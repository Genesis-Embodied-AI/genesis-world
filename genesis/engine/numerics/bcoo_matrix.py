import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field

from .dynamic_exclusive_sum import (
    DynamicExclusiveSumScratch,
    dynamic_exclusive_sum_sizes,
    func_dynamic_exclusive_sum,
    kernel_dynamic_exclusive_sum,
)
from .dynamic_radix_sort import (
    DynamicRadixSortScratch,
    dynamic_radix_sort,
    dynamic_radix_sort_sizes,
    func_dynamic_radix_sort,
)
from .fsr_reduce import func_fsr_reduce

# The triplets a matrix holds at most, padded to a multiple of 64 for the sort and the scan, and the scalars of a block
TRIPLETS = Size("triplets")
PADDED_TRIPLETS = Size("padded_triplets")
BLOCK_SCALARS = Size("block_scalars")


class BCOOMatrixScratch(Data, kind=DataKind.SCRATCH):
    """Block coordinate (BCOO) storage of a sparse matrix and its assembly workspace.

    Writers append (row, column, block) triplets. `func_sort_reduce_bcoo` sorts them by (row, column) and sums the
    duplicates into the reduced BCOO entries. A symmetric matrix stores its upper triangle only, and
    `func_write_triplet` mirrors a lower block there. The capacity in triplets is the shape of the triplet arrays (see
    bcoo_matrix_sizes).
    """

    n_triplets: qd.Tensor = field(int, ndarray=True)
    padded_triplets: qd.Tensor = field(int, ndarray=True)
    has_triplet_overflow: qd.Tensor = field(int, ndarray=True)
    # Uncapped triplet count the writers asked for, which sizes the storage at the next growth
    triplet_demand: qd.Tensor = field(int, ndarray=True)
    triplet_row: qd.Tensor = field(int, shape=(TRIPLETS,), ndarray=True)
    triplet_col: qd.Tensor = field(int, shape=(TRIPLETS,), ndarray=True)
    triplet_val: qd.Tensor = field(float, shape=(TRIPLETS * BLOCK_SCALARS,), ndarray=True)
    triplet_keys: qd.Tensor = field(qd.u64, shape=(PADDED_TRIPLETS,), ndarray=True)
    triplet_perm: qd.Tensor = field(int, shape=(PADDED_TRIPLETS,), ndarray=True)
    sort_keys_out: qd.Tensor = field(qd.u64, shape=(PADDED_TRIPLETS,), ndarray=True)
    sort_perm_out: qd.Tensor = field(int, shape=(PADDED_TRIPLETS,), ndarray=True)
    triplet_sorter: DynamicRadixSortScratch
    seg_flags: qd.Tensor = field(int, shape=(PADDED_TRIPLETS,), ndarray=True)
    seg_ids: qd.Tensor = field(int, shape=(PADDED_TRIPLETS,), ndarray=True)
    segment_scanner: DynamicExclusiveSumScratch
    bcoo_nnz: qd.Tensor = field(int, ndarray=True)
    bcoo_row: qd.Tensor = field(int, shape=(TRIPLETS,), ndarray=True)
    bcoo_col: qd.Tensor = field(int, shape=(TRIPLETS,), ndarray=True)
    bcoo_val: qd.Tensor = field(float, shape=(TRIPLETS * BLOCK_SCALARS,), ndarray=True)


def bcoo_matrix_sizes(capacity: int, block_shape: tuple[int, int]) -> dict:
    """Return the sizes of a matrix holding a bounded number of triplets of blocks of a shape.

    `capacity` is the largest number of triplets and `block_shape` the shape of a block.
    """
    padded_capacity = max(((max(capacity, 1) + 63) // 64) * 64, 64)
    return {
        TRIPLETS: capacity,
        PADDED_TRIPLETS: padded_capacity,
        BLOCK_SCALARS: block_shape[0] * block_shape[1],
        BCOOMatrixScratch.triplet_sorter: dynamic_radix_sort_sizes(key_bits=64, capacity=padded_capacity),
        BCOOMatrixScratch.segment_scanner: dynamic_exclusive_sum_sizes(padded_capacity),
    }


def init_bcoo_matrix(matrix: BCOOMatrixScratch, n_triplets: int) -> None:
    """Set the live count of a matrix just built, with no overflow and no reduced entry.

    `n_triplets` is the number of live triplets.
    """
    matrix.n_triplets.fill(n_triplets)
    matrix.padded_triplets.fill(matrix.triplet_row.shape[0])
    matrix.has_triplet_overflow.fill(0)
    matrix.triplet_demand.fill(0)
    matrix.bcoo_nnz.fill(0)


@qd.func
def func_report_triplet_demand(count: int, matrix: BCOOMatrixScratch):
    """Record the triplet demand, the live triplet count capped at the capacity, and the padded extent of the sort.

    A demand above the capacity raises the overflow flag, and the writers drop the triplets past the capacity (see
    func_write_triplet).
    """
    n_triplets = qd.min(count, matrix.triplet_row.shape[0])
    matrix.triplet_demand[()] = count
    matrix.n_triplets[()] = n_triplets
    matrix.padded_triplets[()] = qd.min(((n_triplets + 63) // 64) * 64, matrix.triplet_keys.shape[0])
    matrix.has_triplet_overflow[()] = gs.qd_int(count > matrix.triplet_row.shape[0])


@qd.func
def func_write_triplet(
    i_t: int, i_r: int, i_c: int, block: qd.template(), matrix: BCOOMatrixScratch, is_symmetric: qd.template()
):
    """Write one block, mirrored into the upper triangle for symmetric storage.

    A triplet past the capacity is dropped: the demand report has raised the overflow flag, and the assembly runs again
    once the storage has grown.
    """
    n_block_rows = qd.static(block.n)
    n_block_cols = qd.static(block.m)
    if i_t < matrix.triplet_row.shape[0]:
        if qd.static(not is_symmetric):
            matrix.triplet_row[i_t] = i_r
            matrix.triplet_col[i_t] = i_c
            for k in qd.static(range(n_block_rows)):
                for j in qd.static(range(n_block_cols)):
                    matrix.triplet_val[i_t * (n_block_rows * n_block_cols) + k * n_block_cols + j] = block[k, j]
        else:
            if i_r <= i_c:
                matrix.triplet_row[i_t] = i_r
                matrix.triplet_col[i_t] = i_c
                for k in qd.static(range(n_block_rows)):
                    for j in qd.static(range(n_block_cols)):
                        matrix.triplet_val[i_t * (n_block_rows * n_block_cols) + k * n_block_cols + j] = block[k, j]
            else:
                matrix.triplet_row[i_t] = i_c
                matrix.triplet_col[i_t] = i_r
                for k in qd.static(range(n_block_rows)):
                    for j in qd.static(range(n_block_cols)):
                        matrix.triplet_val[i_t * (n_block_rows * n_block_cols) + k * n_block_cols + j] = block[j, k]


@qd.func
def func_add_triplet(
    i_t: int, i_r: int, i_c: int, block: qd.template(), matrix: BCOOMatrixScratch, is_symmetric: qd.template()
):
    """Add one block to a triplet that `func_write_triplet` wrote before for the same row and column.

    The block is mirrored as `func_write_triplet` mirrors it, and a triplet past the capacity is dropped.
    """
    n_block_rows = qd.static(block.n)
    n_block_cols = qd.static(block.m)
    if i_t < matrix.triplet_row.shape[0]:
        if qd.static(not is_symmetric):
            for k in qd.static(range(n_block_rows)):
                for j in qd.static(range(n_block_cols)):
                    matrix.triplet_val[i_t * (n_block_rows * n_block_cols) + k * n_block_cols + j] += block[k, j]
        else:
            if i_r <= i_c:
                for k in qd.static(range(n_block_rows)):
                    for j in qd.static(range(n_block_cols)):
                        matrix.triplet_val[i_t * (n_block_rows * n_block_cols) + k * n_block_cols + j] += block[k, j]
            else:
                for k in qd.static(range(n_block_rows)):
                    for j in qd.static(range(n_block_cols)):
                        matrix.triplet_val[i_t * (n_block_rows * n_block_cols) + k * n_block_cols + j] += block[j, k]


@qd.func
def func_read_bcoo(i_nz: int, matrix: BCOOMatrixScratch, block_rows: qd.template(), block_cols: qd.template()):
    """Return one reduced block coordinate (BCOO) entry as (row, column, block)."""
    block = qd.Matrix.zero(gs.qd_float, block_rows, block_cols)
    for k in qd.static(range(block_rows)):
        for j in qd.static(range(block_cols)):
            block[k, j] = matrix.bcoo_val[i_nz * (block_rows * block_cols) + k * block_cols + j]
    return matrix.bcoo_row[i_nz], matrix.bcoo_col[i_nz], block


@qd.func(requires_top_level=True)
def func_compose_bcoo_sort_keys(matrix: BCOOMatrixScratch):
    qd.loop_config(name="bcoo_compose_sort_keys")
    for i_t in range(matrix.padded_triplets[()]):
        if i_t < matrix.n_triplets[()]:
            matrix.triplet_keys[i_t] = (qd.u64(matrix.triplet_row[i_t]) << 32) | qd.u64(matrix.triplet_col[i_t])
        else:
            matrix.triplet_keys[i_t] = qd.u64(0xFFFFFFFFFFFFFFFF)
        matrix.triplet_perm[i_t] = i_t


@qd.func(requires_top_level=True)
def func_mark_bcoo_segment_flags(matrix: BCOOMatrixScratch):
    qd.loop_config(name="bcoo_segment_flags")
    for i_t in range(matrix.padded_triplets[()]):
        flag = gs.qd_int(0)
        if i_t < matrix.n_triplets[()] and (
            i_t == matrix.n_triplets[()] - 1 or matrix.triplet_keys[i_t] != matrix.triplet_keys[i_t + 1]
        ):
            flag = gs.qd_int(1)
        matrix.seg_flags[i_t] = flag


@qd.func(requires_top_level=True)
def func_zero_bcoo_values(matrix: BCOOMatrixScratch, block_scalars: qd.template()):
    for _ in range(1):
        matrix.bcoo_nnz[()] = 0
    qd.loop_config(name="bcoo_zero_values")
    for i_val in range(matrix.n_triplets[()] * block_scalars):
        matrix.bcoo_val[i_val] = 0.0


@qd.func(requires_top_level=True)
def func_extract_unique_bcoo_entries(matrix: BCOOMatrixScratch):
    qd.loop_config(name="bcoo_extract_unique")
    for i_t in range(matrix.n_triplets[()]):
        if matrix.seg_flags[i_t] != 0:
            i_seg = gs.qd_int(matrix.seg_ids[i_t])
            key = matrix.triplet_keys[i_t]
            matrix.bcoo_row[i_seg] = gs.qd_int(key >> 32)
            matrix.bcoo_col[i_seg] = gs.qd_int(key & qd.u64(0xFFFFFFFF))
            if i_t == matrix.n_triplets[()] - 1:
                matrix.bcoo_nnz[()] = i_seg + 1


@qd.func(requires_top_level=True)
def func_sort_reduce_bcoo(matrix: BCOOMatrixScratch, block_scalars: qd.template()):
    """Sort the live triplets by (row, column) and sum the duplicates into the reduced block coordinate (BCOO) entries.

    Called inside a graph kernel (see func_dynamic_radix_sort). `block_scalars` is the number of scalars of a block.
    """
    func_compose_bcoo_sort_keys(matrix)
    func_dynamic_radix_sort(
        matrix.triplet_keys,
        matrix.sort_keys_out,
        matrix.triplet_perm,
        matrix.sort_perm_out,
        matrix.padded_triplets,
        matrix.triplet_sorter,
        key_bits=64,
    )
    func_mark_bcoo_segment_flags(matrix)
    func_dynamic_exclusive_sum(matrix.seg_flags, matrix.seg_ids, matrix.padded_triplets, matrix.segment_scanner)
    func_zero_bcoo_values(matrix, block_scalars)
    func_fsr_reduce(
        matrix.seg_ids,
        matrix.triplet_perm,
        matrix.triplet_keys,
        matrix.triplet_val,
        matrix.bcoo_val,
        matrix.n_triplets,
        matrix.padded_triplets,
        matrix.triplet_keys.shape[0],
        gs.qd_float,
        block_scalars,
    )
    func_extract_unique_bcoo_entries(matrix)


@qd.kernel
def kernel_compose_bcoo_sort_keys(matrix: BCOOMatrixScratch):
    func_compose_bcoo_sort_keys(matrix)


@qd.kernel
def kernel_mark_bcoo_segment_flags(matrix: BCOOMatrixScratch):
    func_mark_bcoo_segment_flags(matrix)


@qd.kernel
def kernel_reduce_bcoo(matrix: BCOOMatrixScratch, block_scalars: qd.template()):
    func_zero_bcoo_values(matrix, block_scalars)
    func_fsr_reduce(
        matrix.seg_ids,
        matrix.triplet_perm,
        matrix.triplet_keys,
        matrix.triplet_val,
        matrix.bcoo_val,
        matrix.n_triplets,
        matrix.padded_triplets,
        matrix.triplet_keys.shape[0],
        gs.qd_float,
        block_scalars,
    )
    func_extract_unique_bcoo_entries(matrix)


def sort_reduce_bcoo(matrix: BCOOMatrixScratch, block_scalars: int) -> None:
    """Run func_sort_reduce_bcoo as separate launches from the host, sharing the compiled sort and scan kernels."""
    kernel_compose_bcoo_sort_keys(matrix)
    dynamic_radix_sort(
        matrix.triplet_sorter,
        matrix.triplet_keys,
        matrix.sort_keys_out,
        matrix.triplet_perm,
        matrix.sort_perm_out,
        matrix.padded_triplets,
        key_bits=64,
    )
    kernel_mark_bcoo_segment_flags(matrix)
    kernel_dynamic_exclusive_sum(matrix.seg_flags, matrix.seg_ids, matrix.padded_triplets, matrix.segment_scanner)
    kernel_reduce_bcoo(matrix, block_scalars)
