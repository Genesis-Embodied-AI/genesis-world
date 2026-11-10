"""Block coordinate (BCOO) sparse matrix, assembled from blocks written in any order by sorting them by row and column
and summing those sharing a position."""

import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field
from genesis.utils.simt import qd_segmented_sum, qd_slot_neighbors

from .exclusive_sum import ExclusiveSumScratch, exclusive_sum_sizes, func_exclusive_sum, kernel_exclusive_sum
from .radix_sort import RadixSortScratch, func_radix_sort, kernel_radix_sort, radix_sort_sizes

# The blocks a matrix holds at most, the same count padded to whole tiles of 64 for the sort and the scan, and the
# scalars of a block
TRIPLETS = Size("triplets")
PADDED_TRIPLETS = Size("padded_triplets")
BLOCK_SCALARS = Size("block_scalars")


class BCOOMatrixScratch(Data, kind=DataKind.SCRATCH):
    """
    A sparse block matrix as the blocks its writers give it (the triplets), and as their sums by position (the entries).

    A writer owns the slots of its triplets and writes each one whole. `n_triplets_demanded` counts the triplets the
    writers ask for, of which the first `n_triplets` fit, `has_triplet_overflow` telling the matrix to grow. A symmetric
    matrix keeps the blocks of its upper triangle alone. The entries are sorted by row then column.
    """

    n_triplets: qd.Tensor = field(int, shape=(), ndarray=True)
    n_padded_triplets: qd.Tensor = field(int, shape=(), ndarray=True)
    n_triplets_demanded: qd.Tensor = field(int, shape=(), ndarray=True)
    has_triplet_overflow: qd.Tensor = field(int, shape=(), ndarray=True)
    triplets_row: qd.Tensor = field(int, shape=(TRIPLETS,), ndarray=True)
    triplets_col: qd.Tensor = field(int, shape=(TRIPLETS,), ndarray=True)
    triplets_value: qd.Tensor = field(float, shape=(TRIPLETS * BLOCK_SCALARS,), ndarray=True)
    triplets_key: qd.Tensor = field(qd.u64, shape=(PADDED_TRIPLETS,), ndarray=True)
    triplets_source_idx: qd.Tensor = field(int, shape=(PADDED_TRIPLETS,), ndarray=True)
    keys_buffer: qd.Tensor = field(qd.u64, shape=(PADDED_TRIPLETS,), ndarray=True)
    sources_idx_buffer: qd.Tensor = field(int, shape=(PADDED_TRIPLETS,), ndarray=True)
    sorter: RadixSortScratch
    triplets_is_entry_end: qd.Tensor = field(int, shape=(PADDED_TRIPLETS,), ndarray=True)
    triplets_entry_idx: qd.Tensor = field(int, shape=(PADDED_TRIPLETS,), ndarray=True)
    scanner: ExclusiveSumScratch
    n_entries: qd.Tensor = field(int, shape=(), ndarray=True)
    entries_row: qd.Tensor = field(int, shape=(TRIPLETS,), ndarray=True)
    entries_col: qd.Tensor = field(int, shape=(TRIPLETS,), ndarray=True)
    entries_value: qd.Tensor = field(float, shape=(TRIPLETS * BLOCK_SCALARS,), ndarray=True)


def bcoo_matrix_sizes(capacity: int, block_shape: tuple[int, int]) -> dict:
    """Return the sizes of a matrix of at most `capacity` blocks of shape `block_shape`."""
    n_padded_triplets = max(((max(capacity, 1) + 63) // 64) * 64, 64)
    return {
        TRIPLETS: capacity,
        PADDED_TRIPLETS: n_padded_triplets,
        BLOCK_SCALARS: block_shape[0] * block_shape[1],
        BCOOMatrixScratch.sorter: radix_sort_sizes(key_bits=64, capacity=n_padded_triplets),
        BCOOMatrixScratch.scanner: exclusive_sum_sizes(n_padded_triplets),
    }


def build_bcoo_matrix(capacity: int, block_shape: tuple[int, int]) -> BCOOMatrixScratch:
    """Build a matrix of at most `capacity` blocks of shape `block_shape`, holding none."""
    matrix = BCOOMatrixScratch(sizes=bcoo_matrix_sizes(capacity, block_shape))
    for counter in (matrix.n_triplets, matrix.n_triplets_demanded, matrix.has_triplet_overflow, matrix.n_entries):
        counter.fill(0)
    matrix.n_padded_triplets.fill(64)
    return matrix


@qd.func
def func_add_by_key(i_key: int, value, out: qd.template(), n_components: qd.template() = 0):
    """
    Add a value into the entry of its key, -1 adding nothing.

    On GPU, the threads of a warp holding the same key sum their values first, so that the runs of equal keys of
    consecutive threads cost one atomic addition each. The 32 threads of every warp take part. A vector value of
    `n_components` components is summed component by component.
    """
    if qd.static(gs.backend == gs.cpu):
        if i_key >= 0:
            qd.atomic_add(out[i_key], value)
    else:
        tid = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
        i_key_prev, i_key_next = qd_slot_neighbors(tid, i_key)
        total = value
        is_tail = False
        if qd.static(n_components > 0):
            for k in qd.static(range(n_components)):
                total_k, is_tail = qd_segmented_sum(tid, i_key, i_key_prev, i_key_next, value[k])
                total[k] = total_k
        else:
            total, is_tail = qd_segmented_sum(tid, i_key, i_key_prev, i_key_next, value)
        if is_tail:
            qd.atomic_add(out[i_key], total)


@qd.func
def func_report_triplet_demand(n_triplets_demanded: int, matrix: BCOOMatrixScratch):
    """Set the triplets the writers ask for, those that fit and the padded extent of their sort."""
    n_triplets = qd.min(n_triplets_demanded, matrix.triplets_row.shape[0])
    matrix.n_triplets_demanded[()] = n_triplets_demanded
    matrix.n_triplets[()] = n_triplets
    matrix.n_padded_triplets[()] = qd.min((n_triplets + 63) // 64 * 64, matrix.triplets_key.shape[0])
    matrix.has_triplet_overflow[()] = gs.qd_int(n_triplets_demanded > matrix.triplets_row.shape[0])


@qd.func
def func_write_triplet(i_t: int, i_r: int, i_c: int, block, matrix: BCOOMatrixScratch, is_symmetric: qd.template()):
    """
    Write a block in a triplet slot, the slots past the capacity dropping it.

    A symmetric matrix stores the block of the lower triangle as the transposed block of the upper one.
    """
    if i_t < matrix.triplets_row.shape[0]:
        i_r_stored = i_r
        i_c_stored = i_c
        block_stored = block
        if qd.static(is_symmetric):
            if i_r > i_c:
                i_r_stored = i_c
                i_c_stored = i_r
                block_stored = block.transpose()
        matrix.triplets_row[i_t] = i_r_stored
        matrix.triplets_col[i_t] = i_c_stored
        for k, j in qd.static(qd.ndrange(block.n, block.m)):
            matrix.triplets_value[i_t * (block.n * block.m) + k * block.m + j] = block_stored[k, j]


@qd.func
def func_discard_triplet(i_t: int, matrix: BCOOMatrixScratch, block_rows: qd.template(), block_cols: qd.template()):
    """Write a zero block at the first row and column in a triplet slot, which adds nothing to the entry it joins."""
    if i_t < matrix.triplets_row.shape[0]:
        matrix.triplets_row[i_t] = 0
        matrix.triplets_col[i_t] = 0
        for k in qd.static(range(block_rows * block_cols)):
            matrix.triplets_value[i_t * (block_rows * block_cols) + k] = 0.0


@qd.func
def func_read_entry(i_e: int, matrix: BCOOMatrixScratch, block_rows: qd.template(), block_cols: qd.template()):
    """Return the row, the column and the block of an entry."""
    block = qd.Matrix.zero(gs.qd_float, block_rows, block_cols)
    for k, j in qd.static(qd.ndrange(block_rows, block_cols)):
        block[k, j] = matrix.entries_value[i_e * (block_rows * block_cols) + k * block_cols + j]
    return matrix.entries_row[i_e], matrix.entries_col[i_e], block


@qd.func(requires_top_level=True)
def func_compose_triplets_key(matrix: BCOOMatrixScratch):
    """Key every triplet by its row then its column, the padding taking the largest key, which sorts last."""
    qd.loop_config(name="bcoo_compose_keys")
    for i_t in range(matrix.n_padded_triplets[()]):
        key = qd.u64(0xFFFFFFFFFFFFFFFF)
        if i_t < matrix.n_triplets[()]:
            key = (qd.u64(matrix.triplets_row[i_t]) << qd.u64(32)) | qd.u64(matrix.triplets_col[i_t])
        matrix.triplets_key[i_t] = key
        matrix.triplets_source_idx[i_t] = i_t


@qd.func(requires_top_level=True)
def func_mark_entry_ends(matrix: BCOOMatrixScratch):
    """Flag the last sorted triplet of every position, whose exclusive sum numbers the entries."""
    qd.loop_config(name="bcoo_mark_entry_ends")
    for i_t in range(matrix.n_padded_triplets[()]):
        is_entry_end = 0
        if i_t < matrix.n_triplets[()]:
            if i_t == matrix.n_triplets[()] - 1:
                is_entry_end = 1
            elif matrix.triplets_key[i_t] != matrix.triplets_key[i_t + 1]:
                is_entry_end = 1
        matrix.triplets_is_entry_end[i_t] = is_entry_end


@qd.func(requires_top_level=True)
def func_sort_triplets(matrix: BCOOMatrixScratch):
    """Sort the triplets by row then column and number the entries their positions make up."""
    func_compose_triplets_key(matrix)
    func_radix_sort(
        matrix.triplets_key,
        matrix.keys_buffer,
        matrix.triplets_source_idx,
        matrix.sources_idx_buffer,
        matrix.n_padded_triplets,
        matrix.sorter,
        key_bits=64,
    )
    func_mark_entry_ends(matrix)
    func_exclusive_sum(
        matrix.triplets_is_entry_end, matrix.triplets_entry_idx, matrix.n_padded_triplets, matrix.scanner
    )


@qd.func(requires_top_level=True)
def func_reduce_triplets(matrix: BCOOMatrixScratch, block_scalars: qd.template()):
    """Sum the blocks of the sorted triplets into the entry of their position (see 'func_sort_triplets')."""
    qd.loop_config(name="bcoo_clear_entries")
    for i_value in range(matrix.n_triplets[()] * block_scalars):
        matrix.entries_value[i_value] = 0.0
    # The loop covers whole warps, which take part in the sums by key together (see 'func_add_by_key')
    qd.loop_config(name="bcoo_sum_triplets", block_dim=256)
    for i_t in range(qd.min((matrix.n_triplets[()] + 31) // 32 * 32, matrix.triplets_key.shape[0])):
        i_e = gs.qd_int(-1)
        i_source = gs.qd_int(0)
        if i_t < matrix.n_triplets[()]:
            i_e = matrix.triplets_entry_idx[i_t]
            i_source = matrix.triplets_source_idx[i_t]
        for k in qd.static(range(block_scalars)):
            value = gs.qd_float(0.0)
            i_value = gs.qd_int(-1)
            if i_e >= 0:
                value = matrix.triplets_value[i_source * block_scalars + k]
                i_value = i_e * block_scalars + k
            func_add_by_key(i_value, value, matrix.entries_value)
    qd.loop_config(name="bcoo_publish_entries")
    for i_t in range(matrix.n_triplets[()]):
        if matrix.triplets_is_entry_end[i_t] != 0:
            i_e = matrix.triplets_entry_idx[i_t]
            key = matrix.triplets_key[i_t]
            matrix.entries_row[i_e] = gs.qd_int(key >> qd.u64(32))
            matrix.entries_col[i_e] = gs.qd_int(key & qd.u64(0xFFFFFFFF))
            if i_t == matrix.n_triplets[()] - 1:
                matrix.n_entries[()] = i_e + 1


@qd.kernel
def kernel_compose_triplets_key(matrix: BCOOMatrixScratch):
    func_compose_triplets_key(matrix)


@qd.kernel
def kernel_mark_entry_ends(matrix: BCOOMatrixScratch):
    func_mark_entry_ends(matrix)


@qd.kernel
def kernel_reduce_triplets(matrix: BCOOMatrixScratch, block_scalars: qd.template()):
    func_reduce_triplets(matrix, block_scalars)


def sort_triplets(matrix: BCOOMatrixScratch) -> None:
    """Sort the triplets of a matrix by row then column (see 'func_sort_triplets'), launching the sort and the scan
    shared by every matrix on their own."""
    kernel_compose_triplets_key(matrix)
    kernel_radix_sort(
        matrix.triplets_key,
        matrix.keys_buffer,
        matrix.triplets_source_idx,
        matrix.sources_idx_buffer,
        matrix.n_padded_triplets,
        matrix.sorter,
        key_bits=64,
    )
    kernel_mark_entry_ends(matrix)
    kernel_exclusive_sum(
        matrix.triplets_is_entry_end, matrix.triplets_entry_idx, matrix.n_padded_triplets, matrix.scanner
    )


def assemble_bcoo_matrix(matrix: BCOOMatrixScratch, block_scalars: int) -> None:
    """Sort the triplets of a matrix and sum them into its entries."""
    sort_triplets(matrix)
    kernel_reduce_triplets(matrix, block_scalars)
