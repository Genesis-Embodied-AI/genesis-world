import numpy as np
import quadrants as qd

from .dynamic_exclusive_sum import DynamicExclusiveSum, func_dynamic_exclusive_sum, kernel_dynamic_exclusive_sum
from .dynamic_radix_sort import (
    DynamicRadixSort,
    dynamic_radix_sort,
    func_dynamic_radix_sort,
    func_dynamic_radix_sort_graph,
)
from .fsr_reduce import func_fsr_reduce, head_segmented_reduce_add


@qd.data_oriented
class BCOOMatrix:
    """Block-COO storage of a matrix and its assembly workspace.

    Writers append (row, column, block) triplets; `func_sort_reduce_bcoo` sorts them by (row, column) and sums the
    duplicates into the reduced BCOO entries. Matrix shape, block shape, scalar type and symmetric storage are fixed at
    construction. A symmetric matrix stores its upper triangle only, and `write_triplet` mirrors a lower block there.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        block_shape: tuple[int, int],
        value_type,
        symmetric: bool,
        initial_triplets: int,
        max_triplets: int,
    ) -> None:
        if min(*shape, *block_shape, initial_triplets, max_triplets) < 0:
            raise ValueError("BCOOMatrix dimensions and capacities must be non-negative")
        if min(*block_shape) == 0:
            raise ValueError("BCOOMatrix block dimensions must be positive")
        if initial_triplets > max_triplets:
            raise ValueError("BCOOMatrix initial triplet count exceeds capacity")
        if symmetric and (shape[0] != shape[1] or block_shape[0] != block_shape[1]):
            raise ValueError("Symmetric BCOOMatrix storage requires square matrix and block shapes")
        if value_type not in (qd.f32, qd.f64):
            raise TypeError("BCOOMatrix supports qd.f32 and qd.f64 scalar values")

        block_scalar_count = block_shape[0] * block_shape[1]
        triplet_capacity = max(max_triplets, 1)
        padded_capacity = max(((triplet_capacity + 63) // 64) * 64, 64)

        def scalar(dtype, value, np_dtype):
            result = qd.ndarray(dtype, shape=())
            result.from_numpy(np.array(value, dtype=np_dtype))
            return result

        self.n_block_rows, self.n_block_cols = shape
        self.block_rows, self.block_cols = block_shape
        self.value_type = value_type
        self.symmetric = bool(symmetric)
        self.block_scalar_count = block_scalar_count
        self.n_triplets = scalar(qd.i32, initial_triplets, np.int32)
        self.max_triplets = scalar(qd.i32, triplet_capacity, np.int32)
        self.padded_triplets = scalar(qd.i32, triplet_capacity, np.int32)
        self.triplet_overflow = scalar(qd.i32, 0, np.int32)
        # The triplet count the writers asked for, uncapped: what a growth must make room for
        self.triplet_demand = scalar(qd.i32, 0, np.int32)
        self.bcoo_valid = scalar(qd.i32, 1, np.int32)
        self.triplet_row = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.triplet_col = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.triplet_val = qd.ndarray(value_type, shape=(triplet_capacity * block_scalar_count,))
        self.triplet_keys = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.triplet_perm = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.sort_keys_out = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.sort_perm_out = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.triplet_sorter = DynamicRadixSort(qd.u64, padded_capacity)
        self.seg_flags = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.seg_ids = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.segment_scanner = DynamicExclusiveSum(padded_capacity)
        self.bcoo_nnz = scalar(qd.i32, 0, np.int32)
        self.bcoo_row = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.bcoo_col = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.bcoo_val = qd.ndarray(value_type, shape=(triplet_capacity * block_scalar_count,))

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_block_rows, self.n_block_cols

    @property
    def block_shape(self) -> tuple[int, int]:
        return self.block_rows, self.block_cols

    def grow(self, capacity: int) -> None:
        """Replace every capacity-sized buffer with a larger one, keeping this object, then reset the live count.

        A capacity the buffers already hold only raises the usable capacity.
        """
        if capacity <= self.triplet_row.shape[0]:
            current = int(self.max_triplets.to_numpy())
            self.max_triplets.from_numpy(np.array(max(current, capacity), dtype=np.int32))
            return
        replacement = BCOOMatrix(
            shape=self.shape,
            block_shape=self.block_shape,
            value_type=self.value_type,
            symmetric=self.symmetric,
            initial_triplets=0,
            max_triplets=capacity,
        )
        for name in (
            "triplet_row",
            "triplet_col",
            "triplet_val",
            "triplet_keys",
            "triplet_perm",
            "sort_keys_out",
            "sort_perm_out",
            "triplet_sorter",
            "seg_flags",
            "seg_ids",
            "segment_scanner",
            "bcoo_row",
            "bcoo_col",
            "bcoo_val",
        ):
            setattr(self, name, getattr(replacement, name))
        self.n_triplets.from_numpy(np.array(0, dtype=np.int32))
        self.max_triplets.from_numpy(np.array(capacity, dtype=np.int32))
        self.padded_triplets.from_numpy(np.array(capacity, dtype=np.int32))

    @qd.func
    def report_triplet_demand(self, count):
        """Publish the live triplet count and the padded extent the sort/reduce works on, raising the overflow flag
        instead of letting a writer go out of bounds."""
        n_triplets = qd.min(count, self.max_triplets[()])
        self.triplet_demand[()] = count
        self.n_triplets[()] = n_triplets
        self.padded_triplets[()] = qd.min(((n_triplets + 63) // 64) * 64, self.triplet_keys.shape[0])
        self.triplet_overflow[()] = qd.i32(count > self.max_triplets[()])

    @qd.func
    def write_triplet(
        self,
        slot,
        row,
        col,
        block: qd.template(),  # qd.Matrix
    ):
        """Write one block, mirrored into the upper triangle for symmetric storage.

        A slot past the capacity is dropped: the demand report has raised the overflow flag, and the assembly runs again
        once the storage has grown.
        """
        if slot < self.max_triplets[()]:
            if qd.static(not self.symmetric):
                self.triplet_row[slot] = row
                self.triplet_col[slot] = col
                for i in qd.static(range(self.block_rows)):
                    for j in qd.static(range(self.block_cols)):
                        self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j] = block[i, j]
            else:
                if row <= col:
                    self.triplet_row[slot] = row
                    self.triplet_col[slot] = col
                    for i in qd.static(range(self.block_rows)):
                        for j in qd.static(range(self.block_cols)):
                            self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j] = block[i, j]
                else:
                    self.triplet_row[slot] = col
                    self.triplet_col[slot] = row
                    for i in qd.static(range(self.block_rows)):
                        for j in qd.static(range(self.block_cols)):
                            self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j] = block[j, i]

    @qd.func
    def read_triplet(self, slot):
        """Return one stored input triplet as (row, column, block)."""
        block = qd.Matrix.zero(self.value_type, self.block_rows, self.block_cols)
        for i in qd.static(range(self.block_rows)):
            for j in qd.static(range(self.block_cols)):
                block[i, j] = self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j]
        return self.triplet_row[slot], self.triplet_col[slot], block

    @qd.func
    def read_bcoo(self, entry):
        """Return one reduced BCOO entry as (row, column, block)."""
        block = qd.Matrix.zero(self.value_type, self.block_rows, self.block_cols)
        for i in qd.static(range(self.block_rows)):
            for j in qd.static(range(self.block_cols)):
                block[i, j] = self.bcoo_val[entry * self.block_scalar_count + i * self.block_cols + j]
        return self.bcoo_row[entry], self.bcoo_col[entry], block


@qd.func(requires_top_level=True)
def func_compose_bcoo_sort_keys(
    matrix: qd.template(),  # BCOOMatrix
):
    qd.loop_config(name="bcoo_compose_sort_keys")
    for i in range(matrix.padded_triplets[()]):
        if i < matrix.n_triplets[()]:
            matrix.triplet_keys[i] = (qd.u64(matrix.triplet_row[i]) << 32) | qd.u64(matrix.triplet_col[i])
        else:
            matrix.triplet_keys[i] = qd.u64(0xFFFFFFFFFFFFFFFF)
        matrix.triplet_perm[i] = i


@qd.func(requires_top_level=True)
def func_mark_bcoo_segment_flags(
    matrix: qd.template(),  # BCOOMatrix
):
    qd.loop_config(name="bcoo_segment_flags")
    for i in range(matrix.padded_triplets[()]):
        flag = qd.i32(0)
        if i < matrix.n_triplets[()] and (
            i == matrix.n_triplets[()] - 1 or matrix.triplet_keys[i] != matrix.triplet_keys[i + 1]
        ):
            flag = qd.i32(1)
        matrix.seg_flags[i] = flag


@qd.func(requires_top_level=True)
def func_zero_bcoo_values(
    matrix: qd.template(),  # BCOOMatrix
):
    for _ in range(1):
        matrix.bcoo_nnz[()] = 0
    qd.loop_config(name="bcoo_zero_values")
    for i in range(matrix.n_triplets[()] * matrix.block_scalar_count):
        matrix.bcoo_val[i] = 0.0


@qd.func(requires_top_level=True)
def func_extract_unique_bcoo_entries(
    matrix: qd.template(),  # BCOOMatrix
):
    qd.loop_config(name="bcoo_extract_unique")
    for i in range(matrix.n_triplets[()]):
        if matrix.seg_flags[i] != 0:
            segment = qd.i32(matrix.seg_ids[i])
            key = matrix.triplet_keys[i]
            matrix.bcoo_row[segment] = qd.i32(key >> 32)
            matrix.bcoo_col[segment] = qd.i32(key & qd.u64(0xFFFFFFFF))
            if i == matrix.n_triplets[()] - 1:
                matrix.bcoo_nnz[()] = segment + 1


@qd.func(requires_top_level=True)
def func_sort_reduce_bcoo(
    matrix: qd.template(),  # BCOOMatrix
):
    """Sort the live triplets by (row, column) and sum the duplicates into the reduced BCOO entries."""
    func_compose_bcoo_sort_keys(matrix)
    func_dynamic_radix_sort(
        matrix.triplet_sorter,
        matrix.triplet_keys,
        matrix.sort_keys_out,
        matrix.triplet_perm,
        matrix.sort_perm_out,
        matrix.padded_triplets[()],
    )
    func_mark_bcoo_segment_flags(matrix)
    func_dynamic_exclusive_sum(matrix.segment_scanner, matrix.seg_flags, matrix.seg_ids, matrix.padded_triplets[()])
    func_zero_bcoo_values(matrix)
    func_fsr_reduce(
        matrix.seg_ids,
        matrix.triplet_perm,
        matrix.triplet_keys,
        matrix.triplet_val,
        matrix.bcoo_val,
        matrix.n_triplets,
        matrix.padded_triplets,
        matrix.triplet_keys.shape[0],
        matrix.value_type,
        matrix.block_scalar_count,
    )
    func_extract_unique_bcoo_entries(matrix)


@qd.func(requires_top_level=True)
def func_sort_reduce_bcoo_graph(
    matrix: qd.template(),  # BCOOMatrix
):
    """func_sort_reduce_bcoo for a graph kernel, with the radix passes in a device loop."""
    func_compose_bcoo_sort_keys(matrix)
    func_dynamic_radix_sort_graph(
        matrix.triplet_sorter,
        matrix.triplet_keys,
        matrix.sort_keys_out,
        matrix.triplet_perm,
        matrix.sort_perm_out,
        matrix.padded_triplets[()],
    )
    func_mark_bcoo_segment_flags(matrix)
    func_dynamic_exclusive_sum(matrix.segment_scanner, matrix.seg_flags, matrix.seg_ids, matrix.padded_triplets[()])
    func_zero_bcoo_values(matrix)
    func_fsr_reduce(
        matrix.seg_ids,
        matrix.triplet_perm,
        matrix.triplet_keys,
        matrix.triplet_val,
        matrix.bcoo_val,
        matrix.n_triplets,
        matrix.padded_triplets,
        matrix.triplet_keys.shape[0],
        matrix.value_type,
        matrix.block_scalar_count,
    )
    func_extract_unique_bcoo_entries(matrix)


@qd.kernel
def kernel_compose_bcoo_sort_keys(matrix: qd.template()):
    func_compose_bcoo_sort_keys(matrix)


@qd.kernel
def kernel_mark_bcoo_segment_flags(matrix: qd.template()):
    func_mark_bcoo_segment_flags(matrix)


@qd.kernel
def kernel_reduce_bcoo(matrix: qd.template()):
    func_zero_bcoo_values(matrix)
    func_fsr_reduce(
        matrix.seg_ids,
        matrix.triplet_perm,
        matrix.triplet_keys,
        matrix.triplet_val,
        matrix.bcoo_val,
        matrix.n_triplets,
        matrix.padded_triplets,
        matrix.triplet_keys.shape[0],
        matrix.value_type,
        matrix.block_scalar_count,
    )
    func_extract_unique_bcoo_entries(matrix)


def sort_reduce_bcoo(matrix: BCOOMatrix) -> None:
    """Host-driven func_sort_reduce_bcoo, sharing the compiled sort and scan launches with every other user."""
    kernel_compose_bcoo_sort_keys(matrix)
    dynamic_radix_sort(
        matrix.triplet_sorter,
        matrix.triplet_keys,
        matrix.sort_keys_out,
        matrix.triplet_perm,
        matrix.sort_perm_out,
        matrix.padded_triplets,
    )
    kernel_mark_bcoo_segment_flags(matrix)
    kernel_dynamic_exclusive_sum(matrix.segment_scanner, matrix.seg_flags, matrix.seg_ids, matrix.padded_triplets)
    kernel_reduce_bcoo(matrix)


@qd.func(requires_top_level=True)
def func_sym_bcoo_spmv(
    matrix: qd.template(),  # BCOOMatrix
    x: qd.template(),  # qd.Ndarray
    y: qd.template(),  # qd.Ndarray
):
    """Accumulate y += A x for a symmetric BCOO matrix, mirroring its stored upper triangle.

    One thread per reduced entry. The entries are row-sorted, so consecutive lanes of a warp holding the same row form
    a segment: a head-segmented warp reduction sums their row products and the segment head adds the sum with one
    atomic. The mirrored product of an off-diagonal entry is added atomically per entry. The caller clears y, so that
    several operators can accumulate into it.
    """
    qd.loop_config(name="bcoo_sym_spmv")
    for t in range(((matrix.bcoo_nnz[()] + 31) // 32) * 32):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
        nnz = matrix.bcoo_nnz[()]
        is_valid = t < nnz
        row = qd.i32(-1)
        product = qd.Vector.zero(matrix.value_type, matrix.block_rows)
        if is_valid:
            row_, col, block = matrix.read_bcoo(t)
            row = row_
            for block_row in qd.static(range(matrix.block_rows)):
                for block_col in qd.static(range(matrix.block_cols)):
                    product[block_row] += block[block_row, block_col] * x[col * matrix.block_cols + block_col]
            if row != col:
                for block_col in qd.static(range(matrix.block_cols)):
                    mirrored = matrix.value_type(0.0)
                    for block_row in qd.static(range(matrix.block_rows)):
                        mirrored += block[block_row, block_col] * x[row * matrix.block_rows + block_row]
                    qd.atomic_add(y[col * matrix.block_cols + block_col], mirrored)
        previous_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.max(lane - 1, 0)))
        next_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.min(lane + 1, 31)))
        is_head = lane == 0 or row != previous_row
        is_tail = lane == 31 or row != next_row
        reduced = qd.Vector.zero(matrix.value_type, matrix.block_rows)
        for block_row in qd.static(range(matrix.block_rows)):
            reduced[block_row] = head_segmented_reduce_add(product[block_row], is_tail)
        if is_valid and is_head:
            for block_row in qd.static(range(matrix.block_rows)):
                qd.atomic_add(y[row * matrix.block_rows + block_row], reduced[block_row])
