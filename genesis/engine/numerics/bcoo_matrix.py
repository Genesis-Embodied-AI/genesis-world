import quadrants as qd

from genesis.utils.misc import qd_to_numpy

from .dynamic_exclusive_sum import DynamicExclusiveSum, func_dynamic_exclusive_sum, kernel_dynamic_exclusive_sum
from .dynamic_radix_sort import DynamicRadixSort, dynamic_radix_sort, func_dynamic_radix_sort
from .fsr_reduce import func_fsr_reduce, head_segmented_reduce_add


@qd.data_oriented
class BCOOMatrix:
    """Block coordinate (BCOO) storage of a sparse matrix and its assembly workspace.

    Writers append (row, column, block) triplets. `func_sort_reduce_bcoo` sorts them by (row, column) and sums the
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
        assert min(*shape, *block_shape, initial_triplets, max_triplets) >= 0
        assert min(*block_shape) > 0
        assert initial_triplets <= max_triplets
        assert not symmetric or (shape[0] == shape[1] and block_shape[0] == block_shape[1])
        assert value_type in (qd.f32, qd.f64)

        block_scalar_count = block_shape[0] * block_shape[1]
        triplet_capacity = max(max_triplets, 1)
        padded_capacity = max(((triplet_capacity + 63) // 64) * 64, 64)

        self.n_block_rows, self.n_block_cols = shape
        self.block_rows, self.block_cols = block_shape
        self.value_type = value_type
        self.symmetric = bool(symmetric)
        self.block_scalar_count = block_scalar_count
        self.n_triplets = qd.ndarray(qd.i32, shape=())
        self.n_triplets.fill(initial_triplets)
        self.max_triplets = qd.ndarray(qd.i32, shape=())
        self.max_triplets.fill(triplet_capacity)
        self.padded_triplets = qd.ndarray(qd.i32, shape=())
        self.padded_triplets.fill(triplet_capacity)
        self.triplet_overflow = qd.ndarray(qd.i32, shape=())
        self.triplet_overflow.fill(0)
        # Uncapped triplet count the writers asked for, which sizes the storage at the next growth
        self.triplet_demand = qd.ndarray(qd.i32, shape=())
        self.triplet_demand.fill(0)
        self.bcoo_valid = qd.ndarray(qd.i32, shape=())
        self.bcoo_valid.fill(1)
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
        self.bcoo_nnz = qd.ndarray(qd.i32, shape=())
        self.bcoo_nnz.fill(0)
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

        A capacity within the current buffers raises the usable capacity alone.
        """
        if capacity <= self.triplet_row.shape[0]:
            self.max_triplets.fill(max(int(qd_to_numpy(self.max_triplets)), capacity))
            return
        replacement = BCOOMatrix(
            shape=self.shape,
            block_shape=self.block_shape,
            value_type=self.value_type,
            symmetric=self.symmetric,
            initial_triplets=0,
            max_triplets=capacity,
        )
        self.triplet_row = replacement.triplet_row
        self.triplet_col = replacement.triplet_col
        self.triplet_val = replacement.triplet_val
        self.triplet_keys = replacement.triplet_keys
        self.triplet_perm = replacement.triplet_perm
        self.sort_keys_out = replacement.sort_keys_out
        self.sort_perm_out = replacement.sort_perm_out
        self.triplet_sorter = replacement.triplet_sorter
        self.seg_flags = replacement.seg_flags
        self.seg_ids = replacement.seg_ids
        self.segment_scanner = replacement.segment_scanner
        self.bcoo_row = replacement.bcoo_row
        self.bcoo_col = replacement.bcoo_col
        self.bcoo_val = replacement.bcoo_val
        self.n_triplets.fill(0)
        self.max_triplets.fill(capacity)
        self.padded_triplets.fill(capacity)

    @qd.func
    def report_triplet_demand(self, count):
        """Record the triplet demand, the live triplet count capped at the capacity, and the padded extent of the sort.

        A demand above the capacity raises the overflow flag, and the writers drop the triplets past the capacity (see
        write_triplet).
        """
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
    def add_triplet(
        self,
        slot,
        row,
        col,
        block: qd.template(),  # qd.Matrix
    ):
        """Add one block to a triplet that `write_triplet` wrote before for the same row and column.

        The block is mirrored as `write_triplet` mirrors it, and a slot past the capacity is dropped.
        """
        if slot < self.max_triplets[()]:
            if qd.static(not self.symmetric):
                for i in qd.static(range(self.block_rows)):
                    for j in qd.static(range(self.block_cols)):
                        self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j] += block[i, j]
            else:
                if row <= col:
                    for i in qd.static(range(self.block_rows)):
                        for j in qd.static(range(self.block_cols)):
                            self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j] += block[i, j]
                else:
                    for i in qd.static(range(self.block_rows)):
                        for j in qd.static(range(self.block_cols)):
                            self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j] += block[j, i]

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
        """Return one reduced block coordinate (BCOO) entry as (row, column, block)."""
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
    """Sort the live triplets by (row, column) and sum the duplicates into the reduced block coordinate (BCOO) entries.

    Called inside a graph kernel (see func_dynamic_radix_sort).
    """
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
    """Run func_sort_reduce_bcoo as separate launches from the host, sharing the compiled sort and scan kernels."""
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
    """Add the product of a symmetric block coordinate (BCOO) matrix and a vector to an output vector.

    The kernel computes y += A x, with A the matrix whose upper triangle is stored, mirrored on the fly. Each thread
    handles one reduced entry. The entries are row-sorted, so consecutive lanes of a warp holding the same row form a
    segment: a head-segmented warp reduction sums their row products and the segment head adds the sum with one atomic.
    The mirrored product of an off-diagonal entry is added atomically per entry. The caller clears y, so that several
    operators can accumulate into it.
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
                    product_mirrored = matrix.value_type(0.0)
                    for block_row in qd.static(range(matrix.block_rows)):
                        product_mirrored += block[block_row, block_col] * x[row * matrix.block_rows + block_row]
                    qd.atomic_add(y[col * matrix.block_cols + block_col], product_mirrored)
        previous_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.max(lane - 1, 0)))
        next_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.min(lane + 1, 31)))
        is_head = lane == 0 or row != previous_row
        is_tail = lane == 31 or row != next_row
        row_sum = qd.Vector.zero(matrix.value_type, matrix.block_rows)
        for block_row in qd.static(range(matrix.block_rows)):
            row_sum[block_row] = head_segmented_reduce_add(product[block_row], is_tail)
        if is_valid and is_head:
            for block_row in qd.static(range(matrix.block_rows)):
                qd.atomic_add(y[row * matrix.block_rows + block_row], row_sum[block_row])
