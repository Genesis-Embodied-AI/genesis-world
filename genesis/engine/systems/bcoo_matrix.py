from __future__ import annotations

import numpy as np
import quadrants as qd
from quadrants.algorithms import (
    exclusive_scan_add,
    exclusive_scan_scratch_slots,
    sort,
    sort_scratch_slots,
)

from .dynamic_exclusive_sum import DynamicExclusiveSum, dynamic_exclusive_sum
from .dynamic_radix_sort import DynamicRadixSort, dynamic_radix_sort
from .fsr_reduce import fast_segmented_reduce_body as fsr_reduce_body


@qd.data_oriented
class BCOOMatrix:
    """General block-COO storage and its assembly workspace.

    Matrix shape, block shape, scalar type, and symmetric storage are explicit
    construction properties. Sorting and duplicate reduction operate only on
    those properties; numerical operations such as SpMV are external.
    """

    n_block_rows: int
    n_block_cols: int
    block_rows: int
    block_cols: int
    value_type_is_f64: bool
    symmetric: bool
    block_scalar_count: int
    sort_end_bit: int
    sort_log256_max_n: int
    scan_log256_max_n: int
    genesis_legacy_sort_reduce_host: bool
    max_triplets_host: int
    padded_triplets_host: int
    n_triplets: qd.Ndarray
    max_triplets: qd.Ndarray
    padded_triplets: qd.Ndarray
    triplet_overflow: qd.Ndarray
    bcoo_valid: qd.Ndarray
    triplet_row: qd.Ndarray
    triplet_col: qd.Ndarray
    triplet_val: qd.Ndarray
    triplet_keys: qd.Ndarray
    triplet_perm: qd.Ndarray
    sort_keys_out: qd.Ndarray
    sort_perm_out: qd.Ndarray
    triplet_sorter: DynamicRadixSort
    sort_scratch: qd.Ndarray
    sort_size: qd.Ndarray
    seg_flags: qd.Ndarray
    seg_ids: qd.Ndarray
    segment_scanner: DynamicExclusiveSum
    scan_scratch: qd.Ndarray
    bcoo_nnz: qd.Ndarray
    bcoo_row: qd.Ndarray
    bcoo_col: qd.Ndarray
    bcoo_val: qd.Ndarray

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        block_shape: tuple[int, int],
        value_type,
        symmetric: bool,
        initial_triplets: int,
        max_triplets: int,
        genesis_legacy_sort_reduce: bool = False,
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
            raise TypeError("BCOOMatrix currently supports qd.f32 and qd.f64 scalar values")

        block_scalar_count = block_shape[0] * block_shape[1]
        sort_log256_max_n = 4
        scan_log256_max_n = 4
        triplet_capacity = max(max_triplets, 1)
        padded_capacity = max(((triplet_capacity + 63) // 64) * 64, 64)

        def scalar(dtype, value, np_dtype):
            result = qd.ndarray(dtype, shape=())
            result.from_numpy(np.array(value, dtype=np_dtype))
            return result

        self.n_block_rows, self.n_block_cols = shape
        self.block_rows, self.block_cols = block_shape
        self.value_type_is_f64 = value_type == qd.f64
        self.symmetric = bool(symmetric)
        self.block_scalar_count = block_scalar_count
        self.sort_end_bit = 64
        self.sort_log256_max_n = sort_log256_max_n
        self.scan_log256_max_n = scan_log256_max_n
        self.genesis_legacy_sort_reduce_host = bool(genesis_legacy_sort_reduce)
        self.max_triplets_host = triplet_capacity
        self.padded_triplets_host = padded_capacity
        self.n_triplets = scalar(qd.i32, initial_triplets, np.int32)
        self.max_triplets = scalar(qd.i32, triplet_capacity, np.int32)
        self.padded_triplets = scalar(qd.i32, triplet_capacity, np.int32)
        self.triplet_overflow = scalar(qd.i32, 0, np.int32)
        self.bcoo_valid = scalar(qd.i32, 1, np.int32)
        self.triplet_row = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.triplet_col = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.triplet_val = qd.ndarray(value_type, shape=(triplet_capacity * block_scalar_count,))
        self.triplet_keys = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.triplet_perm = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.sort_keys_out = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.sort_perm_out = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.triplet_sorter = DynamicRadixSort(qd.u64, padded_capacity)
        self.sort_scratch = qd.ndarray(
            qd.u32,
            shape=(max(sort_scratch_slots(padded_capacity, sort_log256_max_n), 1),),
        )
        self.sort_size = scalar(qd.i32, initial_triplets, np.int32)
        self.seg_flags = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.seg_ids = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.segment_scanner = DynamicExclusiveSum(padded_capacity)
        self.scan_scratch = qd.ndarray(
            qd.i32,
            shape=(max(exclusive_scan_scratch_slots(padded_capacity, scan_log256_max_n), 1),),
        )
        self.bcoo_nnz = scalar(qd.i32, 0, np.int32)
        self.bcoo_row = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.bcoo_col = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.bcoo_val = qd.ndarray(value_type, shape=(triplet_capacity * block_scalar_count,))

    @property
    def shape(self) -> tuple[int, int]:
        """Host-side matrix shape; device code uses scalar dimension members."""
        return self.n_block_rows, self.n_block_cols

    @property
    def block_shape(self) -> tuple[int, int]:
        """Host-side block shape; device code uses scalar dimension members."""
        return self.block_rows, self.block_cols

    def grow(self, capacity: int, *, live_size: int | None = None) -> None:
        """Grow storage while preserving this data-object identity."""
        if capacity < 0 or (live_size is not None and live_size < 0):
            raise ValueError("BCOOMatrix growth sizes must be non-negative")
        if capacity <= self.triplet_row.shape[0]:
            self.max_triplets.from_numpy(np.array(self.triplet_row.shape[0], dtype=np.int32))
            if live_size is not None:
                self.sort_size.from_numpy(np.array(live_size, dtype=np.int32))
            return

        replacement = BCOOMatrix(
            shape=self.shape,
            block_shape=self.block_shape,
            value_type=qd.f64 if self.value_type_is_f64 else qd.f32,
            symmetric=self.symmetric,
            initial_triplets=0 if live_size is None else live_size,
            max_triplets=capacity,
            genesis_legacy_sort_reduce=self.genesis_legacy_sort_reduce_host,
        )
        self.max_triplets_host = replacement.max_triplets_host
        self.padded_triplets_host = replacement.padded_triplets_host
        self.triplet_row = replacement.triplet_row
        self.triplet_col = replacement.triplet_col
        self.triplet_val = replacement.triplet_val
        self.triplet_keys = replacement.triplet_keys
        self.triplet_perm = replacement.triplet_perm
        self.sort_keys_out = replacement.sort_keys_out
        self.sort_perm_out = replacement.sort_perm_out
        self.triplet_sorter = replacement.triplet_sorter
        self.sort_scratch = replacement.sort_scratch
        self.seg_flags = replacement.seg_flags
        self.seg_ids = replacement.seg_ids
        self.segment_scanner = replacement.segment_scanner
        self.scan_scratch = replacement.scan_scratch
        self.bcoo_row = replacement.bcoo_row
        self.bcoo_col = replacement.bcoo_col
        self.bcoo_val = replacement.bcoo_val
        self.max_triplets.from_numpy(np.array(capacity, dtype=np.int32))
        self.padded_triplets.from_numpy(np.array(capacity, dtype=np.int32))
        self.sort_size.from_numpy(np.array(0 if live_size is None else live_size, dtype=np.int32))

    @qd.func
    def set_n_triplets(self, count):
        assert count >= 0 and count <= self.max_triplets[()], (
            f"BCOOMatrix set_n_triplets count={count}, capacity={self.max_triplets[()]}"
        )
        self.n_triplets[()] = count
        self.triplet_overflow[()] = 0
        self.sort_size[()] = count

    @qd.func
    def report_triplet_demand(self, count):
        """Publish required capacity without permitting an out-of-bounds write."""
        assert count >= 0, f"BCOOMatrix report_triplet_demand count={count}"
        self.n_triplets[()] = count
        overflow = count > self.max_triplets[()]
        self.triplet_overflow[()] = qd.i32(overflow)
        if overflow:
            self.sort_size[()] = 0
        else:
            self.sort_size[()] = count

    @qd.func
    def write_triplet(self, slot, row, col, block: qd.template()):
        """Write one block according to this matrix's storage contract."""
        assert slot >= 0 and slot < self.n_triplets[()] and self.n_triplets[()] <= self.max_triplets[()], (
            f"BCOOMatrix write_triplet slot={slot}, row={row}, col={col}, "
            f"live={self.n_triplets[()]}, capacity={self.max_triplets[()]}"
        )
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
        """Return one stored input triplet as ``(row, column, block)``."""
        assert slot >= 0 and slot < self.n_triplets[()], (
            f"BCOOMatrix read_triplet slot={slot}, live={self.n_triplets[()]}, capacity={self.max_triplets[()]}"
        )
        if qd.static(self.value_type_is_f64):
            block = qd.Matrix.zero(qd.f64, self.block_rows, self.block_cols)
            for i in qd.static(range(self.block_rows)):
                for j in qd.static(range(self.block_cols)):
                    block[i, j] = self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j]
            return self.triplet_row[slot], self.triplet_col[slot], block
        else:
            block = qd.Matrix.zero(qd.f32, self.block_rows, self.block_cols)
            for i in qd.static(range(self.block_rows)):
                for j in qd.static(range(self.block_cols)):
                    block[i, j] = self.triplet_val[slot * self.block_scalar_count + i * self.block_cols + j]
            return self.triplet_row[slot], self.triplet_col[slot], block

    @qd.func
    def read_bcoo(self, entry):
        """Return one reduced BCOO entry as ``(row, column, block)``."""
        assert entry >= 0 and entry < self.bcoo_nnz[()], (
            f"BCOOMatrix read_bcoo entry={entry}, nnz={self.bcoo_nnz[()]}, capacity={self.max_triplets[()]}"
        )
        if qd.static(self.value_type_is_f64):
            block = qd.Matrix.zero(qd.f64, self.block_rows, self.block_cols)
            for i in qd.static(range(self.block_rows)):
                for j in qd.static(range(self.block_cols)):
                    block[i, j] = self.bcoo_val[entry * self.block_scalar_count + i * self.block_cols + j]
            return self.bcoo_row[entry], self.bcoo_col[entry], block
        else:
            block = qd.Matrix.zero(qd.f32, self.block_rows, self.block_cols)
            for i in qd.static(range(self.block_rows)):
                for j in qd.static(range(self.block_cols)):
                    block[i, j] = self.bcoo_val[entry * self.block_scalar_count + i * self.block_cols + j]
            return self.bcoo_row[entry], self.bcoo_col[entry], block


@qd.func(requires_top_level=True)
def zero_bcoo_triplets(matrix: qd.template()):
    for i in range(matrix.n_triplets[()] * matrix.block_scalar_count):
        matrix.triplet_val[i] = 0.0


@qd.func(requires_top_level=True)
def compose_bcoo_sort_keys_padded(matrix: qd.template()):
    qd.loop_config(name="body_compose_sort_keys")
    for i in range(matrix.triplet_keys.shape[0]):
        if i < matrix.padded_triplets[()]:
            if i < matrix.n_triplets[()]:
                matrix.triplet_keys[i] = (qd.u64(matrix.triplet_row[i]) << 32) | qd.u64(matrix.triplet_col[i])
                matrix.triplet_perm[i] = i
            else:
                matrix.triplet_keys[i] = qd.u64(0xFFFFFFFFFFFFFFFF)
                matrix.triplet_perm[i] = i


@qd.func(requires_top_level=True)
def sort_bcoo_triplets(matrix: qd.template()):
    if qd.static(matrix.genesis_legacy_sort_reduce_host):
        sort(
            matrix.triplet_keys,
            matrix.sort_keys_out,
            matrix.triplet_perm,
            matrix.sort_perm_out,
            matrix.sort_scratch,
            matrix.sort_size,
            qd.u64,
            True,
            matrix.sort_end_bit,
            matrix.sort_log256_max_n,
        )
    else:
        dynamic_radix_sort(
            matrix.triplet_sorter,
            matrix.triplet_keys,
            matrix.sort_keys_out,
            matrix.triplet_perm,
            matrix.sort_perm_out,
            matrix.padded_triplets[()],
        )


@qd.func(requires_top_level=True)
def mark_bcoo_segment_flags(matrix: qd.template()):
    qd.loop_config(name="body_segment_flags")
    for i in range(matrix.triplet_keys.shape[0]):
        if i < matrix.padded_triplets[()]:
            flag = qd.i32(0)
            if i < matrix.n_triplets[()] and (
                i == matrix.n_triplets[()] - 1 or matrix.triplet_keys[i] != matrix.triplet_keys[i + 1]
            ):
                flag = qd.i32(1)
            matrix.seg_flags[i] = flag


@qd.func(requires_top_level=True)
def scan_bcoo_segments(matrix: qd.template()):
    if qd.static(matrix.genesis_legacy_sort_reduce_host):
        exclusive_scan_add(
            matrix.seg_flags,
            matrix.seg_ids,
            matrix.scan_scratch,
            matrix.n_triplets[()],
            qd.i32,
            matrix.scan_log256_max_n,
        )
    else:
        dynamic_exclusive_sum(
            matrix.segment_scanner,
            matrix.seg_flags,
            matrix.seg_ids,
            matrix.padded_triplets[()],
        )


@qd.func(requires_top_level=True)
def zero_bcoo_values(matrix: qd.template()):
    for _ in range(1):
        matrix.bcoo_nnz[()] = 0
    qd.loop_config(name="body_zero_bcoo")
    for i in range(matrix.bcoo_val.shape[0]):
        if i < matrix.padded_triplets[()] * matrix.block_scalar_count:
            matrix.bcoo_val[i] = 0.0


@qd.func(requires_top_level=True)
def reduce_bcoo_segments(matrix: qd.template()):
    if qd.static(matrix.genesis_legacy_sort_reduce_host):
        qd.loop_config(name="body_fsr_merge_legacy")
        for i in range(matrix.n_triplets[()]):
            source = qd.i32(matrix.triplet_perm[i])
            segment = matrix.seg_ids[i]
            for component in qd.static(range(matrix.block_scalar_count)):
                qd.atomic_add(
                    matrix.bcoo_val[segment * matrix.block_scalar_count + component],
                    matrix.triplet_val[source * matrix.block_scalar_count + component],
                )
    else:
        if qd.static(matrix.value_type_is_f64):
            fsr_reduce_body(
                matrix.seg_ids,
                matrix.triplet_perm,
                matrix.triplet_keys,
                matrix.triplet_val,
                matrix.bcoo_val,
                matrix.n_triplets,
                matrix.padded_triplets,
                matrix.triplet_keys.shape[0],
                qd.f64,
                matrix.block_scalar_count,
            )
        else:
            fsr_reduce_body(
                matrix.seg_ids,
                matrix.triplet_perm,
                matrix.triplet_keys,
                matrix.triplet_val,
                matrix.bcoo_val,
                matrix.n_triplets,
                matrix.padded_triplets,
                matrix.triplet_keys.shape[0],
                qd.f32,
                matrix.block_scalar_count,
            )


@qd.func(requires_top_level=True)
def extract_unique_bcoo_entries(matrix: qd.template()):
    qd.loop_config(name="body_extract_unique")
    for i in range(matrix.triplet_keys.shape[0]):
        if i < matrix.padded_triplets[()] and i < matrix.n_triplets[()] and matrix.seg_flags[i] != 0:
            segment = qd.i32(matrix.seg_ids[i])
            key = matrix.triplet_keys[i]
            matrix.bcoo_row[segment] = qd.i32(key >> 32)
            matrix.bcoo_col[segment] = qd.i32(key & qd.u64(0xFFFFFFFF))
            if i == matrix.n_triplets[()] - 1:
                matrix.bcoo_nnz[()] = segment + 1


@qd.func(requires_top_level=True)
def validate_bcoo(matrix: qd.template()):
    for _ in range(1):
        matrix.bcoo_valid[()] = qd.i32(matrix.triplet_overflow[()] == 0)
    qd.loop_config(name="body_validate_bcoo")
    for i in range(matrix.bcoo_nnz[()]):
        row = matrix.bcoo_row[i]
        col = matrix.bcoo_col[i]
        if row < 0 or row >= matrix.n_block_rows or col < 0 or col >= matrix.n_block_cols:
            matrix.bcoo_valid[()] = 0
        if qd.static(matrix.symmetric):
            if row > col:
                matrix.bcoo_valid[()] = 0
        if i > 0:
            previous_row = matrix.bcoo_row[i - 1]
            previous_col = matrix.bcoo_col[i - 1]
            if previous_row > row or (previous_row == row and previous_col >= col):
                matrix.bcoo_valid[()] = 0


@qd.func(requires_top_level=True)
def sort_reduce_bcoo(matrix: qd.template()):
    compose_bcoo_sort_keys_padded(matrix)
    sort_bcoo_triplets(matrix)
    mark_bcoo_segment_flags(matrix)
    scan_bcoo_segments(matrix)
    zero_bcoo_values(matrix)
    reduce_bcoo_segments(matrix)
    extract_unique_bcoo_entries(matrix)
    validate_bcoo(matrix)
