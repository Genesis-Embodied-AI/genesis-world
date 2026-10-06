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
from .sim_system import SimData

QDDataType = type(qd.f64)


@qd.data_oriented
class BCOOMatrix(SimData):
    """General block-COO storage and its assembly workspace.

    Matrix shape, block shape, scalar type, and symmetric storage are explicit
    construction properties. Sorting and duplicate reduction operate only on
    those properties; numerical operations such as SpMV are external.
    """

    shape: tuple[int, int]
    block_shape: tuple[int, int]
    value_type: QDDataType
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


def get_bcoo_matrix(
    *,
    shape: tuple[int, int],
    block_shape: tuple[int, int],
    value_type,
    symmetric: bool,
    initial_triplets: int,
    max_triplets: int,
    genesis_legacy_sort_reduce: bool = False,
) -> BCOOMatrix:
    if min(*shape, *block_shape, initial_triplets, max_triplets) < 0:
        raise ValueError("BCOOMatrix dimensions and capacities must be non-negative")
    if min(*block_shape) == 0:
        raise ValueError("BCOOMatrix block dimensions must be positive")
    if initial_triplets > max_triplets:
        raise ValueError("BCOOMatrix initial triplet count exceeds capacity")
    if symmetric and (shape[0] != shape[1] or block_shape[0] != block_shape[1]):
        raise ValueError("Symmetric BCOOMatrix storage requires square matrix and block shapes")

    block_scalar_count = block_shape[0] * block_shape[1]
    sort_log256_max_n = 4
    scan_log256_max_n = 4
    triplet_capacity = max(max_triplets, 1)
    padded_capacity = max(((triplet_capacity + 63) // 64) * 64, 64)

    def scalar(dtype, value, np_dtype):
        result = qd.ndarray(dtype, shape=())
        result.from_numpy(np.array(value, dtype=np_dtype))
        return result

    data = BCOOMatrix()
    data.shape = shape
    data.block_shape = block_shape
    data.value_type = value_type
    data.symmetric = bool(symmetric)
    data.block_scalar_count = block_scalar_count
    data.sort_end_bit = 64
    data.sort_log256_max_n = sort_log256_max_n
    data.scan_log256_max_n = scan_log256_max_n
    data.genesis_legacy_sort_reduce_host = bool(genesis_legacy_sort_reduce)
    data.max_triplets_host = triplet_capacity
    data.padded_triplets_host = padded_capacity
    data.n_triplets = scalar(qd.i32, initial_triplets, np.int32)
    data.max_triplets = scalar(qd.i32, triplet_capacity, np.int32)
    data.padded_triplets = scalar(qd.i32, triplet_capacity, np.int32)
    data.triplet_overflow = scalar(qd.i32, 0, np.int32)
    data.bcoo_valid = scalar(qd.i32, 1, np.int32)
    data.triplet_row = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.triplet_col = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.triplet_val = qd.ndarray(value_type, shape=(triplet_capacity * block_scalar_count,))
    data.triplet_keys = qd.ndarray(qd.u64, shape=(padded_capacity,))
    data.triplet_perm = qd.ndarray(qd.i32, shape=(padded_capacity,))
    data.sort_keys_out = qd.ndarray(qd.u64, shape=(padded_capacity,))
    data.sort_perm_out = qd.ndarray(qd.i32, shape=(padded_capacity,))
    data.triplet_sorter = DynamicRadixSort(qd.u64, padded_capacity)
    data.sort_scratch = qd.ndarray(
        qd.u32,
        shape=(max(sort_scratch_slots(padded_capacity, sort_log256_max_n), 1),),
    )
    data.sort_size = scalar(qd.i32, initial_triplets, np.int32)
    data.seg_flags = qd.ndarray(qd.i32, shape=(padded_capacity,))
    data.seg_ids = qd.ndarray(qd.i32, shape=(padded_capacity,))
    data.segment_scanner = DynamicExclusiveSum(padded_capacity)
    data.scan_scratch = qd.ndarray(
        qd.i32,
        shape=(max(exclusive_scan_scratch_slots(padded_capacity, scan_log256_max_n), 1),),
    )
    data.bcoo_nnz = scalar(qd.i32, 0, np.int32)
    data.bcoo_row = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.bcoo_col = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.bcoo_val = qd.ndarray(value_type, shape=(triplet_capacity * block_scalar_count,))
    return data


def grow_bcoo_matrix(
    matrix: BCOOMatrix,
    capacity: int,
    *,
    live_size: int | None = None,
) -> None:
    """Grow matrix buffers in place while preserving the data-object identity."""
    if capacity < 0 or (live_size is not None and live_size < 0):
        raise ValueError("BCOOMatrix growth sizes must be non-negative")
    if capacity <= matrix.triplet_row.shape[0]:
        matrix.max_triplets.from_numpy(np.array(matrix.triplet_row.shape[0], dtype=np.int32))
        if live_size is not None:
            matrix.sort_size.from_numpy(np.array(live_size, dtype=np.int32))
        return
    replacement = get_bcoo_matrix(
        shape=matrix.shape,
        block_shape=matrix.block_shape,
        value_type=matrix.value_type,
        symmetric=matrix.symmetric,
        initial_triplets=0 if live_size is None else live_size,
        max_triplets=capacity,
        genesis_legacy_sort_reduce=matrix.genesis_legacy_sort_reduce_host,
    )
    matrix.max_triplets_host = replacement.max_triplets_host
    matrix.padded_triplets_host = replacement.padded_triplets_host
    matrix.triplet_row = replacement.triplet_row
    matrix.triplet_col = replacement.triplet_col
    matrix.triplet_val = replacement.triplet_val
    matrix.triplet_keys = replacement.triplet_keys
    matrix.triplet_perm = replacement.triplet_perm
    matrix.sort_keys_out = replacement.sort_keys_out
    matrix.sort_perm_out = replacement.sort_perm_out
    matrix.triplet_sorter = replacement.triplet_sorter
    matrix.sort_scratch = replacement.sort_scratch
    matrix.seg_flags = replacement.seg_flags
    matrix.seg_ids = replacement.seg_ids
    matrix.segment_scanner = replacement.segment_scanner
    matrix.scan_scratch = replacement.scan_scratch
    matrix.bcoo_row = replacement.bcoo_row
    matrix.bcoo_col = replacement.bcoo_col
    matrix.bcoo_val = replacement.bcoo_val
    matrix.max_triplets.from_numpy(np.array(capacity, dtype=np.int32))
    matrix.padded_triplets.from_numpy(np.array(capacity, dtype=np.int32))
    matrix.sort_size.from_numpy(np.array(0 if live_size is None else live_size, dtype=np.int32))


@qd.func
def set_bcoo_n_triplets(matrix: qd.template(), count):
    matrix.n_triplets[()] = count
    overflow = count > matrix.max_triplets[()]
    matrix.triplet_overflow[()] = qd.i32(overflow)
    if overflow:
        matrix.sort_size[()] = 0
    else:
        matrix.sort_size[()] = count


@qd.func(requires_top_level=True)
def zero_bcoo_triplets(matrix: qd.template()):
    for i in range(matrix.n_triplets[()] * matrix.block_scalar_count):
        matrix.triplet_val[i] = matrix.value_type(0.0)


@qd.func
def write_bcoo_block(matrix: qd.template(), slot, row, col, block: qd.template()):
    """Write one block according to the matrix storage contract."""
    if qd.static(not matrix.symmetric) or row <= col:
        matrix.triplet_row[slot] = row
        matrix.triplet_col[slot] = col
        for i in qd.static(range(matrix.block_shape[0])):
            for j in qd.static(range(matrix.block_shape[1])):
                matrix.triplet_val[slot * matrix.block_scalar_count + i * matrix.block_shape[1] + j] = block[i, j]
    else:
        matrix.triplet_row[slot] = col
        matrix.triplet_col[slot] = row
        for i in qd.static(range(matrix.block_shape[0])):
            for j in qd.static(range(matrix.block_shape[1])):
                matrix.triplet_val[slot * matrix.block_scalar_count + i * matrix.block_shape[1] + j] = block[j, i]


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
            matrix.bcoo_val[i] = matrix.value_type(0.0)


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
        fsr_reduce_body(
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
        if row < 0 or row >= matrix.shape[0] or col < 0 or col >= matrix.shape[1]:
            matrix.bcoo_valid[()] = 0
        if qd.static(matrix.symmetric) and row > col:
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
