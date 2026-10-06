from __future__ import annotations

import numpy as np
import quadrants as qd
from quadrants.algorithms import (
    exclusive_scan_add,
    exclusive_scan_scratch_slots,
    sort,
    sort_scratch_slots,
)

from .dynamic_exclusive_sum import (
    DynamicExclusiveSum,
    dynamic_exclusive_sum,
)
from .dynamic_radix_sort import (
    DynamicRadixSort,
    dynamic_radix_sort,
)
from .fsr_reduce import fast_segmented_reduce_body as fsr_reduce_body
from .sim_system import SimSystem


@qd.data_oriented
class GlobalLinearSystem(SimSystem):
    """CGQ-compatible global 3x3-block BCOO linear system."""

    def __init__(
        self,
        *,
        genesis_legacy_sort_reduce: bool = False,
    ) -> None:
        super().__init__()
        self.sort_end_bit = 64
        self.sort_log256_max_n = 4
        self.scan_log256_max_n = 4
        self.extent_slot_count_host = 0
        self.is_initialized_host = False
        self.genesis_legacy_sort_reduce_host = bool(genesis_legacy_sort_reduce)

    def do_build(self) -> None:
        pass

    def register_extent_slot(self) -> int:
        if self.is_initialized_host:
            raise RuntimeError("Extent slots must be registered before GlobalLinearSystem.init()")
        slot = self.extent_slot_count_host
        self.extent_slot_count_host += 1
        return slot

    def init(
        self,
        n_block_rows: int,
        n_elastic_triplets: int,
        max_contact_body_triplets: int,
        dof_block_base: int,
        pcg_tol_rate: float,
    ) -> None:
        if self.is_initialized_host:
            raise RuntimeError("GlobalLinearSystem is already initialized")
        if min(n_block_rows, n_elastic_triplets, max_contact_body_triplets, dof_block_base) < 0:
            raise ValueError("GlobalLinearSystem sizes must be non-negative")

        self.n_block_rows_host = n_block_rows
        self.total_dof_host = n_block_rows * 3
        self.max_triplets_host = max(n_elastic_triplets + max_contact_body_triplets, 1)
        self.padded_triplets_host = max(((self.max_triplets_host + 63) // 64) * 64, 64)
        self.pcg_tol_rate = pcg_tol_rate

        self.n_block_rows = qd.ndarray(qd.i32, shape=())
        self.total_dof = qd.ndarray(qd.i32, shape=())
        self.n_triplets = qd.ndarray(qd.i32, shape=())
        self.max_triplets = qd.ndarray(qd.i32, shape=())
        self.padded_triplets = qd.ndarray(qd.i32, shape=())
        self.n_elastic_triplets = qd.ndarray(qd.i32, shape=())
        self.max_contact_body_triplets = qd.ndarray(qd.i32, shape=())
        self.dof_block_base = qd.ndarray(qd.i32, shape=())
        self.n_extent_slots = qd.ndarray(qd.i32, shape=())
        self.n_elastic = qd.ndarray(qd.i32, shape=())
        self.required_block_rows = qd.ndarray(qd.i32, shape=())
        self.triplet_overflow = qd.ndarray(qd.i32, shape=())
        self.bcoo_valid = qd.ndarray(qd.i32, shape=())

        self.n_block_rows.from_numpy(np.array(n_block_rows, dtype=np.int32))
        self.total_dof.from_numpy(np.array(self.total_dof_host, dtype=np.int32))
        self.n_triplets.from_numpy(np.array(n_elastic_triplets, dtype=np.int32))
        self.max_triplets.from_numpy(np.array(self.max_triplets_host, dtype=np.int32))
        self.padded_triplets.from_numpy(np.array(self.max_triplets_host, dtype=np.int32))
        self.n_elastic_triplets.from_numpy(np.array(n_elastic_triplets, dtype=np.int32))
        self.max_contact_body_triplets.from_numpy(np.array(max_contact_body_triplets, dtype=np.int32))
        self.dof_block_base.from_numpy(np.array(dof_block_base, dtype=np.int32))
        self.n_extent_slots.from_numpy(np.array(self.extent_slot_count_host, dtype=np.int32))
        self.n_elastic.from_numpy(np.array(n_elastic_triplets, dtype=np.int32))
        self.required_block_rows.from_numpy(np.array(n_block_rows, dtype=np.int32))
        self.triplet_overflow.from_numpy(np.array(0, dtype=np.int32))
        self.bcoo_valid.from_numpy(np.array(1, dtype=np.int32))

        self.extent_slots = qd.ndarray(qd.i32, shape=(max(self.extent_slot_count_host, 1),))
        self.extent_offsets = qd.ndarray(qd.i32, shape=(max(self.extent_slot_count_host, 1),))
        self.extent_slots.from_numpy(np.zeros(max(self.extent_slot_count_host, 1), dtype=np.int32))
        self.extent_offsets.from_numpy(np.full(max(self.extent_slot_count_host, 1), n_elastic_triplets, dtype=np.int32))

        self.triplet_row = qd.ndarray(qd.i32, shape=(self.max_triplets_host,))
        self.triplet_col = qd.ndarray(qd.i32, shape=(self.max_triplets_host,))
        self.triplet_val = qd.ndarray(qd.f64, shape=(self.max_triplets_host * 9,))
        self.triplet_keys = qd.ndarray(qd.u64, shape=(self.padded_triplets_host,))
        self.triplet_perm = qd.ndarray(qd.i32, shape=(self.padded_triplets_host,))

        self.sort_keys_out = qd.ndarray(qd.u64, shape=(self.padded_triplets_host,))
        self.sort_perm_out = qd.ndarray(qd.i32, shape=(self.padded_triplets_host,))
        self.triplet_sorter = DynamicRadixSort(
            qd.u64,
            self.padded_triplets_host,
        )
        sort_scratch_size = max(
            sort_scratch_slots(self.padded_triplets_host, self.sort_log256_max_n),
            1,
        )
        self.sort_scratch = qd.ndarray(qd.u32, shape=(sort_scratch_size,))
        self.sort_size = qd.ndarray(qd.i32, shape=())
        self.sort_size.from_numpy(np.array(n_elastic_triplets, dtype=np.int32))

        self.seg_flags = qd.ndarray(qd.i32, shape=(self.padded_triplets_host,))
        self.seg_ids = qd.ndarray(qd.i32, shape=(self.padded_triplets_host,))
        self.segment_scanner = DynamicExclusiveSum(
            self.padded_triplets_host,
        )
        scan_scratch_size = max(
            exclusive_scan_scratch_slots(self.padded_triplets_host, self.scan_log256_max_n),
            1,
        )
        self.scan_scratch = qd.ndarray(qd.i32, shape=(scan_scratch_size,))

        self.bcoo_nnz = qd.ndarray(qd.i32, shape=())
        self.bcoo_row = qd.ndarray(qd.i32, shape=(self.max_triplets_host,))
        self.bcoo_col = qd.ndarray(qd.i32, shape=(self.max_triplets_host,))
        self.bcoo_val = qd.ndarray(qd.f64, shape=(self.max_triplets_host * 9,))
        self.bcoo_nnz.from_numpy(np.array(0, dtype=np.int32))

        self.x_sol = qd.ndarray(qd.f64, shape=(max(self.total_dof_host, 1),))
        self.b_rhs = qd.ndarray(qd.f64, shape=(max(self.total_dof_host, 1),))
        self.is_initialized_host = True

    def realloc_triplet_buffers(self, capacity: int, live_size: int | None = None) -> None:
        if capacity <= self.triplet_row.shape[0]:
            self.max_triplets.from_numpy(np.array(self.triplet_row.shape[0], dtype=np.int32))
            if live_size is not None:
                self.sort_size.from_numpy(np.array(live_size, dtype=np.int32))
            return
        padded_capacity = max(((capacity + 63) // 64) * 64, 64)
        self.triplet_row = qd.ndarray(qd.i32, shape=(capacity,))
        self.triplet_col = qd.ndarray(qd.i32, shape=(capacity,))
        self.triplet_val = qd.ndarray(qd.f64, shape=(capacity * 9,))
        self.triplet_keys = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.triplet_perm = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.sort_keys_out = qd.ndarray(qd.u64, shape=(padded_capacity,))
        self.sort_perm_out = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.triplet_sorter = DynamicRadixSort(
            qd.u64,
            padded_capacity,
        )
        self.sort_scratch = qd.ndarray(
            qd.u32,
            shape=(max(sort_scratch_slots(padded_capacity, self.sort_log256_max_n), 1),),
        )
        self.seg_flags = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.seg_ids = qd.ndarray(qd.i32, shape=(padded_capacity,))
        self.segment_scanner = DynamicExclusiveSum(
            padded_capacity,
        )
        self.scan_scratch = qd.ndarray(
            qd.i32,
            shape=(max(exclusive_scan_scratch_slots(padded_capacity, self.scan_log256_max_n), 1),),
        )
        self.bcoo_row = qd.ndarray(qd.i32, shape=(capacity,))
        self.bcoo_col = qd.ndarray(qd.i32, shape=(capacity,))
        self.bcoo_val = qd.ndarray(qd.f64, shape=(capacity * 9,))
        self.max_triplets.from_numpy(np.array(capacity, dtype=np.int32))
        self.padded_triplets.from_numpy(np.array(capacity, dtype=np.int32))
        self.max_contact_body_triplets.from_numpy(np.array(capacity, dtype=np.int32))
        if live_size is not None:
            self.sort_size.from_numpy(np.array(live_size, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def derive_extents(self):
        for _ in range(1):
            total = qd.i32(0)
            for slot in range(self.n_extent_slots[()]):
                self.extent_offsets[slot] = total
                total = total + self.extent_slots[slot]
            self.n_elastic[()] = total
            self.n_triplets[()] = total
            overflow = total > self.max_triplets[()]
            self.triplet_overflow[()] = qd.i32(overflow)
            if overflow:
                self.sort_size[()] = 0
            else:
                self.sort_size[()] = total

    @qd.func(requires_top_level=True)
    def compute_n_triplets(self, contact: qd.template()):
        for _ in range(1):
            total = self.n_elastic[()] + contact.n_unique_triplets[()]
            self.n_triplets[()] = total
            overflow = total > self.max_triplets[()]
            self.triplet_overflow[()] = qd.i32(overflow)
            if overflow:
                self.sort_size[()] = 0
            else:
                self.sort_size[()] = total

    @qd.func(requires_top_level=True)
    def zero_rhs(self):
        for i in range(self.total_dof[()]):
            self.b_rhs[i] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def zero_triplet(self):
        for i in range(self.n_triplets[()] * 9):
            self.triplet_val[i] = qd.f64(0.0)

    @qd.func
    def set_sym(self, slot, row, col, block: qd.template()):
        if row <= col:
            self.triplet_row[slot] = row
            self.triplet_col[slot] = col
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    self.triplet_val[slot * 9 + i * 3 + j] = block[i, j]
        else:
            self.triplet_row[slot] = col
            self.triplet_col[slot] = row
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    self.triplet_val[slot * 9 + i * 3 + j] = block[j, i]

    @qd.func(requires_top_level=True)
    def compose_sort_keys_padded(self):
        qd.loop_config(name="body_compose_sort_keys")
        for i in range(self.triplet_keys.shape[0]):
            if i < self.padded_triplets[()]:
                if i < self.n_triplets[()]:
                    self.triplet_keys[i] = (qd.u64(self.triplet_row[i]) << 32) | qd.u64(self.triplet_col[i])
                    self.triplet_perm[i] = i
                else:
                    self.triplet_keys[i] = qd.u64(0xFFFFFFFFFFFFFFFF)
                    self.triplet_perm[i] = i

    @qd.func(requires_top_level=True)
    def sort_triplets(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            sort(
                self.triplet_keys,
                self.sort_keys_out,
                self.triplet_perm,
                self.sort_perm_out,
                self.sort_scratch,
                self.sort_size,
                qd.u64,
                True,
                self.sort_end_bit,
                self.sort_log256_max_n,
            )
        else:
            dynamic_radix_sort(
                self.triplet_sorter,
                self.triplet_keys,
                self.sort_keys_out,
                self.triplet_perm,
                self.sort_perm_out,
                self.padded_triplets[()],
            )

    @qd.func(requires_top_level=True)
    def segment_flags_body(self):
        qd.loop_config(name="body_segment_flags")
        for i in range(self.triplet_keys.shape[0]):
            if i < self.padded_triplets[()]:
                flag = qd.i32(0)
                if i < self.n_triplets[()] and (
                    i == self.n_triplets[()] - 1 or self.triplet_keys[i] != self.triplet_keys[i + 1]
                ):
                    flag = qd.i32(1)
                self.seg_flags[i] = flag

    @qd.func(requires_top_level=True)
    def scan_body(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            exclusive_scan_add(
                self.seg_flags,
                self.seg_ids,
                self.scan_scratch,
                self.n_triplets[()],
                qd.i32,
                self.scan_log256_max_n,
            )
        else:
            dynamic_exclusive_sum(
                self.segment_scanner,
                self.seg_flags,
                self.seg_ids,
                self.padded_triplets[()],
            )

    @qd.func(requires_top_level=True)
    def zero_bcoo(self):
        for _ in range(1):
            self.bcoo_nnz[()] = 0
        qd.loop_config(name="body_zero_bcoo")
        for i in range(self.bcoo_val.shape[0]):
            if i < self.padded_triplets[()] * 9:
                self.bcoo_val[i] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def fast_segmented_reduce_body(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            qd.loop_config(name="body_fsr_merge_legacy")
            for i in range(self.n_triplets[()]):
                source = qd.i32(self.triplet_perm[i])
                segment = self.seg_ids[i]
                for component in qd.static(range(9)):
                    qd.atomic_add(
                        self.bcoo_val[segment * 9 + component],
                        self.triplet_val[source * 9 + component],
                    )
        else:
            fsr_reduce_body(
                self.seg_ids,
                self.triplet_perm,
                self.triplet_keys,
                self.triplet_val,
                self.bcoo_val,
                self.n_triplets,
                self.padded_triplets,
                self.triplet_keys.shape[0],
            )

    @qd.func(requires_top_level=True)
    def extract_unique_body(self):
        qd.loop_config(name="body_extract_unique")
        for i in range(self.triplet_keys.shape[0]):
            if i < self.padded_triplets[()] and i < self.n_triplets[()] and self.seg_flags[i] != 0:
                segment = qd.i32(self.seg_ids[i])
                key = self.triplet_keys[i]
                self.bcoo_row[segment] = qd.i32(key >> 32)
                self.bcoo_col[segment] = qd.i32(key & qd.u64(0xFFFFFFFF))
                if i == self.n_triplets[()] - 1:
                    self.bcoo_nnz[()] = segment + 1

    @qd.func(requires_top_level=True)
    def validate_bcoo(self):
        for _ in range(1):
            self.bcoo_valid[()] = qd.i32(self.triplet_overflow[()] == 0)
        qd.loop_config(name="body_validate_bcoo")
        for i in range(self.bcoo_nnz[()]):
            row = self.bcoo_row[i]
            col = self.bcoo_col[i]
            if row > col:
                self.bcoo_valid[()] = 0
            if i > 0:
                previous_row = self.bcoo_row[i - 1]
                previous_col = self.bcoo_col[i - 1]
                if previous_row > row or (previous_row == row and previous_col >= col):
                    self.bcoo_valid[()] = 0

    @qd.func(requires_top_level=True)
    def body_sort_reduce(self):
        self.compose_sort_keys_padded()
        self.sort_triplets()
        self.segment_flags_body()
        self.scan_body()
        self.zero_bcoo()
        self.fast_segmented_reduce_body()
        self.extract_unique_body()
        self.validate_bcoo()

    @qd.func(requires_top_level=True)
    def traverse(self, fem_preconditioner: qd.template(), has_fem: qd.template()):
        if qd.static(has_fem):
            fem_preconditioner.memset()
            fem_preconditioner.gather()
            fem_preconditioner.invert()

    @qd.func(requires_top_level=True)
    def spmv(self, x: qd.template(), y: qd.template()):
        for i in range(self.bcoo_nnz[()]):
            row = self.bcoo_row[i]
            col = self.bcoo_col[i]
            block = qd.Matrix.zero(qd.f64, 3, 3)
            for r in qd.static(range(3)):
                for c in qd.static(range(3)):
                    block[r, c] = self.bcoo_val[i * 9 + r * 3 + c]

            x_col = qd.Vector([x[col * 3], x[col * 3 + 1], x[col * 3 + 2]])
            y_row = block @ x_col
            for r in qd.static(range(3)):
                qd.atomic_add(y[row * 3 + r], y_row[r])

            if row != col:
                x_row = qd.Vector([x[row * 3], x[row * 3 + 1], x[row * 3 + 2]])
                y_col = block.transpose() @ x_row
                for r in qd.static(range(3)):
                    qd.atomic_add(y[col * 3 + r], y_col[r])
