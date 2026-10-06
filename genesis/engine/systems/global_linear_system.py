from __future__ import annotations

import numpy as np
import quadrants as qd

from .bcoo_matrix import BCOOMatrix, get_bcoo_matrix, set_bcoo_n_triplets
from .bcoo_operations import sym_bcoo_spmv_naive
from .sim_system import SimData, SimSystem


class GlobalLinearSystem(SimSystem):
    """Global linear problem layout, vectors, and solver entry point."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible global matrix, vector, and extent state."""

        n_block_rows_host: int
        total_dof_host: int
        extent_capacity: int
        n_block_rows: qd.Ndarray
        total_dof: qd.Ndarray
        dof_block_base: qd.Ndarray
        n_extent_slots: qd.Ndarray
        n_elastic: qd.Ndarray
        required_block_rows: qd.Ndarray
        extent_slots: qd.Ndarray
        extent_offsets: qd.Ndarray
        matrix: BCOOMatrix
        x_sol: qd.Ndarray
        b_rhs: qd.Ndarray

    def __init__(
        self,
        *,
        data: Data,
    ) -> None:
        super().__init__()
        self.data = data
        self.extent_slot_count_host = 0

    def build(self) -> None:
        from .pcg_solver import PCGSolver

        self.pcg_solver_system = self.require(PCGSolver)
        self.pcg_operator_action = self.create_action(pcg_apply_operator, self.data)
        self.pcg_solver_system.on_primary_operator(self.pcg_operator_action)

    def register_extent_slot(self) -> int:
        if self.extent_slot_count_host >= self.data.extent_capacity:
            raise RuntimeError("GlobalLinearSystem extent capacity is too small for the registered assembly systems")
        slot = self.extent_slot_count_host
        self.extent_slot_count_host += 1
        self.data.n_extent_slots.from_numpy(np.array(self.extent_slot_count_host, dtype=np.int32))
        return slot


def get_global_linear_system_data(
    *,
    n_block_rows: int,
    n_elastic_triplets: int,
    max_contact_body_triplets: int,
    dof_block_base: int,
    extent_capacity: int,
    n_extent_slots: int = 0,
    genesis_legacy_sort_reduce: bool = False,
) -> GlobalLinearSystem.Data:
    if (
        min(
            n_block_rows,
            n_elastic_triplets,
            max_contact_body_triplets,
            dof_block_base,
            extent_capacity,
            n_extent_slots,
        )
        < 0
    ):
        raise ValueError("Global linear system sizes must be non-negative")
    if n_extent_slots > extent_capacity:
        raise ValueError("Global linear system extent count exceeds capacity")
    total_dof = n_block_rows * 3
    extent_storage = max(extent_capacity, 1)
    dof_storage = max(total_dof, 1)

    def scalar(value):
        result = qd.ndarray(qd.i32, shape=())
        result.from_numpy(np.array(value, dtype=np.int32))
        return result

    extent_slots = qd.ndarray(qd.i32, shape=(extent_storage,))
    extent_offsets = qd.ndarray(qd.i32, shape=(extent_storage,))
    extent_slots.from_numpy(np.zeros(extent_storage, dtype=np.int32))
    extent_offsets.from_numpy(np.full(extent_storage, n_elastic_triplets, dtype=np.int32))
    x_sol = qd.ndarray(qd.f64, shape=(dof_storage,))
    b_rhs = qd.ndarray(qd.f64, shape=(dof_storage,))
    x_sol.from_numpy(np.zeros(dof_storage, dtype=np.float64))
    b_rhs.from_numpy(np.zeros(dof_storage, dtype=np.float64))
    data = GlobalLinearSystem.Data()
    data.n_block_rows_host = n_block_rows
    data.total_dof_host = total_dof
    data.extent_capacity = extent_capacity
    data.n_block_rows = scalar(n_block_rows)
    data.total_dof = scalar(total_dof)
    data.dof_block_base = scalar(dof_block_base)
    data.n_extent_slots = scalar(n_extent_slots)
    data.n_elastic = scalar(n_elastic_triplets)
    data.required_block_rows = scalar(n_block_rows)
    data.extent_slots = extent_slots
    data.extent_offsets = extent_offsets
    data.matrix = get_bcoo_matrix(
        shape=(n_block_rows, n_block_rows),
        block_shape=(3, 3),
        value_type=qd.f64,
        symmetric=True,
        initial_triplets=n_elastic_triplets,
        max_triplets=n_elastic_triplets + max_contact_body_triplets,
        genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
    )
    data.x_sol = x_sol
    data.b_rhs = b_rhs
    return data


@qd.func(requires_top_level=True)
def derive_extents(data: qd.template()):
    for _ in range(1):
        total = qd.i32(0)
        for slot in range(data.n_extent_slots[()]):
            data.extent_offsets[slot] = total
            total = total + data.extent_slots[slot]
        data.n_elastic[()] = total
        set_bcoo_n_triplets(data.matrix, total)


@qd.func(requires_top_level=True)
def compute_n_triplets(data: qd.template(), contact_data: qd.template()):
    for _ in range(1):
        total = data.n_elastic[()] + contact_data.n_unique_triplets[()]
        set_bcoo_n_triplets(data.matrix, total)


@qd.func(requires_top_level=True)
def zero_rhs(data: qd.template()):
    for i in range(data.total_dof[()]):
        data.b_rhs[i] = qd.f64(0.0)


@qd.func(requires_top_level=True)
def pcg_apply_operator(
    data: qd.template(),
    _linear_system_data: qd.template(),
    direction: qd.template(),
    output: qd.template(),
):
    sym_bcoo_spmv_naive(data.matrix, direction, output)
