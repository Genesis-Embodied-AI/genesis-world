from __future__ import annotations

import numpy as np
import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from ..sim_system import SimData, SimSystem
from .finite_element_method import FiniteElementMethod


class FEMDiagPreconditioner(SimSystem):
    """FEM 3x3 block-diagonal preconditioner."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible FEM diagonal blocks and global range."""

        vert_capacity: int
        n_fem_verts: qd.Ndarray
        dof_offset: qd.Ndarray
        precond_inv_diag: qd.Ndarray

    def __init__(self, data: Data) -> None:
        super().__init__()
        self.data = data

    def build(self) -> None:
        from ..pcg_solver import PCGSolver

        self.fem_system = self.require(FiniteElementMethod)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.pcg_solver_system = self.require(PCGSolver)
        self.pcg_preconditioner_action = self.create_action(pcg_apply_preconditioner, self.data)
        self.pcg_solver_system.on_shared_preconditioner(self.pcg_preconditioner_action)


def get_fem_diag_preconditioner_data(
    *,
    vert_capacity: int,
    n_fem_verts: int,
    dof_offset: int,
) -> FEMDiagPreconditioner.Data:
    if min(vert_capacity, n_fem_verts, dof_offset) < 0:
        raise ValueError("FEM diagonal preconditioner dimensions must be non-negative")
    if n_fem_verts > vert_capacity:
        raise ValueError("FEM diagonal preconditioner live vertex count exceeds capacity")
    storage_capacity = max(vert_capacity, 1)
    n_fem_verts_data = qd.ndarray(qd.i32, shape=())
    dof_offset_data = qd.ndarray(qd.i32, shape=())
    precond_inv_diag = qd.ndarray(qd.f64, shape=(storage_capacity, 9))
    n_fem_verts_data.from_numpy(np.array(n_fem_verts, dtype=np.int32))
    dof_offset_data.from_numpy(np.array(dof_offset, dtype=np.int32))
    precond_inv_diag.from_numpy(np.zeros((storage_capacity, 9), dtype=np.float64))
    data = FEMDiagPreconditioner.Data()
    data.vert_capacity = vert_capacity
    data.n_fem_verts = n_fem_verts_data
    data.dof_offset = dof_offset_data
    data.precond_inv_diag = precond_inv_diag
    return data


@qd.func(requires_top_level=True)
def initialize_fem_diag_preconditioner(data: qd.template()):
    for i_vert in range(data.n_fem_verts[()]):
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                data.precond_inv_diag[i_vert, i * 3 + j] = qd.f64(i == j)


@qd.func(requires_top_level=True)
def gather_fem_diag_preconditioner(data: qd.template(), linear_system_data: qd.template()):
    block_offset = data.dof_offset[()] // 3
    matrix = linear_system_data.matrix
    for i_entry in range(matrix.bcoo_nnz[()]):
        row = matrix.bcoo_row[i_entry]
        col = matrix.bcoo_col[i_entry]
        if row == col and row >= block_offset and row < block_offset + data.n_fem_verts[()]:
            local = row - block_offset
            for component in qd.static(range(9)):
                data.precond_inv_diag[local, component] = matrix.bcoo_val[i_entry * 9 + component]


@qd.func(requires_top_level=True)
def invert_fem_diag_preconditioner(data: qd.template()):
    for i_vert in range(data.n_fem_verts[()]):
        block = qd.Matrix.zero(qd.f64, 3, 3)
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                block[i, j] = data.precond_inv_diag[i_vert, i * 3 + j]
        block = block.inverse()
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                data.precond_inv_diag[i_vert, i * 3 + j] = block[i, j]


@qd.func(requires_top_level=True)
def pcg_apply_preconditioner(
    data: qd.template(),
    residual: qd.template(),
    output: qd.template(),
):
    apply_fem_diag_preconditioner(data, residual, output)


@qd.func(requires_top_level=True)
def apply_fem_diag_preconditioner(
    data: qd.template(),
    residual: qd.template(),
    result: qd.template(),
):
    for i_vert in range(data.n_fem_verts[()]):
        global_offset = data.dof_offset[()] + i_vert * 3
        for i in qd.static(range(3)):
            value = qd.f64(0.0)
            for j in qd.static(range(3)):
                value = value + data.precond_inv_diag[i_vert, i * 3 + j] * residual[global_offset + j]
            result[global_offset + i] = value
