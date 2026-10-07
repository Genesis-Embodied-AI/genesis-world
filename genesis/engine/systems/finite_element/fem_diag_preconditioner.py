from __future__ import annotations

import numpy as np
import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from ..sim_system import SimData, SimSystem
from .finite_element_method import FiniteElementMethod


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class FEMDiagPreconditioner(SimSystem):
    """FEM 3x3 block-diagonal preconditioner."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible FEM diagonal blocks and global range."""

        n_fem_verts: qd.Ndarray
        dof_offset: qd.Ndarray
        precond_inv_diag: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()

    def build(self) -> None:
        from ..pcg_solver import PCGSolver

        self.fem_system = self.require(FiniteElementMethod)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.pcg_solver_system = self.require(PCGSolver)
        self.init_action = self.create_action(self.init)
        self.fem_system.on_preconditioner(self.init_action)
        self.pcg_preconditioner_action = self.create_action(pcg_apply_preconditioner, self.data)
        self.pcg_solver_system.on_shared_preconditioner(self.pcg_preconditioner_action)

    def init(self) -> None:
        storage_capacity = self.fem_system.data.x.shape[0]
        self.data.n_fem_verts = self.fem_system.data.n_fem_verts
        self.data.dof_offset = self.fem_system.data.dof_offset
        self.data.precond_inv_diag = qd.ndarray(qd.f64, shape=(storage_capacity, 9))
        self.data.precond_inv_diag.from_numpy(np.zeros((storage_capacity, 9), dtype=np.float64))


@qd.func(requires_top_level=True)
def initialize_fem_diag_preconditioner(
    data: qd.template(),  # FEMDiagPreconditioner.Data
):
    for i_vert in range(data.n_fem_verts[()]):
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                data.precond_inv_diag[i_vert, i * 3 + j] = qd.f64(i == j)


@qd.func(requires_top_level=True)
def gather_fem_diag_preconditioner(
    data: qd.template(),  # FEMDiagPreconditioner.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
):
    matrix = linear_system_data.matrix
    for i_entry in range(matrix.bcoo_nnz[()]):
        block_offset = data.dof_offset[()] // 3
        row, col, block = matrix.read_bcoo(i_entry)
        if row == col and row >= block_offset and row < block_offset + data.n_fem_verts[()]:
            local = row - block_offset
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    data.precond_inv_diag[local, i * 3 + j] = block[i, j]


@qd.func(requires_top_level=True)
def invert_fem_diag_preconditioner(
    data: qd.template(),  # FEMDiagPreconditioner.Data
):
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
    data: qd.template(),  # FEMDiagPreconditioner.Data
    residual: qd.template(),  # qd.Ndarray
    output: qd.template(),  # qd.Ndarray
):
    apply_fem_diag_preconditioner(data, residual, output)


@qd.func(requires_top_level=True)
def apply_fem_diag_preconditioner(
    data: qd.template(),  # FEMDiagPreconditioner.Data
    residual: qd.template(),  # qd.Ndarray
    result: qd.template(),  # qd.Ndarray
):
    for i_vert in range(data.n_fem_verts[()]):
        global_offset = data.dof_offset[()] + i_vert * 3
        for i in qd.static(range(3)):
            value = qd.f64(0.0)
            for j in qd.static(range(3)):
                value = value + data.precond_inv_diag[i_vert, i * 3 + j] * residual[global_offset + j]
            result[global_offset + i] = value
