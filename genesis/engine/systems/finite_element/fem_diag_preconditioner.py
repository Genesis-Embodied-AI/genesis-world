from __future__ import annotations

import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from ..sim_system import SimSystem
from .finite_element_method import FiniteElementMethod


@qd.data_oriented
class FEMDiagPreconditioner(SimSystem):
    """FEM 3x3 block-diagonal preconditioner."""

    def build(self) -> None:
        self.fem = self.require(FiniteElementMethod)
        self.global_linear_system = self.require(GlobalLinearSystem)

    def init(self) -> None:
        self.precond_inv_diag = qd.ndarray(qd.f64, shape=(self.fem.vert_capacity_host, 9))

    @qd.func(requires_top_level=True)
    def memset(self):
        for i_vert in range(self.fem.n_fem_verts[()]):
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    self.precond_inv_diag[i_vert, i * 3 + j] = qd.f64(i == j)

    @qd.func(requires_top_level=True)
    def gather(self):
        block_offset = self.fem.dof_offset[()] // 3
        for i_entry in range(self.global_linear_system.bcoo_nnz[()]):
            row = self.global_linear_system.bcoo_row[i_entry]
            col = self.global_linear_system.bcoo_col[i_entry]
            if row == col and row >= block_offset and row < block_offset + self.fem.n_fem_verts[()]:
                local = row - block_offset
                for component in qd.static(range(9)):
                    self.precond_inv_diag[local, component] = self.global_linear_system.bcoo_val[
                        i_entry * 9 + component
                    ]

    @qd.func(requires_top_level=True)
    def invert(self):
        for i_vert in range(self.fem.n_fem_verts[()]):
            block = qd.Matrix.zero(qd.f64, 3, 3)
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    block[i, j] = self.precond_inv_diag[i_vert, i * 3 + j]
            block = block.inverse()
            for i in qd.static(range(3)):
                for j in qd.static(range(3)):
                    self.precond_inv_diag[i_vert, i * 3 + j] = block[i, j]

    @qd.func(requires_top_level=True)
    def apply(self, residual: qd.template(), result: qd.template()):
        for i_vert in range(self.fem.n_fem_verts[()]):
            global_offset = self.fem.dof_offset[()] + i_vert * 3
            for i in qd.static(range(3)):
                value = qd.f64(0.0)
                for j in qd.static(range(3)):
                    value = value + self.precond_inv_diag[i_vert, i * 3 + j] * residual[global_offset + j]
                result[global_offset + i] = value
