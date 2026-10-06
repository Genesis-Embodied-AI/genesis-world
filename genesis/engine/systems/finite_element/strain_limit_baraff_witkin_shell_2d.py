from __future__ import annotations

import numpy as np
import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from .fem_constitution import FEMConstitution
from .strain_limit_bws_shell_2d import Ds3x2, E, F3x2, ddEddF, dEdF, dFdX


@qd.data_oriented
class StrainLimitBaraffWitkinShell2D(FEMConstitution):
    """Baraff-Witkin shell membrane constitution."""

    def __init__(self) -> None:
        super().__init__()
        self.n_tris_host = 0
        self.extent_slot = -1

    def do_build_constitution(self) -> None:
        self.extent_slot = self.require(GlobalLinearSystem).register_extent_slot()

    def wire_data(
        self,
        tri_indices: np.ndarray,
        mu: np.ndarray,
        lambda_param: np.ndarray,
        strain_limit_multiplier: np.ndarray,
    ) -> None:
        self.n_tris_host = len(tri_indices)
        if not (
            len(mu) == self.n_tris_host
            and len(lambda_param) == self.n_tris_host
            and len(strain_limit_multiplier) == self.n_tris_host
        ):
            raise ValueError("StrainLimitBaraffWitkinShell2D wire-data lengths must match")

        capacity = max(self.n_tris_host, 1)
        self.n_tris = qd.ndarray(qd.i32, shape=())
        self.tri_indices = qd.ndarray(qd.i32, shape=(capacity,))
        self.mu = qd.ndarray(qd.f64, shape=(capacity,))
        self.lambda_ = qd.ndarray(qd.f64, shape=(capacity,))
        self.strain_limit_multiplier = qd.ndarray(qd.f64, shape=(capacity,))
        self.n_tris.from_numpy(np.array(self.n_tris_host, dtype=np.int32))
        self.tri_indices.from_numpy(np.asarray(tri_indices, dtype=np.int32))
        self.mu.from_numpy(np.asarray(mu, dtype=np.float64))
        self.lambda_.from_numpy(np.asarray(lambda_param, dtype=np.float64))
        self.strain_limit_multiplier.from_numpy(np.asarray(strain_limit_multiplier, dtype=np.float64))

    def triplet_count(self) -> int:
        return self.n_tris_host * 6

    @qd.func(requires_top_level=True)
    def report_extent(self, fem: qd.template(), global_linear_system: qd.template()):
        for _ in range(1):
            global_linear_system.extent_slots[self.extent_slot] = self.n_tris[()] * 6

    @qd.func(requires_top_level=True)
    def assemble(
        self,
        fem: qd.template(),
        sim_config: qd.template(),
        global_linear_system: qd.template(),
    ):
        triplet_offset = global_linear_system.extent_offsets[self.extent_slot]
        for i in range(self.n_tris[()]):
            if global_linear_system.triplet_overflow[()] == 0:
                tri = self.tri_indices[i]
                verts = qd.Vector(
                    [
                        fem.tri_indices[tri, 0],
                        fem.tri_indices[tri, 1],
                        fem.tri_indices[tri, 2],
                    ],
                    dt=qd.i32,
                )
                x0 = qd.Vector([fem.x[verts[0], 0], fem.x[verts[0], 1], fem.x[verts[0], 2]])
                x1 = qd.Vector([fem.x[verts[1], 0], fem.x[verts[1], 1], fem.x[verts[1], 2]])
                x2 = qd.Vector([fem.x[verts[2], 0], fem.x[verts[2], 1], fem.x[verts[2], 2]])
                Dm_inv = qd.Matrix(
                    [
                        [fem.Dm_inv_2d[tri, 0], fem.Dm_inv_2d[tri, 1]],
                        [fem.Dm_inv_2d[tri, 2], fem.Dm_inv_2d[tri, 3]],
                    ]
                )
                F = F3x2(Ds3x2(x0, x1, x2), Dm_inv)
                dfdx = dFdX(Dm_inv)
                pk1 = dEdF(
                    F,
                    self.lambda_[i],
                    self.mu[i],
                    self.strain_limit_multiplier[i],
                )
                pk1_flat = qd.Vector.zero(qd.f64, 6)
                for axis in qd.static(range(3)):
                    pk1_flat[axis] = pk1[axis, 0]
                    pk1_flat[axis + 3] = pk1[axis, 1]

                thickness = (fem.thicknesses[verts[0]] + fem.thicknesses[verts[1]] + fem.thicknesses[verts[2]]) / 3.0
                scale = 2.0 * fem.rest_areas[tri] * thickness * sim_config.dt[()] ** 2
                gradient = (dfdx.transpose() @ pk1_flat) * scale
                hessian_F = ddEddF(
                    F,
                    self.lambda_[i],
                    self.mu[i],
                    self.strain_limit_multiplier[i],
                )
                hessian = (dfdx.transpose() @ hessian_F @ dfdx) * scale

                for local in qd.static(range(3)):
                    if fem.is_fixed[verts[local]] == 0:
                        for axis in qd.static(range(3)):
                            qd.atomic_add(
                                global_linear_system.b_rhs[fem.dof_offset[()] + verts[local] * 3 + axis],
                                gradient[local * 3 + axis],
                            )

                slot = triplet_offset + i * 6
                for left in qd.static(range(3)):
                    for right in qd.static(range(left, 3)):
                        block = qd.Matrix.zero(qd.f64, 3, 3)
                        if fem.is_fixed[verts[left]] == 0 and fem.is_fixed[verts[right]] == 0:
                            for row in qd.static(range(3)):
                                for col in qd.static(range(3)):
                                    block[row, col] = hessian[left * 3 + row, right * 3 + col]
                        global_linear_system.set_sym(
                            slot,
                            fem.dof_offset[()] // 3 + verts[left],
                            fem.dof_offset[()] // 3 + verts[right],
                            block,
                        )
                        slot = slot + 1

    @qd.func(requires_top_level=True)
    def energy(self, fem: qd.template(), sim_config: qd.template(), energy: qd.template()):
        for i in range(self.n_tris[()]):
            tri = self.tri_indices[i]
            v0 = fem.tri_indices[tri, 0]
            v1 = fem.tri_indices[tri, 1]
            v2 = fem.tri_indices[tri, 2]
            x0 = qd.Vector([fem.x[v0, 0], fem.x[v0, 1], fem.x[v0, 2]])
            x1 = qd.Vector([fem.x[v1, 0], fem.x[v1, 1], fem.x[v1, 2]])
            x2 = qd.Vector([fem.x[v2, 0], fem.x[v2, 1], fem.x[v2, 2]])
            Dm_inv = qd.Matrix(
                [
                    [fem.Dm_inv_2d[tri, 0], fem.Dm_inv_2d[tri, 1]],
                    [fem.Dm_inv_2d[tri, 2], fem.Dm_inv_2d[tri, 3]],
                ]
            )
            F = F3x2(Ds3x2(x0, x1, x2), Dm_inv)
            thickness = (fem.thicknesses[v0] + fem.thicknesses[v1] + fem.thicknesses[v2]) / 3.0
            volume = 2.0 * fem.rest_areas[tri] * thickness
            psi = E(
                F,
                self.lambda_[i],
                self.mu[i],
                self.strain_limit_multiplier[i],
            )
            qd.atomic_add(energy[()], psi * volume * sim_config.dt[()] ** 2)
