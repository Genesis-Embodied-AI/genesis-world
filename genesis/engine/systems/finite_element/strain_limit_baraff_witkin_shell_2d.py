from __future__ import annotations

import numpy as np
import quadrants as qd

from ..bcoo_matrix import write_bcoo_block
from ..sim_system import SimData
from .fem_constitution import FEMConstitution
from .strain_limit_bws_shell_2d import Ds3x2, E, F3x2, ddEddF, dEdF, dFdX


class StrainLimitBaraffWitkinShell2D(FEMConstitution):
    """Baraff-Witkin shell membrane constitution."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible membrane parameters and extent metadata."""

        n_tris_host: int
        extent_slot: int
        n_tris: qd.Ndarray
        tri_indices: qd.Ndarray
        mu: qd.Ndarray
        lambda_: qd.Ndarray
        strain_limit_multiplier: qd.Ndarray

    def __init__(self, data: Data | None = None) -> None:
        super().__init__()
        self.data = (
            get_strain_limit_baraff_witkin_shell_2d_data(
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
            if data is None
            else data
        )

    def create_constitution_actions(self):
        return (
            self.create_action(report_strain_limit_extent, self.fem_system.data, self.data),
            self.create_action(assemble_strain_limit, self.fem_system.data, self.data),
            self.create_action(compute_strain_limit_energy, self.fem_system.data, self.data),
        )

    def wire_data(
        self,
        tri_indices: np.ndarray,
        mu: np.ndarray,
        lambda_param: np.ndarray,
        strain_limit_multiplier: np.ndarray,
    ) -> None:
        wire_strain_limit_baraff_witkin_shell_2d_data(
            self.data,
            tri_indices,
            mu,
            lambda_param,
            strain_limit_multiplier,
        )

    def triplet_count(self) -> int:
        return self.data.n_tris_host * 6


def _strain_limit_inputs(
    tri_indices: np.ndarray,
    mu: np.ndarray,
    lambda_param: np.ndarray,
    strain_limit_multiplier: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.ascontiguousarray(tri_indices, dtype=np.int32).reshape(-1)
    mu_values = np.ascontiguousarray(mu, dtype=np.float64).reshape(-1)
    lambda_values = np.ascontiguousarray(lambda_param, dtype=np.float64).reshape(-1)
    multiplier_values = np.ascontiguousarray(strain_limit_multiplier, dtype=np.float64).reshape(-1)
    n_tris = len(triangles)
    if not (len(mu_values) == n_tris and len(lambda_values) == n_tris and len(multiplier_values) == n_tris):
        raise ValueError("StrainLimitBaraffWitkinShell2D wire-data lengths must match")
    return triangles, mu_values, lambda_values, multiplier_values


def get_strain_limit_baraff_witkin_shell_2d_data(
    tri_indices: np.ndarray,
    mu: np.ndarray,
    lambda_param: np.ndarray,
    strain_limit_multiplier: np.ndarray,
) -> StrainLimitBaraffWitkinShell2D.Data:
    triangles, mu_values, lambda_values, multiplier_values = _strain_limit_inputs(
        tri_indices,
        mu,
        lambda_param,
        strain_limit_multiplier,
    )
    n_tris = len(triangles)
    capacity = max(n_tris, 1)

    def array(dtype, values):
        result = qd.ndarray(dtype, shape=(capacity,))
        storage = values if n_tris else np.zeros(capacity, dtype=values.dtype)
        result.from_numpy(storage)
        return result

    n_tris_data = qd.ndarray(qd.i32, shape=())
    n_tris_data.from_numpy(np.array(n_tris, dtype=np.int32))
    data = StrainLimitBaraffWitkinShell2D.Data()
    data.n_tris_host = n_tris
    data.extent_slot = -1
    data.n_tris = n_tris_data
    data.tri_indices = array(qd.i32, triangles)
    data.mu = array(qd.f64, mu_values)
    data.lambda_ = array(qd.f64, lambda_values)
    data.strain_limit_multiplier = array(qd.f64, multiplier_values)
    return data


def wire_strain_limit_baraff_witkin_shell_2d_data(
    data: StrainLimitBaraffWitkinShell2D.Data,
    tri_indices: np.ndarray,
    mu: np.ndarray,
    lambda_param: np.ndarray,
    strain_limit_multiplier: np.ndarray,
) -> None:
    triangles, mu_values, lambda_values, multiplier_values = _strain_limit_inputs(
        tri_indices,
        mu,
        lambda_param,
        strain_limit_multiplier,
    )
    n_tris = len(triangles)
    capacity = max(n_tris, 1)
    data.n_tris_host = n_tris
    data.n_tris.from_numpy(np.array(n_tris, dtype=np.int32))
    if data.tri_indices.shape[0] != capacity:
        data.tri_indices = qd.ndarray(qd.i32, shape=(capacity,))
        data.mu = qd.ndarray(qd.f64, shape=(capacity,))
        data.lambda_ = qd.ndarray(qd.f64, shape=(capacity,))
        data.strain_limit_multiplier = qd.ndarray(qd.f64, shape=(capacity,))
    if n_tris:
        data.tri_indices.from_numpy(triangles)
        data.mu.from_numpy(mu_values)
        data.lambda_.from_numpy(lambda_values)
        data.strain_limit_multiplier.from_numpy(multiplier_values)
    else:
        data.tri_indices.from_numpy(np.zeros(capacity, dtype=np.int32))
        data.mu.from_numpy(np.zeros(capacity, dtype=np.float64))
        data.lambda_.from_numpy(np.zeros(capacity, dtype=np.float64))
        data.strain_limit_multiplier.from_numpy(np.zeros(capacity, dtype=np.float64))


@qd.func(requires_top_level=True)
def report_strain_limit_extent(
    fem: qd.template(),
    data: qd.template(),
    global_linear_system_data: qd.template(),
):
    for _ in range(1):
        global_linear_system_data.extent_slots[data.extent_slot] = data.n_tris[()] * 6


@qd.func(requires_top_level=True)
def assemble_strain_limit(
    fem: qd.template(),
    data: qd.template(),
    sim_config: qd.template(),
    global_linear_system_data: qd.template(),
):
    triplet_offset = global_linear_system_data.extent_offsets[data.extent_slot]
    for i in range(data.n_tris[()]):
        if global_linear_system_data.matrix.triplet_overflow[()] == 0:
            tri = data.tri_indices[i]
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
                data.lambda_[i],
                data.mu[i],
                data.strain_limit_multiplier[i],
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
                data.lambda_[i],
                data.mu[i],
                data.strain_limit_multiplier[i],
            )
            hessian = (dfdx.transpose() @ hessian_F @ dfdx) * scale

            for local in qd.static(range(3)):
                if fem.is_fixed[verts[local]] == 0:
                    for axis in qd.static(range(3)):
                        qd.atomic_add(
                            global_linear_system_data.b_rhs[fem.dof_offset[()] + verts[local] * 3 + axis],
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
                    write_bcoo_block(
                        global_linear_system_data.matrix,
                        slot,
                        fem.dof_offset[()] // 3 + verts[left],
                        fem.dof_offset[()] // 3 + verts[right],
                        block,
                    )
                    slot = slot + 1


@qd.func(requires_top_level=True)
def compute_strain_limit_energy(
    fem: qd.template(),
    data: qd.template(),
    sim_config: qd.template(),
):
    for i in range(data.n_tris[()]):
        tri = data.tri_indices[i]
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
            data.lambda_[i],
            data.mu[i],
            data.strain_limit_multiplier[i],
        )
        qd.atomic_add(fem.fem_energy[()], psi * volume * sim_config.dt[()] ** 2)
