from __future__ import annotations

import numpy as np
import quadrants as qd

from ..bcoo_matrix import write_bcoo_block
from ..sim_system import SimData
from .fem_constitution import FEMConstitution


class QuadraticBending(FEMConstitution):
    """Bergou quadratic shell-bending constitution."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible hinge data and extent metadata."""

        n_hinges_host: int
        extent_slot: int
        n_hinges: qd.Ndarray
        hinge_indices: qd.Ndarray
        k: qd.Ndarray
        Q0: qd.Ndarray
        vert_bend_k: qd.Ndarray

    def __init__(self, data: Data | None = None) -> None:
        super().__init__()
        self.data = (
            get_quadratic_bending_data(
                np.empty((0, 4), dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty((0, 4, 4), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
            if data is None
            else data
        )

    def create_constitution_actions(self):
        return (
            self.create_action(report_quadratic_bending_extent, self.fem_system.data, self.data),
            self.create_action(assemble_quadratic_bending, self.fem_system.data, self.data),
            self.create_action(compute_quadratic_bending_energy, self.fem_system.data, self.data),
        )

    def wire_data(
        self,
        hinge_indices: np.ndarray,
        bending_stiffness: np.ndarray,
        Q0: np.ndarray,
        vert_bend_k: np.ndarray,
    ) -> None:
        wire_quadratic_bending_data(
            self.data,
            hinge_indices,
            bending_stiffness,
            Q0,
            vert_bend_k,
        )

    def triplet_count(self) -> int:
        return self.data.n_hinges_host * 10

    @property
    def n_hinges(self):
        return self.data.n_hinges

    @property
    def hinge_indices(self):
        return self.data.hinge_indices

    @property
    def k(self):
        return self.data.k

    @property
    def Q0(self):
        return self.data.Q0


def _quadratic_bending_inputs(
    hinge_indices: np.ndarray,
    bending_stiffness: np.ndarray,
    Q0: np.ndarray,
    vert_bend_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hinges = np.ascontiguousarray(hinge_indices, dtype=np.int32).reshape(-1, 4)
    stiffness = np.ascontiguousarray(bending_stiffness, dtype=np.float64).reshape(-1)
    matrices = np.ascontiguousarray(Q0, dtype=np.float64).reshape(-1, 16)
    vertex_stiffness = np.ascontiguousarray(vert_bend_k, dtype=np.float64).reshape(-1)
    if len(stiffness) != len(hinges) or len(matrices) != len(hinges):
        raise ValueError("QuadraticBending wire-data lengths must match")
    return hinges, stiffness, matrices, vertex_stiffness


def get_quadratic_bending_data(
    hinge_indices: np.ndarray,
    bending_stiffness: np.ndarray,
    Q0: np.ndarray,
    vert_bend_k: np.ndarray,
) -> QuadraticBending.Data:
    hinges, stiffness, matrices, vertex_stiffness = _quadratic_bending_inputs(
        hinge_indices,
        bending_stiffness,
        Q0,
        vert_bend_k,
    )
    n_hinges = len(hinges)
    hinge_capacity = max(n_hinges, 1)
    vertex_capacity = max(len(vertex_stiffness), 1)

    def array(dtype, shape, values):
        result = qd.ndarray(dtype, shape=shape)
        result.from_numpy(values)
        return result

    data = QuadraticBending.Data()
    data.n_hinges_host = n_hinges
    data.extent_slot = -1
    data.n_hinges = array(qd.i32, (), np.array(n_hinges, dtype=np.int32))
    data.hinge_indices = array(
        qd.i32,
        (hinge_capacity, 4),
        hinges if n_hinges else np.zeros((hinge_capacity, 4), dtype=np.int32),
    )
    data.k = array(
        qd.f64,
        (hinge_capacity,),
        stiffness if n_hinges else np.zeros(hinge_capacity, dtype=np.float64),
    )
    data.Q0 = array(
        qd.f64,
        (hinge_capacity, 16),
        matrices if n_hinges else np.zeros((hinge_capacity, 16), dtype=np.float64),
    )
    data.vert_bend_k = array(
        qd.f64,
        (vertex_capacity,),
        vertex_stiffness if len(vertex_stiffness) else np.zeros(vertex_capacity, dtype=np.float64),
    )
    return data


def wire_quadratic_bending_data(
    data: QuadraticBending.Data,
    hinge_indices: np.ndarray,
    bending_stiffness: np.ndarray,
    Q0: np.ndarray,
    vert_bend_k: np.ndarray,
) -> None:
    hinges, stiffness, matrices, vertex_stiffness = _quadratic_bending_inputs(
        hinge_indices,
        bending_stiffness,
        Q0,
        vert_bend_k,
    )
    n_hinges = len(hinges)
    hinge_capacity = max(n_hinges, 1)
    vertex_capacity = max(len(vertex_stiffness), 1)
    data.n_hinges_host = n_hinges
    data.n_hinges.from_numpy(np.array(n_hinges, dtype=np.int32))
    if data.hinge_indices.shape[0] != hinge_capacity:
        data.hinge_indices = qd.ndarray(qd.i32, shape=(hinge_capacity, 4))
        data.k = qd.ndarray(qd.f64, shape=(hinge_capacity,))
        data.Q0 = qd.ndarray(qd.f64, shape=(hinge_capacity, 16))
    if data.vert_bend_k.shape[0] != vertex_capacity:
        data.vert_bend_k = qd.ndarray(qd.f64, shape=(vertex_capacity,))
    if n_hinges:
        data.hinge_indices.from_numpy(hinges)
        data.k.from_numpy(stiffness)
        data.Q0.from_numpy(matrices)
    else:
        data.hinge_indices.from_numpy(np.zeros((hinge_capacity, 4), dtype=np.int32))
        data.k.from_numpy(np.zeros(hinge_capacity, dtype=np.float64))
        data.Q0.from_numpy(np.zeros((hinge_capacity, 16), dtype=np.float64))
    data.vert_bend_k.from_numpy(
        vertex_stiffness if len(vertex_stiffness) else np.zeros(vertex_capacity, dtype=np.float64)
    )


@qd.func(requires_top_level=True)
def report_quadratic_bending_extent(
    fem: qd.template(),
    data: qd.template(),
    global_linear_system_data: qd.template(),
):
    for _ in range(1):
        global_linear_system_data.extent_slots[data.extent_slot] = data.n_hinges[()] * 10


@qd.func(requires_top_level=True)
def assemble_quadratic_bending(
    fem: qd.template(),
    data: qd.template(),
    sim_config: qd.template(),
    global_linear_system_data: qd.template(),
):
    triplet_offset = global_linear_system_data.extent_offsets[data.extent_slot]
    dt2 = sim_config.dt[()] * sim_config.dt[()]
    for i in range(data.n_hinges[()]):
        if global_linear_system_data.matrix.triplet_overflow[()] == 0:
            verts = qd.Vector(
                [
                    data.hinge_indices[i, 0],
                    data.hinge_indices[i, 1],
                    data.hinge_indices[i, 2],
                    data.hinge_indices[i, 3],
                ],
                dt=qd.i32,
            )
            stiffness = data.k[i]

            for left in qd.static(range(4)):
                if fem.is_fixed[verts[left]] == 0:
                    for axis in qd.static(range(3)):
                        gradient = qd.f64(0.0)
                        for right in qd.static(range(4)):
                            gradient = gradient + data.Q0[i, left * 4 + right] * fem.x[verts[right], axis]
                        qd.atomic_add(
                            global_linear_system_data.b_rhs[fem.dof_offset[()] + verts[left] * 3 + axis],
                            stiffness * dt2 * gradient,
                        )

            slot = triplet_offset + i * 10
            for left in qd.static(range(4)):
                for right in qd.static(range(left, 4)):
                    block = qd.Matrix.zero(qd.f64, 3, 3)
                    if fem.is_fixed[verts[left]] == 0 and fem.is_fixed[verts[right]] == 0:
                        value = stiffness * dt2 * data.Q0[i, left * 4 + right]
                        for axis in qd.static(range(3)):
                            block[axis, axis] = value
                    write_bcoo_block(
                        global_linear_system_data.matrix,
                        slot,
                        fem.dof_offset[()] // 3 + verts[left],
                        fem.dof_offset[()] // 3 + verts[right],
                        block,
                    )
                    slot = slot + 1


@qd.func(requires_top_level=True)
def compute_quadratic_bending_energy(
    fem: qd.template(),
    data: qd.template(),
    sim_config: qd.template(),
):
    dt2 = sim_config.dt[()] * sim_config.dt[()]
    for i in range(data.n_hinges[()]):
        value = qd.f64(0.0)
        for left in qd.static(range(4)):
            left_vertex = data.hinge_indices[i, left]
            for right in qd.static(range(4)):
                right_vertex = data.hinge_indices[i, right]
                dot = qd.f64(0.0)
                for axis in qd.static(range(3)):
                    dot = dot + fem.x[left_vertex, axis] * fem.x[right_vertex, axis]
                value = value + data.Q0[i, left * 4 + right] * dot
        qd.atomic_add(fem.fem_energy[()], 0.5 * data.k[i] * value * dt2)
