from __future__ import annotations

import numpy as np
import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from ..sim_system import SimData, SimSystem
from .finite_element_method import FiniteElementMethod


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class QuadraticBending(SimSystem):
    """Bergou quadratic shell-bending constitution."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible hinge data and extent metadata."""

        n_hinges: qd.Ndarray
        hinge_indices: qd.Ndarray
        k: qd.Ndarray
        Q0: qd.Ndarray
        vert_bend_k: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._inputs = None

    def build(self) -> None:
        self.fem_system = self.require(FiniteElementMethod)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.init_action = self.create_action(self.init)
        self.extent_action = self.create_action(report_quadratic_bending_extent, self.fem_system.data, self.data)
        self.assemble_action = self.create_action(assemble_quadratic_bending, self.fem_system.data, self.data)
        self.energy_action = self.create_action(compute_quadratic_bending_energy, self.fem_system.data, self.data)
        self.global_linear_system_system.on_subsystem(
            extent=self.extent_action,
            assemble=self.assemble_action,
        )
        self.fem_system.on_constitution(
            self,
            self.init_action,
            self.extent_action,
            self.assemble_action,
            self.energy_action,
        )

    def wire_data(
        self,
        hinge_indices: np.ndarray,
        bending_stiffness: np.ndarray,
        Q0: np.ndarray,
        vert_bend_k: np.ndarray,
    ) -> None:
        if hasattr(self.data, "n_hinges"):
            raise RuntimeError("QuadraticBending data is already initialized")
        self._inputs = _quadratic_bending_inputs(
            hinge_indices,
            bending_stiffness,
            Q0,
            vert_bend_k,
        )

    def init(self) -> None:
        if self._inputs is None:
            raise RuntimeError("QuadraticBending data has not been wired")
        hinges, stiffness, matrices, vertex_stiffness = self._inputs
        n_hinges = len(hinges)
        hinge_capacity = max(n_hinges, 1)
        vertex_capacity = max(len(vertex_stiffness), 1)

        def array(dtype, shape, values):
            result = qd.ndarray(dtype, shape=shape)
            result.from_numpy(values)
            return result

        self.data.n_hinges = array(qd.i32, (), np.array(n_hinges, dtype=np.int32))
        self.data.hinge_indices = array(
            qd.i32,
            (hinge_capacity, 4),
            hinges if n_hinges else np.zeros((hinge_capacity, 4), dtype=np.int32),
        )
        self.data.k = array(
            qd.f64,
            (hinge_capacity,),
            stiffness if n_hinges else np.zeros(hinge_capacity, dtype=np.float64),
        )
        self.data.Q0 = array(
            qd.f64,
            (hinge_capacity, 16),
            matrices if n_hinges else np.zeros((hinge_capacity, 16), dtype=np.float64),
        )
        self.data.vert_bend_k = array(
            qd.f64,
            (vertex_capacity,),
            vertex_stiffness if len(vertex_stiffness) else np.zeros(vertex_capacity, dtype=np.float64),
        )
        self._inputs = None


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
    fem: qd.template(),  # FiniteElementMethod.Data
    data: qd.template(),  # QuadraticBending.Data
    global_linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    linear_system_id: qd.template(),  # int
):
    for _ in range(1):
        global_linear_system_data.set_subsystem_extent(linear_system_id, data.n_hinges[()] * 10)


@qd.func(requires_top_level=True)
def assemble_quadratic_bending(
    fem: qd.template(),  # FiniteElementMethod.Data
    data: qd.template(),  # QuadraticBending.Data
    sim_config: qd.template(),  # SimConfig.Data
    global_linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    linear_system_id: qd.template(),  # int
):
    for i in range(data.n_hinges[()]):
        triplet_offset = global_linear_system_data.subsystem_offset(linear_system_id)
        dt2 = sim_config.dt[()] * sim_config.dt[()]
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
                        global_linear_system_data.atomic_add_rhs(
                            fem.dof_offset[()] + verts[left] * 3 + axis,
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
                    global_linear_system_data.matrix.write_triplet(
                        slot,
                        fem.dof_offset[()] // 3 + verts[left],
                        fem.dof_offset[()] // 3 + verts[right],
                        block,
                    )
                    slot = slot + 1


@qd.func(requires_top_level=True)
def compute_quadratic_bending_energy(
    fem: qd.template(),  # FiniteElementMethod.Data
    data: qd.template(),  # QuadraticBending.Data
    sim_config: qd.template(),  # SimConfig.Data
):
    for i in range(data.n_hinges[()]):
        dt2 = sim_config.dt[()] * sim_config.dt[()]
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
