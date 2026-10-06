from __future__ import annotations

import numpy as np
import quadrants as qd

from ..bcoo_matrix import write_bcoo_block
from ..global_linear_system import GlobalLinearSystem
from ..sim_system import SimData, SimSystem
from .finite_element_method import FiniteElementMethod


class FEMBDF1(SimSystem):
    """Incremental-potential inertia for FEM vertices."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible BDF1 assembly metadata."""

        n_fem_verts_host: int
        extent_slot: int
        n_fem_verts: qd.Ndarray

    def __init__(self, data: Data) -> None:
        super().__init__()
        self.data = data

    def build(self) -> None:
        self.fem_system = self.require(FiniteElementMethod)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.data.extent_slot = self.global_linear_system_system.register_extent_slot()
        self.predict_action = self.create_action(predict_fem_bdf1, self.fem_system.data, self.data)
        self.extent_action = self.create_action(report_fem_bdf1_extent, self.fem_system.data, self.data)
        self.assemble_action = self.create_action(assemble_fem_bdf1, self.fem_system.data, self.data)
        self.energy_action = self.create_action(compute_fem_bdf1_energy, self.fem_system.data, self.data)
        self.fem_system.register_kinetic_actions(
            self,
            self.predict_action,
            self.extent_action,
            self.assemble_action,
            self.energy_action,
        )

    def wire_data(self, n_fem_verts: int) -> None:
        if n_fem_verts < 0:
            raise ValueError("FEM BDF1 vertex count must be non-negative")
        self.data.n_fem_verts_host = n_fem_verts
        self.data.n_fem_verts.from_numpy(np.array(n_fem_verts, dtype=np.int32))

    def triplet_count(self) -> int:
        return self.data.n_fem_verts_host


def get_fem_bdf1_data(n_fem_verts: int) -> FEMBDF1.Data:
    if n_fem_verts < 0:
        raise ValueError("FEM BDF1 vertex count must be non-negative")
    n_fem_verts_data = qd.ndarray(qd.i32, shape=())
    n_fem_verts_data.from_numpy(np.array(n_fem_verts, dtype=np.int32))
    data = FEMBDF1.Data()
    data.n_fem_verts_host = n_fem_verts
    data.extent_slot = -1
    data.n_fem_verts = n_fem_verts_data
    return data


@qd.func(requires_top_level=True)
def report_fem_bdf1_extent(
    fem: qd.template(),
    data: qd.template(),
    linear_system_data: qd.template(),
):
    for _ in range(1):
        linear_system_data.extent_slots[data.extent_slot] = fem.n_fem_verts[()]


@qd.func(requires_top_level=True)
def predict_fem_bdf1(
    fem: qd.template(),
    data: qd.template(),
    sim_config: qd.template(),
):
    for i_vertex in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            fem.x_tilde[i_vertex, axis] = (
                fem.x_prev[i_vertex, axis]
                + sim_config.dt[()] * fem.velocities[i_vertex, axis]
                + sim_config.dt[()] * sim_config.dt[()] * fem.gravity[i_vertex, axis]
            )


@qd.func(requires_top_level=True)
def compute_fem_bdf1_energy(
    fem: qd.template(),
    data: qd.template(),
    sim_config: qd.template(),
):
    for i_vertex in range(data.n_fem_verts[()]):
        if fem.is_fixed[i_vertex] == 0:
            value = qd.f64(0.0)
            for axis in qd.static(range(3)):
                difference = fem.x[i_vertex, axis] - fem.x_tilde[i_vertex, axis]
                value = value + difference * difference
            qd.atomic_add(fem.fem_energy[()], 0.5 * fem.masses[i_vertex] * value)


@qd.func(requires_top_level=True)
def assemble_fem_bdf1(
    fem: qd.template(),
    data: qd.template(),
    sim_config: qd.template(),
    global_linear_system_data: qd.template(),
):
    triplet_offset = global_linear_system_data.extent_offsets[data.extent_slot]
    for i_vertex in range(data.n_fem_verts[()]):
        if global_linear_system_data.matrix.triplet_overflow[()] == 0:
            block = qd.Matrix.identity(qd.f64, 3)
            if fem.is_fixed[i_vertex] == 0:
                mass = fem.masses[i_vertex]
                block = mass * block
                for axis in qd.static(range(3)):
                    gradient = mass * (fem.x[i_vertex, axis] - fem.x_tilde[i_vertex, axis])
                    global_linear_system_data.b_rhs[fem.dof_offset[()] + i_vertex * 3 + axis] = gradient
            write_bcoo_block(
                global_linear_system_data.matrix,
                triplet_offset + i_vertex,
                fem.dof_offset[()] // 3 + i_vertex,
                fem.dof_offset[()] // 3 + i_vertex,
                block,
            )
