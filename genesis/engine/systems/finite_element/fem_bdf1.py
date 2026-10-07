from __future__ import annotations

import numpy as np
import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from ..sim_system import SimSystem
from .finite_element_method import FiniteElementMethod


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class FEMBDF1(SimSystem):
    """Incremental-potential inertia for FEM vertices."""

    def __init__(self) -> None:
        super().__init__()

    def build(self) -> None:
        self.fem_system = self.require(FiniteElementMethod)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.init_action = self.create_action(self.init)
        self.predict_action = self.create_action(predict_fem_bdf1, self.fem_system.data)
        self.extent_action = self.create_action(report_fem_bdf1_extent, self.fem_system.data)
        self.assemble_action = self.create_action(assemble_fem_bdf1, self.fem_system.data)
        self.energy_action = self.create_action(compute_fem_bdf1_energy, self.fem_system.data)
        self.fem_system.on_kinetic(
            self,
            self.init_action,
            self.predict_action,
            self.extent_action,
            self.assemble_action,
            self.energy_action,
        )
        self.global_linear_system_system.on_subsystem(
            extent=self.extent_action,
            assemble=self.assemble_action,
        )

    def init(self) -> None:
        pass


@qd.func(requires_top_level=True)
def report_fem_bdf1_extent(
    fem: qd.template(),  # FiniteElementMethod.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    linear_system_id: qd.template(),  # int
):
    for _ in range(1):
        linear_system_data.set_subsystem_extent(linear_system_id, fem.n_fem_verts[()])


@qd.func(requires_top_level=True)
def predict_fem_bdf1(
    fem: qd.template(),  # FiniteElementMethod.Data
    sim_config: qd.template(),  # SimConfig.Data
):
    for i_vertex in range(fem.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            fem.x_tilde[i_vertex, axis] = (
                fem.x_prev[i_vertex, axis]
                + sim_config.dt[()] * fem.velocities[i_vertex, axis]
                + sim_config.dt[()] * sim_config.dt[()] * fem.gravity[i_vertex, axis]
            )


@qd.func(requires_top_level=True)
def compute_fem_bdf1_energy(
    fem: qd.template(),  # FiniteElementMethod.Data
    sim_config: qd.template(),  # SimConfig.Data
):
    for i_vertex in range(fem.n_fem_verts[()]):
        if fem.is_fixed[i_vertex] == 0:
            value = qd.f64(0.0)
            for axis in qd.static(range(3)):
                difference = fem.x[i_vertex, axis] - fem.x_tilde[i_vertex, axis]
                value = value + difference * difference
            qd.atomic_add(fem.fem_energy[()], 0.5 * fem.masses[i_vertex] * value)


@qd.func(requires_top_level=True)
def assemble_fem_bdf1(
    fem: qd.template(),  # FiniteElementMethod.Data
    sim_config: qd.template(),  # SimConfig.Data
    global_linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    linear_system_id: qd.template(),  # int
):
    for i_vertex in range(fem.n_fem_verts[()]):
        triplet_offset = global_linear_system_data.subsystem_offset(linear_system_id)
        if global_linear_system_data.matrix.triplet_overflow[()] == 0:
            block = qd.Matrix.identity(qd.f64, 3)
            if fem.is_fixed[i_vertex] == 0:
                mass = fem.masses[i_vertex]
                block = mass * block
                for axis in qd.static(range(3)):
                    gradient = mass * (fem.x[i_vertex, axis] - fem.x_tilde[i_vertex, axis])
                    global_linear_system_data.write_rhs(fem.dof_offset[()] + i_vertex * 3 + axis, gradient)
            global_linear_system_data.matrix.write_triplet(
                triplet_offset + i_vertex,
                fem.dof_offset[()] // 3 + i_vertex,
                fem.dof_offset[()] // 3 + i_vertex,
                block,
            )
