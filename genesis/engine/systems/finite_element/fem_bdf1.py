from __future__ import annotations

import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from ..sim_system import SimSystem
from .finite_element_method import FiniteElementMethod


@qd.data_oriented
class FEMBDF1(SimSystem):
    """Incremental-potential inertia for FEM vertices."""

    def do_build(self) -> None:
        fem = self.require(FiniteElementMethod)
        linear_system = self.require(GlobalLinearSystem)
        self.extent_slot = linear_system.register_extent_slot()
        fem.set_kinetic(self)

    def wire_data(self, n_fem_verts: int) -> None:
        self.n_fem_verts_host = n_fem_verts

    def triplet_count(self) -> int:
        return self.n_fem_verts_host

    @qd.func(requires_top_level=True)
    def report_extent(self, fem: qd.template(), linear_system: qd.template()):
        for _ in range(1):
            linear_system.extent_slots[self.extent_slot] = fem.n_fem_verts[()]

    @qd.func(requires_top_level=True)
    def predict(self, fem: qd.template(), sim_config: qd.template()):
        for i_vertex in range(fem.n_fem_verts[()]):
            for axis in qd.static(range(3)):
                fem.x_tilde[i_vertex, axis] = (
                    fem.x_prev[i_vertex, axis]
                    + sim_config.dt[()] * fem.velocities[i_vertex, axis]
                    + sim_config.dt[()] * sim_config.dt[()] * fem.gravity[i_vertex, axis]
                )

    @qd.func(requires_top_level=True)
    def energy(self, fem: qd.template(), sim_config: qd.template(), energy: qd.template()):
        for i_vertex in range(fem.n_fem_verts[()]):
            if fem.is_fixed[i_vertex] == 0:
                value = qd.f64(0.0)
                for axis in qd.static(range(3)):
                    difference = fem.x[i_vertex, axis] - fem.x_tilde[i_vertex, axis]
                    value = value + difference * difference
                qd.atomic_add(energy[()], 0.5 * fem.masses[i_vertex] * value)

    @qd.func(requires_top_level=True)
    def assemble(
        self,
        fem: qd.template(),
        sim_config: qd.template(),
        global_linear_system: qd.template(),
    ):
        triplet_offset = global_linear_system.extent_offsets[self.extent_slot]
        for i_vertex in range(fem.n_fem_verts[()]):
            if global_linear_system.triplet_overflow[()] == 0:
                block = qd.Matrix.identity(qd.f64, 3)
                if fem.is_fixed[i_vertex] == 0:
                    mass = fem.masses[i_vertex]
                    block = mass * block
                    for axis in qd.static(range(3)):
                        gradient = mass * (fem.x[i_vertex, axis] - fem.x_tilde[i_vertex, axis])
                        global_linear_system.b_rhs[fem.dof_offset[()] + i_vertex * 3 + axis] = gradient
                global_linear_system.set_sym(
                    triplet_offset + i_vertex,
                    fem.dof_offset[()] // 3 + i_vertex,
                    fem.dof_offset[()] // 3 + i_vertex,
                    block,
                )
