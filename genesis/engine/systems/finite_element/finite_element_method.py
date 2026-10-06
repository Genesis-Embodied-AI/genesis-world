from __future__ import annotations

import numpy as np
import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from ..sim_config import SimConfig
from ..sim_system import SimSystem
from .finite_element import FiniteElement


@qd.data_oriented
class FiniteElementMethod(SimSystem):
    """Own FEM state and compose kinetic and constitutive systems."""

    def __init__(self) -> None:
        super().__init__()
        self._kinetic = None
        self._strain_limit_baraff_witkin_shell_2d = None
        self._quadratic_bending = None
        self._is_wired = False

    def build(self) -> None:
        from ..global_body_manager import GlobalBodyManager
        from ..global_vertex_manager import GlobalVertexManager

        self.global_body_manager = self.require(GlobalBodyManager)
        self.global_vertex_manager = self.require(GlobalVertexManager)

    def set_kinetic(self, kinetic) -> None:
        if self._kinetic is not None:
            raise RuntimeError("FiniteElementMethod already has a kinetic system")
        self._kinetic = kinetic

    def add_constitution(self, constitution) -> None:
        from .quadratic_bending import QuadraticBending
        from .strain_limit_baraff_witkin_shell_2d import StrainLimitBaraffWitkinShell2D

        if isinstance(constitution, StrainLimitBaraffWitkinShell2D):
            if self._strain_limit_baraff_witkin_shell_2d is not None:
                raise RuntimeError("StrainLimitBaraffWitkinShell2D is already registered")
            self._strain_limit_baraff_witkin_shell_2d = constitution
        elif isinstance(constitution, QuadraticBending):
            if self._quadratic_bending is not None:
                raise RuntimeError("QuadraticBending is already registered")
            self._quadratic_bending = constitution
        else:
            raise TypeError(f"Unsupported FEMConstitution {type(constitution).__name__}")

    def wire_data(self, finite_element: FiniteElement) -> None:
        if self._is_wired:
            raise RuntimeError("FiniteElementMethod data is already wired")

        self.vert_capacity_host = max(finite_element.n_verts, 1)
        self.tri_capacity_host = max(finite_element.n_tris, 1)

        self.n_fem_verts = qd.ndarray(qd.i32, shape=())
        self.n_tris = qd.ndarray(qd.i32, shape=())
        self.n_bodies = qd.ndarray(qd.i32, shape=())
        self.dof_offset = qd.ndarray(qd.i32, shape=())
        self.global_vert_offset = qd.ndarray(qd.i32, shape=())
        self.global_body_offset = qd.ndarray(qd.i32, shape=())

        self.x = qd.ndarray(qd.f64, shape=(self.vert_capacity_host, 3))
        self.x_prev = qd.ndarray(qd.f64, shape=(self.vert_capacity_host, 3))
        self.velocities = qd.ndarray(qd.f64, shape=(self.vert_capacity_host, 3))
        self.masses = qd.ndarray(qd.f64, shape=(self.vert_capacity_host,))
        self.is_fixed = qd.ndarray(qd.i32, shape=(self.vert_capacity_host,))
        self.gravity = qd.ndarray(qd.f64, shape=(self.vert_capacity_host, 3))
        self.thicknesses = qd.ndarray(qd.f64, shape=(self.vert_capacity_host,))
        self.body_id = qd.ndarray(qd.i32, shape=(self.vert_capacity_host,))
        self.body_vertex_offsets = qd.ndarray(qd.i32, shape=(finite_element.n_bodies + 1,))
        self.self_collision = qd.ndarray(qd.i32, shape=(max(finite_element.n_bodies, 1),))

        self.tri_indices = qd.ndarray(qd.i32, shape=(self.tri_capacity_host, 3))
        self.Dm_inv_2d = qd.ndarray(qd.f64, shape=(self.tri_capacity_host, 4))
        self.rest_areas = qd.ndarray(qd.f64, shape=(self.tri_capacity_host,))

        self.x_tilde = qd.ndarray(qd.f64, shape=(self.vert_capacity_host, 3))
        self.x_temp = qd.ndarray(qd.f64, shape=(self.vert_capacity_host, 3))
        self.dx = qd.ndarray(qd.f64, shape=(self.vert_capacity_host, 3))
        self.fem_energy = qd.ndarray(qd.f64, shape=())
        self._bridge_vertex = qd.ndarray(qd.i32, shape=(self.vert_capacity_host,))
        self._bridge_environment = qd.ndarray(qd.i32, shape=(self.vert_capacity_host,))
        self._scene_elements_v = finite_element.scene_elements_v
        self._scene_frame = finite_element.scene_frame

        self.n_fem_verts.from_numpy(np.array(finite_element.n_verts, dtype=np.int32))
        self.n_tris.from_numpy(np.array(finite_element.n_tris, dtype=np.int32))
        self.n_bodies.from_numpy(np.array(finite_element.n_bodies, dtype=np.int32))
        self.dof_offset.from_numpy(np.array(0, dtype=np.int32))
        self.global_vert_offset.from_numpy(np.array(0, dtype=np.int32))
        self.global_body_offset.from_numpy(np.array(0, dtype=np.int32))
        self.x.from_numpy(finite_element.positions)
        self.x_prev.from_numpy(finite_element.positions)
        self.velocities.from_numpy(finite_element.velocities)
        self.masses.from_numpy(finite_element.masses)
        self.is_fixed.from_numpy(finite_element.is_fixed)
        self.gravity.from_numpy(finite_element.gravity)
        self.thicknesses.from_numpy(finite_element.thicknesses)
        self.body_id.from_numpy(finite_element.body_ids)
        self.body_vertex_offsets.from_numpy(finite_element.body_vertex_offsets)
        self.self_collision.from_numpy(finite_element.self_collision)
        self.tri_indices.from_numpy(finite_element.tri_indices)
        self.Dm_inv_2d.from_numpy(finite_element.Dm_inv_2d.reshape(finite_element.n_tris, 4))
        self.rest_areas.from_numpy(finite_element.rest_areas)
        self.x_tilde.from_numpy(finite_element.positions)
        self.x_temp.from_numpy(finite_element.positions)
        self.dx.from_numpy(np.zeros((self.vert_capacity_host, 3), dtype=np.float64))
        self._bridge_vertex.from_numpy(finite_element.bridge_vertex)
        self._bridge_environment.from_numpy(finite_element.bridge_environment)
        self._is_wired = True

    def report_global_vertex_extent(self) -> int:
        return self.vert_capacity_host

    def receive_global_vertex_range(self, offset: int, count: int) -> None:
        if count != self.vert_capacity_host:
            raise ValueError("FiniteElementMethod global vertex range has the wrong extent")
        self.global_vert_offset.from_numpy(np.array(offset, dtype=np.int32))

    def report_global_body_extent(self) -> int:
        return self.body_vertex_offsets.shape[0] - 1

    def receive_global_body_range(self, offset: int, count: int) -> None:
        if count != self.report_global_body_extent():
            raise ValueError("FiniteElementMethod global body range has the wrong extent")
        self.global_body_offset.from_numpy(np.array(offset, dtype=np.int32))

    def init(self, dof_offset: int) -> None:
        if not self._is_wired:
            raise RuntimeError("FiniteElementMethod.wire_data() must run before init()")
        if self._kinetic is None:
            raise RuntimeError("FiniteElementMethod requires FEMBDF1")

        self.dof_offset.from_numpy(np.array(dof_offset, dtype=np.int32))
        self.kinetic = self._kinetic
        self.strain_limit_baraff_witkin_shell_2d = self._strain_limit_baraff_witkin_shell_2d
        self.quadratic_bending = self._quadratic_bending
        self.has_strain_limit_baraff_witkin_shell_2d = self.strain_limit_baraff_witkin_shell_2d is not None
        self.has_quadratic_bending = self.quadratic_bending is not None
        del (
            self._kinetic,
            self._strain_limit_baraff_witkin_shell_2d,
            self._quadratic_bending,
        )

    def n_elastic_triplets(self) -> int:
        count = self.kinetic.triplet_count()
        if self.has_strain_limit_baraff_witkin_shell_2d:
            count += self.strain_limit_baraff_witkin_shell_2d.triplet_count()
        if self.has_quadratic_bending:
            count += self.quadratic_bending.triplet_count()
        return count

    @qd.func(requires_top_level=True)
    def predict(self, sim_config: qd.template()):
        self.kinetic.predict(self, sim_config)

    @qd.func(requires_top_level=True)
    def initialize_global_vertices(self, vertex: qd.template()):
        for i_vertex in range(self.n_fem_verts[()]):
            global_vertex = self.global_vert_offset[()] + i_vertex
            vertex.body_id[global_vertex] = self.global_body_offset[()] + self.body_id[i_vertex]
            for axis in qd.static(range(3)):
                value = self.x[i_vertex, axis]
                vertex.positions[global_vertex, axis] = value
                vertex.safe_positions[global_vertex, axis] = value
                vertex.trajectory_end_positions[global_vertex, axis] = value
                vertex.x_bar[global_vertex, axis] = value

    @qd.func(requires_top_level=True)
    def forward_global_vertices(self, vertex: qd.template()):
        for i_vertex in range(self.n_fem_verts[()]):
            global_vertex = self.global_vert_offset[()] + i_vertex
            for axis in qd.static(range(3)):
                vertex.positions[global_vertex, axis] = self.x[i_vertex, axis]

    @qd.func(requires_top_level=True)
    def publish_trajectory_end_positions(self, vertex: qd.template()):
        for i_vertex in range(self.n_fem_verts[()]):
            global_vertex = self.global_vert_offset[()] + i_vertex
            for axis in qd.static(range(3)):
                vertex.trajectory_end_positions[global_vertex, axis] = (
                    self.x_temp[i_vertex, axis] + self.dx[i_vertex, axis]
                )

    @qd.func(requires_top_level=True)
    def report_extent(self, global_linear_system: qd.template()):
        self.kinetic.report_extent(self, global_linear_system)
        if qd.static(self.has_strain_limit_baraff_witkin_shell_2d):
            self.strain_limit_baraff_witkin_shell_2d.report_extent(self, global_linear_system)
        if qd.static(self.has_quadratic_bending):
            self.quadratic_bending.report_extent(self, global_linear_system)

    @qd.func(requires_top_level=True)
    def assemble(self, sim_config: qd.template(), global_linear_system: qd.template()):
        self.kinetic.assemble(self, sim_config, global_linear_system)
        if qd.static(self.has_strain_limit_baraff_witkin_shell_2d):
            self.strain_limit_baraff_witkin_shell_2d.assemble(
                self,
                sim_config,
                global_linear_system,
            )
        if qd.static(self.has_quadratic_bending):
            self.quadratic_bending.assemble(self, sim_config, global_linear_system)

    @qd.func(requires_top_level=True)
    def energy(self, sim_config: qd.template()):
        for _ in range(1):
            self.fem_energy[()] = qd.f64(0.0)
        self.kinetic.energy(self, sim_config, self.fem_energy)
        if qd.static(self.has_strain_limit_baraff_witkin_shell_2d):
            self.strain_limit_baraff_witkin_shell_2d.energy(
                self,
                sim_config,
                self.fem_energy,
            )
        if qd.static(self.has_quadratic_bending):
            self.quadratic_bending.energy(self, sim_config, self.fem_energy)

    @qd.func(requires_top_level=True)
    def negate_dx(self, global_linear_system: qd.template()):
        for i_vert in range(self.n_fem_verts[()]):
            for axis in qd.static(range(3)):
                value = qd.f64(0.0)
                if self.is_fixed[i_vert] == 0:
                    value = -global_linear_system.x_sol[self.dof_offset[()] + i_vert * 3 + axis]
                self.dx[i_vert, axis] = value

    @qd.func(requires_top_level=True)
    def contribute_newton_max_disp(self, max_disp: qd.template()):
        for i_vert in range(self.n_fem_verts[()]):
            for axis in qd.static(range(3)):
                qd.atomic_max(max_disp[()], qd.abs(self.dx[i_vert, axis]))

    @qd.func(requires_top_level=True)
    def record_start_point(self):
        for i_vert in range(self.n_fem_verts[()]):
            for axis in qd.static(range(3)):
                self.x_temp[i_vert, axis] = self.x[i_vert, axis]

    @qd.func(requires_top_level=True)
    def step_forward(self, alpha):
        for i_vert in range(self.n_fem_verts[()]):
            if self.is_fixed[i_vert] == 0:
                for axis in qd.static(range(3)):
                    self.x[i_vert, axis] = self.x_temp[i_vert, axis] + alpha * self.dx[i_vert, axis]

    @qd.func(requires_top_level=True)
    def update_velocity(self, sim_config: qd.template()):
        for i_vert in range(self.n_fem_verts[()]):
            for axis in qd.static(range(3)):
                if self.is_fixed[i_vert] != 0:
                    self.velocities[i_vert, axis] = qd.f64(0.0)
                else:
                    self.velocities[i_vert, axis] = (self.x[i_vert, axis] - self.x_prev[i_vert, axis]) / sim_config.dt[
                        ()
                    ]

    @qd.func(requires_top_level=True)
    def copy_x_prev(self):
        for i_vert in range(self.n_fem_verts[()]):
            for axis in qd.static(range(3)):
                self.x_prev[i_vert, axis] = self.x[i_vert, axis]

    @qd.kernel
    def _forward_scene_vertices(self):
        for i_vert in range(self.n_fem_verts[()]):
            scene_vert = self._bridge_vertex[i_vert]
            environment = self._bridge_environment[i_vert]
            for axis in qd.static(range(3)):
                self._scene_elements_v[self._scene_frame, scene_vert, environment].pos[axis] = self.x[i_vert, axis]
                self._scene_elements_v[self._scene_frame, scene_vert, environment].vel[axis] = self.velocities[
                    i_vert, axis
                ]
