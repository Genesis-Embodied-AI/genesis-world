from __future__ import annotations

import numpy as np
import quadrants as qd

from ..sim_system import ActionInvocation, SimAction, SimData, SimSystem
from .finite_element import FiniteElement


class FiniteElementMethod(SimSystem):
    """Own FEM state and compose kinetic and constitutive systems."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible FEM state and host scene bridge."""

        vert_capacity_host: int
        tri_capacity_host: int
        n_bodies_host: int
        n_fem_verts: qd.Ndarray
        n_tris: qd.Ndarray
        n_bodies: qd.Ndarray
        dof_offset: qd.Ndarray
        global_vert_offset: qd.Ndarray
        global_body_offset: qd.Ndarray
        x: qd.Ndarray
        x_prev: qd.Ndarray
        velocities: qd.Ndarray
        masses: qd.Ndarray
        is_fixed: qd.Ndarray
        gravity: qd.Ndarray
        thicknesses: qd.Ndarray
        body_id: qd.Ndarray
        body_vertex_offsets: qd.Ndarray
        self_collision: qd.Ndarray
        tri_indices: qd.Ndarray
        Dm_inv_2d: qd.Ndarray
        rest_areas: qd.Ndarray
        x_tilde: qd.Ndarray
        x_temp: qd.Ndarray
        dx: qd.Ndarray
        fem_energy: qd.Ndarray
        bridge_vertex: qd.Ndarray
        bridge_environment: qd.Ndarray
        scene_elements_v: qd.Field
        scene_vertex_constraints: qd.Field
        scene_frame: int

    def __init__(self, data: Data) -> None:
        super().__init__()
        self.data = data
        self.kinetic_system = None
        self.strain_limit_baraff_witkin_shell_2d_system = None
        self.quadratic_bending_system = None
        self.predict_actions = self.create_action_collection()
        self.extent_actions = self.create_action_collection()
        self.assemble_actions = self.create_action_collection()
        self.energy_actions = self.create_action_collection()

    def build(self) -> None:
        from ..global_body_manager import GlobalBodyManager
        from ..global_vertex_manager import GlobalVertexManager

        self.global_body_system = self.require(GlobalBodyManager)
        self.global_vertex_system = self.require(GlobalVertexManager)

    def on_kinetic(
        self,
        kinetic_system: SimSystem,
        predict_action: SimAction,
        extent_action: SimAction,
        assemble_action: SimAction,
        energy_action: SimAction,
    ) -> None:
        if self.kinetic_system is not None:
            raise RuntimeError("FiniteElementMethod already has a kinetic system")
        self.kinetic_system = kinetic_system
        self.predict_actions.register(predict_action)
        self.extent_actions.register(extent_action)
        self.assemble_actions.register(assemble_action)
        self.energy_actions.register(energy_action)

    def on_constitution(
        self,
        constitution_system: SimSystem,
        extent_action: SimAction,
        assemble_action: SimAction,
        energy_action: SimAction,
    ) -> None:
        from .quadratic_bending import QuadraticBending
        from .strain_limit_baraff_witkin_shell_2d import StrainLimitBaraffWitkinShell2D

        if isinstance(constitution_system, StrainLimitBaraffWitkinShell2D):
            if self.strain_limit_baraff_witkin_shell_2d_system is not None:
                raise RuntimeError("StrainLimitBaraffWitkinShell2D is already registered")
            self.strain_limit_baraff_witkin_shell_2d_system = constitution_system
        elif isinstance(constitution_system, QuadraticBending):
            if self.quadratic_bending_system is not None:
                raise RuntimeError("QuadraticBending is already registered")
            self.quadratic_bending_system = constitution_system
        else:
            raise TypeError(f"Unsupported FEMConstitution {type(constitution_system).__name__}")
        self.extent_actions.register(extent_action)
        self.assemble_actions.register(assemble_action)
        self.energy_actions.register(energy_action)

    def report_global_vertex_extent(self) -> int:
        return self.data.vert_capacity_host

    def receive_global_vertex_range(self, offset: int, count: int) -> None:
        if count != self.data.vert_capacity_host:
            raise ValueError("FiniteElementMethod global vertex range has the wrong extent")
        self.data.global_vert_offset.from_numpy(np.array(offset, dtype=np.int32))

    def report_global_body_extent(self) -> int:
        return self.data.n_bodies_host

    def receive_global_body_range(self, offset: int, count: int) -> None:
        if count != self.report_global_body_extent():
            raise ValueError("FiniteElementMethod global body range has the wrong extent")
        self.data.global_body_offset.from_numpy(np.array(offset, dtype=np.int32))

    def init(self, dof_offset: int) -> None:
        if self.kinetic_system is None:
            raise RuntimeError("FiniteElementMethod requires FEMBDF1")
        self.data.dof_offset.from_numpy(np.array(dof_offset, dtype=np.int32))

    def n_elastic_triplets(self) -> int:
        if self.kinetic_system is None:
            raise RuntimeError("FiniteElementMethod actions have not been built")
        count = self.kinetic_system.triplet_count()
        if self.strain_limit_baraff_witkin_shell_2d_system is not None:
            count += self.strain_limit_baraff_witkin_shell_2d_system.triplet_count()
        if self.quadratic_bending_system is not None:
            count += self.quadratic_bending_system.triplet_count()
        return count

    def resolve_actions(
        self,
    ) -> tuple[
        tuple[ActionInvocation, ...],
        tuple[ActionInvocation, ...],
        tuple[ActionInvocation, ...],
        tuple[ActionInvocation, ...],
    ]:
        predict = self.predict_actions.actions
        extent = self.extent_actions.actions
        assemble = self.assemble_actions.actions
        energy = self.energy_actions.actions
        if len(predict) != 1:
            raise RuntimeError(f"FiniteElementMethod requires exactly one kinetic predictor, got {len(predict)}")
        if not extent or len(extent) != len(assemble) or len(extent) != len(energy):
            raise RuntimeError("FiniteElementMethod requires complete extent, assembly, and energy action protocols")
        return tuple(tuple(action.invocation for action in actions) for actions in (predict, extent, assemble, energy))

    @property
    def x(self):
        return self.data.x

    @property
    def fem_energy(self):
        return self.data.fem_energy

    @property
    def vert_capacity_host(self) -> int:
        return self.data.vert_capacity_host

    @property
    def quadratic_bending(self):
        return self.quadratic_bending_system


def get_finite_element_method_data(finite_element: FiniteElement) -> FiniteElementMethod.Data:
    """Construct complete FEM runtime data before graph action registration."""

    vert_capacity = max(finite_element.n_verts, 1)
    tri_capacity = max(finite_element.n_tris, 1)
    body_capacity = max(finite_element.n_bodies, 1)

    def array(dtype, shape, values):
        result = qd.ndarray(dtype, shape=shape)
        result.from_numpy(values)
        return result

    def padded(values, shape, dtype):
        result = np.zeros(shape, dtype=dtype)
        values = np.asarray(values, dtype=dtype)
        if values.size:
            result[: len(values)] = values
        return result

    positions = padded(finite_element.positions, (vert_capacity, 3), np.float64)
    velocities = padded(finite_element.velocities, (vert_capacity, 3), np.float64)
    masses = padded(finite_element.masses, (vert_capacity,), np.float64)
    fixed = padded(finite_element.is_fixed, (vert_capacity,), np.int32)
    gravity = padded(finite_element.gravity, (vert_capacity, 3), np.float64)
    thicknesses = padded(finite_element.thicknesses, (vert_capacity,), np.float64)
    body_ids = padded(finite_element.body_ids, (vert_capacity,), np.int32)
    self_collision = padded(finite_element.self_collision, (body_capacity,), np.int32)
    triangles = padded(finite_element.tri_indices, (tri_capacity, 3), np.int32)
    dm_inverse = padded(
        finite_element.Dm_inv_2d.reshape(finite_element.n_tris, 4),
        (tri_capacity, 4),
        np.float64,
    )
    rest_areas = padded(finite_element.rest_areas, (tri_capacity,), np.float64)
    bridge_vertex = padded(finite_element.bridge_vertex, (vert_capacity,), np.int32)
    bridge_environment = padded(finite_element.bridge_environment, (vert_capacity,), np.int32)

    data = FiniteElementMethod.Data()
    data.vert_capacity_host = vert_capacity
    data.tri_capacity_host = tri_capacity
    data.n_bodies_host = finite_element.n_bodies
    data.n_fem_verts = array(qd.i32, (), np.array(finite_element.n_verts, dtype=np.int32))
    data.n_tris = array(qd.i32, (), np.array(finite_element.n_tris, dtype=np.int32))
    data.n_bodies = array(qd.i32, (), np.array(finite_element.n_bodies, dtype=np.int32))
    data.dof_offset = array(qd.i32, (), np.array(0, dtype=np.int32))
    data.global_vert_offset = array(qd.i32, (), np.array(0, dtype=np.int32))
    data.global_body_offset = array(qd.i32, (), np.array(0, dtype=np.int32))
    data.x = array(qd.f64, (vert_capacity, 3), positions)
    data.x_prev = array(qd.f64, (vert_capacity, 3), positions)
    data.velocities = array(qd.f64, (vert_capacity, 3), velocities)
    data.masses = array(qd.f64, (vert_capacity,), masses)
    data.is_fixed = array(qd.i32, (vert_capacity,), fixed)
    data.gravity = array(qd.f64, (vert_capacity, 3), gravity)
    data.thicknesses = array(qd.f64, (vert_capacity,), thicknesses)
    data.body_id = array(qd.i32, (vert_capacity,), body_ids)
    data.body_vertex_offsets = array(
        qd.i32,
        (finite_element.n_bodies + 1,),
        np.asarray(finite_element.body_vertex_offsets, dtype=np.int32),
    )
    data.self_collision = array(qd.i32, (body_capacity,), self_collision)
    data.tri_indices = array(qd.i32, (tri_capacity, 3), triangles)
    data.Dm_inv_2d = array(qd.f64, (tri_capacity, 4), dm_inverse)
    data.rest_areas = array(qd.f64, (tri_capacity,), rest_areas)
    data.x_tilde = array(qd.f64, (vert_capacity, 3), positions)
    data.x_temp = array(qd.f64, (vert_capacity, 3), positions)
    data.dx = array(qd.f64, (vert_capacity, 3), np.zeros((vert_capacity, 3), dtype=np.float64))
    data.fem_energy = array(qd.f64, (), np.array(0.0, dtype=np.float64))
    data.bridge_vertex = array(qd.i32, (vert_capacity,), bridge_vertex)
    data.bridge_environment = array(qd.i32, (vert_capacity,), bridge_environment)
    data.scene_elements_v = finite_element.scene_elements_v
    data.scene_vertex_constraints = finite_element.scene_vertex_constraints
    data.scene_frame = finite_element.scene_frame
    return data


@qd.func(requires_top_level=True)
def sync_fem_from_scene(data: qd.template(), vertex: qd.template()):
    for i_vertex in range(data.n_fem_verts[()]):
        scene_vertex = data.bridge_vertex[i_vertex]
        environment = data.bridge_environment[i_vertex]
        constraint = data.scene_vertex_constraints[scene_vertex, environment]
        is_fixed = constraint.is_constrained and not constraint.is_soft_constraint and constraint.link_idx < 0
        data.is_fixed[i_vertex] = qd.cast(is_fixed, qd.i32)
        global_vertex = data.global_vert_offset[()] + i_vertex
        vertex.is_fixed[global_vertex] = qd.cast(is_fixed, qd.i32)
        for axis in qd.static(range(3)):
            position = data.scene_elements_v[data.scene_frame, scene_vertex, environment].pos[axis]
            velocity = data.scene_elements_v[data.scene_frame, scene_vertex, environment].vel[axis]
            if is_fixed:
                position = constraint.target_pos[axis]
                velocity = qd.f64(0.0)
            data.x[i_vertex, axis] = position
            data.x_prev[i_vertex, axis] = position
            data.x_tilde[i_vertex, axis] = position
            data.x_temp[i_vertex, axis] = position
            data.velocities[i_vertex, axis] = velocity
            data.dx[i_vertex, axis] = qd.f64(0.0)
            vertex.positions[global_vertex, axis] = position
            vertex.safe_positions[global_vertex, axis] = position
            vertex.trajectory_end_positions[global_vertex, axis] = position
            vertex.x_bar[global_vertex, axis] = position


@qd.func(requires_top_level=True)
def initialize_fem_global_vertices(data: qd.template(), vertex: qd.template()):
    for i_vertex in range(data.n_fem_verts[()]):
        global_vertex = data.global_vert_offset[()] + i_vertex
        vertex.body_id[global_vertex] = data.global_body_offset[()] + data.body_id[i_vertex]
        for axis in qd.static(range(3)):
            value = data.x[i_vertex, axis]
            vertex.positions[global_vertex, axis] = value
            vertex.safe_positions[global_vertex, axis] = value
            vertex.trajectory_end_positions[global_vertex, axis] = value
            vertex.x_bar[global_vertex, axis] = value


@qd.func(requires_top_level=True)
def forward_fem_global_vertices(data: qd.template(), vertex: qd.template()):
    for i_vertex in range(data.n_fem_verts[()]):
        global_vertex = data.global_vert_offset[()] + i_vertex
        for axis in qd.static(range(3)):
            vertex.positions[global_vertex, axis] = data.x[i_vertex, axis]


@qd.func(requires_top_level=True)
def publish_fem_trajectory_end_positions(data: qd.template(), vertex: qd.template()):
    for i_vertex in range(data.n_fem_verts[()]):
        global_vertex = data.global_vert_offset[()] + i_vertex
        for axis in qd.static(range(3)):
            vertex.trajectory_end_positions[global_vertex, axis] = data.x_temp[i_vertex, axis] + data.dx[i_vertex, axis]


@qd.func(requires_top_level=True)
def reset_fem_energy(data: qd.template()):
    for _ in range(1):
        data.fem_energy[()] = qd.f64(0.0)


@qd.func(requires_top_level=True)
def negate_fem_dx(data: qd.template(), global_linear_system_data: qd.template()):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            value = qd.f64(0.0)
            if data.is_fixed[i_vert] == 0:
                value = -global_linear_system_data.x_sol[data.dof_offset[()] + i_vert * 3 + axis]
            data.dx[i_vert, axis] = value


@qd.func(requires_top_level=True)
def contribute_fem_newton_max_disp(data: qd.template(), max_disp: qd.template()):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            qd.atomic_max(max_disp[()], qd.abs(data.dx[i_vert, axis]))


@qd.func(requires_top_level=True)
def record_fem_start_point(data: qd.template()):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            data.x_temp[i_vert, axis] = data.x[i_vert, axis]


@qd.func(requires_top_level=True)
def step_fem_forward(data: qd.template(), alpha):
    for i_vert in range(data.n_fem_verts[()]):
        if data.is_fixed[i_vert] == 0:
            for axis in qd.static(range(3)):
                data.x[i_vert, axis] = data.x_temp[i_vert, axis] + alpha * data.dx[i_vert, axis]


@qd.func(requires_top_level=True)
def update_fem_velocity(data: qd.template(), sim_config: qd.template()):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            if data.is_fixed[i_vert] != 0:
                data.velocities[i_vert, axis] = qd.f64(0.0)
            else:
                data.velocities[i_vert, axis] = (data.x[i_vert, axis] - data.x_prev[i_vert, axis]) / sim_config.dt[()]


@qd.func(requires_top_level=True)
def copy_fem_x_prev(data: qd.template()):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            data.x_prev[i_vert, axis] = data.x[i_vert, axis]


@qd.kernel
def forward_fem_scene_vertices(data: qd.template()):
    for i_vert in range(data.n_fem_verts[()]):
        scene_vert = data.bridge_vertex[i_vert]
        environment = data.bridge_environment[i_vert]
        for axis in qd.static(range(3)):
            data.scene_elements_v[data.scene_frame, scene_vert, environment].pos[axis] = data.x[i_vert, axis]
            data.scene_elements_v[data.scene_frame, scene_vert, environment].vel[axis] = data.velocities[i_vert, axis]
