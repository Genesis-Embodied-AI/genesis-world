from __future__ import annotations

import numpy as np
import quadrants as qd

from ..sim_system import ActionKind, SimAction, SimData, SimSystem, validate_action_protocol
from .finite_element import FiniteElement


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class FiniteElementMethod(SimSystem):
    """Own FEM state and compose kinetic and constitutive systems."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible FEM state and host scene bridge."""

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
        scene_frame: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._finite_element: FiniteElement | None = None
        self._global_vert_offset: int = 0
        self._global_body_offset: int = 0
        self.predict_actions = self.create_action_collection()
        self.init_actions = self.create_action_collection()
        self.extent_actions = self.create_action_collection()
        self.assemble_actions = self.create_action_collection()
        self.energy_actions = self.create_action_collection()
        self.predict_schedule = ()
        self.extent_schedule = ()
        self.assemble_schedule = ()
        self.energy_schedule = ()

    def wire_data(self, finite_element: FiniteElement) -> None:
        self._finite_element = finite_element

    def build(self) -> None:
        from ..global_body_manager import GlobalBodyManager
        from ..global_linear_system import GlobalLinearSystem
        from ..global_vertex_manager import GlobalVertexManager
        from ..sim_config import SimConfig

        self.global_body_system = self.require(GlobalBodyManager)
        self.global_linear_system = self.require(GlobalLinearSystem)
        self.global_vertex_system = self.require(GlobalVertexManager)
        self.sim_config_system = self.require(SimConfig)

    def on_kinetic(
        self,
        kinetic_system: SimSystem,
        init_action: SimAction,
        predict_action: SimAction,
        extent_action: SimAction,
        assemble_action: SimAction,
        energy_action: SimAction,
    ) -> None:
        validate_action_protocol(
            init_action,
            protocol="FiniteElementMethod.on_kinetic.init",
            expected_kind=ActionKind.HOST,
            transient_arity=0,
        )
        actions = (
            ("predict", predict_action, 1),
            ("extent", extent_action, 2),
            ("assemble", assemble_action, 3),
            ("energy", energy_action, 1),
        )
        for name, action, arity in actions:
            validate_action_protocol(
                action,
                protocol=f"FiniteElementMethod.on_kinetic.{name}",
                expected_kind=ActionKind.TOP_LEVEL_FUNC,
                transient_arity=arity,
            )
        if any(action.owner is not kinetic_system for _, action, _ in actions):
            raise TypeError("FiniteElementMethod kinetic actions must be owned by the kinetic system")
        if self.predict_actions._actions:
            raise RuntimeError("FiniteElementMethod already has a kinetic system")
        self.init_actions.register(init_action)
        self.predict_actions.register(predict_action)
        self.extent_actions.register(extent_action)
        self.assemble_actions.register(assemble_action)
        self.energy_actions.register(energy_action)

    def on_constitution(
        self,
        constitution_system: SimSystem,
        init_action: SimAction,
        extent_action: SimAction,
        assemble_action: SimAction,
        energy_action: SimAction,
    ) -> None:
        validate_action_protocol(
            init_action,
            protocol="FiniteElementMethod.on_constitution.init",
            expected_kind=ActionKind.HOST,
            transient_arity=0,
        )
        actions = (
            ("extent", extent_action, 2),
            ("assemble", assemble_action, 3),
            ("energy", energy_action, 1),
        )
        for name, action, arity in actions:
            validate_action_protocol(
                action,
                protocol=f"FiniteElementMethod.on_constitution.{name}",
                expected_kind=ActionKind.TOP_LEVEL_FUNC,
                transient_arity=arity,
            )
        if any(action.owner is not constitution_system for _, action, _ in actions):
            raise TypeError("FiniteElementMethod constitution actions must be owned by the constitution system")
        if any(action.owner is constitution_system for action in self.extent_actions._actions):
            raise RuntimeError(f"{type(constitution_system).__name__} is already registered")
        self.init_actions.register(init_action)
        self.extent_actions.register(extent_action)
        self.assemble_actions.register(assemble_action)
        self.energy_actions.register(energy_action)

    def on_preconditioner(self, init_action: SimAction) -> None:
        validate_action_protocol(
            init_action,
            protocol="FiniteElementMethod.on_preconditioner.init",
            expected_kind=ActionKind.HOST,
            transient_arity=0,
        )
        self.init_actions.register(init_action)

    def receive_global_vertex_range(self, offset: int, count: int) -> None:
        finite_element = self._finite_element
        if finite_element is None:
            raise RuntimeError("FiniteElementMethod data has not been wired")
        if count != max(finite_element.n_verts, 1):
            raise ValueError("FiniteElementMethod global vertex range has the wrong extent")
        self._global_vert_offset = int(offset)

    def receive_global_body_range(self, offset: int, count: int) -> None:
        finite_element = self._finite_element
        if finite_element is None:
            raise RuntimeError("FiniteElementMethod data has not been wired")
        if count != finite_element.n_bodies:
            raise ValueError("FiniteElementMethod global body range has the wrong extent")
        self._global_body_offset = int(offset)

    def init(self, dof_offset: int) -> None:
        if not self.predict_actions._actions:
            raise RuntimeError("FiniteElementMethod requires FEMBDF1")
        finite_element = self._finite_element
        if finite_element is None:
            raise RuntimeError("FiniteElementMethod data has not been wired")

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
        self.data.n_fem_verts = array(qd.i32, (), np.array(finite_element.n_verts, dtype=np.int32))
        self.data.n_tris = array(qd.i32, (), np.array(finite_element.n_tris, dtype=np.int32))
        self.data.n_bodies = array(qd.i32, (), np.array(finite_element.n_bodies, dtype=np.int32))
        self.data.dof_offset = array(qd.i32, (), np.array(dof_offset, dtype=np.int32))
        self.data.global_vert_offset = array(qd.i32, (), np.array(self._global_vert_offset, dtype=np.int32))
        self.data.global_body_offset = array(qd.i32, (), np.array(self._global_body_offset, dtype=np.int32))
        self.data.x = array(qd.f64, (vert_capacity, 3), positions)
        self.data.x_prev = array(qd.f64, (vert_capacity, 3), positions)
        self.data.velocities = array(
            qd.f64,
            (vert_capacity, 3),
            padded(finite_element.velocities, (vert_capacity, 3), np.float64),
        )
        self.data.masses = array(
            qd.f64,
            (vert_capacity,),
            padded(finite_element.masses, (vert_capacity,), np.float64),
        )
        self.data.is_fixed = array(
            qd.i32,
            (vert_capacity,),
            padded(finite_element.is_fixed, (vert_capacity,), np.int32),
        )
        self.data.gravity = array(
            qd.f64,
            (vert_capacity, 3),
            padded(finite_element.gravity, (vert_capacity, 3), np.float64),
        )
        self.data.thicknesses = array(
            qd.f64,
            (vert_capacity,),
            padded(finite_element.thicknesses, (vert_capacity,), np.float64),
        )
        self.data.body_id = array(
            qd.i32,
            (vert_capacity,),
            padded(finite_element.body_ids, (vert_capacity,), np.int32),
        )
        self.data.body_vertex_offsets = array(
            qd.i32,
            (finite_element.n_bodies + 1,),
            np.asarray(finite_element.body_vertex_offsets, dtype=np.int32),
        )
        self.data.self_collision = array(
            qd.i32,
            (body_capacity,),
            padded(finite_element.self_collision, (body_capacity,), np.int32),
        )
        self.data.tri_indices = array(
            qd.i32,
            (tri_capacity, 3),
            padded(finite_element.tri_indices, (tri_capacity, 3), np.int32),
        )
        self.data.Dm_inv_2d = array(
            qd.f64,
            (tri_capacity, 4),
            padded(
                finite_element.Dm_inv_2d.reshape(finite_element.n_tris, 4),
                (tri_capacity, 4),
                np.float64,
            ),
        )
        self.data.rest_areas = array(
            qd.f64,
            (tri_capacity,),
            padded(finite_element.rest_areas, (tri_capacity,), np.float64),
        )
        self.data.x_tilde = array(qd.f64, (vert_capacity, 3), positions)
        self.data.x_temp = array(qd.f64, (vert_capacity, 3), positions)
        self.data.dx = array(qd.f64, (vert_capacity, 3), np.zeros((vert_capacity, 3), dtype=np.float64))
        self.data.fem_energy = array(qd.f64, (), np.array(0.0, dtype=np.float64))
        self.data.bridge_vertex = array(
            qd.i32,
            (vert_capacity,),
            padded(finite_element.bridge_vertex, (vert_capacity,), np.int32),
        )
        self.data.bridge_environment = array(
            qd.i32,
            (vert_capacity,),
            padded(finite_element.bridge_environment, (vert_capacity,), np.int32),
        )
        self.data.scene_elements_v = finite_element.scene_elements_v
        self.data.scene_vertex_constraints = finite_element.scene_vertex_constraints
        self.data.scene_frame = array(qd.i32, (), np.array(finite_element.scene_frame, dtype=np.int32))
        for action in self.init_actions.actions:
            action.invoke()
        self.resolve_actions()
        self._finite_element = None
        self._global_vert_offset = 0
        self._global_body_offset = 0

    def resolve_actions(
        self,
    ) -> tuple[
        tuple[SimAction, ...],
        tuple[SimAction, ...],
        tuple[SimAction, ...],
        tuple[SimAction, ...],
    ]:
        predict = self.predict_actions.actions
        extent = self.extent_actions.actions
        assemble = self.assemble_actions.actions
        energy = self.energy_actions.actions
        if len(predict) != 1:
            raise RuntimeError(f"FiniteElementMethod requires exactly one kinetic predictor, got {len(predict)}")
        if not extent or len(extent) != len(assemble) or len(extent) != len(energy):
            raise RuntimeError("FiniteElementMethod requires complete extent, assembly, and energy action protocols")
        (
            self.predict_schedule,
            self.extent_schedule,
            self.assemble_schedule,
            self.energy_schedule,
        ) = (predict, extent, assemble, energy)
        return predict, extent, assemble, energy

    @qd.func(requires_top_level=True)
    def on_initialize_global_vertices(self):
        initialize_fem_global_vertices(self.data, self.global_vertex_system.data)

    @qd.func(requires_top_level=True)
    def on_sync_from_scene(self):
        sync_fem_from_scene(self.data, self.global_vertex_system.data)

    @qd.func(requires_top_level=True)
    def on_forward_global_vertices(self):
        forward_fem_global_vertices(self.data, self.global_vertex_system.data)

    @qd.func(requires_top_level=True)
    def on_predict(self):
        for action in qd.static(self.predict_schedule):
            action.invoke((self.sim_config_system.data,))

    @qd.func(requires_top_level=True)
    def on_extent(self):
        for linear_system_id, action in qd.static(enumerate(self.extent_schedule)):
            action.invoke((self.global_linear_system.data, linear_system_id))

    @qd.func(requires_top_level=True)
    def on_assemble(self):
        for linear_system_id, action in qd.static(enumerate(self.assemble_schedule)):
            action.invoke((self.sim_config_system.data, self.global_linear_system.data, linear_system_id))

    @qd.func(requires_top_level=True)
    def on_energy(self):
        for action in qd.static(self.energy_schedule):
            action.invoke((self.sim_config_system.data,))

    @qd.func(requires_top_level=True)
    def on_negate_direction(self):
        negate_fem_dx(self.data, self.global_linear_system.data)

    @qd.func(requires_top_level=True)
    def on_contribute_newton_max_displacement(
        self,
        max_displacement: qd.template(),  # qd.Ndarray
    ):
        contribute_fem_newton_max_disp(self.data, max_displacement)

    @qd.func(requires_top_level=True)
    def on_record_start_point(self):
        record_fem_start_point(self.data)

    @qd.func(requires_top_level=True)
    def on_reset_energy(self):
        reset_fem_energy(self.data)

    @qd.func(requires_top_level=True)
    def on_step_forward(self, alpha):
        step_fem_forward(self.data, alpha)

    @qd.func(requires_top_level=True)
    def on_update_velocity(self):
        update_fem_velocity(self.data, self.sim_config_system.data)

    @qd.func(requires_top_level=True)
    def on_copy_previous_positions(self):
        copy_fem_x_prev(self.data)

    @qd.func(requires_top_level=True)
    def on_publish_trajectory_end_positions(self):
        publish_fem_trajectory_end_positions(self.data, self.global_vertex_system.data)


@qd.func(requires_top_level=True)
def sync_fem_from_scene(
    data: qd.template(),  # FiniteElementMethod.Data
    vertex: qd.template(),  # GlobalVertexManager.Data
):
    for i_vertex in range(data.n_fem_verts[()]):
        scene_vertex = data.bridge_vertex[i_vertex]
        environment = data.bridge_environment[i_vertex]
        constraint = data.scene_vertex_constraints[scene_vertex, environment]
        is_fixed = constraint.is_constrained and not constraint.is_soft_constraint and constraint.link_idx < 0
        data.is_fixed[i_vertex] = qd.cast(is_fixed, qd.i32)
        global_vertex = data.global_vert_offset[()] + i_vertex
        vertex.is_fixed[global_vertex] = qd.cast(is_fixed, qd.i32)
        for axis in qd.static(range(3)):
            position = data.scene_elements_v[data.scene_frame[()], scene_vertex, environment].pos[axis]
            velocity = data.scene_elements_v[data.scene_frame[()], scene_vertex, environment].vel[axis]
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
def initialize_fem_global_vertices(
    data: qd.template(),  # FiniteElementMethod.Data
    vertex: qd.template(),  # GlobalVertexManager.Data
):
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
def forward_fem_global_vertices(
    data: qd.template(),  # FiniteElementMethod.Data
    vertex: qd.template(),  # GlobalVertexManager.Data
):
    for i_vertex in range(data.n_fem_verts[()]):
        global_vertex = data.global_vert_offset[()] + i_vertex
        for axis in qd.static(range(3)):
            vertex.positions[global_vertex, axis] = data.x[i_vertex, axis]


@qd.func(requires_top_level=True)
def publish_fem_trajectory_end_positions(
    data: qd.template(),  # FiniteElementMethod.Data
    vertex: qd.template(),  # GlobalVertexManager.Data
):
    for i_vertex in range(data.n_fem_verts[()]):
        global_vertex = data.global_vert_offset[()] + i_vertex
        for axis in qd.static(range(3)):
            vertex.trajectory_end_positions[global_vertex, axis] = data.x_temp[i_vertex, axis] + data.dx[i_vertex, axis]


@qd.func(requires_top_level=True)
def reset_fem_energy(
    data: qd.template(),  # FiniteElementMethod.Data
):
    for _ in range(1):
        data.fem_energy[()] = qd.f64(0.0)


@qd.func(requires_top_level=True)
def negate_fem_dx(
    data: qd.template(),  # FiniteElementMethod.Data
    global_linear_system_data: qd.template(),  # GlobalLinearSystem.Data
):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            value = qd.f64(0.0)
            if data.is_fixed[i_vert] == 0:
                value = -global_linear_system_data.read_solution(data.dof_offset[()] + i_vert * 3 + axis)
            data.dx[i_vert, axis] = value


@qd.func(requires_top_level=True)
def contribute_fem_newton_max_disp(
    data: qd.template(),  # FiniteElementMethod.Data
    max_disp: qd.template(),  # qd.Ndarray
):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            qd.atomic_max(max_disp[()], qd.abs(data.dx[i_vert, axis]))


@qd.func(requires_top_level=True)
def record_fem_start_point(
    data: qd.template(),  # FiniteElementMethod.Data
):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            data.x_temp[i_vert, axis] = data.x[i_vert, axis]


@qd.func(requires_top_level=True)
def step_fem_forward(
    data: qd.template(),  # FiniteElementMethod.Data
    alpha,
):
    for i_vert in range(data.n_fem_verts[()]):
        if data.is_fixed[i_vert] == 0:
            for axis in qd.static(range(3)):
                data.x[i_vert, axis] = data.x_temp[i_vert, axis] + alpha * data.dx[i_vert, axis]


@qd.func(requires_top_level=True)
def update_fem_velocity(
    data: qd.template(),  # FiniteElementMethod.Data
    sim_config: qd.template(),  # SimConfig.Data
):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            if data.is_fixed[i_vert] != 0:
                data.velocities[i_vert, axis] = qd.f64(0.0)
            else:
                data.velocities[i_vert, axis] = (data.x[i_vert, axis] - data.x_prev[i_vert, axis]) / sim_config.dt[()]


@qd.func(requires_top_level=True)
def copy_fem_x_prev(
    data: qd.template(),  # FiniteElementMethod.Data
):
    for i_vert in range(data.n_fem_verts[()]):
        for axis in qd.static(range(3)):
            data.x_prev[i_vert, axis] = data.x[i_vert, axis]


@qd.kernel
def forward_fem_scene_vertices(
    data: qd.template(),  # FiniteElementMethod.Data
):
    for i_vert in range(data.n_fem_verts[()]):
        scene_vert = data.bridge_vertex[i_vert]
        environment = data.bridge_environment[i_vert]
        for axis in qd.static(range(3)):
            data.scene_elements_v[data.scene_frame[()], scene_vert, environment].pos[axis] = data.x[i_vert, axis]
            data.scene_elements_v[data.scene_frame[()], scene_vert, environment].vel[axis] = data.velocities[
                i_vert, axis
            ]
