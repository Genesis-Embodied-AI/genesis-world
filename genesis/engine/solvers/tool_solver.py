from typing import TYPE_CHECKING

import quadrants as qd

import genesis.utils.array_class as array_class
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities.tool_entity.tool_entity import ToolEntity
from genesis.engine.materials import Tool
from genesis.engine.states.solvers import ToolSolverState

from .base_solver import Solver, TimeBasedMixin

if TYPE_CHECKING:
    from genesis.engine.entities import ToolEntity


@qd.data_oriented
class ToolSolver(TimeBasedMixin, Solver):
    """
    Note
    ----
    !! This class will be removed once we added differntiability to the RigidSolver. This is just a temporary solution to obtain rigid->soft one-way differeitable coupling.
    """

    material_cls = Tool

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self.floor_height = options.floor_height

        # boundary
        self.setup_boundary()

    def build(self):
        super().build()
        # The frames of the substeps the solver holds: those of one step with gradients, whose backward pass replays it
        # whole, otherwise those of one substep (see 'roll_frame_window')
        self._n_frames = self.substeps + 1 if self._requires_grad else 2
        for entity in self._entities:
            entity.build(self._n_frames)

    @property
    def is_active(self):
        return self.n_entities > 0

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self.floor_height)

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "ToolEntity":
        entity = ToolEntity(
            scene=self._scene,
            idx=idx,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            name=name,
        )
        self._entities.append(entity)
        return entity

    def reset_grad(self):
        for entity in self._entities:
            entity.reset_grad()

    def get_state(self, f):
        if self.is_active:
            state = ToolSolverState(self._scene)
            for entity in self._entities:
                state.entities.append(entity.get_state(f))
        else:
            state = None
        return state

    def set_state(self, f, state, envs_idx=None):
        # A state set whole overwrites the output of the last operation recorded (see 'Solver.record_input')
        if self._requires_grad:
            self._rollout_graph.mark_state_changed()
        if state is not None:
            assert len(state) == len(self._entities)
            for i, entity in enumerate(self._entities):
                entity.set_state(f, state[i])

    def substep_pre_coupling(self, f):
        for entity in self._entities:
            entity.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        for entity in self._entities[::-1]:
            entity.substep_pre_coupling_grad(f)

    def substep_post_coupling(self, f):
        for entity in self._entities:
            entity.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        for entity in self._entities[::-1]:
            entity.substep_post_coupling_grad(f)

    @property
    def is_differentiable(self) -> bool:
        return True

    def add_grad_from_state(self, grads, selection, saved_state):
        """Add the gradients of a queried state of the tool given as selection to its frame 0 adjoint."""
        selection.add_grad_from_state(grads)

    def roll_frame_window(self):
        for entity in self._entities:
            entity.copy_frame(self._n_frames - 1, 0)

    def roll_frame_window_grad(self):
        for entity in self._entities:
            entity.copy_grad(0, self._n_frames - 1)
            entity.reset_grad_till_frame(self._n_frames - 1)

    @property
    def data(self):
        """Yield the state of every tool, the frame 0 of which a boundary checkpoint holds (see 'Solver.data')."""
        for i_e, entity in enumerate(self._entities):
            for name, field in (("pos", entity.pos), ("quat", entity.quat), ("vel", entity.vel), ("ang", entity.ang)):
                yield array_class.ArrayItem(f"entities.{i_e}.{name}", array_class.DataKind.STATE, field, is_framed=True)

    @qd.func
    def pbd_collide(self, f, pos_world, thickness, dt):
        for entity in qd.static(self._entities):
            pos_world = entity.pbd_collide(f, pos_world, thickness, dt)
        return pos_world
