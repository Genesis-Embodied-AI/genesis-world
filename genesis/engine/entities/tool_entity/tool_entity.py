from pathlib import Path

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.states.entities import ToolEntityState
from genesis.utils.geom import (
    qd_rotvec_to_quat,
    qd_transform_quat_by_quat,
    transform_pos_quat_by_trans_quat,
)
from genesis.utils.misc import to_gs_tensor

from ..base_entity import Entity
from .mesh import Mesh


@qd.data_oriented
class ToolEntity(Entity):
    # Mesh-based tool body entity
    def __init__(
        self,
        scene,
        idx,
        solver,
        material,
        morph,
        surface,
        name: str | None = None,
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        # The morph pose offset (e.g. an up-axis conversion) is composed onto the morph pose.
        self._init_pos, self._init_quat = transform_pos_quat_by_trans_quat(
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
            np.array(morph.pos, dtype=gs.np_float),
            np.array(morph.quat, dtype=gs.np_float),
        )

        self.mesh = Mesh(
            entity=self,
            material=material,
            morph=morph,
        )

        # for rendering purpose only
        self.latest_pos = qd.Vector.field(3, dtype=gs.qd_float, shape=(1))

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        if isinstance(self._morph, gs.morphs.Mesh):
            return Path(self._morph.file).stem
        return "tool"

    # ------------------------------------------------------------------------------------
    # --------------------------------- initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def reset_grad(self):
        self.pos.grad.fill(0)
        self.quat.grad.fill(0)
        self.vel.grad.fill(0)
        self.ang.grad.fill(0)

    def substep_pre_coupling(self, f):
        self.advect(f)

    def substep_pre_coupling_grad(self, f):
        self.advect.grad(f)

    def substep_post_coupling(self, f):
        self.update_latest_pos(f)

    def substep_post_coupling_grad(self, f):
        pass

    @qd.func
    def collide(self, f, pos_world, vel_mat, i_b):
        return self.mesh.collide(f, pos_world, vel_mat, i_b)

    @qd.func
    def pbd_collide(self, f, pos_world, thickness, dt):
        return self.mesh.pbd_collide(f, pos_world, thickness, dt)

    @qd.kernel
    def update_latest_pos(self, f: qd.i32):
        self.latest_pos[0] = self.pos[f, 0]

    @qd.kernel
    def advect(self, f: qd.i32):
        for i_b in range(self._sim._B):
            self.pos[f + 1, i_b] = self._solver.boundary.impose_pos(
                self.pos[f, i_b] + self.vel[f, i_b] * self._solver.substep_dt
            )
            # rotate in world coordinates about itself.
            self.quat[f + 1, i_b] = qd_transform_quat_by_quat(
                self.quat[f, i_b], qd_rotvec_to_quat(self.ang[f, i_b] * self._solver.substep_dt, gs.EPS)
            )

            self.vel[f + 1, i_b] = self.vel[f, i_b]
            self.ang[f + 1, i_b] = self.ang[f, i_b]

    # state set and copy ...
    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        for i_b in range(self._sim._B):
            self.pos[target, i_b] = self.pos[source, i_b]
            self.quat[target, i_b] = self.quat[source, i_b]
            self.vel[target, i_b] = self.vel[source, i_b]
            self.ang[target, i_b] = self.ang[source, i_b]

    @qd.kernel
    def copy_grad(self, source: qd.i32, target: qd.i32):
        for i_b in range(self._sim._B):
            self.pos.grad[target, i_b] = self.pos.grad[source, i_b]
            self.quat.grad[target, i_b] = self.quat.grad[source, i_b]
            self.vel.grad[target, i_b] = self.vel.grad[source, i_b]
            self.ang.grad[target, i_b] = self.ang.grad[source, i_b]

    @qd.kernel
    def reset_grad_till_frame(self, f: qd.i32):
        for i_b in range(self._sim._B):
            for i_f in range(f):
                self.pos.grad[i_f, i_b].fill(0)
                self.quat.grad[i_f, i_b].fill(0)
                self.vel.grad[i_f, i_b].fill(0)
                self.ang.grad[i_f, i_b].fill(0)

    @qd.kernel
    def get_frame(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        quat: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        ang: qd.types.ndarray(),
    ):
        for i_b in range(self._sim._B):
            for i in qd.static(range(3)):
                pos[i_b, i] = self.pos[f, i_b][i]
            for i in qd.static(range(4)):
                quat[i_b, i] = self.quat[f, i_b][i]
            for i in qd.static(range(3)):
                vel[i_b, i] = self.vel[f, i_b][i]
            for i in qd.static(range(3)):
                ang[i_b, i] = self.ang[f, i_b][i]

    @qd.kernel
    def set_frame(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        quat: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        ang: qd.types.ndarray(),
    ):
        for i_b in range(self._sim._B):
            for i in qd.static(range(3)):
                self.pos[f, i_b][i] = pos[i_b, i]
            for i in qd.static(range(4)):
                self.quat[f, i_b][i] = quat[i_b, i]
            for i in qd.static(range(3)):
                self.vel[f, i_b][i] = vel[i_b, i]
            for i in qd.static(range(3)):
                self.ang[f, i_b][i] = ang[i_b, i]

    @qd.kernel
    def set_frame_add_grad_pos(self, f: qd.i32, pos_grad: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for i in qd.static(range(3)):
                self.pos.grad[f, i_b][i] += pos_grad[i_b, i]

    @qd.kernel
    def set_frame_add_grad_quat(self, f: qd.i32, quat_grad: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for i in qd.static(range(4)):
                self.quat.grad[f, i_b][i] += quat_grad[i_b, i]

    @qd.kernel
    def set_frame_add_grad_vel(self, f: qd.i32, vel_grad: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for i in qd.static(range(3)):
                self.vel.grad[f, i_b][i] += vel_grad[i_b, i]

    @qd.kernel
    def set_frame_add_grad_ang(self, f: qd.i32, ang_grad: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for i in qd.static(range(3)):
                self.ang.grad[f, i_b][i] += ang_grad[i_b, i]

    def get_state(self, f=None):
        state = ToolEntityState(self, self._sim.cur_step_global)

        if f is None:
            f = self._sim.cur_substep_local
        self.get_frame(f, state.pos, state.quat, state.vel, state.ang)
        # The gradient of the state goes to this entity (see 'ToolSolver.add_grad_from_state')
        self._solver.record_read(state.grad_tensors, self)

        return state

    def set_state(self, f, state):
        f = self._sim.cur_substep_local
        self.set_frame(f, state.pos, state.quat, state.vel, state.ang)

    def build(self, n_frames: int):
        """Allocate the state of the tool over the given number of frames (see 'ToolSolver.roll_frame_window')."""
        shape = (n_frames, self._sim._B)
        self.pos = qd.Vector.field(3, gs.qd_float, shape=shape, needs_grad=True)
        # Quaternion (w, x, y, z)
        self.quat = qd.Vector.field(4, gs.qd_float, shape=shape, needs_grad=True)
        self.vel = qd.Vector.field(3, gs.qd_float, shape=shape, needs_grad=True)
        self.ang = qd.Vector.field(3, gs.qd_float, shape=shape, needs_grad=True)
        self.init_state = ToolEntityState(self, 0)
        self.set_init_state(self._init_pos, self._init_quat)

    @qd.kernel
    def set_init_state(self, pos: qd.types.ndarray(), quat: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for i in qd.static(range(3)):
                self.pos[0, i_b][i] = pos[i]
            for i in qd.static(range(4)):
                self.quat[0, i_b][i] = quat[i]

    @qd.kernel
    def set_vel(self, f: qd.i32, vel: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                self.vel[f, i_b][k] = vel[i_b, k]

    @qd.kernel
    def set_ang(self, f: qd.i32, ang: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                self.ang[f, i_b][k] = ang[i_b, k]

    @qd.kernel
    def set_pos(self, f: qd.i32, pos: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                self.pos[f, i_b][k] = pos[i_b, k]

    @qd.kernel
    def set_quat(self, f: qd.i32, quat: qd.types.ndarray()):
        for i_b in range(self._sim._B):
            for k in qd.static(range(4)):
                self.quat[f, i_b][k] = quat[i_b, k]

    def set_velocity(self, vel=None, ang=None):
        vel = None if vel is None else to_gs_tensor(vel).contiguous()
        ang = None if ang is None else to_gs_tensor(ang).contiguous()
        # Between two steps, the state sits in frame 0 (see 'ToolSolver.roll_frame_window')
        with self._solver.recording_input():
            if vel is not None:
                self.set_vel(0, vel)
            if ang is not None:
                self.set_ang(0, ang)

    def set_position(self, pos):
        pos = to_gs_tensor(pos).contiguous()
        # See 'set_velocity'
        with self._solver.recording_input():
            self.set_pos(0, pos)

    def set_quaternion(self, quat):
        quat = to_gs_tensor(quat).contiguous()
        # See 'set_velocity'
        with self._solver.recording_input():
            self.set_quat(0, quat)

    def add_grad_from_state(self, grads):
        """Add the gradients of a queried state of the tool, ordered as its `grad_tensors`, to the frame 0 adjoint."""
        kernels = (
            self.set_frame_add_grad_pos,
            self.set_frame_add_grad_quat,
            self.set_frame_add_grad_vel,
            self.set_frame_add_grad_ang,
        )
        for kernel, grad in zip(kernels, grads):
            if grad is not None:
                kernel(0, grad)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def idx(self):
        return self._idx

    @property
    def scene(self):
        return self._scene

    @property
    def solver(self):
        return self._solver

    @property
    def material(self):
        return self._material

    @property
    def morph(self):
        return self._morph

    @property
    def surface(self):
        return self._surface

    @property
    def init_pos(self):
        return self._init_pos

    @property
    def init_quat(self):
        return self._init_quat
