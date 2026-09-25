from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.states.cache import QueriedStates
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
        self.init_tgt_vars()
        self.init_ckpt()
        self._queried_states = QueriedStates()

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

    def init_tgt_vars(self):
        # temp variable to store targets for next step
        self._tgt = {
            "pos": None,
            "quat": None,
            "vel": None,
            "ang": None,
        }

        self._tgt_buffer = {
            "pos": list(),
            "quat": list(),
            "vel": list(),
            "ang": list(),
        }

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        self.pos.grad.fill(0)
        self.quat.grad.fill(0)
        self.vel.grad.fill(0)
        self.ang.grad.fill(0)

        self._tgt_buffer["pos"].clear()
        self._tgt_buffer["quat"].clear()
        self._tgt_buffer["vel"].clear()
        self._tgt_buffer["ang"].clear()

        self._queried_states.clear()

    @qd.kernel
    def save_ckpt_kernel(
        self, pos: qd.types.ndarray(), quat: qd.types.ndarray(), vel: qd.types.ndarray(), ang: qd.types.ndarray()
    ):
        for i_b in range(self._sim._B):
            for i in qd.static(range(3)):
                pos[i_b, i] = self.pos[0, i_b][i]
                vel[i_b, i] = self.vel[0, i_b][i]
                ang[i_b, i] = self.ang[0, i_b][i]

            for i in qd.static(range(4)):
                quat[i_b, i] = self.quat[0, i_b][i]

    @qd.kernel
    def load_ckpt_kernel(
        self, pos: qd.types.ndarray(), quat: qd.types.ndarray(), vel: qd.types.ndarray(), ang: qd.types.ndarray()
    ):
        for i_b in range(self._sim._B):
            for i in qd.static(range(3)):
                self.pos[0, i_b][i] = pos[i_b, i]
                self.vel[0, i_b][i] = vel[i_b, i]
                self.ang[0, i_b][i] = ang[i_b, i]

            for i in qd.static(range(4)):
                self.quat[0, i_b][i] = quat[i_b, i]

    def save_ckpt(self, ckpt_name):
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = {
                    "pos": torch.zeros((self._sim._B, 3), dtype=gs.tc_float),
                    "quat": torch.zeros((self._sim._B, 4), dtype=gs.tc_float),
                    "vel": torch.zeros((self._sim._B, 3), dtype=gs.tc_float),
                    "ang": torch.zeros((self._sim._B, 3), dtype=gs.tc_float),
                    "_tgt_buffer": dict(),
                }

            self.save_ckpt_kernel(
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["quat"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["ang"],
            )

            self._ckpt[ckpt_name]["_tgt_buffer"]["pos"] = list(self._tgt_buffer["pos"])
            self._ckpt[ckpt_name]["_tgt_buffer"]["quat"] = list(self._tgt_buffer["quat"])
            self._ckpt[ckpt_name]["_tgt_buffer"]["vel"] = list(self._tgt_buffer["vel"])
            self._ckpt[ckpt_name]["_tgt_buffer"]["ang"] = list(self._tgt_buffer["ang"])

            self._tgt_buffer["pos"].clear()
            self._tgt_buffer["quat"].clear()
            self._tgt_buffer["vel"].clear()
            self._tgt_buffer["ang"].clear()

        # restart from frame 0 in memory
        self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)
        self.reset_grad_till_frame(self._sim.substeps_local)

        self.load_ckpt_kernel(
            self._ckpt[ckpt_name]["pos"],
            self._ckpt[ckpt_name]["quat"],
            self._ckpt[ckpt_name]["vel"],
            self._ckpt[ckpt_name]["ang"],
        )

        self._tgt_buffer["pos"] = list(self._ckpt[ckpt_name]["_tgt_buffer"]["pos"])
        self._tgt_buffer["quat"] = list(self._ckpt[ckpt_name]["_tgt_buffer"]["quat"])
        self._tgt_buffer["vel"] = list(self._ckpt[ckpt_name]["_tgt_buffer"]["vel"])
        self._tgt_buffer["ang"] = list(self._ckpt[ckpt_name]["_tgt_buffer"]["ang"])

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

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    def set_state(self, f, state):
        f = self._sim.cur_substep_local
        self.set_frame(f, state.pos, state.quat, state.vel, state.ang)

    def build(self):
        self.pos = qd.Vector.field(3, gs.qd_float, needs_grad=True)  # positon
        self.quat = qd.Vector.field(4, gs.qd_float, needs_grad=True)  # quaternion wxyz
        self.vel = qd.Vector.field(3, gs.qd_float, needs_grad=True)  # velocity
        self.ang = qd.Vector.field(3, gs.qd_float, needs_grad=True)  # angular velocity

        qd.root.dense(qd.ij, (self._sim.substeps_local + 1, self._sim._B)).place(
            self.pos, self.pos.grad, self.quat, self.quat.grad, self.vel, self.vel.grad, self.ang, self.ang.grad
        )
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
    def set_vel(self, s: qd.i32, vel: qd.types.ndarray()):
        f = s * self._sim.substeps
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                self.vel[f, i_b][k] = vel[i_b, k]

    @qd.kernel
    def set_vel_grad(self, s: qd.i32, vel_grad: qd.types.ndarray()):
        f = s * self._sim.substeps
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                vel_grad[i_b, k] += self.vel.grad[f, i_b][k]

    @qd.kernel
    def set_ang(self, s: qd.i32, ang: qd.types.ndarray()):
        f = s * self._sim.substeps
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                self.ang[f, i_b][k] = ang[i_b, k]

    @qd.kernel
    def set_ang_grad(self, s: qd.i32, ang_grad: qd.types.ndarray()):
        f = s * self._sim.substeps
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                ang_grad[i_b, k] += self.ang.grad[f, i_b][k]

    @qd.kernel
    def set_pos(self, s: qd.i32, pos: qd.types.ndarray()):
        f = s * self._sim.substeps
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                self.pos[f, i_b][k] = pos[i_b, k]

    @qd.kernel
    def set_pos_grad(self, s: qd.i32, pos_grad: qd.types.ndarray()):
        f = s * self._sim.substeps
        for i_b in range(self._sim._B):
            for k in qd.static(range(3)):
                pos_grad[i_b, k] += self.pos.grad[f, i_b][k]

    @qd.kernel
    def set_quat(self, s: qd.i32, quat: qd.types.ndarray()):
        f = s * self._sim.substeps
        for i_b in range(self._sim._B):
            for k in qd.static(range(4)):
                self.quat[f, i_b][k] = quat[i_b, k]

    @qd.kernel
    def set_quat_grad(self, s: qd.i32, quat_grad: qd.types.ndarray()):
        f = s * self._sim.substeps
        for i_b in range(self._sim._B):
            for k in qd.static(range(4)):
                quat_grad[i_b, k] += self.quat.grad[f, i_b][k]

    def set_velocity(self, vel=None, ang=None):
        if vel is not None:
            vel = to_gs_tensor(vel)
            self._tgt["vel"] = vel

        if ang is not None:
            ang = to_gs_tensor(ang)
            self._tgt["ang"] = ang

    def set_position(self, pos):
        pos = to_gs_tensor(pos)
        self._tgt["pos"] = pos

    def set_quaternion(self, quat):
        quat = to_gs_tensor(quat)
        self._tgt["quat"] = quat

    def process_input(self, in_backward=False):
        if in_backward:
            self._tgt["pos"] = self._tgt_buffer["pos"][self._sim.cur_step_local]
            self._tgt["quat"] = self._tgt_buffer["quat"][self._sim.cur_step_local]
            self._tgt["vel"] = self._tgt_buffer["vel"][self._sim.cur_step_local]
            self._tgt["ang"] = self._tgt_buffer["ang"][self._sim.cur_step_local]
        else:
            self._tgt_buffer["pos"].append(self._tgt["pos"])
            self._tgt_buffer["quat"].append(self._tgt["quat"])
            self._tgt_buffer["vel"].append(self._tgt["vel"])
            self._tgt_buffer["ang"].append(self._tgt["ang"])

        if self._tgt["pos"] is not None:
            self._tgt["pos"].assert_contiguous()
            self._tgt["pos"].assert_sceneless()
            self.set_pos(self._sim.cur_step_local, self._tgt["pos"])

        if self._tgt["quat"] is not None:
            self._tgt["quat"].assert_contiguous()
            self._tgt["quat"].assert_sceneless()
            self.set_quat(self._sim.cur_step_local, self._tgt["quat"])

        if self._tgt["vel"] is not None:
            self._tgt["vel"].assert_contiguous()
            self._tgt["vel"].assert_sceneless()
            self.set_vel(self._sim.cur_step_local, self._tgt["vel"])

        if self._tgt["ang"] is not None:
            self._tgt["ang"].assert_contiguous()
            self._tgt["ang"].assert_sceneless()
            self.set_ang(self._sim.cur_step_local, self._tgt["ang"])

        self._tgt["pos"] = None
        self._tgt["quat"] = None
        self._tgt["vel"] = None
        self._tgt["ang"] = None

    def process_input_grad(self):
        _tgt_pos = self._tgt_buffer["pos"].pop()
        _tgt_quat = self._tgt_buffer["quat"].pop()
        _tgt_vel = self._tgt_buffer["vel"].pop()
        _tgt_ang = self._tgt_buffer["ang"].pop()

        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_qd(self.set_vel_grad, self._sim.cur_step_local)

        if _tgt_ang is not None and _tgt_ang.requires_grad:
            _tgt_ang._backward_from_qd(self.set_ang_grad, self._sim.cur_step_local)

        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_qd(self.set_pos_grad, self._sim.cur_step_local)

        if _tgt_quat is not None and _tgt_quat.requires_grad:
            _tgt_quat._backward_from_qd(self.set_quat_grad, self._sim.cur_step_local)

    def collect_output_grads(self):
        """
        Collect gradients from external queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            for state in self._queried_states[self._sim.cur_step_global]:
                self.add_grad_from_state(state)

    def add_grad_from_state(self, state):
        if state.pos.grad is not None:
            state.pos.assert_contiguous()
            self.set_frame_add_grad_pos(self._sim.cur_substep_local, state.pos.grad)

        if state.quat.grad is not None:
            state.quat.assert_contiguous()
            self.set_frame_add_grad_quat(self._sim.cur_substep_local, state.quat.grad)

        if state.vel.grad is not None:
            state.vel.assert_contiguous()
            self.set_frame_add_grad_vel(self._sim.cur_substep_local, state.vel.grad)

        if state.ang.grad is not None:
            state.ang.assert_contiguous()
            self.set_frame_add_grad_ang(self._sim.cur_substep_local, state.ang.grad)

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
