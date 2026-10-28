from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.materials import Tool
from genesis.engine.states.entities import ToolEntityState
from genesis.engine.system_data.tool import ToolEntityData
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.utils.mesh import compute_sdf_data, load_mesh
from genesis.utils.misc import to_gs_tensor

from ..base_entity import Entity, EntityDescription, VerticesDescription
from .mesh import Mesh


@dataclass(kw_only=True)
class ToolEntityDescription(VerticesDescription, EntityDescription):
    """The mesh of a tool, normalized and scaled, with its vertex normals, its faces and its signed distance field.

    The signed distance field and its transform from the mesh frame are None when the material disables collision.
    """

    material: Tool
    morph: Morph
    surface: Surface
    name: str | None
    init_pos: np.ndarray
    init_quat: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    sdf_voxels: np.ndarray | None
    T_mesh_to_sdf: np.ndarray | None

    @classmethod
    def resolve(cls, material, morph, surface, name, options):
        """Load the mesh of a tool from its morph, fit it in a unit box centered at the origin, and scale it.

        The signed distance field is sampled on the normalized mesh at the resolution of the material, and its
        transform from the mesh frame accounts for the scale of the morph. 'options' are the entity options the other
        arguments were taken from, kept as authored in the description.
        """
        mesh_orig = load_mesh(morph.file)
        scale = np.linalg.norm(mesh_orig.extents, ord=np.inf)
        center = np.mean(mesh_orig.bounds, axis=0)
        mesh = trimesh.Trimesh(
            vertices=(mesh_orig.vertices - center) / scale,
            faces=mesh_orig.faces,
            vertex_normals=mesh_orig.vertex_normals,
            face_normals=mesh_orig.face_normals,
        )
        T_init = gu.scale_to_T(np.array(morph.scale, dtype=gs.np_float))
        sdf_voxels = T_mesh_to_sdf = None
        if material.collision:
            sdf_data = compute_sdf_data(mesh, material.sdf_res)
            sdf_voxels = sdf_data["voxels"].astype(gs.np_float, order="C", copy=False)
            T_mesh_to_sdf = sdf_data["T_mesh_to_sdf"].astype(gs.np_float, order="C", copy=False) @ gu.inv_T(T_init)
        # The morph pose offset (e.g. an up-axis conversion) is composed onto the morph pose
        init_pos, init_quat = gu.transform_pos_quat_by_trans_quat(
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
            np.array(morph.pos, dtype=gs.np_float),
            np.array(morph.quat, dtype=gs.np_float),
        )
        return cls(
            material=material,
            morph=morph,
            surface=surface,
            name=name,
            init_pos=init_pos,
            init_quat=init_quat,
            init_positions=gu.transform_by_T(mesh.vertices.astype(gs.np_float, order="C", copy=False), T_init),
            normals=mesh.vertex_normals.astype(gs.np_float, order="C", copy=False),
            faces=mesh.faces.astype(gs.np_int, order="C", copy=False),
            sdf_voxels=sdf_voxels,
            T_mesh_to_sdf=T_mesh_to_sdf,
            options=options,
        )


@qd.data_oriented
class ToolEntity(Entity):
    """A mesh-based tool whose pose is driven by the user, and which the soft entities collide with.

    Parameters
    ----------
    scene : Scene
        The simulation scene.
    idx : int
        Index of this entity in the scene.
    solver : ToolSolver
        The solver handling the simulation logic.
    desc : ToolEntityDescription
        The resolved mesh and signed distance field, with the material, morph and surface they are resolved from.
    """

    def __init__(self, scene, idx, solver, desc: ToolEntityDescription):
        super().__init__(idx, scene, solver, desc, desc.morph)
        self._entity_data: ToolEntityData | None = None

        self.mesh = Mesh(entity=self, desc=desc)
        self.init_tgt_vars()
        self.init_ckpt()

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
                    "pos": torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device),
                    "quat": torch.zeros((self._sim._B, 4), dtype=gs.tc_float, device=gs.device),
                    "vel": torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device),
                    "ang": torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device),
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
        pass

    def substep_post_coupling_grad(self, f):
        pass

    @qd.func
    def collide(self, f, pos_world, vel_mat, i_b):
        return self.mesh.collide(f, pos_world, vel_mat, i_b)

    @qd.func
    def pbd_collide(self, f, pos_world, thickness, dt):
        return self.mesh.pbd_collide(f, pos_world, thickness, dt)

    @qd.kernel
    def advect(self, f: qd.i32):
        for i_b in range(self._sim._B):
            self.pos[f + 1, i_b] = self._solver.boundary.impose_pos(
                self.pos[f, i_b] + self.vel[f, i_b] * self._solver.substep_dt
            )
            # rotate in world coordinates about itself.
            self.quat[f + 1, i_b] = gu.qd_transform_quat_by_quat(
                self.quat[f, i_b], gu.qd_rotvec_to_quat(self.ang[f, i_b] * self._solver.substep_dt, gs.EPS)
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
        self._entity_data = self._sim._system_data.get(ToolEntityData, self._solver, self._idx)
        self.init_state = ToolEntityState(self, 0)
        self.set_init_state(self._desc.init_pos, self._desc.init_quat)

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
        inputs = tuple(self._tgt.values())

        # Detached, see ParticleEntity.process_input.
        if self._tgt["pos"] is not None:
            assert self._tgt["pos"].is_contiguous()
            self.set_pos(self._sim.cur_step_local, self._tgt["pos"].detach())

        if self._tgt["quat"] is not None:
            assert self._tgt["quat"].is_contiguous()
            self.set_quat(self._sim.cur_step_local, self._tgt["quat"].detach())

        if self._tgt["vel"] is not None:
            assert self._tgt["vel"].is_contiguous()
            self.set_vel(self._sim.cur_step_local, self._tgt["vel"].detach())

        if self._tgt["ang"] is not None:
            assert self._tgt["ang"].is_contiguous()
            self.set_ang(self._sim.cur_step_local, self._tgt["ang"].detach())

        self._tgt["pos"] = None
        self._tgt["quat"] = None
        self._tgt["vel"] = None
        self._tgt["ang"] = None
        return inputs

    def process_input_grad(self):
        """Return the pairs (input, gradient) of the inputs applied at this step that track gradients."""
        _tgt_pos = self._tgt_buffer["pos"].pop()
        _tgt_quat = self._tgt_buffer["quat"].pop()
        _tgt_vel = self._tgt_buffer["vel"].pop()
        _tgt_ang = self._tgt_buffer["ang"].pop()

        inputs_grad = []
        for tgt, set_grad in (
            (_tgt_vel, self.set_vel_grad),
            (_tgt_ang, self.set_ang_grad),
            (_tgt_pos, self.set_pos_grad),
            (_tgt_quat, self.set_quat_grad),
        ):
            if tgt is not None and tgt.requires_grad:
                grad = torch.zeros_like(tgt)
                set_grad(self._sim.cur_step_local, grad)
                inputs_grad.append((tgt, grad))
        return inputs_grad

    def add_grad_from_state(self, state):
        if state.pos.grad is not None:
            assert state.pos.grad.is_contiguous()
            self.set_frame_add_grad_pos(self._sim.cur_substep_local, state.pos.grad)

        if state.quat.grad is not None:
            assert state.quat.grad.is_contiguous()
            self.set_frame_add_grad_quat(self._sim.cur_substep_local, state.quat.grad)

        if state.vel.grad is not None:
            assert state.vel.grad.is_contiguous()
            self.set_frame_add_grad_vel(self._sim.cur_substep_local, state.vel.grad)

        if state.ang.grad is not None:
            assert state.ang.grad.is_contiguous()
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
    def desc(self) -> ToolEntityDescription:
        return self._desc

    @property
    def pos(self) -> qd.Field:
        """The position of this entity at each local substep and in each environment."""
        return self._entity_data.pos

    @property
    def quat(self) -> qd.Field:
        """The orientation quaternion (w, x, y, z) of this entity at each local substep and in each environment."""
        return self._entity_data.quat

    @property
    def vel(self) -> qd.Field:
        """The linear velocity of this entity at each local substep and in each environment."""
        return self._entity_data.vel

    @property
    def ang(self) -> qd.Field:
        """The angular velocity of this entity at each local substep and in each environment."""
        return self._entity_data.ang

    @property
    def init_pos(self):
        return self._desc.init_pos

    @property
    def init_quat(self):
        return self._desc.init_quat
