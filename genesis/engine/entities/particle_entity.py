import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.utils.misc import broadcast_tensor, to_gs_tensor

from .base_entity import Entity, EntityDescription, VerticesDescription

# Number of vvert-particle distances computed at once when searching the supports of the vverts. Their difference
# vectors take 24 MiB in double precision, while chunks of this size still amortize the per-chunk overhead of numpy.
VVERTS_DIST2_MAX_SIZE = 2**20


@dataclass(kw_only=True)
class ParticleEntityDescription(VerticesDescription, EntityDescription):
    morph: Morph
    surface: Surface
    name: str | None
    particle_size: float
    has_skinning: bool
    sampler: str | None
    origin: np.ndarray
    vmesh: gs.Mesh | None
    vverts: np.ndarray
    vfaces: np.ndarray
    mesh_set_group_ids: np.ndarray | None

    @staticmethod
    def resolve_mesh(morph, surface):
        """Load the visual mesh of a morph, one per member for a mesh set, or None for a morph holding no mesh."""
        if isinstance(morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            meshes = gs.Mesh.from_morph_surface(morph, surface)
            if isinstance(morph, gs.options.morphs.MeshSet):
                return meshes
            if len(meshes) > 1:
                gs.raise_exception("Mesh file with multiple sub-meshes are not supported.")
            return meshes[0]
        surface.update_texture()
        return None

    @classmethod
    def resolve(cls, material, morph, surface, particle_size, name, boundary, *, has_skinning, options):
        """Sample the particles of an entity from its morph, and place them and its visual mesh at the morph pose.

        The particles must lie inside 'boundary', and the visual vertices and faces are kept for skinning when
        'has_skinning' is set. 'options' are the entity options the other arguments were taken from, kept as
        authored in the description.
        """
        sampler = material.sampler
        match sampler.split("-"):
            case ["regular"] | ["random"]:
                pass
            case ["pbs"]:
                sampler = "pbs-32"
            case ["pbs", num] if num.isnumeric():
                pass
            case _:
                gs.raise_exception(
                    "Only one of the following samplers is supported: [`regular`, `random`, `pbs`, `pbs-sdf_res`]. "
                    f"Got: {sampler}."
                )

        if isinstance(morph, gs.options.morphs.MeshSet) and not morph.files:
            gs.raise_exception("A mesh set needs at least one mesh file.")
        vmesh = cls.resolve_mesh(morph, surface)
        mesh_set_group_ids = None
        if isinstance(morph, gs.options.morphs.MeshSet):
            surface = vmesh[0].surface
            particles_per_mesh = []
            for mesh, pos, euler in zip(vmesh, morph.poss, morph.eulers, strict=True):
                pos = np.array(pos, dtype=gs.np_float)
                quat = gu.xyz_to_quat(np.array(euler, dtype=gs.np_float), rpy=True, degrees=True)
                # Sample before posing the mesh, so the lattice and the sampling cache depend on the member shape alone
                particles = mesh.particlize(particle_size, sampler)
                particles_per_mesh.append(gu.transform_by_trans_quat(particles, pos, quat))
                mesh.apply_transform(gu.trans_quat_to_T(pos, quat))
            mesh_set_group_ids = np.concatenate(
                [np.full(len(particles), i, dtype=gs.np_int) for i, particles in enumerate(particles_per_mesh)]
            )
            particles = np.concatenate(particles_per_mesh)
            verts, faces = trimesh.util.append_faces([mesh.verts for mesh in vmesh], [mesh.faces for mesh in vmesh])
            normals = np.concatenate([mesh.trimesh.vertex_normals for mesh in vmesh])
            vmesh = mu.trimesh_to_mesh(
                trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals), scale=1, surface=surface
            )
            origin = np.mean(morph.poss, axis=0, dtype=gs.np_float)
        elif isinstance(morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            surface = vmesh.surface
            particles = vmesh.particlize(particle_size, sampler)
            # The morph pose offset (e.g. an up-axis conversion) is composed onto the morph pose
            pos, quat = gu.transform_pos_quat_by_trans_quat(
                np.array(morph.offset_pos, dtype=gs.np_float),
                np.array(morph.offset_quat, dtype=gs.np_float),
                np.array(morph.pos, dtype=gs.np_float),
                np.array(morph.quat, dtype=gs.np_float),
            )
            vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
            particles = gu.transform_by_trans_quat(particles, pos, quat)
            origin = pos
        elif isinstance(morph, gs.options.morphs.Nowhere):
            particles = pu.nowhere_particles(morph.n_particles)
            origin = gu.nowhere()
        else:
            gs.raise_exception(f"Unsupported morph: {morph}.")

        if len(particles) == 0:
            gs.raise_exception("Entity has zero particles.")
        if not isinstance(morph, gs.options.morphs.Nowhere) and not boundary.is_inside(particles):
            gs.raise_exception(
                "Entity has particles outside solver boundary. Note that for MPMSolver, boundary is slightly tighter "
                f"than the specified domain due to safety padding.\n\nCurrent boundary:\n{boundary}\n\nEntity to be "
                f"added:\nmin: {particles.min(0)}\nmax: {particles.max(0)}\n"
            )
        if has_skinning and vmesh is not None:
            vverts = vmesh.verts.astype(gs.np_float, copy=False)
            vfaces = vmesh.faces.astype(gs.np_int, copy=False)
        else:
            vverts = np.zeros((0, 3), dtype=gs.np_float)
            vfaces = np.zeros((0, 3), dtype=gs.np_int)
        return cls(
            material=material,
            morph=morph,
            surface=surface,
            name=name,
            particle_size=particle_size,
            has_skinning=has_skinning,
            sampler=sampler,
            init_positions=particles.astype(gs.np_float, order="C", copy=False),
            origin=origin,
            vmesh=vmesh,
            vverts=vverts,
            vfaces=vfaces,
            mesh_set_group_ids=mesh_set_group_ids,
            options=options,
        )


def assert_active(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.active:
            gs.raise_exception(f"'{self.__class__.__name__}' is not active. Call `entity.activate()` first.")
        return method(self, *args, **kwargs)

    return wrapper


@qd.data_oriented
class ParticleEntity(Entity):
    """
    Base class for particle-based entity.

    Parameters
    ----------
    scene : Scene
        The scene object that this entity belongs to.
    solver : Solver
        The physics solver responsible for simulating the entity's particles.
    desc : ParticleEntityDescription
        The resolved particles and visual mesh, with the material, morph and surface they are resolved from.
    idx : int
        Index of this entity in the simulation.
    particle_start : int
        Global index offset for this entity's particles in the solver.
    vvert_start : int, optional
        Global index offset for vertex-based rendering, used for skinning.
    vface_start : int, optional
        Global index offset for visual faces, used for skinning.
    """

    def __init__(
        self, scene, solver, desc: ParticleEntityDescription, idx, particle_start, vvert_start=None, vface_start=None
    ):
        super().__init__(idx, scene, solver, desc, desc.morph)

        self._particle_size = desc.particle_size
        self._particle_start = particle_start

        if desc.has_skinning:
            self._vvert_start = vvert_start
            self._vface_start = vface_start
        else:
            self._vvert_start = -1
            self._vface_start = -1

        self.sampler = desc.sampler
        self._vmesh = desc.vmesh
        self._vverts = desc.vverts
        self._vfaces = desc.vfaces
        self._particles = desc.init_positions
        self.mesh_set_group_ids = desc.mesh_set_group_ids
        self._init_particles_offset = torch.tensor(self._particles - desc.origin, dtype=gs.tc_float, device=gs.device)
        self._n_particles = len(self._particles)
        gs.logger.info(f"Sampled ~<{self._n_particles:,}>~ particles.")

        self.init_tgt_vars()
        self.init_ckpt()

        # Note that this attribute must only be used in forward pass
        self.active = False

    def _sanitize_particles_idx_local(self, particles_idx_local=None, envs_idx=None):
        if particles_idx_local is None:
            particles_idx_local = range(self._n_particles)

        if envs_idx is None:
            particles_idx_local_ = broadcast_tensor(particles_idx_local, gs.tc_int, (-1,), ("envs_idx",))
        else:
            particles_idx_local_ = broadcast_tensor(
                particles_idx_local, gs.tc_int, (len(envs_idx), -1), ("envs_idx", "")
            )

        # FIXME: This check is too expensive
        # if not (0 <= particles_idx_local_ & particles_idx_local_ < self._n_particles).all():
        #     gs.raise_exception("Elements of `particles_idx_local' are out-of-range.")

        return particles_idx_local_.contiguous()

    def _sanitize_particles_tensor(
        self, tensor, dtype, particles_idx=None, envs_idx=None, element_shape=(), *, batched=True
    ):
        n_particles = particles_idx.shape[-1] if particles_idx is not None else self._n_particles
        if batched:
            assert envs_idx is not None
            batch_shape = (len(envs_idx), n_particles)
            dim_names = ("envs_idx", "particles_idx", *("" for _ in element_shape))
        else:
            batch_shape = (n_particles,)
            dim_names = ("particles_idx", *("" for _ in element_shape))
        tensor_shape = (*batch_shape, *element_shape)

        return broadcast_tensor(tensor, dtype, tensor_shape, dim_names).contiguous()

    def init_tgt_keys(self):
        """
        Initialize the list of keys used for controlling entity state (position, velocity, activation).
        """
        self._tgt_keys = ("pos", "vel", "act")

    def _add_to_solver(self):
        # For nowhere entity, it will be activated when particles are later added.
        # Note that 'self.active' is not informative for nowhere entity, and such entity is not supported in
        # differentiable mode.
        self.active = not isinstance(self._morph, gs.options.morphs.Nowhere)

        self._add_particles_to_solver()
        if self._desc.has_skinning:
            self._add_vverts_to_solver()

    def _add_particles_to_solver(self):
        raise NotImplementedError

    def _add_vverts_to_solver(self):
        """Compute the supports of the vverts among their neighboring particles, for rendering.

        The vverts are processed by chunks whose distances to all the particles take a bounded memory. A chunk holds one
        vvert at least, whose distances take as much memory as the positions of the particles.
        """
        n_supports = self.solver.n_vvert_supports
        chunk_size = max(VVERTS_DIST2_MAX_SIZE // len(self._particles), 1)
        support_idxs = np.empty((len(self._vverts), n_supports), dtype=gs.np_int)
        for i_vv_start in range(0, len(self._vverts), chunk_size):
            vverts_chunk = self._vverts[i_vv_start : i_vv_start + chunk_size]
            dist2 = np.sum(np.square(vverts_chunk[:, None, :] - self._particles), axis=-1)
            chunk_support_idxs = np.argpartition(dist2, n_supports - 1, axis=1)[:, :n_supports]
            chunk_support_dist2 = np.take_along_axis(dist2, chunk_support_idxs, axis=1)
            sorted_order = np.lexsort((chunk_support_idxs, chunk_support_dist2))
            support_idxs[i_vv_start : i_vv_start + chunk_size] = np.take_along_axis(
                chunk_support_idxs, sorted_order, axis=1
            )
        support_idxs = np.clip(support_idxs, 0, len(self._particles) - 1)

        all_ps = self._particles[support_idxs]
        Ps = all_ps[:, :-1].swapaxes(-2, -1) - np.expand_dims(all_ps[:, -1], axis=-1)
        P_invs = np.linalg.pinv(Ps)

        self._kernel_add_vverts_to_solver(
            vverts=self._vverts,
            particles=self._particles,
            P_invs=P_invs,
            support_idxs_local=support_idxs,
        )

    @qd.kernel
    def _kernel_add_vverts_to_solver(
        self,
        vverts: qd.types.ndarray(element_dim=1),
        particles: qd.types.ndarray(element_dim=1),
        P_invs: qd.types.ndarray(element_dim=2),
        support_idxs_local: qd.types.ndarray(),
    ):
        for i_vv_ in range(self.n_vverts):
            i_vv = i_vv_ + self._vvert_start
            v = vverts[i_vv_]
            P_inv = P_invs[i_vv_]
            b = P_inv @ (v - particles[support_idxs_local[i_vv_, self.solver.n_vvert_supports - 1]])
            for j in range(self.solver.n_vvert_supports - 1):
                self.solver.vverts_info.support_weights[i_vv][j] = b[j]
            self.solver.vverts_info.support_weights[i_vv][self.solver.n_vvert_supports - 1] = 1.0 - b.sum()
            for j in range(self.solver.n_vvert_supports):
                self.solver.vverts_info.support_idxs[i_vv][j] = support_idxs_local[i_vv_, j] + self._particle_start

    def init_tgt_vars(self):
        """
        Initialize target buffers used for controlling and differentiating entity state over time.
        """
        # temp variable to store targets for next step
        self._tgt = dict()
        self._tgt_buffer = dict()
        self.init_tgt_keys()

        for key in self._tgt_keys:
            self._tgt[key] = None
            self._tgt_buffer[key] = list()

    def init_ckpt(self):
        """
        Initialize checkpoint storage for simulation state.
        """
        self._ckpt = dict()

    def save_ckpt(self, ckpt_name):
        """
        Save the current target state buffers to a checkpoint.

        Parameters
        ----------
        ckpt_name : str
            Name of the checkpoint to save.
        """
        if ckpt_name not in self._ckpt:
            self._ckpt[ckpt_name] = {"_tgt_buffer": dict()}

        for key in self._tgt_keys:
            self._ckpt[ckpt_name]["_tgt_buffer"][key] = list(self._tgt_buffer[key])
            self._tgt_buffer[key].clear()

    def load_ckpt(self, ckpt_name):
        """
        Restore target state buffers from a previously saved checkpoint.

        Parameters
        ----------
        ckpt_name : str
            Name of the checkpoint to load.
        """
        for key in self._tgt_keys:
            self._tgt_buffer[key] = list(self._ckpt[ckpt_name]["_tgt_buffer"][key])

    def reset_grad(self):
        """
        Clear target buffers and any externally queried simulation states.

        Used before backpropagation to reset gradients.
        """
        for key in self._tgt_keys:
            self._tgt_buffer[key].clear()
        self._queried_states.clear()

    def _reset_grad(self):
        """
        Clear gradients.
        """
        raise NotImplementedError

    def add_grad_from_state(state):
        raise NotImplementedError

    def process_input(self, in_backward: bool = False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward gradient pass.

        Returns
        -------
        inputs : tuple
            The inputs applied at this step, of any type, for the simulator to tape.
        """
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim._steps_local
            for key in self._tgt_keys:
                self._tgt[key] = self._tgt_buffer[key][index]
        else:
            for key in self._tgt_keys:
                self._tgt_buffer[key].append(self._tgt[key])
        inputs = tuple(self._tgt.values())

        if any(self._tgt[key] is not None for key in self._tgt_keys):
            particles_idx_local = self._sanitize_particles_idx_local(envs_idx=self._scene._envs_idx)

        # Note that setting positions resets velocities to zero, so it must be done BEFORE setting velocities.
        # The setter kernels of every deformable entity take detached inputs: they only read the values, the gradient
        # reaches each input through `process_input_grad`, and a gradient-tracking kernel argument makes Quadrants
        # allocate a `.grad` buffer for it.
        if self._tgt["pos"] is not None:
            assert self._tgt["pos"].is_contiguous()
            self.set_particles_pos(self._tgt["pos"].detach(), particles_idx_local)

        if self._tgt["vel"] is not None:
            assert self._tgt["vel"].is_contiguous()
            self.set_particles_vel(self._tgt["vel"].detach(), particles_idx_local)

        if self._tgt["act"] is not None:
            assert self._tgt["act"].is_contiguous()
            act_values = torch.tensor((gs.ACTIVE, gs.INACTIVE), dtype=gs.tc_int, device=gs.device)
            assert torch.isin(self._tgt["act"], act_values).all()
            self.set_particles_active(self._tgt["act"], particles_idx_local)

        # Only the entities of a muscle material tape an actuation (see init_tgt_keys)
        if "actu" in self._tgt_keys and self._tgt["actu"] is not None:
            assert self._tgt["actu"].is_contiguous()
            self.set_particles_actu(self._tgt["actu"].detach(), particles_idx_local)

        for key in self._tgt_keys:
            self._tgt[key] = None
        return inputs

    def process_input_grad(self):
        """
        Process gradients of input states and propagate them backward.

        Returns
        -------
        inputs_grad : list of tuple
            The pairs (input, gradient) of the inputs applied at this step that track gradients.

        Notes
        -----
        Clears the gradients in the solver to avoid double accumulation.
        """
        _tgt_pos = self._tgt_buffer["pos"].pop()
        _tgt_vel = self._tgt_buffer["vel"].pop()

        inputs_grad = []
        for tgt, set_grad in ((_tgt_pos, self._set_particles_pos_grad), (_tgt_vel, self._set_particles_vel_grad)):
            if tgt is not None and tgt.requires_grad:
                grad = torch.zeros_like(tgt)
                set_grad(grad)
                inputs_grad.append((tgt, grad))

        # Manually zero the grad since manually setting state breaks gradient flow
        if _tgt_vel is not None or _tgt_pos is not None:
            self._reset_grad()
        return inputs_grad

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    @assert_active
    def _set_particles_target_state(self, key, name, element_shape, dtype, tensor, envs_idx=None):
        # Overwriting the position or velocity of the particles cuts the gradient of everything that led to them
        if key in ("pos", "vel") and self.sim.requires_grad and self.sim.cur_t > 0.0:
            gs.logger.warning(
                f"Manually setting particle '{name}'. This is not recommended because it breaks gradient flow."
            )
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vel_ = self._sanitize_particles_tensor(tensor, dtype, None, envs_idx, element_shape)
        self._tgt[key] = to_gs_tensor(vel_)

    def set_position(self, value, envs_idx=None):
        """
        Set the position of all the particles individually, or the center of mass wrt the initial configuration of the
        particles as a whole.

        Parameters
        ----------
        value : array_like, shape ([M,] [n_particles,] 3)
            Tensor of particle positions.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        # Determine whether the position of all the particles has been specified, or only the center of mass
        poss = to_gs_tensor(value, dtype=gs.tc_float)
        if poss.ndim == 1 or (poss.ndim == 2 and poss.shape[0] != self._n_particles):
            poss = self._init_particles_offset + poss[..., None, :]

        self._set_particles_target_state("pos", "position", (3,), gs.tc_float, poss, envs_idx)

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None):
        """
        Set the position of some particles.

        Parameters
        ----------
        poss: torch.Tensor, shape (M, N, 3)
            Target position of each particle.
        particles_idx_local : torch.Tensor, shape (M, N)
            Index of the particles relative to this entity. If None, all particles will be considered. Defaults to None.
        envs_idx : torch.Tensor, shape (M,)
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        raise NotImplementedError

    @gs.assert_built
    def _set_particles_pos_grad(self, poss_grad):
        """
        Set gradients for particle positions at a given substep.

        Parameters
        ----------
        poss_grad : torch.Tensor, shape (M, n_particles, 3)
            The gradients for particle positions.
        """
        raise NotImplementedError

    @gs.assert_built
    def get_particles_pos(self, envs_idx=None):
        """
        Retrieve current particle positions from the solver.

        Parameters
        ----------
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        poss : torch.Tensor, shape (M, n_particles, 3)
            Tensor of particle positions.
        """
        raise NotImplementedError

    def set_velocity(self, vels, envs_idx=None):
        """
        Set the velocity of all the particles individually.

        Parameters
        ----------
        vels : array_like, shape ([M,] [n_particles,] 3)
            Tensor of particle velocities.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        self._set_particles_target_state("vel", "velocity", (3,), gs.tc_float, vels, envs_idx)

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None):
        """
        Set the velocity of some particles.

        Parameters
        ----------
        vels: torch.Tensor, shape (M, N, 3)
            Target velocity of each particle.
        particles_idx_local : torch.Tensor, shape (M, N)
            Index of the particles relative to this entity. If None, all particles will be considered. Defaults to None.
        envs_idx : torch.Tensor, shape (M,)
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        raise NotImplementedError

    @gs.assert_built
    def _set_particles_vel_grad(self, vels_grad):
        """
        Set gradients for particle velocities at a specific frame.

        Parameters
        ----------
        vels_grad : torch.Tensor, shape (M, n_particles, 3)
            The gradients for particle velocities.
        """
        raise NotImplementedError

    @gs.assert_built
    def get_particles_vel(self, envs_idx=None):
        """
        Retrieve current particle velocities from the solver.

        Parameters
        ----------
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        poss : torch.Tensor
            Tensor of particle velocities, shape (M, n_particles, 3).
        """
        raise NotImplementedError

    def set_active(self, actives, envs_idx=None):
        """
        Set the activeness state of all the particles individually.

        Parameters
        ----------
        actives : int, array_like, shape ([M,] [n_particles,])
            Tensor of particle activeness boolean flags.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        self._set_particles_target_state("act", "activeness", (), gs.tc_bool, actives, envs_idx)
        self.active = bool((self._tgt["act"] == gs.ACTIVE).any())

    def activate(self):
        """
        Activate all particles of the entity in simulation, making them eligible for updates.
        """
        gs.logger.info(f"{self.__class__.__name__} <{self._uid}> activated.")
        self.set_active(gs.ACTIVE)

    def deactivate(self):
        """
        Deactivate all particles of the entity in simulation, stopping them from receiving for updates.
        """
        gs.logger.info(f"{self.__class__.__name__} <{self._uid}> deactivated.")
        self.set_active(gs.INACTIVE)

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None):
        """
        Set the velocity of some particles.

        Parameters
        ----------
        actives: torch.Tensor, shape (M, N, 3)
            Activeness boolean flags for each particle.
        particles_idx_local : torch.Tensor, shape (M, N)
            Index of the particles relative to this entity. If None, all particles will be considered. Defaults to None.
        envs_idx : torch.Tensor, shape (M,)
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        raise NotImplementedError

    def get_particles_active(self, envs_idx=None):
        """
        Retrieve current particle activeness boolean flags from the solver.

        Parameters
        ----------
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        poss : torch.Tensor, shape (M, n_particles, 3)
            Tensor of particle activeness boolean flags.
        """
        raise NotImplementedError

    @gs.assert_built
    def get_mass(self, envs_idx=None):
        """
        Return the total mass of the entity.

        Parameters
        ----------
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to query. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        mass : torch.Tensor, shape (M,)
            The computed total mass.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        mass = torch.zeros((len(envs_idx),), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_mass(self._particle_start, self.n_particles, mass, envs_idx)
        return mass

    # ------------------------------------------------------------------------------------
    # -------------------------------------- utils ---------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def find_closest_particle(self, pos, envs_idx=None):
        """
        Find the index of the particle closest to a given position.

        Parameters
        ----------
        pos : array_like, shape ([M,] 3)
            The target position to compare against.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        closest_idx : torch.Tensor, shape (M,)
            The index of the closest particle.
        """
        pos = to_gs_tensor(pos, dtype=gs.tc_float)
        if pos.ndim == 1:
            pos = pos.reshape((1, 3)).expand((self._scene._B, 3))
        if pos.shape != (self._scene._B, 3):
            gs.raise_exception(f"Invalid tensor shape {pos.shape} (expected {pos}).")

        cur_particles = self.get_particles_pos(envs_idx)
        distances = torch.linalg.norm(cur_particles - pos[:, None], dim=-1)
        closest_idx = torch.argmin(distances, dim=-1).to(dtype=gs.tc_int)
        if self._scene.n_envs == 0:
            closest_idx = closest_idx[0]
        return closest_idx

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        morph = self._morph

        if isinstance(morph, gs.morphs.Box):
            return "box"
        if isinstance(morph, gs.morphs.Sphere):
            return "sphere"
        if isinstance(morph, gs.morphs.Cylinder):
            return "cylinder"
        if isinstance(morph, gs.morphs.Mesh):
            return Path(morph.file).stem
        if isinstance(morph, gs.morphs.Nowhere):
            return "emitter"
        return "particle"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """Unique identifier for the entity."""
        return self._uid

    @property
    def idx(self):
        """Index of the entity within the simulation."""
        return self._idx

    @property
    def morph(self):
        """Morphological representation used for particle sampling."""
        return self._morph

    @property
    def vmesh(self):
        """Visual mesh used for skinning and rendering."""
        return self._vmesh

    @property
    def n_vverts(self):
        """Number of visual mesh vertices."""
        return len(self._vverts)

    @property
    def n_vfaces(self):
        """Number of visual mesh faces."""
        return len(self._vfaces)

    @property
    def n_particles(self):
        """Number of particles"""
        return self._n_particles

    @property
    def particle_start(self):
        """Starting index of the entity's particles in the global buffer."""
        return self._particle_start

    @property
    def particle_end(self):
        """Ending index (exclusive) of the entity's particles."""
        return self._particle_start + self._n_particles

    @property
    def vvert_start(self):
        """Starting index for visual mesh vertices."""
        return self._vvert_start

    @property
    def vvert_end(self):
        """Ending index (exclusive) for visual mesh vertices."""
        return self._vvert_start + self.n_vverts

    @property
    def vface_start(self):
        """Starting index for visual mesh faces."""
        return self._vface_start

    @property
    def vface_end(self):
        """Ending index (exclusive) for visual mesh faces."""
        return self._vface_start + self.n_vfaces

    @property
    def particle_size(self):
        """Diameter of individual particles."""
        return self._particle_size

    @property
    def init_particles(self):
        """Initial sampled particle positions."""
        return self._particles

    @property
    def material(self):
        """Material of this entity."""
        return self._material

    @property
    def surface(self):
        """Surface for rendering."""
        return self._surface
