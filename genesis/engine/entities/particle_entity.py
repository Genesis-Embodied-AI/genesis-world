import functools
from pathlib import Path

import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.utils.misc import broadcast_tensor

from .base_entity import Entity

# Number of vvert-particle distances computed at once when searching the supports of the vverts. Their difference
# vectors take 24 MiB in double precision, while chunks of this size still amortize the per-chunk overhead of numpy.
VVERTS_DIST2_MAX_SIZE = 2**20


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
    material : Material
        The material definition, including sampling strategy and physical properties.
    morph : Morph
        Geometry or volumetric shape used for sampling particles (e.g., mesh, primitive).
    surface : Surface
        Surface material or texture information associated with the entity.
    particle_size : float
        Size of each particle, determining resolution and sampling density.
    idx : int
        Index of this entity in the simulation.
    particle_start : int
        Global index offset for this entity's particles in the solver.
    vvert_start : int, optional
        Global index offset for vertex-based rendering, used for skinning.
    vface_start : int, optional
        Global index offset for visual faces, used for skinning.
    need_skinning : bool, default=True
        Whether to enable skinning for rendering this entity's mesh.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        particle_size,
        idx,
        particle_start,
        vvert_start=None,
        vface_start=None,
        need_skinning=True,
        name: str | None = None,
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        self._particle_size = particle_size
        self._particle_start = particle_start

        self._need_skinning = need_skinning
        if need_skinning:
            self._vvert_start = vvert_start
            self._vface_start = vface_start
        else:
            self._vvert_start = -1
            self._vface_start = -1

        # visual mesh
        if isinstance(self._morph, gs.options.morphs.MeshSet):
            self._vmesh = gs.Mesh.from_morph_surface(self.morph, self.surface)
            self._surface = self._vmesh[0].surface

        elif isinstance(self._morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            meshes = gs.Mesh.from_morph_surface(self.morph, self.surface)
            if len(meshes) > 1:
                gs.raise_exception("Mesh file with multiple sub-meshes are not supported.")
            self._vmesh = meshes[0]
            self._surface = self._vmesh.surface

        else:
            surface.update_texture()
            self._vmesh = None

        self.sample()

        # Note that this attribute must only be used in forward pass
        self.active = False

    def init_sampler(self):
        """
        Initialize the particle sampling strategy based on the material's sampler field.

        Raises
        ------
        GenesisException
            If the specified sampler is not supported or incorrectly formatted.
        """
        self.sampler = self._material.sampler

        match self.sampler.split("-"):
            case ["regular"]:
                pass
            case ["random"]:
                pass
            case ["pbs"]:
                # using default sdf_res=32
                self.sampler += "-32"
            case ["pbs", num] if num.isnumeric():
                pass
            case _:
                gs.raise_exception(
                    "Only one of the following samplers is supported: [`regular`, `random`, `pbs`, `pbs-sdf_res`]. "
                    f"Got: {self.sampler}."
                )

    def _add_to_solver(self):
        # For nowhere entity, it will be activated when particles are later added.
        # Note that 'self.active' is not informative for nowhere entity, and such entity is not supported in
        # differentiable mode.
        self.active = not isinstance(self._morph, gs.options.morphs.Nowhere)

        self._add_particles_to_solver()
        if self._need_skinning:
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

        self._solver.add_vverts(
            self._vvert_start, self._particle_start, self._vverts, self._particles, P_invs, support_idxs
        )

    def sample(self):
        """
        Sample particles from the morph based on its type and the specified sampler.

        Raises
        ------
        GenesisException
            If no particles are sampled, or sampled particles lie outside the solver's domain.
        """
        self.init_sampler()

        if isinstance(self._morph, gs.options.morphs.MeshSet):
            particles = []
            for vmesh_i, pos_i, euler_i in zip(self._vmesh, self._morph.poss, self._morph.eulers, strict=True):
                trans_i = np.array(pos_i, dtype=gs.np_float)
                quat_i = gu.xyz_to_quat(np.array(euler_i, dtype=gs.np_float), rpy=True, degrees=True)
                # Sample before posing the mesh, so the lattice and the sampling cache depend on the member shape alone
                particles_i = vmesh_i.particlize(self._particle_size, self.sampler)
                particles.append(gu.transform_by_trans_quat(particles_i, trans_i, quat_i))
                vmesh_i.apply_transform(gu.trans_quat_to_T(trans_i, quat_i))
        elif isinstance(self._morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            particles = self._vmesh.particlize(self._particle_size, self.sampler)
            particles = particles.astype(gs.np_float, order="C", copy=False)
        elif isinstance(self._morph, gs.options.morphs.Nowhere):
            particles = pu.nowhere_particles(self._morph.n_particles)
        else:
            gs.raise_exception(f"Unsupported morph: {self._morph}.")

        if not isinstance(self._morph, gs.options.morphs.MeshSet) and particles.size == 0:
            gs.raise_exception("Entity has zero particles.")

        if isinstance(self._morph, gs.options.morphs.Nowhere):
            self._vverts = np.zeros((0, 3), dtype=gs.np_float)
            self._vfaces = np.zeros((0, 3), dtype=gs.np_int)
            origin = gu.nowhere()
        elif isinstance(self._morph, gs.options.morphs.MeshSet):
            self.mesh_set_group_ids = np.concatenate(
                [np.full((len(v),), fill_value=i, dtype=gs.np_int) for i, v in enumerate(particles)]
            )
            particles = np.concatenate(particles, dtype=gs.np_float)
            if not self._solver.boundary.is_inside(particles):
                gs.raise_exception(
                    "Entity has particles outside solver boundary. Note that for MPMSolver, boundary is slightly "
                    "tighter than the specified domain due to safety padding.\n\nCurrent boundary:\n"
                    f"{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\nmax: {particles.max(0)}\n"
                )

            combined_verts, combined_faces = trimesh.util.append_faces(
                [v.verts for v in self._vmesh],
                [v.faces for v in self._vmesh],
            )
            combined_vert_normals = np.concatenate([v.trimesh.vertex_normals for v in self._vmesh])
            combined_tmesh = trimesh.Trimesh(
                vertices=combined_verts,
                faces=combined_faces,
                vertex_normals=combined_vert_normals,
            )
            self._vmesh = mu.trimesh_to_mesh(combined_tmesh, 1, self._surface)

            if self._need_skinning:
                self._vverts = np.asarray(self._vmesh.verts, dtype=gs.np_float)
                self._vfaces = np.asarray(self._vmesh.faces, dtype=gs.np_int)
            else:
                self._vverts = np.zeros((0, 3), dtype=gs.np_float)
                self._vfaces = np.zeros((0, 3), dtype=gs.np_int)
            origin = np.mean(self._morph.poss, axis=0, dtype=gs.np_float)
        else:
            # transform vmesh (the morph pose offset, e.g. an up-axis conversion, is composed onto the morph pose)
            pos, quat = gu.transform_pos_quat_by_trans_quat(
                np.array(self._morph.offset_pos, dtype=gs.np_float),
                np.array(self._morph.offset_quat, dtype=gs.np_float),
                np.array(self._morph.pos, dtype=gs.np_float),
                np.array(self._morph.quat, dtype=gs.np_float),
            )
            self._vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
            # transform particles
            particles = gu.transform_by_trans_quat(particles, pos, quat)

            if not self._solver.boundary.is_inside(particles):
                gs.raise_exception(
                    "Entity has particles outside solver boundary. Note that for MPMSolver, boundary is slightly "
                    "tighter than the specified domain due to safety padding.\n\n"
                    f"Current boundary:\n{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\n"
                    f"max: {particles.max(0)}\n"
                )

            if self._need_skinning:
                self._vverts = np.asarray(self._vmesh.verts, dtype=gs.np_float)
                self._vfaces = np.asarray(self._vmesh.faces, dtype=gs.np_int)
            else:
                self._vverts = np.zeros((0, 3), dtype=gs.np_float)
                self._vfaces = np.zeros((0, 3), dtype=gs.np_int)
            origin = pos

        self._particles = np.asarray(particles, dtype=gs.np_float, order="C")
        self._init_particles_offset = torch.tensor(self._particles - origin, dtype=gs.tc_float, device=gs.device)
        self._n_particles = len(self._particles)

        gs.logger.info(f"Sampled ~<{self._n_particles:,}>~ particles.")

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    @assert_active
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
        # A value without one entry per particle places the center of mass, each particle keeping its initial offset
        shape = np.shape(value)
        if len(shape) == 1 or (len(shape) == 2 and shape[0] != self._n_particles):
            value = self._init_particles_offset + broadcast_tensor(value, gs.tc_float, (*shape[:-1], 1, 3))
        self.set_particles_pos(value, envs_idx=envs_idx)

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
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        self._solver.set_particles_pos(poss, particles_idx, envs_idx)

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
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        return self._solver.get_particles_pos(particles_idx, envs_idx)

    @assert_active
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
        self.set_particles_vel(vels, envs_idx=envs_idx)

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
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        self._solver.set_particles_vel(vels, particles_idx, envs_idx)

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
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        return self._solver.get_particles_vel(particles_idx, envs_idx)

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
        self.set_particles_active(actives, envs_idx=envs_idx)
        self.active = bool((torch.as_tensor(actives) == gs.ACTIVE).any())

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
        Set the activeness of some particles.

        Parameters
        ----------
        actives: torch.Tensor, shape (M, N, 3)
            Activeness boolean flags for each particle.
        particles_idx_local : torch.Tensor, shape (M, N)
            Index of the particles relative to this entity. If None, all particles will be considered. Defaults to None.
        envs_idx : torch.Tensor, shape (M,)
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        self._solver.set_particles_active(actives, particles_idx, envs_idx)

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
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        return self._solver.get_particles_active(particles_idx, envs_idx)

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
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        return torch.atleast_1d(self._solver.get_particles_mass(particles_idx, envs_idx).sum(dim=-1))

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
        pos = broadcast_tensor(pos, gs.tc_float, (self._scene._B, 3), ("envs_idx", ""))

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
