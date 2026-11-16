import functools
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.materials.base import SAMPLER_PATTERN
from genesis.options.surfaces import Surface
from genesis.utils.misc import broadcast_tensor

from .base_entity import Entity, EntityDescription

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.particle_solver import ParticleSolver


def check_particles_inside_boundary(
    particles_pos: np.ndarray, boundary_lower: np.ndarray, boundary_upper: np.ndarray
) -> None:
    """Raise if any of the given particle positions lies outside the boundary of a particle solver."""
    is_inside = (particles_pos > boundary_lower) & (particles_pos < boundary_upper)
    if not is_inside.all():
        gs.raise_exception(
            textwrap.dedent(
                f"""\
                Particles outside the solver boundary. Note that for MPMSolver, boundary is slightly tighter than the
                specified domain due to safety padding.

                Current boundary:
                lower: {boundary_lower}
                upper: {boundary_upper}

                Particles to be added:
                min: {particles_pos.min(0)}
                max: {particles_pos.max(0)}"""
            )
        )


def resolve_sampler(sampler: str) -> str:
    """Check the name of a particle sampler, completing 'pbs' with the default resolution of its distance field."""
    # The material validates its sampler on construction alone, which an assignment or a copy bypasses
    if not isinstance(sampler, str) or not SAMPLER_PATTERN.fullmatch(sampler):
        gs.raise_exception(
            "The sampler should be 'pbs', 'pbs-<res>' with a positive resolution, 'random', or 'regular'. "
            f"Got: {sampler!r}."
        )
    return "pbs-32" if sampler == "pbs" else sampler


def assert_active(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.active:
            gs.raise_exception(f"'{self.__class__.__name__}' is not active. Call `entity.activate()` first.")
        return method(self, *args, **kwargs)

    return wrapper


@dataclass
class ParticleEntityDescription(EntityDescription):
    """Describe one particle entity as its resolution left it: its sampled particles and its posed visual mesh.

    The particles are sampled from the morph of the spec, possibly at random, so an entity created from this samples
    nothing and reads no asset. The surface is the one the visual mesh resolved, and 'origin' is the point that the
    initial offset of each particle is taken from (see 'ParticleEntity.set_position').
    """

    surface: Surface
    init_particles: np.ndarray
    origin: np.ndarray
    vmesh: "gs.Mesh | None" = None
    mesh_set_group_ids: np.ndarray | None = None

    @classmethod
    def resolve(
        cls,
        spec: "gs.entities.MPM | gs.entities.SPH | gs.entities.PBD",
        surface: Surface,
        particle_size: float,
        boundary_lower: np.ndarray,
        boundary_upper: np.ndarray,
    ) -> Self:
        """Sample the particles of an entity spec, and place them and the visual mesh at the pose of its morph.

        'surface' is the one resolved from the spec, and the particles are sampled 'particle_size' apart. A particle
        outside the box from 'boundary_lower' to 'boundary_upper', the domain of the solver, raises.
        """
        morph = spec.morphs[0]
        sampler = resolve_sampler(spec.material.sampler)
        mesh_set_group_ids = None
        if isinstance(morph, gs.options.morphs.MeshSet):
            vmeshes = gs.Mesh.from_morph_surface(morph, surface)
            particles_per_mesh = []
            for vmesh_i, pos_i, euler_i in zip(vmeshes, morph.poss, morph.eulers, strict=True):
                trans_i = np.array(pos_i, dtype=gs.np_float)
                quat_i = gu.xyz_to_quat(np.array(euler_i, dtype=gs.np_float), rpy=True, degrees=True)
                # Sample before posing the mesh, so the lattice and the sampling cache depend on the member shape alone
                particles_i = vmesh_i.particlize(particle_size, sampler)
                particles_per_mesh.append(gu.transform_by_trans_quat(particles_i, trans_i, quat_i))
                vmesh_i.apply_transform(gu.trans_quat_to_T(trans_i, quat_i))
            mesh_set_group_ids = np.concatenate(
                [np.full((len(v),), fill_value=i, dtype=gs.np_int) for i, v in enumerate(particles_per_mesh)]
            )
            particles = np.concatenate(particles_per_mesh, dtype=gs.np_float)
            combined_verts, combined_faces = trimesh.util.append_faces(
                [v.verts for v in vmeshes], [v.faces for v in vmeshes]
            )
            combined_vert_normals = np.concatenate([v.trimesh.vertex_normals for v in vmeshes])
            combined_tmesh = trimesh.Trimesh(
                vertices=combined_verts, faces=combined_faces, vertex_normals=combined_vert_normals
            )
            surface = vmeshes[0].surface
            vmesh = mu.trimesh_to_mesh(combined_tmesh, 1, surface)
            origin = np.mean(morph.poss, axis=0, dtype=gs.np_float)
        elif isinstance(morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            meshes = gs.Mesh.from_morph_surface(morph, surface)
            if len(meshes) > 1:
                gs.raise_exception("Mesh file with multiple sub-meshes are not supported.")
            vmesh = meshes[0]
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
            surface.update_texture()
            vmesh = None
            particles = np.tile(gu.nowhere(), (morph.n_particles, 1))
            origin = gu.nowhere()
        else:
            gs.raise_exception(f"Unsupported morph: {morph}.")

        if particles.size == 0:
            gs.raise_exception("Entity has zero particles.")
        if not isinstance(morph, gs.options.morphs.Nowhere):
            check_particles_inside_boundary(particles, boundary_lower, boundary_upper)

        gs.logger.info(f"Sampled ~<{len(particles):,}>~ particles.")
        return cls(
            spec,
            surface,
            np.asarray(particles, dtype=gs.np_float, order="C"),
            origin,
            vmesh=vmesh,
            mesh_set_group_ids=mesh_set_group_ids,
        )


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
        The description the entity is created from, holding its spec, its surface and its sampled particles.
    particle_size : float
        Size of each particle, determining resolution and sampling density.
    idx : int
        Index of this entity in the simulation.
    particle_start : int
        Global index offset for this entity's particles in the solver.
    need_skinning : bool, default=True
        Whether to enable skinning for rendering this entity's mesh.
    """

    def __init__(
        self,
        scene: "Scene",
        solver: "ParticleSolver",
        desc: ParticleEntityDescription,
        particle_size: float,
        idx: int,
        particle_start: int,
        need_skinning: bool = True,
    ) -> None:
        super().__init__(idx, scene, solver, desc.surface, desc.spec)
        # The scene names an entity as it is added, so the description takes the spec recording that name
        desc.spec = self._spec
        self._desc = desc

        self._particle_size = particle_size
        self._particle_start = particle_start

        self._need_skinning = need_skinning

        self.sampler = resolve_sampler(self.material.sampler)
        self._vmesh = desc.vmesh
        if need_skinning and desc.vmesh is not None:
            self._vverts = np.asarray(desc.vmesh.verts, dtype=gs.np_float)
            self._vfaces = np.asarray(desc.vmesh.faces, dtype=gs.np_int)
        else:
            self._vverts = np.zeros((0, 3), dtype=gs.np_float)
            self._vfaces = np.zeros((0, 3), dtype=gs.np_int)
        self.mesh_set_group_ids = desc.mesh_set_group_ids
        self._particles = desc.init_particles
        self._init_particles_offset = torch.tensor(self._particles - desc.origin, dtype=gs.tc_float, device=gs.device)
        self._n_particles = len(self._particles)

        # Note that this attribute must only be used in forward pass
        self.active = False

    @property
    def desc(self) -> ParticleEntityDescription:
        """The description this entity was created from, holding its sampled particles and posed visual mesh."""
        return self._desc

    def _add_to_solver(self):
        # For nowhere entity, it will be activated when particles are later added.
        # Note that 'self.active' is not informative for nowhere entity, and such entity is not supported in
        # differentiable mode.
        self.active = not isinstance(self._morph, gs.options.morphs.Nowhere)

        self._add_particles_to_solver()

        self._is_built = True

    def _add_particles_to_solver(self):
        raise NotImplementedError

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
        pos = broadcast_tensor(pos, gs.tc_float, (self._solver.n_envs_, 3), ("envs_idx", ""))

        cur_particles = self.get_particles_pos(envs_idx)
        distances = torch.linalg.norm(cur_particles - pos[:, None], dim=-1)
        closest_idx = torch.argmin(distances, dim=-1).to(dtype=gs.tc_int)
        if self._solver.n_envs == 0:
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
    def particle_size(self):
        """Diameter of individual particles."""
        return self._particle_size

    @property
    def init_particles(self):
        """Initial sampled particle positions."""
        return self._particles

    @property
    def surface(self):
        """Surface for rendering."""
        return self._surface
