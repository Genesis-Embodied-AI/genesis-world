from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.particle_entity import ParticleEntity, ParticleEntityDescription, load_morph_vmesh
from genesis.engine.render_context import DynamicTextureMixin
from genesis.options.surfaces import Surface
from genesis.typing import IndexType

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.pbd_solver import PBDSolver


@dataclass(kw_only=True)
class PBDMeshDescription(ParticleEntityDescription):
    """Describe one PBD entity simulated on a remeshed mesh as its resolution left it: the mesh and its edges.

    The rest lengths of the edges are taken on the particles, so an entity created from this remeshes nothing and reads
    no asset. The visual mesh stands at the pose of the morph, and 'mesh' is its copy remeshed at the particle size.
    """

    mesh: "gs.Mesh"
    mass: float
    edges: np.ndarray
    edges_len_rest: np.ndarray


@dataclass(kw_only=True)
class PBDClothDescription(PBDMeshDescription):
    """Describe one PBD cloth entity: its particles are the vertices of the remeshed mesh, bent across inner edges.

    Each inner edge joins the two vertices of an edge shared by two faces, then the two vertices facing it, and its rest
    length is the distance between the last two.
    """

    inner_edges: np.ndarray
    inner_edges_len_rest: np.ndarray

    @classmethod
    def resolve(cls, spec: "gs.entities.PBD", surface: Surface, particle_size: float) -> Self:
        """Remesh the morph of a cloth entity spec at the pose of the morph, and derive its edges and inner edges.

        'surface' is the one resolved from the spec, and the mesh is remeshed with edges 'particle_size' long.
        """
        vmesh, pos, quat = load_morph_vmesh(spec.morphs[0], surface)
        vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
        if vmesh.area < 1e-6:
            gs.raise_exception("Input mesh has zero surface area.")
        mesh = vmesh.copy()
        mesh.remesh(edge_len_abs=particle_size, fix=False)
        particles = np.asarray(mesh.verts, dtype=gs.np_float)
        edges = np.asarray(mesh.get_unique_edges(), dtype=gs.np_int)
        # Inner edges are two diagonal edges of each quadrilateral formed by adjacent face pairs
        adjacency, adjacent_edges = trimesh.graph.face_adjacency(mesh=mesh.trimesh, return_edges=True)
        v3 = np.sum(mesh.faces[adjacency[:, 0]], axis=1) - adjacent_edges[:, 0] - adjacent_edges[:, 1]
        v4 = np.sum(mesh.faces[adjacency[:, 1]], axis=1) - adjacent_edges[:, 0] - adjacent_edges[:, 1]
        inner_edges = np.stack([adjacent_edges[:, 0], adjacent_edges[:, 1], v3, v4], axis=1, dtype=gs.np_int)
        return cls(
            spec,
            vmesh.surface,
            particles,
            pos,
            vmesh=vmesh,
            mesh=mesh,
            mass=vmesh.area * spec.material.rho,
            edges=edges,
            edges_len_rest=np.linalg.norm(particles[edges[:, 0]] - particles[edges[:, 1]], axis=1),
            inner_edges=inner_edges,
            inner_edges_len_rest=np.linalg.norm(particles[inner_edges[:, 2]] - particles[inner_edges[:, 3]], axis=1),
        )


@dataclass(kw_only=True)
class PBDElasticDescription(PBDMeshDescription):
    """Describe one PBD elastic entity: its particles are the vertices of the tetrahedralized remeshed mesh."""

    elems: np.ndarray
    elems_vol_rest: np.ndarray

    @classmethod
    def resolve(cls, spec: "gs.entities.PBD", surface: Surface, particle_size: float) -> Self:
        """Remesh and tetrahedralize the morph of an elastic entity spec at the pose of the morph.

        'surface' is the one resolved from the spec, and the mesh is remeshed with edges 'particle_size' long. The
        edges are those of the tetrahedra, each listed once.
        """
        morph = spec.morphs[0]
        vmesh, pos, quat = load_morph_vmesh(morph, surface)
        vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
        if vmesh.volume < 1e-6:
            gs.raise_exception("Input mesh has zero volume.")
        mesh = vmesh.copy()
        mesh.remesh(edge_len_abs=particle_size, fix=True)
        particles, elems, *_ = mesh.tetrahedralize(mu.generate_tetgen_config_from_morph(morph))
        particles = particles.astype(gs.np_float, copy=False)
        elems = elems.astype(gs.np_int, copy=False)
        # The six edges of each tetrahedron, sorted so that an edge shared by several elements appears once
        elems_edges = elems[:, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))].reshape((-1, 2))
        edges = np.unique(np.sort(elems_edges, axis=1), axis=0)
        return cls(
            spec,
            vmesh.surface,
            particles,
            pos,
            vmesh=vmesh,
            mesh=mesh,
            mass=vmesh.volume * spec.material.rho,
            edges=edges,
            edges_len_rest=np.linalg.norm(particles[edges[:, 0]] - particles[edges[:, 1]], axis=1),
            elems=elems,
            elems_vol_rest=np.linalg.det(particles[elems[:, 1:]] - particles[elems[:, :1]]) / 6.0,
        )


class PBDBaseEntity(ParticleEntity):
    """
    Base class for PBD entity.
    """

    @gs.assert_built
    def fix_particles_to_link(self, link_idx, particles_idx_local=None, envs_idx=None):
        """
        Attach some particles to a rigid link, which carries them along until they are released.

        Parameters
        ----------
        link_idx : int
            Global index of the rigid link.
        particles_idx_local : int | array_like, shape ([M,] N)
            Index of the particles relative to this entity.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        """
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        self._solver.fix_particles_to_link(link_idx, particles_idx, envs_idx)

    @gs.assert_built
    def fix_particles(self, particles_idx_local=None, envs_idx=None, zero_velocity=True):
        """
        Fix the position of some particles in the simulation.

        Parameters
        ----------
        particles_idx_local : int | array_like, shape ([M,] N)
            Index of the particles relative to this entity.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of the particles. Defaults to True.
        """
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        if zero_velocity:
            self._solver.set_particles_vel(0.0, particles_idx, envs_idx)
        self._solver.fix_particles(particles_idx, envs_idx)

    @gs.assert_built
    def release_particles(
        self,
        particles_idx_local: IndexType | None = None,
        envs_idx: IndexType | None = None,
    ) -> None:
        """
        Release some of the attached particles, allowing them to move freely again.

        Parameters
        ----------
        particles_idx_local : int | array_like, shape ([M,] N)
            Index of the particles relative to this entity.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        """
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        self._solver.release_particles(particles_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        return f"pbd_{super()._get_morph_identifier()}"


@qd.data_oriented
class PBDTetEntity(DynamicTextureMixin, PBDBaseEntity):
    """
    PBD entity represented by tetrahedral elements.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : PBDMeshDescription
        The description the entity is created from, holding its spec, its surface, its mesh and its constraints.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    """

    def __init__(
        self,
        scene: "Scene",
        solver: "PBDSolver",
        desc: PBDMeshDescription,
        particle_size: float,
        idx: int,
        particle_start: int,
        edge_start: int,
    ) -> None:
        super().__init__(scene, solver, desc, particle_size, idx, particle_start)
        self._edge_start = edge_start
        self._mesh = desc.mesh
        self._mass = desc.mass
        self._particle_mass = desc.mass / len(self._particles)
        self._edges = desc.edges
        self._edges_len_rest = desc.edges_len_rest

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._particle_start,
            is_active=True,
            material_type=self._material_type,
            mass=float(self._particle_mass),
            rho_rest=0.0,
            mu_s=self.material.static_friction,
            mu_k=self.material.kinetic_friction,
            air_resistance=self.material.air_resistance if isinstance(self.material, gs.materials.PBD.Cloth) else 0.0,
            density_relaxation=0.0,
            viscosity_relaxation=0.0,
            particles=self._particles,
        )
        self._solver.add_edges(
            self._edge_start,
            self._particle_start,
            self.material.stretch_compliance,
            self.material.stretch_relaxation,
            self._edges,
            self._edges_len_rest,
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def mesh(self):
        """Mesh."""
        return self._mesh

    @property
    def edges(self):
        """Edge array of the mesh."""
        return self._edges

    @property
    def n_edges(self):
        """Number of edges in the mesh."""
        return len(self._edges)


@qd.data_oriented
class PBD2DEntity(PBDTetEntity):
    """
    PBD entity represented by a 2D mesh.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : PBDClothDescription
        The description the entity is created from, holding its spec, its surface, its mesh and its constraints.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    inner_edge_start: int
        Starting index of this entity's inner edges in the global buffer.
    """

    _description_cls = PBDClothDescription

    def __init__(
        self,
        scene: "Scene",
        solver: "PBDSolver",
        desc: PBDClothDescription,
        particle_size: float,
        idx: int,
        particle_start: int,
        edge_start: int,
        inner_edge_start: int,
    ) -> None:
        super().__init__(scene, solver, desc, particle_size, idx, particle_start, edge_start)
        self._inner_edge_start = inner_edge_start
        self._inner_edges = desc.inner_edges
        self._inner_edges_len_rest = desc.inner_edges_len_rest
        self._material_type = int(self.solver.MATERIAL.CLOTH)

    def _add_particles_to_solver(self):
        super()._add_particles_to_solver()
        self._solver.add_inner_edges(
            self._inner_edge_start,
            self._particle_start,
            self.material.bending_compliance,
            self.material.bending_relaxation,
            self._inner_edges,
            self._inner_edges_len_rest,
        )

    @property
    def n_inner_edges(self):
        """The number of inner edges in the 2D mesh."""
        return len(self._inner_edges)


@qd.data_oriented
class PBD3DEntity(PBDTetEntity):
    """
    PBD entity represented by a 3D mesh.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : PBDElasticDescription
        The description the entity is created from, holding its spec, its surface, its mesh and its constraints.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    elem_start: int
        Starting index of this entity's element in the global buffer.
    """

    _description_cls = PBDElasticDescription

    def __init__(
        self,
        scene: "Scene",
        solver: "PBDSolver",
        desc: PBDElasticDescription,
        particle_size: float,
        idx: int,
        particle_start: int,
        edge_start: int,
        elem_start: int,
    ) -> None:
        super().__init__(scene, solver, desc, particle_size, idx, particle_start, edge_start)
        self._elem_start = elem_start
        self._elems = desc.elems
        self._elems_vol_rest = desc.elems_vol_rest
        self._material_type = int(self.solver.MATERIAL.ELASTIC)

    def _add_particles_to_solver(self):
        super()._add_particles_to_solver()
        self._solver.add_elems(
            self._elem_start,
            self._particle_start,
            self.material.volume_compliance,
            self.material.volume_relaxation,
            self._elems,
            self._elems_vol_rest,
        )

    @property
    def n_elems(self):
        """The number of tetrahedral elements in the mesh."""
        return len(self._elems)

    @property
    def elem_start(self):
        """The starting index of the elements in the global solver."""
        return self._elem_start

    @property
    def elem_end(self):
        """The ending index of the elements in the global solver."""
        return self._elem_start + self.n_elems


@qd.data_oriented
class PBDParticleEntity(PBDBaseEntity):
    """
    PBD entity represented solely by particles.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : ParticleEntityDescription
        The description the entity is created from, holding its spec, its surface and its particles.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    """

    _description_cls = ParticleEntityDescription

    def __init__(
        self,
        scene: "Scene",
        solver: "PBDSolver",
        desc: ParticleEntityDescription,
        particle_size: float,
        idx: int,
        particle_start: int,
    ) -> None:
        super().__init__(scene, solver, desc, particle_size, idx, particle_start, need_skinning=False)

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._particle_start,
            is_active=self.active,
            material_type=int(self._solver.MATERIAL.LIQUID),
            mass=self.material.rho,
            rho_rest=self.material.rho,
            mu_s=0.0,
            mu_k=0.0,
            air_resistance=0.0,
            density_relaxation=self.material.density_relaxation,
            viscosity_relaxation=self.material.viscosity_relaxation,
            particles=self._particles,
        )

    @property
    def n_fluid_particles(self):
        """The number of fluid particles."""
        return self.n_particles


@qd.data_oriented
class PBDFreeParticleEntity(PBDBaseEntity):
    """
    PBD-based entity represented by non-physics particles

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : ParticleEntityDescription
        The description the entity is created from, holding its spec, its surface and its particles.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    """

    _description_cls = ParticleEntityDescription

    def __init__(
        self,
        scene: "Scene",
        solver: "PBDSolver",
        desc: ParticleEntityDescription,
        particle_size: float,
        idx: int,
        particle_start: int,
    ) -> None:
        super().__init__(scene, solver, desc, particle_size, idx, particle_start, need_skinning=False)

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._particle_start,
            is_active=self.active,
            material_type=int(self._solver.MATERIAL.PARTICLE),
            mass=self.material.rho,
            rho_rest=self.material.rho,
            mu_s=0.0,
            mu_k=0.0,
            air_resistance=0.0,
            density_relaxation=0.0,
            viscosity_relaxation=0.0,
            particles=self._particles,
        )
