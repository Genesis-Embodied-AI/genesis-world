import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.particle_entity import ParticleEntity


class PBDBaseEntity(ParticleEntity):
    """
    Base class for PBD entity.
    """

    def _add_to_solver(self):
        super()._add_to_solver()

        # Get UVs from vmesh (may be None if no texture)
        uvs = self._vmesh.uvs if self._vmesh is not None else None
        if uvs is None or len(uvs) != self.n_vverts:
            # No UVs available, pass empty array (kernel will skip copy)
            uvs = np.zeros((0, 2), dtype=gs.np_float)

        self._solver.add_uvs_and_faces(self._vvert_start, self._vface_start, uvs, self._vfaces)

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
    def release_particle(self, particles_idx_local=None, envs_idx=None):
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
class PBDTetEntity(PBDBaseEntity):
    """
    PBD entity represented by tetrahedral elements.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
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
        edge_start,
        vvert_start,
        vface_start,
        name: str | None = None,
    ):
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            vvert_start,
            vface_start,
            name=name,
        )
        self._edge_start = edge_start

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self._particle_start,
            active=True,
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

    def sample(self):
        """
        Sample and preprocess the mesh for the PBD tetrahedral entity.

        Applies transformation from the morph, stores mesh vertices and faces, and performs remeshing based on the
        particle size.
        """
        # We don't use ParticleEntity.sample() because we need to maintain the remeshed self._mesh as well. The morph
        # pose offset (e.g. an up-axis conversion) is composed onto the morph pose.
        pos, quat = gu.transform_pos_quat_by_trans_quat(
            np.array(self._morph.offset_pos, dtype=gs.np_float),
            np.array(self._morph.offset_quat, dtype=gs.np_float),
            np.array(self._morph.pos, dtype=gs.np_float),
            np.array(self._morph.quat, dtype=gs.np_float),
        )
        # Composed world placement, used as the origin for the particle offsets in the subclasses' 'sample'.
        self._sampled_pos = pos
        self._vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
        self._vverts = np.asarray(self._vmesh.verts, dtype=gs.np_float)
        self._vfaces = np.asarray(self._vmesh.faces, dtype=gs.np_int)

        self._mesh = self._vmesh.copy()
        self._mesh.remesh(edge_len_abs=self.particle_size, fix=isinstance(self, PBD3DEntity))

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
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
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
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
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
        edge_start,
        inner_edge_start,
        vvert_start,
        vface_start,
        name: str | None = None,
    ):
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            edge_start,
            vvert_start,
            vface_start,
            name=name,
        )

        self._inner_edge_start = inner_edge_start
        self._material_type = int(self.solver.MATERIAL.CLOTH)

    def sample(self):
        """Sample and preprocess the 2D mesh for the PBD cloth-like entity."""
        super().sample()

        if self._vmesh.area < 1e-6:
            gs.raise_exception("Input mesh has zero surface area.")
        self._mass = self._vmesh.area * self.material.rho

        self._particles = np.asarray(self._mesh.verts, dtype=gs.np_float)
        self._init_particles_offset = torch.tensor(
            self._particles - self._sampled_pos, dtype=gs.tc_float, device=gs.device
        )

        self._edges = np.asarray(self._mesh.get_unique_edges(), dtype=gs.np_int)

        self._particle_mass = self._mass / len(self._particles)

        # Inner edges are two diagonal edges of each quadrilateral formed by adjacent face pairs
        adjacency, inner_edges = trimesh.graph.face_adjacency(mesh=self._mesh.trimesh, return_edges=True)
        v3 = np.sum(self._mesh.faces[adjacency[:, 0]], axis=1) - inner_edges[:, 0] - inner_edges[:, 1]
        v4 = np.sum(self._mesh.faces[adjacency[:, 1]], axis=1) - inner_edges[:, 0] - inner_edges[:, 1]
        self._inner_edges = np.stack([inner_edges[:, 0], inner_edges[:, 1], v3, v4], axis=1, dtype=gs.np_int)

        self._edges_len_rest = np.linalg.norm(
            self._particles[self._edges[:, 0]] - self._particles[self._edges[:, 1]], axis=1
        )
        self._inner_edges_len_rest = np.linalg.norm(
            self._particles[self._inner_edges[:, 2]] - self._particles[self._inner_edges[:, 3]], axis=1
        )
        self._n_particles = len(self._particles)

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
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
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
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
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
        edge_start,
        elem_start,
        vvert_start,
        vface_start,
        name: str | None = None,
    ):
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            edge_start,
            vvert_start,
            vface_start,
            name=name,
        )

        self._elem_start = elem_start

        self._material_type = int(self.solver.MATERIAL.ELASTIC)

    def sample(self):
        super().sample()

        if self._vmesh.volume < 1e-6:
            gs.raise_exception("Input mesh has zero volume.")
        self._mass = self._vmesh.volume * self.material.rho

        tet_cfg = mu.generate_tetgen_config_from_morph(self.morph)
        particles, elems, *_ = self._mesh.tetrahedralize(tet_cfg)
        self._particles = particles.astype(gs.np_float, copy=False)
        self._init_particles_offset = torch.tensor(
            self._particles - self._sampled_pos, dtype=gs.tc_float, device=gs.device
        )

        self._elems = elems.astype(gs.np_int, copy=False)
        self._edges = np.array(
            list(
                set(
                    tuple(sorted((self._elems[i, j], self._elems[i, k])))
                    for i in range(len(self._elems))
                    for j in range(4)
                    for k in range(j + 1, 4)
                )
            ),
            dtype=gs.np_int,
        )
        self._particle_mass = self._mass / len(self._particles)

        self._edges_len_rest = np.linalg.norm(
            self._particles[self._edges[:, 0]] - self._particles[self._edges[:, 1]], axis=1
        )
        self._elems_vol_rest = (
            np.linalg.det(self._particles[self._elems[:, 1:]] - self._particles[self._elems[:, :1]]) / 6.0
        )
        self._n_particles = len(self._particles)

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
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    """

    def __init__(
        self, scene, solver, material, morph, surface, particle_size, idx, particle_start, name: str | None = None
    ):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False, name=name
        )

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self._particle_start,
            active=self.active,
            material_type=int(self._solver.MATERIAL.LIQUID),
            mass=self._material.rho,
            rho_rest=self._material.rho,
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
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    """

    def __init__(
        self, scene, solver, material, morph, surface, particle_size, idx, particle_start, name: str | None = None
    ):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False, name=name
        )

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self._particle_start,
            active=self.active,
            material_type=int(self._solver.MATERIAL.PARTICLE),
            mass=self._material.rho,
            rho_rest=self._material.rho,
            mu_s=0.0,
            mu_k=0.0,
            air_resistance=0.0,
            density_relaxation=0.0,
            viscosity_relaxation=0.0,
            particles=self._particles,
        )
