import itertools
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.particle as pu
from genesis.constants import STRUCTURE
from genesis.engine.render_context import (
    ArrayRef,
    DynamicMesh,
    Lines,
    ParticlesPhysics,
    Points,
    ScopedRenderContext,
    SegmentationKeys,
    Shape,
    ShapeRoles,
    select_view_shapes,
)
from genesis.engine.state_change import StateChange, mutates
from genesis.options.solvers import MPMOptions, PBDOptions, SPHOptions
from genesis.typing import UnitIntervalVec4Type
from genesis.utils.data import ENVS, Data, DataKind, Size, eoa, field
from genesis.utils.geom import (
    SpatialHashGrid,
    func_hash_cell_to_slot,
    func_hash_n_cells,
    func_hash_pos_to_grid,
    func_hash_wrap_cell,
)
from genesis.utils.misc import indices_to_mask, qd_to_torch, tensor_to_array

from .base_solver import GravityMixin, Solver, TimeBasedMixin, kernel_set_values

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator
    from genesis.options.surfaces import Surface


# The number of distances between visual vertices and particles computed at once when searching the supports of the
# visual vertices. Their difference vectors take 24 MiB in double precision, while chunks of this size still amortize
# the overhead of numpy per chunk.
VVERTS_DIST2_MAX_SIZE = 2**20

# The visual vertices skinned on the particles, the particles supporting each of them, the entities whose visual mesh
# may tear, and the edges of their meshes.
VVERTS = Size("vverts")
VVERT_SUPPORTS = Size("vvert_supports")
SKINNED_ENTITIES = Size("skinned_entities")
SKINNED_EDGES = Size("skinned_edges")

# The stretch of an edge of a skinned visual mesh, relative to its rest length, past which the particles no longer
# follow the topology of the mesh, such as an elastic body torn apart.
SKINNED_EDGE_STRETCH_MAX = 3.0


class SkinnedMeshes(Data):
    """The visual meshes skinned on the particles of a solver, each vertex the weighted sum of the positions of its
    support particles."""

    support_idxs: qd.Tensor = field((int, VVERT_SUPPORTS), shape=(VVERTS,), kind=DataKind.CONSTANT)
    support_weights: qd.Tensor = field((float, VVERT_SUPPORTS), shape=(VVERTS,), kind=DataKind.CONSTANT)
    pos: qd.Tensor = field((float, 3), shape=(VVERTS, ENVS), kind=DataKind.DERIVED)
    STRETCH_MAX: qd.Tensor = field(float, shape=(), kind=DataKind.INFO)
    # The two vertices of each edge of the meshes of the tearable entities, its rest length, and its entity among them
    edges_vverts_idx: qd.Tensor = field((int, 2), shape=(SKINNED_EDGES,), kind=DataKind.CONSTANT)
    edges_rest_length: qd.Tensor = field(float, shape=(SKINNED_EDGES,), kind=DataKind.CONSTANT)
    edges_entity_idx: qd.Tensor = field(int, shape=(SKINNED_EDGES,), kind=DataKind.CONSTANT)
    # Whether the mesh of each tearable entity holds in each environment, cleared by the refresh it tears at, until a
    # reset of the scene.
    entities_is_intact: qd.Tensor = field(bool, shape=(SKINNED_ENTITIES, ENVS), kind=DataKind.STATE)


class ParticleSolver(GravityMixin, TimeBasedMixin, Solver):
    """Base class of the particle-based solvers, holding what they share: the accessors of the state of the particles.

    The accessors take global particle indices, given once for every environment or one row per environment (see
    '_sanitize_io_variables'), and the kernels of each solver read and write the state the particles hold between two
    steps. A solver assigns the arrays of the position, velocity and activity of its particles and the array of their
    mass at build time, which the accessors read and write.
    """

    def __init__(self, sim: "Simulator", options: MPMOptions | PBDOptions | SPHOptions) -> None:
        super().__init__(sim, options)

        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        # The distance from the walls of the domain to the boundary confining the particles
        self.boundary_padding = 0.0

        # The totals over the entities added so far, which 'add_entity' keeps up to date
        self._n_particles = 0

        # The arrays of the particles the accessors read and write
        self._particles_pos: qd.Tensor | None = None
        self._particles_vel: qd.Tensor | None = None
        self._particles_is_active: qd.Tensor | None = None
        self._particles_mass: qd.Tensor | None = None

    @gs.assert_unbuilt
    def build(self, n_envs: int) -> None:
        # The deprecated backend reconstructing the surface of an entity maps onto the visual options of the solver
        legacy_recon_backends = {
            entity.surface.recon_backend for entity in self._entities if entity.surface.recon_backend is not None
        }
        if legacy_recon_backends:
            vis_options = self._options.vis
            if len(legacy_recon_backends) > 1 or (
                "recon_backend" in vis_options.model_fields_set and legacy_recon_backends != {vis_options.recon_backend}
            ):
                gs.raise_exception(
                    f"The entities of {type(self).__name__} reconstruct their surface with one backend, set by "
                    f"'vis.recon_backend' of its options ({vis_options.recon_backend!r}), got "
                    f"{sorted(legacy_recon_backends)} from the deprecated 'surface.recon_backend'."
                )
            (recon_backend,) = legacy_recon_backends
            vis_options = vis_options.model_copy(update=dict(recon_backend=recon_backend))
            self._options = self._options.model_copy(update=dict(vis=vis_options))
        super().build(n_envs)

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_particles_pos(self, poss, particles_idx=None, envs_idx=None) -> None:
        """
        Set the position of some particles, at rest.

        Parameters
        ----------
        poss : array_like, shape ([M,] [N,] 3)
            Target position of each particle.
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles. If None, all particles will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        poss, particles_idx, envs_idx = self._sanitize_io_variables(
            poss, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_set_particles_pos(particles_idx, envs_idx, poss, self._particles_pos, self._particles_vel)
        self._reset_particles(particles_idx, envs_idx, reset_deformation=True)
        # Overwriting the positions of the particles clears the errors their former state flagged
        self.clear_errno(envs_idx)

    def get_particles_pos(self, particles_idx=None, envs_idx=None, *, copy: bool | None = True) -> torch.Tensor:
        """Get the position of some particles, the same in every environment (see 'set_particles_pos')."""
        tensor = self._read_data(self._particles_pos, envs_idx, particles_idx, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_particles_vel(self, vels, particles_idx=None, envs_idx=None) -> None:
        """Set the velocity of some particles (see 'set_particles_pos')."""
        self._write_data(
            self._particles_vel, vels, particles_idx, self.n_particles, "particles_idx", envs_idx, element_shape=(3,)
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def get_particles_vel(self, particles_idx=None, envs_idx=None, *, copy: bool | None = True) -> torch.Tensor:
        """Get the velocity of some particles, the same in every environment (see 'set_particles_pos')."""
        tensor = self._read_data(self._particles_vel, envs_idx, particles_idx, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    @gs.assert_built
    @mutates(StateChange.GEOMETRY, differentiable=False)
    def set_particles_active(self, actives, particles_idx=None, envs_idx=None) -> None:
        """Set whether some particles take part in the simulation (see 'set_particles_pos')."""
        actives, particles_idx, envs_idx = self._sanitize_io_variables(
            actives, particles_idx, self.n_particles, "particles_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        kernel_set_values(particles_idx, envs_idx, actives, self._particles_is_active)
        self._reset_particles(particles_idx, envs_idx, reset_deformation=False)

    def get_particles_active(self, particles_idx=None, envs_idx=None, *, copy: bool | None = True) -> torch.Tensor:
        """Get whether some particles take part in the simulation (see 'set_particles_pos')."""
        tensor = self._read_data(self._particles_is_active, envs_idx, particles_idx, copy=copy)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_particles_mass(self, particles_idx=None, envs_idx=None) -> torch.Tensor:
        """Get the mass of some particles (see 'set_particles_pos')."""
        # The particles store their mass multiplied by 'particle_mass_scale', the same in every environment
        masses = (qd_to_torch(self._particles_mass) / self.particle_mass_scale)[None].expand(self.n_envs_, -1)
        masses = masses[indices_to_mask(envs_idx, particles_idx, boolean_mask=False, pair_rows=True)]
        return masses[0] if self.n_envs == 0 else masses

    def _reset_particles(self, particles_idx: torch.Tensor, envs_idx: torch.Tensor, reset_deformation: bool) -> None:
        """
        Reset the state of the given particles beside their position and velocity, after they moved or changed activity.

        `reset_deformation` asks a deformable material to restart undeformed.
        """
        pass

    @property
    def is_active(self) -> bool:
        return self.n_particles > 0

    @property
    def particle_mass_scale(self) -> float:
        """The factor the particles store their mass multiplied by, for numerical robustness."""
        return 1.0

    @property
    def n_particles(self) -> int:
        return self._n_particles

    @property
    def particle_size(self) -> float:
        return self._particle_size

    @property
    def particle_radius(self) -> float:
        return self._particle_size / 2.0

    @property
    def upper_bound(self) -> np.ndarray:
        return self._upper_bound

    @property
    def lower_bound(self) -> np.ndarray:
        return self._lower_bound

    @property
    def boundary_lower(self) -> np.ndarray:
        """The lower corner of the boundary confining the particles, inside the domain by the padding."""
        return self._lower_bound + self.boundary_padding

    @property
    def boundary_upper(self) -> np.ndarray:
        """The upper corner of the boundary confining the particles, inside the domain by the padding."""
        return self._upper_bound - self.boundary_padding


class ParticleScopedRenderContext(ScopedRenderContext[ParticleSolver]):
    """The particles of a particle-based solver as drawn, the visual meshes they skin, and the surfaces reconstructed
    from the active particles of the entities drawn as such.

    The structure of the material of an entity decides how it is drawn (see 'Material.structure'): a deformable or
    tearable entity skins a visual mesh on its particles, an amorphous one is drawn as the surface reconstructed from
    them. A tearable entity whose skinned mesh tore is drawn as a reconstructed surface in that environment until the
    next reset.
    """

    debug_vis_modes = gs.entities.MPM._debug_vis_modes
    boundary_color: ClassVar[UnitIntervalVec4Type] = (0.0, 1.0, 1.0, 1.0)
    """The color of the marker of the boundary confining the particles."""

    def build(self) -> None:
        solver = self.owner
        self._skinned_entities = [entity for entity in solver.entities if entity.n_vverts > 0]
        # The first vertex of the visual mesh of each skinned entity among the skinned vertices, entity by entity
        entities_n_vverts = np.array([entity.n_vverts for entity in self._skinned_entities], dtype=gs.np_int)
        self._entities_vvert_start = dict(
            zip(self._skinned_entities, (np.cumsum(entities_n_vverts) - entities_n_vverts).tolist())
        )
        self._tearable_entities = [
            entity for entity in self._skinned_entities if entity.material.structure == STRUCTURE.TEARABLE
        ]
        # The edges of the skinned meshes of the tearable entities, whose stretch is measured at each refresh
        edges_vverts_idx, edges_rest_length = [], []
        for entity in self._tearable_entities:
            edges = entity.vmesh.trimesh.edges_unique
            edges_length = entity.vmesh.trimesh.edges_unique_length
            is_edge_kept = edges_length > gs.EPS
            edges_vverts_idx.append(edges[is_edge_kept] + self._entities_vvert_start[entity])
            edges_rest_length.append(edges_length[is_edge_kept])
        self._skinned = SkinnedMeshes(
            sizes=dict(
                vverts=int(entities_n_vverts.sum()),
                vvert_supports=solver.options.vis.n_support_neighbors,
                skinned_entities=len(self._tearable_entities),
                skinned_edges=sum(map(len, edges_rest_length)),
                envs=solver.n_envs_,
            ),
            values=dict(STRETCH_MAX=SKINNED_EDGE_STRETCH_MAX),
        )
        if self._tearable_entities:
            self._skinned.entities_is_intact.fill(True)
            self._skinned.edges_vverts_idx.from_numpy(np.concatenate(edges_vverts_idx, dtype=gs.np_int))
            self._skinned.edges_rest_length.from_numpy(np.concatenate(edges_rest_length, dtype=gs.np_float))
            self._skinned.edges_entity_idx.from_numpy(
                np.concatenate(
                    [np.full(len(edges), i_t, dtype=gs.np_int) for i_t, edges in enumerate(edges_rest_length)]
                )
            )
        # The particles of each entity, and the surface its reconstructed surface is drawn with for an entity drawn as
        # such, None otherwise, made as the scoped context builds.
        self._entities_points: list[Points] = []
        # The particles standing for the surfaces reconstructed at the latest refresh (see 'update_shapes')
        self._surfaces_points: tuple[Points, ...] = ()
        # The edges of the box confining the particles
        self._boundary: tuple[Lines, ...] = ()
        self._entities_recon_surface: list[Surface | None] = []
        # The weights of a visual vertex are its barycentric coordinates in the simplex of its support particles, its
        # nearest particles at rest. The distances of the vertices to the particles are computed by chunks of bounded
        # memory, a chunk holding one vertex at least.
        n_supports = solver.options.vis.n_support_neighbors
        for entity in self._skinned_entities:
            vverts, particles = entity.vmesh.verts.astype(gs.np_float, copy=False), entity.init_particles
            chunk_size = max(VVERTS_DIST2_MAX_SIZE // len(particles), 1)
            supports_idx = np.empty((len(vverts), n_supports), dtype=gs.np_int)
            for i_vv_start in range(0, len(vverts), chunk_size):
                vverts_chunk = vverts[i_vv_start : i_vv_start + chunk_size]
                dist2 = np.sum(np.square(vverts_chunk[:, None, :] - particles), axis=-1)
                chunk_supports_idx = np.argpartition(dist2, n_supports - 1, axis=1)[:, :n_supports]
                chunk_supports_dist2 = np.take_along_axis(dist2, chunk_supports_idx, axis=1)
                sorted_order = np.lexsort((chunk_supports_idx, chunk_supports_dist2))
                supports_idx[i_vv_start : i_vv_start + chunk_size] = np.take_along_axis(
                    chunk_supports_idx, sorted_order, axis=1
                )
            supports_idx = np.clip(supports_idx, 0, len(particles) - 1)
            supports_pos = particles[supports_idx]
            supports_offset = supports_pos[:, :-1].swapaxes(-2, -1) - supports_pos[:, -1, :, None]
            supports_offset_pinv = np.linalg.pinv(supports_offset)
            kernel_add_vverts(
                self._entities_vvert_start[entity],
                entity.particle_start,
                supports_idx,
                vverts,
                particles,
                supports_offset_pinv,
                self._skinned,
            )

        gravity = tuple(np.atleast_2d(tensor_to_array(solver.get_gravity()))[0].tolist())
        lower_bound, upper_bound = tuple(solver.lower_bound.tolist()), tuple(solver.upper_bound.tolist())
        shapes: list[Shape] = []
        i_t = 0
        for entity in solver.entities:
            seg_keys = SegmentationKeys(entity.idx, entity.idx, entity.idx)
            surface = entity.surface if entity.n_vverts == 0 else entity.vmesh.surface
            # The deprecated 'vis_mode' of an entity shows its particles as the scene looks, and nothing else
            is_pinned = entity.surface.vis_mode is not None
            particles_slice = slice(entity.particle_start, entity.particle_start + entity.n_particles)
            points = Points(
                ArrayRef(solver._particles_pos, particles_slice),
                solver.particle_radius,
                surface=surface,
                seg_keys=seg_keys,
                vel=ArrayRef(solver._particles_vel, particles_slice),
                is_active=ArrayRef(solver._particles_is_active, particles_slice),
                physics=ParticlesPhysics(entity.material.rho, gravity, lower_bound, upper_bound),
                roles=ShapeRoles.of(entity.idx, is_looks=is_pinned, is_debug=True),
                _views=frozenset(("particle",)),
            )
            self._entities_points.append(points)
            shapes.append(points)
            is_reconstructed = entity.material.structure == STRUCTURE.AMORPHOUS or isinstance(
                entity.morph, gs.morphs.Nowhere
            )
            is_torn_reconstructed = False
            if not is_reconstructed and entity.n_vverts > 0:
                # A visual mesh whose UVs do not map its vertices one to one is drawn without texture
                uvs = entity.vmesh.uvs
                if uvs is not None and len(uvs) != entity.n_vverts:
                    uvs = None
                is_tearable = entity.material.structure == STRUCTURE.TEARABLE
                vvert_start = self._entities_vvert_start[entity]
                shapes.append(
                    DynamicMesh(
                        entity.vmesh.faces,
                        surface,
                        ArrayRef(self._skinned.pos, slice(vvert_start, vvert_start + entity.n_vverts)),
                        uvs,
                        seg_keys,
                        particles=points,
                        is_active=ArrayRef(self._skinned.entities_is_intact, i_t) if is_tearable else None,
                        roles=ShapeRoles.of(entity.idx, is_looks=not is_pinned, is_debug=False),
                    )
                )
                i_t += is_tearable
                is_torn_reconstructed = is_tearable
            # An amorphous entity is drawn as a reconstructed surface, and a tearable one in the environments it tore in
            recon_surface = None
            if (is_reconstructed or is_torn_reconstructed) and not is_pinned:
                recon_surface = surface.model_copy(update=dict(smooth=True))
            self._entities_recon_surface.append(recon_surface)
        if solver.options.vis.boundary:
            # The corner i of the box takes along each axis the upper bound where its bit of i is set, so that the
            # corners of an edge differ by one bit.
            corners = np.array(tuple(itertools.product(*zip(solver.boundary_lower, solver.boundary_upper))))
            edges = [(i, j) for i, j in itertools.combinations(range(8), 2) if (i ^ j).bit_count() == 1]
            edges_corners = torch.tensor(corners[np.array(edges)], dtype=gs.tc_float, device=gs.device)
            edges_corners = edges_corners.expand((solver.n_envs_, *edges_corners.shape))
            self._boundary = (
                Lines(edges_corners[..., 0, :], edges_corners[..., 1, :], radius=0.002, color=self.boundary_color),
            )
        self._shapes = tuple(shapes)

    def update_markers(self, envs_idx: torch.Tensor) -> tuple[Shape, ...]:
        return self._boundary

    def update_shapes(self, envs_idx: torch.Tensor | None, view: str) -> tuple[Shape, ...]:
        """Skin the visual meshes, detect their tears, and return the shapes of a representation.

        Particles stand for the reconstructed surfaces among them: those of every amorphous entity, and of each torn
        entity in the environments it tore in.
        """
        if envs_idx is not None:
            solver = self.owner
            if self._skinned_entities:
                kernel_update_skinned_vverts(envs_idx, solver._particles_pos, self._skinned)
            envs_is_intact = None
            if self._tearable_entities:
                kernel_update_skinned_tear(envs_idx, self._skinned)
                envs_is_intact = qd_to_torch(self._skinned.entities_is_intact, envs_idx, transpose=True)
            surfaces_points = []
            i_t = 0
            for entity, points, recon_surface in zip(
                solver.entities, self._entities_points, self._entities_recon_surface
            ):
                is_tearable = entity.n_vverts > 0 and entity.material.structure == STRUCTURE.TEARABLE
                if recon_surface is None:
                    i_t += is_tearable
                    continue
                envs_shown_idx = envs_idx
                if is_tearable:
                    envs_shown_idx = envs_idx[torch.logical_not(envs_is_intact[:, i_t])]
                    i_t += 1
                    if len(envs_shown_idx) == 0:
                        continue
                # The particles stand for the surface reconstructed from them, which draws the entity as it looks. They
                # view the arrays of the particles in every environment, and copy those of the environments showing
                # the surface otherwise.
                particles_slice = slice(entity.particle_start, entity.particle_start + entity.n_particles)
                arrays = [
                    ArrayRef(array, particles_slice)
                    for array in (solver._particles_pos, solver._particles_vel, solver._particles_is_active)
                ]
                if len(envs_shown_idx) < solver.n_envs_:
                    arrays = [array.view(is_per_env=True)[envs_shown_idx] for array in arrays]
                pos, vel, is_active = arrays
                surfaces_points.append(
                    points.fork(
                        uid=gs.UID(),
                        pos=pos,
                        surface=recon_surface,
                        vel=vel,
                        is_active=is_active,
                        envs_idx=None if len(envs_shown_idx) == solver.n_envs_ else envs_shown_idx,
                        roles=ShapeRoles.of(entity.idx, is_looks=True, is_debug=False),
                        _views=frozenset(),
                        is_surface=True,
                    )
                )
            self._surfaces_points = tuple(surfaces_points)
        return select_view_shapes((*self._shapes, *self._surfaces_points), view)

    def reconstruct(self, points: Points) -> tuple[DynamicMesh, ...]:
        solver = self.owner
        # The particles the boundary clamps onto one of its edges or corners crowd along a line or onto a single point,
        # all neighbours of each other, which makes the time and memory of a reconstruction grow with the square of
        # their number. They are left out, the particles resting against a single wall staying in.
        boundary = torch.as_tensor(
            np.stack((solver.boundary_lower, solver.boundary_upper)), dtype=gs.tc_float, device=gs.device
        )
        particles_n_clamped_axes = (
            torch.isclose(points.pos[..., None, :], boundary, rtol=gs.EPS, atol=gs.EPS).any(dim=-2).sum(dim=-1)
        )
        particles_is_kept = torch.logical_and(points.is_active, particles_n_clamped_axes < 2)
        # The particles kept in every environment are selected on the device and copied at once, then split per
        # environment on the host, which spares a synchronization and a copy per environment.
        particles_pos = tensor_to_array(points.pos[particles_is_kept])
        envs_n_kept = tensor_to_array(particles_is_kept.sum(dim=-1))
        meshes = []
        for i_r, positions in enumerate(np.split(particles_pos, np.cumsum(envs_n_kept)[:-1])):
            mesh = pu.particles_to_mesh(
                positions=positions,
                radius=solver.particle_radius,
                backend=solver.options.vis.recon_backend,
            )
            env_idx = points.envs_idx[i_r : i_r + 1]
            # The surface is reconstructed on the host, where the renderers drawing it read its vertices, so that its
            # arrays, made anew at each refresh, take no device memory.
            meshes.append(
                DynamicMesh(
                    torch.from_numpy(mesh.faces),
                    points.surface,
                    torch.from_numpy(mesh.vertices[None]),
                    seg_keys=points.seg_keys,
                    roles=points.roles,
                    particles=points.fork(
                        uid=gs.UID(),
                        pos=points.pos[i_r : i_r + 1],
                        vel=points.vel[i_r : i_r + 1],
                        is_active=points.is_active[i_r : i_r + 1],
                        envs_idx=env_idx,
                    ),
                    envs_idx=env_idx,
                )
            )
        return tuple(meshes)

    def reset(self, envs_idx: torch.Tensor) -> None:
        if self._tearable_entities:
            kernel_reset_skinned_tear(envs_idx, self._skinned)


@qd.kernel(fastcache=True)
def kernel_add_vverts(
    vvert_start: int,
    particle_start: int,
    support_idxs_local: qd.types.ndarray(),
    vverts: qd.types.ndarray(element_dim=1),
    particles: qd.types.ndarray(element_dim=1),
    P_invs: qd.types.ndarray(element_dim=2),
    skinned: SkinnedMeshes,
):
    """
    Bind the given vverts to their supporting particles, whose positions skin them for rendering.

    The weights of a vvert are its barycentric coordinates in the simplex of its support particles, `P_invs` mapping its
    offset from the last support to the others.
    """
    n_supports = support_idxs_local.shape[1]
    for i_vv_ in range(vverts.shape[0]):
        i_vv = i_vv_ + vvert_start
        v = vverts[i_vv_]
        P_inv = P_invs[i_vv_]
        i_p_last = support_idxs_local[i_vv_, n_supports - 1]
        b = P_inv @ (v - particles[i_p_last])
        for i_s in range(n_supports - 1):
            skinned.support_weights[i_vv][i_s] = b[i_s]
        skinned.support_weights[i_vv][n_supports - 1] = 1.0 - b.sum()
        for i_s in range(n_supports):
            skinned.support_idxs[i_vv][i_s] = support_idxs_local[i_vv_, i_s] + particle_start


@qd.kernel(fastcache=True)
def kernel_set_particles_pos(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
):
    """Set the position of some particles, at rest."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(particles_pos))):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_values' in base_solver.py
        if i_p >= 0:
            particles_pos[i_p, i_b] = qd.Vector([poss[i_b_, i_p_, 0], poss[i_b_, i_p_, 1], poss[i_b_, i_p_, 2]])
            particles_vel[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_update_skinned_vverts(envs_idx: qd.Tensor, particles_pos: qd.Tensor, skinned: SkinnedMeshes):
    """Write each visual vertex skinned on the particles, the weighted sum of the positions of its support particles, in
    the environments `envs_idx`."""
    for i_vv, i_b_ in qd.ndrange(skinned.support_idxs.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        vvert_pos = qd.Vector.zero(gs.qd_float, 3)
        for k in qd.static(range(skinned.support_idxs[i_vv].n)):
            i_p = skinned.support_idxs[i_vv][k]
            vvert_pos = vvert_pos + particles_pos[i_p, i_b] * skinned.support_weights[i_vv][k]
        skinned.pos[i_vv, i_b] = vvert_pos


@qd.kernel(fastcache=True)
def kernel_update_skinned_tear(envs_idx: qd.Tensor, skinned: SkinnedMeshes):
    """Mark torn the tearable entities an edge of whose skinned visual mesh stretched past its limit, in the
    environments `envs_idx`."""
    for i_ed, i_b_ in qd.ndrange(skinned.edges_rest_length.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_va = skinned.edges_vverts_idx[i_ed][0]
        i_vb = skinned.edges_vverts_idx[i_ed][1]
        edge_length = (skinned.pos[i_va, i_b] - skinned.pos[i_vb, i_b]).norm()
        if edge_length > skinned.STRETCH_MAX[None] * skinned.edges_rest_length[i_ed]:
            i_e = skinned.edges_entity_idx[i_ed]
            skinned.entities_is_intact[i_e, i_b] = False


# FIXME: quadrants#1036 - a func parameter accepts no subclass of its dataclass annotation, so the funcs shared by the
# solvers take the arrays of their particles one by one.
@qd.func
def func_get_particles_mean_vel(
    i_b: int,
    pos: qd.types.vector(3),
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    hash_grid: SpatialHashGrid,
    half_size: float,
    n_cells: int,
):
    """
    Return the mean velocity of the sorted particles of a solver within a cube around a position, and their count.

    The cube is centered at `pos` with half side `half_size`. The particles are searched for among the cells of the hash
    grid of the solver, `n_cells` along each axis from the one holding the lowest corner of the cube.
    """
    base = func_hash_wrap_cell(func_hash_pos_to_grid(pos - half_size, hash_grid), hash_grid)
    vel_sum = qd.Vector.zero(gs.qd_float, 3)
    n_particles = 0
    n_cells_distinct = func_hash_n_cells(n_cells, hash_grid)
    for offset in qd.grouped(qd.ndrange(n_cells_distinct[0], n_cells_distinct[1], n_cells_distinct[2])):
        i_s = func_hash_cell_to_slot(base + offset, hash_grid)
        i_p_start = hash_grid.slot_start[i_s, i_b]
        for i_p in range(i_p_start, i_p_start + hash_grid.slot_size[i_s, i_b]):
            if qd.abs(pos - particles_pos[i_p, i_b]).max() < half_size:
                vel_sum += particles_vel[i_p, i_b]
                n_particles += 1
    return vel_sum / qd.max(n_particles, 1), n_particles


@qd.func
def func_add_particles_momentum(
    i_b: int,
    pos: qd.types.vector(3),
    momentum: qd.types.vector(3),
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_is_moved: qd.Tensor,
    particles_mass: qd.Tensor,
    hash_grid: SpatialHashGrid,
    half_size: float,
    n_cells: int,
):
    """
    Add a momentum to the sorted particles of a solver in a cube around a position (see func_get_particles_mean_vel).

    Each particle that `particles_is_moved` selects gains the velocity `momentum` over its mass, which `particles_mass`
    holds by particle index.
    """
    base = func_hash_wrap_cell(func_hash_pos_to_grid(pos - half_size, hash_grid), hash_grid)
    n_cells_distinct = func_hash_n_cells(n_cells, hash_grid)
    for offset in qd.grouped(qd.ndrange(n_cells_distinct[0], n_cells_distinct[1], n_cells_distinct[2])):
        i_s = func_hash_cell_to_slot(base + offset, hash_grid)
        i_p_start = hash_grid.slot_start[i_s, i_b]
        for i_p in range(i_p_start, i_p_start + hash_grid.slot_size[i_s, i_b]):
            if qd.abs(pos - particles_pos[i_p, i_b]).max() < half_size and particles_is_moved[i_p, i_b]:
                i_o = hash_grid.particles_idx[i_p, i_b]
                particles_vel[i_p, i_b] = particles_vel[i_p, i_b] + momentum / particles_mass[i_o]


@qd.kernel(fastcache=True)
def kernel_reset_skinned_tear(envs_idx: qd.types.ndarray(), skinned: SkinnedMeshes):
    for i_e, i_b_ in qd.ndrange(skinned.entities_is_intact.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        skinned.entities_is_intact[i_e, i_b] = True
