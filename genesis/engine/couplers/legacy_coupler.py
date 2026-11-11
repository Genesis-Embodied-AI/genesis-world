import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import igl

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.boundaries import func_impose_cube_boundary
from genesis.engine.solvers.base_solver import CheckpointMixin, ErrnoMixin, SimulatorState
from genesis.engine.solvers.fem_solver import SURFACES, VERTICES, FEMInfo, FEMState
from genesis.engine.solvers.mpm_solver import (
    MPMInfo,
    MPMState,
    func_point_vel_after_collision_with_rigid_geom,
    func_point_vel_after_two_way_collision_with_rigid_geom,
    func_quadratic_bspline_stencil,
)
from genesis.engine.solvers.particle_solver import func_add_particles_momentum, func_get_particles_mean_vel
from genesis.engine.solvers.pbd_solver import PBDInfo, PBDState
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.abd.misc import func_add_external_wrench
from genesis.engine.solvers.rigid.data import GeomsInfo, GeomsState, LinksInfo, LinksState, RigidData, RigidInfo
from genesis.engine.solvers.sph_solver import (
    SPHInfo,
    SPHState,
    func_cubic_kernel,
    func_cubic_kernel_derivative,
    func_segment_dist,
    func_segment_volume,
)
from genesis.grad.bridge import reversed_with
from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.data import ENVS, Config, Data, DataItem, DataKind, Size, eoa, field, is_opted_in, iter_data, opt_in
from genesis.utils.geom import (
    PARTICLES,
    SpatialHashGrid,
    func_hash_cell_to_slot,
    func_hash_pos_to_grid,
    func_hash_sort_particles,
    func_hash_wrap_cell,
    hash_grid_sizes,
    qd_inv_transform_by_trans_quat,
    qd_transform_by_quat,
    qd_transform_by_trans_quat,
)
from genesis.utils.misc import get_gpu_core_count, qd_to_numpy, tensor_to_array
from genesis.utils.raycast_qd import triangle_face_normal
from genesis.utils.sdf import SDFInfo
from genesis.utils.simt import (
    ExactSums,
    func_exact_sum_add,
    func_exact_sum_take,
    requires_deterministic_accumulation,
    qd_block_max,
    qd_block_min,
    qd_block_sum,
)

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom
    from genesis.engine.simulator import Simulator


PBD_PARTICLES = Size("pbd_particles")
# The arrays of the coupling of the rigid geoms with the PBD particles, which a scene without it leaves out
RIGID_PBD = "rigid_pbd"


# The segments of the surfaces of the FEM entities and of the rigid geoms, which a scene without their coupling with
# the SPH liquid leaves out
FEM_SPH = "fem_sph"
RIGID_SPH = "rigid_sph"
FEM_SEGMENTS = Size("fem_segments")
RIGID_SEGMENTS = Size("rigid_segments")
RIGID_LINKS = Size("rigid_links")
RIGID_GEOMS = Size("rigid_geoms")
# The points of the quadrature of a segment (see func_segment_volume)
QUAD_POINTS = Size("quad_points")


class SPHSegments(Data, kind=DataKind.SCRATCH, shape=(PARTICLES, ENVS)):
    """
    The segments of the moving boundaries of the SPH liquid over the current substep, triangles of the surfaces.

    The segments of the FEM entities come first, then those of the rigid geoms. `normal` points into the liquid, and the
    hash grid of the segments takes them for particles at their `centroid` (see SPHBoundaries).
    """

    corners: qd.Tensor = field((float, (3, 3)))
    normal: qd.Tensor = field((float, 3))
    centroid: qd.Tensor = field((float, 3))
    vel: qd.Tensor = field((float, 3))
    area: qd.Tensor = field(float)
    is_active: qd.Tensor = field(bool)
    # The largest distance from the centroid of the segment to its corners
    extent: qd.Tensor = field(float)


class SPHBoundaries(Data, kind=DataKind.SCRATCH):
    """The surfaces of the FEM entities and of the rigid geoms as moving boundaries of the SPH liquid."""

    segments: SPHSegments
    # Rebuilt from the segments every substep
    hash_grid: SpatialHashGrid = field(kind=DataKind.SCRATCH)
    # The impulse that the SPH particles give each segment over the substep, which its surface takes
    segments_impulse: qd.Tensor = field((float, 3), shape=(PARTICLES, ENVS))
    # The barycentric coordinates and the weights of the points of the quadrature of a segment, Radon's 7-point rule on
    # the triangle, exact up to degree 5
    quad_bary: qd.Tensor = field((float, 3), shape=(QUAD_POINTS,), kind=DataKind.CONSTANT)
    quad_weight: qd.Tensor = field(float, shape=(QUAD_POINTS,), kind=DataKind.CONSTANT)
    # The surface triangle of each FEM segment, and the barycentric coordinates of its corners in the triangle
    fem_segments_tri_idx: qd.Tensor = field(int, shape=(FEM_SEGMENTS,), kind=DataKind.CONSTANT, labels=FEM_SPH)
    fem_segments_corners_bary: qd.Tensor = field(
        (float, (3, 3)), shape=(FEM_SEGMENTS,), kind=DataKind.CONSTANT, labels=FEM_SPH
    )
    # The FEM segments of each surface triangle span tris_segment_start[i_s] to tris_segment_start[i_s + 1]
    tris_segment_start: qd.Tensor = field(int, shape=(SURFACES + 1,), kind=DataKind.CONSTANT, labels=FEM_SPH)
    # The geom of each rigid segment, and its corners in the frame of the geom
    rigid_segments_geom_idx: qd.Tensor = field(int, shape=(RIGID_SEGMENTS,), kind=DataKind.CONSTANT, labels=RIGID_SPH)
    rigid_segments_corners: qd.Tensor = field(
        (float, (3, 3)), shape=(RIGID_SEGMENTS,), kind=DataKind.CONSTANT, labels=RIGID_SPH
    )
    # Whether each geom takes part in each environment, the variants of a heterogeneous entity being active in some alone
    geoms_is_active: qd.Tensor = field(bool, shape=(RIGID_GEOMS, ENVS), kind=DataKind.CONSTANT, labels=RIGID_SPH)
    # Whether liquid lies around each segment over the substep
    segments_is_wet: qd.Tensor = field(bool, shape=(PARTICLES, ENVS), labels=RIGID_SPH)
    # The rest density of the densest liquid of the scene (see 'links_added_mass')
    rho_rest: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, labels=RIGID_SPH)
    # The mass of the liquid one particle size thick over the wet segments of each rigid link over the substep. The
    # liquid pushes a link a substep late, which diverges for a body lighter than the liquid it drags, so this mass moves
    # to the inertia side of the link while gravity keeps acting on the true mass, and a link at rest takes the full
    # force of the liquid.
    # FIXME: The added mass also slows down the transients of a light body. Compensating it with the force of the added
    # mass at the previous acceleration of the link, which would restore them, makes the light thin bodies diverge again.
    links_added_mass: qd.Tensor = field(float, shape=(RIGID_LINKS, ENVS), labels=RIGID_SPH)
    # The added mass of each link summed over its segments, the same whatever the order of the segments
    links_added_mass_sums: ExactSums = field(labels=RIGID_SPH)
    # The largest extent of the segments of each environment over the substep, which bounds the reach of the particles
    # that look for the segments around them through the hash grid of their centroids
    segments_extent_max: qd.Tensor = field(float, shape=(ENVS,))
    # The boundary each segment belongs to: its rigid geom, or its FEM entity numbered after the rigid geoms
    segments_boundary_idx: qd.Tensor = field(int, shape=(PARTICLES,), kind=DataKind.CONSTANT)


class LegacyCouplerState(Data, kind=DataKind.STATE):
    """
    Runtime state of the legacy coupler.

    The PBD particles animated by a rigid link follow the point `pbd_attach_local_pos` of the link
    `pbd_attach_link_idx`, -1 for none.
    """

    # The error flags of each env, which no kernel of the coupler raises (see 'ErrnoMixin.register_errors')
    errno: qd.Tensor = field(int, shape=(ENVS,))
    pbd_attach_link_idx: qd.Tensor = field(int, shape=(PBD_PARTICLES, ENVS), labels=RIGID_PBD)
    pbd_attach_local_pos: qd.Tensor = field((float, 3), shape=(PBD_PARTICLES, ENVS), labels=RIGID_PBD)


class LegacyCouplerInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the legacy coupler.

    `influence_min` is the influence of a rigid geom at a point below which the geom leaves the particle or vertex there
    untouched, and `animate_inv_dt` the inverse time step the particles attached to links are animated with, that of
    the PBD substep clamped to avoid large jerks. The cells of the MPM grid gather the SPH and PBD particles of a cube
    of `mpm_sph_stencil_size` and `mpm_pbd_stencil_size` cells of their hash grid. `softness_min` bounds away from zero
    the coupling softness of a rigid geom, which its influence divides the distance by.
    """

    influence_min: qd.Tensor = field(float, shape=())
    # The tolerance relative to the speed of a particle within which it meets a bound of its velocity (see
    # kernel_sph_boundaries_impulse)
    velocity_tol: qd.Tensor = field(float, shape=())
    # The determinant of the Gram matrix of unit normals below which they count as dependent (see
    # func_project_on_bounds)
    normals_independence_min: qd.Tensor = field(float, shape=())
    animate_inv_dt: qd.Tensor = field(float, shape=())
    softness_min: qd.Tensor = field(float, shape=())
    mpm_sph_stencil_size: qd.Tensor = field(int, shape=())
    mpm_pbd_stencil_size: qd.Tensor = field(int, shape=())


class LegacyCouplerConfig(Config):
    """The couplings of the legacy coupler that its kernels compile in, between the solvers each one names."""

    has_rigid_mpm: bool
    has_rigid_sph: bool
    has_rigid_pbd: bool
    has_rigid_fem: bool
    has_mpm_sph: bool
    has_mpm_pbd: bool
    has_fem_mpm: bool
    has_fem_sph: bool
    # Whether the rigid links hold their info per environment (see 'RigidOptions.batch_links_info')
    is_links_info_batched: bool
    # Whether the SPH liquid sums its forces on the rigid links exactly (see 'cfrc_external_sums' in LinksState)
    is_external_wrench_exact: bool
    # Whether the lanes of a GPU subgroup take a particle of the SPH liquid together, and a segment of its boundaries,
    # one thread taking it alone on CPU and once the particles, or the segments, of every environment outnumber the GPU
    # cores, which one thread each already saturates
    enable_cooperative_particles: bool
    enable_cooperative_segments: bool


def split_triangles(
    tris_verts_pos: np.ndarray,
    size: float,
    tris_verts_pos_world: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split triangles by halving the longest edge of their parts until all the edges of the parts fit within a size.

    Return the triangle of each part and the barycentric coordinates of its corners in the triangle, the parts sorted by
    triangle.

    With `bounds`, the lower and upper corners of a box, the parts lying beyond one face of the box by more than a size,
    their vertices taken in `tris_verts_pos_world`, are dropped as they are split, which keeps the parts of a large
    triangle crossing the box to those near the box.
    """
    n_tris = len(tris_verts_pos)
    parts_tri_idx, parts_bary = np.arange(n_tris), np.tile(np.eye(3), (n_tris, 1, 1))
    kept_parts = [(parts_tri_idx[:0], parts_bary[:0])]
    while len(parts_tri_idx) > 0:
        if bounds is not None:
            parts_verts_pos_world = parts_bary @ tris_verts_pos_world[parts_tri_idx]
            is_away = (parts_verts_pos_world < bounds[0] - size).all(axis=1) | (
                parts_verts_pos_world > bounds[1] + size
            ).all(axis=1)
            is_near = ~is_away.any(axis=-1)
            parts_tri_idx, parts_bary = parts_tri_idx[is_near], parts_bary[is_near]
        parts_verts_pos = parts_bary @ tris_verts_pos[parts_tri_idx]
        parts_edges = np.linalg.norm(parts_verts_pos - np.roll(parts_verts_pos, -1, axis=1), axis=-1)
        is_split = parts_edges.max(axis=-1) > size
        kept_parts.append((parts_tri_idx[~is_split], parts_bary[~is_split]))
        # The longest edge runs from the corner i_a to the next one, i_b, opposite to the corner i_c
        i_a = parts_edges[is_split].argmax(axis=-1)
        i_b, i_c = (i_a + 1) % 3, (i_a + 2) % 3
        bary = parts_bary[is_split]
        idx = np.arange(len(bary))
        mid = 0.5 * (bary[idx, i_a] + bary[idx, i_b])
        parts_bary = np.concatenate(
            [np.stack([bary[idx, i_a], mid, bary[idx, i_c]], 1), np.stack([mid, bary[idx, i_b], bary[idx, i_c]], 1)]
        )
        parts_tri_idx = np.tile(parts_tri_idx[is_split], 2)
    parts_tri_idx, parts_bary = [np.concatenate(arrays) for arrays in zip(*kept_parts)]
    parts_order = np.argsort(parts_tri_idx, kind="stable")
    return parts_tri_idx[parts_order], parts_bary[parts_order]


def clip_triangles(tris_verts_pos: np.ndarray, tris_verts_dist: np.ndarray) -> np.ndarray:
    """
    Clip triangles to the side of a plane where the signed distance to the plane is nonpositive.

    `tris_verts_dist` is the signed distance of each vertex of the triangles to the plane. Return the vertices of the
    triangles making up the clipped parts, wound as the triangle they come from.
    """
    is_in = tris_verts_dist < 0.0
    n_in = np.count_nonzero(is_in, axis=-1)
    # The vertices turn so that the first one lies alone on its side of the plane
    vert_first = np.where(n_in == 1, is_in.argmax(axis=-1), (~is_in).argmax(axis=-1))
    verts_order = (vert_first[:, None] + np.arange(3)) % 3
    verts_dist = np.take_along_axis(tris_verts_dist, verts_order, axis=-1)
    verts_pos = np.take_along_axis(tris_verts_pos, verts_order[..., None], axis=-2)
    # The points where the edges from the first vertex cross the plane, on the triangles that it cuts
    is_cut = (n_in == 1) | (n_in == 2)
    dists_diff = verts_dist[:, :1] - verts_dist[:, 1:]
    edges_param = np.divide(verts_dist[:, :1], dists_diff, out=np.zeros_like(dists_diff), where=is_cut[:, None])
    cross_pos = verts_pos[:, :1] + edges_param[..., None] * (verts_pos[:, 1:] - verts_pos[:, :1])
    # A triangle with two vertices inside keeps a quad, split in two triangles, the second one being empty when its
    # first vertex lies on the plane
    clipped_parts = (
        (n_in == 3, verts_pos),
        (n_in == 1, np.stack([verts_pos[:, 0], cross_pos[:, 0], cross_pos[:, 1]], axis=1)),
        (n_in == 2, np.stack([cross_pos[:, 0], verts_pos[:, 1], verts_pos[:, 2]], axis=1)),
        ((n_in == 2) & (verts_dist[:, 0] > 0.0), np.stack([cross_pos[:, 0], verts_pos[:, 2], cross_pos[:, 1]], axis=1)),
    )
    return np.concatenate([parts_verts_pos[is_kept] for is_kept, parts_verts_pos in clipped_parts])


class LegacyCoupler(ErrnoMixin, CheckpointMixin, RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # Its whole state, the error flags and the PBD particles attached to rigid links, lives in the arrays 'data'
    # yields (see 'CheckpointMixin').
    is_checkpointable = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, simulator: "Simulator", options: "LegacyCouplerOptions") -> None:
        self.sim = simulator
        self.options = options

        self.rigid_solver = self.sim.rigid_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver

        self.legacy_coupler_state: LegacyCouplerState | None = None
        self.legacy_coupler_info: LegacyCouplerInfo | None = None
        self.legacy_coupler_config: LegacyCouplerConfig | None = None
        self.sph_boundaries: SPHBoundaries | None = None

        # The MPM and PBD particles and the FEM vertices collide with the rigid geoms through their SDF
        opt_in(SDFInfo, cond=lambda: any([self.rigid_couplings[i] for i in (0, 2, 3)]))
        # The surfaces of the FEM entities and of the rigid geoms are moving boundaries of the SPH liquid
        opt_in(SPHState.boundary, cond=lambda: self.has_fem_sph or bool(self.rigid_sph_geoms))
        # The SPH liquid sums its forces on the rigid links exactly when the order of the sums is fixed
        opt_in(
            RigidData.dyn_state.links.cfrc_external_sums,
            cond=lambda: bool(self.rigid_sph_geoms) and requires_deterministic_accumulation(),
        )

    @property
    def data(self) -> Iterator[DataItem]:
        """Yield every array the coupler holds, under the dotted name a checkpoint uses for it (see 'Solver.data')."""
        # A coupler whose build failed holds no array
        if self.legacy_coupler_state is None:
            return
        yield from iter_data(self, "legacy_coupler_state")
        yield from iter_data(self, "sph_boundaries")

    @property
    def has_fem_sph(self) -> bool:
        """Whether the surfaces of the FEM entities couple with the SPH liquid."""
        return self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph

    @property
    def rigid_sph_geoms(self) -> list["RigidGeom"]:
        """The rigid geoms that couple with the SPH liquid through their surface."""
        if not self.rigid_couplings[1]:
            return []
        # A plane bounds the liquid where the domain extends beyond it by more than the support radius. Otherwise, the
        # walls of the domain already bound the liquid there.
        domain_corners = np.stack(
            np.meshgrid(*zip(self.sph_solver.boundary_lower, self.sph_solver.boundary_upper), indexing="ij"), axis=-1
        ).reshape(-1, 3)
        rigid_sph_geoms = []
        for entity in self.rigid_solver.entities:
            is_plane_beyond_domain = False
            if isinstance(entity.morph, gs.morphs.Plane):
                # The pose of the plane composes its offset, as the root link of the entity does
                plane_pos, plane_quat = gu.transform_pos_quat_by_trans_quat(
                    np.asarray(entity.morph.offset_pos),
                    np.asarray(entity.morph.offset_quat),
                    np.asarray(entity.morph.pos),
                    np.asarray(entity.morph.quat),
                )
                normal = gu.transform_by_quat(np.asarray(entity.morph.normal), plane_quat)
                corners_dist = (domain_corners - plane_pos) @ normal
                is_plane_beyond_domain = corners_dist.min() > -self.sph_solver.support_radius
            for link in entity.links:
                for geom in link.geoms:
                    if geom.needs_coup and not (geom.type == gs.GEOM_TYPE.PLANE and is_plane_beyond_domain):
                        rigid_sph_geoms.append(geom)
        return rigid_sph_geoms

    @property
    def rigid_couplings(self) -> tuple[bool, bool, bool, bool]:
        """Whether the rigid geoms couple with the MPM, SPH, PBD and FEM solvers, in this order."""
        return tuple(
            [
                self.rigid_solver.is_active and solver.is_active and is_enabled
                for solver, is_enabled in (
                    (self.mpm_solver, self.options.rigid_mpm),
                    (self.sph_solver, self.options.rigid_sph),
                    (self.pbd_solver, self.options.rigid_pbd),
                    (self.fem_solver, self.options.rigid_fem),
                )
            ]
        )

    def build(self) -> None:
        has_rigid_mpm, _, has_rigid_pbd, has_rigid_fem = self.rigid_couplings
        has_mpm_sph = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph
        has_mpm_pbd = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        has_rigid_sph = self._build_sph_boundaries(self.rigid_sph_geoms)
        n_segments = self.sph_boundaries.segments_impulse.shape[0] if has_rigid_sph or self.has_fem_sph else 0
        self.legacy_coupler_config = LegacyCouplerConfig(
            has_rigid_mpm=has_rigid_mpm,
            has_rigid_sph=has_rigid_sph,
            has_rigid_pbd=has_rigid_pbd,
            has_rigid_fem=has_rigid_fem,
            has_mpm_sph=has_mpm_sph,
            has_mpm_pbd=has_mpm_pbd,
            has_fem_mpm=self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm,
            has_fem_sph=self.has_fem_sph,
            is_links_info_batched=self.rigid_solver.rigid_config.batch_links_info,
            is_external_wrench_exact=has_rigid_sph and is_opted_in(RigidData.dyn_state.links.cfrc_external_sums),
            enable_cooperative_particles=0 < self.sph_solver.n_particles * self.sim._B <= get_gpu_core_count(),
            enable_cooperative_segments=0 < n_segments * self.sim._B <= get_gpu_core_count(),
        )

        self.legacy_coupler_state = LegacyCouplerState(
            sizes=dict(pbd_particles=self.pbd_solver.n_particles, envs=self.sim._B),
            env_first=False,
            unused=dict(rigid_pbd=not has_rigid_pbd),
        )
        if has_rigid_pbd:
            self.legacy_coupler_state.pbd_attach_link_idx.fill(-1)
            self.legacy_coupler_state.pbd_attach_local_pos.fill(0.0)
        self.build_errno(self.sim._B)
        self.register_errors(self.legacy_coupler_state.errno, {})

        # The cells of the MPM grid reach the particles of the hash grid cells they overlap, plus one on each side
        mpm_sph_stencil_size = mpm_pbd_stencil_size = 0
        if has_mpm_sph:
            mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)
        if has_mpm_pbd:
            mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)
        self.legacy_coupler_info = LegacyCouplerInfo(
            values=dict(
                influence_min=0.1,
                velocity_tol=1e-6,
                normals_independence_min=1e-6,
                animate_inv_dt=min(1.0 / self.pbd_solver.substep_dt, 50.0),
                softness_min=1e-10,
                mpm_sph_stencil_size=mpm_sph_stencil_size,
                mpm_pbd_stencil_size=mpm_pbd_stencil_size,
            )
        )

        if self.legacy_coupler_config.has_rigid_mpm and self.mpm_solver.enable_CPIC:
            kernel_reset_mpm_rigid_normal(self.sim.scene._sanitize_envs_idx(None), self.mpm_solver.mpm_state)

    def _build_sph_boundaries(self, rigid_sph_geoms: list["RigidGeom"]) -> bool:
        """Build the segments of the moving boundaries of the SPH liquid, and return whether rigid geoms hold any."""
        has_fem_sph, has_rigid_sph = self.has_fem_sph, bool(rigid_sph_geoms)
        if not (has_fem_sph or has_rigid_sph):
            self.sph_boundaries = SPHBoundaries(unused=True)
            return False

        fem_tris_verts_pos = np.zeros((0, 3, 3), dtype=gs.np_float)
        if has_fem_sph:
            fem_tris_verts_pos = np.concatenate(
                [
                    tensor_to_array(entity.init_positions, dtype=gs.np_float)[entity.surface_triangles]
                    for entity in self.fem_solver.entities
                ]
            )
        fem_segments_tri_idx, fem_segments_corners_bary = split_triangles(
            fem_tris_verts_pos, self.sph_solver.particle_size
        )
        tris_segment_start = np.concatenate(
            [[0], np.cumsum(np.bincount(fem_segments_tri_idx, minlength=len(fem_tris_verts_pos)))]
        )

        geoms = rigid_sph_geoms
        rigid_tris_verts_pos = np.zeros((0, 3, 3), dtype=gs.np_float)
        if geoms:
            rigid_tris_verts_pos = np.concatenate([geom.init_verts[geom.init_faces] for geom in geoms])
        rigid_tris_geom_idx = np.repeat(
            np.array([geom.idx for geom in geoms], dtype=gs.np_int), [len(geom.init_faces) for geom in geoms]
        )
        rigid_tris_is_fixed = np.repeat(
            np.array([geom.is_fixed for geom in geoms], dtype=bool), [len(geom.init_faces) for geom in geoms]
        )
        # The geoms of the variants of a heterogeneous entity are only active in some environments
        geoms_is_active = np.ones((self.rigid_solver.n_geoms, self.sim._B), dtype=bool)
        for geom in geoms:
            if geom.active_envs_mask is not None:
                geoms_is_active[geom.idx] = tensor_to_array(geom.active_envs_mask)
        geoms_pos = qd_to_numpy(self.rigid_solver.dyn_state.geoms.pos, transpose=True)[0]
        geoms_quat = qd_to_numpy(self.rigid_solver.dyn_state.geoms.quat, transpose=True)[0]
        rigid_tris_verts_pos_world = gu.transform_by_trans_quat(
            rigid_tris_verts_pos, geoms_pos[rigid_tris_geom_idx, None], geoms_quat[rigid_tris_geom_idx, None]
        )
        # The fixed geoms, a terrain far larger than the domain among them, keep their segments within the domain alone,
        # where the walls of the domain do not already bound the liquid. The planes take their segments apart.
        rigid_tris_is_plane = np.repeat(
            [geom.type == gs.GEOM_TYPE.PLANE for geom in geoms], [len(geom.init_faces) for geom in geoms]
        )
        segments_parts = []
        for is_fixed in (False, True):
            (tris_idx,) = np.nonzero((rigid_tris_is_fixed == is_fixed) & ~rigid_tris_is_plane)
            tri_idx, corners_bary = split_triangles(
                rigid_tris_verts_pos[tris_idx],
                self.sph_solver.particle_size,
                rigid_tris_verts_pos_world[tris_idx],
                (self.sph_solver.boundary_lower, self.sph_solver.boundary_upper) if is_fixed else None,
            )
            segments_parts.append((tris_idx[tri_idx], corners_bary))
        rigid_segments_tri_idx, rigid_segments_corners_bary = [
            np.concatenate(arrays) for arrays in zip(*segments_parts)
        ]
        segments_centroid_world = np.einsum(
            "ni,nij->nj", rigid_segments_corners_bary.mean(axis=1), rigid_tris_verts_pos_world[rigid_segments_tri_idx]
        )
        is_in_domain = (
            (segments_centroid_world > self.sph_solver.boundary_lower)
            & (segments_centroid_world < self.sph_solver.boundary_upper)
        ).all(axis=-1)
        is_kept = is_in_domain | ~rigid_tris_is_fixed[rigid_segments_tri_idx]
        rigid_segments_tri_idx, rigid_segments_corners_bary = (
            rigid_segments_tri_idx[is_kept],
            rigid_segments_corners_bary[is_kept],
        )
        segments_geom_idx_parts = [rigid_tris_geom_idx[rigid_segments_tri_idx]]
        segments_corners_parts = [
            np.einsum("nci,nij->ncj", rigid_segments_corners_bary, rigid_tris_verts_pos[rigid_segments_tri_idx])
        ]
        # A plane bounds the half-space behind it, which the walls of the domain cut, their ghost particles standing for
        # the liquid beyond them (see func_wall_volume). The segments of the plane cover the boundary of the solid behind
        # it within the domain, so that the kernel volume of the walls and that of the plane add up without overlap: the
        # plane within the domain, wound about its normal into the liquid, and the faces of the domain behind the plane,
        # wound about their normal out of the domain. These faces stop at the support radius from the plane, beyond which
        # no particle in front of the plane reaches them.
        boundary_lower, boundary_upper = self.sph_solver.boundary_lower, self.sph_solver.boundary_upper
        domain_corners_pos = np.stack(
            np.meshgrid(*zip(boundary_lower, boundary_upper), indexing="ij"), axis=-1
        ).reshape((-1, 3))
        # The corner 4 * i_x + 2 * i_y + i_z lies at the upper bound along the axes whose index is one. The edges of the
        # domain join the corners that differ along one axis, and its faces are split in two triangles.
        domain_edges = np.array(
            [[i_c, i_c | axis_bit] for axis_bit in (1, 2, 4) for i_c in range(8) if not i_c & axis_bit]
        )
        domain_faces_verts_pos = domain_corners_pos[
            np.array(
                [
                    [0, 1, 3],
                    [0, 3, 2],
                    [4, 6, 7],
                    [4, 7, 5],
                    [0, 4, 5],
                    [0, 5, 1],
                    [2, 3, 7],
                    [2, 7, 6],
                    [0, 2, 6],
                    [0, 6, 4],
                    [1, 5, 7],
                    [1, 7, 3],
                ]
            )
        ]
        for geom in geoms:
            if geom.type == gs.GEOM_TYPE.PLANE:
                geom_pos, geom_quat = geoms_pos[geom.idx], geoms_quat[geom.idx]
                normal = gu.transform_by_quat(geom.data[:3], geom_quat)
                # The plane cuts the domain along the convex polygon through the points where it crosses the edges of
                # the domain, fanned out in triangles from its first vertex by their angle about the normal of the plane
                edges_verts_dist = ((domain_corners_pos - geom_pos) @ normal)[domain_edges]
                edges_verts_pos = domain_corners_pos[domain_edges]
                is_crossed = (edges_verts_dist[:, 0] < 0.0) != (edges_verts_dist[:, 1] < 0.0)
                edges_dist, edges_pos = edges_verts_dist[is_crossed], edges_verts_pos[is_crossed]
                edges_param = edges_dist[:, :1] / (edges_dist[:, :1] - edges_dist[:, 1:])
                polygon_pos = np.unique(edges_pos[:, 0] + edges_param * (edges_pos[:, 1] - edges_pos[:, 0]), axis=0)
                polygon_offsets = polygon_pos - polygon_pos.mean(axis=0)
                tangent = polygon_offsets[0] / np.linalg.norm(polygon_offsets[0])
                polygon_pos = polygon_pos[
                    np.argsort(
                        np.arctan2(polygon_offsets @ np.cross(normal, tangent), polygon_offsets @ tangent),
                        kind="stable",
                    )
                ]
                plane_tris_verts_pos = np.stack(
                    [
                        np.broadcast_to(polygon_pos[0], (len(polygon_pos) - 2, 3)),
                        polygon_pos[1:-1],
                        polygon_pos[2:],
                    ],
                    axis=1,
                )
                faces_tris_verts_pos = clip_triangles(
                    domain_faces_verts_pos, (domain_faces_verts_pos - geom_pos) @ normal
                )
                faces_tris_verts_pos = clip_triangles(
                    faces_tris_verts_pos,
                    -self.sph_solver.support_radius - (faces_tris_verts_pos - geom_pos) @ normal,
                )
                tris_verts_pos = np.concatenate([plane_tris_verts_pos, faces_tris_verts_pos])
                tri_idx, corners_bary = split_triangles(tris_verts_pos, self.sph_solver.particle_size)
                segments_corners_parts.append(
                    gu.inv_transform_by_trans_quat(corners_bary @ tris_verts_pos[tri_idx], geom_pos, geom_quat)
                )
                segments_geom_idx_parts.append(np.full(len(tri_idx), geom.idx, dtype=gs.np_int))
        rigid_segments_geom_idx = np.concatenate(segments_geom_idx_parts)
        rigid_segments_corners = np.concatenate(segments_corners_parts)
        has_rigid_sph = has_rigid_sph and len(rigid_segments_geom_idx) > 0

        # Radon's 7-point rule on the triangle, exact up to degree 5
        sqrt_15 = math.sqrt(15.0)
        a_1, b_1 = (9.0 - 2.0 * sqrt_15) / 21.0, (6.0 + sqrt_15) / 21.0
        a_2, b_2 = (9.0 + 2.0 * sqrt_15) / 21.0, (6.0 - sqrt_15) / 21.0
        quad_bary = np.array(
            [
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [a_1, b_1, b_1],
                [b_1, a_1, b_1],
                [b_1, b_1, a_1],
                [a_2, b_2, b_2],
                [b_2, a_2, b_2],
                [b_2, b_2, a_2],
            ],
            dtype=gs.np_float,
        )
        quad_weight = np.array(
            [9.0 / 40.0, *[(155.0 + sqrt_15) / 1200.0] * 3, *[(155.0 - sqrt_15) / 1200.0] * 3], dtype=gs.np_float
        )

        self.sph_boundaries = SPHBoundaries(
            sizes=dict(
                particles=len(fem_segments_tri_idx) + len(rigid_segments_geom_idx),
                envs=self.sim._B,
                quad_points=len(quad_weight),
                fem_segments=len(fem_segments_tri_idx),
                surfaces=len(fem_tris_verts_pos),
                rigid_segments=len(rigid_segments_geom_idx),
                rigid_links=self.rigid_solver.n_links,
                rigid_geoms=self.rigid_solver.n_geoms,
                exact_sum_slots=self.rigid_solver.n_links,
                exact_sum_bins=3,
                **hash_grid_sizes(math.prod(self.sph_solver.hash_grid_res)),
            ),
            values=dict(
                hash_grid=dict(cell_size=self.sph_solver.hash_grid_cell_size, grid_res=self.sph_solver.hash_grid_res),
                rho_rest=max([entity.material.rho for entity in self.sph_solver.entities]),
            ),
            env_first={
                (
                    SPHBoundaries.segments,
                    SPHBoundaries.hash_grid,
                    SPHBoundaries.segments_impulse,
                    SPHBoundaries.geoms_is_active,
                    SPHBoundaries.segments_is_wet,
                    SPHBoundaries.links_added_mass,
                ): False
            },
            unused=dict(fem_sph=not has_fem_sph, rigid_sph=not has_rigid_sph),
        )
        self.sph_boundaries.quad_bary.from_numpy(quad_bary)
        fem_tris_entity_idx = np.repeat(
            np.arange(len(self.fem_solver.entities)),
            [len(entity.surface_triangles) for entity in self.fem_solver.entities],
        )
        segments_boundary_idx = np.concatenate(
            [
                self.rigid_solver.n_geoms + fem_tris_entity_idx[fem_segments_tri_idx],
                rigid_segments_geom_idx,
            ]
        )
        self.sph_boundaries.segments_boundary_idx.from_numpy(segments_boundary_idx.astype(gs.np_int))
        self.sph_boundaries.quad_weight.from_numpy(quad_weight)
        if has_fem_sph:
            self.sph_boundaries.fem_segments_tri_idx.from_numpy(fem_segments_tri_idx.astype(gs.np_int))
            self.sph_boundaries.fem_segments_corners_bary.from_numpy(fem_segments_corners_bary.astype(gs.np_float))
            self.sph_boundaries.tris_segment_start.from_numpy(tris_segment_start.astype(gs.np_int))
        if has_rigid_sph:
            self.sph_boundaries.rigid_segments_geom_idx.from_numpy(rigid_segments_geom_idx)
            self.sph_boundaries.rigid_segments_corners.from_numpy(rigid_segments_corners.astype(gs.np_float))
            # Units from 2^-40 kg up to 2^24 kg, 2^32 apart (see ExactSums)
            self.sph_boundaries.links_added_mass_sums.bins_unit.from_numpy(
                2.0 ** np.array([-40.0, -8.0, 24.0], dtype=gs.np_float)
            )
            self.sph_boundaries.geoms_is_active.from_numpy(geoms_is_active)

        # The liquid sampled inside a coupled body, or closer to its surface than the half particle size at which liquid
        # rests against a wall, would start within its boundary, which pushes it out at once at any speed, so it is
        # removed. The surfaces of the bodies at rest form one triangle soup, signed by its winding number.
        # The open surfaces of the terrains and the planes bound no volume of liquid, the samplers placing the liquid
        # that rests on them at the half particle size of the removal. The environments sharing the same active geoms
        # are handled together.
        rigid_tris_is_closed = ~np.isin(
            np.repeat([geom.type for geom in geoms], [len(geom.init_faces) for geom in geoms]),
            (gs.GEOM_TYPE.TERRAIN, gs.GEOM_TYPE.PLANE),
        )
        particles_pos = qd_to_numpy(self.sph_solver.sph_state.particles.pos, transpose=True)[0]
        is_inside = np.zeros((len(particles_pos), self.sim._B), dtype=bool)
        # The variants of a heterogeneous entity may start at other poses in the environments where they are active, so
        # each group of environments takes the poses of its first one
        geoms_pos_envs = qd_to_numpy(self.rigid_solver.dyn_state.geoms.pos, transpose=True)
        geoms_quat_envs = qd_to_numpy(self.rigid_solver.dyn_state.geoms.quat, transpose=True)
        tris_is_active, envs_tris_group = np.unique(geoms_is_active[rigid_tris_geom_idx].T, axis=0, return_inverse=True)
        for i_group, tris_is_active_group in enumerate(tris_is_active):
            i_env = np.flatnonzero(envs_tris_group.reshape(-1) == i_group)[0]
            group_tris_verts_pos_world = gu.transform_by_trans_quat(
                rigid_tris_verts_pos,
                geoms_pos_envs[i_env, rigid_tris_geom_idx, None],
                geoms_quat_envs[i_env, rigid_tris_geom_idx, None],
            )
            tris_verts_pos = np.concatenate(
                [fem_tris_verts_pos, group_tris_verts_pos_world[tris_is_active_group & rigid_tris_is_closed]]
            )
            if len(tris_verts_pos) == 0:
                continue
            particles_dist, *_ = igl.signed_distance(
                particles_pos.astype(np.float64),
                tris_verts_pos.reshape((-1, 3)).astype(np.float64),
                np.arange(tris_verts_pos.size // 3, dtype=np.int64).reshape((-1, 3)),
                sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
            )
            is_inside[:, envs_tris_group.reshape(-1) == i_group] = (
                particles_dist < 0.5 * self.sph_solver.particle_size
            )[:, None]
        for entity in self.sph_solver.entities:
            n_inside = np.count_nonzero(
                is_inside[entity.particle_start : entity.particle_start + entity.n_particles], axis=0
            ).max()
            if n_inside > 0:
                gs.logger.warning(
                    f"{n_inside} particles of the SPH entity {entity.uid} lie inside bodies coupled with the liquid, "
                    f"which displace them: they are removed, a volume of {n_inside * self.sph_solver.particle_volume:.3g} "
                    "m^3."
                )
        if is_inside.any():
            particles_is_active = qd_to_numpy(self.sph_solver.sph_state.particles.is_active)
            particles_is_active[is_inside] = False
            self.sph_solver.sph_state.particles.is_active.from_numpy(particles_is_active)
        return has_rigid_sph

    def reset(self, envs_idx=None) -> None:
        # The MPM particles keep the rigid normals they saw in their state, which 'set_state' writes back before the
        # restart, so the restart leaves them as restored.
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------ PBD particles on links -------------------------------
    # ------------------------------------------------------------------------------------

    def attach_pbd_particles_to_link(self, particles_idx: torch.Tensor, envs_idx: torch.Tensor, link_idx: int) -> None:
        """Animate some PBD particles by a rigid link, keeping their current offset to it."""
        kernel_attach_pbd_to_rigid_link(
            particles_idx,
            envs_idx,
            link_idx,
            self.pbd_solver.pbd_state,
            self.rigid_solver.dyn_state.links,
            self.legacy_coupler_state,
        )

    def detach_pbd_particles(self, particles_idx: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Detach some PBD particles from the links animating them."""
        kernel_detach_pbd_from_rigid_link(particles_idx, envs_idx, self.legacy_coupler_state)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def preprocess(self) -> None:
        if self.legacy_coupler_config.has_fem_sph or self.legacy_coupler_config.has_rigid_sph:
            kernel_place_sph_boundary_segments(
                self.fem_solver.fem_state,
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.sph_boundaries,
                self.fem_solver.fem_info,
                self.sph_solver.sph_info,
                self.rigid_solver.dyn_info.geoms,
                self.legacy_coupler_config,
            )

        # preprocess for MPM CPIC
        if self.legacy_coupler_config.has_rigid_mpm and self.mpm_solver.enable_CPIC:
            kernel_mpm_surface_to_particle(
                self.mpm_solver.mpm_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.rigid_solver.collider.collider_config,
            )

    def couple(self) -> None:
        # MPM <-> all others
        if self.mpm_solver.is_active:
            kernel_mpm_grid_op(
                self.sim.sim_state,
                self.mpm_solver.mpm_state,
                self.sph_solver.sph_state,
                self.pbd_solver.pbd_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.mpm_solver.mpm_info,
                self.sph_solver.sph_info,
                self.pbd_solver.pbd_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.legacy_coupler_info,
                self.mpm_solver.mpm_config,
                self.legacy_coupler_config,
                self.rigid_solver.collider.collider_config,
                self.mpm_solver.force_fields,
                is_backward=False,
            )

        # SPH <-> FEM and Rigid, ahead of the FEM vertices taking their share
        if self.legacy_coupler_config.has_fem_sph or self.legacy_coupler_config.has_rigid_sph:
            kernel_sph_boundaries_impulse(
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.links,
                self.sph_boundaries,
                self.sph_solver.sph_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_info.links,
                self.legacy_coupler_info,
                self.legacy_coupler_config,
            )

        # PBD <-> Rigid
        if self.legacy_coupler_config.has_rigid_pbd:
            kernel_pbd_rigid_collide(
                self.pbd_solver.pbd_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.pbd_solver.pbd_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.rigid_solver.collider.collider_config,
            )

            # 1-way: animate particles by links, over the substep interval and no faster than the clamp allows.
            kernel_pbd_rigid_solve_animate_particles_by_link(
                self.pbd_solver.pbd_state,
                self.rigid_solver.dyn_state.links,
                self.legacy_coupler_state,
                self.legacy_coupler_info,
            )

        if self.fem_solver.is_active:
            kernel_fem_surface_force(
                self.fem_solver.fem_state,
                self.mpm_solver.mpm_state,
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.sph_boundaries,
                self.fem_solver.fem_info,
                self.mpm_solver.mpm_info,
                self.sph_solver.sph_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.legacy_coupler_info,
                self.legacy_coupler_config,
                self.rigid_solver.collider.collider_config,
            )

    @property
    def is_differentiable(self) -> bool:
        """
        Whether the coupler supports a scene with `requires_grad` (see 'Solver.is_differentiable').

        The couplings of the FEM solver with the rigid, MPM and SPH solvers carry no gradient: they overwrite the
        velocity of the surface vertices of the FEM entities in place, which the reverse pass cannot undo.
        """
        return not (
            self.legacy_coupler_config.has_rigid_fem
            or self.legacy_coupler_config.has_fem_mpm
            or self.legacy_coupler_config.has_fem_sph
        )

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_reset_mpm_rigid_normal(envs_idx: qd.types.ndarray(), mpm_state: MPMState):
    for i_p, i_b_ in qd.ndrange(
        mpm_state.particles_cpic.rigid_normal.shape[0],
        envs_idx.shape[0],
        axes=qd.static(eoa(mpm_state.particles_cpic.rigid_normal)),
    ):
        i_b = envs_idx[i_b_]
        for i_g in range(mpm_state.particles_cpic.rigid_normal.shape[1]):
            mpm_state.particles_cpic.rigid_normal[i_p, i_g, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_rigid_geom_influence(
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    collider_config: qd.template(),
    skip_weak_normal: qd.template(),
):
    """
    Return the signed distance of a point to a rigid geom, the influence of the geom there, and its normal there.

    With `skip_weak_normal`, the normal is left zero where the influence is too weak to matter (see
    'LegacyCouplerInfo'), which spares its evaluation.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)

    # bigger coup_softness implies that the coupling influence extends further away from the object.
    softness = qd.max(legacy_coupler_info.softness_min[None], geoms_info.coup_softness[i_g])
    influence = qd.min(qd.exp(-signed_dist / softness), 1)

    normal_rigid = qd.Vector.zero(gs.qd_float, 3)
    is_normal_needed = True
    if qd.static(skip_weak_normal):
        is_normal_needed = influence > legacy_coupler_info.influence_min[None]
    if is_normal_needed:
        normal_rigid = sdf.sdf_func_normal_world(
            i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
        )
    return signed_dist, influence, normal_rigid


@qd.func
def func_point_vel_after_collisions_with_rigid_geoms(
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    mpm_state: MPMState,
    geoms_state: GeomsState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    mpm_config: qd.template(),
    collider_config: qd.template(),
    is_backward: qd.template(),
):
    """
    Return the velocity of a cell of the MPM grid of the given mass once it collided with every rigid geom that couples.

    The momentum change of each collision pushes the link of the geom (see func_add_external_wrench), exactly with
    `is_deterministic` (see MPMConfig), except in the variant autodiff reverses (`is_backward`).
    """
    for i_g in range(geoms_info.needs_coup.shape[0]):
        if geoms_info.needs_coup[i_g]:
            _, influence, normal_rigid = func_rigid_geom_influence(
                i_g,
                i_b,
                pos_world,
                geoms_state,
                geoms_info,
                rigid_info,
                sdf_info,
                legacy_coupler_info,
                collider_config,
                skip_weak_normal=True,
            )
            if influence > legacy_coupler_info.influence_min[None]:
                vel, force, is_colliding = func_point_vel_after_two_way_collision_with_rigid_geom(
                    i_g, i_b, pos_world, vel, mass, normal_rigid, influence, links_state, geoms_info, rigid_info
                )
                if is_colliding:
                    func_add_external_wrench(
                        geoms_info.link_idx[i_g],
                        i_b,
                        pos_world,
                        force,
                        torque=qd.Vector.zero(gs.qd_float, 3),
                        links_state=links_state,
                        is_exact=mpm_config.is_deterministic and not is_backward,
                    )
    return vel


@qd.func
def func_mpm_cell_vel_after_exchange_with_particles(
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    half_size: float,
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_is_moved: qd.Tensor,
    particles_mass: qd.Tensor,
    hash_grid: SpatialHashGrid,
    n_cells: int,
):
    """
    Return the velocity of a cell of the MPM grid once it exchanged momentum with the particles of a solver it holds.

    The cell takes the mean velocity of the sorted particles within a cube of half side `half_size` around its position
    `pos`, and those that `particles_is_moved` selects take the opposite momentum (see 'func_add_particles_momentum').
    A cell holding no particle keeps its velocity.
    """
    particles_vel_mean, n_particles = func_get_particles_mean_vel(
        i_b, pos, particles_pos, particles_vel, hash_grid, half_size, n_cells
    )
    if n_particles > 0:
        particles_dmv = mass * (vel - particles_vel_mean)
        func_add_particles_momentum(
            i_b,
            pos,
            particles_dmv,
            particles_pos,
            particles_vel,
            particles_is_moved,
            particles_mass,
            hash_grid,
            half_size,
            n_cells,
        )
        vel = particles_vel_mean
    return vel


@reversed_with(is_backward=True)
@qd.kernel
def kernel_mpm_grid_op(
    sim_state: SimulatorState,
    mpm_state: MPMState,
    sph_state: SPHState,
    pbd_state: PBDState,
    geoms_state: GeomsState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    sph_info: SPHInfo,
    pbd_info: PBDInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    mpm_config: qd.template(),
    legacy_coupler_config: qd.template(),
    collider_config: qd.template(),
    force_fields: qd.template(),
    is_backward: qd.template(),
):
    """
    Update the velocity of the cells of the MPM grid, which the other solvers exchange momentum with.

    The reactions of the rigid geoms add exactly to the coupling forces of their links with `is_deterministic` (see
    MPMConfig), except in the variant autodiff reverses (`is_backward`).
    """
    for i_x, i_y, i_z, i_b in qd.ndrange(*mpm_state.grid.mass.shape):
        mass_cell = mpm_state.grid.mass[i_x, i_y, i_z, i_b]
        if mass_cell > rigid_info.EPS[None]:
            #################### MPM grid op ####################
            # Momentum to velocity, then gravity
            vel_mpm = (1 / mass_cell) * mpm_state.grid.vel_in[i_x, i_y, i_z, i_b]
            vel_mpm += mpm_info.substep_dt[None] * mpm_info.gravity[i_b]

            pos = (qd.Vector([i_x, i_y, i_z]) + mpm_info.grid_offset[None]) * mpm_info.dx[None]
            mass_mpm = mass_cell / mpm_info.particle_volume_scale[None]

            # External force fields, each one integrated on its own
            for i_ff in qd.static(range(len(force_fields.fields))):
                acc = force_fields.fields[i_ff].get_acc(pos, vel_mpm, sim_state.t[i_b], -1)
                vel_mpm += acc * mpm_info.substep_dt[None]

            #################### MPM <-> Rigid ####################
            if qd.static(legacy_coupler_config.has_rigid_mpm):
                vel_mpm = func_point_vel_after_collisions_with_rigid_geoms(
                    i_b,
                    pos,
                    vel_mpm,
                    mass_mpm,
                    mpm_state,
                    geoms_state,
                    links_state,
                    mpm_info,
                    geoms_info,
                    rigid_info,
                    sdf_info,
                    legacy_coupler_info,
                    mpm_config,
                    collider_config,
                    is_backward,
                )

            #################### MPM <-> SPH, PBD ####################
            # Of the PBD particles, the free ones alone take momentum
            half_size = 0.5 * mpm_info.dx[None]
            if qd.static(legacy_coupler_config.has_mpm_sph):
                vel_mpm = func_mpm_cell_vel_after_exchange_with_particles(
                    i_b,
                    pos,
                    vel_mpm,
                    mass_mpm,
                    half_size,
                    sph_state.particles_reordered.pos,
                    sph_state.particles_reordered.vel,
                    sph_state.particles_reordered.is_active,
                    sph_info.particles.mass,
                    sph_state.hash_grid,
                    legacy_coupler_info.mpm_sph_stencil_size[None],
                )
            if qd.static(legacy_coupler_config.has_mpm_pbd):
                vel_mpm = func_mpm_cell_vel_after_exchange_with_particles(
                    i_b,
                    pos,
                    vel_mpm,
                    mass_mpm,
                    half_size,
                    pbd_state.particles_reordered.pos,
                    pbd_state.particles_reordered.vel,
                    pbd_state.particles_reordered.is_free,
                    pbd_info.particles.mass,
                    pbd_state.hash_grid,
                    legacy_coupler_info.mpm_pbd_stencil_size[None],
                )

            #################### MPM boundary ####################
            _, vel_mpm = func_impose_cube_boundary(pos, vel_mpm, mpm_info.boundary)
            mpm_state.grid.vel_out[i_x, i_y, i_z, i_b] = vel_mpm


@qd.kernel
def kernel_mpm_surface_to_particle(
    mpm_state: MPMState,
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """Update the normal of the rigid geoms each active MPM particle sees, as long as it stays on the same side."""
    for i_p, i_b in qd.ndrange(
        *mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles_cpic.rigid_normal))
    ):
        if mpm_state.particles.is_active[i_p, i_b]:
            for i_g in range(mpm_state.particles_cpic.rigid_normal.shape[1]):
                if geoms_info.needs_coup[i_g]:
                    sdf_normal = sdf.sdf_func_normal_world(
                        i_g,
                        i_b,
                        mpm_state.particles.pos[i_p, i_b],
                        geoms_state,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        collider_config,
                    )
                    # The normal is updated as long as the particle stays on the same side of the geom
                    if sdf_normal.dot(mpm_state.particles_cpic.rigid_normal[i_p, i_g, i_b]) >= 0:
                        mpm_state.particles_cpic.rigid_normal[i_p, i_g, i_b] = sdf_normal


@qd.kernel
def kernel_fem_surface_force(
    fem_state: FEMState,
    mpm_state: MPMState,
    sph_state: SPHState,
    geoms_state: GeomsState,
    links_state: LinksState,
    sph_boundaries: SPHBoundaries,
    fem_info: FEMInfo,
    mpm_info: MPMInfo,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    legacy_coupler_config: qd.template(),
    collider_config: qd.template(),
):
    """Update the velocity of the surface vertices of the FEM entities by their collisions with the other solvers."""
    # TODO: all collisions are on vertices instead of surface and edge
    # Each surface vertex applies the couplings of its incident triangles in increasing triangle index, which
    # reproduces a serial sweep over the triangles without a write race on the vertices they share.
    for i_sv, i_b in qd.ndrange(
        fem_info.surface_vertices.shape[0], fem_state.vertices.pos.shape[1], axes=qd.static(eoa(fem_state.vertices.vel))
    ):
        i_v = fem_info.surface_vertices[i_sv]
        pos = fem_state.vertices.pos[i_v, i_b]
        vel_fem_sv = fem_state.vertices.vel_next[i_v, i_b]
        i_tri_start = fem_info.surface_vertices_tri_start[i_sv]
        n_tris = fem_info.surface_vertices_tri_start[i_sv + 1] - i_tri_start

        # FEM <-> Rigid
        # NOTE: collision only on surface vertices
        if qd.static(legacy_coupler_config.has_rigid_fem):
            # The influence, normal and velocity of a rigid geom at the vertex are the same for all its incident
            # triangles, so they are evaluated once, and the momentum the vertex transfers is applied once.
            for i_g in range(geoms_info.needs_coup.shape[0]):
                if geoms_info.needs_coup[i_g]:
                    _, influence, normal_rigid = func_rigid_geom_influence(
                        i_g,
                        i_b,
                        pos,
                        geoms_state,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        legacy_coupler_info,
                        collider_config,
                        skip_weak_normal=True,
                    )
                    if influence > legacy_coupler_info.influence_min[None]:
                        vel_rigid = func_vel_at_point(geoms_info.link_idx[i_g], i_b, pos, links_state)
                        delta_mv = qd.Vector.zero(gs.qd_float, 3)
                        is_colliding_any = False
                        for i_s_ in range(n_tris):
                            mass = (
                                fem_info.surface_vertices_tri_mass_scaled[i_tri_start + i_s_] / fem_info.vol_scale[None]
                            )
                            vel_new, is_colliding = func_point_vel_after_collision_with_rigid_geom(
                                i_g, vel_fem_sv, vel_rigid, normal_rigid, influence, geoms_info, rigid_info
                            )
                            if is_colliding:
                                # assume element mass uniformly distributed to vertices
                                delta_mv = delta_mv + (mass / 3.0) * (vel_new - vel_fem_sv)
                                is_colliding_any = True
                            vel_fem_sv = vel_new
                        if is_colliding_any:
                            func_add_external_wrench(
                                geoms_info.link_idx[i_g],
                                i_b,
                                pos,
                                force=-delta_mv / rigid_info.substep_dt[None],
                                torque=qd.Vector.zero(gs.qd_float, 3),
                                links_state=links_state,
                                is_exact=False,
                            )

        if qd.static(legacy_coupler_config.has_fem_mpm):
            for i_s_ in range(n_tris):
                i_s = fem_info.surface_vertices_tri_idx[i_tri_start + i_s_]
                mass = fem_info.surface_vertices_tri_mass_scaled[i_tri_start + i_s_] / fem_info.vol_scale[None]

                tri2v = fem_info.surfaces.tri2v[i_s]
                surface_normal = triangle_face_normal(
                    fem_state.vertices.pos[tri2v[0], i_b],
                    fem_state.vertices.pos[tri2v[1], i_b],
                    fem_state.vertices.pos[tri2v[2], i_b],
                    rigid_info.EPS[None],
                )

                # FEM <-> MPM (interact with MPM grid instead of particles)
                # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
                #       however, this function is called after mpm boundary conditions.
                if qd.static(legacy_coupler_config.has_fem_mpm):
                    mass_fem_sv = mass / 4.0  # assume element mass uniformly distributed

                    # Following the transfer of the MPM particles to the grid
                    vel_mpm = qd.Vector.zero(gs.qd_float, 3)
                    mass_mpm = gs.qd_float(0.0)
                    mpm_base, _, mpm_w = func_quadratic_bspline_stencil(pos, mpm_info)
                    new_vel_fem_sv = vel_fem_sv
                    grid_shape = mpm_state.grid.mass.shape
                    grid_res = qd.Vector([grid_shape[0], grid_shape[1], grid_shape[2]])
                    dx = mpm_info.dx[None]
                    for mpm_offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                        cell = mpm_base - mpm_info.grid_offset[None] + mpm_offset
                        # A vertex outside the domain of the grid exchanges nothing with the cells the grid lacks
                        is_cell_in_grid = (cell >= 0).all() and (cell < grid_res).all()
                        mpm_grid_mass = gs.qd_float(0.0)
                        if is_cell_in_grid:
                            mpm_grid_mass = (
                                mpm_state.grid.mass[cell[0], cell[1], cell[2], i_b]
                                / mpm_info.particle_volume_scale[None]
                            )

                        mpm_weight = gs.qd_float(1.0)
                        for d in qd.static(range(3)):
                            mpm_weight *= mpm_w[mpm_offset[d], d]

                        # FEM -> MPM, within one cell of the surface
                        mpm_grid_pos = (cell + mpm_info.grid_offset[None]) * dx
                        signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                        if is_cell_in_grid and signed_dist <= dx:
                            vel_out = mpm_state.grid.vel_out[cell[0], cell[1], cell[2], i_b]
                            mass_mpm_at_cell = mpm_weight * mpm_grid_mass
                            vel_mpm += mpm_weight * vel_out
                            mass_mpm += mass_mpm_at_cell

                            if mass_mpm_at_cell > rigid_info.EPS[None]:
                                # The full mass of the cell, whereas mass_mpm_at_cell carries the weight of the vertex
                                delta_mpm_vel_at_cell = (vel_fem_sv * mpm_weight - vel_out) * (
                                    mpm_grid_mass / mass_fem_sv
                                )
                                qd.atomic_add(
                                    mpm_state.grid.vel_out[cell[0], cell[1], cell[2], i_b], delta_mpm_vel_at_cell
                                )
                                new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                    # MPM -> FEM
                    if mass_mpm > rigid_info.EPS[None]:
                        vel_fem_sv = new_vel_fem_sv

        # FEM <-> SPH, the vertex taking the impulses of the segments of its triangles by the barycentric weight of their
        # centroids
        if qd.static(legacy_coupler_config.has_fem_sph):
            for i_s_ in range(n_tris):
                i_s = fem_info.surface_vertices_tri_idx[i_tri_start + i_s_]
                tri2v = fem_info.surfaces.tri2v[i_s]
                for i_c in qd.static(range(3)):
                    if tri2v[i_c] == i_v:
                        for i_sg in range(
                            sph_boundaries.tris_segment_start[i_s], sph_boundaries.tris_segment_start[i_s + 1]
                        ):
                            corners_bary = sph_boundaries.fem_segments_corners_bary[i_sg]
                            centroid_bary = (corners_bary[0, i_c] + corners_bary[1, i_c] + corners_bary[2, i_c]) / 3.0
                            vel_fem_sv = (
                                vel_fem_sv
                                + centroid_bary
                                * sph_boundaries.segments_impulse[i_sg, i_b]
                                * fem_info.vertices.mass_inv[i_v]
                            )

        fem_state.vertices.vel_next[i_v, i_b] = vel_fem_sv


@qd.func
def func_pbd_collide_with_rigid_geom(
    i_p: int,
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    pbd_state: PBDState,
    geoms_state: GeomsState,
    links_state: LinksState,
    pbd_info: PBDInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """
    Push a sorted particle penetrating a rigid geom back to its surface, and apply the reaction to the geom.

    The particle takes the velocity of its displacement over the substep, friction left out.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)
    contact_normal = sdf.sdf_func_normal_world(
        i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
    )
    new_pos = pos_world
    new_vel = vel
    particle_radius = pbd_info.particle_size[None] / 2
    if signed_dist < particle_radius:
        new_pos = pos_world + contact_normal * (particle_radius - signed_dist)
        new_vel = (new_pos - pbd_state.particles_reordered.ipos[i_p, i_b]) / pbd_info.substep_dt[None]

        delta_mv = mass * (new_vel - vel)
        force = -delta_mv / rigid_info.substep_dt[None]
        func_add_external_wrench(
            geoms_info.link_idx[i_g],
            i_b,
            pos_world,
            force,
            torque=qd.Vector.zero(gs.qd_float, 3),
            links_state=links_state,
            is_exact=False,
        )

    return new_pos, new_vel


@qd.kernel
def kernel_pbd_rigid_collide(
    pbd_state: PBDState,
    geoms_state: GeomsState,
    links_state: LinksState,
    pbd_info: PBDInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """Collide the sorted active PBD particles with the rigid geoms that couple."""
    for i_p, i_b in qd.ndrange(
        *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.pos))
    ):
        if pbd_state.particles_reordered.is_active[i_p, i_b]:
            i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
            # A particle fixed in place or to a link collides as a free one, which is physically implausible
            for i_g in range(geoms_info.needs_coup.shape[0]):
                if geoms_info.needs_coup[i_g]:
                    pos, vel = func_pbd_collide_with_rigid_geom(
                        i_p,
                        i_g,
                        i_b,
                        pbd_state.particles_reordered.pos[i_p, i_b],
                        pbd_state.particles_reordered.vel[i_p, i_b],
                        pbd_info.particles.mass[i_o],
                        pbd_state,
                        geoms_state,
                        links_state,
                        pbd_info,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        collider_config,
                    )
                    pbd_state.particles_reordered.pos[i_p, i_b] = pos
                    pbd_state.particles_reordered.vel[i_p, i_b] = vel


@qd.kernel
def kernel_attach_pbd_to_rigid_link(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: int,
    pbd_state: PBDState,
    links_state: LinksState,
    legacy_coupler_state: LegacyCouplerState,
):
    """Set some particles in some environments to be animated by a link, keeping their current offset to it."""
    for i_p_, i_b_ in qd.ndrange(
        particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(pbd_state.particles.is_free))
    ):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        local_pos = qd_inv_transform_by_trans_quat(
            pbd_state.particles.pos[i_p, i_b], links_state.pos[link_idx, i_b], links_state.quat[link_idx, i_b]
        )
        pbd_state.particles.is_free[i_p, i_b] = False
        legacy_coupler_state.pbd_attach_link_idx[i_p, i_b] = link_idx
        legacy_coupler_state.pbd_attach_local_pos[i_p, i_b] = local_pos


@qd.kernel
def kernel_detach_pbd_from_rigid_link(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), legacy_coupler_state: LegacyCouplerState
):
    """Detach some particles from the links animating them."""
    for i_p_, i_b_ in qd.ndrange(
        particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(legacy_coupler_state.pbd_attach_link_idx))
    ):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        legacy_coupler_state.pbd_attach_link_idx[i_p, i_b] = -1
        legacy_coupler_state.pbd_attach_local_pos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_pbd_rigid_solve_animate_particles_by_link(
    pbd_state: PBDState,
    links_state: LinksState,
    legacy_coupler_state: LegacyCouplerState,
    legacy_coupler_info: LegacyCouplerInfo,
):
    """
    Give every particle animated by a link the velocity that brings it to its target on the link.

    The target moves with the link, whose velocity the particle takes, plus the velocity correcting its position over
    the substep, at an inverse time step clamped to avoid large jerks. It runs after both the rigid and the PBD solvers
    updated, so the correction lags one substep behind.
    """
    for i_r, i_b in qd.ndrange(*pbd_state.particles.pos.shape, axes=qd.static(eoa(pbd_state.particles.pos))):
        i_p = pbd_state.hash_grid.particles_idx[i_r, i_b]
        i_l = -1
        if pbd_state.particles_reordered.is_active[i_r, i_b]:
            i_l = legacy_coupler_state.pbd_attach_link_idx[i_p, i_b]
        if i_l >= 0:
            target_world_pos = qd_transform_by_trans_quat(
                legacy_coupler_state.pbd_attach_local_pos[i_p, i_b],
                links_state.pos[i_l, i_b],
                links_state.quat[i_l, i_b],
            )
            link_com_in_world = links_state.root_COM[i_l, i_b] + links_state.i_pos[i_l, i_b]
            world_arm = target_world_pos - link_com_in_world
            target_world_vel = links_state.cd_vel[i_l, i_b] + links_state.cd_ang[i_l, i_b].cross(world_arm)

            pos_correction = target_world_pos - pbd_state.particles_reordered.pos[i_r, i_b]
            corrective_vel = pos_correction * legacy_coupler_info.animate_inv_dt[None]
            pbd_state.particles_reordered.vel[i_r, i_b] = corrective_vel + target_world_vel


@qd.kernel
def kernel_place_sph_boundary_segments(
    fem_state: FEMState,
    sph_state: SPHState,
    geoms_state: GeomsState,
    links_state: LinksState,
    sph_boundaries: SPHBoundaries,
    fem_info: FEMInfo,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    legacy_coupler_config: qd.template(),
):
    """
    Place the segments of the moving boundaries of the SPH liquid at the start of the substep, and sum up the boundaries
    around every SPH particle.

    The segments are sorted in their hash grid, which every particle reads to sum up the segments around it (see
    SPHBoundaryParticles).
    """
    for i_b in range(sph_boundaries.segments_extent_max.shape[0]):
        sph_boundaries.segments_extent_max[i_b] = 0.0

    n_fem_segments = 0
    if qd.static(legacy_coupler_config.has_fem_sph):
        n_fem_segments = sph_boundaries.fem_segments_tri_idx.shape[0]
    for i_sg, i_b in qd.ndrange(
        *sph_boundaries.segments.centroid.shape, axes=qd.static(eoa(sph_boundaries.segments.centroid))
    ):
        corners = qd.Matrix.zero(gs.qd_float, 3, 3)
        vel = qd.Vector.zero(gs.qd_float, 3)
        is_active = True
        if qd.static(legacy_coupler_config.has_fem_sph):
            if i_sg < n_fem_segments:
                i_t = sph_boundaries.fem_segments_tri_idx[i_sg]
                tri2v = fem_info.surfaces.tri2v[i_t]
                corners_bary = sph_boundaries.fem_segments_corners_bary[i_sg]
                verts_pos = qd.Matrix.zero(gs.qd_float, 3, 3)
                for i_c in qd.static(range(3)):
                    i_v = tri2v[i_c]
                    vert_pos = fem_state.vertices.pos[i_v, i_b]
                    for i_d in qd.static(range(3)):
                        verts_pos[i_c, i_d] = vert_pos[i_d]
                    # The segment moves with the point of the triangle at its centroid
                    centroid_bary = (corners_bary[0, i_c] + corners_bary[1, i_c] + corners_bary[2, i_c]) / 3.0
                    vel = vel + centroid_bary * fem_state.vertices.vel[i_v, i_b]
                corners = corners_bary @ verts_pos
        if qd.static(legacy_coupler_config.has_rigid_sph):
            if i_sg >= n_fem_segments:
                i_rs = i_sg - n_fem_segments
                i_g = sph_boundaries.rigid_segments_geom_idx[i_rs]
                is_active = sph_boundaries.geoms_is_active[i_g, i_b]
                geom_pos, geom_quat = geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                corners_local = sph_boundaries.rigid_segments_corners[i_rs]
                for i_c in qd.static(range(3)):
                    corner = qd_transform_by_trans_quat(
                        qd.Vector([corners_local[i_c, 0], corners_local[i_c, 1], corners_local[i_c, 2]]),
                        geom_pos,
                        geom_quat,
                    )
                    for i_d in qd.static(range(3)):
                        corners[i_c, i_d] = corner[i_d]
        corner_0 = qd.Vector([corners[0, 0], corners[0, 1], corners[0, 2]])
        corner_1 = qd.Vector([corners[1, 0], corners[1, 1], corners[1, 2]])
        corner_2 = qd.Vector([corners[2, 0], corners[2, 1], corners[2, 2]])
        # The surfaces wind counterclockwise about their outward normal, which points into the liquid
        area_normal = 0.5 * (corner_1 - corner_0).cross(corner_2 - corner_0)
        area = area_normal.norm()
        centroid = (corner_0 + corner_1 + corner_2) / 3.0
        if qd.static(legacy_coupler_config.has_rigid_sph):
            if i_sg >= n_fem_segments:
                i_g = sph_boundaries.rigid_segments_geom_idx[i_sg - n_fem_segments]
                i_l = geoms_info.link_idx[i_g]
                vel = func_vel_at_point(i_l, i_b, centroid, links_state)
        sph_boundaries.segments.corners[i_sg, i_b] = corners
        sph_boundaries.segments.normal[i_sg, i_b] = area_normal / qd.max(area, gs.EPS)
        sph_boundaries.segments.centroid[i_sg, i_b] = centroid
        sph_boundaries.segments.vel[i_sg, i_b] = vel
        sph_boundaries.segments.area[i_sg, i_b] = area
        sph_boundaries.segments.is_active[i_sg, i_b] = is_active
        # A deformed FEM segment may stretch beyond its extent at rest
        extent = qd.sqrt(
            qd.max((corner_0 - centroid).norm_sqr(), (corner_1 - centroid).norm_sqr(), (corner_2 - centroid).norm_sqr())
        )
        sph_boundaries.segments.extent[i_sg, i_b] = extent
        if is_active:
            qd.atomic_max(sph_boundaries.segments_extent_max[i_b], extent)

    func_hash_sort_particles(
        sph_boundaries.segments.centroid, sph_boundaries.segments.is_active, sph_boundaries.hash_grid
    )

    # The kernel volume of the solids behind the segments around each particle, its gradient and first moment, and the
    # rate at which the moving segments change it (see SPHBoundaryParticles). On GPU, the lanes of a subgroup take a
    # particle together, striding the segments of each cell around it, then sum their shares along a fixed tree.
    N_LANES = qd.static(32 if legacy_coupler_config.enable_cooperative_particles else 1)
    support_radius, kernel_norm = sph_info.support_radius[None], sph_info.kernel_norm[None]
    n_particles, n_envs = sph_state.particles.pos.shape
    # The environments run fastest, as the arrays keep them innermost
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        volume = gs.qd_float(0.0)
        volume_grad = qd.Vector.zero(gs.qd_float, 3)
        moment = qd.Matrix.zero(gs.qd_float, 3, 3)
        vel_flux = gs.qd_float(0.0)
        surface_dist = gs.qd_float(1e9)
        # The lowest and highest boundary of the segments around the particle
        boundary_lo = sph_boundaries.segments_boundary_idx.shape[0]
        boundary_hi = -1
        if sph_state.particles.is_active[i_p, i_b]:
            pos = sph_state.particles.pos[i_p, i_b]
            # The segments overlapping the support radius of the particle have their centroid within it plus their extent
            reach = support_radius + sph_boundaries.segments_extent_max[i_b]
            cell_lo = func_hash_pos_to_grid(pos - reach, sph_boundaries.hash_grid)
            cell_hi = func_hash_pos_to_grid(pos + reach, sph_boundaries.hash_grid)
            base = func_hash_wrap_cell(cell_lo, sph_boundaries.hash_grid)
            n_cells = qd.min(cell_hi - cell_lo + 1, sph_boundaries.hash_grid.grid_res[None])
            for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
                i_slot = func_hash_cell_to_slot(base + offset, sph_boundaries.hash_grid)
                slot_start = sph_boundaries.hash_grid.slot_start[i_slot, i_b]
                slot_size = sph_boundaries.hash_grid.slot_size[i_slot, i_b]
                for i_q_ in range((slot_size + N_LANES - 1) // N_LANES):
                    i_q = i_q_ * N_LANES + i_lane
                    if i_q < slot_size:
                        i_sg = sph_boundaries.hash_grid.particles_idx[slot_start + i_q, i_b]
                        seg_volume, seg_volume_grad, seg_moment = func_segment_volume(
                            pos,
                            sph_boundaries.segments.corners[i_sg, i_b],
                            sph_boundaries.segments.normal[i_sg, i_b],
                            sph_boundaries.segments.area[i_sg, i_b],
                            sph_boundaries.segments.extent[i_sg, i_b],
                            sph_boundaries.quad_bary,
                            sph_boundaries.quad_weight,
                            support_radius,
                            kernel_norm,
                        )
                        volume = volume + seg_volume
                        volume_grad = volume_grad + seg_volume_grad
                        moment = moment + seg_moment
                        vel_flux = vel_flux + sph_boundaries.segments.vel[i_sg, i_b].dot(seg_volume_grad)
                        surface_dist = qd.min(
                            surface_dist, func_segment_dist(pos, sph_boundaries.segments.corners[i_sg, i_b])
                        )
                        if seg_volume_grad.norm_sqr() > 0.0:
                            i_bd = sph_boundaries.segments_boundary_idx[i_sg]
                            boundary_lo = qd.min(boundary_lo, i_bd)
                            boundary_hi = qd.max(boundary_hi, i_bd)
        if qd.static(N_LANES > 1):
            # The lanes sum scalars alone
            volume = qd_block_sum(volume)
            vel_flux = qd_block_sum(vel_flux)
            surface_dist = qd_block_min(surface_dist)
            boundary_lo = qd_block_min(boundary_lo)
            boundary_hi = qd_block_max(boundary_hi)
            for i_d in qd.static(range(3)):
                volume_grad[i_d] = qd_block_sum(volume_grad[i_d])
                for i_e in qd.static(range(3)):
                    moment[i_d, i_e] = qd_block_sum(moment[i_d, i_e])
        # The divergence theorem counts the kernel at the particle itself out of the volume of a solid that encloses it,
        # which turns it negative
        if volume < -1e-3:
            volume = volume + 1.0
            surface_dist = -surface_dist
        volume = qd.min(qd.max(volume, 0.0), 1.0)
        if i_lane == 0:
            sph_state.boundary.volume[i_p, i_b] = volume
            sph_state.boundary.volume_grad[i_p, i_b] = volume_grad
            sph_state.boundary.moment[i_p, i_b] = moment
            sph_state.boundary.vel_flux[i_p, i_b] = vel_flux
            sph_state.boundary.surface_dist[i_p, i_b] = surface_dist
            sph_state.boundary.impulse_coef[i_p, i_b] = 0.0
            sph_state.boundary.impulse_gravity_coef[i_p, i_b] = 0.0
            sph_state.boundary.boundaries_idx[i_p, i_b] = qd.Vector([boundary_lo, -1, -1, -1], dt=gs.qd_int)
            sph_state.boundary.boundaries_impulse_coef[i_p, i_b] = qd.Vector.zero(gs.qd_float, 4)
            sph_state.boundary.has_several_boundaries[i_p, i_b] = boundary_hi > boundary_lo


@qd.func
def func_project_on_bounds(
    vel_free: qd.types.vector(3),
    normals: qd.types.matrix(4, 3),
    bounds: qd.types.vector(4),
    violation_tol: float,
    legacy_coupler_info: LegacyCouplerInfo,
    n_bounds: int,
):
    """
    Return the velocity nearest to a free velocity that meets at most 4 lower bounds on its components along normals,
    with the indices of the bounds met as equalities and their multipliers.

    Where no velocity meets all the bounds, as for a particle squeezed between two approaching boundaries, the bounds are
    all lowered by the least common slack that lets a velocity meet them, which violates each of the opposing bounds by
    the same amount. By duality, that slack is the largest weighted mean of the bounds over the non-negative weights
    that sum to one and cancel the weighted sum of their normals. The nearest velocity then meets as equalities the
    first set of at most 3 bounds whose multipliers are non-negative and whose solution meets all the others, the
    solution of a strictly convex problem being unique.
    """
    slack = gs.qd_float(0.0)
    for i_mask in range(16):
        # The subsets of at least 2 of the bounds, as the columns of the normals and a row of ones, unused ones padded
        mixing = qd.Matrix.zero(gs.qd_float, 4, 4)
        n_subset = 0
        for i_k in qd.static(range(4)):
            if (i_mask >> i_k) & 1 and i_k < n_bounds:
                for i_d in qd.static(range(3)):
                    mixing[i_d, i_k] = normals[i_k, i_d]
                mixing[3, i_k] = 1.0
                n_subset = n_subset + 1
        if n_subset >= 2 and i_mask < (1 << n_bounds):
            gram = mixing.transpose() @ mixing
            rhs = mixing.transpose() @ qd.Vector([0.0, 0.0, 0.0, 1.0], dt=gs.qd_float)
            for i_k in qd.static(range(4)):
                if not ((i_mask >> i_k) & 1):
                    gram[i_k, i_k] = 1.0
            if gram.determinant() > legacy_coupler_info.normals_independence_min[None]:
                weights = gram.inverse() @ rhs
                residual = (mixing @ weights - qd.Vector([0.0, 0.0, 0.0, 1.0], dt=gs.qd_float)).norm()
                if weights.min() >= 0.0 and residual < legacy_coupler_info.normals_independence_min[None]:
                    slack = qd.max(slack, weights.dot(bounds))
    bounds_relaxed = bounds - slack * qd.Vector([1.0, 1.0, 1.0, 1.0], dt=gs.qd_float)

    vel = vel_free
    set_best = qd.Vector([-1, -1, -1], dt=gs.qd_int)
    lambdas_best = qd.Vector.zero(gs.qd_float, 3)
    violation_best = gs.qd_float(0.0)
    is_found = False
    # The sets are taken as (i_k0 <= i_k1 <= i_k2) of distinct indices, a repeated index ending the set
    for i_k0 in range(n_bounds):
        for i_k1 in range(i_k0, n_bounds):
            for i_k2 in range(i_k1, n_bounds):
                n_set = 1 + qd.select(i_k1 > i_k0, 1, 0) + qd.select(i_k2 > i_k1, 1, 0)
                set_idx = qd.Vector([i_k0, i_k1, i_k2], dt=gs.qd_int)
                if i_k2 > i_k1 and i_k1 == i_k0:
                    n_set = 0
                elif i_k1 > i_k0 and i_k2 == i_k1:
                    set_idx = qd.Vector([i_k0, i_k1, -1], dt=gs.qd_int)
                elif n_set == 1:
                    set_idx = qd.Vector([i_k0, -1, -1], dt=gs.qd_int)
                if n_set > 0 and (not is_found or violation_best > violation_tol):
                    # The bounds of the set as columns, the unused ones left zero with identity equations
                    normals_set = qd.Matrix.zero(gs.qd_float, 3, 3)
                    bounds_set = qd.Vector.zero(gs.qd_float, 3)
                    for i_s in qd.static(range(3)):
                        if i_s < n_set:
                            i_k = set_idx[i_s]
                            for i_d in qd.static(range(3)):
                                normals_set[i_d, i_s] = normals[i_k, i_d]
                            bounds_set[i_s] = bounds_relaxed[i_k]
                    gram = normals_set.transpose() @ normals_set
                    rhs = bounds_set - normals_set.transpose() @ vel_free
                    for i_s in qd.static(range(3)):
                        if i_s >= n_set:
                            gram[i_s, i_s] = 1.0
                            rhs[i_s] = 0.0
                    # The normals are unit vectors, so the determinant of their Gram matrix measures their independence
                    if gram.determinant() > legacy_coupler_info.normals_independence_min[None]:
                        lambdas_set = gram.inverse() @ rhs
                        if lambdas_set.min() >= 0.0:
                            vel_set = vel_free + normals_set @ lambdas_set
                            violation = gs.qd_float(0.0)
                            for i_k in range(n_bounds):
                                normal_k = qd.Vector([normals[i_k, 0], normals[i_k, 1], normals[i_k, 2]])
                                violation = qd.max(violation, bounds_relaxed[i_k] - normal_k.dot(vel_set))
                            if not is_found or violation < violation_best:
                                is_found = True
                                violation_best = violation
                                vel = vel_set
                                set_best = set_idx
                                lambdas_best = lambdas_set
    return vel, set_best, lambdas_best


@qd.kernel
def kernel_sph_boundaries_impulse(
    sph_state: SPHState,
    links_state: LinksState,
    sph_boundaries: SPHBoundaries,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    legacy_coupler_config: qd.template(),
):
    """
    Compute the impulse of the SPH liquid on each segment of the moving boundaries over the substep.

    The impulse is the opposite of the one the segment gives the particles around it, so that the momentum of the
    liquid and of the boundaries is conserved. The links of the rigid geoms take the impulse of their segments as a
    coupling force, the fixed ones aside, while the FEM vertices take theirs in kernel_fem_surface_force.
    """
    support_radius, kernel_norm = sph_info.support_radius[None], sph_info.kernel_norm[None]
    n_fem_segments = 0
    if qd.static(legacy_coupler_config.has_fem_sph):
        n_fem_segments = sph_boundaries.fem_segments_tri_idx.shape[0]

    # A particle reaching the surface of a body loses the velocity that carries it into the body, relative to the
    # surface, the body taking the opposite impulse. The density solve holds the liquid off the body only where the
    # particle has enough liquid neighbors, which a particle at the free surface lacks. Each boundary near the particle
    # bounds the velocity along its own volume gradient, and the bounds are met jointly: a particle pressed between two
    # boundaries, as under a body resting on another, meets both, where their summed gradient cancels along the
    # direction between them. A particle inside a boundary is moved out to its surface, since the bound only stops it
    # from moving further in.
    N_BOUNDARIES = qd.static(4)
    support_radius, kernel_norm = sph_info.support_radius[None], sph_info.kernel_norm[None]
    for i_r, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.vel))
    ):
        if sph_state.particles_reordered.is_active[i_r, i_b]:
            i_p = sph_state.hash_grid.particles_idx[i_r, i_b]
            grad_norm = sph_state.boundary.volume_grad[i_p, i_b].norm()
            if sph_state.boundary.surface_dist[i_p, i_b] < 0.25 * sph_info.particle_size[None] and grad_norm > 0.0:
                pos = sph_state.particles_reordered.pos[i_r, i_b]
                vel_free = sph_state.particles_reordered.vel[i_r, i_b]
                # The kernel volume, its gradient, the rate the moving segments change it and the signed distance to
                # the surface of each boundary around the particle, those beyond the first 3 merged in the last slot.
                # A particle around a single boundary takes them from its sums over all the segments around it.
                boundaries_idx = sph_state.boundary.boundaries_idx[i_p, i_b]
                volumes = qd.Vector.zero(gs.qd_float, N_BOUNDARIES)
                volume_grads = qd.Matrix.zero(gs.qd_float, N_BOUNDARIES, 3)
                vel_fluxes = qd.Vector.zero(gs.qd_float, N_BOUNDARIES)
                surface_dists = qd.Vector([1e9] * N_BOUNDARIES, dt=gs.qd_float)
                volume_grad = sph_state.boundary.volume_grad[i_p, i_b]
                for i_d in qd.static(range(3)):
                    volume_grads[0, i_d] = volume_grad[i_d]
                vel_fluxes[0] = sph_state.boundary.vel_flux[i_p, i_b]
                surface_dists[0] = sph_state.boundary.surface_dist[i_p, i_b]
                n_boundaries = 1
                if sph_state.boundary.has_several_boundaries[i_p, i_b]:
                    boundaries_idx = qd.Vector([-1] * N_BOUNDARIES, dt=gs.qd_int)
                    volume_grads = qd.Matrix.zero(gs.qd_float, N_BOUNDARIES, 3)
                    vel_fluxes = qd.Vector.zero(gs.qd_float, N_BOUNDARIES)
                    surface_dists = qd.Vector([1e9] * N_BOUNDARIES, dt=gs.qd_float)
                    n_boundaries = 0
                    reach = support_radius + sph_boundaries.segments_extent_max[i_b]
                    cell_lo = func_hash_pos_to_grid(pos - reach, sph_boundaries.hash_grid)
                    cell_hi = func_hash_pos_to_grid(pos + reach, sph_boundaries.hash_grid)
                    base = func_hash_wrap_cell(cell_lo, sph_boundaries.hash_grid)
                    n_cells = qd.min(cell_hi - cell_lo + 1, sph_boundaries.hash_grid.grid_res[None])
                    for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
                        i_slot = func_hash_cell_to_slot(base + offset, sph_boundaries.hash_grid)
                        slot_start = sph_boundaries.hash_grid.slot_start[i_slot, i_b]
                        for i_q in range(sph_boundaries.hash_grid.slot_size[i_slot, i_b]):
                            i_sg = sph_boundaries.hash_grid.particles_idx[slot_start + i_q, i_b]
                            corners = sph_boundaries.segments.corners[i_sg, i_b]
                            seg_volume, seg_volume_grad, _ = func_segment_volume(
                                pos,
                                corners,
                                sph_boundaries.segments.normal[i_sg, i_b],
                                sph_boundaries.segments.area[i_sg, i_b],
                                sph_boundaries.segments.extent[i_sg, i_b],
                                sph_boundaries.quad_bary,
                                sph_boundaries.quad_weight,
                                support_radius,
                                kernel_norm,
                            )
                            if seg_volume_grad.norm_sqr() > 0.0:
                                i_bd = sph_boundaries.segments_boundary_idx[i_sg]
                                i_k = -1
                                for i_s in qd.static(range(N_BOUNDARIES)):
                                    if boundaries_idx[i_s] == i_bd:
                                        i_k = i_s
                                if i_k < 0:
                                    if n_boundaries < N_BOUNDARIES:
                                        i_k = n_boundaries
                                        boundaries_idx[i_k] = i_bd
                                        n_boundaries = n_boundaries + 1
                                    else:
                                        i_k = N_BOUNDARIES - 1
                                        boundaries_idx[i_k] = -2
                                volumes[i_k] = volumes[i_k] + seg_volume
                                for i_d in qd.static(range(3)):
                                    volume_grads[i_k, i_d] = volume_grads[i_k, i_d] + seg_volume_grad[i_d]
                                vel_fluxes[i_k] = vel_fluxes[i_k] + sph_boundaries.segments.vel[i_sg, i_b].dot(
                                    seg_volume_grad
                                )
                                surface_dists[i_k] = qd.min(surface_dists[i_k], func_segment_dist(pos, corners))
                    # The divergence theorem counts the kernel at the particle itself out of the volume of a solid
                    # that encloses it, which turns it negative
                    for i_s in qd.static(range(N_BOUNDARIES)):
                        if volumes[i_s] < -1e-3:
                            surface_dists[i_s] = -surface_dists[i_s]

                # The approach velocity of the particle toward each boundary close to it, along the volume gradient
                # of the boundary relative to its segments, is bounded from below by zero
                normals = qd.Matrix.zero(gs.qd_float, 4, 3)
                bounds = qd.Vector.zero(gs.qd_float, 4)
                bounds_boundary = qd.Vector([-1, -1, -1, -1], dt=gs.qd_int)
                n_bounds = 0
                pos_shift = qd.Vector.zero(gs.qd_float, 3)
                for i_s in qd.static(range(N_BOUNDARIES)):
                    volume_grad = qd.Vector([volume_grads[i_s, 0], volume_grads[i_s, 1], volume_grads[i_s, 2]])
                    boundary_grad_norm = volume_grad.norm()
                    surface_dist = surface_dists[i_s]
                    if (
                        i_s < n_boundaries
                        and boundary_grad_norm > 0.0
                        and surface_dist < 0.25 * sph_info.particle_size[None]
                    ):
                        normal = -volume_grad / boundary_grad_norm
                        for i_d in qd.static(range(3)):
                            normals[n_bounds, i_d] = normal[i_d]
                        bounds[n_bounds] = -vel_fluxes[i_s] / boundary_grad_norm
                        bounds_boundary[n_bounds] = i_s
                        n_bounds = n_bounds + 1
                        if surface_dist < 0.0:
                            pos_shift = pos_shift - surface_dist * normal
                violation_tol = legacy_coupler_info.velocity_tol[None] * (1.0 + vel_free.norm())
                vel, set_idx, lambdas = func_project_on_bounds(
                    vel_free, normals, bounds, violation_tol, legacy_coupler_info, n_bounds
                )
                # The multiplier of each bound met as an equality is the velocity the particle takes along its normal
                impulse_coefs = qd.Vector.zero(gs.qd_float, N_BOUNDARIES)
                for i_s in qd.static(range(3)):
                    i_k = set_idx[i_s]
                    if i_k >= 0:
                        i_bs = bounds_boundary[i_k]
                        volume_grad = qd.Vector([volume_grads[i_bs, 0], volume_grads[i_bs, 1], volume_grads[i_bs, 2]])
                        impulse_coefs[i_bs] = sph_info.particles.mass[i_p] * lambdas[i_s] / volume_grad.norm()
                sph_state.particles_reordered.vel[i_r, i_b] = vel
                sph_state.particles_reordered.pos[i_r, i_b] = pos + pos_shift
                sph_state.boundary.boundaries_idx[i_p, i_b] = boundaries_idx
                sph_state.boundary.boundaries_impulse_coef[i_p, i_b] = impulse_coefs

    # Each segment takes the opposite of the impulse it gives the particles around it. On GPU, the lanes of a subgroup
    # take a segment together, striding the particles of each cell around it (see kernel_place_sph_boundary_segments).
    N_LANES = qd.static(32 if legacy_coupler_config.enable_cooperative_segments else 1)
    n_segments, n_envs = sph_boundaries.segments_impulse.shape
    # The environments run fastest, as the arrays keep them innermost
    for i_t in range(n_envs * n_segments * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_sg = i_t // (N_LANES * n_envs)
        impulse = qd.Vector.zero(gs.qd_float, 3)
        n_wet = 0
        if sph_boundaries.segments.is_active[i_sg, i_b]:
            centroid = sph_boundaries.segments.centroid[i_sg, i_b]
            corners = sph_boundaries.segments.corners[i_sg, i_b]
            normal = sph_boundaries.segments.normal[i_sg, i_b]
            area = sph_boundaries.segments.area[i_sg, i_b]
            gravity = sph_info.gravity[i_b]
            # The particles within the support radius of the segment lie within it plus the extent of the segment from its
            # centroid
            extent = sph_boundaries.segments.extent[i_sg, i_b]
            i_bd = sph_boundaries.segments_boundary_idx[i_sg]
            reach = support_radius + extent
            cell_lo = func_hash_pos_to_grid(centroid - reach, sph_state.hash_grid)
            cell_hi = func_hash_pos_to_grid(centroid + reach, sph_state.hash_grid)
            base = func_hash_wrap_cell(cell_lo, sph_state.hash_grid)
            n_cells = qd.min(cell_hi - cell_lo + 1, sph_state.hash_grid.grid_res[None])
            for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
                i_slot = func_hash_cell_to_slot(base + offset, sph_state.hash_grid)
                slot_start = sph_state.hash_grid.slot_start[i_slot, i_b]
                slot_size = sph_state.hash_grid.slot_size[i_slot, i_b]
                for i_r_ in range((slot_size + N_LANES - 1) // N_LANES):
                    i_r = slot_start + i_r_ * N_LANES + i_lane
                    if i_r < slot_start + slot_size:
                        seg_volume, seg_volume_grad, seg_moment = func_segment_volume(
                            sph_state.particles_reordered.pos[i_r, i_b],
                            corners,
                            normal,
                            area,
                            extent,
                            sph_boundaries.quad_bary,
                            sph_boundaries.quad_weight,
                            support_radius,
                            kernel_norm,
                        )
                        if seg_volume_grad.norm_sqr() > 0.0:
                            n_wet = n_wet + 1
                            i_p = sph_state.hash_grid.particles_idx[i_r, i_b]
                            # The impulse of the bound of the boundary of the segment, or of the merged boundaries
                            boundaries_idx = sph_state.boundary.boundaries_idx[i_p, i_b]
                            boundaries_impulse_coef = sph_state.boundary.boundaries_impulse_coef[i_p, i_b]
                            impulse_coef = sph_state.boundary.impulse_coef[i_p, i_b]
                            i_s_matched = qd.select(boundaries_idx[3] == -2, 3, -1)
                            for i_s in qd.static(range(4)):
                                if boundaries_idx[i_s] == i_bd:
                                    i_s_matched = i_s
                            if i_s_matched >= 0:
                                impulse_coef = impulse_coef + boundaries_impulse_coef[i_s_matched]
                            impulse = (
                                impulse
                                + impulse_coef * seg_volume_grad
                                + sph_state.boundary.impulse_gravity_coef[i_p, i_b] * (seg_moment @ gravity)
                            )
        if qd.static(N_LANES > 1):
            # The lanes sum scalars alone
            n_wet = qd_block_sum(n_wet)
            for i_d in qd.static(range(3)):
                impulse[i_d] = qd_block_sum(impulse[i_d])
        if i_lane == 0:
            sph_boundaries.segments_impulse[i_sg, i_b] = impulse
            if qd.static(legacy_coupler_config.has_rigid_sph):
                sph_boundaries.segments_is_wet[i_sg, i_b] = n_wet > 0

    # The links of the rigid geoms take the impulse of their wet segments, and the mass of the liquid one particle size
    # thick over them as added mass (see 'links_added_mass')
    if qd.static(legacy_coupler_config.has_rigid_sph):
        particle_size = sph_info.particle_size[None]
        for i_rs, i_b in qd.ndrange(
            sph_boundaries.rigid_segments_geom_idx.shape[0],
            sph_boundaries.links_added_mass.shape[1],
            axes=qd.static(eoa(sph_boundaries.links_added_mass)),
        ):
            i_sg = n_fem_segments + i_rs
            if sph_boundaries.segments_is_wet[i_sg, i_b]:
                i_g = sph_boundaries.rigid_segments_geom_idx[i_rs]
                i_l = geoms_info.link_idx[i_g]
                segment_mass = sph_boundaries.rho_rest[None] * sph_boundaries.segments.area[i_sg, i_b] * particle_size
                func_exact_sum_add(i_l, i_b, segment_mass, sph_boundaries.links_added_mass_sums)
        for i_l, i_b in qd.ndrange(
            *sph_boundaries.links_added_mass.shape, axes=qd.static(eoa(sph_boundaries.links_added_mass))
        ):
            sph_boundaries.links_added_mass[i_l, i_b] = func_exact_sum_take(
                i_l, i_b, sph_boundaries.links_added_mass_sums
            )
        for i_rs, i_b in qd.ndrange(
            sph_boundaries.rigid_segments_geom_idx.shape[0],
            sph_boundaries.links_added_mass.shape[1],
            axes=qd.static(eoa(sph_boundaries.links_added_mass)),
        ):
            i_sg = n_fem_segments + i_rs
            i_g = sph_boundaries.rigid_segments_geom_idx[i_rs]
            i_l = geoms_info.link_idx[i_g]
            I_l = [i_l, i_b] if qd.static(legacy_coupler_config.is_links_info_batched) else i_l
            if sph_boundaries.segments_is_wet[i_sg, i_b] and not links_info.is_fixed[I_l]:
                mass = links_info.inertial_mass[I_l]
                segment_mass = sph_boundaries.rho_rest[None] * sph_boundaries.segments.area[i_sg, i_b] * particle_size
                force = (
                    sph_boundaries.segments_impulse[i_sg, i_b] / sph_info.substep_dt[None]
                    - segment_mass * sph_info.gravity[i_b]
                ) * (mass / (mass + sph_boundaries.links_added_mass[i_l, i_b]))
                func_add_external_wrench(
                    i_l,
                    i_b,
                    sph_boundaries.segments.centroid[i_sg, i_b],
                    force,
                    torque=qd.Vector.zero(gs.qd_float, 3),
                    links_state=links_state,
                    is_exact=legacy_coupler_config.is_external_wrench_exact,
                )
