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
    qd_quat_to_R,
    qd_transform_by_quat,
    qd_transform_by_trans_quat,
)
from genesis.utils.misc import get_gpu_core_count, qd_from_torch, qd_to_numpy, tensor_to_array
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
# The segments of the fixed rigid geoms other than the planes, which a scene without such geoms coupled with the SPH
# liquid leaves out
FIXED_SPH = "fixed_sph"
FEM_SEGMENTS = Size("fem_segments")
RIGID_SEGMENTS = Size("rigid_segments")
FIXED_TRIS = Size("fixed_tris")
FIXED_CELLS_TRIS = Size("fixed_cells_tris")
FIXED_GEOMS = Size("fixed_geoms")
FIXED_CELLS = Size("fixed_cells")
FIXED_SPARSE_CELLS = Size("fixed_sparse_cells")
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
    # The geom of each rigid segment, and its corners in the frame of the geom, the fixed geoms aside but the planes
    rigid_segments_geom_idx: qd.Tensor = field(int, shape=(RIGID_SEGMENTS,), kind=DataKind.CONSTANT, labels=RIGID_SPH)
    rigid_segments_corners: qd.Tensor = field(
        (float, (3, 3)), shape=(RIGID_SEGMENTS,), kind=DataKind.CONSTANT, labels=RIGID_SPH
    )
    # Whether each geom takes part in each environment, the variants of a heterogeneous entity being active in some alone
    geoms_is_active: qd.Tensor = field(bool, shape=(RIGID_GEOMS, ENVS), kind=DataKind.CONSTANT)
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
    # The triangles of the fixed rigid geoms other than the planes, in the frame of their geom, which a particle reaches
    # wherever the geom stands through the grid of the frame of the geom, whatever the environment (see
    # func_boundary_source_cells). Each triangle splits into segments along a grid, 65536 times whether it fans out of
    # its first corner, plus 256 times its number of rows plus its number of parts across them (see
    # func_boundary_source_segment), which the kernels compute within the reach of each particle alone.
    fixed_tris_corners: qd.Tensor = field(
        (float, (3, 3)), shape=(FIXED_TRIS,), kind=DataKind.CONSTANT, labels=FIXED_SPH
    )
    fixed_tris_split: qd.Tensor = field(int, shape=(FIXED_TRIS,), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    # The triangles of the fixed geoms registered in every cell of the grid of their geom that their box overlaps,
    # sorted by geom and by cell, each the index of its triangle times 32 plus the offset of the cell from the lowest
    # cell of the box along the axes in base 3 (see func_boundary_source_window)
    fixed_cells_tri_idx: qd.Tensor = field(int, shape=(FIXED_CELLS_TRIS,), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    # The index of each fixed geom among the rigid geoms, the lower corner of its grid, the number of cells of the grid
    # along each axis and the largest extent of its segments
    fixed_geoms_idx: qd.Tensor = field(int, shape=(FIXED_GEOMS,), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    fixed_geoms_grid_lower: qd.Tensor = field(
        (float, 3), shape=(FIXED_GEOMS,), kind=DataKind.CONSTANT, labels=FIXED_SPH
    )
    fixed_geoms_grid_res: qd.Tensor = field((int, 3), shape=(FIXED_GEOMS,), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    fixed_geoms_extent_max: qd.Tensor = field(float, shape=(FIXED_GEOMS,), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    # The width of the cells of the grids of the fixed geoms
    fixed_cell_size: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    # The ratio of the Gram determinant of the edges of a fixed triangle to the product of their squared lengths below
    # which the triangle is too thin for the coordinates along its edges to resolve a point
    fixed_tris_det_rtol: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    # The cells of the fixed geoms, those of each geom spanning fixed_geoms_cell_start[i_fg] to
    # fixed_geoms_cell_start[i_fg + 1] in ascending order of their index in its grid, every cell of the grid of a dense
    # geom and those holding triangles of the others, whose indices in the grid of their geom are
    # fixed_sparse_cells_grid_idx[i_c + fixed_geoms_sparse_cell_offset[i_fg]]. The triangles of each cell span
    # fixed_cells_tri_start[i_c] to fixed_cells_tri_start[i_c + 1].
    fixed_geoms_cell_start: qd.Tensor = field(int, shape=(FIXED_GEOMS + 1,), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    fixed_geoms_is_dense: qd.Tensor = field(bool, shape=(FIXED_GEOMS,), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    fixed_geoms_sparse_cell_offset: qd.Tensor = field(
        int, shape=(FIXED_GEOMS,), kind=DataKind.CONSTANT, labels=FIXED_SPH
    )
    fixed_sparse_cells_grid_idx: qd.Tensor = field(
        int, shape=(FIXED_SPARSE_CELLS,), kind=DataKind.CONSTANT, labels=FIXED_SPH
    )
    fixed_cells_tri_start: qd.Tensor = field(int, shape=(FIXED_CELLS + 1,), kind=DataKind.CONSTANT, labels=FIXED_SPH)
    # The fixed geoms whose box, widened by the support radius, overlaps the domain in each environment over the
    # substep, the first fixed_envs_n_geoms[i_b] of them in ascending order (see kernel_select_sph_fixed_geoms)
    fixed_envs_geoms: qd.Tensor = field(int, shape=(FIXED_GEOMS, ENVS), labels=FIXED_SPH)
    fixed_envs_n_geoms: qd.Tensor = field(int, shape=(ENVS,), labels=FIXED_SPH)


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
    EPS: qd.Tensor = field(float, shape=())


class LegacyCouplerConfig(Config):
    """The couplings of the legacy coupler that its kernels compile in, between the solvers each one names."""

    has_rigid_mpm: bool
    has_rigid_sph: bool
    # Whether fixed rigid geoms other than the planes bound the SPH liquid (see 'fixed_tris_corners')
    has_fixed_rigid_sph: bool
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


def split_triangles(tris_verts_pos: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Split triangles by halving the longest edge of their parts until all the edges of the parts fit within a size.

    Return the triangle of each part and the barycentric coordinates of its corners in the triangle, the parts sorted by
    triangle.
    """
    n_tris = len(tris_verts_pos)
    parts_tri_idx, parts_bary = np.arange(n_tris), np.tile(np.eye(3), (n_tris, 1, 1))
    kept_parts = [(parts_tri_idx[:0], parts_bary[:0])]
    while len(parts_tri_idx) > 0:
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
        has_rigid_sph, has_fixed_rigid_sph = self._build_sph_boundaries(self.rigid_sph_geoms)
        n_segments = self.sph_boundaries.segments_impulse.shape[0] if has_rigid_sph or self.has_fem_sph else 0
        self.legacy_coupler_config = LegacyCouplerConfig(
            has_rigid_mpm=has_rigid_mpm,
            has_rigid_sph=has_rigid_sph,
            has_fixed_rigid_sph=has_fixed_rigid_sph,
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
                EPS=gs.EPS,
            )
        )

        if self.legacy_coupler_config.has_rigid_mpm and self.mpm_solver.enable_CPIC:
            kernel_reset_mpm_rigid_normal(self.sim.scene._sanitize_envs_idx(None), self.mpm_solver.mpm_state)

    def _build_sph_boundaries(self, rigid_sph_geoms: list["RigidGeom"]) -> tuple[bool, bool]:
        """
        Build the segments of the moving boundaries of the SPH liquid.

        Return whether rigid geoms hold any, the fixed geoms other than the planes aside, and whether these fixed geoms
        hold any.
        """
        has_fem_sph = self.has_fem_sph
        if not (has_fem_sph or rigid_sph_geoms):
            self.sph_boundaries = SPHBoundaries(unused=True)
            return False, False

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
        # The fixed geoms other than the planes may stand anywhere in each environment, so their triangles stay in the
        # frame of their geom (see 'fixed_tris_corners'). The planes take their segments apart.
        particle_size = self.sph_solver.particle_size
        rigid_tris_is_plane = np.repeat(
            np.array([geom.type == gs.GEOM_TYPE.PLANE for geom in geoms], dtype=bool),
            [len(geom.init_faces) for geom in geoms],
        )
        (tris_idx,) = np.nonzero(~rigid_tris_is_fixed & ~rigid_tris_is_plane)
        tri_idx, corners_bary = split_triangles(rigid_tris_verts_pos[tris_idx], particle_size)
        segments_tri_idx = tris_idx[tri_idx]
        segments_geom_idx_parts = [rigid_tris_geom_idx[segments_tri_idx]]
        segments_corners_parts = [np.einsum("nci,nij->ncj", corners_bary, rigid_tris_verts_pos[segments_tri_idx])]

        # Each fixed triangle splits into segments along a grid whose edges fit within the particle size, regular or
        # fanning out of the corner opposite its shortest edge, whichever gives fewer segments (see
        # func_boundary_source_segment), the triangles too large for 16 x 16 segments being halved down to it first.
        # Each part of a triangle is registered in every cell of the grid of its geom that its box overlaps, the cells
        # half as wide as the largest part, so that a part spans at most 3 cells along each axis.
        n_split_max = 16
        (tris_idx,) = np.nonzero(rigid_tris_is_fixed & ~rigid_tris_is_plane)
        tri_idx, corners_bary = split_triangles(rigid_tris_verts_pos[tris_idx], n_split_max * particle_size)
        fixed_tris_corners = np.einsum("nci,nij->ncj", corners_bary, rigid_tris_verts_pos[tris_idx[tri_idx]])
        # The corners turn so that the first one faces the shortest edge, the winding kept
        fixed_tris_edges = np.linalg.norm(fixed_tris_corners - np.roll(fixed_tris_corners, -1, axis=1), axis=-1)
        corners_order = (fixed_tris_edges.argmin(axis=-1)[:, None] + 2 + np.arange(3)) % 3
        fixed_tris_corners = np.take_along_axis(fixed_tris_corners, corners_order[..., None], axis=1)
        fixed_tris_edges = np.linalg.norm(fixed_tris_corners - np.roll(fixed_tris_corners, -1, axis=1), axis=-1)
        n_regular = np.clip(np.ceil(fixed_tris_edges.max(axis=-1) / particle_size), 1, n_split_max).astype(gs.np_int)
        # The longest edge of the parts of each fan: along the rays, across them, and the diagonals of the cells beyond
        # the first row, the longest at the extreme cells
        fan_n_rays, fan_n_across = [
            n.reshape(-1)
            for n in np.meshgrid(np.arange(1, n_split_max + 1), np.arange(1, n_split_max + 1), indexing="ij")
        ]
        fixed_tris_split = 256 * n_regular
        # A fan holds at least as many parts across as its opposite edge needs, times twice as many rows as its rays
        # need, so that only the triangles whose regular grid holds more parts look for a fan, the thin ones
        fan_n_parts_min = np.ceil(fixed_tris_edges[:, 1] / particle_size) * (
            2 * np.ceil(np.maximum(fixed_tris_edges[:, 0], fixed_tris_edges[:, 2]) / particle_size) - 1
        )
        (tris_thin_idx,) = np.nonzero(fan_n_parts_min < n_regular**2)
        for chunk in np.array_split(tris_thin_idx, max(1, len(tris_thin_idx) // 16384)):
            corners = fixed_tris_corners[chunk, None]
            ray_0, opposite = corners[..., 1, :] - corners[..., 0, :], corners[..., 2, :] - corners[..., 1, :]
            parts_edge_max = np.maximum(
                np.maximum(fixed_tris_edges[chunk, None, 0], fixed_tris_edges[chunk, None, 2]) / fan_n_rays,
                fixed_tris_edges[chunk, None, 1] / fan_n_across,
            )
            for s_end in (2.0 / fan_n_rays, np.ones(len(fan_n_rays))):
                for t_start in (np.zeros(len(fan_n_across)), 1.0 - 1.0 / fan_n_across):
                    diagonal = (ray_0 + t_start[..., None] * opposite) / fan_n_rays[..., None] + (
                        s_end[..., None] / fan_n_across[..., None]
                    ) * opposite
                    is_beyond_first_row = fan_n_rays > 1
                    parts_edge_max = np.where(
                        is_beyond_first_row,
                        np.maximum(parts_edge_max, np.linalg.norm(diagonal, axis=-1)),
                        parts_edge_max,
                    )
            fan_n_parts = np.where(
                parts_edge_max <= particle_size * (1.0 + 1e-6),
                fan_n_across * (2 * fan_n_rays - 1),
                np.inf,
            )
            i_fan = fan_n_parts.argmin(axis=-1)
            is_fan = fan_n_parts[np.arange(len(chunk)), i_fan] < n_regular[chunk] ** 2
            fixed_tris_split[chunk] = np.where(
                is_fan, 65536 + 256 * fan_n_rays[i_fan] + fan_n_across[i_fan], 256 * n_regular[chunk]
            )
        # The regular grid takes its coordinates along the edges from the corner facing the longest edge instead, at
        # the widest angle, whose grid cells are the least skewed, so that the window of the segments within the reach
        # of a particle holds the fewest beyond it (see func_boundary_source_window)
        is_regular = fixed_tris_split < 65536
        fixed_tris_corners[is_regular] = np.take_along_axis(
            fixed_tris_corners[is_regular],
            ((fixed_tris_edges[is_regular].argmax(axis=-1)[:, None] + 2 + np.arange(3)) % 3)[..., None],
            axis=1,
        )
        fixed_geoms_idx, fixed_tris_fixed_geom_idx = np.unique(
            rigid_tris_geom_idx[tris_idx[tri_idx]], return_inverse=True
        )
        fixed_tris_fixed_geom_idx = fixed_tris_fixed_geom_idx.reshape(-1)
        fixed_cell_size = 0.5 * n_split_max * particle_size
        fixed_geoms_grid_lower = np.full((len(fixed_geoms_idx), 3), np.inf)
        fixed_geoms_grid_upper = np.full((len(fixed_geoms_idx), 3), -np.inf)
        np.minimum.at(fixed_geoms_grid_lower, fixed_tris_fixed_geom_idx, fixed_tris_corners.min(axis=1))
        np.maximum.at(fixed_geoms_grid_upper, fixed_tris_fixed_geom_idx, fixed_tris_corners.max(axis=1))
        fixed_geoms_grid_res = (
            np.floor((fixed_geoms_grid_upper - fixed_geoms_grid_lower) / fixed_cell_size).astype(gs.np_int) + 1
        )
        if max([math.prod(grid_res) for grid_res in fixed_geoms_grid_res.tolist()], default=0) >= 2**31:
            gs.raise_exception(
                "A fixed rigid geom coupled with the SPH liquid spans more cells of its grid than an index holds."
            )
        if len(fixed_tris_corners) >= 2**26:
            gs.raise_exception(
                "The fixed rigid geoms coupled with the SPH liquid hold more triangles than an index holds."
            )
        # The segments lie within two thirds of their longest edge, at most the particle size, from their centroid
        fixed_geoms_extent_max = np.full(len(fixed_geoms_idx), 2.0 / 3.0 * particle_size)
        tris_grid_lower = fixed_geoms_grid_lower[fixed_tris_fixed_geom_idx]
        tris_grid_res = fixed_geoms_grid_res[fixed_tris_fixed_geom_idx]
        tris_cell_lo = np.floor((fixed_tris_corners.min(axis=1) - tris_grid_lower) / fixed_cell_size).astype(gs.np_int)
        tris_cell_hi = np.minimum(
            np.floor((fixed_tris_corners.max(axis=1) - tris_grid_lower) / fixed_cell_size).astype(gs.np_int),
            tris_grid_res - 1,
        )
        regs_geom_idx_parts, regs_grid_idx_parts, regs_tri_idx_parts = [], [], []
        for offset in np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), axis=-1).reshape(-1, 3):
            regs_cell = tris_cell_lo + offset
            (regs_tri_idx,) = np.nonzero((regs_cell <= tris_cell_hi).all(axis=-1))
            regs_cell = regs_cell[regs_tri_idx]
            regs_grid_res = tris_grid_res[regs_tri_idx]
            regs_grid_idx = regs_cell[:, 0] * regs_grid_res[:, 1] + regs_cell[:, 1]
            regs_grid_idx_parts.append(regs_grid_idx * regs_grid_res[:, 2] + regs_cell[:, 2])
            regs_geom_idx_parts.append(fixed_tris_fixed_geom_idx[regs_tri_idx])
            # The offset of the cell from the lowest cell of the box of the triangle goes with the triangle, so that
            # the kernels find the lowest cell without recomputing it
            regs_tri_idx_parts.append((32 * regs_tri_idx + 9 * offset[0] + 3 * offset[1] + offset[2]).astype(gs.np_int))
        regs_geom_idx, regs_grid_idx, regs_tri_idx = [
            np.concatenate([np.zeros(0, dtype=gs.np_int), *parts])
            for parts in (regs_geom_idx_parts, regs_grid_idx_parts, regs_tri_idx_parts)
        ]
        regs_order = np.lexsort((regs_grid_idx, regs_geom_idx))
        regs_geom_idx, regs_grid_idx, fixed_cells_tri_idx = [
            array[regs_order] for array in (regs_geom_idx, regs_grid_idx, regs_tri_idx)
        ]
        cells_reg_start = np.flatnonzero(
            (np.diff(regs_geom_idx, prepend=-1) != 0) | (np.diff(regs_grid_idx, prepend=-1) != 0)
        )
        cells_n_tris = np.diff(cells_reg_start, append=len(regs_order))
        cells_fixed_geom_idx, cells_grid_idx = regs_geom_idx[cells_reg_start], regs_grid_idx[cells_reg_start]
        # A geom whose grid holds at most 8 cells per cell holding parts keeps every cell of its grid, which a particle
        # finds at once, while the others keep the cells holding parts alone, found by bisection
        geoms_n_cells = np.bincount(cells_fixed_geom_idx, minlength=len(fixed_geoms_idx))
        fixed_geoms_is_dense = fixed_geoms_grid_res.prod(axis=-1) <= 8 * geoms_n_cells
        geoms_cells_n_tris, geoms_sparse_cells_grid_idx = [], []
        for i_fg, is_dense in enumerate(fixed_geoms_is_dense):
            is_geom_cell = cells_fixed_geom_idx == i_fg
            if is_dense:
                n_grid_cells = fixed_geoms_grid_res[i_fg].prod()
                geoms_cells_n_tris.append(
                    np.bincount(
                        cells_grid_idx[is_geom_cell], weights=cells_n_tris[is_geom_cell], minlength=n_grid_cells
                    ).astype(gs.np_int)
                )
                geoms_sparse_cells_grid_idx.append(cells_grid_idx[:0])
            else:
                geoms_cells_n_tris.append(cells_n_tris[is_geom_cell])
                geoms_sparse_cells_grid_idx.append(cells_grid_idx[is_geom_cell])
        geoms_n_cells = np.array([len(n_tris) for n_tris in geoms_cells_n_tris])
        fixed_geoms_cell_start = np.concatenate([[0], np.cumsum(geoms_n_cells)])
        geoms_n_sparse_cells = np.array([len(grid_idx) for grid_idx in geoms_sparse_cells_grid_idx])
        fixed_geoms_sparse_cell_offset = (
            np.cumsum(geoms_n_sparse_cells) - geoms_n_sparse_cells - fixed_geoms_cell_start[:-1]
        )
        fixed_sparse_cells_grid_idx = np.concatenate([np.zeros(0, dtype=gs.np_int), *geoms_sparse_cells_grid_idx])
        fixed_cells_tri_start = np.concatenate(
            [[0], np.cumsum(np.concatenate([np.zeros(0, dtype=gs.np_int), *geoms_cells_n_tris]))]
        )

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
        has_rigid_sph, has_fixed_rigid_sph = len(rigid_segments_geom_idx) > 0, len(fixed_geoms_idx) > 0

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
                fixed_tris=len(fixed_tris_corners),
                fixed_cells_tris=len(fixed_cells_tri_idx),
                fixed_geoms=len(fixed_geoms_idx),
                fixed_cells=len(fixed_cells_tri_start) - 1,
                fixed_sparse_cells=len(fixed_sparse_cells_grid_idx),
                rigid_links=self.rigid_solver.n_links,
                rigid_geoms=self.rigid_solver.n_geoms,
                exact_sum_slots=self.rigid_solver.n_links,
                exact_sum_bins=3,
                **hash_grid_sizes(math.prod(self.sph_solver.hash_grid_res)),
            ),
            values=dict(
                hash_grid=dict(cell_size=self.sph_solver.hash_grid_cell_size, grid_res=self.sph_solver.hash_grid_res),
                fixed_cell_size=fixed_cell_size,
                fixed_tris_det_rtol=1e-12,
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
                    SPHBoundaries.fixed_envs_geoms,
                ): False
            },
            unused={
                FEM_SPH: not has_fem_sph,
                RIGID_SPH: not has_rigid_sph,
                FIXED_SPH: not has_fixed_rigid_sph,
                SPHBoundaries.geoms_is_active: not (has_rigid_sph or has_fixed_rigid_sph),
            },
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
        # The fixed geoms may bound the liquid alone, their segments lying apart (see 'fixed_tris_corners')
        if len(segments_boundary_idx) > 0:
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
        if has_fixed_rigid_sph:
            # FIXME: quadrants#1100 - from_numpy holds as much memory as it uploads on a unified-memory GPU, so these
            # arrays, scaling with the area of the fixed geoms, go through their zero-copy view.
            for array, values in (
                (self.sph_boundaries.fixed_tris_corners, fixed_tris_corners),
                (self.sph_boundaries.fixed_tris_split, fixed_tris_split),
                (self.sph_boundaries.fixed_cells_tri_idx, fixed_cells_tri_idx),
                (self.sph_boundaries.fixed_geoms_idx, fixed_geoms_idx),
                (self.sph_boundaries.fixed_geoms_grid_lower, fixed_geoms_grid_lower),
                (self.sph_boundaries.fixed_geoms_grid_res, fixed_geoms_grid_res),
                (self.sph_boundaries.fixed_geoms_extent_max, fixed_geoms_extent_max),
                (self.sph_boundaries.fixed_geoms_is_dense, fixed_geoms_is_dense),
                (self.sph_boundaries.fixed_geoms_cell_start, fixed_geoms_cell_start),
                (self.sph_boundaries.fixed_geoms_sparse_cell_offset, fixed_geoms_sparse_cell_offset),
                (self.sph_boundaries.fixed_cells_tri_start, fixed_cells_tri_start),
            ):
                qd_from_torch(array, torch.from_numpy(values))
            if len(fixed_sparse_cells_grid_idx) > 0:
                qd_from_torch(
                    self.sph_boundaries.fixed_sparse_cells_grid_idx, torch.from_numpy(fixed_sparse_cells_grid_idx)
                )
            if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
                torch.mps.synchronize()
        if has_rigid_sph or has_fixed_rigid_sph:
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
        return has_rigid_sph, has_fixed_rigid_sph

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
        if (
            self.legacy_coupler_config.has_fem_sph
            or self.legacy_coupler_config.has_rigid_sph
            or self.legacy_coupler_config.has_fixed_rigid_sph
        ):
            if self.legacy_coupler_config.has_fixed_rigid_sph:
                kernel_select_sph_fixed_geoms(
                    self.rigid_solver.dyn_state.geoms,
                    self.sph_boundaries,
                    self.sph_solver.sph_info,
                    self.legacy_coupler_info,
                )
            kernel_place_sph_boundary_segments(
                self.fem_solver.fem_state,
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.sph_boundaries,
                self.fem_solver.fem_info,
                self.sph_solver.sph_info,
                self.rigid_solver.dyn_info.geoms,
                self.legacy_coupler_info,
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
        if (
            self.legacy_coupler_config.has_fem_sph
            or self.legacy_coupler_config.has_rigid_sph
            or self.legacy_coupler_config.has_fixed_rigid_sph
        ):
            kernel_sph_boundaries_impulse(
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.geoms,
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


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.kernel(fastcache=True)
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


@qd.func
def func_boundary_source_cells(
    i_src: int,
    i_b: int,
    pos: qd.types.vector(3),
    geoms_state: GeomsState,
    sph_boundaries: SPHBoundaries,
    support_radius: float,
    legacy_coupler_config: qd.template(),
    eps: float,
):
    """
    Return the frame of a source of boundary segments, a point in it, and the first cell and number of cells along each
    axis of the source within the reach of the point.

    Source 0 is the hash grid of the segments placed in the world frame every substep, and source 1 + i_k the grid of
    the k-th fixed geom near the domain in the environment in its own frame (see 'fixed_tris_corners' and
    'fixed_envs_geoms'), which a point reaches where it lies within the support radius plus the largest extent of its
    segments from the cells of the grid.
    """
    pos_src = pos
    rot = qd.Matrix.identity(gs.qd_float, 3)
    cell_base = qd.Vector.zero(gs.qd_int, 3)
    n_cells = qd.Vector.zero(gs.qd_int, 3)
    if i_src == 0:
        # The segments overlapping the support radius of the point have their centroid within it plus their extent
        reach = support_radius + sph_boundaries.segments_extent_max[i_b]
        cell_lo = func_hash_pos_to_grid(pos - reach, sph_boundaries.hash_grid)
        cell_hi = func_hash_pos_to_grid(pos + reach, sph_boundaries.hash_grid)
        cell_base = func_hash_wrap_cell(cell_lo, sph_boundaries.hash_grid)
        n_cells = qd.min(cell_hi - cell_lo + 1, sph_boundaries.hash_grid.grid_res[None])
    if qd.static(legacy_coupler_config.has_fixed_rigid_sph):
        if i_src > 0:
            i_fg = sph_boundaries.fixed_envs_geoms[i_src - 1, i_b]
            i_g = sph_boundaries.fixed_geoms_idx[i_fg]
            if sph_boundaries.geoms_is_active[i_g, i_b]:
                geom_quat = geoms_state.quat[i_g, i_b]
                pos_src = qd_inv_transform_by_trans_quat(pos, geoms_state.pos[i_g, i_b], geom_quat)
                rot = qd_quat_to_R(geom_quat, eps)
                reach = support_radius + sph_boundaries.fixed_geoms_extent_max[i_fg]
                cell_size = sph_boundaries.fixed_cell_size[None]
                grid_lower = sph_boundaries.fixed_geoms_grid_lower[i_fg]
                cell_lo = qd.max(qd.floor((pos_src - reach - grid_lower) / cell_size, gs.qd_int), 0)
                cell_hi = qd.min(
                    qd.floor((pos_src + reach - grid_lower) / cell_size, gs.qd_int),
                    sph_boundaries.fixed_geoms_grid_res[i_fg] - 1,
                )
                cell_base = cell_lo
                n_cells = qd.max(cell_hi - cell_lo + 1, 0)
    return pos_src, rot, cell_base, n_cells


@qd.func
def func_boundary_source_cell(
    i_src: int,
    i_b: int,
    cell: qd.types.vector(3, qd.i32),
    sph_boundaries: SPHBoundaries,
    legacy_coupler_config: qd.template(),
):
    """
    Return the first sorted index and the number of the items of a cell of a source of boundary segments.

    The cell of a fixed geom is found at its index in the grid of a dense geom, and by bisection over the indices in
    its grid of the cells holding triangles otherwise. The items of the hash grid of the world frame are segments,
    those of a fixed geom triangles (see func_boundary_source_window).
    """
    item_start = 0
    n_items = 0
    if i_src == 0:
        i_slot = func_hash_cell_to_slot(cell, sph_boundaries.hash_grid)
        item_start = sph_boundaries.hash_grid.slot_start[i_slot, i_b]
        n_items = sph_boundaries.hash_grid.slot_size[i_slot, i_b]
    if qd.static(legacy_coupler_config.has_fixed_rigid_sph):
        if i_src > 0:
            i_fg = sph_boundaries.fixed_envs_geoms[i_src - 1, i_b]
            grid_res = sph_boundaries.fixed_geoms_grid_res[i_fg]
            i_grid = (cell[0] * grid_res[1] + cell[1]) * grid_res[2] + cell[2]
            i_c = sph_boundaries.fixed_geoms_cell_start[i_fg] + i_grid
            is_found = True
            if not sph_boundaries.fixed_geoms_is_dense[i_fg]:
                i_lo = sph_boundaries.fixed_geoms_cell_start[i_fg]
                i_end = sph_boundaries.fixed_geoms_cell_start[i_fg + 1]
                i_hi = i_end
                sparse_offset = sph_boundaries.fixed_geoms_sparse_cell_offset[i_fg]
                # 31 halvings cover any number of cells an index holds
                for _ in range(31):
                    if i_lo < i_hi:
                        i_mid = (i_lo + i_hi) // 2
                        if sph_boundaries.fixed_sparse_cells_grid_idx[i_mid + sparse_offset] < i_grid:
                            i_lo = i_mid + 1
                        else:
                            i_hi = i_mid
                i_c = i_lo
                is_found = False
                if i_lo < i_end:
                    is_found = sph_boundaries.fixed_sparse_cells_grid_idx[i_lo + sparse_offset] == i_grid
            if is_found:
                item_start = sph_boundaries.fixed_cells_tri_start[i_c]
                n_items = sph_boundaries.fixed_cells_tri_start[i_c + 1] - item_start
    return item_start, n_items


@qd.func
def func_boundary_source_window(
    i_src: int,
    i_b: int,
    i_q: int,
    cell: qd.types.vector(3, qd.i32),
    cell_base: qd.types.vector(3, qd.i32),
    pos: qd.types.vector(3),
    sph_boundaries: SPHBoundaries,
    legacy_coupler_config: qd.template(),
    support_radius: float,
):
    """
    Return the window of the segments of an item of a source of boundary segments that a point may reach.

    The caller gets the number of rows of the window, the window, the corners of the triangle of the item and the
    coordinates of the point along its edges. An item of the hash grid of the world frame is a single segment, a window
    of one row and one column. An item of a fixed geom is a triangle registered in every cell its box overlaps, which
    only the lowest of the cells reached by the point among them takes. The window holds the item, its first row and
    number of rows, its first column and number of columns, the split of its grid, negated where
    func_boundary_window_row narrows the columns of each row, and its fixed geom. The rows and columns of the grid of
    the triangle are those that the reach of the point overlaps.
    """
    n_rows = 0
    window = qd.Vector([-1, 0, 1, 0, 1, 0, -1], dt=qd.i32)
    tri_corners = qd.Matrix.zero(gs.qd_float, 3, 3)
    coords = qd.Vector.zero(gs.qd_float, 6)
    if i_src == 0:
        n_rows = 1
        window[0] = sph_boundaries.hash_grid.particles_idx[i_q, i_b]
    if qd.static(legacy_coupler_config.has_fixed_rigid_sph):
        if i_src > 0:
            i_fg = sph_boundaries.fixed_envs_geoms[i_src - 1, i_b]
            tri_reg = sph_boundaries.fixed_cells_tri_idx[i_q]
            i_t = tri_reg // 32
            offset_code = tri_reg % 32
            tri_cell_lo = cell - qd.Vector([offset_code // 9, offset_code // 3 % 3, offset_code % 3], dt=qd.i32)
            owner = qd.max(tri_cell_lo, cell_base)
            if (owner == cell).all():
                tri_corners = sph_boundaries.fixed_tris_corners[i_t]
                corner_0 = qd.Vector([tri_corners[0, 0], tri_corners[0, 1], tri_corners[0, 2]])
                corner_1 = qd.Vector([tri_corners[1, 0], tri_corners[1, 1], tri_corners[1, 2]])
                corner_2 = qd.Vector([tri_corners[2, 0], tri_corners[2, 1], tri_corners[2, 2]])
                window[0] = i_t
                split = sph_boundaries.fixed_tris_split[i_t]
                window[5] = split
                window[6] = i_fg
                is_fan = split >= 65536
                n_rays = split // 256 % 256
                n_across = split % 256
                # The coordinates of the point along the edges from the first corner, and the change of each per unit
                # displacement in the plane of the triangle
                edge_1, edge_2, offset = corner_1 - corner_0, corner_2 - corner_0, pos - corner_0
                g_11, g_12, g_22 = edge_1.dot(edge_1), edge_1.dot(edge_2), edge_2.dot(edge_2)
                # A triangle too thin for its coordinates to resolve the point takes its whole grid
                det = g_11 * g_22 - g_12 * g_12
                is_degenerate = det <= sph_boundaries.fixed_tris_det_rtol[None] * g_11 * g_22
                det = qd.select(is_degenerate, 1.0, det)
                u = (g_22 * offset.dot(edge_1) - g_12 * offset.dot(edge_2)) / det
                v = (g_11 * offset.dot(edge_2) - g_12 * offset.dot(edge_1)) / det
                reach = support_radius + sph_boundaries.fixed_geoms_extent_max[i_fg]
                du = reach * qd.sqrt(g_22 / det)
                dv = reach * qd.sqrt(g_11 / det)
                ds = reach * qd.sqrt(qd.max(g_11 + g_22 - 2.0 * g_12, 0.0) / det)
                # The rows span the first coordinate of the regular grid and the distance along the rays of the fan,
                # the columns the second coordinate of the regular grid and every ray of the fan
                coord_row, d_row = u, du
                if is_fan:
                    coord_row, d_row = u + v, ds
                row_lo = qd.max(qd.floor((coord_row - d_row) * n_rays, qd.i32), 0)
                row_hi = qd.min(qd.floor((coord_row + d_row) * n_rays, qd.i32), n_rays - 1)
                col_lo = qd.max(qd.floor((v - dv) * n_rays, qd.i32), 0)
                col_hi = qd.min(qd.floor((v + dv) * n_rays, qd.i32), n_rays - 1)
                if is_fan:
                    col_lo = 0
                    col_hi = n_across - 1
                if is_degenerate:
                    row_lo, row_hi = 0, n_rays - 1
                    col_lo, col_hi = 0, qd.select(is_fan, n_across, n_rays) - 1
                if row_lo <= row_hi and col_lo <= col_hi:
                    window[1] = row_lo
                    window[2] = row_hi - row_lo + 1
                    window[3] = col_lo
                    window[4] = col_hi - col_lo + 1
                    n_rows = window[2]
                    # A regular grid narrows the columns of each row to the reach (see func_boundary_window_row)
                    if not (is_fan or is_degenerate):
                        coords = qd.Vector([u, v, g_11, g_12, g_22, reach * reach])
                        window[5] = -split
    return n_rows, window, tri_corners, coords


@qd.func
def func_boundary_window_row(i_row: int, window: qd.types.vector(7, qd.i32), coords: qd.types.vector(6)):
    """
    Return the first column and the number of columns of a row of the window of the grid of a triangle (see
    func_boundary_source_window).

    The columns of a row of a regular grid narrow to those whose cells meet the ellipse of the points of the plane of the
    triangle within the reach of the point, in the coordinates along the edges of the triangle from its first corner,
    and to the triangle. The other windows take every column of every row.
    """
    col_lo = window[3]
    n_cols = window[4]
    if window[5] < 0:
        n_rays = -window[5] // 256 % 256
        u, v, g_11, g_12, g_22, reach_sqr = coords[0], coords[1], coords[2], coords[3], coords[4], coords[5]
        det = g_11 * g_22 - g_12 * g_12
        # The offsets of the row along the first coordinate, within the extent of the ellipse along it
        du_max = qd.sqrt(reach_sqr * g_22 / det)
        du_lo = qd.min(qd.max(i_row / n_rays - u, -du_max), du_max)
        du_hi = qd.min(qd.max((i_row + 1) / n_rays - u, -du_max), du_max)
        # The ellipse spans the second coordinate between (-g_12 du -+ sqrt(g_22 reach^2 - det du^2)) / g_22 at an
        # offset du along the first, whose extremes over the row lie at its ends or at -+g_12 reach / sqrt(det g_11)
        du_ext = g_12 * qd.sqrt(reach_sqr / (det * g_11))
        du_top = qd.min(qd.max(-du_ext, du_lo), du_hi)
        du_bottom = qd.min(qd.max(du_ext, du_lo), du_hi)
        dv_top = (-g_12 * du_top + qd.sqrt(qd.max(g_22 * reach_sqr - det * du_top * du_top, 0.0))) / g_22
        dv_bottom = (-g_12 * du_bottom - qd.sqrt(qd.max(g_22 * reach_sqr - det * du_bottom * du_bottom, 0.0))) / g_22
        col_lo = qd.max(qd.floor((v + dv_bottom) * n_rays, qd.i32), 0)
        col_hi = qd.min(qd.floor((v + dv_top) * n_rays, qd.i32), n_rays - 1 - i_row)
        n_cols = qd.max(col_hi - col_lo + 1, 0)
    return col_lo, n_cols


@qd.func
def func_boundary_source_segment(
    i_src: int,
    i_b: int,
    i_row: int,
    i_col: int,
    is_down: bool,
    window: qd.types.vector(7, qd.i32),
    pos: qd.types.vector(3),
    tri_corners: qd.types.matrix(3, 3),
    sph_boundaries: SPHBoundaries,
    legacy_coupler_config: qd.template(),
    support_radius: float,
    eps: float,
):
    """
    Return whether a segment of a window may lie within the support radius of a point, with its shape and velocity.

    Its corners, normal, area, extent and velocity are in the frame of its source, and the boundary of the segment comes
    last. The fixed geoms stand still. The regular grid of a triangle splits it into n x n parts along the lines
    parallel to its edges, two per cell of its rows and columns along its second and third corners. The fan splits it
    along the lines parallel to the edge opposite its first corner and the rays from that corner, a single part per
    column next to the corner and two per cell beyond. A segment of a fixed geom whose centroid lies beyond the reach of
    the point, or which falls outside of its triangle, is rejected before its shape is computed (see
    func_boundary_source_window).
    """
    is_near = True
    corners = qd.Matrix.zero(gs.qd_float, 3, 3)
    normal = qd.Vector.zero(gs.qd_float, 3)
    area = gs.qd_float(0.0)
    extent = gs.qd_float(0.0)
    vel = qd.Vector.zero(gs.qd_float, 3)
    i_bd = -1
    if i_src == 0:
        i_sg = window[0]
        corners = sph_boundaries.segments.corners[i_sg, i_b]
        normal = sph_boundaries.segments.normal[i_sg, i_b]
        area = sph_boundaries.segments.area[i_sg, i_b]
        extent = sph_boundaries.segments.extent[i_sg, i_b]
        vel = sph_boundaries.segments.vel[i_sg, i_b]
        i_bd = sph_boundaries.segments_boundary_idx[i_sg]
    if qd.static(legacy_coupler_config.has_fixed_rigid_sph):
        if i_src > 0:
            split = qd.abs(window[5])
            n_rays = split // 256 % 256
            n_across = split % 256
            corners_grid = qd.Matrix.zero(gs.qd_float, 3, 2)
            if split < 65536:
                # The upward part of the cell of row i_row and column i_col of the regular grid, then the downward one
                is_near = i_row + i_col <= n_rays - 1 - qd.select(is_down, 1, 0)
                corners_grid = qd.Matrix([[i_row, i_col], [i_row + 1, i_col], [i_row, i_col + 1]], dt=gs.qd_float)
                if is_down:
                    corners_grid = qd.Matrix(
                        [[i_row + 1, i_col], [i_row + 1, i_col + 1], [i_row, i_col + 1]], dt=gs.qd_float
                    )
                corners_grid = corners_grid / n_rays
            else:
                # The fan holds a single part per column next to its first corner, then two per cell, taken along the
                # rays at s and across them at t
                is_near = i_row > 0 or not is_down
                corners_st = qd.Matrix([[0, 0], [1, i_col], [1, i_col + 1]], dt=gs.qd_float)
                if i_row > 0:
                    corners_st = qd.Matrix([[i_row, i_col], [i_row + 1, i_col], [i_row + 1, i_col + 1]], dt=gs.qd_float)
                    if is_down:
                        corners_st = qd.Matrix(
                            [[i_row, i_col], [i_row + 1, i_col + 1], [i_row, i_col + 1]], dt=gs.qd_float
                        )
                for i_c in qd.static(range(3)):
                    s_c = corners_st[i_c, 0] / n_rays
                    t_c = corners_st[i_c, 1] / n_across
                    corners_grid[i_c, 0] = s_c * (1.0 - t_c)
                    corners_grid[i_c, 1] = s_c * t_c
            tri_origin = qd.Vector([tri_corners[0, 0], tri_corners[0, 1], tri_corners[0, 2]])
            tri_edges = qd.Matrix.zero(gs.qd_float, 2, 3)
            for i_d in qd.static(range(3)):
                tri_edges[0, i_d] = tri_corners[1, i_d] - tri_corners[0, i_d]
                tri_edges[1, i_d] = tri_corners[2, i_d] - tri_corners[0, i_d]
            i_fg = window[6]
            i_bd = sph_boundaries.fixed_geoms_idx[i_fg]
            centroid_grid = qd.Vector(
                [
                    (corners_grid[0, 0] + corners_grid[1, 0] + corners_grid[2, 0]) / 3.0,
                    (corners_grid[0, 1] + corners_grid[1, 1] + corners_grid[2, 1]) / 3.0,
                ]
            )
            centroid = tri_origin + tri_edges.transpose() @ centroid_grid
            if is_near:
                is_near = (centroid - pos).norm() < support_radius + sph_boundaries.fixed_geoms_extent_max[i_fg]
            if is_near:
                corners = corners_grid @ tri_edges
                for i_c in qd.static(range(3)):
                    for i_d in qd.static(range(3)):
                        corners[i_c, i_d] = corners[i_c, i_d] + tri_origin[i_d]
                corner_0 = qd.Vector([corners[0, 0], corners[0, 1], corners[0, 2]])
                corner_1 = qd.Vector([corners[1, 0], corners[1, 1], corners[1, 2]])
                corner_2 = qd.Vector([corners[2, 0], corners[2, 1], corners[2, 2]])
                area_normal = 0.5 * (corner_1 - corner_0).cross(corner_2 - corner_0)
                area = area_normal.norm()
                normal = area_normal / qd.max(area, eps)
                extent = qd.sqrt(
                    qd.max(
                        (corner_0 - centroid).norm_sqr(),
                        (corner_1 - centroid).norm_sqr(),
                        (corner_2 - centroid).norm_sqr(),
                    )
                )
    return is_near, corners, normal, area, extent, vel, i_bd


@qd.kernel(fastcache=True)
def kernel_select_sph_fixed_geoms(
    geoms_state: GeomsState, sph_boundaries: SPHBoundaries, sph_info: SPHInfo, legacy_coupler_info: LegacyCouplerInfo
):
    """
    List the fixed rigid geoms whose box, widened by the support radius and the largest extent of their segments,
    overlaps the domain of the SPH liquid in each environment, in ascending order.

    The box of a geom spans its grid in its own frame, taken to the world frame as the box bounding it.
    """
    n_fixed_geoms, n_envs = sph_boundaries.fixed_envs_geoms.shape
    cell_size = sph_boundaries.fixed_cell_size[None]
    lower, upper = sph_info.boundary.lower[None], sph_info.boundary.upper[None]
    for i_b in range(n_envs):
        n_geoms = 0
        for i_fg in range(n_fixed_geoms):
            i_g = sph_boundaries.fixed_geoms_idx[i_fg]
            if sph_boundaries.geoms_is_active[i_g, i_b]:
                margin = sph_info.support_radius[None] + sph_boundaries.fixed_geoms_extent_max[i_fg]
                half_size = 0.5 * sph_boundaries.fixed_geoms_grid_res[i_fg] * cell_size + margin
                center_local = (
                    sph_boundaries.fixed_geoms_grid_lower[i_fg]
                    + 0.5 * sph_boundaries.fixed_geoms_grid_res[i_fg] * cell_size
                )
                rot = qd_quat_to_R(geoms_state.quat[i_g, i_b], legacy_coupler_info.EPS[None])
                center = rot @ center_local + geoms_state.pos[i_g, i_b]
                half_size_world = qd.abs(rot) @ half_size
                if (center + half_size_world - lower).min() > 0.0 and (upper - center + half_size_world).min() > 0.0:
                    sph_boundaries.fixed_envs_geoms[n_geoms, i_b] = i_fg
                    n_geoms = n_geoms + 1
        sph_boundaries.fixed_envs_n_geoms[i_b] = n_geoms


@qd.kernel(fastcache=True)
def kernel_place_sph_boundary_segments(
    fem_state: FEMState,
    sph_state: SPHState,
    geoms_state: GeomsState,
    links_state: LinksState,
    sph_boundaries: SPHBoundaries,
    fem_info: FEMInfo,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    legacy_coupler_info: LegacyCouplerInfo,
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
        # The single slot of the segments of fixed geoms bounding the liquid alone holds no segment
        if qd.static(not (legacy_coupler_config.has_fem_sph or legacy_coupler_config.has_rigid_sph)):
            is_active = False
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
        sph_boundaries.segments.normal[i_sg, i_b] = area_normal / qd.max(area, legacy_coupler_info.EPS[None])
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
    eps = legacy_coupler_info.EPS[None]
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
        # The lowest and highest boundary of the segments around the particle, none to start with
        boundary_lo = 2147483647
        boundary_hi = -1
        if sph_state.particles.is_active[i_p, i_b]:
            pos = sph_state.particles.pos[i_p, i_b]
            # The segments of each source are summed in its frame, then taken to the world frame: the hash grid of the
            # segments placed in the world frame, then the grid of each fixed geom near the domain
            n_sources = 1
            if qd.static(legacy_coupler_config.has_fixed_rigid_sph):
                n_sources = 1 + sph_boundaries.fixed_envs_n_geoms[i_b]
            for i_src in range(n_sources):
                pos_src, rot, cell_base, n_cells = func_boundary_source_cells(
                    i_src, i_b, pos, geoms_state, sph_boundaries, support_radius, legacy_coupler_config, eps
                )
                src_volume = gs.qd_float(0.0)
                src_volume_grad = qd.Vector.zero(gs.qd_float, 3)
                src_moment = qd.Matrix.zero(gs.qd_float, 3, 3)
                for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
                    item_start, n_items = func_boundary_source_cell(
                        i_src, i_b, cell_base + offset, sph_boundaries, legacy_coupler_config
                    )
                    # The lanes stride the segments of the world frame, and the segments of the window of each
                    # triangle of a fixed geom, as a triangle holds many. They take the window as a whole, its columns
                    # left as wide in every row, as a row holds too few segments for them.
                    n_item_lanes = qd.select(i_src == 0, N_LANES, 1)
                    n_part_lanes = N_LANES // n_item_lanes
                    for i_q_ in range((n_items + n_item_lanes - 1) // n_item_lanes):
                        i_q = i_q_ * n_item_lanes + qd.select(i_src == 0, i_lane, 0)
                        if i_q < n_items:
                            i_item = item_start + i_q
                            cell = cell_base + offset
                            n_rows, window, tri_corners, coords = func_boundary_source_window(
                                i_src,
                                i_b,
                                i_item,
                                cell,
                                cell_base,
                                pos_src,
                                sph_boundaries,
                                legacy_coupler_config,
                                support_radius,
                            )
                            n_row_iters = n_rows
                            if qd.static(N_LANES > 1):
                                n_row_iters = qd.min(n_rows, 1)
                            for i_row_ in range(n_row_iters):
                                i_row = window[1] + i_row_
                                col_lo, n_cols = window[3], window[4]
                                n_parts = qd.select(i_src == 0, 1, 2 * n_rows * n_cols)
                                if qd.static(N_LANES == 1):
                                    col_lo, n_cols = func_boundary_window_row(i_row, window, coords)
                                    n_parts = qd.select(i_src == 0, 1, 2 * n_cols)
                                for i_part_ in range((n_parts + n_part_lanes - 1) // n_part_lanes):
                                    i_part = i_part_ * n_part_lanes + qd.select(i_src == 0, 0, i_lane)
                                    i_part_row, i_part_col = i_row, col_lo + i_part // 2
                                    if qd.static(N_LANES > 1):
                                        i_part_row = i_row + i_part // 2 // n_cols
                                        i_part_col = col_lo + i_part // 2 % n_cols
                                    is_down = i_part % 2 == 1
                                    is_near, corners, normal, area, extent, vel, i_bd = func_boundary_source_segment(
                                        i_src,
                                        i_b,
                                        i_part_row,
                                        i_part_col,
                                        is_down,
                                        window,
                                        pos_src,
                                        tri_corners,
                                        sph_boundaries,
                                        legacy_coupler_config,
                                        support_radius,
                                        eps,
                                    )
                                    if is_near and i_part < n_parts:
                                        seg_volume, seg_volume_grad, seg_moment = func_segment_volume(
                                            pos_src,
                                            corners,
                                            normal,
                                            area,
                                            extent,
                                            sph_boundaries.quad_bary,
                                            sph_boundaries.quad_weight,
                                            support_radius,
                                            kernel_norm,
                                        )
                                        src_volume = src_volume + seg_volume
                                        src_volume_grad = src_volume_grad + seg_volume_grad
                                        src_moment = src_moment + seg_moment
                                        vel_flux = vel_flux + vel.dot(seg_volume_grad)
                                        surface_dist = qd.min(surface_dist, func_segment_dist(pos_src, corners, eps))
                                        if seg_volume_grad.norm_sqr() > 0.0:
                                            boundary_lo = qd.min(boundary_lo, i_bd)
                                            boundary_hi = qd.max(boundary_hi, i_bd)
                volume = volume + src_volume
                volume_grad = volume_grad + rot @ src_volume_grad
                moment = moment + rot @ src_moment @ rot.transpose()
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
def func_boundary_slot(i_bd: int, boundaries_idx: qd.types.vector(4, qd.i32), n_boundaries: int):
    """
    Return the slot of a boundary among those around a particle, with the boundaries and their number updated.

    A new boundary takes the next free slot, and the boundaries beyond the last slot share it, marked -2.
    """
    N_BOUNDARIES = qd.static(boundaries_idx.n)
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
    return i_k, boundaries_idx, n_boundaries


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

    Where no velocity meets all the bounds within sqrt(3) times the largest bound in magnitude, the speed that three
    bounds at right angles call for, the bounds are all lowered by the least common slack that lets such a velocity
    meet them. This covers a particle squeezed between two approaching boundaries, whose nearly opposite normals would
    call for a speed growing without limit as they turn opposite, and violates each of the opposing bounds by the same
    amount. By duality, a velocity meeting the bounds is at least as fast as the mean of the bounds over the length of
    the mean of their normals, for any non-negative weights summing to one. The slack is the largest excess of the mean
    of the bounds over the speed limit times that length, at the weights cancelling the normals of each subset best.
    The nearest velocity then meets as equalities the first set of at most 3 bounds whose multipliers are non-negative
    and whose solution meets all the others, the solution of a strictly convex problem being unique.
    """
    speed_max = qd.sqrt(3.0) * qd.abs(bounds).max()
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
                weights_sum = weights.sum()
                if weights.min() >= 0.0 and weights_sum > 0.0:
                    weights = weights / weights_sum
                    normals_mean = mixing @ weights
                    normals_mean_norm = qd.Vector([normals_mean[0], normals_mean[1], normals_mean[2]]).norm()
                    slack = qd.max(slack, weights.dot(bounds) - speed_max * normals_mean_norm)
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


@qd.kernel(fastcache=True)
def kernel_sph_boundaries_impulse(
    sph_state: SPHState,
    geoms_state: GeomsState,
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
    eps = legacy_coupler_info.EPS[None]
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
                    n_sources = 1
                    if qd.static(legacy_coupler_config.has_fixed_rigid_sph):
                        n_sources = 1 + sph_boundaries.fixed_envs_n_geoms[i_b]
                    for i_src in range(n_sources):
                        pos_src, rot, cell_base, n_cells = func_boundary_source_cells(
                            i_src, i_b, pos, geoms_state, sph_boundaries, support_radius, legacy_coupler_config, eps
                        )
                        for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
                            item_start, n_items = func_boundary_source_cell(
                                i_src, i_b, cell_base + offset, sph_boundaries, legacy_coupler_config
                            )
                            for i_q in range(n_items):
                                i_item = item_start + i_q
                                cell = cell_base + offset
                                n_rows, window, tri_corners, coords = func_boundary_source_window(
                                    i_src,
                                    i_b,
                                    i_item,
                                    cell,
                                    cell_base,
                                    pos_src,
                                    sph_boundaries,
                                    legacy_coupler_config,
                                    support_radius,
                                )
                                for i_row_ in range(n_rows):
                                    i_row = window[1] + i_row_
                                    col_lo, n_cols = func_boundary_window_row(i_row, window, coords)
                                    n_parts = qd.select(i_src == 0, 1, 2 * n_cols)
                                    for i_part in range(n_parts):
                                        i_col = col_lo + i_part // 2
                                        is_down = i_part % 2 == 1
                                        is_near, corners, normal, area, extent, vel, i_bd = (
                                            func_boundary_source_segment(
                                                i_src,
                                                i_b,
                                                i_row,
                                                i_col,
                                                is_down,
                                                window,
                                                pos_src,
                                                tri_corners,
                                                sph_boundaries,
                                                legacy_coupler_config,
                                                support_radius,
                                                eps,
                                            )
                                        )
                                        if is_near:
                                            seg_volume, seg_volume_grad, _ = func_segment_volume(
                                                pos_src,
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
                                                i_k, boundaries_idx, n_boundaries = func_boundary_slot(
                                                    i_bd, boundaries_idx, n_boundaries
                                                )
                                                volumes[i_k] = volumes[i_k] + seg_volume
                                                seg_volume_grad_world = rot @ seg_volume_grad
                                                for i_d in qd.static(range(3)):
                                                    volume_grads[i_k, i_d] = (
                                                        volume_grads[i_k, i_d] + seg_volume_grad_world[i_d]
                                                    )
                                                vel_fluxes[i_k] = vel_fluxes[i_k] + vel.dot(seg_volume_grad)
                                                surface_dists[i_k] = qd.min(
                                                    surface_dists[i_k], func_segment_dist(pos_src, corners, eps)
                                                )
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
