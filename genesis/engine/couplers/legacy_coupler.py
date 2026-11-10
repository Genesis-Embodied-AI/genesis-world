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
from genesis.engine.solvers.fem_solver import SURFACES, FEMInfo, FEMState
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
from genesis.engine.solvers.rigid.abd.misc import func_apply_coupling_force
from genesis.engine.solvers.rigid.data import GeomsInfo, GeomsState, LinksInfo, LinksState, RigidInfo
from genesis.engine.solvers.sph_solver import (
    SPHBoundaryGhosts,
    SPHInfo,
    SPHState,
    func_cubic_kernel_derivative,
    kernel_sum_boundary_ghosts,
)
from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.data import ENVS, Config, Data, DataItem, DataKind, Size, eoa, field, iter_data, opt_in
from genesis.utils.geom import (
    PARTICLES,
    SpatialHashGrid,
    func_hash_cell_to_slot,
    func_hash_n_cells,
    func_hash_pos_to_grid,
    func_hash_sort_particles,
    func_hash_wrap_cell,
    hash_grid_sizes,
    qd_inv_transform_by_trans_quat,
    qd_transform_by_quat,
    qd_transform_by_trans_quat,
)
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from genesis.utils.raycast_qd import triangle_face_normal
from genesis.utils.sdf import SDFInfo

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom
    from genesis.engine.simulator import Simulator


PBD_PARTICLES = Size("pbd_particles")
# The arrays of the coupling of the rigid geoms with the PBD particles, which a scene without it leaves out
RIGID_PBD = "rigid_pbd"


# The samples of the surfaces of the FEM entities and of the rigid geoms, which a scene without their coupling with the
# SPH liquid leaves out
FEM_SPH = "fem_sph"
RIGID_SPH = "rigid_sph"
FEM_SAMPLES = Size("fem_samples")
RIGID_SAMPLES = Size("rigid_samples")
RIGID_LINKS = Size("rigid_links")


class SPHBoundaries(Data, kind=DataKind.SCRATCH):
    """The surfaces of the FEM entities and of the rigid geoms as moving boundaries of the SPH liquid."""

    # The ghost particle of each sample of a surface, half a particle size inside it, which stands for the layer of
    # liquid one particle size thick under the share of the surface of the sample and moves with it. The samples of the
    # FEM entities come first, then those of the rigid geoms.
    ghosts: SPHBoundaryGhosts
    # The impulse that the SPH particles give each ghost particle over the substep, which its surface takes
    ghosts_impulse: qd.Tensor = field((float, 3), shape=(PARTICLES, ENVS))
    # The triangle of each FEM sample, its barycentric coordinates in the rest triangle, and its share of the triangle
    # area
    samples_tri_idx: qd.Tensor = field(int, shape=(FEM_SAMPLES,), kind=DataKind.CONSTANT, labels=FEM_SPH)
    samples_bary: qd.Tensor = field((float, 3), shape=(FEM_SAMPLES,), kind=DataKind.CONSTANT, labels=FEM_SPH)
    samples_area_frac: qd.Tensor = field(float, shape=(FEM_SAMPLES,), kind=DataKind.CONSTANT, labels=FEM_SPH)
    # The FEM samples of each surface triangle span tris_sample_start[i_s] to tris_sample_start[i_s + 1]
    tris_sample_start: qd.Tensor = field(int, shape=(SURFACES + 1,), kind=DataKind.CONSTANT, labels=FEM_SPH)
    # The geom of each rigid sample, its position and the outward normal of its triangle in the frame of the geom, and
    # its share of the area of the triangle
    rigid_samples_geom_idx: qd.Tensor = field(int, shape=(RIGID_SAMPLES,), kind=DataKind.CONSTANT, labels=RIGID_SPH)
    rigid_samples_pos: qd.Tensor = field((float, 3), shape=(RIGID_SAMPLES,), kind=DataKind.CONSTANT, labels=RIGID_SPH)
    rigid_samples_normal: qd.Tensor = field(
        (float, 3), shape=(RIGID_SAMPLES,), kind=DataKind.CONSTANT, labels=RIGID_SPH
    )
    rigid_samples_area: qd.Tensor = field(float, shape=(RIGID_SAMPLES,), kind=DataKind.CONSTANT, labels=RIGID_SPH)
    # The mass of liquid in the ghost layer of each rigid link, at the rest density of the densest liquid of the scene.
    # The liquid pushes a link a substep late, which diverges for a body lighter than the liquid it drags, so this mass
    # moves to the inertia side of the link while gravity keeps acting on the true mass, and a link at rest takes the
    # full force of the liquid.
    # FIXME: The added mass also slows down the transients of a light body. Compensating it with the force of the added
    # mass at the previous acceleration of the link, which would restore them, makes the light thin bodies diverge again.
    links_added_mass: qd.Tensor = field(float, shape=(RIGID_LINKS,), kind=DataKind.CONSTANT, labels=RIGID_SPH)
    rho_rest: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, labels=RIGID_SPH)


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


def sample_triangles(tris_verts_pos: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample triangles by halving the longest edge of their parts until all the edges of the parts fit within a size.

    The samples of an elongated triangle are as many as its length needs, and sit at the centroids of the parts. Return
    the triangle of each sample, its barycentric coordinates in the triangle and its share of the triangle area, the
    samples sorted by triangle.
    """
    n_tris = len(tris_verts_pos)
    parts_tri_idx, parts_bary, parts_area_frac = (
        np.arange(n_tris),
        np.tile(np.eye(3), (n_tris, 1, 1)),
        np.ones(n_tris),
    )
    samples_parts = [(parts_tri_idx[:0], parts_bary[:0].mean(axis=1), parts_area_frac[:0])]
    while len(parts_tri_idx) > 0:
        parts_verts_pos = parts_bary @ tris_verts_pos[parts_tri_idx]
        parts_edges = np.linalg.norm(parts_verts_pos - np.roll(parts_verts_pos, -1, axis=1), axis=-1)
        is_split = parts_edges.max(axis=-1) > size
        samples_parts.append((parts_tri_idx[~is_split], parts_bary[~is_split].mean(axis=1), parts_area_frac[~is_split]))
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
        parts_area_frac = np.tile(0.5 * parts_area_frac[is_split], 2)
    samples_tri_idx, samples_bary, samples_area_frac = [np.concatenate(arrays) for arrays in zip(*samples_parts)]
    samples_order = np.argsort(samples_tri_idx, kind="stable")
    return samples_tri_idx[samples_order], samples_bary[samples_order], samples_area_frac[samples_order]


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
        opt_in(SPHState.boundary, cond=lambda: self.has_fem_sph or bool(self.sph_geoms))

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
    def sph_geoms(self) -> list["RigidGeom"]:
        """The rigid geoms that couple with the SPH liquid through their surface."""
        if not self.rigid_couplings[1]:
            return []
        # The walls of the domain bound the liquid where a plane would
        return [
            geom
            for entity in self.rigid_solver.entities
            for link in entity.links
            for geom in link.geoms
            if geom.needs_coup and geom.type != gs.GEOM_TYPE.PLANE
        ]

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
        self.legacy_coupler_config = LegacyCouplerConfig(
            has_rigid_mpm=has_rigid_mpm,
            has_rigid_sph=bool(self.sph_geoms),
            has_rigid_pbd=has_rigid_pbd,
            has_rigid_fem=has_rigid_fem,
            has_mpm_sph=has_mpm_sph,
            has_mpm_pbd=has_mpm_pbd,
            has_fem_mpm=self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm,
            has_fem_sph=self.has_fem_sph,
            is_links_info_batched=self.rigid_solver.rigid_config.batch_links_info,
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
                animate_inv_dt=min(1.0 / self.pbd_solver.substep_dt, 50.0),
                softness_min=1e-10,
                mpm_sph_stencil_size=mpm_sph_stencil_size,
                mpm_pbd_stencil_size=mpm_pbd_stencil_size,
            )
        )

        if self.legacy_coupler_config.has_rigid_mpm and self.mpm_solver.enable_CPIC:
            kernel_reset_mpm_rigid_normal(self.sim.scene._sanitize_envs_idx(None), self.mpm_solver.mpm_state)

        self._build_sph_boundaries()

    def _build_sph_boundaries(self) -> None:
        has_fem_sph, has_rigid_sph = self.legacy_coupler_config.has_fem_sph, self.legacy_coupler_config.has_rigid_sph
        if not (has_fem_sph or has_rigid_sph):
            self.sph_boundaries = SPHBoundaries(unused=True)
            return

        fem_tris_verts_pos = np.zeros((0, 3, 3), dtype=gs.np_float)
        if has_fem_sph:
            fem_tris_verts_pos = np.concatenate(
                [
                    tensor_to_array(entity.init_positions, dtype=gs.np_float)[entity.surface_triangles]
                    for entity in self.fem_solver.entities
                ]
            )
        samples_tri_idx, samples_bary, samples_area_frac = sample_triangles(
            fem_tris_verts_pos, self.sph_solver.particle_size
        )
        tris_sample_start = np.concatenate(
            [[0], np.cumsum(np.bincount(samples_tri_idx, minlength=len(fem_tris_verts_pos)))]
        )

        geoms = self.sph_geoms
        rigid_tris_verts_pos = np.zeros((0, 3, 3), dtype=gs.np_float)
        if geoms:
            rigid_tris_verts_pos = np.concatenate([geom.init_verts[geom.init_faces] for geom in geoms])
        rigid_tris_geom_idx = np.repeat([geom.idx for geom in geoms], [len(geom.init_faces) for geom in geoms])
        rigid_tris_area_normal = 0.5 * np.cross(
            rigid_tris_verts_pos[:, 1] - rigid_tris_verts_pos[:, 0],
            rigid_tris_verts_pos[:, 2] - rigid_tris_verts_pos[:, 0],
        )
        rigid_tris_area = np.linalg.norm(rigid_tris_area_normal, axis=-1)
        rigid_samples_tri_idx, rigid_samples_bary, rigid_samples_area_frac = sample_triangles(
            rigid_tris_verts_pos, self.sph_solver.particle_size
        )

        rho_rest = max([entity.material.rho for entity in self.sph_solver.entities])
        self.sph_boundaries = SPHBoundaries(
            sizes=dict(
                particles=len(samples_tri_idx) + len(rigid_samples_tri_idx),
                envs=self.sim._B,
                fem_samples=len(samples_tri_idx),
                surfaces=len(fem_tris_verts_pos),
                rigid_samples=len(rigid_samples_tri_idx),
                rigid_links=self.rigid_solver.n_links,
                **hash_grid_sizes(self.sph_solver.hash_grid_res),
            ),
            values=dict(
                ghosts=dict(
                    hash_grid=dict(
                        cell_size=self.sph_solver.hash_grid_cell_size, grid_res=self.sph_solver.hash_grid_res
                    )
                ),
                rho_rest=rho_rest,
            ),
            env_first={(SPHBoundaries.ghosts, SPHBoundaries.ghosts_impulse): False},
            unused=dict(fem_sph=not has_fem_sph, rigid_sph=not has_rigid_sph),
        )
        if has_fem_sph:
            self.sph_boundaries.samples_tri_idx.from_numpy(samples_tri_idx.astype(gs.np_int))
            self.sph_boundaries.samples_bary.from_numpy(samples_bary.astype(gs.np_float))
            self.sph_boundaries.samples_area_frac.from_numpy(samples_area_frac.astype(gs.np_float))
            self.sph_boundaries.tris_sample_start.from_numpy(tris_sample_start.astype(gs.np_int))
        if has_rigid_sph:
            self.sph_boundaries.rigid_samples_geom_idx.from_numpy(
                rigid_tris_geom_idx[rigid_samples_tri_idx].astype(gs.np_int)
            )
            self.sph_boundaries.rigid_samples_pos.from_numpy(
                np.einsum("ni,nij->nj", rigid_samples_bary, rigid_tris_verts_pos[rigid_samples_tri_idx]).astype(
                    gs.np_float
                )
            )
            self.sph_boundaries.rigid_samples_normal.from_numpy(
                (rigid_tris_area_normal / rigid_tris_area[:, None])[rigid_samples_tri_idx].astype(gs.np_float)
            )
            rigid_samples_area = rigid_samples_area_frac * rigid_tris_area[rigid_samples_tri_idx]
            self.sph_boundaries.rigid_samples_area.from_numpy(rigid_samples_area.astype(gs.np_float))
            rigid_tris_link_idx = np.repeat([geom.link.idx for geom in geoms], [len(geom.init_faces) for geom in geoms])
            links_ghosts_volume = np.bincount(
                rigid_tris_link_idx[rigid_samples_tri_idx],
                weights=rigid_samples_area * self.sph_solver.particle_size,
                minlength=self.rigid_solver.n_links,
            )
            self.sph_boundaries.links_added_mass.from_numpy((rho_rest * links_ghosts_volume).astype(gs.np_float))

        # The liquid sampled inside a coupled body, or closer to its surface than the half particle size at which liquid
        # rests against a wall, would start within its boundary, which pushes it out at once at any speed, so it is
        # removed. The surfaces of the bodies at rest form one triangle soup, signed by its winding number.
        geoms_pos = qd_to_numpy(self.rigid_solver.dyn_state.geoms.pos, transpose=True)[0, rigid_tris_geom_idx, None]
        geoms_quat = qd_to_numpy(self.rigid_solver.dyn_state.geoms.quat, transpose=True)[0, rigid_tris_geom_idx, None]
        tris_verts_pos = np.concatenate(
            [fem_tris_verts_pos, gu.transform_by_trans_quat(rigid_tris_verts_pos, geoms_pos, geoms_quat)]
        )
        particles_pos = qd_to_numpy(self.sph_solver.sph_state.particles.pos, transpose=True)[0]
        particles_dist, *_ = igl.signed_distance(
            particles_pos.astype(np.float64),
            tris_verts_pos.reshape((-1, 3)).astype(np.float64),
            np.arange(tris_verts_pos.size // 3, dtype=np.int64).reshape((-1, 3)),
            sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
        )
        is_inside = particles_dist < 0.5 * self.sph_solver.particle_size
        for entity in self.sph_solver.entities:
            n_inside = np.count_nonzero(is_inside[entity.particle_start : entity.particle_start + entity.n_particles])
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
            kernel_place_sph_boundary_ghosts(
                self.fem_solver.fem_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.sph_boundaries,
                self.fem_solver.fem_info,
                self.sph_solver.sph_info,
                self.rigid_solver.dyn_info.geoms,
                self.legacy_coupler_config,
            )
            kernel_sum_boundary_ghosts(self.sph_boundaries.ghosts, self.sph_solver.sph_state, self.sph_solver.sph_info)

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
                self.legacy_coupler_config,
                self.rigid_solver.collider.collider_config,
                self.mpm_solver.force_fields,
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
    geoms_state: GeomsState,
    links_state: LinksState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    collider_config: qd.template(),
):
    """Return the velocity of a point of the given mass once it collided with every rigid geom that couples."""
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
                vel = func_point_vel_after_two_way_collision_with_rigid_geom(
                    i_g, i_b, pos_world, vel, mass, normal_rigid, influence, links_state, geoms_info, rigid_info
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
    legacy_coupler_config: qd.template(),
    collider_config: qd.template(),
    force_fields: qd.template(),
):
    """Update the velocity of the cells of the MPM grid, which the other solvers exchange momentum with."""
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
                    geoms_state,
                    links_state,
                    geoms_info,
                    rigid_info,
                    sdf_info,
                    legacy_coupler_info,
                    collider_config,
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
                            func_apply_coupling_force(
                                geoms_info.link_idx[i_g],
                                i_b,
                                pos,
                                force=-delta_mv / rigid_info.substep_dt[None],
                                links_state=links_state,
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

        # FEM <-> SPH, the vertex taking the impulses of the ghost particles of its triangles by its barycentric weight
        if qd.static(legacy_coupler_config.has_fem_sph):
            for i_s_ in range(n_tris):
                i_s = fem_info.surface_vertices_tri_idx[i_tri_start + i_s_]
                tri2v = fem_info.surfaces.tri2v[i_s]
                for i_c in qd.static(range(3)):
                    if tri2v[i_c] == i_v:
                        for i_g in range(
                            sph_boundaries.tris_sample_start[i_s], sph_boundaries.tris_sample_start[i_s + 1]
                        ):
                            vel_fem_sv = (
                                vel_fem_sv
                                + sph_boundaries.samples_bary[i_g][i_c]
                                * sph_boundaries.ghosts_impulse[i_g, i_b]
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
        func_apply_coupling_force(geoms_info.link_idx[i_g], i_b, pos_world, force, links_state)

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
def kernel_place_sph_boundary_ghosts(
    fem_state: FEMState,
    geoms_state: GeomsState,
    links_state: LinksState,
    sph_boundaries: SPHBoundaries,
    fem_info: FEMInfo,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    legacy_coupler_config: qd.template(),
):
    """Place the ghost particles of the moving boundaries of the SPH liquid at the start of the substep, and sort them."""
    particle_size = sph_info.particle_size[None]
    n_fem_samples = 0
    if qd.static(legacy_coupler_config.has_fem_sph):
        n_fem_samples = sph_boundaries.samples_tri_idx.shape[0]
    for i_g, i_b in qd.ndrange(
        *sph_boundaries.ghosts.particles.pos.shape, axes=qd.static(eoa(sph_boundaries.ghosts.particles.pos))
    ):
        pos = qd.Vector.zero(gs.qd_float, 3)
        vel = qd.Vector.zero(gs.qd_float, 3)
        volume = gs.qd_float(0.0)
        if qd.static(legacy_coupler_config.has_fem_sph):
            if i_g < n_fem_samples:
                i_s = sph_boundaries.samples_tri_idx[i_g]
                tri2v = fem_info.surfaces.tri2v[i_s]
                bary = sph_boundaries.samples_bary[i_g]
                for i_c in qd.static(range(3)):
                    i_v = tri2v[i_c]
                    pos = pos + bary[i_c] * fem_state.vertices.pos[i_v, i_b]
                    vel = vel + bary[i_c] * fem_state.vertices.vel[i_v, i_b]
                i_v0, i_v1, i_v2 = tri2v[0], tri2v[1], tri2v[2]
                pos_0 = fem_state.vertices.pos[i_v0, i_b]
                area_normal = 0.5 * (fem_state.vertices.pos[i_v1, i_b] - pos_0).cross(
                    fem_state.vertices.pos[i_v2, i_b] - pos_0
                )
                area = area_normal.norm()
                pos = pos - (0.5 * particle_size / area) * area_normal
                volume = sph_boundaries.samples_area_frac[i_g] * area * particle_size
        if qd.static(legacy_coupler_config.has_rigid_sph):
            if i_g >= n_fem_samples:
                # The ghost particle moves with the point of the link it sits on
                i_r = i_g - n_fem_samples
                i_geom = sph_boundaries.rigid_samples_geom_idx[i_r]
                geom_pos, geom_quat = geoms_state.pos[i_geom, i_b], geoms_state.quat[i_geom, i_b]
                normal = qd_transform_by_quat(sph_boundaries.rigid_samples_normal[i_r], geom_quat)
                pos = qd_transform_by_trans_quat(sph_boundaries.rigid_samples_pos[i_r], geom_pos, geom_quat)
                pos = pos - 0.5 * particle_size * normal
                i_l = geoms_info.link_idx[i_geom]
                vel = func_vel_at_point(i_l, i_b, pos, links_state)
                volume = sph_boundaries.rigid_samples_area[i_r] * particle_size
        sph_boundaries.ghosts.particles.pos[i_g, i_b] = pos
        sph_boundaries.ghosts.particles.vel[i_g, i_b] = vel
        sph_boundaries.ghosts.particles.volume[i_g, i_b] = volume
        sph_boundaries.ghosts.particles.is_active[i_g, i_b] = True

    func_hash_sort_particles(
        sph_boundaries.ghosts.particles.pos,
        sph_boundaries.ghosts.particles.is_active,
        sph_boundaries.ghosts.hash_grid,
    )


@qd.kernel
def kernel_sph_boundaries_impulse(
    sph_state: SPHState,
    links_state: LinksState,
    sph_boundaries: SPHBoundaries,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    legacy_coupler_config: qd.template(),
):
    """
    Sum the impulse of the SPH particles on each ghost particle of the moving boundaries over the substep.

    The impulse is opposite to the one the SPH particles take from the ghost particle (see SPHBoundaryParticles). The
    links of the rigid geoms take the impulse of their ghost particles as a coupling force, the fixed ones aside, while
    the FEM vertices take theirs in kernel_fem_surface_force.
    """
    h, k = sph_info.support_radius[None], sph_info.kernel_coef[None]
    n_fem_samples = 0
    if qd.static(legacy_coupler_config.has_fem_sph):
        n_fem_samples = sph_boundaries.samples_tri_idx.shape[0]
    for i_g, i_b in qd.ndrange(
        *sph_boundaries.ghosts_impulse.shape, axes=qd.static(eoa(sph_boundaries.ghosts_impulse))
    ):
        impulse = qd.Vector.zero(gs.qd_float, 3)
        ghost_pos = sph_boundaries.ghosts.particles.pos[i_g, i_b]
        ghost_volume = sph_boundaries.ghosts.particles.volume[i_g, i_b]
        gravity = sph_info.gravity[i_b]
        base = func_hash_wrap_cell(func_hash_pos_to_grid(ghost_pos, sph_state.hash_grid) - 1, sph_state.hash_grid)
        n_cells = func_hash_n_cells(3, sph_state.hash_grid)
        for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
            i_slot = func_hash_cell_to_slot(base + offset, sph_state.hash_grid)
            slot_start = sph_state.hash_grid.slot_start[i_slot, i_b]
            for i_r in range(slot_start, slot_start + sph_state.hash_grid.slot_size[i_slot, i_b]):
                r = ghost_pos - sph_state.particles_reordered.pos[i_r, i_b]
                if r.norm() < h:
                    i_p = sph_state.hash_grid.particles_idx[i_r, i_b]
                    grad = -ghost_volume * func_cubic_kernel_derivative(r, h, k, sph_info.kernel_dist_min[None])
                    impulse_coef = sph_state.boundary.impulse_coef[i_p, i_b] + sph_state.boundary.impulse_gravity_coef[
                        i_p, i_b
                    ] * gravity.dot(r)
                    impulse = impulse + impulse_coef * grad
        sph_boundaries.ghosts_impulse[i_g, i_b] = impulse

        if qd.static(legacy_coupler_config.has_rigid_sph):
            if i_g >= n_fem_samples:
                i_geom = sph_boundaries.rigid_samples_geom_idx[i_g - n_fem_samples]
                i_l = geoms_info.link_idx[i_geom]
                I_l = [i_l, i_b] if qd.static(legacy_coupler_config.is_links_info_batched) else i_l
                if not links_info.is_fixed[I_l]:
                    # The share of the ghost particle in the added mass of the link (see 'links_added_mass')
                    mass = links_info.inertial_mass[I_l]
                    ghost_mass = sph_boundaries.rho_rest[None] * ghost_volume
                    force = (impulse / sph_info.substep_dt[None] - ghost_mass * gravity) * (
                        mass / (mass + sph_boundaries.links_added_mass[i_l])
                    )
                    func_apply_coupling_force(i_l, i_b, ghost_pos, force, links_state)
