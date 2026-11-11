import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary, func_impose_cube_boundary
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.options.solvers import SPHOptions
from genesis.utils.data import ENVS, Config, ConfigItem, Data, DataItem, DataKind, eoa, field, is_opted_in, iter_data
from genesis.utils.geom import (
    PARTICLES,
    SpatialHashGrid,
    func_hash_cell_to_slot,
    func_hash_n_cells,
    func_hash_pos_to_grid,
    func_hash_sort_particles,
    func_hash_wrap_cell,
    hash_grid_sizes,
)
from genesis.utils.misc import get_gpu_core_count
from genesis.utils.simt import (
    BLOCK_LOG2_LANES,
    ExactSums,
    func_exact_sum_add,
    func_exact_sum_take,
    qd_block_sum,
    requires_deterministic_accumulation,
)

from .base_solver import SimulatorState
from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


# The sums of the errors of the DFSPH solve, exact for a fixed order of the sums, or in any order otherwise
DFSPH_EXACT = "dfsph_exact"
DFSPH_ATOMIC = "dfsph_atomic"


class SPHParticles(Data, kind=DataKind.STATE, shape=(PARTICLES, ENVS)):
    """
    Runtime state of the smoothed particle hydrodynamics (SPH) particles, one value per particle and environment.

    `p` is the pressure, which the density solve of divergence-free smoothed particle hydrodynamics (DFSPH) carries over
    from one substep to the next, `dfsph_factor` the factor of its divergence and density solves, and `drho` the rate
    of change of the density they correct.
    """

    pos: qd.Tensor = field((float, 3))
    vel: qd.Tensor = field((float, 3))
    acc: qd.Tensor = field((float, 3))
    rho: qd.Tensor = field(float)
    p: qd.Tensor = field(float)
    dfsph_factor: qd.Tensor = field(float)
    drho: qd.Tensor = field(float)
    is_active: qd.Tensor = field(bool)


class SPHParticlesInfo(Data, kind=DataKind.CONSTANT, shape=(PARTICLES,)):
    """
    Model parameters of the smoothed particle hydrodynamics (SPH) particles, one value per particle.

    `rho` is the rest density, `mu` the viscosity and `surface_tension` the surface tension coefficient, in N/m.
    """

    rho: qd.Tensor = field(float)
    mass: qd.Tensor = field(float)
    mu: qd.Tensor = field(float)
    surface_tension: qd.Tensor = field(float)


class SPHEnvsSolve(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The convergence of the divergence and density solves of divergence-free smoothed particle hydrodynamics (DFSPH) in
    each environment.

    `err` is the density error summed over the active particles, the same whatever the order of the terms so that the
    solve iterates as many times from one run to the next, with deterministic algorithms, and `err_sum` the same sum
    otherwise. `n_active` is the number of active particles, over which it is averaged, and `is_solving` whether the
    environment iterates on.
    """

    err: ExactSums = field(labels=DFSPH_EXACT)
    err_sum: qd.Tensor = field(float, labels=DFSPH_ATOMIC)
    n_active: qd.Tensor = field(int)
    is_solving: qd.Tensor = field(bool)


class SPHSolveLoop(Data, kind=DataKind.SCRATCH, shape=()):
    """
    The passes of a solve of divergence-free smoothed particle hydrodynamics (DFSPH), which runs on the device as long
    as an environment iterates on (see kernel_solve_incompressibility).

    The flag is an ndarray on every backend, since graph.do_while takes the same ndarray on every call.
    """

    n_passes: qd.Tensor = field(int)
    is_looping: qd.Tensor = field(qd.i32, ndarray=True)


class SPHBoundaryParticles(Data, kind=DataKind.SCRATCH, shape=(PARTICLES, ENVS)):
    """The moving boundaries around each SPH particle over the current substep, in the order of the particles."""

    # The kernel volume of the solids behind the boundaries around the particle, with its gradient and first moment, in
    # the units of the walls of the domain (see func_wall_volume and func_segment_volume)
    volume: qd.Tensor = field(float)
    volume_grad: qd.Tensor = field((float, 3))
    moment: qd.Tensor = field((float, (3, 3)))
    # The velocity of the boundaries dotted with the gradient of their kernel volume, summed over the segments around the
    # particle, which moves the density like the neighbors of the particle
    vel_flux: qd.Tensor = field(float)
    # The distance from the particle to the nearest surface of the moving boundaries, negative inside a body
    surface_dist: qd.Tensor = field(float)
    # The particle pushes each segment around it by `impulse_coef * grad + impulse_gravity_coef * moment @ gravity` over
    # the substep, with `grad` and `moment` the volume gradient and the first moment of the segment
    impulse_coef: qd.Tensor = field(float)
    impulse_gravity_coef: qd.Tensor = field(float)
    # The particle pushes the segments of each of the boundaries it meets as an equality on top, by its coefficient times
    # their volume gradient, the boundaries beyond the first 3 near the particle being merged as -2 (see
    # kernel_sph_boundaries_impulse)
    boundaries_idx: qd.Tensor = field((int, 4))
    boundaries_impulse_coef: qd.Tensor = field((float, 4))
    # Whether the segments around the particle belong to several boundaries
    has_several_boundaries: qd.Tensor = field(bool)


class SPHState(Data, kind=DataKind.STATE):
    """
    Runtime state of the smoothed particle hydrodynamics (SPH) solver.

    The substep runs on the copy of the particles sorted by hash grid cell, `particles_reordered`, which the particles
    take back at its end.
    """

    particles: SPHParticles
    particles_reordered: SPHParticles = field(kind=DataKind.SCRATCH)
    # The gradient of the color field of the liquid at each sorted particle, the walls counted as liquid and every
    # particle at its rest volume, which points into the liquid across its free surface
    particles_normal_reordered: qd.Tensor = field((float, 3), shape=(PARTICLES, ENVS), kind=DataKind.SCRATCH)
    # The stiffness that the next Jacobi iteration of DFSPH applies, read by the neighbors of each sorted particle
    particles_k_reordered: qd.Tensor = field(float, shape=(PARTICLES, ENVS), kind=DataKind.SCRATCH)
    envs_solve: SPHEnvsSolve
    solve_loop: SPHSolveLoop
    # The moving boundaries around each particle, which a coupler opts in
    boundary: SPHBoundaryParticles = field(opt_in=True)
    hash_grid: SpatialHashGrid
    # The error flags of each env, which no kernel of the solver raises (see 'ErrnoMixin.register_errors')
    errno: qd.Tensor = field(int, shape=(ENVS,))


class SPHInfo(Data, kind=DataKind.CONSTANT):
    """Model parameters of the smoothed particle hydrodynamics (SPH) solver."""

    particles: SPHParticlesInfo
    boundary: CubeBoundary
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float, shape=())
    particle_size: qd.Tensor = field(float, shape=())
    # The volume a particle stands for
    particle_volume: qd.Tensor = field(float, shape=())
    support_radius: qd.Tensor = field(float, shape=())
    # The normalization of the cubic kernel
    kernel_norm: qd.Tensor = field(float, shape=())
    # The distance below which the gradient of the cubic kernel is zero
    kernel_dist_min: qd.Tensor = field(float, shape=())
    # The squared distance that regularizes the viscosity of a pair of particles
    viscosity_eps: qd.Tensor = field(float, shape=())
    # The magnitude of the color gradient below which a particle takes no part in the surface tension
    surface_normal_min: qd.Tensor = field(float, shape=())
    # The average density error that ends the density solve of DFSPH, in kg/m^3
    density_err_tol: qd.Tensor = field(float, shape=())
    # The average density rate that ends the divergence solve of DFSPH, in kg/m^3/s
    divergence_err_tol: qd.Tensor = field(float, shape=())
    # The stiffness below which a pair of particles exchanges no velocity
    stiffness_eps: qd.Tensor = field(float, shape=())
    jacobi_relaxation: qd.Tensor = field(float, shape=())
    # The share of the pressure of the previous substep that warm-starts the density solve
    warm_start_scale: qd.Tensor = field(float, shape=())
    # The sum of squared kernel gradients below which a particle takes no pressure
    factor_eps: qd.Tensor = field(float, shape=())
    # The number of neighbors below which the density rate of a particle is left out of the divergence solve
    n_neighbors_min: qd.Tensor = field(int, shape=())


class SPHConfig(Config):
    """The parts of the smoothed particle hydrodynamics (SPH) solver that its kernels compile in."""

    # Whether the domain is thinner than twice the support radius along an axis, which puts the walls on both sides of
    # a particle within its support radius (see func_wall_volume)
    is_domain_thin: bool
    # Whether moving boundaries push on the particles (see SPHBoundaryParticles)
    has_boundary: bool
    # Whether the DFSPH solve sums its errors exactly, for a fixed order of the sums (see SPHEnvsSolve)
    is_deterministic: bool
    # The base-2 logarithm of the lanes of a GPU subgroup tile that take a particle and its neighbors together, one thread
    # taking it alone on CPU and once the particles of every environment outnumber the GPU cores, which one thread per
    # particle already saturates (see func_for_all_neighbors)
    log2_neighbor_lanes: int


class SPHSolver(ParticleSolver):
    # Its whole state lives in the arrays 'data' yields (see 'CheckpointMixin')
    is_checkpointable = True

    material_cls = SPH.Base

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: SPHOptions) -> None:
        super().__init__(scene, sim, options)

        self._support_radius = options._support_radius

        # DFSPH parameters
        self._df_max_error_div = options.max_divergence_error
        self._df_max_error_den = options.max_density_error_percent
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations

        # The samplers place one particle per cube of the particle size, the spacing over which the kernel sums to one
        self._particle_volume = self._particle_size**3

        self._hash_grid_cell_size = options.hash_grid_cell_size
        self._hash_grid_res = options._hash_grid_res

        # Coefficient for computing stable timestep
        self._stable_dt_coef = 1.0

        self.sph_state: SPHState | None = None
        self.sph_info: SPHInfo | None = None
        self.sph_config: SPHConfig | None = None

    def build(self) -> None:
        super().build()

        is_deterministic = requires_deterministic_accumulation()
        sizes = dict(envs=self._B, particles=self._n_particles)
        hash_grid_values = dict(cell_size=self._hash_grid_cell_size, grid_res=self._hash_grid_res)
        self.sph_state = SPHState(
            sizes=dict(
                sizes,
                # The density errors of an environment sum into one slot (see SPHEnvsSolve)
                exact_sum_slots=1,
                exact_sum_bins=2,
                **hash_grid_sizes(math.prod(self._hash_grid_res)),
            ),
            values=dict(hash_grid=hash_grid_values),
            # The units of the bins of the exact sums and the loop of the solves span no environment
            env_first={None: False, (SPHState.envs_solve.err.bins_unit, SPHState.solve_loop): None},
            unused={DFSPH_EXACT: not is_deterministic, DFSPH_ATOMIC: is_deterministic} if self.is_active else True,
        )
        self.register_errors(self.sph_state.errno, {})
        if self.is_active:
            # TODO: Support per-particle density
            # The rest density the incompressibility solves target, that of the first particle as the particles store it
            density0 = float(gs.np_float(self._entities[0].material.rho))
            density_err_tol = self._df_max_error_den * 0.01 * density0
            divergence_err_tol = 1.0 / self.substep_dt * self._df_max_error_div * 0.01 * density0
            if is_deterministic:
                # The lowest bin resolves 2^-20 of the lowest tolerance, the bins 2^32 apart (see ExactSums)
                exp_min = math.floor(math.log2(min(density_err_tol, divergence_err_tol))) - 20
                self.sph_state.envs_solve.err.bins_unit.from_numpy(
                    2.0 ** np.array([exp_min, exp_min + 32.0], dtype=gs.np_float)
                )
        self.sph_config = SPHConfig(
            is_domain_thin=self.is_active
            and bool(np.any(self.boundary_upper - self.boundary_lower < 2 * self._support_radius)),
            has_boundary=self.is_active and is_opted_in(SPHState.boundary),
            is_deterministic=is_deterministic,
            # The lanes of a tile stride the particles of each neighbor cell of the hash grid, so a tile takes one lane per
            # particle a cell holds at rest, within a block of the subgroup helpers
            log2_neighbor_lanes=(
                max(
                    0,
                    min(
                        BLOCK_LOG2_LANES, math.floor(math.log2((self._hash_grid_cell_size / self._particle_size) ** 3))
                    ),
                )
                if 0 < self._n_particles * self._B <= get_gpu_core_count()
                else 0
            ),
        )
        info_values = {}
        viscosity_eps = 0.01 * self._support_radius**2
        if self.is_active:
            info_values = dict(
                boundary=dict(lower=self.boundary_lower, upper=self.boundary_upper, restitution=0.0),
                substep_dt=self.substep_dt,
                particle_size=self._particle_size,
                particle_volume=self._particle_volume,
                support_radius=self._support_radius,
                kernel_norm=8.0 / np.pi / self._support_radius**3,
                kernel_dist_min=1e-5,
                viscosity_eps=viscosity_eps,
                # The color gradient of the particles deeper than the first two layers below the free surface is noise,
                # whose direction would corrupt the curvature (see func_task_compute_non_pressure_forces)
                surface_normal_min=0.2 / self._support_radius,
                density_err_tol=density_err_tol,
                divergence_err_tol=divergence_err_tol,
                stiffness_eps=1e-5,
                # The DFSPH factor inverts the diagonal of the pressure system alone, and the stiffness of the neighbors
                # of a particle adds up to as much again, so the diagonally scaled system has eigenvalues up to 2. A
                # full Jacobi update leaves the mode alternating from one particle to the next undamped, and overshoots
                # once the liquid compresses abruptly. Half the update cancels that mode, the others still converging.
                jacobi_relaxation=0.5,
                # FIXME: A larger share saves more density iterations, but turns a thin slab unstable from three quarters
                # on, for a reason not found yet: the excess of the carried-over pressure over hydrostatics neither drifts
                # nor scatters more between neighbors than at half the share.
                warm_start_scale=0.5,
                factor_eps=1e-6,
                n_neighbors_min=20,
            )
        self.sph_info = SPHInfo(sizes=sizes, values=info_values, unused=not self.is_active)
        self._particles_pos = self.sph_state.particles.pos
        self._particles_vel = self.sph_state.particles.vel
        self._particles_is_active = self.sph_state.particles.is_active
        self._particles_mass = self.sph_info.particles.mass
        self._build_render(has_vverts=False)

        if self.is_active:
            for entity in self.entities:
                entity._add_to_solver()

            # Compute stable timestep for the given particle properties (mu) and support radius and print warning
            # if the current timestep is larger than the stable timestep. This formula is derived from
            # 'func_task_compute_non_pressure_forces', which computes the acceleration due to non-pressure forces.
            mu_max = max([entity.material.mu for entity in self._entities])
            stable_dt = self._stable_dt_coef * viscosity_eps / mu_max
            if self.substep_dt > stable_dt:
                gs.logger.warning(
                    f"""Current timestep ({self.substep_dt:.4g}) is larger than the stable timestep ({stable_dt:.4g}) for SPH
solver. Please consider reducing the timestep, liquid viscosity (mu), or increasing the particle size."""
                )

        self._build_gravity(self.sph_info.gravity)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "SPHEntity":
        entity = SPHEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            name=name,
        )
        self.entities.append(entity)
        return entity

    def add_particles(self, is_active: bool, particle_start: int, material: SPH.Base, pos: np.ndarray) -> None:
        """Add the particles of an entity at rest, with the parameters of its material."""
        kernel_add_particles(
            particle_start,
            is_active,
            material.rho,
            material.mu,
            material.surface_tension,
            pos,
            self.sph_state,
            self.sph_info,
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _solve_incompressibility(self, solve_density: bool) -> None:
        n_iters_max = self._df_max_den_iters if solve_density else self._df_max_div_iters

        kernel_solve_incompressibility(
            n_iters_max,
            self.sph_state.solve_loop.is_looping,
            self.sph_state,
            self.sph_info,
            self.sph_config,
            solve_density,
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self):
        if not self.is_active:
            return
        kernel_reorder_particles(self.sph_state)
        kernel_compute_rho(self.sph_state, self.sph_info, self.sph_config)
        kernel_compute_dfsph_factor(self.sph_state, self.sph_info, self.sph_config)
        self._solve_incompressibility(solve_density=False)
        kernel_compute_non_pressure_forces(
            self._sim.sim_state, self.sph_state, self.sph_info, self.sph_config, self._force_fields
        )
        kernel_predict_velocity(self.sph_state, self.sph_info)
        self._solve_incompressibility(solve_density=True)

    def substep_post_coupling(self):
        if self.is_active:
            kernel_advect_position(self.sph_state, self.sph_info)
            kernel_copy_from_reordered(self.sph_state)

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        yield from iter_data(self, "sph_info")
        yield from iter_data(self, "sph_state")
        yield from iter_data(self, "sph_config")

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def _reset_particles(self, particles_idx: torch.Tensor, envs_idx: torch.Tensor, reset_deformation: bool) -> None:
        # The pressure that DFSPH carries over belongs to the neighborhood a particle leaves or joins
        kernel_clear_particles_pressure(particles_idx, envs_idx, self.sph_state)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def particle_volume(self) -> float:
        return self._particle_volume

    @property
    def support_radius(self) -> float:
        return self._support_radius

    @property
    def hash_grid_res(self) -> tuple[int, int, int]:
        return self._hash_grid_res

    @property
    def hash_grid_cell_size(self) -> float:
        return self._hash_grid_cell_size


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_cubic_kernel(dist: float, support_radius: float, kernel_norm: float):
    """
    Return the cubic spline smoothing kernel at a distance.

    `support_radius` is the support radius of the kernel (h in the SPH literature), and `kernel_norm` its
    normalization.
    """
    value = gs.qd_float(0.0)
    dist_rel = dist / support_radius
    if dist_rel <= 1.0:
        if dist_rel <= 0.5:
            q2 = dist_rel**2
            q3 = q2 * dist_rel
            value = kernel_norm * (6.0 * q3 - 6.0 * q2 + 1.0)
        else:
            value = 2 * kernel_norm * (1.0 - dist_rel) ** 3
    return value


@qd.func
def func_cubic_kernel_derivative(
    offset: qd.types.vector(3), support_radius: float, kernel_norm: float, dist_min: float
):
    """Return the gradient of the cubic spline smoothing kernel at a vector (see func_cubic_kernel)."""
    value = qd.Vector.zero(gs.qd_float, 3)
    dist = offset.norm()
    dist_rel = dist / support_radius
    if dist > dist_min and dist_rel <= 1.0:
        grad_q = offset / (dist * support_radius)
        if dist_rel <= 0.5:
            value = 6.0 * kernel_norm * dist_rel * (3.0 * dist_rel - 2.0) * grad_q
        else:
            value = -6.0 * kernel_norm * (1.0 - dist_rel) ** 2 * grad_q
    return value


@qd.func
def func_cubic_kernel_radial_integral(dist: float, support_radius: float, kernel_norm: float):
    """Return the integral of the cubic spline smoothing kernel times r^2 from 0 to a distance."""
    dist_rel = qd.min(dist / support_radius, 1.0)
    u2 = dist_rel * dist_rel
    u3 = u2 * dist_rel
    value = gs.qd_float(0.0)
    if dist_rel <= 0.5:
        value = kernel_norm * (u3 * u3 - 1.2 * u3 * u2 + u3 / 3.0)
    else:
        value = kernel_norm * (0.5**6 - 1.2 * 0.5**5 + 0.5**3 / 3.0) + 2.0 * kernel_norm * (
            (u3 / 3.0 - 0.75 * u2 * u2 + 0.6 * u3 * u2 - u3 * u3 / 6.0)
            - (0.5**3 / 3.0 - 0.75 * 0.5**4 + 0.6 * 0.5**5 - 0.5**6 / 6.0)
        )
    return value * support_radius**3


@qd.func
def func_segment_volume(
    pos: qd.types.vector(3),
    corners: qd.types.matrix(3, 3),
    normal: qd.types.vector(3),
    area: float,
    extent: float,
    quad_bary: qd.template(),
    quad_weight: qd.template(),
    support_radius: float,
    kernel_norm: float,
):
    """
    Return the kernel volume of the solid behind a surface segment around a point, with its gradient and first moment.

    These are the integrals over the solid of the kernel, of its gradient and of its gradient times the offset to the
    point, taken to the segment by the divergence theorem and integrated over it by quadrature.
    """
    volume = gs.qd_float(0.0)
    volume_grad = qd.Vector.zero(gs.qd_float, 3)
    moment = qd.Matrix.zero(gs.qd_float, 3, 3)
    corner_0 = qd.Vector([corners[0, 0], corners[0, 1], corners[0, 2]])
    corner_1 = qd.Vector([corners[1, 0], corners[1, 1], corners[1, 2]])
    corner_2 = qd.Vector([corners[2, 0], corners[2, 1], corners[2, 2]])
    centroid = (corner_0 + corner_1 + corner_2) / 3.0
    if (centroid - pos).norm() < support_radius + extent:
        # The far field of the radial antiderivative is the solid angle of the segment, singular on the surface, which
        # is taken in closed form (Van Oosterom and Strackee) and the smooth remainder by quadrature
        for i_k in range(quad_weight.shape[0]):
            offset = corners.transpose() @ quad_bary[i_k] - pos
            dist = qd.max(offset.norm(), 1e-9)
            weight = quad_weight[i_k] * area
            volume = volume + weight * func_cubic_kernel_radial_integral(
                dist, support_radius, kernel_norm
            ) / dist**3 * offset.dot(normal)
            grad = -weight * func_cubic_kernel(dist, support_radius, kernel_norm) * normal
            volume_grad = volume_grad + grad
            moment = moment + grad.outer_product(offset)
        offset_0, offset_1, offset_2 = corner_0 - pos, corner_1 - pos, corner_2 - pos
        dist_0, dist_1, dist_2 = offset_0.norm(), offset_1.norm(), offset_2.norm()
        solid_angle = 2.0 * qd.atan2(
            offset_0.dot(offset_1.cross(offset_2)),
            dist_0 * dist_1 * dist_2
            + offset_0.dot(offset_1) * dist_2
            + offset_0.dot(offset_2) * dist_1
            + offset_1.dot(offset_2) * dist_0,
        )
        volume = volume - solid_angle / (4.0 * qd.math.pi)
    moment = moment + volume * qd.Matrix.identity(gs.qd_float, 3)
    return volume, volume_grad, moment


@qd.func
def func_segment_dist(pos: qd.types.vector(3), corners: qd.types.matrix(3, 3)):
    """Return the distance from a point to a triangle."""
    corner_0 = qd.Vector([corners[0, 0], corners[0, 1], corners[0, 2]])
    corner_1 = qd.Vector([corners[1, 0], corners[1, 1], corners[1, 2]])
    corner_2 = qd.Vector([corners[2, 0], corners[2, 1], corners[2, 2]])
    area_normal = (corner_1 - corner_0).cross(corner_2 - corner_0)
    normal = area_normal / qd.max(area_normal.norm(), gs.EPS)
    pos_proj = pos - (pos - corner_0).dot(normal) * normal
    # The projection lies inside the triangle where it is on the inner side of its three edges, a degenerate triangle
    # having no inside
    is_inside = (
        area_normal.norm() > gs.EPS
        and (corner_1 - corner_0).cross(pos_proj - corner_0).dot(normal) >= 0.0
        and (corner_2 - corner_1).cross(pos_proj - corner_1).dot(normal) >= 0.0
        and (corner_0 - corner_2).cross(pos_proj - corner_2).dot(normal) >= 0.0
    )
    dist = qd.abs((pos - corner_0).dot(normal))
    if not is_inside:
        dist = gs.qd_float(1e9)
        for i_e in qd.static(range(3)):
            edge_start = corner_0 if i_e == 0 else (corner_1 if i_e == 1 else corner_2)
            edge_end = corner_1 if i_e == 0 else (corner_2 if i_e == 1 else corner_0)
            edge_param = qd.min(
                qd.max(
                    (pos - edge_start).dot(edge_end - edge_start) / qd.max((edge_end - edge_start).norm_sqr(), 1e-30),
                    0.0,
                ),
                1.0,
            )
            dist = qd.min(dist, (pos - edge_start - edge_param * (edge_end - edge_start)).norm())
    return dist


@qd.func
def func_wall_volume(
    i_lane: int,
    pos: qd.types.vector(3),
    sph_info: SPHInfo,
    sph_config: qd.template(),
    log2_n_lanes: qd.template(),
    is_needed: bool,
):
    """
    Return the kernel-weighted volume of the walls of the domain around a point, with its gradient and first moment.

    The walls stand for liquid particles at rest beyond the domain, on the lattice of the samplers continued from each
    wall. The first moment, of the gradient over the offsets to these ghost particles, turns a linear pressure field
    over them into a force. The 2**`log2_n_lanes` lanes of a subgroup tile share the nodes of the lattice, lane `i_lane`
    taking every n-th node, and every lane gets the sum. Every lane of the tile calls it, the volume being zero where
    `is_needed` is false.
    """
    # Without the ghost particles, the particles within the support radius of a wall lack neighbors, and the liquid
    # compresses until they make up for them. The support radius spans two particle sizes, so four nodes per axis on the
    # lattice of the nearest wall cover it. In a thin domain, the nodes within the support radius along an axis are
    # those beyond the lower wall on its own lattice, those inside the domain on the lattice of the nearest wall, and
    # those beyond the upper wall on its own lattice, in this order, five at most since the lattice of the far wall is
    # offset from that of the nearest one. A thin domain alone pays for these nodes, which slow down every step.
    N_NODES_AXIS = qd.static(5 if sph_config.is_domain_thin else 4)
    N_LANES = qd.static(2**log2_n_lanes)
    support_radius, kernel_norm = sph_info.support_radius[None], sph_info.kernel_norm[None]
    particle_size, particle_volume = sph_info.particle_size[None], sph_info.particle_volume[None]
    volume = gs.qd_float(0.0)
    volume_grad = qd.Vector.zero(gs.qd_float, 3)
    moment = qd.Matrix.zero(gs.qd_float, 3, 3)
    dist_lower = pos - sph_info.boundary.lower[None]
    dist_upper = sph_info.boundary.upper[None] - pos
    if is_needed and qd.min(dist_lower.min(), dist_upper.min()) < support_radius:
        dist_near = qd.min(dist_lower, dist_upper)
        near_dir = qd.select(dist_lower <= dist_upper, 1.0, -1.0)
        lattice_start = qd.floor(dist_near / particle_size - 0.5 * (N_NODES_AXIS - 1), gs.qd_int)
        n_lower = qd.Vector.zero(gs.qd_int, 3)
        n_inner = qd.Vector.zero(gs.qd_int, 3)
        n_upper = qd.Vector.zero(gs.qd_int, 3)
        if qd.static(sph_config.is_domain_thin):
            n_lower = qd.max(qd.ceil((support_radius - dist_lower) / particle_size - 0.5, gs.qd_int), 0)
            n_upper = qd.max(qd.ceil((support_radius - dist_upper) / particle_size - 0.5, gs.qd_int), 0)
            lattice_start = qd.max(qd.floor((dist_near - support_radius) / particle_size - 0.5, gs.qd_int) + 1, 0)
            lattice_end = qd.min(
                qd.ceil((dist_near + support_radius) / particle_size - 0.5, gs.qd_int),
                qd.floor((dist_lower + dist_upper) / particle_size - 0.5, gs.qd_int) + 1,
            )
            n_inner = qd.max(lattice_end - lattice_start, 0)
        for i_node_ in range((N_NODES_AXIS**3 + N_LANES - 1) // N_LANES):
            i_node = i_node_ * N_LANES + i_lane
            i_n = qd.Vector(
                [i_node // N_NODES_AXIS**2, i_node // N_NODES_AXIS % N_NODES_AXIS, i_node % N_NODES_AXIS], dt=gs.qd_int
            )
            # The offset to the node along each axis, and whether the node lies beyond a wall
            offset = near_dir * ((lattice_start + i_n + 0.5) * particle_size - dist_near)
            is_ghost = (lattice_start + i_n < 0).any()
            if qd.static(sph_config.is_domain_thin):
                is_lower = i_n < n_lower
                is_upper = i_n >= n_lower + n_inner
                offset = qd.select(
                    is_lower,
                    -dist_lower - (i_n + 0.5) * particle_size,
                    qd.select(
                        is_upper,
                        dist_upper + (i_n - n_lower - n_inner + 0.5) * particle_size,
                        near_dir * ((lattice_start + i_n - n_lower + 0.5) * particle_size - dist_near),
                    ),
                )
                is_ghost = (i_n < n_lower + n_inner + n_upper).all() and (is_lower | is_upper).any()
            if is_ghost and i_node < N_NODES_AXIS**3:
                grad = -particle_volume * func_cubic_kernel_derivative(
                    offset, support_radius, kernel_norm, sph_info.kernel_dist_min[None]
                )
                volume = volume + particle_volume * func_cubic_kernel(offset.norm(), support_radius, kernel_norm)
                volume_grad = volume_grad + grad
                moment = moment + grad.outer_product(offset)
    if qd.static(N_LANES > 1):
        volume = qd_block_sum(volume, log2_n_lanes)
        for i_d in qd.static(range(3)):
            volume_grad[i_d] = qd_block_sum(volume_grad[i_d], log2_n_lanes)
            for j_d in qd.static(range(3)):
                moment[i_d, j_d] = qd_block_sum(moment[i_d, j_d], log2_n_lanes)
    return volume, volume_grad, moment


@qd.func
def func_boundary_volume(
    i_p: int,
    i_b: int,
    i_lane: int,
    sph_state: SPHState,
    sph_info: SPHInfo,
    sph_config: qd.template(),
    log2_n_lanes: qd.template(),
    is_needed: bool,
):
    """
    Return the kernel-weighted volume of the walls and the moving boundaries around a sorted particle, with its gradient
    and first moment (see func_wall_volume and SPHBoundaryParticles).
    """
    pos = sph_state.particles_reordered.pos[i_p, i_b]
    volume, volume_grad, moment = func_wall_volume(
        i_lane, pos, sph_info, sph_config, log2_n_lanes=log2_n_lanes, is_needed=is_needed
    )
    if qd.static(sph_config.has_boundary):
        i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
        volume = volume + sph_state.boundary.volume[i_o, i_b]
        volume_grad = volume_grad + sph_state.boundary.volume_grad[i_o, i_b]
        moment = moment + sph_state.boundary.moment[i_o, i_b]
    return volume, volume_grad, moment


@qd.func
def func_for_all_neighbors(
    i_p: int,
    i_b: int,
    i_lane: int,
    ret: qd.template(),
    sph_state: SPHState,
    sph_info: SPHInfo,
    task: qd.template(),
    n_lanes: qd.template(),
):
    """
    Run a task on every sorted particle within the support radius of a sorted particle, the particle itself excluded.

    `task` takes the indices of both particles, the environment, `ret`, the state and the info, in this order, and
    accumulates its result in `ret`. With `n_lanes` above one, the lanes of a subgroup tile take the particle together,
    lane `i_lane` taking every `n_lanes`-th neighbor of each cell, and their shares of `ret` sum up after.
    """
    pos_i = sph_state.particles_reordered.pos[i_p, i_b]
    base = func_hash_wrap_cell(func_hash_pos_to_grid(pos_i, sph_state.hash_grid) - 1, sph_state.hash_grid)
    n_cells = func_hash_n_cells(3, sph_state.hash_grid)
    for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
        i_s = func_hash_cell_to_slot(base + offset, sph_state.hash_grid)
        slot_start = sph_state.hash_grid.slot_start[i_s, i_b]
        slot_size = sph_state.hash_grid.slot_size[i_s, i_b]
        for j_p_ in range((slot_size + n_lanes - 1) // n_lanes):
            j_p = slot_start + j_p_ * n_lanes + i_lane
            if j_p < slot_start + slot_size:
                offset_ij = pos_i - sph_state.particles_reordered.pos[j_p, i_b]
                if i_p != j_p and offset_ij.norm() < sph_info.support_radius[None]:
                    task(i_p, j_p, i_b, ret, sph_state, sph_info)


@qd.func
def func_copy_particle(i_p_src: int, i_p_dst: int, i_b: int, particles_src: SPHParticles, particles_dst: SPHParticles):
    """Copy the dynamic state of a particle onto a slot of the same environment, its activity aside."""
    particles_dst.pos[i_p_dst, i_b] = particles_src.pos[i_p_src, i_b]
    particles_dst.vel[i_p_dst, i_b] = particles_src.vel[i_p_src, i_b]
    particles_dst.acc[i_p_dst, i_b] = particles_src.acc[i_p_src, i_b]
    particles_dst.rho[i_p_dst, i_b] = particles_src.rho[i_p_src, i_b]
    particles_dst.p[i_p_dst, i_b] = particles_src.p[i_p_src, i_b]
    particles_dst.dfsph_factor[i_p_dst, i_b] = particles_src.dfsph_factor[i_p_src, i_b]
    particles_dst.drho[i_p_dst, i_b] = particles_src.drho[i_p_src, i_b]


@qd.kernel(fastcache=True)
def kernel_reorder_particles(sph_state: SPHState):
    """Sort the particles by the cell of the hash grid they lie in, into the copies the substep runs on."""
    func_hash_sort_particles(sph_state.particles.pos, sph_state.particles.is_active, sph_state.hash_grid)

    # The active particles take the first sorted indices (see func_hash_sort_particles)
    for i_r, i_b in qd.ndrange(
        *sph_state.particles.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.is_active))
    ):
        is_active = i_r < sph_state.hash_grid.n_particles[i_b]
        sph_state.particles_reordered.is_active[i_r, i_b] = is_active
        if is_active:
            i_p = sph_state.hash_grid.particles_idx[i_r, i_b]
            func_copy_particle(i_p, i_r, i_b, sph_state.particles, sph_state.particles_reordered)


@qd.kernel(fastcache=True)
def kernel_copy_from_reordered(sph_state: SPHState):
    """Give the active particles back the state of their sorted copy."""
    for i_r, i_b in qd.ndrange(
        *sph_state.particles.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.is_active))
    ):
        if sph_state.particles_reordered.is_active[i_r, i_b]:
            i_p = sph_state.hash_grid.particles_idx[i_r, i_b]
            func_copy_particle(i_r, i_p, i_b, sph_state.particles_reordered, sph_state.particles)


@qd.func
def func_task_compute_rho(i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo):
    offset_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    support_radius, kernel_norm = sph_info.support_radius[None], sph_info.kernel_norm[None]
    ret.density = ret.density + sph_info.particle_volume[None] * func_cubic_kernel(
        offset_ij.norm(), support_radius, kernel_norm
    )
    ret.normal = ret.normal + sph_info.particle_volume[None] * func_cubic_kernel_derivative(
        offset_ij, support_radius, kernel_norm, sph_info.kernel_dist_min[None]
    )


@qd.kernel(fastcache=True)
def kernel_compute_rho(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """Compute the density and the color gradient of the sorted particles from their neighbours and the walls."""
    # On GPU, the lanes of a subgroup tile take a particle together (see func_for_all_neighbors)
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    # The environments run fastest, as the arrays keep them innermost
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_active = sph_state.particles_reordered.is_active[i_p, i_b]
        ret = qd.Struct(density=gs.qd_float(0.0), normal=qd.Vector.zero(gs.qd_float, 3))
        if is_active:
            func_for_all_neighbors(i_p, i_b, i_lane, ret, sph_state, sph_info, func_task_compute_rho, n_lanes=N_LANES)
        if qd.static(N_LANES > 1):
            ret.density = qd_block_sum(ret.density, sph_config.log2_neighbor_lanes)
            for i_d in qd.static(range(3)):
                ret.normal[i_d] = qd_block_sum(ret.normal[i_d], sph_config.log2_neighbor_lanes)
        wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(
            i_p, i_b, i_lane, sph_state, sph_info, sph_config, sph_config.log2_neighbor_lanes, is_active
        )
        if is_active and i_lane == 0:
            # Base density using the kernel at distance 0
            support_radius, kernel_norm = sph_info.support_radius[None], sph_info.kernel_norm[None]
            rho = sph_info.particle_volume[None] * func_cubic_kernel(0.0, support_radius, kernel_norm)
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            sph_state.particles_reordered.rho[i_p, i_b] = (rho + wall_volume + ret.density) * sph_info.particles.rho[
                i_o
            ]
            sph_state.particles_normal_reordered[i_p, i_b] = wall_volume_grad + ret.normal


@qd.func
def func_task_compute_non_pressure_forces(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    offset_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    dist = offset_ij.norm()
    j_o = sph_state.hash_grid.particles_idx[j_p, i_b]
    mass_j = sph_info.particles.mass[j_o]

    # Surface tension as a continuum surface force: the curvature of the free surface is minus the divergence of its
    # unit normal, summed over the neighbors at the free surface (see surface_normal_min in SPHInfo)
    normal_i = sph_state.particles_normal_reordered[i_p, i_b]
    normal_j = sph_state.particles_normal_reordered[j_p, i_b]
    normal_min = sph_info.surface_normal_min[None]
    if normal_i.norm() > normal_min and normal_j.norm() > normal_min:
        vol_j = mass_j / sph_state.particles_reordered.rho[j_p, i_b]
        grad_ij = func_cubic_kernel_derivative(
            offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None], sph_info.kernel_dist_min[None]
        )
        curvature = -vol_j * (normal_j.normalized() - normal_i.normalized()).dot(grad_ij)
        ret.acc += ret.surface_tension_i * curvature / sph_state.particles_reordered.rho[i_p, i_b] * normal_i

    # Viscosity, normalized by twice the dimension plus two
    viscosity_coef = 2 * (3 + 2)
    vel_ij = (sph_state.particles_reordered.vel[i_p, i_b] - sph_state.particles_reordered.vel[j_p, i_b]).dot(offset_ij)
    rho_j = sph_state.particles_reordered.rho[j_p, i_b]
    ret.acc += (
        viscosity_coef
        * ret.mu_i
        * (mass_j / rho_j)
        * vel_ij
        / (dist**2 + sph_info.viscosity_eps[None])
        * func_cubic_kernel_derivative(
            offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None], sph_info.kernel_dist_min[None]
        )
    )


@qd.kernel
def kernel_compute_non_pressure_forces(
    sim_state: SimulatorState,
    sph_state: SPHState,
    sph_info: SPHInfo,
    sph_config: qd.template(),
    force_fields: qd.template(),
):
    """Compute the acceleration of the sorted particles from gravity, surface tension, viscosity and force fields."""
    # On GPU, the lanes of a subgroup tile take a particle together (see func_for_all_neighbors)
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    # The environments run fastest, as the arrays keep them innermost
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_active = sph_state.particles_reordered.is_active[i_p, i_b]
        # The parameters of the particle, read once for all its neighbors
        i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
        ret = qd.Struct(
            acc=qd.Vector.zero(gs.qd_float, 3),
            surface_tension_i=sph_info.particles.surface_tension[i_o],
            mu_i=sph_info.particles.mu[i_o],
        )
        if is_active:
            func_for_all_neighbors(
                i_p, i_b, i_lane, ret, sph_state, sph_info, func_task_compute_non_pressure_forces, n_lanes=N_LANES
            )
        if qd.static(N_LANES > 1):
            for i_d in qd.static(range(3)):
                ret.acc[i_d] = qd_block_sum(ret.acc[i_d], sph_config.log2_neighbor_lanes)
        if is_active and i_lane == 0:
            sph_state.particles_reordered.acc[i_p, i_b] = force_fields.add_acc(
                sph_info.gravity[i_b] + ret.acc,
                sph_state.particles_reordered.pos[i_p, i_b],
                sph_state.particles_reordered.vel[i_p, i_b],
                sim_state.t[i_b],
                i_p,
            )


@qd.kernel(fastcache=True)
def kernel_advect_position(sph_state: SPHState, sph_info: SPHInfo):
    """Advance the position of the sorted particles by their velocity, inside the boundary."""
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.pos))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            vel = sph_state.particles_reordered.vel[i_p, i_b]
            pos = sph_state.particles_reordered.pos[i_p, i_b] + sph_info.substep_dt[None] * vel
            pos, vel = func_impose_cube_boundary(pos, vel, sph_info.boundary)
            sph_state.particles_reordered.pos[i_p, i_b] = pos
            sph_state.particles_reordered.vel[i_p, i_b] = vel


# ------------------------------------------------------------------------------------
# ------------------------------------- DFSPH ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_task_compute_dfsph_factor(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    offset_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    grad_j = -sph_info.particle_volume[None] * func_cubic_kernel_derivative(
        offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None], sph_info.kernel_dist_min[None]
    )
    ret.grad_sqr_sum = ret.grad_sqr_sum + grad_j.norm_sqr()
    ret.grad_p = ret.grad_p - grad_j


@qd.kernel(fastcache=True)
def kernel_compute_dfsph_factor(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Compute the factor of the divergence and density solves of the sorted particles.

    These are the solves of divergence-free smoothed particle hydrodynamics (DFSPH). Every environment takes part in the
    divergence solve that follows (see func_update_convergence).
    """
    for i_b in range(sph_state.particles_reordered.pos.shape[1]):
        sph_state.envs_solve.is_solving[i_b] = True
        sph_state.envs_solve.n_active[i_b] = 0

    # On GPU, the lanes of a subgroup tile take a particle together (see func_for_all_neighbors)
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    # The environments run fastest, as the arrays keep them innermost
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_active = sph_state.particles_reordered.is_active[i_p, i_b]
        ret = qd.Struct(grad_p=qd.Vector.zero(gs.qd_float, 3), grad_sqr_sum=gs.qd_float(0.0))
        if is_active:
            func_for_all_neighbors(
                i_p, i_b, i_lane, ret, sph_state, sph_info, func_task_compute_dfsph_factor, n_lanes=N_LANES
            )
        if qd.static(N_LANES > 1):
            ret.grad_sqr_sum = qd_block_sum(ret.grad_sqr_sum, sph_config.log2_neighbor_lanes)
            for i_d in qd.static(range(3)):
                ret.grad_p[i_d] = qd_block_sum(ret.grad_p[i_d], sph_config.log2_neighbor_lanes)
        # The walls, at rest, change the density of the particle by its own velocity, which they change with its own
        # stiffness and with the stiffness they take from it among the liquid around it (see func_solver_iteration),
        # weighted by its share of the kernel sum of the liquid.
        wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(
            i_p, i_b, i_lane, sph_state, sph_info, sph_config, sph_config.log2_neighbor_lanes, is_active
        )
        if is_active and i_lane == 0:
            qd.atomic_add(sph_state.envs_solve.n_active[i_b], 1)
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            rho_rel = sph_state.particles_reordered.rho[i_p, i_b] / sph_info.particles.rho[i_o]
            kernel_self = func_cubic_kernel(0.0, sph_info.support_radius[None], sph_info.kernel_norm[None])
            # The relative density of the liquid alone includes the particle itself, so it never vanishes
            self_weight = sph_info.particle_volume[None] * kernel_self / (rho_rel - wall_volume)
            grad_p = ret.grad_p + wall_volume_grad
            sum_grad_p_k = ret.grad_sqr_sum + grad_p.dot(ret.grad_p + (1.0 + self_weight) * wall_volume_grad)

            factor = gs.qd_float(0.0)
            if sum_grad_p_k > sph_info.factor_eps[None]:
                factor = -1.0 / sum_grad_p_k
            else:
                # A particle without neighbors carries no pressure into the next density solve
                sph_state.particles_reordered.p[i_p, i_b] = 0.0
            sph_state.particles_reordered.dfsph_factor[i_p, i_b] = factor


@qd.func
def func_task_compute_density_rate(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    vel_ij = sph_state.particles_reordered.vel[i_p, i_b] - sph_state.particles_reordered.vel[j_p, i_b]
    offset_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    ret.drho += sph_info.particle_volume[None] * vel_ij.dot(
        func_cubic_kernel_derivative(
            offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None], sph_info.kernel_dist_min[None]
        )
    )
    ret.n_neighbors += 1


@qd.func
def func_compute_density_rate(
    sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template(), solve_density: qd.template()
):
    """
    Compute what a solve of divergence-free smoothed particle hydrodynamics (DFSPH) cancels for the sorted particles.

    The divergence solve cancels the positive rate of change of the density, and the density solve the predicted
    relative density at the end of the substep. Each particle also gets the stiffness that the next Jacobi iteration
    applies to cancel it, and adds its error to that of its environment, summed for func_update_convergence.
    """
    # On GPU, the lanes of a subgroup tile take a particle together (see func_for_all_neighbors)
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    # The environments run fastest, as the arrays keep them innermost
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_solved = sph_state.envs_solve.is_solving[i_b] and sph_state.particles_reordered.is_active[i_p, i_b]
        ret = qd.Struct(drho=gs.qd_float(0.0), n_neighbors=0)
        if is_solved:
            func_for_all_neighbors(
                i_p, i_b, i_lane, ret, sph_state, sph_info, func_task_compute_density_rate, n_lanes=N_LANES
            )
        if qd.static(N_LANES > 1):
            ret.drho = qd_block_sum(ret.drho, sph_config.log2_neighbor_lanes)
            ret.n_neighbors = qd_block_sum(ret.n_neighbors, sph_config.log2_neighbor_lanes)
        wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(
            i_p, i_b, i_lane, sph_state, sph_info, sph_config, sph_config.log2_neighbor_lanes, is_solved
        )
        if is_solved and i_lane == 0:
            # The walls are at rest, while the moving boundaries carry their own velocity
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            ret.drho = ret.drho + sph_state.particles_reordered.vel[i_p, i_b].dot(wall_volume_grad)
            if qd.static(sph_config.has_boundary):
                ret.drho = ret.drho - sph_state.boundary.vel_flux[i_o, i_b]
            rho0 = sph_info.particles.rho[i_o]
            err = gs.qd_float(0.0)
            if qd.static(solve_density):
                drho = sph_state.particles_reordered.rho[i_p, i_b] / rho0 + sph_info.substep_dt[None] * ret.drho
                sph_state.particles_reordered.drho[i_p, i_b] = drho

                # The stiffness accumulated by the density solve is carried over from one substep to the next, so the
                # iterations update it in both directions: they take back the part the liquid no longer needs, and
                # clamp it so that the pressure never pulls the liquid together. The update is relaxed (see
                # jacobi_relaxation in SPHInfo).
                kappa = -sph_state.particles_reordered.p[i_p, i_b] / rho0
                k_i = (
                    sph_info.jacobi_relaxation[None]
                    * (drho - 1.0)
                    * sph_state.particles_reordered.dfsph_factor[i_p, i_b]
                )
                sph_state.particles_k_reordered[i_p, i_b] = qd.min(kappa + k_i, 0.0) - kappa

                # A particle under pressure counts its expansion as well, which the iterations correct by taking
                # pressure back, while a particle free of pressure only counts its compression, the liquid being free
                # to expand.
                err_rel = drho - 1.0
                if kappa >= 0.0:
                    err_rel = qd.max(err_rel, 0.0)
                err = rho0 * qd.abs(err_rel)
            else:
                # Only a positive divergence is corrected, and none where the neighbors of the particle are too few, the
                # walls standing for neighbors
                drho = qd.max(ret.drho, 0.0)
                if ret.n_neighbors < sph_info.n_neighbors_min[None] and wall_volume <= 0.0:
                    drho = 0.0
                sph_state.particles_reordered.drho[i_p, i_b] = drho
                sph_state.particles_k_reordered[i_p, i_b] = drho * sph_state.particles_reordered.dfsph_factor[i_p, i_b]
                err = rho0 * drho
            if qd.static(sph_config.is_deterministic):
                func_exact_sum_add(0, i_b, err, sph_state.envs_solve.err)
            else:
                qd.atomic_add(sph_state.envs_solve.err_sum[i_b], err)


@qd.kernel(fastcache=True)
def kernel_predict_velocity(sph_state: SPHState, sph_info: SPHInfo):
    """
    Advance the velocity of the sorted particles by their non-pressure acceleration, ahead of the density solve.

    Every environment takes part in the density solve that follows (see func_update_convergence). The warm start of
    that solve applies part of the pressure carried over from the previous substep to every particle (see
    warm_start_scale in SPHInfo). The density of a particle at rest deep in the liquid sits on either side of its rest
    value, so warm-starting the compressed particles alone would drop the pressure of about half of them to zero, against
    neighbors keeping it, and the pressure gradient between them would shake the liquid.
    """
    for i_b in range(sph_state.particles_reordered.pos.shape[1]):
        sph_state.envs_solve.is_solving[i_b] = True

    dt = sph_info.substep_dt[None]
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.vel))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            sph_state.particles_reordered.vel[i_p, i_b] = (
                sph_state.particles_reordered.vel[i_p, i_b] + dt * sph_state.particles_reordered.acc[i_p, i_b]
            )
            # The density solve corrects a density error over the substep, which turns the DFSPH factor of the
            # divergence solve, a velocity correction per density rate, into an acceleration per density error.
            sph_state.particles_reordered.dfsph_factor[i_p, i_b] = sph_state.particles_reordered.dfsph_factor[
                i_p, i_b
            ] * (1.0 / dt**2)

            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            rho0 = sph_info.particles.rho[i_o]
            p = sph_info.warm_start_scale[None] * sph_state.particles_reordered.p[i_p, i_b]
            sph_state.particles_reordered.p[i_p, i_b] = p
            sph_state.particles_k_reordered[i_p, i_b] = -p / rho0


@qd.func
def func_task_solver_iteration(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    # The stiffness already holds the inverse of the density, which is the same for every particle of a liquid
    k_j = sph_state.particles_k_reordered[j_p, i_b]
    k_sum = ret.k_i + k_j
    offset_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    if qd.abs(k_sum) > sph_info.stiffness_eps[None]:
        grad_p_j = -sph_info.particle_volume[None] * func_cubic_kernel_derivative(
            offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None], sph_info.kernel_dist_min[None]
        )
        ret.dv = ret.dv - ret.dv_scale * k_sum * grad_p_j

    # The stiffness and the position of the neighbors, weighted for the stiffness of the walls (see
    # func_solver_iteration)
    kernel_ij = func_cubic_kernel(offset_ij.norm(), sph_info.support_radius[None], sph_info.kernel_norm[None])
    ret.weight_sum = ret.weight_sum + kernel_ij
    ret.k_sum_w = ret.k_sum_w + kernel_ij * k_j
    ret.pos_sum = ret.pos_sum + kernel_ij * sph_state.particles_reordered.pos[j_p, i_b]


@qd.func
def func_solver_iteration(
    sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template(), solve_density: qd.template()
):
    """
    Run one Jacobi iteration of the divergence or the density solve of divergence-free smoothed particle hydrodynamics
    (DFSPH) on the velocity of the sorted particles.

    The velocity of every particle changes by the stiffness of its own and of its neighbors. The divergence solve
    corrects velocities directly, while the density solve accelerates over the substep. The first pass of the density
    solve is its warm start, and the divergence solve checks its convergence before its first iteration, so its first
    pass leaves the velocities untouched.
    """
    is_first_pass = sph_state.solve_loop.n_passes[None] == 0
    warm_start = False
    is_iterating = True
    if qd.static(solve_density):
        warm_start = is_first_pass
    else:
        is_iterating = not is_first_pass
    # On GPU, the lanes of a subgroup tile take a particle together (see func_for_all_neighbors)
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    # The environments run fastest, as the arrays keep them innermost
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_solved = (
            is_iterating and sph_state.envs_solve.is_solving[i_b] and sph_state.particles_reordered.is_active[i_p, i_b]
        )
        dv_scale = gs.qd_float(1.0)
        if qd.static(solve_density):
            dv_scale = sph_info.substep_dt[None]
        k_i = sph_state.particles_k_reordered[i_p, i_b]
        ret = qd.Struct(
            dv=qd.Vector.zero(gs.qd_float, 3),
            k_i=k_i,
            dv_scale=dv_scale,
            weight_sum=gs.qd_float(0.0),
            k_sum_w=gs.qd_float(0.0),
            pos_sum=qd.Vector.zero(gs.qd_float, 3),
        )
        if is_solved:
            func_for_all_neighbors(
                i_p, i_b, i_lane, ret, sph_state, sph_info, func_task_solver_iteration, n_lanes=N_LANES
            )
        if qd.static(N_LANES > 1):
            ret.weight_sum = qd_block_sum(ret.weight_sum, sph_config.log2_neighbor_lanes)
            ret.k_sum_w = qd_block_sum(ret.k_sum_w, sph_config.log2_neighbor_lanes)
            for i_d in qd.static(range(3)):
                ret.dv[i_d] = qd_block_sum(ret.dv[i_d], sph_config.log2_neighbor_lanes)
                ret.pos_sum[i_d] = qd_block_sum(ret.pos_sum[i_d], sph_config.log2_neighbor_lanes)
        wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(
            i_p, i_b, i_lane, sph_state, sph_info, sph_config, sph_config.log2_neighbor_lanes, is_solved
        )
        if is_solved and i_lane == 0:
            # The particle itself weighs into the stiffness and the position its neighbors give the walls
            pos = sph_state.particles_reordered.pos[i_p, i_b]
            kernel_self = func_cubic_kernel(0.0, sph_info.support_radius[None], sph_info.kernel_norm[None])
            ret.weight_sum = ret.weight_sum + kernel_self
            ret.k_sum_w = ret.k_sum_w + kernel_self * k_i
            ret.pos_sum = ret.pos_sum + kernel_self * pos

            # The ghost particles of the walls take the stiffness of the liquid around the particle (see
            # func_wall_volume). The warm start applies the pressure itself, which adds the hydrostatic
            # pressure of the body force between them, while the iterations correct it.
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            gravity = sph_info.gravity[i_b]
            k_wall_center = ret.k_sum_w / ret.weight_sum
            wall_dv = qd.Vector.zero(gs.qd_float, 3)
            if warm_start:
                k_wall_center = k_wall_center - gravity.dot(pos - ret.pos_sum / ret.weight_sum)
                wall_dv = -wall_moment @ gravity
            wall_dv = wall_dv + (k_i + k_wall_center) * wall_volume_grad
            if qd.static(sph_config.has_boundary):
                # The moving boundaries take the opposite of the impulse they give the particle
                impulse_scale = dv_scale * sph_info.particles.mass[i_o]
                impulse_coef = sph_state.boundary.impulse_coef[i_o, i_b] - impulse_scale * (k_i + k_wall_center)
                sph_state.boundary.impulse_coef[i_o, i_b] = impulse_coef
                if warm_start:
                    sph_state.boundary.impulse_gravity_coef[i_o, i_b] = impulse_scale
            sph_state.particles_reordered.vel[i_p, i_b] = (
                sph_state.particles_reordered.vel[i_p, i_b] + ret.dv + dv_scale * wall_dv
            )

            if qd.static(solve_density):
                # The stiffness k_i is the pressure this pass applies to cancel the density error, up to the -1/rho0
                # scaling carried by dfsph_factor, so summing -k_i * rho0 over the density solve yields the implicit
                # pressure field that enforces incompressibility. The warm start applies the pressure as it stands (see
                # kernel_predict_velocity).
                if not warm_start:
                    rho0 = sph_info.particles.rho[i_o]
                    sph_state.particles_reordered.p[i_p, i_b] = sph_state.particles_reordered.p[i_p, i_b] - k_i * rho0


@qd.func
def func_update_convergence(
    n_iters_max: int, sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template(), solve_density: qd.template()
):
    """
    Stop the solve of divergence-free smoothed particle hydrodynamics (DFSPH) of the environments that converged.

    Each environment stops once the average density error of its own particles meets the tolerance, so that its result
    is the same whatever the other environments simulate alongside it, and the density solve only after its warm start.
    The error summed by the density rate pass is reset once read, for the next pass. The loop runs on as long as an
    environment iterates on, for `n_iters_max` iterations at most after its first pass.
    """
    n_passes = sph_state.solve_loop.n_passes[None]
    can_stop = True
    if qd.static(solve_density):
        can_stop = n_passes > 0
    sph_state.solve_loop.is_looping[()] = 0
    for i_b in range(sph_state.envs_solve.is_solving.shape[0]):
        if sph_state.envs_solve.is_solving[i_b]:
            err_tol = sph_info.divergence_err_tol[None]
            if qd.static(solve_density):
                err_tol = sph_info.density_err_tol[None]
            err = gs.qd_float(0.0)
            if qd.static(sph_config.is_deterministic):
                err = func_exact_sum_take(0, i_b, sph_state.envs_solve.err)
            else:
                err = sph_state.envs_solve.err_sum[i_b]
                sph_state.envs_solve.err_sum[i_b] = 0.0
            if can_stop and err <= err_tol * sph_state.envs_solve.n_active[i_b]:
                sph_state.envs_solve.is_solving[i_b] = False
            elif n_passes < n_iters_max:
                sph_state.solve_loop.is_looping[()] = 1
    sph_state.solve_loop.n_passes[None] = n_passes + 1


@qd.kernel(graph=True, fastcache=True)
def kernel_solve_incompressibility(
    n_iters_max: int,
    is_looping: qd.types.ndarray(qd.i32, ndim=0),
    sph_state: SPHState,
    sph_info: SPHInfo,
    sph_config: qd.template(),
    solve_density: qd.template(),
):
    """
    Run the divergence or the density solve of divergence-free smoothed particle hydrodynamics (DFSPH) on the device.

    The warm start of the density solve applies the pressure carried over from the previous substep, which overshoots
    wherever that pressure corrected a density error at once. The average density error leaves such particles unnoticed,
    so the density solve iterates at least once to take their pressure back. Otherwise, the pressure keeps pushing them
    apart substep after substep. The divergence solve checks its convergence before its first iteration, which a
    divergence-free velocity field leaves without iterations.
    """
    sph_state.solve_loop.n_passes[None] = 0
    is_looping[()] = 1
    # FIXME: quadrants#1083 - the statements before a graph do_while run again on every pass when the loop body opens
    # with bare statements, so the loop body opens with a for loop.
    while qd.graph.do_while(is_looping):
        func_solver_iteration(sph_state, sph_info, sph_config, solve_density)
        func_compute_density_rate(sph_state, sph_info, sph_config, solve_density)
        func_update_convergence(n_iters_max, sph_state, sph_info, sph_config, solve_density)


# ------------------------------------------------------------------------------------
# --------------------------------------- io -----------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_add_particles(
    particle_start: int,
    is_active: bool,
    rho: float,
    mu: float,
    surface_tension: float,
    pos: qd.types.ndarray(),
    sph_state: SPHState,
    sph_info: SPHInfo,
):
    """Add the particles of an entity at rest, with the parameters of its material."""
    for i_p_, i_b in qd.ndrange(
        pos.shape[0], sph_state.particles.pos.shape[1], axes=qd.static(eoa(sph_state.particles.is_active))
    ):
        i_p = i_p_ + particle_start
        sph_state.particles.is_active[i_p, i_b] = is_active
        sph_state.particles.pos[i_p, i_b] = qd.Vector([pos[i_p_, 0], pos[i_p_, 1], pos[i_p_, 2]])
        sph_state.particles.vel[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        sph_state.particles.p[i_p, i_b] = 0.0

    for i_p_ in range(pos.shape[0]):
        i_p = i_p_ + particle_start
        sph_info.particles.rho[i_p] = rho
        sph_info.particles.mu[i_p] = mu
        sph_info.particles.surface_tension[i_p] = surface_tension
        sph_info.particles.mass[i_p] = sph_info.particle_volume[None] * rho


@qd.kernel(fastcache=True)
def kernel_clear_particles_pressure(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), sph_state: SPHState
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(sph_state.particles.p))):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_values' in base_solver.py
        if i_p >= 0:
            sph_state.particles.p[i_p, i_b] = 0.0
