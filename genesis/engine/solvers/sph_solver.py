import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary, func_impose_cube_boundary
from genesis.engine.entities import SPHEntity
from genesis.engine.entities.sph_entity import SPHEntityDescription
from genesis.engine.materials import SPH
from genesis.engine.state_change import StateChange
from genesis.options.solvers import SPHOptions
from genesis.options.surfaces import Surface
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
    qd_block_scan,
    qd_block_shuffle,
    qd_block_sum,
)

from .base_solver import SimulatorState
from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


class SPHParticles(Data, kind=DataKind.STATE, shape=(PARTICLES, ENVS)):
    """
    Runtime state of the smoothed particle hydrodynamics (SPH) particles, one value per particle and environment.

    `p` is the pressure, which the pressure solve carries over from one substep to the next as its first guess (see
    kernel_solve_pressure).
    """

    pos: qd.Tensor = field((float, 3), change=StateChange.GEOMETRY)
    vel: qd.Tensor = field((float, 3), change=StateChange.DYNAMICS)
    # The substep computes them at the positions it starts from, which it moves afterward
    acc: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH)
    rho: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    p: qd.Tensor = field(float)
    is_active: qd.Tensor = field(bool, change=StateChange.GEOMETRY)


class SPHParticlesInfo(Data, kind=DataKind.CONSTANT, shape=(PARTICLES,)):
    """
    Model parameters of the smoothed particle hydrodynamics (SPH) particles, one value per particle.

    `rho` is the rest density, `mu` the viscosity and `surface_tension` the surface tension coefficient, in N/m.
    """

    rho: qd.Tensor = field(float)
    mass: qd.Tensor = field(float)
    mu: qd.Tensor = field(float)
    surface_tension: qd.Tensor = field(float)


class SPHPressureSolve(Data, kind=DataKind.SCRATCH, shape=(PARTICLES, ENVS)):
    """The vectors of the pressure solve of the sorted particles (see kernel_solve_pressure)."""

    # The inverse of the diagonal of the operator, which preconditions the solve
    diag_inv: qd.Tensor = field(float)
    # The relative density excess that the particle would reach at the end of the substep without pressure
    rhs: qd.Tensor = field(float)
    # The gradient of the objective, the relative density deficit of the particle at the end of the substep under the
    # current pressures
    grad: qd.Tensor = field(float)
    # The search direction, its product with the operator, and the velocity change of the particle per unit step along
    # it, or along the pressures while the environment recomputes its gradient
    dir: qd.Tensor = field(float)
    dir_prod: qd.Tensor = field(float)
    dir_vel: qd.Tensor = field((float, 3))
    # The same for the contacts of the particle with the surfaces it meets, whose pressure holds it off them, starting
    # from zero as they rarely hold (see func_contacts)
    p_contacts: qd.Tensor = field((float, 3))
    rhs_contacts: qd.Tensor = field((float, 3))
    grad_contacts: qd.Tensor = field((float, 3))
    dir_contacts: qd.Tensor = field((float, 3))
    dir_prod_contacts: qd.Tensor = field((float, 3))
    # The share of the diagonal of the operator that the bodies giving way to the liquid add, which a coupler opts in
    # (see SPHBoundaryParticles)
    body_compliance: qd.Tensor = field(float, opt_in=True)


class SPHEnvsSolve(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The pressure solve of each environment (see kernel_solve_pressure).

    The sums over the particles are the same whatever the order of the terms, so that the solve iterates as many times
    and takes the same steps from one run to the next.
    """

    # One slot per sum: the preconditioned norms of the chopped gradient and of the reduced free gradient, the product
    # of the preconditioned free gradient with the operator times the search direction, then the dot products of the
    # search direction with its product with the operator and with the gradient. The slots that a stage adds or takes
    # are contiguous, so that it does so in a loop around a single inlined call.
    sums: ExactSums
    is_solving: qd.Tensor = field(bool)
    # Whether the environment recomputes the gradient of its objective from the pressures, after an expansion step or
    # at the start of the solve, takes a proportioning step along the chopped gradient, or expands the active set
    is_recomputing: qd.Tensor = field(bool)
    is_proportioning: qd.Tensor = field(bool)
    is_expanding: qd.Tensor = field(bool)
    # The longest step along the search direction that keeps every pressure non-negative, the step taken, the dot
    # product of the direction with its product with the operator, and the coefficient of the previous direction in the
    # next one
    step_max: qd.Tensor = field(float)
    step: qd.Tensor = field(float)
    dir_prod_norm: qd.Tensor = field(float)
    dir_coef: qd.Tensor = field(float)
    # The largest relative density error of the particles, the largest entry of the projected gradient of the
    # objective, which a maximum keeps the same whatever the order of the particles
    err_max: qd.Tensor = field(float)


class SPHSolveLoop(Data, kind=DataKind.SCRATCH, shape=()):
    """
    The passes of the pressure solve, which runs on the device as long as an environment iterates on (see
    kernel_solve_pressure).

    The flag is an ndarray on every backend, since graph.do_while takes the same ndarray on every call.
    """

    n_passes: qd.Tensor = field(int)
    is_looping: qd.Tensor = field(qd.i32, ndarray=True)


class SPHBoundaryParticles(Data, kind=DataKind.SCRATCH, shape=(PARTICLES, 3, ENVS)):
    """
    The surfaces that each SPH particle meets over the current substep, three at most, in the order of the particles.

    A coupler fills them with the walls of the domain and the surfaces of the moving boundaries nearest to the particle
    (see func_contacts and func_for_all_interactions), and reads back the impulse the particle gives each of them.
    """

    # The plane tangent to the surface at its point closest to the particle, `normal . x = offset`, the unit normal
    # pointing out of the solid
    surfaces_normal: qd.Tensor = field((float, 3))
    surfaces_offset: qd.Tensor = field(float)
    # The velocity of the surface at that point
    surfaces_vel: qd.Tensor = field((float, 3))
    # The surface: -1 for none, below -1 for a wall of the domain, the moving boundary otherwise, whose surfaces the
    # liquid on the same side of it is mirrored across, and the segment of the boundary closest to the particle, -1 for
    # those taking no impulse
    surfaces_key: qd.Tensor = field(int)
    surfaces_segment: qd.Tensor = field(int)
    # The inverse mass of the body of the surface, which gives way to the liquid within the pressure solve, 0 for the
    # walls, the fixed bodies and those whose motion the solve takes as given
    surfaces_inv_mass: qd.Tensor = field(float)
    # The impulse the particle gives the surface over the substep, along its normal
    surfaces_impulse: qd.Tensor = field(float)


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
    pressure_solve: SPHPressureSolve
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
    # The squared distance that regularizes the viscosity of a pair of particles
    viscosity_eps: qd.Tensor = field(float, shape=())
    # The magnitude of the color gradient that regularizes its unit normal in the surface tension
    surface_normal_min: qd.Tensor = field(float, shape=())
    # The largest relative density error of a particle that ends the pressure solve
    density_err_tol: qd.Tensor = field(float, shape=())
    # The share of the density excess of a particle at the start of the substep that the substep corrects
    drift_relaxation: qd.Tensor = field(float, shape=())
    # The fixed step of the projected steepest descent of an expansion step, below twice the inverse of the largest
    # eigenvalue of the preconditioned operator
    expansion_step: qd.Tensor = field(float, shape=())
    # The ratio of the norm of the chopped gradient to that of the reduced free gradient above which the next step is a
    # proportioning step
    proportioning_ratio: qd.Tensor = field(float, shape=())
    # The diagonal of the operator below which a particle takes no pressure
    diag_min: qd.Tensor = field(float, shape=())
    # The distance below which a particle meets a surface (see func_contacts)
    contact_dist: qd.Tensor = field(float, shape=())
    # The inverse of the bulk modulus of the liquid, the relative density change per unit pressure, which keeps the
    # pressure solve feasible where the liquid fills a closed volume
    compliance: qd.Tensor = field(float, shape=())


class SPHConfig(Config):
    """The parts of the smoothed particle hydrodynamics (SPH) solver that its kernels compile in."""

    # Whether moving boundaries push on the particles (see SPHBoundaryParticles)
    has_boundary: bool
    # The base-2 logarithm of the lanes of a GPU subgroup tile that take a particle and its neighbors together, one thread
    # taking it alone on CPU and once the particles of every environment outnumber the GPU cores, which one thread per
    # particle already saturates (see func_for_all_interactions)
    log2_neighbor_lanes: int


class SPHSolver(ParticleSolver):
    entity_spec_cls: ClassVar[type[gs.entities.Spec]] = gs.entities.SPH

    # Its whole state lives in the arrays 'data' yields (see 'CheckpointMixin')
    is_checkpointable = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, sim: "Simulator", options: SPHOptions) -> None:
        super().__init__(sim, options)

        self._support_radius = options._support_radius

        self._density_err_tol = options.max_density_error
        if self._density_err_tol is None:
            self._density_err_tol = 1e-5 if gs.qd_float == qd.f32 else 1e-6
        self._n_pressure_iters_max = options.max_density_solver_iterations

        # The samplers place one particle per cube of the particle size, the spacing over which the kernel sums to one
        self._particle_volume = self._particle_size**3

        self._hash_grid_cell_size = options.hash_grid_cell_size
        self._hash_grid_res = options._hash_grid_res

        # Coefficient for computing stable timestep
        self._stable_dt_coef = 1.0

        self.sph_state: SPHState | None = None
        self.sph_info: SPHInfo | None = None
        self.sph_config: SPHConfig | None = None

    @gs.assert_unbuilt
    def build(self, n_envs: int) -> None:
        super().build(n_envs)

        sizes = dict(envs=self.n_envs_, particles=self._n_particles)
        hash_grid_values = dict(cell_size=self._hash_grid_cell_size, grid_res=self._hash_grid_res)
        self.sph_state = SPHState(
            sizes=dict(
                sizes,
                # Each sum of an environment takes one slot, its terms spanning 96 bits (see SPHEnvsSolve)
                exact_sum_slots=5,
                exact_sum_bins=3,
                **hash_grid_sizes(math.prod(self._hash_grid_res)),
            ),
            values=dict(hash_grid=hash_grid_values),
            # The units of the bins of the exact sums and the loop of the solves span no environment
            env_first={
                None: False,
                (SPHState.envs_solve.sums.bins_unit, SPHState.solve_loop): None,
            },
            unused=not self.is_active,
        )
        self.register_errors(self.sph_state.errno, {})
        if self.is_active:
            # The lowest bin of the sums resolves 2^-20 of their terms at the tolerance, the bins 2^32 apart (see
            # ExactSums). The dot products weigh squared density errors by the inverse diagonal of the operator, which
            # the kernel gradients of the lattice of the samplers set to about 2.5 * rho0 * (particle_size / dt)^2.
            density0_min = min(entity.material.rho for entity in self._entities)
            diag_inv_min = 2.5 * density0_min * (self._particle_size / self.substep_dt) ** 2
            exp_prod = math.floor(math.log2(self._density_err_tol**2 * diag_inv_min))
            exp_min = exp_prod - 20
            bins_unit = 2.0 ** np.array([exp_min, exp_min + 32.0, exp_min + 64.0], dtype=gs.np_float)
            self.sph_state.envs_solve.sums.bins_unit.from_numpy(bins_unit)
        self.sph_config = SPHConfig(
            has_boundary=self.is_active and is_opted_in(SPHState.boundary),
            # A block of the subgroup helpers takes each particle when the particles leave cores idle, its lanes sharing
            # the particles met (see func_for_all_interactions)
            log2_neighbor_lanes=BLOCK_LOG2_LANES if 0 < self._n_particles * self.n_envs_ <= get_gpu_core_count() else 0,
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
                viscosity_eps=viscosity_eps,
                # The color gradient of the particles deeper than the first two layers below the free surface is noise,
                # whose direction would add surface energy in the bulk (see func_task_compute_non_pressure_forces)
                surface_normal_min=0.2 / self._support_radius,
                density_err_tol=self._density_err_tol,
                # Correcting the whole density excess at once kicks the liquid by the velocity that removes it within a
                # substep, which the pressure solve turns into kinetic energy
                drift_relaxation=0.2,
                # Twice the inverse of an estimate of the largest eigenvalue of the preconditioned operator: the ratio
                # of the row sum of the coupling terms of a particle in the bulk to its own diagonal
                expansion_step=1.9 / 4.0,
                proportioning_ratio=1.0,
                diag_min=1e-20,
                # The kernel gradient between a particle and its own image peaks at a third of the support radius
                contact_dist=self._support_radius / 6.0,
                # Water
                compliance=1.0 / 2.2e9,
            )
        self.sph_info = SPHInfo(sizes=sizes, values=info_values, unused=not self.is_active)
        self._particles_pos = self.sph_state.particles.pos
        self._particles_vel = self.sph_state.particles.vel
        self._particles_is_active = self.sph_state.particles.is_active
        self._particles_mass = self.sph_info.particles.mass

        if self.is_active:
            for entity in self.entities:
                entity._add_to_solver()

            # Compute stable timestep for the given particle properties (mu) and support radius and print warning
            # if the current timestep is larger than the stable timestep. This formula is derived from
            # 'func_task_compute_non_pressure_forces', which computes the acceleration due to non-pressure forces. The
            # timestep is compared through its product with the kinematic viscosity, which is zero for an inviscid
            # liquid.
            nu_max = max([entity.material.mu / entity.material.rho for entity in self._entities])
            if self.substep_dt * nu_max > self._stable_dt_coef * viscosity_eps:
                stable_dt = self._stable_dt_coef * viscosity_eps / nu_max
                gs.logger.warning(
                    f"""Current timestep ({self.substep_dt:.4g}) is larger than the stable timestep ({stable_dt:.4g}) for SPH
solver. Please consider reducing the timestep, liquid viscosity (mu), or increasing the particle size."""
                )

        self._build_gravity(self.sph_info.gravity)

        self._is_built = True
        self.changelog.register_arrays(self.data)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(
        self,
        idx: int,
        surface: Surface | None,
        spec: "gs.entities.SPH | None",
        desc: SPHEntityDescription | None = None,
    ) -> SPHEntity:
        """Create an entity from its description, resolved from the entity spec and its surface when none is given."""
        if desc is None:
            desc = SPHEntityDescription.resolve(
                spec, surface, self._particle_size, self.boundary_lower, self.boundary_upper
            )
        entity = SPHEntity(
            scene=self._sim.scene,
            solver=self,
            desc=desc,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
        )
        self._entities.append(entity)
        self._n_particles += entity.n_particles
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
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self):
        if not self.is_active:
            return
        kernel_reorder_particles(self.sph_state)
        kernel_compute_rho_and_diag(self.sph_state, self.sph_info, self.sph_config)
        kernel_compute_non_pressure_forces(
            self._sim.sim_state, self.sph_state, self.sph_info, self.sph_config, self._force_fields
        )
        kernel_predict_velocity(self.sph_state, self.sph_info)
        kernel_compute_rhs(self.sph_state, self.sph_info, self.sph_config)
        kernel_solve_pressure(
            self._n_pressure_iters_max,
            self.sph_state.solve_loop.is_looping,
            self.sph_state,
            self.sph_info,
            self.sph_config,
        )
        kernel_apply_pressure(self.sph_state, self.sph_info, self.sph_config)

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
        # The pressure that the pressure solve carries over belongs to the neighborhood a particle leaves or joins
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
def func_kernel(dist: float, support_radius: float, kernel_norm: float):
    """
    Return the cubic spline smoothing kernel at a distance.

    Parameters
    ----------
    dist : float
        The distance from the center of the kernel.
    support_radius : float
        The support radius of the kernel, beyond which it vanishes.
    kernel_norm : float
        The normalization of the kernel.
    """
    dist_rel = qd.min(dist / support_radius, 1.0)
    value = 2.0 * kernel_norm * (1.0 - dist_rel) ** 3
    if dist_rel <= 0.5:
        value = kernel_norm * (6.0 * dist_rel**3 - 6.0 * dist_rel**2 + 1.0)
    return value


@qd.func
def func_kernel_gradient(offset: qd.types.vector(3), support_radius: float, kernel_norm: float):
    """
    Return the gradient of the cubic spline smoothing kernel at a vector (see func_kernel).

    Parameters
    ----------
    offset : qd.types.vector(3)
        The vector from the neighbor to the point.
    support_radius : float
        The support radius of the kernel.
    kernel_norm : float
        The normalization of the kernel.
    """
    dist = offset.norm()
    dist_rel = qd.min(dist / support_radius, 1.0)
    # The derivative over the distance, finite at 0 as the kernel is smooth there
    deriv_over_dist = 6.0 * kernel_norm * (3.0 * dist_rel - 2.0) / support_radius**2
    if dist_rel > 0.5:
        deriv_over_dist = -6.0 * kernel_norm * (1.0 - dist_rel) ** 2 / (dist * support_radius)
    return deriv_over_dist * offset


@qd.func
def func_kernel_hessian(offset: qd.types.vector(3), support_radius: float, kernel_norm: float):
    """
    Return the Hessian of the cubic spline smoothing kernel at a vector (see func_kernel).

    Parameters
    ----------
    offset : qd.types.vector(3)
        The vector from the neighbor to the point.
    support_radius : float
        The support radius of the kernel.
    kernel_norm : float
        The normalization of the kernel.
    """
    dist = offset.norm()
    dist_rel = dist / support_radius
    # The first derivative over the distance and the second derivative along the distance, both finite at 0
    deriv_over_dist = gs.qd_float(0.0)
    deriv_second = gs.qd_float(0.0)
    if dist_rel <= 0.5:
        deriv_over_dist = kernel_norm * (18.0 * dist_rel - 12.0) / support_radius**2
        deriv_second = kernel_norm * (36.0 * dist_rel - 12.0) / support_radius**2
    elif dist_rel <= 1.0:
        deriv_over_dist = -6.0 * kernel_norm * (1.0 - dist_rel) ** 2 / (dist_rel * support_radius**2)
        deriv_second = 12.0 * kernel_norm * (1.0 - dist_rel) / support_radius**2
    direction = offset / qd.max(dist, 1e-30)
    return deriv_over_dist * qd.Matrix.identity(gs.qd_float, 3) + (
        deriv_second - deriv_over_dist
    ) * direction.outer_product(direction)


@qd.func
def func_segment_closest(pos: qd.types.vector(3), corners: qd.types.matrix(3, 3), eps: float):
    """
    Return the point of a triangle closest to a point.

    Parameters
    ----------
    pos : qd.types.vector(3)
        The point.
    corners : qd.types.matrix(3, 3)
        The corners of the triangle, one per row.
    eps : float
        The area below which the triangle has no inside, its closest point lying on its edges.
    """
    corner_0 = qd.Vector([corners[0, 0], corners[0, 1], corners[0, 2]])
    corner_1 = qd.Vector([corners[1, 0], corners[1, 1], corners[1, 2]])
    corner_2 = qd.Vector([corners[2, 0], corners[2, 1], corners[2, 2]])
    area_normal = (corner_1 - corner_0).cross(corner_2 - corner_0)
    normal = area_normal / qd.max(area_normal.norm(), eps)
    closest = pos - (pos - corner_0).dot(normal) * normal
    # The projection lies inside the triangle where it is on the inner side of its three edges, a degenerate triangle
    # having no inside
    is_inside = (
        area_normal.norm() > eps
        and (corner_1 - corner_0).cross(closest - corner_0).dot(normal) >= 0.0
        and (corner_2 - corner_1).cross(closest - corner_1).dot(normal) >= 0.0
        and (corner_0 - corner_2).cross(closest - corner_2).dot(normal) >= 0.0
    )
    if not is_inside:
        dist = gs.qd_float(1e30)
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
            edge_point = edge_start + edge_param * (edge_end - edge_start)
            if (pos - edge_point).norm() < dist:
                dist = (pos - edge_point).norm()
                closest = edge_point
    return closest


@qd.func
def func_for_all_interactions(
    i_p: int,
    i_b: int,
    i_lane: int,
    ret: qd.template(),
    sph_state: SPHState,
    sph_info: SPHInfo,
    sph_config: qd.template(),
    task: qd.template(),
    n_lanes: qd.template(),
):
    """
    Run a task on every sorted particle that interacts with a sorted particle, as itself or as its mirror image.

    The particle interacts with its neighbors within the support radius, except across a moving boundary, and with the
    mirror images of the particles across the walls of the domain and across the surfaces of the moving boundaries it
    meets, which make them free-slip: an image stands at the reflection of its particle, with its reflected velocity,
    its pressure and its reflected normal. The particle meets its own images too.

    Parameters
    ----------
    i_p : int
        The sorted particle.
    i_b : int
        The environment.
    i_lane : int
        The lane of the subgroup tile that takes the particle, among `n_lanes`. With `n_lanes` above one, the lanes of
        the tile take the particle together, sharing the particles it meets, and their shares of `ret` sum up after.
    ret : qd.template()
        The result `task` accumulates, which holds the position of the particle `ret.pos_i` and the image transform:
        before each call of `task`, `ret.image_rot` and `ret.image_shift` take the second particle to its image at
        `R x + t`, `ret.image_vel` gives the image of a velocity `v` as `R v + u`, and `ret.i_surface` is the slot of
        the surface of a moving boundary the image is reflected across, -1 for a neighbor itself or an image across the
        walls.
    sph_state : SPHState
        The state of the solver.
    sph_info : SPHInfo
        The parameters of the solver.
    sph_config : qd.template()
        The parts of the solver its kernels compile in.
    task : qd.template()
        The task, which takes the indices of both particles, the environment, `ret`, the state and the info, in this
        order, and accumulates its result in `ret`.
    n_lanes : qd.template()
        The number of lanes of the subgroup tile that take the particle together, 1 or 32.
    """
    # The queries are the particle itself and its reflections across the walls it is near, enumerated in mixed radix
    # by the choice along each axis (stay, or reflect across each wall it is near), then its reflections across the
    # surfaces of the moving boundaries it meets, enumerated by the set bits of a mask.
    pos_i = sph_state.particles_reordered.pos[i_p, i_b]
    ret.pos_i = pos_i
    support_radius = sph_info.support_radius[None]
    lower, upper = sph_info.boundary.lower[None], sph_info.boundary.upper[None]
    is_near_lower = pos_i - lower < support_radius
    is_near_upper = upper - pos_i < support_radius
    i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
    n_reflections = 1 + qd.cast(is_near_lower, gs.qd_int) + qd.cast(is_near_upper, gs.qd_int)
    n_reflections_xy = n_reflections[0] * n_reflections[1]
    n_wall_queries = n_reflections_xy * n_reflections[2]
    n_queries = n_wall_queries
    surfaces_mask = 0
    if qd.static(sph_config.has_boundary):
        for i_s in range(3):
            if sph_state.boundary.surfaces_key[i_o, i_s, i_b] >= 0:
                normal = sph_state.boundary.surfaces_normal[i_o, i_s, i_b]
                if normal.dot(pos_i) - sph_state.boundary.surfaces_offset[i_o, i_s, i_b] < support_radius:
                    surfaces_mask = surfaces_mask | (1 << i_s)
                    n_queries = n_queries + 1

    # Each lane looks up the particles of one cell around one query, as many pairs of a query and a cell at a time as
    # lanes, and the lanes number the particles of these cells as one run with a prefix sum over the subgroup, a
    # particle finding its pair by bisection over it. The lanes thus share the particles met by all the queries evenly,
    # whereas striding the particles of each cell would leave idle the lanes beyond the few particles a cell holds.
    n_cells = func_hash_n_cells(3, sph_state.hash_grid)
    n_cells_yz = n_cells[1] * n_cells[2]
    n_cells_all = n_cells[0] * n_cells_yz
    cell_size = sph_state.hash_grid.cell_size[None]
    n_pairs = n_queries * n_cells_all
    for i_chunk_ in range((n_pairs + n_lanes - 1) // n_lanes):
        i_pair = i_chunk_ * n_lanes + i_lane
        i_query = 0
        slot_start = 0
        slot_size = 0
        if i_pair < n_pairs:
            i_k = i_pair // n_cells_all
            i_cell = i_pair % n_cells_all
            pos_query = pos_i
            if i_k < n_wall_queries:
                # The choice along each axis: 0 to stay, 1 to reflect across the lower wall if near it and across the
                # upper wall otherwise, 2 to reflect across the upper wall. The query is numbered by the base-3 digits
                # of its reflections: 0 for none, 1 across the lower wall, 2 across the upper one.
                choices = qd.Vector(
                    [i_k % n_reflections[0], i_k % n_reflections_xy // n_reflections[0], i_k // n_reflections_xy]
                )
                is_lower = (choices == 1) & is_near_lower
                is_upper = (choices == 2) | ((choices == 1) & ~is_near_lower)
                pos_query = qd.select(is_lower, 2.0 * lower - pos_i, qd.select(is_upper, 2.0 * upper - pos_i, pos_i))
                digits = qd.select(is_lower, 1, qd.select(is_upper, 2, 0))
                i_query = digits[0] + 3 * digits[1] + 9 * digits[2]
            else:
                if qd.static(sph_config.has_boundary):
                    # The surface whose rank among the set bits of the mask is that of the query
                    n_skipped = i_k - n_wall_queries
                    i_s_query = -1
                    for i_s in range(3):
                        if i_s_query < 0 and ((surfaces_mask >> i_s) & 1) == 1:
                            if n_skipped == 0:
                                i_s_query = i_s
                            n_skipped = n_skipped - 1
                    normal = sph_state.boundary.surfaces_normal[i_o, i_s_query, i_b]
                    dist = normal.dot(pos_i) - sph_state.boundary.surfaces_offset[i_o, i_s_query, i_b]
                    pos_query = pos_i - 2.0 * dist * normal
                    i_query = 27 + i_s_query
            # The cells beyond the support radius of the query are skipped. Along an axis whose run of cells wraps into
            # fewer than three slots, every cell counts as touching the query.
            cell_query = func_hash_pos_to_grid(pos_query, sph_state.hash_grid)
            offset = qd.Vector([i_cell // n_cells_yz, i_cell // n_cells[2] % n_cells[1], i_cell % n_cells[2]])
            dist_lower = pos_query - cell_query * cell_size
            dist_upper = cell_size - dist_lower
            dist_cell = qd.select(
                n_cells < 3, 0.0, qd.select(offset == 0, dist_lower, qd.select(offset == 2, dist_upper, 0.0))
            )
            if dist_cell.norm_sqr() < support_radius**2:
                cell = func_hash_wrap_cell(cell_query - 1, sph_state.hash_grid) + offset
                i_c = func_hash_cell_to_slot(cell, sph_state.hash_grid)
                slot_start = sph_state.hash_grid.slot_start[i_c, i_b]
                slot_size = sph_state.hash_grid.slot_size[i_c, i_b]
        slots_end = slot_size
        n_run = slot_size
        if qd.static(n_lanes > 1):
            slots_end, n_run = qd_block_scan(slot_size)
        for i_run_ in range((n_run + n_lanes - 1) // n_lanes):
            i_run = i_run_ * n_lanes + i_lane
            j_p = slot_start + i_run
            i_query_run = i_query
            if qd.static(n_lanes > 1):
                i_pair_lane = 0
                for i_step in qd.static(range(5)):
                    if qd_block_shuffle(slots_end, i_pair_lane + (16 >> i_step) - 1) <= i_run:
                        i_pair_lane = i_pair_lane + (16 >> i_step)
                pair_slots_end = qd_block_shuffle(slots_end, i_pair_lane)
                pair_slot_size = qd_block_shuffle(slot_size, i_pair_lane)
                j_p = qd_block_shuffle(slot_start, i_pair_lane) + i_run - (pair_slots_end - pair_slot_size)
                i_query_run = qd_block_shuffle(i_query, i_pair_lane)
            if i_run < n_run:
                ret.i_surface = -1
                ret.image_rot = qd.Matrix.identity(gs.qd_float, 3)
                ret.image_shift = qd.Vector.zero(gs.qd_float, 3)
                ret.image_vel = qd.Vector.zero(gs.qd_float, 3)
                if i_query_run < 27:
                    digits = qd.Vector([i_query_run % 3, i_query_run // 3 % 3, i_query_run // 9])
                    is_lower, is_upper = digits == 1, digits == 2
                    sign = qd.select(is_lower | is_upper, -1.0, 1.0)
                    ret.image_rot = qd.Matrix([[sign[0], 0.0, 0.0], [0.0, sign[1], 0.0], [0.0, 0.0, sign[2]]])
                    ret.image_shift = qd.select(is_lower, 2.0 * lower, qd.select(is_upper, 2.0 * upper, 0.0))
                else:
                    ret.i_surface = i_query_run - 27
                if func_is_interacting(i_p, j_p, i_b, i_query_run, ret, sph_state, sph_info, sph_config):
                    task(i_p, j_p, i_b, ret, sph_state, sph_info)
    ret.i_surface = -1
    ret.image_rot = qd.Matrix.identity(gs.qd_float, 3)
    ret.image_shift = qd.Vector.zero(gs.qd_float, 3)
    ret.image_vel = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_is_interacting(
    i_p: int,
    j_p: int,
    i_b: int,
    i_query: int,
    ret: qd.template(),
    sph_state: SPHState,
    sph_info: SPHInfo,
    sph_config: qd.template(),
):
    """
    Return whether a sorted particle interacts with a particle met by a query of func_for_all_interactions.

    The particle interacts with its neighbors and their images across the walls within the support radius. A neighbor
    is cut when each particle lies behind the tangent plane of the other on the same moving boundary, a boundary
    thinner than the support radius then separating the liquid on both sides. Across the surface of a moving boundary,
    the particle meets the images of the particles on the same side of the same boundary alone, reflected across the
    plane through the mean of the closest points of both particles, normal to the mean of their normals, with the
    velocity of the boundary along that normal.

    Parameters
    ----------
    i_p : int
        The sorted particle.
    j_p : int
        The sorted particle met by the query.
    i_b : int
        The environment.
    i_query : int
        The query: 0 for the particle itself, below 27 for its reflection across the walls, through the transform
        `ret.image_rot` and `ret.image_shift`, and the surface `ret.i_surface` of a moving boundary otherwise, whose
        image transform the particle met sets in `ret`.
    ret : qd.template()
        The result of the task, which holds the position of the particle `ret.pos_i` and the image transform.
    sph_state : SPHState
        The state of the solver.
    sph_info : SPHInfo
        The parameters of the solver.
    sph_config : qd.template()
        The parts of the solver its kernels compile in.
    """
    support_radius = sph_info.support_radius[None]
    pos_j = sph_state.particles_reordered.pos[j_p, i_b]
    i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
    j_o = sph_state.hash_grid.particles_idx[j_p, i_b]
    is_interacting = False
    if i_query < 27:
        pos_image = ret.image_rot @ pos_j + ret.image_shift
        is_interacting = (i_query > 0 or i_p != j_p) and (ret.pos_i - pos_image).norm() < support_radius
        if qd.static(sph_config.has_boundary):
            if i_query == 0 and is_interacting:
                for i_s in range(3):
                    key = sph_state.boundary.surfaces_key[i_o, i_s, i_b]
                    if key >= 0:
                        normal_i = sph_state.boundary.surfaces_normal[i_o, i_s, i_b]
                        if normal_i.dot(pos_j) < sph_state.boundary.surfaces_offset[i_o, i_s, i_b]:
                            for j_s in range(3):
                                if sph_state.boundary.surfaces_key[j_o, j_s, i_b] == key:
                                    normal_j = sph_state.boundary.surfaces_normal[j_o, j_s, i_b]
                                    if normal_j.dot(ret.pos_i) < sph_state.boundary.surfaces_offset[j_o, j_s, i_b]:
                                        is_interacting = False
    else:
        if qd.static(sph_config.has_boundary):
            i_s = ret.i_surface
            key = sph_state.boundary.surfaces_key[i_o, i_s, i_b]
            normal_i = sph_state.boundary.surfaces_normal[i_o, i_s, i_b]
            for j_s in range(3):
                normal_j = sph_state.boundary.surfaces_normal[j_o, j_s, i_b]
                if (
                    not is_interacting
                    and sph_state.boundary.surfaces_key[j_o, j_s, i_b] == key
                    and normal_i.dot(normal_j) > 0.5
                ):
                    normal = (normal_i + normal_j).normalized()
                    closest_i = (
                        ret.pos_i
                        - (normal_i.dot(ret.pos_i) - sph_state.boundary.surfaces_offset[i_o, i_s, i_b]) * normal_i
                    )
                    closest_j = (
                        pos_j - (normal_j.dot(pos_j) - sph_state.boundary.surfaces_offset[j_o, j_s, i_b]) * normal_j
                    )
                    offset = 0.5 * normal.dot(closest_i + closest_j)
                    reflection = qd.Matrix.identity(gs.qd_float, 3) - 2.0 * normal.outer_product(normal)
                    vel_normal = 0.5 * normal.dot(
                        sph_state.boundary.surfaces_vel[i_o, i_s, i_b] + sph_state.boundary.surfaces_vel[j_o, j_s, i_b]
                    )
                    pos_image = reflection @ pos_j + 2.0 * offset * normal
                    if (ret.pos_i - pos_image).norm() < support_radius:
                        is_interacting = True
                        ret.image_rot = reflection
                        ret.image_shift = 2.0 * offset * normal
                        ret.image_vel = 2.0 * vel_normal * normal
    return is_interacting


@qd.func
def func_copy_particle(i_p_src: int, i_p_dst: int, i_b: int, particles_src: SPHParticles, particles_dst: SPHParticles):
    """Copy the dynamic state of a particle onto a slot of the same environment, its activity aside."""
    particles_dst.pos[i_p_dst, i_b] = particles_src.pos[i_p_src, i_b]
    particles_dst.vel[i_p_dst, i_b] = particles_src.vel[i_p_src, i_b]
    particles_dst.acc[i_p_dst, i_b] = particles_src.acc[i_p_src, i_b]
    particles_dst.rho[i_p_dst, i_b] = particles_src.rho[i_p_src, i_b]
    particles_dst.p[i_p_dst, i_b] = particles_src.p[i_p_src, i_b]


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
def func_task_compute_non_pressure_forces(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    # The images of the particles take part like the particles themselves (see func_for_all_interactions)
    offset_ij = ret.pos_i - (ret.image_rot @ sph_state.particles_reordered.pos[j_p, i_b] + ret.image_shift)
    dist = offset_ij.norm()
    j_o = sph_state.hash_grid.particles_idx[j_p, i_b]
    mass_j = sph_info.particles.mass[j_o]

    # Surface tension as the gradient of the surface energy, the surface tension times the area of the free surface,
    # which the magnitude of the color gradient integrates over the volume of the particles. The pair forces are
    # opposite, and the energy does not change under rotations, so the surface tension conserves the linear and the
    # angular momentum. The unit normal is regularized where the color gradient vanishes (see surface_normal_min in
    # SPHInfo), which spares the bulk of the liquid. The image of a neighbor takes its reflected normal.
    normal_min_sqr = sph_info.surface_normal_min[None] ** 2
    normal_i = sph_state.particles_normal_reordered[i_p, i_b]
    normal_j = ret.image_rot @ sph_state.particles_normal_reordered[j_p, i_b]
    unit_normal_i = normal_i / qd.sqrt(normal_i.norm_sqr() + normal_min_sqr)
    unit_normal_j = normal_j / qd.sqrt(normal_j.norm_sqr() + normal_min_sqr)
    surface_tension_ij = 0.5 * (ret.surface_tension_i + sph_info.particles.surface_tension[j_o])
    hessian_ij = func_kernel_hessian(offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None])
    ret.acc += (
        -surface_tension_ij
        * (mass_j / sph_state.particles_reordered.rho[j_p, i_b])
        / ret.rho_i
        * (hessian_ij @ (unit_normal_i - unit_normal_j))
    )

    # Viscosity as the Laplacian of the velocity of Morris et al. (1997), which is the same along every direction of
    # the lattice the samplers place the particles on, where the form projected on the offset of the pair depends on
    # the direction of the flow. The pair takes the mean of the dynamic viscosities and of the densities of both
    # particles, so that the forces between them cancel and the viscosity conserves linear momentum. The kinematic
    # viscosity of the pair is its dynamic viscosity over its density.
    vel_ij = ret.vel_i - (ret.image_rot @ sph_state.particles_reordered.vel[j_p, i_b] + ret.image_vel)
    mu_ij = 0.5 * (ret.mu_i + sph_info.particles.mu[j_o])
    rho_ij = 0.5 * (ret.rho_i + sph_state.particles_reordered.rho[j_p, i_b])
    grad_ij = func_kernel_gradient(offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None])
    ret.acc += (
        2.0
        * (mu_ij / rho_ij)
        * (mass_j / rho_ij)
        * offset_ij.dot(grad_ij)
        / (dist**2 + sph_info.viscosity_eps[None])
        * vel_ij
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
    # On GPU, the lanes of a subgroup tile take a particle together (see func_for_all_interactions)
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
            rho_i=sph_state.particles_reordered.rho[i_p, i_b],
            vel_i=sph_state.particles_reordered.vel[i_p, i_b],
            pos_i=qd.Vector.zero(gs.qd_float, 3),
            image_rot=qd.Matrix.identity(gs.qd_float, 3),
            image_shift=qd.Vector.zero(gs.qd_float, 3),
            image_vel=qd.Vector.zero(gs.qd_float, 3),
            i_surface=-1,
        )
        if is_active:
            # The images share the shear of the liquid, which makes the walls and the moving boundaries free-slip
            func_for_all_interactions(
                i_p,
                i_b,
                i_lane,
                ret,
                sph_state,
                sph_info,
                sph_config,
                func_task_compute_non_pressure_forces,
                n_lanes=N_LANES,
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
# ---------------------------------- pressure solve ----------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_task_compute_rho(i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo):
    offset_ij = ret.pos_i - (ret.image_rot @ sph_state.particles_reordered.pos[j_p, i_b] + ret.image_shift)
    support_radius, particle_volume = sph_info.support_radius[None], sph_info.particle_volume[None]
    grad_ij = particle_volume * func_kernel_gradient(offset_ij, support_radius, sph_info.kernel_norm[None])
    kernel_ij = func_kernel(offset_ij.norm(), support_radius, sph_info.kernel_norm[None])
    ret.density = ret.density + particle_volume * kernel_ij
    ret.normal = ret.normal + grad_ij
    # The body of a surface moves the images across it by its own velocity (see func_is_interacting)
    for i_s in qd.static(range(3)):
        if ret.i_surface == i_s:
            reflected_grad = grad_ij - ret.image_rot @ grad_ij
            for i_d in qd.static(range(3)):
                ret.surfaces_grad[i_s, i_d] = ret.surfaces_grad[i_s, i_d] + reflected_grad[i_d]
    # The image of the particle itself moves with it, by its reflected velocity (see func_for_all_interactions)
    if j_p == i_p:
        ret.grad_self_image = ret.grad_self_image + ret.image_rot @ grad_ij
    else:
        j_o = sph_state.hash_grid.particles_idx[j_p, i_b]
        ret.diag_neighbors = ret.diag_neighbors + grad_ij.norm_sqr() / sph_info.particles.rho[j_o]


@qd.kernel(fastcache=True)
def kernel_compute_rho_and_diag(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Compute the density, the color gradient and the diagonal of the pressure solve of the sorted particles.

    The pressure solve takes the inverse of the diagonal of its operator as preconditioner (see kernel_solve_pressure).
    """
    # On GPU, the lanes of a subgroup tile take a particle together (see func_for_all_interactions)
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    # The environments run fastest, as the arrays keep them innermost
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_active = sph_state.particles_reordered.is_active[i_p, i_b]
        ret = qd.Struct(
            density=gs.qd_float(0.0),
            normal=qd.Vector.zero(gs.qd_float, 3),
            diag_neighbors=gs.qd_float(0.0),
            grad_self_image=qd.Vector.zero(gs.qd_float, 3),
            surfaces_grad=qd.Matrix.zero(gs.qd_float, 3, 3),
            pos_i=qd.Vector.zero(gs.qd_float, 3),
            image_rot=qd.Matrix.identity(gs.qd_float, 3),
            image_shift=qd.Vector.zero(gs.qd_float, 3),
            image_vel=qd.Vector.zero(gs.qd_float, 3),
            i_surface=-1,
        )
        if is_active:
            func_for_all_interactions(
                i_p, i_b, i_lane, ret, sph_state, sph_info, sph_config, func_task_compute_rho, n_lanes=N_LANES
            )
        if qd.static(N_LANES > 1):
            ret.density = qd_block_sum(ret.density, sph_config.log2_neighbor_lanes)
            ret.diag_neighbors = qd_block_sum(ret.diag_neighbors, sph_config.log2_neighbor_lanes)
            for i_d in qd.static(range(3)):
                ret.normal[i_d] = qd_block_sum(ret.normal[i_d], sph_config.log2_neighbor_lanes)
                ret.grad_self_image[i_d] = qd_block_sum(ret.grad_self_image[i_d], sph_config.log2_neighbor_lanes)
                for i_s in qd.static(range(3)):
                    ret.surfaces_grad[i_s, i_d] = qd_block_sum(
                        ret.surfaces_grad[i_s, i_d], sph_config.log2_neighbor_lanes
                    )
        if is_active and i_lane == 0:
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            # The particle itself weighs into its density by the kernel at distance 0
            kernel_self = func_kernel(0.0, sph_info.support_radius[None], sph_info.kernel_norm[None])
            rho0 = sph_info.particles.rho[i_o]
            rho = (sph_info.particle_volume[None] * kernel_self + ret.density) * rho0
            sph_state.particles_reordered.rho[i_p, i_b] = rho
            # The own images of the particle move by its reflected velocity
            sph_state.particles_normal_reordered[i_p, i_b] = ret.normal
            grad_self = ret.normal - ret.grad_self_image
            diag = (
                sph_info.substep_dt[None] ** 2 * (grad_self.norm_sqr() / rho0 + ret.diag_neighbors)
                + sph_info.compliance[None]
            )
            if qd.static(sph_config.has_boundary):
                # Each body takes the reaction of the images across its surfaces, which moves it by its inverse mass.
                # The operator keeps the share of the particle alone, the bodies coupling the particles around them.
                body_compliance = gs.qd_float(0.0)
                for i_s in qd.static(range(3)):
                    surface_grad = qd.Vector(
                        [ret.surfaces_grad[i_s, 0], ret.surfaces_grad[i_s, 1], ret.surfaces_grad[i_s, 2]]
                    )
                    inv_mass = sph_state.boundary.surfaces_inv_mass[i_o, i_s, i_b]
                    body_compliance = body_compliance + inv_mass * surface_grad.norm_sqr()
                body_compliance = (
                    body_compliance
                    * sph_info.substep_dt[None] ** 2
                    * sph_info.particles.mass[i_o]
                    / sph_info.particles.rho[i_o]
                )
                sph_state.pressure_solve.body_compliance[i_p, i_b] = body_compliance
                diag = diag + body_compliance
            diag_inv = gs.qd_float(0.0)
            if diag > sph_info.diag_min[None]:
                diag_inv = 1.0 / diag
            else:
                # A particle without neighbors takes no pressure
                sph_state.particles_reordered.p[i_p, i_b] = 0.0
            sph_state.pressure_solve.diag_inv[i_p, i_b] = diag_inv


@qd.kernel(fastcache=True)
def kernel_predict_velocity(sph_state: SPHState, sph_info: SPHInfo):
    """Advance the velocity of the sorted particles by their non-pressure acceleration, ahead of the pressure solve."""
    dt = sph_info.substep_dt[None]
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.vel))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            sph_state.particles_reordered.vel[i_p, i_b] = (
                sph_state.particles_reordered.vel[i_p, i_b] + dt * sph_state.particles_reordered.acc[i_p, i_b]
            )


@qd.func
def func_task_compute_density_rate(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    offset_ij = ret.pos_i - (ret.image_rot @ sph_state.particles_reordered.pos[j_p, i_b] + ret.image_shift)
    grad_ij = sph_info.particle_volume[None] * func_kernel_gradient(
        offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None]
    )
    vel_ij = ret.vel_i - (ret.image_rot @ sph_state.particles_reordered.vel[j_p, i_b] + ret.image_vel)
    ret.rate = ret.rate + vel_ij.dot(grad_ij)


@qd.func
def func_task_compute_dir_rate(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    offset_ij = ret.pos_i - (ret.image_rot @ sph_state.particles_reordered.pos[j_p, i_b] + ret.image_shift)
    grad_ij = sph_info.particle_volume[None] * func_kernel_gradient(
        offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None]
    )
    vel_ij = ret.vel_i - ret.image_rot @ sph_state.pressure_solve.dir_vel[j_p, i_b]
    ret.rate = ret.rate + vel_ij.dot(grad_ij)


@qd.func
def func_task_compute_pressure_vel(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    offset_ij = ret.pos_i - (ret.image_rot @ sph_state.particles_reordered.pos[j_p, i_b] + ret.image_shift)
    grad_ij = sph_info.particle_volume[None] * func_kernel_gradient(
        offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None]
    )
    pressure_j = sph_state.pressure_solve.dir[j_p, i_b]
    if ret.is_recomputing:
        pressure_j = sph_state.particles_reordered.p[j_p, i_b]
    ret.force = ret.force + (ret.pressure_i + pressure_j) * grad_ij


@qd.func
def func_contacts(i_p: int, i_b: int, sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Return the surfaces that hold a sorted particle off them, with how deep the particle lies within their reach.

    The particle meets three surfaces at most within one particle size of it, which push it along their normal before
    it comes closer than 'contact_dist': the wall nearest along each axis, or the surfaces a coupler fills (see
    SPHBoundaryParticles).

    Parameters
    ----------
    i_p : int
        The sorted particle.
    i_b : int
        The environment.
    sph_state : SPHState
        The state of the solver.
    sph_info : SPHInfo
        The parameters of the solver.
    sph_config : qd.template()
        The parts of the solver its kernels compile in.

    Returns
    -------
    normals : qd.types.matrix(3, 3)
        The unit normals of the surfaces, one per row, out of the solids.
    depths : qd.types.vector(3)
        The depth of the particle within 'contact_dist' of each surface, in particle sizes, negative beyond it.
    normal_vels : qd.types.vector(3)
        The velocity of each surface along its normal.
    contacts_weight : qd.types.vector(3)
        1 for each surface the particle meets, 0 otherwise.
    """
    particle_size = sph_info.particle_size[None]
    pos = sph_state.particles_reordered.pos[i_p, i_b]
    normals = qd.Matrix.zero(gs.qd_float, 3, 3)
    dists = qd.Vector([particle_size, particle_size, particle_size])
    normal_vels = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(sph_config.has_boundary):
        i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
        for i_s in qd.static(range(3)):
            if sph_state.boundary.surfaces_key[i_o, i_s, i_b] != -1:
                normal = sph_state.boundary.surfaces_normal[i_o, i_s, i_b]
                for i_d in qd.static(range(3)):
                    normals[i_s, i_d] = normal[i_d]
                dists[i_s] = normal.dot(pos) - sph_state.boundary.surfaces_offset[i_o, i_s, i_b]
                normal_vels[i_s] = normal.dot(sph_state.boundary.surfaces_vel[i_o, i_s, i_b])
    else:
        dist_lower = pos - sph_info.boundary.lower[None]
        dist_upper = sph_info.boundary.upper[None] - pos
        dists = qd.min(dist_lower, dist_upper)
        sign = qd.select(dist_lower <= dist_upper, 1.0, -1.0)
        normals = qd.Matrix([[sign[0], 0.0, 0.0], [0.0, sign[1], 0.0], [0.0, 0.0, sign[2]]])
    contacts_weight = qd.select(dists < particle_size, 1.0, 0.0)
    return normals, (sph_info.contact_dist[None] - dists) / particle_size, normal_vels, contacts_weight


@qd.func
def func_contacts_diag(i_p: int, i_b: int, sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Return the diagonal of the operator of the pressure solve of each contact of a sorted particle (see func_contacts).

    It is the inverse mass of the particle along the normal, plus that of the body of the surface, which gives way to
    the particle (see SPHBoundaryParticles).

    Parameters
    ----------
    i_p : int
        The sorted particle.
    i_b : int
        The environment.
    sph_state : SPHState
        The state of the solver.
    sph_info : SPHInfo
        The parameters of the solver.
    sph_config : qd.template()
        The parts of the solver its kernels compile in.
    """
    i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
    scale = (sph_info.substep_dt[None] / sph_info.particle_size[None]) ** 2
    diags = qd.Vector([1.0, 1.0, 1.0]) * (scale / sph_info.particles.rho[i_o])
    if qd.static(sph_config.has_boundary):
        volume = sph_info.particles.mass[i_o] / sph_info.particles.rho[i_o]
        for i_s in qd.static(range(3)):
            diags[i_s] = diags[i_s] + scale * volume * sph_state.boundary.surfaces_inv_mass[i_o, i_s, i_b]
    return diags


@qd.func
def func_compute_density_rate(
    i_p: int,
    i_b: int,
    i_lane: int,
    is_needed: bool,
    vel_field: qd.template(),
    sph_state: SPHState,
    sph_info: SPHInfo,
    sph_config: qd.template(),
    task: qd.template(),
):
    """
    Return the rate of change of the relative density of a sorted particle under a field of velocities of the particles.

    The images of the particles move by their reflected velocities, plus the velocity of their surface along its normal
    where `task` adds it (see func_for_all_interactions). Every lane of the tile calls it and gets the result.

    Parameters
    ----------
    i_p : int
        The sorted particle.
    i_b : int
        The environment.
    i_lane : int
        The lane of the subgroup tile that takes the particle.
    is_needed : bool
        Whether the particle takes part, the others getting zero.
    vel_field : qd.template()
        The velocities of the sorted particles.
    sph_state : SPHState
        The state of the solver.
    sph_info : SPHInfo
        The parameters of the solver.
    sph_config : qd.template()
        The parts of the solver its kernels compile in.
    task : qd.template()
        The task reading `vel_field`, func_task_compute_density_rate or func_task_compute_dir_rate.
    """
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    vel_i = qd.Vector.zero(gs.qd_float, 3)
    if is_needed:
        vel_i = vel_field[i_p, i_b]
    ret = qd.Struct(
        rate=gs.qd_float(0.0),
        vel_i=vel_i,
        pos_i=qd.Vector.zero(gs.qd_float, 3),
        image_rot=qd.Matrix.identity(gs.qd_float, 3),
        image_shift=qd.Vector.zero(gs.qd_float, 3),
        image_vel=qd.Vector.zero(gs.qd_float, 3),
        i_surface=-1,
    )
    if is_needed:
        func_for_all_interactions(i_p, i_b, i_lane, ret, sph_state, sph_info, sph_config, task, n_lanes=N_LANES)
    if qd.static(N_LANES > 1):
        ret.rate = qd_block_sum(ret.rate, sph_config.log2_neighbor_lanes)
    return ret.rate


@qd.kernel(fastcache=True)
def kernel_compute_rhs(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Compute the right-hand side of the pressure solve of the sorted particles, and start the solve of every environment.

    It is the relative density excess that each particle would reach at the end of the substep without pressure. The
    excess the particle already has at the start of the substep is corrected over several substeps, at the rate of
    'drift_relaxation', which spares the liquid the kick of correcting it at once, while a particle below its rest
    density may reach it within the substep. The solve starts from the pressure of
    the previous substep, so that it computes the gradient of its objective first (see kernel_solve_pressure).
    """
    for i_b in range(sph_state.particles_reordered.pos.shape[1]):
        sph_state.envs_solve.is_solving[i_b] = True
        sph_state.envs_solve.is_recomputing[i_b] = True
        sph_state.envs_solve.is_proportioning[i_b] = False
        sph_state.envs_solve.is_expanding[i_b] = False
        sph_state.envs_solve.step_max[i_b] = gs.qd_float(float("inf"))
        sph_state.envs_solve.err_max[i_b] = 0.0

    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_active = sph_state.particles_reordered.is_active[i_p, i_b]
        rate = func_compute_density_rate(
            i_p,
            i_b,
            i_lane,
            is_active,
            sph_state.particles_reordered.vel,
            sph_state,
            sph_info,
            sph_config,
            func_task_compute_density_rate,
        )
        if is_active and i_lane == 0:
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            density_excess = sph_state.particles_reordered.rho[i_p, i_b] / sph_info.particles.rho[i_o] - 1.0
            drift_relaxation, dt = sph_info.drift_relaxation[None], sph_info.substep_dt[None]
            # Only the excess beyond the bound is relaxed, the room left below it being usable within the substep
            density_err = qd.select(density_excess > 0.0, drift_relaxation * density_excess, density_excess)
            sph_state.pressure_solve.rhs[i_p, i_b] = density_err + dt * rate
            # The surfaces the particle does not meet take no pressure (see func_contacts)
            normals, depths, normal_vels, contacts_weight = func_contacts(i_p, i_b, sph_state, sph_info, sph_config)
            approach_vels = normals @ sph_state.particles_reordered.vel[i_p, i_b] - normal_vels
            rhs_contacts = contacts_weight * (
                qd.select(depths > 0.0, drift_relaxation * depths, depths)
                - dt / sph_info.particle_size[None] * approach_vels
            )
            sph_state.pressure_solve.rhs_contacts[i_p, i_b] = rhs_contacts
            sph_state.pressure_solve.p_contacts[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_pressure_product(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Multiply the operator of the pressure solve with the pressures or the search direction of the sorted particles.

    The operator is the rate at which the pressures change the relative density of the particles over the substep, which
    the velocity change per unit step along the search direction, computed by the previous pass, gives. An environment
    that recomputes the gradient of its objective multiplies the pressures, and takes the rest of its steps along the
    search direction, which also yields the dot products and the longest feasible step of its next step.
    """
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_solved = sph_state.envs_solve.is_solving[i_b] and sph_state.particles_reordered.is_active[i_p, i_b]
        rate = func_compute_density_rate(
            i_p,
            i_b,
            i_lane,
            is_solved,
            sph_state.pressure_solve.dir_vel,
            sph_state,
            sph_info,
            sph_config,
            func_task_compute_dir_rate,
        )
        if is_solved and i_lane == 0:
            dt = sph_info.substep_dt[None]
            # The liquid gives way under pressure by its compliance
            pressure_src = sph_state.particles_reordered.p[i_p, i_b]
            if not sph_state.envs_solve.is_recomputing[i_b]:
                pressure_src = sph_state.pressure_solve.dir[i_p, i_b]
            prod = sph_info.compliance[None] * pressure_src - dt * rate
            normals, depths, normal_vels, contacts_weight = func_contacts(i_p, i_b, sph_state, sph_info, sph_config)
            prod_contacts = contacts_weight * (
                dt / sph_info.particle_size[None] * (normals @ sph_state.pressure_solve.dir_vel[i_p, i_b])
            )
            if qd.static(sph_config.has_boundary):
                # The bodies giving way to the particle add to the diagonal alone (see 'body_compliance')
                pressures_src = sph_state.pressure_solve.p_contacts[i_p, i_b]
                if not sph_state.envs_solve.is_recomputing[i_b]:
                    pressures_src = sph_state.pressure_solve.dir_contacts[i_p, i_b]
                i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
                particle_diags = (dt / sph_info.particle_size[None]) ** 2 / sph_info.particles.rho[i_o]
                body_diags = func_contacts_diag(i_p, i_b, sph_state, sph_info, sph_config) - particle_diags
                prod = prod + sph_state.pressure_solve.body_compliance[i_p, i_b] * pressure_src
                prod_contacts = prod_contacts + contacts_weight * body_diags * pressures_src
            if sph_state.envs_solve.is_recomputing[i_b]:
                sph_state.pressure_solve.grad[i_p, i_b] = prod - sph_state.pressure_solve.rhs[i_p, i_b]
                grad_contacts = prod_contacts - sph_state.pressure_solve.rhs_contacts[i_p, i_b]
                sph_state.pressure_solve.grad_contacts[i_p, i_b] = grad_contacts
            else:
                direction = sph_state.pressure_solve.dir[i_p, i_b]
                dir_contacts = sph_state.pressure_solve.dir_contacts[i_p, i_b]
                sph_state.pressure_solve.dir_prod[i_p, i_b] = prod
                sph_state.pressure_solve.dir_prod_contacts[i_p, i_b] = prod_contacts
                grad_contacts = sph_state.pressure_solve.grad_contacts[i_p, i_b]
                terms = qd.Vector(
                    [
                        direction * prod + dir_contacts.dot(prod_contacts),
                        direction * sph_state.pressure_solve.grad[i_p, i_b] + dir_contacts.dot(grad_contacts),
                    ]
                )
                # The slots of the dot products of the search direction (see SPHEnvsSolve)
                for i_s in range(2):
                    func_exact_sum_add(3 + i_s, i_b, terms[i_s], sph_state.envs_solve.sums)
                # The longest step that keeps every pressure non-negative
                if direction > 0.0:
                    qd.atomic_min(
                        sph_state.envs_solve.step_max[i_b], sph_state.particles_reordered.p[i_p, i_b] / direction
                    )
                p_contacts = sph_state.pressure_solve.p_contacts[i_p, i_b]
                for i_d in qd.static(range(3)):
                    if dir_contacts[i_d] > 0.0:
                        qd.atomic_min(sph_state.envs_solve.step_max[i_b], p_contacts[i_d] / dir_contacts[i_d])


@qd.func
def func_pressure_dir_vel(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """Compute the velocity change of the sorted particles per unit step along the vector the next product takes."""
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        # One thread clears the loop flag and counts the pass, which spares the pass the launch of serial statements
        # (see func_pressure_next)
        if i_t == 0:
            sph_state.solve_loop.is_looping[()] = 0
            sph_state.solve_loop.n_passes[None] = sph_state.solve_loop.n_passes[None] + 1
        is_solved = sph_state.envs_solve.is_solving[i_b] and sph_state.particles_reordered.is_active[i_p, i_b]
        # The pressures themselves when the solve recomputes its gradient, the search direction otherwise
        is_recomputing = sph_state.envs_solve.is_recomputing[i_b]
        pressure_i = gs.qd_float(0.0)
        pressures_contacts = qd.Vector.zero(gs.qd_float, 3)
        if is_solved:
            pressure_i = sph_state.pressure_solve.dir[i_p, i_b]
            pressures_contacts = sph_state.pressure_solve.dir_contacts[i_p, i_b]
            if is_recomputing:
                pressure_i = sph_state.particles_reordered.p[i_p, i_b]
                pressures_contacts = sph_state.pressure_solve.p_contacts[i_p, i_b]
        ret = qd.Struct(
            force=qd.Vector.zero(gs.qd_float, 3),
            pressure_i=pressure_i,
            is_recomputing=is_recomputing,
            pos_i=qd.Vector.zero(gs.qd_float, 3),
            image_rot=qd.Matrix.identity(gs.qd_float, 3),
            image_shift=qd.Vector.zero(gs.qd_float, 3),
            image_vel=qd.Vector.zero(gs.qd_float, 3),
            i_surface=-1,
        )
        # The pair takes the sum of both pressures, an image of a particle takes its pressure (see
        # func_for_all_interactions), and the contacts push the particle along the normal of their surface
        if is_solved:
            func_for_all_interactions(
                i_p, i_b, i_lane, ret, sph_state, sph_info, sph_config, func_task_compute_pressure_vel, n_lanes=N_LANES
            )
        if qd.static(N_LANES > 1):
            for i_d in qd.static(range(3)):
                ret.force[i_d] = qd_block_sum(ret.force[i_d], sph_config.log2_neighbor_lanes)
        if is_solved and i_lane == 0:
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            normals, depths, normal_vels, contacts_weight = func_contacts(i_p, i_b, sph_state, sph_info, sph_config)
            force = normals.transpose() @ pressures_contacts / sph_info.particle_size[None] - ret.force
            vel = sph_info.substep_dt[None] / sph_info.particles.rho[i_o] * force
            sph_state.pressure_solve.dir_vel[i_p, i_b] = vel


@qd.func
def func_pressure_step(sph_state: SPHState):
    """
    Compute the length of the next step of the pressure solve of each environment.

    A step along the search direction is that of conjugate gradients, or of steepest descent along the chopped gradient,
    the exact minimum of the objective along the direction. A conjugate gradient step that would drive a pressure
    negative stops where the first pressure reaches zero and expands the active set from there (see
    func_pressure_update).
    """
    for i_b in range(sph_state.envs_solve.is_solving.shape[0]):
        if sph_state.envs_solve.is_solving[i_b] and not sph_state.envs_solve.is_recomputing[i_b]:
            # The slots of the dot products of the search direction (see SPHEnvsSolve)
            sums = qd.Vector.zero(gs.qd_float, 2)
            for i_s in range(2):
                sums[i_s] = func_exact_sum_take(3 + i_s, i_b, sph_state.envs_solve.sums)
            dir_prod_sum, grad_dir_sum = sums[0], sums[1]
            step = gs.qd_float(0.0)
            if dir_prod_sum > 0.0:
                step = grad_dir_sum / dir_prod_sum
            step_max = sph_state.envs_solve.step_max[i_b]
            is_expanding = not sph_state.envs_solve.is_proportioning[i_b] and step > step_max
            if is_expanding:
                step = step_max
            sph_state.envs_solve.step[i_b] = step
            sph_state.envs_solve.dir_prod_norm[i_b] = dir_prod_sum
            sph_state.envs_solve.is_expanding[i_b] = is_expanding
            sph_state.envs_solve.step_max[i_b] = gs.qd_float(float("inf"))


@qd.func
def func_pressure_update(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Take the step of the pressure solve of each environment, and sum what decides the next one.

    An expansion step continues with a fixed step of projected steepest descent along the free gradient, which releases
    the particles whose pressure would need to turn negative, after which the environment recomputes its gradient. The
    other steps yield the sums of func_pressure_next: the projected gradient, the norms of the chopped gradient and of
    the reduced free gradient, which decide between conjugate gradients and proportioning, and the product of the free
    gradient with the operator times the search direction, which conjugates the next direction.
    """
    expansion_step = sph_info.expansion_step[None]
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.p))
    ):
        if sph_state.envs_solve.is_solving[i_b] and sph_state.particles_reordered.is_active[i_p, i_b]:
            # The pressure of the particle comes first, then those of its contacts (see func_contacts_diag)
            diags_contacts = func_contacts_diag(i_p, i_b, sph_state, sph_info, sph_config)
            pressures = qd.Vector(
                [
                    sph_state.particles_reordered.p[i_p, i_b],
                    sph_state.pressure_solve.p_contacts[i_p, i_b][0],
                    sph_state.pressure_solve.p_contacts[i_p, i_b][1],
                    sph_state.pressure_solve.p_contacts[i_p, i_b][2],
                ]
            )
            grads = qd.Vector(
                [
                    sph_state.pressure_solve.grad[i_p, i_b],
                    sph_state.pressure_solve.grad_contacts[i_p, i_b][0],
                    sph_state.pressure_solve.grad_contacts[i_p, i_b][1],
                    sph_state.pressure_solve.grad_contacts[i_p, i_b][2],
                ]
            )
            dirs = qd.Vector(
                [
                    sph_state.pressure_solve.dir[i_p, i_b],
                    sph_state.pressure_solve.dir_contacts[i_p, i_b][0],
                    sph_state.pressure_solve.dir_contacts[i_p, i_b][1],
                    sph_state.pressure_solve.dir_contacts[i_p, i_b][2],
                ]
            )
            dir_prods = qd.Vector(
                [
                    sph_state.pressure_solve.dir_prod[i_p, i_b],
                    sph_state.pressure_solve.dir_prod_contacts[i_p, i_b][0],
                    sph_state.pressure_solve.dir_prod_contacts[i_p, i_b][1],
                    sph_state.pressure_solve.dir_prod_contacts[i_p, i_b][2],
                ]
            )
            diags_inv = qd.Vector(
                [
                    sph_state.pressure_solve.diag_inv[i_p, i_b],
                    1.0 / diags_contacts[0],
                    1.0 / diags_contacts[1],
                    1.0 / diags_contacts[2],
                ]
            )
            if not sph_state.envs_solve.is_recomputing[i_b]:
                step = sph_state.envs_solve.step[i_b]
                pressures = qd.max(pressures - step * dirs, 0.0)
                grads = grads - step * dir_prods
            grads_free = qd.select(pressures > 0.0, grads, 0.0)
            if sph_state.envs_solve.is_expanding[i_b]:
                pressures = qd.max(pressures - expansion_step * diags_inv * grads_free, 0.0)
            else:
                grads_chopped = qd.select(pressures > 0.0, 0.0, qd.min(grads, 0.0))
                # The particles free of pressure have no free gradient, and would divide zero by zero
                grads_reduced = qd.select(
                    pressures > 0.0, qd.min(pressures / (expansion_step * diags_inv), grads_free), 0.0
                )
                # The search direction is stale while the environment recomputes its gradient
                free_prod = gs.qd_float(0.0)
                if not sph_state.envs_solve.is_recomputing[i_b]:
                    free_prod = (diags_inv * grads_free * dir_prods).sum()
                terms = qd.Vector(
                    [(diags_inv * grads_chopped**2).sum(), (diags_inv * grads_reduced * grads_free).sum(), free_prod]
                )
                for i_s in range(3):
                    func_exact_sum_add(i_s, i_b, terms[i_s], sph_state.envs_solve.sums)
                qd.atomic_max(sph_state.envs_solve.err_max[i_b], qd.abs(grads_free + grads_chopped).max())
            sph_state.particles_reordered.p[i_p, i_b] = pressures[0]
            sph_state.pressure_solve.p_contacts[i_p, i_b] = qd.Vector([pressures[1], pressures[2], pressures[3]])
            sph_state.pressure_solve.grad[i_p, i_b] = grads[0]
            sph_state.pressure_solve.grad_contacts[i_p, i_b] = qd.Vector([grads[1], grads[2], grads[3]])


@qd.func
def func_pressure_next(n_iters_max: int, sph_state: SPHState, sph_info: SPHInfo):
    """
    Stop the pressure solve of the environments that converged, and choose the next step of the others.

    Each environment stops once the largest relative density error of its particles, the largest entry of the
    projected gradient, meets the tolerance, so that its result is the same whatever the other environments simulate
    alongside it. The next step
    is a conjugate gradient step while the chopped gradient is small against the reduced free gradient, and a
    proportioning step along the chopped gradient otherwise (modified proportioning with reduced gradient projections,
    MPRGP). The loop runs on as long as an environment iterates on, for `n_iters_max` iterations at most.
    """
    for i_b in range(sph_state.envs_solve.is_solving.shape[0]):
        if sph_state.envs_solve.is_solving[i_b]:
            if sph_state.envs_solve.is_expanding[i_b]:
                sph_state.envs_solve.is_expanding[i_b] = False
                sph_state.envs_solve.is_recomputing[i_b] = True
                if sph_state.solve_loop.n_passes[None] < n_iters_max:
                    sph_state.solve_loop.is_looping[()] = 1
            else:
                sums = qd.Vector.zero(gs.qd_float, 3)
                for i_s in range(3):
                    sums[i_s] = func_exact_sum_take(i_s, i_b, sph_state.envs_solve.sums)
                chopped_norm, reduced_norm, free_prod_sum = sums[0], sums[1], sums[2]
                err_max = sph_state.envs_solve.err_max[i_b]
                sph_state.envs_solve.err_max[i_b] = 0.0
                is_conjugating = not (
                    sph_state.envs_solve.is_recomputing[i_b] or sph_state.envs_solve.is_proportioning[i_b]
                )
                is_proportioning = chopped_norm > sph_info.proportioning_ratio[None] * reduced_norm
                dir_coef = gs.qd_float(0.0)
                if is_conjugating and not is_proportioning and sph_state.envs_solve.dir_prod_norm[i_b] > 0.0:
                    dir_coef = free_prod_sum / sph_state.envs_solve.dir_prod_norm[i_b]
                sph_state.envs_solve.dir_coef[i_b] = dir_coef
                sph_state.envs_solve.is_proportioning[i_b] = is_proportioning
                sph_state.envs_solve.is_recomputing[i_b] = False
                if err_max <= sph_info.density_err_tol[None]:
                    sph_state.envs_solve.is_solving[i_b] = False
                elif sph_state.solve_loop.n_passes[None] < n_iters_max:
                    sph_state.solve_loop.is_looping[()] = 1


@qd.func
def func_pressure_dir(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """Compute the next search direction of the pressure solve of each environment (see func_pressure_next)."""
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.pressure_solve.dir))
    ):
        if sph_state.envs_solve.is_solving[i_b] and not sph_state.envs_solve.is_recomputing[i_b]:
            dirs = qd.Vector.zero(gs.qd_float, 4)
            if sph_state.particles_reordered.is_active[i_p, i_b]:
                diags_contacts = func_contacts_diag(i_p, i_b, sph_state, sph_info, sph_config)
                diags_inv = qd.Vector(
                    [
                        sph_state.pressure_solve.diag_inv[i_p, i_b],
                        1.0 / diags_contacts[0],
                        1.0 / diags_contacts[1],
                        1.0 / diags_contacts[2],
                    ]
                )
                pressures = qd.Vector(
                    [
                        sph_state.particles_reordered.p[i_p, i_b],
                        sph_state.pressure_solve.p_contacts[i_p, i_b][0],
                        sph_state.pressure_solve.p_contacts[i_p, i_b][1],
                        sph_state.pressure_solve.p_contacts[i_p, i_b][2],
                    ]
                )
                grads = qd.Vector(
                    [
                        sph_state.pressure_solve.grad[i_p, i_b],
                        sph_state.pressure_solve.grad_contacts[i_p, i_b][0],
                        sph_state.pressure_solve.grad_contacts[i_p, i_b][1],
                        sph_state.pressure_solve.grad_contacts[i_p, i_b][2],
                    ]
                )
                if sph_state.envs_solve.is_proportioning[i_b]:
                    dirs = qd.select(pressures > 0.0, 0.0, diags_inv * qd.min(grads, 0.0))
                else:
                    dirs_prev = qd.Vector(
                        [
                            sph_state.pressure_solve.dir[i_p, i_b],
                            sph_state.pressure_solve.dir_contacts[i_p, i_b][0],
                            sph_state.pressure_solve.dir_contacts[i_p, i_b][1],
                            sph_state.pressure_solve.dir_contacts[i_p, i_b][2],
                        ]
                    )
                    dirs = qd.select(
                        pressures > 0.0, diags_inv * grads - sph_state.envs_solve.dir_coef[i_b] * dirs_prev, 0.0
                    )
            sph_state.pressure_solve.dir[i_p, i_b] = dirs[0]
            sph_state.pressure_solve.dir_contacts[i_p, i_b] = qd.Vector([dirs[1], dirs[2], dirs[3]])


@qd.kernel(graph=True, fastcache=True)
def kernel_solve_pressure(
    n_iters_max: int,
    is_looping: qd.types.ndarray(qd.i32, ndim=0),
    sph_state: SPHState,
    sph_info: SPHInfo,
    sph_config: qd.template(),
):
    """
    Solve the pressures of the sorted particles that keep the liquid incompressible over the substep, on the device.

    The pressures are the non-negative minimizers of a convex quadratic, whose gradient is the relative density deficit
    of the particles at the end of the substep and whose operator, the rate at which the pressures change the density,
    is symmetric positive semi-definite. Its minimum keeps every particle at most at its rest density, and the particles
    under pressure at exactly their rest density. Modified proportioning with reduced gradient projections (MPRGP), a
    conjugate gradient method that expands and shrinks the set of the particles under pressure, solves it, with the
    inverse diagonal of the operator as preconditioner.
    """
    sph_state.solve_loop.n_passes[None] = 0
    is_looping[()] = 1
    # FIXME: quadrants#1083 - the statements before a graph do_while run again on every pass when the loop body opens
    # with bare statements, so the loop body opens with a for loop.
    while qd.graph.do_while(is_looping):
        func_pressure_dir_vel(sph_state, sph_info, sph_config)
        func_pressure_product(sph_state, sph_info, sph_config)
        func_pressure_step(sph_state)
        func_pressure_update(sph_state, sph_info, sph_config)
        func_pressure_next(n_iters_max, sph_state, sph_info)
        func_pressure_dir(sph_state, sph_info, sph_config)


@qd.func
def func_task_apply_pressure(i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo):
    offset_ij = ret.pos_i - (ret.image_rot @ sph_state.particles_reordered.pos[j_p, i_b] + ret.image_shift)
    grad_ij = sph_info.particle_volume[None] * func_kernel_gradient(
        offset_ij, sph_info.support_radius[None], sph_info.kernel_norm[None]
    )
    pressure_j = sph_state.particles_reordered.p[j_p, i_b]
    ret.force = ret.force + (ret.pressure_i + pressure_j) * grad_ij
    # The moving boundary moves the image of the pair by its own velocity, which makes it take the reaction of the
    # pressure of the particle along the reflected part of the kernel gradient (see func_is_interacting)
    for i_s in qd.static(range(3)):
        if ret.i_surface == i_s:
            normal = qd.Vector([ret.surfaces_normal[i_s, 0], ret.surfaces_normal[i_s, 1], ret.surfaces_normal[i_s, 2]])
            reflected_grad = grad_ij - ret.image_rot @ grad_ij
            ret.surfaces_force[i_s] = ret.surfaces_force[i_s] + ret.pressure_i * normal.dot(reflected_grad)


@qd.kernel(fastcache=True)
def kernel_apply_pressure(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """Change the velocity of the sorted particles by their pressures, and give the moving boundaries the reaction."""
    N_LANES = qd.static(2**sph_config.log2_neighbor_lanes)
    n_particles, n_envs = sph_state.particles_reordered.pos.shape
    for i_t in range(n_envs * n_particles * N_LANES):
        i_lane = i_t % N_LANES
        i_b = i_t // N_LANES % n_envs
        i_p = i_t // (N_LANES * n_envs)
        is_active = sph_state.particles_reordered.is_active[i_p, i_b]
        normals, depths, normal_vels, contacts_weight = func_contacts(i_p, i_b, sph_state, sph_info, sph_config)
        ret = qd.Struct(
            force=qd.Vector.zero(gs.qd_float, 3),
            pressure_i=sph_state.particles_reordered.p[i_p, i_b],
            surfaces_force=qd.Vector.zero(gs.qd_float, 3),
            surfaces_normal=normals,
            pos_i=qd.Vector.zero(gs.qd_float, 3),
            image_rot=qd.Matrix.identity(gs.qd_float, 3),
            image_shift=qd.Vector.zero(gs.qd_float, 3),
            image_vel=qd.Vector.zero(gs.qd_float, 3),
            i_surface=-1,
        )
        if is_active:
            func_for_all_interactions(
                i_p, i_b, i_lane, ret, sph_state, sph_info, sph_config, func_task_apply_pressure, n_lanes=N_LANES
            )
        if qd.static(N_LANES > 1):
            for i_d in qd.static(range(3)):
                ret.force[i_d] = qd_block_sum(ret.force[i_d], sph_config.log2_neighbor_lanes)
                ret.surfaces_force[i_d] = qd_block_sum(ret.surfaces_force[i_d], sph_config.log2_neighbor_lanes)
        if is_active and i_lane == 0:
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            dt, particle_size = sph_info.substep_dt[None], sph_info.particle_size[None]
            pressures_contacts = sph_state.pressure_solve.p_contacts[i_p, i_b]
            force = normals.transpose() @ pressures_contacts / particle_size - ret.force
            sph_state.particles_reordered.vel[i_p, i_b] = (
                sph_state.particles_reordered.vel[i_p, i_b] + dt / sph_info.particles.rho[i_o] * force
            )
            if qd.static(sph_config.has_boundary):
                # Each surface takes the opposite of the impulse the particle takes from it, along its normal
                volume = sph_info.particles.mass[i_o] / sph_info.particles.rho[i_o]
                impulses = dt * volume * (ret.surfaces_force - pressures_contacts / particle_size)
                for i_s in qd.static(range(3)):
                    sph_state.boundary.surfaces_impulse[i_o, i_s, i_b] = impulses[i_s]


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
