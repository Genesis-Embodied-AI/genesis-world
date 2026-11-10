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
    func_hash_grid_to_slot,
    func_hash_n_cells,
    func_hash_pos_to_grid,
    func_hash_sort_particles,
    func_hash_wrap_cell,
    hash_grid_sizes,
)

from .base_solver import SimulatorState
from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


# The arrays of divergence-free smoothed particle hydrodynamics (DFSPH), which the weakly compressible solver leaves out
DFSPH = "dfsph"


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

    `rho` is the rest density, `stiffness` and `exponent` the parameters of the equation of state, `mu` the viscosity
    and `surface_tension` the surface tension coefficient, in N/m.
    """

    rho: qd.Tensor = field(float)
    mass: qd.Tensor = field(float)
    stiffness: qd.Tensor = field(float)
    exponent: qd.Tensor = field(float)
    mu: qd.Tensor = field(float)
    surface_tension: qd.Tensor = field(float)


class SPHEnvsSolve(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The convergence of the divergence and density solves of divergence-free smoothed particle hydrodynamics (DFSPH) in
    each environment.

    `err` is the density error summed over the active particles, `n_active` the number of active particles, over which
    it is averaged, and `is_solving` whether the environment iterates on.
    """

    err: qd.Tensor = field(float)
    n_active: qd.Tensor = field(int)
    is_solving: qd.Tensor = field(bool)


class SPHBoundaryParticles(Data, kind=DataKind.SCRATCH, shape=(PARTICLES, ENVS)):
    """The moving boundaries around each SPH particle over the current substep, in the order of the particles."""

    # The ghost particles of the boundaries around the particle, summed up as the walls of the domain (see
    # func_wall_volume)
    volume: qd.Tensor = field(float)
    volume_grad: qd.Tensor = field((float, 3))
    moment: qd.Tensor = field((float, (3, 3)))
    # The volumes of the ghost particles times their velocity dotted with the kernel gradient, summed, which moves the
    # density like the neighbors of the particle
    vel_flux: qd.Tensor = field(float)
    # The particle pushes each ghost particle by `(impulse_coef + impulse_gravity_coef * gravity.dot(r)) * grad` over
    # the substep, with `grad` the volume gradient of the ghost particle and `r` its offset from the particle
    impulse_coef: qd.Tensor = field(float)
    impulse_gravity_coef: qd.Tensor = field(float)


class SPHGhostParticles(Data, kind=DataKind.SCRATCH, shape=(PARTICLES, ENVS)):
    """The ghost particles of a moving boundary over the current substep (see SPHBoundaryGhosts)."""

    pos: qd.Tensor = field((float, 3))
    vel: qd.Tensor = field((float, 3))
    volume: qd.Tensor = field(float)
    is_active: qd.Tensor = field(bool)


class SPHBoundaryGhosts(Data, kind=DataKind.SCRATCH):
    """
    A moving boundary of a smoothed particle hydrodynamics (SPH) liquid, as liquid particles at rest relative to it.

    Their owner places them and sorts them in `hash_grid` every substep, before the SPH solver sums them up around its
    particles (see SPHBoundaryParticles).
    """

    particles: SPHGhostParticles
    hash_grid: SpatialHashGrid


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
    particles_k_reordered: qd.Tensor = field(float, shape=(PARTICLES, ENVS), kind=DataKind.SCRATCH, labels=DFSPH)
    envs_solve: SPHEnvsSolve = field(labels=DFSPH)
    # The moving boundaries around each particle, which a coupler opts in
    boundary: SPHBoundaryParticles = field(opt_in=True)
    hash_grid: SpatialHashGrid
    # The error flags of each env, which no kernel of the solver raises (see 'ErrnoMixin.register_errors')
    errno: qd.Tensor = field(int, shape=(ENVS,))


class SPHColumns(Data, kind=DataKind.SCRATCH):
    """
    The sorted particles of weakly compressible smoothed particle hydrodynamics (WCSPH) sorted by the vertical column
    of the hash grid they lie in, for the height of the free surface above each particle.
    """

    # The position of each sorted particle projected onto the floor of the domain
    pos: qd.Tensor = field((float, 3), shape=(PARTICLES, ENVS))
    grid: SpatialHashGrid


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
    kernel_coef: qd.Tensor = field(float, shape=())
    # The distance below which the gradient of the cubic kernel is zero
    kernel_dist_min: qd.Tensor = field(float, shape=())
    # The squared distance that regularizes the viscosity of a pair of particles
    viscosity_eps: qd.Tensor = field(float, shape=())
    # The magnitude of the color gradient below which a particle takes no part in the surface tension
    surface_normal_min: qd.Tensor = field(float, shape=())
    # The average density error that ends the density solve of DFSPH, in kg/m^3
    density_err_tol: qd.Tensor = field(float, shape=(), labels=DFSPH)
    # The average density rate that ends the divergence solve of DFSPH, in kg/m^3/s
    divergence_err_tol: qd.Tensor = field(float, shape=(), labels=DFSPH)
    # The stiffness below which a pair of particles exchanges no velocity
    stiffness_eps: qd.Tensor = field(float, shape=(), labels=DFSPH)
    jacobi_relaxation: qd.Tensor = field(float, shape=(), labels=DFSPH)
    # The share of the pressure of the previous substep that warm-starts the density solve
    warm_start_scale: qd.Tensor = field(float, shape=(), labels=DFSPH)
    # The sum of squared kernel gradients below which a particle takes no pressure
    factor_eps: qd.Tensor = field(float, shape=(), labels=DFSPH)
    # The number of neighbors below which the density rate of a particle is left out of the divergence solve
    n_neighbors_min: qd.Tensor = field(int, shape=(), labels=DFSPH)


class SPHConfig(Config):
    """The parts of the smoothed particle hydrodynamics (SPH) solver that its kernels compile in."""

    # Whether the domain is thinner than twice the support radius along an axis, which puts the walls on both sides of
    # a particle within its support radius (see func_wall_volume)
    is_domain_thin: bool
    # Whether moving boundaries push on the particles (see SPHBoundaryParticles)
    has_boundary: bool


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
        self._pressure_solver = options.pressure_solver

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
        self.sph_columns: SPHColumns | None = None
        self.sph_info: SPHInfo | None = None
        self.sph_config: SPHConfig | None = None

    def build(self) -> None:
        super().build()

        is_dfsph = self._pressure_solver == "DFSPH"
        unused_labels = dict(dfsph=not is_dfsph) if self.is_active else True
        sizes = dict(envs=self._B, particles=self._n_particles)
        hash_grid_values = dict(cell_size=self._hash_grid_cell_size, grid_res=self._hash_grid_res)
        self.sph_state = SPHState(
            sizes=dict(
                sizes,
                **hash_grid_sizes(math.prod(self._hash_grid_res)),
            ),
            values=dict(hash_grid=hash_grid_values),
            env_first=False,
            unused=unused_labels,
        )
        self.register_errors(self.sph_state.errno, {})
        # The columns of the hash grid, one cell high
        columns_grid_res = (*self._hash_grid_res[:2], 1)
        self.sph_columns = SPHColumns(
            sizes=dict(sizes, **hash_grid_sizes(math.prod(columns_grid_res))),
            values=dict(grid=dict(cell_size=self._hash_grid_cell_size, grid_res=columns_grid_res)),
            env_first=False,
            unused=not self.is_active or is_dfsph,
        )
        self.sph_config = SPHConfig(
            is_domain_thin=self.is_active
            and bool(np.any(self.boundary_upper - self.boundary_lower < 2 * self._support_radius)),
            has_boundary=self.is_active and is_opted_in(SPHState.boundary),
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
                kernel_coef=8.0 / np.pi / self._support_radius**3,
                kernel_dist_min=1e-5,
                viscosity_eps=viscosity_eps,
                # The color gradient of the particles deeper than the first two layers below the free surface is noise,
                # whose direction would corrupt the curvature (see func_task_compute_non_pressure_forces)
                surface_normal_min=0.2 / self._support_radius,
            )
        if self.is_active and is_dfsph:
            # TODO: Support per-particle density
            # The rest density the incompressibility solves target, that of the first particle as the particles store it
            density0 = float(gs.np_float(self._entities[0].material.rho))
            info_values.update(
                density_err_tol=self._df_max_error_den * 0.01 * density0,
                divergence_err_tol=1.0 / self.substep_dt * self._df_max_error_div * 0.01 * density0,
                stiffness_eps=1e-5,
                # The DFSPH factor inverts the diagonal of the pressure system alone, and the stiffness of the neighbors
                # of a particle adds up to as much again, so the diagonally scaled system has eigenvalues up to 2. A
                # full Jacobi update leaves the mode alternating from one particle to the next undamped, and overshoots
                # once the liquid compresses abruptly. Half the update cancels that mode, the others still converging.
                jacobi_relaxation=0.5,
                # FIXME: A larger share saves density iterations but raises the pressure scatter between substeps, which
                # turns a thin slab unstable above a quarter. A better-converged warm start would remove this trade-off.
                warm_start_scale=0.25,
                factor_eps=1e-6,
                n_neighbors_min=20,
            )
        self.sph_info = SPHInfo(sizes=sizes, values=info_values, unused=unused_labels)
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
            material.stiffness,
            material.exponent,
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

        # The convergence is checked before the first iteration of the divergence solve, which a divergence-free
        # velocity field leaves without iterations. The warm start of the density solve applies the pressure carried
        # over from the previous substep, which overshoots wherever that pressure corrected a density error at once.
        # The average density error leaves such particles unnoticed, so the density solve iterates at least once to
        # take their pressure back. Otherwise, the pressure keeps pushing them apart substep after substep.
        if solve_density:
            kernel_solver_iteration(self.sph_state, self.sph_info, self.sph_config, solve_density, warm_start=True)
        kernel_compute_density_rate(self.sph_state, self.sph_info, self.sph_config, solve_density)
        n_envs_solving = kernel_update_convergence(
            self.sph_state, self.sph_info, solve_density, can_stop=not solve_density
        )
        n_iters = 0
        while n_envs_solving > 0 and n_iters < n_iters_max:
            kernel_solver_iteration(self.sph_state, self.sph_info, self.sph_config, solve_density, warm_start=False)
            kernel_compute_density_rate(self.sph_state, self.sph_info, self.sph_config, solve_density)
            n_envs_solving = kernel_update_convergence(self.sph_state, self.sph_info, solve_density, can_stop=True)
            n_iters += 1

        gs.logger.debug(f"DFSPH - {'density' if solve_density else 'divergence'} solve iterations: {n_iters}")

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self):
        if not self.is_active:
            return
        kernel_reorder_particles(self.sph_state)
        kernel_compute_rho(self.sph_state, self.sph_info, self.sph_config)
        if self._pressure_solver == "WCSPH":
            kernel_compute_non_pressure_forces(self._sim.sim_state, self.sph_state, self.sph_info, self._force_fields)
            kernel_compute_pressure_forces(self.sph_state, self.sph_columns, self.sph_info, self.sph_config)
            kernel_advect_velocity(self.sph_state, self.sph_info)
        else:
            kernel_compute_dfsph_factor(self.sph_state, self.sph_info, self.sph_config)
            self._solve_incompressibility(solve_density=False)
            kernel_compute_non_pressure_forces(self._sim.sim_state, self.sph_state, self.sph_info, self._force_fields)
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
        yield from iter_data(self, "sph_columns")
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
    def pressure_solver(self) -> str:
        return self._pressure_solver

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
def func_cubic_kernel(r_norm: float, h: float, k: float):
    """
    Return the cubic spline smoothing kernel at a distance.

    `h` is the support radius of the kernel, and `k` its normalization.
    """
    res = gs.qd_float(0.0)
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = q**2
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
        else:
            res = 2 * k * (1.0 - q) ** 3
    return res


@qd.func
def func_cubic_kernel_derivative(r: qd.types.vector(3), h: float, k: float, dist_min: float):
    """Return the gradient of the cubic spline smoothing kernel at a vector (see func_cubic_kernel)."""
    res = qd.Vector.zero(gs.qd_float, 3)
    r_norm = r.norm()
    q = r_norm / h
    if r_norm > dist_min and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
        else:
            res = -6.0 * k * (1.0 - q) ** 2 * grad_q
    return res


@qd.func
def func_cubic_kernel_plane_integral(dist: float, sph_info: SPHInfo):
    """
    Return the integral of the cubic spline smoothing kernel over the plane at a distance from its center.

    Closed form of 2 * pi * int_{|dist|}^{h} W(r) r dr, with h the support radius. Its own integral along the plane
    normal equals half the kernel normalization: int_{0}^{h} of this quantity over dist is exactly 1/2, which makes
    forces weighted by it integrate to a per-unit-area magnitude over a covering layer of particles.
    """
    res = gs.qd_float(0.0)
    h = sph_info.support_radius[None]
    q = qd.abs(dist) / h
    if q <= 1.0:
        q2 = q**2
        q3 = q2 * q
        if q <= 0.5:
            res = 0.0875 - (1.2 * q3 * q2 - 1.5 * q2**2 + 0.5 * q2)
        else:
            res = 0.1 - (q2 - 2.0 * q3 + 1.5 * q2**2 - 0.4 * q3 * q2)
        res = res * 16.0 / h
    return res


@qd.func
def func_wall_volume(pos: qd.types.vector(3), sph_info: SPHInfo, sph_config: qd.template()):
    """
    Return the kernel-weighted volume of the walls of the domain around a point, with its gradient and first moment.

    The walls stand for liquid particles at rest beyond the domain, on the lattice of the samplers continued from each
    wall. The first moment, of the gradient over the offsets to these ghost particles, turns a linear pressure field
    over them into a force.
    """
    # Without the ghost particles, the particles within the support radius of a wall lack neighbors, and the liquid
    # compresses until they make up for them. The support radius spans two particle sizes, so four nodes per axis on the
    # lattice of the nearest wall cover it. In a thin domain, the nodes within the support radius along an axis are
    # those beyond the lower wall on its own lattice, those inside the domain on the lattice of the nearest wall, and
    # those beyond the upper wall on its own lattice, in this order, five at most since the lattice of the far wall is
    # offset from that of the nearest one. A thin domain alone pays for these nodes, which slow down every step.
    N_NODES_AXIS = qd.static(5 if sph_config.is_domain_thin else 4)
    h, k = sph_info.support_radius[None], sph_info.kernel_coef[None]
    particle_size, particle_volume = sph_info.particle_size[None], sph_info.particle_volume[None]
    volume = gs.qd_float(0.0)
    volume_grad = qd.Vector.zero(gs.qd_float, 3)
    moment = qd.Matrix.zero(gs.qd_float, 3, 3)
    dist_lower = pos - sph_info.boundary.lower[None]
    dist_upper = sph_info.boundary.upper[None] - pos
    if qd.min(dist_lower.min(), dist_upper.min()) < h:
        dist_near = qd.min(dist_lower, dist_upper)
        near_dir = qd.select(dist_lower <= dist_upper, 1.0, -1.0)
        lattice_start = qd.floor(dist_near / particle_size - 0.5 * (N_NODES_AXIS - 1), gs.qd_int)
        n_lower = qd.Vector.zero(gs.qd_int, 3)
        n_inner = qd.Vector.zero(gs.qd_int, 3)
        n_upper = qd.Vector.zero(gs.qd_int, 3)
        if qd.static(sph_config.is_domain_thin):
            n_lower = qd.max(qd.ceil((h - dist_lower) / particle_size - 0.5, gs.qd_int), 0)
            n_upper = qd.max(qd.ceil((h - dist_upper) / particle_size - 0.5, gs.qd_int), 0)
            lattice_start = qd.max(qd.floor((dist_near - h) / particle_size - 0.5, gs.qd_int) + 1, 0)
            lattice_end = qd.min(
                qd.ceil((dist_near + h) / particle_size - 0.5, gs.qd_int),
                qd.floor((dist_lower + dist_upper) / particle_size - 0.5, gs.qd_int) + 1,
            )
            n_inner = qd.max(lattice_end - lattice_start, 0)
        for i_node in range(N_NODES_AXIS**3):
            i_n = qd.Vector(
                [i_node // N_NODES_AXIS**2, i_node // N_NODES_AXIS % N_NODES_AXIS, i_node % N_NODES_AXIS], dt=gs.qd_int
            )
            # The offset to the node along each axis, and whether the node lies beyond a wall
            r = near_dir * ((lattice_start + i_n + 0.5) * particle_size - dist_near)
            is_ghost = (lattice_start + i_n < 0).any()
            if qd.static(sph_config.is_domain_thin):
                is_lower = i_n < n_lower
                is_upper = i_n >= n_lower + n_inner
                r = qd.select(
                    is_lower,
                    -dist_lower - (i_n + 0.5) * particle_size,
                    qd.select(
                        is_upper,
                        dist_upper + (i_n - n_lower - n_inner + 0.5) * particle_size,
                        near_dir * ((lattice_start + i_n - n_lower + 0.5) * particle_size - dist_near),
                    ),
                )
                is_ghost = (i_n < n_lower + n_inner + n_upper).all() and (is_lower | is_upper).any()
            if is_ghost:
                grad = -particle_volume * func_cubic_kernel_derivative(r, h, k, sph_info.kernel_dist_min[None])
                volume = volume + particle_volume * func_cubic_kernel(r.norm(), h, k)
                volume_grad = volume_grad + grad
                moment = moment + grad.outer_product(r)
    return volume, volume_grad, moment


@qd.func
def func_boundary_volume(i_p: int, i_b: int, sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Return the kernel-weighted volume of the walls and the moving boundaries around a sorted particle, with its gradient
    and first moment (see func_wall_volume and SPHBoundaryParticles).
    """
    volume, volume_grad, moment = func_wall_volume(sph_state.particles_reordered.pos[i_p, i_b], sph_info, sph_config)
    if qd.static(sph_config.has_boundary):
        i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
        volume = volume + sph_state.boundary.volume[i_o, i_b]
        volume_grad = volume_grad + sph_state.boundary.volume_grad[i_o, i_b]
        moment = moment + sph_state.boundary.moment[i_o, i_b]
    return volume, volume_grad, moment


@qd.func
def func_for_all_neighbors(
    i_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo, task: qd.template()
):
    """
    Run a task on every sorted particle within the support radius of a sorted particle, the particle itself excluded.

    `task` takes the indices of both particles, the environment, `ret`, the state and the info, in this order, and
    accumulates its result in `ret`.
    """
    pos_i = sph_state.particles_reordered.pos[i_p, i_b]
    base = func_hash_wrap_cell(func_hash_pos_to_grid(pos_i, sph_state.hash_grid) - 1, sph_state.hash_grid)
    n_cells = func_hash_n_cells(3, sph_state.hash_grid)
    for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
        i_s = func_hash_cell_to_slot(base + offset, sph_state.hash_grid)
        slot_start = sph_state.hash_grid.slot_start[i_s, i_b]
        for j_p in range(slot_start, sph_state.hash_grid.slot_size[i_s, i_b] + slot_start):
            d_ij = pos_i - sph_state.particles_reordered.pos[j_p, i_b]
            if i_p != j_p and d_ij.norm() < sph_info.support_radius[None]:
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
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    h, k = sph_info.support_radius[None], sph_info.kernel_coef[None]
    ret.den = ret.den + sph_info.particle_volume[None] * func_cubic_kernel(d_ij.norm(), h, k)
    ret.normal = ret.normal + sph_info.particle_volume[None] * func_cubic_kernel_derivative(
        d_ij, h, k, sph_info.kernel_dist_min[None]
    )


@qd.kernel(fastcache=True)
def kernel_compute_rho(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """Compute the density and the color gradient of the sorted particles from their neighbours and the walls."""
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.rho))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            # Base density using the kernel at distance 0
            h, k = sph_info.support_radius[None], sph_info.kernel_coef[None]
            rho = sph_info.particle_volume[None] * func_cubic_kernel(0.0, h, k)
            wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(i_p, i_b, sph_state, sph_info, sph_config)
            ret = qd.Struct(den=wall_volume, normal=wall_volume_grad)
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_rho)
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            sph_state.particles_reordered.rho[i_p, i_b] = (rho + ret.den) * sph_info.particles.rho[i_o]
            sph_state.particles_normal_reordered[i_p, i_b] = ret.normal


@qd.func
def func_task_compute_non_pressure_forces(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    dist = d_ij.norm()
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
            d_ij, sph_info.support_radius[None], sph_info.kernel_coef[None], sph_info.kernel_dist_min[None]
        )
        curvature = -vol_j * (normal_j.normalized() - normal_i.normalized()).dot(grad_ij)
        ret.acc += ret.surface_tension_i * curvature / sph_state.particles_reordered.rho[i_p, i_b] * normal_i

    # Viscosity, normalized by twice the dimension plus two
    viscosity_coef = 2 * (3 + 2)
    v_ij = (sph_state.particles_reordered.vel[i_p, i_b] - sph_state.particles_reordered.vel[j_p, i_b]).dot(d_ij)
    rho_j = sph_state.particles_reordered.rho[j_p, i_b]
    ret.acc += (
        viscosity_coef
        * ret.mu_i
        * (mass_j / rho_j)
        * v_ij
        / (dist**2 + sph_info.viscosity_eps[None])
        * func_cubic_kernel_derivative(
            d_ij, sph_info.support_radius[None], sph_info.kernel_coef[None], sph_info.kernel_dist_min[None]
        )
    )


@qd.kernel
def kernel_compute_non_pressure_forces(
    sim_state: SimulatorState, sph_state: SPHState, sph_info: SPHInfo, force_fields: qd.template()
):
    """Compute the acceleration of the sorted particles from gravity, surface tension, viscosity and force fields."""
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.acc))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            # The parameters of the particle, read once for all its neighbors
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            ret = qd.Struct(
                acc=sph_info.gravity[i_b],
                surface_tension_i=sph_info.particles.surface_tension[i_o],
                mu_i=sph_info.particles.mu[i_o],
            )
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_non_pressure_forces)
            sph_state.particles_reordered.acc[i_p, i_b] = force_fields.add_acc(
                ret.acc,
                sph_state.particles_reordered.pos[i_p, i_b],
                sph_state.particles_reordered.vel[i_p, i_b],
                sim_state.t[i_b],
                i_p,
            )


@qd.func
def func_task_compute_pressure_forces(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    j_o = sph_state.hash_grid.particles_idx[j_p, i_b]
    rho0_j = sph_info.particles.rho[j_o]
    dp_j = sph_state.particles_reordered.p[j_p, i_b] / sph_state.particles_reordered.rho[j_p, i_b] ** 2

    # The pressure force contribution, symmetric formula
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    ret.acc += (
        -rho0_j
        * sph_info.particle_volume[None]
        * (ret.dp_i + dp_j)
        * func_cubic_kernel_derivative(
            d_ij, sph_info.support_radius[None], sph_info.kernel_coef[None], sph_info.kernel_dist_min[None]
        )
    )


@qd.kernel(fastcache=True)
def kernel_compute_pressure_forces(
    sph_state: SPHState, sph_columns: SPHColumns, sph_info: SPHInfo, sph_config: qd.template()
):
    """
    Add the acceleration of the pressure of the equation of state to the sorted particles.

    This is the pressure model of weakly compressible smoothed particle hydrodynamics (WCSPH).
    """
    # The particles sorted by vertical column, for the height of the free surface that sets the pressure of the walls
    for i_p, i_b in qd.ndrange(*sph_columns.pos.shape, axes=qd.static(eoa(sph_columns.pos))):
        pos = sph_state.particles_reordered.pos[i_p, i_b]
        sph_columns.pos[i_p, i_b] = qd.Vector([pos[0], pos[1], 0.0], dt=gs.qd_float)
    func_hash_sort_particles(sph_columns.pos, sph_state.particles_reordered.is_active, sph_columns.grid)

    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.rho))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            rho0 = sph_info.particles.rho[i_o]
            rho = qd.max(sph_state.particles_reordered.rho[i_p, i_b], rho0)
            sph_state.particles_reordered.rho[i_p, i_b] = rho
            sph_state.particles_reordered.p[i_p, i_b] = sph_info.particles.stiffness[i_o] * (
                qd.pow(rho / rho0, sph_info.particles.exponent[i_o]) - 1.0
            )

    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.acc))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            # The pressure term of the particle, computed once for all its neighbors
            pos = sph_state.particles_reordered.pos[i_p, i_b]
            p_i = sph_state.particles_reordered.p[i_p, i_b]
            dp_i = p_i / sph_state.particles_reordered.rho[i_p, i_b] ** 2
            ret = qd.Struct(acc=qd.Vector.zero(gs.qd_float, 3), dp_i=dp_i)
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_pressure_forces)

            # Wall boundary condition: the ghost particles of the walls take the hydrostatic pressure of their depth
            # under the free surface of the liquid above the particle, and push the particle back through the symmetric
            # pressure force. A pressure interpolated from the liquid around the particle does net work on a liquid at
            # rest, which keeps it oscillating.
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            rho0 = sph_info.particles.rho[i_o]
            gravity = sph_info.gravity[i_b]
            wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(i_p, i_b, sph_state, sph_info, sph_config)
            # The free surface lies half a particle size above the topmost particle of the column of the particle. The
            # slot of the column also holds the columns that the hash grid wraps onto it, which the cell filters out.
            surface_height = pos[2]
            if wall_volume > 0.0:
                cell = func_hash_pos_to_grid(sph_columns.pos[i_p, i_b], sph_columns.grid)
                i_s = func_hash_grid_to_slot(cell, sph_columns.grid)
                slot_start = sph_columns.grid.slot_start[i_s, i_b]
                for i_r in range(slot_start, slot_start + sph_columns.grid.slot_size[i_s, i_b]):
                    j_p = sph_columns.grid.particles_idx[i_r, i_b]
                    cell_j = func_hash_pos_to_grid(sph_columns.pos[j_p, i_b], sph_columns.grid)
                    if cell_j[0] == cell[0] and cell_j[1] == cell[1]:
                        surface_height = qd.max(surface_height, sph_state.particles_reordered.pos[j_p, i_b][2])
            # The vertical depth times the vertical gravity is the depth along gravity times its norm, under a free
            # surface normal to gravity whatever its direction
            depth = surface_height + 0.5 * sph_info.particle_size[None] - pos[2]
            p_wall_center = -rho0 * gravity[2] * depth
            wall_coef = rho0 * dp_i + p_wall_center / rho0
            wall_force = wall_coef * wall_volume_grad + wall_moment @ gravity
            if qd.static(sph_config.has_boundary):
                # The moving boundaries take the opposite of the impulse their ghost particles give the particle
                impulse_scale = sph_info.substep_dt[None] * sph_info.particles.mass[i_o]
                sph_state.boundary.impulse_coef[i_o, i_b] = impulse_scale * wall_coef
                sph_state.boundary.impulse_gravity_coef[i_o, i_b] = impulse_scale
            sph_state.particles_reordered.acc[i_p, i_b] = (
                sph_state.particles_reordered.acc[i_p, i_b] + ret.acc - wall_force
            )


@qd.kernel(fastcache=True)
def kernel_advect_velocity(sph_state: SPHState, sph_info: SPHInfo):
    """Advance the velocity of the sorted particles by their acceleration."""
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.vel))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            sph_state.particles_reordered.vel[i_p, i_b] = (
                sph_state.particles_reordered.vel[i_p, i_b]
                + sph_info.substep_dt[None] * sph_state.particles_reordered.acc[i_p, i_b]
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
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    grad_j = -sph_info.particle_volume[None] * func_cubic_kernel_derivative(
        d_ij, sph_info.support_radius[None], sph_info.kernel_coef[None], sph_info.kernel_dist_min[None]
    )
    ret.grad_sqr_sum = ret.grad_sqr_sum + grad_j.norm_sqr()
    ret.grad_p = ret.grad_p - grad_j


@qd.kernel(fastcache=True)
def kernel_compute_dfsph_factor(sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template()):
    """
    Compute the factor of the divergence and density solves of the sorted particles.

    These are the solves of divergence-free smoothed particle hydrodynamics (DFSPH). Every environment takes part in the
    divergence solve that follows (see kernel_update_convergence).
    """
    for i_b in range(sph_state.particles_reordered.pos.shape[1]):
        sph_state.envs_solve.is_solving[i_b] = True
        sph_state.envs_solve.err[i_b] = 0.0
        sph_state.envs_solve.n_active[i_b] = 0

    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.dfsph_factor))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            qd.atomic_add(sph_state.envs_solve.n_active[i_b], 1)

            ret = qd.Struct(grad_p=qd.Vector.zero(gs.qd_float, 3), grad_sqr_sum=gs.qd_float(0.0))
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_dfsph_factor)

            # The walls, at rest, change the density of the particle by its own velocity, which they change with its own
            # stiffness and with the stiffness they take from it among the liquid around it (see
            # kernel_solver_iteration), weighted by its share of the kernel sum of the liquid.
            wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(i_p, i_b, sph_state, sph_info, sph_config)
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            rho_rel = sph_state.particles_reordered.rho[i_p, i_b] / sph_info.particles.rho[i_o]
            w_ii = func_cubic_kernel(0.0, sph_info.support_radius[None], sph_info.kernel_coef[None])
            # The relative density of the liquid alone includes the particle itself, so it never vanishes
            self_weight = sph_info.particle_volume[None] * w_ii / (rho_rel - wall_volume)
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
    v_ij = sph_state.particles_reordered.vel[i_p, i_b] - sph_state.particles_reordered.vel[j_p, i_b]
    x_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    ret.drho += sph_info.particle_volume[None] * v_ij.dot(
        func_cubic_kernel_derivative(
            x_ij, sph_info.support_radius[None], sph_info.kernel_coef[None], sph_info.kernel_dist_min[None]
        )
    )
    ret.n_neighbors += 1


@qd.kernel(fastcache=True)
def kernel_compute_density_rate(
    sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template(), solve_density: qd.template()
):
    """
    Compute what a solve of divergence-free smoothed particle hydrodynamics (DFSPH) cancels for the sorted particles.

    The divergence solve cancels the positive rate of change of the density, and the density solve the predicted
    relative density at the end of the substep. Each particle also gets the stiffness that the next Jacobi iteration
    applies to cancel it, and adds its error to that of its environment, summed for kernel_update_convergence.
    """
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.drho))
    ):
        if sph_state.envs_solve.is_solving[i_b] and sph_state.particles_reordered.is_active[i_p, i_b]:
            # The walls are at rest, while the moving boundaries carry their own velocity
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(i_p, i_b, sph_state, sph_info, sph_config)
            drho_wall = sph_state.particles_reordered.vel[i_p, i_b].dot(wall_volume_grad)
            if qd.static(sph_config.has_boundary):
                drho_wall = drho_wall - sph_state.boundary.vel_flux[i_o, i_b]
            ret = qd.Struct(drho=drho_wall, n_neighbors=0)
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_density_rate)
            rho0 = sph_info.particles.rho[i_o]
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
                qd.atomic_add(sph_state.envs_solve.err[i_b], rho0 * qd.abs(err_rel))
            else:
                # Only a positive divergence is corrected, and none where the neighbors of the particle are too few, the
                # walls standing for neighbors
                drho = qd.max(ret.drho, 0.0)
                if ret.n_neighbors < sph_info.n_neighbors_min[None] and wall_volume <= 0.0:
                    drho = 0.0
                sph_state.particles_reordered.drho[i_p, i_b] = drho
                sph_state.particles_k_reordered[i_p, i_b] = drho * sph_state.particles_reordered.dfsph_factor[i_p, i_b]
                qd.atomic_add(sph_state.envs_solve.err[i_b], rho0 * drho)


@qd.kernel(fastcache=True)
def kernel_predict_velocity(sph_state: SPHState, sph_info: SPHInfo):
    """
    Advance the velocity of the sorted particles by their non-pressure acceleration, ahead of the density solve.

    Every environment takes part in the density solve that follows (see kernel_update_convergence). The warm start of
    that solve applies part of the pressure carried over from the previous substep (see warm_start_scale in SPHInfo), to
    the particles compressed at the start of the substep.
    """
    for i_b in range(sph_state.particles_reordered.pos.shape[1]):
        sph_state.envs_solve.is_solving[i_b] = True
        sph_state.envs_solve.err[i_b] = 0.0

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
            p = gs.qd_float(0.0)
            if sph_state.particles_reordered.rho[i_p, i_b] > rho0:
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
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    if qd.abs(k_sum) > sph_info.stiffness_eps[None]:
        grad_p_j = -sph_info.particle_volume[None] * func_cubic_kernel_derivative(
            d_ij, sph_info.support_radius[None], sph_info.kernel_coef[None], sph_info.kernel_dist_min[None]
        )
        ret.dv = ret.dv - ret.dv_scale * k_sum * grad_p_j

    # The stiffness and the position of the neighbors, weighted for the stiffness of the walls (see
    # kernel_solver_iteration)
    w_ij = func_cubic_kernel(d_ij.norm(), sph_info.support_radius[None], sph_info.kernel_coef[None])
    ret.weight_sum = ret.weight_sum + w_ij
    ret.k_sum_w = ret.k_sum_w + w_ij * k_j
    ret.pos_sum = ret.pos_sum + w_ij * sph_state.particles_reordered.pos[j_p, i_b]


@qd.kernel(fastcache=True)
def kernel_solver_iteration(
    sph_state: SPHState, sph_info: SPHInfo, sph_config: qd.template(), solve_density: qd.template(), warm_start: bool
):
    """
    Run one Jacobi iteration of the divergence or the density solve of divergence-free smoothed particle hydrodynamics
    (DFSPH) on the velocity of the sorted particles.

    The velocity of every particle changes by the stiffness of its own and of its neighbors. The divergence solve
    corrects velocities directly, while the density solve accelerates over the substep.
    """
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.vel))
    ):
        if sph_state.envs_solve.is_solving[i_b] and sph_state.particles_reordered.is_active[i_p, i_b]:
            dv_scale = gs.qd_float(1.0)
            if qd.static(solve_density):
                dv_scale = sph_info.substep_dt[None]
            k_i = sph_state.particles_k_reordered[i_p, i_b]
            pos = sph_state.particles_reordered.pos[i_p, i_b]
            w_ii = func_cubic_kernel(0.0, sph_info.support_radius[None], sph_info.kernel_coef[None])
            ret = qd.Struct(
                dv=qd.Vector.zero(gs.qd_float, 3),
                k_i=k_i,
                dv_scale=dv_scale,
                weight_sum=w_ii,
                k_sum_w=w_ii * k_i,
                pos_sum=w_ii * pos,
            )
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_solver_iteration)

            # The ghost particles of the walls take the stiffness of the liquid around the particle (see
            # kernel_compute_pressure_forces). The warm start applies the pressure itself, which adds the hydrostatic
            # pressure of the body force between them, while the iterations correct it.
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            gravity = sph_info.gravity[i_b]
            wall_volume, wall_volume_grad, wall_moment = func_boundary_volume(i_p, i_b, sph_state, sph_info, sph_config)
            k_wall_center = ret.k_sum_w / ret.weight_sum
            wall_dv = qd.Vector.zero(gs.qd_float, 3)
            if warm_start:
                k_wall_center = k_wall_center - gravity.dot(pos - ret.pos_sum / ret.weight_sum)
                wall_dv = -wall_moment @ gravity
            wall_dv = wall_dv + (k_i + k_wall_center) * wall_volume_grad
            if qd.static(sph_config.has_boundary):
                # The moving boundaries take the opposite of the impulse their ghost particles give the particle
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
                # pressure field that enforces incompressibility, the quantity the equation of state gives under WCSPH.
                # The warm start applies the pressure as it stands (see kernel_predict_velocity).
                if not warm_start:
                    rho0 = sph_info.particles.rho[i_o]
                    sph_state.particles_reordered.p[i_p, i_b] = sph_state.particles_reordered.p[i_p, i_b] - k_i * rho0


@qd.kernel(fastcache=True)
def kernel_update_convergence(
    sph_state: SPHState, sph_info: SPHInfo, solve_density: qd.template(), can_stop: bool
) -> int:
    """
    Return the number of environments whose divergence-free smoothed particle hydrodynamics (DFSPH) solve iterates on.

    Each environment stops once the average density error of its own particles meets the tolerance, so that its result
    is the same whatever the other environments simulate alongside it, and only once `can_stop` allows it. The error
    summed by the density rate kernels is reset once read, for the next call.
    """
    n_envs_solving = 0
    for i_b in range(sph_state.envs_solve.err.shape[0]):
        if sph_state.envs_solve.is_solving[i_b]:
            err_tol = sph_info.divergence_err_tol[None]
            if qd.static(solve_density):
                err_tol = sph_info.density_err_tol[None]
            if can_stop and sph_state.envs_solve.err[i_b] <= err_tol * sph_state.envs_solve.n_active[i_b]:
                sph_state.envs_solve.is_solving[i_b] = False
            else:
                qd.atomic_add(n_envs_solving, 1)
            sph_state.envs_solve.err[i_b] = 0.0
    return n_envs_solving


# ------------------------------------------------------------------------------------
# --------------------------------------- io -----------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_add_particles(
    particle_start: int,
    is_active: bool,
    rho: float,
    stiffness: float,
    exponent: float,
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
        sph_info.particles.stiffness[i_p] = stiffness
        sph_info.particles.exponent[i_p] = exponent
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


@qd.kernel(fastcache=True)
def kernel_sum_boundary_ghosts(boundary_ghosts: SPHBoundaryGhosts, sph_state: SPHState, sph_info: SPHInfo):
    """Sum up the ghost particles of a moving boundary around every particle (see SPHBoundaryParticles)."""
    h, k = sph_info.support_radius[None], sph_info.kernel_coef[None]
    for i_p, i_b in qd.ndrange(*sph_state.particles.pos.shape, axes=qd.static(eoa(sph_state.boundary.volume))):
        volume = gs.qd_float(0.0)
        volume_grad = qd.Vector.zero(gs.qd_float, 3)
        moment = qd.Matrix.zero(gs.qd_float, 3, 3)
        vel_flux = gs.qd_float(0.0)
        if sph_state.particles.is_active[i_p, i_b]:
            pos = sph_state.particles.pos[i_p, i_b]
            base = func_hash_wrap_cell(
                func_hash_pos_to_grid(pos, boundary_ghosts.hash_grid) - 1, boundary_ghosts.hash_grid
            )
            n_cells = func_hash_n_cells(3, boundary_ghosts.hash_grid)
            for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
                i_s = func_hash_cell_to_slot(base + offset, boundary_ghosts.hash_grid)
                slot_start = boundary_ghosts.hash_grid.slot_start[i_s, i_b]
                for i_r in range(slot_start, slot_start + boundary_ghosts.hash_grid.slot_size[i_s, i_b]):
                    i_g = boundary_ghosts.hash_grid.particles_idx[i_r, i_b]
                    r = boundary_ghosts.particles.pos[i_g, i_b] - pos
                    if r.norm() < h:
                        ghost_volume = boundary_ghosts.particles.volume[i_g, i_b]
                        grad = -ghost_volume * func_cubic_kernel_derivative(r, h, k, sph_info.kernel_dist_min[None])
                        volume = volume + ghost_volume * func_cubic_kernel(r.norm(), h, k)
                        volume_grad = volume_grad + grad
                        moment = moment + grad.outer_product(r)
                        vel_flux = vel_flux + boundary_ghosts.particles.vel[i_g, i_b].dot(grad)
        sph_state.boundary.volume[i_p, i_b] = volume
        sph_state.boundary.volume_grad[i_p, i_b] = volume_grad
        sph_state.boundary.moment[i_p, i_b] = moment
        sph_state.boundary.vel_flux[i_p, i_b] = vel_flux
        sph_state.boundary.impulse_coef[i_p, i_b] = 0.0
        sph_state.boundary.impulse_gravity_coef[i_p, i_b] = 0.0
