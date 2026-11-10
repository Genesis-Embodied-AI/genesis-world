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
from genesis.utils.data import ENVS, ConfigItem, Data, DataItem, DataKind, eoa, field, iter_data
from genesis.utils.geom import (
    PARTICLES,
    SpatialHashGrid,
    func_hash_grid_to_slot,
    func_hash_pos_to_grid,
    func_hash_sort_particles,
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
    and `gamma` the surface tension.
    """

    rho: qd.Tensor = field(float)
    mass: qd.Tensor = field(float)
    stiffness: qd.Tensor = field(float)
    exponent: qd.Tensor = field(float)
    mu: qd.Tensor = field(float)
    gamma: qd.Tensor = field(float)


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


class SPHState(Data, kind=DataKind.STATE):
    """
    Runtime state of the smoothed particle hydrodynamics (SPH) solver.

    The substep runs on the copy of the particles sorted by hash grid cell, `particles_reordered`, which the particles
    take back at its end. `particles_k_reordered` is the stiffness that the next Jacobi iteration of divergence-free
    smoothed particle hydrodynamics (DFSPH) applies, read by the neighbors of each sorted particle.
    """

    particles: SPHParticles
    particles_reordered: SPHParticles = field(kind=DataKind.SCRATCH)
    particles_k_reordered: qd.Tensor = field(float, shape=(PARTICLES, ENVS), kind=DataKind.SCRATCH, labels=DFSPH)
    envs_solve: SPHEnvsSolve = field(labels=DFSPH)
    hash_grid: SpatialHashGrid


class SPHInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the smoothed particle hydrodynamics (SPH) solver.

    `particle_volume` is the volume a particle stands for, `support_radius` the radius of the smoothing kernels,
    `kernel_coef` the normalization of the cubic one, `kernel_dist_min` the distance below which the gradient of the
    cubic kernel is zero, and `viscosity_eps` the squared distance the viscosity of a pair of particles is regularized
    by. The tolerances of the divergence-free smoothed particle hydrodynamics (DFSPH) solves are the average density
    error that ends the density solve, in kg/m^3, the average density rate that ends the divergence solve, in kg/m^3/s,
    the stiffness below which a pair of particles exchanges no velocity, the factor of the Jacobi updates, the sum of
    squared kernel gradients below which a particle takes no pressure, and the number of neighbors below which the
    density rate of a particle is left out of the divergence solve.
    """

    particles: SPHParticlesInfo
    boundary: CubeBoundary
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float, shape=())
    particle_size: qd.Tensor = field(float, shape=())
    particle_volume: qd.Tensor = field(float, shape=())
    support_radius: qd.Tensor = field(float, shape=())
    kernel_coef: qd.Tensor = field(float, shape=())
    kernel_dist_min: qd.Tensor = field(float, shape=())
    viscosity_eps: qd.Tensor = field(float, shape=())
    density_err_tol: qd.Tensor = field(float, shape=(), labels=DFSPH)
    divergence_err_tol: qd.Tensor = field(float, shape=(), labels=DFSPH)
    stiffness_eps: qd.Tensor = field(float, shape=(), labels=DFSPH)
    jacobi_relaxation: qd.Tensor = field(float, shape=(), labels=DFSPH)
    factor_eps: qd.Tensor = field(float, shape=(), labels=DFSPH)
    n_neighbors_min: qd.Tensor = field(int, shape=(), labels=DFSPH)


class SPHSolver(ParticleSolver):
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

        # 0.8 is an empirical value
        self._particle_volume = 0.8 * self._particle_size**3

        self._hash_grid_cell_size = options.hash_grid_cell_size
        self._hash_grid_res = options._hash_grid_res

        # Coefficient for computing stable timestep
        self._stable_dt_coef = 1.0

        self.sph_state: SPHState | None = None
        self.sph_info: SPHInfo | None = None

    def build(self) -> None:
        super().build()

        is_dfsph = self._pressure_solver == "DFSPH"
        unused_labels = dict(dfsph=not is_dfsph) if self.is_active else True
        sizes = dict(envs=self._B, particles=self._n_particles)
        hash_grid_values = dict(cell_size=self._hash_grid_cell_size, grid_res=self._hash_grid_res)
        self.sph_state = SPHState(
            sizes=dict(sizes, hash_slots=np.prod(self._hash_grid_res)),
            values=dict(hash_grid=hash_grid_values),
            env_first=False,
            unused=unused_labels,
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
            material.gamma,
            pos,
            self.sph_state,
            self.sph_info,
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _solve_incompressibility(self, solve_density: bool) -> None:
        n_iters_max = self._df_max_den_iters if solve_density else self._df_max_div_iters

        # The convergence is checked before the first iteration: an environment that the warm start of the density
        # solve, or a divergence-free velocity field, already brings within tolerance does not iterate at all.
        if solve_density:
            kernel_solver_iteration(self.sph_state, self.sph_info, solve_density, warm_start=True)
        kernel_compute_density_rate(self.sph_state, self.sph_info, solve_density)
        n_envs_solving = kernel_update_convergence(self.sph_state, self.sph_info, solve_density)
        n_iters = 0
        while n_envs_solving > 0 and n_iters < n_iters_max:
            kernel_solver_iteration(self.sph_state, self.sph_info, solve_density, warm_start=False)
            kernel_compute_density_rate(self.sph_state, self.sph_info, solve_density)
            n_envs_solving = kernel_update_convergence(self.sph_state, self.sph_info, solve_density)
            n_iters += 1

        gs.logger.debug(f"DFSPH - {'density' if solve_density else 'divergence'} solve iterations: {n_iters}")

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        kernel_reorder_particles(self.sph_state)
        kernel_compute_rho(self.sph_state, self.sph_info)
        if self._pressure_solver == "WCSPH":
            kernel_compute_non_pressure_forces(self._sim.sim_state, self.sph_state, self.sph_info, self._force_fields)
            kernel_compute_pressure_forces(self.sph_state, self.sph_info)
            kernel_advect_velocity(self.sph_state, self.sph_info)
        else:
            kernel_compute_dfsph_factor(self.sph_state, self.sph_info)
            self._solve_incompressibility(solve_density=False)
            kernel_compute_non_pressure_forces(self._sim.sim_state, self.sph_state, self.sph_info, self._force_fields)
            kernel_predict_velocity(self.sph_state, self.sph_info)
            self._solve_incompressibility(solve_density=True)

    def substep_post_coupling(self, f):
        if self.is_active:
            kernel_advect_position(self.sph_state, self.sph_info)
            kernel_copy_from_reordered(self.sph_state)

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        yield from iter_data(self, "sph_info")
        yield from iter_data(self, "sph_state")

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
def func_for_all_neighbors(
    i_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo, task: qd.template()
):
    """
    Run a task on every sorted particle within the support radius of a sorted particle, the particle itself excluded.

    `task` takes the indices of both particles, the environment, `ret`, the state and the info, in this order, and
    accumulates its result in `ret`.
    """
    pos_i = sph_state.particles_reordered.pos[i_p, i_b]
    base = func_hash_pos_to_grid(pos_i, sph_state.hash_grid)
    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
        i_s = func_hash_grid_to_slot(base + offset, sph_state.hash_grid)
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
    ret += sph_info.particle_volume[None] * func_cubic_kernel(
        d_ij.norm(), sph_info.support_radius[None], sph_info.kernel_coef[None]
    )


@qd.kernel(fastcache=True)
def kernel_compute_rho(sph_state: SPHState, sph_info: SPHInfo):
    """Compute the density of the sorted particles from their neighbours."""
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.rho))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            # Base density using the kernel at distance 0
            h, k = sph_info.support_radius[None], sph_info.kernel_coef[None]
            rho = sph_info.particle_volume[None] * func_cubic_kernel(0.0, h, k)
            den = gs.qd_float(0.0)
            func_for_all_neighbors(i_p, i_b, den, sph_state, sph_info, func_task_compute_rho)
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
            sph_state.particles_reordered.rho[i_p, i_b] = (rho + den) * sph_info.particles.rho[i_o]


@qd.func
def func_task_compute_non_pressure_forces(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    dist = d_ij.norm()
    j_o = sph_state.hash_grid.particles_idx[j_p, i_b]
    mass_j = sph_info.particles.mass[j_o]

    # Surface tension, the distance clamped to the particle size, which bounds the force of overlapping particles
    particle_size = sph_info.particle_size[None]
    effective_dist = dist if dist > particle_size else particle_size
    ret.acc -= (
        ret.gamma_i
        / ret.mass_i
        * mass_j
        * d_ij
        * func_cubic_kernel(effective_dist, sph_info.support_radius[None], sph_info.kernel_coef[None])
    )

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
                gamma_i=sph_info.particles.gamma[i_o],
                mass_i=sph_info.particles.mass[i_o],
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
def kernel_compute_pressure_forces(sph_state: SPHState, sph_info: SPHInfo):
    """
    Add the acceleration of the pressure of the equation of state to the sorted particles.

    This is the pressure model of weakly compressible smoothed particle hydrodynamics (WCSPH).
    """
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
            dp_i = sph_state.particles_reordered.p[i_p, i_b] / sph_state.particles_reordered.rho[i_p, i_b] ** 2
            ret = qd.Struct(acc=qd.Vector.zero(gs.qd_float, 3), dp_i=dp_i)
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_pressure_forces)
            sph_state.particles_reordered.acc[i_p, i_b] = sph_state.particles_reordered.acc[i_p, i_b] + ret.acc


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
def kernel_compute_dfsph_factor(sph_state: SPHState, sph_info: SPHInfo):
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
            sum_grad_p_k = ret.grad_sqr_sum + ret.grad_p.norm_sqr()

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
def kernel_compute_density_rate(sph_state: SPHState, sph_info: SPHInfo, solve_density: qd.template()):
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
            ret = qd.Struct(drho=gs.qd_float(0.0), n_neighbors=0)
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_density_rate)
            i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
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
                # Only a positive divergence is corrected, and none where the neighbors of the particle are too few
                drho = qd.max(ret.drho, 0.0)
                if ret.n_neighbors < sph_info.n_neighbors_min[None]:
                    drho = 0.0
                sph_state.particles_reordered.drho[i_p, i_b] = drho
                sph_state.particles_k_reordered[i_p, i_b] = drho * sph_state.particles_reordered.dfsph_factor[i_p, i_b]
                qd.atomic_add(sph_state.envs_solve.err[i_b], rho0 * drho)


@qd.kernel(fastcache=True)
def kernel_predict_velocity(sph_state: SPHState, sph_info: SPHInfo):
    """
    Advance the velocity of the sorted particles by their non-pressure acceleration, ahead of the density solve.

    Every environment takes part in the density solve that follows (see kernel_update_convergence). The warm start of
    that solve applies the stiffness carried over from the previous substep in the pressure, to the particles compressed
    at the start of the substep.
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
            k_i = gs.qd_float(0.0)
            if sph_state.particles_reordered.rho[i_p, i_b] > rho0:
                k_i = -sph_state.particles_reordered.p[i_p, i_b] / rho0
            sph_state.particles_k_reordered[i_p, i_b] = k_i


@qd.func
def func_task_solver_iteration(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    # The stiffness already holds the inverse of the density, which is the same for every particle of a liquid
    k_sum = ret.k_i + sph_state.particles_k_reordered[j_p, i_b]
    if qd.abs(k_sum) > sph_info.stiffness_eps[None]:
        d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
        grad_p_j = -sph_info.particle_volume[None] * func_cubic_kernel_derivative(
            d_ij, sph_info.support_radius[None], sph_info.kernel_coef[None], sph_info.kernel_dist_min[None]
        )
        ret.dv = ret.dv - ret.dv_scale * k_sum * grad_p_j


@qd.kernel(fastcache=True)
def kernel_solver_iteration(sph_state: SPHState, sph_info: SPHInfo, solve_density: qd.template(), warm_start: bool):
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
            ret = qd.Struct(dv=qd.Vector.zero(gs.qd_float, 3), k_i=k_i, dv_scale=dv_scale)
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_solver_iteration)
            sph_state.particles_reordered.vel[i_p, i_b] = sph_state.particles_reordered.vel[i_p, i_b] + ret.dv

            if qd.static(solve_density):
                # The stiffness k_i is the pressure this pass applies to cancel the density error, up to the -1/rho0
                # scaling carried by dfsph_factor, so summing -k_i * rho0 over the density solve yields the implicit
                # pressure field that enforces incompressibility, the quantity the equation of state gives under WCSPH.
                # The warm start applies the pressure as it stands and drops the pressure of the particles it leaves
                # out, which the neighbors reading it in the meantime leave out the same way.
                i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
                rho0 = sph_info.particles.rho[i_o]
                if warm_start:
                    if sph_state.particles_reordered.rho[i_p, i_b] <= rho0:
                        sph_state.particles_reordered.p[i_p, i_b] = 0.0
                else:
                    sph_state.particles_reordered.p[i_p, i_b] = sph_state.particles_reordered.p[i_p, i_b] - k_i * rho0


@qd.kernel(fastcache=True)
def kernel_update_convergence(sph_state: SPHState, sph_info: SPHInfo, solve_density: qd.template()) -> int:
    """
    Return the number of environments whose divergence-free smoothed particle hydrodynamics (DFSPH) solve iterates on.

    Each environment stops once the average density error of its own particles meets the tolerance, so that its result
    is the same whatever the other environments simulate alongside it. The error summed by the density rate kernels is
    reset once read, for the next call.
    """
    n_envs_solving = 0
    for i_b in range(sph_state.envs_solve.err.shape[0]):
        if sph_state.envs_solve.is_solving[i_b]:
            err_tol = sph_info.divergence_err_tol[None]
            if qd.static(solve_density):
                err_tol = sph_info.density_err_tol[None]
            if sph_state.envs_solve.err[i_b] <= err_tol * sph_state.envs_solve.n_active[i_b]:
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
    gamma: float,
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
        sph_info.particles.gamma[i_p] = gamma
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
