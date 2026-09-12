from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine.states.solvers import SPHSolverState

from .base_solver import GravityMixin, Solver, TimeBasedMixin

if TYPE_CHECKING:
    from genesis.engine.entities import SPHEntity


@qd.data_oriented
class SPHSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = SPH.Base
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius
        self._pressure_solver = options.pressure_solver

        # DFSPH parameters
        self._df_max_error_div = options.max_divergence_error
        self._df_max_error_den = options.max_density_error_percent
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations
        self._df_eps = 1e-5

        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        self._particle_volume = 0.8 * self._particle_size**3  # 0.8 is an empirical value

        # spatial hasher
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )
        # boundary
        self.setup_boundary()

        # Coefficient for computing stable timestep
        self._stable_dt_coef = 1.0

    def setup_boundary(self):
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
            # restitution=0.5,
        )

    def init_particle_fields(self):
        # dynamic particle state
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            acc=gs.qd_vec3,  # acceleration
            rho=gs.qd_float,  # density
            p=gs.qd_float,  # pressure
            dfsph_factor=gs.qd_float,  # DFSPH use: Factor for Divergence and density solver
            drho=gs.qd_float,  # density deritivate
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
        )

        # static particle info
        struct_particle_info = qd.types.struct(
            rho=gs.qd_float,  # rest density
            mass=gs.qd_float,  # mass
            stiffness=gs.qd_float,
            exponent=gs.qd_float,
            mu=gs.qd_float,  # viscosity
            gamma=gs.qd_float,  # surface tension
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            active=gs.qd_bool,
        )

        # construct fields
        self.particles = struct_particle_state.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_info = struct_particle_info.field(
            shape=(self._n_particles,), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_reordered = struct_particle_state.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_info_reordered = struct_particle_info.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles

        self._coupler = self.sim._coupler

        if self.is_active:
            self.sh.build(self._B)
            self.init_particle_fields()
            self.init_ckpt()

            for entity in self.entities:
                entity._add_to_solver()

            # Compute stable timestep for the given particle properties (mu) and support radius and print warning
            # if the current timestep is larger than the stable timestep. This formula is derived from
            # [_task_compute_non_pressure_forces] function, which computes the acceleration due to non-pressure forces.
            stable_dt = (
                self._stable_dt_coef * (0.01 * self._support_radius**2) / self.particles_info.mu.to_numpy().max()
            )
            if self.substep_dt > stable_dt:
                gs.logger.warning(
                    f"Current timestep ({self.substep_dt:.4g}) is larger than the stable timestep "
                    f"({stable_dt:.4g}) for SPH solver. "
                    "Please consider reducing the timestep, liquid viscosity (mu), or increasing the particle size."
                )

            # TODO: Support per-particle density
            self._density0 = self.particles_info[0].rho

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity(as_field=True)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

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

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )

        # copy to reordered
        self.particles_ng_reordered.active.fill(False)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]
                self.particles_info_reordered[reordered_idx, i_b] = self.particles_info[i_p]
                self.particles_ng_reordered[reordered_idx, i_b].active = self.particles_ng[i_p, i_b].active

        if qd.static(self._coupler._rigid_sph):
            for i_p, i_g, i_b in qd.ndrange(self._n_particles, self._coupler.rigid_solver.n_geoms, self._B):
                if self.particles_ng[i_p, i_b].active:
                    self._coupler.sph_rigid_normal_reordered[self.particles_ng[i_p, i_b].reordered_idx, i_g, i_b] = (
                        self._coupler.sph_rigid_normal[i_p, i_g, i_b]
                    )

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                # only need to copy back dynamic state, i.e. self.particles
                self.particles[i_p, i_b] = self.particles_reordered[self.particles_ng[i_p, i_b].reordered_idx, i_b]

        if qd.static(self._coupler._rigid_sph):
            for i_p, i_g, i_b in qd.ndrange(self._n_particles, self._coupler.rigid_solver.n_geoms, self._B):
                if self.particles_ng[i_p, i_b].active:
                    self._coupler.sph_rigid_normal[i_p, i_g, i_b] = self._coupler.sph_rigid_normal_reordered[
                        self.particles_ng[i_p, i_b].reordered_idx, i_g, i_b
                    ]

    @qd.func
    def _task_compute_rho(self, i, j, ret: qd.template(), i_b):
        ret += self._particle_volume * self.cubic_kernel(
            (self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos).norm()
        )

    @qd.kernel
    def _kernel_compute_rho(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                # Base density using the kernel at distance 0
                self.particles_reordered[i_p, i_b].rho = self._particle_volume * self.cubic_kernel(0.0)

                den = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, den, self._task_compute_rho, i_b
                )
                self.particles_reordered[i_p, i_b].rho += den

                self.particles_reordered[i_p, i_b].rho *= self.particles_info_reordered[i_p, i_b].rho

    @qd.func
    def _task_compute_non_pressure_forces(self, i, j, ret: qd.template(), i_b: qd.i32):
        d_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        dist = d_ij.norm()

        gamma_i = self.particles_info_reordered[i, i_b].gamma
        mass_i = self.particles_info_reordered[i, i_b].mass
        mu_i = self.particles_info_reordered[i, i_b].mu

        mass_j = self.particles_info_reordered[j, i_b].mass

        # -----------------------------
        # Surface Tension term
        # -----------------------------
        # If distance is bigger than _particle_size, use d_ij.norm() directly; otherwise clamp.
        effective_dist = dist if dist > self._particle_size else self._particle_size

        ret -= gamma_i / mass_i * mass_j * d_ij * self.cubic_kernel(effective_dist)

        # -----------------------------
        # Viscosity Force
        # -----------------------------
        v_ij = (self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel).dot(d_ij)

        # Some constant factor used in the viscosity formula
        d = 2 * (3 + 2)

        # The density of the neighbor j in batch b
        rho_j = self.particles_reordered[j, i_b].rho

        f_v = (
            d
            * mu_i
            * (mass_j / rho_j)
            * v_ij
            / (dist**2 + 0.01 * self._support_radius**2)
            * self.cubic_kernel_derivative(d_ij)
        )
        ret += f_v

    @qd.kernel
    def _kernel_compute_non_pressure_forces(self, f: qd.i32, t: qd.f32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                acc = self._gravity[i_b]
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    acc,
                    self._task_compute_non_pressure_forces,
                    i_b,
                )

                # external force fields
                for i_ff in qd.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(
                        self.particles_reordered[i_p, i_b].pos, self.particles_reordered[i_p, i_b].vel, t, i_p
                    )
                self.particles_reordered[i_p, i_b].acc = acc

    @qd.func
    def _task_compute_pressure_forces(self, i, j, ret: qd.template(), i_b):
        dp_i = self.particles_reordered[i, i_b].p / self.particles_reordered[i, i_b].rho ** 2
        rho_j = (
            self.particles_reordered[j, i_b].rho
            * self.particles_info_reordered[j, i_b].rho
            / self.particles_info_reordered[j, i_b].rho
        )
        dp_j = self.particles_reordered[j, i_b].p / rho_j**2

        # Compute the pressure force contribution, Symmetric Formula
        ret += (
            -self.particles_info_reordered[j, i_b].rho
            * self._particle_volume
            * (dp_i + dp_j)
            * self.cubic_kernel_derivative(self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos)
        )

    @qd.kernel
    def _kernel_compute_pressure_forces(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                rho0 = self.particles_info_reordered[i_p, i_b].rho
                stiff = self.particles_info_reordered[i_p, i_b].stiffness
                expnt = self.particles_info_reordered[i_p, i_b].exponent

                self.particles_reordered[i_p, i_b].rho = qd.max(self.particles_reordered[i_p, i_b].rho, rho0)

                self.particles_reordered[i_p, i_b].p = stiff * (
                    qd.pow(self.particles_reordered[i_p, i_b].rho / rho0, expnt) - 1.0
                )

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                acc = qd.Vector.zero(gs.qd_float, 3)

                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,  # shape [n_particles, B, 3] or similar
                    self._support_radius,
                    acc,
                    self._task_compute_pressure_forces,
                    i_b,
                )
                self.particles_reordered[i_p, i_b].acc += acc

    @qd.kernel
    def _kernel_advect_velocity(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].acc

    @qd.kernel
    def _kernel_advect_position(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                pos = self.particles_reordered[i_p, i_b].pos
                new_vel = self.particles_reordered[i_p, i_b].vel

                # advect
                new_pos = pos + self._substep_dt * new_vel

                # impose boundary (pass b if needed)
                corrected_pos, corrected_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                # update
                self.particles_reordered[i_p, i_b].vel = corrected_vel
                self.particles_reordered[i_p, i_b].pos = corrected_pos

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------
    @qd.func
    def _task_compute_DFSPH_factor(self, i, j, ret: qd.template(), i_b):
        # Fluid neighbors
        grad_j = -self._particle_volume * self.cubic_kernel_derivative(
            self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        )
        ret[3] += grad_j.norm_sqr()  # sum_grad_p_k
        for ii in qd.static(range(3)):  # grad_p_i
            ret[ii] -= grad_j[ii]

    @qd.kernel
    def _kernel_compute_DFSPH_factor(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                sum_grad_p_k = gs.qd_float(0.0)
                grad_p_i = qd.Vector.zero(gs.qd_float, 3)

                # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
                ret = qd.Vector.zero(gs.qd_float, 4)

                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, ret, self._task_compute_DFSPH_factor, i_b
                )

                sum_grad_p_k = ret[3]
                for ii in qd.static(range(3)):
                    grad_p_i[ii] = ret[ii]
                sum_grad_p_k += grad_p_i.norm_sqr()

                # Compute pressure stiffness denominator
                factor = gs.qd_float(0.0)
                if sum_grad_p_k > 1e-6:
                    factor = -1.0 / sum_grad_p_k
                else:
                    factor = 0.0
                self.particles_reordered[i_p, i_b].dfsph_factor = factor

                # Reset the implicit pressure accumulated by the density solve (see
                # _kernel_density_solve_iteration).
                self.particles_reordered[i_p, i_b].p = 0.0

    @qd.func
    def _task_compute_density_time_derivative(self, i, j, ret: qd.template(), i_b):
        v_i = self.particles_reordered[i, i_b].vel
        v_j = self.particles_reordered[j, i_b].vel

        x_i = self.particles_reordered[i, i_b].pos
        x_j = self.particles_reordered[j, i_b].pos

        # Fluid neighbors
        ret.drho += self._particle_volume * (v_i - v_j).dot(self.cubic_kernel_derivative(x_i - x_j))
        ret.num_neighbors += 1

    @qd.kernel
    def _kernel_compute_density_time_derivative(self):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                ret = qd.Struct(drho=0.0, num_neighbors=0)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_density_time_derivative,
                    i_b,
                )

                # only correct positive divergence
                drho = qd.max(ret.drho, 0.0)
                num_neighbors = ret.num_neighbors

                # Do not perform divergence solve when paritlce deficiency happens
                if num_neighbors < 20:
                    drho = 0.0

                self.particles_reordered[i_p, i_b].drho = drho

    @qd.func
    def _task_divergence_solver_iteration(self, i, j, ret: qd.template(), i_b):
        # Fluid neighbors
        b_j = self.particles_reordered[j, i_b].drho
        k_j = b_j * self.particles_reordered[j, i_b].dfsph_factor
        k_sum = (
            self._density0 / self._density0 * ret.k_i + k_j
        )  # TODO: make the neighbor density different for multiphase fluid
        if qd.abs(k_sum) > self._df_eps:
            grad_p_j = -self._particle_volume * self.cubic_kernel_derivative(
                self.particles_reordered.pos[i, i_b] - self.particles_reordered.pos[j, i_b]
            )
            ret.dv -= (
                k_sum * grad_p_j
            )  # ki, kj already contain inverse density, i.e., density canceled if not mutiphase flow

    @qd.kernel
    def _kernel_divergence_solver_iteration(self):
        # Perform Jacobi iteration
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                # evaluate rhs
                b_i = self.particles_reordered[i_p, i_b].drho
                k_i = b_i * self.particles_reordered[i_p, i_b].dfsph_factor
                ret = qd.Struct(dv=qd.Vector.zero(gs.qd_float, 3), k_i=k_i)
                # TODO: if warm start
                # get_kappa_V += k_i
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_divergence_solver_iteration,
                    i_b,
                )
                self.particles_reordered.vel[i_p, i_b] = self.particles_reordered.vel[i_p, i_b] + ret.dv

    @qd.kernel
    def _kernel_compute_density_error(self, offset: float) -> float:
        density_error = gs.qd_float(0.0)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                density_error += self._density0 * self.particles_reordered[i_p, i_b].drho - offset
        return density_error

    def _divergence_solver_iteration(self):
        self._kernel_divergence_solver_iteration()
        self._kernel_compute_density_time_derivative()
        density_err = self._kernel_compute_density_error(0.0)
        return density_err / self._n_particles

    def _divergence_solve(self, f: qd.i32):
        # TODO: warm start
        # Compute velocity of density change
        self._kernel_compute_density_time_derivative()
        inv_dt = 1 / self._substep_dt
        # self._kernel_multiply_time_step(self.ps.dfsph_factor, inv_dt)

        # Start solver
        iteration = 0
        avg_density_err = 0.0
        while iteration < self._df_max_div_iters:
            avg_density_err = self._divergence_solver_iteration()
            # Max allowed density fluctuation
            # The SI unit for divergence is s^-1, use max density error divided by time step size
            eta = inv_dt * self._df_max_error_div * 0.01 * self._density0
            if avg_density_err <= eta:
                break
            iteration += 1

        gs.logger.debug(f"DFSPH - iteration V: {iteration} Avg divergence err: {avg_density_err / self._density0:.4f}")

        # Multiply by h, the time step size has to be removed
        # to make the stiffness value independent
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v

        # self._kernel_multiply_time_step(self.ps.dfsph_factor, self.dt[None])

    @qd.kernel
    def _kernel_predict_velocity(self, f: qd.i32):
        # compute new velocities only considering non-pressure forces
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].acc

    @qd.func
    def _task_compute_density_star(self, i, j, ret: qd.template(), i_b):
        v_i = self.particles_reordered[i, i_b].vel
        v_j = self.particles_reordered[j, i_b].vel
        x_i = self.particles_reordered[i, i_b].pos
        x_j = self.particles_reordered[j, i_b].pos
        ret += self._particle_volume * (v_i - v_j).dot(self.cubic_kernel_derivative(x_i - x_j))

    @qd.kernel
    def _kernel_compute_density_star(self):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                delta = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, delta, self._task_compute_density_star, i_b
                )
                drho = self.particles_reordered[i_p, i_b].rho / self._density0 + self._substep_dt * delta
                self.particles_reordered[i_p, i_b].drho = qd.max(drho, 1.0)  # - 1.0

    @qd.func
    def density_solve_iteration_task(self, i, j, ret: qd.template(), i_b):
        # Fluid neighbors
        b_j = self.particles_reordered[j, i_b].drho - 1.0
        k_j = b_j * self.particles_reordered[j, i_b].dfsph_factor
        k_sum = (
            self._density0 / self._density0 * ret.k_i + k_j
        )  # TODO: make the neighbor density0 different for multiphase fluid
        if qd.abs(k_sum) > self._df_eps:
            grad_p_j = -self._particle_volume * self.cubic_kernel_derivative(
                self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
            )
            # Directly update velocities instead of storing pressure accelerations
            ret.dv -= (
                self._substep_dt * k_sum * grad_p_j
            )  # ki, kj already contain inverse density, i.e., density canceled if not mutiphase flow

    @qd.kernel
    def _kernel_density_solve_iteration(self):
        # Compute pressure forces
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                # Evaluate rhs
                b_i = self.particles_reordered[i_p, i_b].drho - 1.0
                k_i = b_i * self.particles_reordered[i_p, i_b].dfsph_factor

                ret = qd.Struct(dv=qd.Vector.zero(gs.qd_float, 3), k_i=k_i)

                # TODO: if warmstart
                # get kappa V
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, ret, self.density_solve_iteration_task, i_b
                )
                self.particles_reordered[i_p, i_b].vel = self.particles_reordered[i_p, i_b].vel + ret.dv

                # The stiffness k_i is the pressure this Jacobi iteration applies to cancel the current density
                # error, up to the -1/rho0 scaling carried by dfsph_factor, so summing -k_i * rho0 over the
                # iterations of the density solve yields the implicit pressure field that enforces
                # incompressibility over the substep. This exposes in particles.p the same physical quantity
                # that the equation of state provides under WCSPH. The field is zeroed once per substep in
                # _kernel_compute_DFSPH_factor.
                rho0 = self.particles_info_reordered[i_p, i_b].rho
                self.particles_reordered[i_p, i_b].p = self.particles_reordered[i_p, i_b].p - k_i * rho0

    def _density_solve_iteration(self):
        self._kernel_density_solve_iteration()
        self._kernel_compute_density_star()
        density_err = self._kernel_compute_density_error(self._density0)
        return density_err / self._n_particles

    @qd.kernel
    def _kernel_multiply_time_step(self, field: qd.template(), time_step: float):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                field[i_p, i_b] *= time_step

    def _density_solve(self, f: qd.i32):
        inv_dt2 = 1.0 / self._substep_dt**2

        # Compute density star
        self._kernel_compute_density_star()

        self._kernel_multiply_time_step(self.particles_reordered.dfsph_factor, inv_dt2)

        # Start solver
        iteration = 0
        avg_density_err = 0.0
        while iteration < self._df_max_den_iters:
            avg_density_err = self._density_solve_iteration()
            # Max allowed density fluctuation
            eta = self._df_max_error_den * 0.01 * self._density0
            if avg_density_err <= eta:
                break
            iteration += 1

        gs.logger.debug(f"DFSPH - iterations: {iteration} Avg density err: {avg_density_err:.4f} kg/m^3")

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def cubic_kernel(self, r_norm):
        """
        Cubic spline smoothing kernel.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        k = 8.0 / np.pi / h**3
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
    def cubic_kernel_derivative(self, r):
        """
        Derivative of cubic spline smoothing kernel.
        """
        res = qd.Vector.zero(gs.qd_float, 3)

        r_norm = r.norm()
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
            else:
                res = -6.0 * k * (1.0 - q) ** 2 * grad_q
        return res

    @qd.func
    def cubic_kernel_plane_integral(self, dist):
        """
        Integral of the cubic spline smoothing kernel over the plane at distance dist from its center.

        Closed form of 2 * pi * int_{|dist|}^{h} W(r) r dr. Its own integral along the plane normal equals
        half the kernel normalization: int_{0}^{h} of this quantity over dist is exactly 1/2, which makes
        forces weighted by it integrate to a per-unit-area magnitude over a covering layer of particles.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
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

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self.entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        if self.is_active:
            self._kernel_reorder_particles(f)
            if self._pressure_solver == "WCSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_compute_pressure_forces(f)
                self._kernel_advect_velocity(f)
            elif self._pressure_solver == "DFSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_DFSPH_factor(f)
                self._divergence_solve(f)
                self._kernel_advect_velocity(f)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_predict_velocity(f)
                self._density_solve(f)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_advect_position(f)
            self._kernel_copy_from_reordered(f)

    def substep_post_coupling_grad(self, f):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.active)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
            self.particles_ng[i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        if self.is_active:
            state = SPHSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
            active[i_b, i_p] = self.particles_ng[i_p, i_b].active

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        mat_rho: qd.f32,
        mat_stiffness: qd.f32,
        mat_exponent: qd.f32,
        mat_mu: qd.f32,
        mat_gamma: qd.f32,
        pos: qd.types.ndarray(),
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles_ng[i_p, i_b].active = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].p = 0

        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            self.particles_info[i_p].rho = mat_rho
            self.particles_info[i_p].stiffness = mat_stiffness
            self.particles_info[i_p].exponent = mat_exponent
            self.particles_info[i_p].mu = mat_mu
            self.particles_info[i_p].gamma = mat_gamma
            self.particles_info[i_p].mass = self._particle_volume * mat_rho

    # ----------------------------------------------------------------------

    @qd.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
            self.particles[i_p, i_b].vel.fill(0.0)
            self.particles[i_p, i_b].acc.fill(0.0)

    @qd.kernel
    def _kernel_get_particles_pos(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[i_p, i_b].pos[i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]
            self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_get_particles_vel(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[i_p, i_b].vel[i]

    @qd.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles_ng[i_p, i_b].active = actives[i_b_, i_p_]

    @qd.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_get_mass(
        self, particle_start: qd.i32, n_particles: qd.i32, mass: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            mass[i_b_] += self.particles[i_p, i_b].m

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def support_radius(self):
        return self._support_radius

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
