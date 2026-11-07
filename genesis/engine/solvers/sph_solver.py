from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine.states.solvers import SPHSolverState

from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.entities import SPHEntity


@qd.data_oriented
class SPHSolver(ParticleSolver):
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

        if self._pressure_solver == "DFSPH":
            # Stiffness that the next Jacobi iteration applies, read by the neighbors of each particle
            self.particles_k_reordered = qd.field(gs.qd_float, shape=(self._n_particles, self._B), needs_grad=False)

            struct_env_df = qd.types.struct(
                err=gs.qd_float,  # density error summed over the active particles
                n_active=gs.qd_int,  # number of active particles, over which the density error is averaged
                is_solving=gs.qd_bool,
            )
            self.envs_df = struct_env_df.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.SOA)

            struct_df_info = qd.types.struct(
                substep_dt=gs.qd_float,
                density_err_tol=gs.qd_float,  # average density error that ends the density solve, kg/m^3
                divergence_err_tol=gs.qd_float,  # average density rate that ends the divergence solve, kg/m^3/s
                stiffness_eps=gs.qd_float,
                jacobi_relaxation=gs.qd_float,
                factor_eps=gs.qd_float,
                n_neighbors_min=gs.qd_int,  # fewer neighbors leave the density rate of a particle out of the solve
            )
            self.df_info = struct_df_info.field(shape=(), needs_grad=False)

    def build(self):
        super().build()

        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles

        self._coupler = self.sim._coupler

        if self.is_active:
            self.sh.build(self._n_particles, self._B)
            self.init_particle_fields()

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

            if self._pressure_solver == "DFSPH":
                self.df_info.substep_dt[None] = self.substep_dt
                self.df_info.density_err_tol[None] = self._df_max_error_den * 0.01 * self._density0
                self.df_info.divergence_err_tol[None] = (
                    1.0 / self.substep_dt * self._df_max_error_div * 0.01 * self._density0
                )
                self.df_info.stiffness_eps[None] = 1e-5
                # The DFSPH factor inverts the diagonal of the pressure system alone, and the stiffness of the neighbors
                # of a particle adds up to as much again, so the diagonally scaled system has eigenvalues up to 2. A
                # full Jacobi update leaves the mode alternating from one particle to the next undamped, and overshoots
                # once the liquid compresses abruptly. Half the update cancels that mode, the others still converging.
                self.df_info.jacobi_relaxation[None] = 0.5
                self.df_info.factor_eps[None] = 1e-6
                self.df_info.n_neighbors_min[None] = 20

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity()

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

        # Copy to reordered, the active particles taking the first sorted indices (see compute_reordered_idx).
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles_ng_reordered[i_p, i_b].active = i_p < self.sh.cur_cnt[i_b]
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]
                self.particles_info_reordered[reordered_idx, i_b] = self.particles_info[i_p]

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
    def _kernel_compute_non_pressure_forces(self, f: qd.i32, t: float):
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
        # Every environment takes part in the divergence solve that follows (see _kernel_update_convergence).
        for i_b in range(self._B):
            self.envs_df[i_b].is_solving = True
            self.envs_df[i_b].err = 0.0
            self.envs_df[i_b].n_active = 0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                qd.atomic_add(self.envs_df[i_b].n_active, 1)

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
                if sum_grad_p_k > self.df_info[None].factor_eps:
                    factor = -1.0 / sum_grad_p_k
                else:
                    # A particle without neighbors carries no pressure into the next density solve.
                    factor = 0.0
                    self.particles_reordered[i_p, i_b].p = 0.0
                self.particles_reordered[i_p, i_b].dfsph_factor = factor

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
            if self.envs_df[i_b].is_solving and self.particles_ng_reordered[i_p, i_b].active:
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
                if num_neighbors < self.df_info[None].n_neighbors_min:
                    drho = 0.0

                self.particles_reordered[i_p, i_b].drho = drho

                # Stiffness that the next Jacobi iteration applies to cancel the density rate of the particle.
                self.particles_k_reordered[i_p, i_b] = drho * self.particles_reordered[i_p, i_b].dfsph_factor

                # The divergence error of the environment, summed for _kernel_update_convergence.
                qd.atomic_add(self.envs_df[i_b].err, self.particles_info_reordered[i_p, i_b].rho * drho)

    @qd.kernel
    def _kernel_predict_velocity(self, f: qd.i32):
        # Every environment takes part in the density solve that follows (see _kernel_update_convergence).
        for i_b in range(self._B):
            self.envs_df[i_b].is_solving = True
            self.envs_df[i_b].err = 0.0

        dt = self.df_info[None].substep_dt
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                # compute new velocities only considering non-pressure forces
                self.particles_reordered[i_p, i_b].vel = (
                    self.particles_reordered[i_p, i_b].vel + dt * self.particles_reordered[i_p, i_b].acc
                )

                # The density solve corrects a density error over the substep, which turns the DFSPH factor of the
                # divergence solve, a velocity correction per density rate, into an acceleration per density error.
                self.particles_reordered[i_p, i_b].dfsph_factor = self.particles_reordered[i_p, i_b].dfsph_factor * (
                    1.0 / dt**2
                )

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
            if self.envs_df[i_b].is_solving and self.particles_ng_reordered[i_p, i_b].active:
                delta = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, delta, self._task_compute_density_star, i_b
                )
                rho0 = self.particles_info_reordered[i_p, i_b].rho
                drho = self.particles_reordered[i_p, i_b].rho / rho0 + self.df_info[None].substep_dt * delta
                self.particles_reordered[i_p, i_b].drho = drho

                # Stiffness that the next Jacobi iteration applies to cancel the density error of the particle. The
                # stiffness accumulated by the density solve is carried over from one substep to the next, so the
                # iterations update it in both directions: they take back the part the liquid no longer needs, and
                # clamp it so that the pressure never pulls the liquid together. The update is relaxed (see
                # jacobi_relaxation in build).
                kappa = -self.particles_reordered[i_p, i_b].p / rho0
                k_i = (
                    self.df_info[None].jacobi_relaxation
                    * (drho - 1.0)
                    * self.particles_reordered[i_p, i_b].dfsph_factor
                )
                self.particles_k_reordered[i_p, i_b] = qd.min(kappa + k_i, 0.0) - kappa

                # The density error of the environment, summed for _kernel_update_convergence. A particle under
                # pressure counts its expansion as well, which the iterations correct by taking pressure back, while
                # a particle free of pressure only counts its compression, the liquid being free to expand.
                err_rel = drho - 1.0
                if kappa >= 0.0:
                    err_rel = qd.max(err_rel, 0.0)
                qd.atomic_add(self.envs_df[i_b].err, rho0 * qd.abs(err_rel))

    @qd.func
    def _func_stiffness(self, i_p: int, i_b: int, warm_start: bool):
        # The warm start applies the stiffness carried over from the previous substep in particles.p (see
        # _kernel_solver_iteration) to the particles compressed at the start of the substep.
        k_i = self.particles_k_reordered[i_p, i_b]
        if warm_start:
            k_i = 0.0
            rho0 = self.particles_info_reordered[i_p, i_b].rho
            if self.particles_reordered[i_p, i_b].rho > rho0:
                k_i = -self.particles_reordered[i_p, i_b].p / rho0
        return k_i

    @qd.func
    def _task_solver_iteration(self, i_p: int, j_p: int, ret: qd.template(), i_b: int):
        # The stiffness already holds the inverse of the density, which is the same for every particle of a liquid.
        k_sum = ret.k_i + self._func_stiffness(j_p, i_b, ret.is_warm_start)
        if qd.abs(k_sum) > self.df_info[None].stiffness_eps:
            grad_p_j = -self._particle_volume * self.cubic_kernel_derivative(
                self.particles_reordered[i_p, i_b].pos - self.particles_reordered[j_p, i_b].pos
            )
            ret.dv = ret.dv - ret.dv_scale * k_sum * grad_p_j

    @qd.kernel
    def _kernel_solver_iteration(self, solve_density: qd.template(), warm_start: bool):
        # Jacobi iteration: the velocity of every particle changes by the stiffness of its own and of its neighbors.
        # The divergence solve corrects velocities directly, while the density solve accelerates over the substep.
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_df[i_b].is_solving and self.particles_ng_reordered[i_p, i_b].active:
                dv_scale = gs.qd_float(1.0)
                if qd.static(solve_density):
                    dv_scale = self.df_info[None].substep_dt
                k_i = self._func_stiffness(i_p, i_b, warm_start)
                ret = qd.Struct(dv=qd.Vector.zero(gs.qd_float, 3), k_i=k_i, dv_scale=dv_scale, is_warm_start=warm_start)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, ret, self._task_solver_iteration, i_b
                )
                self.particles_reordered[i_p, i_b].vel = self.particles_reordered[i_p, i_b].vel + ret.dv

                if qd.static(solve_density):
                    # The stiffness k_i is the pressure this pass applies to cancel the density error, up to the
                    # -1/rho0 scaling carried by dfsph_factor, so summing -k_i * rho0 over the density solve yields
                    # the implicit pressure field that enforces incompressibility. This exposes in particles.p the
                    # same physical quantity that the equation of state provides under WCSPH. The warm start applies
                    # the pressure as it stands and drops the pressure of the particles it leaves out, which the
                    # neighbors reading it in the meantime leave out the same way.
                    rho0 = self.particles_info_reordered[i_p, i_b].rho
                    if warm_start:
                        if self.particles_reordered[i_p, i_b].rho <= rho0:
                            self.particles_reordered[i_p, i_b].p = 0.0
                    else:
                        self.particles_reordered[i_p, i_b].p = self.particles_reordered[i_p, i_b].p - k_i * rho0

    @qd.kernel
    def _kernel_update_convergence(self, solve_density: qd.template()) -> int:
        # Each environment stops iterating once the average density error of its own particles meets the tolerance,
        # so that its result is the same whatever the other environments simulate alongside it. The density rate
        # kernels sum the error of each environment, which is reset once read for the next call.
        n_envs_solving = 0
        for i_b in range(self._B):
            if self.envs_df[i_b].is_solving:
                err_tol = self.df_info[None].divergence_err_tol
                if qd.static(solve_density):
                    err_tol = self.df_info[None].density_err_tol
                if self.envs_df[i_b].err <= err_tol * self.envs_df[i_b].n_active:
                    self.envs_df[i_b].is_solving = False
                else:
                    qd.atomic_add(n_envs_solving, 1)
                self.envs_df[i_b].err = 0.0
        return n_envs_solving

    def _solve_incompressibility(self, solve_density: bool) -> None:
        if solve_density:
            kernel_compute_density_rate = self._kernel_compute_density_star
            n_iters_max = self._df_max_den_iters
        else:
            kernel_compute_density_rate = self._kernel_compute_density_time_derivative
            n_iters_max = self._df_max_div_iters

        # The convergence is checked before the first iteration: an environment that the warm start of the density
        # solve, or a divergence-free velocity field, already brings within tolerance does not iterate at all.
        if solve_density:
            self._kernel_solver_iteration(solve_density, warm_start=True)
        kernel_compute_density_rate()
        n_envs_solving = self._kernel_update_convergence(solve_density)
        n_iters = 0
        while n_envs_solving > 0 and n_iters < n_iters_max:
            self._kernel_solver_iteration(solve_density, warm_start=False)
            kernel_compute_density_rate()
            n_envs_solving = self._kernel_update_convergence(solve_density)
            n_iters += 1

        gs.logger.debug(f"DFSPH - {'density' if solve_density else 'divergence'} solve iterations: {n_iters}")

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
                self._solve_incompressibility(solve_density=False)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_predict_velocity(f)
                self._solve_incompressibility(solve_density=True)

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_advect_position(f)
            self._kernel_copy_from_reordered(f)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.active)
            if self._pressure_solver == "DFSPH":
                self._kernel_set_pressure(state.pressure)

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
            if self._pressure_solver == "DFSPH":
                self._kernel_get_pressure(state.pressure)
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

    @qd.kernel
    def _kernel_set_pressure(self, pressure: qd.types.ndarray()):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles[i_p, i_b].p = pressure[i_b, i_p]

    @qd.kernel
    def _kernel_get_pressure(self, pressure: qd.types.ndarray()):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            pressure[i_b, i_p] = self.particles[i_p, i_b].p

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
    def add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        mat_rho: float,
        mat_stiffness: float,
        mat_exponent: float,
        mat_mu: float,
        mat_gamma: float,
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
            # The earlier occurrences of a repeated particle are -1 (see 'drop_repeated_index')
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
                self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
                self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)
                # The pressure that DFSPH carries over belongs to the neighborhood the particle leaves.
                self.particles[i_p, i_b].p = 0.0

    @qd.kernel
    def _kernel_get_particles_pos(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
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
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]
                self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_get_particles_vel(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
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
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_ng[i_p, i_b].active = actives[i_b_, i_p_]
                self.particles[i_p, i_b].p = 0.0

    @qd.kernel
    def _kernel_get_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_get_particles_mass(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), masses: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            masses[i_b_, i_p_] = self.particles[i_p, i_b].m

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
    def pressure_solver(self) -> str:
        return self._pressure_solver

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
