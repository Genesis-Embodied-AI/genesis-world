from __future__ import annotations

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.constraint import linesearch, solver
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver, func_step_1, func_step_2

from .sim_system import SimSystem


@qd.data_oriented
class RigidSystem(SimSystem):
    """Expose the Genesis minimal-coordinate Rigid numerical core to the global Newton runtime."""

    def __init__(self, rigid_solver: RigidSolver) -> None:
        super().__init__()
        if gs.qd_float != qd.f64:
            raise RuntimeError("The Rigid Newton framework requires double precision")
        if rigid_solver._requires_grad:
            raise RuntimeError("The Rigid Newton framework does not support differentiable simulation")
        if rigid_solver.rigid_config.solver_type != gs.constraint_solver.Newton:
            raise RuntimeError("The Rigid Newton framework requires the native Newton constraint formulation")
        if rigid_solver._options.noslip_iterations > 0:
            raise RuntimeError("The Rigid Newton framework does not support the noslip post-processing solve")
        if rigid_solver.rigid_config.use_hibernation:
            raise RuntimeError("The Rigid Newton framework does not support hibernation")

        self.dyn_state = rigid_solver.dyn_state
        self.constraint_state = rigid_solver.constraint_solver.constraint_state
        self.dyn_info = rigid_solver.dyn_info
        self.rigid_info = rigid_solver.rigid_info
        self.rigid_config = rigid_solver.rigid_config
        self.collider_state = rigid_solver.collider.collider_state
        self.collider_config = rigid_solver.collider.collider_config
        self.errno = rigid_solver._errno

        self.has_constraints = not rigid_solver._disable_constraint
        self.has_collision = False

        self.n_dofs_per_instance_host = rigid_solver.n_dofs
        self.n_instances_host = rigid_solver._B
        self.dof_count_host = self.n_dofs_per_instance_host * self.n_instances_host
        self.storage_dof_count_host = ((self.dof_count_host + 2) // 3) * 3
        self.n_dofs_per_instance = qd.ndarray(qd.i32, shape=())
        self.n_instances = qd.ndarray(qd.i32, shape=())
        self.n_links = qd.ndarray(qd.i32, shape=())
        self.n_dofs = qd.ndarray(qd.i32, shape=())
        self.dof_offset = qd.ndarray(qd.i32, shape=())
        self.n_storage_dofs = qd.ndarray(qd.i32, shape=())
        self.gradient_squared = qd.ndarray(qd.f64, shape=())
        self.rigid_energy = qd.ndarray(qd.f64, shape=())
        self.qacc_temp = qd.ndarray(qd.f64, shape=self.constraint_state.qacc.shape)
        self.Ma_temp = qd.ndarray(qd.f64, shape=self.constraint_state.Ma.shape)
        self.Jaref_temp = qd.ndarray(qd.f64, shape=self.constraint_state.Jaref.shape)
        self.n_dofs_per_instance.from_numpy(np.array(self.n_dofs_per_instance_host, dtype=np.int32))
        self.n_instances.from_numpy(np.array(self.n_instances_host, dtype=np.int32))
        self.n_links.from_numpy(np.array(rigid_solver.n_links, dtype=np.int32))
        self.n_dofs.from_numpy(np.array(self.dof_count_host, dtype=np.int32))
        self.dof_offset.from_numpy(np.array(0, dtype=np.int32))
        self.n_storage_dofs.from_numpy(np.array(self.storage_dof_count_host, dtype=np.int32))
        self.h = rigid_solver._substep_dt
        self.h4 = self.h**4
        self.is_forward_pos_updated = rigid_solver._is_forward_pos_updated
        self.is_forward_vel_updated = rigid_solver._is_forward_vel_updated

    def do_build(self) -> None:
        pass

    def init(self, dof_offset: int) -> None:
        self.dof_offset.from_numpy(np.array(dof_offset, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def predict(self):
        func_step_1(
            self.dyn_state,
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self.is_forward_pos_updated,
            self.is_forward_vel_updated,
            False,
        )

    @qd.func(requires_top_level=True)
    def assemble_candidate_rows(self):
        if qd.static(self.has_constraints):
            solver.func_add_equality_constraints(
                self.dyn_state,
                self.collider_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

        for i_b in range(self.n_instances[()]):
            self.collider_state.n_contacts[i_b] = 0
            self.collider_state.n_contacts_hibernated[i_b] = 0

        if qd.static(self.has_constraints):
            solver.func_add_inequality_constraints(
                self.dyn_state,
                self.collider_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                self.collider_config,
            )

    @qd.func(requires_top_level=True)
    def initialize_newton(self):
        if qd.static(self.has_constraints):
            solver.func_solve_init(
                self.dyn_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                True,
            )
            self.set_newton_active(1)
            self.build_preconditioner(compute_envelope=True)
            solver.func_update_gradient_no_solve(
                self.dyn_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    @qd.func(requires_top_level=True)
    def set_newton_active(self, is_active):
        for i_b in range(self.n_instances[()]):
            has_constraints = self.constraint_state.n_constraints[i_b] > 0 and is_active != 0
            self.constraint_state.improved[i_b] = has_constraints
            for i_island in range(self.constraint_state.island.n_islands[i_b]):
                self.constraint_state.island.improved[i_island, i_b] = has_constraints

    @qd.func(requires_top_level=True)
    def build_preconditioner(self, compute_envelope: qd.template()):
        solver.func_hessian_and_cholesky_factor_direct(
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            compute_envelope=compute_envelope,
        )

    @qd.func(requires_top_level=True)
    def assemble(
        self,
        sim_config: qd.template(),
        global_linear_system: qd.template(),
        displacement_coordinates: qd.template(),
    ):
        for _ in range(1):
            self.gradient_squared[()] = qd.f64(0.0)
        gradient_scale = self.h4
        if qd.static(displacement_coordinates):
            gradient_scale = self.h * self.h
        for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
            i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
            gradient = qd.f64(0.0)
            gradient_unscaled = qd.f64(0.0)
            has_live_constraints = False
            if qd.static(self.has_constraints):
                has_live_constraints = self.constraint_state.n_constraints[i_b] > 0
            if has_live_constraints:
                gradient_unscaled = self.constraint_state.grad[i_d, i_b]
                gradient = gradient_scale * gradient_unscaled
            global_linear_system.b_rhs[i_global] = gradient
            qd.atomic_add(self.gradient_squared[()], gradient_unscaled * gradient_unscaled)

        for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
            global_linear_system.b_rhs[self.dof_offset[()] + i_padding] = qd.f64(0.0)

    @qd.func(requires_top_level=True)
    def energy(self, sim_config: qd.template()):
        for _ in range(1):
            self.rigid_energy[()] = qd.f64(0.0)
        if qd.static(self.has_constraints):
            for i_b in range(self.n_instances[()]):
                qd.atomic_add(self.rigid_energy[()], self.h4 * self.constraint_state.cost[i_b])

    @qd.func(requires_top_level=True)
    def apply_hessian(
        self,
        x: qd.template(),
        y: qd.template(),
        displacement_coordinates: qd.template(),
    ):
        for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
            i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
            self.constraint_state.search[i_d, i_b] = x[i_global]

        for i_b in range(self.n_instances[()]):
            if qd.static(self.has_constraints) and self.constraint_state.n_constraints[i_b] > 0:
                for i_island in range(self.constraint_state.island.n_islands[i_b]):
                    n_dofs = self.constraint_state.island.dof_slices.n[i_island, i_b]
                    if qd.static(self.rigid_config.is_single_island):
                        n_dofs = self.n_dofs_per_instance[()]
                    i_dof_start = self.constraint_state.island.dof_slices.start[i_island, i_b]

                    for j_d_local in range(n_dofs):
                        j_d = j_d_local
                        if qd.static(not self.rigid_config.is_single_island or self.rigid_config.sparse_solve):
                            j_d = self.constraint_state.island.dof_id[i_dof_start + j_d_local, i_b]
                        value = gs.qd_float(0.0)
                        for i_d_local in range(j_d_local, n_dofs):
                            i_d = i_d_local
                            if qd.static(not self.rigid_config.is_single_island or self.rigid_config.sparse_solve):
                                i_d = self.constraint_state.island.dof_id[i_dof_start + i_d_local, i_b]
                            scale = gs.qd_float(1.0)
                            if qd.static(self.rigid_config.enable_jacobi_equilibration):
                                scale = self.constraint_state.nt_jacobi[i_d, i_b]
                            value = value + (
                                self.constraint_state.nt_H[i_b, i_d, j_d]
                                * self.constraint_state.search[i_d, i_b]
                                / scale
                            )
                        self.constraint_state.Mgrad[j_d, i_b] = value

                    for i_d_local in range(n_dofs):
                        i_d = i_d_local
                        if qd.static(not self.rigid_config.is_single_island or self.rigid_config.sparse_solve):
                            i_d = self.constraint_state.island.dof_id[i_dof_start + i_d_local, i_b]
                        value = gs.qd_float(0.0)
                        for j_d_local in range(i_d_local + 1):
                            j_d = j_d_local
                            if qd.static(not self.rigid_config.is_single_island or self.rigid_config.sparse_solve):
                                j_d = self.constraint_state.island.dof_id[i_dof_start + j_d_local, i_b]
                            value = value + (
                                self.constraint_state.nt_H[i_b, i_d, j_d] * self.constraint_state.Mgrad[j_d, i_b]
                            )
                        scale = gs.qd_float(1.0)
                        if qd.static(self.rigid_config.enable_jacobi_equilibration):
                            scale = self.constraint_state.nt_jacobi[i_d, i_b]
                        self.constraint_state.grad[i_d, i_b] = value / scale
            else:
                for i_d in range(self.n_dofs_per_instance[()]):
                    value = gs.qd_float(0.0)
                    for j_d in range(self.n_dofs_per_instance[()]):
                        value = value + (
                            self.rigid_info.mass_mat[i_d, j_d, i_b] * self.constraint_state.search[j_d, i_b]
                        )
                    self.constraint_state.grad[i_d, i_b] = value

        hessian_scale = self.h4
        if qd.static(displacement_coordinates):
            hessian_scale = 1.0
        for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
            i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
            y[i_global] = y[i_global] + hessian_scale * self.constraint_state.grad[i_d, i_b]

        for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
            i_global = self.dof_offset[()] + i_padding
            y[i_global] = y[i_global] + x[i_global]

    @qd.func(requires_top_level=True)
    def apply_preconditioner(
        self,
        residual: qd.template(),
        result: qd.template(),
        displacement_coordinates: qd.template(),
    ):
        if qd.static(self.has_constraints):
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
                self.constraint_state.grad[i_d, i_b] = residual[i_global]

            for i_b in range(self.n_instances[()]):
                if self.constraint_state.n_constraints[i_b] > 0:
                    for i_island in range(self.constraint_state.island.n_islands[i_b]):
                        solver.func_cholesky_solve_batch(
                            i_b,
                            i_island,
                            rhs=self.constraint_state.grad,
                            out=self.constraint_state.Mgrad,
                            constraint_state=self.constraint_state,
                            rigid_config=self.rigid_config,
                        )

            preconditioner_scale = 1.0 / self.h4
            if qd.static(displacement_coordinates):
                preconditioner_scale = 1.0
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
                if self.constraint_state.n_constraints[i_b] > 0:
                    result[i_global] = preconditioner_scale * self.constraint_state.Mgrad[i_d, i_b]

        for i_padding in range(self.n_dofs[()], self.n_storage_dofs[()]):
            i_global = self.dof_offset[()] + i_padding
            result[i_global] = residual[i_global]

    @qd.func(requires_top_level=True)
    def negate_dq(
        self,
        global_linear_system: qd.template(),
        displacement_coordinates: qd.template(),
    ):
        if qd.static(self.has_constraints):
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                i_global = self.dof_offset[()] + i_b * self.n_dofs_per_instance[()] + i_d
                direction = global_linear_system.x_sol[i_global]
                if qd.static(displacement_coordinates):
                    direction = direction / (self.h * self.h)
                self.constraint_state.search[i_d, i_b] = -direction
                self.constraint_state.Mgrad[i_d, i_b] = direction
            for i_b in range(self.n_instances[()]):
                if self.constraint_state.n_constraints[i_b] > 0:
                    linesearch.func_mv_jv_dense(i_b, self.constraint_state, self.rigid_info)

    @qd.func(requires_top_level=True)
    def record_start_point(self):
        if qd.static(self.has_constraints):
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                self.qacc_temp[i_d, i_b] = self.constraint_state.qacc[i_d, i_b]
                self.Ma_temp[i_d, i_b] = self.constraint_state.Ma[i_d, i_b]
            for i_c, i_b in qd.ndrange(self.constraint_state.Jaref.shape[0], self.n_instances[()]):
                if i_c < self.constraint_state.n_constraints[i_b]:
                    self.Jaref_temp[i_c, i_b] = self.constraint_state.Jaref[i_c, i_b]

    @qd.func(requires_top_level=True)
    def step_forward(self, alpha):
        if qd.static(self.has_constraints):
            for i_d, i_b in qd.ndrange(self.n_dofs_per_instance[()], self.n_instances[()]):
                self.constraint_state.qacc[i_d, i_b] = (
                    self.qacc_temp[i_d, i_b] + alpha * self.constraint_state.search[i_d, i_b]
                )
                self.constraint_state.Ma[i_d, i_b] = self.Ma_temp[i_d, i_b] + alpha * self.constraint_state.mv[i_d, i_b]
            for i_c, i_b in qd.ndrange(self.constraint_state.Jaref.shape[0], self.n_instances[()]):
                if i_c < self.constraint_state.n_constraints[i_b]:
                    self.constraint_state.Jaref[i_c, i_b] = (
                        self.Jaref_temp[i_c, i_b] + alpha * self.constraint_state.jv[i_c, i_b]
                    )

            solver.func_update_constraint(
                qacc=self.constraint_state.qacc,
                Ma=self.constraint_state.Ma,
                cost=self.constraint_state.cost,
                dyn_state=self.dyn_state,
                constraint_state=self.constraint_state,
                rigid_config=self.rigid_config,
            )
            self.set_newton_active(1)
            solver.func_update_gradient_no_solve(
                self.dyn_state,
                self.constraint_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    @qd.func(requires_top_level=True)
    def update_velocity(self):
        if qd.static(self.has_constraints):
            solver.func_update_qacc(self.dyn_state, self.constraint_state, self.rigid_config, self.errno)
        func_step_2(
            self.dyn_state,
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            False,
            self.errno,
        )
