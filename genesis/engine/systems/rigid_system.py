from __future__ import annotations

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils import array_class
from genesis.engine.solvers.rigid.constraint import linesearch, solver
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver, func_step_1, func_step_2

from .sim_system import SimData, SimSystem


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class RigidSystem(SimSystem):
    """Expose the existing Genesis RigidSolver numerical state to the Newton runtime."""

    @qd.data_oriented
    class Data(SimData):
        """Stable references to RigidSolver-owned state plus coupling scratch."""

        n_dofs_per_instance: qd.Ndarray
        n_instances: qd.Ndarray
        n_links: qd.Ndarray
        n_dofs: qd.Ndarray
        dof_offset: qd.Ndarray
        n_storage_dofs: qd.Ndarray
        h: qd.Ndarray
        gradient_squared: qd.Ndarray
        rigid_energy: qd.Ndarray
        qacc_temp: qd.Ndarray
        Ma_temp: qd.Ndarray
        Jaref_temp: qd.Ndarray
        dyn_state: array_class.DynState
        constraint_state: array_class.ConstraintState
        dyn_info: array_class.DynInfo
        rigid_info: array_class.RigidInfo
        rigid_config: array_class.RigidSimStaticConfig
        collider_state: array_class.ColliderState
        collider_config: array_class.ColliderStaticConfig
        errno: qd.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._rigid_solver: RigidSolver | None = None
        self.has_constraints = False
        self.has_collision = False
        self.n_dofs_per_instance = 0
        self.n_instances = 0
        self.dof_count = 0
        self.storage_dof_count = 0
        self.h = 0.0
        self.h4 = 0.0
        self.is_forward_pos_updated = False
        self.is_forward_vel_updated = False

    def wire_solver(self, rigid_solver: RigidSolver) -> None:
        self._rigid_solver = rigid_solver
        self.has_constraints = not rigid_solver._disable_constraint
        self.n_dofs_per_instance = rigid_solver.n_dofs
        self.n_instances = rigid_solver._B
        self.dof_count = self.n_dofs_per_instance * self.n_instances
        self.storage_dof_count = ((self.dof_count + 2) // 3) * 3
        self.h = rigid_solver._substep_dt
        self.h4 = self.h**4
        self.is_forward_pos_updated = rigid_solver._is_forward_pos_updated
        self.is_forward_vel_updated = rigid_solver._is_forward_vel_updated

    def build(self) -> None:
        from .global_linear_system import GlobalLinearSystem
        from .pcg_solver import PCGSolver
        from .sim_config import SimConfig

        pcg_solver = self.require(PCGSolver)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.sim_config_system = self.require(SimConfig)
        self.pcg_operator_action = self.create_action(pcg_apply_operator, self)
        self.pcg_reduced_operator_action = self.create_action(pcg_apply_reduced_operator, self)
        self.pcg_preconditioner_action = self.create_action(pcg_apply_preconditioner, self)
        pcg_solver.on_solve_contribution(
            self.pcg_operator_action,
            self.pcg_reduced_operator_action,
            self.pcg_preconditioner_action,
        )

    def init(self, dof_offset: int) -> None:
        rigid_solver = self._rigid_solver
        if rigid_solver is None:
            raise RuntimeError("RigidSystem solver has not been wired")
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
        data = self.data
        data.dyn_state = rigid_solver.dyn_state
        data.constraint_state = rigid_solver.constraint_solver.constraint_state
        data.dyn_info = rigid_solver.dyn_info
        data.rigid_info = rigid_solver.rigid_info
        data.rigid_config = rigid_solver.rigid_config
        data.collider_state = rigid_solver.collider.collider_state
        data.collider_config = rigid_solver.collider.collider_config
        data.errno = rigid_solver._errno
        data.n_dofs_per_instance = qd.ndarray(qd.i32, shape=())
        data.n_instances = qd.ndarray(qd.i32, shape=())
        data.n_links = qd.ndarray(qd.i32, shape=())
        data.n_dofs = qd.ndarray(qd.i32, shape=())
        data.dof_offset = qd.ndarray(qd.i32, shape=())
        data.n_storage_dofs = qd.ndarray(qd.i32, shape=())
        data.h = qd.ndarray(qd.f64, shape=())
        data.gradient_squared = qd.ndarray(qd.f64, shape=())
        data.rigid_energy = qd.ndarray(qd.f64, shape=())
        data.qacc_temp = qd.ndarray(qd.f64, shape=data.constraint_state.qacc.shape)
        data.Ma_temp = qd.ndarray(qd.f64, shape=data.constraint_state.Ma.shape)
        data.Jaref_temp = qd.ndarray(qd.f64, shape=data.constraint_state.Jaref.shape)
        data.n_dofs_per_instance.from_numpy(np.array(self.n_dofs_per_instance, dtype=np.int32))
        data.n_instances.from_numpy(np.array(self.n_instances, dtype=np.int32))
        data.n_links.from_numpy(np.array(rigid_solver.n_links, dtype=np.int32))
        data.n_dofs.from_numpy(np.array(self.dof_count, dtype=np.int32))
        data.dof_offset.from_numpy(np.array(dof_offset, dtype=np.int32))
        data.n_storage_dofs.from_numpy(np.array(self.storage_dof_count, dtype=np.int32))
        data.h.from_numpy(np.array(self.h, dtype=np.float64))
        data.gradient_squared.from_numpy(np.array(0.0, dtype=np.float64))
        data.rigid_energy.from_numpy(np.array(0.0, dtype=np.float64))
        data.qacc_temp.from_numpy(np.zeros(data.constraint_state.qacc.shape, dtype=np.float64))
        data.Ma_temp.from_numpy(np.zeros(data.constraint_state.Ma.shape, dtype=np.float64))
        data.Jaref_temp.from_numpy(np.zeros(data.constraint_state.Jaref.shape, dtype=np.float64))
        self._rigid_solver = None

    @qd.func(requires_top_level=True)
    def on_predict(self):
        predict(self)

    @qd.func(requires_top_level=True)
    def on_assemble_candidate_rows(self):
        assemble_candidate_rows(self)

    @qd.func(requires_top_level=True)
    def on_initialize_newton(self):
        initialize_newton(self)

    @qd.func(requires_top_level=True)
    def on_assemble(self, displacement_coordinates: qd.template()):
        assemble(
            self,
            self.sim_config_system.data,
            self.global_linear_system_system.data,
            displacement_coordinates,
        )

    @qd.func(requires_top_level=True)
    def on_negate_direction(self, displacement_coordinates: qd.template()):
        negate_dq(
            self,
            self.global_linear_system_system.data,
            displacement_coordinates,
        )

    @qd.func(requires_top_level=True)
    def on_record_start_point(self):
        record_start_point(self)

    @qd.func(requires_top_level=True)
    def on_energy(self):
        energy(self, self.sim_config_system.data)

    @qd.func(requires_top_level=True)
    def on_step_forward(self, alpha):
        step_forward(self, alpha)

    @qd.func(requires_top_level=True)
    def on_set_newton_active(self, is_active):
        set_newton_active(self, is_active)

    @qd.func(requires_top_level=True)
    def on_build_preconditioner(self, compute_envelope: qd.template()):
        build_preconditioner(self, compute_envelope)

    @qd.func(requires_top_level=True)
    def on_update_velocity(self):
        update_velocity(self)


@qd.func(requires_top_level=True)
def predict(
    system: qd.template(),
):
    data = system.data
    func_step_1(
        data.dyn_state,
        data.constraint_state,
        data.dyn_info,
        data.rigid_info,
        data.rigid_config,
        system.is_forward_pos_updated,
        system.is_forward_vel_updated,
        False,
    )


@qd.func(requires_top_level=True)
def assemble_candidate_rows(
    system: qd.template(),
):
    data = system.data
    if qd.static(system.has_constraints):
        solver.func_add_equality_constraints(
            data.dyn_state,
            data.collider_state,
            data.constraint_state,
            data.dyn_info,
            data.rigid_info,
            data.rigid_config,
        )

    for i_b in range(data.n_instances[()]):
        data.collider_state.n_contacts[i_b] = 0
        data.collider_state.n_contacts_hibernated[i_b] = 0

    if qd.static(system.has_constraints):
        solver.func_add_inequality_constraints(
            data.dyn_state,
            data.collider_state,
            data.constraint_state,
            data.dyn_info,
            data.rigid_info,
            data.rigid_config,
            data.collider_config,
        )


@qd.func(requires_top_level=True)
def initialize_newton(
    system: qd.template(),
):
    data = system.data
    if qd.static(system.has_constraints):
        solver.func_solve_init(
            data.dyn_state,
            data.constraint_state,
            data.dyn_info,
            data.rigid_info,
            data.rigid_config,
            True,
        )
        set_newton_active(system, 1)
        build_preconditioner(system, compute_envelope=True)
        solver.func_update_gradient_no_solve(
            data.dyn_state,
            data.constraint_state,
            data.dyn_info,
            data.rigid_info,
            data.rigid_config,
        )


@qd.func(requires_top_level=True)
def set_newton_active(system: qd.template(), is_active):
    data = system.data
    for i_b in range(data.n_instances[()]):
        has_constraints = data.constraint_state.n_constraints[i_b] > 0 and is_active != 0
        data.constraint_state.improved[i_b] = has_constraints
        for i_island in range(data.constraint_state.island.n_islands[i_b]):
            data.constraint_state.island.improved[i_island, i_b] = has_constraints


@qd.func(requires_top_level=True)
def build_preconditioner(system: qd.template(), compute_envelope: qd.template()):
    data = system.data
    solver.func_hessian_and_cholesky_factor_direct(
        data.constraint_state,
        data.dyn_info,
        data.rigid_info,
        data.rigid_config,
        compute_envelope=compute_envelope,
    )


@qd.func(requires_top_level=True)
def assemble(
    system: qd.template(),
    sim_config: qd.template(),
    global_linear_system_data: qd.template(),
    displacement_coordinates: qd.template(),
):
    data = system.data
    for _ in range(1):
        data.gradient_squared[()] = qd.f64(0.0)
    gradient_scale = system.h4
    if qd.static(displacement_coordinates):
        gradient_scale = system.h * system.h
    for i_d, i_b in qd.ndrange(data.n_dofs_per_instance[()], data.n_instances[()]):
        i_global = data.dof_offset[()] + i_b * data.n_dofs_per_instance[()] + i_d
        gradient = qd.f64(0.0)
        gradient_unscaled = qd.f64(0.0)
        has_live_constraints = False
        if qd.static(system.has_constraints):
            has_live_constraints = data.constraint_state.n_constraints[i_b] > 0
        if has_live_constraints:
            gradient_unscaled = data.constraint_state.grad[i_d, i_b]
            gradient = gradient_scale * gradient_unscaled
        global_linear_system_data.write_rhs(i_global, gradient)
        qd.atomic_add(data.gradient_squared[()], gradient_unscaled * gradient_unscaled)

    for i_padding in range(data.n_dofs[()], data.n_storage_dofs[()]):
        global_linear_system_data.write_rhs(data.dof_offset[()] + i_padding, qd.f64(0.0))


@qd.func(requires_top_level=True)
def energy(system: qd.template(), sim_config: qd.template()):
    data = system.data
    for _ in range(1):
        data.rigid_energy[()] = qd.f64(0.0)
    if qd.static(system.has_constraints):
        for i_b in range(data.n_instances[()]):
            qd.atomic_add(data.rigid_energy[()], system.h4 * data.constraint_state.cost[i_b])


@qd.func(requires_top_level=True)
def pcg_apply_operator(
    system: qd.template(),
    linear_system_data: qd.template(),
    direction: qd.template(),
    output: qd.template(),
):
    apply_hessian(system, direction, output, False)


@qd.func(requires_top_level=True)
def pcg_apply_reduced_operator(
    system: qd.template(),
    linear_system_data: qd.template(),
    direction: qd.template(),
    output: qd.template(),
):
    apply_hessian(system, direction, output, True)


@qd.func(requires_top_level=True)
def apply_hessian(
    system: qd.template(),
    x: qd.template(),
    y: qd.template(),
    displacement_coordinates: qd.template(),
):
    data = system.data
    for i_d, i_b in qd.ndrange(data.n_dofs_per_instance[()], data.n_instances[()]):
        i_global = data.dof_offset[()] + i_b * data.n_dofs_per_instance[()] + i_d
        data.constraint_state.search[i_d, i_b] = x[i_global]

    for i_b in range(data.n_instances[()]):
        has_live_constraints = False
        if qd.static(system.has_constraints):
            has_live_constraints = data.constraint_state.n_constraints[i_b] > 0
        if has_live_constraints:
            for i_island in range(data.constraint_state.island.n_islands[i_b]):
                n_dofs = data.constraint_state.island.dof_slices.n[i_island, i_b]
                if qd.static(data.rigid_config.is_single_island):
                    n_dofs = data.n_dofs_per_instance[()]
                i_dof_start = data.constraint_state.island.dof_slices.start[i_island, i_b]

                for j_d_local in range(n_dofs):
                    j_d = j_d_local
                    if qd.static(not data.rigid_config.is_single_island or data.rigid_config.sparse_solve):
                        j_d = data.constraint_state.island.dof_id[i_dof_start + j_d_local, i_b]
                    value = gs.qd_float(0.0)
                    for i_d_local in range(j_d_local, n_dofs):
                        i_d = i_d_local
                        if qd.static(not data.rigid_config.is_single_island or data.rigid_config.sparse_solve):
                            i_d = data.constraint_state.island.dof_id[i_dof_start + i_d_local, i_b]
                        scale = gs.qd_float(1.0)
                        if qd.static(data.rigid_config.enable_jacobi_equilibration):
                            scale = data.constraint_state.nt_jacobi[i_d, i_b]
                        value = value + (
                            data.constraint_state.nt_H[i_b, i_d, j_d] * data.constraint_state.search[i_d, i_b] / scale
                        )
                    data.constraint_state.Mgrad[j_d, i_b] = value

                for i_d_local in range(n_dofs):
                    i_d = i_d_local
                    if qd.static(not data.rigid_config.is_single_island or data.rigid_config.sparse_solve):
                        i_d = data.constraint_state.island.dof_id[i_dof_start + i_d_local, i_b]
                    value = gs.qd_float(0.0)
                    for j_d_local in range(i_d_local + 1):
                        j_d = j_d_local
                        if qd.static(not data.rigid_config.is_single_island or data.rigid_config.sparse_solve):
                            j_d = data.constraint_state.island.dof_id[i_dof_start + j_d_local, i_b]
                        value = value + (
                            data.constraint_state.nt_H[i_b, i_d, j_d] * data.constraint_state.Mgrad[j_d, i_b]
                        )
                    scale = gs.qd_float(1.0)
                    if qd.static(data.rigid_config.enable_jacobi_equilibration):
                        scale = data.constraint_state.nt_jacobi[i_d, i_b]
                    data.constraint_state.grad[i_d, i_b] = value / scale
        else:
            for i_d in range(data.n_dofs_per_instance[()]):
                value = gs.qd_float(0.0)
                for j_d in range(data.n_dofs_per_instance[()]):
                    value = value + (data.rigid_info.mass_mat[i_d, j_d, i_b] * data.constraint_state.search[j_d, i_b])
                data.constraint_state.grad[i_d, i_b] = value

    hessian_scale = system.h4
    if qd.static(displacement_coordinates):
        hessian_scale = 1.0
    for i_d, i_b in qd.ndrange(data.n_dofs_per_instance[()], data.n_instances[()]):
        i_global = data.dof_offset[()] + i_b * data.n_dofs_per_instance[()] + i_d
        y[i_global] = y[i_global] + hessian_scale * data.constraint_state.grad[i_d, i_b]

    for i_padding in range(data.n_dofs[()], data.n_storage_dofs[()]):
        i_global = data.dof_offset[()] + i_padding
        y[i_global] = y[i_global] + x[i_global]


@qd.func(requires_top_level=True)
def pcg_apply_preconditioner(system: qd.template(), residual: qd.template(), output: qd.template()):
    apply_preconditioner(system, residual, output, False)


@qd.func(requires_top_level=True)
def apply_preconditioner(
    system: qd.template(),
    residual: qd.template(),
    result: qd.template(),
    displacement_coordinates: qd.template(),
):
    data = system.data
    if qd.static(system.has_constraints):
        for i_d, i_b in qd.ndrange(data.n_dofs_per_instance[()], data.n_instances[()]):
            i_global = data.dof_offset[()] + i_b * data.n_dofs_per_instance[()] + i_d
            data.constraint_state.grad[i_d, i_b] = residual[i_global]

        for i_b in range(data.n_instances[()]):
            if data.constraint_state.n_constraints[i_b] > 0:
                for i_island in range(data.constraint_state.island.n_islands[i_b]):
                    solver.func_cholesky_solve_batch(
                        i_b,
                        i_island,
                        rhs=data.constraint_state.grad,
                        out=data.constraint_state.Mgrad,
                        constraint_state=data.constraint_state,
                        rigid_config=data.rigid_config,
                    )

        preconditioner_scale = 1.0 / system.h4
        if qd.static(displacement_coordinates):
            preconditioner_scale = 1.0
        for i_d, i_b in qd.ndrange(data.n_dofs_per_instance[()], data.n_instances[()]):
            i_global = data.dof_offset[()] + i_b * data.n_dofs_per_instance[()] + i_d
            if data.constraint_state.n_constraints[i_b] > 0:
                result[i_global] = preconditioner_scale * data.constraint_state.Mgrad[i_d, i_b]

    for i_padding in range(data.n_dofs[()], data.n_storage_dofs[()]):
        i_global = data.dof_offset[()] + i_padding
        result[i_global] = residual[i_global]


@qd.func(requires_top_level=True)
def negate_dq(
    system: qd.template(),
    global_linear_system_data: qd.template(),
    displacement_coordinates: qd.template(),
):
    data = system.data
    if qd.static(system.has_constraints):
        for i_d, i_b in qd.ndrange(data.n_dofs_per_instance[()], data.n_instances[()]):
            i_global = data.dof_offset[()] + i_b * data.n_dofs_per_instance[()] + i_d
            direction = global_linear_system_data.read_solution(i_global)
            if qd.static(displacement_coordinates):
                direction = direction / (system.h * system.h)
            data.constraint_state.search[i_d, i_b] = -direction
            data.constraint_state.Mgrad[i_d, i_b] = direction
        for i_b in range(data.n_instances[()]):
            if data.constraint_state.n_constraints[i_b] > 0:
                linesearch.func_mv_jv_dense(i_b, data.constraint_state, data.rigid_info)


@qd.func(requires_top_level=True)
def record_start_point(
    system: qd.template(),
):
    data = system.data
    if qd.static(system.has_constraints):
        for i_d, i_b in qd.ndrange(data.n_dofs_per_instance[()], data.n_instances[()]):
            data.qacc_temp[i_d, i_b] = data.constraint_state.qacc[i_d, i_b]
            data.Ma_temp[i_d, i_b] = data.constraint_state.Ma[i_d, i_b]
        for i_c, i_b in qd.ndrange(data.constraint_state.Jaref.shape[0], data.n_instances[()]):
            if i_c < data.constraint_state.n_constraints[i_b]:
                data.Jaref_temp[i_c, i_b] = data.constraint_state.Jaref[i_c, i_b]


@qd.func(requires_top_level=True)
def step_forward(system: qd.template(), alpha):
    data = system.data
    if qd.static(system.has_constraints):
        for i_d, i_b in qd.ndrange(data.n_dofs_per_instance[()], data.n_instances[()]):
            data.constraint_state.qacc[i_d, i_b] = (
                data.qacc_temp[i_d, i_b] + alpha * data.constraint_state.search[i_d, i_b]
            )
            data.constraint_state.Ma[i_d, i_b] = data.Ma_temp[i_d, i_b] + alpha * data.constraint_state.mv[i_d, i_b]
        for i_c, i_b in qd.ndrange(data.constraint_state.Jaref.shape[0], data.n_instances[()]):
            if i_c < data.constraint_state.n_constraints[i_b]:
                data.constraint_state.Jaref[i_c, i_b] = (
                    data.Jaref_temp[i_c, i_b] + alpha * data.constraint_state.jv[i_c, i_b]
                )

        solver.func_update_constraint(
            qacc=data.constraint_state.qacc,
            Ma=data.constraint_state.Ma,
            cost=data.constraint_state.cost,
            dyn_state=data.dyn_state,
            constraint_state=data.constraint_state,
            rigid_config=data.rigid_config,
        )
        set_newton_active(system, 1)
        solver.func_update_gradient_no_solve(
            data.dyn_state,
            data.constraint_state,
            data.dyn_info,
            data.rigid_info,
            data.rigid_config,
        )


@qd.func(requires_top_level=True)
def update_velocity(
    system: qd.template(),
):
    data = system.data
    if qd.static(system.has_constraints):
        solver.func_update_qacc(data.dyn_state, data.constraint_state, data.rigid_config, data.errno)
    func_step_2(
        data.dyn_state,
        data.constraint_state,
        data.dyn_info,
        data.rigid_info,
        data.rigid_config,
        False,
        data.errno,
    )
