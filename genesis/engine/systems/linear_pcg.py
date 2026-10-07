from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimData, SimSystem


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class LinearPCG(SimSystem):
    """Python lifecycle system for the graph-native linear PCG runtime."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible storage for the linear PCG algorithm."""

        total_dof: qd.Ndarray
        residual: qd.Ndarray
        preconditioned_residual: qd.Ndarray
        direction: qd.Ndarray
        operator_direction: qd.Ndarray
        dot_partials: qd.Ndarray
        residual_preconditioned: qd.Ndarray
        residual_preconditioned_initial: qd.Ndarray
        residual_preconditioned_next: qd.Ndarray
        direction_operator_direction: qd.Ndarray
        alpha: qd.Ndarray
        beta: qd.Ndarray
        condition: qd.Ndarray
        is_active: qd.Ndarray
        n_iterations: qd.Ndarray
        is_failed: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()

    def build(self) -> None:
        from .global_linear_system import GlobalLinearSystem
        from .pcg_solver import PCGSolver

        self.global_linear_system = self.require(GlobalLinearSystem)
        pcg_solver = self.require(PCGSolver)
        self.initialize_action = self.create_action(init_linear_pcg, self.data)
        self.solve_action = self.create_action(solve_linear_pcg, self.data)
        pcg_solver.on_solve(
            self.initialize_action,
            self.solve_action,
        )

    def init(self) -> None:
        capacity = self.global_linear_system.data.b_rhs.shape[0]

        def array(dtype, shape, values):
            result = qd.ndarray(dtype, shape=shape)
            result.from_numpy(values)
            return result

        zero_scalar_f64 = np.array(0.0, dtype=np.float64)
        zero_scalar_i32 = np.array(0, dtype=np.int32)
        self.data.total_dof = self.global_linear_system.data.total_dof
        self.data.residual = array(qd.f64, (capacity,), np.zeros(capacity, dtype=np.float64))
        self.data.preconditioned_residual = array(qd.f64, (capacity,), np.zeros(capacity, dtype=np.float64))
        self.data.direction = array(qd.f64, (capacity,), np.zeros(capacity, dtype=np.float64))
        self.data.operator_direction = array(qd.f64, (capacity,), np.zeros(capacity, dtype=np.float64))
        self.data.dot_partials = array(qd.f64, (64,), np.zeros(64, dtype=np.float64))
        self.data.residual_preconditioned = array(qd.f64, (), zero_scalar_f64)
        self.data.residual_preconditioned_initial = array(qd.f64, (), zero_scalar_f64)
        self.data.residual_preconditioned_next = array(qd.f64, (), zero_scalar_f64)
        self.data.direction_operator_direction = array(qd.f64, (), zero_scalar_f64)
        self.data.alpha = array(qd.f64, (), zero_scalar_f64)
        self.data.beta = array(qd.f64, (), zero_scalar_f64)
        self.data.condition = array(qd.i32, (), zero_scalar_i32)
        self.data.is_active = array(qd.i32, (), zero_scalar_i32)
        self.data.n_iterations = array(qd.i32, (), zero_scalar_i32)
        self.data.is_failed = array(qd.i32, (), zero_scalar_i32)


@qd.func(requires_top_level=True)
def init_linear_pcg(
    data: qd.template(),  # LinearPCG.Data
):
    """Reset the diagnostics of one preallocated linear PCG implementation."""
    for _ in range(1):
        data.n_iterations[()] = 0
        data.is_failed[()] = 0


@qd.func(requires_top_level=True)
def pcg_dot_rz(
    data: qd.template(),  # LinearPCG.Data
    lhs: qd.template(),  # qd.Ndarray
    rhs: qd.template(),  # qd.Ndarray
    output: qd.template(),  # qd.Ndarray
    gate_active: qd.template(),  # bool
):
    qd.loop_config(name="pcg_dot_rz_partial", block_dim=256)
    for thread in range(16384):
        value = qd.f64(0.0)
        index = thread
        while index < data.total_dof[()]:
            if qd.static(gate_active):
                if data.is_active[()] != 0:
                    value = value + lhs[index] * rhs[index]
            else:
                value = value + lhs[index] * rhs[index]
            index = index + 16384
        block_sum = qd.simt.block.reduce_add(value, 256, qd.f64)
        if thread % 256 == 0:
            data.dot_partials[thread // 256] = block_sum

    qd.loop_config(name="pcg_dot_rz_final", block_dim=64)
    for thread in range(64):
        value = qd.simt.block.reduce_add(
            data.dot_partials[thread],
            64,
            qd.f64,
        )
        if thread == 0:
            output[()] = value


@qd.func(requires_top_level=True)
def pcg_dot_pAp(
    data: qd.template(),  # LinearPCG.Data
):
    qd.loop_config(name="pcg_dot_pAp_partial", block_dim=256)
    for thread in range(16384):
        value = qd.f64(0.0)
        index = thread
        while index < data.total_dof[()]:
            if data.is_active[()] != 0:
                value = value + data.direction[index] * data.operator_direction[index]
            index = index + 16384
        block_sum = qd.simt.block.reduce_add(value, 256, qd.f64)
        if thread % 256 == 0:
            data.dot_partials[thread // 256] = block_sum

    qd.loop_config(name="pcg_dot_pAp_final", block_dim=64)
    for thread in range(64):
        value = qd.simt.block.reduce_add(
            data.dot_partials[thread],
            64,
            qd.f64,
        )
        if thread == 0:
            data.direction_operator_direction[()] = value


@qd.func(requires_top_level=True)
def initialize_linear_pcg(
    data: qd.template(),  # LinearPCG.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    action_provider: qd.template(),  # PCGSolver
):
    for i_d in range(data.total_dof[()]):
        linear_system_data.write_solution(i_d, qd.f64(0.0))
        data.residual[i_d] = linear_system_data.read_rhs(i_d)
        data.preconditioned_residual[i_d] = qd.f64(0.0)

    for action in qd.static(action_provider.preconditioner_actions):
        action.invoke((data.residual, data.preconditioned_residual))

    for i_d in range(data.total_dof[()]):
        data.direction[i_d] = data.preconditioned_residual[i_d]

    for _ in range(1):
        data.n_iterations[()] = 0
        data.is_failed[()] = 0
    pcg_dot_rz(
        data,
        data.residual,
        data.preconditioned_residual,
        data.residual_preconditioned,
        False,
    )

    for _ in range(1):
        data.residual_preconditioned_initial[()] = data.residual_preconditioned[()]
        data.condition[()] = 1
        data.is_active[()] = qd.i32(data.residual_preconditioned[()] > 0.0)


@qd.func(requires_top_level=True)
def iterate_linear_pcg(
    data: qd.template(),  # LinearPCG.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    action_provider: qd.template(),  # PCGSolver
    tolerance: qd.template(),  # float
    max_iterations: qd.template(),  # int
):
    for i_d in range(data.total_dof[()]):
        if data.is_active[()] != 0:
            data.operator_direction[i_d] = qd.f64(0.0)

    for action in qd.static(action_provider.operator_actions):
        action.invoke((linear_system_data, data.direction, data.operator_direction))

    pcg_dot_pAp(data)

    for _ in range(1):
        if data.is_active[()] != 0:
            if data.direction_operator_direction[()] > 0.0:
                data.alpha[()] = data.residual_preconditioned[()] / data.direction_operator_direction[()]
            else:
                data.alpha[()] = qd.f64(0.0)
                data.is_failed[()] = 1
                data.is_active[()] = 0

    for i_d in range(data.total_dof[()]):
        if data.is_active[()] != 0:
            linear_system_data.add_solution(i_d, data.alpha[()] * data.direction[i_d])
            data.residual[i_d] = data.residual[i_d] - data.alpha[()] * data.operator_direction[i_d]
            data.preconditioned_residual[i_d] = qd.f64(0.0)

    for action in qd.static(action_provider.preconditioner_actions):
        action.invoke((data.residual, data.preconditioned_residual))

    pcg_dot_rz(
        data,
        data.residual,
        data.preconditioned_residual,
        data.residual_preconditioned_next,
        True,
    )

    for _ in range(1):
        if data.is_active[()] != 0:
            is_converged = qd.abs(data.residual_preconditioned_next[()]) <= tolerance * qd.abs(
                data.residual_preconditioned_initial[()]
            )
            if is_converged:
                data.is_active[()] = 0
            elif qd.abs(data.residual_preconditioned[()]) > 0.0:
                data.beta[()] = data.residual_preconditioned_next[()] / data.residual_preconditioned[()]
                data.residual_preconditioned[()] = data.residual_preconditioned_next[()]
            else:
                data.is_failed[()] = 1
                data.is_active[()] = 0
        if data.is_failed[()] == 0 and data.residual_preconditioned_initial[()] > 0.0:
            data.n_iterations[()] = data.n_iterations[()] + 1
        if data.n_iterations[()] >= max_iterations and data.is_active[()] != 0:
            data.is_failed[()] = 1
            data.is_active[()] = 0
        data.condition[()] = data.is_active[()]

    for i_d in range(data.total_dof[()]):
        if data.is_active[()] != 0:
            data.direction[i_d] = data.preconditioned_residual[i_d] + data.beta[()] * data.direction[i_d]


@qd.func(requires_top_level=True)
def solve_linear_pcg(
    data: qd.template(),  # LinearPCG.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    action_provider: qd.template(),  # PCGSolver
    tolerance: qd.template(),  # float
    max_iterations: qd.template(),  # int
    max_pcg_iterations: qd.template(),  # qd.Ndarray
    total_pcg_iterations: qd.template(),  # qd.Ndarray
):
    initialize_linear_pcg(data, linear_system_data, action_provider)
    while qd.graph.do_while(data.condition):
        iterate_linear_pcg(
            data,
            linear_system_data,
            action_provider,
            tolerance,
            max_iterations,
        )
    for _ in range(1):
        max_pcg_iterations[()] = qd.max(
            max_pcg_iterations[()],
            data.n_iterations[()],
        )
        total_pcg_iterations[()] = total_pcg_iterations[()] + data.n_iterations[()]
