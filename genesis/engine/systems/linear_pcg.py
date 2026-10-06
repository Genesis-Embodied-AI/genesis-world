from __future__ import annotations

import numpy as np
import quadrants as qd


@qd.data_oriented
class LinearPCG:
    """Internal standard-PCG algorithm."""

    def __init__(self, total_dof: int) -> None:
        self.dof_capacity = total_dof
        n_storage_dofs = max(total_dof, 1)

        self.total_dof = qd.ndarray(qd.i32, shape=())
        self.residual = qd.ndarray(qd.f64, shape=(n_storage_dofs,))
        self.preconditioned_residual = qd.ndarray(qd.f64, shape=(n_storage_dofs,))
        self.direction = qd.ndarray(qd.f64, shape=(n_storage_dofs,))
        self.operator_direction = qd.ndarray(qd.f64, shape=(n_storage_dofs,))
        self.dot_partials = qd.ndarray(qd.f64, shape=(64,))

        self.residual_preconditioned = qd.ndarray(qd.f64, shape=())
        self.residual_preconditioned_initial = qd.ndarray(qd.f64, shape=())
        self.residual_preconditioned_next = qd.ndarray(qd.f64, shape=())
        self.direction_operator_direction = qd.ndarray(qd.f64, shape=())
        self.alpha = qd.ndarray(qd.f64, shape=())
        self.beta = qd.ndarray(qd.f64, shape=())
        self.condition = qd.ndarray(qd.i32, shape=())
        self.is_active = qd.ndarray(qd.i32, shape=())
        self.n_iterations = qd.ndarray(qd.i32, shape=())
        self.is_failed = qd.ndarray(qd.i32, shape=())
        self.total_dof.from_numpy(np.array(total_dof, dtype=np.int32))
        self.dot_partials.from_numpy(np.zeros(64, dtype=np.float64))

    @qd.func(requires_top_level=True)
    def pcg_dot_rz(
        self,
        lhs: qd.template(),
        rhs: qd.template(),
        output: qd.template(),
        gate_active: qd.template(),
    ):
        qd.loop_config(name="pcg_dot_rz_partial", block_dim=256)
        for thread in range(16384):
            value = qd.f64(0.0)
            index = thread
            while index < self.total_dof[()]:
                if qd.static(not gate_active) or self.is_active[()] != 0:
                    value = value + lhs[index] * rhs[index]
                index = index + 16384
            block_sum = qd.simt.block.reduce_add(value, 256, qd.f64)
            if thread % 256 == 0:
                self.dot_partials[thread // 256] = block_sum

        qd.loop_config(name="pcg_dot_rz_final", block_dim=64)
        for thread in range(64):
            value = qd.simt.block.reduce_add(
                self.dot_partials[thread],
                64,
                qd.f64,
            )
            if thread == 0:
                output[()] = value

    @qd.func(requires_top_level=True)
    def pcg_dot_pAp(self):
        qd.loop_config(name="pcg_dot_pAp_partial", block_dim=256)
        for thread in range(16384):
            value = qd.f64(0.0)
            index = thread
            while index < self.total_dof[()]:
                if self.is_active[()] != 0:
                    value = value + self.direction[index] * self.operator_direction[index]
                index = index + 16384
            block_sum = qd.simt.block.reduce_add(value, 256, qd.f64)
            if thread % 256 == 0:
                self.dot_partials[thread // 256] = block_sum

        qd.loop_config(name="pcg_dot_pAp_final", block_dim=64)
        for thread in range(64):
            value = qd.simt.block.reduce_add(
                self.dot_partials[thread],
                64,
                qd.f64,
            )
            if thread == 0:
                self.direction_operator_direction[()] = value

    @qd.func(requires_top_level=True)
    def initialize(
        self,
        linear_system: qd.template(),
        rigid: qd.template(),
        rigid_forest: qd.template(),
        fem_preconditioner: qd.template(),
        has_rigid: qd.template(),
        has_rigid_forest: qd.template(),
        has_fem: qd.template(),
    ):
        for i_d in range(self.total_dof[()]):
            linear_system.x_sol[i_d] = qd.f64(0.0)
            self.residual[i_d] = linear_system.b_rhs[i_d]
            self.preconditioned_residual[i_d] = qd.f64(0.0)

        if qd.static(has_rigid_forest):
            rigid_forest.apply_preconditioner(
                self.residual,
                self.preconditioned_residual,
            )
        elif qd.static(has_rigid):
            rigid.apply_preconditioner(
                self.residual,
                self.preconditioned_residual,
                False,
            )
        if qd.static(has_fem):
            fem_preconditioner.apply(self.residual, self.preconditioned_residual)

        for i_d in range(self.total_dof[()]):
            self.direction[i_d] = self.preconditioned_residual[i_d]

        for _ in range(1):
            self.n_iterations[()] = 0
            self.is_failed[()] = 0
        self.pcg_dot_rz(
            self.residual,
            self.preconditioned_residual,
            self.residual_preconditioned,
            False,
        )

        for _ in range(1):
            self.residual_preconditioned_initial[()] = self.residual_preconditioned[()]
            self.condition[()] = 1
            self.is_active[()] = qd.i32(self.residual_preconditioned[()] > 0.0)

    @qd.func(requires_top_level=True)
    def iteration(
        self,
        linear_system: qd.template(),
        rigid: qd.template(),
        rigid_forest: qd.template(),
        fem_preconditioner: qd.template(),
        has_rigid: qd.template(),
        has_rigid_forest: qd.template(),
        has_fem: qd.template(),
        tolerance: qd.template(),
        max_iterations: qd.template(),
    ):
        for i_d in range(self.total_dof[()]):
            if self.is_active[()] != 0:
                self.operator_direction[i_d] = qd.f64(0.0)
                if qd.static(has_rigid_forest):
                    rigid_forest.physical_Ap[i_d] = qd.f64(0.0)

        if qd.static(has_rigid_forest):
            rigid_forest.prepare_physical_direction(self.direction)
            linear_system.spmv(rigid_forest.physical_p, rigid_forest.physical_Ap)
            rigid_forest.finish_reduced_spmv(
                self.direction,
                self.operator_direction,
            )
        else:
            linear_system.spmv(self.direction, self.operator_direction)
        if qd.static(has_rigid):
            rigid.apply_hessian(
                self.direction,
                self.operator_direction,
                has_rigid_forest,
            )

        self.pcg_dot_pAp()

        for _ in range(1):
            if self.is_active[()] != 0:
                if self.direction_operator_direction[()] > 0.0:
                    self.alpha[()] = self.residual_preconditioned[()] / self.direction_operator_direction[()]
                else:
                    self.alpha[()] = qd.f64(0.0)
                    self.is_failed[()] = 1
                    self.is_active[()] = 0

        for i_d in range(self.total_dof[()]):
            if self.is_active[()] != 0:
                linear_system.x_sol[i_d] = linear_system.x_sol[i_d] + self.alpha[()] * self.direction[i_d]
                self.residual[i_d] = self.residual[i_d] - self.alpha[()] * self.operator_direction[i_d]
                self.preconditioned_residual[i_d] = qd.f64(0.0)

        if qd.static(has_rigid_forest):
            rigid_forest.apply_preconditioner(
                self.residual,
                self.preconditioned_residual,
            )
        elif qd.static(has_rigid):
            rigid.apply_preconditioner(
                self.residual,
                self.preconditioned_residual,
                False,
            )
        if qd.static(has_fem):
            fem_preconditioner.apply(self.residual, self.preconditioned_residual)

        self.pcg_dot_rz(
            self.residual,
            self.preconditioned_residual,
            self.residual_preconditioned_next,
            True,
        )

        for _ in range(1):
            if self.is_active[()] != 0:
                is_converged = qd.abs(self.residual_preconditioned_next[()]) <= tolerance * qd.abs(
                    self.residual_preconditioned_initial[()]
                )
                if is_converged:
                    self.is_active[()] = 0
                elif qd.abs(self.residual_preconditioned[()]) > 0.0:
                    self.beta[()] = self.residual_preconditioned_next[()] / self.residual_preconditioned[()]
                    self.residual_preconditioned[()] = self.residual_preconditioned_next[()]
                else:
                    self.is_failed[()] = 1
                    self.is_active[()] = 0
            if self.is_failed[()] == 0 and self.residual_preconditioned_initial[()] > 0.0:
                self.n_iterations[()] = self.n_iterations[()] + 1
            if self.n_iterations[()] >= max_iterations and self.is_active[()] != 0:
                self.is_failed[()] = 1
                self.is_active[()] = 0
            self.condition[()] = self.is_active[()]

        for i_d in range(self.total_dof[()]):
            if self.is_active[()] != 0:
                self.direction[i_d] = self.preconditioned_residual[i_d] + self.beta[()] * self.direction[i_d]
