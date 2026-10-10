# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Per-environment control of the damped Newton iterations: residual norms, line search and convergence."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .data import LINEAR_TOLERANCE, LINESEARCH, SOLVE_STATUS, MochiInfo, MochiIslandState, MochiState

# Forcing term of the adaptive linear tolerance ('choice 2' of Eisenstat and Walker): the tolerance of the next linear
# solve is EW_GAMMA * (res / res_prev) ** EW_ALPHA, capped at EW_MAX_ETA (also the tolerance of the first iteration,
# where no ratio exists yet) and floored at EW_MIN_ETA_EPS times the machine epsilon so it stays above the noise
# floor. The cap is much tighter than the textbook 0.9 because an assembly costs more than a linear solve here.
EW_GAMMA = 0.9
EW_ALPHA = 1.618033988749895
EW_MAX_ETA = 0.01
EW_MIN_ETA_EPS = 100.0


@qd.func
def func_is_env_active(i_b, mochi_state: MochiState, skip_ls_done):
    """Whether the Newton solve of an environment is still running, optionally excluding the environments whose line
    search has already accepted an iterate."""
    is_active = mochi_state.is_active[i_b]
    if skip_ls_done:
        is_active = is_active and not mochi_state.ls_is_done[i_b]
    return is_active


@qd.func
def func_reset_newton(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.is_active[i_b] = True
        mochi_state.status[i_b] = SOLVE_STATUS.RUNNING
        mochi_state.n_iter[i_b] = 0
        mochi_state.n_pcg_iter[i_b] = 0
        mochi_state.ls_alpha[i_b] = 1.0
        mochi_state.ls_is_done[i_b] = False
        mochi_state.res_norm0[i_b] = 0.0


@qd.kernel
def kernel_reset_newton(mochi_state: MochiState, rigid_config: qd.template()):
    func_reset_newton(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        rigid_config,
    )


@qd.func
def func_residual_norms(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    skip_ls_done,
):
    """Plain and convergence-weighted squared norms of the residual of every running environment, the weighted one
    also per entity."""
    n_dofs = mochi_state.res.shape[0]
    n_nodes = island_state.nodes_res_w_sq.shape[0]
    _B = mochi_state.res.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            mochi_state.res_norm_sq[i_b] = 0.0
            mochi_state.res_w_sq[i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_n, i_slot in qd.ndrange(n_nodes, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_nodes, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            island_state.nodes_res_w_sq[i_n, i_b] = 0.0
    qd.loop_config(
        # A per-environment scalar accumulation contends on one cache line per environment: keep it
        # serial on the CPU backend unless environments themselves are the parallel axis.
        serialize=qd.static(
            rigid_config.para_level < gs.PARA_LEVEL.PARTIAL
            or (rigid_config.backend == gs.cpu and rigid_config.para_level < gs.PARA_LEVEL.ALL)
        )
    )
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            r = mochi_state.res[i_d, i_b]
            r_w_sq = mochi_state.conv_w[i_d, i_b] * r * r
            qd.atomic_add(mochi_state.res_norm_sq[i_b], r * r)
            qd.atomic_add(mochi_state.res_w_sq[i_b], r_w_sq)
            qd.atomic_add(island_state.nodes_res_w_sq[island_state.dofs_node[i_d], i_b], r_w_sq)


@qd.kernel
def kernel_residual_norms(
    mochi_state: MochiState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    skip_ls_done: qd.i32,
):
    func_residual_norms(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_state,
        island_state,
        rigid_config,
        skip_ls_done,
    )


@qd.func
def func_store_initial_norms(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    rigid_info: array_class.RigidInfo,
    mochi_state: MochiState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
):
    """Record the residual norms of the initial iterate, the reference of the relative convergence and divergence
    tests, and take it as the first line search reference."""
    n_qs = rigid_info.qpos.shape[0]
    n_nodes = island_state.nodes_res_w_sq.shape[0]
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.res_norm0[i_b] = qd.sqrt(mochi_state.res_norm_sq[i_b])
        mochi_state.ls_ref_norm_sq[i_b] = mochi_state.res_norm_sq[i_b]
        mochi_state.obj_ref[i_b] = mochi_state.obj[i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_n, i_slot in qd.ndrange(n_nodes, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_nodes, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        island_state.nodes_res_norm0_w[i_n, i_b] = qd.sqrt(island_state.nodes_res_w_sq[i_n, i_b])
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_slot in qd.ndrange(n_qs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_qs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.qpos_ls_ref[i_q, i_b] = rigid_info.qpos[i_q, i_b]


@qd.kernel
def kernel_store_initial_norms(
    rigid_info: array_class.RigidInfo,
    mochi_state: MochiState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
):
    func_store_initial_norms(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        rigid_info,
        mochi_state,
        island_state,
        rigid_config,
    )


@qd.func
def func_linesearch_begin(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    rigid_info: array_class.RigidInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Open a line search along the freshly solved Newton step: full step first, the current (accepted) iterate as
    reference, and the directional derivative of the objective along the step for the Armijo rule."""
    n_qs = rigid_info.qpos.shape[0]
    n_dofs = mochi_state.res.shape[0]
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.is_active[i_b]:
            mochi_state.ls_alpha[i_b] = 1.0
            mochi_state.ls_is_done[i_b] = False
            mochi_state.ls_ref_norm_sq[i_b] = mochi_state.res_norm_sq[i_b]
            mochi_state.obj_ref[i_b] = mochi_state.obj[i_b]
            mochi_state.ls_slope[i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_slot in qd.ndrange(n_qs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_qs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.is_active[i_b]:
            mochi_state.qpos_ls_ref[i_q, i_b] = rigid_info.qpos[i_q, i_b]
    qd.loop_config(
        # A per-environment scalar accumulation contends on one cache line per environment: keep it
        # serial on the CPU backend unless environments themselves are the parallel axis.
        serialize=qd.static(
            rigid_config.para_level < gs.PARA_LEVEL.PARTIAL
            or (rigid_config.backend == gs.cpu and rigid_config.para_level < gs.PARA_LEVEL.ALL)
        )
    )
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.is_active[i_b]:
            # The step taken is -dx, so the slope of the objective along it is -res . dx.
            qd.atomic_add(mochi_state.ls_slope[i_b], -mochi_state.res[i_d, i_b] * mochi_state.dx[i_d, i_b])


@qd.kernel
def kernel_linesearch_begin(
    rigid_info: array_class.RigidInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    func_linesearch_begin(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        rigid_info,
        mochi_state,
        rigid_config,
    )


@qd.func
def func_apply_increment(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Move every joint of the environments still searching to the trial iterate: the line search reference minus the
    scaled Newton step, composed in the tangent space of the joint (body-frame rotation increment for the quaternion
    coordinates of free and spherical joints, as the kinematic solver's velocity convention)."""
    n_joints = dyn_info.joints.type.shape[0]
    _B = mochi_state.is_active.shape[0]
    EPS = mochi_info.EPS[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_slot in qd.ndrange(n_joints, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_joints, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, True):
            continue
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]
        if joint_type == gs.JOINT_TYPE.FIXED:
            continue
        q_start = dyn_info.joints.q_start[I_j]
        q_end = dyn_info.joints.q_end[I_j]
        dof_start = dyn_info.joints.dof_start[I_j]
        alpha = mochi_state.ls_alpha[i_b]
        if joint_type == gs.JOINT_TYPE.FREE or joint_type == gs.JOINT_TYPE.SPHERICAL:
            rot_offset = 0
            if joint_type == gs.JOINT_TYPE.FREE:
                rot_offset = 3
                for k in qd.static(range(3)):
                    rigid_info.qpos[q_start + k, i_b] = (
                        mochi_state.qpos_ls_ref[q_start + k, i_b] - alpha * mochi_state.dx[dof_start + k, i_b]
                    )
            rotvec = -alpha * qd.Vector(
                [
                    mochi_state.dx[dof_start + rot_offset, i_b],
                    mochi_state.dx[dof_start + rot_offset + 1, i_b],
                    mochi_state.dx[dof_start + rot_offset + 2, i_b],
                ],
                dt=gs.qd_float,
            )
            quat_ref = qd.Vector(
                [
                    mochi_state.qpos_ls_ref[q_start + rot_offset, i_b],
                    mochi_state.qpos_ls_ref[q_start + rot_offset + 1, i_b],
                    mochi_state.qpos_ls_ref[q_start + rot_offset + 2, i_b],
                    mochi_state.qpos_ls_ref[q_start + rot_offset + 3, i_b],
                ],
                dt=gs.qd_float,
            )
            quat = gu.qd_transform_quat_by_quat(gu.qd_rotvec_to_quat(rotvec, EPS), quat_ref)
            for k in qd.static(range(4)):
                rigid_info.qpos[q_start + rot_offset + k, i_b] = quat[k]
        else:
            for i_q_ in range(q_end - q_start):
                rigid_info.qpos[q_start + i_q_, i_b] = (
                    mochi_state.qpos_ls_ref[q_start + i_q_, i_b] - alpha * mochi_state.dx[dof_start + i_q_, i_b]
                )


@qd.kernel
def kernel_apply_increment(
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    func_apply_increment(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_info,
        rigid_info,
        mochi_info,
        mochi_state,
        rigid_config,
    )


@qd.func
def func_linesearch_decide(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    is_last,
):
    """Accept the trial iterate of every searching environment when it improves on the reference (or on the last
    trial regardless, so that the solve always progresses), else halve the step."""
    n_qs = rigid_info.qpos.shape[0]
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, True):
            continue
        is_improved = True
        if qd.static(mochi_config.linesearch_type == LINESEARCH.RESIDUAL_NORM):
            is_improved = mochi_state.res_norm_sq[i_b] <= mochi_state.ls_ref_norm_sq[i_b]
        elif qd.static(mochi_config.linesearch_type == LINESEARCH.ARMIJO):
            is_improved = mochi_state.obj[i_b] <= (
                mochi_state.obj_ref[i_b]
                + mochi_info.linesearch_wolfe1[None] * mochi_state.ls_alpha[i_b] * mochi_state.ls_slope[i_b]
            )
        if is_improved or is_last:
            mochi_state.ls_is_done[i_b] = True
            mochi_state.ls_ref_norm_sq[i_b] = mochi_state.res_norm_sq[i_b]
            mochi_state.obj_ref[i_b] = mochi_state.obj[i_b]
        else:
            mochi_state.ls_alpha[i_b] = mochi_state.ls_alpha[i_b] * mochi_info.linesearch_alpha[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_slot in qd.ndrange(n_qs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_qs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.is_active[i_b] and mochi_state.ls_is_done[i_b]:
            mochi_state.qpos_ls_ref[i_q, i_b] = rigid_info.qpos[i_q, i_b]


@qd.kernel
def kernel_linesearch_decide(
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    is_last: qd.template(),
):
    func_linesearch_decide(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        rigid_info,
        mochi_info,
        mochi_state,
        rigid_config,
        mochi_config,
        is_last,
    )


@qd.func
def func_convergence_check(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    increment_iter,
    errno: qd.Tensor,
):
    """Classify every running environment from the residual of its accepted iterate: converged when every entity's
    weighted norm is within the absolute tolerance or the relative tolerance of its initial value (mochi's per-actor
    test), diverged on residual blow-up of the environment, stopped at the iteration budget."""
    n_nodes = island_state.nodes_res_w_sq.shape[0]
    _B = mochi_state.is_active.shape[0]
    abs_tol = mochi_info.newton_abs_tol[None]
    rel_tol = mochi_info.newton_rel_tol[None]
    div_abs_tol = mochi_info.explosion_abs_tol[None]
    div_rel_tol = mochi_info.explosion_rel_tol[None]
    max_iter = mochi_info.n_newton_iterations[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.is_active[i_b]:
            continue
        if increment_iter:
            mochi_state.n_iter[i_b] = mochi_state.n_iter[i_b] + 1
        res_norm = qd.sqrt(mochi_state.res_norm_sq[i_b])
        res_norm_w = qd.sqrt(mochi_state.res_w_sq[i_b])
        is_diverged = qd.math.isnan(res_norm)
        if div_abs_tol > 0.0:
            is_diverged |= res_norm > div_abs_tol
            is_diverged |= (res_norm > div_rel_tol * mochi_state.res_norm0[i_b]) and (res_norm_w > abs_tol)
        is_converged = True
        for i_n in range(n_nodes):
            node_norm_w = qd.sqrt(island_state.nodes_res_w_sq[i_n, i_b])
            node_norm0_w = island_state.nodes_res_norm0_w[i_n, i_b]
            if not (node_norm_w <= abs_tol or (node_norm0_w > 0.0 and node_norm_w <= rel_tol * node_norm0_w)):
                is_converged = False
        if is_diverged:
            mochi_state.status[i_b] = SOLVE_STATUS.DIVERGED
            mochi_state.is_active[i_b] = False
            qd.atomic_or(errno[i_b], array_class.ErrorCode.MOCHI_DIVERGED)
        elif is_converged:
            mochi_state.status[i_b] = SOLVE_STATUS.CONVERGED
            mochi_state.is_active[i_b] = False
        elif mochi_state.n_iter[i_b] >= max_iter:
            mochi_state.status[i_b] = SOLVE_STATUS.STOPPED
            mochi_state.is_active[i_b] = False


@qd.kernel
def kernel_convergence_check(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    increment_iter: qd.template(),
    errno: qd.Tensor,
):
    func_convergence_check(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        mochi_state,
        island_state,
        rigid_config,
        increment_iter,
        errno,
    )


@qd.func
def func_update_linear_tolerance(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """Relative tolerance the linear solve of the coming Newton step must reach in every running environment."""
    _B = mochi_state.is_active.shape[0]
    EPS = mochi_info.EPS[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.is_active[i_b]:
            continue
        if qd.static(mochi_config.linear_tolerance == LINEAR_TOLERANCE.ADAPTIVE):
            res_norm = qd.sqrt(mochi_state.res_norm_sq[i_b])
            eta = gs.qd_float(EW_MAX_ETA)
            if mochi_state.n_iter[i_b] > 0:
                eta = EW_GAMMA * qd.pow(res_norm / mochi_state.res_norm_prev[i_b], EW_ALPHA)
                eta = qd.min(qd.max(eta, EW_MIN_ETA_EPS * EPS), EW_MAX_ETA)
            mochi_state.pcg_rel_tol[i_b] = eta
            mochi_state.res_norm_prev[i_b] = res_norm
        else:
            mochi_state.pcg_rel_tol[i_b] = mochi_info.pcg_rel_tol[None]


@qd.kernel
def kernel_update_linear_tolerance(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    func_update_linear_tolerance(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        mochi_state,
        rigid_config,
        mochi_config,
    )


@qd.kernel
def kernel_any_active(mochi_state: MochiState, skip_ls_done: qd.template()) -> qd.i32:
    n_active = 0
    for i_b in range(mochi_state.is_active.shape[0]):
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            n_active += 1
    return n_active
