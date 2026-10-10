# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Time integration of the MochiSolver: multistep history, stage-start extrapolation (backward Euler and BDF2) of the
generalized coordinates and of the link velocities, and the finite-difference velocities recovered from the solved
configuration."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .data import INTEGRATOR, N_HISTORY, SOLVE_STATUS, MochiInfo, MochiState
from .lie import sym, vee, vsym_from_omega

# BDF2 extrapolation coefficient of the step before the previous one: x_start = x_-1 + BDF2_ALPHA_2 (x_-2 - x_-1).
BDF2_ALPHA_2 = -1.0 / 3.0
BDF2_BETA = 2.0 / 3.0


@qd.func
def func_read_quat(qpos: qd.Tensor, q_start, i_b):
    return qd.Vector(
        [qpos[q_start, i_b], qpos[q_start + 1, i_b], qpos[q_start + 2, i_b], qpos[q_start + 3, i_b]], dt=gs.qd_float
    )


@qd.func
def func_read_quat_hist(qpos: qd.Tensor, slot, q_start, i_b):
    return qd.Vector(
        [
            qpos[slot, q_start, i_b],
            qpos[slot, q_start + 1, i_b],
            qpos[slot, q_start + 2, i_b],
            qpos[slot, q_start + 3, i_b],
        ],
        dt=gs.qd_float,
    )


@qd.func
def func_quat_extrapolate(quat_1, quat_2, alpha, eps):
    """quat_1 composed with alpha times the body-frame rotation from quat_1 to quat_2."""
    rotvec = gu.qd_quat_to_rotvec(gu.qd_quat_mul(gu.qd_inv_quat(quat_1), quat_2), eps)
    return gu.qd_transform_quat_by_quat(gu.qd_rotvec_to_quat(alpha * rotvec, eps), quat_1)


@qd.func
def func_step_start(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """Shift the multistep history by one step and build the stage-start reference of the new step: the previous state
    for backward Euler, the two-step extrapolation for BDF2, which is also the warm start of the Newton solve."""
    n_qs = rigid_info.qpos.shape[0]
    n_dofs = dyn_state.dofs.vel.shape[0]
    n_links = dyn_state.links.pos.shape[0]
    n_joints = dyn_info.joints.type.shape[0]
    _B = mochi_state.n_hist.shape[0]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.n_hist[i_b] = qd.min(mochi_state.n_hist[i_b] + 1, N_HISTORY)
        beta = gs.qd_float(1.0)
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):  # noqa: SIM102
            if mochi_state.n_hist[i_b] >= 2:
                beta = BDF2_BETA
        mochi_state.dt_stage[i_b] = beta * mochi_info.dt[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_slot in qd.ndrange(n_qs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_qs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.qpos_prev[1, i_q, i_b] = mochi_state.qpos_prev[0, i_q, i_b]
        mochi_state.qpos_prev[0, i_q, i_b] = rigid_info.qpos[i_q, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.dofs_vel_prev[1, i_d, i_b] = mochi_state.dofs_vel_prev[0, i_d, i_b]
        mochi_state.dofs_vel_prev[0, i_d, i_b] = dyn_state.dofs.vel[i_d, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.links_vel_prev[1, i_l, i_b] = mochi_state.links_vel_prev[0, i_l, i_b]
        mochi_state.links_ang_prev[1, i_l, i_b] = mochi_state.links_ang_prev[0, i_l, i_b]
        mochi_state.links_vsym_prev[1, i_l, i_b] = mochi_state.links_vsym_prev[0, i_l, i_b]
        mochi_state.links_vel_prev[0, i_l, i_b] = mochi_state.links_vel[i_l, i_b]
        mochi_state.links_ang_prev[0, i_l, i_b] = mochi_state.links_ang[i_l, i_b]
        mochi_state.links_vsym_prev[0, i_l, i_b] = mochi_state.links_vsym[i_l, i_b]

    # Step-start generalized coordinates, joint by joint: linear extrapolation of the coordinates of scalar joints
    # and of free-joint translations, geodesic extrapolation of the quaternions.
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_slot in qd.ndrange(n_joints, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_joints, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]
        q_start = dyn_info.joints.q_start[I_j]
        q_end = dyn_info.joints.q_end[I_j]
        is_bdf2 = False
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):
            is_bdf2 = mochi_state.n_hist[i_b] >= 2
        for i_q in range(q_start, q_end):
            x1 = mochi_state.qpos_prev[0, i_q, i_b]
            value = x1
            if is_bdf2:
                value = x1 + BDF2_ALPHA_2 * (mochi_state.qpos_prev[1, i_q, i_b] - x1)
            mochi_state.qpos_step_start[i_q, i_b] = value
        if is_bdf2 and (joint_type == gs.JOINT_TYPE.FREE or joint_type == gs.JOINT_TYPE.SPHERICAL):
            rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
            quat_1 = func_read_quat_hist(mochi_state.qpos_prev, 0, q_start + rot_offset, i_b)
            quat_2 = func_read_quat_hist(mochi_state.qpos_prev, 1, q_start + rot_offset, i_b)
            quat_0 = func_quat_extrapolate(quat_1, quat_2, BDF2_ALPHA_2, EPS)
            for k in qd.static(range(4)):
                mochi_state.qpos_step_start[q_start + rot_offset + k, i_b] = quat_0[k]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_slot in qd.ndrange(n_qs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_qs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        # Single-stage schemes: the stage starts at the step start, which is also the warm start of the solve.
        mochi_state.qpos_stage_start[i_q, i_b] = mochi_state.qpos_step_start[i_q, i_b]
        rigid_info.qpos[i_q, i_b] = mochi_state.qpos_step_start[i_q, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        v1 = mochi_state.dofs_vel_prev[0, i_d, i_b]
        value = v1
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):  # noqa: SIM102
            if mochi_state.n_hist[i_b] >= 2:
                value = v1 + BDF2_ALPHA_2 * (mochi_state.dofs_vel_prev[1, i_d, i_b] - v1)
        mochi_state.dofs_vel_stage_start[i_d, i_b] = value

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        vel_1 = mochi_state.links_vel_prev[0, i_l, i_b]
        ang_1 = mochi_state.links_ang_prev[0, i_l, i_b]
        vsym_1 = mochi_state.links_vsym_prev[0, i_l, i_b]
        vel = vel_1
        ang = ang_1
        vsym = vsym_1
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):  # noqa: SIM102
            if mochi_state.n_hist[i_b] >= 2:
                vel = vel_1 + BDF2_ALPHA_2 * (mochi_state.links_vel_prev[1, i_l, i_b] - vel_1)
                ang = ang_1 + BDF2_ALPHA_2 * (mochi_state.links_ang_prev[1, i_l, i_b] - ang_1)
                vsym = vsym_1 + BDF2_ALPHA_2 * (mochi_state.links_vsym_prev[1, i_l, i_b] - vsym_1)
        mochi_state.links_vel_stage_start[i_l, i_b] = vel
        mochi_state.links_ang_stage_start[i_l, i_b] = ang
        mochi_state.links_vsym_stage_start[i_l, i_b] = vsym


@qd.kernel
def kernel_step_start(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    func_step_start(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        rigid_info,
        mochi_info,
        mochi_state,
        rigid_config,
        mochi_config,
    )


@qd.func
def func_store_stage_start_poses(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Snapshot the link and geom poses of the stage-start configuration (forward kinematics must have been run on
    the stage-start coordinates)."""
    n_links = dyn_state.links.pos.shape[0]
    n_geoms = dyn_state.geoms.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.links_pos_stage_start[i_l, i_b] = dyn_state.links.pos[i_l, i_b]
        mochi_state.links_quat_stage_start[i_l, i_b] = dyn_state.links.quat[i_l, i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_slot in qd.ndrange(n_geoms, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_geoms, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.geoms_pos_stage_start[i_g, i_b] = dyn_state.geoms.pos[i_g, i_b]
        mochi_state.geoms_quat_stage_start[i_g, i_b] = dyn_state.geoms.quat[i_g, i_b]


@qd.kernel
def kernel_store_stage_start_poses(
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    func_store_stage_start_poses(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        mochi_state,
        rigid_config,
    )


@qd.func
def func_post_stage(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Recover the velocities by finite differences over the stage: per degree of freedom (in the tangent space of
    its joint, so that the kinematic solver's velocity conventions hold), and per link (the angular velocity is the
    antisymmetric part of (R - R_start)/h R^T and the symmetric part is kept so the next step extrapolates the same
    rotation increment). A diverged environment is reset to its previous configuration at rest."""
    n_joints = dyn_info.joints.type.shape[0]
    n_links = dyn_state.links.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_slot in qd.ndrange(n_joints, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_joints, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]
        if joint_type == gs.JOINT_TYPE.FIXED:
            continue
        q_start = dyn_info.joints.q_start[I_j]
        q_end = dyn_info.joints.q_end[I_j]
        dof_start = dyn_info.joints.dof_start[I_j]
        if mochi_state.status[i_b] == SOLVE_STATUS.DIVERGED:
            for i_q in range(q_start, q_end):
                rigid_info.qpos[i_q, i_b] = mochi_state.qpos_prev[0, i_q, i_b]
            for i_d in range(dof_start, dyn_info.joints.dof_end[I_j]):
                dyn_state.dofs.vel[i_d, i_b] = 0.0
            continue
        h = mochi_state.dt_stage[i_b]
        if joint_type == gs.JOINT_TYPE.FREE or joint_type == gs.JOINT_TYPE.SPHERICAL:
            rot_offset = 0
            if joint_type == gs.JOINT_TYPE.FREE:
                rot_offset = 3
                for k in qd.static(range(3)):
                    dyn_state.dofs.vel[dof_start + k, i_b] = (
                        rigid_info.qpos[q_start + k, i_b] - mochi_state.qpos_stage_start[q_start + k, i_b]
                    ) / h
            quat = func_read_quat(rigid_info.qpos, q_start + rot_offset, i_b)
            quat_start = func_read_quat(mochi_state.qpos_stage_start, q_start + rot_offset, i_b)
            # Body-frame angular velocity, the tangent space of the quaternion degrees of freedom.
            rotvec = gu.qd_quat_to_rotvec(gu.qd_quat_mul(gu.qd_inv_quat(quat_start), quat), EPS)
            # Sine-based finite-difference angular velocity, vee(((R - R_ss) / h) R^T) expressed in the joint frame,
            # so that joint velocities match the rigid-body history (mochi convention).
            angle = rotvec.norm()
            sinc = gs.qd_float(1.0)
            if angle > EPS:
                sinc = qd.sin(angle) / angle
            for k in qd.static(range(3)):
                dyn_state.dofs.vel[dof_start + rot_offset + k, i_b] = rotvec[k] * sinc / h
        else:
            for i_q_ in range(q_end - q_start):
                dq = rigid_info.qpos[q_start + i_q_, i_b] - mochi_state.qpos_stage_start[q_start + i_q_, i_b]
                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    # Sine-based finite difference of the joint rotation, consistent with the angular velocities.
                    dq = qd.sin(dq)
                dyn_state.dofs.vel[dof_start + i_q_, i_b] = dq / h

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_info.links.is_dynamic[i_l]:
            continue
        if mochi_state.status[i_b] == SOLVE_STATUS.DIVERGED:
            mochi_state.links_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
            mochi_state.links_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
            mochi_state.links_vsym[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            continue
        h = mochi_state.dt_stage[i_b]
        # The link poses still hold the solved configuration (forward kinematics ran on the accepted iterate).
        pos = dyn_state.links.pos[i_l, i_b]
        R = gu.qd_quat_to_R(dyn_state.links.quat[i_l, i_b], EPS)
        R_start = gu.qd_quat_to_R(mochi_state.links_quat_stage_start[i_l, i_b], EPS)
        F = ((R - R_start) / h) @ R.transpose()
        # The translational history is the finite-difference velocity of the center of mass, the point at which
        # the inertia is assembled (it differs from the link-origin velocity by (skew(w) + S) r_c).
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        com_local = dyn_info.links.inertial_pos[I_l]
        pos_c = pos + R @ com_local
        pos_c_start = mochi_state.links_pos_stage_start[i_l, i_b] + R_start @ com_local
        mochi_state.links_vel[i_l, i_b] = (pos_c - pos_c_start) / h
        mochi_state.links_ang[i_l, i_b] = vee(F)
        mochi_state.links_vsym[i_l, i_b] = sym(F)


@qd.kernel
def kernel_post_stage(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    func_post_stage(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        rigid_info,
        mochi_info,
        mochi_state,
        rigid_config,
    )


@qd.kernel
def kernel_reset_history(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Invalidate the multistep history of the given environments (their next step is a backward Euler step) and
    rebuild the link velocities from the kinematic solver's velocity state, as required after the state has been set
    from outside the solver."""
    n_links = dyn_state.links.pos.shape[0]
    dt = mochi_info.dt[None]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        mochi_state.n_hist[envs_idx[i_b_]] = 0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b_ in qd.ndrange(n_links, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        vel = qd.Vector.zero(gs.qd_float, 3)
        ang = qd.Vector.zero(gs.qd_float, 3)
        S = qd.Matrix.zero(gs.qd_float, 3, 3)
        if mochi_info.links.is_dynamic[i_l]:
            ang = dyn_state.links.cd_ang[i_l, i_b]
            vel = dyn_state.links.cd_vel[i_l, i_b] + ang.cross(dyn_state.links.i_pos[i_l, i_b])
            S = vsym_from_omega(ang, dt, EPS)
        mochi_state.links_vel[i_l, i_b] = vel
        mochi_state.links_ang[i_l, i_b] = ang
        mochi_state.links_vsym[i_l, i_b] = S
