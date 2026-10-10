# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Articulated bodies: projection of the link-space residual and Hessian blocks onto the degrees of freedom through
the link Jacobians (Gauss-Newton, the second derivative of the kinematic map is dropped), joint-space terms (limits,
damping, stiffness, armature, drives) and the convergence weights of the degrees of freedom."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .data import MochiInfo, MochiState
from .newton import func_is_env_active


@qd.func
def func_link_dof_jacobian(i_l, i_d, i_b, dyn_state: array_class.DynState):
    """Column of the link Jacobian for an ancestor degree of freedom: world-frame velocity of the link origin and
    angular velocity per unit velocity of the degree of freedom. The motion subspace is stored about the tree's
    center of mass, hence the shift to the link origin."""
    ang = dyn_state.dofs.cdof_ang[i_d, i_b]
    vel = dyn_state.dofs.cdof_vel[i_d, i_b] + ang.cross(
        dyn_state.links.pos[i_l, i_b] - dyn_state.links.root_COM[i_l, i_b]
    )
    return vel, ang


@qd.func
def func_jacobian_column_dot(i_l, i_d, i_b, vec6, dyn_state: array_class.DynState):
    """J_col^T vec6 for one degree of freedom."""
    vel, ang = func_link_dof_jacobian(i_l, i_d, i_b, dyn_state)
    return (
        vel[0] * vec6[0] + vel[1] * vec6[1] + vel[2] * vec6[2] + ang[0] * vec6[3] + ang[1] * vec6[4] + ang[2] * vec6[5]
    )


@qd.func
def func_jacobian_times_dofs(
    i_l,
    i_b,
    dofs: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Link-space 6-vector J_l dofs (translational then angular), summing over the ancestor degrees of freedom."""
    out = qd.Vector.zero(gs.qd_float, 6)
    i_a = i_l
    while i_a != -1:
        I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
        for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
            vel, ang = func_link_dof_jacobian(i_l, i_d, i_b, dyn_state)
            value = dofs[i_d, i_b]
            for k in qd.static(range(3)):
                out[k] += vel[k] * value
                out[3 + k] += ang[k] * value
        i_a = dyn_info.links.parent_idx[I_a]
    return out


@qd.func
def func_jacobian_transpose_add(
    i_l,
    i_b,
    vec6,
    out: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """out += J_l^T vec6 over the ancestor degrees of freedom of the link (atomic)."""
    i_a = i_l
    while i_a != -1:
        I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
        for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
            qd.atomic_add(out[i_d, i_b], func_jacobian_column_dot(i_l, i_d, i_b, vec6, dyn_state))
        i_a = dyn_info.links.parent_idx[I_a]


@qd.func
def func_joint_dof_displacement(
    i_j, i_d_, q_start, i_b, joint_type, rigid_info: array_class.RigidInfo, mochi_state: MochiState, eps
):
    """Tangent-space displacement of the i_d_-th degree of freedom of a joint since the stage start: a coordinate
    difference for scalar joints and the translation of a free joint, the body-frame rotation vector for the rotational
    part of free and spherical joints."""
    dq = gs.qd_float(0.0)
    if joint_type == gs.JOINT_TYPE.FREE and i_d_ < 3:
        dq = rigid_info.qpos[q_start + i_d_, i_b] - mochi_state.qpos_stage_start[q_start + i_d_, i_b]
    elif joint_type == gs.JOINT_TYPE.FREE or joint_type == gs.JOINT_TYPE.SPHERICAL:
        rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
        k = i_d_ - rot_offset
        quat = qd.Vector(
            [
                rigid_info.qpos[q_start + rot_offset, i_b],
                rigid_info.qpos[q_start + rot_offset + 1, i_b],
                rigid_info.qpos[q_start + rot_offset + 2, i_b],
                rigid_info.qpos[q_start + rot_offset + 3, i_b],
            ],
            dt=gs.qd_float,
        )
        quat_start = qd.Vector(
            [
                mochi_state.qpos_stage_start[q_start + rot_offset, i_b],
                mochi_state.qpos_stage_start[q_start + rot_offset + 1, i_b],
                mochi_state.qpos_stage_start[q_start + rot_offset + 2, i_b],
                mochi_state.qpos_stage_start[q_start + rot_offset + 3, i_b],
            ],
            dt=gs.qd_float,
        )
        rotvec = gu.qd_quat_to_rotvec(gu.qd_quat_mul(gu.qd_inv_quat(quat_start), quat), eps)
        dq = rotvec[k]
    else:
        dq = rigid_info.qpos[q_start + i_d_, i_b] - mochi_state.qpos_stage_start[q_start + i_d_, i_b]
    return dq


@qd.func
def func_project_links_residual(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    skip_ls_done,
):
    """res += J_l^T links_res_l for every moving link."""
    n_links = dyn_state.links.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_info.links.is_dynamic[i_l] or not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        func_jacobian_transpose_add(
            i_l, i_b, mochi_state.links_res[i_l, i_b], mochi_state.res, dyn_state, dyn_info, rigid_config
        )


@qd.kernel
def kernel_project_links_residual(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    skip_ls_done: qd.i32,
):
    func_project_links_residual(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        rigid_config,
        skip_ls_done,
    )


@qd.func
def func_assemble_joints(
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
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Joint-space terms of the incremental potential, per degree of freedom: joint damping and armature (relative to
    the stage start), joint stiffness and soft range limits of scalar joints, and the drives (force, velocity and
    position control) written as the potential of the actuator law evaluated implicitly at the iterate."""
    n_joints = dyn_info.joints.type.shape[0]
    _B = dyn_state.links.pos.shape[1]
    EPS = mochi_info.EPS[None]
    limit_stiffness = mochi_info.joint_limit_stiffness[None]
    limit_damping = mochi_info.joint_limit_damping[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_slot in qd.ndrange(n_joints, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_joints, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]
        if joint_type == gs.JOINT_TYPE.FIXED:
            continue
        q_start = dyn_info.joints.q_start[I_j]
        dof_start = dyn_info.joints.dof_start[I_j]
        n_dofs = dyn_info.joints.dof_end[I_j] - dof_start
        h = mochi_state.dt_stage[i_b]
        is_scalar = joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.PRISMATIC

        for i_d_ in range(n_dofs):
            i_d = dof_start + i_d_
            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
            dq = func_joint_dof_displacement(i_j, i_d_, q_start, i_b, joint_type, rigid_info, mochi_state, EPS)
            v_start = mochi_state.dofs_vel_stage_start[i_d, i_b]
            energy = gs.qd_float(0.0)
            grad = gs.qd_float(0.0)
            hess = gs.qd_float(0.0)

            damping = dyn_info.dofs.damping[I_d]
            if damping > 0.0:
                kappa = damping / h
                energy += 0.5 * kappa * dq * dq
                grad += kappa * dq
                hess += kappa

            armature = dyn_info.dofs.armature[I_d]
            if armature > 0.0:
                dv = dq / h - v_start
                energy += 0.5 * armature * dv * dv
                grad += (armature / h) * dv
                hess += armature / (h * h)

            if is_scalar:
                q = rigid_info.qpos[q_start, i_b]
                q_rel = q - rigid_info.qpos0[q_start, i_b]
                stiffness = dyn_info.dofs.stiffness[I_d]
                if stiffness > 0.0:
                    energy += 0.5 * stiffness * q_rel * q_rel
                    grad += stiffness * q_rel
                    hess += stiffness

                # Soft range limit, with the violation damped relative to its stage-start value.
                limit = dyn_info.dofs.limit[I_d]
                q_start_value = mochi_state.qpos_stage_start[q_start, i_b]
                violation = gs.qd_float(0.0)
                violation_start = gs.qd_float(0.0)
                side = gs.qd_float(0.0)
                if q > limit[1]:
                    violation = q - limit[1]
                    violation_start = qd.max(q_start_value - limit[1], 0.0)
                    side = 1.0
                elif q < limit[0]:
                    violation = limit[0] - q
                    violation_start = qd.max(limit[0] - q_start_value, 0.0)
                    side = -1.0
                if side != 0.0:
                    kappa = limit_damping / h
                    dviolation = violation - violation_start
                    energy += 0.5 * limit_stiffness * violation * violation + 0.5 * kappa * dviolation * dviolation
                    grad += side * (limit_stiffness * violation + kappa * dviolation)
                    hess += limit_stiffness + kappa

                # Drives: the actuator torque is a linear function of the coordinate and of the implicit velocity
                # dq / h, so it derives from a quadratic potential.
                ctrl_mode = dyn_state.dofs.ctrl_mode[i_d, i_b]
                if ctrl_mode == gs.CTRL_MODE.POSITION:
                    gain = dyn_info.dofs.act_gain[I_d]
                    bias = dyn_info.dofs.act_bias[I_d]
                    torque = (
                        gain * (dyn_state.dofs.ctrl_pos[i_d, i_b] - q_rel)
                        + bias[0]
                        + (gain + bias[1]) * q_rel
                        + bias[2] * (dq / h - dyn_state.dofs.ctrl_vel[i_d, i_b])
                    )
                    dtorque = bias[1] + bias[2] / h
                    energy += -torque * dq + 0.5 * dtorque * dq * dq
                    grad -= torque
                    hess -= dtorque
                elif ctrl_mode == gs.CTRL_MODE.VELOCITY:
                    bias = dyn_info.dofs.act_bias[I_d]
                    torque = -bias[2] * (dyn_state.dofs.ctrl_vel[i_d, i_b] - dq / h)
                    dtorque = bias[2] / h
                    energy += -torque * dq + 0.5 * dtorque * dq * dq
                    grad -= torque
                    hess -= dtorque
            if dyn_state.dofs.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                force = dyn_state.dofs.ctrl_force[i_d, i_b]
                energy -= force * dq
                grad -= force

            if qd.static(assem_obj):
                qd.atomic_add(mochi_state.obj[i_b], energy)
            if qd.static(assem_res):
                mochi_state.res[i_d, i_b] += grad
            if assem_dres:
                mochi_state.dofs_H_diag[i_d, i_b] += hess


@qd.kernel
def kernel_assemble_joints(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_assemble_joints(
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
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )


@qd.func
def func_update_conv_weights(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Convergence weights of the degrees of freedom: the inverse of the generalized inertia seen by each degree of
    freedom times the mass of its entity and the square of a reference acceleration of max(1, |g|), so that a unit
    weighted residual norm means a unit acceleration error."""
    n_dofs = mochi_state.conv_w.shape[0]
    n_links = dyn_state.links.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.conv_w[i_d, i_b] = 0.0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_info.links.is_dynamic[i_l]:
            continue
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        mass = mochi_info.links.mass[i_l]
        R_c = gu.qd_quat_to_R(dyn_state.links.quat[i_l, i_b], EPS) @ gu.qd_quat_to_R(
            dyn_info.links.inertial_quat[I_l], EPS
        )
        I_w = R_c @ mochi_info.links.inertia[i_l] @ R_c.transpose()
        i_a = i_l
        while i_a != -1:
            I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
            for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
                vel, ang = func_link_dof_jacobian(i_l, i_d, i_b, dyn_state)
                qd.atomic_add(mochi_state.conv_w[i_d, i_b], mass * vel.norm_sqr() + ang.dot(I_w @ ang))
            i_a = dyn_info.links.parent_idx[I_a]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        generalized_mass = mochi_state.conv_w[i_d, i_b]
        entity_mass = mochi_info.dofs_entity_mass[i_d]
        # A degree of freedom carrying no inertia at all (massless dummy link) has no acceleration scale to normalize
        # by: it enters the weighted norm with a unit weight rather than with the enormous weight a floored inertia
        # would produce, which would make its residual alone decide convergence for the whole entity.
        w = gs.qd_float(1.0)
        if generalized_mass > 0.0 and entity_mass > 0.0:
            a_ref = qd.max(1.0, mochi_info.gravity[i_b].norm())
            w = 1.0 / (a_ref * a_ref * entity_mass * generalized_mass)
        mochi_state.conv_w[i_d, i_b] = w


@qd.kernel
def kernel_update_conv_weights(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    func_update_conv_weights(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        rigid_config,
    )
