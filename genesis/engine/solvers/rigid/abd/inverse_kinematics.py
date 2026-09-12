"""
Inverse kinematics for rigid body entities.

This module contains the inverse kinematics kernel for computing joint configurations
that achieve desired end-effector poses.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.linalg as lu
import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_forward_kinematics_entity


@qd.func
def func_forward_kinematics_scratch(
    i_col_out,
    i_col_q,
    i_b,
    entity_idx,
    link_offset,
    joint_offset,
    q_offset,
    qpos: qd.Tensor,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Forward kinematics of one entity on caller-owned scratch buffers.

    Reads the configuration from qpos[:, i_col_q] (entity-local q coordinates keyed by q_offset) and writes link
    poses and joint frames at column i_col_out of the given tensors; the live solver state is only read (model
    info and the fixed root pose). Handles every joint type, so callers get correct kinematics without mutating
    solver state.
    """
    for i_l_ in range(dyn_info.entities.link_start[entity_idx], dyn_info.entities.link_end[entity_idx]):
        i_l = gs.qd_int(i_l_)
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        pos = dyn_info.links.pos[I_l]
        quat = dyn_info.links.quat[I_l]
        if dyn_info.links.parent_idx[I_l] != -1:
            parent_pos = links_pos[dyn_info.links.parent_idx[I_l] - link_offset, i_col_out]
            parent_quat = links_quat[dyn_info.links.parent_idx[I_l] - link_offset, i_col_out]
            pos = parent_pos + gu.qd_transform_by_quat(dyn_info.links.pos[I_l], parent_quat)
            quat = gu.qd_transform_quat_by_quat(dyn_info.links.quat[I_l], parent_quat)

        for i_j_ in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
            i_j = gs.qd_int(i_j_)
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
            joint_type = dyn_info.joints.type[I_j]
            q_start = dyn_info.joints.q_start[I_j]
            dof_start = dyn_info.joints.dof_start[I_j]
            I_d = [dof_start, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start

            q_loc = q_start - q_offset
            if joint_type == gs.JOINT_TYPE.FREE:
                # Six-DOF root: the configuration carries the base position (3) then orientation quaternion (4).
                base_pos = qd.Vector(
                    [qpos[q_loc, i_col_q], qpos[q_loc + 1, i_col_q], qpos[q_loc + 2, i_col_q]], dt=gs.qd_float
                )
                joints_xanchor[i_j - joint_offset, i_col_out] = base_pos
                joints_xaxis[i_j - joint_offset, i_col_out] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
                base_quat = qd.Vector(
                    [
                        qpos[q_loc + 3, i_col_q],
                        qpos[q_loc + 4, i_col_q],
                        qpos[q_loc + 5, i_col_q],
                        qpos[q_loc + 6, i_col_q],
                    ],
                    dt=gs.qd_float,
                )
                pos = base_pos
                quat = base_quat / base_quat.norm()
            elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                # Three-DOF ball joint: the configuration carries a local orientation quaternion (4).
                q_loc_quat = qd.Vector(
                    [
                        qpos[q_loc, i_col_q],
                        qpos[q_loc + 1, i_col_q],
                        qpos[q_loc + 2, i_col_q],
                        qpos[q_loc + 3, i_col_q],
                    ],
                    dt=gs.qd_float,
                )
                joints_xanchor[i_j - joint_offset, i_col_out] = (
                    gu.qd_transform_by_quat(dyn_info.joints.pos[I_j], quat) + pos
                )
                joints_xaxis[i_j - joint_offset, i_col_out] = gu.qd_transform_by_quat(
                    qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float), quat
                )
                quat = gu.qd_transform_quat_by_quat(q_loc_quat, quat)
                pos = joints_xanchor[i_j - joint_offset, i_col_out] - gu.qd_transform_by_quat(
                    dyn_info.joints.pos[I_j], quat
                )
            elif joint_type != gs.JOINT_TYPE.FIXED:
                axis = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    axis = dyn_info.dofs.motion_ang[I_d]
                else:
                    axis = dyn_info.dofs.motion_vel[I_d]
                joints_xanchor[i_j - joint_offset, i_col_out] = (
                    gu.qd_transform_by_quat(dyn_info.joints.pos[I_j], quat) + pos
                )
                joints_xaxis[i_j - joint_offset, i_col_out] = gu.qd_transform_by_quat(axis, quat)

                q_delta = qpos[q_loc, i_col_q] - rigid_info.qpos0[q_start, i_b]
                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    qloc = gu.qd_rotvec_to_quat(axis * q_delta, rigid_info.EPS[None])
                    quat = gu.qd_transform_quat_by_quat(qloc, quat)
                    pos = joints_xanchor[i_j - joint_offset, i_col_out] - gu.qd_transform_by_quat(
                        dyn_info.joints.pos[I_j], quat
                    )
                else:
                    pos = pos + joints_xaxis[i_j - joint_offset, i_col_out] * q_delta

        # Fixed root links keep their live pose, exactly like the solver FK (users may overwrite them manually).
        if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
            pos = dyn_state.links.pos[i_l, i_b]
            quat = dyn_state.links.quat[i_l, i_b]
        links_pos[i_l - link_offset, i_col_out] = pos
        links_quat[i_l - link_offset, i_col_out] = quat


@qd.func
def func_jacobian_scratch(
    i_col,
    i_b,
    entity_idx,
    joint_offset,
    ee_link,
    ee_pos,
    jacobian: qd.Tensor,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Spatial Jacobian (6 x n_dofs) of the goal point at column i_col.

    Reads the joint frames that func_forward_kinematics_scratch placed in the per-column scratch.

    Walks the kinematic path from the target link to the entity root, filling each joint's columns from its world
    axis and anchor (already placed by func_forward_kinematics_scratch into joints_xaxis / joints_xanchor). A
    SPHERICAL joint contributes nothing, matching the solver-state Jacobian. Unmasked.
    """
    dof_offset = dyn_info.entities.dof_start[entity_idx]
    for i_row, i_d in qd.ndrange(6, jacobian.shape[1]):
        jacobian[i_row, i_d, i_col] = 0.0

    i_l = ee_link
    while i_l != -1:
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        for i_j_ in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
            i_j = gs.qd_int(i_j_)
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
            i_d_jac = dyn_info.joints.dof_start[I_j] - dof_offset
            if dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                rotation = joints_xaxis[i_j - joint_offset, i_col]
                translation = rotation.cross(ee_pos - joints_xanchor[i_j - joint_offset, i_col])
                for i in qd.static(range(3)):
                    jacobian[i, i_d_jac, i_col] = translation[i]
                    jacobian[i + 3, i_d_jac, i_col] = rotation[i]
            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                translation = joints_xaxis[i_j - joint_offset, i_col]
                for i in qd.static(range(3)):
                    jacobian[i, i_d_jac, i_col] = translation[i]
            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.FREE:
                # Base translation moves the goal point directly; base rotation contributes a lever-arm term about
                # the root position (the FREE joint's scratch anchor, placed by func_forward_kinematics_scratch).
                base_pos = joints_xanchor[i_j - joint_offset, i_col]
                for i_d_ in qd.static(range(3)):
                    jacobian[i_d_, dyn_info.joints.dof_start[I_j] + i_d_ - dof_offset, i_col] = 1.0
                for i_d_ in qd.static(range(3)):
                    i_d = dyn_info.joints.dof_start[I_j] + i_d_ + 3
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                    rotation = dyn_info.dofs.motion_ang[I_d]
                    translation = rotation.cross(ee_pos - base_pos)
                    for i in qd.static(range(3)):
                        jacobian[i, i_d - dof_offset, i_col] = translation[i]
                        jacobian[i + 3, i_d - dof_offset, i_col] = rotation[i]
        i_l = dyn_info.links.parent_idx[I_l]


@qd.func
def func_get_jacobian(
    i_b,
    tgt_link_idx,
    dof_start,
    p_local,
    jacobian: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Full spatial Jacobian (6 x n_dofs) of a target link's local point.

    Written into column i_b of the buffer.

    Sums the velocity contribution of each joint on the path to the root; dof_start offsets global joint DOFs into
    the buffer rows. Axis masking is left to the caller (the inverse-kinematics solver masks the stacked copy).
    """
    n_dofs = jacobian.shape[1]
    for i_row, i_d in qd.ndrange(6, n_dofs):
        jacobian[i_row, i_d, i_b] = 0.0

    tgt_link_pos = dyn_state.links.pos[tgt_link_idx, i_b] + gu.qd_transform_by_quat(
        p_local, dyn_state.links.quat[tgt_link_idx, i_b]
    )
    i_l = tgt_link_idx
    while i_l > -1:
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j

            if dyn_info.joints.type[I_j] == gs.JOINT_TYPE.FIXED:
                pass

            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                i_d_jac = dyn_info.joints.dof_start[I_j] - dof_start
                rotation = dyn_state.joints.xaxis[i_j, i_b]
                translation = rotation.cross(tgt_link_pos - dyn_state.joints.xanchor[i_j, i_b])

                for i in qd.static(range(3)):
                    jacobian[i, i_d_jac, i_b] = translation[i]
                    jacobian[i + 3, i_d_jac, i_b] = rotation[i]

            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                i_d_jac = dyn_info.joints.dof_start[I_j] - dof_start
                translation = dyn_state.joints.xaxis[i_j, i_b]

                for i in qd.static(range(3)):
                    jacobian[i, i_d_jac, i_b] = translation[i]

            elif dyn_info.joints.type[I_j] == gs.JOINT_TYPE.FREE:
                for i_d_ in qd.static(range(3)):
                    i_d_jac = dyn_info.joints.dof_start[I_j] + i_d_ - dof_start
                    jacobian[i_d_, i_d_jac, i_b] = 1.0

                for i_d_ in qd.static(range(3)):
                    i_d = dyn_info.joints.dof_start[I_j] + i_d_ + 3
                    i_d_jac = i_d - dof_start
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                    rotation = dyn_info.dofs.motion_ang[I_d]
                    translation = rotation.cross(tgt_link_pos - dyn_state.links.pos[i_l, i_b])

                    for i in qd.static(range(3)):
                        jacobian[i, i_d_jac, i_b] = translation[i]
                        jacobian[i + 3, i_d_jac, i_b] = rotation[i]

        i_l = dyn_info.links.parent_idx[I_l]


@qd.kernel
def kernel_get_jacobian(
    tgt_link_idx: int,
    dof_start: int,
    p_local: qd.types.ndarray(),
    jacobian: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    n_batch: int,
):
    """Full spatial Jacobian of a link-local point, for every environment column."""
    p_vec = qd.Vector([p_local[0], p_local[1], p_local[2]], dt=gs.qd_float)
    for i_b in range(n_batch):
        func_get_jacobian(i_b, tgt_link_idx, dof_start, p_vec, jacobian, dyn_state, dyn_info, rigid_config)


@qd.kernel(fastcache=True)
def kernel_get_jacobian_zero(
    tgt_link_idx: int,
    dof_start: int,
    jacobian: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    n_batch: int,
):
    """Full spatial Jacobian of a link origin, for every environment column."""
    for i_b in range(n_batch):
        func_get_jacobian(
            i_b, tgt_link_idx, dof_start, qd.Vector.zero(gs.qd_float, 3), jacobian, dyn_state, dyn_info, rigid_config
        )


@qd.func
def func_integrate_dq_scratch(
    i_col,
    i_b,
    entity_idx,
    q_offset,
    dq: qd.Tensor,
    qpos: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    respect_joint_limit,
):
    """Integrate a joint-space step into the scratch configuration column.

    Mirrors the solver-state integrator on caller-owned qpos, leaving the live state untouched. dq is indexed by
    entity-local DOF; a FREE root advances its position additively and its orientation by a world-frame delta
    rotation, and a REVOLUTE / PRISMATIC DOF advances additively with optional limit clamping.
    """
    EPS = rigid_info.EPS[None]

    for i_l in range(dyn_info.entities.link_start[entity_idx], dyn_info.entities.link_end[entity_idx]):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.n_dofs[I_l] == 0:
            continue

        i_j = dyn_info.links.joint_start[I_l]
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]

        q_loc = dyn_info.links.q_start[I_l] - q_offset
        dof_start = dyn_info.links.dof_start[I_l]
        dq_start = dof_start - dyn_info.entities.dof_start[entity_idx]

        if joint_type == gs.JOINT_TYPE.FREE:
            pos = qd.Vector([qpos[q_loc, i_col], qpos[q_loc + 1, i_col], qpos[q_loc + 2, i_col]])
            pos = pos + qd.Vector([dq[dq_start, i_col], dq[dq_start + 1, i_col], dq[dq_start + 2, i_col]])
            quat = qd.Vector(
                [qpos[q_loc + 3, i_col], qpos[q_loc + 4, i_col], qpos[q_loc + 5, i_col], qpos[q_loc + 6, i_col]]
            )
            dquat = gu.qd_rotvec_to_quat(
                qd.Vector([dq[dq_start + 3, i_col], dq[dq_start + 4, i_col], dq[dq_start + 5, i_col]], dt=gs.qd_float),
                EPS,
            )
            # The delta rotation is expressed in the world frame, so it composes on the left of the orientation.
            quat = gu.qd_transform_quat_by_quat(quat, dquat)
            for j in qd.static(range(3)):
                qpos[q_loc + j, i_col] = pos[j]
            for j in qd.static(range(4)):
                qpos[q_loc + j + 3, i_col] = quat[j]

        elif joint_type == gs.JOINT_TYPE.FIXED:
            pass

        else:
            for i_d_ in range(dyn_info.links.n_dofs[I_l]):
                qpos[q_loc + i_d_, i_col] = qpos[q_loc + i_d_, i_col] + dq[dq_start + i_d_, i_col]
                if respect_joint_limit:
                    I_d = [dof_start + i_d_, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start + i_d_
                    qpos[q_loc + i_d_, i_col] = qd.math.clamp(
                        qpos[q_loc + i_d_, i_col], dyn_info.dofs.limit[I_d][0], dyn_info.dofs.limit[I_d][1]
                    )


@qd.func
def func_inverse_kinematics(
    entity_idx,
    q_offset,
    link_offset,
    joint_offset,
    dyn_state: array_class.DynState,
    ik_state: array_class.IKState,
    fk: array_class.IKScratchFK,
    targets: array_class.IKTargets,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    n_qs,
    entity_n_dofs,
    n_links,
    n_dofs,
    n_envs,
    custom_init_qpos,
    max_samples,
    max_solver_iters,
    damping,
    pos_tol,
    rot_tol,
    max_step_size,
    seed,
    respect_joint_limit,
):
    """Damped-least-squares inverse kinematics for the target links.

    Writes the best configuration found per column into ik_state.qpos_best.

    Runs up to max_samples restarts of at most max_solver_iters Gauss-Newton steps against the stacked multi-target
    pose error, keeping the least-error configuration. Every evaluation and integration happens on the caller-owned
    forward-kinematics scratch (fk), so the live solver state is only read; unconverged restarts resample
    joint-limited DOFs with a counter-based draw so the result is deterministic across parallel schedules.
    """
    EPS = rigid_info.EPS[None]

    n_error_dims = 6 * n_links

    for i_b_ in range(n_envs):
        i_b = targets.envs_idx[i_b_]

        # Seed the working configuration from the requested initial guess or the live solver configuration.
        for i_q in range(n_qs):
            if custom_init_qpos:
                fk.qpos[i_q, i_b] = targets.init_qpos[i_b, i_q]
            else:
                fk.qpos[i_q, i_b] = rigid_info.qpos[i_q + q_offset, i_b]

        for i_error in range(n_error_dims):
            ik_state.err_pose_best[i_error, i_b] = 1e4

        solved = False
        for i_sample in range(max_samples):
            for _ in range(max_solver_iters):
                func_forward_kinematics_scratch(
                    i_b,
                    i_b,
                    i_b,
                    entity_idx,
                    link_offset,
                    joint_offset,
                    q_offset,
                    fk.qpos,
                    fk.links_pos,
                    fk.links_quat,
                    fk.joints_xanchor,
                    fk.joints_xaxis,
                    dyn_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                )

                solved = True
                for i_ee in range(n_links):
                    i_l_ee = targets.links_idx[i_ee]
                    ee_quat = fk.links_quat[i_l_ee - link_offset, i_b]
                    ee_pos = fk.links_pos[i_l_ee - link_offset, i_b] + gu.qd_transform_by_quat(
                        targets.local_point[i_ee], ee_quat
                    )

                    err_pos_i = targets.pos[i_ee, i_b] - ee_pos
                    for k in range(3):
                        err_pos_i[k] = err_pos_i[k] * (targets.pos_mask[k] * targets.link_pos_mask[i_ee])
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    err_rot_i = gu.qd_quat_to_rotvec(
                        gu.qd_transform_quat_by_quat(gu.qd_inv_quat(ee_quat), targets.quat[i_ee, i_b]), EPS
                    )
                    for k in range(3):
                        err_rot_i[k] = err_rot_i[k] * (targets.rot_mask[k] * targets.link_rot_mask[i_ee])
                    if err_rot_i.norm() > rot_tol:
                        solved = False

                    for k in range(3):
                        ik_state.err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        ik_state.err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

                if solved:
                    break

                for i_ee in range(n_links):
                    i_l_ee = targets.links_idx[i_ee]
                    ee_quat = fk.links_quat[i_l_ee - link_offset, i_b]
                    ee_pos = fk.links_pos[i_l_ee - link_offset, i_b] + gu.qd_transform_by_quat(
                        targets.local_point[i_ee], ee_quat
                    )
                    # Full single-link Jacobian over the entity DOFs; axis masking is applied while stacking below.
                    func_jacobian_scratch(
                        i_b,
                        i_b,
                        entity_idx,
                        joint_offset,
                        i_l_ee,
                        ee_pos,
                        ik_state.jacobian,
                        fk.joints_xanchor,
                        fk.joints_xaxis,
                        dyn_info,
                        rigid_config,
                    )

                    for i_d_ in range(n_dofs):
                        i_d = targets.dofs_idx[i_d_]
                        for i_error in qd.static(range(3)):
                            ik_state.jacobian_stacked[i_ee * 6 + i_error, i_d_, i_b] = (
                                ik_state.jacobian[i_error, i_d, i_b] * targets.pos_mask[i_error]
                            )
                            ik_state.jacobian_stacked[i_ee * 6 + i_error + 3, i_d_, i_b] = (
                                ik_state.jacobian[i_error + 3, i_d, i_b] * targets.rot_mask[i_error]
                            )

                lu.mat_transpose(ik_state.jacobian_stacked, ik_state.jacobian_stacked_t, n_error_dims, n_dofs, i_b)
                lu.mat_mul(
                    ik_state.jacobian_stacked,
                    ik_state.jacobian_stacked_t,
                    ik_state.mat,
                    n_error_dims,
                    n_dofs,
                    n_error_dims,
                    i_b,
                )
                lu.mat_add_eye(ik_state.mat, damping**2, n_error_dims, i_b)
                lu.mat_inverse(
                    ik_state.mat,
                    ik_state.lu_lower,
                    ik_state.lu_upper,
                    ik_state.lu_y,
                    ik_state.inv,
                    n_error_dims,
                    i_b,
                )
                lu.mat_mul_vec(ik_state.inv, ik_state.err_pose, ik_state.vec, n_error_dims, n_error_dims, i_b)

                for i_d_ in range(entity_n_dofs):
                    ik_state.delta_qpos[i_d_, i_b] = 0
                for i_d_ in range(n_dofs):
                    i_d = targets.dofs_idx[i_d_]
                    for j in range(n_error_dims):
                        ik_state.delta_qpos[i_d, i_b] = ik_state.delta_qpos[i_d, i_b] + (
                            ik_state.jacobian_stacked_t[i_d_, j, i_b] * ik_state.vec[j, i_b]
                        )

                for i_d_ in range(entity_n_dofs):
                    ik_state.delta_qpos[i_d_, i_b] = qd.math.clamp(
                        ik_state.delta_qpos[i_d_, i_b], -max_step_size, max_step_size
                    )

                func_integrate_dq_scratch(
                    i_b,
                    i_b,
                    entity_idx,
                    q_offset,
                    ik_state.delta_qpos,
                    fk.qpos,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                    respect_joint_limit,
                )

            if not solved:
                # Recompute the residual for the final configuration when the iteration budget ran out.
                func_forward_kinematics_scratch(
                    i_b,
                    i_b,
                    i_b,
                    entity_idx,
                    link_offset,
                    joint_offset,
                    q_offset,
                    fk.qpos,
                    fk.links_pos,
                    fk.links_quat,
                    fk.joints_xanchor,
                    fk.joints_xaxis,
                    dyn_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                )
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = targets.links_idx[i_ee]
                    ee_quat = fk.links_quat[i_l_ee - link_offset, i_b]
                    ee_pos = fk.links_pos[i_l_ee - link_offset, i_b] + gu.qd_transform_by_quat(
                        targets.local_point[i_ee], ee_quat
                    )
                    err_pos_i = targets.pos[i_ee, i_b] - ee_pos
                    for k in range(3):
                        err_pos_i[k] = err_pos_i[k] * (targets.pos_mask[k] * targets.link_pos_mask[i_ee])
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    err_rot_i = gu.qd_quat_to_rotvec(
                        gu.qd_transform_quat_by_quat(gu.qd_inv_quat(ee_quat), targets.quat[i_ee, i_b]), EPS
                    )
                    for k in range(3):
                        err_rot_i[k] = err_rot_i[k] * (targets.rot_mask[k] * targets.link_rot_mask[i_ee])
                    if err_rot_i.norm() > rot_tol:
                        solved = False

                    for k in range(3):
                        ik_state.err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        ik_state.err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

            if solved:
                for i_q in range(n_qs):
                    ik_state.qpos_best[i_q, i_b] = fk.qpos[i_q, i_b]
                for i_error in range(n_error_dims):
                    ik_state.err_pose_best[i_error, i_b] = ik_state.err_pose[i_error, i_b]
                break

            else:
                improved = True
                for i_ee in range(n_links):
                    error_pos_i = qd.Vector([ik_state.err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3)])
                    error_rot_i = qd.Vector([ik_state.err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)])
                    error_pos_best = qd.Vector(
                        [ik_state.err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3)]
                    )
                    error_rot_best = qd.Vector(
                        [ik_state.err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)]
                    )
                    if error_pos_i.norm() > error_pos_best.norm() or error_rot_i.norm() > error_rot_best.norm():
                        improved = False
                        break

                if improved:
                    for i_q in range(n_qs):
                        ik_state.qpos_best[i_q, i_b] = fk.qpos[i_q, i_b]
                    for i_error in range(n_error_dims):
                        ik_state.err_pose_best[i_error, i_b] = ik_state.err_pose[i_error, i_b]

                if respect_joint_limit and i_sample < max_samples - 1:
                    entity_dof_start = dyn_info.entities.dof_start[entity_idx]
                    for i_l in range(dyn_info.entities.link_start[entity_idx], dyn_info.entities.link_end[entity_idx]):
                        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

                        must_resample = False
                        for i_d_ in range(n_dofs):
                            i_d = targets.dofs_idx[i_d_]
                            link_dof_start_local = dyn_info.links.dof_start[I_l] - entity_dof_start
                            link_dof_end_local = dyn_info.links.dof_end[I_l] - entity_dof_start
                            if link_dof_start_local <= i_d and i_d < link_dof_end_local:
                                must_resample = True
                                break
                        if not must_resample:
                            continue

                        for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                            i_d = dyn_info.joints.dof_start[I_j]
                            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d

                            dof_limit = dyn_info.dofs.limit[I_d]
                            if (
                                dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                                or dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                            ) and not (qd.math.isinf(dof_limit[0]) or qd.math.isinf(dof_limit[1])):
                                q_start = dyn_info.joints.q_start[I_j]
                                # Counter-based draw keyed on (env, restart, DOF): a qd.random() per-thread stream
                                # would order by the parallel schedule, making the resampled - and returned -
                                # configuration nondeterministic across runs.
                                fk.qpos[q_start - q_offset, i_b] = dof_limit[0] + gu.qd_hash01(
                                    seed, i_b, i_sample, q_start
                                ) * (dof_limit[1] - dof_limit[0])


@qd.kernel(fastcache=True)
def kernel_inverse_kinematics_entity(
    entity_idx: int,
    q_offset: int,
    link_offset: int,
    joint_offset: int,
    dyn_state: array_class.DynState,
    ik_state: array_class.IKState,
    fk: array_class.IKScratchFK,
    targets: array_class.IKTargets,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    n_qs: int,
    entity_n_dofs: int,
    n_links: int,
    n_dofs: int,
    n_envs: int,
    custom_init_qpos: int,
    max_samples: int,
    max_solver_iters: int,
    damping: float,
    pos_tol: float,
    rot_tol: float,
    max_step_size: float,
    seed: int,
    respect_joint_limit: int,
):
    """Entity-facing launch wrapper forwarding the IK scratch and model scalars to func_inverse_kinematics."""
    func_inverse_kinematics(
        entity_idx,
        q_offset,
        link_offset,
        joint_offset,
        dyn_state,
        ik_state,
        fk,
        targets,
        dyn_info,
        rigid_info,
        rigid_config,
        n_qs,
        entity_n_dofs,
        n_links,
        n_dofs,
        n_envs,
        custom_init_qpos,
        max_samples,
        max_solver_iters,
        damping,
        pos_tol,
        rot_tol,
        max_step_size,
        seed,
        respect_joint_limit,
    )


@qd.kernel(fastcache=True)
def kernel_set_ik_targets(
    links_idx: qd.types.ndarray(),
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    quats: qd.types.ndarray(),
    local_points: qd.types.ndarray(),
    init_qpos: qd.types.ndarray(),
    pos_mask: qd.types.ndarray(),
    rot_mask: qd.types.ndarray(),
    link_pos_mask: qd.types.ndarray(),
    link_rot_mask: qd.types.ndarray(),
    targets: array_class.IKTargets,
):
    """Copy externally supplied inverse-kinematics targets into the solver-side struct (non-zero-copy fallback)."""
    for i_ee in range(links_idx.shape[0]):
        targets.links_idx[i_ee] = links_idx[i_ee]
        targets.link_pos_mask[i_ee] = link_pos_mask[i_ee]
        targets.link_rot_mask[i_ee] = link_rot_mask[i_ee]
        targets.local_point[i_ee] = qd.Vector([local_points[i_ee, 0], local_points[i_ee, 1], local_points[i_ee, 2]])
        for i_c in range(envs_idx.shape[0]):
            i_b = envs_idx[i_c]
            targets.pos[i_ee, i_b] = qd.Vector([poss[i_ee, i_c, 0], poss[i_ee, i_c, 1], poss[i_ee, i_c, 2]])
            targets.quat[i_ee, i_b] = qd.Vector(
                [quats[i_ee, i_c, 0], quats[i_ee, i_c, 1], quats[i_ee, i_c, 2], quats[i_ee, i_c, 3]]
            )
    for i_d in range(dofs_idx.shape[0]):
        targets.dofs_idx[i_d] = dofs_idx[i_d]
    for i_c in range(envs_idx.shape[0]):
        targets.envs_idx[i_c] = envs_idx[i_c]
        for i_q in range(init_qpos.shape[1]):
            targets.init_qpos[envs_idx[i_c], i_q] = init_qpos[i_c, i_q]
    for k in qd.static(range(3)):
        targets.pos_mask[k] = pos_mask[k]
        targets.rot_mask[k] = rot_mask[k]


@qd.kernel
def kernel_forward_kinematics_query(
    entity_idx: int,
    qs_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    links_pos: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    qpos: qd.types.ndarray(),
    qpos_cache: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Link poses of one entity at a queried configuration, leaving the live configuration as it was.

    The queried configuration is written into the solver, forward kinematics runs, the poses are read out, and the
    saved configuration is restored and re-propagated. qpos_cache spans the solver configuration, so the global q
    indices index it directly.
    """
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
        qpos_cache[qs_idx[i_q_], envs_idx[i_b_]] = rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]]
        rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        func_forward_kinematics_entity(
            entity_idx,
            envs_idx[i_b_],
            rigid_info.qpos,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            is_backward=False,
        )

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        for i in qd.static(range(3)):
            links_pos[i_b_, i_l_, i] = dyn_state.links.pos[links_idx[i_l_], envs_idx[i_b_]][i]
        for i in qd.static(range(4)):
            links_quat[i_b_, i_l_, i] = dyn_state.links.quat[links_idx[i_l_], envs_idx[i_b_]][i]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
        rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos_cache[qs_idx[i_q_], envs_idx[i_b_]]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        func_forward_kinematics_entity(
            entity_idx,
            envs_idx[i_b_],
            rigid_info.qpos,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            is_backward=False,
        )
