"""
Inverse kinematics for rigid body entities.

This module contains the inverse kinematics kernel for computing joint configurations
that achieve desired end-effector poses.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_forward_kinematics_root
from genesis.engine.solvers.rigid.data import (
    ENTITY_DOFS,
    ENTITY_JOINTS,
    ENTITY_LINKS,
    ENTITY_QS,
    DynInfo,
    DynState,
    RigidInfo,
)
from genesis.utils.data import ENVS, Data, DataKind, Size, eoa, field
from genesis.utils.misc import qd_isfinite


# The cap of inverse-kinematics targets
IK_TARGETS = Size("ik_targets")
# The pose residuals of the targets stacked, six per target
IK_ERRORS = 6 * IK_TARGETS


class IKState(Data, kind=DataKind.SCRATCH):
    """Damped-least-squares inverse-kinematics scratch, one column per environment or per restart of the planner."""

    # Spatial Jacobian of one target, rebuilt per target link and copied into the stacked block
    jacobian: qd.Tensor = field(float, shape=(6, ENTITY_DOFS, ENVS))
    # Jacobian of all targets stacked
    jacobian_stacked: qd.Tensor = field(float, shape=(IK_ERRORS, ENTITY_DOFS, ENVS))
    # Lower Cholesky factor of the damped normal matrix, J J^T + damping^2 I in the error space or J^T J + damping^2 I
    # in the dof space, whichever is smaller (see func_inverse_kinematics), its upper triangle unused
    chol: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    # Pose residual of all targets stacked, the best one across restarts, and the solution of the normal equations
    err_pose: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    err_pose_best: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    vec: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    # Joint-space step, and the best configuration across restarts
    delta_qpos: qd.Tensor = field(float, shape=(ENTITY_DOFS, ENVS))
    qpos_best: qd.Tensor = field(float, shape=(ENTITY_QS, ENVS))


class IKScratchFK(Data, kind=DataKind.SCRATCH):
    """Forward kinematics of the inverse-kinematics solve of an entity, apart from the state of the solver.

    It holds the configuration the solve iterates and the link and joint frames of that configuration, one column per
    environment.
    """

    qpos: qd.Tensor = field(float, shape=(ENTITY_QS, ENVS))
    # Written by func_forward_kinematics_scratch
    links_pos: qd.Tensor = field((float, 3), shape=(ENTITY_LINKS, ENVS))
    links_quat: qd.Tensor = field((float, 4), shape=(ENTITY_LINKS, ENVS))
    joints_xanchor: qd.Tensor = field((float, 3), shape=(ENTITY_JOINTS, ENVS))
    joints_xaxis: qd.Tensor = field((float, 3), shape=(ENTITY_JOINTS, ENVS))


class IKTargets(Data, kind=DataKind.SCRATCH):
    """Targets and dofs of an inverse-kinematics solve, one column per environment or per restart of the planner.

    The caller writes every field before the solve.
    """

    # Global indices of the target links, of the dofs to move and of the columns to solve
    links_idx: qd.Tensor = field(int, shape=(IK_TARGETS,))
    dofs_idx: qd.Tensor = field(int, shape=(ENTITY_DOFS,))
    envs_idx: qd.Tensor = field(int, shape=(ENVS,))
    # Target pose per (link, column) in the world frame, and the link-local point it drives
    pos: qd.Tensor = field((float, 3), shape=(IK_TARGETS, ENVS))
    quat: qd.Tensor = field((float, 4), shape=(IK_TARGETS, ENVS))
    local_point: qd.Tensor = field((float, 3), shape=(IK_TARGETS,))
    # Initial configuration per column, read when custom_init_qpos is set
    init_qpos: qd.Tensor = field(float, shape=(ENTITY_QS, ENVS))
    # The position and rotation axes to solve for all links, and the position and rotation switches of each link
    pos_mask: qd.Tensor = field(float, shape=(3,))
    rot_mask: qd.Tensor = field(float, shape=(3,))
    link_pos_mask: qd.Tensor = field(int, shape=(IK_TARGETS,))
    link_rot_mask: qd.Tensor = field(int, shape=(IK_TARGETS,))


class IKScratch(Data, kind=DataKind.SCRATCH):
    """The scratch of the inverse-kinematics solves of every entity of a solver.

    It spans the largest entity and the target cap, and a solve writes the leading rows its entity spans.
    """

    state: IKState
    fk: IKScratchFK
    targets: IKTargets


@qd.func
def func_forward_kinematics_scratch(
    i_col_out: int,
    i_col_q: int,
    i_b: int,
    entity_idx: int,
    link_offset: int,
    joint_offset: int,
    q_offset: int,
    qpos: qd.Tensor,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
):
    """Forward kinematics of one entity on caller-owned scratch buffers.

    Reads the configuration from qpos[:, i_col_q] (entity-local q coordinates keyed by q_offset) and writes link
    poses and joint frames at column i_col_out of the given tensors. The live solver state is only read (model
    info and the fixed root pose). Handles every joint type, so callers get correct kinematics without mutating
    solver state.
    """
    for i_l_ in range(dyn_info.entities.link_start[entity_idx], dyn_info.entities.link_end[entity_idx]):
        i_l = gs.qd_int(i_l_)
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        pos = dyn_info.links.pos[I_l]
        quat = dyn_info.links.quat[I_l]
        i_l_parent = dyn_info.links.parent_idx[I_l]
        if i_l_parent != -1:
            parent_pos = links_pos[i_l_parent - link_offset, i_col_out]
            parent_quat = links_quat[i_l_parent - link_offset, i_col_out]
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
                    qloc = gu.qd_rotvec_to_quat(axis * q_delta)
                    quat = gu.qd_transform_quat_by_quat(qloc, quat)
                    pos = joints_xanchor[i_j - joint_offset, i_col_out] - gu.qd_transform_by_quat(
                        dyn_info.joints.pos[I_j], quat
                    )
                else:
                    pos = pos + joints_xaxis[i_j - joint_offset, i_col_out] * q_delta

        # Fixed root links keep their live pose, exactly like the solver FK (users may overwrite them manually).
        if i_l_parent == -1 and dyn_info.links.is_fixed[I_l]:
            pos = dyn_state.links.pos[i_l, i_b]
            quat = dyn_state.links.quat[i_l, i_b]
        links_pos[i_l - link_offset, i_col_out] = pos
        links_quat[i_l - link_offset, i_col_out] = quat


@qd.func
def func_jacobian_scratch(
    i_col: int,
    i_b: int,
    entity_idx: int,
    joint_offset: int,
    ee_link: int,
    ee_pos: qd.types.vector(3),
    jacobian: qd.Tensor,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    dyn_info: DynInfo,
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

    # Every parent comes before its children in the links of an entity, which bounds the ancestors of the target link
    # by the links preceding it.
    link_start = dyn_info.entities.link_start[entity_idx]
    i_l = ee_link
    is_on_path = True
    for _ in range(ee_link - link_start + 1):
        if is_on_path:
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
                    # Base translation moves the goal point directly, and base rotation contributes a lever-arm term
                    # about the root position, the scratch anchor of the FREE joint.
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
            is_on_path = i_l >= link_start


@qd.func
def func_get_jacobian(
    i_b: int,
    tgt_link_idx: int,
    dof_start: int,
    p_local: qd.types.vector(3),
    jacobian: qd.Tensor,
    dyn_state: DynState,
    dyn_info: DynInfo,
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


@qd.kernel(fastcache=True)
def kernel_get_jacobian(
    tgt_link_idx: int,
    dof_start: int,
    p_local: qd.types.ndarray(),
    jacobian: qd.Tensor,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
    n_batch: int,
):
    """Full spatial Jacobian of a link-local point, for every environment column."""
    p_vec = qd.Vector([p_local[0], p_local[1], p_local[2]], dt=gs.qd_float)
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(n_batch):
        func_get_jacobian(i_b, tgt_link_idx, dof_start, p_vec, jacobian, dyn_state, dyn_info, rigid_config)


@qd.kernel(fastcache=True)
def kernel_get_jacobian_zero(
    tgt_link_idx: int,
    dof_start: int,
    jacobian: qd.Tensor,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
    n_batch: int,
):
    """Full spatial Jacobian of a link origin, for every environment column."""
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(n_batch):
        func_get_jacobian(
            i_b, tgt_link_idx, dof_start, qd.Vector.zero(gs.qd_float, 3), jacobian, dyn_state, dyn_info, rigid_config
        )


@qd.func
def func_integrate_dq_scratch(
    i_col: int,
    i_b: int,
    entity_idx: int,
    q_offset: int,
    dq: qd.Tensor,
    qpos: qd.Tensor,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
    respect_joint_limit: bool,
):
    """Integrate a joint-space step into the scratch configuration column.

    Mirrors the solver-state integrator on caller-owned qpos, leaving the live state untouched. dq is indexed by
    entity-local DOF. A FREE root advances its position additively and its orientation by a world-frame delta
    rotation, and a REVOLUTE / PRISMATIC DOF advances additively with optional limit clamping.
    """

    for i_l in range(dyn_info.entities.link_start[entity_idx], dyn_info.entities.link_end[entity_idx]):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.n_dofs[I_l] > 0:
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
                    qd.Vector(
                        [dq[dq_start + 3, i_col], dq[dq_start + 4, i_col], dq[dq_start + 5, i_col]], dt=gs.qd_float
                    )
                )
                # The delta rotation is expressed in the world frame, so it composes on the left of the orientation.
                quat = gu.qd_transform_quat_by_quat(quat, dquat)
                for j in qd.static(range(3)):
                    qpos[q_loc + j, i_col] = pos[j]
                for j in qd.static(range(4)):
                    qpos[q_loc + j + 3, i_col] = quat[j]

            elif joint_type != gs.JOINT_TYPE.FIXED:
                for i_d_ in range(dyn_info.links.n_dofs[I_l]):
                    qpos[q_loc + i_d_, i_col] = qpos[q_loc + i_d_, i_col] + dq[dq_start + i_d_, i_col]
                    if respect_joint_limit:
                        I_d = [dof_start + i_d_, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start + i_d_
                        qpos[q_loc + i_d_, i_col] = qd.math.clamp(
                            qpos[q_loc + i_d_, i_col], dyn_info.dofs.limit[I_d][0], dyn_info.dofs.limit[I_d][1]
                        )


@qd.func
def func_inverse_kinematics(
    entity_idx: int,
    q_offset: int,
    link_offset: int,
    joint_offset: int,
    seed: int,
    dyn_state: DynState,
    ik_state: IKState,
    fk: IKScratchFK,
    targets: IKTargets,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    n_qs: int,
    entity_n_dofs: int,
    n_links: int,
    n_dofs: int,
    n_envs: int,
    max_samples: int,
    max_solver_iters: int,
    damping: float,
    pos_tol: float,
    rot_tol: float,
    max_step_size: float,
    custom_init_qpos: bool,
    respect_joint_limit: bool,
):
    """Damped-least-squares inverse kinematics for the target links.

    Writes the best configuration found per column into ik_state.qpos_best.

    Runs up to max_samples restarts of at most max_solver_iters Gauss-Newton steps against the stacked multi-target
    pose error, keeping the least-error configuration. Every evaluation and integration happens on the caller-owned
    forward-kinematics scratch (fk), so the live solver state is only read. Unconverged restarts resample
    joint-limited DOFs with a counter-based draw, so the result is deterministic across parallel schedules.
    """

    n_error_dims = 6 * n_links

    # The damped normal equations are solved in the smaller of the error and dof spaces, the step
    # J^T (J J^T + damping^2 I)^-1 err being equal to (J^T J + damping^2 I)^-1 J^T err. Both factor the damped Gram
    # matrix A A^T + damping^2 I of the rows of A, those of J in the error space and those of J^T in the dof space.
    is_dof_space = n_dofs < n_error_dims
    n_rows = n_dofs if is_dof_space else n_error_dims
    n_cols = n_error_dims if is_dof_space else n_dofs

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(n_envs):
        i_b = targets.envs_idx[i_b_]

        # Seed the working configuration from the requested initial guess or the live solver configuration.
        for i_q in range(n_qs):
            if custom_init_qpos:
                fk.qpos[i_q, i_b] = targets.init_qpos[i_q, i_b]
            else:
                fk.qpos[i_q, i_b] = rigid_info.qpos[i_q + q_offset, i_b]

        for i_error in range(n_error_dims):
            ik_state.err_pose_best[i_error, i_b] = 1e4

        is_solved = False
        for i_sample in range(max_samples):
            # The last iteration only evaluates the residual of the configuration the iteration budget ends on.
            for i_iter in range(max_solver_iters + 1):
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

                is_solved = True
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
                        is_solved = False

                    err_rot_i = gu.qd_quat_to_rotvec(
                        gu.qd_transform_quat_by_quat(gu.qd_inv_quat(ee_quat), targets.quat[i_ee, i_b])
                    )
                    for k in range(3):
                        err_rot_i[k] = err_rot_i[k] * (targets.rot_mask[k] * targets.link_rot_mask[i_ee])
                    if err_rot_i.norm() > rot_tol:
                        is_solved = False

                    for k in range(3):
                        ik_state.err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        ik_state.err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

                    if i_iter < max_solver_iters:
                        # Full single-link Jacobian over the entity DOFs, the axes being masked while stacking below
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

                if is_solved or i_iter == max_solver_iters:
                    break

                # In-place Cholesky factorization of the damped Gram matrix, forming each entry when it reaches it
                for i_row in range(n_rows):
                    for j_row in range(i_row + 1):
                        val = gs.qd_float(0.0)
                        for i_col in range(n_cols):
                            i_error = i_col if is_dof_space else i_row
                            j_error = i_col if is_dof_space else j_row
                            i_d_ = i_row if is_dof_space else i_col
                            j_d_ = j_row if is_dof_space else i_col
                            val = val + (
                                ik_state.jacobian_stacked[i_error, i_d_, i_b]
                                * ik_state.jacobian_stacked[j_error, j_d_, i_b]
                            )
                        for k_row in range(j_row):
                            val = val - ik_state.chol[i_row, k_row, i_b] * ik_state.chol[j_row, k_row, i_b]
                        if i_row == j_row:
                            ik_state.chol[i_row, i_row, i_b] = qd.sqrt(val + damping**2)
                        else:
                            ik_state.chol[i_row, j_row, i_b] = val / ik_state.chol[j_row, j_row, i_b]

                # Forward then backward substitution, the right-hand side being err in the error space and J^T err in
                # the dof space
                for i_row in range(n_rows):
                    val = gs.qd_float(0.0)
                    if is_dof_space:
                        for i_error in range(n_error_dims):
                            val = val + ik_state.jacobian_stacked[i_error, i_row, i_b] * ik_state.err_pose[i_error, i_b]
                    else:
                        val = ik_state.err_pose[i_row, i_b]
                    for k_row in range(i_row):
                        val = val - ik_state.chol[i_row, k_row, i_b] * ik_state.vec[k_row, i_b]
                    ik_state.vec[i_row, i_b] = val / ik_state.chol[i_row, i_row, i_b]
                for i_row_ in range(n_rows):
                    i_row = n_rows - 1 - i_row_
                    val = ik_state.vec[i_row, i_b]
                    for k_row in range(i_row + 1, n_rows):
                        val = val - ik_state.chol[k_row, i_row, i_b] * ik_state.vec[k_row, i_b]
                    ik_state.vec[i_row, i_b] = val / ik_state.chol[i_row, i_row, i_b]

                # The joint-space step is the solution in the dof space, and J^T times it in the error space
                for i_d_ in range(entity_n_dofs):
                    ik_state.delta_qpos[i_d_, i_b] = 0.0
                for i_d_ in range(n_dofs):
                    i_d = targets.dofs_idx[i_d_]
                    val = gs.qd_float(0.0)
                    if is_dof_space:
                        val = ik_state.vec[i_d_, i_b]
                    else:
                        for i_error in range(n_error_dims):
                            val = val + ik_state.jacobian_stacked[i_error, i_d_, i_b] * ik_state.vec[i_error, i_b]
                    ik_state.delta_qpos[i_d, i_b] = qd.math.clamp(val, -max_step_size, max_step_size)

                func_integrate_dq_scratch(
                    i_b,
                    i_b,
                    entity_idx,
                    q_offset,
                    ik_state.delta_qpos,
                    fk.qpos,
                    dyn_info,
                    rigid_config,
                    respect_joint_limit,
                )

            if is_solved:
                for i_q in range(n_qs):
                    ik_state.qpos_best[i_q, i_b] = fk.qpos[i_q, i_b]
                for i_error in range(n_error_dims):
                    ik_state.err_pose_best[i_error, i_b] = ik_state.err_pose[i_error, i_b]
                break

            is_improved = True
            for i_ee in range(n_links):
                error_pos_i = qd.Vector([ik_state.err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3)])
                error_rot_i = qd.Vector([ik_state.err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)])
                error_pos_best = qd.Vector([ik_state.err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3)])
                error_rot_best = qd.Vector([ik_state.err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)])
                if error_pos_i.norm() > error_pos_best.norm() or error_rot_i.norm() > error_rot_best.norm():
                    is_improved = False

            if is_improved:
                for i_q in range(n_qs):
                    ik_state.qpos_best[i_q, i_b] = fk.qpos[i_q, i_b]
                for i_error in range(n_error_dims):
                    ik_state.err_pose_best[i_error, i_b] = ik_state.err_pose[i_error, i_b]

            if respect_joint_limit and i_sample < max_samples - 1:
                entity_dof_start = dyn_info.entities.dof_start[entity_idx]
                for i_l in range(dyn_info.entities.link_start[entity_idx], dyn_info.entities.link_end[entity_idx]):
                    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                    link_dof_start_local = dyn_info.links.dof_start[I_l] - entity_dof_start
                    link_dof_end_local = dyn_info.links.dof_end[I_l] - entity_dof_start

                    is_resampled = False
                    for i_d_ in range(n_dofs):
                        i_d = targets.dofs_idx[i_d_]
                        if link_dof_start_local <= i_d and i_d < link_dof_end_local:
                            is_resampled = True

                    if is_resampled:
                        for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                            i_d = dyn_info.joints.dof_start[I_j]
                            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d

                            dof_limit = dyn_info.dofs.limit[I_d]
                            if (
                                dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                                or dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                            ) and qd_isfinite(dof_limit):
                                q_start = dyn_info.joints.q_start[I_j]
                                # Counter-based draw keyed on (env, restart, DOF). A qd.random() per-thread stream
                                # would order by the parallel schedule, which makes the resampled configuration, and
                                # thus the returned one, nondeterministic across runs.
                                fk.qpos[q_start - q_offset, i_b] = dof_limit[0] + gu.qd_hash01(
                                    seed, i_b, i_sample, q_start
                                ) * (dof_limit[1] - dof_limit[0])


@qd.kernel(fastcache=True)
def kernel_inverse_kinematics_entity(
    entity_idx: int,
    q_offset: int,
    link_offset: int,
    joint_offset: int,
    seed: int,
    dyn_state: DynState,
    ik_state: IKState,
    fk: IKScratchFK,
    targets: IKTargets,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    n_qs: int,
    entity_n_dofs: int,
    n_links: int,
    n_dofs: int,
    n_envs: int,
    max_samples: int,
    max_solver_iters: int,
    damping: float,
    pos_tol: float,
    rot_tol: float,
    max_step_size: float,
    custom_init_qpos: bool,
    respect_joint_limit: bool,
):
    """Entity-facing launch wrapper forwarding the IK scratch and model scalars to func_inverse_kinematics."""
    func_inverse_kinematics(
        entity_idx,
        q_offset,
        link_offset,
        joint_offset,
        seed,
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
        max_samples,
        max_solver_iters,
        damping,
        pos_tol,
        rot_tol,
        max_step_size,
        custom_init_qpos,
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
    targets: IKTargets,
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
            targets.init_qpos[i_q, envs_idx[i_c]] = init_qpos[i_c, i_q]
    for k in qd.static(range(3)):
        targets.pos_mask[k] = pos_mask[k]
        targets.rot_mask[k] = rot_mask[k]


@qd.kernel(fastcache=True)
def kernel_forward_kinematics_query(
    entity_idx: int,
    qs_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    links_pos: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    qpos: qd.types.ndarray(),
    qpos_cache: qd.Tensor,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
):
    """Link poses of one entity at a queried configuration, leaving the live configuration as it was.

    The queried configuration is written into the solver, forward kinematics runs, the poses are read out, and the
    saved configuration is restored and re-propagated. qpos_cache spans the solver configuration, so the global q
    indices index it directly.
    """
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(rigid_info.qpos))):
        qpos_cache[qs_idx[i_q_], envs_idx[i_b_]] = rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]]
        rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]

    i_l_base = dyn_info.entities.link_start[entity_idx]
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        I_l_base = [i_l_base, i_b] if qd.static(rigid_config.batch_links_info) else i_l_base
        i_l_root = dyn_info.links.root_idx[I_l_base]
        func_forward_kinematics_root(
            i_l_root, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
        )

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(dyn_state.links.pos))):
        for i in qd.static(range(3)):
            links_pos[i_b_, i_l_, i] = dyn_state.links.pos[links_idx[i_l_], envs_idx[i_b_]][i]
        for i in qd.static(range(4)):
            links_quat[i_b_, i_l_, i] = dyn_state.links.quat[links_idx[i_l_], envs_idx[i_b_]][i]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(rigid_info.qpos))):
        rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos_cache[qs_idx[i_q_], envs_idx[i_b_]]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        I_l_base = [i_l_base, i_b] if qd.static(rigid_config.batch_links_info) else i_l_base
        i_l_root = dyn_info.links.root_idx[I_l_base]
        func_forward_kinematics_root(
            i_l_root, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
        )
