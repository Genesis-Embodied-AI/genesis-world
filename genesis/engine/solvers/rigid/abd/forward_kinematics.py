"""
Forward kinematics, velocity propagation, and geometry updates for rigid body simulation.

This module contains Quadrants kernels and functions for:
- Forward kinematics computation (link and joint pose updates)
- Velocity propagation through kinematic chains
- Geometry pose and vertex updates
- Center of mass calculations
- AABB updates for collision detection
- Hibernation management for inactive entities
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from .misc import (
    func_atomic_add_if,
    func_check_index_range,
    func_is_tree_awake,
    func_read_field_if,
    func_write_and_read_field_if,
    func_write_field_if,
)


@qd.kernel(fastcache=True)
def kernel_forward_kinematics_links_geoms(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_update_cartesian_space_batch(
            i_b,
            rigid_info.qpos,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            force_update_fixed_geoms=True,
            is_backward=False,
        )
        func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(fastcache=True)
def kernel_masked_forward_kinematics_links_geoms(
    envs_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_update_cartesian_space_batch(
                i_b,
                rigid_info.qpos,
                dyn_state,
                dyn_info,
                rigid_info,
                rigid_config,
                force_update_fixed_geoms=True,
                is_backward=False,
            )
            func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(fastcache=True)
def kernel_forward_kinematics(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_forward_kinematics_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
        func_COM_links(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
        func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(fastcache=True)
def kernel_masked_forward_kinematics(
    envs_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_forward_kinematics_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
            func_COM_links(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
            func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(fastcache=True)
def kernel_forward_velocity(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_masked_forward_velocity(
    envs_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_COM_links(
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the center of mass of every kinematic root of one environment and the inertial of every link about it."""
    i_b = qd.cast(i_b, qd.i32)

    for i_r in range(rigid_info.roots_link_idx.shape[0]):
        func_COM_root(i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_COM_root(
    i_r,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the center of mass of kinematic root i_r of env i_b and the inertial of its links about it.

    The links of a root sleep as a unit, so the root link tells whether they are awake, and a sleeping root keeps the
    values of its last awake step.
    """
    i_l_root = rigid_info.roots_link_idx[i_r]
    is_awake = True
    if qd.static(rigid_config.use_hibernation):
        is_awake = not dyn_state.links.is_hibernated[i_l_root, i_b]
    if is_awake:
        func_COM_links_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_COM_links_root(
    i_l_root,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the center of mass of the links of root link i_l_root and the inertial of each of them about it.

    One call handles the whole root, walking its link span and gating each link on its root (see roots_link_idx in
    array_class.py), so that a root spanning several entities, one attached beneath another, accumulates every link of
    its own before it divides. Mirrors the root walk of func_crb_fold. The callers pass awake roots, whose links sleep
    as a unit, so no link of it is hibernated here.
    """
    EPS = rigid_info.EPS[None]
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)
    i_l_end = rigid_info.links_root_end[i_l_root]

    dyn_state.links.root_COM_bw[i_l_root, i_b].fill(0.0)
    dyn_state.links.mass_sum[i_l_root, i_b] = 0.0

    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root:
            mass = dyn_info.links.inertial_mass[I_l]
            (dyn_state.links.i_pos_bw[i_l, i_b], dyn_state.links.i_quat[i_l, i_b]) = (
                gu.qd_transform_pos_quat_by_trans_quat(
                    dyn_info.links.inertial_pos[I_l],
                    dyn_info.links.inertial_quat[I_l],
                    dyn_state.links.pos[i_l, i_b],
                    dyn_state.links.quat[i_l, i_b],
                )
            )

            dyn_state.links.mass_sum[i_l_root, i_b] = dyn_state.links.mass_sum[i_l_root, i_b] + mass
            dyn_state.links.root_COM_bw[i_l_root, i_b] = (
                dyn_state.links.root_COM_bw[i_l_root, i_b] + mass * dyn_state.links.i_pos_bw[i_l, i_b]
            )

    mass_sum = dyn_state.links.mass_sum[i_l_root, i_b]
    if mass_sum > EPS:
        dyn_state.links.root_COM[i_l_root, i_b] = dyn_state.links.root_COM_bw[i_l_root, i_b] / mass_sum
    else:
        dyn_state.links.root_COM[i_l_root, i_b] = dyn_state.links.i_pos_bw[i_l_root, i_b]

    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root:
            dyn_state.links.root_COM[i_l, i_b] = dyn_state.links.root_COM[i_l_root, i_b]

    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root:
            dyn_state.links.i_pos[i_l, i_b] = dyn_state.links.i_pos_bw[i_l, i_b] - dyn_state.links.root_COM[i_l, i_b]

            (
                dyn_state.links.cinr_inertial[i_l, i_b],
                dyn_state.links.cinr_pos[i_l, i_b],
                dyn_state.links.cinr_quat[i_l, i_b],
                dyn_state.links.cinr_mass[i_l, i_b],
            ) = gu.qd_transform_inertia_by_trans_quat(
                dyn_info.links.inertial_i[I_l],
                dyn_info.links.inertial_mass[I_l],
                dyn_state.links.i_pos[i_l, i_b],
                dyn_state.links.i_quat[i_l, i_b],
                rigid_info.EPS[None],
            )

    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root:
            if dyn_info.links.n_dofs[I_l] > 0:
                i_p = dyn_info.links.parent_idx[I_l]

                _i_j = dyn_info.links.joint_start[I_l]
                _I_j = [_i_j, i_b] if qd.static(rigid_config.batch_joints_info) else _i_j
                joint_type = dyn_info.joints.type[_I_j]

                p_pos = qd.Vector.zero(gs.qd_float, 3)
                p_quat = gu.qd_identity_quat()
                if i_p != -1:
                    p_pos = dyn_state.links.pos[i_p, i_b]
                    p_quat = dyn_state.links.quat[i_p, i_b]

                if joint_type == gs.JOINT_TYPE.FREE or (dyn_info.links.is_fixed[I_l] and i_p == -1):
                    dyn_state.links.j_pos[i_l, i_b] = dyn_state.links.pos[i_l, i_b]
                    dyn_state.links.j_quat[i_l, i_b] = dyn_state.links.quat[i_l, i_b]
                else:
                    (dyn_state.links.j_pos_bw[i_l, 0, i_b], dyn_state.links.j_quat_bw[i_l, 0, i_b]) = (
                        gu.qd_transform_pos_quat_by_trans_quat(
                            dyn_info.links.pos[I_l], dyn_info.links.quat[I_l], p_pos, p_quat
                        )
                    )

                    n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]

                    for i_j_ in range(n_joints):
                        i_j = i_j_ + dyn_info.links.joint_start[I_l]

                        curr_i_j = 0 if qd.static(not BW) else i_j_
                        next_i_j = 0 if qd.static(not BW) else i_j_ + 1

                        if func_check_index_range(
                            i_j, dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l], BW
                        ):
                            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j

                            (
                                dyn_state.links.j_pos_bw[i_l, next_i_j, i_b],
                                dyn_state.links.j_quat_bw[i_l, next_i_j, i_b],
                            ) = gu.qd_transform_pos_quat_by_trans_quat(
                                dyn_info.joints.pos[I_j],
                                gu.qd_identity_quat(),
                                dyn_state.links.j_pos_bw[i_l, curr_i_j, i_b],
                                dyn_state.links.j_quat_bw[i_l, curr_i_j, i_b],
                            )

                    i_j_ = 0 if qd.static(not BW) else n_joints
                    dyn_state.links.j_pos[i_l, i_b] = dyn_state.links.j_pos_bw[i_l, i_j_, i_b]
                    dyn_state.links.j_quat[i_l, i_b] = dyn_state.links.j_quat_bw[i_l, i_j_, i_b]

    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root:
            if dyn_info.links.n_dofs[I_l] > 0:
                for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                    offset_pos = dyn_state.links.root_COM[i_l, i_b] - dyn_state.joints.xanchor[i_j, i_b]
                    I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                    joint_type = dyn_info.joints.type[I_j]

                    dof_start = dyn_info.joints.dof_start[I_j]

                    if joint_type == gs.JOINT_TYPE.REVOLUTE:
                        dyn_state.dofs.cdof_ang[dof_start, i_b] = dyn_state.joints.xaxis[i_j, i_b]
                        dyn_state.dofs.cdof_vel[dof_start, i_b] = dyn_state.joints.xaxis[i_j, i_b].cross(offset_pos)
                    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                        dyn_state.dofs.cdof_ang[dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                        dyn_state.dofs.cdof_vel[dof_start, i_b] = dyn_state.joints.xaxis[i_j, i_b]
                    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                        xmat_T = gu.qd_quat_to_R(dyn_state.links.quat[i_l, i_b], EPS).transpose()
                        for i in qd.static(range(3)):
                            dyn_state.dofs.cdof_ang[i + dof_start, i_b] = xmat_T[i, :]
                            dyn_state.dofs.cdof_vel[i + dof_start, i_b] = xmat_T[i, :].cross(offset_pos)
                    elif joint_type == gs.JOINT_TYPE.FREE:
                        for i in qd.static(range(3)):
                            dyn_state.dofs.cdof_ang[i + dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                            dyn_state.dofs.cdof_vel[i + dof_start, i_b] = qd.Vector.zero(gs.qd_float, 3)
                            dyn_state.dofs.cdof_vel[i + dof_start, i_b][i] = 1.0

                        xmat_T = gu.qd_quat_to_R(dyn_state.links.quat[i_l, i_b], EPS).transpose()
                        for i in qd.static(range(3)):
                            dyn_state.dofs.cdof_ang[i + dof_start + 3, i_b] = xmat_T[i, :]
                            dyn_state.dofs.cdof_vel[i + dof_start + 3, i_b] = xmat_T[i, :].cross(offset_pos)

                    for i_d in range(dof_start, dyn_info.joints.dof_end[I_j]):
                        dyn_state.dofs.cdofvel_ang[i_d, i_b] = (
                            dyn_state.dofs.cdof_ang[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
                        )
                        dyn_state.dofs.cdofvel_vel[i_d, i_b] = (
                            dyn_state.dofs.cdof_vel[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
                        )


@qd.func
def func_forward_kinematics_link(
    i_l,
    i_b,
    qpos: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Pose link i_l of env i_b from its parent's pose and its joint coordinates.

    Also writes the anchor and axis of each of its joints and the dof positions its coordinates map to. The parent's
    pose must be current.
    """
    BW = qd.static(is_backward)
    W = qd.static(func_write_field_if)
    R = qd.static(func_read_field_if)
    WR = qd.static(func_write_and_read_field_if)
    i_b = qd.cast(i_b, qd.i32)

    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    I_l0 = (i_l, 0, i_b)

    pos = W(I_l0, dyn_info.links.pos[I_l], dyn_state.links.pos_bw, BW)
    quat = W(I_l0, dyn_info.links.quat[I_l], dyn_state.links.quat_bw, BW)
    if dyn_info.links.parent_idx[I_l] != -1:
        parent_pos = dyn_state.links.pos[dyn_info.links.parent_idx[I_l], i_b]
        parent_quat = dyn_state.links.quat[dyn_info.links.parent_idx[I_l], i_b]
        pos_ = parent_pos + gu.qd_transform_by_quat(dyn_info.links.pos[I_l], parent_quat)
        quat_ = gu.qd_transform_quat_by_quat(dyn_info.links.quat[I_l], parent_quat)

        pos = W(I_l0, pos_, dyn_state.links.pos_bw, BW)
        quat = W(I_l0, quat_, dyn_state.links.quat_bw, BW)

    n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]

    for i_j_ in range(n_joints):
        i_j = i_j_ + dyn_info.links.joint_start[I_l]

        curr_I = (i_l, 0 if qd.static(not BW) else i_j_, i_b)
        next_I = (i_l, 0 if qd.static(not BW) else i_j_ + 1, i_b)

        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]
        q_start = dyn_info.joints.q_start[I_j]
        dof_start = dyn_info.joints.dof_start[I_j]
        I_d = [dof_start, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start

        # compute axis and anchor
        if joint_type == gs.JOINT_TYPE.FREE:
            dyn_state.joints.xanchor[i_j, i_b] = qd.Vector(
                [
                    qpos[q_start, i_b],
                    qpos[q_start + 1, i_b],
                    qpos[q_start + 2, i_b],
                ]
            )
            dyn_state.joints.xaxis[i_j, i_b] = qd.Vector([0.0, 0.0, 1.0])
        elif joint_type == gs.JOINT_TYPE.FIXED:
            pass
        else:
            axis = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
            if joint_type == gs.JOINT_TYPE.REVOLUTE:
                axis = dyn_info.dofs.motion_ang[I_d]
            elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                axis = dyn_info.dofs.motion_vel[I_d]

            pos_ = R(curr_I, pos, dyn_state.links.pos_bw, BW)
            quat_ = R(curr_I, quat, dyn_state.links.quat_bw, BW)

            dyn_state.joints.xanchor[i_j, i_b] = gu.qd_transform_by_quat(dyn_info.joints.pos[I_j], quat_) + pos_
            dyn_state.joints.xaxis[i_j, i_b] = gu.qd_transform_by_quat(axis, quat_)

        if joint_type == gs.JOINT_TYPE.FREE:
            pos_ = qd.Vector(
                [
                    qpos[q_start, i_b],
                    qpos[q_start + 1, i_b],
                    qpos[q_start + 2, i_b],
                ],
                dt=gs.qd_float,
            )
            quat_ = qd.Vector(
                [
                    qpos[q_start + 3, i_b],
                    qpos[q_start + 4, i_b],
                    qpos[q_start + 5, i_b],
                    qpos[q_start + 6, i_b],
                ],
                dt=gs.qd_float,
            )
            quat_ = quat_ / quat_.norm()
            pos = WR(next_I, pos_, dyn_state.links.pos_bw, BW)
            quat = WR(next_I, quat_, dyn_state.links.quat_bw, BW)

            xyz = gu.qd_quat_to_xyz(quat, rigid_info.EPS[None])
            for j in qd.static(range(3)):
                dyn_state.dofs.pos[dof_start + j, i_b] = pos[j]
                dyn_state.dofs.pos[dof_start + 3 + j, i_b] = xyz[j]
        elif joint_type == gs.JOINT_TYPE.FIXED:
            pass
        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
            qloc = qd.Vector(
                [
                    qpos[q_start, i_b],
                    qpos[q_start + 1, i_b],
                    qpos[q_start + 2, i_b],
                    qpos[q_start + 3, i_b],
                ],
                dt=gs.qd_float,
            )
            xyz = gu.qd_quat_to_xyz(qloc, rigid_info.EPS[None])
            for j in qd.static(range(3)):
                dyn_state.dofs.pos[dof_start + j, i_b] = xyz[j]
            quat_ = gu.qd_transform_quat_by_quat(qloc, R(curr_I, quat, dyn_state.links.quat_bw, BW))
            quat = WR(next_I, quat_, dyn_state.links.quat_bw, BW)
            pos_ = dyn_state.joints.xanchor[i_j, i_b] - gu.qd_transform_by_quat(dyn_info.joints.pos[I_j], quat)
            pos = W(next_I, pos_, dyn_state.links.pos_bw, BW)
        elif joint_type == gs.JOINT_TYPE.REVOLUTE:
            axis = dyn_info.dofs.motion_ang[I_d]
            dyn_state.dofs.pos[dof_start, i_b] = qpos[q_start, i_b] - rigid_info.qpos0[q_start, i_b]
            qloc = gu.qd_rotvec_to_quat(axis * dyn_state.dofs.pos[dof_start, i_b], rigid_info.EPS[None])
            quat_ = gu.qd_transform_quat_by_quat(qloc, R(curr_I, quat, dyn_state.links.quat_bw, BW))
            quat = WR(next_I, quat_, dyn_state.links.quat_bw, BW)
            pos_ = dyn_state.joints.xanchor[i_j, i_b] - gu.qd_transform_by_quat(dyn_info.joints.pos[I_j], quat)
            pos = W(next_I, pos_, dyn_state.links.pos_bw, BW)
        else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
            dyn_state.dofs.pos[dof_start, i_b] = qpos[q_start, i_b] - rigid_info.qpos0[q_start, i_b]
            pos_ = (
                R(curr_I, pos, dyn_state.links.pos_bw, BW)
                + dyn_state.joints.xaxis[i_j, i_b] * dyn_state.dofs.pos[dof_start, i_b]
            )
            pos = W(next_I, pos_, dyn_state.links.pos_bw, BW)
            # A prismatic joint leaves the link orientation unchanged, but the backward per-joint cache still
            # needs the next slot populated: the final R(quat_bw, I_jf, ...) below reads it in backward mode and
            # would get uninitialized memory (NaN gradients on qpos) otherwise.
            quat = W(next_I, quat, dyn_state.links.quat_bw, BW)

    # Skip link pose update for fixed root links to let users manually overwrite them
    I_jf = (i_l, 0 if qd.static(not BW) else n_joints, i_b)
    if not (dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]):
        dyn_state.links.pos[i_l, i_b] = R(I_jf, pos, dyn_state.links.pos_bw, BW)
        dyn_state.links.quat[i_l, i_b] = R(I_jf, quat, dyn_state.links.quat_bw, BW)


@qd.func
def func_forward_kinematics_tree(
    i_t,
    i_b,
    qpos: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Pose the links of kinematic tree i_t of env i_b, parents first (see trees_root_idx in array_class.py).

    A tree sleeps as a unit, so a hibernated tree keeps the poses of its last awake step.
    """
    if func_is_tree_awake(i_t, i_b, dyn_state, rigid_info, rigid_config):
        for i_l in range(rigid_info.trees_root_idx[i_t], rigid_info.trees_link_end[i_t]):
            if rigid_info.links_tree_idx[i_l] == i_t:
                func_forward_kinematics_link(i_l, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_forward_kinematics_static_links(
    i_r,
    i_b,
    qpos: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Pose the static links of kinematic root i_r of env i_b, parents first (see roots_link_idx in array_class.py).

    A static link moves only when a setter writes the pose of its fixed root, so the setters run this pass and the step
    sweep skips it (see func_update_cartesian_space).
    """
    i_l_root = rigid_info.roots_link_idx[i_r]
    for i_l in range(i_l_root, rigid_info.links_root_end[i_l_root]):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root and rigid_info.links_tree_idx[i_l] == -1:
            func_forward_kinematics_link(i_l, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_forward_kinematics_entity(
    i_e,
    i_b,
    qpos: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Pose every link of entity i_e of env i_b, parents first, its sleeping links included.

    The callers write the joint coordinates of the entity and read the poses back (inverse kinematics, path planning).
    """
    for i_l in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
        func_forward_kinematics_link(i_l, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_forward_kinematics_batch(
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Pose every link of env i_b: the static links of each root, then the awake trees."""
    i_b = qd.cast(i_b, qd.i32)

    for i_r in range(rigid_info.roots_link_idx.shape[0]):
        func_forward_kinematics_static_links(
            i_r, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
        )
    for i_t in range(rigid_info.trees_root_idx.shape[0]):
        func_forward_kinematics_tree(
            i_t, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
        )


@qd.kernel(fastcache=True)
def kernel_forward_kinematics_entity(
    i_e: qd.int32,
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)

        func_forward_kinematics_entity(
            i_e, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
        )


@qd.func
def func_update_geoms_link(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Pose the geoms of link i_l of env i_b from its pose, the fixed geoms only under force_update_fixed_geoms.

    The verts follow lazily (see verts_updated).
    """
    BW = qd.static(is_backward)
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    i_g_start = dyn_info.links.geom_start[I_l]
    i_g_end = dyn_info.links.geom_end[I_l]

    for i_g in range(i_g_start, i_g_end):
        if func_check_index_range(i_g, i_g_start, i_g_end, BW):
            if force_update_fixed_geoms or not dyn_info.geoms.is_fixed[i_g]:
                dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b] = gu.qd_transform_pos_quat_by_trans_quat(
                    dyn_info.geoms.pos[i_g],
                    dyn_info.geoms.quat[i_g],
                    dyn_state.links.pos[i_l, i_b],
                    dyn_state.links.quat[i_l, i_b],
                )
                dyn_state.geoms.verts_updated[i_g, i_b] = False


@qd.func
def func_update_geoms_tree(
    i_t,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Pose the geoms of the links of kinematic tree i_t of env i_b.

    A hibernated tree keeps the geom poses of its last awake step (see func_forward_kinematics_tree).
    """
    if func_is_tree_awake(i_t, i_b, dyn_state, rigid_info, rigid_config):
        for i_l in range(rigid_info.trees_root_idx[i_t], rigid_info.trees_link_end[i_t]):
            if rigid_info.links_tree_idx[i_l] == i_t:
                func_update_geoms_link(
                    i_l, i_b, dyn_state, dyn_info, rigid_config, force_update_fixed_geoms, is_backward
                )


@qd.func
def func_update_geoms_static_links(
    i_r,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Pose the geoms of the static links of kinematic root i_r of env i_b (see func_forward_kinematics_static_links)."""
    i_l_root = rigid_info.roots_link_idx[i_r]
    for i_l in range(i_l_root, rigid_info.links_root_end[i_l_root]):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root and rigid_info.links_tree_idx[i_l] == -1:
            func_update_geoms_link(i_l, i_b, dyn_state, dyn_info, rigid_config, force_update_fixed_geoms, is_backward)


@qd.func
def func_update_geoms_batch(
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Pose the geoms of every awake link of env i_b, and those of the static links under force_update_fixed_geoms."""
    i_b = qd.cast(i_b, qd.i32)

    if qd.static(force_update_fixed_geoms):
        for i_r in range(rigid_info.roots_link_idx.shape[0]):
            func_update_geoms_static_links(
                i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
            )
    for i_t in range(rigid_info.trees_root_idx.shape[0]):
        func_update_geoms_tree(
            i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
        )


@qd.func
def func_update_geoms(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Pose the geoms of every env, one thread per kinematic tree.

    The loops are the outermost ones of their kernel, which differentiability requires.
    """
    if qd.static(force_update_fixed_geoms):
        qd.loop_config(name="update_geoms_static", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
            func_update_geoms_static_links(
                i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
            )
    qd.loop_config(name="update_geoms", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], dyn_state.links.pos.shape[1]):
        func_update_geoms_tree(
            i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
        )


@qd.kernel(fastcache=True)
def kernel_update_geoms(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)

        func_update_geoms_batch(
            i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward=False
        )


@qd.func
def func_forward_velocity_link(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Cartesian velocity of link i_l of env i_b from its parent's velocity and its dof velocities.

    Also writes the velocity derivative of the motion subspace of each of its dofs. The parent's velocity must be
    current.
    """
    BW = qd.static(is_backward)
    W = qd.static(func_write_field_if)
    R = qd.static(func_read_field_if)
    A = qd.static(func_atomic_add_if)
    i_b = qd.cast(i_b, qd.i32)

    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]

    I_j0 = (i_l, 0, i_b)
    cvel_vel = W(I_j0, qd.Vector.zero(gs.qd_float, 3), dyn_state.links.cd_vel_bw, BW)
    cvel_ang = W(I_j0, qd.Vector.zero(gs.qd_float, 3), dyn_state.links.cd_ang_bw, BW)

    if dyn_info.links.parent_idx[I_l] != -1:
        cvel_vel = W(I_j0, dyn_state.links.cd_vel[dyn_info.links.parent_idx[I_l], i_b], dyn_state.links.cd_vel_bw, BW)
        cvel_ang = W(I_j0, dyn_state.links.cd_ang[dyn_info.links.parent_idx[I_l], i_b], dyn_state.links.cd_ang_bw, BW)

    for i_j_ in range(n_joints):
        i_j = i_j_ + dyn_info.links.joint_start[I_l]

        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]
        dof_start = dyn_info.joints.dof_start[I_j]

        curr_I = (i_l, 0 if qd.static(not BW) else i_j_, i_b)
        next_I = (i_l, 0 if qd.static(not BW) else i_j_ + 1, i_b)

        if joint_type == gs.JOINT_TYPE.FREE:
            for i_3 in qd.static(range(3)):
                _vel = dyn_state.dofs.cdof_vel[dof_start + i_3, i_b] * dyn_state.dofs.vel[dof_start + i_3, i_b]
                _ang = dyn_state.dofs.cdof_ang[dof_start + i_3, i_b] * dyn_state.dofs.vel[dof_start + i_3, i_b]

                cvel_vel = cvel_vel + A(curr_I, _vel, dyn_state.links.cd_vel_bw, BW)
                cvel_ang = cvel_ang + A(curr_I, _ang, dyn_state.links.cd_ang_bw, BW)

            for i_3 in qd.static(range(3)):
                dyn_state.dofs.cdofd_ang[dof_start + i_3, i_b], dyn_state.dofs.cdofd_vel[dof_start + i_3, i_b] = (
                    qd.Vector.zero(gs.qd_float, 3),
                    qd.Vector.zero(gs.qd_float, 3),
                )

                (
                    dyn_state.dofs.cdofd_ang[dof_start + i_3 + 3, i_b],
                    dyn_state.dofs.cdofd_vel[dof_start + i_3 + 3, i_b],
                ) = gu.motion_cross_motion(
                    R(curr_I, cvel_ang, dyn_state.links.cd_ang_bw, BW),
                    R(curr_I, cvel_vel, dyn_state.links.cd_vel_bw, BW),
                    dyn_state.dofs.cdof_ang[dof_start + i_3 + 3, i_b],
                    dyn_state.dofs.cdof_vel[dof_start + i_3 + 3, i_b],
                )

            if qd.static(BW):
                dyn_state.links.cd_vel_bw[next_I] = dyn_state.links.cd_vel_bw[curr_I]
                dyn_state.links.cd_ang_bw[next_I] = dyn_state.links.cd_ang_bw[curr_I]

            for i_3 in qd.static(range(3)):
                _vel = dyn_state.dofs.cdof_vel[dof_start + i_3 + 3, i_b] * dyn_state.dofs.vel[dof_start + i_3 + 3, i_b]
                _ang = dyn_state.dofs.cdof_ang[dof_start + i_3 + 3, i_b] * dyn_state.dofs.vel[dof_start + i_3 + 3, i_b]
                cvel_vel = cvel_vel + A(next_I, _vel, dyn_state.links.cd_vel_bw, BW)
                cvel_ang = cvel_ang + A(next_I, _ang, dyn_state.links.cd_ang_bw, BW)

        else:
            for i_d in range(dof_start, dyn_info.joints.dof_end[I_j]):
                dyn_state.dofs.cdofd_ang[i_d, i_b], dyn_state.dofs.cdofd_vel[i_d, i_b] = gu.motion_cross_motion(
                    R(curr_I, cvel_ang, dyn_state.links.cd_ang_bw, BW),
                    R(curr_I, cvel_vel, dyn_state.links.cd_vel_bw, BW),
                    dyn_state.dofs.cdof_ang[i_d, i_b],
                    dyn_state.dofs.cdof_vel[i_d, i_b],
                )

            if qd.static(BW):
                dyn_state.links.cd_vel_bw[next_I] = dyn_state.links.cd_vel_bw[curr_I]
                dyn_state.links.cd_ang_bw[next_I] = dyn_state.links.cd_ang_bw[curr_I]

            for i_d in range(dof_start, dyn_info.joints.dof_end[I_j]):
                _vel = dyn_state.dofs.cdof_vel[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
                _ang = dyn_state.dofs.cdof_ang[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
                cvel_vel = cvel_vel + A(next_I, _vel, dyn_state.links.cd_vel_bw, BW)
                cvel_ang = cvel_ang + A(next_I, _ang, dyn_state.links.cd_ang_bw, BW)

    I_jf = (i_l, 0 if qd.static(not BW) else n_joints, i_b)
    dyn_state.links.cd_vel[i_l, i_b] = R(I_jf, cvel_vel, dyn_state.links.cd_vel_bw, BW)
    dyn_state.links.cd_ang[i_l, i_b] = R(I_jf, cvel_ang, dyn_state.links.cd_ang_bw, BW)


@qd.func
def func_forward_velocity_tree(
    i_t,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Cartesian velocities of the links of kinematic tree i_t of env i_b, parents first.

    A hibernated tree keeps the zero velocities it was put to sleep with (see func_hibernate_link in misc.py).
    """
    if func_is_tree_awake(i_t, i_b, dyn_state, rigid_info, rigid_config):
        for i_l in range(rigid_info.trees_root_idx[i_t], rigid_info.trees_link_end[i_t]):
            if rigid_info.links_tree_idx[i_l] == i_t:
                func_forward_velocity_link(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_forward_velocity_static_links(
    i_r,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Cartesian velocities of the static links of kinematic root i_r of env i_b, zero by construction.

    The backward mode runs it to fill the per-joint caches the reverse reads for every link.
    """
    i_l_root = rigid_info.roots_link_idx[i_r]
    for i_l in range(i_l_root, rigid_info.links_root_end[i_l_root]):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root and rigid_info.links_tree_idx[i_l] == -1:
            func_forward_velocity_link(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_forward_velocity_batch(
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Cartesian velocities of every awake link of env i_b.

    A static link keeps its zero velocity, so the forward sweep skips it (see func_forward_velocity_static_links).
    """
    i_b = qd.cast(i_b, qd.i32)

    if qd.static(is_backward):
        for i_r in range(rigid_info.roots_link_idx.shape[0]):
            func_forward_velocity_static_links(i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    for i_t in range(rigid_info.trees_root_idx.shape[0]):
        func_forward_velocity_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_forward_velocity(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Cartesian velocities of every env, one thread per kinematic tree (see func_forward_velocity_batch).

    The loops are the outermost ones of their kernel, which differentiability requires.
    """
    if qd.static(is_backward):
        qd.loop_config(
            name="forward_velocity_static", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        )
        for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
            func_forward_velocity_static_links(i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    qd.loop_config(name="forward_velocity", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], dyn_state.links.pos.shape[1]):
        func_forward_velocity_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_update_verts_for_geoms(
    geoms_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    n_geoms = geoms_idx.shape[0]
    _B = dyn_state.geoms.verts_updated.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g_, i_b in qd.ndrange(n_geoms, _B):
        i_g = geoms_idx[i_g_]
        func_update_verts_for_geom(i_g, i_b, dyn_state, dyn_info)


@qd.func
def func_update_verts_for_geom(
    i_g: qd.i32, i_b: qd.i32, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo
):
    _B = dyn_state.geoms.verts_updated.shape[1]

    if not dyn_state.geoms.verts_updated[i_g, i_b]:
        i_v_start = dyn_info.geoms.vert_start[i_g]
        if dyn_info.verts.is_fixed[i_v_start]:
            for i_v in range(i_v_start, dyn_info.geoms.vert_end[i_g]):
                verts_state_idx = dyn_info.verts.verts_state_idx[i_v]
                dyn_state.fixed_verts.pos[verts_state_idx] = gu.qd_transform_by_trans_quat(
                    dyn_info.verts.init_pos[i_v], dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]
                )
            for j_b in range(_B):
                dyn_state.geoms.verts_updated[i_g, j_b] = True
        else:
            for i_v in range(i_v_start, dyn_info.geoms.vert_end[i_g]):
                verts_state_idx = dyn_info.verts.verts_state_idx[i_v]
                dyn_state.free_verts.pos[verts_state_idx, i_b] = gu.qd_transform_by_trans_quat(
                    dyn_info.verts.init_pos[i_v], dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]
                )
            dyn_state.geoms.verts_updated[i_g, i_b] = True


@qd.func
def func_update_all_verts(dyn_state: array_class.DynState, dyn_info: array_class.DynInfo, rigid_config: qd.template()):
    n_geoms, _B = dyn_state.geoms.pos.shape

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        func_update_verts_for_geom(i_g, i_b, dyn_state, dyn_info)


@qd.kernel(fastcache=True)
def kernel_update_all_verts(
    dyn_state: array_class.DynState, dyn_info: array_class.DynInfo, rigid_config: qd.template()
):
    func_update_all_verts(dyn_state, dyn_info, rigid_config)


@qd.kernel(fastcache=True)
def kernel_update_geom_aabbs(
    geoms_init_AABB: array_class.GeomsInitAABB, dyn_state: array_class.DynState, rigid_config: qd.template()
):
    n_geoms = dyn_state.geoms.pos.shape[0]
    _B = dyn_state.geoms.pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        g_pos = dyn_state.geoms.pos[i_g, i_b]
        g_quat = dyn_state.geoms.quat[i_g, i_b]

        lower = gu.qd_vec3(qd.math.inf)
        upper = gu.qd_vec3(-qd.math.inf)
        for i_corner in qd.static(range(8)):
            corner_pos = gu.qd_transform_by_trans_quat(geoms_init_AABB[i_g, i_corner], g_pos, g_quat)
            lower = qd.min(lower, corner_pos)
            upper = qd.max(upper, corner_pos)

        dyn_state.geoms.aabb_min[i_g, i_b] = lower
        dyn_state.geoms.aabb_max[i_g, i_b] = upper


@qd.kernel(fastcache=True)
def kernel_update_vgeoms(dyn_state: array_class.DynState, dyn_info: array_class.DynInfo, rigid_config: qd.template()):
    """
    Vgeoms are only for visualization purposes. Updates vgeom world transforms from link state.
    """
    n_vgeoms = dyn_info.vgeoms.link_idx.shape[0]
    _B = dyn_state.links.pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in qd.ndrange(n_vgeoms, _B):
        i_l = dyn_info.vgeoms.link_idx[i_g]
        dyn_state.vgeoms.pos[i_g, i_b], dyn_state.vgeoms.quat[i_g, i_b] = gu.qd_transform_pos_quat_by_trans_quat(
            dyn_info.vgeoms.pos[i_g],
            dyn_info.vgeoms.quat[i_g],
            dyn_state.links.pos[i_l, i_b],
            dyn_state.links.quat[i_l, i_b],
        )


@qd.kernel(fastcache=True)
def kernel_update_vverts_for_vgeoms(
    vgeoms_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """
    Refresh vverts_state.pos for the requested vgeom range from FK output. Only iterates vverts that have a slot in
    the custom buffer (vverts_state_idx != -1); other vverts are computed on the fly by their consumers, so they have
    no persistent storage here.
    """
    n_vgeoms_in = vgeoms_idx.shape[0]
    _B = dyn_state.vgeoms.pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_vg_, i_b in qd.ndrange(n_vgeoms_in, _B):
        i_vg = vgeoms_idx[i_vg_]
        v_start = dyn_info.vgeoms.vvert_start[i_vg]
        v_end = dyn_info.vgeoms.vvert_end[i_vg]
        for i_vv in range(v_start, v_end):
            i_state = dyn_info.vverts.vverts_state_idx[i_vv]
            if i_state >= 0:
                dyn_state.vverts.pos[i_state, i_b] = gu.qd_transform_by_trans_quat(
                    dyn_info.vverts.init_pos[i_vv], dyn_state.vgeoms.pos[i_vg, i_b], dyn_state.vgeoms.quat[i_vg, i_b]
                )


@qd.func
def func_update_cartesian_space_batch(
    i_b,
    qpos: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Update the Cartesian space of one environment: the static links, the awake trees, the centers of mass of the
    roots, then the geoms."""
    i_b = qd.cast(i_b, qd.i32)

    for i_r in range(rigid_info.roots_link_idx.shape[0]):
        func_forward_kinematics_static_links(i_r, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    for i_t in range(rigid_info.trees_root_idx.shape[0]):
        func_forward_kinematics_tree(i_t, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_COM_links(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_update_geoms_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward)


@qd.func
def func_update_cartesian_space(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Update the Cartesian space of every env, one thread per kinematic tree.

    Each awake tree poses its links and geoms, then each root computes its centers of mass, which span every tree of
    the root. A static link moves through the setters alone, which run the per-env pass with force_update_fixed_geoms
    (see func_update_cartesian_space_batch), so the sweep poses the static links under that flag and in backward mode
    only, whose per-joint caches the reverse reads for every link. The loops are the outermost ones of their kernel,
    which differentiability requires.
    """
    # Autodiff takes a kernel made of loops alone, so the loop bounds are read inline
    if qd.static(force_update_fixed_geoms or is_backward):
        qd.loop_config(
            name="update_cartesian_space_static", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        )
        for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
            func_forward_kinematics_static_links(
                i_r, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
            )
            func_update_geoms_static_links(
                i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
            )
    qd.loop_config(name="update_cartesian_space", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], dyn_state.links.pos.shape[1]):
        func_forward_kinematics_tree(
            i_t, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
        )
        func_update_geoms_tree(
            i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
        )
    qd.loop_config(
        name="update_cartesian_space_com", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    )
    for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
        func_COM_root(i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_update_cartesian_space(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    func_update_cartesian_space(dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward)


# Standalone forward-replay kernels for the update_cartesian_space sub-stages, used only in the backward pass
# (substep_pre_coupling_grad). The backward unroll replays each sub-stage with is_backward=True (static loops) and
# then reverses it, stage by stage: COM-links and geom-pose updates reverse cleanly through Quadrants autodiff (their
# replay kernel's .grad is called directly), while forward kinematics and forward velocity are reversed manually
# (see kernel_manual_forward_kinematics_bw / kernel_manual_forward_velocity_bw in manual_bw.py) because autodiff
# silently drops their gradient. TODO: once every sub-stage reverses correctly under autodiff, drop these kernels and
# the manual reverses, and differentiate kernel_update_cartesian_space as a whole.
@qd.kernel(fastcache=True)
def kernel_forward_kinematics_replay(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_forward_kinematics_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_update_geoms_replay(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    func_update_geoms(
        dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms=False, is_backward=is_backward
    )


@qd.kernel(fastcache=True)
def kernel_COM_links_replay(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    for i_b in range(dyn_state.links.pos.shape[1]):
        func_COM_links(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
