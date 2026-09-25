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

from enum import IntEnum

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from .misc import (
    func_add_safe_backward,
    func_atomic_add_if,
    func_check_index_range,
    func_is_awake_link,
    func_is_awake_tree,
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
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_update_cartesian_space_root(
            i_l_root,
            i_b,
            rigid_info.qpos,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            force_update_all_geoms=True,
            is_backward=False,
        )
        func_forward_velocity_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(fastcache=True)
def kernel_masked_forward_kinematics_links_geoms(
    envs_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_mask.shape[0]):
        if envs_mask[i_b]:
            i_l_root = rigid_info.roots_link_idx[i_r]
            func_update_cartesian_space_root(
                i_l_root,
                i_b,
                rigid_info.qpos,
                dyn_state,
                dyn_info,
                rigid_info,
                rigid_config,
                force_update_all_geoms=True,
                is_backward=False,
            )
            func_forward_velocity_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(fastcache=True)
def kernel_forward_kinematics(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_update_kinematics_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config)


@qd.kernel(fastcache=True)
def kernel_masked_forward_kinematics(
    envs_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_mask.shape[0]):
        if envs_mask[i_b]:
            i_l_root = rigid_info.roots_link_idx[i_r]
            func_update_kinematics_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config)


@qd.kernel(fastcache=True)
def kernel_forward_velocity(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_forward_velocity_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_masked_forward_velocity(
    envs_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_mask.shape[0]):
        if envs_mask[i_b]:
            i_l_root = rigid_info.roots_link_idx[i_r]
            func_forward_velocity_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_update_kinematics_root(
    i_l_root,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Compute the poses of the links of kinematic root i_l_root of env i_b, its center of mass and its link velocities.

    One thread walks the whole span of the root (see roots_link_idx in array_class.py), parents first, both its static
    links and those of its trees, so a tree reads the pose of the static link it hangs from after the walk wrote it.
    The velocities read the center of mass, so they follow in a second walk. Runs for a scene simulated without
    dynamics, where no link sleeps.
    """
    func_forward_kinematics_root(
        i_l_root, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
    )
    func_COM_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
    func_forward_velocity_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.func
def func_COM_root_sum(
    i_l_root,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Compute the mass and the center of mass of a kinematic root.

    Sums the links of kinematic root i_l_root of env i_b, reading the world pose of the inertial frame of each, which
    must be current.
    """
    EPS = rigid_info.EPS[None]
    i_l_end = rigid_info.links_root_end[i_l_root]

    dyn_state.links.root_COM_bw[i_l_root, i_b].fill(0.0)
    dyn_state.links.mass_sum[i_l_root, i_b] = 0.0
    # The walk covers the link span of the root and gates each link on its root (see roots_link_idx in array_class.py),
    # so a root spanning several entities, one attached beneath another, sums every link of its own before it divides
    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root:
            mass = dyn_info.links.inertial_mass[I_l]
            dyn_state.links.mass_sum[i_l_root, i_b] = dyn_state.links.mass_sum[i_l_root, i_b] + mass
            dyn_state.links.root_COM_bw[i_l_root, i_b] = (
                dyn_state.links.root_COM_bw[i_l_root, i_b] + mass * dyn_state.links.i_pos_bw[i_l, i_b]
            )

    mass_sum = dyn_state.links.mass_sum[i_l_root, i_b]
    if mass_sum > EPS:
        dyn_state.links.root_COM[i_l_root, i_b] = dyn_state.links.root_COM_bw[i_l_root, i_b] / mass_sum
    else:
        dyn_state.links.root_COM[i_l_root, i_b] = dyn_state.links.i_pos_bw[i_l_root, i_b]


@qd.func
def func_COM_link(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the quantities of a link that follow from the center of mass of its kinematic root.

    Writes, for link i_l of env i_b, the center of mass of its root, its inertia about it, the pose of its joint frame,
    and the motion subspace and velocity of its dofs about the center of mass. The center of mass of the root must be
    current (see func_COM_root_sum).
    """
    EPS = rigid_info.EPS[None]
    BW = qd.static(is_backward)
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    i_l_root = dyn_info.links.root_idx[I_l]
    # The uses below read the copy back from the field rather than a local loaded from the root: autodiff reverses this
    # function (see kernel_COM_links_replay), and on the root link the copy overwrites its own source, which breaks the
    # gradient of a local held across it.
    dyn_state.links.root_COM[i_l, i_b] = dyn_state.links.root_COM[i_l_root, i_b]
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
        EPS,
    )

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
                gu.qd_transform_pos_quat_by_trans_quat(dyn_info.links.pos[I_l], dyn_info.links.quat[I_l], p_pos, p_quat)
            )

            n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]

            for i_j_ in range(n_joints):
                i_j = i_j_ + dyn_info.links.joint_start[I_l]

                curr_i_j = 0 if qd.static(not BW) else i_j_
                next_i_j = 0 if qd.static(not BW) else i_j_ + 1

                if func_check_index_range(i_j, dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l], BW):
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
                dyn_state.dofs.cdofvel_ang[i_d, i_b] = dyn_state.dofs.cdof_ang[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
                dyn_state.dofs.cdofvel_vel[i_d, i_b] = dyn_state.dofs.cdof_vel[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]


@qd.func
def func_COM_root(
    i_l_root,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the center of mass of a kinematic root and the inertia of each of its links about it.

    One call handles kinematic root i_l_root of env i_b in a single thread: the inertial frames of its links, their
    mass-weighted mean position, then the quantities of each link that follow from it (see func_COM_link). A sleeping
    link keeps a valid pose, so the walk reads every link of the root whatever its sleep state. A root that is itself a
    tree root sleeps with its tree and keeps the values of its last awake step. A fixed root never sleeps, so its center
    of mass follows its trees whatever their sleep state.
    """
    i_b = qd.cast(i_b, qd.i32)
    if func_is_awake_link(i_l_root, i_b, dyn_state, rigid_config):
        i_l_end = rigid_info.links_root_end[i_l_root]
        for i_l in range(i_l_root, i_l_end):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if dyn_info.links.root_idx[I_l] == i_l_root:
                (dyn_state.links.i_pos_bw[i_l, i_b], dyn_state.links.i_quat[i_l, i_b]) = (
                    gu.qd_transform_pos_quat_by_trans_quat(
                        dyn_info.links.inertial_pos[I_l],
                        dyn_info.links.inertial_quat[I_l],
                        dyn_state.links.pos[i_l, i_b],
                        dyn_state.links.quat[i_l, i_b],
                    )
                )
        func_COM_root_sum(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config)
        for i_l in range(i_l_root, i_l_end):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if dyn_info.links.root_idx[I_l] == i_l_root:
                func_COM_link(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


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
    """Compute the pose of link i_l of env i_b from its parent's pose and its joint coordinates.

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
    i_p = dyn_info.links.parent_idx[I_l]
    if i_p != -1:
        parent_pos = dyn_state.links.pos[i_p, i_b]
        parent_quat = dyn_state.links.quat[i_p, i_b]
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

        if joint_type == gs.JOINT_TYPE.FREE:
            dyn_state.joints.xanchor[i_j, i_b] = qd.Vector(
                [qpos[q_start, i_b], qpos[q_start + 1, i_b], qpos[q_start + 2, i_b]]
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
            pos_ = qd.Vector([qpos[q_start, i_b], qpos[q_start + 1, i_b], qpos[q_start + 2, i_b]], dt=gs.qd_float)
            quat_ = qd.Vector(
                [qpos[q_start + 3, i_b], qpos[q_start + 4, i_b], qpos[q_start + 5, i_b], qpos[q_start + 6, i_b]],
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
                [qpos[q_start, i_b], qpos[q_start + 1, i_b], qpos[q_start + 2, i_b], qpos[q_start + 3, i_b]],
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
        else:
            dyn_state.dofs.pos[dof_start, i_b] = qpos[q_start, i_b] - rigid_info.qpos0[q_start, i_b]
            pos_ = (
                R(curr_I, pos, dyn_state.links.pos_bw, BW)
                + dyn_state.joints.xaxis[i_j, i_b] * dyn_state.dofs.pos[dof_start, i_b]
            )
            pos = W(next_I, pos_, dyn_state.links.pos_bw, BW)
            # A prismatic joint leaves the link orientation unchanged, but the backward per-joint cache still needs the
            # next slot populated: the final R(quat_bw, I_jf, ...) below reads it in backward mode and would get
            # uninitialized memory (NaN gradients on qpos) otherwise.
            quat = W(next_I, quat, dyn_state.links.quat_bw, BW)

    # Skip link pose update for fixed root links to let users manually overwrite them
    I_jf = (i_l, 0 if qd.static(not BW) else n_joints, i_b)
    if not (i_p == -1 and dyn_info.links.is_fixed[I_l]):
        dyn_state.links.pos[i_l, i_b] = R(I_jf, pos, dyn_state.links.pos_bw, BW)
        dyn_state.links.quat[i_l, i_b] = R(I_jf, quat, dyn_state.links.quat_bw, BW)


@qd.func
def func_forward_kinematics_root(
    i_l_root,
    i_b,
    qpos: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the pose of every link of kinematic root i_l_root of env i_b, parents first, its sleeping links included.

    The callers write the joint coordinates of an entity and read poses back (inverse kinematics, path planning). An
    entity attached to another shares its root (see roots_link_idx in array_class.py), so the walk over the whole root
    carries the attached entities along with the joints written.
    """
    i_l_end = rigid_info.links_root_end[i_l_root]
    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root:
            func_forward_kinematics_link(i_l, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_update_geoms_link(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    force_update_all_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Place the movable geoms of the variant of link i_l active in env i_b on the link.

    Under force_update_all_geoms every geom of the link is placed instead, the fixed ones and those of the variants the
    env holds inactive included, for the link poses imposed from outside the step (build, setters). The verts follow
    lazily (see verts_updated).
    """
    BW = qd.static(is_backward)
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    i_g_start = dyn_info.links.geom_start[I_l]
    i_g_end = dyn_info.links.geom_end[I_l]
    if qd.static(force_update_all_geoms):
        # The variants of a link are spread over the geoms of its entity
        i_e = dyn_info.links.entity_idx[I_l]
        i_g_start = dyn_info.entities.geom_start[i_e]
        i_g_end = dyn_info.entities.geom_end[i_e]

    for i_g in range(i_g_start, i_g_end):
        is_link_geom = True
        if qd.static(force_update_all_geoms):
            is_link_geom = dyn_info.geoms.link_idx[i_g] == i_l
        if is_link_geom and func_check_index_range(i_g, i_g_start, i_g_end, BW):
            if force_update_all_geoms or not dyn_info.geoms.is_fixed[i_g]:
                dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b] = gu.qd_transform_pos_quat_by_trans_quat(
                    dyn_info.geoms.pos[i_g],
                    dyn_info.geoms.quat[i_g],
                    dyn_state.links.pos[i_l, i_b],
                    dyn_state.links.quat[i_l, i_b],
                )
                dyn_state.geoms.verts_updated[i_g, i_b] = False


@qd.func
def func_update_geoms_root(
    i_l_root,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_all_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Place the geoms of every awake link of kinematic root i_l_root of env i_b on their link.

    Every geom is placed under force_update_all_geoms alone (see func_update_geoms_link, func_update_kinematics_root).
    """
    i_l_end = rigid_info.links_root_end[i_l_root]
    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root and func_is_awake_link(i_l, i_b, dyn_state, rigid_config):
            func_update_geoms_link(i_l, i_b, dyn_state, dyn_info, rigid_config, force_update_all_geoms, is_backward)


@qd.func
def func_update_geoms(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_all_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Place the geoms of every env on their links, one thread per awake kinematic tree, or per root under
    force_update_all_geoms.

    The root walk places the fixed geoms of the static links too. See func_update_cartesian_space for the two walks,
    and for why the loops are the outermost ones of their kernel.
    """
    if qd.static(force_update_all_geoms):
        qd.loop_config(name="update_geoms_roots", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
            i_l_root = rigid_info.roots_link_idx[i_r]
            func_update_geoms_root(
                i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms, is_backward
            )
    else:
        qd.loop_config(name="update_geoms", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], dyn_state.links.pos.shape[1]):
            if func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config):
                i_l_start = rigid_info.trees_root_idx[i_t]
                i_l_end = rigid_info.trees_link_end[i_t]
                for i_l in range(i_l_start, i_l_end):
                    if rigid_info.links_tree_idx[i_l] == i_t:
                        func_update_geoms_link(
                            i_l, i_b, dyn_state, dyn_info, rigid_config, force_update_all_geoms, is_backward
                        )


@qd.kernel(fastcache=True)
def kernel_update_geoms(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_all_geoms: qd.template(),
):
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_update_geoms_root(
            i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms, is_backward=False
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
    """Compute the Cartesian velocity of link i_l of env i_b from its parent's velocity and its dof velocities.

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

    i_p = dyn_info.links.parent_idx[I_l]
    if i_p != -1:
        cvel_vel = W(I_j0, dyn_state.links.cd_vel[i_p, i_b], dyn_state.links.cd_vel_bw, BW)
        cvel_ang = W(I_j0, dyn_state.links.cd_ang[i_p, i_b], dyn_state.links.cd_ang_bw, BW)

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
def func_update_acc_link(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    update_cacc: qd.template(),
    is_backward: qd.template(),
):
    """Compute the Cartesian acceleration of a link.

    The acceleration of link i_l of env i_b follows from the acceleration of its parent and the velocities and
    accelerations of its dofs.

    The parent's acceleration must be current. A root link starts from the gravity left uncompensated by its entity.
    """
    BW = qd.static(is_backward)
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    i_p = dyn_info.links.parent_idx[I_l]

    if i_p == -1:
        i_e = dyn_info.links.entity_idx[I_l]
        dyn_state.links.cdd_vel[i_l, i_b] = -rigid_info.gravity[i_b] * (1 - dyn_info.entities.gravity_compensation[i_e])
        dyn_state.links.cdd_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(update_cacc):
            dyn_state.links.cacc_lin[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
            dyn_state.links.cacc_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
    else:
        dyn_state.links.cdd_vel[i_l, i_b] = dyn_state.links.cdd_vel[i_p, i_b]
        dyn_state.links.cdd_ang[i_l, i_b] = dyn_state.links.cdd_ang[i_p, i_b]
        if qd.static(update_cacc):
            dyn_state.links.cacc_lin[i_l, i_b] = dyn_state.links.cacc_lin[i_p, i_b]
            dyn_state.links.cacc_ang[i_l, i_b] = dyn_state.links.cacc_ang[i_p, i_b]

    for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
        # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
        local_cdd_vel = dyn_state.dofs.cdofd_vel[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]
        local_cdd_ang = dyn_state.dofs.cdofd_ang[i_d, i_b] * dyn_state.dofs.vel[i_d, i_b]

        func_add_safe_backward([i_l, i_b], local_cdd_vel, dyn_state.links.cdd_vel, BW)
        func_add_safe_backward([i_l, i_b], local_cdd_ang, dyn_state.links.cdd_ang, BW)
        if qd.static(update_cacc):
            func_add_safe_backward(
                [i_l, i_b],
                local_cdd_vel + dyn_state.dofs.cdof_vel[i_d, i_b] * dyn_state.dofs.acc[i_d, i_b],
                dyn_state.links.cacc_lin,
                BW,
            )
            func_add_safe_backward(
                [i_l, i_b],
                local_cdd_ang + dyn_state.dofs.cdof_ang[i_d, i_b] * dyn_state.dofs.acc[i_d, i_b],
                dyn_state.links.cacc_ang,
                BW,
            )


@qd.func
def func_forward_velocity_root(
    i_l_root,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the Cartesian velocities of every awake link of kinematic root i_l_root of env i_b.

    See func_update_kinematics_root for the root walk. A static link comes out with zero velocity, which the backward
    mode computes all the same to fill the per-joint caches its reverse reads for every link.
    """
    i_l_end = rigid_info.links_root_end[i_l_root]
    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root and func_is_awake_link(i_l, i_b, dyn_state, rigid_config):
            func_forward_velocity_link(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_forward_velocity(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the Cartesian velocities of every env, per awake kinematic tree, or per root in backward mode.

    A tree takes one thread, or one block that sweeps it level by level under enable_tree_level_sweep (see
    func_update_cartesian_space).

    The manual reverse of this pass walks each root leaf to root and reads the per-joint cache of every link of the
    root, static ones included, so backward mode fills that cache with a root walk (see func_forward_velocity_root),
    where the step's forward mode skips the static links, whose velocity is constant. See func_update_cartesian_space
    for why the loops are the outermost ones of their kernel.
    """
    if qd.static(is_backward):
        qd.loop_config(
            name="forward_velocity_roots", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        )
        for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
            i_l_root = rigid_info.roots_link_idx[i_r]
            func_forward_velocity_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    elif qd.static(rigid_config.enable_tree_level_sweep):
        func_sweep_links_by_level(
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            sweep_pass=LINK_SWEEP_PASS.FORWARD_VELOCITY,
            force_update_all_geoms=False,
            update_cacc=False,
            is_backward=is_backward,
        )
    else:
        qd.loop_config(name="forward_velocity", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], dyn_state.links.pos.shape[1]):
            # A hibernated tree keeps the zero velocities it was put to sleep with (see func_hibernate_link in misc.py)
            if func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config):
                i_l_start = rigid_info.trees_root_idx[i_t]
                i_l_end = rigid_info.trees_link_end[i_t]
                for i_l in range(i_l_start, i_l_end):
                    if rigid_info.links_tree_idx[i_l] == i_t:
                        func_forward_velocity_link(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


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
def func_update_cartesian_space_root(
    i_l_root,
    i_b,
    qpos: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_all_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Update the Cartesian space of kinematic root i_l_root of env i_b: link poses and geoms, then center of mass.

    See func_update_kinematics_root for the root walk.
    """
    i_l_end = rigid_info.links_root_end[i_l_root]
    for i_l in range(i_l_root, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l_root and func_is_awake_link(i_l, i_b, dyn_state, rigid_config):
            func_forward_kinematics_link(i_l, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
            func_update_geoms_link(i_l, i_b, dyn_state, dyn_info, rigid_config, force_update_all_geoms, is_backward)
    func_COM_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


class LINK_SWEEP_PASS(IntEnum):
    """Per-link pass run by func_sweep_links_by_level."""

    FORWARD_KINEMATICS = 0
    FORWARD_VELOCITY = 1
    ACCELERATION = 2


@qd.func
def func_sweep_links_by_level(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    sweep_pass: qd.template(),
    force_update_all_geoms: qd.template(),
    update_cacc: qd.template(),
    is_backward: qd.template(),
):
    """Run a per-link pass over every env, sweeping each kinematic tree or root level by level.

    The pass sweep_pass (see LINK_SWEEP_PASS) runs on every link of the awake trees for the forward kinematics and
    velocities, and on every awake link of each root for the accelerations, level by level (see links_depth in
    array_class.py). force_update_all_geoms goes to the forward kinematics and update_cacc to the accelerations.
    """
    BLOCK_DIM = qd.static(rigid_config.level_sweep_block_dim)
    IS_PER_ROOT = qd.static(sweep_pass == LINK_SWEEP_PASS.ACCELERATION)
    _B = dyn_state.links.pos.shape[1]
    n_groups = rigid_info.trees_root_idx.shape[0]
    if qd.static(IS_PER_ROOT):
        n_groups = rigid_info.roots_link_idx.shape[0]
    qd.loop_config(name="sweep_links_by_level", block_dim=BLOCK_DIM)
    for i_flat in range(n_groups * _B * BLOCK_DIM):
        i_group = i_flat // (_B * BLOCK_DIM)
        i_b = (i_flat // BLOCK_DIM) % _B
        tid = i_flat % BLOCK_DIM
        is_awake = True
        i_l_start = 0
        i_l_end = 0
        n_levels = 0
        if qd.static(IS_PER_ROOT):
            i_l_start = rigid_info.roots_link_idx[i_group]
            i_l_end = rigid_info.links_root_end[i_l_start]
            n_levels = rigid_info.roots_n_levels[i_group]
        else:
            # A hibernated tree is skipped, see func_update_cartesian_space and func_forward_velocity
            is_awake = func_is_awake_tree(i_group, i_b, dyn_state, rigid_info, rigid_config)
            i_l_start = rigid_info.trees_root_idx[i_group]
            i_l_end = rigid_info.trees_link_end[i_group]
            n_levels = rigid_info.trees_n_levels[i_group]
        for i_level in range(n_levels):
            if is_awake:
                for i_chunk_ in range((i_l_end - i_l_start + BLOCK_DIM - 1) // BLOCK_DIM):
                    i_l = i_l_start + i_chunk_ * BLOCK_DIM + tid
                    if i_l < i_l_end:
                        is_member = rigid_info.links_depth[i_l] == i_level
                        if qd.static(IS_PER_ROOT):
                            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                            is_member = (
                                is_member
                                and dyn_info.links.root_idx[I_l] == i_l_start
                                and func_is_awake_link(i_l, i_b, dyn_state, rigid_config)
                            )
                        else:
                            is_member = is_member and rigid_info.links_tree_idx[i_l] == i_group
                        if is_member:
                            if qd.static(sweep_pass == LINK_SWEEP_PASS.FORWARD_KINEMATICS):
                                func_forward_kinematics_link(
                                    i_l,
                                    i_b,
                                    rigid_info.qpos,
                                    dyn_state,
                                    dyn_info,
                                    rigid_info,
                                    rigid_config,
                                    is_backward,
                                )
                                func_update_geoms_link(
                                    i_l, i_b, dyn_state, dyn_info, rigid_config, force_update_all_geoms, is_backward
                                )
                            elif qd.static(sweep_pass == LINK_SWEEP_PASS.FORWARD_VELOCITY):
                                func_forward_velocity_link(
                                    i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
                                )
                            else:
                                func_update_acc_link(
                                    i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, update_cacc, is_backward
                                )
            qd.simt.block.sync()


@qd.func
def func_update_cartesian_space(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_all_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Update the Cartesian space of every env: the link poses and geoms, then the centers of mass.

    The link poses and the geom placement run per awake kinematic tree, or per root under force_update_all_geoms and in
    backward mode. A tree takes one thread, or one block that sweeps it level by level under enable_tree_level_sweep.
    The centers of mass take one thread per link and one per root.

    A static link moves only when a setter writes the pose of its fixed root, so the step sweep walks the awake trees
    alone (see trees_root_idx in array_class.py), whose static parents hold still, and a tree sleeps as a unit, so a
    hibernated tree keeps the poses of its last awake step. The initial placement and the backward replay cover the
    static links too and walk each root whole (see func_update_kinematics_root): the manual reverse reads the
    per-joint pose cache of every link of a root. Autodiff takes a kernel made of loops alone, so the loops are the
    outermost ones of their kernel.
    """
    if qd.static(force_update_all_geoms or is_backward):
        qd.loop_config(
            name="update_cartesian_space_roots", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        )
        for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
            i_l_root = rigid_info.roots_link_idx[i_r]
            i_l_end = rigid_info.links_root_end[i_l_root]
            for i_l in range(i_l_root, i_l_end):
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                if dyn_info.links.root_idx[I_l] == i_l_root and func_is_awake_link(i_l, i_b, dyn_state, rigid_config):
                    func_forward_kinematics_link(
                        i_l, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
                    )
                    func_update_geoms_link(
                        i_l, i_b, dyn_state, dyn_info, rigid_config, force_update_all_geoms, is_backward
                    )
    elif qd.static(rigid_config.enable_tree_level_sweep):
        func_sweep_links_by_level(
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            sweep_pass=LINK_SWEEP_PASS.FORWARD_KINEMATICS,
            force_update_all_geoms=force_update_all_geoms,
            update_cacc=False,
            is_backward=is_backward,
        )
    else:
        qd.loop_config(
            name="update_cartesian_space", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        )
        for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], dyn_state.links.pos.shape[1]):
            if func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config):
                i_l_start = rigid_info.trees_root_idx[i_t]
                i_l_end = rigid_info.trees_link_end[i_t]
                for i_l in range(i_l_start, i_l_end):
                    if rigid_info.links_tree_idx[i_l] == i_t:
                        func_forward_kinematics_link(
                            i_l, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
                        )
                        func_update_geoms_link(
                            i_l, i_b, dyn_state, dyn_info, rigid_config, force_update_all_geoms, is_backward
                        )
    # The center of mass runs in three sweeps so that only its mass-weighted sum walks the links of a root serially: one
    # thread per link for the inertial frames, one per root for the sum, one per link for what follows from it. A link
    # follows the sleep state of its root (see func_COM_root).
    qd.loop_config(
        name="update_cartesian_space_com_inertial", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    )
    for i_l, i_b in qd.ndrange(dyn_state.links.pos.shape[0], dyn_state.links.pos.shape[1]):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        i_l_root = dyn_info.links.root_idx[I_l]
        if func_is_awake_link(i_l_root, i_b, dyn_state, rigid_config):
            (dyn_state.links.i_pos_bw[i_l, i_b], dyn_state.links.i_quat[i_l, i_b]) = (
                gu.qd_transform_pos_quat_by_trans_quat(
                    dyn_info.links.inertial_pos[I_l],
                    dyn_info.links.inertial_quat[I_l],
                    dyn_state.links.pos[i_l, i_b],
                    dyn_state.links.quat[i_l, i_b],
                )
            )
    qd.loop_config(
        name="update_cartesian_space_com_sum", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    )
    for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
        i_l_root = rigid_info.roots_link_idx[i_r]
        if func_is_awake_link(i_l_root, i_b, dyn_state, rigid_config):
            func_COM_root_sum(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config)
    qd.loop_config(
        name="update_cartesian_space_com", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    )
    for i_l, i_b in qd.ndrange(dyn_state.links.pos.shape[0], dyn_state.links.pos.shape[1]):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        i_l_root = dyn_info.links.root_idx[I_l]
        if func_is_awake_link(i_l_root, i_b, dyn_state, rigid_config):
            func_COM_link(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_update_cartesian_space(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_all_geoms: qd.template(),
    is_backward: qd.template(),
):
    func_update_cartesian_space(dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms, is_backward)


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
    # No link sleeps under gradients, so the walk of every root covers every link
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_forward_kinematics_root(
            i_l_root, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
        )


@qd.kernel(fastcache=True)
def kernel_update_geoms_replay(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    func_update_geoms(
        dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms=False, is_backward=is_backward
    )


@qd.kernel(fastcache=True)
def kernel_COM_links_replay(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_COM_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
