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
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .misc import (
    func_check_index_range,
    func_read_field_if,
    func_write_field_if,
    func_write_and_read_field_if,
    func_atomic_add_if,
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
    """Compute the center of mass of every kinematic tree of one environment and the inertial of every link about it."""
    i_b = qd.cast(i_b, qd.i32)

    for i_t in range(rigid_info.trees_root_idx.shape[0]):
        # A tree sleeps as a unit, so its root tells whether it is awake
        i_l_root = rigid_info.trees_root_idx[i_t]
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.links.is_hibernated[i_l_root, i_b]
        if is_awake:
            func_COM_links_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_COM_links_tree(
    i_t,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Compute the center of mass of one kinematic tree and the inertial of each of its links about it.

    One call handles the whole tree, walking its link span and gating each link on its root (see trees_root_idx in
    array_class.py), so that a tree spanning several entities, one attached beneath another, accumulates every link of
    its own before it divides. Mirrors the tree walk of func_crb_fold. The callers pass awake trees, and a tree sleeps
    as a unit, so no link of it is hibernated here.
    """
    EPS = rigid_info.EPS[None]
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)
    i_l_root = rigid_info.trees_root_idx[i_t]
    i_l_end = rigid_info.trees_link_end[i_t]

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
    BW = qd.static(is_backward)
    W = qd.static(func_write_field_if)
    R = qd.static(func_read_field_if)
    WR = qd.static(func_write_and_read_field_if)
    i_b = qd.cast(i_b, qd.i32)

    for i_l_ in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
        i_l = gs.qd_int(i_l_)
        # A hibernated link's pose is frozen and still valid, so skip recomputing it. All links of a component sleep
        # together, so a hibernated link never has an awake child whose pose depends on it.
        if qd.static(rigid_config.use_hibernation):
            if dyn_state.links.is_hibernated[i_l, i_b]:
                continue

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
def func_forward_kinematics_batch(
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_e_ in (
        range(rigid_info.n_awake_entities[i_b])
        if qd.static(rigid_config.use_hibernation)
        else range(dyn_info.entities.n_links.shape[0])
    ):
        if func_check_index_range(i_e_, 0, rigid_info.n_awake_entities[i_b], rigid_config.use_hibernation):
            i_e = rigid_info.awake_entities[i_e_, i_b] if qd.static(rigid_config.use_hibernation) else i_e_

            func_forward_kinematics_entity(
                i_e, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
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
def func_update_geoms_entity(
    i_e,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_g_ in (
        # Dynamic inner loop for forward pass
        range(dyn_info.entities.n_geoms[i_e])
        if qd.static(not BW)
        else qd.static(range(rigid_config.max_n_geoms_per_entity))  # Static inner loop for backward pass
    ):
        i_g = dyn_info.entities.geom_start[i_e] + i_g_
        if qd.static(rigid_config.use_hibernation):
            if dyn_state.geoms.is_hibernated[i_g, i_b]:
                continue
        if func_check_index_range(i_g, dyn_info.entities.geom_start[i_e], dyn_info.entities.geom_end[i_e], BW):
            if force_update_fixed_geoms or not dyn_info.geoms.is_fixed[i_g]:
                dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b] = gu.qd_transform_pos_quat_by_trans_quat(
                    dyn_info.geoms.pos[i_g],
                    dyn_info.geoms.quat[i_g],
                    dyn_state.links.pos[dyn_info.geoms.link_idx[i_g], i_b],
                    dyn_state.links.quat[dyn_info.geoms.link_idx[i_g], i_b],
                )
                dyn_state.geoms.verts_updated[i_g, i_b] = False


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
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_e_ in (
        range(rigid_info.n_awake_entities[i_b])
        if qd.static(rigid_config.use_hibernation)
        else range(dyn_info.entities.n_links.shape[0])
    ):
        if func_check_index_range(i_e_, 0, rigid_info.n_awake_entities[i_b], rigid_config.use_hibernation):
            i_e = rigid_info.awake_entities[i_e_, i_b] if qd.static(rigid_config.use_hibernation) else i_e_

            func_update_geoms_entity(
                i_e, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
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
    # This loop must be the outermost loop to be differentiable
    if qd.static(rigid_config.use_hibernation):
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(dyn_state.links.pos.shape[1]):
            func_update_geoms_batch(
                i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
            )
    else:
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.links.pos.shape[1]):
            func_update_geoms_entity(
                i_e, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
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
def func_forward_velocity_entity(
    i_e,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    W = qd.static(func_write_field_if)
    R = qd.static(func_read_field_if)
    A = qd.static(func_atomic_add_if)
    i_b = qd.cast(i_b, qd.i32)

    for i_l_ in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
        i_l = gs.qd_int(i_l_)
        # A hibernated link's velocity is zero and frozen; skip it. Components sleep as a unit, so a hibernated link
        # never has an awake child whose velocity propagates from it.
        if qd.static(rigid_config.use_hibernation):
            if dyn_state.links.is_hibernated[i_l, i_b]:
                continue

        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        n_joints = dyn_info.links.joint_end[I_l] - dyn_info.links.joint_start[I_l]

        I_j0 = (i_l, 0, i_b)
        cvel_vel = W(I_j0, qd.Vector.zero(gs.qd_float, 3), dyn_state.links.cd_vel_bw, BW)
        cvel_ang = W(I_j0, qd.Vector.zero(gs.qd_float, 3), dyn_state.links.cd_ang_bw, BW)

        if dyn_info.links.parent_idx[I_l] != -1:
            cvel_vel = W(
                I_j0, dyn_state.links.cd_vel[dyn_info.links.parent_idx[I_l], i_b], dyn_state.links.cd_vel_bw, BW
            )
            cvel_ang = W(
                I_j0, dyn_state.links.cd_ang[dyn_info.links.parent_idx[I_l], i_b], dyn_state.links.cd_ang_bw, BW
            )

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
                    _vel = (
                        dyn_state.dofs.cdof_vel[dof_start + i_3 + 3, i_b] * dyn_state.dofs.vel[dof_start + i_3 + 3, i_b]
                    )
                    _ang = (
                        dyn_state.dofs.cdof_ang[dof_start + i_3 + 3, i_b] * dyn_state.dofs.vel[dof_start + i_3 + 3, i_b]
                    )
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
def func_forward_velocity_batch(
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    for i_e_ in (
        range(rigid_info.n_awake_entities[i_b])
        if qd.static(rigid_config.use_hibernation)
        else range(dyn_info.entities.n_links.shape[0])
    ):
        if func_check_index_range(i_e_, 0, rigid_info.n_awake_entities[i_b], rigid_config.use_hibernation):
            i_e = rigid_info.awake_entities[i_e_, i_b] if qd.static(rigid_config.use_hibernation) else i_e_

            func_forward_velocity_entity(i_e, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_forward_velocity(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    # This loop must be the outermost loop to be differentiable
    if qd.static(rigid_config.use_hibernation):
        qd.loop_config(name="forward_velocity_batch", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(dyn_state.links.pos.shape[1]):
            func_forward_velocity_batch(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    else:
        qd.loop_config(
            name="forward_velocity_entity", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        )
        for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.links.pos.shape[1]):
            func_forward_velocity_entity(i_e, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


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
def func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    _B = dyn_state.links.is_hibernated.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for i_island in range(constraint_state.island.n_islands[i_b]):
            was_island_hibernated = constraint_state.island.is_hibernated[i_island, i_b]

            if not was_island_hibernated:
                are_all_links_ready_to_sleep = True
                link_ref_n = constraint_state.island.link_slices.n[i_island, i_b]
                link_ref_start = constraint_state.island.link_slices.start[i_island, i_b]

                # Invariant check: ensure link_id access won't exceed buffer
                if link_ref_start + link_ref_n > constraint_state.island.link_id.shape[0]:
                    errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_HIBERNATION_ISLANDS
                    continue

                max_vel_thresh = rigid_info.hibernation_thresh_vel[None]
                for i_link_ref_offset_ in range(link_ref_n):
                    link_ref = link_ref_start + i_link_ref_offset_
                    link_idx = constraint_state.island.link_id[link_ref, i_b]

                    # Hibernated links already have zero velocity.
                    if dyn_state.links.is_hibernated[link_idx, i_b]:
                        continue

                    # A link is ready to sleep once its maximum DOF speed has stayed below the tolerance for
                    # hibernation_min_steps consecutive steps. Every awake link is visited each step so its counter
                    # stays current even when its island will not sleep this step; the loop never breaks early. Each
                    # DOF velocity is weighted by dofs_info.dof_length (1 for translation, the swept radius for
                    # rotation) so the tolerance is a single linear speed across mixed DOFs: rotational jitter of a
                    # small body produces a tiny surface speed and no longer keeps it awake.
                    min_steps = qd.static(rigid_config.hibernation_min_steps)
                    link_I = [link_idx, i_b] if qd.static(rigid_config.batch_links_info) else link_idx
                    max_vel = gs.qd_float(0.0)
                    for i_d in range(dyn_info.links.dof_start[link_I], dyn_info.links.dof_end[link_I]):
                        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                        max_vel = qd.max(max_vel, dyn_info.dofs.dof_length[I_d] * qd.abs(dyn_state.dofs.vel[i_d, i_b]))

                    if max_vel < max_vel_thresh:
                        if dyn_state.links.awake_steps[link_idx, i_b] < min_steps:
                            dyn_state.links.awake_steps[link_idx, i_b] = dyn_state.links.awake_steps[link_idx, i_b] + 1
                    else:
                        dyn_state.links.awake_steps[link_idx, i_b] = 0

                    if dyn_state.links.awake_steps[link_idx, i_b] < min_steps:
                        are_all_links_ready_to_sleep = False

                # Hibernate the whole island (component) once all its links are ready to sleep. The awake-island
                # sort-buffer refresh that used to live in the other branch is now handled by the broad phase, which
                # refreshes every awake geom's extents each step regardless of hibernation.
                if are_all_links_ready_to_sleep and link_ref_n > 0:
                    prev_link_idx = constraint_state.island.link_id[link_ref_start + link_ref_n - 1, i_b]

                    for i_link_ref_offset_ in range(link_ref_n):
                        link_ref = link_ref_start + i_link_ref_offset_
                        link_idx = constraint_state.island.link_id[link_ref, i_b]

                        func_hibernate_link_and_zero_dof_velocities(link_idx, i_b, dyn_state, dyn_info, rigid_config)

                        # store links of the hibernated island by daisy chaining them
                        constraint_state.island.hibernated_next_link[prev_link_idx, i_b] = link_idx
                        prev_link_idx = link_idx


@qd.func
def func_aggregate_awake_entities(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_entities = dyn_state.entities.is_hibernated.shape[0]
    n_links = dyn_state.links.is_hibernated.shape[0]
    _B = dyn_state.entities.is_hibernated.shape[1]

    # Recompute each entity's hibernation flag from its links: with per-component islands a single entity's free bodies
    # can sleep independently, so the entity is hibernated only when every one of its movable links is. Fixed (welded
    # to the world) links never hibernate, so they are ignored - otherwise a ground plane living in a multi-free-body
    # entity's worldbody would keep that entity awake forever and force its whole forward-kinematics pass every step.
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_b in qd.ndrange(n_entities, _B):
        are_all_links_hibernated = True
        for i_l in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
            link_idx = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if not dyn_info.links.is_fixed[link_idx] and not dyn_state.links.is_hibernated[i_l, i_b]:
                are_all_links_hibernated = False
                break
        dyn_state.entities.is_hibernated[i_e, i_b] = are_all_links_hibernated

    # Reset counts once per batch (not per entity!)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        rigid_info.n_awake_entities[i_b] = 0
        rigid_info.n_awake_links[i_b] = 0
        rigid_info.n_awake_dofs[i_b] = 0

    # Awake links and their DOFs are gathered per-link, so a partially-awake entity contributes only its awake
    # components (the forward-dynamics passes iterate these lists and skip the sleeping ones).
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if dyn_state.links.is_hibernated[i_l, i_b]:
            continue

        next_awake_link_idx = qd.atomic_add(rigid_info.n_awake_links[i_b], 1)
        rigid_info.awake_links[next_awake_link_idx, i_b] = i_l

        link_I = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        n_dofs = dyn_info.links.n_dofs[link_I]
        if n_dofs > 0:
            link_dofs_base_idx = dyn_info.links.dof_start[link_I]
            awake_dofs_base_idx = qd.atomic_add(rigid_info.n_awake_dofs[i_b], n_dofs)
            for i_d_ in range(n_dofs):
                rigid_info.awake_dofs[awake_dofs_base_idx + i_d_, i_b] = link_dofs_base_idx + i_d_

    # Awake entities (the entity-level forward-kinematics passes traverse the whole entity tree, so an entity is awake
    # whenever any of its links is - i.e. it is not fully hibernated).
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_b in qd.ndrange(n_entities, _B):
        if dyn_state.entities.is_hibernated[i_e, i_b] or dyn_info.entities.n_dofs[i_e] == 0:
            continue

        next_awake_entity_idx = qd.atomic_add(rigid_info.n_awake_entities[i_b], 1)
        rigid_info.awake_entities[next_awake_entity_idx, i_b] = i_e


@qd.func
def func_hibernate_link_and_zero_dof_velocities(
    i_l: int, i_b: int, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo, rigid_config: qd.template()
):
    """Mark a link, its DOFs, and its geoms as hibernated, and zero out the DOF and link velocities and accelerations.

    The next-velocity buffer is zeroed too: the integration copy runs over every dof, so a stale value there would be
    restored as the sleeper's velocity on the following substep. The link Cartesian velocity is zeroed as well: the
    velocity pass skips a sleeping link (see func_forward_velocity_entity), so the value it holds at the transition is
    what every velocity getter reports for as long as the link sleeps, and a restored state recomputes it from the
    zeroed dof velocities.
    """
    dyn_state.links.is_hibernated[i_l, i_b] = True
    dyn_state.links.cd_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
    dyn_state.links.cd_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

    link_I = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    for i_d in range(dyn_info.links.dof_start[link_I], dyn_info.links.dof_end[link_I]):
        dyn_state.dofs.is_hibernated[i_d, i_b] = True
        dyn_state.dofs.vel[i_d, i_b] = 0.0
        dyn_state.dofs.vel_next[i_d, i_b] = 0.0
        dyn_state.dofs.acc[i_d, i_b] = 0.0

    for i_g in range(dyn_info.links.geom_start[link_I], dyn_info.links.geom_end[link_I]):
        dyn_state.geoms.is_hibernated[i_g, i_b] = True


@qd.func
def func_is_entity_in_tree(
    j_e,
    i_b,
    i_l_start,
    i_l_end,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Whether entity j_e belongs to the kinematic tree of the entity holding links [i_l_start, i_l_end).

    An entity roots at its own base link, or, once attached beneath another entity, at a link of that entity. The
    root of its base link therefore tells which entity carries it.
    """
    j_l_start = dyn_info.entities.link_start[j_e]
    J_l_start = [j_l_start, i_b] if qd.static(rigid_config.batch_links_info) else j_l_start
    j_l_root = dyn_info.links.root_idx[J_l_start]
    return i_l_start <= j_l_root and j_l_root < i_l_end


@qd.func
def func_update_cartesian_space_tree(
    i_e,
    i_b,
    qpos: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    """Update the kinematic trees of entity i_e and of every entity attached beneath one of its links.

    An attached entity roots at a link of the entity it hangs from, which may be a jointed link rather than the base,
    and comes after that entity. The forward kinematics run over these entities in order, so the poses a parent passes
    down are final when its children read them; the center of mass of each tree then walks the tree whole (see
    func_COM_links_tree), and the geoms follow.
    """
    i_l_start = dyn_info.entities.link_start[i_e]
    i_l_end = dyn_info.entities.link_end[i_e]
    n_entities = dyn_info.entities.n_links.shape[0]

    for j_e in range(i_e, n_entities):
        if func_is_entity_in_tree(j_e, i_b, i_l_start, i_l_end, dyn_info, rigid_config):
            func_forward_kinematics_entity(j_e, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    for i_l in range(i_l_start, i_l_end):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.root_idx[I_l] == i_l:
            func_COM_links_tree(
                rigid_info.links_tree_idx[i_l], i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward
            )
    for j_e in range(i_e, n_entities):
        if func_is_entity_in_tree(j_e, i_b, i_l_start, i_l_end, dyn_info, rigid_config):
            func_update_geoms_entity(
                j_e, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
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
    """Update the Cartesian space of one environment: forward kinematics, centers of mass, then geoms."""
    BW = qd.static(is_backward)
    i_b = qd.cast(i_b, qd.i32)

    # These loops are considered inner loops
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e_ in (
        range(rigid_info.n_awake_entities[i_b])
        if qd.static(rigid_config.use_hibernation)
        else range(dyn_info.entities.n_links.shape[0])
    ):
        i_e = rigid_info.awake_entities[i_e_, i_b] if qd.static(rigid_config.use_hibernation) else i_e_
        func_forward_kinematics_entity(i_e, i_b, qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_COM_links(i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e_ in (
        range(rigid_info.n_awake_entities[i_b])
        if qd.static(rigid_config.use_hibernation)
        else range(dyn_info.entities.n_links.shape[0])
    ):
        i_e = rigid_info.awake_entities[i_e_, i_b] if qd.static(rigid_config.use_hibernation) else i_e_
        func_update_geoms_entity(
            i_e, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_fixed_geoms, is_backward
        )


@qd.func
def func_update_cartesian_space(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_fixed_geoms: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # This loop must be the outermost loop to be differentiable
    if qd.static(rigid_config.use_hibernation):
        qd.loop_config(name="update_carteisan_space_batch", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(dyn_state.links.pos.shape[1]):
            func_update_cartesian_space_batch(
                i_b,
                rigid_info.qpos,
                dyn_state,
                dyn_info,
                rigid_info,
                rigid_config,
                force_update_fixed_geoms,
                is_backward,
            )
    else:
        # FIXME: Implement parallelization at tree-level (based on root_idx) instead of entity-level
        qd.loop_config(
            name="update_cartesian_space", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        )
        for i_e, i_b in qd.ndrange(dyn_info.entities.n_links.shape[0], dyn_state.links.pos.shape[1]):
            # An entity roots at one of its own links unless it hangs from another entity, whose thread carries it.
            i_l_start = dyn_info.entities.link_start[i_e]
            I_l_start = [i_l_start, i_b] if qd.static(rigid_config.batch_links_info) else i_l_start
            i_l_root = dyn_info.links.root_idx[I_l_start]
            if i_l_start <= i_l_root and i_l_root < dyn_info.entities.link_end[i_e]:
                func_update_cartesian_space_tree(
                    i_e,
                    i_b,
                    rigid_info.qpos,
                    dyn_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                    force_update_fixed_geoms,
                    is_backward,
                )


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
