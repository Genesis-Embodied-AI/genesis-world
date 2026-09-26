"""
Backward pass functions for the rigid body solver.

This module contains functions used during the backward pass (gradient computation)
of the rigid body simulation. These functions handle:
- Copying state between next and current time steps
- Saving and loading adjoint cache for gradient computation
- Preparing and beginning backward substeps
- Gradient validity checking
- Cartesian space copying for adjoint computation
- Acceleration copying and dq integration

These functions are extracted from the main rigid_solver module to improve
code organization and maintainability.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from .forward_kinematics import func_update_cartesian_space, func_forward_velocity


@qd.func
def func_copy_next_to_curr(
    dyn_state: array_class.DynState, rigid_info: array_class.RigidInfo, rigid_config: qd.template(), errno: qd.Tensor
):
    n_qs = rigid_info.qpos.shape[0]
    n_dofs = dyn_state.dofs.vel.shape[0]
    _B = dyn_state.dofs.vel.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # Prevent nan propagation
        is_valid = True
        for i_d in range(n_dofs):
            e = dyn_state.dofs.vel_next[i_d, i_b]
            is_valid &= not qd.math.isnan(e)
        for i_q in range(n_qs):
            e = rigid_info.qpos_next[i_q, i_b]
            is_valid &= not qd.math.isnan(e)

        if is_valid:
            for i_d in range(n_dofs):
                dyn_state.dofs.vel[i_d, i_b] = dyn_state.dofs.vel_next[i_d, i_b]

            for i_q in range(n_qs):
                rigid_info.qpos[i_q, i_b] = rigid_info.qpos_next[i_q, i_b]
        else:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_ACC_NAN


@qd.func
def func_copy_next_to_curr_grad(
    f: qd.int32,
    dyn_state: array_class.DynState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = dyn_state.dofs.vel.shape[0]
    n_qs = rigid_info.qpos.shape[0]
    _B = dyn_state.dofs.vel.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dyn_state.dofs.vel_next.grad[i_d, i_b] = dyn_state.dofs.vel.grad[i_d, i_b]
        dyn_state.dofs.vel.grad[i_d, i_b] = 0.0
        dyn_state.dofs.vel[i_d, i_b] = rigid_adjoint_cache.dofs_vel[f, i_d, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in qd.ndrange(n_qs, _B):
        rigid_info.qpos_next.grad[i_q, i_b] = rigid_info.qpos.grad[i_q, i_b]
        rigid_info.qpos.grad[i_q, i_b] = 0.0
        rigid_info.qpos[i_q, i_b] = rigid_adjoint_cache.qpos[f, i_q, i_b]


@qd.kernel(fastcache=True)
def kernel_save_adjoint_cache(
    f: qd.int32,
    dyn_state: array_class.DynState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    func_save_adjoint_cache(f, dyn_state, rigid_adjoint_cache, rigid_info, rigid_config)


@qd.func
def func_save_adjoint_cache(
    f: qd.int32,
    dyn_state: array_class.DynState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = dyn_state.dofs.vel.shape[0]
    n_qs = rigid_info.qpos.shape[0]
    _B = dyn_state.dofs.vel.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        rigid_adjoint_cache.dofs_vel[f, i_d, i_b] = dyn_state.dofs.vel[i_d, i_b]
        rigid_adjoint_cache.dofs_acc[f, i_d, i_b] = dyn_state.dofs.acc[i_d, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in qd.ndrange(n_qs, _B):
        rigid_adjoint_cache.qpos[f, i_q, i_b] = rigid_info.qpos[i_q, i_b]


@qd.func
def func_load_adjoint_cache(
    f: qd.int32,
    dyn_state: array_class.DynState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = dyn_state.dofs.vel.shape[0]
    n_qs = rigid_info.qpos.shape[0]
    _B = dyn_state.dofs.vel.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dyn_state.dofs.vel[i_d, i_b] = rigid_adjoint_cache.dofs_vel[f, i_d, i_b]
        dyn_state.dofs.acc[i_d, i_b] = rigid_adjoint_cache.dofs_acc[f, i_d, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in qd.ndrange(n_qs, _B):
        rigid_info.qpos[i_q, i_b] = rigid_adjoint_cache.qpos[f, i_q, i_b]


@qd.kernel(fastcache=True)
def kernel_prepare_backward_substep(
    f: qd.int32,
    dyn_state: array_class.DynState,
    dyn_state_adjoint_cache: array_class.DynState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    # Load the current state from adjoint cache
    func_load_adjoint_cache(f, dyn_state, rigid_adjoint_cache, rigid_info, rigid_config)

    # If mujoco compatibility is disabled, update the cartesian space and save the results to adjoint cache. This is
    # because the cartesian space is overwritten later by other kernels if mujoco compatibility was disabled.
    if qd.static(not rigid_config.enable_mujoco_compatibility):
        func_update_cartesian_space(
            dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms=False, is_backward=True
        )
        func_forward_velocity(dyn_state, dyn_info, rigid_info, rigid_config, is_backward=True)

        # FIXME: Parameter pruning for ndarray is buggy for now and requires match variable and arg names.
        # Save results of [update_cartesian_space] to adjoint cache
        func_copy_cartesian_space(dyn_state, dyn_state_adjoint_cache, rigid_config)


@qd.kernel(fastcache=True)
def kernel_begin_backward_substep(
    f: qd.int32,
    dyn_state: array_class.DynState,
    dyn_state_adjoint_cache: array_class.DynState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
) -> qd.i32:
    is_grad_valid = func_is_grad_valid(dyn_state, rigid_info, rigid_config)
    if is_grad_valid:
        func_copy_next_to_curr_grad(f, dyn_state, rigid_adjoint_cache, rigid_info, rigid_config)

        if not rigid_config.enable_mujoco_compatibility:
            # Restore the cartesian space that was overwritten by the post-integrate forward replay in the backward
            # substep (see _update_cartesian_grad in rigid_solver.py).
            func_copy_cartesian_space(dyn_state_adjoint_cache, dyn_state, rigid_config)

    return is_grad_valid


@qd.func
def func_is_grad_valid(dyn_state: array_class.DynState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()):
    is_valid = True
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for I in qd.grouped(qd.ndrange(*rigid_info.qpos.shape)):
        if qd.math.isnan(rigid_info.qpos.grad[I]):
            is_valid = False

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for I in qd.grouped(qd.ndrange(*dyn_state.dofs.vel.shape)):
        if qd.math.isnan(dyn_state.dofs.vel.grad[I]):
            is_valid = False

    return is_valid


@qd.func
def func_copy_cartesian_space(
    dyn_state: array_class.DynState, dyn_state_adjoint_cache: array_class.DynState, rigid_config: qd.template()
):
    # Copy outputs of [kernel_update_cartesian_space] among [dofs, links, joints, geoms] states. This is used to restore
    # the outputs that were overwritten if we disabled mujoco compatibility for backward pass.

    # dofs state
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for I in qd.grouped(qd.ndrange(*dyn_state.dofs.pos.shape)):
        # pos, cdof_ang, cdof_vel, cdofd_ang, cdofd_vel
        dyn_state_adjoint_cache.dofs.pos[I] = dyn_state.dofs.pos[I]
        dyn_state_adjoint_cache.dofs.cdof_ang[I] = dyn_state.dofs.cdof_ang[I]
        dyn_state_adjoint_cache.dofs.cdof_vel[I] = dyn_state.dofs.cdof_vel[I]
        dyn_state_adjoint_cache.dofs.cdofd_ang[I] = dyn_state.dofs.cdofd_ang[I]
        dyn_state_adjoint_cache.dofs.cdofd_vel[I] = dyn_state.dofs.cdofd_vel[I]

    # links state
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for I in qd.grouped(qd.ndrange(*dyn_state.links.pos.shape)):
        # pos, quat, root_COM, mass_sum, i_pos, i_quat, cinr_inertial, cinr_pos, cinr_quat, cinr_mass, cd_vel, cd_ang
        dyn_state_adjoint_cache.links.pos[I] = dyn_state.links.pos[I]
        dyn_state_adjoint_cache.links.quat[I] = dyn_state.links.quat[I]
        dyn_state_adjoint_cache.links.root_COM[I] = dyn_state.links.root_COM[I]
        dyn_state_adjoint_cache.links.mass_sum[I] = dyn_state.links.mass_sum[I]
        dyn_state_adjoint_cache.links.i_pos[I] = dyn_state.links.i_pos[I]
        dyn_state_adjoint_cache.links.i_quat[I] = dyn_state.links.i_quat[I]
        dyn_state_adjoint_cache.links.cinr_inertial[I] = dyn_state.links.cinr_inertial[I]
        dyn_state_adjoint_cache.links.cinr_pos[I] = dyn_state.links.cinr_pos[I]
        dyn_state_adjoint_cache.links.cinr_quat[I] = dyn_state.links.cinr_quat[I]
        dyn_state_adjoint_cache.links.cinr_mass[I] = dyn_state.links.cinr_mass[I]
        dyn_state_adjoint_cache.links.cd_vel[I] = dyn_state.links.cd_vel[I]
        dyn_state_adjoint_cache.links.cd_ang[I] = dyn_state.links.cd_ang[I]

    # joints state
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for I in qd.grouped(qd.ndrange(*dyn_state.joints.xanchor.shape)):
        # xanchor, xaxis
        dyn_state_adjoint_cache.joints.xanchor[I] = dyn_state.joints.xanchor[I]
        dyn_state_adjoint_cache.joints.xaxis[I] = dyn_state.joints.xaxis[I]

    # geoms state
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for I in qd.grouped(qd.ndrange(*dyn_state.geoms.pos.shape)):
        # pos, quat, verts_updated
        dyn_state_adjoint_cache.geoms.pos[I] = dyn_state.geoms.pos[I]
        dyn_state_adjoint_cache.geoms.quat[I] = dyn_state.geoms.quat[I]
        dyn_state_adjoint_cache.geoms.verts_updated[I] = dyn_state.geoms.verts_updated[I]


@qd.kernel(fastcache=True)
def kernel_copy_acc(
    f: qd.int32,
    dyn_state: array_class.DynState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    rigid_config: qd.template(),
):
    n_dofs = dyn_state.dofs.vel.shape[0]
    _B = dyn_state.dofs.vel.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dyn_state.dofs.acc[i_d, i_b] = rigid_adjoint_cache.dofs_acc[f, i_d, i_b]


@qd.kernel(fastcache=True)
def kernel_copy_next_to_curr_no_check(
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Unguarded copy of the _next slots to current, used in the backward substep right before the forward replay so
    the backward kernels see the post-integrate qpos / vel."""
    n_qs = rigid_info.qpos.shape[0]
    n_dofs = dyn_state.dofs.vel.shape[0]
    _B = dyn_state.dofs.vel.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in qd.ndrange(n_qs, _B):
        rigid_info.qpos[i_q, i_b] = rigid_info.qpos_next[i_q, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dyn_state.dofs.vel[i_d, i_b] = dyn_state.dofs.vel_next[i_d, i_b]


@qd.func
def func_integrate_dq_entity(
    i_e,
    i_b,
    dq,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    respect_joint_limit,
):
    EPS = rigid_info.EPS[None]

    for i_l in range(dyn_info.entities.link_start[i_e], dyn_info.entities.link_end[i_e]):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.n_dofs[I_l] == 0:
            continue

        i_j = dyn_info.links.joint_start[I_l]
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]

        q_start = dyn_info.links.q_start[I_l]
        dof_start = dyn_info.links.dof_start[I_l]
        dq_start = dyn_info.links.dof_start[I_l] - dyn_info.entities.dof_start[i_e]

        if joint_type == gs.JOINT_TYPE.FREE:
            pos = qd.Vector(
                [rigid_info.qpos[q_start, i_b], rigid_info.qpos[q_start + 1, i_b], rigid_info.qpos[q_start + 2, i_b]]
            )
            dpos = qd.Vector([dq[dq_start, i_b], dq[dq_start + 1, i_b], dq[dq_start + 2, i_b]])
            pos = pos + dpos

            quat = qd.Vector(
                [
                    rigid_info.qpos[q_start + 3, i_b],
                    rigid_info.qpos[q_start + 4, i_b],
                    rigid_info.qpos[q_start + 5, i_b],
                    rigid_info.qpos[q_start + 6, i_b],
                ]
            )
            dquat = gu.qd_rotvec_to_quat(
                qd.Vector([dq[dq_start + 3, i_b], dq[dq_start + 4, i_b], dq[dq_start + 5, i_b]], dt=gs.qd_float), EPS
            )
            quat = gu.qd_transform_quat_by_quat(
                quat, dquat
            )  # Note that this order is different from integrateing vel. Here dq is w.r.t to world.

            for j in qd.static(range(3)):
                rigid_info.qpos[q_start + j, i_b] = pos[j]

            for j in qd.static(range(4)):
                rigid_info.qpos[q_start + j + 3, i_b] = quat[j]

        elif joint_type == gs.JOINT_TYPE.FIXED:
            pass

        else:
            for i_d_ in range(dyn_info.links.n_dofs[I_l]):
                rigid_info.qpos[q_start + i_d_, i_b] = rigid_info.qpos[q_start + i_d_, i_b] + dq[dq_start + i_d_, i_b]

                if respect_joint_limit:
                    I_d = [dof_start + i_d_, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start + i_d_
                    rigid_info.qpos[q_start + i_d_, i_b] = qd.math.clamp(
                        rigid_info.qpos[q_start + i_d_, i_b], dyn_info.dofs.limit[I_d][0], dyn_info.dofs.limit[I_d][1]
                    )
