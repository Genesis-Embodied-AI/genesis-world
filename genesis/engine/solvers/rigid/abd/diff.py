"""
State updates of the rigid body solver outside its dynamics, and the finiteness checks guarding them.

These functions handle:
- Copying the integrated state to the current one, and flagging the envs whose integration is not finite
- Integrating the positions of an entity by given velocities
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.utils.misc import qd_isfinite

from .forward_dynamics import func_clear_coupling_wrench


@qd.func
def func_copy_next_to_curr(
    dyn_state: array_class.DynState, rigid_info: array_class.RigidInfo, rigid_config: qd.template(), errno: qd.Tensor
):
    """Copy the integrated velocities and positions of every env to its state, unless one of them is not finite.

    A 'nan' or an 'inf' raises INVALID_ACC_NAN for its env, which halts the simulation, and leaves the state of that env
    as it was so that the value does not propagate. Every velocity and position takes its own thread.
    """
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_dq, i_b in qd.ndrange(qd.max(dyn_state.dofs.vel.shape[0], rigid_info.qpos.shape[0]), errno.shape[0]):
        is_valid = True
        if i_dq < dyn_state.dofs.vel.shape[0]:
            is_valid = qd_isfinite(dyn_state.dofs.vel_next[i_dq, i_b])
        if i_dq < rigid_info.qpos.shape[0]:
            is_valid = is_valid and qd_isfinite(rigid_info.qpos_next[i_dq, i_b])
        if not is_valid:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_ACC_NAN

    # The sizes are read in the loop ranges, since autodiff takes a top level made of loops alone
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_dq, i_b in qd.ndrange(qd.max(dyn_state.dofs.vel.shape[0], rigid_info.qpos.shape[0]), errno.shape[0]):
        if (errno[i_b] & array_class.ErrorCode.INVALID_ACC_NAN) == 0:
            if i_dq < dyn_state.dofs.vel.shape[0]:
                dyn_state.dofs.vel[i_dq, i_b] = dyn_state.dofs.vel_next[i_dq, i_b]
            if i_dq < rigid_info.qpos.shape[0]:
                rigid_info.qpos[i_dq, i_b] = rigid_info.qpos_next[i_dq, i_b]


@qd.kernel(fastcache=True)
def kernel_copy_next_to_curr_clear_coupling(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """Copy the integrated state to the current one, then clear the coupling wrench (see func_clear_coupling_wrench)."""
    func_copy_next_to_curr(dyn_state, rigid_info, rigid_config, errno)
    func_clear_coupling_wrench(dyn_state, dyn_info, rigid_info, rigid_config)


@qd.func
def func_integrate_dq_entity(
    i_e: int,
    i_b: int,
    dq: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    respect_joint_limit: bool,
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
