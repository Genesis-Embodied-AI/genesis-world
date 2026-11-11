"""
State updates of the rigid body solver outside its dynamics, and the finiteness checks guarding them.

These functions handle:
- Copying the integrated state to the current one, and flagging the envs whose integration is not finite
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.data import DynInfo, DynState, RigidErrorCode, RigidInfo
from genesis.utils.data import eoa
from genesis.utils.misc import qd_isfinite

from .forward_dynamics import func_clear_external_wrench


@qd.func
def func_copy_next_to_curr(dyn_state: DynState, rigid_info: RigidInfo, rigid_config: qd.template(), errno: qd.Tensor):
    """Copy the integrated velocities and positions of every env to its state, unless one of them is not finite.

    A 'nan' or an 'inf' raises INVALID_ACC_NAN for its env, which halts the simulation, and leaves the state of that env
    as it was so that the value does not propagate. Every velocity and position takes its own thread.
    """
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_dq, i_b in qd.ndrange(
        qd.max(dyn_state.dofs.vel.shape[0], rigid_info.qpos.shape[0]),
        errno.shape[0],
        axes=qd.static(eoa(dyn_state.dofs.vel_next)),
    ):
        is_valid = True
        if i_dq < dyn_state.dofs.vel.shape[0]:
            is_valid = qd_isfinite(dyn_state.dofs.vel_next[i_dq, i_b])
        if i_dq < rigid_info.qpos.shape[0]:
            is_valid = is_valid and qd_isfinite(rigid_info.qpos_next[i_dq, i_b])
        if not is_valid:
            errno[i_b] = errno[i_b] | RigidErrorCode.INVALID_ACC_NAN

    # The sizes are read in the loop ranges, since autodiff takes a top level made of loops alone
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_dq, i_b in qd.ndrange(
        qd.max(dyn_state.dofs.vel.shape[0], rigid_info.qpos.shape[0]),
        errno.shape[0],
        axes=qd.static(eoa(dyn_state.dofs.vel)),
    ):
        if (errno[i_b] & RigidErrorCode.INVALID_ACC_NAN) == 0:
            if i_dq < dyn_state.dofs.vel.shape[0]:
                dyn_state.dofs.vel[i_dq, i_b] = dyn_state.dofs.vel_next[i_dq, i_b]
            if i_dq < rigid_info.qpos.shape[0]:
                rigid_info.qpos[i_dq, i_b] = rigid_info.qpos_next[i_dq, i_b]


@qd.kernel(fastcache=True)
def kernel_copy_next_to_curr_clear_coupling(
    dyn_state: DynState, dyn_info: DynInfo, rigid_info: RigidInfo, rigid_config: qd.template(), errno: qd.Tensor
):
    """Copy the integrated state to the current one, then clear the external wrench (see func_clear_external_wrench)."""
    func_copy_next_to_curr(dyn_state, rigid_info, rigid_config, errno)
    func_clear_external_wrench(dyn_state, dyn_info, rigid_info, rigid_config)
