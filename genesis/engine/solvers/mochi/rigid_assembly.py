# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Per-link terms of the incremental potential of a rigid body: inertia, gravity and damping, assembled at the center
of mass and expressed in the link-frame Lie coordinates (translation of the link origin, world-frame rotation vector)."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .data import MochiInfo, MochiState
from .lie import (
    project_sym_psd3,
    rotation_difference_gradient,
    rotation_difference_hessian,
    rotation_difference_matrix,
    rotation_difference_merit,
    skew,
    vee,
)
from .newton import func_is_env_active


@qd.func
def func_assemble_links(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Add the inertia, gravity and damping of every moving link to the objective and to the link-space residual and
    diagonal Hessian blocks.

    The terms are formed in the center-of-mass frame, where translation and rotation decouple and the rotation merit
    is weighted by the second moment of inertia, then mapped to the link Lie coordinates (translation of the link
    origin and world-frame rotation vector) through the constant offset of the center of mass. Only the rotational
    inertia block is projected onto the positive semi-definite cone; every other term is convex.
    """
    n_links = dyn_state.links.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    EPS = mochi_info.EPS[None]
    I3 = qd.Matrix.identity(gs.qd_float, 3)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_info.links.is_dynamic[i_l] or not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        h = mochi_state.dt_stage[i_b]
        h2i = 1.0 / (h * h)
        mass = mochi_info.links.mass[i_l]

        # Current and stage-start poses of the center-of-mass frame.
        pos = dyn_state.links.pos[i_l, i_b]
        R = gu.qd_quat_to_R(dyn_state.links.quat[i_l, i_b], EPS)
        R_c0 = gu.qd_quat_to_R(dyn_info.links.inertial_quat[I_l], EPS)
        com_local = dyn_info.links.inertial_pos[I_l]
        r_c = R @ com_local
        pos_c = pos + r_c
        R_c = R @ R_c0

        R_start = gu.qd_quat_to_R(mochi_state.links_quat_stage_start[i_l, i_b], EPS)
        r_c_start = R_start @ com_local
        pos_c_start = mochi_state.links_pos_stage_start[i_l, i_b] + r_c_start
        R_c_start = R_start @ R_c0
        vel_start = mochi_state.links_vel_stage_start[i_l, i_b]
        omega_start = mochi_state.links_ang_stage_start[i_l, i_b]
        vel_c_start = vel_start
        S_start = mochi_state.links_vsym_stage_start[i_l, i_b]

        energy = gs.qd_float(0.0)
        g_t = qd.Vector.zero(gs.qd_float, 3)
        g_r = qd.Vector.zero(gs.qd_float, 3)
        H_t = qd.Matrix.zero(gs.qd_float, 3, 3)
        H_r = qd.Matrix.zero(gs.qd_float, 3, 3)

        # Inertia
        dv = (pos_c - pos_c_start) / h - vel_c_start
        energy += 0.5 * mass * dv.norm_sqr()
        g_t += (mass / h) * dv
        H_t += (mass * h2i) * I3
        H_r_inertia = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(mochi_config.use_newton_euler_inertia):
            F = ((R_c - R_c_start) / h) @ R_c.transpose()
            omega = vee(F)
            domega = omega - omega_start
            M = R_c @ mochi_info.links.inertia[i_l] @ R_c.transpose()
            energy += 0.5 * domega.dot(M @ domega)
            g_r += (M @ domega) / h + omega.cross(M @ omega)
            H_r_inertia = M * h2i
        else:
            W = mochi_info.links.second_moment[i_l] * h2i
            # Explicit rotation extrapolated from the stage-start velocity, and the implied rotation two stages back.
            R_tilde = R_c_start + h * ((skew(omega_start) + S_start) @ R_c_start)
            R_old = 2.0 * R_c_start - R_tilde
            M = rotation_difference_matrix(R_c, R_tilde, W)
            energy += -rotation_difference_merit(R_c, R_old, W) + 2.0 * rotation_difference_merit(R_c, R_c_start, W)
            g_r += rotation_difference_gradient(M)
            H_r_inertia = rotation_difference_hessian(M)
        H_r += project_sym_psd3(H_r_inertia, EPS)

        # Gravity
        if mochi_info.links.has_gravity[i_l]:
            mg = mass * mochi_info.gravity[i_b]
            energy -= pos_c.dot(mg)
            g_t -= mg

        # Center of mass -> link origin: d pos_c = d pos - [r_c]x d theta.
        S_c = skew(r_c)
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], energy)
        if qd.static(assem_res):
            g_r_link = g_r + r_c.cross(g_t)
            for k in qd.static(range(3)):
                mochi_state.links_res[i_l, i_b][k] += g_t[k]
                mochi_state.links_res[i_l, i_b][3 + k] += g_r_link[k]
        if assem_dres:
            H_tr = -H_t @ S_c
            H_rr = -S_c @ H_t @ S_c + H_r
            for k, l in qd.static(qd.ndrange(3, 3)):
                mochi_state.H_diag[i_l, i_b][k, l] += H_t[k, l]
                mochi_state.H_diag[i_l, i_b][k, 3 + l] += H_tr[k, l]
                mochi_state.H_diag[i_l, i_b][3 + k, l] += H_tr[l, k]
                mochi_state.H_diag[i_l, i_b][3 + k, 3 + l] += H_rr[k, l]


@qd.kernel
def kernel_assemble_links(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_assemble_links(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        rigid_config,
        mochi_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )
