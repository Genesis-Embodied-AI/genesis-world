# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Equality constraints of the articulations in the MochiSolver (connect, weld and joint couplings, i.e. loop
closures) as soft penalties of the incremental potential: E = 1/2 k |c|^2 + 1/2 (d / h) |c - c_stage_start|^2 on the
constraint violation c, with Gauss-Newton Hessian blocks on the two constrained links or joints."""

import dataclasses

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class
from genesis.utils.array_class import V_MAT, V

from .data import MochiInfo, MochiState
from .lie import skew
from .newton import func_is_env_active


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiEqualitiesInfo:
    # Constraint type (gs.EQUALITY_TYPE), the two constrained links (connect, weld) or joints (joint), and the data
    # of the constraint (anchors, relative pose and torque scale, or polynomial coefficients).
    eq_type: qd.Tensor
    eq_obj1id: qd.Tensor
    eq_obj2id: qd.Tensor
    eq_data: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiEqualitiesState:
    # Constraint violation at the stage start (6 rows at most), the 6x6 block coupling the two links of a connect or
    # weld constraint and the scalar coupling the two joints of a joint constraint.
    c_stage_start: qd.Tensor
    H_off: qd.Tensor
    joint_h12: qd.Tensor


def get_mochi_equalities_info(solver, equalities):
    n_eq_ = max(1, len(equalities))
    info = MochiEqualitiesInfo(
        eq_type=V(dtype=gs.qd_int, shape=(n_eq_,)),
        eq_obj1id=V(dtype=gs.qd_int, shape=(n_eq_,)),
        eq_obj2id=V(dtype=gs.qd_int, shape=(n_eq_,)),
        eq_data=V(dtype=gs.qd_vec11, shape=(n_eq_,)),
    )
    if equalities:
        info.eq_type.from_numpy(np.array([int(eq.type) for eq in equalities], dtype=gs.np_int))
        info.eq_obj1id.from_numpy(np.array([eq.eq_obj1id for eq in equalities], dtype=gs.np_int))
        info.eq_obj2id.from_numpy(np.array([eq.eq_obj2id for eq in equalities], dtype=gs.np_int))
        data = np.zeros((len(equalities), 11), dtype=gs.np_float)
        for i_eq, eq in enumerate(equalities):
            values = np.asarray(eq.eq_data, dtype=gs.np_float).reshape((-1,))
            data[i_eq, : len(values)] = values
        info.eq_data.from_numpy(data)
    return info


def get_mochi_equalities_state(solver, n_equalities):
    _B = solver._B
    n_eq_ = max(1, n_equalities)
    return MochiEqualitiesState(
        c_stage_start=V(dtype=gs.qd_vec6, shape=(n_eq_, _B)),
        H_off=V_MAT(n=6, m=6, dtype=gs.qd_float, shape=(n_eq_, _B)),
        joint_h12=V(dtype=gs.qd_float, shape=(n_eq_, _B)),
    )


@qd.func
def func_equality_anchors(i_eq, i_b, links_pos: qd.Tensor, links_quat: qd.Tensor, eq_info: MochiEqualitiesInfo):
    """World lever arms (from the link origins) and anchor positions of a connect or weld constraint. Connect stores
    (anchor1, anchor2), weld stores (anchor2, anchor1)."""
    data = eq_info.eq_data[i_eq]
    i_la = eq_info.eq_obj1id[i_eq]
    i_lb = eq_info.eq_obj2id[i_eq]
    local_a = qd.Vector([data[0], data[1], data[2]], dt=gs.qd_float)
    local_b = qd.Vector([data[3], data[4], data[5]], dt=gs.qd_float)
    if eq_info.eq_type[i_eq] == gs.EQUALITY_TYPE.WELD:
        local_a, local_b = local_b, local_a
    rho_a = gu.qd_transform_by_quat(local_a, links_quat[i_la, i_b])
    rho_b = gu.qd_transform_by_quat(local_b, links_quat[i_lb, i_b])
    return rho_a, rho_b, links_pos[i_la, i_b] + rho_a, links_pos[i_lb, i_b] + rho_b


@qd.func
def func_weld_rotation_error(i_eq, i_b, links_quat: qd.Tensor, eq_info: MochiEqualitiesInfo, EPS):
    """World-frame rotation vector taking the orientation of link b to the target orientation q_a * relpose."""
    data = eq_info.eq_data[i_eq]
    i_la = eq_info.eq_obj1id[i_eq]
    i_lb = eq_info.eq_obj2id[i_eq]
    relpose = qd.Vector([data[6], data[7], data[8], data[9]], dt=gs.qd_float)
    q_target = gu.qd_transform_quat_by_quat(relpose, links_quat[i_la, i_b])
    q_err = gu.qd_transform_quat_by_quat(gu.qd_inv_quat(links_quat[i_lb, i_b]), q_target)
    return gu.qd_quat_to_rotvec(q_err, EPS)


@qd.func
def func_joint_coupling(
    i_eq,
    i_b,
    qpos: qd.Tensor,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    eq_info: MochiEqualitiesInfo,
    rigid_config: qd.template(),
):
    """Violation c = (q1 - q1_0) - poly(q2 - q2_0) of a joint coupling, its derivative with respect to q2 and the two
    degrees of freedom (the second is -1 when the coupling has a single joint)."""
    data = eq_info.eq_data[i_eq]
    i_j1 = eq_info.eq_obj1id[i_eq]
    i_j2 = eq_info.eq_obj2id[i_eq]
    I_j1 = [i_j1, i_b] if qd.static(rigid_config.batch_joints_info) else i_j1
    i_q1 = dyn_info.joints.q_start[I_j1]
    i_d1 = dyn_info.joints.dof_start[I_j1]
    c = qpos[i_q1, i_b] - rigid_info.qpos0[i_q1, i_b] - data[0]
    deriv = gs.qd_float(0.0)
    i_d2 = -1
    if i_j2 >= 0:
        I_j2 = [i_j2, i_b] if qd.static(rigid_config.batch_joints_info) else i_j2
        i_q2 = dyn_info.joints.q_start[I_j2]
        i_d2 = dyn_info.joints.dof_start[I_j2]
        x = qpos[i_q2, i_b] - rigid_info.qpos0[i_q2, i_b]
        power = gs.qd_float(1.0)
        for i in qd.static(range(1, 5)):
            c -= data[i] * power * x
            deriv -= data[i] * i * power
            power *= x
    return c, deriv, i_d1, i_d2


@qd.func
def func_equalities_stage_start(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
):
    """Constraint violations at the stage start, the reference of the penalty damping."""
    n_eq = eq_info.eq_type.shape[0]
    _B = mochi_state.is_active.shape[0]
    EPS = mochi_info.EPS[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_eq, i_slot in qd.ndrange(n_eq, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_eq, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        c = qd.Vector.zero(gs.qd_float, 6)
        eq_type = eq_info.eq_type[i_eq]
        if eq_type == gs.EQUALITY_TYPE.JOINT:
            c0, _deriv, _i_d1, _i_d2 = func_joint_coupling(
                i_eq, i_b, mochi_state.qpos_stage_start, dyn_info, rigid_info, eq_info, rigid_config
            )
            c[0] = c0
        else:
            _rho_a, _rho_b, p_a, p_b = func_equality_anchors(
                i_eq, i_b, mochi_state.links_pos_stage_start, mochi_state.links_quat_stage_start, eq_info
            )
            dp = p_a - p_b
            for k in qd.static(range(3)):
                c[k] = dp[k]
            if eq_type == gs.EQUALITY_TYPE.WELD:
                e = func_weld_rotation_error(i_eq, i_b, mochi_state.links_quat_stage_start, eq_info, EPS)
                for k in qd.static(range(3)):
                    c[3 + k] = e[k]
        eq_state.c_stage_start[i_eq, i_b] = c


@qd.kernel
def kernel_equalities_stage_start(
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
):
    func_equalities_stage_start(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_info,
        rigid_info,
        mochi_info,
        mochi_state,
        eq_info,
        eq_state,
        rigid_config,
    )


@qd.func
def func_assemble_equalities(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Penalty of every equality constraint: residual on the links (connect, weld: through the point Jacobians
    J = [I, -[rho]x] of the anchors and the identity on the rotation error) or on the joint coordinates (joint
    coupling), Gauss-Newton Hessian blocks on the links and the coupling block between them."""
    n_eq = eq_info.eq_type.shape[0]
    _B = mochi_state.is_active.shape[0]
    EPS = mochi_info.EPS[None]
    k = mochi_info.equality_stiffness[None]
    d = mochi_info.equality_damping[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_eq, i_slot in qd.ndrange(n_eq, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_eq, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        h = mochi_state.dt_stage[i_b]
        kappa = d / h
        K = k + kappa
        c_start = eq_state.c_stage_start[i_eq, i_b]
        eq_type = eq_info.eq_type[i_eq]

        if eq_type == gs.EQUALITY_TYPE.JOINT:
            c, deriv, i_d1, i_d2 = func_joint_coupling(
                i_eq, i_b, rigid_info.qpos, dyn_info, rigid_info, eq_info, rigid_config
            )
            dc = c - c_start[0]
            g = k * c + kappa * dc
            if qd.static(assem_obj):
                qd.atomic_add(mochi_state.obj[i_b], 0.5 * k * c * c + 0.5 * kappa * dc * dc)
            if qd.static(assem_res):
                qd.atomic_add(mochi_state.res[i_d1, i_b], g)
                if i_d2 >= 0:
                    qd.atomic_add(mochi_state.res[i_d2, i_b], g * deriv)
            if assem_dres:
                qd.atomic_add(mochi_state.dofs_H_diag[i_d1, i_b], K)
                eq_state.joint_h12[i_eq, i_b] = 0.0
                if i_d2 >= 0:
                    qd.atomic_add(mochi_state.dofs_H_diag[i_d2, i_b], K * deriv * deriv)
                    eq_state.joint_h12[i_eq, i_b] = K * deriv
            continue

        i_la = eq_info.eq_obj1id[i_eq]
        i_lb = eq_info.eq_obj2id[i_eq]
        is_dynamic_a = mochi_info.links.is_dynamic[i_la]
        is_dynamic_b = mochi_info.links.is_dynamic[i_lb]
        rho_a, rho_b, p_a, p_b = func_equality_anchors(i_eq, i_b, dyn_state.links.pos, dyn_state.links.quat, eq_info)
        c_t = p_a - p_b
        c_t_start = qd.Vector([c_start[0], c_start[1], c_start[2]], dt=gs.qd_float)
        g_t = k * c_t + kappa * (c_t - c_t_start)
        energy = 0.5 * k * c_t.dot(c_t) + 0.5 * kappa * (c_t - c_t_start).dot(c_t - c_t_start)
        g_r = qd.Vector.zero(gs.qd_float, 3)
        K_r = gs.qd_float(0.0)
        if eq_type == gs.EQUALITY_TYPE.WELD:
            torque_scale = eq_info.eq_data[i_eq][10]
            e = func_weld_rotation_error(i_eq, i_b, dyn_state.links.quat, eq_info, EPS)
            e_start = qd.Vector([c_start[3], c_start[4], c_start[5]], dt=gs.qd_float)
            g_r = torque_scale * (k * e + kappa * (e - e_start))
            K_r = torque_scale * K
            energy += torque_scale * (0.5 * k * e.dot(e) + 0.5 * kappa * (e - e_start).dot(e - e_start))
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], energy)
        if qd.static(assem_res):
            torque_a = rho_a.cross(g_t) + g_r
            torque_b = rho_b.cross(g_t) + g_r
            if is_dynamic_a:
                for kk in qd.static(range(3)):
                    qd.atomic_add(mochi_state.links_res[i_la, i_b][kk], g_t[kk])
                    qd.atomic_add(mochi_state.links_res[i_la, i_b][3 + kk], torque_a[kk])
            if is_dynamic_b:
                for kk in qd.static(range(3)):
                    qd.atomic_add(mochi_state.links_res[i_lb, i_b][kk], -g_t[kk])
                    qd.atomic_add(mochi_state.links_res[i_lb, i_b][3 + kk], -torque_b[kk])
        if assem_dres:
            S_a = skew(rho_a)
            S_b = skew(rho_b)
            I3 = qd.Matrix.identity(gs.qd_float, 3)
            if is_dynamic_a:
                SS = -K * (S_a @ S_a) + K_r * I3
                for kk, ll in qd.static(qd.ndrange(3, 3)):
                    qd.atomic_add(mochi_state.H_diag[i_la, i_b][kk, ll], K * I3[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_la, i_b][kk, 3 + ll], -K * S_a[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_la, i_b][3 + kk, ll], K * S_a[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_la, i_b][3 + kk, 3 + ll], SS[kk, ll])
            if is_dynamic_b:
                SS = -K * (S_b @ S_b) + K_r * I3
                for kk, ll in qd.static(qd.ndrange(3, 3)):
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][kk, ll], K * I3[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][kk, 3 + ll], -K * S_b[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][3 + kk, ll], K * S_b[kk, ll])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][3 + kk, 3 + ll], SS[kk, ll])
            H_off = qd.Matrix.zero(gs.qd_float, 6, 6)
            if is_dynamic_a and is_dynamic_b:
                # -J_a^T K J_b with J_a = [I, -S_a], J_b = [I, -S_b], minus the rotation coupling of a weld.
                SaSb = S_a @ S_b
                for kk, ll in qd.static(qd.ndrange(3, 3)):
                    H_off[kk, ll] = -K * I3[kk, ll]
                    H_off[kk, 3 + ll] = K * S_b[kk, ll]
                    H_off[3 + kk, ll] = -K * S_a[kk, ll]
                    H_off[3 + kk, 3 + ll] = K * SaSb[kk, ll] - K_r * I3[kk, ll]
            eq_state.H_off[i_eq, i_b] = H_off


@qd.kernel
def kernel_assemble_equalities(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_assemble_equalities(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        rigid_info,
        mochi_info,
        mochi_state,
        eq_info,
        eq_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )
