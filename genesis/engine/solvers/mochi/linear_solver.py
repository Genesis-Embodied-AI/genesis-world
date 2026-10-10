# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Linear solve of the Newton system H dx = res, where H is the projection of the per-link and per-contact-pair 6x6
blocks onto the degrees of freedom through the link Jacobians plus the joint-space diagonal: a dense Cholesky
factorization of the condensed matrix, or a Jacobi-preconditioned conjugate gradient applying the blocks on the fly."""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .articulated import (
    func_jacobian_column_dot,
    func_jacobian_times_dofs,
    func_jacobian_transpose_add,
    func_link_dof_jacobian,
)
from .data import (
    REDUCE_BLOCK,
    REDUCE_CHUNKS,
    REDUCE_LANES,
    REDUCE_TILE,
    MochiContactState,
    MochiInfo,
    MochiSoftInfo,
    MochiSoftState,
    MochiState,
)
from .equalities import MochiEqualitiesInfo, MochiEqualitiesState
from .islands import MochiIslandState
from .rod_solver import func_rod_band_factor
from .soft import func_soft_hit_counts_max, func_soft_matvec, func_soft_precondition

# Divergence factor of the conjugate gradient stopping test, on the norm of the preconditioned residual: an
# environment is dropped once that norm grows by this much, which also catches a non-finite residual. The stopping
# test carries no absolute floor: the preconditioned residual is scaled by the inverse of the Hessian diagonal, so a
# fixed floor would mean a different accuracy for every contact stiffness, and at a stiff default it truncates the
# solve after a couple of iterations.
PCG_DIV_REL_TOL = 1e10


@qd.func
def func_add_projected_block(
    i_la,
    i_lb,
    i_b,
    block,
    H_dense: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    is_symmetric_pair: qd.template(),
):
    """H_dense += J_a^T block J_b over the ancestor degrees of freedom of links a and b (and the transpose block when
    the two links differ)."""
    i_a = i_la
    while i_a != -1:
        I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
        for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
            vel, ang = func_link_dof_jacobian(i_la, i_d, i_b, dyn_state)
            column = qd.Vector([vel[0], vel[1], vel[2], ang[0], ang[1], ang[2]], dt=gs.qd_float)
            row = block.transpose() @ column
            i_c = i_lb
            while i_c != -1:
                I_c = [i_c, i_b] if qd.static(rigid_config.batch_links_info) else i_c
                for j_d in range(dyn_info.links.dof_start[I_c], dyn_info.links.dof_end[I_c]):
                    value = func_jacobian_column_dot(i_lb, j_d, i_b, row, dyn_state)
                    qd.atomic_add(H_dense[i_b, i_d, j_d], value)
                    if qd.static(is_symmetric_pair):
                        qd.atomic_add(H_dense[i_b, j_d, i_d], value)
                i_c = dyn_info.links.parent_idx[I_c]
        i_a = dyn_info.links.parent_idx[I_a]


@qd.func
def func_condense_dense(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    island_state: MochiIslandState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
    has_equalities: qd.template(),
):
    """Assemble the dense Hessian of every environment solved directly from the projected blocks, the joint diagonal
    and the equality constraint couplings."""
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    max_pairs = contact_state.pair_link_a.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_slot, i_d, j_d in (
        qd.ndrange(n_envs[None], n_dofs, n_dofs) if qd.static(not per_env) else qd.ndrange(1, n_dofs, n_dofs)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.is_active[i_b] and island_state.uses_dense[i_b]:
            mochi_state.H_dense[i_b, i_d, j_d] = 0.0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.is_active[i_b] and island_state.uses_dense[i_b]:
            mochi_state.H_dense[i_b, i_d, i_d] = mochi_state.dofs_H_diag[i_d, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]) or not mochi_info.links.is_dynamic[i_l]:
            continue
        func_add_projected_block(
            i_l, i_l, i_b, mochi_state.H_diag[i_l, i_b], mochi_state.H_dense, dyn_state, dyn_info, rigid_config, False
        )

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_slot in qd.ndrange(max_pairs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(max_pairs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]) or i_p >= contact_state.n_pairs[i_b]:
            continue
        if contact_state.n_hits[i_p, i_b] == 0:
            continue
        i_la = contact_state.pair_link_a[i_p, i_b]
        i_lb = contact_state.pair_link_b[i_p, i_b]
        if not (mochi_info.links.is_dynamic[i_la] and mochi_info.links.is_dynamic[i_lb]):
            continue
        func_add_projected_block(
            i_la, i_lb, i_b, mochi_state.H_off[i_p, i_b], mochi_state.H_dense, dyn_state, dyn_info, rigid_config, True
        )

    if qd.static(has_equalities):
        n_eq = eq_info.eq_type.shape[0]
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_eq, i_slot in qd.ndrange(n_eq, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_eq, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if not (mochi_state.is_active[i_b] and island_state.uses_dense[i_b]):
                continue
            if eq_info.eq_type[i_eq] == gs.EQUALITY_TYPE.JOINT:
                h12 = eq_state.joint_h12[i_eq, i_b]
                if h12 != 0.0:
                    i_j1 = eq_info.eq_obj1id[i_eq]
                    i_j2 = eq_info.eq_obj2id[i_eq]
                    I_j1 = [i_j1, i_b] if qd.static(rigid_config.batch_joints_info) else i_j1
                    I_j2 = [i_j2, i_b] if qd.static(rigid_config.batch_joints_info) else i_j2
                    i_d1 = dyn_info.joints.dof_start[I_j1]
                    i_d2 = dyn_info.joints.dof_start[I_j2]
                    mochi_state.H_dense[i_b, i_d1, i_d2] += h12
                    mochi_state.H_dense[i_b, i_d2, i_d1] += h12
            else:
                i_la = eq_info.eq_obj1id[i_eq]
                i_lb = eq_info.eq_obj2id[i_eq]
                if mochi_info.links.is_dynamic[i_la] and mochi_info.links.is_dynamic[i_lb]:
                    func_add_projected_block(
                        i_la,
                        i_lb,
                        i_b,
                        eq_state.H_off[i_eq, i_b],
                        mochi_state.H_dense,
                        dyn_state,
                        dyn_info,
                        rigid_config,
                        True,
                    )


@qd.kernel
def kernel_condense_dense(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    island_state: MochiIslandState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
    has_equalities: qd.template(),
):
    func_condense_dense(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        island_state,
        eq_info,
        eq_state,
        rigid_config,
        has_equalities,
    )


@qd.func
def func_matvec(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    src: qd.Tensor,
    dst: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    update_p: qd.template(),
):
    """dst = H src for the running conjugate gradient environments, applying the projected blocks on the fly.

    With `update_p` the conjugate-gradient direction update p = z + beta p is fused into the initialization pass
    (src must be pcg_p): the previous iteration computed z and beta, and updating the direction here saves the
    standalone pass and skips it entirely in converged environments."""
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    max_pairs = contact_state.pair_link_a.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.pcg_is_active[i_b]:
            x = src[i_d, i_b]
            if qd.static(update_p):
                x = mochi_state.pcg_z[i_d, i_b] + mochi_state.pcg_beta[i_b] * x
                src[i_d, i_b] = x
            dst[i_d, i_b] = mochi_state.dofs_H_diag[i_d, i_b] * x

    # One segmented pass covers the link blocks, the contact pair couplings and the equality couplings: they are
    # independent atomic accumulations into dst, and each offloaded loop costs a fixed launch on the device.
    n_eq = eq_info.eq_type.shape[0] if qd.static(mochi_config.has_equalities) else 0
    n_rigid_items = n_links + max_pairs + n_eq
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_x, i_slot in (
        qd.ndrange(n_rigid_items, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_rigid_items, 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.pcg_is_active[i_b]:
            continue
        if i_x < n_links:
            i_l = i_x
            if mochi_info.links.is_dynamic[i_l]:
                v = func_jacobian_times_dofs(i_l, i_b, src, dyn_state, dyn_info, rigid_config)
                func_jacobian_transpose_add(
                    i_l, i_b, mochi_state.H_diag[i_l, i_b] @ v, dst, dyn_state, dyn_info, rigid_config
                )
        elif i_x < n_links + max_pairs:
            i_p = i_x - n_links
            if i_p < contact_state.n_pairs[i_b] and contact_state.n_hits[i_p, i_b] != 0:
                i_la = contact_state.pair_link_a[i_p, i_b]
                i_lb = contact_state.pair_link_b[i_p, i_b]
                if mochi_info.links.is_dynamic[i_la] and mochi_info.links.is_dynamic[i_lb]:
                    H_off = mochi_state.H_off[i_p, i_b]
                    v_a = func_jacobian_times_dofs(i_la, i_b, src, dyn_state, dyn_info, rigid_config)
                    v_b = func_jacobian_times_dofs(i_lb, i_b, src, dyn_state, dyn_info, rigid_config)
                    func_jacobian_transpose_add(i_la, i_b, H_off @ v_b, dst, dyn_state, dyn_info, rigid_config)
                    func_jacobian_transpose_add(
                        i_lb, i_b, H_off.transpose() @ v_a, dst, dyn_state, dyn_info, rigid_config
                    )
        elif qd.static(mochi_config.has_equalities):
            i_eq = i_x - n_links - max_pairs
            if eq_info.eq_type[i_eq] == gs.EQUALITY_TYPE.JOINT:
                h12 = eq_state.joint_h12[i_eq, i_b]
                if h12 != 0.0:
                    i_j1 = eq_info.eq_obj1id[i_eq]
                    i_j2 = eq_info.eq_obj2id[i_eq]
                    I_j1 = [i_j1, i_b] if qd.static(rigid_config.batch_joints_info) else i_j1
                    I_j2 = [i_j2, i_b] if qd.static(rigid_config.batch_joints_info) else i_j2
                    i_d1 = dyn_info.joints.dof_start[I_j1]
                    i_d2 = dyn_info.joints.dof_start[I_j2]
                    qd.atomic_add(dst[i_d1, i_b], h12 * src[i_d2, i_b])
                    qd.atomic_add(dst[i_d2, i_b], h12 * src[i_d1, i_b])
            else:
                i_la = eq_info.eq_obj1id[i_eq]
                i_lb = eq_info.eq_obj2id[i_eq]
                if mochi_info.links.is_dynamic[i_la] and mochi_info.links.is_dynamic[i_lb]:
                    H_off = eq_state.H_off[i_eq, i_b]
                    v_a = func_jacobian_times_dofs(i_la, i_b, src, dyn_state, dyn_info, rigid_config)
                    v_b = func_jacobian_times_dofs(i_lb, i_b, src, dyn_state, dyn_info, rigid_config)
                    func_jacobian_transpose_add(i_la, i_b, H_off @ v_b, dst, dyn_state, dyn_info, rigid_config)
                    func_jacobian_transpose_add(
                        i_lb, i_b, H_off.transpose() @ v_a, dst, dyn_state, dyn_info, rigid_config
                    )

    if qd.static(mochi_config.has_soft):
        func_soft_matvec(
            i_b_env,
            per_env,
            envs,
            n_envs,
            src,
            dst,
            dyn_state,
            dyn_info,
            mochi_info,
            mochi_state,
            soft_info,
            soft_state,
            rigid_config,
        )


@qd.func
def func_apply_preconditioner(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    r: qd.Tensor,
    z: qd.Tensor,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """z = M^-1 r: Jacobi on the rigid degrees of freedom, block Jacobi (3x3 per vertex) on the deformable ones."""
    n_dofs = mochi_state.res.shape[0]
    _B = mochi_state.is_active.shape[0]
    EPS = mochi_info.EPS[None]
    if qd.static(mochi_config.has_soft):
        # The deformable preconditioner covers the rigid dofs in its segmented pass as well.
        func_soft_precondition(
            i_b_env, per_env, envs, n_envs, r, z, mochi_state, soft_info, soft_state, rigid_config, EPS
        )
    else:
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if mochi_state.pcg_is_active[i_b]:
                z[i_d, i_b] = r[i_d, i_b] / qd.max(mochi_state.pcg_diag[i_d, i_b], EPS)


@qd.func
def func_pcg_init(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """Start the conjugate gradient from dx = 0 with the preconditioner built from the projected diagonal, for the
    running environments not solved directly."""
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    _B = mochi_state.is_active.shape[0]

    if qd.static(mochi_config.has_soft):
        func_soft_hit_counts_max(i_b_env, per_env, envs, n_envs, soft_state, rigid_config)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.pcg_is_active[i_b] = mochi_state.is_active[i_b] and not island_state.uses_dense[i_b]
        mochi_state.pcg_rTz[i_b] = 0.0
        mochi_state.pcg_zTz[i_b] = 0.0
    if qd.static(mochi_config.has_soft and mochi_config.has_rod_band):
        func_rod_band_factor(
            i_b_env,
            per_env,
            envs,
            n_envs,
            mochi_state,
            soft_info,
            soft_state,
            rigid_config,
            mochi_info.EPS[None],
        )
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.pcg_is_active[i_b]:
            mochi_state.dx[i_d, i_b] = 0.0
            mochi_state.pcg_r[i_d, i_b] = mochi_state.res[i_d, i_b]
            mochi_state.pcg_diag[i_d, i_b] = mochi_state.dofs_H_diag[i_d, i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.pcg_is_active[i_b] or not mochi_info.links.is_dynamic[i_l]:
            continue
        H = mochi_state.H_diag[i_l, i_b]
        i_a = i_l
        while i_a != -1:
            I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
            for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
                vel, ang = func_link_dof_jacobian(i_l, i_d, i_b, dyn_state)
                column = qd.Vector([vel[0], vel[1], vel[2], ang[0], ang[1], ang[2]], dt=gs.qd_float)
                qd.atomic_add(mochi_state.pcg_diag[i_d, i_b], column.dot(H @ column))
            i_a = dyn_info.links.parent_idx[I_a]
    func_apply_preconditioner(
        i_b_env,
        per_env,
        envs,
        n_envs,
        mochi_state.pcg_r,
        mochi_state.pcg_z,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        mochi_config,
    )
    qd.loop_config(
        # A per-environment scalar accumulation contends on one cache line per environment: keep it
        # serial on the CPU backend unless environments themselves are the parallel axis.
        serialize=qd.static(
            rigid_config.para_level < gs.PARA_LEVEL.PARTIAL
            or (rigid_config.backend == gs.cpu and rigid_config.para_level < gs.PARA_LEVEL.ALL)
        )
    )
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.pcg_is_active[i_b]:
            r = mochi_state.pcg_r[i_d, i_b]
            z = mochi_state.pcg_z[i_d, i_b]
            mochi_state.pcg_p[i_d, i_b] = z
            qd.atomic_add(mochi_state.pcg_rTz[i_b], r * z)
            qd.atomic_add(mochi_state.pcg_zTz[i_b], z * z)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        mochi_state.pcg_zTz0[i_b] = mochi_state.pcg_zTz[i_b]
        mochi_state.pcg_beta[i_b] = 0.0
        mochi_state.pcg_pTAp[i_b] = 0.0
        # An environment whose right-hand side vanishes takes no iteration at all; the negated comparison also drops
        # a non-finite norm.
        if not (mochi_state.pcg_zTz[i_b] > 0.0):
            mochi_state.pcg_is_active[i_b] = False


@qd.kernel
def kernel_pcg_init(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    func_pcg_init(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        island_state,
        rigid_config,
        mochi_config,
    )


@qd.func
def func_pcg_iter(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    n_dofs = mochi_state.res.shape[0]
    _B = mochi_state.is_active.shape[0]
    abs_tol = mochi_info.pcg_abs_tol[None]

    func_matvec(
        i_b_env,
        per_env,
        envs,
        n_envs,
        mochi_state.pcg_p,
        mochi_state.pcg_Ap,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        soft_info,
        soft_state,
        eq_info,
        eq_state,
        rigid_config,
        mochi_config,
        True,
    )

    n_env_tiles = (n_envs[None] + REDUCE_LANES - 1) // REDUCE_LANES
    n_dof_tiles = (n_dofs + REDUCE_TILE - 1) // REDUCE_TILE
    if qd.static(not per_env and rigid_config.para_level >= gs.PARA_LEVEL.PARTIAL and rigid_config.backend != gs.cpu):
        qd.loop_config(block_dim=REDUCE_BLOCK)
        for i_flat in range(n_env_tiles * n_dof_tiles * REDUCE_BLOCK):
            tid = i_flat % REDUCE_BLOCK
            i_block = i_flat // REDUCE_BLOCK
            lane = tid % REDUCE_LANES
            chunk = tid // REDUCE_LANES
            i_slot = REDUCE_LANES * (i_block % n_env_tiles) + lane
            d0 = REDUCE_TILE * (i_block // n_env_tiles)
            sh = qd.simt.block.SharedArray((REDUCE_BLOCK,), gs.qd_float)
            acc = gs.qd_float(0.0)
            i_b = 0
            is_live = i_slot < n_envs[None]
            if is_live:
                i_b = envs[i_slot]
                is_live = mochi_state.pcg_is_active[i_b] != 0
            if is_live:
                for k in range(REDUCE_TILE // REDUCE_CHUNKS):
                    i_d = d0 + REDUCE_CHUNKS * k + chunk
                    if i_d < n_dofs:
                        acc += mochi_state.pcg_p[i_d, i_b] * mochi_state.pcg_Ap[i_d, i_b]
            sh[tid] = acc
            qd.simt.block.sync()
            if chunk == 0 and is_live:
                total = gs.qd_float(0.0)
                for c in qd.static(range(REDUCE_CHUNKS)):
                    total += sh[lane + REDUCE_LANES * c]
                qd.atomic_add(mochi_state.pcg_pTAp[i_b], total)
    else:
        qd.loop_config(
            # A per-environment scalar accumulation contends on one cache line per environment: keep it
            # serial on the CPU backend unless environments themselves are the parallel axis.
            serialize=qd.static(
                rigid_config.para_level < gs.PARA_LEVEL.PARTIAL
                or (rigid_config.backend == gs.cpu and rigid_config.para_level < gs.PARA_LEVEL.ALL)
            )
        )
        for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if mochi_state.pcg_is_active[i_b]:
                qd.atomic_add(mochi_state.pcg_pTAp[i_b], mochi_state.pcg_p[i_d, i_b] * mochi_state.pcg_Ap[i_d, i_b])

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.pcg_is_active[i_b]:
            mochi_state.n_pcg_iter[i_b] += 1
            if mochi_state.pcg_pTAp[i_b] <= 0.0:
                mochi_state.pcg_is_active[i_b] = False
            mochi_state.pcg_rTz_new[i_b] = 0.0
            mochi_state.pcg_rTz_cross[i_b] = 0.0
            mochi_state.pcg_zTz[i_b] = 0.0
    if qd.static(not per_env and rigid_config.para_level >= gs.PARA_LEVEL.PARTIAL and rigid_config.backend != gs.cpu):
        qd.loop_config(block_dim=REDUCE_BLOCK)
        for i_flat in range(n_env_tiles * n_dof_tiles * REDUCE_BLOCK):
            tid = i_flat % REDUCE_BLOCK
            i_block = i_flat // REDUCE_BLOCK
            lane = tid % REDUCE_LANES
            chunk = tid // REDUCE_LANES
            i_slot = REDUCE_LANES * (i_block % n_env_tiles) + lane
            d0 = REDUCE_TILE * (i_block // n_env_tiles)
            sh = qd.simt.block.SharedArray((REDUCE_BLOCK,), gs.qd_float)
            acc = gs.qd_float(0.0)
            i_b = 0
            is_live = i_slot < n_envs[None]
            if is_live:
                i_b = envs[i_slot]
                is_live = mochi_state.pcg_is_active[i_b] != 0
            if is_live:
                alpha = mochi_state.pcg_rTz[i_b] / mochi_state.pcg_pTAp[i_b]
                for k in range(REDUCE_TILE // REDUCE_CHUNKS):
                    i_d = d0 + REDUCE_CHUNKS * k + chunk
                    if i_d < n_dofs:
                        mochi_state.dx[i_d, i_b] += alpha * mochi_state.pcg_p[i_d, i_b]
                        r = mochi_state.pcg_r[i_d, i_b] - alpha * mochi_state.pcg_Ap[i_d, i_b]
                        mochi_state.pcg_r[i_d, i_b] = r
                        # The preconditioned residual of the previous iteration is still in pcg_z; its product with
                        # the new residual tells the Polak-Ribiere direction from the Fletcher-Reeves one.
                        acc += r * mochi_state.pcg_z[i_d, i_b]
            sh[tid] = acc
            qd.simt.block.sync()
            if chunk == 0 and is_live:
                total = gs.qd_float(0.0)
                for c in qd.static(range(REDUCE_CHUNKS)):
                    total += sh[lane + REDUCE_LANES * c]
                qd.atomic_add(mochi_state.pcg_rTz_cross[i_b], total)
    else:
        qd.loop_config(
            # A per-environment scalar accumulation contends on one cache line per environment: keep it
            # serial on the CPU backend unless environments themselves are the parallel axis.
            serialize=qd.static(
                rigid_config.para_level < gs.PARA_LEVEL.PARTIAL
                or (rigid_config.backend == gs.cpu and rigid_config.para_level < gs.PARA_LEVEL.ALL)
            )
        )
        for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if mochi_state.pcg_is_active[i_b]:
                alpha = mochi_state.pcg_rTz[i_b] / mochi_state.pcg_pTAp[i_b]
                mochi_state.dx[i_d, i_b] += alpha * mochi_state.pcg_p[i_d, i_b]
                r = mochi_state.pcg_r[i_d, i_b] - alpha * mochi_state.pcg_Ap[i_d, i_b]
                mochi_state.pcg_r[i_d, i_b] = r
                # The preconditioned residual of the previous iteration is still in pcg_z; its product with the new
                # residual is the term that tells the Polak-Ribiere direction from the Fletcher-Reeves one.
                qd.atomic_add(mochi_state.pcg_rTz_cross[i_b], r * mochi_state.pcg_z[i_d, i_b])
    func_apply_preconditioner(
        i_b_env,
        per_env,
        envs,
        n_envs,
        mochi_state.pcg_r,
        mochi_state.pcg_z,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        rigid_config,
        mochi_config,
    )
    if qd.static(not per_env and rigid_config.para_level >= gs.PARA_LEVEL.PARTIAL and rigid_config.backend != gs.cpu):
        qd.loop_config(block_dim=REDUCE_BLOCK)
        for i_flat in range(n_env_tiles * n_dof_tiles * REDUCE_BLOCK):
            tid = i_flat % REDUCE_BLOCK
            i_block = i_flat // REDUCE_BLOCK
            lane = tid % REDUCE_LANES
            chunk = tid // REDUCE_LANES
            i_slot = REDUCE_LANES * (i_block % n_env_tiles) + lane
            d0 = REDUCE_TILE * (i_block // n_env_tiles)
            sh_rz = qd.simt.block.SharedArray((REDUCE_BLOCK,), gs.qd_float)
            sh_zz = qd.simt.block.SharedArray((REDUCE_BLOCK,), gs.qd_float)
            acc_rz = gs.qd_float(0.0)
            acc_zz = gs.qd_float(0.0)
            i_b = 0
            is_live = i_slot < n_envs[None]
            if is_live:
                i_b = envs[i_slot]
                is_live = mochi_state.pcg_is_active[i_b] != 0
            if is_live:
                for k in range(REDUCE_TILE // REDUCE_CHUNKS):
                    i_d = d0 + REDUCE_CHUNKS * k + chunk
                    if i_d < n_dofs:
                        z = mochi_state.pcg_z[i_d, i_b]
                        acc_rz += mochi_state.pcg_r[i_d, i_b] * z
                        acc_zz += z * z
            sh_rz[tid] = acc_rz
            sh_zz[tid] = acc_zz
            qd.simt.block.sync()
            if chunk == 0 and is_live:
                total_rz = gs.qd_float(0.0)
                total_zz = gs.qd_float(0.0)
                for c in qd.static(range(REDUCE_CHUNKS)):
                    total_rz += sh_rz[lane + REDUCE_LANES * c]
                    total_zz += sh_zz[lane + REDUCE_LANES * c]
                qd.atomic_add(mochi_state.pcg_rTz_new[i_b], total_rz)
                qd.atomic_add(mochi_state.pcg_zTz[i_b], total_zz)
    else:
        qd.loop_config(
            # A per-environment scalar accumulation contends on one cache line per environment: keep it
            # serial on the CPU backend unless environments themselves are the parallel axis.
            serialize=qd.static(
                rigid_config.para_level < gs.PARA_LEVEL.PARTIAL
                or (rigid_config.backend == gs.cpu and rigid_config.para_level < gs.PARA_LEVEL.ALL)
            )
        )
        for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if mochi_state.pcg_is_active[i_b]:
                z = mochi_state.pcg_z[i_d, i_b]
                qd.atomic_add(mochi_state.pcg_rTz_new[i_b], mochi_state.pcg_r[i_d, i_b] * z)
                qd.atomic_add(mochi_state.pcg_zTz[i_b], z * z)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if mochi_state.pcg_is_active[i_b]:
            rel_tol = mochi_state.pcg_rel_tol[i_b]
            zTz0 = mochi_state.pcg_zTz0[i_b]
            zTz = mochi_state.pcg_zTz[i_b]
            is_converged = not (zTz > zTz0 * rel_tol * rel_tol) or zTz <= abs_tol * abs_tol
            if is_converged or zTz > (zTz0 * PCG_DIV_REL_TOL) * PCG_DIV_REL_TOL:
                mochi_state.pcg_is_active[i_b] = False
            beta = (mochi_state.pcg_rTz_new[i_b] - mochi_state.pcg_rTz_cross[i_b]) / mochi_state.pcg_rTz[i_b]
            mochi_state.pcg_rTz[i_b] = mochi_state.pcg_rTz_new[i_b]
            mochi_state.pcg_beta[i_b] = beta
            mochi_state.pcg_pTAp[i_b] = 0.0


@qd.kernel
def kernel_pcg_iter(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    func_pcg_iter(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        soft_info,
        soft_state,
        eq_info,
        eq_state,
        rigid_config,
        mochi_config,
    )


@qd.kernel
def kernel_pcg_any_active(mochi_state: MochiState) -> qd.i32:
    n_active = 0
    for i_b in range(mochi_state.pcg_is_active.shape[0]):
        if mochi_state.pcg_is_active[i_b]:
            n_active += 1
    return n_active
