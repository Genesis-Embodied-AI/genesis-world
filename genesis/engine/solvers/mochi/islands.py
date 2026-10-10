# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Simulation islands of the MochiSolver: bodies coupled through this step's contact candidates are grouped per
environment, and the direct linear solver factorizes the Newton system island by island."""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .data import (
    COLLIDER_TYPE,
    MochiContactState,
    MochiInfo,
    MochiIslandState,
    MochiSoftInfo,
    MochiSoftState,
    MochiState,
)
from .equalities import MochiEqualitiesInfo


@qd.func
def func_find_root(i_n, i_b, island_state: MochiIslandState):
    root = i_n
    while island_state.nodes_parent[root, i_b] != root:
        parent = island_state.nodes_parent[root, i_b]
        island_state.nodes_parent[root, i_b] = island_state.nodes_parent[parent, i_b]
        root = island_state.nodes_parent[root, i_b]
    return root


@qd.func
def func_union(i_na, i_nb, i_b, island_state: MochiIslandState):
    """Union by minimum index (the root of a component is its smallest node)."""
    if i_na >= 0 and i_nb >= 0:
        root_a = func_find_root(i_na, i_b, island_state)
        root_b = func_find_root(i_nb, i_b, island_state)
        if root_a < root_b:
            island_state.nodes_parent[root_b, i_b] = root_a
        elif root_b < root_a:
            island_state.nodes_parent[root_a, i_b] = root_b


@qd.func
def func_aabbs_overlap(min_a, max_a, min_b, max_b):
    return not ((max_a < min_b).any() or (min_a > max_b).any())


@qd.func
def func_build_islands(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    eq_info: MochiEqualitiesInfo,
    dense_max_dofs: int,
    n_rigid_entities: int,
    rigid_config: qd.template(),
    has_soft: qd.template(),
    has_dense: qd.template(),
    has_equalities: qd.template(),
):
    """Group the bodies of every running environment into islands from this step's contact candidates: the
    (link, collider) and (deformable entity, collider) pairs of the broadphase, and the deformable bodies whose
    conservative bounds overlap those of a deformable collider or of a dynamic link. Then list the degrees of
    freedom island by island and decide whether the environment is solved by the island-wise direct solver."""
    n_nodes = island_state.nodes_parent.shape[0]
    n_dofs = island_state.dofs_node.shape[0]
    n_links = island_state.links_node.shape[0]
    _B = island_state.n_islands.shape[0]
    max_pairs = contact_state.pair_link_a.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_n, i_slot in qd.ndrange(n_nodes, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_nodes, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        island_state.nodes_parent[i_n, i_b] = i_n

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.is_active[i_b]:
            continue
        for i_p in range(qd.min(contact_state.n_pairs[i_b], max_pairs)):
            i_la = contact_state.pair_link_a[i_p, i_b]
            i_lb = contact_state.pair_link_b[i_p, i_b]
            if mochi_info.links.is_dynamic[i_la] and mochi_info.links.is_dynamic[i_lb]:
                func_union(island_state.links_node[i_la], island_state.links_node[i_lb], i_b, island_state)
        if qd.static(has_equalities):
            # Connect and weld constraints couple their two links (joint couplings stay within an articulation).
            for i_eq in range(eq_info.eq_type.shape[0]):
                if eq_info.eq_type[i_eq] != gs.EQUALITY_TYPE.JOINT:
                    i_la = eq_info.eq_obj1id[i_eq]
                    i_lb = eq_info.eq_obj2id[i_eq]
                    if mochi_info.links.is_dynamic[i_la] and mochi_info.links.is_dynamic[i_lb]:
                        func_union(island_state.links_node[i_la], island_state.links_node[i_lb], i_b, island_state)
        if qd.static(has_soft):
            n_soft_entities = soft_state.entities_step_aabb_min.shape[0]
            max_soft_pairs = soft_state.pair_entity_a.shape[0]
            for i_p in range(qd.min(soft_state.n_pairs[i_b], max_soft_pairs)):
                i_lb = soft_state.pair_link_b[i_p, i_b]
                if mochi_info.links.is_dynamic[i_lb]:
                    func_union(
                        n_rigid_entities + soft_state.pair_entity_a[i_p, i_b],
                        island_state.links_node[i_lb],
                        i_b,
                        island_state,
                    )
            # Attachments couple the deformable entity with the link they anchor to.
            for i_a in range(soft_info.att_vert.shape[0]):
                if i_a >= soft_info.n_attachments[None]:
                    continue
                if soft_info.att_link_is_dynamic[i_a] != 0:
                    func_union(
                        n_rigid_entities + soft_info.verts_entity_idx[soft_info.att_vert[i_a]],
                        island_state.links_node[soft_info.att_link[i_a]],
                        i_b,
                        island_state,
                    )
            for i_e in range(n_soft_entities):
                if soft_info.entities_collider_type[i_e] == COLLIDER_TYPE.NONE:
                    continue
                aabb_min = soft_state.entities_step_aabb_min[i_e, i_b]
                aabb_max = soft_state.entities_step_aabb_max[i_e, i_b]
                for j_e in range(n_soft_entities):
                    if j_e == i_e or not soft_info.entities_pair_enabled[j_e, i_e]:
                        continue
                    if soft_info.entities_sample_end[j_e] <= soft_info.entities_sample_start[j_e]:
                        continue
                    if func_aabbs_overlap(
                        aabb_min,
                        aabb_max,
                        soft_state.entities_step_aabb_min[j_e, i_b],
                        soft_state.entities_step_aabb_max[j_e, i_b],
                    ):
                        func_union(n_rigid_entities + i_e, n_rigid_entities + j_e, i_b, island_state)
                for i_l in range(n_links):
                    if not mochi_info.links.is_dynamic[i_l] or not soft_info.entities_links_pair_enabled[i_e, i_l]:
                        continue
                    if mochi_info.links.sample_end[i_l] <= mochi_info.links.sample_start[i_l]:
                        continue
                    if func_aabbs_overlap(
                        aabb_min,
                        aabb_max,
                        contact_state.links_step_aabb_min[i_l, i_b],
                        contact_state.links_step_aabb_max[i_l, i_b],
                    ):
                        func_union(n_rigid_entities + i_e, island_state.links_node[i_l], i_b, island_state)

        # Compact island indices in order of the smallest node of each component.
        n_islands = 0
        for i_n in range(n_nodes):
            root = func_find_root(i_n, i_b, island_state)
            if root == i_n:
                island_state.nodes_island[i_n, i_b] = n_islands
                n_islands += 1
        for i_n in range(n_nodes):
            root = func_find_root(i_n, i_b, island_state)
            island_state.nodes_island[i_n, i_b] = island_state.nodes_island[root, i_b]
        island_state.n_islands[i_b] = n_islands

        # Degrees of freedom grouped by island.
        for i_isl in range(n_nodes):
            island_state.island_n_dofs[i_isl, i_b] = 0
        for i_d in range(n_dofs):
            i_isl = island_state.nodes_island[island_state.dofs_node[i_d], i_b]
            island_state.dofs_island[i_d, i_b] = i_isl
            island_state.island_n_dofs[i_isl, i_b] += 1
        island_state.island_start[0, i_b] = 0
        max_dofs = 0
        for i_isl in range(n_nodes):
            island_state.island_start[i_isl + 1, i_b] = (
                island_state.island_start[i_isl, i_b] + island_state.island_n_dofs[i_isl, i_b]
            )
            max_dofs = qd.max(max_dofs, island_state.island_n_dofs[i_isl, i_b])
        for i_isl in range(n_nodes):
            island_state.island_n_dofs[i_isl, i_b] = 0
        for i_d in range(n_dofs):
            i_isl = island_state.dofs_island[i_d, i_b]
            slot = island_state.island_start[i_isl, i_b] + island_state.island_n_dofs[i_isl, i_b]
            island_state.island_dofs[slot, i_b] = i_d
            island_state.island_n_dofs[i_isl, i_b] += 1
        island_state.island_max_dofs[i_b] = max_dofs
        island_state.uses_dense[i_b] = False
        if qd.static(has_dense):
            island_state.uses_dense[i_b] = max_dofs <= dense_max_dofs


@qd.kernel
def kernel_build_islands(
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    eq_info: MochiEqualitiesInfo,
    dense_max_dofs: int,
    n_rigid_entities: int,
    rigid_config: qd.template(),
    has_soft: qd.template(),
    has_dense: qd.template(),
    has_equalities: qd.template(),
):
    func_build_islands(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        soft_info,
        soft_state,
        island_state,
        eq_info,
        dense_max_dofs,
        n_rigid_entities,
        rigid_config,
        has_soft,
        has_dense,
        has_equalities,
    )


@qd.func
def func_cholesky_solve_islands(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
):
    """In-place Cholesky factorization of the dense matrix of every island of the running environments solved
    directly, followed by the two triangular solves; islands are independent so they run in parallel. The pivot is
    floored relative to the original diagonal so a nearly singular row still factors."""
    n_nodes = island_state.nodes_parent.shape[0]
    _B = island_state.n_islands.shape[0]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_isl, i_slot in qd.ndrange(n_nodes, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_nodes, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.is_active[i_b] or not island_state.uses_dense[i_b]:
            continue
        if i_isl >= island_state.n_islands[i_b]:
            continue
        s0 = island_state.island_start[i_isl, i_b]
        n = island_state.island_n_dofs[i_isl, i_b]
        for i_ in range(n):
            i_d = island_state.island_dofs[s0 + i_, i_b]
            diag = mochi_state.H_dense[i_b, i_d, i_d]
            tmp = diag
            for k_ in range(i_):
                k_d = island_state.island_dofs[s0 + k_, i_b]
                tmp = tmp - mochi_state.H_dense[i_b, i_d, k_d] ** 2
            mochi_state.H_dense[i_b, i_d, i_d] = qd.sqrt(qd.max(tmp, EPS * qd.max(diag, EPS)))
            inv = 1.0 / mochi_state.H_dense[i_b, i_d, i_d]
            for j_ in range(i_ + 1, n):
                j_d = island_state.island_dofs[s0 + j_, i_b]
                dot = gs.qd_float(0.0)
                for k_ in range(i_):
                    k_d = island_state.island_dofs[s0 + k_, i_b]
                    dot = dot + mochi_state.H_dense[i_b, j_d, k_d] * mochi_state.H_dense[i_b, i_d, k_d]
                mochi_state.H_dense[i_b, j_d, i_d] = (mochi_state.H_dense[i_b, j_d, i_d] - dot) * inv
        # L y = res
        for i_ in range(n):
            i_d = island_state.island_dofs[s0 + i_, i_b]
            s = mochi_state.res[i_d, i_b]
            for k_ in range(i_):
                k_d = island_state.island_dofs[s0 + k_, i_b]
                s = s - mochi_state.H_dense[i_b, i_d, k_d] * mochi_state.dx[k_d, i_b]
            mochi_state.dx[i_d, i_b] = s / mochi_state.H_dense[i_b, i_d, i_d]
        # L^T dx = y
        for i__ in range(n):
            i_ = n - 1 - i__
            i_d = island_state.island_dofs[s0 + i_, i_b]
            s = mochi_state.dx[i_d, i_b]
            for k_ in range(i_ + 1, n):
                k_d = island_state.island_dofs[s0 + k_, i_b]
                s = s - mochi_state.H_dense[i_b, k_d, i_d] * mochi_state.dx[k_d, i_b]
            mochi_state.dx[i_d, i_b] = s / mochi_state.H_dense[i_b, i_d, i_d]


@qd.kernel
def kernel_cholesky_solve_islands(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
):
    func_cholesky_solve_islands(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        mochi_info,
        mochi_state,
        island_state,
        rigid_config,
    )
