# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""One mochi step per environment in a single kernel.

The multi-kernel pipeline of `MochiSolver` launches every stage of the step as its own kernel with the host driving
the Newton, line-search and conjugate gradient loops; at one environment each launch costs more than the work it
carries. This kernel runs the whole step for one environment as one serial program (the same per-item functions as the
pipeline, in their per-environment form), environments in parallel: no host round trips, one launch per step. It is
used on the CPU, and on the GPU when the environments are small enough to be worth one thread each.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .articulated import func_assemble_joints, func_project_links_residual, func_update_conv_weights
from .contact import (
    func_broadphase_pairs,
    func_conservative_bounds,
    func_contact_eval,
    func_pairs_to_blocks,
    func_zero_assembly,
)
from .data import (
    LINESEARCH,
    MochiContactState,
    MochiHitReadback,
    MochiInfo,
    MochiIslandState,
    MochiSoftInfo,
    MochiSoftState,
    MochiState,
)
from .equalities import (
    MochiEqualitiesInfo,
    MochiEqualitiesState,
    func_assemble_equalities,
    func_equalities_stage_start,
)
from .integration import func_post_stage, func_step_start, func_store_stage_start_poses
from .islands import func_build_islands, func_cholesky_solve_islands
from .kinematics import func_forward_kinematics_env
from .linear_solver import func_condense_dense, func_pcg_init, func_pcg_iter
from .newton import (
    func_apply_increment,
    func_convergence_check,
    func_linesearch_begin,
    func_linesearch_decide,
    func_reset_newton,
    func_residual_norms,
    func_store_initial_norms,
    func_update_linear_tolerance,
)
from .rigid_assembly import func_assemble_links
from .sleep import func_update_sleep
from .soft import (
    func_assemble_attachments,
    func_attachments_stage_start,
    func_pc_collider_eval,
    func_pc_hash_build,
    func_rod_apply_increment,
    func_rod_assemble,
    func_rod_post_stage,
    func_rod_step_start,
    func_rod_store_ls_ref,
    func_rod_update_conv_weights,
    func_shell_assemble,
    func_shell_stage_start,
    func_soft_apply_increment,
    func_soft_assemble_elements,
    func_soft_broadphase,
    func_soft_collider_eval,
    func_soft_condense_dense,
    func_soft_conservative_bounds,
    func_soft_contact_eval,
    func_soft_dirichlet,
    func_soft_pairs_to_blocks,
    func_soft_post_stage,
    func_soft_step_start,
    func_soft_store_ls_ref,
    func_soft_update_conv_weights,
    func_soft_zero_assembly,
    func_tet_tree_refit,
)


@qd.func
def func_update_kinematics_env(
    i_b,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    with_velocity: qd.template(),
):
    """Link poses, motion subspaces, geom poses and bounds of one environment (joint-space velocities on request)."""
    func_forward_kinematics_env(i_b, dyn_state, dyn_info, rigid_info, rigid_config, with_velocity)
    n_geoms = dyn_state.geoms.pos.shape[0]
    for i_g in range(n_geoms):
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


@qd.func
def func_assemble(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    hit_readback: MochiHitReadback,
    island_state: MochiIslandState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    has_shell: qd.template(),
    has_rod: qd.template(),
    max_samples_per_soft_entity: int,
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
    errno: qd.Tensor,
):
    """Residual and/or Hessian of the environments of the list at their current iterate (contact re-detected)."""
    assem_obj = qd.static(mochi_config.linesearch_type == LINESEARCH.ARMIJO)
    func_zero_assembly(
        i_b_env,
        per_env,
        envs,
        n_envs,
        dyn_state,
        mochi_state,
        contact_state,
        hit_readback,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
        False,
    )
    if qd.static(mochi_config.has_soft):
        func_soft_zero_assembly(
            i_b_env,
            per_env,
            envs,
            n_envs,
            mochi_state,
            soft_state,
            rigid_config,
            assem_dres,
            skip_ls_done,
            False,
        )
    func_contact_eval(
        i_b_env,
        per_env,
        envs,
        n_envs,
        dyn_state,
        dyn_info,
        sdf_info,
        mochi_info,
        mochi_state,
        contact_state,
        hit_readback,
        rigid_config,
        mochi_config,
        assem_dres,
        skip_ls_done,
        False,
        errno,
    )
    if qd.static(mochi_config.has_soft):
        func_soft_contact_eval(
            i_b_env,
            per_env,
            envs,
            n_envs,
            dyn_state,
            dyn_info,
            sdf_info,
            mochi_info,
            mochi_state,
            soft_info,
            soft_state,
            hit_readback,
            rigid_config,
            mochi_config,
            max_samples_per_soft_entity,
            assem_res,
            assem_dres,
            skip_ls_done,
            False,
            errno,
        )
        if qd.static(mochi_config.has_pc_colliders):
            func_pc_hash_build(
                i_b_env,
                per_env,
                envs,
                n_envs,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
                skip_ls_done,
            )
            func_pc_collider_eval(
                i_b_env,
                per_env,
                envs,
                n_envs,
                dyn_state,
                dyn_info,
                mochi_info,
                mochi_state,
                soft_info,
                soft_state,
                hit_readback,
                rigid_config,
                mochi_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
                False,
                errno,
            )
        if qd.static(mochi_config.has_soft_colliders):
            func_tet_tree_refit(
                i_b_env,
                per_env,
                envs,
                n_envs,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
                mochi_config,
                skip_ls_done,
            )
            func_soft_collider_eval(
                i_b_env,
                per_env,
                envs,
                n_envs,
                dyn_state,
                dyn_info,
                mochi_info,
                mochi_state,
                soft_info,
                soft_state,
                hit_readback,
                rigid_config,
                mochi_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
                False,
                errno,
            )
    func_pairs_to_blocks(
        i_b_env,
        per_env,
        envs,
        n_envs,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )
    func_assemble_links(
        i_b_env,
        per_env,
        envs,
        n_envs,
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
    func_assemble_joints(
        i_b_env,
        per_env,
        envs,
        n_envs,
        dyn_state,
        dyn_info,
        rigid_info,
        mochi_info,
        mochi_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )
    if qd.static(mochi_config.has_equalities):
        func_assemble_equalities(
            i_b_env,
            per_env,
            envs,
            n_envs,
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
    if qd.static(mochi_config.has_attachments):
        func_assemble_attachments(
            i_b_env,
            per_env,
            envs,
            n_envs,
            dyn_state,
            mochi_state,
            soft_info,
            soft_state,
            rigid_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
        )
    if qd.static(mochi_config.has_soft):
        func_soft_pairs_to_blocks(
            i_b_env,
            per_env,
            envs,
            n_envs,
            dyn_state,
            mochi_info,
            mochi_state,
            soft_state,
            rigid_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
        )
        if qd.static(mochi_config.has_tets):
            func_soft_assemble_elements(
                i_b_env,
                per_env,
                envs,
                n_envs,
                mochi_info,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
                mochi_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
            )
        if qd.static(has_shell):
            func_shell_assemble(
                i_b_env,
                per_env,
                envs,
                n_envs,
                mochi_info,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
                mochi_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
            )
        if qd.static(has_rod):
            func_rod_assemble(
                i_b_env,
                per_env,
                envs,
                n_envs,
                mochi_info,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
            )
    if qd.static(assem_res):
        if qd.static(mochi_config.has_soft):
            func_soft_dirichlet(
                i_b_env,
                per_env,
                envs,
                n_envs,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
                skip_ls_done,
            )
        func_project_links_residual(
            i_b_env,
            per_env,
            envs,
            n_envs,
            dyn_state,
            dyn_info,
            mochi_info,
            mochi_state,
            rigid_config,
            skip_ls_done,
        )
        func_residual_norms(
            i_b_env,
            per_env,
            envs,
            n_envs,
            mochi_state,
            island_state,
            rigid_config,
            mochi_config,
            skip_ls_done,
        )


@qd.func
def func_linear_solve_env(
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    hit_readback: MochiHitReadback,
    island_state: MochiIslandState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    n_pcg: int,
):
    """Newton step of one environment: island-wise direct solve when the islands fit the dense limit, matrix-free
    conjugate gradient otherwise."""
    is_dense = False
    if qd.static(mochi_config.has_dense):
        is_dense = island_state.uses_dense[i_b]
    if is_dense:
        if qd.static(mochi_config.has_dense):
            func_condense_dense(
                i_b,
                True,
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
                mochi_config.has_equalities,
            )
            if qd.static(mochi_config.has_soft):
                func_soft_condense_dense(
                    i_b,
                    True,
                    mochi_state.all_envs,
                    mochi_state.n_envs_all,
                    dyn_state,
                    dyn_info,
                    mochi_state,
                    soft_info,
                    soft_state,
                    island_state,
                    rigid_config,
                )
            func_cholesky_solve_islands(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                mochi_info,
                mochi_state,
                island_state,
                rigid_config,
            )
    else:
        func_update_linear_tolerance(
            i_b,
            True,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            mochi_info,
            mochi_state,
            rigid_config,
            mochi_config,
        )
        func_pcg_init(
            i_b,
            True,
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
        for _ in range(n_pcg):
            if not mochi_state.pcg_is_active[i_b]:
                break
            func_pcg_iter(
                i_b,
                True,
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
def kernel_step_monolith(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    hit_readback: MochiHitReadback,
    island_state: MochiIslandState,
    eq_info: MochiEqualitiesInfo,
    eq_state: MochiEqualitiesState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    has_shell: qd.template(),
    has_rod: qd.template(),
    n_trials: int,
    dense_max_dofs: int,
    n_rigid_entities: int,
    n_newton: int,
    n_pcg: int,
    max_samples_per_soft_entity: int,
    errno: qd.Tensor,
):
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        # Stage start: history, warm start, kinematics, conservative bounds, candidate pairs and islands.
        func_step_start(
            i_b,
            True,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            dyn_state,
            dyn_info,
            rigid_info,
            mochi_info,
            mochi_state,
            island_state,
            rigid_config,
            mochi_config,
        )
        if qd.static(mochi_config.has_soft):
            func_soft_step_start(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
                mochi_config,
            )
            if qd.static(has_shell):
                func_shell_stage_start(
                    i_b,
                    True,
                    mochi_state.all_envs,
                    mochi_state.n_envs_all,
                    soft_info,
                    soft_state,
                    rigid_config,
                )
            if qd.static(has_rod):
                func_rod_step_start(
                    i_b,
                    True,
                    mochi_state.all_envs,
                    mochi_state.n_envs_all,
                    mochi_state,
                    soft_info,
                    soft_state,
                    rigid_config,
                    mochi_config,
                )
        func_update_kinematics_env(i_b, geoms_init_AABB, dyn_state, dyn_info, rigid_info, rigid_config, False)
        func_store_stage_start_poses(
            i_b,
            True,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            dyn_state,
            mochi_state,
            rigid_config,
        )
        if qd.static(mochi_config.has_equalities):
            func_equalities_stage_start(
                i_b,
                True,
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
        if qd.static(mochi_config.has_attachments):
            func_attachments_stage_start(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
            )
        func_reset_newton(
            i_b,
            True,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            mochi_state,
            rigid_config,
        )
        func_conservative_bounds(
            i_b,
            True,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            dyn_state,
            dyn_info,
            mochi_info,
            mochi_state,
            contact_state,
            rigid_config,
        )
        func_broadphase_pairs(
            i_b,
            True,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            dyn_state,
            dyn_info,
            mochi_info,
            mochi_state,
            contact_state,
            rigid_config,
            errno,
        )
        if qd.static(mochi_config.has_soft):
            func_soft_conservative_bounds(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                mochi_info,
                soft_info,
                soft_state,
                rigid_config,
            )
            func_soft_broadphase(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                dyn_state,
                dyn_info,
                mochi_info,
                mochi_state,
                contact_state,
                soft_info,
                soft_state,
                rigid_config,
                errno,
            )
        func_build_islands(
            i_b,
            True,
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
            mochi_config.has_soft,
            mochi_config.has_dense,
            mochi_config.has_equalities,
            mochi_config.use_sleeping,
        )

        # Newton iterations with the residual-norm line search. Every round evaluates the residual at the current
        # iterate through the single assembly call site below: a round with i_ls == 0 is the full assembly (with the
        # Hessian) at the accepted iterate that opens a Newton iteration, a round with i_ls > 0 is a line-search trial
        # (residual only) after moving to the trial iterate.
        func_update_conv_weights(
            i_b,
            True,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            dyn_state,
            dyn_info,
            mochi_info,
            mochi_state,
            rigid_config,
        )
        if qd.static(mochi_config.has_soft):
            func_soft_update_conv_weights(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                mochi_info,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
            )
            if qd.static(has_rod):
                func_rod_update_conv_weights(
                    i_b,
                    True,
                    mochi_state.all_envs,
                    mochi_state.n_envs_all,
                    mochi_info,
                    mochi_state,
                    soft_info,
                    soft_state,
                    rigid_config,
                )
        i_ls = 0
        is_first = True
        for _ in range(n_newton * (n_trials + 1) + 2):
            if not mochi_state.is_active[i_b]:
                break
            if i_ls > 0:
                func_apply_increment(
                    i_b,
                    True,
                    mochi_state.all_envs,
                    mochi_state.n_envs_all,
                    dyn_info,
                    rigid_info,
                    mochi_info,
                    mochi_state,
                    rigid_config,
                )
                if qd.static(mochi_config.has_soft):
                    func_soft_apply_increment(
                        i_b,
                        True,
                        mochi_state.all_envs,
                        mochi_state.n_envs_all,
                        mochi_state,
                        soft_info,
                        soft_state,
                        rigid_config,
                    )
                    if qd.static(has_rod):
                        func_rod_apply_increment(
                            i_b,
                            True,
                            mochi_state.all_envs,
                            mochi_state.n_envs_all,
                            mochi_state,
                            soft_info,
                            soft_state,
                            rigid_config,
                        )
                func_update_kinematics_env(i_b, geoms_init_AABB, dyn_state, dyn_info, rigid_info, rigid_config, False)
            func_assemble(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                dyn_state,
                dyn_info,
                rigid_info,
                sdf_info,
                mochi_info,
                mochi_state,
                contact_state,
                hit_readback,
                island_state,
                eq_info,
                eq_state,
                soft_info,
                soft_state,
                rigid_config,
                mochi_config,
                has_shell,
                has_rod,
                max_samples_per_soft_entity,
                True,
                i_ls == 0,
                i_ls > 0,
                errno,
            )
            if i_ls == 0:
                if is_first:
                    func_store_initial_norms(
                        i_b,
                        True,
                        mochi_state.all_envs,
                        mochi_state.n_envs_all,
                        rigid_info,
                        mochi_state,
                        island_state,
                        rigid_config,
                    )
                    if qd.static(mochi_config.has_soft):
                        func_soft_store_ls_ref(
                            i_b,
                            True,
                            mochi_state.all_envs,
                            mochi_state.n_envs_all,
                            mochi_state,
                            soft_state,
                            rigid_config,
                            False,
                        )
                        if qd.static(has_rod):
                            func_rod_store_ls_ref(
                                i_b,
                                True,
                                mochi_state.all_envs,
                                mochi_state.n_envs_all,
                                mochi_state,
                                soft_state,
                                rigid_config,
                                False,
                            )
                    is_first = False
                func_convergence_check(
                    i_b,
                    True,
                    mochi_state.all_envs,
                    mochi_state.n_envs_all,
                    mochi_info,
                    mochi_state,
                    island_state,
                    rigid_config,
                    False,
                    errno,
                )
                if mochi_state.is_active[i_b]:
                    func_linear_solve_env(
                        i_b,
                        dyn_state,
                        dyn_info,
                        mochi_info,
                        mochi_state,
                        contact_state,
                        hit_readback,
                        island_state,
                        eq_info,
                        eq_state,
                        soft_info,
                        soft_state,
                        rigid_config,
                        mochi_config,
                        n_pcg,
                    )
                    func_linesearch_begin(
                        i_b,
                        True,
                        mochi_state.all_envs,
                        mochi_state.n_envs_all,
                        rigid_info,
                        mochi_state,
                        rigid_config,
                    )
                    if qd.static(mochi_config.has_soft):
                        func_soft_store_ls_ref(
                            i_b,
                            True,
                            mochi_state.all_envs,
                            mochi_state.n_envs_all,
                            mochi_state,
                            soft_state,
                            rigid_config,
                            False,
                        )
                        if qd.static(has_rod):
                            func_rod_store_ls_ref(
                                i_b,
                                True,
                                mochi_state.all_envs,
                                mochi_state.n_envs_all,
                                mochi_state,
                                soft_state,
                                rigid_config,
                                False,
                            )
                    i_ls = 1
            else:
                func_linesearch_decide(
                    i_b,
                    True,
                    mochi_state.all_envs,
                    mochi_state.n_envs_all,
                    rigid_info,
                    mochi_info,
                    mochi_state,
                    rigid_config,
                    mochi_config,
                    i_ls == n_trials,
                )
                if qd.static(mochi_config.has_soft):
                    func_soft_store_ls_ref(
                        i_b,
                        True,
                        mochi_state.all_envs,
                        mochi_state.n_envs_all,
                        mochi_state,
                        soft_state,
                        rigid_config,
                        True,
                    )
                    if qd.static(has_rod):
                        func_rod_store_ls_ref(
                            i_b,
                            True,
                            mochi_state.all_envs,
                            mochi_state.n_envs_all,
                            mochi_state,
                            soft_state,
                            rigid_config,
                            True,
                        )
                if mochi_state.ls_is_done[i_b]:
                    func_convergence_check(
                        i_b,
                        True,
                        mochi_state.all_envs,
                        mochi_state.n_envs_all,
                        mochi_info,
                        mochi_state,
                        island_state,
                        rigid_config,
                        True,
                        errno,
                    )
                    i_ls = 0
                else:
                    i_ls += 1

        # Stage end: finite-difference velocities, history, final kinematics with the joint-space velocities.
        func_post_stage(
            i_b,
            True,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            dyn_state,
            dyn_info,
            rigid_info,
            mochi_info,
            mochi_state,
            rigid_config,
        )
        if qd.static(mochi_config.has_soft):
            func_soft_post_stage(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
            )
            if qd.static(has_rod):
                func_rod_post_stage(
                    i_b,
                    True,
                    mochi_state.all_envs,
                    mochi_state.n_envs_all,
                    mochi_state,
                    soft_info,
                    soft_state,
                    rigid_config,
                )
        if qd.static(mochi_config.use_sleeping):
            func_update_sleep(
                i_b,
                True,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                n_rigid_entities,
                dyn_state,
                dyn_info,
                mochi_info,
                mochi_state,
                soft_info,
                soft_state,
                island_state,
                rigid_config,
                mochi_config.has_soft,
            )
        func_update_kinematics_env(i_b, geoms_init_AABB, dyn_state, dyn_info, rigid_info, rigid_config, True)
