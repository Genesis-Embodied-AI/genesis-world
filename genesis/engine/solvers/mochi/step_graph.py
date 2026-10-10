"""One graph-launched kernel per step: the Newton, line-search and conjugate-gradient loops run as device-side graph
loops (`qd.graph.do_while`) whose bodies are the batched step functions (parallel over items and environments), so a
step costs one Python launch and, on GPUs without device-side graph conditionals, one flag readback per loop round
instead of one launch per stage.

Round structure (do-while semantics: a body runs before its counter is tested, so every loop is primed with one useful
round and a body that has nothing to do runs on empty environment lists):

    prologue: stage start, kinematics, bounds, candidate pairs, islands, convergence weights
    while newton_counter:                      # n_newton + 1 bodies at most
        while round_counter:                   # trials L1..Ln of the previous iteration, then the S round
            [L] trial iterate: increments, kinematics
            [S] convergence check of the iterate accepted by the line search (increments the iteration count)
            ONE assembly call site: residual (+ Hessian in the S round)
            [S] initial norms (first round), convergence check, linear tolerance, dense arm, PCG init
            [L] line-search decision
        while pcg_counter:                     # k conjugate-gradient iterations per round
        [S] line-search begin
    epilogue: post stage, kinematics with velocities

The environment lists are the identity list gated to 0 or every environment per phase; the functions predicate on the
per-environment flags as in the pipeline.
"""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .articulated import func_update_conv_weights
from .contact import func_broadphase_pairs, func_conservative_bounds
from .data import (
    MochiContactState,
    MochiHitReadback,
    MochiInfo,
    MochiIslandState,
    MochiSoftInfo,
    MochiSoftState,
    MochiState,
)
from .equalities import MochiEqualitiesInfo, MochiEqualitiesState, func_equalities_stage_start
from .integration import func_post_stage, func_step_start, func_store_stage_start_poses
from .islands import func_build_islands, func_cholesky_solve_islands
from .kinematics import func_update_kinematics
from .linear_solver import func_condense_dense, func_pcg_init, func_pcg_iter
from .newton import (
    func_apply_increment,
    func_convergence_check,
    func_linesearch_begin,
    func_linesearch_decide,
    func_reset_newton,
    func_store_initial_norms,
    func_update_linear_tolerance,
)
from .soft import (
    func_attachments_stage_start,
    func_rod_apply_increment,
    func_rod_post_stage,
    func_rod_step_start,
    func_rod_store_ls_ref,
    func_rod_update_conv_weights,
    func_shell_stage_start,
    func_soft_apply_increment,
    func_soft_broadphase,
    func_soft_condense_dense,
    func_soft_conservative_bounds,
    func_soft_post_stage,
    func_soft_step_start,
    func_soft_store_ls_ref,
    func_soft_update_conv_weights,
)
from .step_monolith import func_assemble


@qd.func
def func_graph_prime(
    newton_counter: qd.types.ndarray(),
    round_counter: qd.types.ndarray(),
    mochi_state: MochiState,
    n_newton,
):
    """Counters of a step: the Newton loop gets one extra body for the trials of the last iteration, the round loop
    starts with the single S round of iteration 0."""
    for _ in range(1):
        newton_counter[None] = n_newton + 1
        round_counter[None] = 1
        mochi_state.graph_is_first[None] = 1


@qd.func
def func_round_begin(round_counter: qd.types.ndarray(), mochi_state: MochiState):
    """Kind of the round (round_counter 1 = S, 2 = last trial) and the gates of its phases."""
    _B = mochi_state.is_active.shape[0]
    for _ in range(1):
        is_s = round_counter[None] == 1
        mochi_state.graph_round_is_s[None] = 1 if is_s else 0
        mochi_state.graph_round_is_l[None] = 0 if is_s else 1
        mochi_state.graph_round_is_last_trial[None] = 1 if round_counter[None] == 2 else 0
        mochi_state.gate_ls[None] = 0 if is_s else _B
        mochi_state.gate_newton[None] = _B if is_s else 0
        mochi_state.gate_first[None] = _B if (is_s and mochi_state.graph_is_first[None] == 1) else 0
        mochi_state.gate_post_ls[None] = _B if (is_s and mochi_state.graph_is_first[None] == 0) else 0


@qd.func
def func_any_in_linesearch(mochi_state: MochiState, rigid_config: qd.template()):
    """Whether some environment still runs its line search (graph_any)."""
    _B = mochi_state.is_active.shape[0]
    for _ in range(1):
        mochi_state.graph_any[None] = 0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        if mochi_state.is_active[i_b] and not mochi_state.ls_is_done[i_b]:
            qd.atomic_max(mochi_state.graph_any[None], 1)


@qd.func
def func_round_end(
    round_counter: qd.types.ndarray(), pcg_counter: qd.types.ndarray(), mochi_state: MochiState, n_pcg_rounds
):
    """After an S round: leave the round loop and prime the conjugate-gradient loop; after a trial: next trial, or
    straight to the S round when every line search is done."""
    for _ in range(1):
        if mochi_state.graph_round_is_s[None] == 1:
            round_counter[None] = 0
            pcg_counter[None] = n_pcg_rounds
            mochi_state.graph_is_first[None] = 0
        elif mochi_state.graph_any[None] == 0:
            round_counter[None] = 1
        else:
            round_counter[None] = qd.max(round_counter[None] - 1, 1)


@qd.func
def func_any_pcg_active(mochi_state: MochiState, rigid_config: qd.template()):
    _B = mochi_state.is_active.shape[0]
    for _ in range(1):
        mochi_state.graph_any[None] = 0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        if mochi_state.pcg_is_active[i_b]:
            qd.atomic_max(mochi_state.graph_any[None], 1)
    for _ in range(1):
        mochi_state.gate_pcg[None] = _B if mochi_state.graph_any[None] == 1 else 0


@qd.func
def func_pcg_cap(
    pcg_counter: qd.types.ndarray(), mochi_state: MochiState, rigid_config: qd.template(), n_pcg_rounds, k, j, n_pcg
):
    """Deactivate the environments that reached the iteration budget inside an unrolled round."""
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        i_iter = (n_pcg_rounds - pcg_counter[None]) * k + j
        if i_iter >= n_pcg:
            mochi_state.pcg_is_active[i_b] = False


@qd.func
def func_pcg_round_end(pcg_counter: qd.types.ndarray(), mochi_state: MochiState):
    for _ in range(1):
        if mochi_state.graph_any[None] == 1:
            pcg_counter[None] = qd.max(pcg_counter[None] - 1, 0)
        else:
            pcg_counter[None] = 0


@qd.func
def func_newton_end(
    newton_counter: qd.types.ndarray(),
    round_counter: qd.types.ndarray(),
    mochi_state: MochiState,
    rigid_config: qd.template(),
    n_trials,
):
    """Another Newton body (its trials, then its S round) while some environment is active."""
    _B = mochi_state.is_active.shape[0]
    for _ in range(1):
        mochi_state.graph_any[None] = 0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        if mochi_state.is_active[i_b]:
            qd.atomic_max(mochi_state.graph_any[None], 1)
    for _ in range(1):
        if mochi_state.graph_any[None] == 1:
            newton_counter[None] = qd.max(newton_counter[None] - 1, 0)
            round_counter[None] = n_trials + 1
        else:
            newton_counter[None] = 0


@qd.kernel(graph=True)
def kernel_step_graph(
    newton_counter: qd.types.ndarray(),
    round_counter: qd.types.ndarray(),
    pcg_counter: qd.types.ndarray(),
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
    pcg_unroll: qd.template(),
    n_trials: int,
    dense_max_dofs: int,
    n_rigid_entities: int,
    n_newton: int,
    n_pcg: int,
    n_pcg_rounds: int,
    max_samples_per_soft_entity: int,
    errno: qd.Tensor,
):
    # ---- prologue: stage start, kinematics, conservative bounds, candidate pairs and islands (every environment) ----
    func_step_start(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        rigid_info,
        mochi_info,
        mochi_state,
        rigid_config,
        mochi_config,
    )
    if qd.static(mochi_config.has_soft):
        func_soft_step_start(
            0,
            False,
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
                0, False, mochi_state.all_envs, mochi_state.n_envs_all, soft_info, soft_state, rigid_config
            )
        if qd.static(has_rod):
            func_rod_step_start(
                0,
                False,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
                mochi_config,
            )
    func_update_kinematics(
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        geoms_init_AABB,
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
        False,
    )
    func_store_stage_start_poses(
        0, False, mochi_state.all_envs, mochi_state.n_envs_all, dyn_state, mochi_state, rigid_config
    )
    if qd.static(mochi_config.has_equalities):
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
    if qd.static(mochi_config.has_attachments):
        func_attachments_stage_start(
            0,
            False,
            mochi_state.all_envs,
            mochi_state.n_envs_all,
            mochi_state,
            soft_info,
            soft_state,
            rigid_config,
        )
    func_reset_newton(0, False, mochi_state.all_envs, mochi_state.n_envs_all, mochi_state, rigid_config)
    func_conservative_bounds(
        0,
        False,
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
        0,
        False,
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
            0, False, mochi_state.all_envs, mochi_state.n_envs_all, mochi_info, soft_info, soft_state, rigid_config
        )
        func_soft_broadphase(
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
            rigid_config,
            errno,
        )
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
        mochi_config.has_soft,
        mochi_config.has_dense,
        mochi_config.has_equalities,
    )
    func_update_conv_weights(
        0,
        False,
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
            0,
            False,
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
                0,
                False,
                mochi_state.all_envs,
                mochi_state.n_envs_all,
                mochi_info,
                mochi_state,
                soft_info,
                soft_state,
                rigid_config,
            )
    func_graph_prime(newton_counter, round_counter, mochi_state, n_newton)

    while qd.graph.do_while(newton_counter):
        while qd.graph.do_while(round_counter):
            func_round_begin(round_counter, mochi_state)
            # [L] the trial iterate
            func_apply_increment(
                0,
                False,
                mochi_state.all_envs,
                mochi_state.gate_ls,
                dyn_info,
                rigid_info,
                mochi_info,
                mochi_state,
                rigid_config,
            )
            if qd.static(mochi_config.has_soft):
                func_soft_apply_increment(
                    0,
                    False,
                    mochi_state.all_envs,
                    mochi_state.gate_ls,
                    mochi_state,
                    soft_info,
                    soft_state,
                    rigid_config,
                )
                if qd.static(has_rod):
                    func_rod_apply_increment(
                        0,
                        False,
                        mochi_state.all_envs,
                        mochi_state.gate_ls,
                        mochi_state,
                        soft_info,
                        soft_state,
                        rigid_config,
                    )
            func_update_kinematics(
                mochi_state.all_envs,
                mochi_state.gate_ls,
                geoms_init_AABB,
                dyn_state,
                dyn_info,
                rigid_info,
                rigid_config,
                False,
            )
            # [S] the iterate accepted by the line search of the previous iteration
            func_convergence_check(
                0,
                False,
                mochi_state.all_envs,
                mochi_state.gate_post_ls,
                mochi_info,
                mochi_state,
                island_state,
                rigid_config,
                True,
                errno,
            )
            # the single assembly call site: residual at the current iterate, Hessian in the S round
            func_assemble(
                0,
                False,
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
                mochi_state.graph_round_is_s[None] == 1,
                mochi_state.graph_round_is_l[None] == 1,
                errno,
            )
            # [S] Newton start
            func_store_initial_norms(
                0,
                False,
                mochi_state.all_envs,
                mochi_state.gate_first,
                rigid_info,
                mochi_state,
                island_state,
                rigid_config,
            )
            if qd.static(mochi_config.has_soft):
                func_soft_store_ls_ref(
                    0, False, mochi_state.all_envs, mochi_state.gate_first, mochi_state, soft_state, rigid_config, False
                )
                if qd.static(has_rod):
                    func_rod_store_ls_ref(
                        0,
                        False,
                        mochi_state.all_envs,
                        mochi_state.gate_first,
                        mochi_state,
                        soft_state,
                        rigid_config,
                        False,
                    )
            func_convergence_check(
                0,
                False,
                mochi_state.all_envs,
                mochi_state.gate_newton,
                mochi_info,
                mochi_state,
                island_state,
                rigid_config,
                False,
                errno,
            )
            if qd.static(mochi_config.has_dense):
                func_condense_dense(
                    0,
                    False,
                    mochi_state.all_envs,
                    mochi_state.gate_newton,
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
                        0,
                        False,
                        mochi_state.all_envs,
                        mochi_state.gate_newton,
                        dyn_state,
                        dyn_info,
                        mochi_state,
                        soft_info,
                        soft_state,
                        island_state,
                        rigid_config,
                    )
                func_cholesky_solve_islands(
                    0,
                    False,
                    mochi_state.all_envs,
                    mochi_state.gate_newton,
                    mochi_info,
                    mochi_state,
                    island_state,
                    rigid_config,
                )
            func_update_linear_tolerance(
                0,
                False,
                mochi_state.all_envs,
                mochi_state.gate_newton,
                mochi_info,
                mochi_state,
                rigid_config,
                mochi_config,
            )
            func_pcg_init(
                0,
                False,
                mochi_state.all_envs,
                mochi_state.gate_newton,
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
            # [L] the line-search decision
            func_linesearch_decide(
                0,
                False,
                mochi_state.all_envs,
                mochi_state.gate_ls,
                rigid_info,
                mochi_info,
                mochi_state,
                rigid_config,
                mochi_config,
                mochi_state.graph_round_is_last_trial[None] == 1,
            )
            if qd.static(mochi_config.has_soft):
                func_soft_store_ls_ref(
                    0, False, mochi_state.all_envs, mochi_state.gate_ls, mochi_state, soft_state, rigid_config, True
                )
                if qd.static(has_rod):
                    func_rod_store_ls_ref(
                        0, False, mochi_state.all_envs, mochi_state.gate_ls, mochi_state, soft_state, rigid_config, True
                    )
            func_any_in_linesearch(mochi_state, rigid_config)
            func_round_end(round_counter, pcg_counter, mochi_state, n_pcg_rounds)
        # conjugate gradient of the S round (empty rounds when no environment needs it)
        while qd.graph.do_while(pcg_counter):
            func_any_pcg_active(mochi_state, rigid_config)
            for j in qd.static(range(pcg_unroll)):
                if qd.static(j > 0):
                    func_pcg_cap(pcg_counter, mochi_state, rigid_config, n_pcg_rounds, pcg_unroll, j, n_pcg)
                func_pcg_iter(
                    0,
                    False,
                    mochi_state.all_envs,
                    mochi_state.gate_pcg,
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
            func_any_pcg_active(mochi_state, rigid_config)
            func_pcg_round_end(pcg_counter, mochi_state)
        # [S] the line search of this iteration starts
        func_linesearch_begin(
            0, False, mochi_state.all_envs, mochi_state.gate_newton, rigid_info, mochi_state, rigid_config
        )
        if qd.static(mochi_config.has_soft):
            func_soft_store_ls_ref(
                0, False, mochi_state.all_envs, mochi_state.gate_newton, mochi_state, soft_state, rigid_config, False
            )
            if qd.static(has_rod):
                func_rod_store_ls_ref(
                    0,
                    False,
                    mochi_state.all_envs,
                    mochi_state.gate_newton,
                    mochi_state,
                    soft_state,
                    rigid_config,
                    False,
                )
        func_newton_end(newton_counter, round_counter, mochi_state, rigid_config, n_trials)

    # ---- epilogue: finite-difference velocities, history, final kinematics with the joint-space velocities ----
    func_post_stage(
        0,
        False,
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
            0, False, mochi_state.all_envs, mochi_state.n_envs_all, mochi_state, soft_info, soft_state, rigid_config
        )
        if qd.static(has_rod):
            func_rod_post_stage(
                0, False, mochi_state.all_envs, mochi_state.n_envs_all, mochi_state, soft_info, soft_state, rigid_config
            )
    func_update_kinematics(
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        geoms_init_AABB,
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
        True,
    )
