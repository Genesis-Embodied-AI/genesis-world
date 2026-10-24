import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import qd_to_torch
from . import linesearch
from . import solver

# Shared-memory reduction block size for the _jv path.
_JV_BLOCK = 32

# Maximum allowed alpha (prevents divergence from degenerate steps).
LS_ALPHA_MAX = 1e4


# ============================================== Shared iteration funcs ================================================


@qd.func
def _func_update_constraint_forces_body(
    i_c: int, i_b: int, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Per-element body for ``_func_update_constraint_forces``. Factored out so the two
    ndrange orderings (coalescing-optimal for each layout) share a single implementation."""
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    ncone = nef
    if qd.static(rigid_config.enable_elliptic_friction):
        ncone = ncone + constraint_state.n_constraints_cone[i_b]

    if qd.static(rigid_config.enable_elliptic_friction) and (nef <= i_c and i_c < ncone):
        # Elliptic cone (one-thread-per-row): only the head thread resolves the coupled rows and writes all of
        # them; the friction-row threads are no-ops (race-free). The coupled middle-zone cost is discarded here; the
        # linesearch evaluates the cone cost delta directly.
        if (i_c - nef) % qd.static(rigid_config.rows_per_contact) == 0:
            solver.func_cone_update_rows(i_c, i_b, constraint_state, rigid_config)
    else:
        if qd.static(
            rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.enable_elliptic_friction
        ):
            constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
        constraint_state.active[i_c, i_b] = True
        floss_force = gs.qd_float(0.0)

        if ne <= i_c and i_c < nef:
            f = constraint_state.efc_frictionloss[i_c, i_b]
            r = constraint_state.diag[i_c, i_b]
            rf = r * f
            linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
            linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
            constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
            floss_force = linear_neg * f + linear_pos * -f
        elif nef <= i_c:
            constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

        constraint_state.efc_force[i_c, i_b] = floss_force + (
            -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        )


@qd.func
def _func_update_constraint_forces_row(
    i_c: int, i_b: int, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Update the active flag and the force of one row of an iterating env, a row standing still keeping both."""
    if solver.func_is_row_moving(i_c, i_b, constraint_state, skip_settled_islands=True):
        _func_update_constraint_forces_body(i_c, i_b, constraint_state, rigid_config)
    elif qd.static(
        rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.enable_elliptic_friction
    ):
        constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]


@qd.func
def _func_update_constraint_forces(constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Compute active flags and efc_force, parallelized over (constraint, env).

    The cooperative kernels update the rows in the line search instead, see _func_update_constraint_forces_batch.
    """
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    # Snapshot prev_active in its own parallel pass so every row is captured before any active recompute: the cone head
    # thread rewrites its two tangent rows' active, which would otherwise race the tangent threads capturing
    # prev_active. Pyramidal threads only write their own row, so they snapshot inline in the body (no extra pass).
    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton and rigid_config.enable_elliptic_friction):
        qd.loop_config(name="snapshot_prev_active", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_c, i_b in qd.ndrange(len_constraints, _B):
            if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

    # A row of an island standing still keeps its values and shows no flip to the incremental factor, see
    # func_update_constraint_batch.
    qd.loop_config(name="update_constraint_forces", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(len_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            _func_update_constraint_forces_row(i_c, i_b, constraint_state, rigid_config)


@qd.func
def _func_update_constraint_forces_batch(
    i_b: int, tid: int, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Compute the active flags and efc_force of the rows of one env that moved, by the _K lanes of its block.

    The lanes stride over the rows of the env, which fills a small share of the constraint capacity. The block waits
    for the step applied to every row first, and under the elliptic cone the previous active flags are all captured
    before any is recomputed, see _func_update_constraint_forces.
    """
    _K = qd.static(32)
    n_con = constraint_state.n_constraints[i_b]
    qd.simt.block.sync()
    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton and rigid_config.enable_elliptic_friction):
        for i_chunk_ in range((n_con + _K - 1) // _K):
            i_c = i_chunk_ * _K + tid
            if i_c < n_con:
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]
        qd.simt.block.sync()
    for i_chunk_ in range((n_con + _K - 1) // _K):
        i_c = i_chunk_ * _K + tid
        if i_c < n_con:
            _func_update_constraint_forces_row(i_c, i_b, constraint_state, rigid_config)


@qd.func
def _func_update_qfrc_constraint_per_dof(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
    update_gradient: qd.template(),
):
    """Compute qfrc_constraint = J^T @ efc_force with one thread per (dof, env).

    Under update_gradient the thread also forms the gradient of its dof from it, see func_update_gradient_no_solve.

    A DOF only couples to constraints in its own island (a constraint touching the DOF is always in its island), so
    the sum runs over that island's constraints (constraint_id) rather than all n_con - identical result, but O(nnz)
    instead of O(n_dofs * n_con). The per-step constraint order is fixed, so the sum stays deterministic.

    Under ``enable_cooperative_constraint_kernels`` the outer ndrange is swapped so adjacent lanes vary i_d: the
    qfrc_constraint write coalesces under the flipped DOF-vec layout.
    """
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_qfrc", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if rigid_config.enable_cooperative_constraint_kernels else None)
    ):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            # A dof of an island standing still keeps its value, see func_update_constraint_batch. A single-island
            # scene sums the env's rows by index, its one island holding every row in order.
            is_island_moving = True
            con_base = 0
            con_n = constraint_state.n_constraints[i_b]
            if qd.static(not rigid_config.is_single_island):
                i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
                is_island_moving = constraint_state.island.improved[i_island, i_b]
                con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
                con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
            if is_island_moving:
                qfrc = gs.qd_float(0.0)
                for i_lcon in range(con_n):
                    i_c = con_base + i_lcon
                    if qd.static(not rigid_config.is_single_island):
                        i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
                    qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = qfrc
                if qd.static(update_gradient):
                    constraint_state.grad[i_d, i_b] = (
                        constraint_state.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b] - qfrc
                    )


@qd.func
def _func_update_gradient(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Step 5: update the gradient of every env still iterating, one env per thread.

    Where the solve is seeded one env per thread (see has_per_env_solve_seed in array_class.py), the Hessian of the env
    is assembled and factored first.
    """
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="update_gradient", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            if qd.static(rigid_config.has_per_env_solve_seed):
                solver.func_hessian_and_cholesky_factor_direct_batch(
                    i_b, constraint_state, dyn_info, rigid_info, rigid_config
                )
            solver.func_update_gradient_batch(i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)


@qd.func
def _func_islands_linesearch_and_apply(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Steps 1 to 4: the lockstep line search of every island of an env and the step applied, by the lanes of the env's
    block (see linesearch.py).

    A single-island scene runs the search of its one island by the lanes with the state in registers
    (func_search_single_island). Without the cooperative kernels (a GPU where they are disabled, the arm pinned
    regardless) lane 0 runs the serial sweeps.

    Under the cooperative kernels the lanes of an env that moved then update the active flag and the force of its rows
    (_func_update_constraint_forces_batch), while its rows are still in cache.

    The first lane opens the iteration of the graph loop: it counts the iteration and ends the loop, which the search
    direction of any env that iterates on resumes (see _func_islands_update_search_direction).
    """
    _K = qd.static(32)
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="islands_linesearch", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        if i_flat == 0:
            constraint_state.solver_iter_counter[()] = constraint_state.solver_iter_counter[()] + 1
            constraint_state.graph_counter_next[()] = graph_counter[()] - 1
            graph_counter[()] = 0
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            if qd.static(rigid_config.enable_cooperative_constraint_kernels and rigid_config.is_single_island):
                linesearch.func_mv_jv_coop(i_b, tid, constraint_state, rigid_info)
                is_moved = linesearch.func_search_single_island(
                    i_b, tid, _K, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_coop=True
                )
                if tid == 0:
                    constraint_state.improved[i_b] = is_moved
                if is_moved:
                    _func_update_constraint_forces_batch(i_b, tid, constraint_state, rigid_config)
            elif qd.static(rigid_config.enable_cooperative_constraint_kernels):
                sh_acc = qd.simt.block.SharedArray((10 * _K,), gs.qd_float)
                # The regime changes of the 'signorini' cost alone, a single slot otherwise (see func_cone_head_kinks in
                # linesearch.py)
                sh_kinks = qd.simt.block.SharedArray(
                    (qd.static(3 * _K if rigid_config.enable_signorini_contact else 1),), gs.qd_float
                )
                sh_alphas = qd.simt.block.SharedArray((3 * _K,), gs.qd_float)
                sh_n_alphas = qd.simt.block.SharedArray((_K,), gs.qd_int)
                sh_pending = qd.simt.block.SharedArray((_K,), gs.qd_int)
                sh_alpha = qd.simt.block.SharedArray((_K,), gs.qd_float)
                is_moved = linesearch.func_linesearch_islands_coop(
                    i_b,
                    tid,
                    sh_acc,
                    sh_kinks,
                    sh_alphas,
                    sh_n_alphas,
                    sh_pending,
                    sh_alpha,
                    dyn_state,
                    constraint_state,
                    dyn_info,
                    rigid_info,
                    rigid_config,
                )
                if tid == 0:
                    constraint_state.improved[i_b] = is_moved
                if is_moved:
                    _func_update_constraint_forces_batch(i_b, tid, constraint_state, rigid_config)
            else:
                if tid == 0:
                    constraint_state.improved[i_b] = linesearch.func_linesearch_islands_serial(
                        i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config
                    )


@qd.func
def _func_islands_update_search_direction(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Step 6: every island's exit test and next direction, see _func_islands_linesearch_and_apply for the lane
    arrangement.

    An env that iterates on resumes the graph loop with the iterations left, every such env writing the same count.
    """
    _K = qd.static(32)
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="islands_update_search_direction", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            if qd.static(rigid_config.enable_cooperative_constraint_kernels and rigid_config.is_single_island):
                improved = linesearch.func_exit_single_island(
                    i_b, tid, _K, constraint_state, rigid_info, rigid_config, is_coop=True, certify=False
                )
                if tid == 0:
                    constraint_state.improved[i_b] = improved
                    if improved:
                        graph_counter[()] = constraint_state.graph_counter_next[()]
            elif qd.static(rigid_config.enable_cooperative_constraint_kernels):
                sh_acc = qd.simt.block.SharedArray((7 * _K,), gs.qd_float)
                sh_pending = qd.simt.block.SharedArray((_K,), gs.qd_int)
                sh_alpha = qd.simt.block.SharedArray((_K,), gs.qd_float)
                improved = linesearch.func_exit_islands_coop(
                    i_b, tid, sh_acc, sh_pending, sh_alpha, constraint_state, rigid_info, rigid_config, certify=False
                )
                if tid == 0:
                    constraint_state.improved[i_b] = improved
                    if improved:
                        graph_counter[()] = constraint_state.graph_counter_next[()]
            else:
                if tid == 0:
                    improved = linesearch.func_exit_islands_serial(
                        i_b, constraint_state, rigid_info, rigid_config, certify=False
                    )
                    constraint_state.improved[i_b] = improved
                    if improved:
                        graph_counter[()] = constraint_state.graph_counter_next[()]


# ============================================== Solve body dispatch ================================================


@qd.kernel(graph=True, fastcache=True)
def _kernel_solve_graph(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    while qd.graph.do_while(graph_counter):
        # Every island's line search in lockstep, the step applied and the CG gradients saved per island
        _func_islands_linesearch_and_apply(
            graph_counter, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config
        )
        if qd.static(not rigid_config.enable_cooperative_constraint_kernels):
            _func_update_constraint_forces(constraint_state, rigid_config)
        # The tiled Newton factor below reads the gradient alone, so the thread of a dof forms it with its force
        is_tiled_newton = qd.static(
            rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.has_per_env_solve_seed
        )
        _func_update_qfrc_constraint_per_dof(dyn_state, constraint_state, rigid_config, update_gradient=is_tiled_newton)
        if qd.static(is_tiled_newton):
            # Every island's Hessian block maintained in nt_H, patched with the rows whose active state flipped, then a
            # tiled factor + solve of every island in its own shared tile, which reads the block without consuming it,
            # the coupled elliptic-cone block bracketed around the factor (see func_wrap_cone_hessian).
            solver.func_island_hessian_assemble_all(constraint_state, rigid_info, rigid_config, patch=True)
            solver.func_wrap_cone_hessian(constraint_state, rigid_config, is_removal=False, is_enabled=True)
            solver.func_island_tiled_factor_solve_all(
                constraint_state, dyn_info, rigid_info, rigid_config, write_L=False
            )
            solver.func_wrap_cone_hessian(constraint_state, rigid_config, is_removal=True, is_enabled=True)
        else:
            _func_update_gradient(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
        _func_islands_update_search_direction(graph_counter, dyn_state, constraint_state, rigid_info, rigid_config)


@solver.func_solve_body.register(
    is_compatible=lambda *args, **kwargs: (
        not (rigid_config := solver._get_static_config(*args, **kwargs)).requires_grad
        and rigid_config.prefer_decomposed_solver != 0
    )
)
def func_solve_decomposed(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, _n_iterations):
    """
    GPU graph accelerated solver loop with parallel grid-search linesearch and GPU-side iteration via graph.do_while.

    On CUDA SM 9.0+ (Hopper), the entire iteration loop runs on the GPU with no host involvement. On older CUDA GPUs,
    falls back to a host-side do-while loop that still benefits from CUDA graph kernel launch batching. On other GPUs,
    falls back to a host-side C++-side loop, that still reduces python launch overhead.

    Early exits when all batch elements have converged (no improved[i_b] is True).

    The per-iteration factor/solve runs per island over the (env, island) work-list, an unpartitioned env being a
    single island spanning every dof, except where the solve is seeded one env per thread (see has_per_env_solve_seed
    in array_class.py): the seed and every factor then run one env per thread.
    """
    # The graph maintains the assembled Hessian in nt_H, so the seed leaves it there (write_L=False).
    solver.func_solve_init(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, write_L=False)
    if _n_iterations <= 0:
        return
    if gs.use_zerocopy:
        graph_counter = qd_to_torch(constraint_state.graph_counter, copy=False)
        graph_counter.fill_(_n_iterations)
        if gs.backend == gs.metal:
            torch.mps.synchronize()
    else:
        constraint_state.graph_counter.fill(_n_iterations)
    _kernel_solve_graph(constraint_state.graph_counter, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
