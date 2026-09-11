import sys

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from . import linesearch
from . import solver

# Shared-memory reduction block size for the _jv path.
_JV_BLOCK = 32

# Maximum allowed alpha (prevents divergence from degenerate steps).
LS_ALPHA_MAX = 1e4


# ============================================== Shared iteration funcs ================================================


@qd.func
def _func_update_constraint_forces_body(
    i_c, i_b, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
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
def _func_update_constraint_forces(constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Compute active flags and efc_force, parallelized over (constraint, env).

    Iteration order is picked at compile time so adjacent lanes always cover the *physical* contiguous dimension of the
    layout-flippable constraint-state tensors:
      - layout False (canonical [i_c, i_b], physical [i_c, i_b]):  ndrange(len_constraints, _B)
      - layout True  (canonical [i_c, i_b], physical [i_b, i_c]):  ndrange(_B, len_constraints)
    """
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    # Snapshot prev_active in its own parallel pass so every row is captured before any active recompute: the cone head
    # thread rewrites its two tangent rows' active, which would otherwise race the tangent threads capturing
    # prev_active. Pyramidal threads only write their own row, so they snapshot inline in the body (no extra pass).
    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton and rigid_config.enable_elliptic_friction):
        qd.loop_config(name="snapshot_prev_active")
        for i_c, i_b in qd.ndrange(
            len_constraints, _B, axes=qd.static((1, 0) if rigid_config.enable_cooperative_constraint_kernels else None)
        ):
            if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

    # A row of an island standing still keeps its values and shows no flip to the incremental factor, see
    # func_update_constraint_batch.
    qd.loop_config(name="update_constraint_forces")
    for i_c, i_b in qd.ndrange(
        len_constraints, _B, axes=qd.static((1, 0) if rigid_config.enable_cooperative_constraint_kernels else None)
    ):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            if solver.func_is_row_moving(i_c, i_b, constraint_state, skip_settled_islands=True):
                _func_update_constraint_forces_body(i_c, i_b, constraint_state, rigid_config)
            elif qd.static(
                rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.enable_elliptic_friction
            ):
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]


@qd.func
def _func_update_qfrc_constraint_per_dof(constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Compute qfrc_constraint = J^T @ efc_force with one thread per (dof, env).

    A DOF only couples to constraints in its own island (a constraint touching the DOF is always in its island), so
    the sum runs over that island's constraints (constraint_id) rather than all n_con - identical result, but O(nnz)
    instead of O(n_dofs * n_con). The per-step constraint order is fixed, so the sum stays deterministic.

    Under ``enable_cooperative_constraint_kernels`` the outer ndrange is swapped so adjacent lanes vary i_d: the
    qfrc_constraint write coalesces under the flipped DOF-vec layout.
    """
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_qfrc")
    for i_d, i_b in qd.ndrange(
        n_dofs, _B, axes=qd.static((1, 0) if rigid_config.enable_cooperative_constraint_kernels else None)
    ):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            # A dof of an island standing still keeps its value, see func_update_constraint_batch
            i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
            if constraint_state.island.improved[i_island, i_b]:
                qfrc = gs.qd_float(0.0)
                con_base = constraint_state.island.constraint_slices.start[i_island, i_b]
                con_n = constraint_state.island.constraint_slices.n[i_island, i_b]
                for i_lcon in range(con_n):
                    i_c = constraint_state.island.constraint_id[con_base + i_lcon, i_b]
                    qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = qfrc


@qd.func
def _func_update_gradient(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Step 5: Update gradient"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="update_gradient", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_update_gradient_batch(i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)


@qd.func
def _func_islands_linesearch_and_apply(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Steps 1 to 4: the lockstep line search of every island of an env and the step applied, by the lanes of the env's
    block (see linesearch.py).

    Without the cooperative kernels (a GPU where they are disabled, the arm pinned regardless) lane 0 runs the serial
    sweeps.
    """
    _K = qd.static(32)
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="islands_linesearch", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        sh_acc = qd.simt.block.SharedArray((9 * _K,), gs.qd_float)
        sh_alphas = qd.simt.block.SharedArray((3 * _K,), gs.qd_float)
        sh_n_alphas = qd.simt.block.SharedArray((_K,), gs.qd_int)
        sh_pending = qd.simt.block.SharedArray((_K,), gs.qd_int)
        sh_alpha = qd.simt.block.SharedArray((_K,), gs.qd_float)
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            if qd.static(rigid_config.enable_cooperative_constraint_kernels):
                is_moved = linesearch.func_linesearch_islands_coop(
                    i_b,
                    tid,
                    sh_acc,
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
            else:
                if tid == 0:
                    constraint_state.improved[i_b] = linesearch.func_linesearch_islands_serial(
                        i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config
                    )


@qd.func
def _func_islands_update_search_direction(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Step 6: every island's exit test and next direction, see
    _func_islands_linesearch_and_apply for the lane arrangement."""
    _K = qd.static(32)
    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="islands_update_search_direction", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        sh_acc = qd.simt.block.SharedArray((7 * _K,), gs.qd_float)
        sh_pending = qd.simt.block.SharedArray((_K,), gs.qd_int)
        sh_alpha = qd.simt.block.SharedArray((_K,), gs.qd_float)
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            if qd.static(rigid_config.enable_cooperative_constraint_kernels):
                improved = linesearch.func_exit_islands_coop(
                    i_b, tid, sh_acc, sh_pending, sh_alpha, constraint_state, rigid_info, rigid_config, certify=False
                )
                if tid == 0:
                    constraint_state.improved[i_b] = improved
            else:
                if tid == 0:
                    constraint_state.improved[i_b] = linesearch.func_exit_islands_serial(
                        i_b, constraint_state, rigid_info, rigid_config, certify=False
                    )


@qd.func
def _func_check_early_exit(
    graph_counter: qd.types.ndarray(qd.i32, ndim=0), constraint_state: array_class.ConstraintState
):
    """Decrement iteration counter and exit early if no batch element improved. solver_iter_counter counts the
    iterations the graph ran, whichever path it took."""
    qd.loop_config(name="check_early_exit_reset_flag")
    for _ in range(1):
        graph_counter[()] = graph_counter[()] - 1
        constraint_state.solver_iter_counter[()] = constraint_state.solver_iter_counter[()] + 1
        constraint_state.early_exit_flag[()] = 0

    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="check_early_exit_scan_values")
    for i_b in range(_B):
        if constraint_state.improved[i_b]:
            qd.atomic_max(constraint_state.early_exit_flag[()], 1)

    qd.loop_config(name="check_early_exit_set_counter")
    for _ in range(1):
        if constraint_state.early_exit_flag[()] == 0:
            graph_counter[()] = 0


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
        _func_islands_linesearch_and_apply(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
        _func_update_constraint_forces(constraint_state, rigid_config)
        _func_update_qfrc_constraint_per_dof(constraint_state, rigid_config)
        if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton):
            # Every island's Hessian block maintained in nt_H, patched with the rows whose active state flipped, then a
            # tiled factor + solve of every island in its own shared tile, which reads the block without consuming it,
            # the coupled elliptic-cone block bracketed around the factor (see func_wrap_cone_hessian).
            solver.func_island_hessian_assemble_all(constraint_state, rigid_info, rigid_config, patch=True)
            solver.func_update_gradient_no_solve(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
            solver.func_wrap_cone_hessian(constraint_state, rigid_config, is_removal=False, is_enabled=True)
            solver.func_island_tiled_factor_solve_all(
                constraint_state, dyn_info, rigid_info, rigid_config, write_L=False
            )
            solver.func_wrap_cone_hessian(constraint_state, rigid_config, is_removal=True, is_enabled=True)
        else:
            _func_update_gradient(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
        _func_islands_update_search_direction(dyn_state, constraint_state, rigid_info, rigid_config)
        _func_check_early_exit(graph_counter, constraint_state)


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
    single island spanning every dof.
    """
    # The graph maintains the assembled Hessian in nt_H, so the seed leaves it there (write_L=False).
    solver.func_solve_init(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, write_L=False)
    if _n_iterations <= 0:
        return
    constraint_state.graph_counter.from_numpy(np.array(_n_iterations, dtype=np.int32))
    _kernel_solve_graph(constraint_state.graph_counter, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)
