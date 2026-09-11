"""Per-island Newton / CG iteration of the rigid constraint solver, every island of an env advancing in lockstep.

The island partition (see island.py) makes the mass matrix, the Jacobian and hence the cost block-diagonal, so each
island owns a line search, a convergence test and a search direction of its own, measured against the trace of its own
mass block (IslandState.inertia in array_class.py), and a converged island stands still while the others iterate. The
monolith arm (one thread per env) takes the islands one after the other, each search sweeping its own dofs and rows
through the island-ordered dof_id / constraint_id lists with its state in registers. The decomposed arm runs all the
islands of a group of _K in lockstep with the lanes of the env's block, sweeping the group's range of those lists in
chunks of _K items and reducing each chunk by island with a segmented subgroup sum whose segment tail is the single
writer of its island's slot, so the accumulation order is that of the chunks whatever the lanes do. A range of either
list holding consecutive indices (every island of a single-island env, a group of whole trees in tree order) is swept by
offset without reading the list, see func_list_range_start.

The line search of one island is MuJoCo's bracketing line search (a Newton walk along the search direction until the
directional derivative changes sign, then a three-candidate refinement between the bracket points), written as a state
machine (phase, literal in the code): 0 evaluates the first trial step, 1 walks the bracket one Newton step at a time, 2
refines with three candidates at once, 3 is done, -1 marks an island that stands still this iteration (converged or
asleep), its statistics of the iteration that converged it left untouched. A round evaluates the pending candidates and
advances the island by one transition, until it is done. The scalar functions below are shared by both arms.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from . import solver as constraint_solver

# ======================================================================================================================
# ================================================== Island item lists ================================================
# ======================================================================================================================


@qd.func
def func_list_range_start(ids: qd.Tensor, lo, hi, i_b):
    """First index of the ascending id list ids[lo:hi] of one env when its indices are consecutive, -1 otherwise (0 for
    an empty range).

    A sweep over a consecutive range indexes its items directly, see func_list_item. The constraint list ascends by
    construction; the dof list only where dof_range_start (array_class.py) says so, since the CPU skyline path reorders
    it.
    """
    start = 0
    if hi > lo:
        start = ids[lo, i_b]
        if ids[hi - 1, i_b] - start + 1 != hi - lo:
            start = -1
    return start


@qd.func
def func_list_item(ids: qd.Tensor, i_pos, lo, range_start, i_b):
    """Item at position i_pos of the id list ids[lo:...] of one env: an offset from ``range_start`` when the range is
    consecutive (see func_list_range_start), otherwise the list entry."""
    i = range_start + (i_pos - lo)
    if range_start < 0:
        i = ids[i_pos, i_b]
    return i


@qd.func
def func_group_dof_range_start(i_b, tid, base, n_group, dof_lo, dof_hi, constraint_state: array_class.ConstraintState):
    """First dof of a group of _K islands whose dof lists, laid end to end, hold consecutive dofs, -1 otherwise.

    Every island of the group is in build order (dof_range_start) and the whole range consecutive, decided by all the
    lanes.
    """
    is_in_order = True
    if tid < n_group:
        is_in_order = constraint_state.island.dof_range_start[base + tid, i_b] >= 0
    range_start = -1
    if qd.simt.subgroup.all_true(is_in_order) != 0:
        range_start = func_list_range_start(constraint_state.island.dof_id, dof_lo, dof_hi, i_b)
    return range_start


# ======================================================================================================================
# ================================================== Per-item terms ===================================================
# ======================================================================================================================


@qd.func
def func_row_p0_terms(
    i_c, i_b, ne, nef, ncone, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Linear and quadratic coefficients along the search direction, at alpha = 0, of one constraint row.

    The equality rows count apart (they never deactivate) and the active rows of any kind contribute, an elliptic cone
    through its head row only.
    """
    eq_1 = gs.qd_float(0.0)
    eq_2 = gs.qd_float(0.0)
    tot_1 = gs.qd_float(0.0)
    tot_2 = gs.qd_float(0.0)
    is_cone_row = False
    if qd.static(rigid_config.enable_elliptic_friction):
        is_cone_row = nef <= i_c and i_c < ncone
    if is_cone_row:
        n_rows = qd.static(rigid_config.rows_per_contact)
        if (i_c - nef) % n_rows == 0:
            rows_efc_D, rows_friction, con_mu, rows_jaref = constraint_solver._func_cone_head_load(
                i_c, i_b, constraint_state, rigid_config
            )
            rows_jv = qd.Vector.zero(gs.qd_float, n_rows)
            for i_r in qd.static(range(n_rows)):
                rows_jv[i_r] = constraint_state.jv[i_c + i_r, i_b]
            _cost_c, grad_c, hess_c = constraint_solver._func_cone_cost_along_alpha(
                rows_jaref, rows_jv, 0.0, rows_efc_D, con_mu, rows_friction, rigid_config
            )
            tot_1 = grad_c
            tot_2 = 0.5 * hess_c
    else:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        if i_c < ne:
            eq_1 = qf_1
            eq_2 = qf_2
            tot_1 = qf_1
            tot_2 = qf_2
        elif i_c < nef:
            f = constraint_state.efc_frictionloss[i_c, i_b]
            rf = constraint_state.diag[i_c, i_b] * f
            linear_neg = Jaref_c <= -rf
            linear_pos = Jaref_c >= rf
            if linear_neg or linear_pos:
                qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
                qf_2 = 0.0
            tot_1 = qf_1
            tot_2 = qf_2
        else:
            active = Jaref_c < 0
            tot_1 = qf_1 * active
            tot_2 = qf_2 * active
    return eq_1, eq_2, tot_1, tot_2


@qd.func
def func_row_alpha_terms(
    i_c,
    i_b,
    n_alphas,
    alphas,
    ne,
    nef,
    ncone,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """The (const, linear, quad) coefficient triple of one constraint row at each pending candidate step.

    The triples are taken at the first n_alphas candidates of alphas, in the shifted convention cost(alpha) - cost(0)
    (see ls_improvement in IslandState, array_class.py): an equality row contributes nothing (its terms sit in the
    initialization sums), an elliptic cone contributes through its head row only.

    Returns the triples as t_0 (const per candidate), t_1 (linear), t_2 (quad), zero past n_alphas.
    """
    t_0 = qd.Vector.zero(gs.qd_float, 3)
    t_1 = qd.Vector.zero(gs.qd_float, 3)
    t_2 = qd.Vector.zero(gs.qd_float, 3)
    is_cone_row = False
    if qd.static(rigid_config.enable_elliptic_friction):
        is_cone_row = i_c < ncone
    if i_c < ne:
        pass
    elif i_c < nef:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f = constraint_state.efc_frictionloss[i_c, i_b]
        r = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        rf = r * f
        ln0 = Jaref_c <= -rf
        lp0 = Jaref_c >= rf
        cost0 = qf_0
        if ln0 or lp0:
            cost0 = ln0 * f * (-0.5 * rf - Jaref_c) + lp0 * f * (-0.5 * rf + Jaref_c)
        for k in qd.static(range(3)):
            if k < n_alphas:
                x = Jaref_c + alphas[k] * jv_c
                ln = x <= -rf
                lp = x >= rf
                ak_qf_0, ak_qf_1, ak_qf_2 = qf_0, qf_1, qf_2
                if ln or lp:
                    ak_qf_0 = ln * f * (-0.5 * rf - Jaref_c) + lp * f * (-0.5 * rf + Jaref_c)
                    ak_qf_1 = ln * (-f * jv_c) + lp * (f * jv_c)
                    ak_qf_2 = 0.0
                t_0[k] = ak_qf_0 - cost0
                t_1[k] = ak_qf_1
                t_2[k] = ak_qf_2
    elif is_cone_row:
        n_rows = qd.static(rigid_config.rows_per_contact)
        if (i_c - nef) % n_rows == 0:
            rows_efc_D, rows_friction, con_mu, rows_jaref = constraint_solver._func_cone_head_load(
                i_c, i_b, constraint_state, rigid_config
            )
            rows_jv = qd.Vector.zero(gs.qd_float, n_rows)
            for i_r in qd.static(range(n_rows)):
                rows_jv[i_r] = constraint_state.jv[i_c + i_r, i_b]
            for k in qd.static(range(3)):
                if k < n_alphas:
                    alpha_k = alphas[k]
                    cost_diff_c, grad_c, hess_c = constraint_solver._func_cone_cost_diff_along_alpha(
                        rows_jaref, rows_jv, alpha_k, rows_efc_D, con_mu, rows_friction, rigid_config
                    )
                    t_0[k] = cost_diff_c - grad_c * alpha_k + 0.5 * hess_c * alpha_k * alpha_k
                    t_1[k] = grad_c - hess_c * alpha_k
                    t_2[k] = 0.5 * hess_c
    else:
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        act0 = gs.qd_bool(Jaref_c < 0)
        for k in qd.static(range(3)):
            if k < n_alphas:
                act = gs.qd_bool(Jaref_c + alphas[k] * jv_c < 0)
                t_0[k] = qf_0 * act - qf_0 * act0
                t_1[k] = qf_1 * act
                t_2[k] = qf_2 * act
    return t_0, t_1, t_2


@qd.func
def func_dof_p0_terms(i_d, i_b, dyn_state: array_class.DynState, constraint_state: array_class.ConstraintState):
    """Per-dof terms of the line search initialization: squared search direction, squared gradient, and the Gauss
    (unconstrained) linear and quadratic coefficients along the search direction."""
    s = constraint_state.search[i_d, i_b]
    return (
        s * s,
        constraint_state.grad[i_d, i_b] ** 2,
        s * constraint_state.Ma[i_d, i_b] - s * dyn_state.dofs.force[i_d, i_b],
        0.5 * s * constraint_state.mv[i_d, i_b],
    )


@qd.func
def func_dof_exit_terms(i_d, i_b, constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Per-dof terms of the convergence test: squared gradient, the descent grad .

    Mgrad, and for CG the Hager-Zhang products (d . y, y . My, y . Mgrad, d . grad, |d|^2), see func_exit_decision.
    """
    grad = constraint_state.grad[i_d, i_b]
    Mgrad = constraint_state.Mgrad[i_d, i_b]
    terms = qd.Vector.zero(gs.qd_float, 7)
    terms[0] = grad * grad
    terms[1] = grad * Mgrad
    if qd.static(rigid_config.solver_type == gs.constraint_solver.CG):
        search = constraint_state.search[i_d, i_b]
        y = grad - constraint_state.cg_prev_grad[i_d, i_b]
        My = Mgrad - constraint_state.cg_prev_Mgrad[i_d, i_b]
        terms[2] = search * y
        terms[3] = y * My
        terms[4] = y * Mgrad
        terms[5] = search * grad
        terms[6] = search * search
    return terms


# ======================================================================================================================
# ============================================ Register-resident island state ==========================================
# ======================================================================================================================


@qd.func
def func_ls_state_init(sums, inertia, rigid_info: array_class.RigidInfo, rigid_config: qd.template()):
    """Initialize one island's line search from its eight initialization sums (|s|^2, |g|^2, the Gauss linear and
    quadratic coefficients, the equality ones, the all-rows ones), on the island's own inertia scale.

    Returns the phase (3 with status 1 for a vanishing search direction), the gradient tolerance, the constant
    coefficients of every evaluation, the derivatives at alpha = 0 and the first trial step.
    """
    EPS = rigid_info.EPS[None]
    snorm = qd.sqrt(sums[0])
    # The mass scale of the convergence tests, see func_exit_decision
    scale = inertia
    ls_ratio = rigid_info.tolerance[None] * rigid_info.ls_tolerance[None]
    if qd.static(not rigid_config.enable_mujoco_compatibility):
        # The gradient norm carries the scale: the line search tolerance then follows the residual left to reduce
        # rather than the inertia, which a solve far from convergence would let dominate.
        scale = qd.sqrt(sums[1])
        LS_NOISE_RATIO = qd.static(10.0)
        ls_ratio = qd.max(ls_ratio, LS_NOISE_RATIO * EPS)
    gtol = ls_ratio * snorm * scale
    base_1 = sums[2] + sums[4]
    base_2 = sums[3] + sums[5]
    p0_deriv_0 = sums[2] + sums[6]
    p0_deriv_1 = 2.0 * (sums[3] + sums[7])
    if p0_deriv_1 <= 0.0:
        p0_deriv_1 = EPS
    phase = 0
    ls_result = 0
    alpha_0 = -p0_deriv_0 / p0_deriv_1
    if snorm < EPS:
        phase = 3
        ls_result = 1
        alpha_0 = 0.0
    return phase, ls_result, gtol, base_1, base_2, p0_deriv_0, p0_deriv_1, alpha_0


@qd.func
def func_ls_state_advance(
    acc,
    n_alphas,
    alphas,
    phase,
    p1,
    p2,
    p2update,
    direction,
    ls_it,
    gtol,
    base_1,
    base_2,
    p0_deriv_0,
    p0_deriv_1,
    rigid_info: array_class.RigidInfo,
):
    """Advance one island's line search by one transition from the accumulated terms of its pending candidates.

    The transition covers the Newton walk, the three-candidate refinement and their fallbacks, from the terms acc of
    the n_alphas pending candidates alphas, the bracket points p1 / p2 carried between rounds as (alpha, cost, first,
    second derivative).

    Returns the new phase, candidate count and candidates, bracket points, flags and evaluation count, and once done
    the accepted step, its improvement and the status code.
    """
    EPS = rigid_info.EPS[None]
    ls_iter_limit = rigid_info.ls_iterations[None]
    costs = qd.Vector.zero(gs.qd_float, 3)
    grads = qd.Vector.zero(gs.qd_float, 3)
    hess = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        alpha_k = alphas[k]
        t1 = base_1 + acc[3 * k + 1]
        t2 = base_2 + acc[3 * k + 2]
        costs[k] = alpha_k * alpha_k * t2 + alpha_k * t1 + acc[3 * k]
        grads[k] = 2.0 * alpha_k * t2 + t1
        hess[k] = 2.0 * t2
        if hess[k] <= 0.0:
            hess[k] = EPS
    ls_it = ls_it + n_alphas

    p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = p1[0], p1[1], p1[2], p1[3]
    p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p2[0], p2[1], p2[2], p2[3]
    alphas_next = qd.Vector.zero(gs.qd_float, 3)
    res_alpha = gs.qd_float(0.0)
    res_cost = gs.qd_float(0.0)
    ls_result = 0
    is_done = False
    is_walking = False

    if phase == 0:
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alphas[0], costs[0], grads[0], hess[0]
        # Costs are shifted deltas from alpha = 0 (see ls_improvement in array_class.py): a positive one means no
        # improvement over alpha = 0, which takes its place.
        if p1_cost > 0.0:
            p1_alpha = 0.0
            p1_cost = 0.0
            p1_deriv_0 = p0_deriv_0
            p1_deriv_1 = p0_deriv_1
        if qd.abs(p1_deriv_0) < gtol:
            if qd.abs(p1_alpha) < EPS:
                ls_result = 2
            res_alpha = p1_alpha
            res_cost = p1_cost
            is_done = True
        else:
            direction = (p1_deriv_0 < 0) * 2 - 1
            p2update = 0
            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
            is_walking = True
    elif phase == 1:
        p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1 = alphas[0], costs[0], grads[0], hess[0]
        if qd.abs(p1_deriv_0) < gtol:
            res_alpha = p1_alpha
            res_cost = p1_cost
            is_done = True
        else:
            is_walking = True
    else:
        best_a = gs.qd_float(0.0)
        best_c = gs.qd_float(0.0)
        has_best = False
        for k in qd.static(range(3)):
            if qd.abs(grads[k]) < gtol and (not has_best or costs[k] < best_c):
                best_a = alphas[k]
                best_c = costs[k]
                has_best = True
        if has_best:
            res_alpha = best_a
            res_cost = best_c
            is_done = True
        else:
            b1, p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, p1_next = constraint_solver.update_bracket_no_eval_local(
                p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1, alphas, costs, grads, hess
            )
            b2, p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1, p2_next = constraint_solver.update_bracket_no_eval_local(
                p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1, alphas, costs, grads, hess
            )
            if b1 == 0 and b2 == 0:
                if costs[2] >= 0.0:
                    ls_result = 7
                res_alpha = alphas[2]
                res_cost = costs[2]
                is_done = True
            elif ls_it < ls_iter_limit:
                alphas_next[0] = p1_next
                alphas_next[1] = p2_next
                alphas_next[2] = (p1_alpha + p2_alpha) * 0.5
            else:
                if p1_cost <= p2_cost and p1_cost < 0.0:
                    ls_result = 4
                    res_alpha = p1_alpha
                    res_cost = p1_cost
                elif p2_cost <= p1_cost and p2_cost < 0.0:
                    ls_result = 4
                    res_alpha = p2_alpha
                    res_cost = p2_cost
                else:
                    ls_result = 5
                is_done = True

    if is_walking:
        # One Newton step along the bracket while the derivative keeps its sign and the budget lasts, then the
        # three-candidate refinement between the two bracket points.
        if p1_deriv_0 * direction <= -gtol and ls_it < ls_iter_limit:
            p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1 = p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1
            p2update = 1
            alphas_next[0] = p1_alpha - p1_deriv_0 / p1_deriv_1
            n_alphas = 1
            phase = 1
        elif ls_it >= ls_iter_limit:
            ls_result = 3
            res_alpha = p1_alpha
            res_cost = p1_cost
            is_done = True
        elif p2update == 0:
            ls_result = 6
            res_alpha = p1_alpha
            res_cost = p1_cost
            is_done = True
        else:
            alphas_next[0] = p1_alpha - p1_deriv_0 / p1_deriv_1
            alphas_next[1] = p1_alpha
            alphas_next[2] = (p1_alpha + p2_alpha) * 0.5
            n_alphas = 3
            phase = 2

    improvement = -res_cost
    if is_done:
        # Status 7: both brackets stalled and the midpoint does not improve on alpha = 0. Reject the step
        if ls_result == 7:
            res_alpha = 0.0
            improvement = 0.0
        phase = 3
    p1_next_pt = qd.Vector([p1_alpha, p1_cost, p1_deriv_0, p1_deriv_1])
    p2_next_pt = qd.Vector([p2_alpha, p2_cost, p2_deriv_0, p2_deriv_1])
    return (
        phase,
        n_alphas,
        alphas_next,
        p1_next_pt,
        p2_next_pt,
        p2update,
        direction,
        ls_it,
        res_alpha,
        improvement,
        ls_result,
    )


@qd.func
def func_exit_decision(terms, improvement, inertia, rigid_info: array_class.RigidInfo, rigid_config: qd.template()):
    """Convergence test of one island from its exit terms (see func_dof_exit_terms) and the improvement of its last line
    search, on the island's own inertia scale: a flat gradient or a stalled improvement stops the iteration, and for
    Newton the descent grad .

    Mgrad must stay above the tolerance; for CG also the Hager-Zhang conjugate coefficient the direction update reads.
    """
    tol_scaled = inertia * rigid_info.tolerance[None]
    grad_norm = qd.sqrt(terms[0])
    descent = terms[1]
    is_flat = grad_norm <= tol_scaled
    is_stalled = improvement > 0.0 and improvement < tol_scaled
    improved = not (is_flat or is_stalled)
    if qd.static(rigid_config.enable_signorini_contact):
        improved = not (is_flat and 0.5 * descent <= tol_scaled)
    elif qd.static(
        rigid_config.solver_type == gs.constraint_solver.Newton and not rigid_config.enable_mujoco_compatibility
    ):
        # TODO: Remove the MuJoCo compatibility gate once on MuJoCo 3.11, which carries this criterion natively
        improved = improved and qd.max(0.5 * descent, 0.0) >= tol_scaled
    cg_beta = gs.qd_float(0.0)
    if qd.static(rigid_config.solver_type == gs.constraint_solver.CG):
        d_dot_y = terms[2]
        if improved and d_dot_y >= rigid_info.EPS[None]:
            beta_hz = (terms[4] - 2.0 * (terms[3] / d_dot_y) * terms[5]) / d_dot_y
            eta_k = -1.0 / qd.max(rigid_info.EPS[None], qd.sqrt(terms[6]) * qd.min(gs.qd_float(0.01), grad_norm))
            cg_beta = qd.max(eta_k, beta_hz)
    return improved, cg_beta


@qd.func
def func_certify_decision(terms, inertia, rigid_info: array_class.RigidInfo, rigid_config: qd.template()):
    """Warm-start certificate of one island from its exit terms.

    The warm-started acceleration is kept as the solution when its Newton decrement and, for Newton or the Signorini
    contact, its gradient norm sit below the tolerance.
    """
    tolerance_scaled = inertia * rigid_info.tolerance[None]
    is_converged = qd.max(0.5 * terms[1], 0.0) < tolerance_scaled
    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton or rigid_config.enable_signorini_contact):
        is_converged = is_converged and qd.sqrt(terms[0]) <= tolerance_scaled
    return is_converged


# ======================================================================================================================
# ================================================ Monolith arm (serial) ==============================================
# ======================================================================================================================


@qd.func
def func_mv_jv_dense(i_b, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo):
    """mv = M @ search over the mass blocks and jv = J @ search over every row and dof of one env, the dense read of an
    env whose single island spans it (see is_single_island), whose loads carry no dependent index."""
    n_dofs = constraint_state.search.shape[0]
    for i_d1 in range(n_dofs):
        mv = gs.qd_float(0.0)
        for i_d2 in range(rigid_info.dofs_mass_block_start[i_d1], rigid_info.dofs_mass_block_end[i_d1]):
            mv = mv + rigid_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
        constraint_state.mv[i_d1, i_b] = mv
    for i_c in range(constraint_state.n_constraints[i_b]):
        jv = gs.qd_float(0.0)
        for i_d in range(n_dofs):
            jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
        constraint_state.jv[i_c, i_b] = jv


@qd.func
def func_mv_jv_islands(i_b, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo):
    """mv and jv of the islands of one env still iterating, each row read over its sparse support, so the cost follows
    the islands' sizes."""
    for i_island in range(constraint_state.island.n_islands[i_b]):
        if constraint_state.island.improved[i_island, i_b]:
            dof_lo = constraint_state.island.dof_slices.start[i_island, i_b]
            dof_hi = dof_lo + constraint_state.island.dof_slices.n[i_island, i_b]
            for i_pos in range(dof_lo, dof_hi):
                i_d1 = constraint_state.island.dof_id[i_pos, i_b]
                mv = gs.qd_float(0.0)
                for i_d2 in range(rigid_info.dofs_mass_block_start[i_d1], rigid_info.dofs_mass_block_end[i_d1]):
                    mv = mv + rigid_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
                constraint_state.mv[i_d1, i_b] = mv
            row_lo = constraint_state.island.constraint_slices.start[i_island, i_b]
            row_hi = row_lo + constraint_state.island.constraint_slices.n[i_island, i_b]
            for i_pos in range(row_lo, row_hi):
                i_c = constraint_state.island.constraint_id[i_pos, i_b]
                jv = gs.qd_float(0.0)
                for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                    jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
                constraint_state.jv[i_c, i_b] = jv


@qd.func
def func_linesearch_islands_serial(
    i_b,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Line search of every island of one env by a single thread, then the step applied.

    mv and jv are formed over the env, then each awake island is searched in turn, its state in registers, its sums and
    evaluations sweeping its own dofs and rows through the island-ordered dof_id / constraint_id lists.

    Returns whether any island moved.
    """
    n_dofs = constraint_state.search.shape[0]
    n_islands = constraint_state.island.n_islands[i_b]
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    ncone = nef
    if qd.static(rigid_config.enable_elliptic_friction):
        ncone = ncone + constraint_state.n_constraints_cone[i_b]

    # mv = M @ search and jv = J @ search over the islands still moving, through the island lists. An env holding one
    # island reads every row densely, whose loads carry no dependent index.
    if qd.static(rigid_config.is_single_island):
        func_mv_jv_dense(i_b, constraint_state, rigid_info)
    else:
        if n_islands == 1:
            func_mv_jv_dense(i_b, constraint_state, rigid_info)
        else:
            func_mv_jv_islands(i_b, constraint_state, rigid_info)
    is_moved = False
    for i_island in range(n_islands):
        if constraint_state.island.improved[i_island, i_b]:
            dof_lo = constraint_state.island.dof_slices.start[i_island, i_b]
            dof_hi = dof_lo + constraint_state.island.dof_slices.n[i_island, i_b]
            if qd.static(rigid_config.is_single_island):
                dof_lo = 0
                dof_hi = constraint_state.search.shape[0]
            row_lo = constraint_state.island.constraint_slices.start[i_island, i_b]
            row_hi = row_lo + constraint_state.island.constraint_slices.n[i_island, i_b]
            dof_base = constraint_state.island.dof_range_start[i_island, i_b]
            row_base = func_list_range_start(constraint_state.island.constraint_id, row_lo, row_hi, i_b)
            sums = qd.Vector.zero(gs.qd_float, 8)
            for i_pos in range(dof_lo, dof_hi):
                s_sq, g_sq, gauss_1, gauss_2 = func_dof_p0_terms(
                    func_list_item(constraint_state.island.dof_id, i_pos, dof_lo, dof_base, i_b),
                    i_b,
                    dyn_state,
                    constraint_state,
                )
                sums[0] = sums[0] + s_sq
                sums[1] = sums[1] + g_sq
                sums[2] = sums[2] + gauss_1
                sums[3] = sums[3] + gauss_2
            for i_pos in range(row_lo, row_hi):
                eq_1, eq_2, tot_1, tot_2 = func_row_p0_terms(
                    func_list_item(constraint_state.island.constraint_id, i_pos, row_lo, row_base, i_b),
                    i_b,
                    ne,
                    nef,
                    ncone,
                    constraint_state,
                    rigid_config,
                )
                sums[4] = sums[4] + eq_1
                sums[5] = sums[5] + eq_2
                sums[6] = sums[6] + tot_1
                sums[7] = sums[7] + tot_2
            phase, ls_result, gtol, base_1, base_2, p0_deriv_0, p0_deriv_1, alpha_0 = func_ls_state_init(
                sums, constraint_state.island.inertia[i_island, i_b], rigid_info, rigid_config
            )
            n_alphas = 1
            alphas = qd.Vector.zero(gs.qd_float, 3)
            alphas[0] = alpha_0
            p1 = qd.Vector.zero(gs.qd_float, 4)
            p2 = qd.Vector.zero(gs.qd_float, 4)
            p2update = 0
            direction = 0
            ls_it = 1
            res_alpha = gs.qd_float(0.0)
            improvement = gs.qd_float(0.0)
            while phase < 3:
                acc = qd.Vector.zero(gs.qd_float, 9)
                for i_pos in range(row_lo, row_hi):
                    t_0, t_1, t_2 = func_row_alpha_terms(
                        func_list_item(constraint_state.island.constraint_id, i_pos, row_lo, row_base, i_b),
                        i_b,
                        n_alphas,
                        alphas,
                        ne,
                        nef,
                        ncone,
                        constraint_state,
                        rigid_config,
                    )
                    for k in qd.static(range(3)):
                        acc[3 * k] = acc[3 * k] + t_0[k]
                        acc[3 * k + 1] = acc[3 * k + 1] + t_1[k]
                        acc[3 * k + 2] = acc[3 * k + 2] + t_2[k]
                (
                    phase,
                    n_alphas,
                    alphas,
                    p1,
                    p2,
                    p2update,
                    direction,
                    ls_it,
                    res_alpha,
                    improvement,
                    ls_result,
                ) = func_ls_state_advance(
                    acc,
                    n_alphas,
                    alphas,
                    phase,
                    p1,
                    p2,
                    p2update,
                    direction,
                    ls_it,
                    gtol,
                    base_1,
                    base_2,
                    p0_deriv_0,
                    p0_deriv_1,
                    rigid_info,
                )
            # A null step converges the island; otherwise its dofs and rows take the step
            if qd.abs(res_alpha) < rigid_info.EPS[None]:
                res_alpha = 0.0
                constraint_state.island.improved[i_island, i_b] = False
            else:
                is_moved = True
                for i_pos in range(dof_lo, dof_hi):
                    i_d = func_list_item(constraint_state.island.dof_id, i_pos, dof_lo, dof_base, i_b)
                    constraint_state.qacc[i_d, i_b] = (
                        constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * res_alpha
                    )
                    constraint_state.Ma[i_d, i_b] = (
                        constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * res_alpha
                    )
                    if qd.static(rigid_config.solver_type == gs.constraint_solver.CG):
                        constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                        constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]
                for i_pos in range(row_lo, row_hi):
                    i_c = func_list_item(constraint_state.island.constraint_id, i_pos, row_lo, row_base, i_b)
                    constraint_state.Jaref[i_c, i_b] = (
                        constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * res_alpha
                    )
            constraint_state.island.ls_improvement[i_island, i_b] = improvement
    return is_moved


@qd.func
def func_exit_islands_serial(
    i_b,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    certify: qd.template(),
):
    """Convergence test, or under certify the warm-start certificate, of every awake island of one env, serially.

    Each island's exit terms are summed over its own dofs, then the search direction of the dofs of every island that
    iterates on is set.

    Returns whether any island does.
    """
    n_islands = constraint_state.island.n_islands[i_b]
    improved_any = False
    for i_island in range(n_islands):
        if constraint_state.island.improved[i_island, i_b]:
            dof_lo = constraint_state.island.dof_slices.start[i_island, i_b]
            dof_hi = dof_lo + constraint_state.island.dof_slices.n[i_island, i_b]
            if qd.static(rigid_config.is_single_island):
                dof_lo = 0
                dof_hi = constraint_state.search.shape[0]
            dof_base = constraint_state.island.dof_range_start[i_island, i_b]
            terms = qd.Vector.zero(gs.qd_float, 7)
            for i_pos in range(dof_lo, dof_hi):
                terms = terms + func_dof_exit_terms(
                    func_list_item(constraint_state.island.dof_id, i_pos, dof_lo, dof_base, i_b),
                    i_b,
                    constraint_state,
                    rigid_config,
                )
            inertia = constraint_state.island.inertia[i_island, i_b]
            improved = False
            cg_beta = gs.qd_float(0.0)
            if qd.static(certify):
                improved = not func_certify_decision(terms, inertia, rigid_info, rigid_config)
            else:
                improved, cg_beta = func_exit_decision(
                    terms, constraint_state.island.ls_improvement[i_island, i_b], inertia, rigid_info, rigid_config
                )
            constraint_state.island.improved[i_island, i_b] = improved
            if improved:
                improved_any = True
                if qd.static(not certify):
                    for i_pos in range(dof_lo, dof_hi):
                        i_d = func_list_item(constraint_state.island.dof_id, i_pos, dof_lo, dof_base, i_b)
                        if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton):
                            constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
                        else:
                            constraint_state.search[i_d, i_b] = (
                                -constraint_state.Mgrad[i_d, i_b] + cg_beta * constraint_state.search[i_d, i_b]
                            )
    return improved_any


# ======================================================================================================================
# ============================================ Decomposed arm (cooperative) ===========================================
# ======================================================================================================================


@qd.func
def func_segment_add(tid, i_slot, n_valid, value, i_slot_prev, i_slot_next, sh_acc, k):
    """Segmented sum of ``value`` over the lanes of a chunk sharing an island slot, the tail lane of each segment adding
    the segment's total into row ``k`` of the shared accumulator.

    ``i_slot_prev`` / ``i_slot_next`` are the slots of the neighboring lanes, -1 past the chunk's valid lanes.
    """
    _K = qd.static(32)
    is_head = 1
    if tid > 0 and i_slot_prev == i_slot:
        is_head = 0
    total = qd.simt.subgroup.segmented_reduce_add_tiled(value, is_head, 5)
    if tid < n_valid and i_slot >= 0 and i_slot_next != i_slot:
        sh_acc[k * _K + i_slot] = sh_acc[k * _K + i_slot] + total


@qd.func
def func_linesearch_islands_coop(
    i_b,
    tid,
    sh_acc,
    sh_alphas,
    sh_n_alphas,
    sh_pending,
    sh_alpha,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Lockstep line search of every island of one env by the _K lanes of its block, then the step applied.

    Islands are taken in groups of _K, lane tid owning island base + tid and holding its search state in registers
    between rounds, and a group's dofs and rows are contiguous ranges of the island-ordered dof_id / constraint_id
    lists. A group of few islands sums each island in turn, every lane striding over the island's own items and one
    plain subgroup reduction closing each sum. A group of many islands sweeps the group's range in chunks of _K items,
    every chunk reduced by island with a segmented subgroup sum whose segment tail is the single writer of the island's
    slot in the shared accumulator (func_segment_add), so the accumulation order is the chunk order. The rows read each
    island's pending flag and candidates from shared memory, the owner lane reads its sums back from it.

    Returns whether any island moved.
    """
    _K = qd.static(32)
    n_dofs = constraint_state.search.shape[0]
    n_islands = constraint_state.island.n_islands[i_b]
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    ncone = nef
    if qd.static(rigid_config.enable_elliptic_friction):
        ncone = ncone + constraint_state.n_constraints_cone[i_b]
    n_con = constraint_state.n_constraints[i_b]

    i_d1 = tid
    while i_d1 < n_dofs:
        mv = gs.qd_float(0.0)
        for i_d2 in range(rigid_info.dofs_mass_block_start[i_d1], rigid_info.dofs_mass_block_end[i_d1]):
            mv = mv + rigid_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
        constraint_state.mv[i_d1, i_b] = mv
        i_d1 = i_d1 + _K
    i_c = tid
    while i_c < n_con:
        jv = gs.qd_float(0.0)
        for i_jd in range(constraint_state.jac_n_dofs[i_c, i_b]):
            i_d = constraint_state.jac_dofs_idx[i_c, i_jd, i_b]
            jv = jv + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
        constraint_state.jv[i_c, i_b] = jv
        i_c = i_c + _K
    qd.simt.block.sync()

    is_moved = False
    for i_group in range((n_islands + _K - 1) // _K):
        base = i_group * _K
        n_group = qd.min(n_islands - base, _K)
        i_last = base + n_group - 1
        dof_lo = constraint_state.island.dof_slices.start[base, i_b]
        dof_hi = (
            constraint_state.island.dof_slices.start[i_last, i_b] + constraint_state.island.dof_slices.n[i_last, i_b]
        )
        row_lo = constraint_state.island.constraint_slices.start[base, i_b]
        row_hi = (
            constraint_state.island.constraint_slices.start[i_last, i_b]
            + constraint_state.island.constraint_slices.n[i_last, i_b]
        )
        dof_base = func_group_dof_range_start(i_b, tid, base, n_group, dof_lo, dof_hi, constraint_state)
        # The rows of several islands interleave in constraint order, so a sweep in that order would spread one
        # island's rows over several segments of a chunk, whose tails would all add into its slot at once (see
        # func_segment_add). A group of several islands sweeps its rows in list order, one segment per island per
        # chunk.
        row_base = -1
        if n_group == 1:
            row_base = func_list_range_start(constraint_state.island.constraint_id, row_lo, row_hi, i_b)
        # A group of few islands sums each island by every lane over the island's own items and one plain subgroup
        # reduction per sum, whose count follows the islands; a group of many small islands takes the segmented sums
        # per chunk of the group's lists (func_segment_add), whose count follows the rows swept.
        is_per_island = n_group <= 2 * ((row_hi - row_lo + _K - 1) // _K)
        i_island = base + tid
        is_owner = tid < n_group
        is_active = False
        if is_owner:
            is_active = constraint_state.island.improved[i_island, i_b]
        for k in qd.static(range(9)):
            sh_acc[k * _K + tid] = 0.0
        sh_pending[tid] = 0
        sh_alpha[tid] = 0.0
        qd.simt.block.sync()

        # Initialization sums over each island's dofs then rows (an island standing still contributes nothing)
        if is_per_island:
            for i_g in range(n_group):
                j_island = base + i_g
                if constraint_state.island.improved[j_island, i_b]:
                    sums_i = qd.Vector.zero(gs.qd_float, 8)
                    dof_lo_i = constraint_state.island.dof_slices.start[j_island, i_b]
                    dof_hi_i = dof_lo_i + constraint_state.island.dof_slices.n[j_island, i_b]
                    dof_base_i = constraint_state.island.dof_range_start[j_island, i_b]
                    i_pos = dof_lo_i + tid
                    while i_pos < dof_hi_i:
                        i_d = func_list_item(constraint_state.island.dof_id, i_pos, dof_lo_i, dof_base_i, i_b)
                        s_sq, g_sq, gauss_1, gauss_2 = func_dof_p0_terms(i_d, i_b, dyn_state, constraint_state)
                        sums_i[0] = sums_i[0] + s_sq
                        sums_i[1] = sums_i[1] + g_sq
                        sums_i[2] = sums_i[2] + gauss_1
                        sums_i[3] = sums_i[3] + gauss_2
                        i_pos = i_pos + _K
                    row_lo_i = constraint_state.island.constraint_slices.start[j_island, i_b]
                    row_hi_i = row_lo_i + constraint_state.island.constraint_slices.n[j_island, i_b]
                    row_base_i = func_list_range_start(constraint_state.island.constraint_id, row_lo_i, row_hi_i, i_b)
                    i_pos = row_lo_i + tid
                    while i_pos < row_hi_i:
                        i_c = func_list_item(constraint_state.island.constraint_id, i_pos, row_lo_i, row_base_i, i_b)
                        eq_1, eq_2, tot_1, tot_2 = func_row_p0_terms(
                            i_c, i_b, ne, nef, ncone, constraint_state, rigid_config
                        )
                        sums_i[4] = sums_i[4] + eq_1
                        sums_i[5] = sums_i[5] + eq_2
                        sums_i[6] = sums_i[6] + tot_1
                        sums_i[7] = sums_i[7] + tot_2
                        i_pos = i_pos + _K
                    for k in qd.static(range(8)):
                        total = qd.simt.subgroup.reduce_all_add_tiled(sums_i[k], 5)
                        if tid == i_g:
                            sh_acc[k * _K + tid] = total
            qd.simt.block.sync()
        else:
            for i_chunk in range((dof_hi - dof_lo + _K - 1) // _K):
                i_pos = dof_lo + i_chunk * _K + tid
                n_valid = qd.min(dof_hi - dof_lo - i_chunk * _K, _K)
                i_slot = -1
                s_sq = gs.qd_float(0.0)
                g_sq = gs.qd_float(0.0)
                gauss_1 = gs.qd_float(0.0)
                gauss_2 = gs.qd_float(0.0)
                if i_pos < dof_hi:
                    i_d = func_list_item(constraint_state.island.dof_id, i_pos, dof_lo, dof_base, i_b)
                    i_slot = 0
                    if n_group > 1:
                        i_slot = constraint_state.island.dofs_island_idx[i_d, i_b] - base
                    if constraint_state.island.improved[base + i_slot, i_b]:
                        s_sq, g_sq, gauss_1, gauss_2 = func_dof_p0_terms(i_d, i_b, dyn_state, constraint_state)
                i_slot_prev = qd.simt.subgroup.shuffle_up(i_slot, qd.u32(1))
                i_slot_next = qd.simt.subgroup.shuffle_down(i_slot, qd.u32(1))
                if tid == _K - 1:
                    i_slot_next = -1
                func_segment_add(tid, i_slot, n_valid, s_sq, i_slot_prev, i_slot_next, sh_acc, 0)
                func_segment_add(tid, i_slot, n_valid, g_sq, i_slot_prev, i_slot_next, sh_acc, 1)
                func_segment_add(tid, i_slot, n_valid, gauss_1, i_slot_prev, i_slot_next, sh_acc, 2)
                func_segment_add(tid, i_slot, n_valid, gauss_2, i_slot_prev, i_slot_next, sh_acc, 3)
                qd.simt.block.sync()
            for i_chunk in range((row_hi - row_lo + _K - 1) // _K):
                i_pos = row_lo + i_chunk * _K + tid
                n_valid = qd.min(row_hi - row_lo - i_chunk * _K, _K)
                i_slot = -1
                eq_1 = gs.qd_float(0.0)
                eq_2 = gs.qd_float(0.0)
                tot_1 = gs.qd_float(0.0)
                tot_2 = gs.qd_float(0.0)
                if i_pos < row_hi:
                    i_c = func_list_item(constraint_state.island.constraint_id, i_pos, row_lo, row_base, i_b)
                    i_slot = 0
                    if n_group > 1:
                        i_slot = constraint_state.island.constraint_island_idx[i_c, i_b] - base
                    if constraint_state.island.improved[base + i_slot, i_b]:
                        eq_1, eq_2, tot_1, tot_2 = func_row_p0_terms(
                            i_c, i_b, ne, nef, ncone, constraint_state, rigid_config
                        )
                i_slot_prev = qd.simt.subgroup.shuffle_up(i_slot, qd.u32(1))
                i_slot_next = qd.simt.subgroup.shuffle_down(i_slot, qd.u32(1))
                if tid == _K - 1:
                    i_slot_next = -1
                func_segment_add(tid, i_slot, n_valid, eq_1, i_slot_prev, i_slot_next, sh_acc, 4)
                func_segment_add(tid, i_slot, n_valid, eq_2, i_slot_prev, i_slot_next, sh_acc, 5)
                func_segment_add(tid, i_slot, n_valid, tot_1, i_slot_prev, i_slot_next, sh_acc, 6)
                func_segment_add(tid, i_slot, n_valid, tot_2, i_slot_prev, i_slot_next, sh_acc, 7)
                qd.simt.block.sync()

        # The owner lane closes the initialization of its island and keeps the search state in registers
        phase = -1
        ls_result = 0
        gtol = gs.qd_float(0.0)
        base_1 = gs.qd_float(0.0)
        base_2 = gs.qd_float(0.0)
        p0_deriv_0 = gs.qd_float(0.0)
        p0_deriv_1 = gs.qd_float(0.0)
        n_alphas = 0
        alphas = qd.Vector.zero(gs.qd_float, 3)
        p1 = qd.Vector.zero(gs.qd_float, 4)
        p2 = qd.Vector.zero(gs.qd_float, 4)
        p2update = 0
        direction = 0
        ls_it = 1
        res_alpha = gs.qd_float(0.0)
        improvement = gs.qd_float(0.0)
        if is_active:
            sums = qd.Vector.zero(gs.qd_float, 8)
            for k in qd.static(range(8)):
                sums[k] = sh_acc[k * _K + tid]
            phase, ls_result, gtol, base_1, base_2, p0_deriv_0, p0_deriv_1, alpha_0 = func_ls_state_init(
                sums, constraint_state.island.inertia[i_island, i_b], rigid_info, rigid_config
            )
            alphas[0] = alpha_0
            n_alphas = 1
        qd.simt.block.sync()
        is_pending = phase >= 0 and phase < 3
        sh_pending[tid] = is_pending
        sh_n_alphas[tid] = n_alphas
        for k in qd.static(range(3)):
            sh_alphas[k * _K + tid] = alphas[k]
        for k in qd.static(range(9)):
            sh_acc[k * _K + tid] = 0.0
        qd.simt.block.sync()

        # Lockstep rounds: one sweep of the group's rows evaluates every pending island's candidates
        while qd.simt.subgroup.any_true(is_pending) != 0:
            if is_per_island:
                for i_g in range(n_group):
                    if sh_pending[i_g] != 0:
                        j_island = base + i_g
                        acc_i = qd.Vector.zero(gs.qd_float, 9)
                        row_alphas = qd.Vector([sh_alphas[i_g], sh_alphas[_K + i_g], sh_alphas[2 * _K + i_g]])
                        n_alphas_i = sh_n_alphas[i_g]
                        row_lo_i = constraint_state.island.constraint_slices.start[j_island, i_b]
                        row_hi_i = row_lo_i + constraint_state.island.constraint_slices.n[j_island, i_b]
                        row_base_i = func_list_range_start(
                            constraint_state.island.constraint_id, row_lo_i, row_hi_i, i_b
                        )
                        i_pos = row_lo_i + tid
                        while i_pos < row_hi_i:
                            i_c = func_list_item(
                                constraint_state.island.constraint_id, i_pos, row_lo_i, row_base_i, i_b
                            )
                            t_0, t_1, t_2 = func_row_alpha_terms(
                                i_c, i_b, n_alphas_i, row_alphas, ne, nef, ncone, constraint_state, rigid_config
                            )
                            for k in qd.static(range(3)):
                                acc_i[3 * k] = acc_i[3 * k] + t_0[k]
                                acc_i[3 * k + 1] = acc_i[3 * k + 1] + t_1[k]
                                acc_i[3 * k + 2] = acc_i[3 * k + 2] + t_2[k]
                            i_pos = i_pos + _K
                        for k in qd.static(range(9)):
                            total = qd.simt.subgroup.reduce_all_add_tiled(acc_i[k], 5)
                            if tid == i_g:
                                sh_acc[k * _K + tid] = total
                qd.simt.block.sync()
            else:
                for i_chunk in range((row_hi - row_lo + _K - 1) // _K):
                    i_pos = row_lo + i_chunk * _K + tid
                    n_valid = qd.min(row_hi - row_lo - i_chunk * _K, _K)
                    i_slot = -1
                    t_0 = qd.Vector.zero(gs.qd_float, 3)
                    t_1 = qd.Vector.zero(gs.qd_float, 3)
                    t_2 = qd.Vector.zero(gs.qd_float, 3)
                    if i_pos < row_hi:
                        i_c = func_list_item(constraint_state.island.constraint_id, i_pos, row_lo, row_base, i_b)
                        i_slot = 0
                        if n_group > 1:
                            i_slot = constraint_state.island.constraint_island_idx[i_c, i_b] - base
                        if sh_pending[i_slot] != 0:
                            row_alphas = qd.Vector(
                                [sh_alphas[i_slot], sh_alphas[_K + i_slot], sh_alphas[2 * _K + i_slot]]
                            )
                            t_0, t_1, t_2 = func_row_alpha_terms(
                                i_c,
                                i_b,
                                sh_n_alphas[i_slot],
                                row_alphas,
                                ne,
                                nef,
                                ncone,
                                constraint_state,
                                rigid_config,
                            )
                    i_slot_prev = qd.simt.subgroup.shuffle_up(i_slot, qd.u32(1))
                    i_slot_next = qd.simt.subgroup.shuffle_down(i_slot, qd.u32(1))
                    if tid == _K - 1:
                        i_slot_next = -1
                    for k in qd.static(range(3)):
                        func_segment_add(tid, i_slot, n_valid, t_0[k], i_slot_prev, i_slot_next, sh_acc, 3 * k)
                        func_segment_add(tid, i_slot, n_valid, t_1[k], i_slot_prev, i_slot_next, sh_acc, 3 * k + 1)
                        func_segment_add(tid, i_slot, n_valid, t_2[k], i_slot_prev, i_slot_next, sh_acc, 3 * k + 2)
                    qd.simt.block.sync()
            if is_pending:
                acc = qd.Vector.zero(gs.qd_float, 9)
                for k in qd.static(range(9)):
                    acc[k] = sh_acc[k * _K + tid]
                (
                    phase,
                    n_alphas,
                    alphas,
                    p1,
                    p2,
                    p2update,
                    direction,
                    ls_it,
                    res_alpha,
                    improvement,
                    ls_result,
                ) = func_ls_state_advance(
                    acc,
                    n_alphas,
                    alphas,
                    phase,
                    p1,
                    p2,
                    p2update,
                    direction,
                    ls_it,
                    gtol,
                    base_1,
                    base_2,
                    p0_deriv_0,
                    p0_deriv_1,
                    rigid_info,
                )
            qd.simt.block.sync()
            is_pending = phase >= 0 and phase < 3
            sh_pending[tid] = is_pending
            sh_n_alphas[tid] = n_alphas
            for k in qd.static(range(3)):
                sh_alphas[k * _K + tid] = alphas[k]
            for k in qd.static(range(9)):
                sh_acc[k * _K + tid] = 0.0
            qd.simt.block.sync()

        # The owner records its island's search, a null step converging the island; then the group's step is applied
        if phase == 3:
            if qd.abs(res_alpha) < rigid_info.EPS[None]:
                res_alpha = 0.0
                constraint_state.island.improved[i_island, i_b] = False
            constraint_state.island.ls_improvement[i_island, i_b] = improvement
        sh_alpha[tid] = res_alpha
        qd.simt.block.sync()
        if qd.simt.subgroup.any_true(res_alpha != 0.0) != 0:
            is_moved = True
            i_pos = dof_lo + tid
            while i_pos < dof_hi:
                i_d = func_list_item(constraint_state.island.dof_id, i_pos, dof_lo, dof_base, i_b)
                i_slot = 0
                if n_group > 1:
                    i_slot = constraint_state.island.dofs_island_idx[i_d, i_b] - base
                alpha = sh_alpha[i_slot]
                constraint_state.qacc[i_d, i_b] = (
                    constraint_state.qacc[i_d, i_b] + constraint_state.search[i_d, i_b] * alpha
                )
                constraint_state.Ma[i_d, i_b] = constraint_state.Ma[i_d, i_b] + constraint_state.mv[i_d, i_b] * alpha
                if qd.static(rigid_config.solver_type == gs.constraint_solver.CG):
                    constraint_state.cg_prev_grad[i_d, i_b] = constraint_state.grad[i_d, i_b]
                    constraint_state.cg_prev_Mgrad[i_d, i_b] = constraint_state.Mgrad[i_d, i_b]
                i_pos = i_pos + _K
            i_pos = row_lo + tid
            while i_pos < row_hi:
                i_c = func_list_item(constraint_state.island.constraint_id, i_pos, row_lo, row_base, i_b)
                i_slot = 0
                if n_group > 1:
                    i_slot = constraint_state.island.constraint_island_idx[i_c, i_b] - base
                alpha = sh_alpha[i_slot]
                constraint_state.Jaref[i_c, i_b] = (
                    constraint_state.Jaref[i_c, i_b] + constraint_state.jv[i_c, i_b] * alpha
                )
                i_pos = i_pos + _K
        qd.simt.block.sync()
    return is_moved


@qd.func
def func_exit_islands_coop(
    i_b,
    tid,
    sh_acc,
    sh_pending,
    sh_alpha,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    certify: qd.template(),
):
    """Convergence test, or under certify the warm-start certificate, of every island of one env by its block.

    Islands come in groups of _K as in func_linesearch_islands_coop: the exit terms are reduced per island over the
    group's dofs into sh_acc, the owner lane decides, then the search direction of every dof follows its island, whose
    decision and conjugate coefficient it reads from sh_pending and sh_alpha.

    Returns whether any island iterates on.
    """
    _K = qd.static(32)
    n_islands = constraint_state.island.n_islands[i_b]
    improved_any = False
    for i_group in range((n_islands + _K - 1) // _K):
        base = i_group * _K
        n_group = qd.min(n_islands - base, _K)
        i_last = base + n_group - 1
        dof_lo = constraint_state.island.dof_slices.start[base, i_b]
        dof_hi = (
            constraint_state.island.dof_slices.start[i_last, i_b] + constraint_state.island.dof_slices.n[i_last, i_b]
        )
        dof_base = func_group_dof_range_start(i_b, tid, base, n_group, dof_lo, dof_hi, constraint_state)
        i_island = base + tid
        is_active = False
        if tid < n_group:
            is_active = constraint_state.island.improved[i_island, i_b]
        for k in qd.static(range(7)):
            sh_acc[k * _K + tid] = 0.0
        qd.simt.block.sync()
        for i_chunk in range((dof_hi - dof_lo + _K - 1) // _K):
            i_pos = dof_lo + i_chunk * _K + tid
            n_valid = qd.min(dof_hi - dof_lo - i_chunk * _K, _K)
            i_slot = -1
            terms = qd.Vector.zero(gs.qd_float, 7)
            if i_pos < dof_hi:
                i_d = func_list_item(constraint_state.island.dof_id, i_pos, dof_lo, dof_base, i_b)
                i_slot = 0
                if n_group > 1:
                    i_slot = constraint_state.island.dofs_island_idx[i_d, i_b] - base
                if constraint_state.island.improved[base + i_slot, i_b]:
                    terms = func_dof_exit_terms(i_d, i_b, constraint_state, rigid_config)
            i_slot_prev = qd.simt.subgroup.shuffle_up(i_slot, qd.u32(1))
            i_slot_next = qd.simt.subgroup.shuffle_down(i_slot, qd.u32(1))
            if tid == _K - 1:
                i_slot_next = -1
            for k in qd.static(range(7)):
                func_segment_add(tid, i_slot, n_valid, terms[k], i_slot_prev, i_slot_next, sh_acc, k)
            qd.simt.block.sync()
        improved = False
        cg_beta = gs.qd_float(0.0)
        if is_active:
            terms = qd.Vector.zero(gs.qd_float, 7)
            for k in qd.static(range(7)):
                terms[k] = sh_acc[k * _K + tid]
            inertia = constraint_state.island.inertia[i_island, i_b]
            if qd.static(certify):
                improved = not func_certify_decision(terms, inertia, rigid_info, rigid_config)
            else:
                improved, cg_beta = func_exit_decision(
                    terms, constraint_state.island.ls_improvement[i_island, i_b], inertia, rigid_info, rigid_config
                )
            constraint_state.island.improved[i_island, i_b] = improved
        qd.simt.block.sync()
        sh_pending[tid] = improved
        sh_alpha[tid] = cg_beta
        qd.simt.block.sync()
        if qd.simt.subgroup.any_true(improved) != 0:
            improved_any = True
            if qd.static(not certify):
                i_pos = dof_lo + tid
                while i_pos < dof_hi:
                    i_d = func_list_item(constraint_state.island.dof_id, i_pos, dof_lo, dof_base, i_b)
                    i_slot = 0
                    if n_group > 1:
                        i_slot = constraint_state.island.dofs_island_idx[i_d, i_b] - base
                    if sh_pending[i_slot] != 0:
                        if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton):
                            constraint_state.search[i_d, i_b] = -constraint_state.Mgrad[i_d, i_b]
                        else:
                            constraint_state.search[i_d, i_b] = (
                                -constraint_state.Mgrad[i_d, i_b] + sh_alpha[i_slot] * constraint_state.search[i_d, i_b]
                            )
                    i_pos = i_pos + _K
        qd.simt.block.sync()
    return improved_any
