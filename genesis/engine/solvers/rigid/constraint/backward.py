import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from . import solver


@qd.func
def func_matvec_Ap(
    i_b,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """
    Compute Ap = (M + J^T * diag(D) * J) * p on the current active set, which is used for solving the adjoint u.

    Specifically, M = mass matrix, J = Jacobian, diag(D) = diagonal matrix of efc_D, and p = search direction.
    """
    n_dofs = constraint_state.bw_p.shape[0]
    for i_d in range(n_dofs):
        constraint_state.bw_Ap[i_d, i_b] = 0.0

    # Mp: Block multiplication
    n_entities = dyn_info.entities.n_links.shape[0]
    for i_e in range(n_entities):
        for i_d1 in range(dyn_info.entities.dof_start[i_e], dyn_info.entities.dof_end[i_e]):
            acc = gs.qd_float(0.0)
            for i_d2 in range(dyn_info.entities.dof_start[i_e], dyn_info.entities.dof_end[i_e]):
                acc += rigid_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.bw_p[i_d2, i_b]
            constraint_state.bw_Ap[i_d1, i_b] += acc

    # tmp = J v
    for i_c in range(constraint_state.n_constraints[i_b]):
        jv = gs.qd_float(0.0)
        if qd.static(rigid_config.sparse_solve):
            for k in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, k, i_b]
                jv += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_p[i_d, i_b]
        else:
            for i_d in range(n_dofs):
                jv += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_p[i_d, i_b]
        # only active constraints contribute
        jv *= constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
        # out += J^T (D * J v)
        if qd.static(rigid_config.sparse_solve):
            for k in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, k, i_b]
                constraint_state.bw_Ap[i_d, i_b] += constraint_state.jac[i_c, i_d, i_b] * jv
        else:
            for i_d in range(n_dofs):
                constraint_state.bw_Ap[i_d, i_b] += constraint_state.jac[i_c, i_d, i_b] * jv


@qd.func
def func_solve_adjoint_u_cg_batch(
    i_b,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Conjugate-gradient (CG) solve of A u = g for a single environment [i_b].

    A = M + J^T diag(D) J is applied implicitly by func_matvec_Ap, which reads rigid_info.mass_mat directly and loops
    only over the active constraints, so this also solves the unconstrained case A = M (empty J term).
    """
    n_dofs = constraint_state.bw_u.shape[0]

    # r = g - A*0 = g ; p = r ; u = 0
    num = gs.qd_float(0.0)
    for i_d in range(n_dofs):
        constraint_state.bw_u[i_d, i_b] = 0.0
        constraint_state.bw_r[i_d, i_b] = constraint_state.dL_dqacc[i_d, i_b]
        constraint_state.bw_p[i_d, i_b] = constraint_state.bw_r[i_d, i_b]
        num += constraint_state.bw_r[i_d, i_b] * constraint_state.bw_r[i_d, i_b]

    # The stopping target is relative to the seed |g|^2: an absolute threshold either exits at a huge relative
    # residual when g is small (each backward substep shrinks the upstream gradient by roughly the loss scale) or,
    # past convergence, lets the clamped alpha / beta denominators inject garbage steps that corrupt u. The
    # denominator break exits once p collapses to the round-off floor, where p^T A p underflows for a positive
    # semi-definite (PSD) A; alpha and beta then never need clamping.
    num_target = num * rigid_info.EPS[None] * rigid_info.EPS[None]
    for _ in range(rigid_info.iterations[None]):
        if num <= num_target:
            break
        func_matvec_Ap(i_b, constraint_state, dyn_info, rigid_info, rigid_config)

        # alpha = (r,r)/(p,Ap)
        den = gs.qd_float(0.0)
        for i_d in range(n_dofs):
            den += constraint_state.bw_p[i_d, i_b] * constraint_state.bw_Ap[i_d, i_b]
        if den <= 0.0:
            break
        alpha = num / den

        # u += alpha p ; r -= alpha Ap
        for i_d in range(n_dofs):
            constraint_state.bw_u[i_d, i_b] += alpha * constraint_state.bw_p[i_d, i_b]
            constraint_state.bw_r[i_d, i_b] -= alpha * constraint_state.bw_Ap[i_d, i_b]

        # beta = (r_new,r_new)/(r_old,r_old) ; p = r + beta p
        num_new = gs.qd_float(0.0)
        for i_d in range(n_dofs):
            num_new += constraint_state.bw_r[i_d, i_b] * constraint_state.bw_r[i_d, i_b]
        beta = num_new / num
        num = num_new
        for i_d in range(n_dofs):
            constraint_state.bw_p[i_d, i_b] = constraint_state.bw_r[i_d, i_b] + beta * constraint_state.bw_p[i_d, i_b]


@qd.kernel
def kernel_solve_adjoint_u(
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    r"""
    Solve for the adjoint vector [u] from Au = g, where A = dF/dqacc (primal Hessian on the active set) and g = dL/dqacc.
    Intuitively, [u] is a sensitivity vector that translates the upstream gradient dL/dqacc into the primal space.
    This adjoint vector [u] can be used an intermediate variable to compute the downstream gradients. Since A is a
    Semi-Positive Definite (SPD) matrix, we can solve A * u = g using either Cholesky decomposition or CG solver.
    When Newton solver was used, we reuse the Cholesky decomposition of A (= L * L^T) to solve A * u = g. Otherwise,
    we use CG solver.

    Specifically, A = M + J^T * diag(D) * J, where M = mass matrix, J = Jacobian, diag(D) = diagonal matrix of efc_D.
    """
    n_dofs = constraint_state.bw_u.shape[0]
    _B = constraint_state.bw_u.shape[1]

    # Initialize u
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        constraint_state.bw_u[i_d, i_b] = 0.0

    if qd.static(rigid_config.solver_type == gs.constraint_solver.Newton):
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] == 0:
                # No active constraint: A = M. The forward's constrained-Hessian Cholesky nt_H is unreliable for
                # these envs (the GPU tiled factorization skips them), so solve M u = g via CG, which reads mass_mat
                # directly and never touches nt_H.
                func_solve_adjoint_u_cg_batch(i_b, constraint_state, dyn_info, rigid_info, rigid_config)
            else:
                # Reuse the forward's Cholesky decomposition A = L * L^T to solve A u = g. The stored L factors the
                # scaled Hessian, so the right-hand side is pre-scaled and u unscaled at the end (see nt_jacobi in
                # array_class.py); the intermediate z equals the unscaled L^-1 g exactly.
                # z = L^{-1} g  (forward substitution); saved to bw_r
                for i_d in range(n_dofs):
                    z = constraint_state.dL_dqacc[i_d, i_b]
                    if qd.static(rigid_config.enable_jacobi_equilibration):
                        z = z * constraint_state.nt_jacobi[i_d, i_b]
                    for j_d in range(i_d):
                        z -= constraint_state.nt_H[i_b, i_d, j_d] * constraint_state.bw_r[j_d, i_b]
                    z /= constraint_state.nt_H[i_b, i_d, i_d]
                    constraint_state.bw_r[i_d, i_b] = z

                # u = L^{-T} z  (back substitution)
                for i_d_ in range(n_dofs):
                    i_d = n_dofs - 1 - i_d_
                    u = constraint_state.bw_r[i_d, i_b]
                    for j_d in range(i_d + 1, n_dofs):
                        u -= constraint_state.nt_H[i_b, j_d, i_d] * constraint_state.bw_u[j_d, i_b]
                    u /= constraint_state.nt_H[i_b, i_d, i_d]
                    constraint_state.bw_u[i_d, i_b] = u
                if qd.static(rigid_config.enable_jacobi_equilibration):
                    for i_d in range(n_dofs):
                        constraint_state.bw_u[i_d, i_b] = (
                            constraint_state.bw_u[i_d, i_b] * constraint_state.nt_jacobi[i_d, i_b]
                        )
    else:
        # CG solver for A * u = g (parallelized over the batch dimension).
        for i_b in range(_B):
            func_solve_adjoint_u_cg_batch(i_b, constraint_state, dyn_info, rigid_info, rigid_config)


@qd.kernel
def kernel_compute_gradients(
    constraint_state: array_class.ConstraintState, dyn_info: array_class.DynInfo, rigid_config: qd.template()
):
    r"""
    Compute gradients of the loss with respect to the input variables to this solver. Note that we use the intermediate
    adjoint vector [u] computed in [kernel_solve_adjoint_u] to compute these gradients.

    Specifically, the gradients are computed as follows:
    - dL_dM = -u * qacc^T
    - dL_djac = -[u * y^T + qacc * (D \odot (Ju))^T] (y = D \odot w, w = (Jqacc - aref))
    - dL_daref = Ju \odot D
    - dL_defc_D = -Ju \odot (Jqacc - aref)
    - dL_dforce = u
    """
    _B = constraint_state.bw_u.shape[1]
    n_dofs = constraint_state.bw_u.shape[0]
    n_constraints = constraint_state.bw_Ju.shape[0]

    # clear grads
    for i_d0, i_d1, i_b in qd.ndrange(n_dofs, n_dofs, _B):
        constraint_state.dL_dM[i_d0, i_d1, i_b] = gs.qd_float(0.0)
    for i_c, i_d, i_b in qd.ndrange(n_constraints, n_dofs, _B):
        constraint_state.dL_djac[i_c, i_d, i_b] = gs.qd_float(0.0)
    for i_c, i_b in qd.ndrange(n_constraints, _B):
        constraint_state.dL_daref[i_c, i_b] = gs.qd_float(0.0)
        constraint_state.dL_defc_D[i_c, i_b] = gs.qd_float(0.0)
        constraint_state.bw_Ju[i_c, i_b] = gs.qd_float(0.0)
        constraint_state.bw_y[i_c, i_b] = gs.qd_float(0.0)
        constraint_state.bw_w[i_c, i_b] = gs.qd_float(0.0)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        constraint_state.dL_dforce[i_d, i_b] = gs.qd_float(0.0)

    # Ju, w, y
    for i_b in range(_B):
        # Ju
        for i_c in range(constraint_state.n_constraints[i_b]):
            s = gs.qd_float(0.0)
            if qd.static(rigid_config.sparse_solve):
                for k in range(constraint_state.jac_n_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_dofs_idx[i_c, k, i_b]
                    s += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_u[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    s += constraint_state.jac[i_c, i_d, i_b] * constraint_state.bw_u[i_d, i_b]
            constraint_state.bw_Ju[i_c, i_b] = s

        # w = J qacc - aref
        # y = D \odot w
        for i_c in range(constraint_state.n_constraints[i_b]):
            t = gs.qd_float(0.0)
            if qd.static(rigid_config.sparse_solve):
                for k in range(constraint_state.jac_n_dofs[i_c, i_b]):
                    i_d = constraint_state.jac_dofs_idx[i_c, k, i_b]
                    t += constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            else:
                for i_d in range(n_dofs):
                    t += constraint_state.jac[i_c, i_d, i_b] * constraint_state.qacc[i_d, i_b]
            constraint_state.bw_w[i_c, i_b] = t - constraint_state.aref[i_c, i_b]
            constraint_state.bw_y[i_c, i_b] = constraint_state.efc_D[i_c, i_b] * constraint_state.bw_w[i_c, i_b]

        # grads
        # force: u
        for i_d in range(n_dofs):
            constraint_state.dL_dforce[i_d, i_b] += constraint_state.bw_u[i_d, i_b]

        # per-constraint (mask by active)
        # aref: Ju \odot D
        # D: -Ju \odot w
        # J: -[u * y^T + qacc * (D \odot (Ju)^T)]
        for i_c in range(constraint_state.n_constraints[i_b]):
            if constraint_state.active[i_c, i_b] != 0:
                # aref: Ju \odot D
                constraint_state.dL_daref[i_c, i_b] += (
                    constraint_state.efc_D[i_c, i_b] * constraint_state.bw_Ju[i_c, i_b]
                )
                # D: -Ju \odot w
                constraint_state.dL_defc_D[i_c, i_b] -= (
                    constraint_state.bw_Ju[i_c, i_b] * constraint_state.bw_w[i_c, i_b]
                )

                # J: -[u * y^T + qacc * (D \odot (Ju))^T]
                DJu_i = constraint_state.efc_D[i_c, i_b] * constraint_state.bw_Ju[i_c, i_b]
                y_i = constraint_state.bw_y[i_c, i_b]

                if qd.static(rigid_config.sparse_solve):
                    for k in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, k, i_b]
                        constraint_state.dL_djac[i_c, i_d, i_b] += -(
                            constraint_state.bw_u[i_d, i_b] * y_i + constraint_state.qacc[i_d, i_b] * DJu_i
                        )
                else:
                    for i_d in range(n_dofs):
                        constraint_state.dL_djac[i_c, i_d, i_b] += -(
                            constraint_state.bw_u[i_d, i_b] * y_i + constraint_state.qacc[i_d, i_b] * DJu_i
                        )

        # M: -u * qacc^T
        n_entities = dyn_info.entities.n_links.shape[0]
        for i_e in range(n_entities):
            s = dyn_info.entities.dof_start[i_e]
            e = dyn_info.entities.dof_end[i_e]
            for i in range(s, e):
                for j in range(s, e):
                    val0 = -constraint_state.bw_u[i, i_b] * constraint_state.qacc[j, i_b]
                    val1 = -constraint_state.bw_u[j, i_b] * constraint_state.qacc[i, i_b]
                    constraint_state.dL_dM[i, j, i_b] += (val0 + val1) * 0.5  # symmetrize


@qd.kernel(fastcache=True)
def kernel_load_dL_dqacc_from_acc_grad(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
):
    """Copy the acc grad into constraint_state.dL_dqacc (the input buffer consumed by kernel_solve_adjoint_u) and
    zero the source grad so the downstream implicit-function-theorem path does not re-consume it.
    """
    _B = dyn_state.dofs.acc.shape[1]
    n_dofs = dyn_state.dofs.acc.shape[0]
    qd.loop_config(
        name="kernel_load_dL_dqacc_from_acc_grad",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        constraint_state.dL_dqacc[i_d, i_b] = dyn_state.dofs.acc.grad[i_d, i_b]
        dyn_state.dofs.acc.grad[i_d, i_b] = gs.qd_float(0.0)


@qd.kernel(fastcache=True)
def kernel_accumulate_constraint_solver_grads(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Fold the constraint-solver adjoint outputs into the autodiff grad fields:
    dyn_state.dofs.force.grad += constraint_state.dL_dforce
    rigid_info.mass_mat.grad  += constraint_state.dL_dM
    """
    _B = dyn_state.dofs.force.shape[1]
    n_dofs = dyn_state.dofs.force.shape[0]
    qd.loop_config(
        name="kernel_accumulate_constraint_solver_grads",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dyn_state.dofs.force.grad[i_d, i_b] += constraint_state.dL_dforce[i_d, i_b]
    for i, j, i_b in qd.ndrange(n_dofs, n_dofs, _B):
        rigid_info.mass_mat.grad[i, j, i_b] += constraint_state.dL_dM[i, j, i_b]


# ---------------------------------------------------------------------------
# Manual reverses of the constraint-row assembly (equality, frictionloss, collision, joint-limit). Shared
# conventions for the kernels below.
#
# These reverses are manual because the constraint rows are built inside the forward solver with a data-dependent
# count and ordering - n_con is assigned by atomic_add as active constraints are discovered - and the row index is
# a runtime quantity outside what autograd can tape.
#
# Upstream grads: kernel_compute_gradients populates, per constraint row n_con, constraint_state.dL_daref[n_con]
# (dL/d aref), dL_defc_D[n_con] (dL/d efc_D), and dL_djac[n_con, i_d] (dL/d jac). The collision and equality
# reverses use dL_djac; the frictionloss and joint-limit reverses ignore it (their jac entries are the constants
# 1.0 and piecewise +-1 respectively, so the sub-gradient w.r.t. jac is 0). Each kernel consumes these and
# accumulates into its own differentiable inputs.
#
# n_con row layout: the forward adds constraints in the order equality -> frictionloss -> collision -> joint-limit
# (see add_equality_constraints / add_inequality_constraints in solver.py). All three equality sub-types (JOINT,
# CONNECT, WELD) are differentiated by kernel_manual_add_equality_constraints_bw. The manual reverses re-walk the
# same forward loops deterministically to recover their own n_con:
#   n_eq   = constraint_state.n_constraints_equality[i_b]
#   n_fric = constraint_state.n_constraints_frictionloss[i_b]
#   equality     : seed counter at 0
#   frictionloss : seed counter at n_eq
#   collision    : n_con = n_eq + n_fric + i_col_ * 4 + i, with i_col_ the logical (sorted) contact index
#   joint-limit  : seed counter at n_eq + n_fric (+ 4 * n_contacts if collision on)
# ---------------------------------------------------------------------------
@qd.kernel(fastcache=True)
def kernel_manual_add_joint_limit_constraints_bw(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    enable_collision: qd.template(),
):
    """Manual reverse of `add_joint_limit_constraints`. See the section header
    above for the shared `n_con` layout and upstream-grad conventions.

    Accumulates into rigid_info.qpos.grad[i_q] and dyn_state.dofs.vel.grad[i_d].

    Chain rule (per active joint, `pos_delta < 0`):

        Forward:
            pos_delta_min = qpos[i_q] - limit_lo
            pos_delta_max = limit_hi - qpos[i_q]
            pos_delta     = min(pos_delta_min, pos_delta_max)
            sign          = +1 if pos_delta_min < pos_delta_max else -1
            jac_qvel      = sign * dofs_vel[i_d]
            imp, aref     = gu.imp_aref(sol_params, pos_delta, jac_qvel, pos_delta)
            diag_raw      = invweight * (1 - imp) / imp
            diag          = max(diag_raw, EPS)
            efc_D         = 1 / diag

        d(pos_delta) / d(qpos) = sign      (chosen branch of `min`)
        d(jac_qvel) / d(vel)   = sign

        dL/d(imp) = g_aref * d(aref)/d(imp) + g_efc_D * d(efc_D)/d(imp)
                  g_aref = dL_daref[n_con],  g_efc_D = dL_defc_D[n_con]

        dL/d(pos_delta) = g_aref * d(aref)/d(pos_delta)|_direct
                        + dL/d(imp) * d(imp)/d(imp_x) * d(imp_x)/d(pos_delta)

        dL/d(jac_qvel)  = g_aref * d(aref)/d(jac_qvel) = -g_aref * b_coef

        dL/d(qpos)      += sign * dL/d(pos_delta)
        dL/d(vel)       += sign * dL/d(jac_qvel)
    """
    EPS = rigid_info.EPS[None]
    _B = constraint_state.jac.shape[2]
    n_links = dyn_info.links.root_idx.shape[0]

    qd.loop_config(
        name="kernel_manual_add_joint_limit_constraints_bw",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL),
    )
    for i_b in range(_B):
        # Forward row layout: equality -> frictionloss -> collision -> joint-limit.
        # Seed the joint-limit counter past equality + frictionloss (always)
        # and collision (when on).
        n_con_counter = gs.qd_int(
            constraint_state.n_constraints_equality[i_b] + constraint_state.n_constraints_frictionloss[i_b]
        )
        if qd.static(enable_collision):
            n_con_counter = n_con_counter + gs.qd_int(
                collider_state.n_contacts[i_b] * qd.static(rigid_config.rows_per_contact)
            )

        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j

                if (
                    dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                    or dyn_info.joints.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                ):
                    i_q = dyn_info.joints.q_start[I_j]
                    i_d = dyn_info.joints.dof_start[I_j]
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d

                    pos_delta_min = rigid_info.qpos[i_q, i_b] - dyn_info.dofs.limit[I_d][0]
                    pos_delta_max = dyn_info.dofs.limit[I_d][1] - rigid_info.qpos[i_q, i_b]
                    pos_delta = qd.min(pos_delta_min, pos_delta_max)

                    if pos_delta < 0:
                        n_con = n_con_counter
                        n_con_counter = n_con_counter + 1

                        # Replay forward intermediates (cheap, avoids stashing).
                        sign_pos = (pos_delta_min < pos_delta_max) * 2 - 1
                        sign_f = gs.qd_float(sign_pos)

                        sol_params = dyn_info.joints.sol_params[I_j]
                        imp, b_coef, k_coef, d_imp_d_imp_x = gu.imp_aref_grad(sol_params, pos_delta)
                        width = sol_params[4]

                        invweight = dyn_info.dofs.invweight[I_d]
                        diag_raw = invweight * (1.0 - imp) / imp
                        diag = qd.max(diag_raw, EPS)

                        # Upstream grads.
                        g_aref = constraint_state.dL_daref[n_con, i_b]
                        g_efc_D = constraint_state.dL_defc_D[n_con, i_b]

                        # --- Partials of forward outputs w.r.t. intermediates ---
                        # aref = -b_coef * jac_qvel - k_coef * imp * pos_delta
                        d_aref_d_imp = -k_coef * pos_delta
                        d_aref_d_jac_qvel = -b_coef
                        d_aref_d_pos_delta_direct = -k_coef * imp

                        # diag_raw = invweight * (1-imp)/imp gives d(diag_raw)/d(imp) = -invweight/imp^2;
                        # diag = max(diag_raw, EPS) and efc_D = 1/diag, with zero derivative when clamped at EPS.
                        d_diag_d_imp = gs.qd_float(0.0)
                        if diag_raw > EPS:
                            d_diag_d_imp = -invweight / (imp * imp)
                        d_efc_D_d_imp = -d_diag_d_imp / (diag * diag)

                        # d(imp_x)/d(pos_delta) = sign(pos_delta)/width, and pos_delta < 0 here
                        d_imp_d_pos_delta = d_imp_d_imp_x * (-1.0 / width)

                        # --- Combine ---
                        dL_d_imp = g_aref * d_aref_d_imp + g_efc_D * d_efc_D_d_imp
                        dL_d_pos_delta = g_aref * d_aref_d_pos_delta_direct + dL_d_imp * d_imp_d_pos_delta
                        dL_d_jac_qvel = g_aref * d_aref_d_jac_qvel

                        # --- Propagate ---
                        rigid_info.qpos.grad[i_q, i_b] += sign_f * dL_d_pos_delta
                        dyn_state.dofs.vel.grad[i_d, i_b] += sign_f * dL_d_jac_qvel


@qd.kernel(fastcache=True)
def kernel_manual_add_collision_constraints_bw(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Manual reverse of `add_collision_constraints`. See the section header
    above for the shared `n_con` layout and upstream-grad conventions.

    Produces the gradients w.r.t. the collision constraint's differentiable inputs:
        collider_state.contact_data.{pos, normal, penetration}.grad   (-> collider.backward)
        dyn_state.dofs.{cdof_ang, cdof_vel, vel}.grad
        dyn_state.links.root_COM.grad
    (cdof / root_COM / vel grads feed the COM / forward-dynamics reverse chain;
    contact_data grads feed `collider.backward`.)

    Forward recap (per contact `i_col`, per friction-pyramid row `i` in 0..3):
        d1, d2 = qd_orthogonals(normal);  d = s_i * (d1 if i<2 else d2),  s_i = 2*(i%2)-1
        n      = d * friction - normal
        jac[n_con, i_d] = sum_chain (sign * vel_motion(i_d)) . n
            vel_motion = cdof_vel - t_pos x cdof_ang,  t_pos = contact_pos - root_COM[link]
        jac_qvel = sum_chain jac[n_con, i_d] * dofs_vel[i_d]
        imp, aref = imp_aref(sol_params, -penetration, jac_qvel, -penetration)
        diag = (invweight + mu2 invweight) * 2 mu2 (1-imp)/imp, mu2 = friction^2/impratio ; efc_D = 1/diag
    """
    EPS = rigid_info.EPS[None]
    _B = dyn_state.dofs.ctrl_mode.shape[1]
    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]
    max_contact_pairs = collider_state.contact_data.link_a.shape[0]

    qd.loop_config(
        name="kernel_manual_add_collision_constraints_bw",
        # Per-contact reverses are independent (grad writes accumulate atomically); same gate as the forward
        # per-contact assembly in solver.py.
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL),
    )
    for flat_idx in range(max_contact_pairs * _B):
        i_b = flat_idx % _B
        i_col_ = flat_idx // _B
        if i_col_ < collider_state.n_contacts[i_b]:
            # The forward assembles the contact rows in logical (sorted) contact order: row group i_col_ maps to
            # physical contact contact_sort_idx[i_col_] (see add_inequality_constraints).
            i_col = collider_state.contact_sort_idx[i_col_, i_b]
            link_a = collider_state.contact_data.link_a[i_col, i_b]
            link_b = collider_state.contact_data.link_b[i_col, i_b]
            contact_pos = collider_state.contact_data.pos[i_col, i_b]
            normal = collider_state.contact_data.normal[i_col, i_b]
            friction = collider_state.contact_data.friction[i_col, i_b]
            sol_params = collider_state.contact_data.sol_params[i_col, i_b]
            penetration = collider_state.contact_data.penetration[i_col, i_b]

            link_a_maybe_batch = [link_a, i_b] if qd.static(rigid_config.batch_links_info) else link_a
            invweight = dyn_info.links.invweight[link_a_maybe_batch][0]
            if link_b > -1:
                link_b_maybe_batch = [link_b, i_b] if qd.static(rigid_config.batch_links_info) else link_b
                invweight = invweight + dyn_info.links.invweight[link_b_maybe_batch][0]

            # --- forward intermediates of qd_orthogonals(normal) ---
            #   b_raw branches on |normal[1]| < 0.5; b = normalize(b_raw)
            #   d1 = b x normal, d2 = b
            n0, n1, n2 = normal[0], normal[1], normal[2]
            is_branch_a = qd.abs(n1) < 0.5
            b_raw = gs.qd_vec3(0.0, 0.0, 0.0)
            if is_branch_a:
                b_raw = gs.qd_vec3(-n0 * n1, 1.0 - n1 * n1, -n2 * n1)
            else:
                b_raw = gs.qd_vec3(-n0 * n2, -n1 * n2, 1.0 - n2 * n2)
            b_raw_norm = b_raw.norm()
            b = b_raw / b_raw_norm
            d1 = b.cross(normal)
            d2 = b

            neg_pen = -penetration
            imp, b_coef, k_coef, d_imp_d_imp_x = gu.imp_aref_grad(sol_params, neg_pen)
            # d(imp_x)/d(penetration) = -sign(neg_pen)/width
            sign_neg = gs.qd_float(1.0) if neg_pen >= 0 else gs.qd_float(-1.0)
            d_imp_x_d_pen = -sign_neg / sol_params[4]

            # diag = C0 * (1-imp)/imp with the impratio-regularized cone coefficient of the forward:
            # friction_sq_reg = friction^2 / impratio (see add_collision_constraints in solver.py).
            friction_sq_reg = friction * friction / rigid_info.impratio[None]
            C0 = (invweight + friction_sq_reg * invweight) * 2.0 * friction_sq_reg
            diag_raw = C0 * (1.0 - imp) / imp
            diag = qd.max(diag_raw, EPS)

            d_diag_d_imp = gs.qd_float(0.0)
            if diag_raw > EPS:
                d_diag_d_imp = -C0 / (imp * imp)
            d_efc_D_d_imp = -d_diag_d_imp / (diag * diag)

            # Accumulators for this contact's differentiable inputs.
            g_pos = gs.qd_vec3(0.0, 0.0, 0.0)
            g_normal = gs.qd_vec3(0.0, 0.0, 0.0)
            g_pen = gs.qd_float(0.0)
            g_d1 = gs.qd_vec3(0.0, 0.0, 0.0)
            g_d2 = gs.qd_vec3(0.0, 0.0, 0.0)

            # Forward row layout: equality -> frictionloss -> collision -> joint-limit.
            # Offset past equality + frictionloss.
            const_start = (
                constraint_state.n_constraints_equality[i_b] + constraint_state.n_constraints_frictionloss[i_b]
            )
            for i in range(qd.static(rigid_config.rows_per_contact)):
                s_i = gs.qd_float(2 * (i % 2) - 1)
                d = s_i * d1 if i < 2 else s_i * d2
                n = d * friction - normal
                n_con = const_start + i_col_ * qd.static(rigid_config.rows_per_contact) + i

                g_aref = constraint_state.dL_daref[n_con, i_b]
                g_efc_D = constraint_state.dL_defc_D[n_con, i_b]

                # aref = -b_coef*jac_qvel + k_coef*imp*penetration  (pos arg = -penetration)
                d_aref_d_imp = k_coef * penetration
                d_aref_d_pen_direct = k_coef * imp
                d_aref_d_jac_qvel = -b_coef

                dL_d_imp = g_aref * d_aref_d_imp + g_efc_D * d_efc_D_d_imp
                dL_d_pen = g_aref * d_aref_d_pen_direct + dL_d_imp * d_imp_d_imp_x * d_imp_x_d_pen
                g_pen += dL_d_pen
                dL_d_jac_qvel = g_aref * d_aref_d_jac_qvel

                # d(jac_qvel)/d(vel[i_d]) = jac[n_con, i_d]: accumulate once per relevant dof recorded by the
                # forward row assembly - the chain walk below visits shared ancestor dofs of same-root pairs
                # twice (see _append_relevant_dof in solver.py).
                for k in range(constraint_state.jac_n_dofs[n_con, i_b]):
                    i_d = constraint_state.jac_dofs_idx[n_con, k, i_b]
                    dyn_state.dofs.vel.grad[i_d, i_b] += dL_d_jac_qvel * constraint_state.jac[n_con, i_d, i_b]

                # Reverse jac[n_con, i_d] over the kinematic chain.
                dL_dn = gs.qd_vec3(0.0, 0.0, 0.0)
                for i_ab in range(2):
                    sign = gs.qd_float(-1.0)
                    link = link_a
                    if i_ab == 1:
                        sign = gs.qd_float(1.0)
                        link = link_b
                    while link > -1:
                        link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link
                        for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                            i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_

                            cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                            cdof_vel = dyn_state.dofs.cdof_vel[i_d, i_b]
                            t_pos = contact_pos - dyn_state.links.root_COM[link, i_b]
                            vel_motion = cdof_vel - t_pos.cross(cdof_ang)

                            g_jac = (
                                constraint_state.dL_djac[n_con, i_d, i_b] + dL_d_jac_qvel * dyn_state.dofs.vel[i_d, i_b]
                            )

                            # jac_contrib = (sign * vel_motion) . n
                            dL_dn += g_jac * sign * vel_motion
                            # dL/d(vel_motion)
                            g_vm = g_jac * sign * n

                            # vel_motion = cdof_vel - t_pos x cdof_ang
                            dyn_state.dofs.cdof_vel.grad[i_d, i_b] += g_vm
                            dyn_state.dofs.cdof_ang.grad[i_d, i_b] += t_pos.cross(g_vm)
                            # dL/d(t_pos)
                            g_t_pos = -cdof_ang.cross(g_vm)
                            g_pos += g_t_pos
                            dyn_state.links.root_COM.grad[link, i_b] += -g_t_pos

                        link = dyn_info.links.parent_idx[link_maybe_batch]

                # n = d*friction - normal
                g_normal += -dL_dn
                g_d = dL_dn * friction
                if i < 2:
                    g_d1 += s_i * g_d
                else:
                    g_d2 += s_i * g_d

            # Reverse qd_orthogonals: d1 = b x normal, d2 = b, b = normalize(b_raw(normal)).
            dL_db = g_d2 + normal.cross(g_d1)
            g_normal += g_d1.cross(b)
            # b = b_raw / |b_raw|
            dL_db_raw = (dL_db - dL_db.dot(b) * b) / b_raw_norm
            # b_raw(normal) branch Jacobian
            if is_branch_a:
                # b_raw = (-n0 n1, 1 - n1^2, -n2 n1)
                g_normal[0] += dL_db_raw[0] * (-n1)
                g_normal[1] += dL_db_raw[0] * (-n0) + dL_db_raw[1] * (-2.0 * n1) + dL_db_raw[2] * (-n2)
                g_normal[2] += dL_db_raw[2] * (-n1)
            else:
                # b_raw = (-n0 n2, -n1 n2, 1 - n2^2)
                g_normal[0] += dL_db_raw[0] * (-n2)
                g_normal[1] += dL_db_raw[1] * (-n2)
                g_normal[2] += dL_db_raw[0] * (-n0) + dL_db_raw[1] * (-n1) + dL_db_raw[2] * (-2.0 * n2)

            for j in qd.static(range(3)):
                collider_state.contact_data.pos.grad[i_col, i_b][j] = g_pos[j]
                collider_state.contact_data.normal.grad[i_col, i_b][j] = g_normal[j]
            collider_state.contact_data.penetration.grad[i_col, i_b] = g_pen


@qd.kernel(fastcache=True)
def kernel_manual_add_frictionloss_constraints_bw(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Manual reverse of `add_frictionloss_constraints`. See the section header
    above for the shared `n_con` layout and upstream-grad conventions.

    Accumulates into dyn_state.dofs.vel.grad[i_d] only (the conservative, kinematic-
    state-only target). Model parameters (frictionloss, sol_params, invweight)
    are not differentiated.

    Forward recap (per dof with `frictionloss[I_d] > EPS`, `pos_delta = 0`):
        jac[n_con, i_d] = 1.0
        jac_qvel = jac * vel[i_d] = vel[i_d]
        imp, aref = imp_aref(sol_params, 0, jac_qvel, 0)
        diag = max(invweight * (1 - imp) / imp, EPS); efc_D = 1/diag

    Reverse: pos_delta = 0 kills both the `imp  *  pos_delta` term in `aref`
    and the entire `imp` sensitivity to anything (imp_x = 0 => within_clamp
    is False => d_imp / d_anything = 0). What survives is the direct
    `aref = -b_coef  *  jac_qvel` term, so

        dL/d_vel[i_d] += dL_daref[n_con] * (-b_coef)
    """
    EPS = rigid_info.EPS[None]
    _B = constraint_state.jac.shape[2]
    n_links = dyn_info.links.root_idx.shape[0]

    qd.loop_config(
        name="kernel_manual_add_frictionloss_constraints_bw",
        # Same serialize condition as the forward; see add_frictionloss_constraints in solver.py for the Metal gate
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL and rigid_config.backend != gs.metal),
    )
    for i_b in range(_B):
        # Forward row layout: equality -> frictionloss -> collision -> joint-limit.
        # Frictionloss row counter starts past the equality block (which may be
        # nonzero when JOINT-type equalities are present).
        n_con_counter = gs.qd_int(constraint_state.n_constraints_equality[i_b])

        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                for i_d in range(dyn_info.joints.dof_start[I_j], dyn_info.joints.dof_end[I_j]):
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d

                    if dyn_info.dofs.frictionloss[I_d] > EPS:
                        n_con = n_con_counter
                        n_con_counter = n_con_counter + 1

                        sol_params = dyn_info.joints.sol_params[I_j]
                        timeconst = sol_params[0]
                        dmax = sol_params[3]
                        b_coef = 2.0 / (dmax * timeconst)

                        g_aref = constraint_state.dL_daref[n_con, i_b]
                        # jac = 1.0 constant => dL/d_vel = dL/d_jac_qvel.
                        dyn_state.dofs.vel.grad[i_d, i_b] += g_aref * (-b_coef)


@qd.func
def func_cddb_ang_bw(
    i_b,
    link,
    g_cddb_ang,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Reverse of the chain contraction cddb_ang = sum_d cdofd_ang[d] * vel[d] (see func_equality_jdotv in
    solver.py), accumulating into cdofd_ang.grad and vel.grad over the link's ancestor dofs."""
    i_l = link
    while i_l > -1:
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
            dyn_state.dofs.cdofd_ang.grad[i_d, i_b] += g_cddb_ang * dyn_state.dofs.vel[i_d, i_b]
            dyn_state.dofs.vel.grad[i_d, i_b] += g_cddb_ang.dot(dyn_state.dofs.cdofd_ang[i_d, i_b])
        i_l = dyn_info.links.parent_idx[I_l]


@qd.func
def func_equality_jdotv_bw(
    i_b,
    link,
    anchor_pos,
    g_jdotv,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Reverse of func_equality_jdotv (see solver.py) for one chain, given the upstream gradient g_jdotv of its
    linear Jdot @ qvel.

    Accumulates into the chain dofs' cdofd_ang / cdofd_vel / vel grads and the link's cd_ang / cd_vel / root_COM
    grads, and returns the gradient w.r.t. the world anchor position (the caller owns the anchor -> link pos / quat
    chain). Returns zero for the world (link == -1)."""
    g_anchor = gs.qd_vec3(0.0, 0.0, 0.0)
    if link > -1:
        # Replay the chain contraction; only cddb_ang is consumed by the adjoint below
        _jdotv, cddb_ang = solver.func_equality_jdotv(i_b, link, anchor_pos, dyn_state, dyn_info, rigid_config)
        offset = anchor_pos - dyn_state.links.root_COM[link, i_b]
        pvel = dyn_state.links.cd_vel[link, i_b] + dyn_state.links.cd_ang[link, i_b].cross(offset)

        # jdotv = cddb_vel + cddb_ang x offset + cd_ang x pvel, using g_u = w x g and g_w = g x u for c = u x w
        g_cddb_vel = g_jdotv
        g_cddb_ang = offset.cross(g_jdotv)
        g_offset = g_jdotv.cross(cddb_ang)
        g_cd_ang = pvel.cross(g_jdotv)
        g_pvel = g_jdotv.cross(dyn_state.links.cd_ang[link, i_b])

        # pvel = cd_vel + cd_ang x offset
        g_cd_vel = g_pvel
        g_cd_ang = g_cd_ang + offset.cross(g_pvel)
        g_offset = g_offset + g_pvel.cross(dyn_state.links.cd_ang[link, i_b])

        # offset = anchor_pos - root_COM[link]
        g_anchor = g_offset
        dyn_state.links.root_COM.grad[link, i_b] += -g_offset
        dyn_state.links.cd_ang.grad[link, i_b] += g_cd_ang
        dyn_state.links.cd_vel.grad[link, i_b] += g_cd_vel

        # cddb_{ang,vel} = sum_d cdofd_{ang,vel}[d] * vel[d] over the ancestor chain
        i_l = link
        while i_l > -1:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                dyn_state.dofs.cdofd_ang.grad[i_d, i_b] += g_cddb_ang * dyn_state.dofs.vel[i_d, i_b]
                dyn_state.dofs.cdofd_vel.grad[i_d, i_b] += g_cddb_vel * dyn_state.dofs.vel[i_d, i_b]
                dyn_state.dofs.vel.grad[i_d, i_b] += g_cddb_ang.dot(dyn_state.dofs.cdofd_ang[i_d, i_b])
                dyn_state.dofs.vel.grad[i_d, i_b] += g_cddb_vel.dot(dyn_state.dofs.cdofd_vel[i_d, i_b])
            i_l = dyn_info.links.parent_idx[I_l]
    return g_anchor


@qd.kernel(fastcache=True)
def kernel_manual_add_equality_constraints_bw(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Manual reverse of `add_equality_constraints` (JOINT + CONNECT + WELD).

    * JOINT   - constrains one scalar dof, optionally against another via a quartic polynomial (1 row).
    * CONNECT - 3 rows pinning `global_anchor1` to `global_anchor2` in world.
    * WELD    - 6 rows: 3 position + 3 orientation, all sharing a single
                combined `pos_imp = ||all_error||` (6D).

    Accumulates into kinematic-state grads only (conservative):
        JOINT:
            rigid_info.qpos.grad[i_qpos1], and qpos.grad[i_qpos2] when joint2 exists
            dyn_state.dofs.vel.grad[i_dof1], and vel.grad[i_dof2] when joint2 exists
        CONNECT / WELD:
            dyn_state.links.{pos, quat, root_COM, cd_ang, cd_vel}.grad[link1 / link2]
            dyn_state.dofs.{cdof_ang, cdof_vel, cdofd_ang, cdofd_vel, vel}.grad over each link chain
    Model parameters (`sol_params`, `eq_data`, `dyn_info.dofs.invweight`,
    `dyn_info.links.invweight`) are not differentiated.

    Forward recap (per equality of type JOINT):
        diff = qpos[i_qpos2] - qpos0[i_qpos2], or 0 when joint2 is omitted
        pos_poly = a0 + a1 * diff + a2 * diff^2 + a3 * diff^3 + a4 * diff^4
        pos = qpos[i_qpos1] - qpos0[i_qpos1] - pos_poly
        deriv = d(pos_poly)/d(diff) = a1 + 2 * a2 * diff + 3 * a3 * diff^2 + 4 * a4 * diff^3
        jac[n_con, i_dof1] = 1.0
        jac[n_con, i_dof2] = -deriv, when joint2 exists
        jac_qvel = vel[i_dof1] - deriv * vel[i_dof2], with the second term omitted when joint2 is omitted
        imp, aref = imp_aref(sol_params, -|pos|, jac_qvel, pos)
            aref = -b * jac_qvel - k * imp * pos
        diag = max(invweight * (1 - imp) / imp, EPS); efc_D = 1/diag
    """
    EPS = rigid_info.EPS[None]
    _B = constraint_state.jac.shape[2]

    qd.loop_config(
        name="kernel_manual_add_equality_constraints_bw",
        serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL),
    )
    for i_b in range(_B):
        # Equality is the first constraint group; row counter starts at 0.
        n_con_counter = gs.qd_int(0)

        for i_e in range(constraint_state.qd_n_equalities[i_b]):
            if dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.JOINT:
                n_con = n_con_counter
                n_con_counter = n_con_counter + 1

                # ---- Replay forward intermediates ----
                I_joint1 = (
                    [dyn_info.equalities.eq_obj1id[i_e, i_b], i_b]
                    if qd.static(rigid_config.batch_joints_info)
                    else dyn_info.equalities.eq_obj1id[i_e, i_b]
                )
                i_qpos1 = dyn_info.joints.q_start[I_joint1]
                i_dof1 = dyn_info.joints.dof_start[I_joint1]
                I_dof1 = [i_dof1, i_b] if qd.static(rigid_config.batch_dofs_info) else i_dof1

                pos1 = rigid_info.qpos[i_qpos1, i_b]
                ref1 = rigid_info.qpos0[i_qpos1, i_b]
                i_joint2 = dyn_info.equalities.eq_obj2id[i_e, i_b]
                i_qpos2 = -1
                i_dof2 = -1
                diff = gs.qd_float(0.0)
                if i_joint2 >= 0:
                    I_joint2 = [i_joint2, i_b] if qd.static(rigid_config.batch_joints_info) else i_joint2
                    i_qpos2 = dyn_info.joints.q_start[I_joint2]
                    i_dof2 = dyn_info.joints.dof_start[I_joint2]
                    diff = rigid_info.qpos[i_qpos2, i_b] - rigid_info.qpos0[i_qpos2, i_b]

                a0 = dyn_info.equalities.eq_data[i_e, i_b][0]
                a1 = dyn_info.equalities.eq_data[i_e, i_b][1]
                a2 = dyn_info.equalities.eq_data[i_e, i_b][2]
                a3 = dyn_info.equalities.eq_data[i_e, i_b][3]
                a4 = dyn_info.equalities.eq_data[i_e, i_b][4]

                diff2 = diff * diff
                diff3 = diff2 * diff
                diff4 = diff3 * diff
                pos = pos1 - ref1 - a0 - a1 * diff - a2 * diff2 - a3 * diff3 - a4 * diff4
                deriv = a1 + 2.0 * a2 * diff + 3.0 * a3 * diff2 + 4.0 * a4 * diff3
                d_deriv_d_diff = 2.0 * a2 + 6.0 * a3 * diff + 12.0 * a4 * diff2
                if i_joint2 < 0:
                    deriv = gs.qd_float(0.0)
                    d_deriv_d_diff = gs.qd_float(0.0)

                jac1 = gs.qd_float(1.0)
                jac2 = gs.qd_float(0.0)
                vel1 = dyn_state.dofs.vel[i_dof1, i_b]
                vel2 = gs.qd_float(0.0)
                jac_qvel = jac1 * vel1
                invweight = dyn_info.dofs.invweight[I_dof1]
                if i_joint2 >= 0:
                    I_dof2 = [i_dof2, i_b] if qd.static(rigid_config.batch_dofs_info) else i_dof2
                    jac2 = -deriv
                    vel2 = dyn_state.dofs.vel[i_dof2, i_b]
                    jac_qvel = jac_qvel + jac2 * vel2
                    invweight = invweight + dyn_info.dofs.invweight[I_dof2]

                sol_params = dyn_info.equalities.sol_params[i_e, i_b]
                imp, b_coef, k_coef, d_imp_d_imp_x = gu.imp_aref_grad(sol_params, pos)
                width = sol_params[4]

                diag_raw = invweight * (1.0 - imp) / imp
                diag = qd.max(diag_raw, EPS)

                # ---- Upstream grads ----
                g_aref = constraint_state.dL_daref[n_con, i_b]
                g_efc_D = constraint_state.dL_defc_D[n_con, i_b]
                # jac[dof1] = 1.0 (constant) => no chain through dL_djac[n_con, i_dof1].
                g_jac2 = gs.qd_float(0.0)
                if i_joint2 >= 0:
                    g_jac2 = constraint_state.dL_djac[n_con, i_dof2, i_b]

                # ---- Partials ----
                # aref = -b * jac_qvel - k * imp * pos
                d_aref_d_jac_qvel = -b_coef
                d_aref_d_pos_direct = -k_coef * imp
                d_aref_d_imp = -k_coef * pos

                # diag = max(diag_raw, EPS); efc_D = 1/diag
                d_diag_d_imp = gs.qd_float(0.0)
                if diag_raw > EPS:
                    d_diag_d_imp = -invweight / (imp * imp)
                d_efc_D_d_imp = -d_diag_d_imp / (diag * diag)

                # imp_x = |pos|/width  =>  d_imp_x/d_pos = sign(pos) / width
                sign_pos_f = gs.qd_float(1.0)
                if pos < 0.0:
                    sign_pos_f = gs.qd_float(-1.0)
                d_imp_d_pos = d_imp_d_imp_x * sign_pos_f / width

                # ---- Combine ----
                dL_d_imp = g_aref * d_aref_d_imp + g_efc_D * d_efc_D_d_imp
                dL_d_jac_qvel = g_aref * d_aref_d_jac_qvel
                dL_d_pos = g_aref * d_aref_d_pos_direct + dL_d_imp * d_imp_d_pos
                # deriv enters via (jac_qvel through jac2 = -deriv) and (jac[dof2] = -deriv).
                dL_d_deriv = dL_d_jac_qvel * (-vel2) + g_jac2 * (-1.0)

                # ---- Propagate ----
                dyn_state.dofs.vel.grad[i_dof1, i_b] += dL_d_jac_qvel * jac1
                # qpos1 enters only via pos (d_pos/d_pos1 = 1).
                rigid_info.qpos.grad[i_qpos1, i_b] += dL_d_pos
                if i_joint2 >= 0:
                    dyn_state.dofs.vel.grad[i_dof2, i_b] += dL_d_jac_qvel * jac2
                    # qpos2 enters via pos (d_pos/d_pos2 = -deriv) and deriv (d_deriv/d_diff).
                    rigid_info.qpos.grad[i_qpos2, i_b] += dL_d_pos * (-deriv) + dL_d_deriv * d_deriv_d_diff
            elif dyn_info.equalities.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.CONNECT:
                # ----------------------------------------------------------
                # CONNECT: 3 rows pin global_anchor1 == global_anchor2.
                #
                # Forward recap (per row i_3 in {0,1,2}):
                #   ga1 = trans(dyn_state.links.pos[link1], dyn_state.links.quat[link1]) * eq_data[0:3]
                #   ga2 = trans(dyn_state.links.pos[link2], dyn_state.links.quat[link2]) * eq_data[3:6]
                #   For each link in (link1, link2) chain, for each dof on that link:
                #       t_pos = ga_link - root_COM[link]
                #       vel_motion = cdof_vel - t_pos x cdof_ang
                #       jac_i3 = sign * vel_motion[i_3]   (sign = +1 for link1, -1 for link2)
                #       jac[n_con, i_d] += jac_i3
                #       jac_qvel       += jac_i3 * vel[i_d]
                #   pos_diff   = ga1 - ga2
                #   penetration = ||pos_diff||
                #   imp, aref = imp_aref(sol_params, -penetration, jac_qvel, pos_diff[i_3])
                #       aref = -b * jac_qvel - k * imp * pos_diff[i_3]
                #   stored aref[n_con] = aref - jdotv[i_3], jdotv = jdotv1 - jdotv2 (func_equality_jdotv)
                #   diag = max(invweight * (1 - imp) / imp, EPS); efc_D = 1/diag
                # ----------------------------------------------------------
                link1_idx = dyn_info.equalities.eq_obj1id[i_e, i_b]
                link2_idx = dyn_info.equalities.eq_obj2id[i_e, i_b]
                link1_maybe_batch = [link1_idx, i_b] if qd.static(rigid_config.batch_links_info) else link1_idx
                link2_maybe_batch = [link2_idx, i_b] if qd.static(rigid_config.batch_links_info) else link2_idx

                anchor1_local = gs.qd_vec3(
                    dyn_info.equalities.eq_data[i_e, i_b][0],
                    dyn_info.equalities.eq_data[i_e, i_b][1],
                    dyn_info.equalities.eq_data[i_e, i_b][2],
                )
                anchor2_local = gs.qd_vec3(
                    dyn_info.equalities.eq_data[i_e, i_b][3],
                    dyn_info.equalities.eq_data[i_e, i_b][4],
                    dyn_info.equalities.eq_data[i_e, i_b][5],
                )

                quat1 = dyn_state.links.quat[link1_idx, i_b]
                quat2 = dyn_state.links.quat[link2_idx, i_b]
                trans1 = dyn_state.links.pos[link1_idx, i_b]
                trans2 = dyn_state.links.pos[link2_idx, i_b]
                ga1 = gu.qd_transform_by_trans_quat(pos=anchor1_local, trans=trans1, quat=quat1)
                ga2 = gu.qd_transform_by_trans_quat(pos=anchor2_local, trans=trans2, quat=quat2)
                pos_diff = ga1 - ga2
                penetration = pos_diff.norm()

                invweight = (
                    dyn_info.links.invweight[link1_maybe_batch][0] + dyn_info.links.invweight[link2_maybe_batch][0]
                )

                sol_params = dyn_info.equalities.sol_params[i_e, i_b]
                imp, b_coef, k_coef, d_imp_d_imp_x = gu.imp_aref_grad(sol_params, -penetration)
                width = sol_params[4]

                diag_raw = invweight * (1.0 - imp) / imp
                diag = qd.max(diag_raw, EPS)

                # All 3 rows share imp / penetration / pos_diff. Per-row partials
                # only differ in which axis pos_diff[i_3] is used as ref_arg.
                d_diag_d_imp = gs.qd_float(0.0)
                if diag_raw > EPS:
                    d_diag_d_imp = -invweight / (imp * imp)
                d_efc_D_d_imp = -d_diag_d_imp / (diag * diag)

                # Accumulate dL/d_ga over the 3 rows so we propagate to
                # dyn_state.links.{pos,quat} only once per anchor.
                g_ga1 = gs.qd_vec3(0.0, 0.0, 0.0)
                g_ga2 = gs.qd_vec3(0.0, 0.0, 0.0)
                # The stored rows are aref - jdotv[i_3] (see func_equality_connect in solver.py); collect the bias
                # gradient per row and reverse it through both chains after the row loop.
                g_jdotv = gs.qd_vec3(0.0, 0.0, 0.0)

                for i_3 in range(3):
                    n_con = n_con_counter
                    n_con_counter = n_con_counter + 1

                    g_aref = constraint_state.dL_daref[n_con, i_b]
                    g_efc_D = constraint_state.dL_defc_D[n_con, i_b]

                    d_aref_d_jac_qvel = -b_coef
                    d_aref_d_pos_diff_i3_direct = -k_coef * imp
                    d_aref_d_imp = -k_coef * pos_diff[i_3]
                    g_jdotv[i_3] = -g_aref

                    dL_d_imp = g_aref * d_aref_d_imp + g_efc_D * d_efc_D_d_imp
                    dL_d_jac_qvel = g_aref * d_aref_d_jac_qvel

                    # d(jac_qvel)/d(vel[i_d]) over the forward-recorded dof support (see the collision reverse
                    # above for why the chain walk cannot accumulate this).
                    for k in range(constraint_state.jac_n_dofs[n_con, i_b]):
                        i_d = constraint_state.jac_dofs_idx[n_con, k, i_b]
                        dyn_state.dofs.vel.grad[i_d, i_b] += dL_d_jac_qvel * constraint_state.jac[n_con, i_d, i_b]

                    # dL/d_pos_diff: (a) direct axis-i_3 term, (b) via penetration / imp.
                    g_pos_diff = gs.qd_vec3(0.0, 0.0, 0.0)
                    g_pos_diff[i_3] = g_aref * d_aref_d_pos_diff_i3_direct
                    if penetration > EPS:
                        coef_pen = dL_d_imp * d_imp_d_imp_x / (width * penetration)
                        for j in qd.static(range(3)):
                            g_pos_diff[j] = g_pos_diff[j] + coef_pen * pos_diff[j]

                    # Walk both chains, accumulating cdof / vel / root_COM grads
                    # and the anchor portion that flows back through t_pos.
                    g_anchor1_row = gs.qd_vec3(0.0, 0.0, 0.0)
                    g_anchor2_row = gs.qd_vec3(0.0, 0.0, 0.0)
                    for i_ab in range(2):
                        sign = gs.qd_float(1.0)
                        link = link1_idx
                        anchor_pos = ga1
                        if i_ab == 1:
                            sign = gs.qd_float(-1.0)
                            link = link2_idx
                            anchor_pos = ga2

                        while link > -1:
                            link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link
                            for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                                i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_

                                cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                                cdof_vel = dyn_state.dofs.cdof_vel[i_d, i_b]
                                t_pos = anchor_pos - dyn_state.links.root_COM[link, i_b]

                                # jac_i3 = sign * vel_motion[i_3]
                                # upstream: dL_djac[n_con, i_d] + dL_d_jac_qvel * vel[i_d]
                                g_jac_i3 = (
                                    constraint_state.dL_djac[n_con, i_d, i_b]
                                    + dL_d_jac_qvel * dyn_state.dofs.vel[i_d, i_b]
                                )

                                # dL/d_vel_motion (only component i_3)
                                g_vm = gs.qd_vec3(0.0, 0.0, 0.0)
                                g_vm[i_3] = g_jac_i3 * sign

                                # vel_motion = cdof_vel - t_pos x cdof_ang
                                dyn_state.dofs.cdof_vel.grad[i_d, i_b] += g_vm
                                dyn_state.dofs.cdof_ang.grad[i_d, i_b] += t_pos.cross(g_vm)
                                g_t_pos = -cdof_ang.cross(g_vm)
                                # t_pos = anchor_pos - root_COM[link]
                                if i_ab == 0:
                                    g_anchor1_row = g_anchor1_row + g_t_pos
                                else:
                                    g_anchor2_row = g_anchor2_row + g_t_pos
                                dyn_state.links.root_COM.grad[link, i_b] += -g_t_pos

                            link = dyn_info.links.parent_idx[link_maybe_batch]

                    # pos_diff = ga1 - ga2 => g_ga1 += g_pos_diff, g_ga2 += -g_pos_diff.
                    g_ga1 = g_ga1 + g_pos_diff + g_anchor1_row
                    g_ga2 = g_ga2 - g_pos_diff + g_anchor2_row

                # Reverse the velocity-product bias jdotv = jdotv1 - jdotv2 through both chains
                g_ga1 = g_ga1 + func_equality_jdotv_bw(i_b, link1_idx, ga1, g_jdotv, dyn_state, dyn_info, rigid_config)
                g_ga2 = g_ga2 + func_equality_jdotv_bw(i_b, link2_idx, ga2, -g_jdotv, dyn_state, dyn_info, rigid_config)

                # Propagate accumulated g_ga grads back to dyn_state.links.{pos, quat}.
                # ga = trans + R(quat) * anchor_local; anchor_local is model param.
                g_quat1 = gu.qd_transform_by_quat_grad_quat(anchor1_local, quat1, g_ga1)
                g_quat2 = gu.qd_transform_by_quat_grad_quat(anchor2_local, quat2, g_ga2)
                dyn_state.links.pos.grad[link1_idx, i_b] += g_ga1
                dyn_state.links.pos.grad[link2_idx, i_b] += g_ga2
                dyn_state.links.quat.grad[link1_idx, i_b] += g_quat1
                dyn_state.links.quat.grad[link2_idx, i_b] += g_quat2
            else:
                # ----------------------------------------------------------
                # WELD: 6 rows - 3 position + 3 orientation, all sharing
                # a single combined pos_imp = ||all_error|| (6D).
                #
                # Forward recap:
                #   ga1 = trans(dyn_state.links.pos[link1], dyn_state.links.quat[link1]) * eq_data[3:6]
                #   ga2 = trans(dyn_state.links.pos[link2], dyn_state.links.quat[link2]) * eq_data[0:3]
                #   pos_error = ga1 - ga2
                #   inv_q2 = inv_quat(dyn_state.links.quat[link2])
                #   q       = quat_mul(dyn_state.links.quat[link1], relpose)
                #   error_quat = quat_mul(inv_q2, q)
                #   rot_error  = error_quat.xyz * torquescale
                #   all_error  = (pos_error, rot_error);  pos_imp = ||all_error||
                #
                # Position rows (i_3 = 0..2): same chain structure as CONNECT
                # with invweight[0]; ref_arg = pos_error[i_3].
                # Orientation rows (i_3 = 0..2): jac_phase1 = sign * cdof_ang[i_d]
                # then quat post-process
                #   quat2_d = qd_quat_mul_axis(inv_q2, jac_phase1[i_d])
                #   quat3_d = qd_quat_mul(quat2_d, q)
                #   jac[n_con+3+i_3, i_d] = 0.5 * torquescale * quat3_d[i_3+1]
                # ----------------------------------------------------------
                link1_idx = dyn_info.equalities.eq_obj1id[i_e, i_b]
                link2_idx = dyn_info.equalities.eq_obj2id[i_e, i_b]
                link1_maybe_batch = [link1_idx, i_b] if qd.static(rigid_config.batch_links_info) else link1_idx
                link2_maybe_batch = [link2_idx, i_b] if qd.static(rigid_config.batch_links_info) else link2_idx

                # WELD eq_data layout (per forward comment):
                # [0:3] anchor2 (local), [3:6] anchor1 (local), [6:10] relpose, [10] torquescale
                anchor1_local = gs.qd_vec3(
                    dyn_info.equalities.eq_data[i_e, i_b][3],
                    dyn_info.equalities.eq_data[i_e, i_b][4],
                    dyn_info.equalities.eq_data[i_e, i_b][5],
                )
                anchor2_local = gs.qd_vec3(
                    dyn_info.equalities.eq_data[i_e, i_b][0],
                    dyn_info.equalities.eq_data[i_e, i_b][1],
                    dyn_info.equalities.eq_data[i_e, i_b][2],
                )
                relpose = qd.Vector(
                    [
                        dyn_info.equalities.eq_data[i_e, i_b][6],
                        dyn_info.equalities.eq_data[i_e, i_b][7],
                        dyn_info.equalities.eq_data[i_e, i_b][8],
                        dyn_info.equalities.eq_data[i_e, i_b][9],
                    ],
                    dt=gs.qd_float,
                )
                torquescale = dyn_info.equalities.eq_data[i_e, i_b][10]

                quat_body1 = dyn_state.links.quat[link1_idx, i_b]
                quat_body2 = dyn_state.links.quat[link2_idx, i_b]
                trans1 = dyn_state.links.pos[link1_idx, i_b]
                trans2 = dyn_state.links.pos[link2_idx, i_b]
                ga1 = gu.qd_transform_by_trans_quat(pos=anchor1_local, trans=trans1, quat=quat_body1)
                ga2 = gu.qd_transform_by_trans_quat(pos=anchor2_local, trans=trans2, quat=quat_body2)
                pos_error = ga1 - ga2

                inv_q2 = gu.qd_inv_quat(quat_body2)
                q_var = gu.qd_quat_mul(quat_body1, relpose)
                error_quat = gu.qd_quat_mul(inv_q2, q_var)
                rot_error = gs.qd_vec3(error_quat[1], error_quat[2], error_quat[3]) * torquescale

                # all_error = (pos_error, rot_error) ; pos_imp = ||all_error||
                pos_imp = qd.sqrt(
                    pos_error[0] * pos_error[0]
                    + pos_error[1] * pos_error[1]
                    + pos_error[2] * pos_error[2]
                    + rot_error[0] * rot_error[0]
                    + rot_error[1] * rot_error[1]
                    + rot_error[2] * rot_error[2]
                )

                invweight_pos = (
                    dyn_info.links.invweight[link1_maybe_batch][0] + dyn_info.links.invweight[link2_maybe_batch][0]
                )
                invweight_rot = (
                    dyn_info.links.invweight[link1_maybe_batch][1] + dyn_info.links.invweight[link2_maybe_batch][1]
                )

                sol_params = dyn_info.equalities.sol_params[i_e, i_b]
                imp, b_coef, k_coef, d_imp_d_imp_x = gu.imp_aref_grad(sol_params, -pos_imp)
                width = sol_params[4]

                # Per-group diag/efc_D depend on invweight; same imp.
                diag_raw_pos = invweight_pos * (1.0 - imp) / imp
                diag_pos = qd.max(diag_raw_pos, EPS)
                d_diag_d_imp_pos = gs.qd_float(0.0)
                if diag_raw_pos > EPS:
                    d_diag_d_imp_pos = -invweight_pos / (imp * imp)
                d_efc_D_d_imp_pos = -d_diag_d_imp_pos / (diag_pos * diag_pos)

                diag_raw_rot = invweight_rot * (1.0 - imp) / imp
                diag_rot = qd.max(diag_raw_rot, EPS)
                d_diag_d_imp_rot = gs.qd_float(0.0)
                if diag_raw_rot > EPS:
                    d_diag_d_imp_rot = -invweight_rot / (imp * imp)
                d_efc_D_d_imp_rot = -d_diag_d_imp_rot / (diag_rot * diag_rot)

                # Accumulators across all 6 rows.
                g_ga1 = gs.qd_vec3(0.0, 0.0, 0.0)
                g_ga2 = gs.qd_vec3(0.0, 0.0, 0.0)
                g_rot_error = gs.qd_vec3(0.0, 0.0, 0.0)
                dL_d_imp_total = gs.qd_float(0.0)
                # Per-row jac_qvel grad (for the orientation chain walk below).
                dL_d_jac_qvel_orient = gs.qd_vec3(0.0, 0.0, 0.0)
                # The stored position rows are aref - jdotv[i_3] and the rotation rows carry the rotational bias
                # 0.5 * (t1 + t2 + t3)[i_3 + 1] * torquescale (see func_equality_weld in solver.py); collect both
                # bias gradients per row and reverse them after the row loops.
                g_jdotv = gs.qd_vec3(0.0, 0.0, 0.0)
                g_t_quat = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.qd_float)

                # ---- Position rows (3) -- mirrors CONNECT structure ----
                n_con_orient_base = n_con_counter + 3  # rotation rows start here
                for i_3 in range(3):
                    n_con = n_con_counter
                    n_con_counter = n_con_counter + 1

                    g_aref = constraint_state.dL_daref[n_con, i_b]
                    g_efc_D = constraint_state.dL_defc_D[n_con, i_b]

                    d_aref_d_jac_qvel = -b_coef
                    d_aref_d_ref_direct = -k_coef * imp
                    d_aref_d_imp = -k_coef * pos_error[i_3]
                    g_jdotv[i_3] = -g_aref

                    dL_d_imp_total = dL_d_imp_total + g_aref * d_aref_d_imp + g_efc_D * d_efc_D_d_imp_pos
                    dL_d_jac_qvel = g_aref * d_aref_d_jac_qvel
                    # Direct ref-axis contribution (pos_error[i_3]):
                    g_pos_error_direct = g_aref * d_aref_d_ref_direct

                    # d(jac_qvel)/d(vel[i_d]) over the forward-recorded dof support (see the collision reverse
                    # above for why the chain walk cannot accumulate this).
                    for k in range(constraint_state.jac_n_dofs[n_con, i_b]):
                        i_d = constraint_state.jac_dofs_idx[n_con, k, i_b]
                        dyn_state.dofs.vel.grad[i_d, i_b] += dL_d_jac_qvel * constraint_state.jac[n_con, i_d, i_b]

                    # Chain walk (same shape as CONNECT pos chain):
                    g_anchor1_row = gs.qd_vec3(0.0, 0.0, 0.0)
                    g_anchor2_row = gs.qd_vec3(0.0, 0.0, 0.0)
                    for i_ab in range(2):
                        sign = gs.qd_float(1.0)
                        link = link1_idx
                        anchor_pos = ga1
                        if i_ab == 1:
                            sign = gs.qd_float(-1.0)
                            link = link2_idx
                            anchor_pos = ga2

                        while link > -1:
                            link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link
                            for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                                i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_

                                cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                                cdof_vel = dyn_state.dofs.cdof_vel[i_d, i_b]
                                t_pos = anchor_pos - dyn_state.links.root_COM[link, i_b]

                                g_jac_i3 = (
                                    constraint_state.dL_djac[n_con, i_d, i_b]
                                    + dL_d_jac_qvel * dyn_state.dofs.vel[i_d, i_b]
                                )

                                g_vm = gs.qd_vec3(0.0, 0.0, 0.0)
                                g_vm[i_3] = g_jac_i3 * sign

                                dyn_state.dofs.cdof_vel.grad[i_d, i_b] += g_vm
                                dyn_state.dofs.cdof_ang.grad[i_d, i_b] += t_pos.cross(g_vm)
                                g_t_pos = -cdof_ang.cross(g_vm)
                                if i_ab == 0:
                                    g_anchor1_row = g_anchor1_row + g_t_pos
                                else:
                                    g_anchor2_row = g_anchor2_row + g_t_pos
                                dyn_state.links.root_COM.grad[link, i_b] += -g_t_pos

                            link = dyn_info.links.parent_idx[link_maybe_batch]

                    # pos_error = ga1 - ga2 => direct g splits to g_ga1, g_ga2 oppositely.
                    g_ga1[i_3] = g_ga1[i_3] + g_pos_error_direct
                    g_ga2[i_3] = g_ga2[i_3] - g_pos_error_direct
                    g_ga1 = g_ga1 + g_anchor1_row
                    g_ga2 = g_ga2 + g_anchor2_row

                # Reverse the velocity-product bias jdotv = jdotv1 - jdotv2 through both chains
                g_ga1 = g_ga1 + func_equality_jdotv_bw(i_b, link1_idx, ga1, g_jdotv, dyn_state, dyn_info, rigid_config)
                g_ga2 = g_ga2 + func_equality_jdotv_bw(i_b, link2_idx, ga2, -g_jdotv, dyn_state, dyn_info, rigid_config)

                # ---- Orientation rows (3) ----
                # Direct contributions: rot_error[i_3] via ref, dL_d_imp via imp.
                for i_3 in range(3):
                    n_con = n_con_counter
                    n_con_counter = n_con_counter + 1
                    g_aref = constraint_state.dL_daref[n_con, i_b]
                    g_efc_D = constraint_state.dL_defc_D[n_con, i_b]

                    d_aref_d_jac_qvel = -b_coef
                    d_aref_d_ref_direct = -k_coef * imp
                    d_aref_d_imp = -k_coef * rot_error[i_3]
                    # Rotational bias: stored aref = aref - 0.5 * (t1 + t2 + t3)[i_3 + 1] * torquescale, the same
                    # upstream gradient reaching each of t1 / t2 / t3.
                    g_t_quat[i_3 + 1] = -0.5 * torquescale * g_aref

                    dL_d_imp_total = dL_d_imp_total + g_aref * d_aref_d_imp + g_efc_D * d_efc_D_d_imp_rot
                    dL_d_jac_qvel_orient[i_3] = g_aref * d_aref_d_jac_qvel
                    g_rot_error[i_3] = g_rot_error[i_3] + g_aref * d_aref_d_ref_direct

                    # d(jac_qvel)/d(vel[i_d]) over the forward-recorded dof support (see the collision reverse
                    # above for why the chain walk cannot accumulate this).
                    for k in range(constraint_state.jac_n_dofs[n_con, i_b]):
                        i_d = constraint_state.jac_dofs_idx[n_con, k, i_b]
                        dyn_state.dofs.vel.grad[i_d, i_b] += (
                            dL_d_jac_qvel_orient[i_3] * constraint_state.jac[n_con, i_d, i_b]
                        )

                # Orientation chain walk: per i_d on chain, build g_quat3_d from
                # the 3 orient rows, then back-prop through quat_mul/quat_mul_axis.
                g_inv_q2 = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.qd_float)
                g_q = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.qd_float)
                for i_ab in range(2):
                    sign_chain = gs.qd_float(1.0)
                    link = link1_idx
                    if i_ab == 1:
                        sign_chain = gs.qd_float(-1.0)
                        link = link2_idx

                    while link > -1:
                        link_maybe_batch = [link, i_b] if qd.static(rigid_config.batch_links_info) else link
                        for i_d_ in range(dyn_info.links.n_dofs[link_maybe_batch]):
                            i_d = dyn_info.links.dof_end[link_maybe_batch] - 1 - i_d_

                            # Build g_quat3_d (only xyz components feed jac).
                            g_quat3_d = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.qd_float)
                            for i_3 in qd.static(range(3)):
                                row = n_con_orient_base + i_3
                                gjac = (
                                    constraint_state.dL_djac[row, i_d, i_b]
                                    + dL_d_jac_qvel_orient[i_3] * dyn_state.dofs.vel[i_d, i_b]
                                )
                                # jac[row, i_d] = 0.5 * torquescale * quat3_d[i_3+1]
                                g_quat3_d[i_3 + 1] = g_quat3_d[i_3 + 1] + gjac * 0.5 * torquescale

                            # Replay quat3_d, quat2_d, jac_diff_r_d
                            cdof_ang = dyn_state.dofs.cdof_ang[i_d, i_b]
                            jac_diff_r_d = sign_chain * cdof_ang
                            quat2_d = gu.qd_quat_mul_axis(inv_q2, jac_diff_r_d)

                            # quat3_d = quat_mul(quat2_d, q_var)
                            g_quat2_d = gu.qd_quat_mul_grad_lhs(quat2_d, q_var, g_quat3_d)
                            g_q_contrib = gu.qd_quat_mul_grad_rhs(quat2_d, q_var, g_quat3_d)
                            g_q = g_q + g_q_contrib

                            # quat2_d = quat_mul_axis(inv_q2, jac_diff_r_d)
                            #        = quat_mul(inv_q2, [0, jac_diff_r_d])
                            v_padded = qd.Vector(
                                [0.0, jac_diff_r_d[0], jac_diff_r_d[1], jac_diff_r_d[2]],
                                dt=gs.qd_float,
                            )
                            g_inv_q2_contrib = gu.qd_quat_mul_grad_lhs(inv_q2, v_padded, g_quat2_d)
                            g_v_padded = gu.qd_quat_mul_grad_rhs(inv_q2, v_padded, g_quat2_d)
                            g_inv_q2 = g_inv_q2 + g_inv_q2_contrib
                            g_jac_diff_r_d = gs.qd_vec3(g_v_padded[1], g_v_padded[2], g_v_padded[3])

                            # jac_diff_r_d = sign_chain  *  cdof_ang[i_d]
                            dyn_state.dofs.cdof_ang.grad[i_d, i_b] += sign_chain * g_jac_diff_r_d

                        link = dyn_info.links.parent_idx[link_maybe_batch]

                # Via-penetration contribution (shared across all 6 rows).
                if pos_imp > EPS:
                    coef_pen = dL_d_imp_total * d_imp_d_imp_x / (width * pos_imp)
                    # pos_error part
                    for j in qd.static(range(3)):
                        g_pos_error_via_pen = coef_pen * pos_error[j]
                        g_ga1[j] = g_ga1[j] + g_pos_error_via_pen
                        g_ga2[j] = g_ga2[j] - g_pos_error_via_pen
                    # rot_error part
                    g_rot_error = g_rot_error + coef_pen * rot_error

                # rot_error = error_quat.xyz  *  torquescale
                # => g_error_quat = (0, g_rot_error * torquescale)
                g_error_quat = qd.Vector(
                    [
                        0.0,
                        g_rot_error[0] * torquescale,
                        g_rot_error[1] * torquescale,
                        g_rot_error[2] * torquescale,
                    ],
                    dt=gs.qd_float,
                )
                g_inv_q2_eq = gu.qd_quat_mul_grad_lhs(inv_q2, q_var, g_error_quat)
                g_q_eq = gu.qd_quat_mul_grad_rhs(inv_q2, q_var, g_error_quat)
                g_inv_q2 = g_inv_q2 + g_inv_q2_eq
                g_q = g_q + g_q_eq

                # ---- Rotational bias t1 + t2 + t3 (see func_equality_weld in solver.py) ----
                # Replay the bias intermediates from the same state the forward read.
                omega1 = dyn_state.links.cd_ang[link1_idx, i_b]
                omega2 = dyn_state.links.cd_ang[link2_idx, i_b]
                domega = omega1 - omega2
                p_omega1 = qd.Vector([0.0, omega1[0], omega1[1], omega1[2]], dt=gs.qd_float)
                p_omega2 = qd.Vector([0.0, omega2[0], omega2[1], omega2[2]], dt=gs.qd_float)
                p_domega = qd.Vector([0.0, domega[0], domega[1], domega[2]], dt=gs.qd_float)
                qdot_body1 = 0.5 * gu.qd_quat_mul(p_omega1, quat_body1)
                qdot0r = gu.qd_quat_mul(qdot_body1, relpose)
                qdot_body2 = 0.5 * gu.qd_quat_mul(p_omega2, quat_body2)
                inv_qdot2 = gu.qd_inv_quat(qdot_body2)
                m1 = gu.qd_quat_mul_axis(inv_qdot2, domega)
                m3 = gu.qd_quat_mul_axis(inv_q2, domega)

                g_omega1 = gs.qd_vec3(0.0, 0.0, 0.0)
                g_omega2 = gs.qd_vec3(0.0, 0.0, 0.0)
                g_domega = gs.qd_vec3(0.0, 0.0, 0.0)

                # t2 = quat_mul(quat_mul_axis(inv_q2, djrdv), q_var), with quat_mul_axis(u, a) = quat_mul(u, [0, a])
                # and djrdv = cddb1_ang - cddb2_ang the difference of the chains' angular Jdot @ qvel.
                _jdotv1, cddb1_ang = solver.func_equality_jdotv(i_b, link1_idx, ga1, dyn_state, dyn_info, rigid_config)
                _jdotv2, cddb2_ang = solver.func_equality_jdotv(i_b, link2_idx, ga2, dyn_state, dyn_info, rigid_config)
                djrdv = cddb1_ang - cddb2_ang
                p_djrdv = qd.Vector([0.0, djrdv[0], djrdv[1], djrdv[2]], dt=gs.qd_float)
                m2 = gu.qd_quat_mul_axis(inv_q2, djrdv)
                g_m2 = gu.qd_quat_mul_grad_lhs(m2, q_var, g_t_quat)
                g_q = g_q + gu.qd_quat_mul_grad_rhs(m2, q_var, g_t_quat)
                g_inv_q2 = g_inv_q2 + gu.qd_quat_mul_grad_lhs(inv_q2, p_djrdv, g_m2)
                g_p_djrdv = gu.qd_quat_mul_grad_rhs(inv_q2, p_djrdv, g_m2)
                g_djrdv = gs.qd_vec3(g_p_djrdv[1], g_p_djrdv[2], g_p_djrdv[3])
                func_cddb_ang_bw(i_b, link1_idx, g_djrdv, dyn_state, dyn_info, rigid_config)
                func_cddb_ang_bw(i_b, link2_idx, -g_djrdv, dyn_state, dyn_info, rigid_config)

                # t3 = quat_mul(quat_mul_axis(inv_q2, domega), qdot0r)
                g_m3 = gu.qd_quat_mul_grad_lhs(m3, qdot0r, g_t_quat)
                g_qdot0r = gu.qd_quat_mul_grad_rhs(m3, qdot0r, g_t_quat)
                g_inv_q2 = g_inv_q2 + gu.qd_quat_mul_grad_lhs(inv_q2, p_domega, g_m3)
                g_p_domega3 = gu.qd_quat_mul_grad_rhs(inv_q2, p_domega, g_m3)
                for j in qd.static(range(3)):
                    g_domega[j] = g_domega[j] + g_p_domega3[j + 1]
                # qdot0r = quat_mul(qdot_body1, relpose); relpose is a model param
                g_qdot_body1 = gu.qd_quat_mul_grad_lhs(qdot_body1, relpose, g_qdot0r)
                # qdot_body1 = 0.5 * quat_mul([0, omega1], quat_body1)
                g_p_omega1 = gu.qd_quat_mul_grad_lhs(p_omega1, quat_body1, 0.5 * g_qdot_body1)
                g_quat1_bias = gu.qd_quat_mul_grad_rhs(p_omega1, quat_body1, 0.5 * g_qdot_body1)
                for j in qd.static(range(3)):
                    g_omega1[j] = g_omega1[j] + g_p_omega1[j + 1]

                # t1 = quat_mul(quat_mul_axis(inv_quat(qdot_body2), domega), q_var)
                g_m1 = gu.qd_quat_mul_grad_lhs(m1, q_var, g_t_quat)
                g_q = g_q + gu.qd_quat_mul_grad_rhs(m1, q_var, g_t_quat)
                g_inv_qdot2 = gu.qd_quat_mul_grad_lhs(inv_qdot2, p_domega, g_m1)
                g_p_domega1 = gu.qd_quat_mul_grad_rhs(inv_qdot2, p_domega, g_m1)
                for j in qd.static(range(3)):
                    g_domega[j] = g_domega[j] + g_p_domega1[j + 1]
                # inv_quat flips the xyz signs of the incoming gradient
                g_qdot_body2 = qd.Vector(
                    [g_inv_qdot2[0], -g_inv_qdot2[1], -g_inv_qdot2[2], -g_inv_qdot2[3]], dt=gs.qd_float
                )
                # qdot_body2 = 0.5 * quat_mul([0, omega2], quat_body2)
                g_p_omega2 = gu.qd_quat_mul_grad_lhs(p_omega2, quat_body2, 0.5 * g_qdot_body2)
                g_quat2_bias = gu.qd_quat_mul_grad_rhs(p_omega2, quat_body2, 0.5 * g_qdot_body2)
                for j in qd.static(range(3)):
                    g_omega2[j] = g_omega2[j] + g_p_omega2[j + 1]

                # domega = omega1 - omega2; omega_c = links.cd_ang[link_c]
                g_omega1 = g_omega1 + g_domega
                g_omega2 = g_omega2 - g_domega
                dyn_state.links.cd_ang.grad[link1_idx, i_b] += g_omega1
                dyn_state.links.cd_ang.grad[link2_idx, i_b] += g_omega2

                # inv_q2 = inv_quat(quat_body2): (w, x, y, z) -> (w, -x, -y, -z)
                # => g_quat_body2 (from inv_q2 chain) = (g_inv_q2[0], -g_inv_q2[1], -g_inv_q2[2], -g_inv_q2[3])
                g_quat2_from_inv = qd.Vector(
                    [g_inv_q2[0], -g_inv_q2[1], -g_inv_q2[2], -g_inv_q2[3]],
                    dt=gs.qd_float,
                )

                # q_var = quat_mul(quat_body1, relpose); relpose const => drop g_v.
                g_quat1_from_q = gu.qd_quat_mul_grad_lhs(quat_body1, relpose, g_q)

                # Anchor chain ga1, ga2 -> dyn_state.links.{pos, quat}.
                g_quat1_anchor = gu.qd_transform_by_quat_grad_quat(anchor1_local, quat_body1, g_ga1)
                g_quat2_anchor = gu.qd_transform_by_quat_grad_quat(anchor2_local, quat_body2, g_ga2)

                dyn_state.links.pos.grad[link1_idx, i_b] += g_ga1
                dyn_state.links.pos.grad[link2_idx, i_b] += g_ga2
                dyn_state.links.quat.grad[link1_idx, i_b] += g_quat1_anchor + g_quat1_from_q + g_quat1_bias
                dyn_state.links.quat.grad[link2_idx, i_b] += g_quat2_anchor + g_quat2_from_inv + g_quat2_bias
