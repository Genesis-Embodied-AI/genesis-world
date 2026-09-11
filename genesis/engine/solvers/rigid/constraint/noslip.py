import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class


@qd.func
def func_solve_mass_block(i_d0, i_b, vec: qd.Tensor, rigid_info: array_class.RigidInfo):
    """LDL^T forward-backward substitution on vec, restricted to the mass block containing dof i_d0.

    The factor is block-diagonal per mass block (see dofs_mass_block_start in array_class.py), with constant block
    bounds shared by every member dof, so a block is solvable independently from any member dof.
    """
    block_start = rigid_info.dofs_mass_block_start[i_d0]
    block_end = rigid_info.dofs_mass_block_end[i_d0]

    # Step 1: Solve w s.t. L^T @ w = y (backward substitution)
    for i_d_ in range(block_end - block_start):
        i_d = block_end - i_d_ - 1
        curr = vec[i_d, i_b]
        for j_d in range(i_d + 1, block_end):
            curr = curr - rigid_info.mass_mat_L[j_d, i_d, i_b] * vec[j_d, i_b]
        vec[i_d, i_b] = curr

    # Step 2: z = D^{-1} @ w
    for i_d in range(block_start, block_end):
        vec[i_d, i_b] = vec[i_d, i_b] * rigid_info.mass_mat_D_inv[i_d, i_b]

    # Step 3: Solve x s.t. L @ x = z (forward substitution)
    for i_d in range(block_start, block_end):
        curr = vec[i_d, i_b]
        for j_d in range(block_start, i_d):
            curr = curr - rigid_info.mass_mat_L[i_d, j_d, i_b] * vec[j_d, i_b]
        vec[i_d, i_b] = curr


@qd.func
def func_apply_Minv_rows(
    i_row_0,
    i_row_1,
    i_b,
    jac_dofs_idx: qd.Tensor,
    coef_0,
    coef_1,
    vec: qd.Tensor,
    jac: qd.Tensor,
    jac_n_dofs: qd.Tensor,
    rigid_info: array_class.RigidInfo,
):
    """Compute vec = M^{-1} (coef_0 * J[i_row_0]^T + coef_1 * J[i_row_1]^T) over the touched mass blocks.

    Both rows must share the same dof support (e.g. the two edges of a friction-pyramid pair); the walk is driven by
    i_row_0. The mass matrix is block-diagonal per kinematic tree, so scattering into fully-zeroed blocks and solving
    only those blocks is exact. Row dofs are sorted, so same-block dofs are contiguous and each block is visited once
    per pass. Working at block granularity (not entity granularity) keeps the touched dof range within the row's own
    island, which makes concurrent per-island sweeps of the same env race-free.
    """
    tree_start_prev = gs.qd_int(-1)
    for i_d_ in range(jac_n_dofs[i_row_0, i_b]):
        i_d = jac_dofs_idx[i_row_0, i_d_, i_b]
        block_start = rigid_info.dofs_mass_block_start[i_d]
        if block_start != tree_start_prev:
            for j_d in range(block_start, rigid_info.dofs_mass_block_end[i_d]):
                vec[j_d, i_b] = gs.qd_float(0.0)
            tree_start_prev = block_start
        vec[i_d, i_b] = coef_0 * jac[i_row_0, i_d, i_b] + coef_1 * jac[i_row_1, i_d, i_b]

    tree_start_prev = gs.qd_int(-1)
    for i_d_ in range(jac_n_dofs[i_row_0, i_b]):
        i_d = jac_dofs_idx[i_row_0, i_d_, i_b]
        block_start = rigid_info.dofs_mass_block_start[i_d]
        if block_start != tree_start_prev:
            func_solve_mass_block(i_d, i_b, vec, rigid_info)
            tree_start_prev = block_start


@qd.func
def func_accumulate_row_blocks(
    i_row,
    i_b,
    jac_dofs_idx: qd.Tensor,
    vec_src: qd.Tensor,
    vec_dst: qd.Tensor,
    jac_n_dofs: qd.Tensor,
    rigid_info: array_class.RigidInfo,
):
    """Add vec_src to vec_dst over the mass blocks touched by constraint row i_row."""
    tree_start_prev = gs.qd_int(-1)
    for i_d_ in range(jac_n_dofs[i_row, i_b]):
        i_d = jac_dofs_idx[i_row, i_d_, i_b]
        block_start = rigid_info.dofs_mass_block_start[i_d]
        if block_start != tree_start_prev:
            for j_d in range(block_start, rigid_info.dofs_mass_block_end[i_d]):
                vec_dst[j_d, i_b] = vec_dst[j_d, i_b] + vec_src[j_d, i_b]
            tree_start_prev = block_start


@qd.func
def func_dot_row(i_row, i_b, jac_dofs_idx: qd.Tensor, vec: qd.Tensor, jac: qd.Tensor, jac_n_dofs: qd.Tensor):
    """Sparse dot product J[i_row] * vec over the row dof support."""
    s = gs.qd_float(0.0)
    for i_d_ in range(jac_n_dofs[i_row, i_b]):
        i_d = jac_dofs_idx[i_row, i_d_, i_b]
        s += jac[i_row, i_d, i_b] * vec[i_d, i_b]
    return s


@qd.func
def func_refresh_qacc_batch(
    i_b,
    i_island,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Recompute qacc = acc_smooth + M^{-1} J^T f from the current constraint forces, over one island.

    The force-update sweep maintains qacc incrementally; recomputing it exactly at the start of every iteration keeps
    the accumulated floating-point drift bounded to a single sweep. Only the island's own dofs and constraint rows are
    visited (mass blocks never straddle islands, and each block's first dof appears exactly once in the island's dof
    list regardless of the skyline dof reorder).
    """
    n_dofs = constraint_state.island.dof_slices.n[i_island, i_b]
    n_rows = constraint_state.island.constraint_slices.n[i_island, i_b]
    dof_start = constraint_state.island.dof_slices.start[i_island, i_b]
    row_start = constraint_state.island.constraint_slices.start[i_island, i_b]

    for i_d_ in range(n_dofs):
        i_d = constraint_state.island.dof_id[dof_start + i_d_, i_b]
        constraint_state.qfrc_constraint[i_d, i_b] = gs.qd_float(0.0)
        constraint_state.qacc[i_d, i_b] = gs.qd_float(0.0)

    for i_c_ in range(n_rows):
        i_c = constraint_state.island.constraint_id[row_start + i_c_, i_b]
        force = constraint_state.efc_force[i_c, i_b]
        for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
            i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = (
                constraint_state.qfrc_constraint[i_d, i_b] + constraint_state.jac[i_c, i_d, i_b] * force
            )

    for i_d_ in range(n_dofs):
        i_d = constraint_state.island.dof_id[dof_start + i_d_, i_b]
        # Solve each mass block once, when visiting its first dof (order-robust, unlike previous-block tracking,
        # since the island dof list may be permuted by the fill-reducing reorder).
        if i_d == rigid_info.dofs_mass_block_start[i_d]:
            constraint_state.qacc[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
            for j_d in range(i_d + 1, rigid_info.dofs_mass_block_end[i_d]):
                constraint_state.qacc[j_d, i_b] = constraint_state.qfrc_constraint[j_d, i_b]
            func_solve_mass_block(i_d, i_b, constraint_state.qacc, rigid_info)

    for i_d_ in range(n_dofs):
        i_d = constraint_state.island.dof_id[dof_start + i_d_, i_b]
        constraint_state.qacc[i_d, i_b] = constraint_state.qacc[i_d, i_b] + dyn_state.dofs.acc_smooth[i_d, i_b]


@qd.func
def func_noslip_batch(
    i_b,
    i_island,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Matrix-free noslip force-update sweep over one island.

    The dual residual of row r is res_r = A f + b = J_r * qacc - aref_r with qacc = acc_smooth + M^{-1} J^T f, so
    the sweep maintains qacc instead of materializing the dense dual matrix AR = J M^{-1} J^T: each force update
    propagates to qacc through an M^{-1} solve restricted to the mass blocks the row touches, and the 1x1/2x2
    diagonal blocks of A needed by the updates are recomputed on the fly the same way. This keeps the pass linear in
    the number of constraints (times the row support size) per iteration, instead of quadratic. A is block-diagonal
    by island and the updates touch only the row's own mass blocks, so concurrent sweeps of different islands of the
    same env are race-free and equivalent to the env-wide sweep.
    """
    EPS = rigid_info.EPS[None]

    # temp variables
    res = qd.Vector.zero(gs.qd_float, 2)
    old_force = qd.Vector.zero(gs.qd_float, 2)
    bc = qd.Vector.zero(gs.qd_float, 2)
    Ac = qd.Vector.zero(gs.qd_float, 4)

    n_dofs = constraint_state.qfrc_constraint.shape[0]
    ne = constraint_state.n_constraints_equality[i_b]
    nf = constraint_state.n_constraints_frictionloss[i_b]
    const_start = ne + nf
    const_end = const_start + qd.static(rigid_config.rows_per_contact) * collider_state.n_contacts[i_b]

    n_dofs = constraint_state.island.dof_slices.n[i_island, i_b]
    n_rows = constraint_state.island.constraint_slices.n[i_island, i_b]
    row_start = constraint_state.island.constraint_slices.start[i_island, i_b]

    scale = 1.0 / (rigid_info.meaninertia[i_b] * qd.max(1.0, n_dofs))

    for i_iter in range(rigid_info.noslip_iterations[None]):
        func_refresh_qacc_batch(i_b, i_island, dyn_state, constraint_state, rigid_info, rigid_config)

        improvement = gs.qd_float(0.0)

        # Sweep the island's constraint rows in ascending order (the per-island grouping is index-ordered):
        # dry-friction (dof frictionloss) rows get a 1-dof update, and every other collision row is the base of an
        # opposing pyramid-edge pair (j_efc, j_efc + 1) projected with the normal force fixed. Equality and joint
        # limit rows only contribute to the iter-0 improvement correction.
        for i_c_ in range(n_rows):
            i_c = constraint_state.island.constraint_id[row_start + i_c_, i_b]

            if i_iter == 0:
                improvement += 0.5 * constraint_state.efc_force[i_c, i_b] ** 2 * constraint_state.diag[i_c, i_b]

            if i_c >= ne and i_c < ne + nf:
                # Each row runs two phases through the single func_apply_Minv_rows call site: phase 0 computes the
                # diagonal entry A[i_c, i_c] = J M^{-1} J^T and updates the force, phase 1 propagates the force
                # change to qacc (skipped when the force did not move).
                delta = gs.qd_float(0.0)
                for i_phase in range(2):
                    coef = gs.qd_float(1.0)
                    if i_phase == 1:
                        coef = delta
                    if i_phase == 0 or delta != 0.0:
                        func_apply_Minv_rows(
                            i_c,
                            i_c,
                            i_b,
                            constraint_state.jac_dofs_idx,
                            coef,
                            0.0,
                            constraint_state.Mgrad,
                            constraint_state.jac,
                            constraint_state.jac_n_dofs,
                            rigid_info,
                        )
                        if i_phase == 0:
                            A_diag = func_dot_row(
                                i_c,
                                i_b,
                                constraint_state.jac_dofs_idx,
                                constraint_state.Mgrad,
                                constraint_state.jac,
                                constraint_state.jac_n_dofs,
                            )
                            res[0] = (
                                func_dot_row(
                                    i_c,
                                    i_b,
                                    constraint_state.jac_dofs_idx,
                                    constraint_state.qacc,
                                    constraint_state.jac,
                                    constraint_state.jac_n_dofs,
                                )
                                - constraint_state.aref[i_c, i_b]
                            )

                            old_force[0] = constraint_state.efc_force[i_c, i_b]
                            constraint_state.efc_force[i_c, i_b] -= res[0] / A_diag
                            if constraint_state.efc_force[i_c, i_b] < -constraint_state.efc_frictionloss[i_c, i_b]:
                                constraint_state.efc_force[i_c, i_b] = -constraint_state.efc_frictionloss[i_c, i_b]
                            elif constraint_state.efc_force[i_c, i_b] > constraint_state.efc_frictionloss[i_c, i_b]:
                                constraint_state.efc_force[i_c, i_b] = constraint_state.efc_frictionloss[i_c, i_b]
                            delta = constraint_state.efc_force[i_c, i_b] - old_force[0]
                            improvement -= 0.5 * delta**2 * A_diag + delta * res[0]
                        else:
                            func_accumulate_row_blocks(
                                i_c,
                                i_b,
                                constraint_state.jac_dofs_idx,
                                constraint_state.Mgrad,
                                constraint_state.qacc,
                                constraint_state.jac_n_dofs,
                                rigid_info,
                            )
            elif i_c >= const_start and i_c < const_end and (i_c - const_start) % 2 == 0:
                j_efc = i_c

                # Three phases through the single func_apply_Minv_rows call site: phases 0 and 1 compute the
                # symmetric 2x2 block of A (both rows share the same dof support, so two block solves and three
                # sparse dots suffice), then the force update runs at the end of phase 1, and phase 2 propagates the
                # force change to qacc (skipped when the forces did not move).
                delta_0 = gs.qd_float(0.0)
                delta_1 = gs.qd_float(0.0)
                for i_phase in range(3):
                    coef_0 = gs.qd_float(0.0)
                    coef_1 = gs.qd_float(0.0)
                    if i_phase == 0:
                        coef_0 = 1.0
                    elif i_phase == 1:
                        coef_1 = 1.0
                    else:
                        coef_0 = delta_0
                        coef_1 = delta_1
                    if i_phase < 2 or delta_0 != 0.0 or delta_1 != 0.0:
                        func_apply_Minv_rows(
                            j_efc,
                            j_efc + 1,
                            i_b,
                            constraint_state.jac_dofs_idx,
                            coef_0,
                            coef_1,
                            constraint_state.Mgrad,
                            constraint_state.jac,
                            constraint_state.jac_n_dofs,
                            rigid_info,
                        )
                        if i_phase == 2:
                            func_accumulate_row_blocks(
                                j_efc,
                                i_b,
                                constraint_state.jac_dofs_idx,
                                constraint_state.Mgrad,
                                constraint_state.qacc,
                                constraint_state.jac_n_dofs,
                                rigid_info,
                            )
                        else:
                            for i2 in qd.static(range(2)):
                                if i_phase == 0 or i2 == 1:
                                    s = func_dot_row(
                                        j_efc + i2,
                                        i_b,
                                        constraint_state.jac_dofs_idx,
                                        constraint_state.Mgrad,
                                        constraint_state.jac,
                                        constraint_state.jac_n_dofs,
                                    )
                                    if i_phase == 0:
                                        Ac[i2] = s
                                    else:
                                        Ac[3] = s

                    if i_phase == 1:
                        Ac[2] = Ac[1]
                        for i2 in qd.static(range(2)):
                            res[i2] = (
                                func_dot_row(
                                    j_efc + i2,
                                    i_b,
                                    constraint_state.jac_dofs_idx,
                                    constraint_state.qacc,
                                    constraint_state.jac,
                                    constraint_state.jac_n_dofs,
                                )
                                - constraint_state.aref[j_efc + i2, i_b]
                            )
                            old_force[i2] = constraint_state.efc_force[j_efc + i2, i_b]

                        for j in qd.static(range(2)):
                            bc[j] = res[j]
                            for k in qd.static(range(2)):
                                bc[j] -= Ac[j * 2 + k] * old_force[k]
                        mid = 0.5 * (
                            constraint_state.efc_force[j_efc, i_b] + constraint_state.efc_force[j_efc + 1, i_b]
                        )
                        y = 0.5 * (constraint_state.efc_force[j_efc, i_b] - constraint_state.efc_force[j_efc + 1, i_b])
                        K1 = Ac[0] + Ac[3] - Ac[1] - Ac[2]
                        K0 = mid * (Ac[0] - Ac[3]) + bc[0] - bc[1]
                        if K1 < EPS:
                            constraint_state.efc_force[j_efc, i_b] = constraint_state.efc_force[j_efc + 1, i_b] = mid
                        else:
                            y = -K0 / K1
                            if y < -mid:
                                constraint_state.efc_force[j_efc, i_b] = 0
                                constraint_state.efc_force[j_efc + 1, i_b] = 2 * mid
                            elif y > mid:
                                constraint_state.efc_force[j_efc, i_b] = 2 * mid
                                constraint_state.efc_force[j_efc + 1, i_b] = 0
                            else:
                                constraint_state.efc_force[j_efc, i_b] = mid + y
                                constraint_state.efc_force[j_efc + 1, i_b] = mid - y
                        cost_change = func_cost_change(
                            i_b, j_efc, Ac, old_force, res, EPS, constraint_state.efc_force, 2
                        )

                        improvement -= cost_change

                        delta_0 = constraint_state.efc_force[j_efc, i_b] - old_force[0]
                        delta_1 = constraint_state.efc_force[j_efc + 1, i_b] - old_force[1]

        improvement *= scale

        if improvement < rigid_info.noslip_tolerance[None]:
            break


@qd.func
def func_dual_finish_batch(
    i_b,
    i_island,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Map the final constraint forces back to joint space over one island.

    The refresh recomputes qfrc_constraint = J^T f and qacc = acc_smooth + M^{-1} J^T f exactly from the swept
    forces; the remaining work is copying them into the per-dof state.
    """
    func_refresh_qacc_batch(i_b, i_island, dyn_state, constraint_state, rigid_info, rigid_config)

    n_dofs = constraint_state.island.dof_slices.n[i_island, i_b]
    dof_start = constraint_state.island.dof_slices.start[i_island, i_b]

    for i_d_ in range(n_dofs):
        i_d = constraint_state.island.dof_id[dof_start + i_d_, i_b]
        dyn_state.dofs.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        dyn_state.dofs.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dyn_state.dofs.force[i_d, i_b] = dyn_state.dofs.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]


@qd.kernel(fastcache=True)
def kernel_noslip(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Noslip pass: matrix-free force-update sweep followed by the dual finish, fused per island.

    The sweep is a sequential Gauss-Seidel process within an island; islands are independent (A is block-diagonal by
    island and both phases touch only the island's own rows and dofs), so each (env, island) pair runs sweep and
    finish end-to-end in one thread.
    """
    _B = constraint_state.jac.shape[2]

    # max_islands bounds the per-env island count (at most one island per tree); the guard skips the unused tail.
    # Iterate islands-major so that consecutive GPU lanes sweep the same island index across consecutive envs: envs
    # are replicas of one scene, so lanes execute identical control flow (island sizes match) and the batch-contiguous
    # field reads coalesce, instead of adjacent lanes diverging on different islands of one env.
    max_islands = constraint_state.island.dof_slices.start.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_island, i_b in qd.ndrange(max_islands, _B):
        if i_island < constraint_state.island.n_islands[i_b]:
            run_island = True
            if qd.static(rigid_config.use_hibernation):
                run_island = not constraint_state.island.is_hibernated[i_island, i_b]
            if run_island:
                func_noslip_batch(i_b, i_island, dyn_state, collider_state, constraint_state, rigid_info, rigid_config)
                func_dual_finish_batch(i_b, i_island, dyn_state, constraint_state, rigid_info, rigid_config)


@qd.func
def func_cost_change(i_b: int, force_start: int, Ac, old_force, res, eps, force: qd.Tensor, dim: int):
    change = gs.qd_float(0.0)
    if dim == 1:
        delta = force[force_start, i_b] - old_force[0]
        change = 0.5 * Ac[0] * delta * delta + delta * res[0]
    else:
        delta = qd.Vector.zero(gs.qd_float, 2)
        for i in range(dim):
            delta[i] = force[force_start + i, i_b] - old_force[i]
        for i in range(dim):
            for j in range(dim):
                change += 0.5 * Ac[i * dim + j] * delta[i] * delta[j]
            change += delta[i] * res[i]
    if change > eps:
        for i in range(dim):
            force[force_start + i, i_b] = old_force[i]
        change = 0.0
    return change
