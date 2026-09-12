import os

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

# Warp width of the colored noslip kernel (lanes per env). MUST match the lane dim of ConstraintState.noslip_minv.
NOSLIP_COOP_T = 32

# Iteration-count override for the colored sweep (0 = use the scene's noslip_iterations).
NOSLIP_COOP_ITERS = int(os.environ.get("GS_NOSLIP_COOP_ITERS", "0"))

# NOSLIP_COLOR_MAXC caps the greedy coloring's color count (first axis of ConstraintState.color_block_used); a scene
# needing more colors falls back to a serial single-lane sweep for that env (correct, slower), so this is a perf knob,
# not a correctness bound. These bake in as qd.static, so change them only with a dedicated compile cache
# (QD_OFFLINE_CACHE_FILE_PATH).
NOSLIP_COLOR_OMEGA = float(os.environ.get("GS_NOSLIP_COLOR_OMEGA", "1.0"))
NOSLIP_COLOR_MAXC = int(os.environ.get("GS_NOSLIP_COLOR_MAXC", "64"))


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


# ======================================================================================================================
# Per-lane building blocks for the warp-per-env colored sweep below. Each mirrors a scalar helper (func_solve_mass_block
# / func_apply_Minv_rows / ...) but writes the per-lane scratch noslip_minv[i_d, i_lane, i_b], whose lane dimension lets
# concurrent rows compute their own M^-1 J^T without clobbering a shared buffer.
# ======================================================================================================================


@qd.func
def func_solve_mass_block_lane(i_d0, i_b, i_lane, vec: qd.Tensor, rigid_info: array_class.RigidInfo):
    """func_solve_mass_block on per-lane scratch vec[i_d, i_lane, i_b]."""
    block_start = rigid_info.dofs_mass_block_start[i_d0]
    block_end = rigid_info.dofs_mass_block_end[i_d0]
    for i_d_ in range(block_end - block_start):
        i_d = block_end - i_d_ - 1
        curr = vec[i_d, i_lane, i_b]
        for j_d in range(i_d + 1, block_end):
            curr = curr - rigid_info.mass_mat_L[j_d, i_d, i_b] * vec[j_d, i_lane, i_b]
        vec[i_d, i_lane, i_b] = curr
    for i_d in range(block_start, block_end):
        vec[i_d, i_lane, i_b] = vec[i_d, i_lane, i_b] * rigid_info.mass_mat_D_inv[i_d, i_b]
    for i_d in range(block_start, block_end):
        curr = vec[i_d, i_lane, i_b]
        for j_d in range(block_start, i_d):
            curr = curr - rigid_info.mass_mat_L[i_d, j_d, i_b] * vec[j_d, i_lane, i_b]
        vec[i_d, i_lane, i_b] = curr


@qd.func
def func_apply_Minv_rows_lane(
    i_row_0,
    i_row_1,
    i_b,
    i_lane,
    jac_dofs_idx: qd.Tensor,
    coef_0,
    coef_1,
    vec: qd.Tensor,
    jac: qd.Tensor,
    jac_n_dofs: qd.Tensor,
    rigid_info: array_class.RigidInfo,
):
    """func_apply_Minv_rows writing into per-lane scratch vec[i_d, i_lane, i_b]."""
    tree_start_prev = gs.qd_int(-1)
    for i_d_ in range(jac_n_dofs[i_row_0, i_b]):
        i_d = jac_dofs_idx[i_row_0, i_d_, i_b]
        block_start = rigid_info.dofs_mass_block_start[i_d]
        if block_start != tree_start_prev:
            for j_d in range(block_start, rigid_info.dofs_mass_block_end[i_d]):
                vec[j_d, i_lane, i_b] = gs.qd_float(0.0)
            tree_start_prev = block_start
        vec[i_d, i_lane, i_b] = coef_0 * jac[i_row_0, i_d, i_b] + coef_1 * jac[i_row_1, i_d, i_b]

    tree_start_prev = gs.qd_int(-1)
    for i_d_ in range(jac_n_dofs[i_row_0, i_b]):
        i_d = jac_dofs_idx[i_row_0, i_d_, i_b]
        block_start = rigid_info.dofs_mass_block_start[i_d]
        if block_start != tree_start_prev:
            func_solve_mass_block_lane(i_d, i_b, i_lane, vec, rigid_info)
            tree_start_prev = block_start


@qd.func
def func_dot_row_lane(
    i_row, i_b, i_lane, jac_dofs_idx: qd.Tensor, vec: qd.Tensor, jac: qd.Tensor, jac_n_dofs: qd.Tensor
):
    """Sparse dot J[i_row] . vec[:, i_lane, :] over the row dof support (vec has a lane dim)."""
    s = gs.qd_float(0.0)
    for i_d_ in range(jac_n_dofs[i_row, i_b]):
        i_d = jac_dofs_idx[i_row, i_d_, i_b]
        s += jac[i_row, i_d, i_b] * vec[i_d, i_lane, i_b]
    return s


@qd.func
def func_accumulate_row_blocks_lane(
    i_row,
    i_b,
    i_lane,
    scale,
    jac_dofs_idx: qd.Tensor,
    vec_src: qd.Tensor,
    vec_dst: qd.Tensor,
    jac_n_dofs: qd.Tensor,
    rigid_info: array_class.RigidInfo,
):
    """vec_dst[block, i_b] += scale * vec_src[block, i_lane, i_b] over the mass blocks row i_row touches.

    Same-color rows touch disjoint mass blocks, so concurrent lanes write disjoint vec_dst entries (race-free)."""
    tree_start_prev = gs.qd_int(-1)
    for i_d_ in range(jac_n_dofs[i_row, i_b]):
        i_d = jac_dofs_idx[i_row, i_d_, i_b]
        block_start = rigid_info.dofs_mass_block_start[i_d]
        if block_start != tree_start_prev:
            for j_d in range(block_start, rigid_info.dofs_mass_block_end[i_d]):
                vec_dst[j_d, i_b] = vec_dst[j_d, i_b] + scale * vec_src[j_d, i_lane, i_b]
            tree_start_prev = block_start


@qd.func
def func_coop_zero_qfrc(tid, T, i_b, n_dofs, constraint_state: array_class.ConstraintState):
    i_d = tid
    while i_d < n_dofs:
        constraint_state.qfrc_constraint[i_d, i_b] = gs.qd_float(0.0)
        i_d += T


@qd.func
def func_coop_accum_qfrc(tid, T, i_b, n_rows, constraint_state: array_class.ConstraintState):
    """qfrc_constraint += J^T f, on a single lane. qfrc_constraint is batch-first under the cooperative layout, where an
    atomic scatter mis-addresses, so this accumulate is serialized (the costly M^-1 blocks stay cooperative)."""
    if tid == 0:
        for i_c in range(n_rows):
            force = constraint_state.efc_force[i_c, i_b]
            for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                constraint_state.qfrc_constraint[i_d, i_b] = (
                    constraint_state.qfrc_constraint[i_d, i_b] + constraint_state.jac[i_c, i_d, i_b] * force
                )


@qd.func
def func_coop_solve_qacc(tid, T, i_b, n_dofs, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo):
    """qacc = M^-1 qfrc_constraint, cooperatively over mass blocks (each block owned by its first dof's lane)."""
    i_d = tid
    while i_d < n_dofs:
        if i_d == rigid_info.dofs_mass_block_start[i_d]:
            constraint_state.qacc[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
            for j_d in range(i_d + 1, rigid_info.dofs_mass_block_end[i_d]):
                constraint_state.qacc[j_d, i_b] = constraint_state.qfrc_constraint[j_d, i_b]
            func_solve_mass_block(i_d, i_b, constraint_state.qacc, rigid_info)
        i_d += T


@qd.func
def func_coop_add_smooth(tid, T, i_b, n_dofs, constraint_state: array_class.ConstraintState, dyn_state: array_class.DynState):
    i_d = tid
    while i_d < n_dofs:
        constraint_state.qacc[i_d, i_b] = constraint_state.qacc[i_d, i_b] + dyn_state.dofs.acc_smooth[i_d, i_b]
        i_d += T


@qd.func
def func_coop_finish_copy(tid, T, i_b, n_dofs, constraint_state: array_class.ConstraintState, dyn_state: array_class.DynState):
    i_d = tid
    while i_d < n_dofs:
        constraint_state.qacc[i_d, i_b] = constraint_state.qacc[i_d, i_b] + dyn_state.dofs.acc_smooth[i_d, i_b]
        dyn_state.dofs.acc[i_d, i_b] = constraint_state.qacc[i_d, i_b]
        dyn_state.dofs.qf_constraint[i_d, i_b] = constraint_state.qfrc_constraint[i_d, i_b]
        dyn_state.dofs.force[i_d, i_b] = (
            dyn_state.dofs.qf_smooth[i_d, i_b] + constraint_state.qfrc_constraint[i_d, i_b]
        )
        i_d += T


@qd.func
def func_noslip_update_frictionloss_lane(i_c, i_b, i_lane, omega, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo):
    """1x1 dry-friction update: A_diag = J M^-1 J^T, res = J qacc - aref, project into [-fl, fl]. omega under-relaxes
    (omega=1 = the exact projected step)."""
    func_apply_Minv_rows_lane(
        i_c, i_c, i_b, i_lane, constraint_state.jac_dofs_idx, 1.0, 0.0,
        constraint_state.noslip_minv, constraint_state.jac, constraint_state.jac_n_dofs, rigid_info,
    )
    A_diag = func_dot_row_lane(
        i_c, i_b, i_lane, constraint_state.jac_dofs_idx, constraint_state.noslip_minv,
        constraint_state.jac, constraint_state.jac_n_dofs,
    )
    res = (
        func_dot_row(i_c, i_b, constraint_state.jac_dofs_idx, constraint_state.qacc, constraint_state.jac, constraint_state.jac_n_dofs)
        - constraint_state.aref[i_c, i_b]
    )
    old = constraint_state.efc_force[i_c, i_b]
    f = old - res / A_diag
    fl = constraint_state.efc_frictionloss[i_c, i_b]
    if f < -fl:
        f = -fl
    elif f > fl:
        f = fl
    constraint_state.efc_force[i_c, i_b] = old + omega * (f - old)


@qd.func
def func_noslip_update_collision_pair_lane(j_efc, i_b, i_lane, EPS, omega, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo):
    """2x2 pyramid-pair update: symmetric block A recomputed matrix-free, forces projected onto the pyramid. omega
    under-relaxes (omega=1 = the exact projected step)."""
    Ac = qd.Vector.zero(gs.qd_float, 4)
    res = qd.Vector.zero(gs.qd_float, 2)
    old_force = qd.Vector.zero(gs.qd_float, 2)
    bc = qd.Vector.zero(gs.qd_float, 2)

    # Ac[0] = J0 M^-1 J0^T, Ac[1] = Ac[2] = J1 M^-1 J0^T, Ac[3] = J1 M^-1 J1^T.
    func_apply_Minv_rows_lane(
        j_efc, j_efc + 1, i_b, i_lane, constraint_state.jac_dofs_idx, 1.0, 0.0,
        constraint_state.noslip_minv, constraint_state.jac, constraint_state.jac_n_dofs, rigid_info,
    )
    Ac[0] = func_dot_row_lane(j_efc, i_b, i_lane, constraint_state.jac_dofs_idx, constraint_state.noslip_minv, constraint_state.jac, constraint_state.jac_n_dofs)
    Ac[1] = func_dot_row_lane(j_efc + 1, i_b, i_lane, constraint_state.jac_dofs_idx, constraint_state.noslip_minv, constraint_state.jac, constraint_state.jac_n_dofs)
    func_apply_Minv_rows_lane(
        j_efc, j_efc + 1, i_b, i_lane, constraint_state.jac_dofs_idx, 0.0, 1.0,
        constraint_state.noslip_minv, constraint_state.jac, constraint_state.jac_n_dofs, rigid_info,
    )
    Ac[3] = func_dot_row_lane(j_efc + 1, i_b, i_lane, constraint_state.jac_dofs_idx, constraint_state.noslip_minv, constraint_state.jac, constraint_state.jac_n_dofs)
    Ac[2] = Ac[1]

    for i2 in qd.static(range(2)):
        res[i2] = (
            func_dot_row(j_efc + i2, i_b, constraint_state.jac_dofs_idx, constraint_state.qacc, constraint_state.jac, constraint_state.jac_n_dofs)
            - constraint_state.aref[j_efc + i2, i_b]
        )
        old_force[i2] = constraint_state.efc_force[j_efc + i2, i_b]

    for j in qd.static(range(2)):
        bc[j] = res[j]
        for k in qd.static(range(2)):
            bc[j] -= Ac[j * 2 + k] * old_force[k]
    mid = 0.5 * (constraint_state.efc_force[j_efc, i_b] + constraint_state.efc_force[j_efc + 1, i_b])
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
    # Under-relax toward the pre-update forces before the cost-change safeguard evaluates the step.
    constraint_state.efc_force[j_efc, i_b] = old_force[0] + omega * (constraint_state.efc_force[j_efc, i_b] - old_force[0])
    constraint_state.efc_force[j_efc + 1, i_b] = old_force[1] + omega * (constraint_state.efc_force[j_efc + 1, i_b] - old_force[1])
    # Reject a step that raised the cost.
    func_cost_change(i_b, j_efc, Ac, old_force, res, EPS, constraint_state.efc_force, 2)


# ======================================================================================================================
# Colored Gauss-Seidel (warp-per-env) matrix-free noslip.
#
# The scalar kernel_noslip runs one env per thread, starving the GPU at small batch. Here a block of NOSLIP_COOP_T lanes
# handles an env, with the rows graph-colored so same-color rows touch disjoint mass blocks and update in parallel.
# Sweeping the colors in order keeps it true Gauss-Seidel (each color sees the previous colors' updates).
# ======================================================================================================================


@qd.func
def func_color_clear(tid, T, i_b, n_dofs, n_rows, constraint_state: array_class.ConstraintState):
    """Reset the greedy-coloring workspace: color_block_used[*, *, i_b] = 0 and row_color[*, i_b] = -1. Lane-strided."""
    MAXC = qd.static(NOSLIP_COLOR_MAXC)
    total = MAXC * n_dofs
    i = tid
    while i < total:
        c = i // n_dofs
        i_d = i % n_dofs
        constraint_state.color_block_used[c, i_d, i_b] = gs.qd_int(0)
        i += T
    i_c = tid
    while i_c < n_rows:
        constraint_state.row_color[i_c, i_b] = gs.qd_int(-1)
        i_c += T


@qd.func
def func_color_assign(
    i_b, ne, nf, const_start, const_end, n_rows,
    constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo,
):
    """Greedy graph-coloring of this env's noslip rows on a single lane (caller guards tid == 0).

    Atoms are the friction rows and the pyramid bases j_efc (the base's dof support also covers its j_efc+1 partner).
    Two atoms conflict iff they share a mass block (keyed on the first dof). Each atom takes the lowest conflict-free
    color and claims its blocks; equality / joint-limit rows keep -1 so the sweep skips them (matching the scalar
    kernel). Overflow past NOSLIP_COLOR_MAXC colors sets n_colors = -1, which the kernel handles by falling back to a
    serial single-lane sweep for that env."""
    MAXC = qd.static(NOSLIP_COLOR_MAXC)
    max_color = gs.qd_int(-1)
    overflow = False
    for i_c in range(n_rows):
        is_friction = i_c >= ne and i_c < ne + nf
        is_pair_base = i_c >= const_start and i_c < const_end and (i_c - const_start) % 2 == 0
        if is_friction or is_pair_base:
            chosen = gs.qd_int(-1)
            for c in range(MAXC):
                if chosen < 0:
                    ok = True
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        blk = rigid_info.dofs_mass_block_start[i_d]
                        if constraint_state.color_block_used[c, blk, i_b] == 1:
                            ok = False
                    if ok:
                        chosen = c
            if chosen < 0:
                overflow = True
                chosen = MAXC - 1  # clamp so the marking below stays in bounds; the sentinel flags the failure
            for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                blk = rigid_info.dofs_mass_block_start[i_d]
                constraint_state.color_block_used[chosen, blk, i_b] = gs.qd_int(1)
            constraint_state.row_color[i_c, i_b] = chosen
            if is_pair_base:
                constraint_state.row_color[i_c + 1, i_b] = chosen
            if chosen > max_color:
                max_color = chosen
    if overflow:
        constraint_state.n_colors[i_b] = gs.qd_int(-1)
    else:
        constraint_state.n_colors[i_b] = max_color + 1


@qd.func
def func_color_apply_row(
    i_c, i_b, i_lane, ne, nf, const_start, const_end, EPS, omega,
    constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo,
):
    """Project one noslip row at lane i_lane and scatter its force delta into qacc. Friction rows get the 1x1 update,
    pyramid bases the 2x2; equality / joint-limit rows are skipped. The update leaves noslip_minv holding M^-1 J^T, so
    propagation is a scaled accumulate (no extra solve)."""
    if i_c >= ne and i_c < ne + nf:
        old = constraint_state.efc_force[i_c, i_b]
        func_noslip_update_frictionloss_lane(i_c, i_b, i_lane, omega, constraint_state, rigid_info)
        dfc = constraint_state.efc_force[i_c, i_b] - old
        func_accumulate_row_blocks_lane(
            i_c, i_b, i_lane, dfc, constraint_state.jac_dofs_idx, constraint_state.noslip_minv,
            constraint_state.qacc, constraint_state.jac_n_dofs, rigid_info,
        )
    elif i_c >= const_start and i_c < const_end and (i_c - const_start) % 2 == 0:
        old0 = constraint_state.efc_force[i_c, i_b]
        old1 = constraint_state.efc_force[i_c + 1, i_b]
        func_noslip_update_collision_pair_lane(i_c, i_b, i_lane, EPS, omega, constraint_state, rigid_info)
        df0 = constraint_state.efc_force[i_c, i_b] - old0
        df1 = constraint_state.efc_force[i_c + 1, i_b] - old1
        func_apply_Minv_rows_lane(
            i_c, i_c + 1, i_b, i_lane, constraint_state.jac_dofs_idx, df0, df1,
            constraint_state.noslip_minv, constraint_state.jac, constraint_state.jac_n_dofs, rigid_info,
        )
        func_accumulate_row_blocks_lane(
            i_c, i_b, i_lane, 1.0, constraint_state.jac_dofs_idx, constraint_state.noslip_minv,
            constraint_state.qacc, constraint_state.jac_n_dofs, rigid_info,
        )


@qd.kernel(fastcache=True)
def kernel_noslip_color(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Colored Gauss-Seidel matrix-free noslip; one block of NOSLIP_COOP_T lanes per env. Whole-env sweep only
    (per-island solve off).

    Lane 0 colors the rows, then each iteration refreshes qacc = acc_smooth + M^-1 J^T f once and sweeps the colors in
    order, updating each color's rows in parallel and scattering their force deltas back into qacc before the next
    color. Same-color rows touch disjoint mass blocks, so the update and the qacc scatter are race-free.
    """
    _B = constraint_state.jac.shape[2]
    _T = qd.static(32)  # == NOSLIP_COOP_T

    qd.loop_config(name="noslip_color", block_dim=_T)
    for i_flat in range(_B * _T):
        tid = i_flat % _T
        i_b = i_flat // _T

        n_dofs = constraint_state.qfrc_constraint.shape[0]
        n_rows = constraint_state.n_constraints[i_b]
        ne = constraint_state.n_constraints_equality[i_b]
        nf = constraint_state.n_constraints_frictionloss[i_b]
        const_start = ne + nf
        const_end = const_start + qd.static(rigid_config.rows_per_contact) * collider_state.n_contacts[i_b]
        EPS = rigid_info.EPS[None]

        n_coop_iters = rigid_info.noslip_iterations[None]
        if qd.static(NOSLIP_COOP_ITERS > 0):
            n_coop_iters = qd.static(NOSLIP_COOP_ITERS)

        if n_rows > 0:
            func_color_clear(tid, _T, i_b, n_dofs, n_rows, constraint_state)
            qd.simt.block.sync()
            if tid == 0:
                func_color_assign(i_b, ne, nf, const_start, const_end, n_rows, constraint_state, rigid_info)
            qd.simt.block.sync()
            n_colors = constraint_state.n_colors[i_b]

            for i_iter in range(n_coop_iters):
                # Refresh qacc from the current forces once per iteration (bounds fp drift); colors then propagate
                # their deltas into qacc incrementally.
                func_coop_zero_qfrc(tid, _T, i_b, n_dofs, constraint_state)
                qd.simt.block.sync()
                func_coop_accum_qfrc(tid, _T, i_b, n_rows, constraint_state)
                qd.simt.block.sync()
                func_coop_solve_qacc(tid, _T, i_b, n_dofs, constraint_state, rigid_info)
                qd.simt.block.sync()
                func_coop_add_smooth(tid, _T, i_b, n_dofs, constraint_state, dyn_state)
                qd.simt.block.sync()

                if n_colors >= 0:
                    for c in range(n_colors):
                        # Same-color rows own disjoint mass blocks, so updating them and scattering their deltas into
                        # qacc are race-free across lanes without a mid-color sync.
                        i_c = tid
                        while i_c < n_rows:
                            if constraint_state.row_color[i_c, i_b] == c:
                                func_color_apply_row(
                                    i_c, i_b, tid, ne, nf, const_start, const_end, EPS,
                                    qd.static(NOSLIP_COLOR_OMEGA), constraint_state, rigid_info,
                                )
                            i_c += _T
                        qd.simt.block.sync()
                else:
                    # Coloring overflowed NOSLIP_COLOR_MAXC: fall back to a serial single-lane Gauss-Seidel sweep over
                    # all rows in index order. Slower for this env, but correct and race-free (one lane), and it still
                    # rides the cooperative qacc refresh above.
                    if tid == 0:
                        for i_c in range(n_rows):
                            func_color_apply_row(
                                i_c, i_b, 0, ne, nf, const_start, const_end, EPS,
                                qd.static(NOSLIP_COLOR_OMEGA), constraint_state, rigid_info,
                            )
                    qd.simt.block.sync()

            # Dual finish: rebuild qacc / qfrc from the final forces and write them to dof state.
            func_coop_zero_qfrc(tid, _T, i_b, n_dofs, constraint_state)
            qd.simt.block.sync()
            func_coop_accum_qfrc(tid, _T, i_b, n_rows, constraint_state)
            qd.simt.block.sync()
            func_coop_solve_qacc(tid, _T, i_b, n_dofs, constraint_state, rigid_info)
            qd.simt.block.sync()
            func_coop_finish_copy(tid, _T, i_b, n_dofs, constraint_state, dyn_state)
