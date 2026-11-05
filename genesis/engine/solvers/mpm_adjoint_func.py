"""Layer-1 stages of the adjoint of the implicit material point method (MPM) substep.

The substep maps the particle state p of frame f to that of frame f + 1 through the grid displacement u* that minimizes
the incremental potential E(u, p), so that grad_u E(u*, p) = 0. Its backward pass consumes the adjoint of frame f + 1 in
the `.grad` of the particle arrays, clears it, and adds the adjoint of frame f in four steps:

1. the reverse of the grid-to-particle transfer (G2P) and of the advection, which gives the direct terms of frame f and
   the adjoint u_bar of u*,
2. the adjoint solve H lambda = u_bar by the preconditioned conjugate gradient (PCG) of the forward solve, on the
   Hessian at u* with its symmetric positive definite (SPD) projection,
3. the implicit-function pullback p_bar = -(d grad_u E / dp)^T lambda over the kinetic and elastic terms, with the exact
   differential of the first Piola-Kirchhoff stress,
4. the reverse of the particle-to-grid transfer (P2G) from the adjoint of the node momentum and mass.

The gradients are approximate where the projection clamps a mode of the elastic Hessian. The wall rows enter the adjoint
solve through the Hessian and the pullback through their weights and coefficients (see func_wall_contact_adjoint), as
the rows of the couplings do (see mpm_rigid_adjoint_func), and the constraint springs enter it through the particle
position and the pose of their links (see func_constraint_adjoint). The return maps of G2P reverse by hand (see
return_map_vjp), and the snow hardening reverses through the Lame parameters read by the energy.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.array_class import LinksState

from .mpm_data import (
    BLOCK_CELLS,
    CONTACT_ROWS,
    MpmGridScratch,
    MpmInfo,
    MpmLinearScratch,
    MpmMaterialTable,
    MpmNewtonScratch,
    MpmParticles,
    MpmParticlesInfo,
    MpmParticlesNG,
)
from .mpm_func import (
    LIQUID_J_MIN,
    PLASTIC_LIQUID,
    PLASTIC_NONE,
    PLASTIC_SNOW,
    SNOW_HARDENING,
    active_fibre_stiffness,
    isotropic_first_piola_differential,
    local_offset,
    material_param,
    node_offset,
    particle_elastic_stress,
    particle_of_sorted,
    q1_stencil,
    q1_weight_hessian,
    return_map_principal,
    stencil_node_slots,
    trial_deformation,
    unpack_block,
)


@qd.func
def return_map_vjp(
    slot: int,
    kind: int,
    F: qd.types.matrix(3, 3),
    Jp: float,
    F_new_bar: qd.types.matrix(3, 3),
    Jp_new_bar: float,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Reverse of `return_map`, from the adjoints of the returned deformation gradient and plastic state.

    Returns the adjoints of the trial deformation gradient F and of the plastic state Jp, given F_new_bar and
    Jp_new_bar, the adjoints of the returned F_new and Jp_new.

    Every return map but the liquid's is F_new = U diag(h(sigma, Jp)) V^T, with F = U diag(sigma) V^T the singular value
    decomposition (SVD), an isotropic matrix function: its reverse is its differential (see
    isotropic_first_piola_differential) with the transposed Jacobian dh/dsigma, the twist and flip blocks of the axis
    pairs being symmetric. The dependence of Jp_new on sigma adds U diag(dJp_new/dsigma) V^T Jp_new_bar to the adjoint
    of F, and h depends on Jp for sand. The liquid's F_new = J^(1/3) I, J being the volume ratio clamped to
    [LIQUID_J_MIN, 1], reverses through det F where the clamp is inactive.
    """
    F_bar = F_new_bar
    Jp_bar = Jp_new_bar
    if kind == PLASTIC_LIQUID:
        det = F.determinant()
        F_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
        if det > qd.static(LIQUID_J_MIN) and det < 1.0:
            # d det F / dF is its cofactor, det F F^-T
            J_bar = (F_new_bar[0, 0] + F_new_bar[1, 1] + F_new_bar[2, 2]) / (3.0 * qd.pow(det, 2.0 / 3.0))
            F_bar = (J_bar * det) * F.inverse().transpose()
    elif kind != PLASTIC_NONE:
        U, S, V = gu.qd_svd(F)
        sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float)
        h, Jp_new, is_changed, dh, dh_dJp, dJp_new_dsigma, dJp_new_dJp = return_map_principal(
            slot, kind, sigma, Jp, materials, mpm_info
        )
        if is_changed:
            F_bar = isotropic_first_piola_differential(U, V, sigma, h, dh.transpose(), F_new_bar)
            F_bar += Jp_new_bar * (
                U
                @ qd.Matrix(
                    [[dJp_new_dsigma[0], 0.0, 0.0], [0.0, dJp_new_dsigma[1], 0.0], [0.0, 0.0, dJp_new_dsigma[2]]],
                    dt=gs.qd_float,
                )
                @ V.transpose()
            )
            G = U.transpose() @ F_new_bar @ V
            Jp_bar = Jp_new_bar * dJp_new_dJp + dh_dJp[0] * G[0, 0] + dh_dJp[1] * G[1, 1] + dh_dJp[2] * G[2, 2]
    return F_bar, Jp_bar


@qd.func
def band_row_bw(
    sol_params: qd.types.vector(7), phi: float, vel: float, diag_coef: float, dt: float, D_bar: float, t_bar: float
):
    """Reverse of `band_row` (see soft_row), from the adjoints of the coefficients of a row to those of its inputs.

    Returns the adjoints of phi, vel and diag_coef, given D_bar and t_bar, the adjoints of the coefficients D and t.
    """
    phi_bar = gs.qd_float(0.0)
    vel_bar = gs.qd_float(0.0)
    width = sol_params[4]
    # The impedance of the row, at the edge of the range phi = 0 short of it, chosen at runtime so that
    # `imp_aref_grad` is inlined once
    phi_row = gs.qd_float(0.0)
    if phi <= 0.0:
        phi_row = phi
    imp, b, k, d_imp_d_x = gu.imp_aref_grad(sol_params, phi_row)
    diag_raw = diag_coef * (1.0 - imp) / imp
    # Zero where diag_raw is clamped at the floor gs.EPS
    diag_raw_bar = gs.qd_float(0.0)
    if diag_raw > gs.EPS:
        D = 1.0 / diag_raw
        diag_raw_bar = -D * D * D_bar
    diag_coef_bar = diag_raw_bar * (1.0 - imp) / imp
    if phi <= 0.0:
        # t = dt vel + dt^2 (-b vel - k imp phi), imp of |phi| / width
        imp_bar = diag_raw_bar * (-diag_coef / (imp * imp)) - dt * dt * k * phi * t_bar
        vel_bar = (dt - dt * dt * b) * t_bar
        phi_bar = -dt * dt * k * imp * t_bar - imp_bar * d_imp_d_x / width
    else:
        phi_bar = -t_bar
    return phi_bar, vel_bar, diag_coef_bar


@qd.func
def polar_rotation_bw(
    U: qd.types.matrix(3, 3), S: qd.types.matrix(3, 3), V: qd.types.matrix(3, 3), R_bar: qd.types.matrix(3, 3)
):
    """Reverse of the rotation of the polar decomposition, from its adjoint to that of the deformation gradient.

    U, S and V hold the singular value decomposition (SVD) F = U S V^T, and the rotation of the polar decomposition F =
    R (V S V^T) is R = U V^T. With M = U^T dF V, dR = U X V^T where X_ij = (M_ij - M_ji) / (s_i + s_j), so the returned
    adjoint of F is F_bar = U Y V^T where Y_ij = (G_ij - G_ji) / (s_i + s_j) and G = U^T R_bar V."""
    G = U.transpose() @ R_bar @ V
    Y = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if qd.static(i != j):
            Y[i, j] = (G[i, j] - G[j, i]) / qd.max(S[i, i] + S[j, j], gs.EPS)
    return U @ Y @ V.transpose()


@qd.func(requires_top_level=True)
def func_wall_contact_adjoint(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: MpmLinearScratch,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Pull the wall rows of every valid particle (see func_build_wall_rows) back into the adjoint of the particles.

    The adjoints add to frame f. A row k on the trial displacement Delta = sum_a w_a u_a, of normal J = +-e_d, adds D c
    J . delta to lambda^T grad_u E when active, with c = J . Delta - t and delta = sum_a w_a lambda_a. Its pullback is
    that of the Rigid rows (see func_rigid_contact_adjoint) with a fixed wall in place of the link: through the weights,
    and through the coefficients via the gap, the velocity along the normal and the inverse weight of the particle."""
    qd.loop_config(name="mpm_wall_contact_adjoint")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        has_row = False
        for d in qd.static(range(3)):
            has_row = has_row or newton.contact_D[CONTACT_ROWS * s + d] > 0.0
        if has_row:
            slots = stencil_node_slots(env, cell, grid)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            w, grad_w = q1_stencil(x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
            Delta = qd.Vector.zero(gs.qd_float, 3)
            delta = qd.Vector.zero(gs.qd_float, 3)
            for a in qd.static(range(8)):
                Delta += w[a] * qd.Vector([newton.u[3 * slots[a] + e] for e in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * slots[a]
                delta += w[a] * qd.Vector([linear.solution[row + e] for e in qd.static(range(3))], dt=gs.qd_float)
            U, S, V = gu.qd_svd(particles.F[f, i_p, i_b])
            R = U @ V.transpose()
            W = particles_info.contact_invweight[i_p]
            sol_params = mpm_info.wall_sol_params[()]
            x_bar = qd.Vector.zero(gs.qd_float, 3)
            v_bar = qd.Vector.zero(gs.qd_float, 3)
            R_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
            Delta_bar = qd.Vector.zero(gs.qd_float, 3)
            delta_bar = qd.Vector.zero(gs.qd_float, 3)
            for d in qd.static(range(3)):
                k = CONTACT_ROWS * s + d
                D = newton.contact_D[k]
                J = newton.contact_dir[k]
                c = J.dot(Delta) - newton.contact_t[k]
                if D > 0.0 and c < 0.0:
                    a_k = J.dot(delta)
                    Delta_bar -= D * a_k * J
                    delta_bar -= D * c * J
                    # The row as func_build_wall_rows built it
                    phi = x_p[d] - mpm_info.wall_lower[()][d]
                    sign = gs.qd_float(1.0)
                    if mpm_info.wall_upper[()][d] - x_p[d] < phi:
                        phi = mpm_info.wall_upper[()][d] - x_p[d]
                        sign = -1.0
                    n = qd.Vector.zero(gs.qd_float, 3)
                    n[d] = sign
                    n_rest = R.transpose() @ n
                    invweight = n_rest.dot(W @ n_rest)
                    phi_bar, vel_bar, invweight_bar = band_row_bw(
                        sol_params, phi, sign * v_p[d], invweight, dt, -c * a_k, D * a_k
                    )
                    x_bar[d] += sign * phi_bar
                    v_bar[d] += sign * vel_bar
                    R_bar += (2.0 * invweight_bar) * n.outer_product(W @ n_rest)
            for a in qd.static(range(8)):
                u_a = qd.Vector([newton.u[3 * slots[a] + e] for e in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * slots[a]
                lam_a = qd.Vector([linear.solution[row + e] for e in qd.static(range(3))], dt=gs.qd_float)
                grad_w_a = qd.Vector([grad_w[a, e] for e in qd.static(range(3))], dt=gs.qd_float)
                x_bar += (Delta_bar.dot(u_a) + delta_bar.dot(lam_a)) * grad_w_a
            particles.pos.grad[f, i_p, i_b] += x_bar
            particles.vel.grad[f, i_p, i_b] += v_bar
            particles.F.grad[f, i_p, i_b] += polar_rotation_bw(U, S, V, R_bar)


@qd.func(requires_top_level=True)
def func_constraint_adjoint(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: MpmLinearScratch,
    links_state: LinksState,
    mpm_info: MpmInfo,
    constraints: qd.template(),
):
    """Pull the constraint springs (see func_build_constraint_rows) back into the adjoints of the particles and links.

    The adjoints add to frame f of the particles and to the poses of the links that hold them. A spring adds delta . g
    to lambda^T grad_u E, with g = kappa (x + Delta - T) + gamma Delta, kappa = dt^2 k, gamma = dt c, Delta = sum_a w_a
    u_a and delta = sum_a w_a lambda_a. The position enters through the weights, the gap and Delta, x_bar = -[kappa
    delta + sum_a (lambda_a . g + (kappa + gamma) u_a . delta) grad w_a], and the target through the gap, T_bar = kappa
    delta. The stiffness and the mass that scales the damping are model constants."""
    qd.loop_config(name="mpm_constraint_adjoint")
    for s in range(grid.n_valid[()]):
        spring = newton.constraint_spring[s]
        if spring > 0.0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
            slots = stencil_node_slots(env, cell, grid)
            x_p = particles.pos[f, i_p, i_b]
            w, grad_w = q1_stencil(x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
            Delta = qd.Vector.zero(gs.qd_float, 3)
            delta = qd.Vector.zero(gs.qd_float, 3)
            for a in qd.static(range(8)):
                Delta += w[a] * qd.Vector([newton.u[3 * slots[a] + e] for e in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * slots[a]
                delta += w[a] * qd.Vector([linear.solution[row + e] for e in qd.static(range(3))], dt=gs.qd_float)
            damping = newton.constraint_damping[s]
            g = spring * (x_p + Delta - newton.constraint_target[s]) + damping * Delta
            x_bar = -spring * delta
            for a in qd.static(range(8)):
                u_a = qd.Vector([newton.u[3 * slots[a] + e] for e in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * slots[a]
                lam_a = qd.Vector([linear.solution[row + e] for e in qd.static(range(3))], dt=gs.qd_float)
                grad_w_a = qd.Vector([grad_w[a, e] for e in qd.static(range(3))], dt=gs.qd_float)
                x_bar -= (lam_a.dot(g) + (spring + damping) * u_a.dot(delta)) * grad_w_a
            particles.pos.grad[f, i_p, i_b] += x_bar
            i_l = constraints[i_p, i_b].link_idx
            if i_l >= 0:
                target_bar = spring * delta
                links_state.pos.grad[i_l, i_b] += target_bar
                links_state.quat.grad[i_l, i_b] += gu.qd_transform_by_quat_grad_quat(
                    constraints[i_p, i_b].link_local_pos, links_state.quat[i_l, i_b], target_bar
                )


@qd.func(requires_top_level=True)
def func_adjoint_reset_u_bar(grid: MpmGridScratch, newton: MpmNewtonScratch):
    """Clear the adjoint of u*, which the reverse of the grid-to-particle transfer (G2P) accumulates into `newton.du`
    (free once the forward solve is over)."""
    qd.loop_config(name="mpm_adjoint_reset_u_bar")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        newton.du[i] = 0.0


@qd.func(requires_top_level=True)
def func_adjoint_g2p(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Reverse of the grid-to-particle transfer (G2P) and of the advection, from the adjoint of the next frame.

    The forward maps of frame f to frame f + 1 are x' = x + dt v, v = sum_a w_a u_a / dt, C = sum_a u_a grad w_a^T / dt
    and (F', Jp') = return_map((I + dt C) F, Jp). The direct terms add to the adjoint of frame f, and the adjoint of u*
    to `newton.du` by atomics over the stencil. The position enters through the weights and their gradients.
    """
    qd.loop_config(name="mpm_adjoint_g2p")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        slots = stencil_node_slots(env, cell, grid)
        fx = particles.pos[f, i_p, i_b] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
        C = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            v_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float) / dt
            C += v_a.outer_product(qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float))
        F = particles.F[f, i_p, i_b]
        slot = particles_info.material_idx[i_p]
        F_trial = (qd.Matrix.identity(gs.qd_float, 3) + dt * C) @ F
        F_trial_bar, Jp_bar = return_map_vjp(
            slot,
            materials.mat_plastic[slot],
            F_trial,
            particles.Jp[f, i_p, i_b],
            particles.F.grad[f + 1, i_p, i_b],
            particles.Jp.grad[f + 1, i_p, i_b],
            materials,
            mpm_info,
        )

        x_bar_next = particles.pos.grad[f + 1, i_p, i_b]
        v_bar = particles.vel.grad[f + 1, i_p, i_b] + dt * x_bar_next
        C_bar = particles.C.grad[f + 1, i_p, i_b] + dt * F_trial_bar @ F.transpose()

        x_bar = x_bar_next
        for a in qd.static(range(8)):
            v_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float) / dt
            grad_w_a = qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
            u_bar = (w[a] * v_bar + C_bar @ grad_w_a) / dt
            for d in qd.static(range(3)):
                qd.atomic_add(newton.du[3 * slots[a] + d], u_bar[d])
            x_bar += v_a.dot(v_bar) * grad_w_a + q1_weight_hessian(fx, mpm_info.inv_dx[()], a) @ (
                C_bar.transpose() @ v_a
            )

        particles.pos.grad[f, i_p, i_b] += x_bar
        particles.F.grad[f, i_p, i_b] += (qd.Matrix.identity(gs.qd_float, 3) + dt * C).transpose() @ F_trial_bar
        particles.Jp.grad[f, i_p, i_b] += Jp_bar
        particles.actu.grad[f, i_p, i_b] += particles.actu.grad[f + 1, i_p, i_b]


@qd.func(requires_top_level=True)
def func_adjoint_rhs(grid: MpmGridScratch, newton: MpmNewtonScratch, linear: MpmLinearScratch):
    """Set the right-hand side of the adjoint solve to the adjoint of u*, zero on the fixed nodes, whose rows of
    the Newton matrix are the identity."""
    qd.loop_config(name="mpm_adjoint_rhs")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        value = newton.du[i]
        if grid.node_fixed[i // 3] != 0:
            value = 0.0
        linear.b_rhs[i] = value


@qd.func(requires_top_level=True)
def func_adjoint_nodes(
    t: float,
    dt: float,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: MpmLinearScratch,
    force_fields: qd.template(),
    mpm_info: MpmInfo,
):
    """Pull lambda back through the kinetic term and reverse the free velocity of every node.

    The node adjoints of momentum and mass go to `newton.adj_mom` and `newton.adj_mass`. lambda is `linear.solution`,
    and the free velocity is v_free = v + dt acc(x_a, v) with v = mom / m. grad_{u_a} E = m_a (u_a - dt v_free_a) gives
    v_free_bar = m_a dt lambda_a and m_bar = -lambda_a . (u_a - dt v_free_a). The force fields add dt (d acc / dv)^T
    v_free_bar to v_bar = v_free_bar (see 'ForceField.get_acc_vel_vjp'), the node position x_a being fixed, so mom_bar =
    v_bar / m_a and m_bar -= v_a . v_bar / m_a. A fixed or massless node, whose velocity is zero, gets zero adjoints.
    """
    qd.loop_config(name="mpm_adjoint_nodes")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        mass = grid.node_mass[node]
        mass_bar = gs.qd_float(0.0)
        mom_bar = qd.Vector.zero(gs.qd_float, 3)
        if mass > 0.0 and grid.node_fixed[node] == 0:
            lam = qd.Vector([linear.solution[3 * node + d] for d in qd.static(range(3))])
            u = qd.Vector([newton.u[3 * node + d] for d in qd.static(range(3))], dt=gs.qd_float)
            v = grid.node_v[node]
            v_free = grid.node_v_free[node]
            v_free_bar = (mass * dt) * lam
            v_bar = v_free_bar
            if qd.static(len(force_fields.force_fields) > 0):
                env, block = unpack_block(grid.block_coords[node // BLOCK_CELLS])
                x_n = (block * 4 + local_offset(node % BLOCK_CELLS)).cast(gs.qd_float) * mpm_info.dx[()]
                for i_ff in qd.static(range(len(force_fields.force_fields))):
                    v_bar += dt * force_fields.force_fields[i_ff].get_acc_vel_vjp(x_n, v, t, -1, v_free_bar)
            mom_bar = v_bar / mass
            mass_bar = -lam.dot(u - dt * v_free) - v.dot(v_bar) / mass
        newton.adj_mom[node] = mom_bar
        newton.adj_mass[node] = mass_bar


@qd.func(requires_top_level=True)
def func_adjoint_particles(
    f: int,
    dt: float,
    gravity: qd.template(),
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: MpmLinearScratch,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Pull lambda back through the elastic term and reverse the particle-to-grid transfer (P2G) of valid particles.

    The adjoints add to frame f. With A = I + sum_a u_a grad w_a^T, Lambda = sum_a lambda_a grad w_a^T and F_trial = A
    F, the elastic gradient gives phi = V dt^2 P(F_trial) : (Lambda F), and p_bar = -d phi / dp. Hence F_bar -= V dt^2
    (A^T Q + Lambda^T P) with Q = dP[Lambda F], the actuation enters through the fibre stiffness, and the position
    through the weight Hessians H_a, x_bar -= V dt^2 sum_a H_a F (Q^T u_a + P^T lambda_a). P2G spreads m_p (v_p + dt g +
    C_p (x_a - x_p)) and m_p by w_a, which reverses into v, C and x from the node adjoints of `func_adjoint_nodes`.
    """
    qd.loop_config(name="mpm_adjoint_particles")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        slots = stencil_node_slots(env, cell, grid)
        x_p = particles.pos[f, i_p, i_b]
        fx = x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
        F_trial, grad_w_, F = trial_deformation(f, i_p, i_b, cell, slots, particles, newton, mpm_info)

        Lambda = qd.Matrix.zero(gs.qd_float, 3, 3)
        grad_u = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            grad_w_a = qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
            lam_a = qd.Vector([linear.solution[3 * slots[a] + d] for d in qd.static(range(3))])
            u_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
            Lambda += lam_a.outer_product(grad_w_a)
            grad_u += u_a.outer_product(grad_w_a)
        F_increment = qd.Matrix.identity(gs.qd_float, 3) + grad_u

        slot, U, sigma, V, g, A_sigma, P = particle_elastic_stress(
            f, i_p, i_b, F_trial, particles, particles_info, materials, mpm_info
        )
        Lambda_F = Lambda @ F
        Q = isotropic_first_piola_differential(U, V, sigma, g, A_sigma, Lambda_F)
        # Snow scales mu and lam by exp(SNOW_HARDENING (1 - Jp)), to which its elastic energy and compression penalty
        # are proportional, so phi = V dt^2 P : (Lambda F) moves by -SNOW_HARDENING phi with Jp
        Jp_bar = gs.qd_float(0.0)
        if materials.mat_plastic[slot] == PLASTIC_SNOW:
            phi = gs.qd_float(0.0)
            for i, j in qd.static(qd.ndrange(3, 3)):
                phi += P[i, j] * Lambda_F[i, j]
            Jp_bar = qd.static(SNOW_HARDENING) * mpm_info.particle_volume[()] * dt * dt * phi
        muscle_direction = particles_info.muscle_direction[i_p]
        active_stiffness, dactive_stiffness = active_fibre_stiffness(
            slot, particles.actu[f, i_p, i_b], materials, mpm_info
        )
        F_trial_m = F_trial @ muscle_direction
        Lambda_F_m = Lambda_F @ muscle_direction
        P += active_stiffness * F_trial_m.outer_product(muscle_direction)
        Q += active_stiffness * Lambda_F_m.outer_product(muscle_direction)

        scale = mpm_info.particle_volume[()] * dt * dt
        F_bar = -scale * (F_increment.transpose() @ Q + Lambda.transpose() @ P)
        actu_bar = -scale * dactive_stiffness * F_trial_m.dot(Lambda_F_m)
        x_bar = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(8)):
            lam_a = qd.Vector([linear.solution[3 * slots[a] + d] for d in qd.static(range(3))])
            u_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
            H_a = q1_weight_hessian(fx, mpm_info.inv_dx[()], a)
            x_bar -= scale * (H_a @ (F @ (Q.transpose() @ u_a + P.transpose() @ lam_a)))

        # Reverse of the particle-to-grid transfer (P2G)
        m_p = particles_info.mass[i_p] / mpm_info.mass_scale[()]
        v_p = particles.vel[f, i_p, i_b] + dt * gravity[i_b]
        C_p = particles.C[f, i_p, i_b]
        v_bar = qd.Vector.zero(gs.qd_float, 3)
        C_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            grad_w_a = qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
            x_a = (cell + node_offset(a)).cast(gs.qd_float) * mpm_info.dx[()]
            mom_bar = newton.adj_mom[slots[a]]
            mass_bar = newton.adj_mass[slots[a]]
            v_tilde = v_p + C_p @ (x_a - x_p)
            v_bar += (w[a] * m_p) * mom_bar
            C_bar += (w[a] * m_p) * mom_bar.outer_product(x_a - x_p)
            x_bar += (m_p * (v_tilde.dot(mom_bar) + mass_bar)) * grad_w_a - (w[a] * m_p) * (C_p.transpose() @ mom_bar)

        particles.pos.grad[f, i_p, i_b] += x_bar
        particles.vel.grad[f, i_p, i_b] += v_bar
        particles.C.grad[f, i_p, i_b] += C_bar
        particles.F.grad[f, i_p, i_b] += F_bar
        particles.Jp.grad[f, i_p, i_b] += Jp_bar
        particles.actu.grad[f, i_p, i_b] += actu_bar


@qd.func(requires_top_level=True)
def func_adjoint_consume(f: int, particles: MpmParticles, particles_ng: MpmParticlesNG, mpm_info: MpmInfo):
    """Clear the adjoint of the next frame, which the substep consumed.

    The substep carries an inactive particle over unchanged, so its adjoint of frame f + 1 first adds to that of frame
    f.
    """
    qd.loop_config(name="mpm_adjoint_consume")
    for i_p, i_b in qd.ndrange(mpm_info.n_particles[()], mpm_info.n_envs[()]):
        if not particles_ng.active[f, i_p, i_b]:
            particles.pos.grad[f, i_p, i_b] += particles.pos.grad[f + 1, i_p, i_b]
            particles.vel.grad[f, i_p, i_b] += particles.vel.grad[f + 1, i_p, i_b]
            particles.C.grad[f, i_p, i_b] += particles.C.grad[f + 1, i_p, i_b]
            particles.F.grad[f, i_p, i_b] += particles.F.grad[f + 1, i_p, i_b]
            particles.Jp.grad[f, i_p, i_b] += particles.Jp.grad[f + 1, i_p, i_b]
            particles.actu.grad[f, i_p, i_b] += particles.actu.grad[f + 1, i_p, i_b]
        particles.pos.grad[f + 1, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        particles.vel.grad[f + 1, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        particles.C.grad[f + 1, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        particles.F.grad[f + 1, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        particles.Jp.grad[f + 1, i_p, i_b] = gs.qd_float(0.0)
        particles.actu.grad[f + 1, i_p, i_b] = gs.qd_float(0.0)
