"""Layer-1 stages of the adjoint of the implicit material point method (MPM) substep.

The substep maps the particle state p of frame f to frame f + 1 through the grid displacement u* with grad_u E(u*, p) =
0. Its backward pass consumes the adjoint of frame f + 1 in the `.grad` of the particle arrays and adds that of frame f
in four steps:

1. the reverse of the grid-to-particle transfer (G2P) and of the advection, giving the adjoint u_bar of u*, 2. the
adjoint solve H lambda = u_bar by the preconditioned conjugate gradient (PCG) of the forward solve, 3. the
implicit-function pullback p_bar = -(d grad_u E / dp)^T lambda over the kinetic, elastic, wall and spring terms, 4. the
reverse of the particle-to-grid transfer (P2G).

The gradients are approximate where the projection clamps a mode of the elastic Hessian."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.array_class import LinksState

from .newton_linear import LinearBlock, NewtonLinearScratch
from .mpm_data import (
    BLOCK_CELLS,
    MpmGridScratch,
    MpmInfo,
    MpmMaterialTable,
    MpmNewtonScratch,
    MpmParticles,
    MpmParticlesInfo,
    MpmParticlesNG,
    MpmWallPairs,
)
from .mpm_func import (
    LIQUID_J_MIN,
    PLASTIC_LIQUID,
    PLASTIC_NONE,
    PLASTIC_SNOW,
    SNOW_HARDENING,
    active_fibre_stiffness,
    diagonal,
    diagonal_of,
    isotropic_first_piola_differential,
    local_offset,
    node_offset,
    particle_elastic_stress,
    particle_node_sum,
    particle_of_sorted,
    q1_stencil,
    q1_weight_hessian,
    return_map_principal,
    stencil_dots,
    stencil_gradient,
    stencil_node_slots,
    trial_deformation,
    unpack_block,
    wall_normal,
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
    """Reverse of `return_map`: the adjoints of the trial deformation gradient F and of the plastic state Jp, given
    F_new_bar and Jp_new_bar.

    Every map but the liquid's is the isotropic matrix function F_new = U diag(h(sigma, Jp)) V^T, which reverses by its
    differential with the transposed Jacobian dh/dsigma, plus the dependence of Jp_new on sigma. The liquid's F_new =
    J^(1/3) I reverses through det F where the clamp of J is inactive."""
    F_bar = F_new_bar
    Jp_bar = Jp_new_bar
    if kind == PLASTIC_LIQUID:
        det = F.determinant()
        F_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
        if det > qd.static(LIQUID_J_MIN) and det < 1.0:
            # d det F / dF is its cofactor, det F F^-T
            J_bar = F_new_bar.trace() / (3.0 * qd.pow(det, 2.0 / 3.0))
            F_bar = (J_bar * det) * F.inverse().transpose()
    elif kind != PLASTIC_NONE:
        U, S, V = gu.qd_svd(F)
        sigma = diagonal_of(S)
        h, Jp_new, is_changed, dh, dh_dJp, dJp_new_dsigma, dJp_new_dJp = return_map_principal(
            slot, kind, sigma, Jp, materials, mpm_info
        )
        if is_changed:
            F_bar = isotropic_first_piola_differential(U, V, sigma, h, dh.transpose(), F_new_bar)
            F_bar += Jp_new_bar * (U @ diagonal(dJp_new_dsigma) @ V.transpose())
            Jp_bar = Jp_new_bar * dJp_new_dJp + dh_dJp.dot(diagonal_of(U.transpose() @ F_new_bar @ V))
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
    """Reverse of the rotation R = U V^T of the polar decomposition of F = U S V^T: F_bar = U Y V^T with Y_ij = (G_ij -
    G_ji) / (s_i + s_j) and G = U^T R_bar V."""
    G = U.transpose() @ R_bar @ V
    s = diagonal_of(S)
    ones = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)
    return U @ ((G - G.transpose()) / qd.max(s.outer_product(ones) + ones.outer_product(s), gs.EPS)) @ V.transpose()


@qd.func(requires_top_level=True)
def func_wall_contact_adjoint(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    pairs: MpmWallPairs,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Pull the wall pairs (see func_build_wall_pairs) back into the adjoint of frame f of the particles, as the Rigid
    rows with a fixed wall (see func_rigid_contact_adjoint): through the weights, and through the coefficients via the
    gap, the normal velocity and the inverse weight of the particle."""
    qd.loop_config(name="mpm_wall_contact_adjoint")
    for i_pair in range(pairs.n_pairs[()]):
        s = pairs.s[i_pair]
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        slots = stencil_node_slots(env, cell, grid)
        x_p = particles.pos[f, i_p, i_b]
        v_p = particles.vel[f, i_p, i_b]
        w, grad_w = q1_stencil(x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
        Delta = particle_node_sum(slots, w, newton.u)
        delta = particle_node_sum(rows.row_start[()] + slots, w, linear.solution)
        D = pairs.D[i_pair]
        J = wall_normal(pairs.axis[i_pair], pairs.sign[i_pair])
        c = J.dot(Delta) - pairs.t[i_pair]
        if D > 0.0 and c < 0.0:
            a_k = J.dot(delta)
            Delta_bar = -D * a_k * J
            delta_bar = -D * c * J
            # The row as func_build_wall_pairs built it: the gap to the wall along J, whose gradient in x_p is J
            phi = J.dot(x_p - mpm_info.wall_lower[()])
            if pairs.sign[i_pair] < 0.0:
                phi = J.dot(x_p - mpm_info.wall_upper[()])
            U, S, V = gu.qd_svd(particles.F[f, i_p, i_b])
            R = U @ V.transpose()
            W = particles_info.contact_invweight[i_p]
            n_rest = R.transpose() @ J
            invweight = n_rest.dot(W @ n_rest)
            phi_bar, vel_bar, invweight_bar = band_row_bw(
                mpm_info.wall_sol_params[()], phi, J.dot(v_p), invweight, dt, -c * a_k, D * a_k
            )
            x_bar = phi_bar * J
            v_bar = vel_bar * J
            R_bar = (2.0 * invweight_bar) * J.outer_product(W @ n_rest)
            weights_bar = stencil_dots(slots, newton.u, Delta_bar)
            weights_bar += stencil_dots(rows.row_start[()] + slots, linear.solution, delta_bar)
            x_bar += grad_w.transpose() @ weights_bar
            particles.pos.grad[f, i_p, i_b] += x_bar
            particles.vel.grad[f, i_p, i_b] += v_bar
            particles.F.grad[f, i_p, i_b] += polar_rotation_bw(U, S, V, R_bar)


@qd.func(requires_top_level=True)
def func_constraint_adjoint(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    links_state: LinksState,
    mpm_info: MpmInfo,
    constraints: qd.template(),
):
    """Pull the constraint springs (see func_build_constraint_rows) back into the adjoints of frame f of the particles
    and of the poses of the links that hold them.

    A spring adds delta . g to lambda^T grad_u E, with g = kappa (x + Delta - T) + gamma Delta, kappa = dt^2 k, gamma =
    dt c, Delta = sum_a w_a u_a and delta = sum_a w_a lambda_a. Hence x_bar = -[kappa delta + sum_a (lambda_a . g +
    (kappa + gamma) u_a . delta) grad w_a] and T_bar = kappa delta."""
    qd.loop_config(name="mpm_constraint_adjoint")
    for s in range(grid.n_valid[()]):
        spring = newton.constraint_spring[s]
        if spring > 0.0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
            slots = stencil_node_slots(env, cell, grid)
            x_p = particles.pos[f, i_p, i_b]
            w, grad_w = q1_stencil(x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
            Delta = particle_node_sum(slots, w, newton.u)
            delta = particle_node_sum(rows.row_start[()] + slots, w, linear.solution)
            damping = newton.constraint_damping[s]
            g = spring * (x_p + Delta - newton.constraint_target[s]) + damping * Delta
            weights_bar = stencil_dots(rows.row_start[()] + slots, linear.solution, g)
            weights_bar += stencil_dots(slots, newton.u, (spring + damping) * delta)
            x_bar = -spring * delta - grad_w.transpose() @ weights_bar
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
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        newton.du[node] = qd.Vector.zero(gs.qd_float, 3)


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

    The forward maps are x' = x + dt v, v = sum_a w_a u_a / dt, C = sum_a u_a grad w_a^T / dt and (F', Jp') =
    return_map((I + dt C) F, Jp). The direct terms add to frame f, and the adjoint of u* to `newton.du`."""
    qd.loop_config(name="mpm_adjoint_g2p")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        slots = stencil_node_slots(env, cell, grid)
        fx = particles.pos[f, i_p, i_b] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
        C = stencil_gradient(slots, grad_w, newton.u) / dt
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
            v_a = newton.u[slots[a]] / dt
            grad_w_a = grad_w[a, :]
            u_bar = (w[a] * v_bar + C_bar @ grad_w_a) / dt
            qd.atomic_add(newton.du[slots[a]], u_bar)
            x_bar += v_a.dot(v_bar) * grad_w_a + q1_weight_hessian(fx, mpm_info.inv_dx[()], a) @ (
                C_bar.transpose() @ v_a
            )

        particles.pos.grad[f, i_p, i_b] += x_bar
        particles.F.grad[f, i_p, i_b] += (qd.Matrix.identity(gs.qd_float, 3) + dt * C).transpose() @ F_trial_bar
        particles.Jp.grad[f, i_p, i_b] += Jp_bar
        particles.actu.grad[f, i_p, i_b] += particles.actu.grad[f + 1, i_p, i_b]


@qd.func(requires_top_level=True)
def func_adjoint_rhs(grid: MpmGridScratch, newton: MpmNewtonScratch, linear: NewtonLinearScratch, rows: LinearBlock):
    """Set the right-hand side of the adjoint solve to the adjoint of u*, zero on the fixed nodes, whose rows of
    the Newton matrix are the identity."""
    qd.loop_config(name="mpm_adjoint_rhs")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        value = newton.du[node]
        if grid.node_fixed[node] != 0:
            value = qd.Vector.zero(gs.qd_float, 3)
        linear.b_rhs[rows.row_start[()] + node] = value


@qd.func(requires_top_level=True)
def func_adjoint_nodes(
    t: float,
    dt: float,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    force_fields: qd.template(),
    mpm_info: MpmInfo,
):
    """Pull lambda back through the kinetic term and reverse the free velocity of every node into `newton.adj_mom` and
    `newton.adj_mass`.

    From grad_{u_a} E = m_a (u_a - dt v_free_a): v_free_bar = m_a dt lambda_a and m_bar = -lambda_a . (u_a - dt
    v_free_a), the force fields adding dt (d acc / dv)^T v_free_bar to v_bar (see 'ForceField.get_acc_vel_vjp'). A fixed
    or massless node gets zero adjoints."""
    qd.loop_config(name="mpm_adjoint_nodes")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        mass = grid.node_mass[node]
        mass_bar = gs.qd_float(0.0)
        mom_bar = qd.Vector.zero(gs.qd_float, 3)
        if mass > 0.0 and grid.node_fixed[node] == 0:
            lam = linear.solution[rows.row_start[()] + node]
            u = newton.u[node]
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
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Pull lambda back through the elastic term and reverse the particle-to-grid transfer (P2G) of valid particles.

    With A = I + sum_a u_a grad w_a^T, Lambda = sum_a lambda_a grad w_a^T and F_trial = A F, the elastic gradient gives
    phi = V dt^2 P(F_trial) : (Lambda F) and p_bar = -d phi / dp: F_bar -= V dt^2 (A^T Q + Lambda^T P) with Q =
    dP[Lambda F], and x_bar -= V dt^2 sum_a H_a F (Q^T u_a + P^T lambda_a) through the weight Hessians H_a. P2G then
    reverses into v, C and x from the node adjoints."""
    qd.loop_config(name="mpm_adjoint_particles")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        slots = stencil_node_slots(env, cell, grid)
        x_p = particles.pos[f, i_p, i_b]
        fx = x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
        F_trial, grad_w_, F = trial_deformation(f, i_p, i_b, cell, slots, particles, newton, mpm_info)
        Lambda = stencil_gradient(rows.row_start[()] + slots, grad_w, linear.solution)
        F_increment = qd.Matrix.identity(gs.qd_float, 3) + stencil_gradient(slots, grad_w, newton.u)

        slot, U, sigma, V, g, A_sigma, P = particle_elastic_stress(
            f, i_p, i_b, F_trial, particles, particles_info, materials, mpm_info
        )
        Lambda_F = Lambda @ F
        Q = isotropic_first_piola_differential(U, V, sigma, g, A_sigma, Lambda_F)
        # Snow scales mu and lam by exp(SNOW_HARDENING (1 - Jp)), to which its elastic energy and compression penalty
        # are proportional, so phi = V dt^2 P : (Lambda F) moves by -SNOW_HARDENING phi with Jp
        Jp_bar = gs.qd_float(0.0)
        if materials.mat_plastic[slot] == PLASTIC_SNOW:
            Jp_bar = qd.static(SNOW_HARDENING) * mpm_info.particle_volume[()] * dt * dt * (P * Lambda_F).sum()
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
            lam_a = linear.solution[rows.row_start[()] + slots[a]]
            u_a = newton.u[slots[a]]
            H_a = q1_weight_hessian(fx, mpm_info.inv_dx[()], a)
            x_bar -= scale * (H_a @ (F @ (Q.transpose() @ u_a + P.transpose() @ lam_a)))

        # Reverse of the particle-to-grid transfer (P2G)
        m_p = particles_info.mass[i_p] / mpm_info.mass_scale[()]
        v_p = particles.vel[f, i_p, i_b] + dt * gravity[i_b]
        C_p = particles.C[f, i_p, i_b]
        v_bar = qd.Vector.zero(gs.qd_float, 3)
        C_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            grad_w_a = grad_w[a, :]
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
    """Clear the adjoint of the next frame, which the substep consumed, after adding that of an inactive particle, which
    the substep carries over unchanged, to frame f."""
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
