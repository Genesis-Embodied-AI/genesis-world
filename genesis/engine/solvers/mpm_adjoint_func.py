"""Layer-1 stages of the adjoint of the implicit MPM substep (design §8.2).

The substep maps the particle state of frame f to that of frame f + 1 through the grid displacement u* that minimizes
the incremental potential E(u; p), `grad_u E(u*; p) = 0`. Its backward pass consumes the adjoint of frame f + 1 in the
`.grad` of the particle arrays, clears it, and adds the adjoint of frame f:

1. reverse of G2P and advection, which gives the direct terms of frame f and the adjoint u_bar of u*,
2. the adjoint solve H lambda = u_bar by the PCG of the forward solve, on its SPD-projected Hessian at u*,
3. the implicit-function pullback p_bar = -(d grad_u E / dp)^T lambda over the kinetic and elastic terms, with the
   exact differential of the first Piola-Kirchhoff stress,
4. reverse of P2G from the adjoint of the node momentum and mass.

The gradients are approximate where the projection clamps a mode of the elastic Hessian. The wall rows enter the
adjoint solve through the Hessian and the pullback through their weights and coefficients (see
func_wall_contact_adjoint), as the rows of the couplings do (see mpm_rigid_adjoint_func), and the constraint springs
through the particle position and the pose of their links (see func_constraint_adjoint). The return maps are the
identity: only the materials without plasticity reach here.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from .mpm_data import BLOCK_CELLS, CONTACT_ROWS, MpmMaterialTable, MpmParticles, MpmParticlesInfo, MpmParticlesNG
from .mpm_func import (
    PARAM_FIBRE,
    elastic_principal,
    isotropic_first_piola_differential,
    local_offset,
    material_param,
    material_params,
    node_offset,
    particle_of_sorted,
    q1_stencil,
    q1_weight_hessian,
    stencil_node_slots,
    trial_deformation,
    unpack_block,
)


@qd.func
def band_row_bw(sol_params, phi, vel, diag_coef, dt, D_bar, t_bar):
    """Reverse of `band_row` (see soft_row): the adjoints of phi, vel and diag_coef from those of D and t."""
    phi_bar = gs.qd_float(0.0)
    vel_bar = gs.qd_float(0.0)
    diag_coef_bar = gs.qd_float(0.0)
    width = sol_params[4]
    if phi <= 0.0:
        imp, b, k, d_imp_d_x = gu.imp_aref_grad(sol_params, phi)
        diag_raw = diag_coef * (1.0 - imp) / imp
        D = 1.0 / qd.max(diag_raw, gs.EPS)
        # Zero where the floor holds D
        diag_raw_bar = gs.qd_float(0.0)
        if diag_raw > gs.EPS:
            diag_raw_bar = -D * D * D_bar
        diag_coef_bar = diag_raw_bar * (1.0 - imp) / imp
        # t = dt vel + dt^2 (-b vel - k imp phi), imp of |phi| / width
        imp_bar = diag_raw_bar * (-diag_coef / (imp * imp)) - dt * dt * k * phi * t_bar
        vel_bar = (dt - dt * dt * b) * t_bar
        phi_bar = -dt * dt * k * imp * t_bar - imp_bar * d_imp_d_x / width
    else:
        imp, b, k, d_imp_d_x = gu.imp_aref_grad(sol_params, 0.0)
        diag_raw = diag_coef * (1.0 - imp) / imp
        if diag_raw > gs.EPS:
            D = 1.0 / diag_raw
            diag_coef_bar = -D * D * D_bar * (1.0 - imp) / imp
        phi_bar = -t_bar
    return phi_bar, vel_bar, diag_coef_bar


@qd.func
def polar_rotation_bw(U, S, V, R_bar):
    """Reverse of the rotation R = U V^T of the polar decomposition F = R S, F = U S V^T: with M = U^T dF V, dR =
    U X V^T, X_ij = (M_ij - M_ji) / (s_i + s_j), so F_bar = U Y V^T, Y_ij = (G_ij - G_ji) / (s_i + s_j) for
    G = U^T R_bar V."""
    G = U.transpose() @ R_bar @ V
    Y = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if qd.static(i != j):
            Y[i, j] = (G[i, j] - G[j, i]) / qd.max(S[i, i] + S[j, j], gs.EPS)
    return U @ Y @ V.transpose()


@qd.func(requires_top_level=True)
def func_wall_contact_adjoint(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """The pullback of the wall rows of every valid particle (see func_build_wall_rows) into the adjoint of frame f.

    A row k on the trial displacement Delta = sum_a w_a u_a, of normal J = +-e_d, adds D c J . delta to lambda^T
    grad_u E, c = J . Delta - t and delta = sum_a w_a lambda_a, when active. Its pullback is that of the Rigid rows
    (see func_rigid_contact_adjoint) without the link: the weights, and the coefficients through the gap and the
    velocity along the normal and the inverse weight of the particle."""
    qd.loop_config(name="mpm_wall_contact_adjoint")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
        has_row = False
        for d in qd.static(range(3)):
            has_row = has_row or newton.contact_D[CONTACT_ROWS * s + d] > 0.0
        if has_row:
            slots = stencil_node_slots(grid, env, cell)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            w, grad_w = q1_stencil(x_p * params.inv_dx[()] - cell.cast(gs.qd_float), params.inv_dx[()])
            Delta = qd.Vector.zero(gs.qd_float, 3)
            delta = qd.Vector.zero(gs.qd_float, 3)
            for a in qd.static(range(8)):
                Delta += w[a] * qd.Vector([newton.u[3 * slots[a] + e] for e in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * (linear.row_offset + slots[a])
                delta += w[a] * qd.Vector([linear.solution[row + e] for e in qd.static(range(3))], dt=gs.qd_float)
            U, S, V = qd.svd(particles.F[f, i_p, i_b], gs.qd_float)
            R = U @ V.transpose()
            W = particles_info.contact_invweight[i_p]
            sol_params = qd.Vector([params.wall_sol_params[i] for i in qd.static(range(7))], dt=gs.qd_float)
            x_bar = qd.Vector.zero(gs.qd_float, 3)
            v_bar = qd.Vector.zero(gs.qd_float, 3)
            R_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
            Delta_bar = qd.Vector.zero(gs.qd_float, 3)
            delta_bar = qd.Vector.zero(gs.qd_float, 3)
            for d in qd.static(range(3)):
                k = CONTACT_ROWS * s + d
                D = newton.contact_D[k]
                J = qd.Vector([newton.contact_dir[3 * k + e] for e in qd.static(range(3))], dt=gs.qd_float)
                c = J.dot(Delta) - newton.contact_t[k]
                if D > 0.0 and c < 0.0:
                    a_k = J.dot(delta)
                    Delta_bar -= D * a_k * J
                    delta_bar -= D * c * J
                    # The row as func_build_wall_rows built it
                    phi = x_p[d] - params.wall_lower[d]
                    sign = gs.qd_float(1.0)
                    if params.wall_upper[d] - x_p[d] < phi:
                        phi = params.wall_upper[d] - x_p[d]
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
                row = 3 * (linear.row_offset + slots[a])
                lam_a = qd.Vector([linear.solution[row + e] for e in qd.static(range(3))], dt=gs.qd_float)
                grad_w_a = qd.Vector([grad_w[a, e] for e in qd.static(range(3))], dt=gs.qd_float)
                x_bar += (Delta_bar.dot(u_a) + delta_bar.dot(lam_a)) * grad_w_a
            particles.pos.grad[f, i_p, i_b] += x_bar
            particles.vel.grad[f, i_p, i_b] += v_bar
            particles.F.grad[f, i_p, i_b] += polar_rotation_bw(U, S, V, R_bar)


@qd.func(requires_top_level=True)
def func_constraint_adjoint(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    constraints: qd.template(),  # qd.StructField of the particle constraints (see MPMSolver.set_particle_constraints)
    links_state: array_class.LinksState,
    f,
):
    """The pullback of the constraint springs (see func_build_constraint_rows) into the adjoint of frame f and of the
    pose of the links the particles are attached to.

    A spring adds delta . g to lambda^T grad_u E, g = kappa (x + Delta - T) + gamma Delta with kappa = dt^2 k and
    gamma = dt c, Delta = sum_a w_a u_a and delta = sum_a w_a lambda_a. The position enters through the weights, the
    gap and Delta, x_bar = -[kappa delta + sum_a (lambda_a . g + (kappa + gamma) u_a . delta) grad w_a], and the target
    through the gap, T_bar = kappa delta. The stiffness and the mass behind the damping are model constants."""
    qd.loop_config(name="mpm_constraint_adjoint")
    for s in range(grid.n_valid[()]):
        spring = newton.constraint_spring[s]
        if spring > 0.0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            slots = stencil_node_slots(grid, env, cell)
            x_p = particles.pos[f, i_p, i_b]
            w, grad_w = q1_stencil(x_p * params.inv_dx[()] - cell.cast(gs.qd_float), params.inv_dx[()])
            Delta = qd.Vector.zero(gs.qd_float, 3)
            delta = qd.Vector.zero(gs.qd_float, 3)
            for a in qd.static(range(8)):
                Delta += w[a] * qd.Vector([newton.u[3 * slots[a] + e] for e in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * (linear.row_offset + slots[a])
                delta += w[a] * qd.Vector([linear.solution[row + e] for e in qd.static(range(3))], dt=gs.qd_float)
            damping = newton.constraint_damping[s]
            target = qd.Vector([newton.constraint_target[3 * s + e] for e in qd.static(range(3))], dt=gs.qd_float)
            g = spring * (x_p + Delta - target) + damping * Delta
            x_bar = -spring * delta
            for a in qd.static(range(8)):
                u_a = qd.Vector([newton.u[3 * slots[a] + e] for e in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * (linear.row_offset + slots[a])
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
def func_adjoint_reset_u_bar(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
):
    """Clear the adjoint of u*, which the reverse of G2P accumulates into `newton.du`, free once the solve is over."""
    qd.loop_config(name="mpm_adjoint_reset_u_bar")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        newton.du[i] = 0.0


@qd.func(requires_top_level=True)
def func_adjoint_g2p(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """Reverse of G2P and advection: x' = x + dt v, v = sum_a w_a u_a / dt, C = sum_a u_a grad w_a^T / dt and
    F' = (I + dt C) F, given the adjoint of frame f + 1.

    The direct terms add to the adjoint of frame f, the adjoint of u* to `newton.du`, by atomics over the stencil.
    The position enters through the weights and their gradients.
    """
    qd.loop_config(name="mpm_adjoint_g2p")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
        slots = stencil_node_slots(grid, env, cell)
        fx = particles.pos[f, i_p, i_b] * params.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, params.inv_dx[()])
        C = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            v_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float) / dt
            C += v_a.outer_product(qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float))
        F = particles.F[f, i_p, i_b]

        x_bar_next = particles.pos.grad[f + 1, i_p, i_b]
        F_bar_next = particles.F.grad[f + 1, i_p, i_b]
        v_bar = particles.vel.grad[f + 1, i_p, i_b] + dt * x_bar_next
        C_bar = particles.C.grad[f + 1, i_p, i_b] + dt * F_bar_next @ F.transpose()

        x_bar = x_bar_next
        for a in qd.static(range(8)):
            v_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float) / dt
            grad_w_a = qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
            u_bar = (w[a] * v_bar + C_bar @ grad_w_a) / dt
            for d in qd.static(range(3)):
                qd.atomic_add(newton.du[3 * slots[a] + d], u_bar[d])
            x_bar += v_a.dot(v_bar) * grad_w_a + q1_weight_hessian(fx, a, params.inv_dx[()]) @ (C_bar.transpose() @ v_a)

        particles.pos.grad[f, i_p, i_b] += x_bar
        particles.F.grad[f, i_p, i_b] += (qd.Matrix.identity(gs.qd_float, 3) + dt * C).transpose() @ F_bar_next
        particles.Jp.grad[f, i_p, i_b] += particles.Jp.grad[f + 1, i_p, i_b]
        particles.actu.grad[f, i_p, i_b] += particles.actu.grad[f + 1, i_p, i_b]


@qd.func(requires_top_level=True)
def func_adjoint_rhs(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
):
    """The right-hand side of the adjoint solve, the adjoint of u* with its fixed components, whose rows of the Newton
    matrix are the identity, at zero."""
    qd.loop_config(name="mpm_adjoint_rhs")
    for i in range(3 * grid.n_blocks[()] * BLOCK_CELLS):
        value = newton.du[i]
        if grid.dof_fixed[i] != 0:
            value = 0.0
        linear.b_rhs[3 * linear.row_offset + i] = value


@qd.func(requires_top_level=True)
def func_adjoint_nodes(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    force_fields: qd.template(),  # MpmForceFields
    params: qd.template(),  # MpmParams
    t,
    dt,
):
    """The kinetic pullback of lambda (`linear.solution`) and the reverse of v_free = v + dt acc(x_a, v), v = mom / m,
    on every node.

    grad_{u_a} E = m_a (u_a - dt v_free_a) gives v_free_bar = m_a dt lambda_a and m_bar = -lambda_a . (u_a - dt
    v_free_a). The force fields add dt (d acc / dv)^T v_free_bar to v_bar = v_free_bar (see
    'ForceField.get_acc_vel_vjp'), the node position x_a being fixed, so mom_bar = v_bar / m_a and
    m_bar -= v_a . v_bar / m_a. A fixed or massless node, whose velocity is zero, gets none.
    """
    qd.loop_config(name="mpm_adjoint_nodes")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        mass = grid.node_mass[node]
        mass_bar = gs.qd_float(0.0)
        mom_bar = qd.Vector.zero(gs.qd_float, 3)
        if mass > 0.0 and grid.node_fixed[node] == 0:
            lam = qd.Vector([linear.solution[3 * (linear.row_offset + node) + d] for d in qd.static(range(3))])
            u = qd.Vector([newton.u[3 * node + d] for d in qd.static(range(3))], dt=gs.qd_float)
            v = qd.Vector([grid.node_v[3 * node + d] for d in qd.static(range(3))], dt=gs.qd_float)
            v_free = qd.Vector([grid.node_v_free[3 * node + d] for d in qd.static(range(3))], dt=gs.qd_float)
            v_free_bar = (mass * dt) * lam
            v_bar = v_free_bar
            if qd.static(len(force_fields.force_fields) > 0):
                env, block = unpack_block(grid.block_coords[node // BLOCK_CELLS])
                x_n = (block * 4 + local_offset(node % BLOCK_CELLS)).cast(gs.qd_float) * params.dx[()]
                for i_ff in qd.static(range(len(force_fields.force_fields))):
                    v_bar += dt * force_fields.force_fields[i_ff].get_acc_vel_vjp(x_n, v, t, -1, v_free_bar)
            mom_bar = v_bar / mass
            mass_bar = -lam.dot(u - dt * v_free) - v.dot(v_bar) / mass
        for d in qd.static(range(3)):
            newton.adj_mom[3 * node + d] = mom_bar[d]
        newton.adj_mass[node] = mass_bar


@qd.func(requires_top_level=True)
def func_adjoint_particles(
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
    materials: MpmMaterialTable,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    params: qd.template(),  # MpmParams
    gravity: qd.template(),  # qd.Tensor of per-environment gravity, shape (B,) of vec3
    f,
    dt,
):
    """The elastic pullback of lambda and the reverse of P2G of every valid particle, into the adjoint of frame f.

    With A = I + sum_a u_a grad w_a^T, Lambda = sum_a lambda_a grad w_a^T and F_trial = A F, the elastic gradient
    gives phi = V dt^2 P(F_trial) : (Lambda F), and p_bar = -d phi / dp: F_bar -= V dt^2 (A^T Q + Lambda^T P) with
    Q = dP[Lambda F], the actuation through the fibre stiffness, and the position through the weight Hessians H_a,
    x_bar -= V dt^2 sum_a H_a F (Q^T u_a + P^T lambda_a). P2G spreads m_p (v_p + dt g + C_p (x_a - x_p)) and m_p
    by w_a, which reverses into v, C and x from the node adjoints of `func_adjoint_nodes`.
    """
    qd.loop_config(name="mpm_adjoint_particles")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
        slots = stencil_node_slots(grid, env, cell)
        x_p = particles.pos[f, i_p, i_b]
        fx = x_p * params.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, params.inv_dx[()])
        F_tr, grad_w_, F = trial_deformation(particles, grid, newton, params, f, i_p, i_b, cell, slots)

        Lam = qd.Matrix.zero(gs.qd_float, 3, 3)
        grad_u = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            grad_w_a = qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
            lam_a = qd.Vector([linear.solution[3 * (linear.row_offset + slots[a]) + d] for d in qd.static(range(3))])
            u_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
            Lam += lam_a.outer_product(grad_w_a)
            grad_u += u_a.outer_product(grad_w_a)
        A_def = qd.Matrix.identity(gs.qd_float, 3) + grad_u

        U, S, V = qd.svd(F_tr, gs.qd_float)
        sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float)
        slot = particles_info.material_idx[i_p]
        mu, lam = material_params(materials, params, slot, particles.Jp[f, i_p, i_b])
        psi, g, A_sigma = elastic_principal(materials, slot, sigma, mu, lam)
        P = U @ qd.Matrix([[g[0], 0.0, 0.0], [0.0, g[1], 0.0], [0.0, 0.0, g[2]]], dt=gs.qd_float) @ V.transpose()
        LamF = Lam @ F
        Q = isotropic_first_piola_differential(U, V, sigma, g, A_sigma, LamF)
        stiffness = material_param(materials, params, PARAM_FIBRE, slot)
        m_dir = particles_info.muscle_direction[i_p]
        ka = stiffness * particles.actu[f, i_p, i_b]
        F_tr_m = F_tr @ m_dir
        LamF_m = LamF @ m_dir
        P += ka * F_tr_m.outer_product(m_dir)
        Q += ka * LamF_m.outer_product(m_dir)

        scale = params.particle_volume[()] * dt * dt
        F_bar = -scale * (A_def.transpose() @ Q + Lam.transpose() @ P)
        actu_bar = -scale * stiffness * F_tr_m.dot(LamF_m)
        x_bar = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(8)):
            lam_a = qd.Vector([linear.solution[3 * (linear.row_offset + slots[a]) + d] for d in qd.static(range(3))])
            u_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
            H_a = q1_weight_hessian(fx, a, params.inv_dx[()])
            x_bar -= scale * (H_a @ (F @ (Q.transpose() @ u_a + P.transpose() @ lam_a)))

        # Reverse of P2G
        m_p = particles_info.mass[i_p] / params.mass_scale[()]
        v_p = particles.vel[f, i_p, i_b] + dt * gravity[i_b]
        C_p = particles.C[f, i_p, i_b]
        v_bar = qd.Vector.zero(gs.qd_float, 3)
        C_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            grad_w_a = qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
            x_a = (cell + node_offset(a)).cast(gs.qd_float) * params.dx[()]
            mom_bar = qd.Vector([newton.adj_mom[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
            mass_bar = newton.adj_mass[slots[a]]
            v_tilde = v_p + C_p @ (x_a - x_p)
            v_bar += (w[a] * m_p) * mom_bar
            C_bar += (w[a] * m_p) * mom_bar.outer_product(x_a - x_p)
            x_bar += (m_p * (v_tilde.dot(mom_bar) + mass_bar)) * grad_w_a - (w[a] * m_p) * (C_p.transpose() @ mom_bar)

        particles.pos.grad[f, i_p, i_b] += x_bar
        particles.vel.grad[f, i_p, i_b] += v_bar
        particles.C.grad[f, i_p, i_b] += C_bar
        particles.F.grad[f, i_p, i_b] += F_bar
        particles.actu.grad[f, i_p, i_b] += actu_bar


@qd.func(requires_top_level=True)
def func_adjoint_consume(
    particles: MpmParticles,
    particles_ng: MpmParticlesNG,
    params: qd.template(),  # MpmParams
    f,
):
    """Clear the adjoint of frame f + 1, which the substep consumed; an inactive particle, which the substep carries
    over unchanged, hands it to frame f."""
    qd.loop_config(name="mpm_adjoint_consume")
    for i_p, i_b in qd.ndrange(params.n_particles[()], params.n_envs[()]):
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
