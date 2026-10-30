"""Layer-1 stages of the adjoint of the implicit MPM substep (design §8.2).

The substep maps the particle state of frame f to that of frame f + 1 through the grid displacement u* that minimizes
the incremental potential E(u; p), `grad_u E(u*; p) = 0`. Its backward pass consumes the adjoint of frame f + 1 in the
`.grad` of the particle arrays, clears it, and adds the adjoint of frame f:

1. reverse of G2P and advection, which gives the direct terms of frame f and the adjoint u_bar of u*,
2. the adjoint solve H lambda = u_bar by the PCG of the forward solve, on its SPD-projected Hessian at u*,
3. the implicit-function pullback p_bar = -(d grad_u E / dp)^T lambda over the kinetic and elastic terms, with the
   exact differential of the first Piola-Kirchhoff stress,
4. reverse of P2G from the adjoint of the node momentum and mass.

The gradients are approximate where the projection clamps a mode of the elastic Hessian. The contact rows enter the
adjoint solve through the Hessian, their coefficients held fixed in the pullback. The return maps are the identity:
only the materials without plasticity reach here.
"""

import quadrants as qd

import genesis as gs

from .mpm_data import BLOCK_CELLS, MpmMaterialTable, MpmParticles, MpmParticlesInfo, MpmParticlesNG
from .mpm_func import (
    PARAM_FIBRE,
    elastic_principal,
    isotropic_first_piola_differential,
    material_param,
    material_params,
    node_offset,
    particle_of_sorted,
    q1_stencil,
    q1_weight_hessian,
    stencil_node_slots,
    trial_deformation,
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
    dt,
):
    """The kinetic pullback of lambda (`linear.solution`) and the reverse of v_free = mom / m on every node.

    grad_{u_a} E = m_a (u_a - dt v_free_a) gives v_free_bar = m_a dt lambda_a and m_bar = -lambda_a . (u_a - dt
    v_free_a), so mom_bar = v_free_bar / m_a and m_bar -= v_free_a . v_free_bar / m_a. A fixed or massless node, whose
    velocity is zero, gets none.
    """
    qd.loop_config(name="mpm_adjoint_nodes")
    for node in range(grid.n_blocks[()] * BLOCK_CELLS):
        mass = grid.node_mass[node]
        mass_bar = gs.qd_float(0.0)
        mom_bar = qd.Vector.zero(gs.qd_float, 3)
        if mass > 0.0 and grid.node_fixed[node] == 0:
            lam = qd.Vector([linear.solution[3 * (linear.row_offset + node) + d] for d in qd.static(range(3))])
            u = qd.Vector([newton.u[3 * node + d] for d in qd.static(range(3))], dt=gs.qd_float)
            v_free = qd.Vector([grid.node_v_free[3 * node + d] for d in qd.static(range(3))], dt=gs.qd_float)
            v_free_bar = (mass * dt) * lam
            mom_bar = v_free_bar / mass
            mass_bar = -lam.dot(u - dt * v_free) - v_free.dot(v_free_bar) / mass
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
