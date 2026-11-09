"""Layer-1 stages of the adjoint of the implicit material point method (MPM) substep."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.array_class import LinksState

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
from .newton_linear import LinearBlock, NewtonLinearScratch


@qd.func
def return_map_bw(
    i_mat: int,
    kind: int,
    F: qd.types.matrix(3, 3),
    Jp: float,
    F_new_bar: qd.types.matrix(3, 3),
    Jp_new_bar: float,
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
):
    """Reverse of `return_map`: the adjoints of the trial deformation gradient F and of the plastic state Jp, given
    F_new_bar and Jp_new_bar."""
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
            i_mat, kind, sigma, Jp, materials, mpm_info
        )
        if is_changed:
            F_bar = isotropic_first_piola_differential(U, V, sigma, h, dh.transpose(), F_new_bar)
            F_bar += Jp_new_bar * (U @ diagonal(dJp_new_dsigma) @ V.transpose())
            Jp_bar = Jp_new_bar * dJp_new_dJp + dh_dJp.dot(diagonal_of(U.transpose() @ F_new_bar @ V))
    return F_bar, Jp_bar


@qd.func
def band_row_bw(
    sol_params: qd.types.vector(7),
    phi: float,
    vel: float,
    diag_coef: float,
    dt: float,
    stiffness_bar: float,
    target_bar: float,
):
    """Reverse of `band_row` (see soft_row), from the adjoints of the coefficients of a row to those of its inputs."""
    phi_bar = gs.qd_float(0.0)
    vel_bar = gs.qd_float(0.0)
    width = sol_params[4]
    # The impedance of the row, at the edge of the range phi = 0 short of it, chosen at runtime so that
    # `imp_aref_grad` is inlined once
    phi_row = gs.qd_float(0.0)
    if phi <= 0.0:
        phi_row = phi
    imp, damping, spring, d_imp_d_x = gu.imp_aref_grad(sol_params, phi_row)
    diag_raw = diag_coef * (1.0 - imp) / imp
    # Zero where diag_raw is clamped at the floor gs.EPS
    diag_raw_bar = gs.qd_float(0.0)
    if diag_raw > gs.EPS:
        stiffness = 1.0 / diag_raw
        diag_raw_bar = -stiffness * stiffness * stiffness_bar
    diag_coef_bar = diag_raw_bar * (1.0 - imp) / imp
    if phi <= 0.0:
        # target = dt vel + dt^2 (-damping vel - spring imp phi), imp of |phi| / width
        imp_bar = diag_raw_bar * (-diag_coef / (imp * imp)) - dt * dt * spring * phi * target_bar
        vel_bar = (dt - dt * dt * damping) * target_bar
        phi_bar = -dt * dt * spring * imp * target_bar - imp_bar * d_imp_d_x / width
    else:
        phi_bar = -target_bar
    return phi_bar, vel_bar, diag_coef_bar


@qd.func
def polar_rotation_bw(
    U: qd.types.matrix(3, 3), S: qd.types.matrix(3, 3), V: qd.types.matrix(3, 3), R_bar: qd.types.matrix(3, 3)
):
    """Reverse of the rotation R = U V^T of the polar decomposition of F = U S V^T: F_bar = U Y V^T with Y_ij = (G_ij -
    G_ji) / (sigma_i + sigma_j) and G = U^T R_bar V."""
    G = U.transpose() @ R_bar @ V
    sigma = diagonal_of(S)
    ones = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)
    return (
        U
        @ ((G - G.transpose()) / qd.max(sigma.outer_product(ones) + ones.outer_product(sigma), gs.EPS))
        @ V.transpose()
    )


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
        i_ps = pairs.sorted_idx[i_pair]
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        slots = stencil_node_slots(i_b, cell, grid)
        x_p = particles.pos[f, i_p, i_b]
        v_p = particles.vel[f, i_p, i_b]
        w, grad_w = q1_stencil(x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
        u_p = particle_node_sum(slots, w, newton.u)
        lam_p = particle_node_sum(rows.row_start[()] + slots, w, linear.solution)
        stiffness = pairs.stiffness[i_pair]
        direction = wall_normal(pairs.axis[i_pair], pairs.sign[i_pair])
        gap = direction.dot(u_p) - pairs.target[i_pair]
        if stiffness > 0.0 and gap < 0.0:
            lam_gap = direction.dot(lam_p)
            u_p_bar = -stiffness * lam_gap * direction
            lam_p_bar = -stiffness * gap * direction
            # The row as func_build_wall_pairs built it: the gap to the wall along the row direction, which is its
            # gradient in x_p
            phi = direction.dot(x_p - mpm_info.wall_lower[()])
            if pairs.sign[i_pair] < 0.0:
                phi = direction.dot(x_p - mpm_info.wall_upper[()])
            U, S, V = gu.qd_svd(particles.F[f, i_p, i_b])
            R = U @ V.transpose()
            invweight_p = particles_info.contact_invweight[i_p]
            n_rest = R.transpose() @ direction
            invweight = n_rest.dot(invweight_p @ n_rest)
            phi_bar, vel_bar, invweight_bar = band_row_bw(
                mpm_info.wall_sol_params[()],
                phi,
                direction.dot(v_p),
                invweight,
                dt,
                -gap * lam_gap,
                stiffness * lam_gap,
            )
            x_bar = phi_bar * direction
            v_bar = vel_bar * direction
            R_bar = (2.0 * invweight_bar) * direction.outer_product(invweight_p @ n_rest)
            weights_bar = stencil_dots(slots, newton.u, u_p_bar)
            weights_bar += stencil_dots(rows.row_start[()] + slots, linear.solution, lam_p_bar)
            x_bar += grad_w.transpose() @ weights_bar
            # A particle has a pair per wall it nears, up to three
            qd.atomic_add(particles.pos.grad[f, i_p, i_b], x_bar)
            qd.atomic_add(particles.vel.grad[f, i_p, i_b], v_bar)
            qd.atomic_add(particles.F.grad[f, i_p, i_b], polar_rotation_bw(U, S, V, R_bar))


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
    and of the poses of the links that hold them."""
    qd.loop_config(name="mpm_constraint_adjoint")
    for i_ps in range(grid.n_valid[()]):
        spring = newton.constraint_spring[i_ps]
        if spring > 0.0:
            i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
            slots = stencil_node_slots(i_b, cell, grid)
            x_p = particles.pos[f, i_p, i_b]
            w, grad_w = q1_stencil(x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
            u_p = particle_node_sum(slots, w, newton.u)
            lam_p = particle_node_sum(rows.row_start[()] + slots, w, linear.solution)
            damping = newton.constraint_damping[i_ps]
            grad_spring = spring * (x_p + u_p - newton.constraint_target[i_ps]) + damping * u_p
            weights_bar = stencil_dots(rows.row_start[()] + slots, linear.solution, grad_spring)
            weights_bar += stencil_dots(slots, newton.u, (spring + damping) * lam_p)
            x_bar = -spring * lam_p - grad_w.transpose() @ weights_bar
            particles.pos.grad[f, i_p, i_b] = particles.pos.grad[f, i_p, i_b] + x_bar
            i_l = constraints[i_p, i_b].link_idx
            if i_l >= 0:
                target_bar = spring * lam_p
                qd.atomic_add(links_state.pos.grad[i_l, i_b], target_bar)
                qd.atomic_add(
                    links_state.quat.grad[i_l, i_b],
                    gu.qd_transform_by_quat_grad_quat(
                        constraints[i_p, i_b].link_local_pos, links_state.quat[i_l, i_b], target_bar
                    ),
                )


@qd.func(requires_top_level=True)
def func_adjoint_reset_u_bar(grid: MpmGridScratch, newton: MpmNewtonScratch):
    """Clear the adjoint of u*, which the reverse of the grid-to-particle transfer (G2P) accumulates into `newton.du`
    (free once the forward solve is over)."""
    qd.loop_config(name="mpm_adjoint_reset_u_bar")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        newton.du[i_n] = qd.Vector.zero(gs.qd_float, 3)


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
    """Reverse of the grid-to-particle transfer (G2P) and of the advection, from the adjoint of the next frame."""
    qd.loop_config(name="mpm_adjoint_g2p")
    for i_ps in range(grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        slots = stencil_node_slots(i_b, cell, grid)
        fx = particles.pos[f, i_p, i_b] * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
        C = stencil_gradient(slots, grad_w, newton.u) / dt
        F = particles.F[f, i_p, i_b]
        i_mat = particles_info.material_idx[i_p]
        F_trial = (qd.Matrix.identity(gs.qd_float, 3) + dt * C) @ F
        F_trial_bar, Jp_bar = return_map_bw(
            i_mat,
            materials.materials_plastic[i_mat],
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

        F_bar = (qd.Matrix.identity(gs.qd_float, 3) + dt * C).transpose() @ F_trial_bar
        particles.pos.grad[f, i_p, i_b] = particles.pos.grad[f, i_p, i_b] + x_bar
        particles.F.grad[f, i_p, i_b] = particles.F.grad[f, i_p, i_b] + F_bar
        particles.Jp.grad[f, i_p, i_b] = particles.Jp.grad[f, i_p, i_b] + Jp_bar
        particles.actu.grad[f, i_p, i_b] = particles.actu.grad[f, i_p, i_b] + particles.actu.grad[f + 1, i_p, i_b]


@qd.func(requires_top_level=True)
def func_adjoint_rhs(grid: MpmGridScratch, newton: MpmNewtonScratch, linear: NewtonLinearScratch, rows: LinearBlock):
    """Set the right-hand side of the adjoint solve to the adjoint of u*, zero on the fixed nodes, whose rows of
    the Newton matrix are the identity."""
    qd.loop_config(name="mpm_adjoint_rhs")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        value = newton.du[i_n]
        if grid.nodes_is_fixed[i_n] != 0:
            value = qd.Vector.zero(gs.qd_float, 3)
        linear.b_rhs[rows.row_start[()] + i_n] = value


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
    """Pull lambda back through the kinetic term and reverse the free velocity of every node into
    `newton.nodes_momentum_bar` and `newton.nodes_mass_bar`."""
    qd.loop_config(name="mpm_adjoint_nodes")
    for i_n in range(grid.n_blocks[()] * BLOCK_CELLS):
        mass = grid.nodes_mass[i_n]
        mass_bar = gs.qd_float(0.0)
        momentum_bar = qd.Vector.zero(gs.qd_float, 3)
        if mass > 0.0 and grid.nodes_is_fixed[i_n] == 0:
            lam = linear.solution[rows.row_start[()] + i_n]
            u = newton.u[i_n]
            v = grid.nodes_vel[i_n]
            v_free = grid.nodes_vel_free[i_n]
            v_free_bar = (mass * dt) * lam
            v_bar = v_free_bar
            if qd.static(len(force_fields.force_fields) > 0):
                i_b, block = unpack_block(grid.blocks_key[i_n // BLOCK_CELLS])
                x_n = (block * 4 + local_offset(i_n % BLOCK_CELLS)).cast(gs.qd_float) * mpm_info.dx[()]
                for i_ff in qd.static(range(len(force_fields.force_fields))):
                    v_bar += dt * force_fields.force_fields[i_ff].get_acc_grad_vel(x_n, v, t, -1, v_free_bar)
            momentum_bar = v_bar / mass
            mass_bar = -lam.dot(u - dt * v_free) - v.dot(v_bar) / mass
        newton.nodes_momentum_bar[i_n] = momentum_bar
        newton.nodes_mass_bar[i_n] = mass_bar


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
    """Pull lambda back through the elastic term and reverse the particle-to-grid transfer (P2G) of valid particles."""
    qd.loop_config(name="mpm_adjoint_particles")
    for i_ps in range(grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
        slots = stencil_node_slots(i_b, cell, grid)
        x_p = particles.pos[f, i_p, i_b]
        fx = x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float)
        w, grad_w = q1_stencil(fx, mpm_info.inv_dx[()])
        F_trial, _, F = trial_deformation(f, i_p, i_b, cell, slots, particles, newton, mpm_info)
        lam_grad = stencil_gradient(rows.row_start[()] + slots, grad_w, linear.solution)
        F_increment = qd.Matrix.identity(gs.qd_float, 3) + stencil_gradient(slots, grad_w, newton.u)

        i_mat, U, sigma, V, g, A, P = particle_elastic_stress(
            f, i_p, i_b, F_trial, particles, particles_info, materials, mpm_info
        )
        lam_grad_F = lam_grad @ F
        dP = isotropic_first_piola_differential(U, V, sigma, g, A, lam_grad_F)
        # Snow scales mu and lam by exp(SNOW_HARDENING (1 - Jp)), to which its elastic energy and compression penalty
        # are proportional, so phi = V dt^2 P : (lam_grad F) moves by -SNOW_HARDENING phi with Jp
        Jp_bar = gs.qd_float(0.0)
        if materials.materials_plastic[i_mat] == PLASTIC_SNOW:
            Jp_bar = qd.static(SNOW_HARDENING) * mpm_info.particle_volume[()] * dt * dt * (P * lam_grad_F).sum()
        muscle_direction = particles_info.muscle_direction[i_p]
        active_stiffness, dactive_stiffness = active_fibre_stiffness(
            i_mat, particles.actu[f, i_p, i_b], materials, mpm_info
        )
        F_fibre = F_trial @ muscle_direction
        lam_grad_F_fibre = lam_grad_F @ muscle_direction
        P += active_stiffness * F_fibre.outer_product(muscle_direction)
        dP += active_stiffness * lam_grad_F_fibre.outer_product(muscle_direction)

        scale = mpm_info.particle_volume[()] * dt * dt
        F_bar = -scale * (F_increment.transpose() @ dP + lam_grad.transpose() @ P)
        actu_bar = -scale * dactive_stiffness * F_fibre.dot(lam_grad_F_fibre)
        x_bar = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(8)):
            lam_a = linear.solution[rows.row_start[()] + slots[a]]
            u_a = newton.u[slots[a]]
            H_a = q1_weight_hessian(fx, mpm_info.inv_dx[()], a)
            x_bar -= scale * (H_a @ (F @ (dP.transpose() @ u_a + P.transpose() @ lam_a)))

        # Reverse of the particle-to-grid transfer (P2G)
        m_p = particles_info.mass[i_p] / mpm_info.mass_scale[()]
        v_p = particles.vel[f, i_p, i_b] + dt * gravity[i_b]
        C_p = particles.C[f, i_p, i_b]
        v_bar = qd.Vector.zero(gs.qd_float, 3)
        C_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a in qd.static(range(8)):
            grad_w_a = grad_w[a, :]
            x_a = (cell + node_offset(a)).cast(gs.qd_float) * mpm_info.dx[()]
            momentum_bar = newton.nodes_momentum_bar[slots[a]]
            mass_bar = newton.nodes_mass_bar[slots[a]]
            v_tilde = v_p + C_p @ (x_a - x_p)
            v_bar += (w[a] * m_p) * momentum_bar
            C_bar += (w[a] * m_p) * momentum_bar.outer_product(x_a - x_p)
            x_bar += (m_p * (v_tilde.dot(momentum_bar) + mass_bar)) * grad_w_a - (w[a] * m_p) * (
                C_p.transpose() @ momentum_bar
            )

        particles.pos.grad[f, i_p, i_b] = particles.pos.grad[f, i_p, i_b] + x_bar
        particles.vel.grad[f, i_p, i_b] = particles.vel.grad[f, i_p, i_b] + v_bar
        particles.C.grad[f, i_p, i_b] = particles.C.grad[f, i_p, i_b] + C_bar
        particles.F.grad[f, i_p, i_b] = particles.F.grad[f, i_p, i_b] + F_bar
        particles.Jp.grad[f, i_p, i_b] = particles.Jp.grad[f, i_p, i_b] + Jp_bar
        particles.actu.grad[f, i_p, i_b] = particles.actu.grad[f, i_p, i_b] + actu_bar


@qd.func(requires_top_level=True)
def func_adjoint_consume(f: int, particles: MpmParticles, particles_ng: MpmParticlesNG, mpm_info: MpmInfo):
    """Clear the adjoint of the next frame, which the substep consumed, after adding that of an inactive particle, which
    the substep carries over unchanged, to frame f."""
    qd.loop_config(name="mpm_adjoint_consume")
    for i_p, i_b in qd.ndrange(mpm_info.n_particles[()], mpm_info.n_envs[()]):
        if not particles_ng.active[f, i_p, i_b]:
            particles.pos.grad[f, i_p, i_b] = particles.pos.grad[f, i_p, i_b] + particles.pos.grad[f + 1, i_p, i_b]
            particles.vel.grad[f, i_p, i_b] = particles.vel.grad[f, i_p, i_b] + particles.vel.grad[f + 1, i_p, i_b]
            particles.C.grad[f, i_p, i_b] = particles.C.grad[f, i_p, i_b] + particles.C.grad[f + 1, i_p, i_b]
            particles.F.grad[f, i_p, i_b] = particles.F.grad[f, i_p, i_b] + particles.F.grad[f + 1, i_p, i_b]
            particles.Jp.grad[f, i_p, i_b] = particles.Jp.grad[f, i_p, i_b] + particles.Jp.grad[f + 1, i_p, i_b]
            particles.actu.grad[f, i_p, i_b] = particles.actu.grad[f, i_p, i_b] + particles.actu.grad[f + 1, i_p, i_b]
        particles.pos.grad[f + 1, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        particles.vel.grad[f + 1, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        particles.C.grad[f + 1, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        particles.F.grad[f + 1, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        particles.Jp.grad[f + 1, i_p, i_b] = gs.qd_float(0.0)
        particles.actu.grad[f + 1, i_p, i_b] = gs.qd_float(0.0)
