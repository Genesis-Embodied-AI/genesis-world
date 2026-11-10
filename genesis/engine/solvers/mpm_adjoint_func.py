"""Layer-0 reverses and Layer-1 pullbacks of the adjoint of the implicit material point method (MPM) substep."""

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
    N_MATERIAL_PARAMS,
    PARAM_FIBRE,
    PLASTIC_LIQUID,
    PLASTIC_NONE,
    PLASTIC_SNOW,
    active_fibre_stiffness,
    contact_invweight,
    diagonal,
    diagonal_of,
    elastic_scale,
    first_piola,
    g2p_advect,
    gather_stencil,
    gather_stencil_scalars,
    is_free_node,
    isotropic_first_piola_differential,
    local_offset,
    material_values,
    node_offset,
    node_position,
    particle_elastic_state,
    particle_mass,
    particle_of_sorted,
    q1_weight_hessian,
    return_map_principal,
    spring_terms,
    stencil_gradient,
    stencil_interpolate,
    stencil_weights,
    unpack_block,
    wall_distance,
    wall_normal,
)
from .newton_linear import LinearBlock, NewtonLinearScratch

# ================================================ Layer 0: reverses =================================================


@qd.func
def return_map_backward(
    kind: int,
    F: qd.types.matrix(3, 3),
    Jp: float,
    F_new_bar: qd.types.matrix(3, 3),
    Jp_new_bar: float,
    params: qd.types.vector(N_MATERIAL_PARAMS),
    von_mises_sigma_min: float,
    sand_sigma_min: float,
    liquid_j_min: float,
):
    """Reverse of `return_map`, from the adjoints of the returned state to those of the trial state."""
    F_bar = F_new_bar
    Jp_bar = Jp_new_bar
    if kind == PLASTIC_LIQUID:
        det = F.determinant()
        F_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
        if det > liquid_j_min and det < 1.0:
            # d det F / dF is its cofactor, det F F^-T
            J_bar = F_new_bar.trace() / (3.0 * qd.pow(det, 2.0 / 3.0))
            F_bar = (J_bar * det) * F.inverse().transpose()
    elif kind != PLASTIC_NONE:
        U, S, V = gu.qd_svd(F)
        sigma = diagonal_of(S)
        h, Jp_new, is_changed, dh, dh_dJp, dJp_new_dsigma, dJp_new_dJp = return_map_principal(
            kind, sigma, Jp, params, von_mises_sigma_min, sand_sigma_min
        )
        if is_changed:
            F_bar = isotropic_first_piola_differential(U, V, sigma, h, dh.transpose(), F_new_bar)
            F_bar += Jp_new_bar * (U @ diagonal(dJp_new_dsigma) @ V.transpose())
            Jp_bar = Jp_new_bar * dJp_new_dJp + dh_dJp.dot(diagonal_of(U.transpose() @ F_new_bar @ V))
    return F_bar, Jp_bar


@qd.func
def band_row_backward(
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
def polar_rotation_backward(
    U: qd.types.matrix(3, 3), S: qd.types.matrix(3, 3), V: qd.types.matrix(3, 3), R_bar: qd.types.matrix(3, 3)
):
    """Reverse of the rotation of the polar decomposition of a deformation gradient."""
    G = U.transpose() @ R_bar @ V
    sigma = diagonal_of(S)
    ones = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)
    return (
        U
        @ ((G - G.transpose()) / qd.max(sigma.outer_product(ones) + ones.outer_product(sigma), gs.EPS))
        @ V.transpose()
    )


@qd.func
def soft_row_backward(
    direction: qd.types.vector(3),
    stiffness: float,
    target: float,
    u_rel: qd.types.vector(3),
    lam_rel: qd.types.vector(3),
):
    """Reverse of the gradient of a soft row (see soft_row_terms) contracted with the adjoint solution lam_rel: whether
    the row is active, and the adjoints of its stiffness, target and direction and of the relative displacement and of
    the adjoint solution along it."""
    stiffness_bar = gs.qd_float(0.0)
    target_bar = gs.qd_float(0.0)
    direction_bar = qd.Vector.zero(gs.qd_float, 3)
    u_rel_bar = qd.Vector.zero(gs.qd_float, 3)
    lam_rel_bar = qd.Vector.zero(gs.qd_float, 3)
    gap = direction.dot(u_rel) - target
    is_active = stiffness > 0.0 and gap < 0.0
    if is_active:
        lam_gap = direction.dot(lam_rel)
        stiffness_bar = -gap * lam_gap
        target_bar = stiffness * lam_gap
        direction_bar = -stiffness * (lam_gap * u_rel + gap * lam_rel)
        u_rel_bar = -stiffness * lam_gap * direction
        lam_rel_bar = -stiffness * gap * direction
    return is_active, stiffness_bar, target_bar, direction_bar, u_rel_bar, lam_rel_bar


@qd.func
def contact_invweight_backward(
    U: qd.types.matrix(3, 3),
    S: qd.types.matrix(3, 3),
    V: qd.types.matrix(3, 3),
    invweight_rest: qd.types.matrix(3, 3),
    n: qd.types.vector(3),
    invweight_bar: float,
):
    """Reverse of `contact_invweight`, from the adjoint of the inverse weight to those of the direction and of the
    deformation gradient."""
    R = U @ V.transpose()
    W_n_rest = invweight_rest @ (R.transpose() @ n)
    R_bar = (2.0 * invweight_bar) * n.outer_product(W_n_rest)
    return 2.0 * invweight_bar * (R @ W_n_rest), polar_rotation_backward(U, S, V, R_bar)


@qd.func
def stencil_position_backward(
    grad_w: qd.types.matrix(8, 3),
    u_nodes: qd.types.matrix(8, 3),
    lam_nodes: qd.types.matrix(8, 3),
    u_bar: qd.types.vector(3),
    lam_bar: qd.types.vector(3),
    x_bar: qd.types.vector(3),
):
    """Add to the adjoint of the position of a particle the reverse of the weights of its stencil in sum_a w_a u_a and
    sum_a w_a lam_a, from the adjoints of these sums."""
    return x_bar + grad_w.transpose() @ (u_nodes @ u_bar + lam_nodes @ lam_bar)


@qd.func
def wall_pair_backward(
    axis: int,
    sign: float,
    stiffness: float,
    target: float,
    sol_params: qd.types.vector(7),
    lower: qd.types.vector(3),
    upper: qd.types.vector(3),
    x_p: qd.types.vector(3),
    v_p: qd.types.vector(3),
    F: qd.types.matrix(3, 3),
    invweight_rest: qd.types.matrix(3, 3),
    w: qd.types.vector(8),
    grad_w: qd.types.matrix(8, 3),
    u_nodes: qd.types.matrix(8, 3),
    lam_nodes: qd.types.matrix(8, 3),
    dt: float,
):
    """Reverse of the gradient of a wall row (see func_build_wall_pairs) contracted with the adjoint solution: whether
    the row is active, and the adjoints of the position, velocity and deformation gradient of its particle."""
    normal = wall_normal(axis, sign)
    is_active, stiffness_bar, target_bar, _, u_p_bar, lam_p_bar = soft_row_backward(
        normal, stiffness, target, stencil_interpolate(w, u_nodes), stencil_interpolate(w, lam_nodes)
    )
    x_bar = qd.Vector.zero(gs.qd_float, 3)
    v_bar = qd.Vector.zero(gs.qd_float, 3)
    F_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
    if is_active:
        U, S, V = gu.qd_svd(F)
        phi_bar, vel_bar, invweight_bar = band_row_backward(
            sol_params,
            wall_distance(sign, normal, x_p, lower, upper),
            normal.dot(v_p),
            contact_invweight(U, V, invweight_rest, normal),
            dt,
            stiffness_bar,
            target_bar,
        )
        _, F_bar = contact_invweight_backward(U, S, V, invweight_rest, normal, invweight_bar)
        x_bar = stencil_position_backward(grad_w, u_nodes, lam_nodes, u_p_bar, lam_p_bar, phi_bar * normal)
        v_bar = vel_bar * normal
    return is_active, x_bar, v_bar, F_bar


@qd.func
def constraint_spring_backward(
    spring: float,
    damping: float,
    target: qd.types.vector(3),
    x_p: qd.types.vector(3),
    w: qd.types.vector(8),
    grad_w: qd.types.matrix(8, 3),
    u_nodes: qd.types.matrix(8, 3),
    lam_nodes: qd.types.matrix(8, 3),
):
    """Reverse of the gradient of a constraint spring (see spring_terms) contracted with the adjoint solution, to the
    adjoints of the position of its particle and of its target."""
    u_p = stencil_interpolate(w, u_nodes)
    lam_p = stencil_interpolate(w, lam_nodes)
    _, grad_spring, hess_spring = spring_terms(spring, damping, target, x_p, u_p)
    weights_bar = lam_nodes @ grad_spring + u_nodes @ (hess_spring * lam_p)
    return -spring * lam_p - grad_w.transpose() @ weights_bar, spring * lam_p


@qd.func
def g2p_advect_backward(
    u_nodes: qd.types.matrix(8, 3),
    w: qd.types.vector(8),
    grad_w: qd.types.matrix(8, 3),
    fx: qd.types.vector(3),
    inv_dx: float,
    dt: float,
    C: qd.types.matrix(3, 3),
    F: qd.types.matrix(3, 3),
    F_trial_bar: qd.types.matrix(3, 3),
    x_bar_next: qd.types.vector(3),
    v_bar_next: qd.types.vector(3),
    C_bar_next: qd.types.matrix(3, 3),
):
    """Reverse of `g2p_advect`, from the adjoints of the next state of a particle, its trial deformation gradient's
    through the return map, to those of the displacements of the nodes of its stencil, of its position and of its
    deformation gradient."""
    v_bar = v_bar_next + dt * x_bar_next
    C_bar = C_bar_next + dt * F_trial_bar @ F.transpose()
    u_bars = qd.Matrix.zero(gs.qd_float, 8, 3)
    x_bar = x_bar_next
    for a in qd.static(range(8)):
        v_a = u_nodes[a, :] / dt
        grad_w_a = grad_w[a, :]
        u_bar = (w[a] * v_bar + C_bar @ grad_w_a) / dt
        for d in qd.static(range(3)):
            u_bars[a, d] = u_bar[d]
        x_bar += v_a.dot(v_bar) * grad_w_a + q1_weight_hessian(a, fx, inv_dx) @ (C_bar.transpose() @ v_a)
    return u_bars, x_bar, (qd.Matrix.identity(gs.qd_float, 3) + dt * C).transpose() @ F_trial_bar


@qd.func
def free_velocity_adjoint(mass: float, dt: float, lam: qd.types.vector(3)):
    """Adjoint of the free velocity of a node from the adjoint solution, through its kinetic gradient (see
    node_gradient)."""
    return (mass * dt) * lam


@qd.func
def node_backward(
    mass: float,
    lam: qd.types.vector(3),
    u: qd.types.vector(3),
    v: qd.types.vector(3),
    v_free: qd.types.vector(3),
    acc_vel_bar: qd.types.vector(3),
    dt: float,
):
    """Reverse of the free velocity and of the kinetic gradient of a free node, from the adjoint solution to the
    adjoints of its momentum and mass, with the velocity adjoint of the force fields at the adjoint of its free velocity
    (see ForceField.get_acc_grad_vel)."""
    v_bar = free_velocity_adjoint(mass, dt, lam) + dt * acc_vel_bar
    return v_bar / mass, -lam.dot(u - dt * v_free) - v.dot(v_bar) / mass


@qd.func
def elastic_pullback(
    u_nodes: qd.types.matrix(8, 3),
    lam_nodes: qd.types.matrix(8, 3),
    grad_w: qd.types.matrix(8, 3),
    fx: qd.types.vector(3),
    inv_dx: float,
    F: qd.types.matrix(3, 3),
    F_trial: qd.types.matrix(3, 3),
    U: qd.types.matrix(3, 3),
    sigma: qd.types.vector(3),
    V: qd.types.matrix(3, 3),
    g: qd.types.vector(3),
    A: qd.types.matrix(3, 3),
    plastic: int,
    params: qd.types.vector(N_MATERIAL_PARAMS),
    actu: float,
    muscle_direction: qd.types.vector(3),
    scale: float,
    snow_hardening: float,
):
    """Reverse of the elastic gradient of a particle contracted with the adjoint solution, to the adjoints of its
    position, deformation gradient, plastic state and actuation."""
    lam_grad = stencil_gradient(grad_w, lam_nodes)
    F_increment = qd.Matrix.identity(gs.qd_float, 3) + stencil_gradient(grad_w, u_nodes)
    P = first_piola(U, g, V)
    lam_grad_F = lam_grad @ F
    dP = isotropic_first_piola_differential(U, V, sigma, g, A, lam_grad_F)
    # Snow scales mu and lam by exp(snow_hardening (1 - Jp)), to which its elastic energy and compression penalty
    # are proportional, so phi = V dt^2 P : (lam_grad F) moves by -snow_hardening phi with Jp
    Jp_bar = gs.qd_float(0.0)
    if plastic == PLASTIC_SNOW:
        Jp_bar = snow_hardening * scale * (P * lam_grad_F).sum()
    active_stiffness, dactive_stiffness = active_fibre_stiffness(params[PARAM_FIBRE], actu)
    F_fibre = F_trial @ muscle_direction
    lam_grad_F_fibre = lam_grad_F @ muscle_direction
    P += active_stiffness * F_fibre.outer_product(muscle_direction)
    dP += active_stiffness * lam_grad_F_fibre.outer_product(muscle_direction)
    F_bar = -scale * (F_increment.transpose() @ dP + lam_grad.transpose() @ P)
    actu_bar = -scale * dactive_stiffness * F_fibre.dot(lam_grad_F_fibre)
    x_bar = qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(8)):
        x_bar -= scale * (
            q1_weight_hessian(a, fx, inv_dx) @ (F @ (dP.transpose() @ u_nodes[a, :] + P.transpose() @ lam_nodes[a, :]))
        )
    return x_bar, F_bar, Jp_bar, actu_bar


@qd.func
def p2g_pullback(
    w: qd.types.vector(8),
    grad_w: qd.types.matrix(8, 3),
    cell,
    dx: float,
    x_p: qd.types.vector(3),
    v_p: qd.types.vector(3),
    gravity: qd.types.vector(3),
    dt: float,
    C_p: qd.types.matrix(3, 3),
    mass: float,
    momentum_bar_nodes: qd.types.matrix(8, 3),
    mass_bar_nodes: qd.types.vector(8),
    x_bar: qd.types.vector(3),
):
    """Reverse of the particle-to-grid transfer (P2G) of a particle (see apic_contribution), from the adjoints of the
    momentum and mass of the nodes of its stencil, adding to the adjoint of its position."""
    v_moved = v_p + dt * gravity
    v_bar = qd.Vector.zero(gs.qd_float, 3)
    C_bar = qd.Matrix.zero(gs.qd_float, 3, 3)
    for a in qd.static(range(8)):
        grad_w_a = grad_w[a, :]
        arm = node_position(cell + node_offset(a), dx) - x_p
        momentum_bar = momentum_bar_nodes[a, :]
        v_tilde = v_moved + C_p @ arm
        v_bar += (w[a] * mass) * momentum_bar
        C_bar += (w[a] * mass) * momentum_bar.outer_product(arm)
        x_bar += (mass * (v_tilde.dot(momentum_bar) + mass_bar_nodes[a])) * grad_w_a - (w[a] * mass) * (
            C_p.transpose() @ momentum_bar
        )
    return x_bar, v_bar, C_bar


# ================================================ Layer 1: pullbacks ================================================


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
    """Pull the wall pairs (see func_build_wall_pairs) back into the adjoint of the particles, as the Rigid rows with a
    fixed wall (see kernel_rigid_contact_adjoint)."""
    qd.loop_config(name="mpm_wall_contact_adjoint")
    for i_pair in range(pairs.n_pairs[()]):
        i_p, i_b, cell = particle_of_sorted(pairs.sorted_idx[i_pair], grid, mpm_info)
        x_p = particles.pos[f, i_p, i_b]
        slots, fx, w, grad_w = stencil_weights(i_b, cell, x_p, grid, mpm_info)
        is_active, x_bar, v_bar, F_bar = wall_pair_backward(
            pairs.axis[i_pair],
            pairs.sign[i_pair],
            pairs.stiffness[i_pair],
            pairs.target[i_pair],
            mpm_info.wall_sol_params[()],
            mpm_info.wall_lower[()],
            mpm_info.wall_upper[()],
            x_p,
            particles.vel[f, i_p, i_b],
            particles.F[f, i_p, i_b],
            particles_info.contact_invweight[i_p],
            w,
            grad_w,
            gather_stencil(slots, newton.u),
            gather_stencil(rows.row_start[()] + slots, linear.solution),
            dt,
        )
        if is_active:
            # A particle has a pair per wall it nears, up to three
            qd.atomic_add(particles.pos.grad[f, i_p, i_b], x_bar)
            qd.atomic_add(particles.vel.grad[f, i_p, i_b], v_bar)
            qd.atomic_add(particles.F.grad[f, i_p, i_b], F_bar)


@qd.func(requires_top_level=True)
def func_constraint_adjoint(
    f: int,
    constraints: qd.template(),
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    links_state: LinksState,
    mpm_info: MpmInfo,
):
    """Pull the constraint springs (see func_build_constraint_rows) back into the adjoints of the particles and of the
    poses of the links that hold them."""
    qd.loop_config(name="mpm_constraint_adjoint")
    for i_ps in range(grid.n_valid[()]):
        spring = newton.constraint_spring[i_ps]
        if spring > 0.0:
            i_p, i_b, cell = particle_of_sorted(i_ps, grid, mpm_info)
            x_p = particles.pos[f, i_p, i_b]
            slots, fx, w, grad_w = stencil_weights(i_b, cell, x_p, grid, mpm_info)
            x_bar, target_bar = constraint_spring_backward(
                spring,
                newton.constraint_damping[i_ps],
                newton.constraint_target[i_ps],
                x_p,
                w,
                grad_w,
                gather_stencil(slots, newton.u),
                gather_stencil(rows.row_start[()] + slots, linear.solution),
            )
            particles.pos.grad[f, i_p, i_b] = particles.pos.grad[f, i_p, i_b] + x_bar
            i_l = constraints[i_p, i_b].link_idx
            if i_l >= 0:
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
        x_p = particles.pos[f, i_p, i_b]
        slots, fx, w, grad_w = stencil_weights(i_b, cell, x_p, grid, mpm_info)
        u_nodes = gather_stencil(slots, newton.u)
        F = particles.F[f, i_p, i_b]
        x_new, v_new, C, F_trial = g2p_advect(u_nodes, w, grad_w, dt, x_p, F)
        family, plastic, params = material_values(particles_info.material_idx[i_p], materials, mpm_info)
        F_trial_bar, Jp_bar = return_map_backward(
            plastic,
            F_trial,
            particles.Jp[f, i_p, i_b],
            particles.F.grad[f + 1, i_p, i_b],
            particles.Jp.grad[f + 1, i_p, i_b],
            params,
            mpm_info.von_mises_sigma_min[()],
            mpm_info.sand_sigma_min[()],
            mpm_info.liquid_j_min[()],
        )
        u_bars, x_bar, F_bar = g2p_advect_backward(
            u_nodes,
            w,
            grad_w,
            fx,
            mpm_info.inv_dx[()],
            dt,
            C,
            F,
            F_trial_bar,
            particles.pos.grad[f + 1, i_p, i_b],
            particles.vel.grad[f + 1, i_p, i_b],
            particles.C.grad[f + 1, i_p, i_b],
        )
        for a in qd.static(range(8)):
            qd.atomic_add(newton.du[slots[a]], u_bars[a, :])
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
        i_r = rows.row_start[()] + i_n
        linear.b_rhs[i_r] = value


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
        if is_free_node(mass, grid.nodes_is_fixed[i_n]):
            lam = linear.solution[rows.row_start[()] + i_n]
            v = grid.nodes_vel[i_n]
            acc_vel_bar = qd.Vector.zero(gs.qd_float, 3)
            if qd.static(len(force_fields.force_fields) > 0):
                i_b, block = unpack_block(grid.blocks_key[i_n // BLOCK_CELLS])
                x_n = node_position(block * 4 + local_offset(i_n % BLOCK_CELLS), mpm_info.dx[()])
                v_free_bar = free_velocity_adjoint(mass, dt, lam)
                for i_ff in qd.static(range(len(force_fields.force_fields))):
                    acc_vel_bar += force_fields.force_fields[i_ff].get_acc_grad_vel(x_n, v, t, -1, v_free_bar)
            momentum_bar, mass_bar = node_backward(
                mass, lam, newton.u[i_n], v, grid.nodes_vel_free[i_n], acc_vel_bar, dt
            )
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
        x_p = particles.pos[f, i_p, i_b]
        slots, fx, w, grad_w = stencil_weights(i_b, cell, x_p, grid, mpm_info)
        u_nodes = gather_stencil(slots, newton.u)
        lam_nodes = gather_stencil(rows.row_start[()] + slots, linear.solution)
        F, F_trial, plastic, params, U, sigma, V, psi, g, A = particle_elastic_state(
            f, i_p, i_b, u_nodes, grad_w, particles, particles_info, materials, mpm_info
        )
        x_bar, F_bar, Jp_bar, actu_bar = elastic_pullback(
            u_nodes,
            lam_nodes,
            grad_w,
            fx,
            mpm_info.inv_dx[()],
            F,
            F_trial,
            U,
            sigma,
            V,
            g,
            A,
            plastic,
            params,
            particles.actu[f, i_p, i_b],
            particles_info.muscle_direction[i_p],
            elastic_scale(mpm_info.particle_volume[()], dt),
            mpm_info.snow_hardening[()],
        )
        x_bar, v_bar, C_bar = p2g_pullback(
            w,
            grad_w,
            cell,
            mpm_info.dx[()],
            x_p,
            particles.vel[f, i_p, i_b],
            gravity[i_b],
            dt,
            particles.C[f, i_p, i_b],
            particle_mass(particles_info.mass[i_p], mpm_info.mass_scale[()]),
            gather_stencil(slots, newton.nodes_momentum_bar),
            gather_stencil_scalars(slots, newton.nodes_mass_bar),
            x_bar,
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
    the substep carries over unchanged, to the start frame."""
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
