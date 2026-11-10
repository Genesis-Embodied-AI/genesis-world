"""Adjoint of the contact rows between the material point method (MPM) and Rigid in the joint Newton solve."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.mpm_adjoint_func import (
    band_row_backward,
    contact_invweight_backward,
    soft_row_backward,
    stencil_position_backward,
)
from genesis.engine.solvers.mpm_data import (
    PYRAMID_ROWS,
    MpmGridScratch,
    MpmInfo,
    MpmNewtonScratch,
    MpmParticles,
    MpmParticlesInfo,
)
from genesis.engine.solvers.mpm_func import (
    contact_friction,
    contact_invweight,
    gather_stencil,
    particle_of_sorted,
    pyramid_directions,
    stencil_interpolate,
    stencil_weights,
)
from genesis.engine.solvers.newton_linear import LinearBlock, NewtonLinearScratch
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.newton_func import RigidNewtonScratch, dof_motion
from genesis.utils.array_class import DofsState, GeomsInfo, GeomsState, LinksInfo, LinksState, RigidInfo, SDFInfo

from .mpm_rigid_func import (
    MpmRigidPairs,
    contact_arm,
    geom_contact_reverse,
    geom_normal,
    geom_sdf_grid,
    point_motion,
    rigid_row_coefficients,
)

# ================================================ Layer 0: reverses =================================================


@qd.func
def rigid_relative(
    w: qd.types.vector(8),
    values: qd.types.matrix(8, 3),
    lin: qd.types.vector(3),
    ang: qd.types.vector(3),
    r: qd.types.vector(3),
):
    """Value of a particle relative to the contact point of its link at arm r, from those of the nodes of its stencil
    and the spatial value (lin, ang) of the link: its relative displacement, or its relative adjoint solution."""
    return stencil_interpolate(w, values) - point_motion(lin, ang, r)


@qd.func
def pyramid_rows_backward(
    normal: qd.types.vector(3),
    friction: float,
    stiffness: qd.types.vector(PYRAMID_ROWS),
    target: qd.types.vector(PYRAMID_ROWS),
    u_rel: qd.types.vector(3),
    lam_rel: qd.types.vector(3),
):
    """Reverse of the gradient of the rows of a friction pyramid (see pyramid_terms) contracted with the adjoint
    solution, to the adjoints of their stiffness, target and direction, and of the relative displacement and of the
    adjoint solution along them (see soft_row_backward)."""
    directions = pyramid_directions(normal, friction)
    stiffness_bar = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
    target_bar = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
    direction_bar = qd.Matrix.zero(gs.qd_float, PYRAMID_ROWS, 3)
    u_rel_bar = qd.Vector.zero(gs.qd_float, 3)
    lam_rel_bar = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(PYRAMID_ROWS)):
        _, stiffness_bar_k, target_bar_k, direction_bar_k, u_rel_bar_k, lam_rel_bar_k = soft_row_backward(
            directions[k, :], stiffness[k], target[k], u_rel, lam_rel
        )
        stiffness_bar[k] = stiffness_bar_k
        target_bar[k] = target_bar_k
        for j in qd.static(range(3)):
            direction_bar[k, j] = direction_bar_k[j]
        u_rel_bar += u_rel_bar_k
        lam_rel_bar += lam_rel_bar_k
    return stiffness_bar, target_bar, direction_bar, u_rel_bar, lam_rel_bar


@qd.func
def rigid_relative_backward(
    grad_w: qd.types.matrix(8, 3),
    u_nodes: qd.types.matrix(8, 3),
    lam_nodes: qd.types.matrix(8, 3),
    u_ang: qd.types.vector(3),
    lam_ang: qd.types.vector(3),
    r: qd.types.vector(3),
    u_rel_bar: qd.types.vector(3),
    lam_rel_bar: qd.types.vector(3),
):
    """Reverse of rigid_relative through the weights of the stencil and the arm of the contact point, to the adjoints
    of the position of the particle and of the root center of mass (COM) of the link, with the arm crossed with the
    adjoints for the pullback onto the dofs of the chain (see dof_motion_backward)."""
    com_bar = u_rel_bar.cross(u_ang) + lam_rel_bar.cross(lam_ang)
    x_bar = stencil_position_backward(grad_w, u_nodes, lam_nodes, u_rel_bar, lam_rel_bar, -com_bar)
    return x_bar, com_bar, r.cross(u_rel_bar), r.cross(lam_rel_bar)


@qd.func
def dof_motion_backward(
    x_d: float,
    lam_d: float,
    u_rel_bar: qd.types.vector(3),
    lam_rel_bar: qd.types.vector(3),
    r_cross_u_rel_bar: qd.types.vector(3),
    r_cross_lam_rel_bar: qd.types.vector(3),
):
    """Reverse of the motion of the contact point by a dof of the chain of its link (see dof_motion), from the adjoints
    of the relative displacement and adjoint solution to those of the motion of the dof."""
    return -(x_d * u_rel_bar + lam_d * lam_rel_bar), -(x_d * r_cross_u_rel_bar + lam_d * r_cross_lam_rel_bar)


@qd.func
def rigid_pair_backward(
    normal: qd.types.vector(3),
    friction: float,
    diag_coef: float,
    diag_coef_scale: float,
    sol_params_row: qd.types.vector(7),
    phi: float,
    v_p: qd.types.vector(3),
    v_r: qd.types.vector(3),
    U: qd.types.matrix(3, 3),
    S: qd.types.matrix(3, 3),
    V: qd.types.matrix(3, 3),
    invweight_rest: qd.types.matrix(3, 3),
    dt: float,
    stiffness_bar: qd.types.vector(PYRAMID_ROWS),
    target_bar: qd.types.vector(PYRAMID_ROWS),
    direction_bar: qd.types.matrix(PYRAMID_ROWS, 3),
):
    """Reverse of the row coefficients and directions of a pair of a particle with a Rigid geom (see rigid_pair_rows and
    rigid_row_coefficients), to the adjoints of the gap, the normal, the velocity of the particle and of the contact
    point, and the deformation gradient of the particle through its inverse weight."""
    v_rel = v_p - v_r
    tangent1, tangent2 = gu.qd_orthogonals(normal)
    phi_bar = gs.qd_float(0.0)
    n_bar = qd.Vector.zero(gs.qd_float, 3)
    v_r_bar = qd.Vector.zero(gs.qd_float, 3)
    v_rel_bar = qd.Vector.zero(gs.qd_float, 3)
    diag_coef_bar = gs.qd_float(0.0)
    tangent1_bar = qd.Vector.zero(gs.qd_float, 3)
    tangent2_bar = qd.Vector.zero(gs.qd_float, 3)
    for k in range(PYRAMID_ROWS):
        is_second_tangent = k >= 2
        tangent = tangent1
        if is_second_tangent:
            tangent = tangent2
        sign = gs.qd_float(2 * (k % 2) - 1)
        direction = normal + (sign * friction) * tangent
        direction_bar_k = qd.Vector([direction_bar[k, j] for j in qd.static(range(3))], dt=gs.qd_float)
        # target = target_soft + dt direction . v_r
        direction_bar_k += dt * target_bar[k] * v_r
        v_r_bar += dt * target_bar[k] * direction
        phi_bar_k, vel_bar_k, diag_coef_bar_k = band_row_backward(
            sol_params_row, phi, direction.dot(v_rel), diag_coef, dt, stiffness_bar[k], target_bar[k]
        )
        phi_bar += phi_bar_k
        diag_coef_bar += diag_coef_bar_k
        direction_bar_k += vel_bar_k * v_rel
        v_rel_bar += vel_bar_k * direction
        n_bar += direction_bar_k
        if is_second_tangent:
            tangent2_bar += (sign * friction) * direction_bar_k
        else:
            tangent1_bar += (sign * friction) * direction_bar_k
    n_bar += gu.qd_orthogonals_backward(normal, tangent1_bar, tangent2_bar)
    # The inverse weight n^T R invweight_rest R^T n of the particle, that of the link being a constant
    n_bar_invweight, F_bar = contact_invweight_backward(
        U, S, V, invweight_rest, normal, diag_coef_bar * diag_coef_scale
    )
    return phi_bar, n_bar + n_bar_invweight, v_rel_bar, v_r_bar - v_rel_bar, F_bar


@qd.func
def point_velocity_backward(
    v_r_bar: qd.types.vector(3),
    r: qd.types.vector(3),
    cd_ang: qd.types.vector(3),
    x_bar: qd.types.vector(3),
    com_bar: qd.types.vector(3),
):
    """Reverse of the velocity v_r = cd_vel + cd_ang x r of a contact point (see func_vel_at_point), adding to the
    adjoints of the position of the particle and of the root center of mass (COM) of the link, with the adjoint of the
    angular velocity of the link, that of its linear velocity being v_r_bar."""
    return r.cross(v_r_bar), x_bar + v_r_bar.cross(cd_ang), com_bar - v_r_bar.cross(cd_ang)


# ============================================== Layer 1: pullback ===============================================


@qd.func
def link_chain_spatial(
    i_l: int,
    i_b: int,
    src: qd.template(),
    dofs_state: DofsState,
    links_info: LinksInfo,
    config: qd.template(),
):
    """Spatial displacement of a link from a dof vector (see func_link_spatial_disp)."""
    lin = qd.Vector.zero(gs.qd_float, 3)
    ang = qd.Vector.zero(gs.qd_float, 3)
    j_l = i_l
    while j_l >= 0:
        J_l = [j_l, i_b] if qd.static(config.rigid_batch_links_info) else j_l
        for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
            lin_d, ang_d = dof_motion(dofs_state.cdof_vel[i_d, i_b], dofs_state.cdof_ang[i_d, i_b], src[i_d, i_b])
            lin += lin_d
            ang += ang_d
        j_l = links_info.parent_idx[J_l]
    return lin, ang


@qd.func
def geom_contact_backward(
    i_g: int,
    i_b: int,
    x: qd.types.vector(3),
    phi_bar: float,
    n_bar: qd.types.vector(3),
    x_bar: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
):
    """Add to the adjoint of a world point the reverse of the signed distance `geom_sdf` and of the normal
    `geom_normal` of a geom there, and their adjoints of the pose of the geom to its state."""
    geom_type = geoms_info.type[i_g]
    grad_mesh = qd.Vector.zero(gs.qd_float, 3)
    hess_mesh = qd.Matrix.zero(gs.qd_float, 3, 3)
    if geom_type == gs.GEOM_TYPE.PLANE:
        geom_data = geoms_info.data[i_g]
        grad_mesh = qd.Vector([geom_data[0], geom_data[1], geom_data[2]], dt=gs.qd_float)
    elif geom_type != gs.GEOM_TYPE.SPHERE:
        _, grad_mesh, hess_mesh = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info, is_hessian=True)
    x_bar, pos_bar, quat_bar = geom_contact_reverse(
        geom_type == gs.GEOM_TYPE.SPHERE,
        x,
        geoms_state.pos[i_g, i_b],
        geoms_state.quat[i_g, i_b],
        grad_mesh,
        hess_mesh,
        phi_bar,
        n_bar,
        rigid_info.EPS[None],
        x_bar,
    )
    qd.atomic_add(geoms_state.pos.grad[i_g, i_b], pos_bar)
    qd.atomic_add(geoms_state.quat.grad[i_g, i_b], quat_bar)
    return x_bar


@qd.kernel
def kernel_rigid_contact_adjoint(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: NewtonLinearScratch,
    rows: LinearBlock,
    pairs: MpmRigidPairs,
    rigid_data: RigidNewtonScratch,
    geoms_state: GeomsState,
    links_state: LinksState,
    dofs_state: DofsState,
    particles_info: MpmParticlesInfo,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    mpm_info: MpmInfo,
    config: qd.template(),
):
    """Pull the MPM-Rigid pairs back into the adjoints of the particles and of Rigid's state."""
    dt = mpm_info.substep_dt[()]
    qd.loop_config(name="mpm_rigid_contact_adjoint")
    for i_pair in range(pairs.n_pairs[()]):
        i_l = pairs.link_idx[i_pair]
        if i_l >= 0:
            i_p, i_b, cell = particle_of_sorted(pairs.sorted_idx[i_pair], grid, mpm_info)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            slots, fx, w, grad_w = stencil_weights(i_b, cell, x_p, grid, mpm_info)
            u_nodes = gather_stencil(slots, newton.u)
            lam_nodes = gather_stencil(rows.row_start[()] + slots, linear.solution)
            r = contact_arm(x_p, links_state.root_COM[i_l, i_b])
            u_lin, u_ang = link_chain_spatial(i_l, i_b, rigid_data.x, dofs_state, links_info, config)
            lam_lin, lam_ang = link_chain_spatial(i_l, i_b, rigid_data.solution, dofs_state, links_info, config)

            # The adjoints of the coefficients of the active rows, and of u_rel and lam_rel
            stiffness_bar, target_bar, direction_bar, u_rel_bar, lam_rel_bar = pyramid_rows_backward(
                pairs.normal[i_pair],
                pairs.friction[i_pair],
                pairs.stiffness[i_pair],
                pairs.target[i_pair],
                rigid_relative(w, u_nodes, u_lin, u_ang, r),
                rigid_relative(w, lam_nodes, lam_lin, lam_ang, r),
            )

            # u_rel and lam_rel through the weights, the arm and the chain
            x_bar, com_bar, r_cross_u_rel_bar, r_cross_lam_rel_bar = rigid_relative_backward(
                grad_w, u_nodes, lam_nodes, u_ang, lam_ang, r, u_rel_bar, lam_rel_bar
            )
            j_l = i_l
            while j_l >= 0:
                J_l = [j_l, i_b] if qd.static(config.rigid_batch_links_info) else j_l
                for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                    cdof_vel_bar, cdof_ang_bar = dof_motion_backward(
                        rigid_data.x[i_d, i_b],
                        rigid_data.solution[i_d, i_b],
                        u_rel_bar,
                        lam_rel_bar,
                        r_cross_u_rel_bar,
                        r_cross_lam_rel_bar,
                    )
                    qd.atomic_add(dofs_state.cdof_vel.grad[i_d, i_b], cdof_vel_bar)
                    qd.atomic_add(dofs_state.cdof_ang.grad[i_d, i_b], cdof_ang_bar)
                j_l = links_info.parent_idx[J_l]

            # The row coefficients, replayed as MpmRigidContact.build_pair built them
            i_g = pairs.geom_idx[i_pair]
            I_l = [i_l, i_b] if qd.static(config.rigid_batch_links_info) else i_l
            v_r = func_vel_at_point(i_l, i_b, x_p, links_state)
            normal = geom_normal(i_g, i_b, x_p, geoms_state, geoms_info, rigid_info, sdf_info, config)
            friction = contact_friction(
                geoms_info.friction[i_g], geoms_state.friction_ratio[i_g, i_b], mpm_info.contact_min_friction[()]
            )
            U, S, V = gu.qd_svd(particles.F[f, i_p, i_b])
            invweight_rest = particles_info.contact_invweight[i_p]
            diag_coef, diag_coef_scale, sol_params_row = rigid_row_coefficients(
                friction,
                rigid_info.impratio[None],
                contact_invweight(U, V, invweight_rest, normal),
                links_info.invweight[I_l][0],
                geoms_info.sol_params[i_g],
                dt,
            )
            phi_bar, n_bar, v_p_bar, v_r_bar, F_bar = rigid_pair_backward(
                normal,
                friction,
                diag_coef,
                diag_coef_scale,
                sol_params_row,
                pairs.phi[i_pair],
                v_p,
                v_r,
                U,
                S,
                V,
                invweight_rest,
                dt,
                stiffness_bar,
                target_bar,
                direction_bar,
            )

            # v_r = cd_vel + cd_ang x (x_p - root_COM)
            cd_ang_bar, x_bar, com_bar = point_velocity_backward(
                v_r_bar, r, links_state.cd_ang[i_l, i_b], x_bar, com_bar
            )
            qd.atomic_add(links_state.cd_vel.grad[i_l, i_b], v_r_bar)
            qd.atomic_add(links_state.cd_ang.grad[i_l, i_b], cd_ang_bar)
            qd.atomic_add(links_state.root_COM.grad[i_l, i_b], com_bar)

            # The distance and the normal of the geom at the particle
            x_bar = geom_contact_backward(
                i_g, i_b, x_p, phi_bar, n_bar, x_bar, geoms_state, geoms_info, rigid_info, sdf_info
            )

            # A particle has one pair at most, whose thread alone adds to its adjoint
            particles.pos.grad[f, i_p, i_b] = particles.pos.grad[f, i_p, i_b] + x_bar
            particles.vel.grad[f, i_p, i_b] = particles.vel.grad[f, i_p, i_b] + v_p_bar
            particles.F.grad[f, i_p, i_b] = particles.F.grad[f, i_p, i_b] + F_bar
