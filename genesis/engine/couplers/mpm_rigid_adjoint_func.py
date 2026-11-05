"""Adjoint of the contact rows between the material point method (MPM) and Rigid in the joint Newton solve.

The rows (see mpm_rigid_func) reverse by hand in the implicit-function pullback, as Rigid reverses the assembly of its
own contact rows.

The joint solve minimizes E(y, theta) over y = (u, x). Its backward pass solves H lambda = y_bar and pulls back
theta_bar = -d/dtheta [lambda^T grad_y E] at the solution, with y and lambda held. The contact rows' share of lambda^T
grad_y E is Phi = sum_s g_s . delta_s over the particles with Rigid rows, g_s = sum_k D_k c_k J_k over the active rows
of particle s, c_k = J_k . Delta_s - t_k, with Delta_s = sum_a w_a u_a - dr(x) the displacement of the particle relative
to its contact point and delta_s the same map of lambda. Phi depends on theta through the weights and the arm of the
particle, the cdofs of its link, and the row coefficients J, D and t, which `func_build_rigid_rows` derives from the
particle state, the pose of the geom and the velocity of the link.

The friction, the impedance parameters and the inverse weight matrix of the particle are model constants (see
func_contact_invweight).
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.solvers.mpm_adjoint_func import band_row_bw, polar_rotation_bw
from genesis.engine.solvers.mpm_data import (
    CONTACT_ROWS,
    PYRAMID_ROWS,
    WALL_ROWS,
    MpmGridScratch,
    MpmInfo,
    MpmLinearScratch,
    MpmNewtonScratch,
    MpmParticles,
    MpmParticlesInfo,
)
from genesis.engine.solvers.mpm_func import particle_of_sorted, q1_stencil, stencil_node_slots
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.newton_func import RigidNewtonScratch
from genesis.utils.array_class import DofsState, GeomsInfo, GeomsState, LinksInfo, LinksState, RigidInfo, SDFInfo

from .mpm_rigid_func import geom_normal

# ============================================== Layer 0: reverses ===============================================


@qd.func
def geom_sdf_grid_hessian(i_g: int, i_b: int, x: qd.types.vector(3), geoms_state: GeomsState, sdf_info: SDFInfo):
    """Gradient and Hessian, in the mesh frame, of the signed distance of `geom_sdf_grid` at a world point.

    Inside the grid, they are those of the trilinear interpolant, whose pure second derivatives vanish. Outside, a
    clamped component follows the distance |o| to the grid, of Hessian (I - o o^T) / |o| over the clamped components."""
    pos_mesh = gu.qd_inv_transform_by_trans_quat(x, geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b])
    pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
    sdf_res = sdf_info.geoms_info.sdf_res[i_g]
    cell_size = sdf_info.geoms_info.sdf_cell_size[i_g]
    upper = (sdf_res - 1).cast(gs.qd_float)
    q = qd.min(qd.max(pos_sdf, 0.0), upper)
    base = qd.max(qd.min(qd.floor(q, gs.qd_int), sdf_res - 2), 0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    hess = qd.Matrix.zero(gs.qd_float, 3, 3)
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        node = base + offset
        w = 1.0 - qd.abs(q - node.cast(gs.qd_float))
        sign = (2 * offset - 1).cast(gs.qd_float)
        sdf_val = sdf_info.geoms_sdf_val[sdf.sdf_func_ravel_cell_idx(node, i_g, sdf_res, sdf_info)]
        grad[0] += sign[0] * w[1] * w[2] * sdf_val / cell_size[0]
        grad[1] += w[0] * sign[1] * w[2] * sdf_val / cell_size[1]
        grad[2] += w[0] * w[1] * sign[2] * sdf_val / cell_size[2]
        hess[0, 1] += sign[0] * sign[1] * w[2] * sdf_val / (cell_size[0] * cell_size[1])
        hess[0, 2] += sign[0] * w[1] * sign[2] * sdf_val / (cell_size[0] * cell_size[2])
        hess[1, 2] += w[0] * sign[1] * sign[2] * sdf_val / (cell_size[1] * cell_size[2])
    hess[1, 0] = hess[0, 1]
    hess[2, 0] = hess[0, 2]
    hess[2, 1] = hess[1, 2]
    outside = qd.Vector([(pos_sdf[d] - q[d]) * cell_size[d] for d in qd.static(range(3))], dt=gs.qd_float)
    distance = outside.norm()
    if distance > 0.0:
        o = outside / distance
        eye = qd.Matrix.identity(gs.qd_float, 3)
        is_outside = qd.Vector.zero(gs.qd_int, 3)
        for d in qd.static(range(3)):
            if pos_sdf[d] < 0.0 or pos_sdf[d] > upper[d]:
                is_outside[d] = 1
                grad[d] = o[d]
                for e in qd.static(range(3)):
                    hess[d, e] = 0.0
                    hess[e, d] = 0.0
        for d, e in qd.static(qd.ndrange(3, 3)):
            if is_outside[d] != 0 and is_outside[e] != 0:
                hess[d, e] = (eye[d, e] - o[d] * o[e]) / distance
    return grad, hess


@qd.func
def geom_contact_bw(
    i_g: int,
    i_b: int,
    x: qd.types.vector(3),
    phi_bar: float,
    n_bar: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
):
    """Reverse of the signed distance `geom_sdf` and of the normal `geom_normal` of a geom at a world point.

    Returns the adjoint of x from phi_bar and n_bar, and adds the adjoints of the pose of geom i_g to its gradients."""
    EPS = rigid_info.EPS[None]
    geom_pos = geoms_state.pos[i_g, i_b]
    geom_quat = geoms_state.quat[i_g, i_b]
    x_bar = qd.Vector.zero(gs.qd_float, 3)
    pos_bar = qd.Vector.zero(gs.qd_float, 3)
    quat_bar = qd.Vector.zero(gs.qd_float, 4)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE:
        e = x - geom_pos
        e_norm = qd.max(e.norm(), EPS)
        n = e / e_norm
        x_bar = phi_bar * n + (n_bar - n.dot(n_bar) * n) / e_norm
        pos_bar = -x_bar
    else:
        # The mesh-frame gradient of the distance and its Hessian, and the world gradient that gives the normal
        grad_mesh = qd.Vector.zero(gs.qd_float, 3)
        hess_mesh = qd.Matrix.zero(gs.qd_float, 3, 3)
        if geom_type == gs.GEOM_TYPE.PLANE:
            geom_data = geoms_info.data[i_g]
            grad_mesh = qd.Vector([geom_data[0], geom_data[1], geom_data[2]], dt=gs.qd_float)
        else:
            grad_mesh, hess_mesh = geom_sdf_grid_hessian(i_g, i_b, x, geoms_state, sdf_info)
        grad_world = gu.qd_transform_by_quat(grad_mesh, geom_quat)
        grad_world_norm = qd.max(grad_world.norm(), EPS)
        n = grad_world / grad_world_norm
        grad_world_bar = (n_bar - n.dot(n_bar) * n) / grad_world_norm
        quat_bar += gu.qd_transform_by_quat_grad_quat(grad_mesh, geom_quat, grad_world_bar)
        quat_inv = gu.qd_inv_quat(geom_quat)
        grad_mesh_bar = gu.qd_transform_by_quat(grad_world_bar, quat_inv)
        # pos_mesh = R^T (x - pos)
        pos_mesh_bar = phi_bar * grad_mesh + hess_mesh @ grad_mesh_bar
        x_bar = gu.qd_transform_by_quat(pos_mesh_bar, geom_quat)
        pos_bar = -x_bar
        quat_inv_bar = gu.qd_transform_by_quat_grad_quat(x - geom_pos, quat_inv, pos_mesh_bar)
        quat_bar += qd.Vector([quat_inv_bar[0], -quat_inv_bar[1], -quat_inv_bar[2], -quat_inv_bar[3]], dt=gs.qd_float)
    geoms_state.pos.grad[i_g, i_b] += pos_bar
    geoms_state.quat.grad[i_g, i_b] += quat_bar
    return x_bar


@qd.func
def link_chain_spatial(
    i_l: int,
    i_b: int,
    src: qd.template(),
    dofs_state: DofsState,
    links_info: LinksInfo,
    config: qd.template(),
):
    """Spatial displacement of a link from a dof vector (see func_link_spatial_disp).

    Returns the linear and angular parts of the sum of cdof src_d over the dofs of the chain of link i_l."""
    lin = qd.Vector.zero(gs.qd_float, 3)
    ang = qd.Vector.zero(gs.qd_float, 3)
    i_a = i_l
    while i_a >= 0:
        I_a = [i_a, i_b] if qd.static(config.rigid_batch_links_info) else i_a
        for i_d in range(links_info.dof_start[I_a], links_info.dof_end[I_a]):
            lin += dofs_state.cdof_vel[i_d, i_b] * src[i_d, i_b]
            ang += dofs_state.cdof_ang[i_d, i_b] * src[i_d, i_b]
        i_a = links_info.parent_idx[I_a]
    return lin, ang


# ============================================== Layer 1: pullback ===============================================


@qd.func(requires_top_level=True)
def func_rigid_contact_adjoint(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    linear: MpmLinearScratch,
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
    """Pull the Rigid contact rows of every particle back into the adjoints of the particles and of Rigid's state.

    The adjoints add to frame f of the particles, from the joint solution (u*, x*) and the adjoint lambda =
    (`linear.solution`, `rigid_data.sol`). With a_k = J_k . delta over the active rows, Phi has D_k_bar = -c_k a_k,
    t_k_bar = D_k a_k, J_k_bar = -D_k (a_k Delta + c_k delta), Delta_bar = -sum D_k a_k J_k and delta_bar = -sum D_k c_k
    J_k. Delta and delta move with the weights, the arm r = x_p - root_COM of the particle and the cdofs of its chain.
    The rows move with the particle state, with the geom pose through its distance and normal, and with the link
    velocity at the particle."""
    dt = mpm_info.substep_dt[()]
    qd.loop_config(name="mpm_rigid_contact_adjoint")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
            slots = stencil_node_slots(env, cell, grid)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            w, grad_w = q1_stencil(x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
            dxp = qd.Vector.zero(gs.qd_float, 3)
            lam_p = qd.Vector.zero(gs.qd_float, 3)
            for a in qd.static(range(8)):
                u_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * slots[a]
                lam_a = qd.Vector([linear.solution[row + d] for d in qd.static(range(3))], dt=gs.qd_float)
                dxp += w[a] * u_a
                lam_p += w[a] * lam_a
            r = x_p - links_state.root_COM[i_l, i_b]
            V_lin, V_ang = link_chain_spatial(i_l, i_b, rigid_data.x, dofs_state, links_info, config)
            L_lin, L_ang = link_chain_spatial(i_l, i_b, rigid_data.sol, dofs_state, links_info, config)
            Delta = dxp - (V_lin + V_ang.cross(r))
            delta = lam_p - (L_lin + L_ang.cross(r))

            # The adjoints of the coefficients of the active rows, and of Delta and delta
            D_bar = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
            t_bar = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
            J_bar = qd.Matrix.zero(gs.qd_float, PYRAMID_ROWS, 3)
            Delta_bar = qd.Vector.zero(gs.qd_float, 3)
            delta_bar = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(PYRAMID_ROWS)):
                k = CONTACT_ROWS * s + WALL_ROWS + i
                D = newton.contact_D[k]
                J = newton.contact_dir[k]
                c = J.dot(Delta) - newton.contact_t[k]
                if D > 0.0 and c < 0.0:
                    a_k = J.dot(delta)
                    D_bar[i] = -c * a_k
                    t_bar[i] = D * a_k
                    J_bar_i = -D * (a_k * Delta + c * delta)
                    for d in qd.static(range(3)):
                        J_bar[i, d] = J_bar_i[d]
                    Delta_bar -= D * a_k * J
                    delta_bar -= D * c * J

            # Delta and delta through the weights, the arm and the chain
            x_bar = -Delta_bar.cross(V_ang) - delta_bar.cross(L_ang)
            for a in qd.static(range(8)):
                u_a = qd.Vector([newton.u[3 * slots[a] + d] for d in qd.static(range(3))], dt=gs.qd_float)
                row = 3 * slots[a]
                lam_a = qd.Vector([linear.solution[row + d] for d in qd.static(range(3))], dt=gs.qd_float)
                grad_w_a = qd.Vector([grad_w[a, d] for d in qd.static(range(3))], dt=gs.qd_float)
                x_bar += (Delta_bar.dot(u_a) + delta_bar.dot(lam_a)) * grad_w_a
            com_bar = Delta_bar.cross(V_ang) + delta_bar.cross(L_ang)
            r_cross_Delta_bar = r.cross(Delta_bar)
            r_cross_delta_bar = r.cross(delta_bar)
            i_a = i_l
            while i_a >= 0:
                I_a = [i_a, i_b] if qd.static(config.rigid_batch_links_info) else i_a
                for i_d in range(links_info.dof_start[I_a], links_info.dof_end[I_a]):
                    x_d = rigid_data.x[i_d, i_b]
                    lam_d = rigid_data.sol[i_d, i_b]
                    dofs_state.cdof_vel.grad[i_d, i_b] += -(x_d * Delta_bar + lam_d * delta_bar)
                    dofs_state.cdof_ang.grad[i_d, i_b] += -(x_d * r_cross_Delta_bar + lam_d * r_cross_delta_bar)
                i_a = links_info.parent_idx[I_a]

            # The row coefficients, replayed as func_build_rigid_rows built them
            i_g = newton.query_idx[s]
            phi = newton.query_phi[s]
            I_l = [i_l, i_b] if qd.static(config.rigid_batch_links_info) else i_l
            v_r = func_vel_at_point(i_l, i_b, x_p, links_state)
            v_rel = v_p - v_r
            normal = geom_normal(i_g, i_b, x_p, geoms_state, geoms_info, rigid_info, sdf_info, config)
            friction = qd.max(geoms_info.friction[i_g] * geoms_state.friction_ratio[i_g, i_b], 1e-2)
            friction_sq_reg = friction * friction / rigid_info.impratio[None]
            U, S, V = gu.qd_svd(particles.F[f, i_p, i_b])
            R = U @ V.transpose()
            n_rest = R.transpose() @ normal
            W = particles_info.contact_invweight[i_p]
            invweight = n_rest.dot(W @ n_rest) + links_info.invweight[I_l][0]
            diag_coef_scale = (1.0 + friction_sq_reg) * 2.0 * friction_sq_reg
            diag_coef = invweight * diag_coef_scale
            sol_params = geoms_info.sol_params[i_g]
            sol_params[0] = qd.max(sol_params[0], 2.0 * dt)
            d1, d2 = gu.qd_orthogonals(normal)

            phi_bar = gs.qd_float(0.0)
            n_bar = qd.Vector.zero(gs.qd_float, 3)
            v_r_bar = qd.Vector.zero(gs.qd_float, 3)
            v_rel_bar = qd.Vector.zero(gs.qd_float, 3)
            diag_coef_bar = gs.qd_float(0.0)
            g_d1 = qd.Vector.zero(gs.qd_float, 3)
            g_d2 = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(PYRAMID_ROWS)):
                d = d1
                if qd.static(i >= 2):
                    d = d2
                sign = qd.static(2 * (i % 2) - 1)
                J = normal + (sign * friction) * d
                J_bar_i = qd.Vector([J_bar[i, e] for e in qd.static(range(3))], dt=gs.qd_float)
                # t = t_soft + dt J . v_r
                J_bar_i += dt * t_bar[i] * v_r
                v_r_bar += dt * t_bar[i] * J
                phi_bar_i, vel_bar_i, diag_coef_bar_i = band_row_bw(
                    sol_params, phi, J.dot(v_rel), diag_coef, dt, D_bar[i], t_bar[i]
                )
                phi_bar += phi_bar_i
                diag_coef_bar += diag_coef_bar_i
                J_bar_i += vel_bar_i * v_rel
                v_rel_bar += vel_bar_i * J
                n_bar += J_bar_i
                if qd.static(i >= 2):
                    g_d2 += (sign * friction) * J_bar_i
                else:
                    g_d1 += (sign * friction) * J_bar_i
            n_bar += gu.qd_orthogonals_bw(normal, g_d1, g_d2)

            # The inverse weight n^T R W R^T n
            invweight_bar = diag_coef_bar * diag_coef_scale
            n_bar += 2.0 * invweight_bar * (R @ (W @ n_rest))
            R_bar = (2.0 * invweight_bar) * normal.outer_product(W @ n_rest)
            F_bar = polar_rotation_bw(U, S, V, R_bar)

            # v_rel = v_p - v_r, v_r = cd_vel + cd_ang x (x_p - root_COM)
            v_p_bar = v_rel_bar
            v_r_bar -= v_rel_bar
            cd_ang = links_state.cd_ang[i_l, i_b]
            links_state.cd_vel.grad[i_l, i_b] += v_r_bar
            links_state.cd_ang.grad[i_l, i_b] += r.cross(v_r_bar)
            x_bar += v_r_bar.cross(cd_ang)
            com_bar -= v_r_bar.cross(cd_ang)
            links_state.root_COM.grad[i_l, i_b] += com_bar

            # The distance and the normal of the geom at the particle
            x_bar += geom_contact_bw(i_g, i_b, x_p, phi_bar, n_bar, geoms_state, geoms_info, rigid_info, sdf_info)

            particles.pos.grad[f, i_p, i_b] += x_bar
            particles.vel.grad[f, i_p, i_b] += v_p_bar
            particles.F.grad[f, i_p, i_b] += F_bar
