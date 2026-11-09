"""Adjoint of the contact rows between the material point method (MPM) and Rigid in the joint Newton solve."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.solvers.mpm_adjoint_func import band_row_bw, polar_rotation_bw
from genesis.engine.solvers.mpm_data import (
    PYRAMID_ROWS,
    MpmGridScratch,
    MpmInfo,
    MpmNewtonScratch,
    MpmParticles,
    MpmParticlesInfo,
)
from genesis.engine.solvers.mpm_func import (
    cofactor,
    cofactor_jacobian,
    particle_node_sum,
    particle_of_sorted,
    pyramid_directions,
    q1_stencil,
    stencil_dots,
    stencil_node_slots,
)
from genesis.engine.solvers.newton_linear import LinearBlock, NewtonLinearScratch
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.newton_func import RigidNewtonScratch
from genesis.utils.array_class import DofsState, GeomsInfo, GeomsState, LinksInfo, LinksState, RigidInfo, SDFInfo

from .mpm_rigid_func import MpmRigidPairs, geom_normal

# ============================================== Layer 0: reverses ===============================================


@qd.func
def geom_sdf_grid_hessian(i_g: int, i_b: int, x: qd.types.vector(3), geoms_state: GeomsState, sdf_info: SDFInfo):
    """Gradient and Hessian, in the mesh frame, of the signed distance of `geom_sdf_grid` at a world point: those of the
    trilinear interpolant inside the grid, and over a clamped component outside those of the distance to the grid."""
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
        sdf_val = sdf_info.geoms_sdf_val[sdf.sdf_func_ravel_cell_idx(node, i_g, sdf_res, sdf_info)]
        axis_dw = (2 * offset - 1).cast(gs.qd_float) / cell_size
        grad += sdf_val * axis_dw * cofactor(w)
        hess += sdf_val * axis_dw.outer_product(axis_dw) * cofactor_jacobian(w)
    outside = (pos_sdf - q) * cell_size
    distance = outside.norm()
    if distance > 0.0:
        outside_dir = outside / distance
        is_outside = ((pos_sdf < 0.0) | (pos_sdf > upper)).cast(gs.qd_float)
        grad = qd.select(is_outside > 0.0, outside_dir, grad)
        hess *= (1.0 - is_outside).outer_product(1.0 - is_outside)
        hess += (
            is_outside.outer_product(is_outside)
            * (qd.Matrix.identity(gs.qd_float, 3) - outside_dir.outer_product(outside_dir))
            / distance
        )
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
    """Reverse of the signed distance `geom_sdf` and of the normal `geom_normal` of a geom at a world point."""
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
        quat_bar += gu.qd_inv_quat(quat_inv_bar)
    qd.atomic_add(geoms_state.pos.grad[i_g, i_b], pos_bar)
    qd.atomic_add(geoms_state.quat.grad[i_g, i_b], quat_bar)
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
    """Spatial displacement of a link from a dof vector (see func_link_spatial_disp)."""
    lin = qd.Vector.zero(gs.qd_float, 3)
    ang = qd.Vector.zero(gs.qd_float, 3)
    j_l = i_l
    while j_l >= 0:
        J_l = [j_l, i_b] if qd.static(config.rigid_batch_links_info) else j_l
        for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
            lin += dofs_state.cdof_vel[i_d, i_b] * src[i_d, i_b]
            ang += dofs_state.cdof_ang[i_d, i_b] * src[i_d, i_b]
        j_l = links_info.parent_idx[J_l]
    return lin, ang


# ============================================== Layer 1: pullback ===============================================


@qd.func(requires_top_level=True)
def func_rigid_contact_adjoint(
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
    """Pull the MPM-Rigid pairs back into the adjoints of frame f of the particles and of Rigid's state."""
    dt = mpm_info.substep_dt[()]
    qd.loop_config(name="mpm_rigid_contact_adjoint")
    for i_pair in range(pairs.n_pairs[()]):
        i_l = pairs.link_idx[i_pair]
        if i_l >= 0:
            i_p, i_b, cell = particle_of_sorted(pairs.sorted_idx[i_pair], grid, mpm_info)
            slots = stencil_node_slots(i_b, cell, grid)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            w, grad_w = q1_stencil(x_p * mpm_info.inv_dx[()] - cell.cast(gs.qd_float), mpm_info.inv_dx[()])
            rows_p = rows.row_start[()] + slots
            u_p = particle_node_sum(slots, w, newton.u)
            lam_p = particle_node_sum(rows_p, w, linear.solution)
            r = x_p - links_state.root_COM[i_l, i_b]
            u_lin, u_ang = link_chain_spatial(i_l, i_b, rigid_data.x, dofs_state, links_info, config)
            lam_lin, lam_ang = link_chain_spatial(i_l, i_b, rigid_data.solution, dofs_state, links_info, config)
            u_rel = u_p - (u_lin + u_ang.cross(r))
            lam_rel = lam_p - (lam_lin + lam_ang.cross(r))

            # The adjoints of the coefficients of the active rows, and of u_rel and lam_rel
            stiffness_bar = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
            target_bar = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
            direction_bar = qd.Matrix.zero(gs.qd_float, PYRAMID_ROWS, 3)
            u_rel_bar = qd.Vector.zero(gs.qd_float, 3)
            lam_rel_bar = qd.Vector.zero(gs.qd_float, 3)
            directions = pyramid_directions(pairs.normal[i_pair], pairs.friction[i_pair])
            for k in qd.static(range(PYRAMID_ROWS)):
                stiffness = pairs.stiffness[i_pair][k]
                direction = directions[k, :]
                gap = direction.dot(u_rel) - pairs.target[i_pair][k]
                if stiffness > 0.0 and gap < 0.0:
                    lam_gap = direction.dot(lam_rel)
                    stiffness_bar[k] = -gap * lam_gap
                    target_bar[k] = stiffness * lam_gap
                    direction_bar_k = -stiffness * (lam_gap * u_rel + gap * lam_rel)
                    for j in qd.static(range(3)):
                        direction_bar[k, j] = direction_bar_k[j]
                    u_rel_bar -= stiffness * lam_gap * direction
                    lam_rel_bar -= stiffness * gap * direction

            # u_rel and lam_rel through the weights, the arm and the chain
            weights_bar = stencil_dots(slots, newton.u, u_rel_bar) + stencil_dots(rows_p, linear.solution, lam_rel_bar)
            x_bar = grad_w.transpose() @ weights_bar - u_rel_bar.cross(u_ang) - lam_rel_bar.cross(lam_ang)
            com_bar = u_rel_bar.cross(u_ang) + lam_rel_bar.cross(lam_ang)
            r_cross_u_rel_bar = r.cross(u_rel_bar)
            r_cross_lam_rel_bar = r.cross(lam_rel_bar)
            j_l = i_l
            while j_l >= 0:
                J_l = [j_l, i_b] if qd.static(config.rigid_batch_links_info) else j_l
                for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                    x_d = rigid_data.x[i_d, i_b]
                    lam_d = rigid_data.solution[i_d, i_b]
                    qd.atomic_add(dofs_state.cdof_vel.grad[i_d, i_b], -(x_d * u_rel_bar + lam_d * lam_rel_bar))
                    qd.atomic_add(
                        dofs_state.cdof_ang.grad[i_d, i_b], -(x_d * r_cross_u_rel_bar + lam_d * r_cross_lam_rel_bar)
                    )
                j_l = links_info.parent_idx[J_l]

            # The row coefficients, replayed as MpmRigidContact.build_pair built them
            i_g = pairs.geom_idx[i_pair]
            phi = pairs.phi[i_pair]
            I_l = [i_l, i_b] if qd.static(config.rigid_batch_links_info) else i_l
            v_r = func_vel_at_point(i_l, i_b, x_p, links_state)
            v_rel = v_p - v_r
            normal = geom_normal(i_g, i_b, x_p, geoms_state, geoms_info, rigid_info, sdf_info, config)
            friction = qd.max(geoms_info.friction[i_g] * geoms_state.friction_ratio[i_g, i_b], 1e-2)
            friction_sq_reg = friction * friction / rigid_info.impratio[None]
            U, S, V = gu.qd_svd(particles.F[f, i_p, i_b])
            R = U @ V.transpose()
            n_rest = R.transpose() @ normal
            invweight_p = particles_info.contact_invweight[i_p]
            invweight = n_rest.dot(invweight_p @ n_rest) + links_info.invweight[I_l][0]
            diag_coef_scale = (1.0 + friction_sq_reg) * 2.0 * friction_sq_reg
            diag_coef = invweight * diag_coef_scale
            sol_params = geoms_info.sol_params[i_g]
            sol_params[0] = qd.max(sol_params[0], 2.0 * dt)
            tangent1, tangent2 = gu.qd_orthogonals(normal)

            phi_bar = gs.qd_float(0.0)
            n_bar = qd.Vector.zero(gs.qd_float, 3)
            v_r_bar = qd.Vector.zero(gs.qd_float, 3)
            v_rel_bar = qd.Vector.zero(gs.qd_float, 3)
            diag_coef_bar = gs.qd_float(0.0)
            tangent1_bar = qd.Vector.zero(gs.qd_float, 3)
            tangent2_bar = qd.Vector.zero(gs.qd_float, 3)
            for k in qd.static(range(PYRAMID_ROWS)):
                tangent = tangent1
                if qd.static(k >= 2):
                    tangent = tangent2
                sign = qd.static(2 * (k % 2) - 1)
                direction = normal + (sign * friction) * tangent
                direction_bar_k = direction_bar[k, :]
                # target = target_soft + dt direction . v_r
                direction_bar_k += dt * target_bar[k] * v_r
                v_r_bar += dt * target_bar[k] * direction
                phi_bar_k, vel_bar_k, diag_coef_bar_k = band_row_bw(
                    sol_params, phi, direction.dot(v_rel), diag_coef, dt, stiffness_bar[k], target_bar[k]
                )
                phi_bar += phi_bar_k
                diag_coef_bar += diag_coef_bar_k
                direction_bar_k += vel_bar_k * v_rel
                v_rel_bar += vel_bar_k * direction
                n_bar += direction_bar_k
                if qd.static(k >= 2):
                    tangent2_bar += (sign * friction) * direction_bar_k
                else:
                    tangent1_bar += (sign * friction) * direction_bar_k
            n_bar += gu.qd_orthogonals_bw(normal, tangent1_bar, tangent2_bar)

            # The inverse weight n^T R invweight_p R^T n
            invweight_bar = diag_coef_bar * diag_coef_scale
            n_bar += 2.0 * invweight_bar * (R @ (invweight_p @ n_rest))
            R_bar = (2.0 * invweight_bar) * normal.outer_product(invweight_p @ n_rest)
            F_bar = polar_rotation_bw(U, S, V, R_bar)

            # v_rel = v_p - v_r, v_r = cd_vel + cd_ang x (x_p - root_COM)
            v_p_bar = v_rel_bar
            v_r_bar -= v_rel_bar
            cd_ang = links_state.cd_ang[i_l, i_b]
            qd.atomic_add(links_state.cd_vel.grad[i_l, i_b], v_r_bar)
            qd.atomic_add(links_state.cd_ang.grad[i_l, i_b], r.cross(v_r_bar))
            x_bar += v_r_bar.cross(cd_ang)
            com_bar -= v_r_bar.cross(cd_ang)
            qd.atomic_add(links_state.root_COM.grad[i_l, i_b], com_bar)

            # The distance and the normal of the geom at the particle
            x_bar += geom_contact_bw(i_g, i_b, x_p, phi_bar, n_bar, geoms_state, geoms_info, rigid_info, sdf_info)

            # A particle has one pair at most, whose thread alone adds to its adjoint
            particles.pos.grad[f, i_p, i_b] = particles.pos.grad[f, i_p, i_b] + x_bar
            particles.vel.grad[f, i_p, i_b] = particles.vel.grad[f, i_p, i_b] + v_p_bar
            particles.F.grad[f, i_p, i_b] = particles.F.grad[f, i_p, i_b] + F_bar
