from __future__ import annotations

import quadrants as qd

from .contact_constitution import ContactConstitution
from .contact_function.cipc_simplex_scatter import (
    cipc_pair_area_weight,
    cipc_scatter_doublets,
    cipc_scatter_triplets_upper,
    cipc_scatter_triplets_upper_rank1,
)
from .contact_function.codim_thickness import (
    pair_thickness_ee,
    pair_thickness_ph,
    pair_thickness_pt,
)
from .contact_function.contact_table_query import (
    ct_enabled,
    ct_enabled_ee,
    ct_enabled_pt,
    ct_query,
    ct_query_ee,
    ct_query_pt,
)
from .contact_function.distance_flag import (
    _get_offset,
    _popcount4,
    ee_distance_flag,
    ee_flagged_distance2,
    flag_active_offsets,
    flag_active_offsets_for_ee,
    pt_distance_flag,
    pt_flagged_distance2,
)
from .contact_function.friction_contact_function import (
    gipc_friction_energy_ee,
    gipc_friction_energy_pe,
    gipc_friction_energy_pp,
    gipc_friction_energy_pt,
    gipc_friction_grad_hess_ee,
    gipc_friction_grad_hess_pe,
    gipc_friction_grad_hess_pp,
    gipc_friction_grad_hess_pt,
)
from .contact_function.gipc_barrier import (
    gipc_barrier_energy,
    gipc_barrier_energy_mollified,
    gipc_barrier_first_derivative,
    gipc_barrier_second_derivative,
    gipc_normal_force,
)
from .contact_function.gipc_barrier_gradient_hessian import (
    gipc_barrier_grad_hess_ee_mollified,
    gipc_barrier_grad_hess_ee_rank1,
    gipc_barrier_grad_hess_pe,
    gipc_barrier_grad_hess_pe_mollified,
    gipc_barrier_grad_hess_pp,
    gipc_barrier_grad_hess_pp_mollified,
    gipc_barrier_grad_hess_pt,
)
from .contact_function.gipc_distance import gipc_d_EE
from .contact_function.halfplane_contact import (
    halfplane_barrier_gradient,
    halfplane_barrier_hessian,
    halfplane_friction_energy,
    halfplane_friction_grad_hess,
    halfplane_signed_distance,
)
from .contact_function.pair_d_hat import pair_d_hat_ee, pair_d_hat_ph, pair_d_hat_pt


@qd.func
def pair_vert_kappa_scale(contact: qd.template(), vertex: qd.template(), global_ids: qd.template(), count):
    scale = contact.contact_kappa_scale[()]
    mode = contact.adaptive_kappa_mode[()]
    if mode == 2:
        for index in qd.static(range(4)):
            if index < count:
                scale = qd.max(scale, contact.vertex_kappa_scale[global_ids[index]])
    elif mode == 3:
        for index in qd.static(range(4)):
            if index < count:
                scale = qd.max(scale, contact.body_kappa_scale[vertex.body_id[global_ids[index]]])
    return scale


@qd.func
def report_pair_min_gap(
    contact: qd.template(),
    vertex: qd.template(),
    gap,
    global_ids: qd.template(),
    count,
):
    qd.atomic_min(contact.min_gap_ratio[()], gap)
    qd.atomic_min(contact.iter_min_gap_ratio[()], gap)
    mode = contact.adaptive_kappa_mode[()]
    for index in qd.static(range(4)):
        if index < count:
            vertex_id = global_ids[index]
            if mode == 2:
                qd.atomic_min(contact.vertex_min_gap[vertex_id], gap)
            elif mode == 3:
                body_id = vertex.body_id[vertex_id]
                qd.atomic_min(contact.body_min_gap[body_id], gap)
                qd.atomic_min(contact.iter_body_min_gap[body_id], gap)


@qd.func
def _pt_enabled(contact: qd.template(), ids: qd.template()):
    return ct_enabled_pt(
        contact.enable_table,
        contact.n_contact_elements[()],
        contact.vert_contact_element_ids[ids[0]],
        contact.vert_contact_element_ids[ids[1]],
        contact.vert_contact_element_ids[ids[2]],
        contact.vert_contact_element_ids[ids[3]],
    )


@qd.func
def _ee_enabled(contact: qd.template(), ids: qd.template()):
    return ct_enabled_ee(
        contact.enable_ee_table,
        contact.n_contact_elements[()],
        contact.vert_contact_element_ids[ids[0]],
        contact.vert_contact_element_ids[ids[1]],
        contact.vert_contact_element_ids[ids[2]],
        contact.vert_contact_element_ids[ids[3]],
    )


@qd.func
def _ph_enabled(contact: qd.template(), vertex_id):
    return ct_enabled(
        contact.enable_table,
        contact.n_contact_elements[()],
        contact.vert_contact_element_ids[vertex_id],
        0,
    )


@qd.func
def _embed_pe(
    positions: qd.template(),
    offsets: qd.template(),
    d_hat,
    xi,
    kappa,
    gradient: qd.template(),
    hessian: qd.template(),
):
    gradient_pe = qd.Vector.zero(qd.f64, 9)
    hessian_pe = qd.Vector.zero(qd.f64, 81)
    gipc_barrier_grad_hess_pe(
        positions[offsets[0], 0],
        positions[offsets[0], 1],
        positions[offsets[0], 2],
        positions[offsets[1], 0],
        positions[offsets[1], 1],
        positions[offsets[1], 2],
        positions[offsets[2], 0],
        positions[offsets[2], 1],
        positions[offsets[2], 2],
        d_hat,
        xi,
        kappa,
        gradient_pe,
        hessian_pe,
    )
    for left in qd.static(range(3)):
        for axis in qd.static(range(3)):
            gradient[offsets[left] * 3 + axis] = gradient_pe[left * 3 + axis]
        for right in qd.static(range(3)):
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    hessian[(offsets[left] * 3 + row) * 12 + offsets[right] * 3 + column] = hessian_pe[
                        (left * 3 + row) * 9 + right * 3 + column
                    ]


@qd.func
def _embed_pp(
    positions: qd.template(),
    offsets: qd.template(),
    d_hat,
    xi,
    kappa,
    gradient: qd.template(),
    hessian: qd.template(),
):
    gradient_pp = qd.Vector.zero(qd.f64, 6)
    hessian_pp = qd.Vector.zero(qd.f64, 36)
    gipc_barrier_grad_hess_pp(
        positions[offsets[0], 0],
        positions[offsets[0], 1],
        positions[offsets[0], 2],
        positions[offsets[1], 0],
        positions[offsets[1], 1],
        positions[offsets[1], 2],
        d_hat,
        xi,
        kappa,
        gradient_pp,
        hessian_pp,
    )
    for left in qd.static(range(2)):
        for axis in qd.static(range(3)):
            gradient[offsets[left] * 3 + axis] = gradient_pp[left * 3 + axis]
        for right in qd.static(range(2)):
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    hessian[(offsets[left] * 3 + row) * 12 + offsets[right] * 3 + column] = hessian_pp[
                        (left * 3 + row) * 6 + right * 3 + column
                    ]


@qd.func
def _embed_friction_pe(
    current: qd.template(),
    lagged: qd.template(),
    offsets: qd.template(),
    d_hat,
    xi,
    kappa,
    mu,
    eps_vh,
    gradient: qd.template(),
    hessian: qd.template(),
):
    current_pe = qd.Matrix.zero(qd.f64, 3, 3)
    lagged_pe = qd.Matrix.zero(qd.f64, 3, 3)
    for point in qd.static(range(3)):
        for axis in qd.static(range(3)):
            current_pe[point, axis] = current[offsets[point], axis]
            lagged_pe[point, axis] = lagged[offsets[point], axis]
    gradient_pe = qd.Vector.zero(qd.f64, 9)
    hessian_pe = qd.Vector.zero(qd.f64, 81)
    gipc_friction_grad_hess_pe(
        current_pe,
        lagged_pe,
        d_hat,
        kappa,
        mu,
        eps_vh,
        xi,
        gradient_pe,
        hessian_pe,
    )
    for left in qd.static(range(3)):
        for axis in qd.static(range(3)):
            gradient[offsets[left] * 3 + axis] = gradient_pe[left * 3 + axis]
        for right in qd.static(range(3)):
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    hessian[(offsets[left] * 3 + row) * 12 + offsets[right] * 3 + column] = hessian_pe[
                        (left * 3 + row) * 9 + right * 3 + column
                    ]


@qd.func
def _embed_friction_pp(
    current: qd.template(),
    lagged: qd.template(),
    offsets: qd.template(),
    d_hat,
    xi,
    kappa,
    mu,
    eps_vh,
    gradient: qd.template(),
    hessian: qd.template(),
):
    current_pp = qd.Matrix.zero(qd.f64, 2, 3)
    lagged_pp = qd.Matrix.zero(qd.f64, 2, 3)
    for point in qd.static(range(2)):
        for axis in qd.static(range(3)):
            current_pp[point, axis] = current[offsets[point], axis]
            lagged_pp[point, axis] = lagged[offsets[point], axis]
    gradient_pp = qd.Vector.zero(qd.f64, 6)
    hessian_pp = qd.Vector.zero(qd.f64, 36)
    gipc_friction_grad_hess_pp(
        current_pp,
        lagged_pp,
        d_hat,
        kappa,
        mu,
        eps_vh,
        xi,
        gradient_pp,
        hessian_pp,
    )
    for left in qd.static(range(2)):
        for axis in qd.static(range(3)):
            gradient[offsets[left] * 3 + axis] = gradient_pp[left * 3 + axis]
        for right in qd.static(range(2)):
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    hessian[(offsets[left] * 3 + row) * 12 + offsets[right] * 3 + column] = hessian_pp[
                        (left * 3 + row) * 6 + right * 3 + column
                    ]


@qd.func
def _scale_pair(gradient: qd.template(), hessian: qd.template(), scale):
    for component in qd.static(range(12)):
        gradient[component] = gradient[component] * scale
    for component in range(144):
        hessian[component] = hessian[component] * scale


@qd.func(requires_top_level=True)
def friction_pair_filter_pt_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_pt[()]):
        vert_index = contact.pairs_pt[pair_index, 0]
        face_index = contact.pairs_pt[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        ids[0] = surface.surf_verts[vert_index]
        for corner in qd.static(range(3)):
            ids[corner + 1] = surface.surf_triangles[face_index, corner]
        lagged = qd.Matrix.zero(qd.f64, 4, 3)
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                lagged[point, axis] = contact.lagged_positions[ids[point], axis]
        flag = pt_distance_flag(
            lagged[0, 0],
            lagged[0, 1],
            lagged[0, 2],
            lagged[1, 0],
            lagged[1, 1],
            lagged[1, 2],
            lagged[2, 0],
            lagged[2, 1],
            lagged[2, 2],
            lagged[3, 0],
            lagged[3, 1],
            lagged[3, 2],
        )
        distance_squared = pt_flagged_distance2(flag, lagged)
        d_hat = pair_d_hat_pt(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_pt(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        if distance_squared > xi * xi and distance_squared < (d_hat + xi) * (d_hat + xi) and _pt_enabled(contact, ids):
            output = qd.atomic_add(contact.n_friction_pairs_pt[()], 1)
            count = _popcount4(flag)
            if output < contact.max_friction_pairs_pt[()]:
                contact.friction_pairs_pt[output, 0] = vert_index
                contact.friction_pairs_pt[output, 1] = face_index
                contact.friction_flags_pt[output] = flag
                qd.atomic_add(contact.n_friction_demand_doublets[()], count)
                qd.atomic_add(contact.n_friction_demand_triplets[()], count * (count + 1) // 2)
            else:
                contact.friction_overflow_flag[()] = 1


@qd.func(requires_top_level=True)
def friction_pair_filter_ee_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_ee[()]):
        edge_a = contact.pairs_ee[pair_index, 0]
        edge_b = contact.pairs_ee[pair_index, 1]
        ids = qd.Vector(
            [
                surface.surf_edges[edge_a, 0],
                surface.surf_edges[edge_a, 1],
                surface.surf_edges[edge_b, 0],
                surface.surf_edges[edge_b, 1],
            ]
        )
        lagged = qd.Matrix.zero(qd.f64, 4, 3)
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                lagged[point, axis] = contact.lagged_positions[ids[point], axis]
        flag = ee_distance_flag(
            lagged[0, 0],
            lagged[0, 1],
            lagged[0, 2],
            lagged[1, 0],
            lagged[1, 1],
            lagged[1, 2],
            lagged[2, 0],
            lagged[2, 1],
            lagged[2, 2],
            lagged[3, 0],
            lagged[3, 1],
            lagged[3, 2],
        )
        distance_squared = ee_flagged_distance2(flag, lagged)
        d_hat = pair_d_hat_ee(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_ee(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        count = _popcount4(flag)
        keep = (
            distance_squared > xi * xi and distance_squared < (d_hat + xi) * (d_hat + xi) and _ee_enabled(contact, ids)
        )
        if count == 4:
            edge_a_x = lagged[1, 0] - lagged[0, 0]
            edge_a_y = lagged[1, 1] - lagged[0, 1]
            edge_a_z = lagged[1, 2] - lagged[0, 2]
            edge_b_x = lagged[3, 0] - lagged[2, 0]
            edge_b_y = lagged[3, 1] - lagged[2, 1]
            edge_b_z = lagged[3, 2] - lagged[2, 2]
            cross_x = edge_a_y * edge_b_z - edge_a_z * edge_b_y
            cross_y = edge_a_z * edge_b_x - edge_a_x * edge_b_z
            cross_z = edge_a_x * edge_b_y - edge_a_y * edge_b_x
            cross_squared = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
            edge_a_squared = edge_a_x * edge_a_x + edge_a_y * edge_a_y + edge_a_z * edge_a_z
            edge_b_squared = edge_b_x * edge_b_x + edge_b_y * edge_b_y + edge_b_z * edge_b_z
            if cross_squared < 1.0e-3 * edge_a_squared * edge_b_squared:
                keep = False
        if keep:
            output = qd.atomic_add(contact.n_friction_pairs_ee[()], 1)
            if output < contact.max_friction_pairs_ee[()]:
                contact.friction_pairs_ee[output, 0] = edge_a
                contact.friction_pairs_ee[output, 1] = edge_b
                contact.friction_flags_ee[output] = flag
                qd.atomic_add(contact.n_friction_demand_doublets[()], count)
                qd.atomic_add(contact.n_friction_demand_triplets[()], count * (count + 1) // 2)
            else:
                contact.friction_overflow_flag[()] = 1


@qd.func(requires_top_level=True)
def halfplane_friction_pair_filter_kernel(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
):
    for pair_index in range(contact.n_pairs_ph[()]):
        surface_vertex = contact.pairs_ph[pair_index, 0]
        plane = contact.pairs_ph[pair_index, 1]
        vertex_id = surface.surf_verts[surface_vertex]
        distance = halfplane_signed_distance(
            contact.lagged_positions[vertex_id, 0],
            contact.lagged_positions[vertex_id, 1],
            contact.lagged_positions[vertex_id, 2],
            contact.halfplane_positions[plane, 0],
            contact.halfplane_positions[plane, 1],
            contact.halfplane_positions[plane, 2],
            contact.halfplane_normals[plane, 0],
            contact.halfplane_normals[plane, 1],
            contact.halfplane_normals[plane, 2],
        )
        d_hat = pair_d_hat_ph(vertex.d_hats, contact.d_hat[()], vertex_id)
        xi = pair_thickness_ph(vertex.thicknesses, vertex_id)
        if distance > xi and distance < d_hat + xi and _ph_enabled(contact, vertex_id):
            output = qd.atomic_add(contact.n_friction_pairs_ph[()], 1)
            if output < contact.max_friction_pairs_ph[()]:
                contact.friction_pairs_ph[output, 0] = surface_vertex
                contact.friction_pairs_ph[output, 1] = plane
                contact.friction_flags_ph[output] = 1
                qd.atomic_add(contact.n_friction_demand_doublets[()], 1)
                qd.atomic_add(contact.n_friction_demand_triplets[()], 1)
            else:
                contact.friction_overflow_flag[()] = 1


@qd.func(requires_top_level=True)
def cipc_friction_assemble_pt_kernel(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    dt,
):
    for pair_index in range(contact.n_friction_pairs_pt[()]):
        vert_index = contact.friction_pairs_pt[pair_index, 0]
        face_index = contact.friction_pairs_pt[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        ids[0] = surface.surf_verts[vert_index]
        for corner in qd.static(range(3)):
            ids[corner + 1] = surface.surf_triangles[face_index, corner]
        current = qd.Matrix.zero(qd.f64, 4, 3)
        lagged = qd.Matrix.zero(qd.f64, 4, 3)
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                current[point, axis] = vertex.positions[ids[point], axis]
                lagged[point, axis] = contact.lagged_positions[ids[point], axis]
        flag = contact.friction_flags_pt[pair_index]
        count = _popcount4(flag)
        d_hat = pair_d_hat_pt(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_pt(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        kappa = ct_query_pt(
            contact.kappa_table,
            contact.n_contact_elements[()],
            contact.vert_contact_element_ids[ids[0]],
            contact.vert_contact_element_ids[ids[1]],
            contact.vert_contact_element_ids[ids[2]],
            contact.vert_contact_element_ids[ids[3]],
        )
        kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, count)
        mu = ct_query_pt(
            contact.mu_table,
            contact.n_contact_elements[()],
            contact.vert_contact_element_ids[ids[0]],
            contact.vert_contact_element_ids[ids[1]],
            contact.vert_contact_element_ids[ids[2]],
            contact.vert_contact_element_ids[ids[3]],
        )
        gradient = qd.Vector.zero(qd.f64, 12)
        hessian = qd.Vector.zero(qd.f64, 144)
        if count == 4:
            gipc_friction_grad_hess_pt(
                current,
                lagged,
                d_hat,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                xi,
                gradient,
                hessian,
            )
        elif count == 3:
            offsets = qd.Vector.zero(qd.i32, 4)
            flag_active_offsets(flag, offsets)
            _embed_friction_pe(
                current,
                lagged,
                offsets,
                d_hat,
                xi,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                gradient,
                hessian,
            )
        else:
            offsets = qd.Vector.zero(qd.i32, 4)
            flag_active_offsets(flag, offsets)
            _embed_friction_pp(
                current,
                lagged,
                offsets,
                d_hat,
                xi,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                gradient,
                hessian,
            )
        scale = contact.dt_sq[()] * cipc_pair_area_weight(
            surface.vert_area_weights[vert_index],
            surface.face_area_weights[face_index],
            d_hat,
        )
        _scale_pair(gradient, hessian, scale)
        cipc_scatter_doublets(contact, gradient, flag, ids)
        cipc_scatter_triplets_upper(contact, hessian, flag, ids)


@qd.func(requires_top_level=True)
def cipc_friction_assemble_ee_kernel(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    dt,
):
    for pair_index in range(contact.n_friction_pairs_ee[()]):
        edge_a = contact.friction_pairs_ee[pair_index, 0]
        edge_b = contact.friction_pairs_ee[pair_index, 1]
        ids = qd.Vector(
            [
                surface.surf_edges[edge_a, 0],
                surface.surf_edges[edge_a, 1],
                surface.surf_edges[edge_b, 0],
                surface.surf_edges[edge_b, 1],
            ]
        )
        current = qd.Matrix.zero(qd.f64, 4, 3)
        lagged = qd.Matrix.zero(qd.f64, 4, 3)
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                current[point, axis] = vertex.positions[ids[point], axis]
                lagged[point, axis] = contact.lagged_positions[ids[point], axis]
        flag = contact.friction_flags_ee[pair_index]
        count = _popcount4(flag)
        d_hat = pair_d_hat_ee(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_ee(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        kappa = ct_query_ee(
            contact.kappa_table,
            contact.n_contact_elements[()],
            contact.vert_contact_element_ids[ids[0]],
            contact.vert_contact_element_ids[ids[1]],
            contact.vert_contact_element_ids[ids[2]],
            contact.vert_contact_element_ids[ids[3]],
        )
        kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, count)
        mu = ct_query_ee(
            contact.mu_table,
            contact.n_contact_elements[()],
            contact.vert_contact_element_ids[ids[0]],
            contact.vert_contact_element_ids[ids[1]],
            contact.vert_contact_element_ids[ids[2]],
            contact.vert_contact_element_ids[ids[3]],
        )
        gradient = qd.Vector.zero(qd.f64, 12)
        hessian = qd.Vector.zero(qd.f64, 144)
        if count == 4:
            gipc_friction_grad_hess_ee(
                current,
                lagged,
                d_hat,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                xi,
                gradient,
                hessian,
            )
        elif count == 3:
            offsets = qd.Vector.zero(qd.i32, 4)
            is_point_on_a = qd.Vector.zero(qd.i32, 1)
            flag_active_offsets_for_ee(flag, offsets, is_point_on_a)
            _embed_friction_pe(
                current,
                lagged,
                offsets,
                d_hat,
                xi,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                gradient,
                hessian,
            )
        else:
            offsets = qd.Vector.zero(qd.i32, 4)
            flag_active_offsets(flag, offsets)
            _embed_friction_pp(
                current,
                lagged,
                offsets,
                d_hat,
                xi,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                gradient,
                hessian,
            )
        scale = contact.dt_sq[()] * cipc_pair_area_weight(
            surface.edge_area_weights[edge_a],
            surface.edge_area_weights[edge_b],
            d_hat,
        )
        _scale_pair(gradient, hessian, scale)
        cipc_scatter_doublets(contact, gradient, flag, ids)
        cipc_scatter_triplets_upper(contact, hessian, flag, ids)


@qd.func(requires_top_level=True)
def cipc_friction_energy_pt_kernel(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    dt,
):
    for pair_index in range(contact.n_friction_pairs_pt[()]):
        vert_index = contact.friction_pairs_pt[pair_index, 0]
        face_index = contact.friction_pairs_pt[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        ids[0] = surface.surf_verts[vert_index]
        for corner in qd.static(range(3)):
            ids[corner + 1] = surface.surf_triangles[face_index, corner]
        current = qd.Matrix.zero(qd.f64, 4, 3)
        lagged = qd.Matrix.zero(qd.f64, 4, 3)
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                current[point, axis] = vertex.positions[ids[point], axis]
                lagged[point, axis] = contact.lagged_positions[ids[point], axis]
        flag = contact.friction_flags_pt[pair_index]
        count = _popcount4(flag)
        d_hat = pair_d_hat_pt(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_pt(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        kappa = ct_query_pt(
            contact.kappa_table,
            contact.n_contact_elements[()],
            contact.vert_contact_element_ids[ids[0]],
            contact.vert_contact_element_ids[ids[1]],
            contact.vert_contact_element_ids[ids[2]],
            contact.vert_contact_element_ids[ids[3]],
        )
        kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, count)
        mu = ct_query_pt(
            contact.mu_table,
            contact.n_contact_elements[()],
            contact.vert_contact_element_ids[ids[0]],
            contact.vert_contact_element_ids[ids[1]],
            contact.vert_contact_element_ids[ids[2]],
            contact.vert_contact_element_ids[ids[3]],
        )
        energy = qd.f64(0.0)
        if count == 4:
            energy = gipc_friction_energy_pt(
                current,
                lagged,
                d_hat,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                xi,
            )
        elif count == 3:
            offsets = qd.Vector.zero(qd.i32, 4)
            flag_active_offsets(flag, offsets)
            current_pe = qd.Matrix.zero(qd.f64, 3, 3)
            lagged_pe = qd.Matrix.zero(qd.f64, 3, 3)
            for point in qd.static(range(3)):
                for axis in qd.static(range(3)):
                    current_pe[point, axis] = current[offsets[point], axis]
                    lagged_pe[point, axis] = lagged[offsets[point], axis]
            energy = gipc_friction_energy_pe(
                current_pe,
                lagged_pe,
                d_hat,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                xi,
            )
        else:
            offsets = qd.Vector.zero(qd.i32, 4)
            flag_active_offsets(flag, offsets)
            current_pp = qd.Matrix.zero(qd.f64, 2, 3)
            lagged_pp = qd.Matrix.zero(qd.f64, 2, 3)
            for point in qd.static(range(2)):
                for axis in qd.static(range(3)):
                    current_pp[point, axis] = current[offsets[point], axis]
                    lagged_pp[point, axis] = lagged[offsets[point], axis]
            energy = gipc_friction_energy_pp(
                current_pp,
                lagged_pp,
                d_hat,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                xi,
            )
        scale = contact.dt_sq[()] * cipc_pair_area_weight(
            surface.vert_area_weights[vert_index],
            surface.face_area_weights[face_index],
            d_hat,
        )
        qd.atomic_add(contact.friction_energy[()], energy * scale)


@qd.func(requires_top_level=True)
def cipc_friction_energy_ee_kernel(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    dt,
):
    for pair_index in range(contact.n_friction_pairs_ee[()]):
        edge_a = contact.friction_pairs_ee[pair_index, 0]
        edge_b = contact.friction_pairs_ee[pair_index, 1]
        ids = qd.Vector(
            [
                surface.surf_edges[edge_a, 0],
                surface.surf_edges[edge_a, 1],
                surface.surf_edges[edge_b, 0],
                surface.surf_edges[edge_b, 1],
            ]
        )
        current = qd.Matrix.zero(qd.f64, 4, 3)
        lagged = qd.Matrix.zero(qd.f64, 4, 3)
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                current[point, axis] = vertex.positions[ids[point], axis]
                lagged[point, axis] = contact.lagged_positions[ids[point], axis]
        flag = contact.friction_flags_ee[pair_index]
        count = _popcount4(flag)
        d_hat = pair_d_hat_ee(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_ee(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        kappa = ct_query_ee(
            contact.kappa_table,
            contact.n_contact_elements[()],
            contact.vert_contact_element_ids[ids[0]],
            contact.vert_contact_element_ids[ids[1]],
            contact.vert_contact_element_ids[ids[2]],
            contact.vert_contact_element_ids[ids[3]],
        )
        kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, count)
        mu = ct_query_ee(
            contact.mu_table,
            contact.n_contact_elements[()],
            contact.vert_contact_element_ids[ids[0]],
            contact.vert_contact_element_ids[ids[1]],
            contact.vert_contact_element_ids[ids[2]],
            contact.vert_contact_element_ids[ids[3]],
        )
        energy = qd.f64(0.0)
        if count == 4:
            energy = gipc_friction_energy_ee(
                current,
                lagged,
                d_hat,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                xi,
            )
        elif count == 3:
            offsets = qd.Vector.zero(qd.i32, 4)
            is_point_on_a = qd.Vector.zero(qd.i32, 1)
            flag_active_offsets_for_ee(flag, offsets, is_point_on_a)
            current_pe = qd.Matrix.zero(qd.f64, 3, 3)
            lagged_pe = qd.Matrix.zero(qd.f64, 3, 3)
            for point in qd.static(range(3)):
                for axis in qd.static(range(3)):
                    current_pe[point, axis] = current[offsets[point], axis]
                    lagged_pe[point, axis] = lagged[offsets[point], axis]
            energy = gipc_friction_energy_pe(
                current_pe,
                lagged_pe,
                d_hat,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                xi,
            )
        else:
            offsets = qd.Vector.zero(qd.i32, 4)
            flag_active_offsets(flag, offsets)
            current_pp = qd.Matrix.zero(qd.f64, 2, 3)
            lagged_pp = qd.Matrix.zero(qd.f64, 2, 3)
            for point in qd.static(range(2)):
                for axis in qd.static(range(3)):
                    current_pp[point, axis] = current[offsets[point], axis]
                    lagged_pp[point, axis] = lagged[offsets[point], axis]
            energy = gipc_friction_energy_pp(
                current_pp,
                lagged_pp,
                d_hat,
                kappa,
                mu,
                contact.friction_eps_v[()] * dt,
                xi,
            )
        scale = contact.dt_sq[()] * cipc_pair_area_weight(
            surface.edge_area_weights[edge_a],
            surface.edge_area_weights[edge_b],
            d_hat,
        )
        qd.atomic_add(contact.friction_energy[()], energy * scale)


@qd.func(requires_top_level=True)
def cipc_count_active_pt_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_pt[()]):
        vert_index = contact.pairs_pt[pair_index, 0]
        face_index = contact.pairs_pt[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        positions = qd.Matrix.zero(qd.f64, 4, 3)
        ids[0] = surface.surf_verts[vert_index]
        for corner in qd.static(range(3)):
            ids[corner + 1] = surface.surf_triangles[face_index, corner]
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                positions[point, axis] = vertex.positions[ids[point], axis]
        flag = pt_distance_flag(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            positions[1, 0],
            positions[1, 1],
            positions[1, 2],
            positions[2, 0],
            positions[2, 1],
            positions[2, 2],
            positions[3, 0],
            positions[3, 1],
            positions[3, 2],
        )
        distance_squared = pt_flagged_distance2(flag, positions)
        d_hat = pair_d_hat_pt(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_pt(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        if distance_squared <= xi * xi:
            contact.intersection_flag[()] = 1
        elif distance_squared < (d_hat + xi) * (d_hat + xi):
            count = _popcount4(flag)
            qd.atomic_add(contact.n_counted_doublets[()], count)
            qd.atomic_add(contact.n_counted_triplets[()], count * (count + 1) // 2)


@qd.func(requires_top_level=True)
def cipc_count_active_ee_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_ee[()]):
        edge_a = contact.pairs_ee[pair_index, 0]
        edge_b = contact.pairs_ee[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        positions = qd.Matrix.zero(qd.f64, 4, 3)
        ids[0] = surface.surf_edges[edge_a, 0]
        ids[1] = surface.surf_edges[edge_a, 1]
        ids[2] = surface.surf_edges[edge_b, 0]
        ids[3] = surface.surf_edges[edge_b, 1]
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                positions[point, axis] = vertex.positions[ids[point], axis]
        flag = ee_distance_flag(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            positions[1, 0],
            positions[1, 1],
            positions[1, 2],
            positions[2, 0],
            positions[2, 1],
            positions[2, 2],
            positions[3, 0],
            positions[3, 1],
            positions[3, 2],
        )
        distance_squared = ee_flagged_distance2(flag, positions)
        d_hat = pair_d_hat_ee(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_ee(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        if distance_squared <= xi * xi:
            contact.intersection_flag[()] = 1
        elif distance_squared < (d_hat + xi) * (d_hat + xi):
            count = _popcount4(flag)
            edge_a_x = positions[1, 0] - positions[0, 0]
            edge_a_y = positions[1, 1] - positions[0, 1]
            edge_a_z = positions[1, 2] - positions[0, 2]
            edge_b_x = positions[3, 0] - positions[2, 0]
            edge_b_y = positions[3, 1] - positions[2, 1]
            edge_b_z = positions[3, 2] - positions[2, 2]
            cross_x = edge_a_y * edge_b_z - edge_a_z * edge_b_y
            cross_y = edge_a_z * edge_b_x - edge_a_x * edge_b_z
            cross_z = edge_a_x * edge_b_y - edge_a_y * edge_b_x
            I1 = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
            rest_a_x = vertex.x_bar[ids[1], 0] - vertex.x_bar[ids[0], 0]
            rest_a_y = vertex.x_bar[ids[1], 1] - vertex.x_bar[ids[0], 1]
            rest_a_z = vertex.x_bar[ids[1], 2] - vertex.x_bar[ids[0], 2]
            rest_b_x = vertex.x_bar[ids[3], 0] - vertex.x_bar[ids[2], 0]
            rest_b_y = vertex.x_bar[ids[3], 1] - vertex.x_bar[ids[2], 1]
            rest_b_z = vertex.x_bar[ids[3], 2] - vertex.x_bar[ids[2], 2]
            eps_x = (
                1.0e-3
                * (rest_a_x * rest_a_x + rest_a_y * rest_a_y + rest_a_z * rest_a_z)
                * (rest_b_x * rest_b_x + rest_b_y * rest_b_y + rest_b_z * rest_b_z)
            )
            demand = count
            if I1 < eps_x:
                demand = 4
            qd.atomic_add(contact.n_counted_doublets[()], demand)
            qd.atomic_add(contact.n_counted_triplets[()], demand * (demand + 1) // 2)


@qd.func(requires_top_level=True)
def cipc_count_active_ph_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_ph[()]):
        surface_vertex = contact.pairs_ph[pair_index, 0]
        plane = contact.pairs_ph[pair_index, 1]
        vertex_id = surface.surf_verts[surface_vertex]
        distance = halfplane_signed_distance(
            vertex.positions[vertex_id, 0],
            vertex.positions[vertex_id, 1],
            vertex.positions[vertex_id, 2],
            contact.halfplane_positions[plane, 0],
            contact.halfplane_positions[plane, 1],
            contact.halfplane_positions[plane, 2],
            contact.halfplane_normals[plane, 0],
            contact.halfplane_normals[plane, 1],
            contact.halfplane_normals[plane, 2],
        )
        d_hat = pair_d_hat_ph(vertex.d_hats, contact.d_hat[()], vertex_id)
        xi = pair_thickness_ph(vertex.thicknesses, vertex_id)
        if distance > xi and distance < d_hat + xi and _ph_enabled(contact, vertex_id):
            qd.atomic_add(contact.n_counted_doublets[()], 1)
            qd.atomic_add(contact.n_counted_triplets[()], 1)


@qd.func(requires_top_level=True)
def cipc_filter_assemble_pt_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_pt[()]):
        vert_index = contact.pairs_pt[pair_index, 0]
        face_index = contact.pairs_pt[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        positions = qd.Matrix.zero(qd.f64, 4, 3)
        ids[0] = surface.surf_verts[vert_index]
        for corner in qd.static(range(3)):
            ids[corner + 1] = surface.surf_triangles[face_index, corner]
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                positions[point, axis] = vertex.positions[ids[point], axis]
        flag = pt_distance_flag(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            positions[1, 0],
            positions[1, 1],
            positions[1, 2],
            positions[2, 0],
            positions[2, 1],
            positions[2, 2],
            positions[3, 0],
            positions[3, 1],
            positions[3, 2],
        )
        distance_squared = pt_flagged_distance2(flag, positions)
        d_hat = pair_d_hat_pt(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_pt(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        if distance_squared > xi * xi and distance_squared < (d_hat + xi) * (d_hat + xi) and _pt_enabled(contact, ids):
            count = _popcount4(flag)
            report_pair_min_gap(
                contact,
                vertex,
                (qd.sqrt(distance_squared) - xi) / d_hat,
                ids,
                count,
            )
            kappa = ct_query_pt(
                contact.kappa_table,
                contact.n_contact_elements[()],
                contact.vert_contact_element_ids[ids[0]],
                contact.vert_contact_element_ids[ids[1]],
                contact.vert_contact_element_ids[ids[2]],
                contact.vert_contact_element_ids[ids[3]],
            )
            kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, count)
            gradient = qd.Vector.zero(qd.f64, 12)
            hessian = qd.Vector.zero(qd.f64, 144)
            if count == 4:
                gipc_barrier_grad_hess_pt(
                    positions[0, 0],
                    positions[0, 1],
                    positions[0, 2],
                    positions[1, 0],
                    positions[1, 1],
                    positions[1, 2],
                    positions[2, 0],
                    positions[2, 1],
                    positions[2, 2],
                    positions[3, 0],
                    positions[3, 1],
                    positions[3, 2],
                    d_hat,
                    xi,
                    kappa,
                    gradient,
                    hessian,
                )
            elif count == 3:
                offsets = qd.Vector.zero(qd.i32, 4)
                flag_active_offsets(flag, offsets)
                _embed_pe(positions, offsets, d_hat, xi, kappa, gradient, hessian)
            else:
                offsets = qd.Vector.zero(qd.i32, 4)
                flag_active_offsets(flag, offsets)
                _embed_pp(positions, offsets, d_hat, xi, kappa, gradient, hessian)
            scale = contact.dt_sq[()] * cipc_pair_area_weight(
                surface.vert_area_weights[vert_index],
                surface.face_area_weights[face_index],
                d_hat,
            )
            _scale_pair(gradient, hessian, scale)
            qd.atomic_add(contact.n_active_pairs[()], 1)
            cipc_scatter_doublets(contact, gradient, flag, ids)
            cipc_scatter_triplets_upper(contact, hessian, flag, ids)


@qd.func(requires_top_level=True)
def cipc_filter_assemble_ee_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    qd.loop_config(name="cipc_filter_assemble_ee")
    for pair_index in range(contact.n_pairs_ee[()]):
        edge_a = contact.pairs_ee[pair_index, 0]
        edge_b = contact.pairs_ee[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        positions = qd.Matrix.zero(qd.f64, 4, 3)
        ids[0] = surface.surf_edges[edge_a, 0]
        ids[1] = surface.surf_edges[edge_a, 1]
        ids[2] = surface.surf_edges[edge_b, 0]
        ids[3] = surface.surf_edges[edge_b, 1]
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                positions[point, axis] = vertex.positions[ids[point], axis]
        flag = ee_distance_flag(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            positions[1, 0],
            positions[1, 1],
            positions[1, 2],
            positions[2, 0],
            positions[2, 1],
            positions[2, 2],
            positions[3, 0],
            positions[3, 1],
            positions[3, 2],
        )
        distance_squared = ee_flagged_distance2(flag, positions)
        d_hat = pair_d_hat_ee(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_ee(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        if distance_squared > xi * xi and distance_squared < (d_hat + xi) * (d_hat + xi) and _ee_enabled(contact, ids):
            count = _popcount4(flag)
            report_pair_min_gap(
                contact,
                vertex,
                (qd.sqrt(distance_squared) - xi) / d_hat,
                ids,
                count,
            )
            kappa = ct_query_ee(
                contact.kappa_table,
                contact.n_contact_elements[()],
                contact.vert_contact_element_ids[ids[0]],
                contact.vert_contact_element_ids[ids[1]],
                contact.vert_contact_element_ids[ids[2]],
                contact.vert_contact_element_ids[ids[3]],
            )
            kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, count)

            edge_a_x = positions[1, 0] - positions[0, 0]
            edge_a_y = positions[1, 1] - positions[0, 1]
            edge_a_z = positions[1, 2] - positions[0, 2]
            edge_b_x = positions[3, 0] - positions[2, 0]
            edge_b_y = positions[3, 1] - positions[2, 1]
            edge_b_z = positions[3, 2] - positions[2, 2]
            cross_x = edge_a_y * edge_b_z - edge_a_z * edge_b_y
            cross_y = edge_a_z * edge_b_x - edge_a_x * edge_b_z
            cross_z = edge_a_x * edge_b_y - edge_a_y * edge_b_x
            I1 = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
            rest_a_x = vertex.x_bar[ids[1], 0] - vertex.x_bar[ids[0], 0]
            rest_a_y = vertex.x_bar[ids[1], 1] - vertex.x_bar[ids[0], 1]
            rest_a_z = vertex.x_bar[ids[1], 2] - vertex.x_bar[ids[0], 2]
            rest_b_x = vertex.x_bar[ids[3], 0] - vertex.x_bar[ids[2], 0]
            rest_b_y = vertex.x_bar[ids[3], 1] - vertex.x_bar[ids[2], 1]
            rest_b_z = vertex.x_bar[ids[3], 2] - vertex.x_bar[ids[2], 2]
            eps_x = (
                1.0e-3
                * (rest_a_x * rest_a_x + rest_a_y * rest_a_y + rest_a_z * rest_a_z)
                * (rest_b_x * rest_b_x + rest_b_y * rest_b_y + rest_b_z * rest_b_z)
            )

            scale = contact.dt_sq[()] * cipc_pair_area_weight(
                surface.edge_area_weights[edge_a],
                surface.edge_area_weights[edge_b],
                d_hat,
            )
            if I1 >= eps_x and count == 4:
                values = qd.Vector.zero(qd.f64, 12)
                grad_scale = qd.Vector.zero(qd.f64, 1)
                hess_coef = qd.Vector.zero(qd.f64, 1)
                gipc_barrier_grad_hess_ee_rank1(
                    positions[0, 0],
                    positions[0, 1],
                    positions[0, 2],
                    positions[1, 0],
                    positions[1, 1],
                    positions[1, 2],
                    positions[2, 0],
                    positions[2, 1],
                    positions[2, 2],
                    positions[3, 0],
                    positions[3, 1],
                    positions[3, 2],
                    d_hat,
                    xi,
                    kappa,
                    values,
                    grad_scale,
                    hess_coef,
                )
                gradient = qd.Vector.zero(qd.f64, 12)
                for component in qd.static(range(12)):
                    gradient[component] = values[component] * grad_scale[0] * scale
                qd.atomic_add(contact.n_active_pairs[()], 1)
                cipc_scatter_doublets(contact, gradient, flag, ids)
                cipc_scatter_triplets_upper_rank1(contact, values, hess_coef[0] * scale, ids)
                continue

            gradient = qd.Vector.zero(qd.f64, 12)
            hessian = qd.Vector.zero(qd.f64, 144)
            scatter_ids = qd.Vector.zero(qd.i32, 4)
            for index in qd.static(range(4)):
                scatter_ids[index] = ids[index]
            scatter_flag = flag
            if I1 < eps_x:
                scatter_flag = qd.i32(0xF)
                if count == 4:
                    interior_distance_squared = gipc_d_EE(
                        positions[0, 0],
                        positions[0, 1],
                        positions[0, 2],
                        positions[1, 0],
                        positions[1, 1],
                        positions[1, 2],
                        positions[2, 0],
                        positions[2, 1],
                        positions[2, 2],
                        positions[3, 0],
                        positions[3, 1],
                        positions[3, 2],
                    )
                    if not (
                        interior_distance_squared > xi * xi and interior_distance_squared < (d_hat + xi) * (d_hat + xi)
                    ):
                        interior_distance_squared = distance_squared
                    gipc_barrier_grad_hess_ee_mollified(
                        positions[0, 0],
                        positions[0, 1],
                        positions[0, 2],
                        positions[1, 0],
                        positions[1, 1],
                        positions[1, 2],
                        positions[2, 0],
                        positions[2, 1],
                        positions[2, 2],
                        positions[3, 0],
                        positions[3, 1],
                        positions[3, 2],
                        vertex.x_bar[ids[0], 0],
                        vertex.x_bar[ids[0], 1],
                        vertex.x_bar[ids[0], 2],
                        vertex.x_bar[ids[1], 0],
                        vertex.x_bar[ids[1], 1],
                        vertex.x_bar[ids[1], 2],
                        vertex.x_bar[ids[2], 0],
                        vertex.x_bar[ids[2], 1],
                        vertex.x_bar[ids[2], 2],
                        vertex.x_bar[ids[3], 0],
                        vertex.x_bar[ids[3], 1],
                        vertex.x_bar[ids[3], 2],
                        d_hat,
                        xi,
                        kappa,
                        interior_distance_squared,
                        gradient,
                        hessian,
                    )
                elif count == 3:
                    offsets = qd.Vector.zero(qd.i32, 4)
                    is_point_on_a = qd.Vector.zero(qd.i32, 1)
                    flag_active_offsets_for_ee(flag, offsets, is_point_on_a)
                    missing = 6 - offsets[0] - offsets[1] - offsets[2]
                    scatter_ids[0] = ids[offsets[0]]
                    scatter_ids[1] = ids[offsets[1]]
                    scatter_ids[2] = ids[offsets[2]]
                    scatter_ids[3] = ids[missing]
                    gipc_barrier_grad_hess_pe_mollified(
                        positions[offsets[0], 0],
                        positions[offsets[0], 1],
                        positions[offsets[0], 2],
                        positions[offsets[1], 0],
                        positions[offsets[1], 1],
                        positions[offsets[1], 2],
                        positions[offsets[2], 0],
                        positions[offsets[2], 1],
                        positions[offsets[2], 2],
                        positions[missing, 0],
                        positions[missing, 1],
                        positions[missing, 2],
                        vertex.x_bar[ids[offsets[0]], 0],
                        vertex.x_bar[ids[offsets[0]], 1],
                        vertex.x_bar[ids[offsets[0]], 2],
                        vertex.x_bar[ids[offsets[1]], 0],
                        vertex.x_bar[ids[offsets[1]], 1],
                        vertex.x_bar[ids[offsets[1]], 2],
                        vertex.x_bar[ids[offsets[2]], 0],
                        vertex.x_bar[ids[offsets[2]], 1],
                        vertex.x_bar[ids[offsets[2]], 2],
                        vertex.x_bar[ids[missing], 0],
                        vertex.x_bar[ids[missing], 1],
                        vertex.x_bar[ids[missing], 2],
                        d_hat,
                        xi,
                        kappa,
                        gradient,
                        hessian,
                    )
                else:
                    offsets = qd.Vector.zero(qd.i32, 4)
                    flag_active_offsets(flag, offsets)
                    a_other = offsets[0] ^ 1
                    b_other = 5 - offsets[1]
                    scatter_ids[0] = ids[offsets[0]]
                    scatter_ids[1] = ids[offsets[1]]
                    scatter_ids[2] = ids[a_other]
                    scatter_ids[3] = ids[b_other]
                    gipc_barrier_grad_hess_pp_mollified(
                        positions[offsets[0], 0],
                        positions[offsets[0], 1],
                        positions[offsets[0], 2],
                        positions[offsets[1], 0],
                        positions[offsets[1], 1],
                        positions[offsets[1], 2],
                        positions[a_other, 0],
                        positions[a_other, 1],
                        positions[a_other, 2],
                        positions[b_other, 0],
                        positions[b_other, 1],
                        positions[b_other, 2],
                        vertex.x_bar[ids[offsets[0]], 0],
                        vertex.x_bar[ids[offsets[0]], 1],
                        vertex.x_bar[ids[offsets[0]], 2],
                        vertex.x_bar[ids[offsets[1]], 0],
                        vertex.x_bar[ids[offsets[1]], 1],
                        vertex.x_bar[ids[offsets[1]], 2],
                        vertex.x_bar[ids[a_other], 0],
                        vertex.x_bar[ids[a_other], 1],
                        vertex.x_bar[ids[a_other], 2],
                        vertex.x_bar[ids[b_other], 0],
                        vertex.x_bar[ids[b_other], 1],
                        vertex.x_bar[ids[b_other], 2],
                        d_hat,
                        xi,
                        kappa,
                        gradient,
                        hessian,
                    )
            else:
                if count == 3:
                    offsets = qd.Vector.zero(qd.i32, 4)
                    is_point_on_a = qd.Vector.zero(qd.i32, 1)
                    flag_active_offsets_for_ee(flag, offsets, is_point_on_a)
                    _embed_pe(positions, offsets, d_hat, xi, kappa, gradient, hessian)
                else:
                    offsets = qd.Vector.zero(qd.i32, 4)
                    flag_active_offsets(flag, offsets)
                    _embed_pp(positions, offsets, d_hat, xi, kappa, gradient, hessian)
            _scale_pair(gradient, hessian, scale)
            qd.atomic_add(contact.n_active_pairs[()], 1)
            cipc_scatter_doublets(contact, gradient, scatter_flag, scatter_ids)
            cipc_scatter_triplets_upper(contact, hessian, scatter_flag, scatter_ids)


@qd.func(requires_top_level=True)
def cipc_filter_assemble_ph_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_ph[()]):
        surface_vertex = contact.pairs_ph[pair_index, 0]
        plane = contact.pairs_ph[pair_index, 1]
        vertex_id = surface.surf_verts[surface_vertex]
        normal_x = contact.halfplane_normals[plane, 0]
        normal_y = contact.halfplane_normals[plane, 1]
        normal_z = contact.halfplane_normals[plane, 2]
        distance = halfplane_signed_distance(
            vertex.positions[vertex_id, 0],
            vertex.positions[vertex_id, 1],
            vertex.positions[vertex_id, 2],
            contact.halfplane_positions[plane, 0],
            contact.halfplane_positions[plane, 1],
            contact.halfplane_positions[plane, 2],
            normal_x,
            normal_y,
            normal_z,
        )
        d_hat = pair_d_hat_ph(vertex.d_hats, contact.d_hat[()], vertex_id)
        xi = pair_thickness_ph(vertex.thicknesses, vertex_id)
        if distance > xi and distance < d_hat + xi and _ph_enabled(contact, vertex_id):
            element = contact.vert_contact_element_ids[vertex_id]
            kappa = ct_query(
                contact.kappa_table,
                contact.n_contact_elements[()],
                element,
                0,
            )
            ids = qd.Vector([vertex_id, 0, 0, 0])
            report_pair_min_gap(contact, vertex, (distance - xi) / d_hat, ids, 1)
            kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, 1)
            distance_squared = distance * distance
            gradient = qd.Vector.zero(qd.f64, 3)
            hessian = qd.Vector.zero(qd.f64, 9)
            first = gipc_barrier_first_derivative(distance_squared, d_hat, xi, kappa)
            second = gipc_barrier_second_derivative(distance_squared, d_hat, xi, kappa)
            halfplane_barrier_gradient(first, distance, normal_x, normal_y, normal_z, gradient)
            halfplane_barrier_hessian(
                second,
                first,
                distance_squared,
                normal_x,
                normal_y,
                normal_z,
                hessian,
            )
            scale = contact.dt_sq[()] * surface.vert_area_weights[surface_vertex] * d_hat
            output_doublet = qd.atomic_add(contact.n_contact_doublets[()], 1)
            contact.contact_doublet_vertices[output_doublet] = vertex_id
            for axis in qd.static(range(3)):
                contact.contact_doublet_gradients[output_doublet, axis] = gradient[axis] * scale
            output_triplet = qd.atomic_add(contact.n_contact_triplets[()], 1)
            contact.contact_triplet_rows[output_triplet] = vertex_id
            contact.contact_triplet_cols[output_triplet] = vertex_id
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    contact.contact_triplet_values[output_triplet, row, column] = hessian[row * 3 + column] * scale
            qd.atomic_add(contact.n_active_pairs[()], 1)


@qd.func(requires_top_level=True)
def cipc_halfplane_friction_assemble_kernel(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    dt,
):
    for pair_index in range(contact.n_friction_pairs_ph[()]):
        surface_vertex = contact.friction_pairs_ph[pair_index, 0]
        plane = contact.friction_pairs_ph[pair_index, 1]
        vertex_id = surface.surf_verts[surface_vertex]
        current = qd.Vector(
            [
                vertex.positions[vertex_id, 0],
                vertex.positions[vertex_id, 1],
                vertex.positions[vertex_id, 2],
            ]
        )
        lagged = qd.Vector(
            [
                contact.lagged_positions[vertex_id, 0],
                contact.lagged_positions[vertex_id, 1],
                contact.lagged_positions[vertex_id, 2],
            ]
        )
        normal_x = contact.halfplane_normals[plane, 0]
        normal_y = contact.halfplane_normals[plane, 1]
        normal_z = contact.halfplane_normals[plane, 2]
        lagged_distance = halfplane_signed_distance(
            lagged[0],
            lagged[1],
            lagged[2],
            contact.halfplane_positions[plane, 0],
            contact.halfplane_positions[plane, 1],
            contact.halfplane_positions[plane, 2],
            normal_x,
            normal_y,
            normal_z,
        )
        d_hat = pair_d_hat_ph(vertex.d_hats, contact.d_hat[()], vertex_id)
        xi = pair_thickness_ph(vertex.thicknesses, vertex_id)
        element = contact.vert_contact_element_ids[vertex_id]
        kappa = ct_query(contact.kappa_table, contact.n_contact_elements[()], element, 0)
        ids = qd.Vector([vertex_id, 0, 0, 0])
        kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, 1)
        mu = ct_query(contact.mu_table, contact.n_contact_elements[()], element, 0)
        normal_force = gipc_normal_force(lagged_distance * lagged_distance, d_hat, xi, kappa)
        gradient = qd.Vector.zero(qd.f64, 3)
        hessian = qd.Vector.zero(qd.f64, 9)
        halfplane_friction_grad_hess(
            current,
            lagged,
            normal_x,
            normal_y,
            normal_z,
            mu,
            normal_force,
            contact.friction_eps_v[()] * dt,
            gradient,
            hessian,
        )
        scale = contact.dt_sq[()] * surface.vert_area_weights[surface_vertex] * d_hat
        output_doublet = qd.atomic_add(contact.n_contact_doublets[()], 1)
        contact.contact_doublet_vertices[output_doublet] = vertex_id
        for axis in qd.static(range(3)):
            contact.contact_doublet_gradients[output_doublet, axis] = gradient[axis] * scale
        output_triplet = qd.atomic_add(contact.n_contact_triplets[()], 1)
        contact.contact_triplet_rows[output_triplet] = vertex_id
        contact.contact_triplet_cols[output_triplet] = vertex_id
        for row in qd.static(range(3)):
            for column in qd.static(range(3)):
                contact.contact_triplet_values[output_triplet, row, column] = hessian[row * 3 + column] * scale


@qd.func(requires_top_level=True)
def cipc_halfplane_friction_energy_kernel(
    contact: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    dt,
):
    for pair_index in range(contact.n_friction_pairs_ph[()]):
        surface_vertex = contact.friction_pairs_ph[pair_index, 0]
        plane = contact.friction_pairs_ph[pair_index, 1]
        vertex_id = surface.surf_verts[surface_vertex]
        current = qd.Vector(
            [
                vertex.positions[vertex_id, 0],
                vertex.positions[vertex_id, 1],
                vertex.positions[vertex_id, 2],
            ]
        )
        lagged = qd.Vector(
            [
                contact.lagged_positions[vertex_id, 0],
                contact.lagged_positions[vertex_id, 1],
                contact.lagged_positions[vertex_id, 2],
            ]
        )
        normal_x = contact.halfplane_normals[plane, 0]
        normal_y = contact.halfplane_normals[plane, 1]
        normal_z = contact.halfplane_normals[plane, 2]
        lagged_distance = halfplane_signed_distance(
            lagged[0],
            lagged[1],
            lagged[2],
            contact.halfplane_positions[plane, 0],
            contact.halfplane_positions[plane, 1],
            contact.halfplane_positions[plane, 2],
            normal_x,
            normal_y,
            normal_z,
        )
        d_hat = pair_d_hat_ph(vertex.d_hats, contact.d_hat[()], vertex_id)
        xi = pair_thickness_ph(vertex.thicknesses, vertex_id)
        element = contact.vert_contact_element_ids[vertex_id]
        kappa = ct_query(contact.kappa_table, contact.n_contact_elements[()], element, 0)
        ids = qd.Vector([vertex_id, 0, 0, 0])
        kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, 1)
        mu = ct_query(contact.mu_table, contact.n_contact_elements[()], element, 0)
        normal_force = gipc_normal_force(lagged_distance * lagged_distance, d_hat, xi, kappa)
        energy = halfplane_friction_energy(
            current,
            lagged,
            normal_x,
            normal_y,
            normal_z,
            mu,
            normal_force,
            contact.friction_eps_v[()] * dt,
        )
        scale = contact.dt_sq[()] * surface.vert_area_weights[surface_vertex] * d_hat
        qd.atomic_add(contact.friction_energy[()], energy * scale)


@qd.func(requires_top_level=True)
def cipc_filter_energy_pt_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_pt[()]):
        vert_index = contact.pairs_pt[pair_index, 0]
        face_index = contact.pairs_pt[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        positions = qd.Matrix.zero(qd.f64, 4, 3)
        ids[0] = surface.surf_verts[vert_index]
        for corner in qd.static(range(3)):
            ids[corner + 1] = surface.surf_triangles[face_index, corner]
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                positions[point, axis] = vertex.positions[ids[point], axis]
        flag = pt_distance_flag(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            positions[1, 0],
            positions[1, 1],
            positions[1, 2],
            positions[2, 0],
            positions[2, 1],
            positions[2, 2],
            positions[3, 0],
            positions[3, 1],
            positions[3, 2],
        )
        distance_squared = pt_flagged_distance2(flag, positions)
        d_hat = pair_d_hat_pt(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_pt(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        if distance_squared > xi * xi and distance_squared < (d_hat + xi) * (d_hat + xi) and _pt_enabled(contact, ids):
            count = _popcount4(flag)
            kappa = ct_query_pt(
                contact.kappa_table,
                contact.n_contact_elements[()],
                contact.vert_contact_element_ids[ids[0]],
                contact.vert_contact_element_ids[ids[1]],
                contact.vert_contact_element_ids[ids[2]],
                contact.vert_contact_element_ids[ids[3]],
            )
            kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, count)
            energy = gipc_barrier_energy(distance_squared, d_hat, xi, kappa)
            scale = contact.dt_sq[()] * cipc_pair_area_weight(
                surface.vert_area_weights[vert_index],
                surface.face_area_weights[face_index],
                d_hat,
            )
            qd.atomic_add(contact.barrier_energy[()], energy * scale)


@qd.func(requires_top_level=True)
def cipc_filter_energy_ee_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_ee[()]):
        edge_a = contact.pairs_ee[pair_index, 0]
        edge_b = contact.pairs_ee[pair_index, 1]
        ids = qd.Vector.zero(qd.i32, 4)
        positions = qd.Matrix.zero(qd.f64, 4, 3)
        ids[0] = surface.surf_edges[edge_a, 0]
        ids[1] = surface.surf_edges[edge_a, 1]
        ids[2] = surface.surf_edges[edge_b, 0]
        ids[3] = surface.surf_edges[edge_b, 1]
        for point in qd.static(range(4)):
            for axis in qd.static(range(3)):
                positions[point, axis] = vertex.positions[ids[point], axis]
        flag = ee_distance_flag(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            positions[1, 0],
            positions[1, 1],
            positions[1, 2],
            positions[2, 0],
            positions[2, 1],
            positions[2, 2],
            positions[3, 0],
            positions[3, 1],
            positions[3, 2],
        )
        distance_squared = ee_flagged_distance2(flag, positions)
        d_hat = pair_d_hat_ee(vertex.d_hats, contact.d_hat[()], ids[0], ids[1], ids[2], ids[3])
        xi = pair_thickness_ee(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3])
        if distance_squared > xi * xi and distance_squared < (d_hat + xi) * (d_hat + xi) and _ee_enabled(contact, ids):
            count = _popcount4(flag)
            kappa = ct_query_ee(
                contact.kappa_table,
                contact.n_contact_elements[()],
                contact.vert_contact_element_ids[ids[0]],
                contact.vert_contact_element_ids[ids[1]],
                contact.vert_contact_element_ids[ids[2]],
                contact.vert_contact_element_ids[ids[3]],
            )
            kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, count)
            edge_a_x = positions[1, 0] - positions[0, 0]
            edge_a_y = positions[1, 1] - positions[0, 1]
            edge_a_z = positions[1, 2] - positions[0, 2]
            edge_b_x = positions[3, 0] - positions[2, 0]
            edge_b_y = positions[3, 1] - positions[2, 1]
            edge_b_z = positions[3, 2] - positions[2, 2]
            cross_x = edge_a_y * edge_b_z - edge_a_z * edge_b_y
            cross_y = edge_a_z * edge_b_x - edge_a_x * edge_b_z
            cross_z = edge_a_x * edge_b_y - edge_a_y * edge_b_x
            I1 = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
            rest_a_x = vertex.x_bar[ids[1], 0] - vertex.x_bar[ids[0], 0]
            rest_a_y = vertex.x_bar[ids[1], 1] - vertex.x_bar[ids[0], 1]
            rest_a_z = vertex.x_bar[ids[1], 2] - vertex.x_bar[ids[0], 2]
            rest_b_x = vertex.x_bar[ids[3], 0] - vertex.x_bar[ids[2], 0]
            rest_b_y = vertex.x_bar[ids[3], 1] - vertex.x_bar[ids[2], 1]
            rest_b_z = vertex.x_bar[ids[3], 2] - vertex.x_bar[ids[2], 2]
            eps_x = (
                1.0e-3
                * (rest_a_x * rest_a_x + rest_a_y * rest_a_y + rest_a_z * rest_a_z)
                * (rest_b_x * rest_b_x + rest_b_y * rest_b_y + rest_b_z * rest_b_z)
            )
            energy = gipc_barrier_energy(distance_squared, d_hat, xi, kappa)
            if I1 < eps_x:
                mollified_distance_squared = distance_squared
                if count == 4:
                    interior_distance_squared = gipc_d_EE(
                        positions[0, 0],
                        positions[0, 1],
                        positions[0, 2],
                        positions[1, 0],
                        positions[1, 1],
                        positions[1, 2],
                        positions[2, 0],
                        positions[2, 1],
                        positions[2, 2],
                        positions[3, 0],
                        positions[3, 1],
                        positions[3, 2],
                    )
                    if interior_distance_squared > xi * xi and interior_distance_squared < (d_hat + xi) * (d_hat + xi):
                        mollified_distance_squared = interior_distance_squared
                energy = gipc_barrier_energy_mollified(
                    mollified_distance_squared,
                    d_hat,
                    xi,
                    kappa,
                    I1,
                    eps_x,
                )
            scale = contact.dt_sq[()] * cipc_pair_area_weight(
                surface.edge_area_weights[edge_a],
                surface.edge_area_weights[edge_b],
                d_hat,
            )
            qd.atomic_add(contact.barrier_energy[()], energy * scale)


@qd.func(requires_top_level=True)
def cipc_filter_energy_ph_kernel(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    for pair_index in range(contact.n_pairs_ph[()]):
        surface_vertex = contact.pairs_ph[pair_index, 0]
        plane = contact.pairs_ph[pair_index, 1]
        vertex_id = surface.surf_verts[surface_vertex]
        distance = halfplane_signed_distance(
            vertex.positions[vertex_id, 0],
            vertex.positions[vertex_id, 1],
            vertex.positions[vertex_id, 2],
            contact.halfplane_positions[plane, 0],
            contact.halfplane_positions[plane, 1],
            contact.halfplane_positions[plane, 2],
            contact.halfplane_normals[plane, 0],
            contact.halfplane_normals[plane, 1],
            contact.halfplane_normals[plane, 2],
        )
        d_hat = pair_d_hat_ph(vertex.d_hats, contact.d_hat[()], vertex_id)
        xi = pair_thickness_ph(vertex.thicknesses, vertex_id)
        if distance > xi and distance < d_hat + xi and _ph_enabled(contact, vertex_id):
            element = contact.vert_contact_element_ids[vertex_id]
            kappa = ct_query(contact.kappa_table, contact.n_contact_elements[()], element, 0)
            ids = qd.Vector([vertex_id, 0, 0, 0])
            kappa = kappa * pair_vert_kappa_scale(contact, vertex, ids, 1)
            energy = gipc_barrier_energy(distance * distance, d_hat, xi, kappa)
            scale = contact.dt_sq[()] * surface.vert_area_weights[surface_vertex] * d_hat
            qd.atomic_add(contact.barrier_energy[()], energy * scale)


@qd.func(requires_top_level=True)
def snapshot_lagged_positions(contact: qd.template(), vertex: qd.template()):
    for i_vertex in range(vertex.n_verts[()]):
        for axis in qd.static(range(3)):
            contact.lagged_positions[i_vertex, axis] = vertex.positions[i_vertex, axis]
    for _ in range(1):
        contact.n_friction_pairs_pt[()] = 0
        contact.n_friction_pairs_ee[()] = 0
        contact.n_friction_pairs_pe[()] = 0
        contact.n_friction_pairs_pp[()] = 0
        contact.n_friction_pairs_ph[()] = 0
        contact.n_friction_demand_doublets[()] = 0
        contact.n_friction_demand_triplets[()] = 0
        contact.friction_overflow_flag[()] = 0


@qd.func(requires_top_level=True)
def friction_pair_filter_pt(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    friction_pair_filter_pt_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def friction_pair_filter_ee(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    friction_pair_filter_ee_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def friction_pair_filter_ph(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    halfplane_friction_pair_filter_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def friction_snapshot(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    snapshot_lagged_positions(contact, vertex)
    friction_pair_filter_pt(contact, surface, vertex)
    friction_pair_filter_ee(contact, surface, vertex)
    if qd.static(contact.has_halfplanes):
        friction_pair_filter_ph(contact, surface, vertex)


@qd.func(requires_top_level=True)
def count_active_pt(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_count_active_pt_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def count_active_ee(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_count_active_ee_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def count_active_ph(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_count_active_ph_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def count_active(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    count_active_pt(contact, surface, vertex)
    count_active_ee(contact, surface, vertex)
    if qd.static(contact.has_halfplanes):
        count_active_ph(contact, surface, vertex)


@qd.func(requires_top_level=True)
def filter_assemble_pt(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_filter_assemble_pt_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def filter_assemble_ee(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_filter_assemble_ee_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def filter_assemble_ph(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_filter_assemble_ph_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def friction_assemble_pt(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_friction_assemble_pt_kernel(
        contact,
        surface,
        vertex,
        qd.sqrt(contact.dt_sq[()]),
    )


@qd.func(requires_top_level=True)
def friction_assemble_ee(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_friction_assemble_ee_kernel(
        contact,
        surface,
        vertex,
        qd.sqrt(contact.dt_sq[()]),
    )


@qd.func(requires_top_level=True)
def friction_assemble_ph(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_halfplane_friction_assemble_kernel(
        contact,
        surface,
        vertex,
        qd.sqrt(contact.dt_sq[()]),
    )


@qd.func(requires_top_level=True)
def filter_assemble(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    filter_assemble_pt(contact, surface, vertex)
    filter_assemble_ee(contact, surface, vertex)
    if qd.static(contact.has_friction):
        friction_assemble_pt(contact, surface, vertex)
        friction_assemble_ee(contact, surface, vertex)
    if qd.static(contact.has_halfplanes):
        filter_assemble_ph(contact, surface, vertex)
        if qd.static(contact.has_friction):
            friction_assemble_ph(contact, surface, vertex)


@qd.func(requires_top_level=True)
def filter_energy_pt(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_filter_energy_pt_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def filter_energy_ee(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_filter_energy_ee_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def filter_energy_ph(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_filter_energy_ph_kernel(contact, surface, vertex)


@qd.func(requires_top_level=True)
def friction_energy_pt(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_friction_energy_pt_kernel(
        contact,
        surface,
        vertex,
        qd.sqrt(contact.dt_sq[()]),
    )


@qd.func(requires_top_level=True)
def friction_energy_ee(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_friction_energy_ee_kernel(
        contact,
        surface,
        vertex,
        qd.sqrt(contact.dt_sq[()]),
    )


@qd.func(requires_top_level=True)
def friction_energy_ph(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    cipc_halfplane_friction_energy_kernel(
        contact,
        surface,
        vertex,
        qd.sqrt(contact.dt_sq[()]),
    )


@qd.func(requires_top_level=True)
def contact_energy(contact: qd.template(), surface: qd.template(), vertex: qd.template()):
    filter_energy_pt(contact, surface, vertex)
    filter_energy_ee(contact, surface, vertex)
    if qd.static(contact.has_friction):
        friction_energy_pt(contact, surface, vertex)
        friction_energy_ee(contact, surface, vertex)
    if qd.static(contact.has_halfplanes):
        filter_energy_ph(contact, surface, vertex)
        if qd.static(contact.has_friction):
            friction_energy_ph(contact, surface, vertex)


class ConsistentIPCContactConstitution(ContactConstitution):
    """Production Consistent IPC contact action provider."""

    def create_action_protocol(self) -> dict[str, object]:
        contact = self.contact_system.data
        surface = self.surface_system.data
        vertex = self.vertex_system.data
        protocol = {
            "snapshot_lagged_positions": (snapshot_lagged_positions, (contact, vertex)),
            "friction_pair_filter_pt": (friction_pair_filter_pt, (contact, surface, vertex)),
            "friction_pair_filter_ee": (friction_pair_filter_ee, (contact, surface, vertex)),
            "friction_pair_filter_ph": (friction_pair_filter_ph, (contact, surface, vertex)),
            "friction_snapshot": (friction_snapshot, (contact, surface, vertex)),
            "count_active_pt": (count_active_pt, (contact, surface, vertex)),
            "count_active_ee": (count_active_ee, (contact, surface, vertex)),
            "count_active_ph": (count_active_ph, (contact, surface, vertex)),
            "count_active": (count_active, (contact, surface, vertex)),
            "filter_assemble_pt": (filter_assemble_pt, (contact, surface, vertex)),
            "filter_assemble_ee": (filter_assemble_ee, (contact, surface, vertex)),
            "filter_assemble_ph": (filter_assemble_ph, (contact, surface, vertex)),
            "friction_assemble_pt": (friction_assemble_pt, (contact, surface, vertex)),
            "friction_assemble_ee": (friction_assemble_ee, (contact, surface, vertex)),
            "friction_assemble_ph": (friction_assemble_ph, (contact, surface, vertex)),
            "filter_assemble": (filter_assemble, (contact, surface, vertex)),
            "filter_energy_pt": (filter_energy_pt, (contact, surface, vertex)),
            "filter_energy_ee": (filter_energy_ee, (contact, surface, vertex)),
            "filter_energy_ph": (filter_energy_ph, (contact, surface, vertex)),
            "friction_energy_pt": (friction_energy_pt, (contact, surface, vertex)),
            "friction_energy_ee": (friction_energy_ee, (contact, surface, vertex)),
            "friction_energy_ph": (friction_energy_ph, (contact, surface, vertex)),
            "contact_energy": (contact_energy, (contact, surface, vertex)),
        }
        return {name: self.create_action(kernel, *action_data) for name, (kernel, action_data) in protocol.items()}
