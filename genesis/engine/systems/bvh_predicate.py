from __future__ import annotations

import quadrants as qd

from .contact_function.contact_table_query import ct_enabled_ee, ct_enabled_pt
from .global_body_manager import is_body_contact_ignored


@qd.func
def _node_pair_enabled(body_mgr: qd.template(), left_body, right_body):
    enabled = True
    if (
        left_body >= 0
        and right_body >= 0
        and (
            (left_body == right_body and body_mgr.self_collision[left_body] == 0)
            or is_body_contact_ignored(body_mgr, left_body, right_body)
        )
    ):
        enabled = False
    return enabled


@qd.func
def _ee_pair_enabled(
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    edge_a,
    edge_b,
):
    ea0 = surface.surf_edges[edge_a, 0]
    ea1 = surface.surf_edges[edge_a, 1]
    eb0 = surface.surf_edges[edge_b, 0]
    eb1 = surface.surf_edges[edge_b, 1]
    body_a = vertex.body_id[ea0]
    body_b = vertex.body_id[eb0]
    accept = qd.i32(1)
    if body_a == body_b and body_a >= 0 and body.self_collision[body_a] == 0:
        accept = 0
    if body_a >= 0 and body_b >= 0 and is_body_contact_ignored(body, body_a, body_b):
        accept = 0
    if ea0 == eb0 or ea0 == eb1 or ea1 == eb0 or ea1 == eb1:
        accept = 0
    if not ct_enabled_ee(
        contact.enable_ee_table,
        contact.n_contact_elements[()],
        contact.vert_contact_element_ids[ea0],
        contact.vert_contact_element_ids[ea1],
        contact.vert_contact_element_ids[eb0],
        contact.vert_contact_element_ids[eb1],
    ):
        accept = 0
    return accept


@qd.func
def _ee_emit_pair(
    emit,
    edge_a,
    edge_b,
    pairs: qd.template(),
    n_pairs: qd.template(),
    max_pairs,
    overflow_flag: qd.template(),
):
    lane = qd.simt.subgroup.invocation_id()
    mask = qd.simt.subgroup.ballot(qd.i32(emit))
    count = qd.i32(qd.math.popcnt(mask))
    base = qd.i32(0)
    if lane == 0 and count > 0:
        base = qd.atomic_add(n_pairs[()], count)
    base = qd.simt.subgroup.broadcast_first(base)
    if emit:
        lane_lt = (qd.u64(1) << qd.u64(lane)) - qd.u64(1)
        output = base + qd.i32(qd.math.popcnt(mask & lane_lt))
        if output < max_pairs:
            pairs[output, 0] = edge_a
            pairs[output, 1] = edge_b
        else:
            qd.atomic_or(overflow_flag[()], 1)


@qd.func
def _pt_pair_enabled(
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    vertex_index,
    face,
):
    triangle_a = surface.surf_triangles[face, 0]
    triangle_b = surface.surf_triangles[face, 1]
    triangle_c = surface.surf_triangles[face, 2]
    vertex_body = vertex.body_id[vertex_index]
    triangle_body = vertex.body_id[triangle_a]
    if vertex.body_id[triangle_b] != triangle_body or vertex.body_id[triangle_c] != triangle_body:
        triangle_body = -1

    accept = qd.i32(1)
    if vertex_body == triangle_body and vertex_body >= 0 and body.self_collision[vertex_body] == 0:
        accept = 0
    if (
        vertex_body >= 0
        and triangle_body >= 0
        and vertex_body != triangle_body
        and is_body_contact_ignored(body, vertex_body, triangle_body)
    ):
        accept = 0
    if vertex_index == triangle_a or vertex_index == triangle_b or vertex_index == triangle_c:
        accept = 0
    if not ct_enabled_pt(
        contact.enable_table,
        contact.n_contact_elements[()],
        contact.vert_contact_element_ids[vertex_index],
        contact.vert_contact_element_ids[triangle_a],
        contact.vert_contact_element_ids[triangle_b],
        contact.vert_contact_element_ids[triangle_c],
    ):
        accept = 0
    return accept


@qd.func
def _pt_emit_pair(
    emit,
    query,
    face,
    pairs: qd.template(),
    n_pairs: qd.template(),
    max_pairs,
    overflow_flag: qd.template(),
):
    lane = qd.simt.subgroup.invocation_id()
    mask = qd.simt.subgroup.ballot(qd.i32(emit))
    count = qd.i32(qd.math.popcnt(mask))
    base = qd.i32(0)
    if lane == 0 and count > 0:
        base = qd.atomic_add(n_pairs[()], count)
    base = qd.simt.subgroup.broadcast_first(base)
    if emit:
        lane_lt = (qd.u64(1) << qd.u64(lane)) - qd.u64(1)
        output = base + qd.i32(qd.math.popcnt(mask & lane_lt))
        if output < max_pairs:
            pairs[output, 0] = query
            pairs[output, 1] = face
        else:
            qd.atomic_or(overflow_flag[()], 1)
