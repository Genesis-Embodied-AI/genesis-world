"""Conservative advancement for mixed linear and rigid-screw vertex paths."""

from __future__ import annotations

import quadrants as qd

from .directional_ccd import closest_ee, closest_pt


@qd.func
def _load_screw_vertex(vertex: qd.template(), vertex_id, path: qd.template(), meta: qd.template()):
    for axis in qd.static(range(3)):
        x0 = vertex.positions[vertex_id, axis]
        path[0, axis] = x0
        path[1, axis] = vertex.trajectory_end_positions[vertex_id, axis] - x0
        path[2, axis] = vertex.path_rot[vertex_id, axis]
        path[3, axis] = vertex.path_pivot[vertex_id, axis]
        path[4, axis] = vertex.path_pivot_disp[vertex_id, axis]
    theta = qd.sqrt(path[2, 0] * path[2, 0] + path[2, 1] * path[2, 1] + path[2, 2] * path[2, 2])
    lever = qd.f64(0.0)
    if theta > 0.0:
        dx = path[0, 0] - path[3, 0]
        dy = path[0, 1] - path[3, 1]
        dz = path[0, 2] - path[3, 2]
        lever = qd.sqrt(dx * dx + dy * dy + dz * dz)
    meta[0] = theta
    meta[1] = lever


@qd.func
def _screw_eval(path: qd.template(), meta: qd.template(), time):
    result = qd.Vector.zero(qd.f64, 3)
    if meta[0] == 0.0:
        for axis in qd.static(range(3)):
            result[axis] = path[0, axis] + time * path[1, axis]
    else:
        rotation = qd.Vector(
            [
                time * path[2, 0],
                time * path[2, 1],
                time * path[2, 2],
            ]
        )
        angle = rotation.norm()
        quaternion = qd.Vector([1.0, 0.0, 0.0, 0.0])
        if angle > 1.0e-12:
            half = 0.5 * angle
            scale = qd.sin(half) / angle
            quaternion = qd.Vector(
                [
                    qd.cos(half),
                    scale * rotation[0],
                    scale * rotation[1],
                    scale * rotation[2],
                ]
            )
        lever = qd.Vector(
            [
                path[0, 0] - path[3, 0],
                path[0, 1] - path[3, 1],
                path[0, 2] - path[3, 2],
            ]
        )
        qw = quaternion[0]
        qv = qd.Vector([quaternion[1], quaternion[2], quaternion[3]])
        rotated = lever + 2.0 * qv.cross(qv.cross(lever) + qw * lever)
        for axis in qd.static(range(3)):
            result[axis] = path[3, axis] + time * path[4, axis] + rotated[axis]
    return result


@qd.func
def _screw_remainder(path: qd.template(), current: qd.template()):
    return qd.Vector(
        [
            path[0, 0] + path[1, 0] - current[0],
            path[0, 1] + path[1, 1] - current[1],
            path[0, 2] + path[1, 2] - current[2],
        ]
    )


@qd.func
def _screw_curvature(meta: qd.template(), time):
    remainder_angle = (1.0 - time) * meta[0]
    return remainder_angle * remainder_angle * meta[1]


@qd.func
def _advance_fraction(gap, rate, curvature):
    result = qd.f64(2.0)
    if curvature <= 0.0:
        if rate > 0.0:
            result = gap / rate
    else:
        coefficient = rate + curvature
        if coefficient > 0.0:
            discriminant = coefficient * coefficient - 4.0 * curvature * gap
            if discriminant >= 0.0:
                result = 2.0 * gap / (coefficient + qd.sqrt(discriminant))
    return result


@qd.func
def screw_point_triangle_ccd(
    vertex: qd.template(),
    point_id,
    triangle_0_id,
    triangle_1_id,
    triangle_2_id,
    eta,
    thickness,
    max_iters: qd.template(),
    result: qd.template(),
):
    point_path = qd.Matrix.zero(qd.f64, 5, 3)
    triangle_0_path = qd.Matrix.zero(qd.f64, 5, 3)
    triangle_1_path = qd.Matrix.zero(qd.f64, 5, 3)
    triangle_2_path = qd.Matrix.zero(qd.f64, 5, 3)
    point_meta = qd.Vector.zero(qd.f64, 2)
    triangle_0_meta = qd.Vector.zero(qd.f64, 2)
    triangle_1_meta = qd.Vector.zero(qd.f64, 2)
    triangle_2_meta = qd.Vector.zero(qd.f64, 2)
    _load_screw_vertex(vertex, point_id, point_path, point_meta)
    _load_screw_vertex(vertex, triangle_0_id, triangle_0_path, triangle_0_meta)
    _load_screw_vertex(vertex, triangle_1_id, triangle_1_path, triangle_1_meta)
    _load_screw_vertex(vertex, triangle_2_id, triangle_2_path, triangle_2_meta)

    point = _screw_eval(point_path, point_meta, 0.0)
    triangle_0 = _screw_eval(triangle_0_path, triangle_0_meta, 0.0)
    triangle_1 = _screw_eval(triangle_1_path, triangle_1_meta, 0.0)
    triangle_2 = _screw_eval(triangle_2_path, triangle_2_meta, 0.0)
    closest = qd.Vector.zero(qd.f64, 4)
    closest_pt(
        point[0],
        point[1],
        point[2],
        triangle_0[0],
        triangle_0[1],
        triangle_0[2],
        triangle_1[0],
        triangle_1[1],
        triangle_1[2],
        triangle_2[0],
        triangle_2[1],
        triangle_2[2],
        closest,
    )
    distance = closest[3]
    initial_distance = distance
    time = qd.f64(0.0)
    done = qd.i32(distance <= thickness)
    iterations = qd.i32(0)
    while iterations < max_iters and done == 0:
        iterations = iterations + 1
        normal = qd.Vector(
            [
                (closest[0] - point[0]) / distance,
                (closest[1] - point[1]) / distance,
                (closest[2] - point[2]) / distance,
            ]
        )
        point_delta = _screw_remainder(point_path, point)
        triangle_0_delta = _screw_remainder(triangle_0_path, triangle_0)
        triangle_1_delta = _screw_remainder(triangle_1_path, triangle_1)
        triangle_2_delta = _screw_remainder(triangle_2_path, triangle_2)
        gap = qd.min(
            qd.min(normal.dot(triangle_0), normal.dot(triangle_1)),
            normal.dot(triangle_2),
        ) - normal.dot(point)
        rate = normal.dot(point_delta) + qd.max(
            qd.max(-normal.dot(triangle_0_delta), -normal.dot(triangle_1_delta)),
            -normal.dot(triangle_2_delta),
        )
        if gap <= thickness:
            gap = distance
            rate = point_delta.norm() + qd.sqrt(
                qd.max(
                    triangle_0_delta.dot(triangle_0_delta),
                    qd.max(
                        triangle_1_delta.dot(triangle_1_delta),
                        triangle_2_delta.dot(triangle_2_delta),
                    ),
                )
            )
        curvature = 0.5 * (
            _screw_curvature(point_meta, time)
            + _screw_curvature(triangle_0_meta, time)
            + _screw_curvature(triangle_1_meta, time)
            + _screw_curvature(triangle_2_meta, time)
        )
        if rate <= 0.0 and curvature <= 0.0:
            time = 1.0
            done = 1
        if done == 0:
            fraction = _advance_fraction(
                (1.0 - eta) * (gap - thickness),
                rate,
                curvature,
            )
            if time + fraction * (1.0 - time) >= 1.0:
                time = 1.0
                done = 1
            else:
                time = time + fraction * (1.0 - time)
                point = _screw_eval(point_path, point_meta, time)
                triangle_0 = _screw_eval(triangle_0_path, triangle_0_meta, time)
                triangle_1 = _screw_eval(triangle_1_path, triangle_1_meta, time)
                triangle_2 = _screw_eval(triangle_2_path, triangle_2_meta, time)
                closest_pt(
                    point[0],
                    point[1],
                    point[2],
                    triangle_0[0],
                    triangle_0[1],
                    triangle_0[2],
                    triangle_1[0],
                    triangle_1[1],
                    triangle_1[2],
                    triangle_2[0],
                    triangle_2[1],
                    triangle_2[2],
                    closest,
                )
                distance = closest[3]
                if distance - thickness < eta * (initial_distance - thickness):
                    done = 1
    result[0] = time


@qd.func
def screw_edge_edge_ccd(
    vertex: qd.template(),
    edge_a_0_id,
    edge_a_1_id,
    edge_b_0_id,
    edge_b_1_id,
    eta,
    thickness,
    max_iters: qd.template(),
    result: qd.template(),
):
    paths = qd.Matrix.zero(qd.f64, 20, 3)
    metadata = qd.Matrix.zero(qd.f64, 4, 2)
    path = qd.Matrix.zero(qd.f64, 5, 3)
    meta = qd.Vector.zero(qd.f64, 2)
    ids = qd.Vector([edge_a_0_id, edge_a_1_id, edge_b_0_id, edge_b_1_id])
    for vertex_slot in qd.static(range(4)):
        _load_screw_vertex(vertex, ids[vertex_slot], path, meta)
        for row in qd.static(range(5)):
            for axis in qd.static(range(3)):
                paths[vertex_slot * 5 + row, axis] = path[row, axis]
        for component in qd.static(range(2)):
            metadata[vertex_slot, component] = meta[component]

    current = qd.Matrix.zero(qd.f64, 4, 3)
    deltas = qd.Matrix.zero(qd.f64, 4, 3)
    for vertex_slot in qd.static(range(4)):
        for row in qd.static(range(5)):
            for axis in qd.static(range(3)):
                path[row, axis] = paths[vertex_slot * 5 + row, axis]
        for component in qd.static(range(2)):
            meta[component] = metadata[vertex_slot, component]
        value = _screw_eval(path, meta, 0.0)
        for axis in qd.static(range(3)):
            current[vertex_slot, axis] = value[axis]

    closest = qd.Vector.zero(qd.f64, 7)
    closest_ee(
        current[0, 0],
        current[0, 1],
        current[0, 2],
        current[1, 0],
        current[1, 1],
        current[1, 2],
        current[2, 0],
        current[2, 1],
        current[2, 2],
        current[3, 0],
        current[3, 1],
        current[3, 2],
        closest,
    )
    distance = closest[6]
    initial_distance = distance
    time = qd.f64(0.0)
    done = qd.i32(distance <= thickness)
    iterations = qd.i32(0)
    while iterations < max_iters and done == 0:
        iterations = iterations + 1
        normal = qd.Vector(
            [
                (closest[3] - closest[0]) / distance,
                (closest[4] - closest[1]) / distance,
                (closest[5] - closest[2]) / distance,
            ]
        )
        curvature = qd.f64(0.0)
        for vertex_slot in qd.static(range(4)):
            for row in qd.static(range(5)):
                for axis in qd.static(range(3)):
                    path[row, axis] = paths[vertex_slot * 5 + row, axis]
            for component in qd.static(range(2)):
                meta[component] = metadata[vertex_slot, component]
            point = qd.Vector(
                [
                    current[vertex_slot, 0],
                    current[vertex_slot, 1],
                    current[vertex_slot, 2],
                ]
            )
            delta = _screw_remainder(path, point)
            for axis in qd.static(range(3)):
                deltas[vertex_slot, axis] = delta[axis]
            curvature = curvature + _screw_curvature(meta, time)
        curvature = 0.5 * curvature
        edge_a_0 = qd.Vector([current[0, 0], current[0, 1], current[0, 2]])
        edge_a_1 = qd.Vector([current[1, 0], current[1, 1], current[1, 2]])
        edge_b_0 = qd.Vector([current[2, 0], current[2, 1], current[2, 2]])
        edge_b_1 = qd.Vector([current[3, 0], current[3, 1], current[3, 2]])
        delta_a_0 = qd.Vector([deltas[0, 0], deltas[0, 1], deltas[0, 2]])
        delta_a_1 = qd.Vector([deltas[1, 0], deltas[1, 1], deltas[1, 2]])
        delta_b_0 = qd.Vector([deltas[2, 0], deltas[2, 1], deltas[2, 2]])
        delta_b_1 = qd.Vector([deltas[3, 0], deltas[3, 1], deltas[3, 2]])
        gap = qd.min(normal.dot(edge_b_0), normal.dot(edge_b_1)) - qd.max(
            normal.dot(edge_a_0),
            normal.dot(edge_a_1),
        )
        rate = qd.max(normal.dot(delta_a_0), normal.dot(delta_a_1)) + qd.max(
            -normal.dot(delta_b_0),
            -normal.dot(delta_b_1),
        )
        if gap <= thickness:
            gap = distance
            rate = qd.sqrt(qd.max(delta_a_0.dot(delta_a_0), delta_a_1.dot(delta_a_1))) + qd.sqrt(
                qd.max(delta_b_0.dot(delta_b_0), delta_b_1.dot(delta_b_1))
            )
        if rate <= 0.0 and curvature <= 0.0:
            time = 1.0
            done = 1
        if done == 0:
            fraction = _advance_fraction(
                (1.0 - eta) * (gap - thickness),
                rate,
                curvature,
            )
            if time + fraction * (1.0 - time) >= 1.0:
                time = 1.0
                done = 1
            else:
                time = time + fraction * (1.0 - time)
                for vertex_slot in qd.static(range(4)):
                    for row in qd.static(range(5)):
                        for axis in qd.static(range(3)):
                            path[row, axis] = paths[vertex_slot * 5 + row, axis]
                    for component in qd.static(range(2)):
                        meta[component] = metadata[vertex_slot, component]
                    value = _screw_eval(path, meta, time)
                    for axis in qd.static(range(3)):
                        current[vertex_slot, axis] = value[axis]
                closest_ee(
                    current[0, 0],
                    current[0, 1],
                    current[0, 2],
                    current[1, 0],
                    current[1, 1],
                    current[1, 2],
                    current[2, 0],
                    current[2, 1],
                    current[2, 2],
                    current[3, 0],
                    current[3, 1],
                    current[3, 2],
                    closest,
                )
                distance = closest[6]
                if distance - thickness < eta * (initial_distance - thickness):
                    done = 1
    result[0] = time


@qd.func
def screw_halfplane_ccd(
    vertex: qd.template(),
    vertex_id,
    normal: qd.template(),
    plane_distance,
    eta,
    thickness,
    max_iters: qd.template(),
    result: qd.template(),
):
    path = qd.Matrix.zero(qd.f64, 5, 3)
    meta = qd.Vector.zero(qd.f64, 2)
    _load_screw_vertex(vertex, vertex_id, path, meta)
    current = _screw_eval(path, meta, 0.0)
    distance = normal.dot(current) - plane_distance
    initial_distance = distance
    time = qd.f64(0.0)
    done = qd.i32(distance <= thickness)
    iterations = qd.i32(0)
    while iterations < max_iters and done == 0:
        iterations = iterations + 1
        delta = _screw_remainder(path, current)
        rate = -normal.dot(delta)
        curvature = 0.5 * _screw_curvature(meta, time)
        if rate <= 0.0 and curvature <= 0.0:
            time = 1.0
            done = 1
        if done == 0:
            fraction = _advance_fraction(
                (1.0 - eta) * (distance - thickness),
                rate,
                curvature,
            )
            if time + fraction * (1.0 - time) >= 1.0:
                time = 1.0
                done = 1
            else:
                time = time + fraction * (1.0 - time)
                current = _screw_eval(path, meta, time)
                distance = normal.dot(current) - plane_distance
                if distance - thickness < eta * (initial_distance - thickness):
                    done = 1
    result[0] = time
