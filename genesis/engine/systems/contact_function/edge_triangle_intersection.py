from __future__ import annotations

import quadrants as qd


@qd.func
def point_in_triangle(
    point: qd.template(),  # qd.Vector
    a: qd.template(),  # qd.Vector
    b: qd.template(),  # qd.Vector
    c: qd.template(),  # qd.Vector
):
    u = b - a
    v = c - a
    w = point - a
    uu = u.dot(u)
    uv = u.dot(v)
    vv = v.dot(v)
    wu = w.dot(u)
    wv = w.dot(v)
    denominator = uv * uv - uu * vv
    inside = False
    if qd.abs(denominator) >= 1.0e-30:
        s = (uv * wv - vv * wu) / denominator
        t = (uv * wu - uu * wv) / denominator
        inside = s >= -1.0e-12 and t >= -1.0e-12 and s + t <= 1.0 + 1.0e-12
    return inside


@qd.func
def segment_segment_intersect(
    a: qd.template(),  # qd.Vector
    direction_a: qd.template(),  # qd.Vector
    b: qd.template(),  # qd.Vector
    direction_b: qd.template(),  # qd.Vector
):
    ab = b - a
    cross_directions = direction_a.cross(direction_b)
    denominator = cross_directions.dot(cross_directions)
    intersects = False
    if denominator >= 1.0e-30:
        u = ab.cross(direction_b).dot(cross_directions) / denominator
        v = ab.cross(direction_a).dot(cross_directions) / denominator
        intersects = u >= -1.0e-12 and u <= 1.0 + 1.0e-12 and v >= -1.0e-12 and v <= 1.0 + 1.0e-12
    return intersects


@qd.func
def triangle_edge_intersect(
    triangle_a: qd.template(),  # qd.Vector
    triangle_b: qd.template(),  # qd.Vector
    triangle_c: qd.template(),  # qd.Vector
    edge_a: qd.template(),  # qd.Vector
    edge_b: qd.template(),  # qd.Vector
):
    normal = (triangle_b - triangle_a).cross(triangle_c - triangle_a)
    distance_a = (edge_a - triangle_a).dot(normal)
    distance_b = (edge_b - triangle_a).dot(normal)
    intersects = False

    if not ((distance_a > 0.0 and distance_b > 0.0) or (distance_a < 0.0 and distance_b < 0.0)):
        if qd.abs(distance_a) < 1.0e-14 and qd.abs(distance_b) < 1.0e-14:
            intersects = (
                point_in_triangle(edge_a, triangle_a, triangle_b, triangle_c)
                or point_in_triangle(edge_b, triangle_a, triangle_b, triangle_c)
                or segment_segment_intersect(
                    edge_a,
                    edge_b - edge_a,
                    triangle_a,
                    triangle_b - triangle_a,
                )
                or segment_segment_intersect(
                    edge_a,
                    edge_b - edge_a,
                    triangle_b,
                    triangle_c - triangle_b,
                )
                or segment_segment_intersect(
                    edge_a,
                    edge_b - edge_a,
                    triangle_c,
                    triangle_a - triangle_c,
                )
            )
        else:
            interpolation = distance_a / (distance_a - distance_b)
            if interpolation >= 0.0 and interpolation <= 1.0:
                point = edge_a + interpolation * (edge_b - edge_a)
                intersects = point_in_triangle(
                    point,
                    triangle_a,
                    triangle_b,
                    triangle_c,
                )
    return intersects
