"""Intersection test between an edge and a triangle, which detects the penetrations that IPC contact must never reach."""

import quadrants as qd


@qd.func
def func_point_in_triangle(
    point: qd.types.vector(3),
    t0: qd.types.vector(3),
    t1: qd.types.vector(3),
    t2: qd.types.vector(3),
    tolerance: float,
) -> bool:
    """Test whether a point of the plane of a triangle lies inside it, its barycentric coordinates within a tolerance.

    A triangle whose corner at t0 has a sine below 'tolerance' contains no point.
    """
    u = t1 - t0
    v = t2 - t0
    w = point - t0
    uu = u.dot(u)
    uv = u.dot(v)
    vv = v.dot(v)
    wu = w.dot(u)
    wv = w.dot(v)
    # -|u x v|^2, compared with uu vv so that the test holds at any scale
    denom = uv * uv - uu * vv
    is_inside = False
    if qd.abs(denom) >= tolerance * tolerance * uu * vv:
        s = (uv * wv - vv * wu) / denom
        t = (uv * wu - uu * wv) / denom
        is_inside = s >= -tolerance and t >= -tolerance and s + t <= 1.0 + tolerance
    return is_inside


@qd.func
def func_edge_triangle_intersect(
    t0: qd.types.vector(3),
    t1: qd.types.vector(3),
    t2: qd.types.vector(3),
    e0: qd.types.vector(3),
    e1: qd.types.vector(3),
    tolerance: float,
) -> bool:
    """Test whether a segment intersects a triangle, coplanar configurations included.

    'tolerance' is relative to the longest of the edge and the sides of the triangle, so that the test holds at any
    scale: it bounds the squared area of a degenerate triangle, which intersects nothing, the distance to the plane
    under which the edge counts as coplanar, and the barycentric coordinates of the inside tests. A coplanar edge
    intersects the triangle when one of its vertices lies inside it or it crosses one of its sides in the plane of the
    triangle, a side whose angle with the edge has a sine below 'tolerance' being parallel to it. Otherwise the point
    where the edge crosses the plane must lie inside the triangle.
    """
    u = t1 - t0
    v = t2 - t0
    w = t2 - t1
    edge = e1 - e0
    length2 = qd.max(qd.max(edge.dot(edge), u.dot(u)), qd.max(v.dot(v), w.dot(w)))
    normal = u.cross(v)
    normal_norm2 = normal.dot(normal)
    is_hit = False
    if normal_norm2 > tolerance * tolerance * length2 * length2:
        height0 = (e0 - t0).dot(normal)
        height1 = (e1 - t0).dot(normal)
        if not ((height0 > 0.0 and height1 > 0.0) or (height0 < 0.0 and height1 < 0.0)):
            coplanar_height = tolerance * qd.sqrt(length2 * normal_norm2)
            if qd.abs(height0) <= coplanar_height and qd.abs(height1) <= coplanar_height:
                is_hit = func_point_in_triangle(e0, t0, t1, t2, tolerance) or func_point_in_triangle(
                    e1, t0, t1, t2, tolerance
                )
                corners = (t0, t1, t2)
                for k in qd.static(range(3)):
                    side = corners[(k + 1) % 3] - corners[k]
                    # Cross products projected on the normal, so that the tilt of the edge out of the plane that the
                    # coplanar band admits does not enter: in 3D it dominates the cross product of an edge parallel to
                    # a side in the plane, whose crossing coordinates then are rounding noise
                    denom = edge.cross(side).dot(normal)
                    if (
                        not is_hit
                        and denom * denom >= tolerance * tolerance * edge.dot(edge) * side.dot(side) * normal_norm2
                    ):
                        offset = corners[k] - e0
                        coord_edge = offset.cross(side).dot(normal) / denom
                        coord_side = offset.cross(edge).dot(normal) / denom
                        is_hit = (
                            coord_edge >= -tolerance
                            and coord_edge <= 1.0 + tolerance
                            and coord_side >= -tolerance
                            and coord_side <= 1.0 + tolerance
                        )
            else:
                t = height0 / (height0 - height1)
                if t >= 0.0 and t <= 1.0:
                    is_hit = func_point_in_triangle(e0 + (e1 - e0) * t, t0, t1, t2, tolerance)
    return is_hit
