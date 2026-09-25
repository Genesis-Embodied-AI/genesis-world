import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.constants import GEOM_TYPE


@qd.func
def func_closest_points_on_segments(seg_a_p1, seg_a_p2, seg_b_p1, seg_b_p2, EPS):
    """
    Compute closest points on two line segments using analytical solution.

    References
    ----------
    Real-Time Collision Detection by Christer Ericson, Chapter 5.1.9
    """
    segment_a_dir = seg_a_p2 - seg_a_p1
    segment_b_dir = seg_b_p2 - seg_b_p1
    vec_between_segment_origins = seg_a_p1 - seg_b_p1

    a_squared_len = segment_a_dir.dot(segment_a_dir)
    dot_product_dir = segment_a_dir.dot(segment_b_dir)
    b_squared_len = segment_b_dir.dot(segment_b_dir)
    d = segment_a_dir.dot(vec_between_segment_origins)
    e = segment_b_dir.dot(vec_between_segment_origins)

    denom = a_squared_len * b_squared_len - dot_product_dir * dot_product_dir

    s = gs.qd_float(0.0)
    t = gs.qd_float(0.0)

    if denom < EPS:
        # Segments are parallel or one/both are degenerate
        s = 0.0
        if b_squared_len > EPS:
            t = qd.math.clamp(e / b_squared_len, 0.0, 1.0)
        else:
            t = 0.0
    else:
        # General case: solve for optimal parameters
        s = (dot_product_dir * e - b_squared_len * d) / denom
        t = (a_squared_len * e - dot_product_dir * d) / denom

        s = qd.math.clamp(s, 0.0, 1.0)

        # Recompute t for clamped s
        t = qd.math.clamp((dot_product_dir * s + e) / b_squared_len if b_squared_len > EPS else 0.0, 0.0, 1.0)

        # Recompute s for clamped t (ensures we're on segment boundaries)
        s_new = qd.math.clamp((dot_product_dir * t - d) / a_squared_len if a_squared_len > EPS else 0.0, 0.0, 1.0)

        # Use refined s if it improves the solution
        if a_squared_len > EPS:
            s = s_new

    seg_a_closest = seg_a_p1 + s * segment_a_dir
    seg_b_closest = seg_b_p1 + t * segment_b_dir

    return seg_a_closest, seg_b_closest


@qd.func
def func_det3(v1, v2, v3):
    """
    Compute the determinant of a 3x3 matrix M = [v1 | v2 | v3].
    """
    return (
        v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
        - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
        + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
    )


@qd.func
def func_point_in_geom_aabb(
    i_g: int, i_b: int, point: qd.types.vector(3), expansion: float, dyn_state: array_class.DynState
):
    aabb_min = dyn_state.geoms.aabb_min[i_g, i_b] - expansion
    aabb_max = dyn_state.geoms.aabb_max[i_g, i_b] + expansion
    return (point > aabb_min).all() and (point < aabb_max).all()


@qd.func
def func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, dyn_state: array_class.DynState):
    return not (
        (dyn_state.geoms.aabb_max[i_ga, i_b] <= dyn_state.geoms.aabb_min[i_gb, i_b]).any()
        or (dyn_state.geoms.aabb_min[i_ga, i_b] >= dyn_state.geoms.aabb_max[i_gb, i_b]).any()
    )


@qd.func
def func_is_discrete_geom(i_g, dyn_info: array_class.DynInfo):
    """
    Check if the given geom is a discrete geometry.
    """
    geom_type = dyn_info.geoms.type[i_g]
    return geom_type == GEOM_TYPE.MESH or geom_type == GEOM_TYPE.BOX


@qd.func
def func_is_discrete_geoms(i_ga, i_gb, dyn_info: array_class.DynInfo):
    """
    Check if the given geoms are discrete geometries.
    """
    return func_is_discrete_geom(i_ga, dyn_info) and func_is_discrete_geom(i_gb, dyn_info)


@qd.func
def func_is_equal_vec(a, b, eps):
    """
    Check if two vectors are equal within a small tolerance.
    """
    return (qd.abs(a - b) < eps).all()
