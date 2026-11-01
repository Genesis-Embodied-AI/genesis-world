import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.constants import GEOM_TYPE


@qd.func
def func_closest_points_on_segments(
    seg_a_p1: qd.types.vector(3),
    seg_a_p2: qd.types.vector(3),
    seg_b_p1: qd.types.vector(3),
    seg_b_p2: qd.types.vector(3),
    eps: float,
):
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

    if denom < eps:
        # Segments are parallel or one/both are degenerate
        s = 0.0
        if b_squared_len > eps:
            t = qd.math.clamp(e / b_squared_len, 0.0, 1.0)
        else:
            t = 0.0
    else:
        # General case: solve for optimal parameters
        s = (dot_product_dir * e - b_squared_len * d) / denom
        t = (a_squared_len * e - dot_product_dir * d) / denom

        s = qd.math.clamp(s, 0.0, 1.0)

        # Recompute t for clamped s
        t = qd.math.clamp((dot_product_dir * s + e) / b_squared_len if b_squared_len > eps else 0.0, 0.0, 1.0)

        # Recompute s for clamped t (ensures we're on segment boundaries)
        s_new = qd.math.clamp((dot_product_dir * t - d) / a_squared_len if a_squared_len > eps else 0.0, 0.0, 1.0)

        # Use refined s if it improves the solution
        if a_squared_len > eps:
            s = s_new

    seg_a_closest = seg_a_p1 + s * segment_a_dir
    seg_b_closest = seg_b_p1 + t * segment_b_dir

    return seg_a_closest, seg_b_closest


@qd.func
def func_det3(v1: qd.types.vector(3), v2: qd.types.vector(3), v3: qd.types.vector(3)):
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
def func_is_geom_boxes_overlap(
    i_ga: int,
    i_gb: int,
    i_b: int,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    """Check whether the world-aligned boxes of two geoms overlap, and for two convex geoms their oriented boxes too.

    The oriented box of a geom is its rest-pose box moved with it. It bounds a convex geom much tighter than the
    world-aligned box of a rotated geom, so the separating axis test of both oriented boxes drops most of the pairs that
    the world-aligned boxes keep without any contact. Planes and terrains collide beyond their box, and non-convex geoms
    go through their own detection, so their pairs keep the world-aligned test alone.
    """
    is_overlapping = not (
        (dyn_state.geoms.aabb_max[i_ga, i_b] <= dyn_state.geoms.aabb_min[i_gb, i_b]).any()
        or (dyn_state.geoms.aabb_min[i_ga, i_b] >= dyn_state.geoms.aabb_max[i_gb, i_b]).any()
    )
    # The pair comes by increasing geom type (see func_broad_phase_geoms), the plane and the terrain being the first and
    # the last ones
    if (
        is_overlapping
        and dyn_info.geoms.is_convex[i_ga]
        and dyn_info.geoms.is_convex[i_gb]
        and dyn_info.geoms.type[i_ga] != GEOM_TYPE.PLANE
        and dyn_info.geoms.type[i_gb] != GEOM_TYPE.TERRAIN
    ):
        EPS = rigid_info.EPS[None]
        half_a = 0.5 * (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0])
        half_b = 0.5 * (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0])
        center_a = 0.5 * (geoms_init_AABB[i_ga, 7] + geoms_init_AABB[i_ga, 0])
        center_b = 0.5 * (geoms_init_AABB[i_gb, 7] + geoms_init_AABB[i_gb, 0])

        # The axes of box b and the offset of its center, in the frame of box a
        quat_a = dyn_state.geoms.quat[i_ga, i_b]
        R_ab = gu.qd_quat_to_R(gu.qd_quat_mul(gu.qd_inv_quat(quat_a), dyn_state.geoms.quat[i_gb, i_b]), EPS)
        R_ab_abs = qd.abs(R_ab)
        pos_ab = gu.qd_inv_transform_by_quat(dyn_state.geoms.pos[i_gb, i_b] - dyn_state.geoms.pos[i_ga, i_b], quat_a)
        offset = pos_ab + R_ab @ center_b - center_a

        # The detections report a contact within their tolerance of touching, a small fraction of the pair scale (see
        # func_compute_geom_pair_scale), which this margin stays well above
        margin = 1e-3 * qd.min(half_a.norm(), half_b.norm())

        # The face axes of both boxes
        extent_a = R_ab_abs @ half_b + half_a
        extent_b = R_ab_abs.transpose() @ half_a + half_b
        is_overlapping = (qd.abs(offset) <= extent_a + margin).all() and (
            qd.abs(R_ab.transpose() @ offset) <= extent_b + margin
        ).all()

        # The axes crossing axis i_3 of box a with axis j_3 of box b, which lie in the plane of the other axes i_31 and
        # i_32 of box a and project box b on its other axes j_31 and j_32. They stay unnormalized: the margin grows
        # relative to both projections as the edges turn parallel, which only keeps more pairs.
        for i_3 in qd.static(range(3)):
            i_31, i_32 = qd.static((i_3 + 1) % 3, (i_3 + 2) % 3)
            for j_3 in qd.static(range(3)):
                j_31, j_32 = qd.static((j_3 + 1) % 3, (j_3 + 2) % 3)
                extent = (
                    half_a[i_31] * R_ab_abs[i_32, j_3]
                    + half_a[i_32] * R_ab_abs[i_31, j_3]
                    + half_b[j_31] * R_ab_abs[i_3, j_32]
                    + half_b[j_32] * R_ab_abs[i_3, j_31]
                )
                if qd.abs(offset[i_32] * R_ab[i_31, j_3] - offset[i_31] * R_ab[i_32, j_3]) > extent + margin:
                    is_overlapping = False
    return is_overlapping


@qd.func
def func_is_discrete_geom(i_g: int, dyn_info: array_class.DynInfo):
    """
    Check if the given geom is a discrete geometry.
    """
    geom_type = dyn_info.geoms.type[i_g]
    return geom_type == GEOM_TYPE.MESH or geom_type == GEOM_TYPE.BOX


@qd.func
def func_is_discrete_geoms(i_ga: int, i_gb: int, dyn_info: array_class.DynInfo):
    """
    Check if the given geoms are discrete geometries.
    """
    return func_is_discrete_geom(i_ga, dyn_info) and func_is_discrete_geom(i_gb, dyn_info)
