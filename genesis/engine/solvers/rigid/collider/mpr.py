import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import support_field
from .constants import PORTAL_STATUS
from .contact import func_compute_geom_pair_scale


class MPR:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver

        self._mpr_info = array_class.get_mpr_info(
            # Dimensionless, every site scaling it by the magnitudes of its own operands so the degeneracy tests
            # ignore the scene scale; it bounds when a triangle, a segment or a portal counts as degenerate, and the
            # support lookup scales it by its own factor for tie width.
            CCD_EPS=1e-5 if gs.qd_float == qd.f32 else 1e-10,
            # A fraction of the geom pair scale, turned into a length per pair: how close the portal must come to
            # the surface to count as having reached it, and the magnitude the ray is re-seeded with when the two
            # geoms' centres coincide.
            CCD_TOLERANCE=1e-6 if rigid_solver._enable_mujoco_compatibility else 1e-5,
            # Bounds that refinement, which is not otherwise guaranteed to terminate.
            CCD_ITERATIONS=50,
        )
        self.mpr_state = array_class.get_mpr_state(self._solver._B)


@qd.kernel
def clear(mpr_state: qd.template()):
    mpr_state.simplex_size.fill(0)


@qd.func
def mpr_swap(i_ga, i_gb, i_b, j, mpr_state: array_class.MPRState, i):
    mpr_state.simplex_support.v1[i, i_b], mpr_state.simplex_support.v1[j, i_b] = (
        mpr_state.simplex_support.v1[j, i_b],
        mpr_state.simplex_support.v1[i, i_b],
    )
    mpr_state.simplex_support.v2[i, i_b], mpr_state.simplex_support.v2[j, i_b] = (
        mpr_state.simplex_support.v2[j, i_b],
        mpr_state.simplex_support.v2[i, i_b],
    )
    mpr_state.simplex_support.v[i, i_b], mpr_state.simplex_support.v[j, i_b] = (
        mpr_state.simplex_support.v[j, i_b],
        mpr_state.simplex_support.v[i, i_b],
    )


@qd.func
def mpr_point_segment_dist2(P, A, B, collider_info: array_class.ColliderInfo):
    AB = B - A
    AP = P - A
    AB_AB = AB.dot(AB)
    AP_AB = AP.dot(AB)
    t = AP_AB / AB_AB
    if t < collider_info.mpr.CCD_EPS[None]:
        t = gs.qd_float(0.0)
    elif t > 1.0 - collider_info.mpr.CCD_EPS[None]:
        t = gs.qd_float(1.0)
    Q = A + AB * t

    return (P - Q).norm_sqr(), Q


@qd.func
def mpr_point_tri_depth(P, x0, B, C, collider_info: array_class.ColliderInfo):
    d1 = B - x0
    d2 = C - x0
    a = x0 - P
    v = d1.dot(d1)
    w = d2.dot(d2)
    p = a.dot(d1)
    q = a.dot(d2)
    r = d1.dot(d2)

    d = w * v - r * r
    dist = s = t = gs.qd_float(0.0)
    pdir = gs.qd_vec3([0.0, 0.0, 0.0])
    # d = |d1|^2 * |d2|^2 - (d1 . d2)^2 = |d1 x d2|^2 is the Gram determinant of the triangle edges (length^4);
    # comparing it to its own scale v * w makes the degeneracy test the dimensionless sin^2 of the edge angle.
    # Must use <= so that degenerate (zero-length) edges are still classified as degenerate.
    if qd.abs(d) <= collider_info.mpr.CCD_EPS[None] * v * w:
        s = t = -1.0
    else:
        s = (q * r - w * p) / d
        t = (-s * r - q) / w

    if (
        (s > -collider_info.mpr.CCD_EPS[None])
        and (s < 1.0 + collider_info.mpr.CCD_EPS[None])
        and (t > -collider_info.mpr.CCD_EPS[None])
        and (t < 1.0 + collider_info.mpr.CCD_EPS[None])
        and (t + s < 1.0 + collider_info.mpr.CCD_EPS[None])
    ):
        pdir = x0 + d1 * s + d2 * t
        dist = (P - pdir).norm_sqr()
    else:
        dist, pdir = mpr_point_segment_dist2(P, x0, B, collider_info)
        dist2, pdir2 = mpr_point_segment_dist2(P, x0, C, collider_info)
        if dist2 < dist:
            dist = dist2
            pdir = pdir2

        dist2, pdir2 = mpr_point_segment_dist2(P, B, C, collider_info)
        if dist2 < dist:
            dist = dist2
            pdir = pdir2

    return qd.sqrt(dist), pdir


@qd.func
def mpr_portal_dir(i_ga, i_gb, i_b, mpr_state: array_class.MPRState):
    v2v1 = mpr_state.simplex_support.v[2, i_b] - mpr_state.simplex_support.v[1, i_b]
    v3v1 = mpr_state.simplex_support.v[3, i_b] - mpr_state.simplex_support.v[1, i_b]
    direction = v2v1.cross(v3v1).normalized()
    return direction


@qd.func
def mpr_portal_encapsules_origin(
    i_ga, i_gb, i_b, direction, mpr_state: array_class.MPRState, collider_info: array_class.ColliderInfo
):
    # Pure sign test: at the boundary both outcomes are equally valid and the result is value-continuous, so any
    # epsilon would only shift the decision without protecting anything.
    dot = mpr_state.simplex_support.v[1, i_b].dot(direction)
    return dot > 0.0


@qd.func
def mpr_portal_can_encapsule_origin(v, direction, collider_info: array_class.ColliderInfo):
    # Pure sign test (see mpr_portal_encapsules_origin).
    dot = v.dot(direction)
    return dot > 0.0


@qd.func
def mpr_portal_reach_tolerance(
    i_ga, i_gb, i_b, ccd_tol, v, direction, mpr_state: array_class.MPRState, collider_info: array_class.ColliderInfo
):
    pv1 = mpr_state.simplex_support.v[1, i_b]
    pv2 = mpr_state.simplex_support.v[2, i_b]
    pv3 = mpr_state.simplex_support.v[3, i_b]
    dv = v.dot(direction)
    dot1 = qd.min(dv - pv1.dot(direction), dv - pv2.dot(direction), dv - pv3.dot(direction))
    # A portal plane through the origin makes dv and the three pv projections pure rounding noise.
    # The rounding error of a dot product scales with its operands, so the tolerance floor scales with the vertices.
    # Without the floor a noise gap counts as progress, the portal re-adds a held vertex, and refinement never ends.
    scale = qd.max(qd.abs(v).sum(), qd.abs(pv1).sum(), qd.abs(pv2).sum(), qd.abs(pv3).sum())
    return dot1 < ccd_tol + collider_info.mpr.CCD_EPS[None] * scale


@qd.func
def support_driver(
    i_g,
    i_b,
    direction,
    pos: qd.types.vector(3),
    quat: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    v = qd.Vector.zero(gs.qd_float, 3)
    geom_type = dyn_info.geoms.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE:
        v, v_, vid = support_field._func_support_sphere(i_g, direction, pos, quat, shrink=False, dyn_info=dyn_info)
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        v = support_field._func_support_ellipsoid(i_g, direction, pos, quat, dyn_info)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        v = support_field._func_support_capsule(i_g, direction, pos, quat, shrink=False, dyn_info=dyn_info)
    elif geom_type == gs.GEOM_TYPE.CYLINDER:
        v = support_field._func_support_cylinder(i_g, direction, pos, quat, shrink=False, dyn_info=dyn_info)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, v_, vid = support_field._func_support_box(i_g, direction, pos, quat, dyn_info)
    elif geom_type == gs.GEOM_TYPE.TERRAIN:
        if qd.static(collider_static_config.has_terrain):
            # Terrain support doesn't depend on geometry pos/quat - uses collider_state.prism
            # Terrain is global and not perturbed, so we use the global state directly
            v, _ = support_field._func_support_prism(i_b, direction, collider_state)
    else:
        # The reference engine's fallback pipeline scans mesh vertices exhaustively for its supports; the sampled
        # support table it replaces may return a different vertex of a tied flat face (see _func_support_world).
        v, v_, vid = support_field._func_support_world(
            i_g,
            direction,
            pos,
            quat,
            dyn_info,
            collider_info,
            exhaustive=qd.static(rigid_config.enable_mujoco_compatibility),
        )

    return v


@qd.func
def compute_support(
    i_ga,
    i_gb,
    i_b,
    direction,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    v1 = support_driver(
        i_ga,
        i_b,
        direction,
        pos_a,
        quat_a,
        collider_state,
        dyn_info,
        collider_info,
        rigid_config,
        collider_static_config,
    )
    v2 = support_driver(
        i_gb,
        i_b,
        -direction,
        pos_b,
        quat_b,
        collider_state,
        dyn_info,
        collider_info,
        rigid_config,
        collider_static_config,
    )

    v = v1 - v2
    return v, v1, v2


@qd.func
def func_geom_support(i_g, direction, pos: qd.types.vector(3), quat: qd.types.vector(4), dyn_info: array_class.DynInfo):
    direction_in_init_frame = gu.qd_inv_transform_by_quat(direction, quat)

    dot_max = gs.qd_float(-1e10)
    v = qd.Vector.zero(gs.qd_float, 3)
    vid = 0

    for i_v in range(dyn_info.geoms.vert_start[i_g], dyn_info.geoms.vert_end[i_g]):
        pos_local = dyn_info.verts.init_pos[i_v]
        dot = pos_local.dot(direction_in_init_frame)
        if dot > dot_max:
            v = pos_local
            dot_max = dot
            vid = i_v
    v_world = gu.qd_transform_by_trans_quat(v, pos, quat)

    return v_world, vid


@qd.func
def mpr_refine_portal(
    i_ga,
    i_gb,
    i_b,
    ccd_tol,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    ret = 1
    while True:
        direction = mpr_portal_dir(i_ga, i_gb, i_b, mpr_state)

        if mpr_portal_encapsules_origin(i_ga, i_gb, i_b, direction, mpr_state, collider_info):
            ret = 0
            break

        v, v1, v2 = compute_support(
            i_ga,
            i_gb,
            i_b,
            direction,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            collider_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )

        if not mpr_portal_can_encapsule_origin(v, direction, collider_info) or mpr_portal_reach_tolerance(
            i_ga, i_gb, i_b, ccd_tol, v, direction, mpr_state, collider_info
        ):
            ret = -1
            break

        mpr_expand_portal(i_ga, i_gb, i_b, v, v1, v2, mpr_state)
    return ret


@qd.func
def mpr_find_pos(
    i_ga,
    i_gb,
    i_b,
    mpr_state: array_class.MPRState,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
):
    b = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=gs.qd_float)

    # Only look into the direction of the portal for consistency with penetration depth computation
    if qd.static(rigid_config.enable_mujoco_compatibility):
        for i in qd.static(range(4)):
            i1, i2, i3 = (i % 2) + 1, (i + 2) % 4, 3 * ((i + 1) % 2)
            vec = mpr_state.simplex_support.v[i1, i_b].cross(mpr_state.simplex_support.v[i2, i_b])
            b[i] = vec.dot(mpr_state.simplex_support.v[i3, i_b]) * (1 - 2 * (((i + 1) // 2) % 2))

    sum_ = b.sum()

    # Must use <= so that the all-zero weights of the non-mujoco-compatible path (which skips the tetrahedron
    # volumes above) still enter the portal-direction fallback.
    if sum_ <= collider_info.mpr.CCD_EPS[None] * qd.abs(b).sum():
        direction = mpr_portal_dir(i_ga, i_gb, i_b, mpr_state)
        b[0] = 0.0
        for i in qd.static(range(1, 4)):
            i1, i2 = i % 3 + 1, (i + 1) % 3 + 1
            vec = mpr_state.simplex_support.v[i1, i_b].cross(mpr_state.simplex_support.v[i2, i_b])
            b[i] = vec.dot(direction)
        sum_ = b.sum()

    # A portal spanning no volume along its own direction leaves every weight at zero, and the position is a ratio of
    # them: weighting the three portal vertices equally puts the contact at their centroid, which lies on the portal
    # and keeps the ratio finite.
    if sum_ <= collider_info.mpr.CCD_EPS[None] * qd.abs(b).sum():
        b[0] = 0.0
        for i in qd.static(range(1, 4)):
            b[i] = 1.0
        sum_ = 3.0

    p1 = gs.qd_vec3([0.0, 0.0, 0.0])
    p2 = gs.qd_vec3([0.0, 0.0, 0.0])
    for i in qd.static(range(4)):
        p1 = p1 + b[i] * mpr_state.simplex_support.v1[i, i_b]
        p2 = p2 + b[i] * mpr_state.simplex_support.v2[i, i_b]

    return (0.5 / sum_) * (p1 + p2)


@qd.func
def mpr_find_penetr_touch(i_ga, i_gb, i_b, mpr_state: array_class.MPRState):
    is_col = True
    penetration = gs.qd_float(0.0)
    normal = -mpr_state.simplex_support.v[0, i_b].normalized()
    pos = (mpr_state.simplex_support.v1[1, i_b] + mpr_state.simplex_support.v2[1, i_b]) * 0.5
    return is_col, normal, penetration, pos


@qd.func
def mpr_find_penetr_segment(i_ga, i_gb, i_b, mpr_state: array_class.MPRState):
    is_col = True
    # Anchor the contact direction to the ray (v1 - v0) rather than the raw support point: a degenerate flat-face
    # support tie can leave v1 with an arbitrary lateral offset, making its direction noise while the ray stays
    # meaningful. Fall back to v1 if both coincide (fully degenerate difference).
    direction = mpr_state.simplex_support.v[1, i_b] - mpr_state.simplex_support.v[0, i_b]
    if direction.norm_sqr() == 0.0:
        direction = mpr_state.simplex_support.v[1, i_b]
    direction = direction.normalized()
    penetration = mpr_state.simplex_support.v[1, i_b].dot(direction)
    normal = -direction
    pos = (mpr_state.simplex_support.v1[1, i_b] + mpr_state.simplex_support.v2[1, i_b]) * 0.5

    return is_col, normal, penetration, pos


@qd.func
def mpr_find_penetration(
    i_ga,
    i_gb,
    i_b,
    ccd_tol,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    # How far the origin's projection may extrapolate beyond the portal triangle, as a fraction of the triangle
    # (barycentric), before the infinite-plane penetration is deemed an unreliable extrapolation (portal INVALID ->
    # refine with GJK). FIXME: This is a compile-time constant instead of an MPRInfo scalar field because one extra
    # field read pushes '_func_narrowphase_multicontact' past Metal's limit of 31 buffer bindings per kernel. Move it
    # back to MPRInfo once quadrants packs root buffers below that limit (e.g. via Metal argument buffers).
    CCD_EXTRAPOLATION_TOL = qd.static(1.0)

    iterations = 0

    is_col = False
    pos = gs.qd_vec3([0.0, 0.0, 0.0])
    normal = gs.qd_vec3([0.0, 0.0, 0.0])
    penetration = gs.qd_float(0.0)

    while True:
        direction = mpr_portal_dir(i_ga, i_gb, i_b, mpr_state)
        v, v1, v2 = compute_support(
            i_ga,
            i_gb,
            i_b,
            direction,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            collider_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )
        reached = mpr_portal_reach_tolerance(i_ga, i_gb, i_b, ccd_tol, v, direction, mpr_state, collider_info)
        if reached or iterations > collider_info.mpr.CCD_ITERATIONS[None]:
            # The contact point is defined as the projection of the origin onto the portal, i.e. the closest point to
            # the origin that lies inside the portal. Let's consider the portal as an infinite plane rather than a face
            # triangle. This makes sense because the projection of the origin must be strictly included into the portal
            # triangle for it to correspond to the true penetration depth. For reference about this property, see
            # 'Collision Handling with Variable-Step Integrators' Theorem 4.2:
            # https://modiasim.github.io/Modia3D.jl/resources/documentation/CollisionHandling_Neumayr_Otter_2017.pdf
            #
            # In theory, the center should have been shifted to end up with the one and only portal satisfying this
            # condition. However, a naive implementation of this process must be avoided because it would be very
            # costly. In practice, assuming the portal is infinite provides a decent approximation of the true
            # penetration depth (it is actually a lower-bound estimate according to Theorem 4.3) and normal without
            # requiring any additional computations.
            # See: https://github.com/danfis/libccd/issues/71#issuecomment-660415008
            #
            # An improved version of MPR has been proposed to find the right portal in an efficient way.
            # See: https://arxiv.org/pdf/2304.07357
            # Implementation: https://github.com/weigao95/mind-fcl/blob/main/include/fcl/cvx_collide/mpr.h
            #
            # The original paper introducing MPR algorithm is available here:
            # https://archive.org/details/game-programming-gems-7
            if qd.static(rigid_config.enable_mujoco_compatibility):
                penetration, pdir = mpr_point_tri_depth(
                    gs.qd_vec3([0.0, 0.0, 0.0]),
                    mpr_state.simplex_support.v[1, i_b],
                    mpr_state.simplex_support.v[2, i_b],
                    mpr_state.simplex_support.v[3, i_b],
                    collider_info,
                )
                normal = -pdir.normalized()
            else:
                penetration = direction.dot(mpr_state.simplex_support.v[1, i_b])
                normal = -direction

            # Classify what the depth is worth by how far the origin's projection extrapolates beyond the portal
            # triangle. b1,b2,b3 are the (unnormalized) barycentric coordinates of that projection; all >= 0 means the
            # origin projects inside, so the depth is exact (Thm 4.2 -> EXACT). Outside, -min(b)/sum is the
            # extrapolation as a fraction of the triangle: a small overshoot leaves the depth a valid lower bound
            # (Thm 4.3 -> LOWER_BOUND), a large one makes the infinite-plane depth an unreliable extrapolation
            # (-> EXTRAPOLATED). The extrapolation is the criterion; the triangle's own shape matters only through it.
            pv1 = mpr_state.simplex_support.v[1, i_b]
            pv2 = mpr_state.simplex_support.v[2, i_b]
            pv3 = mpr_state.simplex_support.v[3, i_b]
            b1 = pv2.cross(pv3).dot(direction)
            b2 = pv3.cross(pv1).dot(direction)
            b3 = pv1.cross(pv2).dot(direction)
            bsum = b1 + b2 + b3
            babs = qd.abs(b1) + qd.abs(b2) + qd.abs(b3)
            min_b = qd.min(b1, qd.min(b2, b3))
            if not reached:
                mpr_state.portal_status[i_b] = PORTAL_STATUS.UNCONVERGED
            elif bsum <= collider_info.mpr.CCD_EPS[None] * babs:
                # A portal spanning no area leaves every coordinate at zero, which satisfies min_b >= 0: the
                # degeneracy test must come first or a portal whose depth means nothing reads as exact.
                mpr_state.portal_status[i_b] = PORTAL_STATUS.EXTRAPOLATED
            elif min_b >= 0.0:
                mpr_state.portal_status[i_b] = PORTAL_STATUS.EXACT
            elif (-min_b) <= CCD_EXTRAPOLATION_TOL * bsum:
                mpr_state.portal_status[i_b] = PORTAL_STATUS.LOWER_BOUND
            else:
                mpr_state.portal_status[i_b] = PORTAL_STATUS.EXTRAPOLATED

            is_col = True
            pos = mpr_find_pos(i_ga, i_gb, i_b, mpr_state, collider_info, rigid_config)
            break

        mpr_expand_portal(i_ga, i_gb, i_b, v, v1, v2, mpr_state)
        iterations += 1

    return is_col, normal, penetration, pos


@qd.func
def mpr_expand_portal(i_ga, i_gb, i_b, v, v1, v2, mpr_state: array_class.MPRState):
    v4v0 = v.cross(mpr_state.simplex_support.v[0, i_b])
    dot = mpr_state.simplex_support.v[1, i_b].dot(v4v0)

    i_s = gs.qd_int(0)
    if dot > 0:
        dot = mpr_state.simplex_support.v[2, i_b].dot(v4v0)
        i_s = 1 if dot > 0 else 3

    else:
        dot = mpr_state.simplex_support.v[3, i_b].dot(v4v0)
        i_s = 2 if dot > 0 else 1

    mpr_state.simplex_support.v1[i_s, i_b] = v1
    mpr_state.simplex_support.v2[i_s, i_b] = v2
    mpr_state.simplex_support.v[i_s, i_b] = v


@qd.func
def mpr_discover_portal(
    i_ga,
    i_gb,
    i_b,
    ccd_tol,
    center_a,
    center_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    mpr_state.simplex_support.v1[0, i_b] = center_a
    mpr_state.simplex_support.v2[0, i_b] = center_b
    mpr_state.simplex_support.v[0, i_b] = center_a - center_b
    mpr_state.simplex_size[i_b] = 1

    # Coincident centers (within the solver's absolute length resolution) leave the ray direction undefined; probe
    # the three axes and pick the one with the largest Minkowski support extent, which gives portal discovery the
    # most room to enclose the origin.
    if (qd.abs(mpr_state.simplex_support.v[0, i_b]) < ccd_tol).all():
        best_extent = gs.qd_float(0.0)
        best_dir = qd.Vector.zero(gs.qd_float, 3)
        for i_axis in range(3):
            probe = qd.Vector.zero(gs.qd_float, 3)
            probe[i_axis] = 1.0
            probe_v, probe_v1, probe_v2 = compute_support(
                i_ga,
                i_gb,
                i_b,
                probe,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                collider_state,
                dyn_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )
            extent = probe_v.dot(probe)
            if i_axis == 0 or extent > best_extent:
                best_extent = extent
                best_dir = probe
        mpr_state.simplex_support.v[0, i_b] = best_dir * (10.0 * ccd_tol)

    direction = -mpr_state.simplex_support.v[0, i_b].normalized()

    v, v1, v2 = compute_support(
        i_ga,
        i_gb,
        i_b,
        direction,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        collider_state,
        dyn_info,
        collider_info,
        rigid_config,
        collider_static_config,
    )

    mpr_state.simplex_support.v1[1, i_b] = v1
    mpr_state.simplex_support.v2[1, i_b] = v2
    mpr_state.simplex_support.v[1, i_b] = v
    mpr_state.simplex_size[i_b] = 2

    dot = v.dot(direction)

    ret = 0
    if dot < 0.0:
        ret = -1
    else:
        direction = mpr_state.simplex_support.v[0, i_b].cross(mpr_state.simplex_support.v[1, i_b])
        # The touch test must come first and independently: for a vanishing v1 both sides of the relative
        # collinearity test vanish as well, which would fall through to normalizing a zero cross product.
        if (
            qd.abs(mpr_state.simplex_support.v[1, i_b])
            < collider_info.mpr.CCD_EPS[None] * mpr_state.simplex_support.v[0, i_b].norm()
        ).all():
            ret = 1
        # Relative collinearity test: |v0 x v1|^2 = |v0|^2 * |v1|^2 * sin^2(angle). An absolute epsilon (length^4)
        # would misclassify non-collinear cm-scale pairs as degenerate, fabricating deep contacts via the segment
        # path. Must use <= so that an exactly zero cross product is still classified as degenerate.
        elif (
            direction.dot(direction)
            <= collider_info.mpr.CCD_EPS[None]
            * mpr_state.simplex_support.v[0, i_b].norm_sqr()
            * mpr_state.simplex_support.v[1, i_b].norm_sqr()
        ):
            ret = 2
        else:
            direction = direction.normalized()
            v, v1, v2 = compute_support(
                i_ga,
                i_gb,
                i_b,
                direction,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                collider_state,
                dyn_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )
            dot = v.dot(direction)
            if dot < 0.0:
                ret = -1
            else:
                mpr_state.simplex_support.v1[2, i_b] = v1
                mpr_state.simplex_support.v2[2, i_b] = v2
                mpr_state.simplex_support.v[2, i_b] = v
                mpr_state.simplex_size[i_b] = 3

                va = mpr_state.simplex_support.v[1, i_b] - mpr_state.simplex_support.v[0, i_b]
                vb = mpr_state.simplex_support.v[2, i_b] - mpr_state.simplex_support.v[0, i_b]
                direction = va.cross(vb)
                direction = direction.normalized()

                dot = direction.dot(mpr_state.simplex_support.v[0, i_b])
                if dot > 0:
                    mpr_swap(i_ga, i_gb, i_b, 2, mpr_state, 1)
                    direction = -direction

                # FIXME: This algorithm may get stuck in an infinite loop if the actual penetration is smaller than
                # CCD_EPS and at least one geometry's center is outside its convex hull. Since this deadlock happens
                # very rarely, a simple fix is to abort computation after a few trials.
                num_trials = gs.qd_int(0)
                while mpr_state.simplex_size[i_b] < 4:
                    v, v1, v2 = compute_support(
                        i_ga,
                        i_gb,
                        i_b,
                        direction,
                        pos_a,
                        quat_a,
                        pos_b,
                        quat_b,
                        collider_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                    dot = v.dot(direction)
                    if dot < 0.0:
                        ret = -1
                        break

                    cont = False

                    va = mpr_state.simplex_support.v[1, i_b].cross(v)
                    dot = va.dot(mpr_state.simplex_support.v[0, i_b])
                    if dot < 0.0:
                        mpr_state.simplex_support.v1[2, i_b] = v1
                        mpr_state.simplex_support.v2[2, i_b] = v2
                        mpr_state.simplex_support.v[2, i_b] = v
                        cont = True

                    if not cont:
                        va = v.cross(mpr_state.simplex_support.v[2, i_b])
                        dot = va.dot(mpr_state.simplex_support.v[0, i_b])
                        if dot < 0.0:
                            mpr_state.simplex_support.v1[1, i_b] = v1
                            mpr_state.simplex_support.v2[1, i_b] = v2
                            mpr_state.simplex_support.v[1, i_b] = v
                            cont = True

                    if cont:
                        va = mpr_state.simplex_support.v[1, i_b] - mpr_state.simplex_support.v[0, i_b]
                        vb = mpr_state.simplex_support.v[2, i_b] - mpr_state.simplex_support.v[0, i_b]
                        direction = va.cross(vb)
                        direction = direction.normalized()
                        num_trials = num_trials + 1
                        if num_trials == 15:
                            ret = -1
                            break
                    else:
                        mpr_state.simplex_support.v1[3, i_b] = v1
                        mpr_state.simplex_support.v2[3, i_b] = v2
                        mpr_state.simplex_support.v[3, i_b] = v
                        mpr_state.simplex_size[i_b] = 4

    return ret


@qd.func
def guess_geoms_center(
    i_ga,
    i_gb,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    normal_ws,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
):
    # MPR algorithm was initially design to check whether a pair of convex geometries was colliding. The author
    # proposed to extend its application to collision detection as it can provide the contact normal and penetration
    # depth in some cases, i.e. when the original of the Minkowski difference can be projected inside the refined
    # portal. Beyond this specific scenario, it only provides an approximation, that gets worst and worst as the
    # ray casting and portal normal are misaligned.
    # For convex shape, one can show that everything should be fine for low penetration-to-size ratio for each
    # geometry, and the probability to accurately estimate the contact point decreases as this ratio increases.
    #
    # This issue can be avoided by initializing the algorithm with the good seach direction, basically the one
    # from the previous simulation timestep would do fine, as the penetration was smaller at that time and so the
    # likely for this direction to be valid was larger. Alternatively, the direction of the linear velocity would
    # be a good option.
    #
    # Enforcing a specific search direction to vanilla MPR is not straightforward, because the direction of the ray
    # control by v0, which is defined as the difference between the respective centers of each geometry.
    # The only option is to change the way the center of each geometry are defined, so as to make the ray casting
    # from origin to v0 as colinear as possible with the direction we are interested, while remaining included in
    # their respective geometry.
    # The idea is to offset the original centers of each geometry by a ratio that corresponds to their respective
    # (rotated) bounding box size along each axe. Each center cannot be moved more than half of its bound-box size
    # along each axe. This could lead to a center that is outside the geometries if they do not collide, but
    # should be fine otherwise. Anyway, this is not a big deal in practice and MPR is robust enough to converge to
    # a meaningful solution and if the center is slightly off of each geometry. Nevertheless, if it turns out this
    # is a real issue, one way to address it is to evaluate the exact signed distance of each center wrt their
    # respective geometry. If one of the center is off, its offset from the original center is divided by 2 and the
    # signed distance is computed once again until to find a valid point. This procedure should be cheap.
    EPS = rigid_info.EPS[None]

    # Transform geometry centers to world space using thread-local pos/quat
    center_a = gu.qd_transform_by_trans_quat(dyn_info.geoms.center[i_ga], pos_a, quat_a)
    center_b = gu.qd_transform_by_trans_quat(dyn_info.geoms.center[i_gb], pos_b, quat_b)

    # Completely different center logics if a normal guess is provided
    if qd.static(not rigid_config.enable_mujoco_compatibility):
        if (qd.abs(normal_ws) > collider_info.mpr.CCD_EPS[None]).any():
            # Must start from the center of each bounding box
            center_a_local = 0.5 * (geoms_init_AABB[i_ga, 7] + geoms_init_AABB[i_ga, 0])
            center_a = gu.qd_transform_by_trans_quat(center_a_local, pos_a, quat_a)
            center_b_local = 0.5 * (geoms_init_AABB[i_gb, 7] + geoms_init_AABB[i_gb, 0])
            center_b = gu.qd_transform_by_trans_quat(center_b_local, pos_b, quat_b)
            delta = center_a - center_b

            # Skip offset if normal is roughly pointing in the same direction already.
            # Note that a threshold of 0.5 would probably make more sense, but this means that the center of each
            # geometry would significantly affect collision detection, which is undesirable.
            normal = delta.normalized()
            if normal_ws.cross(normal).norm() > 0.01:
                # Compute the target offset
                offset = delta.dot(normal_ws) * normal_ws - delta
                offset_norm = offset.norm()

                if offset_norm > EPS:
                    # Compute the size of the bounding boxes along the target offset direction.
                    # First, move the direction in local box frame
                    dir_offset = offset / offset_norm
                    dir_offset_local_a = gu.qd_inv_transform_by_quat(dir_offset, quat_a)
                    dir_offset_local_b = gu.qd_inv_transform_by_quat(dir_offset, quat_b)
                    box_size_a = geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]
                    box_size_b = geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]
                    length_a = box_size_a.dot(qd.abs(dir_offset_local_a))
                    length_b = box_size_b.dot(qd.abs(dir_offset_local_b))

                    # Shift the center of each geometry
                    offset_ratio = qd.min(offset_norm / (length_a + length_b), 0.5)
                    center_a = center_a + dir_offset * length_a * offset_ratio
                    center_b = center_b - dir_offset * length_b * offset_ratio

    return center_a, center_b


@qd.func
def func_mpr_contact_from_centers(
    i_ga,
    i_gb,
    i_b,
    center_a,
    center_b,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    geoms_init_AABB: array_class.GeomsInitAABB,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    ccd_tol = collider_info.mpr.CCD_TOLERANCE[None]
    if qd.static(not rigid_config.enable_mujoco_compatibility):
        ccd_tol = ccd_tol * func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)

    res = mpr_discover_portal(
        i_ga,
        i_gb,
        i_b,
        ccd_tol,
        center_a,
        center_b,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        collider_state,
        mpr_state,
        dyn_info,
        collider_info,
        rigid_config,
        collider_static_config,
    )

    is_col = False
    pos = gs.qd_vec3([0.0, 0.0, 0.0])
    normal = gs.qd_vec3([0.0, 0.0, 0.0])
    penetration = gs.qd_float(0.0)

    # Default for the degenerate touch/segment paths (and refine failure): a contact with no reusable refined portal.
    # The refined-portal path classifies the portal precisely inside mpr_find_penetration.
    mpr_state.portal_status[i_b] = PORTAL_STATUS.NONE

    if res == 1:
        is_col, normal, penetration, pos = mpr_find_penetr_touch(i_ga, i_gb, i_b, mpr_state)
    elif res == 2:
        is_col, normal, penetration, pos = mpr_find_penetr_segment(i_ga, i_gb, i_b, mpr_state)
    elif res == 0:
        res = mpr_refine_portal(
            i_ga,
            i_gb,
            i_b,
            ccd_tol,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            collider_state,
            mpr_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )
        if res >= 0:
            is_col, normal, penetration, pos = mpr_find_penetration(
                i_ga,
                i_gb,
                i_b,
                ccd_tol,
                pos_a,
                quat_a,
                pos_b,
                quat_b,
                collider_state,
                mpr_state,
                dyn_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )
    return is_col, normal, penetration, pos


@qd.func
def func_mpr_contact(
    i_ga,
    i_gb,
    i_b,
    normal_ws,
    pos_a: qd.types.vector(3),
    quat_a: qd.types.vector(4),
    pos_b: qd.types.vector(3),
    quat_b: qd.types.vector(4),
    geoms_init_AABB: array_class.GeomsInitAABB,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    center_a, center_b = guess_geoms_center(
        i_ga,
        i_gb,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        normal_ws,
        geoms_init_AABB,
        dyn_info,
        rigid_info,
        collider_info,
        rigid_config,
    )
    return func_mpr_contact_from_centers(
        i_ga,
        i_gb,
        i_b,
        center_a,
        center_b,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        geoms_init_AABB,
        collider_state,
        mpr_state,
        dyn_info,
        collider_info,
        rigid_config,
        collider_static_config,
    )


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.mpr_decomp")
