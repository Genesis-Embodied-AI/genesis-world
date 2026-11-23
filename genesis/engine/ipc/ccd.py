"""Continuous collision detection (CCD) of IPC contact stencils by conservative advancement, and the broad-phase rule.

A stencil has two sides, each given as the rows of a matrix: one point against three (point-triangle), two against two
(edge-edge), one against two (point-edge) or one against one (point-point). Every routine advances the pair by a proven
lower bound of its time of contact per iteration, so the time it reaches is safe to return at any point. It returns 0
when the pair already violates its contact thickness at the start, 1 when the whole step is free of contact, and
otherwise a conservative time of contact, reached once the gap above the thickness has shrunk below 'eta' times its
value at the start. An iteration budget bounds the work: a pair still advancing when it is spent returns the time it
reached.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_isfinite

from .distance import (
    func_closest_points,
    func_edge_edge_distance_flag,
    func_edge_edge_distance_type,
    func_edge_edge_flagged_distance2,
    func_point_edge_distance_flag,
    func_point_edge_flagged_distance2,
    func_point_triangle_distance_flag,
    func_point_triangle_flagged_distance2,
)


@qd.func
def _func_stencil_distance2(points_a, points_b, parallel_tolerance: float, is_parallel_projected: qd.template()):
    """Compute the squared distance of the closest features of a stencil.

    Edge-edge stencils are classified by func_edge_edge_distance_type when 'is_parallel_projected' holds, and by
    func_edge_edge_distance_flag otherwise, with 'parallel_tolerance' as tolerance.
    """
    n_a, n_b = qd.static(points_a.n, points_b.n)
    distance2 = gs.qd_float(0.0)
    if qd.static(n_a == 1 and n_b == 3):
        p, t0, t1, t2 = points_a[0, :], points_b[0, :], points_b[1, :], points_b[2, :]
        distance2, _ = func_point_triangle_flagged_distance2(
            func_point_triangle_distance_flag(p, t0, t1, t2), p, t0, t1, t2
        )
    elif qd.static(n_a == 2 and n_b == 2):
        ea0, ea1, eb0, eb1 = points_a[0, :], points_a[1, :], points_b[0, :], points_b[1, :]
        flag = 0
        if qd.static(is_parallel_projected):
            flag = func_edge_edge_distance_type(ea0, ea1, eb0, eb1, parallel_tolerance)
        else:
            flag = func_edge_edge_distance_flag(ea0, ea1, eb0, eb1, parallel_tolerance)
        distance2, _ = func_edge_edge_flagged_distance2(flag, ea0, ea1, eb0, eb1)
    elif qd.static(n_a == 1 and n_b == 2):
        p, e0, e1 = points_a[0, :], points_b[0, :], points_b[1, :]
        distance2, _ = func_point_edge_flagged_distance2(func_point_edge_distance_flag(p, e0, e1), p, e0, e1)
    else:
        distance2 = (points_a[0, :] - points_b[0, :]).norm_sqr()
    return distance2


@qd.func
def _func_max_row_norm(rows):
    norm2 = rows[0, :].norm_sqr()
    for i_v in qd.static(range(1, rows.n)):
        norm2 = qd.max(norm2, rows[i_v, :].norm_sqr())
    return qd.sqrt(norm2)


@qd.func
def _func_projected_gap(normal: qd.types.vector(3), points_a, points_b) -> float:
    """Compute the gap between the projections of the two sides on an axis, a lower bound of their distance."""
    proj_a = normal.dot(points_a[0, :])
    for i_v in qd.static(range(1, points_a.n)):
        proj_a = qd.max(proj_a, normal.dot(points_a[i_v, :]))
    proj_b = normal.dot(points_b[0, :])
    for i_v in qd.static(range(1, points_b.n)):
        proj_b = qd.min(proj_b, normal.dot(points_b[i_v, :]))
    return proj_b - proj_a


@qd.func
def _func_closing_rate(normal: qd.types.vector(3), displacements_a, displacements_b) -> float:
    """Compute how fast the projected gap along an axis can close, the support of the relative displacements."""
    rate_a = normal.dot(displacements_a[0, :])
    for i_v in qd.static(range(1, displacements_a.n)):
        rate_a = qd.max(rate_a, normal.dot(displacements_a[i_v, :]))
    rate_b = -normal.dot(displacements_b[0, :])
    for i_v in qd.static(range(1, displacements_b.n)):
        rate_b = qd.max(rate_b, -normal.dot(displacements_b[i_v, :]))
    return rate_a + rate_b


@qd.func
def func_additive_ccd(
    points_a,
    points_b,
    displacements_a,
    displacements_b,
    eta: float,
    thickness: float,
    max_iters: int,
    parallel_tolerance: float,
) -> float:
    """Compute a conservative time of contact of a stencil moving linearly, bounded by an isotropic displacement bound.

    The displacements of the vertices over the step are taken relative to their mean, and each iteration advances by
    (1 - eta) times the gap above the thickness over the sum of the largest displacement of each side. Edge-edge
    stencils are classified by func_edge_edge_distance_type with 'parallel_tolerance', and an edge-edge pair advanced
    into its thickness falls back to the distance of its closest vertices. See the module docstring for the contract.
    """
    n_a, n_b = qd.static(points_a.n, points_b.n)
    thickness2 = thickness * thickness
    positions_a = points_a
    positions_b = points_b
    relative_a = displacements_a
    relative_b = displacements_b
    max_displacement = gs.qd_float(0.0)
    gap = gs.qd_float(0.0)
    step = gs.qd_float(0.0)
    toc = gs.qd_float(0.0)
    toi = gs.qd_float(0.0)
    for i_iter in range(max_iters + 1):
        distance2 = _func_stencil_distance2(positions_a, positions_b, parallel_tolerance, is_parallel_projected=True)
        if qd.static(n_a == 2 and n_b == 2):
            if i_iter > 0 and distance2 - thickness2 <= 0.0:
                distance2 = qd.min(
                    qd.min(
                        (positions_a[0, :] - positions_b[0, :]).norm_sqr(),
                        (positions_a[0, :] - positions_b[1, :]).norm_sqr(),
                    ),
                    qd.min(
                        (positions_a[1, :] - positions_b[0, :]).norm_sqr(),
                        (positions_a[1, :] - positions_b[1, :]).norm_sqr(),
                    ),
                )
        excess = distance2 - thickness2
        distance = qd.sqrt(distance2)
        if i_iter == 0:
            if distance2 <= thickness2:
                toi = 0.0
                break
            # The displacements are summed in pairs in a fixed order per stencil, which sets the rounding of their mean
            total = displacements_a[0, :] + displacements_b[0, :]
            if qd.static(n_a == 1 and n_b == 3):
                total = (displacements_b[0, :] + displacements_b[1, :]) + (
                    displacements_b[2, :] + displacements_a[0, :]
                )
            elif qd.static(n_a == 2 and n_b == 2):
                total = (displacements_a[0, :] + displacements_a[1, :]) + (
                    displacements_b[0, :] + displacements_b[1, :]
                )
            elif qd.static(n_a == 1 and n_b == 2):
                total = (displacements_a[0, :] + displacements_b[0, :]) + displacements_b[1, :]
            shift = -(1.0 / (n_a + n_b)) * total
            for i_v in qd.static(range(n_a)):
                for k in qd.static(range(3)):
                    relative_a[i_v, k] = displacements_a[i_v, k] + shift[k]
            for i_v in qd.static(range(n_b)):
                for k in qd.static(range(3)):
                    relative_b[i_v, k] = displacements_b[i_v, k] + shift[k]
            max_displacement = _func_max_row_norm(relative_a) + _func_max_row_norm(relative_b)
            if max_displacement <= 0.0:
                toi = 1.0
                break
            gap = eta * excess / (distance + thickness)
        else:
            if toc > 0.0 and excess / (distance + thickness) < gap:
                toi = toc
                break
            toc += step
            if toc > 1.0:
                toi = 1.0
                break
        if i_iter == max_iters:
            toi = toc
            break
        step = (1.0 - eta) * excess / ((distance + thickness) * max_displacement)
        positions_a += relative_a * step
        positions_b += relative_b * step
    return toi


@qd.func
def func_directional_ccd(
    points_a, points_b, displacements_a, displacements_b, eta: float, thickness: float, max_iters: int
) -> float:
    """Compute a conservative time of contact of a stencil moving linearly, bounded by its closing rate along an axis.

    The axis n is the unit vector between the closest points of the two sides, the gap is the gap between their
    projections on n, and the closing rate is the largest approach of the projections over the step, which is never
    larger than the isotropic displacement bound and much smaller for sliding pairs. Each iteration advances by (1 -
    eta) times the gap above the thickness over the closing rate, a pair whose projections never approach being free
    over the whole step. Should rounding put the projected gap within the thickness, the isotropic bound takes over. See
    the module docstring for the contract.
    """
    is_point_point = qd.static(points_a.n == 1 and points_b.n == 1)
    positions_a = points_a
    positions_b = points_b
    distance_start = gs.qd_float(0.0)
    toc = gs.qd_float(0.0)
    toi = gs.qd_float(0.0)
    for i_iter in range(max_iters + 1):
        distance, closest_a, closest_b = func_closest_points(positions_a, positions_b)
        if i_iter == 0:
            distance_start = distance
            if distance <= thickness:
                toi = 0.0
                break
        elif distance - thickness < eta * (distance_start - thickness):
            toi = toc
            break
        if i_iter == max_iters:
            toi = toc
            break
        normal = (closest_b - closest_a) / distance
        gap = distance
        rate = _func_closing_rate(normal, displacements_a, displacements_b)
        if qd.static(not is_point_point):
            gap = _func_projected_gap(normal, positions_a, positions_b)
            if gap <= thickness:
                gap = distance
                rate = _func_max_row_norm(displacements_a) + _func_max_row_norm(displacements_b)
        if rate <= 0.0:
            toi = 1.0
            break
        step = (1.0 - eta) * (gap - thickness) / rate
        if toc + step >= 1.0:
            toi = 1.0
            break
        positions_a += displacements_a * step
        positions_b += displacements_b * step
        toc += step
    return toi


@qd.func
def _func_screw_positions(
    t: float,
    points,
    chords,
    rotation: qd.types.vector(3),
    pivot: qd.types.vector(3),
    pivot_displacement: qd.types.vector(3),
):
    """Evaluate the positions of the vertices of a side on its screw path at time t of the step.

    A side without rotation moves along its chords.
    """
    positions = points + t * chords
    if rotation.norm_sqr() > 0.0:
        quat = gu.qd_rotvec_to_quat(t * rotation)
        center = pivot + t * pivot_displacement
        for i_v in qd.static(range(points.n)):
            position = center + gu.qd_transform_by_quat_fast(points[i_v, :] - pivot, quat)
            for k in qd.static(range(3)):
                positions[i_v, k] = position[k]
    return positions


@qd.func
def _func_advance_fraction(gap: float, rate: float, curvature: float) -> float:
    """Compute the largest fraction s of the remaining step such that s * rate + s * (1 - s) * curvature <= gap.

    The gap is positive and the curvature non-negative. Returns a value above 1 when the whole remaining step is
    certified. The smaller root takes the cancellation-free form 2 gap / (b + sqrt(disc)), which degenerates smoothly to
    gap / rate as the curvature vanishes.
    """
    fraction = gs.qd_float(2.0)
    if curvature <= 0.0:
        if rate > 0.0:
            fraction = gap / rate
    else:
        slope = rate + curvature
        if slope > 0.0:
            disc = slope * slope - 4.0 * curvature * gap
            if disc >= 0.0:
                fraction = 2.0 * gap / (slope + qd.sqrt(disc))
    return fraction


@qd.func
def func_screw_ccd(
    points_a,
    chords_a,
    rotation_a: qd.types.vector(3),
    pivot_a: qd.types.vector(3),
    pivot_displacement_a: qd.types.vector(3),
    points_b,
    chords_b,
    rotation_b: qd.types.vector(3),
    pivot_b: qd.types.vector(3),
    pivot_displacement_b: qd.types.vector(3),
    t_start: float,
    t_end: float,
    distance_start: float,
    eta: float,
    thickness: float,
    max_iters: int,
    is_same_rigid_motion: bool,
):
    """Advance a stencil whose sides move on screw paths over the part [t_start, t_end] of the step.

    Each side moves rigidly, x(t) = pivot + t * pivot_displacement + R(t * rotation) (x(0) - pivot), the chords being
    x(1) - x(0), or linearly along its chords when its rotation is zero. The bound is the directional one of
    func_directional_ccd over the chords of the remaining interval, plus a curvature term bounding how far the arcs bulge
    out of their chords, which vanishes as the interval shrinks. Each step also evaluates the bound in a frame rotating
    with the faster side and takes the larger certified fraction, so that sides rotating together certify at once.
    'is_same_rigid_motion' tells that both sides move with the same rigid motion (they belong to the same link).

    The stop rule compares the gap with 'distance_start', the distance at the start of the whole step, so that the step
    can be split into intervals without changing where it stops: over [0, 1] with the distance at time 0 this is the
    whole-step advancement. Returns the time reached, the number of iterations, and the status: 0 when the whole interval
    is certified (the time is t_end), 1 when the stop rule is met or the thickness is violated at t_start (the time is
    the stop time), 2 when the iteration budget is spent (the time is the last certified one).
    """
    n_a, n_b = qd.static(points_a.n, points_b.n)
    is_point_point = qd.static(n_a == 1 and n_b == 1)
    theta_a = rotation_a.norm()
    theta_b = rotation_b.norm()
    levers_a = qd.Vector(
        [qd.select(theta_a > 0.0, (points_a[i_v, :] - pivot_a).norm(), 0.0) for i_v in qd.static(range(n_a))]
    )
    levers_b = qd.Vector(
        [qd.select(theta_b > 0.0, (points_b[i_v, :] - pivot_b).norm(), 0.0) for i_v in qd.static(range(n_b))]
    )

    # The faster rotating side defines the co-rotating frame, in which its vertices stand still
    is_frame_a = theta_a >= theta_b
    frame_rotation = qd.select(is_frame_a, rotation_a, rotation_b)
    frame_pivot = qd.select(is_frame_a, pivot_a, pivot_b)
    frame_pivot_displacement = qd.select(is_frame_a, pivot_displacement_a, pivot_displacement_b)
    is_frame_active = qd.max(theta_a, theta_b) > 0.0
    is_on_frame_a = is_frame_active and (is_frame_a or is_same_rigid_motion)
    is_on_frame_b = is_frame_active and (not is_frame_a or is_same_rigid_motion)

    ends_a = points_a + chords_a
    ends_b = points_b + chords_b
    if t_end < 1.0:
        ends_a = _func_screw_positions(t_end, points_a, chords_a, rotation_a, pivot_a, pivot_displacement_a)
        ends_b = _func_screw_positions(t_end, points_b, chords_b, rotation_b, pivot_b, pivot_displacement_b)

    positions_a = points_a
    positions_b = points_b
    t = t_start
    t_reached = t_start
    n_iters = 0
    status = 2
    for i_iter in range(max_iters + 1):
        if i_iter > 0 or t_start > 0.0:
            positions_a = _func_screw_positions(t, points_a, chords_a, rotation_a, pivot_a, pivot_displacement_a)
            positions_b = _func_screw_positions(t, points_b, chords_b, rotation_b, pivot_b, pivot_displacement_b)
        distance, closest_a, closest_b = func_closest_points(positions_a, positions_b)
        t_reached = t
        if distance - thickness < eta * (distance_start - thickness) or (i_iter == 0 and distance <= thickness):
            status = 1
            break
        if i_iter == max_iters:
            break
        n_iters = i_iter + 1
        normal = (closest_b - closest_a) / distance
        span = t_end - t
        chords_rem_a = ends_a - positions_a
        chords_rem_b = ends_b - positions_b

        gap = distance
        rate = _func_closing_rate(normal, chords_rem_a, chords_rem_b)
        is_isotropic = False
        if qd.static(not is_point_point):
            gap = _func_projected_gap(normal, positions_a, positions_b)
            is_isotropic = gap <= thickness
            if is_isotropic:
                gap = distance
                rate = _func_max_row_norm(chords_rem_a) + _func_max_row_norm(chords_rem_b)
        curvature_terms_a = (span * theta_a) * (span * theta_a) * levers_a
        curvature_terms_b = (span * theta_b) * (span * theta_b) * levers_b
        curvature_sum = gs.qd_float(0.0)
        for i_v in qd.static(range(n_a)):
            curvature_sum += curvature_terms_a[i_v]
        for i_v in qd.static(range(n_b)):
            curvature_sum += curvature_terms_b[i_v]
        curvature = 0.5 * curvature_sum

        rate_rel = rate
        curvature_rel = curvature
        if is_frame_active:
            # Chords and curvature bounds of the remaining interval in the co-rotating frame
            quat_inv = gu.qd_rotvec_to_quat(-span * frame_rotation)
            frame_center = frame_pivot + t * frame_pivot_displacement
            frame_center_end = frame_pivot + t_end * frame_pivot_displacement
            frame_rotation_rem = span * frame_rotation
            theta_frame = frame_rotation_rem.norm()
            frame_axis = frame_rotation_rem
            if theta_frame > 0.0:
                frame_axis = frame_rotation_rem / theta_frame
            curvature_sum_rel = gs.qd_float(0.0)
            chords_rel_a = qd.Matrix.zero(gs.qd_float, n_a, 3)
            chords_rel_b = qd.Matrix.zero(gs.qd_float, n_b, 3)
            sides = (
                (
                    rotation_a,
                    pivot_a,
                    pivot_displacement_a,
                    theta_a,
                    levers_a,
                    chords_a,
                    positions_a,
                    ends_a,
                    is_on_frame_a,
                ),
                (
                    rotation_b,
                    pivot_b,
                    pivot_displacement_b,
                    theta_b,
                    levers_b,
                    chords_b,
                    positions_b,
                    ends_b,
                    is_on_frame_b,
                ),
            )
            for i_side in qd.static(range(2)):
                rotation, pivot, pivot_displacement, theta, levers, chords, positions, ends, is_on_frame = sides[i_side]
                if not is_on_frame:
                    rotation_rem = span * rotation
                    theta_rel = (rotation_rem - frame_rotation_rem).norm()
                    theta_sum = theta_frame + rotation_rem.norm()
                    commutator = frame_rotation_rem.cross(rotation_rem).norm()
                    for i_v in qd.static(range(chords.n)):
                        chord_rel = (
                            gu.qd_transform_by_quat_fast(ends[i_v, :] - frame_center_end, quat_inv)
                            + frame_center
                            - positions[i_v, :]
                        )
                        for k in qd.static(range(3)):
                            if qd.static(i_side == 0):
                                chords_rel_a[i_v, k] = chord_rel[k]
                            else:
                                chords_rel_b[i_v, k] = chord_rel[k]
                        bound = (theta_rel * theta_rel + commutator + theta_sum * theta_sum * theta_rel) * levers[i_v]
                        # Transport of the vertex by the frame: rigid vertices ride their own pivot, the lever being
                        # bounded above, linear ones put their whole offset into it
                        offset = positions[i_v, :] - frame_center
                        offset_rate = span * chords[i_v, :] - span * frame_pivot_displacement
                        if theta > 0.0:
                            offset = (pivot + t * pivot_displacement) - frame_center
                            offset_rate = span * (pivot_displacement - frame_pivot_displacement)
                        offset_across = offset.norm()
                        if theta_frame > 0.0:
                            offset_across = (offset - frame_axis * frame_axis.dot(offset)).norm()
                        bound += 2.0 * theta_frame * offset_rate.norm() + theta_frame * theta_frame * (
                            offset_across + offset_rate.norm()
                        )
                        curvature_sum_rel += bound
            curvature_rel = 0.5 * curvature_sum_rel
            if is_isotropic:
                rate_rel = _func_max_row_norm(chords_rel_a) + _func_max_row_norm(chords_rel_b)
            else:
                rate_rel = _func_closing_rate(normal, chords_rel_a, chords_rel_b)

        if (rate <= 0.0 and curvature <= 0.0) or (rate_rel <= 0.0 and curvature_rel <= 0.0):
            t_reached = t_end
            status = 0
            break
        gap_safe = (1.0 - eta) * (gap - thickness)
        fraction = _func_advance_fraction(gap_safe, rate, curvature)
        if is_frame_active:
            fraction = qd.max(fraction, _func_advance_fraction(gap_safe, rate_rel, curvature_rel))
        if t + fraction * span >= t_end:
            t_reached = t_end
            status = 0
            break
        t += fraction * span
    return t_reached, n_iters, status


@qd.func
def func_point_halfplane_ccd(
    x: qd.types.vector(3),
    displacement: qd.types.vector(3),
    plane_point: qd.types.vector(3),
    normal: qd.types.vector(3),
    eta: float,
    thickness: float,
):
    """Compute a conservative time of contact of a point moving linearly towards a half-plane of unit normal.

    It is (1 - eta) times the time the point takes to reach the thickness at its approach speed, unclamped. A point
    moving away from or along the plane imposes no bound. Returns the time and whether the point bounds the step.
    """
    gap = normal.dot(x - plane_point) - thickness
    approach_speed = -normal.dot(displacement)
    toi = gs.qd_float(1.0)
    if approach_speed > 0.0:
        toi = gap / approach_speed * (1.0 - eta)
    return toi, approach_speed > 0.0


@qd.func
def func_point_halfplane_screw_ccd(
    x: qd.types.vector(3),
    chord: qd.types.vector(3),
    rotation: qd.types.vector(3),
    pivot: qd.types.vector(3),
    pivot_displacement: qd.types.vector(3),
    plane_point: qd.types.vector(3),
    normal: qd.types.vector(3),
    eta: float,
    thickness: float,
    max_iters: int,
) -> float:
    """Compute a conservative time of contact of a point moving on a screw path towards a half-plane of unit normal.

    The screw path is the one of func_screw_ccd. Each iteration advances along the remaining chord with the curvature
    term of the arc. See the module docstring for the contract.
    """
    plane_offset = normal.dot(plane_point)
    point = qd.Matrix.rows([x])
    chords = qd.Matrix.rows([chord])
    lever = (x - pivot).norm()
    theta = rotation.norm()
    position = x
    distance_start = gs.qd_float(0.0)
    t = gs.qd_float(0.0)
    toi = gs.qd_float(0.0)
    for i_iter in range(max_iters + 1):
        if i_iter > 0:
            position = _func_screw_positions(t, point, chords, rotation, pivot, pivot_displacement)[0, :]
        distance = normal.dot(position) - plane_offset
        if i_iter == 0:
            distance_start = distance
            if distance <= thickness:
                toi = 0.0
                break
        elif distance - thickness < eta * (distance_start - thickness):
            toi = t
            break
        if i_iter == max_iters:
            toi = t
            break
        rate = -normal.dot((x + chord) - position)
        theta_rem = (1.0 - t) * theta
        curvature = 0.5 * (theta_rem * theta_rem * lever)
        if rate <= 0.0 and curvature <= 0.0:
            toi = 1.0
            break
        fraction = _func_advance_fraction((1.0 - eta) * (distance - thickness), rate, curvature)
        if t + fraction * (1.0 - t) >= 1.0:
            toi = 1.0
            break
        t += fraction * (1.0 - t)
    return toi


@qd.func
def func_candidate_reach(
    points_a,
    points_b,
    ends_a,
    ends_b,
    inflations_a,
    inflations_b,
    speeds_a,
    speeds_b,
    band: float,
    parallel_tolerance: float,
    is_directional: bool,
) -> int:
    """Decide whether a candidate stencil of the broad phase can enter the contact band during the step.

    'ends' are the positions of the vertices at the end of their trajectory over the step, 'inflations' how far their
    paths bulge out of their chords (the sagitta of a screw path, zero for a straight one), 'speeds' an upper bound of
    the distance they travel if positive (otherwise the chord length plus the inflation is taken), and 'band' the
    contact band of the pair, its scaled activation distance plus its thickness. Edge-edge stencils are classified with
    'parallel_tolerance' (see func_edge_edge_distance_flag).

    The isotropic test drops a pair farther than the band plus the largest travel of each side. With 'is_directional',
    the pairs it keeps also go through the directional test along the axis between their closest points, where the gap
    between the projections of the sides closes at most by how far each side moves towards the other. Returns 0 when
    the pair never enters the band under any step fractions of its two sides, 2 when it never does provided both sides
    take the same step fraction, and 1 when it may enter the band.
    """
    n_a, n_b = qd.static(points_a.n, points_b.n)
    distance2 = _func_stencil_distance2(points_a, points_b, parallel_tolerance, is_parallel_projected=False)
    travels_a = qd.Vector(
        [
            qd.select(
                speeds_a[i_v] > 0.0, speeds_a[i_v], (ends_a[i_v, :] - points_a[i_v, :]).norm() + inflations_a[i_v]
            )
            for i_v in qd.static(range(n_a))
        ]
    )
    travels_b = qd.Vector(
        [
            qd.select(
                speeds_b[i_v] > 0.0, speeds_b[i_v], (ends_b[i_v, :] - points_b[i_v, :]).norm() + inflations_b[i_v]
            )
            for i_v in qd.static(range(n_b))
        ]
    )
    reach = band + (travels_a.max() + travels_b.max())
    verdict = 1
    # A non-finite reach keeps the pair
    if distance2 >= reach * reach:
        verdict = 0
    elif is_directional and qd_isfinite(band):
        distance, closest_a, closest_b = func_closest_points(points_a, points_b)
        if distance > 0.0:
            normal = (closest_b - closest_a) / distance
            gap = _func_projected_gap(normal, points_a, points_b) - inflations_a.max() - inflations_b.max()
            closing_a = normal.dot(ends_a[0, :] - points_a[0, :])
            for i_v in qd.static(range(1, n_a)):
                closing_a = qd.max(closing_a, normal.dot(ends_a[i_v, :] - points_a[i_v, :]))
            closing_b = -normal.dot(ends_b[0, :] - points_b[0, :])
            for i_v in qd.static(range(1, n_b)):
                closing_b = qd.max(closing_b, -normal.dot(ends_b[i_v, :] - points_b[i_v, :]))
            if not (gap - qd.max(closing_a, 0.0) - qd.max(closing_b, 0.0) < band):
                verdict = 0
            elif not (gap - qd.max(closing_a + closing_b, 0.0) < band):
                verdict = 2
    return verdict


@qd.func
def _func_swept_kdop(points, displacements):
    lower = qd.Vector([1e300] * 7, dt=gs.qd_float)
    upper = qd.Vector([-1e300] * 7, dt=gs.qd_float)
    for i_v in qd.static(range(points.n)):
        for i_end in qd.static(range(2)):
            q = points[i_v, :] + i_end * displacements[i_v, :]
            axes = qd.Vector(
                [q[0], q[1], q[2], q[0] + q[1] + q[2], q[0] + q[1] - q[2], q[0] - q[1] + q[2], q[0] - q[1] - q[2]]
            )
            lower = qd.min(lower, axes)
            upper = qd.max(upper, axes)
    return lower, upper


@qd.func
def func_is_swept_within(points_a, points_b, displacements_a, displacements_b, distance: float) -> bool:
    """Decide whether the two sides of a stencil may come within a distance of each other over a linear step.

    Each side is bounded by the 14-sided discrete oriented polytope of its vertices at the start and at the end of the
    step, along the axes x, y, z and the four diagonals of the cube. The sides are apart once their bounds are farther
    than 'distance' along some axis, the unnormalized diagonals taking sqrt(3) times the distance.
    """
    SQRT3 = qd.static(1.7320508075688772)
    lower_a, upper_a = _func_swept_kdop(points_a, displacements_a)
    lower_b, upper_b = _func_swept_kdop(points_b, displacements_b)
    is_apart = False
    for k in qd.static(range(7)):
        margin = qd.static(1.0 if k < 3 else SQRT3) * distance
        if lower_a[k] - upper_b[k] > margin or lower_b[k] - upper_a[k] > margin:
            is_apart = True
    return not is_apart


@qd.func
def func_point_halfplane_candidate_reach(
    x: qd.types.vector(3),
    end: qd.types.vector(3),
    plane_point: qd.types.vector(3),
    normal: qd.types.vector(3),
    inflation: float,
    band: float,
) -> bool:
    """Decide whether a point can enter the contact band of a half-plane of unit normal during the step.

    The signed distance of the point to the plane sweeps the interval between its values at the start and at the end
    of its trajectory, widened on both sides by the inflation of its path (see func_candidate_reach). The point is kept
    when that interval reaches above the plane and below 'band', the activation distance plus the thickness.
    """
    distance = normal.dot(x - plane_point)
    distance_change = normal.dot(end - x)
    distance_min = distance + qd.min(0.0, distance_change) - inflation
    distance_max = distance + qd.max(0.0, distance_change) + inflation
    return not (distance_max <= 0.0 or distance_min >= band)
