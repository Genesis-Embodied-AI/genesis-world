"""
Box collision contact detection functions.

This module contains specialized contact detection algorithms for box geometries:
- Plane-box contact detection
- Sphere-box contact detection
- Box-box contact detection (MuJoCo algorithm)
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
from . import support_field

from .contact import func_add_contact, func_compute_geom_pair_scale, rotaxis, rotmatx


@qd.func
def func_sphere_box_contact(
    i_ga, i_gb, ga_pos, ga_quat, gb_pos, gb_quat, dyn_info: array_class.DynInfo, rigid_info: array_class.RigidInfo
):
    """
    Analytical sphere-box collision detection.

    Reduces to finding the closest point on the box (in the box's local frame) to the sphere center,
    then comparing the distance to the sphere radius. Produces an exact normal aligned with one of the
    box's face axes, avoiding the small tilt that MPR/GJK introduces for shallow contacts.

    Parameters
    ----------
    ga_pos, ga_quat : Position and orientation of geom A (may be perturbed for multi-contact).
    gb_pos, gb_quat : Position and orientation of geom B (may be perturbed for multi-contact).
    """
    EPS = rigid_info.EPS[None]

    # Caller guarantees sphere is i_ga and box is i_gb (geoms are sorted by ascending type).
    sphere_center = ga_pos
    box_center = gb_pos
    box_quat = gb_quat
    sphere_radius = dyn_info.geoms.data[i_ga][0]
    box_half_size = qd.Vector(
        [0.5 * dyn_info.geoms.data[i_gb][0], 0.5 * dyn_info.geoms.data[i_gb][1], 0.5 * dyn_info.geoms.data[i_gb][2]],
        dt=gs.qd_float,
    )

    # Express the sphere center in the box's local frame
    p_local = gu.qd_inv_transform_by_trans_quat(sphere_center, box_center, box_quat)

    # Closest point on the box to the sphere center, in the box's local frame
    closest_local = qd.Vector(
        [
            qd.math.clamp(p_local[0], -box_half_size[0], box_half_size[0]),
            qd.math.clamp(p_local[1], -box_half_size[1], box_half_size[1]),
            qd.math.clamp(p_local[2], -box_half_size[2], box_half_size[2]),
        ],
        dt=gs.qd_float,
    )
    diff_local = p_local - closest_local
    dist_sq = diff_local.dot(diff_local)

    is_col = False
    normal_unit = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    penetration = gs.qd_float(0.0)

    if dist_sq < sphere_radius * sphere_radius:
        is_col = True
        normal_local = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        if dist_sq > EPS:
            # Sphere center outside the box, normal points from box surface to sphere center
            dist = qd.sqrt(dist_sq)
            normal_local = diff_local / dist
            penetration = sphere_radius - dist
        else:
            # Sphere center inside the box. The gap to the nearest face along each axis is half_size[i] - |p_local[i]|;
            # the smallest of those three gaps picks the contact axis, with the normal sign determined by which side
            # the sphere center is on.
            gaps = box_half_size - qd.abs(p_local)
            sign_x = gs.qd_float(1.0) if p_local[0] >= 0.0 else gs.qd_float(-1.0)
            sign_y = gs.qd_float(1.0) if p_local[1] >= 0.0 else gs.qd_float(-1.0)
            sign_z = gs.qd_float(1.0) if p_local[2] >= 0.0 else gs.qd_float(-1.0)
            normal_local = qd.Vector([sign_x, 0.0, 0.0], dt=gs.qd_float)
            min_gap = gaps[0]
            if gaps[1] < min_gap:
                min_gap = gaps[1]
                normal_local = qd.Vector([0.0, sign_y, 0.0], dt=gs.qd_float)
            if gaps[2] < min_gap:
                min_gap = gaps[2]
                normal_local = qd.Vector([0.0, 0.0, sign_z], dt=gs.qd_float)
            penetration = sphere_radius + min_gap

        normal_unit = gu.qd_transform_by_quat(normal_local, box_quat)
        contact_pos = sphere_center - (sphere_radius - 0.5 * penetration) * normal_unit

    return is_col, normal_unit, contact_pos, penetration


@qd.func
def func_plane_box_contact(
    i_ga,
    i_gb,
    i_b,
    i_pair,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    ga_pos, ga_quat = dyn_state.geoms.pos[i_ga, i_b], dyn_state.geoms.quat[i_ga, i_b]
    gb_pos, gb_quat = dyn_state.geoms.pos[i_gb, i_b], dyn_state.geoms.quat[i_gb, i_b]

    plane_dir = qd.Vector(
        [dyn_info.geoms.data[i_ga][0], dyn_info.geoms.data[i_ga][1], dyn_info.geoms.data[i_ga][2]], dt=gs.qd_float
    )
    plane_dir = gu.qd_transform_by_quat_fast(plane_dir, ga_quat)
    normal = -plane_dir.normalized()

    v1, _, _ = support_field._func_support_box(i_gb, normal, gb_pos, gb_quat, dyn_info)
    penetration = normal.dot(v1 - ga_pos)

    if penetration > 0.0:
        contact_pos = v1 - 0.5 * penetration * normal
        func_add_contact(
            i_ga,
            i_gb,
            i_b,
            i_pair,
            normal,
            contact_pos,
            penetration,
            dyn_state,
            collider_state,
            dyn_info,
            rigid_info,
            collider_info,
            use_atomic=False,
            errno=errno,
        )

        if qd.static(rigid_config.enable_multi_contact):
            n_con = 1
            contact_pos_0 = contact_pos
            tolerance = collider_info.mc_tolerance[None] * func_compute_geom_pair_scale(
                i_ga, i_gb, geoms_init_AABB, dyn_info
            )
            for i_v in range(dyn_info.geoms.vert_start[i_gb], dyn_info.geoms.vert_end[i_gb]):
                # Plane-box pairs are sized with the convex cap (they are not in the large-contact mask), so the
                # emission must stay within it to avoid overflowing a buffer allocated as a convex pair.
                if n_con < qd.static(collider_static_config.n_contacts_per_convex_pair):
                    pos_corner = gu.qd_transform_by_trans_quat_fast(dyn_info.verts.init_pos[i_v], gb_pos, gb_quat)
                    penetration = normal.dot(pos_corner - ga_pos)
                    if penetration > 0.0:
                        contact_pos = pos_corner - 0.5 * penetration * normal
                        if (contact_pos - contact_pos_0).norm() > tolerance:
                            func_add_contact(
                                i_ga,
                                i_gb,
                                i_b,
                                i_pair,
                                normal,
                                contact_pos,
                                penetration,
                                dyn_state,
                                collider_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                use_atomic=False,
                                errno=errno,
                            )
                            n_con = n_con + 1


@qd.func
def func_box_box_contact(
    i_ga,
    i_gb,
    i_b,
    i_pair,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    """
    Use Mujoco's box-box contact detection algorithm for more stable collision detection.

    The compilation and running time of this function is longer than the MPR-based contact detection.

    Algorithm is from

    https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c
    """
    EPS = rigid_info.EPS[None]

    n = 0
    code = -1
    margin = gs.qd_float(0.0)
    is_return = False
    cle1, cle2 = 0, 0
    in_ = 0
    tmp2 = qd.Vector.zero(gs.qd_float, 3)
    margin2 = margin * margin
    rotmore = qd.Matrix.zero(gs.qd_float, 3, 3)

    ga_pos = dyn_state.geoms.pos[i_ga, i_b]
    gb_pos = dyn_state.geoms.pos[i_gb, i_b]
    ga_quat = dyn_state.geoms.quat[i_ga, i_b]
    gb_quat = dyn_state.geoms.quat[i_gb, i_b]

    size1 = (
        qd.Vector(
            [dyn_info.geoms.data[i_ga][0], dyn_info.geoms.data[i_ga][1], dyn_info.geoms.data[i_ga][2]], dt=gs.qd_float
        )
        / 2
    )
    size2 = (
        qd.Vector(
            [dyn_info.geoms.data[i_gb][0], dyn_info.geoms.data[i_gb][1], dyn_info.geoms.data[i_gb][2]], dt=gs.qd_float
        )
        / 2
    )

    pos1, pos2 = ga_pos, gb_pos
    mat1, mat2 = gu.qd_quat_to_R(ga_quat, EPS), gu.qd_quat_to_R(gb_quat, EPS)

    tmp1 = pos2 - pos1
    pos21 = mat1.transpose() @ tmp1

    tmp1 = pos1 - pos2
    pos12 = mat2.transpose() @ tmp1

    rot = mat1.transpose() @ mat2
    rott = rot.transpose()

    rotabs = qd.abs(rot)
    rottabs = qd.abs(rott)

    plen2 = rotabs @ size2
    plen1 = rotabs.transpose() @ size1
    penetration = margin
    for i in qd.static(range(3)):
        penetration = penetration + size1[i] * 3 + size2[i] * 3
    for i in qd.static(range(3)):
        c1 = -qd.abs(pos21[i]) + size1[i] + plen2[i]
        c2 = -qd.abs(pos12[i]) + size2[i] + plen1[i]

        if (c1 < -margin) or (c2 < -margin):
            is_return = True

        if c1 < penetration:
            penetration = c1
            code = i + 3 * (pos21[i] < 0) + 0

        if c2 < penetration:
            penetration = c2
            code = i + 3 * (pos12[i] < 0) + 6
    clnorm = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
    for i, j in qd.static(qd.ndrange(3, 3)):
        rj0 = rott[j, 0]
        rj1 = rott[j, 1]
        rj2 = rott[j, 2]
        if i == 0:
            tmp2 = qd.Vector([0.0, -rj2, +rj1], dt=gs.qd_float)
        elif i == 1:
            tmp2 = qd.Vector([+rj2, 0.0, -rj0], dt=gs.qd_float)
        else:
            tmp2 = qd.Vector([-rj1, +rj0, 0.0], dt=gs.qd_float)

        c1 = tmp2.norm()
        tmp2 = tmp2 / c1
        if c1 >= EPS:
            c2 = pos21.dot(tmp2)

            c3 = gs.qd_float(0.0)

            for k in qd.static(range(3)):
                if k != i:
                    c3 = c3 + size1[k] * qd.abs(tmp2[k])

            for k in qd.static(range(3)):
                if k != j:
                    m = i
                    n = 3 - k - j
                    if k - j > 3:
                        m = m - 1
                        n = n + 3
                    c3 = c3 + size2[k] * rotabs[m, n] / c1

            c3 = c3 - qd.abs(c2)

            if c3 < -margin:
                is_return = True

            if c3 < penetration * (1.0 - 1e-12):
                penetration = c3
                cle1 = 0
                for k in qd.static(range(3)):
                    if (k != i) and ((tmp2[k] > 0) ^ (c2 < 0)):
                        cle1 = cle1 + (1 << k)

                cle2 = 0
                for k in qd.static(range(3)):
                    if k != j:
                        m = i
                        n = 3 - k - j
                        if k - j > 3:
                            m = m - 1
                            n = n + 3
                        if (rot[m, n] > 0) ^ (c2 < 0) ^ (qd.raw_mod(k - j + 3, 3) == 1):
                            cle2 = cle2 + (1 << k)

                code = 12 + i * 3 + j
                clnorm = tmp2
                in_ = c2 < 0
    if code == -1:
        is_return = True

    if not is_return:
        if code < 12:
            q1 = code % 6
            q2 = code // 6

            if q1 == 0:
                rotmore[0, 2] = -1
                rotmore[1, 1] = +1
                rotmore[2, 0] = +1
            elif q1 == 1:
                rotmore[0, 0] = +1
                rotmore[1, 2] = -1
                rotmore[2, 1] = +1
            elif q1 == 2:
                rotmore[0, 0] = +1
                rotmore[1, 1] = +1
                rotmore[2, 2] = +1
            elif q1 == 3:
                rotmore[0, 2] = +1
                rotmore[1, 1] = +1
                rotmore[2, 0] = -1
            elif q1 == 4:
                rotmore[0, 0] = +1
                rotmore[1, 2] = +1
                rotmore[2, 1] = -1
            elif q1 == 5:
                rotmore[0, 0] = -1
                rotmore[1, 1] = +1
                rotmore[2, 2] = -1

            i0 = 0
            i1 = 1
            i2 = 2
            f0 = f1 = f2 = 1
            if q1 == 0:
                i0 = 2
                f0 = -1
                i2 = 0
            elif q1 == 1:
                i1 = 2
                f1 = -1
                i2 = 1
            elif q1 == 3:
                i0 = 2
                i2 = 0
                f2 = -1
            elif q1 == 4:
                i1 = 2
                i2 = 1
                f2 = -1
            elif q1 == 5:
                f0 = -1
                f2 = -1

            r = qd.Matrix.zero(gs.qd_float, 3, 3)
            p = qd.Vector.zero(gs.qd_float, 3)
            s = qd.Vector.zero(gs.qd_float, 3)
            if q2:
                r = rotmore @ rot.transpose()
                p = rotaxis(pos12, i0, i1, i2, f0, f1, f2)
                tmp1 = rotaxis(size2, i0, i1, i2, f0, f1, f2)
                s = size1
            else:
                r = rotmatx(rot, i0, i1, i2, f0, f1, f2)
                p = rotaxis(pos21, i0, i1, i2, f0, f1, f2)
                tmp1 = rotaxis(size1, i0, i1, i2, f0, f1, f2)
                s = size2

            rt = r.transpose()
            ss = qd.abs(tmp1)
            lx = ss[0]
            ly = ss[1]
            hz = ss[2]
            p[2] = p[2] - hz
            lp = p

            clcorner = 0

            for i in qd.static(range(3)):
                if r[2, i] < 0:
                    clcorner = clcorner + (1 << i)

            for i in qd.static(range(3)):
                lp = lp + rt[i, :] * s[i] * (1 if (clcorner & (1 << i)) else -1)

            m, k = 0, 0
            collider_state.box_pts[m, i_b] = lp
            m = m + 1

            for i in qd.static(range(3)):
                if qd.abs(r[2, i]) < 0.5:
                    collider_state.box_pts[m, i_b] = rt[i, :] * s[i] * (-2 if (clcorner & (1 << i)) else 2)
                    m = m + 1

            collider_state.box_pts[3, i_b] = collider_state.box_pts[0, i_b] + collider_state.box_pts[1, i_b]
            collider_state.box_pts[4, i_b] = collider_state.box_pts[0, i_b] + collider_state.box_pts[2, i_b]
            collider_state.box_pts[5, i_b] = collider_state.box_pts[3, i_b] + collider_state.box_pts[2, i_b]

            if m > 1:
                collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[0, i_b]
                collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[1, i_b]
                k = k + 1

            if m > 2:
                collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[0, i_b]
                collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[2, i_b]
                k = k + 1

                collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[3, i_b]
                collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[2, i_b]
                k = k + 1

                collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[4, i_b]
                collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[1, i_b]
                k = k + 1

            for i in range(k):
                for q in qd.static(range(2)):
                    a = collider_state.box_lines[i, i_b][0 + q]
                    b = collider_state.box_lines[i, i_b][3 + q]
                    c = collider_state.box_lines[i, i_b][1 - q]
                    d = collider_state.box_lines[i, i_b][4 - q]
                    if qd.abs(b) > EPS:
                        for _j in qd.static(range(2)):
                            j = 2 * _j - 1
                            l = ss[q] * j
                            c1 = (l - a) / b
                            if 0 <= c1 and c1 <= 1:
                                c2 = c + d * c1
                                if qd.abs(c2) <= ss[1 - q]:
                                    collider_state.box_points[n, i_b] = (
                                        collider_state.box_lines[i, i_b][0:3]
                                        + collider_state.box_lines[i, i_b][3:6] * c1
                                    )
                                    n = n + 1
            a = collider_state.box_pts[1, i_b][0]
            b = collider_state.box_pts[2, i_b][0]
            c = collider_state.box_pts[1, i_b][1]
            d = collider_state.box_pts[2, i_b][1]
            c1 = a * d - b * c

            if m > 2:
                for i in qd.static(range(4)):
                    llx = lx if (i // 2) else -lx
                    lly = ly if (i % 2) else -ly

                    x = llx - collider_state.box_pts[0, i_b][0]
                    y = lly - collider_state.box_pts[0, i_b][1]

                    u = (x * d - y * b) / c1
                    v = (y * a - x * c) / c1

                    if 0 < u and u < 1 and 0 < v and v < 1:
                        collider_state.box_points[n, i_b] = qd.Vector(
                            [
                                llx,
                                lly,
                                collider_state.box_pts[0, i_b][2]
                                + u * collider_state.box_pts[1, i_b][2]
                                + v * collider_state.box_pts[2, i_b][2],
                            ]
                        )
                        n = n + 1

            for i in range(1 << (m - 1)):
                tmp1 = collider_state.box_pts[0 if i == 0 else i + 2, i_b]
                if not (i and (tmp1[0] <= -lx or tmp1[0] >= lx or tmp1[1] <= -ly or tmp1[1] >= ly)):
                    collider_state.box_points[n, i_b] = tmp1
                    n = n + 1
            m = n
            n = 0

            for i in range(m):
                if collider_state.box_points[i, i_b][2] <= margin:
                    collider_state.box_points[n, i_b] = collider_state.box_points[i, i_b]
                    collider_state.box_depth[n, i_b] = collider_state.box_points[n, i_b][2]
                    collider_state.box_points[n, i_b][2] = collider_state.box_points[n, i_b][2] * 0.5
                    n = n + 1
            r = (mat2 if q2 else mat1) @ rotmore.transpose()
            p = pos2 if q2 else pos1
            tmp2 = qd.Vector(
                [(-1 if q2 else 1) * r[0, 2], (-1 if q2 else 1) * r[1, 2], (-1 if q2 else 1) * r[2, 2]], dt=gs.qd_float
            )
            normal_0 = tmp2

            n_added = 0
            n_start = collider_state.n_contacts[i_b]
            for i in range(n):
                if n_added < qd.static(collider_static_config.n_contacts_per_nonconvex_pair):
                    dist = collider_state.box_points[i, i_b][2]
                    collider_state.box_points[i, i_b][2] = collider_state.box_points[i, i_b][2] + hz
                    contact_pos = p + r @ collider_state.box_points[i, i_b]

                    # Filter out redundant contact points
                    is_valid = True
                    for j_ in range(n_added):
                        j = n_start + j_
                        if (qd.abs(contact_pos - collider_state.contact_data.pos[j, i_b]) < EPS).all():
                            is_valid = False

                    if is_valid:
                        func_add_contact(
                            i_ga,
                            i_gb,
                            i_b,
                            i_pair,
                            -normal_0,
                            contact_pos,
                            -dist,
                            dyn_state,
                            collider_state,
                            dyn_info,
                            rigid_info,
                            collider_info,
                            use_atomic=False,
                            errno=errno,
                        )
                        n_added = n_added + 1
        else:
            code = code - 12

            q1 = code // 3
            q2 = code % 3

            ax1, ax2 = 0, 0
            pax1, pax2 = 0, 0

            if q2 == 0:
                ax1, ax2 = 1, 2
            elif q2 == 1:
                ax1, ax2 = 0, 2
            elif q2 == 2:
                ax1, ax2 = 1, 0

            if q1 == 0:
                pax1, pax2 = 1, 2
            elif q1 == 1:
                pax1, pax2 = 0, 2
            elif q1 == 2:
                pax1, pax2 = 1, 0
            if rotabs[q1, ax1] < rotabs[q1, ax2]:
                ax1 = ax2
                ax2 = 3 - q2 - ax1

            if rottabs[q2, pax1] < rottabs[q2, pax2]:
                pax1 = pax2
                pax2 = 3 - q1 - pax1

            clface = 0
            if cle1 & (1 << pax2):
                clface = pax2
            else:
                clface = pax2 + 3

            rotmore.fill(0.0)
            if clface == 0:
                rotmore[0, 2], rotmore[1, 1], rotmore[2, 0] = -1, +1, +1
            elif clface == 1:
                rotmore[0, 0], rotmore[1, 2], rotmore[2, 1] = +1, -1, +1
            elif clface == 2:
                rotmore[0, 0], rotmore[1, 1], rotmore[2, 2] = +1, +1, +1
            elif clface == 3:
                rotmore[0, 2], rotmore[1, 1], rotmore[2, 0] = +1, +1, -1
            elif clface == 4:
                rotmore[0, 0], rotmore[1, 2], rotmore[2, 1] = +1, +1, -1
            elif clface == 5:
                rotmore[0, 0], rotmore[1, 1], rotmore[2, 2] = -1, +1, -1

            i0, i1, i2 = 0, 1, 2
            f0, f1, f2 = 1, 1, 1

            if clface == 0:
                i0, i2, f0 = 2, 0, -1
            elif clface == 1:
                i1, i2, f1 = 2, 1, -1
            elif clface == 3:
                i0, i2, f2 = 2, 0, -1
            elif clface == 4:
                i1, i2, f2 = 2, 1, -1
            elif clface == 5:
                f0, f2 = -1, -1

            p = rotaxis(pos21, i0, i1, i2, f0, f1, f2)
            rnorm = rotaxis(clnorm, i0, i1, i2, f0, f1, f2)
            r = rotmatx(rot, i0, i1, i2, f0, f1, f2)

            # TODO
            tmp1 = rotmore.transpose() @ size1

            s = qd.abs(tmp1)
            rt = r.transpose()

            lx, ly, hz = s[0], s[1], s[2]
            p[2] = p[2] - hz

            n = 0
            collider_state.box_points[n, i_b] = p

            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[ax1, :] * size2[ax1] * (
                1 if (cle2 & (1 << ax1)) else -1
            )
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[ax2, :] * size2[ax2] * (
                1 if (cle2 & (1 << ax2)) else -1
            )

            collider_state.box_points[n + 1, i_b] = collider_state.box_points[n, i_b]
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[q2, :] * size2[q2]

            n = 1
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] - rt[q2, :] * size2[q2]

            n = 2
            collider_state.box_points[n, i_b] = p
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[ax1, :] * size2[ax1] * (
                -1 if (cle2 & (1 << ax1)) else 1
            )
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[ax2, :] * size2[ax2] * (
                1 if (cle2 & (1 << ax2)) else -1
            )

            collider_state.box_points[n + 1, i_b] = collider_state.box_points[n, i_b]
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[q2, :] * size2[q2]

            n = 3
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] - rt[q2, :] * size2[q2]

            n = 4
            collider_state.box_axi[0, i_b] = collider_state.box_points[0, i_b]
            collider_state.box_axi[1, i_b] = collider_state.box_points[1, i_b] - collider_state.box_points[0, i_b]
            collider_state.box_axi[2, i_b] = collider_state.box_points[2, i_b] - collider_state.box_points[0, i_b]

            if qd.abs(rnorm[2]) < EPS:
                is_return = True
            if not is_return:
                innorm = (1 / rnorm[2]) * (-1 if in_ else 1)

                for i in qd.static(range(4)):
                    c1 = -collider_state.box_points[i, i_b][2] / rnorm[2]
                    collider_state.box_pu[i, i_b] = collider_state.box_points[i, i_b]
                    collider_state.box_points[i, i_b] = collider_state.box_points[i, i_b] + c1 * rnorm

                    collider_state.box_ppts2[i, 0, i_b] = collider_state.box_points[i, i_b][0]
                    collider_state.box_ppts2[i, 1, i_b] = collider_state.box_points[i, i_b][1]
                collider_state.box_pts[0, i_b] = collider_state.box_points[0, i_b]
                collider_state.box_pts[1, i_b] = collider_state.box_points[1, i_b] - collider_state.box_points[0, i_b]
                collider_state.box_pts[2, i_b] = collider_state.box_points[2, i_b] - collider_state.box_points[0, i_b]

                m = 3
                k = 0
                n = 0

                if m > 1:
                    collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[0, i_b]
                    collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[1, i_b]
                    collider_state.box_linesu[k, i_b][0:3] = collider_state.box_axi[0, i_b]
                    collider_state.box_linesu[k, i_b][3:6] = collider_state.box_axi[1, i_b]
                    k = k + 1

                if m > 2:
                    collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[0, i_b]
                    collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[2, i_b]
                    collider_state.box_linesu[k, i_b][0:3] = collider_state.box_axi[0, i_b]
                    collider_state.box_linesu[k, i_b][3:6] = collider_state.box_axi[2, i_b]
                    k = k + 1

                    collider_state.box_lines[k, i_b][0:3] = (
                        collider_state.box_pts[0, i_b] + collider_state.box_pts[1, i_b]
                    )
                    collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[2, i_b]
                    collider_state.box_linesu[k, i_b][0:3] = (
                        collider_state.box_axi[0, i_b] + collider_state.box_axi[1, i_b]
                    )
                    collider_state.box_linesu[k, i_b][3:6] = collider_state.box_axi[2, i_b]
                    k = k + 1

                    collider_state.box_lines[k, i_b][0:3] = (
                        collider_state.box_pts[0, i_b] + collider_state.box_pts[2, i_b]
                    )
                    collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[1, i_b]
                    collider_state.box_linesu[k, i_b][0:3] = (
                        collider_state.box_axi[0, i_b] + collider_state.box_axi[2, i_b]
                    )
                    collider_state.box_linesu[k, i_b][3:6] = collider_state.box_axi[1, i_b]
                    k = k + 1

                for i in range(k):
                    for q in qd.static(range(2)):
                        a = collider_state.box_lines[i, i_b][q]
                        b = collider_state.box_lines[i, i_b][q + 3]
                        c = collider_state.box_lines[i, i_b][1 - q]
                        d = collider_state.box_lines[i, i_b][4 - q]

                        if qd.abs(b) > EPS:
                            for _j in qd.static(range(2)):
                                j = 2 * _j - 1
                                l = s[q] * j
                                c1 = (l - a) / b
                                if 0 <= c1 and c1 <= 1:
                                    c2 = c + d * c1
                                    if (qd.abs(c2) <= s[1 - q]) and (
                                        (
                                            collider_state.box_linesu[i, i_b][2]
                                            + collider_state.box_linesu[i, i_b][5] * c1
                                        )
                                        * innorm
                                        <= margin
                                    ):
                                        collider_state.box_points[n, i_b] = (
                                            collider_state.box_linesu[i, i_b][0:3] * 0.5
                                            + c1 * 0.5 * collider_state.box_linesu[i, i_b][3:6]
                                        )
                                        collider_state.box_points[n, i_b][q] = (
                                            collider_state.box_points[n, i_b][q] + 0.5 * l
                                        )
                                        collider_state.box_points[n, i_b][1 - q] = (
                                            collider_state.box_points[n, i_b][1 - q] + 0.5 * c2
                                        )
                                        collider_state.box_depth[n, i_b] = (
                                            collider_state.box_points[n, i_b][2] * innorm * 2
                                        )
                                        n = n + 1

                nl = n
                a = collider_state.box_pts[1, i_b][0]
                b = collider_state.box_pts[2, i_b][0]
                c = collider_state.box_pts[1, i_b][1]
                d = collider_state.box_pts[2, i_b][1]
                c1 = a * d - b * c

                for i in range(4):
                    llx = lx if (i // 2) else -lx
                    lly = ly if (i % 2) else -ly

                    x = llx - collider_state.box_pts[0, i_b][0]
                    y = lly - collider_state.box_pts[0, i_b][1]

                    u = (x * d - y * b) / c1
                    v = (y * a - x * c) / c1

                    if nl == 0:
                        if (u < 0 or u > 1) and (v < 0 or v > 1):
                            continue
                    elif u < 0 or u > 1 or v < 0 or v > 1:
                        continue

                    u = qd.math.clamp(u, 0, 1)
                    v = qd.math.clamp(v, 0, 1)
                    tmp1 = (
                        collider_state.box_pu[0, i_b] * (1 - u - v)
                        + collider_state.box_pu[1, i_b] * u
                        + collider_state.box_pu[2, i_b] * v
                    )
                    collider_state.box_points[n, i_b][0] = llx
                    collider_state.box_points[n, i_b][1] = lly
                    collider_state.box_points[n, i_b][2] = 0

                    tmp2 = collider_state.box_points[n, i_b] - tmp1

                    c2 = tmp2.dot(tmp2)

                    if not (tmp1[2] > 0 and c2 > margin2):
                        collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + tmp1
                        collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] * 0.5

                        collider_state.box_depth[n, i_b] = qd.sqrt(c2) * (-1 if tmp1[2] < 0 else 1)
                        n = n + 1

                nf = n

                for i in range(4):
                    x, y = collider_state.box_ppts2[i, 0, i_b], collider_state.box_ppts2[i, 1, i_b]

                    if nl == 0:
                        if (nf != 0) and (x < -lx or x > lx) and (y < -ly or y > ly):
                            continue
                    elif x < -lx or x > lx or y < -ly or y > ly:
                        continue

                    c1 = 0
                    for j in qd.static(range(2)):
                        if collider_state.box_ppts2[i, j, i_b] < -s[j]:
                            c1 = c1 + (collider_state.box_ppts2[i, j, i_b] + s[j]) ** 2
                        elif collider_state.box_ppts2[i, j, i_b] > s[j]:
                            c1 = c1 + (collider_state.box_ppts2[i, j, i_b] - s[j]) ** 2

                    c1 = c1 + (collider_state.box_pu[i, i_b][2] * innorm) ** 2

                    if collider_state.box_pu[i, i_b][2] > 0 and c1 > margin2:
                        continue

                    tmp1 = qd.Vector(
                        [
                            collider_state.box_ppts2[i, 0, i_b] * 0.5,
                            collider_state.box_ppts2[i, 1, i_b] * 0.5,
                            0,
                        ],
                        dt=gs.qd_float,
                    )

                    for j in qd.static(range(2)):
                        if collider_state.box_ppts2[i, j, i_b] < -s[j]:
                            tmp1[j] = -s[j] * 0.5
                        elif collider_state.box_ppts2[i, j, i_b] > s[j]:
                            tmp1[j] = s[j] * 0.5

                    tmp1 = tmp1 + collider_state.box_pu[i, i_b] * 0.5
                    collider_state.box_points[n, i_b] = tmp1

                    collider_state.box_depth[n, i_b] = qd.sqrt(c1) * (-1 if collider_state.box_pu[i, i_b][2] < 0 else 1)
                    n = n + 1

                r = mat1 @ rotmore.transpose()

                normal_0 = (-1 if in_ else 1) * (r @ rnorm)

                n_added = 0
                n_start = collider_state.n_contacts[i_b]
                for i in range(n):
                    if n_added < qd.static(collider_static_config.n_contacts_per_nonconvex_pair):
                        dist = collider_state.box_depth[i, i_b]
                        collider_state.box_points[i, i_b][2] = collider_state.box_points[i, i_b][2] + hz
                        contact_pos = pos1 + (r @ collider_state.box_points[i, i_b])

                        # Filter out redundant contact points
                        is_valid = True
                        for j_ in range(n_added):
                            j = n_start + j_
                            if (qd.abs(contact_pos - collider_state.contact_data.pos[j, i_b]) < EPS).all():
                                is_valid = False

                        if is_valid:
                            func_add_contact(
                                i_ga,
                                i_gb,
                                i_b,
                                i_pair,
                                -normal_0,
                                contact_pos,
                                -dist,
                                dyn_state,
                                collider_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                use_atomic=False,
                                errno=errno,
                            )
                            n_added = n_added + 1
