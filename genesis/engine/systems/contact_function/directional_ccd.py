"""Directional CCD — device (@qd.func) implementation.

Port of the pinned reference ``directional_ccd.h`` (closest-point conservative advancement)
and ``halfplane_contact_kernels.cu`` (analytic ray-plane).

Closest-point routines follow Ericson, *Real-Time Collision Detection*:
``ClosestPtPointTriangle`` (``closest_pt``) and
``ClosestPtSegmentSegment`` (``closest_ee``).

quadrants does not support ``return`` inside non-static ``if``, so all
functions write results to output templates and avoid early returns.
"""

# ruff: noqa: SIM102

from __future__ import annotations

import quadrants as qd

# Pinned reference gipc_accd.h: constexpr int32_t ACCD_MAX_ITERS = 50000;
# Structural invariant: never changed at runtime. Passed as a qd.template()
# parameter from callers so it enters the fastcache key (Standing rule 5).
CCD_MAX_ITERS_DEFAULT = 50000


# ---------------------------------------------------------------------------
# Closest-point queries (Ericson) — @qd.func
# ---------------------------------------------------------------------------


@qd.func
def closest_pt(
    px: qd.f64,
    py: qd.f64,
    pz: qd.f64,
    t0x: qd.f64,
    t0y: qd.f64,
    t0z: qd.f64,
    t1x: qd.f64,
    t1y: qd.f64,
    t1z: qd.f64,
    t2x: qd.f64,
    t2y: qd.f64,
    t2z: qd.f64,
    out: qd.template(),  # qd.Vector
):
    """Closest point on triangle to point. Writes ``out[0..2]`` = closest point, ``out[3]`` = dist.

    Port of the pinned reference ``detail::closest_pt``. Uses flag-based dispatch to avoid
    early returns (quadrants limitation).
    """
    abx = t1x - t0x
    aby = t1y - t0y
    abz = t1z - t0z
    acx = t2x - t0x
    acy = t2y - t0y
    acz = t2z - t0z
    apx = px - t0x
    apy = py - t0y
    apz = pz - t0z
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz

    bpx = px - t1x
    bpy = py - t1y
    bpz = pz - t1z
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz

    cpx = px - t2x
    cpy = py - t2y
    cpz = pz - t2z
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # region=0: vertex t0
    # region=1: vertex t1
    # region=2: edge t0-t1
    # region=3: vertex t2
    # region=4: edge t0-t2
    # region=5: edge t1-t2
    # region=6: interior
    region = qd.i32(6)
    if d1 <= 0.0 and d2 <= 0.0:
        region = 0
    if d3 >= 0.0 and d4 <= d3:
        if region == 6:
            region = 1
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        if region == 6:
            region = 2
    if d6 >= 0.0 and d5 <= d6:
        if region == 6:
            region = 3
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        if region == 6:
            region = 4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        if region == 6:
            region = 5

    cx = qd.f64(0.0)
    cy = qd.f64(0.0)
    cz = qd.f64(0.0)

    if region == 0:
        cx = t0x
        cy = t0y
        cz = t0z
    if region == 1:
        cx = t1x
        cy = t1y
        cz = t1z
    if region == 2:
        v = d1 / (d1 - d3)
        cx = t0x + v * abx
        cy = t0y + v * aby
        cz = t0z + v * abz
    if region == 3:
        cx = t2x
        cy = t2y
        cz = t2z
    if region == 4:
        w = d2 / (d2 - d6)
        cx = t0x + w * acx
        cy = t0y + w * acy
        cz = t0z + w * acz
    if region == 5:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        cx = t1x + w * (t2x - t1x)
        cy = t1y + w * (t2y - t1y)
        cz = t1z + w * (t2z - t1z)
    if region == 6:
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        cx = t0x + abx * v + acx * w
        cy = t0y + aby * v + acy * w
        cz = t0z + abz * v + acz * w

    out[0] = cx
    out[1] = cy
    out[2] = cz
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    out[3] = qd.sqrt(dx * dx + dy * dy + dz * dz)


@qd.func
def closest_ee(
    p1x: qd.f64,
    p1y: qd.f64,
    p1z: qd.f64,
    q1x: qd.f64,
    q1y: qd.f64,
    q1z: qd.f64,
    p2x: qd.f64,
    p2y: qd.f64,
    p2z: qd.f64,
    q2x: qd.f64,
    q2y: qd.f64,
    q2z: qd.f64,
    out: qd.template(),  # qd.Vector
):
    """Closest points of two segments. Writes ``out[0..2]`` = ca, ``out[3..5]`` = cb, ``out[6]`` = dist.

    Port of the pinned reference ``detail::closest_ee``. Flag-based dispatch.
    """
    d1x = q1x - p1x
    d1y = q1y - p1y
    d1z = q1z - p1z
    d2x = q2x - p2x
    d2y = q2y - p2y
    d2z = q2z - p2z
    rx = p1x - p2x
    ry = p1y - p2y
    rz = p1z - p2z
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    c = d1x * rx + d1y * ry + d1z * rz
    b = d1x * d2x + d1y * d2y + d1z * d2z

    TINY = qd.f64(1e-300)
    s = qd.f64(0.0)
    t = qd.f64(0.0)

    # case=0: both degenerate
    # case=1: a degenerate
    # case=2: e degenerate
    # case=3: general
    case = qd.i32(3)
    if a <= TINY and e <= TINY:
        case = 0
    if a <= TINY and case == 3:
        case = 1
    if e <= TINY and case == 3:
        case = 2

    if case == 0:
        s = 0.0
        t = 0.0
    if case == 1:
        s = 0.0
        t = qd.min(qd.max(f / e, 0.0), 1.0)
    if case == 2:
        t = 0.0
        s = qd.min(qd.max(-c / a, 0.0), 1.0)
    if case == 3:
        denom = a * e - b * b
        if denom > 0.0:
            s = qd.min(qd.max((b * f - c * e) / denom, 0.0), 1.0)
        else:
            s = 0.0
        t = (b * s + f) / e
        if t < 0.0:
            t = 0.0
            s = qd.min(qd.max(-c / a, 0.0), 1.0)
        if t > 1.0:
            t = 1.0
            s = qd.min(qd.max((b - c) / a, 0.0), 1.0)

    out[0] = p1x + s * d1x
    out[1] = p1y + s * d1y
    out[2] = p1z + s * d1z
    out[3] = p2x + t * d2x
    out[4] = p2y + t * d2y
    out[5] = p2z + t * d2z
    dx = out[0] - out[3]
    dy = out[1] - out[4]
    dz = out[2] - out[5]
    out[6] = qd.sqrt(dx * dx + dy * dy + dz * dz)


# ---------------------------------------------------------------------------
# Directional CCD — @qd.func
# ---------------------------------------------------------------------------


@qd.func
def directional_point_triangle_ccd(
    px: qd.f64,
    py: qd.f64,
    pz: qd.f64,
    t0x: qd.f64,
    t0y: qd.f64,
    t0z: qd.f64,
    t1x: qd.f64,
    t1y: qd.f64,
    t1z: qd.f64,
    t2x: qd.f64,
    t2y: qd.f64,
    t2z: qd.f64,
    dpx: qd.f64,
    dpy: qd.f64,
    dpz: qd.f64,
    dt0x: qd.f64,
    dt0y: qd.f64,
    dt0z: qd.f64,
    dt1x: qd.f64,
    dt1y: qd.f64,
    dt1z: qd.f64,
    dt2x: qd.f64,
    dt2y: qd.f64,
    dt2z: qd.f64,
    eta: qd.f64,
    thickness: qd.f64,
    max_iters: qd.template(),  # int
    result: qd.template(),  # qd.Vector
):
    """Directional CCD for point-triangle. Writes ``result[0]`` = toc.

    Port of the pinned reference ``directional_point_triangle_ccd``.
    """
    cp = qd.Vector.zero(qd.f64, 4)
    closest_pt(px, py, pz, t0x, t0y, t0z, t1x, t1y, t1z, t2x, t2y, t2z, cp)
    dist = cp[3]

    toc = qd.f64(0.0)
    done = qd.i32(0)
    if dist <= thickness:
        done = 1

    d0 = dist
    cbx = cp[0]
    cby = cp[1]
    cbz = cp[2]

    lpx = px
    lpy = py
    lpz = pz
    lt0x = t0x
    lt0y = t0y
    lt0z = t0z
    lt1x = t1x
    lt1y = t1y
    lt1z = t1z
    lt2x = t2x
    lt2y = t2y
    lt2z = t2z

    count = qd.i32(0)
    while count < max_iters and done == 0:
        count = count + 1
        if done == 0:
            nx = (cbx - lpx) / dist
            ny = (cby - lpy) / dist
            nz = (cbz - lpz) / dist

            n_t0 = nx * lt0x + ny * lt0y + nz * lt0z
            n_t1 = nx * lt1x + ny * lt1y + nz * lt1z
            n_t2 = nx * lt2x + ny * lt2y + nz * lt2z
            n_p = nx * lpx + ny * lpy + nz * lpz
            gap = qd.min(qd.min(n_t0, n_t1), n_t2) - n_p

            n_dp = nx * dpx + ny * dpy + nz * dpz
            neg_n_dt0 = -(nx * dt0x + ny * dt0y + nz * dt0z)
            neg_n_dt1 = -(nx * dt1x + ny * dt1y + nz * dt1z)
            neg_n_dt2 = -(nx * dt2x + ny * dt2y + nz * dt2z)
            rate = n_dp + qd.max(qd.max(neg_n_dt0, neg_n_dt1), neg_n_dt2)

            if gap <= thickness:
                gap = dist
                dp_sq = dpx * dpx + dpy * dpy + dpz * dpz
                dt0_sq = dt0x * dt0x + dt0y * dt0y + dt0z * dt0z
                dt1_sq = dt1x * dt1x + dt1y * dt1y + dt1z * dt1z
                dt2_sq = dt2x * dt2x + dt2y * dt2y + dt2z * dt2z
                rate = qd.sqrt(dp_sq) + qd.sqrt(qd.max(dt0_sq, qd.max(dt1_sq, dt2_sq)))
            if rate <= 0.0:
                toc = 1.0
                done = 1
            if done == 0:
                step = (1.0 - eta) * (gap - thickness) / rate
                if toc + step >= 1.0:
                    toc = 1.0
                    done = 1
                if done == 0:
                    lpx = lpx + dpx * step
                    lpy = lpy + dpy * step
                    lpz = lpz + dpz * step
                    lt0x = lt0x + dt0x * step
                    lt0y = lt0y + dt0y * step
                    lt0z = lt0z + dt0z * step
                    lt1x = lt1x + dt1x * step
                    lt1y = lt1y + dt1y * step
                    lt1z = lt1z + dt1z * step
                    lt2x = lt2x + dt2x * step
                    lt2y = lt2y + dt2y * step
                    lt2z = lt2z + dt2z * step
                    toc = toc + step
                    closest_pt(lpx, lpy, lpz, lt0x, lt0y, lt0z, lt1x, lt1y, lt1z, lt2x, lt2y, lt2z, cp)
                    dist = cp[3]
                    cbx = cp[0]
                    cby = cp[1]
                    cbz = cp[2]
                    if dist - thickness < eta * (d0 - thickness):
                        done = 1
    result[0] = toc


@qd.func
def directional_edge_edge_ccd(
    ea0x: qd.f64,
    ea0y: qd.f64,
    ea0z: qd.f64,
    ea1x: qd.f64,
    ea1y: qd.f64,
    ea1z: qd.f64,
    eb0x: qd.f64,
    eb0y: qd.f64,
    eb0z: qd.f64,
    eb1x: qd.f64,
    eb1y: qd.f64,
    eb1z: qd.f64,
    dea0x: qd.f64,
    dea0y: qd.f64,
    dea0z: qd.f64,
    dea1x: qd.f64,
    dea1y: qd.f64,
    dea1z: qd.f64,
    deb0x: qd.f64,
    deb0y: qd.f64,
    deb0z: qd.f64,
    deb1x: qd.f64,
    deb1y: qd.f64,
    deb1z: qd.f64,
    eta: qd.f64,
    thickness: qd.f64,
    max_iters: qd.template(),  # int
    result: qd.template(),  # qd.Vector
):
    """Directional CCD for edge-edge. Writes ``result[0]`` = toc.

    Port of the pinned reference ``directional_edge_edge_ccd``.
    """
    cp = qd.Vector.zero(qd.f64, 7)
    closest_ee(ea0x, ea0y, ea0z, ea1x, ea1y, ea1z, eb0x, eb0y, eb0z, eb1x, eb1y, eb1z, cp)
    dist = cp[6]

    toc = qd.f64(0.0)
    done = qd.i32(0)
    if dist <= thickness:
        done = 1

    d0 = dist
    cax = cp[0]
    cay = cp[1]
    caz = cp[2]
    cbx = cp[3]
    cby = cp[4]
    cbz = cp[5]

    la0x = ea0x
    la0y = ea0y
    la0z = ea0z
    la1x = ea1x
    la1y = ea1y
    la1z = ea1z
    lb0x = eb0x
    lb0y = eb0y
    lb0z = eb0z
    lb1x = eb1x
    lb1y = eb1y
    lb1z = eb1z

    count = qd.i32(0)
    while count < max_iters and done == 0:
        count = count + 1
        if done == 0:
            nx = (cbx - cax) / dist
            ny = (cby - cay) / dist
            nz = (cbz - caz) / dist

            n_b0 = nx * lb0x + ny * lb0y + nz * lb0z
            n_b1 = nx * lb1x + ny * lb1y + nz * lb1z
            n_a0 = nx * la0x + ny * la0y + nz * la0z
            n_a1 = nx * la1x + ny * la1y + nz * la1z
            gap = qd.min(n_b0, n_b1) - qd.max(n_a0, n_a1)

            n_da0 = nx * dea0x + ny * dea0y + nz * dea0z
            n_da1 = nx * dea1x + ny * dea1y + nz * dea1z
            neg_n_db0 = -(nx * deb0x + ny * deb0y + nz * deb0z)
            neg_n_db1 = -(nx * deb1x + ny * deb1y + nz * deb1z)
            rate = qd.max(n_da0, n_da1) + qd.max(neg_n_db0, neg_n_db1)

            if gap <= thickness:
                gap = dist
                da0_sq = dea0x * dea0x + dea0y * dea0y + dea0z * dea0z
                da1_sq = dea1x * dea1x + dea1y * dea1y + dea1z * dea1z
                db0_sq = deb0x * deb0x + deb0y * deb0y + deb0z * deb0z
                db1_sq = deb1x * deb1x + deb1y * deb1y + deb1z * deb1z
                rate = qd.sqrt(qd.max(da0_sq, da1_sq)) + qd.sqrt(qd.max(db0_sq, db1_sq))
            if rate <= 0.0:
                toc = 1.0
                done = 1
            if done == 0:
                step = (1.0 - eta) * (gap - thickness) / rate
                if toc + step >= 1.0:
                    toc = 1.0
                    done = 1
                if done == 0:
                    la0x = la0x + dea0x * step
                    la0y = la0y + dea0y * step
                    la0z = la0z + dea0z * step
                    la1x = la1x + dea1x * step
                    la1y = la1y + dea1y * step
                    la1z = la1z + dea1z * step
                    lb0x = lb0x + deb0x * step
                    lb0y = lb0y + deb0y * step
                    lb0z = lb0z + deb0z * step
                    lb1x = lb1x + deb1x * step
                    lb1y = lb1y + deb1y * step
                    lb1z = lb1z + deb1z * step
                    toc = toc + step
                    closest_ee(la0x, la0y, la0z, la1x, la1y, la1z, lb0x, lb0y, lb0z, lb1x, lb1y, lb1z, cp)
                    dist = cp[6]
                    cax = cp[0]
                    cay = cp[1]
                    caz = cp[2]
                    cbx = cp[3]
                    cby = cp[4]
                    cbz = cp[5]
                    if dist - thickness < eta * (d0 - thickness):
                        done = 1
    result[0] = toc


@qd.func
def halfplane_ccd(
    x0: qd.f64,
    x1: qd.f64,
    x2: qd.f64,
    dx0: qd.f64,
    dx1: qd.f64,
    dx2: qd.f64,
    Px: qd.f64,
    Py: qd.f64,
    Pz: qd.f64,
    Nx: qd.f64,
    Ny: qd.f64,
    Nz: qd.f64,
    eta: qd.f64,
    thickness: qd.f64,
    result: qd.template(),  # qd.Vector
):
    """Analytic ray-plane CCD. Writes ``result[0]`` = toi.

    Port of the pinned reference halfplane_ccd_alpha_kernel (linear path).
    """
    current_dist = Nx * (x0 - Px) + Ny * (x1 - Py) + Nz * (x2 - Pz)
    safe_dist = current_dist - thickness
    toi = qd.f64(1.0)
    if safe_dist <= 0.0:
        toi = 0.0
    else:
        approach_speed = -(Nx * dx0 + Ny * dx1 + Nz * dx2)
        if approach_speed > 0.0:
            raw = safe_dist / approach_speed * (1.0 - eta)
            toi = qd.min(raw, 1.0)
    result[0] = toi
