"""Flag-based distance type classification and dispatch (@qd.func Quadrants version).

Ports the pinned reference ``distance_flag.h`` as quadrants @qd.func functions for use
inside graph kernels.

All vertex data is passed as flat f64 arrays and i32 indices.

NOTE: quadrants does not support ``return`` inside non-static ``if``.
All flag functions use a result variable with elif chains instead of
early returns.
"""

from __future__ import annotations

import quadrants as qd

from .distance_primitives import (
    ee_distance2,
    pe_distance2,
    pp_distance2,
    pt_distance2,
)

# ---------------------------------------------------------------------------
# PT flag classification
# ---------------------------------------------------------------------------


@qd.func
def pt_distance_flag(
    p0: qd.f64,
    p1: qd.f64,
    p2: qd.f64,
    t00: qd.f64,
    t01: qd.f64,
    t02: qd.f64,
    t10: qd.f64,
    t11: qd.f64,
    t12: qd.f64,
    t20: qd.f64,
    t21: qd.f64,
    t22: qd.f64,
) -> qd.i32:
    """Classify PT contact sub-type. Returns 4-bit flag."""
    b0x = t10 - t00
    b0y = t11 - t01
    b0z = t12 - t02
    b1x = t20 - t00
    b1y = t21 - t01
    b1z = t22 - t02
    b2x = p0 - t00
    b2y = p1 - t01
    b2z = p2 - t02
    nx = b0y * b1z - b0z * b1y
    ny = b0z * b1x - b0x * b1z
    nz = b0x * b1y - b0y * b1x

    # Edge t0-t1
    e1x = b0y * nz - b0z * ny
    e1y = b0z * nx - b0x * nz
    e1z = b0x * ny - b0y * nx
    D = b0x * (e1y * nz - e1z * ny) - b0y * (e1x * nz - e1z * nx) + b0z * (e1x * ny - e1y * nx)
    D1 = b2x * (e1y * nz - e1z * ny) - b2y * (e1x * nz - e1z * nx) + b2z * (e1x * ny - e1y * nx)
    D2 = b0x * (b2y * nz - b2z * ny) - b0y * (b2x * nz - b2z * nx) + b0z * (b2x * ny - b2y * nx)
    p0x_v = D1 / D
    p0y_v = D2 / D

    # Edge t1-t2
    b0x2 = t20 - t10
    b0y2 = t21 - t11
    b0z2 = t22 - t12
    e1x2 = b0y2 * nz - b0z2 * ny
    e1y2 = b0z2 * nx - b0x2 * nz
    e1z2 = b0x2 * ny - b0y2 * nx
    b2x2 = p0 - t10
    b2y2 = p1 - t11
    b2z2 = p2 - t12
    D_2 = b0x2 * (e1y2 * nz - e1z2 * ny) - b0y2 * (e1x2 * nz - e1z2 * nx) + b0z2 * (e1x2 * ny - e1y2 * nx)
    D1_2 = b2x2 * (e1y2 * nz - e1z2 * ny) - b2y2 * (e1x2 * nz - e1z2 * nx) + b2z2 * (e1x2 * ny - e1y2 * nx)
    D2_2 = b0x2 * (b2y2 * nz - b2z2 * ny) - b0y2 * (b2x2 * nz - b2z2 * nx) + b0z2 * (b2x2 * ny - b2y2 * nx)
    p1x_v = D1_2 / D_2
    p1y_v = D2_2 / D_2

    # Edge t2-t0
    b0x3 = t00 - t20
    b0y3 = t01 - t21
    b0z3 = t02 - t22
    e1x3 = b0y3 * nz - b0z3 * ny
    e1y3 = b0z3 * nx - b0x3 * nz
    e1z3 = b0x3 * ny - b0y3 * nx
    b2x3 = p0 - t20
    b2y3 = p1 - t21
    b2z3 = p2 - t22
    D_3 = b0x3 * (e1y3 * nz - e1z3 * ny) - b0y3 * (e1x3 * nz - e1z3 * nx) + b0z3 * (e1x3 * ny - e1y3 * nx)
    D1_3 = b2x3 * (e1y3 * nz - e1z3 * ny) - b2y3 * (e1x3 * nz - e1z3 * nx) + b2z3 * (e1x3 * ny - e1y3 * nx)
    D2_3 = b0x3 * (b2y3 * nz - b2z3 * ny) - b0y3 * (b2x3 * nz - b2z3 * nx) + b0z3 * (b2x3 * ny - b2y3 * nx)
    p2x_v = D1_3 / D_3
    p2y_v = D2_3 / D_3

    result = qd.i32(0xF)  # default: full PT
    if p0x_v > 0.0 and p0x_v < 1.0 and p0y_v >= 0.0:
        result = 0x7  # PE t0t1
    elif p1x_v > 0.0 and p1x_v < 1.0 and p1y_v >= 0.0:
        result = 0xD  # PE t1t2
    elif p2x_v > 0.0 and p2x_v < 1.0 and p2y_v >= 0.0:
        result = 0xB  # PE t2t0
    elif p0x_v <= 0.0 and p2x_v >= 1.0:
        result = 0x3  # PP t0
    elif p1x_v <= 0.0 and p0x_v >= 1.0:
        result = 0x5  # PP t1
    elif p2x_v <= 0.0 and p1x_v >= 1.0:
        result = 0x9  # PP t2

    return result


# ---------------------------------------------------------------------------
# EE flag classification
# ---------------------------------------------------------------------------


@qd.func
def ee_distance_flag_gipc_legacy(
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
) -> qd.i32:
    """Classify EE contact sub-type, GIPC legacy rules. Returns a 4-bit flag.

    Port of the pinned reference ``ee_distance_flag_gipc_legacy`` (``distance_flag.h``), which
    the pinned reference calls only from ``gipc_contact.cu``. The ConsistentIPC path uses
    ``ee_distance_flag`` below instead; see its docstring for what
    differs and why.
    """
    ux = ea1x - ea0x
    uy = ea1y - ea0y
    uz = ea1z - ea0z
    vx = eb1x - eb0x
    vy = eb1y - eb0y
    vz = eb1z - eb0z
    wx = ea0x - eb0x
    wy = ea0y - eb0y
    wz = ea0z - eb0z

    a = ux * ux + uy * uy + uz * uz
    b = ux * vx + uy * vy + uz * vz
    c = vx * vx + vy * vy + vz * vz
    d = ux * wx + uy * wy + uz * wz
    e = vx * wx + vy * wy + vz * wz

    D_val = a * c - b * b
    defaultCase = qd.i32(8)

    sN = b * e - c * d
    tN = qd.f64(0.0)
    tD = D_val

    if sN <= 0.0:
        tN = e
        tD = c
        defaultCase = 2
    elif sN >= D_val:
        tN = e + b
        tD = c
        defaultCase = 5
    else:
        tN = a * e - b * d
        uxv_x = uy * vz - uz * vy
        uxv_y = uz * vx - ux * vz
        uxv_z = ux * vy - uy * vx
        uxv2 = uxv_x * uxv_x + uxv_y * uxv_y + uxv_z * uxv_z
        uxv_dot_w = uxv_x * wx + uxv_y * wy + uxv_z * wz
        if tN > 0.0 and tN < tD and (uxv_dot_w == 0.0 or uxv2 < 1.0e-20 * a * c):
            if sN < D_val / 2:
                tN = e
                tD = c
                defaultCase = 2
            else:
                tN = e + b
                tD = c
                defaultCase = 5

    result = qd.i32(0xF)  # default: full EE
    neg_d = -d

    if tN <= 0.0:
        if neg_d <= 0.0:
            result = 0x5  # PP: ea0, eb0
        elif neg_d >= a:
            result = 0x6  # PP: ea1, eb0
        else:
            result = 0x7  # PE: eb0 on edgeA
    elif tN >= tD:
        neg_d_plus_b = neg_d + b
        if neg_d_plus_b <= 0.0:
            result = 0x9  # PP: ea0, eb1
        elif neg_d_plus_b >= a:
            result = 0xA  # PP: ea1, eb1
        else:
            result = 0xB  # PE: eb1 on edgeA
    elif defaultCase == 0:
        result = 0x5
    elif defaultCase == 1:
        result = 0x9
    elif defaultCase == 2:
        result = 0xD
    elif defaultCase == 3:
        result = 0x6
    elif defaultCase == 4:
        result = 0xA
    elif defaultCase == 5:
        result = 0xE
    elif defaultCase == 6:
        result = 0x7
    elif defaultCase == 7:
        result = 0xB

    return result


@qd.func
def ee_distance_flag(
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
) -> qd.i32:
    """Classify EE contact sub-type, ConsistentIPC rules. Returns a 4-bit flag.

    Port of the pinned reference ``ee_distance_flag`` (``distance_flag.h``), the classifier
    ``consistent_ipc_contact.cu`` calls. It differs from the GIPC legacy variant
    above in exactly two places, both of which the pinned reference documents at length:

    1. **A near-parallel guard.** ``D = |u x v|^2``, so ``D ~ 0`` means the two
       edges are nearly parallel and the segment parameter ``s`` is
       *indeterminate* -- for parallel segments the closest points form a
       continuum, not a point, and the sign tests below then pivot on rounding
       noise. Following Ericson's canonical segment-segment routine, ``s`` is
       pinned to 0 and the t-clamp plus s-reclamp select the true closest
       feature. Unlike endpoint enumeration this still reaches PE-on-edge-A,
       which is correct when a short edge lies inside a long collinear one.

    2. **GIPC's re-clamp in the interior branch is gone.** It fired on
       ``(u x v)·w == 0`` (the four points coplanar) or ``|u x v|^2 < 1e-20 a c``.
       Coplanarity is not degeneracy: with ``s`` and ``t`` both interior it means
       the segments genuinely cross, so the closest feature *is*
       interior-interior, which the clamp destroyed by reporting a far-away
       endpoint instead. It is also an exact float equality, so it never fires
       in generic position and always fires on axis-aligned meshes -- a
       classifier discontinuous across a 1-ulp perturbation. The ``1e-20`` test
       is unreachable once the guard above diverts everything below ``1e-6 a c``.

    Not bitwise-faithful in one respect: the pinned reference spells every operation with
    ``nofma_*`` so the classification is reproducible across compilers, and
    the Quadrants source language currently exposes no no-FMA primitive. The guard is a coarse
    threshold spanning six orders of magnitude, so contraction cannot flip it
    except for a pair sitting essentially on ``D == 1e-6 a c``; that is far more
    robust than the ungarded form, where the branch pivots on rounding at
    ``D ~ 1e-13``. Pairs near the threshold remain a known residual risk.
    """
    ux = ea1x - ea0x
    uy = ea1y - ea0y
    uz = ea1z - ea0z
    vx = eb1x - eb0x
    vy = eb1y - eb0y
    vz = eb1z - eb0z
    wx = ea0x - eb0x
    wy = ea0y - eb0y
    wz = ea0z - eb0z

    a = ux * ux + uy * uy + uz * uz
    b = ux * vx + uy * vy + uz * vz
    c = vx * vx + vy * vy + vz * vz
    d = ux * wx + uy * wy + uz * wz
    e = vx * wx + vy * wy + vz * wz

    D_val = a * c - b * b
    defaultCase = qd.i32(8)

    sN = b * e - c * d
    tN = qd.f64(0.0)
    tD = D_val

    near_parallel = qd.i32(0)
    if a > 0.0 and c > 0.0 and D_val <= 1.0e-6 * a * c:
        near_parallel = 1

    if near_parallel == 1 or sN <= 0.0:
        tN = e
        tD = c
        defaultCase = 2
    elif sN >= D_val:
        tN = e + b
        tD = c
        defaultCase = 5
    else:
        # s is interior and the edges are not near-parallel, so the parametric
        # solve is well conditioned -- keep it.
        tN = a * e - b * d

    result = qd.i32(0xF)  # default: full EE
    neg_d = -d

    if tN <= 0.0:
        if neg_d <= 0.0:
            result = 0x5  # PP: ea0, eb0
        elif neg_d >= a:
            result = 0x6  # PP: ea1, eb0
        else:
            result = 0x7  # PE: eb0 on edgeA
    elif tN >= tD:
        neg_d_plus_b = neg_d + b
        if neg_d_plus_b <= 0.0:
            result = 0x9  # PP: ea0, eb1
        elif neg_d_plus_b >= a:
            result = 0xA  # PP: ea1, eb1
        else:
            result = 0xB  # PE: eb1 on edgeA
    elif defaultCase == 0:
        result = 0x5
    elif defaultCase == 1:
        result = 0x9
    elif defaultCase == 2:
        result = 0xD
    elif defaultCase == 3:
        result = 0x6
    elif defaultCase == 4:
        result = 0xA
    elif defaultCase == 5:
        result = 0xE
    elif defaultCase == 6:
        result = 0x7
    elif defaultCase == 7:
        result = 0xB

    return result


# ---------------------------------------------------------------------------
# Flagged distance dispatch (device)
# ---------------------------------------------------------------------------


@qd.func
def _popcount4(flag: qd.i32) -> qd.i32:
    """Popcount of lower 4 bits."""
    c = qd.i32(0)
    for i in qd.static(range(4)):
        if flag & (1 << i):
            c = c + 1
    return c


@qd.func
def _get_offset(flag: qd.i32, k: qd.i32) -> qd.i32:
    """Get the k-th set bit position in lower 4 bits of flag."""
    count = qd.i32(0)
    result = qd.i32(0)
    for i in qd.static(range(4)):
        if flag & (1 << i):
            if count == k:
                result = i
            count = count + 1
    return result


@qd.func
def flag_active_offsets(
    flag: qd.i32,
    offsets: qd.template(),  # qd.Vector
):
    count = qd.i32(0)
    for index in qd.static(range(4)):
        if flag & (1 << index):
            offsets[count] = index
            count = count + 1


@qd.func
def flag_active_offsets_for_ee(
    flag: qd.i32,
    offsets: qd.template(),  # qd.Vector
    is_point_on_a: qd.template(),  # bool
):
    inactive = qd.i32(0)
    for index in qd.static(range(4)):
        if not (flag & (1 << index)):
            inactive = index
    if inactive < 2:
        offsets[0] = 1 - inactive
        offsets[1] = 2
        offsets[2] = 3
        is_point_on_a[0] = 1
    else:
        offsets[0] = 5 - inactive
        offsets[1] = 0
        offsets[2] = 1
        is_point_on_a[0] = 0


@qd.func
def pt_flagged_distance2(
    flag: qd.i32,
    verts: qd.template(),  # qd.Vector
) -> qd.f64:
    """Dispatch PT d² by flag sub-type. verts is (4, 3) field/ndarray."""
    pc = _popcount4(flag)
    result = qd.f64(0.0)
    if pc == 2:
        o0 = _get_offset(flag, 0)
        o1 = _get_offset(flag, 1)
        result = pp_distance2(
            verts[o0, 0],
            verts[o0, 1],
            verts[o0, 2],
            verts[o1, 0],
            verts[o1, 1],
            verts[o1, 2],
        )
    elif pc == 3:
        o0 = _get_offset(flag, 0)
        o1 = _get_offset(flag, 1)
        o2 = _get_offset(flag, 2)
        result = pe_distance2(
            verts[o0, 0],
            verts[o0, 1],
            verts[o0, 2],
            verts[o1, 0],
            verts[o1, 1],
            verts[o1, 2],
            verts[o2, 0],
            verts[o2, 1],
            verts[o2, 2],
        )
    else:
        result = pt_distance2(
            verts[0, 0],
            verts[0, 1],
            verts[0, 2],
            verts[1, 0],
            verts[1, 1],
            verts[1, 2],
            verts[2, 0],
            verts[2, 1],
            verts[2, 2],
            verts[3, 0],
            verts[3, 1],
            verts[3, 2],
        )
    return result


@qd.func
def gipc_d_EE(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    v2x: qd.f64,
    v2y: qd.f64,
    v2z: qd.f64,
    v3x: qd.f64,
    v3y: qd.f64,
    v3z: qd.f64,
) -> qd.f64:
    """Interior edge-edge d², the projection onto the edge cross product.

    Ports the pinned reference ``gipc_distance.h::gipc_d_EE``. Kept separate from the codegen'd
    ``ee_distance2`` (same quantity, different expression) so the value
    fed to the mollified barrier is the one computed by the pinned reference.
    """
    bx = (v1y - v0y) * (v3z - v2z) - (v1z - v0z) * (v3y - v2y)
    by = (v1z - v0z) * (v3x - v2x) - (v1x - v0x) * (v3z - v2z)
    bz = (v1x - v0x) * (v3y - v2y) - (v1y - v0y) * (v3x - v2x)
    aTb = (v2x - v0x) * bx + (v2y - v0y) * by + (v2z - v0z) * bz
    return aTb * aTb / (bx * bx + by * by + bz * bz)


@qd.func
def ee_flagged_distance2(
    flag: qd.i32,
    verts: qd.template(),  # qd.Vector
) -> qd.f64:
    """Dispatch EE d² by flag sub-type. verts is (4, 3) field/ndarray."""
    pc = _popcount4(flag)
    result = qd.f64(0.0)
    if pc == 2:
        o0 = _get_offset(flag, 0)
        o1 = _get_offset(flag, 1)
        result = pp_distance2(
            verts[o0, 0],
            verts[o0, 1],
            verts[o0, 2],
            verts[o1, 0],
            verts[o1, 1],
            verts[o1, 2],
        )
    elif pc == 3:
        inactive = qd.i32(0)
        for i in qd.static(range(4)):
            if not (flag & (1 << i)):
                inactive = i
        if inactive < 2:
            pt_idx = 1 - inactive
            result = pe_distance2(
                verts[pt_idx, 0],
                verts[pt_idx, 1],
                verts[pt_idx, 2],
                verts[2, 0],
                verts[2, 1],
                verts[2, 2],
                verts[3, 0],
                verts[3, 1],
                verts[3, 2],
            )
        else:
            pt_idx = 5 - inactive
            result = pe_distance2(
                verts[pt_idx, 0],
                verts[pt_idx, 1],
                verts[pt_idx, 2],
                verts[0, 0],
                verts[0, 1],
                verts[0, 2],
                verts[1, 0],
                verts[1, 1],
                verts[1, 2],
            )
    else:
        result = ee_distance2(
            verts[0, 0],
            verts[0, 1],
            verts[0, 2],
            verts[1, 0],
            verts[1, 1],
            verts[1, 2],
            verts[2, 0],
            verts[2, 1],
            verts[2, 2],
            verts[3, 0],
            verts[3, 1],
            verts[3, 2],
        )
    return result
