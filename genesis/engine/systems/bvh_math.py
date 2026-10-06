"""BVH math primitives: AABB operations, Morton codes, Karras 2012 helpers.

All functions are ``@qd.func`` for use inside graph kernels. They port
``bvh_types.cuh`` from CGQ, with the documented stricter outward rounding in
``aabb_expand``.
"""

# ruff: noqa: SIM102

from __future__ import annotations

import quadrants as qd

# ---------------------------------------------------------------------------
# Bounding-volume helpers. AABB uses six contiguous f64 values. DOP14f uses
# fourteen active f32 values in a sixteen-value row: lo[7], hi[7], padding[2].
# The 64-byte row stride mirrors CGQ's alignas(64) DOP14f.
# ---------------------------------------------------------------------------


@qd.func
def f64_to_f32_rd(value: qd.f64):
    """CGQ ``bvf_rd``: convert f64 to f32 with round-toward-negative-infinity."""
    result = qd.f32(value)
    if qd.f64(result) > value:
        bits = qd.bit_cast(result, qd.u32)
        if (bits & qd.u32(0x80000000)) != 0:
            bits = bits + qd.u32(1)
        else:
            bits = bits - qd.u32(1)
        result = qd.bit_cast(bits, qd.f32)
    return result


@qd.func
def f64_to_f32_ru(value: qd.f64):
    """CGQ ``bvf_ru``: convert f64 to f32 with round-toward-positive-infinity."""
    result = qd.f32(value)
    if qd.f64(result) < value:
        bits = qd.bit_cast(result, qd.u32)
        if (bits & qd.u32(0x80000000)) != 0:
            bits = bits - qd.u32(1)
        else:
            bits = bits + qd.u32(1)
        result = qd.bit_cast(bits, qd.f32)
    return result


@qd.func
def aabb_init(
    aabbs: qd.template(),
    idx: qd.i32,
    half: qd.template(),
    use_dop14f: qd.template(),
):
    """Reset aabbs[idx] to the empty sentinel (lower=+inf, upper=-inf).

    The fp64 AABB uses +-1e32 and DOP14f uses CGQ's +-3e38f. They are
    sentinels for "no bound yet" that min/max expansion reduces away, not a
    length scale. They are spelled inline because a module constant read from
    device code never enters the fastcache key, so editing it would silently
    reuse a stale kernel.
    """
    if qd.static(use_dop14f):
        for k in qd.static(range(half)):
            aabbs[idx, k] = qd.f32(3e38)
            aabbs[idx, half + k] = qd.f32(-3e38)
    else:
        for k in qd.static(range(half)):
            aabbs[idx, k] = qd.f64(1e32)
            aabbs[idx, half + k] = qd.f64(-1e32)


@qd.func
def aabb_expand(
    aabbs: qd.template(),
    idx: qd.i32,
    r: qd.f64,
    half: qd.template(),
    use_dop14f: qd.template(),
):
    """Expand a bound by ``r``.

    DOP14f rounds the final lower/upper values outward. CGQ currently rounds
    only ``r`` upward and then uses round-to-nearest f32 arithmetic; adversarial
    values prove that can move the final bound inward by one ulp.
    """
    for k in qd.static(range(half)):
        scale = qd.f64(1.0)
        if qd.static(k >= 3):
            scale = qd.f64(1.7320508075688772)
        if qd.static(use_dop14f):
            radius = r * scale
            aabbs[idx, k] = f64_to_f32_rd(qd.f64(aabbs[idx, k]) - radius)
            aabbs[idx, half + k] = f64_to_f32_ru(qd.f64(aabbs[idx, half + k]) + radius)
        else:
            aabbs[idx, k] = aabbs[idx, k] - r * scale
            aabbs[idx, half + k] = aabbs[idx, half + k] + r * scale


@qd.func
def aabb_combine_point(
    aabbs: qd.template(),
    idx: qd.i32,
    px: qd.f64,
    py: qd.f64,
    pz: qd.f64,
    half: qd.template(),
    use_dop14f: qd.template(),
):
    """Expand aabbs[idx] to include point (px, py, pz)."""
    projections = qd.Vector([px, py, pz, px + py + pz, px + py - pz, px - py + pz, px - py - pz])
    for axis in qd.static(range(half)):
        if qd.static(use_dop14f):
            aabbs[idx, axis] = qd.min(aabbs[idx, axis], f64_to_f32_rd(projections[axis]))
            aabbs[idx, half + axis] = qd.max(
                aabbs[idx, half + axis],
                f64_to_f32_ru(projections[axis]),
            )
        else:
            aabbs[idx, axis] = qd.min(aabbs[idx, axis], projections[axis])
            aabbs[idx, half + axis] = qd.max(aabbs[idx, half + axis], projections[axis])


@qd.func
def aabb_combine_aabb(
    dst: qd.template(),
    dst_idx: qd.i32,
    src: qd.template(),
    src_idx: qd.i32,
    half: qd.template(),
):
    """Expand dst[dst_idx] to include src[src_idx]."""
    for k in qd.static(range(half)):
        dst[dst_idx, k] = qd.min(dst[dst_idx, k], src[src_idx, k])
        dst[dst_idx, half + k] = qd.max(dst[dst_idx, half + k], src[src_idx, half + k])


@qd.func
def aabb_overlap(
    a: qd.template(),
    a_idx: qd.i32,
    b: qd.template(),
    b_idx: qd.i32,
    half: qd.template(),
):
    """Return 1 if AABB a and b overlap, 0 otherwise.

    Matches cgq ``aabb_overlap`` (plain overlap, no gap tolerance).
    """
    result = qd.i32(1)
    for axis in qd.static(range(half)):
        if b[b_idx, axis] > a[a_idx, half + axis] or a[a_idx, axis] > b[b_idx, half + axis]:
            result = 0
    return result


@qd.func
def aabb_overlap_gap(
    a: qd.template(),
    a_idx: qd.i32,
    b: qd.template(),
    b_idx: qd.i32,
    gap: qd.f64,
    half: qd.template(),
):
    """Return 1 if AABB a and b overlap within gap tolerance, 0 otherwise.

    Matches cgq ``aabb_overlap_gap``: boxes overlap iff separation along
    every axis is strictly less than *gap*.
    """
    result = qd.i32(1)
    for axis in qd.static(range(half)):
        axis_gap = gap
        if qd.static(axis >= 3):
            axis_gap = gap * 1.7320508075688772
        if (b[b_idx, axis] - a[a_idx, half + axis]) >= axis_gap or (a[a_idx, axis] - b[b_idx, half + axis]) >= axis_gap:
            result = 0
    return result


@qd.func
def aabb_center(
    aabbs: qd.template(),
    idx: qd.i32,
    half: qd.template(),
):
    """Return the center of aabbs[idx]."""
    c = qd.Vector.zero(qd.f64, 3)
    for k in qd.static(range(3)):
        c[k] = (aabbs[idx, k] + aabbs[idx, half + k]) * 0.5
    return c


# ---------------------------------------------------------------------------
# Morton codes (30-bit, from cgq bvh_types.cuh)
# ---------------------------------------------------------------------------


@qd.func
def expand_bits(v: qd.u32):
    """Spread 10-bit value into 30-bit 3-interleaved pattern."""
    v = (v * qd.u32(0x00010001)) & qd.u32(0xFF0000FF)
    v = (v * qd.u32(0x00000101)) & qd.u32(0x0F00F00F)
    v = (v * qd.u32(0x00000011)) & qd.u32(0xC30C30C3)
    v = (v * qd.u32(0x00000005)) & qd.u32(0x49249249)
    return v


@qd.func
def morton_code_30bit(x: qd.f64, y: qd.f64, z: qd.f64):
    """Compute 30-bit Morton code from [0,1]-normalized coordinates."""
    resolution = 1024.0
    xc = qd.min(qd.max(x * resolution, 0.0), resolution - 1.0)
    yc = qd.min(qd.max(y * resolution, 0.0), resolution - 1.0)
    zc = qd.min(qd.max(z * resolution, 0.0), resolution - 1.0)
    xx = expand_bits(qd.u32(xc))
    yy = expand_bits(qd.u32(yc))
    zz = expand_bits(qd.u32(zc))
    return (xx << qd.u32(2)) | (yy << qd.u32(1)) | zz


# ---------------------------------------------------------------------------
# Karras 2012 helpers (from cgq bvh_types.cuh)
# ---------------------------------------------------------------------------


@qd.func
def common_upper_bits(a: qd.u64, b: qd.u64):
    """Count leading zeros of (a XOR b).  Equivalent to ``__clzll(a ^ b)``."""
    return qd.i32(qd.math.clz(a ^ b))


@qd.func
def determine_range(
    codes: qd.template(),
    n: qd.i32,
    idx: qd.i32,
):
    """Karras 2012 determine_range: find the range [first, last] for internal node *idx*.

    *codes* is a 1-D array of sorted u64 Morton keys (shape ``(n,)``).
    Returns ``(first, last)`` as an i32 Vector of length 2.
    """
    result = qd.Vector.zero(qd.i32, 2)

    # idx == 0 special case: range is the full array
    # (no early return allowed in quadrants runtime if)
    result[0] = 0
    result[1] = n - 1

    if idx > 0:
        self_code = codes[idx]
        L_delta = common_upper_bits(self_code, codes[idx - 1])
        R_delta = common_upper_bits(self_code, codes[idx + 1])
        d = 1
        if R_delta <= L_delta:
            d = -1

        delta_min = qd.min(L_delta, R_delta)
        l_max = 2
        delta = qd.i32(-1)
        i_tmp = idx + d * l_max
        if i_tmp >= 0:
            if i_tmp < n:
                delta = common_upper_bits(self_code, codes[i_tmp])
        while delta > delta_min:
            l_max = l_max << 1
            i_tmp = idx + d * l_max
            delta = qd.i32(-1)
            if i_tmp >= 0:
                if i_tmp < n:
                    delta = common_upper_bits(self_code, codes[i_tmp])

        ll = qd.i32(0)
        t = l_max >> 1
        while t > 0:
            i_tmp = idx + (ll + t) * d
            delta = qd.i32(-1)
            if i_tmp >= 0:
                if i_tmp < n:
                    delta = common_upper_bits(self_code, codes[i_tmp])
            if delta > delta_min:
                ll = ll + t
            t = t >> 1

        jdx = idx + ll * d

        first = idx
        last = jdx
        if d < 0:
            first = jdx
            last = idx

        result[0] = first
        result[1] = last

    return result


@qd.func
def find_split(
    codes: qd.template(),
    n: qd.i32,
    first: qd.i32,
    last: qd.i32,
):
    """Karras 2012 find_split: binary search for the split position."""
    first_code = codes[first]
    last_code = codes[last]

    # Equal codes: split at midpoint (no early return in runtime if)
    split = (first + last) >> 1

    if first_code != last_code:
        delta_node = common_upper_bits(first_code, last_code)
        split = first
        stride = last - first
        cont = qd.i32(1)
        while cont != 0:
            stride = (stride + 1) >> 1
            middle = split + stride
            if middle < last:
                delta = common_upper_bits(first_code, codes[middle])
                if delta > delta_node:
                    split = middle
            if stride <= 1:
                cont = 0

    return split
