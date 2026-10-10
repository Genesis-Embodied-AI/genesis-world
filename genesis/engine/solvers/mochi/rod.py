# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Discrete elastic rods (Kirchhoff): stretching of the segments (1D Saint Venant-Kirchhoff), bending and twisting at the
interior nodes from the integrated curvature binormal of adjacent segments and of their material axes.

The unknowns are the node positions and one twist angle per segment (stored at the segment's first node). The material
axis of a segment is a slaved state: at every trial iterate it is parallel transported from the reference iterate's
tangent to the new one and rotated by the twist increment about the new tangent; the stencils below therefore see the
current axes and linearize the twist about zero (d a / d theta = t x a).

Tangents: the axial stencil carries its exact tangent with the geometric part floored (positive semidefinite); the bend
and twist stencil carries the Gauss-Newton tangent only (exact at rest), as mochi does.
"""

import quadrants as qd

import genesis as gs

from .lie import skew


@qd.func
def func_rod_normalize(v, tiny):
    return v / qd.max(v.norm(), tiny)


@qd.func
def func_rod_dnormalize(v, tiny):
    """d(v / |v|) / dv."""
    norm = qd.max(v.norm(), tiny)
    u = v / norm
    return (qd.Matrix.identity(gs.qd_float, 3) - u.outer_product(u)) / norm


@qd.func
def func_rod_parallel_transport(n0, n, tiny):
    """Minimal rotation carrying the unit vector n0 to the unit vector n."""
    c = n0.dot(n)
    v = n0.cross(n)
    return c * qd.Matrix.identity(gs.qd_float, 3) + skew(v) + v.outer_product(v) / qd.max(1.0 + c, tiny)


@qd.func
def func_rod_dparallel_transported(n0, n, v, tiny):
    """Derivatives of PT(n0, n) v with respect to n0 and n."""
    cr = n0.cross(n)
    c = 1.0 / qd.max(1.0 + n0.dot(n), tiny)
    cr_dot_v = cr.dot(v)
    proj_v = v - (c * c * cr_dot_v) * cr
    inner = -skew(v) + c * (cr.outer_product(v) + cr_dot_v * qd.Matrix.identity(gs.qd_float, 3))
    d_n0 = proj_v.outer_product(n) + inner @ (-skew(n))
    d_n = proj_v.outer_product(n0) + inner @ skew(n0)
    return d_n0, d_n


@qd.func
def func_rod_curvature_binormal(e0, e1, tiny):
    """Integrated curvature binormal 2 e0 x e1 / (1 + e0 . e1) of two unit vectors."""
    return (2.0 / qd.max(1.0 + e0.dot(e1), tiny)) * e0.cross(e1)


@qd.func
def func_rod_dcurvature_binormal(e0, e1, tiny):
    cr = e0.cross(e1)
    inv_denom = 1.0 / qd.max(1.0 + e0.dot(e1), tiny)
    t = 2.0 * inv_denom
    d_e0 = t * (-skew(e1) - inv_denom * cr.outer_product(e1))
    d_e1 = t * (skew(e0) - inv_denom * cr.outer_product(e0))
    return d_e0, d_e1


@qd.func
def func_rod_rotation(axis, angle):
    aa = axis.outer_product(axis)
    return aa + qd.cos(angle) * (qd.Matrix.identity(gs.qd_float, 3) - aa) + qd.sin(angle) * skew(axis)


@qd.func
def func_rod_transport_axis(base_tangent, new_tangent, twist, axis, tiny):
    """Material axis after the tangent moved from base_tangent to new_tangent and the segment twisted by the angle
    twist about the new tangent."""
    return func_rod_rotation(new_tangent, twist) @ (func_rod_parallel_transport(base_tangent, new_tangent, tiny) @ axis)


# ------------------------------------------------------------------------------------
# ---------------------------------------- axial -------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_rod_axial_strain(x0, x1, L):
    q = (x1 - x0) / L
    return 0.5 * (q.norm_sqr() - 1.0), q


@qd.func
def func_rod_axial(x0, x1, L, EA, f, strain_ss, eps, assem_dres):
    """Stretching of a segment of reference length L: energy, force on the two nodes as the residual pair
    (res on node 0 = -g, on node 1 = +g) and the symmetric 3x3 tangent T (node blocks +T, -T; -T, +T). f is the
    stiffness damping factor beta / h and strain_ss the stage-start strain."""
    strain, q = func_rod_axial_strain(x0, x1, L)
    EA_eff = (1.0 + f) * EA
    ss_weight = f / (1.0 + f)
    strain_eff = strain - ss_weight * strain_ss
    stress = EA_eff * strain_eff
    energy = 0.5 * stress * strain_eff * L
    g = stress * q
    T = qd.Matrix.zero(gs.qd_float, 3, 3)
    if assem_dres:
        geometric = stress / L
        geometric = max(geometric, eps)
        T = (EA_eff / L) * q.outer_product(q) + geometric * qd.Matrix.identity(gs.qd_float, 3)
    return energy, g, T


# ------------------------------------------------------------------------------------
# ------------------------------------- bend + twist ---------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_rod_bend_twist_measures(x0, x1, x2, a0, a1, L, tiny):
    """Curvature components along the averaged material axes and the integrated twist at the central node of a
    three-node stencil with segment axes a0 (x0 -> x1) and a1 (x1 -> x2); L is the reference Voronoi length."""
    e0 = x1 - x0
    e1 = x2 - x1
    e0_hat = func_rod_normalize(e0, tiny)
    e1_hat = func_rod_normalize(e1, tiny)
    e_avg = func_rod_normalize(e0_hat + e1_hat, tiny)
    a0_node = func_rod_parallel_transport(e0_hat, e_avg, tiny) @ a0
    a1_node = func_rod_parallel_transport(e1_hat, e_avg, tiny) @ a1
    a_avg = func_rod_normalize(a0_node + a1_node, tiny)
    b_avg = func_rod_normalize(e_avg.cross(a0_node) + e_avg.cross(a1_node), tiny)
    l_cur = 0.5 * (e0.norm() + e1.norm())
    k = (l_cur / (L * L)) * func_rod_curvature_binormal(e0_hat, e1_hat, tiny)
    tw = func_rod_curvature_binormal(a0_node, a1_node, tiny).dot(e_avg)
    return k.dot(a_avg), k.dot(b_avg), tw


@qd.func
def func_rod_bend_twist(
    x0,
    x1,
    x2,
    a0,
    a1,
    L,
    ka_ref,
    kb_ref,
    tw_ref,
    ka_ss,
    kb_ss,
    tw_ss,
    EI1,
    EI2,
    GJ,
    f,
    tiny,
    assem_dres,
):
    """Bending and twisting energy at the central node, the residual over the 11 stencil coordinates
    [x0, theta0, x1, theta1, x2] and its Gauss-Newton tangent. f is the stiffness damping factor beta / h; the *_ss are
    the stage-start measures and the *_ref the reference ones."""
    e0 = x1 - x0
    e1 = x2 - x1
    e0_norm = qd.max(e0.norm(), tiny)
    e1_norm = qd.max(e1.norm(), tiny)
    e0_hat = e0 / e0_norm
    e1_hat = e1 / e1_norm
    e_sum = e0_hat + e1_hat
    e_avg = func_rod_normalize(e_sum, tiny)
    PT0 = func_rod_parallel_transport(e0_hat, e_avg, tiny)
    PT1 = func_rod_parallel_transport(e1_hat, e_avg, tiny)
    a0_node = PT0 @ a0
    a1_node = PT1 @ a1
    b0_node = e_avg.cross(a0_node)
    b1_node = e_avg.cross(a1_node)
    a_sum = a0_node + a1_node
    b_sum = b0_node + b1_node
    a_avg = func_rod_normalize(a_sum, tiny)
    b_avg = func_rod_normalize(b_sum, tiny)
    l_cur = 0.5 * (e0_norm + e1_norm)
    L2_inv = 1.0 / (L * L)
    k_int = func_rod_curvature_binormal(e0_hat, e1_hat, tiny)
    k = (l_cur * L2_inv) * k_int
    ka = k.dot(a_avg)
    kb = k.dot(b_avg)
    a_bin = func_rod_curvature_binormal(a0_node, a1_node, tiny)
    tw = a_bin.dot(e_avg)

    scale = 1.0 + f
    ss_weight = f / scale
    ka_eff = (ka - ka_ref) - ss_weight * (ka_ss - ka_ref)
    kb_eff = (kb - kb_ref) - ss_weight * (kb_ss - kb_ref)
    tw_eff = (tw - tw_ref) - ss_weight * (tw_ss - tw_ref)
    EI1_eff = scale * EI1
    EI2_eff = scale * EI2
    GJ_eff = scale * GJ
    energy = 0.5 * (EI1_eff * ka_eff * ka_eff + EI2_eff * kb_eff * kb_eff) * L + 0.5 * GJ_eff * tw_eff * tw_eff / L

    # Derivatives of the measures with respect to e0, e1 (gradient vectors) and the two twists.
    I3 = qd.Matrix.identity(gs.qd_float, 3)
    de0_hat = (I3 - e0_hat.outer_product(e0_hat)) / e0_norm
    de1_hat = (I3 - e1_hat.outer_product(e1_hat)) / e1_norm
    de_avg = func_rod_dnormalize(e_sum, tiny)
    da0_de0_hat = a0.outer_product(e0_hat) - e0_hat.outer_product(a0)
    da1_de1_hat = a1.outer_product(e1_hat) - e1_hat.outer_product(a1)
    da0_dth0 = e0_hat.cross(a0)
    da1_dth1 = e1_hat.cross(a1)
    dPT0_n0, dPT0_n = func_rod_dparallel_transported(e0_hat, e_avg, a0, tiny)
    dPT1_n0, dPT1_n = func_rod_dparallel_transported(e1_hat, e_avg, a1, tiny)
    da0n_de1h = dPT0_n @ de_avg
    da1n_de0h = dPT1_n @ de_avg
    da0n_de0h = dPT0_n0 + da0n_de1h + PT0 @ da0_de0_hat
    da1n_de1h = dPT1_n0 + da1n_de0h + PT1 @ da1_de1_hat
    da0n_dth0 = PT0 @ da0_dth0
    da1n_dth1 = PT1 @ da1_dth1
    S_e = skew(e_avg)
    db0n_desum = (-skew(a0_node)) @ de_avg
    db1n_desum = (-skew(a1_node)) @ de_avg
    db0n_de0h = S_e @ da0n_de0h + db0n_desum
    db0n_de1h = S_e @ da0n_de1h + db0n_desum
    db1n_de0h = S_e @ da1n_de0h + db1n_desum
    db1n_de1h = S_e @ da1n_de1h + db1n_desum
    db0n_dth0 = S_e @ da0n_dth0
    db1n_dth1 = S_e @ da1n_dth1
    da_avg = func_rod_dnormalize(a_sum, tiny)
    db_avg = func_rod_dnormalize(b_sum, tiny)
    daavg_de0 = (da_avg @ (da0n_de0h + da1n_de0h)) @ de0_hat
    daavg_de1 = (da_avg @ (da0n_de1h + da1n_de1h)) @ de1_hat
    dbavg_de0 = (db_avg @ (db0n_de0h + db1n_de0h)) @ de0_hat
    dbavg_de1 = (db_avg @ (db0n_de1h + db1n_de1h)) @ de1_hat
    daavg_dth0 = da_avg @ da0n_dth0
    daavg_dth1 = da_avg @ da1n_dth1
    dbavg_dth0 = db_avg @ db0n_dth0
    dbavg_dth1 = db_avg @ db1n_dth1
    dkint_de0h, dkint_de1h = func_rod_dcurvature_binormal(e0_hat, e1_hat, tiny)
    dk_de0 = (l_cur * L2_inv) * (dkint_de0h @ de0_hat) + (0.5 * L2_inv) * k_int.outer_product(e0_hat)
    dk_de1 = (l_cur * L2_inv) * (dkint_de1h @ de1_hat) + (0.5 * L2_inv) * k_int.outer_product(e1_hat)
    dka_de0 = dk_de0.transpose() @ a_avg + daavg_de0.transpose() @ k
    dka_de1 = dk_de1.transpose() @ a_avg + daavg_de1.transpose() @ k
    dkb_de0 = dk_de0.transpose() @ b_avg + dbavg_de0.transpose() @ k
    dkb_de1 = dk_de1.transpose() @ b_avg + dbavg_de1.transpose() @ k
    dka_dth0 = k.dot(daavg_dth0)
    dka_dth1 = k.dot(daavg_dth1)
    dkb_dth0 = k.dot(dbavg_dth0)
    dkb_dth1 = k.dot(dbavg_dth1)
    dbin_da0n, dbin_da1n = func_rod_dcurvature_binormal(a0_node, a1_node, tiny)
    dbin_de0h = dbin_da0n @ da0n_de0h + dbin_da1n @ da1n_de0h
    dbin_de1h = dbin_da1n @ da1n_de1h + dbin_da0n @ da0n_de1h
    dtw_de0 = (dbin_de0h @ de0_hat).transpose() @ e_avg
    dtw_de1 = (dbin_de1h @ de1_hat).transpose() @ e_avg
    dtw_dth0 = (dbin_da0n @ da0n_dth0).dot(e_avg)
    dtw_dth1 = (dbin_da1n @ da1n_dth1).dot(e_avg)

    # Chain to the stencil coordinates [x0 (0..2), theta0 (3), x1 (4..6), theta1 (7), x2 (8..10)].
    Dka = qd.Vector.zero(gs.qd_float, 11)
    Dkb = qd.Vector.zero(gs.qd_float, 11)
    Dtw = qd.Vector.zero(gs.qd_float, 11)
    for i in qd.static(range(3)):
        Dka[i] = -dka_de0[i]
        Dka[4 + i] = dka_de0[i] - dka_de1[i]
        Dka[8 + i] = dka_de1[i]
        Dkb[i] = -dkb_de0[i]
        Dkb[4 + i] = dkb_de0[i] - dkb_de1[i]
        Dkb[8 + i] = dkb_de1[i]
        Dtw[i] = -dtw_de0[i]
        Dtw[4 + i] = dtw_de0[i] - dtw_de1[i]
        Dtw[8 + i] = dtw_de1[i]
    Dka[3] = dka_dth0
    Dka[7] = dka_dth1
    Dkb[3] = dkb_dth0
    Dkb[7] = dkb_dth1
    Dtw[3] = dtw_dth0
    Dtw[7] = dtw_dth1

    w_a = EI1_eff * L
    w_b = EI2_eff * L
    w_t = GJ_eff / L
    res = (w_a * ka_eff) * Dka + (w_b * kb_eff) * Dkb + (w_t * tw_eff) * Dtw
    K = qd.Matrix.zero(gs.qd_float, 11, 11)
    if assem_dres:
        K = w_a * Dka.outer_product(Dka) + w_b * Dkb.outer_product(Dkb) + w_t * Dtw.outer_product(Dtw)
    return energy, res, K, ka, kb, tw
