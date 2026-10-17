# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Thin shells: linear triangles with a Saint Venant-Kirchhoff membrane on the metric of the mid-surface and a
Koiter-type bending term on its discrete second fundamental form, assembled per triangle over a stencil of the three
vertices and the three opposite vertices of the neighboring triangles (extrapolated across boundary edges).

Two-by-two tensors are symmetric; the covariant strain measures are mixed tensors S = A^-1 (...) and the constitutive
response is Psi(S) = 1/2 lambda tr(S)^2 + mu tr(S S) with derivative lambda tr(S) A^-1 + 2 mu S A^-1 and constant
second derivative D2[k][l][m][n] = 2 mu A^-1[k][m] A^-1[l][n] + lambda A^-1[k][l] A^-1[m][n], contracted with the
derivatives of the metric or of the curvature (full double contractions, off-diagonals counted twice).
"""

import quadrants as qd

import genesis as gs

# d2 a / dx_a dx_b of the metric, identical for every spatial component (blocks indexed [a][b], a <= b).
_METRIC_HESSIAN = (
    ((2.0, 2.0, 2.0), (-2.0, -1.0, 0.0), (0.0, -1.0, -2.0)),
    (None, (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (None, None, (0.0, 0.0, 2.0)),
)


@qd.func
def func_colon(P, Q):
    """Full double contraction of two symmetric 2x2 tensors."""
    return P[0, 0] * Q[0, 0] + 2.0 * P[0, 1] * Q[0, 1] + P[1, 1] * Q[1, 1]


@qd.func
def func_sym2(a, b, c):
    return qd.Matrix([[a, b], [b, c]], dt=gs.qd_float)


@qd.func
def func_svk_energy(S, lam, mu):
    tr = S[0, 0] + S[1, 1]
    return 0.5 * lam * tr * tr + mu * (S @ S).trace()


@qd.func
def func_svk_stress(S, A_inv, lam, mu):
    """d Psi / d S contracted to the contravariant stress lambda tr(S) A^-1 + 2 mu S A^-1 (symmetric)."""
    tr = S[0, 0] + S[1, 1]
    T = lam * tr * A_inv + 2.0 * mu * (S @ A_inv)
    return func_sym2(T[0, 0], 0.5 * (T[0, 1] + T[1, 0]), T[1, 1])


@qd.func
def func_svk_tangent_dof(G, A_inv):
    """Per-dof factors of the tangent pairs: the metric product A^-1 G A^-1 (symmetric, three numbers) and the trace
    A^-1 : G, so that G1 : D2Psi : G2 = 2 mu G1 : (A^-1 G2 A^-1) + lam (A^-1 : G1)(A^-1 : G2) costs a handful of
    multiplications per pair instead of two matrix products."""
    H = A_inv @ G @ A_inv
    return qd.Vector([H[0, 0], H[0, 1], H[1, 1]], dt=gs.qd_float), func_colon(A_inv, G)


@qd.func
def func_project_psd_metric(S, A_inv, eps):
    """Clamp the eigenvalues of the symmetric 2x2 stress S in the metric A^-1 = L L^T: B = L^-1 S L^-T, eigenvalues of B
    clamped to at least eps, S' = L B' L^T."""
    l00 = qd.sqrt(qd.max(A_inv[0, 0], eps))
    l10 = A_inv[1, 0] / l00
    l11 = qd.sqrt(qd.max(A_inv[1, 1] - l10 * l10, eps))
    # L^-1 = [[1/l00, 0], [-l10/(l00 l11), 1/l11]]
    i00 = 1.0 / l00
    i10 = -l10 / (l00 * l11)
    i11 = 1.0 / l11
    L_inv = qd.Matrix([[i00, 0.0], [i10, i11]], dt=gs.qd_float)
    B = L_inv @ S @ L_inv.transpose()
    b00 = B[0, 0]
    b01 = 0.5 * (B[0, 1] + B[1, 0])
    b11 = B[1, 1]
    mean = 0.5 * (b00 + b11)
    radius = qd.sqrt(0.25 * (b00 - b11) ** 2 + b01 * b01)
    lam_min = mean - radius
    lam_max = mean + radius
    B_proj = B
    if lam_min < eps:
        # Eigenvector of the smallest eigenvalue.
        v = qd.Vector([b01, lam_min - b00], dt=gs.qd_float)
        if v.norm() < eps:
            v = qd.Vector([1.0, 0.0], dt=gs.qd_float)
        v = v / v.norm()
        u = qd.Vector([-v[1], v[0]], dt=gs.qd_float)
        B_proj = qd.max(lam_min, eps) * v.outer_product(v) + qd.max(lam_max, eps) * u.outer_product(u)
    L = qd.Matrix([[l00, 0.0], [l10, l11]], dt=gs.qd_float)
    T = L @ B_proj @ L.transpose()
    return func_sym2(T[0, 0], 0.5 * (T[0, 1] + T[1, 0]), T[1, 1])


@qd.func
def func_extrapolate_stencil(x, is_missing):
    """Complete the opposite vertices missing across boundary edges by flat continuation: x[3+v] = x[v1] + x[v2] - x[v]."""
    y = x
    for v0 in qd.static(range(3)):
        v1 = (v0 + 1) % 3
        v2 = (v0 + 2) % 3
        if is_missing[v0]:
            for k in qd.static(range(3)):
                y[3 + v0, k] = x[v1, k] + x[v2, k] - x[v0, k]
    return y


@qd.func
def func_row(M, r):
    return qd.Vector([M[r, 0], M[r, 1], M[r, 2]], dt=gs.qd_float)


@qd.func
def func_shell_metric(x):
    """Metric a of the triangle (x0, x1, x2) and its edges e0 = x1 - x0, e1 = x2 - x0."""
    e0 = func_row(x, 1) - func_row(x, 0)
    e1 = func_row(x, 2) - func_row(x, 0)
    return e0, e1, func_sym2(e0.dot(e0), e0.dot(e1), e1.dot(e1))


@qd.func
def func_shell_edges_normals(x, eps):
    """Edges v[0..5] of the stencil, unit normals of the central and the three neighboring triangles, and the averaged
    (Kelvin-inverted) normals across the three edges."""
    v = qd.Matrix.zero(gs.qd_float, 6, 3)
    for k in qd.static(range(3)):
        for i in qd.static(range(3)):
            v[k, i] = x[(k + 1) % 3, i] - x[k, i]
            v[k + 3, i] = x[k + 3, i] - x[(k + 1) % 3, i]
    n = qd.Matrix.zero(gs.qd_float, 4, 3)
    n0 = func_row(v, 2).cross(func_row(v, 0))
    n1 = func_row(v, 3).cross(func_row(v, 1))
    n2 = func_row(v, 4).cross(func_row(v, 2))
    n3 = func_row(v, 5).cross(func_row(v, 0))
    for i in qd.static(range(3)):
        n[0, i] = n0[i]
        n[1, i] = n1[i]
        n[2, i] = n2[i]
        n[3, i] = n3[i]
    n_hat = qd.Matrix.zero(gs.qd_float, 4, 3)
    for f in qd.static(range(4)):
        nf = func_row(n, f)
        nf = nf / qd.max(nf.norm(), eps)
        for i in qd.static(range(3)):
            n_hat[f, i] = nf[i]
    n_avg = qd.Matrix.zero(gs.qd_float, 3, 3)
    for f in qd.static(range(3)):
        m = 0.5 * (func_row(n_hat, 0) + func_row(n_hat, f + 1))
        m = m / qd.max(m.norm_sqr(), eps)
        for i in qd.static(range(3)):
            n_avg[f, i] = m[i]
    return v, n, n_hat, n_avg


@qd.func
def func_second_fundamental_form(v, n_avg):
    v0 = func_row(v, 0)
    v2 = func_row(v, 2)
    b00 = 2.0 * (func_row(n_avg, 1) - func_row(n_avg, 0)).dot(v0)
    b01 = 2.0 * func_row(n_avg, 0).dot(v2)
    b11 = 2.0 * (func_row(n_avg, 0) - func_row(n_avg, 2)).dot(v2)
    return func_sym2(b00, b01, b11)


@qd.func
def func_shell_rest_data(X, is_missing, eps):
    """Reference area, inverse metric and rest second fundamental form of a stencil (missing vertices extrapolated)."""
    Xe = func_extrapolate_stencil(X, is_missing)
    _E0, _E1, A = func_shell_metric(Xe)
    det_A = qd.max(A.determinant(), eps)
    area = 0.5 * qd.sqrt(det_A)
    A_inv = func_sym2(A[1, 1] / det_A, -A[0, 1] / det_A, A[0, 0] / det_A)
    V, _N, _N_hat, N_avg = func_shell_edges_normals(Xe, eps)
    B = func_second_fundamental_form(V, N_avg)
    return area, A_inv, B


@qd.func
def func_shell_strains(x, X, is_missing, A_inv, B, eps):
    """Mixed membrane strain A^-1 eps_flat (cancellation-free form) and mixed bending strain A^-1 (B - b) of a stencil."""
    xe = func_extrapolate_stencil(x, is_missing)
    Xe = func_extrapolate_stencil(X, is_missing)
    e0, e1, _a = func_shell_metric(xe)
    E0, E1, _A = func_shell_metric(Xe)
    du0 = e0 - E0
    du1 = e1 - E1
    eps_flat = func_sym2(
        E0.dot(du0) + 0.5 * du0.dot(du0),
        0.5 * (E0.dot(du1) + du0.dot(E1 + du1)),
        E1.dot(du1) + 0.5 * du1.dot(du1),
    )
    eps_m = A_inv @ eps_flat
    v, _n, _n_hat, n_avg = func_shell_edges_normals(xe, eps)
    b = func_second_fundamental_form(v, n_avg)
    s = A_inv @ (B - b)
    return eps_m, s


@qd.func
def func_shell_bending_gradients(v, n, n_hat, n_avg, eps):
    """Derivatives of the three components (b00, b01, b11) of the second fundamental form with respect to the 18
    coordinates of the (extrapolated) stencil, by forward propagation through the edges, normals and averages."""
    G = qd.Matrix.zero(gs.qd_float, 18, 3)
    v0 = func_row(v, 0)
    v1 = func_row(v, 1)
    v2 = func_row(v, 2)
    v3 = func_row(v, 3)
    v4 = func_row(v, 4)
    v5 = func_row(v, 5)
    for a in qd.static(range(6)):
        for i in qd.static(range(3)):
            d = qd.Vector.zero(gs.qd_float, 3)
            d[i] = 1.0
            # Edge perturbations.
            dv = qd.Matrix.zero(gs.qd_float, 6, 3)
            for k in qd.static(range(3)):
                if qd.static(a == (k + 1) % 3):
                    for c in qd.static(range(3)):
                        dv[k, c] += d[c]
                if qd.static(a == k):
                    for c in qd.static(range(3)):
                        dv[k, c] -= d[c]
                if qd.static(a == k + 3):
                    for c in qd.static(range(3)):
                        dv[k + 3, c] += d[c]
                if qd.static(a == (k + 1) % 3):
                    for c in qd.static(range(3)):
                        dv[k + 3, c] -= d[c]
            dv0 = func_row(dv, 0)
            dv1 = func_row(dv, 1)
            dv2 = func_row(dv, 2)
            dv3 = func_row(dv, 3)
            dv4 = func_row(dv, 4)
            dv5 = func_row(dv, 5)
            # Normal perturbations.
            dn = qd.Matrix.zero(gs.qd_float, 4, 3)
            dn0 = dv2.cross(v0) + v2.cross(dv0)
            dn1 = dv3.cross(v1) + v3.cross(dv1)
            dn2 = dv4.cross(v2) + v4.cross(dv2)
            dn3 = dv5.cross(v0) + v5.cross(dv0)
            for c in qd.static(range(3)):
                dn[0, c] = dn0[c]
                dn[1, c] = dn1[c]
                dn[2, c] = dn2[c]
                dn[3, c] = dn3[c]
            dn_hat = qd.Matrix.zero(gs.qd_float, 4, 3)
            for f in qd.static(range(4)):
                nf = func_row(n, f)
                nh = func_row(n_hat, f)
                dnf = func_row(dn, f)
                dh = (dnf - nh * nh.dot(dnf)) / qd.max(nf.norm(), eps)
                for c in qd.static(range(3)):
                    dn_hat[f, c] = dh[c]
            dn_avg = qd.Matrix.zero(gs.qd_float, 3, 3)
            for f in qd.static(range(3)):
                m = 0.5 * (func_row(n_hat, 0) + func_row(n_hat, f + 1))
                dm = 0.5 * (func_row(dn_hat, 0) + func_row(dn_hat, f + 1))
                m_sq = qd.max(m.norm_sqr(), eps)
                da = dm / m_sq - (2.0 * m.dot(dm) / (m_sq * m_sq)) * m
                for c in qd.static(range(3)):
                    dn_avg[f, c] = da[c]
            na0 = func_row(n_avg, 0)
            na1 = func_row(n_avg, 1)
            na2 = func_row(n_avg, 2)
            dna0 = func_row(dn_avg, 0)
            dna1 = func_row(dn_avg, 1)
            dna2 = func_row(dn_avg, 2)
            G[3 * a + i, 0] = 2.0 * ((dna1 - dna0).dot(v0) + (na1 - na0).dot(dv0))
            G[3 * a + i, 1] = 2.0 * (dna0.dot(v2) + na0.dot(dv2))
            G[3 * a + i, 2] = 2.0 * ((dna0 - dna2).dot(v2) + (na0 - na2).dot(dv2))
    return G


@qd.func
def func_fold_missing(G, is_missing):
    """Redistribute the derivative rows of the extrapolated vertices onto the triangle vertices (x[3+v] = x[v1] + x[v2]
    - x[v]) and zero them."""
    H = G
    for v0 in qd.static(range(3)):
        v1 = (v0 + 1) % 3
        v2 = (v0 + 2) % 3
        if is_missing[v0]:
            for i in qd.static(range(3)):
                for c in qd.static(range(G.m)):
                    g = H[3 * (3 + v0) + i, c]
                    H[3 * v1 + i, c] += g
                    H[3 * v2 + i, c] += g
                    H[3 * v0 + i, c] -= g
                    H[3 * (3 + v0) + i, c] = 0.0
    return H


@qd.func
def func_shell_elastic(
    x,
    X,
    is_missing,
    area,
    A_inv,
    B,
    lam,
    mu,
    alpha,
    beta,
    ss_weight,
    eps_ss,
    s_ss,
    eps,
    project: qd.template(),
    assem_dres,
):
    """Membrane and bending energy of a stencil, the residual on its 18 coordinates and the 18x18 tangent (membrane:
    material term plus the metric-projected geometric term; bending: material term only)."""
    xe = func_extrapolate_stencil(x, is_missing)
    Xe = func_extrapolate_stencil(X, is_missing)
    e0, e1, _a = func_shell_metric(xe)
    E0, E1, _A = func_shell_metric(Xe)
    du0 = e0 - E0
    du1 = e1 - E1
    eps_flat = func_sym2(
        E0.dot(du0) + 0.5 * du0.dot(du0),
        0.5 * (E0.dot(du1) + du0.dot(E1 + du1)),
        E1.dot(du1) + 0.5 * du1.dot(du1),
    )
    eps_m = A_inv @ eps_flat - ss_weight * eps_ss
    energy = area * func_svk_energy(eps_m, lam, mu)
    dpsi_da = 0.5 * func_svk_stress(eps_m, A_inv, lam, mu)

    res = qd.Vector.zero(gs.qd_float, 18)
    K = qd.Matrix.zero(gs.qd_float, 18, 18)
    # Metric derivatives da/dx_a[i], a = 0..2, as symmetric 2x2 tensors (stored per dof).
    Ga = qd.Matrix.zero(gs.qd_float, 9, 3)
    for i in qd.static(range(3)):
        Ga[i, 0] = -2.0 * e0[i]
        Ga[i, 1] = -(e0[i] + e1[i])
        Ga[i, 2] = -2.0 * e1[i]
        Ga[3 + i, 0] = 2.0 * e0[i]
        Ga[3 + i, 1] = e1[i]
        Ga[3 + i, 2] = 0.0
        Ga[6 + i, 0] = 0.0
        Ga[6 + i, 1] = e0[i]
        Ga[6 + i, 2] = 2.0 * e1[i]
    for p in qd.static(range(9)):
        res[p] += area * func_colon(func_sym2(Ga[p, 0], Ga[p, 1], Ga[p, 2]), dpsi_da)
    if assem_dres:
        S_proj = dpsi_da
        if qd.static(project):
            S_proj = func_project_psd_metric(dpsi_da, A_inv, eps)
        # G_p : D2Psi : G_q over the pairs of dofs from per-dof factors (see func_svk_tangent_dof)
        Ha = qd.Matrix.zero(gs.qd_float, 9, 3)
        ta = qd.Vector.zero(gs.qd_float, 9)
        for p in qd.static(range(9)):
            h, t = func_svk_tangent_dof(func_sym2(Ga[p, 0], Ga[p, 1], Ga[p, 2]), A_inv)
            for k in qd.static(range(3)):
                Ha[p, k] = h[k]
            ta[p] = t
        for p in qd.static(range(9)):
            for q in qd.static(range(p, 9)):
                colon = Ga[p, 0] * Ha[q, 0] + 2.0 * Ga[p, 1] * Ha[q, 1] + Ga[p, 2] * Ha[q, 2]
                K[p, q] += 0.25 * area * (2.0 * mu * colon + lam * ta[p] * ta[q])
        for na in qd.static(range(3)):
            for nb in qd.static(range(na, 3)):
                h = qd.static(_METRIC_HESSIAN[na][nb])
                value = area * func_colon(func_sym2(h[0], h[1], h[2]), S_proj)
                for i in qd.static(range(3)):
                    K[3 * na + i, 3 * nb + i] += value

    # Bending.
    v, n, n_hat, n_avg = func_shell_edges_normals(xe, eps)
    b = func_second_fundamental_form(v, n_avg)
    s = A_inv @ (B - b) - ss_weight * s_ss
    energy += area * func_svk_energy(s, alpha, 0.5 * beta)
    dpsi_db = -func_svk_stress(s, A_inv, alpha, 0.5 * beta)
    Gb = func_fold_missing(func_shell_bending_gradients(v, n, n_hat, n_avg, eps), is_missing)
    for p in qd.static(range(18)):
        res[p] += area * func_colon(func_sym2(Gb[p, 0], Gb[p, 1], Gb[p, 2]), dpsi_db)
    if assem_dres:
        Hb = qd.Matrix.zero(gs.qd_float, 18, 3)
        tb = qd.Vector.zero(gs.qd_float, 18)
        for p in qd.static(range(18)):
            h, t = func_svk_tangent_dof(func_sym2(Gb[p, 0], Gb[p, 1], Gb[p, 2]), A_inv)
            for k in qd.static(range(3)):
                Hb[p, k] = h[k]
            tb[p] = t
        for p in qd.static(range(18)):
            for q in qd.static(range(p, 18)):
                colon = Gb[p, 0] * Hb[q, 0] + 2.0 * Gb[p, 1] * Hb[q, 1] + Gb[p, 2] * Hb[q, 2]
                K[p, q] += area * (beta * colon + alpha * tb[p] * tb[q])
        for p in qd.static(range(18)):
            for q in qd.static(range(p + 1, 18)):
                K[q, p] = K[p, q]
    return energy, res, K, eps_m, s
