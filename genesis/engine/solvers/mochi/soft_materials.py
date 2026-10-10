# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Constitutive models of the deformable bodies: energy density, first Piola-Kirchhoff stress and the
positive-semidefinite projected tangent d2Psi/dF2 of linear tetrahedra, plus the Kelvin-Voigt stiffness damping.

Tangents are 9x9 matrices over the row-major flattening of F (index 3 i + j for F[i, j]). The projection clamps the
eigenvalues of the analytic eigensystem of the tangent (Smith et al. 2019): three "scaling" modes u_i v_i^T mixed by a
symmetric 3x3 matrix, and the "twist" u_j v_k^T - u_k v_j^T and "flip" u_j v_k^T + u_k v_j^T modes, whose unnormalized
matrices carry a factor 1/2 on their eigenvalue.
"""

from enum import IntEnum

import quadrants as qd

import genesis as gs

from .lie import sym_eig3


class ELASTIC_MODEL(IntEnum):
    STABLE_NEOHOOKEAN = 0
    STVK = 1
    LINEAR = 2


ELASTIC_MODEL_BY_NAME = {
    "stable_neohookean": ELASTIC_MODEL.STABLE_NEOHOOKEAN,
    "stvk": ELASTIC_MODEL.STVK,
    "linear": ELASTIC_MODEL.LINEAR,
}

# (j, k) pairs of the twist and flip modes n = 0, 1, 2.
_MODE_PAIRS = ((1, 2), (0, 2), (0, 1))


@qd.func
def func_flatten(M):
    """Row-major 9-vector of a 3x3 matrix."""
    v = qd.Vector.zero(gs.qd_float, 9)
    for i, j in qd.static(qd.ndrange(3, 3)):
        v[3 * i + j] = M[i, j]
    return v


@qd.func
def func_cofactor(F):
    """Cofactor matrix dJ/dF of F."""
    f0 = qd.Vector([F[0, 0], F[1, 0], F[2, 0]], dt=gs.qd_float)
    f1 = qd.Vector([F[0, 1], F[1, 1], F[2, 1]], dt=gs.qd_float)
    f2 = qd.Vector([F[0, 2], F[1, 2], F[2, 2]], dt=gs.qd_float)
    return qd.Matrix.cols([f1.cross(f2), f2.cross(f0), f0.cross(f1)])


@qd.func
def func_max_eigenvalue_sym3(G):
    """Largest eigenvalue of a symmetric 3x3 matrix (trigonometric closed form)."""
    q = G.trace() / 3.0
    p1 = G[0, 1] ** 2 + G[0, 2] ** 2 + G[1, 2] ** 2
    p2 = (G[0, 0] - q) ** 2 + (G[1, 1] - q) ** 2 + (G[2, 2] - q) ** 2 + 2.0 * p1
    lam_max = q
    if p2 > 0.0:
        p = qd.sqrt(p2 / 6.0)
        B = (G - q * qd.Matrix.identity(gs.qd_float, 3)) / p
        r = qd.math.clamp(0.5 * B.determinant(), -1.0, 1.0)
        lam_max = q + 2.0 * p * qd.cos(qd.acos(r) / 3.0)
    return lam_max


@qd.func
def func_rotation_variant_svd(F):
    """SVD F = U diag(sigma) V^T with det(U) >= 0 and det(V) >= 0, so that sigma[2] < 0 exactly when F is inverted."""
    U, S, V = qd.svd(F)
    sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float)
    if U.determinant() < 0.0:
        for i in qd.static(range(3)):
            U[i, 2] = -U[i, 2]
        sigma[2] = -sigma[2]
    if V.determinant() < 0.0:
        for i in qd.static(range(3)):
            V[i, 2] = -V[i, 2]
        sigma[2] = -sigma[2]
    return U, sigma, V


@qd.func
def func_tangent_from_eigensystem(U, V, A, twist, flip, eps, project: qd.template()):
    """Assemble the 9x9 tangent from the scaling matrix A (in the u_i v_i^T basis) and the twist and flip eigenvalues,
    clamping every eigenvalue to at least eps when projecting."""
    evals, Q = sym_eig3(A, n_sweeps=8)
    C = qd.Matrix.zero(gs.qd_float, 9, 9)
    for n in qd.static(range(3)):
        lam_n = evals[n]
        if qd.static(project):
            lam_n = qd.max(lam_n, eps)
        M = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            u_i = qd.Vector([U[0, i], U[1, i], U[2, i]], dt=gs.qd_float)
            v_i = qd.Vector([V[0, i], V[1, i], V[2, i]], dt=gs.qd_float)
            M += Q[i, n] * u_i.outer_product(v_i)
        m = func_flatten(M)
        C += lam_n * m.outer_product(m)
    for n in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[n])
        u_j = qd.Vector([U[0, j], U[1, j], U[2, j]], dt=gs.qd_float)
        u_k = qd.Vector([U[0, k], U[1, k], U[2, k]], dt=gs.qd_float)
        v_j = qd.Vector([V[0, j], V[1, j], V[2, j]], dt=gs.qd_float)
        v_k = qd.Vector([V[0, k], V[1, k], V[2, k]], dt=gs.qd_float)
        t = func_flatten(u_j.outer_product(v_k) - u_k.outer_product(v_j))
        fl = func_flatten(u_j.outer_product(v_k) + u_k.outer_product(v_j))
        lam_t = twist[n]
        lam_f = flip[n]
        if qd.static(project):
            lam_t = qd.max(lam_t, eps)
            lam_f = qd.max(lam_f, eps)
        C += (0.5 * lam_t) * t.outer_product(t)
        C += (0.5 * lam_f) * fl.outer_product(fl)
    return C


# ------------------------------------------------------------------------------------
# Stable neo-Hookean (Smith et al. 2018), in the reparametrization mu' = 4/3 mu, lambda' = lambda + 5/6 mu.
# ------------------------------------------------------------------------------------


@qd.func
def func_smith_params(mu, lam):
    mu_hat = 4.0 / 3.0 * mu
    lam_hat = lam + 5.0 / 6.0 * mu
    alpha = 1.0 + 0.75 * mu_hat / lam_hat
    return mu_hat, lam_hat, alpha


@qd.func
def func_smith_nh_energy(F, mu, lam):
    mu_hat, lam_hat, alpha = func_smith_params(mu, lam)
    Ic = F.norm_sqr()
    J = F.determinant()
    return 0.5 * mu_hat * (Ic - 3.0) + 0.5 * lam_hat * (J - alpha) ** 2 - 0.5 * mu_hat * qd.log(Ic + 1.0)


@qd.func
def func_smith_nh_stress(F, mu, lam):
    mu_hat, lam_hat, alpha = func_smith_params(mu, lam)
    Ic = F.norm_sqr()
    J = F.determinant()
    return mu_hat * (1.0 - 1.0 / (Ic + 1.0)) * F + lam_hat * (J - alpha) * func_cofactor(F)


@qd.func
def func_smith_nh_direct_tangent(F, mu_hat, lam_hat, alpha, Ic, J):
    """Exact tangent d2Psi/dF2 of the Smith neo-Hookean energy (row-major 9x9):
    c3 f f^T + lam_hat cof cof^T + lam_hat (J - alpha) d2J/dF2 + c2 I with c2 = mu_hat (1 - 1/(Ic+1)) and
    c3 = 2 mu_hat/(Ic+1)^2; d2J/dF2 pairs the columns a != b of F through the skew matrix of the third column."""
    Ic1 = Ic + 1.0
    c2 = mu_hat * (1.0 - 1.0 / Ic1)
    c3 = 2.0 * mu_hat / (Ic1 * Ic1)
    f = func_flatten(F)
    cof = func_flatten(func_cofactor(F))
    C = qd.Matrix.zero(gs.qd_float, 9, 9)
    for p in qd.static(range(9)):
        for q in qd.static(range(9)):
            C[p, q] = c3 * f[p] * f[q] + lam_hat * cof[p] * cof[q]
        C[p, p] += c2
    coeff = lam_hat * (J - alpha)
    for a in qd.static(range(3)):
        for b in qd.static(range(3)):
            if qd.static(a != b):
                c = qd.static(3 - a - b)
                # -eps_abc: the derivative of column a's cofactor with respect to column b.
                sign = qd.static(-1.0 if (a, b) in ((0, 1), (1, 2), (2, 0)) else 1.0)
                v0, v1, v2 = F[0, c], F[1, c], F[2, c]
                C[3 * 0 + a, 3 * 1 + b] += coeff * sign * (-v2)
                C[3 * 0 + a, 3 * 2 + b] += coeff * sign * v1
                C[3 * 1 + a, 3 * 0 + b] += coeff * sign * v2
                C[3 * 1 + a, 3 * 2 + b] += coeff * sign * (-v0)
                C[3 * 2 + a, 3 * 0 + b] += coeff * sign * (-v1)
                C[3 * 2 + a, 3 * 1 + b] += coeff * sign * v0
    return C


@qd.func
def func_smith_nh_tangent(F, mu, lam, eps, project: qd.template()):
    """Newton tangent of the Smith neo-Hookean energy. Only the twist and flip modes can make it indefinite, with
    eigenvalues +-lam_hat (J - alpha) sigma_i + mu_hat_k, mu_hat_k = mu_hat (1 - 1/(Ic+1)); when J >= 0 and
    lam_hat^2 (J - alpha)^2 max_i sigma_i^2 <= mu_hat_k^2 the exact tangent is positive semidefinite and is returned
    directly, without any decomposition, as in mochi's PSD oracle (max_i sigma_i^2 is the largest eigenvalue of F^T F).
    The eigensystem path projects the remaining elements."""
    mu_hat, lam_hat, alpha = func_smith_params(mu, lam)
    Ic = F.norm_sqr()
    J = F.determinant()
    C = qd.Matrix.zero(gs.qd_float, 9, 9)
    use_direct = True
    if qd.static(project):
        use_direct = func_smith_nh_is_psd(F, mu_hat, lam_hat, alpha, Ic, J)
    if use_direct:
        C = func_smith_nh_direct_tangent(F, mu_hat, lam_hat, alpha, Ic, J)
    else:
        U, V, A, twist, flip = func_smith_nh_eigensystem(F, mu_hat, lam_hat, alpha)
        C = func_tangent_from_eigensystem(U, V, A, twist, flip, eps, project)
    return C


@qd.func
def func_smith_nh_is_psd(F, mu_hat, lam_hat, alpha, Ic, J):
    """mochi's oracle: only the twist and flip modes can make the Smith neo-Hookean tangent indefinite, with
    eigenvalues +-lam_hat (J - alpha) sigma_i + mu_hat_k, mu_hat_k = mu_hat (1 - 1/(Ic+1)); when J >= 0 and
    lam_hat^2 (J - alpha)^2 max_i sigma_i^2 <= mu_hat_k^2 the exact tangent is positive semidefinite (max_i sigma_i^2 is
    the largest eigenvalue of F^T F)."""
    mu_hat_k = mu_hat * (1.0 - 1.0 / (Ic + 1.0))
    max_sigma_sq = func_max_eigenvalue_sym3(F.transpose() @ F)
    return (lam_hat * (J - alpha)) ** 2 * max_sigma_sq <= mu_hat_k * mu_hat_k and J >= 0.0


@qd.func
def func_smith_nh_eigensystem(F, mu_hat, lam_hat, alpha):
    """Analytic eigensystem of the Smith neo-Hookean tangent: the rotation-variant SVD, the scaling matrix in the
    u_i v_i^T basis and the twist and flip eigenvalues."""
    U, sigma, V = func_rotation_variant_svd(F)
    Ic = sigma.norm_sqr()
    J = sigma[0] * sigma[1] * sigma[2]
    Ic1 = Ic + 1.0
    coeff0 = mu_hat * (1.0 - 1.0 / Ic1)
    coeff1 = lam_hat * (J - alpha)
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[i])
        A[i, i] = (
            (2.0 * mu_hat * sigma[i] ** 2 - mu_hat * Ic1) / (Ic1 * Ic1)
            + lam_hat * sigma[j] ** 2 * sigma[k] ** 2
            + mu_hat
        )
    for i in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[i])
        # The off-diagonal (j, k) couples the modes other than i through sigma_i.
        A[j, k] = (2.0 * J - alpha) * lam_hat * sigma[i] + 2.0 * mu_hat * sigma[j] * sigma[k] / (Ic1 * Ic1)
        A[k, j] = A[j, k]
    twist = qd.Vector.zero(gs.qd_float, 3)
    flip = qd.Vector.zero(gs.qd_float, 3)
    for n in qd.static(range(3)):
        twist[n] = coeff1 * sigma[n] + coeff0
        flip[n] = -coeff1 * sigma[n] + coeff0
    return U, V, A, twist, flip


# ------------------------------------------------------------------------------------
# Saint Venant-Kirchhoff
# ------------------------------------------------------------------------------------


@qd.func
def func_green_strain(F):
    return 0.5 * (F.transpose() @ F - qd.Matrix.identity(gs.qd_float, 3))


@qd.func
def func_stvk_energy(F, mu, lam):
    G = func_green_strain(F)
    return mu * G.norm_sqr() + 0.5 * lam * G.trace() ** 2


@qd.func
def func_stvk_stress(F, mu, lam):
    G = func_green_strain(F)
    S = 2.0 * mu * G + lam * G.trace() * qd.Matrix.identity(gs.qd_float, 3)
    return F @ S


@qd.func
def func_stvk_tangent(F, mu, lam, eps, project: qd.template()):
    U, V, A, twist, flip = func_stvk_eigensystem(F, mu, lam)
    return func_tangent_from_eigensystem(U, V, A, twist, flip, eps, project)


@qd.func
def func_stvk_eigensystem(F, mu, lam):
    """Analytic eigensystem of the Saint Venant-Kirchhoff tangent (see func_smith_nh_eigensystem)."""
    U, sigma, V = func_rotation_variant_svd(F)
    I2 = sigma.norm_sqr()
    base = -mu + 0.5 * lam * (I2 - 3.0)
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        A[i, i] = base + (lam + 3.0 * mu) * sigma[i] ** 2
    for i in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[i])
        A[j, k] = lam * sigma[j] * sigma[k]
        A[k, j] = A[j, k]
    twist = qd.Vector.zero(gs.qd_float, 3)
    flip = qd.Vector.zero(gs.qd_float, 3)
    for n in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[n])
        twist[n] = base + mu * (sigma[j] ** 2 + sigma[k] ** 2 - sigma[j] * sigma[k])
        flip[n] = base + mu * (sigma[j] ** 2 + sigma[k] ** 2 + sigma[j] * sigma[k])
    return U, V, A, twist, flip


# ------------------------------------------------------------------------------------
# Linear elasticity (small strain)
# ------------------------------------------------------------------------------------


@qd.func
def func_linear_strain(F):
    return 0.5 * (F + F.transpose()) - qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def func_linear_energy(F, mu, lam):
    e = func_linear_strain(F)
    return mu * e.norm_sqr() + 0.5 * lam * e.trace() ** 2


@qd.func
def func_linear_stress(F, mu, lam):
    e = func_linear_strain(F)
    return 2.0 * mu * e + lam * e.trace() * qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def func_linear_tangent(mu, lam):
    """C_ijkl = lambda d_ij d_kl + mu (d_ik d_jl + d_il d_jk), constant and positive definite."""
    C = qd.Matrix.zero(gs.qd_float, 9, 9)
    for i, j in qd.static(qd.ndrange(3, 3)):
        C[3 * i + i, 3 * j + j] += lam
        C[3 * i + j, 3 * i + j] += mu
        C[3 * i + j, 3 * j + i] += mu
    return C


# ------------------------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------------------------


@qd.func
def func_elastic_energy(model, F, mu, lam):
    energy = gs.qd_float(0.0)
    if model == ELASTIC_MODEL.STABLE_NEOHOOKEAN:
        energy = func_smith_nh_energy(F, mu, lam)
    elif model == ELASTIC_MODEL.STVK:
        energy = func_stvk_energy(F, mu, lam)
    else:
        energy = func_linear_energy(F, mu, lam)
    return energy


@qd.func
def func_elastic_stress(model, F, mu, lam):
    P = qd.Matrix.zero(gs.qd_float, 3, 3)
    if model == ELASTIC_MODEL.STABLE_NEOHOOKEAN:
        P = func_smith_nh_stress(F, mu, lam)
    elif model == ELASTIC_MODEL.STVK:
        P = func_stvk_stress(F, mu, lam)
    else:
        P = func_linear_stress(F, mu, lam)
    return P


@qd.func
def func_elastic_tangent(model, F, mu, lam, eps, project: qd.template()):
    C = qd.Matrix.zero(gs.qd_float, 9, 9)
    if model == ELASTIC_MODEL.STABLE_NEOHOOKEAN:
        C = func_smith_nh_tangent(F, mu, lam, eps, project)
    elif model == ELASTIC_MODEL.STVK:
        C = func_stvk_tangent(F, mu, lam, eps, project)
    else:
        C = func_linear_tangent(mu, lam)
    return C


# ------------------------------------------------------------------------------------
# Element stiffness as node blocks
# ------------------------------------------------------------------------------------


@qd.func
def func_add_mode_blocks(K, lam_n, M, grads, vol):
    """Add vol lam_n (M g_f)(M g_g)^T to every node block (f, g) of the 12x12 element stiffness: the contribution of one
    tangent eigenmode M (3x3, eigenvalue lam_n) contracted with the shape gradients, since
    g_f^T (m m^T)_{rc} g_g = (M g_f)_r (M g_g)_c for the row-major flattening m of M."""
    p = qd.Matrix.zero(gs.qd_float, 4, 3)
    for f in qd.static(range(4)):
        g_f = qd.Vector([grads[f, 0], grads[f, 1], grads[f, 2]], dt=gs.qd_float)
        Mg = M @ g_f
        for r in qd.static(range(3)):
            p[f, r] = Mg[r]
    # upper node blocks only (f <= g); func_blocks_from_eigensystem mirrors them once after the last mode
    scale = vol * lam_n
    for f in qd.static(range(4)):
        for g in qd.static(range(f, 4)):
            for r in qd.static(range(3)):
                for c in qd.static(range(3)):
                    K[3 * f + r, 3 * g + c] += scale * p[f, r] * p[g, c]
    return K


@qd.func
def func_blocks_from_eigensystem(U, V, A, twist, flip, eps, project: qd.template(), grads, vol):
    """Element stiffness vol g_f^T C g_g of a tangent given by its analytic eigensystem (see
    func_tangent_from_eigensystem), accumulated mode by mode without forming the 9x9 tangent."""
    K = qd.Matrix.zero(gs.qd_float, 12, 12)
    evals, Q = sym_eig3(A, n_sweeps=8)
    for n in qd.static(range(3)):
        lam_n = evals[n]
        if qd.static(project):
            lam_n = qd.max(lam_n, eps)
        M = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            u_i = qd.Vector([U[0, i], U[1, i], U[2, i]], dt=gs.qd_float)
            v_i = qd.Vector([V[0, i], V[1, i], V[2, i]], dt=gs.qd_float)
            M += Q[i, n] * u_i.outer_product(v_i)
        K = func_add_mode_blocks(K, lam_n, M, grads, vol)
    for n in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[n])
        u_j = qd.Vector([U[0, j], U[1, j], U[2, j]], dt=gs.qd_float)
        u_k = qd.Vector([U[0, k], U[1, k], U[2, k]], dt=gs.qd_float)
        v_j = qd.Vector([V[0, j], V[1, j], V[2, j]], dt=gs.qd_float)
        v_k = qd.Vector([V[0, k], V[1, k], V[2, k]], dt=gs.qd_float)
        T = u_j.outer_product(v_k) - u_k.outer_product(v_j)
        Fl = u_j.outer_product(v_k) + u_k.outer_product(v_j)
        lam_t = twist[n]
        lam_f = flip[n]
        if qd.static(project):
            lam_t = qd.max(lam_t, eps)
            lam_f = qd.max(lam_f, eps)
        K = func_add_mode_blocks(K, 0.5 * lam_t, T, grads, vol)
        K = func_add_mode_blocks(K, 0.5 * lam_f, Fl, grads, vol)
    for f in qd.static(range(4)):
        for g in qd.static(range(f + 1, 4)):
            for r in qd.static(range(3)):
                for c in qd.static(range(3)):
                    K[3 * g + c, 3 * f + r] = K[3 * f + r, 3 * g + c]
    return K


@qd.func
def func_smith_nh_direct_blocks(F, mu_hat, lam_hat, alpha, Ic, J, grads, vol):
    """Element stiffness of the exact Smith neo-Hookean tangent c3 f f^T + lam_hat cof cof^T + c2 I
    + lam_hat (J - alpha) d2J/dF2 in closed form per node block (f, g):
    vol [c3 (F g_f)(F g_g)^T + lam_hat (cof g_f)(cof g_g)^T + c2 (g_f . g_g) I + coeff S(F (g_f x g_g))], where
    S(w)_rc = eps_rck w_k is the contraction of d2J/dF2 = eps_rck eps_mnl F_kl with the two gradients."""
    Ic1 = Ic + 1.0
    c2 = mu_hat * (1.0 - 1.0 / Ic1)
    c3 = 2.0 * mu_hat / (Ic1 * Ic1)
    coeff = lam_hat * (J - alpha)
    cof = func_cofactor(F)
    a = qd.Matrix.zero(gs.qd_float, 4, 3)
    b = qd.Matrix.zero(gs.qd_float, 4, 3)
    for f in qd.static(range(4)):
        g_f = qd.Vector([grads[f, 0], grads[f, 1], grads[f, 2]], dt=gs.qd_float)
        Fg = F @ g_f
        cg = cof @ g_f
        for r in qd.static(range(3)):
            a[f, r] = Fg[r]
            b[f, r] = cg[r]
    K = qd.Matrix.zero(gs.qd_float, 12, 12)
    for f in qd.static(range(4)):
        g_f = qd.Vector([grads[f, 0], grads[f, 1], grads[f, 2]], dt=gs.qd_float)
        for g in qd.static(range(f, 4)):
            g_g = qd.Vector([grads[g, 0], grads[g, 1], grads[g, 2]], dt=gs.qd_float)
            gg = g_f.dot(g_g)
            block = qd.Matrix.zero(gs.qd_float, 3, 3)
            for r in qd.static(range(3)):
                for c in qd.static(range(3)):
                    block[r, c] = c3 * a[f, r] * a[g, c] + lam_hat * b[f, r] * b[g, c]
                block[r, r] += c2 * gg
            if qd.static(f != g):
                w = F @ g_f.cross(g_g)
                block[0, 1] += coeff * w[2]
                block[0, 2] -= coeff * w[1]
                block[1, 0] -= coeff * w[2]
                block[1, 2] += coeff * w[0]
                block[2, 0] += coeff * w[1]
                block[2, 1] -= coeff * w[0]
            for r in qd.static(range(3)):
                for c in qd.static(range(3)):
                    K[3 * f + r, 3 * g + c] = vol * block[r, c]
                    if qd.static(f != g):
                        K[3 * g + c, 3 * f + r] = vol * block[r, c]
    return K


@qd.func
def func_linear_blocks(mu, lam, grads, vol):
    """Element stiffness of the constant linear tangent: vol [lam g_f g_g^T + mu (g_f . g_g) I + mu g_g g_f^T]."""
    K = qd.Matrix.zero(gs.qd_float, 12, 12)
    for f in qd.static(range(4)):
        g_f = qd.Vector([grads[f, 0], grads[f, 1], grads[f, 2]], dt=gs.qd_float)
        for g in qd.static(range(4)):
            g_g = qd.Vector([grads[g, 0], grads[g, 1], grads[g, 2]], dt=gs.qd_float)
            gg = g_f.dot(g_g)
            for r in qd.static(range(3)):
                for c in qd.static(range(3)):
                    value = lam * g_f[r] * g_g[c] + mu * g_g[r] * g_f[c]
                    if qd.static(r == c):
                        value += mu * gg
                    K[3 * f + r, 3 * g + c] = vol * value
    return K


@qd.func
def func_tet_stiffness(model, F, mu, lam, eps, project: qd.template(), grads, vol):
    """Elastic part of the 12x12 stiffness of a linear tetrahedron, vol g_f^T (d2Psi/dF2) g_g per node block, with the
    projected tangent of func_elastic_tangent but assembled block by block: the Smith neo-Hookean tangent in closed
    form when mochi's oracle proves it definite, the analytic eigenmodes otherwise (and for Saint Venant-Kirchhoff),
    the constant linear tangent directly."""
    K = qd.Matrix.zero(gs.qd_float, 12, 12)
    if model == ELASTIC_MODEL.STABLE_NEOHOOKEAN:
        mu_hat, lam_hat, alpha = func_smith_params(mu, lam)
        Ic = F.norm_sqr()
        J = F.determinant()
        use_direct = True
        if qd.static(project):
            use_direct = func_smith_nh_is_psd(F, mu_hat, lam_hat, alpha, Ic, J)
        if use_direct:
            K = func_smith_nh_direct_blocks(F, mu_hat, lam_hat, alpha, Ic, J, grads, vol)
        else:
            U, V, A, twist, flip = func_smith_nh_eigensystem(F, mu_hat, lam_hat, alpha)
            K = func_blocks_from_eigensystem(U, V, A, twist, flip, eps, project, grads, vol)
    elif model == ELASTIC_MODEL.STVK:
        U, V, A, twist, flip = func_stvk_eigensystem(F, mu, lam)
        K = func_blocks_from_eigensystem(U, V, A, twist, flip, eps, project, grads, vol)
    else:
        K = func_linear_blocks(mu, lam, grads, vol)
    return K


# ------------------------------------------------------------------------------------
# Kelvin-Voigt stiffness damping
# ------------------------------------------------------------------------------------


@qd.func
def func_stiffness_damping_stress(F, F_start, mu, lam, kappa):
    """Viscous second Piola-Kirchhoff stress kappa C0 : (E(F) - E(F_start)) with the rest-state isotropic tangent C0
    (Lame parameters lambda, mu) acting on the Green strain increment of the stage, and the corresponding energy
    density 1/2 dE : S."""
    dE = func_green_strain(F) - func_green_strain(F_start)
    S = kappa * (lam * dE.trace() * qd.Matrix.identity(gs.qd_float, 3) + 2.0 * mu * dE)
    energy = 0.5 * (dE * S).sum()
    return energy, S


@qd.func
def func_stiffness_damping_block(F, g_f, g_g, mu, lam, kappa):
    """Material tangent block d(F S_visc g_f)/d x_g = kappa [lambda (F g_f)(F g_g)^T + mu (g_f . g_g) F F^T
    + mu (F g_g)(F g_f)^T] (the geometric part is dropped, as mochi does by default)."""
    Fg_f = F @ g_f
    Fg_g = F @ g_g
    return kappa * (
        lam * Fg_f.outer_product(Fg_g) + mu * g_f.dot(g_g) * (F @ F.transpose()) + mu * Fg_g.outer_product(Fg_f)
    )
