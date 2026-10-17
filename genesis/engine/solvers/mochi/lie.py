# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Lie-algebra helpers for rotations parametrized by a left (world-frame) rotation-vector perturbation, and the
merit of a weighted rotation difference used by the rigid inertia term."""

import quadrants as qd

import genesis as gs


@qd.func
def skew(v):
    return qd.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dt=gs.qd_float)


@qd.func
def vee(M):
    """Vector v such that skew(v) is the antisymmetric part of M."""
    return 0.5 * qd.Vector([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]], dt=gs.qd_float)


@qd.func
def sym(M):
    return 0.5 * (M + M.transpose())


@qd.func
def rotation_difference_merit(R, Q, W):
    """Psi = 1/2 tr((R - Q) W (R - Q)^T), which is zero at R = Q and shares its derivatives with -tr(R W Q^T)."""
    D = R - Q
    return 0.5 * (D @ W @ D.transpose()).trace()


@qd.func
def rotation_difference_matrix(R, Q, W):
    """M = -R W Q^T, from which the Lie gradient and Hessian of the merit with respect to R follow."""
    return -(R @ W @ Q.transpose())


@qd.func
def rotation_difference_gradient(M):
    """d tr(R M') / d theta for the left perturbation R <- exp(skew(theta)) R, with M = R M'."""
    return -2.0 * vee(M)


@qd.func
def rotation_difference_hessian(M):
    """d^2 tr(R M') / d theta^2 for the left perturbation, with M = R M'."""
    return sym(M) - M.trace() * qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def sym_eig3(A):
    """Eigen-decomposition of a symmetric 3x3 matrix: eigenvalues in descending order and the rotation whose columns
    are the eigenvectors.

    The eigenvalue farthest from the other two solves the characteristic cubic, its eigenvector is the largest row of
    the cofactor matrix of A - lambda I, and a Jacobi rotation diagonalizes A restricted to the orthogonal complement
    (D. Eberly, "A Robust Eigensolver for 3x3 Symmetric Matrices", 2014). The decomposition is backward stable for
    every spectrum, including repeated eigenvalues, with no trigonometric function and no iteration.
    """
    eps = gs.qd_float(1.1920929e-07) if qd.static(gs.qd_float == qd.f32) else gs.qd_float(2.220446049250313e-16)
    real_min = gs.qd_float(1.1754944e-38) if qd.static(gs.qd_float == qd.f32) else gs.qd_float(2.2250738585072014e-308)

    # Normalize by the mean absolute entry, and zero the off-diagonal entries below the rounding error, so that a matrix
    # diagonal to within rounding gets the coordinate axes as eigenvectors.
    scale = (qd.abs(A[0, 0]) + qd.abs(A[1, 1]) + qd.abs(A[2, 2])) / 6.0
    scale += (qd.abs(A[0, 1]) + qd.abs(A[0, 2]) + qd.abs(A[1, 2])) / 6.0
    scale = qd.math.clamp(scale, real_min, 1.0 / real_min)
    inv_scale = 1.0 / scale
    a = A[0, 0] * inv_scale
    b = A[1, 1] * inv_scale
    c = A[2, 2] * inv_scale
    d = A[0, 1] * inv_scale
    e = A[0, 2] * inv_scale
    f = A[1, 2] * inv_scale
    d = qd.select(qd.abs(d) <= eps, 0.0, d)
    e = qd.select(qd.abs(e) <= eps, 0.0, e)
    f = qd.select(qd.abs(f) <= eps, 0.0, f)

    # With q = tr(A) / 3, p^2 = tr((A - q I)^2) / 6 and cos(3 theta) = det(A - q I) / (2 p^3), the eigenvalues are
    # q + 2 p cos(theta + 2 pi k / 3). The one farthest from the other two is q + 2 p sign(cos 3 theta) cos_sep, with
    # cos_sep = cos(acos(|cos 3 theta|) / 3). The diagonal of A - q I comes from differences of diagonal entries, exact
    # for close entries, so that clustered eigenvalues keep their relative accuracy.
    a_b = a - b
    b_c = b - c
    c_a = c - a
    b0 = (a_b - c_a) / 3.0
    b1 = (b_c - a_b) / 3.0
    b2 = (c_a - b_c) / 3.0
    q = (a + b + c) / 3.0
    p_sq = (b0 * b0 + b1 * b1 + b2 * b2 + 2.0 * (d * d + e * e + f * f)) / 6.0
    p = qd.sqrt(p_sq)
    det_b = b0 * (b1 * b2 - f * f) - d * (d * b2 - e * f) + e * (d * f - b1 * e)
    cos_3theta = qd.math.clamp(det_b / qd.max(2.0 * p * p_sq, real_min), -1.0, 1.0)
    # cos_sep, the largest root of 4 c^3 - 3 c = |cos 3 theta|, is 2F1(-1/3, 1/3; 1/2; t / 2) in t = 1 - |cos 3 theta|.
    # Its degree-7 Taylor polynomial is an upper bound within 5.1e-5, from which Newton's method decreases
    # monotonically, to 5.1e-9 after one step and 5.2e-17 after two.
    abs_cos_3theta = qd.abs(cos_3theta)
    t = 1.0 - abs_cos_3theta
    t2 = t * t
    terms_01 = 1.0 - t / 9.0
    terms_23 = 4.0 / 243.0 + 28.0 / 6561.0 * t
    terms_45 = 80.0 / 59049.0 + 2288.0 / 4782969.0 * t
    terms_67 = 23296.0 / 129140163.0 + 82688.0 / 1162261467.0 * t
    cos_sep = terms_01 - t2 * (terms_23 + t2 * (terms_45 + t2 * terms_67))
    for _ in qd.static(range(1 if gs.qd_float == qd.f32 else 2)):
        cos_sep_sq = cos_sep * cos_sep
        cos_sep -= (cos_sep * (4.0 * cos_sep_sq - 3.0) - abs_cos_3theta) / (12.0 * cos_sep_sq - 3.0)
    signed_two_p = qd.select(cos_3theta >= 0.0, 2.0, -2.0) * p
    lambda_sep = q + signed_two_p * cos_sep
    is_sep_smallest = signed_two_p < 0.0

    # The rows of the cofactor matrix of A - lambda_sep I are parallel to the eigenvector of lambda_sep. An error in
    # lambda_sep tilts them towards the other eigenvectors by the ratio of that error to the eigenvalue gap, so the
    # residual of the largest row stays of the order of that error however small the gaps are.
    m00 = a - lambda_sep
    m11 = b - lambda_sep
    m22 = c - lambda_sep
    cof00 = m11 * m22 - f * f
    cof01 = e * f - d * m22
    cof02 = d * f - m11 * e
    cof11 = m00 * m22 - e * e
    cof12 = d * e - m00 * f
    cof22 = m00 * m11 - d * d
    norm0 = cof00 * cof00 + cof01 * cof01 + cof02 * cof02
    norm1 = cof01 * cof01 + cof11 * cof11 + cof12 * cof12
    norm2 = cof02 * cof02 + cof12 * cof12 + cof22 * cof22
    row = qd.Vector([cof00, cof01, cof02], dt=gs.qd_float)
    row_norm_sq = norm0
    if norm1 > row_norm_sq:
        row = qd.Vector([cof01, cof11, cof12], dt=gs.qd_float)
        row_norm_sq = norm1
    if norm2 > row_norm_sq:
        row = qd.Vector([cof02, cof12, cof22], dt=gs.qd_float)
        row_norm_sq = norm2
    # The rows all vanish only if A = lambda_sep I to within underflow, where any unit vector is an eigenvector.
    if row_norm_sq < real_min:
        row = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        row_norm_sq = 1.0
    row_norm = qd.sqrt(row_norm_sq)

    # e0 = row / |row|, and the orthonormal basis (u, v) of its complement with u x v = e0, from a single division
    # (Duff et al., "Building an Orthonormal Basis, Revisited", 2017).
    sign = qd.select(row[2] >= 0.0, 1.0, -1.0)
    s = sign * row_norm + row[2]
    inv_norm_s = 1.0 / (row_norm * s)
    e0 = row * (s * inv_norm_s)
    xyk = -row[0] * row[1] * inv_norm_s
    u = qd.Vector([1.0 - sign * row[0] * row[0] * inv_norm_s, sign * xyk, -sign * e0[0]], dt=gs.qd_float)
    v = qd.Vector([xyk, sign - row[1] * row[1] * inv_norm_s, -e0[1]], dt=gs.qd_float)

    # Jacobi rotation of A restricted to span(u, v), accurate for any gap between its eigenvalues. Below
    # real_min / eps, the restriction is a multiple of the identity to within rounding and needs no rotation.
    A_n = qd.Matrix([[a, d, e], [d, b, f], [e, f, c]], dt=gs.qd_float)
    Au = A_n @ u
    a_uv = u.dot(Au)
    b_uv = v.dot(A_n @ v)
    c_uv = v.dot(Au)
    half_diff = 0.5 * (a_uv - b_uv)
    radius_sq = half_diff * half_diff + c_uv * c_uv
    is_scalar = radius_sq < real_min / eps
    w = qd.select(is_scalar, 2.0, qd.abs(half_diff) + qd.sqrt(radius_sq))
    tan_rot = qd.select(is_scalar, 0.0, c_uv) / w
    cos_rot = qd.sqrt(w / (2.0 * qd.select(is_scalar, 1.0, qd.sqrt(radius_sq))))
    sin_rot = tan_rot * cos_rot
    x = qd.select(half_diff >= 0.0, cos_rot, sin_rot)
    y = qd.select(half_diff >= 0.0, sin_rot, cos_rot)
    lambda1 = qd.max(a_uv, b_uv) + tan_rot * c_uv
    lambda2 = qd.min(a_uv, b_uv) - tan_rot * c_uv
    e1 = x * u + y * v
    e2 = x * v - y * u

    # The Rayleigh quotient of e0, clamped so that rounding cannot break the descending order. Both (e0, e1, e2) and
    # (e1, e2, e0) are right-handed.
    rayleigh = e0.dot(A_n @ e0)
    eigenvalues = qd.Vector([qd.max(rayleigh, lambda1), lambda1, lambda2], dt=gs.qd_float)
    Q = qd.Matrix.cols([e0, e1, e2])
    if is_sep_smallest:
        eigenvalues = qd.Vector([lambda1, lambda2, qd.min(rayleigh, lambda2)], dt=gs.qd_float)
        Q = qd.Matrix.cols([e1, e2, e0])
    return scale * eigenvalues, Q


@qd.func
def project_sym_psd3(A, eps):
    """Clamp the eigenvalues of a symmetric 3x3 matrix to at least eps. A matrix that is already positive definite
    (strict Sylvester test) is returned untouched, sparing the decomposition and its rounding."""
    m1 = A[0, 0]
    m2 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    m3 = A.determinant()
    B = A
    if not (m1 > 0.0 and m2 > 0.0 and m3 > 0.0):
        eigenvalues, Q = sym_eig3(A)
        L = qd.Matrix.zero(gs.qd_float, 3, 3)
        for k in qd.static(range(3)):
            L[k, k] = qd.max(eigenvalues[k], eps)
        B = Q @ L @ Q.transpose()
    return B


@qd.func
def vsym_from_omega(omega, dt_stage, eps):
    """Symmetric rotation-derivative correction that makes R + h (skew(w) + S) R an exact rotation for the given
    angular velocity w (valid for |w| < 1/h; the discriminant is clamped otherwise)."""
    norm_sq = omega.norm_sqr()
    disc = qd.max(0.0, 1.0 - dt_stage * dt_stage * norm_sq)
    x = (qd.sqrt(disc) - 1.0) / dt_stage
    u1 = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    if norm_sq > eps * eps:
        u1 = omega / qd.sqrt(norm_sq)
    u2 = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
    if qd.abs(u1[1]) > 0.9:
        u2 = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    u2 = u2 - u1.dot(u2) * u1
    u2 = u2 / u2.norm()
    u3 = u1.cross(u2)
    return x * (u2.outer_product(u2) + u3.outer_product(u3))
