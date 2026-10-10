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
def sym_eig3(A, n_sweeps: qd.template()):
    """Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations: eigenvalues and the matrix whose
    columns are the eigenvectors."""
    D = A
    Q = qd.Matrix.identity(gs.qd_float, 3)
    for _ in qd.static(range(n_sweeps)):
        for p, q in qd.static(((0, 1), (0, 2), (1, 2))):
            if qd.abs(D[p, q]) > 0.0:
                theta = (D[q, q] - D[p, p]) / (2.0 * D[p, q])
                t = 1.0 / (qd.abs(theta) + qd.sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                c = 1.0 / qd.sqrt(t * t + 1.0)
                s = t * c
                # D <- G^T D G and Q <- Q G with the Givens rotation G acting on the (p, q) plane.
                for k in qd.static(range(3)):
                    d_kp = D[k, p]
                    d_kq = D[k, q]
                    D[k, p] = c * d_kp - s * d_kq
                    D[k, q] = s * d_kp + c * d_kq
                for k in qd.static(range(3)):
                    d_pk = D[p, k]
                    d_qk = D[q, k]
                    D[p, k] = c * d_pk - s * d_qk
                    D[q, k] = s * d_pk + c * d_qk
                for k in qd.static(range(3)):
                    q_kp = Q[k, p]
                    q_kq = Q[k, q]
                    Q[k, p] = c * q_kp - s * q_kq
                    Q[k, q] = s * q_kp + c * q_kq
    return qd.Vector([D[0, 0], D[1, 1], D[2, 2]], dt=gs.qd_float), Q


@qd.func
def project_sym_psd3(A, eps):
    """Clamp the eigenvalues of a symmetric 3x3 matrix to at least eps. A matrix that is already positive definite
    (strict Sylvester test) is returned untouched, sparing the decomposition and its rounding."""
    m1 = A[0, 0]
    m2 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    m3 = A.determinant()
    B = A
    if not (m1 > 0.0 and m2 > 0.0 and m3 > 0.0):
        eigenvalues, Q = sym_eig3(A, n_sweeps=8)
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
