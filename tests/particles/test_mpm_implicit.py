import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mpm_func import (
    HENCKY_SIGMA_MIN,
    extended_hencky_principal,
    fixed_corotated_energy,
    fixed_corotated_principal,
    isotropic_pair_block,
    isotropic_projected_modes,
    liquid_principal,
    q1_stencil,
    signed_singular_values,
    stable_neohookean_principal,
)
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_corotated(
    F_in: qd.types.ndarray(),
    r_a: qd.types.ndarray(),
    r_b: qd.types.ndarray(),
    mu: qd.f64,
    lam: qd.f64,
    out_psi: qd.types.ndarray(),
    out_P: qd.types.ndarray(),
    out_block: qd.types.ndarray(),
):
    for _ in range(1):
        F = qd.Matrix.zero(qd.f64, 3, 3)
        for i, j in qd.static(qd.ndrange(3, 3)):
            F[i, j] = F_in[i, j]
        U, S, V = qd.svd(F, qd.f64)
        sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=qd.f64)
        psi, g, A = fixed_corotated_principal(sigma, mu, lam)
        S_block, alp, bet = isotropic_projected_modes(sigma, g, A)
        P = U @ qd.Matrix([[g[0], 0.0, 0.0], [0.0, g[1], 0.0], [0.0, 0.0, g[2]]], dt=qd.f64) @ V.transpose()
        ra = qd.Vector([r_a[0], r_a[1], r_a[2]], dt=qd.f64)
        rb = qd.Vector([r_b[0], r_b[1], r_b[2]], dt=qd.f64)
        Z = isotropic_pair_block(V.transpose() @ ra, V.transpose() @ rb, S_block, alp, bet)
        block = U @ Z @ U.transpose()
        out_psi[()] = psi
        for i, j in qd.static(qd.ndrange(3, 3)):
            out_P[i, j] = P[i, j]
            out_block[i, j] = block[i, j]


def _evaluate(F, r_a, r_b, mu, lam):
    F_nd = qd.ndarray(qd.f64, shape=(3, 3))
    ra_nd, rb_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=(3,))
    psi_nd = qd.ndarray(qd.f64, shape=())
    P_nd, block_nd = qd.ndarray(qd.f64, shape=(3, 3)), qd.ndarray(qd.f64, shape=(3, 3))
    F_nd.from_numpy(F)
    ra_nd.from_numpy(r_a)
    rb_nd.from_numpy(r_b)
    kernel_corotated(F_nd, ra_nd, rb_nd, mu, lam, psi_nd, P_nd, block_nd)
    return float(qd_to_numpy(psi_nd)), qd_to_numpy(P_nd), qd_to_numpy(block_nd)


def _reference_psi(F, mu, lam):
    U, sigma, Vt = np.linalg.svd(F)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return mu * np.sum((F - R) ** 2) + 0.5 * lam * (np.linalg.det(F) - 1.0) ** 2


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_fixed_corotated_derivatives_match_finite_differences():
    rng = np.random.default_rng(1)
    mu, lam = 3.0, 5.0
    h = 1e-6
    for _ in range(4):
        F = np.eye(3) + 0.05 * rng.standard_normal((3, 3))
        r_a, r_b = rng.standard_normal(3), rng.standard_normal(3)
        psi, P, block = _evaluate(F, r_a, r_b, mu, lam)
        np.testing.assert_allclose(psi, _reference_psi(F, mu, lam), rtol=1e-10, atol=1e-12)

        # dpsi/dF
        P_fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                dF = np.zeros((3, 3))
                dF[i, j] = h
                P_fd[i, j] = (_evaluate(F + dF, r_a, r_b, mu, lam)[0] - _evaluate(F - dF, r_a, r_b, mu, lam)[0]) / (
                    2 * h
                )
        np.testing.assert_allclose(P, P_fd, rtol=1e-6, atol=1e-7)

        # Near rest the Hessian is positive definite, so the projected block is the exact one:
        # block[c, d] = d/dF (P : e_c r_a^T) along e_d r_b^T
        block_fd = np.zeros((3, 3))
        for d in range(3):
            dF = h * np.outer(np.eye(3)[d], r_b)
            P_plus = _evaluate(F + dF, r_a, r_b, mu, lam)[1]
            P_minus = _evaluate(F - dF, r_a, r_b, mu, lam)[1]
            block_fd[:, d] = (P_plus - P_minus) @ r_a / (2 * h)
        np.testing.assert_allclose(block, block_fd, rtol=1e-5, atol=1e-6)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_fixed_corotated_is_finite_and_psd_when_inverted():
    rng = np.random.default_rng(2)
    mu, lam = 3.0, 5.0
    for _ in range(8):
        F = rng.standard_normal((3, 3))
        if np.linalg.det(F) > 0:
            F[:, 0] *= -1
        psi, P, _ = _evaluate(F, np.zeros(3), np.zeros(3), mu, lam)
        assert np.isfinite(psi) and np.all(np.isfinite(P))
        np.testing.assert_allclose(psi, _reference_psi(F, mu, lam), rtol=1e-10, atol=1e-12)
        # A block r^T H r over a random r must be non-negative for the projected Hessian
        for _ in range(4):
            r = rng.standard_normal(3)
            _, _, block = _evaluate(F, r, r, mu, lam)
            assert np.all(np.isfinite(block))
            assert np.linalg.eigvalsh(0.5 * (block + block.T)).min() > -1e-9


@qd.kernel
def kernel_energy_from_singular_values(
    F_in: qd.types.ndarray(),
    mu: qd.f64,
    lam: qd.f64,
    out_sigma: qd.types.ndarray(),
    out_psi: qd.types.ndarray(),
):
    for _ in range(1):
        F = qd.Matrix.zero(qd.f64, 3, 3)
        for i, j in qd.static(qd.ndrange(3, 3)):
            F[i, j] = F_in[i, j]
        sigma = signed_singular_values(F)
        for i in qd.static(range(3)):
            out_sigma[i] = sigma[i]
        out_psi[()] = fixed_corotated_energy(sigma, mu, lam)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_energy_from_signed_singular_values():
    rng = np.random.default_rng(4)
    mu, lam = 3.0, 5.0
    F_nd, sigma_nd, psi_nd = qd.ndarray(qd.f64, (3, 3)), qd.ndarray(qd.f64, (3,)), qd.ndarray(qd.f64, ())
    samples = [np.eye(3), np.diag([1.2, 0.7, 0.9]), np.diag([0.5, -1.1, 0.8])]
    samples += [np.eye(3) + 0.3 * rng.standard_normal((3, 3)) for _ in range(6)]
    samples += [rng.standard_normal((3, 3)) for _ in range(6)]
    for F in samples:
        F_nd.from_numpy(F)
        kernel_energy_from_singular_values(F_nd, mu, lam, sigma_nd, psi_nd)
        sigma = qd_to_numpy(sigma_nd)
        reference = np.linalg.svd(F, compute_uv=False)
        reference[-1] *= np.sign(np.linalg.det(F))
        np.testing.assert_allclose(sigma, reference, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(float(qd_to_numpy(psi_nd)), _reference_psi(F, mu, lam), rtol=1e-8, atol=1e-10)


_FAMILIES = (extended_hencky_principal, stable_neohookean_principal, liquid_principal)


@qd.kernel
def kernel_principal(
    family: qd.template(),  # int, the index of the energy in _FAMILIES
    sigma_in: qd.types.ndarray(),
    mu: qd.f64,
    lam: qd.f64,
    out_psi: qd.types.ndarray(),
    out_g: qd.types.ndarray(),
    out_A: qd.types.ndarray(),
):
    for _ in range(1):
        sigma = qd.Vector([sigma_in[0], sigma_in[1], sigma_in[2]], dt=qd.f64)
        psi, g, A = qd.static(_FAMILIES[family])(sigma, mu, lam)
        out_psi[()] = psi
        for i in qd.static(range(3)):
            out_g[i] = g[i]
            for j in qd.static(range(3)):
                out_A[i, j] = A[i, j]


def _reference_extended_hencky(sigma, mu, lam):
    s = HENCKY_SIGMA_MIN
    l = np.where(sigma >= s, np.log(np.maximum(sigma, s)), np.log(s) + (sigma - s) / s)
    return mu * l @ l + 0.5 * lam * l.sum() ** 2


def _reference_stable_neohookean(sigma, mu, lam):
    lam_s = lam + mu
    alpha = 1.0 + mu / lam_s
    return 0.5 * mu * (sigma @ sigma - 3.0) + 0.5 * lam_s * (np.prod(sigma) - alpha) ** 2 - mu * mu / (2.0 * lam_s)


def _reference_liquid(sigma, mu, lam):
    J = np.prod(sigma)
    return mu * np.sum((sigma - 1.0) ** 2) + 0.5 * lam * min(J - 1.0, 0.0) ** 2


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize(
    "family, reference, sigmas",
    [
        # Above and below the tangent continuation, and inverted
        (0, _reference_extended_hencky, ((0.5, 1.5), (0.01, 0.04), (-0.5, 0.04))),
        (1, _reference_stable_neohookean, ((0.5, 1.5), (-1.2, -0.2))),
        # Compressed and expanded
        (2, _reference_liquid, ((0.6, 0.95), (1.05, 1.4))),
    ],
    ids=["extended_hencky", "stable_neohookean", "liquid"],
)
def test_principal_derivatives(family, reference, sigmas):
    rng = np.random.default_rng(5)
    mu, lam, h = 3.0, 5.0, 1e-7
    sigma_nd, psi_nd = qd.ndarray(qd.f64, (3,)), qd.ndarray(qd.f64, ())
    g_nd, A_nd = qd.ndarray(qd.f64, (3,)), qd.ndarray(qd.f64, (3, 3))

    def evaluate(sigma):
        sigma_nd.from_numpy(sigma)
        kernel_principal(family, sigma_nd, mu, lam, psi_nd, g_nd, A_nd)
        return float(qd_to_numpy(psi_nd)), qd_to_numpy(g_nd).copy(), qd_to_numpy(A_nd).copy()

    for low, high in sigmas:
        for _ in range(4):
            sigma = rng.uniform(low, high, 3)
            psi, g, A = evaluate(sigma)
            np.testing.assert_allclose(psi, reference(sigma, mu, lam), rtol=1e-10, atol=1e-12)
            for i in range(3):
                d = np.zeros(3)
                d[i] = h
                g_fd = (evaluate(sigma + d)[0] - evaluate(sigma - d)[0]) / (2 * h)
                A_fd = (evaluate(sigma + d)[1] - evaluate(sigma - d)[1]) / (2 * h)
                np.testing.assert_allclose(g[i], g_fd, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(A[:, i], A_fd, rtol=1e-5, atol=1e-5)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_stable_neohookean_rest_state():
    """The rest state is stress-free with zero energy, and the small-strain response has the Lame parameters."""
    mu, lam = 3.0, 5.0
    sigma_nd, psi_nd = qd.ndarray(qd.f64, (3,)), qd.ndarray(qd.f64, ())
    g_nd, A_nd = qd.ndarray(qd.f64, (3,)), qd.ndarray(qd.f64, (3, 3))
    sigma_nd.from_numpy(np.ones(3))
    kernel_principal(1, sigma_nd, mu, lam, psi_nd, g_nd, A_nd)
    np.testing.assert_allclose(float(qd_to_numpy(psi_nd)), 0.0, atol=1e-12)
    np.testing.assert_allclose(qd_to_numpy(g_nd), 0.0, atol=1e-12)
    # The principal Hessian of linear elasticity mu |eps|^2 + lam / 2 (tr eps)^2 is 2 mu I + lam 1 1^T
    np.testing.assert_allclose(qd_to_numpy(A_nd), 2.0 * mu * np.eye(3) + lam * np.ones((3, 3)), atol=1e-10)


@qd.kernel
def kernel_q1(fx_in: qd.types.ndarray(), w_out: qd.types.ndarray(), grad_out: qd.types.ndarray()):
    for _ in range(1):
        fx = qd.Vector([fx_in[0], fx_in[1], fx_in[2]], dt=qd.f64)
        w, grad_w = q1_stencil(fx, 2.0)
        for a in qd.static(range(8)):
            w_out[a] = w[a]
            for d in qd.static(range(3)):
                grad_out[a, d] = grad_w[a, d]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_q1_stencil_partition_of_unity():
    rng = np.random.default_rng(3)
    fx_nd, w_nd, grad_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=(8,)), qd.ndarray(qd.f64, (8, 3))
    for _ in range(5):
        fx = rng.random(3)
        fx_nd.from_numpy(fx)
        kernel_q1(fx_nd, w_nd, grad_nd)
        w, grad = qd_to_numpy(w_nd), qd_to_numpy(grad_nd)
        offsets = np.array([[(a >> 2) & 1, (a >> 1) & 1, a & 1] for a in range(8)], dtype=np.float64)
        np.testing.assert_allclose(w.sum(), 1.0)
        np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-12)
        # Q1 reproduces linear fields: sum_a w_a x_a = x and sum_a x_a grad w_a^T = I / dx (dx = 1 / 2)
        np.testing.assert_allclose(w @ offsets, fx)
        np.testing.assert_allclose(offsets.T @ grad / 2.0, np.eye(3), atol=1e-12)
