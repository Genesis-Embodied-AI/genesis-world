import numpy as np
import pytest

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mpm_adjoint_func import return_map_vjp
from genesis.engine.solvers.mpm_data import MpmMaterialTable, MpmParams, get_mpm_material_table
from genesis.engine.solvers.mpm_func import (
    EXTENSION_SIGMA_MIN,
    EXTENSION_STIFFNESS,
    HENCKY_SIGMA_MIN,
    N_MATERIAL_PARAMS,
    PARAM_LAM,
    PARAM_MU,
    PARAM_SAND_ALPHA,
    PARAM_YIELD_HIGHER,
    PARAM_YIELD_LOWER,
    PARAM_YIELD_STRESS,
    PLASTIC_CLAMP,
    PLASTIC_LIQUID,
    PLASTIC_SAND,
    PLASTIC_SNOW,
    PLASTIC_VON_MISES,
    extended_hencky_principal,
    fixed_corotated_energy,
    fixed_corotated_principal,
    isotropic_first_piola_differential,
    isotropic_pair_block,
    isotropic_projected_modes,
    liquid_principal,
    q1_stencil,
    q1_weight_hessian,
    return_map,
    signed_singular_values,
    stable_neohookean_principal,
)
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


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
        S_block, alpha, beta = isotropic_projected_modes(sigma, g, A)
        P = U @ qd.Matrix([[g[0], 0.0, 0.0], [0.0, g[1], 0.0], [0.0, 0.0, g[2]]], dt=qd.f64) @ V.transpose()
        ra = qd.Vector([r_a[0], r_a[1], r_a[2]], dt=qd.f64)
        rb = qd.Vector([r_b[0], r_b[1], r_b[2]], dt=qd.f64)
        Z = isotropic_pair_block(V.transpose() @ ra, V.transpose() @ rb, S_block, alpha, beta)
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


def _signed_singular_values(F):
    sigma = np.linalg.svd(F, compute_uv=False)
    sigma[-1] *= np.sign(np.linalg.det(F))
    return sigma


def _reference_penalty(sigma, mu, lam):
    """Compute the compression penalty k / 2 |min(sigma - s, 0)|^2 of the fixed corotated and Stable Neo-Hookean
    energies."""
    k = EXTENSION_STIFFNESS * mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    d = np.minimum(sigma - EXTENSION_SIGMA_MIN, 0.0)
    return 0.5 * k * d @ d


def _reference_extended_corotated(sigma, mu, lam):
    J = np.prod(sigma)
    return mu * np.sum((sigma - 1.0) ** 2) + 0.5 * lam * (J - 1.0) ** 2 + _reference_penalty(sigma, mu, lam)


def _reference_stable_neohookean(sigma, mu, lam):
    """Compute the Stable Neo-Hookean energy of Smith et al. 2018 (Eq. 14) with its Lame reparameterization, shifted
    to vanish at rest, plus the compression penalty."""
    mu_s, lam_s = 4.0 / 3.0 * mu, lam + 5.0 / 6.0 * mu
    alpha = 1.0 + mu_s / lam_s - mu_s / (4.0 * lam_s)
    psi, psi_rest = (
        0.5 * mu_s * (x @ x - 3.0) + 0.5 * lam_s * (np.prod(x) - alpha) ** 2 - 0.5 * mu_s * np.log(x @ x + 1.0)
        for x in (sigma, np.ones(3))
    )
    return psi - psi_rest + _reference_penalty(sigma, mu, lam)


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
        assert_allclose(psi, _reference_psi(F, mu, lam), rtol=1e-10, atol=1e-12)

        # The first Piola-Kirchhoff stress P is dpsi/dF
        P_fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                dF = np.zeros((3, 3))
                dF[i, j] = h
                psi_plus = _evaluate(F + dF, r_a, r_b, mu, lam)[0]
                psi_minus = _evaluate(F - dF, r_a, r_b, mu, lam)[0]
                P_fd[i, j] = (psi_plus - psi_minus) / (2 * h)
        assert_allclose(P, P_fd, rtol=1e-6, atol=1e-7)

        # Near rest the Hessian is positive definite, so the projected block is the exact one:
        # block[c, d] = d/dF (P : e_c r_a^T) along e_d r_b^T
        block_fd = np.zeros((3, 3))
        for d in range(3):
            dF = h * np.outer(np.eye(3)[d], r_b)
            P_plus = _evaluate(F + dF, r_a, r_b, mu, lam)[1]
            P_minus = _evaluate(F - dF, r_a, r_b, mu, lam)[1]
            block_fd[:, d] = (P_plus - P_minus) @ r_a / (2 * h)
        assert_allclose(block, block_fd, rtol=1e-5, atol=1e-6)


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
        assert_allclose(psi, _reference_extended_corotated(_signed_singular_values(F), mu, lam), rtol=1e-10, atol=1e-12)
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
    F_nd, sigma_nd = qd.ndarray(qd.f64, shape=(3, 3)), qd.ndarray(qd.f64, shape=(3,))
    psi_nd = qd.ndarray(qd.f64, shape=())
    samples = [np.eye(3), np.diag([1.2, 0.7, 0.9]), np.diag([0.5, -1.1, 0.8])]
    samples += [np.eye(3) + 0.3 * rng.standard_normal((3, 3)) for _ in range(6)]
    samples += [rng.standard_normal((3, 3)) for _ in range(6)]
    for F in samples:
        F_nd.from_numpy(F)
        kernel_energy_from_singular_values(F_nd, mu, lam, sigma_nd, psi_nd)
        sigma = qd_to_numpy(sigma_nd)
        reference = _signed_singular_values(F)
        assert_allclose(sigma, reference, tol=1e-7)
        assert_allclose(qd_to_numpy(psi_nd), _reference_extended_corotated(reference, mu, lam), rtol=1e-8, atol=1e-10)


@qd.func
def _stable_neohookean_unpenalized(sigma, mu, lam):
    return stable_neohookean_principal(sigma, mu, lam, is_penalized=False)


@qd.func
def _fixed_corotated_unpenalized(sigma, mu, lam):
    return fixed_corotated_principal(sigma, mu, lam, is_penalized=False)


_FAMILIES = (
    extended_hencky_principal,
    stable_neohookean_principal,
    liquid_principal,
    fixed_corotated_principal,
    _stable_neohookean_unpenalized,
    _fixed_corotated_unpenalized,
)


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
        # Above the penalty threshold, below it along every axis, and inverted
        (1, _reference_stable_neohookean, ((0.5, 1.5), (0.05, 0.35), (-1.2, -0.2))),
        # Compressed and expanded
        (2, _reference_liquid, ((0.6, 0.95), (1.05, 1.4))),
        # Above the penalty threshold, below it along every axis, across it with inversion, and below it under a large
        # stretch of another axis
        (3, _reference_extended_corotated, ((0.5, 1.5), (0.05, 0.35), (-0.5, 0.8), (0.1, 20.0))),
    ],
    ids=["extended_hencky", "stable_neohookean", "liquid", "fixed_corotated"],
)
def test_principal_derivatives(family, reference, sigmas):
    rng = np.random.default_rng(5)
    mu, lam, h = 3.0, 5.0, 1e-7
    sigma_nd, psi_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=())
    g_nd, A_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=(3, 3))

    def evaluate(sigma):
        sigma_nd.from_numpy(sigma)
        kernel_principal(family, sigma_nd, mu, lam, psi_nd, g_nd, A_nd)
        return float(qd_to_numpy(psi_nd)), qd_to_numpy(g_nd, copy=True), qd_to_numpy(A_nd, copy=True)

    for low, high in sigmas:
        for _ in range(4):
            sigma = rng.uniform(low, high, 3)
            psi, g, A = evaluate(sigma)
            assert_allclose(psi, reference(sigma, mu, lam), rtol=1e-10, atol=1e-12)
            for i in range(3):
                d = np.zeros(3)
                d[i] = h
                g_fd = (evaluate(sigma + d)[0] - evaluate(sigma - d)[0]) / (2 * h)
                A_fd = (evaluate(sigma + d)[1] - evaluate(sigma - d)[1]) / (2 * h)
                assert_allclose(g[i], g_fd, tol=1e-6)
                assert_allclose(A[:, i], A_fd, tol=1e-5)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_stable_neohookean_rest_state():
    mu, lam = 3.0, 5.0
    sigma_nd, psi_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=())
    g_nd, A_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=(3, 3))
    sigma_nd.from_numpy(np.ones(3))
    kernel_principal(1, sigma_nd, mu, lam, psi_nd, g_nd, A_nd)
    assert_allclose(qd_to_numpy(psi_nd), 0.0, atol=1e-12)
    assert_allclose(qd_to_numpy(g_nd), 0.0, atol=1e-12)
    # The principal Hessian of linear elasticity mu |eps|^2 + lam / 2 (tr eps)^2 is 2 mu I + lam 1 1^T
    assert_allclose(qd_to_numpy(A_nd), 2.0 * mu * np.eye(3) + lam * np.ones((3, 3)), rtol=1e-7, atol=1e-10)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("family, unextended", [(1, 4), (3, 5)], ids=["stable_neohookean", "fixed_corotated"])
def test_crushed_states_restore(family, unextended):
    mu, lam = 3.0, 5.0
    sigma_nd, psi_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=())
    g_nd, A_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=(3, 3))

    def gradient(index, sigma):
        sigma_nd.from_numpy(sigma)
        kernel_principal(index, sigma_nd, mu, lam, psi_nd, g_nd, A_nd)
        return qd_to_numpy(g_nd)

    # Under uniform compression sigma = delta 1, the energy decreases toward rest all the way down to F = 0
    for delta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.6, 0.9):
        assert (gradient(family, np.full(3, delta)) < 0.0).all()
    # The extension multiplies the Kirchhoff stress sigma_i dpsi/dsigma_i, through which a crushed particle pushes its
    # grid nodes, whereas this stress vanishes with sigma for the energy without extension
    for delta in (0.1, 0.2):
        sigma = np.full(3, delta)
        assert (sigma * gradient(family, sigma) < 5.0 * sigma * gradient(unextended, sigma)).all()


@qd.kernel
def kernel_q1(fx_in: qd.types.ndarray(), w_out: qd.types.ndarray(), grad_out: qd.types.ndarray()):
    for _ in range(1):
        fx = qd.Vector([fx_in[0], fx_in[1], fx_in[2]], dt=qd.f64)
        w, grad_w = q1_stencil(fx, inv_dx=2.0)
        for a in qd.static(range(8)):
            w_out[a] = w[a]
            for d in qd.static(range(3)):
                grad_out[a, d] = grad_w[a, d]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_q1_stencil_partition_of_unity():
    rng = np.random.default_rng(3)
    fx_nd, w_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=(8,))
    grad_nd = qd.ndarray(qd.f64, shape=(8, 3))
    for _ in range(5):
        fx = rng.random(3)
        fx_nd.from_numpy(fx)
        kernel_q1(fx_nd, w_nd, grad_nd)
        w, grad = qd_to_numpy(w_nd), qd_to_numpy(grad_nd)
        offsets = np.array([[(a >> 2) & 1, (a >> 1) & 1, a & 1] for a in range(8)], dtype=np.float64)
        assert_allclose(w.sum(), 1.0, rtol=1e-7)
        assert_allclose(grad.sum(axis=0), 0.0, atol=1e-12)
        # Q1 reproduces linear fields: sum_a w_a x_a = x and sum_a x_a grad w_a^T = I / dx (dx = 1 / 2)
        assert_allclose(w @ offsets, fx, rtol=1e-7)
        assert_allclose(offsets.T @ grad / 2.0, np.eye(3), rtol=1e-7, atol=1e-12)


@qd.kernel
def kernel_q1_hessian(fx_in: qd.types.ndarray(), H_out: qd.types.ndarray()):
    for _ in range(1):
        fx = qd.Vector([fx_in[0], fx_in[1], fx_in[2]], dt=qd.f64)
        for a in qd.static(range(8)):
            H = q1_weight_hessian(fx, a, inv_dx=2.0)
            for i, j in qd.static(qd.ndrange(3, 3)):
                H_out[a, i, j] = H[i, j]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_q1_weight_hessian():
    rng = np.random.default_rng(4)
    inv_dx, h = 2.0, 1e-6
    fx_nd, w_nd = qd.ndarray(qd.f64, shape=(3,)), qd.ndarray(qd.f64, shape=(8,))
    grad_nd, H_nd = qd.ndarray(qd.f64, shape=(8, 3)), qd.ndarray(qd.f64, shape=(8, 3, 3))

    def gradient(fx):
        fx_nd.from_numpy(fx)
        kernel_q1(fx_nd, w_nd, grad_nd)
        return qd_to_numpy(grad_nd, copy=True)

    fx = rng.random(3)
    fx_nd.from_numpy(fx)
    kernel_q1_hessian(fx_nd, H_nd)
    H = qd_to_numpy(H_nd)
    for d in range(3):
        step = np.zeros(3)
        step[d] = h
        # A step dx of the position moves fx by dx * inv_dx
        H_fd = (gradient(fx + step) - gradient(fx - step)) / (2.0 * h / inv_dx)
        assert_allclose(H[:, :, d], H_fd, tol=1e-6)


_STRESS_FAMILIES = (fixed_corotated_principal, stable_neohookean_principal, extended_hencky_principal)


@qd.kernel
def kernel_first_piola(
    family: qd.template(),  # int, the index of the energy in _STRESS_FAMILIES
    F_in: qd.types.ndarray(),
    dF_in: qd.types.ndarray(),
    mu: qd.f64,
    lam: qd.f64,
    P_out: qd.types.ndarray(),
    dP_out: qd.types.ndarray(),
):
    for _ in range(1):
        F = qd.Matrix([[F_in[i, j] for j in range(3)] for i in range(3)], dt=qd.f64)
        dF = qd.Matrix([[dF_in[i, j] for j in range(3)] for i in range(3)], dt=qd.f64)
        U, S, V = qd.svd(F, qd.f64)
        sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=qd.f64)
        psi, g, A = qd.static(_STRESS_FAMILIES[family])(sigma, mu, lam)
        P = U @ qd.Matrix([[g[0], 0.0, 0.0], [0.0, g[1], 0.0], [0.0, 0.0, g[2]]], dt=qd.f64) @ V.transpose()
        dP = isotropic_first_piola_differential(U, V, sigma, g, A, dF)
        for i, j in qd.static(qd.ndrange(3, 3)):
            P_out[i, j] = P[i, j]
            dP_out[i, j] = dP[i, j]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("family", [0, 1, 2], ids=["fixed_corotated", "stable_neohookean", "extended_hencky"])
def test_first_piola_differential(family):
    rng = np.random.default_rng(6)
    mu, lam, h = 3.0, 5.0, 1e-6
    F_nd, dF_nd = qd.ndarray(qd.f64, shape=(3, 3)), qd.ndarray(qd.f64, shape=(3, 3))
    P_nd, dP_nd = qd.ndarray(qd.f64, shape=(3, 3)), qd.ndarray(qd.f64, shape=(3, 3))

    def evaluate(F, dF):
        F_nd.from_numpy(F)
        dF_nd.from_numpy(dF)
        kernel_first_piola(family, F_nd, dF_nd, mu, lam, P_nd, dP_nd)
        return qd_to_numpy(P_nd, copy=True), qd_to_numpy(dP_nd, copy=True)

    for _ in range(4):
        # Away from inversion and from repeated singular values, where the decomposition is smooth
        F = np.eye(3) + 0.3 * rng.standard_normal((3, 3))
        assert np.linalg.det(F) > 0.1
        dF = rng.standard_normal((3, 3))
        _, dP = evaluate(F, dF)
        dP_fd = (evaluate(F + h * dF, dF)[0] - evaluate(F - h * dF, dF)[0]) / (2.0 * h)
        assert_allclose(dP, dP_fd, tol=1e-5)


@qd.kernel
def kernel_return_map(
    materials: MpmMaterialTable,
    params: qd.template(),
    kind: qd.i32,
    F_in: qd.types.ndarray(),
    Jp: qd.f64,
    F_bar_in: qd.types.ndarray(),
    Jp_bar: qd.f64,
    F_out: qd.types.ndarray(),
    F_bar_out: qd.types.ndarray(),
    Jp_out: qd.types.ndarray(),
):
    for _ in range(1):
        F = qd.Matrix([[F_in[i, j] for j in range(3)] for i in range(3)], dt=qd.f64)
        F_bar = qd.Matrix([[F_bar_in[i, j] for j in range(3)] for i in range(3)], dt=qd.f64)
        F_new, Jp_new = return_map(materials, params, slot=0, kind=kind, F=F, Jp=Jp)
        F_bar_old, Jp_bar_old = return_map_vjp(
            materials, params, slot=0, kind=kind, F=F, Jp=Jp, F_new_bar=F_bar, Jp_new_bar=Jp_bar
        )
        for i, j in qd.static(qd.ndrange(3, 3)):
            F_out[i, j] = F_new[i, j]
            F_bar_out[i, j] = F_bar_old[i, j]
        Jp_out[0] = Jp_new
        Jp_out[1] = Jp_bar_old


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize(
    "kind, sigma, Jp",
    [
        (PLASTIC_VON_MISES, (1.3, 0.95, 0.8), 1.0),
        (PLASTIC_VON_MISES, (1.03, 1.0, 0.98), 1.0),
        (PLASTIC_CLAMP, (1.1, 0.95, 0.8), 1.0),
        (PLASTIC_SNOW, (1.1, 0.95, 0.8), 1.2),
        (PLASTIC_SAND, (1.2, 1.1, 1.05), -0.1),
        (PLASTIC_SAND, (1.2, 0.9, 0.7), 0.02),
        (PLASTIC_SAND, (0.9, 0.88, 0.86), 0.0),
        (PLASTIC_LIQUID, (0.95, 0.9, 0.8), 1.0),
    ],
    ids=[
        "von_mises_yield",
        "von_mises_elastic",
        "clamp",
        "snow",
        "sand_expand",
        "sand_flow",
        "sand_stick",
        "liquid",
    ],
)
def test_return_map_vjp(kind, sigma, Jp):
    rng = np.random.default_rng(7)
    columns = np.zeros(N_MATERIAL_PARAMS)
    columns[[PARAM_MU, PARAM_LAM]] = 3.0, 5.0
    columns[[PARAM_YIELD_LOWER, PARAM_YIELD_HIGHER, PARAM_YIELD_STRESS, PARAM_SAND_ALPHA]] = 0.1, 0.05, 0.5, 0.3
    materials = get_mpm_material_table(columns, family=np.array([0]), plastic=np.array([kind]))
    params = MpmParams(
        n_envs=1,
        n_particles=1,
        substep_dt=1e-3,
        dx=0.1,
        particle_volume=1e-3,
        mass_scale=1.0,
        n_materials=1,
        wall_sol_params=np.zeros(7),
        wall_lower=np.zeros(3),
        wall_upper=np.ones(3),
    )
    F_nd, F_bar_nd = qd.ndarray(qd.f64, shape=(3, 3)), qd.ndarray(qd.f64, shape=(3, 3))
    F_out, F_bar_out = qd.ndarray(qd.f64, shape=(3, 3)), qd.ndarray(qd.f64, shape=(3, 3))
    Jp_out = qd.ndarray(qd.f64, shape=(2,))

    U, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    F = (U * np.linalg.det(U)) @ np.diag(sigma) @ (V * np.linalg.det(V)).T
    W, w_Jp = rng.standard_normal((3, 3)), rng.standard_normal()
    F_bar_nd.from_numpy(W)

    def evaluate(F, Jp):
        F_nd.from_numpy(F)
        kernel_return_map(materials, params, kind, F_nd, Jp, F_bar_nd, w_Jp, F_out, F_bar_out, Jp_out)
        Jp_values = qd_to_numpy(Jp_out)
        return np.sum(W * qd_to_numpy(F_out)) + w_Jp * Jp_values[0], qd_to_numpy(F_bar_out, copy=True), Jp_values[1]

    _, F_bar, Jp_bar = evaluate(F, Jp)
    h = 1e-6
    F_bar_fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            dF = np.zeros((3, 3))
            dF[i, j] = h
            F_bar_fd[i, j] = (evaluate(F + dF, Jp)[0] - evaluate(F - dF, Jp)[0]) / (2.0 * h)
    Jp_bar_fd = (evaluate(F, Jp + h)[0] - evaluate(F, Jp - h)[0]) / (2.0 * h)
    assert_allclose(F_bar, F_bar_fd, rtol=1e-6, atol=1e-7)
    assert_allclose(Jp_bar, Jp_bar_fd, rtol=1e-6, atol=1e-7)
