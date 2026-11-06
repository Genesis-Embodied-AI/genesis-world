import numpy as np
import pytest

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mpm_adjoint_func import return_map_vjp
from genesis.engine.solvers.mpm_data import MpmInfo, MpmMaterialTable
from genesis.engine.solvers.mpm_func import (
    EXTENSION_SIGMA_MIN,
    EXTENSION_STIFFNESS,
    HENCKY_SIGMA_MIN,
    N_MATERIAL_PARAMS,
    PARAM_FIBRE,
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
    active_fibre_stiffness,
    cubic_first_root,
    det_cubic,
    extended_hencky_principal,
    fibre_terms,
    fixed_corotated_principal,
    invert_node_block,
    isotropic_first_piola_differential,
    isotropic_pair_block,
    isotropic_projected_modes,
    liquid_principal,
    q1_stencil,
    q1_weight_hessian,
    return_map,
    stable_neohookean_principal,
)
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


def _run(kernel, *args, shapes):
    args_nd = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            arg_nd = qd.ndarray(gs.qd_float, shape=arg.shape)
            arg_nd.from_numpy(arg)
            arg = arg_nd
        args_nd.append(arg)
    outs_nd = [qd.ndarray(gs.qd_float, shape=shape) for shape in shapes]
    kernel(*args_nd, *outs_nd)
    return tuple(qd_to_numpy(out_nd, copy=True) for out_nd in outs_nd)


def _single_material(params_idx, params_values, plastic_kind=0):
    columns = np.zeros(N_MATERIAL_PARAMS)
    columns[params_idx] = params_values
    materials = MpmMaterialTable(sizes=dict(materials=1, material_params=N_MATERIAL_PARAMS))
    materials.material_table.from_numpy(columns)
    materials.mat_family.from_numpy(np.array([0], dtype=gs.np_int))
    materials.mat_plastic.from_numpy(np.array([plastic_kind], dtype=gs.np_int))
    mpm_info = MpmInfo(
        values=dict(
            n_envs=1,
            n_particles=1,
            substep_dt=1e-3,
            dx=0.1,
            inv_dx=10.0,
            particle_volume=1e-3,
            mass_scale=1.0,
            n_materials=1,
            liquid_band=0.0,
            wall_sol_params=(0.0,) * 7,
            wall_lower=(0.0,) * 3,
            wall_upper=(1.0,) * 3,
        )
    )
    return materials, mpm_info


_STRESS_FAMILIES = (fixed_corotated_principal, stable_neohookean_principal, extended_hencky_principal)


@qd.kernel
def kernel_first_piola(
    # The index of the energy in _STRESS_FAMILIES
    family: qd.template(),
    F_in: qd.types.ndarray(),
    dF_in: qd.types.ndarray(),
    r_a: qd.types.ndarray(),
    r_b: qd.types.ndarray(),
    mu: float,
    lam: float,
    out_psi: qd.types.ndarray(),
    out_P: qd.types.ndarray(),
    out_dP: qd.types.ndarray(),
    out_block: qd.types.ndarray(),
):
    for _ in range(1):
        F = qd.Matrix([[F_in[i, j] for j in range(3)] for i in range(3)], dt=gs.qd_float)
        dF = qd.Matrix([[dF_in[i, j] for j in range(3)] for i in range(3)], dt=gs.qd_float)
        ra = qd.Vector([r_a[0], r_a[1], r_a[2]], dt=gs.qd_float)
        rb = qd.Vector([r_b[0], r_b[1], r_b[2]], dt=gs.qd_float)
        U, S, V = qd.svd(F, gs.qd_float)
        sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float)
        psi, g, A = qd.static(_STRESS_FAMILIES[family])(sigma, mu, lam)
        P = U @ qd.Matrix([[g[0], 0.0, 0.0], [0.0, g[1], 0.0], [0.0, 0.0, g[2]]], dt=gs.qd_float) @ V.transpose()
        dP = isotropic_first_piola_differential(U, V, sigma, g, A, dF)
        S_block, alpha, beta = isotropic_projected_modes(sigma, g, A)
        Z = isotropic_pair_block(V.transpose() @ ra, V.transpose() @ rb, S_block, alpha, beta)
        block = U @ Z @ U.transpose()
        out_psi[()] = psi
        for i, j in qd.static(qd.ndrange(3, 3)):
            out_P[i, j] = P[i, j]
            out_dP[i, j] = dP[i, j]
            out_block[i, j] = block[i, j]


def _evaluate_first_piola(family, F, dF, r_a, r_b, mu, lam):
    psi, P, dP, block = _run(kernel_first_piola, family, F, dF, r_a, r_b, mu, lam, shapes=((), (3, 3), (3, 3), (3, 3)))
    return float(psi), P, dP, block


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


def _reference_extended_hencky(sigma, mu, lam):
    s = HENCKY_SIGMA_MIN
    l = np.where(sigma >= s, np.log(np.maximum(sigma, s)), np.log(s) + (sigma - s) / s)
    return mu * l @ l + 0.5 * lam * l.sum() ** 2


# Smoothing band of the liquid energy in the tests of the energy families
LIQUID_BAND = 0.05


def _reference_liquid(sigma, mu, lam):
    """The one-sided log volume term, below the smoothing band, where the quintic does not take its place."""
    J = np.prod(sigma)
    assert J <= 1.0 - LIQUID_BAND or J >= 1.0
    volume = 0.25 * lam * (J * J - 1.0 - 2.0 * np.log(J)) if J < 1.0 else 0.0
    return mu * np.sum((sigma - 1.0) ** 2) + volume


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "family, reference",
    [(0, _reference_extended_corotated), (1, _reference_stable_neohookean), (2, _reference_extended_hencky)],
    ids=["fixed_corotated", "stable_neohookean", "extended_hencky"],
)
def test_first_piola_and_projected_block_match_finite_differences(family, reference):
    rng = np.random.default_rng(1)
    mu, lam, h = 3.0, 5.0, 1e-6
    for _ in range(4):
        # Near rest the Hessian is positive definite, so the projected block is the exact one
        F = np.eye(3) + 0.05 * rng.standard_normal((3, 3))
        r_a, r_b = rng.standard_normal(3), rng.standard_normal(3)
        psi, P, _, block = _evaluate_first_piola(family, F, np.zeros((3, 3)), r_a, r_b, mu, lam)
        assert_allclose(psi, reference(_signed_singular_values(F), mu, lam), rtol=1e-10, atol=1e-12)
        if family == 0:
            assert_allclose(psi, _reference_psi(F, mu, lam), rtol=1e-10, atol=1e-12)

        # The first Piola-Kirchhoff stress P is dpsi/dF
        P_fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                dF = np.zeros((3, 3))
                dF[i, j] = h
                psi_plus = _evaluate_first_piola(family, F + dF, np.zeros((3, 3)), r_a, r_b, mu, lam)[0]
                psi_minus = _evaluate_first_piola(family, F - dF, np.zeros((3, 3)), r_a, r_b, mu, lam)[0]
                P_fd[i, j] = (psi_plus - psi_minus) / (2 * h)
        assert_allclose(P, P_fd, rtol=1e-6, atol=1e-7)

        # block[c, d] = d/dF (P : e_c r_a^T) along e_d r_b^T
        block_fd = np.zeros((3, 3))
        for d in range(3):
            dF = h * np.outer(np.eye(3)[d], r_b)
            P_plus = _evaluate_first_piola(family, F + dF, np.zeros((3, 3)), r_a, r_b, mu, lam)[1]
            P_minus = _evaluate_first_piola(family, F - dF, np.zeros((3, 3)), r_a, r_b, mu, lam)[1]
            block_fd[:, d] = (P_plus - P_minus) @ r_a / (2 * h)
        assert_allclose(block, block_fd, rtol=1e-5, atol=1e-6)

    for _ in range(4):
        # Away from inversion and from repeated singular values, where the decomposition is smooth
        F = np.eye(3) + 0.3 * rng.standard_normal((3, 3))
        assert np.linalg.det(F) > 0.1
        dF = rng.standard_normal((3, 3))
        dP = _evaluate_first_piola(family, F, dF, r_a, r_b, mu, lam)[2]
        P_plus = _evaluate_first_piola(family, F + h * dF, dF, r_a, r_b, mu, lam)[1]
        P_minus = _evaluate_first_piola(family, F - h * dF, dF, r_a, r_b, mu, lam)[1]
        assert_allclose(dP, (P_plus - P_minus) / (2.0 * h), tol=1e-5)


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
        psi, P, _, _ = _evaluate_first_piola(0, F, np.zeros((3, 3)), np.zeros(3), np.zeros(3), mu, lam)
        assert np.isfinite(psi) and np.all(np.isfinite(P))
        assert_allclose(psi, _reference_extended_corotated(_signed_singular_values(F), mu, lam), rtol=1e-10, atol=1e-12)
        # A block r^T H r over a random r must be non-negative for the projected Hessian
        for _ in range(4):
            r = rng.standard_normal(3)
            block = _evaluate_first_piola(0, F, np.zeros((3, 3)), r, r, mu, lam)[3]
            assert np.all(np.isfinite(block))
            assert np.linalg.eigvalsh(0.5 * (block + block.T)).min() > -1e-9


@qd.func
def _stable_neohookean_unpenalized(sigma, mu, lam):
    return stable_neohookean_principal(sigma, mu, lam, is_penalized=False)


@qd.func
def _fixed_corotated_unpenalized(sigma, mu, lam):
    return fixed_corotated_principal(sigma, mu, lam, is_penalized=False)


@qd.func
def _liquid_banded(sigma, mu, lam):
    return liquid_principal(sigma, mu, lam, LIQUID_BAND)


_FAMILIES = (
    extended_hencky_principal,
    stable_neohookean_principal,
    _liquid_banded,
    fixed_corotated_principal,
    _stable_neohookean_unpenalized,
    _fixed_corotated_unpenalized,
)


@qd.kernel
def kernel_principal(
    # The index of the energy in _FAMILIES
    family: qd.template(),
    sigma_in: qd.types.ndarray(),
    mu: float,
    lam: float,
    out_psi: qd.types.ndarray(),
    out_g: qd.types.ndarray(),
    out_A: qd.types.ndarray(),
):
    for _ in range(1):
        sigma = qd.Vector([sigma_in[0], sigma_in[1], sigma_in[2]], dt=gs.qd_float)
        psi, g, A = qd.static(_FAMILIES[family])(sigma, mu, lam)
        out_psi[()] = psi
        for i in qd.static(range(3)):
            out_g[i] = g[i]
            for j in qd.static(range(3)):
                out_A[i, j] = A[i, j]


def _evaluate_principal(family, sigma, mu, lam):
    psi, g, A = _run(kernel_principal, family, sigma, mu, lam, shapes=((), (3,), (3, 3)))
    return float(psi), g, A


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize(
    "family, reference, sigmas, rest_volume_scale",
    [
        # Above and below the tangent continuation, and inverted
        (0, _reference_extended_hencky, ((0.5, 1.5), (0.01, 0.04), (-0.5, 0.04)), 1.0),
        # Above the penalty threshold, below it along every axis, and inverted
        (1, _reference_stable_neohookean, ((0.5, 1.5), (0.05, 0.35), (-1.2, -0.2)), 1.0),
        # Compressed below the smoothing band and expanded. The volume term vanishes at rest.
        (2, _reference_liquid, ((0.6, 0.98), (1.05, 1.4)), 0.0),
        # Above the penalty threshold, below it along every axis, across it with inversion, and below it under a large
        # stretch of another axis
        (3, _reference_extended_corotated, ((0.5, 1.5), (0.05, 0.35), (-0.5, 0.8), (0.1, 20.0)), 1.0),
    ],
    ids=["extended_hencky", "stable_neohookean", "liquid", "fixed_corotated"],
)
def test_principal_derivatives(family, reference, sigmas, rest_volume_scale):
    rng = np.random.default_rng(5)
    mu, lam, h = 3.0, 5.0, 1e-7
    for low, high in sigmas:
        for _ in range(4):
            sigma = rng.uniform(low, high, 3)
            psi, g, A = _evaluate_principal(family, sigma, mu, lam)
            assert_allclose(psi, reference(sigma, mu, lam), rtol=1e-10, atol=1e-12)
            for i in range(3):
                d = np.zeros(3)
                d[i] = h
                plus = _evaluate_principal(family, sigma + d, mu, lam)
                minus = _evaluate_principal(family, sigma - d, mu, lam)
                assert_allclose(g[i], (plus[0] - minus[0]) / (2 * h), tol=1e-6)
                assert_allclose(A[:, i], (plus[1] - minus[1]) / (2 * h), tol=1e-5)

    # The rest state is stress free, and its principal Hessian is the one of linear elasticity mu |eps|^2 + lam / 2
    # (tr eps)^2, which is 2 mu I + lam 1 1^T
    psi, g, A = _evaluate_principal(family, np.ones(3), mu, lam)
    assert_allclose(psi, 0.0, atol=1e-12)
    assert_allclose(g, 0.0, atol=1e-12)
    assert_allclose(A, 2.0 * mu * np.eye(3) + rest_volume_scale * lam * np.ones((3, 3)), rtol=1e-7, atol=1e-10)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_liquid_smoothing_band():
    rng = np.random.default_rng(8)
    lam, h = 5.0, 1e-7

    # Inside the band, the gradient and the Hessian are the derivatives of the energy
    for _ in range(6):
        sigma = rng.uniform(0.984, 0.999, 3)
        assert 1.0 - LIQUID_BAND < np.prod(sigma) < 1.0
        psi, g, A = _evaluate_principal(2, sigma, 0.0, lam)
        for i in range(3):
            d = np.zeros(3)
            d[i] = h
            plus, minus = _evaluate_principal(2, sigma + d, 0.0, lam), _evaluate_principal(2, sigma - d, 0.0, lam)
            assert_allclose(g[i], (plus[0] - minus[0]) / (2 * h), tol=1e-6)
            assert_allclose(A[:, i], (plus[1] - minus[1]) / (2 * h), tol=1e-5)
    # The energy, the gradient and the Hessian are continuous where the band meets the log form and at J = 1, where the
    # compression stiffness vanishes instead of jumping to lam
    for J in (1.0 - LIQUID_BAND, 1.0):
        below = _evaluate_principal(2, np.full(3, (J * (1.0 - 1e-9)) ** (1.0 / 3.0)), 0.0, lam)
        above = _evaluate_principal(2, np.full(3, (J * (1.0 + 1e-9)) ** (1.0 / 3.0)), 0.0, lam)
        for value_below, value_above in zip(below, above):
            assert_allclose(value_below, value_above, tol=1e-6 * lam)
        if J == 1.0:
            for value in below:
                assert_allclose(value, 0.0, tol=1e-6 * lam)
    # The curvature 1^T A 1 of the energy along a uniform compression sigma = s 1 stays positive across the band
    for J in np.linspace(1.0 - LIQUID_BAND, 1.0, 12)[:-1]:
        A = _evaluate_principal(2, np.full(3, J ** (1.0 / 3.0)), 0.0, lam)[2]
        assert A.sum() > 0.0


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("family, unextended", [(1, 4), (3, 5)], ids=["stable_neohookean", "fixed_corotated"])
def test_crushed_states_restore(family, unextended):
    mu, lam = 3.0, 5.0

    # Under uniform compression sigma = delta 1, the energy decreases toward rest all the way down to F = 0
    for delta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.6, 0.9):
        assert (_evaluate_principal(family, np.full(3, delta), mu, lam)[1] < 0.0).all()
    # The extension multiplies the Kirchhoff stress sigma_i dpsi/dsigma_i, through which a crushed particle pushes its
    # grid nodes, whereas this stress vanishes with sigma for the energy without extension
    for delta in (0.1, 0.2):
        sigma = np.full(3, delta)
        g_extended = _evaluate_principal(family, sigma, mu, lam)[1]
        g_unextended = _evaluate_principal(unextended, sigma, mu, lam)[1]
        assert (sigma * g_extended < 5.0 * sigma * g_unextended).all()


@qd.kernel
def kernel_fibre(
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
    actu: float,
    m_in: qd.types.ndarray(),
    F_in: qd.types.ndarray(),
    out_psi: qd.types.ndarray(),
    out_P: qd.types.ndarray(),
    out_stiffness: qd.types.ndarray(),
):
    for _ in range(1):
        F = qd.Matrix([[F_in[i, j] for j in range(3)] for i in range(3)], dt=gs.qd_float)
        m = qd.Vector([m_in[0], m_in[1], m_in[2]], dt=gs.qd_float)
        psi, P, curvature = fibre_terms(0, actu, m, F, materials, mpm_info)
        ka, dka = active_fibre_stiffness(0, actu, materials, mpm_info)
        out_psi[()] = psi
        for i, j in qd.static(qd.ndrange(3, 3)):
            out_P[i, j] = P[i, j]
        out_stiffness[0] = curvature
        out_stiffness[1] = dka


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_fibre_terms():
    rng = np.random.default_rng(9)
    stiffness, h = 2.0, 1e-7
    materials, mpm_info = _single_material([PARAM_MU, PARAM_LAM, PARAM_FIBRE], [3.0, 5.0, stiffness])
    m = np.array([0.0, 0.6, 0.8])
    F = np.eye(3) + 0.2 * rng.standard_normal((3, 3))

    def evaluate(F, actu):
        psi, P, stiffnesses = _run(kernel_fibre, materials, mpm_info, actu, m, F, shapes=((), (3, 3), (2,)))
        return float(psi), P, stiffnesses

    # An active fibre pulls along its direction, with the stress of its energy and the actuation derivative the adjoint
    # takes
    actu = 0.7
    psi, P, (curvature, dka) = evaluate(F, actu)
    assert_allclose(psi, 0.5 * stiffness * actu * np.sum((F @ m) ** 2), tol=1e-12)
    assert_allclose(curvature, stiffness * actu, tol=1e-12)
    for i, j in np.ndindex(3, 3):
        dF = np.zeros((3, 3))
        dF[i, j] = h
        assert_allclose(P[i, j], (evaluate(F + dF, actu)[0] - evaluate(F - dF, actu)[0]) / (2 * h), tol=1e-6)
    assert_allclose(
        dka * 0.5 * np.sum((F @ m) ** 2), (evaluate(F, actu + h)[0] - evaluate(F, actu - h)[0]) / (2 * h), tol=1e-6
    )

    # A muscle only contracts: a negative actuation activates nothing, so the term stays convex and bounded below
    psi, P, stiffnesses = evaluate(F, -0.4)
    assert psi == 0.0 and not P.any() and not stiffnesses.any()


@qd.kernel
def kernel_invert_node_block(D_in: qd.types.ndarray(), D_inv_out: qd.types.ndarray()):
    for _ in range(1):
        D = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i, j in qd.static(qd.ndrange(3, 3)):
            D[i, j] = D_in[i, j]
        D_inv = invert_node_block(D)
        for i, j in qd.static(qd.ndrange(3, 3)):
            D_inv_out[i, j] = D_inv[i, j]


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_invert_node_block():
    D_nd, D_inv_nd = qd.ndarray(gs.qd_float, shape=(3, 3)), qd.ndarray(gs.qd_float, shape=(3, 3))

    def invert(D):
        D_nd.from_numpy(np.asarray(D, dtype=gs.np_float))
        kernel_invert_node_block(D_nd, D_inv_nd)
        return qd_to_numpy(D_inv_nd)

    # The block m I of a node of vanishing mass, whose determinant m^3 underflows in single precision
    mass = 2.2e-14
    assert_allclose(invert(mass * np.eye(3)), np.eye(3) / mass, rtol=1e-5)
    # The block of no particle
    assert_allclose(invert(np.zeros((3, 3))), np.eye(3), tol=gs.EPS)
    rng = np.random.default_rng(5)
    a = rng.standard_normal((3, 3))
    D = a @ a.T + np.diag([1e-3, 1.0, 1e3])
    assert_allclose(invert(D), np.linalg.inv(D), rtol=1e-4)


@qd.kernel
def kernel_inversion_root(F0_in: qd.types.ndarray(), dF_in: qd.types.ndarray(), t_max: float, out: qd.types.ndarray()):
    for _ in range(1):
        F0 = qd.Matrix.zero(gs.qd_float, 3, 3)
        dF = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i, j in qd.static(qd.ndrange(3, 3)):
            F0[i, j] = F0_in[i, j]
            dF[i, j] = dF_in[i, j]
        c0, c1, c2, c3 = det_cubic(F0, dF)
        out[()] = cubic_first_root(c0, c1, c2, c3, t_max)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_inversion_step_root():
    rng = np.random.default_rng(7)
    t_max = 1.0 / 0.9
    n_roots = 0
    for _ in range(40):
        F0 = np.eye(3) + 0.2 * rng.standard_normal((3, 3))
        dF = 2.0 * rng.standard_normal((3, 3))
        (root,) = _run(kernel_inversion_root, F0, dF, t_max, shapes=((),))
        # The first real root in (0, t_max] of the cubic det(F0 + t dF), interpolated from four of its values
        coeffs = [np.linalg.det(F0 + t * dF) for t in (0.0, 1.0, -1.0, 2.0)]
        c = np.linalg.solve(np.vander([0.0, 1.0, -1.0, 2.0], 4, increasing=True), coeffs)
        candidates = [r.real for r in np.roots(c[::-1]) if abs(r.imag) < 1e-12 and 0.0 < r.real <= t_max]
        if candidates:
            n_roots += 1
            assert_allclose(root, min(candidates), tol=1e-9)
        else:
            assert root > t_max
    # Both outcomes are exercised
    assert 0 < n_roots < 40


@qd.kernel
def kernel_q1(fx_in: qd.types.ndarray(), w_out: qd.types.ndarray(), grad_out: qd.types.ndarray()):
    for _ in range(1):
        fx = qd.Vector([fx_in[0], fx_in[1], fx_in[2]], dt=gs.qd_float)
        w, grad_w = q1_stencil(fx, inv_dx=2.0)
        for a in qd.static(range(8)):
            w_out[a] = w[a]
            for d in qd.static(range(3)):
                grad_out[a, d] = grad_w[a, d]


@qd.kernel
def kernel_q1_hessian(fx_in: qd.types.ndarray(), H_out: qd.types.ndarray()):
    for _ in range(1):
        fx = qd.Vector([fx_in[0], fx_in[1], fx_in[2]], dt=gs.qd_float)
        for a in qd.static(range(8)):
            H = q1_weight_hessian(fx, inv_dx=2.0, a=a)
            for i, j in qd.static(qd.ndrange(3, 3)):
                H_out[a, i, j] = H[i, j]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu])
def test_q1_stencil_partition_of_unity_and_weight_hessian():
    rng = np.random.default_rng(3)
    inv_dx, h = 2.0, 1e-6
    offsets = np.array([[(a >> 2) & 1, (a >> 1) & 1, a & 1] for a in range(8)], dtype=np.float64)
    for _ in range(5):
        fx = rng.random(3)
        w, grad = _run(kernel_q1, fx, shapes=((8,), (8, 3)))
        assert_allclose(w.sum(), 1.0, rtol=1e-7)
        assert_allclose(grad.sum(axis=0), 0.0, atol=1e-12)
        # Q1 reproduces linear fields: sum_a w_a x_a = x and sum_a x_a grad w_a^T = I / dx (dx = 1 / 2)
        assert_allclose(w @ offsets, fx, rtol=1e-7)
        assert_allclose(offsets.T @ grad / inv_dx, np.eye(3), rtol=1e-7, atol=1e-12)

        (H,) = _run(kernel_q1_hessian, fx, shapes=((8, 3, 3),))
        for d in range(3):
            step = np.zeros(3)
            step[d] = h
            # A step dx of the position moves fx by dx * inv_dx
            grad_plus = _run(kernel_q1, fx + step, shapes=((8,), (8, 3)))[1]
            grad_minus = _run(kernel_q1, fx - step, shapes=((8,), (8, 3)))[1]
            assert_allclose(H[:, :, d], (grad_plus - grad_minus) / (2.0 * h / inv_dx), tol=1e-6)


@qd.kernel
def kernel_return_map(
    materials: MpmMaterialTable,
    mpm_info: MpmInfo,
    kind: int,
    F_in: qd.types.ndarray(),
    Jp: float,
    F_bar_in: qd.types.ndarray(),
    Jp_bar: float,
    F_out: qd.types.ndarray(),
    F_bar_out: qd.types.ndarray(),
    Jp_out: qd.types.ndarray(),
):
    for _ in range(1):
        F = qd.Matrix([[F_in[i, j] for j in range(3)] for i in range(3)], dt=gs.qd_float)
        F_bar = qd.Matrix([[F_bar_in[i, j] for j in range(3)] for i in range(3)], dt=gs.qd_float)
        F_new, Jp_new = return_map(slot=0, kind=kind, F=F, Jp=Jp, materials=materials, mpm_info=mpm_info)
        F_bar_old, Jp_bar_old = return_map_vjp(
            slot=0, kind=kind, F=F, Jp=Jp, F_new_bar=F_bar, Jp_new_bar=Jp_bar, materials=materials, mpm_info=mpm_info
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
    ids=["von_mises_yield", "von_mises_elastic", "clamp", "snow", "sand_expand", "sand_flow", "sand_stick", "liquid"],
)
def test_return_map_vjp(kind, sigma, Jp):
    rng = np.random.default_rng(7)
    materials, mpm_info = _single_material(
        [PARAM_MU, PARAM_LAM, PARAM_YIELD_LOWER, PARAM_YIELD_HIGHER, PARAM_YIELD_STRESS, PARAM_SAND_ALPHA],
        [3.0, 5.0, 0.1, 0.05, 0.5, 0.3],
        plastic_kind=kind,
    )

    U, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    F = (U * np.linalg.det(U)) @ np.diag(sigma) @ (V * np.linalg.det(V)).T
    W, w_Jp = rng.standard_normal((3, 3)), rng.standard_normal()

    def evaluate(F, Jp):
        F_new, F_bar, Jp_values = _run(
            kernel_return_map, materials, mpm_info, kind, F, Jp, W, w_Jp, shapes=((3, 3), (3, 3), (2,))
        )
        return np.sum(W * F_new) + w_Jp * Jp_values[0], F_bar, Jp_values[1]

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
