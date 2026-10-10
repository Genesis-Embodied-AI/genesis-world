import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.mochi.soft_materials import ELASTIC_MODEL, func_elastic_tangent, func_tet_stiffness
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def _kernel_stiffness(
    F_in: qd.types.ndarray(),
    Dm_inv_in: qd.types.ndarray(),
    model: qd.i32,
    mu: qd.f64,
    lam: qd.f64,
    K_blocks: qd.types.ndarray(),
    K_contracted: qd.types.ndarray(),
):
    for i in range(F_in.shape[0]):
        F = qd.Matrix.zero(gs.qd_float, 3, 3)
        Dm_inv = qd.Matrix.zero(gs.qd_float, 3, 3)
        for r, c in qd.static(qd.ndrange(3, 3)):
            F[r, c] = F_in[i, r, c]
            Dm_inv[r, c] = Dm_inv_in[i, r, c]
        g0 = qd.Vector([Dm_inv[0, 0], Dm_inv[0, 1], Dm_inv[0, 2]], dt=gs.qd_float)
        g1 = qd.Vector([Dm_inv[1, 0], Dm_inv[1, 1], Dm_inv[1, 2]], dt=gs.qd_float)
        g2 = qd.Vector([Dm_inv[2, 0], Dm_inv[2, 1], Dm_inv[2, 2]], dt=gs.qd_float)
        grads = qd.Matrix.rows([g0, g1, g2, -(g0 + g1 + g2)])
        vol = 1.7
        K = func_tet_stiffness(model, F, mu, lam, 1e-12, True, grads, vol)
        C = func_elastic_tangent(model, F, mu, lam, 1e-12, True)
        for f, g in qd.static(qd.ndrange(4, 4)):
            for r, c in qd.static(qd.ndrange(3, 3)):
                value = 0.0
                for m, n in qd.static(qd.ndrange(3, 3)):
                    value += grads[f, m] * C[3 * r + m, 3 * c + n] * grads[g, n]
                K_contracted[i, 3 * f + r, 3 * g + c] = vol * value
                K_blocks[i, 3 * f + r, 3 * g + c] = K[3 * f + r, 3 * g + c]


@pytest.mark.precision("64")
@pytest.mark.parametrize("model", [ELASTIC_MODEL.STABLE_NEOHOOKEAN, ELASTIC_MODEL.STVK, ELASTIC_MODEL.LINEAR])
def test_tet_stiffness_blocks_match_tangent_contraction(model):
    # Deformation gradients in tension, compression, shear and inversion: the direct and the projected eigenmode paths
    # of the neo-Hookean tangent are both exercised.
    rng = np.random.default_rng(3)
    n = 64
    F = np.tile(np.eye(3), (n, 1, 1))
    F[: n // 4] *= rng.uniform(1.05, 1.6, (n // 4, 1, 1))
    F[n // 4 : n // 2] *= rng.uniform(0.3, 0.9, (n // 4, 1, 1))
    F[n // 2 : 3 * n // 4] += 0.4 * rng.standard_normal((n // 4, 3, 3))
    F[3 * n // 4 :] = rng.standard_normal((n - 3 * n // 4, 3, 3))
    F[3 * n // 4 :, 0] *= -1.0
    Dm_inv = rng.standard_normal((n, 3, 3)) * 5.0
    K_blocks = np.zeros((n, 12, 12))
    K_contracted = np.zeros((n, 12, 12))
    _kernel_stiffness(F, Dm_inv, int(model), 3.0e5, 7.0e5, K_blocks, K_contracted)
    scale = np.abs(K_contracted).max(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(K_blocks / scale, K_contracted / scale, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(K_blocks / scale, np.swapaxes(K_blocks, 1, 2) / scale, rtol=0.0, atol=1e-10)
