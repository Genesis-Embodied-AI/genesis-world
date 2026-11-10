from typing import Any, Literal

import quadrants as qd

import genesis as gs

from genesis.typing import PositiveFloat

from .base import Base


@qd.data_oriented
class Elastic(Base):
    """
    The elastic material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 3e5.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    model : str, optional
        Stress model ('corotation', 'neohooken'). Default is 'corotation'.
    """

    E: PositiveFloat = 3e5
    model: Literal["corotation", "neohooken"] = "corotation"

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        self.update_F_S_Jp = self._update_F_S_Jp_elastic
        if self.model == "corotation":
            self.update_stress = self._update_stress_corotation
            # corotation stress uses U @ V.T from SVD(F_tmp).
            self.needs_svd = True
        elif self.model == "neohooken":
            self.update_stress = self._update_stress_neohooken
            # neohooken stress only reads F_tmp and J=det(F_tmp); SVD can be skipped.
            self.needs_svd = False

    @qd.func
    def _update_F_S_Jp_elastic(self, J, F_tmp, U, S, V, Jp):
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def _update_stress_corotation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress

    @qd.func
    def _update_stress_neohooken(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = self.mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (
            self.lam * qd.log(J) - self.mu
        )
        return stress


# ------------------------------------------------------------------------------------
# --------------------------- implicit solver: isotropy ------------------------------
# ------------------------------------------------------------------------------------

# The energy densities of the implicit solver are isotropic, functions of the signed singular values sigma of the
# deformation gradient (see 'MPMModel'). Each one returns the density psi, its gradient g and its Hessian A in sigma.
# The quantities of the pairs of principal axes come in the order (0, 1), (0, 2), (1, 2).


@qd.func
def func_diagonal_matrix(v: qd.types.vector(3)):
    return qd.Matrix(
        [
            [v[0], 0.0, 0.0],
            [0.0, v[1], 0.0],
            [0.0, 0.0, v[2]],
        ],
        dt=gs.qd_float,
    )


@qd.func
def func_cofactor(v: qd.types.vector(3)):
    """Return the product of the two other components of every component of a vector, the gradient of the product of
    all three."""
    return qd.Vector([v[1] * v[2], v[0] * v[2], v[0] * v[1]], dt=gs.qd_float)


@qd.func
def func_cofactor_jacobian(v: qd.types.vector(3)):
    """Return the Jacobian of 'func_cofactor', the third component off the diagonal and zero on it."""
    return qd.Matrix(
        [
            [0.0, v[2], v[1]],
            [v[2], 0.0, v[0]],
            [v[1], v[0], 0.0],
        ],
        dt=gs.qd_float,
    )


@qd.func
def func_compression_penalty(
    sigma: qd.types.vector(3), mu: float, lam: float, sigma_min: float, stiffness_ratio: float
):
    """
    Return the penalty against the compression of the singular values below `sigma_min`, with its gradient and the
    diagonal of its Hessian.

    Its stiffness is `stiffness_ratio` times the Young's modulus, which keeps an elastic material from collapsing where
    its own energy grows too slowly under compression (Stomakhin et al. 2012).
    """
    young_modulus = gs.qd_float(0.0)
    if lam + mu > 0.0:
        young_modulus = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    stiffness = stiffness_ratio * young_modulus
    deficit = qd.min(sigma - sigma_min, 0.0)
    return (
        0.5 * stiffness * deficit.dot(deficit),
        stiffness * deficit,
        stiffness * (sigma < sigma_min).cast(gs.qd_float),
    )


@qd.func
def func_fixed_corotated_principal(sigma: qd.types.vector(3), mu: float, lam: float):
    """Return the fixed corotated energy density, its gradient and its Hessian in the signed singular values."""
    J = sigma[0] * sigma[1] * sigma[2]
    cofactor = func_cofactor(sigma)
    psi = mu * (sigma - 1.0).dot(sigma - 1.0) + 0.5 * lam * (J - 1.0) ** 2
    g = 2.0 * mu * (sigma - 1.0) + lam * (J - 1.0) * cofactor
    A = 2.0 * mu * qd.Matrix.identity(gs.qd_float, 3)
    A += lam * (cofactor.outer_product(cofactor) + (J - 1.0) * func_cofactor_jacobian(sigma))
    return psi, g, A


@qd.func
def func_stable_neohookean_principal(sigma: qd.types.vector(3), mu: float, lam: float):
    """Return the Stable Neo-Hookean energy density (Smith et al. 2018, Eq. 14), its gradient and its Hessian in the
    signed singular values, zero at rest and defined for inverted states."""
    mu_s = 4.0 / 3.0 * mu
    lam_s = lam + 5.0 / 6.0 * mu
    alpha = gs.qd_float(1.0)
    if lam_s > 0.0:
        alpha = 1.0 + mu_s / lam_s - mu_s / (4.0 * lam_s)
    psi_rest = 0.5 * lam_s * (1.0 - alpha) ** 2 - 0.5 * mu_s * qd.log(4.0)
    J = sigma[0] * sigma[1] * sigma[2]
    cofactor = func_cofactor(sigma)
    I_C = sigma.dot(sigma)
    barrier = 1.0 / (I_C + 1.0)
    psi = 0.5 * mu_s * (I_C - 3.0) + 0.5 * lam_s * (J - alpha) ** 2 - 0.5 * mu_s * qd.log(I_C + 1.0) - psi_rest
    g = mu_s * (1.0 - barrier) * sigma + lam_s * (J - alpha) * cofactor
    A = mu_s * (1.0 - barrier) * qd.Matrix.identity(gs.qd_float, 3)
    A += 2.0 * mu_s * barrier * barrier * sigma.outer_product(sigma)
    A += lam_s * (cofactor.outer_product(cofactor) + (J - alpha) * func_cofactor_jacobian(sigma))
    return psi, g, A


@qd.func
def func_isotropic_pair_eigenvalues(sigma: qd.types.vector(3), g: qd.types.vector(3), A: qd.types.matrix(3, 3)):
    """
    Return the twist and the flip eigenvalues of an isotropic energy for the pairs of principal axes.

    The flip eigenvalue of two equal singular values is its limit, from the Hessian in the singular values.
    """
    EPS_PAIR = qd.static(1e-6)
    twist = qd.Vector.zero(gs.qd_float, 3)
    flip = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        i, j = qd.static(((0, 1), (0, 2), (1, 2))[k])
        sigma_diff = sigma[i] - sigma[j]
        if qd.abs(sigma_diff) > EPS_PAIR:
            flip[k] = (g[i] - g[j]) / sigma_diff
        else:
            flip[k] = A[i, i] - A[i, j]
        sigma_sum = sigma[i] + sigma[j]
        if qd.abs(sigma_sum) < EPS_PAIR:
            sigma_sum = EPS_PAIR if sigma_sum >= 0.0 else -EPS_PAIR
        twist[k] = (g[i] + g[j]) / sigma_sum
    return twist, flip


@qd.func
def func_isotropic_projected_modes(sigma: qd.types.vector(3), g: qd.types.vector(3), A: qd.types.matrix(3, 3)):
    """
    Return the positive semi-definite projection of the Hessian of an isotropic energy in the deformation gradient, as
    its block in the singular values and the coefficients alpha and beta of every pair of principal axes.

    Each mode of the Hessian is clamped at zero: the eigenvalues of the block, and the twist and flip eigenvalues of
    every pair, whose half sum and half difference are alpha and beta (see 'func_isotropic_pair_block').
    """
    S = 0.5 * (A + A.transpose())
    twist, flip = func_isotropic_pair_eigenvalues(sigma, g, A)
    eigenvalues, Q = qd.sym_eig(S, gs.qd_float)
    S_projected = qd.Matrix.zero(gs.qd_float, 3, 3)
    for k in qd.static(range(3)):
        eigenvector = Q[:, k]
        S_projected += qd.max(eigenvalues[k], 0.0) * eigenvector.outer_product(eigenvector)
    twist = qd.max(twist, 0.0)
    flip = qd.max(flip, 0.0)
    return S_projected, 0.5 * (flip + twist), 0.5 * (flip - twist)


@qd.func
def func_isotropic_pair_block(
    r_a: qd.types.vector(3),
    r_b: qd.types.vector(3),
    S: qd.types.matrix(3, 3),
    alpha: qd.types.vector(3),
    beta: qd.types.vector(3),
):
    """
    Return the Hessian block of two nodes in the frame V of the singular value decomposition, from the modes of
    'func_isotropic_projected_modes'.

    `r_a` and `r_b` are the gradients of the weights of the nodes, times the deformation gradient at the start of the
    substep, expressed in the frame V.
    """
    Z = S * r_a.outer_product(r_b)
    for k in qd.static(range(3)):
        i, j = qd.static(((0, 1), (0, 2), (1, 2))[k])
        Z[i, i] += alpha[k] * r_a[j] * r_b[j]
        Z[j, j] += alpha[k] * r_a[i] * r_b[i]
        Z[i, j] += beta[k] * r_a[j] * r_b[i]
        Z[j, i] += beta[k] * r_a[i] * r_b[j]
    return Z
