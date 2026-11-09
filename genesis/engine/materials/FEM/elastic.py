from enum import IntEnum
from typing import Literal

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu

from .base import Base


class FEMModel(IntEnum):
    """The constitutive models of the elastic FEM materials, by which the FEM kernels dispatch each element."""

    LINEAR = 0
    STABLE_NEOHOOKEAN = 1
    LINEAR_COROTATED = 2


class Elastic(Base):
    """
    The elastic material class for FEM.

    Parameters
    ----------
    E : float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio, describing the material's volume change under stress. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    hydroelastic_modulus : float, optional
        Hydroelastic modulus for hydroelastic contact. Default is 1e7.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    model : str, optional
        Constitutive model to use for stress computation. Options are:
        - 'linear': Linear elasticity model
        - 'stable_neohookean': A numerically stable Neo-Hookean model
        - 'linear_corotated': Linear corotated elasticity model
        Default is 'linear'.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. ``None`` uses the coupler global
        default. Default is None.
    """

    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "linear"

    @property
    def model_id(self) -> FEMModel:
        """The constitutive model of the material, as the FEM kernels dispatch on it."""
        return FEMModel[self.model.upper()]


# --- Linear model ---


@qd.func
def func_linear_stress(mu: float, lam: float, F: qd.types.matrix(3, 3)):
    """Return the first Piola-Kirchhoff stress of the linear model."""
    I = qd.Matrix.identity(gs.qd_float, 3)
    return mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I


@qd.func
def func_linear_hessian_product(mu: float, lam: float, dF: qd.types.matrix(3, 3)):
    """Return the product of the Hessian of the energy density of the linear model with a change of deformation."""
    return mu * (dF + dF.transpose()) + lam * dF.trace() * qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def func_linear_energy_gradient(mu: float, lam: float, F: qd.types.matrix(3, 3)):
    """Return the energy density of the linear model and its gradient with respect to the deformation gradient."""
    I = qd.Matrix.identity(gs.qd_float, 3)
    eps = 0.5 * (F + F.transpose()) - I
    trEps = eps.trace()
    energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
    gradient = 2.0 * mu * eps + lam * trEps * I
    return energy, gradient


@qd.func
def func_linear_energy(mu: float, lam: float, F: qd.types.matrix(3, 3)):
    """Return the energy density of the linear model."""
    eps = 0.5 * (F + F.transpose()) - qd.Matrix.identity(gs.qd_float, 3)
    return mu * eps.norm_sqr() + 0.5 * lam * eps.trace() ** 2


# --- Stable Neo-Hookean model ---


@qd.func
def func_stable_neohookean_stress(mu: float, lam: float, J: float, F: qd.types.matrix(3, 3)):
    """Return the first Piola-Kirchhoff stress of the stable Neo-Hookean model."""
    IC = F.norm_sqr()
    dJdF = qd.Matrix.cols([F[:, 1].cross(F[:, 2]), F[:, 2].cross(F[:, 0]), F[:, 0].cross(F[:, 1])])
    alpha = 1 + 0.75 * mu / lam
    return mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF


@qd.func
def func_stable_neohookean_energy(mu: float, lam: float, J: float, F: qd.types.matrix(3, 3)):
    """Return the energy density of the stable Neo-Hookean model."""
    lam_shifted = lam + mu
    alpha = 1.0 + mu / lam_shifted
    return 0.5 * (mu * (F.norm_sqr() - 3.0) + lam_shifted * (J - alpha) ** 2)


# --- Linear corotated model ---


@qd.func
def func_linear_corotated_rotation(F: qd.types.matrix(3, 3)):
    """Return the rotation of the polar decomposition of a deformation gradient, taken at the start of a substep."""
    U, S, V = gu.qd_svd(F)
    return U @ V.transpose()


@qd.func
def func_linear_corotated_hessian_product(mu: float, lam: float, R: qd.types.matrix(3, 3), dF: qd.types.matrix(3, 3)):
    """
    Return the product of the Hessian of the energy density of the linear corotated model with a change of deformation.

    The Hessian depends on the rotation `R` of the element alone.
    """
    return mu * dF + mu * R @ dF.transpose() @ R + lam * (R * dF).sum() * R


@qd.func
def func_linear_corotated_energy_gradient(mu: float, lam: float, R: qd.types.matrix(3, 3), F: qd.types.matrix(3, 3)):
    """
    Return the energy density of the linear corotated model and its gradient with respect to the deformation gradient.

    Both are linearized about the rotation `R` of the element.
    """
    F_hat = R.transpose() @ F
    eps = 0.5 * (F_hat + F_hat.transpose()) - qd.Matrix.identity(gs.qd_float, 3)
    trEps = eps.trace()
    energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
    gradient = 2.0 * mu * R @ eps + lam * trEps * R
    return energy, gradient


@qd.func
def func_linear_corotated_energy(mu: float, lam: float, R: qd.types.matrix(3, 3), F: qd.types.matrix(3, 3)):
    """
    Return the energy density of the linear corotated model.

    It is linearized about the rotation `R` of the element.
    """
    F_hat = R.transpose() @ F
    eps = 0.5 * (F_hat + F_hat.transpose()) - qd.Matrix.identity(gs.qd_float, 3)
    return mu * eps.norm_sqr() + 0.5 * lam * eps.trace() ** 2


@qd.func
def func_linear_corotated_rotation_grad(
    mu: float,
    lam: float,
    R: qd.types.matrix(3, 3),
    F_start: qd.types.matrix(3, 3),
    F: qd.types.matrix(3, 3),
    dF_damping: qd.types.matrix(3, 3),
    gradient_grad: qd.types.matrix(3, 3),
):
    """
    Return the adjoint of the start deformation gradient of a substep through the rotation of the corotated model.

    The adjoint comes from the adjoint `gradient_grad` of the energy gradient at the end of the substep. The rotation R
    enters the energy gradient 2 mu R E + lam tr(E) R with E = sym(R^T F) and the damping stress
    mu D + mu R D^T R + lam (R : D) R with D = `dF_damping`. With F_start = U S V^T, a change dF turns the rotation
    R = U V^T by U W V^T, where W is skew with W_ij = (A_ij - A_ji) / (S_ii + S_jj) and A = U^T dF V. The singular
    values are signed (see 'qd_svd'), so a sum vanishes only for an element turned inside out, where the rotation has
    no derivative: the adjoint of the state then turns non-finite, which halts the backward pass
    (see 'StepOperation.backward').
    """
    I_3 = qd.Matrix.identity(gs.qd_float, 3)
    E = 0.5 * (R.transpose() @ F + F.transpose() @ R) - I_3
    M = 0.5 * (R.transpose() @ gradient_grad + gradient_grad.transpose() @ R)
    gradient_dot_R = (gradient_grad * R).sum()
    R_grad = 2.0 * mu * (gradient_grad @ E + F @ M) + lam * (gradient_dot_R * F + E.trace() * gradient_grad)
    R_grad = R_grad + mu * (gradient_grad @ R.transpose() @ dF_damping + dF_damping @ R.transpose() @ gradient_grad)
    R_grad = R_grad + lam * (gradient_dot_R * dF_damping + (R * dF_damping).sum() * gradient_grad)

    U, S, V = gu.qd_svd(F_start)
    K = U.transpose() @ R_grad @ V
    X = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i_x, j_x in qd.static(qd.ndrange(3, 3)):
        if qd.static(i_x != j_x):
            X[i_x, j_x] = (K[i_x, j_x] - K[j_x, i_x]) / (S[i_x, i_x] + S[j_x, j_x])
    return U @ X @ V.transpose()
