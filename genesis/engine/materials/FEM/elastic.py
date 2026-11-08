from typing import Any, Literal

import quadrants as qd
from pydantic import PrivateAttr, model_validator

import genesis as gs
import genesis.utils.geom as gu

from .base import Base


@qd.func
def partialJpartialF(F):
    pJpF0 = F[:, 1].cross(F[:, 2])
    pJpF1 = F[:, 2].cross(F[:, 0])
    pJpF2 = F[:, 0].cross(F[:, 1])
    pJpF = qd.Matrix.cols([pJpF0, pJpF1, pJpF2])
    return pJpF


@qd.data_oriented
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

    # Internal buffer for linear corotated rotation matrix
    _R: Any = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_deprecated_model(cls, data: dict) -> dict:
        if data.get("model") == "stable_neohooken":
            gs.logger.warning("The 'stable_neohooken' model is deprecated. Use 'stable_neohookean' instead.")
            data["model"] = "stable_neohookean"
        return data

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        if self.model == "linear":
            self.update_stress = self._update_stress_linear
            self.hessian_product = self._hessian_product_linear
            self.compute_energy_gradient = self._compute_energy_gradient_linear
            self.compute_energy = self._compute_energy_linear
        elif self.model == "stable_neohookean":
            self.update_stress = self._update_stress_stable_neohookean
            self.hessian_product = self._hessian_product_stable_neohookean
            self.compute_energy_gradient = self._compute_energy_gradient_stable_neohookean
            self.compute_energy = self._compute_energy_stable_neohookean
        elif self.model == "linear_corotated":
            self.build = self._build_linear_corotated
            self.pre_compute = self._pre_compute_linear_corotated
            self.pre_compute_grad = self._pre_compute_grad_linear_corotated
            self.update_stress = self._update_stress_linear_corotated
            self.hessian_product = self._hessian_product_linear_corotated
            self.compute_energy_gradient = self._compute_energy_gradient_linear_corotated
            self.compute_energy = self._compute_energy_linear_corotated

    def _build_linear_corotated(self, fem_solver):
        self._R = qd.field(dtype=gs.qd_mat3, shape=(fem_solver._B, fem_solver.n_elements))

    @qd.func
    def _pre_compute_linear_corotated(
        self, i_e: int, i_b: int, J: float, F: qd.types.matrix(3, 3), m_dir: qd.types.vector(3)
    ):
        U, S, V = gu.qd_svd(F)
        R = U @ V.transpose()
        self._R[i_b, i_e] = R

    @qd.func
    def _pre_compute_grad_linear_corotated(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        F_start: qd.types.matrix(3, 3),
        F: qd.types.matrix(3, 3),
        dF_damping: qd.types.matrix(3, 3),
        gradient_grad: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        # The adjoint of the rotation R, which enters the energy gradient 2 mu R E + lam tr(E) R with
        # E = sym(R^T F) - I and the damping stress mu D + mu R D^T R + lam (R : D) R with D = dF_damping.
        R = self._R[i_b, i_e]
        I_3 = qd.Matrix.identity(gs.qd_float, 3)
        E = 0.5 * (R.transpose() @ F + F.transpose() @ R) - I_3
        M = 0.5 * (R.transpose() @ gradient_grad + gradient_grad.transpose() @ R)
        gradient_dot_R = (gradient_grad * R).sum()
        R_grad = 2.0 * mu * (gradient_grad @ E + F @ M) + lam * (gradient_dot_R * F + E.trace() * gradient_grad)
        R_grad = R_grad + mu * (gradient_grad @ R.transpose() @ dF_damping + dF_damping @ R.transpose() @ gradient_grad)
        R_grad = R_grad + lam * (gradient_dot_R * dF_damping + (R * dF_damping).sum() * gradient_grad)

        # With F_start = U S V^T, a change dF turns the rotation R = U V^T by U W V^T, where W is skew with
        # W_ij = (A_ij - A_ji) / (S_ii + S_jj) and A = U^T dF V. The singular values are signed (see 'qd_svd'), so a
        # sum vanishes only for an element turned inside out, where the rotation is not differentiable: the adjoint
        # of the state then turns non-finite, which halts the backward pass (see 'StepOperation.backward').
        U, S, V = gu.qd_svd(F_start)
        K = U.transpose() @ R_grad @ V
        X = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i_x, j_x in qd.static(qd.ndrange(3, 3)):
            if qd.static(i_x != j_x):
                X[i_x, j_x] = (K[i_x, j_x] - K[j_x, i_x]) / (S[i_x, i_x] + S[j_x, j_x])
        return U @ X @ V.transpose()

    # --- Linear model ---

    @qd.func
    def _update_stress_linear(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I
        return stress

    @qd.func
    def _hessian_product_linear(self, i_e: int, i_b: int, mu: float, lam: float, dF: qd.types.matrix(3, 3)):
        return mu * (dF + dF.transpose()) + lam * dF.trace() * qd.Matrix.identity(dt=gs.qd_float, n=3)

    @qd.func
    def _compute_energy_gradient_linear(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * eps + lam * trEps * I
        return energy, gradient

    @qd.func
    def _compute_energy_linear(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        return energy

    # --- Stable Neo-Hookean model ---

    @qd.func
    def _update_stress_stable_neohookean(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        IC = F.norm_sqr()
        dJdF0 = F[:, 1].cross(F[:, 2])
        dJdF1 = F[:, 2].cross(F[:, 0])
        dJdF2 = F[:, 0].cross(F[:, 1])
        dJdF = qd.Matrix.cols([dJdF0, dJdF1, dJdF2])
        alpha = 1 + 0.75 * mu / lam
        stress = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF
        return stress

    @qd.func
    def _hessian_product_stable_neohookean(self, i_e: int, i_b: int, mu: float, lam: float, dF: qd.types.matrix(3, 3)):
        raise NotImplementedError("Hessian computation is not implemented for stable_neohookean model.")

    @qd.func
    def _compute_energy_gradient_stable_neohookean(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        gs.raise_exception("Gradient computation is not implemented for stable_neohookean model.")

    @qd.func
    def _compute_energy_stable_neohookean(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        _lambda = lam + mu
        _alpha = 1.0 + mu / _lambda
        Ic = F.norm_sqr()
        Jminus1 = J - _alpha
        energy = 0.5 * (mu * (Ic - 3.0) + _lambda * Jminus1**2)
        return energy

    # --- Linear Corotated model ---

    @qd.func
    def _update_stress_linear_corotated(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        gs.raise_exception("Linear corotated stress update is not implemented yet.")

    @qd.func
    def _hessian_product_linear_corotated(self, i_e: int, i_b: int, mu: float, lam: float, dF: qd.types.matrix(3, 3)):
        # The Hessian depends on the rotation of the element alone, taken at the start of the substep (see pre_compute)
        R = self._R[i_b, i_e]
        return mu * dF + mu * R @ dF.transpose() @ R + lam * (R * dF).sum() * R

    @qd.func
    def _compute_energy_gradient_linear_corotated(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        F_hat = self._R[i_b, i_e].transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * self._R[i_b, i_e] @ eps + lam * trEps * self._R[i_b, i_e]
        return energy, gradient

    @qd.func
    def _compute_energy_linear_corotated(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        F_hat = self._R[i_b, i_e].transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        return energy
