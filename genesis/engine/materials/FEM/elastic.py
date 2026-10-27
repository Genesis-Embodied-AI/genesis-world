from typing import Any, Literal

import quadrants as qd
from pydantic import PrivateAttr, model_validator

import genesis as gs

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
        Material density (kg/m³). Default is 1000.
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
            self.compute_energy_gradient_hessian = self._compute_energy_gradient_hessian_linear
            self.compute_energy_gradient = self._compute_energy_gradient_linear
            self.compute_energy = self._compute_energy_linear
            self.hessian_invariant = True
        elif self.model == "stable_neohookean":
            self.update_stress = self._update_stress_stable_neohookean
            self.compute_energy_gradient_hessian = self._compute_energy_gradient_hessian_stable_neohookean
            self.compute_energy_gradient = self._compute_energy_gradient_stable_neohookean
            self.compute_energy = self._compute_energy_stable_neohookean
            self.hessian_invariant = False
        elif self.model == "linear_corotated":
            self.build = self._build_linear_corotated
            self.pre_compute = self._pre_compute_linear_corotated
            self.update_stress = self._update_stress_linear_corotated
            self.compute_energy_gradient_hessian = self._compute_energy_gradient_hessian_linear_corotated
            self.compute_energy_gradient = self._compute_energy_gradient_linear_corotated
            self.compute_energy = self._compute_energy_linear_corotated

    def _build_linear_corotated(self, fem_solver):
        self._R = qd.field(dtype=gs.qd_mat3, shape=(fem_solver._B, fem_solver.n_elements))

    @qd.func
    def _pre_compute_linear_corotated(self, J, F, i_e, i_b):
        U, S, V = qd.svd(F)
        R = U @ V.transpose()
        self._R[i_b, i_e] = R

    # ─── Linear model ───

    @qd.func
    def _update_stress_linear(self, mu, lam, J, F, actu, m_dir):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I
        return stress

    @qd.func
    def _compute_energy_gradient_hessian_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * eps + lam * trEps * I

        for i in qd.static(qd.grouped(qd.ndrange(3, 3))):
            hessian_field[i_b, i, i_e].fill(0.0)

        for i, k in qd.static(qd.ndrange(3, 3)):
            hessian_field[i_b, i, i, i_e][k, k] = mu

        hessian_field[i_b, 0, 0, i_e][0, 0] += mu + lam
        hessian_field[i_b, 1, 1, i_e][1, 1] += mu + lam
        hessian_field[i_b, 2, 2, i_e][2, 2] += mu + lam

        hessian_field[i_b, 0, 1, i_e][1, 0] = hessian_field[i_b, 1, 0, i_e][0, 1] = mu
        hessian_field[i_b, 0, 2, i_e][2, 0] = hessian_field[i_b, 2, 0, i_e][0, 2] = mu
        hessian_field[i_b, 1, 2, i_e][2, 1] = hessian_field[i_b, 2, 1, i_e][1, 2] = mu

        hessian_field[i_b, 0, 1, i_e][0, 1] = hessian_field[i_b, 0, 2, i_e][0, 2] = lam
        hessian_field[i_b, 1, 0, i_e][1, 0] = hessian_field[i_b, 2, 0, i_e][2, 0] = lam
        hessian_field[i_b, 1, 2, i_e][1, 2] = hessian_field[i_b, 2, 1, i_e][2, 1] = lam
        return energy, gradient

    @qd.func
    def _compute_energy_gradient_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * eps + lam * trEps * I
        return energy, gradient

    @qd.func
    def _compute_energy_linear(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        return energy

    # ─── Stable Neo-Hookean model ───

    @qd.func
    def _update_stress_stable_neohookean(self, mu, lam, J, F, actu, m_dir):
        IC = F.norm_sqr()
        dJdF0 = F[:, 1].cross(F[:, 2])
        dJdF1 = F[:, 2].cross(F[:, 0])
        dJdF2 = F[:, 0].cross(F[:, 1])
        dJdF = qd.Matrix.cols([dJdF0, dJdF1, dJdF2])
        alpha = 1 + 0.75 * mu / lam
        stress = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF
        return stress

    @qd.func
    def _compute_energy_gradient_hessian_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        raise NotImplementedError("Hessian computation is not implemented for stable_neohookean model.")

    @qd.func
    def _compute_energy_gradient_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        gs.raise_exception("Gradient computation is not implemented for stable_neohookean model.")

    @qd.func
    def _compute_energy_stable_neohookean(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        _lambda = lam + mu
        _alpha = 1.0 + mu / _lambda
        Ic = F.norm_sqr()
        Jminus1 = J - _alpha
        energy = 0.5 * (mu * (Ic - 3.0) + _lambda * Jminus1**2)
        return energy

    # ─── Linear Corotated model ───

    @qd.func
    def _update_stress_linear_corotated(self, mu, lam, J, F, actu, m_dir):
        gs.raise_exception("Linear corotated stress update is not implemented yet.")

    @qd.func
    def _compute_energy_gradient_hessian_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b, hessian_field):
        R = self._R[i_b, i_e]
        F_hat = R.transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * R @ eps + lam * trEps * R

        for i in qd.static(qd.grouped(qd.ndrange(3, 3))):
            hessian_field[i_b, i, i_e].fill(0.0)

        for i, k in qd.static(qd.ndrange(3, 3)):
            hessian_field[i_b, i, i, i_e][k, k] = mu

        for i, j, alpha, beta in qd.ndrange(3, 3, 3, 3):
            hessian_field[i_b, j, beta, i_e][i, alpha] += mu * R[i, beta] * R[alpha, j] + lam * R[alpha, beta] * R[i, j]

        return energy, gradient

    @qd.func
    def _compute_energy_gradient_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        F_hat = self._R[i_b, i_e].transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * self._R[i_b, i_e] @ eps + lam * trEps * self._R[i_b, i_e]
        return energy, gradient

    @qd.func
    def _compute_energy_linear_corotated(self, mu, lam, J, F, actu, m_dir, i_e, i_b):
        F_hat = self._R[i_b, i_e].transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        return energy
