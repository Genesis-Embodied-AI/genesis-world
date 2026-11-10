import math
from typing import Any, Literal

import quadrants as qd
from pydantic import Field

import genesis as gs
from genesis.typing import PositiveFloat, ValidFloat

from .base import Base, SamplerType
from .elastic import func_diagonal_matrix


@qd.data_oriented
class Sand(Base):
    """
    The sand material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    sampler : str, optional
        Particle sampler. Default is 'random'.
    friction_angle : float, optional
        Friction angle in degrees, used to compute internal pressure-dependent plasticity. Default is 45.
    """

    sampler: SamplerType = "random"
    friction_angle: PositiveFloat = 45.0

    # Derived from friction_angle, set in model_post_init.
    alpha: ValidFloat = Field(default=0.0, exclude=True)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._default_Jp = 0.0
        sin_phi = math.sin(math.radians(self.friction_angle))
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        self.update_F_S_Jp = self._update_F_S_Jp_sand
        self.update_stress = self._update_stress_sand

    @qd.func
    def _update_F_S_Jp_sand(self, J, F_tmp, U, S, V, Jp):
        sigma_new, Jp_new = func_sand_return_map(
            qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float), Jp, self.mu, self.lam, self.alpha
        )
        S_new = func_diagonal_matrix(sigma_new)
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def _update_stress_sand(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        log_S_sum = gs.qd_float(0.0)
        center = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            log_S_sum += qd.log(S[i, i])
            center[i, i] = 2.0 * self.mu * qd.log(S[i, i]) * (1 / S[i, i])
        for i in qd.static(range(3)):
            center[i, i] += self.lam * log_S_sum * (1 / S[i, i])
        stress = U @ center @ V.transpose() @ F_new.transpose()
        return stress


@qd.func
def func_sand_return_map(sigma: qd.types.vector(3), Jp: float, mu: float, lam: float, alpha: float):
    """
    Return the singular values of the deformation gradient of sand after its Drucker-Prager return map, and its volume
    correction.

    `Jp` is the log volume the particle gained by dilating, which a compressed particle gives back first. Under tension,
    sand loses its strain and keeps the dilation as volume correction. Under compression beyond the friction cone of
    coefficient `alpha`, its deviatoric strain is projected back onto the cone.
    """
    epsilon = qd.log(qd.max(qd.abs(sigma), 1e-4))
    trace = epsilon.sum() + Jp
    epsilon_hat = epsilon - trace / 3.0
    epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
    sigma_new = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)
    Jp_new = trace
    if trace < 0.0:
        Jp_new = 0.0
        delta_gamma = epsilon_hat_norm + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * trace * alpha
        sigma_new = qd.exp(epsilon - qd.max(delta_gamma, 0.0) / epsilon_hat_norm * epsilon_hat)
    return sigma_new, Jp_new


@qd.func
def func_extended_hencky_principal(sigma: qd.types.vector(3), mu: float, lam: float, sigma_min: float):
    """
    Return the extended Hencky energy density of sand, its gradient and its Hessian in the signed singular values.

    The log of a singular value below `sigma_min` continues by its tangent line, which keeps the energy finite as a
    singular value goes to zero or below.
    """
    log_stretch = qd.Vector.zero(gs.qd_float, 3)
    dlog_stretch = qd.Vector.zero(gs.qd_float, 3)
    d2log_stretch = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        if sigma[k] >= sigma_min:
            log_stretch[k] = qd.log(sigma[k])
            dlog_stretch[k] = 1.0 / sigma[k]
            d2log_stretch[k] = -1.0 / (sigma[k] * sigma[k])
        else:
            log_stretch[k] = qd.log(sigma_min) + (sigma[k] - sigma_min) / sigma_min
            dlog_stretch[k] = 1.0 / sigma_min
    trace = log_stretch.sum()
    tau = 2.0 * mu * log_stretch + lam * trace
    psi = mu * log_stretch.dot(log_stretch) + 0.5 * lam * trace * trace
    A = lam * dlog_stretch.outer_product(dlog_stretch)
    A += func_diagonal_matrix(2.0 * mu * dlog_stretch * dlog_stretch + tau * d2log_stretch)
    return psi, tau * dlog_stretch, A
