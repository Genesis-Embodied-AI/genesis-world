import math
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.constants import STRUCTURE
from genesis.typing import PositiveFloat

from ..base import SamplerType
from .base import Base


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

    structure: ClassVar[STRUCTURE] = STRUCTURE.AMORPHOUS
    _default_Jp: ClassVar[float] = 0.0

    sampler: SamplerType = "random"
    friction_angle: PositiveFloat = 45.0

    @property
    def alpha(self) -> float:
        """The coefficient of the Drucker-Prager yield surface, derived from the friction angle."""
        sin_phi = math.sin(math.radians(self.friction_angle))
        return math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

    @qd.func
    def _sand_projection(self, S, Jp, eps):
        S_out = qd.Matrix.zero(gs.qd_float, 3, 3)
        epsilon = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(3)):
            epsilon[i] = qd.log(max(abs(S[i, i]), 1e-4))
            S_out[i, i] = 1
        tr = epsilon.sum() + Jp
        epsilon_hat = epsilon - tr / 3
        epsilon_hat_norm = epsilon_hat.norm(eps)

        Jp_new = gs.qd_float(0.0)
        if tr >= 0.0:
            Jp_new = tr
        else:
            Jp_new = 0.0
            delta_gamma = epsilon_hat_norm + (3 * self.lam + 2 * self.mu) / (2 * self.mu) * tr * self.alpha
            for i in qd.static(range(3)):
                S_out[i, i] = qd.exp(epsilon[i] - max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat[i])

        return S_out, Jp_new

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
        S_new, Jp_new = self._sand_projection(S, Jp, eps)
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        log_S_sum = gs.qd_float(0.0)
        center = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            log_S_sum += qd.log(S[i, i])
            center[i, i] = 2.0 * self.mu * qd.log(S[i, i]) * (1 / S[i, i])
        for i in qd.static(range(3)):
            center[i, i] += self.lam * log_S_sum * (1 / S[i, i])
        stress = U @ center @ V.transpose() @ F_new.transpose()
        return stress
