from typing import Any

import quadrants as qd
from pydantic import StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base


@qd.data_oriented
class ElastoPlastic(Base):
    """
    The elasto-plastic material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    yield_lower : float, optional
        Lower bound for the yield clamp (ignored if using von Mises). Default is 2.5e-2.
    yield_higher : float, optional
        Upper bound for the yield clamp (ignored if using von Mises). Default is 4.5e-3.
    use_von_mises : bool, optional
        Whether to use von Mises yield criterion. Default is True.
    von_mises_yield_stress : float, optional
        Yield stress for von Mises criterion. Default is 10000.
    """

    yield_lower: NonNegativeFloat = 2.5e-2
    yield_higher: NonNegativeFloat = 4.5e-3
    use_von_mises: StrictBool = True
    von_mises_yield_stress: PositiveFloat = 10000.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.update_F_S_Jp = self._update_F_S_Jp_elasto_plastic

    @qd.func
    def _update_F_S_Jp_elasto_plastic(self, J, F_tmp, U, S, V, Jp, eps):
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(self.use_von_mises):
            S_new = qd.max(S, 0.05)  # to prevent NaN
            epsilon = qd.Vector([qd.log(S_new[0, 0]), qd.log(S_new[1, 1]), qd.log(S_new[2, 2])])
            epsilon_hat = epsilon - (epsilon.sum() / 3)
            epsilon_hat_norm = epsilon_hat.norm(eps)
            delta_gamma = epsilon_hat_norm - self.von_mises_yield_stress / (2 * self.mu)

            if delta_gamma > 0:  # Yields
                epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
                S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
                for d in qd.static(range(3)):
                    S_new[d, d] = qd.exp(epsilon[d])
                F_new = U @ S_new @ V.transpose()
            else:
                F_new = F_tmp

        else:
            S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
            for d in qd.static(range(3)):
                S_new[d, d] = min(max(S[d, d], 1 - self.yield_lower), 1 + self.yield_higher)
            F_new = U @ S_new @ V.transpose()

        Jp_new = Jp
        return F_new, S_new, Jp_new
