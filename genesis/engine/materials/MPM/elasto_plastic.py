from typing import Any

import quadrants as qd
from pydantic import StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base
from .elastic import func_diagonal_matrix


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
    def _update_F_S_Jp_elasto_plastic(self, J, F_tmp, U, S, V, Jp):
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float)
        if qd.static(self.use_von_mises):
            S_new = qd.max(S, 0.05)  # to prevent NaN
            sigma_new, is_yielding = func_von_mises_return_map(sigma, self.mu, self.von_mises_yield_stress)
            if is_yielding:
                S_new = func_diagonal_matrix(sigma_new)
                F_new = U @ S_new @ V.transpose()
            else:
                F_new = F_tmp
        else:
            S_new = func_diagonal_matrix(func_clamp_return_map(sigma, self.yield_lower, self.yield_higher))
            F_new = U @ S_new @ V.transpose()

        Jp_new = Jp
        return F_new, S_new, Jp_new


@qd.func
def func_von_mises_return_map(sigma: qd.types.vector(3), mu: float, yield_stress: float):
    """
    Return the singular values of the deformation gradient after the von Mises return map, and whether it yields.

    The logarithmic strain yields once its deviatoric part exceeds the yield stress over twice the shear modulus, and is
    projected back onto the yield surface. The singular values are floored at 0.05 before taking their log.
    """
    epsilon = qd.log(qd.max(sigma, 0.05))
    epsilon_hat = epsilon - epsilon.sum() / 3.0
    epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
    delta_gamma = epsilon_hat_norm - yield_stress / (2.0 * mu)
    sigma_new = sigma
    is_yielding = delta_gamma > 0.0
    if is_yielding:
        sigma_new = qd.exp(epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat)
    return sigma_new, is_yielding


@qd.func
def func_clamp_return_map(sigma: qd.types.vector(3), yield_lower: float, yield_higher: float):
    """Return the singular values of the deformation gradient clamped between 1 - `yield_lower` and
    1 + `yield_higher`, beyond which the material yields."""
    return qd.min(qd.max(sigma, 1.0 - yield_lower), 1.0 + yield_higher)
