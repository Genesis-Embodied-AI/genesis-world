from typing import Any

import quadrants as qd
from pydantic import StrictBool, model_validator

import genesis as gs
from genesis.typing import PositiveFloat

from .base import SamplerType
from .elasto_plastic import ElastoPlastic


@qd.data_oriented
class Snow(ElastoPlastic):
    """
    The snow material class for MPM.

    Note
    ----
    Snow is a special type of ElastoPlastic that gets harder when compressed.
    It does not support von Mises yield criterion.

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
    yield_lower : float, optional
        Lower bound of yield condition. Default is 2.5e-2.
    yield_higher : float, optional
        Upper bound of yield condition. Default is 4.5e-3.
    """

    sampler: SamplerType = "random"
    use_von_mises: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _enforce_no_von_mises(cls, data: dict) -> dict:
        if data.get("use_von_mises", False):
            gs.raise_exception("Snow does not support use_von_mises=True.")
        data["use_von_mises"] = False
        return data

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.update_F_S_Jp = self._update_F_S_Jp_snow
        self.update_stress = self._update_stress_snow

    @qd.func
    def _update_F_S_Jp_snow(self, J, F_tmp, U, S, V, Jp):
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        Jp_new = Jp
        for d in qd.static(range(3)):
            S_new[d, d] = min(max(S[d, d], 1 - self.yield_lower), 1 + self.yield_higher)
            Jp_new *= S[d, d] / S_new[d, d]
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def _update_stress_snow(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # Hardening coefficient: material harder when compressed
        h = qd.exp(10 * (1.0 - Jp))
        mu, lam = self.mu * h, self.lam * h

        r = U @ V.transpose()
        stress = 2 * mu * (F_new - r) @ F_new.transpose() + qd.Matrix.identity(gs.qd_float, 3) * lam * J * (J - 1)

        return stress
