from typing import Any

import quadrants as qd
from pydantic import StrictBool, model_validator

import genesis as gs
from genesis.typing import PositiveFloat

from .base import SamplerType
from .elastic import func_diagonal_matrix
from .elasto_plastic import ElastoPlastic, func_clamp_return_map


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
        sigma_new, Jp_new = func_snow_return_map(
            qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float), Jp, self.yield_lower, self.yield_higher
        )
        S_new = func_diagonal_matrix(sigma_new)
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp_new

    @qd.func
    def _update_stress_snow(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        h = func_snow_hardening(Jp)
        mu, lam = self.mu * h, self.lam * h

        r = U @ V.transpose()
        stress = 2 * mu * (F_new - r) @ F_new.transpose() + qd.Matrix.identity(gs.qd_float, 3) * lam * J * (J - 1)

        return stress


@qd.func
def func_snow_hardening(Jp: float):
    """
    Return the hardening factor of the Lame parameters of snow, which grows as the snow compacts.

    `Jp` is the plastic volume ratio of the snow.
    """
    return qd.exp(10.0 * (1.0 - Jp))


@qd.func
def func_snow_return_map(sigma: qd.types.vector(3), Jp: float, yield_lower: float, yield_higher: float):
    """Return the singular values of the deformation gradient of snow after its return map (see
    'func_clamp_return_map'), and its plastic volume ratio, which takes the volume the clamp removed."""
    sigma_new = func_clamp_return_map(sigma, yield_lower, yield_higher)
    Jp_new = Jp
    for k in qd.static(range(3)):
        Jp_new *= sigma[k] / sigma_new[k]
    return sigma_new, Jp_new
