import numbers
from typing import Any, ClassVar

from pydantic import StrictBool, model_validator

import quadrants as qd

import genesis as gs
from genesis.constants import STRUCTURE

from .base import Base


@qd.data_oriented
class Liquid(Base):
    """
    The liquid material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    viscous : bool, optional
        Whether the liquid is viscous. A non-viscous liquid takes mu=0 and refuses another value. Default is False.
    """

    structure: ClassVar[STRUCTURE] = STRUCTURE.AMORPHOUS

    viscous: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _zero_mu_if_inviscid(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not data.get("viscous", False):
            # The options exported from an inviscid liquid hold mu=0 exactly, which the liquid accepts back
            mu = data.get("mu")
            if isinstance(mu, numbers.Real) and abs(mu) > 0.0:
                gs.raise_exception("An inviscid liquid takes mu=0. Set 'viscous=True' to give it a viscosity.")
            data = {**data, "mu": 0.0}
        return data

    @property
    def needs_svd(self) -> bool:
        # The viscous term 2 * mu * (F_tmp - U @ V.T) @ F_tmp.T vanishes with mu, which a non-viscous liquid zeroes
        return self.viscous

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
        F_new = qd.Matrix.identity(gs.qd_float, 3) * qd.pow(J, 1.0 / 3.0)
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self.mu * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress
