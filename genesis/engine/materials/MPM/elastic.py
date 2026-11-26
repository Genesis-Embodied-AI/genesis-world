from typing import Literal

import quadrants as qd

import genesis as gs
from genesis.typing import PositiveFloat

from .base import Base


@qd.data_oriented
class Elastic(Base):
    """
    The elastic material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 3e5.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    model : str, optional
        Stress model ('corotation', 'neohooken'). Default is 'corotation'.
    """

    E: PositiveFloat = 3e5
    model: Literal["corotation", "neohooken"] = "corotation"

    @property
    def needs_svd(self) -> bool:
        # The corotated stress reads U @ V.T, while the neo-Hookean one reads F_tmp and J alone
        return self.model == "corotation"

    @qd.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp, eps):
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        return self._update_stress_elastic(U, S, V, F_tmp, F_new, J)

    @qd.func
    def _update_stress_elastic(self, U, S, V, F_tmp, F_new, J):
        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(self.model == "corotation"):
            stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
                gs.qd_float, 3
            ) * self.lam * J * (J - 1)
        else:
            stress = self.mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (
                self.lam * qd.log(J) - self.mu
            )
        return stress
