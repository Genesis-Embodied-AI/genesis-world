from typing import Literal

import quadrants as qd

from genesis.typing import PositiveFloat, PositiveInt

from .elastic import Elastic


@qd.data_oriented
class Muscle(Elastic):
    """
    The muscle material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    model : str, optional
        Stress model ('corotation', 'neohooken'). Default is 'neohooken'.
    n_groups : int, optional
        Number of muscle groups. Default is 1.
    """

    E: PositiveFloat = 1e6
    model: Literal["corotation", "neohooken"] = "neohooken"
    n_groups: PositiveInt = 1

    @qd.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # The fibers pull along their direction with the stiffness of the material
        stress = self._update_stress_elastic(U, S, V, F_tmp, F_new, J)
        AAt = m_dir.outer_product(m_dir)
        stress += self.E * actu * F_tmp @ AAt @ F_tmp.transpose()
        return stress
