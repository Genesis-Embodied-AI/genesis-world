from typing import Literal

from genesis.typing import PositiveFloat, PositiveInt

from .elastic import Elastic


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
