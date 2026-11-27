from typing import ClassVar

from pydantic import StrictBool

from genesis.constants import STRUCTURE

from .base import Base


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
        Whether the liquid resists shear, with the second Lame parameter mu, which an inviscid liquid ignores. Default
        is False.
    """

    structure: ClassVar[STRUCTURE] = STRUCTURE.AMORPHOUS

    viscous: StrictBool = False
