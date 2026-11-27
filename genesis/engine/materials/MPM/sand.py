from typing import ClassVar

from genesis.constants import STRUCTURE
from genesis.typing import PositiveFloat

from ..base import SamplerType
from .base import Base


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

    sampler: SamplerType = "random"
    friction_angle: PositiveFloat = 45.0
