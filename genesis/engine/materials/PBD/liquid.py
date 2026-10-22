import platform
import sys
from typing import Literal

from genesis.typing import PositiveFloat, ValidFloat

from .base import Base


SamplerType = Literal["pbs", "random", "regular"]
DEFAULT_SAMPLER: SamplerType = "pbs" if (sys.platform == "linux" and platform.machine() == "x86_64") else "random"


class Liquid(Base):
    """
    The liquid material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The rest density of the fluid in kg/m³. Default is 1000.0.
    sampler : str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    density_relaxation : float, optional
        Relaxation factor for solving the density constraint. Default is 0.2.
    viscosity_relaxation : float, optional
        Relaxation factor used in the viscosity solver. Default is 0.01.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = DEFAULT_SAMPLER
    density_relaxation: ValidFloat = 0.2
    viscosity_relaxation: ValidFloat = 0.01
