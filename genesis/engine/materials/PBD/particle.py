import platform
import sys
from typing import Literal

from genesis.typing import PositiveFloat

from .base import Base


SamplerType = Literal["pbs", "random", "regular"]
DEFAULT_SAMPLER: SamplerType = "pbs" if (sys.platform == "linux" and platform.machine() == "x86_64") else "random"


class Particle(Base):
    """
    Particle-based entities with no inter-particle interaction.

    Note
    ----
    This creates particle-based entities that have no inter-particle interaction at all; i.e. they are only affected
    by external forces. This is useful for creating particle-based animations. Hosted under the PBD system.

    Parameters
    ----------
    rho : float, optional
        The rest density. Default is 1000.0.
    sampler : str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = DEFAULT_SAMPLER
