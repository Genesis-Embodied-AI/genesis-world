from typing import Literal

from genesis.typing import PositiveFloat

from .base import Base


SamplerType = Literal["pbs", "random", "regular"]


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
        Particle sampler. Defaults to 'pbs', which relaxes the particles to a uniform density inside the shape, at the
        cost of a slower first sampling of each shape, cached afterwards. 'regular' places them on a grid, which leaves
        the surface stair-stepped. 'random' draws them uniformly, which leaves local density fluctuations.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = "pbs"
