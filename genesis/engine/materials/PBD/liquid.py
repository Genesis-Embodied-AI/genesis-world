from typing import TYPE_CHECKING, Literal

from genesis.typing import PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBDParticleEntity

SamplerType = Literal["pbs", "random", "regular"]


class Liquid(Base["PBDParticleEntity"]):
    """
    The liquid material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The rest density of the fluid in kg/m³. Default is 1000.0.
    sampler : str, optional
        Particle sampler. Defaults to 'pbs', which relaxes the particles to a uniform density inside the shape, at the
        cost of a slower first sampling of each shape, cached afterwards. 'regular' places them on a grid, which leaves
        the surface stair-stepped. 'random' draws them uniformly, which leaves local density fluctuations.
    density_relaxation : float, optional
        Relaxation factor for solving the density constraint. Default is 0.2.
    viscosity_relaxation : float, optional
        Relaxation factor used in the viscosity solver. Default is 0.01.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = "pbs"
    density_relaxation: ValidFloat = 0.2
    viscosity_relaxation: ValidFloat = 0.01
