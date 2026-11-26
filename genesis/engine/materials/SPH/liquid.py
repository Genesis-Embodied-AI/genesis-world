from genesis.typing import NonNegativeFloat, PositiveFloat

from ..base import SamplerType
from .base import Base


class Liquid(Base):
    """
    The liquid material class for SPH.

    Parameters
    ----------
    rho : float, optional
        The rest density (kg/m³). Default is 1000.0.
    mu : float, optional
        The dynamic viscosity of the liquid (Pa s), which damps the relative motion of neighboring particles. A higher
        value settles the liquid faster and makes it flow like honey rather than water, at the cost of a smaller stable
        time step. Default is 0.001, the viscosity of water at 20 degrees Celsius.
    surface_tension : float, optional
        The surface tension of the liquid (N/m), which pulls a curved free surface toward its center of curvature, so
        that droplets round up and jets break into drops. A higher value holds the liquid together more firmly, at the
        cost of a smaller stable time step for small droplets. 0 leaves the free surface unconstrained. Default is
        0.0728, the surface tension of water at 20 degrees Celsius.
    sampler : str, optional
        Particle sampler. Defaults to 'regular' for numerical stability with SPH.
    """

    rho: PositiveFloat = 1000.0
    mu: NonNegativeFloat = 0.001
    surface_tension: NonNegativeFloat = 0.0728
    sampler: SamplerType = "regular"
