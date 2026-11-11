from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base, SamplerType


class Liquid(Base):
    """
    The liquid material class for SPH.

    Parameters
    ----------
    rho : float, optional
        The rest density (kg/m³). Default is 1000.0.
    stiffness : float, optional
        State stiffness (N/m²). Controls how pressure increases with compression. Default is 50000.0.
    exponent : float, optional
        State exponent. Controls how nonlinearly pressure scales with density. Default is 7.0.
    mu : float, optional
        The viscosity of the liquid. Default is 0.005.
    surface_tension : float, optional
        The surface tension of the liquid (N/m), which pulls a curved free surface toward its center of curvature, so
        that droplets round up and jets break into drops. A higher value holds the liquid together more firmly, at the
        cost of a smaller stable time step for small droplets. 0 leaves the free surface unconstrained. Default is
        0.0728, the surface tension of water at 20 degrees Celsius.
    sampler : str, optional
        Particle sampler. Defaults to 'regular' for numerical stability with SPH.
    """

    rho: PositiveFloat = 1000.0
    stiffness: PositiveFloat = 50000.0
    exponent: PositiveFloat = 7.0
    mu: NonNegativeFloat = 0.005
    surface_tension: NonNegativeFloat = 0.0728
    sampler: SamplerType = "regular"
