from typing import TYPE_CHECKING

from genesis.typing import PositiveFloat, ValidFloat

from ..base import ParticleSampling
from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBDParticleEntity


class Liquid(Base["PBDParticleEntity"], ParticleSampling):
    """
    The liquid material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The rest density of the fluid in kg/m³. Default is 1000.0.
    density_relaxation : float, optional
        Relaxation factor for solving the density constraint. Default is 0.2.
    viscosity_relaxation : float, optional
        Relaxation factor used in the viscosity solver. Default is 0.01.
    """

    rho: PositiveFloat = 1000.0
    density_relaxation: ValidFloat = 0.2
    viscosity_relaxation: ValidFloat = 0.01
