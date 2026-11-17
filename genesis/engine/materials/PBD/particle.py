from typing import TYPE_CHECKING

from genesis.typing import PositiveFloat

from ..base import ParticleSampling
from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBDFreeParticleEntity


class Particle(Base["PBDFreeParticleEntity"], ParticleSampling):
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
    """

    rho: PositiveFloat = 1000.0
