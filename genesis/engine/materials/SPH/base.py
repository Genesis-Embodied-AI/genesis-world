from typing import TYPE_CHECKING

from ..base import Material, ParticleSampling

if TYPE_CHECKING:
    from genesis.engine.entities.sph_entity import SPHEntity


class Base(Material["SPHEntity"], ParticleSampling):
    """
    The base class of SPH materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """
