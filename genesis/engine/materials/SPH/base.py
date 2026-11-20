from typing import TYPE_CHECKING, ClassVar

from genesis.constants import STRUCTURE

from ..base import Material, SamplerType

if TYPE_CHECKING:
    from genesis.engine.entities.sph_entity import SPHEntity


class Base(Material["SPHEntity"]):
    """
    The base class of SPH materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    sampler : str, optional
        Particle sampler. Defaults to 'pbs', which relaxes the particles to a uniform density inside the shape, at the
        cost of a slower first sampling of each shape, cached afterwards. 'regular' places them on a grid, which leaves
        the surface stair-stepped. 'random' draws them uniformly, which leaves local density fluctuations. 'pbs-<res>'
        sets the resolution of the signed distance field of 'pbs', 32 by default.
    """

    structure: ClassVar[STRUCTURE] = STRUCTURE.AMORPHOUS

    sampler: SamplerType = "pbs"
