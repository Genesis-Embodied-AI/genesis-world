import platform
import sys
from typing import TYPE_CHECKING, Literal

from ..base import MaterialOptions

if TYPE_CHECKING:
    from genesis.engine.entities.sph_entity import SPHEntity

SamplerType = Literal["pbs", "random", "regular"]
DEFAULT_SAMPLER: SamplerType = "pbs" if (sys.platform == "linux" and platform.machine() == "x86_64") else "random"


class Base(MaterialOptions["SPHEntity"]):
    """
    The base class of SPH materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    sampler : str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    """

    sampler: SamplerType = DEFAULT_SAMPLER
