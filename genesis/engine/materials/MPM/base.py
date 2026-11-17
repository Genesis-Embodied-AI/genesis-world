import re
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BeforeValidator, Field
from pydantic_core import PydanticCustomError

from genesis.typing import PositiveFloat, ValidFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.mpm_entity import MPMEntity

_SAMPLER_PATTERN = re.compile(r"^pbs(-\d+)?$|^random$|^regular$")


def _validate_sampler(v: str) -> str:
    if not isinstance(v, str) or not _SAMPLER_PATTERN.match(v):
        raise PydanticCustomError(
            "invalid_sampler",
            "Input should be 'pbs', 'pbs-<res>', 'random', or 'regular'",
            {"value": v},
        )
    return v


SamplerType = Annotated[str, BeforeValidator(_validate_sampler)]


class Base(Material["MPMEntity"]):
    """
    The base class of MPM materials.

    Note
    ----
    This class should *not* be instantiated directly. A subclass simulates with the physics registered for it, or for
    its closest base class (see 'MPMMaterialPhysics' in genesis/engine/solvers/mpm_material_physics.py).

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m³). Default is 1000.
    lam : float, optional
        The first Lame's parameter. Default is None, computed from E and nu.
    mu : float, optional
        The second Lame's parameter. Default is None, computed from E and nu.
    sampler : str, optional
        Particle sampler. Defaults to 'pbs', which relaxes the particles to a uniform density inside the shape, at the
        cost of a slower first sampling of each shape, cached afterwards. 'regular' places them on a grid, which leaves
        the surface stair-stepped. 'random' draws them uniformly, which leaves local density fluctuations.
    """

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0
    lam: ValidFloat | None = None
    mu: ValidFloat | None = None
    sampler: SamplerType = "pbs"

    def model_post_init(self, context: Any) -> None:
        if self.mu is None:
            self.mu = self.E / (2.0 * (1.0 + self.nu))
        if self.lam is None:
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
