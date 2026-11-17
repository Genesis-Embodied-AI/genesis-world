import re
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, StrictBool, model_validator

import genesis as gs
from genesis.options.options import Options
from genesis.typing import PositiveInt

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


class Material[EntityT: "Entity"](Options):
    """
    The base class of materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    use_visual_raycasting: StrictBool = False


SamplerType = Annotated[gs.sampler, Field(strict=False)]


class ParticleSampling(Options):
    """
    The fields of a material whose entities are sampled into particles.

    Parameters
    ----------
    sampler : gs.sampler, optional
        Particle sampler. 'pbs' relaxes the particles to a uniform density inside the shape, at the cost of a slower
        first sampling of each shape, cached afterwards. 'regular' places them on a grid, which leaves the surface
        stair-stepped. 'random' draws them uniformly, which leaves local density fluctuations.
    sdf_resolution : int, optional
        Resolution of the signed distance field that the 'pbs' sampler relaxes the particles in. A finer field keeps
        thinner features of the shape, at the cost of a slower first sampling. The other samplers ignore it. Default
        is 32.
    """

    sampler: SamplerType = gs.sampler.pbs
    sdf_resolution: PositiveInt = 32

    @model_validator(mode="before")
    @classmethod
    def _split_resolution_from_sampler(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        sampler, sdf_resolution = split_legacy_sampler(data.get("sampler"))
        if sdf_resolution is None:
            return data
        if data.get("sdf_resolution", sdf_resolution) != sdf_resolution:
            gs.raise_exception(f"Sampler '{data['sampler']}' contradicts 'sdf_resolution={data['sdf_resolution']}'.")
        return {**data, "sampler": sampler, "sdf_resolution": sdf_resolution}


def split_legacy_sampler(sampler: Any) -> tuple[Any, int | None]:
    """Split the deprecated 'pbs-<res>' spelling of a sampler into the 'pbs' sampler and its distance field resolution.

    Any other sampler comes back as given, with no resolution.
    """
    match = re.fullmatch(r"pbs-(\d+)", sampler) if isinstance(sampler, str) else None
    if match is None:
        return sampler, None
    # A material may be created before 'gs.init', which sets up the logger
    if gs.logger is not None:
        gs.logger.warning(
            f"Sampler '{sampler}' is deprecated and will be removed in a future release. Use "
            f"'sampler=gs.sampler.pbs' with 'sdf_resolution={match[1]}' instead."
        )
    return gs.sampler.pbs, int(match[1])
