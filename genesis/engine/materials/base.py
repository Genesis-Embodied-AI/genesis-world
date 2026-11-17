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
        sampler = data.get("sampler")
        match = re.fullmatch(r"pbs-(\d+)", sampler) if isinstance(sampler, str) else None
        if match is None:
            return data
        gs.logger.warning(
            f"Sampler '{sampler}' is deprecated and will be removed in a future release. Use "
            f"'sampler=gs.sampler.pbs' with 'sdf_resolution={match[1]}' instead."
        )
        sdf_resolution = int(match[1])
        if data.get("sdf_resolution", sdf_resolution) != sdf_resolution:
            gs.raise_exception(f"Sampler '{sampler}' contradicts 'sdf_resolution={data['sdf_resolution']}'.")
        return {**data, "sampler": gs.sampler.pbs, "sdf_resolution": sdf_resolution}
