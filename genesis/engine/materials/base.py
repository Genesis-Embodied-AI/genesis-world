from typing import TYPE_CHECKING, ClassVar

from pydantic import StrictBool, field_validator

import genesis as gs
from genesis.constants import STRUCTURE
from genesis.options.options import Options

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


class Material[EntityT: "Entity"](Options):
    """
    The base class of materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    structure: ClassVar[STRUCTURE]
    """How the shape and the connectivity of the entities of the material change, which decides how they are drawn and
    whether their visual geoms admit DynamicVertices."""

    use_visual_raycasting: StrictBool | None = None

    @field_validator("use_visual_raycasting")
    @classmethod
    def _deprecate_use_visual_raycasting(cls, value: bool | None) -> bool | None:
        if value is not None:
            gs.logger.warning(
                "'use_visual_raycasting' of the material is deprecated. Set it on the entity spec instead, such as "
                f"'gs.entities.Rigid(use_visual_raycasting={value})'."
            )
        return value
