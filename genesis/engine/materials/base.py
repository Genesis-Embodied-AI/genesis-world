from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import StrictBool

from genesis.options.options import Options

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity

EntityT = TypeVar("EntityT", bound="Entity")


class Material(Options, Generic[EntityT]):
    """
    The base class of materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    use_visual_raycasting: StrictBool = False
