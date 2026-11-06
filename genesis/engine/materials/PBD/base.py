from typing import TYPE_CHECKING

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


class Base[EntityT: "Entity"](Material[EntityT]):
    """
    The base class of PBD materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """
