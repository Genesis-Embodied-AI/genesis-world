import copy
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar
from typing_extensions import Self

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

    # The fields holding functions bound to the material by 'model_post_init', which a copy binds to itself instead
    _dispatch_fields: ClassVar[tuple[str, ...]] = ()

    use_visual_raycasting: StrictBool = False

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> Self:
        """Copy the field values alone, so that 'model_post_init' binds the dispatch functions to the copy."""
        if not self._dispatch_fields:
            return super().__deepcopy__(memo)
        values = {name: copy.deepcopy(value, memo) for name, value in self if name not in self._dispatch_fields}
        return type(self).model_construct(_fields_set=set(self.model_fields_set), **values)
