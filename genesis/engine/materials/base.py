from typing import TYPE_CHECKING, Any

from pydantic import StrictBool

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

    use_visual_raycasting: StrictBool = False

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "Material":
        # A material binds its kernels to itself as it is built, so a copy is built anew from the fields given
        # explicitly rather than holding the kernels of the original. The fields the solver sets are left out.
        return type(self)(
            **{
                name: value
                for name, value in self
                if name in self.model_fields_set and not type(self).model_fields[name].exclude
            }
        )
