"""
The registry binding each material options class to the physics a solver simulates it with.

A solver plugs in with a generic subclass of 'MaterialPhysics' declaring its own empty `PHYSICS_TYPES_MAP`, and binds
each material with a subclass of it parameterized by the options class (`class MyPhysics(MySolverPhysics[MyMaterial])`).
"""

from typing import ClassVar, TypeVar, get_args, get_origin

import genesis as gs
from genesis.engine.materials.base import Material


class MaterialPhysics[MaterialT: Material]:
    """
    The physics of a material for one solver, wrapping the material options it is built from.

    A subclass parameterized by a material class registers itself in the `PHYSICS_TYPES_MAP` of the physics base of its
    solver, and serves the subclasses of that material class that no other physics names. A subclass keeping a type
    parameter registers nothing, which lets several registered physics share code. Subclassing a registered physics
    raises a TypeError.
    """

    PHYSICS_TYPES_MAP: ClassVar[dict[type[Material], type["MaterialPhysics"]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for physics_base in cls.__mro__[1:]:
            physics_types_map = physics_base.__dict__.get("PHYSICS_TYPES_MAP")
            if physics_types_map is None:
                continue
            for physics_cls in cls.__mro__[1:]:
                if physics_cls in physics_types_map.values():
                    (generic_base,) = physics_cls.__dict__["__orig_bases__"]
                    raise TypeError(
                        f"{cls.__name__} subclasses the registered physics {physics_cls.__name__}. Subclass its "
                        f"generic base {get_origin(generic_base).__name__} instead."
                    )
        for base in cls.__dict__.get("__orig_bases__", ()):
            origin = get_origin(base)
            if origin is None or not issubclass(origin, MaterialPhysics):
                continue
            (material_cls,) = get_args(base)
            if isinstance(material_cls, TypeVar):
                break
            for physics_base in cls.__mro__:
                if "PHYSICS_TYPES_MAP" in physics_base.__dict__:
                    physics_base.PHYSICS_TYPES_MAP[material_cls] = cls
                    break
            else:
                raise TypeError(
                    f"{cls.__name__} binds {material_cls.__name__}, but none of its base classes declares the "
                    "'PHYSICS_TYPES_MAP' of a solver."
                )
            break

    def __init__(self, material: MaterialT) -> None:
        self.material = material

    @classmethod
    def from_material(cls, material: Material) -> "MaterialPhysics":
        """Build the physics of a material, bound to its options class or to the closest base class of it."""
        for options_cls in type(material).__mro__:
            physics_cls = cls.PHYSICS_TYPES_MAP.get(options_cls)
            if physics_cls is not None:
                return physics_cls(material)
        gs.raise_exception(
            f"No {cls.__name__} is registered for the material {type(material).__name__}. Define a subclass "
            f"'{cls.__name__}[{type(material).__name__}]'."
        )
