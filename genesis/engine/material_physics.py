"""
The registry binding each material options class to the physics a solver simulates it with.

A solver plugs in with a generic subclass of 'MaterialPhysics' declaring its own empty `PHYSICS_TYPES_MAP`, and binds
each material with a subclass of it parameterized by the options class (`class MyPhysics(MySolverPhysics[MyMaterial])`).
"""

from typing import ClassVar

import genesis as gs
from genesis.engine.materials.base import Material
from genesis.utils.misc import get_generic_args


class MaterialPhysics[MaterialT: Material]:
    """
    The physics of a material for one solver, wrapping the material options it is built from.

    A subclass parameterized by a material class registers itself in the `PHYSICS_TYPES_MAP` of the physics base of its
    solver, and serves the subclasses of that material class that no other physics names. A subclass keeping a type
    parameter registers nothing, which lets several registered physics share code. A registered physics has no
    subclass, and each material class has at most one physics.
    """

    PHYSICS_TYPES_MAP: ClassVar[dict[type[Material], type["MaterialPhysics"]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        physics_types_map = next(
            (base.__dict__["PHYSICS_TYPES_MAP"] for base in cls.__mro__ if "PHYSICS_TYPES_MAP" in base.__dict__), None
        )
        if physics_types_map is not None:
            registered_bases = [base for base in cls.__mro__[1:] if base in physics_types_map.values()]
            assert not registered_bases, (
                f"{cls.__name__} subclasses the registered physics {registered_bases[0].__name__}."
            )
        generic_args = get_generic_args(cls, MaterialPhysics)
        if not generic_args or generic_args[0] is None:
            return
        (material_cls,) = generic_args
        assert physics_types_map is not None, (
            f"{cls.__name__} binds {material_cls.__name__}, but none of its base classes declares a "
            "'PHYSICS_TYPES_MAP'."
        )
        assert material_cls not in physics_types_map, (
            f"{cls.__name__} binds {material_cls.__name__}, which {physics_types_map[material_cls].__name__} "
            "already binds."
        )
        physics_types_map[material_cls] = cls

    def __init__(self, material: MaterialT) -> None:
        self.material = material

    @classmethod
    def from_material(cls, material: Material) -> "MaterialPhysics":
        """Build the physics of a material, bound to its options class or to the closest base class of it."""
        for options_cls in type(material).__mro__:
            physics_cls = cls.PHYSICS_TYPES_MAP.get(options_cls)
            if physics_cls is not None:
                return physics_cls(material)
        gs.raise_exception(f"No {cls.__name__} is registered for the material {type(material).__name__}.")
