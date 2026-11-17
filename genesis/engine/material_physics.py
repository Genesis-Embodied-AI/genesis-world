"""
The registry binding each material options class to the physics a solver simulates it with.

A solver plugs in with a direct subclass of 'MaterialPhysics' declaring its own empty `PHYSICS_TYPES_MAP`, and binds
each material with a subclass of it naming the options class in `material_cls`.
"""

from typing import ClassVar

import genesis as gs
from genesis.engine.materials.base import Material


class MaterialPhysics:
    """
    The physics of a material for one solver, wrapping the material options it is built from.

    A subclass naming `material_cls` registers itself in the `PHYSICS_TYPES_MAP` of the physics base of its solver,
    and serves the subclasses of `material_cls` that no other physics names.
    """

    PHYSICS_TYPES_MAP: ClassVar[dict[type[Material], type["MaterialPhysics"]]]
    material_cls: ClassVar[type[Material]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "material_cls" in cls.__dict__:
            cls.PHYSICS_TYPES_MAP[cls.material_cls] = cls

    def __init__(self, material: Material) -> None:
        self.material = material

    @classmethod
    def from_material(cls, material: Material) -> "MaterialPhysics":
        """Build the physics of a material, bound to its options class or to the closest base class of it."""
        for options_cls in type(material).__mro__:
            physics_cls = cls.PHYSICS_TYPES_MAP.get(options_cls)
            if physics_cls is not None:
                return physics_cls(material)
        gs.raise_exception(
            f"No {cls.__name__} is registered for the material {type(material).__name__}. Define a subclass of "
            f"{cls.__name__} setting `material_cls = {type(material).__name__}`."
        )
