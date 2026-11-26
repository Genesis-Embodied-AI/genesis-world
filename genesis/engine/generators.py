"""The generation of the geometry of generator morphs, which the simulator runs as it adds their entities."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import genesis as gs
import genesis.utils.terrain as tu
from genesis.options.morphs import Generator, Morph, Terrain

if TYPE_CHECKING:
    from genesis.engine.entities import Entity


class GeneratorContext(NamedTuple):
    """The scene that a generator morph may generate its geometry from, as the simulator passes it."""

    entities: Sequence["Entity"]
    """The entities of the scene added before the generated one, of every kind, which each generator filters."""


def _generate_terrain(morph: Terrain, context: GeneratorContext) -> Terrain:
    if morph.height_field is not None:
        return morph
    return morph.model_copy(update={"height_field": tu.generate_height_field(morph)})


_GENERATE_BY_MORPH_CLS: dict[type[Generator], Callable[[Generator, GeneratorContext], Morph]] = {
    Terrain: _generate_terrain,
}


def generate(morph: Generator, context: GeneratorContext) -> Morph:
    """Return the morph holding the geometry that a generator morph generates, which its entity is built from.

    Parameters
    ----------
    morph : Generator
        The generator morph of the entity spec.
    context : GeneratorContext
        The scene the geometry may be generated from.
    """
    generate_morph = _GENERATE_BY_MORPH_CLS.get(type(morph))
    if generate_morph is None:
        gs.raise_exception(f"No generation is defined for gs.morphs.{type(morph).__name__}.")
    return generate_morph(morph, context)
