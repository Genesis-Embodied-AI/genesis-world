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


_GENERATE_BY_MORPH_CLS: dict[type[Generator], Callable[[Generator, GeneratorContext], Morph]] = {}


def register_generator(morph_cls: type[Generator]) -> Callable[[Callable], Callable]:
    """Register the function generating the geometry of a kind of generator morph, as a decorator.

    The function takes the generator morph and the scene (see 'GeneratorContext') and returns the morph its entity is
    built from, which the kind of entity of the spec must load, such as a ``gs.morphs.MJCF`` holding an inline model
    for a rigid entity. A user defines a generator by subclassing ``gs.morphs.Generator`` and registering its function::

        class Chain(gs.morphs.Generator):
            n_links: int

        @gs.register_generator(Chain)
        def generate_chain(morph: Chain, context: gs.engine.generators.GeneratorContext) -> gs.morphs.MJCF: ...

    Parameters
    ----------
    morph_cls : type[gs.morphs.Generator]
        The kind of generator morph whose geometry the decorated function generates.
    """

    def _register(generate_morph: Callable[[Generator, GeneratorContext], Morph]) -> Callable:
        _GENERATE_BY_MORPH_CLS[morph_cls] = generate_morph
        return generate_morph

    return _register


# The generated terrain is its generator with the height field set, so it holds every option of it and records none
@register_generator(Terrain)
def _generate_terrain(morph: Terrain, context: GeneratorContext) -> Terrain:
    if morph.height_field is not None:
        return morph
    return morph.model_copy(update={"height_field": tu.generate_height_field(morph)})


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
        gs.raise_exception(
            f"No generation is registered for gs.morphs.{type(morph).__name__}. Register one with gs.register_generator."
        )
    return generate_morph(morph, context)
