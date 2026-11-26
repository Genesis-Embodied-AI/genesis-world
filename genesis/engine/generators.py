"""The generation of the geometry of generator morphs, which the simulator runs as it adds their entities."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.terrain as tu
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.entities.rigid_entity.rigid_entity import compute_forward_kinematics
from genesis.options.morphs import Generator, LinkShellGenerator, MeshSet, Morph, Terrain

if TYPE_CHECKING:
    from genesis.engine.entities import Entity


class GeneratorContext(NamedTuple):
    """The scene that a generator morph may generate its geometry from, as the simulator passes it."""

    entities: Sequence["Entity"]
    """The entities of the scene added before the generated one, of every kind, which each generator filters."""


_GENERATE_FUNC_BY_MORPH_CLS: dict[type[Generator], Callable[[Generator, GeneratorContext], Morph]] = {}


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
        _GENERATE_FUNC_BY_MORPH_CLS[morph_cls] = generate_morph
        return generate_morph

    return _register


# The generated terrain is its generator with the height field set, so it holds every option of it and records none
@register_generator(Terrain)
def _generate_terrain(morph: Terrain, context: GeneratorContext) -> Terrain:
    if morph.height_field is not None:
        return morph
    return morph.model_copy(update={"height_field": tu.generate_height_field(morph)})


@register_generator(LinkShellGenerator)
def _generate_link_shell(morph: LinkShellGenerator, context: GeneratorContext) -> MeshSet:
    rigid_entities = [entity for entity in context.entities if isinstance(entity, RigidEntity)]
    rigid_links = [link for entity in rigid_entities for link in entity.links]
    # The volume stands around the links at the neutral configuration of their entities. A heterogeneous entity holds
    # the geoms of every variant on its links, which no single volume fits.
    links = [rigid_links[i_l] for i_l in morph.links]
    for link in links:
        if link.entity.desc.variants:
            gs.raise_exception(
                f"gs.morphs.LinkShellGenerator grows no volume around link '{link.name}' of a heterogeneous entity."
            )
        if not link.geoms:
            gs.raise_exception(f"Link '{link.name}' has no collision geom to grow a volume around.")
    links_pos, links_quat, *_ = compute_forward_kinematics(rigid_entities, n_envs=1)
    meshes, poss, eulers = [], [], []
    for link in links:
        for geom in link.geoms:
            pos, quat = gu.transform_pos_quat_by_trans_quat(
                geom.init_pos, geom.init_quat, links_pos[0, link.idx], links_quat[0, link.idx]
            )
            verts = geom.init_verts + geom.init_normals * morph.thickness
            meshes.append(trimesh.Trimesh(vertices=verts, faces=geom.init_faces))
            poss.append(tuple(pos))
            eulers.append(tuple(gu.quat_to_xyz(quat, rpy=True, degrees=True)))
    return MeshSet(
        files=meshes,
        poss=poss,
        eulers=eulers,
    )


def generate(morph: Generator, context: GeneratorContext) -> Morph:
    """Return the morph holding the geometry that a generator morph generates, which its entity is built from.

    Parameters
    ----------
    morph : Generator
        The generator morph of the entity spec.
    context : GeneratorContext
        The scene the geometry may be generated from.
    """
    # A subclass of a registered generator morph is generated as its closest registered ancestor
    generate_morph = next(
        (
            _GENERATE_FUNC_BY_MORPH_CLS[morph_cls]
            for morph_cls in type(morph).__mro__
            if morph_cls in _GENERATE_FUNC_BY_MORPH_CLS
        ),
        None,
    )
    if generate_morph is None:
        gs.raise_exception(
            f"No generation is registered for gs.morphs.{type(morph).__name__}. Register one with gs.register_generator."
        )
    return generate_morph(morph, context)
