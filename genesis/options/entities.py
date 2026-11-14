"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

from collections.abc import Iterable
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Self, TypeVar, overload

from pydantic import Field, InstanceOf, PrivateAttr, StrictBool, model_validator

import genesis as gs
from genesis.engine import materials
from genesis.engine.materials.base import Material
from genesis.repr_base import get_public_name

from . import indices
from .blocks import Geom, SelectionBlock, Vertex, _FrozenOptions
from .morphs import Morph
from .options import Options
from .surfaces import Surface

if TYPE_CHECKING:
    from genesis.engine.entities import (
        Entity,
        FEMEntity,
        HybridEntity,
        KinematicEntity,
        MPMEntity,
        RigidEntity,
        SPHEntity,
    )
    from genesis.engine.entities.pbd_entity import PBDBaseEntity


EntityT = TypeVar("EntityT", bound="Entity")


def get_entity_options_cls(material: Material) -> type["BaseEntity"]:
    """Return the class of entity options of the solver simulating a material, raising if no solver simulates it.

    A material goes to the class of the nearest of its bases that the solvers of Genesis simulate, so a subclass a user
    declares goes where the material it derives from goes.
    """
    for material_cls in type(material).__mro__:
        options_cls = _OPTIONS_CLS_BY_MATERIAL_CLS.get(material_cls)
        if options_cls is not None:
            return options_cls
    gs.raise_exception(f"No entity options admit material {type(material).__name__}.")


def get_entity_options(
    morph: Morph | Iterable[Morph] | None = None,
    material: Material | None = None,
    surface: Surface | None = None,
    visualize_contact: bool | None = None,
    vis_mode: str | None = None,
    name: str | None = None,
) -> "BaseEntity":
    """Return the entity options that the individual arguments of ``Scene.add_entity`` stand for.

    The material decides the class of the options (see ``get_entity_options_cls``), ``gs.entities.Rigid`` if None. Each
    argument given fills the field of the options it stands for, 'vis_mode' filling that of a copy of the surface, and
    the options refuse an argument their class has no field for.
    """
    options_cls = Rigid if material is None else get_entity_options_cls(material)
    # Heterogeneous morphs may come as any iterable of morphs
    if isinstance(morph, Iterable) and not isinstance(morph, Morph):
        morph = tuple(morph)
    if vis_mode is not None:
        surface = (gs.surfaces.Default() if surface is None else surface).model_copy(update={"vis_mode": vis_mode})
    # Only rigid entities have contacts to draw, and 'False' asks for nothing, so every kind of entity accepts it
    if "visualize_contact" not in options_cls.model_fields:
        if visualize_contact:
            gs.raise_exception(
                f"'visualize_contact' only applies to rigid entities, not to gs.entities.{options_cls.__name__}."
            )
        visualize_contact = None
    fields = {"morph": morph, "material": material, "surface": surface, "visualize_contact": visualize_contact}
    return options_cls(**{field: value for field, value in {**fields, "name": name}.items() if value is not None})


class BaseEntity(_FrozenOptions, Generic[EntityT]):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits,
    the selectors of its parts and the blocks of quantities set on them. The fields are fixed at construction, while
    ``select`` records into the options the quantities set on parts of the entity, where the kind of entity has parts.
    The entity holds a frozen copy, which records nothing more, so later records leave the entity as it is. A copy
    of the options (``model_copy()``) records again, starting from the blocks recorded so far.

    Parameters
    ----------
    morph : Morph | tuple[Morph, ...] | None, optional
        The geometry or asset of the entity. A sequence of morphs makes the entity heterogeneous: each parallel
        environment simulates one of them (see ``Scene.add_entity``). It must be set once the options reach
        ``Scene.add_entity``, while the options given to ``Scene.add_emitter`` leave it None, since the emitter sets it.
    surface : Surface | None, optional
        The appearance of the entity, its visualization mode included. If None, use ``gs.surfaces.Default()``.
    name : str | None, optional
        The name of the entity, unique within the scene. If None, a name is generated from the morph.
    selection_blocks : tuple[gs.options.SelectionBlock, ...], optional
        The blocks of quantities set on selections of parts of the entity, in the order they were set, which ``select``
        and its setters alone record (see ``select``). Default is none.
    """

    # Whether the kind of entity admits several morphs, one per variant of a heterogeneous entity
    _admits_variants: ClassVar[bool] = False
    # Selectors of the parts of the entity, from the coarsest level to the finest, and blocks of the quantities set on
    # them, that the kind of entity admits
    _selector_types: ClassVar[tuple[type[indices.Link | indices.Geom | indices.Vertex], ...]] = ()
    _block_types: ClassVar[tuple[type[Geom | Vertex], ...]] = ()

    # Checked by instance alone, since validation would run the validators of 'Morph' on the sequence of morphs
    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(min_length=1, strict=False)] | None = None
    surface: Surface | None = None
    name: str | None = None
    selection_blocks: tuple[SelectionBlock, ...] = ()

    _is_frozen: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _check_material_solver(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        material = data.get("material")
        if isinstance(material, Material) and "material" in cls.model_fields:
            options_cls = get_entity_options_cls(material)
            if not issubclass(cls, options_cls):
                gs.raise_exception(
                    f"gs.entities.{cls.__name__} cannot take material {type(material).__name__} of another solver. "
                    f"Pass it to gs.entities.{options_cls.__name__} instead."
                )
        return data

    @model_validator(mode="after")
    def _check_selection_blocks(self) -> Self:
        if isinstance(self.morph, tuple) and not self._admits_variants:
            gs.raise_exception(
                f"An entity of gs.entities.{type(self).__name__} is created from one morph, got {len(self.morph)}."
            )
        for selection_block in self.selection_blocks:
            self._check_selection_block(selection_block)
        return self

    # A copy records again, frozen or not
    def __copy__(self) -> Self:
        options = super().__copy__()
        options._is_frozen = False
        return options

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        options = super().__deepcopy__(memo)
        options._is_frozen = False
        return options

    def _check_selection_block(self, selection_block: SelectionBlock) -> None:
        # Each selector selects at the level of the one before it or a finer one
        levels = []
        for selector in selection_block.selectors:
            if not isinstance(selector, self._selector_types):
                gs.raise_exception(
                    f"gs.entities.{type(self).__name__} selects no part with gs.{get_public_name(type(selector))}."
                )
            levels.append(next(i for i, cls in enumerate(self._selector_types) if isinstance(selector, cls)))
        if any(level < previous_level for previous_level, level in zip(levels, levels[1:])):
            gs.raise_exception(
                f"A selection selects at its own level or a finer one, got {selection_block.selection_repr}."
            )
        if not isinstance(selection_block.block, self._block_types):
            gs.raise_exception(
                f"gs.entities.{type(self).__name__} sets no gs.options.{type(selection_block.block).__name__} on its "
                "parts."
            )

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant of a heterogeneous entity and a single one otherwise."""
        if self.morph is None:
            return ()
        return self.morph if isinstance(self.morph, tuple) else (self.morph,)

    def _frozen_copy(self, update: dict[str, Any] | None = None) -> Self:
        """Return a copy of these options, with the fields 'update' gives, that records no more blocks."""
        options = self.model_copy(update=update)
        options._is_frozen = True
        return options

    def _record(self, selection_blocks: tuple[SelectionBlock, ...]) -> None:
        """Record blocks after those recorded so far, each checked as at construction."""
        if self._is_frozen:
            gs.raise_exception(
                "The options an entity holds are frozen. Select parts on a copy of them ('model_copy()') instead."
            )
        for selection_block in selection_blocks:
            self._check_selection_block(selection_block)
        # The field is frozen for users, so that 'select' and its setters are the only way to record a block
        object.__setattr__(self, "selection_blocks", (*self.selection_blocks, *selection_blocks))
        self.__pydantic_fields_set__.add("selection_blocks")


OptionsT = TypeVar("OptionsT", bound=BaseEntity)


class _PartSelection(Generic[OptionsT]):
    """A selection of parts of an entity, recording into its entity options the quantities set on it.

    It holds the entity options and the selector of each level leading to the parts, and owns no data. Every setter
    returns the selection, so that calls chain on it, ``end`` returns the entity options to start another selection
    from, and ``with`` binds the selection to a name for a block.
    """

    def __init__(self, options: OptionsT, selectors: tuple[indices.Link | indices.Geom | indices.Vertex, ...]) -> None:
        self._options = options
        self._selectors = selectors

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None

    def end(self) -> OptionsT:
        """Return the entity options this selection records into, however deeply it is nested.

        It ends a chain inline, so that one expression configures several selections and gives the options to
        ``add_entity``: ``options.select(top).set_is_fixed(True).end().select(bottom).set_is_fixed(True).end()``.
        """
        return self._options

    def _set(self, blocks: tuple[Geom | Vertex, ...]) -> Self:
        self._options._record(tuple(SelectionBlock(selectors=self._selectors, block=block) for block in blocks))
        return self


class _GeomSetters(_PartSelection[OptionsT]):
    """The setters of the quantities of collision geoms, which a link selection applies to every geom of its links."""

    def set(self, *blocks: Geom) -> Self:
        """Set on the selected geoms the fields each block gives explicitly, a later call overriding an earlier one."""
        return self._set(blocks)

    def set_rho(self, rho: float) -> Self:
        """Set the density of the selected collision geoms (see ``gs.options.Geom.rho``)."""
        return self._set((Geom(rho=rho),))

    def set_friction(self, friction: float) -> Self:
        """Set the sliding friction coefficient of the selected collision geoms (see ``gs.options.Geom.friction``)."""
        return self._set((Geom(friction=friction),))

    def set_friction_torsional(self, friction_torsional: float) -> Self:
        """Set the torsional friction coefficient of the selected collision geoms."""
        return self._set((Geom(friction_torsional=friction_torsional),))

    def set_friction_rolling(self, friction_rolling: float) -> Self:
        """Set the rolling friction coefficient of the selected collision geoms."""
        return self._set((Geom(friction_rolling=friction_rolling),))


class GeomSelection(_GeomSetters[OptionsT]):
    """A selection of collision geoms of an entity (see ``gs.entities.Rigid.select``)."""

    def select(self, selector: indices.Geom) -> "GeomSelection[OptionsT]":
        """Return the selection of the geoms of this selection that the selector holds, resolved among them.

        Two rules decide which value a field of a geom takes:

        1. A value set on a finer level overrides one set on a coarser level: the material, then a link selection, then
           a geom selection. The level is that of the finest selector, however the selection is written.
        2. Between two selections of one level, the later call applies.
        """
        if not isinstance(selector, indices.Geom):
            gs.raise_exception(
                f"A geom selection selects with gs.indices.Geom, got gs.{get_public_name(type(selector))}."
            )
        return GeomSelection(self._options, (*self._selectors, selector))


class LinkSelection(_GeomSetters[OptionsT]):
    """A selection of links of an entity, whose geom setters apply to every geom of the links (see ``select``)."""

    @overload
    def select(self, selector: indices.Link) -> "LinkSelection[OptionsT]": ...

    @overload
    def select(self, selector: indices.Geom) -> GeomSelection[OptionsT]: ...

    def select(self, selector: indices.Link | indices.Geom) -> "LinkSelection[OptionsT] | GeomSelection[OptionsT]":
        """Return the selection of the links, or the geoms, of this selection that the selector holds.

        The selector resolves among these links, and a geom ordinal counts each link apart (see ``gs.indices.Geom.at``).

        Two rules decide which value a field of a geom takes:

        1. A value set on a finer level overrides one set on a coarser level: the material, then a link selection, then
           a geom selection. The level is that of the finest selector, however the selection is written.
        2. Between two selections of one level, the later call applies.
        """
        if isinstance(selector, indices.Geom):
            return GeomSelection(self._options, (*self._selectors, selector))
        if not isinstance(selector, indices.Link):
            gs.raise_exception(
                "A link selection selects with gs.indices.Link or gs.indices.Geom, got "
                f"gs.{get_public_name(type(selector))}."
            )
        return LinkSelection(self._options, (*self._selectors, selector))


class VertexSelection(_PartSelection[OptionsT]):
    """A selection of vertices of a deformable entity (see ``gs.entities.FEM.select``)."""

    def select(self, selector: indices.Vertex) -> "VertexSelection[OptionsT]":
        """Return the selection of the vertices of this selection that the selector holds, resolved among them.

        Two rules decide which value a field of a vertex takes:

        1. A value set on a vertex selection overrides the material, however the selection is written.
        2. Between two vertex selections, the later call applies.
        """
        if not isinstance(selector, indices.Vertex):
            gs.raise_exception(
                f"A vertex selection selects with gs.indices.Vertex, got gs.{get_public_name(type(selector))}."
            )
        return VertexSelection(self._options, (*self._selectors, selector))

    def set(self, *blocks: Vertex) -> Self:
        """Set on the selected vertices the fields each block gives explicitly, overriding an earlier call."""
        return self._set(blocks)

    def set_is_fixed(self, is_fixed: bool) -> Self:
        """Set whether the selected vertices stay at their initial positions (see ``gs.options.Vertex.is_fixed``)."""
        return self._set((Vertex(is_fixed=is_fixed),))


class Rigid(BaseEntity["RigidEntity"]):
    """Options for an entity simulated with rigid-body dynamics.

    Parameters
    ----------
    material : gs.materials.Rigid, optional
        The material of the entity. Defaults to ``gs.materials.Rigid()``.
    visualize_contact : bool, optional
        Whether to draw the contact forces applied to the entity as arrows in the viewer and rendered images. Each
        arrow is drawn every frame, so reserve it for the entities being inspected. Default is False.
    """

    _admits_variants = True
    _selector_types = (indices.Link, indices.Geom)
    _block_types = (Geom,)

    material: materials.Rigid = Field(default_factory=materials.Rigid)
    visualize_contact: StrictBool = False

    @overload
    def select(self, selector: indices.Link) -> LinkSelection[Self]: ...

    @overload
    def select(self, selector: indices.Geom) -> GeomSelection[Self]: ...

    def select(self, selector: indices.Link | indices.Geom) -> LinkSelection[Self] | GeomSelection[Self]:
        """Select the links, or the collision geoms, that the selector holds, to set the quantities of geoms on.

        Selectors combine with ``|`` and ``&``, each resolving over the whole entity. A selection selects again with
        ``select``, at its own level or a finer one, the nested selector resolving among the parts the selection holds:
        ``options.select(fingers).select(gs.indices.Geom.at(slice(1, 6)))`` holds geoms 1 to 5 of each finger, and
        ``options.select(fingers).select(gs.indices.Geom.named("pad*")).select(gs.indices.Geom.at(0))`` the first pad of
        each finger. A quantity set on links applies to every collision geom of those links.

        Two rules decide which value a field of a geom takes:

        1. A value set on a finer level overrides one set on a coarser level: the material, then a link selection, then
           a geom selection. The level is that of the finest selector of the selection, so how a selection is written
           (nesting, ``|``, ``&``, several names) never makes it more specific.
        2. Between two selections of one level, the later call applies. The three calls here select links, so the thumb
           ends at 1.0::

               options.select(gs.indices.Link.subtree("hand")).select(gs.indices.Link.named("thumb")).set_friction(2.0)
               options.select(gs.indices.Link.subtree("hand") & gs.indices.Link.named("thumb")).set_friction(3.0)
               options.select(gs.indices.Link.subtree("hand")).set_friction(1.0)

        A selection that holds no geom of the entity, or of one of its heterogeneous variants, raises in ``add_entity``.
        """
        if not isinstance(selector, self._selector_types):
            gs.raise_exception(
                f"gs.entities.Rigid selects with gs.indices.Link or gs.indices.Geom, got "
                f"gs.{get_public_name(type(selector))}."
            )
        if isinstance(selector, indices.Geom):
            return GeomSelection(self, (selector,))
        return LinkSelection(self, (selector,))


class Kinematic(BaseEntity["KinematicEntity"]):
    """Options for an entity simulated with kinematics only, with no collision.

    Parameters
    ----------
    material : gs.materials.Kinematic, optional
        The material of the entity. Defaults to ``gs.materials.Kinematic()``.
    """

    _admits_variants = True

    material: materials.Kinematic = Field(default_factory=materials.Kinematic)


class FEM(BaseEntity["FEMEntity"]):
    """Options for an entity simulated with the finite element method (FEM).

    Parameters
    ----------
    material : gs.materials.FEM.Base
        One of ``gs.materials.FEM.Elastic``, ``Muscle`` or ``Cloth``, a Cloth entity alone refusing any selection.
    """

    _selector_types = (indices.Vertex,)
    _block_types = (Vertex,)

    material: materials.FEM.Base

    def _check_selection_block(self, selection_block: SelectionBlock) -> None:
        if isinstance(self.material, materials.FEM.Cloth):
            gs.raise_exception("gs.entities.FEM cannot honour quantities of vertices for a Cloth material.")
        super()._check_selection_block(selection_block)

    def select(self, selector: indices.Vertex) -> VertexSelection[Self]:
        """Select the vertices of the volume mesh that the selector holds, to set quantities on.

        Selectors combine with ``|`` and ``&``, each resolving over the whole mesh. A selection selects again with
        ``select``, the nested selector resolving among the vertices the selection holds. With
        ``top = gs.indices.Vertex.in_box(lower, upper)`` and
        ``near = gs.indices.Vertex.closest_to(clamp, max_distance=0.01)``, ``options.select(top).select(near)`` holds
        the vertex of the top face nearest the clamp, where ``options.select(top & near)`` holds the vertex of the whole
        mesh nearest the clamp, if it lies on the top face.

        Two rules decide which value a field of a vertex takes:

        1. A value set on a vertex selection overrides the material. Vertices are a single level, so how a selection is
           written (nesting, ``|``, ``&``) never makes it more specific.
        2. Between two vertex selections, the later call applies.

        A selection that holds no vertex raises in ``add_entity``.
        """
        if not isinstance(selector, indices.Vertex):
            gs.raise_exception(
                f"gs.entities.FEM selects with gs.indices.Vertex, got gs.{get_public_name(type(selector))}."
            )
        return VertexSelection(self, (selector,))


class MPM(BaseEntity["MPMEntity"]):
    """Options for an entity simulated with the material point method (MPM).

    Parameters
    ----------
    material : gs.materials.MPM.Base
        One of ``gs.materials.MPM.Elastic``, ``ElastoPlastic``, ``Liquid``, ``Sand``, ``Snow`` or ``Muscle``.
    """

    material: materials.MPM.Base


class SPH(BaseEntity["SPHEntity"]):
    """Options for an entity simulated with smoothed particle hydrodynamics (SPH).

    Parameters
    ----------
    material : gs.materials.SPH.Base, optional
        The material of the liquid. Defaults to ``gs.materials.SPH.Liquid()``.
    """

    material: materials.SPH.Base = Field(default_factory=materials.SPH.Liquid)


class PBD(BaseEntity["PBDBaseEntity"]):
    """Options for an entity simulated with position-based dynamics (PBD).

    Parameters
    ----------
    material : gs.materials.PBD.Base
        The material of the entity, one of ``gs.materials.PBD.Cloth``, ``Elastic``, ``Liquid`` or ``Particle``.
    """

    material: materials.PBD.Base


class Hybrid(BaseEntity["HybridEntity"]):
    """Options for an entity simulated with a rigid skeleton inside a material point method (MPM) skin.

    Parameters
    ----------
    material : gs.materials.Hybrid
        The material of the skeleton and of the skin, and how they are coupled.
    """

    material: materials.Hybrid


# The class of entity options each material class goes to, the one each options class declares for its material. A
# material takes the class of the nearest of its bases (see 'get_entity_options_cls').
_OPTIONS_CLS_BY_MATERIAL_CLS: dict[type[Any] | None, type[BaseEntity]] = {
    options_cls.model_fields["material"].annotation: options_cls
    for options_cls in (Rigid, Kinematic, FEM, MPM, SPH, PBD, Hybrid)
}
