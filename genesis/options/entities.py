"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Self, TypeVar, overload

from pydantic import ConfigDict, Field, InstanceOf, PrivateAttr, StrictBool, model_validator

import genesis as gs
from genesis.engine import materials
from genesis.engine.materials.base import Material
from genesis.typing import ValidFloat

from .indices import GeomIndex, LinkIndex, VertexIndex
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

Selector = LinkIndex | GeomIndex | VertexIndex


def get_entity_options_cls(material: Material) -> type["BaseEntity"] | None:
    """Return the class of entity options of the solver simulating a material, or None if no solver simulates it.

    A material goes to the class of the nearest of its bases that the solvers of Genesis simulate, so a subclass a user
    declares goes where the material it derives from goes.
    """
    for material_cls in type(material).__mro__:
        options_cls = _OPTIONS_CLS_BY_MATERIAL_CLS.get(material_cls)
        if options_cls is not None:
            return options_cls
    return None


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
    if options_cls is None:
        gs.raise_exception(f"No entity options admit material {type(material).__name__}.")
    # Heterogeneous morphs may come as any iterable of morphs
    if isinstance(morph, Iterable) and not isinstance(morph, Morph):
        morph = tuple(morph)
    if vis_mode is not None:
        surface = (gs.surfaces.Default() if surface is None else surface).model_copy(update={"vis_mode": vis_mode})
    fields = {"morph": morph, "material": material, "surface": surface, "visualize_contact": visualize_contact}
    return options_cls(**{field: value for field, value in {**fields, "name": name}.items() if value is not None})


class _FrozenOptions(Options):
    # Frozen, since assigning a field skips the validators
    model_config = ConfigDict(frozen=True)


class Geom(_FrozenOptions):
    """Quantities of the collision geoms of a rigid entity, set on a selection of its links or geoms.

    The fields given explicitly override those of the entity material for the selected geoms (see ``Rigid.select``).
    The fields left unset keep theirs.

    Parameters
    ----------
    rho : float | None, optional
        The density (kg/m^3). It estimates the mass and the inertia of a link the asset states none for, or of every
        moving link under the 'recompute_inertia' option of the morph. An aligned body is then simulated unaligned.
        The IPC coupler rejects it. If None, the geoms keep the density of the material. Default is None.
    friction : float | None, optional
        The sliding friction coefficient of the contacts of the geoms (see ``gs.materials.Rigid.friction``). If None,
        the geoms keep that of the material. Default is None.
    friction_torsional : float | None, optional
        The torsional friction coefficient (see ``gs.materials.Rigid.friction_torsional``). If None, the geoms keep
        that of the material. Default is None.
    friction_rolling : float | None, optional
        The rolling friction coefficient (see ``gs.materials.Rigid.friction_rolling``). If None, the geoms keep that of
        the material. Default is None.
    """

    rho: ValidFloat | None = None
    friction: Annotated[ValidFloat, Field(ge=0.01, le=5.0)] | None = None
    friction_torsional: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_rolling: Annotated[ValidFloat, Field(ge=0.0)] | None = None


class Vertex(_FrozenOptions):
    """Quantities of the vertices of a deformable entity, set on a selection of its vertices.

    Parameters
    ----------
    is_fixed : bool, optional
        Whether the vertices stay at their initial positions, as a hard constraint the solver enforces at every
        substep. It holds an entity in place by a few vertices, such as a soft pad clamped by one face. The vertices are
        vertex constraints from the build on, which ``FEMEntity.remove_vertex_constraints`` releases like any other. The
        IPC coupler does not support it, so a Cloth material rejects it. Default is False.
    """

    is_fixed: StrictBool = False


class Setting(_FrozenOptions):
    """Quantities set on a selection of parts of an entity, as ``select`` records them into its entity options.

    Parameters
    ----------
    selectors : tuple[tuple[gs.LinkIndex | gs.GeomIndex | gs.VertexIndex, ...], ...]
        The groups of selectors of the parts. A part is selected when every group holds it, and a group holds the parts
        any of its selectors holds. The selectors of one group are of one kind.
    block : gs.options.Geom | gs.options.Vertex
        The quantities set on the parts, of which the fields given explicitly apply.
    """

    selectors: Annotated[
        tuple[Annotated[tuple[LinkIndex | GeomIndex | VertexIndex, ...], Field(min_length=1, strict=False)], ...],
        Field(min_length=1, strict=False),
    ]
    block: Geom | Vertex

    @model_validator(mode="after")
    def _check_setting(self) -> Self:
        for group in self.selectors:
            if len({type(selector) for selector in group}) > 1:
                gs.raise_exception(
                    f"A list of selectors holds the parts any of them holds, so they select one kind of part, got "
                    f"{[type(selector).__name__ for selector in group]}."
                )
        if not self.block.model_fields_set:
            gs.raise_exception(f"gs.options.{type(self.block).__name__} sets no quantity. Give at least one field.")
        return self

    @property
    def selection_repr(self) -> str:
        """The selection of the setting on one line, for error messages."""
        return " and ".join(
            "["
            + ", ".join(f"{type(selector).__name__}{selector.model_dump(exclude_none=True)}" for selector in group)
            + "]"
            for group in self.selectors
        )


class BaseEntity(_FrozenOptions, Generic[EntityT]):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits,
    the selectors of its parts and the blocks of quantities set on them. The fields are fixed at construction, while
    ``select`` records into the options the quantities set on parts of the entity, where the kind of entity has parts.
    ``Scene.add_entity`` keeps a frozen copy, which records nothing more, so later records leave the entity as it is. A
    copy of the options (``model_copy()``) records again, starting from the settings recorded so far.

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
    settings : tuple[gs.options.Setting, ...], optional
        The quantities set on selections of parts of the entity, in the order they were set (see ``select``). Default
        is none.
    """

    # Selectors of the parts of the entity, and blocks of the quantities set on them, that the kind of entity admits
    _selector_types: ClassVar[tuple[type[LinkIndex | GeomIndex | VertexIndex], ...]] = ()
    _block_types: ClassVar[tuple[type[Geom | Vertex], ...]] = ()

    # Checked by instance alone, since validation would run the validators of 'Morph' on the sequence of morphs
    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(strict=False)] | None = None
    surface: Surface | None = None
    name: str | None = None
    settings: tuple[Setting, ...] = ()

    _is_frozen: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _check_material_solver(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        material = data.get("material")
        if isinstance(material, Material) and "material" in cls.model_fields:
            options_cls = get_entity_options_cls(material)
            if options_cls is None or not issubclass(cls, options_cls):
                options_name = "the entity options of its solver"
                if options_cls is not None:
                    options_name = f"gs.entities.{options_cls.__name__}"
                gs.raise_exception(
                    f"gs.entities.{cls.__name__} cannot take material {type(material).__name__} of another solver. "
                    f"Pass it to {options_name} instead."
                )
        return data

    @model_validator(mode="after")
    def _check_settings(self) -> Self:
        for setting in self.settings:
            for group in setting.selectors:
                if not isinstance(group[0], self._selector_types):
                    gs.raise_exception(
                        f"gs.entities.{type(self).__name__} selects no part with gs.{type(group[0]).__name__}."
                    )
            if not isinstance(setting.block, self._block_types):
                gs.raise_exception(
                    f"gs.entities.{type(self).__name__} sets no gs.options.{type(setting.block).__name__} on its parts."
                )
        return self

    def __copy__(self) -> Self:
        options = super().__copy__()
        options._is_frozen = False
        return options

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant of a heterogeneous entity and a single one otherwise."""
        if self.morph is None:
            return ()
        return self.morph if isinstance(self.morph, tuple) else (self.morph,)

    def _frozen_copy(self, update: dict[str, Any] | None = None) -> Self:
        """Return a copy of these options, with the fields 'update' gives, that records no more settings."""
        options = self.model_copy(update=update)
        options._is_frozen = True
        return options

    def _record(self, settings: tuple[Setting, ...]) -> None:
        """Record settings after those recorded so far, validated as the options would be at construction."""
        if self._is_frozen:
            gs.raise_exception(
                "The options an entity holds are frozen. Select parts on a copy of them ('model_copy()') instead."
            )
        # Every validator runs on the options holding the new settings, whose settings are then adopted
        options = type(self)(**{**dict(self), "settings": (*self.settings, *settings)})
        object.__setattr__(self, "settings", options.settings)
        self.__pydantic_fields_set__.add("settings")


OptionsT = TypeVar("OptionsT", bound=BaseEntity)


class _PartSelection(Generic[OptionsT]):
    """A selection of parts of an entity, recording into its entity options the quantities set on it.

    It holds the entity options and the selectors leading to the parts, and owns no data. Every setter returns the
    selection, so that calls chain on it. ``select`` narrows the selection to the parts that the given selectors hold
    too, at the level of the selection or a finer one, and ``with`` binds the selection to a name for a block.
    """

    def __init__(
        self, options: OptionsT, selectors: tuple[tuple[LinkIndex | GeomIndex | VertexIndex, ...], ...]
    ) -> None:
        self._options = options
        self._selectors = selectors

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None

    def _set(self, blocks: tuple[Geom | Vertex, ...]) -> Self:
        self._options._record(tuple(Setting(selectors=self._selectors, block=block) for block in blocks))
        return self


def _get_selector_groups(
    selectors: tuple["Selector | Sequence[Selector]", ...], selector_types: tuple[type["Selector"], ...]
) -> tuple[type["Selector"], tuple[tuple["Selector", ...], ...]]:
    """Return the type of the selectors given to 'select', and their groups, a list holding what any member holds.

    The selectors are of one type among 'selector_types', the selectors of a level the selection admits.
    """
    groups = tuple(
        (selector,) if isinstance(selector, (LinkIndex, GeomIndex, VertexIndex)) else tuple(selector)
        for selector in selectors
    )
    selector_types_given = {type(selector) for group in groups for selector in group}
    if len(selector_types_given) != 1:
        gs.raise_exception(
            "'select' takes selectors of one type, and selections of different types nest. Got "
            f"{sorted(selector_type.__name__ for selector_type in selector_types_given)}."
        )
    (selector_type,) = selector_types_given
    if not issubclass(selector_type, selector_types):
        gs.raise_exception(
            f"This selection selects with {[selector_type.__name__ for selector_type in selector_types]}, got "
            f"{selector_type.__name__}."
        )
    return selector_type, groups


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

    def select(self, *selectors: GeomIndex | Sequence[GeomIndex]) -> "GeomSelection[OptionsT]":
        """Return the selection of the geoms of this selection that every argument holds."""
        _, groups = _get_selector_groups(selectors, (GeomIndex,))
        return GeomSelection(self._options, (*self._selectors, *groups))


class LinkSelection(_GeomSetters[OptionsT]):
    """A selection of links of an entity, whose geom setters apply to every geom of the links (see ``select``)."""

    @overload
    def select(self, *selectors: LinkIndex | Sequence[LinkIndex]) -> "LinkSelection[OptionsT]": ...

    @overload
    def select(self, *selectors: GeomIndex | Sequence[GeomIndex]) -> GeomSelection[OptionsT]: ...

    def select(
        self, *selectors: LinkIndex | GeomIndex | Sequence[LinkIndex] | Sequence[GeomIndex]
    ) -> "LinkSelection[OptionsT] | GeomSelection[OptionsT]":
        """Return the selection of the links, or the geoms, of this selection that every argument holds."""
        selector_type, groups = _get_selector_groups(selectors, (LinkIndex, GeomIndex))
        if selector_type is GeomIndex:
            return GeomSelection(self._options, (*self._selectors, *groups))
        return LinkSelection(self._options, (*self._selectors, *groups))


class VertexSelection(_PartSelection[OptionsT]):
    """A selection of vertices of a deformable entity (see ``gs.entities.FEM.select``)."""

    def select(self, *selectors: VertexIndex | Sequence[VertexIndex]) -> "VertexSelection[OptionsT]":
        """Return the selection of the vertices of this selection that every argument holds."""
        _, groups = _get_selector_groups(selectors, (VertexIndex,))
        return VertexSelection(self._options, (*self._selectors, *groups))

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

    _selector_types = (LinkIndex, GeomIndex)
    _block_types = (Geom,)

    material: materials.Rigid = Field(default_factory=materials.Rigid)
    visualize_contact: StrictBool = False

    @overload
    def select(self, *selectors: LinkIndex | Sequence[LinkIndex]) -> LinkSelection[Self]: ...

    @overload
    def select(self, *selectors: GeomIndex | Sequence[GeomIndex]) -> GeomSelection[Self]: ...

    def select(
        self, *selectors: LinkIndex | GeomIndex | Sequence[LinkIndex] | Sequence[GeomIndex]
    ) -> LinkSelection[Self] | GeomSelection[Self]:
        """Select the links, or the collision geoms, that every argument holds, to set quantities of their geoms on.

        An argument is a selector, or a list of selectors holding the parts any of them holds, all of one type. Select
        links and geoms together by nesting: ``options.select(fingers).select(gs.GeomIndex.named("pad*"))``.

        A quantity set on links applies to every collision geom of those links. For each field of each geom, a value
        set on a geom selection overrides a value set on links, which overrides the material. Between two selections of
        one of these levels, the later one applies. A selection that holds no geom of the entity, or of one of its
        heterogeneous variants, raises when the entity is added.
        """
        selector_type, groups = _get_selector_groups(selectors, self._selector_types)
        if selector_type is GeomIndex:
            return GeomSelection(self, groups)
        return LinkSelection(self, groups)


class Kinematic(BaseEntity["KinematicEntity"]):
    """Options for an entity simulated with kinematics only, with no collision.

    Parameters
    ----------
    material : gs.materials.Kinematic, optional
        The material of the entity. Defaults to ``gs.materials.Kinematic()``.
    """

    material: materials.Kinematic = Field(default_factory=materials.Kinematic)


class FEM(BaseEntity["FEMEntity"]):
    """Options for an entity simulated with the finite element method (FEM).

    Parameters
    ----------
    material : gs.materials.FEM.Base
        The material of the entity, one of ``gs.materials.FEM.Elastic``, ``Cloth`` or ``Muscle``. A Cloth material
        takes no settings.
    """

    _selector_types = (VertexIndex,)
    _block_types = (Vertex,)

    material: materials.FEM.Base

    @model_validator(mode="after")
    def _check_cloth_settings(self) -> Self:
        if isinstance(self.material, materials.FEM.Cloth) and self.settings:
            gs.raise_exception("gs.entities.FEM cannot honour settings of vertices for a Cloth material.")
        return self

    def select(self, *selectors: VertexIndex | Sequence[VertexIndex]) -> VertexSelection[Self]:
        """Select the vertices of the volume mesh that every argument holds, to set quantities on.

        An argument is a selector, or a list of selectors holding the vertices any of them holds. Between two selections
        setting one field of one vertex, the later one applies. A selection that holds no vertex raises when the entity
        is added.
        """
        _, groups = _get_selector_groups(selectors, self._selector_types)
        return VertexSelection(self, groups)


class MPM(BaseEntity["MPMEntity"]):
    """Options for an entity simulated with the material point method (MPM).

    Parameters
    ----------
    material : gs.materials.MPM.Base
        The material of the entity, one of ``gs.materials.MPM.Elastic``, ``ElastoPlastic``, ``Liquid``, ``Sand``,
        ``Snow`` or ``Muscle``.
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


# The class of entity options each material class goes to, a material taking that of the nearest of its bases
_OPTIONS_CLS_BY_MATERIAL_CLS: dict[type[Material], type[BaseEntity]] = {
    materials.Rigid: Rigid,
    materials.Kinematic: Kinematic,
    materials.FEM.Base: FEM,
    materials.MPM.Base: MPM,
    materials.SPH.Base: SPH,
    materials.PBD.Base: PBD,
    materials.Hybrid: Hybrid,
}
