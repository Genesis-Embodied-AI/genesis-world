"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

from collections.abc import Iterable
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Self, TypeVar, overload

from pydantic import Field, InstanceOf, StrictBool, model_validator

import genesis as gs
from genesis.engine import materials
from genesis.engine.materials.base import Material
from genesis.repr_base import get_public_name

from . import indices
from .blocks import Geom, Setting, Vertex, _FrozenOptions
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

Selector = indices.Link | indices.Geom | indices.Vertex


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
    fields = {"morph": morph, "material": material, "surface": surface, "visualize_contact": visualize_contact}
    return options_cls(**{field: value for field, value in {**fields, "name": name}.items() if value is not None})


class BaseEntity(_FrozenOptions, Generic[EntityT]):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits,
    the selectors of its parts and the blocks of quantities set on them. The fields are fixed at construction, while
    ``select`` records into the options the quantities set on parts of the entity, where the kind of entity has parts.
    The entity holds a frozen copy, whose settings are a tuple recording nothing more, so later records leave the
    entity as it is. A copy of the options (``model_copy()``) records into a list of its own, starting from the settings
    recorded so far.

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
    settings : list[gs.options.Setting] | tuple[gs.options.Setting, ...], optional
        The quantities set on selections of parts of the entity, in the order they were set (see ``select``), a tuple
        once frozen. Default is none.
    """

    # Whether the kind of entity admits several morphs, one per variant of a heterogeneous entity
    _admits_variants: ClassVar[bool] = False
    # Selectors of the parts of the entity, and blocks of the quantities set on them, that the kind of entity admits
    _selector_types: ClassVar[tuple[type[indices.Link | indices.Geom | indices.Vertex], ...]] = ()
    _block_types: ClassVar[tuple[type[Geom | Vertex], ...]] = ()

    # Checked by instance alone, since validation would run the validators of 'Morph' on the sequence of morphs
    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(strict=False)] | None = None
    surface: Surface | None = None
    name: str | None = None
    settings: list[Setting] | tuple[Setting, ...] = Field(default_factory=list)

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
    def _check_settings(self) -> Self:
        if isinstance(self.morph, tuple) and not self._admits_variants:
            gs.raise_exception(
                f"An entity of gs.entities.{type(self).__name__} is created from one morph, got {len(self.morph)}."
            )
        for setting in self.settings:
            self._check_setting(setting)
        return self

    def __copy__(self) -> Self:
        options = super().__copy__()
        # A copy records into a list of its own, frozen or not
        options.__dict__["settings"] = list(self.settings)
        return options

    def _check_setting(self, setting: Setting) -> None:
        for group in setting.selectors:
            if not isinstance(group[0], self._selector_types):
                gs.raise_exception(
                    f"gs.entities.{type(self).__name__} selects no part with gs.{get_public_name(type(group[0]))}."
                )
        if not isinstance(setting.block, self._block_types):
            gs.raise_exception(
                f"gs.entities.{type(self).__name__} sets no gs.options.{type(setting.block).__name__} on its parts."
            )

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant of a heterogeneous entity and a single one otherwise."""
        if self.morph is None:
            return ()
        return self.morph if isinstance(self.morph, tuple) else (self.morph,)

    def _frozen_copy(self, update: dict[str, Any] | None = None) -> Self:
        """Return a copy of these options, with the fields 'update' gives, that records no more settings."""
        return self.model_copy(update={**(update or {}), "settings": tuple(self.settings)})

    def _record(self, settings: tuple[Setting, ...]) -> None:
        """Record settings after those recorded so far, each checked as at construction."""
        if isinstance(self.settings, tuple):
            gs.raise_exception(
                "The options an entity holds are frozen. Select parts on a copy of them ('model_copy()') instead."
            )
        for setting in settings:
            self._check_setting(setting)
        self.settings.extend(settings)
        self.__pydantic_fields_set__.add("settings")


OptionsT = TypeVar("OptionsT", bound=BaseEntity)


class _PartSelection(Generic[OptionsT]):
    """A selection of parts of an entity, recording into its entity options the quantities set on it.

    It holds the entity options and the selectors leading to the parts, and owns no data. Every setter returns the
    selection, so that calls chain on it. ``select`` narrows the selection to the parts that the given selectors hold
    too, at the level of the selection or a finer one, and ``with`` binds the selection to a name for a block.
    """

    def __init__(
        self, options: OptionsT, selectors: tuple[tuple[indices.Link | indices.Geom | indices.Vertex, ...], ...]
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


def _get_selector_type(
    selectors: tuple["Selector", ...], selector_types: tuple[type["Selector"], ...]
) -> type["Selector"]:
    """Return the type of the selectors given to one 'select' call, one among 'selector_types' a selection admits."""
    selector_types_given = {type(selector) for selector in selectors}
    if len(selector_types_given) != 1:
        gs.raise_exception(
            "'select' takes at least one selector, all of one type, and selections of different types nest. Got "
            f"{sorted(f'gs.{get_public_name(selector_type)}' for selector_type in selector_types_given)}."
        )
    (selector_type,) = selector_types_given
    if not issubclass(selector_type, selector_types):
        selector_names = [f"gs.{get_public_name(selector_type)}" for selector_type in selector_types]
        gs.raise_exception(f"This selection selects with {selector_names}, got gs.{get_public_name(selector_type)}.")
    return selector_type


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

    def select(self, *selectors: indices.Geom) -> "GeomSelection[OptionsT]":
        """Return the selection of the geoms of this selection that any of the selectors holds."""
        _get_selector_type(selectors, (indices.Geom,))
        return GeomSelection(self._options, (*self._selectors, selectors))


class LinkSelection(_GeomSetters[OptionsT]):
    """A selection of links of an entity, whose geom setters apply to every geom of the links (see ``select``)."""

    @overload
    def select(self, *selectors: indices.Link) -> "LinkSelection[OptionsT]": ...

    @overload
    def select(self, *selectors: indices.Geom) -> GeomSelection[OptionsT]: ...

    def select(self, *selectors: indices.Link | indices.Geom) -> "LinkSelection[OptionsT] | GeomSelection[OptionsT]":
        """Return the selection of the links, or the geoms, of this selection that any of the selectors holds."""
        if _get_selector_type(selectors, (indices.Link, indices.Geom)) is indices.Geom:
            return GeomSelection(self._options, (*self._selectors, selectors))
        return LinkSelection(self._options, (*self._selectors, selectors))


class VertexSelection(_PartSelection[OptionsT]):
    """A selection of vertices of a deformable entity (see ``gs.entities.FEM.select``)."""

    def select(self, *selectors: indices.Vertex) -> "VertexSelection[OptionsT]":
        """Return the selection of the vertices of this selection that any of the selectors holds."""
        _get_selector_type(selectors, (indices.Vertex,))
        return VertexSelection(self._options, (*self._selectors, selectors))

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
    def select(self, *selectors: indices.Link) -> LinkSelection[Self]: ...

    @overload
    def select(self, *selectors: indices.Geom) -> GeomSelection[Self]: ...

    def select(self, *selectors: indices.Link | indices.Geom) -> LinkSelection[Self] | GeomSelection[Self]:
        """Select the links, or the collision geoms, that any of the selectors holds, to set the quantities of geoms on.

        The selectors of one call are of one type, and a call on a selection keeps the parts both hold, so links and
        geoms select together by nesting: ``options.select(fingers).select(gs.indices.Geom.named("pad*"))``.

        A quantity set on links applies to every collision geom of those links, and a geom ordinal selected on links
        counts the geoms of each link apart (see ``gs.indices.Geom.at``). For each field of each geom, a value
        set on a geom selection overrides a value set on links, which overrides the material. Between two selections of
        one of these levels, the later one applies. A selection that holds no geom of the entity, or of one of its
        heterogeneous variants, raises when the entity is added.
        """
        if _get_selector_type(selectors, self._selector_types) is indices.Geom:
            return GeomSelection(self, (selectors,))
        return LinkSelection(self, (selectors,))


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
        The material of the entity, one of ``gs.materials.FEM.Elastic``, ``Cloth`` or ``Muscle``. A Cloth material
        takes no settings.
    """

    _selector_types = (indices.Vertex,)
    _block_types = (Vertex,)

    material: materials.FEM.Base

    def _check_setting(self, setting: Setting) -> None:
        if isinstance(self.material, materials.FEM.Cloth):
            gs.raise_exception("gs.entities.FEM cannot honour settings of vertices for a Cloth material.")
        super()._check_setting(setting)

    def select(self, *selectors: indices.Vertex) -> VertexSelection[Self]:
        """Select the vertices of the volume mesh that any of the selectors holds, to set quantities on.

        A call on a selection keeps the vertices both hold, an ordinal counting the vertices of that selection. Between
        two selections setting one field of one vertex, the later one applies. A selection that holds no vertex raises
        when the entity is added.
        """
        _get_selector_type(selectors, self._selector_types)
        return VertexSelection(self, (selectors,))


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


# The class of entity options each material class goes to, the one each options class declares for its material. A
# material takes the class of the nearest of its bases (see 'get_entity_options_cls').
_OPTIONS_CLS_BY_MATERIAL_CLS: dict[type[Any] | None, type[BaseEntity]] = {
    options_cls.model_fields["material"].annotation: options_cls
    for options_cls in (Rigid, Kinematic, FEM, MPM, SPH, PBD, Hybrid)
}
