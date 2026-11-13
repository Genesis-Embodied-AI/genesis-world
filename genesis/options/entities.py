"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

import copy
from collections.abc import Iterable, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Self, TypeVar

from pydantic import ConfigDict, Field, InstanceOf, StrictBool, ValidationInfo, field_validator, model_validator

import genesis as gs
from genesis.engine import materials
from genesis.engine.materials.base import Material
from genesis.typing import ValidFloat

from .indices import GeomIndex, LinkIndex, VertexIndex
from .morphs import Morph
from .options import Options
from .solvers import FEMOptions, KinematicOptions, MPMOptions, PBDOptions, RigidOptions, SPHOptions
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

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Return a copy of these options with the fields 'update' gives, validated as a new instance would be.

        The fields given explicitly stay given, so the copy inherits from its context only what the original left
        unset. 'deep' copies the values held as well.
        """
        fields = {name: getattr(self, name) for name in self.model_fields_set}
        if deep:
            fields = copy.deepcopy(fields)
        return type(self)(**{**fields, **(update or {})})


class BaseEntity(_FrozenOptions, Generic[EntityT]):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits.
    The options are a builder until ``Scene.add_entity`` takes them: assign their fields, or set quantities on parts of
    the entity through ``select`` where the kind of entity has parts. ``Scene.add_entity`` keeps a read-only copy, so
    later edits leave the entity as it is.

    Parameters
    ----------
    morph : Morph | tuple[Morph, ...]
        The geometry or asset of the entity. A sequence of morphs makes the entity heterogeneous: each parallel
        environment simulates one of them (see ``Scene.add_entity``). The options of a particle entity given to
        ``Scene.add_emitter`` leave it None, since the emitter sets it.
    surface : Surface | None, optional
        The appearance of the entity, its visualization mode included. If None, use ``gs.surfaces.Default()``.
    name : str | None, optional
        The name of the entity, unique within the scene. If None, a name is generated from the morph.
    """

    # Assigning a field runs the validators, as constructing the options does
    model_config = ConfigDict(frozen=False, validate_assignment=True)

    # Options class of the solver simulating the entity, None for a hybrid entity, whose parts are entities of their own
    _solver_options_cls: ClassVar[type[Options] | None] = None

    # Checked by instance alone, since validation would run the validators of 'Morph' on the sequence of morphs
    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(strict=False)]
    surface: Surface | None = None
    name: str | None = None

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

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant of a heterogeneous entity and a single one otherwise."""
        return self.morph if isinstance(self.morph, tuple) else (self.morph,)


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


class _EntitySelection:
    """A selection of parts of an entity, recording into its entity options the quantities set on it.

    Every setter returns the selection, so that calls chain on it. ``select`` narrows the selection to the parts that
    the given selectors hold too, and ``with`` binds the selection to a name for a block.
    """

    def __init__(self, options: "BaseEntity", selectors: tuple["Selector | Sequence[Selector]", ...]) -> None:
        if not selectors:
            gs.raise_exception("'select' takes at least one selector.")
        self._options = options
        self._selectors = tuple(
            (selector,) if isinstance(selector, (LinkIndex, GeomIndex, VertexIndex)) else tuple(selector)
            for selector in selectors
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None

    def select(self, *selectors: "Selector | Sequence[Selector]") -> Self:
        """Return the selection of the parts of this selection that every argument holds.

        An argument is a selector, or a list of selectors of one kind holding the parts any of them holds.
        """
        return type(self)(self._options, (*self._selectors, *selectors))

    def set(self, *blocks: "Geom | Vertex") -> Self:
        """Set on the selected parts the fields each block gives explicitly, a later call overriding an earlier one."""
        settings = tuple(Setting(selectors=self._selectors, block=block) for block in blocks)
        self._options.settings = (*self._options.settings, *settings)
        return self


class RigidSelection(_EntitySelection):
    """A selection of links and collision geoms of a rigid entity (see ``Rigid.select``)."""

    def set_rho(self, rho: float) -> Self:
        """Set the density of the selected collision geoms (see ``gs.options.Geom.rho``)."""
        return self.set(Geom(rho=rho))

    def set_friction(self, friction: float) -> Self:
        """Set the sliding friction coefficient of the selected collision geoms (see ``gs.options.Geom.friction``)."""
        return self.set(Geom(friction=friction))

    def set_friction_torsional(self, friction_torsional: float) -> Self:
        """Set the torsional friction coefficient of the selected collision geoms."""
        return self.set(Geom(friction_torsional=friction_torsional))

    def set_friction_rolling(self, friction_rolling: float) -> Self:
        """Set the rolling friction coefficient of the selected collision geoms."""
        return self.set(Geom(friction_rolling=friction_rolling))


class FEMSelection(_EntitySelection):
    """A selection of vertices of a deformable entity (see ``FEM.select``)."""

    def set_is_fixed(self, is_fixed: bool) -> Self:
        """Set whether the selected vertices stay at their initial positions (see ``gs.options.Vertex.is_fixed``)."""
        return self.set(Vertex(is_fixed=is_fixed))


class Rigid(BaseEntity["RigidEntity"]):
    """Options for an entity simulated with rigid-body dynamics.

    Parameters
    ----------
    material : gs.materials.Rigid, optional
        The material of the entity. Defaults to ``gs.materials.Rigid()``.
    visualize_contact : bool, optional
        Whether to draw the contact forces applied to the entity as arrows in the viewer and rendered images. Each
        arrow is drawn every frame, so reserve it for the entities being inspected. Default is False.
    settings : tuple[gs.options.Setting, ...], optional
        The quantities set on selections of links and collision geoms, in the order they were set (see ``select``).
        Default is none.
    """

    _solver_options_cls = RigidOptions

    material: materials.Rigid = Field(default_factory=materials.Rigid)
    visualize_contact: StrictBool = False
    settings: tuple[Setting, ...] = ()

    # A field validator, since a model validator failing on an assignment leaves the assigned value in place
    @field_validator("settings")
    @classmethod
    def _check_settings(cls, settings: tuple[Setting, ...]) -> tuple[Setting, ...]:
        for setting in settings:
            if not all(
                isinstance(selector, (LinkIndex, GeomIndex)) for group in setting.selectors for selector in group
            ):
                gs.raise_exception("gs.entities.Rigid selects parts with gs.LinkIndex and gs.GeomIndex alone.")
            if not isinstance(setting.block, Geom):
                gs.raise_exception("gs.entities.Rigid sets gs.options.Geom alone on its parts.")
        return settings

    def select(self, *selectors: "LinkIndex | GeomIndex | Sequence[LinkIndex | GeomIndex]") -> RigidSelection:
        """Select the links and the collision geoms that every argument holds, to set quantities of their geoms on.

        An argument is a selector, or a list of selectors of one kind holding the parts any of them holds.

        A quantity set on links applies to every collision geom of those links. For each field of each geom, a value
        set on a selection holding a ``gs.GeomIndex`` overrides a value set on links alone, which overrides the
        material. Between two selections of one of these levels, the later one applies. A selection that holds no
        geom of the entity raises when the entity is added. A link the asset parser merges into its parent has no name
        of its own, so keep it with the 'links_to_keep' option of the morph to select it.
        """
        return RigidSelection(self, selectors)


class Kinematic(BaseEntity["KinematicEntity"]):
    """Options for an entity simulated with kinematics only, with no collision.

    Parameters
    ----------
    material : gs.materials.Kinematic, optional
        The material of the entity. Defaults to ``gs.materials.Kinematic()``.
    """

    _solver_options_cls = KinematicOptions

    material: materials.Kinematic = Field(default_factory=materials.Kinematic)


class FEM(BaseEntity["FEMEntity"]):
    """Options for an entity simulated with the finite element method (FEM).

    Parameters
    ----------
    material : gs.materials.FEM.Base
        The material of the entity, one of ``gs.materials.FEM.Elastic``, ``Cloth`` or ``Muscle``.
    settings : tuple[gs.options.Setting, ...], optional
        The quantities set on selections of vertices of the volume mesh, in the order they were set (see ``select``). A
        Cloth material takes none. Default is none.
    """

    _solver_options_cls = FEMOptions

    material: materials.FEM.Base
    settings: tuple[Setting, ...] = ()

    # Field validators, since a model validator failing on an assignment leaves the assigned value in place. Each field
    # checks the other one as it stands, so that assigning either keeps a Cloth material free of settings.
    @field_validator("material", "settings")
    @classmethod
    def _check_cloth_settings(cls, value: Any, info: ValidationInfo) -> Any:
        material = value if info.field_name == "material" else info.data.get("material")
        settings = value if info.field_name == "settings" else info.data.get("settings")
        if isinstance(material, materials.FEM.Cloth) and settings:
            gs.raise_exception("gs.entities.FEM cannot honour settings of vertices for a Cloth material.")
        return value

    @field_validator("settings")
    @classmethod
    def _check_settings(cls, settings: tuple[Setting, ...]) -> tuple[Setting, ...]:
        for setting in settings:
            if not all(isinstance(selector, VertexIndex) for group in setting.selectors for selector in group):
                gs.raise_exception("gs.entities.FEM selects parts with gs.VertexIndex alone.")
            if not isinstance(setting.block, Vertex):
                gs.raise_exception("gs.entities.FEM sets gs.options.Vertex alone on its parts.")
        return settings

    def select(self, *selectors: "VertexIndex | Sequence[VertexIndex]") -> FEMSelection:
        """Select the vertices of the volume mesh that every argument holds, to set quantities on.

        An argument is a selector, or a list of selectors holding the vertices any of them holds.

        Between two selections setting one field of one vertex, the later one applies. A selection that holds no vertex
        raises when the entity is added.
        """
        return FEMSelection(self, selectors)


class _EmittableParticles(BaseEntity[EntityT]):
    """Options of the particle entities an emitter can hold, whose morph ``Scene.add_emitter`` sets."""

    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(strict=False)] | None = None


class MPM(_EmittableParticles["MPMEntity"]):
    """Options for an entity simulated with the material point method (MPM).

    Parameters
    ----------
    material : gs.materials.MPM.Base
        The material of the entity, one of ``gs.materials.MPM.Elastic``, ``ElastoPlastic``, ``Liquid``, ``Sand``,
        ``Snow`` or ``Muscle``.
    """

    _solver_options_cls = MPMOptions

    material: materials.MPM.Base


class SPH(_EmittableParticles["SPHEntity"]):
    """Options for an entity simulated with smoothed particle hydrodynamics (SPH).

    Parameters
    ----------
    material : gs.materials.SPH.Base, optional
        The material of the liquid. Defaults to ``gs.materials.SPH.Liquid()``.
    """

    _solver_options_cls = SPHOptions

    material: materials.SPH.Base = Field(default_factory=materials.SPH.Liquid)


class PBD(_EmittableParticles["PBDBaseEntity"]):
    """Options for an entity simulated with position-based dynamics (PBD).

    Parameters
    ----------
    material : gs.materials.PBD.Base
        The material of the entity, one of ``gs.materials.PBD.Cloth``, ``Elastic``, ``Liquid`` or ``Particle``.
    """

    _solver_options_cls = PBDOptions

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
