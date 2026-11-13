"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Self, TypeVar

from pydantic import ConfigDict, Field, InstanceOf, StrictBool, model_validator

import genesis as gs
from genesis.engine import materials
from genesis.engine.materials.base import Material
from genesis.typing import FrozenDictType, ValidFloat

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
    The options are immutable: derive a variant with ``model_copy(update=...)``.

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


class _GeomFields(_FrozenOptions):
    rho: ValidFloat | None = None
    friction: Annotated[ValidFloat, Field(ge=0.01, le=5.0)] | None = None
    friction_torsional: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_rolling: Annotated[ValidFloat, Field(ge=0.0)] | None = None


class Geom(_GeomFields):
    """Options of the collision geoms of a rigid entity that a ``gs.GeomIndex`` selects.

    The fields given explicitly override those of the entity material for the selected geoms. Given in the 'geoms' of
    link options, they override the link options as well, while in the 'geoms' of entity options the link options
    override them. The fields left unset keep theirs.

    Parameters
    ----------
    rho : float | None, optional
        The density (kg/m^3). It estimates the mass and the inertia of a link the asset states none for, or of every
        moving link under the 'recompute_inertia' option of the morph. An aligned body is then simulated unaligned.
        The IPC coupler rejects it. If None, the geoms take the density of their link. Default is None.
    friction : float | None, optional
        The sliding friction coefficient of the contacts of the geoms (see ``gs.materials.Rigid.friction``). If None,
        the geoms take that of their link. Default is None.
    friction_torsional : float | None, optional
        The torsional friction coefficient (see ``gs.materials.Rigid.friction_torsional``). If None, the geoms take that
        of their link. Default is None.
    friction_rolling : float | None, optional
        The rolling friction coefficient (see ``gs.materials.Rigid.friction_rolling``). If None, the geoms take that of
        their link. Default is None.
    """


class Link(_GeomFields):
    """Options of the links of a rigid entity that a ``gs.LinkIndex`` selects.

    The density and the friction coefficients given explicitly apply to every collision geom of the selected links,
    overriding those of the entity material and of the geom options of the entity. If None, the links keep those.

    Parameters
    ----------
    geoms : dict[gs.GeomIndex, gs.options.Geom], optional
        The options of collision geoms of each selected link, a ``gs.GeomIndex`` ordinal counting the collision geoms of
        that link alone (see ``gs.GeomIndex.at``). They override the options of the link. Default is no geom options.
    """

    geoms: FrozenDictType[GeomIndex, Geom] = {}


class Rigid(BaseEntity["RigidEntity"]):
    """Options for an entity simulated with rigid-body dynamics.

    Parameters
    ----------
    material : gs.materials.Rigid, optional
        The material of the entity. Defaults to ``gs.materials.Rigid()``.
    visualize_contact : bool, optional
        Whether to draw the contact forces applied to the entity as arrows in the viewer and rendered images. Each
        arrow is drawn every frame, so reserve it for the entities being inspected. Default is False.
    links : dict[gs.LinkIndex, gs.options.Link], optional
        The options of the links each key selects, which override the material of the entity. A link the asset parser
        merges into its parent has no name of its own, so keep it with the 'links_to_keep' option of the morph to
        select it. Where two keys give one field of one link, the first key applies, with a warning. Default is no link
        options.
    geoms : dict[gs.GeomIndex, gs.options.Geom], optional
        The options of the collision geoms each key selects among the collision geoms of the whole entity, by ordinal
        (see ``gs.GeomIndex.at``) or by name in any of its links, raising if it selects no geom. They override the
        material of the entity, and the options of the links override them in turn. Where two keys give one field of
        one geom, the first key applies, with a warning. Default is no geom options.
    """

    _solver_options_cls = RigidOptions

    material: materials.Rigid = Field(default_factory=materials.Rigid)
    visualize_contact: StrictBool = False
    links: FrozenDictType[LinkIndex, Link] = {}
    geoms: FrozenDictType[GeomIndex, Geom] = {}


class Kinematic(BaseEntity["KinematicEntity"]):
    """Options for an entity simulated with kinematics only, with no collision.

    Parameters
    ----------
    material : gs.materials.Kinematic, optional
        The material of the entity. Defaults to ``gs.materials.Kinematic()``.
    """

    _solver_options_cls = KinematicOptions

    material: materials.Kinematic = Field(default_factory=materials.Kinematic)


class Vertex(_FrozenOptions):
    """Options of the vertices of a deformable entity that a ``gs.VertexIndex`` selects.

    Parameters
    ----------
    is_fixed : bool, optional
        Whether the vertices stay at their initial positions, as a hard constraint the solver enforces at every
        substep. It holds an entity in place by a few vertices, such as a soft pad clamped by one face. The vertices are
        vertex constraints from the build on, which ``FEMEntity.remove_vertex_constraints`` releases like any other. The
        IPC coupler does not support it, so a Cloth material rejects it. Default is False.
    """

    is_fixed: StrictBool = False


class FEM(BaseEntity["FEMEntity"]):
    """Options for an entity simulated with the finite element method (FEM).

    Parameters
    ----------
    material : gs.materials.FEM.Base
        The material of the entity, one of ``gs.materials.FEM.Elastic``, ``Cloth`` or ``Muscle``.
    vertices : dict[gs.VertexIndex, gs.options.Vertex], optional
        The options of the vertices each key selects among the vertices of the volume mesh. Where several keys give
        one field of one vertex, the first key applies, with a warning. A Cloth material takes none. Default is no
        vertex options.
    """

    _solver_options_cls = FEMOptions

    material: materials.FEM.Base
    vertices: FrozenDictType[VertexIndex, Vertex] = {}

    @model_validator(mode="after")
    def _check_vertices(self) -> "FEM":
        if isinstance(self.material, materials.FEM.Cloth) and self.vertices:
            gs.raise_exception("gs.entities.FEM cannot honour 'vertices' for a Cloth material.")
        return self


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
