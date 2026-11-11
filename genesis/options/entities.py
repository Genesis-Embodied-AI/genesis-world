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
    # Assigning to a field of options skips their validators, so the entity options and the options they nest are
    # immutable, which keeps what their validators checked true
    model_config = ConfigDict(frozen=True)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Return a copy of these options with the fields 'update' gives, validated as a new instance would be.

        The fields given explicitly stay given, so the copy inherits from its context only what the original left
        unset. 'deep' copies the values held as well.
        """
        fields = {name: getattr(self, name) for name in self.model_fields_set}
        if deep:
            fields = copy.deepcopy(fields)
        return type(self)(**fields, **(update or {}))


class BaseEntity(_FrozenOptions, Generic[EntityT]):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits.
    The options are immutable: derive a variant with ``model_copy(update=...)``. The entity is built from the morph
    and the surface they hold, whose unset defaults it fills in, as when they are passed to ``Scene.add_entity``
    directly.

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

    # The class of the scene options configuring the solver that simulates the entity, or None for an entity simulated
    # by no solver of its own, such as a hybrid entity, whose parts are entities of their own
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

    The fields given explicitly override those of the link and of the entity material for the selected geoms, while the
    fields left unset keep theirs.

    Parameters
    ----------
    rho : float | None, optional
        The density (kg/m^3). It estimates the mass and the inertia of a link the asset states none for, or of every
        moving link under the 'recompute_inertia' option of the morph. The frame of an aligned body stays where the
        densities of the asset put it, so the body is then simulated as unaligned. The IPC coupler rejects it on a link
        it simulates. If None, the geoms take the density of their link. Default is None.
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
    overriding those of the entity material. If None, the links take those of the entity material. Parameters follow
    ``gs.options.Geom``, plus the following.

    Parameters
    ----------
    geoms : dict[gs.GeomIndex, gs.options.Geom], optional
        The options of collision geoms of each selected link, a ``gs.GeomIndex`` counting the collision geoms of the
        link as its asset states them. They override the options of the link. Default is no geom options.
    """

    geoms: FrozenDictType[GeomIndex, Geom] = {}


class Rigid(BaseEntity["RigidEntity"]):
    """Options of a rigid entity, which the rigid solver simulates with contacts, joints and constraints.

    Use it for every entity whose motion follows from physics. Its links collide, so every geom adds to the cost of
    collision detection and constraint solving. Parameters follow ``gs.entities.BaseEntity``, plus the following.

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
        select it. Two keys selecting one link stand at equal specificity whatever the number of links they name, so
        they cannot give one field. Default is no link options.
    geoms : dict[gs.GeomIndex, gs.options.Geom], optional
        The options of the collision geoms each key selects, which override those of the entity and the material of
        the links. They stand at the specificity of the geom options of the links, so the two giving one field of one
        geom raise an error.
        Each key is ``gs.GeomIndex.all()``, selecting every collision geom of the entity, or ``gs.GeomIndex.named``,
        selecting the geoms of the given names in any of its links and raising if no link has one. Select geoms by
        ordinal through the options of their link in 'links'. Default is no geom options.
    """

    _solver_options_cls = RigidOptions

    material: materials.Rigid = Field(default_factory=materials.Rigid)
    visualize_contact: StrictBool = False
    links: FrozenDictType[LinkIndex, Link] = {}
    geoms: FrozenDictType[GeomIndex, Geom] = {}

    @model_validator(mode="after")
    def _check_consistency(self) -> "Rigid":
        if any(not geom_index.is_all and geom_index.names is None for geom_index in self.geoms):
            gs.raise_exception(
                "Geom options of an entity select geoms with 'gs.GeomIndex.all()' or 'gs.GeomIndex.named'. Select geoms "
                "by ordinal through the options of their link."
            )
        return self


class Kinematic(BaseEntity["KinematicEntity"]):
    """Options of a kinematic entity, which follows the poses it is given and takes no part in physics.

    Use it for a ghost or a reference motion: forward kinematics is all it costs, while it collides with nothing and
    nothing pushes it. Parameters follow ``gs.entities.BaseEntity``, plus the following.

    Parameters
    ----------
    material : gs.materials.Kinematic, optional
        The material of the entity, which a rigid material cannot stand for. Defaults to ``gs.materials.Kinematic()``.
    """

    _solver_options_cls = KinematicOptions

    material: materials.Kinematic = Field(default_factory=materials.Kinematic)


class Vertex(_FrozenOptions):
    """Options of the vertices of a deformable entity that a ``gs.VertexIndex`` selects.

    Parameters
    ----------
    is_fixed : bool, optional
        Whether the vertices stay at their initial positions, as a hard constraint the solver enforces at every
        substep. It holds an entity in place by a few vertices, such as a soft pad clamped by one face. The IPC
        coupler does not support it, so a Cloth material rejects it. Default is False.
    """

    is_fixed: StrictBool = False


class FEM(BaseEntity["FEMEntity"]):
    """Options of a deformable entity, which the finite element method (FEM) solver simulates on a volume mesh.

    Use it for a solid whose deformation matters, such as a soft gripper pad or a rubber block. Every element of the
    volume mesh adds to the cost of each substep, so a coarse mesh simulates faster while a fine one resolves the
    deformation more accurately. A Cloth material simulates a shell on the surface mesh instead, under the IPC
    coupler alone. Parameters follow ``gs.entities.BaseEntity``, plus the following.

    Parameters
    ----------
    material : gs.materials.FEM.Base
        The material of the entity, one of ``gs.materials.FEM.Elastic``, ``Cloth`` or ``Muscle``, which makes it a
        muscle.
    vertices : dict[gs.VertexIndex, gs.options.Vertex], optional
        The options of the vertices each key selects among the vertices of the volume mesh. A Cloth material takes
        none. Default is no vertex options.
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
    """Options of a particle entity, which the material point method (MPM) solver simulates on a background grid.

    Use it for a continuum that deforms without bound, tears or flows, such as sand, snow, dough or a liquid. Its
    accuracy and cost both follow the particle size and the grid resolution of the solver. Parameters follow
    ``gs.entities.BaseEntity``, plus the following.

    Parameters
    ----------
    material : gs.materials.MPM.Base
        The material of the entity, one of ``gs.materials.MPM.Elastic``, ``ElastoPlastic``, ``Liquid``, ``Sand``,
        ``Snow`` or ``Muscle``.
    """

    _solver_options_cls = MPMOptions

    material: materials.MPM.Base


class SPH(_EmittableParticles["SPHEntity"]):
    """Options of a liquid entity, which the smoothed particle hydrodynamics (SPH) solver simulates as particles.

    Use it for a liquid whose free surface and splashes matter. Each particle interacts with its neighbors, so the cost
    grows with the particle count that the particle size of the solver sets. Parameters follow
    ``gs.entities.BaseEntity``, plus the following.

    Parameters
    ----------
    material : gs.materials.SPH.Base, optional
        The material of the liquid. Defaults to ``gs.materials.SPH.Liquid()``.
    """

    _solver_options_cls = SPHOptions

    material: materials.SPH.Base = Field(default_factory=materials.SPH.Liquid)


class PBD(_EmittableParticles["PBDBaseEntity"]):
    """Options of an entity the position-based dynamics (PBD) solver simulates, whose material decides its kind.

    Use it where speed and stability matter more than physically accurate stiffness, pressure and viscosity, which the
    finite element method (FEM), the material point method (MPM) and smoothed particle hydrodynamics (SPH) offer at a
    higher cost. A Cloth material makes a sheet, such as a towel or a flag, remeshed to the particle size of the
    solver. An Elastic material makes a soft solid on a volume mesh. A Liquid material makes a liquid of incompressible
    particles, each interacting with its neighbors so that the cost grows with the particle count. A Particle material
    makes free particles. Parameters follow ``gs.entities.BaseEntity``, plus the following.

    Parameters
    ----------
    material : gs.materials.PBD.Base
        The material of the entity, one of ``gs.materials.PBD.Cloth``, ``Elastic``, ``Liquid`` or ``Particle``.
    """

    _solver_options_cls = PBDOptions

    material: materials.PBD.Base


class Hybrid(BaseEntity["HybridEntity"]):
    """Options of a hybrid entity, a soft skin simulated by the material point method around a rigid skeleton.

    Use it for a soft robot actuated through its joints. The scene adds the rigid skeleton and the soft skin as two
    entities of their own, so the entity costs as much as both of them together. The morph is a URDF, whose links the
    skin is grown around, or a mesh, whose skeleton is extracted from the skin. Parameters follow
    ``gs.entities.BaseEntity``, plus the following.

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
