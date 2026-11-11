"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, TypeVar

import numpy as np

from pydantic import BeforeValidator, ConfigDict, Field, InstanceOf, StrictBool, model_validator

import genesis as gs
from genesis.engine import materials
from genesis.engine.materials.base import Material
from genesis.typing import FrozenDictType

from .indices import GeomIndex, LinkIndex, VertexIndex
from .morphs import Mesh, Morph, Primitive
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

# The fields of a rigid material that the options of a link or a geom honour, the others applying to the whole entity
_LINK_MATERIAL_FIELDS = ("rho", "friction", "friction_torsional", "friction_rolling")


def _merge_fields(fields: dict, options: Options, target: str) -> None:
    """Add the fields given explicitly in 'options' to 'fields', raising on a field already given for 'target'.

    A field given as None leaves the value to the scope above, as one left unset does.
    """
    for name, value in options:
        if name not in options.model_fields_set or value is None:
            continue
        if name in fields:
            gs.raise_exception(f"Two selections of equal specificity give '{name}' of {target}.")
        fields[name] = value


def _key_links_by_index(links):
    """Key the link options by ``LinkIndex``, a name or a tuple of names standing for the links of those names."""
    if not isinstance(links, Mapping):
        return links
    links_by_index = {}
    for key, link_options in links.items():
        if isinstance(key, str):
            key = LinkIndex.named(key)
        elif isinstance(key, tuple):
            key = LinkIndex.named(*key)
        links_by_index[key] = link_options
    return links_by_index


def _reject_material_entity_fields(material: "materials.Rigid", target: str) -> None:
    """Raise an error naming the fields of the material of a link or a geom that apply to the whole entity."""
    names = sorted(material.model_fields_set.difference(_LINK_MATERIAL_FIELDS))
    if names:
        gs.raise_exception(
            f"The material of a {target} cannot honour {', '.join(names)}, which apply to the whole entity."
        )


def get_entity_options_cls(material: Material) -> tuple[type["BaseEntity"], ...]:
    """Return the public classes of the entity options admitting a material, in declaration order.

    Each class admits the materials of its declared 'material' field, except those it excludes explicitly.
    """
    classes, pending = [], [BaseEntity]
    while pending:
        options_cls = pending.pop()
        classes.append(options_cls)
        pending.extend(reversed(options_cls.__subclasses__()))
    # Private bases and their parametrizations by the entity class declare a material field too
    return tuple(
        options_cls
        for options_cls in classes
        if not options_cls.__name__.startswith("_")
        and options_cls.__pydantic_generic_metadata__["origin"] is None
        and "material" in options_cls.model_fields
        and options_cls._admits(material)
    )


class _FrozenOptions(Options):
    # Assigning to a field of options skips their validators, so the entity options and the options they nest are
    # immutable, which keeps what their validators checked true and 'entity.options' as authored
    model_config = ConfigDict(frozen=True)


class BaseEntity(_FrozenOptions, Generic[EntityT]):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits.
    The options are immutable: derive a variant with ``model_copy(update=...)``. The scene keeps its own copy of them
    as ``entity.options``, so one instance can construct several entities.

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

    # Validated again wherever an instance is handed over, since 'model_copy(update=...)' derives a variant unvalidated
    model_config = ConfigDict(revalidate_instances="always")

    # The class of the scene options configuring the solver that simulates the entity, or None for an entity simulated
    # by no solver of its own, such as a hybrid entity, whose parts are entities of their own
    _solver_options_cls: ClassVar[type[Options] | None] = None
    # The materials the 'material' field accepts by type that belong to the entity options of another solver
    _excluded_materials: ClassVar[tuple[type[Material], ...]] = ()

    # Checked by instance alone, since validation would run the validators of 'Morph' on the sequence of morphs
    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(strict=False)]
    surface: Surface | None = None
    name: str | None = None

    @classmethod
    def _admits(cls, material: Material) -> bool:
        return isinstance(material, cls.model_fields["material"].annotation) and not isinstance(
            material, cls._excluded_materials
        )

    @model_validator(mode="before")
    @classmethod
    def _check_material_solver(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        material = data.get("material")
        if isinstance(material, Material) and "material" in cls.model_fields and not cls._admits(material):
            options_names = " or ".join(
                f"gs.entities.{options_cls.__name__}" for options_cls in get_entity_options_cls(material)
            )
            gs.raise_exception(
                f"gs.entities.{cls.__name__} cannot take material {type(material).__name__} of another solver. "
                f"Pass it to {options_names or 'the entity options of its solver'} instead."
            )
        return data

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant of a heterogeneous entity and a single one otherwise."""
        return self.morph if isinstance(self.morph, tuple) else (self.morph,)


class Geom(_FrozenOptions):
    """Options of the collision geoms of a rigid entity that a ``gs.GeomIndex`` selects.

    Parameters
    ----------
    material : gs.materials.Rigid | None, optional
        The density and the friction coefficients given explicitly override those of the link and of the entity for
        the selected geoms, while the fields left unset keep theirs. The friction coefficients apply to the contacts
        of the geoms. The density estimates the mass and the inertia of a link the asset states none for, or of every
        moving link under the 'recompute_inertia' option of the morph. The frame of an aligned body stays where the
        densities of the asset put it, so the body is then simulated as unaligned. The IPC coupler rejects the density
        on a link it simulates. The other fields of the material apply to the whole entity, so a geom rejects them.
        If None, the geoms take the material of their link. Default is None.
    """

    material: materials.Rigid | None = None

    @model_validator(mode="after")
    def _check_material(self) -> "Geom":
        if self.material is not None:
            _reject_material_entity_fields(self.material, "geom")
        return self


class Link(_FrozenOptions):
    """Options of the links of a rigid entity that a ``gs.LinkIndex`` or link names select.

    Parameters
    ----------
    material : gs.materials.Rigid | None, optional
        The density and the friction coefficients given explicitly override those of the entity for every collision
        geom of the selected links (see ``gs.options.Geom.material``). If None, the links take the material of the
        entity. Default is None.
    geoms : dict[gs.GeomIndex, gs.options.Geom], optional
        The options of collision geoms of each selected link, a ``gs.GeomIndex`` counting the collision geoms of the
        link as its asset states them. They override the options of the link. Default is no geom options.
    """

    material: materials.Rigid | None = None
    geoms: FrozenDictType[GeomIndex, Geom] = {}

    @model_validator(mode="after")
    def _check_material(self) -> "Link":
        if self.material is not None:
            _reject_material_entity_fields(self.material, "link")
        return self


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
    links : dict[gs.LinkIndex | str | tuple[str, ...], gs.options.Link], optional
        The options of the links each key selects, which override the material of the entity. A name or a tuple of
        names stands for ``gs.LinkIndex.named(...)``. A link the asset parser merges into its parent has no name of
        its own, so keep it with the 'links_to_keep' option of the morph to select it. Two keys selecting one link
        stand at equal specificity whatever the number of links they name, so they cannot give one field. Default is
        no link options.
    geoms : dict[gs.GeomIndex, gs.options.Geom], optional
        The options of the collision geoms each key selects, which override those of the entity and the material of
        the links. They stand at the specificity of the geom options of the links, so the two giving one field of one
        geom raise an error.
        ``gs.GeomIndex.all()`` selects every collision geom of the entity, and ``gs.GeomIndex.named`` the geoms of the
        given names in any of its links, raising if no link has one. A selection by ordinal requires a mesh or a
        primitive morph, whose entity has a single link. Select the geoms of one link of another asset through 'links'
        instead. Default is no geom options.
    """

    _solver_options_cls = RigidOptions

    material: materials.Rigid = Field(default_factory=materials.Rigid)
    visualize_contact: StrictBool = False
    links: Annotated[FrozenDictType[LinkIndex, Link], BeforeValidator(_key_links_by_index)] = {}
    geoms: FrozenDictType[GeomIndex, Geom] = {}

    @model_validator(mode="after")
    def _check_consistency(self) -> "Rigid":
        if any(not geom_index.is_all and geom_index.names is None for geom_index in self.geoms) and not all(
            isinstance(morph, (Mesh, Primitive)) for morph in self.morphs
        ):
            gs.raise_exception(
                "Geom options of an entity select geoms by ordinal only if its morph is a mesh or a primitive. Select "
                "them with 'gs.GeomIndex.all()' or 'gs.GeomIndex.named', or through the options of their link instead."
            )
        return self

    def get_geoms_material(self, link_path: tuple[str, ...], geoms_name: list[str]) -> list[materials.Rigid | None]:
        """Return the material the link and geom options give each collision geom of a link, or None where none.

        'link_path' holds the names from the root of the kinematic tree down to the link (see ``gs.LinkIndex``), and
        'geoms_name' the names of the collision geoms of the link as its asset states them, which a ``gs.GeomIndex``
        counts and matches. A material holds the fields given explicitly alone, those of the geom options overriding
        those of the link options. Two selections of equal specificity giving one field of one geom raise an error.
        A selection by name may match no geom of a given link, the description checking that it matches one of the
        entity.
        """
        link_name = link_path[-1]
        link_fields = {}
        geoms_options = list(self.geoms.items())
        for link_index, link_options in self.links.items():
            if link_index.matches(link_path):
                if link_options.material is not None:
                    _merge_fields(link_fields, link_options.material, f"link '{link_name}'")
                geoms_options.extend(link_options.geoms.items())
        geoms_fields = [{} for _ in geoms_name]
        for geom_index, geom_options in geoms_options:
            if geom_options.material is not None:
                for i_g in geom_index.resolve(geoms_name):
                    _merge_fields(geoms_fields[i_g], geom_options.material, f"geom {i_g} of link '{link_name}'")
        return [
            materials.Rigid(**{**link_fields, **geom_fields}) if link_fields or geom_fields else None
            for geom_fields in geoms_fields
        ]


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
    _excluded_materials = (materials.Rigid,)

    material: materials.Kinematic = Field(default_factory=materials.Kinematic)


class FEMVertexSolverOptions(_FrozenOptions):
    """Options of the finite element method (FEM) solver specific to vertices of an entity.

    Parameters
    ----------
    is_fixed : bool, optional
        Whether the vertices stay at their initial positions, as a hard constraint the solver enforces at every
        substep. It holds an entity in place by a few vertices, such as a soft pad clamped by one face. The IPC
        coupler does not support it, so a Cloth material rejects it. Default is False.
    """

    is_fixed: StrictBool = False


class Vertex(_FrozenOptions):
    """Options of the vertices of a deformable entity that a ``gs.VertexIndex`` selects.

    Parameters
    ----------
    solver_options : FEMVertexSolverOptions | None, optional
        The options of the solver specific to the selected vertices. If None, the vertices take the defaults of
        ``FEMVertexSolverOptions``. Default is None.
    """

    solver_options: FEMVertexSolverOptions | None = None


class FEM(BaseEntity["FEMEntity"]):
    """Options of a deformable entity, which the finite element method (FEM) solver simulates on a volume mesh.

    Use it for a solid whose deformation matters, such as a soft gripper pad or a rubber block. Every element of the
    volume mesh adds to the cost of each substep, so a coarse mesh simulates faster while a fine one resolves the
    deformation more accurately. A Cloth material simulates a shell on the surface mesh instead, under the IPC
    coupler alone. Parameters follow ``gs.entities.BaseEntity``, plus the following.

    Parameters
    ----------
    material : gs.materials.FEM.Elastic | gs.materials.FEM.Cloth
        The material of the entity, a ``gs.materials.FEM.Muscle`` making it a muscle.
    vertices : dict[gs.VertexIndex, gs.options.Vertex], optional
        The options of the vertices each key selects among the vertices of the volume mesh. A Cloth material takes
        none. Default is no vertex options.
    """

    _solver_options_cls = FEMOptions

    material: materials.FEM.Elastic | materials.FEM.Cloth
    vertices: FrozenDictType[VertexIndex, Vertex] = {}

    @model_validator(mode="after")
    def _check_vertices(self) -> "FEM":
        if isinstance(self.material, materials.FEM.Cloth) and self.vertices:
            gs.raise_exception("gs.entities.FEM cannot honour 'vertices' for a Cloth material.")
        return self

    def get_fixed_verts_idx(self, verts_pos: np.ndarray) -> np.ndarray:
        """Return the sorted indices of the vertices the vertex options fix, given their positions in the morph frame.

        Two selections of equal specificity giving 'is_fixed' of one vertex raise an error.
        """
        verts_fields = {}
        for vertex_index, vertex_options in self.vertices.items():
            solver_options = vertex_options.solver_options
            if solver_options is not None and solver_options.model_fields_set:
                for i_v in vertex_index.resolve(verts_pos):
                    _merge_fields(verts_fields.setdefault(i_v, {}), solver_options, f"vertex {i_v}")
        return np.array(
            sorted(i_v for i_v, fields in verts_fields.items() if fields.get("is_fixed", False)), dtype=gs.np_int
        )


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
    material : gs.materials.SPH.Liquid, optional
        The material of the liquid. Defaults to ``gs.materials.SPH.Liquid()``.
    """

    _solver_options_cls = SPHOptions

    material: materials.SPH.Liquid = Field(default_factory=materials.SPH.Liquid)


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
