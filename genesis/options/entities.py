"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

import platform
import re
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Callable, ClassVar, Generic, Literal, NamedTuple, TypeVar

import numpy as np

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, InstanceOf, StrictBool, model_validator

import genesis as gs
from genesis.typing import FrozenDictType, NonNegativeFloat, PositiveFloat, PositiveInt, StrictInt, ValidFloat

from .indices import GeomIndex, LinkIndex, VertexIndex, VGeomIndex
from .materials import BaseMaterial, Cloth, Elastic, ElastoPlastic, Liquid, Material, Muscle, Sand, Snow
from .morphs import URDF, Mesh, Morph, Primitive
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
        PBD2DEntity,
        PBD3DEntity,
        PBDFreeParticleEntity,
        PBDParticleEntity,
        RigidEntity,
        SPHEntity,
    )


DEFAULT_SAMPLER = "pbs" if (sys.platform == "linux" and platform.machine() == "x86_64") else "random"

ParticleSamplerType = Literal["pbs", "random", "regular"]

CoupType = Literal["two_way_soft_constraint", "external_articulation", "ipc_only"]


def _validate_mpm_sampler(sampler: str) -> str:
    if re.fullmatch(r"pbs(-\d+)?|random|regular", sampler) is None:
        raise ValueError("Input should be 'pbs', 'pbs-<sdf resolution>', 'random', or 'regular'")
    return sampler


def _reject_fields(
    options: "EntityOptions", sub_options: Options, field_names: tuple[str, ...], reason: str = ""
) -> None:
    """Raise an error naming the fields of a material or of solver options that these entity options cannot honour.

    A field counts as given when it is set explicitly to a value other than None. 'reason' completes the message.
    """
    names = sorted(
        name
        for name, value in dict(sub_options).items()
        if name in field_names and name in sub_options.model_fields_set and value is not None
    )
    if names:
        gs.raise_exception(
            f"{type(options).__name__} cannot honour {', '.join(names)} of {type(sub_options).__name__}{reason}."
        )


EntityT = TypeVar("EntityT", bound="Entity")


class EntityOptions(Options, Generic[EntityT]):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits
    along with the options of that solver specific to the entity ('solver_options'). The options are immutable: derive
    a variant with ``model_copy(update=...)``. The scene keeps its own copy of them as ``entity.options``, so one
    instance can construct several entities.

    Parameters
    ----------
    morph : Morph | tuple[Morph, ...]
        The geometry or asset of the entity. A sequence of morphs makes the entity heterogeneous: each parallel
        environment simulates one of them (see ``Scene.add_entity``).
    surface : Surface | None, optional
        The appearance of the entity, its visualization mode included. If None, use ``gs.surfaces.Default()``.
    name : str | None, optional
        The name of the entity, unique within the scene. If None, a name is generated from the morph.
    """

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    # The class of the scene options configuring the solver that simulates the entity, or None for an entity simulated
    # by no solver of its own, such as a hybrid entity, whose parts are entities of their own
    _solver_options_cls: ClassVar[type[Options] | None] = None

    # Checked by instance alone, since validation would run the validators of 'Morph' on the sequence of morphs
    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(strict=False)]
    surface: Surface | None = None
    name: str | None = None


class RigidEntitySolverOptions(Options):
    """Options of the rigid solver specific to one rigid entity: its coupling with other solvers and its collision grid.

    Parameters
    ----------
    needs_coup : bool, optional
        Whether the links of the entity interact with the entities of other solvers, unless the solver options of a
        link override it (see ``RigidLinkSolverOptions``). Disabling it spares the cost of the coupling for an entity
        that never touches them. Default is True.
    coup_friction : float, optional
        The friction coefficient of the contacts with the particles of other solvers. Default is 0.1.
    coup_softness : float, optional
        The distance over which the entity pushes the particles of other solvers away, in m. A larger value holds the
        particles off more smoothly, at the cost of a visible gap. Default is 0.002.
    coup_restitution : float, optional
        The restitution coefficient of the contacts with the particles of other solvers, between 0 and 1. A non-zero
        value bounces particles off, at the cost of numerical stability. Default is 0.0.
    coup_type : str | None, optional
        How the IPC coupler simulates the entity, which requires a coupled link (see 'needs_coup').
        'two_way_soft_constraint' couples each link with a soft constraint, and alone couples some links of the entity
        without the others. 'external_articulation' couples the joint positions, for articulated entities. 'ipc_only'
        lets IPC simulate the entity alone, gravity included, and is restricted to non-articulated entities. If None,
        'external_articulation' for a fixed-base articulated entity, 'two_way_soft_constraint' for a floating-base one
        and 'ipc_only' for a non-articulated one, unless some links of the entity are coupled alone, which takes
        'two_way_soft_constraint'. Default is None.
    enable_coup_collision : bool, optional
        Whether the links of the entity collide in the IPC coupler. Disabling it keeps the coupling forces while
        dropping the contacts, whereas ``needs_coup=False`` leaves the entity out of the coupler entirely. Default is
        True.
    contact_resistance : float | None, optional
        The contact stiffness of the entity in the IPC coupler. A stiffer contact lets less interpenetration through,
        at the cost of smaller time steps. If None, ``IPCCouplerOptions.contact_resistance``. Default is None.
    sdf_cell_size : float, optional
        The target cell size of the signed distance field sampled around each collision geometry, in m. A finer grid
        resolves the contacts of non-convex geometry better, at the cost of memory and build time. Default is 0.005.
    sdf_min_res : int, optional
        The minimum number of cells of the signed distance field along each axis, at least 16. Default is 32.
    sdf_max_res : int, optional
        The maximum number of cells of the signed distance field along each axis, at least 'sdf_min_res'. A primitive
        geometry caps it at 32, which resolves its shape exactly. Default is 128.
    """

    needs_coup: StrictBool = True
    coup_friction: NonNegativeFloat = 0.1
    coup_softness: NonNegativeFloat = 0.002
    coup_restitution: Annotated[ValidFloat, Field(ge=0.0, le=1.0)] = 0.0
    coup_type: CoupType | None = None
    enable_coup_collision: StrictBool = True
    contact_resistance: PositiveFloat | None = None
    sdf_cell_size: PositiveFloat = 0.005
    sdf_min_res: Annotated[StrictInt, Field(ge=16)] = 32
    sdf_max_res: Annotated[StrictInt, Field(ge=16)] = 128

    @model_validator(mode="after")
    def _check_consistency(self) -> "RigidEntitySolverOptions":
        if self.sdf_min_res > self.sdf_max_res:
            gs.raise_exception("'sdf_min_res' must be smaller than or equal to 'sdf_max_res'.")
        if self.coup_restitution > 0.0:
            gs.logger.warning("Non-zero 'coup_restitution' could lead to instability. Use with caution.")
        return self


class RigidLinkSolverOptions(Options):
    """Options of the rigid solver specific to one link: its coupling with other solvers.

    A field left to None takes the value of the entity (see ``RigidEntitySolverOptions``).

    Parameters
    ----------
    needs_coup : bool | None, optional
        Whether the link interacts with the entities of other solvers. Coupling some links of an entity alone, by
        disabling it on the others or by enabling it on these with ``needs_coup=False`` for the entity, takes the
        'two_way_soft_constraint' coupling under the IPC coupler, which a 'coup_type' left to None resolves to.
        Default is None.
    enable_coup_collision : bool | None, optional
        Whether the link collides in the IPC coupler. Disabling it keeps the coupling forces of the link while dropping
        its contacts, and enabling it on some links with ``enable_coup_collision=False`` for the entity limits the
        contacts to these links. Default is None.
    """

    model_config = ConfigDict(frozen=True)

    needs_coup: StrictBool | None = None
    enable_coup_collision: StrictBool | None = None


class GeomOptions(Options):
    """Options of the collision geoms of a rigid entity that a ``gs.GeomIndex`` selects.

    Parameters
    ----------
    material : gs.materials.Material | None, optional
        The material fields given explicitly override those of the link and of the entity for the selected geoms,
        while the fields left unset keep theirs. The friction coefficients apply to the contacts of the geoms. The
        density estimates the mass and the inertia of a link the asset states none for, or of every moving link under
        the 'recompute_inertia' option of the morph. The frame of an aligned body stays where the densities of the
        asset put it, so the body is then simulated as unaligned. The IPC coupler rejects the density on a link it
        simulates. If None, the geoms take the material of their link. Default is None.
    """

    model_config = ConfigDict(frozen=True)

    material: Material | None = None


class VGeomOptions(Options):
    """Options of the visual geoms of a rigid or kinematic entity that a ``gs.VGeomIndex`` selects.

    Parameters
    ----------
    surface : gs.surfaces.Surface | None, optional
        The appearance of the selected visual geoms, which replaces that of their link and of the entity. The textures
        it leaves unset keep those the asset and the surface of the entity give the geoms. If None, the geoms take the
        surface of their link. Default is None.
    """

    model_config = ConfigDict(frozen=True)

    surface: Surface | None = None


class LinkOptions(Options):
    """Options of the links of a rigid entity that a ``gs.LinkIndex`` or link names select.

    Parameters
    ----------
    material : gs.materials.Material | None, optional
        The material fields given explicitly override those of the entity for every collision geom of the selected
        links (see ``GeomOptions.material``). If None, the links take the material of the entity. Default is None.
    geoms : dict[gs.GeomIndex, GeomOptions], optional
        The options of collision geoms of each selected link, a ``gs.GeomIndex`` counting the collision geoms of the
        link as its asset states them. They override the options of the link. Default is no geom options.
    solver_options : RigidLinkSolverOptions | None, optional
        The options of the rigid solver specific to the selected links. If None, the links take those of the entity.
        Default is None.
    use_visual_raycasting : bool | None, optional
        Whether to expose the visual mesh of the selected links to visual raycasting, as the option of the same name
        of the entity does for all of them. If None, the links take the value of the entity. Default is None.
    surface : gs.surfaces.Surface | None, optional
        The appearance of every visual geom of the selected links, which replaces that of the entity (see
        ``VGeomOptions.surface``). If None, the links take the surface of the entity. Default is None.
    vgeoms : dict[gs.VGeomIndex, VGeomOptions], optional
        The options of visual geoms of each selected link, a ``gs.VGeomIndex`` counting the visual geoms of the link as
        its asset splits them. They override the options of the link. Default is no visual geom options.
    """

    model_config = ConfigDict(frozen=True)

    material: Material | None = None
    geoms: FrozenDictType[GeomIndex, GeomOptions] = {}
    solver_options: RigidLinkSolverOptions | None = None
    use_visual_raycasting: StrictBool | None = None
    surface: Surface | None = None
    vgeoms: FrozenDictType[VGeomIndex, VGeomOptions] = {}


def _merge_fields(fields: dict, options: Options, target: str) -> None:
    """Add the fields given explicitly in 'options' to 'fields', raising on a field already given for 'target'.

    A field given as None leaves the value to the scope above, as one left unset does.
    """
    for name, value in options.model_dump(exclude_unset=True, exclude_none=True).items():
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


class _ArticulatedEntityOptions(EntityOptions[EntityT]):
    """Options shared by the entities made of links connected by joints, rigid and kinematic alike.

    See ``KinematicEntityOptions`` for the parameters they add to ``EntityOptions``.
    """

    use_visual_raycasting: StrictBool = False
    links: Annotated[FrozenDictType[LinkIndex, LinkOptions], BeforeValidator(_key_links_by_index)] = {}
    vgeoms: FrozenDictType[VGeomIndex, VGeomOptions] = {}

    @model_validator(mode="after")
    def _check_vgeoms(self) -> "_ArticulatedEntityOptions":
        morphs = self.morph if isinstance(self.morph, tuple) else (self.morph,)
        if any(
            not vgeom_index.is_all and vgeom_index.names is None and vgeom_index.material_names is None
            for vgeom_index in self.vgeoms
        ) and not all(isinstance(morph, (Mesh, Primitive)) for morph in morphs):
            gs.raise_exception(
                "Visual geom options of an entity select visual geoms by ordinal only if its morph is a mesh or a "
                "primitive. Select them with 'gs.VGeomIndex.all()', 'named' or 'with_material', or through the options "
                "of their link instead."
            )
        return self

    def get_vgeoms_surface(
        self, link_path: tuple[str, ...], vgeoms_name: list[str], vgeoms_material_name: list[str]
    ) -> list[Surface | None]:
        """Return the surface the link and visual geom options give each visual geom of a link, or None where none.

        'link_path' holds the names from the root of the kinematic tree down to the link (see ``gs.LinkIndex``), while
        'vgeoms_name' and 'vgeoms_material_name' hold the names and the asset material names of the visual geoms of the
        link as its asset splits them, which a ``gs.VGeomIndex`` counts and matches. The surface of the visual geom
        options overrides that of the link options. Two selections of equal specificity giving a surface to one visual
        geom raise an error.
        """
        link_name = link_path[-1]
        link_surfaces = []
        vgeoms_options = list(self.vgeoms.items())
        for link_index, link_options in self.links.items():
            if link_index.matches(link_path):
                if link_options.surface is not None:
                    link_surfaces.append(link_options.surface)
                vgeoms_options.extend(link_options.vgeoms.items())
        if len(link_surfaces) > 1:
            gs.raise_exception(f"Two selections of equal specificity give 'surface' of link '{link_name}'.")
        vgeoms_surface = [link_surfaces[0] if link_surfaces else None for _ in vgeoms_name]
        is_vgeom_given = [False for _ in vgeoms_name]
        for vgeom_index, vgeom_options in vgeoms_options:
            if vgeom_options.surface is not None:
                for i_vg in vgeom_index.resolve(vgeoms_name, vgeoms_material_name):
                    if is_vgeom_given[i_vg]:
                        gs.raise_exception(
                            f"Two selections of equal specificity give 'surface' of visual geom {i_vg} of link "
                            f"'{link_name}'."
                        )
                    vgeoms_surface[i_vg] = vgeom_options.surface
                    is_vgeom_given[i_vg] = True
        return vgeoms_surface

    def get_link_use_visual_raycasting(self, link_path: tuple[str, ...]) -> bool:
        """Return whether a link exposes its visual mesh to visual raycasting.

        'link_path' holds the names from the root of the kinematic tree down to the link (see ``gs.LinkIndex``).
        """
        values = [
            link_options.use_visual_raycasting
            for link_index, link_options in self.links.items()
            if link_index.matches(link_path) and link_options.use_visual_raycasting is not None
        ]
        if len(values) > 1:
            gs.raise_exception(
                f"Two selections of equal specificity give 'use_visual_raycasting' of link '{link_path[-1]}'."
            )
        return values[0] if values else self.use_visual_raycasting


class RigidEntityOptions(_ArticulatedEntityOptions["RigidEntity"]):
    """Options of a rigid entity, which the rigid solver simulates with contacts, joints and constraints.

    Use it for every entity whose motion follows from physics. Its links collide, so every geom adds to the cost of
    collision detection and constraint solving. Parameters follow ``KinematicEntityOptions``, plus the following.

    Parameters
    ----------
    material : gs.materials.Material, optional
        The physical properties of the entity. Defaults to ``gs.materials.Material()``.
    solver_options : RigidEntitySolverOptions, optional
        The options of the rigid solver specific to the entity. Defaults to ``RigidEntitySolverOptions()``.
    visualize_contact : bool, optional
        Whether to draw the contact forces applied to the entity as arrows in the viewer and rendered images. Each
        arrow is drawn every frame, so reserve it for the entities being inspected. Default is False.
    gravity_compensation : float, optional
        The fraction of gravity cancelled on the entity, 1.0 letting it float. Default is 0.0. An entity simulated by
        IPC alone (``coup_type='ipc_only'``) falls under the gravity of IPC instead, so it requires 0.0.
    geoms : dict[gs.GeomIndex, GeomOptions], optional
        The options of the collision geoms each key selects, which override those of the entity and the material of
        the links. They stand at the specificity of the geom options of the links, so the two giving one field of one
        geom raise an error.
        ``gs.GeomIndex.all()`` selects every collision geom of the entity, and ``gs.GeomIndex.named`` the geoms of the
        given names in any of its links, raising if no link has one. A selection by ordinal requires a mesh or a
        primitive morph, whose entity has a single link. Select the geoms of one link of another asset through 'links'
        instead. Default is no geom options.
    """

    _solver_options_cls = RigidOptions

    material: Material = Field(default_factory=Material)
    solver_options: RigidEntitySolverOptions = Field(default_factory=RigidEntitySolverOptions)
    visualize_contact: StrictBool = False
    gravity_compensation: ValidFloat = 0.0
    geoms: FrozenDictType[GeomIndex, GeomOptions] = {}

    @model_validator(mode="after")
    def _check_consistency(self) -> "RigidEntityOptions":
        materials = [self.material]
        geoms_options = list(self.geoms.values())
        for link_options in self.links.values():
            materials.append(link_options.material)
            geoms_options.extend(link_options.geoms.values())
        materials.extend(geom_options.material for geom_options in geoms_options)
        for material in materials:
            friction = None if material is None else material.friction
            if friction is not None and not 0.01 <= friction <= 5.0:
                gs.raise_exception(f"The friction of a rigid entity must lie between 0.01 and 5, got {friction}.")
        morphs = self.morph if isinstance(self.morph, tuple) else (self.morph,)
        if any(not geom_index.is_all and geom_index.names is None for geom_index in self.geoms) and not all(
            isinstance(morph, (Mesh, Primitive)) for morph in morphs
        ):
            gs.raise_exception(
                "Geom options of an entity select geoms by ordinal only if its morph is a mesh or a primitive. Select "
                "them with 'gs.GeomIndex.all()' or 'gs.GeomIndex.named', or through the options of their link instead."
            )
        solver_options = self.solver_options
        if solver_options.coup_type == "ipc_only" and abs(self.gravity_compensation) > gs.EPS:
            gs.raise_exception("'gravity_compensation' must be 0 with coup_type='ipc_only', since IPC applies gravity.")
        links_solver_options = [
            link_options.solver_options
            for link_options in self.links.values()
            if link_options.solver_options is not None
        ]
        links_needs_coup = [
            link_solver_options.needs_coup
            for link_solver_options in links_solver_options
            if link_solver_options.needs_coup is not None
        ]
        if solver_options.coup_type is not None and not (solver_options.needs_coup or any(links_needs_coup)):
            gs.raise_exception(f"'coup_type' requires a coupled link, got coup_type={solver_options.coup_type!r}.")
        # A link coupled differently from its entity filters the links of the coupling
        is_link_filtered = any(needs_coup != solver_options.needs_coup for needs_coup in links_needs_coup)
        if is_link_filtered and solver_options.coup_type not in (None, "two_way_soft_constraint"):
            gs.raise_exception(
                "Coupling links differently from their entity requires the 'two_way_soft_constraint' coupling, got "
                f"coup_type={solver_options.coup_type!r}."
            )
        return self

    def has_link_density(self, link_path: tuple[str, ...]) -> bool:
        """Whether the link and geom options give a density to a link or to one of its geoms.

        'link_path' holds the names from the root of the kinematic tree down to the link (see ``gs.LinkIndex``).
        """
        geoms_options = list(self.geoms.values())
        for link_index, link_options in self.links.items():
            if link_index.matches(link_path):
                if link_options.material is not None and link_options.material.density is not None:
                    return True
                geoms_options.extend(link_options.geoms.values())
        return any(
            geom_options.material is not None and geom_options.material.density is not None
            for geom_options in geoms_options
        )

    def get_link_solver_options(self, link_path: tuple[str, ...]) -> RigidLinkSolverOptions:
        """Return the solver options of a link, the fields its link options leave unset inherited.

        'link_path' holds the names from the root of the kinematic tree down to the link (see ``gs.LinkIndex``).
        """
        fields = {}
        for link_index, link_options in self.links.items():
            if link_index.matches(link_path) and link_options.solver_options is not None:
                _merge_fields(fields, link_options.solver_options, f"link '{link_path[-1]}'")
        return RigidLinkSolverOptions(
            needs_coup=fields.get("needs_coup", self.solver_options.needs_coup),
            enable_coup_collision=fields.get("enable_coup_collision", self.solver_options.enable_coup_collision),
        )

    def get_geoms_material(self, link_path: tuple[str, ...], geoms_name: list[str]) -> list[Material | None]:
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
            Material(**{**link_fields, **geom_fields}) if link_fields or geom_fields else None
            for geom_fields in geoms_fields
        ]


class KinematicEntityOptions(_ArticulatedEntityOptions["KinematicEntity"]):
    """Options of a kinematic entity, which follows the poses it is given and takes no part in physics.

    Use it for a ghost or a reference motion: forward kinematics is all it costs, while it collides with nothing and
    nothing pushes it, so it has no material. Parameters follow ``EntityOptions``, plus the following.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        Whether to expose the visual mesh of the entity to every consumer casting rays against visual geometry:
        raycaster sensors, and viewer plugins built on RaycasterViewerPlugin with use_visual_geom set. Each consumer
        then scans the visual triangles of the entity again as it moves, so reserve it for the entities meant to be
        sensed or picked. The options of a link override it. Default is False.
    links : dict[gs.LinkIndex | str | tuple[str, ...], LinkOptions], optional
        The options of the links each key selects, which override those of the entity. A name or a tuple of names
        stands for ``gs.LinkIndex.named(...)``. A link the asset parser merges into its parent has no name of its own,
        so keep it with the 'links_to_keep' option of the morph to select it. Two keys selecting one link stand at equal
        specificity whatever the number of links they name, so they cannot give one field. A kinematic entity takes
        the 'use_visual_raycasting', 'surface' and 'vgeoms' of its link options alone. Default is no link options.
    vgeoms : dict[gs.VGeomIndex, VGeomOptions], optional
        The options of the visual geoms each key selects, which override the surface of the entity and of their link.
        They stand at the specificity of the visual geom options of the links, so the two giving a surface to one
        visual geom raise an error. ``gs.VGeomIndex.all()`` selects every visual geom of the entity, while
        ``gs.VGeomIndex.named`` and ``gs.VGeomIndex.with_material`` select those of the given names or asset materials
        in any of its links, raising if no link has one. A selection by ordinal requires a mesh or a primitive morph,
        whose entity has a single link. Default is no visual geom options.
    """

    _solver_options_cls = KinematicOptions

    @model_validator(mode="after")
    def _check_links(self) -> "KinematicEntityOptions":
        for link_options in self.links.values():
            _reject_fields(self, link_options, ("material", "geoms", "solver_options"), ", which takes no physics")
        return self


class FEMEntitySolverOptions(Options):
    """Options of the finite element method (FEM) solver specific to one entity.

    Parameters
    ----------
    model : str, optional
        The constitutive model of an Elastic or Muscle material. 'linear' is the cheapest and suits small deformations.
        'stable_neohookean' stays accurate under large deformations, at a higher cost per step, and only the explicit
        solver supports it. 'linear_corotated' handles large rotations under small strain, and only the implicit solver
        supports it. The IPC coupler always simulates a stable Neo-Hookean solid. Default is 'linear'.
    hydroelastic_modulus : float, optional
        The pressure at the deepest point of the entity under hydroelastic contact, in Pa. A larger value makes the
        contact stiffer. Default is 1e7.
    contact_resistance : float | None, optional
        The contact stiffness of the entity in the IPC coupler. A stiffer contact lets less interpenetration through,
        at the cost of smaller time steps. If None, ``IPCCouplerOptions.contact_resistance``. Default is None.
    n_groups : int, optional
        The number of muscle groups of a Muscle material, each actuated on its own. Default is 1.
    """

    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "linear"
    hydroelastic_modulus: PositiveFloat = 1e7
    contact_resistance: PositiveFloat | None = None
    n_groups: PositiveInt = 1


class FEMVertexSolverOptions(Options):
    """Options of the finite element method (FEM) solver specific to vertices of an entity.

    Parameters
    ----------
    is_fixed : bool, optional
        Whether the vertices stay at their initial positions, as a hard constraint the solver enforces at every
        substep. It holds an entity in place by a few vertices, such as a soft pad clamped by one face. The implicit
        solver enables ``FEMOptions.enable_vertex_constraints`` for it, and the IPC coupler does not support it, so a
        Cloth material rejects it. Default is False.
    """

    model_config = ConfigDict(frozen=True)

    is_fixed: StrictBool = False


class VertexOptions(Options):
    """Options of the vertices of a deformable entity that a ``gs.VertexIndex`` selects.

    Parameters
    ----------
    solver_options : FEMVertexSolverOptions | None, optional
        The options of the solver specific to the selected vertices. If None, the vertices take those of the entity.
        Default is None.
    """

    model_config = ConfigDict(frozen=True)

    solver_options: FEMVertexSolverOptions | None = None


class FEMMaterialParams(NamedTuple):
    """The parameters of a material as the FEM solver simulates it, the defaults of the solver filled in."""

    youngs_modulus: float
    poisson_ratio: float
    density: float
    friction: float


class FEMEntityOptions(EntityOptions["FEMEntity"]):
    """Options of a deformable entity, which the finite element method (FEM) solver simulates on a volume mesh.

    Use it for a solid whose deformation matters, such as a soft gripper pad or a rubber block. Every element of the
    volume mesh adds to the cost of each substep, so a coarse mesh simulates faster while a fine one resolves the
    deformation more accurately. A Cloth material simulates a shell on the surface mesh instead, under the IPC
    coupler alone. Parameters follow ``EntityOptions``, plus the following.

    Parameters
    ----------
    material : gs.materials.Elastic | gs.materials.Cloth | gs.materials.Muscle
        The physical law of the entity and its parameters.
    solver_options : FEMEntitySolverOptions, optional
        The options of the FEM solver specific to the entity. Defaults to ``FEMEntitySolverOptions()``.
    vertices : dict[gs.VertexIndex, VertexOptions], optional
        The options of the vertices each key selects among the vertices of the volume mesh. A Cloth material takes
        none. Default is no vertex options.
    """

    _solver_options_cls = FEMOptions

    material: Elastic | Cloth | Muscle
    solver_options: FEMEntitySolverOptions = Field(default_factory=FEMEntitySolverOptions)
    vertices: FrozenDictType[VertexIndex, VertexOptions] = {}

    @model_validator(mode="after")
    def _check_solver_options(self) -> "FEMEntityOptions":
        if isinstance(self.material, Cloth):
            _reject_fields(self, self.solver_options, ("model",), " for a Cloth material")
            if self.vertices:
                gs.raise_exception("FEMEntityOptions cannot honour 'vertices' for a Cloth material.")
        if not isinstance(self.material, Muscle):
            _reject_fields(self, self.solver_options, ("n_groups",), " for a material other than Muscle")
        return self

    def get_fixed_verts_idx(self, verts_pos: np.ndarray) -> np.ndarray:
        """Return the sorted indices of the vertices the vertex options fix, given their positions in the morph frame.

        Two selections of equal specificity giving 'is_fixed' of one vertex raise an error.
        """
        verts_is_fixed = {}
        for vertex_index, vertex_options in self.vertices.items():
            solver_options = vertex_options.solver_options
            if solver_options is not None and "is_fixed" in solver_options.model_fields_set:
                for i_v in vertex_index.resolve(verts_pos):
                    if i_v in verts_is_fixed:
                        gs.raise_exception(f"Two selections of equal specificity give 'is_fixed' of vertex {i_v}.")
                    verts_is_fixed[i_v] = solver_options.is_fixed
        return np.array(sorted(i_v for i_v, is_fixed in verts_is_fixed.items() if is_fixed), dtype=gs.np_int)

    @property
    def material_params(self) -> FEMMaterialParams:
        """The parameters of the material, the defaults of the FEM solver filled in.

        The density of Cloth defaults to 200 kg/m^3 and the friction coefficient to 0.1.
        """
        material = self.material
        return FEMMaterialParams(
            youngs_modulus=material.youngs_modulus,
            poisson_ratio=material.poisson_ratio,
            density=200.0 if material.density is None else material.density,
            friction=0.1 if material.friction is None else material.friction,
        )


class MPMEntitySolverOptions(Options):
    """Options of the material point method (MPM) solver specific to one entity.

    Parameters
    ----------
    sampler : str | None, optional
        How the morph is filled with particles. 'regular' lays them on a grid, which is cheap but shows aliasing,
        'random' scatters them uniformly, and 'pbs' spreads them evenly by Poisson disk sampling, the most uniform but
        also the slowest to build, and supported on Linux x86 alone. 'pbs-<N>' samples with a signed distance field of
        resolution N. If None, 'random' for Sand and Snow, and otherwise 'pbs' on Linux x86 and 'random' elsewhere.
        Default is None.
    model : str | None, optional
        The stress model of an Elastic or Muscle material. 'corotation' handles large rotations accurately, while
        'neohooken' stays robust under large compression and skips a decomposition per particle, which makes it
        faster. If None, 'corotation' for Elastic and 'neohooken' for Muscle. Default is None.
    n_groups : int, optional
        The number of muscle groups of a Muscle material, each actuated on its own. Default is 1.
    """

    sampler: Annotated[str, AfterValidator(_validate_mpm_sampler)] | None = None
    model: Literal["corotation", "neohooken"] | None = None
    n_groups: PositiveInt = 1


class MPMMaterialParams(NamedTuple):
    """The parameters of a material as the MPM solver simulates it, the defaults of the solver filled in.

    The Young's modulus and the Poisson ratio give the Lame parameters of the elastic law. The stress model is None
    for a material other than Elastic and Muscle.
    """

    youngs_modulus: float
    poisson_ratio: float
    model: str | None
    sampler: str


class MPMEntityOptions(EntityOptions["MPMEntity"]):
    """Options of a particle entity, which the material point method (MPM) solver simulates on a background grid.

    Use it for a continuum that deforms without bound, tears or flows, such as sand, snow, dough or a liquid. Its
    accuracy and cost both follow the particle size and the grid resolution of the solver. Parameters follow
    ``EntityOptions``, plus the following.

    Parameters
    ----------
    material : gs.materials.BaseMaterial
        The physical law of the entity and its parameters. One of Elastic, ElastoPlastic, Liquid, Sand, Snow or Muscle.
    solver_options : MPMEntitySolverOptions, optional
        The options of the MPM solver specific to the entity. Defaults to ``MPMEntitySolverOptions()``.
    """

    _solver_options_cls = MPMOptions

    material: Elastic | ElastoPlastic | Liquid | Sand | Snow | Muscle
    solver_options: MPMEntitySolverOptions = Field(default_factory=MPMEntitySolverOptions)

    @model_validator(mode="after")
    def _check_material(self) -> "MPMEntityOptions":
        _reject_fields(self, self.material, ("friction", "surface_tension"))
        if not isinstance(self.material, (Elastic, Muscle)):
            _reject_fields(self, self.solver_options, ("model",), " for a material other than Elastic and Muscle")
        if not isinstance(self.material, Muscle):
            _reject_fields(self, self.solver_options, ("n_groups",), " for a material other than Muscle")
        return self

    @property
    def material_params(self) -> MPMMaterialParams:
        """The parameters of the material, the defaults of the MPM solver filled in.

        A Liquid takes the Lame parameters of a Young's modulus of 1e6 Pa and a Poisson ratio of 0.2 (see
        'MPMLiquidModel'). The stress model defaults to 'corotation' for Elastic and 'neohooken' for Muscle, and the
        sampler to 'random' for Sand and Snow and to 'pbs' on Linux x86 and 'random' elsewhere otherwise.
        """
        material, solver_options = self.material, self.solver_options
        if isinstance(material, Liquid):
            youngs_modulus, poisson_ratio = 1e6, 0.2
        else:
            youngs_modulus, poisson_ratio = material.youngs_modulus, material.poisson_ratio
        model = solver_options.model
        if model is None and isinstance(material, (Elastic, Muscle)):
            model = "corotation" if isinstance(material, Elastic) else "neohooken"
        sampler = solver_options.sampler
        if sampler is None:
            sampler = "random" if isinstance(material, (Sand, Snow)) else DEFAULT_SAMPLER
        return MPMMaterialParams(youngs_modulus, poisson_ratio, model, sampler)


class SPHEntitySolverOptions(Options):
    """Options of the smoothed particle hydrodynamics (SPH) solver specific to one entity.

    Parameters
    ----------
    sampler : str, optional
        How the morph is filled with particles. 'regular' lays them on a grid, which keeps the density uniform and the
        simulation stable at rest, 'random' scatters them uniformly, and 'pbs' spreads them evenly by Poisson disk
        sampling, supported on Linux x86 alone. Default is 'regular'.
    exponent : float, optional
        The exponent of the equation of state, relating pressure to density. A larger exponent resists compression
        more sharply, at the cost of a stiffer and less stable simulation. Default is 7.0.
    """

    sampler: ParticleSamplerType = "regular"
    exponent: PositiveFloat = 7.0


class SPHMaterialParams(NamedTuple):
    """The parameters of a Liquid as the SPH solver simulates it, the defaults of the solver filled in.

    The stiffness is the bulk modulus divided by the exponent of the equation of state.
    """

    stiffness: float
    viscosity: float
    surface_tension: float


class SPHEntityOptions(EntityOptions["SPHEntity"]):
    """Options of a liquid entity, which the smoothed particle hydrodynamics (SPH) solver simulates as particles.

    Use it for a liquid whose free surface and splashes matter. Each particle interacts with its neighbors, so the cost
    grows with the particle count that the particle size of the solver sets. Parameters follow ``EntityOptions``, plus
    the following.

    Parameters
    ----------
    material : gs.materials.Liquid, optional
        The physical properties of the liquid. Defaults to ``gs.materials.Liquid()``.
    solver_options : SPHEntitySolverOptions, optional
        The options of the SPH solver specific to the entity. Defaults to ``SPHEntitySolverOptions()``.
    """

    _solver_options_cls = SPHOptions

    material: Liquid = Field(default_factory=Liquid)
    solver_options: SPHEntitySolverOptions = Field(default_factory=SPHEntitySolverOptions)

    @property
    def material_params(self) -> SPHMaterialParams:
        """The parameters of the liquid, the defaults of the SPH solver filled in.

        The stiffness defaults to 5e4 Pa, the viscosity to 0.005 Pa*s and the surface tension to 0.01.
        """
        material = self.material
        return SPHMaterialParams(
            stiffness=5e4 if material.bulk_modulus is None else material.bulk_modulus / self.solver_options.exponent,
            viscosity=0.005 if material.viscosity is None else material.viscosity,
            surface_tension=0.01 if material.surface_tension is None else material.surface_tension,
        )


class PBDMaterialParams(NamedTuple):
    """The parameters of a cloth or an elastic material as the PBD solver simulates it, the defaults filled in.

    The density is a mass per unit area for a cloth, in kg/m^2, and a mass per unit volume otherwise, in kg/m^3. The
    friction coefficient holds a particle that sticks, and the kinetic friction coefficient slows one that slides.
    """

    density: float
    friction: float
    kinetic_friction: float


class PBDClothEntitySolverOptions(Options):
    """Options of the position-based dynamics (PBD) solver specific to one cloth entity.

    A compliance is the inverse stiffness of a constraint: 0 makes it rigid, and a larger value lets it give way. A
    relaxation scales each correction of a constraint: a smaller value weakens it but converges more smoothly.

    Parameters
    ----------
    stretch_compliance : float, optional
        The compliance of the edges against stretching, in m/N. Default is 1e-7.
    bending_compliance : float, optional
        The compliance of the sheet against bending, in rad/N. Default is 1e-5.
    stretch_relaxation : float, optional
        The relaxation of the stretching constraints. Default is 0.3.
    bending_relaxation : float, optional
        The relaxation of the bending constraints. Default is 0.1.
    air_resistance : float, optional
        The damping of the particle velocities by air drag. A larger value slows the fall and the swing of the sheet.
        Default is 1e-3.
    kinetic_friction : float | None, optional
        The friction coefficient of a particle sliding on a contact, the friction of the material holding a particle
        that sticks. A smaller value lets the sheet slide on more easily once it moves. If None, the friction of the
        material. Default is None.
    """

    stretch_compliance: NonNegativeFloat = 1e-7
    bending_compliance: NonNegativeFloat = 1e-5
    stretch_relaxation: ValidFloat = 0.3
    bending_relaxation: ValidFloat = 0.1
    air_resistance: NonNegativeFloat = 1e-3
    kinetic_friction: NonNegativeFloat | None = None


class PBDClothEntityOptions(EntityOptions["PBD2DEntity"]):
    """Options of a cloth entity, which the position-based dynamics (PBD) solver simulates as a mesh of particles.

    Use it for a thin sheet that stretches and bends, such as a towel or a flag. The surface mesh is remeshed to the
    particle size of the solver, so a smaller particle size gives finer folds at a higher cost. Parameters follow
    ``EntityOptions``, plus the following.

    Parameters
    ----------
    material : gs.materials.Cloth, optional
        The physical properties of the sheet. Defaults to ``gs.materials.Cloth()``.
    solver_options : PBDClothEntitySolverOptions, optional
        The options of the PBD solver specific to the entity, among which its stiffness. Defaults to
        ``PBDClothEntitySolverOptions()``.
    """

    _solver_options_cls = PBDOptions

    material: Cloth = Field(default_factory=Cloth)
    solver_options: PBDClothEntitySolverOptions = Field(default_factory=PBDClothEntitySolverOptions)

    @model_validator(mode="after")
    def _check_material(self) -> "PBDClothEntityOptions":
        _reject_fields(self, self.material, ("youngs_modulus", "poisson_ratio", "bending_stiffness"))
        # The mass per unit area defaults to a value of its own when no density is given, which leaves the thickness out
        if self.material.density is None:
            _reject_fields(self, self.material, ("thickness",), " without a density")
        return self

    @property
    def material_params(self) -> PBDMaterialParams:
        """The parameters of the sheet, the defaults of the PBD solver filled in.

        The mass per unit area is ``density * thickness``, and defaults to 4.0 kg/m^2. The friction coefficient
        defaults to 0.15, and the kinetic one to the friction coefficient.
        """
        material = self.material
        friction = 0.15 if material.friction is None else material.friction
        kinetic_friction = self.solver_options.kinetic_friction
        return PBDMaterialParams(
            density=4.0 if material.density is None else material.density * material.thickness,
            friction=friction,
            kinetic_friction=friction if kinetic_friction is None else kinetic_friction,
        )


class PBDElasticEntitySolverOptions(Options):
    """Options of the position-based dynamics (PBD) solver specific to one elastic entity.

    See ``PBDClothEntitySolverOptions`` for the meaning of a compliance and a relaxation.

    Parameters
    ----------
    stretch_compliance : float, optional
        The compliance of the edges against stretching, in m/N. Default is 0.0.
    bending_compliance : float, optional
        The compliance against bending, in rad/N. Default is 0.0.
    volume_compliance : float, optional
        The compliance of the elements against a change of volume, in m^3/N. Default is 0.0.
    stretch_relaxation : float, optional
        The relaxation of the stretching constraints. Default is 0.1.
    bending_relaxation : float, optional
        The relaxation of the bending constraints. Default is 0.1.
    volume_relaxation : float, optional
        The relaxation of the volume constraints. Default is 0.1.
    kinetic_friction : float | None, optional
        See ``PBDClothEntitySolverOptions``. If None, the friction of the material. Default is None.
    """

    stretch_compliance: NonNegativeFloat = 0.0
    bending_compliance: NonNegativeFloat = 0.0
    volume_compliance: NonNegativeFloat = 0.0
    stretch_relaxation: ValidFloat = 0.1
    bending_relaxation: ValidFloat = 0.1
    volume_relaxation: ValidFloat = 0.1
    kinetic_friction: NonNegativeFloat | None = None


class PBDElasticEntityOptions(EntityOptions["PBD3DEntity"]):
    """Options of an elastic solid, which the position-based dynamics (PBD) solver simulates on a volume mesh.

    Use it for a soft solid where speed and stability matter more than physically accurate stiffness, which the
    finite element method (FEM) solver offers at a higher cost (see ``FEMEntityOptions``). Parameters follow
    ``EntityOptions``, plus the following.

    Parameters
    ----------
    material : gs.materials.Elastic, optional
        The physical properties of the solid. Defaults to ``gs.materials.Elastic()``.
    solver_options : PBDElasticEntitySolverOptions, optional
        The options of the PBD solver specific to the entity, among which its stiffness. Defaults to
        ``PBDElasticEntitySolverOptions()``.
    """

    _solver_options_cls = PBDOptions

    material: Elastic = Field(default_factory=Elastic)
    solver_options: PBDElasticEntitySolverOptions = Field(default_factory=PBDElasticEntitySolverOptions)

    @model_validator(mode="after")
    def _check_material(self) -> "PBDElasticEntityOptions":
        _reject_fields(self, self.material, ("youngs_modulus", "poisson_ratio"))
        return self

    @property
    def material_params(self) -> PBDMaterialParams:
        """The parameters of the solid, the friction coefficient defaulting to 0.15 and the kinetic one to it."""
        friction = 0.15 if self.material.friction is None else self.material.friction
        kinetic_friction = self.solver_options.kinetic_friction
        return PBDMaterialParams(
            density=self.material.density,
            friction=friction,
            kinetic_friction=friction if kinetic_friction is None else kinetic_friction,
        )


class PBDLiquidEntitySolverOptions(Options):
    """Options of the position-based dynamics (PBD) solver specific to one liquid entity.

    Parameters
    ----------
    sampler : str, optional
        How the morph is filled with particles. 'regular' lays them on a grid, 'random' scatters them uniformly, and
        'pbs' spreads them evenly by Poisson disk sampling, supported on Linux x86 alone. Defaults to 'pbs' on Linux
        x86 and 'random' elsewhere.
    density_relaxation : float, optional
        The relaxation of the constraint keeping the density at rest. A larger value keeps the volume more constant
        but converges less smoothly. Default is 0.2.
    viscosity_relaxation : float, optional
        The relaxation of the viscosity, which smooths the velocities of neighboring particles. A larger value damps
        the flow more, as honey does. Default is 0.01.
    """

    sampler: ParticleSamplerType = DEFAULT_SAMPLER
    density_relaxation: ValidFloat = 0.2
    viscosity_relaxation: ValidFloat = 0.01


class PBDLiquidEntityOptions(EntityOptions["PBDParticleEntity"]):
    """Options of a liquid entity, which the position-based dynamics (PBD) solver simulates as incompressible particles.

    Use it for a liquid where speed and stability matter more than accurate pressure and viscosity. Each particle
    interacts with its neighbors, so the cost grows with the particle count. Parameters follow ``EntityOptions``, plus
    the following.

    Parameters
    ----------
    material : gs.materials.Liquid, optional
        The physical properties of the liquid. Defaults to ``gs.materials.Liquid()``.
    solver_options : PBDLiquidEntitySolverOptions, optional
        The options of the PBD solver specific to the entity, among which its viscosity. Defaults to
        ``PBDLiquidEntitySolverOptions()``.
    """

    _solver_options_cls = PBDOptions

    material: Liquid = Field(default_factory=Liquid)
    solver_options: PBDLiquidEntitySolverOptions = Field(default_factory=PBDLiquidEntitySolverOptions)

    @model_validator(mode="after")
    def _check_material(self) -> "PBDLiquidEntityOptions":
        _reject_fields(self, self.material, ("bulk_modulus", "viscosity", "surface_tension"))
        return self


class PBDParticleEntitySolverOptions(Options):
    """Options of the position-based dynamics (PBD) solver specific to one set of free particles.

    Parameters
    ----------
    sampler : str, optional
        See ``PBDLiquidEntitySolverOptions``. Defaults to 'pbs' on Linux x86 and 'random' elsewhere.
    """

    sampler: ParticleSamplerType = DEFAULT_SAMPLER


class PBDParticleEntityOptions(EntityOptions["PBDFreeParticleEntity"]):
    """Options of a set of free particles, which the position-based dynamics (PBD) solver moves under external forces.

    Use it for particle effects and animations: the particles ignore each other, which makes them the cheapest
    particle entity, at the cost of any volume or cohesion. Parameters follow ``EntityOptions``, plus the following.

    Parameters
    ----------
    material : gs.materials.Liquid, optional
        The physical properties of the particles, of which only the density applies. Defaults to
        ``gs.materials.Liquid()``.
    solver_options : PBDParticleEntitySolverOptions, optional
        The options of the PBD solver specific to the entity. Defaults to ``PBDParticleEntitySolverOptions()``.
    """

    _solver_options_cls = PBDOptions

    material: Liquid = Field(default_factory=Liquid)
    solver_options: PBDParticleEntitySolverOptions = Field(default_factory=PBDParticleEntitySolverOptions)

    @model_validator(mode="after")
    def _check_material(self) -> "PBDParticleEntityOptions":
        _reject_fields(self, self.material, ("bulk_modulus", "viscosity", "surface_tension"))
        return self


class HybridEntitySolverOptions(Options):
    """Options of the solvers simulating the two parts of a hybrid entity.

    Parameters
    ----------
    rigid : RigidEntitySolverOptions, optional
        The options of the rigid solver specific to the skeleton. The 'needs_coup' of a URDF skeleton follows
        'use_default_coupling' of the hybrid entity, and must equal it if given. Defaults to
        ``RigidEntitySolverOptions()``.
    soft : MPMEntitySolverOptions, optional
        The options of the material point method (MPM) solver specific to the skin. Its 'n_groups' can be given under
        'use_default_coupling' alone, since the skin otherwise has one muscle group per link it follows. Defaults to
        ``MPMEntitySolverOptions()``.
    """

    rigid: RigidEntitySolverOptions = Field(default_factory=RigidEntitySolverOptions)
    soft: MPMEntitySolverOptions = Field(default_factory=MPMEntitySolverOptions)


class HybridEntityOptions(EntityOptions["HybridEntity"]):
    """Options of a hybrid entity, a soft skin simulated by the material point method around a rigid skeleton.

    Use it for a soft robot actuated through its joints. The scene adds the rigid skeleton and the soft skin as two
    entities of their own, so the entity costs as much as both of them together. The morph is a URDF, whose links the
    skin is grown around, or a mesh, whose skeleton is extracted from the skin. Parameters follow ``EntityOptions``,
    plus the following.

    Parameters
    ----------
    material_rigid : gs.materials.Material, optional
        The physical properties of the skeleton. Defaults to ``gs.materials.Material()``.
    material_soft : gs.materials.BaseMaterial
        The physical law of the skin and its parameters. One of Elastic, ElastoPlastic, Liquid, Sand, Snow or Muscle,
        a Muscle material letting each link actuate its own group.
    solver_options : HybridEntitySolverOptions, optional
        The options of the solvers simulating both parts. Defaults to ``HybridEntitySolverOptions()``.
    gravity_compensation : float, optional
        The fraction of gravity cancelled on the skeleton, 1.0 letting it float. Default is 0.0.
    use_default_coupling : bool, optional
        Whether the skin and the skeleton interact through the coupler of the scene alone. Otherwise, the skin follows
        the links it is associated with, which holds it in place far more firmly. Default is False.
    damping : float, optional
        The damping of the velocity the skin takes from the skeleton, in 1/s. Default is 0.0.
    thickness : float, optional
        The thickness of the skin grown around the links of a URDF skeleton, in m. Default is 0.05.
    soft_dv_coef : float, optional
        The fraction of the velocity of the skeleton transferred to the skin at each substep. A larger value makes the
        skin follow more tightly, at the cost of stability. Default is 0.01.
    func_instantiate_rigid_from_soft : callable | None, optional
        The function building the skeleton from the mesh of the skin. If None, the skeleton is extracted from the
        mesh. Default is None.
    func_instantiate_soft_from_rigid : callable | None, optional
        The function building the skin from the links of the skeleton. If None, the skin is grown around them by
        'thickness'. Default is None.
    func_instantiate_rigid_soft_association : callable | None, optional
        The function associating the particles of the skin with the links of the skeleton. If None, each particle
        follows its closest link. Default is None.
    """

    material_rigid: Material = Field(default_factory=Material)
    material_soft: Elastic | ElastoPlastic | Liquid | Sand | Snow | Muscle
    solver_options: HybridEntitySolverOptions = Field(default_factory=HybridEntitySolverOptions)
    gravity_compensation: ValidFloat = 0.0
    use_default_coupling: StrictBool = False
    damping: NonNegativeFloat = 0.0
    thickness: ValidFloat = 0.05
    soft_dv_coef: ValidFloat = 0.01
    func_instantiate_rigid_from_soft: Callable | None = Field(default=None, exclude=True)
    func_instantiate_soft_from_rigid: Callable | None = Field(default=None, exclude=True)
    func_instantiate_rigid_soft_association: Callable | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_parts(self) -> "HybridEntityOptions":
        rigid, soft = self.solver_options.rigid, self.solver_options.soft
        if isinstance(self.morph, URDF) and "needs_coup" in rigid.model_fields_set:
            if rigid.needs_coup != self.use_default_coupling:
                gs.raise_exception(
                    "The URDF skeleton of a hybrid entity interacts with its skin through the coupler exactly when "
                    f"'use_default_coupling' is set, got needs_coup={rigid.needs_coup} and "
                    f"use_default_coupling={self.use_default_coupling}."
                )
        if not self.use_default_coupling:
            _reject_fields(self, soft, ("n_groups",), ", which is one per link the skin follows")
        # Each part is an entity of its own, whose options reject what its solver cannot honour
        for part_options_cls, material, solver_options in (
            (RigidEntityOptions, self.material_rigid, self.rigid_solver_options),
            (MPMEntityOptions, self.material_soft, soft),
        ):
            try:
                part_options_cls(morph=self.morph, material=material, solver_options=solver_options)
            except gs.GenesisException as e:
                gs.raise_exception_from(f"Invalid part of a hybrid entity: {e}", e)
        return self

    @property
    def rigid_solver_options(self) -> RigidEntitySolverOptions:
        """The solver options the skeleton is built with.

        A URDF skeleton interacts with its skin through the coupler exactly when 'use_default_coupling' is set.
        """
        rigid = self.solver_options.rigid
        if isinstance(self.morph, URDF):
            rigid = rigid.model_copy(update={"needs_coup": self.use_default_coupling})
        return rigid


def get_entity_options_cls(material: BaseMaterial) -> tuple[type[EntityOptions], ...]:
    """Return the classes of the entity options admitting a material, in declaration order.

    Each class admits the materials of its declared 'material' field. A kinematic or hybrid entity has no such field.
    """
    classes, pending = [], [EntityOptions]
    while pending:
        options_cls = pending.pop()
        classes.append(options_cls)
        pending.extend(reversed(options_cls.__subclasses__()))
    return tuple(
        options_cls
        for options_cls in classes
        if "material" in options_cls.model_fields
        and isinstance(material, options_cls.model_fields["material"].annotation)
    )
