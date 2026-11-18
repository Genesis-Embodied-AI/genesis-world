"""Typed specs an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

from collections.abc import Iterable
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Self, TypeVar, overload

from pydantic import Field, InstanceOf, PrivateAttr, StrictBool, model_validator

import genesis as gs
from genesis.constants import STRUCTURE
from genesis.engine import materials
from genesis.engine.materials.base import Material
from genesis.repr_base import get_public_name

from . import indices
from .morphs import Morph
from .options import Options
from .patches import Geom, Patch, Vertex, VGeom, _FrozenOptions
from .surfaces import DynamicVertices, Surface

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


def get_entity_spec_cls(material: Material) -> type["Spec"]:
    """Return the entity spec class of the solver simulating a material, raising if no solver simulates it.

    A material goes to the class of the nearest of its bases that the solvers of Genesis simulate, so a subclass a user
    declares goes where the material it derives from goes.
    """
    for material_cls in type(material).__mro__:
        spec_cls = _SPEC_CLS_BY_MATERIAL_CLS.get(material_cls)
        if spec_cls is not None:
            return spec_cls
    gs.raise_exception(f"No entity spec class admits material {type(material).__name__}.")


def get_entity_spec(
    morph: Morph | Iterable[Morph] | None = None,
    material: Material | None = None,
    surface: Surface | None = None,
    visualize_contact: bool | None = None,
    vis_mode: str | None = None,
    name: str | None = None,
) -> "Spec":
    """Return the entity spec that the individual arguments of ``Scene.add_entity`` stand for.

    The material decides the class of the spec (see ``get_entity_spec_cls``), ``gs.entities.Rigid`` if None. Each
    argument given fills the field of the spec it stands for, 'vis_mode' filling that of a copy of the surface, and
    the spec refuses an argument its class has no field for.
    """
    spec_cls = Rigid if material is None else get_entity_spec_cls(material)
    # Heterogeneous morphs may come as any iterable of morphs
    if isinstance(morph, Iterable) and not isinstance(morph, Morph):
        morph = tuple(morph)
    if vis_mode is not None:
        surface = (gs.surfaces.Default() if surface is None else surface).model_copy(update={"vis_mode": vis_mode})
    # Only rigid entities have contacts to draw, and 'False' asks for nothing, so every kind of entity accepts it
    if "visualize_contact" not in spec_cls.model_fields:
        if visualize_contact:
            gs.raise_exception(
                f"'visualize_contact' only applies to rigid entities, not to gs.entities.{spec_cls.__name__}."
            )
        visualize_contact = None
    fields = {"morph": morph, "material": material, "surface": surface, "visualize_contact": visualize_contact}
    return spec_cls(**{field: value for field, value in {**fields, "name": name}.items() if value is not None})


class Spec(_FrozenOptions, Generic[EntityT]):
    """Spec an entity is constructed from, as passed to ``Scene.add_entity(spec=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits,
    the selectors of its parts and the patches of options set on them. The fields are fixed at construction, while
    ``select`` records into the spec the options set on parts of the entity, where the kind of entity has parts.
    The entity holds a frozen copy, which records nothing more, so later records leave the entity as it is. A copy
    of the spec (``model_copy()``) records again, starting from the patches recorded so far.

    Parameters
    ----------
    morph : Morph | tuple[Morph, ...] | None, optional
        The geometry or asset of the entity. A sequence of morphs makes the entity heterogeneous: each parallel
        environment simulates one of them (see ``Scene.add_entity``). It must be set once the spec reaches
        ``Scene.add_entity``, while the spec given to ``Scene.add_emitter`` leaves it None, since the emitter sets it.
    surface : Surface | None, optional
        The appearance of the entity, its debug representation included (see ``Surface.debug_vis_mode``). If None, use
        ``gs.surfaces.Default()``.
    name : str | None, optional
        The name of the entity, unique within the scene. If None, a name is generated from the morph.
    patches : tuple[genesis.options.patches.Patch, ...], optional
        The patches of options set on selections of parts of the entity, in the order they were set, which ``select``
        and its setters alone record (see ``select``). Default is none.
    """

    # Whether the kind of entity admits several morphs, one per variant of a heterogeneous entity
    _admits_variants: ClassVar[bool] = False
    # The representations of the kind of entity in the debug view (see 'Surface.debug_vis_mode'), the first by default
    _debug_vis_modes: ClassVar[tuple[str, ...]] = ()
    # The patches of options the kind of entity admits, each with the selectors of the parts it is set on, from the
    # coarsest level to the finest.
    _patches_selector_types: ClassVar[dict[type[Options], tuple[type[Options], ...]]] = {}

    # Checked by instance alone, since validation would run the validators of 'Morph' on the sequence of morphs
    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(min_length=1, strict=False)] | None = None
    surface: Surface | None = None
    name: str | None = None
    patches: tuple[Patch, ...] = ()

    _is_frozen: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _check_material_solver(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        material = data.get("material")
        if isinstance(material, Material) and "material" in cls.model_fields:
            spec_cls = get_entity_spec_cls(material)
            if not issubclass(cls, spec_cls):
                gs.raise_exception(
                    f"gs.entities.{cls.__name__} cannot take material {type(material).__name__} of another solver. "
                    f"Pass it to gs.entities.{spec_cls.__name__} instead."
                )
            # The deprecated 'use_visual_raycasting' of the material stands for that of the spec
            if material.use_visual_raycasting is not None and "use_visual_raycasting" in cls.model_fields:
                if data.get("use_visual_raycasting", material.use_visual_raycasting) != material.use_visual_raycasting:
                    gs.raise_exception(
                        "Conflicting values set for 'use_visual_raycasting' of the entity spec and of its material. "
                        "Set the former alone."
                    )
                data = {**data, "use_visual_raycasting": material.use_visual_raycasting}
        return data

    @model_validator(mode="after")
    def _check_fields(self) -> Self:
        if isinstance(self.morph, tuple) and not self._admits_variants:
            gs.raise_exception(
                f"An entity of gs.entities.{type(self).__name__} is created from one morph, got {len(self.morph)}."
            )
        if self.surface is not None:
            self._check_surface(self.surface)
        for patch in self.patches:
            self._check_patch(patch)
        return self

    def _check_surface(self, surface: Surface) -> None:
        """Check that the kind of entity draws what a surface of the entity or of its visual geoms asks for."""
        name = f"gs.entities.{type(self).__name__}"
        if surface.debug_vis_mode is not None and not self._debug_vis_modes:
            gs.raise_exception(
                f"{name} is visual only and has no debug representation, which the debug view leaves out, got "
                f"'debug_vis_mode={surface.debug_vis_mode}'."
            )
        if surface.debug_vis_mode is not None and surface.debug_vis_mode not in self._debug_vis_modes:
            gs.raise_exception(
                f"{name} is drawn in the debug view as one of {list(self._debug_vis_modes)}, got "
                f"'debug_vis_mode={surface.debug_vis_mode}'."
            )
        # The deprecated 'vis_mode' names the representation the material decides, 'visual' for a mesh and 'recon' for a
        # reconstructed surface, or a debug representation of the entity, which then shows it in every view.
        if surface.vis_mode is not None:
            vis_mode = "recon" if self.structure == STRUCTURE.AMORPHOUS else "visual"
            if surface.vis_mode != vis_mode and surface.vis_mode not in self._debug_vis_modes:
                gs.raise_exception(
                    f"The material {type(self.material).__name__} is drawn as '{vis_mode}' outside of the debug view, "
                    f"got 'vis_mode={surface.vis_mode}'. Leave 'vis_mode' unset, and draw the entity as "
                    f"{list(self._debug_vis_modes)} with 'surface.debug_vis_mode' and 'VisOptions(debug=True)'."
                )
            if surface.vis_mode in self._debug_vis_modes and surface.debug_vis_mode not in (None, surface.vis_mode):
                gs.raise_exception(
                    f"'vis_mode={surface.vis_mode}' shows the entity as its debug representation '{surface.vis_mode}' "
                    f"in every view, which conflicts with 'debug_vis_mode={surface.debug_vis_mode}'."
                )

    @property
    def structure(self) -> STRUCTURE:
        """How the shape and the connectivity of the entity change, which its material decides."""
        return self.material.structure

    # A copy records again, frozen or not
    def __copy__(self) -> Self:
        spec = super().__copy__()
        spec._is_frozen = False
        return spec

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        spec = super().__deepcopy__(memo)
        spec._is_frozen = False
        return spec

    def _check_patch(self, patch: Patch) -> None:
        options_name = type(patch.options).__name__
        selector_types = self._patches_selector_types.get(type(patch.options))
        if selector_types is None:
            gs.raise_exception(f"gs.entities.{type(self).__name__} sets no gs.options.{options_name} on its parts.")
        # Each selector selects at the level of the one before it or a finer one
        levels = []
        for selector in patch.selectors:
            if not isinstance(selector, selector_types):
                gs.raise_exception(
                    f"gs.entities.{type(self).__name__} sets no gs.options.{options_name} on parts selected with "
                    f"gs.{get_public_name(type(selector))}."
                )
            levels.append(next(i for i, cls in enumerate(selector_types) if isinstance(selector, cls)))
        if any(level < previous_level for previous_level, level in zip(levels, levels[1:])):
            gs.raise_exception(f"A selection selects at its own level or a finer one, got {patch.selection_repr}.")
        if isinstance(patch.options, VGeom) and patch.options.surface is not None:
            if patch.options.surface.vis_mode in self._debug_vis_modes:
                gs.raise_exception(
                    f"'vis_mode={patch.options.surface.vis_mode}' shows a whole entity as its debug representation, "
                    "which the surface of a visual geom cannot ask for."
                )
            self._check_surface(patch.options.surface)
        # The user places the vertices of a visual geom whose shape never changes, which a solver places otherwise
        if (
            isinstance(patch.options, VGeom)
            and patch.options.dynamic_vertices is not None
            and self.structure != STRUCTURE.RIGID
        ):
            gs.raise_exception(
                f"gs.entities.{type(self).__name__} cannot show DynamicVertices, since its solver places its vertices. "
                "Only the entities of a rigid structure (gs.entities.Rigid and gs.entities.Kinematic) show them."
            )

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant of a heterogeneous entity and a single one otherwise."""
        if self.morph is None:
            return ()
        return self.morph if isinstance(self.morph, tuple) else (self.morph,)

    def _frozen_copy(self, update: dict[str, Any] | None = None) -> Self:
        """Return a copy of this spec, with the fields 'update' gives, that records no more patches."""
        spec = self.model_copy(update=update)
        spec._is_frozen = True
        return spec

    def _record(self, patches: tuple[Patch, ...]) -> None:
        """Record patches after those recorded so far, each checked as at construction."""
        if self._is_frozen:
            gs.raise_exception(
                "The spec an entity holds is frozen. Select parts on a copy of it ('model_copy()') instead."
            )
        for patch in patches:
            self._check_patch(patch)
        # The field is frozen for users, so that 'select' and its setters are the only way to record a patch
        object.__setattr__(self, "patches", (*self.patches, *patches))
        self.__pydantic_fields_set__.add("patches")


SpecT = TypeVar("SpecT", bound=Spec)


class _PartSelection(Generic[SpecT]):
    """A selection of parts of an entity, recording into its entity spec the options set on it.

    It holds the entity spec and the selector of each level leading to the parts, and owns no data. Every setter
    returns the selection, so that calls chain on it, ``end`` returns the entity spec to start another selection
    from, and ``with`` binds the selection to a name for a patch.
    """

    def __init__(
        self, spec: SpecT, selectors: tuple[indices.Link | indices.Geom | indices.VGeom | indices.Vertex, ...]
    ) -> None:
        self._spec = spec
        self._selectors = selectors

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None

    def end(self) -> SpecT:
        """Return the entity spec this selection records into, however deeply it is nested.

        It ends a chain inline, so that one expression configures several selections and gives the spec to
        ``add_entity``: ``spec.select(top).set_is_fixed(True).end().select(bottom).set_is_fixed(True).end()``.
        """
        return self._spec

    def _set(self, patches_options: tuple[Geom | VGeom | Vertex, ...]) -> Self:
        self._spec._record(
            tuple(Patch(selectors=self._selectors, options=patch_options) for patch_options in patches_options)
        )
        return self


class _GeomSetters(_PartSelection[SpecT]):
    """The setters of the quantities of collision geoms, which a link selection applies to every geom of its links."""

    def set(self, *options: Geom) -> Self:
        """Set on the selected geoms the fields each of the options gives explicitly, a later call overriding an earlier one."""
        return self._set(options)

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

    def set_visualize_contact(self, visualize_contact: bool) -> Self:
        """Set whether the contacts of the selected collision geoms are drawn as markers (see ``gs.options.Geom``)."""
        return self._set((Geom(visualize_contact=visualize_contact),))


class _VGeomSetters(_PartSelection[SpecT]):
    """The setters of the appearance of visual geoms, which a link selection applies to each visual geom it holds."""

    def set_surface(self, surface: Surface | None) -> Self:
        """Set the surface of the selected visual geoms (see ``gs.options.VGeom.surface``)."""
        return self._set((VGeom(surface=surface),))


class VGeomSelection(_VGeomSetters[SpecT]):
    """A selection of visual geoms of an entity (see ``gs.entities.Rigid.select``)."""

    def set(self, *options: VGeom) -> Self:
        """Set on the selected visual geoms the fields each of the options gives, overriding an earlier call."""
        return self._set(options)

    def select(self, selector: indices.VGeom) -> "VGeomSelection[SpecT]":
        """Return the selection of the visual geoms of this selection that the selector holds, resolved among them.

        Two rules decide which surface a visual geom takes:

        1. A surface set on a finer level overrides one set on a coarser level: the entity, then a link selection, then
           a visual geom selection. The level is that of the finest selector, however the selection is written.
        2. Between two selections of one level, the later call applies.
        """
        if not isinstance(selector, indices.VGeom):
            gs.raise_exception(
                f"A visual geom selection selects with gs.indices.VGeom, got gs.{get_public_name(type(selector))}."
            )
        return VGeomSelection(self._spec, (*self._selectors, selector))


class GeomSelection(_GeomSetters[SpecT]):
    """A selection of collision geoms of an entity (see ``gs.entities.Rigid.select``)."""

    def select(self, selector: indices.Geom) -> "GeomSelection[SpecT]":
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
        return GeomSelection(self._spec, (*self._selectors, selector))


class LinkSelection(_GeomSetters[SpecT], _VGeomSetters[SpecT]):
    """A selection of links of an entity, whose setters apply to each geom or visual geom of the links (see ``select``).

    A kinematic entity has no collision geoms, so its link selections take ``set_surface`` alone.
    """

    def set(self, *options: Geom | VGeom) -> Self:
        """Set on the geoms of the selected links the fields each of the options gives, overriding an earlier call."""
        return self._set(options)

    @overload
    def select(self, selector: indices.Link) -> "LinkSelection[SpecT]": ...

    @overload
    def select(self, selector: indices.Geom) -> GeomSelection[SpecT]: ...

    @overload
    def select(self, selector: indices.VGeom) -> VGeomSelection[SpecT]: ...

    def select(
        self, selector: indices.Link | indices.Geom | indices.VGeom
    ) -> "LinkSelection[SpecT] | GeomSelection[SpecT] | VGeomSelection[SpecT]":
        """Return the selection of the links, the geoms or the visual geoms of this selection that the selector holds.

        The selector resolves among these links, and a geom ordinal counts each link apart (see ``gs.indices.Geom.at``).

        Two rules decide which value a field of a geom takes:

        1. A value set on a finer level overrides one set on a coarser level: the material, or the surface of the
           entity, then a link selection, then a geom or visual geom selection. The level is that of the finest
           selector, however the selection is written.
        2. Between two selections of one level, the later call applies.
        """
        if isinstance(selector, indices.Geom):
            return GeomSelection(self._spec, (*self._selectors, selector))
        if isinstance(selector, indices.VGeom):
            return VGeomSelection(self._spec, (*self._selectors, selector))
        if not isinstance(selector, indices.Link):
            gs.raise_exception(
                "A link selection selects with gs.indices.Link, gs.indices.Geom or gs.indices.VGeom, got "
                f"gs.{get_public_name(type(selector))}."
            )
        return LinkSelection(self._spec, (*self._selectors, selector))


class VertexSelection(_PartSelection[SpecT]):
    """A selection of vertices of a deformable entity (see ``gs.entities.FEM.select``)."""

    def select(self, selector: indices.Vertex) -> "VertexSelection[SpecT]":
        """Return the selection of the vertices of this selection that the selector holds, resolved among them.

        Two rules decide which value a field of a vertex takes:

        1. A value set on a vertex selection overrides the material, however the selection is written.
        2. Between two vertex selections, the later call applies.
        """
        if not isinstance(selector, indices.Vertex):
            gs.raise_exception(
                f"A vertex selection selects with gs.indices.Vertex, got gs.{get_public_name(type(selector))}."
            )
        return VertexSelection(self._spec, (*self._selectors, selector))

    def set(self, *options: Vertex) -> Self:
        """Set on the selected vertices the fields each of the options gives explicitly, overriding an earlier call."""
        return self._set(options)

    def set_is_fixed(self, is_fixed: bool) -> Self:
        """Set whether the selected vertices stay at their initial positions (see ``gs.options.Vertex.is_fixed``)."""
        return self._set((Vertex(is_fixed=is_fixed),))


class Rigid(Spec["RigidEntity"]):
    """Spec of an entity simulated with rigid-body dynamics.

    Parameters
    ----------
    material : gs.materials.Rigid, optional
        The material of the entity. Defaults to ``gs.materials.Rigid()``.
    visualize_contact : bool, optional
        Whether the contacts of every collision geom of the entity are drawn as markers, the shorthand of
        ``gs.options.Geom.visualize_contact`` set on all of them. Each contact is read back and drawn at every frame,
        so reserve it for the entities being inspected. Default is False.
    dynamic_vertices : gs.surfaces.DynamicVertices | None, optional
        The vertices of every visual geom of the entity, the shorthand of ``gs.options.VGeom.dynamic_vertices`` set on
        all of them. Default is None, the meshes of the asset.
    use_visual_raycasting : bool, optional
        Whether the ray casting on visual geometry hits the visual meshes of the entity: the raycaster sensors and the
        viewer plugins casting on visual geoms (``RaycasterViewerPlugin`` with ``use_visual_geom``). Each of them then
        refits its trees over the visual triangles of the entity as it moves, so reserve it for the entities meant to
        be sensed or picked by their appearance. Default is False.
    """

    _admits_variants = True
    _debug_vis_modes = ("collision", "sdf")
    _patches_selector_types = {Geom: (indices.Link, indices.Geom), VGeom: (indices.Link, indices.VGeom)}

    material: materials.Rigid = Field(default_factory=materials.Rigid)
    visualize_contact: StrictBool = False
    dynamic_vertices: DynamicVertices | None = None
    use_visual_raycasting: StrictBool = False

    @overload
    def select(self, selector: indices.Link) -> LinkSelection[Self]: ...

    @overload
    def select(self, selector: indices.Geom) -> GeomSelection[Self]: ...

    @overload
    def select(self, selector: indices.VGeom) -> VGeomSelection[Self]: ...

    def select(
        self, selector: indices.Link | indices.Geom | indices.VGeom
    ) -> LinkSelection[Self] | GeomSelection[Self] | VGeomSelection[Self]:
        """Select the links, the collision geoms or the visual geoms that the selector holds, to set quantities on.

        Selectors combine with ``|`` and ``&``, each resolving over the whole entity. A selection selects again with
        ``select``, at its own level or a finer one, the nested selector resolving among the parts the selection holds:
        ``spec.select(fingers).select(gs.indices.Geom.at(slice(1, 6)))`` holds geoms 1 to 5 of each finger, and
        ``spec.select(fingers).select(gs.indices.Geom.named("pad*")).select(gs.indices.Geom.at(0))`` the first pad of
        each finger. A quantity set on links applies to every collision geom of them, a surface to every visual geom.

        Two rules decide which value a field of a geom takes:

        1. A value set on a finer level overrides one set on a coarser level: the material or the entity surface, then a
           link selection, then a geom or visual geom selection. The level is that of the finest selector of the
           selection, so how a selection is written (nesting, ``|``, ``&``, several names) never makes it more specific.
        2. Between two selections of one level, the later call applies. The three calls here select links, so the thumb
           ends at 1.0::

               spec.select(gs.indices.Link.subtree("hand")).select(gs.indices.Link.named("thumb")).set_friction(2.0)
               spec.select(gs.indices.Link.subtree("hand") & gs.indices.Link.named("thumb")).set_friction(3.0)
               spec.select(gs.indices.Link.subtree("hand")).set_friction(1.0)

        A selection that holds no geom of the entity, or of one of its heterogeneous variants, raises in ``add_entity``.
        """
        if isinstance(selector, indices.Geom):
            return GeomSelection(self, (selector,))
        if isinstance(selector, indices.VGeom):
            return VGeomSelection(self, (selector,))
        if not isinstance(selector, indices.Link):
            gs.raise_exception(
                "gs.entities.Rigid selects with gs.indices.Link, gs.indices.Geom or gs.indices.VGeom, got "
                f"gs.{get_public_name(type(selector))}."
            )
        return LinkSelection(self, (selector,))


class Kinematic(Spec["KinematicEntity"]):
    """Spec of an entity simulated with kinematics only, with no collision.

    Parameters
    ----------
    material : gs.materials.Kinematic, optional
        The material of the entity. Defaults to ``gs.materials.Kinematic()``.
    dynamic_vertices : gs.surfaces.DynamicVertices | None, optional
        See ``gs.entities.Rigid``.
    use_visual_raycasting : bool, optional
        See ``gs.entities.Rigid``.
    """

    _admits_variants = True
    _patches_selector_types = {VGeom: (indices.Link, indices.VGeom)}

    material: materials.Kinematic = Field(default_factory=materials.Kinematic)
    dynamic_vertices: DynamicVertices | None = None
    use_visual_raycasting: StrictBool = False

    @overload
    def select(self, selector: indices.Link) -> LinkSelection[Self]: ...

    @overload
    def select(self, selector: indices.VGeom) -> VGeomSelection[Self]: ...

    def select(self, selector: indices.Link | indices.VGeom) -> LinkSelection[Self] | VGeomSelection[Self]:
        """Select the links, or the visual geoms, that the selector holds, to set their surface on.

        Selectors combine with ``|`` and ``&``, and a selection selects again with ``select``, as for
        ``gs.entities.Rigid.select``. A kinematic entity has no collision geoms, so its selections set surfaces alone.

        Two rules decide which surface a visual geom takes:

        1. A surface set on a finer level overrides one set on a coarser level: the entity, then a link selection, then
           a visual geom selection. The level is that of the finest selector of the selection.
        2. Between two selections of one level, the later call applies.

        A selection holding no visual geom of the entity, or of a heterogeneous variant, raises in ``add_entity``.
        """
        if isinstance(selector, indices.VGeom):
            return VGeomSelection(self, (selector,))
        if not isinstance(selector, indices.Link):
            gs.raise_exception(
                "gs.entities.Kinematic selects with gs.indices.Link or gs.indices.VGeom, got "
                f"gs.{get_public_name(type(selector))}."
            )
        return LinkSelection(self, (selector,))


class FEM(Spec["FEMEntity"]):
    """Spec of an entity simulated with the finite element method (FEM).

    Parameters
    ----------
    material : gs.materials.FEM.Base
        One of ``gs.materials.FEM.Elastic``, ``Muscle`` or ``Cloth``, a Cloth entity refusing any vertex selection.
    """

    _debug_vis_modes = ("tet",)
    _patches_selector_types = {Vertex: (indices.Vertex,), VGeom: (indices.VGeom,)}

    material: materials.FEM.Base

    def _check_patch(self, patch: Patch) -> None:
        if isinstance(patch.options, Vertex) and isinstance(self.material, materials.FEM.Cloth):
            gs.raise_exception("gs.entities.FEM cannot honour quantities of vertices for a Cloth material.")
        super()._check_patch(patch)

    @overload
    def select(self, selector: indices.Vertex) -> VertexSelection[Self]: ...

    @overload
    def select(self, selector: indices.VGeom) -> VGeomSelection[Self]: ...

    def select(self, selector: indices.Vertex | indices.VGeom) -> VertexSelection[Self] | VGeomSelection[Self]:
        """Select the vertices of the volume mesh, or the visual geoms, that the selector holds, to set quantities on.

        A visual geom selection sets the surface of the visual geoms it holds, over the surface of the entity, the later
        call applying between two of them. A visual geom ordinal counts the visual meshes of the entity as its asset
        splits them (see ``gs.indices.VGeom``).

        Selectors combine with ``|`` and ``&``, each resolving over the whole mesh. A selection selects again with
        ``select``, the nested selector resolving among the vertices the selection holds. With
        ``top = gs.indices.Vertex.in_box(lower, upper)`` and
        ``near = gs.indices.Vertex.closest_to(clamp, max_distance=0.01)``, ``spec.select(top).select(near)`` holds
        the vertex of the top face nearest the clamp, where ``spec.select(top & near)`` holds the vertex of the whole
        mesh nearest the clamp, if it lies on the top face.

        Two rules decide which value a field of a vertex takes:

        1. A value set on a vertex selection overrides the material. Vertices are a single level, so how a selection is
           written (nesting, ``|``, ``&``) never makes it more specific.
        2. Between two vertex selections, the later call applies.

        A selection that holds no vertex, or no visual geom, raises in ``add_entity``.
        """
        if isinstance(selector, indices.VGeom):
            return VGeomSelection(self, (selector,))
        if not isinstance(selector, indices.Vertex):
            gs.raise_exception(
                "gs.entities.FEM selects with gs.indices.Vertex or gs.indices.VGeom, got "
                f"gs.{get_public_name(type(selector))}."
            )
        return VertexSelection(self, (selector,))


class MPM(Spec["MPMEntity"]):
    """Spec of an entity simulated with the material point method (MPM).

    Parameters
    ----------
    material : gs.materials.MPM.Base
        One of ``gs.materials.MPM.Elastic``, ``ElastoPlastic``, ``Liquid``, ``Sand``, ``Snow`` or ``Muscle``.
    """

    _debug_vis_modes = ("particle",)

    material: materials.MPM.Base


class SPH(Spec["SPHEntity"]):
    """Spec of an entity simulated with smoothed particle hydrodynamics (SPH).

    Parameters
    ----------
    material : gs.materials.SPH.Base, optional
        The material of the liquid. Defaults to ``gs.materials.SPH.Liquid()``.
    """

    _debug_vis_modes = ("particle",)

    material: materials.SPH.Base = Field(default_factory=materials.SPH.Liquid)


class PBD(Spec["PBDBaseEntity"]):
    """Spec of an entity simulated with position-based dynamics (PBD).

    Parameters
    ----------
    material : gs.materials.PBD.Base
        The material of the entity, one of ``gs.materials.PBD.Cloth``, ``Elastic``, ``Liquid`` or ``Particle``.
    """

    _debug_vis_modes = ("particle",)

    material: materials.PBD.Base


class Hybrid(Spec["HybridEntity"]):
    """Spec of an entity simulated with a rigid skeleton inside a material point method (MPM) skin.

    Parameters
    ----------
    material : gs.materials.Hybrid
        The material of the skeleton and of the skin, and how they are coupled.
    """

    _debug_vis_modes = ("particle",)

    material: materials.Hybrid

    @property
    def structure(self) -> STRUCTURE:
        """How the shape and the connectivity of the entity change, which the material of its skin decides."""
        return self.material.material_soft.structure


# The entity spec class each material class goes to, the one each spec class declares for its material. A
# material takes the class of the nearest of its bases (see 'get_entity_spec_cls').
_SPEC_CLS_BY_MATERIAL_CLS: dict[type[Any] | None, type[Spec]] = {
    spec_cls.model_fields["material"].annotation: spec_cls
    for spec_cls in (Rigid, Kinematic, FEM, MPM, SPH, PBD, Hybrid)
}
