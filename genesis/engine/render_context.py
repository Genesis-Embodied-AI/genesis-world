"""The visual data of a scene, which every renderer, the ray casting on visual geoms and the visual queries read."""

import dataclasses
import functools
import itertools
import types
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, NamedTuple, TypeVar, get_args, get_origin

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.state_change import StateChange, Subscriber
from genesis.typing import (
    PositiveVec2FType,
    PositiveVec3FType,
    UnitIntervalVec4Type,
    UnitVec3FType,
    UnitVec4FType,
    Vec3FType,
)
from genesis.utils.misc import qd_to_torch, tensor_to_array

if TYPE_CHECKING:
    import trimesh

    from genesis.engine.entities.base_entity import Entity
    from genesis.engine.solvers.base_solver import Solver
    from genesis.options.surfaces import Surface
    from genesis.typing import IndexType


# ------------------------------------------------------------------------------------
# ------------------------------------- shapes ---------------------------------------
# ------------------------------------------------------------------------------------

# The reads of the scene the parts of a shape belong to (see 'Shape.reads'): the scene as it looks, and the scene as the
# debug views show it, each entity in its debug representation or as it looks (see 'RenderContext.set_debug'). A
# representation every entity of a scoped context takes is named after its debug_vis_mode, one of
# 'ScopedRenderContext.debug_vis_modes'.
NORMAL = "normal"
DEBUG = "debug"

# The metadata of the fields of a shape holding one row per environment of the shape
PER_ENV = types.MappingProxyType({"is_per_env": True})


class LinkKey(NamedTuple):
    """The segmentation key of a link."""

    entity: int
    """The index of the entity of the link."""
    link: int
    """The index of the link."""


class GeomKey(NamedTuple):
    """The segmentation key of a geom or a visual geom."""

    entity: int
    """The index of the entity of the geom."""
    link: int
    """The index of the link of the geom."""
    geom: int
    """The index of the geom."""


class MeshKey(NamedTuple):
    """The segmentation key of a visual mesh of an entity without links."""

    entity: int
    """The index of the entity."""
    mesh: int
    """The position of the mesh among those of the entity."""


class SegmentationKeys(NamedTuple):
    """The segmentation key of a drawn part at each level of ``VisOptions.segmentation_level``."""

    entity: int
    """The index of the entity of the part."""
    link: int | LinkKey
    """The key of the part at the link level, the index of its entity for a part without links."""
    geom: int | GeomKey | MeshKey
    """The key of the part at the geom level, for a part without links the index of its entity (particles) or the key of
    its mesh (FEM)."""


class TexturePaths(NamedTuple):
    """The files shipped with Genesis holding the textures of a surface, None for a texture of another origin."""

    diffuse: str | None
    normal: str | None
    emissive: str | None


class PrimitiveAsset(NamedTuple):
    """The primitive shape a mesh is made from, as the morph making it states it."""

    geom_type: gs.GEOM_TYPE
    """The type of the primitive shape."""
    size: tuple[float, ...]
    """The size of the primitive shape."""


class MeshSource(NamedTuple):
    """The origin of the vertices of a mesh loaded unmodified from a file or made from a primitive shape."""

    file: str | None
    """The file holding the mesh, None for a mesh made from a primitive shape."""
    is_zup: bool
    """Whether the file states its vertices with the z axis up."""
    primitive: PrimitiveAsset | None
    """The primitive shape the mesh is made from, None for a mesh loaded from a file."""
    scale: PositiveVec3FType
    """The scale applied to the vertices."""
    is_visual_overwritten: bool
    """Whether the surface replaces the materials of the file."""
    textures_path: TexturePaths
    """The files of the textures of the surface shipped with Genesis."""


class EntityAsset(NamedTuple):
    """The file an entity is loaded from, holding the meshes of all its links, which its meshes share."""

    file: str
    """The file the entity is loaded from."""
    is_zup: bool
    """Whether the file states its vertices with the z axis up."""
    is_merged: bool
    """Whether the fixed links of the file were merged into their parent."""
    scale: PositiveVec3FType
    """The scale the entity places the file at as the scene builds."""
    pos: Vec3FType
    """The position the entity places the file at as the scene builds."""
    quat: UnitVec4FType
    """The orientation the entity places the file at as the scene builds."""


class MeshAsset(NamedTuple):
    """The place of a mesh in the file its entity is loaded from."""

    entity: EntityAsset
    """The file the entity of the mesh is loaded from."""
    link_name: str
    """The link holding the mesh."""
    index: int
    """The position of the mesh among the visual meshes of its link."""


class ShapeRoles(NamedTuple):
    """What each part of a shape draws: each mesh of a StaticMeshBatch, the whole shape otherwise."""

    entities_idx: np.ndarray
    """The entity each part draws."""
    is_looks: np.ndarray
    """Whether each part draws its entity as it looks."""
    is_debug: np.ndarray
    """Whether each part draws the debug representation of its entity."""

    @classmethod
    def of(
        cls, entities_idx: "np.typing.ArrayLike", is_looks: "np.typing.ArrayLike", is_debug: "np.typing.ArrayLike"
    ) -> "ShapeRoles":
        """Return the roles of the parts drawing some entities.

        Parameters
        ----------
        entities_idx : array_like
            The entity each part draws.
        is_looks : array_like
            Whether each part draws its entity as it looks, one flag standing for all.
        is_debug : array_like
            Whether each part draws the debug representation of its entity, one flag standing for all.
        """
        entities_idx = np.array(entities_idx, dtype=gs.np_int, ndmin=1)
        return cls(
            entities_idx, np.broadcast_to(is_looks, entities_idx.shape), np.broadcast_to(is_debug, entities_idx.shape)
        )


class ShapeReads(NamedTuple):
    """Which reads of the scene hold each part of a shape (see ShapeRoles)."""

    normal: np.ndarray
    """Whether the read of the scene as it looks holds each part."""
    debug: np.ndarray
    """Whether the read of the debug views holds each part."""


class Floor(NamedTuple):
    """The plane a mesh stands for, which a renderer reflects the scene in."""

    normal: UnitVec3FType
    """The normal of the plane."""
    size: PositiveVec2FType
    """The size of the plane."""


class ParticlesPhysics(NamedTuple):
    """The physical setting of the particles of an entity, which a renderer simulating foam reads."""

    density: float
    """The density of the material of the particles."""
    gravity: Vec3FType
    """The gravity of the solver of the particles as the scene builds."""
    lower_bound: Vec3FType
    """The lower corner of the domain of the solver of the particles."""
    upper_bound: Vec3FType
    """The upper corner of the domain of the solver of the particles."""


def _to_device(array: np.ndarray) -> torch.Tensor:
    dtype = (
        gs.tc_bool if array.dtype == np.bool_ else gs.tc_int if np.issubdtype(array.dtype, np.integer) else gs.tc_float
    )
    return torch.as_tensor(array, dtype=dtype, device=gs.device)


class ArrayRef(NamedTuple):
    """Some elements of an array of a scoped context, of a solver or of the host.

    A field of a shape given as an ArrayRef is a read-only view of the array, which the render context makes, its
    reference kept in 'Shape.refs' for the kernels reading the array itself, such as those of the ray casting.
    """

    array: "qd.Tensor | torch.Tensor | np.ndarray"
    """The array holding the elements."""
    cols: "slice | int | None" = None
    """The elements of the array, all of them if None."""

    def view(self, is_per_env: bool) -> torch.Tensor:
        """Return the elements of the array, one row per environment for an array holding a value per environment,
        viewing the array with zero-copy and copying it otherwise."""
        if isinstance(self.array, (torch.Tensor, np.ndarray)):
            # An array of the host, such as a mesh computed in memory, is copied to the device
            array = _to_device(self.array) if isinstance(self.array, np.ndarray) else self.array
            if is_per_env:
                return array if self.cols is None else array[:, self.cols]
            return array if self.cols is None else array[self.cols]
        if is_per_env:
            return qd_to_torch(self.array, None, self.cols, keepdim=False, transpose=True)
        return qd_to_torch(self.array, self.cols, keepdim=False, transpose=True)


@dataclass(frozen=True, eq=False)
class MeshGeometry:
    """The constant geometry of one mesh, which a StaticMesh places in each environment."""

    surface: "Surface | None"
    """The surface the mesh is drawn with, None for a marker."""
    uvs: np.ndarray | None = None
    """The texture coordinates (UVs) of the vertices, None for a mesh without."""
    visual: "trimesh.visual.ColorVisuals | trimesh.visual.TextureVisuals | None" = None
    """The appearance of the mesh as its asset states it (materials, textures, colors), None for a mesh drawn with its
    surface alone."""
    seg_keys: SegmentationKeys | None = None
    """The segmentation keys of the mesh, None for a marker."""
    source: MeshSource | None = None
    """Where the vertices of the mesh come from, None whenever 'asset' is."""
    asset: MeshAsset | None = None
    """The place of the mesh in the file its entity is loaded from, for a renderer loading the meshes from their files.
    None for a mesh computed in memory, of an entity whose meshes are drawn apart, or of the debug representation."""
    floor: Floor | None = None
    """The plane the mesh stands for, None for another mesh."""
    verts_ref: ArrayRef = dataclasses.field(kw_only=True, repr=False)
    """The array the vertices view (see 'verts')."""
    normals_ref: ArrayRef = dataclasses.field(kw_only=True, repr=False)
    """The array the normals view (see 'normals')."""
    faces_ref: ArrayRef = dataclasses.field(kw_only=True, repr=False)
    """The array the triangles view, counting the vertices of the array of 'verts_ref' from its start (see 'faces')."""
    is_ray_target: bool = dataclasses.field(default=False, kw_only=True)
    """Whether the ray casting on visual geoms hits the mesh (see ``gs.entities.Rigid.use_visual_raycasting``)."""
    is_smooth: bool = dataclasses.field(default=False, kw_only=True)
    """Whether a marker is drawn smooth, the surface of another mesh deciding."""

    # The views are made at each read: a consumer reads them once to make its own resources, and keeping the triangles
    # counted from the first vertex of the mesh would hold device memory for the life of the scene.
    @property
    def verts(self) -> torch.Tensor:
        """The vertices of the mesh in the frame of its pose, as a read-only view."""
        return self.verts_ref.view(is_per_env=False)

    @property
    def normals(self) -> torch.Tensor:
        """The normals of the vertices in the frame of the pose, those of the asset where it states them."""
        return self.normals_ref.view(is_per_env=False)

    @property
    def faces(self) -> torch.Tensor:
        """The triangles of the mesh, as vertices of 'verts'."""
        vert_start = 0 if self.verts_ref.cols is None else self.verts_ref.cols.start
        return self.faces_ref.view(is_per_env=False) - vert_start


@dataclass(frozen=True, eq=False)
class Shape:
    """A part of the scene as a renderer draws it, in some of its environments.

    Each array of a shape holding a value per environment holds one row per environment of 'envs_idx'. The arrays are
    read-only tensors on the device, a scoped context giving them as host arrays too. The lifetime of a shape follows
    its identity (see 'ScopedRenderContext.update_shapes').
    """

    envs_idx: torch.Tensor | None = dataclasses.field(default=None, kw_only=True)
    """The environments the shape appears in, all of them if None."""
    is_shared: bool = dataclasses.field(default=False, kw_only=True)
    """Whether the shape is drawn once where it stands, in every environment where a view draws one environment and
    once among the environments a view lays out side by side, its arrays holding a single row."""
    roles: ShapeRoles | None = dataclasses.field(default=None, kw_only=True, repr=False)
    """What the parts of the shape draw, which the reads of the scene holding each part follow. The scene as it looks
    and the debug views hold the whole shape if None."""
    reads: ShapeReads | None = dataclasses.field(default=None, init=False, repr=False)
    """Which reads of the scene hold each part of the shape. The render context writes them as the debug state changes,
    so that the shape keeps its identity (see 'RenderContext.set_debug')."""
    # The representations named after a debug_vis_mode the shape belongs to (see 'select_view_shapes')
    _views: frozenset[str] = dataclasses.field(default=frozenset(), kw_only=True, repr=False)
    stale_on: frozenset[StateChange] | None = dataclasses.field(default=None, kw_only=True)
    """The changes of state of a source that leave a marker wrong until the next step in the environments they touch,
    such as a marker showing what a step computed, None for those of its scoped context (see
    'ScopedRenderContext.markers_stale_on')."""
    refs: dict[str, ArrayRef] = dataclasses.field(default_factory=dict, init=False, repr=False)
    """The arrays the fields given as ArrayRef view, by field name."""

    @functools.cached_property
    def envs(self) -> tuple[int, ...]:
        """The environments of the shape on the host."""
        return tuple(tensor_to_array(self.envs_idx).tolist())


@dataclass(frozen=True, eq=False)
class StaticMesh(Shape):
    """One mesh placed across environments, 'static' referring to its vertex data, never to its transform.

    Its arrays are views of the pose block of its StaticMeshBatch.
    """

    geometry: MeshGeometry
    """The geometry of the mesh."""
    pos: torch.Tensor = dataclasses.field(metadata=PER_ENV)
    """The position of the mesh in each environment."""
    quat: torch.Tensor = dataclasses.field(metadata=PER_ENV)
    """The orientation of the mesh in each environment."""
    is_active: torch.Tensor | None = dataclasses.field(default=None, metadata=PER_ENV)
    """Whether each environment holds the mesh, None when every environment does."""
    scale: torch.Tensor | None = dataclasses.field(default=None, metadata=PER_ENV)
    """The linear map applied to the mesh in the frame of its pose in each environment (see 'StaticMeshBatch.scale')."""


@dataclass(frozen=True, eq=False)
class StaticMeshBatch(Shape):
    """The static meshes of a solver, with their transforms packed contiguously in one pose block.

    Indexing and iterating the batch give its meshes as StaticMesh shapes viewing their column of the block, the same
    objects at each call.
    """

    geometries: tuple[MeshGeometry, ...]
    """The geometry of each mesh."""
    pos: "torch.Tensor | ArrayRef" = dataclasses.field(metadata=PER_ENV)
    """The position of each mesh in each environment, one column per mesh."""
    quat: "torch.Tensor | ArrayRef" = dataclasses.field(metadata=PER_ENV)
    """The orientation of each mesh in each environment, one column per mesh."""
    is_active: "torch.Tensor | ArrayRef | None" = dataclasses.field(default=None, metadata=PER_ENV)
    """Whether each environment holds each mesh, None when every environment holds every mesh. Only heterogeneous
    entities leave a mesh out of some environments."""
    aabb_min: "torch.Tensor | ArrayRef | None" = dataclasses.field(default=None, metadata=PER_ENV)
    """The lower corner of the world bounding box of each mesh in each environment, kept current with the poses, None
    when unknown."""
    aabb_max: "torch.Tensor | ArrayRef | None" = dataclasses.field(default=None, metadata=PER_ENV)
    """The upper corner of the world bounding box of each mesh in each environment (see 'aabb_min')."""
    scale: torch.Tensor | None = dataclasses.field(default=None, metadata=PER_ENV)
    """The 3x3 linear map applied to each mesh in each environment in the frame of its pose, its scale and its shear.
    None for meshes drawn at the size of their geometry, such as those of the entities."""

    @functools.cached_property
    def _meshes(self) -> tuple[StaticMesh, ...]:
        return tuple(
            StaticMesh(
                geometry,
                self.pos[:, i_m],
                self.quat[:, i_m],
                is_active=None if self.is_active is None else self.is_active[:, i_m],
                scale=None if self.scale is None else self.scale[:, i_m],
                envs_idx=self.envs_idx,
                is_shared=self.is_shared,
                _views=self._views,
            )
            for i_m, geometry in enumerate(self.geometries)
        )

    def __len__(self) -> int:
        return len(self.geometries)

    def __getitem__(self, i_m: int) -> StaticMesh:
        return self._meshes[i_m]

    def __iter__(self):
        return iter(self._meshes)


@dataclass(frozen=True, eq=False)
class DynamicMesh(Shape):
    """A mesh whose state is its vertices, which move at runtime.

    A surface reconstructed from particles changes topology from one environment to the next, so it is a dynamic mesh
    per environment, made anew at each refresh, its triangles and vertices on the host where it is reconstructed.
    """

    faces: torch.Tensor
    """The triangles of the mesh."""
    surface: "Surface | None"
    """The surface the mesh is drawn with."""
    verts: "torch.Tensor | ArrayRef" = dataclasses.field(metadata=PER_ENV)
    """The world position of the vertices in each environment."""
    uvs: np.ndarray | None = None
    """The texture coordinates (UVs) of the vertices, None for a mesh without."""
    seg_keys: SegmentationKeys | None = None
    """The segmentation keys of the mesh, None for a marker."""
    verts_idx: torch.Tensor | None = None
    """The vertex of 'verts' each vertex of the mesh is at, which a renderer gathers as it uploads them, such as the
    simulated vertices of an FEM entity shared by the vertices of its visual mesh. None when 'verts' holds the vertices
    of the mesh in order."""
    particles: "Points | None" = None
    """The particles the mesh is skinned on or reconstructed from, None otherwise."""
    is_active: "torch.Tensor | ArrayRef | None" = dataclasses.field(default=None, metadata=PER_ENV)
    """Whether each environment shows the mesh, None when every environment does."""
    colors: torch.Tensor | None = dataclasses.field(default=None, metadata=PER_ENV)
    """The RGBA color of each vertex of a marker in each environment."""
    is_ray_target: bool = dataclasses.field(default=False, kw_only=True)
    """Whether the ray casting on visual geoms hits the mesh (see 'MeshGeometry.is_ray_target')."""


@dataclass(frozen=True, eq=False)
class Points(Shape):
    """Points drawn as spheres, such as the particles of an entity or a marker."""

    pos: "torch.Tensor | ArrayRef" = dataclasses.field(metadata=PER_ENV)
    """The position of each point in each environment."""
    radius: float = 0.0
    """The radius of the spheres, zero drawing each point as one pixel. That of the particles of an entity, which a
    renderer scales by ``VisOptions.particle_size_scale``."""
    color: "UnitIntervalVec4Type | torch.Tensor | None" = None
    """The RGBA color of a marker, one color for every point or one per point."""
    surface: "Surface | None" = None
    """The surface the particles of an entity are drawn with."""
    seg_keys: SegmentationKeys | None = None
    """The segmentation keys of the particles of an entity."""
    vel: "torch.Tensor | ArrayRef | None" = dataclasses.field(default=None, metadata=PER_ENV)
    """The velocity of each particle in each environment, which a renderer may stretch them along."""
    is_active: "torch.Tensor | ArrayRef | None" = dataclasses.field(default=None, metadata=PER_ENV)
    """Whether each point is drawn in each environment, None for all of them."""
    physics: ParticlesPhysics | None = None
    """The physical setting of the particles of an entity."""
    is_surface: bool = dataclasses.field(default=False, kw_only=True)
    """Whether the particles stand for the surface reconstructed from them (see 'ScopedRenderContext.reconstruct')."""


@dataclass(frozen=True, eq=False)
class Arrows(Shape):
    """Arrows in each environment.

    An arrow of zero length is left out, which pads the environments holding fewer arrows.
    """

    pos: torch.Tensor = dataclasses.field(metadata=PER_ENV)
    """The start of each arrow in each environment."""
    vec: torch.Tensor = dataclasses.field(metadata=PER_ENV)
    """The vector from the start to the tip of each arrow in each environment."""
    radius: "float | torch.Tensor" = 0.006
    """The radius of the shafts, one for every arrow or one per arrow, in each environment or in all of them."""
    color: "UnitIntervalVec4Type | torch.Tensor" = (1.0, 0.0, 0.0, 1.0)
    """The RGBA color, one for every arrow or one per arrow."""


@dataclass(frozen=True, eq=False)
class Lines(Shape):
    """Lines in each environment.

    A line of zero length is left out, which pads the environments holding fewer lines.
    """

    start: torch.Tensor = dataclasses.field(metadata=PER_ENV)
    """The start of each line in each environment."""
    end: torch.Tensor = dataclasses.field(metadata=PER_ENV)
    """The end of each line in each environment."""
    radius: "float | torch.Tensor" = 0.003
    """The radius of the lines (see 'Arrows.radius')."""
    color: "UnitIntervalVec4Type | torch.Tensor" = (1.0, 0.0, 0.0, 1.0)
    """The RGBA color (see 'Arrows.color')."""


@dataclass(frozen=True, eq=False)
class Frames(Shape):
    """Coordinate frames in each environment."""

    pos: "torch.Tensor | ArrayRef" = dataclasses.field(metadata=PER_ENV)
    """The position of each frame in each environment."""
    quat: "torch.Tensor | ArrayRef" = dataclasses.field(metadata=PER_ENV)
    """The orientation of each frame in each environment."""
    length: float
    """The length of the axes."""
    is_active: torch.Tensor | None = dataclasses.field(default=None, metadata=PER_ENV)
    """Whether each frame is drawn in each environment, None for all of them."""


# The radius and the color of the markers of the constraints of the solvers (see 'gs.options.RigidVisOptions')
CONSTRAINT_POINT_RADIUS = 0.01
CONSTRAINT_LINE_RADIUS = 0.003
CONSTRAINT_COLOR = (1.0, 0.5, 0.0, 1.0)


def select_view_shapes(shapes: "Iterable[Shape]", view: str) -> tuple[Shape, ...]:
    """Return the shapes some part of which a representation of the scene may hold.

    A shape stays returned as the debug state changes, its reads telling which parts are drawn (see 'Shape.reads').

    Parameters
    ----------
    shapes : Iterable[Shape]
        The shapes to select from.
    view : str
        The representation. The scene as it looks (NORMAL) holds the parts drawing their entity as it looks, and the
        debug views (DEBUG) the parts drawing it either way, whatever the representation each entity takes. A
        representation named after a debug_vis_mode holds the shapes stating it.
    """
    if view not in (NORMAL, DEBUG):
        return tuple(shape for shape in shapes if view in shape._views)
    return tuple(
        shape
        for shape in shapes
        if shape.roles is None or (shape.roles.is_looks | (view == DEBUG and shape.roles.is_debug)).any()
    )


# ------------------------------------------------------------------------------------
# ---------------------------------- scoped contexts ---------------------------------
# ------------------------------------------------------------------------------------

# The class of the component of the scene a scoped context draws
OwnerT = TypeVar("OwnerT")


# The scoped context class of each class of component drawing in the render context, inherited by its subclasses, as
# 'ScopedRenderContext.__init_subclass__' records it.
SCOPED_RENDER_CONTEXTS_MAP: dict[type, type["ScopedRenderContext"]] = {}


class ScopedRenderContext(Generic[OwnerT]):
    """The part of the render context one component of the scene draws: a solver, a sensor or the scene itself.

    A class of scoped context states the class of the components it draws as its generic parameter, such as
    ``class KinematicScopedRenderContext(ScopedRenderContext[KinematicSolver])``, which records it for that class and
    its subclasses. The build of a solver, of a sensor or of the scene registers the component, which holds the scoped
    context it gets as ``render_context``, a component drawing nothing holding one that reads empty.
    """

    debug_vis_modes: ClassVar[tuple[str, ...]] = ()
    """The debug representations the scoped context draws its entities in (see ``Surface.debug_vis_mode``)."""

    stale_on: ClassVar[frozenset[StateChange]] = frozenset((StateChange.GEOMETRY,))
    """The changes of state of a source that make the shapes and markers stale, which the next read refreshes. A step
    makes every scoped context stale."""

    markers_stale_on: ClassVar[frozenset[StateChange]] = frozenset()
    """The changes of state of a source that leave the markers wrong until the next step, for the markers stating none
    (see 'Shape.stale_on'). A marker viewing the state, such as the frames of the links, is current once refreshed."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is not None and issubclass(origin, ScopedRenderContext):
                (owner_cls,) = get_args(base)
                if not isinstance(owner_cls, TypeVar):
                    SCOPED_RENDER_CONTEXTS_MAP[owner_cls] = cls

    def __init__(self, owner: OwnerT, context: "RenderContext") -> None:
        self.owner = owner
        self._context = context
        # The markers the user hides (see 'set_hidden')
        self._hidden_markers: set[Shape] = set()
        # The representation the debug views show the entities asking for one in, by entity index (see 'set_debug')
        self._entities_debug: dict[int, bool] = {}

    @property
    def sources(self) -> "tuple[Solver, ...]":
        """The solvers whose changes of state make the scoped context stale (see 'stale_on'), the owner by default."""
        return (self.owner,)

    def get_shapes(
        self,
        envs_idx: "IndexType | None" = None,
        *,
        debug_vis_mode: str | None = None,
        reconstruct_surfaces: bool = True,
    ) -> tuple[Shape, ...]:
        """Return the shapes of every entity of the scoped context in one representation, refreshing it alone first.

        The shapes of each kind follow the order of the entities of the scoped context, a batch covering several
        entities (see StaticMeshBatch) laying out its meshes in that order.

        Parameters
        ----------
        envs_idx : None | int | array_like, optional
            The environments refreshed, all of them if None. Defaults to None.
        debug_vis_mode : str | None, optional
            One of 'debug_vis_modes', or None for the entities as the scene looks. Whatever debug representation each
            entity takes in the debug view, every entity takes the one asked for, such as the collision geometry of
            every rigid entity. Defaults to None.
        reconstruct_surfaces : bool, optional
            See 'RenderContext.get_shapes'. Defaults to True.
        """
        if debug_vis_mode is not None and debug_vis_mode not in self.debug_vis_modes:
            gs.raise_exception(
                f"{type(self).__name__} draws its entities in the debug representations {self.debug_vis_modes}, got "
                f"'{debug_vis_mode}'."
            )
        view = NORMAL if debug_vis_mode is None else debug_vis_mode
        return self._context._read(envs_idx, view, scoped_contexts=(self,), reconstruct_surfaces=reconstruct_surfaces)

    def build(self) -> None:
        """Build the data of the scoped context once its owner is built, the shapes it keeps included."""

    def update_shapes(self, envs_idx: torch.Tensor | None, view: str) -> tuple[Shape, ...]:
        """Refresh the data of the shapes, and return every shape the scoped context shows in a representation.

        The lifetime of a shape follows its identity: a shape returned again as the same object is the same shape, whose
        arrays the scoped context refreshes in place and whose resources a renderer keeps, a new object is a new shape,
        and a shape no longer returned is gone. The render context keeps the shapes the latest call returned for each
        representation, releasing those no representation returns any longer, and refreshes a scoped context at most
        once per change of state and selection of environments. A field of a shape viewing an array of the scoped
        context or of a solver is an ArrayRef.

        It runs for a read of the shapes alone, never for a read of the markers. Points 'is_surface' stand for the
        surface that 'reconstruct' makes from them.

        Parameters
        ----------
        envs_idx : torch.Tensor | None
            The environments the read needs, None for a read of another representation of the data as last refreshed,
            which needs none. A scoped context may refresh more of them, every environment at once where that costs
            less, and leaves out what it cannot bring current, such as what a step computed after a change of state (see
            'Shape.stale_on').
        view : str
            The representation being read (see NORMAL and DEBUG).
        """
        return ()

    def update_markers(self, envs_idx: torch.Tensor) -> tuple[Shape, ...]:
        """Return every marker the scoped context shows, with their arrays written in some environments.

        It runs for a read of the markers alone, so that a view without markers never pays for them. The lifetime of a
        marker follows the rules of the shapes (see 'update_shapes'). A scoped context toggles a marker by returning it
        or not, keeping the object and its arrays meanwhile.

        A marker showing what a step computed (contacts, sensor readings) is wrong once a change of state in its
        'Shape.stale_on' touches some environments, which the render context draws it hidden in until the next step,
        overwriting its arrays there. The render context skips the refreshes while the user hides every marker (see
        'set_hidden').

        Parameters
        ----------
        envs_idx : torch.Tensor
            The environments the read needs, which the scoped context may exceed (see 'update_shapes').
        """
        return ()

    def is_hidden(self, marker: Shape) -> bool:
        """Whether the user hides a marker of the scoped context (see 'set_hidden')."""
        return marker in self._hidden_markers

    def set_hidden(self, marker: Shape, hidden: bool) -> None:
        """Hide a marker the scoped context shows, or show it again, whatever its staleness.

        The read following showing a marker again refreshes it in every environment.
        """
        if hidden:
            self._hidden_markers.add(marker)
        else:
            self._hidden_markers.discard(marker)
        self._context.invalidate((self,))

    def get_debug(self, entity: "Entity") -> bool | None:
        """Whether the debug views show an entity in its debug representation, None for an entity following the render
        context (see 'RenderContext.debug')."""
        return self._entities_debug.get(entity.idx)

    def set_debug(self, entity: "Entity", debug: bool | None) -> None:
        """Show an entity in its debug representation in the debug views, or as it looks, whatever the render context
        asks (see 'RenderContext.set_debug'), or as the render context asks if None.

        The shapes keep their identity: their parts state what they draw (see 'Shape.roles'), from which the render
        context writes the reads holding each part (see 'Shape.reads').
        """
        if debug is not None and entity.surface.debug_vis_mode is None:
            gs.raise_exception(f"Entity '{entity.name}' has no debug representation.")
        if debug is None:
            self._entities_debug.pop(entity.idx, None)
        else:
            self._entities_debug[entity.idx] = debug
        self._context._resolve_reads(self)

    def reconstruct(self, points: Points) -> tuple[DynamicMesh, ...]:
        """Return the surfaces reconstructed from particles standing for them, one per environment."""
        raise NotImplementedError(f"{type(self).__name__} reconstructs no surface.")

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Reset the appearance the scoped context holds in some environments, as a reset of the scene asks."""


# ------------------------------------------------------------------------------------
# -------------------------------------- context -------------------------------------
# ------------------------------------------------------------------------------------


class _Refresh(NamedTuple):
    epoch: int
    # The environments refreshed on the host, None for all of them
    envs_key: tuple[int, ...] | None
    # The number of changes of state recorded for the stale markers (see '_ScopedRecord.changes')
    n_changes: int = 0


@dataclass(eq=False)
class _ScopedRecord:
    scoped_context: ScopedRenderContext
    # The markers the scoped context shows since its latest refresh of the markers
    markers: tuple[Shape, ...] = ()
    # The number of times the scoped context was made stale, and the latest refresh of its shapes and of its markers
    epoch: int = 0
    shapes_refresh: _Refresh | None = None
    markers_refresh: _Refresh | None = None
    # The shapes the scoped context shows in each representation read, and the refresh of the data they were returned
    # at (see 'ScopedRenderContext.update_shapes').
    views_shapes: dict[str, tuple[Shape, ...]] = dataclasses.field(default_factory=dict)
    views_refresh: dict[str, _Refresh] = dataclasses.field(default_factory=dict)
    # The surfaces reconstructed from the particles standing for them
    reconstructed: dict[Points, tuple[DynamicMesh, ...]] = dataclasses.field(default_factory=dict)
    # The copies, without zero-copy, of the arrays each shape or marker views, refreshed with it
    shapes_copies: dict[Shape, list[tuple[torch.Tensor, ArrayRef]]] = dataclasses.field(default_factory=dict)
    markers_copies: dict[Shape, list[tuple[torch.Tensor, ArrayRef]]] = dataclasses.field(default_factory=dict)
    # The changes of state of the sources since the latest step, with the environments the setters gave them for,
    # which leave the markers stating them wrong until the next step (see 'Shape.stale_on').
    changes: list[tuple[StateChange, "IndexType | None"]] = dataclasses.field(default_factory=list)
    # The masks the context draws the markers it hides with, by marker, for the markers telling none
    markers_is_active: dict[Shape, torch.Tensor] = dataclasses.field(default_factory=dict)
    # Whether the user hid every marker at the latest refresh of the markers, which the next refreshes skip until the
    # user shows one again.
    is_markers_hidden: bool = False
    subscribers: set[Subscriber] = dataclasses.field(default_factory=set)


class RenderContext:
    """The visual data of a scene, which the renderers, the ray casting on visual geoms and the visual queries read.

    The simulator owns it. Each component of the scene registers with it as it builds, and holds its scoped context
    (see ScopedRenderContext). A read refreshes the stale scoped contexts it reads first. The visual data is appearance,
    outside every checkpoint.
    """

    def __init__(self, is_debug: bool) -> None:
        # The scoped contexts registered, not built yet
        self._pending: list[ScopedRenderContext] = []
        self._records: list[_ScopedRecord] = []
        # The records of the scoped contexts of each source (see 'ScopedRenderContext.sources')
        self._sources_records: dict["Solver", list[_ScopedRecord]] = {}
        # The record of each shape the scoped contexts show, which notifies the subscribers of the shapes (see
        # 'subscribe'). The shapes are the keys, so that no other object takes their identity while a scoped context
        # shows them, and a shape a scoped context no longer returns leaves.
        self._shapes_record: dict[Shape, _ScopedRecord] = {}
        # The indices of every environment
        self._envs_idx: torch.Tensor | None = None
        # The number of parallel environments, and the sanitizer of a selection of them, which the simulator gives as
        # the context builds.
        self._n_envs = 0
        self._sanitize_envs_idx: "Callable[[IndexType | None], torch.Tensor] | None" = None
        # The subscribers of the context itself (see 'subscribe')
        self._subscribers: set[Subscriber] = set()
        # Whether the debug views show the entities in their debug representation (see 'set_debug')
        self._is_debug = is_debug

    def register(self, owner: object, *, is_empty: bool = False, **kwargs: object) -> ScopedRenderContext:
        """Return the scoped context of a component of the scene, which the component holds.

        The scoped context is of the class recorded for the nearest class in the method resolution order of the
        component. It builds as the context builds, or at the next read for a component registering once the context is
        built. That of a component whose class records none reads empty.

        Parameters
        ----------
        owner : object
            The component registering.
        is_empty : bool, optional
            Whether the component draws nothing in this scene, such as a solver without entity, its scoped context then
            reading empty. Defaults to False.
        **kwargs : object
            The arguments the component passes down to the constructor of its scoped context.
        """
        scoped_context_cls = next(
            (SCOPED_RENDER_CONTEXTS_MAP[cls] for cls in type(owner).__mro__ if cls in SCOPED_RENDER_CONTEXTS_MAP),
            ScopedRenderContext,
        )
        if is_empty or scoped_context_cls is ScopedRenderContext:
            # The base class draws nothing, accepting the debug representations of the class it stands for
            scoped_context = ScopedRenderContext(owner, self)
            scoped_context.debug_vis_modes = scoped_context_cls.debug_vis_modes
            return scoped_context
        scoped_context = scoped_context_cls(owner, self, **kwargs)
        self._pending.append(scoped_context)
        return scoped_context

    def _make_pending(self) -> None:
        # The scoped contexts registered once the context is built build at the first read, once their owner is built
        for scoped_context in self._pending:
            record = _ScopedRecord(scoped_context)
            for source in scoped_context.sources:
                self._sources_records.setdefault(source, []).append(record)
            scoped_context.build()
            self._records.append(record)
        self._pending.clear()

    def _record_shape(self, record: _ScopedRecord, shape: Shape, is_marker: bool) -> None:
        # The fields given as ArrayRef are resolved into views of their arrays, kept current by the refreshes of the
        # scoped context: the array itself with zero-copy, a copy refreshed with the shapes or the markers otherwise.
        # The shapes are frozen to their readers alone, which is why the context writes their fields in place.
        if shape in self._shapes_record:
            return
        self._shapes_record[shape] = record
        self._set_reads(record, shape)
        if shape.envs_idx is None:
            object.__setattr__(shape, "envs_idx", self._envs_idx)
        is_recorded_before = bool(shape.refs)
        for shape_field in dataclasses.fields(shape):
            name = shape_field.name
            value = getattr(shape, name)
            if isinstance(value, ArrayRef):
                shape.refs[name] = value
                object.__setattr__(shape, name, value.view(is_per_env=True))
            elif isinstance(value, np.ndarray) and "torch.Tensor" in str(shape_field.type):
                # The arrays a scoped context computes on the host are copied to the device once, the texture
                # coordinates of a mesh staying on the host where the renderers read them.
                object.__setattr__(shape, name, _to_device(value))
        if not gs.use_zerocopy and shape.refs:
            shape_copies = [(getattr(shape, name), ref) for name, ref in shape.refs.items()]
            (record.markers_copies if is_marker else record.shapes_copies)[shape] = shape_copies
            # A shape returned again once gone holds the copies made as it was first returned
            if is_recorded_before:
                for dst, ref in shape_copies:
                    dst.copy_(ref.view(is_per_env=True))

    def _set_reads(self, record: _ScopedRecord, shape: Shape) -> None:
        # The reads holding each part of a shape stating its roles (see 'Shape.reads'). The debug read holds the debug
        # representation of an entity the debug views show in it, which leaves out an entity without one, such as a
        # kinematic entity.
        roles = shape.roles
        if roles is not None:
            entities_debug = record.scoped_context._entities_debug
            entities_is_debug = np.array(
                [entities_debug.get(i_e, self._is_debug) for i_e in roles.entities_idx], dtype=gs.np_bool
            )
            is_debug = np.where(entities_is_debug, roles.is_debug, roles.is_looks)
            object.__setattr__(shape, "reads", ShapeReads(roles.is_looks, is_debug))

    def build(
        self, n_envs: int, solvers: "Sequence[Solver]", sanitize_envs_idx: "Callable[[IndexType | None], torch.Tensor]"
    ) -> None:
        """Build the scoped contexts registered so far, and refresh their shapes once.

        Parameters
        ----------
        n_envs : int
            The number of parallel environments, 0 for a scene without them.
        solvers : Sequence[Solver]
            The active solvers.
        sanitize_envs_idx : Callable
            The function returning the indices of a selection of environments as a tensor, all of them for None.
        """
        self._n_envs = n_envs
        self._sanitize_envs_idx = sanitize_envs_idx
        self._envs_idx = sanitize_envs_idx(None)

        # The context subscribes to every solver as it builds, ahead of every consumer, so that a change of state
        # reaches the scoped contexts, including those made later, before the consumers reading the context.
        def on_change(solver: "Solver", change: StateChange, envs_idx: "IndexType | None") -> None:
            for record in self._sources_records.get(solver, ()):
                if change in record.scoped_context.stale_on:
                    record.epoch += 1
                # The environments resolve at the next read of the markers, so that a setter pays no read of the device
                record.changes.append((change, envs_idx))

        for solver in solvers:
            solver.subscribe(Subscriber(to=frozenset(StateChange), callback=functools.partial(on_change, solver)))
        self._make_pending()
        self._refresh(self._envs_idx, self._records, is_markers=False, view=NORMAL)

    @property
    def debug(self) -> bool:
        """Whether the debug views show the entities in their debug representation (see 'set_debug')."""
        return self._is_debug

    def set_debug(self, debug: bool) -> None:
        """Show the entities in their debug representation in the debug views, or as they look, except the entities
        their scoped context asks otherwise for (see 'ScopedRenderContext.set_debug').

        The debug views are the interactive viewer and the debug cameras, which draw the markers too. The initial state
        is ``VisOptions.debug``.
        """
        self._is_debug = debug
        for record in self._records:
            self._resolve_reads(record.scoped_context)

    def _resolve_reads(self, scoped_context: ScopedRenderContext) -> None:
        # The parts of the shapes of a scoped context the debug read holds follow the representation each entity takes,
        # which the subscribers of the shapes and of the context are notified of as a change of their appearance.
        for record in self._records:
            if record.scoped_context is scoped_context:
                shapes = itertools.chain(*record.views_shapes.values(), *record.reconstructed.values())
                for shape in dict.fromkeys(shapes):
                    self._set_reads(record, shape)
                for subscriber in record.subscribers:
                    subscriber.notify(StateChange.VISUAL)
        for subscriber in self._subscribers:
            subscriber.notify(StateChange.VISUAL)

    def invalidate(
        self,
        scoped_contexts: "tuple[ScopedRenderContext, ...] | None" = None,
        change: StateChange = StateChange.GEOMETRY,
    ) -> None:
        """Make some scoped contexts stale and notify the subscribers of the context (see 'subscribe').

        A step, a restart, a control of the user changing what a scoped context shows and a write of the appearance of
        an entity (such as 'RigidVisGeom.set_vverts') call it.

        Parameters
        ----------
        scoped_contexts : tuple[ScopedRenderContext, ...] | None, optional
            The scoped contexts to make stale, all of them if None. Defaults to None.
        change : StateChange, optional
            The change the subscribers are notified of: VISUAL for a write of the appearance alone, which a consumer
            syncing once per step shows at once, GEOMETRY otherwise. Defaults to GEOMETRY.
        """
        for record in self._records:
            if scoped_contexts is None or record.scoped_context in scoped_contexts:
                record.epoch += 1
        for subscriber in self._subscribers:
            subscriber.notify(change)

    def advance(self) -> None:
        """Make every scoped context stale after a step, which shows again the markers a change of state hid."""
        # The step recomputes what the markers show, so that the changes before it no longer hide any
        for record in self._records:
            record.changes.clear()
        self.invalidate()

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Reset the appearance the scoped contexts hold in some environments, as a reset of the scene asks."""
        for record in self._records:
            record.scoped_context.reset(envs_idx)
        self.invalidate()

    def subscribe(self, subscriber: Subscriber, shapes: "tuple[Shape, ...] | None" = None) -> None:
        """Register the subscriber of a consumer keeping its own copy of shapes, until 'unsubscribe'.

        Parameters
        ----------
        subscriber : Subscriber
            The subscriber of the consumer.
        shapes : tuple[Shape, ...] | None, optional
            The shapes whose changes the subscriber is notified of: a change of GEOMETRY at each refresh of their scoped
            contexts, and a change of VISUAL as the reads holding their parts change (see 'Shape.reads'). None
            subscribes to the context itself, notified of a change of GEOMETRY as some scoped context is made stale (see
            'invalidate'), and of a change of VISUAL as the reads of some shapes change, which tells a consumer syncing
            once per step to read the context again. Defaults to None.
        """
        if shapes is None:
            self._subscribers.add(subscriber)
            return
        for shape in shapes:
            record = self._shapes_record.get(shape)
            if record is not None:
                record.subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """End the subscription of a subscriber (see 'subscribe')."""
        self._subscribers.discard(subscriber)
        for record in self._records:
            record.subscribers.discard(subscriber)

    def _refresh(
        self, envs_idx: torch.Tensor, records: "list[_ScopedRecord]", is_markers: bool, view: str = NORMAL
    ) -> None:
        # A scoped context refreshes when stale since its latest refresh, or last refreshed for other environments than
        # all of them or those read. A read of the shapes of a representation not read since the latest refresh asks
        # the scoped context for them alone, refreshing nothing (see 'ScopedRenderContext.update_shapes'). A selection
        # viewing the indices of every environment, as a slice or a range does, is never read back on the host.
        envs_key = None
        if envs_idx.data_ptr() != self._envs_idx.data_ptr() or len(envs_idx) != len(self._envs_idx):
            envs_key = tuple(tensor_to_array(envs_idx).tolist())
            if envs_key == tuple(range(len(self._envs_idx))):
                envs_key = None
        for record in records:
            latest = record.markers_refresh if is_markers else record.shapes_refresh
            n_changes = len(record.changes) if is_markers else 0
            is_current = (
                latest is not None
                and latest.epoch == record.epoch
                and latest.n_changes == n_changes
                and latest.envs_key in (None, envs_key)
            )
            if is_current and (is_markers or record.views_refresh.get(view) is latest):
                continue
            scoped_context = record.scoped_context
            if is_markers and record.is_markers_hidden and all(map(scoped_context.is_hidden, record.markers)):
                continue
            if is_markers:
                self._update_markers(record, envs_idx)
                copies = record.markers_copies
                record.markers_refresh = _Refresh(record.epoch, envs_key, n_changes)
            else:
                shapes = scoped_context.update_shapes(envs_idx=None if is_current else envs_idx, view=view)
                for shape in shapes:
                    self._record_shape(record, shape, is_marker=False)
                shapes_before = record.views_shapes.get(view, ())
                record.views_shapes[view] = shapes
                # A shape that no representation returns any longer is gone
                for shape in set(shapes_before).difference(itertools.chain(*record.views_shapes.values())):
                    del self._shapes_record[shape]
                    record.shapes_copies.pop(shape, None)
                    record.reconstructed.pop(shape, None)
                if not is_current:
                    record.shapes_refresh = _Refresh(record.epoch, envs_key)
                record.views_refresh[view] = record.shapes_refresh
                copies = record.shapes_copies
            if is_current:
                continue
            for shape_copies in copies.values():
                for dst, ref in shape_copies:
                    dst.copy_(ref.view(is_per_env=True))
            for subscriber in record.subscribers:
                subscriber.notify(StateChange.GEOMETRY)

    def _update_markers(self, record: _ScopedRecord, envs_idx: torch.Tensor) -> None:
        scoped_context = record.scoped_context
        markers = scoped_context.update_markers(envs_idx)
        for marker in markers:
            self._record_shape(record, marker, is_marker=True)
        for marker in set(record.markers).difference(markers):
            del self._shapes_record[marker]
            record.markers_copies.pop(marker, None)
            record.markers_is_active.pop(marker, None)
        record.markers = markers
        # The markers hidden at the latest refresh are drawn again, then left out of the environments a change of state
        # since the latest step left them wrong in, and of all of them where the user hides them.
        for is_active in record.markers_is_active.values():
            is_active[:] = True
        envs_is_stale_by_changes: dict[frozenset[StateChange], torch.Tensor] = {}
        for marker in markers:
            stale_on = scoped_context.markers_stale_on if marker.stale_on is None else marker.stale_on
            if scoped_context.is_hidden(marker):
                rows = slice(None)
            elif record.changes and stale_on:
                envs_is_stale = envs_is_stale_by_changes.get(stale_on)
                if envs_is_stale is None:
                    envs_is_stale = torch.zeros(len(self._envs_idx), dtype=torch.bool, device=gs.device)
                    for change, envs_idx in record.changes:
                        if change in stale_on:
                            envs_is_stale[self._sanitize_envs_idx(envs_idx)] = True
                    envs_is_stale_by_changes[stale_on] = envs_is_stale
                rows = envs_is_stale[marker.envs_idx]
            else:
                continue
            if isinstance(marker, Arrows):
                marker.vec[rows] = 0.0
            elif isinstance(marker, Lines):
                marker.end[rows] = marker.start[rows]
            elif isinstance(marker, (Points, DynamicMesh, Frames, StaticMeshBatch)):
                if marker.is_active is None:
                    if isinstance(marker, DynamicMesh):
                        is_active_shape = marker.verts.shape[:1]
                    else:
                        is_active_shape = marker.pos.shape[:2]
                    is_active = torch.ones(is_active_shape, dtype=gs.tc_bool, device=gs.device)
                    object.__setattr__(marker, "is_active", is_active)
                    record.markers_is_active[marker] = is_active
                marker.is_active[rows] = False
        record.is_markers_hidden = bool(markers) and all(map(scoped_context.is_hidden, markers))

    def _read(
        self,
        envs_idx: "IndexType | None",
        view: str,
        *,
        scoped_contexts: "tuple[ScopedRenderContext, ...] | None" = None,
        reconstruct_surfaces: bool = True,
    ) -> tuple[Shape, ...]:
        """Return the shapes of the given scoped contexts, all of them if None, in one representation of the scene (see
        NORMAL and DEBUG), refreshing these scoped contexts first in the environments `envs_idx`."""
        # A read of given scoped contexts makes no pending scoped context, whose owner may still be building, such as a
        # sensor registered by the base build of a ray caster reading the scoped contexts of the solvers as it builds.
        if scoped_contexts is None:
            self._make_pending()
        # A scene without parallel environments holds a single one, which every selection reads
        envs_idx = self._envs_idx if envs_idx is None or self._n_envs == 0 else self._sanitize_envs_idx(envs_idx)
        records = [
            record for record in self._records if scoped_contexts is None or record.scoped_context in scoped_contexts
        ]
        self._refresh(envs_idx, records, is_markers=False, view=view)
        shapes: list[Shape] = []
        for record in records:
            for shape in record.views_shapes[view]:
                if isinstance(shape, Points) and shape.is_surface and reconstruct_surfaces:
                    reconstructed = record.reconstructed.get(shape)
                    if reconstructed is None:
                        reconstructed = record.reconstructed[shape] = record.scoped_context.reconstruct(shape)
                        for mesh in reconstructed:
                            self._set_reads(record, mesh)
                    shapes += reconstructed
                else:
                    shapes.append(shape)
        return tuple(shapes)

    def get_shapes(
        self, envs_idx: "IndexType | None" = None, *, debug: bool, reconstruct_surfaces: bool = True
    ) -> tuple[Shape, ...]:
        """Return the shapes the entities are drawn with, refreshing the stale visual data first.

        The shapes of every component of the scene are returned, a consumer telling them apart by their type, and the
        meshes by their segmentation keys. The lifetime of a shape follows its identity (see
        'ScopedRenderContext.update_shapes'): the meshes of the geoms, the visual meshes of the deformable entities and
        the particles are the same objects at each call, so that a renderer builds its resources once.

        Parameters
        ----------
        envs_idx : None | int | array_like, optional
            The environments refreshed, all of them if None. A renderer passes the environments the scene renders
            (``VisOptions.rendered_envs_idx``), so that a large batched scene refreshes those alone, the surfaces
            reconstructed from particles and the detection of torn meshes included. A shape spans its own environments
            (see 'Shape.envs_idx'), a renderer reading the rows of the environments it draws. Defaults to None.
        debug : bool
            False returns the entities as the scene looks, which the structure of their material decides (see
            ``Material.structure``), every part of each shape being drawn. True returns them as the debug views show
            them: each entity in its debug representation (see ``Surface.debug_vis_mode``), the collision geometry of a
            rigid entity, the particles of a particle-based one, the tetrahedra of an FEM one, or as it looks, as its
            scoped context or else the render context asks (see 'set_debug'). Every shape some part of which may take
            part in the debug views is then returned, whatever the debug state, its reads telling which of its parts are
            drawn (see 'Shape.reads').
        reconstruct_surfaces : bool, optional
            False for a consumer reconstructing the surfaces of the entities itself: wherever a surface reconstructed
            from the particles of an entity would be returned (an amorphous entity, or a torn body in the environments
            it tore in), the particles of that entity are returned instead (see Points), and no surface is
            reconstructed. Defaults to True.
        """
        return self._read(envs_idx, view=DEBUG if debug else NORMAL, reconstruct_surfaces=reconstruct_surfaces)

    def get_markers(self, envs_idx: "IndexType | None" = None) -> tuple[Shape, ...]:
        """Return the markers, refreshing the stale visual data first.

        A marker overlays what is not the geometry of an entity: the readings of a sensor, the contacts, link frames and
        constraints of a solver, the boundary confining its particles, and what ``scene.draw_debug_*`` draws. The
        options of their owner decide which markers exist (the ``draw_debug`` of a sensor, the ``vis`` options of a
        solver), and a view wanting them asks for them. Those showing what a step computed (contacts, sensor readings)
        are left out of the environments a change of their state touched until the next step. Markers take no part in
        segmentation.

        Parameters
        ----------
        envs_idx : None | int | array_like, optional
            The environments refreshed, all of them if None (see 'get_shapes'). Defaults to None.
        """
        self._make_pending()
        envs_idx = self._envs_idx if envs_idx is None or self._n_envs == 0 else self._sanitize_envs_idx(envs_idx)
        self._refresh(envs_idx, self._records, is_markers=True)
        return tuple(marker for record in self._records for marker in record.markers)

    def destroy(self) -> None:
        """Release the data, the scoped contexts and the shapes, which the build of the scene makes anew."""
        self._pending.clear()
        self._records.clear()
        self._sources_records.clear()
        self._shapes_record.clear()
        self._subscribers.clear()
        self._envs_idx = None
