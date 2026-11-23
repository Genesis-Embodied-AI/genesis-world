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
from genesis.engine.state_change import ChangeLog, StateChange, Subscriber, TrackableMixin, Tracker
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
    its identity, its 'uid' (see 'ScopedRenderContext.update_shapes').
    """

    uid: gs.UID = dataclasses.field(default_factory=gs.UID, kw_only=True, repr=False)
    """The identity of the shape, which two shapes compare equal by. A fork keeps it (see 'fork'), so that a renderer
    keeps its resources for the shape. A shape made anew takes a new one."""
    envs_idx: torch.Tensor | None = dataclasses.field(default=None, kw_only=True)
    """The environments the shape appears in, all of them if None."""
    is_shared: bool = dataclasses.field(default=False, kw_only=True)
    """Whether the shape is drawn once where it stands, in every environment where a view draws one environment and
    once among the environments a view lays out side by side, its arrays holding a single row."""
    roles: ShapeRoles | None = dataclasses.field(default=None, kw_only=True, repr=False)
    """What the parts of the shape draw, which the reads of the scene holding each part follow. The scene as it looks
    and the debug views hold the whole shape if None."""
    reads: ShapeReads | None = dataclasses.field(default=None, kw_only=True, repr=False)
    """Which reads of the scene hold each part of a shape the render context returns. As the debug state changes, the
    render context returns forks of the shapes holding the new reads (see 'RenderContext.set_debug')."""
    # The representations named after a debug_vis_mode the shape belongs to (see 'select_view_shapes')
    _views: frozenset[str] = dataclasses.field(default=frozenset(), kw_only=True, repr=False)
    invalidated_by: dict[TrackableMixin, tuple[StateChange, ...]] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    """The categories of change of each trackable the content of the shape follows, which the render context reads to
    tell a shape that changed since the latest read. The scoped context refreshes the shape after such a change, or
    hides it where it cannot, a marker (see 'ScopedRenderContext.update_markers'). A shape following nothing changes
    with the controls of its scoped context alone."""
    refs: dict[str, ArrayRef] = dataclasses.field(default_factory=dict, kw_only=True, repr=False)
    """The arrays the fields given as ArrayRef view in a shape the render context returns, by field name."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Shape) and self.uid is other.uid

    def fork(self, **changes: object) -> "Shape":
        """Return a copy of the shape with some fields changed, standing for the same shape.

        A fork keeps the identity of the shape (see 'uid'), so that a renderer drawing the shape keeps its resources and
        draws the fork in its place. A scoped context returns a fork to change what one of its shapes draws, such as the
        flags of the items of a marker to hide some of them, and returns the fork from then on. A shape standing for
        something else is made anew, or forked with 'uid=gs.UID()' to take an identity of its own.

        Parameters
        ----------
        **changes : object
            The new value of each field changed, by field name. The fork holds the arrays of the shape for the others.
        """
        return dataclasses.replace(self, **changes)

    def __hash__(self) -> int:
        return hash(self.uid)

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
    is_active: torch.Tensor | None = dataclasses.field(default=None, metadata=PER_ENV)
    """Whether each arrow is drawn in each environment, None for all of them."""


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
    is_active: torch.Tensor | None = dataclasses.field(default=None, metadata=PER_ENV)
    """Whether each line is drawn in each environment, None for all of them."""


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


class ScopedRenderContext(TrackableMixin, Generic[OwnerT]):
    """The part of the render context one component of the scene draws: a solver, a sensor or the scene itself.

    A class of scoped context states the class of the components it draws as its generic parameter, such as
    ``class KinematicScopedRenderContext(ScopedRenderContext[KinematicSolver])``, which records it for that class and
    its subclasses. The build of a solver, of a sensor or of the scene registers the component, which holds the scoped
    context it gets as ``render_context``, a component drawing nothing holding one that reads empty.
    """

    debug_vis_modes: ClassVar[tuple[str, ...]] = ()
    """The debug representations the scoped context draws its entities in (see ``EntityVisOptions.debug_vis_mode``)."""

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

        The lifetime of a shape follows its identity (see 'Shape.uid'): a shape returned again, as the same object or a
        fork of it, is the same shape, whose arrays the scoped context refreshes in place and whose resources a renderer
        keeps, a shape made anew is a new shape, and a shape no longer returned is gone. The render context keeps the
        shapes the latest call returned for each representation, releasing those no representation returns any longer,
        and refreshes a scoped context at most once per change of state and selection of environments. A field of a
        shape viewing an array of the scoped context or of a solver is an ArrayRef. The render context returns a fork of
        each shape, its fields resolved, and leaves the shapes of the scoped context as they are.

        It runs for a read of the shapes alone, never for a read of the markers. Points 'is_surface' stand for the
        surface that 'reconstruct' makes from them.

        Parameters
        ----------
        envs_idx : torch.Tensor | None
            The environments the read needs, None for a read of another representation of the data as last refreshed,
            which needs none. A scoped context may refresh more of them, every environment at once where that costs
            less.
        view : str
            The representation being read (see NORMAL and DEBUG).
        """
        return ()

    def update_markers(self, envs_idx: torch.Tensor) -> tuple[Shape, ...]:
        """Return every marker the scoped context shows, with their arrays written in some environments.

        It runs for a read of the markers alone, so that a view without markers never pays for them. The lifetime of a
        marker follows the rules of the shapes (see 'update_shapes'). A scoped context toggles a marker by returning it
        or not, keeping the object and its arrays meanwhile.

        A marker showing what the owner computed draws what its getters deliver, and nothing where they return None.
        The render context skips the refreshes while the user hides every marker (see 'set_hidden').

        Parameters
        ----------
        envs_idx : torch.Tensor
            The environments the read needs, which the scoped context may exceed (see 'update_shapes').
        """
        return ()

    def is_hidden(self, marker: Shape) -> bool:
        """Whether the user hides a marker of the scoped context (see 'set_hidden')."""
        return marker in self._hidden_markers

    def resize_marker(self, marker: Shape, n_items: int) -> Shape:
        """Return a marker holding a number of points, arrows, lines or frames, its arrays viewing that many rows.

        The rows past the number held before are active, and the scoped context writes them before returning the
        marker. The renderers draw arrays of a size of their own, which the render context grows and shrinks seldom.

        Parameters
        ----------
        marker : Shape
            The marker, made with the flags of its items ('is_active').
        n_items : int
            The number of items the marker holds from then on.

        Returns
        -------
        marker : Shape
            A fork of the marker (see 'Shape.fork'), which the scoped context returns from then on.
        """
        return self._context._resize_marker(marker, n_items)

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
        if debug is not None and entity.spec.vis.debug_vis_mode is None:
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


@dataclass(eq=False)
class _MarkerStorage:
    # The latest marker the scoped context resized, whose arrays view the first 'n_items' rows of those of the marker
    # the renderers draw, which hold 'capacity' rows (see 'ScopedRenderContext.resize_marker').
    marker: Shape
    static: Shape
    capacity: int
    n_items: int


class _RecordedShape(NamedTuple):
    # The object of a shape the scoped context returned last, and the fork of it the render context returns
    returned: Shape
    resolved: Shape
    record: "_ScopedRecord"


@dataclass(eq=False)
class _ScopedRecord:
    scoped_context: ScopedRenderContext
    # The markers the scoped context shows since its latest refresh of the markers, and the markers the renderers draw
    # for them (see '_update_markers').
    markers: tuple[Shape, ...] = ()
    markers_drawn: tuple[Shape, ...] = ()
    # Whether a control of the scoped context made the shapes and the markers stale (see 'invalidate'), the
    # environments the latest refresh of the shapes and of the markers covered, None for all of them, and the number of
    # refreshes of the shapes.
    is_shapes_stale: bool = True
    is_markers_stale: bool = True
    shapes_envs_key: tuple[int, ...] | None = None
    markers_envs_key: tuple[int, ...] | None = None
    n_shapes_refreshes: int = 0
    # The shapes the scoped context shows in each representation read, and the refresh of the shapes they were
    # returned at (see 'ScopedRenderContext.update_shapes').
    views_shapes: dict[str, tuple[Shape, ...]] = dataclasses.field(default_factory=dict)
    # What the shapes of every representation follow, merged as they are returned (see 'Shape.invalidated_by'), and the
    # changes of each trackable the shapes and the markers follow since their latest refresh.
    shapes_invalidated_by: dict[TrackableMixin, tuple[StateChange, ...]] = dataclasses.field(default_factory=dict)
    shapes_trackers: dict[TrackableMixin, Tracker] = dataclasses.field(default_factory=dict)
    markers_trackers: dict[TrackableMixin, Tracker] = dataclasses.field(default_factory=dict)
    views_n_shapes_refreshes: dict[str, int] = dataclasses.field(default_factory=dict)
    # The surfaces reconstructed from the particles standing for them
    reconstructed: dict[Points, tuple[DynamicMesh, ...]] = dataclasses.field(default_factory=dict)
    # The copies, without zero-copy, of the arrays each shape or marker views, refreshed with it
    shapes_copies: dict[Shape, list[tuple[torch.Tensor, ArrayRef]]] = dataclasses.field(default_factory=dict)
    markers_copies: dict[Shape, list[tuple[torch.Tensor, ArrayRef]]] = dataclasses.field(default_factory=dict)
    # Whether the user hid every marker at the latest refresh of the markers, which the next refreshes skip until the
    # user shows one again.
    is_markers_hidden: bool = False


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
        # The record of each shape the scoped contexts show, which notifies the subscribers of the shapes (see
        # 'subscribe'). A shape a scoped context no longer returns leaves.
        self._shapes_record: dict[Shape, _RecordedShape] = {}
        # The storage of the markers of a size of their own, by marker of the scoped contexts
        self._markers_storage: dict[Shape, _MarkerStorage] = {}
        # The indices of every environment
        self._envs_idx: torch.Tensor | None = None
        # The number of parallel environments, and the sanitizer of a selection of them, which the simulator gives as
        # the context builds.
        self._n_envs = 0
        self._sanitize_envs_idx: "Callable[[IndexType | None], torch.Tensor] | None" = None
        # The eager subscribers of the context, called back as a scoped context is made stale (see 'subscribe')
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
            scoped_context.changelog = ChangeLog(max(self._n_envs, 1))
            scoped_context.build()
            self._records.append(record)
        self._pending.clear()

    def _record_shape(self, record: _ScopedRecord, shape: Shape, is_marker: bool) -> Shape:
        # The render context returns a fork of each object a scoped context returns, and keeps it while the scoped
        # context returns the same object. The fields given as ArrayRef are resolved into views of their arrays, kept
        # current by the refreshes of the scoped context: the array itself with zero-copy, a copy refreshed with the
        # shapes or the markers otherwise.
        recorded = self._shapes_record.get(shape)
        if recorded is not None and recorded.returned is shape:
            return recorded.resolved
        values = {}
        refs = {}
        for shape_field in dataclasses.fields(shape):
            name = shape_field.name
            value = getattr(shape, name)
            if isinstance(value, ArrayRef):
                refs[name] = value
                values[name] = value.view(is_per_env=True)
            elif isinstance(value, np.ndarray) and "torch.Tensor" in str(shape_field.type):
                # The arrays a scoped context computes on the host are copied to the device once, the texture
                # coordinates of a mesh staying on the host where the renderers read them.
                values[name] = _to_device(value)
        envs_idx = self._envs_idx if shape.envs_idx is None else shape.envs_idx
        reads = self._get_reads(record, shape)
        resolved = shape.fork(envs_idx=envs_idx, refs=refs, reads=reads, **values)
        if not gs.use_zerocopy and refs:
            shape_copies = [(values[name], ref) for name, ref in refs.items()]
            (record.markers_copies if is_marker else record.shapes_copies)[shape] = shape_copies
        self._shapes_record[shape] = _RecordedShape(shape, resolved, record)
        return resolved

    def _get_reads(self, record: _ScopedRecord, shape: Shape) -> ShapeReads | None:
        # The reads holding each part of a shape stating its roles (see 'Shape.reads'). The debug read holds the debug
        # representation of an entity the debug views show in it, which leaves out an entity without one, such as a
        # kinematic entity.
        roles = shape.roles
        if roles is None:
            return None
        entities_debug = record.scoped_context._entities_debug
        entities_is_debug = np.array(
            [entities_debug.get(i_e, self._is_debug) for i_e in roles.entities_idx], dtype=gs.np_bool
        )
        return ShapeReads(roles.is_looks, np.where(entities_is_debug, roles.is_debug, roles.is_looks))

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
                # The shapes are returned as forks holding the new reads, which keep their identity (see 'Shape.uid')
                shapes_fork = {
                    shape: shape.fork(reads=self._get_reads(record, shape))
                    for shape in itertools.chain(*record.views_shapes.values(), *record.reconstructed.values())
                }
                for shape, fork in shapes_fork.items():
                    self._shapes_record[shape] = self._shapes_record[shape]._replace(resolved=fork)
                for view, shapes in record.views_shapes.items():
                    record.views_shapes[view] = tuple(map(shapes_fork.get, shapes))
                for points, meshes in record.reconstructed.items():
                    record.reconstructed[points] = tuple(map(shapes_fork.get, meshes))
                if scoped_context.changelog is not None:
                    scoped_context.changelog.add(StateChange.VISUAL)
        for subscriber in self._subscribers:
            subscriber.notify(StateChange.VISUAL)

    def invalidate(
        self,
        scoped_contexts: "tuple[ScopedRenderContext, ...] | None" = None,
        change: StateChange = StateChange.GEOMETRY,
    ) -> None:
        """Make some scoped contexts stale and call back the eager subscribers of the context.

        The subscribers are those registered through 'subscribe'. A restart, a control of the user changing what a
        scoped context shows and a write of the appearance of an entity (such as 'RigidVisGeom.set_vverts') call it.

        Parameters
        ----------
        scoped_contexts : tuple[ScopedRenderContext, ...] | None, optional
            The scoped contexts to make stale, all of them if None. Defaults to None.
        change : StateChange, optional
            The change the subscribers are called back for: VISUAL for a write of the appearance alone, which a consumer
            syncing once per step shows at once, GEOMETRY otherwise. Defaults to GEOMETRY.
        """
        for record in self._records:
            if scoped_contexts is None or record.scoped_context in scoped_contexts:
                record.is_shapes_stale = record.is_markers_stale = True
        for subscriber in self._subscribers:
            subscriber.notify(change)

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Reset the appearance the scoped contexts hold in some environments, as a reset of the scene asks."""
        for record in self._records:
            record.scoped_context.reset(envs_idx)
        self.invalidate()

    def subscribe(self, subscriber: Subscriber) -> None:
        """Register an eager subscriber, called back as a scoped context is made stale or the reads of shapes change.

        The subscriber stays until 'unsubscribe'. It is called back by 'invalidate' and as the reads of some shapes
        change (see 'Shape.reads'). A consumer keeping its own copy of some shapes tracks them instead (see 'track').
        """
        self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """End the subscription of a subscriber (see 'subscribe')."""
        self._subscribers.discard(subscriber)

    def track(self, shape: Shape, *changes: StateChange) -> Tracker:
        """Return a tracker of the changes of a shape, for a consumer keeping its own copy of it.

        A change of GEOMETRY is a refresh of the arrays of the shape, and a change of VISUAL a change of the reads holding
        its parts (see 'Shape.reads'). The scoped context showing the shape records them for all its shapes at once.

        Parameters
        ----------
        shape : Shape
            A shape the render context returned.
        *changes : StateChange
            The categories the tracker reports.
        """
        return self._shapes_record[shape].record.scoped_context.changelog.track(*changes)

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
            scoped_context = record.scoped_context
            if is_markers:
                # Every read asks the scoped context for its markers, which refreshes them as it sees fit. They changed
                # if one was added or removed, if a change one of them states occurred since the latest refresh, if a
                # control made them stale, or if other environments are read (see 'Shape.invalidated_by').
                if record.is_markers_hidden and all(map(scoped_context.is_hidden, record.markers)):
                    continue
                markers_before = record.markers
                is_current = (
                    not record.is_markers_stale
                    and record.markers_envs_key in (None, envs_key)
                    and not any(
                        trackable not in record.markers_trackers
                        or record.markers_trackers[trackable].is_modified(*changes)
                        for marker in markers_before
                        for trackable, changes in marker.invalidated_by.items()
                    )
                )
                self._update_markers(record, envs_idx)
                markers_trackers = {}
                for marker in record.markers:
                    for trackable in marker.invalidated_by:
                        markers_trackers[trackable] = (
                            record.markers_trackers.get(trackable) or trackable.changelog.track()
                        )
                for tracker in markers_trackers.values():
                    tracker.pull()
                record.markers_trackers = markers_trackers
                is_current = is_current and set(record.markers) == set(markers_before)
                copies = record.markers_copies
                record.markers_envs_key = envs_key
                record.is_markers_stale = False
            else:
                # The shapes changed if a change one of them states occurred since their latest refresh, keyed by the
                # record apart from the markers, if a control made them stale, or if other environments are read.
                is_current = (
                    not record.is_shapes_stale
                    and record.shapes_envs_key in (None, envs_key)
                    and not any(
                        trackable not in record.shapes_trackers
                        or record.shapes_trackers[trackable].is_modified(*changes)
                        for trackable, changes in record.shapes_invalidated_by.items()
                    )
                )
                if is_current and record.views_n_shapes_refreshes.get(view) == record.n_shapes_refreshes:
                    continue
                # The trackers are pulled as the refresh starts, so that a change the refresh records itself counts for
                # the next one, and those of the trackables the new shapes follow first as it ends.
                for tracker in record.shapes_trackers.values():
                    tracker.pull()
                shapes = tuple(
                    self._record_shape(record, shape, is_marker=False)
                    for shape in scoped_context.update_shapes(envs_idx=None if is_current else envs_idx, view=view)
                )
                shapes_before = record.views_shapes.get(view, ())
                record.views_shapes[view] = shapes
                shapes_invalidated_by: dict[TrackableMixin, tuple[StateChange, ...]] = {}
                for shape in itertools.chain(*record.views_shapes.values()):
                    for trackable, changes in shape.invalidated_by.items():
                        trackable_changes = {*shapes_invalidated_by.get(trackable, ()), *changes}
                        shapes_invalidated_by[trackable] = tuple(sorted(trackable_changes))
                shapes_trackers = {}
                for trackable in shapes_invalidated_by:
                    tracker = record.shapes_trackers.get(trackable)
                    if tracker is None:
                        tracker = trackable.changelog.track()
                        tracker.pull()
                    shapes_trackers[trackable] = tracker
                record.shapes_invalidated_by, record.shapes_trackers = shapes_invalidated_by, shapes_trackers
                # A shape that no representation returns any longer is gone
                for shape in set(shapes_before).difference(itertools.chain(*record.views_shapes.values())):
                    del self._shapes_record[shape]
                    record.shapes_copies.pop(shape, None)
                    for mesh in record.reconstructed.pop(shape, ()):
                        del self._shapes_record[mesh]
                if not is_current:
                    record.shapes_envs_key = envs_key
                    record.n_shapes_refreshes += 1
                    record.is_shapes_stale = False
                record.views_n_shapes_refreshes[view] = record.n_shapes_refreshes
                copies = record.shapes_copies
            if is_current:
                continue
            for shape_copies in copies.values():
                for dst, ref in shape_copies:
                    dst.copy_(ref.view(is_per_env=True))
            scoped_context.changelog.add(StateChange.GEOMETRY, envs_idx=envs_idx)

    def _update_markers(self, record: _ScopedRecord, envs_idx: torch.Tensor) -> None:
        scoped_context = record.scoped_context
        markers = scoped_context.update_markers(envs_idx)
        for marker in set(record.markers).difference(markers):
            del self._shapes_record[marker]
            record.markers_copies.pop(marker, None)
            storage = self._markers_storage.pop(marker, None)
            if storage is not None:
                self._shapes_record.pop(storage.static, None)
        # A marker of a size of its own is drawn through the arrays of its storage, a fork of the marker the scoped
        # context resized (a hidden one) drawing its flags over their rows. A marker the user hides is drawn as a fork
        # without any item, which leaves the flags of the scoped context as it wrote them.
        markers_drawn = []
        for marker in markers:
            marker_drawn = self._record_shape(record, marker, is_marker=True)
            is_hidden = scoped_context.is_hidden(marker)
            storage = self._markers_storage.get(marker)
            if storage is not None:
                static = self._record_shape(record, storage.static, is_marker=True)
                if is_hidden or marker is not storage.marker:
                    is_active = torch.zeros_like(static.is_active)
                    if not is_hidden:
                        is_active[:, : storage.n_items] = marker_drawn.is_active
                    static = static.fork(is_active=is_active)
                marker_drawn = static
            elif is_hidden:
                if marker_drawn.is_active is not None:
                    is_active_shape = marker_drawn.is_active.shape
                elif isinstance(marker_drawn, DynamicMesh):
                    is_active_shape = marker_drawn.verts.shape[:1]
                elif isinstance(marker_drawn, Lines):
                    is_active_shape = marker_drawn.start.shape[:2]
                else:
                    is_active_shape = marker_drawn.pos.shape[:2]
                is_active = torch.zeros(is_active_shape, dtype=gs.tc_bool, device=gs.device)
                marker_drawn = marker_drawn.fork(is_active=is_active)
            markers_drawn.append(marker_drawn)
        record.markers = markers
        record.markers_drawn = tuple(markers_drawn)
        record.is_markers_hidden = bool(markers) and all(map(scoped_context.is_hidden, markers))

    def _resize_marker(self, marker: Shape, n_items: int) -> Shape:
        # The capacity doubles once exceeded and halves once above four times the number of items, so that a number of
        # items going back and forth reallocates seldom.
        storage = self._markers_storage.get(marker)
        items = marker.start if isinstance(marker, Lines) else marker.pos
        n_items_before = items.shape[1] if storage is None else storage.n_items
        capacity = 0 if storage is None else storage.capacity
        # The rows past the number of items are hidden through the flags, which a resized marker is made with
        assert marker.is_active is not None
        # The arrays holding a row per item, along their second axis
        names = [
            field.name
            for field in dataclasses.fields(marker)
            if isinstance(value := getattr(marker, field.name), torch.Tensor)
            and value.ndim >= 2
            and value.shape[1] == n_items_before
        ]
        if storage is None or n_items > capacity or n_items < capacity // 4:
            if n_items > capacity:
                capacity = max(n_items, 2 * capacity)
            while n_items < capacity // 4:
                capacity //= 2
            arrays = {}
            n_kept = min(n_items_before, n_items)
            for name in names:
                value = getattr(marker, name)
                array = torch.zeros((value.shape[0], capacity, *value.shape[2:]), dtype=value.dtype, device=gs.device)
                array[:, :n_kept] = value[:, :n_kept]
                arrays[name] = array
            # The renderers draw arrays of another size, which a marker of a new identity tells them
            static = marker.fork(uid=gs.UID(), **arrays)
            if storage is not None:
                self._shapes_record.pop(storage.static, None)
            storage = self._markers_storage[marker] = _MarkerStorage(marker, static, capacity, n_items)
        is_active = storage.static.is_active
        is_active[:, n_items_before:n_items] = True
        is_active[:, n_items:] = False
        storage.n_items = n_items
        storage.marker = marker.fork(**{name: getattr(storage.static, name)[:, :n_items] for name in names})
        return storage.marker

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
                    # Reconstruction runs on the host per environment, so a surface no view draws is skipped. A shape
                    # stating no reads is drawn by every read (see 'Shape.reads').
                    reads = shape.reads
                    if reads is not None and not (reads.debug if view == DEBUG else reads.normal).any():
                        continue
                    reconstructed = record.reconstructed.get(shape)
                    if reconstructed is None:
                        reconstructed = record.reconstructed[shape] = tuple(
                            self._record_shape(record, mesh, is_marker=False)
                            for mesh in record.scoped_context.reconstruct(shape)
                        )
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
            False returns the entities as the scene looks (see ``Material.structure``). True returns them as the debug
            views show them, each in its debug representation (see ``EntityVisOptions.debug_vis_mode``) or as it looks
            (see 'set_debug'). Every shape that may take part in the debug views is then returned, its reads telling
            which parts are drawn (see 'Shape.reads'), but a surface reconstructed from particles only while drawn.
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
        solver), and a view wanting them asks for them. Markers take no part in segmentation.

        Parameters
        ----------
        envs_idx : None | int | array_like, optional
            The environments refreshed, all of them if None (see 'get_shapes'). Defaults to None.
        """
        self._make_pending()
        envs_idx = self._envs_idx if envs_idx is None or self._n_envs == 0 else self._sanitize_envs_idx(envs_idx)
        self._refresh(envs_idx, self._records, is_markers=True)
        return tuple(marker for record in self._records for marker in record.markers_drawn)

    def destroy(self) -> None:
        """Release the data, the scoped contexts and the shapes, which the build of the scene makes anew."""
        self._pending.clear()
        self._records.clear()
        self._shapes_record.clear()
        self._markers_storage.clear()
        self._subscribers.clear()
        self._envs_idx = None
