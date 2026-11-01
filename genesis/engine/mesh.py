import functools
import math
import os
import pickle
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

import fast_simplification
import igl
import trimesh
from scipy.spatial import ConvexHull, QhullError

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.gltf as gltf_utils
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
import genesis.utils.point_cloud as pc
from genesis.constants import GLTF_FORMATS, MESH_FORMATS
from genesis.options.surfaces import Surface
from genesis.repr_base import RBC
from genesis.typing import Matrix3x3Type, Vec3FType
from genesis.utils import serialization
from genesis.utils.cache import OwnedCache, get_gsd_path, get_hashkey, get_remesh_cache
from genesis.utils.misc import redirect_libc_stderr

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.description import SignedDistanceFieldDescription


class InertialProperties(NamedTuple):
    """A rigid body's intrinsic inertial: mass, center of mass 'com', and inertia tensor 'i' about that COM."""

    mass: float
    com: Vec3FType
    i: Matrix3x3Type


class SDFGrid(NamedTuple):
    """Signed distance field (SDF) of a geometry sampled at the nodes of a regular grid.

    'value' and 'grad' hold the signed distance and its gradient at each node, 'closest_verts_idx' the index of the
    vertex of the geometry closest to each node, 'max_value' the largest signed distance of the grid, and
    'T_mesh_to_sdf' the transform from the mesh frame to the grid frame, in which the nodes sit at integer coordinates.
    """

    value: np.ndarray
    grad: np.ndarray
    closest_verts_idx: np.ndarray
    max_value: float
    T_mesh_to_sdf: np.ndarray


class SupportTable(NamedTuple):
    """Support table of a geometry: the vertices supporting the directions of each cell of a grid of directions.

    'slots_vid' holds the first two candidates of each cell, the first one repeated when the cell has a single one,
    'extras_vid' the other candidates of all cells, in cell order, and 'cells_n_extras' their count per cell. Vertex
    indices are local to the geometry (see 'SupportField').
    """

    slots_vid: np.ndarray
    extras_vid: np.ndarray
    cells_n_extras: np.ndarray


# Rotation from the mesh frame to the frame of each spherical chart of the support table. The second chart carries its
# poles a quarter turn away from those of the first, onto +/- y (see _func_direction_cell in support_field.py).
CHARTS_ROT = np.stack((np.eye(3), gu.euler_to_R(np.array((90.0, 0.0, 0.0)))))

# Data derived from the trimesh of a mesh alone, shared by the meshes of one trimesh (see 'Mesh.tmesh_hash')
_TMESH_CACHE = OwnedCache()


def cached_by_tmesh(method):
    """Wrap a method of 'Mesh' computing data from its trimesh alone, so that the meshes of one trimesh share it.

    The value is computed by the method on the first call, keyed on its name, the hash of the trimesh (see
    'Mesh.tmesh_hash') and its arguments, which must be hashable.
    """

    @functools.wraps(method)
    def wrapper(self, *args):
        return _TMESH_CACHE.get(self, (method.__name__, self.tmesh_hash, *args), functools.partial(method, self, *args))

    return wrapper


def mutates_tmesh(method):
    """Wrap a method of 'Mesh' changing its trimesh, so that the mesh gives up the data derived from its former one.

    The data derived from the trimesh alone is shared through the trimesh cache (see 'Mesh.tmesh_hash'), which the mesh
    releases once the method returns, its hash being computed again from the new trimesh when next asked.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._tmesh_hash = None
        _TMESH_CACHE.release(self)
        return result

    return wrapper


class Mesh(RBC, serialization.SerializationMixin):
    """
    Genesis's own triangle mesh object.

    This is a wrapper of `trimesh.Trimesh` with some additional features and attributes. The internal trimesh object
    can be accessed via `self.trimesh`.

    Parameters
    ----------
    surface : genesis.Surface
        The mesh's surface object.
    uvs : np.ndarray
        The mesh's uv coordinates.
    convexify : bool
        Whether to convexify the mesh.
    decimate : bool
        Whether to decimate the mesh.
    decimate_face_num : int
        The target number of faces after decimation.
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters
        the original geometry if necessary. 8 does what needs to be done at all costs. Default to 0.
    metadata : dict
        The metadata of the mesh.
    """

    def __init__(
        self,
        mesh,
        surface: Surface | None = None,
        uvs: "np.typing.NDArray | None" = None,
        scale: "np.typing.NDArray | float | None" = None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=0,
        metadata=None,
        is_mesh_zup: bool = True,
    ):
        self._uid = gs.UID()
        self._mesh = mesh  # .copy() FIXME: For some reason forcing copy is causing some tests to fails...
        self._surface = surface
        if uvs is not None:
            uvs = uvs.astype(gs.np_float, copy=False)
        self._uvs = uvs
        self._metadata: dict[str, Any] = metadata or {}
        self._color = np.array([1.0, 1.0, 1.0, 1.0], dtype=gs.np_float)

        # The data derived from the trimesh alone is shared through the trimesh cache (see 'tmesh_hash')
        self._tmesh_hash: "str | None" = None

        # By default, all meshes are considered zup, unless the "FileMorph.file_meshes_are_zup" option was set to False
        self._metadata.setdefault("imported_as_zup", True)

        # By default, all meshes are considered having their original visual
        self._metadata.setdefault("is_visual_overwritten", False)

        if not is_mesh_zup:
            if self._metadata["imported_as_zup"]:
                self._mesh.apply_transform(mu.Y_UP_TRANSFORM.T)
            self._metadata["imported_as_zup"] = False

        if scale is not None:
            scale = np.atleast_1d(np.asarray(scale))
            assert scale.ndim == 1 and scale.size in (1, 3)
            self._mesh.apply_scale(scale)

        if self._surface.requires_uv:  # check uvs here
            if self._uvs is None:
                if "mesh_path" in self._metadata:
                    gs.logger.warning(
                        f"Texture given but asset missing uv info (or failed to load): {self._metadata['mesh_path']}"
                    )
                else:
                    gs.logger.warning("Texture given but asset missing uv info (or failed to load).")
        else:
            self._uvs = None

        if convexify:
            self.convexify()

        if decimate:
            self.decimate(decimate_face_num, decimate_aggressiveness)

    @mutates_tmesh
    def convexify(self):
        """
        Convexify the mesh.
        """
        if self._mesh.vertices.shape[0] > 3:
            self._mesh = mu.convex_hull(self._mesh)
            self._metadata["convexified"] = True
        self.clear_visuals()

    @mutates_tmesh
    def watertighten(self, aggressiveness=7):
        """Replace `self._mesh` with a closed manifold wrap of the current geometry.

        `aggressiveness` is the integer 0..8 controlling the wrap's quadric-error decimation cost cutoff; see
        `genesis.utils.watertighten.watertighten_mesh` for the full pipeline.
        """
        if self._mesh.vertices.shape[0] <= 3:
            return
        from genesis.utils.watertighten import watertighten_mesh

        v, f = watertighten_mesh(
            np.asarray(self._mesh.vertices),
            np.asarray(self._mesh.faces, dtype=np.int32),
            aggressiveness=aggressiveness,
        )
        self._mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        self._metadata["watertightened"] = True
        self.clear_visuals()

    @mutates_tmesh
    def decimate(self, decimate_face_num, decimate_aggressiveness):
        """
        Decimate the mesh.
        """
        if self._mesh.vertices.shape[0] > 3 and len(self._mesh.faces) > decimate_face_num:
            self._mesh.process(validate=True)
            self._mesh = trimesh.Trimesh(
                *fast_simplification.simplify(
                    self._mesh.vertices,
                    self._mesh.faces,
                    target_count=decimate_face_num,
                    agg=decimate_aggressiveness,
                    lossless=(decimate_aggressiveness == 0),
                ),
            )
            self._metadata["decimated"] = True

        self.clear_visuals()

    @mutates_tmesh
    def remesh(self, edge_len_abs=None, edge_len_ratio=0.01, fix=True):
        """
        Remesh for tetrahedralization.
        """
        cache = get_remesh_cache(self.verts, self.faces, edge_len_abs, edge_len_ratio, fix)

        remeshed = cache.load()
        if remeshed is not None:
            gs.logger.debug("Remeshed file (`.rm`) found in cache.")
        else:
            # Importing pymeshlab is very slow and not used very often. Let's delay import.
            with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
                import pymeshlab

            gs.logger.info("Remeshing for tetrahedralization...")
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=self.verts, face_matrix=self.faces))
            if edge_len_abs is not None:
                ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(edge_len_abs))
            else:
                ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(edge_len_ratio * 100))
            m = ms.current_mesh()
            remeshed = (m.vertex_matrix(), m.face_matrix())
            # Maybe we need to fix the mesh in some extreme cases with open3d
            # if fix:
            #     verts, faces = pymeshfix.clean_from_arrays(verts, faces)
            cache.save(remeshed)

        verts, faces = remeshed
        self._mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        self.clear_visuals()

    def tetrahedralize(self, tet_cfg):
        """
        Tetrahedralize the mesh.
        """
        return mu.tetrahedralize_mesh(self._mesh, tet_cfg)

    def particlize(
        self,
        p_size=0.01,
        sampler="random",
    ):
        """
        Sample particles using the mesh volume.
        """
        if "pbs" in sampler:
            return pu.trimesh_to_particles_pbs(self._mesh, p_size, sampler)
        return pu.trimesh_to_particles_simple(self._mesh, p_size, sampler)

    def sample_point_cloud(
        self,
        n_points: int,
        *,
        n_candidates: int | None = None,
        seed: int | None = None,
        use_cache: bool = True,
        return_normals: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Sample `n_points` from mesh in local coordinates using Furthest Point Sampling.

        If ``return_normals`` is True, returns ``(positions, normals)`` with face normals aligned to each sample.
        """
        return pc.sample_mesh_point_cloud(
            self.verts,
            self.faces,
            n_points,
            n_candidates=n_candidates,
            seed=seed,
            use_cache=use_cache,
            return_normals=return_normals,
        )

    def clear_visuals(self):
        """
        Clear the mesh's visual attributes by resetting the surface to gs.surfaces.Default().
        """
        self._surface = gs.surfaces.Default()
        self._surface.update_texture()

    @property
    def tmesh_hash(self) -> str:
        """
        Hash of the vertices and faces of the mesh, identifying its trimesh.

        The meshes of one trimesh share the data derived from it alone, which is computed once for all of them and
        kept as long as any of them lives (see 'OwnedCache').
        """
        if self._tmesh_hash is None:
            self._tmesh_hash = get_hashkey(self.verts, self.faces)
        return self._tmesh_hash

    @cached_by_tmesh
    def get_unique_edges(self):
        """
        Get the unique edges of the mesh.
        """
        # The key of an edge packs its two vertex indices, smaller first, into one integer, which sorts and compares
        # millions of edges much faster than rows of vertex pairs.
        n_verts = len(self.verts)
        faces_rolled = np.roll(self.faces, 1, axis=1)
        edges_key = np.minimum(self.faces, faces_rolled) * n_verts + np.maximum(self.faces, faces_rolled)
        edges_key = np.sort(edges_key, axis=None)
        edges_key = edges_key[np.diff(edges_key, prepend=-1) != 0]
        edges = np.stack(np.divmod(edges_key, n_verts), axis=-1)
        return edges[edges[:, 0] != edges[:, 1]]

    @property
    @cached_by_tmesh
    def inertial(self):
        """
        Mass, center of mass and inertia tensor of the geometry in its own frame, at unit density.

        Non-watertight geometry is closed by its convex hull first so the volume integral is well-defined. A degenerate
        geometry has no mass, and composes as a geom of no mass at the origin. The result is shared by reference across
        the meshes of one trimesh.
        """
        # A degenerate geometry carries no inertia: one enclosing no volume has no center of mass, and the convex hull
        # closing a non-watertight one raises when it has fewer than 4 non-coplanar vertices.
        try:
            tmesh = trimesh.Trimesh(self.verts, self.faces, process=False)
            # The watertightness of the mesh is cached and counted on edge keys (see 'is_watertight'), while this fresh
            # trimesh would group the rows of all its edges again.
            if not self.is_watertight:
                gs.logger.warning(
                    "Mesh is not watertight. Falling back to convex hull for estimating inertial properties."
                )
                tmesh = mu.convex_hull(tmesh)
            volume = mu.get_signed_volume(tmesh.vertices, tmesh.faces)
        except QhullError:
            volume = 0.0
        if volume < 0.0:
            # Inward-facing winding gives a negative volume and inverted mass properties (e.g. a closed terrain block,
            # which bypasses watertighten since it is already watertight). Flip the mesh so that the inertia integral
            # reflects the actual solid.
            tmesh = tmesh.copy()
            tmesh.invert()
            volume = -volume
        if volume > 0.0:
            return InertialProperties(tmesh.mass, tmesh.center_mass, tmesh.moment_inertia)
        return InertialProperties(0.0, np.zeros(3), np.zeros((3, 3)))

    @cached_by_tmesh
    def get_sdf_grid(self, sdf: "SignedDistanceFieldDescription") -> SDFGrid:
        """Return the signed distance field (SDF) of the geometry, sampled on a grid centered on its bounding box.

        'sdf' describes the grid. It carries the gradient of the field if the description asks for it, and zeros
        otherwise. The signed distance is measured against the mesh 'sdf_mesh' of the metadata when it holds one.

        The grid is loaded from the preprocessing cache on disk when it holds it, and is computed and stored there
        otherwise. It is shared by reference across the meshes of one trimesh (see 'tmesh_hash').
        """
        sdf_tmesh = self._metadata.get("sdf_mesh", self._mesh)
        mesh_descr = f"({self._metadata['mesh_path']})" if "mesh_path" in self._metadata else ""
        return _get_sdf_grid(self.verts, self.faces, sdf_tmesh, sdf, mesh_descr)

    @cached_by_tmesh
    def get_support_table(self, support_res: int) -> SupportTable:
        """Return the support table of the geometry, over a grid of directions of a given resolution.

        'support_res' is the number of cells of each spherical chart of the grid along its azimuth. The table is shared
        by reference across the meshes of one trimesh (see 'tmesh_hash').
        """
        return _get_support_table(self.verts, support_res)

    @cached_by_tmesh
    def get_vert_adjacency(self):
        """
        Get the per-vertex adjacency graph as flat arrays (vert_neighbors, vert_n_neighbors, vert_neighbor_start).

        vert_neighbors concatenates each vertex's neighbor indices, vert_n_neighbors holds the neighbor count of each
        vertex, and vert_neighbor_start the offset of each vertex's slice into vert_neighbors.
        """
        # The unique edges are sorted with their smaller vertex first, so listing them from their larger vertex then
        # their smaller one and grouping by vertex with a stable sort gives the neighbors of each vertex in order.
        edges = self.get_unique_edges()
        verts_idx = np.concatenate((edges[:, 1], edges[:, 0]))
        vert_neighbors = np.concatenate((edges[:, 0], edges[:, 1]))[np.argsort(verts_idx, kind="stable")]
        vert_n_neighbors = np.bincount(verts_idx, minlength=len(self.verts))
        vert_neighbor_start = np.concatenate(((0,), np.cumsum(vert_n_neighbors[:-1])), dtype=gs.np_int)
        return vert_neighbors.astype(gs.np_int), vert_n_neighbors.astype(gs.np_int), vert_neighbor_start

    def copy(self):
        """
        Copy the mesh.
        """
        return Mesh(
            mesh=self._mesh.copy(**(dict(include_cache=True) if isinstance(self._mesh, trimesh.Trimesh) else {})),
            surface=self._surface.model_copy(),
            uvs=self._uvs.copy() if self._uvs is not None else None,
            metadata=self._metadata.copy(),
        )

    @classmethod
    def from_trimesh(
        cls,
        mesh,
        scale=None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=2,
        metadata=None,
        surface=None,
        is_mesh_zup=True,
        share_arrays=False,
    ):
        """
        Create a genesis.Mesh from a trimesh.Trimesh object.

        The mesh holds a copy of the trimesh, unless 'share_arrays' asks for its arrays to be wrapped in a trimesh of
        its own, so that the meshes built from one trimesh share its geometry. Sharing requires that nothing writes into
        those arrays in place, which holds for a trimesh that Genesis alone holds, transforms replacing the arrays they
        change.
        """
        if surface is None:
            surface = gs.surfaces.Default()
            surface.update_texture()
        else:
            surface = surface.model_copy()

        # Only the normals cached by the given trimesh are carried over, read from its cache since trimesh offers no
        # public query for them. Normals keep apart the vertices they distinguish when merging, so computing the
        # missing ones here would prevent the welding of vertex soups later on.
        if share_arrays:
            assert isinstance(mesh, trimesh.Trimesh)
            tmesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                face_normals=mesh._cache["face_normals"],
                vertex_normals=mesh._cache["vertex_normals"],
                process=False,
            )
        else:
            tmesh = mesh.copy(**(dict(include_cache=True) if isinstance(mesh, trimesh.Trimesh) else {}))

        # Always parse uvs if available because roughness and normal map also need uvs.
        # Note that some visual may not have uv, e.g. ColorVisuals.
        uvs = None
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and mesh.visual.uv is not None:
            # Note that 'trimesh' uses uvs starting from top left corner.
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]

        metadata = metadata or {}
        must_update_surface = True
        roughness_factor = None
        color_image = None
        color_factor = None
        opacity = 1.0

        visual = mesh.visual
        if isinstance(visual, trimesh.visual.texture.TextureVisuals) and visual.defined:
            if visual.kind == "texture":
                material = visual.material

                # TODO: Parsing PBR in obj or not
                # trimesh from .obj file will never use PBR material, but that from .glb file will
                if isinstance(material, trimesh.visual.material.PBRMaterial):
                    if material.baseColorTexture is not None:
                        color_image = mu.PIL_to_array(material.baseColorTexture)
                    if material.baseColorFactor is not None:
                        color_factor = tuple(np.array(material.baseColorFactor, dtype=np.float32) / 255.0)

                    if material.roughnessFactor is not None:
                        roughness_factor = (material.roughnessFactor,)

                elif isinstance(material, trimesh.visual.material.SimpleMaterial):
                    if material.image is not None:
                        color_image = mu.PIL_to_array(material.image)
                    elif material.diffuse is not None:
                        color_factor = tuple(np.array(material.diffuse, dtype=np.float32) / 255.0)

                    if material.glossiness is not None:
                        roughness_factor = (mu.glossiness_to_roughness(material.glossiness),)

                    opacity = float(material.kwargs.get("d", [1.0])[0])
                    if opacity < 1.0:
                        if color_factor is None:
                            color_factor = (1.0, 1.0, 1.0, opacity)
                        else:
                            color_factor = (*color_factor[:3], color_factor[3] * opacity)
                else:
                    gs.raise_exception(f"Unsupported Trimesh material type '{type(material)}'.")
            else:
                # TODO: support vertex/face colors in luisa
                color_factor = tuple(np.array(visual.main_color, dtype=np.float32) / 255.0)
        elif isinstance(surface.texture, gs.textures.ColorTexture):
            color_factor = surface.texture.color
        elif (isinstance(visual, trimesh.visual.color.ColorVisuals) and visual.defined) or (
            isinstance(visual, trimesh.visual.color.VertexColor) and visual.vertex_colors.size > 0
        ):
            # Color is already vertex-based. It is not only necessary to create a new visual.
            must_update_surface = False
        else:
            # use white color as default
            color_factor = (1.0, 1.0, 1.0, 1.0)

        if must_update_surface:
            metadata["is_visual_overwritten"] = isinstance(surface.texture, gs.textures.ColorTexture)

            color_texture = mu.create_texture(color_image, color_factor, "srgb")
            opacity_texture = None
            if color_texture is not None:
                opacity_texture = color_texture.check_dim(3)
            roughness_texture = mu.create_texture(None, roughness_factor, "linear")

            surface.update_texture(
                color_texture=color_texture,
                opacity_texture=opacity_texture,
                roughness_texture=roughness_texture,
            )
            tmesh.visual = mu.surface_uvs_to_trimesh_visual(surface, uvs, len(mesh.vertices))
        elif share_arrays:
            tmesh.visual = mesh.visual.copy()

        return cls(
            mesh=tmesh,
            surface=surface,
            uvs=uvs,
            scale=scale,
            convexify=convexify,
            decimate=decimate,
            decimate_face_num=decimate_face_num,
            decimate_aggressiveness=decimate_aggressiveness,
            metadata=metadata,
            is_mesh_zup=is_mesh_zup,
        )

    @classmethod
    def from_attrs(
        cls, verts, faces, normals=None, surface=None, uvs=None, scale=None, metadata=None, is_mesh_zup=True
    ):
        """
        Create a genesis.Mesh from mesh attributes including vertices, faces, and normals.
        """
        if surface is None:
            surface = gs.surfaces.Default()

        metadata = metadata or {}
        metadata["is_visual_overwritten"] = metadata.get("is_visual_overwritten") or (surface.texture is not None)
        visual = mu.surface_uvs_to_trimesh_visual(surface, uvs, len(verts))

        tmesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            visual=visual,
            process=False,
        )

        return cls(
            mesh=tmesh,
            surface=surface,
            uvs=uvs,
            scale=scale,
            metadata=metadata,
            is_mesh_zup=is_mesh_zup,
        )

    @classmethod
    def from_morph_surface(cls, morph, surface=None) -> "list[gs.Mesh]":
        """
        Create genesis.Mesh objects from morph and surface options.

        A list is always returned: a Mesh morph (morphs.Mesh) may contain multiple sub-meshes, while primitive
        morphs yield a single mesh.
        """
        if isinstance(morph, gs.options.morphs.Mesh):
            if morph.is_format(MESH_FORMATS):
                if morph.is_format(GLTF_FORMATS):
                    meshes = gltf_utils.parse_mesh_glb(
                        morph.file, morph.group_by_material, morph.scale, morph.file_meshes_are_zup, surface
                    )
                else:
                    meshes = mu.parse_mesh_trimesh(
                        morph.file, morph.group_by_material, morph.scale, morph.file_meshes_are_zup, surface
                    )
            elif isinstance(morph, gs.options.morphs.MeshSet):
                assert all(isinstance(mesh, trimesh.Trimesh) for mesh in morph.files)
                meshes = [mu.trimesh_to_mesh(mesh, morph.scale, surface) for mesh in morph.files]
            else:
                gs.raise_exception(f"File type not supported: {morph.file}")

            return meshes

        if isinstance(morph, gs.options.morphs.Box):
            tmesh = mu.create_box(extents=morph.size)
        elif isinstance(morph, gs.options.morphs.Cylinder):
            tmesh = mu.create_cylinder(radius=morph.radius, height=morph.height)
        elif isinstance(morph, gs.options.morphs.Sphere):
            tmesh = mu.create_sphere(radius=morph.radius)
        else:
            gs.raise_exception(f"Morph {morph} not supported by this method.")

        return [cls.from_trimesh(tmesh, surface=surface)]

    def set_color(self, color):
        """
        Set the mesh's color.
        """
        self._color = color
        color_texture = gs.textures.ColorTexture(color=tuple(color))
        opacity_texture = color_texture.check_dim(3)
        self._surface.update_texture(color_texture=color_texture, opacity_texture=opacity_texture, force=True)
        self.update_trimesh_visual()

    def update_trimesh_visual(self):
        """
        Update the trimesh obj's visual attributes using its surface and uvs.
        """
        self._mesh.visual = mu.surface_uvs_to_trimesh_visual(self.surface, self.uvs, len(self.verts))
        self._metadata["is_visual_overwritten"] = True

    @mutates_tmesh
    def apply_transform(self, T):
        """
        Apply a 4x4 transformation matrix (translation on the right column) to the mesh.
        """
        self._mesh.apply_transform(T)

    @property
    def uid(self):
        """
        Return the mesh's uid.
        """
        return self._uid

    @property
    def trimesh(self):
        """
        Return the mesh's trimesh object.
        """
        return self._mesh

    @property
    def is_convex(self) -> bool:
        """
        Whether the mesh is convex.
        """
        # The flag of a mesh replaced by its convex hull spares the test. It describes that mesh alone, so it stays out
        # of the trimesh cache, which only holds the test shared by the meshes of one trimesh.
        if "convexified" in self.metadata:
            return self.metadata["convexified"]
        key = ("is_convex", self.tmesh_hash)
        is_convex = _TMESH_CACHE.find(key)
        if is_convex is None:
            # trimesh deems a mesh that is not watertight non-convex. Testing that first spares its own test, which
            # builds the face adjacency of the whole mesh, on such a mesh, a terrain surface among them.
            is_convex = self.is_watertight and trimesh.Trimesh(self.verts, self.faces, process=False).is_convex
        return _TMESH_CACHE.own(self, key, is_convex)

    @property
    @cached_by_tmesh
    def is_watertight(self) -> bool:
        """
        Whether the mesh is a closed manifold surface.
        """
        # Every edge must be shared by exactly two faces, counted on the sorted edge keys (see 'get_unique_edges').
        # Bracketing the keys by values out of their range makes each group of equal keys start at a change, and leaves
        # a mesh without faces watertight, as trimesh has it.
        n_verts = len(self.verts)
        faces_rolled = np.roll(self.faces, 1, axis=1)
        edges_key = np.minimum(self.faces, faces_rolled) * n_verts + np.maximum(self.faces, faces_rolled)
        edges_key = np.sort(edges_key, axis=None)
        is_edge_start = np.diff(edges_key, prepend=-1, append=n_verts * n_verts) != 0
        return (np.diff(np.flatnonzero(is_edge_start)) == 2).all()

    @property
    def metadata(self):
        """
        Metadata of the mesh.
        """
        return self._metadata

    @property
    def verts(self):
        """
        Vertices of the mesh.
        """
        return self._mesh.vertices

    @verts.setter
    @mutates_tmesh
    def verts(self, verts):
        """
        Set the vertices of the mesh.
        """
        assert len(verts) == len(self.verts)
        self._mesh.vertices = verts

    @property
    def faces(self):
        """
        Faces of the mesh.
        """
        return self._mesh.faces

    @property
    def normals(self):
        """
        Normals of the mesh.
        """
        return self._mesh.vertex_normals

    @property
    def color(self):
        """
        Color of the mesh, as red, green, blue and alpha.
        """
        return self._color

    @property
    def surface(self):
        """
        Surface of the mesh.
        """
        return self._surface

    @property
    def uvs(self):
        """
        UVs of the mesh.
        """
        return self._uvs

    @property
    def area(self):
        """
        Surface area of the mesh.
        """
        return self._mesh.area

    @property
    def volume(self):
        """
        Volume of the mesh.
        """
        return mu.get_signed_volume(self._mesh.vertices, self._mesh.faces)

    def export(self, exporting: serialization.Exporting) -> dict:
        """Export the geometry, uvs, surface and metadata a mesh holds.

        A mesh is written this way rather than through its fields because construction convexifies, decimates and
        rescales it: what a file carries is the geometry as it now stands, so reading one back processes nothing.
        """
        return {
            "geometry": _exported_geometry(self.trimesh, exporting),
            "uvs": None if self.uvs is None else exporting.array(self.uvs),
            "surface": exporting.value(self.surface, Surface),
            "metadata": exporting.value(self.metadata, Any),
        }

    @classmethod
    def load(cls, raw: dict, loading: serialization.Loading) -> "Mesh":
        """Recreate the mesh exactly as exported, processing none of its geometry again.

        Meshes written from one trimesh share the data derived from it, as the meshes of one asset do (see
        'tmesh_hash'). Each keeps its own trimesh and surface, so a mesh drawn in its own colour still costs one copy of
        the geometry.
        """
        mesh = cls(
            mesh=_loaded_geometry(raw["geometry"], loading),
            surface=loading.value(raw["surface"], Surface),
            uvs=None if raw["uvs"] is None else loading.array(raw["uvs"]),
            # The metadata says what was already done to the geometry, so it is handed back rather than acted on again
            metadata=loading.value(raw["metadata"], Any),
        )
        # Construction derives the visual from the surface and the uvs unless the mesh states vertex colours, which are
        # the only visual a file carries (see '_exported_geometry'), so a textured mesh gets its visual back this way.
        if raw["geometry"]["colours"] is None:
            mesh._mesh.visual = mu.surface_uvs_to_trimesh_visual(mesh.surface, mesh.uvs, len(mesh.verts))
        return mesh


def _get_sdf_grid(
    verts: np.ndarray,
    faces: np.ndarray,
    sdf_tmesh: trimesh.Trimesh,
    sdf: "SignedDistanceFieldDescription",
    mesh_descr: str,
) -> SDFGrid:
    """Load the signed distance field (SDF) grid of a triangle mesh from the cache on disk, or compute and store it.

    The grid is centered on the bounding box of 'verts', described by 'sdf', and measures the signed distance to
    'sdf_tmesh' (see 'Mesh.get_sdf_grid'). 'mesh_descr' names the mesh in the warnings.
    """
    n_nodes, cell_size = np.array(sdf.n_nodes), np.array(sdf.cell_size)
    gsd_path = get_gsd_path(verts, faces, n_nodes, cell_size)
    sdf_grid = None
    if os.path.exists(gsd_path):
        gs.logger.debug(f"Preprocessed file (`.gsd`) found in cache: {gsd_path}.")
        try:
            with open(gsd_path, "rb") as file:
                sdf_grid = pickle.load(file)
        except (EOFError, ModuleNotFoundError, pickle.UnpicklingError, TypeError, MemoryError):
            gs.logger.info("Ignoring corrupted cache.")

    if sdf_grid is None:
        sdf_verts, sdf_faces = np.ascontiguousarray(sdf_tmesh.vertices), np.ascontiguousarray(sdf_tmesh.faces)
        if len(sdf_faces) > 50000:
            gs.logger.warning(
                f"Beware that SDF pre-processing of mesh {mesh_descr} having more than 50000 faces may take a "
                "very long time (>10min) and require large RAM allocation (>20Gb). Please either enable convexify "
                "or decimation. (see FileMorph options)"
            )
        with gs.logger.timer(f"Preprocessing SDF of mesh ~<{gsd_path}>~."):
            lower = verts.min(axis=0)
            upper = verts.max(axis=0)
            center = (upper + lower) / 2.0

            # The grid spans 'res - 1' cells from the center of its lower node to the center of its upper one
            grid_size = (n_nodes - 1) * cell_size
            voxel_lower = center - grid_size / 2.0
            voxel_upper = center + grid_size / 2.0
            X, Y, Z = np.meshgrid(
                *(np.linspace(voxel_lower[i], voxel_upper[i], n_nodes[i]) for i in range(3)), indexing="ij"
            )
            query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

            signed_distance, sdf_closest_faces, closest_points, _ = igl.signed_distance(
                query_points, sdf_verts, sdf_faces
            )
            sdf_value = signed_distance.astype(gs.np_float, copy=False).reshape(n_nodes)

            # The closest vertex belongs to the mesh itself, whose closest faces are those of the signed distance when
            # it is measured against that mesh
            closest_faces = sdf_closest_faces
            if not (np.array_equal(sdf_verts, verts) and np.array_equal(sdf_faces, faces)):
                _, closest_faces, *_ = igl.signed_distance(query_points, verts, faces)
            verts_idx = faces[closest_faces]
            verts_dist = np.linalg.norm(verts[verts_idx] - query_points[:, None, :], axis=-1)
            sdf_closest_verts_idx = verts_idx[np.arange(len(query_points)), np.argmin(verts_dist, axis=-1)]

            T_mesh_to_centered = np.eye(4)
            T_mesh_to_centered[:3, 3] = -center
            T_centered_to_sdf = np.eye(4)
            T_centered_to_sdf[:3, :3] *= (n_nodes - 1) / grid_size
            T_centered_to_sdf[:3, 3] = (n_nodes - 1) / 2
            T_mesh_to_sdf = T_centered_to_sdf @ T_mesh_to_centered

            # The gradient of a signed distance is the unit vector from the closest point of the surface, oriented by
            # the side of the node, and the normal of the closest face on the surface itself. On the medial axis, where
            # the field has a kink, it is the gradient on the side of the closest face found.
            sdf_grad = np.zeros((*n_nodes, 3), dtype=gs.np_float)
            if sdf.has_grad:
                offset = query_points - closest_points
                dist = np.linalg.norm(offset, axis=-1, keepdims=True)
                grad = np.where(
                    dist > gs.EPS,
                    np.sign(signed_distance)[:, None] * offset / np.maximum(dist, gs.EPS),
                    sdf_tmesh.face_normals[sdf_closest_faces],
                )
                sdf_grad[:] = grad.reshape((*n_nodes, 3))

        sdf_grid = SDFGrid(
            sdf_value, sdf_grad, sdf_closest_verts_idx.reshape(n_nodes), np.max(sdf_value), T_mesh_to_sdf
        )
        os.makedirs(os.path.dirname(gsd_path), exist_ok=True)
        with open(gsd_path, "wb") as file:
            pickle.dump(sdf_grid, file)

    return sdf_grid


def _get_support_table(verts: np.ndarray, support_res: int) -> SupportTable:
    """Compute the support table of the convex hull of a set of vertices over the cells of a grid of directions.

    The grid is made of two spherical charts of 'support_res' cells along their azimuth, each one covering the band
    of directions 45 to 135 degrees off its own poles (see 'SupportField').
    """
    band_start, band_rows = support_res // 4, support_res // 2
    n_geom_cells = 2 * support_res * band_rows

    # Unit direction in the mesh frame of every node bounding the cells of the charts, as [chart, azimuth, polar]
    theta = np.arange(support_res) / support_res * 2 * math.pi - math.pi
    phi = np.arange(band_start, band_start + band_rows + 1) / support_res * math.pi
    nodes_chart = np.stack(
        np.broadcast_arrays(np.sin(phi) * np.cos(theta[:, None]), np.sin(phi) * np.sin(theta[:, None]), np.cos(phi)),
        axis=-1,
    )
    nodes = nodes_chart @ CHARTS_ROT[:, None]

    # Four corner nodes of each cell
    cells_chart, cells_i, cells_j, corners_di, corners_dj = np.meshgrid(
        np.arange(2), np.arange(support_res), np.arange(band_rows), (0, 1), (0, 1), indexing="ij"
    )
    cells_corner = np.ravel_multi_index(
        (cells_chart, (cells_i + corners_di) % support_res, cells_j + corners_dj), nodes.shape[:3]
    ).reshape((n_geom_cells, 4))

    # Planes of the meridians and polar angles of the parallels bounding the cells of a chart, in its own frame
    meridians_theta = theta[: support_res // 2]
    meridians_normal = np.stack(
        (-np.sin(meridians_theta), np.cos(meridians_theta), np.zeros_like(meridians_theta)), axis=-1
    )
    parallels_z = np.cos(phi)
    # Corners of the closure of a cell around a direction, widened by a margin in cell units
    cells_corner_margin = 1e-6 * np.array(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)))

    # Normal cones of the hull vertices, as the faces and edges of their union: the normals of the hull faces, each
    # supported by the vertices of its face, and the arcs between the normals of adjacent hull faces, along which both
    # vertices of their shared edge tie. A flat geom joins its two opposite face normals through the outward normal of
    # each outline edge by two quarter arcs. A segment or a point ties both of its ends along the great circle
    # orthogonal to it, cut in four quarter arcs.
    verts_centered = verts - verts.mean(axis=0)
    verts_rank = np.linalg.matrix_rank(verts_centered)
    # Principal axes of the vertices, from the eigenvectors of their 3x3 second moment, of fixed size
    _, _, frame = np.linalg.svd(verts_centered.T @ verts_centered)
    if verts_rank == 3:
        hull = ConvexHull(verts_centered)
        hull_vid = hull.vertices
        faces_idx, i_vert = np.nonzero(hull.neighbors > np.arange(len(hull.simplices))[:, None])
        faces_normal = hull.equations[:, :3]
        faces_direction = np.repeat(faces_normal, repeats=3, axis=0)
        faces_vid = hull.simplices.reshape(-1)
        arcs_start = faces_normal[faces_idx]
        arcs_end = faces_normal[hull.neighbors[faces_idx, i_vert]]
        # The neighbour of a face across its i-th vertex shares the edge made of its two other vertices
        arcs_vid = np.stack(
            (hull.simplices[faces_idx, (i_vert + 1) % 3], hull.simplices[faces_idx, (i_vert + 2) % 3]),
            axis=-1,
        )
    elif verts_rank == 2:
        outline = ConvexHull(verts_centered @ frame[:2].T)
        hull_vid = outline.vertices
        edges_normal = outline.equations[:, :2] @ frame[:2]
        faces_normal = np.stack((frame[2], -frame[2]))
        n_edges = len(edges_normal)
        faces_direction = np.repeat(faces_normal, repeats=len(hull_vid), axis=0)
        faces_vid = np.tile(hull_vid, reps=2)
        arcs_start = np.concatenate((np.tile(faces_normal[0], reps=(n_edges, 1)), edges_normal))
        arcs_end = np.concatenate((edges_normal, np.tile(faces_normal[1], reps=(n_edges, 1))))
        arcs_vid = np.tile(outline.simplices, reps=(2, 1))
    else:
        verts_axial = verts_centered @ frame[0]
        hull_vid = np.array((np.argmin(verts_axial), np.argmax(verts_axial)))
        faces_direction = np.zeros((0, 3))
        faces_vid = np.zeros((0,), dtype=gs.np_int)
        arcs_start = np.stack((frame[1], frame[2], -frame[1], -frame[2]))
        arcs_end = np.roll(arcs_start, shift=-1, axis=0)
        arcs_vid = np.tile(hull_vid, reps=(4, 1))

    # Support of every node, searched among the hull vertices alone, by chunks of nodes whose dot products take a
    # bounded memory whatever the vertex count
    nodes_chunks = np.array_split(nodes.reshape((-1, 3)), max(nodes[..., 0].size * len(hull_vid) // 2**24, 1))
    nodes_vid = np.concatenate(
        [hull_vid[np.argmax(nodes_chunk @ verts[hull_vid].T, axis=-1)] for nodes_chunk in nodes_chunks]
    )

    # Every arc is cut where it crosses the meridians and parallels of a chart, so that each piece lies in a single
    # cell, the cell of its midpoint. A crossing outside of the arc is clamped to one of its ends. The cuts lie on the
    # boundaries of the cells, which an arc tangent to a parallel touches without crossing, so they are listed along
    # with the midpoints.
    arcs_cos = np.clip(np.sum(arcs_start * arcs_end, axis=-1), -1.0, 1.0)
    arcs_angle = np.arccos(arcs_cos)
    arcs_orthogonal = arcs_end - arcs_cos[:, None] * arcs_start
    arcs_orthogonal_norm = np.linalg.norm(arcs_orthogonal, axis=-1, keepdims=True)
    arcs_orthogonal = np.divide(
        arcs_orthogonal,
        arcs_orthogonal_norm,
        out=np.zeros_like(arcs_orthogonal),
        where=arcs_orthogonal_norm > 0,
    )

    # The corners of a cell mostly share their support, which is listed once
    corners_vid = nodes_vid[cells_corner]
    is_repeat = np.tril(corners_vid[:, :, None] == corners_vid[:, None, :], k=-1).any(axis=-1)
    corners_cell, i_corner = np.nonzero(~is_repeat)
    pairs = [np.stack((corners_cell, corners_vid[corners_cell, i_corner]), axis=-1)]
    for i_chart in range(2):
        arcs_start_chart = arcs_start @ CHARTS_ROT[i_chart].T
        arcs_orthogonal_chart = arcs_orthogonal @ CHARTS_ROT[i_chart].T
        arcs_meridian = np.mod(
            np.arctan2(-arcs_start_chart @ meridians_normal.T, arcs_orthogonal_chart @ meridians_normal.T),
            math.pi,
        )
        arcs_amplitude = np.hypot(arcs_start_chart[:, 2], arcs_orthogonal_chart[:, 2])
        arcs_phase = np.arctan2(arcs_orthogonal_chart[:, 2], arcs_start_chart[:, 2])
        arcs_ratio = parallels_z / np.maximum(arcs_amplitude, gs.EPS)[:, None]
        # A parallel the great circle of an arc misses, or touches up to rounding, cuts it at its extreme point
        arcs_half = np.arccos(np.clip(arcs_ratio, -1.0, 1.0))
        arcs_cuts = np.concatenate(
            (
                np.zeros_like(arcs_angle)[:, None],
                arcs_meridian,
                np.mod(arcs_phase[:, None] + arcs_half, 2 * math.pi),
                np.mod(arcs_phase[:, None] - arcs_half, 2 * math.pi),
                arcs_angle[:, None],
            ),
            axis=-1,
        )
        arcs_cuts = np.sort(np.minimum(arcs_cuts, arcs_angle[:, None]), axis=-1)
        arcs_idx, i_cut = np.nonzero(arcs_cuts[:, 1:] > arcs_cuts[:, :-1])
        # The midpoint and the start of every piece, the starts covering every distinct cut but the end of the arc, a
        # face normal listed with its face
        points_angle = np.concatenate(
            (
                0.5 * (arcs_cuts[arcs_idx, i_cut] + arcs_cuts[arcs_idx, i_cut + 1]),
                arcs_cuts[arcs_idx, i_cut],
            )
        )
        points_arc_idx = np.concatenate((arcs_idx, arcs_idx))
        points_direction = (
            np.cos(points_angle)[:, None] * arcs_start[points_arc_idx]
            + np.sin(points_angle)[:, None] * arcs_orthogonal[points_arc_idx]
        )

        # Every face normal lists the vertices of its face, and every point of arc both ends of its hull edge, in the
        # cells whose closure holds it. A direction on a cell boundary belongs to every cell around it, since the
        # vertices it supports may support directions of any of them. The closure is widened by a margin covering the
        # rounding of the cell coordinates, which lists a vertex in one cell more at most.
        for directions, directions_vid in (
            (faces_direction, faces_vid[:, None]),
            (points_direction, arcs_vid[points_arc_idx]),
        ):
            directions_chart = directions @ CHARTS_ROT[i_chart].T
            theta_chart = np.arctan2(directions_chart[:, 1], directions_chart[:, 0])
            phi_chart = np.arccos(np.clip(directions_chart[:, 2], -1.0, 1.0))
            cells_coord = np.stack(
                (
                    (theta_chart + math.pi) / (2 * math.pi) * support_res,
                    phi_chart / math.pi * support_res,
                ),
                axis=-1,
            )
            cells_ij = np.floor(cells_coord[:, None] + cells_corner_margin)
            cells_i = cells_ij[..., 0].astype(gs.np_int) % support_res
            cells_j = cells_ij[..., 1].astype(gs.np_int) - band_start
            is_in_band = (cells_j >= 0) & (cells_j < band_rows)
            directions_idx, _ = np.nonzero(is_in_band)
            cells_idx = ((i_chart * support_res + cells_i) * band_rows + cells_j)[is_in_band]
            cells_idx = np.repeat(cells_idx, repeats=directions_vid.shape[-1])
            pairs.append(np.stack((cells_idx, directions_vid[directions_idx].reshape(-1)), axis=-1))

    # Each pair is encoded as one integer key, cells major, so that a flat sort removes the repeated ones
    pairs = np.concatenate(pairs)
    cells_idx, verts_idx = np.divmod(np.unique(pairs[:, 0] * len(verts) + pairs[:, 1]), len(verts))
    # Every cell lists the supports of its corners, so the candidates of each cell come in a row, in cell order. Records
    # are laid out as described in SupportFieldInfo.
    candidates_idx = np.arange(len(cells_idx))
    is_first = np.concatenate(((True,), cells_idx[1:] != cells_idx[:-1]))
    candidates_rank = candidates_idx - np.maximum.accumulate(np.where(is_first, candidates_idx, 0))
    slots_vid = np.repeat(verts_idx[is_first, None], repeats=2, axis=-1)
    is_slot = candidates_rank < 2
    slots_vid[cells_idx[is_slot], candidates_rank[is_slot]] = verts_idx[is_slot]
    return SupportTable(slots_vid, verts_idx[~is_slot], np.bincount(cells_idx[~is_slot], minlength=n_geom_cells))


def _exported_geometry(mesh: trimesh.Trimesh, exporting: serialization.Exporting) -> dict:
    """Export the geometry of a trimesh: its vertices, the faces joining them, their normals and their colours.

    Only colours the mesh states are exported. Asking a mesh coloured by anything else for its vertex colours makes
    trimesh invent a default one per vertex, which a file would then carry as if the author had chosen it.
    """
    colours = mesh.visual.vertex_colors if mesh.visual.kind == "vertex" else None
    return {
        "verts": exporting.array(mesh.vertices),
        "faces": exporting.array(mesh.faces),
        "normals": exporting.array(mesh.vertex_normals),
        "colours": None if colours is None else exporting.array(colours),
    }


def _loaded_geometry(raw: dict, loading: serialization.Loading) -> trimesh.Trimesh:
    """Recreate the trimesh as it was exported, with the vertices, faces and normals the file holds."""
    mesh = trimesh.Trimesh(
        vertices=loading.array(raw["verts"]),
        faces=loading.array(raw["faces"]),
        vertex_normals=loading.array(raw["normals"]),
        process=False,
    )
    if raw["colours"] is not None:
        mesh.visual.vertex_colors = loading.array(raw["colours"])
    return mesh


serialization.register(trimesh.Trimesh, _exported_geometry, _loaded_geometry)
