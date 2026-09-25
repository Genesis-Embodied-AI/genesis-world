import os
from itertools import chain
from typing import Any, NamedTuple

import fast_simplification
import numpy as np
import trimesh
from scipy.spatial import QhullError

import genesis as gs
import genesis.utils.gltf as gltf_utils
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.constants import GLTF_FORMATS, MESH_FORMATS
import genesis.utils.point_cloud as pc
from genesis.options.surfaces import Surface
from genesis.repr_base import RBC
from genesis.typing import Matrix3x3Type, Vec3FType
from genesis.utils import serialization
from genesis.utils.misc import redirect_libc_stderr


class InertialProperties(NamedTuple):
    """A rigid body's intrinsic inertial: mass, center of mass 'com', and inertia tensor 'i' about that COM."""

    mass: float
    com: Vec3FType
    i: Matrix3x3Type


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

        # Geometry-derived data (independent of appearance) computed lazily and reused. When the same processed
        # collision geometry backs many entities, these are shared by reference so each entity reads them instead of
        # recomputing the unique-edge list and the vertex-adjacency graph.
        self._unique_edges: "np.ndarray | None" = None
        self._vert_adjacency: "tuple[np.ndarray, np.ndarray, np.ndarray] | None" = None
        self._is_convex: "bool | None" = None
        self._inertial: "InertialProperties | None" = None
        # A mesh sharing another's processed geometry (a cached collision template) delegates its inertia query to that
        # source, so the (convex-hull) mass-property computation runs at most once per geometry and only when an entity
        # actually needs it - never eagerly for e.g. fixed or articulated assets whose link frames are not aligned.
        self._inertial_source: "Mesh | None" = None

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

    def convexify(self):
        """
        Convexify the mesh.
        """
        if self._mesh.vertices.shape[0] > 3:
            self._mesh = trimesh.convex.convex_hull(self._mesh)
            self._metadata["convexified"] = True
        self._invalidate_geometry_cache()
        self.clear_visuals()

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
        self._invalidate_geometry_cache()
        self.clear_visuals()

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

        self._invalidate_geometry_cache()
        self.clear_visuals()

    def remesh(self, edge_len_abs=None, edge_len_ratio=0.01, fix=True):
        """
        Remesh for tetrahedralization.
        """
        cache = mu.get_remesh_cache(self.verts, self.faces, edge_len_abs, edge_len_ratio, fix)

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
        self._invalidate_geometry_cache()
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

    def _invalidate_geometry_cache(self):
        # Drop memoized geometry-derived data (unique edges, vertex adjacency, convexity, inertia) when the underlying
        # trimesh is replaced or its vertices change, so the next query recomputes from the current geometry instead of
        # returning stale topology or inertia.
        self._unique_edges = None
        self._vert_adjacency = None
        self._is_convex = None
        self._inertial = None
        self._inertial_source = None

    def get_unique_edges(self):
        """
        Get the unique edges of the mesh.
        """
        if self._unique_edges is None:
            r_face = np.roll(self.faces, 1, axis=1)
            edges = np.concatenate(np.array([self.faces, r_face]).T)

            # do a first pass to remove duplicates
            edges.sort(axis=1)
            edges = np.unique(edges, axis=0)
            self._unique_edges = edges[edges[:, 0] != edges[:, 1]]

        return self._unique_edges

    @property
    def inertial(self):
        """
        Mass, center of mass and inertia tensor of the geometry in its own frame, at unit density.

        Non-watertight geometry is closed by its convex hull first so the volume integral is well-defined; a degenerate
        geometry has no mass, and composes as a geom of no mass at the origin. The result is memoized and shared by
        reference across entities backed by the same geometry.
        """
        if self._inertial_source is not None:
            return self._inertial_source.inertial
        if self._inertial is None:
            # A degenerate geometry (zero / ill-defined volume) makes trimesh's mass-property integral divide by zero,
            # yielding a non-finite center of mass; a more degenerate one (fewer than 4 non-coplanar vertices) makes the
            # convex-hull closure of a non-watertight mesh raise instead. Either way the geom carries no inertia:
            # report no mass, rather than crashing or letting NaNs propagate into the composed inertia.
            with np.errstate(invalid="ignore", divide="ignore"):
                try:
                    tmesh = self._mesh
                    if not self._mesh.is_watertight:
                        gs.logger.warning(
                            "Mesh is not watertight. Falling back to convex hull for estimating inertial properties."
                        )
                        tmesh = self._mesh.convex_hull
                    volume = float(tmesh.volume)
                    if volume < 0.0:
                        # Inward-facing winding gives a negative volume and inverted mass properties (e.g. a closed
                        # terrain block, which bypasses watertighten since it is already watertight). Flip the mesh so
                        # the inertia integral reflects the actual solid rather than estimating from an inverted one.
                        tmesh = tmesh.copy()
                        tmesh.invert()
                        volume = -volume
                    center_mass = tmesh.center_mass
                except QhullError:
                    volume, center_mass = 0.0, None
                if volume > 0.0 and np.all(np.isfinite(center_mass)):
                    self._inertial = InertialProperties(tmesh.mass, center_mass, tmesh.moment_inertia)
                else:
                    self._inertial = InertialProperties(0.0, np.zeros(3), np.zeros((3, 3)))
        return self._inertial

    def get_vert_adjacency(self):
        """
        Get the per-vertex adjacency graph as flat arrays (vert_neighbors, vert_n_neighbors, vert_neighbor_start).

        vert_neighbors concatenates each vertex's neighbor indices, vert_n_neighbors holds the neighbor count of each
        vertex, and vert_neighbor_start the offset of each vertex's slice into vert_neighbors.
        """
        if self._vert_adjacency is None:
            tmesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces, process=False)
            vert_neighbors_list = tmesh.vertex_neighbors
            vert_neighbors = np.array(tuple(chain.from_iterable(vert_neighbors_list)), dtype=gs.np_int)
            vert_n_neighbors = np.array(tuple(map(len, vert_neighbors_list)), dtype=gs.np_int)
            vert_neighbor_start = np.array((0, *np.cumsum(vert_n_neighbors)[:-1]), dtype=gs.np_int)
            self._vert_adjacency = (vert_neighbors, vert_n_neighbors, vert_neighbor_start)

        return self._vert_adjacency

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
    ):
        """
        Create a genesis.Mesh from a trimesh.Trimesh object.
        """
        if surface is None:
            surface = gs.surfaces.Default()
            surface.update_texture()
        else:
            surface = surface.model_copy()

        mesh = mesh.copy(**(dict(include_cache=True) if isinstance(mesh, trimesh.Trimesh) else {}))

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
            mesh.visual = mu.surface_uvs_to_trimesh_visual(surface, uvs, len(mesh.vertices))

        return cls(
            mesh=mesh,
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

    def apply_transform(self, T):
        """
        Apply a 4x4 transformation matrix (translation on the right column) to the mesh.
        """
        self._mesh.apply_transform(T)
        self._invalidate_geometry_cache()

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
        # 'dict.get' would evaluate the (expensive) convexity test eagerly even when the flag is already set, so branch
        # explicitly. The trimesh fallback is memoized to stay cheap when many entities share the same geometry.
        if "convexified" in self.metadata:
            return self.metadata["convexified"]
        if self._is_convex is None:
            self._is_convex = self._mesh.is_convex
        return self._is_convex

    @property
    def is_watertight(self) -> bool:
        """
        Whether the mesh is a closed manifold surface.
        """
        return self._mesh.is_watertight

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
    def verts(self, verts):
        """
        Set the vertices of the mesh.
        """
        assert len(verts) == len(self.verts)
        self._mesh.vertices = verts
        self._invalidate_geometry_cache()

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
        return self._mesh.volume

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

        Meshes written from one geometry are handed the edges, the vertex adjacency and the inertia of the first of
        them, exactly as 'postprocess_collision_geoms' hands them to the entities it builds from one asset. Each
        keeps its own trimesh and surface, so a mesh drawn in its own colour still costs one copy of the geometry.
        """
        mesh = cls(
            mesh=_loaded_geometry(raw["geometry"], loading),
            surface=loading.value(raw["surface"], Surface),
            uvs=None if raw["uvs"] is None else loading.array(raw["uvs"]),
            # The metadata says what was already done to the geometry, so it is handed back rather than acted on again
            metadata=loading.value(raw["metadata"], Any),
        )
        source = loading.shared.setdefault((raw["geometry"]["verts"], raw["geometry"]["faces"]), mesh)
        if source is not mesh:
            mesh._unique_edges = source.get_unique_edges()
            mesh._vert_adjacency = source.get_vert_adjacency()
            mesh._inertial_source = source
        return mesh


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
