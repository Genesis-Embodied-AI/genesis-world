import hashlib
import marshal
import math
import os
import pickle as pkl
from functools import lru_cache
from pathlib import Path

import coacd
import igl
import Imath
import numpy as np
import OpenEXR
import tetgen
import trimesh
from PIL import Image

import genesis as gs

from . import geom as gu
from .misc import (
    SizeCappedCache,
    get_assets_dir,
    get_cvx_cache_dir,
    get_exr_cache_dir,
    get_gnd_cache_dir,
    get_gsd_cache_dir,
    get_ptc_cache_dir,
    get_remesh_cache_dir,
    get_src_dir,
    get_tet_cache_dir,
    get_usd_cache_dir,
    get_wt_cache_dir,
    get_wth_cache_dir,
    register_cache_clear,
)

MESH_REPAIR_ERROR_THRESHOLD = 0.01
# Largest coordinate discrepancy, as a fraction of the bounding-box diagonal, under which two meshes are considered
# identical by the on-disk caches. Orders of magnitude below the spacing of genuinely distinct geometry, and orders of
# magnitude above the last-significant-digit noise separating two exports of the same model.
MESH_CACHE_MATCH_RTOL = 1e-6
# Vertex welding quantum: co-located vertices are authored with identical coordinates, so a tight absolute quantum
# absorbs sub-nanometer parsing noise while keeping genuinely distinct vertices separate.
VERT_WELD_QUANTIZE_FACTOR = 1e-8
Y_UP_TRANSFORM = np.asarray(  # translation on the bottom row
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
)
DEFAULT_PLANE_TEXTURE_PATH = "textures/checker.png"  # use checkerboard texture by default

# Bumped when watertighten output changes for a fixed (mesh, aggressiveness): forces a cache miss on stale entries.
WT_CACHE_VERSION = 7
# Bumped when the wall-thickness estimate changes for a fixed (mesh, quantile): forces a cache miss on stale entries.
WTH_CACHE_VERSION = 1


def color_f32_to_u8(color) -> np.ndarray:
    return np.round(np.asarray(color, dtype=np.float32) * 255.0).astype(np.uint8)


def color_u8_to_f32(color) -> np.ndarray:
    return np.asarray(color, dtype=np.uint8).astype(np.float32) / 255.0


def glossiness_to_roughness(glossiness: float) -> float:
    return (2 / (glossiness + 2)) ** (1.0 / 4.0)


def get_wall_thickness(verts: np.ndarray, faces: np.ndarray, quantile: float = 0.25) -> float:
    """Estimate a watertight mesh's characteristic wall thickness by probing its local diameter with inward rays.

    A ray is cast inward along the face normal from a stride-subsampled set of face centroids to the first opposite
    surface. The `quantile` of those hit distances, weighted by probed face area so the estimate measures surface
    rather than tessellation density (a thin wall spanned by a handful of large faces must not be outvoted by many
    small decorative facets), is returned: a low quantile approximates the thinnest wall and the median the typical
    one. The mesh must be watertight: inward rays escape through holes of an open surface, so the estimate is
    meaningless there and an exception is raised (the caller is expected to skip the probe for non-watertight meshes).

    The estimate is deliberately a scalar, not per-axis: the SDF grid it sizes does not only resolve walls along
    their normal - it certifies contact penetrations through Lipschitz cone bounds whose slack grows with the
    lattice spacing in every direction around the contact point, tangent axes included. On a closed shell each
    wall's tangent directions are other walls' normals (a mug's vertical wall lies tangent to the vertical axis
    even though the only wall facing that axis, the thick bottom, would justify coarse vertical cells), so relaxing
    any one axis to the thickness of the walls facing it measurably degrades the certified pens of every wall
    tangent to it, and no bound-side search can recover the loss (the lateral sample offset is a lattice property).
    """
    cache = get_wth_cache(verts, faces, quantile)
    wall_thickness = cache.load()
    if wall_thickness is not None:
        return wall_thickness

    mesh = trimesh.Trimesh(verts, faces, process=False)
    if not mesh.is_watertight:
        gs.raise_exception("Wall-thickness estimation requires a watertight mesh.")
    diag = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    stride = max(1, len(faces) // 1000)
    centers = mesh.triangles_center[::stride]
    normals = mesh.face_normals[::stride]
    origins = centers - 1e-3 * diag * normals
    locations, ray_idx, _ = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
    thickness = np.linalg.norm(locations - origins[ray_idx], axis=1)
    # A hit at the ray origin's own face reads as zero thickness; drop those self-hits before taking the quantile.
    is_hit_valid = thickness > 1e-4 * diag
    thickness = thickness[is_hit_valid]
    areas = mesh.area_faces[::stride][ray_idx][is_hit_valid]
    order = np.argsort(thickness)
    areas_cum = np.cumsum(areas[order])
    wall_thickness = thickness[order][np.searchsorted(areas_cum, quantile * areas_cum[-1])]

    cache.save(wall_thickness)
    return wall_thickness


class MeshInfo:
    def __init__(self):
        self.surface = None
        self.metadata = {}
        self.verts = []
        self.faces = []
        self.normals = []
        self.uvs = []
        self.n_points = 0

    def set_property(self, surface=None, metadata=None):
        self.surface = surface
        self.metadata = metadata

    def append(self, verts, faces, normals, uvs):
        faces += self.n_points
        self.verts.append(verts)
        self.faces.append(faces)
        self.normals.append(normals)
        self.uvs.append(uvs)
        self.n_points += len(verts)

    def export_mesh(self, scale: float, is_mesh_zup: bool) -> "gs.Mesh":
        uvs = None
        if self.uvs:
            for i, (uvs, verts) in enumerate(zip(self.uvs, self.verts)):
                if uvs is None:
                    self.uvs[i] = np.zeros((len(verts), 2), dtype=gs.np_float)
            uvs = np.concatenate(self.uvs, axis=0)

        verts = np.concatenate(self.verts, axis=0)
        faces = np.concatenate(self.faces, axis=0)
        normals = np.concatenate(self.normals, axis=0)

        return gs.Mesh.from_attrs(
            verts=verts,
            faces=faces,
            normals=normals,
            surface=self.surface,
            uvs=uvs,
            scale=scale,
            metadata=self.metadata,
            is_mesh_zup=is_mesh_zup,
        )


class MeshInfoGroup:
    def __init__(self):
        self.infos: dict[str, MeshInfo] = {}

    def get(self, name: str):
        first_created = False
        mesh_info = self.infos.get(name)
        if mesh_info is None:
            mesh_info = self.infos.setdefault(name, MeshInfo())
            first_created = True
        return mesh_info, first_created

    def export_meshes(self, scale, is_mesh_zup) -> "list[gs.Mesh]":
        return [mesh_info.export_mesh(scale, is_mesh_zup) for mesh_info in self.infos.values()]


def get_asset_path(file):
    return os.path.join(get_src_dir(), "assets", file)


def get_gsd_path(verts, faces, sdf_res, sdf_cell_size):
    # The grid is fully determined by the mesh plus the resolved per-axis resolution and cell size, so the key is
    # built from those rather than the material defaults: the resolution now also depends on wall thickness, so the
    # defaults no longer identify the grid. Schema tag bumped when the on-disk SDF layout changes (e.g. scalar ->
    # per-axis cell size); forces a cache miss on stale entries without manually clearing the cache.
    schema = "v3-res-keyed"
    hashkey = get_hashkey(verts, faces, sdf_res, sdf_cell_size, schema)
    return os.path.join(get_gsd_cache_dir(), f"{hashkey}.gsd")


def get_gnd_path(name, subterrain_types, subterrain_size, horizontal_scale, vertical_scale, n_subterrains):
    hashkey = get_hashkey(name, subterrain_types, subterrain_size, horizontal_scale, vertical_scale, n_subterrains)
    return os.path.join(get_gnd_cache_dir(), f"{hashkey}.gnd")


def get_cvx_cache(verts, faces, coacd_options):
    return MeshCache(get_cvx_cache_dir(), "cvx", verts, faces, coacd_options.__dict__)


def get_ptc_cache(verts, faces, p_size, sampler):
    return MeshCache(get_ptc_cache_dir(), "ptc", verts, faces, p_size, sampler)


def get_tet_cache(verts, faces, tet_cfg):
    return MeshCache(get_tet_cache_dir(), "tet", verts, faces, tet_cfg)


def get_remesh_cache(verts, faces, edge_len_abs, edge_len_ratio, fix):
    return MeshCache(get_remesh_cache_dir(), "rm", verts, faces, edge_len_abs, edge_len_ratio, fix)


def get_wt_cache(verts, faces, aggressiveness):
    return MeshCache(get_wt_cache_dir(), "wt", verts, faces, aggressiveness, WT_CACHE_VERSION)


def get_wth_cache(verts, faces, quantile):
    return MeshCache(get_wth_cache_dir(), "wth", verts, faces, quantile, WTH_CACHE_VERSION)


def get_exr_path(file_path):
    hashkey = get_hashkey(Path(file_path))
    return os.path.join(get_exr_cache_dir(), f"{hashkey}.exr")


def get_usd_zip_path(file_path):
    hashkey = get_hashkey(Path(file_path))
    return os.path.join(get_usd_cache_dir(), "zip", hashkey)


def get_usd_bake_path(file_path):
    hashkey = get_hashkey(Path(file_path))
    return os.path.join(get_usd_cache_dir(), "bake", hashkey)


def get_hashkey(*args):
    hasher = hashlib.sha256()
    for arg in (*args, gs.__version__.encode()):
        if isinstance(arg, Path):
            file_stats = arg.stat()
            arg = (str(arg).encode(), file_stats.st_size, file_stats.st_mtime)
        if isinstance(arg, str):
            arg = arg.encode()
        elif not isinstance(arg, bytes):
            try:
                arg = bytes(memoryview(arg))
            except TypeError:
                arg = marshal.dumps(arg)
        hasher.update(arg)
    return hasher.hexdigest()


class MeshCache:
    """On-disk cache of data derived from a mesh, keyed on its geometry up to the noise of a re-export.

    Exporting the same model twice from an authoring tool perturbs its coordinates in the last significant digits,
    which changes any exact hash of them and forces every derived artifact to be recomputed from scratch. Rounding
    the coordinates onto a grid before hashing does not fix this: a coordinate lying near a grid boundary changes
    bucket under an arbitrarily small perturbation, and a single such coordinate out of the tens of thousands of a
    mesh already changes the key. Entries are therefore grouped into buckets keyed on what is exactly reproducible -
    the vertex count, the face connectivity and the generation parameters - and a lookup that misses the exact vertex
    key scans its bucket, accepting the first candidate matching within 'MESH_CACHE_MATCH_RTOL'. Both arms compare
    the vertices, so a hit is always verified rather than inferred from a key.

    An entry stores its reference vertices ahead of the payload in the same file, so the scan deserializes only that
    leading record for the candidates it rejects.

    Parameters
    ----------
    cache_dir : str
        Directory holding the buckets of this kind of derived data.
    ext : str
        Extension of the cache files, identifying the kind of derived data they hold.
    verts : np.ndarray
        Vertices identifying the mesh, in the frame the derived data is expressed in. Pass them normalized whenever
        the derived data is invariant to a transform, so that meshes related by one share a single entry.
    faces : np.ndarray
        Face connectivity identifying the mesh.
    *params
        Every generation parameter the derived data depends on. Must be exactly reproducible across runs.
    """

    def __init__(self, cache_dir: str, ext: str, verts: np.ndarray, faces: np.ndarray, *params):
        self._verts = np.ascontiguousarray(verts)
        bucket_key = get_hashkey(len(self._verts), np.ascontiguousarray(faces), *params)
        self._bucket_dir = os.path.join(cache_dir, bucket_key)
        self._filename = f"{get_hashkey(self._verts)}.{ext}"

    def load(self):
        """Return the payload of the matching entry, or None if this bucket holds none."""
        if not os.path.isdir(self._bucket_dir):
            return None
        atol = MESH_CACHE_MATCH_RTOL * np.linalg.norm(self._verts.max(axis=0) - self._verts.min(axis=0))
        # Trying the exact key first keeps an unchanged asset from ever paying for the scan.
        for filename in sorted(os.listdir(self._bucket_dir), key=lambda name: name != self._filename):
            try:
                with open(os.path.join(self._bucket_dir, filename), "rb") as file:
                    # Each record needs its own unpickler: one reused across both would carry its memo from the
                    # vertices into the payload and misread it.
                    verts_ref = pkl.load(file)
                    if verts_ref.shape != self._verts.shape or (np.abs(self._verts - verts_ref) > atol).any():
                        continue
                    return pkl.load(file)
            except (EOFError, ModuleNotFoundError, pkl.UnpicklingError, TypeError, MemoryError):
                gs.logger.info("Ignoring corrupted cache.")
        return None

    def save(self, payload) -> None:
        """Store the payload as the entry of this mesh. It must not be None, which 'load' reserves for a miss."""
        assert payload is not None
        os.makedirs(self._bucket_dir, exist_ok=True)
        with open(os.path.join(self._bucket_dir, self._filename), "wb") as file:
            pkl.dump(self._verts, file, protocol=pkl.HIGHEST_PROTOCOL)
            pkl.dump(payload, file, protocol=pkl.HIGHEST_PROTOCOL)


def load_mesh(file):
    if isinstance(file, (str, Path)):
        try:
            return trimesh.load_mesh(file, force="mesh", skip_texture=False)
        except Exception as e:
            gs.logger.warning(f"Failed to load mesh with texture: {e}")
            # try loading without texture data
            return trimesh.load_mesh(file, force="mesh", skip_texture=True)
    return file


def compute_sdf_data(mesh, res):
    """
    Convert mesh to sdf voxels and a transformation matrix from mesh frame to voxel frame.
    """
    voxels_radius = 0.6
    x = np.linspace(-voxels_radius, voxels_radius, res)
    y = np.linspace(-voxels_radius, voxels_radius, res)
    z = np.linspace(-voxels_radius, voxels_radius, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

    voxels, *_ = igl.signed_distance(query_points, mesh.vertices, mesh.faces)
    voxels = voxels.reshape((res, res, res)).astype(gs.np_float, copy=False)

    T_mesh_to_sdf = np.eye(4, dtype=gs.np_float)
    T_mesh_to_sdf[:3, :3] *= (res - 1) / (voxels_radius * 2)
    T_mesh_to_sdf[:3, 3] = (res - 1) / 2

    sdf_data = {
        "voxels": voxels,
        "T_mesh_to_sdf": T_mesh_to_sdf,
    }
    return sdf_data


def surface_uvs_to_trimesh_visual(surface, uvs=None, n_verts=None):
    texture = surface.get_rgba()

    # 'trimesh' uses uvs starting from the top-left corner, so flip them to Genesis' convention.
    flipped_uvs = None
    if uvs is not None:
        flipped_uvs = uvs.copy()
        flipped_uvs[:, 1] = 1.0 - flipped_uvs[:, 1]

    # Composite emissive additively on top of the base color, but only when the base color is the packed albedo
    # (present, nonblack, and distinct from the emissive). Otherwise get_rgba already returned the emissive as the
    # albedo and re-adding it would double it. Baking it into the material makes every renderer path pick it up through
    # from_trimesh (rigid and deformable alike), and forces a PBRMaterial since SimpleMaterial has no emissive channel.
    # An image emissive samples UVs, so it is composited only when the mesh has them (else the shader references an
    # undeclared uv_0); a flat emissive color needs none.
    base = surface.texture
    emission = surface.emission
    emissive_kwargs = {}
    if emission is not None and base is not None and base is not emission and not base.is_black:
        if (
            isinstance(emission, gs.textures.ImageTexture)
            and emission.image_array is not None
            and flipped_uvs is not None
        ):
            emissive_kwargs = dict(
                emissiveTexture=Image.fromarray(emission.image_array), emissiveFactor=emission.image_color
            )
        elif isinstance(emission, gs.textures.ColorTexture):
            emissive_kwargs = dict(emissiveFactor=emission.color)

    if isinstance(texture, gs.textures.ImageTexture):
        if flipped_uvs is not None:
            assert texture.image_array.dtype == np.uint8
            image = Image.fromarray(texture.image_array)
            if emissive_kwargs:
                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=image, baseColorFactor=(255, 255, 255, 255), **emissive_kwargs
                )
            else:
                material = trimesh.visual.material.SimpleMaterial(image=image, diffuse=(1.0, 1.0, 1.0, 1.0))
            visual = trimesh.visual.TextureVisuals(uv=flipped_uvs, material=material)
        else:
            # fall back to color texture
            visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(texture.mean_color, [n_verts, 1]))
    elif isinstance(texture, gs.textures.ColorTexture):
        if n_verts is None:
            gs.raise_exception("n_verts is required for color texture.")
        if emissive_kwargs:
            # The flat base color applies through a factor, but an image emissive still samples the UVs, so pass them.
            material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=color_f32_to_u8(texture.color), **emissive_kwargs
            )
            visual = trimesh.visual.TextureVisuals(uv=flipped_uvs, material=material)
        else:
            visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(np.array(texture.color), [n_verts, 1]))
            assert visual.defined
    else:
        gs.raise_exception("Cannot get texture when generating trimesh visual.")

    return visual


def convex_decompose(mesh, coacd_options):
    # The decomposition of a scaled mesh is the decomposition of the mesh scaled, so the cache is keyed on vertices
    # normalized by the mesh scale and the hulls are rescaled on the way out: every scaled copy of an asset shares a
    # single entry.
    mesh_scale = float(np.linalg.norm(mesh.extents))
    assert not (np.isinf(mesh_scale) or np.isnan(mesh_scale) or mesh_scale <= 0.0)
    cache = get_cvx_cache(mesh.vertices / mesh_scale, mesh.faces, coacd_options)

    # loading pre-computed cache if available
    is_cached_loaded = False
    loaded_cache = cache.load()
    if loaded_cache is not None:
        gs.logger.debug("Convex decomposition file (.cvx) found in cache.")
        mesh_parts = loaded_cache["mesh_parts"]
        cached_mesh_scale = loaded_cache["mesh_scale"]

        # rescale loaded mesh parts
        if not (np.isinf(cached_mesh_scale) or np.isnan(cached_mesh_scale) or cached_mesh_scale <= 0.0):
            rescale_factor = mesh_scale / cached_mesh_scale
            for mesh_part in mesh_parts:
                mesh_part.vertices *= rescale_factor
            is_cached_loaded = True

    if not is_cached_loaded:
        with gs.logger.timer("Running convex decomposition."):
            mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            args = coacd_options
            result = coacd.run_coacd(
                mesh,
                threshold=args.threshold,
                max_convex_hull=args.max_convex_hull,
                preprocess_mode=args.preprocess_mode,
                preprocess_resolution=args.preprocess_resolution,
                resolution=args.resolution,
                mcts_nodes=args.mcts_nodes,
                mcts_iterations=args.mcts_iterations,
                mcts_max_depth=args.mcts_max_depth,
                pca=args.pca,
                merge=args.merge,
                decimate=args.decimate,
                max_ch_vertex=args.max_ch_vertex,
                extrude=args.extrude,
                extrude_margin=args.extrude_margin,
                apx_mode=args.apx_mode,
                seed=args.seed,
            )
            mesh_parts = []
            for vs, fs in result:
                mesh_parts.append(trimesh.Trimesh(vs, fs))
            cache.save({"mesh_parts": mesh_parts, "mesh_scale": mesh_scale})

    return mesh_parts


# 512 MiB of processed collision geometry. Sized by the geometry footprint actually retained (vertices and faces of
# the cached meshes), so a few large assets or many small ones are both bounded without an arbitrary entry count.
_COLLISION_GEOMS_CACHE = SizeCappedCache(max_bytes=512 * 1024 * 1024)


def postprocess_collision_geoms(
    g_infos,
    decimate,
    decimate_face_num,
    decimate_aggressiveness,
    convexify,
    decompose_error_threshold,
    coacd_options,
    watertighten,
):
    # Convexification / decomposition of a collision mesh (convex hull and coacd decomposition) is the dominant cost of
    # adding a file-based entity, and a scene built from many copies of the same asset would otherwise redo it for every
    # entity. The result is memoized on the geometry of the input collision meshes and on every option / per-geom field
    # that can change the outcome. The processed meshes are immutable (their vertices are only read downstream,
    # alignment is folded into the link frame), so the cached ones are shared across entities, which also collapses
    # their memory footprint. Only fresh g_info dicts are handed back so the caller can re-express the per-geom pose in
    # place without corrupting the template.
    if not g_infos:
        return []

    # Keyed on the vertices themselves rather than matched within a tolerance the way the on-disk caches are: one
    # entry per distinct input keeps every variant of a same topology evictable on its own, which is what bounds the
    # footprint of this cache. Re-exports of one asset are what tolerant matching buys, and they cost nothing here
    # since the expensive steps this memoizes hit their own tolerant on-disk cache anyway.
    key_parts = [
        bool(decimate),
        int(decimate_face_num),
        int(decimate_aggressiveness),
        bool(convexify),
        float(decompose_error_threshold),
        -1 if watertighten is None else int(watertighten),
        coacd_options.model_dump(),
    ]
    for g_info in g_infos:
        mesh = g_info["mesh"]
        geom_type = g_info.get("type")
        friction = g_info.get("friction")
        density = g_info.get("density")
        sol_params = g_info.get("sol_params")
        key_parts += [
            np.ascontiguousarray(mesh.verts),
            np.ascontiguousarray(mesh.faces),
            -1 if geom_type is None else int(geom_type),
            int(g_info.get("contype", 0)),
            int(g_info.get("conaffinity", 0)),
            float("nan") if friction is None else float(friction),
            float("nan") if density is None else float(density),
            np.zeros(0) if sol_params is None else np.ascontiguousarray(sol_params, dtype=np.float64),
            np.ascontiguousarray(g_info.get("pos", gu.zero_pos()), dtype=np.float64),
            np.ascontiguousarray(g_info.get("quat", gu.identity_quat()), dtype=np.float64),
        ]
    key = get_hashkey(*key_parts)

    cached = _COLLISION_GEOMS_CACHE.get(key)
    if cached is None:
        cached = _postprocess_collision_geoms_impl(
            g_infos,
            decimate,
            decimate_face_num,
            decimate_aggressiveness,
            convexify,
            decompose_error_threshold,
            coacd_options,
            watertighten,
        )
        n_bytes = sum(g_info["mesh"].verts.nbytes + g_info["mesh"].faces.nbytes for g_info in cached)
        _COLLISION_GEOMS_CACHE.put(key, cached, n_bytes)

    # Hand back per-entity geoms that share the cached geometry and its derived data (edges, vertex adjacency, inertia)
    # with the template, but own their trimesh and surface. This keeps appearance per-entity (each entity gets an
    # independent, e.g. randomized, collision color) while the heavy geometry is computed once and its arrays are
    # shared. Edges and adjacency are always needed, so they are populated up front; inertia is queried lazily through
    # the template (only entities that align their link frame need it) to avoid eager convex-hull work.
    result = []
    for g_info in cached:
        template = g_info["mesh"]
        template_tmesh = template.trimesh
        mesh = gs.Mesh(
            mesh=trimesh.Trimesh(vertices=template_tmesh.vertices, faces=template_tmesh.faces, process=False),
            surface=gs.surfaces.Collision(),
            uvs=template.uvs,
            metadata=template.metadata.copy(),
        )
        mesh._unique_edges = template.get_unique_edges()
        mesh._vert_adjacency = template.get_vert_adjacency()
        mesh._inertial_source = template
        result.append({**g_info, "mesh": mesh})
    return result


def _postprocess_collision_geoms_impl(
    g_infos,
    decimate,
    decimate_face_num,
    decimate_aggressiveness,
    convexify,
    decompose_error_threshold,
    coacd_options,
    watertighten,
):
    # Early return if there is no geometry to process
    if not g_infos:
        return []

    # Check whether the geometries are authored, ie they are all watertight and convex, as a cheap proxy
    is_authored = all(
        (tmesh := g_info["mesh"].trimesh).is_winding_consistent and tmesh.is_watertight and tmesh.is_convex
        for g_info in g_infos
    )

    # Weld coincident vertices of non-convex collision meshes onto a separate copy. Formats that store unshared
    # per-face vertices (notably STL) yield a vertex soup whose duplicates differ only by float rounding; the
    # downstream exact dedup at geom build only partially fuses them, leaving a degraded (sliver) mesh that corrupts
    # the SDF. The convex path skips this (the hull / decomposition replaces the surface anyway). The collision mesh
    # may be the very same trimesh as the visual geom, so the weld must not be done in place: only the collision geom
    # is swapped for the welded copy, leaving the visual vertices untouched. Already-shared meshes (OBJ, glTF) weld to
    # no fewer vertices and keep their original geom.
    if not is_authored and not convexify:
        for g_info in g_infos:
            if g_info["type"] != gs.GEOM_TYPE.MESH:
                continue
            welded = g_info["mesh"].trimesh.copy()
            welded.merge_vertices()
            if len(welded.vertices) < len(g_info["mesh"].trimesh.vertices):
                g_info["mesh"] = gs.Mesh.from_trimesh(
                    mesh=welded,
                    surface=gs.surfaces.Collision(),
                    metadata=g_info["mesh"].metadata.copy(),
                )

    # Try the repair meshes that seems to be "broken" but not beyond repair.
    # Note that this procedure is only applied if the estimated volume is significantly different before and after
    # repair, to avoid altering the original mesh without actual benefit. Moreover, only duplicate faces are removed,
    # which is less aggressive than `Trimesh.process(validate=True)`.
    if not is_authored:
        for g_info in g_infos:
            mesh = g_info["mesh"]
            tmesh = mesh.trimesh
            if g_info["type"] != gs.GEOM_TYPE.MESH:
                continue
            if tmesh.is_winding_consistent and not tmesh.is_watertight:
                tmesh_repaired = tmesh.copy()
                tmesh_repaired.update_faces(tmesh_repaired.unique_faces())
                if tmesh_repaired.volume < 0.0:
                    tmesh_repaired.invert()
                if abs(tmesh_repaired.volume) < gs.EPS:
                    continue
                if abs(abs(tmesh.volume / tmesh_repaired.volume) - 1.0) > MESH_REPAIR_ERROR_THRESHOLD:
                    gs.logger.info(
                        "Collision mesh is not watertight and has ill-defined volume. It will be repaired by removing "
                        "duplicate faces."
                    )
                    tmesh.update_faces(tmesh.unique_faces())
                    tmesh._cache.clear()
                    tmesh.visual._cache.clear()

    # Check which geometries can be convexified without decomposition
    geoms_must_decompose = [False] * len(g_infos)
    volume_err_max = 0.0
    if not is_authored and convexify:
        for i_g, g_info in enumerate(g_infos):
            mesh = g_info["mesh"]
            tmesh = mesh.trimesh

            # Skip geometries that do not corresponds to mesh or have no enclosed volume
            if g_info["type"] != gs.GEOM_TYPE.MESH:
                continue
            cmesh = trimesh.convex.convex_hull(tmesh)
            if abs(cmesh.volume) < gs.EPS:
                continue

            # Fix mesh temporarily to make volume computation more reliable
            if not tmesh.is_winding_consistent:
                tmesh = tmesh.copy()
                tmesh.process(validate=True)

            # Fix negative volume by inverting faces
            if tmesh.volume < 0.0:
                tmesh = tmesh.copy()
                tmesh.invert()

            # Compute volume approximation error between true geometry and its convex hull conservatively
            if not tmesh.is_winding_consistent:
                if not math.isinf(decompose_error_threshold):
                    geoms_must_decompose[i_g] = True
                    volume_err_max = float("inf")
            elif abs(tmesh.volume) > gs.EPS:
                volume_err = abs(cmesh.volume / abs(tmesh.volume) - 1.0)
                if volume_err > decompose_error_threshold:
                    geoms_must_decompose[i_g] = True
                    volume_err_max = max(volume_err_max, volume_err)
    must_decompose = any(geoms_must_decompose)

    # Fuse collision geoms that share a collision group and contact parameters into one closed surface per consistent
    # sub-group, transforming each sub-mesh into its group's first geom frame so the fused geom keeps that pose. One
    # surface per group exposes the whole-object topology needed to detect enclosure correctly and to estimate wall
    # thickness (which sets the grid SDF cell size, hence collision accuracy), and it is cheaper - fewer vertices to
    # scan and roughly one contact set per link instead of per geom. All geom types can be merged except planes and
    # terrains. On the nonconvex path, sub-groups are systematically fused unless watertightening is disabled; on the
    # convex path, only when one of their members requires decomposition.
    geoms_is_fused = [False] * len(g_infos)
    if len(g_infos) > 1 and ((not convexify and watertighten is not None) or (not is_authored and must_decompose)):
        fusion_groups: list[list[int]] = []
        for i, g_info in enumerate(g_infos):
            # Join the first existing sub-group with a matching collision group and contact params, else open a new one.
            if g_info["type"] not in (gs.GEOM_TYPE.PLANE, gs.GEOM_TYPE.TERRAIN):
                for fusion_group in fusion_groups:
                    first_g_info = g_infos[fusion_group[0]]
                    if (
                        first_g_info["type"] not in (gs.GEOM_TYPE.PLANE, gs.GEOM_TYPE.TERRAIN)
                        and all(first_g_info.get(name) == g_info.get(name) for name in ("contype", "conaffinity"))
                        and all(
                            np.allclose(first_g_info.get(name, np.nan), g_info.get(name, np.nan), equal_nan=True)
                            # A fused mesh carries a single density, so geoms with different authored densities must
                            # stay separate for the density-weighted link inertial to survive fusion.
                            for name in ("friction", "sol_params", "density")
                        )
                    ):
                        fusion_group.append(i)
                        break
                else:
                    fusion_groups.append([i])
            else:
                fusion_groups.append([i])

        fused_infos = []
        geoms_is_fused = []
        for fusion_group in fusion_groups:
            if len(fusion_group) == 1 or (convexify and not any(geoms_must_decompose[i] for i in fusion_group)):
                fused_infos.extend(g_infos[i] for i in fusion_group)
                geoms_is_fused.extend([False] * len(fusion_group))
                continue
            first_g_info = g_infos[fusion_group[0]]
            T_first_inv = np.linalg.inv(
                gu.trans_quat_to_T(first_g_info.get("pos", gu.zero_pos()), first_g_info.get("quat", gu.identity_quat()))
            )
            tmeshes = []
            metadata = set(first_g_info["mesh"].metadata.items())
            for i in fusion_group:
                g_info = g_infos[i]
                mesh = g_info["mesh"]
                tmesh = mesh.trimesh
                T_rel = T_first_inv @ gu.trans_quat_to_T(
                    g_info.get("pos", gu.zero_pos()), g_info.get("quat", gu.identity_quat())
                )
                if not np.allclose(T_rel, np.eye(4)):
                    tmesh = tmesh.copy()
                    tmesh.apply_transform(T_rel)
                metadata &= set(mesh.metadata.items())
                tmeshes.append(tmesh)
            tmesh = trimesh.util.concatenate(tmeshes)
            mesh = gs.Mesh.from_trimesh(
                mesh=tmesh, surface=gs.surfaces.Collision(), metadata=dict(metadata) | {"merged": True}
            )
            fused_infos.append({**first_g_info, **dict(type=gs.GEOM_TYPE.MESH, data=None, mesh=mesh)})
            geoms_is_fused.append(True)
        g_infos = fused_infos

        # A genuinely fused mesh (a sub-group of more than one geom) re-enters the pipeline so its convex hull /
        # decomposition is recomputed on the union; requiring an actual merge stops infinite recursion once every
        # sub-group is either a singleton or accurate enough to skip decomposition.
        if convexify and any(geoms_is_fused):
            return _postprocess_collision_geoms_impl(
                g_infos,
                decimate,
                decimate_face_num,
                decimate_aggressiveness,
                convexify,
                decompose_error_threshold,
                coacd_options,
                watertighten,
            )

    # Nonconvex: watertighten each fused surface into a closed mesh so the grid SDF is reliable. The convex path skips
    # this (its hull / decomposition replaces the surface anyway, so an alpha-wrap would be wasted work).
    if not convexify and watertighten is not None:
        from .watertighten import watertighten_mesh

        for g_info, is_fused in zip(g_infos, geoms_is_fused):
            # Fused geoms are always watertightened, as their sub-meshes may overlap while being individually
            # watertight. Other geoms are skipped if they are not generic meshes or already watertight or convex.
            tmesh = g_info["mesh"].trimesh
            if not is_fused and (g_info["type"] != gs.GEOM_TYPE.MESH or tmesh.is_watertight or tmesh.is_convex):
                continue

            # On-disk cache keyed by (vertices, faces, aggressiveness): a repeated build on the same geom is a file read
            # instead of a multi-second SDF + DC + QEM rebuild.
            cache = get_wt_cache(tmesh.vertices, tmesh.faces, watertighten)
            cached_mesh = cache.load()
            if cached_mesh is None:
                cached_mesh = watertighten_mesh(tmesh.vertices, tmesh.faces, aggressiveness=watertighten)
                cache.save(cached_mesh)
            v_out, f_out = cached_mesh
            fused = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
            metadata = g_info["mesh"].metadata.copy()
            metadata["watertightened"] = True
            g_info["mesh"] = gs.Mesh.from_trimesh(mesh=fused, surface=gs.surfaces.Collision(), metadata=metadata)

    if not is_authored and must_decompose:
        if math.isinf(volume_err_max):
            gs.logger.info(
                "Collision mesh has inconsistent winding and 'decompose_error_threshold' != float('inf'). "
                "Falling back to more expensive convex decomposition (see FileMorph options)."
            )
        else:
            gs.logger.info(
                f"Convex hull is not accurate enough for collision detection ({volume_err_max:.3f}). Falling back to "
                "more expensive convex decomposition (see FileMorph options)."
            )
        _g_infos = []
        for g_info in g_infos:
            mesh = g_info["mesh"]
            tmesh = mesh.trimesh
            if g_info["type"] != gs.GEOM_TYPE.MESH:
                volume_err = 0.0
            elif not tmesh.is_winding_consistent:
                volume_err = float("inf")
            elif abs(tmesh.volume) < gs.EPS:
                volume_err = 0.0
            else:
                cmesh = trimesh.convex.convex_hull(tmesh)
                volume_err = abs(cmesh.volume / abs(tmesh.volume) - 1.0)
            if volume_err > decompose_error_threshold:  # Note that 'inf' is not larger than 'inf'
                tmeshes = convex_decompose(tmesh, coacd_options)
                meshes = [
                    gs.Mesh.from_trimesh(
                        tmesh, surface=gs.surfaces.Collision(), metadata={**mesh.metadata, "decomposed": True}
                    )
                    for tmesh in tmeshes
                ]
                _g_infos += [{**g_info, **dict(mesh=mesh)} for mesh in meshes]
            else:
                _g_infos.append(g_info)
        g_infos = _g_infos

    # Process of meshes sequentially
    _g_infos = []
    for g_info in g_infos:
        mesh = g_info["mesh"]
        tmesh = mesh.trimesh

        num_faces = len(tmesh.faces)
        if not decimate and num_faces > 5000:
            gs.logger.warning(
                f"At least one of the meshes contain many faces ({num_faces}). Consider setting "
                "'morph.decimate=True' to speed up collision detection and improve numerical stability."
            )
        if decimate and decimate_face_num < 100:
            gs.logger.warning(
                "`decimate_face_num` should be greater than 100 to ensure sufficient geometry details are preserved."
            )

        # Decimation is an independent step applied after watertightening (which runs its own internal QEM): when
        # 'decimate' is requested it simplifies the collision mesh - including a watertighten wrap - down to
        # 'decimate_face_num'. It is gated on the mesh being either above that target or watertight, since decimating
        # a low-poly non-watertight mesh would be unreliable.
        must_decimate = num_faces > decimate_face_num or tmesh.is_watertight
        if not must_decimate:
            gs.logger.debug(
                "Collision mesh is not watertight. Decimate would be unreliable. Skipping as mesh is already low-poly."
            )

        mesh = gs.Mesh.from_trimesh(
            mesh=tmesh,
            convexify=convexify,
            decimate=decimate and must_decimate,
            decimate_face_num=decimate_face_num,
            decimate_aggressiveness=decimate_aggressiveness,
            surface=gs.surfaces.Collision(),
            metadata=mesh.metadata.copy(),
        )
        _g_infos.append({**g_info, **dict(mesh=mesh)})

    return _g_infos


@lru_cache(maxsize=32)
def _load_trimesh_scene_cached(path, group_by_material, mtime) -> "trimesh.Scene":
    # Parsing a mesh file from disk (read + decode) is by far the dominant cost of adding a file-based entity, and a
    # scene built from many copies of the same asset (e.g. a grid of identical objects) would otherwise re-parse the
    # exact same bytes for every entity. The result is keyed by file modification time so an edited asset is reloaded.
    # The returned scene is treated as an immutable template - callers copy each geometry out before applying scale or
    # surface - so it is never mutated and is safe to share across entities.
    return trimesh.load(path, force="scene", group_material=group_by_material, process=False)


register_cache_clear(_load_trimesh_scene_cached.cache_clear)


def parse_mesh_trimesh(path, group_by_material, scale, is_mesh_zup, surface) -> "list[gs.Mesh]":
    meshes: list[gs.Mesh] = []
    scene = _load_trimesh_scene_cached(path, group_by_material, os.path.getmtime(path))
    for tmesh in scene.geometry.values():
        if not isinstance(tmesh, trimesh.Trimesh):
            gs.raise_exception(f"Mesh type not supported: {path}")
        mesh = gs.Mesh.from_trimesh(
            mesh=tmesh, scale=scale, surface=surface, is_mesh_zup=is_mesh_zup, metadata={"mesh_path": path}
        )
        meshes.append(mesh)
    return meshes


def trimesh_to_mesh(mesh, scale, surface) -> "gs.Mesh":
    return gs.Mesh.from_trimesh(mesh=mesh, scale=scale, surface=surface)


def adjust_alpha_cutoff(alpha_cutoff, alpha_mode):
    if alpha_mode == 0:  # OPAQUE
        return 0.0
    if alpha_mode == 1:  # MASK
        return alpha_cutoff
    return None  # BLEND


def PIL_to_array(image):
    # A paletted image stores indices into its palette, so its texels are the colors those indices stand for
    if isinstance(image, Image.Image) and image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    return np.array(image)


def tonemapped(image):
    exposure = 0.5
    return (np.clip(np.power(image / 255 * np.power(2, exposure), 1 / 2.2), 0, 1) * 255).astype(np.uint8)


def create_texture(image, factor, encoding):
    if image is not None:
        return gs.textures.ImageTexture(image_array=image, image_color=factor, encoding=encoding)
    if factor is not None:
        return gs.textures.ColorTexture(color=factor)
    return None


def apply_transform(transform, positions, normals=None):
    # Note that here transform's translation is on the bottom row, different from that in Genesis and trimesh.
    transformed_positions = (np.column_stack([positions, np.ones(len(positions))]) @ transform)[:, :3]

    transformed_normals = normals
    if normals is not None:
        rot_mat = transform[:3, :3]
        if np.abs(3.0 - np.trace(rot_mat)) > gs.EPS**2:  # has rotation or scaling
            transformed_normals = normals @ rot_mat
            scale = np.linalg.norm(rot_mat, axis=1, keepdims=True)
            if np.any(np.abs(scale - 1.0) > gs.EPS):  # has scale
                transformed_normals /= np.linalg.norm(transformed_normals, axis=1, keepdims=True)

    return transformed_positions, transformed_normals


def create_frame(
    origin_radius=0.012, axis_radius=0.005, axis_length=1.0, head_radius=0.01, head_length=0.03, sections=12
):
    origin = create_sphere(radius=origin_radius, subdivisions=2)

    x = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.7, 0.0, 0.0, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )
    y = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.0, 0.7, 0.0, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )
    z = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.0, 0.0, 0.7, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )

    x.vertices = gu.transform_by_R(x.vertices, gu.euler_to_R((0.0, 90.0, 0.0)))
    y.vertices = gu.transform_by_R(y.vertices, gu.euler_to_R((-90.0, 0.0, 0.0)))

    return trimesh.util.concatenate([origin, x, y, z])


def create_camera_frustum(camera, color):
    # camera
    camera_mesh = trimesh.load(os.path.join(get_src_dir(), "assets", "meshes", "camera/camera.glb"), force="mesh")
    camera_mesh.visual = camera_mesh.visual.to_color()
    camera_mesh.apply_translation([0.0, 0.0, 1.0])
    camera_mesh.apply_scale(0.05)

    # frustum
    near_half_height = camera.near * np.tan(np.deg2rad(camera.fov / 2))
    near_half_width = near_half_height * camera.aspect_ratio
    far_half_height = camera.far * np.tan(np.deg2rad(camera.fov / 2))
    far_half_width = far_half_height * camera.aspect_ratio

    # Define the vertices of the frustum
    vertices = np.array(
        [
            [0, 0, 0],  # apex
            [-near_half_width, -near_half_height, -camera.near],  # near bottom left
            [near_half_width, -near_half_height, -camera.near],  # near bottom right
            [near_half_width, near_half_height, -camera.near],  # near top right
            [-near_half_width, near_half_height, -camera.near],  # near top left
            [-far_half_width, -far_half_height, -camera.far],  # far bottom left
            [far_half_width, -far_half_height, -camera.far],  # far bottom right
            [far_half_width, far_half_height, -camera.far],  # far top right
            [-far_half_width, far_half_height, -camera.far],  # far top left
        ]
    )

    # Define the faces of the frustum
    faces = np.array(
        [
            # # near face
            # [1, 2, 3, 4],
            # # far face
            # [5, 6, 7, 8],
            # side face
            [2, 1, 5, 6],
            [3, 2, 6, 7],
            [4, 3, 7, 8],
            [1, 4, 8, 5],
        ]
    )

    # Create the frustum mesh
    frustum_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    frustum_mesh.visual.vertex_colors = np.asarray(color, dtype=np.float32)
    return trimesh.util.concatenate([camera_mesh, frustum_mesh])


def create_tets_mesh(n_tets=1, halfsize=1.0, quats=None, randomize_halfsize=True):
    """
    Create artistic tet-based mesh for rendering particles as tets.
    """
    # create tet-based particles given positions
    vert_per_tet = 12
    face_per_tet = 20
    if quats is None:
        quats = np.tile(gu.random_quaternion(n_tets), [1, vert_per_tet]).reshape(-1, 4)

    if randomize_halfsize:
        halfsize = (
            np.tile(np.random.uniform(0.3, 1.9, size=(n_tets, 1)), [1, vert_per_tet * 3]).reshape(-1, 3) * halfsize
        )
        halfsize = (
            np.tile(np.random.uniform(0.3, 1.9, size=(n_tets * 4, 1)), [1, vert_per_tet // 4 * 3]).reshape(-1, 3)
            * halfsize
        )
        # halfsize = np.random.uniform(0.2, 1.9, size=(n_tets * vert_per_tet, 3)) * halfsize

    vertices = (
        np.tile(
            np.array(
                [
                    [0.91835, 0.836701, 0.91835],
                    [0.91835, 0.91835, 0.836701],
                    [0.836701, 0.91835, 0.91835],
                    [-0.836701, 0.91835, -0.91835],
                    [-0.91835, 0.836701, -0.91835],
                    [-0.91835, 0.91835, -0.836701],
                    [-0.836701, -0.91835, 0.91835],
                    [-0.91835, -0.836701, 0.91835],
                    [-0.91835, -0.91835, 0.836701],
                    [0.91835, -0.836701, -0.91835],
                    [0.91835, -0.91835, -0.836701],
                    [0.836701, -0.91835, -0.91835],
                ]
            ),
            [n_tets, 1],
        )
        * halfsize
    )
    vertices = gu.transform_by_quat(vertices, quats)

    faces = np.tile(
        np.array(
            [
                [0, 6, 10],
                [11, 8, 4],
                [2, 5, 7],
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [0, 10, 9],
                [9, 1, 0],
                [1, 3, 5],
                [5, 2, 1],
                [2, 7, 6],
                [6, 0, 2],
                [4, 8, 7],
                [7, 5, 4],
                [8, 11, 10],
                [10, 6, 8],
                [3, 9, 11],
                [11, 4, 3],
                [1, 9, 3],
            ]
        ),
        [n_tets, 1],
    )
    faces_offset = np.tile(np.arange(0, n_tets).reshape(-1, 1) * vert_per_tet, [1, face_per_tet * 3]).reshape(
        n_tets * face_per_tet, 3
    )
    faces += faces_offset

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def transform_tets_mesh_verts(vertices, positions, zs=None):
    vert_per_tet = 12
    assert len(vertices) == len(positions) * vert_per_tet
    vertices = vertices.reshape(-1, vert_per_tet, 3)
    if zs is not None:
        assert len(zs) == len(positions)
        vertices = gu.transform_by_R(vertices, gu.z_up_to_R(zs))
    return (vertices + positions[:, np.newaxis]).reshape((-1, 3))


@lru_cache(maxsize=32)
def _create_unit_sphere_impl(subdivisions):
    mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=subdivisions)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    attrs = {"vertex_normals": mesh.vertex_normals.copy(), "face_normals": mesh.face_normals.copy()}
    for data in (vertices, faces, *attrs.values()):
        data.flags.writeable = False
    return vertices, faces, attrs


def create_sphere(radius, subdivisions=3, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces, attrs = _create_unit_sphere_impl(subdivisions=subdivisions)
    vertices = vertices * radius
    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.tile(color_f32_to_u8(color), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache.update(attrs)
    return mesh


@lru_cache(maxsize=32)
def _create_unit_cylinder_impl(sections):
    mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=sections)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    attrs = {"vertex_normals": mesh.vertex_normals.copy(), "face_normals": mesh.face_normals.copy()}
    for data in (vertices, faces, *attrs.values()):
        data.flags.writeable = False
    return vertices, faces, attrs


def create_cylinder(radius, height, sections=None, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces, attrs = _create_unit_cylinder_impl(sections=sections)
    vertices = vertices * (radius, radius, height)
    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.tile(color_f32_to_u8(color), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache.update(attrs)
    return mesh


@lru_cache(maxsize=32)
def _create_unit_cone_impl(sections):
    mesh = trimesh.creation.cone(radius=1.0, height=1.0, sections=sections)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    attrs = {"vertex_normals": mesh.vertex_normals.copy(), "face_normals": mesh.face_normals.copy()}
    for data in (vertices, faces, *attrs.values()):
        data.flags.writeable = False
    return vertices, faces, attrs


def create_cone(radius, height, sections=None, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces, attrs = _create_unit_cone_impl(sections=sections)
    vertices = vertices * (radius, radius, height)
    for name, normals in attrs.items():
        normals = normals / (radius, radius, height)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        attrs[name] = normals
    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.tile(color_f32_to_u8(color), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache.update(attrs)
    return mesh


def create_arrow(
    length=1.0,
    radius=0.02,
    l_ratio=0.25,
    r_ratio=1.5,
    body_color=(1.0, 1.0, 1.0, 1.0),
    head_color=(1.0, 1.0, 1.0, 1.0),
    sections=12,
):
    r_head = radius * r_ratio
    r_body = radius
    l_head = length * l_ratio
    l_body = length - l_head

    head = create_cone(r_head, l_head, sections=sections, color=head_color)
    body = create_cylinder(r_body, l_body, sections=sections, color=body_color)
    face_normals = np.vstack((body._cache["face_normals"], head._cache["face_normals"]))
    face_normals.flags.writeable = False
    head._data["vertices"] += np.array([0.0, 0.0, l_body])
    body._data["vertices"] += np.array([0.0, 0.0, l_body / 2])

    vertices = np.vstack((body.vertices, head.vertices))
    faces = np.vstack((body.faces, head.faces + len(body.vertices)))
    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.vstack((body.visual.vertex_colors, head.visual.vertex_colors))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache["face_normals"] = face_normals
    return mesh


def create_line(start, end, radius=0.002, color=(1.0, 1.0, 1.0, 1.0), sections=12):
    vec = end - start
    length = np.linalg.norm(vec)
    mesh = create_cylinder(radius, length, sections, color)  # along z-axis
    mesh._data["vertices"][:, -1] += length / 2.0
    mesh.vertices = gu.transform_by_trans_R(mesh._data["vertices"], start, gu.z_up_to_R(vec))
    return mesh


@lru_cache(maxsize=1)
def _create_unit_box_impl():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    attrs = {"vertex_normals": mesh.vertex_normals.copy(), "face_normals": mesh.face_normals.copy()}
    for data in (vertices, faces, *attrs.values()):
        data.flags.writeable = False
    return vertices, faces, attrs


def create_box(extents=None, color=(1.0, 1.0, 1.0, 1.0), bounds=None, wireframe=False, wireframe_radius=0.002):
    if bounds is not None:
        bounds = np.asarray(bounds)
        extents = bounds[1] - bounds[0]
        pos = bounds.mean(axis=0)
    elif extents is not None:
        extents = np.asarray(extents)
        pos = np.zeros(3)
    else:
        gs.raise_exception("Neither `extents` nor `bounds` is provided.")

    if wireframe:
        box_vertices = np.asarray(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        box_vertices = box_vertices * extents + pos
        box_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

        n_verts = 0
        vertices, faces, attrs = [], [], {}
        for v_start, v_end in box_edges:
            p_start, p_end = box_vertices[v_start], box_vertices[v_end]
            vec = p_end - p_start
            length = np.linalg.norm(vec)

            line_vertices, line_faces, line_attrs = _create_unit_cylinder_impl(sections=12)
            line_vertices = line_vertices * (wireframe_radius, wireframe_radius, length)
            line_vertices[:, -1] += length / 2.0
            line_vertices = gu.transform_by_trans_R(line_vertices, p_start, gu.z_up_to_R(vec))

            vertices.append(line_vertices)
            faces.append(line_faces + n_verts)
            for name, value in line_attrs.items():
                attrs.setdefault(name, []).append(value)
            n_verts += len(line_vertices)

        for vertex in box_vertices:
            sphere_vertices, sphere_faces, sphere_attrs = _create_unit_sphere_impl(subdivisions=3)

            vertices.append(sphere_vertices * wireframe_radius + vertex)
            faces.append(sphere_faces + n_verts)
            for name, value in sphere_attrs.items():
                attrs.setdefault(name, []).append(value)
            n_verts += len(sphere_vertices)

        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        for name, values in attrs.items():
            attrs[name] = np.concatenate(values)
    else:
        vertices, faces, attrs = _create_unit_box_impl()
        vertices = vertices * extents + pos

    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.tile(color_f32_to_u8(color), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache.update(attrs)

    return mesh


def create_plane(
    normal=(0.0, 0.0, 1.0),
    plane_size=(1e3, 1e3),
    tile_size=(1, 1),
    color_or_texture=DEFAULT_PLANE_TEXTURE_PATH,
    double_sided=False,
):
    if isinstance(color_or_texture, str):
        color, texture_path = None, color_or_texture
    else:
        color, texture_path = color_or_texture, None

    thickness = 1e-2  # for safety
    mesh = trimesh.creation.box(extents=[plane_size[0], plane_size[1], thickness])
    mesh.vertices[:, 2] -= thickness / 2
    mesh.vertices = gu.transform_by_R(mesh.vertices, gu.z_up_to_R(np.asarray(normal, dtype=np.float32)))

    half_x, half_y = (plane_size[0] * 0.5, plane_size[1] * 0.5)
    verts = np.array(
        [
            [-half_x, -half_y, 0.0],
            [half_x, -half_y, 0.0],
            [half_x, half_y, 0.0],
            [-half_x, -half_y, 0.0],
            [half_x, half_y, 0.0],
            [-half_x, half_y, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.arange(6, dtype=np.int32).reshape(-1, 3)

    if double_sided:
        # Add reversed faces for back-facing visibility
        faces = np.vstack([faces, faces[:, ::-1]])

    vmesh = trimesh.Trimesh(verts, faces, process=False)
    # Align visual surface with the collision top face, both at z=0 in link-local frame, so morph.pos specifies
    # the actual rendered surface position.
    vmesh.vertices = gu.transform_by_R(vmesh.vertices, gu.z_up_to_R(np.asarray(normal, dtype=np.float32)))

    if texture_path is not None:
        n_tile_x, n_tile_y = plane_size[0] / tile_size[0], plane_size[1] / tile_size[1]
        uv_coords = np.array(
            [
                [0, 0],
                [n_tile_x, 0],
                [n_tile_x, n_tile_y],
                [0, 0],
                [n_tile_x, n_tile_y],
                [0, n_tile_y],
            ],
            dtype=np.float32,
        )
        if double_sided:
            # Duplicate UV coords for back faces
            uv_coords = np.vstack([uv_coords, uv_coords])

        vmesh.visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            material=trimesh.visual.material.SimpleMaterial(
                image=Image.open(os.path.join(get_assets_dir(), texture_path)),
            ),
        )
    else:
        vmesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(np.asarray(color, dtype=np.float32), (len(vmesh.vertices), 1))
        )

    return vmesh, mesh


def generate_tetgen_config_from_morph(morph):
    if not isinstance(morph, gs.options.morphs.TetGenMixin):
        raise TypeError(
            f"Expected an instance of a class that inherits from TetGenMixin, but got an instance of {type(morph).name}."
        )
    return dict(
        order=morph.order,
        mindihedral=morph.mindihedral,
        minratio=morph.minratio,
        nobisect=morph.nobisect,
        quality=morph.quality,
        maxvolume=morph.maxvolume,
        verbose=morph.verbose,
    )


def make_tetgen_switches(cfg):
    """Build a TetGen switches string from a config dict."""
    flags = ["p"]

    if cfg.get("quality", True):
        r = cfg.get("minratio", 1.1)
        di = cfg.get("mindihedral", 10)
        flags.append(f"q{r}/{di}")

    a = cfg.get("maxvolume", -1.0)
    if a > 0:
        flags.append(f"a{a}")

    o = cfg.get("order", 1)
    if o != 1:
        flags.append(f"o{o}")

    if cfg.get("nobisect", False):
        flags.append("Y")

    v = cfg.get("verbose", 0)
    if v > 0:
        flags.append("V" * v)

    return "".join(flags)


def merge_submeshes(verts_list, faces_list):
    """Concatenate sub-meshes into a single mesh, welding vertices that share a position.

    Welding covers both duplicated vertices within a sub-mesh (e.g. texture-seam splits) and vertices shared across
    sub-meshes, so that downstream consumers (e.g. tetrahedralization) see one connected surface.

    Parameters
    ----------
    verts_list : list of np.ndarray
        Per-sub-mesh vertex arrays with shape (n_verts, 3).
    faces_list : list of np.ndarray
        Per-sub-mesh face arrays with shape (n_faces, 3), indexing into the matching vertex array.

    Returns
    -------
    verts : np.ndarray
        Welded vertices, ordered by first occurrence in the concatenated input.
    faces : np.ndarray
        Concatenated faces, re-indexed against the welded vertices.
    verts_maps : list of np.ndarray
        For each sub-mesh, the map from its local vertex indices to indices into the welded vertices.
    """
    submeshes_offset = (0, *np.cumsum([len(verts) for verts in verts_list]))
    stacked_verts = np.concatenate(verts_list, axis=0)
    stacked_faces = np.concatenate([faces + offset for faces, offset in zip(faces_list, submeshes_offset)], axis=0)

    # Weld by quantized position (see VERT_WELD_QUANTIZE_FACTOR). int64 is required: quantized metre-scale
    # coordinates overflow int32.
    quantized = np.round(stacked_verts / VERT_WELD_QUANTIZE_FACTOR).astype(np.int64)
    _, unique_idx, remap = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    # np.unique orders groups by quantized key; rank the first-occurrence indices to restore input order instead.
    rank = np.empty(len(unique_idx), dtype=gs.np_int)
    rank[np.argsort(unique_idx)] = np.arange(len(unique_idx))
    remap = rank[remap]

    verts = stacked_verts[np.sort(unique_idx)]
    faces = remap[stacked_faces]
    verts_maps = [remap[start:end] for start, end in zip(submeshes_offset, submeshes_offset[1:])]
    return verts, faces, verts_maps


def tetrahedralize_mesh(mesh, tet_cfg):
    tet = tetgen.TetGen(mesh.vertices.astype(np.float64, copy=False), mesh.faces.astype(np.int32, copy=False))

    # Build and apply the switches string directly, since
    # the Python wrapper sometimes ignores certain kwargs
    # (e.g. maxvolume). See: https://github.com/pyvista/tetgen/issues/24
    verts, elems, *_ = tet.tetrahedralize(switches=make_tetgen_switches(tet_cfg))

    return verts, elems


def visualize_tet(tet, mesh, show_surface=True, plot_cell_qual=False):
    grid = tet.grid
    if show_surface:
        grid.plot(show_edges=True)
    else:
        # get cell centroids
        cells = grid.cells.reshape(-1, 5)[:, 1:]
        cell_center = grid.points[cells].mean(axis=1)

        # extract cells below the 0 xy plane
        cell_ind = (cell_center[:, 2] < 0.0).nonzero(as_tuple=False)
        subgrid = grid.extract_cells(cell_ind)

        # advanced plotting
        if plot_cell_qual:
            cell_qual = subgrid.compute_cell_quality()["CellQuality"]
            subgrid.plot(
                scalars=cell_qual, stitle="Quality", cmap="bwr", clim=[0, 1], flip_scalars=True, show_edges=True
            )
        else:
            # Delaying import of 'pyvista' because it is an optional dependency
            import pyvista as pv

            faces = np.concatenate([np.full((mesh.faces.shape[0], 1), mesh.faces.shape[1]), mesh.faces], axis=1)
            pv_data = pv.PolyData(mesh.vertices, faces)

            plotter = pv.Plotter()
            plotter.add_mesh(subgrid, "lightgrey", lighting=True, show_edges=True)
            plotter.add_mesh(pv_data, "r", "wireframe")
            plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
            plotter.show()


def check_exr_compression(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    exr_header = exr_file.header()
    if exr_header["compression"].v > Imath.Compression.PIZ_COMPRESSION:
        new_exr_path = get_exr_path(exr_path)
        if os.path.exists(new_exr_path):
            gs.logger.info(f"Assets of fixed compression detected and used: {new_exr_path}.")
        else:
            gs.logger.warning(
                f"EXR image {exr_path}'s compression type {exr_header['compression']} is not supported. "
                f"Converting to compression type ZIP_COMPRESSION and saving to {new_exr_path}."
            )

            channel_data = {channel: exr_file.channel(channel) for channel in exr_header["channels"]}
            exr_header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)

            os.makedirs(os.path.dirname(new_exr_path), exist_ok=True)
            new_exr_file = OpenEXR.OutputFile(new_exr_path, exr_header)
            new_exr_file.writePixels(channel_data)
            new_exr_file.close()

        exr_path = new_exr_path

    exr_file.close()
    return exr_path
