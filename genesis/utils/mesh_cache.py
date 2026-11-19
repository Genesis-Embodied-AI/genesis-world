"""On-disk caches of data derived from a mesh, keyed on its geometry up to the noise of a re-export."""

import importlib.metadata
import os

import numpy as np

import genesis as gs

from . import serialization
from .cache import (
    CACHE_FORMAT_VERSION,
    get_cvx_cache_dir,
    get_hashkey,
    get_ptc_cache_dir,
    get_remesh_cache_dir,
    get_tet_cache_dir,
    get_wt_cache_dir,
    get_wth_cache_dir,
)

# Largest coordinate discrepancy, as a fraction of the bounding-box diagonal, under which two meshes are considered
# identical by the on-disk caches. Orders of magnitude below the spacing of genuinely distinct geometry, and orders of
# magnitude above the last-significant-digit noise separating two exports of the same model.
MESH_CACHE_MATCH_RTOL = 1e-6
# Bumped when watertighten output changes for a fixed (mesh, aggressiveness): forces a cache miss on stale entries.
WT_CACHE_VERSION = 8
# Bumped when the wall-thickness estimate changes for a fixed mesh and parameters: forces a cache miss on stale entries.
WTH_CACHE_VERSION = 3
# Bumped when the particles sampled for a fixed key change or when the layout of an entry does: forces a cache miss on
# stale entries.
PTC_CACHE_VERSION = 3


def get_cvx_cache(verts, faces, coacd_options):
    # Keyed on the coacd version, since a coacd release may change the decomposition of a fixed mesh.
    coacd_version = importlib.metadata.version("coacd")
    return MeshCache(get_cvx_cache_dir(), "cvx", verts, faces, coacd_options.__dict__, coacd_version)


def get_ptc_cache(verts, faces, p_size, sampler, random_state):
    return MeshCache(get_ptc_cache_dir(), "ptc", verts, faces, p_size, sampler, random_state, PTC_CACHE_VERSION)


def get_tet_cache(verts, faces, tet_cfg):
    return MeshCache(get_tet_cache_dir(), "tet", verts, faces, tet_cfg)


def get_remesh_cache(verts, faces, edge_len_abs, edge_len_ratio, fix):
    return MeshCache(get_remesh_cache_dir(), "rm", verts, faces, edge_len_abs, edge_len_ratio, fix)


def get_wt_cache(verts, faces, aggressiveness):
    return MeshCache(get_wt_cache_dir(), "wt", verts, faces, aggressiveness, WT_CACHE_VERSION)


def get_wth_cache(verts, faces, quantile, weight_by_area):
    return MeshCache(get_wth_cache_dir(), "wth", verts, faces, quantile, weight_by_area, WTH_CACHE_VERSION)


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

    An entry holds two records (see 'serialization.write_record'), the reference vertices then the payload, so the
    scan reads only the vertices of the candidates it rejects. Nothing is pickled, so an entry loads in any environment
    sharing the cache, whatever its Python, NumPy or optional packages. The payload is any value a record holds.

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
        bucket_key = get_hashkey(len(self._verts), np.ascontiguousarray(faces), *params, CACHE_FORMAT_VERSION)
        self._bucket_dir = os.path.join(cache_dir, bucket_key)
        self._filename = f"{get_hashkey(self._verts)}.{ext}"

    def load(self):
        """Return the payload of the matching entry, or None if this bucket holds none."""
        if not os.path.isdir(self._bucket_dir):
            return None
        atol = None
        # Trying the exact key first keeps an unchanged asset from ever paying for the scan.
        for filename in sorted(os.listdir(self._bucket_dir), key=lambda name: name != self._filename):
            try:
                with open(os.path.join(self._bucket_dir, filename), "rb") as file:
                    verts_ref = serialization.read_record(file)
                    # Vertices equal bit for bit, as those of an unchanged asset are, spare the reductions of the
                    # tolerance check, which cost far more than the rest of a load.
                    if not np.array_equal(self._verts, verts_ref):
                        if atol is None:
                            verts_extent = self._verts.max(axis=0) - self._verts.min(axis=0)
                            atol = MESH_CACHE_MATCH_RTOL * np.linalg.norm(verts_extent)
                        if verts_ref.shape != self._verts.shape or (np.abs(self._verts - verts_ref) > atol).any():
                            continue
                    return serialization.read_record(file)
            except (ValueError, MemoryError):
                gs.logger.info("Ignoring corrupted cache.")
        return None

    def save(self, payload) -> None:
        """Store the payload as the entry of this mesh. It must not be None, which 'load' reserves for a miss."""
        assert payload is not None
        os.makedirs(self._bucket_dir, exist_ok=True)
        with open(os.path.join(self._bucket_dir, self._filename), "wb") as file:
            serialization.write_record(file, self._verts)
            serialization.write_record(file, payload)
