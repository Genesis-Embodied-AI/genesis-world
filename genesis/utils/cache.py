"""Caches of Genesis: where they live on disk, how their keys are computed, and the in-memory caches shared across
scenes."""

import dataclasses
import functools
import hashlib
import marshal
import os
import pickle as pkl
import sys
import types
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

import genesis as gs

# Largest coordinate discrepancy, as a fraction of the bounding-box diagonal, under which two meshes are considered
# identical by the on-disk caches. Orders of magnitude below the spacing of genuinely distinct geometry, and orders of
# magnitude above the last-significant-digit noise separating two exports of the same model.
MESH_CACHE_MATCH_RTOL = 1e-6
# Bumped when watertighten output changes for a fixed (mesh, aggressiveness): forces a cache miss on stale entries.
WT_CACHE_VERSION = 7
# Bumped when the wall-thickness estimate changes for a fixed (mesh, quantile): forces a cache miss on stale entries.
WTH_CACHE_VERSION = 2
# Bumped when the particles sampled for a fixed key change or when the layout of an entry does: forces a cache miss on
# stale entries.
PTC_CACHE_VERSION = 2


def get_cache_dir():
    cache_dir = os.environ.get("GS_CACHE_FILE_PATH")
    if cache_dir is not None:
        return cache_dir
    root_cache_dir = None
    if sys.platform == "linux":
        root_cache_dir = os.environ.get("XDG_CACHE_HOME")
    if root_cache_dir is None:
        root_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root_cache_dir, "genesis")


def get_gsd_cache_dir():
    return os.path.join(get_cache_dir(), "gsd")


def get_gnd_cache_dir():
    return os.path.join(get_cache_dir(), "terrain")


def get_cvx_cache_dir():
    return os.path.join(get_cache_dir(), "cvx")


def get_ptc_cache_dir():
    return os.path.join(get_cache_dir(), "ptc")


def get_fps_pc_cache_dir():
    return os.path.join(get_cache_dir(), "fps_pc")


def get_tet_cache_dir():
    return os.path.join(get_cache_dir(), "tet")


def get_gel_cache_dir():
    return os.path.join(get_cache_dir(), "gel")


def get_remesh_cache_dir():
    return os.path.join(get_cache_dir(), "rm")


def get_wt_cache_dir():
    return os.path.join(get_cache_dir(), "wt")


def get_wth_cache_dir():
    return os.path.join(get_cache_dir(), "wth")


def get_exr_cache_dir():
    return os.path.join(get_cache_dir(), "exr")


def get_usd_cache_dir():
    return os.path.join(get_cache_dir(), "usd")


_CLEARABLE_CACHES: list[Callable[[], None]] = []


def register_cache_clear(cache_clear: Callable[[], None]) -> None:
    """Register a callback that drops a module-level cache, invoked by clear_caches on genesis teardown.

    Pass the cache's own clearing method, e.g. the cache_clear of a functools.lru_cache or the clear of a manual dict
    cache. This lets module-level asset caches (parsed meshes, baked textures, ...) release the large arrays they hold
    for destroyed scenes without the teardown path having to know about each one. A bound method is held through a weak
    reference, so that registering a cache never keeps it alive.
    """
    _CLEARABLE_CACHES.append(
        weakref.WeakMethod(cache_clear) if isinstance(cache_clear, types.MethodType) else cache_clear
    )


def clear_caches() -> None:
    """Drop every cache registered through register_cache_clear, the ones already collected aside."""
    for cache_clear in _CLEARABLE_CACHES:
        if isinstance(cache_clear, weakref.WeakMethod):
            cache_clear = cache_clear()
        if cache_clear is not None:
            cache_clear()


class SizeCappedCache:
    """An LRU cache bounded by the total byte footprint of its values rather than by their count.

    Each value is stored together with an explicit size in bytes; once the running total exceeds max_bytes the
    least-recently-used entries are evicted until it fits again (the most-recent entry is always kept). This suits
    caching a handful of large, scene-independent arrays - such as processed collision geometry - where the number of
    distinct entries is a poor proxy for the memory actually held. An optional max_entries also caps the entry count,
    so values whose reported size is small or zero (e.g. flat-color textures) cannot accumulate without bound. It
    registers itself with register_cache_clear so it is dropped together with the other asset caches on genesis
    teardown.
    """

    def __init__(self, max_bytes: int, max_entries: "int | None" = None) -> None:
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._store: "OrderedDict[Any, tuple[Any, int]]" = OrderedDict()
        self._total_bytes = 0
        register_cache_clear(self.clear)

    def get(self, key: Any) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return None
        self._store.move_to_end(key)
        return entry[0]

    def put(self, key: Any, value: Any, n_bytes: int) -> None:
        previous = self._store.pop(key, None)
        if previous is not None:
            self._total_bytes -= previous[1]
        self._store[key] = (value, n_bytes)
        self._total_bytes += n_bytes
        while len(self._store) > 1 and (
            self._total_bytes > self._max_bytes
            or (self._max_entries is not None and len(self._store) > self._max_entries)
        ):
            _, (_, evicted_bytes) = self._store.popitem(last=False)
            self._total_bytes -= evicted_bytes

    def clear(self) -> None:
        self._store.clear()
        self._total_bytes = 0


# Whether the entries of the owned caches are dropped once their last owner goes (see 'disable_owned_cache_release')
_is_owned_cache_release_enabled = True
# Every owned cache, for 'enable_owned_cache_release' to drop the entries left without owner meanwhile
_OWNED_CACHES: weakref.WeakSet["OwnedCache"] = weakref.WeakSet()


def disable_owned_cache_release() -> None:
    """Keep the entries of the owned caches whose last owner goes, until 'enable_owned_cache_release' is called.

    Rebuilding a scene in place destroys the objects owning its cached data, its meshes among them, before the rebuilt
    scene asks for the same data again. Disabling the release around the rebuild keeps that data instead of computing
    it anew, the way 'gc.disable' holds off the cyclic garbage collector.
    """
    global _is_owned_cache_release_enabled
    _is_owned_cache_release_enabled = False


def enable_owned_cache_release() -> None:
    """Drop the entries of the owned caches that no object owns any more, and the next ones as their last owner goes."""
    global _is_owned_cache_release_enabled
    _is_owned_cache_release_enabled = True
    for cache in _OWNED_CACHES:
        for key in [key for key, entry in cache._entries.items() if entry.n_owners == 0]:
            del cache._entries[key]


def is_owned_cache_release_enabled() -> bool:
    """Whether the entries of the owned caches are dropped once their last owner goes.

    See 'disable_owned_cache_release'.
    """
    return _is_owned_cache_release_enabled


@dataclasses.dataclass
class OwnedCacheEntry:
    """Value of an entry of an 'OwnedCache', with the number of objects that own it."""

    value: Any
    n_owners: int


class OwnedCache:
    """Cache whose entries live as long as any of the objects that requested them.

    The object requesting an entry becomes one of its owners, and the entry is dropped once the last of its owners is
    garbage collected or gives it up, unless the release is disabled (see 'disable_owned_cache_release'). Objects of
    several scenes asking for one entry share it, whichever scene goes first, and an entry never outlives the objects
    using it, which bounds the memory it holds without evicting anything still in use. It registers itself with
    register_cache_clear so it is dropped together with the other caches on genesis teardown.
    """

    def __init__(self) -> None:
        self._entries: dict[Any, OwnedCacheEntry] = {}
        # The entries each owner holds, by a weak reference to the owner, which compares equal to any other weak
        # reference to it while it lives
        self._owners_entries: dict[weakref.ReferenceType, dict[Any, OwnedCacheEntry]] = {}
        register_cache_clear(self.clear)
        _OWNED_CACHES.add(self)

    def get(self, owner: Any, key: Any, compute: Callable[[], Any]) -> Any:
        """Return the value of an entry, computed if no entry holds it yet, and make the object asking for it own it.

        'key' identifies the entry, 'compute' computes its value, and 'owner' asks for it (see 'own').
        """
        value = self.find(key)
        return self.own(owner, key, compute() if value is None else value)

    def own(self, owner: Any, key: Any, value: Any) -> Any:
        """Make an object own an entry, holding a given value if no entry holds one yet, and return its value.

        'key' identifies the entry and 'owner' owns it, the entry holding 'value' unless it already holds another one.
        'owner' must support weak references and must not be referenced by the value, which would keep it alive.
        """
        owner_ref = weakref.ref(owner)
        owner_entries = self._owners_entries.get(owner_ref)
        if owner_entries is None:
            # The callback reaches the cache through a weak reference, so that a pending callback never keeps it alive
            owner_ref = weakref.ref(owner, functools.partial(_release_collected_owner, weakref.ref(self)))
            owner_entries = {}
            self._owners_entries[owner_ref] = owner_entries
        entry = owner_entries.get(key)
        if entry is None:
            entry = self._entries.get(key)
            if entry is None:
                entry = OwnedCacheEntry(value, 0)
            entry.n_owners += 1
            owner_entries[key] = entry
            # A collection running since the lookup may have dropped the entry along with its last other owner
            self._entries[key] = entry
        return entry.value

    def find(self, key: Any) -> Any:
        """Return the value of an entry, or None if no entry holds it, without owning it."""
        entry = self._entries.get(key)
        return None if entry is None else entry.value

    def release(self, owner: Any) -> None:
        """Give up the entries an object owns, dropping those it was the last owner of.

        'owner' may ask for entries again afterwards.
        """
        _release_owned_entries(self._entries, self._owners_entries.get(weakref.ref(owner), {}))

    def clear(self) -> None:
        """Drop every entry, whoever owns it."""
        self._entries.clear()
        self._owners_entries.clear()


def _release_owned_entries(entries: dict[Any, OwnedCacheEntry], owner_entries: dict[Any, OwnedCacheEntry]) -> None:
    """Give up the entries an owner of an 'OwnedCache' holds, dropping those it was the last owner of."""
    for key, entry in owner_entries.items():
        entry.n_owners -= 1
        # The entry stored under the key may be another one, the cache having been cleared since
        if entry.n_owners == 0 and _is_owned_cache_release_enabled and entries.get(key) is entry:
            del entries[key]
    owner_entries.clear()


def _release_collected_owner(cache_ref: weakref.ReferenceType[OwnedCache], owner_ref: weakref.ReferenceType) -> None:
    """Give up the entries an owner of an 'OwnedCache' held, once it is garbage collected."""
    cache = cache_ref()
    if cache is not None:
        _release_owned_entries(cache._entries, cache._owners_entries.pop(owner_ref, {}))


def get_gsd_path(verts, faces, sdf_res, sdf_cell_size):
    # The mesh, the resolution and the cell size determine the grid. The schema tag is bumped whenever the layout of a
    # file changes, which forces a cache miss on stale entries.
    schema = "v4-sdf-grid"
    hashkey = get_hashkey(verts, faces, sdf_res, sdf_cell_size, schema)
    return os.path.join(get_gsd_cache_dir(), f"{hashkey}.gsd")


def get_gnd_path(
    name, subterrain_types, subterrain_size, horizontal_scale, vertical_scale, n_subterrains, subterrain_parameters
):
    hashkey = get_hashkey(
        name, subterrain_types, subterrain_size, horizontal_scale, vertical_scale, n_subterrains, subterrain_parameters
    )
    return os.path.join(get_gnd_cache_dir(), f"{hashkey}.gnd")


def get_cvx_cache(verts, faces, coacd_options):
    return MeshCache(get_cvx_cache_dir(), "cvx", verts, faces, coacd_options.__dict__)


def get_ptc_cache(verts, faces, p_size, sampler, random_state):
    return MeshCache(get_ptc_cache_dir(), "ptc", verts, faces, p_size, sampler, random_state, PTC_CACHE_VERSION)


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
    # Containers are replaced by the tuples of their items, sorted for sets and for dicts other than OrderedDict since
    # equal ones may iterate in different orders (string hashes are salted per process). Items are replaced before their
    # container, so that sorting compares canonical representations, which order items of any type.
    values, nodes = [], [(args, False)]
    while nodes:
        node, is_expanded = nodes.pop()
        if is_expanded:
            items = [values.pop() for _ in range(len(node))]
            if isinstance(node, (set, frozenset)) or (isinstance(node, dict) and not isinstance(node, OrderedDict)):
                items.sort(key=repr)
            values.append(tuple(items))
        elif isinstance(node, (dict, list, tuple, set, frozenset)):
            nodes.append((node, True))
            nodes.extend((item, False) for item in (node.items() if isinstance(node, dict) else node))
        else:
            values.append(node)

    hasher = hashlib.sha256()
    for arg in (*values[0], gs.__version__.encode()):
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
