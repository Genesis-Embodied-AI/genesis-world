"""
Furthest point sampling (FPS) on triangle mesh surfaces.
"""

import numpy as np
import trimesh

import genesis as gs

from . import mesh as msu
from .misc import get_fps_pc_cache_dir


def get_fps_pc_cache(verts, faces, n_points: int, n_candidates: int, return_normals: bool, seed: int):
    return msu.MeshCache(get_fps_pc_cache_dir(), "fps_pc", verts, faces, n_points, n_candidates, return_normals, seed)


def _furthest_point_sample_impl(
    points: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None,
    *,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    if n_samples == 0:
        empty_pts = np.zeros((0, 3), dtype=gs.np_float)
        if return_indices:
            return empty_pts, np.zeros((0,), dtype=np.int32)
        return empty_pts

    first_idx = int(rng.integers(0, n)) if rng is not None else 0

    selected = np.empty((n_samples,), dtype=np.int32)
    selected[0] = first_idx
    min_dist_sq = np.sum((points - points[first_idx]) ** 2, axis=1)

    for k in range(1, n_samples):
        next_idx = int(np.argmax(min_dist_sq))
        selected[k] = next_idx
        d2 = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, d2)

    out_pts = points[selected].astype(gs.np_float, copy=False)
    if return_indices:
        return out_pts, selected
    return out_pts


def furthest_point_sample(points: np.ndarray, n_samples: int, *, seed: int | None = None) -> np.ndarray:
    """
    Greedy furthest-point sampling on a finite set of 3D points.

    Parameters
    ----------
    points : np.ndarray
        Candidate positions, shape (N, 3).
    n_samples : int
        Number of points to select. Must satisfy 0 <= n_samples <= N.
    seed : int, optional
        If given, the first point index is chosen uniformly at random with this
        RNG seed; otherwise the first point is index 0.

    Returns
    -------
    np.ndarray
        Selected points, shape (n_samples, 3), dtype ``gs.np_float``.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        gs.raise_exception("points must have shape (N, 3).")
    n = points.shape[0]
    if n_samples < 0 or n_samples > n:
        gs.raise_exception(f"n_samples must be in [0, N] with N={n}, got n_samples={n_samples}.")
    rng = np.random.default_rng(seed) if seed is not None else None
    return _furthest_point_sample_impl(points, n_samples, rng, return_indices=False)


def sample_mesh_point_cloud(
    verts: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    *,
    n_candidates: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
    return_normals: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Sample ``n_points`` points on the mesh surface using FPS in mesh local coordinates.

    Parameters
    ----------
    verts : np.ndarray
        Vertex positions, shape (V, 3).
    faces : np.ndarray
        Triangle indices, shape (F, 3).
    n_points : int
        Number of output points.
    n_candidates : int, optional
        Number of area-weighted surface samples to draw before FPS.
    seed : int, optional
        The random seed for `trimesh.sample.sample_surface` and the first FPS pick. If None, a random seed is generated.
    use_cache : bool
        If True, load and store results under the cache directory.
    return_normals : bool
        If True, also return triangle face normals aligned with each FPS sample (same candidate face as the point).

    Returns
    -------
    points: np.ndarray
        Positions of shape (n_points, 3), dtype ``gs.np_float``.
    normals: np.ndarray, optional
        Normals of shape (n_points, 3), dtype ``gs.np_float``. Only returned if ``return_normals`` is True.
    """
    if faces.ndim != 2 or faces.shape[1] != 3:
        gs.raise_exception("faces must have shape (F, 3).")
    if faces.shape[0] == 0:
        gs.raise_exception("Mesh has no faces.")

    if n_candidates is None:
        n_candidates = max(256, n_points * 4)
    n_candidates = max(n_candidates, n_points)

    if seed is None:
        seed = int(np.random.SeedSequence().entropy)

    cache = get_fps_pc_cache(verts, faces, n_points, n_candidates, return_normals, seed)

    if use_cache:
        data = cache.load()
        if data is not None:
            gs.logger.debug("FPS point cloud cache file found.")
            if return_normals:
                pts, nrm = data
                return pts.astype(gs.np_float, copy=False), nrm.astype(gs.np_float, copy=False)
            return data.astype(gs.np_float, copy=False)

    tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    candidates, face_idx = trimesh.sample.sample_surface(tmesh, n_candidates, seed=seed)
    candidates = np.asarray(candidates, dtype=np.float32)

    fps_rng = np.random.default_rng(seed)
    if return_normals:
        candidate_normals = np.asarray(tmesh.face_normals[face_idx], dtype=np.float32)
        points, selected = _furthest_point_sample_impl(candidates, n_points, fps_rng, return_indices=True)
        normals = candidate_normals[selected].astype(gs.np_float, copy=False)
        output = (points, normals)
    else:
        output = _furthest_point_sample_impl(candidates, n_points, fps_rng, return_indices=False)

    if use_cache:
        cache.save(output)

    return output
