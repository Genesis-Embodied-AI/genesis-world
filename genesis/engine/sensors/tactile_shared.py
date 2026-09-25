import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, NamedTuple, TypeVar

import numpy as np
import quadrants as qd
import torch
import torch.nn.functional as F

import genesis as gs
import genesis.utils.geom as gu
from genesis.options.sensors.options import ProbesWithNormalSensorOptionsMixin
from genesis.utils.misc import concat_with_tensor, gaussian_crosstalk_kernel, make_tensor_field

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


_GRID_TOL = 1.0e-5  # Tolerance for grid-regularity / orthogonality / normal-uniformity checks.


def next_pow2(n: int) -> int:
    """
    Smallest power of 2 >= ``n`` (1 if ``n == 0``).
    """
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


# ==================== BVH helpers (shared by point-cloud and triangle-mesh sensors) ====================


BVH_LEAF_SIZE = 8
BVH_STACK_SIZE = 32


def get_mesh_geom_chunks(link, prefer_visual: bool) -> list[tuple[object, np.ndarray, np.ndarray]]:
    """
    Return per-geom mesh chunks ``(geom, verts_link, faces)`` in link-local frame.

    ``prefer_visual`` picks vgeoms over geoms when both exist; falls back to the other type when the
    preferred one is absent. Empty meshes are dropped from the list.
    """
    if prefer_visual:
        geoms = list(link.vgeoms) if link.vgeoms else list(link.geoms)
        use_vverts = bool(link.vgeoms)
    else:
        geoms = list(link.geoms) if link.geoms else list(link.vgeoms)
        use_vverts = not bool(link.geoms) and bool(link.vgeoms)

    chunks: list[tuple[object, np.ndarray, np.ndarray]] = []
    for geom in geoms:
        # init_*verts / init_*faces come from loaded mesh data whose dtype is not under our control, so coerce here.
        if use_vverts:
            verts = np.asarray(geom.init_vverts, dtype=gs.np_float)
            faces = np.asarray(geom.init_vfaces, dtype=gs.np_int)
        else:
            verts = np.asarray(geom.init_verts, dtype=gs.np_float)
            faces = np.asarray(geom.init_faces, dtype=gs.np_int)
        if verts.size == 0 or faces.size == 0:
            continue
        verts_link = gu.transform_by_trans_quat(verts, geom.init_pos, geom.init_quat)
        chunks.append((geom, verts_link.astype(gs.np_float, copy=False), faces))
    return chunks


def build_static_chunk_bvh(
    centroids: np.ndarray, aabb_mins: np.ndarray, aabb_maxs: np.ndarray, global_rows: np.ndarray, leaf_size: int
) -> tuple[np.ndarray, ...]:
    """
    Median-split AABB BVH over a static set of elements (points, triangles, etc.) in link-local frame.

    Split decisions use ``centroids`` along the longest-spread axis; node AABBs union the per-element
    ``aabb_mins``/``aabb_maxs``. For point-cloud BVHs, callers pass ``centroids == aabb_mins == aabb_maxs``
    (the points themselves); for triangle BVHs, callers pass per-triangle centroid + min/max bounds.

    Leaves carry the caller-provided ``global_rows`` (absolute rows into the sensor-class element table);
    the kernel indexes directly into that table with no extra indirection. Internal nodes use -1 for
    ``node_left`` / ``node_right``. Returns ``(node_min, node_max, node_left, node_right, node_elem_start,
    node_elem_n, elem_idx)``.
    """
    node_min: list[np.ndarray] = []
    node_max: list[np.ndarray] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_elem_start: list[int] = []
    node_elem_n: list[int] = []
    elem_idx: list[int] = []

    def _alloc() -> int:
        i = len(node_min)
        node_min.append(np.zeros(3, dtype=gs.np_float))
        node_max.append(np.zeros(3, dtype=gs.np_float))
        node_left.append(-1)
        node_right.append(-1)
        node_elem_start.append(-1)
        node_elem_n.append(0)
        return i

    def _build(rows: np.ndarray, cents: np.ndarray, a_mins: np.ndarray, a_maxs: np.ndarray) -> int:
        nid = _alloc()
        bmin = a_mins.min(axis=0).astype(gs.np_float)
        bmax = a_maxs.max(axis=0).astype(gs.np_float)
        node_min[nid] = bmin
        node_max[nid] = bmax
        if rows.shape[0] <= leaf_size:
            start = len(elem_idx)
            elem_idx.extend(int(r) for r in rows)
            node_elem_start[nid] = start
            node_elem_n[nid] = int(rows.shape[0])
            return nid
        axis = int(np.argmax(bmax - bmin))
        order = np.argsort(cents[:, axis], kind="stable")
        mid = order.shape[0] // 2
        node_left[nid] = _build(rows[order[:mid]], cents[order[:mid]], a_mins[order[:mid]], a_maxs[order[:mid]])
        node_right[nid] = _build(rows[order[mid:]], cents[order[mid:]], a_mins[order[mid:]], a_maxs[order[mid:]])
        return nid

    if centroids.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=gs.np_float),
            np.zeros((0, 3), dtype=gs.np_float),
            np.zeros((0,), dtype=gs.np_int),
            np.zeros((0,), dtype=gs.np_int),
            np.zeros((0,), dtype=gs.np_int),
            np.zeros((0,), dtype=gs.np_int),
            np.zeros((0,), dtype=gs.np_int),
        )

    root = _build(
        global_rows.astype(gs.np_int, copy=False),
        centroids.astype(gs.np_float, copy=False),
        aabb_mins.astype(gs.np_float, copy=False),
        aabb_maxs.astype(gs.np_float, copy=False),
    )
    assert root == 0
    return (
        np.stack(node_min, axis=0),
        np.stack(node_max, axis=0),
        np.asarray(node_left, dtype=gs.np_int),
        np.asarray(node_right, dtype=gs.np_int),
        np.asarray(node_elem_start, dtype=gs.np_int),
        np.asarray(node_elem_n, dtype=gs.np_int),
        np.asarray(elem_idx, dtype=gs.np_int),
    )


@qd.func
def func_vec3_at(i: int, values: qd.types.ndarray()) -> qd.types.vector(3):
    return qd.Vector([values[i, 0], values[i, 1], values[i, 2]], dt=float)


@qd.func
def func_sphere_intersects_aabb(center, radius_sq, bmin, bmax):  # -> bool
    """
    Squared-distance sphere-vs-AABB test: True iff the closest AABB point to ``center`` is within ``radius_sq``.

    Reused as a closest-point cull by passing ``radius_sq = current_best_dist_sq``.
    """
    d_sq = gs.qd_float(0.0)
    for k in qd.static(range(3)):
        v = center[k]
        lo = bmin[k]
        hi = bmax[k]
        if v < lo:
            d = lo - v
            d_sq = d_sq + d * d
        elif v > hi:
            d = v - hi
            d_sq = d_sq + d * d
    return d_sq <= radius_sq


@qd.func
def func_aabb_intersects_aabb(amin, amax, bmin, bmax):  # -> bool
    """
    Standard 6-axis AABB-vs-AABB overlap test.
    """
    return (
        amin[0] <= bmax[0]
        and amax[0] >= bmin[0]
        and amin[1] <= bmax[1]
        and amax[1] >= bmin[1]
        and amin[2] <= bmax[2]
        and amax[2] >= bmin[2]
    )


@dataclass(eq=True, kw_only=False, frozen=True)
class ChunkedBVHData:
    """
    Bundle of the flat ``BVHMetadata`` scaffolding tensors passed to a traversal kernel as one argument.

    See ``BVHMetadata`` for the field semantics. Element payload tables (``tri_verts``, point-cloud
    positions, ...) stay separate kernel arguments because they differ per sensor class.
    """

    sensor_chunk_start: qd.types.ndarray()
    sensor_chunk_count: qd.types.ndarray()
    chunk_link_idx: qd.types.ndarray()
    chunk_node_start: qd.types.ndarray()
    node_min: qd.types.ndarray()
    node_max: qd.types.ndarray()
    node_left: qd.types.ndarray()
    node_right: qd.types.ndarray()
    node_leaf_start: qd.types.ndarray()
    node_leaf_count: qd.types.ndarray()
    leaf_elem_idx: qd.types.ndarray()


@dataclass
class BVHMetadata:
    """
    Element-agnostic scaffolding for a static, link-local, chunked AABB BVH shared across one sensor class.

    One *chunk* per (sensor, tracked_link): each chunk is a small subtree built once at scene init in the
    tracked link's local frame and never rebuilt. Subclasses (PointCloudBVH, TriangleMeshBVH) layer on
    element-specific payload tables; ``leaf_elem_idx`` entries are absolute rows into those tables.

    Per-sensor slice into the chunk arrays:
        ``chunks[sensor_chunk_start[s] : sensor_chunk_start[s] + sensor_chunk_count[s]]``
    Per-chunk slice into the flat node arrays:
        ``nodes[chunk_node_start[c] : chunk_node_start[c] + chunk_node_count[c]]``
    Per-leaf slice into ``leaf_elem_idx``:
        ``leaf_elem_idx[node_leaf_start[n] : node_leaf_start[n] + node_leaf_count[n]]``
    ``node_left == -1`` marks a leaf; otherwise ``node_left``/``node_right`` are absolute child indices.
    """

    sensor_chunk_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_chunk_count: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    chunk_link_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    chunk_node_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    chunk_node_count: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    node_min: torch.Tensor = make_tensor_field((0, 3))
    node_max: torch.Tensor = make_tensor_field((0, 3))
    node_left: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    node_right: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    node_leaf_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    node_leaf_count: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    leaf_elem_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    # Cached scaffolding bundle for kernel calls, built once on first use (the BVH is static after scene init).
    _kernel_bvh: "ChunkedBVHData | None" = field(default=None, init=False, compare=False, repr=False)

    @property
    def kernel_bvh(self) -> ChunkedBVHData:
        """
        The scaffolding fields bundled into a single ``ChunkedBVHData`` traversal-kernel argument.
        """
        if self._kernel_bvh is None:
            self._kernel_bvh = ChunkedBVHData(
                sensor_chunk_start=self.sensor_chunk_start,
                sensor_chunk_count=self.sensor_chunk_count,
                chunk_link_idx=self.chunk_link_idx,
                chunk_node_start=self.chunk_node_start,
                node_min=self.node_min,
                node_max=self.node_max,
                node_left=self.node_left,
                node_right=self.node_right,
                node_leaf_start=self.node_leaf_start,
                node_leaf_count=self.node_leaf_count,
                leaf_elem_idx=self.leaf_elem_idx,
            )
        return self._kernel_bvh


# ============================ FFT helpers ============================


@dataclass
class GridFFTConvMetadataMixin:
    """
    Shared per-sensor-class state for the per-grid 2D-FFT convolution passes.

    Attributes
    ----------
    grid_fft_meta : list of NamedTuple
        Per-grid-FFT-sensor metadata records. The leading 5 fields are always
        ``(sensor_idx, g_ny, g_nx, probe_start, cache_start)``; sensors append their kernel params after that
        (e.g. ``GridFFTMeta`` for HydroShear dilation).
    grid_fft_max_n : (int, int)
        Global FFT size ``(fft_ny, fft_nx)``, the elementwise max over all registered grid sensors. Build-time only.
    grid_fft_kernels_stacked : torch.Tensor
        Stacked complex ``rfft2`` kernels (half spectrum), shape ``(n_grid, n_planes, fft_ny, fft_nx // 2 + 1)``.
        Recomputed when the FFT size grows.
    grid_fft_buffer : torch.Tensor
        Reused per-step real buffer: ``(B, n_grid, n_channels, fft_ny, fft_nx)`` when registered with
        ``n_buffer_channels > 0``, else ``(B, n_grid, fft_ny, fft_nx)``. Reallocated on each registration.
    any_grid_fft : bool
        Python fast-path flag; True iff at least one grid-FFT sensor is registered.
    """

    grid_fft_meta: list[NamedTuple] = field(default_factory=list)
    grid_fft_max_n: tuple[int, int] = (0, 0)
    grid_fft_kernels_stacked: torch.Tensor = make_tensor_field((0, 0, 0, 0), dtype_factory=lambda: torch.complex64)
    grid_fft_buffer: torch.Tensor = make_tensor_field((0, 0, 0, 0))
    any_grid_fft: bool = False


def register_grid_fft_sensor(
    metadata: GridFFTConvMetadataMixin,
    meta_entry: NamedTuple,
    this_fft_n: tuple[int, int],
    kernel_builder: Callable[[NamedTuple, tuple[int, int]], torch.Tensor],
    n_buffer_channels: int,
    batch_size: int,
) -> None:
    """
    Register one grid-shaped sensor for FFT convolution; (re)build the stacked kernels and the per-step buffer.

    Parameters
    ----------
    meta_entry : NamedTuple
        Metadata record appended to ``grid_fft_meta``; its leading 5 fields must be
        ``(sensor_idx, g_ny, g_nx, probe_start, cache_start)``, followed by any sensor-specific kernel params.
    this_fft_n : (int, int)
        The ``(ny, nx)`` FFT size this sensor needs. The shared ``grid_fft_max_n`` grows to the elementwise max;
        when it grows, every prior sensor's kernel is recomputed at the new size (frequency-domain padding is not
        equivalent to spatial zero-padding).
    kernel_builder : callable
        ``kernel_builder(meta_entry, fft_n) -> (n_planes, fft_ny, fft_nx // 2 + 1)`` complex tensor (an ``rfft2``
        half spectrum). Must be deterministic from the meta record, since it is re-invoked whenever the FFT size grows.
    n_buffer_channels : int
        When ``> 0``, allocate a 5D ``(B, n_grid, n_buffer_channels, ny, nx)`` per-step buffer; else a 4D
        ``(B, n_grid, ny, nx)`` one.
    """
    metadata.grid_fft_meta.append(meta_entry)
    cur = metadata.grid_fft_max_n
    new_n = (max(cur[0], this_fft_n[0]), max(cur[1], this_fft_n[1]))
    metadata.grid_fft_max_n = new_n
    n_grid = len(metadata.grid_fft_meta)
    metadata.grid_fft_kernels_stacked = torch.stack([kernel_builder(m, new_n) for m in metadata.grid_fft_meta], dim=0)
    buffer_shape = (
        (batch_size, n_grid, n_buffer_channels, new_n[0], new_n[1])
        if n_buffer_channels > 0
        else (batch_size, n_grid, new_n[0], new_n[1])
    )
    metadata.grid_fft_buffer = torch.zeros(buffer_shape, dtype=gs.tc_float, device=gs.device)
    metadata.any_grid_fft = True


def expand_probe_normals(normals: np.ndarray, n_probes: int, probe_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast ``normals`` to a flat ``(n_probes, 3)`` array.

    Accepts a single shared normal of shape ``(3,)``, a grid-shaped ``(*probe_shape, 3)`` array, or an already-flat
    ``(n_probes, 3)``. Any other shape raises.
    """
    normals = np.asarray(normals, dtype=gs.np_float)
    if normals.ndim == 1:
        return np.broadcast_to(normals, (n_probes, 3)).copy()
    if normals.shape == (*probe_shape, 3):
        return normals.reshape(n_probes, 3).copy()
    if normals.shape == (n_probes, 3):
        return normals.copy()
    gs.raise_exception(
        "probe_local_normal must be one normal or match probe_local_pos shape. "
        f"Got normal shape {normals.shape} for probe shape {probe_shape}."
    )


def normalize_grid_probe_layout(
    probe_pos: np.ndarray, probe_normals: np.ndarray, is_grid: bool
) -> tuple[np.ndarray, np.ndarray, bool, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate a probe layout and extract grid-FFT metadata when the layout qualifies.

    Returns ``(flat_positions, flat_normals, use_grid_fft, is_grid_regular, grid_normal, tangent_u, tangent_v,
    grid_spacing)``.

    ``use_grid_fft`` is True when the layout has shape ``(ny, nx, 3)`` with ``ny, nx >= 2`` and non-degenerate
    spacing along both axes -- the FFT path is usable and the grid metadata is populated as a best-fit
    approximation (average step vectors over all adjacent pairs, average unit normal over all probes).

    ``is_grid_regular`` is True when, in addition, the layout is strictly regular: normals uniform within
    tolerance, tangents orthogonal, both tangents in the plane perpendicular to the normal, and all probes lie
    on the regular ``(spacing_u, spacing_v)`` rectangle implied by the averaged steps. Callers that proceed
    with FFT on an irregular layout (``use_grid_fft`` and not ``is_grid_regular``) should warn the user.

    When ``use_grid_fft`` is False, the tangent / spacing / normal entries are zero.
    """
    probe_shape = probe_pos.shape[:-1]
    flat = probe_pos.reshape(-1, 3)
    # ``probe_normals is None`` means the sensor carries no per-probe normal (e.g. KinematicTaxel): derive the grid's
    # plane normal from its tangents below. The only such caller is spatial crosstalk, which always runs on a regular
    # planar grid, so the geometric plane normal is exact.
    derive_normal = probe_normals is None
    if derive_normal:
        normals = np.zeros((flat.shape[0], 3), dtype=gs.np_float)
    else:
        normals = expand_probe_normals(probe_normals, flat.shape[0], probe_shape)
        normal_norms = np.linalg.norm(normals, axis=1)
        if np.any(normal_norms < gs.EPS):
            gs.raise_exception("probe_local_normal entries must be non-zero.")
        normals = normals / normal_norms[:, None]

    use_grid_fft = False
    is_grid_regular = False
    grid_normal = np.zeros(3, dtype=gs.np_float)
    tangent_u = np.zeros(3, dtype=gs.np_float)
    tangent_v = np.zeros(3, dtype=gs.np_float)
    grid_spacing = np.zeros(2, dtype=gs.np_float)

    if is_grid:
        if len(probe_shape) != 2:
            gs.raise_exception("Grid probe_local_pos must have shape (ny, nx, 3).")
        ny, nx = int(probe_shape[0]), int(probe_shape[1])
        if nx >= 2 and ny >= 2:
            grid = probe_pos.reshape(ny, nx, 3)
            # Averaged step vectors across all adjacent pairs along each axis -- robust to local jitter.
            avg_step_u = (grid[:, 1:, :] - grid[:, :-1, :]).reshape(-1, 3).mean(axis=0)
            avg_step_v = (grid[1:, :, :] - grid[:-1, :, :]).reshape(-1, 3).mean(axis=0)
            spacing_u = float(np.linalg.norm(avg_step_u))
            spacing_v = float(np.linalg.norm(avg_step_v))
            if spacing_u >= gs.EPS and spacing_v >= gs.EPS:
                tangent_u_candidate = (avg_step_u / spacing_u).astype(gs.np_float)
                tangent_v_candidate = (avg_step_v / spacing_v).astype(gs.np_float)
                if derive_normal:
                    # No per-probe normal supplied: the plane normal is the (unit) cross of the grid tangents.
                    # Its sign is irrelevant -- crosstalk only uses it to project force into normal vs shear.
                    cross = np.cross(tangent_u_candidate, tangent_v_candidate)
                    cross_norm = float(np.linalg.norm(cross))
                    normal_candidate = (cross / cross_norm).astype(gs.np_float) if cross_norm >= gs.EPS else cross
                    normals[:] = normal_candidate
                    normals_are_uniform = True
                else:
                    # Average unit normal across all probes. If they cancel out (e.g. opposing normals), fall back
                    # to the first probe's normal so downstream FFT still has a defined orientation.
                    avg_normal = normals.mean(axis=0)
                    normal_norm = float(np.linalg.norm(avg_normal))
                    if normal_norm < gs.EPS:
                        normal_candidate = normals[0].astype(gs.np_float, copy=False)
                    else:
                        normal_candidate = (avg_normal / normal_norm).astype(gs.np_float)
                    normals_are_uniform = bool(np.all(normals @ normal_candidate >= 1.0 - _GRID_TOL))
                axes_are_orthogonal = abs(float(tangent_u_candidate @ tangent_v_candidate)) <= _GRID_TOL
                axes_in_plane = (
                    abs(float(tangent_u_candidate @ normal_candidate)) <= _GRID_TOL
                    and abs(float(tangent_v_candidate @ normal_candidate)) <= _GRID_TOL
                )
                expected = (
                    grid[0, 0]
                    + np.arange(nx, dtype=gs.np_float)[None, :, None] * avg_step_u[None, None, :]
                    + np.arange(ny, dtype=gs.np_float)[:, None, None] * avg_step_v[None, None, :]
                )
                is_regular = bool(np.max(np.linalg.norm(grid - expected, axis=-1)) <= _GRID_TOL)

                use_grid_fft = True
                is_grid_regular = normals_are_uniform and axes_are_orthogonal and axes_in_plane and is_regular
                grid_normal = normal_candidate
                tangent_u = tangent_u_candidate
                tangent_v = tangent_v_candidate
                grid_spacing = np.array((spacing_u, spacing_v), dtype=gs.np_float)

    return (
        flat.astype(gs.np_float, copy=False),
        normals.astype(gs.np_float, copy=False),
        use_grid_fft,
        is_grid_regular,
        grid_normal.astype(gs.np_float, copy=False),
        tangent_u.astype(gs.np_float, copy=False),
        tangent_v.astype(gs.np_float, copy=False),
        grid_spacing.astype(gs.np_float, copy=False),
    )


# ============================ Contact prefilter ============================


@dataclass
class ContactPrefilterMetadataMixin:
    """
    Per-(env, sensor) prefilter buffers shared by tactile sensors whose kernels query the collider's contacts per
    probe (KinematicTaxel, ContactDepthProbe), populated each step in ``kinematic_tactile.py``:

    - ``sensor_contacts_idx`` / ``sensor_n_contacts``: compact list of contact indices whose ``link_a`` or
      ``link_b`` matches the sensor's tracked link (``_kernel_build_sensor_contact_idx``). Feeds the raycast
      path's BVH-mask builder. Shape ``(B, n_sensors, max_contacts)`` / ``(B, n_sensors)``; the per-sensor cap
      (``_MAX_CONTACTS_PER_SENSOR``) is read off ``sensor_contacts_idx.shape[2]``.
    - ``sensor_geoms_idx`` / ``sensor_n_geoms``: compact *deduplicated* list of the opposing contacting geoms
      for the same link (``_kernel_build_sensor_geom_idx``). Feeds the SDF path so each probe queries one SDF
      per distinct geom instead of one per contact point. Shape ``(B, n_sensors, max_geoms)`` /
      ``(B, n_sensors)``; the cap (``_MAX_GEOMS_PER_SENSOR``) is read off ``sensor_geoms_idx.shape[2]``.
    """

    sensor_contacts_idx: torch.Tensor = make_tensor_field((0, 0, 0), dtype_factory=lambda: gs.tc_int)
    sensor_n_contacts: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    sensor_geoms_idx: torch.Tensor = make_tensor_field((0, 0, 0), dtype_factory=lambda: gs.tc_int)
    sensor_n_geoms: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)


# ============================ Contact depth query mode (SDF vs raycast) ============================


@dataclass
class ContactDepthQueryMetadataMixin:
    """
    Shared per-sensor-class state for the contact-depth query backend.

    ``contact_depth_query`` is the resolved mode for every sensor of this class - ``"sdf"`` or ``"raycast"``. The
    backend is dispatched once per sensor class (a single kernel covers all of the class's sensors), so every sensor
    must agree: each sensor's ``build()`` calls ``resolve_contact_depth_query``, which records the first non-``None``
    mode and raises if a later sensor requests a different one. ``None`` defers to whatever another sensor set;
    ``None`` at update time falls back to ``"sdf"``.

    When mode is ``"raycast"``, the collision/visual BVHs come from the shared ``RaycastContext`` (the sensor's
    ``shared_context``), built lazily on raycast opt-in and refreshed once per step by ``SensorManager``.
    ``sensor_candidate_geom_mask`` is a per-(env, sensor, geom) bool gate - scattered per step from the contact
    prefilter (KinematicTactile family) or once at build from ``sensor_track_geom_idx`` (ElastomerTaxel) - so BVH
    leaves whose ``faces_info.geom_idx`` falls outside the mask are skipped.
    """

    contact_depth_query: str | None = None
    sensor_candidate_geom_mask: torch.Tensor = make_tensor_field((0, 0, 0), dtype_factory=lambda: gs.tc_bool)


def resolve_contact_depth_query(metadata: ContactDepthQueryMetadataMixin, mode: str | None, sensor_name: str) -> None:
    """
    Record the class-wide contact-depth backend for one sensor, rejecting a conflict with an earlier sensor.

    The backend is dispatched per sensor class, so every sensor of a class must agree on ``"sdf"`` vs ``"raycast"``.
    ``mode is None`` defers to whatever another sensor set (default ``"sdf"``); a different non-``None`` ``mode``
    raises rather than silently switching the whole class by build order.
    """
    if mode is None:
        return
    existing = metadata.contact_depth_query
    if existing is not None and existing != mode:
        gs.raise_exception(
            f"{sensor_name} sensors disagree on contact_depth_query ({existing!r} vs {mode!r}). All sensors of a "
            "tactile class share one contact-depth backend; use the same mode for every sensor of this class (build "
            "separate scenes to compare backends)."
        )
    metadata.contact_depth_query = mode


class ContactDepthQuerySensorMixin:
    """
    Sensor-side counterpart to ``ContactDepthQueryMetadataMixin``.

    Its ``build()`` resolves the class-wide contact-depth backend (raising if this class's sensors disagree) and
    activates only the chosen backend: the shared ``RaycastContext`` BVH in raycast mode, else the collider SDF.
    Mixing this in (alongside the metadata mixin) means subclasses get the resolution from the super-``build()`` chain
    and need not call ``resolve_contact_depth_query`` themselves. Requires ``_shared_metadata``,
    ``_options.contact_depth_query``, and a ``_shared_context`` ``RaycastContext``.
    """

    def build(self):
        super().build()
        resolve_contact_depth_query(self._shared_metadata, self._options.contact_depth_query, type(self).__name__)
        if self._shared_metadata.contact_depth_query == "raycast":
            self._shared_context.activate()  # builds the BVH lazily, only on raycast opt-in
        else:
            self._shared_metadata.solver.collider.activate_sdf()


# ============================ ViscoelasticHysteresis ============================


@dataclass
class ViscoelasticHysteresisMetadataMixin:
    hysteresis_strength: torch.Tensor = make_tensor_field((0,))
    hysteresis_alpha: torch.Tensor = make_tensor_field((0,))
    viscoelastic_xi: torch.Tensor = make_tensor_field((0, 0))
    viscoelastic_prev_input: torch.Tensor = make_tensor_field((0, 0))
    viscoelastic_strength_row: torch.Tensor = make_tensor_field((0,))
    viscoelastic_alpha_row: torch.Tensor = make_tensor_field((0,))
    has_any_hysteresis: bool = False


ViscoelasticHysteresisSharedMetadataT = TypeVar(
    "ViscoelasticHysteresisSharedMetadataT", bound=ViscoelasticHysteresisMetadataMixin
)


class ViscoelasticHysteresisMixin(Generic[ViscoelasticHysteresisSharedMetadataT]):
    """
    Viscoelastic hysteresis (single Maxwell element, equilibrium gain normalized to 1).

    Per simulation step::
        alpha = exp(-dt / tau)
        xi    <- alpha * xi + (x - x_prev)
        output = x + strength * xi
        x_prev <- x

    After a step input from 0 to X the output jumps to ``X * (1 + strength)`` and decays back to ``X`` with time
    constant ``tau``. On cyclic input the output overshoots on rising edges and undershoots on falling edges.
    """

    _shared_metadata: ViscoelasticHysteresisSharedMetadataT

    def build(self):
        super().build()
        # Every sensor mixing in ViscoelasticHysteresisMixin uses an options class that inherits
        # ViscoelasticHysteresisOptionsMixin, so the hysteresis fields are always declared.
        strength = float(self._options.hysteresis_strength)
        tau = float(self._options.hysteresis_tau)
        alpha = math.exp(-self._dt / tau) if tau > 0.0 else 0.0
        self._shared_metadata.hysteresis_strength = concat_with_tensor(
            self._shared_metadata.hysteresis_strength, strength, expand=(1,)
        )
        self._shared_metadata.hysteresis_alpha = concat_with_tensor(
            self._shared_metadata.hysteresis_alpha, alpha, expand=(1,)
        )
        if strength > 0.0 and tau > 0.0:
            self._shared_metadata.has_any_hysteresis = True
        # Invalidate lazy rows so they rebuild on first apply against the final cache width. Per-column state tensors
        # are allocated lazily at the same time, so sensor classes that never enable hysteresis pay no memory cost.
        self._shared_metadata.viscoelastic_strength_row = torch.empty((0,), dtype=gs.tc_float, device=gs.device)
        self._shared_metadata.viscoelastic_alpha_row = torch.empty((0,), dtype=gs.tc_float, device=gs.device)

    @classmethod
    def reset(
        cls, shared_metadata: ViscoelasticHysteresisSharedMetadataT, shared_ground_truth_cache: torch.Tensor, envs_idx
    ):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        if shared_metadata.viscoelastic_xi.numel() > 0:
            shared_metadata.viscoelastic_xi[envs_idx] = 0.0
            shared_metadata.viscoelastic_prev_input[envs_idx] = 0.0

    @classmethod
    def _apply_transform(
        cls,
        shared_metadata: ViscoelasticHysteresisSharedMetadataT,
        data: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ):
        super()._apply_transform(shared_metadata, data, timeline, is_measured=is_measured)
        if not is_measured or not shared_metadata.has_any_hysteresis:
            return

        B, n_cols, *_ = data.shape
        # Lazily build the per-cache-column strength/alpha rows and state buffers
        if shared_metadata.viscoelastic_strength_row.shape != (n_cols,):
            sensor_col_idx = []
            for i_s, size in enumerate(shared_metadata.cache_sizes):
                sensor_col_idx.extend([i_s] * size)
            idx_t = torch.tensor(sensor_col_idx, dtype=torch.long, device=gs.device)
            shared_metadata.viscoelastic_strength_row = shared_metadata.hysteresis_strength[idx_t].to(dtype=data.dtype)
            shared_metadata.viscoelastic_alpha_row = shared_metadata.hysteresis_alpha[idx_t].to(dtype=data.dtype)
            shared_metadata.viscoelastic_xi = torch.zeros((B, n_cols), dtype=data.dtype, device=gs.device)
            shared_metadata.viscoelastic_prev_input = torch.zeros((B, n_cols), dtype=data.dtype, device=gs.device)

        xi = shared_metadata.viscoelastic_xi
        prev = shared_metadata.viscoelastic_prev_input
        xi.mul_(shared_metadata.viscoelastic_alpha_row.unsqueeze(0))
        xi.add_(data).sub_(prev)
        prev.copy_(data)
        data.addcmul_(xi, shared_metadata.viscoelastic_strength_row.unsqueeze(0))


# ============================ Spatial crosstalk ============================


class CrosstalkMeta(NamedTuple):
    """
    Per-sensor spatial-crosstalk layout record for one grid-shaped taxel sensor.

    ``g_ny``/``g_nx`` are the grid dims; ``probe_start``/``cache_start`` locate the sensor in the shared probe and
    cache arrays; ``n_groups`` selects how the per-group kernels map onto the 6 force/torque channels (see
    ``apply_spatial_crosstalk``).
    """

    g_ny: int
    g_nx: int
    probe_start: int
    cache_start: int
    n_groups: int


@dataclass
class SpatialCrosstalkMetadataMixin:
    """
    Shared per-sensor-class state for grid spatial crosstalk (measured branch).

    ``crosstalk_meta[i]`` is the layout record for the i-th registered crosstalk sensor; ``crosstalk_kernels[i]``
    holds its per-group depthwise ``(C, 1, kh, kw)`` conv weights and ``crosstalk_normals[i]`` its link-local grid
    normal (for the normal/shear force decomposition). ``any_crosstalk`` is a Python fast-path flag.
    """

    crosstalk_meta: list["CrosstalkMeta"] = field(default_factory=list)
    crosstalk_kernels: list[list[torch.Tensor]] = field(default_factory=list)
    crosstalk_normals: list[torch.Tensor] = field(default_factory=list)
    any_crosstalk: bool = False


def _depthwise_kernel(kernel_2d: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Shape an ``(kh, kw)`` kernel into the ``(n_channels, 1, kh, kw)`` weight a depthwise ``F.conv2d`` needs."""
    return kernel_2d.view(1, 1, *kernel_2d.shape).repeat(n_channels, 1, 1, 1)


def _conv_crosstalk(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Same-size depthwise 2D convolution: ``kernel`` is ``(C, 1, kh, kw)`` matching ``field``'s channel count."""
    n_channels = field.shape[1]
    padding = (kernel.shape[2] // 2, kernel.shape[3] // 2)
    return F.conv2d(field, kernel.to(field.dtype), groups=n_channels, padding=padding)


def build_crosstalk_kernels(
    options, grid_spacing: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> tuple[list[torch.Tensor], int]:
    """
    Build the per-group depthwise crosstalk kernels for one sensor from its options.

    With an explicit ``crosstalk_kernel`` of shape ``(N, M)`` -> 1 group, ``(2, N, M)`` -> 2 groups
    (normal | shear+torque), ``(3, N, M)`` -> 3 groups (normal | shear | torque); the array is used as-is.
    Otherwise the Gaussian path builds a single ``(1 - strength) * delta + strength * gaussian`` kernel sized from
    ``crosstalk_sigma`` and the grid spacing. Per-group channel counts: 1-group -> [6] (raw force+torque),
    2-group -> [1, 6], 3-group -> [1, 3, 3]. Returns ``(kernels, n_groups)``.
    """
    explicit = options.crosstalk_kernel
    if explicit is not None:
        arr = np.asarray(explicit, dtype=gs.np_float)
        planes = [arr] if arr.ndim == 2 else [arr[i] for i in range(arr.shape[0])]
        group_sums = [float(p.sum()) for p in planes]
        if any(abs(s - 1.0) > 1e-3 for s in group_sums):
            gs.logger.warning(
                f"crosstalk_kernel group sums {[round(s, 4) for s in group_sums]} are not ~1; total measured force "
                "is not conserved (sum > 1 amplifies, sum < 1 attenuates). Normalize each group to sum 1 unless this "
                "is intentional."
            )
        base = [torch.as_tensor(p, dtype=dtype, device=device) for p in planes]
        n_groups = len(planes)
    else:
        sigma = float(options.crosstalk_sigma)
        strength = float(options.crosstalk_strength)
        spacing_u = float(grid_spacing[0].item())
        spacing_v = float(grid_spacing[1].item())
        r_v = max(1, int(math.ceil(3.0 * sigma / spacing_v)))
        r_u = max(1, int(math.ceil(3.0 * sigma / spacing_u)))
        gaussian = gaussian_crosstalk_kernel(2 * r_v + 1, 2 * r_u + 1, sigma, spacing=(spacing_v, spacing_u))
        kernel = strength * gaussian
        kernel[r_v, r_u] += 1.0 - strength  # identity blend folded into the center (self) tap
        base = [torch.as_tensor(kernel, dtype=dtype, device=device)]
        n_groups = 1

    channels = {1: [6], 2: [1, 6], 3: [1, 3, 3]}[n_groups]
    return [_depthwise_kernel(k, c) for k, c in zip(base, channels)], n_groups


def apply_spatial_crosstalk(
    crosstalk_meta: list["CrosstalkMeta"],
    crosstalk_kernels: list[list[torch.Tensor]],
    crosstalk_normals: list[torch.Tensor],
    cache_data: torch.Tensor,
    probe_radii: torch.Tensor,
) -> None:
    """
    Apply per-sensor grid spatial crosstalk to the force/torque cache (measured branch), mutating ``cache_data``
    (``(B, total_cols)``) in place.

    Each sensor's slice spans ``2 * n_probes * 3`` columns: probe-major force xyz then probe-major torque xyz, with
    probes in ``iy * nx + ix`` order. A 1-group kernel blurs the raw 6 channels. A 2/3-group kernel first splits the
    force into a normal component (along the sensor's grid normal) and a shear component, blurs each group with its
    kernel, then recombines; torque rides the shear kernel (2-group) or its own twist kernel (3-group).
    """
    if not crosstalk_meta:
        return
    B = cache_data.shape[0]
    for meta, kernels, normal in zip(crosstalk_meta, crosstalk_kernels, crosstalk_normals):
        n_probes = meta.g_ny * meta.g_nx
        f0 = meta.cache_start
        t0 = f0 + n_probes * 3
        # Cache is probe-major (iy * nx + ix) with 3 inner cols; reshape to (B, ny, nx, 3) -> (B, 3, ny, nx).
        force = cache_data[:, f0:t0].view(B, meta.g_ny, meta.g_nx, 3).permute(0, 3, 1, 2)
        torque = cache_data[:, t0 : t0 + n_probes * 3].view(B, meta.g_ny, meta.g_nx, 3).permute(0, 3, 1, 2)

        if meta.n_groups == 1:
            field_in = torch.cat((force, torque), dim=1).contiguous()  # (B, 6, ny, nx)
            blurred = _conv_crosstalk(field_in, kernels[0])
            force_out, torque_out = blurred[:, 0:3], blurred[:, 3:6]
        else:
            n = normal.view(1, 3, 1, 1)
            f_normal = (force * n).sum(dim=1, keepdim=True)  # (B, 1, ny, nx): scalar normal force
            f_shear = force - f_normal * n  # (B, 3, ny, nx): tangential force
            f_normal_out = _conv_crosstalk(f_normal.contiguous(), kernels[0])
            if meta.n_groups == 2:
                rest = torch.cat((f_shear, torque), dim=1).contiguous()  # (B, 6, ny, nx): shear + torque
                rest_out = _conv_crosstalk(rest, kernels[1])
                f_shear_out, torque_out = rest_out[:, 0:3], rest_out[:, 3:6]
            else:
                f_shear_out = _conv_crosstalk(f_shear.contiguous(), kernels[1])
                torque_out = _conv_crosstalk(torque.contiguous(), kernels[2])
            force_out = f_normal_out * n + f_shear_out

        # Zero inactive filler probes (probe_radius == 0): the blur leaks neighbor signal into their cells.
        active = (probe_radii[meta.probe_start : meta.probe_start + n_probes] > 0.0).to(force_out.dtype)
        active = active.view(1, 1, meta.g_ny, meta.g_nx)
        force_out = force_out * active
        torque_out = torque_out * active

        # Inverse of the build permute: (B, 3, ny, nx) -> (B, ny, nx, 3) -> flat (B, ny*nx*3).
        cache_data[:, f0:t0] = force_out.permute(0, 2, 3, 1).reshape(B, n_probes * 3)
        cache_data[:, t0 : t0 + n_probes * 3] = torque_out.permute(0, 2, 3, 1).reshape(B, n_probes * 3)


SpatialCrosstalkSharedMetadataT = TypeVar("SpatialCrosstalkSharedMetadataT", bound=SpatialCrosstalkMetadataMixin)


class SpatialCrosstalkMixin(Generic[SpatialCrosstalkSharedMetadataT]):
    """
    Adds grid spatial crosstalk to a taxel sensor on the measured branch.

    The sensor populates ``_grid_normal`` / ``_grid_spacing`` / ``_use_grid_crosstalk`` in ``__init__`` (via
    ``_setup_crosstalk_grid``) and calls ``_register_crosstalk`` in ``build()`` when crosstalk is enabled.
    """

    _shared_metadata: SpatialCrosstalkSharedMetadataT

    def _setup_crosstalk_grid(self, options) -> None:
        """Resolve the grid frame for crosstalk and validate the grid requirement; sets the ``_grid_*`` attributes."""
        sensor_name = type(self).__name__
        is_grid = len(self._probe_layout_shape) == 2
        # Sensors without a per-probe normal (KinematicTaxel) pass None so the grid plane normal is derived from the
        # layout geometry; ProximityTaxel and the elastomer FFT path pass their explicit probe_local_normal.
        has_normal = isinstance(options, ProbesWithNormalSensorOptionsMixin)
        probe_normals = options.probe_local_normal if has_normal else None
        _, _, use_grid, is_regular, grid_normal, _, _, grid_spacing = normalize_grid_probe_layout(
            np.asarray(options.probe_local_pos, dtype=gs.np_float),
            None if probe_normals is None else np.asarray(probe_normals, dtype=gs.np_float),
            is_grid,
        )
        self._use_grid_crosstalk = use_grid
        self._grid_normal = torch.tensor(grid_normal, dtype=gs.tc_float, device=gs.device)
        self._grid_spacing = torch.tensor(grid_spacing, dtype=gs.tc_float, device=gs.device)
        if options.is_crosstalk_enabled:
            if not use_grid:
                gs.raise_exception(
                    f"{sensor_name} crosstalk requires a 2D grid-shaped probe_local_pos (shape (ny, nx, 3) with "
                    f"ny, nx >= 2 and non-degenerate spacing); got shape {tuple(self._probe_layout_shape)}."
                )
            if not is_regular:
                gs.logger.warning(
                    f"{sensor_name} crosstalk grid is not strictly regular (uniform spacing, uniform normals, "
                    "orthogonal tangents); crosstalk will use averaged spacing and normal as a best-fit approximation."
                )

    def _register_crosstalk(self) -> None:
        """Build this sensor's per-group crosstalk kernels and append them to the shared metadata lists."""
        sm = self._shared_metadata
        sensor_idx = sm.n_probes_per_sensor.shape[0] - 1  # this sensor was just registered by ProbeSensorMixin.build
        kernels, n_groups = build_crosstalk_kernels(self._options, self._grid_spacing, gs.device, gs.tc_float)
        sm.crosstalk_meta.append(
            CrosstalkMeta(
                g_ny=int(self._probe_layout_shape[0]),
                g_nx=int(self._probe_layout_shape[1]),
                probe_start=int(sm.sensor_probe_start[sensor_idx].item()),
                cache_start=int(sm.sensor_cache_start[sensor_idx].item()),
                n_groups=n_groups,
            )
        )
        sm.crosstalk_kernels.append(kernels)
        sm.crosstalk_normals.append(self._grid_normal.to(dtype=gs.tc_float))
        sm.any_crosstalk = True

    @classmethod
    def _apply_transform(
        cls,
        shared_metadata: SpatialCrosstalkSharedMetadataT,
        data: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ):
        super()._apply_transform(shared_metadata, data, timeline, is_measured=is_measured)
        if not is_measured or not shared_metadata.any_crosstalk:
            return
        apply_spatial_crosstalk(
            shared_metadata.crosstalk_meta,
            shared_metadata.crosstalk_kernels,
            shared_metadata.crosstalk_normals,
            data,
            shared_metadata.probe_radii,
        )
