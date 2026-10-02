import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.typing import EnvsIdxType
from genesis.utils.misc import gaussian_crosstalk_kernel

from .raycaster import RaycastContext

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.options.sensors.tactile import SpatialCrosstalkOptionsMixin
    from genesis.utils.ring_buffer import TensorRingBuffer


# Tolerance of the grid regularity, tangent orthogonality and normal uniformity checks
_GRID_TOL = 1.0e-5


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


def get_mesh_geom_chunks(link: "RigidLink", prefer_visual: bool) -> list[tuple[object, np.ndarray, np.ndarray]]:
    """
    Return per-geom mesh chunks ``(geom, verts_link, faces)`` in link-local frame.

    With ``prefer_visual`` the vgeoms are picked over the geoms when both exist, and the other type serves when the
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
    Build a median-split axis-aligned bounding box (AABB) bounding volume hierarchy (BVH) over a static set of elements
    (points, triangles) in link-local frame.

    The splits follow ``centroids`` along the axis of the longest spread, and the node AABBs are the union of the
    per-element ``aabb_mins`` and ``aabb_maxs``. A point-cloud BVH passes ``centroids == aabb_mins == aabb_maxs`` (the
    points themselves), a triangle BVH the centroid and the min and max bounds of each triangle.

    The leaves carry the ``global_rows`` the caller provides, absolute rows into the element table of the type, so the
    kernel indexes that table directly. A leaf has ``-1`` in ``node_left`` and ``node_right``. It returns ``(node_min,
    node_max, node_left, node_right, node_elem_start, node_elem_n, elem_idx)``.
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
def func_sphere_intersects_aabb(center, radius_sq, bmin, bmax):
    """
    Return whether the closest point of an axis-aligned bounding box (AABB) to ``center`` lies within ``radius_sq``, a
    squared-distance sphere test.

    Passing ``radius_sq = current_best_dist_sq`` makes it a closest-point cull.
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
def func_aabb_intersects_aabb(amin, amax, bmin, bmax):
    """Return whether two axis-aligned bounding boxes (AABBs) overlap."""
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

    See ``BVHMetadata`` for the field semantics. The element payload tables (``tri_verts``, point-cloud positions,
    ...) differ per sensor type and travel as separate kernel arguments.
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
    Element-agnostic scaffolding for a static, link-local, chunked axis-aligned bounding box (AABB) bounding volume
    hierarchy (BVH) shared by every sensor of one type.

    One *chunk* per (sensor, tracked_link): each chunk is a small subtree built once at scene build in the local frame
    of the tracked link. Subclasses (PointCloudBVH, TriangleMeshBVH) layer on element-specific payload tables, and
    ``leaf_elem_idx`` entries are absolute rows into those tables.

    Per-sensor slice into the chunk arrays:
        ``chunks[sensor_chunk_start[s] : sensor_chunk_start[s] + sensor_chunk_count[s]]``
    Per-chunk slice into the flat node arrays:
        ``nodes[chunk_node_start[c] : chunk_node_start[c] + chunk_node_count[c]]``
    Per-leaf slice into ``leaf_elem_idx``:
        ``leaf_elem_idx[node_leaf_start[n] : node_leaf_start[n] + node_leaf_count[n]]``

    ``node_left == -1`` marks a leaf, any other value an inner node whose ``node_left`` and ``node_right`` are absolute
    child indices.

    A subclass builds itself whole from every sensor of the type through its ``build`` factory.
    """

    sensor_chunk_start: torch.Tensor
    sensor_chunk_count: torch.Tensor

    chunk_link_idx: torch.Tensor
    chunk_node_start: torch.Tensor
    chunk_node_count: torch.Tensor

    node_min: torch.Tensor
    node_max: torch.Tensor
    node_left: torch.Tensor
    node_right: torch.Tensor

    node_leaf_start: torch.Tensor
    node_leaf_count: torch.Tensor
    leaf_elem_idx: torch.Tensor

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


class GridFFTConvArrayMixin:
    """
    Array mixin holding the per-grid 2D-FFT convolution passes of every grid-shaped sensor of the type.

    Attributes
    ----------
    grid_fft_meta : list of NamedTuple
        One record per grid-FFT sensor. The leading 5 fields are always
        ``(sensor_idx, g_ny, g_nx, probe_start, cache_start)``; sensors append their kernel params after that
        (e.g. ``GridFFTMeta`` for HydroShear dilation).
    grid_fft_kernels_stacked : torch.Tensor
        Stacked complex ``rfft2`` kernels (half spectrum), shape ``(n_grid, n_planes, fft_ny, fft_nx // 2 + 1)``.
        Built at the FFT size covering every grid sensor.
    grid_fft_buffer : torch.Tensor
        Reused per-step real buffer: ``(B, n_grid, n_channels, fft_ny, fft_nx)`` when registered with
        ``n_buffer_channels > 0``, else ``(B, n_grid, fft_ny, fft_nx)``.
    """


def build_grid_fft(
    array: GridFFTConvArrayMixin,
    meta_entries: list[NamedTuple],
    fft_sizes: list[tuple[int, int]],
    kernel_builder: Callable[[NamedTuple, tuple[int, int]], torch.Tensor],
    n_buffer_channels: int,
    batch_size: int,
) -> None:
    """
    Set up the FFT convolution of every grid-shaped sensor of the type at once: the stacked kernels and the per-step
    buffer, at the FFT size covering every sensor.

    Parameters
    ----------
    meta_entries : list of NamedTuple
        Records of the grid sensors, kept in ``grid_fft_meta``; the leading 5 fields of each must be
        ``(sensor_idx, g_ny, g_nx, probe_start, cache_start)``, followed by any sensor-specific kernel params.
    fft_sizes : list of (int, int)
        The ``(ny, nx)`` FFT size each sensor needs. All kernels are built at their elementwise max.
    kernel_builder : callable
        ``kernel_builder(meta_entry, fft_n) -> (n_planes, fft_ny, fft_nx // 2 + 1)`` complex tensor (an ``rfft2``
        half spectrum).
    n_buffer_channels : int
        When ``> 0``, allocate a 5D ``(B, n_grid, n_buffer_channels, ny, nx)`` per-step buffer, and a 4D
        ``(B, n_grid, ny, nx)`` one when 0.
    """
    # One FFT size for every kernel: frequency-domain padding differs from spatial zero-padding, so a kernel built
    # at a smaller size cannot be padded up afterwards
    # FIXME: every sensor pays the largest size of the type; group the sensors by FFT size instead.
    array.grid_fft_meta = list(meta_entries)
    if not meta_entries:
        array.grid_fft_kernels_stacked = torch.empty((0, 0, 0, 0), dtype=torch.complex64, device=gs.device)
        array.grid_fft_buffer = torch.empty((0, 0, 0, 0), dtype=gs.tc_float, device=gs.device)
        return
    fft_n = (max(n[0] for n in fft_sizes), max(n[1] for n in fft_sizes))
    array.grid_fft_kernels_stacked = torch.stack([kernel_builder(m, fft_n) for m in meta_entries], dim=0)
    buffer_shape = (
        (batch_size, len(meta_entries), n_buffer_channels, *fft_n)
        if n_buffer_channels > 0
        else (batch_size, len(meta_entries), *fft_n)
    )
    array.grid_fft_buffer = torch.zeros(buffer_shape, dtype=gs.tc_float, device=gs.device)


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


class GridFrame(NamedTuple):
    """
    The grid frame of one probe sensor, as ``normalize_grid_probe_layout`` resolves it.

    ``use_grid_fft`` is True when the layout has shape ``(ny, nx, 3)`` with ``ny, nx >= 2`` and non-degenerate
    spacing along both axes: the FFT path is usable and the frame is a best-fit approximation (average step vectors
    over all adjacent pairs, average unit normal over all probes). ``is_grid_regular`` is True when, in addition, the
    layout is strictly regular (see ``normalize_grid_probe_layout``). When ``use_grid_fft`` is False, the normal,
    tangents and spacing are zero.
    """

    use_grid_fft: bool
    is_grid_regular: bool
    grid_normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    grid_spacing: np.ndarray


def normalize_grid_probe_layout(probe_pos: np.ndarray, probe_normals: np.ndarray | None, is_grid: bool) -> GridFrame:
    """
    Validate a probe layout and resolve its grid frame when the layout qualifies (see ``GridFrame``).

    ``probe_normals`` is the per-probe normal of the sensor, or None for a sensor carrying none, whose grid plane
    normal is then the unit cross product of the grid tangents.

    ``is_grid_regular`` is True when, in addition, the layout is strictly regular: normals uniform within
    tolerance, tangents orthogonal, both tangents in the plane perpendicular to the normal, and all probes lie
    on the regular ``(spacing_u, spacing_v)`` rectangle implied by the averaged steps. Callers that proceed
    with FFT on an irregular layout (``use_grid_fft`` and not ``is_grid_regular``) should warn the user.
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

    return GridFrame(
        use_grid_fft,
        is_grid_regular,
        grid_normal.astype(gs.np_float, copy=False),
        tangent_u.astype(gs.np_float, copy=False),
        tangent_v.astype(gs.np_float, copy=False),
        grid_spacing.astype(gs.np_float, copy=False),
    )


# ============================ Contact prefilter ============================


# Per-(env, sensor) cap on the prefiltered contact list consumed by the BVH-mask builder. Sensors track a
# single rigid link; even with multicontact and many neighbouring geoms, the count of contacts touching one
# link rarely exceeds a few hundred.
MAX_CONTACTS_PER_SENSOR = 1024

# Per-(env, sensor) cap on the deduplicated opposing-geom list consumed by ``_func_query_contact_depth`` and
# ``_func_query_contact_depth_penetration`` (the SDF path). Unlike the contact list, this counts *distinct*
# contacting geoms, not contact points: one pressing object is a single entry regardless of how many contact
# points multicontact emits. A single rigid sensor link touching >64 distinct geoms at once is implausible,
# so 64 is generous; overflow silently truncates, matching ``MAX_CONTACTS_PER_SENSOR``.
MAX_GEOMS_PER_SENSOR = 64


class ContactPrefilterArrayMixin:
    """
    Array mixin holding the per-(env, sensor) prefilter buffers of the tactile sensors whose kernels query the contacts
    of the collider per probe.

    KinematicTaxel and ContactDepthProbe populate them each step in ``kinematic_tactile.py``:

    - ``sensor_contacts_idx`` / ``sensor_n_contacts``: the compact list of the contact indices whose ``link_a`` or
      ``link_b`` matches the tracked link of the sensor (``_kernel_build_sensor_contact_idx``). It feeds the BVH-mask
      builder of the raycast path. The shapes are ``(B, n_sensors, max_contacts)`` and ``(B, n_sensors)``, and the
      per-sensor cap (``MAX_CONTACTS_PER_SENSOR``) is read off ``sensor_contacts_idx.shape[2]``.
    - ``sensor_geoms_idx`` / ``sensor_n_geoms``: the compact deduplicated list of the opposing contacting geoms of the
      same link (``_kernel_build_sensor_geom_idx``). It feeds the SDF path, so each probe queries one SDF per distinct
      geom instead of one per contact point. The shapes are ``(B, n_sensors, max_geoms)`` and ``(B, n_sensors)``, and
      the cap (``MAX_GEOMS_PER_SENSOR``) is read off ``sensor_geoms_idx.shape[2]``.
    """

    def build(self) -> None:
        super().build()

        # The per-(env, sensor) contact prefilter buffers the per-step kernel writes into, sized once for every sensor
        self.sensor_contacts_idx = torch.zeros(
            (self._sim._B, self.n_sensors, MAX_CONTACTS_PER_SENSOR), dtype=gs.tc_int, device=gs.device
        )
        self.sensor_n_contacts = torch.zeros((self._sim._B, self.n_sensors), dtype=gs.tc_int, device=gs.device)
        self.sensor_geoms_idx = torch.zeros(
            (self._sim._B, self.n_sensors, MAX_GEOMS_PER_SENSOR), dtype=gs.tc_int, device=gs.device
        )
        self.sensor_n_geoms = torch.zeros((self._sim._B, self.n_sensors), dtype=gs.tc_int, device=gs.device)


# ============================ Contact depth query mode (SDF vs raycast) ============================


class ContactDepthQueryArrayMixin:
    """
    Array mixin holding the contact-depth query backend of every sensor of the type.

    ``contact_depth_query`` is the resolved mode for every sensor of the type - ``"sdf"`` or ``"raycast"``. The
    backend is dispatched once per array (one kernel covers all its sensors), so every sensor must agree: ``build``
    resolves the mode from all of them at once and raises when two request different ones. ``None`` defers to the
    others; ``None`` at update time falls back to ``"sdf"``.

    When mode is ``"raycast"``, the collision/visual BVHs come from the ``RaycastContext`` the array fetches from
    ``SensorManager`` at build and reads as ``self._raycast``, built lazily on raycast opt-in and refreshed once per
    step by ``SensorManager``. ``sensor_candidate_geom_mask`` is a per-(env, sensor, geom) bool gate - scattered per step from
    the contact prefilter (KinematicTactile family) or once at build from ``sensor_track_geom_idx`` (ElastomerTaxel) -
    so BVH leaves whose ``faces_info.geom_idx`` falls outside the mask are skipped.
    """

    def build(self) -> None:
        """Resolve the contact-depth backend from every sensor of the type and activate it (see the class docstring)."""
        super().build()

        modes = {sensor.options.contact_depth_query for sensor in self._sensors} - {None}
        if len(modes) > 1:
            gs.raise_exception(
                f"{type(self._sensors[0]).__name__} sensors disagree on contact_depth_query ({sorted(modes)}). All "
                "sensors of a tactile type share one contact-depth backend; use the same mode for every sensor of "
                "this type (build separate scenes to compare backends)."
            )
        self.contact_depth_query = modes.pop() if modes else None
        self._raycast = self._manager.get_context(RaycastContext)
        if self.contact_depth_query == "raycast":
            self._raycast.activate()
            self.sensor_candidate_geom_mask = torch.zeros(
                (self._sim._B, self.n_sensors, self._sim.rigid_solver.n_geoms), dtype=gs.tc_bool, device=gs.device
            )
        else:
            self._sim.rigid_solver.collider.activate_sdf()


# ============================ ViscoelasticHysteresis ============================


class ViscoelasticHysteresisArrayMixin:
    """
    Viscoelastic hysteresis (single Maxwell element, equilibrium gain normalized to 1), on the measured branch.

    Per simulation step::

        alpha = exp(-dt / tau)
        xi    <- alpha * xi + (x - x_prev)
        output = x + strength * xi
        x_prev <- x

    After a step input from 0 to X the output jumps to ``X * (1 + strength)`` and decays back to ``X`` with time
    constant ``tau``. On cyclic input the output overshoots on rising edges and undershoots on falling edges.
    """

    def build(self) -> None:
        super().build()

        # Every array mixing this in serves an options class that inherits ViscoelasticHysteresisOptionsMixin, so the
        # hysteresis fields are always declared
        strengths = [sensor.options.hysteresis_strength for sensor in self._sensors]
        taus = [sensor.options.hysteresis_tau for sensor in self._sensors]
        alphas = [math.exp(-self._dt / tau) if tau > 0.0 else 0.0 for tau in taus]
        self.hysteresis_strength = torch.tensor(strengths, dtype=gs.tc_float, device=gs.device)
        self.hysteresis_alpha = torch.tensor(alphas, dtype=gs.tc_float, device=gs.device)
        self.has_any_hysteresis = any(strength > 0.0 and tau > 0.0 for strength, tau in zip(strengths, taus))
        # The per-sensor parameters spread over the cache columns of each sensor, and the Maxwell state per column
        cache_sizes = torch.tensor(self.cache_sizes, device=gs.device)
        self.viscoelastic_strength_row = torch.repeat_interleave(self.hysteresis_strength, cache_sizes)
        self.viscoelastic_alpha_row = torch.repeat_interleave(self.hysteresis_alpha, cache_sizes)
        state_shape = (self._sim._B, sum(self.cache_sizes))
        self.viscoelastic_xi = torch.zeros(state_shape, dtype=gs.tc_float, device=gs.device)
        self.viscoelastic_prev_input = torch.zeros(state_shape, dtype=gs.tc_float, device=gs.device)

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        super().reset(envs_idx)

        self.viscoelastic_xi[envs_idx] = 0.0
        self.viscoelastic_prev_input[envs_idx] = 0.0

    def _apply_transform(self, data: torch.Tensor, timeline: "TensorRingBuffer", *, is_measured: bool) -> None:
        super()._apply_transform(data, timeline, is_measured=is_measured)
        if not is_measured or not self.has_any_hysteresis:
            return
        xi = self.viscoelastic_xi
        prev = self.viscoelastic_prev_input
        xi.mul_(self.viscoelastic_alpha_row.unsqueeze(0))
        xi.add_(data).sub_(prev)
        prev.copy_(data)
        data.addcmul_(xi, self.viscoelastic_strength_row.unsqueeze(0))


# ============================ Spatial crosstalk ============================


class CrosstalkMeta(NamedTuple):
    """
    Spatial-crosstalk layout record of one grid-shaped taxel sensor.

    ``g_ny`` and ``g_nx`` are the grid dimensions. ``probe_start`` and ``cache_start`` locate the sensor in the shared
    probe and cache arrays. ``n_groups`` selects how the per-group kernels map onto the 6 force and torque channels
    (see ``apply_spatial_crosstalk``).
    """

    g_ny: int
    g_nx: int
    probe_start: int
    cache_start: int
    n_groups: int


class SpatialCrosstalkArrayMixin:
    """
    Array mixin holding the grid spatial crosstalk (measured branch) of every sensor of the type.

    ``crosstalk_meta[i]`` is the layout record for the i-th registered crosstalk sensor; ``crosstalk_kernels[i]``
    holds its per-group depthwise ``(C, 1, kh, kw)`` conv weights and ``crosstalk_normals[i]`` its link-local grid
    normal (for the normal/shear force decomposition). ``has_any_crosstalk`` is a Python fast-path flag.
    """

    def build(self) -> None:
        super().build()

        self.crosstalk_meta = []
        self.crosstalk_kernels = []
        self.crosstalk_normals = []
        self.has_any_crosstalk = False
        for i_s, (sensor, layout_shape, grid_frame) in enumerate(
            zip(self._sensors, self._sensors_probe_layout_shape, self._sensors_grid_frame)
        ):
            options = sensor.options
            if not options.is_crosstalk_enabled:
                continue
            sensor_name = type(sensor).__name__
            if not grid_frame.use_grid_fft:
                gs.raise_exception(
                    f"{sensor_name} crosstalk requires a 2D grid-shaped probe_local_pos (shape (ny, nx, 3) with "
                    f"ny, nx >= 2 and non-degenerate spacing); got shape {tuple(layout_shape)}."
                )
            if not grid_frame.is_grid_regular:
                gs.logger.warning(
                    f"{sensor_name} crosstalk grid is not strictly regular (uniform spacing, uniform normals, "
                    "orthogonal tangents); crosstalk will use averaged spacing and normal as a best-fit approximation."
                )
            grid_spacing_t = torch.tensor(grid_frame.grid_spacing, dtype=gs.tc_float, device=gs.device)
            kernels, n_groups = build_crosstalk_kernels(options, grid_spacing_t, gs.device, gs.tc_float)
            self.crosstalk_meta.append(
                CrosstalkMeta(
                    g_ny=int(layout_shape[0]),
                    g_nx=int(layout_shape[1]),
                    probe_start=int(self.sensor_probe_start[i_s].item()),
                    cache_start=int(self.sensors_cache_start[i_s].item()),
                    n_groups=n_groups,
                )
            )
            self.crosstalk_kernels.append(kernels)
            self.crosstalk_normals.append(torch.tensor(grid_frame.grid_normal, dtype=gs.tc_float, device=gs.device))
            self.has_any_crosstalk = True

    def _apply_transform(self, data: torch.Tensor, timeline: "TensorRingBuffer", *, is_measured: bool) -> None:
        super()._apply_transform(data, timeline, is_measured=is_measured)
        if not is_measured or not self.has_any_crosstalk:
            return
        apply_spatial_crosstalk(
            self.crosstalk_meta, self.crosstalk_kernels, self.crosstalk_normals, data, self.probe_radii
        )


def _depthwise_kernel(kernel_2d: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Shape an ``(kh, kw)`` kernel into the ``(n_channels, 1, kh, kw)`` weight a depthwise ``F.conv2d`` needs."""
    return kernel_2d.view(1, 1, *kernel_2d.shape).repeat(n_channels, 1, 1, 1)


def _conv_crosstalk(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolve ``field`` with the depthwise 2D ``kernel`` of shape ``(C, 1, kh, kw)``, ``C`` being its channel count, at
    the same size.
    """
    n_channels = field.shape[1]
    padding = (kernel.shape[2] // 2, kernel.shape[3] // 2)
    return F.conv2d(field, kernel.to(field.dtype), groups=n_channels, padding=padding)


def build_crosstalk_kernels(
    options: "SpatialCrosstalkOptionsMixin", grid_spacing: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> tuple[list[torch.Tensor], int]:
    """
    Build the per-group depthwise crosstalk kernels for one sensor from its options.

    An explicit ``crosstalk_kernel`` of shape ``(N, M)`` gives 1 group, ``(2, N, M)`` 2 groups (normal | shear+torque),
    ``(3, N, M)`` 3 groups (normal | shear | torque), the array used as is. Without one, the Gaussian path builds a
    single ``(1 - strength) * delta + strength * gaussian`` kernel sized from ``crosstalk_sigma`` and the grid
    spacing. Per-group channel counts: 1-group -> [6] (raw force+torque), 2-group -> [1, 6], 3-group -> [1, 3, 3].
    Returns ``(kernels, n_groups)``.
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
    force into a normal component (along the grid normal of the sensor) and a shear component, blurs each group with
    its kernel, then recombines them. The torque takes the shear kernel (2-group) or its own twist kernel (3-group).
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
