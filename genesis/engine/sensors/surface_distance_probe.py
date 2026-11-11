import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import func_bvh_traverse
from genesis.engine.solvers.rigid.data import DynState
from genesis.options.sensors import SurfaceDistanceProbe as SurfaceDistanceProbeOptions
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.raycast_qd import closest_point_on_triangle, func_sphere_intersects_aabb

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata
from .probe import ProbeSensorMetadataMixin, ProbeSensorMixin, func_noised_probe_radius, get_measured_bufs
from .tactile_shared import (
    BVHMetadata,
    ChunkedBVHData,
    func_vec3_at,
    get_mesh_geom_chunks,
)

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


# The triangles of the chunks of a surface distance probe class
CHUNK_TRIANGLES = Size("chunk_triangles")


class ChunkTriangles(Data, kind=DataKind.CONSTANT, shape=(CHUNK_TRIANGLES,)):
    """The triangles of the chunks of a sensor class, as their three vertices in the frame of their tracked link."""

    verts: qd.Tensor = field((float, 3), shape=(CHUNK_TRIANGLES, 3))


# The probes of a surface distance probe class
SURFACE_PROBES = Size("surface_probes")


class ClosestPoints(Data, kind=DataKind.SCRATCH, shape=(SURFACE_PROBES, ENVS)):
    """The closest point of the triangles of a chunk to each probe, in the frame of the chunk."""

    point: qd.Tensor = field((float, 3))


@qd.data_oriented
class ClosestTriangle:
    """The visitor of the triangles of a chunk within a radius of a probe, its bound being the squared distance of the
    closest one.

    The query is the probe in the frame of the chunk, the query index being the probe. It reads the triangles of the
    chunks (see ChunkTriangles) and records the closest point of each probe in each environment (see ClosestPoints).
    """

    is_instance_level = False

    def __init__(self, triangles: ChunkTriangles, points: ClosestPoints) -> None:
        self.triangles = triangles
        self.points = points
        self.Query = qd.types.struct(probe=qd.types.vector(3, gs.qd_float))

    @qd.func
    def is_node_hit(
        self,
        i_t: int,
        i_q: int,
        node_min: qd.types.vector(3),
        node_max: qd.types.vector(3),
        query,
        bound: float,
    ) -> bool:
        return func_sphere_intersects_aabb(query.probe, bound, node_min, node_max)

    @qd.func
    def visit_leaf(self, i_t: int, i_q: int, i_k: int, i_a: int, query, bound: float) -> float:
        closest = closest_point_on_triangle(
            query.probe, self.triangles.verts[i_a, 0], self.triangles.verts[i_a, 1], self.triangles.verts[i_a, 2]
        )
        dist_sq = (closest - query.probe).norm_sqr()
        if dist_sq < bound:
            bound = dist_sq
            self.points.point[i_q, i_t] = closest
        return bound


@dataclass
class TriangleMeshBVH(BVHMetadata):
    """
    BVH over tracked mesh triangles for one sensor class.

    The triangles follow the chunks in order, as their link-local vertices (see ChunkTriangles). See ``BVHMetadata``
    for the shared scaffolding semantics. Rigid-link assumption: built once at scene init, never rebuilt.
    """

    # The vertices of the triangles of every chunk, until the triangles are built
    chunks_tri_verts: list[np.ndarray] = dataclasses.field(default_factory=list)
    # The closest-point query of the probes of the class, built with the trees
    closest_query: ClosestTriangle | None = None

    def append_sensor(self, track_link_idx: np.ndarray, solver: "RigidSolver") -> None:
        """Append the chunks of one sensor, one per tracked link with triangles, in the frame of that link."""
        links_idx, chunks_aabb_min, chunks_aabb_max = [], [], []
        for link_idx in track_link_idx.tolist():
            geom_chunks = get_mesh_geom_chunks(solver.links[link_idx], prefer_visual=False)
            if not geom_chunks:
                continue
            # The triangles of every geom of the link in one chunk, as their three vertices
            tri_verts = np.concatenate([verts_link[faces] for _, verts_link, faces in geom_chunks])
            links_idx.append(link_idx)
            chunks_aabb_min.append(tri_verts.min(axis=1))
            chunks_aabb_max.append(tri_verts.max(axis=1))
            self.chunks_tri_verts.append(tri_verts)
        self.append_chunks(links_idx, chunks_aabb_min, chunks_aabb_max)

    def build_closest_query(self, n_probes: int, n_envs: int) -> None:
        """Build the trees and the triangles of the chunks and the closest-point query of the probes of the class."""
        self.build()
        triangles = ChunkTriangles(sizes=dict(chunk_triangles=sum([len(verts) for verts in self.chunks_tri_verts])))
        if self.chunks_tri_verts:
            triangles.verts.from_numpy(np.concatenate(self.chunks_tri_verts))
        self.chunks_tri_verts.clear()
        points = ClosestPoints(sizes=dict(surface_probes=n_probes, envs=n_envs))
        self.closest_query = ClosestTriangle(triangles, points)


# Fastcache stays off for a kernel taking a visitor or a target (see RaycastTarget)
@qd.kernel
def _kernel_surface_distance_probe_bvh(
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    bvh: ChunkedBVHData,
    closest_query: qd.template(),
    positions_gt: qd.types.ndarray(),
    positions_measured: qd.types.ndarray(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    dyn_state: DynState,
    eps: float,
):
    """Find the closest point of the tracked triangles to each probe in every environment, within its radius.

    The probe goes into the frame of the tracked link of each chunk of its sensor, whose tree the search walks (see
    ClosestTriangle). A single search within the larger of the nominal radius and the noised one serves both: a branch
    whose radius holds no triangle reads its radius as distance and the probe as closest point. The closest points
    in the world are written to `positions_*` and the distances to `output_*`.
    """
    for i_p, i_b in qd.ndrange(probe_positions_local.shape[0], output_gt.shape[-1]):
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        probe_world = gu.qd_transform_by_trans_quat(
            func_vec3_at(i_p, probe_positions_local),
            dyn_state.links.pos[sensor_link_idx, i_b],
            dyn_state.links.quat[sensor_link_idx, i_b],
        )

        radius_gt = probe_radii[i_p]
        radius_m = radius_gt
        if probe_radii_noise[i_p] > eps:
            radius_m = func_noised_probe_radius(radius_gt, probe_radii_noise[i_p], eps)
        radius = qd.max(radius_gt, radius_m)

        point = probe_world
        dist_sq = radius * radius
        for i_c in range(bvh.sensor_chunk_start[i_s], bvh.sensor_chunk_start[i_s] + bvh.sensor_chunk_count[i_s]):
            track_link_idx = bvh.chunk_link_idx[i_c]
            track_pos = dyn_state.links.pos[track_link_idx, i_b]
            track_quat = dyn_state.links.quat[track_link_idx, i_b]
            # The tree of the chunk lives in the frame of its tracked link
            probe = gu.qd_inv_transform_by_trans_quat(probe_world, track_pos, track_quat)
            i_root = bvh.trees.trees_root[i_c]
            leaf_offset = bvh.trees.trees_leaf_offset[i_c]
            dist_sq_chunk = func_bvh_traverse(
                i_b,
                i_p,
                i_root,
                leaf_offset,
                leaf_base=0,
                query=closest_query.Query(probe=probe),
                bound=dist_sq,
                tree=bvh.trees.tree,
                visitor=closest_query,
            )
            # The visits record the closest point in the frame of the chunk, a closer one than the chunks before
            if dist_sq_chunk < dist_sq:
                dist_sq = dist_sq_chunk
                point = gu.qd_transform_by_trans_quat(closest_query.points.point[i_p, i_b], track_pos, track_quat)

        i_cache = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
        point_gt = probe_world
        if dist_sq < radius_gt * radius_gt:
            point_gt = point
        point_m = probe_world
        if dist_sq < radius_m * radius_m:
            point_m = point
        output_gt[i_cache, i_b] = qd.sqrt(qd.min(dist_sq, radius_gt * radius_gt))
        output_measured[i_cache, i_b] = qd.sqrt(qd.min(dist_sq, radius_m * radius_m))
        for j in qd.static(range(3)):
            positions_gt[i_b, i_p, j] = point_gt[j]
            positions_measured[i_b, i_p, j] = point_m[j]


@dataclass
class SurfaceDistanceProbeSensorMetadataMixin(ProbeSensorMetadataMixin):
    """
    Shared metadata for surface distance probe sensors: tracked-link bookkeeping, nearest-point buffer,
    and the per-class static triangle-mesh BVH consumed by ``_kernel_surface_distance_probe_bvh``.
    """

    track_link_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_link_end: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    track_link_flat: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    nearest_positions: torch.Tensor = make_tensor_field((0, 0, 3))
    nearest_positions_measured: torch.Tensor = make_tensor_field((0, 0, 3))
    bvh: TriangleMeshBVH = dataclasses.field(default_factory=TriangleMeshBVH)


@dataclass
class SurfaceDistanceProbeMetadata(
    SurfaceDistanceProbeSensorMetadataMixin, RigidSensorMetadataMixin, SimpleSensorMetadata
):
    """Shared metadata for the SurfaceDistanceProbe sensor class."""


class SurfaceDistanceProbeSensor(
    ProbeSensorMixin[SurfaceDistanceProbeMetadata],
    RigidSensorMixin[SurfaceDistanceProbeMetadata],
    SimpleSensor[SurfaceDistanceProbeOptions, None, SurfaceDistanceProbeMetadata, tuple],
):
    """Surface distance probe: distance and nearest point from probe positions to tracked mesh surfaces."""

    def __init__(
        self, options: SurfaceDistanceProbeOptions, idx: int, shared_context, shared_metadata, manager: "SensorManager"
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._nearest_points_slice: slice | None = None

    def _get_return_format(self) -> tuple[int, ...]:
        # Mirror the probe layout so a grid ``probe_local_pos`` (M, N, 3) reads back as (..., M, N), consistent with
        # the other grid tactile sensors; a flat layout stays (..., n_probes). The cache is flat either way.
        return self._probe_layout_shape

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    def build(self):
        super().build()

        track_link_idx = np.asarray(self._options.track_link_idx, dtype=gs.np_int)
        n_tracked = len(track_link_idx)
        start = (
            int(self._shared_metadata.track_link_flat.shape[0])
            if self._shared_metadata.track_link_flat.numel() > 0
            else 0
        )
        self._shared_metadata.track_link_start = concat_with_tensor(
            self._shared_metadata.track_link_start, start, expand=(1,)
        )
        self._shared_metadata.track_link_end = concat_with_tensor(
            self._shared_metadata.track_link_end, start + n_tracked, expand=(1,)
        )
        track_flat = torch.tensor(track_link_idx, dtype=gs.tc_int, device=gs.device)
        self._shared_metadata.track_link_flat = concat_with_tensor(
            self._shared_metadata.track_link_flat, track_flat, expand=(n_tracked,)
        )

        self._shared_metadata.nearest_positions = torch.zeros(
            (self._manager._sim._B, self._shared_metadata.total_n_probes, 3), dtype=gs.tc_float, device=gs.device
        )
        self._shared_metadata.nearest_positions_measured = torch.zeros(
            (self._manager._sim._B, self._shared_metadata.total_n_probes, 3), dtype=gs.tc_float, device=gs.device
        )
        slice_start = self._shared_metadata.sensor_probe_start[self._idx]
        self._nearest_points_slice = slice(slice_start, slice_start + self._n_probes)

        # Build the per-(sensor, tracked-link) triangle BVH in link-local frame. Rigid links don't deform,
        # so this is a one-shot scene-build cost; per-step queries traverse the static structure.
        self._shared_metadata.bvh.append_sensor(track_link_idx, self._shared_metadata.solver)
        if self._idx == len(self._manager._sensors_by_type[type(self)]) - 1:
            self._shared_metadata.bvh.build_closest_query(self._shared_metadata.total_n_probes, self._manager._sim._B)

    @classmethod
    def reset(cls, shared_metadata: SurfaceDistanceProbeMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        # Pre-first-step placeholder. The kernel writes world-frame nearest points on each step; before that, an
        # uninitialized read returns zeros rather than misleading link-local positions.
        shared_metadata.nearest_positions[envs_idx] = 0.0
        shared_metadata.nearest_positions_measured[envs_idx] = 0.0

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_context: None,
        shared_metadata: SurfaceDistanceProbeMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        measured, measured_cols_b = get_measured_bufs(
            shared_metadata, current_ground_truth_data_T, measured_data_timeline
        )
        bvh = shared_metadata.bvh
        _kernel_surface_distance_probe_bvh(
            shared_metadata.probe_sensor_idx,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.probe_positions,
            shared_metadata.probe_radii,
            shared_metadata.probe_radii_noise,
            bvh.kernel_bvh,
            bvh.closest_query,
            shared_metadata.nearest_positions,
            shared_metadata.nearest_positions_measured,
            current_ground_truth_data_T,
            measured_cols_b,
            solver.dyn_state,
            eps=gs.EPS,
        )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext", envs_idx: list[int]) -> list[tuple[None, "Mesh", None]]:
        """Make the probes, their nearest surface points and the lines joining them, for the first rendered env."""
        if 0 not in envs_idx:
            return []
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        # Single env: drop the leading env axis to a bare (3,) / (4,); squeeze(0) leaves an unbatched vector untouched.
        link_pos = self._link.get_pos(env_idx, relative=False).squeeze(0)
        link_quat = self._link.get_quat(env_idx, relative=False).squeeze(0)
        probe_world = tensor_to_array(
            gu.transform_by_trans_quat(self._probe_local_pos.reshape(-1, 3), link_pos, link_quat)
        ).reshape(-1, 3)
        points = tensor_to_array(self.nearest_points[env_idx]).reshape(-1, 3)

        rgb = tuple(float(c) for c in self._options.debug_probe_color)
        line_color = (*rgb, 1.0)
        nodes = self._draw_probe_spheres(context, probe_world, rgb)
        nodes.append(
            context.make_debug_spheres(
                poss=points, radius=float(self._options.debug_probe_center_radius), color=line_color
            )
        )
        for i in range(len(probe_world)):
            nodes.append(
                context.make_debug_line(
                    probe_world[i],
                    points[i],
                    radius=float(self._options.debug_probe_center_radius) / 4.0,
                    color=line_color,
                )
            )
        return [(None, node, None) for node in nodes]

    @property
    def nearest_points(self) -> torch.Tensor:
        """Nearest mesh points for the measured (noisy-radius) query, aligned with ``read()`` -- a grid
        ``probe_local_pos`` (M, N, 3) reads back as (..., M, N, 3), a flat layout as (..., n_probes, 3)."""
        points = self._shared_metadata.nearest_positions_measured[..., self._nearest_points_slice, :]
        return points.reshape(*points.shape[:-2], *self._probe_layout_shape, 3)

    @property
    def nearest_points_ground_truth(self) -> torch.Tensor:
        """Nearest mesh points for the nominal-radius ground-truth query, aligned with ``read_ground_truth()``."""
        points = self._shared_metadata.nearest_positions[..., self._nearest_points_slice, :]
        return points.reshape(*points.shape[:-2], *self._probe_layout_shape, 3)
