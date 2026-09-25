from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.options.sensors import SurfaceDistanceProbe as SurfaceDistanceProbeOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.raycast_qd import closest_point_on_triangle

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata
from .probe import ProbeSensorMetadataMixin, ProbeSensorMixin, func_noised_probe_radius, get_measured_bufs
from .tactile_shared import (
    BVH_LEAF_SIZE,
    BVH_STACK_SIZE,
    BVHMetadata,
    ChunkedBVHData,
    build_static_chunk_bvh,
    func_sphere_intersects_aabb,
    func_vec3_at,
    get_mesh_geom_chunks,
)

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@dataclass
class TriangleMeshBVH(BVHMetadata):
    """
    BVH over tracked mesh triangles for one sensor class.

    ``leaf_elem_idx`` entries are absolute rows into ``tri_verts``, a flat per-class table of link-local triangle
    vertices (shape ``(total_n_tri, 3, 3)``: per triangle, three xyz vertex positions). See ``BVHMetadata`` for the
    shared scaffolding semantics. Rigid-link assumption: built once at scene init, never rebuilt.
    """

    tri_verts: torch.Tensor = make_tensor_field((0, 3, 3))

    def append_sensor(self, track_link_idx: np.ndarray, solver) -> None:
        """
        Build per-tracked-link chunks for one sensor (link-local triangle BVH) and append into the flat tensors.

        Sensors with no tracked-link geometry register zero chunks; the kernel's per-sensor chunk loop iterates
        ``[0, sensor_chunk_count[i_s])`` and is a no-op for those.
        """
        new_chunk_link_idx: list[int] = []
        new_chunk_node_start: list[int] = []
        new_chunk_node_count: list[int] = []
        chunk_node_min: list[np.ndarray] = []
        chunk_node_max: list[np.ndarray] = []
        chunk_node_left: list[np.ndarray] = []
        chunk_node_right: list[np.ndarray] = []
        chunk_node_leaf_start: list[np.ndarray] = []
        chunk_node_leaf_count: list[np.ndarray] = []
        chunk_leaf_elem_idx: list[np.ndarray] = []
        chunk_tri_verts: list[np.ndarray] = []

        chunk_start_for_sensor = int(self.chunk_link_idx.shape[0])
        node_offset = int(self.node_min.shape[0])
        leaf_offset = int(self.leaf_elem_idx.shape[0])
        tri_offset = int(self.tri_verts.shape[0])

        for i_l in range(int(track_link_idx.shape[0])):
            link_idx = int(track_link_idx[i_l])
            link = solver.links[link_idx]
            geom_chunks = get_mesh_geom_chunks(link, prefer_visual=False)
            if not geom_chunks:
                continue
            # Concatenate triangles from all geoms of this link into one chunk.
            tri_v0_list: list[np.ndarray] = []
            tri_v1_list: list[np.ndarray] = []
            tri_v2_list: list[np.ndarray] = []
            for _geom, verts_link, faces in geom_chunks:
                tri_v0_list.append(verts_link[faces[:, 0]])
                tri_v1_list.append(verts_link[faces[:, 1]])
                tri_v2_list.append(verts_link[faces[:, 2]])
            v0 = np.concatenate(tri_v0_list, axis=0).astype(gs.np_float, copy=False)
            v1 = np.concatenate(tri_v1_list, axis=0).astype(gs.np_float, copy=False)
            v2 = np.concatenate(tri_v2_list, axis=0).astype(gs.np_float, copy=False)
            n_tri = int(v0.shape[0])
            if n_tri == 0:
                continue

            centroids = (v0 + v1 + v2) / 3.0
            aabb_mins = np.minimum(np.minimum(v0, v1), v2)
            aabb_maxs = np.maximum(np.maximum(v0, v1), v2)

            tri_stack = np.stack((v0, v1, v2), axis=1)  # (n_tri, 3, 3)
            global_rows = (tri_offset + np.arange(n_tri, dtype=gs.np_int)).astype(gs.np_int)

            nmin, nmax, nleft, nright, lstart, lcount, eidx = build_static_chunk_bvh(
                centroids, aabb_mins, aabb_maxs, global_rows, BVH_LEAF_SIZE
            )

            new_chunk_link_idx.append(link_idx)
            new_chunk_node_start.append(node_offset)
            new_chunk_node_count.append(int(nmin.shape[0]))

            chunk_node_min.append(nmin)
            chunk_node_max.append(nmax)
            # Rebase intra-chunk child / leaf-start indices into the flat tensors' absolute space.
            chunk_node_left.append(np.where(nleft >= 0, nleft + node_offset, nleft).astype(gs.np_int))
            chunk_node_right.append(np.where(nright >= 0, nright + node_offset, nright).astype(gs.np_int))
            chunk_node_leaf_start.append(np.where(lcount > 0, lstart + leaf_offset, lstart).astype(gs.np_int))
            chunk_node_leaf_count.append(lcount)
            chunk_leaf_elem_idx.append(eidx)
            chunk_tri_verts.append(tri_stack.astype(gs.np_float, copy=False))

            node_offset += int(nmin.shape[0])
            leaf_offset += int(eidx.shape[0])
            tri_offset += n_tri

        if not new_chunk_link_idx:
            # No tracked links contributed geometry; record zero chunks for this sensor.
            self.sensor_chunk_start = concat_with_tensor(self.sensor_chunk_start, chunk_start_for_sensor, expand=(1,))
            self.sensor_chunk_count = concat_with_tensor(self.sensor_chunk_count, 0, expand=(1,))
            return

        node_min_cat = torch.tensor(np.concatenate(chunk_node_min, axis=0), dtype=gs.tc_float, device=gs.device)
        node_max_cat = torch.tensor(np.concatenate(chunk_node_max, axis=0), dtype=gs.tc_float, device=gs.device)
        node_left_cat = torch.tensor(np.concatenate(chunk_node_left, axis=0), dtype=gs.tc_int, device=gs.device)
        node_right_cat = torch.tensor(np.concatenate(chunk_node_right, axis=0), dtype=gs.tc_int, device=gs.device)
        node_leaf_start_cat = torch.tensor(
            np.concatenate(chunk_node_leaf_start, axis=0), dtype=gs.tc_int, device=gs.device
        )
        node_leaf_count_cat = torch.tensor(
            np.concatenate(chunk_node_leaf_count, axis=0), dtype=gs.tc_int, device=gs.device
        )
        leaf_elem_idx_cat = torch.tensor(np.concatenate(chunk_leaf_elem_idx, axis=0), dtype=gs.tc_int, device=gs.device)
        tri_verts_cat = torch.tensor(np.concatenate(chunk_tri_verts, axis=0), dtype=gs.tc_float, device=gs.device)
        chunk_link_idx_cat = torch.tensor(new_chunk_link_idx, dtype=gs.tc_int, device=gs.device)
        chunk_node_start_cat = torch.tensor(new_chunk_node_start, dtype=gs.tc_int, device=gs.device)
        chunk_node_count_cat = torch.tensor(new_chunk_node_count, dtype=gs.tc_int, device=gs.device)

        self.node_min = concat_with_tensor(self.node_min, node_min_cat, expand=(node_min_cat.shape[0], 3))
        self.node_max = concat_with_tensor(self.node_max, node_max_cat, expand=(node_max_cat.shape[0], 3))
        self.node_left = concat_with_tensor(self.node_left, node_left_cat, expand=(node_left_cat.shape[0],))
        self.node_right = concat_with_tensor(self.node_right, node_right_cat, expand=(node_right_cat.shape[0],))
        self.node_leaf_start = concat_with_tensor(
            self.node_leaf_start, node_leaf_start_cat, expand=(node_leaf_start_cat.shape[0],)
        )
        self.node_leaf_count = concat_with_tensor(
            self.node_leaf_count, node_leaf_count_cat, expand=(node_leaf_count_cat.shape[0],)
        )
        self.leaf_elem_idx = concat_with_tensor(
            self.leaf_elem_idx, leaf_elem_idx_cat, expand=(leaf_elem_idx_cat.shape[0],)
        )
        self.tri_verts = concat_with_tensor(self.tri_verts, tri_verts_cat, expand=(tri_verts_cat.shape[0], 3, 3))
        self.chunk_link_idx = concat_with_tensor(
            self.chunk_link_idx, chunk_link_idx_cat, expand=(chunk_link_idx_cat.shape[0],)
        )
        self.chunk_node_start = concat_with_tensor(
            self.chunk_node_start, chunk_node_start_cat, expand=(chunk_node_start_cat.shape[0],)
        )
        self.chunk_node_count = concat_with_tensor(
            self.chunk_node_count, chunk_node_count_cat, expand=(chunk_node_count_cat.shape[0],)
        )
        self.sensor_chunk_start = concat_with_tensor(self.sensor_chunk_start, chunk_start_for_sensor, expand=(1,))
        self.sensor_chunk_count = concat_with_tensor(self.sensor_chunk_count, len(new_chunk_link_idx), expand=(1,))


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
    bvh_tri_verts: qd.types.ndarray(),
    positions_gt: qd.types.ndarray(),
    positions_measured: qd.types.ndarray(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    dyn_state: array_class.DynState,
):
    """
    BVH-accelerated surface-distance query.

    Per ``(probe, env)``: transform the probe into each tracked-link's local frame, traverse the
    per-(sensor, tracked-link) static BVH with a fixed-depth stack, cull nodes via sphere-vs-AABB with
    radius squared = current best (the larger of GT / measured branch), and on leaf nodes call
    closest-point-on-triangle against the stored link-local vertices. The closest world-frame point is
    written to ``positions_*`` and the distance to ``output_*``.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        link_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        link_quat = dyn_state.links.quat[sensor_link_idx, i_b]

        probe_local = func_vec3_at(i_p, probe_positions_local)
        probe_world = link_pos + gu.qd_transform_by_quat(probe_local, link_quat)

        max_r_gt = probe_radii[i_p]
        best_dist_sq_gt = max_r_gt * max_r_gt
        best_point_gt = probe_world

        probe_radius_noise = probe_radii_noise[i_p]
        use_noised_radius = probe_radius_noise > gs.EPS
        max_r_m = max_r_gt
        if use_noised_radius:
            max_r_m = func_noised_probe_radius(max_r_gt, probe_radius_noise)
        best_dist_sq_m = max_r_m * max_r_m
        best_point_m = probe_world

        chunk_start = bvh.sensor_chunk_start[i_s]
        n_chunks = bvh.sensor_chunk_count[i_s]
        for c_off in range(n_chunks):
            i_c = chunk_start + c_off
            track_link_idx = bvh.chunk_link_idx[i_c]
            track_pos = dyn_state.links.pos[track_link_idx, i_b]
            track_quat = dyn_state.links.quat[track_link_idx, i_b]
            # BVH lives in the tracked link's local frame; bring the probe over.
            probe_link = gu.qd_inv_transform_by_trans_quat(probe_world, track_pos, track_quat)

            stack = qd.Vector.zero(gs.qd_int, qd.static(BVH_STACK_SIZE))
            stack[0] = bvh.chunk_node_start[i_c]
            stack_idx = 1

            while stack_idx > 0:
                stack_idx -= 1
                n = stack[stack_idx]
                bmin = func_vec3_at(n, bvh.node_min)
                bmax = func_vec3_at(n, bvh.node_max)
                # Cull when min distance from probe to AABB exceeds the conservative current best.
                cull_radius_sq = qd.max(best_dist_sq_gt, best_dist_sq_m)
                if not func_sphere_intersects_aabb(probe_link, cull_radius_sq, bmin, bmax):
                    continue
                left = bvh.node_left[n]
                if left == -1:
                    fstart = bvh.node_leaf_start[n]
                    fn = bvh.node_leaf_count[n]
                    for j in range(fn):
                        i_f = bvh.leaf_elem_idx[fstart + j]
                        v0 = qd.Vector(
                            [bvh_tri_verts[i_f, 0, 0], bvh_tri_verts[i_f, 0, 1], bvh_tri_verts[i_f, 0, 2]],
                            dt=gs.qd_float,
                        )
                        v1 = qd.Vector(
                            [bvh_tri_verts[i_f, 1, 0], bvh_tri_verts[i_f, 1, 1], bvh_tri_verts[i_f, 1, 2]],
                            dt=gs.qd_float,
                        )
                        v2 = qd.Vector(
                            [bvh_tri_verts[i_f, 2, 0], bvh_tri_verts[i_f, 2, 1], bvh_tri_verts[i_f, 2, 2]],
                            dt=gs.qd_float,
                        )
                        closest_link = closest_point_on_triangle(probe_link, v0, v1, v2)
                        diff = closest_link - probe_link
                        dist_sq = diff.dot(diff)
                        if dist_sq < best_dist_sq_gt or (use_noised_radius and dist_sq < best_dist_sq_m):
                            # Transform the hit back to world frame and record on whichever branch tightened.
                            closest_world = track_pos + gu.qd_transform_by_quat(closest_link, track_quat)
                            if dist_sq < best_dist_sq_gt:
                                best_dist_sq_gt = dist_sq
                                best_point_gt = closest_world
                            if use_noised_radius and dist_sq < best_dist_sq_m:
                                best_dist_sq_m = dist_sq
                                best_point_m = closest_world
                else:
                    right = bvh.node_right[n]
                    # Median split bounds depth at log2(N / leaf_size) << BVH_STACK_SIZE; the guard mirrors the
                    # global rigid-BVH kernel so a future build strategy can't silently overflow the stack.
                    if stack_idx < qd.static(BVH_STACK_SIZE - 2):
                        stack[stack_idx] = left
                        stack[stack_idx + 1] = right
                        stack_idx += 2

        best_dist_gt = qd.sqrt(best_dist_sq_gt)
        best_dist_m = best_dist_gt
        if use_noised_radius:
            best_dist_m = qd.sqrt(best_dist_sq_m)
        else:
            for j in qd.static(range(3)):
                best_point_m[j] = best_point_gt[j]

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]

        output_gt[cache_start + probe_idx_in_sensor, i_b] = best_dist_gt
        output_measured[cache_start + probe_idx_in_sensor, i_b] = best_dist_m
        for j in qd.static(range(3)):
            positions_gt[i_b, i_p, j] = best_point_gt[j]
            positions_measured[i_b, i_p, j] = best_point_m[j]


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
    bvh: TriangleMeshBVH = field(default_factory=TriangleMeshBVH)


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
            bvh.tri_verts,
            shared_metadata.nearest_positions,
            shared_metadata.nearest_positions_measured,
            current_ground_truth_data_T,
            measured_cols_b,
            solver.dyn_state,
        )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        for obj in self._debug_objects:
            context.clear_debug_object(obj)
        self._debug_objects.clear()

        # Single env: drop the leading env axis to a bare (3,) / (4,); squeeze(0) leaves an unbatched vector untouched.
        link_pos = self._link.get_pos(env_idx, relative=False).squeeze(0)
        link_quat = self._link.get_quat(env_idx, relative=False).squeeze(0)
        probe_world = tensor_to_array(
            gu.transform_by_trans_quat(self._probe_local_pos.reshape(-1, 3), link_pos, link_quat)
        ).reshape(-1, 3)
        points = tensor_to_array(self.nearest_points[env_idx]).reshape(-1, 3)

        rgb = tuple(float(c) for c in self._options.debug_probe_color)
        line_color = (*rgb, 1.0)
        self._debug_objects.extend(self._draw_probe_spheres(context, probe_world, rgb))
        self._debug_objects.append(
            context.draw_debug_spheres(
                poss=points, radius=float(self._options.debug_probe_center_radius), color=line_color
            )
        )
        for i in range(len(probe_world)):
            self._debug_objects.append(
                context.draw_debug_line(
                    probe_world[i],
                    points[i],
                    radius=float(self._options.debug_probe_center_radius) / 4.0,
                    color=line_color,
                )
            )

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
