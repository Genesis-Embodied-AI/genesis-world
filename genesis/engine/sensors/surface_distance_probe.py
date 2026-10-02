from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.options.sensors import SurfaceDistanceProbe as SurfaceDistanceProbeOptions
from genesis.typing import EnvsIdxType
from genesis.utils.misc import tensor_to_array
from genesis.utils.raycast_qd import closest_point_on_triangle

from .base_sensor import LinkAttachedSensorMixin, RigidSensorArrayMixin, SimpleSensor, SimpleSensorArray
from .probe import ProbeSensorArrayMixin, ProbeSensorMixin, func_noised_probe_radius
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
    from genesis.engine.solvers import RigidSolver
    from genesis.vis.rasterizer_context import RasterizerContext


@dataclass
class TriangleMeshBVH(BVHMetadata):
    """
    Bounding volume hierarchy (BVH) over the tracked mesh triangles of every sensor of one type.

    ``leaf_elem_idx`` entries are absolute rows into ``tri_verts``, a flat table of the link-local triangle vertices of
    the type (shape ``(total_n_tri, 3, 3)``: per triangle, three xyz vertex positions). See ``BVHMetadata`` for the
    shared scaffolding semantics.
    """

    tri_verts: torch.Tensor

    @classmethod
    def build(cls, sensors_track_link_idx: list[np.ndarray], solver: "RigidSolver") -> "TriangleMeshBVH":
        """
        Build the per-(sensor, tracked link) chunks of every sensor at once, one link-local triangle BVH each.

        A sensor whose tracked links carry no mesh geometry holds zero chunks, so the kernel's per-sensor chunk loop
        over ``[0, sensor_chunk_count[i_s])`` is empty for it.
        """
        sensor_chunk_start: list[int] = []
        sensor_chunk_count: list[int] = []
        chunk_link_idx: list[int] = []
        chunk_node_start: list[int] = []
        chunk_node_count: list[int] = []
        # Seeded with typed empties so the tables come out well-formed without any chunk
        node_min = [np.empty((0, 3), dtype=gs.np_float)]
        node_max = [np.empty((0, 3), dtype=gs.np_float)]
        node_left = [np.empty((0,), dtype=gs.np_int)]
        node_right = [np.empty((0,), dtype=gs.np_int)]
        node_leaf_start = [np.empty((0,), dtype=gs.np_int)]
        node_leaf_count = [np.empty((0,), dtype=gs.np_int)]
        leaf_elem_idx = [np.empty((0,), dtype=gs.np_int)]
        tri_verts = [np.empty((0, 3, 3), dtype=gs.np_float)]

        node_offset = 0
        leaf_offset = 0
        tri_offset = 0
        for track_link_idx in sensors_track_link_idx:
            sensor_chunk_start.append(len(chunk_link_idx))
            for link_idx in track_link_idx.tolist():
                geom_chunks = get_mesh_geom_chunks(solver.links[link_idx], prefer_visual=False)
                if not geom_chunks:
                    continue
                # One chunk per link, over the triangles of all its geoms
                v0 = np.concatenate([verts_link[faces[:, 0]] for _geom, verts_link, faces in geom_chunks], axis=0)
                v1 = np.concatenate([verts_link[faces[:, 1]] for _geom, verts_link, faces in geom_chunks], axis=0)
                v2 = np.concatenate([verts_link[faces[:, 2]] for _geom, verts_link, faces in geom_chunks], axis=0)
                n_tri = v0.shape[0]
                if n_tri == 0:
                    continue
                centroids = (v0 + v1 + v2) / 3.0
                aabb_mins = np.minimum(np.minimum(v0, v1), v2)
                aabb_maxs = np.maximum(np.maximum(v0, v1), v2)
                global_rows = tri_offset + np.arange(n_tri, dtype=gs.np_int)
                nmin, nmax, nleft, nright, lstart, lcount, eidx = build_static_chunk_bvh(
                    centroids, aabb_mins, aabb_maxs, global_rows, BVH_LEAF_SIZE
                )
                chunk_link_idx.append(link_idx)
                chunk_node_start.append(node_offset)
                chunk_node_count.append(nmin.shape[0])
                node_min.append(nmin)
                node_max.append(nmax)
                # Rebase intra-chunk child / leaf-start indices into the flat tensors' absolute space.
                node_left.append(np.where(nleft >= 0, nleft + node_offset, nleft))
                node_right.append(np.where(nright >= 0, nright + node_offset, nright))
                node_leaf_start.append(np.where(lcount > 0, lstart + leaf_offset, lstart))
                node_leaf_count.append(lcount)
                leaf_elem_idx.append(eidx)
                tri_verts.append(np.stack((v0, v1, v2), axis=1))
                node_offset += nmin.shape[0]
                leaf_offset += eidx.shape[0]
                tri_offset += n_tri
            sensor_chunk_count.append(len(chunk_link_idx) - sensor_chunk_start[-1])

        return cls(
            sensor_chunk_start=torch.tensor(sensor_chunk_start, dtype=gs.tc_int, device=gs.device),
            sensor_chunk_count=torch.tensor(sensor_chunk_count, dtype=gs.tc_int, device=gs.device),
            chunk_link_idx=torch.tensor(chunk_link_idx, dtype=gs.tc_int, device=gs.device),
            chunk_node_start=torch.tensor(chunk_node_start, dtype=gs.tc_int, device=gs.device),
            chunk_node_count=torch.tensor(chunk_node_count, dtype=gs.tc_int, device=gs.device),
            node_min=torch.tensor(np.concatenate(node_min), dtype=gs.tc_float, device=gs.device),
            node_max=torch.tensor(np.concatenate(node_max), dtype=gs.tc_float, device=gs.device),
            node_left=torch.tensor(np.concatenate(node_left), dtype=gs.tc_int, device=gs.device),
            node_right=torch.tensor(np.concatenate(node_right), dtype=gs.tc_int, device=gs.device),
            node_leaf_start=torch.tensor(np.concatenate(node_leaf_start), dtype=gs.tc_int, device=gs.device),
            node_leaf_count=torch.tensor(np.concatenate(node_leaf_count), dtype=gs.tc_int, device=gs.device),
            leaf_elem_idx=torch.tensor(np.concatenate(leaf_elem_idx), dtype=gs.tc_int, device=gs.device),
            tri_verts=torch.tensor(np.concatenate(tri_verts), dtype=gs.tc_float, device=gs.device),
        )


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
    eps: float,
):
    """
    Query the surface distance of every probe through a bounding volume hierarchy (BVH).

    For each (probe, env) pair the kernel transforms the probe into the local frame of each tracked link, traverses the
    static BVH of the (sensor, tracked link) pair with a fixed-depth stack, culls the nodes with a sphere versus
    axis-aligned bounding box (AABB) test at the squared radius of the current best (the larger of the ground truth (GT)
    and measured branches), and runs closest-point-on-triangle against the stored link-local vertices at the leaves. The
    closest world-frame point goes to ``positions_*`` and the distance to ``output_*``.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[0]

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
        use_noised_radius = probe_radius_noise > eps
        max_r_m = max_r_gt
        if use_noised_radius:
            max_r_m = func_noised_probe_radius(max_r_gt, probe_radius_noise, eps)
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

        output_gt[i_b, cache_start + probe_idx_in_sensor] = best_dist_gt
        output_measured[i_b, cache_start + probe_idx_in_sensor] = best_dist_m
        for j in qd.static(range(3)):
            positions_gt[i_b, i_p, j] = best_point_gt[j]
            positions_measured[i_b, i_p, j] = best_point_m[j]


class SurfaceDistanceProbeSensorArray(
    ProbeSensorArrayMixin, RigidSensorArrayMixin, SimpleSensorArray[SurfaceDistanceProbeOptions]
):
    """
    Array of every surface distance probe of the scene: distance and nearest point from the probes to the tracked mesh
    surfaces.

    The queries run through one static triangle-mesh BVH over the tracked links (see
    ``_kernel_surface_distance_probe_bvh``).
    """

    def build(self) -> None:
        super().build()

        _B = self._sim._B
        sensors_track_link_idx = [
            np.asarray(sensor.options.track_link_idx, dtype=gs.np_int) for sensor in self._sensors
        ]
        self.nearest_positions = torch.zeros((_B, self.total_n_probes, 3), dtype=gs.tc_float, device=gs.device)
        self.nearest_positions_measured = torch.zeros((_B, self.total_n_probes, 3), dtype=gs.tc_float, device=gs.device)
        # Rigid links keep their shape, so the link-local triangle BVH is built once for the whole scene
        self.bvh = TriangleMeshBVH.build(sensors_track_link_idx, self._sim.rigid_solver)

    def _get_return_format(self, options: SurfaceDistanceProbeOptions) -> tuple[int, ...]:
        # Mirror the probe layout so a grid ``probe_local_pos`` (M, N, 3) reads back as (..., M, N), consistent with
        # the other grid tactile sensors; a flat layout stays (..., n_probes). The cache is flat either way.
        return torch.tensor(options.probe_local_pos, dtype=gs.tc_float, device=gs.device).shape[:-1]

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        super().reset(envs_idx)

        # Pre-first-step placeholder. The kernel writes world-frame nearest points on each step; before that, an
        # uninitialized read returns zeros rather than misleading link-local positions.
        self.nearest_positions[envs_idx] = 0.0
        self.nearest_positions_measured[envs_idx] = 0.0

    def _update_current_timestep_data(self, ground_truth_slot_0: torch.Tensor, measured_slot_0: torch.Tensor) -> None:
        _kernel_surface_distance_probe_bvh(
            self.probe_sensor_idx,
            self.links_idx,
            self.sensors_cache_start,
            self.sensor_probe_start,
            self.probe_positions,
            self.probe_radii,
            self.probe_radii_noise,
            self.bvh.kernel_bvh,
            self.bvh.tri_verts,
            self.nearest_positions,
            self.nearest_positions_measured,
            ground_truth_slot_0,
            measured_slot_0,
            self.solver.dyn_state,
            eps=gs.EPS,
        )

    def nearest_points(self, i_s: int, is_ground_truth: bool = False) -> torch.Tensor:
        """The nearest mesh points of sensor ``i_s``, aligned with its readings.

        A grid ``probe_local_pos`` (M, N, 3) reads back as (..., M, N, 3), a flat layout as (..., n_probes, 3). The
        measured query uses the noisy radius, the ground-truth one the nominal radius.
        """
        positions = self.nearest_positions if is_ground_truth else self.nearest_positions_measured
        points = positions[..., self._probe_slice(i_s), :]
        return points.reshape(*points.shape[:-2], *self._sensors_probe_layout_shape[i_s], 3)

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        options = self._sensors[i_s].options
        env_idx = context.rendered_envs_idx[0] if self._sim.n_envs > 0 else None
        debug_objects = self._debug_objects[i_s]
        for obj in debug_objects:
            context.clear_debug_object(obj)
        debug_objects.clear()

        # Single env: drop the leading env axis to a bare (3,) / (4,); squeeze(0) leaves an unbatched vector untouched.
        link = self._links[i_s]
        link_pos = link.get_pos(env_idx, relative=False).squeeze(0)
        link_quat = link.get_quat(env_idx, relative=False).squeeze(0)
        probe_world = tensor_to_array(
            gu.transform_by_trans_quat(self._sensors_probe_local_pos[i_s], link_pos, link_quat)
        ).reshape(-1, 3)
        points = tensor_to_array(self.nearest_points(i_s)[env_idx]).reshape(-1, 3)
        rgb = options.debug_probe_color
        line_color = (*rgb, 1.0)
        debug_objects.extend(self._draw_probe_spheres(i_s, context, probe_world, rgb))
        debug_objects.append(
            context.draw_debug_spheres(poss=points, radius=options.debug_probe_center_radius, color=line_color)
        )
        for i in range(len(probe_world)):
            debug_objects.append(
                context.draw_debug_line(
                    probe_world[i],
                    points[i],
                    radius=options.debug_probe_center_radius / 4.0,
                    color=line_color,
                )
            )


class SurfaceDistanceProbeSensor(
    ProbeSensorMixin,
    LinkAttachedSensorMixin,
    SimpleSensor[SurfaceDistanceProbeOptions, SurfaceDistanceProbeSensorArray],
):
    """Sensor reading the distance and the nearest point from each of its probes to the tracked mesh surfaces."""

    @property
    def nearest_points(self) -> torch.Tensor:
        """Nearest mesh points for the measured (noisy-radius) query, aligned with ``read()``."""
        return self._array.nearest_points(self._idx)

    @property
    def nearest_points_ground_truth(self) -> torch.Tensor:
        """Nearest mesh points for the nominal-radius ground-truth query, aligned with ``read_ground_truth()``."""
        return self._array.nearest_points(self._idx, is_ground_truth=True)
