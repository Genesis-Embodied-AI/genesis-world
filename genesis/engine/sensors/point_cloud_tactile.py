import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Final, NamedTuple

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.options.sensors import (
    ElastomerTaxel as ElastomerTaxelSensorOptions,
    ProximityTaxel as ProximityTaxelOptions,
)
from genesis.typing import EnvsIdxType
from genesis.utils.misc import tensor_to_array
from genesis.utils.point_cloud import sample_mesh_point_cloud
from genesis.utils.raycast_qd import closest_point_on_triangle, get_triangle_vertices, triangle_face_normal

from .base_sensor import LinkAttachedSensorMixin, RigidSensorArrayMixin, SimpleSensor, SimpleSensorArray
from .probe import (
    ProbeSensorArrayMixin,
    ProbeSensorMixin,
    ProbesWithNormalSensorArrayMixin,
    func_noised_probe_radius,
)
from .tactile_shared import (
    BVH_LEAF_SIZE,
    BVH_STACK_SIZE,
    BVHMetadata,
    ChunkedBVHData,
    ContactDepthQueryArrayMixin,
    GridFFTConvArrayMixin,
    SpatialCrosstalkArrayMixin,
    ViscoelasticHysteresisArrayMixin,
    build_grid_fft,
    build_static_chunk_bvh,
    func_aabb_intersects_aabb,
    func_sphere_intersects_aabb,
    func_vec3_at,
    get_mesh_geom_chunks,
    next_pow2,
)

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom
    from genesis.engine.solvers import RigidSolver
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext


# Conservative cap for global-BVH closest-point walks in raycast mode. Points farther than this from every candidate
# triangle map to depth = 0 (so the elastomer "out of contact" branch fires). Sized to cover realistic elastomer
# penetrations -- bumping it widens BVH traversal cost but doesn't change correctness for in-contact probes.
_ELASTOMER_RAYCAST_QUERY_DIST = 0.1


def _n_sample_points_per_link(n_sample_points: int | list | tuple, n_links: int) -> list[int]:
    if n_links <= 0:
        return []
    if isinstance(n_sample_points, (list, tuple)):
        counts = [int(x) for x in n_sample_points]
        if len(counts) != n_links:
            gs.raise_exception(
                f"Point cloud tactile n_sample_points length must match track_link_idx ({n_links}), got {len(counts)}."
            )
        if any(c < 0 for c in counts):
            gs.raise_exception("n_sample_points entries must be non-negative.")
        return counts
    n_total = int(n_sample_points)
    if n_total < 0:
        gs.raise_exception("n_sample_points must be non-negative.")
    base, rem = divmod(n_total, n_links)
    return [base + (1 if i < rem else 0) for i in range(n_links)]


class GridFFTMeta(NamedTuple):
    """
    Record of one grid-shaped sensor on the fast Fourier transform (FFT) dilation path of HydroShear.

    ``sensor_idx``, ``g_ny``, ``g_nx``, ``probe_start`` and ``cache_start`` are the leading fields every grid-FFT sensor
    shares, the contract ``build_grid_fft`` relies on. ``lambda_d``, ``spacing_u``, ``spacing_v``, ``compressibility``
    and ``dilation_reg`` are the HydroShear kernel parameters ``_dilate_kernel_builder`` consumes: ``compressibility``
    blends the local Gaussian kernel (1) and the incompressible 1/r kernel (0), and ``dilation_reg`` is the resolved
    epsilon in meters.
    """

    sensor_idx: int
    g_ny: int
    g_nx: int
    probe_start: int
    cache_start: int
    lambda_d: float
    spacing_u: float
    spacing_v: float
    compressibility: float
    dilation_reg: float
    elastomer_thickness: float = 0.0


def _fill_candidate_geom_mask(
    mask: torch.Tensor, geom_starts: torch.Tensor, geom_ns: torch.Tensor, geom_idx: torch.Tensor
) -> None:
    """
    Mark in the ``(B, n_sensors, n_geoms)`` bool ``mask``, per sensor, which scene geoms are candidates.

    ``geom_idx`` is the flat per-sensor concatenation of candidate geom indices; ``geom_starts``/``geom_ns`` give
    each sensor's slice into it. The marks are identical across all ``B`` environments.
    """
    starts = tensor_to_array(geom_starts)
    ns = tensor_to_array(geom_ns)
    idx = tensor_to_array(geom_idx)
    for i_s in range(mask.shape[1]):
        lo = int(starts[i_s])
        hi = lo + int(ns[i_s])
        if hi > lo:
            mask[:, i_s, idx[lo:hi]] = True


def _mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    tris = verts[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _split_count_by_area(n_total: int, geom_chunks: list[tuple[object, np.ndarray, np.ndarray]]) -> list[int]:
    n_chunks = len(geom_chunks)
    if n_chunks <= 0:
        return []
    if n_total <= 0:
        return [0] * n_chunks

    areas = np.asarray([_mesh_area(verts, faces) for _, verts, faces in geom_chunks], dtype=gs.np_float)
    if float(areas.sum()) <= gs.EPS:
        areas.fill(1.0)

    if n_total < n_chunks:
        counts = np.zeros(n_chunks, dtype=gs.np_int)
        counts[np.argsort(-areas)[:n_total]] = 1
        return counts.tolist()

    raw_extra = (n_total - n_chunks) * areas / float(areas.sum())
    extra = np.floor(raw_extra).astype(gs.np_int)
    remainder = n_total - n_chunks - int(extra.sum())
    if remainder > 0:
        extra[np.argsort(-(raw_extra - extra))[:remainder]] += 1
    return (extra + 1).tolist()


def _active_envs_mask_tensor(geom: "RigidGeom", batch_size: int) -> torch.Tensor:
    if geom.active_envs_mask is None:
        return torch.ones((batch_size,), dtype=gs.tc_bool, device=gs.device)
    return geom.active_envs_mask.to(device=gs.device, dtype=gs.tc_bool)


def _group_geoms_by_variant(
    geom_chunks: list[tuple[object, np.ndarray, np.ndarray]], batch_size: int
) -> list[tuple[torch.Tensor, list[tuple[object, np.ndarray, np.ndarray]]]]:
    """
    Partition the geoms of a link into heterogeneous-variant groups by ``active_envs_mask``.

    Geoms sharing a mask form one variant. ``None`` masks, the homogeneous case, collapse into a single all-True group.
    It returns ``[(mask, geom_chunks_for_variant), ...]`` and preserves the original geom order within each group.
    """
    groups: dict[bytes, tuple[torch.Tensor, list[tuple[object, np.ndarray, np.ndarray]]]] = {}
    for chunk in geom_chunks:
        geom = chunk[0]
        mask = _active_envs_mask_tensor(geom, batch_size)
        key = tensor_to_array(mask).astype(np.bool_).tobytes()
        if key not in groups:
            groups[key] = (mask, [])
        groups[key][1].append(chunk)
    return list(groups.values())


def _sample_track_links_point_cloud_tensors(
    solver: "RigidSolver", track_link_idx: np.ndarray, n_sample_points: int | list | tuple, prefer_visual: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample the meshes of the tracked links by farthest point sampling (FPS) into concatenated link-local positions.

    The per-link budget from ``n_sample_points`` goes to every heterogeneous variant of a link (geoms grouped by
    ``active_envs_mask``), so each parallel environment sees the full requested point count whatever variant is active.
    Within a variant, the budget is split across geoms by surface area.

    Returns
    -------
    idx_cat, pos_cat, active_cat
        Global link index per row, positions (N, 3), and active env mask (N, B), all on ``gs.device``.
    """
    n_per_link = _n_sample_points_per_link(n_sample_points, int(track_link_idx.shape[0]))
    if sum(n_per_link) == 0:
        gs.raise_exception("n_sample_points must allocate at least one sample in total.")

    link_idx_chunks: list[torch.Tensor] = []
    pos_chunks: list[torch.Tensor] = []
    active_chunks: list[torch.Tensor] = []

    for i_l in range(int(track_link_idx.shape[0])):
        n_pts = n_per_link[i_l]
        link_idx = int(track_link_idx[i_l])
        link = solver.links[link_idx]
        geom_chunks = get_mesh_geom_chunks(link, prefer_visual)
        if not geom_chunks:
            gs.raise_exception(f"No mesh geometry on tracked link index {link_idx}.")
        for variant_mask, variant_chunks in _group_geoms_by_variant(geom_chunks, solver._B):
            for n_geom_pts, (geom, verts, faces) in zip(_split_count_by_area(n_pts, variant_chunks), variant_chunks):
                if n_geom_pts <= 0:
                    continue
                # Fixed seed: the cache key already discriminates between meshes (vertices+faces hashed), so the same
                # mesh always resolves to the same sample, which keeps tactile readings reproducible across
                # build/reset cycles.
                pts_np = sample_mesh_point_cloud(verts, faces, n_geom_pts, seed=0, use_cache=True)

                li = torch.full((pts_np.shape[0],), link_idx, dtype=gs.tc_int, device=gs.device)
                link_idx_chunks.append(li)
                pos_chunks.append(torch.tensor(pts_np, dtype=gs.tc_float, device=gs.device))
                active_chunks.append(variant_mask.expand(pts_np.shape[0], solver._B))

    if not pos_chunks:
        gs.raise_exception("PointCloudTactile sensor produced an empty object point cloud.")

    return torch.cat(link_idx_chunks, dim=0), torch.cat(pos_chunks, dim=0), torch.cat(active_chunks, dim=0)


_ELASTOMER_QUERY_AABB_MARGIN = 1e-3


@dataclass
class PointCloudBVH(BVHMetadata):
    """
    BVH over the tracked point clouds of one sensor type.

    ``leaf_elem_idx`` entries are absolute rows into ``pc_pos_link`` / ``pc_active_envs_mask`` so a leaf hit resolves
    to per-point data with one indirection. See ``BVHMetadata`` for the shared scaffolding semantics.
    """

    # Inverse of sensor_chunk_start/count: chunk_sensor_idx[i_c] is the owning sensor's index. Enables
    # (env, chunk)-parallel kernels (e.g. ElastomerTaxel surface state) without rescanning sensor_chunk_start
    # in every thread; ProximityTaxel parallelizes per-probe and does not consume this field.
    chunk_sensor_idx: torch.Tensor

    @classmethod
    def build(cls, sensors_points: list[tuple[torch.Tensor, torch.Tensor]]) -> "PointCloudBVH":
        """
        Build the per-(sensor, tracked link) chunks of every sensor at once into the flat tensors.

        ``sensors_points`` holds the ``(link index, position)`` of the sampled points of each sensor. Each leaf's
        element index is an absolute row into the point tables of the type, which lay the points sensor after sensor
        in the same order.
        """
        sensor_chunk_start: list[int] = []
        sensor_chunk_count: list[int] = []
        chunk_link_idx: list[int] = []
        chunk_sensor_idx: list[int] = []
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

        pc_start_row = 0
        node_offset = 0
        point_offset = 0
        for i_s, (idx_cat, pos_cat) in enumerate(sensors_points):
            if pos_cat.shape[0] == 0:
                gs.raise_exception("PointCloudBVH cannot be built over an empty point cloud.")
            idx_np = tensor_to_array(idx_cat)
            pos_np = tensor_to_array(pos_cat)
            unique_links = np.unique(idx_np)
            sensor_chunk_start.append(len(chunk_link_idx))
            sensor_chunk_count.append(len(unique_links))
            for link_idx in unique_links.tolist():
                local_rows = np.flatnonzero(idx_np == link_idx)
                # The BVH builder indexes with the solver's integer width
                global_rows = (pc_start_row + local_rows).astype(gs.np_int)
                pts_link = pos_np[local_rows]
                # Point cloud: AABB per element is degenerate (the point itself), so pass the points as both
                # centroids and the per-element min/max bounds.
                nmin, nmax, nleft, nright, npstart, npn, pidx = build_static_chunk_bvh(
                    pts_link, pts_link, pts_link, global_rows, BVH_LEAF_SIZE
                )
                chunk_link_idx.append(link_idx)
                chunk_sensor_idx.append(i_s)
                chunk_node_start.append(node_offset)
                chunk_node_count.append(nmin.shape[0])
                node_min.append(nmin)
                node_max.append(nmax)
                # Rebase intra-chunk child / leaf-start indices into the flat tensors' absolute space.
                node_left.append(np.where(nleft >= 0, nleft + node_offset, nleft))
                node_right.append(np.where(nright >= 0, nright + node_offset, nright))
                node_leaf_start.append(np.where(npn > 0, npstart + point_offset, npstart))
                node_leaf_count.append(npn)
                leaf_elem_idx.append(pidx)
                node_offset += nmin.shape[0]
                point_offset += pidx.shape[0]
            pc_start_row += pos_cat.shape[0]

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
            chunk_sensor_idx=torch.tensor(chunk_sensor_idx, dtype=gs.tc_int, device=gs.device),
        )


@qd.kernel
def _kernel_point_cloud_proximity_taxel_bvh(
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    bvh: ChunkedBVHData,
    pc_pos_link: qd.types.ndarray(),
    pc_active_envs_mask: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_gains: qd.types.ndarray(),
    stiffness: qd.types.ndarray(),
    shear_coupling: qd.types.ndarray(),
    twist_scalar: qd.types.ndarray(),
    proximity_density_scale: qd.types.ndarray(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    taxel_signal_buf: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    eps: float,
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[0]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        s_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        s_quat = dyn_state.links.quat[sensor_link_idx, i_b]

        k_stiff = stiffness[i_s]
        k_shear = shear_coupling[i_s]
        k_twist = twist_scalar[i_s]
        dens = proximity_density_scale[i_s, i_b]
        n_probes = n_probes_per_sensor[i_s]
        cache_start = sensor_cache_start[i_s]
        _i_p = i_p - sensor_probe_start[i_s]

        s_vel = dyn_state.links.cd_vel[sensor_link_idx, i_b]
        s_ang = dyn_state.links.cd_ang[sensor_link_idx, i_b]
        s_com = dyn_state.links.root_COM[sensor_link_idx, i_b]

        probe_local = func_vec3_at(i_p, probe_positions_local)
        probe_world = s_pos + gu.qd_transform_by_quat(probe_local, s_quat)

        a_loc = func_vec3_at(i_p, probe_local_normal)
        a_w = gu.qd_transform_by_quat(a_loc, s_quat)
        a_norm = qd.sqrt(a_w.dot(a_w)) + eps
        for j in qd.static(range(3)):
            a_w[j] = a_w[j] / a_norm

        R_gt = probe_radii[i_p]
        R_gt_sq = R_gt * R_gt
        probe_radius_noise = probe_radii_noise[i_p]
        use_noised_radius = probe_radius_noise > eps
        R_m = R_gt
        if use_noised_radius:
            R_m = func_noised_probe_radius(R_gt, probe_radius_noise, eps)
        R_m_sq = R_m * R_m
        # Conservative traversal radius covers both branches; exact tests run per leaf candidate.
        R_query = qd.max(R_gt, R_m)
        R_query_sq = R_query * R_query

        v_tax = s_vel + s_ang.cross(probe_world - s_com)

        sum_p_gt = gs.qd_float(0.0)
        fv_gt = qd.Vector.zero(gs.qd_float, 3)
        omega_w_gt = gs.qd_float(0.0)
        sum_p_m = gs.qd_float(0.0)
        fv_m = qd.Vector.zero(gs.qd_float, 3)
        omega_w_m = gs.qd_float(0.0)

        chunk_start = bvh.sensor_chunk_start[i_s]
        n_chunks = bvh.sensor_chunk_count[i_s]
        for c_off in range(n_chunks):
            i_c = chunk_start + c_off
            track_link_idx = bvh.chunk_link_idx[i_c]
            track_pos = dyn_state.links.pos[track_link_idx, i_b]
            track_quat = dyn_state.links.quat[track_link_idx, i_b]
            rcom_o = dyn_state.links.root_COM[track_link_idx, i_b]
            cdv_o = dyn_state.links.cd_vel[track_link_idx, i_b]
            cda_o = dyn_state.links.cd_ang[track_link_idx, i_b]
            omega_c = (cda_o - s_ang).dot(a_w)  # Relative spin rate of the tracked link about the normal
            # BVH nodes live in tracked-link local frame: bring the probe sphere center over.
            probe_link = gu.qd_inv_transform_by_trans_quat(probe_world, track_pos, track_quat)

            stack = qd.Vector.zero(gs.qd_int, qd.static(BVH_STACK_SIZE))
            stack[0] = bvh.chunk_node_start[i_c]
            stack_idx = 1

            while stack_idx > 0:
                stack_idx -= 1
                n = stack[stack_idx]
                bmin = func_vec3_at(n, bvh.node_min)
                bmax = func_vec3_at(n, bvh.node_max)
                if not func_sphere_intersects_aabb(probe_link, R_query_sq, bmin, bmax):
                    continue
                left = bvh.node_left[n]
                if left == -1:
                    pstart = bvh.node_leaf_start[n]
                    pn = bvh.node_leaf_count[n]
                    for j in range(pn):
                        i_o = bvh.leaf_elem_idx[pstart + j]
                        if not pc_active_envs_mask[i_o, i_b]:
                            continue
                        pos_l = func_vec3_at(i_o, pc_pos_link)
                        d_link = pos_l - probe_link
                        dsq = d_link.dot(d_link)
                        dist = qd.sqrt(dsq)

                        hit_gt = dsq <= R_gt_sq and dist > eps
                        hit_m = use_noised_radius and dsq <= R_m_sq and dist > eps
                        if hit_gt or hit_m:
                            # d_link is the probe->point offset in the tracked-link frame; rotating it to world
                            # and adding to probe_world yields the point's world position without a second transform.
                            d_world = gu.qd_transform_by_quat(d_link, track_quat)
                            pw = probe_world + d_world
                            v_pc = cdv_o + cda_o.cross(pw - rcom_o)
                            v_rel = v_pc - v_tax
                            vdota = v_rel.dot(a_w)
                            v_t = qd.Vector.zero(gs.qd_float, 3)
                            for k2 in qd.static(range(3)):
                                v_t[k2] = v_rel[k2] - a_w[k2] * vdota

                            if hit_gt:
                                P_i_gt = R_gt - dist
                                if P_i_gt > 0.0:
                                    sum_p_gt = sum_p_gt + P_i_gt
                                    omega_w_gt = omega_w_gt + P_i_gt * omega_c
                                    for k2 in qd.static(range(3)):
                                        fv_gt[k2] = fv_gt[k2] + P_i_gt * v_t[k2]
                            if hit_m:
                                P_i_m = R_m - dist
                                if P_i_m > 0.0:
                                    sum_p_m = sum_p_m + P_i_m
                                    omega_w_m = omega_w_m + P_i_m * omega_c
                                    for k2 in qd.static(range(3)):
                                        fv_m[k2] = fv_m[k2] + P_i_m * v_t[k2]
                else:
                    right = bvh.node_right[n]
                    # Median split bounds depth at log2(N / leaf_size) << BVH_STACK_SIZE; the guard mirrors the
                    # global rigid-BVH kernel so a future build strategy can't silently overflow the stack.
                    if stack_idx < qd.static(BVH_STACK_SIZE - 2):
                        stack[stack_idx] = left
                        stack[stack_idx + 1] = right
                        stack_idx += 2

        if not use_noised_radius:
            sum_p_m = sum_p_gt
            omega_w_m = omega_w_gt
            for j in qd.static(range(3)):
                fv_m[j] = fv_gt[j]

        # Penetration-weighted average of the spin rate over contacts; the per-probe gain cancels in the ratio, so
        # this uses the pre-gain accumulators.
        omega_n_gt = gs.qd_float(0.0)
        if sum_p_gt > eps:
            omega_n_gt = omega_w_gt / sum_p_gt
        omega_n_m = gs.qd_float(0.0)
        if sum_p_m > eps:
            omega_n_m = omega_w_m / sum_p_m

        # Apply the per-(env, probe) gain to the measured accumulators; force is linear in them, so this scales it.
        gain_m = probe_gains[i_b, i_p]
        sum_p_m = sum_p_m * gain_m
        for j in qd.static(range(3)):
            fv_m[j] = fv_m[j] * gain_m

        taxel_signal_buf[i_b, i_p] = sum_p_m

        # Lever arm from the sensor link origin to the taxel, in world frame.
        lever_world = probe_world - s_pos

        force_world_gt = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(3)):
            force_world_gt[j] = k_stiff * dens * sum_p_gt * a_w[j]
        if k_shear > eps:
            for j in qd.static(range(3)):
                force_world_gt[j] = force_world_gt[j] + k_shear * dens * fv_gt[j]

        # Torque is the moment of the full force about the sensor link origin, minus a spin term along the normal
        # driven by the relative twist rate.
        torque_world_gt = lever_world.cross(force_world_gt)
        for j in qd.static(range(3)):
            torque_world_gt[j] = torque_world_gt[j] - a_w[j] * (k_twist * omega_n_gt)

        force_link_gt = gu.qd_inv_transform_by_quat(force_world_gt, s_quat)
        torque_link_gt = gu.qd_inv_transform_by_quat(torque_world_gt, s_quat)

        force_world_measured = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(3)):
            force_world_measured[j] = k_stiff * dens * sum_p_m * a_w[j]
        if k_shear > eps:
            for j in qd.static(range(3)):
                force_world_measured[j] = force_world_measured[j] + k_shear * dens * fv_m[j]

        torque_world_measured = lever_world.cross(force_world_measured)
        for j in qd.static(range(3)):
            torque_world_measured[j] = torque_world_measured[j] - a_w[j] * (k_twist * omega_n_m)

        force_link_measured = gu.qd_inv_transform_by_quat(force_world_measured, s_quat)
        torque_link_measured = gu.qd_inv_transform_by_quat(torque_world_measured, s_quat)

        force_start = cache_start + _i_p * 3
        torque_start = cache_start + n_probes * 3 + _i_p * 3
        for j in qd.static(range(3)):
            output_gt[i_b, force_start + j] = force_link_gt[j]
        for j in qd.static(range(3)):
            output_gt[i_b, torque_start + j] = torque_link_gt[j]
        for j in qd.static(range(3)):
            output_measured[i_b, force_start + j] = force_link_measured[j]
        for j in qd.static(range(3)):
            output_measured[i_b, torque_start + j] = torque_link_measured[j]


class PointCloudTactileArrayMixin(ProbeSensorArrayMixin, RigidSensorArrayMixin):
    """Array of point-cloud-tracked tactile sensors: the probes of every sensor plus the merged point cloud sampled on
    their tracked links, and the BVH over it."""

    def build(self) -> None:
        super().build()

        # The tracked links of each sensor, sampled into link-local points laid sensor after sensor
        sensors_points = [
            _sample_track_links_point_cloud_tensors(
                self._sim.rigid_solver,
                np.asarray(sensor.options.track_link_idx, dtype=gs.np_int),
                sensor.options.n_sample_points,
                sensor.options.use_visual_mesh,
            )
            for sensor in self._sensors
        ]
        self.pc_link_idx = torch.cat([idx_cat for idx_cat, _, _ in sensors_points])
        self.pc_pos_link = torch.cat([pos_cat for _, pos_cat, _ in sensors_points])
        self.pc_active_envs_mask = torch.cat([active_cat for _, _, active_cat in sensors_points])
        n_points = [pos_cat.shape[0] for _, pos_cat, _ in sensors_points]
        self.pc_starts = list(itertools.accumulate(n_points, initial=0))[:-1]
        self.sensor_pc_start = torch.tensor(self.pc_starts, dtype=gs.tc_int, device=gs.device)
        self.sensor_pc_n = torch.tensor(n_points, dtype=gs.tc_int, device=gs.device)
        # The leaves index the point tables just laid, sensor after sensor
        self.pc_bvh = PointCloudBVH.build([(idx_cat, pos_cat) for idx_cat, pos_cat, _ in sensors_points])

    def _pc_slice(self, i_s: int) -> slice:
        """The rows of sensor ``i_s`` in the merged point tables."""
        return slice(self.pc_starts[i_s], self.pc_starts[i_s] + int(self.sensor_pc_n[i_s].item()))

    def _draw_debug_probes(
        self,
        i_s: int,
        context: "RasterizerContext",
        color_groups_fn: Callable[[list[int] | None], list[tuple]] | None = None,
    ) -> tuple[list[int] | None, int, np.ndarray | None]:
        envs_idx, n_debug_envs, env_offsets = super()._draw_debug_probes(i_s, context, color_groups_fn)
        options = self._sensors[i_s].options
        # The sensor's slice of the merged point tables, one chunk per tracked link
        pc_slice = self._pc_slice(i_s)
        pc_link_idx = self.pc_link_idx[pc_slice]
        pc_pos_link = self.pc_pos_link[pc_slice]
        pc_active_envs_mask = self.pc_active_envs_mask[pc_slice]
        world_chunks: list[np.ndarray] = []
        for link_idx in torch.unique(pc_link_idx).tolist():
            mask = pc_link_idx == link_idx
            pos_local, active_envs_mask = pc_pos_link[mask], pc_active_envs_mask[mask]
            track_link = self._sim.rigid_solver.links[link_idx]
            if envs_idx is not None:
                active_mask = tensor_to_array(active_envs_mask[:, envs_idx].T).astype(bool)
                if not active_mask.any():
                    continue
                track_pos = track_link.get_pos(envs_idx, relative=False)[:, None, :]
                track_quat = track_link.get_quat(envs_idx, relative=False)[:, None, :]
                pc_world = gu.transform_by_trans_quat(pos_local[None, :, :], track_pos, track_quat)
                pc_world = tensor_to_array(pc_world) + env_offsets[:, None, :]
                world_chunks.append(pc_world[active_mask])
            else:
                active_mask = active_envs_mask[:, 0]
                pos_active = pos_local[active_mask]
                if pos_active.numel() == 0:
                    continue
                track_pos = track_link.get_pos(envs_idx, relative=False).reshape(3)
                track_quat = track_link.get_quat(envs_idx, relative=False).reshape(4)
                world_chunks.append(tensor_to_array(gu.transform_by_trans_quat(pos_active, track_pos, track_quat)))
        if world_chunks:
            self._debug_objects[i_s].append(
                context.draw_debug_spheres(
                    poss=np.concatenate(world_chunks, axis=0),
                    radius=float(options.debug_point_cloud_radius),
                    color=options.debug_point_cloud_color,
                )
            )
        return envs_idx, n_debug_envs, env_offsets


class ProximityTaxelReturnType(NamedTuple):
    """Force and torque estimates per taxel in the link frame."""

    force: torch.Tensor
    torque: torch.Tensor


class ProximityTaxelSensorArray(
    ViscoelasticHysteresisArrayMixin,
    SpatialCrosstalkArrayMixin,
    PointCloudTactileArrayMixin,
    ProbesWithNormalSensorArrayMixin,
    SimpleSensorArray[ProximityTaxelOptions, ProximityTaxelReturnType],
):
    """
    Array of every proximity taxel of the scene, reading the force and torque of each spherical taxel in the link frame
    against the tracked point clouds.
    """

    # Two channel groups: force xyz followed by torque xyz (probe-major within each group)
    _taxel_channel_groups = 2

    def build(self) -> None:
        super().build()

        sensors_options = [sensor.options for sensor in self._sensors]
        self.stiffness = torch.tensor(
            [sensor_options.stiffness for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.shear_coupling = torch.tensor(
            [sensor_options.shear_coupling for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.twist_scalar = torch.tensor(
            [sensor_options.twist_scalar for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        # Per sensor and environment, the density scalar over the number of its points active there
        density_scales = []
        for i_s, sensor_options in enumerate(sensors_options):
            active_count = self.pc_active_envs_mask[self._pc_slice(i_s)].sum(dim=0).clamp_min(1)
            density_scales.append(sensor_options.density_scalar / active_count.to(dtype=gs.tc_float))
        self.proximity_density_scale = torch.stack(density_scales)
        self.taxel_signal_buf = torch.zeros((self._sim._B, self.total_n_probes), dtype=gs.tc_float, device=gs.device)

    def _get_return_format(self, options: ProximityTaxelOptions) -> tuple[tuple[int, ...], ...]:
        shape = (*np.shape(options.probe_local_pos)[:-1], 3)
        return shape, shape

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        super().reset(envs_idx)

        self.taxel_signal_buf[envs_idx] = 0.0

    def _update_current_timestep_data(self, ground_truth_slot_0: torch.Tensor, measured_slot_0: torch.Tensor) -> None:
        bvh = self.pc_bvh
        _kernel_point_cloud_proximity_taxel_bvh(
            self.probe_sensor_idx,
            self.links_idx,
            self.sensors_cache_start,
            self.sensor_probe_start,
            self.probe_positions,
            self.probe_local_normal,
            self.n_probes_per_sensor,
            bvh.kernel_bvh,
            self.pc_pos_link,
            self.pc_active_envs_mask,
            self.probe_radii,
            self.probe_radii_noise,
            self.probe_gains,
            self.stiffness,
            self.shear_coupling,
            self.twist_scalar,
            self.proximity_density_scale,
            ground_truth_slot_0,
            measured_slot_0,
            self.taxel_signal_buf,
            self.solver.dyn_state,
            gs.EPS,
        )

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        def mask(envs_idx: EnvsIdxType | None) -> torch.Tensor:
            signals = self.taxel_signal_buf[:, self._probe_slice(i_s)]
            return tensor_to_array(signals[0] if envs_idx is None else signals[envs_idx]) >= gs.EPS

        self._draw_debug_probes(i_s, context, self._tactile_color_groups_fn(i_s, mask))


class ProximityTaxelSensor(
    ProbeSensorMixin,
    LinkAttachedSensorMixin,
    SimpleSensor[ProximityTaxelOptions, ProximityTaxelSensorArray],
):
    """
    Sensor reading the force and torque of each of its spherical taxels in the link frame against the tracked point
    clouds.
    """


@qd.func
def _func_elastomer_min_sdf_over_active_geoms(
    i_b: int,
    geom_start: int,
    geom_idx: qd.types.ndarray(),
    point_world: qd.types.vector(3),
    geom_n: int,
    geom_active_envs_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
) -> float:
    min_sdf = float(1.0e6)
    geom_end = geom_start + geom_n
    for i_gm in range(geom_start, geom_end):
        if not geom_active_envs_mask[i_gm, i_b]:
            continue
        i_g = geom_idx[i_gm]
        # AABB pre-cull: the geom is fully contained in its world AABB, so a point strictly outside
        # the AABB has sdf > 0 and can't be the min when any other geom contains the point. If no
        # geom contains the point, min_sdf stays at 1.0e6 -- callers map that to depth=0 and the
        # surface-state "exit" branch (sdf > sdf_exit), both correct.
        amin = dyn_state.geoms.aabb_min[i_g, i_b]
        amax = dyn_state.geoms.aabb_max[i_g, i_b]
        if (
            point_world[0] < amin[0]
            or point_world[0] > amax[0]
            or point_world[1] < amin[1]
            or point_world[1] > amax[1]
            or point_world[2] < amin[2]
            or point_world[2] > amax[2]
        ):
            continue
        sd = sdf.sdf_func_world(i_g, i_b, point_world, dyn_state.geoms, dyn_info.geoms, collider_info.sdf)
        if sd < min_sdf:
            min_sdf = sd
    return min_sdf


@qd.func
def _func_elastomer_tangent(vec: qd.types.vector(3), normal: qd.types.vector(3)) -> qd.types.vector(3):
    return vec - normal * vec.dot(normal)


@qd.func
def _func_elastomer_update_surface_anchor(
    i_b: int,
    i_o: int,
    sdf_value: float,
    point_sensor: qd.types.vector(3),
    sdf_enter: float,
    sdf_exit: float,
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
):
    if sdf_value > sdf_exit:
        surface_initialized_buf[i_b, i_o] = False
        for k in qd.static(range(3)):
            surface_entry_pos_sensor_buf[i_b, i_o, k] = 0.0
    elif (not surface_initialized_buf[i_b, i_o]) and sdf_value < -sdf_enter:
        surface_initialized_buf[i_b, i_o] = True
        for k in qd.static(range(3)):
            surface_entry_pos_sensor_buf[i_b, i_o, k] = point_sensor[k]


@qd.func
def _func_elastomer_direct_dilate_contribution(
    source_pos: qd.types.vector(3),
    target_pos: qd.types.vector(3),
    target_normal: qd.types.vector(3),
    depth: float,
    lam: float,
    scale: float,
    normal_exponent: float,
    compressibility: float,
    eps: float,
) -> qd.types.vector(3):
    """
    Dilation contribution of a single tracked point to a target probe.

    Tangential spreading is linear in penetration depth, while the out-of-plane bulge follows a
    ``depth ** normal_exponent`` power law (mirrors the FFT path's H / H**normal_exponent channel split). The normal
    bulge always keeps the Gaussian falloff; the in-plane term is set by ``compressibility`` (1 = local
    Gaussian first-moment, 0 = incompressible ``r_hat/r``, in-between = peak-normalized blend).
    """
    planar_diff = _func_elastomer_tangent(target_pos - source_pos, target_normal)
    r2 = planar_diff.dot(planar_diff)
    gaussian = qd.exp(-lam * r2)
    normal_bulge = target_normal * qd.pow(depth, normal_exponent) * gaussian
    w = depth * gaussian  # compressibility >= 1: pure local Gaussian
    if compressibility < 1.0:
        inv = gs.qd_float(1.0) / (r2 + eps * eps)
        if compressibility <= 0.0:  # pure incompressible r_hat / r
            w = depth * inv
        else:  # blend, each kernel peak-normalized (see the FFT builder for the closed-form peaks)
            # math.exp(-0.5) == 1/sqrt(e) is the peak of r*exp(-lambda r^2); read it inline rather than from a
            # module constant so the fastcache purity check stays happy (the math module is an allowed capture).
            norm_g = gs.qd_float(qd.static(math.exp(-0.5))) / qd.sqrt(gs.qd_float(2.0) * lam)
            norm_i = gs.qd_float(1.0) / (gs.qd_float(2.0) * eps)
            w = depth * (compressibility * gaussian / norm_g + (gs.qd_float(1.0) - compressibility) * inv / norm_i)
    return (planar_diff * w + normal_bulge) * scale


@qd.func
def _func_elastomer_direct_shear_contribution(
    point_sensor: qd.types.vector(3),
    entry_sensor: qd.types.vector(3),
    probe_pos: qd.types.vector(3),
    probe_normal: qd.types.vector(3),
    depth: float,
    lam: float,
    scale: float,
    eps: float,
) -> qd.types.vector(3):
    shear_disp = point_sensor - entry_sensor
    shear_tangent = _func_elastomer_tangent(shear_disp, probe_normal)
    contribution = qd.Vector.zero(gs.qd_float, 3)
    if shear_tangent.dot(shear_tangent) > eps * eps:
        diff = probe_pos - point_sensor
        planar_diff = _func_elastomer_tangent(diff, probe_normal)
        contribution = shear_tangent * (depth * qd.exp(-lam * planar_diff.dot(planar_diff)) * scale)
    return contribution


def _collect_collision_geom_idx(solver: "RigidSolver", track_link_idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    geom_idx: list[int] = []
    active_masks: list[torch.Tensor] = []
    for link_idx in track_link_idx:
        link_i = int(link_idx)
        if link_i < 0 or link_i >= len(solver.links):
            gs.raise_exception(f"ElastomerTaxel track_link_idx contains invalid global link index {link_i}.")
        link = solver.links[link_i]
        for geom in link.geoms:
            geom_idx.append(int(geom.idx))
            active_masks.append(_active_envs_mask_tensor(geom, solver._B))
    if not geom_idx:
        gs.raise_exception("ElastomerTaxel tracked links must have collision geometry for SDF queries.")
    return torch.tensor(geom_idx, dtype=gs.tc_int, device=gs.device), torch.stack(active_masks, dim=0)


# Clamp bounds for q = |k| * h in _bonded_layer_transfer, chosen where S(q) is already flat.
_LAYER_Q_MIN: Final[float] = 1e-3
_LAYER_Q_MAX: Final[float] = 30.0


@torch.jit.script
def _bonded_layer_transfer(q: torch.Tensor, q_min: float = _LAYER_Q_MIN, q_max: float = _LAYER_Q_MAX) -> torch.Tensor:
    """In-plane transfer ``S(q)`` of an incompressible elastic layer of thickness ``h`` bonded to a rigid base.

    The top surface is shear-free with a prescribed normal displacement. For dimensionless wavenumber ``q = |k| * h``,
    the tangential surface displacement spectrum is ``-i * k_hat * S(q) * H_hat`` from the height spectrum ``H_hat``.

    ``S(q) = 2 q^2 / (sinh(2q) - 2q)`` is the exact per-mode solution: it grows as ``1.5/q`` for small ``q``
    (thin-layer squeeze flow, the free-space ``1/r``), peaks near ``q ~ 1``, and decays to ``0`` for large ``q``
    (incompressible half-space limit). ``q`` is clamped to where ``S`` is already flat.
    """
    q = q.clamp(min=q_min, max=q_max)
    two_q = 2.0 * q
    two_q_sq = two_q * two_q
    # sinh(2q) - 2q cancels for small q: sum its Taylor series (2q)^(2n+1)/(2n+1)! by recurrence, direct otherwise.
    term = two_q * two_q_sq / 6.0  # first term, (2q)^3 / 3!
    series = term
    for n in range(2, 6):  # n_terms = 5, float32-accurate at the direct-form crossover
        term = term * two_q_sq / ((2 * n) * (2 * n + 1))
        series = series + term
    denom = torch.where(two_q < 1.0, series, torch.sinh(two_q) - two_q)
    return 2.0 * q * q / denom


@torch.jit.script
def _precompute_hydroshear_dilate_kernel_fft(
    lambda_d: float,
    grid_spacing: tuple[float, float],
    fft_n: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
    compressibility: float = 1.0,
    dilation_reg: float = 0.0,
    elastomer_thickness: float = 0.0,
) -> torch.Tensor:
    """Real FFT of the 3-plane HydroShear dilation kernel ``(Ku, Kv, Kn)``.

    ``fft_n`` is ``(fft_ny, fft_nx)`` row-major: axis 0 spans the tangent_v direction, axis 1 the tangent_u
    direction. ``grid_spacing`` is ``(spacing_u, spacing_v)``. The output is a complex
    ``(3, fft_ny, fft_nx // 2 + 1)`` half-spectrum ready to multiply against ``rfft2(field)``.

    The in-plane planes ``(Ku, Kv)`` blend a local and a global kernel by ``compressibility`` (1 = local only,
    0 = global only, each peak-normalized in between). Local: the first-moment Gaussian
    ``offset * exp(-lambda_d r^2)``. Global: with ``elastomer_thickness`` set, the exact bonded incompressible
    layer transfer ``-i k_hat S(|k| h)`` (see ``_bonded_layer_transfer``), built directly in k-space; with a zero
    thickness the free-space ``offset / (r^2 + eps^2)`` (gradient of the 2D inverse-Laplacian, ``~1/r``). The normal
    plane ``Kn`` is the Gaussian bulge in both cases.
    """
    iv = torch.arange(fft_n[0], dtype=dtype, device=device)
    iu = torch.arange(fft_n[1], dtype=dtype, device=device)
    vv, uu = torch.meshgrid(
        (iv - fft_n[0] // 2) * grid_spacing[1], (iu - fft_n[1] // 2) * grid_spacing[0], indexing="ij"
    )
    r2 = uu * uu + vv * vv
    g = torch.exp(torch.tensor(-lambda_d, dtype=dtype, device=device) * r2)
    if compressibility >= 1.0:
        k = torch.stack((uu * g, vv * g, g), dim=0)
        return torch.fft.rfft2(torch.fft.ifftshift(k, dim=(-2, -1)))

    if elastomer_thickness > 0.0:
        kv1 = 2.0 * math.pi * torch.fft.fftfreq(fft_n[0], d=grid_spacing[1], dtype=dtype, device=device)
        ku1 = 2.0 * math.pi * torch.fft.rfftfreq(fft_n[1], d=grid_spacing[0], dtype=dtype, device=device)
        kvv, kuu = torch.meshgrid(kv1, ku1, indexing="ij")
        kmag = torch.sqrt(kvv * kvv + kuu * kuu)
        s_tf = torch.where(kmag > 0.0, _bonded_layer_transfer(kmag * elastomer_thickness), torch.zeros_like(kmag))
        kmag_safe = kmag.clamp(min=eps)
        gu_hat = (-1j) * (kuu / kmag_safe) * s_tf
        gv_hat = (-1j) * (kvv / kmag_safe) * s_tf
        # Peak of the real-space kernel magnitude, for the blend normalization below.
        norm_i = float(
            torch.sqrt(torch.fft.irfft2(gu_hat, s=fft_n) ** 2 + torch.fft.irfft2(gv_hat, s=fft_n) ** 2).max()
        )
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        gu_hat = gu_hat.to(cdtype)
        gv_hat = gv_hat.to(cdtype)
    else:
        eps_reg = dilation_reg if dilation_reg > 0.0 else 0.5 * (grid_spacing[0] + grid_spacing[1])
        inv = 1.0 / (r2 + eps_reg * eps_reg)
        sp = torch.fft.rfft2(torch.fft.ifftshift(torch.stack((uu * inv, vv * inv), dim=0), dim=(-2, -1)))
        gu_hat, gv_hat = sp[0], sp[1]
        norm_i = 1.0 / (2.0 * eps_reg)  # peak of r/(r^2+eps^2) at r=eps_reg

    kn_hat = torch.fft.rfft2(torch.fft.ifftshift(g, dim=(-2, -1)))
    if compressibility <= 0.0:
        ku_hat, kv_hat = gu_hat, gv_hat
    else:
        loc = torch.fft.rfft2(torch.fft.ifftshift(torch.stack((uu * g, vv * g), dim=0), dim=(-2, -1)))
        norm_g = math.exp(-0.5) / math.sqrt(2.0 * lambda_d)  # 1/sqrt(e) is the peak of r*exp(-lambda_d r^2)
        c = compressibility
        ku_hat = c * loc[0] / norm_g + (1.0 - c) * gu_hat / norm_i
        kv_hat = c * loc[1] / norm_g + (1.0 - c) * gv_hat / norm_i
    return torch.stack((ku_hat, kv_hat, kn_hat), dim=0)


def _dilate_kernel_builder(meta_entry: GridFFTMeta, fft_n: tuple[int, int]) -> torch.Tensor:
    """Build the three HydroShear dilation kernel planes ``(Ku, Kv, Kn)`` for ``build_grid_fft``."""
    return _precompute_hydroshear_dilate_kernel_fft(
        meta_entry.lambda_d,
        (meta_entry.spacing_u, meta_entry.spacing_v),
        fft_n,
        gs.device,
        gs.tc_float,
        gs.EPS,
        meta_entry.compressibility,
        meta_entry.dilation_reg,
        meta_entry.elastomer_thickness,
    )


@qd.func
def _func_elastomer_min_signed_dist_bvh(
    i_t: int,
    i_b: int,
    i_s: int,
    probe_world: qd.types.vector(3),
    track_geom_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    bvh_tree_state: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    max_query_dist: float,
) -> float:
    """
    Return the signed distance from ``probe_world`` to the nearest triangle of any geom flagged for this sensor in
    ``track_geom_mask``, through a bounding volume hierarchy (BVH).

    ``track_geom_mask`` has shape ``(B, n_sensors, n_geoms)``. The sign is positive when the probe is outside the
    surface (the face normal of the closest triangle points away from the probe) and negative inside. The return mirrors
    ``_func_elastomer_min_sdf_over_active_geoms``, so callers consume ``max(0, -signed)`` identically.

    ``max_query_dist`` is the cull radius of the BVH: a probe farther than that from every candidate triangle counts as
    fully outside and returns ``+max_query_dist``, which maps to a depth of 0 downstream.
    """
    # The tree's own leaf count: a compacted-subset tree (see RaycastContext.activate) has fewer leaves than faces.
    n_triangles = bvh_tree_state.leaves_idx.shape[1]
    best_dist = max_query_dist
    best_dist_sq = best_dist * best_dist
    best_signed = max_query_dist

    # Stack depth, see func_bvh_query_aabb in genesis.engine.bvh
    STACK_SIZE = qd.static(64)
    node_stack = qd.Vector.zero(gs.qd_int, STACK_SIZE)
    node_stack[0] = 0
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]

        if not func_sphere_intersects_aabb(
            probe_world, best_dist_sq, bvh_tree_state.nodes_min[i_t, node_idx], bvh_tree_state.nodes_max[i_t, node_idx]
        ):
            continue

        i_left = bvh_tree_state.nodes_left[i_t, node_idx]
        if i_left == -1:
            sorted_leaf_idx = node_idx - (n_triangles - 1)
            i_f = bvh_tree_state.leaves_idx[i_t, sorted_leaf_idx]
            i_g = dyn_info.faces.geom_idx[i_f]
            if not track_geom_mask[i_b, i_s, i_g]:
                continue

            tri = get_triangle_vertices(i_f, i_b, dyn_state, dyn_info)
            v0 = tri[:, 0]
            v1 = tri[:, 1]
            v2 = tri[:, 2]
            closest = closest_point_on_triangle(probe_world, v0, v1, v2)
            diff = probe_world - closest
            d_sq = diff.dot(diff)
            if d_sq < best_dist_sq:
                d = qd.sqrt(d_sq)
                fn = triangle_face_normal(v0, v1, v2)
                # Sign: probe outside if (probe - closest) aligns with outward face normal.
                sign_v = qd.select(diff.dot(fn) >= gs.qd_float(0.0), gs.qd_float(1.0), gs.qd_float(-1.0))
                best_signed = d * sign_v
                best_dist = d
                best_dist_sq = d_sq
        else:
            if stack_idx < STACK_SIZE - 2:
                node_stack[stack_idx] = i_left
                node_stack[stack_idx + 1] = bvh_tree_state.nodes_right[i_t, node_idx]
                stack_idx += 2

    return best_signed


@qd.kernel(fastcache=True)
def _kernel_elastomer_probe_depth_bvh(
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    env_bvh_idx_a: qd.types.ndarray(),
    env_bvh_idx_b: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    track_geom_mask: qd.types.ndarray(),
    probe_depth_buf: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    bvh_tree_state_a: array_class.BVHTreeState,
    bvh_tree_state_b: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    max_query_dist: float,
    is_split: qd.template(),
):
    """
    Compute the contact depth of each probe from the collision bounding volume hierarchy (BVH) entries of the rigid
    solver, folded when split and gated by ``track_geom_mask``.

    It writes ``probe_depth_buf`` like ``_kernel_elastomer_probe_depth``, and the dilate accumulator consumes the same
    buffer downstream.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = probe_depth_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        if probe_radii[i_p] <= gs.qd_float(0.0):
            probe_depth_buf[i_b, i_p] = gs.qd_float(0.0)
            continue
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        link_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        link_quat = dyn_state.links.quat[sensor_link_idx, i_b]
        probe_local = func_vec3_at(i_p, probe_positions_local)
        probe_world = link_pos + gu.qd_transform_by_quat(probe_local, link_quat)

        signed = _func_elastomer_min_signed_dist_bvh(
            env_bvh_idx_a[i_b],
            i_b,
            i_s,
            probe_world,
            track_geom_mask,
            dyn_state,
            bvh_tree_state_a,
            dyn_info,
            max_query_dist,
        )
        if is_split:
            # The collision faces are partitioned over two trees (see RaycastContext.activate); |signed| is the
            # distance the query minimizes, so the smaller magnitude is the globally nearest triangle's answer.
            signed_b = _func_elastomer_min_signed_dist_bvh(
                env_bvh_idx_b[i_b],
                i_b,
                i_s,
                probe_world,
                track_geom_mask,
                dyn_state,
                bvh_tree_state_b,
                dyn_info,
                max_query_dist,
            )
            if qd.abs(signed_b) < qd.abs(signed):
                signed = signed_b
        probe_depth_buf[i_b, i_p] = qd.max(gs.qd_float(0.0), -signed)


@qd.kernel(fastcache=True)
def _kernel_elastomer_probe_depth(
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_track_geom_start: qd.types.ndarray(),
    track_geom_idx: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    sensor_track_geom_n: qd.types.ndarray(),
    track_geom_active_envs_mask: qd.types.ndarray(),
    probe_depth_buf: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Compute the contact depth of each probe from the signed distance fields (SDFs) of the tracked geoms, parallel over
    (env, probe).

    It writes only ``probe_depth_buf``. The dilate accumulation runs in a separate target-major kernel without atomics.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = probe_depth_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        # Inactive filler probe: no SDF query, contributes no dilation.
        if probe_radii[i_p] <= gs.qd_float(0.0):
            probe_depth_buf[i_b, i_p] = gs.qd_float(0.0)
            continue
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        link_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        link_quat = dyn_state.links.quat[sensor_link_idx, i_b]
        probe_local = func_vec3_at(i_p, probe_positions_local)
        probe_world = link_pos + gu.qd_transform_by_quat(probe_local, link_quat)

        min_sdf = _func_elastomer_min_sdf_over_active_geoms(
            i_b,
            sensor_track_geom_start[i_s],
            track_geom_idx,
            probe_world,
            sensor_track_geom_n[i_s],
            track_geom_active_envs_mask,
            dyn_state,
            dyn_info,
            collider_info,
        )

        probe_depth_buf[i_b, i_p] = qd.max(gs.qd_float(0.0), -min_sdf)


@qd.kernel(fastcache=True)
def _kernel_elastomer_dilate_accumulate(
    probe_sensor_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    use_grid_fft: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    lambda_d: qd.types.ndarray(),
    dilate_scale: qd.types.ndarray(),
    normal_exponent: qd.types.ndarray(),
    compressibility: qd.types.ndarray(),
    dilation_reg: qd.types.ndarray(),
    probe_depth_buf: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    """
    Accumulate the dilation onto each target probe of the sensors on the direct path.

    Each (env, target_probe) thread sums the Gaussian contributions of every in-contact source probe of its sensor into
    a register and overwrites its output slot, so no atomic add and no pre-zeroing is needed (see
    ``_update_current_timestep_data`` for the write order). Grid sensors are skipped and take the FFT path.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = probe_depth_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]
        if use_grid_fft[i_s]:
            continue
        n_probes = n_probes_per_sensor[i_s]
        probe_start = sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        lam = lambda_d[i_s]
        scale = dilate_scale[i_s]
        n_exp = normal_exponent[i_s]
        comp = compressibility[i_s]
        eps = dilation_reg[i_s]
        _i_p = i_p - probe_start

        # Inactive filler probe: reads zero, no dilation accumulated.
        if probe_radii[i_p] <= gs.qd_float(0.0):
            for k in qd.static(range(3)):
                output[i_b, cache_start + _i_p * 3 + k] = gs.qd_float(0.0)
            continue

        target_local = func_vec3_at(i_p, probe_positions_local)
        target_normal = func_vec3_at(i_p, probe_local_normal)

        acc = qd.Vector.zero(gs.qd_float, 3)
        for j in range(n_probes):
            j_p = probe_start + j
            src_depth = probe_depth_buf[i_b, j_p]
            if src_depth <= gs.qd_float(0.0):
                continue
            contribution = _func_elastomer_direct_dilate_contribution(
                func_vec3_at(j_p, probe_positions_local),
                target_local,
                target_normal,
                src_depth,
                lam,
                scale,
                n_exp,
                comp,
                eps,
            )
            for k in qd.static(range(3)):
                acc[k] = acc[k] + contribution[k]

        for k in qd.static(range(3)):
            output[i_b, cache_start + _i_p * 3 + k] = acc[k]


@qd.kernel(fastcache=True)
def _kernel_elastomer_surface_state_bvh(
    links_idx: qd.types.ndarray(),
    sensor_elastomer_geom_start: qd.types.ndarray(),
    elastomer_geom_idx: qd.types.ndarray(),
    bvh_chunk_sensor_idx: qd.types.ndarray(),
    sensor_elastomer_geom_n: qd.types.ndarray(),
    elastomer_geom_active_envs_mask: qd.types.ndarray(),
    bvh: ChunkedBVHData,
    pc_pos_link: qd.types.ndarray(),
    pc_active_envs_mask: qd.types.ndarray(),
    sdf_enter: qd.types.ndarray(),
    sdf_exit: qd.types.ndarray(),
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
    surface_candidate_buf: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    aabb_margin: float,
    bvh_stack_size: qd.template(),
):
    """
    Compute the query axis-aligned bounding box (AABB) of each (env, chunk) pair in registers, traverse the bounding
    volume hierarchy (BVH) and write the surface state of each candidate point.

    The AABB fill and the BVH traversal share one kernel, so the AABB stays in thread-local state instead of a (B,
    n_chunks, 3) buffer. The shear contribution is accumulated in a separate target-major kernel that reads
    ``surface_pos_sensor_buf``, ``surface_depth_buf`` and ``surface_entry_pos_sensor_buf``.
    """
    n_batches = surface_pos_sensor_buf.shape[0]
    n_chunks = bvh_chunk_sensor_idx.shape[0]

    for i_b, i_c in qd.ndrange(n_batches, n_chunks):
        i_s = bvh_chunk_sensor_idx[i_c]

        # 1) Build the world-space elastomer-geom union AABB for sensor i_s, env i_b.
        wmin = qd.Vector([gs.qd_float(qd.math.inf), gs.qd_float(qd.math.inf), gs.qd_float(qd.math.inf)], dt=gs.qd_float)
        wmax = qd.Vector(
            [gs.qd_float(-qd.math.inf), gs.qd_float(-qd.math.inf), gs.qd_float(-qd.math.inf)], dt=gs.qd_float
        )
        any_active = False
        gm_start = sensor_elastomer_geom_start[i_s]
        gm_n = sensor_elastomer_geom_n[i_s]
        for i_gm in range(gm_start, gm_start + gm_n):
            if not elastomer_geom_active_envs_mask[i_gm, i_b]:
                continue
            i_g = elastomer_geom_idx[i_gm]
            gmin = dyn_state.geoms.aabb_min[i_g, i_b]
            gmax = dyn_state.geoms.aabb_max[i_g, i_b]
            for k in qd.static(range(3)):
                if gmin[k] < wmin[k]:
                    wmin[k] = gmin[k]
                if gmax[k] > wmax[k]:
                    wmax[k] = gmax[k]
            any_active = True

        if not any_active:
            continue

        # 2) Expand by sdf_exit + margin so any point with sdf <= sdf_exit (the surface-state
        # exit threshold) is inside the AABB.
        expand = sdf_exit[i_s] + gs.qd_float(aabb_margin)
        for k in qd.static(range(3)):
            wmin[k] = wmin[k] - expand
            wmax[k] = wmax[k] + expand

        # 3) Transform 8 corners into the chunk's tracked-link local frame to get qmin/qmax.
        track_link_idx = bvh.chunk_link_idx[i_c]
        track_pos = dyn_state.links.pos[track_link_idx, i_b]
        track_quat = dyn_state.links.quat[track_link_idx, i_b]
        qmin = qd.Vector([gs.qd_float(qd.math.inf), gs.qd_float(qd.math.inf), gs.qd_float(qd.math.inf)], dt=gs.qd_float)
        qmax = qd.Vector(
            [gs.qd_float(-qd.math.inf), gs.qd_float(-qd.math.inf), gs.qd_float(-qd.math.inf)], dt=gs.qd_float
        )
        for cx in qd.static(range(2)):
            for cy in qd.static(range(2)):
                for cz in qd.static(range(2)):
                    cw_x = wmax[0] if cx == 1 else wmin[0]
                    cw_y = wmax[1] if cy == 1 else wmin[1]
                    cw_z = wmax[2] if cz == 1 else wmin[2]
                    corner_world = qd.Vector([cw_x, cw_y, cw_z], dt=gs.qd_float)
                    corner_link = gu.qd_inv_transform_by_trans_quat(corner_world, track_pos, track_quat)
                    for k in qd.static(range(3)):
                        if corner_link[k] < qmin[k]:
                            qmin[k] = corner_link[k]
                        if corner_link[k] > qmax[k]:
                            qmax[k] = corner_link[k]

        # 4) BVH-traverse the chunk with the chunk-local query AABB. For each visited active point:
        # mark candidate, write point_sensor / depth, run anchor (enter/exit hysteresis).
        sensor_link_idx = links_idx[i_s]
        sensor_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        sensor_quat = dyn_state.links.quat[sensor_link_idx, i_b]

        stack = qd.Vector.zero(gs.qd_int, qd.static(bvh_stack_size))
        stack[0] = bvh.chunk_node_start[i_c]
        stack_idx = 1

        while stack_idx > 0:
            stack_idx -= 1
            n = stack[stack_idx]
            bmin = func_vec3_at(n, bvh.node_min)
            bmax = func_vec3_at(n, bvh.node_max)
            if not func_aabb_intersects_aabb(bmin, bmax, qmin, qmax):
                continue
            left = bvh.node_left[n]
            if left == -1:
                pstart = bvh.node_leaf_start[n]
                pn = bvh.node_leaf_count[n]
                for j in range(pn):
                    i_o = bvh.leaf_elem_idx[pstart + j]
                    if not pc_active_envs_mask[i_o, i_b]:
                        continue
                    surface_candidate_buf[i_b, i_o] = True

                    point_link = func_vec3_at(i_o, pc_pos_link)
                    point_world = track_pos + gu.qd_transform_by_quat(point_link, track_quat)
                    point_sensor = gu.qd_inv_transform_by_trans_quat(point_world, sensor_pos, sensor_quat)
                    for k in qd.static(range(3)):
                        surface_pos_sensor_buf[i_b, i_o, k] = point_sensor[k]

                    min_sdf = _func_elastomer_min_sdf_over_active_geoms(
                        i_b,
                        sensor_elastomer_geom_start[i_s],
                        elastomer_geom_idx,
                        point_world,
                        sensor_elastomer_geom_n[i_s],
                        elastomer_geom_active_envs_mask,
                        dyn_state,
                        dyn_info,
                        collider_info,
                    )

                    surface_depth_buf[i_b, i_o] = qd.max(gs.qd_float(0.0), -min_sdf)

                    _func_elastomer_update_surface_anchor(
                        i_b,
                        i_o,
                        min_sdf,
                        point_sensor,
                        sdf_enter[i_s],
                        sdf_exit[i_s],
                        surface_entry_pos_sensor_buf,
                        surface_initialized_buf,
                    )
            else:
                right = bvh.node_right[n]
                # Median split bounds depth at log2(N / leaf_size) << BVH_STACK_SIZE; the guard mirrors the
                # global rigid-BVH kernel so a future build strategy can't silently overflow the stack.
                if stack_idx < qd.static(bvh_stack_size - 2):
                    stack[stack_idx] = left
                    stack[stack_idx + 1] = right
                    stack_idx += 2


@qd.kernel(fastcache=True)
def _kernel_elastomer_surface_state_via_global_bvh(
    links_idx: qd.types.ndarray(),
    env_bvh_idx_a: qd.types.ndarray(),
    env_bvh_idx_b: qd.types.ndarray(),
    sensor_elastomer_geom_start: qd.types.ndarray(),
    elastomer_geom_idx: qd.types.ndarray(),
    bvh_chunk_sensor_idx: qd.types.ndarray(),
    sensor_elastomer_geom_n: qd.types.ndarray(),
    elastomer_geom_active_envs_mask: qd.types.ndarray(),
    elastomer_candidate_geom_mask: qd.types.ndarray(),
    bvh: ChunkedBVHData,
    pc_pos_link: qd.types.ndarray(),
    pc_active_envs_mask: qd.types.ndarray(),
    sdf_enter: qd.types.ndarray(),
    sdf_exit: qd.types.ndarray(),
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    surface_initialized_buf: qd.types.ndarray(),
    surface_candidate_buf: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    global_bvh_tree_a: array_class.BVHTreeState,
    global_bvh_tree_b: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    aabb_margin: float,
    max_query_dist: float,
    bvh_stack_size: qd.template(),
    is_split: qd.template(),
):
    """
    Raycast variant of ``_kernel_elastomer_surface_state_bvh``.

    Same outer (env, chunk) traversal over the point-cloud BVH per tracked link, but the inner signed-distance query
    at each PC point uses ``_func_elastomer_min_signed_dist_bvh`` over the rigid solver's collision BVH entries
    (folded when split, gated by ``elastomer_candidate_geom_mask``) instead of the analytic SDF. It writes the same
    buffers as the SDF variant, so the dilate and shear kernels consume either.
    """
    n_batches = surface_pos_sensor_buf.shape[0]
    n_chunks = bvh_chunk_sensor_idx.shape[0]

    for i_b, i_c in qd.ndrange(n_batches, n_chunks):
        i_s = bvh_chunk_sensor_idx[i_c]

        wmin = qd.Vector([gs.qd_float(qd.math.inf), gs.qd_float(qd.math.inf), gs.qd_float(qd.math.inf)], dt=gs.qd_float)
        wmax = qd.Vector(
            [gs.qd_float(-qd.math.inf), gs.qd_float(-qd.math.inf), gs.qd_float(-qd.math.inf)], dt=gs.qd_float
        )
        any_active = False
        gm_start = sensor_elastomer_geom_start[i_s]
        gm_n = sensor_elastomer_geom_n[i_s]
        for i_gm in range(gm_start, gm_start + gm_n):
            if not elastomer_geom_active_envs_mask[i_gm, i_b]:
                continue
            i_g = elastomer_geom_idx[i_gm]
            gmin = dyn_state.geoms.aabb_min[i_g, i_b]
            gmax = dyn_state.geoms.aabb_max[i_g, i_b]
            for k in qd.static(range(3)):
                if gmin[k] < wmin[k]:
                    wmin[k] = gmin[k]
                if gmax[k] > wmax[k]:
                    wmax[k] = gmax[k]
            any_active = True

        if not any_active:
            continue

        expand = sdf_exit[i_s] + gs.qd_float(aabb_margin)
        for k in qd.static(range(3)):
            wmin[k] = wmin[k] - expand
            wmax[k] = wmax[k] + expand

        track_link_idx = bvh.chunk_link_idx[i_c]
        track_pos = dyn_state.links.pos[track_link_idx, i_b]
        track_quat = dyn_state.links.quat[track_link_idx, i_b]
        qmin = qd.Vector([gs.qd_float(qd.math.inf), gs.qd_float(qd.math.inf), gs.qd_float(qd.math.inf)], dt=gs.qd_float)
        qmax = qd.Vector(
            [gs.qd_float(-qd.math.inf), gs.qd_float(-qd.math.inf), gs.qd_float(-qd.math.inf)], dt=gs.qd_float
        )
        for cx in qd.static(range(2)):
            for cy in qd.static(range(2)):
                for cz in qd.static(range(2)):
                    cw_x = wmax[0] if cx == 1 else wmin[0]
                    cw_y = wmax[1] if cy == 1 else wmin[1]
                    cw_z = wmax[2] if cz == 1 else wmin[2]
                    corner_world = qd.Vector([cw_x, cw_y, cw_z], dt=gs.qd_float)
                    corner_link = gu.qd_inv_transform_by_trans_quat(corner_world, track_pos, track_quat)
                    for k in qd.static(range(3)):
                        if corner_link[k] < qmin[k]:
                            qmin[k] = corner_link[k]
                        if corner_link[k] > qmax[k]:
                            qmax[k] = corner_link[k]

        sensor_link_idx = links_idx[i_s]
        sensor_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        sensor_quat = dyn_state.links.quat[sensor_link_idx, i_b]

        stack = qd.Vector.zero(gs.qd_int, qd.static(bvh_stack_size))
        stack[0] = bvh.chunk_node_start[i_c]
        stack_idx = 1

        while stack_idx > 0:
            stack_idx -= 1
            n = stack[stack_idx]
            bmin = func_vec3_at(n, bvh.node_min)
            bmax = func_vec3_at(n, bvh.node_max)
            if not func_aabb_intersects_aabb(bmin, bmax, qmin, qmax):
                continue
            left = bvh.node_left[n]
            if left == -1:
                pstart = bvh.node_leaf_start[n]
                pn = bvh.node_leaf_count[n]
                for j in range(pn):
                    i_o = bvh.leaf_elem_idx[pstart + j]
                    if not pc_active_envs_mask[i_o, i_b]:
                        continue
                    surface_candidate_buf[i_b, i_o] = True

                    point_link = func_vec3_at(i_o, pc_pos_link)
                    point_world = track_pos + gu.qd_transform_by_quat(point_link, track_quat)
                    point_sensor = gu.qd_inv_transform_by_trans_quat(point_world, sensor_pos, sensor_quat)
                    for k in qd.static(range(3)):
                        surface_pos_sensor_buf[i_b, i_o, k] = point_sensor[k]

                    min_sdf = _func_elastomer_min_signed_dist_bvh(
                        env_bvh_idx_a[i_b],
                        i_b,
                        i_s,
                        point_world,
                        elastomer_candidate_geom_mask,
                        dyn_state,
                        global_bvh_tree_a,
                        dyn_info,
                        max_query_dist,
                    )
                    if is_split:
                        # See _kernel_elastomer_probe_depth_bvh for the two-tree fold.
                        min_sdf_b = _func_elastomer_min_signed_dist_bvh(
                            env_bvh_idx_b[i_b],
                            i_b,
                            i_s,
                            point_world,
                            elastomer_candidate_geom_mask,
                            dyn_state,
                            global_bvh_tree_b,
                            dyn_info,
                            max_query_dist,
                        )
                        if qd.abs(min_sdf_b) < qd.abs(min_sdf):
                            min_sdf = min_sdf_b

                    surface_depth_buf[i_b, i_o] = qd.max(gs.qd_float(0.0), -min_sdf)

                    _func_elastomer_update_surface_anchor(
                        i_b,
                        i_o,
                        min_sdf,
                        point_sensor,
                        sdf_enter[i_s],
                        sdf_exit[i_s],
                        surface_entry_pos_sensor_buf,
                        surface_initialized_buf,
                    )
            else:
                right = bvh.node_right[n]
                # Median split bounds depth at log2(N / leaf_size) << BVH_STACK_SIZE; the guard mirrors the
                # global rigid-BVH kernel so a future build strategy can't silently overflow the stack.
                if stack_idx < qd.static(bvh_stack_size - 2):
                    stack[stack_idx] = left
                    stack[stack_idx + 1] = right
                    stack_idx += 2


@qd.kernel(fastcache=True)
def _kernel_elastomer_shear_accumulate(
    probe_sensor_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    sensor_pc_start: qd.types.ndarray(),
    shear_active_pc_idx: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_local_normal: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    lambda_s: qd.types.ndarray(),
    shear_scale: qd.types.ndarray(),
    surface_pos_sensor_buf: qd.types.ndarray(),
    surface_entry_pos_sensor_buf: qd.types.ndarray(),
    surface_depth_buf: qd.types.ndarray(),
    shear_active_pc_count: qd.types.ndarray(),
    output: qd.types.ndarray(),
    eps: float,
):
    """
    Accumulate the shear contribution of the active surface points of each sensor into ``output``, target-major.

    Per (env, target_probe), the thread iterates over the compact active surface-point index of the sensor, sums the
    Gaussian contributions into a register and adds the result to its own output slot, so no atomic add is needed. It
    consumes the compact index ``_build_shear_active_pc_index`` produces, which runs after the surface-state kernel and
    after the post-kernel ``surface_initialized_buf &= candidate`` cleanup. The inner loop costs O(active_count[i_b,
    i_s]), so the kernel scales with the contact density.
    """
    total_n_probes = probe_positions_local.shape[0]
    n_batches = surface_pos_sensor_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]
        scale = shear_scale[i_s]
        if scale <= gs.qd_float(0.0):
            continue
        # Inactive filler probe: dilate already wrote 0 to this output slot.
        if probe_radii[i_p] <= gs.qd_float(0.0):
            continue
        lam = lambda_s[i_s]
        cache_start = sensor_cache_start[i_s]
        _i_p = i_p - sensor_probe_start[i_s]
        pc_start = sensor_pc_start[i_s]
        n_active = shear_active_pc_count[i_b, i_s]

        probe_local = func_vec3_at(i_p, probe_positions_local)
        probe_normal = func_vec3_at(i_p, probe_local_normal)

        acc = qd.Vector.zero(gs.qd_float, 3)
        for j in range(n_active):
            i_o = shear_active_pc_idx[i_b, pc_start + j]
            depth = surface_depth_buf[i_b, i_o]
            if depth <= eps:
                continue
            point_sensor = qd.Vector(
                [
                    surface_pos_sensor_buf[i_b, i_o, 0],
                    surface_pos_sensor_buf[i_b, i_o, 1],
                    surface_pos_sensor_buf[i_b, i_o, 2],
                ],
                dt=gs.qd_float,
            )
            entry = qd.Vector(
                [
                    surface_entry_pos_sensor_buf[i_b, i_o, 0],
                    surface_entry_pos_sensor_buf[i_b, i_o, 1],
                    surface_entry_pos_sensor_buf[i_b, i_o, 2],
                ],
                dt=gs.qd_float,
            )
            contribution = _func_elastomer_direct_shear_contribution(
                point_sensor, entry, probe_local, probe_normal, depth, lam, scale, eps
            )
            for k in qd.static(range(3)):
                acc[k] = acc[k] + contribution[k]

        for k in qd.static(range(3)):
            output[i_b, cache_start + _i_p * 3 + k] = output[i_b, cache_start + _i_p * 3 + k] + acc[k]


# FIXME: three host transfers then one nonzero per active sensor, every step; compact in one kernel over the
# sensor segments.
def _build_shear_active_pc_index(
    surface_initialized_buf: torch.Tensor,
    sensor_pc_start: torch.Tensor,
    sensor_pc_n: torch.Tensor,
    shear_scale: torch.Tensor,
    active_pc_idx: torch.Tensor,
    active_pc_count: torch.Tensor,
) -> None:
    """
    Build the compact per-(env, sensor) active surface-point index ``_kernel_elastomer_shear_accumulate`` consumes.

    It mutates ``active_pc_idx`` and ``active_pc_count`` in place.

    For each sensor ``s`` with ``shear_scale[s] > 0``, it gathers the indices of the True entries of
    ``surface_initialized_buf[:, pc_start[s] : pc_start[s] + pc_n[s]]`` into the compact slice ``active_pc_idx[:,
    pc_start[s] : pc_start[s] + active_count[:, s]]`` and writes the count of each (env, sensor) pair to
    ``active_pc_count[:, s]``. A sensor with ``shear_scale == 0`` is skipped and its count stays at zero, which the
    outer early exit of the kernel handles with no extra work.

    The per-sensor scatter runs as an exclusive cumsum and one ``torch.nonzero`` over the whole pass, in O(B *
    total_n_surface) torch ops.
    """
    active_pc_count.zero_()
    n_sensors = sensor_pc_start.shape[0]
    if n_sensors == 0:
        return
    # Single host sync up front so the per-sensor loop is metadata-only on the Python side.
    pc_starts = sensor_pc_start.tolist()
    pc_ns = sensor_pc_n.tolist()
    scales = shear_scale.tolist()
    idx_dtype = active_pc_idx.dtype
    for i_s in range(n_sensors):
        if scales[i_s] <= 0.0:
            continue
        pc_start = int(pc_starts[i_s])
        pc_n = int(pc_ns[i_s])
        if pc_n == 0:
            continue
        mask = surface_initialized_buf[:, pc_start : pc_start + pc_n]  # (B, pc_n) bool
        int_mask = mask.to(idx_dtype)
        write_pos = torch.cumsum(int_mask, dim=1) - int_mask  # exclusive cumsum
        active_pc_count[:, i_s] = int_mask.sum(dim=1)
        bs, js = torch.nonzero(mask, as_tuple=True)
        if bs.numel() > 0:
            active_pc_idx[bs, pc_start + write_pos[bs, js]] = (pc_start + js).to(idx_dtype)


@torch.jit.script
def _elastomer_taxel_grid_fft_dilate(
    grid_fft_meta: list[GridFFTMeta],
    grid_fft_kernels_stacked: torch.Tensor,
    probe_depth_buf: torch.Tensor,
    probe_radii: torch.Tensor,
    grid_fft_buffer: torch.Tensor,
    dilate_scale: torch.Tensor,
    normal_exponent: torch.Tensor,
    grid_normal: torch.Tensor,
    grid_tangent_u: torch.Tensor,
    grid_tangent_v: torch.Tensor,
    grid_dilate_out_buffer: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Dilate the elastomer markers of every grid sensor by a 2D fast Fourier transform (FFT) in the tangent basis of its
    probes.

    All grid sensors share one FFT size, the last two dimensions of ``grid_fft_buffer``, and their kernels are stacked
    into ``grid_fft_kernels_stacked`` of shape (n_grid, 3, fft_ny, fft_nx). The four heavy FFTs (the FFT of H, the FFT
    of H**normal_exponent, the inverse FFT for Ku, Kv and Kn) run as batched operations over the grid-sensor axis, in 4
    launches instead of 4*n_grid. The H fill and the write-back stay per sensor, as small Python loops over views and
    copies and a per-sensor tangent decomposition. The grid axes are ``(ny, nx)`` row-major throughout, matching the
    probe flat index ``iy * nx + ix``, so the fill and the write-back read the grid as it is.
    """
    if len(grid_fft_meta) == 0:
        return
    n_batches = probe_depth_buf.shape[0]
    fft_ny, fft_nx = grid_fft_buffer.shape[-2], grid_fft_buffer.shape[-1]

    # 1) Fill the active region of the (B, n_grid, fft_ny, fft_nx) depth buffer. The zero-padding region is never
    # written here and stays zero from allocation, so no per-step ``zero_()`` is needed.
    for grid_pos, meta in enumerate(grid_fft_meta):
        depth_slice = probe_depth_buf[:, meta.probe_start : meta.probe_start + meta.g_ny * meta.g_nx]
        grid_fft_buffer[:, grid_pos, : meta.g_ny, : meta.g_nx].copy_(depth_slice.view(n_batches, meta.g_ny, meta.g_nx))

    # 2) Batched real FFTs across (B, n_grid). Inputs are real so ``rfft2`` (half spectrum) is ~2x cheaper than the
    # full complex ``fft2``. Kernels broadcast over B when multiplying.
    H_fft = torch.fft.rfft2(grid_fft_buffer)
    # The normal channel follows depth ** normal_exponent, so it convolves the per-grid powered depth field;
    # the tangential (u, v) channels stay linear in depth and convolve the raw field H.
    sensors_idx = torch.tensor([meta.sensor_idx for meta in grid_fft_meta], device=normal_exponent.device)
    exps = normal_exponent[sensors_idx].reshape(1, -1, 1, 1)
    Hp_fft = torch.fft.rfft2(grid_fft_buffer.pow(exps))
    Ku_all = grid_fft_kernels_stacked[:, 0]  # (n_grid, fft_ny, fft_nx // 2 + 1) complex
    Kv_all = grid_fft_kernels_stacked[:, 1]
    Kn_all = grid_fft_kernels_stacked[:, 2]
    disp_u_all = torch.fft.irfft2(H_fft * Ku_all, s=(fft_ny, fft_nx))  # (B, n_grid, fft_ny, fft_nx)
    disp_v_all = torch.fft.irfft2(H_fft * Kv_all, s=(fft_ny, fft_nx))
    disp_n_all = torch.fft.irfft2(Hp_fft * Kn_all, s=(fft_ny, fft_nx))

    # 3) Per-sensor write-back: slice to (g_ny, g_nx), apply scale + tangent decomposition, copy
    # into the sensor's output range. Tangent vectors are per-sensor so can't trivially batch here.
    for grid_pos, meta in enumerate(grid_fft_meta):
        sensor_idx, g_ny, g_nx = meta.sensor_idx, meta.g_ny, meta.g_nx
        probe_start, cache_start = meta.probe_start, meta.cache_start
        scale_s = dilate_scale[sensor_idx]
        disp_u = disp_u_all[:, grid_pos, :g_ny, :g_nx] * scale_s
        disp_v = disp_v_all[:, grid_pos, :g_ny, :g_nx] * scale_s
        disp_n = disp_n_all[:, grid_pos, :g_ny, :g_nx] * scale_s
        # (B, g_ny, g_nx) reshapes directly to the probe flat index iy*nx+ix -- no transpose.
        disp_u_flat = disp_u.reshape(n_batches, -1)
        disp_v_flat = disp_v.reshape(n_batches, -1)
        disp_n_flat = disp_n.reshape(n_batches, -1)
        grid_size = g_ny * g_nx * 3
        out_block = grid_dilate_out_buffer[:, :grid_size]
        tangent_u = grid_tangent_u[sensor_idx]
        tangent_v = grid_tangent_v[sensor_idx]
        normal = grid_normal[sensor_idx]
        # Zero inactive filler probes (probe_radius == 0): they are non-sources, but the FFT still smears
        # neighbour dilation into their cells, so mask the per-probe write-back.
        active = (probe_radii[probe_start : probe_start + g_ny * g_nx] > 0.0).to(disp_u_flat.dtype)
        for k in range(3):
            out_block[:, k:grid_size:3] = (
                disp_u_flat * tangent_u[k] + disp_v_flat * tangent_v[k] + disp_n_flat * normal[k]
            ) * active
        output[:, cache_start : cache_start + grid_size].copy_(out_block)


class ElastomerTaxelSensorArray(
    ViscoelasticHysteresisArrayMixin,
    GridFFTConvArrayMixin,
    ContactDepthQueryArrayMixin,
    PointCloudTactileArrayMixin,
    ProbesWithNormalSensorArrayMixin,
    SimpleSensorArray[ElastomerTaxelSensorOptions],
):
    """Array of every elastomer taxel of the scene: the displacement of a soft elastomer surface under contact, dilated
    over the grid of probes and sheared by the anchored contact points."""

    def build(self) -> None:
        super().build()

        _B = self._sim._B
        sensors_options = [sensor.options for sensor in self._sensors]

        for link in self._links:
            if link is None:
                gs.raise_exception("ElastomerTaxel must be attached to a rigid link with collision geometry.")

        # The collision geoms of each sensor's own link and of its tracked links, laid end to end with their spans
        elastomer_geoms = [
            _collect_collision_geom_idx(self._sim.rigid_solver, np.asarray((link.idx,), dtype=gs.np_int))
            for link in self._links
        ]
        self.elastomer_geom_idx = torch.cat([geom_idx for geom_idx, _ in elastomer_geoms])
        self.elastomer_geom_active_envs_mask = torch.cat([active_mask for _, active_mask in elastomer_geoms])
        n_elastomer_geoms = [geom_idx.shape[0] for geom_idx, _ in elastomer_geoms]
        self.sensor_elastomer_geom_start = torch.tensor(
            list(itertools.accumulate(n_elastomer_geoms, initial=0))[:-1], dtype=gs.tc_int, device=gs.device
        )
        self.sensor_elastomer_geom_n = torch.tensor(n_elastomer_geoms, dtype=gs.tc_int, device=gs.device)
        track_geoms = [
            _collect_collision_geom_idx(
                self._sim.rigid_solver, np.asarray(sensor_options.track_link_idx, dtype=gs.np_int)
            )
            for sensor_options in sensors_options
        ]
        self.track_geom_idx = torch.cat([geom_idx for geom_idx, _ in track_geoms])
        self.track_geom_active_envs_mask = torch.cat([active_mask for _, active_mask in track_geoms])
        n_track_geoms = [geom_idx.shape[0] for geom_idx, _ in track_geoms]
        self.sensor_track_geom_start = torch.tensor(
            list(itertools.accumulate(n_track_geoms, initial=0))[:-1], dtype=gs.tc_int, device=gs.device
        )
        self.sensor_track_geom_n = torch.tensor(n_track_geoms, dtype=gs.tc_int, device=gs.device)

        # A 2D layout with non-degenerate spacing takes the FFT dilation path (an irregular one too, with averaged
        # spacing and normal and a warning), the others the direct dilation kernel (see ProbeSensorArrayMixin.build)
        for grid_frame in self._sensors_grid_frame:
            if grid_frame.use_grid_fft and not grid_frame.is_grid_regular:
                gs.logger.warning(
                    "ElastomerTaxel grid is not strictly regular (uniform spacing, uniform normals, orthogonal "
                    "tangents); FFT dilation will use averaged spacing and normal as a best-fit approximation."
                )

        self.lambda_d = torch.tensor(
            [sensor_options.lambda_d for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.lambda_s = torch.tensor(
            [sensor_options.lambda_s for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.dilate_scale = torch.tensor(
            [sensor_options.dilate_scale for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.normal_exponent = torch.tensor(
            [sensor_options.normal_exponent for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.shear_scale = torch.tensor(
            [sensor_options.shear_scale for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        # The in-plane dilation blend weight (1 = local Gaussian, 0 = incompressible 1/r) and the incompressible-kernel
        # regularization epsilon, shared by the direct kernel and the FFT path (baked into GridFFTMeta). The physical
        # scale is elastomer_thickness: grid sensors use it in the exact spectral layer kernel, the direct path
        # approximates the layer by regularizing 1/r at epsilon = h. Without a thickness, epsilon is a numerical guard
        # at the probe spacing (grid step, else sqrt(in-plane area / n_probes)).
        dilation_regs = []
        for sensor_options, grid_frame in zip(sensors_options, self._sensors_grid_frame):
            if sensor_options.elastomer_thickness > 0.0:
                dilation_regs.append(sensor_options.elastomer_thickness)
            elif grid_frame.use_grid_fft:
                dilation_regs.append(0.5 * (grid_frame.grid_spacing[0] + grid_frame.grid_spacing[1]))
            else:
                pos = np.asarray(sensor_options.probe_local_pos, dtype=gs.np_float).reshape(-1, 3)
                ext = np.sort(pos.max(axis=0) - pos.min(axis=0))[::-1]
                area = ext[0] * ext[1] if ext[1] > gs.EPS else ext[0] * ext[0]
                dilation_regs.append(np.sqrt(max(area, gs.EPS) / max(pos.shape[0], 1)))
        self.compressibility = torch.tensor(
            [sensor_options.compressibility for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.dilation_reg = torch.tensor(dilation_regs, dtype=gs.tc_float, device=gs.device)
        # The shear-anchor gate as signed-distance margins: a surface point anchors when its sd < -contact_threshold and
        # releases when sd > -release_threshold, the latter falling back to the former
        self.shear_anchor_sd_enter = torch.tensor(
            [sensor_options.contact_threshold for sensor_options in sensors_options],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.shear_anchor_sd_exit = torch.tensor(
            [
                -(
                    sensor_options.contact_threshold
                    if sensor_options.release_threshold is None
                    else sensor_options.release_threshold
                )
                for sensor_options in sensors_options
            ],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # Python flag gating the per-step shear work without an O(n_sensors) reduction and a device sync
        self.has_any_shear = any(sensor_options.shear_scale > 0.0 for sensor_options in sensors_options)

        total_n_surface = self.pc_pos_link.shape[0]
        self.probe_depth_buf = torch.zeros((_B, self.total_n_probes), dtype=gs.tc_float, device=gs.device)
        self.surface_pos_sensor_buf = torch.zeros((_B, total_n_surface, 3), dtype=gs.tc_float, device=gs.device)
        self.surface_entry_pos_sensor_buf = torch.zeros((_B, total_n_surface, 3), dtype=gs.tc_float, device=gs.device)
        self.surface_depth_buf = torch.zeros((_B, total_n_surface), dtype=gs.tc_float, device=gs.device)
        self.surface_initialized_buf = torch.zeros((_B, total_n_surface), dtype=gs.tc_bool, device=gs.device)
        # Per-(env, pc-row) BVH-candidate flag, zeroed each step and written True by the surface-state kernel for every
        # visited active point, so the post-kernel torch ops can invalidate the stale surface_initialized /
        # surface_entry_pos of the points the BVH skipped this step
        self.surface_candidate_buf = torch.zeros((_B, total_n_surface), dtype=gs.tc_bool, device=gs.device)
        # Compact per-(env, sensor) active surface-point index, rebuilt every step right after the
        # ``surface_initialized_buf &= candidate`` cleanup and consumed by ``_kernel_elastomer_shear_accumulate``: for
        # sensor ``s`` in env ``i_b``, the first ``shear_active_pc_count[i_b, s]`` entries of
        # ``shear_active_pc_idx[i_b, sensor_pc_start[s]:]`` hold the global pc-row indices whose
        # ``surface_initialized_buf`` is True. Zero-initialized whatever the sensors' shear: the per-step index build
        # leaves the count of a sensor without shear at 0, so its unread region stays harmless zeros
        self.shear_active_pc_idx = torch.zeros((_B, total_n_surface), dtype=gs.tc_int, device=gs.device)
        self.shear_active_pc_count = torch.zeros((_B, self.n_sensors), dtype=gs.tc_int, device=gs.device)

        # The candidate-geom masks of the BVH walks of the raycast mode: the tracked geoms (probe depth kernel) and the
        # elastomer geoms of the sensor itself (surface-state kernel, gating the triangles back to the sensor's
        # elastomer surface)
        if self.contact_depth_query == "raycast":
            _fill_candidate_geom_mask(
                self.sensor_candidate_geom_mask,
                self.sensor_track_geom_start,
                self.sensor_track_geom_n,
                self.track_geom_idx,
            )
            self.elastomer_candidate_geom_mask = torch.zeros_like(self.sensor_candidate_geom_mask)
            _fill_candidate_geom_mask(
                self.elastomer_candidate_geom_mask,
                self.sensor_elastomer_geom_start,
                self.sensor_elastomer_geom_n,
                self.elastomer_geom_idx,
            )

        # The FFT dilation of the grid-shaped sensors (the others take the direct dilation kernel), all at one FFT size,
        # and the tangent basis of each grid the dilation write-back decomposes along (see GridFFTMeta for the
        # per-sensor record layout)
        self.use_grid_fft = torch.tensor(
            [grid_frame.use_grid_fft for grid_frame in self._sensors_grid_frame], dtype=gs.tc_bool, device=gs.device
        )
        self.grid_normal = torch.tensor(
            np.stack([grid_frame.grid_normal for grid_frame in self._sensors_grid_frame]),
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.grid_tangent_u = torch.tensor(
            np.stack([grid_frame.tangent_u for grid_frame in self._sensors_grid_frame]),
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.grid_tangent_v = torch.tensor(
            np.stack([grid_frame.tangent_v for grid_frame in self._sensors_grid_frame]),
            dtype=gs.tc_float,
            device=gs.device,
        )
        meta_entries = []
        fft_sizes = []
        grid_sizes = []
        for i_s, (sensor_options, layout_shape, grid_frame) in enumerate(
            zip(sensors_options, self._sensors_probe_layout_shape, self._sensors_grid_frame)
        ):
            if grid_frame.use_grid_fft:
                nx, ny = int(layout_shape[1]), int(layout_shape[0])
                meta_entries.append(
                    GridFFTMeta(
                        sensor_idx=i_s,
                        g_ny=ny,
                        g_nx=nx,
                        probe_start=int(self.sensor_probe_start[i_s].item()),
                        cache_start=int(self.sensors_cache_start[i_s].item()),
                        lambda_d=float(sensor_options.lambda_d),
                        spacing_u=float(grid_frame.grid_spacing[0]),
                        spacing_v=float(grid_frame.grid_spacing[1]),
                        compressibility=sensor_options.compressibility,
                        dilation_reg=dilation_regs[i_s],
                        elastomer_thickness=sensor_options.elastomer_thickness,
                    )
                )
                # FFT size is (ny, nx) row-major. Sizing each axis to ``2n - 1`` (the full linear-convolution support)
                # rounded up to a power of 2 guarantees zero circular wraparound regardless of the dilation kernel's
                # decay -- the ``x*g`` / ``y*g`` first-moment kernels decay slower than the Gaussian itself.
                fft_sizes.append((next_pow2(2 * ny - 1), next_pow2(2 * nx - 1)))
                grid_sizes.append(nx * ny * 3)
        build_grid_fft(self, meta_entries, fft_sizes, _dilate_kernel_builder, n_buffer_channels=0, batch_size=_B)
        # Scratch for the per-sensor tangent-decomposition write-back, sized to the largest grid
        self.grid_dilate_out_buffer = torch.empty((_B, max(grid_sizes, default=0)), dtype=gs.tc_float, device=gs.device)

    def _get_return_format(self, options: ElastomerTaxelSensorOptions) -> tuple[int, ...]:
        return (*np.shape(options.probe_local_pos)[:-1], 3)

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        super().reset(envs_idx)

        # Clearing the anchor state is enough: the other surface buffers are read only where it is True, probe_depth_buf
        # is overwritten every step and surface_candidate_buf is zeroed at step start
        self.surface_initialized_buf[envs_idx] = False

    def _apply_transform(self, data: torch.Tensor, timeline: "TensorRingBuffer", *, is_measured: bool) -> None:
        super()._apply_transform(data, timeline, is_measured=is_measured)
        if not is_measured:
            return
        # ElastomerTaxel's kernel writes a single output used for both GT and measured (measured is .copy_'d from
        # GT), so per-probe gain is applied here as a post-step multiplication on the measured branch only.
        # Approximation note: tangential dilation and shear scale linearly with gain (exact), but the H^2
        # normal-dilation term ideally scales as gain^2 -- here we apply gain^1 across all components. For typical
        # gains near 1 this is a small error; for large deviations the normal component will be slightly off.
        gain_per_col = self.probe_gains[:, self.cache_col_probe_idx]
        data.mul_(gain_per_col)

    def _update_current_timestep_data(self, ground_truth_slot_0: torch.Tensor, measured_slot_0: torch.Tensor) -> None:
        solver = self.solver
        # No pre-zeros: probe_depth is fully overwritten by _kernel_elastomer_probe_depth; the ground-truth slot is
        # fully overwritten by FFT-dilate union dilate-accumulate (then shear-accumulate += on top); surface_depth_buf
        # is only read where surface_initialized=True, which is set in lockstep with that same depth write; the
        # measured slot is copied at the end.
        if (self.contact_depth_query or "sdf") == "sdf":
            _kernel_elastomer_probe_depth(
                self.probe_sensor_idx,
                self.links_idx,
                self.sensor_track_geom_start,
                self.track_geom_idx,
                self.probe_positions,
                self.probe_radii,
                self.sensor_track_geom_n,
                self.track_geom_active_envs_mask,
                self.probe_depth_buf,
                solver.dyn_state,
                solver.dyn_info,
                solver.collider.collider_info,
            )
        else:
            collision_bvh_contexts = self._raycast.collision_bvh_contexts
            entry_a, entry_b = collision_bvh_contexts[0], collision_bvh_contexts[-1]
            _kernel_elastomer_probe_depth_bvh(
                self.probe_sensor_idx,
                self.links_idx,
                entry_a.env_bvh_idx,
                entry_b.env_bvh_idx,
                self.probe_positions,
                self.probe_radii,
                self.sensor_candidate_geom_mask,
                self.probe_depth_buf,
                solver.dyn_state,
                entry_a.bvh_state.tree,
                entry_b.bvh_state.tree,
                solver.dyn_info,
                _ELASTOMER_RAYCAST_QUERY_DIST,
                is_split=entry_b is not entry_a,
            )
        _kernel_elastomer_dilate_accumulate(
            self.probe_sensor_idx,
            self.sensors_cache_start,
            self.sensor_probe_start,
            self.use_grid_fft,
            self.probe_positions,
            self.probe_local_normal,
            self.probe_radii,
            self.n_probes_per_sensor,
            self.lambda_d,
            self.dilate_scale,
            self.normal_exponent,
            self.compressibility,
            self.dilation_reg,
            self.probe_depth_buf,
            ground_truth_slot_0,
        )
        # FFT runs after the qd dilate kernel: on Metal, write-only kernel outputs zero unwritten slots on copy-back,
        # which would erase the grid range the FFT just wrote.
        _elastomer_taxel_grid_fft_dilate(
            self.grid_fft_meta,
            self.grid_fft_kernels_stacked,
            self.probe_depth_buf,
            self.probe_radii,
            self.grid_fft_buffer,
            self.dilate_scale,
            self.normal_exponent,
            self.grid_normal,
            self.grid_tangent_u,
            self.grid_tangent_v,
            self.grid_dilate_out_buffer,
            ground_truth_slot_0,
        )
        if self.has_any_shear:
            bvh = self.pc_bvh
            self.surface_candidate_buf.zero_()
            if (self.contact_depth_query or "sdf") == "sdf":
                _kernel_elastomer_surface_state_bvh(
                    self.links_idx,
                    self.sensor_elastomer_geom_start,
                    self.elastomer_geom_idx,
                    bvh.chunk_sensor_idx,
                    self.sensor_elastomer_geom_n,
                    self.elastomer_geom_active_envs_mask,
                    bvh.kernel_bvh,
                    self.pc_pos_link,
                    self.pc_active_envs_mask,
                    self.shear_anchor_sd_enter,
                    self.shear_anchor_sd_exit,
                    self.surface_pos_sensor_buf,
                    self.surface_entry_pos_sensor_buf,
                    self.surface_depth_buf,
                    self.surface_initialized_buf,
                    self.surface_candidate_buf,
                    solver.dyn_state,
                    solver.dyn_info,
                    solver.collider.collider_info,
                    _ELASTOMER_QUERY_AABB_MARGIN,
                    BVH_STACK_SIZE,
                )
            else:
                collision_bvh_contexts = self._raycast.collision_bvh_contexts
                entry_a, entry_b = collision_bvh_contexts[0], collision_bvh_contexts[-1]
                _kernel_elastomer_surface_state_via_global_bvh(
                    self.links_idx,
                    entry_a.env_bvh_idx,
                    entry_b.env_bvh_idx,
                    self.sensor_elastomer_geom_start,
                    self.elastomer_geom_idx,
                    bvh.chunk_sensor_idx,
                    self.sensor_elastomer_geom_n,
                    self.elastomer_geom_active_envs_mask,
                    self.elastomer_candidate_geom_mask,
                    bvh.kernel_bvh,
                    self.pc_pos_link,
                    self.pc_active_envs_mask,
                    self.shear_anchor_sd_enter,
                    self.shear_anchor_sd_exit,
                    self.surface_pos_sensor_buf,
                    self.surface_entry_pos_sensor_buf,
                    self.surface_depth_buf,
                    self.surface_initialized_buf,
                    self.surface_candidate_buf,
                    solver.dyn_state,
                    entry_a.bvh_state.tree,
                    entry_b.bvh_state.tree,
                    solver.dyn_info,
                    _ELASTOMER_QUERY_AABB_MARGIN,
                    _ELASTOMER_RAYCAST_QUERY_DIST,
                    BVH_STACK_SIZE,
                    is_split=entry_b is not entry_a,
                )
            # Invalidate stale surface state for points the BVH did not visit. surface_initialized
            # and entry-pos survive across steps; depth/pos are gated by initialized downstream so
            # they don't need clearing. The shear accumulator below reads from a compact index
            # rebuilt from surface_initialized -- without this step, stale True from a prior step
            # would inject phantom contributions.
            cand = self.surface_candidate_buf
            self.surface_initialized_buf &= cand
            # Implicit bool->float broadcast zeros entries where cand=False, no `~` allocation.
            self.surface_entry_pos_sensor_buf.mul_(cand.unsqueeze(-1))
            _build_shear_active_pc_index(
                self.surface_initialized_buf,
                self.sensor_pc_start,
                self.sensor_pc_n,
                self.shear_scale,
                self.shear_active_pc_idx,
                self.shear_active_pc_count,
            )
            _kernel_elastomer_shear_accumulate(
                self.probe_sensor_idx,
                self.sensors_cache_start,
                self.sensor_probe_start,
                self.sensor_pc_start,
                self.shear_active_pc_idx,
                self.probe_positions,
                self.probe_local_normal,
                self.probe_radii,
                self.lambda_s,
                self.shear_scale,
                self.surface_pos_sensor_buf,
                self.surface_entry_pos_sensor_buf,
                self.surface_depth_buf,
                self.shear_active_pc_count,
                ground_truth_slot_0,
                gs.EPS,
            )
        measured_slot_0.copy_(ground_truth_slot_0)

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        def mask(envs_idx: EnvsIdxType | None) -> torch.Tensor:
            disp = self.read(i_s, envs_idx, is_ground_truth=True)
            if self.history_lengths[i_s] > 0:
                disp = disp.select(1 if self._sim.n_envs > 0 else 0, -1)
            return torch.linalg.norm(disp, dim=-1) >= gs.EPS

        self._draw_debug_probes(i_s, context, self._tactile_color_groups_fn(i_s, mask))


class ElastomerTaxelSensor(
    ProbeSensorMixin, LinkAttachedSensorMixin, SimpleSensor[ElastomerTaxelSensorOptions, ElastomerTaxelSensorArray]
):
    """Sensor reading the displacement of a soft elastomer surface under contact, per probe in the link frame."""
