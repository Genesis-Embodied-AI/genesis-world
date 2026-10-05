from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.solvers.rigid.collider.utils import func_point_in_geom_aabb
from genesis.options.sensors import ContactDepthProbe as ContactDepthProbeOptions
from genesis.options.sensors import ContactProbe as ContactProbeOptions
from genesis.options.sensors import KinematicTaxel as KinematicTaxelOptions
from genesis.typing import EnvsIdxType
from genesis.utils.raycast_qd import closest_point_on_triangle, get_triangle_vertices, triangle_face_normal

from .base_sensor import LinkAttachedSensorMixin, RigidSensorArrayMixin, SimpleSensor, SimpleSensorArray
from .contact_force import ContactFilterArrayMixin, _func_link_is_filtered
from .probe import ProbeSensorArrayMixin, ProbeSensorMixin, func_noised_probe_radius
from .tactile_shared import (
    ContactDepthQueryArrayMixin,
    ContactPrefilterArrayMixin,
    SpatialCrosstalkArrayMixin,
    ViscoelasticHysteresisArrayMixin,
    func_sphere_intersects_aabb,
)

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext


@qd.func
def _func_query_contact_depth_penetration(
    i_b: int,
    i_s: int,
    sensor_geoms_idx: qd.types.ndarray(),
    probe_pos: qd.types.vector(3),
    probe_radius_gt: float,
    probe_radius_m: float,
    sensor_n_geoms: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Return the largest probe penetration over the distinct opposing geoms of the sensor link from their signed distance
    fields (SDFs), for both probe radii.
    """
    max_pen_gt = gs.qd_float(0.0)
    max_pen_m = gs.qd_float(0.0)

    n_g = sensor_n_geoms[i_b, i_s]
    for i_g_ in range(n_g):
        i_g = sensor_geoms_idx[i_b, i_s, i_g_]
        g_pos = dyn_state.geoms.pos[i_g, i_b]
        g_quat = dyn_state.geoms.quat[i_g, i_b]
        sd = sdf.sdf_func_world_local(i_g, probe_pos, g_pos, g_quat, dyn_info.geoms, collider_info.sdf)
        pen_gt = probe_radius_gt - sd
        if pen_gt > max_pen_gt:
            max_pen_gt = pen_gt
        pen_m = probe_radius_m - sd
        if pen_m > max_pen_m:
            max_pen_m = pen_m

    return max_pen_gt, max_pen_m


@qd.kernel(fastcache=True)
def _kernel_build_sensor_contact_idx(
    sensor_link_idx: qd.types.ndarray(),
    filter_links_idx: qd.types.ndarray(),
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    """
    Build the compact contact index of each (environment, sensor) pair for the KinematicTaxel pre-pass.

    The kernel parallelizes over ``(n_batches, n_sensors)``, and the per-probe contact scan of the main kernel drops
    from O(n_probes * n_contacts) to O(n_probes * sensor_n_contacts). The counterpart filter applies here, before the
    per-sensor cap: a contact tied to the sensor link only through a filtered counterpart is dropped and consumes no cap
    slot, so a large filtered manifold such as the ground cannot starve an allowed contact. An overflow of the cap (a
    count reaching the last dimension of ``sensor_contacts_idx``) truncates the list silently (see the
    ``MAX_CONTACTS_PER_SENSOR`` comment in tactile_shared.py).
    """
    n_sensors = sensor_link_idx.shape[0]
    n_batches = sensor_n_contacts.shape[0]
    max_per_sensor = sensor_contacts_idx.shape[2]
    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        link = sensor_link_idx[i_s]
        count = gs.qd_int(0)
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            if count >= max_per_sensor:
                break
            link_a = collider_state.contact_data.link_a[i_c, i_b]
            link_b = collider_state.contact_data.link_b[i_c, i_b]
            is_on_a = link_a == link and not _func_link_is_filtered(i_s, link_b, filter_links_idx)
            is_on_b = link_b == link and not _func_link_is_filtered(i_s, link_a, filter_links_idx)
            if is_on_a or is_on_b:
                sensor_contacts_idx[i_b, i_s, count] = i_c
                count = count + 1
        sensor_n_contacts[i_b, i_s] = count


@qd.kernel(fastcache=True)
def _kernel_build_sensor_geom_idx(
    sensor_link_idx: qd.types.ndarray(),
    filter_links_idx: qd.types.ndarray(),
    sensor_geoms_idx: qd.types.ndarray(),
    sensor_n_geoms: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    """
    Build the compact, deduplicated list of the opposing contacting geoms of each (environment, sensor) pair for the
    signed distance field (SDF) query path.

    The kernel parallelizes over ``(n_batches, n_sensors)`` and records the opposing geom of each contact, the side away
    from the sensor link. Deduplication collapses the fan-out of a multicontact (tens of contacts on one pressing object
    give one geom), so the per-probe loop of the SDF path runs once per distinct contacting geom. A contact whose
    counterpart link is in the ``filter_links_idx`` row of sensor ``i_s`` is skipped, so the SDF query loop never sees a
    filtered geom (the raycast path filters the same way in ``_kernel_build_sensor_candidate_geom_mask``) and the filter
    stays out of the per-probe loops. An overflow of the cap (a count reaching the last dimension of
    ``sensor_geoms_idx``) truncates the list silently (see the ``MAX_GEOMS_PER_SENSOR`` comment in tactile_shared.py).
    """
    n_sensors = sensor_link_idx.shape[0]
    n_batches = sensor_n_geoms.shape[0]
    max_per_sensor = sensor_geoms_idx.shape[2]
    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        link = sensor_link_idx[i_s]
        count = gs.qd_int(0)
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            link_a = collider_state.contact_data.link_a[i_c, i_b]
            link_b = collider_state.contact_data.link_b[i_c, i_b]
            # A self-contact (sensor link on both sides) is deduped naturally below.
            for side in qd.static(range(2)):
                c_link = link_a if side == 0 else link_b
                counterpart_link = link_b if side == 0 else link_a
                if c_link == link and not _func_link_is_filtered(i_s, counterpart_link, filter_links_idx):
                    i_g = (
                        collider_state.contact_data.geom_b[i_c, i_b]
                        if side == 0
                        else collider_state.contact_data.geom_a[i_c, i_b]
                    )
                    already = False
                    for i_seen in range(count):
                        if sensor_geoms_idx[i_b, i_s, i_seen] == i_g:
                            already = True
                    if not already and count < max_per_sensor:
                        sensor_geoms_idx[i_b, i_s, count] = i_g
                        count = count + 1
        sensor_n_geoms[i_b, i_s] = count


@qd.func
def _func_query_contact_depth(
    i_b: int,
    i_s: int,
    sensor_geoms_idx: qd.types.ndarray(),
    probe_pos: qd.types.vector(3),
    probe_radius_gt: float,
    probe_radius_m: float,
    sensor_n_geoms: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    eps: float,
):
    """
    Query the contact of one probe at two radii in one signed distance field (SDF) and normal pass, for the ground truth
    (GT) radius and the noised radius.

    The func iterates over the deduplicated list of opposing geoms of the (env, sensor) pair that
    ``_kernel_build_sensor_geom_idx`` built. Every geom in that list contacts the tracked link of the sensor, so the
    reported contact link is ``geoms_info.link_idx[i_g]``, the link owning the opposing geom. The axis-aligned bounding
    box (AABB) prefilter expands by ``max(probe_radius_gt, probe_radius_m)`` and covers both radii. A caller without a
    noised radius passes ``probe_radius_m == probe_radius_gt``.
    """
    max_pen_gt = gs.qd_float(0.0)
    contact_link_gt = gs.qd_int(-1)
    contact_normal_gt = qd.Vector.zero(gs.qd_float, 3)
    max_pen_m = gs.qd_float(0.0)
    contact_link_m = gs.qd_int(-1)
    contact_normal_m = qd.Vector.zero(gs.qd_float, 3)

    aabb_expansion = qd.max(probe_radius_gt, probe_radius_m)
    n_g = sensor_n_geoms[i_b, i_s]
    for i_g_ in range(n_g):
        i_g = sensor_geoms_idx[i_b, i_s, i_g_]
        if func_point_in_geom_aabb(i_g, i_b, probe_pos, aabb_expansion, dyn_state):
            g_pos = dyn_state.geoms.pos[i_g, i_b]
            g_quat = dyn_state.geoms.quat[i_g, i_b]
            sd = sdf.sdf_func_world_local(i_g, probe_pos, g_pos, g_quat, dyn_info.geoms, collider_info.sdf)
            pen_gt = probe_radius_gt - sd
            pen_m = probe_radius_m - sd
            # Compute the SDF normal at most once across both branches.
            need_normal = (pen_gt > max_pen_gt and pen_gt > eps) or (pen_m > max_pen_m and pen_m > eps)
            if need_normal:
                normal = sdf.sdf_func_normal_world_local(
                    i_g, probe_pos, g_pos, g_quat, dyn_info.geoms, rigid_info, collider_info.sdf, collider_static_config
                )
                contact_link = dyn_info.geoms.link_idx[i_g]
                if pen_gt > max_pen_gt and pen_gt > eps:
                    max_pen_gt = pen_gt
                    contact_link_gt = contact_link
                    contact_normal_gt = normal
                if pen_m > max_pen_m and pen_m > eps:
                    max_pen_m = pen_m
                    contact_link_m = contact_link
                    contact_normal_m = normal

    return max_pen_gt, contact_link_gt, contact_normal_gt, max_pen_m, contact_link_m, contact_normal_m


@qd.func
def _func_kinematic_spring_damper(
    i_b: int,
    sensor_link_idx: int,
    max_penetration: float,
    contact_link: int,
    contact_normal: qd.types.vector(3),
    probe_pos: qd.types.vector(3),
    probe_pos_local: qd.types.vector(3),
    link_quat: qd.types.vector(4),
    normal_stiffness: float,
    normal_damping: float,
    normal_exponent: float,
    shear_scalar: float,
    twist_scalar: float,
    dyn_state: array_class.DynState,
):
    """
    Return the spring-damper force and torque in the sensor link frame from the contact query of one probe.

    The ground truth (GT) and measured branches of ``_kernel_kinematic_taxel`` share it and differ only in the
    dual-radius query result they feed in. It returns ``(force_local, torque_local)``, both zero when ``max_penetration
    <= 0``.
    """
    force_local = qd.Vector.zero(gs.qd_float, 3)
    torque_local = qd.Vector.zero(gs.qd_float, 3)
    if max_penetration > 0:
        contact_normal_local = gu.qd_inv_transform_by_quat(contact_normal, link_quat)
        s = qd.pow(max_penetration, normal_exponent)
        force_local = contact_normal_local * (normal_stiffness * s)

        if contact_link >= 0:
            contact_vel = dyn_state.links.cd_vel[contact_link, i_b] + dyn_state.links.cd_ang[contact_link, i_b].cross(
                probe_pos - dyn_state.links.root_COM[contact_link, i_b]
            )
            sensor_vel = dyn_state.links.cd_vel[sensor_link_idx, i_b] + dyn_state.links.cd_ang[
                sensor_link_idx, i_b
            ].cross(probe_pos - dyn_state.links.root_COM[sensor_link_idx, i_b])
            rel_vel_world = contact_vel - sensor_vel
            rel_vel_local = gu.qd_inv_transform_by_quat(rel_vel_world, link_quat)

            vn_dot = rel_vel_local.dot(contact_normal_local)
            v_t_local = rel_vel_local - contact_normal_local * vn_dot
            force_local += contact_normal_local * (normal_damping * s * vn_dot) - shear_scalar * v_t_local

            rel_ang_world = dyn_state.links.cd_ang[contact_link, i_b] - dyn_state.links.cd_ang[sensor_link_idx, i_b]
            omega_n = rel_ang_world.dot(contact_normal)
            torque_local = probe_pos_local.cross(force_local) - contact_normal_local * (twist_scalar * omega_n)
        else:
            torque_local = probe_pos_local.cross(force_local)

    return force_local, torque_local


@qd.kernel(fastcache=True)
def _kernel_kinematic_taxel(
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    sensor_geoms_idx: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_gains: qd.types.ndarray(),
    normal_stiffness: qd.types.ndarray(),
    normal_damping: qd.types.ndarray(),
    normal_exponent: qd.types.ndarray(),
    shear_scalar: qd.types.ndarray(),
    twist_scalar: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    sensor_n_geoms: qd.types.ndarray(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    eps: float,
    measured_equals_gt: int,
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[0]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]
        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        force_start = cache_start + probe_idx_in_sensor * 3
        torque_start = cache_start + n_probes * 3 + probe_idx_in_sensor * 3

        # Inactive filler probe (probe_radius == 0): reads zero force/torque, no contact query.
        if probe_radii[i_p] <= gs.qd_float(0.0):
            for j in qd.static(range(3)):
                output_gt[i_b, force_start + j] = gs.qd_float(0.0)
                output_gt[i_b, torque_start + j] = gs.qd_float(0.0)
                output_measured[i_b, force_start + j] = gs.qd_float(0.0)
                output_measured[i_b, torque_start + j] = gs.qd_float(0.0)
            continue

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )

        sensor_link_idx = links_idx[i_s]
        link_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        link_quat = dyn_state.links.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        probe_radius = probe_radii[i_p]
        probe_radius_noise = probe_radii_noise[i_p]
        use_noised_radius = probe_radius_noise > eps
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise, eps) if use_noised_radius else probe_radius
        )

        (
            max_penetration_gt,
            contact_link_gt,
            contact_normal_gt,
            max_penetration_m,
            contact_link_m,
            contact_normal_m,
        ) = _func_query_contact_depth(
            i_b,
            i_s,
            sensor_geoms_idx,
            probe_pos,
            probe_radius,
            probe_radius_m,
            sensor_n_geoms,
            dyn_state,
            dyn_info,
            rigid_info,
            collider_info,
            collider_static_config,
            eps,
        )

        force_local_gt, torque_local_gt = _func_kinematic_spring_damper(
            i_b,
            sensor_link_idx,
            max_penetration_gt,
            contact_link_gt,
            contact_normal_gt,
            probe_pos,
            probe_pos_local,
            link_quat,
            normal_stiffness[i_s],
            normal_damping[i_s],
            normal_exponent[i_s],
            shear_scalar[i_s],
            twist_scalar[i_s],
            dyn_state,
        )

        force_local_m = force_local_gt
        torque_local_m = torque_local_gt
        if measured_equals_gt == 0:
            # The measured branch differs from GT: either some probe has a noised sensing radius or a non-unit
            # per-(env, probe) gain. Gain scales the measured penetration only; force / torque then scale as
            # ``gain ** normal_exponent`` since they derive from ``s = max_penetration_m ** normal_exponent``.
            max_penetration_m = max_penetration_m * probe_gains[i_b, i_p]
            force_local_m, torque_local_m = _func_kinematic_spring_damper(
                i_b,
                sensor_link_idx,
                max_penetration_m,
                contact_link_m,
                contact_normal_m,
                probe_pos,
                probe_pos_local,
                link_quat,
                normal_stiffness[i_s],
                normal_damping[i_s],
                normal_exponent[i_s],
                shear_scalar[i_s],
                twist_scalar[i_s],
                dyn_state,
            )

        for j in qd.static(range(3)):
            output_gt[i_b, force_start + j] = force_local_gt[j]
            output_gt[i_b, torque_start + j] = torque_local_gt[j]
            output_measured[i_b, force_start + j] = force_local_m[j]
            output_measured[i_b, torque_start + j] = torque_local_m[j]


@qd.kernel(fastcache=True)
def _kernel_contact_depth_probe(
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    sensor_geoms_idx: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_gains: qd.types.ndarray(),
    sensor_n_geoms: qd.types.ndarray(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    eps: float,
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[0]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]

        # Inactive filler probe (probe_radius == 0): reads zero depth (which contact-probe interprets as no contact).
        if probe_radii[i_p] <= gs.qd_float(0.0):
            cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
            output_gt[i_b, cache_idx] = gs.qd_float(0.0)
            output_measured[i_b, cache_idx] = gs.qd_float(0.0)
            continue

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )

        sensor_link_idx = links_idx[i_s]
        link_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        link_quat = dyn_state.links.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        probe_radius = probe_radii[i_p]
        probe_radius_noise = probe_radii_noise[i_p]
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise, eps)
            if probe_radius_noise > eps
            else probe_radius
        )

        max_penetration_gt, max_penetration_m = _func_query_contact_depth_penetration(
            i_b,
            i_s,
            sensor_geoms_idx,
            probe_pos,
            probe_radius,
            probe_radius_m,
            sensor_n_geoms,
            dyn_state,
            dyn_info,
            collider_info,
        )
        max_penetration_m = max_penetration_m * probe_gains[i_b, i_p]  # gain on measured branch only
        cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
        output_gt[i_b, cache_idx] = max_penetration_gt
        output_measured[i_b, cache_idx] = max_penetration_m


# ============================ Raycast / BVH contact-depth path ============================


@qd.kernel(fastcache=True)
def _kernel_build_sensor_candidate_geom_mask(
    sensor_link_idx: qd.types.ndarray(),
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
    sensor_candidate_geom_mask: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    """
    Scatter the per-(env, sensor) candidate-geom bitmask from the prefiltered contact list.

    It runs only when the array is in ``contact_depth_query="raycast"`` mode. The leaf loop of the bounding volume
    hierarchy (BVH) consults the mask to skip the triangles whose owning geom is outside the current contact list of the
    sensor. Only the geom on the side opposite the sensor link is marked, as in the ``i_g = <other geom>`` selection of
    the SDF path: marking the own geom of the sensor would let the closest-point test of the BVH latch onto its own
    surface and pin the reported depth to ``probe_radius``, whatever the pressing object. The counterpart filter is
    already applied while building ``sensor_contacts_idx``, so every listed contact is an allowed one.
    """
    n_batches = sensor_n_contacts.shape[0]
    n_sensors = sensor_n_contacts.shape[1]
    n_geoms = sensor_candidate_geom_mask.shape[2]
    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        for i_g in range(n_geoms):
            sensor_candidate_geom_mask[i_b, i_s, i_g] = False
        link = sensor_link_idx[i_s]
        n_c = sensor_n_contacts[i_b, i_s]
        for i_c_ in range(n_c):
            i_c = sensor_contacts_idx[i_b, i_s, i_c_]
            if collider_state.contact_data.link_a[i_c, i_b] == link:
                sensor_candidate_geom_mask[i_b, i_s, collider_state.contact_data.geom_b[i_c, i_b]] = True
            if collider_state.contact_data.link_b[i_c, i_b] == link:
                sensor_candidate_geom_mask[i_b, i_s, collider_state.contact_data.geom_a[i_c, i_b]] = True


@qd.func
def _func_query_contact_depth_penetration_bvh(
    i_t: int,
    i_b: int,
    i_s: int,
    probe_pos: qd.types.vector(3),
    probe_radius_gt: float,
    probe_radius_m: float,
    sensor_candidate_geom_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    bvh_tree_state: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
):
    """
    Return the probe penetration at two radii from the nearest candidate triangle of a bounding volume hierarchy (BVH).

    The signed distance to the nearest candidate triangle takes its sign from the face normal of that triangle, negative
    when the probe is inside the surface, like ``_func_elastomer_min_signed_dist_bvh``. The penetration is ``max(0, R -
    sd)`` per radius, the ``pen = R - sd`` of the SDF path: it keeps growing as the probe penetrates, where an unsigned
    closest-point distance would fold back at ``R``. The return mirrors ``_func_query_contact_depth_penetration`` with
    the signed distance of the nearest triangle appended, so a split-tree caller selects the globally nearest answer
    (see the fold in the kernels).
    """
    # The tree's own leaf count: a compacted-subset tree (see RaycastContext.activate) has fewer leaves than faces.
    n_triangles = bvh_tree_state.leaves_idx.shape[1]
    radius_query = qd.max(probe_radius_gt, probe_radius_m)
    best_dist_sq = radius_query * radius_query
    best_signed = radius_query

    # Stack depth, see func_bvh_query_aabb in genesis.engine.bvh
    STACK_SIZE = qd.static(64)
    node_stack = qd.Vector.zero(gs.qd_int, STACK_SIZE)
    node_stack[0] = 0
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]

        if not func_sphere_intersects_aabb(
            probe_pos, best_dist_sq, bvh_tree_state.nodes_min[i_t, node_idx], bvh_tree_state.nodes_max[i_t, node_idx]
        ):
            continue

        i_left = bvh_tree_state.nodes_left[i_t, node_idx]
        if i_left == -1:
            sorted_leaf_idx = node_idx - (n_triangles - 1)
            i_f = bvh_tree_state.leaves_idx[i_t, sorted_leaf_idx]
            i_g = dyn_info.faces.geom_idx[i_f]
            if not sensor_candidate_geom_mask[i_b, i_s, i_g]:
                continue

            tri = get_triangle_vertices(i_f, i_b, dyn_state, dyn_info)
            v0 = tri[:, 0]
            v1 = tri[:, 1]
            v2 = tri[:, 2]

            closest = closest_point_on_triangle(probe_pos, v0, v1, v2)
            diff = probe_pos - closest
            d_sq = diff.dot(diff)
            if d_sq < best_dist_sq:
                d = qd.sqrt(d_sq)
                fn = triangle_face_normal(v0, v1, v2)
                sign_v = qd.select(diff.dot(fn) >= gs.qd_float(0.0), gs.qd_float(1.0), gs.qd_float(-1.0))
                best_signed = d * sign_v
                best_dist_sq = d_sq
        else:
            if stack_idx < STACK_SIZE - 2:
                node_stack[stack_idx] = i_left
                node_stack[stack_idx + 1] = bvh_tree_state.nodes_right[i_t, node_idx]
                stack_idx += 2

    max_pen_gt = qd.max(gs.qd_float(0.0), probe_radius_gt - best_signed)
    max_pen_m = qd.max(gs.qd_float(0.0), probe_radius_m - best_signed)
    return max_pen_gt, max_pen_m, best_signed


@qd.func
def _func_query_contact_depth_bvh(
    i_t: int,
    i_b: int,
    i_s: int,
    probe_pos: qd.types.vector(3),
    probe_radius_gt: float,
    probe_radius_m: float,
    sensor_candidate_geom_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    bvh_tree_state: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
):
    """
    Query the contact of one probe at two radii against a bounding volume hierarchy (BVH), with the contact normal and
    link, in the return layout of ``_func_query_contact_depth``.

    The nearest candidate triangle gives the signed distance, negative when the probe is inside the surface, and ``pen =
    R - sd`` matches the SDF path. The returned contact normal is the outward face normal of the nearest triangle, which
    the spring-damper model uses as the surface normal. The signed distance is appended, so a split-tree caller selects
    the globally nearest answer (see the fold in the kernels).
    """
    # The tree's own leaf count: a compacted-subset tree (see RaycastContext.activate) has fewer leaves than faces.
    n_triangles = bvh_tree_state.leaves_idx.shape[1]
    radius_query = qd.max(probe_radius_gt, probe_radius_m)
    best_dist_sq = radius_query * radius_query
    best_signed = radius_query
    contact_link = gs.qd_int(-1)
    contact_normal = qd.Vector.zero(gs.qd_float, 3)

    # Stack depth, see func_bvh_query_aabb in genesis.engine.bvh
    STACK_SIZE = qd.static(64)
    node_stack = qd.Vector.zero(gs.qd_int, STACK_SIZE)
    node_stack[0] = 0
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]

        if not func_sphere_intersects_aabb(
            probe_pos, best_dist_sq, bvh_tree_state.nodes_min[i_t, node_idx], bvh_tree_state.nodes_max[i_t, node_idx]
        ):
            continue

        i_left = bvh_tree_state.nodes_left[i_t, node_idx]
        if i_left == -1:
            sorted_leaf_idx = node_idx - (n_triangles - 1)
            i_f = bvh_tree_state.leaves_idx[i_t, sorted_leaf_idx]
            i_g = dyn_info.faces.geom_idx[i_f]
            if not sensor_candidate_geom_mask[i_b, i_s, i_g]:
                continue

            tri = get_triangle_vertices(i_f, i_b, dyn_state, dyn_info)
            v0 = tri[:, 0]
            v1 = tri[:, 1]
            v2 = tri[:, 2]

            closest = closest_point_on_triangle(probe_pos, v0, v1, v2)
            diff = probe_pos - closest
            d_sq = diff.dot(diff)
            if d_sq < best_dist_sq:
                d = qd.sqrt(d_sq)
                fn = triangle_face_normal(v0, v1, v2)
                sign_v = qd.select(diff.dot(fn) >= gs.qd_float(0.0), gs.qd_float(1.0), gs.qd_float(-1.0))
                best_signed = d * sign_v
                best_dist_sq = d_sq
                contact_link = dyn_info.geoms.link_idx[i_g]
                contact_normal = fn
        else:
            if stack_idx < STACK_SIZE - 2:
                node_stack[stack_idx] = i_left
                node_stack[stack_idx + 1] = bvh_tree_state.nodes_right[i_t, node_idx]
                stack_idx += 2

    # Penetration only; the link / normal are meaningful only for the branch that actually reports contact.
    max_pen_gt = qd.max(gs.qd_float(0.0), probe_radius_gt - best_signed)
    max_pen_m = qd.max(gs.qd_float(0.0), probe_radius_m - best_signed)
    contact_link_gt = contact_link if max_pen_gt > gs.qd_float(0.0) else gs.qd_int(-1)
    contact_link_m = contact_link if max_pen_m > gs.qd_float(0.0) else gs.qd_int(-1)
    contact_normal_gt = contact_normal if max_pen_gt > gs.qd_float(0.0) else qd.Vector.zero(gs.qd_float, 3)
    contact_normal_m = contact_normal if max_pen_m > gs.qd_float(0.0) else qd.Vector.zero(gs.qd_float, 3)
    return max_pen_gt, contact_link_gt, contact_normal_gt, max_pen_m, contact_link_m, contact_normal_m, best_signed


@qd.kernel(fastcache=True)
def _kernel_contact_depth_probe_bvh(
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    env_bvh_idx_a: qd.types.ndarray(),
    env_bvh_idx_b: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_gains: qd.types.ndarray(),
    sensor_candidate_geom_mask: qd.types.ndarray(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    bvh_tree_state_a: array_class.BVHTreeState,
    bvh_tree_state_b: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    eps: float,
    is_split: qd.template(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[0]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]

        if probe_radii[i_p] <= gs.qd_float(0.0):
            cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
            output_gt[i_b, cache_idx] = gs.qd_float(0.0)
            output_measured[i_b, cache_idx] = gs.qd_float(0.0)
            continue

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )

        sensor_link_idx = links_idx[i_s]
        link_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        link_quat = dyn_state.links.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        probe_radius = probe_radii[i_p]
        probe_radius_noise = probe_radii_noise[i_p]
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise, eps)
            if probe_radius_noise > eps
            else probe_radius
        )

        max_penetration_gt, max_penetration_m, signed_dist = _func_query_contact_depth_penetration_bvh(
            env_bvh_idx_a[i_b],
            i_b,
            i_s,
            probe_pos,
            probe_radius,
            probe_radius_m,
            sensor_candidate_geom_mask,
            dyn_state,
            bvh_tree_state_a,
            dyn_info,
        )
        if is_split:
            # The collision faces are partitioned over two trees (see RaycastContext.activate). Each query answers
            # for its own nearest triangle, and penetration alone cannot decide between them - a farther inside
            # triangle out-penetrates a nearer outside one - so the globally nearest answer is the one with the
            # smaller signed-distance magnitude, taken wholesale.
            max_penetration_gt_b, max_penetration_m_b, signed_dist_b = _func_query_contact_depth_penetration_bvh(
                env_bvh_idx_b[i_b],
                i_b,
                i_s,
                probe_pos,
                probe_radius,
                probe_radius_m,
                sensor_candidate_geom_mask,
                dyn_state,
                bvh_tree_state_b,
                dyn_info,
            )
            if qd.abs(signed_dist_b) < qd.abs(signed_dist):
                max_penetration_gt = max_penetration_gt_b
                max_penetration_m = max_penetration_m_b
        max_penetration_m = max_penetration_m * probe_gains[i_b, i_p]
        cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
        output_gt[i_b, cache_idx] = max_penetration_gt
        output_measured[i_b, cache_idx] = max_penetration_m


@qd.kernel(fastcache=True)
def _kernel_kinematic_taxel_bvh(
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    env_bvh_idx_a: qd.types.ndarray(),
    env_bvh_idx_b: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    probe_gains: qd.types.ndarray(),
    normal_stiffness: qd.types.ndarray(),
    normal_damping: qd.types.ndarray(),
    normal_exponent: qd.types.ndarray(),
    shear_scalar: qd.types.ndarray(),
    twist_scalar: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    sensor_candidate_geom_mask: qd.types.ndarray(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    bvh_tree_state_a: array_class.BVHTreeState,
    bvh_tree_state_b: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    measured_equals_gt: int,
    eps: float,
    is_split: qd.template(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[0]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]
        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        force_start = cache_start + probe_idx_in_sensor * 3
        torque_start = cache_start + n_probes * 3 + probe_idx_in_sensor * 3

        if probe_radii[i_p] <= gs.qd_float(0.0):
            for j in qd.static(range(3)):
                output_gt[i_b, force_start + j] = gs.qd_float(0.0)
                output_gt[i_b, torque_start + j] = gs.qd_float(0.0)
                output_measured[i_b, force_start + j] = gs.qd_float(0.0)
                output_measured[i_b, torque_start + j] = gs.qd_float(0.0)
            continue

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )

        sensor_link_idx = links_idx[i_s]
        link_pos = dyn_state.links.pos[sensor_link_idx, i_b]
        link_quat = dyn_state.links.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        probe_radius = probe_radii[i_p]
        probe_radius_noise = probe_radii_noise[i_p]
        use_noised_radius = probe_radius_noise > eps
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise, eps) if use_noised_radius else probe_radius
        )

        (
            max_penetration_gt,
            contact_link_gt,
            contact_normal_gt,
            max_penetration_m,
            contact_link_m,
            contact_normal_m,
            signed_dist,
        ) = _func_query_contact_depth_bvh(
            env_bvh_idx_a[i_b],
            i_b,
            i_s,
            probe_pos,
            probe_radius,
            probe_radius_m,
            sensor_candidate_geom_mask,
            dyn_state,
            bvh_tree_state_a,
            dyn_info,
        )
        if is_split:
            # See _kernel_contact_depth_probe_bvh for the two-tree fold: the globally nearest answer is the one
            # with the smaller signed-distance magnitude, taken wholesale so link and normal stay consistent.
            (
                max_penetration_gt_b,
                contact_link_gt_b,
                contact_normal_gt_b,
                max_penetration_m_b,
                contact_link_m_b,
                contact_normal_m_b,
                signed_dist_b,
            ) = _func_query_contact_depth_bvh(
                env_bvh_idx_b[i_b],
                i_b,
                i_s,
                probe_pos,
                probe_radius,
                probe_radius_m,
                sensor_candidate_geom_mask,
                dyn_state,
                bvh_tree_state_b,
                dyn_info,
            )
            if qd.abs(signed_dist_b) < qd.abs(signed_dist):
                max_penetration_gt = max_penetration_gt_b
                contact_link_gt = contact_link_gt_b
                contact_normal_gt = contact_normal_gt_b
                max_penetration_m = max_penetration_m_b
                contact_link_m = contact_link_m_b
                contact_normal_m = contact_normal_m_b

        gained_pen_m = max_penetration_m * probe_gains[i_b, i_p]

        force_gt, torque_gt = _func_kinematic_spring_damper(
            i_b,
            sensor_link_idx,
            max_penetration_gt,
            contact_link_gt,
            contact_normal_gt,
            probe_pos,
            probe_pos_local,
            link_quat,
            normal_stiffness[i_s],
            normal_damping[i_s],
            normal_exponent[i_s],
            shear_scalar[i_s],
            twist_scalar[i_s],
            dyn_state,
        )
        for j in qd.static(range(3)):
            output_gt[i_b, force_start + j] = force_gt[j]
            output_gt[i_b, torque_start + j] = torque_gt[j]

        if measured_equals_gt == 1:
            for j in qd.static(range(3)):
                output_measured[i_b, force_start + j] = force_gt[j]
                output_measured[i_b, torque_start + j] = torque_gt[j]
        else:
            force_m, torque_m = _func_kinematic_spring_damper(
                i_b,
                sensor_link_idx,
                gained_pen_m,
                contact_link_m,
                contact_normal_m,
                probe_pos,
                probe_pos_local,
                link_quat,
                normal_stiffness[i_s],
                normal_damping[i_s],
                normal_exponent[i_s],
                shear_scalar[i_s],
                twist_scalar[i_s],
                dyn_state,
            )
            for j in qd.static(range(3)):
                output_measured[i_b, force_start + j] = force_m[j]
                output_measured[i_b, torque_start + j] = torque_m[j]


class ContactDepthProbeSensorArray(
    ViscoelasticHysteresisArrayMixin,
    ProbeSensorArrayMixin,
    ContactFilterArrayMixin,
    ContactPrefilterArrayMixin,
    ContactDepthQueryArrayMixin,
    RigidSensorArrayMixin,
    SimpleSensorArray[ContactDepthProbeOptions],
):
    """Array of every contact depth probe of the scene, reading the contact depth of each probe in meters."""

    def _get_return_format(self, options: ContactDepthProbeOptions) -> tuple[int, ...]:
        return np.shape(options.probe_local_pos)[:-1]

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def _update_current_timestep_data(self, ground_truth_slot_0: torch.Tensor, measured_slot_0: torch.Tensor) -> None:
        solver = self.solver
        ground_truth_slot_0.zero_()
        measured_slot_0.zero_()
        if (self.contact_depth_query or "sdf") == "sdf":
            _kernel_build_sensor_geom_idx(
                self.links_idx,
                self.filter_links_idx,
                self.sensor_geoms_idx,
                self.sensor_n_geoms,
                solver.collider.collider_state,
            )
            _kernel_contact_depth_probe(
                self.probe_sensor_idx,
                self.links_idx,
                self.sensors_cache_start,
                self.sensor_probe_start,
                self.sensor_geoms_idx,
                self.probe_positions,
                self.probe_radii,
                self.probe_radii_noise,
                self.probe_gains,
                self.sensor_n_geoms,
                ground_truth_slot_0,
                measured_slot_0,
                solver.dyn_state,
                solver.dyn_info,
                solver.collider.collider_info,
                eps=gs.EPS,
            )
        else:
            _kernel_build_sensor_contact_idx(
                self.links_idx,
                self.filter_links_idx,
                self.sensor_contacts_idx,
                self.sensor_n_contacts,
                solver.collider.collider_state,
            )
            _kernel_build_sensor_candidate_geom_mask(
                self.links_idx,
                self.sensor_contacts_idx,
                self.sensor_n_contacts,
                self.sensor_candidate_geom_mask,
                solver.collider.collider_state,
            )
            collision_bvh_contexts = self._raycast.collision_bvh_contexts
            entry_a, entry_b = collision_bvh_contexts[0], collision_bvh_contexts[-1]
            _kernel_contact_depth_probe_bvh(
                self.probe_sensor_idx,
                self.links_idx,
                entry_a.env_bvh_idx,
                entry_b.env_bvh_idx,
                self.sensors_cache_start,
                self.sensor_probe_start,
                self.probe_positions,
                self.probe_radii,
                self.probe_radii_noise,
                self.probe_gains,
                self.sensor_candidate_geom_mask,
                ground_truth_slot_0,
                measured_slot_0,
                solver.dyn_state,
                entry_a.bvh_state.tree,
                entry_b.bvh_state.tree,
                solver.dyn_info,
                eps=gs.EPS,
                is_split=entry_b is not entry_a,
            )

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        def mask(envs_idx: EnvsIdxType | None) -> torch.Tensor:
            depth = self.read(i_s, envs_idx, is_ground_truth=True)
            if self.history_lengths[i_s] > 0:
                depth = depth.select(1 if self._sim.n_envs > 0 else 0, -1)
            return depth >= gs.EPS

        self._draw_debug_probes(i_s, context, self._tactile_color_groups_fn(i_s, mask))


class ContactDepthProbeSensor(
    ProbeSensorMixin, LinkAttachedSensorMixin, SimpleSensor[ContactDepthProbeOptions, ContactDepthProbeSensorArray]
):
    """Sensor reading the contact depth of each of its probes, in meters."""


class ContactProbeSensorArray(ContactDepthProbeSensorArray):
    """
    Array of every contact probe of the scene, reading a boolean contact per probe with an optional Schmitt-trigger
    hysteresis.

    It shares the kernel of the depth probes. The contact bit latches on when the depth exceeds ``contact_threshold``
    and releases when the depth drops to ``release_threshold`` or below. With ``release_threshold`` unset, its default,
    it equals ``contact_threshold`` and the latch reduces to a stateless threshold. The latch state comes from the
    return-space ring of each branch, so the ground truth (GT) and measured branches latch independently and reset with
    the environment (the ring is zeroed on reset).
    """

    def build(self) -> None:
        super().build()

        sensors_options = [sensor.options for sensor in self._sensors]
        self.contact_threshold = torch.tensor(
            [sensor_options.contact_threshold for sensor_options in sensors_options],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # The latch releases at the contact threshold when no release threshold is set, a stateless threshold
        self.release_threshold = torch.tensor(
            [
                sensor_options.contact_threshold
                if sensor_options.release_threshold is None
                else sensor_options.release_threshold
                for sensor_options in sensors_options
            ],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # The gate levels per cache column, one column per probe in probe order
        cache_sizes = torch.tensor(self.cache_sizes, device=gs.device)
        self.enter_row = torch.repeat_interleave(self.contact_threshold, cache_sizes)
        self.exit_row = torch.repeat_interleave(self.release_threshold, cache_sizes)

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_bool

    def _get_intermediate_dtype(self) -> torch.dtype:
        return gs.tc_float

    def _post_process(self, tensor: torch.Tensor, timeline: "TensorRingBuffer", *, is_measured: bool) -> torch.Tensor:
        above_enter = tensor > self.enter_row.unsqueeze(0)
        above_exit = tensor > self.exit_row.unsqueeze(0)
        prev_state = timeline.at(0, copy=False)
        return above_enter | (prev_state & above_exit)

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        def mask(envs_idx: EnvsIdxType | None) -> torch.Tensor:
            contact = self.read(i_s, envs_idx, is_ground_truth=True)
            if self.history_lengths[i_s] > 0:
                contact = contact.select(1 if self._sim.n_envs > 0 else 0, -1)
            return contact

        self._draw_debug_probes(i_s, context, self._tactile_color_groups_fn(i_s, mask))


class ContactProbeSensor(
    ProbeSensorMixin, LinkAttachedSensorMixin, SimpleSensor[ContactProbeOptions, ContactProbeSensorArray]
):
    """
    Sensor reading a boolean contact per probe, with an optional Schmitt-trigger hysteresis (see
    ContactProbeSensorArray).
    """


class KinematicTaxelReturnType(NamedTuple):
    """Estimated contact force and torque per probe in the link frame, from the kinematic spring-damper model."""

    force: torch.Tensor
    torque: torch.Tensor


class KinematicTaxelSensorArray(
    ViscoelasticHysteresisArrayMixin,
    SpatialCrosstalkArrayMixin,
    ProbeSensorArrayMixin,
    ContactFilterArrayMixin,
    ContactPrefilterArrayMixin,
    ContactDepthQueryArrayMixin,
    RigidSensorArrayMixin,
    SimpleSensorArray[KinematicTaxelOptions, KinematicTaxelReturnType],
):
    """Array of every kinematic taxel of the scene: spring-damper force and torque per probe from contact geometry
    and relative motion."""

    # Two channel groups: force xyz followed by torque xyz (probe-major within each group). See
    # ``ProbeSensorArrayMixin._taxel_channel_groups`` for how this drives dead-taxel cache-col -> probe mapping.
    _taxel_channel_groups = 2

    def build(self) -> None:
        super().build()

        sensors_options = [sensor.options for sensor in self._sensors]
        self.normal_stiffness = torch.tensor(
            [sensor_options.normal_stiffness for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.normal_damping = torch.tensor(
            [sensor_options.normal_damping for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.normal_exponent = torch.tensor(
            [sensor_options.normal_exponent for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.shear_scalar = torch.tensor(
            [sensor_options.shear_scalar for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )
        self.twist_scalar = torch.tensor(
            [sensor_options.twist_scalar for sensor_options in sensors_options], dtype=gs.tc_float, device=gs.device
        )

    def _get_return_format(self, options: KinematicTaxelOptions) -> tuple[tuple[int, ...], ...]:
        shape = (*np.shape(options.probe_local_pos)[:-1], 3)
        return shape, shape

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def _update_current_timestep_data(self, ground_truth_slot_0: torch.Tensor, measured_slot_0: torch.Tensor) -> None:
        solver = self.solver
        ground_truth_slot_0.zero_()
        measured_slot_0.zero_()
        # The measured branch is provably identical to GT (and the kernel can skip recomputing it) when no probe
        # has a noised sensing radius and no probe has a non-unit measured-branch gain.
        measured_equals_gt = int(not self.has_any_probe_radius_noise and not self.has_any_probe_gain)
        if (self.contact_depth_query or "sdf") == "sdf":
            _kernel_build_sensor_geom_idx(
                self.links_idx,
                self.filter_links_idx,
                self.sensor_geoms_idx,
                self.sensor_n_geoms,
                solver.collider.collider_state,
            )
            _kernel_kinematic_taxel(
                self.probe_sensor_idx,
                self.links_idx,
                self.sensors_cache_start,
                self.sensor_probe_start,
                self.sensor_geoms_idx,
                self.probe_positions,
                self.probe_radii,
                self.probe_radii_noise,
                self.probe_gains,
                self.normal_stiffness,
                self.normal_damping,
                self.normal_exponent,
                self.shear_scalar,
                self.twist_scalar,
                self.n_probes_per_sensor,
                self.sensor_n_geoms,
                ground_truth_slot_0,
                measured_slot_0,
                solver.dyn_state,
                solver.dyn_info,
                solver.rigid_info,
                solver.collider.collider_info,
                solver.collider.collider_config,
                gs.EPS,
                measured_equals_gt,
            )
        else:
            _kernel_build_sensor_contact_idx(
                self.links_idx,
                self.filter_links_idx,
                self.sensor_contacts_idx,
                self.sensor_n_contacts,
                solver.collider.collider_state,
            )
            _kernel_build_sensor_candidate_geom_mask(
                self.links_idx,
                self.sensor_contacts_idx,
                self.sensor_n_contacts,
                self.sensor_candidate_geom_mask,
                solver.collider.collider_state,
            )
            collision_bvh_contexts = self._raycast.collision_bvh_contexts
            entry_a, entry_b = collision_bvh_contexts[0], collision_bvh_contexts[-1]
            _kernel_kinematic_taxel_bvh(
                self.probe_sensor_idx,
                self.links_idx,
                entry_a.env_bvh_idx,
                entry_b.env_bvh_idx,
                self.sensors_cache_start,
                self.sensor_probe_start,
                self.probe_positions,
                self.probe_radii,
                self.probe_radii_noise,
                self.probe_gains,
                self.normal_stiffness,
                self.normal_damping,
                self.normal_exponent,
                self.shear_scalar,
                self.twist_scalar,
                self.n_probes_per_sensor,
                self.sensor_candidate_geom_mask,
                ground_truth_slot_0,
                measured_slot_0,
                solver.dyn_state,
                entry_a.bvh_state.tree,
                entry_b.bvh_state.tree,
                solver.dyn_info,
                measured_equals_gt,
                eps=gs.EPS,
                is_split=entry_b is not entry_a,
            )

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        def mask(envs_idx: EnvsIdxType | None) -> torch.Tensor:
            force = self.read(i_s, envs_idx, is_ground_truth=True).force
            if self.history_lengths[i_s] > 0:
                force = force.select(1 if self._sim.n_envs > 0 else 0, -1)
            return torch.linalg.norm(force, dim=-1) >= gs.EPS

        self._draw_debug_probes(i_s, context, self._tactile_color_groups_fn(i_s, mask))


class KinematicTaxelSensor(
    ProbeSensorMixin,
    LinkAttachedSensorMixin,
    SimpleSensor[KinematicTaxelOptions, KinematicTaxelSensorArray],
):
    """
    Sensor reading the spring-damper force and torque of each of its probes from the contact geometry and the relative
    motion.
    """
