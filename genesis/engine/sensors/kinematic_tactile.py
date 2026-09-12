from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.bvh import STACK_SIZE as _BVH_STACK_SIZE
from genesis.engine.solvers.rigid.collider.utils import func_point_in_geom_aabb
from genesis.options.sensors import ContactDepthProbe as ContactDepthProbeOptions
from genesis.options.sensors import ContactProbe as ContactProbeOptions
from genesis.options.sensors import KinematicTaxel as KinematicTaxelOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.raycast_qd import closest_point_on_triangle, get_triangle_vertices, triangle_face_normal

from .raycaster import RaycastContext

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata
from .contact_force import ContactFilterMetadataMixin, _func_link_is_filtered
from .probe import (
    ProbeSensorMetadataMixin,
    ProbeSensorMixin,
    ProbeSensorSharedMetadataT,
    func_noised_probe_radius,
    get_measured_bufs,
)
from .tactile_shared import (
    ContactDepthQueryMetadataMixin,
    ContactDepthQuerySensorMixin,
    ContactPrefilterMetadataMixin,
    SpatialCrosstalkMetadataMixin,
    SpatialCrosstalkMixin,
    ViscoelasticHysteresisMetadataMixin,
    ViscoelasticHysteresisMixin,
    func_sphere_intersects_aabb,
)

if TYPE_CHECKING:
    from genesis.options.sensors import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


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
    Max probe penetration from SDF over the sensor link's unique opposing geoms, dual-radius.
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


# Per-(env, sensor) cap on the prefiltered contact list consumed by the BVH-mask builder. Sensors track a
# single rigid link; even with multicontact and many neighbouring geoms, the count of contacts touching one
# link rarely exceeds a few hundred.
_MAX_CONTACTS_PER_SENSOR = 1024

# Per-(env, sensor) cap on the deduplicated opposing-geom list consumed by ``_func_query_contact_depth`` and
# ``_func_query_contact_depth_penetration`` (the SDF path). Unlike the contact list, this counts *distinct*
# contacting geoms, not contact points: one pressing object is a single entry regardless of how many contact
# points multicontact emits. A single rigid sensor link touching >64 distinct geoms at once is implausible,
# so 64 is generous; overflow silently truncates, matching ``_MAX_CONTACTS_PER_SENSOR``.
_MAX_GEOMS_PER_SENSOR = 64


@qd.kernel
def _kernel_build_sensor_contact_idx(
    sensor_link_idx: qd.types.ndarray(),
    filter_links_idx: qd.types.ndarray(),
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    """
    Per-(env, sensor) compact contact index for the KinematicTaxel pre-pass.

    Parallelizes over ``(n_batches, n_sensors)`` so the main kernel's per-probe contact-list scan drops from
    O(n_probes * n_contacts) to O(n_probes * sensor_n_contacts). The counterpart filter is applied here, before the
    per-sensor cap: a contact whose only tie to the sensor link is through a filtered counterpart is dropped and
    never consumes a cap slot, so a large filtered manifold (e.g. the ground) cannot starve an allowed contact.
    Cap-overflows (count >= last dim of ``sensor_contacts_idx``) silently truncate; see the module-level
    ``_MAX_CONTACTS_PER_SENSOR`` comment.
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


@qd.kernel
def _kernel_build_sensor_geom_idx(
    sensor_link_idx: qd.types.ndarray(),
    filter_links_idx: qd.types.ndarray(),
    sensor_geoms_idx: qd.types.ndarray(),
    sensor_n_geoms: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    """
    Per-(env, sensor) compact, deduplicated list of opposing contacting geoms for the SDF query path.

    Parallelizes over ``(n_batches, n_sensors)``, recording each contact's opposing geom (the side not on the
    sensor link). Deduping collapses the multicontact fan-out (tens of contacts on one pressing object -> one
    geom) so the SDF path's per-probe loop runs once per distinct contacting geom, not once per contact point.
    A contact whose counterpart link is in sensor ``i_s``'s ``filter_links_idx`` row is skipped here, so the SDF
    query loop never sees the filtered geom (the raycast path filters symmetrically in
    ``_kernel_build_sensor_candidate_geom_mask``), keeping the filter out of the per-probe hot loops.
    Cap-overflows (count >= last dim of ``sensor_geoms_idx``) silently truncate; see the module-level
    ``_MAX_GEOMS_PER_SENSOR`` comment.
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
    Dual-radius probe query: single SDF + normal pass yielding both GT and noised-radius results.

    Iterates the per-(env, sensor) deduplicated opposing-geom list built by ``_kernel_build_sensor_geom_idx``;
    every geom in that list contacts the sensor's tracked link, so the reported contact link is recovered as
    ``geoms_info.link_idx[i_g]`` (the link owning the opposing geom). AABB pre-filter expands by
    ``max(probe_radius_gt, probe_radius_m)`` so neither branch is
    silently skipped. Callers without a noised radius pass ``probe_radius_m == probe_radius_gt``.
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
    Kinematic spring-damper force / torque in the sensor link frame from a single probe's contact query.

    Shared by the GT and measured branches of ``_kernel_kinematic_taxel`` (they differ only in which dual-radius
    query result is fed in). Returns ``(force_local, torque_local)``; both zero when ``max_penetration <= 0``.
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


@qd.kernel
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
    n_batches = output_gt.shape[-1]

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
                output_gt[force_start + j, i_b] = gs.qd_float(0.0)
                output_gt[torque_start + j, i_b] = gs.qd_float(0.0)
                output_measured[force_start + j, i_b] = gs.qd_float(0.0)
                output_measured[torque_start + j, i_b] = gs.qd_float(0.0)
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
            func_noised_probe_radius(probe_radius, probe_radius_noise) if use_noised_radius else probe_radius
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
            output_gt[force_start + j, i_b] = force_local_gt[j]
            output_gt[torque_start + j, i_b] = torque_local_gt[j]
            output_measured[force_start + j, i_b] = force_local_m[j]
            output_measured[torque_start + j, i_b] = torque_local_m[j]


@qd.kernel
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
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]

        # Inactive filler probe (probe_radius == 0): reads zero depth (which contact-probe interprets as no contact).
        if probe_radii[i_p] <= gs.qd_float(0.0):
            cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
            output_gt[cache_idx, i_b] = gs.qd_float(0.0)
            output_measured[cache_idx, i_b] = gs.qd_float(0.0)
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
            func_noised_probe_radius(probe_radius, probe_radius_noise) if probe_radius_noise > gs.EPS else probe_radius
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
        output_gt[cache_idx, i_b] = max_penetration_gt
        output_measured[cache_idx, i_b] = max_penetration_m


# ============================ Raycast / BVH contact-depth path ============================


@qd.kernel
def _kernel_build_sensor_candidate_geom_mask(
    sensor_link_idx: qd.types.ndarray(),
    sensor_contacts_idx: qd.types.ndarray(),
    sensor_n_contacts: qd.types.ndarray(),
    sensor_candidate_geom_mask: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    """
    Scatter the per-(env, sensor) candidate-geom bitmask from the prefiltered contact list.

    Run only when the sensor class is in ``contact_depth_query="raycast"`` mode; the BVH leaf loop consults this mask
    to skip triangles whose owning geom isn't in the sensor's current contact list. Only the geom on the side opposite
    the sensor link is marked (mirroring the SDF path's ``i_g = <other geom>`` selection); marking the sensor's own
    geom would let the BVH closest-point test latch onto the sensor's own surface, pinning the reported depth to
    ``probe_radius`` regardless of the pressing object. The counterpart filter is already applied while building
    ``sensor_contacts_idx``, so every listed contact is an allowed one.
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
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    sensor_candidate_geom_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
):
    """
    BVH-based dual-radius probe penetration.

    Finds the signed distance to the nearest candidate triangle (sign from the closest triangle's face normal:
    negative when the probe is inside the surface, like ``_func_elastomer_min_signed_dist_bvh``) and returns
    ``max(0, R - sd)`` per radius. This matches the SDF path's ``pen = R - sd`` -- in particular it keeps growing as
    the probe penetrates, rather than folding back at ``R`` like an unsigned closest-point distance. Mirrors
    ``_func_query_contact_depth_penetration``'s return, with the nearest triangle's signed distance appended so a
    split-tree caller can select the globally nearest answer (see the kernels' fold).
    """
    # The tree's own leaf count: a compacted-subset tree (see RaycastContext.activate) has fewer leaves than faces.
    n_triangles = bvh_morton_codes.shape[1]
    radius_query = qd.max(probe_radius_gt, probe_radius_m)
    best_dist_sq = radius_query * radius_query
    best_signed = radius_query

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(_BVH_STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = bvh_nodes[i_t, node_idx]

        if not func_sphere_intersects_aabb(probe_pos, best_dist_sq, node.bound.min, node.bound.max):
            continue

        if node.left == -1:
            sorted_leaf_idx = node_idx - (n_triangles - 1)
            i_f = qd.cast(bvh_morton_codes[i_t, sorted_leaf_idx][1], gs.qd_int)
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
            if stack_idx < qd.static(_BVH_STACK_SIZE - 2):
                node_stack[stack_idx] = node.left
                node_stack[stack_idx + 1] = node.right
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
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    sensor_candidate_geom_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
):
    """
    BVH-based dual-radius probe query with contact normal and link, mirroring ``_func_query_contact_depth``'s return.

    Finds the nearest candidate triangle and its signed distance (sign from the face normal; negative when the probe
    is inside the surface), yielding ``pen = R - sd`` to match the SDF path. The returned contact normal is the
    nearest triangle's outward face normal, which the spring-damper model uses as the surface normal. The signed
    distance is appended so a split-tree caller can select the globally nearest answer (see the kernels' fold).
    """
    # The tree's own leaf count: a compacted-subset tree (see RaycastContext.activate) has fewer leaves than faces.
    n_triangles = bvh_morton_codes.shape[1]
    radius_query = qd.max(probe_radius_gt, probe_radius_m)
    best_dist_sq = radius_query * radius_query
    best_signed = radius_query
    contact_link = gs.qd_int(-1)
    contact_normal = qd.Vector.zero(gs.qd_float, 3)

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(_BVH_STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = bvh_nodes[i_t, node_idx]

        if not func_sphere_intersects_aabb(probe_pos, best_dist_sq, node.bound.min, node.bound.max):
            continue

        if node.left == -1:
            sorted_leaf_idx = node_idx - (n_triangles - 1)
            i_f = qd.cast(bvh_morton_codes[i_t, sorted_leaf_idx][1], gs.qd_int)
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
            if stack_idx < qd.static(_BVH_STACK_SIZE - 2):
                node_stack[stack_idx] = node.left
                node_stack[stack_idx + 1] = node.right
                stack_idx += 2

    # Penetration only; the link / normal are meaningful only for the branch that actually reports contact.
    max_pen_gt = qd.max(gs.qd_float(0.0), probe_radius_gt - best_signed)
    max_pen_m = qd.max(gs.qd_float(0.0), probe_radius_m - best_signed)
    contact_link_gt = contact_link if max_pen_gt > gs.qd_float(0.0) else gs.qd_int(-1)
    contact_link_m = contact_link if max_pen_m > gs.qd_float(0.0) else gs.qd_int(-1)
    contact_normal_gt = contact_normal if max_pen_gt > gs.qd_float(0.0) else qd.Vector.zero(gs.qd_float, 3)
    contact_normal_m = contact_normal if max_pen_m > gs.qd_float(0.0) else qd.Vector.zero(gs.qd_float, 3)
    return max_pen_gt, contact_link_gt, contact_normal_gt, max_pen_m, contact_link_m, contact_normal_m, best_signed


@qd.kernel(fastcache=False)
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
    bvh_nodes_a: qd.template(),
    bvh_morton_codes_a: qd.template(),
    bvh_nodes_b: qd.template(),
    bvh_morton_codes_b: qd.template(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    is_split: qd.template(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]

        if probe_radii[i_p] <= gs.qd_float(0.0):
            cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
            output_gt[cache_idx, i_b] = gs.qd_float(0.0)
            output_measured[cache_idx, i_b] = gs.qd_float(0.0)
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
            func_noised_probe_radius(probe_radius, probe_radius_noise) if probe_radius_noise > gs.EPS else probe_radius
        )

        max_penetration_gt, max_penetration_m, signed_dist = _func_query_contact_depth_penetration_bvh(
            env_bvh_idx_a[i_b],
            i_b,
            i_s,
            probe_pos,
            probe_radius,
            probe_radius_m,
            bvh_nodes_a,
            bvh_morton_codes_a,
            sensor_candidate_geom_mask,
            dyn_state,
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
                bvh_nodes_b,
                bvh_morton_codes_b,
                sensor_candidate_geom_mask,
                dyn_state,
                dyn_info,
            )
            if qd.abs(signed_dist_b) < qd.abs(signed_dist):
                max_penetration_gt = max_penetration_gt_b
                max_penetration_m = max_penetration_m_b
        max_penetration_m = max_penetration_m * probe_gains[i_b, i_p]
        cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
        output_gt[cache_idx, i_b] = max_penetration_gt
        output_measured[cache_idx, i_b] = max_penetration_m


@qd.kernel(fastcache=False)
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
    bvh_nodes_a: qd.template(),
    bvh_morton_codes_a: qd.template(),
    bvh_nodes_b: qd.template(),
    bvh_morton_codes_b: qd.template(),
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    measured_equals_gt: int,
    is_split: qd.template(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]
        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        force_start = cache_start + probe_idx_in_sensor * 3
        torque_start = cache_start + n_probes * 3 + probe_idx_in_sensor * 3

        if probe_radii[i_p] <= gs.qd_float(0.0):
            for j in qd.static(range(3)):
                output_gt[force_start + j, i_b] = gs.qd_float(0.0)
                output_gt[torque_start + j, i_b] = gs.qd_float(0.0)
                output_measured[force_start + j, i_b] = gs.qd_float(0.0)
                output_measured[torque_start + j, i_b] = gs.qd_float(0.0)
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
        use_noised_radius = probe_radius_noise > gs.EPS
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise) if use_noised_radius else probe_radius
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
            bvh_nodes_a,
            bvh_morton_codes_a,
            sensor_candidate_geom_mask,
            dyn_state,
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
                bvh_nodes_b,
                bvh_morton_codes_b,
                sensor_candidate_geom_mask,
                dyn_state,
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
            output_gt[force_start + j, i_b] = force_gt[j]
            output_gt[torque_start + j, i_b] = torque_gt[j]

        if measured_equals_gt == 1:
            for j in qd.static(range(3)):
                output_measured[force_start + j, i_b] = force_gt[j]
                output_measured[torque_start + j, i_b] = torque_gt[j]
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
                output_measured[force_start + j, i_b] = force_m[j]
                output_measured[torque_start + j, i_b] = torque_m[j]


class KinematicTactileSensorMixin(ContactDepthQuerySensorMixin, ProbeSensorMixin[ProbeSensorSharedMetadataT]):
    """Contact-depth probe family (ContactDepthProbe, ContactProbe, KinematicTaxel).

    The class-wide SDF/raycast backend is resolved and activated by ``ContactDepthQuerySensorMixin.build``;
    subclasses add their own metadata.
    """

    def build(self):
        super().build()
        self._shared_metadata.append_filter(self._options.filter_link_idx)


@dataclass
class ContactDepthProbeMetadata(
    ViscoelasticHysteresisMetadataMixin,
    ProbeSensorMetadataMixin,
    ContactFilterMetadataMixin,
    ContactPrefilterMetadataMixin,
    ContactDepthQueryMetadataMixin,
    RigidSensorMetadataMixin,
    SimpleSensorMetadata,
):
    pass


class ContactDepthProbeSensor(
    ViscoelasticHysteresisMixin[ContactDepthProbeMetadata],
    KinematicTactileSensorMixin[ContactDepthProbeMetadata],
    RigidSensorMixin[ContactDepthProbeMetadata],
    SimpleSensor[ContactDepthProbeOptions, RaycastContext, ContactDepthProbeMetadata, tuple],
):
    """
    Returns contact depth in meters per probe.
    """

    def build(self):
        super().build()
        # Re-allocate the per-(env, sensor) contact prefilter buffers to absorb the newly-registered sensor.
        B = self._manager._sim._B
        n_sensors_built = self._shared_metadata.n_probes_per_sensor.shape[0]
        self._shared_metadata.sensor_contacts_idx = torch.zeros(
            (B, n_sensors_built, _MAX_CONTACTS_PER_SENSOR), dtype=gs.tc_int, device=gs.device
        )
        self._shared_metadata.sensor_n_contacts = torch.zeros((B, n_sensors_built), dtype=gs.tc_int, device=gs.device)
        self._shared_metadata.sensor_geoms_idx = torch.zeros(
            (B, n_sensors_built, _MAX_GEOMS_PER_SENSOR), dtype=gs.tc_int, device=gs.device
        )
        self._shared_metadata.sensor_n_geoms = torch.zeros((B, n_sensors_built), dtype=gs.tc_int, device=gs.device)

    def _get_return_format(self) -> tuple[int, ...]:
        return self._probe_layout_shape

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_context: RaycastContext,
        shared_metadata: ContactDepthProbeMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        measured, measured_cols_b = get_measured_bufs(
            shared_metadata, current_ground_truth_data_T, measured_data_timeline
        )
        if (shared_metadata.contact_depth_query or "sdf") == "sdf":
            _kernel_build_sensor_geom_idx(
                shared_metadata.links_idx,
                shared_metadata.filter_links_idx,
                shared_metadata.sensor_geoms_idx,
                shared_metadata.sensor_n_geoms,
                solver.collider.collider_state,
            )
            _kernel_contact_depth_probe(
                shared_metadata.probe_sensor_idx,
                shared_metadata.links_idx,
                shared_metadata.sensor_cache_start,
                shared_metadata.sensor_probe_start,
                shared_metadata.sensor_geoms_idx,
                shared_metadata.probe_positions,
                shared_metadata.probe_radii,
                shared_metadata.probe_radii_noise,
                shared_metadata.probe_gains,
                shared_metadata.sensor_n_geoms,
                current_ground_truth_data_T,
                measured_cols_b,
                solver.dyn_state,
                solver.dyn_info,
                solver.collider.collider_info,
            )
        else:
            _kernel_build_sensor_contact_idx(
                shared_metadata.links_idx,
                shared_metadata.filter_links_idx,
                shared_metadata.sensor_contacts_idx,
                shared_metadata.sensor_n_contacts,
                solver.collider.collider_state,
            )
            B, n_sensors = shared_metadata.sensor_n_contacts.shape
            mask_shape = (B, n_sensors, solver.n_geoms)
            if tuple(shared_metadata.sensor_candidate_geom_mask.shape) != mask_shape:
                shared_metadata.sensor_candidate_geom_mask = torch.zeros(mask_shape, dtype=gs.tc_bool, device=gs.device)
            _kernel_build_sensor_candidate_geom_mask(
                shared_metadata.links_idx,
                shared_metadata.sensor_contacts_idx,
                shared_metadata.sensor_n_contacts,
                shared_metadata.sensor_candidate_geom_mask,
                solver.collider.collider_state,
            )
            collision_bvh_contexts = shared_context.collision_bvh_contexts
            entry_a, entry_b = collision_bvh_contexts[0], collision_bvh_contexts[-1]
            _kernel_contact_depth_probe_bvh(
                shared_metadata.probe_sensor_idx,
                shared_metadata.links_idx,
                entry_a.env_bvh_idx,
                entry_b.env_bvh_idx,
                shared_metadata.sensor_cache_start,
                shared_metadata.sensor_probe_start,
                shared_metadata.probe_positions,
                shared_metadata.probe_radii,
                shared_metadata.probe_radii_noise,
                shared_metadata.probe_gains,
                shared_metadata.sensor_candidate_geom_mask,
                entry_a.bvh.nodes,
                entry_a.bvh.morton_codes,
                entry_b.bvh.nodes,
                entry_b.bvh.morton_codes,
                current_ground_truth_data_T,
                measured_cols_b,
                solver.dyn_state,
                solver.dyn_info,
                is_split=entry_b is not entry_a,
            )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        def mask(envs_idx):
            depth = self.read_ground_truth(envs_idx)
            if self._options.history_length > 0:
                depth = depth.select(1 if self._manager._sim.n_envs > 0 else 0, -1)
            return depth >= gs.EPS

        self._draw_debug_probes(context, self._tactile_color_groups_fn(mask))


@dataclass
class ContactProbeMetadata(ContactDepthProbeMetadata):
    contact_threshold: torch.Tensor = make_tensor_field((0,))
    release_threshold: torch.Tensor = make_tensor_field((0,))
    # Per-probe gate levels scattered into intermediate-cache layout, computed lazily on first `_post_process`.
    enter_row: torch.Tensor = make_tensor_field((0,))
    exit_row: torch.Tensor = make_tensor_field((0,))


class ContactProbeSensor(
    ContactDepthProbeSensor, SimpleSensor[ContactProbeOptions, RaycastContext, ContactProbeMetadata, tuple]
):
    """
    Returns boolean contact per probe with optional Schmitt-trigger hysteresis.

    Shares the depth-probe kernel. The contact bit latches on when depth exceeds ``contact_threshold`` and releases
    when depth drops to or below ``release_threshold``. When ``release_threshold`` is left unset (the default; it then
    falls back to ``contact_threshold``), the latch is degenerate and behavior matches a stateless threshold. Latch
    state is read from the per-branch return-space ring, so GT and measured branches latch independently and reset
    cleanly with the env (the manager zeros the ring on reset).
    """

    def build(self):
        super().build()
        self._shared_metadata.contact_threshold = concat_with_tensor(
            self._shared_metadata.contact_threshold, self._options.contact_threshold, expand=(1,)
        )
        exit_level = (
            self._options.contact_threshold
            if self._options.release_threshold is None
            else self._options.release_threshold
        )
        self._shared_metadata.release_threshold = concat_with_tensor(
            self._shared_metadata.release_threshold, exit_level, expand=(1,)
        )

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_bool

    @classmethod
    def _get_intermediate_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _post_process(
        cls,
        shared_metadata: ContactProbeMetadata,
        tensor: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ) -> torch.Tensor:
        if shared_metadata.enter_row.shape != (tensor.shape[1],) or shared_metadata.enter_row.dtype != tensor.dtype:
            i_p = torch.arange(shared_metadata.total_n_probes, device=gs.device, dtype=gs.tc_int)
            i_s = shared_metadata.probe_sensor_idx
            cache_idx = shared_metadata.sensor_cache_start[i_s] + i_p - shared_metadata.sensor_probe_start[i_s]
            cache_idx_64 = cache_idx.to(dtype=torch.int64)
            enter_row = torch.zeros((tensor.shape[1],), dtype=tensor.dtype, device=gs.device)
            enter_row.scatter_(0, cache_idx_64, shared_metadata.contact_threshold[i_s].to(dtype=tensor.dtype))
            exit_row = torch.zeros((tensor.shape[1],), dtype=tensor.dtype, device=gs.device)
            exit_row.scatter_(0, cache_idx_64, shared_metadata.release_threshold[i_s].to(dtype=tensor.dtype))
            shared_metadata.enter_row = enter_row
            shared_metadata.exit_row = exit_row
        above_enter = tensor > shared_metadata.enter_row.unsqueeze(0)
        above_exit = tensor > shared_metadata.exit_row.unsqueeze(0)
        prev_state = timeline.at(0, copy=False)
        return above_enter | (prev_state & above_exit)

    def _draw_debug(self, context: "RasterizerContext"):
        def mask(envs_idx):
            contact = self.read_ground_truth(envs_idx)
            if self._options.history_length > 0:
                contact = contact.select(1 if self._manager._sim.n_envs > 0 else 0, -1)
            return contact

        self._draw_debug_probes(context, self._tactile_color_groups_fn(mask))


class KinematicTaxelReturnType(NamedTuple):
    """
    Parameters
    ----------
    force: torch.Tensor, shape ([n_envs,] n_probes, 3)
        Estimated contact force in the link frame from the kinematic spring-damper model.
    torque: torch.Tensor, shape ([n_envs,] n_probes, 3)
    """

    force: torch.Tensor
    torque: torch.Tensor


@dataclass
class KinematicTaxelMetadata(
    ViscoelasticHysteresisMetadataMixin,
    SpatialCrosstalkMetadataMixin,
    ProbeSensorMetadataMixin,
    ContactFilterMetadataMixin,
    ContactPrefilterMetadataMixin,
    ContactDepthQueryMetadataMixin,
    RigidSensorMetadataMixin,
    SimpleSensorMetadata,
):
    normal_stiffness: torch.Tensor = make_tensor_field((0,))
    normal_damping: torch.Tensor = make_tensor_field((0,))
    normal_exponent: torch.Tensor = make_tensor_field((0,))
    shear_scalar: torch.Tensor = make_tensor_field((0,))
    twist_scalar: torch.Tensor = make_tensor_field((0,))


class KinematicTaxelSensor(
    ViscoelasticHysteresisMixin[KinematicTaxelMetadata],
    SpatialCrosstalkMixin[KinematicTaxelMetadata],
    KinematicTactileSensorMixin[KinematicTaxelMetadata],
    RigidSensorMixin[KinematicTaxelMetadata],
    SimpleSensor[KinematicTaxelOptions, RaycastContext, KinematicTaxelMetadata, KinematicTaxelReturnType],
):
    """Kinematic taxels: spring-damper force and torque per probe from contact geometry and relative motion."""

    # Two channel groups: force xyz followed by torque xyz (probe-major within each group). See
    # ``ProbeSensorMixin._taxel_channel_groups`` for how this drives dead-taxel cache-col -> probe mapping.
    _taxel_channel_groups: int = 2

    def __init__(
        self, options: KinematicTaxelOptions, idx: int, shared_context, shared_metadata, manager: "SensorManager"
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        # Resolve the grid frame for spatial crosstalk (flat pos/normals are already populated by the base mixins).
        self._setup_crosstalk_grid(options)

    def build(self):
        super().build()

        self._shared_metadata.normal_stiffness = concat_with_tensor(
            self._shared_metadata.normal_stiffness, float(self._options.normal_stiffness), expand=(1,)
        )
        self._shared_metadata.normal_damping = concat_with_tensor(
            self._shared_metadata.normal_damping, float(self._options.normal_damping), expand=(1,)
        )
        self._shared_metadata.normal_exponent = concat_with_tensor(
            self._shared_metadata.normal_exponent, float(self._options.normal_exponent), expand=(1,)
        )
        self._shared_metadata.shear_scalar = concat_with_tensor(
            self._shared_metadata.shear_scalar, float(self._options.shear_scalar), expand=(1,)
        )
        self._shared_metadata.twist_scalar = concat_with_tensor(
            self._shared_metadata.twist_scalar, float(self._options.twist_scalar), expand=(1,)
        )

        if self._options.is_crosstalk_enabled and self._use_grid_crosstalk:
            self._register_crosstalk()

        # Re-allocate the per-(env, sensor) contact prefilter buffers to absorb the newly-registered sensor.
        # Sized at build time; the per-step kernel writes into the same buffers without further allocation.
        B = self._manager._sim._B
        n_sensors_built = self._shared_metadata.n_probes_per_sensor.shape[0]
        self._shared_metadata.sensor_contacts_idx = torch.zeros(
            (B, n_sensors_built, _MAX_CONTACTS_PER_SENSOR), dtype=gs.tc_int, device=gs.device
        )
        self._shared_metadata.sensor_n_contacts = torch.zeros((B, n_sensors_built), dtype=gs.tc_int, device=gs.device)
        self._shared_metadata.sensor_geoms_idx = torch.zeros(
            (B, n_sensors_built, _MAX_GEOMS_PER_SENSOR), dtype=gs.tc_int, device=gs.device
        )
        self._shared_metadata.sensor_n_geoms = torch.zeros((B, n_sensors_built), dtype=gs.tc_int, device=gs.device)

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        shape = (*self._probe_layout_shape, 3)
        return shape, shape

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_context: RaycastContext,
        shared_metadata: KinematicTaxelMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver
        measured, measured_cols_b = get_measured_bufs(
            shared_metadata, current_ground_truth_data_T, measured_data_timeline
        )
        # The measured branch is provably identical to GT (and the kernel can skip recomputing it) when no probe
        # has a noised sensing radius and no probe has a non-unit measured-branch gain.
        measured_equals_gt = int(
            not shared_metadata.has_any_probe_radius_noise and not shared_metadata.has_any_probe_gain
        )
        if (shared_metadata.contact_depth_query or "sdf") == "sdf":
            _kernel_build_sensor_geom_idx(
                shared_metadata.links_idx,
                shared_metadata.filter_links_idx,
                shared_metadata.sensor_geoms_idx,
                shared_metadata.sensor_n_geoms,
                solver.collider.collider_state,
            )
            _kernel_kinematic_taxel(
                shared_metadata.probe_sensor_idx,
                shared_metadata.links_idx,
                shared_metadata.sensor_cache_start,
                shared_metadata.sensor_probe_start,
                shared_metadata.sensor_geoms_idx,
                shared_metadata.probe_positions,
                shared_metadata.probe_radii,
                shared_metadata.probe_radii_noise,
                shared_metadata.probe_gains,
                shared_metadata.normal_stiffness,
                shared_metadata.normal_damping,
                shared_metadata.normal_exponent,
                shared_metadata.shear_scalar,
                shared_metadata.twist_scalar,
                shared_metadata.n_probes_per_sensor,
                shared_metadata.sensor_n_geoms,
                current_ground_truth_data_T,
                measured_cols_b,
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
                shared_metadata.links_idx,
                shared_metadata.filter_links_idx,
                shared_metadata.sensor_contacts_idx,
                shared_metadata.sensor_n_contacts,
                solver.collider.collider_state,
            )
            B, n_sensors = shared_metadata.sensor_n_contacts.shape
            mask_shape = (B, n_sensors, solver.n_geoms)
            if tuple(shared_metadata.sensor_candidate_geom_mask.shape) != mask_shape:
                shared_metadata.sensor_candidate_geom_mask = torch.zeros(mask_shape, dtype=gs.tc_bool, device=gs.device)
            _kernel_build_sensor_candidate_geom_mask(
                shared_metadata.links_idx,
                shared_metadata.sensor_contacts_idx,
                shared_metadata.sensor_n_contacts,
                shared_metadata.sensor_candidate_geom_mask,
                solver.collider.collider_state,
            )
            collision_bvh_contexts = shared_context.collision_bvh_contexts
            entry_a, entry_b = collision_bvh_contexts[0], collision_bvh_contexts[-1]
            _kernel_kinematic_taxel_bvh(
                shared_metadata.probe_sensor_idx,
                shared_metadata.links_idx,
                entry_a.env_bvh_idx,
                entry_b.env_bvh_idx,
                shared_metadata.sensor_cache_start,
                shared_metadata.sensor_probe_start,
                shared_metadata.probe_positions,
                shared_metadata.probe_radii,
                shared_metadata.probe_radii_noise,
                shared_metadata.probe_gains,
                shared_metadata.normal_stiffness,
                shared_metadata.normal_damping,
                shared_metadata.normal_exponent,
                shared_metadata.shear_scalar,
                shared_metadata.twist_scalar,
                shared_metadata.n_probes_per_sensor,
                shared_metadata.sensor_candidate_geom_mask,
                entry_a.bvh.nodes,
                entry_a.bvh.morton_codes,
                entry_b.bvh.nodes,
                entry_b.bvh.morton_codes,
                current_ground_truth_data_T,
                measured_cols_b,
                solver.dyn_state,
                solver.dyn_info,
                measured_equals_gt,
                is_split=entry_b is not entry_a,
            )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        def mask(envs_idx):
            force = self.read_ground_truth(envs_idx).force
            if self._options.history_length > 0:
                force = force.select(1 if self._manager._sim.n_envs > 0 else 0, -1)
            return torch.linalg.norm(force, dim=-1) >= gs.EPS

        self._draw_debug_probes(context, self._tactile_color_groups_fn(mask))
