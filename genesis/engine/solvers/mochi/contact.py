# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Contact of the MochiSolver: surface sample points, conservative broadphase over (link, collider geom) pairs, and
the assembly of the smooth penalty response into the residual and Hessian blocks."""

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .colliders import query_collider, query_collider_lower_bound
from .contact_utils import collision_response, func_mat3_to_sym6, func_sym6_to_mat3
from .data import (
    COLLIDER_TYPE,
    MochiContactState,
    MochiHitReadback,
    MochiInfo,
    MochiSamplesInfo,
    MochiSoftInfo,
    MochiSoftState,
    MochiState,
)
from .lie import skew
from .newton import func_is_env_active

# Barycentric coordinates and reference weights (integrating to 1/2, the reference triangle area) of the triangle
# quadrature rules placing the contact samples: centroid, degree-2 and degree-4 (Dunavant) rules.
TRIANGLE_QUADRATURES = {
    "P1Q1": (np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([0.5])),
    "P1Q3": (
        np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]),
        np.array([1 / 6, 1 / 6, 1 / 6]),
    ),
    "P1Q6": (
        np.array(
            [
                [0.816847572980459, 0.091576213509771, 0.091576213509771],
                [0.091576213509771, 0.816847572980459, 0.091576213509771],
                [0.091576213509771, 0.091576213509771, 0.816847572980459],
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
            ]
        ),
        np.array([0.109951743655322 / 2] * 3 + [0.223381589678011 / 2] * 3),
    ),
}

# Motion bound of the per-step conservative bounding boxes: the stage-start speed is doubled and a large acceleration
# is added so that the pairs found at the start of the step cover every iterate of the solve.
CONSERVATIVE_SPEED_SCALE = 2.0
CONSERVATIVE_MAX_ACCEL = 400.0


def build_geom_samples(geom, quadrature):
    """Quadrature points of the collision triangles of a geom, expressed in the link frame, with their area weights
    and outward normals. Returns (positions, normals, weights)."""
    bary, ref_weights = TRIANGLE_QUADRATURES[quadrature]
    verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
    faces = geom.init_faces
    tri = verts[faces]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(edge_1, edge_2)
    areas_2 = np.linalg.norm(cross, axis=-1)
    normals = cross / np.maximum(areas_2, gs.EPS)[:, None]
    # The quadrature weights of the reference triangle integrate to 1/2; the physical weight scales by |det J| = 2 A.
    positions = np.einsum("qk,fkd->fqd", bary, tri).reshape((-1, 3))
    normals = np.repeat(normals, len(bary), axis=0)
    weights = (ref_weights[None, :] * areas_2[:, None]).reshape((-1,))
    return positions, normals, weights


@qd.kernel
def kernel_init_mochi_fields(
    links_is_dynamic: qd.types.ndarray(),
    links_has_gravity: qd.types.ndarray(),
    links_mass: qd.types.ndarray(),
    links_inertia: qd.types.ndarray(),
    links_sample_start: qd.types.ndarray(),
    links_sample_end: qd.types.ndarray(),
    links_samples_aabb_min: qd.types.ndarray(),
    links_samples_aabb_max: qd.types.ndarray(),
    geoms_collider_type: qd.types.ndarray(),
    geoms_contact_params: qd.types.ndarray(),
    samples_pos: qd.types.ndarray(),
    samples_normal: qd.types.ndarray(),
    samples_weight: qd.types.ndarray(),
    samples_link_idx: qd.types.ndarray(),
    samples_geom_idx: qd.types.ndarray(),
    links_pair_enabled: qd.types.ndarray(),
    dofs_entity_mass: qd.types.ndarray(),
    mochi_info: MochiInfo,
    rigid_config: qd.template(),
):
    n_links = links_is_dynamic.shape[0]
    n_dofs = dofs_entity_mass.shape[0]
    n_geoms = geoms_collider_type.shape[0]
    n_samples = samples_weight.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l in range(n_links):
        mochi_info.links.is_dynamic[i_l] = links_is_dynamic[i_l]
        mochi_info.links.has_gravity[i_l] = links_has_gravity[i_l]
        mochi_info.links.mass[i_l] = links_mass[i_l]
        mochi_info.links.sample_start[i_l] = links_sample_start[i_l]
        mochi_info.links.sample_end[i_l] = links_sample_end[i_l]
        trace = gs.qd_float(0.0)
        for k, l in qd.static(qd.ndrange(3, 3)):
            mochi_info.links.inertia[i_l][k, l] = links_inertia[i_l, k, l]
        for k in qd.static(range(3)):
            trace += links_inertia[i_l, k, k]
        for k, l in qd.static(qd.ndrange(3, 3)):
            mochi_info.links.second_moment[i_l][k, l] = -links_inertia[i_l, k, l]
        for k in qd.static(range(3)):
            mochi_info.links.second_moment[i_l][k, k] += 0.5 * trace
            mochi_info.links.samples_aabb_min[i_l][k] = links_samples_aabb_min[i_l, k]
            mochi_info.links.samples_aabb_max[i_l][k] = links_samples_aabb_max[i_l, k]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g in range(n_geoms):
        mochi_info.geoms.collider_type[i_g] = geoms_collider_type[i_g]
        mochi_info.geoms.penalty_coefficient[i_g] = geoms_contact_params[i_g, 0]
        mochi_info.geoms.penalty_smoothing_half_distance[i_g] = geoms_contact_params[i_g, 1]
        mochi_info.geoms.penalty_threshold[i_g] = geoms_contact_params[i_g, 2]
        mochi_info.geoms.friction[i_g] = geoms_contact_params[i_g, 3]
        mochi_info.geoms.friction_falloff_vel[i_g] = geoms_contact_params[i_g, 4]
        mochi_info.geoms.viscous_friction[i_g] = geoms_contact_params[i_g, 5]
        mochi_info.geoms.normal_viscous_damping[i_g] = geoms_contact_params[i_g, 6]
        mochi_info.geoms.max_alignment_normals[i_g] = geoms_contact_params[i_g, 7]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s in range(n_samples):
        for k in qd.static(range(3)):
            mochi_info.samples.pos[i_s][k] = samples_pos[i_s, k]
            mochi_info.samples.normal[i_s][k] = samples_normal[i_s, k]
        mochi_info.samples.weight[i_s] = samples_weight[i_s]
        mochi_info.samples.link_idx[i_s] = samples_link_idx[i_s]
        mochi_info.samples.geom_idx[i_s] = samples_geom_idx[i_s]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_la, i_lb in qd.ndrange(n_links, n_links):
        mochi_info.links_pair_enabled[i_la, i_lb] = links_pair_enabled[i_la, i_lb]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d in range(n_dofs):
        mochi_info.dofs_entity_mass[i_d] = dofs_entity_mass[i_d]


@qd.kernel
def kernel_set_links_pair_enabled(
    links_pair_enabled: qd.types.ndarray(),
    mochi_info: MochiInfo,
    rigid_config: qd.template(),
):
    n_links = links_pair_enabled.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_la, i_lb in qd.ndrange(n_links, n_links):
        mochi_info.links_pair_enabled[i_la, i_lb] = links_pair_enabled[i_la, i_lb]


@qd.func
def func_zero_assembly(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
    record: qd.template(),
):
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    max_pairs = contact_state.pair_link_a.shape[0]
    max_hits = hit_readback.hit_sample.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            if qd.static(assem_obj):
                mochi_state.obj[i_b] = 0.0
            if qd.static(record):
                hit_readback.n_hits_total[i_b] = 0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_slot in qd.ndrange(n_dofs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_dofs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            if qd.static(assem_res):
                mochi_state.res[i_d, i_b] = 0.0
            if assem_dres:
                mochi_state.dofs_H_diag[i_d, i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            if qd.static(assem_res):
                mochi_state.links_res[i_l, i_b] = qd.Vector.zero(gs.qd_float, 6)
            if assem_dres:
                mochi_state.H_diag[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)
            if qd.static(record):
                dyn_state.links.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(record):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_h, i_slot in qd.ndrange(max_hits, n_envs[None]) if qd.static(not per_env) else qd.ndrange(max_hits, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            hit_readback.hit_geom_a[i_h, i_b] = -1
            hit_readback.hit_geom_b[i_h, i_b] = -1
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_slot in qd.ndrange(max_pairs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(max_pairs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done) and i_p < contact_state.n_pairs[i_b]:
            contact_state.acc_f[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            contact_state.acc_q[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            contact_state.acc_D[i_p, i_b] = qd.Vector.zero(gs.qd_float, 6)
            contact_state.acc_SD[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            contact_state.acc_SDS[i_p, i_b] = qd.Vector.zero(gs.qd_float, 6)
            contact_state.acc_obj[i_p, i_b] = 0.0
            contact_state.n_hits[i_p, i_b] = 0


@qd.kernel
def kernel_zero_assembly(
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.template(),
    skip_ls_done: qd.i32,
    record: qd.template(),
):
    func_zero_assembly(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        mochi_state,
        contact_state,
        hit_readback,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
        record,
    )


@qd.func
def func_conservative_bounds(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
):
    """World bounds of every link's sample cloud that hold for the whole step, and the motion padding of every link,
    evaluated at the stage-start configuration."""
    n_links = dyn_state.links.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    dt = mochi_info.dt[None]
    margin = mochi_info.broadphase_margin[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        aabb_min = mochi_info.links.samples_aabb_min[i_l]
        aabb_max = mochi_info.links.samples_aabb_max[i_l]
        pad = margin
        if mochi_info.links.is_dynamic[i_l]:
            vel = mochi_state.links_vel_stage_start[i_l, i_b]
            omega = mochi_state.links_ang_stage_start[i_l, i_b]
            radius = qd.max(aabb_min.norm(), aabb_max.norm())
            speed = vel.norm() + omega.norm() * radius
            speed = CONSERVATIVE_SPEED_SCALE * speed + CONSERVATIVE_MAX_ACCEL * dt
            if mochi_info.links.has_gravity[i_l]:
                speed += mochi_info.gravity[i_b].norm() * dt
            pad += speed * dt
        contact_state.links_step_pad[i_l, i_b] = pad

        pos = dyn_state.links.pos[i_l, i_b]
        quat = dyn_state.links.quat[i_l, i_b]
        world_min = qd.Vector([gs.qd_float(1e30)] * 3, dt=gs.qd_float)
        world_max = -world_min
        for corner in qd.static(range(8)):
            local = qd.Vector.zero(gs.qd_float, 3)
            for k in qd.static(range(3)):
                local[k] = aabb_max[k] if qd.static((corner >> k) & 1) else aabb_min[k]
            world = gu.qd_transform_by_trans_quat(local, pos, quat)
            world_min = qd.min(world_min, world)
            world_max = qd.max(world_max, world)
        contact_state.links_step_aabb_min[i_l, i_b] = world_min - pad
        contact_state.links_step_aabb_max[i_l, i_b] = world_max + pad


@qd.kernel
def kernel_conservative_bounds(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
):
    func_conservative_bounds(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        rigid_config,
    )


@qd.func
def func_broadphase_pairs(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """Enumerate the (link with samples, collider geom) pairs whose conservative bounds overlap within the step. Both
    orderings of two links are emitted when both carry samples and colliders."""
    n_links = dyn_state.links.pos.shape[0]
    n_geoms = dyn_state.geoms.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    max_pairs = contact_state.pair_link_a.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        contact_state.n_pairs[i_b] = 0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_la, i_gb, i_slot in (
        qd.ndrange(n_links, n_geoms, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, n_geoms, 1)
    ):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not mochi_state.is_active[i_b]:
            continue
        if mochi_info.links.sample_end[i_la] <= mochi_info.links.sample_start[i_la]:
            continue
        if mochi_info.geoms.collider_type[i_gb] == COLLIDER_TYPE.NONE:
            continue
        i_lb = dyn_info.geoms.link_idx[i_gb]
        if i_lb == i_la:
            continue
        # A static body is only a collider: its samples never collide (mochi's rule); the collider may be static.
        if not mochi_info.links.is_dynamic[i_la]:
            continue
        if not mochi_info.links_pair_enabled[i_la, i_lb]:
            continue
        # A plane is a half-space: its mesh bounds say nothing about the penetration side, so it is never culled.
        if mochi_info.geoms.collider_type[i_gb] != COLLIDER_TYPE.PLANE:
            band = contact_state.links_step_pad[i_lb, i_b] + mochi_info.geoms.penalty_threshold[i_gb]
            geom_min = dyn_state.geoms.aabb_min[i_gb, i_b] - band
            geom_max = dyn_state.geoms.aabb_max[i_gb, i_b] + band
            if (contact_state.links_step_aabb_max[i_la, i_b] < geom_min).any():
                continue
            if (contact_state.links_step_aabb_min[i_la, i_b] > geom_max).any():
                continue
        i_p = qd.atomic_add(contact_state.n_pairs[i_b], 1)
        if i_p < max_pairs:
            contact_state.pair_link_a[i_p, i_b] = i_la
            contact_state.pair_link_b[i_p, i_b] = i_lb
            contact_state.pair_geom_b[i_p, i_b] = i_gb
        else:
            qd.atomic_or(errno[i_b], array_class.ErrorCode.OVERFLOW_MOCHI_CONTACT_PAIRS)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        contact_state.n_pairs[i_b] = qd.min(contact_state.n_pairs[i_b], max_pairs)


@qd.kernel
def kernel_broadphase_pairs(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    func_broadphase_pairs(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        rigid_config,
        errno,
    )


@qd.func
def func_contact_eval_sample(
    i_p,
    i_s,
    i_b,
    i_la,
    i_lb,
    i_gb,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_dres,
    record: qd.template(),
    errno: qd.Tensor,
):
    """Evaluate one sample of a candidate pair against its collider at the current iterate and accumulate the pair's
    force, torque and Hessian sums."""
    max_hits = hit_readback.hit_sample.shape[0]
    EPS = mochi_info.EPS[None]
    i_ga = mochi_info.samples.geom_idx[i_s]
    contype_a = dyn_info.geoms.contype[i_ga]
    conaffinity_a = dyn_info.geoms.conaffinity[i_ga]
    contype_b = dyn_info.geoms.contype[i_gb]
    conaffinity_b = dyn_info.geoms.conaffinity[i_gb]
    is_hit = (contype_a & conaffinity_b) != 0 or (contype_b & conaffinity_a) != 0

    pos_a = dyn_state.links.pos[i_la, i_b]
    quat_a = dyn_state.links.quat[i_la, i_b]
    pos = gu.qd_transform_by_trans_quat(mochi_info.samples.pos[i_s], pos_a, quat_a)
    thr = mochi_info.geoms.penalty_threshold[i_gb]
    h = mochi_info.geoms.penalty_smoothing_half_distance[i_gb]
    # Contact range: the penalty and its derivatives vanish beyond the threshold (mochi's detection range).
    band = thr
    if is_hit and mochi_info.geoms.collider_type[i_gb] != COLLIDER_TYPE.PLANE:
        if (pos < dyn_state.geoms.aabb_min[i_gb, i_b] - band).any():
            is_hit = False
        if (pos > dyn_state.geoms.aabb_max[i_gb, i_b] + band).any():
            is_hit = False

    pos_g = dyn_state.geoms.pos[i_gb, i_b]
    quat_g = dyn_state.geoms.quat[i_gb, i_b]
    pos_geom = gu.qd_inv_transform_by_trans_quat(pos, pos_g, quat_g)
    d = gs.qd_float(0.0)
    grad = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    if is_hit:
        is_valid, d_query, grad_query = query_collider(
            i_gb, pos_geom, dyn_info.geoms, mochi_info.geoms, sdf_info, mochi_config
        )
        d = d_query
        grad = grad_query
        if not is_valid or d > band:
            is_hit = False

    if is_hit:
        # Colliding normal and stage displacement of the sample, in the collider frame.
        normal_world = gu.qd_transform_by_quat(mochi_info.samples.normal[i_s], quat_a)
        normal_geom = gu.qd_inv_transform_by_quat(normal_world, quat_g)
        pos_start = gu.qd_transform_by_trans_quat(
            mochi_info.samples.pos[i_s],
            mochi_state.links_pos_stage_start[i_la, i_b],
            mochi_state.links_quat_stage_start[i_la, i_b],
        )
        pos_geom_start = gu.qd_inv_transform_by_trans_quat(
            pos_start, mochi_state.geoms_pos_stage_start[i_gb, i_b], mochi_state.geoms_quat_stage_start[i_gb, i_b]
        )
        p_rel = pos_geom - pos_geom_start
        d_start = d - grad.dot(p_rel)

        is_static_b = not mochi_info.links.is_dynamic[i_lb]
        k = qd.sqrt(mochi_info.geoms.penalty_coefficient[i_ga] * mochi_info.geoms.penalty_coefficient[i_gb])
        falloff = qd.sqrt(mochi_info.geoms.friction_falloff_vel[i_ga] * mochi_info.geoms.friction_falloff_vel[i_gb])
        if is_static_b:
            k = mochi_info.geoms.penalty_coefficient[i_ga]
            falloff = mochi_info.geoms.friction_falloff_vel[i_ga]
        mu = qd.sqrt(mochi_info.geoms.friction[i_ga] * mochi_info.geoms.friction[i_gb])
        c_visc = qd.sqrt(mochi_info.geoms.viscous_friction[i_ga] * mochi_info.geoms.viscous_friction[i_gb])
        c_ndamp = qd.sqrt(mochi_info.geoms.normal_viscous_damping[i_ga] * mochi_info.geoms.normal_viscous_damping[i_gb])
        max_align = mochi_info.geoms.max_alignment_normals[i_gb]

        energy, force_geom, dforce_geom, _ = collision_response(
            d,
            grad,
            normal_geom,
            p_rel,
            d_start,
            k,
            h,
            thr,
            mu,
            falloff,
            c_visc,
            c_ndamp,
            max_align,
            mochi_state.dt_stage[i_b],
            EPS,
            mochi_config,
        )

        w = mochi_info.samples.weight[i_s]
        R_g = gu.qd_quat_to_R(quat_g, EPS)
        force = R_g @ force_geom
        r_b = pos - dyn_state.links.pos[i_lb, i_b]
        qd.atomic_add(contact_state.acc_f[i_p, i_b], w * force)
        qd.atomic_add(contact_state.acc_q[i_p, i_b], w * r_b.cross(force))
        qd.atomic_add(contact_state.acc_obj[i_p, i_b], w * energy)
        qd.atomic_add(contact_state.n_hits[i_p, i_b], 1)
        # The three Hessian sums are read by kernel_pairs_to_blocks under the same flag, and they carry most of the
        # atomic traffic of this kernel: the line search re-evaluates contact for the residual alone.
        if assem_dres:
            D = -w * (R_g @ dforce_geom @ R_g.transpose())
            S_b = skew(r_b)
            qd.atomic_add(contact_state.acc_D[i_p, i_b], func_mat3_to_sym6(D))
            qd.atomic_add(contact_state.acc_SD[i_p, i_b], S_b @ D)
            qd.atomic_add(contact_state.acc_SDS[i_p, i_b], func_mat3_to_sym6(S_b @ D @ S_b))

        if qd.static(record):
            qd.atomic_add(dyn_state.links.contact_force[i_la, i_b], w * force)
            qd.atomic_add(dyn_state.links.contact_force[i_lb, i_b], -w * force)
            i_h = qd.atomic_add(hit_readback.n_hits_total[i_b], 1)
            if i_h < max_hits:
                hit_readback.hit_link_a[i_h, i_b] = i_la
                hit_readback.hit_geom_a[i_h, i_b] = i_ga
                hit_readback.hit_link_b[i_h, i_b] = i_lb
                hit_readback.hit_geom_b[i_h, i_b] = i_gb
                hit_readback.hit_sample[i_h, i_b] = i_s
                hit_readback.hit_pos[i_h, i_b] = pos
                hit_readback.hit_normal[i_h, i_b] = gu.qd_normalize(R_g @ grad, EPS)
                hit_readback.hit_force[i_h, i_b] = w * force
                hit_readback.hit_distance[i_h, i_b] = d
                hit_readback.hit_weight[i_h, i_b] = w
            else:
                qd.atomic_or(errno[i_b], array_class.ErrorCode.OVERFLOW_MOCHI_CONTACTS)


@qd.func
def func_contact_eval(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_dres,
    skip_ls_done,
    record: qd.template(),
    errno: qd.Tensor,
):
    """Evaluate the samples of every candidate pair that can lie within the penalty band of the collider at the
    current iterate, and accumulate the per-pair force, torque and Hessian sums. Contact is re-detected here at every
    call, so the set of active samples follows the Newton iterates; the sample hierarchy of the colliding link is
    traversed depth-first and a node whose distance lower bound exceeds the band is skipped with all its samples."""
    max_pairs = contact_state.pair_link_a.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_slot in qd.ndrange(max_pairs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(max_pairs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        if i_p >= contact_state.n_pairs[i_b]:
            continue
        i_la = contact_state.pair_link_a[i_p, i_b]
        i_lb = contact_state.pair_link_b[i_p, i_b]
        i_gb = contact_state.pair_geom_b[i_p, i_b]
        pos_a = dyn_state.links.pos[i_la, i_b]
        quat_a = dyn_state.links.quat[i_la, i_b]
        pos_g = dyn_state.geoms.pos[i_gb, i_b]
        quat_g = dyn_state.geoms.quat[i_gb, i_b]
        band = mochi_info.geoms.penalty_threshold[i_gb]
        i_node = mochi_info.links.tree_start[i_la]
        i_node_end = mochi_info.links.tree_end[i_la]
        while i_node < i_node_end:
            center = gu.qd_transform_by_trans_quat(mochi_info.samples.tree_center[i_node], pos_a, quat_a)
            center_geom = gu.qd_inv_transform_by_trans_quat(center, pos_g, quat_g)
            lower = query_collider_lower_bound(
                i_gb,
                center_geom,
                mochi_info.samples.tree_radius[i_node],
                dyn_info.geoms,
                mochi_info.geoms,
                sdf_info,
                mochi_config,
            )
            if lower > band:
                i_node = mochi_info.samples.tree_escape[i_node]
            else:
                if mochi_info.samples.tree_is_leaf[i_node] != 0:
                    i_s_start = mochi_info.samples.tree_first[i_node]
                    for i_s in range(i_s_start, i_s_start + mochi_info.samples.tree_count[i_node]):
                        func_contact_eval_sample(
                            i_p,
                            i_s,
                            i_b,
                            i_la,
                            i_lb,
                            i_gb,
                            dyn_state,
                            dyn_info,
                            sdf_info,
                            mochi_info,
                            mochi_state,
                            contact_state,
                            hit_readback,
                            rigid_config,
                            mochi_config,
                            assem_dres,
                            record,
                            errno,
                        )
                i_node = i_node + 1


@qd.kernel
def kernel_contact_eval(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    hit_readback: MochiHitReadback,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
    record: qd.template(),
    errno: qd.Tensor,
):
    func_contact_eval(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        sdf_info,
        mochi_info,
        mochi_state,
        contact_state,
        hit_readback,
        rigid_config,
        mochi_config,
        assem_dres,
        skip_ls_done,
        record,
        errno,
    )


@qd.func
def func_pairs_to_blocks(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres,
    skip_ls_done,
):
    """Expand the per-pair sums into the residual and the 6x6 Hessian blocks of the two links.

    With the point Jacobians J_a = [I, -[p - t_a]x] and J_b = -[I, -[p - t_b]x] and the per-sample matrix D = -w df/dp,
    the blocks J^T D J only need the sums of D, [r_b]x D and [r_b]x D [r_b]x, since [p - t_a]x = [r_b]x - [t_a - t_b]x.
    """
    max_pairs = contact_state.pair_link_a.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_slot in qd.ndrange(max_pairs, n_envs[None]) if qd.static(not per_env) else qd.ndrange(max_pairs, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        if i_p >= contact_state.n_pairs[i_b] or contact_state.n_hits[i_p, i_b] == 0:
            continue
        i_la = contact_state.pair_link_a[i_p, i_b]
        i_lb = contact_state.pair_link_b[i_p, i_b]
        is_dynamic_a = mochi_info.links.is_dynamic[i_la]
        is_dynamic_b = mochi_info.links.is_dynamic[i_lb]

        F = contact_state.acc_f[i_p, i_b]
        Q = contact_state.acc_q[i_p, i_b]
        c = dyn_state.links.pos[i_la, i_b] - dyn_state.links.pos[i_lb, i_b]

        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], contact_state.acc_obj[i_p, i_b])

        if qd.static(assem_res):
            if is_dynamic_a:
                torque_a = Q - c.cross(F)
                for k in qd.static(range(3)):
                    qd.atomic_add(mochi_state.links_res[i_la, i_b][k], -F[k])
                    qd.atomic_add(mochi_state.links_res[i_la, i_b][3 + k], -torque_a[k])
            if is_dynamic_b:
                for k in qd.static(range(3)):
                    qd.atomic_add(mochi_state.links_res[i_lb, i_b][k], F[k])
                    qd.atomic_add(mochi_state.links_res[i_lb, i_b][3 + k], Q[k])

        if assem_dres:
            Dbar = func_sym6_to_mat3(contact_state.acc_D[i_p, i_b])
            Sh = contact_state.acc_SD[i_p, i_b]
            Sh2 = func_sym6_to_mat3(contact_state.acc_SDS[i_p, i_b])
            ShT = Sh.transpose()
            C = skew(c)
            if is_dynamic_a:
                AA_tr = Dbar @ C + ShT
                AA_rr = Sh @ C - C @ ShT - Sh2 - C @ Dbar @ C
                for k, l in qd.static(qd.ndrange(3, 3)):
                    qd.atomic_add(mochi_state.H_diag[i_la, i_b][k, l], Dbar[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_la, i_b][k, 3 + l], AA_tr[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_la, i_b][3 + k, l], AA_tr[l, k])
                    qd.atomic_add(mochi_state.H_diag[i_la, i_b][3 + k, 3 + l], AA_rr[k, l])
            if is_dynamic_b:
                for k, l in qd.static(qd.ndrange(3, 3)):
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][k, l], Dbar[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][k, 3 + l], ShT[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][3 + k, l], Sh[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][3 + k, 3 + l], -Sh2[k, l])
            if is_dynamic_a and is_dynamic_b:
                AB_rt = -Sh + C @ Dbar
                AB_rr = Sh2 + C @ ShT
                H_off = qd.Matrix.zero(gs.qd_float, 6, 6)
                for k, l in qd.static(qd.ndrange(3, 3)):
                    H_off[k, l] = -Dbar[k, l]
                    H_off[k, 3 + l] = -ShT[k, l]
                    H_off[3 + k, l] = AB_rt[k, l]
                    H_off[3 + k, 3 + l] = AB_rr[k, l]
                mochi_state.H_off[i_p, i_b] = H_off


@qd.kernel
def kernel_pairs_to_blocks(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.i32,
    skip_ls_done: qd.i32,
):
    func_pairs_to_blocks(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        rigid_config,
        assem_obj,
        assem_res,
        assem_dres,
        skip_ls_done,
    )


# ------------------------------------------------------------------------------------
# --------------------------------------- readback -----------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_write_record(
    i_rec,
    i_b,
    entity_a,
    entity_b,
    link_a,
    link_b,
    geom_a,
    geom_b,
    verts_a,
    bary_a,
    verts_b,
    bary_b,
    pos,
    normal,
    force,
    distance,
    weight,
    out_entity_a: qd.template(),
    out_entity_b: qd.template(),
    out_link_a: qd.template(),
    out_link_b: qd.template(),
    out_geom_a: qd.template(),
    out_geom_b: qd.template(),
    out_verts_a: qd.template(),
    out_bary_a: qd.template(),
    out_verts_b: qd.template(),
    out_bary_b: qd.template(),
    out_pos: qd.template(),
    out_normal: qd.template(),
    out_force: qd.template(),
    out_distance: qd.template(),
    out_weight: qd.template(),
):
    """One readback record into the (n_envs, n_records, ...) output tensors."""
    out_entity_a[i_b, i_rec] = entity_a
    out_entity_b[i_b, i_rec] = entity_b
    out_link_a[i_b, i_rec] = link_a
    out_link_b[i_b, i_rec] = link_b
    out_geom_a[i_b, i_rec] = geom_a
    out_geom_b[i_b, i_rec] = geom_b
    for k in qd.static(range(3)):
        out_verts_a[i_b, i_rec, k] = verts_a[k]
        out_bary_a[i_b, i_rec, k] = bary_a[k]
        out_pos[i_b, i_rec, k] = pos[k]
        out_normal[i_b, i_rec, k] = normal[k]
        out_force[i_b, i_rec, k] = force[k]
    for k in qd.static(range(4)):
        out_verts_b[i_b, i_rec, k] = verts_b[k]
        out_bary_b[i_b, i_rec, k] = bary_b[k]
    out_distance[i_b, i_rec] = distance
    out_weight[i_b, i_rec] = weight


@qd.func
def func_soft_sample_verts(i_s, soft_info: MochiSoftInfo):
    """Entity-local vertices and weights of a deformable contact sample."""
    i_e = soft_info.samples_entity_idx[i_s]
    v_start = soft_info.entities_vert_start[i_e]
    tri = soft_info.samples_tri[i_s]
    verts = qd.Vector([tri[0] - v_start, tri[1] - v_start, tri[2] - v_start], dt=gs.qd_int)
    return i_e, verts, soft_info.samples_bary[i_s]


@qd.kernel
def kernel_count_contact_records(
    hit_readback: MochiHitReadback,
    soft_state: MochiSoftState,
    n_records: qd.types.ndarray(),
    rigid_config: qd.template(),
    has_soft: qd.template(),
):
    """Number of contact points recorded per environment by the last recording pass, all kinds together."""
    _B = n_records.shape[0]
    max_hits = hit_readback.hit_sample.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        n = qd.min(hit_readback.n_hits_total[i_b], max_hits)
        if qd.static(has_soft):
            n += qd.min(soft_state.n_soft_hits[i_b], soft_state.hit_sample.shape[0])
            n += qd.min(soft_state.n_sc_hits[i_b], soft_state.sc_hit_kind_a.shape[0])
            n += qd.min(soft_state.n_pc_hits[i_b], soft_state.pc_hit_kind_a.shape[0])
        n_records[i_b] = n


@qd.kernel
def kernel_gather_contact_records(
    links_entity_idx: qd.types.ndarray(),
    geoms_link_idx: qd.types.ndarray(),
    soft_entities_idx: qd.types.ndarray(),
    samples_info: MochiSamplesInfo,
    hit_readback: MochiHitReadback,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    has_soft: qd.template(),
    out_entity_a: qd.types.ndarray(),
    out_entity_b: qd.types.ndarray(),
    out_link_a: qd.types.ndarray(),
    out_link_b: qd.types.ndarray(),
    out_geom_a: qd.types.ndarray(),
    out_geom_b: qd.types.ndarray(),
    out_verts_a: qd.types.ndarray(),
    out_bary_a: qd.types.ndarray(),
    out_verts_b: qd.types.ndarray(),
    out_bary_b: qd.types.ndarray(),
    out_pos: qd.types.ndarray(),
    out_normal: qd.types.ndarray(),
    out_force: qd.types.ndarray(),
    out_distance: qd.types.ndarray(),
    out_weight: qd.types.ndarray(),
):
    """Compact the contact points recorded by the last evaluation (rigid samples on rigid colliders, then deformable
    samples on rigid colliders, samples on deformable colliders and on point-cloud colliders) into the unified
    per-environment readback records, resolving links, geoms and entities to scene indices. The outputs hold the
    largest per-environment count (see kernel_count_contact_records)."""
    max_hits = hit_readback.hit_sample.shape[0]
    _B = out_entity_a.shape[0]
    no_verts3 = qd.Vector([-1, -1, -1], dt=gs.qd_int)
    no_verts4 = qd.Vector([-1, -1, -1, -1], dt=gs.qd_int)
    zero3 = qd.Vector.zero(gs.qd_float, 3)
    zero4 = qd.Vector.zero(gs.qd_float, 4)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_h, i_b in qd.ndrange(max_hits, _B):
        n_rigid = qd.min(hit_readback.n_hits_total[i_b], max_hits)
        if i_h >= n_rigid:
            continue
        i_la = hit_readback.hit_link_a[i_h, i_b]
        i_lb = hit_readback.hit_link_b[i_h, i_b]
        func_write_record(
            i_h,
            i_b,
            links_entity_idx[i_la],
            links_entity_idx[i_lb],
            i_la,
            i_lb,
            hit_readback.hit_geom_a[i_h, i_b],
            hit_readback.hit_geom_b[i_h, i_b],
            no_verts3,
            zero3,
            no_verts4,
            zero4,
            hit_readback.hit_pos[i_h, i_b],
            hit_readback.hit_normal[i_h, i_b],
            hit_readback.hit_force[i_h, i_b],
            hit_readback.hit_distance[i_h, i_b],
            hit_readback.hit_weight[i_h, i_b],
            out_entity_a,
            out_entity_b,
            out_link_a,
            out_link_b,
            out_geom_a,
            out_geom_b,
            out_verts_a,
            out_bary_a,
            out_verts_b,
            out_bary_b,
            out_pos,
            out_normal,
            out_force,
            out_distance,
            out_weight,
        )

    if qd.static(has_soft):
        max_soft_hits = soft_state.hit_sample.shape[0]
        max_sc_hits = soft_state.sc_hit_kind_a.shape[0]
        max_pc_hits = soft_state.pc_hit_kind_a.shape[0]

        # Deformable samples on rigid colliders.
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_h, i_b in qd.ndrange(max_soft_hits, _B):
            n_rigid = qd.min(hit_readback.n_hits_total[i_b], max_hits)
            n_soft = qd.min(soft_state.n_soft_hits[i_b], max_soft_hits)
            if i_h >= n_soft:
                continue
            i_e, verts_a, bary_a = func_soft_sample_verts(soft_state.hit_sample[i_h, i_b], soft_info)
            i_gb = hit_readback.soft_hit_geom_b[i_h, i_b]
            i_lb = geoms_link_idx[i_gb]
            func_write_record(
                n_rigid + i_h,
                i_b,
                soft_entities_idx[i_e],
                links_entity_idx[i_lb],
                -1,
                i_lb,
                -1,
                i_gb,
                verts_a,
                bary_a,
                no_verts4,
                zero4,
                hit_readback.soft_hit_pos[i_h, i_b],
                hit_readback.soft_hit_normal[i_h, i_b],
                hit_readback.soft_hit_force[i_h, i_b],
                hit_readback.soft_hit_distance[i_h, i_b],
                soft_info.samples_weight[soft_state.hit_sample[i_h, i_b]],
                out_entity_a,
                out_entity_b,
                out_link_a,
                out_link_b,
                out_geom_a,
                out_geom_b,
                out_verts_a,
                out_bary_a,
                out_verts_b,
                out_bary_b,
                out_pos,
                out_normal,
                out_force,
                out_distance,
                out_weight,
            )

        # Samples of either kind on deformable (tetrahedral) colliders.
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_h, i_b in qd.ndrange(max_sc_hits, _B):
            n_rigid = qd.min(hit_readback.n_hits_total[i_b], max_hits)
            n_soft = qd.min(soft_state.n_soft_hits[i_b], max_soft_hits)
            n_sc = qd.min(soft_state.n_sc_hits[i_b], max_sc_hits)
            if i_h >= n_sc:
                continue
            i_sample = soft_state.sc_hit_sample_a[i_h, i_b]
            entity_a = -1
            link_a = -1
            geom_a = -1
            verts_a = no_verts3
            bary_a = zero3
            weight = gs.qd_float(0.0)
            if soft_state.sc_hit_kind_a[i_h, i_b] == 0:
                link_a = samples_info.link_idx[i_sample]
                geom_a = samples_info.geom_idx[i_sample]
                entity_a = links_entity_idx[link_a]
                weight = samples_info.weight[i_sample]
            else:
                i_ea, verts_a, bary_a = func_soft_sample_verts(i_sample, soft_info)
                entity_a = soft_entities_idx[i_ea]
                weight = soft_info.samples_weight[i_sample]
            i_el = soft_state.sc_hit_elem_b[i_h, i_b]
            i_eb = soft_info.elems_entity_idx[i_el]
            v_start = soft_info.entities_vert_start[i_eb]
            v = soft_info.elems_v[i_el]
            verts_b = qd.Vector([v[0] - v_start, v[1] - v_start, v[2] - v_start, v[3] - v_start], dt=gs.qd_int)
            func_write_record(
                n_rigid + n_soft + i_h,
                i_b,
                entity_a,
                soft_entities_idx[i_eb],
                link_a,
                -1,
                geom_a,
                -1,
                verts_a,
                bary_a,
                verts_b,
                soft_state.sc_hit_bary_b[i_h, i_b],
                hit_readback.sc_hit_pos[i_h, i_b],
                hit_readback.sc_hit_normal[i_h, i_b],
                hit_readback.sc_hit_force[i_h, i_b],
                hit_readback.sc_hit_distance[i_h, i_b],
                weight,
                out_entity_a,
                out_entity_b,
                out_link_a,
                out_link_b,
                out_geom_a,
                out_geom_b,
                out_verts_a,
                out_bary_a,
                out_verts_b,
                out_bary_b,
                out_pos,
                out_normal,
                out_force,
                out_distance,
                out_weight,
            )

        # Samples of either kind on point-cloud colliders (one collider vertex).
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_h, i_b in qd.ndrange(max_pc_hits, _B):
            n_rigid = qd.min(hit_readback.n_hits_total[i_b], max_hits)
            n_soft = qd.min(soft_state.n_soft_hits[i_b], max_soft_hits)
            n_sc = qd.min(soft_state.n_sc_hits[i_b], max_sc_hits)
            n_pc = qd.min(soft_state.n_pc_hits[i_b], max_pc_hits)
            if i_h >= n_pc:
                continue
            i_sample = soft_state.pc_hit_sample_a[i_h, i_b]
            entity_a = -1
            link_a = -1
            geom_a = -1
            verts_a = no_verts3
            bary_a = zero3
            weight = gs.qd_float(0.0)
            if soft_state.pc_hit_kind_a[i_h, i_b] == 0:
                link_a = samples_info.link_idx[i_sample]
                geom_a = samples_info.geom_idx[i_sample]
                entity_a = links_entity_idx[link_a]
                weight = samples_info.weight[i_sample]
            else:
                i_ea, verts_a, bary_a = func_soft_sample_verts(i_sample, soft_info)
                entity_a = soft_entities_idx[i_ea]
                weight = soft_info.samples_weight[i_sample]
            i_vb = soft_state.pc_hit_vert_b[i_h, i_b]
            i_eb = soft_info.verts_entity_idx[i_vb]
            verts_b = qd.Vector([i_vb - soft_info.entities_vert_start[i_eb], -1, -1, -1], dt=gs.qd_int)
            bary_b = qd.Vector([1.0, 0.0, 0.0, 0.0], dt=gs.qd_float)
            func_write_record(
                n_rigid + n_soft + n_sc + i_h,
                i_b,
                entity_a,
                soft_entities_idx[i_eb],
                link_a,
                -1,
                geom_a,
                -1,
                verts_a,
                bary_a,
                verts_b,
                bary_b,
                hit_readback.pc_hit_pos[i_h, i_b],
                hit_readback.pc_hit_normal[i_h, i_b],
                hit_readback.pc_hit_force[i_h, i_b],
                hit_readback.pc_hit_distance[i_h, i_b],
                weight,
                out_entity_a,
                out_entity_b,
                out_link_a,
                out_link_b,
                out_geom_a,
                out_geom_b,
                out_verts_a,
                out_bary_a,
                out_verts_b,
                out_bary_b,
                out_pos,
                out_normal,
                out_force,
                out_distance,
                out_weight,
            )
