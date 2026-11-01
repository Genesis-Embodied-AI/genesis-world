"""
Broad-phase collision detection functions.

The broad phase runs at two levels. The geoms of a link keep their relative pose, so the bounding boxes of the links are
swept first, either by sweep-and-prune or all-vs-all over the valid link pairs, and the geom pairs are then tested only
under the link pairs whose boxes overlap, spread over the geoms of the larger link.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from .contact import func_collider_clear_env, func_is_link_pair_hibernated
from .utils import func_is_geom_boxes_overlap


@qd.func
def func_check_links_collision_valid(
    i_la: int,
    i_lb: int,
    i_b: int,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Check whether the geoms of two links of a valid link pair may collide in an environment at this step.

    The link pair is dropped when a dynamic weld or screw constraint holds the two links, whose geometries may overlap
    by design (a nut on its bolt), and with hibernation when both links stand still.
    """
    is_valid = True

    for i_eq in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
        eq_type = dyn_info.equalities.eq_type[i_eq, i_b]
        if eq_type == gs.EQUALITY_TYPE.WELD or eq_type == gs.EQUALITY_TYPE.SCREW:
            i_leqa = dyn_info.equalities.eq_obj1id[i_eq, i_b]
            i_leqb = dyn_info.equalities.eq_obj2id[i_eq, i_b]
            if (i_leqa == i_la and i_leqb == i_lb) or (i_leqa == i_lb and i_leqb == i_la):
                is_valid = False

    # A pair standing still keeps its contacts from the last awake solve (see func_collider_clear_env), and an awake
    # link striking either one is the only motion that reaches them.
    if qd.static(rigid_config.use_hibernation):
        if rigid_info.n_awake_dofs[i_b] < dyn_state.dofs.is_hibernated.shape[0]:
            if func_is_link_pair_hibernated(i_la, i_lb, i_b, dyn_state, dyn_info, rigid_config):
                is_valid = False

    return is_valid


@qd.func
def func_collision_clear(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """Clear the contacts and the candidate link pairs of every environment, and fit the bounding box of every link.

    The contact cache of the candidate pairs of the last broad phase whose bounding boxes no longer overlap is cleared
    too. The narrow phase writes the cache of the candidate pairs alone, so a pair holding a cached normal was a
    candidate of the last broad phase. Clearing those that left the overlap makes a later re-contact cold, at a cost
    scaling with the candidate count instead of the valid pair count. Every chunk of an environment reads its candidate
    geom pairs, so their count is reset by the link sweep that follows.
    """
    _B = collider_state.n_contacts.shape[0]
    n_links = dyn_info.links.geom_start.shape[0]
    max_broad_pairs = collider_state.broad_collision_pairs.shape[0]

    # The pairs and links of an env are split over chunks so that the GPU cores are kept busy whatever the env count,
    # as in func_narrowphase_contact0, the first chunk clearing the contacts of the env
    n_chunks = qd.max((collider_static_config.gpu_cores + _B - 1) // _B, 1)
    qd.loop_config(name="collision_clear", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_b_chunk in range(_B * n_chunks):
        i_b = i_b_chunk // n_chunks
        i_chunk = i_b_chunk % n_chunks
        if i_chunk == 0:
            func_collider_clear_env(i_b, dyn_state, collider_state, dyn_info, rigid_info, rigid_config)
            collider_state.n_broad_link_pairs[i_b] = 0
        if qd.static(
            collider_static_config.has_non_box_plane_convex_convex and not rigid_config.enable_mujoco_compatibility
        ):
            n_pairs = qd.min(collider_state.n_broad_pairs[i_b], max_broad_pairs)
            for i_pair_ in range(i_chunk * n_pairs // n_chunks, (i_chunk + 1) * n_pairs // n_chunks):
                i_ga = collider_state.broad_collision_pairs[i_pair_, i_b][0]
                i_gb = collider_state.broad_collision_pairs[i_pair_, i_b][1]
                if not func_is_geom_boxes_overlap(i_ga, i_gb, i_b, geoms_init_AABB, dyn_state, dyn_info, rigid_info):
                    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    collider_state.contact_cache.penetration[i_pair, i_b] = 0.0

        # The box of a link is the box of the link frame moved with the link, which bounds its geoms since their pose
        # in the link is fixed. A link carrying no geom in an environment gets an empty box, which overlaps nothing.
        for i_l in range(i_chunk * n_links // n_chunks, (i_chunk + 1) * n_links // n_chunks):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            lower = qd.Vector([qd.math.inf, qd.math.inf, qd.math.inf], dt=gs.qd_float)
            upper = qd.Vector([-qd.math.inf, -qd.math.inf, -qd.math.inf], dt=gs.qd_float)
            if dyn_info.links.geom_start[I_l] < dyn_info.links.geom_end[I_l]:
                # The bounding box of a rotated box spans its moved center plus or minus its half sizes rotated by |R|
                l_quat = dyn_state.links.quat[i_l, i_b]
                aabb_center = 0.5 * (collider_info.links_init_aabb_max[i_l] + collider_info.links_init_aabb_min[i_l])
                aabb_half_size = 0.5 * (collider_info.links_init_aabb_max[i_l] - collider_info.links_init_aabb_min[i_l])
                center = gu.qd_transform_by_trans_quat(aabb_center, dyn_state.links.pos[i_l, i_b], l_quat)
                half_size = qd.abs(gu.qd_quat_to_R(l_quat, rigid_info.EPS[None])) @ aabb_half_size
                lower = center - half_size
                upper = center + half_size
            collider_state.links_aabb_min[i_l, i_b] = lower
            collider_state.links_aabb_max[i_l, i_b] = upper


@qd.func
def func_add_broad_link_pair(
    i_la: int,
    i_lb: int,
    i_b: int,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
):
    """Record a link pair as a candidate of an environment when it is valid and the boxes of its links overlap."""
    i_lp = collider_info.link_pair_idx[i_la, i_lb]
    if i_lp != -1:
        is_link_aabbs_overlapping = not (
            (collider_state.links_aabb_max[i_la, i_b] <= collider_state.links_aabb_min[i_lb, i_b]).any()
            or (collider_state.links_aabb_min[i_la, i_b] >= collider_state.links_aabb_max[i_lb, i_b]).any()
        )
        if is_link_aabbs_overlapping:
            if func_check_links_collision_valid(
                i_la, i_lb, i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config
            ):
                # The list holds every valid link pair, so it cannot overflow
                i_lp_ = qd.atomic_add(collider_state.n_broad_link_pairs[i_b], 1)
                collider_state.broad_link_pairs[i_lp_, i_b] = i_lp


@qd.func
def _func_broad_phase_links_sap(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
):
    """Find the link pairs of every environment whose boxes overlap by sweep-and-prune (SAP) along the x axis.

    The link boxes are kept sorted from one call to the next, so the insertion sort runs in near linear time.
    """
    n_links, _B = collider_state.active_buffer.shape

    qd.loop_config(name="sweep_links", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # See func_collision_clear
        collider_state.n_broad_pairs[i_b] = 0
        if collider_state.first_time[i_b]:
            for i_l in range(n_links):
                collider_state.sort_buffer.i_l[2 * i_l, i_b] = i_l
                collider_state.sort_buffer.is_max[2 * i_l, i_b] = False
                collider_state.sort_buffer.i_l[2 * i_l + 1, i_b] = i_l
                collider_state.sort_buffer.is_max[2 * i_l + 1, i_b] = True
            collider_state.first_time[i_b] = False

        for i in range(2 * n_links):
            i_l = collider_state.sort_buffer.i_l[i, i_b]
            if collider_state.sort_buffer.is_max[i, i_b]:
                collider_state.sort_buffer.value[i, i_b] = collider_state.links_aabb_max[i_l, i_b][0]
            else:
                collider_state.sort_buffer.value[i, i_b] = collider_state.links_aabb_min[i_l, i_b][0]

        # Insertion sort, whose complexity is near linear on the nearly sorted boxes of the last call
        for i in range(1, 2 * n_links):
            key_value = collider_state.sort_buffer.value[i, i_b]
            key_is_max = collider_state.sort_buffer.is_max[i, i_b]
            key_i_l = collider_state.sort_buffer.i_l[i, i_b]
            j = i
            for k in range(i):
                if key_value >= collider_state.sort_buffer.value[i - 1 - k, i_b]:
                    break
                collider_state.sort_buffer.value[i - k, i_b] = collider_state.sort_buffer.value[i - 1 - k, i_b]
                collider_state.sort_buffer.is_max[i - k, i_b] = collider_state.sort_buffer.is_max[i - 1 - k, i_b]
                collider_state.sort_buffer.i_l[i - k, i_b] = collider_state.sort_buffer.i_l[i - 1 - k, i_b]
                j = i - 1 - k
            collider_state.sort_buffer.value[j, i_b] = key_value
            collider_state.sort_buffer.is_max[j, i_b] = key_is_max
            collider_state.sort_buffer.i_l[j, i_b] = key_i_l

        # Sweep the sorted boxes, an opening box being tested against every box still open
        n_active = 0
        for i in range(2 * n_links):
            i_l = collider_state.sort_buffer.i_l[i, i_b]
            if not collider_state.sort_buffer.is_max[i, i_b]:
                for j in range(n_active):
                    i_l_active = collider_state.active_buffer[j, i_b]
                    func_add_broad_link_pair(
                        i_l,
                        i_l_active,
                        i_b,
                        dyn_state,
                        collider_state,
                        constraint_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                    )
                collider_state.active_buffer[n_active, i_b] = i_l
                n_active = n_active + 1
            else:
                # The empty box of a link carrying no geom closes before it opens, so it is never found open
                is_removed = False
                for j in range(n_active):
                    if is_removed:
                        collider_state.active_buffer[j - 1, i_b] = collider_state.active_buffer[j, i_b]
                    elif collider_state.active_buffer[j, i_b] == i_l:
                        is_removed = True
                if is_removed:
                    n_active = n_active - 1


@qd.func
def _func_broad_phase_links_all_vs_all(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
):
    """Find the link pairs of every environment whose boxes overlap by testing every valid link pair in parallel."""
    _B = collider_state.n_contacts.shape[0]
    n_valid_link_pairs = collider_info.n_valid_link_pairs[None]

    qd.loop_config(name="traverse_valid_links", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_lp, i_b in qd.ndrange(n_valid_link_pairs, _B):
        # See func_collision_clear. No geom pair is ever a candidate when no link pair is valid.
        if i_lp == 0:
            collider_state.n_broad_pairs[i_b] = 0
        link_pair = collider_info.valid_link_pairs[i_lp]
        func_add_broad_link_pair(
            link_pair[0],
            link_pair[1],
            i_b,
            dyn_state,
            collider_state,
            constraint_state,
            dyn_info,
            rigid_info,
            collider_info,
            rigid_config,
        )


@qd.func
def func_broad_phase_geoms(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """Find the geom pairs whose boxes overlap under the candidate link pairs of every environment.

    Each geom of the outer link of a candidate pair, the link holding the most geoms, is tested against the box of the
    inner link first, then against each geom of the inner link, so the work spreads over the geoms of the larger link.
    """
    n_geoms = dyn_state.geoms.aabb_min.shape[0]
    _B = collider_state.n_contacts.shape[0]

    qd.loop_config(name="traverse_link_pairs_geoms", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        i_l = dyn_info.geoms.link_idx[i_g]
        for i_lp_ in range(collider_state.n_broad_link_pairs[i_b]):
            i_lp = collider_state.broad_link_pairs[i_lp_, i_b]
            link_pair = collider_info.valid_link_pairs[i_lp]
            if link_pair[0] == i_l:
                i_l_inner = link_pair[1]
                is_overlapping_inner = not (
                    (dyn_state.geoms.aabb_max[i_g, i_b] <= collider_state.links_aabb_min[i_l_inner, i_b]).any()
                    or (dyn_state.geoms.aabb_min[i_g, i_b] >= collider_state.links_aabb_max[i_l_inner, i_b]).any()
                )
                if is_overlapping_inner:
                    I_l_inner = [i_l_inner, i_b] if qd.static(rigid_config.batch_links_info) else i_l_inner
                    for i_g_inner in range(dyn_info.links.geom_start[I_l_inner], dyn_info.links.geom_end[I_l_inner]):
                        i_ga = qd.min(i_g, i_g_inner)
                        i_gb = qd.max(i_g, i_g_inner)
                        if collider_info.collision_pair_idx[i_ga, i_gb] != -1:
                            # A pair is stored by increasing geom type, then by index, the order every narrow phase
                            # detection reads it in
                            if dyn_info.geoms.type[i_ga] > dyn_info.geoms.type[i_gb]:
                                i_ga, i_gb = i_gb, i_ga
                            if func_is_geom_boxes_overlap(
                                i_ga, i_gb, i_b, geoms_init_AABB, dyn_state, dyn_info, rigid_info
                            ):
                                n_broad = qd.atomic_add(collider_state.n_broad_pairs[i_b], 1)
                                if n_broad < collider_info.max_collision_pairs_broad[None]:
                                    collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                                    collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                                else:
                                    errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS


@qd.func
def func_broad_phase(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    """Run the broad phase, sweeping the links with the traversal the rigid config selects then testing their geoms."""
    func_collision_clear(
        geoms_init_AABB,
        dyn_state,
        collider_state,
        dyn_info,
        rigid_info,
        collider_info,
        rigid_config,
        collider_static_config,
    )
    if qd.static(rigid_config.broadphase_traversal == gs.broadphase_traversal.ALL_VS_ALL):
        _func_broad_phase_links_all_vs_all(
            dyn_state, collider_state, constraint_state, dyn_info, rigid_info, collider_info, rigid_config
        )
    else:
        _func_broad_phase_links_sap(
            dyn_state, collider_state, constraint_state, dyn_info, rigid_info, collider_info, rigid_config
        )
    func_broad_phase_geoms(
        geoms_init_AABB, dyn_state, collider_state, dyn_info, rigid_info, collider_info, rigid_config, errno
    )
