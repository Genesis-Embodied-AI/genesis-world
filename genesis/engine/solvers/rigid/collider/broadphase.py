"""
Broad-phase collision detection functions.

This module contains AABB operations, sweep-and-prune algorithms,
and collision pair validation for the rigid body collider.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .contact import func_collider_clear_env
from .utils import func_is_geom_aabbs_overlap


@qd.func
def func_check_collision_valid(
    i_ga,
    i_gb,
    i_b,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
):
    is_valid = collider_info.collision_pair_idx[i_ga, i_gb] != -1

    if is_valid:
        i_la = dyn_info.geoms.link_idx[i_ga]
        i_lb = dyn_info.geoms.link_idx[i_gb]

        # Filter out collision pairs that are involved in dynamically registered weld equality constraints
        for i_eq in range(rigid_info.n_equalities[None], constraint_state.qd_n_equalities[i_b]):
            if dyn_info.equalities.eq_type[i_eq, i_b] == gs.EQUALITY_TYPE.WELD:
                i_leqa = dyn_info.equalities.eq_obj1id[i_eq, i_b]
                i_leqb = dyn_info.equalities.eq_obj2id[i_eq, i_b]
                if (i_leqa == i_la and i_leqb == i_lb) or (i_leqa == i_lb and i_leqb == i_la):
                    is_valid = False

        # hibernated <-> fixed links
        if qd.static(rigid_config.use_hibernation):
            I_la = [i_la, i_b] if qd.static(rigid_config.batch_links_info) else i_la
            I_lb = [i_lb, i_b] if qd.static(rigid_config.batch_links_info) else i_lb

            if (dyn_state.links.is_hibernated[i_la, i_b] and dyn_info.links.is_fixed[I_lb]) or (
                dyn_state.links.is_hibernated[i_lb, i_b] and dyn_info.links.is_fixed[I_la]
            ):
                is_valid = False

    return is_valid


@qd.func
def func_collision_clear(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    _B = collider_state.n_contacts.shape[0]

    qd.loop_config(name="collision_clear", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        func_collider_clear_env(i_b, dyn_state, collider_state, dyn_info, rigid_config)


@qd.kernel(fastcache=True)
def _func_broad_phase_sap(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """
    Sweep and Prune (SAP) for broad-phase collision detection.

    This function sorts the geometry axis-aligned bounding boxes (AABBs) along a specified axis and checks for
    potential collision pairs based on the AABB overlap.
    """
    n_geoms, _B = collider_state.active_buffer.shape
    n_links = dyn_info.links.geom_start.shape[0]

    # Clear collider state
    func_collision_clear(dyn_state, collider_state, dyn_info, rigid_config)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        axis = 0

        # Calculate the number of active geoms for this environment
        # (for heterogeneous entities, different envs may have different geoms)
        env_n_geoms = 0
        for i_l in range(n_links):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            env_n_geoms = env_n_geoms + dyn_info.links.geom_end[I_l] - dyn_info.links.geom_start[I_l]

        # copy updated geom aabbs to buffer for sorting
        if collider_state.first_time[i_b]:
            i_buffer = 0
            for i_l in range(n_links):
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                for i_g in range(dyn_info.links.geom_start[I_l], dyn_info.links.geom_end[I_l]):
                    collider_state.sort_buffer.value[2 * i_buffer, i_b] = dyn_state.geoms.aabb_min[i_g, i_b][axis]
                    collider_state.sort_buffer.i_g[2 * i_buffer, i_b] = i_g
                    collider_state.sort_buffer.is_max[2 * i_buffer, i_b] = False

                    collider_state.sort_buffer.value[2 * i_buffer + 1, i_b] = dyn_state.geoms.aabb_max[i_g, i_b][axis]
                    collider_state.sort_buffer.i_g[2 * i_buffer + 1, i_b] = i_g
                    collider_state.sort_buffer.is_max[2 * i_buffer + 1, i_b] = True

                    dyn_state.geoms.min_buffer_idx[i_buffer, i_b] = 2 * i_g
                    dyn_state.geoms.max_buffer_idx[i_buffer, i_b] = 2 * i_g + 1
                    i_buffer = i_buffer + 1

            collider_state.first_time[i_b] = False

        else:
            # Warm start: refresh the sort-buffer extents from the current AABBs. Hibernated geoms do not move, so
            # their entries stay valid and are skipped; refreshing the awake ones here (rather than relying solely
            # on the hibernation decision kernel, which runs at substep end) ensures a body woken between substeps -
            # e.g. by a state setter - is seen by the very next broad phase instead of lagging a step.
            if qd.static(not rigid_config.use_hibernation):
                for i in range(env_n_geoms * 2):
                    if collider_state.sort_buffer.is_max[i, i_b]:
                        collider_state.sort_buffer.value[i, i_b] = dyn_state.geoms.aabb_max[
                            collider_state.sort_buffer.i_g[i, i_b], i_b
                        ][axis]
                    else:
                        collider_state.sort_buffer.value[i, i_b] = dyn_state.geoms.aabb_min[
                            collider_state.sort_buffer.i_g[i, i_b], i_b
                        ][axis]
            else:
                for i in range(env_n_geoms * 2):
                    i_g = collider_state.sort_buffer.i_g[i, i_b]
                    if not dyn_state.geoms.is_hibernated[i_g, i_b]:
                        if collider_state.sort_buffer.is_max[i, i_b]:
                            collider_state.sort_buffer.value[i, i_b] = dyn_state.geoms.aabb_max[i_g, i_b][axis]
                        else:
                            collider_state.sort_buffer.value[i, i_b] = dyn_state.geoms.aabb_min[i_g, i_b][axis]

        # insertion sort, which has complexity near O(n) for nearly sorted array
        for i in range(1, 2 * env_n_geoms):
            key_value = collider_state.sort_buffer.value[i, i_b]
            key_is_max = collider_state.sort_buffer.is_max[i, i_b]
            key_i_g = collider_state.sort_buffer.i_g[i, i_b]

            j = i - 1
            while j >= 0 and key_value < collider_state.sort_buffer.value[j, i_b]:
                collider_state.sort_buffer.value[j + 1, i_b] = collider_state.sort_buffer.value[j, i_b]
                collider_state.sort_buffer.is_max[j + 1, i_b] = collider_state.sort_buffer.is_max[j, i_b]
                collider_state.sort_buffer.i_g[j + 1, i_b] = collider_state.sort_buffer.i_g[j, i_b]

                if qd.static(rigid_config.use_hibernation):
                    if collider_state.sort_buffer.is_max[j, i_b]:
                        dyn_state.geoms.max_buffer_idx[collider_state.sort_buffer.i_g[j, i_b], i_b] = j + 1
                    else:
                        dyn_state.geoms.min_buffer_idx[collider_state.sort_buffer.i_g[j, i_b], i_b] = j + 1

                j -= 1
            collider_state.sort_buffer.value[j + 1, i_b] = key_value
            collider_state.sort_buffer.is_max[j + 1, i_b] = key_is_max
            collider_state.sort_buffer.i_g[j + 1, i_b] = key_i_g

            if qd.static(rigid_config.use_hibernation):
                if key_is_max:
                    dyn_state.geoms.max_buffer_idx[key_i_g, i_b] = j + 1
                else:
                    dyn_state.geoms.min_buffer_idx[key_i_g, i_b] = j + 1

        # sweep over the sorted AABBs to find potential collision pairs
        n_broad = 0
        if qd.static(not rigid_config.use_hibernation):
            n_active = 0
            for i in range(2 * env_n_geoms):
                if not collider_state.sort_buffer.is_max[i, i_b]:
                    for j in range(n_active):
                        i_ga = collider_state.active_buffer[j, i_b]
                        i_gb = collider_state.sort_buffer.i_g[i, i_b]
                        if i_ga > i_gb:
                            i_ga, i_gb = i_gb, i_ga

                        if not func_check_collision_valid(
                            i_ga,
                            i_gb,
                            i_b,
                            dyn_state,
                            constraint_state,
                            dyn_info,
                            rigid_info,
                            collider_info,
                            rigid_config,
                        ):
                            continue

                        if not func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, dyn_state):
                            # Clear collision normal cache if not in contact
                            if qd.static(not rigid_config.enable_mujoco_compatibility):
                                i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                                collider_state.contact_cache.penetration[i_pair, i_b] = 0.0
                            continue

                        if n_broad == collider_info.max_collision_pairs_broad[None]:
                            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
                            break
                        collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                        collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                        n_broad = n_broad + 1

                    collider_state.active_buffer[n_active, i_b] = collider_state.sort_buffer.i_g[i, i_b]
                    n_active = n_active + 1
                else:
                    i_g_to_remove = collider_state.sort_buffer.i_g[i, i_b]
                    for j in range(n_active):
                        if collider_state.active_buffer[j, i_b] == i_g_to_remove:
                            if j < n_active - 1:
                                for k in range(j, n_active - 1):
                                    collider_state.active_buffer[k, i_b] = collider_state.active_buffer[k + 1, i_b]
                            n_active = n_active - 1
                            break
        else:
            if rigid_info.n_awake_dofs[i_b] > 0:
                n_active_awake = 0
                n_active_hib = 0
                for i in range(2 * env_n_geoms):
                    is_incoming_geom_hibernated = dyn_state.geoms.is_hibernated[
                        collider_state.sort_buffer.i_g[i, i_b], i_b
                    ]

                    if not collider_state.sort_buffer.is_max[i, i_b]:
                        # both awake and hibernated geom check with active awake geoms
                        for j in range(n_active_awake):
                            i_ga = collider_state.active_buffer_awake[j, i_b]
                            i_gb = collider_state.sort_buffer.i_g[i, i_b]
                            if i_ga > i_gb:
                                i_ga, i_gb = i_gb, i_ga

                            if not func_check_collision_valid(
                                i_ga,
                                i_gb,
                                i_b,
                                dyn_state,
                                constraint_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                rigid_config,
                            ):
                                continue

                            if not func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, dyn_state):
                                # Clear collision normal cache if not in contact
                                if qd.static(not rigid_config.enable_mujoco_compatibility):
                                    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                                    collider_state.contact_cache.penetration[i_pair, i_b] = 0.0
                                continue

                            collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                            collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                            n_broad = n_broad + 1

                        # if incoming geom is awake, also need to check with hibernated geoms
                        if not is_incoming_geom_hibernated:
                            for j in range(n_active_hib):
                                i_ga = collider_state.active_buffer_hib[j, i_b]
                                i_gb = collider_state.sort_buffer.i_g[i, i_b]
                                if i_ga > i_gb:
                                    i_ga, i_gb = i_gb, i_ga

                                if not func_check_collision_valid(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    dyn_state,
                                    constraint_state,
                                    dyn_info,
                                    rigid_info,
                                    collider_info,
                                    rigid_config,
                                ):
                                    continue

                                if not func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, dyn_state):
                                    # Clear collision normal cache if not in contact
                                    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                                    collider_state.contact_cache.penetration[i_pair, i_b] = 0.0
                                    continue

                                collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
                                collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
                                n_broad = n_broad + 1

                        if is_incoming_geom_hibernated:
                            collider_state.active_buffer_hib[n_active_hib, i_b] = collider_state.sort_buffer.i_g[i, i_b]
                            n_active_hib = n_active_hib + 1
                        else:
                            collider_state.active_buffer_awake[n_active_awake, i_b] = collider_state.sort_buffer.i_g[
                                i, i_b
                            ]
                            n_active_awake = n_active_awake + 1
                    else:
                        i_g_to_remove = collider_state.sort_buffer.i_g[i, i_b]
                        if is_incoming_geom_hibernated:
                            for j in range(n_active_hib):
                                if collider_state.active_buffer_hib[j, i_b] == i_g_to_remove:
                                    if j < n_active_hib - 1:
                                        for k in range(j, n_active_hib - 1):
                                            collider_state.active_buffer_hib[k, i_b] = collider_state.active_buffer_hib[
                                                k + 1, i_b
                                            ]
                                    n_active_hib = n_active_hib - 1
                                    break
                        else:
                            for j in range(n_active_awake):
                                if collider_state.active_buffer_awake[j, i_b] == i_g_to_remove:
                                    if j < n_active_awake - 1:
                                        for k in range(j, n_active_awake - 1):
                                            collider_state.active_buffer_awake[k, i_b] = (
                                                collider_state.active_buffer_awake[k + 1, i_b]
                                            )
                                    n_active_awake = n_active_awake - 1
                                    break
        collider_state.n_broad_pairs[i_b] = n_broad


@qd.kernel(fastcache=True)
def _func_broad_phase_all_vs_all(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """
    All-vs-all broad-phase collision detection.

    Iterates over pre-filtered valid geom pairs in parallel across pairs and batches, checking 3D AABB overlap.
    Passing pairs are appended to the output buffer via atomic add.
    """

    func_collision_clear(dyn_state, collider_state, dyn_info, rigid_config)

    _B = collider_state.n_contacts.shape[0]
    qd.loop_config(name="init_broad_pairs", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        collider_state.n_broad_pairs[i_b] = 0

    n_valid_pairs = collider_info.n_valid_pairs[None]
    qd.loop_config(name="traverse_valid")
    for i_vp, i_b in qd.ndrange(n_valid_pairs, _B):
        pair = collider_info.valid_collision_pairs[i_vp]
        i_ga = pair[0]
        i_gb = pair[1]

        if not func_check_collision_valid(
            i_ga, i_gb, i_b, dyn_state, constraint_state, dyn_info, rigid_info, collider_info, rigid_config
        ):
            continue

        if not func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, dyn_state):
            if qd.static(not rigid_config.enable_mujoco_compatibility):
                i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
                collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                collider_state.contact_cache.penetration[i_pair, i_b] = 0.0
            continue

        n_broad = qd.atomic_add(collider_state.n_broad_pairs[i_b], 1)
        if n_broad < collider_info.max_collision_pairs_broad[None]:
            collider_state.broad_collision_pairs[n_broad, i_b][0] = i_ga
            collider_state.broad_collision_pairs[n_broad, i_b][1] = i_gb
        else:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS


def func_broad_phase(
    dyn_state, dyn_info, rigid_info, rigid_config, constraint_state, collider_state, collider_info, errno
):
    """Dispatch to the appropriate broad-phase kernel based on config."""
    if rigid_config.broadphase_traversal == gs.broadphase_traversal.ALL_VS_ALL:
        _func_broad_phase_all_vs_all(
            dyn_state, collider_state, constraint_state, dyn_info, rigid_info, collider_info, rigid_config, errno
        )
    else:
        _func_broad_phase_sap(
            dyn_state, collider_state, constraint_state, dyn_info, rigid_info, collider_info, rigid_config, errno
        )
