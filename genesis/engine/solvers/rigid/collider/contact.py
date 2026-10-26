"""
Contact management and utility functions for the rigid body collider.

This module contains functions for adding contacts, computing tolerances,
and managing contact data including reset/clear operations.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.func
def func_refine_smooth_contact_pos(
    geom_type: int,
    geom_data: qd.types.vector(7),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    normal: qd.types.vector(3),
    penetration: float,
    ccd_contact_pos: qd.types.vector(3),
):
    """
    Reconstruct the contact position analytically from the smooth side of the contact.

    MPR/GJK leave a position-dependent bias in the reported contact position that, on static contacts against
    rotationally-symmetric geometry, becomes torque on the smooth body and drives a persistent tangential drift (the
    lever arm becomes non-zero on what should be a face-aligned contact). For smooth primitives we have a closed-form
    surface point given the CCD-reported normal, so we can replace the biased contact position with the exact midpoint
    between that surface point and the inferred polytope-side surface. The result has the lever arm parallel to the
    contact normal, so the constraint force creates no spurious torque.

    Conventions: normal points from geom B to geom A (geom A is the one being refined). The refined contact position
    is the midpoint between A's surface (in the -normal direction from A's center) and the implicit B surface (offset
    by penetration along normal). Idempotent on the analytical paths (sphere-box, sphere-capsule, capsule-capsule)
    since those use the same closed-form expression.
    """
    refined = ccd_contact_pos
    if geom_type == gs.GEOM_TYPE.SPHERE:
        radius = geom_data[0]
        refined = geom_pos - (radius - 0.5 * penetration) * normal
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        # Surface point on ellipsoid in direction -normal, in local frame, is at p = -(a^2 n_x, b^2 n_y, c^2 n_z) /
        # sqrt(a^2 n_x^2 + b^2 n_y^2 + c^2 n_z^2). This comes from the Lagrangian "closest point in direction d" with
        # f(p) = (px/a)^2 + ... - 1 = 0.
        a = geom_data[0]
        b = geom_data[1]
        c = geom_data[2]
        n_local = gu.qd_inv_transform_by_quat(normal, geom_quat)
        denom = qd.sqrt(
            a * a * n_local[0] * n_local[0] + b * b * n_local[1] * n_local[1] + c * c * n_local[2] * n_local[2]
        )
        p_local = qd.Vector(
            [-a * a * n_local[0] / denom, -b * b * n_local[1] / denom, -c * c * n_local[2] / denom], dt=gs.qd_float
        )
        surface_pt = gu.qd_transform_by_trans_quat(p_local, geom_pos, geom_quat)
        refined = surface_pt + 0.5 * penetration * normal
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        # Capsule axis is along local +z. Project ccd_contact_pos onto the axis (clamped to the segment), then offset by
        # radius along -normal. The clamp lets cap contacts degenerate to the sphere case automatically. Barrel contacts
        # inherit the axial coordinate from ccd_contact_pos, which is only as good as the CCD's axial estimate.
        radius = geom_data[0]
        half_length = 0.5 * geom_data[1]
        axis_dir = gu.qd_transform_by_quat_fast(qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float), geom_quat)
        t_axial = (ccd_contact_pos - geom_pos).dot(axis_dir)
        t_clamped = qd.math.clamp(t_axial, -half_length, half_length)
        axis_point = geom_pos + t_clamped * axis_dir
        refined = axis_point - (radius - 0.5 * penetration) * normal
    elif geom_type == gs.GEOM_TYPE.CYLINDER:
        # Cylinder axis is along local +z. Barrel vs cap is decided from the normal, not the axial coordinate: a barrel
        # (or barrel-edge) contact has a radial normal perpendicular to the axis, while a flat-cap contact has an axial
        # normal. The axial coordinate alone is ambiguous for a side-resting cylinder, whose end contacts sit exactly at
        # the rim (|t_axial| == half_length) yet are genuine barrel contacts that must be snapped. A barrel contact is
        # identical to the capsule barrel: project onto the axis (clamped to the barrel extent so a rim contact lands at
        # the cap plane) and offset by the radius along -normal, removing the CCD's radial position bias. A cap contact
        # is on a flat end face with no curvature to refine, so the CCD position is kept.
        radius = geom_data[0]
        half_length = 0.5 * geom_data[1]
        axis_dir = gu.qd_transform_by_quat_fast(qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float), geom_quat)
        if qd.abs(normal.dot(axis_dir)) < 0.5:
            t_axial = (ccd_contact_pos - geom_pos).dot(axis_dir)
            t_clamped = qd.math.clamp(t_axial, -half_length, half_length)
            axis_point = geom_pos + t_clamped * axis_dir
            refined = axis_point - (radius - 0.5 * penetration) * normal
    return refined


@qd.func
def func_apply_smooth_refinement(
    i_ga: int,
    i_gb: int,
    normal: qd.types.vector(3),
    penetration: float,
    contact_pos: qd.types.vector(3),
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """
    Reconstruct the contact position analytically from the smooth side when one of the geoms is a smooth primitive.

    Idempotent on analytical contact paths; on MPR/GJK paths it removes the position-dependent bias that drives
    spurious torque and drift on static smooth-vs-polytope contacts. The pose inputs (ga_*/gb_*) must be in the same
    frame as contact_pos and normal: the detection pose for a directly-added contact, or the unperturbed pose for a
    multi-contact perturbed contact, which is refined only after the perturbation is reverted so the result lands in
    the canonical frame the constraint solver stores.
    """
    if qd.static(not rigid_config.enable_mujoco_compatibility):
        # Geom pairs are sorted by ascending type, so smooth primitives (SPHERE/ELLIPSOID/CAPSULE) always sit on the
        # A side when paired with a polytope (BOX/MESH/TERRAIN/PLANE). Smooth-vs-smooth pairs go through analytical
        # fast paths and never reach this helper, so at most one side ever needs refinement.
        type_a = dyn_info.geoms.type[i_ga]
        type_b = dyn_info.geoms.type[i_gb]
        if (
            type_a == gs.GEOM_TYPE.SPHERE
            or type_a == gs.GEOM_TYPE.ELLIPSOID
            or type_a == gs.GEOM_TYPE.CAPSULE
            or type_a == gs.GEOM_TYPE.CYLINDER
        ):
            contact_pos = func_refine_smooth_contact_pos(
                type_a, dyn_info.geoms.data[i_ga], ga_pos, ga_quat, normal, penetration, contact_pos
            )
        elif (
            type_b == gs.GEOM_TYPE.SPHERE
            or type_b == gs.GEOM_TYPE.ELLIPSOID
            or type_b == gs.GEOM_TYPE.CAPSULE
            or type_b == gs.GEOM_TYPE.CYLINDER
        ):
            contact_pos = func_refine_smooth_contact_pos(
                type_b, dyn_info.geoms.data[i_gb], gb_pos, gb_quat, -normal, penetration, contact_pos
            )
    return contact_pos


@qd.func
def rotaxis(i0: int, i1: int, i2: int, vecin: qd.types.vector(3), f0: int, f1: int, f2: int):
    vecres = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
    vecres[0] = vecin[i0] * f0
    vecres[1] = vecin[i1] * f1
    vecres[2] = vecin[i2] * f2
    return vecres


@qd.func
def rotmatx(i0: int, i1: int, i2: int, matin: qd.types.matrix(3, 3), f0: int, f1: int, f2: int):
    matres = qd.Matrix.zero(gs.qd_float, 3, 3)
    matres[0, :] = matin[i0, :] * f0
    matres[1, :] = matin[i1, :] * f1
    matres[2, :] = matin[i2, :] * f2
    return matres


@qd.func
def func_collider_reset_env(
    i_b: int,
    collider_state: array_class.ColliderState,
    collider_static_config: qd.template(),
    cache_only: qd.template(),
):
    """Drop what the collision detection of env i_b carries over from the previous steps.

    The cache of contact normals and penetrations warm-starts the next detection. Unless 'cache_only', the next
    detection also runs as the first one.
    """
    if qd.static(not cache_only):
        collider_state.first_time[i_b] = True

    # The contact cache is only held for the convex-convex pairs (see get_contact_cache in array_class.py)
    if qd.static(collider_static_config.has_non_box_plane_convex_convex):
        for i_pair in range(collider_state.contact_cache.normal.shape[0]):
            collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
            collider_state.contact_cache.penetration[i_pair, i_b] = 0.0


@qd.kernel(fastcache=True)
def collider_kernel_reset(
    envs_idx: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    cache_only: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        func_collider_reset_env(i_b, collider_state, collider_static_config, cache_only=cache_only)


@qd.func
def func_add_contacts_force_to_links(
    i_b: int, dyn_state: array_class.DynState, collider_state: array_class.ColliderState
):
    """Add the force of every contact of env i_b to the contact force of the two links it pairs.

    The forces are summed in contact order, which keeps the rounding of the sum fixed.
    """
    for i_c in range(collider_state.n_contacts[i_b]):
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        i_la = collider_state.contact_data.link_a[i_col, i_b]
        i_lb = collider_state.contact_data.link_b[i_col, i_b]
        force = collider_state.contact_data.force[i_col, i_b]
        dyn_state.links.contact_force[i_la, i_b] = dyn_state.links.contact_force[i_la, i_b] - force
        dyn_state.links.contact_force[i_lb, i_b] = dyn_state.links.contact_force[i_lb, i_b] + force


@qd.func
def func_is_link_pair_hibernated(
    i_la: int,
    i_lb: int,
    i_b: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Check whether two links stand still in an environment: one sleeps and the other sleeps or is fixed.

    The broadphase skips such a pair and the clear pass keeps its contacts from the last solve it took part in.
    """
    I_la = [i_la, i_b] if qd.static(rigid_config.batch_links_info) else i_la
    I_lb = [i_lb, i_b] if qd.static(rigid_config.batch_links_info) else i_lb
    is_a_hibernated = dyn_state.links.is_hibernated[i_la, i_b]
    is_b_hibernated = dyn_state.links.is_hibernated[i_lb, i_b]
    return (is_a_hibernated and (is_b_hibernated or dyn_info.links.is_fixed[I_lb])) or (
        is_b_hibernated and dyn_info.links.is_fixed[I_la]
    )


@qd.func
def func_collider_clear_env(
    i_b: int,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    if qd.static(rigid_config.use_hibernation):
        # Advect the contacts of the sleepers: a pair standing still (func_is_link_pair_hibernated) stays where it is,
        # so its contact is kept at the front of the buffer with the force of the last solve it took part in (see
        # n_contacts_hibernated in array_class.py for what reads the kept range). The contacts are first flagged on
        # their raw slot, in the sort key the narrowphase rewrites before reading it, then compacted in raw order: every
        # slot written to was already read, so no kept contact is overwritten. The kept range then lists them in their
        # last logical order (see func_sort_contacts), the raw order following the narrowphase's slot allocation, which
        # the GPU leaves to atomics. An env with no sleeper keeps none (see n_awake_dofs in array_class.py).
        n_hib = 0
        if rigid_info.n_awake_dofs[i_b] < dyn_state.dofs.is_hibernated.shape[0]:
            n_raw = 0
            for i_c_ in range(collider_state.n_contacts[i_b]):
                n_raw = qd.max(n_raw, collider_state.contact_sort_idx[i_c_, i_b] + 1)
            for i_c in range(n_raw):
                collider_state.contact_sort_key[i_c, i_b] = 0.0
            for i_c_ in range(collider_state.n_contacts[i_b]):
                i_c = collider_state.contact_sort_idx[i_c_, i_b]
                i_la = collider_state.contact_data.link_a[i_c, i_b]
                i_lb = collider_state.contact_data.link_b[i_c, i_b]
                if func_is_link_pair_hibernated(i_la, i_lb, i_b, dyn_state, dyn_info, rigid_config):
                    collider_state.contact_sort_key[i_c, i_b] = 1.0
            n_hib = 0
            for i_c in range(n_raw):
                if collider_state.contact_sort_key[i_c, i_b] > 0.0:
                    # The key takes the compact slot, read back by the logical walk below
                    collider_state.contact_sort_key[i_c, i_b] = n_hib + 1.0
                    if i_c != n_hib:
                        # fmt: off
                        collider_state.contact_data.geom_a[n_hib, i_b] = collider_state.contact_data.geom_a[i_c, i_b]
                        collider_state.contact_data.geom_b[n_hib, i_b] = collider_state.contact_data.geom_b[i_c, i_b]
                        collider_state.contact_data.penetration[n_hib, i_b] = collider_state.contact_data.penetration[i_c, i_b]
                        collider_state.contact_data.normal[n_hib, i_b] = collider_state.contact_data.normal[i_c, i_b]
                        collider_state.contact_data.pos[n_hib, i_b] = collider_state.contact_data.pos[i_c, i_b]
                        collider_state.contact_data.friction[n_hib, i_b] = collider_state.contact_data.friction[i_c, i_b]
                        collider_state.contact_data.friction_torsional[n_hib, i_b] = collider_state.contact_data.friction_torsional[i_c, i_b]
                        collider_state.contact_data.friction_rolling[n_hib, i_b] = collider_state.contact_data.friction_rolling[i_c, i_b]
                        collider_state.contact_data.sol_params[n_hib, i_b] = collider_state.contact_data.sol_params[i_c, i_b]
                        collider_state.contact_data.force[n_hib, i_b] = collider_state.contact_data.force[i_c, i_b]
                        collider_state.contact_data.link_a[n_hib, i_b] = collider_state.contact_data.link_a[i_c, i_b]
                        collider_state.contact_data.link_b[n_hib, i_b] = collider_state.contact_data.link_b[i_c, i_b]
                        # fmt: on
                    n_hib = n_hib + 1
            # Rank r of the kept range never overtakes the logical position it reads, so the walk is in place
            rank = 0
            for i_c_ in range(collider_state.n_contacts[i_b]):
                i_c = collider_state.contact_sort_idx[i_c_, i_b]
                slot_key = collider_state.contact_sort_key[i_c, i_b]
                if slot_key > 0.0:
                    collider_state.contact_sort_idx[rank, i_b] = qd.cast(slot_key, gs.qd_int) - 1
                    rank = rank + 1
        collider_state.n_contacts_hibernated[i_b] = n_hib

    for i_c in range(collider_state.n_contacts[i_b]):
        should_clear = True
        if qd.static(rigid_config.use_hibernation):
            should_clear = i_c >= collider_state.n_contacts_hibernated[i_b]
        if should_clear:
            collider_state.contact_data.link_a[i_c, i_b] = -1
            collider_state.contact_data.link_b[i_c, i_b] = -1
            collider_state.contact_data.geom_a[i_c, i_b] = -1
            collider_state.contact_data.geom_b[i_c, i_b] = -1
            collider_state.contact_data.penetration[i_c, i_b] = 0.0
            collider_state.contact_data.pos[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            collider_state.contact_data.normal[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            collider_state.contact_data.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)

    if qd.static(rigid_config.use_hibernation):
        collider_state.n_contacts[i_b] = collider_state.n_contacts_hibernated[i_b]
    else:
        collider_state.n_contacts_hibernated[i_b] = 0
        collider_state.n_contacts[i_b] = 0


@qd.func
def func_promote_woken_contacts(
    i_b: int,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Move the kept contacts of the links of env i_b that woke this step among the live contacts.

    A kept contact holds where its sleeper rests, and the narrowphase left the sleeper's pairs out while it slept, so
    the woken link would solve its wake step without its support otherwise. A contact is promoted once its pair no
    longer stands still (func_is_link_pair_hibernated). The promoted contact moves to the end of the kept range, which
    shrinks past it onto the first live slot (see n_contacts_hibernated in array_class.py), and the kept contacts
    behind it close the gap in their order, so the getters keep listing the contacts of the links still asleep as they
    stood. The caller then sorts the live range.
    """
    n_hib = collider_state.n_contacts_hibernated[i_b]
    i_c_ = 0
    while i_c_ < n_hib:
        i_c = collider_state.contact_sort_idx[i_c_, i_b]
        i_la = collider_state.contact_data.link_a[i_c, i_b]
        i_lb = collider_state.contact_data.link_b[i_c, i_b]
        if func_is_link_pair_hibernated(i_la, i_lb, i_b, dyn_state, dyn_info, rigid_config):
            i_c_ = i_c_ + 1
        else:
            n_hib = n_hib - 1
            for j_c_ in range(i_c_, n_hib):
                collider_state.contact_sort_idx[j_c_, i_b] = collider_state.contact_sort_idx[j_c_ + 1, i_b]
            collider_state.contact_sort_idx[n_hib, i_b] = i_c
    collider_state.n_contacts_hibernated[i_b] = n_hib


@qd.kernel(fastcache=True)
def kernel_collider_clear(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        func_collider_reset_env(i_b, collider_state, collider_static_config, cache_only=False)
        func_collider_clear_env(i_b, dyn_state, collider_state, dyn_info, rigid_info, rigid_config)
        # Only the contacts of the sleepers remain listed, so their links alone keep a contact force
        for i_l in range(dyn_state.links.contact_force.shape[0]):
            dyn_state.links.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        func_add_contacts_force_to_links(i_b, dyn_state, collider_state)


@qd.kernel(fastcache=True)
def kernel_masked_collider_clear(
    envs_mask: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(envs_mask.shape[0]):
        if envs_mask[i_b]:
            func_collider_reset_env(i_b, collider_state, collider_static_config, cache_only=False)
            func_collider_clear_env(i_b, dyn_state, collider_state, dyn_info, rigid_info, rigid_config)
            # See 'kernel_collider_clear'
            for i_l in range(dyn_state.links.contact_force.shape[0]):
                dyn_state.links.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
            func_add_contacts_force_to_links(i_b, dyn_state, collider_state)


@qd.kernel(fastcache=True)
def collider_kernel_get_contacts(
    iout: qd.types.ndarray(),
    fout: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    rigid_config: qd.template(),
    is_padded: qd.template(),
):
    _B = collider_state.active_buffer.shape[1]

    # TODO: Better implementation from Quadrants for this kind of reduction.
    n_contacts_max = gs.qd_int(0)
    qd.loop_config(serialize=True)
    for i_b in range(_B):
        n_contacts = collider_state.n_contacts[i_b]
        if n_contacts > n_contacts_max:
            n_contacts_max = n_contacts

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        i_c_start = gs.qd_int(0)
        if qd.static(is_padded):
            i_c_start = i_b * n_contacts_max
        else:
            for j_b in range(i_b):
                i_c_start = i_c_start + collider_state.n_contacts[j_b]

        for i_c_ in range(collider_state.n_contacts[i_b]):
            i_c = i_c_start + i_c_
            i_col = collider_state.contact_sort_idx[i_c_, i_b]

            iout[i_c, 0] = collider_state.contact_data.link_a[i_col, i_b]
            iout[i_c, 1] = collider_state.contact_data.link_b[i_col, i_b]
            iout[i_c, 2] = collider_state.contact_data.geom_a[i_col, i_b]
            iout[i_c, 3] = collider_state.contact_data.geom_b[i_col, i_b]
            fout[i_c, 0] = collider_state.contact_data.penetration[i_col, i_b]
            for j in qd.static(range(3)):
                fout[i_c, 1 + j] = collider_state.contact_data.pos[i_col, i_b][j]
                fout[i_c, 4 + j] = collider_state.contact_data.normal[i_col, i_b][j]
                fout[i_c, 7 + j] = collider_state.contact_data.force[i_col, i_b][j]


@qd.func
def func_add_contact(
    i_ga: int,
    i_gb: int,
    i_b: int,
    i_pair: int,
    normal: qd.types.vector(3),
    contact_pos: qd.types.vector(3),
    penetration: float,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    use_atomic: qd.template(),
    errno: qd.Tensor,
):
    """Append a contact to the buffer of an environment, returning its slot, or -1 when the buffer is full."""
    i_c = func_reserve_contacts(
        i_b,
        n_contacts=1,
        collider_state=collider_state,
        collider_info=collider_info,
        use_atomic=use_atomic,
        errno=errno,
    )
    i_c_added = -1
    if i_c < collider_info.max_candidate_contacts[None]:
        i_c_added = i_c
        func_set_contact(
            i_ga,
            i_gb,
            i_b,
            i_c,
            i_pair,
            normal,
            contact_pos,
            penetration,
            dyn_state,
            collider_state,
            dyn_info,
            rigid_info,
            collider_info,
            errno,
        )
    return i_c_added


@qd.func
def func_reserve_contacts(
    i_b: int,
    n_contacts: int,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    use_atomic: qd.template(),
    errno: qd.Tensor,
):
    """Reserve consecutive slots of the contact buffer of an environment, returning the first of them.

    The slots past the end of the buffer are flagged as an overflow, and the caller drops their contacts. Without
    atomics, the count of the buffer only grows by slots it holds.
    """
    i_c = 0
    if qd.static(use_atomic):
        i_c = qd.atomic_add(collider_state.n_contacts[i_b], n_contacts)
    else:
        i_c = collider_state.n_contacts[i_b]
    if i_c + n_contacts <= collider_info.max_candidate_contacts[None]:
        if not qd.static(use_atomic):
            collider_state.n_contacts[i_b] = i_c + n_contacts
    else:
        errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
    return i_c


@qd.func
def func_set_contact(
    i_ga: int,
    i_gb: int,
    i_b: int,
    i_c: int,
    i_pair: int,
    normal: qd.types.vector(3),
    contact_pos: qd.types.vector(3),
    penetration: float,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    errno: qd.Tensor,
):
    """
    Set the contact data for the contact [i_c]. This is used for the backward pass, which parallelizes over the entire
    contact data, and for the split narrowphase multi-contact writes.
    """
    friction_a = dyn_info.geoms.friction[i_ga] * dyn_state.geoms.friction_ratio[i_ga, i_b]
    friction_b = dyn_info.geoms.friction[i_gb] * dyn_state.geoms.friction_ratio[i_gb, i_b]
    friction_torsional_a = dyn_info.geoms.friction_torsional[i_ga] * dyn_state.geoms.friction_ratio[i_ga, i_b]
    friction_torsional_b = dyn_info.geoms.friction_torsional[i_gb] * dyn_state.geoms.friction_ratio[i_gb, i_b]
    friction_rolling_a = dyn_info.geoms.friction_rolling[i_ga] * dyn_state.geoms.friction_ratio[i_ga, i_b]
    friction_rolling_b = dyn_info.geoms.friction_rolling[i_gb] * dyn_state.geoms.friction_ratio[i_gb, i_b]

    # Every contact the solver sees is written here, so a non-finite position, normal or penetration is flagged here
    # rather than several stages later as a force gone wrong. Flagged rather than dropped: a missing contact lets
    # bodies pass through each other just as silently. The magnitude test catches 'inf' and 'nan' at once, where the
    # bit-pattern intrinsics ('qd.math.isnan' / 'qd.math.isinf') are assumed away under fast math for the infinities
    # and have no reverse-mode adjoint - this write is shared with the differentiated narrowphase.
    residual = contact_pos[0] + contact_pos[1] + contact_pos[2] + normal[0] + normal[1] + normal[2] + penetration
    if not (qd.abs(residual) < qd.math.inf):
        errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_CONTACT_NAN

    # b to a
    collider_state.contact_data.geom_a[i_c, i_b] = i_ga
    collider_state.contact_data.geom_b[i_c, i_b] = i_gb
    collider_state.contact_data.normal[i_c, i_b] = normal
    collider_state.contact_data.pos[i_c, i_b] = contact_pos
    collider_state.contact_data.penetration[i_c, i_b] = penetration
    collider_state.contact_data.friction[i_c, i_b] = qd.max(qd.max(friction_a, friction_b), 1e-2)
    collider_state.contact_data.friction_torsional[i_c, i_b] = qd.max(friction_torsional_a, friction_torsional_b)
    collider_state.contact_data.friction_rolling[i_c, i_b] = qd.max(friction_rolling_a, friction_rolling_b)
    # The constraint time constant is floored on the mixed value rather than on each geom's own (see the geom
    # sanitize site in rigid_solver.py); 2.0 is TIME_CONSTANT_SAFETY_FACTOR (rigid_solver.py).
    sol_params = 0.5 * (dyn_info.geoms.sol_params[i_ga] + dyn_info.geoms.sol_params[i_gb])
    sol_params[0] = qd.max(sol_params[0], 2.0 * rigid_info.substep_dt[None])
    collider_state.contact_data.sol_params[i_c, i_b] = sol_params
    collider_state.contact_data.link_a[i_c, i_b] = dyn_info.geoms.link_idx[i_ga]
    collider_state.contact_data.link_b[i_c, i_b] = dyn_info.geoms.link_idx[i_gb]
    collider_state.contact_data.pair_idx[i_c, i_b] = i_pair


@qd.func
def func_add_diff_contact_input(
    i_ga: int,
    i_gb: int,
    i_b: int,
    i_d: int,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    collider_info: array_class.ColliderInfo,
):
    i_c = collider_state.n_contacts[i_b]
    if i_c < collider_info.max_candidate_contacts[None]:
        collider_state.diff_contact_input.geom_a[i_b, i_c] = i_ga
        collider_state.diff_contact_input.geom_b[i_b, i_c] = i_gb
        collider_state.diff_contact_input.local_pos1_a[i_b, i_c] = gjk_state.diff_contact_input.local_pos1_a[i_b, i_d]
        collider_state.diff_contact_input.local_pos1_b[i_b, i_c] = gjk_state.diff_contact_input.local_pos1_b[i_b, i_d]
        collider_state.diff_contact_input.local_pos1_c[i_b, i_c] = gjk_state.diff_contact_input.local_pos1_c[i_b, i_d]
        collider_state.diff_contact_input.local_pos2_a[i_b, i_c] = gjk_state.diff_contact_input.local_pos2_a[i_b, i_d]
        collider_state.diff_contact_input.local_pos2_b[i_b, i_c] = gjk_state.diff_contact_input.local_pos2_b[i_b, i_d]
        collider_state.diff_contact_input.local_pos2_c[i_b, i_c] = gjk_state.diff_contact_input.local_pos2_c[i_b, i_d]
        collider_state.diff_contact_input.w_local_pos1[i_b, i_c] = gjk_state.diff_contact_input.w_local_pos1[i_b, i_d]
        collider_state.diff_contact_input.w_local_pos2[i_b, i_c] = gjk_state.diff_contact_input.w_local_pos2[i_b, i_d]
        # The first contact point is the reference contact point
        collider_state.diff_contact_input.ref_id[i_b, i_c] = i_c - i_d
        collider_state.diff_contact_input.ref_penetration[i_b, i_c] = gjk_state.diff_contact_input.ref_penetration[
            i_b, i_d
        ]


@qd.func
def func_compute_geom_rbound(i_g: int, geoms_init_AABB: array_class.GeomsInitAABB, dyn_info: array_class.DynInfo):
    """Compute the bounding sphere radius for a geom, matching MuJoCo's geom_rbound."""
    geom_type = dyn_info.geoms.type[i_g]
    rbound = gs.qd_float(0.0)
    if geom_type == gs.GEOM_TYPE.SPHERE:
        rbound = dyn_info.geoms.data[i_g][0]
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        # radius + half_length (MuJoCo stores size as [radius, half_length])
        # Genesis stores data as [radius, full_length], so half_length = 0.5 * data[1]
        rbound = dyn_info.geoms.data[i_g][0] + 0.5 * dyn_info.geoms.data[i_g][1]
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        rbound = qd.max(dyn_info.geoms.data[i_g][0], qd.max(dyn_info.geoms.data[i_g][1], dyn_info.geoms.data[i_g][2]))
    elif geom_type == gs.GEOM_TYPE.BOX:
        d0 = dyn_info.geoms.data[i_g][0]
        d1 = dyn_info.geoms.data[i_g][1]
        d2 = dyn_info.geoms.data[i_g][2]
        rbound = qd.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    else:
        # For mesh and other types, approximate as half AABB diagonal
        rbound = 0.5 * (geoms_init_AABB[i_g, 7] - geoms_init_AABB[i_g, 0]).norm()
    return rbound


@qd.func
def func_compute_geom_pair_scale(
    i_ga: int, i_gb: int, geoms_init_AABB: array_class.GeomsInitAABB, dyn_info: array_class.DynInfo
):
    # Intrinsic length scale of a geom pair: half the smaller geom's world-aligned bounding-box diagonal. The
    # original (rest-pose) AABB is used so the scale is a constant independent of the current orientation, which
    # makes sense since the size of the geometries is an intrinsic property. Multiply by a relative tolerance to
    # turn it into an absolute one.
    aabb_size_b = (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]).norm()
    aabb_size = aabb_size_b
    if dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.PLANE:
        aabb_size_a = (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]).norm()
        aabb_size = qd.min(aabb_size_a, aabb_size_b)

    return 0.5 * aabb_size


@qd.func
def func_compute_geom_pair_scale_mj(
    i_ga: int, i_gb: int, geoms_init_AABB: array_class.GeomsInitAABB, dyn_info: array_class.DynInfo
):
    """Geom-pair length scale matching MuJoCo's formula: min(rbound_g1, rbound_g2). Multiply by a relative tolerance
    to recover MuJoCo's absolute tolerance."""
    rbound_a = func_compute_geom_rbound(i_ga, geoms_init_AABB, dyn_info)
    rbound_b = func_compute_geom_rbound(i_gb, geoms_init_AABB, dyn_info)
    return qd.min(rbound_a, rbound_b)


@qd.func
def func_compute_mc_tolerance(
    i_ga: int,
    i_gb: int,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
):
    """Absolute multi-contact acceptance tolerance of a geom pair.

    Shared by every convex narrowphase arm so the accepted contact set stays backend-independent. The relative
    tolerance scales with the reference engine's pair scale under MuJoCo compatibility and with the intrinsic pair
    scale otherwise."""
    scale = func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
    if qd.static(rigid_config.enable_mujoco_compatibility):
        scale = func_compute_geom_pair_scale_mj(i_ga, i_gb, geoms_init_AABB, dyn_info)
    return collider_info.mc_tolerance[None] * scale


@qd.func
def func_contact_orthogonals(
    i_ga: int,
    i_gb: int,
    i_b: int,
    normal: qd.types.vector(3),
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    EPS = rigid_info.EPS[None]

    axis_0 = qd.Vector.zero(gs.qd_float, 3)
    axis_1 = qd.Vector.zero(gs.qd_float, 3)

    if qd.static(rigid_config.enable_mujoco_compatibility):
        # Choose between world axes Y or Z to avoid colinearity issue
        if qd.abs(normal[1]) < 0.5:
            axis_0[1] = 1.0
        else:
            axis_0[2] = 1.0

        # Project axis on orthogonal plane to contact normal
        axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()

        # Complete orthonormal frame (matching MuJoCo's mju_makeFrame)
        axis_1 = normal.cross(axis_0)
        axis_0 = axis_1.cross(normal)
    else:
        # The reference geometry is the one that will have the largest impact on the position of
        # the contact point. Basically, the smallest one between the two, which can be approximated
        # by the volume of their respective bounding box.
        i_g = i_gb
        if dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.PLANE:
            size_ga = geoms_init_AABB[i_ga, 7]
            volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
            size_gb = geoms_init_AABB[i_gb, 7]
            volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
            i_g = i_ga if volume_ga < volume_gb else i_gb

        # The basis is built in the reference geom's local inertial frame, the physical anchor that does not depend on
        # the link origin, maintained by forward kinematics as links.quat composed with the build-time local inertial
        # quat, then rotated back to world. Building the orthogonals on the LOCAL normal keeps the construction's branch
        # decisions fixed to the body, so a scene and any rigidly rotated copy of it perturb along the same directions
        # relative to the geometry and find the same manifold.
        i_l = dyn_info.geoms.link_idx[i_g]
        rot = gu.qd_quat_to_R(dyn_state.links.i_quat[i_l, i_b], EPS)
        axis_0_local, axis_1_local = gu.qd_orthogonals(rot.transpose() @ normal)
        # Tilting along a principal direction can leave an entire edge at equal depth, allowing repeated contacts.
        # Diagonal directions break these ties on faces aligned with the inertial frame
        axis_0 = qd.sqrt(0.5) * (rot @ (axis_0_local + axis_1_local))
        axis_1 = qd.sqrt(0.5) * (rot @ (axis_1_local - axis_0_local))

    return axis_0, axis_1


@qd.func
def func_rotate_frame(
    pos: qd.types.vector(3), quat: qd.types.vector(4), contact_pos: qd.types.vector(3), qrot: qd.types.vector(4)
) -> tuple[qd.types.vector(3), qd.types.vector(4)]:
    """
    Instead of modifying geoms_state in place, this function takes thread-local
    pos/quat and returns the updated values.
    """
    new_quat = gu.qd_transform_quat_by_quat(quat, qrot)

    rel = contact_pos - pos
    vec = gu.qd_transform_by_quat(rel, qrot)
    vec = vec - rel
    new_pos = pos - vec

    return new_pos, new_quat


@qd.func
def func_contact_order_key(pos: qd.types.vector(3)):
    """Order a contact position along one generic direction, as a single scalar.

    Comparing components in turn tests each for equality, and in a frame attached to the geometry the contacts of one
    patch share components exactly - a box face puts two corners at the same local x - so the ordering follows the
    rounding of a mathematically tied quantity. Projecting on a direction no face of a box or regular prism is parallel
    to separates the points of a patch by a margin of their own spacing. The weights are successive powers of the
    golden ratio, as far from any rational direction as a pair of weights gets.
    """
    return pos[0] + 1.618033988749895 * pos[1] + 2.618033988749895 * pos[2]


@qd.kernel
def func_set_upstream_grad(
    dL_dposition: qd.types.ndarray(),
    dL_dnormal: qd.types.ndarray(),
    dL_dpenetration: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
):
    _B = dL_dposition.shape[0]
    _C = dL_dposition.shape[1]
    for i_b, i_c in qd.ndrange(_B, _C):
        # The upstream gradients follow the logical order of the contacts that get_contacts returns, which
        # contact_sort_idx maps to the physical slots of contact_data.
        i_col = i_c
        if i_c < collider_state.n_contacts[i_b]:
            i_col = collider_state.contact_sort_idx[i_c, i_b]
        for j in qd.static(range(3)):
            collider_state.contact_data.pos.grad[i_col, i_b][j] = dL_dposition[i_b, i_c, j]
            collider_state.contact_data.normal.grad[i_col, i_b][j] = dL_dnormal[i_b, i_c, j]
        collider_state.contact_data.penetration.grad[i_col, i_b] = dL_dpenetration[i_b, i_c]
