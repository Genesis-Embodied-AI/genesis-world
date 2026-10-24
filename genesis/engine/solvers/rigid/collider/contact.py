"""
Contact management and utility functions for the rigid body collider.

This module contains functions for adding contacts, computing tolerances,
and managing contact data including reset/clear operations.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.simt as su

from .constants import CONTACT_ORDER


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


@qd.func
def func_contact_frame_order_key(
    i_c: int, i_b: int, dyn_state: array_class.DynState, collider_state: array_class.ColliderState
):
    """Order key of a contact position in the frame of its second geom (see func_contact_order_key)."""
    i_gb = collider_state.contact_data.geom_b[i_c, i_b]
    return func_contact_order_key(
        gu.qd_inv_transform_by_quat(
            collider_state.contact_data.pos[i_c, i_b] - dyn_state.geoms.pos[i_gb, i_b], dyn_state.geoms.quat[i_gb, i_b]
        )
    )


@qd.func
def func_contact_order_get(i_b: int, i_o: int, collider_state: array_class.ColliderState, order: qd.template()):
    """Read the contact at position 'i_o' of the ordering sorted by func_contact_heapsort in the given order."""
    i_c = 0
    if qd.static(order == CONTACT_ORDER.POSITION):
        i_c = collider_state.contact_lex_idx[i_o, i_b]
    else:
        i_c = collider_state.contact_sort_idx[i_o, i_b]
    return i_c


@qd.func
def func_contact_order_set(
    i_b: int, i_o: int, i_c: int, collider_state: array_class.ColliderState, order: qd.template()
):
    """Write the contact at position 'i_o' of the ordering sorted by func_contact_heapsort in the given order."""
    if qd.static(order == CONTACT_ORDER.POSITION):
        collider_state.contact_lex_idx[i_o, i_b] = i_c
    else:
        collider_state.contact_sort_idx[i_o, i_b] = i_c


@qd.func
def func_contact_is_after(
    i_b: int, i_x: int, i_y: int, collider_state: array_class.ColliderState, order: qd.template()
):
    """Whether contact 'i_x' sorts after contact 'i_y' in the given order (see CONTACT_ORDER), ties broken by index.

    The intrinsic order compares the whole data of the contacts, their positions and normals last, so that only
    identical contacts tie, whose order does not matter. Its first keys already separate all other contacts, so the
    others are read only when those tie, sparing the sort the latency of their reads.
    """
    # Sort key of each contact, compared lexicographically, in two stages read one after the other
    keys = qd.Matrix.zero(gs.qd_float, 2, 3)
    for i_k, i_c in qd.static(enumerate((i_x, i_y))):
        key = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(order == CONTACT_ORDER.LINK_PAIR):
            i_la = collider_state.contact_data.link_a[i_c, i_b]
            i_lb = collider_state.contact_data.link_b[i_c, i_b]
            key[0], key[1], key[2] = qd.min(i_la, i_lb), qd.max(i_la, i_lb), i_c
        elif qd.static(order == CONTACT_ORDER.INTRINSIC):
            key[0] = collider_state.contact_data.geom_a[i_c, i_b]
            key[1] = collider_state.contact_data.geom_b[i_c, i_b]
            key[2] = collider_state.contact_proj_v[i_c, i_b]
        else:
            key[0], key[1], key[2] = (
                collider_state.contact_sort_key[i_c, i_b],
                collider_state.contact_proj_v[i_c, i_b],
                i_c,
            )
        for i_3 in qd.static(range(3)):
            keys[i_k, i_3] = key[i_3]
    is_after = False
    is_tied = True
    for i_3 in qd.static(range(3)):
        if is_tied:
            if keys[0, i_3] > keys[1, i_3]:
                is_after = True
                is_tied = False
            elif keys[0, i_3] < keys[1, i_3]:
                is_tied = False
    if qd.static(order == CONTACT_ORDER.INTRINSIC):
        if is_tied:
            keys_tail = qd.Matrix.zero(gs.qd_float, 2, 7)
            for i_k, i_c in qd.static(enumerate((i_x, i_y))):
                pos = collider_state.contact_data.pos[i_c, i_b]
                normal = collider_state.contact_data.normal[i_c, i_b]
                keys_tail[i_k, 0] = collider_state.contact_data.penetration[i_c, i_b]
                for i_3 in qd.static(range(3)):
                    keys_tail[i_k, 1 + i_3] = pos[i_3]
                    keys_tail[i_k, 4 + i_3] = normal[i_3]
            for i_7 in qd.static(range(7)):
                if is_tied:
                    if keys_tail[0, i_7] > keys_tail[1, i_7]:
                        is_after = True
                        is_tied = False
                    elif keys_tail[0, i_7] < keys_tail[1, i_7]:
                        is_tied = False
    return is_after


@qd.func
def func_contact_heapsort(
    i_b: int, i_o_start: int, i_o_end: int, collider_state: array_class.ColliderState, order: qd.template()
):
    """Sort a range of contacts in place by heapsort, in linearithmic time and without extra memory.

    In the POSITION order (see CONTACT_ORDER), the bucket-logical indices in 'contact_lex_idx' are sorted by the
    projections of the contacts on the plane, held in 'contact_sort_key' and 'contact_proj_v'. Otherwise the contact
    indices in 'contact_sort_idx' are sorted by link pair, or by their intrinsic data with the frame-local order key
    held in 'contact_proj_v' at the contact index. Position alone would leave coincident contacts of different geoms
    (adjacent ring wedges touching the pole at one shared point) tied, in an order that changes from run to run.
    """
    n_o = i_o_end - i_o_start
    # The first n_o // 2 rounds sift each internal node down to build the max-heap, the next n_o - 1 rounds move its
    # root behind the shrinking heap and sift the new root down. A single sift serves both.
    for i_round in range(n_o // 2 + n_o - 1):
        i_root = n_o // 2 - 1 - i_round
        n_heap = n_o
        if i_round >= n_o // 2:
            i_root = 0
            n_heap = n_o - 1 - (i_round - n_o // 2)
            i_top = func_contact_order_get(i_b, i_o_start, collider_state, order)
            i_last = func_contact_order_get(i_b, i_o_start + n_heap, collider_state, order)
            func_contact_order_set(i_b, i_o_start, i_last, collider_state, order)
            func_contact_order_set(i_b, i_o_start + n_heap, i_top, collider_state, order)
        i_node = func_contact_order_get(i_b, i_o_start + i_root, collider_state, order)
        while 2 * i_root + 1 < n_heap:
            # Each step compares the later child with the earlier one, then the larger child with the node, both in
            # one loop so that the comparison is inlined once
            i_child = 2 * i_root + 1
            i_c = func_contact_order_get(i_b, i_o_start + i_child, collider_state, order)
            i_s = i_c
            if i_child + 1 < n_heap:
                i_s = func_contact_order_get(i_b, i_o_start + i_child + 1, collider_state, order)
            is_after = False
            for i_cmp in range(2):
                i_x, i_y = i_s, i_c
                if i_cmp == 1:
                    i_x, i_y = i_c, i_node
                is_after = False
                if i_cmp == 1 or i_child + 1 < n_heap:
                    is_after = func_contact_is_after(i_b, i_x, i_y, collider_state, order)
                if i_cmp == 0 and is_after:
                    i_child += 1
                    i_c = i_s
            if is_after:
                func_contact_order_set(i_b, i_o_start + i_root, i_c, collider_state, order)
                i_root = i_child
            else:
                break
        func_contact_order_set(i_b, i_o_start + i_root, i_node, collider_state, order)


@qd.func
def func_contact_sort_key(i_la: int, i_lb: int, i_ga: int, i_gb: int, frame_key: float):
    """Leading keys of a contact in the order of phase 1 of the contact pruning (see func_contact_is_before).

    Returns the link pair, the geom pair and the frame order key 'frame_key' of the contact between geoms 'i_ga' and
    'i_gb' of links 'i_la' and 'i_lb'.
    """
    return qd.Vector([qd.min(i_la, i_lb), qd.max(i_la, i_lb), i_ga, i_gb, frame_key], dt=gs.qd_float)


@qd.func
def func_contact_is_before(
    i_b: int,
    i_q: int,
    i_p: int,
    key_q: qd.types.vector(5),
    key_p: qd.types.vector(5),
    collider_state: array_class.ColliderState,
):
    """Whether one contact sorts before another in the order of phase 1 of the contact pruning.

    Contacts are ordered by link pair, then geom pair, then frame order key ('contact_proj_v', see
    func_contact_frame_order_key), then intrinsic order (see func_contact_is_after), the lower index going first among
    identical contacts. The link indices are compared themselves, as a key packing them loses exactness and lets
    distinct pairs interleave in the order of their contact indices, which the racy atomic slot reservation of the
    narrowphase sets. 'key_q' and 'key_p' are the leading keys of both contacts (see func_contact_sort_key), and the
    intrinsic order alone reads the contact data.

    Returns whether contact 'i_q' of environment 'i_b' sorts before contact 'i_p'.
    """
    is_before = False
    is_tied = True
    for i_k in qd.static(range(5)):
        if is_tied:
            if key_q[i_k] < key_p[i_k]:
                is_before = True
                is_tied = False
            elif key_q[i_k] > key_p[i_k]:
                is_tied = False
    if is_tied:
        # One loop runs both comparisons, so that the comparison is inlined once
        is_after_pq = False
        is_after_qp = False
        for i_cmp in range(2):
            i_x, i_y = i_p, i_q
            if i_cmp == 1:
                i_x, i_y = i_q, i_p
            is_after = func_contact_is_after(i_b, i_x, i_y, collider_state, CONTACT_ORDER.INTRINSIC)
            if i_cmp == 0:
                is_after_pq = is_after
            else:
                is_after_qp = is_after
        is_before = is_after_pq or (i_q < i_p and not is_after_qp)
    return is_before


@qd.func
def func_contact_link_pair_end(i_b: int, i_cb_start: int, n_con: int, collider_state: array_class.ColliderState):
    """End of the bucket of contacts sharing the link pair of the contact at 'i_cb_start', sorted by link pair.

    Bucket boundaries are derived from the link ids, which the sort groups contiguously.
    """
    i_pc0 = collider_state.contact_sort_idx[i_cb_start, i_b]
    i_la0 = collider_state.contact_data.link_a[i_pc0, i_b]
    i_lb0 = collider_state.contact_data.link_b[i_pc0, i_b]
    i_cb_end = i_cb_start + 1
    while i_cb_end < n_con:
        i_pc = collider_state.contact_sort_idx[i_cb_end, i_b]
        i_la = collider_state.contact_data.link_a[i_pc, i_b]
        i_lb = collider_state.contact_data.link_b[i_pc, i_b]
        if qd.min(i_la, i_lb) != qd.min(i_la0, i_lb0) or qd.max(i_la, i_lb) != qd.max(i_la0, i_lb0):
            break
        i_cb_end = i_cb_end + 1
    return i_cb_end


@qd.func
def func_contact_patch_end(
    i_b: int, i_cb_start: int, n_con: int, collider_state: array_class.ColliderState, cos_tol: float
):
    """Find the end of the patch that a contact opens, in a bucket of contacts grouped by func_contact_group_by_normal.

    The patch of the contact at 'i_cb_start', among the first 'n_con' contacts, runs up to the first contact of another
    link pair or whose normal leaves the axis of its own by an angle of cosine below 'cos_tol'. A contact of a later
    patch agrees with no earlier first contact, or it would have joined its patch.
    """
    i_pc_start = collider_state.contact_sort_idx[i_cb_start, i_b]
    normal_start = collider_state.contact_data.normal[i_pc_start, i_b]
    i_cb_pair_end = func_contact_link_pair_end(i_b, i_cb_start, n_con, collider_state)
    i_cb_end = i_cb_pair_end
    for i_cb in range(i_cb_start + 1, i_cb_pair_end):
        i_pc = collider_state.contact_sort_idx[i_cb, i_b]
        if i_cb < i_cb_end and qd.abs(collider_state.contact_data.normal[i_pc, i_b].dot(normal_start)) < cos_tol:
            i_cb_end = i_cb
    return i_cb_end


@qd.func
def func_contact_patch_end_coop(
    tid: int,
    i_b: int,
    i_cb_start: int,
    n_con: int,
    collider_state: array_class.ColliderState,
    cos_tol: float,
    is_patch: qd.template(),
):
    """End of the bucket or of the patch that a contact opens, found by the 32 lanes of a block together.

    The bucket of the contact at 'i_cb_start', among the first 'n_con' contacts sorted by link pair, ends at the first
    contact of another link pair, as func_contact_link_pair_end finds it. The patch ('is_patch' set) ends there too, or
    earlier at the first contact whose normal leaves the axis of its own by an angle of cosine below 'cos_tol', as
    func_contact_patch_end finds it. The lanes test 32 contacts at a time, so that the walk waits on one read of the
    contacts per 32 of them instead of one per contact. Every lane receives the end.
    """
    i_pc_start = collider_state.contact_sort_idx[i_cb_start, i_b]
    i_la_start = collider_state.contact_data.link_a[i_pc_start, i_b]
    i_lb_start = collider_state.contact_data.link_b[i_pc_start, i_b]
    i_l_min_start = qd.min(i_la_start, i_lb_start)
    i_l_max_start = qd.max(i_la_start, i_lb_start)
    normal_start = collider_state.contact_data.normal[i_pc_start, i_b]
    i_cb_end = n_con
    for i_chunk_ in range((n_con - i_cb_start - 1 + 31) // 32):
        if i_cb_end == n_con:
            i_cb = i_cb_start + 1 + i_chunk_ * 32 + tid
            i_cb_end_l = n_con
            if i_cb < n_con:
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                i_la = collider_state.contact_data.link_a[i_pc, i_b]
                i_lb = collider_state.contact_data.link_b[i_pc, i_b]
                is_end = qd.min(i_la, i_lb) != i_l_min_start or qd.max(i_la, i_lb) != i_l_max_start
                if qd.static(is_patch):
                    is_end = is_end or qd.abs(collider_state.contact_data.normal[i_pc, i_b].dot(normal_start)) < cos_tol
                if is_end:
                    i_cb_end_l = i_cb
            i_cb_end = su.qd_block_min(i_cb_end_l)
    return i_cb_end


@qd.func
def func_contact_group_by_normal(
    i_b: int, i_cb_start: int, i_cb_end: int, collider_state: array_class.ColliderState, cos_tol: float
):
    """Group a link-pair bucket of contacts into patches of agreeing normals, keeping their order within each patch.

    A patch holds the contacts whose normals lie along the axis of its first contact within an angle of cosine
    'cos_tol', and the support polygon prunes within a patch only: a contact along another normal transmits a wrench
    the others do not span. Each contact of the bucket between 'i_cb_start' and 'i_cb_end' joins the first patch it
    agrees with, in the order of the bucket, which ends up sorted by patch, each patch running up to
    func_contact_patch_end.
    """
    # The bucket-logical offsets of the first contacts of the patches go to contact_hull_stack, free until the support
    # polygon is built, the patch of each contact to contact_sort_key at its index, free once the bucket is sorted, and
    # the bucket gathered patch by patch to contact_lex_idx, free alike
    n_patches = 0
    for i_cb in range(i_cb_start, i_cb_end):
        i_pc = collider_state.contact_sort_idx[i_cb, i_b]
        normal = collider_state.contact_data.normal[i_pc, i_b]
        i_cb_patch_ = -1
        for i_patch in range(n_patches):
            if i_cb_patch_ < 0:
                i_cb_head_ = collider_state.contact_hull_stack[i_cb_start + i_patch, i_b]
                i_pc_head = collider_state.contact_sort_idx[i_cb_start + i_cb_head_, i_b]
                if qd.abs(collider_state.contact_data.normal[i_pc_head, i_b].dot(normal)) >= cos_tol:
                    i_cb_patch_ = i_cb_head_
        if i_cb_patch_ < 0:
            i_cb_patch_ = i_cb - i_cb_start
            collider_state.contact_hull_stack[i_cb_start + n_patches, i_b] = i_cb_patch_
            n_patches += 1
        collider_state.contact_sort_key[i_pc, i_b] = i_cb_patch_
    i_cb_out = i_cb_start
    for i_patch in range(n_patches):
        i_cb_head_ = collider_state.contact_hull_stack[i_cb_start + i_patch, i_b]
        for i_cb in range(i_cb_start + i_cb_head_, i_cb_end):
            i_pc = collider_state.contact_sort_idx[i_cb, i_b]
            if qd.cast(collider_state.contact_sort_key[i_pc, i_b], gs.qd_int) == i_cb_head_:
                collider_state.contact_lex_idx[i_cb_out, i_b] = i_pc
                i_cb_out += 1
    for i_cb in range(i_cb_start, i_cb_end):
        collider_state.contact_sort_idx[i_cb, i_b] = collider_state.contact_lex_idx[i_cb, i_b]


@qd.func
def func_contact_hull_vertex_pos(
    i_b: int, i_cb_start: int, i_h: int, n_hull: int, collider_state: array_class.ColliderState
):
    """Projected position of the hull vertex 'i_h', counted cyclically, whether removed or not.

    A removed vertex keeps its contact index encoded as negative (see func_contact_support_hull).
    """
    i_c = collider_state.contact_hull_stack[i_cb_start + (i_h + n_hull) % n_hull, i_b]
    if i_c < 0:
        i_c = -1 - i_c
    return qd.Vector([collider_state.contact_sort_key[i_c, i_b], collider_state.contact_proj_v[i_c, i_b]])


@qd.func
def func_contact_support_hull(
    i_b: int,
    i_cb_start: int,
    i_cb_end: int,
    collider_state: array_class.ColliderState,
    tol: float,
    rounding: float,
    eps: float,
    sort_positions: bool,
):
    """Find the support polygon of a bucket of coplanar contacts, as the vertices of their convex hull in the plane.

    The contacts of the bucket, identified by their bucket-logical indices in [i_cb_start, i_cb_end), are projected
    beforehand on the plane, with their coordinates in 'contact_sort_key' and 'contact_proj_v'. They are sorted
    lexicographically in 'contact_lex_idx' (see func_contact_heapsort), unless 'sort_positions' is unset because the
    caller sorted them already, and Andrew's monotone chain keeps the vertices of their exact convex hull. A single pass
    along the hull then drops some of the vertices whose removal shrinks the hull by less than the fraction 'tol' of its
    area, the points of its edges among them. The hull visits the points of an edge in their order along it, while the
    sort does not as soon as the edge is nearly parallel to the first sort axis, so a tolerance on the turns of the
    chain itself would drop a corner in place of a point of the edge. Every step is linear in the number of contacts
    but the sort, which is linearithmic.

    'rounding' is the length within which points of the bucket count as lying on one line, relative to the size of the
    bucket so that the outcome does not depend on where it stands. 'eps' bounds the relative rounding error of the float
    type (see gs.EPS).

    Return the simplified hull size and the deepest original corner's bucket-logical index (-1 if absent). Hull
    vertices are stored in 'contact_hull_stack' from 'i_cb_start' on.
    """
    if sort_positions:
        func_contact_heapsort(i_b, i_cb_start, i_cb_end, collider_state, CONTACT_ORDER.POSITION)

    # Track the top two hull-stack entries in locals rather than re-reading the just-written contact_hull_stack slots.
    # On Apple Metal, reading a slot written in the previous iteration can return a stale value (a compiler bug), so
    # the re-read is avoided and only the deeper entry is reloaded.
    n_cb = i_cb_end - i_cb_start
    n_hull = 0
    i_ht = qd.i32(-1)
    i_hs = qd.i32(-1)
    for i_pass in range(2):
        # The lower chain walks the sorted contacts forward, the upper one backward from the second to last, down to
        # the leftmost one, which already sits at the bottom of the stack
        n_hull_base = 0 if i_pass == 0 else n_hull
        for i_step in range(n_cb if i_pass == 0 else n_cb - 1):
            i_cb = i_cb_start + i_step if i_pass == 0 else i_cb_end - 2 - i_step
            i_p = collider_state.contact_lex_idx[i_cb, i_b]
            pos_p = qd.Vector([collider_state.contact_sort_key[i_p, i_b], collider_state.contact_proj_v[i_p, i_b]])
            while n_hull >= n_hull_base + 2 - i_pass:
                pos_s = qd.Vector(
                    [collider_state.contact_sort_key[i_hs, i_b], collider_state.contact_proj_v[i_hs, i_b]]
                )
                pos_t = qd.Vector(
                    [collider_state.contact_sort_key[i_ht, i_b], collider_state.contact_proj_v[i_ht, i_b]]
                )
                dir_t, dir_p = pos_t - pos_s, pos_p - pos_s
                # Under fused multiply-add contraction, the difference of two equal products evaluates to the rounding
                # error of one of them, of either sign, so coincident and collinear points would turn either way
                # depending on the compiler. A turn within the rounding error of its products counts as none.
                cross_lhs = dir_t[0] * dir_p[1]
                cross_rhs = dir_t[1] * dir_p[0]
                if cross_lhs - cross_rhs <= 2.0 * eps * (qd.abs(cross_lhs) + qd.abs(cross_rhs)):
                    n_hull -= 1
                    i_ht = i_hs
                    if n_hull >= 2:
                        i_hs = collider_state.contact_hull_stack[i_cb_start + n_hull - 2, i_b]
                else:
                    break
            # The upper chain ends at the leftmost point, which opens the lower one, so that it only pops the points
            # its return makes redundant, its duplicates included. The n_hull < n_cb guard bounds the stack to the
            # bucket size.
            if (i_pass == 0 or i_step < n_cb - 2) and n_hull < n_cb:
                collider_state.contact_hull_stack[i_cb_start + n_hull, i_b] = i_p
                i_hs = i_ht
                i_ht = i_p
                n_hull = n_hull + 1

    # Area-based removal limits each lost triangle to 'tol' of the original hull area. Removing local minima keeps
    # adjacent corners for the next pass. Projected positions order near-equal losses (see area_rounding below).
    # Negative contact indices mark removed corners while their neighbours still read them during the pass.
    hull_area_2 = gs.qd_float(0.0)
    hull_perimeter = gs.qd_float(0.0)
    # Collinear edge points follow the deep-interior rule. Preserve the deepest corner during area simplification
    i_cb_deepest = -1
    penetration_max = gs.qd_float(-1.0)
    pos_prev = func_contact_hull_vertex_pos(i_b, i_cb_start, n_hull - 1, n_hull, collider_state)
    for i_h in range(n_hull):
        pos_p = func_contact_hull_vertex_pos(i_b, i_cb_start, i_h, n_hull, collider_state)
        pos_n = func_contact_hull_vertex_pos(i_b, i_cb_start, i_h + 1, n_hull, collider_state)
        hull_area_2 += pos_p[0] * pos_n[1] - pos_p[1] * pos_n[0]
        hull_perimeter += (pos_n - pos_p).norm()
        dir_in, dir_out = pos_p - pos_prev, pos_n - pos_p
        area_corner = qd.abs(dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0])
        if n_hull <= 2 or area_corner > rounding * (dir_in.norm() + dir_out.norm()):
            i_cb = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
            i_pc = collider_state.contact_sort_idx[i_cb, i_b]
            penetration = collider_state.contact_data.penetration[i_pc, i_b]
            if penetration > penetration_max:
                penetration_max = penetration
                i_cb_deepest = i_cb
        pos_prev = pos_p
    area_tol = tol * qd.abs(hull_area_2)
    # Near-equal area losses use the projected positions to keep corner selection stable under positional rounding.
    area_rounding = rounding * hull_perimeter

    # A hull whose doubled area is within 'rounding' times its perimeter is a segment, every point within rounding of
    # one line. Its support is its two ends, the two vertices farthest apart, which a sweep to the farthest vertex from
    # any one, then from that one, finds. The turns along it, all of rounding size, would decide nothing.
    is_hull_segment = n_hull > 2 and qd.abs(hull_area_2) <= rounding * hull_perimeter
    if is_hull_segment:
        i_end_0 = 0
        i_end_1 = 0
        for i_sweep in qd.static(range(2)):
            pos_from = func_contact_hull_vertex_pos(i_b, i_cb_start, i_end_0, n_hull, collider_state)
            dist_sqr_max = gs.qd_float(-1.0)
            for i_h in range(n_hull):
                dist_sqr = (
                    func_contact_hull_vertex_pos(i_b, i_cb_start, i_h, n_hull, collider_state) - pos_from
                ).norm_sqr()
                if dist_sqr > dist_sqr_max:
                    dist_sqr_max = dist_sqr
                    i_end_1 = i_h
            if qd.static(i_sweep == 0):
                i_end_0 = i_end_1
        i_c_0 = collider_state.contact_hull_stack[i_cb_start + i_end_0, i_b]
        i_c_1 = collider_state.contact_hull_stack[i_cb_start + i_end_1, i_b]
        collider_state.contact_hull_stack[i_cb_start, i_b] = i_c_0
        collider_state.contact_hull_stack[i_cb_start + 1, i_b] = i_c_1
        n_hull = 2
    # A vertex turning by less than 'rounding' along its edges is collinear with its neighbours and goes whatever the
    # area it carries, while a genuine triangle keeps its three vertices, each carrying its whole area. Collinear
    # vertices obey the rule that spares one of two neighbours too: copies of one point a few roundings apart, which
    # overlapping geoms report, are each collinear with the other, and removing both would remove their corner. The
    # pass repeats until it removes nothing, each pass taking its losses on the hull as it found it, through a window
    # sliding along the hull over the vertex, its two neighbours and the next one, and the losses of the first three.
    n_kept = n_hull
    is_converged = False
    for i_pass in range(n_hull):
        if not is_converged and n_kept > 2:
            poss = qd.Matrix.zero(gs.qd_float, 5, 2)
            losses = qd.Vector.zero(gs.qd_float, 3)
            for i_k in qd.static(range(5)):
                pos = func_contact_hull_vertex_pos(i_b, i_cb_start, i_k - 2, n_kept, collider_state)
                poss[i_k, 0], poss[i_k, 1] = pos[0], pos[1]
            for i_k in qd.static(range(3)):
                dir_in = qd.Vector([poss[i_k + 1, 0] - poss[i_k, 0], poss[i_k + 1, 1] - poss[i_k, 1]])
                dir_out = qd.Vector([poss[i_k + 2, 0] - poss[i_k + 1, 0], poss[i_k + 2, 1] - poss[i_k + 1, 1]])
                losses[i_k] = qd.abs(dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0])
            for i_k in qd.static(range(4)):
                poss[i_k, 0], poss[i_k, 1] = poss[i_k + 1, 0], poss[i_k + 1, 1]
            for i_h in range(n_kept):
                dir_in = qd.Vector([poss[1, 0] - poss[0, 0], poss[1, 1] - poss[0, 1]])
                dir_out = qd.Vector([poss[2, 0] - poss[1, 0], poss[2, 1] - poss[1, 1]])
                is_between = dir_in.dot(dir_out) > 0.0
                is_vertex_collinear = losses[1] <= rounding * (dir_in.norm() + dir_out.norm())
                is_least = True
                for i_k in qd.static((0, 2)):
                    # Removal eligibility takes precedence over positional tie-breaking.
                    is_tie = qd.abs(losses[1] - losses[i_k]) <= area_rounding
                    is_tie = is_tie and ((losses[1] <= area_tol) == (losses[i_k] <= area_tol))
                    is_before = poss[1, 0] < poss[i_k, 0] or (poss[1, 0] <= poss[i_k, 0] and poss[1, 1] < poss[i_k, 1])
                    is_least = is_least and ((not is_tie and losses[1] < losses[i_k]) or (is_tie and is_before))
                if is_between and is_least and (is_vertex_collinear or losses[1] <= area_tol):
                    i_c = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                    collider_state.contact_hull_stack[i_cb_start + i_h, i_b] = -1 - i_c
                pos_last = func_contact_hull_vertex_pos(i_b, i_cb_start, i_h + 3, n_kept, collider_state)
                for i_k in qd.static(range(3)):
                    poss[i_k, 0], poss[i_k, 1] = poss[i_k + 1, 0], poss[i_k + 1, 1]
                poss[3, 0], poss[3, 1] = pos_last[0], pos_last[1]
                losses[0], losses[1] = losses[1], losses[2]
                dir_in = qd.Vector([poss[2, 0] - poss[1, 0], poss[2, 1] - poss[1, 1]])
                dir_out = qd.Vector([poss[3, 0] - poss[2, 0], poss[3, 1] - poss[2, 1]])
                losses[2] = qd.abs(dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0])
            n_left = 0
            for i_h in range(n_kept):
                i_c = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                if i_c >= 0:
                    collider_state.contact_hull_stack[i_cb_start + n_left, i_b] = i_c
                    n_left += 1
            is_converged = n_left == n_kept
            n_kept = n_left
    return n_kept, i_cb_deepest


@qd.func
def func_contact_patch_basis(normal: qd.types.vector(3)):
    """Orthonormal basis of the plane of a contact patch of unit normal 'normal'.

    The first axis is the world axis least aligned with the normal, projected onto the plane and normalized, the
    second one is the normal crossed with the first one. The world axis follows the normal alone, so every lane that
    holds the same normal gets the same basis.
    """
    normal_abs = qd.abs(normal)
    axis = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    if normal_abs[1] < normal_abs[0] and normal_abs[1] < normal_abs[2]:
        axis = qd.Vector([0.0, 1.0, 0.0])
    elif normal_abs[2] < normal_abs[0] and normal_abs[2] <= normal_abs[1]:
        axis = qd.Vector([0.0, 0.0, 1.0])
    axis_u = axis - axis.dot(normal) * normal
    axis_u /= axis_u.norm()
    return axis_u, normal.cross(axis_u)


@qd.func
def func_clamp_prune_contacts(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    """Clamp + (optional) link-pair pruning, in one per-env loop pass.

    Builds a logical-to-physical contact permutation in contact_sort_idx rather than rewriting contact_data. After this
    func runs, downstream consumers read contact i_col by indirecting through
    contact_data.X[contact_sort_idx[i_col, i_b], i_b]. The physical layout of contact_data is left intact.

    Phases per env (gated at compile time by collider_static_config):
    - Always: clamp n_contacts to max_candidate_contacts; initialise contact_sort_idx to the identity.
    - If has_prunable_contacts and not requires_grad: prune redundant contacts via 2D convex hull on the
      contact-patch plane (skipped at runtime when contact_pruning_tolerance is 0). Drops are realised by compacting
      contact_sort_idx rather than contact_data.
    - Always: clamp the surviving n_contacts to max_contacts (the budget sizing the contact constraint buffers) and
      flag OVERFLOW_CONTACTS in errno, which halts the simulation at the next errno check.

    Deterministic ordering of the kept contacts (independent of the racy atomic_add narrowphase layout) is applied
    later in add_inequality_constraints, not here.

    The pruning logic groups contacts by canonical (min(link_a, link_b), max(link_a, link_b)), splits each bucket into
    patches of normals lying along one axis (see func_contact_group_by_normal) and, for each patch of >= 3 contacts
    whose positions lie in a single plane (perpendicular to the patch's folded mean normal), keeps the 2D convex hull
    vertices and the deepest original corner beyond rounding tolerance. Patches whose positions span multiple planes are
    left untouched. Surviving contacts retain their normals. The patch mean normal sets the projection direction.

    The ``tol`` parameter bounds the angle in radians between the normals of a patch, and the depth gate as a
    dimensionless slop fraction:
      max |out-of-plane offset| / in-plane radius <= tol.

    Phases (per env, scratch sized to max_candidate_contacts):
    1. Group by canonical link-pair: heapsort ``contact_sort_idx`` by (min_link, max_link), then each bucket by the
       intrinsic data of its contacts (see CONTACT_ORDER), then group each bucket into patches.
    2. Per patch of >= 3 contacts: compute mean normal (folded to a common hemisphere). Check depth coplanarity of
       contact positions. If they share a plane, project to (u, v) and find their support polygon (see
       func_contact_support_hull). Mark survivors in contact_keep[] (indexed by bucket-logical position).
    3. Compact: squeeze dropped slots out of ``contact_sort_idx`` and update ``n_contacts``.
    """
    _B = collider_state.n_contacts.shape[0]
    max_candidate_contacts = collider_info.max_candidate_contacts[None]
    max_contacts = collider_info.max_contacts[None]
    tol = collider_info.contact_pruning_tolerance[None]
    # The normals of a patch lie along the axis of its first contact within the angle 'tol' (see
    # func_contact_group_by_normal)
    cos_tol = 1.0 - 0.5 * tol * tol
    prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
    EPS = rigid_info.EPS[None]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        n_con = qd.min(collider_state.n_contacts[i_b], max_candidate_contacts)
        collider_state.n_contacts[i_b] = n_con
        # The kept contacts of the sleepers lead the buffer in the order func_collider_clear_env gave them, and the
        # prune runs on the live contacts after them (see n_contacts_hibernated in array_class.py)
        n_hib = collider_state.n_contacts_hibernated[i_b]

        # Identity permutation of the live contacts. Required so downstream consumers can always indirect through
        # contact_sort_idx, even when pruning is inactive.
        for i_c in range(n_hib, n_con):
            collider_state.contact_sort_idx[i_c, i_b] = i_c

        # === Pruning phase (link-pair support polygon). Gated by static config: only emitted when the
        # scene has multi-geom links / nonconvex / terrain, and not in autodiff mode. Skipped at runtime
        # when contact_pruning_tolerance is 0.
        if qd.static(collider_static_config.has_prunable_contacts and not rigid_config.requires_grad):
            if n_con - n_hib >= 3 and tol > gs.qd_float(0.0):
                # Phase 1: sort contact_sort_idx by canonical (min_link, max_link) pair. The sort_idx already holds
                # the identity from the unconditional init above.
                func_contact_heapsort(i_b, n_hib, n_con, collider_state, CONTACT_ORDER.LINK_PAIR)

                # Deterministic within-bucket order. The sort above only orders by link pair, so contacts sharing one
                # keep the non-deterministic physical layout (atomic_add slot reservation, multi-pass narrowphase).
                # Sorting each bucket by the intrinsic data of its contacts makes the sums over the bucket and the
                # survivor set reproducible. The frame-local order key of each contact is held in contact_proj_v at its
                # index, so every bucket is sorted before the walk below, whose projections overwrite these keys. A
                # single contact is already in order. The bucket is then grouped into patches, which the walk below
                # prunes one by one (see func_contact_group_by_normal).
                i_cb_start = n_hib
                while i_cb_start < n_con:
                    i_cb_end = func_contact_link_pair_end(i_b, i_cb_start, n_con, collider_state)
                    if i_cb_end - i_cb_start >= 2:
                        for i_cb in range(i_cb_start, i_cb_end):
                            i_p = collider_state.contact_sort_idx[i_cb, i_b]
                            collider_state.contact_proj_v[i_p, i_b] = func_contact_frame_order_key(
                                i_p, i_b, dyn_state, collider_state
                            )
                        func_contact_heapsort(i_b, i_cb_start, i_cb_end, collider_state, CONTACT_ORDER.INTRINSIC)
                    if func_contact_patch_end(i_b, i_cb_start, n_con, collider_state, cos_tol) < i_cb_end:
                        func_contact_group_by_normal(i_b, i_cb_start, i_cb_end, collider_state, cos_tol)
                    i_cb_start = i_cb_end

                # Default: keep everything. Patches that pass the gates flip their entries to drop and then mark only
                # hull-vertex contacts as keep again.
                for i_c in range(n_con):
                    collider_state.contact_keep[i_c, i_b] = 1

                # Phase 2: walk the patches of each link pair (logical-contiguous after the sorts above).
                i_cb_start = n_hib
                while i_cb_start < n_con:
                    i_cb_end = func_contact_patch_end(i_b, i_cb_start, n_con, collider_state, cos_tol)
                    n_cb = i_cb_end - i_cb_start

                    if n_cb >= 3:
                        i_pc0 = collider_state.contact_sort_idx[i_cb_start, i_b]

                        # Mean normal (folded to the hemisphere of contact at i_cb_start) and centroid.
                        normal_ref = collider_state.contact_data.normal[i_pc0, i_b]
                        mean_normal = qd.Vector.zero(gs.qd_float, 3)
                        centroid = qd.Vector.zero(gs.qd_float, 3)
                        for i_cb in range(i_cb_start, i_cb_end):
                            i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                            normal_c = collider_state.contact_data.normal[i_pc, i_b]
                            sign = gs.qd_float(1.0)
                            if normal_ref.dot(normal_c) < 0.0:
                                sign = -1.0
                            mean_normal += sign * normal_c
                            centroid += collider_state.contact_data.pos[i_pc, i_b]
                        centroid *= 1.0 / n_cb
                        mean_normal_norm = mean_normal.norm()

                        # Hoisted out so the hull-build branch below can read it (quadrants scopes per if).
                        max_in_plane_r2 = gs.qd_float(0.0)

                        is_coplanar = mean_normal_norm > EPS
                        if is_coplanar:
                            mean_normal /= mean_normal_norm

                            # Depth coplanarity: positions must lie in a single plane perpendicular to the mean normal. No
                            # per-contact normal check: a contact whose normal is diagonal (e.g. an edge-vs-edge contact at a
                            # corner of the contact patch) still participates in the 2D hull because its position is a vertex of
                            # the patch; dropping a collinear-edge contact in the same bucket is justified by the positional
                            # support polygon regardless of that contact's normal direction.
                            max_depth = gs.qd_float(0.0)
                            for i_cb in range(i_cb_start, i_cb_end):
                                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                                delta = collider_state.contact_data.pos[i_pc, i_b] - centroid
                                depth = qd.abs(delta.dot(mean_normal))
                                if depth > max_depth:
                                    max_depth = depth
                                radius_sq = delta.norm_sqr() - depth * depth
                                if radius_sq > max_in_plane_r2:
                                    max_in_plane_r2 = radius_sq

                            if max_depth > tol * qd.sqrt(max_in_plane_r2):
                                is_coplanar = False

                        if is_coplanar:
                            axis_u, axis_v = func_contact_patch_basis(mean_normal)

                            # Project bucket contacts to (u, v). sort_key holds u, contact_proj_v holds v. Both are
                            # indexed by bucket-logical position so the (u, v) sort below can read them without another
                            # indirection. The projections are taken relative to the centroid, so that their rounding
                            # scales with the patch rather than with its distance to the origin, which the area of the
                            # support polygon would otherwise cancel down to.
                            for i_cb in range(i_cb_start, i_cb_end):
                                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                                delta = collider_state.contact_data.pos[i_pc, i_b] - centroid
                                collider_state.contact_sort_key[i_cb, i_b] = delta.dot(axis_u)
                                collider_state.contact_proj_v[i_cb, i_b] = delta.dot(axis_v)

                            for i_cb in range(i_cb_start, i_cb_end):
                                collider_state.contact_lex_idx[i_cb, i_b] = i_cb
                            n_hull, i_cb_deepest = func_contact_support_hull(
                                i_b,
                                i_cb_start,
                                i_cb_end,
                                collider_state,
                                tol,
                                rounding=qd.sqrt(EPS * max_in_plane_r2),
                                eps=EPS,
                                sort_positions=True,
                            )

                            # Overwrite contact_keep[b_start..b_end) with the final drop/keep flags: drop everything,
                            # then mark hull vertices keep.
                            for i_cb in range(i_cb_start, i_cb_end):
                                collider_state.contact_keep[i_cb, i_b] = 0
                            for i_h in range(n_hull):
                                i_hv = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                                collider_state.contact_keep[i_hv, i_b] = 1

                            # Restore non-hull contacts whose penetration is much deeper than the hull boundary's
                            # average. The support-polygon argument says interior contacts are wrench-redundant only
                            # when ALL contacts share the same normal and penetration; a contact with substantially
                            # higher penetration than the hull's average represents a distinct physical support (the
                            # body of a fork resting beyond its tines, the deep middle of a long body) and dropping it
                            # lets the body sink into the surface. The 3x factor is well above the typical ~1.x
                            # penetration spread on transient/rocking faces (so non-uniform-penetration buckets like
                            # irregular mesh contacts keep only the hull) but well below the deep interior penetrations
                            # seen when a non-flat body rests inside its convex envelope (so genuine deep supports are
                            # restored).
                            hull_pen_max = gs.qd_float(0.0)
                            for i_h in range(n_hull):
                                i_hv = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                                i_pc = collider_state.contact_sort_idx[i_hv, i_b]
                                pen = collider_state.contact_data.penetration[i_pc, i_b]
                                if pen > hull_pen_max:
                                    hull_pen_max = pen
                            # Preserve the deepest corner when simplifying the hull. Estimate depth rounding from the
                            # two surface witnesses and the patch radius.
                            if i_cb_deepest >= 0:
                                i_pc_deepest = collider_state.contact_sort_idx[i_cb_deepest, i_b]
                                penetration_max = collider_state.contact_data.penetration[i_pc_deepest, i_b]
                                depth_rounding = EPS * (
                                    2.0 * collider_state.contact_data.pos[i_pc_deepest, i_b].norm()
                                    + penetration_max
                                    + 2.0 * qd.sqrt(max_in_plane_r2)
                                )
                                if penetration_max > hull_pen_max + depth_rounding:
                                    collider_state.contact_keep[i_cb_deepest, i_b] = 1
                            deep_keep_threshold = prune_deep_penetration_ratio * hull_pen_max
                            for i_cb in range(i_cb_start, i_cb_end):
                                if collider_state.contact_keep[i_cb, i_b] == 0:
                                    i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                                    if collider_state.contact_data.penetration[i_pc, i_b] > deep_keep_threshold:
                                        collider_state.contact_keep[i_cb, i_b] = 1

                    i_cb_start = i_cb_end

                # Phase 3: compact contact_sort_idx by squeezing out dropped slots.
                i_cw = n_hib
                for i_cr in range(n_hib, n_con):
                    if collider_state.contact_keep[i_cr, i_b] != 0:
                        if i_cw != i_cr:
                            collider_state.contact_sort_idx[i_cw, i_b] = collider_state.contact_sort_idx[i_cr, i_b]
                        i_cw = i_cw + 1
                collider_state.n_contacts[i_b] = i_cw

        # The contact constraint buffers are sized to 4 * max_contacts, so any surviving contact beyond that budget
        # would write out of bounds. Clamp and flag the env: check_errno halts the simulation with a request to
        # increase 'max_contacts'.
        if collider_state.n_contacts[i_b] > max_contacts:
            collider_state.n_contacts[i_b] = max_contacts
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CONTACTS


@qd.func
def func_clamp_prune_contacts_coop(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    errno: qd.Tensor,
):
    """GPU-only cooperative warp-per-env variant of func_clamp_prune_contacts.

    Only dispatched when pruning is enabled, so it prunes unconditionally (no static gate). Same clamp + prune
    algorithm and same contract (mandatory clamp + identity-init contact_sort_idx + phase-3 compact) as
    func_clamp_prune_contacts. Deterministic ordering of the kept contacts is applied later in
    add_inequality_constraints.
    Difference from func_clamp_prune_contacts: a block of warps splits the per-env work:
      - PARALLEL over contacts, every thread: per-contact init and the phase-1 sort.
      - PARALLEL over contacts, the lanes of one warp: bucket and patch ends and the phase-3 compact.
      - PARALLEL over patches, one warp each: mean-normal / centroid reductions, coplanarity reduction, in-plane
        projection writes and lexicographic ranks.
      - PARALLEL over patches, one thread each: support polygon, hull-mark and deep-pen restore.
      - SERIAL on lane 0: grouping of a bucket of several patches.
    """
    _B = collider_state.n_contacts.shape[0]
    max_candidate_contacts = collider_info.max_candidate_contacts[None]
    max_contacts = collider_info.max_contacts[None]
    tol = collider_info.contact_pruning_tolerance[None]
    # See func_clamp_prune_contacts
    cos_tol = 1.0 - 0.5 * tol * tol
    prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
    EPS = rigid_info.EPS[None]

    # A block of _W warps of _K lanes runs each env. The warps sort the contacts together, then share out the patches
    # of a round: each warp computes the statistics of its patches with its own lanes, and each lane of the block
    # builds the support polygon of one patch.
    _K = qd.static(32)
    _W = qd.static(4)
    # Contacts whose sort keys the block holds in shared memory, two per thread (see phase 1)
    _N_STAGED = qd.static(2 * _K * _W)
    qd.loop_config(name="clamp_prune_contacts_coop", block_dim=_K * _W)
    for i_flat in range(_B * _K * _W):
        i_thread = i_flat % (_K * _W)
        tid = i_thread % _K
        i_warp = i_thread // _K
        i_b = i_flat // (_K * _W)
        sh_patch_start = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_patch_end = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_is_hull = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_max_in_plane_r2 = qd.simt.block.SharedArray((_K * _W,), gs.qd_float)
        sh_round = qd.simt.block.SharedArray((2,), gs.qd_int)
        sh_sort_geoms = qd.simt.block.SharedArray((_N_STAGED, 2), gs.qd_int)
        sh_sort_v = qd.simt.block.SharedArray((_N_STAGED,), gs.qd_float)
        sh_sort_idx = qd.simt.block.SharedArray((_N_STAGED,), gs.qd_int)
        n_con = qd.min(collider_state.n_contacts[i_b], max_candidate_contacts)
        n_hib = collider_state.n_contacts_hibernated[i_b]
        qd.simt.block.sync()
        if i_thread == 0:
            collider_state.n_contacts[i_b] = n_con

        # PARALLEL: clamp+init. Mirrors the fused kernel's unconditional init block: every env (including n_con < 5
        # where the prune/sort branch below is skipped) needs contact_sort_idx set to identity over the live contacts
        # so downstream consumers that always indirect through contact_sort_idx (constraint solver, sensors) read
        # valid permutations rather than stale data from the previous step. contact_keep default-keep is set here for
        # the same reason.
        for i_chunk_ in range((n_con - n_hib + _K * _W - 1) // (_K * _W)):
            i_c = n_hib + i_chunk_ * _K * _W + i_thread
            if i_c < n_con:
                collider_state.contact_keep[i_c, i_b] = 1
                collider_state.contact_sort_idx[i_c, i_b] = i_c

        if n_con - n_hib >= 3:
            # Phase 1: contacts in a deterministic order, grouped by link pair then sorted within each pair (see
            # func_clamp_prune_contacts). The threads sort the live contacts in place by a bitonic network, whose every
            # step puts the earlier contact of a pair of slots in the lower slot: the first step of a level pairs each
            # slot with its mirror in a block of twice the size of the previous level, and the following ones pair slots
            # half as far apart each time. The live contacts are padded with slots that sort after every contact up to a
            # power of two, which never move under such steps, so that the pairs reaching them are skipped.
            # The contacts of an env lie one environment stride apart in memory, so that the reads of a block share no
            # cache line as soon as there are several envs. While they fit, the block reads their geom pairs and frame
            # order keys once into shared memory, sorts them there, then writes the permutation back. The intrinsic
            # order of tied keys reads the contact data still. A larger capacity measured slower on phase 2, whose
            # reads go through the cache.
            n_live = n_con - n_hib
            is_staged = n_live <= _N_STAGED and _B > 1
            for i_chunk_ in range((n_live + _K * _W - 1) // (_K * _W)):
                i_s = i_chunk_ * _K * _W + i_thread
                if i_s < n_live:
                    i_c = n_hib + i_s
                    frame_key = func_contact_frame_order_key(i_c, i_b, dyn_state, collider_state)
                    collider_state.contact_proj_v[i_c, i_b] = frame_key
                    if is_staged:
                        sh_sort_geoms[i_s, 0] = collider_state.contact_data.geom_a[i_c, i_b]
                        sh_sort_geoms[i_s, 1] = collider_state.contact_data.geom_b[i_c, i_b]
                        sh_sort_v[i_s] = frame_key
                        sh_sort_idx[i_s] = i_c
            qd.simt.block.sync()
            n_level = 0
            for i_bit in range(31):
                if (1 << i_bit) < n_live:
                    n_level = i_bit + 1
            for i_level in range(n_level):
                for i_step in range(i_level + 1):
                    slot_mask = 1 << (i_level - i_step)
                    if i_step == 0:
                        slot_mask = (2 << i_level) - 1
                    # The lower slot of each pair has the bit of the pair distance cleared, so that the threads take
                    # the pairs directly by inserting a cleared bit at that position into their pair index
                    i_bit_pair = i_level - i_step
                    for i_chunk_ in range(((1 << (n_level - 1)) + _K * _W - 1) // (_K * _W)):
                        i_pair = i_chunk_ * _K * _W + i_thread
                        i_s = ((i_pair >> i_bit_pair) << (i_bit_pair + 1)) | (i_pair & ((1 << i_bit_pair) - 1))
                        i_t = i_s ^ slot_mask
                        if i_t < n_live:
                            i_p = 0
                            i_q = 0
                            key_p = qd.Vector.zero(gs.qd_float, 5)
                            key_q = qd.Vector.zero(gs.qd_float, 5)
                            for i_k_ in range(2):
                                i_ss = i_s if i_k_ == 0 else i_t
                                i_c = 0
                                i_la = 0
                                i_lb = 0
                                i_ga = 0
                                i_gb = 0
                                frame_key = gs.qd_float(0.0)
                                if is_staged:
                                    # A contact touches the links of its geoms, which every env shares
                                    i_c = sh_sort_idx[i_ss]
                                    i_ga = sh_sort_geoms[i_ss, 0]
                                    i_gb = sh_sort_geoms[i_ss, 1]
                                    i_la = dyn_info.geoms.link_idx[i_ga]
                                    i_lb = dyn_info.geoms.link_idx[i_gb]
                                    frame_key = sh_sort_v[i_ss]
                                else:
                                    i_c = collider_state.contact_sort_idx[n_hib + i_ss, i_b]
                                    i_la = collider_state.contact_data.link_a[i_c, i_b]
                                    i_lb = collider_state.contact_data.link_b[i_c, i_b]
                                    i_ga = collider_state.contact_data.geom_a[i_c, i_b]
                                    i_gb = collider_state.contact_data.geom_b[i_c, i_b]
                                    frame_key = collider_state.contact_proj_v[i_c, i_b]
                                key = func_contact_sort_key(i_la, i_lb, i_ga, i_gb, frame_key)
                                if i_k_ == 0:
                                    i_p = i_c
                                    key_p = key
                                else:
                                    i_q = i_c
                                    key_q = key
                            if func_contact_is_before(i_b, i_q, i_p, key_q, key_p, collider_state):
                                if is_staged:
                                    sh_sort_idx[i_s] = i_q
                                    sh_sort_idx[i_t] = i_p
                                    for i_k in qd.static(range(2)):
                                        i_g = sh_sort_geoms[i_s, i_k]
                                        sh_sort_geoms[i_s, i_k] = sh_sort_geoms[i_t, i_k]
                                        sh_sort_geoms[i_t, i_k] = i_g
                                    sh_sort_v[i_s] = key_q[4]
                                    sh_sort_v[i_t] = key_p[4]
                                else:
                                    collider_state.contact_sort_idx[n_hib + i_s, i_b] = i_q
                                    collider_state.contact_sort_idx[n_hib + i_t, i_b] = i_p
                    qd.simt.block.sync()
            if is_staged:
                for i_chunk_ in range((n_live + _K * _W - 1) // (_K * _W)):
                    i_s = i_chunk_ * _K * _W + i_thread
                    if i_s < n_live:
                        collider_state.contact_sort_idx[n_hib + i_s, i_b] = sh_sort_idx[i_s]
                qd.simt.block.sync()

            # Group the contacts of each link pair into patches (see func_contact_group_by_normal). The lanes of the
            # first warp find the end of the first patch of a bucket together, which ends the bucket unless a contact of
            # the same link pair follows, and lane 0 groups the rare bucket of several patches.
            if i_warp == 0:
                i_cb_start = n_hib
                while i_cb_start < n_con:
                    i_cb_end = func_contact_patch_end_coop(
                        tid, i_b, i_cb_start, n_con, collider_state, cos_tol, is_patch=True
                    )
                    if i_cb_end < n_con:
                        i_pc_start = collider_state.contact_sort_idx[i_cb_start, i_b]
                        i_pc_end = collider_state.contact_sort_idx[i_cb_end, i_b]
                        i_la_start = collider_state.contact_data.link_a[i_pc_start, i_b]
                        i_lb_start = collider_state.contact_data.link_b[i_pc_start, i_b]
                        i_la_end = collider_state.contact_data.link_a[i_pc_end, i_b]
                        i_lb_end = collider_state.contact_data.link_b[i_pc_end, i_b]
                        if qd.min(i_la_end, i_lb_end) == qd.min(i_la_start, i_lb_start) and qd.max(
                            i_la_end, i_lb_end
                        ) == qd.max(i_la_start, i_lb_start):
                            i_cb_end = func_contact_patch_end_coop(
                                tid, i_b, i_cb_end, n_con, collider_state, cos_tol, is_patch=False
                            )
                            if tid == 0:
                                func_contact_group_by_normal(i_b, i_cb_start, i_cb_end, collider_state, cos_tol)
                            qd.simt.subgroup.sync()
                    i_cb_start = i_cb_end
            qd.simt.block.sync()

            # Phase 2: the patches are taken in rounds of up to one per thread. The first warp finds the patches of a
            # round, each warp computes the mean normal, the centroid, the coplanarity check, the projections and the
            # lexicographic ranks of its share of them with its own lanes, then each thread builds the support polygon
            # of one patch and restores its deep contacts, a serial chain of reads that the patches of a round run side
            # by side. Each patch writes the scratch of its own range of contacts alone.
            i_cb_next = n_hib
            while i_cb_next < n_con:
                if i_warp == 0:
                    i_cb_start = i_cb_next
                    n_patches_round = 0
                    while i_cb_start < n_con and n_patches_round < _K * _W:
                        i_cb_end = func_contact_patch_end_coop(
                            tid, i_b, i_cb_start, n_con, collider_state, cos_tol, is_patch=True
                        )
                        if tid == 0:
                            sh_patch_start[n_patches_round] = i_cb_start
                            sh_patch_end[n_patches_round] = i_cb_end
                        n_patches_round += 1
                        i_cb_start = i_cb_end
                    if tid == 0:
                        sh_round[0] = n_patches_round
                        sh_round[1] = i_cb_start
                qd.simt.block.sync()
                n_patches_round = sh_round[0]
                i_cb_next = sh_round[1]

                for i_patch_ in range((n_patches_round + _W - 1) // _W):
                    i_patch = i_patch_ * _W + i_warp
                    if i_patch < n_patches_round:
                        i_cb_start_p = sh_patch_start[i_patch]
                        i_cb_end_p = sh_patch_end[i_patch]
                        n_cb = i_cb_end_p - i_cb_start_p
                        is_hull = False
                        max_in_plane_r2 = gs.qd_float(0.0)

                        if n_cb >= 3:
                            i_pc0 = collider_state.contact_sort_idx[i_cb_start_p, i_b]
                            normal_ref = collider_state.contact_data.normal[i_pc0, i_b]
                            mean_normal_l = qd.Vector.zero(gs.qd_float, 3)
                            centroid_l = qd.Vector.zero(gs.qd_float, 3)
                            i_cb_ = i_cb_start_p + tid
                            while i_cb_ < i_cb_end_p:
                                i_pc = collider_state.contact_sort_idx[i_cb_, i_b]
                                normal_c = collider_state.contact_data.normal[i_pc, i_b]
                                sign = gs.qd_float(1.0)
                                if normal_ref.dot(normal_c) < 0.0:
                                    sign = -1.0
                                mean_normal_l += sign * normal_c
                                centroid_l += collider_state.contact_data.pos[i_pc, i_b]
                                i_cb_ += _K

                            mean_normal = qd.Vector.zero(gs.qd_float, 3)
                            centroid = qd.Vector.zero(gs.qd_float, 3)
                            for i_3 in qd.static(range(3)):
                                mean_normal[i_3] = su.qd_block_sum(mean_normal_l[i_3])
                                centroid[i_3] = su.qd_block_sum(centroid_l[i_3])

                            # Every lane holds the sums lane 0 does (see qd_block_sum in utils/simt.py), so the
                            # post-reduce math runs on all 32 lanes and agrees bit for bit across them.
                            centroid *= 1.0 / n_cb
                            mean_normal_norm = mean_normal.norm()

                            is_coplanar = mean_normal_norm > EPS
                            if is_coplanar:
                                mean_normal /= mean_normal_norm

                                # COOP coplanarity check (stage 3). Each lane strides [i_cb_start_p + tid, i_cb_end_p)
                                # by _K, locally tracking max_depth / max_in_plane_r2. Wasted work per warp is at most
                                # n_cb/_K. The upstream algo no longer checks per-contact normals (a contact with a
                                # diagonal normal at the corner of a patch still participates in the 2D hull because its
                                # position is a vertex), so we only do the depth coplanarity gate here.
                                max_depth_l = gs.qd_float(0.0)
                                max_radius_sq_l = gs.qd_float(0.0)
                                i_cb_ = i_cb_start_p + tid
                                while i_cb_ < i_cb_end_p:
                                    i_pc = collider_state.contact_sort_idx[i_cb_, i_b]
                                    delta = collider_state.contact_data.pos[i_pc, i_b] - centroid
                                    depth = qd.abs(delta.dot(mean_normal))
                                    if depth > max_depth_l:
                                        max_depth_l = depth
                                    radius_sq = delta.norm_sqr() - depth * depth
                                    if radius_sq > max_radius_sq_l:
                                        max_radius_sq_l = radius_sq
                                    i_cb_ += _K

                                max_depth = su.qd_block_max(max_depth_l)
                                max_in_plane_r2 = su.qd_block_max(max_radius_sq_l)

                                if max_depth > tol * qd.sqrt(max_in_plane_r2):
                                    is_coplanar = False

                            if is_coplanar:
                                # Every lane computes the same basis from the mean normal the reduce broadcast to it
                                axis_u, axis_v = func_contact_patch_basis(mean_normal)

                                # COOP projection relative to the centroid (see func_clamp_prune_contacts): 32 lanes
                                # stride writes to contact_sort_key + contact_proj_v.
                                i_cb_ = i_cb_start_p + tid
                                while i_cb_ < i_cb_end_p:
                                    i_pc = collider_state.contact_sort_idx[i_cb_, i_b]
                                    delta = collider_state.contact_data.pos[i_pc, i_b] - centroid
                                    collider_state.contact_sort_key[i_cb_, i_b] = delta.dot(axis_u)
                                    collider_state.contact_proj_v[i_cb_, i_b] = delta.dot(axis_v)
                                    i_cb_ += _K

                                # COOP mark-drop: stride writes to contact_keep[i_pc].
                                i_cb_ = i_cb_start_p + tid
                                while i_cb_ < i_cb_end_p:
                                    i_pc = collider_state.contact_sort_idx[i_cb_, i_b]
                                    collider_state.contact_keep[i_pc, i_b] = 0
                                    i_cb_ += _K

                                # COOP lexicographic order of the projections (see func_contact_support_hull): each lane
                                # ranks its contacts among those of the bucket
                                qd.simt.subgroup.sync()
                                i_cb_ = i_cb_start_p + tid
                                while i_cb_ < i_cb_end_p:
                                    i_lex = i_cb_start_p
                                    pos_u = collider_state.contact_sort_key[i_cb_, i_b]
                                    pos_v = collider_state.contact_proj_v[i_cb_, i_b]
                                    for i_cr in range(i_cb_start_p, i_cb_end_p):
                                        pos_ur = collider_state.contact_sort_key[i_cr, i_b]
                                        pos_vr = collider_state.contact_proj_v[i_cr, i_b]
                                        if pos_ur < pos_u or (
                                            pos_ur <= pos_u and (pos_vr < pos_v or (pos_vr <= pos_v and i_cr < i_cb_))
                                        ):
                                            i_lex += 1
                                    collider_state.contact_lex_idx[i_lex, i_b] = i_cb_
                                    i_cb_ += _K

                                # SYNC between coop writes (sort_key, proj_v, lex_idx, contact_keep[i_pc]) and the hull
                                # build that reads them.
                                qd.simt.subgroup.sync()

                            is_hull = is_coplanar

                        if tid == 0:
                            sh_is_hull[i_patch] = 1 if is_hull else 0
                            sh_max_in_plane_r2[i_patch] = max_in_plane_r2
                qd.simt.block.sync()

                # Consecutive patches go to different warps, which step through their diverging hull builds
                # independently, whereas the lanes of one warp step through them in turn
                i_patch_hull = tid * _W + i_warp
                i_cb_start_l = 0
                i_cb_end_l = 0
                is_hull_l = False
                max_in_plane_r2_l = gs.qd_float(0.0)
                if i_patch_hull < n_patches_round:
                    i_cb_start_l = sh_patch_start[i_patch_hull]
                    i_cb_end_l = sh_patch_end[i_patch_hull]
                    is_hull_l = sh_is_hull[i_patch_hull] != 0
                    max_in_plane_r2_l = sh_max_in_plane_r2[i_patch_hull]
                if is_hull_l:
                    n_hull, i_cb_deepest = func_contact_support_hull(
                        i_b,
                        i_cb_start_l,
                        i_cb_end_l,
                        collider_state,
                        tol,
                        rounding=qd.sqrt(EPS * max_in_plane_r2_l),
                        eps=EPS,
                        sort_positions=False,
                    )

                    for i_h in range(n_hull):
                        i_hv = collider_state.contact_hull_stack[i_cb_start_l + i_h, i_b]
                        i_pc = collider_state.contact_sort_idx[i_hv, i_b]
                        collider_state.contact_keep[i_pc, i_b] = 1

                    # Deep-penetration restore. See func_clamp_prune_contacts for the rationale. Indices here live in
                    # orig-space because the cycle-permute is fused into phase 3 below (contact_data is still in
                    # pre-sort order, so we translate sort-space hull/bucket indices through contact_sort_idx).
                    hull_pen_max = gs.qd_float(0.0)
                    for i_h in range(n_hull):
                        i_hv = collider_state.contact_hull_stack[i_cb_start_l + i_h, i_b]
                        i_pc = collider_state.contact_sort_idx[i_hv, i_b]
                        pen = collider_state.contact_data.penetration[i_pc, i_b]
                        if pen > hull_pen_max:
                            hull_pen_max = pen
                    # See the depth-preservation rounding estimate in func_clamp_prune_contacts
                    if i_cb_deepest >= 0:
                        i_pc_deepest = collider_state.contact_sort_idx[i_cb_deepest, i_b]
                        penetration_max = collider_state.contact_data.penetration[i_pc_deepest, i_b]
                        depth_rounding = EPS * (
                            2.0 * collider_state.contact_data.pos[i_pc_deepest, i_b].norm()
                            + penetration_max
                            + 2.0 * qd.sqrt(max_in_plane_r2_l)
                        )
                        if penetration_max > hull_pen_max + depth_rounding:
                            collider_state.contact_keep[i_pc_deepest, i_b] = 1
                    deep_keep_threshold = prune_deep_penetration_ratio * hull_pen_max
                    for i_cb in range(i_cb_start_l, i_cb_end_l):
                        i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                        if collider_state.contact_keep[i_pc, i_b] == 0:
                            if collider_state.contact_data.penetration[i_pc, i_b] > deep_keep_threshold:
                                collider_state.contact_keep[i_pc, i_b] = 1
                qd.simt.block.sync()

            # Phase 3 (compact): squeeze dropped slots out of contact_sort_idx and update n_contacts. The survivors
            # keep the order phase 1 and the per-bucket sort leave them in, which is what holds a scene and a rotated
            # copy of it to the same reported contacts. contact_keep is indexed physically, so the read indirects
            # through the permutation while the write compacts it in place. The lanes take 32 slots at a time, each
            # survivor writing at its rank among them, which trails every slot not read yet. Fewer contacts than a
            # patch prunes keep their identity permutation, which needs no compaction.
            qd.simt.block.sync()
            if i_warp == 0:
                i_cw = n_hib
                for i_chunk_ in range((n_con - n_hib + _K - 1) // _K):
                    i_c = n_hib + i_chunk_ * _K + tid
                    i_pc = 0
                    is_kept = 0
                    if i_c < n_con:
                        i_pc = collider_state.contact_sort_idx[i_c, i_b]
                        if collider_state.contact_keep[i_pc, i_b] != 0:
                            is_kept = 1
                    i_rank_incl, n_kept = su.qd_block_scan(is_kept)
                    if is_kept != 0:
                        collider_state.contact_sort_idx[i_cw + i_rank_incl - 1, i_b] = i_pc
                    i_cw += n_kept
                if tid == 0:
                    collider_state.n_contacts[i_b] = i_cw

        # The contact constraint buffers are sized to 4 * max_contacts, so any surviving contact beyond that budget
        # would write out of bounds. Clamp and flag the env: check_errno halts the simulation with a request to
        # increase 'max_contacts'.
        if i_thread == 0:
            if collider_state.n_contacts[i_b] > max_contacts:
                collider_state.n_contacts[i_b] = max_contacts
                errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CONTACTS


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
