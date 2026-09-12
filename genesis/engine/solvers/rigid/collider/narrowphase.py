"""
Narrow-phase collision detection functions.

This module contains SDF-based contact detection, convex-convex contact,
terrain detection, box-box contact, and multi-contact search algorithms.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf

from . import capsule_contact, diff_gjk, gjk, mpr
from .box_contact import func_box_box_contact, func_plane_box_contact, func_sphere_box_contact
from .constants import CCD_ALGORITHM_CODE, PORTAL_STATUS
from .contact import (
    func_add_contact,
    func_add_diff_contact_input,
    func_apply_smooth_refinement,
    func_compute_geom_pair_scale,
    func_compute_mc_tolerance,
    func_contact_orthogonals,
    func_rotate_frame,
    func_set_contact,
)
from .utils import func_point_in_geom_aabb


@qd.func
def func_contact_sphere_sdf(
    i_ga,
    i_gb,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
):
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    sphere_center = dyn_state.geoms.pos[i_ga, i_b]
    sphere_radius = dyn_info.geoms.data[i_ga][0]

    center_to_b_dist = sdf.sdf_func_world(i_gb, i_b, sphere_center, dyn_state.geoms, dyn_info.geoms, collider_info.sdf)
    if center_to_b_dist < sphere_radius:
        is_col = True
        normal = sdf.sdf_func_normal_world(
            i_gb,
            i_b,
            sphere_center,
            dyn_state.geoms,
            dyn_info.geoms,
            rigid_info,
            collider_info.sdf,
            collider_static_config,
        )
        penetration = sphere_radius - center_to_b_dist
        contact_pos = sphere_center - (sphere_radius - 0.5 * penetration) * normal

    return is_col, normal, penetration, contact_pos


@qd.func
def func_add_polytope_vertex_contacts_sdf(
    i_ga,
    i_gb,
    i_b,
    i_pair,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    tolerance,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    seeded: qd.template(),
    errno: qd.Tensor,
):
    # Emit up to n_max contacts at the deepest spatially-diverse vertices of A penetrating (or near-touching) B's
    # surface. Pass 1 scans every vertex of A, evaluates B's grid SDF at each, and keeps the n_max deepest in a small
    # buffer; spatial diversity is enforced during insertion so the buffer doesn't collapse onto one feature. Pass 2
    # emits a contact at each kept vertex with a normal sampled from B's SDF gradient (per-vertex when the local
    # gradient is reliable, falling back to A's centre gradient otherwise). The deepest selection is required for
    # fine-tessellated A (spoons, ico-spheres): with O(n_verts) body work, the buffer must capture the actual contact
    # patch and not arbitrary body verts that would starve the solver of normal-force capacity. The margin is sized to
    # B's smallest SDF cell so verts within one cell of the surface still register a contact even when the kernel pen
    # reads <= 0; the synthetic pen tapers smoothly across the band to avoid a discontinuity that would drive a settled
    # body into a limit-cycle oscillation.
    n_max = qd.static(collider_static_config.n_contacts_per_nonconvex_pair if rigid_config.enable_multi_contact else 1)
    EPS = rigid_info.EPS[None]
    gb_cell = collider_info.sdf.geoms_info.sdf_cell_size[i_gb]
    margin = qd.min(qd.min(gb_cell[0], gb_cell[1]), gb_cell[2])
    synthetic_pen_max = 1e-4

    # Bounding-sphere-vs-SDF coarse reject at A's centre. Every point of A lies within rbound_a of
    # geoms_info.center[i_ga], so when B's SDF at A's centre exceeds rbound_a no point of A can reach B's surface and
    # the O(n_verts) scan is skipped. rbound_a is the tight sphere around A's AABB centred at geoms_info.center[i_ga]
    # (which is not necessarily the AABB midpoint for decomposed convex pieces). The reject is gated on
    # can_use_sd_reject: only valid when B's SDF query is exact - true for SPHERE and PLANE (analytical) and for grid
    # B with the query point inside the grid; outside the grid the SDF falls back to a proxy that can over-report
    # distance and silently miss a contact. A directional/SAT bound that uses the SDF gradient at A's centre would be
    # tighter but is unsafe on nonconvex B: the centre gradient is a local linearisation, so an A vertex on the
    # opposite side can still reach a different feature of B that the centre points away from.
    center_local = dyn_info.geoms.center[i_ga]
    rbound_a_sq = gs.qd_float(0.0)
    for k in qd.static(range(8)):
        delta = geoms_init_AABB[i_ga, k] - center_local
        d_sq = delta.dot(delta)
        if d_sq > rbound_a_sq:
            rbound_a_sq = d_sq
    rbound_a = qd.sqrt(rbound_a_sq)
    center_a_world = gu.qd_transform_by_trans_quat(center_local, ga_pos, ga_quat)
    can_use_sd_reject = (
        dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.SPHERE or dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.PLANE
    )
    if not can_use_sd_reject:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(center_a_world, gb_pos, gb_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[i_gb])
        can_use_sd_reject = not sdf.sdf_func_is_outside_sdf_grid(i_gb, pos_sdf, collider_info.sdf)
    sd_center = sdf.sdf_func_world_local(i_gb, center_a_world, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf)

    # Contacts already emitted for this pair by the first scan, recovered from the buffer where the pair's contacts
    # sit contiguously on top: the repeated-contact check and the shared pair budget below cover both scans. The
    # count and buffer top are snapshotted for seeding, since the emission below grows both.
    n_added = 0
    i_contact_top = collider_state.n_contacts[i_b]
    if qd.static(seeded):
        while (
            n_added < i_contact_top and collider_state.contact_data.pair_idx[i_contact_top - 1 - n_added, i_b] == i_pair
        ):
            n_added = n_added + 1
    n_prev = n_added
    if (not can_use_sd_reject) or sd_center <= rbound_a:
        # Pass 1: select the n_max deepest spatially-diverse vertices of A by grid SDF pen. The buffer keeps verts at
        # least `diversity_radius` apart in world space: a candidate close to an existing entry replaces that entry
        # only when deeper, otherwise it displaces the weakest entry. The radius starts at the default `tolerance`
        # and is widened by the "needle extent" of A - how much of the long axis sticks out beyond twice the
        # cross-section - divided across n_max slots. A near-cube/sphere/spoon-bowl has zero (clipped) needle extent
        # and stays at `tolerance`, which lets a small curved patch keep its tight cluster of contacts without
        # picking up rim verts whose grad is tilted relative to the surface and would inject torque; a 1:1:16 rod
        # recovers a radius near rbound_a/n_max so the buffer spreads along the long axis instead of collapsing onto
        # the deepest tip and letting the body pivot about a single contact patch.
        ext = geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]
        ext_max = qd.max(qd.max(ext[0], ext[1]), ext[2])
        ext_sum_other = ext[0] + ext[1] + ext[2] - ext_max
        needle_extent = ext_max - gs.qd_float(2.0) * ext_sum_other
        diversity_radius = qd.max(tolerance, needle_extent * gs.qd_float(0.5 / n_max))
        rbound_b_sq = gs.qd_float(0.0)
        b_center_local = dyn_info.geoms.center[i_gb]
        for k in qd.static(range(8)):
            delta_b = geoms_init_AABB[i_gb, k] - b_center_local
            d_sq_b = delta_b.dot(delta_b)
            if d_sq_b > rbound_b_sq:
                rbound_b_sq = d_sq_b
        center_b_world = gu.qd_transform_by_trans_quat(b_center_local, gb_pos, gb_quat)
        top_iv = qd.Vector.zero(gs.qd_int, n_max)
        top_pen = qd.Vector.zero(gs.qd_float, n_max)
        for k in range(n_max):
            top_pen[k] = -gs.qd_float(1e30)
            top_iv[k] = -1
        if qd.static(not seeded):
            # Pull B's world AABB back into A's local frame (its exact enclosing box, padded to absorb the
            # rounding of the two transform paths) and walk only the spatial cells of A overlapping it: a vert
            # outside B's AABB contributes nothing, so far cells are skipped wholesale, without per-vertex work.
            # The exact world-AABB gate below is unchanged, and the cell binning shares the kernel's monotone
            # single-precision cell mapping, so every vert that can pass the gate lies in a visited cell.
            aabb_center = 0.5 * (dyn_state.geoms.aabb_min[i_gb, i_b] + dyn_state.geoms.aabb_max[i_gb, i_b])
            aabb_half_size = 0.5 * (dyn_state.geoms.aabb_max[i_gb, i_b] - dyn_state.geoms.aabb_min[i_gb, i_b])
            box_center = gu.qd_inv_transform_by_trans_quat(aabb_center, ga_pos, ga_quat)
            box_half_size = qd.abs(gu.qd_quat_to_R(ga_quat, EPS)).transpose() @ aabb_half_size
            box_min = box_center - box_half_size
            box_max = box_center + box_half_size
            box_pad = 1e-6 * (1.0 + box_min.norm() + box_max.norm())
            n_cells_per_axis = 8
            spatial_origin = collider_info.verts_spatial_grid.geoms_origin[i_ga]
            spatial_inv_cell_size = collider_info.verts_spatial_grid.geoms_inv_cell_size[i_ga]
            cell_min = qd.floor((box_min - box_pad - spatial_origin) * spatial_inv_cell_size, gs.qd_int)
            cell_max = qd.floor((box_max + box_pad - spatial_origin) * spatial_inv_cell_size, gs.qd_int)
            # The box overlaps the grid only where its cell interval meets [0, n_cells_per_axis - 1] on every
            # axis; a broadphase pair whose pulled-back box clears the grid on some axis (common for rotated
            # meshes) scans nothing. Clamping is applied only inside this guard, so indexing stays within the
            # geom's own cell-range block.
            if (cell_max >= 0).all() and (cell_min <= n_cells_per_axis - 1).all():
                cell_min = qd.max(cell_min, 0)
                cell_max = qd.min(cell_max, n_cells_per_axis - 1)
                for i_cx in range(cell_min[0], cell_max[0] + 1):
                    for i_cy in range(cell_min[1], cell_max[1] + 1):
                        # Cells along z are contiguous in the cell-range layout, so the whole z-run is one vert range.
                        i_cell = (
                            i_ga * (n_cells_per_axis**3 + 1)
                            + (i_cx * n_cells_per_axis + i_cy) * n_cells_per_axis
                            + cell_min[2]
                        )
                        for i_sv in range(
                            collider_info.verts_spatial_grid.cells_vert_start[i_cell],
                            collider_info.verts_spatial_grid.cells_vert_start[i_cell + cell_max[2] - cell_min[2] + 1],
                        ):
                            vertex_pos = gu.qd_transform_by_trans_quat(
                                collider_info.verts_spatial_grid.verts_pos[i_sv], ga_pos, ga_quat
                            )
                            if func_point_in_geom_aabb(i_gb, i_b, vertex_pos, 0.0, dyn_state):
                                is_in_band, sd_v = sdf.sdf_func_world_local_banded(
                                    i_gb, vertex_pos, gb_pos, gb_quat, margin, dyn_info, collider_info
                                )
                                pen_v = -sd_v
                                if is_in_band:
                                    i_v = collider_info.verts_spatial_grid.verts_idx[i_sv]
                                    close_idx = -1
                                    for k in range(n_max):
                                        if close_idx < 0 and top_iv[k] >= 0:
                                            other_pos = gu.qd_transform_by_trans_quat(
                                                dyn_info.verts.init_pos[top_iv[k]], ga_pos, ga_quat
                                            )
                                            if (vertex_pos - other_pos).norm() < diversity_radius:
                                                close_idx = k
                                    if close_idx >= 0:
                                        if pen_v > top_pen[close_idx]:
                                            top_pen[close_idx] = pen_v
                                            top_iv[close_idx] = i_v
                                    else:
                                        weakest_idx = 0
                                        for k in range(1, n_max):
                                            if top_pen[k] < top_pen[weakest_idx]:
                                                weakest_idx = k
                                        if pen_v > top_pen[weakest_idx]:
                                            top_pen[weakest_idx] = pen_v
                                            top_iv[weakest_idx] = i_v
        else:
            # Seeded verification: what the pair's first scan cannot see is a feature of A (rim, edge) crossing one
            # of B's faces BETWEEN B's verts; any such feature lies in B's near-surface band, connected on A's vertex
            # graph to a local SDF minimum. So instead of scanning every A vert, hill-climb down B's SDF from each
            # first-scan contact plus the image of B's center (covers the zero-contact crossing, e.g. a thin blade
            # fully through a plate), and consider only the minimum and its 1-ring. The early stop is a convergence
            # dedup: seeds of the same patch resolve to an already-visited closest vert or climb into a known
            # minimum, so only the first seed of each basin pays for a climb. Cost is O(basins * climb * ring),
            # affordable for every pair including world-anchored meshes with huge vert counts.
            seen_verts = qd.Vector.zero(gs.qd_int, n_max)
            n_seen = 0
            for i_seed in range(n_max + 1):
                i_v_min = -1
                if i_seed == 0:
                    i_v_min = sdf.sdf_func_find_closest_vert(
                        i_ga, i_b, center_b_world, dyn_state, dyn_info, collider_info
                    )
                elif i_seed <= n_prev:
                    seed_pos = collider_state.contact_data.pos[i_contact_top - i_seed, i_b]
                    i_v_min = sdf.sdf_func_find_closest_vert(i_ga, i_b, seed_pos, dyn_state, dyn_info, collider_info)
                if i_v_min >= 0:
                    for k in range(n_max):
                        if k < n_seen and seen_verts[k] == i_v_min:
                            i_v_min = -1
                if i_v_min >= 0:
                    i_v_start = i_v_min
                    pos_min = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v_min], ga_pos, ga_quat)
                    sd_min = sdf.sdf_func_world_local(i_gb, pos_min, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf)
                    i_v_cur = -1
                    while i_v_cur != i_v_min:
                        i_v_cur = i_v_min
                        for i_neighbor_ in range(
                            collider_info.vert_neighbor_start[i_v_cur],
                            collider_info.vert_neighbor_start[i_v_cur] + collider_info.vert_n_neighbors[i_v_cur],
                        ):
                            i_neighbor = collider_info.vert_neighbors[i_neighbor_]
                            pos_neighbor = gu.qd_transform_by_trans_quat(
                                dyn_info.verts.init_pos[i_neighbor], ga_pos, ga_quat
                            )
                            sd_neighbor = sdf.sdf_func_world_local(
                                i_gb, pos_neighbor, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf
                            )
                            # Strict decrease guarantees termination: values strictly descend over a finite vertex
                            # set, so no vertex can be revisited even on numerically flat plateaus.
                            if sd_neighbor < sd_min:
                                i_v_min = i_neighbor
                                sd_min = sd_neighbor
                    # The final minimum is checked against the basins found so far BEFORE this seed's own start is
                    # recorded, so a seed that starts at the minimum still emits.
                    for k in range(n_max):
                        if k < n_seen and seen_verts[k] == i_v_min:
                            i_v_min = -1
                    if n_seen < n_max:
                        seen_verts[n_seen] = i_v_start
                        n_seen = n_seen + 1
                    if i_v_min >= 0 and i_v_min != i_v_start and n_seen < n_max:
                        seen_verts[n_seen] = i_v_min
                        n_seen = n_seen + 1
                if i_v_min >= 0:
                    # Candidates are the minimum itself and its 1-ring (the leading index selects the minimum).
                    # Unlike the full scan, the verification only emits UNAMBIGUOUS penetrations: the first scan
                    # already supplies the near-surface approach band from the other side, and duplicating it here
                    # with grid-noise-level pens and grid-smoothed normals disturbs tuned interfaces (it decouples
                    # the pitch of a threaded nut-bolt pair and injects residual jitter into a settled pile). A
                    # quarter-cell floor keeps grid noise out while a genuine wall crossing (at least half a wall
                    # thickness deep) clears it long before locking.
                    for i_c in range(
                        collider_info.vert_neighbor_start[i_v_min] - 1,
                        collider_info.vert_neighbor_start[i_v_min] + collider_info.vert_n_neighbors[i_v_min],
                    ):
                        i_v = i_v_min
                        if i_c >= collider_info.vert_neighbor_start[i_v_min]:
                            i_v = collider_info.vert_neighbors[i_c]
                        vertex_pos = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v], ga_pos, ga_quat)
                        pen_v = -sdf.sdf_func_world_local(
                            i_gb, vertex_pos, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf
                        )
                        if pen_v > 0.25 * margin:
                            close_idx = -1
                            for k in range(n_max):
                                if close_idx < 0 and top_iv[k] >= 0:
                                    other_pos = gu.qd_transform_by_trans_quat(
                                        dyn_info.verts.init_pos[top_iv[k]], ga_pos, ga_quat
                                    )
                                    if (vertex_pos - other_pos).norm() < diversity_radius:
                                        close_idx = k
                            if close_idx >= 0:
                                if pen_v > top_pen[close_idx]:
                                    top_pen[close_idx] = pen_v
                                    top_iv[close_idx] = i_v
                            else:
                                weakest_idx = 0
                                for k in range(1, n_max):
                                    if top_pen[k] < top_pen[weakest_idx]:
                                        weakest_idx = k
                                if pen_v > top_pen[weakest_idx]:
                                    top_pen[weakest_idx] = pen_v
                                    top_iv[weakest_idx] = i_v

        # Pass 2: emit contacts at the selected vertices. Reference normal is the grid SDF gradient sampled at A's
        # center, which gives a stable per-patch normal whenever A is small relative to B's features. Per-vertex
        # grads agreeing with that reference (positive dot product) are kept verbatim - this is what allows an A
        # wedged at a concave L-corner to expose both axis-aligned face normals (floor + wall). A per-vertex grad
        # that opposes the reference indicates the vertex is closer to B's OPPOSITE surface across a thin feature,
        # i.e. A has partially tunneled through; using its raw grad would push A further through, so we fall back
        # to the reference direction.
        grad_center = sdf.sdf_func_grad_world_local(
            i_gb, center_a_world, gb_pos, gb_quat, dyn_info.geoms, rigid_info, collider_info.sdf, collider_static_config
        )
        normal_center = gu.qd_normalize(grad_center, EPS)
        # When two comparably-sized bodies meet "across" each other (the crossed-thin-rod regime: A's center sits within
        # one A-bounding-radius of B's surface, with both bodies of similar bounding-sphere size), the grid SDF gradient
        # at A's center is poorly conditioned - it lands on B's local radial direction, which is perpendicular to the
        # closing motion. The geometric line from B's origin to A's center is a stronger reference there: it points
        # along the relative-pose offset, which for a head-on closing pair coincides with the closing direction. The
        # per-vertex grad is also locally radial and just as biased, so in this regime we use the closing direction as
        # the FINAL normal rather than letting per-vertex grad override it. The size-ratio gate keeps the existing
        # behavior for one-big-one-small pairs where the SDF grad at A's center is reliable and the closing-direction
        # line is wrong (sphere on a large floor mesh).
        is_closing_regime = qd.abs(sd_center) < rbound_a and rbound_a_sq > 0.25 * rbound_b_sq
        approach_depth_pair = gs.qd_float(0.0)
        closing_normal = qd.Vector.zero(gs.qd_float, 3)
        # Set when A wraps around B so that B passes through A along the center-to-center axis. There the SDF gradient
        # at A's center is ill-conditioned (it sits inside B, away from any surface), so per-vertex grads are trusted
        # directly rather than filtered against that unreliable reference normal.
        is_enclosed_regime = False
        # Set when A and B are two concave shells resting on each other (nested cups/bowls). Both SDF-based normals
        # are unreliable there, so the center-to-center line is used as the contact normal for the whole pair.
        is_axis_normal_regime = False
        if is_closing_regime:
            closing_dir = center_a_world - center_b_world
            if closing_dir.norm() > EPS:
                closing_normal = gu.qd_normalize(closing_dir, EPS)
                # Reject the override when A wraps around B: the center-to-center line is then B's through-axis and
                # the axis "overlap" is the pass-through extent, not a real interpenetration, so resolving along it
                # would eject A. The pose-robust signature is that A's own center lies in a cavity rather than inside
                # A's material (the build-time is_hollow flag): true for such a hollow/annular A, false for the
                # solid A of the genuine crossed-thin-geom regime.
                if dyn_info.geoms.is_hollow[i_ga]:
                    is_closing_regime = False
                    if sd_center < 0.0:
                        # B's material occupies A's center: B passes through A's cavity (a nut around a bolt shaft).
                        # The grad at A's center is ill-conditioned (deep inside B), so trust each vertex's own grad,
                        # which is radial around B and balances across the contact ring.
                        is_enclosed_regime = True
                    else:
                        # A is hollow but its center sits OUTSIDE B: two concave shells resting on each other (nested
                        # cups/bowls). BOTH SDF-based normals are unreliable here - the thin curved wall makes the
                        # per-vertex grads point laterally, and A's center sits in a concave pocket where the grad
                        # sampled at A's center can even be sign-flipped (pointing into the stack). The center-to-center
                        # line is the stacking axis and the robust contact normal, so use it directly (b->a) for every
                        # contact of the pair.
                        normal_center = closing_normal
                        is_axis_normal_regime = True
                else:
                    # Mirror the enclosure test on B: when B's own center sits in a cavity of B (mug, cup, torus),
                    # the center-to-center axis passes through that cavity, so the axis overlap measures pass-through
                    # extent rather than material interpenetration - flooring pen_emit with it catapults a solid A
                    # resting inside or beside the hollow B. The genuine crossed-thin-geom regime this override
                    # targets has both bodies solid (their centers inside their own material), so requiring a
                    # non-hollow B preserves it while restoring the standard SDF-gradient path for hollow B.
                    if dyn_info.geoms.is_hollow[i_gb]:
                        is_closing_regime = False
                        sd_b_center = sdf.sdf_func_world_local(
                            i_ga, center_b_world, ga_pos, ga_quat, dyn_info.geoms, collider_info.sdf
                        )
                        if sd_b_center < 0.0:
                            # A's material occupies B's center: B wraps around A (scanning the bolt of a nut-on-bolt
                            # pair). Mirror of the hollow-A enclosed case: the reference grad at A's center is
                            # ill-conditioned (it sits on B's bore axis where the axisymmetric grad vanishes), so
                            # trust each vertex's own grad, which is radial around B's bore and balances across the
                            # contact ring.
                            is_enclosed_regime = True
                    else:
                        normal_center = closing_normal
                        # Approach depth along the closing axis, measured on the SDFs: bisect the center-to-center
                        # segment for each body's own surface crossing (both centers are inside their own material
                        # here, per the enclosure tests), then overlap = depth_a + depth_b - center distance. This is
                        # the exact material overlap along the axis for both regimes the floor serves: crossed thin
                        # rods (depth = rod radius) and compact solids (depth = support extent). An OBB-projection
                        # bound (half_ext.dot(|d_local|)) is NOT usable here: it overestimates by up to sqrt(3)
                        # off-axis, so two compact solids barely touching along a local diagonal would read a fake
                        # overlap near 0.7 * (rbound_a + rbound_b) and be catapulted apart. It is much larger than
                        # the per-vertex SDF "distance to nearest surface" for crossed thin geoms, where most A verts
                        # sit on A's outer skin one radial gap away from B's lateral surface. A body whose center-side
                        # segment endpoint is still inside the other body (deep overlap) keeps the full segment length
                        # as its depth, degrading gracefully to a conservative overlap.
                        seg_len = closing_dir.norm()
                        depth_b = sdf.sdf_func_ray_exit_distance(
                            i_gb,
                            center_b_world,
                            closing_normal,
                            seg_len,
                            tolerance,
                            gb_pos,
                            gb_quat,
                            dyn_info,
                            collider_info,
                        )
                        depth_a = sdf.sdf_func_ray_exit_distance(
                            i_ga,
                            center_a_world,
                            -closing_normal,
                            seg_len,
                            tolerance,
                            ga_pos,
                            ga_quat,
                            dyn_info,
                            collider_info,
                        )
                        approach_depth_pair = depth_a + depth_b - seg_len
            else:
                is_closing_regime = False
        for k in range(n_max):
            if top_iv[k] >= 0:
                i_v = top_iv[k]
                vertex_pos = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v], ga_pos, ga_quat)
                pen_v = top_pen[k]
                grad_v = sdf.sdf_func_grad_world_local(
                    i_gb,
                    vertex_pos,
                    gb_pos,
                    gb_quat,
                    dyn_info.geoms,
                    rigid_info,
                    collider_info.sdf,
                    collider_static_config,
                )
                # Per-vertex grad magnitude classifies the local grid SDF regime.
                # - Smoothed (|grad| < 0.5): trilinear interpolation has smoothed across a feature. The pen value is
                #   unreliable. Emit a tapered synthetic pen for approach detection.
                # - Edge (0.5 <= |grad| <= 0.9): the vertex sits on a concave seam. The kernel pen is unreliable. Emit
                #   a bounded synthetic pen so the contact registers without dominating the clean face contacts that
                #   wedge the body at the seam.
                # - Clean (|grad| > 0.9): trust the kernel pen.
                grad_norm = grad_v.norm()
                pen_emit = gs.qd_float(0.0)
                contact_pos_v = vertex_pos
                if grad_norm < 0.5:
                    if pen_v > 0.0:
                        pen_emit = qd.min(pen_v, margin)
                    else:
                        pen_emit = synthetic_pen_max * (1.0 + pen_v / margin)
                elif grad_norm > 0.9:
                    if pen_v > 0.0:
                        pen_emit = pen_v
                elif pen_v > 0.0:
                    pen_emit = synthetic_pen_max
                normal_v = normal_center
                if is_enclosed_regime or is_axis_normal_regime:
                    # Two concave shells (nested cups/bowls) or B passing through A's cavity (nut on bolt): the
                    # pair-level reference normal is unreliable (sign-flipped in a concave pocket, or vertical-only so
                    # it cannot resist lateral shear). Orient the contact from A's own exact vertex surface normal
                    # (precomputed): A's face at the contact points into B, so the b->a normal opposes A's outward
                    # normal. On a tilted bowl wall that normal is correctly tilted - it carries both the vertical
                    # support and the radial component that resists a nested stack shearing sideways. When B's grid grad
                    # is well-conditioned, take its axis (it can resolve concave seams a single vertex normal cannot)
                    # but fix its sign from A's normal (the grad's sign inverts once the vertex tunnels past B's thin
                    # wall). When the grad is smoothed (coarse grid across the thin wall), use A's vertex normal
                    # directly rather than the vertical reference, which is what was leaving the side walls unsupported.
                    a_vnormal = gu.qd_normalize(gu.qd_transform_by_quat(dyn_info.verts.init_normal[i_v], ga_quat), EPS)
                    if grad_norm > 0.5:
                        normal_v = gu.qd_normalize(grad_v, EPS)
                        if normal_v.dot(a_vnormal) > 0.0:
                            normal_v = -normal_v
                    else:
                        normal_v = -a_vnormal
                elif (
                    not is_closing_regime
                    and not is_axis_normal_regime
                    and grad_norm > 0.9
                    and grad_v.dot(normal_center) > 0.0
                ):
                    # Trust a per-vertex grad as the contact normal only in the clean band (the same |grad| > 0.9 band
                    # where the kernel pen is trusted): this is what exposes both face normals for an A wedged at a
                    # concave L-corner. In the edge/smoothed bands the per-vertex grad is a partially-interpolated
                    # direction (a box corner straddling a B feature reads a grad tilted tens of degrees off the true
                    # surface normal); there the reference normal sampled at A's center is the more reliable direction.
                    normal_v = gu.qd_normalize(grad_v, EPS)
                # In the closing-direction regime, the SDF "distance to nearest surface" measured at a vertex on A's
                # outer skin is the small radial gap to B's lateral, not the much larger approach depth along the
                # closing axis. Use the geometric approach depth as a floor on pen_emit so the constraint solver sees
                # the actual overlap rather than just the radial gap.
                if is_closing_regime and pen_v > 0.0 and approach_depth_pair > pen_emit:
                    pen_emit = approach_depth_pair
                repeated = False
                for j in range(n_added):
                    idx_prev = collider_state.n_contacts[i_b] - 1 - j
                    if (contact_pos_v - collider_state.contact_data.pos[idx_prev, i_b]).norm() < tolerance:
                        repeated = True
                if not repeated and pen_emit > 0.0:
                    # Snap the contact position onto A's smooth surface when A is a smooth primitive
                    # (SPHERE/ELLIPSOID/CAPSULE). The tessellation vertex sits an O(tessellation chord error) inboard
                    # of the true surface; on a settled static contact that offset becomes a torque arm and drives a
                    # slow tangential drift. The refinement is a no-op for polytope-typed A.
                    contact_pos_v = func_apply_smooth_refinement(
                        i_ga,
                        i_gb,
                        normal_v,
                        pen_emit,
                        contact_pos_v,
                        ga_pos,
                        ga_quat,
                        gb_pos,
                        gb_quat,
                        dyn_info,
                        rigid_config,
                    )
                    if n_added < n_max:
                        func_add_contact(
                            i_ga,
                            i_gb,
                            i_b,
                            i_pair,
                            normal_v,
                            contact_pos_v,
                            pen_emit,
                            dyn_state,
                            collider_state,
                            dyn_info,
                            rigid_info,
                            collider_info,
                            use_atomic=False,
                            errno=errno,
                        )
                        n_added = n_added + 1
                    else:
                        # The pair budget is shared between the two scans: at the cap, a new contact may only
                        # displace the pair's least-penetrating one, keeping the n_max deepest.
                        weakest_idx = -1
                        weakest_pen = pen_emit
                        for j in range(n_max):
                            if j < n_added:
                                idx_prev = collider_state.n_contacts[i_b] - 1 - j
                                if collider_state.contact_data.penetration[idx_prev, i_b] < weakest_pen:
                                    weakest_pen = collider_state.contact_data.penetration[idx_prev, i_b]
                                    weakest_idx = idx_prev
                        if weakest_idx >= 0:
                            func_set_contact(
                                i_ga,
                                i_gb,
                                i_b,
                                weakest_idx,
                                i_pair,
                                normal_v,
                                contact_pos_v,
                                pen_emit,
                                dyn_state,
                                collider_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                errno,
                            )


@qd.func
def func_add_polytope_vertex_contacts_sdf_shell(
    i_ga,
    i_gb,
    i_b,
    i_pair,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    tolerance,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    """
    Sector-aggregated manifold for nested-shell pairs (both bodies hollow, see the dispatcher gate).

    The annular contact of nested shells makes the top-k vertex manifold churn: kept-set swaps, dedup toggles and
    regime flips each relocate a finite force, and a tall stack rectifies that noise into a sideways walk. Here
    every band vert instead accumulates into a fixed azimuthal sector, and each active sector emits ONE aggregated
    contact, so the emitted force field is continuous in pose and the pair budget holds by construction.
    """
    # The bucket table is sized at the pair budget: each bucket emits at most one contact, from whichever scan
    # covers it (see emission).
    n_buckets = qd.static(
        collider_static_config.n_contacts_per_nonconvex_pair if rigid_config.enable_multi_contact else 1
    )
    EPS = rigid_info.EPS[None]
    synthetic_pen_max = 1e-4
    center_a_world = gu.qd_transform_by_trans_quat(dyn_info.geoms.center[i_ga], ga_pos, ga_quat)
    center_b_world = gu.qd_transform_by_trans_quat(dyn_info.geoms.center[i_gb], gb_pos, gb_quat)
    # Shared azimuthal-sector frame for both scan directions: a vertex found by the swapped scan joins the
    # aggregate its azimuth belongs to instead of spawning a contact of its own. Derived from the closing line,
    # which is pose-continuous.
    closing_line = center_a_world - center_b_world
    frame_axis = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    if closing_line.norm() > EPS:
        frame_axis = gu.qd_normalize(closing_line, EPS)
    # Fixed oblique reference direction: any axis-dependent choice relabels every bucket in one step when the
    # axis crosses its switching surface; a constant reference is continuous everywhere except exact alignment,
    # which no physical stacking axis hits.
    e_ref = qd.Vector([0.36, 0.48, 0.8], dt=gs.qd_float)
    t1_ref = gu.qd_normalize(frame_axis.cross(e_ref), EPS)
    t2_ref = frame_axis.cross(t1_ref)
    frame_origin = 0.5 * (center_a_world + center_b_world)

    # Per-bucket accumulators, one table per scan direction. A vertex enters or leaves a bucket with weight
    # pen_emit -> 0, which is what keeps the emitted force field continuous.
    acc_w = qd.Vector.zero(gs.qd_float, n_buckets)
    acc_pos = qd.Matrix.zero(gs.qd_float, n_buckets, 3)
    acc_n = qd.Matrix.zero(gs.qd_float, n_buckets, 3)
    acc_pen_max = qd.Vector.zero(gs.qd_float, n_buckets)
    # The swapped scan fills a shadow table consumed only where the primary scan is blind (see emission):
    # both scans sample the same patch through two different tessellations, and summing them makes their
    # sampling washboards beat - a low-frequency force modulation that ratchets a soft stack sideways.
    acc2_w = qd.Vector.zero(gs.qd_float, n_buckets)
    acc2_pos = qd.Matrix.zero(gs.qd_float, n_buckets, 3)
    acc2_n = qd.Matrix.zero(gs.qd_float, n_buckets, 3)
    acc2_pen_max = qd.Vector.zero(gs.qd_float, n_buckets)
    sup_w = qd.Vector.zero(gs.qd_float, n_buckets)
    sup_pos = qd.Matrix.zero(gs.qd_float, n_buckets, 3)
    sup_n = qd.Matrix.zero(gs.qd_float, n_buckets, 3)

    for i_phase in qd.static(range(2)):
        # Phase 0 scans every vert of A against B's SDF; phase 1 scans B's verts against A's SDF into the shadow
        # table, covering features of B crossing A's faces BETWEEN A's verts. Phase-1 normals are negated onto
        # the phase-0 orientation convention.
        j_ga = i_ga
        j_gb = i_gb
        ja_pos = ga_pos
        ja_quat = ga_quat
        jb_pos = gb_pos
        jb_quat = gb_quat
        phase_sign = gs.qd_float(1.0)
        if qd.static(i_phase == 1):
            j_ga = i_gb
            j_gb = i_ga
            ja_pos = gb_pos
            ja_quat = gb_quat
            jb_pos = ga_pos
            jb_quat = ga_quat
            phase_sign = -1.0
        is_phase_active = True
        if qd.static(i_phase == 1):
            # A plane's handful of far-flung verts carry no contact information.
            is_phase_active = dyn_info.geoms.type[j_ga] != gs.GEOM_TYPE.PLANE
        jb_cell = collider_info.sdf.geoms_info.sdf_cell_size[j_gb]
        margin = qd.min(qd.min(jb_cell[0], jb_cell[1]), jb_cell[2])
        ja_center_local = dyn_info.geoms.center[j_ga]
        rbound_a_sq = gs.qd_float(0.0)
        for k in qd.static(range(8)):
            delta = geoms_init_AABB[j_ga, k] - ja_center_local
            d_sq = delta.dot(delta)
            if d_sq > rbound_a_sq:
                rbound_a_sq = d_sq
        rbound_a = qd.sqrt(rbound_a_sq)
        ja_center_world = gu.qd_transform_by_trans_quat(ja_center_local, ja_pos, ja_quat)
        can_use_sd_reject = (
            dyn_info.geoms.type[j_gb] == gs.GEOM_TYPE.SPHERE or dyn_info.geoms.type[j_gb] == gs.GEOM_TYPE.PLANE
        )
        if not can_use_sd_reject:
            pos_mesh = gu.qd_inv_transform_by_trans_quat(ja_center_world, jb_pos, jb_quat)
            pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[j_gb])
            can_use_sd_reject = not sdf.sdf_func_is_outside_sdf_grid(j_gb, pos_sdf, collider_info.sdf)
        sd_center = sdf.sdf_func_world_local(j_gb, ja_center_world, jb_pos, jb_quat, dyn_info.geoms, collider_info.sdf)
        if is_phase_active and ((not can_use_sd_reject) or sd_center <= rbound_a):
            rbound_b_sq = gs.qd_float(0.0)
            jb_center_local = dyn_info.geoms.center[j_gb]
            for k in qd.static(range(8)):
                delta_b = geoms_init_AABB[j_gb, k] - jb_center_local
                d_sq_b = delta_b.dot(delta_b)
                if d_sq_b > rbound_b_sq:
                    rbound_b_sq = d_sq_b
            jb_center_world = gu.qd_transform_by_trans_quat(jb_center_local, jb_pos, jb_quat)

            grad_center = sdf.sdf_func_grad_world_local_consistent(
                j_gb, ja_center_world, jb_pos, jb_quat, dyn_info, rigid_info, collider_info
            )
            normal_center = gu.qd_normalize(grad_center, EPS)
            # Regime detection and per-vertex pen/normal policy identical to the per-vertex manifold, which
            # documents the rationale of each test.
            is_closing_regime = qd.abs(sd_center) < rbound_a and rbound_a_sq > gs.qd_float(0.25) * rbound_b_sq
            approach_depth_pair = gs.qd_float(0.0)
            closing_normal = qd.Vector.zero(gs.qd_float, 3)
            is_enclosed_regime = False
            is_axis_normal_regime = False
            if is_closing_regime:
                closing_dir = ja_center_world - jb_center_world
                if closing_dir.norm() > EPS:
                    closing_normal = gu.qd_normalize(closing_dir, EPS)
                    if dyn_info.geoms.is_hollow[j_ga]:
                        is_closing_regime = False
                        if sd_center < 0.0:
                            is_enclosed_regime = True
                        else:
                            normal_center = closing_normal
                            is_axis_normal_regime = True
                    elif dyn_info.geoms.is_hollow[j_gb]:
                        is_closing_regime = False
                        sd_b_center = sdf.sdf_func_world_local(
                            j_ga, jb_center_world, ja_pos, ja_quat, dyn_info.geoms, collider_info.sdf
                        )
                        if sd_b_center < 0.0:
                            is_enclosed_regime = True
                    else:
                        normal_center = closing_normal
                        seg_len = closing_dir.norm()
                        depth_b = sdf.sdf_func_ray_exit_distance(
                            j_gb,
                            jb_center_world,
                            closing_normal,
                            seg_len,
                            tolerance,
                            jb_pos,
                            jb_quat,
                            dyn_info,
                            collider_info,
                        )
                        depth_a = sdf.sdf_func_ray_exit_distance(
                            j_ga,
                            ja_center_world,
                            -closing_normal,
                            seg_len,
                            tolerance,
                            ja_pos,
                            ja_quat,
                            dyn_info,
                            collider_info,
                        )
                        approach_depth_pair = depth_a + depth_b - seg_len
                else:
                    is_closing_regime = False
            # Both directions run the same full scan: phase 0 checks A's verts against B's SDF, phase 1
            # B's verts against A's (features of one body crossing the other's faces BETWEEN its verts are
            # only visible to the swapped scan). A full scan keeps membership continuous in pose - verts
            # enter and leave the admission band with weight pen_emit -> 0 - whereas a seeded local search
            # toggles whole member clusters at finite weight whenever a hill-climb seed relocates, and that
            # force modulation ratchets a softly-stacked column sideways.
            for i_v in range(dyn_info.geoms.vert_start[j_ga], dyn_info.geoms.vert_end[j_ga]):
                vertex_pos = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v], ja_pos, ja_quat)
                if func_point_in_geom_aabb(j_gb, i_b, vertex_pos, 0.0, dyn_state):
                    is_in_band, sd_v = sdf.sdf_func_world_local_banded(
                        j_gb, vertex_pos, jb_pos, jb_quat, margin, dyn_info, collider_info
                    )
                    pen_v = -sd_v
                    if is_in_band:
                        grad_v = sdf.sdf_func_grad_world_local_consistent(
                            j_gb, vertex_pos, jb_pos, jb_quat, dyn_info, rigid_info, collider_info
                        )
                        grad_norm = grad_v.norm()
                        pen_emit = gs.qd_float(0.0)
                        is_support = False
                        if grad_norm > 0.9:
                            if pen_v > 0.0:
                                pen_emit = pen_v
                            elif qd.static(i_phase == 0):
                                # Zero-depth support contacts: verts touching within the grid-noise pen scale damp
                                # the closing velocity without exerting static force. Crossed solids are excluded
                                # (damping the far side of the overlap stalls the escape).
                                is_support = pen_v > -synthetic_pen_max and not is_closing_regime
                        elif is_closing_regime or is_axis_normal_regime or is_enclosed_regime:
                            # Sub-cell thin walls are uncertifiable by the lattice, yet these band verts are the
                            # only carriers of the coupling across the wall: keep the synthetic depths, continuous
                            # across pen_v = 0.
                            if grad_norm < 0.5:
                                if pen_v > 0.0:
                                    pen_emit = qd.min(synthetic_pen_max + pen_v, margin)
                                else:
                                    pen_emit = synthetic_pen_max * (1.0 + pen_v / margin)
                            elif pen_v > 0.0:
                                pen_emit = synthetic_pen_max
                        else:
                            # Outside the thin-wall regimes an uncertain (smoothed or seam) pen is not trusted at
                            # all; the vert still damps the closing velocity as a zero-depth support contact.
                            if qd.static(i_phase == 0):
                                is_support = pen_v > -synthetic_pen_max
                        normal_v = normal_center
                        if is_enclosed_regime or is_axis_normal_regime:
                            # Concave shells: the pair-level reference normal cannot resist lateral shear, so orient
                            # from B's grad when well-conditioned (sign fixed from A's exact vertex normal) and from
                            # A's vertex normal when the grad is smoothed across the thin wall.
                            a_vnormal = gu.qd_normalize(
                                gu.qd_transform_by_quat(dyn_info.verts.init_normal[i_v], ja_quat), EPS
                            )
                            if grad_norm > 0.5:
                                normal_v = gu.qd_normalize(grad_v, EPS)
                                if normal_v.dot(a_vnormal) > 0.0:
                                    # Grad agreeing with A's outward normal = tunneled past B's thin wall: no
                                    # support contact there, damping the far side stalls the crossing-escape creep.
                                    normal_v = -normal_v
                                    is_support = False
                            else:
                                normal_v = -a_vnormal
                        elif (
                            not is_closing_regime
                            and not is_axis_normal_regime
                            and grad_norm > 0.9
                            and grad_v.dot(normal_center) > 0.0
                        ):
                            # Trust a per-vertex grad as the contact normal only in the clean band where the kernel
                            # pen is trusted; in the edge/smoothed bands the reference normal at A's center is more
                            # reliable.
                            normal_v = gu.qd_normalize(grad_v, EPS)
                        # In the closing regime the per-vertex pen is only the radial gap to B's lateral; floor it
                        # with the geometric approach depth so the solver sees the actual overlap.
                        if is_closing_regime and pen_v > 0.0 and approach_depth_pair > pen_emit:
                            pen_emit = approach_depth_pair
                        rel_ref = vertex_pos - frame_origin
                        az_ref = qd.atan2(rel_ref.dot(t2_ref), rel_ref.dot(t1_ref))
                        # Two sub-buckets per azimuth sector, split by normal hemisphere: a crossed wall has
                        # penetrating verts on BOTH faces with opposed normals, and merging them would cancel the
                        # aggregate exactly when the crossing needs resistance from each side. The split also keeps
                        # phase-1 members from annihilating phase-0 members of the opposite face.
                        i_bucket = 0
                        i_bucket_2 = 0
                        w_hat = gs.qd_float(1.0)
                        if qd.static(n_buckets >= 2):
                            # Hat-weighted azimuthal binning: a member splits its weight linearly between the two
                            # nearest sector centers, so rotating past a sector boundary moves weight continuously
                            # instead of relocating it in one step - a hard handoff modulates the emitted forces
                            # and, combined with any standing lateral bias, ratchets a softly-stacked column.
                            az_scaled = (az_ref + qd.math.pi) * ((n_buckets // 2) / (2.0 * qd.math.pi))
                            u_hat = az_scaled - 0.5
                            i_sec = qd.cast(qd.floor(u_hat), gs.qd_int)
                            frac_hat = u_hat - qd.cast(i_sec, gs.qd_float)
                            i_sec = qd.raw_mod(i_sec + (n_buckets // 2), n_buckets // 2)
                            i_sec_2 = qd.raw_mod(i_sec + 1, n_buckets // 2)
                            w_hat = 1.0 - frac_hat
                            i_bucket = 2 * i_sec
                            i_bucket_2 = 2 * i_sec_2
                            # Hemisphere sign from whichever direction is well-conditioned for this face: seating
                            # faces align with the closing axis, side walls are perpendicular to it and split on
                            # the local radial instead - otherwise their sub-bucket assignment is FP-noise-driven
                            # and opposite faces can still share a bucket and cancel.
                            normal_folded = phase_sign * normal_v
                            rad_ref = gu.qd_normalize(rel_ref - rel_ref.dot(frame_axis) * frame_axis, EPS)
                            split_dot = normal_folded.dot(frame_axis)
                            rad_dot = normal_folded.dot(rad_ref)
                            if qd.abs(rad_dot) > qd.abs(split_dot):
                                split_dot = rad_dot
                            if split_dot > 0.0:
                                i_bucket = i_bucket + 1
                                i_bucket_2 = i_bucket_2 + 1
                        if pen_emit > 0.0:
                            for k_hat in qd.static(range(2)):
                                i_bucket_k = i_bucket if k_hat == 0 else i_bucket_2
                                w_b = (w_hat if k_hat == 0 else 1.0 - w_hat) * pen_emit
                                if qd.static(i_phase == 0):
                                    acc_w[i_bucket_k] += w_b
                                    for j in qd.static(range(3)):
                                        acc_pos[i_bucket_k, j] += w_b * vertex_pos[j]
                                        acc_n[i_bucket_k, j] += phase_sign * w_b * normal_v[j]
                                else:
                                    acc2_w[i_bucket_k] += w_b
                                    for j in qd.static(range(3)):
                                        acc2_pos[i_bucket_k, j] += w_b * vertex_pos[j]
                                        acc2_n[i_bucket_k, j] += phase_sign * w_b * normal_v[j]
                            # Depth is a per-violation quantity: one physical violation provisions ONE constraint
                            # row's corrective impulse, so a member's pen goes to its primary bucket only while
                            # the hat split shares its weight. A sector fed purely by secondary spill emits with
                            # zero depth (direction and damping only): its positional correction already lives in
                            # the primary row. The handoff at a boundary crossing is smooth because at the swap
                            # both buckets hold nearly the same members, so the depth moves between two
                            # geometrically near-identical rows. Scaling the pen by the hat share instead would
                            # under-provision the primary row (a softened correction destabilizes what it
                            # supports), and writing it into both buckets would double the corrective capacity
                            # for as long as the member sits in the overlap, pulsing the emitted forces at entry
                            # and exit.
                            if qd.static(i_phase == 0):
                                if pen_emit > acc_pen_max[i_bucket]:
                                    acc_pen_max[i_bucket] = pen_emit
                            elif pen_emit > acc2_pen_max[i_bucket]:
                                acc2_pen_max[i_bucket] = pen_emit
                        elif is_support:
                            for k_hat in qd.static(range(2)):
                                i_bucket_k = i_bucket if k_hat == 0 else i_bucket_2
                                w_b = w_hat if k_hat == 0 else 1.0 - w_hat
                                sup_w[i_bucket_k] += w_b
                                for j in qd.static(range(3)):
                                    sup_pos[i_bucket_k, j] += w_b * vertex_pos[j]
                                    sup_n[i_bucket_k, j] += phase_sign * w_b * normal_v[j]
    # Emit one aggregated contact per active bucket. A bucket whose members disagree on direction (the weighted
    # normal nearly cancels) is ambiguous and skipped; buckets holding only zero-depth support verts emit a support
    # contact at their weight-averaged centroid.
    for i_bucket in range(n_buckets):
        emit_pen = gs.qd_float(0.0)
        is_emit = False
        contact_pos = qd.Vector.zero(gs.qd_float, 3)
        normal_c = qd.Vector.zero(gs.qd_float, 3)
        if acc_w[i_bucket] > 0.0:
            inv_w = 1.0 / acc_w[i_bucket]
            contact_pos = qd.Vector(
                [acc_pos[i_bucket, 0] * inv_w, acc_pos[i_bucket, 1] * inv_w, acc_pos[i_bucket, 2] * inv_w],
                dt=gs.qd_float,
            )
            # Rescale the pen-weighted sum to the mean member normal BEFORE normalizing: the sum's magnitude is
            # O(pen), where the eps guard under norm() is no longer negligible and would emit a non-unit normal
            # (a silently softened constraint row). The mean is O(1) by the coherence gate, so the guard is inert.
            normal_c = qd.Vector([acc_n[i_bucket, 0], acc_n[i_bucket, 1], acc_n[i_bucket, 2]], dt=gs.qd_float)
            normal_c = normal_c / acc_w[i_bucket]
            if normal_c.norm() > 0.1:
                # The emitted depth is the deepest member pen: the solver consumes pen as a geometric depth (it
                # drives impedance and error correction, and is reported by get_contacts), so a summed magnitude
                # would inflate a shallow many-vert ring and saturate a genuinely deep crossing. The direction
                # still composes the member forces via sum(pen_i * n_i).
                emit_pen = acc_pen_max[i_bucket]
                normal_c = gu.qd_normalize(normal_c, EPS)
                is_emit = True
        elif acc2_w[i_bucket] > 0.0:
            # The primary scan is blind in this sector (a feature of the other body crosses between its verts):
            # fall back to the swapped scan's aggregate, which is the coverage this scan direction exists for.
            inv_w = 1.0 / acc2_w[i_bucket]
            contact_pos = qd.Vector(
                [acc2_pos[i_bucket, 0] * inv_w, acc2_pos[i_bucket, 1] * inv_w, acc2_pos[i_bucket, 2] * inv_w],
                dt=gs.qd_float,
            )
            normal_c = qd.Vector([acc2_n[i_bucket, 0], acc2_n[i_bucket, 1], acc2_n[i_bucket, 2]], dt=gs.qd_float)
            normal_c = normal_c / acc2_w[i_bucket]
            if normal_c.norm() > 0.1:
                emit_pen = acc2_pen_max[i_bucket]
                normal_c = gu.qd_normalize(normal_c, EPS)
                is_emit = True
        elif sup_w[i_bucket] > 0.0:
            inv_w = 1.0 / sup_w[i_bucket]
            contact_pos = qd.Vector(
                [sup_pos[i_bucket, 0] * inv_w, sup_pos[i_bucket, 1] * inv_w, sup_pos[i_bucket, 2] * inv_w],
                dt=gs.qd_float,
            )
            normal_c = qd.Vector([sup_n[i_bucket, 0], sup_n[i_bucket, 1], sup_n[i_bucket, 2]], dt=gs.qd_float)
            if normal_c.norm() > 0.1 * sup_w[i_bucket]:
                normal_c = gu.qd_normalize(normal_c, EPS)
                is_emit = True
        if is_emit:
            # Snap onto A's smooth surface when A is a smooth primitive; a no-op for polytope-typed A.
            contact_pos = func_apply_smooth_refinement(
                i_ga, i_gb, normal_c, emit_pen, contact_pos, ga_pos, ga_quat, gb_pos, gb_quat, dyn_info, rigid_config
            )
            func_add_contact(
                i_ga,
                i_gb,
                i_b,
                i_pair,
                normal_c,
                contact_pos,
                emit_pen,
                dyn_state,
                collider_state,
                dyn_info,
                rigid_info,
                collider_info,
                use_atomic=False,
                errno=errno,
            )


@qd.func
def func_contact_vertex_sdf(
    i_ga,
    i_gb,
    i_b,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
):
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    for i_v in range(dyn_info.geoms.vert_start[i_ga], dyn_info.geoms.vert_end[i_ga]):
        vertex_pos = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v], ga_pos, ga_quat)
        if func_point_in_geom_aabb(i_gb, i_b, vertex_pos, 0.0, dyn_state):
            new_penetration = -sdf.sdf_func_world_local(
                i_gb, vertex_pos, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf
            )
            if new_penetration > penetration:
                is_col = True
                contact_pos = vertex_pos
                penetration = new_penetration

    if is_col:
        # Sample B's SDF gradient at A's geometric center when A is a convex geom (primitive or convexified mesh). For
        # any convex shape the centroid of its vertices is by construction inside its hull, so the sampling point sits
        # deep inside A's surface and several SDF cells away from B's zero-isosurface in this post-penetration
        # configuration; the tri-linear interpolation of B's gradient there is dominated by the smooth interior and not
        # by the cell-aligned noise that contaminates the gradient evaluated at the deepest iterated vertex. Without
        # this override the contact normal acquires a tangential component that flips frame to frame as the deepest
        # vertex migrates between neighbouring tessellated points, driving spheres at rest into a ~m/s vertical jitter
        # and a slow lateral drift. We use `geoms_info.center` - already populated and used by MPR for the same
        # purpose - rather than A's frame origin so the property generalises beyond primitives whose modeller happened
        # to centre the pivot.
        center_a = gu.qd_transform_by_trans_quat(dyn_info.geoms.center[i_ga], ga_pos, ga_quat)
        normal_sample = center_a if dyn_info.geoms.is_convex[i_ga] else contact_pos
        normal = sdf.sdf_func_normal_world_local(
            i_gb, normal_sample, gb_pos, gb_quat, dyn_info.geoms, rigid_info, collider_info.sdf, collider_static_config
        )

        # Shift contact_pos from the deepest vertex (interior side) by half the penetration along the outward normal
        # so it lands at the midpoint between A's surface and B's surface.
        contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos


@qd.func
def func_contact_nonconvex_convex_sdf(
    i_ga,
    i_gb,
    i_b,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
):
    # Contact between geoms in a mixed-convexity pair where A is the smaller-AABB side (whether convex or not) and B
    # is the other side, used in lieu of the symmetric two-pass dispatch of func_narrow_phase_nonconvex_vs_nonterrain.
    # A's vertex enumeration alone finds the deepest penetration into B - in a mixed pair the smaller geom is always
    # the one penetrating the larger - and skipping the other side's O(n_verts) scan kills the cost that dominates
    # when the larger side is a static mesh with many tens of thousands of vertices. A bounding-sphere-vs-SDF coarse
    # reject at A's centre mirrors MPR's "if no support certifies overlap, exit" pattern: every point of A lies within
    # rbound_a of geoms_info.center[i_ga], so when B's SDF at A's centre exceeds rbound_a no point of A can reach B's
    # zero level set and the entire vertex scan is skipped.
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    # rbound_a is the radius of the smallest sphere centred at geoms_info.center[i_ga] that contains A's AABB.
    # func_compute_geom_rbound returns half the AABB diagonal, which is correct only when the geom centre coincides
    # with the AABB midpoint - true for primitives but not for arbitrary decomposed convex mesh pieces, whose centroid
    # (used as geoms_info.center by MPR) is offset. Iterating the 8 AABB corners gives the tight per-centre bound and
    # ensures the reject test never discards a pair whose vertices could still be inside B.
    center_local = dyn_info.geoms.center[i_ga]
    rbound_a_sq = gs.qd_float(0.0)
    for k in qd.static(range(8)):
        delta = geoms_init_AABB[i_ga, k] - center_local
        d_sq = delta.dot(delta)
        if d_sq > rbound_a_sq:
            rbound_a_sq = d_sq
    rbound_a = qd.sqrt(rbound_a_sq)

    # The reject is only valid when the SDF value at A's centre is the true distance to B's surface. For SPHERE and
    # PLANE B that is the analytical branch of sdf_func_world_local - exact everywhere. For grid-based SDFs the query
    # returns the true trilinear interpolation only when the point falls inside the SDF grid; outside the grid
    # sdf_func_world_local routes to sdf_func_proxy_sdf, which returns ||P - grid_center||_world + sdf_max. That proxy
    # is NOT a one-sided bound on true_sdf: with R_mesh the mesh's bounding radius from grid_center, the gap
    # proxy - true_sdf lies in [sdf_max - R_mesh, sdf_max + R_mesh]. Typical grids use a 20% padding so
    # sdf_max ~ 0.2*R_mesh < R_mesh, and the proxy can overestimate true_sdf by up to ~1.2*R_mesh - large enough to
    # satisfy proxy > rbound_a while true_sdf < rbound_a, silently missing a real contact. Falling through to the
    # vertex scan when outside the grid costs at most one extra func_contact_vertex_sdf pass on the smaller side and
    # keeps the reject correct in all cases.
    center_a_world = gu.qd_transform_by_trans_quat(center_local, ga_pos, ga_quat)
    # SPHERE B has an analytical SDF; grid-based B is only safe when the query point falls inside the grid.
    can_use_sd_reject = dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.SPHERE
    if not can_use_sd_reject:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(center_a_world, gb_pos, gb_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[i_gb])
        can_use_sd_reject = not sdf.sdf_func_is_outside_sdf_grid(i_gb, pos_sdf, collider_info.sdf)
    sd_center = sdf.sdf_func_world_local(i_gb, center_a_world, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf)

    if (not can_use_sd_reject) or sd_center <= rbound_a:
        is_col, normal, penetration, contact_pos = func_contact_vertex_sdf(
            i_ga,
            i_gb,
            i_b,
            ga_pos,
            ga_quat,
            gb_pos,
            gb_quat,
            dyn_state,
            dyn_info,
            rigid_info,
            collider_info,
            collider_static_config,
        )

    return is_col, normal, penetration, contact_pos


@qd.func
def func_contact_convex_convex_sdf(
    i_ga,
    i_gb,
    i_b,
    i_va_ws,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    enable_edge_detection_fallback: qd.template(),
):
    EPS = rigid_info.EPS[None]

    gb_vert_start = dyn_info.geoms.vert_start[i_gb]
    ga_pos = dyn_state.geoms.pos[i_ga, i_b]
    ga_quat = dyn_state.geoms.quat[i_ga, i_b]
    gb_pos = dyn_state.geoms.pos[i_gb, i_b]
    gb_quat = dyn_state.geoms.quat[i_gb, i_b]

    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    i_va = i_va_ws
    if i_va == -1:
        # start traversing on the vertex graph with a smart initial vertex
        pos_vb = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[gb_vert_start], gb_pos, gb_quat)
        i_va = sdf.sdf_func_find_closest_vert(i_ga, i_b, pos_vb, dyn_state, dyn_info, collider_info)
    i_v_closest = i_va
    pos_v_closest = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v_closest], ga_pos, ga_quat)
    sd_v_closest = sdf.sdf_func_world(i_gb, i_b, pos_v_closest, dyn_state.geoms, dyn_info.geoms, collider_info.sdf)

    while True:
        for i_neighbor_ in range(
            collider_info.vert_neighbor_start[i_va],
            collider_info.vert_neighbor_start[i_va] + collider_info.vert_n_neighbors[i_va],
        ):
            i_neighbor = collider_info.vert_neighbors[i_neighbor_]
            pos_neighbor = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_neighbor], ga_pos, ga_quat)
            sd_neighbor = sdf.sdf_func_world(
                i_gb, i_b, pos_neighbor, dyn_state.geoms, dyn_info.geoms, collider_info.sdf
            )
            if sd_neighbor < sd_v_closest - 1e-5:  # 1e-5 (0.01mm) to avoid endless loop due to numerical instability
                i_v_closest = i_neighbor
                sd_v_closest = sd_neighbor
                pos_v_closest = pos_neighbor

        if i_v_closest == i_va:  # no better neighbor
            break
        else:
            i_va = i_v_closest

    # i_va is the deepest vertex
    pos_a = pos_v_closest
    if sd_v_closest < 0.0:
        is_col = True
        normal = sdf.sdf_func_normal_world(
            i_gb, i_b, pos_a, dyn_state.geoms, dyn_info.geoms, rigid_info, collider_info.sdf, collider_static_config
        )
        penetration = -sd_v_closest
        contact_pos = pos_a
    elif enable_edge_detection_fallback:  # check edge surrounding it
        for i_neighbor_ in range(
            collider_info.vert_neighbor_start[i_va],
            collider_info.vert_neighbor_start[i_va] + collider_info.vert_n_neighbors[i_va],
        ):
            i_neighbor = collider_info.vert_neighbors[i_neighbor_]

            p_0 = pos_v_closest
            p_1 = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_neighbor], ga_pos, ga_quat)
            vec_01 = gu.qd_normalize(p_1 - p_0, EPS)

            sdf_grad_0_b = sdf.sdf_func_grad_world(
                i_gb, i_b, p_0, dyn_state, dyn_info, rigid_info, collider_info, collider_static_config
            )
            sdf_grad_1_b = sdf.sdf_func_grad_world(
                i_gb, i_b, p_1, dyn_state, dyn_info, rigid_info, collider_info, collider_static_config
            )

            # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
            sdf_grad_0_a = sdf.sdf_func_grad_world(
                i_ga, i_b, p_0, dyn_state, dyn_info, rigid_info, collider_info, collider_static_config
            )
            sdf_grad_1_a = sdf.sdf_func_grad_world(
                i_ga, i_b, p_1, dyn_state, dyn_info, rigid_info, collider_info, collider_static_config
            )
            normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
            normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

            if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
                # check if closest point is between the two points
                if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:
                    cur_length = (p_1 - p_0).norm()
                    ga_sdf_cell_size_vec = collider_info.sdf.geoms_info.sdf_cell_size[i_ga]
                    ga_sdf_cell_size = qd.min(
                        qd.min(ga_sdf_cell_size_vec[0], ga_sdf_cell_size_vec[1]), ga_sdf_cell_size_vec[2]
                    )
                    while cur_length > ga_sdf_cell_size:
                        p_mid = 0.5 * (p_0 + p_1)
                        side = sdf.sdf_func_grad_world(
                            i_gb, i_b, p_mid, dyn_state, dyn_info, rigid_info, collider_info, collider_static_config
                        ).dot(vec_01)
                        if side < 0:
                            p_0 = p_mid
                        else:
                            p_1 = p_mid

                        cur_length = 0.5 * cur_length

                    p = 0.5 * (p_0 + p_1)
                    new_penetration = -sdf.sdf_func_world(
                        i_gb, i_b, p, dyn_state.geoms, dyn_info.geoms, collider_info.sdf
                    )

                    if new_penetration > 0.0:
                        is_col = True
                        normal = sdf.sdf_func_normal_world(
                            i_gb,
                            i_b,
                            p,
                            dyn_state.geoms,
                            dyn_info.geoms,
                            rigid_info,
                            collider_info.sdf,
                            collider_static_config,
                        )
                        contact_pos = p
                        penetration = new_penetration
                        break

    # The contact point must be offsetted by half the penetration depth, for consistency with MPR
    contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos, i_va


@qd.func
def func_contact_mpr_terrain(
    i_ga,
    i_gb,
    i_b,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    ga_pos, ga_quat = dyn_state.geoms.pos[i_ga, i_b], dyn_state.geoms.quat[i_ga, i_b]
    gb_pos, gb_quat = dyn_state.geoms.pos[i_gb, i_b], dyn_state.geoms.quat[i_gb, i_b]
    margin = gs.qd_float(0.0)
    EPS = rigid_info.EPS[None]

    multi_contact = (
        qd.static(rigid_config.enable_multi_contact)
        and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.SPHERE
        and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
    )

    is_return = False
    tolerance = collider_info.mc_tolerance[None] * func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)

    if not is_return:
        # Transform to terrain's frame (using local variables, not modifying global state)
        ga_pos_terrain_frame, ga_quat_terrain_frame = gu.qd_transform_pos_quat_by_trans_quat(
            ga_pos - gb_pos, ga_quat, qd.Vector.zero(gs.qd_float, 3), gu.qd_inv_quat(gb_quat)
        )
        gb_pos_terrain_frame = qd.Vector.zero(gs.qd_float, 3)
        gb_quat_terrain_frame = gu.qd_identity_quat()

        for i_axis, i_m in qd.ndrange(3, 2):
            direction = qd.Vector.zero(gs.qd_float, 3)
            if i_m == 0:
                direction[i_axis] = 1.0
            else:
                direction[i_axis] = -1.0
            v1 = mpr.support_driver(
                i_ga,
                i_b,
                direction,
                ga_pos_terrain_frame,
                ga_quat_terrain_frame,
                collider_state,
                dyn_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )
            collider_state.xyz_max_min[3 * i_m + i_axis, i_b] = v1[i_axis]

        for i in qd.static(range(3)):
            collider_state.prism[i, i_b][2] = collider_info.terrain_xyz_maxmin[5]

            if (
                collider_info.terrain_xyz_maxmin[i] < collider_state.xyz_max_min[i + 3, i_b] - margin
                or collider_info.terrain_xyz_maxmin[i + 3] > collider_state.xyz_max_min[i, i_b] + margin
            ):
                is_return = True

        if not is_return:
            sh = collider_info.terrain_scale[0]
            r_min = gs.qd_int(qd.floor((collider_state.xyz_max_min[3, i_b] - collider_info.terrain_xyz_maxmin[3]) / sh))
            r_max = gs.qd_int(qd.ceil((collider_state.xyz_max_min[0, i_b] - collider_info.terrain_xyz_maxmin[3]) / sh))
            c_min = gs.qd_int(qd.floor((collider_state.xyz_max_min[4, i_b] - collider_info.terrain_xyz_maxmin[4]) / sh))
            c_max = gs.qd_int(qd.ceil((collider_state.xyz_max_min[1, i_b] - collider_info.terrain_xyz_maxmin[4]) / sh))

            r_min = qd.max(0, r_min)
            c_min = qd.max(0, c_min)
            r_max = qd.min(collider_info.terrain_rc[0] - 1, r_max)
            c_max = qd.min(collider_info.terrain_rc[1] - 1, c_max)

            # Multi-contact perturbation state. The initial detection (i_detection == 0) finds the first face-vs-face
            # contact; subsequent passes rotate geom A by a small angle about an axis orthogonal to that contact's
            # normal, which tips the box face onto a different corner. After undoing the rotation each perturbed contact
            # lands at a different corner of the contact patch, stabilizing a flat box on a triangulated cell.
            # Perturbation is only applied when the initial contact is face-vs-face (snap fired); cliff-edge contacts
            # are kept as a single MPR contact.
            is_col_0 = False
            face_face_0 = False
            contact_pos_0 = qd.Vector.zero(gs.qd_float, 3)
            normal_0 = qd.Vector.zero(gs.qd_float, 3)
            axis_0 = qd.Vector.zero(gs.qd_float, 3)
            axis_1 = qd.Vector.zero(gs.qd_float, 3)
            qrot = qd.Vector.zero(gs.qd_float, 4)
            n_con = 0
            n_detections = 5 if multi_contact else 1

            ga_pos_tf = ga_pos_terrain_frame
            ga_quat_tf = ga_quat_terrain_frame
            for i_detection in range(n_detections):
                if i_detection > 0 and not face_face_0:
                    break
                if i_detection > 0:
                    axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                    qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)
                    ga_pos_curr, ga_quat_curr = func_rotate_frame(ga_pos, ga_quat, contact_pos_0, qrot)
                    ga_pos_tf, ga_quat_tf = gu.qd_transform_pos_quat_by_trans_quat(
                        ga_pos_curr - gb_pos, ga_quat_curr, qd.Vector.zero(gs.qd_float, 3), gu.qd_inv_quat(gb_quat)
                    )
                center_a = gu.qd_transform_by_trans_quat(dyn_info.geoms.center[i_ga], ga_pos_tf, ga_quat_tf)

                for r in range(r_min, r_max):
                    nvert = 0
                    for c in range(c_min, c_max + 1):
                        for i in range(2):
                            if n_con < qd.static(collider_static_config.n_contacts_per_convex_pair):
                                nvert = nvert + 1
                                func_add_prism_vert(
                                    i_b,
                                    sh * (r + i) + collider_info.terrain_xyz_maxmin[3],
                                    sh * c + collider_info.terrain_xyz_maxmin[4],
                                    collider_info.terrain_hf[r + i, c] + margin,
                                    collider_state,
                                )
                                if nvert > 2 and (
                                    collider_state.prism[3, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                    or collider_state.prism[4, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                    or collider_state.prism[5, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                ):
                                    center_b = qd.Vector.zero(gs.qd_float, 3)
                                    for i_p in qd.static(range(6)):
                                        center_b = center_b + collider_state.prism[i_p, i_b]
                                    center_b = center_b / 6.0

                                    is_col, normal, penetration, contact_pos = mpr.func_mpr_contact_from_centers(
                                        i_ga,
                                        i_gb,
                                        i_b,
                                        center_a,
                                        center_b,
                                        ga_pos_tf,
                                        ga_quat_tf,
                                        gb_pos_terrain_frame,
                                        gb_quat_terrain_frame,
                                        geoms_init_AABB,
                                        collider_state,
                                        mpr_state,
                                        dyn_info,
                                        collider_info,
                                        rigid_config,
                                        collider_static_config,
                                    )
                                    if is_col:
                                        snap_fired = False
                                        face_face = False
                                        # Snap normal to the prism's top face normal when MPR's reported normal is
                                        # already close to it. Cell boundaries on a SMOOTH heightfield are
                                        # discretization artefacts, not physical edges, and MPR's polytope-edge radial
                                        # normal there picks up a small position-dependent bias relative to the exact
                                        # face normal. Only snap when the bias is small (dot > 0.95) so that contacts on
                                        # real cliff edges, where MPR's normal is genuinely far from any single cell's
                                        # top face normal, keep MPR's result.
                                        e1 = collider_state.prism[4, i_b] - collider_state.prism[3, i_b]
                                        e2 = collider_state.prism[5, i_b] - collider_state.prism[3, i_b]
                                        top_face_normal = e1.cross(e2).normalized()
                                        if top_face_normal[2] < 0.0:
                                            top_face_normal = -top_face_normal
                                        if top_face_normal.dot(normal) > 0.95:
                                            normal = top_face_normal
                                            snap_fired = True
                                            # Cell is essentially horizontal in the terrain's local frame (within ~8
                                            # deg). Independent of the terrain's world orientation.
                                            face_face = top_face_normal[2] > 0.99

                                        normal = gu.qd_transform_by_quat(normal, gb_quat)
                                        contact_pos = gu.qd_transform_by_quat(contact_pos, gb_quat)
                                        contact_pos = contact_pos + gb_pos

                                        # No perturbation correction: the perturbation magnitude (mc_perturbation) is
                                        # small enough that the perturbed contact_pos stays close to the unperturbed
                                        # contact patch. The deduplication tolerance downstream picks the unique corner
                                        # contacts and the constraint solver tolerates the residual offset.

                                        contact_pos = func_apply_smooth_refinement(
                                            i_ga,
                                            i_gb,
                                            normal,
                                            penetration,
                                            contact_pos,
                                            dyn_state.geoms.pos[i_ga, i_b],
                                            dyn_state.geoms.quat[i_ga, i_b],
                                            dyn_state.geoms.pos[i_gb, i_b],
                                            dyn_state.geoms.quat[i_gb, i_b],
                                            dyn_info,
                                            rigid_config,
                                        )

                                        valid = True
                                        i_c = collider_state.n_contacts[i_b]
                                        for j in range(n_con):
                                            if (
                                                contact_pos - collider_state.contact_data.pos[i_c - j - 1, i_b]
                                            ).norm() < tolerance:
                                                valid = False
                                                break
                                        if valid and i_detection > 0 and not snap_fired:
                                            # Perturbed contacts are only kept when they still describe a face-vs-face
                                            # contact on the same horizontal cell face. A perturbed corner that landed
                                            # against a cliff wall (snap did not fire) is a fictitious contact.
                                            valid = False

                                        if valid:
                                            i_pair = collider_info.collision_pair_idx[
                                                (i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)
                                            ]
                                            func_add_contact(
                                                i_ga,
                                                i_gb,
                                                i_b,
                                                i_pair,
                                                normal,
                                                contact_pos,
                                                penetration,
                                                dyn_state,
                                                collider_state,
                                                dyn_info,
                                                rigid_info,
                                                collider_info,
                                                use_atomic=False,
                                                errno=errno,
                                            )
                                            n_con = n_con + 1
                                            if i_detection == 0 and not is_col_0:
                                                is_col_0 = True
                                                # Perturbation only applies on essentially-horizontal cell faces (in the
                                                # terrain's local frame, within ~8 deg of the heightfield +z): on
                                                # steeper slopes the corner contacts generated by perturbation map to
                                                # different parts of neighbouring cells and create constraint-solver
                                                # oscillation.
                                                face_face_0 = face_face
                                                contact_pos_0 = contact_pos
                                                normal_0 = normal
                                                if face_face_0:
                                                    axis_0, axis_1 = func_contact_orthogonals(
                                                        i_ga,
                                                        i_gb,
                                                        i_b,
                                                        normal_0,
                                                        geoms_init_AABB,
                                                        dyn_state,
                                                        dyn_info,
                                                        rigid_info,
                                                        rigid_config,
                                                    )


@qd.func
def func_add_prism_vert(i_b, x, y, z, collider_state: array_class.ColliderState):
    collider_state.prism[0, i_b] = collider_state.prism[1, i_b]
    collider_state.prism[1, i_b] = collider_state.prism[2, i_b]
    collider_state.prism[3, i_b] = collider_state.prism[4, i_b]
    collider_state.prism[4, i_b] = collider_state.prism[5, i_b]

    collider_state.prism[2, i_b][0] = x
    collider_state.prism[5, i_b][0] = x
    collider_state.prism[2, i_b][1] = y
    collider_state.prism[5, i_b][1] = y
    collider_state.prism[5, i_b][2] = z


@qd.func
def func_recompute_perturbed_contact(
    i_ga,
    i_gb,
    i_scratch,
    normal: qd.types.vector(3),
    penetration,
    contact_pos: qd.types.vector(3),
    normal_0: qd.types.vector(3),
    contact_pos_0: qd.types.vector(3),
    qrot: qd.types.vector(4),
    ga_pos_original: qd.types.vector(3),
    ga_quat_original: qd.types.vector(4),
    gb_pos_original: qd.types.vector(3),
    gb_quat_original: qd.types.vector(4),
    used_gjk,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
):
    """
    Recompute a perturbed multi-contact point exactly, by un-rotating the portal the perturbed detection found.

    Multi-contact spreads contact points by detecting collisions on slightly rotated copies of the two geometries.
    The contact normal and penetration must be recovered for the unperturbed configuration. The contact normal is a
    property of the Minkowski difference (both geometries), so neither geom's surface normal alone captures it.
    Instead the MPR portal - the triangle of support-point pairs bounding the contact - is un-rotated back to the
    unperturbed pose (each support point by the inverse of its own geom's perturbation), and the face normal of the
    resulting Minkowski triangle gives the exact contact normal, with the penetration as the portal's distance to the
    Minkowski origin. The position comes last, from the recovered normal and penetration, since only those say how far
    apart the two surfaces are.
    """
    # qrot is applied to geom A and its inverse to geom B; precompute the rotation matrix once (R for qrot, its
    # transpose for the inverse) and reuse it for every un-rotation below instead of re-deriving it per call.
    R = gu.qd_quat_to_R(qrot, rigid_info.EPS[None])
    R_inv = R.transpose()
    contact_point_a = R_inv @ ((contact_pos - 0.5 * penetration * normal) - contact_pos_0) + contact_pos_0
    contact_point_b = R @ ((contact_pos + 0.5 * penetration * normal) - contact_pos_0) + contact_pos_0

    # The unperturbed contact normal is recovered per detection method, using only the data that method exposes. The
    # multi-contact perturbation is symmetric (geom A by +qrot, geom B by -qrot, over +/- axis pairs), so methods that
    # keep the perturbed normal still yield an unbiased contact set: the per-contact tilts cancel in aggregate (no
    # drift), and the pruning kernel's mean normal averages them back to the true normal (the patch stays coplanar).
    #  - PLANE: the normal is rigid to the plane geom (geom A, rotated by qrot), so un-rotating it by qrot is exact.
    #  - CAPSULE-CAPSULE: an analytic closest-segment contact, with no portal or witness pair; the only available
    #    correction is the first-order twist of the perturbed normal back towards the unperturbed one.
    #  - MPR: it exposes no witness pair, only a portal; the un-rotated portal support simplex gives the exact normal
    #    as the Minkowski-triangle face normal (vertex-face / edge-edge contacts included).
    #  - GJK: same construction from the EPA polytope face nearest to the origin (its three support pairs).
    # is_exact reports whether the recovered penetration is exact (a true contact depth) rather than an approximate
    # first-order value. The caller uses it to pick the contact-acceptance threshold: an exact penetration can be
    # discarded as soon as it is non-positive (fictitious contact), while an approximate one keeps a negative tolerance.
    is_exact = False
    needs_twist = False
    if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE:
        normal = R_inv @ normal
        penetration = normal.dot(contact_point_b - contact_point_a)
        is_exact = True
    elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
        # Analytic closest-segment contact: no portal or witness pair (and its portal_status / nearest_face are stale,
        # since it runs neither MPR nor GJK), so the only correction available is the first-order twist.
        needs_twist = True
    elif used_gjk and gjk_state.nearest_face[i_scratch] < 0:
        # Shallow GJK contact (no EPA polytope was built): no support face, but the perturbed witness delta is the
        # perturbed normal by construction, so keep it; the +/- symmetry keeps the contact set unbiased in aggregate.
        pass
    elif not used_gjk and mpr_state.portal_status[i_scratch] < PORTAL_STATUS.LOWER_BOUND:
        # MPR left no trustworthy refined contact-face portal (degenerate touch/segment path, or the origin projects
        # far outside the portal); reconstructing from it would yield a spurious edge/corner normal. A LOWER_BOUND
        # portal keeps the reconstruction: its support triangle still spans the contact face, only its depth is
        # inexact.
        needs_twist = True
    else:
        # Support pairs of the contact face: the MPR portal (indices 1-3), or the GJK EPA face nearest to the origin.
        a1 = mpr_state.simplex_support.v1[1, i_scratch]
        b1 = mpr_state.simplex_support.v2[1, i_scratch]
        a2 = mpr_state.simplex_support.v1[2, i_scratch]
        b2 = mpr_state.simplex_support.v2[2, i_scratch]
        a3 = mpr_state.simplex_support.v1[3, i_scratch]
        b3 = mpr_state.simplex_support.v2[3, i_scratch]
        if used_gjk:
            i_f = gjk_state.nearest_face[i_scratch]
            iv1 = gjk_state.polytope_faces.verts_idx[i_scratch, i_f][0]
            iv2 = gjk_state.polytope_faces.verts_idx[i_scratch, i_f][1]
            iv3 = gjk_state.polytope_faces.verts_idx[i_scratch, i_f][2]
            a1 = gjk_state.polytope_verts.obj1[i_scratch, iv1]
            b1 = gjk_state.polytope_verts.obj2[i_scratch, iv1]
            a2 = gjk_state.polytope_verts.obj1[i_scratch, iv2]
            b2 = gjk_state.polytope_verts.obj2[i_scratch, iv2]
            a3 = gjk_state.polytope_verts.obj1[i_scratch, iv3]
            b3 = gjk_state.polytope_verts.obj2[i_scratch, iv3]
        # contact_pos_0 cancels in the edge differences, so the face normal needs only support-point deltas.
        edge1 = R_inv @ (a2 - a1) - R @ (b2 - b1)
        edge2 = R_inv @ (a3 - a1) - R @ (b3 - b1)
        portal_normal = edge1.cross(edge2)
        portal_norm_sqr = portal_normal.norm_sqr()
        # The face normal is reliable only when the support triangle is well-conditioned. For a nearly coplanar
        # contact (e.g. flat box-on-box) the support points can be almost collinear, making the face normal
        # numerically unstable; fall back to the twist there. Compared squared to avoid the edge-length square roots.
        if portal_norm_sqr > 0.01 * edge1.norm_sqr() * edge2.norm_sqr():
            normal = portal_normal / qd.sqrt(portal_norm_sqr)
            if normal.dot(normal_0) < 0.0:
                normal = -normal
            # The depth read off the portal is exact only when the origin projects inside it (Theorem 4.2); a
            # LOWER_BOUND portal keeps the reconstructed normal but leaves the depth to the first-order witness
            # separation below, and its candidate to the lenient acceptance.
            is_exact = used_gjk or mpr_state.portal_status[i_scratch] == PORTAL_STATUS.EXACT
            if is_exact:
                # m1 (one un-rotated Minkowski support point on the face) is only needed for the exact penetration.
                m1 = R_inv @ (a1 - contact_pos_0) - R @ (b1 - contact_pos_0)
                penetration = -normal.dot(m1)
        else:
            needs_twist = True

    # Single first-order fallback for every case that could not recover an exact normal (analytic capsule-capsule,
    # degenerate MPR, near-collinear portal). Computed once, and only when actually needed.
    if needs_twist:
        mc_perturbation = collider_info.mc_perturbation[None]
        twist_rotvec = qd.math.clamp(normal.cross(normal_0), -mc_perturbation, mc_perturbation)
        normal = normal + twist_rotvec.cross(normal)
    if not is_exact:
        penetration = normal.dot(contact_point_b - contact_point_a)

    # Each un-rotated witness point is an exact material point of its own geom, but the two un-rotations turn in
    # opposite directions and so slide the pair tangentially past each other, by twice the perturbation angle times
    # their distance from contact 0. Only their separation along the recovered normal carries contact information, so
    # the position steps back from geom B's witness by half of it, landing on the feature the perturbation found.
    # Averaging the two witnesses instead subtracts the whole separation, sliding the contact off that feature by an
    # amount that grows with the square of the perturbation angle, which no precision reaches.
    contact_pos = contact_point_b - 0.5 * penetration * normal

    # Apply the smooth-primitive position reconstruction here, after the perturbation has been reverted, so it uses the
    # final (corrected) normal and the unperturbed pose - the canonical state the solver stores.
    contact_pos = func_apply_smooth_refinement(
        i_ga,
        i_gb,
        normal,
        penetration,
        contact_pos,
        ga_pos_original,
        ga_quat_original,
        gb_pos_original,
        gb_quat_original,
        dyn_info,
        rigid_config,
    )
    return normal, penetration, contact_pos, is_exact


@qd.func
def func_prefer_gjk_refinement(
    i_pair,
    i_b,
    portal_status,
    penetration,
    geom_pair_scale,
    tolerance,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
):
    """Whether GJK should refine the contact MPR resolved, decided on what that contact's depth is worth.

    Nothing guarantees the portal MPR converged on is the globally minimal one, and the risk it is not grows with the
    overlap, so the depth is compared against a threshold bounded inside [tolerance, overlap_ratio * pair scale]: a
    depth at or below the manifold tolerance never refines - it carries nothing a refinement could act on, and the
    refinement is the arm that can fail to return a contact at all - while a depth past the overlap cap always does.
    An EXACT portal recovers the true depth (Thm 4.2) and earns the larger cap once a warm cache corroborates it; the
    warm cache also tightens the threshold to the jump over the cached depth, so a depth that grew by more than
    mpr_to_gjk_penetration_ratio in one step refines. A cold pair whose portal only bounds the depth gates on the
    tolerance, as does a depth MPR could not stand behind at all - unconverged, or read off an extrapolation of the
    portal's plane - whatever the cache says.
    """
    cached_penetration = collider_state.contact_cache.penetration[i_pair, i_b]
    overlap_ratio = collider_info.mpr_to_gjk_overlap_ratio[None]
    if portal_status == PORTAL_STATUS.EXACT and cached_penetration > 0.0:
        overlap_ratio = collider_info.mpr_to_gjk_overlap_ratio_valid[None]
    gjk_threshold = qd.max(overlap_ratio * geom_pair_scale, tolerance)
    if cached_penetration > 0.0:
        gjk_threshold = qd.math.clamp(
            collider_info.mpr_to_gjk_penetration_ratio[None] * cached_penetration, tolerance, gjk_threshold
        )
    elif portal_status != PORTAL_STATUS.EXACT:
        # A cold pair carries no history to weigh the depth against, so MPR is trusted to the overlap cap only on a
        # portal that recovers the true depth; anything less gates on the tolerance until the cache warms.
        gjk_threshold = tolerance
    if (portal_status == PORTAL_STATUS.UNCONVERGED) or (portal_status == PORTAL_STATUS.EXTRAPOLATED):
        gjk_threshold = tolerance
    return penetration > gjk_threshold


@qd.func
def func_convex_convex_contact(
    i_ga,
    i_gb,
    i_b,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    diff_contact_input: array_class.DiffContactInput,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    errno: qd.Tensor,
):
    if not (dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX):
        EPS = rigid_info.EPS[None]

        # Disabling multi-contact for pairs of decomposed geoms would speed up simulation but may cause physical
        # instabilities in the few cases where multiple contact points are actually needed. Increasing the tolerance
        # criteria to get rid of redundant contact points seems to be a better option.
        multi_contact = (
            rigid_config.enable_multi_contact
            # and not (self._solver.dyn_info.geoms[i_ga].is_decomposed and self._solver.dyn_info.geoms[i_gb].is_decomposed)
            and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.SPHERE
            and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
            and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.SPHERE
            and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
        )
        if qd.static(rigid_config.enable_mujoco_compatibility):
            # The reference engine resolves capsule-capsule pairs analytically with a single contact, outside the
            # generic convex pipeline that carries its perturbation-based multi-contact.
            multi_contact = multi_contact and not (
                dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE
            )

        geom_pair_scale = func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
        tolerance = func_compute_mc_tolerance(i_ga, i_gb, geoms_init_AABB, dyn_info, collider_info, rigid_config)
        diff_pos_tolerance = collider_info.diff_pos_tolerance[None] * geom_pair_scale
        diff_normal_tolerance = collider_info.diff_normal_tolerance[None]

        # Load original geometry state into thread-local variables
        # These are the UNPERTURBED states used as reference point for each independent perturbation
        ga_pos_original = dyn_state.geoms.pos[i_ga, i_b]
        ga_quat_original = dyn_state.geoms.quat[i_ga, i_b]
        gb_pos_original = dyn_state.geoms.pos[i_gb, i_b]
        gb_quat_original = dyn_state.geoms.quat[i_gb, i_b]

        # Current (possibly perturbed) state - initialized to original, updated during perturbations
        ga_pos_current = ga_pos_original
        ga_quat_current = ga_quat_original
        gb_pos_current = gb_pos_original
        gb_quat_current = gb_quat_original

        # Pre-allocate some buffers Note that the variables post-fixed with _0 are the values of these variables for
        # contact 0 (used for multi-contact).
        is_col_0 = False
        penetration_0 = gs.qd_float(0.0)
        normal_0 = qd.Vector.zero(gs.qd_float, 3)
        contact_pos_0 = qd.Vector.zero(gs.qd_float, 3)

        # Whether narrowphase detected a contact.
        is_col = False
        penetration = gs.qd_float(0.0)
        normal = qd.Vector.zero(gs.qd_float, 3)
        contact_pos = qd.Vector.zero(gs.qd_float, 3)

        n_con = gs.qd_int(0)
        axis_0 = qd.Vector.zero(gs.qd_float, 3)
        axis_1 = qd.Vector.zero(gs.qd_float, 3)
        qrot = qd.Vector.zero(gs.qd_float, 4)

        i_pair = collider_info.collision_pair_idx[(i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)]
        for i_detection in range(5):
            prefer_gjk = (
                collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.GJK
                or collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK
            )

            # Apply perturbations to thread-local state
            if multi_contact and is_col_0:
                if qd.static(rigid_config.enable_mujoco_compatibility):
                    # Match MuJoCo's perturbation pattern: single axis at a time i_detection 1: (axis_0, -angle), 2:
                    # (axis_0, +angle), 3: (axis_1, -angle), 4: (axis_1, +angle)
                    axis_idx = (i_detection - 1) // 2
                    angle_sign = 2 * ((i_detection - 1) % 2) - 1
                    axis = axis_0 if axis_idx == 0 else axis_1
                    qrot = gu.qd_rotvec_to_quat(angle_sign * collider_info.mc_perturbation[None] * axis, EPS)
                else:
                    # Perturbation axis must not be aligned with the principal axes of inertia of the geometry,
                    # otherwise it would be more sensitive to ill-conditioning.
                    axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                    qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

                # Apply perturbation starting from original state
                ga_pos_current, ga_quat_current = func_rotate_frame(
                    ga_pos_original, ga_quat_original, contact_pos_0, qrot
                )
                gb_pos_current, gb_quat_current = func_rotate_frame(
                    gb_pos_original, gb_quat_original, contact_pos_0, gu.qd_inv_quat(qrot)
                )

            if (multi_contact and is_col_0) or (i_detection == 0):
                if (
                    dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE
                    and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE
                ):
                    is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(
                        i_ga,
                        i_gb,
                        ga_pos_current,
                        ga_quat_current,
                        gb_pos_current,
                        gb_quat_current,
                        dyn_info,
                        rigid_info,
                    )
                elif (
                    dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE
                    and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.SPHERE
                ):
                    is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_sphere_contact(
                        i_ga, i_gb, ga_pos_current, gb_pos_current, dyn_info, rigid_info
                    )
                elif (
                    dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE
                    and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE
                ):
                    is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(
                        i_ga,
                        i_gb,
                        ga_pos_current,
                        ga_quat_current,
                        gb_pos_current,
                        gb_quat_current,
                        dyn_info,
                        rigid_info,
                    )
                elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX:
                    is_col, normal, contact_pos, penetration = func_sphere_box_contact(
                        i_ga,
                        i_gb,
                        ga_pos_current,
                        ga_quat_current,
                        gb_pos_current,
                        gb_quat_current,
                        dyn_info,
                        rigid_info,
                    )
                elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE:
                    plane_dir = qd.Vector(
                        [dyn_info.geoms.data[i_ga][0], dyn_info.geoms.data[i_ga][1], dyn_info.geoms.data[i_ga][2]],
                        dt=gs.qd_float,
                    )
                    plane_dir = gu.qd_transform_by_quat(plane_dir, ga_quat_current)
                    normal = -plane_dir.normalized()

                    v1 = mpr.support_driver(
                        i_gb,
                        i_b,
                        normal,
                        gb_pos_current,
                        gb_quat_current,
                        collider_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                    penetration = normal.dot(v1 - ga_pos_current)
                    contact_pos = v1 - 0.5 * penetration * normal
                    if qd.static(rigid_config.enable_mujoco_compatibility):
                        is_col = penetration > 0.0
                    else:
                        # Grazing band and clamp: see the plane branch of _func_multicontact_run_detection.
                        is_col = penetration > -EPS * geom_pair_scale
                        penetration = qd.max(penetration, 0.0)
                else:
                    ### MPR, MJ_MPR
                    if qd.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)
                    ):
                        # Try using MPR before anything else
                        is_mpr_updated = False
                        normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                        is_mpr_guess_direction_available = (qd.abs(normal_ws) > EPS).any()
                        for i_mpr in range(2):
                            if i_mpr == 1:
                                # Try without warm-start if no contact was detected using it. When penetration depth is
                                # very shallow, MPR may wrongly classify two geometries as not in contact while they
                                # actually are. This helps to improve contact persistence without increasing much the
                                # overall computational cost since the fallback should not be triggered very often.
                                if qd.static(not rigid_config.enable_mujoco_compatibility):
                                    if (i_detection == 0) and not is_col and is_mpr_guess_direction_available:
                                        normal_ws = qd.Vector.zero(gs.qd_float, 3)
                                        is_mpr_guess_direction_available = False
                                        is_mpr_updated = False

                            if not is_mpr_updated:
                                is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    normal_ws,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                    geoms_init_AABB,
                                    collider_state,
                                    mpr_state,
                                    dyn_info,
                                    rigid_info,
                                    collider_info,
                                    rigid_config,
                                    collider_static_config,
                                )
                                is_mpr_updated = True

                        if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                            prefer_gjk = func_prefer_gjk_refinement(
                                i_pair,
                                i_b,
                                mpr_state.portal_status[i_b],
                                penetration,
                                geom_pair_scale,
                                tolerance,
                                collider_state,
                                collider_info,
                            )

                    ### GJK, MJ_GJK
                    # TODO: Add support of smooth refinement to differentiable contact.
                    if qd.static(collider_static_config.ccd_algorithm != CCD_ALGORITHM_CODE.MJ_MPR):
                        if prefer_gjk:
                            if qd.static(rigid_config.requires_grad):
                                diff_gjk.func_gjk_contact(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                    diff_pos_tolerance,
                                    diff_normal_tolerance,
                                    geoms_init_AABB,
                                    dyn_state,
                                    collider_state,
                                    gjk_state,
                                    diff_contact_input,
                                    dyn_info,
                                    rigid_info,
                                    collider_info,
                                    rigid_config,
                                    collider_static_config,
                                )
                            else:
                                gjk.func_gjk_contact(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                    dyn_state,
                                    collider_state,
                                    gjk_state,
                                    dyn_info,
                                    rigid_info,
                                    collider_info,
                                    rigid_config,
                                    collider_static_config,
                                    gjk_static_config,
                                )

                            is_col = gjk_state.is_col[i_b] == 1
                            penetration = gjk_state.penetration[i_b]
                            n_contacts = gjk_state.n_contacts[i_b]

                            if is_col:
                                if qd.static(rigid_config.requires_grad):
                                    for i_c in range(n_contacts):
                                        func_add_diff_contact_input(
                                            i_ga, i_gb, i_b, i_c, collider_state, gjk_state, collider_info
                                        )
                                        func_add_contact(
                                            i_ga,
                                            i_gb,
                                            i_b,
                                            i_pair,
                                            gjk_state.normal[i_b, i_c],
                                            gjk_state.contact_pos[i_b, i_c],
                                            gjk_state.diff_penetration[i_b, i_c],
                                            dyn_state,
                                            collider_state,
                                            dyn_info,
                                            rigid_info,
                                            collider_info,
                                            use_atomic=False,
                                            errno=errno,
                                        )
                                    break
                                else:
                                    if gjk_state.multi_contact_flag[i_b]:
                                        # Since we already found multiple contact points, add the discovered contact
                                        # points and stop multi-contact search.
                                        for i_c in range(n_contacts):
                                            # Ignore contact points if the number of contacts exceeds the limit.
                                            if i_c < collider_static_config.n_contacts_per_convex_pair:
                                                contact_pos = gjk_state.contact_pos[i_b, i_c]
                                                normal = gjk_state.normal[i_b, i_c]
                                                contact_pos = func_apply_smooth_refinement(
                                                    i_ga,
                                                    i_gb,
                                                    normal,
                                                    penetration,
                                                    contact_pos,
                                                    ga_pos_current,
                                                    ga_quat_current,
                                                    gb_pos_current,
                                                    gb_quat_current,
                                                    dyn_info,
                                                    rigid_config,
                                                )
                                                func_add_contact(
                                                    i_ga,
                                                    i_gb,
                                                    i_b,
                                                    i_pair,
                                                    normal,
                                                    contact_pos,
                                                    penetration,
                                                    dyn_state,
                                                    collider_state,
                                                    dyn_info,
                                                    rigid_info,
                                                    collider_info,
                                                    use_atomic=False,
                                                    errno=errno,
                                                )

                                        break
                                    else:
                                        contact_pos = gjk_state.contact_pos[i_b, 0]
                                        normal = gjk_state.normal[i_b, 0]

            # Refine the unperturbed (i_detection == 0) contact here; perturbed contacts are refined inside
            # func_recompute_perturbed_contact after the perturbation is reverted, on the canonical (unperturbed) pose.
            if is_col and i_detection == 0:
                contact_pos = func_apply_smooth_refinement(
                    i_ga,
                    i_gb,
                    normal,
                    penetration,
                    contact_pos,
                    ga_pos_current,
                    ga_quat_current,
                    gb_pos_current,
                    gb_quat_current,
                    dyn_info,
                    rigid_config,
                )

            if i_detection == 0:
                is_col_0, normal_0, penetration_0, contact_pos_0 = is_col, normal, penetration, contact_pos
                if is_col_0:
                    func_add_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        i_pair,
                        normal_0,
                        contact_pos_0,
                        penetration_0,
                        dyn_state,
                        collider_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        use_atomic=False,
                        errno=errno,
                    )
                    if multi_contact:
                        # Perturb geom_a around two orthogonal axes to find multiple contacts
                        axis_0, axis_1 = func_contact_orthogonals(
                            i_ga, i_gb, i_b, normal, geoms_init_AABB, dyn_state, dyn_info, rigid_info, rigid_config
                        )
                        n_con = 1

                    if qd.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)
                    ):
                        collider_state.contact_cache.normal[i_pair, i_b] = normal
                        collider_state.contact_cache.penetration[i_pair, i_b] = penetration
                else:
                    # Clear the cached normal AND penetration when not in contact, so a later re-contact is treated as
                    # cold (warm-start penetration 0) instead of reading a stale penetration across the gap.
                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    collider_state.contact_cache.penetration[i_pair, i_b] = 0.0
            elif multi_contact and is_col:
                # For perturbed iterations (i_detection > 0), recompute the contact from the deepest contact points
                # discovered by the perturbed detection, evaluated on the unperturbed geometries. This applies to all
                # collision methods when multi-contact is enabled, except mujoco compatible. When the correction is
                # skipped (mujoco compatible), is_exact stays False so the lenient acceptance threshold is used.
                is_exact = False
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    _used_gjk = prefer_gjk
                    normal, penetration, contact_pos, is_exact = func_recompute_perturbed_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        normal,
                        penetration,
                        contact_pos,
                        normal_0,
                        contact_pos_0,
                        qrot,
                        ga_pos_original,
                        ga_quat_original,
                        gb_pos_original,
                        gb_quat_original,
                        _used_gjk,
                        mpr_state,
                        gjk_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                    )

                # For MuJoCo-compatible GJK, set penetration of perturbed contacts to equal the initial contact's
                # penetration, matching MuJoCo's behavior (engine_collision_convex.c:1010).
                if qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    penetration = penetration_0

                # Discard repeated contact points
                repeated = False
                for i_c in range(n_con):
                    if not repeated:
                        idx_prev = collider_state.n_contacts[i_b] - 1 - i_c
                        prev_contact = collider_state.contact_data.pos[idx_prev, i_b]
                        if (contact_pos - prev_contact).norm() < tolerance:
                            repeated = True

                if not repeated:
                    # When the correction is exact, a fictitious candidate (one that only touches because of the
                    # perturbation) reverts to a non-positive penetration and is discarded, within a guard of rounding
                    # width so a grazing candidate is accepted consistently across rotated copies. When the correction
                    # is only approximate, keep the negative tolerance so first-order error drops no genuine contact.
                    if penetration > (-EPS * geom_pair_scale if is_exact else -tolerance):
                        penetration = qd.max(penetration, 0.0)
                        func_add_contact(
                            i_ga,
                            i_gb,
                            i_b,
                            i_pair,
                            normal,
                            contact_pos,
                            penetration,
                            dyn_state,
                            collider_state,
                            dyn_info,
                            rigid_info,
                            collider_info,
                            use_atomic=False,
                            errno=errno,
                        )
                        n_con = n_con + 1


@qd.func
def _func_multicontact_run_detection(
    i_ga,
    i_gb,
    i_scratch,
    i_b,
    i_pair,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    use_gjk: qd.template(),
    is_initial_detection: qd.template(),
):
    """Run one detection (capsule/plane/MPR/GJK) and return (is_col, normal, contact_pos, penetration, used_gjk)."""
    EPS = rigid_info.EPS[None]
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    used_gjk = False
    geom_pair_scale = func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
    tolerance = func_compute_mc_tolerance(i_ga, i_gb, geoms_init_AABB, dyn_info, collider_info, rigid_config)

    if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
        is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(
            i_ga, i_gb, ga_pos, ga_quat, gb_pos, gb_quat, dyn_info, rigid_info
        )
    elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.SPHERE:
        is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_sphere_contact(
            i_ga, i_gb, ga_pos, gb_pos, dyn_info, rigid_info
        )
    elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
        is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(
            i_ga, i_gb, ga_pos, ga_quat, gb_pos, gb_quat, dyn_info, rigid_info
        )
    elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX:
        is_col, normal, contact_pos, penetration = func_sphere_box_contact(
            i_ga, i_gb, ga_pos, ga_quat, gb_pos, gb_quat, dyn_info, rigid_info
        )
    elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE:
        plane_dir = qd.Vector(
            [dyn_info.geoms.data[i_ga][0], dyn_info.geoms.data[i_ga][1], dyn_info.geoms.data[i_ga][2]], dt=gs.qd_float
        )
        plane_dir = gu.qd_transform_by_quat(plane_dir, ga_quat)
        normal = -plane_dir.normalized()
        v1 = mpr.support_driver(
            i_gb,
            i_b,
            normal,
            gb_pos,
            gb_quat,
            collider_state,
            dyn_info,
            collider_info,
            rigid_config,
            collider_static_config,
        )
        penetration = normal.dot(v1 - ga_pos)
        contact_pos = v1 - 0.5 * penetration * normal
        if qd.static(rigid_config.enable_mujoco_compatibility):
            is_col = penetration > 0.0
        else:
            # A grazing pair's depth sits within rounding of zero, so a bare sign test accepts or rejects on rounding
            # alone: accept within a band of rounding width and clamp, so the contact carries zero force until pressed.
            is_col = penetration > -EPS * geom_pair_scale
            penetration = qd.max(penetration, 0.0)
    else:
        if qd.static(
            collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR) and not use_gjk
        ):
            is_mpr_updated = False
            normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
            is_mpr_guess_direction_available = (qd.abs(normal_ws) > EPS).any()
            for i_mpr in range(2):
                if i_mpr == 1:
                    if qd.static(is_initial_detection and not rigid_config.enable_mujoco_compatibility):
                        if not is_col and is_mpr_guess_direction_available:
                            normal_ws = qd.Vector.zero(gs.qd_float, 3)
                            is_mpr_guess_direction_available = False
                            is_mpr_updated = False

                if not is_mpr_updated:
                    is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                        i_ga,
                        i_gb,
                        i_scratch,
                        normal_ws,
                        ga_pos,
                        ga_quat,
                        gb_pos,
                        gb_quat,
                        geoms_init_AABB,
                        collider_state,
                        mpr_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                    is_mpr_updated = True

        if qd.static(
            collider_static_config.ccd_algorithm != CCD_ALGORITHM_CODE.MJ_MPR
            and use_gjk
            and not rigid_config.requires_grad
        ):
            gjk.func_gjk_contact(
                i_ga,
                i_gb,
                i_scratch,
                ga_pos,
                ga_quat,
                gb_pos,
                gb_quat,
                dyn_state,
                collider_state,
                gjk_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
                gjk_static_config,
            )
            is_col = gjk_state.is_col[i_scratch] == 1
            penetration = gjk_state.penetration[i_scratch]
            if is_col:
                contact_pos = gjk_state.contact_pos[i_scratch, 0]
                normal = gjk_state.normal[i_scratch, 0]
            used_gjk = True

    return is_col, normal, contact_pos, penetration, used_gjk


@qd.func
def _func_multicontact_mpr(
    i_scratch,
    i_b,
    i_ga,
    i_gb,
    i_pair,
    contact_pos_0: qd.types.vector(3),
    normal_0: qd.types.vector(3),
    penetration_0,
    prefer_gjk_0: bool,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    errno: qd.Tensor,
):
    """Compute all contacts for a pair and write them contiguously via a single atomic reservation.

    Contact 0 uses GJK when prefer_gjk_0 is set (its penetration gate fired in the contact0 kernel), expanding a GJK
    native multi-contact manifold and skipping perturbation when one is returned; otherwise it uses the MPR seed
    passed from the contact0 kernel. Each perturbed contact tries MPR first and falls back to GJK on its own when its
    penetration gate fires, except under the pure GJK algorithm where every contact is detected with GJK. The single
    atomic reservation gives deterministic per-pair contact ordering."""
    EPS = rigid_info.EPS[None]

    ga_pos_original = dyn_state.geoms.pos[i_ga, i_b]
    ga_quat_original = dyn_state.geoms.quat[i_ga, i_b]
    gb_pos_original = dyn_state.geoms.pos[i_gb, i_b]
    gb_quat_original = dyn_state.geoms.quat[i_gb, i_b]

    multi_contact = (
        rigid_config.enable_multi_contact
        and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.SPHERE
        and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
        and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.SPHERE
        and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
    )
    if qd.static(rigid_config.enable_mujoco_compatibility):
        # Capsule-capsule exclusion under compatibility: see func_convex_convex_contact.
        multi_contact = multi_contact and not (
            dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE
        )

    geom_pair_scale = func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
    tolerance = func_compute_mc_tolerance(i_ga, i_gb, geoms_init_AABB, dyn_info, collider_info, rigid_config)

    n_con = gs.qd_int(0)
    local_contact_pos = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 3)
    local_normal = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 3)
    local_penetration = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 1)
    gjk_multi_done = False

    contact0_normal = normal_0
    contact0_pos = contact_pos_0

    if prefer_gjk_0:
        # Contact 0 fell back to GJK. Re-detect it with GJK on the unperturbed pose, expanding a native multi-contact
        # manifold (and skipping perturbation) when one is found.
        is_col, normal, contact_pos, penetration, _used_gjk = _func_multicontact_run_detection(
            i_ga,
            i_gb,
            i_scratch,
            i_b,
            i_pair,
            ga_pos_original,
            ga_quat_original,
            gb_pos_original,
            gb_quat_original,
            geoms_init_AABB,
            dyn_state,
            collider_state,
            mpr_state,
            gjk_state,
            dyn_info,
            rigid_info,
            collider_info,
            rigid_config,
            collider_static_config,
            gjk_static_config,
            use_gjk=True,
            is_initial_detection=True,
        )
        if is_col:
            collider_state.contact_cache.normal[i_pair, i_b] = normal
            collider_state.contact_cache.penetration[i_pair, i_b] = penetration
            if _used_gjk:
                # GJK populated gjk_state: take the single contact (i_c 0) or, when a native manifold was returned, all
                # of its points (which sets gjk_multi_done to skip perturbation). The single contact is gjk_state[0],
                # so it is just a one-point manifold.
                gjk_multi_done = gjk_state.multi_contact_flag[i_scratch] == 1
                n_contacts_gjk = gjk_state.n_contacts[i_scratch] if gjk_multi_done else 1
                for i_c in range(n_contacts_gjk):
                    if n_con < collider_static_config.n_contacts_per_convex_pair:
                        gjk_normal = gjk_state.normal[i_scratch, i_c]
                        gjk_contact_pos = func_apply_smooth_refinement(
                            i_ga,
                            i_gb,
                            gjk_normal,
                            penetration,
                            gjk_state.contact_pos[i_scratch, i_c],
                            ga_pos_original,
                            ga_quat_original,
                            gb_pos_original,
                            gb_quat_original,
                            dyn_info,
                            rigid_config,
                        )
                        for i_3 in qd.static(range(3)):
                            local_contact_pos[n_con, i_3] = gjk_contact_pos[i_3]
                            local_normal[n_con, i_3] = gjk_normal[i_3]
                        local_penetration[n_con, 0] = penetration
                        if i_c == 0:
                            contact0_normal = gjk_normal
                            contact0_pos = gjk_contact_pos
                        n_con = n_con + 1
            else:
                # Analytic detection (plane/capsule/sphere-box) leaves gjk_state untouched, so use the returned contact
                # directly. Reading gjk_state here would pick up a stale contact from another pair on this thread,
                # which is non-deterministic because the thread-to-pair assignment is racy.
                contact_pos = func_apply_smooth_refinement(
                    i_ga,
                    i_gb,
                    normal,
                    penetration,
                    contact_pos,
                    ga_pos_original,
                    ga_quat_original,
                    gb_pos_original,
                    gb_quat_original,
                    dyn_info,
                    rigid_config,
                )
                contact0_normal = normal
                contact0_pos = contact_pos
                for i_3 in qd.static(range(3)):
                    local_contact_pos[0, i_3] = contact_pos[i_3]
                    local_normal[0, i_3] = normal[i_3]
                local_penetration[0, 0] = penetration
                n_con = 1
        else:
            collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
            collider_state.contact_cache.penetration[i_pair, i_b] = 0.0
    else:
        # Contact 0 from the MPR seed already detected, refined and cached by the contact0 kernel.
        for i_3 in qd.static(range(3)):
            local_contact_pos[0, i_3] = contact_pos_0[i_3]
            local_normal[0, i_3] = normal_0[i_3]
        local_penetration[0, 0] = penetration_0
        n_con = 1

    if multi_contact and n_con > 0 and not gjk_multi_done:
        axis_0, axis_1 = func_contact_orthogonals(
            i_ga, i_gb, i_b, contact0_normal, geoms_init_AABB, dyn_state, dyn_info, rigid_info, rigid_config
        )

        for i_detection in range(4):
            i_det = i_detection + 1
            # Declared ahead of the static branches that assign it, as quadrants scoping requires.
            qrot = qd.Vector.zero(gs.qd_float, 4)
            if qd.static(rigid_config.enable_mujoco_compatibility):
                # Single-axis perturbation pattern: see the twin loop in func_convex_convex_contact.
                axis_idx = (i_det - 1) // 2
                angle_sign = 2 * ((i_det - 1) % 2) - 1
                axis = axis_0 if axis_idx == 0 else axis_1
                qrot = gu.qd_rotvec_to_quat(angle_sign * collider_info.mc_perturbation[None] * axis, EPS)
            else:
                # Combined-axes perturbation: see the twin loop in func_convex_convex_contact.
                axis = (2 * (i_det % 2) - 1) * axis_0 + (1 - 2 * ((i_det // 2) % 2)) * axis_1
                qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

            ga_pos_current, ga_quat_current = func_rotate_frame(ga_pos_original, ga_quat_original, contact0_pos, qrot)
            gb_pos_current, gb_quat_current = func_rotate_frame(
                gb_pos_original, gb_quat_original, contact0_pos, gu.qd_inv_quat(qrot)
            )

            is_col, normal, contact_pos, penetration, _used_gjk = _func_multicontact_run_detection(
                i_ga,
                i_gb,
                i_scratch,
                i_b,
                i_pair,
                ga_pos_current,
                ga_quat_current,
                gb_pos_current,
                gb_quat_current,
                geoms_init_AABB,
                dyn_state,
                collider_state,
                mpr_state,
                gjk_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
                gjk_static_config,
                # Perturbed contacts try MPR first under the MPR algorithm (falling back to GJK per contact below).
                # The pure GJK algorithms detect every perturbed contact with GJK directly.
                use_gjk=qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.GJK, CCD_ALGORITHM_CODE.MJ_GJK)
                ),
                is_initial_detection=False,
            )

            if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                if is_col:
                    # When the refinement is called for it applies to this perturbed contact only, rather than
                    # upgrading the whole pair, which keeps the MPR-first model.
                    prefer_gjk = func_prefer_gjk_refinement(
                        i_pair,
                        i_b,
                        mpr_state.portal_status[i_scratch],
                        penetration,
                        geom_pair_scale,
                        tolerance,
                        collider_state,
                        collider_info,
                    )
                    if prefer_gjk:
                        is_col, normal, contact_pos, penetration, _used_gjk = _func_multicontact_run_detection(
                            i_ga,
                            i_gb,
                            i_scratch,
                            i_b,
                            i_pair,
                            ga_pos_current,
                            ga_quat_current,
                            gb_pos_current,
                            gb_quat_current,
                            geoms_init_AABB,
                            dyn_state,
                            collider_state,
                            mpr_state,
                            gjk_state,
                            dyn_info,
                            rigid_info,
                            collider_info,
                            rigid_config,
                            collider_static_config,
                            gjk_static_config,
                            use_gjk=True,
                            is_initial_detection=False,
                        )

            if is_col:
                # The perturbed contact is refined inside func_recompute_perturbed_contact (after the perturbation is
                # reverted, on the canonical pose); no pre-reversal refinement is needed here.
                is_exact = False
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    normal, penetration, contact_pos, is_exact = func_recompute_perturbed_contact(
                        i_ga,
                        i_gb,
                        i_scratch,
                        normal,
                        penetration,
                        contact_pos,
                        contact0_normal,
                        contact0_pos,
                        qrot,
                        ga_pos_original,
                        ga_quat_original,
                        gb_pos_original,
                        gb_quat_original,
                        _used_gjk,
                        mpr_state,
                        gjk_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                    )

                # Perturbed contacts carry the initial contact's penetration under compatibility: see the twin
                # acceptance in func_convex_convex_contact.
                if qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    penetration = local_penetration[0, 0]

                repeated = False
                for i_c in range(n_con):
                    if not repeated:
                        prev = qd.Vector(
                            [local_contact_pos[i_c, 0], local_contact_pos[i_c, 1], local_contact_pos[i_c, 2]],
                            dt=gs.qd_float,
                        )
                        if (contact_pos - prev).norm() < tolerance:
                            repeated = True

                if not repeated:
                    # Rounding-scale guard for exact candidates: see the twin acceptance in func_convex_convex_contact.
                    if penetration > (-EPS * geom_pair_scale if is_exact else -tolerance):
                        penetration = qd.max(penetration, 0.0)
                        for i_3 in qd.static(range(3)):
                            local_contact_pos[n_con, i_3] = contact_pos[i_3]
                            local_normal[n_con, i_3] = normal[i_3]
                        local_penetration[n_con, 0] = penetration
                        n_con = n_con + 1

    if n_con > 0:
        # Non-atomic pre-check to avoid reserving slots we cannot fill. A rare race between the read and the
        # atomic_add below may still overshoot; in that case we write only the contacts that fit and set errno.
        max_candidate_contacts = collider_info.max_candidate_contacts[None]
        if collider_state.n_contacts[i_b] + n_con > max_candidate_contacts:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
        else:
            start_idx = qd.atomic_add(collider_state.n_contacts[i_b], n_con)
            n_con = qd.math.clamp(max_candidate_contacts - start_idx, 0, n_con)
            if n_con == 0:
                errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
            for i in range(n_con):
                i_c = start_idx + i
                pos_i = qd.Vector(
                    [local_contact_pos[i, 0], local_contact_pos[i, 1], local_contact_pos[i, 2]], dt=gs.qd_float
                )
                normal_i = qd.Vector([local_normal[i, 0], local_normal[i, 1], local_normal[i, 2]], dt=gs.qd_float)
                func_set_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    i_c,
                    i_pair,
                    normal_i,
                    pos_i,
                    local_penetration[i, 0],
                    dyn_state,
                    collider_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    errno,
                )


@qd.kernel(fastcache=True)
def _func_narrowphase_multicontact(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    n_total_threads: qd.template(),
    max_items_per_thread: qd.template(),
    errno: qd.Tensor,
):
    for i_tid in range(n_total_threads):
        for _iter in range(max_items_per_thread):
            idx = qd.atomic_add(collider_state.narrowphase_work_queues.mpr_work_counter[0], 1)
            if idx >= collider_state.narrowphase_work_queues.mpr_queue_size[0]:
                break
            i_b = collider_state.narrowphase_work_queues.mpr_i_b[idx]
            i_ga = collider_state.narrowphase_work_queues.mpr_i_ga[idx]
            i_gb = collider_state.narrowphase_work_queues.mpr_i_gb[idx]
            i_pair = collider_state.narrowphase_work_queues.mpr_i_pair[idx]
            contact_pos_0 = collider_state.narrowphase_work_queues.mpr_contact_pos_0[idx]
            normal_0 = collider_state.narrowphase_work_queues.mpr_normal_0[idx]
            penetration_0 = collider_state.narrowphase_work_queues.mpr_penetration_0[idx]
            prefer_gjk_0 = collider_state.narrowphase_work_queues.mpr_prefer_gjk[idx] == 1

            _func_multicontact_mpr(
                i_tid,
                i_b,
                i_ga,
                i_gb,
                i_pair,
                contact_pos_0,
                normal_0,
                penetration_0,
                prefer_gjk_0,
                geoms_init_AABB,
                dyn_state,
                collider_state,
                mpr_state,
                gjk_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
                gjk_static_config,
                errno,
            )


@qd.kernel(fastcache=True)
def _func_reset_narrowphase_work_queues(collider_state: array_class.ColliderState):
    for _i in range(1):
        collider_state.narrowphase_work_queues.mpr_queue_size[0] = 0
        collider_state.narrowphase_work_queues.mpr_work_counter[0] = 0


@qd.func
def _func_enqueue_for_multicontact(
    i_b,
    i_ga,
    i_gb,
    i_pair,
    contact_pos_0: qd.types.vector(3),
    normal_0: qd.types.vector(3),
    penetration_0,
    prefer_gjk: bool,
    collider_state: array_class.ColliderState,
):
    idx = qd.atomic_add(collider_state.narrowphase_work_queues.mpr_queue_size[0], 1)
    collider_state.narrowphase_work_queues.mpr_i_b[idx] = i_b
    collider_state.narrowphase_work_queues.mpr_i_ga[idx] = i_ga
    collider_state.narrowphase_work_queues.mpr_i_gb[idx] = i_gb
    collider_state.narrowphase_work_queues.mpr_i_pair[idx] = i_pair
    collider_state.narrowphase_work_queues.mpr_contact_pos_0[idx] = contact_pos_0
    collider_state.narrowphase_work_queues.mpr_normal_0[idx] = normal_0
    collider_state.narrowphase_work_queues.mpr_penetration_0[idx] = penetration_0
    collider_state.narrowphase_work_queues.mpr_prefer_gjk[idx] = 1 if prefer_gjk else 0


@qd.kernel(fastcache=True)
def _func_narrowphase_contact0(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    n_envs: qd.template(),
    n_chunks: qd.template(),
    errno: qd.Tensor,
):
    _grid_size = n_envs * n_chunks
    max_broad_pairs = collider_state.broad_collision_pairs.shape[0]

    for flat_idx in range(_grid_size):
        i_b = flat_idx // n_chunks
        chunk = flat_idx % n_chunks
        n_pairs = collider_state.n_broad_pairs[i_b]
        pair_start = chunk * n_pairs // n_chunks
        pair_end = (chunk + 1) * n_pairs // n_chunks

        for i_pair_local in range(max_broad_pairs):
            i_pair_idx = pair_start + i_pair_local
            if i_pair_idx >= pair_end:
                break

            i_ga = collider_state.broad_collision_pairs[i_pair_idx, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair_idx, i_b][1]

            if dyn_info.geoms.type[i_ga] > dyn_info.geoms.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if not (
                dyn_info.geoms.is_convex[i_ga]
                and dyn_info.geoms.is_convex[i_gb]
                and not dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.TERRAIN
                and not (
                    qd.static(rigid_config.box_box_detection)
                    and dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.BOX
                    and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX
                )
            ):
                continue

            if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX:
                continue

            EPS = rigid_info.EPS[None]

            multi_contact = (
                rigid_config.enable_multi_contact
                and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.SPHERE
                and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
                and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.SPHERE
                and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
            )

            geom_pair_scale = func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
            tolerance = func_compute_mc_tolerance(i_ga, i_gb, geoms_init_AABB, dyn_info, collider_info, rigid_config)

            ga_pos = dyn_state.geoms.pos[i_ga, i_b]
            ga_quat = dyn_state.geoms.quat[i_ga, i_b]
            gb_pos = dyn_state.geoms.pos[i_gb, i_b]
            gb_quat = dyn_state.geoms.quat[i_gb, i_b]

            is_col = False
            penetration = gs.qd_float(0.0)
            normal = qd.Vector.zero(gs.qd_float, 3)
            contact_pos = qd.Vector.zero(gs.qd_float, 3)
            prefer_gjk = (
                collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.GJK
                or collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK
            )

            i_pair = collider_info.collision_pair_idx[(i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)]

            if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
                is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(
                    i_ga, i_gb, ga_pos, ga_quat, gb_pos, gb_quat, dyn_info, rigid_info
                )
            elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.SPHERE:
                is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_sphere_contact(
                    i_ga, i_gb, ga_pos, gb_pos, dyn_info, rigid_info
                )
            elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
                is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(
                    i_ga, i_gb, ga_pos, ga_quat, gb_pos, gb_quat, dyn_info, rigid_info
                )
            elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX:
                is_col, normal, contact_pos, penetration = func_sphere_box_contact(
                    i_ga, i_gb, ga_pos, ga_quat, gb_pos, gb_quat, dyn_info, rigid_info
                )
            elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE:
                plane_dir = qd.Vector(
                    [dyn_info.geoms.data[i_ga][0], dyn_info.geoms.data[i_ga][1], dyn_info.geoms.data[i_ga][2]],
                    dt=gs.qd_float,
                )
                plane_dir = gu.qd_transform_by_quat(plane_dir, ga_quat)
                normal = -plane_dir.normalized()
                v1 = mpr.support_driver(
                    i_gb,
                    i_b,
                    normal,
                    gb_pos,
                    gb_quat,
                    collider_state,
                    dyn_info,
                    collider_info,
                    rigid_config,
                    collider_static_config,
                )
                penetration = normal.dot(v1 - ga_pos)
                contact_pos = v1 - 0.5 * penetration * normal
                if qd.static(rigid_config.enable_mujoco_compatibility):
                    is_col = penetration > 0.0
                else:
                    # Grazing band and clamp: see the plane branch of _func_multicontact_run_detection.
                    is_col = penetration > -EPS * geom_pair_scale
                    penetration = qd.max(penetration, 0.0)
            else:
                if qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.GJK, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    gjk.clear_cache(flat_idx, gjk_state)
                    distance = gjk.func_gjk(
                        i_ga,
                        i_gb,
                        flat_idx,
                        ga_pos,
                        ga_quat,
                        gb_pos,
                        gb_quat,
                        shrink_sphere=False,
                        collider_state=collider_state,
                        gjk_state=gjk_state,
                        dyn_info=dyn_info,
                        collider_info=collider_info,
                        rigid_config=rigid_config,
                        collider_static_config=collider_static_config,
                    )
                    is_col = distance < collider_info.gjk.collision_eps[None]
                    if distance >= 0.5 * collider_info.gjk.FLOAT_MAX[None]:
                        # func_gjk (fp32 on GPU) can spuriously separate a pair that was genuinely in contact last
                        # step. Trust temporal coherence: if the multicontact pass cached a contact normal for this
                        # pair, re-enqueue so the robust pass re-decides, instead of dropping it on a marginal frame.
                        normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                        is_col = normal_ws.dot(normal_ws) > 0.0

                if qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)
                ):
                    is_mpr_updated = False
                    normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                    is_mpr_guess_direction_available = (qd.abs(normal_ws) > EPS).any()
                    for i_mpr in range(2):
                        if i_mpr == 1:
                            if qd.static(not rigid_config.enable_mujoco_compatibility):
                                if not is_col and is_mpr_guess_direction_available:
                                    normal_ws = qd.Vector.zero(gs.qd_float, 3)
                                    is_mpr_guess_direction_available = False
                                    is_mpr_updated = False

                        if not is_mpr_updated:
                            is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                                i_ga,
                                i_gb,
                                flat_idx,
                                normal_ws,
                                ga_pos,
                                ga_quat,
                                gb_pos,
                                gb_quat,
                                geoms_init_AABB,
                                collider_state,
                                mpr_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                rigid_config,
                                collider_static_config,
                            )
                            is_mpr_updated = True

                    if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                        prefer_gjk = func_prefer_gjk_refinement(
                            i_pair,
                            i_b,
                            mpr_state.portal_status[flat_idx],
                            penetration,
                            geom_pair_scale,
                            tolerance,
                            collider_state,
                            collider_info,
                        )

            if is_col:
                if qd.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)):
                    collider_state.contact_cache.normal[i_pair, i_b] = normal
                    collider_state.contact_cache.penetration[i_pair, i_b] = penetration
                # Refine the contact position before enqueueing or storing it. The downstream multicontact functions
                # store this as the initial contact (index 0 of local_contact_pos) without re-refining, so refinement
                # must happen here to stay consistent with the monolithic path's consolidated refinement at the start
                # of each i_detection iteration.
                contact_pos = func_apply_smooth_refinement(
                    i_ga,
                    i_gb,
                    normal,
                    penetration,
                    contact_pos,
                    dyn_state.geoms.pos[i_ga, i_b],
                    dyn_state.geoms.quat[i_ga, i_b],
                    dyn_state.geoms.pos[i_gb, i_b],
                    dyn_state.geoms.quat[i_gb, i_b],
                    dyn_info,
                    rigid_config,
                )
                if multi_contact or prefer_gjk:
                    # Enqueue for the multicontact pass, which writes all contacts (including contact 0) contiguously
                    # via a single atomic reservation. The prefer_gjk flag selects GJK for contact 0; perturbed
                    # contacts always try MPR first and fall back to GJK per contact. prefer_gjk is never set for the
                    # MJ_MPR algorithm (no GJK), so a non-multi_contact MJ_MPR pair always takes the fast path below.
                    _func_enqueue_for_multicontact(
                        i_b, i_ga, i_gb, i_pair, contact_pos, normal, penetration, prefer_gjk, collider_state
                    )
                else:
                    func_add_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        i_pair,
                        normal,
                        contact_pos,
                        penetration,
                        dyn_state,
                        collider_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        use_atomic=True,
                        errno=errno,
                    )
            elif not is_col:
                collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                collider_state.contact_cache.penetration[i_pair, i_b] = 0.0


@qd.kernel(fastcache=True)
def func_narrow_phase_convex_vs_convex(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    diff_contact_input: array_class.DiffContactInput,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    errno: qd.Tensor,
):
    _B = collider_state.active_buffer.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if dyn_info.geoms.type[i_ga] > dyn_info.geoms.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if (
                dyn_info.geoms.is_convex[i_ga]
                and dyn_info.geoms.is_convex[i_gb]
                and not dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.TERRAIN
                and not (
                    rigid_config.box_box_detection
                    and dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.BOX
                    and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX
                )
            ):
                if not (
                    dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX
                ):
                    func_convex_convex_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        geoms_init_AABB,
                        dyn_state,
                        collider_state,
                        mpr_state,
                        gjk_state,
                        diff_contact_input,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                        gjk_static_config,
                        errno,
                    )


@qd.kernel(fastcache=True)
def func_narrow_phase_diff_convex_vs_convex(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    diff_contact_input: array_class.DiffContactInput,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    # Compute reference contacts
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(collider_state.contact_data.pos.shape[0], collider_state.active_buffer.shape[1]):
        if i_c < collider_state.n_contacts[i_b]:
            ref_id = collider_state.diff_contact_input.ref_id[i_b, i_c]
            is_ref = i_c == ref_id
            i_ga = collider_state.diff_contact_input.geom_a[i_b, i_c]
            i_gb = collider_state.diff_contact_input.geom_b[i_b, i_c]

            if is_ref:
                ref_penetration = -1.0
                contact_pos = gs.qd_vec3(0.0, 0.0, 0.0)
                contact_normal = gs.qd_vec3(0.0, 0.0, 0.0)
                penetration = gs.qd_float(0.0)
                weight = gs.qd_float(0.0)
                if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE:
                    contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_plane_contact(
                        i_ga, i_gb, i_b, i_c, dyn_state, diff_contact_input, dyn_info
                    )
                elif (
                    dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE
                    and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.SPHERE
                ):
                    contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_sphere_contact(
                        i_ga, i_gb, i_b, dyn_state, dyn_info, rigid_info
                    )
                else:
                    contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_contact(
                        i_ga, i_gb, i_b, i_c, ref_penetration, dyn_state, diff_contact_input, collider_info
                    )
                collider_state.diff_contact_input.ref_penetration[i_b, i_c] = penetration

                func_set_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    i_c,
                    collider_state.contact_data.pair_idx[i_c, i_b],
                    contact_normal,
                    contact_pos,
                    penetration * weight,
                    dyn_state,
                    collider_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    errno,
                )

    # Compute other contacts
    for i_c, i_b in qd.ndrange(collider_state.contact_data.pos.shape[0], collider_state.active_buffer.shape[1]):
        if i_c < collider_state.n_contacts[i_b]:
            ref_id = collider_state.diff_contact_input.ref_id[i_b, i_c]
            is_ref = i_c == ref_id
            i_ga = collider_state.diff_contact_input.geom_a[i_b, i_c]
            i_gb = collider_state.diff_contact_input.geom_b[i_b, i_c]

            if not is_ref:
                ref_penetration = collider_state.diff_contact_input.ref_penetration[i_b, ref_id]
                contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_contact(
                    i_ga, i_gb, i_b, i_c, ref_penetration, dyn_state, diff_contact_input, collider_info
                )

                func_set_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    i_c,
                    collider_state.contact_data.pair_idx[i_c, i_b],
                    contact_normal,
                    contact_pos,
                    penetration * weight,
                    dyn_state,
                    collider_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    errno,
                )


@qd.kernel(fastcache=True)
def kernel_fill_diff_contact_input_analytic(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Populate diff_contact_input for the contacts of analytic detection paths.

    The analytic paths leave diff_contact_input unfilled, so the differentiable narrow-phase reverse would have
    nothing to reconstruct. The plane paths (func_plane_box_contact, the plane branch of func_convex_convex_contact)
    share the convention contact_pos = v - 0.5 * penetration * normal with
    normal = -normalize(R(quat_plane) @ plane_local_dir), so the convex support point is recovered as
    v = contact_pos + 0.5 * penetration * normal, and its pose-independent "core" (box vertex / sphere center /
    capsule nearest endpoint) as v - radius * normal, stored in the convex geom's local frame. PLANE is the smallest
    GEOM_TYPE so it is always geom_a after the canonical type-ordered swap. A sphere-sphere contact
    (func_sphere_sphere_contact) is reconstructed in closed form from the geom centers and radii alone, so only the
    geom identities and a self-referential ref_id are stored.
    """
    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(collider_state.contact_data.pos.shape[0], _B):
        if i_c < collider_state.n_contacts[i_b]:
            i_ga = collider_state.contact_data.geom_a[i_c, i_b]
            i_gb = collider_state.contact_data.geom_b[i_c, i_b]
            if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.SPHERE:
                collider_state.diff_contact_input.geom_a[i_b, i_c] = i_ga
                collider_state.diff_contact_input.geom_b[i_b, i_c] = i_gb
                collider_state.diff_contact_input.ref_id[i_b, i_c] = i_c
                collider_state.diff_contact_input.valid[i_b, i_c] = 1
            elif dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE:
                trans_convex = dyn_state.geoms.pos[i_gb, i_b]
                quat_convex = dyn_state.geoms.quat[i_gb, i_b]

                normal, radius = diff_gjk.func_plane_contact_frame(i_b, i_ga, i_gb, dyn_state, dyn_info)

                penetration = collider_state.contact_data.penetration[i_c, i_b]
                contact_pos = collider_state.contact_data.pos[i_c, i_b]
                support_pos = contact_pos + 0.5 * penetration * normal
                core_world = support_pos - radius * normal
                core_local = gu.qd_transform_by_quat(core_world - trans_convex, gu.qd_inv_quat(quat_convex))

                collider_state.diff_contact_input.geom_a[i_b, i_c] = i_ga
                collider_state.diff_contact_input.geom_b[i_b, i_c] = i_gb
                collider_state.diff_contact_input.core_local[i_b, i_c] = core_local
                collider_state.diff_contact_input.ref_id[i_b, i_c] = i_c
                collider_state.diff_contact_input.valid[i_b, i_c] = 1


@qd.kernel(fastcache=True)
def func_narrow_phase_convex_specializations(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if dyn_info.geoms.type[i_ga] > dyn_info.geoms.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX:
                func_plane_box_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    i_pair,
                    geoms_init_AABB,
                    dyn_state,
                    collider_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    rigid_config,
                    collider_static_config,
                    errno,
                )

            if qd.static(rigid_config.box_box_detection):
                if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.BOX and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX:
                    func_box_box_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        i_pair,
                        dyn_state,
                        collider_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        collider_static_config,
                        errno,
                    )


@qd.kernel(fastcache=True)
def func_narrow_phase_any_vs_terrain(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    """
    NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
    Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
    Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
    Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
    """
    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if qd.static(collider_static_config.has_terrain):
                if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.TERRAIN:
                    i_ga, i_gb = i_gb, i_ga

                if dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.TERRAIN:
                    func_contact_mpr_terrain(
                        i_ga,
                        i_gb,
                        i_b,
                        geoms_init_AABB,
                        dyn_state,
                        collider_state,
                        mpr_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                        errno,
                    )


@qd.kernel(fastcache=True)
def func_narrow_phase_nonconvex_vs_nonterrain(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    EPS = rigid_info.EPS[None]

    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if qd.static(collider_static_config.has_nonconvex_nonterrain):
                if (
                    not (dyn_info.geoms.is_convex[i_ga] and dyn_info.geoms.is_convex[i_gb])
                    and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.TERRAIN
                ):
                    # Place the bounded side at i_ga: the polytope helper iterates A's verts and queries B's SDF.
                    # PLANE has infinite extent so it must be i_gb. For non-PLANE pairs the smaller-AABB side goes to
                    # i_ga (its verts are dense relative to the contact patch, while the larger side's scan wastes
                    # cycles on verts far from any contact). Type-ascending order is preserved as a tiebreaker so
                    # smooth primitives (SPHERE/ELLIPSOID/CAPSULE) stay at i_ga, matching the smooth-contact refinement
                    # convention shared with the convex-vs-convex narrowphase.
                    if dyn_info.geoms.type[i_ga] > dyn_info.geoms.type[i_gb]:
                        i_ga, i_gb = i_gb, i_ga
                    if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE:
                        i_ga, i_gb = i_gb, i_ga
                    elif dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.PLANE:
                        diag_a_sq = (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]).norm_sqr()
                        diag_b_sq = (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]).norm_sqr()
                        if diag_a_sq > diag_b_sq:
                            i_ga, i_gb = i_gb, i_ga

                    tolerance = collider_info.mc_tolerance[None] * func_compute_geom_pair_scale(
                        i_ga, i_gb, geoms_init_AABB, dyn_info
                    )

                    # enable_multi_contact controls how many contacts the helper emits per pair (n_max=1 vs
                    # n_contacts_per_convex_pair); the dispatch is unconditional so disabling multi-contact never drops
                    # collisions.
                    ga_pos = dyn_state.geoms.pos[i_ga, i_b]
                    ga_quat = dyn_state.geoms.quat[i_ga, i_b]
                    gb_pos = dyn_state.geoms.pos[i_gb, i_b]
                    gb_quat = dyn_state.geoms.quat[i_gb, i_b]
                    # Nested-shell pair detection: BOTH bodies hollow (their own centers sit in their cavities),
                    # A's center outside B's material, comparable sizes. Such pairs contact along an annular line
                    # where the per-vertex manifold churns, so they use the sector-aggregated manifold; every other
                    # pair - including a hollow body near a solid one (nut around a bolt) - keeps the per-vertex one.
                    is_shell_pair = False
                    center_a_w = gu.qd_transform_by_trans_quat(dyn_info.geoms.center[i_ga], ga_pos, ga_quat)
                    rb_a_sq = gs.qd_float(0.0)
                    rb_b_sq = gs.qd_float(0.0)
                    for k in qd.static(range(8)):
                        d_a = geoms_init_AABB[i_ga, k] - dyn_info.geoms.center[i_ga]
                        d_b = geoms_init_AABB[i_gb, k] - dyn_info.geoms.center[i_gb]
                        if d_a.dot(d_a) > rb_a_sq:
                            rb_a_sq = d_a.dot(d_a)
                        if d_b.dot(d_b) > rb_b_sq:
                            rb_b_sq = d_b.dot(d_b)
                    sd_center_ab = sdf.sdf_func_world_local(
                        i_gb, center_a_w, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf
                    )
                    # Nested-only: A's center must lie inside B's local AABB. Adjacent shells share every
                    # other property of a nested pair, but their contact is a small lens ON the closing
                    # line, where the azimuthal sectors degenerate to a single bucket and the aggregate
                    # churns (cm-scale contact relocation every step, with intermittently silent steps);
                    # the per-vertex manifold handles them well. The two populations sit far from this
                    # boundary - a nested center is deep inside, an adjacent center a full bound outside -
                    # so the gate is stable in practice.
                    center_a_in_b = gu.qd_inv_transform_by_trans_quat(center_a_w, gb_pos, gb_quat)
                    is_nested = True
                    for j in qd.static(range(3)):
                        if (
                            center_a_in_b[j] < geoms_init_AABB[i_gb, 0][j]
                            or center_a_in_b[j] > geoms_init_AABB[i_gb, 7][j]
                        ):
                            is_nested = False
                    if (
                        is_nested
                        and dyn_info.geoms.is_hollow[i_ga]
                        and dyn_info.geoms.is_hollow[i_gb]
                        and qd.abs(sd_center_ab) < qd.sqrt(rb_a_sq)
                        and rb_a_sq > 0.25 * rb_b_sq
                        and sd_center_ab >= 0.0
                    ):
                        is_shell_pair = True
                    if is_shell_pair:
                        func_add_polytope_vertex_contacts_sdf_shell(
                            i_ga,
                            i_gb,
                            i_b,
                            i_pair,
                            ga_pos,
                            ga_quat,
                            gb_pos,
                            gb_quat,
                            tolerance,
                            geoms_init_AABB,
                            dyn_state,
                            collider_state,
                            dyn_info,
                            rigid_info,
                            collider_info,
                            rigid_config,
                            collider_static_config,
                            errno,
                        )
                    else:
                        func_add_polytope_vertex_contacts_sdf(
                            i_ga,
                            i_gb,
                            i_b,
                            i_pair,
                            ga_pos,
                            ga_quat,
                            gb_pos,
                            gb_quat,
                            tolerance,
                            geoms_init_AABB,
                            dyn_state,
                            collider_state,
                            dyn_info,
                            rigid_info,
                            collider_info,
                            rigid_config,
                            collider_static_config,
                            seeded=False,
                            errno=errno,
                        )
                        # Swapped-role verification: A's vertex scan cannot see a feature of B crossing one of A's
                        # faces BETWEEN A's verts, so B's verts are checked against A's SDF as well - as a seeded
                        # local search rather than a full scan. Skipped only for PLANE (its handful of far-flung
                        # verts carry no contact information).
                        if dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.PLANE:
                            func_add_polytope_vertex_contacts_sdf(
                                i_gb,
                                i_ga,
                                i_b,
                                i_pair,
                                gb_pos,
                                gb_quat,
                                ga_pos,
                                ga_quat,
                                tolerance,
                                geoms_init_AABB,
                                dyn_state,
                                collider_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                rigid_config,
                                collider_static_config,
                                seeded=True,
                                errno=errno,
                            )
