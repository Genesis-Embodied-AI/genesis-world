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
import genesis.utils.simt as su

from . import capsule_contact, diff_gjk, gjk, mpr
from .box_contact import func_box_box_contact, func_plane_box_contact, func_sphere_box_contact
from .constants import CCD_ALGORITHM_CODE, MULTICONTACT_ENTRY, MULTICONTACT_SLOT, PORTAL_STATUS
from .contact import (
    func_add_contact,
    func_add_diff_contact_input,
    func_apply_smooth_refinement,
    func_compute_geom_pair_scale,
    func_compute_mc_tolerance,
    func_contact_orthogonals,
    func_reserve_contacts,
    func_rotate_frame,
    func_set_contact,
)
from .utils import func_closest_points_on_segments, func_is_discrete_geoms, func_point_in_geom_aabb


@qd.func
def func_contact_sphere_sdf(
    i_ga: int,
    i_gb: int,
    i_b: int,
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
    i_ga: int,
    i_gb: int,
    i_b: int,
    i_pair: int,
    tid: int,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    tolerance: float,
    n_prev: int,
    contacts_slot,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    block_dim: qd.template(),
    seeded: bool,
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
    # body into a limit-cycle oscillation. The pair's first n_prev contacts sit at the buffer slots contacts_slot lists,
    # and the count and slots of its contacts once this scan emitted its own are returned. The block_dim lanes of a
    # block detect the pair together, lane tid of the block being this thread, every lane returning the same result.
    # The per-pair buffers (kept verts, visited minima, contact slots) are spread over the lanes (see
    # su.qd_lane_vector_get), which keeps a block_dim-th of each in every lane instead of a full copy in local memory.
    n_max = qd.static(collider_static_config.n_contacts_per_nonconvex_pair if rigid_config.enable_multi_contact else 1)
    N_SLOTS = qd.static((n_max + block_dim - 1) // block_dim)
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

    # The pairs of an env are detected in parallel, so the contacts of the pair's first scan are found through their
    # slots: the repeated-contact check and the shared pair budget below cover both scans.
    n_added = n_prev
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
        top_iv = qd.Vector.zero(gs.qd_int, N_SLOTS)
        top_pen = qd.Vector.zero(gs.qd_float, N_SLOTS)
        for i_k in qd.static(range(N_SLOTS)):
            top_pen[i_k] = -gs.qd_float(1e30)
            top_iv[i_k] = -1
        # Both passes walk groups of candidate verts of A (all the columns of cells as one group for the full scan, a
        # 1-ring per seed for the verification) and pick the group and the vert in runtime branches, so that the SDF
        # query and the insertion below are inlined once. This trades some runtime for compile time: the per-vert
        # branches on 'seeded' make the scan slower than two dedicated loops, each with its own copy of the query and
        # the insertion.
        n_cells_per_axis = qd.static(8)
        cell_min = qd.Vector.zero(gs.qd_int, 3)
        cell_max = qd.Vector.zero(gs.qd_int, 3)
        n_columns = 0
        n_groups = 0
        # The full scan keeps the verts within one SDF cell of B's surface. The verification only keeps UNAMBIGUOUS
        # penetrations: the first scan already supplies the near-surface approach band from the other side, and
        # duplicating it here with grid-noise-level pens and grid-smoothed normals disturbs tuned interfaces (it
        # decouples the pitch of a threaded nut-bolt pair and injects residual jitter into a settled pile). A
        # quarter-cell floor keeps grid noise out while a genuine wall crossing (at least half a wall thickness deep)
        # clears it long before locking.
        band = margin
        if not seeded:
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
                # One column per xy cell: cells along z are contiguous in the cell-range layout, so the whole z-run is
                # one vert range.
                n_columns = (cell_max[0] - cell_min[0] + 1) * (cell_max[1] - cell_min[1] + 1)
        else:
            # Seeded verification: what the pair's first scan cannot see is a feature of A (rim, edge) crossing one
            # of B's faces BETWEEN B's verts; any such feature lies in B's near-surface band, connected on A's vertex
            # graph to a local SDF minimum. So instead of scanning every A vert, hill-climb down B's SDF from each
            # first-scan contact plus the image of B's center (covers the zero-contact crossing, e.g. a thin blade
            # fully through a plate), and consider only the minimum and its 1-ring, one group per seed. The early stop
            # is a convergence dedup: seeds of the same patch resolve to an already-visited closest vert or climb into
            # a known minimum, so only the first seed of each basin pays for a climb. Cost is O(basins * climb * ring),
            # affordable for every pair including world-anchored meshes with huge vert counts. Only B's center and the
            # n_prev contacts of the first scan seed a group.
            n_groups = n_prev + 1
            band = -0.25 * margin
        seen_verts = qd.Vector.zero(gs.qd_int, N_SLOTS)
        n_seen = 0
        weakest_idx = 0
        # The vert ranges of the columns of cells are read by the lanes at once, each holding a block_dim-th of them
        # (see su.qd_lane_vector_get). A column holds a few verts only, so with several lanes the full scan walks the
        # verts of all the columns as one sequence in column order, a batch of verts spanning as many columns as it
        # takes to fill the lanes of the block. A single lane walks the columns one group after the other.
        N_COLUMN_SLOTS = qd.static((n_cells_per_axis**2 + block_dim - 1) // block_dim)
        columns_vert_range = qd.Matrix.zero(gs.qd_int, N_COLUMN_SLOTS, 2)
        n_cells_y = cell_max[1] - cell_min[1] + 1
        n_scan_verts = 0
        if not seeded:
            cells_stride = qd.Vector([n_cells_per_axis**2, n_cells_per_axis, 1])
            for i_k in range(N_COLUMN_SLOTS):
                i_column = i_k * block_dim + tid
                if i_column < n_columns:
                    cell = cell_min + qd.Vector([i_column // n_cells_y, i_column % n_cells_y, 0])
                    i_cell = i_ga * (n_cells_per_axis**3 + 1) + cell.dot(cells_stride)
                    i_cell_end = i_cell + cell_max[2] - cell_min[2] + 1
                    column_vert_range = qd.Vector(
                        [collider_info.verts_spatial_grid.cells_vert_start[i] for i in qd.static((i_cell, i_cell_end))]
                    )
                    su.qd_lane_local_set(columns_vert_range, i_k, column_vert_range, block_dim)
                    n_scan_verts = n_scan_verts + column_vert_range[1] - column_vert_range[0]
            n_groups = n_columns
            if qd.static(block_dim > 1):
                n_scan_verts = su.qd_block_sum(n_scan_verts)
                n_groups = qd.min(n_columns, 1)
        # Most columns are empty, so the batches jump from one non-empty column to the next through a ballot of the
        # lanes per row of columns.
        lanes_column_left = qd.Vector.zero(qd.u32, N_COLUMN_SLOTS)
        if qd.static(block_dim > 1):
            for i_k in qd.static(range(N_COLUMN_SLOTS)):
                column_vert_range = su.qd_lane_local_get(columns_vert_range, i_k, block_dim)
                lanes_column_left[i_k] = su.qd_block_ballot(
                    i_k * block_dim + tid < n_columns and column_vert_range[1] > column_vert_range[0]
                )
        # The start and count of the verts left in the column the batches of the full scan walk
        column_vert_start = 0
        n_column_verts_left = 0
        for i_group in range(n_groups):
            i_v_min = -1
            i_c_start = 0
            i_c_end = 0
            if not seeded:
                if qd.static(block_dim > 1):
                    i_c_end = n_scan_verts
                else:
                    column_vert_range = su.qd_lane_slot_read(columns_vert_range, i_group, block_dim)
                    i_c_start = column_vert_range[0]
                    i_c_end = column_vert_range[1]
            else:
                if i_group == 0:
                    i_v_min = sdf.sdf_func_find_closest_vert(
                        i_ga, i_b, center_b_world, dyn_state, dyn_info, collider_info
                    )
                elif i_group <= n_prev:
                    i_c_seed = su.qd_lane_slot_read(contacts_slot, n_prev - i_group, block_dim)
                    seed_pos = collider_state.contact_data.pos[i_c_seed, i_b]
                    i_v_min = sdf.sdf_func_find_closest_vert(i_ga, i_b, seed_pos, dyn_state, dyn_info, collider_info)
                if i_v_min >= 0:
                    if su.qd_is_lane_slot_seen(seen_verts, n_seen, i_v_min, tid, block_dim):
                        i_v_min = -1
                if i_v_min >= 0:
                    i_v_start = i_v_min
                    pos_min = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v_min], ga_pos, ga_quat)
                    sd_min = sdf.sdf_func_world_local(i_gb, pos_min, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf)
                    i_v_cur = -1
                    while i_v_cur != i_v_min:
                        i_v_cur = i_v_min
                        i_n_start = collider_info.vert_neighbor_start[i_v_cur]
                        n_neighbors = collider_info.vert_n_neighbors[i_v_cur]
                        # The lanes evaluate a block_dim of neighbors at once, the block keeping the lowest SD and,
                        # among equal ones, the first neighbor in order, which is the one the serial walk keeps. A
                        # lane without neighbor holds the current minimum, which never displaces it.
                        for i_chunk_ in range((n_neighbors + block_dim - 1) // block_dim):
                            i_n_ = i_chunk_ * block_dim + tid
                            i_neighbor = -1
                            sd_neighbor = sd_min
                            if i_n_ < n_neighbors:
                                i_neighbor = collider_info.vert_neighbors[i_n_start + i_n_]
                                pos_neighbor = gu.qd_transform_by_trans_quat(
                                    dyn_info.verts.init_pos[i_neighbor], ga_pos, ga_quat
                                )
                                sd_neighbor = sdf.sdf_func_world_local(
                                    i_gb, pos_neighbor, gb_pos, gb_quat, dyn_info.geoms, collider_info.sdf
                                )
                            if qd.static(block_dim > 1):
                                sd_best = su.qd_block_min(sd_neighbor)
                                i_lane = qd.math.ffs(su.qd_block_ballot(sd_neighbor == sd_best)) - 1
                                i_neighbor = su.qd_block_shuffle(i_neighbor, i_lane)
                                sd_neighbor = sd_best
                            # Strict decrease guarantees termination: values strictly descend over a finite vertex
                            # set, so no vertex can be revisited even on numerically flat plateaus.
                            if sd_neighbor < sd_min:
                                i_v_min = i_neighbor
                                sd_min = sd_neighbor
                    # The final minimum is checked against the basins found so far BEFORE this seed's own start is
                    # recorded, so a seed that starts at the minimum still emits.
                    if su.qd_is_lane_slot_seen(seen_verts, n_seen, i_v_min, tid, block_dim):
                        i_v_min = -1
                    if n_seen < n_max:
                        su.qd_lane_slot_write(seen_verts, n_seen, i_v_start, tid, block_dim)
                        n_seen = n_seen + 1
                    if i_v_min >= 0 and i_v_min != i_v_start and n_seen < n_max:
                        su.qd_lane_slot_write(seen_verts, n_seen, i_v_min, tid, block_dim)
                        n_seen = n_seen + 1
                if i_v_min >= 0:
                    # The leading index of the range selects the minimum itself, the others its 1-ring.
                    i_c_start = collider_info.vert_neighbor_start[i_v_min] - 1
                    i_c_end = collider_info.vert_neighbor_start[i_v_min] + collider_info.vert_n_neighbors[i_v_min]
            # The lanes split the vert range in batches of block_dim verts, each lane running the gate and the SDF query
            # of one vert of the batch. The verts kept for insertion are then inserted one at a time in vert order,
            # every lane updating its own copy of the buffer alike, so the buffer is the one the serial scan selects.
            for i_batch in range((i_c_end - i_c_start + block_dim - 1) // block_dim):
                # Read by every lane ahead of the per-vert branches, which only some lanes enter
                pen_weakest = su.qd_lane_slot_read(top_pen, weakest_idx, block_dim)
                i_c = i_c_start + i_batch * block_dim + tid
                is_vert = i_c < i_c_end
                if qd.static(block_dim > 1):
                    if not seeded:
                        # The lanes take the next verts of the columns in order, from the column the last batch stopped in.
                        i_c = -1
                        n_lanes_filled = 0
                        for i_fill_ in range(n_columns):
                            if n_lanes_filled == block_dim:
                                break
                            if n_column_verts_left == 0:
                                i_column = su.qd_lane_mask_pop_first(lanes_column_left, n_columns, block_dim)
                                if i_column == n_columns:
                                    break
                                column_vert_range = su.qd_lane_slot_read(columns_vert_range, i_column, block_dim)
                                column_vert_start = column_vert_range[0]
                                n_column_verts_left = column_vert_range[1] - column_vert_range[0]
                            n_taken = qd.min(n_column_verts_left, block_dim - n_lanes_filled)
                            if tid >= n_lanes_filled and tid < n_lanes_filled + n_taken:
                                i_c = column_vert_start + tid - n_lanes_filled
                            n_lanes_filled = n_lanes_filled + n_taken
                            column_vert_start = column_vert_start + n_taken
                            n_column_verts_left = n_column_verts_left - n_taken
                        is_vert = i_c >= 0
                i_v = i_v_min
                vertex_pos = qd.Vector.zero(gs.qd_float, 3)
                pen_v = gs.qd_float(0.0)
                is_inserted = False
                if is_vert:
                    vertex_local = qd.Vector.zero(gs.qd_float, 3)
                    if not seeded:
                        i_v = collider_info.verts_spatial_grid.verts_idx[i_c]
                        vertex_local = collider_info.verts_spatial_grid.verts_pos[i_c]
                    else:
                        if i_c > i_c_start:
                            i_v = collider_info.vert_neighbors[i_c]
                        vertex_local = dyn_info.verts.init_pos[i_v]
                    vertex_pos = gu.qd_transform_by_trans_quat(vertex_local, ga_pos, ga_quat)
                    is_candidate = True
                    if not seeded:
                        is_candidate = func_point_in_geom_aabb(i_gb, i_b, vertex_pos, 0.0, dyn_state)
                    if is_candidate:
                        is_in_band, sd_v = sdf.sdf_func_world_local_banded(
                            i_gb, vertex_pos, gb_pos, gb_quat, band, dyn_info, collider_info
                        )
                        pen_v = -sd_v
                        # A vert no deeper than the shallowest kept one (at weakest_idx, the first of minimal pen, an
                        # empty slot being the shallowest) can neither replace a close kept vert nor displace the
                        # weakest, which spares the insertion for most verts of a dense mesh once the buffer is full.
                        # Insertions only ever deepen the buffer, so a vert failing this test at the start of its batch
                        # fails it at its turn as well.
                        is_inserted = is_in_band and pen_v > pen_weakest
                # The lanes holding a vert to insert, taken in lane order, which is vert order
                n_inserted = qd.cast(is_inserted, gs.qd_int)
                lanes_inserted = qd.u32(0)
                if qd.static(block_dim > 1):
                    lanes_inserted = su.qd_block_ballot(is_inserted)
                    n_inserted = qd.cast(qd.math.popcnt(lanes_inserted), gs.qd_int)
                for i_rank in range(n_inserted):
                    i_v_new = i_v
                    pos_new = vertex_pos
                    pen_new = pen_v
                    if qd.static(block_dim > 1):
                        i_lane = qd.math.ffs(lanes_inserted) - 1
                        lanes_inserted = lanes_inserted & (lanes_inserted - 1)
                        i_v_new = su.qd_block_shuffle(i_v, i_lane)
                        pen_new = su.qd_block_shuffle(pen_v, i_lane)
                        for j in qd.static(range(3)):
                            pos_new[j] = su.qd_block_shuffle(vertex_pos[j], i_lane)
                    if pen_new > su.qd_lane_slot_read(top_pen, weakest_idx, block_dim):
                        # Each lane tests the kept verts it holds, the block keeping the first one closer than the
                        # diversity radius.
                        i_close = n_max
                        for i_k in range(N_SLOTS):
                            k = i_k * block_dim + tid
                            if k < n_max:
                                i_v_kept = su.qd_lane_local_get(top_iv, i_k, block_dim)
                                if i_v_kept >= 0:
                                    other_pos = gu.qd_transform_by_trans_quat(
                                        dyn_info.verts.init_pos[i_v_kept], ga_pos, ga_quat
                                    )
                                    if (pos_new - other_pos).norm() < diversity_radius:
                                        i_close = k
                                        break
                        if qd.static(block_dim > 1):
                            i_close = su.qd_lane_slot_min(i_close, n_max, N_SLOTS, block_dim)
                        i_put = weakest_idx
                        if i_close < n_max:
                            i_put = -1
                            if pen_new > su.qd_lane_slot_read(top_pen, i_close, block_dim):
                                i_put = i_close
                        if i_put >= 0:
                            su.qd_lane_slot_write(top_pen, i_put, pen_new, tid, block_dim)
                            su.qd_lane_slot_write(top_iv, i_put, i_v_new, tid, block_dim)
                            # Each lane finds the first slot of minimal pen among the slots it holds, the block keeping
                            # the lowest slot holding the minimum.
                            pen_min = su.qd_lane_slot_read(top_pen, weakest_idx, block_dim)
                            weakest_idx = n_max
                            for i_k in range(N_SLOTS):
                                k = i_k * block_dim + tid
                                if k < n_max:
                                    pen_k = su.qd_lane_local_get(top_pen, i_k, block_dim)
                                    if weakest_idx == n_max or pen_k < pen_min:
                                        pen_min = pen_k
                                        weakest_idx = k
                            if qd.static(block_dim > 1):
                                pen_min_block = su.qd_block_min(pen_min)
                                weakest_idx = su.qd_lane_slot_min(
                                    weakest_idx if pen_min <= pen_min_block else n_max, n_max, N_SLOTS, block_dim
                                )

        # Pass 2: emit contacts at the selected vertices. Reference normal is the grid SDF gradient sampled at A's
        # center, which gives a stable per-patch normal whenever A is small relative to B's features. Per-vertex
        # grads agreeing with that reference (positive dot product) are kept verbatim - this is what allows an A
        # wedged at a concave L-corner to expose both axis-aligned face normals (floor + wall). A per-vertex grad
        # that opposes the reference indicates the vertex is closer to B's OPPOSITE surface across a thin feature,
        # i.e. A has partially tunneled through; using its raw grad would push A further through, so we fall back
        # to the reference direction.
        # The pair-level normal and depth below only serve the kept verts, so a single lane skips them for a scan that
        # kept none, where the lanes of a block run them at no gain. The buffer fills from its first slot, the weakest
        # slot being the first empty one, so it holds a vert if that one does.
        is_vert_kept = True
        if qd.static(block_dim == 1):
            is_vert_kept = top_iv[0] >= 0
        grad_center = qd.Vector.zero(gs.qd_float, 3)
        if is_vert_kept:
            grad_center = sdf.sdf_func_grad_world_local(
                i_gb,
                center_a_world,
                gb_pos,
                gb_quat,
                dyn_info.geoms,
                rigid_info,
                collider_info.sdf,
                collider_static_config,
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
        is_closing_regime = is_vert_kept and qd.abs(sd_center) < rbound_a and rbound_a_sq > 0.25 * rbound_b_sq
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
        # The normal and the penetration of every kept vert depend on that vert alone, so each lane works out those of
        # the slots it holds (see su.qd_lane_vector_get), and the contacts are then emitted one slot after the other.
        emit_pen = qd.Vector.zero(gs.qd_float, N_SLOTS)
        emit_normal = qd.Matrix.zero(gs.qd_float, N_SLOTS, 3)
        for i_k in range(N_SLOTS):
            i_v = su.qd_lane_local_get(top_iv, i_k, block_dim)
            pen_v = su.qd_lane_local_get(top_pen, i_k, block_dim)
            if i_k * block_dim + tid < n_max and i_v >= 0:
                vertex_pos = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v], ga_pos, ga_quat)
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
                su.qd_lane_local_set(emit_pen, i_k, pen_emit, block_dim)
                su.qd_lane_local_set(emit_normal, i_k, normal_v, block_dim)
        # Most slots are left empty by the scan, so on GPU the emission only visits the occupied ones, taken in slot
        # order from a ballot of the lanes per row of slots.
        n_emit_slots = n_max
        lanes_occupied = qd.Vector.zero(qd.u32, N_SLOTS)
        if qd.static(block_dim > 1):
            n_emit_slots = 0
            for i_k in qd.static(range(N_SLOTS)):
                lanes_occupied[i_k] = su.qd_block_ballot(
                    i_k * block_dim + tid < n_max and su.qd_lane_local_get(top_iv, i_k, block_dim) >= 0
                )
                n_emit_slots = n_emit_slots + qd.cast(qd.math.popcnt(lanes_occupied[i_k]), gs.qd_int)
        # Where the scan cannot reach the pair budget, on GPU its contacts are decided first, each lane keeping the
        # positions of the slots it holds for the repeated-contact check, then written at once from a single reservation
        # of the contact buffer, one lane per contact. At the budget, a contact may displace an earlier one, so they are
        # written one after the other.
        is_batched = False
        if qd.static(block_dim > 1):
            is_batched = n_added + n_emit_slots <= n_max
        lanes_emitted = qd.Vector.zero(qd.u32, N_SLOTS)
        emit_pos = qd.Matrix.zero(gs.qd_float, N_SLOTS, 3)
        for i_emit in range(n_emit_slots):
            k = i_emit
            if qd.static(block_dim > 1):
                k = su.qd_lane_mask_pop_first(lanes_occupied, n_max, block_dim)
            i_v = su.qd_lane_slot_read(top_iv, k, block_dim)
            if i_v >= 0:
                pen_emit = su.qd_lane_slot_read(emit_pen, k, block_dim)
                normal_v = su.qd_lane_slot_read(emit_normal, k, block_dim)
                contact_pos_v = gu.qd_transform_by_trans_quat(dyn_info.verts.init_pos[i_v], ga_pos, ga_quat)
                # Each lane tests the contacts of the pair whose slots it holds, the block combining the answers
                is_repeated = False
                for i_k in range(N_SLOTS):
                    if i_k * block_dim + tid < n_added:
                        idx_prev = su.qd_lane_local_get(contacts_slot, i_k, block_dim)
                        if (contact_pos_v - collider_state.contact_data.pos[idx_prev, i_b]).norm() < tolerance:
                            is_repeated = True
                if qd.static(block_dim > 1):
                    for i_k in qd.static(range(N_SLOTS)):
                        if (lanes_emitted[i_k] >> qd.u32(tid)) & qd.u32(1) != 0:
                            if (contact_pos_v - emit_pos[i_k, :]).norm() < tolerance:
                                is_repeated = True
                    is_repeated = su.qd_block_any(is_repeated)
                if not is_repeated and pen_emit > 0.0:
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
                    if is_batched:
                        su.qd_lane_slot_write(emit_pos, k, contact_pos_v, tid, block_dim)
                        for i_k in qd.static(range(N_SLOTS)):
                            if k // block_dim == i_k:
                                lanes_emitted[i_k] = lanes_emitted[i_k] | (qd.u32(1) << qd.u32(k % block_dim))
                    else:
                        # The first lane writes the contacts of the block, whose slots every lane reads back. A new
                        # contact takes a fresh slot of the buffer, and the contact it displaces at the cap otherwise,
                        # both written by the same call.
                        i_c = -1
                        if n_added < n_max:
                            if tid == 0:
                                i_c = func_reserve_contacts(
                                    i_b,
                                    n_contacts=1,
                                    collider_state=collider_state,
                                    collider_info=collider_info,
                                    use_atomic=qd.static(rigid_config.para_level >= gs.PARA_LEVEL.PARTIAL),
                                    errno=errno,
                                )
                                if i_c >= collider_info.max_candidate_contacts[None]:
                                    i_c = -1
                            if qd.static(block_dim > 1):
                                i_c = su.qd_block_shuffle(i_c, 0)
                            if i_c >= 0:
                                su.qd_lane_slot_write(contacts_slot, n_added, i_c, tid, block_dim)
                                n_added = n_added + 1
                        else:
                            # The pair budget is shared between the two scans: at the cap, a new contact may only
                            # displace the pair's least-penetrating one, keeping the n_max deepest.
                            weakest_pen = pen_emit
                            for j in range(n_max):
                                if j < n_added:
                                    idx_prev = su.qd_lane_slot_read(contacts_slot, n_added - 1 - j, block_dim)
                                    if collider_state.contact_data.penetration[idx_prev, i_b] < weakest_pen:
                                        weakest_pen = collider_state.contact_data.penetration[idx_prev, i_b]
                                        i_c = idx_prev
                        if i_c >= 0 and tid == 0:
                            func_set_contact(
                                i_ga,
                                i_gb,
                                i_b,
                                i_c,
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
                        # Every lane reads the contact back, for the repeated-contact check and the seeds.
                        if qd.static(block_dim > 1):
                            qd.simt.block.sync()
        if is_batched:
            n_emitted = 0
            for i_k in qd.static(range(N_SLOTS)):
                n_emitted = n_emitted + qd.cast(qd.math.popcnt(lanes_emitted[i_k]), gs.qd_int)
            if n_emitted > 0:
                i_c_base = 0
                if tid == 0:
                    i_c_base = func_reserve_contacts(
                        i_b,
                        n_emitted,
                        collider_state,
                        collider_info,
                        use_atomic=qd.static(rigid_config.para_level >= gs.PARA_LEVEL.PARTIAL),
                        errno=errno,
                    )
                i_c_base = su.qd_block_shuffle(i_c_base, 0)
                max_contacts = collider_info.max_candidate_contacts[None]
                # The contacts take the reserved slots in the order of the slots of their verts
                n_emitted_before = 0
                for i_k in range(N_SLOTS):
                    lanes = su.qd_lane_local_get(lanes_emitted, i_k, block_dim)
                    if (lanes >> qd.u32(tid)) & qd.u32(1) != 0:
                        i_c = i_c_base + n_emitted_before
                        i_c = i_c + qd.cast(qd.math.popcnt(lanes & ((qd.u32(1) << qd.u32(tid)) - qd.u32(1))), gs.qd_int)
                        if i_c < max_contacts:
                            func_set_contact(
                                i_ga,
                                i_gb,
                                i_b,
                                i_c,
                                i_pair,
                                su.qd_lane_local_get(emit_normal, i_k, block_dim),
                                su.qd_lane_local_get(emit_pos, i_k, block_dim),
                                su.qd_lane_local_get(emit_pen, i_k, block_dim),
                                dyn_state,
                                collider_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                errno,
                            )
                    n_emitted_before = n_emitted_before + qd.cast(qd.math.popcnt(lanes), gs.qd_int)
                n_stored = qd.max(qd.min(max_contacts - i_c_base, n_emitted), 0)
                for i_k in qd.static(range(N_SLOTS)):
                    i_added = i_k * block_dim + tid
                    if i_added >= n_added and i_added < n_added + n_stored:
                        contacts_slot[i_k] = i_c_base + i_added - n_added
                n_added = n_added + n_stored
                # Every lane reads the contacts back in the next scan of the pair, for its seeds and repeated contacts.
                if qd.static(block_dim > 1):
                    qd.simt.block.sync()
    return n_added, contacts_slot


@qd.func
def func_add_polytope_vertex_contacts_sdf_shell(
    i_ga: int,
    i_gb: int,
    i_b: int,
    i_pair: int,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    tolerance: float,
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
                use_atomic=qd.static(rigid_config.para_level >= gs.PARA_LEVEL.PARTIAL),
                errno=errno,
            )


@qd.func
def func_contact_vertex_sdf(
    i_ga: int,
    i_gb: int,
    i_b: int,
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
    i_ga: int,
    i_gb: int,
    i_b: int,
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
    i_ga: int,
    i_gb: int,
    i_b: int,
    i_va_ws: int,
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
    i_ga: int,
    i_gb: int,
    i_scratch: int,
    i_b: int,
    tid: int,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    block_dim: qd.template(),
    errno: qd.Tensor,
):
    """Detect the contacts of a convex geom against the terrain, the lanes of a block sharing the work.

    The geom is tested with MPR against the prism under every triangle of the height field its bounds overlap, the
    block_dim lanes of the block sharing the work, lane tid on slot i_scratch of the prism scratch and of mpr_state. The
    contacts are those of a serial walk over the detections and triangles, whatever the number of lanes.
    """
    ga_pos, ga_quat = dyn_state.geoms.pos[i_ga, i_b], dyn_state.geoms.quat[i_ga, i_b]
    gb_pos, gb_quat = dyn_state.geoms.pos[i_gb, i_b], dyn_state.geoms.quat[i_gb, i_b]
    margin = gs.qd_float(0.0)
    EPS = rigid_info.EPS[None]
    n_max = qd.static(collider_static_config.n_contacts_per_convex_pair)
    N_PERTURBATIONS = qd.static(4)

    multi_contact = (
        qd.static(rigid_config.enable_multi_contact)
        and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.SPHERE
        and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
    )

    is_return = False
    tolerance = collider_info.mc_tolerance[None] * func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
    i_pair = collider_info.collision_pair_idx[i_ga, i_gb]

    # Transform to terrain's frame (using local variables, not modifying global state)
    ga_pos_terrain_frame, ga_quat_terrain_frame = gu.qd_transform_pos_quat_by_trans_quat(
        ga_pos - gb_pos, ga_quat, qd.Vector.zero(gs.qd_float, 3), gu.qd_inv_quat(gb_quat)
    )
    gb_pos_terrain_frame = qd.Vector.zero(gs.qd_float, 3)
    gb_quat_terrain_frame = gu.qd_identity_quat()

    # Bounds of the geom in the frame of the terrain, max corner then min corner, one support query per lane
    xyz_max_min = qd.Vector.zero(gs.qd_float, 6)
    for i_dir_ in range((6 + block_dim - 1) // block_dim):
        i_dir = i_dir_ * block_dim + tid
        extent = gs.qd_float(0.0)
        if i_dir < 6:
            i_axis = i_dir % 3
            direction = qd.Vector.zero(gs.qd_float, 3)
            direction[i_axis] = 1.0 if i_dir < 3 else -1.0
            v1 = mpr.support_driver(
                i_ga,
                i_scratch,
                direction,
                ga_pos_terrain_frame,
                ga_quat_terrain_frame,
                collider_state,
                dyn_info,
                collider_info,
                rigid_config,
                collider_static_config,
            )
            extent = v1[i_axis]
        if qd.static(block_dim > 1):
            for k in qd.static(range(6)):
                xyz_max_min[k] = su.qd_block_shuffle(extent, k)
        else:
            xyz_max_min[i_dir] = extent

    for i in qd.static(range(3)):
        if (
            collider_info.terrain_xyz_maxmin[i] < xyz_max_min[i + 3] - margin
            or collider_info.terrain_xyz_maxmin[i + 3] > xyz_max_min[i] + margin
        ):
            is_return = True

    if not is_return:
        sh = collider_info.terrain_scale[0]
        r_min = gs.qd_int(qd.floor((xyz_max_min[3] - collider_info.terrain_xyz_maxmin[3]) / sh))
        r_max = gs.qd_int(qd.ceil((xyz_max_min[0] - collider_info.terrain_xyz_maxmin[3]) / sh))
        c_min = gs.qd_int(qd.floor((xyz_max_min[4] - collider_info.terrain_xyz_maxmin[4]) / sh))
        c_max = gs.qd_int(qd.ceil((xyz_max_min[1] - collider_info.terrain_xyz_maxmin[4]) / sh))

        r_min = qd.max(0, r_min)
        c_min = qd.max(0, c_min)
        r_max = qd.min(collider_info.terrain_rc[0] - 1, r_max)
        c_max = qd.min(collider_info.terrain_rc[1] - 1, c_max)

        # Row r lists its verts (r, c_min), (r + 1, c_min), (r, c_min + 1), ..., every three consecutive verts making a
        # triangle, which cuts every cell along the diagonal of the visual mesh (see genesis/utils/terrain.py)
        n_triangles_row = 2 * qd.max(c_max - c_min, 0)
        n_triangles = qd.max(r_max - r_min, 0) * n_triangles_row

        # The perturbed detections rotate geom A by a small angle about an axis orthogonal to the normal of the first
        # contact, which tips a box face onto another corner of the contact patch. They only run when the first contact
        # is face-vs-face (snap fired).
        is_col_0 = False
        face_face_0 = False
        contact_pos_0 = qd.Vector.zero(gs.qd_float, 3)
        normal_0 = qd.Vector.zero(gs.qd_float, 3)
        axis_0 = qd.Vector.zero(gs.qd_float, 3)
        axis_1 = qd.Vector.zero(gs.qd_float, 3)
        n_con = 0
        # Contacts of the pair, spread over the lanes (see su.qd_lane_vector_get), which a new contact merges against
        # since the other pairs of the env append theirs to the buffer concurrently
        contacts_pos = qd.Matrix.zero(gs.qd_float, qd.static((n_max + block_dim - 1) // block_dim), 3)

        # Pass 0 walks the triangles of the first detection, pass 1 those of every perturbed detection in turn
        for i_pass in range(2):
            n_items = n_triangles
            if i_pass == 1:
                n_items = 0
                if multi_contact and face_face_0:
                    n_items = N_PERTURBATIONS * n_triangles
            for i_batch_ in range((n_items + block_dim - 1) // block_dim):
                if n_con < n_max:
                    i_item = i_batch_ * block_dim + tid
                    i_detection = 0
                    i_triangle = i_item
                    if i_pass == 1:
                        i_detection = 1 + i_item // n_triangles
                        i_triangle = i_item % n_triangles
                    ga_pos_tf = ga_pos_terrain_frame
                    ga_quat_tf = ga_quat_terrain_frame
                    if i_detection > 0:
                        axis = axis_0 if i_detection < 3 else axis_1
                        angle_sign = 2 * (i_detection % 2) - 1
                        qrot = gu.qd_rotvec_to_quat(angle_sign * collider_info.mc_perturbation[None] * axis, EPS)
                        ga_pos_curr, ga_quat_curr = func_rotate_frame(ga_pos, ga_quat, contact_pos_0, qrot)
                        ga_pos_tf, ga_quat_tf = gu.qd_transform_pos_quat_by_trans_quat(
                            ga_pos_curr - gb_pos, ga_quat_curr, qd.Vector.zero(gs.qd_float, 3), gu.qd_inv_quat(gb_quat)
                        )
                    center_a = gu.qd_transform_by_trans_quat(dyn_info.geoms.center[i_ga], ga_pos_tf, ga_quat_tf)
                    is_col = False
                    is_contact_candidate = False
                    face_face = False
                    normal = qd.Vector.zero(gs.qd_float, 3)
                    contact_pos = qd.Vector.zero(gs.qd_float, 3)
                    penetration = gs.qd_float(0.0)
                    if i_item < n_items:
                        r = r_min + i_triangle // n_triangles_row
                        i_v_end = i_triangle % n_triangles_row + 2
                        # A triangle all of whose verts lie below the geom cannot touch it
                        is_reachable = False
                        for i_p in qd.static(range(3)):
                            r_v = r + (i_v_end - 2 + i_p) % 2
                            c_v = c_min + (i_v_end - 2 + i_p) // 2
                            x = sh * r_v + collider_info.terrain_xyz_maxmin[3]
                            y = sh * c_v + collider_info.terrain_xyz_maxmin[4]
                            z = collider_info.terrain_hf[r_v, c_v] + margin
                            collider_state.prism[i_p, i_scratch] = qd.Vector(
                                [x, y, collider_info.terrain_xyz_maxmin[5]], dt=gs.qd_float
                            )
                            collider_state.prism[i_p + 3, i_scratch] = qd.Vector([x, y, z], dt=gs.qd_float)
                            if z >= xyz_max_min[5]:
                                is_reachable = True
                        if is_reachable:
                            center_b = qd.Vector.zero(gs.qd_float, 3)
                            for i_p in qd.static(range(6)):
                                center_b = center_b + collider_state.prism[i_p, i_scratch]
                            center_b = center_b / 6.0

                            is_col, normal, penetration, contact_pos = mpr.func_mpr_contact_from_centers(
                                i_ga,
                                i_gb,
                                i_scratch,
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
                        # Snap normal to the prism's top face normal when MPR's reported normal is already close to it.
                        # Cell boundaries on a SMOOTH heightfield are discretization artefacts, not physical edges, and
                        # MPR's polytope-edge radial normal there picks up a small position-dependent bias relative to
                        # the exact face normal. Only snap when the bias is small (dot > 0.95) so that contacts on real
                        # cliff edges, where MPR's normal is genuinely far from any single cell's top face normal, keep
                        # MPR's result.
                        e1 = collider_state.prism[4, i_scratch] - collider_state.prism[3, i_scratch]
                        e2 = collider_state.prism[5, i_scratch] - collider_state.prism[3, i_scratch]
                        top_face_normal = e1.cross(e2).normalized()
                        if top_face_normal[2] < 0.0:
                            top_face_normal = -top_face_normal
                        if top_face_normal.dot(normal) > 0.95:
                            normal = top_face_normal
                            snap_fired = True
                            # Cell is essentially horizontal in the terrain's local frame (within ~8 deg). Independent
                            # of the terrain's world orientation.
                            face_face = top_face_normal[2] > 0.99

                        normal = gu.qd_transform_by_quat(normal, gb_quat)
                        contact_pos = gu.qd_transform_by_quat(contact_pos, gb_quat)
                        contact_pos = contact_pos + gb_pos

                        # No perturbation correction: the perturbation magnitude (mc_perturbation) is small enough that
                        # the perturbed contact_pos stays close to the unperturbed contact patch. The deduplication
                        # tolerance downstream picks the unique corner contacts and the constraint solver tolerates the
                        # residual offset.

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

                        # Perturbed contacts are only kept when they still describe a face-vs-face contact on the same
                        # horizontal cell face. A perturbed corner that landed against a cliff wall (snap did not fire)
                        # is a fictitious contact.
                        is_contact_candidate = i_detection == 0 or snap_fired

                    lanes_candidate = qd.cast(is_contact_candidate, qd.u32)
                    if qd.static(block_dim > 1):
                        lanes_candidate = su.qd_block_ballot(is_contact_candidate)
                    for _ in range(qd.cast(qd.math.popcnt(lanes_candidate), gs.qd_int)):
                        i_lane = qd.cast(qd.math.ffs(lanes_candidate), gs.qd_int) - 1
                        lanes_candidate = lanes_candidate & (lanes_candidate - 1)
                        contact_pos_lane = contact_pos
                        normal_lane = normal
                        is_face_face_lane = face_face
                        if qd.static(block_dim > 1):
                            contact_pos_lane = qd.Vector(
                                [su.qd_block_shuffle(contact_pos[k], i_lane) for k in qd.static(range(3))]
                            )
                            normal_lane = qd.Vector(
                                [su.qd_block_shuffle(normal[k], i_lane) for k in qd.static(range(3))]
                            )
                            is_face_face_lane = su.qd_block_shuffle(qd.cast(face_face, gs.qd_int), i_lane) != 0

                        if n_con < n_max:
                            is_contact_duplicate = False
                            for i_k in range(contacts_pos.n):
                                if i_k * block_dim + tid < n_con:
                                    contact_pos_prev = su.qd_lane_local_get(contacts_pos, i_k, block_dim)
                                    if (contact_pos_lane - contact_pos_prev).norm() < tolerance:
                                        is_contact_duplicate = True
                            if qd.static(block_dim > 1):
                                is_contact_duplicate = su.qd_block_any(is_contact_duplicate)

                            if not is_contact_duplicate:
                                if tid == i_lane:
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
                                su.qd_lane_slot_write(contacts_pos, n_con, contact_pos_lane, tid, block_dim)
                                n_con = n_con + 1
                                if i_pass == 0 and not is_col_0:
                                    is_col_0 = True
                                    # Perturbation only applies on essentially-horizontal cell faces (in the terrain's
                                    # local frame, within ~8 deg of the heightfield +z): on steeper slopes the corner
                                    # contacts generated by perturbation map to different parts of neighbouring cells
                                    # and create constraint-solver oscillation.
                                    face_face_0 = is_face_face_lane
                                    contact_pos_0 = contact_pos_lane
                                    normal_0 = normal_lane
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
def func_recompute_perturbed_contact(
    i_ga: int,
    i_gb: int,
    i_scratch: int,
    normal: qd.types.vector(3),
    penetration: float,
    contact_pos: qd.types.vector(3),
    normal_0: qd.types.vector(3),
    contact_pos_0: qd.types.vector(3),
    penetration_0: float,
    qrot: qd.types.vector(4),
    ga_pos_original: qd.types.vector(3),
    ga_quat_original: qd.types.vector(4),
    gb_pos_original: qd.types.vector(3),
    gb_quat_original: qd.types.vector(4),
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    geom_pair_scale: float,
    used_gjk: bool,
    used_gjk_0: bool,
):
    """
    Bring a contact found on the perturbed geoms back to the unperturbed geoms, on the patch of contact 0 or its own.

    The witnesses of the perturbed detection, un-rotated by the perturbation 'qrot' of their own geom, are exact
    material points of the unperturbed geoms. A contact whose depth along the normal of contact 0 'normal_0' can be
    estimated at its witnesses lies on the patch of contact 0 and takes that normal, and one of negative depth, past the
    edge of the patch, moves towards contact 0 ('contact_pos_0', 'penetration_0') to where its depth vanishes. Any
    other contact keeps the normal of the face of the Minkowski difference that the detection found, on its own patch.
    'used_gjk_0' says whether Gilbert-Johnson-Keerthi (GJK) rather than Minkowski portal refinement (MPR) detected
    contact 0, which sets the accuracy of its depth.

    Returns the normal, the non-negative depth and the position.
    """
    EPS = rigid_info.EPS[None]

    # qrot is applied to geom A and its inverse to geom B
    R = gu.qd_quat_to_R(qrot, EPS)
    R_inv = R.transpose()
    witness_a = contact_pos - 0.5 * penetration * normal
    witness_b = contact_pos + 0.5 * penetration * normal
    if used_gjk:
        # The polytope face normal can differ from the witness separation, so recover each material point directly
        witness_a = gjk_state.witness.point_obj1[i_scratch, 0]
        witness_b = gjk_state.witness.point_obj2[i_scratch, 0]
    witness_a = R_inv @ (witness_a - contact_pos_0) + contact_pos_0
    witness_b = R @ (witness_b - contact_pos_0) + contact_pos_0

    # Geom A lies on the positive side of the normal, so that the depth at the witness of geom A is the height of the
    # surface of geom B above it, and the depth at the witness of geom B its height above the surface of geom A
    depth = normal_0.dot(witness_b - witness_a)
    # The point steps back from the witness of geom B by half the depth: averaging both witnesses would slide it off the
    # feature by an amount quadratic in the perturbation angle
    contact_pos = witness_b - 0.5 * depth * normal_0
    has_depth_estimate = False
    # Whether the witness of geom B rather than geom A stands on the feature of the contact, where the depth is measured
    is_witness_b_anchor = False
    # Support points of the detection on the unperturbed geoms, one per row, of geom A and of geom B
    supports_a = qd.Matrix.zero(gs.qd_float, 3, 3)
    supports_b = qd.Matrix.zero(gs.qd_float, 3, 3)
    has_supports = False
    if dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.PLANE:
        # The plane of geom A passes through its witness of contact 0, with the normal of contact 0
        depth = normal_0.dot(witness_b - contact_pos_0) + 0.5 * penetration_0
        contact_pos = witness_b - 0.5 * depth * normal_0
        has_depth_estimate = True
    elif not (
        (dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE)
        or (used_gjk and gjk_state.nearest_face[i_scratch] < 0)
        or (not used_gjk and mpr_state.portal_status[i_scratch] == PORTAL_STATUS.NONE)
    ):
        # Support points of the detection: the Minkowski portal refinement (MPR) portal (indices 1-3), or the face of
        # the expanding polytope algorithm (EPA) of Gilbert-Johnson-Keerthi (GJK) nearest to the origin. The analytic
        # capsule-capsule detection runs neither, a shallow GJK contact builds no polytope, and the degenerate paths of
        # MPR leave the portal unwritten.
        has_supports = True
        for i_v in qd.static(range(3)):
            support_a = qd.Vector.zero(gs.qd_float, 3)
            support_b = qd.Vector.zero(gs.qd_float, 3)
            if used_gjk:
                i_f = gjk_state.nearest_face[i_scratch]
                i_pv = gjk_state.polytope_faces.verts_idx[i_scratch, i_f][i_v]
                # Boxes, meshes and spheres provide native local supports. Other supports are stored in world space.
                geom_type_a = dyn_info.geoms.type[i_ga]
                geom_type_b = dyn_info.geoms.type[i_gb]
                if (
                    geom_type_a == gs.GEOM_TYPE.BOX
                    or geom_type_a == gs.GEOM_TYPE.MESH
                    or geom_type_a == gs.GEOM_TYPE.SPHERE
                ):
                    support_a = gu.qd_transform_by_trans_quat(
                        gjk_state.polytope_verts.local_obj1[i_scratch, i_pv], ga_pos_original, ga_quat_original
                    )
                else:
                    support_a = gjk_state.polytope_verts.obj1[i_scratch, i_pv]
                    support_a = R_inv @ (support_a - contact_pos_0) + contact_pos_0
                if (
                    geom_type_b == gs.GEOM_TYPE.BOX
                    or geom_type_b == gs.GEOM_TYPE.MESH
                    or geom_type_b == gs.GEOM_TYPE.SPHERE
                ):
                    support_b = gu.qd_transform_by_trans_quat(
                        gjk_state.polytope_verts.local_obj2[i_scratch, i_pv], gb_pos_original, gb_quat_original
                    )
                else:
                    support_b = gjk_state.polytope_verts.obj2[i_scratch, i_pv]
                    support_b = R @ (support_b - contact_pos_0) + contact_pos_0
            else:
                support_a = mpr_state.simplex_support.v1[i_v + 1, i_scratch]
                support_b = mpr_state.simplex_support.v2[i_v + 1, i_scratch]
                support_a = R_inv @ (support_a - contact_pos_0) + contact_pos_0
                support_b = R @ (support_b - contact_pos_0) + contact_pos_0
            supports_a[i_v, :] = support_a
            supports_b[i_v, :] = support_b

        # The support points of a geom that line up along the normal of contact 0 span a vertex or an edge of it, whose
        # two farthest points are the ends of the edge, rows 2 * i_g of 'ends' for geom i_g
        is_supports_degenerate = qd.Vector([0, 0], dt=gs.qd_int)
        is_supports_edge = qd.Vector([0, 0], dt=gs.qd_int)
        ends = qd.Matrix.zero(gs.qd_float, 4, 3)
        for i_g in qd.static(range(2)):
            supports_g = supports_a if qd.static(i_g == 0) else supports_b
            v_1, v_2, v_3 = supports_g[0, :], supports_g[1, :], supports_g[2, :]
            edge_1, edge_2 = v_2 - v_1, v_3 - v_1
            # The triangle spans no area when its height is within the rounding of the coordinates of its points, which
            # the un-rotation about contact 0 makes relative to their distance from the origin. Rounding alone can also
            # split one support vertex into two of its points, so the height is measured against its longest edge. Both
            # sides are compared squared, (a + b)^2 <= 2 (a^2 + b^2) widening the rounding bound by at most root two.
            edge_max_sqr = qd.max(edge_1.norm_sqr(), edge_2.norm_sqr(), (v_3 - v_2).norm_sqr())
            area_normal = edge_1.cross(edge_2).dot(normal_0)
            if area_normal**2 <= 2.0 * EPS**2 * edge_max_sqr * (edge_max_sqr + 4.0 * contact_pos_0.norm_sqr()):
                is_supports_degenerate[i_g] = 1
                # The ends of the edge are the two points farthest apart along the contact plane, where it lies
                span_12 = edge_1 - edge_1.dot(normal_0) * normal_0
                span_13 = edge_2 - edge_2.dot(normal_0) * normal_0
                span_23 = (v_3 - v_2) - (v_3 - v_2).dot(normal_0) * normal_0
                end_0, end_1, span = v_1, v_2, span_12
                if span_13.norm_sqr() > span.norm_sqr():
                    end_1, span = v_3, span_13
                if span_23.norm_sqr() > span.norm_sqr():
                    end_0, end_1, span = v_2, v_3, span_23
                if span.norm_sqr() > EPS * geom_pair_scale**2:
                    is_supports_edge[i_g] = 1
                    ends[2 * i_g, :] = end_0
                    ends[2 * i_g + 1, :] = end_1
        is_witness_b_anchor = is_supports_degenerate[1] == 1 and is_supports_degenerate[0] == 0

        if is_supports_edge[0] == 1 and is_supports_edge[1] == 1:
            # Two edges touch where they cross seen along the normal of contact 0, both closest points being material
            # points on one line along it, which makes the depth exact
            edge_a_0, edge_a_1, edge_b_0, edge_b_1 = ends[0, :], ends[1, :], ends[2, :], ends[3, :]
            proj_a_0 = edge_a_0 - edge_a_0.dot(normal_0) * normal_0
            proj_a_1 = edge_a_1 - edge_a_1.dot(normal_0) * normal_0
            proj_b_0 = edge_b_0 - edge_b_0.dot(normal_0) * normal_0
            proj_b_1 = edge_b_1 - edge_b_1.dot(normal_0) * normal_0
            len_sqr_a = (proj_a_1 - proj_a_0).norm_sqr()
            len_sqr_b = (proj_b_1 - proj_b_0).norm_sqr()
            if len_sqr_a > EPS * geom_pair_scale**2 and len_sqr_b > EPS * geom_pair_scale**2:
                # The parallel test of the helper compares the squared sine of the angle between both edges, times the
                # product of their squared lengths, to its tolerance
                closest_a, closest_b = func_closest_points_on_segments(
                    proj_a_0, proj_a_1, proj_b_0, proj_b_1, EPS * len_sqr_a * len_sqr_b
                )
                if (closest_b - closest_a).norm_sqr() <= EPS * geom_pair_scale**2:
                    ratio_a = (closest_a - proj_a_0).dot(proj_a_1 - proj_a_0) / len_sqr_a
                    ratio_b = (closest_b - proj_b_0).dot(proj_b_1 - proj_b_0) / len_sqr_b
                    witness_a = edge_a_0 + ratio_a * (edge_a_1 - edge_a_0)
                    witness_b = edge_b_0 + ratio_b * (edge_b_1 - edge_b_0)
                    depth = normal_0.dot(witness_b - witness_a)
                    contact_pos = 0.5 * (witness_a + witness_b)
                    has_depth_estimate = True
        else:
            # Along an edge lying on the face of the other geom, the depth varies linearly, so that the point the
            # detection lands on along it is arbitrary up to rounding when the tilt hardly deepens one end more than
            # the other. The perturbation looks for the end of the patch that its tilt deepens, which is one end of the
            # edge, an exact material point. Each geom turns about contact 0, geom A by 'qrot' towards the negative
            # side of the normal and geom B by its inverse towards the positive one.
            ends_deepened = qd.Matrix.zero(gs.qd_float, 2, 3)
            has_end_deepened = qd.Vector.zero(gs.qd_int, 2)
            for i_g in qd.static(range(2)):
                if is_supports_edge[i_g] == 1:
                    end_0, end_1 = ends[2 * i_g, :], ends[2 * i_g + 1, :]
                    R_g = R if qd.static(i_g == 0) else R_inv
                    lift_0 = normal_0.dot(R_g @ (end_0 - contact_pos_0) - (end_0 - contact_pos_0))
                    lift_1 = normal_0.dot(R_g @ (end_1 - contact_pos_0) - (end_1 - contact_pos_0))
                    if qd.static(i_g == 0):
                        lift_0, lift_1 = -lift_0, -lift_1
                    if qd.abs(lift_1 - lift_0) > EPS * (qd.abs(lift_0) + qd.abs(lift_1)):
                        end = end_1 if lift_1 > lift_0 else end_0
                        has_end_deepened[i_g] = 1
                        ends_deepened[i_g, :] = end

            # The support triangle of each geom, where it covers the witness of the other geom along the normal of
            # contact 0, bounds its surface there, which gives a lower bound of the depth. The deepened end of an edge
            # replaces the witness of its geom only where covered, since the edge may run past the other geom.
            for i_g in qd.static(range(2)):
                supports_other = supports_b if qd.static(i_g == 0) else supports_a
                v_1, v_2, v_3 = supports_other[0, :], supports_other[1, :], supports_other[2, :]
                edge_1, edge_2 = v_2 - v_1, v_3 - v_1
                area = edge_1.cross(edge_2).dot(normal_0)
                is_covered = False
                for i_w in qd.static(range(2)):
                    # The deepened end first, then the witness of the detection
                    witness = witness_a if qd.static(i_g == 0) else witness_b
                    is_candidate = not is_covered
                    if qd.static(i_w == 0):
                        witness = ends_deepened[i_g, :]
                        is_candidate = has_end_deepened[i_g] == 1
                    if is_candidate:
                        if is_supports_degenerate[1 - i_g] == 0:
                            offset = witness - v_1
                            lambda_2 = offset.cross(edge_2).dot(normal_0) / area
                            lambda_3 = edge_1.cross(offset).dot(normal_0) / area
                            if lambda_2 >= 0.0 and lambda_3 >= 0.0 and lambda_2 + lambda_3 <= 1.0:
                                is_covered = True
                                if qd.static(i_g == 0):
                                    witness_a = witness
                                else:
                                    witness_b = witness
                                height = normal_0.dot(v_1 + lambda_2 * edge_1 + lambda_3 * edge_2)
                                bound = height - normal_0.dot(witness_a)
                                if qd.static(i_g == 1):
                                    bound = normal_0.dot(witness_b) - height
                                if not has_depth_estimate or bound > depth:
                                    depth = bound
                                    contact_pos = witness + (0.5 if qd.static(i_g == 0) else -0.5) * bound * normal_0
                                    has_depth_estimate = True

    # Available support triangles define the unperturbed normal below. Un-rotating the detection normal instead mixes
    # lateral witness separation into depth when the detection combines different support features.
    if not has_depth_estimate and not has_supports:
        # The un-rotated perturbed normal is the normal of the tangent plane of the other geom at its witness, against
        # which the anchor witness measures its depth. A plane tilted beyond the relative rotation of both geoms belongs
        # to another face, along which this depth is ill-conditioned, so that the separation of both witnesses stays.
        tangent_normal = R @ normal
        if is_witness_b_anchor:
            tangent_normal = R_inv @ normal
        tangent_slope = tangent_normal.dot(normal_0)
        if tangent_slope >= qd.cos(2.0 * collider_info.mc_perturbation[None]):
            depth = tangent_normal.dot(witness_b - witness_a) / tangent_slope
            contact_pos = witness_a + 0.5 * depth * normal_0
            if is_witness_b_anchor:
                contact_pos = witness_b - 0.5 * depth * normal_0
            has_depth_estimate = True

    # When the projected support triangles miss the witnesses, their Minkowski normal measures the separation on the
    # unperturbed geoms. The detection normal stands in for a support triangle too thin to be oriented.
    contact_normal = normal_0
    if not has_depth_estimate:
        contact_normal = normal
        if has_supports:
            minkowski = supports_a - supports_b
            minkowski_1, minkowski_2, minkowski_3 = minkowski[0, :], minkowski[1, :], minkowski[2, :]
            edge_1, edge_2 = minkowski_2 - minkowski_1, minkowski_3 - minkowski_1
            face_normal = edge_1.cross(edge_2)
            # See the support triangles above for the degeneracy bound
            edge_max_sqr = qd.max(edge_1.norm_sqr(), edge_2.norm_sqr(), (minkowski_3 - minkowski_2).norm_sqr())
            if face_normal.norm_sqr() > 2.0 * EPS**2 * edge_max_sqr * (edge_max_sqr + 4.0 * contact_pos_0.norm_sqr()):
                contact_normal = face_normal.normalized()
                if contact_normal.dot(normal) < 0.0:
                    contact_normal = -contact_normal
        depth = contact_normal.dot(witness_b - witness_a)
        contact_pos = 0.5 * (witness_a + witness_b)

    # A contact of negative depth lies past the edge of the patch, and moves towards contact 0 to where the depth
    # interpolated between both vanishes, a point of the patch since the depth across the contact plane is concave. The
    # interpolation divides by their difference of depth, which must exceed its own error: the rounding of both depths
    # and the stopping tolerance of the detection of contact 0.
    depth_noise = EPS * geom_pair_scale
    if used_gjk_0:
        if not func_is_discrete_geoms(i_ga, i_gb, dyn_info):
            depth_noise = qd.max(depth_noise, collider_info.gjk.tolerance[None])
    else:
        depth_noise = qd.max(depth_noise, collider_info.mpr.CCD_TOLERANCE[None] * geom_pair_scale)
    if has_depth_estimate and depth < 0.0 and penetration_0 - depth > depth_noise:
        contact_pos += depth / (depth - penetration_0) * (contact_pos_0 - contact_pos)
    depth = qd.max(depth, 0.0)

    # The smooth-primitive position reconstruction runs on the unperturbed pose, the canonical state the solver stores
    contact_pos = func_apply_smooth_refinement(
        i_ga,
        i_gb,
        contact_normal,
        depth,
        contact_pos,
        ga_pos_original,
        ga_quat_original,
        gb_pos_original,
        gb_quat_original,
        dyn_info,
        rigid_config,
    )
    return contact_normal, depth, contact_pos


@qd.func
def func_prefer_gjk_refinement(
    i_ga: int,
    i_gb: int,
    i_pair: int,
    i_b: int,
    i_scratch: int,
    portal_status: int,
    penetration: float,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geom_pair_scale: float,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """Whether GJK should refine the contact MPR resolved, on an unconverged portal or a cold non-minimal one.

    A depth MPR could not converge on always refines. A cold contact is also checked against the overlap of both geoms
    along the frame axes of either: a depth exceeding it along one of them proves the portal non-minimal and refines. MPR
    searches along the line joining the geom centers, which runs sideways between flat geoms stacked near their edges,
    so a shallow depth may still come with a wrong normal. Every other contact keeps the portal of MPR, whose depth may
    exceed the minimal one by a margin that leaves the dynamics plausible.
    """
    cached_penetration = collider_state.contact_cache.penetration[i_pair, i_b]
    is_gjk_preferred = portal_status == PORTAL_STATUS.UNCONVERGED
    # The support queries are too costly for every contact, so only a cold one is checked
    if not is_gjk_preferred and cached_penetration <= 0.0 and 0.0 < penetration:
        EPS = rigid_info.EPS[None]
        R_a = gu.qd_quat_to_R(ga_quat, EPS)
        R_b = gu.qd_quat_to_R(gb_quat, EPS)
        # Both senses of the frame axes of either geom, one per row
        directions = qd.Matrix.zero(gs.qd_float, 12, 3)
        for i_dir in qd.static(range(12)):
            for k in qd.static(range(3)):
                if qd.static(i_dir < 6):
                    directions[i_dir, k] = R_a[k, i_dir // 2]
                else:
                    directions[i_dir, k] = R_b[k, i_dir // 2 - 3]
                if qd.static(i_dir % 2 == 1):
                    directions[i_dir, k] = -directions[i_dir, k]
        # The MPR convergence tolerance absorbs the depth error of a portal that is the minimal one
        depth_max = penetration - collider_info.mpr.CCD_TOLERANCE[None] * geom_pair_scale
        # Any known point of the Minkowski difference bounds from below its depth along every direction, so a direction
        # is only queried while no known point rules it out. The portal vertices come first when MPR built a portal,
        # its degenerate paths leaving them unwritten, then every support found.
        depths_lower = qd.Vector([-1e20] * 12, dt=gs.qd_float)
        is_portal_non_minimal = False
        for i_point in range(15):
            has_point = False
            point = qd.Vector.zero(gs.qd_float, 3)
            if i_point < 3:
                if portal_status != PORTAL_STATUS.NONE:
                    point = mpr_state.simplex_support.v[i_point + 1, i_scratch]
                    has_point = True
            elif not is_portal_non_minimal:
                i_dir = i_point - 3
                direction = qd.Vector([directions[i_dir, 0], directions[i_dir, 1], directions[i_dir, 2]])
                if depths_lower[i_dir] < depth_max:
                    point, _, _ = mpr.compute_support(
                        i_ga,
                        i_gb,
                        i_b,
                        direction,
                        ga_pos,
                        ga_quat,
                        gb_pos,
                        gb_quat,
                        collider_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                    is_portal_non_minimal = direction.dot(point) < depth_max
                    has_point = True
            if has_point:
                depths_lower = qd.max(depths_lower, directions @ point)
        is_gjk_preferred = is_portal_non_minimal
    return is_gjk_preferred and 0.0 < penetration


@qd.func
def func_contact_overlap_axes(
    i_ga: int,
    i_gb: int,
    i_b: int,
    normal: qd.types.vector(3),
    contact_pos: qd.types.vector(3),
    axis_0: qd.types.vector(3),
    axis_1: qd.types.vector(3),
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    axes: qd.types.matrix(4, 3),
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    geom_pair_scale: float,
):
    """Aim the four perturbed detections of a contact at the corners of the overlap of both geoms around it.

    The face of each geom touching the other one is sampled along the contact normal 'normal' tilted toward directions
    in the plane spanned by 'axis_0' and 'axis_1', with a fixed limit on support queries. The overlap intersects the
    convex hulls of these samples where the depth between their fitted planes is non-negative, in the plane coordinates
    around contact 0 'contact_pos'.

    A perturbed detection returns the point of the overlap that its tilt deepens the most, which is the corner whose
    normal cone (the in-plane directions along which it lies farthest out) contains the deepening direction. Each
    detection deepens its corner along the direction of the normal cone that leaves the depth gradient farthest inside
    the cone, so that it can neither tie with nor return a neighbouring corner. An overlap of more than four corners
    targets the four that carry the most area together with the corner of contact 0, which needs no detection, so that
    the contacts of the pair are as many corners of the patch.

    Returns the unit rotation axes and the merge distance. Targets within that distance of contact 0 or an earlier
    target receive a zero axis. Degenerate overlaps retain the default axes and the geom-scale merge distance.
    """
    EPS = rigid_info.EPS[None]
    # Support points that coincide up to the rounding of their world coordinates and of contact 0 they are taken from,
    # which grows with their distance to the origin, are one point. Distinct points lie a fraction of the geometry
    # apart, far beyond the square root of the machine precision.
    coincidence_tolerance = qd.sqrt(EPS) * geom_pair_scale + 2.0 * EPS * contact_pos.norm()

    # Two opposite supports seed a convex outline. Each query adds its support to the hull, retaining at most eight
    # vertices in contact-plane coordinates plus height along the normal. Geom A lies on the positive side of the
    # normal, so its face looks backwards. Sorted geom types put a plane at A and a terrain at B.
    # TODO: Ten vertices per face and 40 support queries raise the area spanned by the contacts on a 32-sided face from
    # at worst 67% to at least 75% of the largest five-corner patch, for 0.1-0.4% more step time on average. The worst
    # case grows faster, the search of the target corners being quadratic in the 2n+1 overlap corners of n vertices.
    polygons = qd.Matrix.zero(gs.qd_float, 16, 3)
    polygons_size = qd.Vector.zero(gs.qd_int, 2)
    # A sphere, an ellipsoid or a capsule has no planar face of positive area, so their overlap with any face is empty.
    # Against a plane, the overlap is the whole face of the other geom, a corner of which every tilt of the default
    # perturbations reaches, so the pair keeps them.
    is_outlined = dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.PLANE and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.TERRAIN
    for i_g in qd.static(range(2)):
        geom_type = dyn_info.geoms.type[i_ga if i_g == 0 else i_gb]
        if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.ELLIPSOID or geom_type == gs.GEOM_TYPE.CAPSULE:
            is_outlined = False
    if is_outlined:
        # Prioritize the edges longest relative to their distance to contact 0 (squared length over squared distance),
        # so the vertex budget resolves the portion around the contact first without spending itself on short edges
        # next to it while a long edge farther away hides a corner of the overlap past it. Surviving edges retain their
        # settled bits when an insertion removes the visible chain.
        edges_settled = qd.Vector.zero(gs.qd_int, 2)
        i_g_refine = 0
        i_e_refine = 0
        for i_query in range(4 + 2 * (6 + 8)):
            i_g = i_query // 2
            i_e = 0
            direction_x = 1.0 if i_query % 2 == 0 else -1.0
            direction_y = gs.qd_float(0.0)
            is_query = i_query < 4
            if not is_query:
                for _ in range(2):
                    if i_g_refine < 2:
                        n_points = polygons_size[i_g_refine]
                        if n_points < 2 or n_points >= 8 or edges_settled[i_g_refine] == (1 << n_points) - 1:
                            i_g_refine += 1
                            i_e_refine = 0
                if i_g_refine < 2:
                    i_e_refine = 0
                    span_max = gs.qd_float(-1.0)
                    for j_e in range(polygons_size[i_g_refine]):
                        if edges_settled[i_g_refine] & (1 << j_e) == 0:
                            j_k0 = 8 * i_g_refine + j_e
                            j_k1 = 8 * i_g_refine + (j_e + 1) % polygons_size[i_g_refine]
                            edge_x = polygons[j_k1, 0] - polygons[j_k0, 0]
                            edge_y = polygons[j_k1, 1] - polygons[j_k0, 1]
                            edge_length_sqr = edge_x**2 + edge_y**2
                            edge_fraction = -(polygons[j_k0, 0] * edge_x + polygons[j_k0, 1] * edge_y)
                            edge_fraction = qd.min(qd.max(edge_fraction / qd.max(edge_length_sqr, EPS**2), 0.0), 1.0)
                            distance = (polygons[j_k0, 0] + edge_fraction * edge_x) ** 2
                            distance += (polygons[j_k0, 1] + edge_fraction * edge_y) ** 2
                            span = edge_length_sqr / qd.max(distance, EPS**2)
                            if span > span_max:
                                span_max = span
                                i_e_refine = j_e
                    # Both directed seed edges span the same segment. Contact 0 selects the side to refine first
                    if polygons_size[i_g_refine] == 2 and edges_settled[i_g_refine] == 0:
                        i_k0 = 8 * i_g_refine
                        edge_x = polygons[i_k0 + 1, 0] - polygons[i_k0, 0]
                        edge_y = polygons[i_k0 + 1, 1] - polygons[i_k0, 1]
                        side = edge_y * polygons[i_k0, 0] - edge_x * polygons[i_k0, 1]
                        i_e_refine = 0 if side <= 0.0 else 1
                    i_g, i_e = i_g_refine, i_e_refine
                    n_points = polygons_size[i_g]
                    i_k0 = 8 * i_g + i_e
                    i_k1 = 8 * i_g + (i_e + 1) % n_points
                    edge_x = polygons[i_k1, 0] - polygons[i_k0, 0]
                    edge_y = polygons[i_k1, 1] - polygons[i_k0, 1]
                    edge_norm = qd.sqrt(edge_x**2 + edge_y**2)
                    direction_x, direction_y = edge_y / edge_norm, -edge_x / edge_norm
                    is_query = True
            if is_query:
                is_b = i_g == 1
                # The relative perturbation sets the sampling angle so curved surfaces are sampled near the contact.
                direction = 2.0 * collider_info.mc_perturbation[None] * (
                    direction_x * axis_0 + direction_y * axis_1
                ) + (normal if is_b else -normal)
                support_offset = (
                    mpr.support_driver(
                        i_gb if is_b else i_ga,
                        i_b,
                        direction.normalized(EPS),
                        gb_pos if is_b else ga_pos,
                        gb_quat if is_b else ga_quat,
                        collider_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                    )
                    - contact_pos
                )
                corner = qd.Vector([support_offset.dot(axis_0), support_offset.dot(axis_1), support_offset.dot(normal)])
                n_points = polygons_size[i_g]
                if i_query < 4:
                    # Coincident opposite supports leave a single projected point.
                    is_distinct = n_points == 0
                    if n_points > 0:
                        i_k_last = 8 * i_g + n_points - 1
                        gap_x, gap_y = corner[0] - polygons[i_k_last, 0], corner[1] - polygons[i_k_last, 1]
                        is_distinct = gap_x**2 + gap_y**2 > coincidence_tolerance**2
                        if i_query % 2 == 1 and n_points > 1:
                            gap_x, gap_y = corner[0] - polygons[8 * i_g, 0], corner[1] - polygons[8 * i_g, 1]
                            is_distinct = is_distinct and gap_x**2 + gap_y**2 > coincidence_tolerance**2
                    if is_distinct:
                        for i_3 in qd.static(range(3)):
                            polygons[8 * i_g + n_points, i_3] = corner[i_3]
                        polygons_size[i_g] = n_points + 1
                else:
                    i_k0 = 8 * i_g + i_e
                    i_k1 = 8 * i_g + (i_e + 1) % n_points
                    edge_x, edge_y = polygons[i_k1, 0] - polygons[i_k0, 0], polygons[i_k1, 1] - polygons[i_k0, 1]
                    side = edge_x * (corner[1] - polygons[i_k0, 1]) - edge_y * (corner[0] - polygons[i_k0, 0])
                    if side < -coincidence_tolerance * qd.sqrt(edge_x**2 + edge_y**2):
                        is_local = True
                        for j_e_ in range(2):
                            j_e = (i_e + n_points - 1 + 2 * j_e_) % n_points
                            j_k0 = 8 * i_g + j_e
                            j_k1 = 8 * i_g + (j_e + 1) % n_points
                            neighbor_x = polygons[j_k1, 0] - polygons[j_k0, 0]
                            neighbor_y = polygons[j_k1, 1] - polygons[j_k0, 1]
                            neighbor_side = neighbor_x * (corner[1] - polygons[j_k0, 1])
                            neighbor_side -= neighbor_y * (corner[0] - polygons[j_k0, 0])
                            if neighbor_side < 0.0:
                                is_local = False
                        if is_local:
                            for i_v_ in range(n_points - i_e - 1):
                                i_v = n_points - i_v_
                                for i_3 in qd.static(range(3)):
                                    polygons[8 * i_g + i_v, i_3] = polygons[8 * i_g + i_v - 1, i_3]
                            for i_3 in qd.static(range(3)):
                                polygons[8 * i_g + i_e + 1, i_3] = corner[i_3]
                            polygons_size[i_g] = n_points + 1
                            edges_lower = edges_settled[i_g] & ((1 << i_e) - 1)
                            edges_upper = (edges_settled[i_g] >> (i_e + 1)) << (i_e + 2)
                            edges_settled[i_g] = edges_lower | edges_upper
                        else:
                            edges_visible = gs.qd_int(0)
                            for j_e in range(n_points):
                                j_k0 = 8 * i_g + j_e
                                j_k1 = 8 * i_g + (j_e + 1) % n_points
                                edge_x = polygons[j_k1, 0] - polygons[j_k0, 0]
                                edge_y = polygons[j_k1, 1] - polygons[j_k0, 1]
                                side = edge_x * (corner[1] - polygons[j_k0, 1])
                                side -= edge_y * (corner[0] - polygons[j_k0, 0])
                                if side < 0.0:
                                    edges_visible |= 1 << j_e
                            i_v_start = 0
                            i_v_end = 0
                            n_visible_runs = 0
                            for j_e in range(n_points):
                                is_visible = edges_visible & (1 << j_e) != 0
                                is_prev_visible = edges_visible & (1 << ((j_e + n_points - 1) % n_points)) != 0
                                if is_prev_visible and not is_visible:
                                    i_v_start = j_e
                                if is_visible and not is_prev_visible:
                                    i_v_end = j_e
                                    n_visible_runs += 1
                            if n_visible_runs == 1:
                                n_kept = (i_v_end + n_points - i_v_start) % n_points + 1
                                edges_settled_next = (edges_settled[i_g] >> i_v_start) | (
                                    edges_settled[i_g] << (n_points - i_v_start)
                                )
                                edges_settled_next &= (1 << (n_kept - 1)) - 1
                                if i_v_start > 0:
                                    # The kept chain starts at vertex 0 after a rotation by i_v_start, made of as many
                                    # cycles as the greatest common divisor of both counts. Euclid's algorithm takes at
                                    # most 5 steps for counts up to 8.
                                    n_cycles = n_points
                                    n_remainder = i_v_start
                                    for _ in range(5):
                                        if n_remainder > 0:
                                            n_next = n_cycles % n_remainder
                                            n_cycles = n_remainder
                                            n_remainder = n_next
                                    for j_v in range(n_cycles):
                                        i_k = 8 * i_g + j_v
                                        corner_cycle = qd.Vector([polygons[i_k, 0], polygons[i_k, 1], polygons[i_k, 2]])
                                        i_v = j_v
                                        for _ in range(n_points // n_cycles - 1):
                                            i_v_next = (i_v + i_v_start) % n_points
                                            for i_3 in qd.static(range(3)):
                                                polygons[8 * i_g + i_v, i_3] = polygons[8 * i_g + i_v_next, i_3]
                                            i_v = i_v_next
                                        for i_3 in qd.static(range(3)):
                                            polygons[8 * i_g + i_v, i_3] = corner_cycle[i_3]
                                for i_3 in qd.static(range(3)):
                                    polygons[8 * i_g + n_kept, i_3] = corner[i_3]
                                polygons_size[i_g] = n_kept + 1
                                edges_settled[i_g] = edges_settled_next
                            else:
                                # Roundoff can split nearly collinear visible edges into several chains. Rebuild the
                                # hull to restore convex vertex order before clipping.
                                for i_3 in qd.static(range(3)):
                                    polygons[8 * i_g + n_points, i_3] = corner[i_3]
                                n_points += 1
                                i_k_start = 0
                                for j_k_ in range(1, n_points):
                                    i_k = 8 * i_g + i_k_start
                                    j_k = 8 * i_g + j_k_
                                    is_before = polygons[j_k, 0] < polygons[i_k, 0]
                                    is_before = is_before or (
                                        polygons[j_k, 0] <= polygons[i_k, 0] and polygons[j_k, 1] < polygons[i_k, 1]
                                    )
                                    if is_before:
                                        i_k_start = j_k_
                                for i_3 in qd.static(range(3)):
                                    coordinate = polygons[8 * i_g, i_3]
                                    polygons[8 * i_g, i_3] = polygons[8 * i_g + i_k_start, i_3]
                                    polygons[8 * i_g + i_k_start, i_3] = coordinate
                                n_hull = 0
                                is_closed = False
                                for i_v in range(n_points):
                                    if not is_closed:
                                        n_hull = i_v + 1
                                        i_k0 = 8 * i_g + i_v
                                        i_k_next = 1 if i_v == 0 and n_points > 1 else 0
                                        for j_k_ in range(i_v + 1, n_points):
                                            i_k1 = 8 * i_g + i_k_next
                                            j_k = 8 * i_g + j_k_
                                            edge_x = polygons[i_k1, 0] - polygons[i_k0, 0]
                                            edge_y = polygons[i_k1, 1] - polygons[i_k0, 1]
                                            delta_x = polygons[j_k, 0] - polygons[i_k0, 0]
                                            delta_y = polygons[j_k, 1] - polygons[i_k0, 1]
                                            side = edge_x * delta_y - edge_y * delta_x
                                            distance = delta_x**2 + delta_y**2
                                            edge_length_sqr = edge_x**2 + edge_y**2
                                            side_tolerance = EPS * qd.sqrt(edge_length_sqr * distance)
                                            is_farther = qd.abs(side) <= side_tolerance and distance > edge_length_sqr
                                            if side < -side_tolerance or is_farther:
                                                i_k_next = j_k_
                                        is_closed = i_k_next == 0
                                        if not is_closed:
                                            for i_3 in qd.static(range(3)):
                                                coordinate = polygons[8 * i_g + n_hull, i_3]
                                                polygons[8 * i_g + n_hull, i_3] = polygons[8 * i_g + i_k_next, i_3]
                                                polygons[8 * i_g + i_k_next, i_3] = coordinate
                                polygons_size[i_g] = n_hull
                                edges_settled[i_g] = 0
                    else:
                        edges_settled[i_g] |= 1 << i_e

    # Two eight-vertex hulls and the depth half-plane bound at most seventeen overlap vertices
    overlap = qd.Matrix.zero(gs.qd_float, 17, 2)
    n_overlap = 0
    slope = qd.Vector.zero(gs.qd_float, 2)
    if is_outlined:
        # Depth in the contact plane, the height of the face of geom B minus that of geom A, from the plane of each
        # polygon: the normal of its points (Newell's method) through their centroid. Contact 0 lies on both planes of
        # flat faces, while the outline of a rounded geom samples it far above the contact, where its depth leaves no
        # part of the overlap pressed.
        depth_origin = gs.qd_float(0.0)
        for i_g in qd.static(range(2)):
            plane_normal = qd.Vector.zero(gs.qd_float, 3)
            centroid = qd.Vector.zero(gs.qd_float, 3)
            n_points = polygons_size[i_g]
            for i_k in range(n_points):
                i_k0 = 8 * i_g + i_k
                i_k1 = 8 * i_g + (i_k + 1) % n_points
                plane_normal[0] += (polygons[i_k0, 1] - polygons[i_k1, 1]) * (polygons[i_k0, 2] + polygons[i_k1, 2])
                plane_normal[1] += (polygons[i_k0, 2] - polygons[i_k1, 2]) * (polygons[i_k0, 0] + polygons[i_k1, 0])
                plane_normal[2] += (polygons[i_k0, 0] - polygons[i_k1, 0]) * (polygons[i_k0, 1] + polygons[i_k1, 1])
                for i_3 in qd.static(range(3)):
                    centroid[i_3] += polygons[i_k0, i_3] / n_points
            gradient = qd.Vector.zero(gs.qd_float, 2)
            if qd.abs(plane_normal[2]) > EPS * plane_normal.norm():
                gradient = -qd.Vector([plane_normal[0], plane_normal[1]]) / plane_normal[2]
            height_origin = centroid[2] - gradient[0] * centroid[0] - gradient[1] * centroid[1]
            if qd.static(i_g == 1):
                slope += gradient
                depth_origin += height_origin
            else:
                slope -= gradient
                depth_origin -= height_origin

        # Clip the polygon of geom A by that of geom B (Sutherland-Hodgman), which keeps the orientation of the former,
        # then by the half-plane where the faces interpenetrate, the patch being the part of their overlap that the
        # depth bounds. A polygon of fewer than three points bounds no patch.
        n_overlap = polygons_size[0]
        for i_k in range(n_overlap):
            overlap[i_k, 0] = polygons[i_k, 0]
            overlap[i_k, 1] = polygons[i_k, 1]
        n_clip = polygons_size[1]
        if n_clip < 3:
            n_overlap = 0
        # The overlap lies within the subject polygon, which clipping only cuts, and within the convex hull of the clip
        # polygon, which holds the intersection of the half-planes of its edges. It is thus empty when the depth, affine
        # in the plane, is negative at every corner of either polygon, which spares the clipping of the faces apart from
        # each other.
        is_pressed_subject = False
        for i_k in range(n_overlap):
            if depth_origin + slope[0] * overlap[i_k, 0] + slope[1] * overlap[i_k, 1] >= 0.0:
                is_pressed_subject = True
        is_pressed_clip = False
        for i_e in range(n_clip):
            if depth_origin + slope[0] * polygons[8 + i_e, 0] + slope[1] * polygons[8 + i_e, 1] >= 0.0:
                is_pressed_clip = True
        if not (is_pressed_subject and is_pressed_clip):
            n_overlap = 0
            n_clip = 0
        clip_area = gs.qd_float(0.0)
        for i_e in range(n_clip):
            i_e_next = (i_e + 1) % n_clip
            clip_area += (
                polygons[8 + i_e, 0] * polygons[8 + i_e_next, 1] - polygons[8 + i_e, 1] * polygons[8 + i_e_next, 0]
            )
        orientation = 1.0 if clip_area >= 0.0 else -1.0
        for i_e in range(n_clip + 1):
            # Each clip keeps the side where the affine function 'offset + inward_normal . p' is non-negative
            inward_normal = slope
            offset = depth_origin
            if i_e < n_clip:
                corner_x, corner_y = polygons[8 + i_e, 0], polygons[8 + i_e, 1]
                edge_x = polygons[8 + (i_e + 1) % n_clip, 0] - corner_x
                edge_y = polygons[8 + (i_e + 1) % n_clip, 1] - corner_y
                inward_normal = orientation * qd.Vector([-edge_y, edge_x])
                offset = -(inward_normal[0] * corner_x + inward_normal[1] * corner_y)
            overlap_out = qd.Matrix.zero(gs.qd_float, 17, 2)
            n_out = 0
            for i_k in range(n_overlap):
                p_x, p_y = overlap[i_k, 0], overlap[i_k, 1]
                q_x, q_y = overlap[(i_k + 1) % n_overlap, 0], overlap[(i_k + 1) % n_overlap, 1]
                side_p = offset + inward_normal[0] * p_x + inward_normal[1] * p_y
                side_q = offset + inward_normal[0] * q_x + inward_normal[1] * q_y
                if side_p >= 0.0 and n_out < 17:
                    overlap_out[n_out, 0], overlap_out[n_out, 1] = p_x, p_y
                    n_out += 1
                if side_p * side_q < 0.0 and n_out < 17:
                    crossing_fraction = side_p / (side_p - side_q)
                    overlap_out[n_out, 0] = p_x + crossing_fraction * (q_x - p_x)
                    overlap_out[n_out, 1] = p_y + crossing_fraction * (q_y - p_y)
                    n_out += 1
            overlap = overlap_out
            n_overlap = n_out

        # Where an edge of one polygon crosses a corner of the other, clipping places a point of each polygon at that
        # corner, and both points are merged into one corner
        overlap_distinct = qd.Matrix.zero(gs.qd_float, 17, 2)
        n_distinct = 0
        for i_k in range(n_overlap):
            is_distinct = n_distinct == 0
            if n_distinct > 0:
                gap = qd.Vector(
                    [
                        overlap[i_k, 0] - overlap_distinct[n_distinct - 1, 0],
                        overlap[i_k, 1] - overlap_distinct[n_distinct - 1, 1],
                    ]
                )
                is_distinct = gap.norm_sqr() > coincidence_tolerance**2
            if is_distinct:
                overlap_distinct[n_distinct, 0], overlap_distinct[n_distinct, 1] = overlap[i_k, 0], overlap[i_k, 1]
                n_distinct += 1
        if n_distinct > 1:
            gap = qd.Vector(
                [
                    overlap_distinct[n_distinct - 1, 0] - overlap_distinct[0, 0],
                    overlap_distinct[n_distinct - 1, 1] - overlap_distinct[0, 1],
                ]
            )
            if gap.norm_sqr() <= coincidence_tolerance**2:
                n_distinct -= 1
        overlap = overlap_distinct
        n_overlap = n_distinct

    # An overlap spanning no area beyond the rounding of its corners, as that of an edge lying on a face, has no corner
    # to aim at, and its clipping keeps a number of corners that rounding decides
    overlap_area = gs.qd_float(0.0)
    for i_k in range(n_overlap):
        i_k_next = (i_k + 1) % n_overlap
        overlap_area += 0.5 * (overlap[i_k, 0] * overlap[i_k_next, 1] - overlap[i_k, 1] * overlap[i_k_next, 0])
    overlap_perimeter = gs.qd_float(0.0)
    for i_k in range(n_overlap):
        i_next = (i_k + 1) % n_overlap
        edge = qd.Vector([overlap[i_next, 0] - overlap[i_k, 0], overlap[i_next, 1] - overlap[i_k, 1]])
        overlap_perimeter += edge.norm()
    is_overlap_degenerate = qd.abs(overlap_area) <= EPS * geom_pair_scale * overlap_perimeter

    # The normal of each edge of the overlap, pointing outwards
    orientation = 1.0 if overlap_area >= 0.0 else -1.0
    edges_normal = qd.Matrix.zero(gs.qd_float, 17, 2)
    for i_k in range(n_overlap):
        i_k_next = (i_k + 1) % n_overlap
        edges_normal[i_k, 0] = orientation * (overlap[i_k_next, 1] - overlap[i_k, 1])
        edges_normal[i_k, 1] = orientation * (overlap[i_k, 0] - overlap[i_k_next, 0])

    targets = qd.Vector([0, 1, 2, 3], dt=gs.qd_int)
    if n_overlap > 4:
        i_k_origin = -1
        for i_k in range(n_overlap):
            if overlap[i_k, 0] ** 2 + overlap[i_k, 1] ** 2 <= coincidence_tolerance**2:
                i_k_origin = i_k
        # A diagonal splits a convex quadrilateral into independently maximal triangles. With contact 0 fixed,
        # rotate the corner order past it and split the pentagon into three triangles, two independently maximal.
        n_search = n_overlap - (1 if i_k_origin >= 0 else 0)
        i_k_start = i_k_origin + 1 if i_k_origin >= 0 else 0
        area_best = gs.qd_float(-1.0)
        for i_k0_ in range(n_search - 3):
            i_k0 = (i_k_start + i_k0_) % n_overlap
            corner_0 = qd.Vector([overlap[i_k0, 0], overlap[i_k0, 1]])
            corner_anchor = corner_0
            if i_k_origin >= 0:
                corner_anchor = qd.Vector([overlap[i_k_origin, 0], overlap[i_k_origin, 1]])
            # Convexity makes each maximizing endpoint advance as the diagonal rotates. Each endpoint traverses
            # its corner interval once for a fixed first corner, bounding the total search quadratically.
            i_k1_best_ = i_k0_ + 1
            i_k3_best_ = i_k0_ + 3
            for i_k2_ in range(i_k0_ + 2, n_search - 1):
                i_k2 = (i_k_start + i_k2_) % n_overlap
                corner_2 = qd.Vector([overlap[i_k2, 0], overlap[i_k2, 1]])
                area_origin = gs.qd_float(0.0)
                if i_k_origin >= 0:
                    edge_0, edge_2 = corner_0 - corner_anchor, corner_2 - corner_anchor
                    area_origin = orientation * (edge_0[0] * edge_2[1] - edge_0[1] * edge_2[0])
                i_k3_best_ = qd.max(i_k3_best_, i_k2_ + 1)
                area_mid_best = gs.qd_float(0.0)
                area_last_best = gs.qd_float(0.0)
                for i_side in range(2):
                    i_k_best_ = i_k1_best_
                    i_k_end_ = i_k2_
                    if i_side == 1:
                        i_k_best_ = i_k3_best_
                        i_k_end_ = n_search
                    area_triangle_best = gs.qd_float(-1.0)
                    for i_k_ in range(i_k_best_, i_k_end_):
                        i_k = (i_k_start + i_k_) % n_overlap
                        corner = qd.Vector([overlap[i_k, 0], overlap[i_k, 1]])
                        edge_1, edge_2 = corner - corner_0, corner_2 - corner_0
                        if i_side == 1:
                            edge_1, edge_2 = corner_2 - corner_anchor, corner - corner_anchor
                        area = orientation * (edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0])
                        if area <= area_triangle_best:
                            break
                        area_triangle_best = area
                        i_k_best_ = i_k_
                    if i_side == 0:
                        i_k1_best_ = i_k_best_
                        area_mid_best = area_triangle_best
                    else:
                        i_k3_best_ = i_k_best_
                        area_last_best = area_triangle_best
                area = area_origin + area_mid_best + area_last_best
                if area > area_best:
                    area_best = area
                    i_k1_best = (i_k_start + i_k1_best_) % n_overlap
                    i_k3_best = (i_k_start + i_k3_best_) % n_overlap
                    targets = qd.Vector([i_k0, i_k1_best, i_k2, i_k3_best], dt=gs.qd_int)

    # Area divided by perimeter scales with the narrow width of a patch, protecting thin overlaps from merging
    overlap_merge_tolerance = collider_info.mc_tolerance[None] * geom_pair_scale
    if not is_overlap_degenerate:
        overlap_merge_tolerance = qd.min(overlap_merge_tolerance, 0.005 * qd.abs(overlap_area) / overlap_perimeter)
    target_tolerance = qd.max(EPS * geom_pair_scale, overlap_merge_tolerance)
    # Unit axes keep the perturbation angle independent of the selected target corner
    for i_t in qd.static(range(4)):
        axis = qd.Vector([axes[i_t, 0], axes[i_t, 1], axes[i_t, 2]])
        if not is_overlap_degenerate:
            axis = qd.Vector.zero(gs.qd_float, 3)
            i_k = targets[i_t]
            if i_t < n_overlap:
                is_target_distinct = overlap[i_k, 0] ** 2 + overlap[i_k, 1] ** 2 > target_tolerance**2
                for j_t in range(i_t):
                    j_k = targets[j_t]
                    axis_kept = qd.Vector([axes[j_t, 0], axes[j_t, 1], axes[j_t, 2]])
                    gap = qd.Vector([overlap[i_k, 0] - overlap[j_k, 0], overlap[i_k, 1] - overlap[j_k, 1]])
                    if axis_kept.norm_sqr() > 0.0 and gap.norm_sqr() <= target_tolerance**2:
                        is_target_distinct = False
                if is_target_distinct:
                    i_k_prev = (i_k + n_overlap - 1) % n_overlap
                    normal_prev = qd.Vector([edges_normal[i_k_prev, 0], edges_normal[i_k_prev, 1]]).normalized(EPS)
                    normal_next = qd.Vector([edges_normal[i_k, 0], edges_normal[i_k, 1]]).normalized(EPS)
                    # Inward normals of both sides of the normal cone of the corner, which the outward normals of its
                    # two edges bound. The cone holds the gradient of the tilted depth 'slope + gain *
                    # deepening_direction' as long as the corner is the deepest point of the overlap. The smallest
                    # margin of the gradient is maximized: the corner that the depth leaves farthest behind gets the
                    # whole tilt, while flat faces deepen along the middle of the cone.
                    side_prev = (normal_next - normal_next.dot(normal_prev) * normal_prev).normalized(EPS)
                    side_next = (normal_prev - normal_prev.dot(normal_next) * normal_next).normalized(EPS)
                    gain = 2.0 * collider_info.mc_perturbation[None]
                    margin_prev, margin_next = side_prev.dot(slope), side_next.dot(slope)
                    deepening_direction = side_prev
                    if margin_next + gain <= margin_prev + gain * side_prev.dot(side_next):
                        deepening_direction = side_next
                    elif margin_prev + gain > margin_next + gain * side_next.dot(side_prev):
                        # Both margins are equal along the deepening direction that maximizes them
                        sides_gap = side_prev - side_next
                        sides_gap_norm = sides_gap.norm()
                        direction_along = sides_gap / sides_gap_norm
                        direction_across = qd.Vector([-direction_along[1], direction_along[0]])
                        if direction_across.dot(side_prev) < 0.0:
                            direction_across = -direction_across
                        ratio = (margin_next - margin_prev) / (gain * sides_gap_norm)
                        deepening_direction = ratio * direction_along + qd.sqrt(qd.max(1.0 - ratio**2, 0.0)) * (
                            direction_across
                        )
                    axis = gu.qd_normalize(
                        normal.cross(deepening_direction[0] * axis_0 + deepening_direction[1] * axis_1), EPS
                    )
        for i_3 in qd.static(range(3)):
            axes[i_t, i_3] = axis[i_3]
    return axes, overlap_merge_tolerance


@qd.func
def func_convex_convex_contact(
    i_ga: int,
    i_gb: int,
    i_b: int,
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
        normal_cos_tol = qd.cos(2.0 * collider_info.mc_perturbation[None])
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
        is_gjk_used_0 = False

        # Whether narrowphase detected a contact.
        is_col = False
        penetration = gs.qd_float(0.0)
        normal = qd.Vector.zero(gs.qd_float, 3)
        contact_pos = qd.Vector.zero(gs.qd_float, 3)

        n_con = gs.qd_int(0)
        axis_0 = qd.Vector.zero(gs.qd_float, 3)
        axis_1 = qd.Vector.zero(gs.qd_float, 3)
        qrot = qd.Vector.zero(gs.qd_float, 4)
        overlap_axes = qd.Matrix.zero(gs.qd_float, 4, 3)
        # A zero axis skips the detection: see func_contact_overlap_axes
        has_target_corner = True

        i_pair = collider_info.collision_pair_idx[i_ga, i_gb]
        for i_detection in range(5):
            prefer_gjk = (
                collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.GJK
                or collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK
            )
            # Whether GJK ran, which the analytic detections and the plane bypass whatever the algorithm prefers
            is_gjk_used = False

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
                    # Each detection aims at a corner of the contact patch (see func_contact_overlap_axes)
                    axis = qd.Vector.zero(gs.qd_float, 3)
                    for i_t in qd.static(range(4)):
                        if i_detection == i_t + 1:
                            axis = qd.Vector([overlap_axes[i_t, 0], overlap_axes[i_t, 1], overlap_axes[i_t, 2]])
                    has_target_corner = axis.norm_sqr() > 0.0
                    qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

                # Apply perturbation starting from original state
                ga_pos_current, ga_quat_current = func_rotate_frame(
                    ga_pos_original, ga_quat_original, contact_pos_0, qrot
                )
                gb_pos_current, gb_quat_current = func_rotate_frame(
                    gb_pos_original, gb_quat_original, contact_pos_0, gu.qd_inv_quat(qrot)
                )

            # A detection left without corner to target runs nothing, and must not pass on the outcome of the last one
            is_col = False
            if (multi_contact and is_col_0 and has_target_corner) or (i_detection == 0):
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
                                i_ga,
                                i_gb,
                                i_pair,
                                i_b,
                                i_b,
                                mpr_state.portal_status[i_b],
                                penetration,
                                ga_pos_current,
                                ga_quat_current,
                                gb_pos_current,
                                gb_quat_current,
                                geom_pair_scale,
                                collider_state,
                                mpr_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                rigid_config,
                                collider_static_config,
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

                            is_gjk_used = True
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
                is_gjk_used_0 = is_gjk_used
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
                        if qd.static(not rigid_config.enable_mujoco_compatibility):
                            overlap_axes = qd.Matrix.rows([axis_0, -axis_0, axis_1, -axis_1])
                            if dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.PLANE:
                                overlap_axes, overlap_merge_tolerance = func_contact_overlap_axes(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    normal,
                                    contact_pos,
                                    axis_0,
                                    axis_1,
                                    ga_pos_original,
                                    ga_quat_original,
                                    gb_pos_original,
                                    gb_quat_original,
                                    overlap_axes,
                                    collider_state,
                                    dyn_info,
                                    rigid_info,
                                    collider_info,
                                    rigid_config,
                                    collider_static_config,
                                    geom_pair_scale,
                                )
                                tolerance = qd.min(tolerance, overlap_merge_tolerance)
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
                # collision methods when multi-contact is enabled, except mujoco compatible.
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    normal, penetration, contact_pos = func_recompute_perturbed_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        normal,
                        penetration,
                        contact_pos,
                        normal_0,
                        contact_pos_0,
                        penetration_0,
                        qrot,
                        ga_pos_original,
                        ga_quat_original,
                        gb_pos_original,
                        gb_quat_original,
                        mpr_state,
                        gjk_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                        geom_pair_scale,
                        is_gjk_used,
                        is_gjk_used_0,
                    )

                # For MuJoCo-compatible GJK, set penetration of perturbed contacts to equal the initial contact's
                # penetration, matching MuJoCo's behavior (engine_collision_convex.c:1010).
                if qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    penetration = penetration_0

                # Discard repeated contact points, a contact at the same point on another patch being kept: see
                # _func_multicontact_accept
                is_contact_duplicate = False
                for i_c in range(n_con):
                    if not is_contact_duplicate:
                        idx_prev = collider_state.n_contacts[i_b] - 1 - i_c
                        contact_pos_kept = collider_state.contact_data.pos[idx_prev, i_b]
                        # See the merge distance and precision floor in _func_multicontact_accept
                        merge_tolerance = tolerance
                        if qd.static(
                            collider_static_config.ccd_algorithm
                            not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                        ):
                            merge_tolerance = qd.sqrt(EPS) * geom_pair_scale
                            merge_tolerance += EPS * (contact_pos.norm() + contact_pos_kept.norm())
                            merge_tolerance = qd.max(merge_tolerance, tolerance)
                        if (contact_pos - contact_pos_kept).norm() < merge_tolerance:
                            is_contact_duplicate = True
                            if qd.static(
                                collider_static_config.ccd_algorithm
                                not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                            ):
                                normal_kept = collider_state.contact_data.normal[idx_prev, i_b]
                                is_contact_duplicate = normal.dot(normal_kept) >= normal_cos_tol

                if not is_contact_duplicate:
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
    i_ga: int,
    i_gb: int,
    i_scratch: int,
    i_b: int,
    i_pair: int,
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
    geom_pair_scale: float,
    use_gjk: bool,
    is_initial_detection: bool,
):
    """Run one detection (capsule/plane/MPR/GJK) and return (is_col, normal, contact_pos, penetration, used_gjk).

    use_gjk selects GJK over MPR where both are compiled, is_initial_detection lets MPR retry without the cached
    normal. Both are runtime flags so the kernel inlines this pipeline once whatever the mix of its detections.
    """
    EPS = rigid_info.EPS[None]
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    used_gjk = False

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
        if qd.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)):
            if not use_gjk:
                is_mpr_updated = False
                normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                is_mpr_guess_direction_available = (qd.abs(normal_ws) > EPS).any()
                for i_mpr in range(2):
                    if i_mpr == 1:
                        if qd.static(not rigid_config.enable_mujoco_compatibility):
                            if is_initial_detection and not is_col and is_mpr_guess_direction_available:
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
            collider_static_config.ccd_algorithm != CCD_ALGORITHM_CODE.MJ_MPR and not rigid_config.requires_grad
        ):
            if use_gjk:
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
def _func_multicontact_detect(
    i_scratch: int,
    i_work: int,
    i_det_start: int,
    i_det_end: int,
    overlap_axes: qd.types.matrix(4, 3),
    overlap_merge_tolerance: float,
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
    """Run a range of the detections of a queued pair and store each contact it finds.

    The call runs the detections [i_det_start, i_det_end) of the pair of multicontact queue entry i_work, with the
    Gilbert-Johnson-Keerthi (GJK) and Minkowski portal refinement (MPR) scratch of slot i_scratch. Detection 0 takes
    contact 0 from slot 0 of the entry, or detects it with GJK when the contact0 kernel left it to GJK (see
    MULTICONTACT_ENTRY in constants.py). Detections 1 to 4 are the perturbed contacts, each tried again with GJK when
    its own gate fires, except under the pure GJK algorithm where every contact is detected with GJK. A range starting
    past 0 requires contact 0 in slot 0.

    Without the contact patch, detection i stores its contact in slot i of the entry (see NarrowphaseWorkQueue in
    array_class.py) for _func_multicontact_gather. With the contact patch, the call takes the manifold GJK returns for
    contact 0 when there is one, which skips perturbation, accepts each contact as it comes and writes those of the
    pair itself once its last detection ran.
    """
    EPS = rigid_info.EPS[None]

    i_b = collider_state.work_queue.i_b[i_work]
    i_ga = collider_state.work_queue.i_ga[i_work]
    i_gb = collider_state.work_queue.i_gb[i_work]
    i_pair = collider_state.work_queue.i_pair[i_work]
    contact0_status = collider_state.work_queue.contact0_status[i_work]
    is_gjk_preferred_0 = contact0_status == MULTICONTACT_ENTRY.GJK
    # Whether GJK detected contact 0, in an earlier stage (see MULTICONTACT_ENTRY) or in this one below
    is_gjk_used_0 = contact0_status == MULTICONTACT_ENTRY.DETECTED

    ga_pos_original = dyn_state.geoms.pos[i_ga, i_b]
    ga_quat_original = dyn_state.geoms.quat[i_ga, i_b]
    gb_pos_original = dyn_state.geoms.pos[i_gb, i_b]
    gb_quat_original = dyn_state.geoms.quat[i_gb, i_b]

    is_multi_contact = (
        rigid_config.enable_multi_contact
        and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.SPHERE
        and dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
        and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.SPHERE
        and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
    )
    if qd.static(rigid_config.enable_mujoco_compatibility):
        # Capsule-capsule exclusion under compatibility: see func_convex_convex_contact.
        is_multi_contact = is_multi_contact and not (
            dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.CAPSULE and dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.CAPSULE
        )

    geom_pair_scale = func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
    tolerance = func_compute_mc_tolerance(i_ga, i_gb, geoms_init_AABB, dyn_info, collider_info, rigid_config)
    if overlap_merge_tolerance > 0.0:
        tolerance = qd.min(tolerance, overlap_merge_tolerance)
    normal_cos_tol = qd.cos(2.0 * collider_info.mc_perturbation[None])

    contact0_normal = collider_state.work_queue.contact_normal[i_work, 0]
    contact0_pos = collider_state.work_queue.contact_pos[i_work, 0]
    contact0_penetration = collider_state.work_queue.contact_penetration[i_work, 0]
    has_contact0 = contact0_status == MULTICONTACT_ENTRY.SEED or contact0_status == MULTICONTACT_ENTRY.DETECTED
    is_gjk_multi_done = False
    n_con = gs.qd_int(0)
    contact_pos_local = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 3)
    normal_local = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 3)
    penetration_local = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 1)

    # One runtime loop drives every detection of the call, so the detection pipeline is inlined once in the kernel
    axis_0 = qd.Vector.zero(gs.qd_float, 3)
    axis_1 = qd.Vector.zero(gs.qd_float, 3)
    is_axes_computed = False
    for i_det in range(i_det_start, i_det_end):
        # A zero axis skips the detection: see func_contact_overlap_axes
        has_target_corner = True
        is_initial_detection = i_det == 0
        is_active = is_gjk_preferred_0
        if not is_initial_detection:
            is_active = is_multi_contact and has_contact0 and not is_gjk_multi_done
        slot_status = MULTICONTACT_SLOT.EMPTY
        slot_normal = qd.Vector.zero(gs.qd_float, 3)
        slot_pos = qd.Vector.zero(gs.qd_float, 3)
        slot_penetration = gs.qd_float(0.0)
        is_slot_written = False
        if is_initial_detection and not is_gjk_preferred_0:
            # Slot 0 holds contact 0, detected, refined and cached by the contact0 kernel or by GJK in the first stage
            if qd.static(gjk_static_config.enable_contact_patch):
                slot_status = MULTICONTACT_SLOT.BASE
                slot_normal = contact0_normal
                slot_pos = contact0_pos
                slot_penetration = collider_state.work_queue.contact_penetration[i_work, 0]
            else:
                collider_state.work_queue.contact_status[i_work, 0] = MULTICONTACT_SLOT.BASE
                is_slot_written = True
        if is_active:
            if not is_initial_detection and not is_axes_computed:
                if qd.static(rigid_config.enable_mujoco_compatibility):
                    axis_0, axis_1 = func_contact_orthogonals(
                        i_ga, i_gb, i_b, contact0_normal, geoms_init_AABB, dyn_state, dyn_info, rigid_info, rigid_config
                    )
                is_axes_computed = True
            # Declared ahead of the branches that assign them, as quadrants scoping requires.
            qrot = qd.Vector.zero(gs.qd_float, 4)
            ga_pos_current = ga_pos_original
            ga_quat_current = ga_quat_original
            gb_pos_current = gb_pos_original
            gb_quat_current = gb_quat_original
            if not is_initial_detection:
                if qd.static(rigid_config.enable_mujoco_compatibility):
                    # Single-axis perturbation pattern: see the twin loop in func_convex_convex_contact.
                    axis_idx = (i_det - 1) // 2
                    angle_sign = 2 * ((i_det - 1) % 2) - 1
                    axis = axis_0 if axis_idx == 0 else axis_1
                    qrot = gu.qd_rotvec_to_quat(angle_sign * collider_info.mc_perturbation[None] * axis, EPS)
                else:
                    # Each detection aims at a corner of the patch: see the twin loop in func_convex_convex_contact
                    axis = qd.Vector.zero(gs.qd_float, 3)
                    for i_t in qd.static(range(4)):
                        if i_det == i_t + 1:
                            axis = qd.Vector([overlap_axes[i_t, 0], overlap_axes[i_t, 1], overlap_axes[i_t, 2]])
                    has_target_corner = axis.norm_sqr() > 0.0
                    qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)
                ga_pos_current, ga_quat_current = func_rotate_frame(
                    ga_pos_original, ga_quat_original, contact0_pos, qrot
                )
                gb_pos_current, gb_quat_current = func_rotate_frame(
                    gb_pos_original, gb_quat_original, contact0_pos, gu.qd_inv_quat(qrot)
                )

            # Perturbed contacts try MPR first under the MPR algorithm and fall back to GJK per contact. The pure GJK
            # algorithms detect every contact with GJK directly.
            use_gjk = is_initial_detection
            if qd.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.GJK, CCD_ALGORITHM_CODE.MJ_GJK)):
                use_gjk = True
            is_col = False
            normal = qd.Vector.zero(gs.qd_float, 3)
            contact_pos = qd.Vector.zero(gs.qd_float, 3)
            penetration = gs.qd_float(0.0)
            is_gjk_used = False
            is_detected = not has_target_corner
            for i_try in range(2):
                if not is_detected:
                    is_col, normal, contact_pos, penetration, is_gjk_used = _func_multicontact_run_detection(
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
                        geom_pair_scale,
                        use_gjk,
                        is_initial_detection,
                    )
                    is_detected = True
                    if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                        # When the refinement is called for it applies to this perturbed contact only, rather than
                        # upgrading the whole pair, which keeps the MPR-first model.
                        if not is_initial_detection and is_col and not use_gjk:
                            is_gjk_preferred = func_prefer_gjk_refinement(
                                i_ga,
                                i_gb,
                                i_pair,
                                i_b,
                                i_scratch,
                                mpr_state.portal_status[i_scratch],
                                penetration,
                                ga_pos_current,
                                ga_quat_current,
                                gb_pos_current,
                                gb_quat_current,
                                geom_pair_scale,
                                collider_state,
                                mpr_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                rigid_config,
                                collider_static_config,
                            )
                            if is_gjk_preferred:
                                use_gjk = True
                                is_detected = False

            if is_initial_detection:
                if is_col:
                    collider_state.contact_cache.normal[i_pair, i_b] = normal
                    collider_state.contact_cache.penetration[i_pair, i_b] = penetration
                    # Only the contact patch returns a manifold, whose points are all accepted as they are and replace
                    # perturbation. GJK otherwise returns the single contact it found, and an analytic detection returns
                    # its contact directly.
                    n_contacts_manifold = 0
                    if qd.static(gjk_static_config.enable_contact_patch):
                        if is_gjk_used and gjk_state.multi_contact_flag[i_scratch] == 1:
                            n_contacts_manifold = qd.min(
                                gjk_state.n_contacts[i_scratch], collider_static_config.n_contacts_per_convex_pair
                            )
                            is_gjk_multi_done = True
                    for i_c in range(n_contacts_manifold):
                        manifold_normal = gjk_state.normal[i_scratch, i_c]
                        manifold_pos = func_apply_smooth_refinement(
                            i_ga,
                            i_gb,
                            manifold_normal,
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
                            contact_pos_local[n_con, i_3] = manifold_pos[i_3]
                            normal_local[n_con, i_3] = manifold_normal[i_3]
                        penetration_local[n_con, 0] = penetration
                        n_con = n_con + 1
                    if not is_gjk_multi_done:
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
                        contact0_penetration = penetration
                        is_gjk_used_0 = is_gjk_used
                        slot_status = MULTICONTACT_SLOT.BASE
                        slot_normal = normal
                        slot_pos = contact_pos
                        slot_penetration = penetration
                    has_contact0 = True
                else:
                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    collider_state.contact_cache.penetration[i_pair, i_b] = 0.0
            elif is_col:
                # The perturbed contact is refined inside func_recompute_perturbed_contact (after the perturbation is
                # reverted, on the canonical pose); no pre-reversal refinement is needed here.
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    normal, penetration, contact_pos = func_recompute_perturbed_contact(
                        i_ga,
                        i_gb,
                        i_scratch,
                        normal,
                        penetration,
                        contact_pos,
                        contact0_normal,
                        contact0_pos,
                        contact0_penetration,
                        qrot,
                        ga_pos_original,
                        ga_quat_original,
                        gb_pos_original,
                        gb_quat_original,
                        mpr_state,
                        gjk_state,
                        dyn_info,
                        rigid_info,
                        collider_info,
                        rigid_config,
                        geom_pair_scale,
                        is_gjk_used,
                        is_gjk_used_0,
                    )
                slot_status = MULTICONTACT_SLOT.PERTURBED
                slot_normal = normal
                slot_pos = contact_pos
                slot_penetration = penetration

        if qd.static(gjk_static_config.enable_contact_patch):
            if slot_status != MULTICONTACT_SLOT.EMPTY:
                is_contact_kept, slot_penetration = _func_multicontact_accept(
                    n_con,
                    slot_status,
                    slot_pos,
                    slot_normal,
                    slot_penetration,
                    tolerance,
                    normal_cos_tol,
                    contact_pos_local,
                    normal_local,
                    penetration_local,
                    collider_static_config,
                    geom_pair_scale,
                    EPS,
                )
                if is_contact_kept:
                    for i_3 in qd.static(range(3)):
                        contact_pos_local[n_con, i_3] = slot_pos[i_3]
                        normal_local[n_con, i_3] = slot_normal[i_3]
                    penetration_local[n_con, 0] = slot_penetration
                    n_con = n_con + 1
        elif not is_slot_written:
            # The MPR seed marks its slot above, and every other detection stores its outcome here, empty when it did
            # not run or found no contact
            collider_state.work_queue.contact_pos[i_work, i_det] = slot_pos
            collider_state.work_queue.contact_normal[i_work, i_det] = slot_normal
            collider_state.work_queue.contact_penetration[i_work, i_det] = slot_penetration
            collider_state.work_queue.contact_status[i_work, i_det] = slot_status

    # A contact 0 found with Gilbert-Johnson-Keerthi (GJK) is stored in slot 0 as the contact0 kernel stores the MPR
    # seed, for the stages that aim and perturb around it, unless nothing is left to detect for the pair
    is_detected_for_later = False
    if is_gjk_preferred_0:
        contact0_status = MULTICONTACT_ENTRY.DONE
        if has_contact0 and is_multi_contact and not is_gjk_multi_done:
            contact0_status = MULTICONTACT_ENTRY.DETECTED if is_gjk_used_0 else MULTICONTACT_ENTRY.SEED
            is_detected_for_later = True
            # Without the contact patch, detection 0 stored its contact in slot 0 already
            if qd.static(gjk_static_config.enable_contact_patch):
                collider_state.work_queue.contact_pos[i_work, 0] = contact0_pos
                collider_state.work_queue.contact_normal[i_work, 0] = contact0_normal
                collider_state.work_queue.contact_penetration[i_work, 0] = contact0_penetration
        collider_state.work_queue.contact0_status[i_work] = contact0_status

    if qd.static(gjk_static_config.enable_contact_patch):
        if not is_detected_for_later:
            _func_multicontact_write(
                i_b,
                i_ga,
                i_gb,
                i_pair,
                n_con,
                contact_pos_local,
                normal_local,
                penetration_local,
                dyn_state,
                collider_state,
                dyn_info,
                rigid_info,
                collider_info,
                errno,
            )


@qd.func
def _func_multicontact_accept(
    n_con: int,
    status: int,
    contact_pos: qd.types.vector(3),
    normal: qd.types.vector(3),
    penetration: float,
    tolerance: float,
    normal_cos_tol: float,
    contact_pos_local,
    normal_local,
    penetration_local,
    collider_static_config: qd.template(),
    geom_pair_scale: float,
    EPS: float,
):
    """Decide whether a candidate contact of a pair is kept, given the contacts of the pair kept so far.

    The first n_con rows of contact_pos_local, normal_local and penetration_local hold the contacts kept so far. A
    contact of status BASE (see MULTICONTACT_SLOT in constants.py) is kept as it is. A perturbed contact is kept unless
    it lies within the tolerance of a contact already kept whose normal agrees with its own, the cosine of their angle
    being at least normal_cos_tol. Returns whether the contact is kept and its penetration.

    A contact projected onto the patch of the initial contact carries the normal of the initial contact, and a contact
    left on its own patch carries the normal of a face of the Minkowski difference, or the normal of its detection
    when that face is degenerate, which the perturbation tilts by twice its angle at most. Contacts of one patch thus
    agree within twice the perturbation angle, while two patches meeting at one point keep a contact each. The
    MuJoCo-compatible algorithms compare positions alone, as MuJoCo does.
    """
    is_contact_duplicate = False
    if status != MULTICONTACT_SLOT.BASE:
        # Perturbed contacts carry the initial contact's penetration under compatibility: see the twin acceptance in
        # func_convex_convex_contact.
        if qd.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)):
            penetration = penetration_local[0, 0]

        for i_c in range(n_con):
            if not is_contact_duplicate:
                contact_pos_kept = qd.Vector(
                    [contact_pos_local[i_c, 0], contact_pos_local[i_c, 1], contact_pos_local[i_c, 2]], dt=gs.qd_float
                )
                merge_tolerance = tolerance
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    # Surface reconstruction collapses edges below EPS * geom_pair_scale**2 in squared length.
                    # Keep that feature resolution and world-position rounding as a floor on the patch merge distance
                    merge_tolerance = qd.sqrt(EPS) * geom_pair_scale
                    merge_tolerance += EPS * (contact_pos.norm() + contact_pos_kept.norm())
                    merge_tolerance = qd.max(merge_tolerance, tolerance)
                if (contact_pos - contact_pos_kept).norm() < merge_tolerance:
                    is_contact_duplicate = True
                    if qd.static(
                        collider_static_config.ccd_algorithm
                        not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                    ):
                        normal_kept = qd.Vector(
                            [normal_local[i_c, 0], normal_local[i_c, 1], normal_local[i_c, 2]], dt=gs.qd_float
                        )
                        is_contact_duplicate = normal.dot(normal_kept) >= normal_cos_tol
    return not is_contact_duplicate, penetration


@qd.func
def _func_multicontact_write(
    i_b: int,
    i_ga: int,
    i_gb: int,
    i_pair: int,
    n_con: int,
    contact_pos_local,
    normal_local,
    penetration_local,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    errno: qd.Tensor,
):
    """Write the contacts kept for a pair contiguously via a single atomic reservation.

    The first n_con rows of the local arrays hold the contacts, and the single reservation gives deterministic per-pair
    contact ordering. The contacts that no longer fit in the contact buffer are dropped and flag the overflow in errno.
    """
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
                    [contact_pos_local[i, 0], contact_pos_local[i, 1], contact_pos_local[i, 2]], dt=gs.qd_float
                )
                normal_i = qd.Vector([normal_local[i, 0], normal_local[i, 1], normal_local[i, 2]], dt=gs.qd_float)
                func_set_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    i_c,
                    i_pair,
                    normal_i,
                    pos_i,
                    penetration_local[i, 0],
                    dyn_state,
                    collider_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    errno,
                )


@qd.func
def _func_multicontact_gather(
    i_work: int,
    overlap_merge_tolerance: float,
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
    """Accept the candidate contacts of a queued pair in slot order and write those it keeps.

    The slots of multicontact queue entry i_work (see NarrowphaseWorkQueue in array_class.py) go through
    _func_multicontact_accept in order, and the contacts kept are written by _func_multicontact_write.
    """
    i_b = collider_state.work_queue.i_b[i_work]
    i_ga = collider_state.work_queue.i_ga[i_work]
    i_gb = collider_state.work_queue.i_gb[i_work]
    i_pair = collider_state.work_queue.i_pair[i_work]

    tolerance = func_compute_mc_tolerance(i_ga, i_gb, geoms_init_AABB, dyn_info, collider_info, rigid_config)
    if overlap_merge_tolerance > 0.0:
        tolerance = qd.min(tolerance, overlap_merge_tolerance)
    EPS = rigid_info.EPS[None]
    geom_pair_scale = func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
    normal_cos_tol = qd.cos(2.0 * collider_info.mc_perturbation[None])

    n_con = gs.qd_int(0)
    contact_pos_local = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 3)
    normal_local = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 3)
    penetration_local = qd.Matrix.zero(gs.qd_float, collider_static_config.n_contacts_per_convex_pair, 1)
    for i_s in range(collider_state.work_queue.contact_status.shape[1]):
        status = collider_state.work_queue.contact_status[i_work, i_s]
        if status != MULTICONTACT_SLOT.EMPTY:
            contact_pos = collider_state.work_queue.contact_pos[i_work, i_s]
            normal = collider_state.work_queue.contact_normal[i_work, i_s]
            is_contact_kept, penetration = _func_multicontact_accept(
                n_con,
                status,
                contact_pos,
                normal,
                collider_state.work_queue.contact_penetration[i_work, i_s],
                tolerance,
                normal_cos_tol,
                contact_pos_local,
                normal_local,
                penetration_local,
                collider_static_config,
                geom_pair_scale,
                EPS,
            )
            if is_contact_kept:
                for i_3 in qd.static(range(3)):
                    contact_pos_local[n_con, i_3] = contact_pos[i_3]
                    normal_local[n_con, i_3] = normal[i_3]
                penetration_local[n_con, 0] = penetration
                n_con = n_con + 1

    _func_multicontact_write(
        i_b,
        i_ga,
        i_gb,
        i_pair,
        n_con,
        contact_pos_local,
        normal_local,
        penetration_local,
        dyn_state,
        collider_state,
        dyn_info,
        rigid_info,
        collider_info,
        errno,
    )


@qd.func
def func_narrowphase_multicontact(
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
    """Detect the perturbed contacts of every pair queued for the multi-contact pass.

    The pass runs each queued pair in three stages, the stages of a pair sharing a block whose barrier separates them,
    so that each stage reads every slot the previous one wrote: the first unit of the pair stores contact 0, detecting
    it with Gilbert-Johnson-Keerthi (GJK) when the contact0 kernel left it so, then aims the perturbed detections at the
    corners of the contact patch around it (see func_contact_overlap_axes), then every unit runs its share of the
    perturbed detections. While the GPU cores outnumber the work units, a pair splits into one unit per perturbed
    detection, which read contact 0 and nothing of one another. A queue long enough to saturate the GPU keeps one unit
    per pair, since the split then adds work without adding lanes, and so does the contact patch, which gathers the
    contacts of a pair in its unit. A block claims as many pairs at once as it has units, which keeps the units of a
    pair in the same block. Its first unit accepts the contacts after all detections finish. Axes and merge distances
    stay in shared memory until acceptance. Every stage has one call site so the detection and outline compile once.
    """
    N_PERTURBATIONS = qd.static(4)
    BLOCK_DIM = qd.static(32)
    qd.loop_config(name="narrowphase_multicontact", block_dim=BLOCK_DIM)
    for i_tid in range(collider_static_config.gpu_cores):
        i_lane = i_tid % BLOCK_DIM
        sh_work_start = qd.simt.block.SharedArray((1,), gs.qd_int)
        sh_aim_axes = qd.simt.block.SharedArray((BLOCK_DIM, 4, 3), gs.qd_float)
        sh_n_aims = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_int)
        sh_merge_tolerance = qd.simt.block.SharedArray((BLOCK_DIM,), gs.qd_float)
        n_queue = collider_state.work_queue.n_pairs[0]
        n_units = 1
        if qd.static(not gjk_static_config.enable_contact_patch):
            if N_PERTURBATIONS * n_queue <= collider_static_config.gpu_cores:
                n_units = N_PERTURBATIONS
        n_works_per_claim = BLOCK_DIM // n_units
        for i_claim_ in range(n_queue + 1):
            # A block takes its first works by its rank and the next ones from the shared counter, offset past the
            # works taken by rank, so that the blocks a short queue leaves idle exit without an atomic on the counter,
            # which every block of the launch would otherwise hit at once.
            i_work_start = (i_tid // BLOCK_DIM) * n_works_per_claim
            if i_claim_ > 0:
                # FIXME: quadrants#925 - every lane of the block issues the claim atomic, the other lanes adding 0,
                # instead of one atomic per block.
                i_claim = qd.atomic_add(collider_state.work_queue.n_claimed[0], n_works_per_claim if i_lane == 0 else 0)
                if i_lane == 0:
                    n_blocks = (collider_static_config.gpu_cores + BLOCK_DIM - 1) // BLOCK_DIM
                    sh_work_start[0] = n_blocks * n_works_per_claim + i_claim
                qd.simt.block.sync()
                i_work_start = sh_work_start[0]
                qd.simt.block.sync()
            i_work = i_work_start + i_lane // n_units
            if i_work_start >= n_queue:
                break
            i_unit = i_lane % n_units
            if i_work < n_queue and i_unit == 0:
                sh_n_aims[i_lane // n_units] = N_PERTURBATIONS
                sh_merge_tolerance[i_lane // n_units] = 0.0
                if qd.static(not gjk_static_config.enable_contact_patch):
                    for i_s in range(1, N_PERTURBATIONS + 1):
                        collider_state.work_queue.contact_status[i_work, i_s] = MULTICONTACT_SLOT.EMPTY
            for i_stage in range(3):
                contact0_status = MULTICONTACT_ENTRY.DONE
                if i_work < n_queue:
                    contact0_status = collider_state.work_queue.contact0_status[i_work]
                is_contact0_known = (
                    contact0_status == MULTICONTACT_ENTRY.SEED or contact0_status == MULTICONTACT_ENTRY.DETECTED
                )
                if i_stage == 1:
                    if qd.static(rigid_config.enable_multi_contact and not rigid_config.enable_mujoco_compatibility):
                        if i_unit == 0 and is_contact0_known:
                            # Contact 0 is read from slot 0, where the contact0 kernel stores the Minkowski portal
                            # refinement (MPR) seed and the first stage stores a Gilbert-Johnson-Keerthi (GJK) contact
                            i_b = collider_state.work_queue.i_b[i_work]
                            i_ga = collider_state.work_queue.i_ga[i_work]
                            i_gb = collider_state.work_queue.i_gb[i_work]
                            normal = collider_state.work_queue.contact_normal[i_work, 0]
                            contact_pos = collider_state.work_queue.contact_pos[i_work, 0]
                            geom_pair_scale = func_compute_geom_pair_scale(i_ga, i_gb, geoms_init_AABB, dyn_info)
                            axis_0, axis_1 = func_contact_orthogonals(
                                i_ga, i_gb, i_b, normal, geoms_init_AABB, dyn_state, dyn_info, rigid_info, rigid_config
                            )
                            overlap_axes = qd.Matrix.rows([axis_0, -axis_0, axis_1, -axis_1])
                            if dyn_info.geoms.type[i_ga] != gs.GEOM_TYPE.PLANE:
                                overlap_axes, overlap_merge_tolerance = func_contact_overlap_axes(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    normal,
                                    contact_pos,
                                    axis_0,
                                    axis_1,
                                    dyn_state.geoms.pos[i_ga, i_b],
                                    dyn_state.geoms.quat[i_ga, i_b],
                                    dyn_state.geoms.pos[i_gb, i_b],
                                    dyn_state.geoms.quat[i_gb, i_b],
                                    overlap_axes,
                                    collider_state,
                                    dyn_info,
                                    rigid_info,
                                    collider_info,
                                    rigid_config,
                                    collider_static_config,
                                    geom_pair_scale,
                                )
                                sh_merge_tolerance[i_lane // n_units] = overlap_merge_tolerance
                            # Compact valid aims so inactive lanes can skip the detection call entirely
                            n_aims = 0
                            for i_t in range(N_PERTURBATIONS):
                                axis = qd.Vector([overlap_axes[i_t, 0], overlap_axes[i_t, 1], overlap_axes[i_t, 2]])
                                if axis.norm_sqr() > 0.0:
                                    for i_3 in qd.static(range(3)):
                                        sh_aim_axes[i_lane // n_units, n_aims, i_3] = axis[i_3]
                                    n_aims += 1
                            sh_n_aims[i_lane // n_units] = n_aims
                            for i_t in range(n_aims, N_PERTURBATIONS):
                                for i_3 in qd.static(range(3)):
                                    sh_aim_axes[i_lane // n_units, i_t, i_3] = 0.0
                else:
                    # The first stage stores contact 0, which marks the slot of the MPR seed and detects the contact the
                    # GJK pairs left to it. The last one runs the perturbed detections of the unit, all of them on a
                    # single unit, and stores the empty slots of a pair left without any. The contact patch reads
                    # contact 0 back from slot 0 as its first contact, and gathers the contacts of a pair in its last
                    # stage unless the first one held them all.
                    i_det_start = 0
                    i_det_end = 0
                    if i_work < n_queue:
                        if i_stage == 0:
                            if i_unit == 0:
                                if qd.static(gjk_static_config.enable_contact_patch):
                                    if contact0_status == MULTICONTACT_ENTRY.GJK:
                                        i_det_end = 1
                                else:
                                    i_det_end = 1
                        else:
                            if qd.static(gjk_static_config.enable_contact_patch):
                                if is_contact0_known:
                                    i_det_end = N_PERTURBATIONS + 1
                            else:
                                i_det_start = i_unit + 1
                                i_det_end = i_unit + 2 if n_units > 1 else N_PERTURBATIONS + 1
                    if i_stage == 2 and i_work < n_queue:
                        i_det_end = qd.min(i_det_end, sh_n_aims[i_lane // n_units] + 1)
                    if i_det_start < i_det_end:
                        overlap_merge_tolerance = gs.qd_float(0.0)
                        if i_stage == 2:
                            overlap_merge_tolerance = sh_merge_tolerance[i_lane // n_units]
                        overlap_axes = qd.Matrix.zero(gs.qd_float, 4, 3)
                        if qd.static(
                            rigid_config.enable_multi_contact and not rigid_config.enable_mujoco_compatibility
                        ):
                            if i_stage == 2 and is_contact0_known:
                                for i_t in qd.static(range(4)):
                                    for i_3 in qd.static(range(3)):
                                        overlap_axes[i_t, i_3] = sh_aim_axes[i_lane // n_units, i_t, i_3]
                        _func_multicontact_detect(
                            i_tid,
                            i_work,
                            i_det_start,
                            i_det_end,
                            overlap_axes,
                            overlap_merge_tolerance,
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
                # The slots live in device memory, which the block barrier of the Standard Portable Intermediate
                # Representation (SPIR-V) backends leaves unordered
                qd.simt.grid.mem_fence()
                qd.simt.block.sync()

            if qd.static(not gjk_static_config.enable_contact_patch):
                if i_work < n_queue and i_unit == 0:
                    _func_multicontact_gather(
                        i_work,
                        sh_merge_tolerance[i_lane // n_units],
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


@qd.func
def _func_enqueue_for_multicontact(
    i_b: int,
    i_ga: int,
    i_gb: int,
    i_pair: int,
    contact_pos_0: qd.types.vector(3),
    normal_0: qd.types.vector(3),
    penetration_0: float,
    collider_state: array_class.ColliderState,
    prefer_gjk: bool,
):
    idx = qd.atomic_add(collider_state.work_queue.n_pairs[0], 1)
    collider_state.work_queue.i_b[idx] = i_b
    collider_state.work_queue.i_ga[idx] = i_ga
    collider_state.work_queue.i_gb[idx] = i_gb
    collider_state.work_queue.i_pair[idx] = i_pair
    collider_state.work_queue.contact_pos[idx, 0] = contact_pos_0
    collider_state.work_queue.contact_normal[idx, 0] = normal_0
    collider_state.work_queue.contact_penetration[idx, 0] = penetration_0
    collider_state.work_queue.contact0_status[idx] = MULTICONTACT_ENTRY.GJK if prefer_gjk else MULTICONTACT_ENTRY.SEED


@qd.func
def func_narrowphase_contact0(
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
    errno: qd.Tensor,
):
    n_envs = collider_state.n_broad_pairs.shape[0]
    n_chunks = (collider_static_config.gpu_cores + n_envs - 1) // n_envs
    _grid_size = n_envs * n_chunks
    max_broad_pairs = collider_state.broad_collision_pairs.shape[0]

    # The queue the loop below fills is emptied first. The top-level loops of a kernel run in order, so the reset
    # costs no launch of its own
    for _ in range(1):
        collider_state.work_queue.n_pairs[0] = 0
        collider_state.work_queue.n_claimed[0] = 0

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for flat_idx in range(_grid_size):
        i_b = flat_idx // n_chunks
        chunk = flat_idx % n_chunks
        n_pairs = qd.min(collider_state.n_broad_pairs[i_b], max_broad_pairs)
        pair_start = chunk * n_pairs // n_chunks
        pair_end = (chunk + 1) * n_pairs // n_chunks

        for i_pair_local in range(max_broad_pairs):
            i_pair_idx = pair_start + i_pair_local
            if i_pair_idx >= pair_end:
                break

            i_ga = collider_state.broad_collision_pairs[i_pair_idx, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair_idx, i_b][1]

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

            i_pair = collider_info.collision_pair_idx[i_ga, i_gb]

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
                        collider_state,
                        gjk_state,
                        dyn_info,
                        collider_info,
                        rigid_config,
                        collider_static_config,
                        shrink_sphere=False,
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
                            i_ga,
                            i_gb,
                            i_pair,
                            i_b,
                            flat_idx,
                            mpr_state.portal_status[flat_idx],
                            penetration,
                            ga_pos,
                            ga_quat,
                            gb_pos,
                            gb_quat,
                            geom_pair_scale,
                            collider_state,
                            mpr_state,
                            dyn_info,
                            rigid_info,
                            collider_info,
                            rigid_config,
                            collider_static_config,
                        )

            if is_col:
                if qd.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)):
                    collider_state.contact_cache.normal[i_pair, i_b] = normal
                    collider_state.contact_cache.penetration[i_pair, i_b] = penetration
                # Refine the contact position before enqueueing or storing it. The downstream multicontact functions
                # store this as the initial contact (index 0 of contact_pos_local) without re-refining, so refinement
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
                        i_b, i_ga, i_gb, i_pair, contact_pos, normal, penetration, collider_state, prefer_gjk
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


@qd.func
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
        for i_pair in range(qd.min(collider_state.n_broad_pairs[i_b], collider_state.broad_collision_pairs.shape[0])):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

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
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
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


@qd.func
def func_fill_diff_contact_input_analytic(
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


@qd.func
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
        for i_pair in range(qd.min(collider_state.n_broad_pairs[i_b], collider_state.broad_collision_pairs.shape[0])):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

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


@qd.func
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
    """Detect the contacts of every convex geom against the terrain.

    The terrain pairs of every env are queued, then claimed one at a time by the blocks of a pool of threads, which
    balances the blocks whatever the cost of their pairs. A block detects its pair with its lanes (see
    func_contact_mpr_terrain), every thread on its own slot of mpr_state and of the prism scratch.
    """
    if qd.static(collider_static_config.has_terrain):
        is_coop = qd.static(rigid_config.backend != gs.cpu and rigid_config.para_level >= gs.PARA_LEVEL.PARTIAL)
        BLOCK_DIM = qd.static(32 if is_coop else 1)
        _B = collider_state.active_buffer.shape[1]
        max_broad_pairs = collider_state.broad_collision_pairs.shape[0]
        n_threads = collider_state.prism.shape[1]

        # The top-level loops of a kernel run in order, so the reset costs no launch of its own
        for _ in range(1):
            collider_state.work_queue.n_pairs[0] = 0
            collider_state.work_queue.n_claimed[0] = 0

        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b, i_pair in qd.ndrange(_B, max_broad_pairs):
            if i_pair < collider_state.n_broad_pairs[i_b]:
                i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
                i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]
                if dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.TERRAIN:
                    # The terrain is detected against the convex hull of the other geom, which would fill the cavities
                    # of a non-convex geom
                    if dyn_info.geoms.is_convex[i_ga]:
                        i_work = qd.atomic_add(collider_state.work_queue.n_pairs[0], 1)
                        collider_state.work_queue.i_b[i_work] = i_b
                        collider_state.work_queue.i_ga[i_work] = i_ga
                        collider_state.work_queue.i_gb[i_work] = i_gb
                    else:
                        errno[i_b] = errno[i_b] | array_class.ErrorCode.UNSUPPORTED_COLLISION_PAIR

        qd.loop_config(
            name="narrowphase_terrain",
            serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL,
            block_dim=qd.static(BLOCK_DIM if is_coop else None),
        )
        for i_scratch in range(n_threads):
            tid = i_scratch % BLOCK_DIM
            n_queue = collider_state.work_queue.n_pairs[0]
            # A block takes its first pair by its rank and the next ones from the counter, which spares an atomic to the
            # blocks a short queue leaves idle
            i_work = i_scratch // BLOCK_DIM
            for i_claim_ in range(n_queue + 1):
                if i_claim_ > 0:
                    # One claim per block, issued by its first lane
                    i_claim = 0
                    if tid == 0:
                        i_claim = qd.atomic_add(collider_state.work_queue.n_claimed[0], 1)
                    if qd.static(BLOCK_DIM > 1):
                        i_claim = su.qd_block_shuffle(i_claim, 0)
                    i_work = n_threads // BLOCK_DIM + i_claim
                if i_work >= n_queue:
                    break
                func_contact_mpr_terrain(
                    collider_state.work_queue.i_ga[i_work],
                    collider_state.work_queue.i_gb[i_work],
                    i_scratch,
                    collider_state.work_queue.i_b[i_work],
                    tid,
                    geoms_init_AABB,
                    dyn_state,
                    collider_state,
                    mpr_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    rigid_config,
                    collider_static_config,
                    block_dim=BLOCK_DIM,
                    errno=errno,
                )


@qd.func
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

    if qd.static(collider_static_config.has_nonconvex_nonterrain):
        n_max = qd.static(
            collider_static_config.n_contacts_per_nonconvex_pair if rigid_config.enable_multi_contact else 1
        )
        # The pairs of an env are split over chunks, every pair detected by one block whose lanes split its work. Pair
        # costs span orders of magnitude, so the chunks outnumber the blocks the GPU holds at once 8 times, capped by
        # the pairs an env can hold, a block done early taking the next one. They run chunk by chunk over the envs,
        # which keeps the loaded chunks of envs with few pairs side by side.
        is_coop = qd.static(rigid_config.backend != gs.cpu and rigid_config.para_level >= gs.PARA_LEVEL.PARTIAL)
        BLOCK_DIM = qd.static(32 if is_coop else 1)
        _B = collider_state.active_buffer.shape[1]
        max_broad_pairs = collider_state.broad_collision_pairs.shape[0]
        n_chunks = qd.max(
            qd.min((8 * collider_static_config.gpu_cores // BLOCK_DIM + _B - 1) // _B, max_broad_pairs), 1
        )
        qd.loop_config(
            name="narrowphase_nonconvex",
            serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL,
            block_dim=qd.static(BLOCK_DIM if is_coop else None),
        )
        for i_flat in range(_B * n_chunks * BLOCK_DIM):
            tid = i_flat % BLOCK_DIM
            i_b_chunk = i_flat // BLOCK_DIM
            i_b = i_b_chunk % _B
            i_chunk = i_b_chunk // _B
            n_pairs = qd.min(collider_state.n_broad_pairs[i_b], max_broad_pairs)
            for i_pair in range(i_chunk * n_pairs // n_chunks, (i_chunk + 1) * n_pairs // n_chunks):
                i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
                i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]
                if (
                    not (dyn_info.geoms.is_convex[i_ga] and dyn_info.geoms.is_convex[i_gb])
                    and dyn_info.geoms.type[i_gb] != gs.GEOM_TYPE.TERRAIN
                ):
                    # Place the bounded side at i_ga: the polytope helper iterates A's verts and queries B's SDF.
                    # PLANE has infinite extent so it must be i_gb. For non-PLANE pairs the smaller-AABB side goes to
                    # i_ga (its verts are dense relative to the contact patch, while the larger side's scan wastes
                    # cycles on verts far from any contact). The type-ascending order of the pair is the tiebreaker so
                    # smooth primitives (SPHERE/ELLIPSOID/CAPSULE) stay at i_ga, matching the smooth-contact refinement
                    # convention shared with the convex-vs-convex narrowphase.
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
                    if is_shell_pair and tid == 0:
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
                    elif not is_shell_pair:
                        # Swapped-role verification: A's vertex scan cannot see a feature of B crossing one of A's
                        # faces BETWEEN A's verts, so B's verts are checked against A's SDF as well - as a seeded
                        # local search rather than a full scan. Skipped only for PLANE (its handful of far-flung
                        # verts carry no contact information).
                        n_scans = 1 if dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.PLANE else 2
                        n_added = 0
                        contacts_slot = qd.Vector.zero(gs.qd_int, (n_max + BLOCK_DIM - 1) // BLOCK_DIM)
                        for i_scan in range(n_scans):
                            i_g1, i_g2 = i_ga, i_gb
                            g1_pos, g1_quat, g2_pos, g2_quat = ga_pos, ga_quat, gb_pos, gb_quat
                            if i_scan > 0:
                                i_g1, i_g2 = i_gb, i_ga
                                g1_pos, g1_quat, g2_pos, g2_quat = gb_pos, gb_quat, ga_pos, ga_quat
                            n_added, contacts_slot = func_add_polytope_vertex_contacts_sdf(
                                i_g1,
                                i_g2,
                                i_b,
                                i_pair,
                                tid,
                                g1_pos,
                                g1_quat,
                                g2_pos,
                                g2_quat,
                                tolerance,
                                n_added,
                                contacts_slot,
                                geoms_init_AABB,
                                dyn_state,
                                collider_state,
                                dyn_info,
                                rigid_info,
                                collider_info,
                                rigid_config,
                                collider_static_config,
                                block_dim=BLOCK_DIM,
                                seeded=i_scan > 0,
                                errno=errno,
                            )
