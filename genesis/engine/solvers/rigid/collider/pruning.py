"""Reduce redundant rigid contacts while preserving distributed support and separation constraints."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.simt as su

from .constants import CONTACT_ORDER
from .contact import func_contact_order_key


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
    """Whether a contact sorts after another in the requested contact order.

    Contact 'i_x' is compared with contact 'i_y' of environment 'i_b' in the order 'order' (see CONTACT_ORDER).
    Link-pair and intrinsic ordering compare contact data to resolve ties. Later keys are read only when earlier keys
    tie, reducing memory accesses during sorting.
    """
    is_after = False
    is_tied = True
    for i_group in qd.static(range(2 if order == CONTACT_ORDER.LINK_PAIR else 1)):
        if is_tied:
            keys = qd.Matrix.zero(gs.qd_float, 2, 3)
            for i_k, i_c in qd.static(enumerate((i_x, i_y))):
                key = qd.Vector.zero(gs.qd_float, 3)
                if qd.static(order == CONTACT_ORDER.LINK_PAIR and i_group == 0):
                    i_la = collider_state.contact_data.link_a[i_c, i_b]
                    i_lb = collider_state.contact_data.link_b[i_c, i_b]
                    key[0], key[1], key[2] = qd.min(i_la, i_lb), qd.max(i_la, i_lb), 0.0
                elif qd.static(order != CONTACT_ORDER.POSITION):
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
            for i_3 in qd.static(range(3)):
                if is_tied:
                    if keys[0, i_3] > keys[1, i_3]:
                        is_after = True
                        is_tied = False
                    elif keys[0, i_3] < keys[1, i_3]:
                        is_tied = False
    if qd.static(order != CONTACT_ORDER.POSITION):
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

    Returns the link pair of links 'i_la' and 'i_lb', lower index first, the geom pair 'i_ga' and 'i_gb', and the frame
    order key 'frame_key'.
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

    Contacts are ordered by link pair, then geom pair, then frame order key (in 'contact_proj_v', the order key of the
    contact position in the frame of its second geom, see func_contact_order_key), then intrinsic order (see
    func_contact_is_after), the lower index going first among identical contacts. The link indices are compared
    themselves, as a key packing them loses exactness and lets distinct pairs interleave in the order of their contact
    indices, which the racy atomic slot reservation of the narrowphase sets. 'key_q' and 'key_p' are the leading keys of
    both contacts (see func_contact_sort_key), and the intrinsic order alone reads the contact data.

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
def func_contact_link_pair(i_b: int, i_c: int, collider_state: array_class.ColliderState):
    """Link pair of a contact of the ordering in contact_sort_idx, lower link index first.

    The contact is the one at position 'i_c' of the ordering of environment 'i_b'.
    """
    i_pc = collider_state.contact_sort_idx[i_c, i_b]
    i_la = collider_state.contact_data.link_a[i_pc, i_b]
    i_lb = collider_state.contact_data.link_b[i_pc, i_b]
    return qd.min(i_la, i_lb), qd.max(i_la, i_lb)


@qd.func
def func_contact_patch_end(
    tid: int,
    i_b: int,
    i_cb_start: int,
    n_con: int,
    collider_state: array_class.ColliderState,
    cos_tol: float,
    use_coop: qd.template(),
    split_by_normal: bool,
):
    """Find the end of the bucket or of the patch that a contact opens, among contacts sorted by link pair.

    The bucket of the contact at 'i_cb_start', among the first 'n_con' contacts, ends at the first contact of another
    link pair. Its patch ('split_by_normal' set) ends there too, or earlier at the first contact whose normal leaves
    the axis of its own by an angle of cosine below 'cos_tol'. Once its bucket is grouped by
    func_contact_group_by_normal, a contact of a later patch agrees with no earlier first contact, or it would have
    joined its patch. 'use_coop' tests 32 contacts at a time on the lanes of a warp, every lane receiving the end, so
    that the walk waits on one read of the contacts per 32 of them.
    """
    n_lanes = qd.static(32 if use_coop else 1)
    i_l_min_start, i_l_max_start = func_contact_link_pair(i_b, i_cb_start, collider_state)
    i_pc_start = collider_state.contact_sort_idx[i_cb_start, i_b]
    normal_start = collider_state.contact_data.normal[i_pc_start, i_b]
    i_cb_end = n_con
    for i_chunk_ in range((n_con - i_cb_start - 1 + n_lanes - 1) // n_lanes):
        i_cb = i_cb_start + 1 + i_chunk_ * n_lanes + tid
        i_cb_end_lane = n_con
        if i_cb < n_con:
            i_l_min, i_l_max = func_contact_link_pair(i_b, i_cb, collider_state)
            is_end = i_l_min != i_l_min_start or i_l_max != i_l_max_start
            if split_by_normal:
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                is_end = is_end or qd.abs(collider_state.contact_data.normal[i_pc, i_b].dot(normal_start)) < cos_tol
            if is_end:
                i_cb_end_lane = i_cb
        i_cb_end = i_cb_end_lane
        if qd.static(use_coop):
            i_cb_end = su.qd_block_min(i_cb_end_lane)
        if i_cb_end < n_con:
            break
    return i_cb_end


@qd.func
def func_contact_group_by_normal(
    tid: int,
    i_b: int,
    i_cb_start: int,
    i_cb_end: int,
    collider_state: array_class.ColliderState,
    cos_tol: float,
    use_coop: qd.template(),
):
    """Group a link-pair bucket of contacts into patches of agreeing normals, keeping their order within each patch.

    A patch holds the contacts whose normals lie along the axis of its first contact within an angle of cosine
    'cos_tol', and the support polygon prunes within a patch only: a contact along another normal transmits a wrench
    the others do not span. The contact of the bucket [i_cb_start, i_cb_end) with the smallest position not grouped yet
    opens the next patch, whatever the number of lanes, and the bucket ends up sorted by patch, each patch running up
    to func_contact_patch_end. 'use_coop' shares the contacts over the 32 lanes of a warp.
    """
    n_lanes = qd.static(32 if use_coop else 1)
    n_cb = i_cb_end - i_cb_start
    # Hull storage holds assignment flags until the grouped permutation is complete
    for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
        i_cb = i_cb_start + i_chunk_ * n_lanes + tid
        if i_cb < i_cb_end:
            collider_state.contact_hull_stack[i_cb, i_b] = 0
    if qd.static(use_coop):
        qd.simt.subgroup.sync()
    i_cb_head = i_cb_start
    i_cb_out = i_cb_start
    for i_patch_ in range(n_cb):
        if i_cb_head >= i_cb_end:
            break
        i_pc_head = collider_state.contact_sort_idx[i_cb_head, i_b]
        normal_ref = collider_state.contact_data.normal[i_pc_head, i_b]
        i_cb_next = i_cb_end
        for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
            i_cb = i_cb_start + i_chunk_ * n_lanes + tid
            i_pc = 0
            is_member = False
            if i_cb < i_cb_end:
                if collider_state.contact_hull_stack[i_cb, i_b] == 0:
                    i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                    normal = collider_state.contact_data.normal[i_pc, i_b]
                    is_member = i_cb == i_cb_head or qd.abs(normal_ref.dot(normal)) >= cos_tol
                    if is_member:
                        collider_state.contact_hull_stack[i_cb, i_b] = 1
                    else:
                        i_cb_next = qd.min(i_cb_next, i_cb)
            i_rank = qd.cast(is_member, gs.qd_int)
            n_chunk = i_rank
            if qd.static(use_coop):
                i_rank, n_chunk = su.qd_block_scan(i_rank)
            if is_member:
                collider_state.contact_lex_idx[i_cb_out + i_rank - 1, i_b] = i_pc
            i_cb_out += n_chunk
        if qd.static(use_coop):
            i_cb_next = su.qd_block_min(i_cb_next)
            qd.simt.subgroup.sync()
        i_cb_head = i_cb_next
    for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
        i_cb = i_cb_start + i_chunk_ * n_lanes + tid
        if i_cb < i_cb_end:
            collider_state.contact_sort_idx[i_cb, i_b] = collider_state.contact_lex_idx[i_cb, i_b]
    if qd.static(use_coop):
        qd.simt.subgroup.sync()


@qd.func
def func_contact_group_patches(
    tid: int,
    i_b: int,
    i_cb_start: int,
    n_con: int,
    collider_state: array_class.ColliderState,
    cos_tol: float,
    use_coop: qd.template(),
):
    """Group into patches of agreeing normals every link-pair bucket of the ordering that holds several of them.

    The buckets cover the positions from 'i_cb_start' up to 'n_con' of the ordering, sorted by link pair. See
    func_contact_group_by_normal for the patches and 'cos_tol'. 'use_coop' shares the walk over the 32 lanes of a warp.
    """
    while i_cb_start < n_con:
        # The first patch of a bucket ends it unless a contact of the same link pair follows, in which case a second
        # walk finds the end of the bucket. One loop runs both, so that the walk is inlined once.
        i_cb_end = i_cb_start
        has_patches = False
        for i_walk in range(2):
            if i_walk == 0 or has_patches:
                i_cb_end = func_contact_patch_end(
                    tid, i_b, i_cb_end, n_con, collider_state, cos_tol, use_coop, split_by_normal=i_walk == 0
                )
                if i_walk == 0 and i_cb_end < n_con:
                    i_l_min_start, i_l_max_start = func_contact_link_pair(i_b, i_cb_start, collider_state)
                    i_l_min_end, i_l_max_end = func_contact_link_pair(i_b, i_cb_end, collider_state)
                    has_patches = i_l_min_end == i_l_min_start and i_l_max_end == i_l_max_start
        if has_patches:
            func_contact_group_by_normal(tid, i_b, i_cb_start, i_cb_end, collider_state, cos_tol, use_coop)
        i_cb_start = i_cb_end


@qd.func
def func_contact_compact(
    tid: int, i_b: int, i_c_start: int, n_con: int, collider_state: array_class.ColliderState, use_coop: qd.template()
):
    """Compact the ordering to the contacts that contact_keep keeps, and return the end of the kept ones.

    The positions from 'i_c_start' to 'n_con' are compacted in place in contact_sort_idx. contact_keep is indexed
    physically, so that the read goes through the ordering. 'use_coop' takes 32 positions at a time on the lanes of a
    warp, each kept contact going to its rank among them, which trails every position not read yet.
    """
    n_lanes = qd.static(32 if use_coop else 1)
    i_c_out = i_c_start
    for i_chunk_ in range((n_con - i_c_start + n_lanes - 1) // n_lanes):
        i_c = i_c_start + i_chunk_ * n_lanes + tid
        i_pc = 0
        is_kept = False
        if i_c < n_con:
            i_pc = collider_state.contact_sort_idx[i_c, i_b]
            is_kept = collider_state.contact_keep[i_pc, i_b] != 0
        i_rank = qd.cast(is_kept, gs.qd_int)
        n_chunk = i_rank
        if qd.static(use_coop):
            i_rank, n_chunk = su.qd_block_scan(i_rank)
        if is_kept:
            collider_state.contact_sort_idx[i_c_out + i_rank - 1, i_b] = i_pc
        i_c_out += n_chunk
    return i_c_out


@qd.func
def func_contact_lanes_argmax(i_value: int, value: float, value_min: float, use_coop: qd.template()):
    """Largest of the values the lanes hold above a floor, and the smallest index holding it, or -1 if none is above.

    Each lane holds the first of its largest values in increasing index order, so that the result is the one of a
    sequential scan over all indices keeping a strictly larger value only. 'use_coop' reduces over the 32 lanes of a
    warp, every lane receiving the result, and keeps the value of the single lane otherwise.
    """
    value_max, i_max = value, i_value
    if qd.static(use_coop):
        value_max, i_max = su.qd_block_argmax(i_value, value)
    if value_max <= value_min:
        i_max = -1
    return value_max, i_max


@qd.func
def func_contact_filter_interior(
    tid: int,
    i_b: int,
    i_cb_start: int,
    i_cb_end: int,
    radius_sqr: float,
    collider_state: array_class.ColliderState,
    eps: float,
    use_coop: qd.template(),
):
    """Gather the candidate vertices of the support polygon of a projected patch, rejecting points strictly inside it.

    A patch of more than 32 contacts drops the points strictly inside the quadrilateral of its extreme points along the
    two diagonals of the projection plane, which no support polygon of the patch can hold. 'radius_sqr' scales the
    rounding margin of that test. The bucket-logical indices of the remaining contacts are written to contact_lex_idx
    from 'i_cb_start' on, in order, and their count is returned. 'use_coop' shares the work over the 32 lanes of a warp.
    """
    n_lanes = qd.static(32 if use_coop else 1)
    n_cb = i_cb_end - i_cb_start
    if qd.static(use_coop):
        qd.simt.subgroup.sync()
    extrema = qd.Matrix.zero(gs.qd_float, 4, 2)
    if n_cb > 32:
        # Strictly interior points of an inscribed polygon cannot support the full hull
        for i_extreme in range(4):
            direction_u = 1.0
            direction_v = 1.0
            if i_extreme == 1 or i_extreme == 2:
                direction_u = -1.0
            if i_extreme >= 2:
                direction_v = -1.0
            score_max = -1e30
            i_cb_extreme = i_cb_start
            for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
                i_cb = i_cb_start + i_chunk_ * n_lanes + tid
                if i_cb < i_cb_end:
                    score = direction_u * collider_state.contact_sort_key[i_cb, i_b]
                    score += direction_v * collider_state.contact_proj_v[i_cb, i_b]
                    if score > score_max:
                        score_max = score
                        i_cb_extreme = i_cb
            score_max, i_cb_extreme = func_contact_lanes_argmax(i_cb_extreme, score_max, -1e30, use_coop)
            extrema[i_extreme, 0] = collider_state.contact_sort_key[i_cb_extreme, i_b]
            extrema[i_extreme, 1] = collider_state.contact_proj_v[i_cb_extreme, i_b]
    n_boundary = 0
    for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
        i_cb = i_cb_start + i_chunk_ * n_lanes + tid
        is_boundary = i_cb < i_cb_end
        if is_boundary and n_cb > 32:
            point_u = collider_state.contact_sort_key[i_cb, i_b]
            point_v = collider_state.contact_proj_v[i_cb, i_b]
            is_interior = True
            for i_edge in range(4):
                i_edge_next = (i_edge + 1) % 4
                edge_u = extrema[i_edge_next, 0] - extrema[i_edge, 0]
                edge_v = extrema[i_edge_next, 1] - extrema[i_edge, 1]
                cross_lhs = edge_u * (point_v - extrema[i_edge, 1])
                cross_rhs = edge_v * (point_u - extrema[i_edge, 0])
                rounding = 8 * eps * (qd.abs(cross_lhs) + qd.abs(cross_rhs) + radius_sqr)
                is_interior = is_interior and cross_lhs - cross_rhs > rounding
            is_boundary = not is_interior
        i_rank = qd.cast(is_boundary, gs.qd_int)
        n_chunk = i_rank
        if qd.static(use_coop):
            i_rank, n_chunk = su.qd_block_scan(i_rank)
        if is_boundary:
            collider_state.contact_lex_idx[i_cb_start + n_boundary + i_rank - 1, i_b] = i_cb
        n_boundary += n_chunk
    if qd.static(use_coop):
        qd.simt.subgroup.sync()
    return n_boundary


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
def func_contact_restore_depth(
    tid: int,
    i_b: int,
    i_cb_start: int,
    i_cb_end: int,
    i_cb_u: int,
    i_ca: int,
    i_ch: int,
    n_hull: int,
    penetration_min: float,
    penetration_max: float,
    penetration_hull: float,
    collider_state: array_class.ColliderState,
    prune_deep_penetration_ratio: float,
    tol: float,
    eps: float,
    use_coop: qd.template(),
    is_constant_depth: bool,
):
    """Keep the contacts of a patch deeper than the depth its retained support polygon represents.

    The patch holds the contacts at positions [i_cb_start, i_cb_end) of the ordering, its support polygon the 'n_hull'
    vertices of contact_hull_stack from 'i_cb_start' on, the deepest of them 'i_ch' with penetration 'penetration_hull'.
    A contact deeper than 'prune_deep_penetration_ratio' times 'penetration_hull' is kept. Unless the depth range from
    'penetration_min' to 'penetration_max' lies within 'tol' of the smallest penetration ('is_constant_depth'), the
    penetration is checked for an affine function of the position, fitted on the first contact, the contact 'i_cb_u'
    farthest from it in its plane (-1 if all coincide) and the contact farthest from the line through both. Otherwise
    the deepest contact 'i_ca' is kept, or 'i_ch' when within tolerance of it, and every other contact deeper than the
    penetration interpolated linearly over the fan of triangles joining that apex to the edges of the polygon, within
    the relative tolerance 'tol'. 'eps' bounds the relative rounding error of the float type. 'use_coop' shares the
    contacts over the 32 lanes of a warp.
    """
    n_lanes = qd.static(32 if use_coop else 1)
    n_cb = i_cb_end - i_cb_start
    # Every contact lies below the deepest vertex within the tolerance when the depth range does
    if not is_constant_depth or prune_deep_penetration_ratio < 1.0 + tol:
        for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
            i_cb = i_cb_start + i_chunk_ * n_lanes + tid
            if i_cb < i_cb_end:
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                if collider_state.contact_data.penetration[i_pc, i_b] > prune_deep_penetration_ratio * penetration_hull:
                    collider_state.contact_keep[i_pc, i_b] = 1

    is_affine = is_constant_depth
    if not is_constant_depth:
        i_pc0 = collider_state.contact_sort_idx[i_cb_start, i_b]
        normal = collider_state.contact_data.normal[i_pc0, i_b].normalized()
        position_ref = collider_state.contact_data.pos[i_pc0, i_b]
        penetration_ref = collider_state.contact_data.penetration[i_pc0, i_b]
        is_affine = True
        length_sqr = gs.qd_float(0.0)
        direction_u = qd.Vector.zero(gs.qd_float, 3)
        gradient = qd.Vector.zero(gs.qd_float, 3)
        if i_cb_u >= 0:
            i_pu = collider_state.contact_sort_idx[i_cb_u, i_b]
            direction_u = collider_state.contact_data.pos[i_pu, i_b] - position_ref
            direction_u -= direction_u.dot(normal) * normal
            length_sqr = direction_u.norm_sqr()
            direction_u /= qd.sqrt(length_sqr)
            penetration_u = collider_state.contact_data.penetration[i_pu, i_b] - penetration_ref
            gradient = direction_u * (penetration_u / qd.sqrt(length_sqr))
        width_sqr = gs.qd_float(0.0)
        i_cb_v = -1
        for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
            i_cb = i_cb_start + i_chunk_ * n_lanes + tid
            if i_cb < i_cb_end:
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                delta = collider_state.contact_data.pos[i_pc, i_b] - position_ref
                delta -= delta.dot(normal) * normal
                offset = delta - delta.dot(direction_u) * direction_u
                if offset.norm_sqr() > width_sqr:
                    width_sqr = offset.norm_sqr()
                    i_cb_v = i_cb
        width_sqr, i_cb_v = func_contact_lanes_argmax(i_cb_v, width_sqr, 0.0, use_coop)
        if i_cb_v >= 0 and width_sqr > eps * length_sqr:
            i_pv = collider_state.contact_sort_idx[i_cb_v, i_b]
            delta = collider_state.contact_data.pos[i_pv, i_b] - position_ref
            delta -= delta.dot(normal) * normal
            direction_v = delta - delta.dot(direction_u) * direction_u
            penetration_v = collider_state.contact_data.penetration[i_pv, i_b] - penetration_ref
            gradient += direction_v * ((penetration_v - delta.dot(gradient)) / width_sqr)
        for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
            i_cb = i_cb_start + i_chunk_ * n_lanes + tid
            if i_cb < i_cb_end:
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                position = collider_state.contact_data.pos[i_pc, i_b]
                penetration = collider_state.contact_data.penetration[i_pc, i_b]
                prediction = penetration_ref + (position - position_ref).dot(gradient)
                rounding = eps * (position.norm() + position_ref.norm() + penetration) * (1 + gradient.norm())
                if qd.abs(penetration - prediction) > tol * qd.max(penetration, penetration_ref) + rounding:
                    is_affine = False
        if qd.static(use_coop):
            is_affine = su.qd_block_all(is_affine)

    if not is_affine:
        i_pa = collider_state.contact_sort_idx[i_ca, i_b]
        i_ph = collider_state.contact_sort_idx[i_ch, i_b]
        penetration_apex = collider_state.contact_data.penetration[i_pa, i_b]
        penetration_vertex = collider_state.contact_data.penetration[i_ph, i_b]
        rounding = eps * (2 * collider_state.contact_data.pos[i_pa, i_b].norm() + penetration_apex)
        if penetration_apex <= penetration_vertex + tol * penetration_apex + rounding:
            i_ca = i_ch
            penetration_apex = penetration_vertex
            i_pa = i_ph
        if tid == 0:
            collider_state.contact_keep[i_pa, i_b] = 1
        if qd.static(use_coop):
            qd.simt.subgroup.sync()
        position_a = qd.Vector([collider_state.contact_sort_key[i_ca, i_b], collider_state.contact_proj_v[i_ca, i_b]])
        for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
            i_cb = i_cb_start + i_chunk_ * n_lanes + tid
            i_pc = 0
            is_kept = True
            if i_cb < i_cb_end:
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                is_kept = collider_state.contact_keep[i_pc, i_b] != 0
            if not is_kept:
                offset = (
                    qd.Vector([collider_state.contact_sort_key[i_cb, i_b], collider_state.contact_proj_v[i_cb, i_b]])
                    - position_a
                )
                prediction = gs.qd_float(-1.0)
                if n_hull == 1:
                    prediction = penetration_apex
                for i_h in range(n_hull):
                    i_cu = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
                    i_cv = collider_state.contact_hull_stack[i_cb_start + (i_h + 1) % n_hull, i_b]
                    i_pu = collider_state.contact_sort_idx[i_cu, i_b]
                    i_pv = collider_state.contact_sort_idx[i_cv, i_b]
                    direction_u = (
                        func_contact_hull_vertex_pos(i_b, i_cb_start, i_h, n_hull, collider_state) - position_a
                    )
                    direction_v = (
                        func_contact_hull_vertex_pos(i_b, i_cb_start, i_h + 1, n_hull, collider_state) - position_a
                    )
                    if n_hull == 2:
                        length_sqr = direction_u.norm_sqr()
                        if length_sqr > 0.0:
                            weight = offset.dot(direction_u) / length_sqr
                            if weight >= -eps and weight <= 1 + eps:
                                weight = qd.min(1.0, qd.max(0.0, weight))
                                depth = (1 - weight) * penetration_apex
                                depth += weight * collider_state.contact_data.penetration[i_pu, i_b]
                                prediction = qd.max(prediction, depth)
                    else:
                        determinant = direction_u[0] * direction_v[1] - direction_u[1] * direction_v[0]
                        rounding_area = eps * qd.sqrt(direction_u.norm_sqr() * direction_v.norm_sqr())
                        if determinant > rounding_area:
                            weight_u = (offset[0] * direction_v[1] - offset[1] * direction_v[0]) / determinant
                            weight_v = (direction_u[0] * offset[1] - direction_u[1] * offset[0]) / determinant
                            if weight_u >= -eps and weight_v >= -eps and weight_u + weight_v <= 1 + eps:
                                weight_u = qd.max(0.0, weight_u)
                                weight_v = qd.max(0.0, weight_v)
                                weight_a = qd.max(0.0, 1 - weight_u - weight_v)
                                depth = weight_a * penetration_apex
                                depth += weight_u * collider_state.contact_data.penetration[i_pu, i_b]
                                depth += weight_v * collider_state.contact_data.penetration[i_pv, i_b]
                                depth /= weight_a + weight_u + weight_v
                                prediction = qd.max(prediction, depth)
                penetration = collider_state.contact_data.penetration[i_pc, i_b]
                rounding = eps * (2 * collider_state.contact_data.pos[i_pc, i_b].norm() + penetration + offset.norm())
                if penetration > prediction + tol * penetration + rounding:
                    collider_state.contact_keep[i_pc, i_b] = 1


@qd.func
def func_contact_support_hull(
    i_b: int,
    i_cb_start: int,
    i_cb_end: int,
    collider_state: array_class.ColliderState,
    tol: float,
    rounding: float,
    eps: float,
):
    """Find the support polygon of a bucket of coplanar contacts, as the vertices of their convex hull in the plane.

    The contacts of the bucket, identified by their bucket-logical indices in [i_cb_start, i_cb_end), are projected
    beforehand on the plane, with their coordinates in 'contact_sort_key' and 'contact_proj_v', and sorted
    lexicographically in 'contact_lex_idx' (see func_contact_sort_positions). Andrew's monotone chain keeps the vertices
    of their exact convex hull. A single pass along the hull then drops some of the vertices whose removal shrinks the
    hull by less than the fraction 'tol' of its area, the points of its edges among them. The hull visits the points of
    an edge in their order along it, while the sort does not as soon as the edge is nearly parallel to the first sort
    axis, so a tolerance on the turns of the chain itself would drop a corner in place of a point of the edge. Every
    step is linear in the number of contacts.

    'rounding' is the length within which points of the bucket count as lying on one line, relative to the size of the
    bucket so that the outcome does not depend on where it stands. 'eps' bounds the relative rounding error of the float
    type (see gs.EPS).

    Return the simplified hull size and the deepest original corner's bucket-logical index (-1 if absent). Hull
    vertices are stored in 'contact_hull_stack' from 'i_cb_start' on.
    """
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
    # Near-equal area losses use the projected positions to keep corner selection stable under positional rounding
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
                    # Removal eligibility takes precedence over positional tie-breaking
                    is_tie = qd.abs(losses[1] - losses[i_k]) <= area_rounding
                    is_tie = is_tie and ((losses[1] <= area_tol) == (losses[i_k] <= area_tol))
                    delta_u = poss[1, 0] - poss[i_k, 0]
                    # Resolve positional rounding before comparing symmetric corners' secondary coordinates
                    is_before = delta_u < -rounding or (
                        qd.abs(delta_u) <= rounding
                        and (poss[1, 1] < poss[i_k, 1] or (poss[1, 1] <= poss[i_k, 1] and delta_u < 0.0))
                    )
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
def func_contact_patch_is_prunable(
    i_b: int,
    i_cb_start: int,
    i_cb_end: int,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
):
    """Whether a patch of contacts is reduced to its support polygon.

    The patch holds the contacts at positions [i_cb_start, i_cb_end) of the ordering. A patch needs 3 contacts to hold a
    removable one. Under contact patch detection, the manifold of a single convex geom pair is already bounded by its
    detection budget, so that only patches spanning several geom pairs are reduced.
    """
    n_cb = i_cb_end - i_cb_start
    is_prunable = n_cb >= 3
    if qd.static(gjk_static_config.enable_contact_patch):
        if is_prunable and n_cb <= collider_static_config.n_contacts_per_convex_pair:
            i_pc0 = collider_state.contact_sort_idx[i_cb_start, i_b]
            i_ga = collider_state.contact_data.geom_a[i_pc0, i_b]
            i_gb = collider_state.contact_data.geom_b[i_pc0, i_b]
            pair_ref = qd.Vector([qd.min(i_ga, i_gb), qd.max(i_ga, i_gb)])
            is_single_convex_pair = dyn_info.geoms.is_convex[i_ga] and dyn_info.geoms.is_convex[i_gb]
            for i_cb in range(i_cb_start + 1, i_cb_end):
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                i_ga = collider_state.contact_data.geom_a[i_pc, i_b]
                i_gb = collider_state.contact_data.geom_b[i_pc, i_b]
                if qd.min(i_ga, i_gb) != pair_ref[0] or qd.max(i_ga, i_gb) != pair_ref[1]:
                    is_single_convex_pair = False
            is_prunable = not is_single_convex_pair
    return is_prunable


@qd.func
def func_contact_patch_project(
    tid: int,
    i_b: int,
    i_cb_start: int,
    i_cb_end: int,
    collider_state: array_class.ColliderState,
    tol: float,
    eps: float,
    use_coop: qd.template(),
):
    """Check whether a patch of contacts lies in one plane, and if so project its contacts onto it and drop them all.

    The patch holds the contacts at positions [i_cb_start, i_cb_end) of the ordering. Its plane is normal to the mean
    of their normals, folded to the hemisphere of the first one, and passes through their centroid. The patch lies in
    it when its positions deviate from it by at most 'tol' times its in-plane radius, either surface of the contact
    being allowed to tilt (see the witnesses below). The coordinates of the contacts of a coplanar patch in the plane,
    relative to the centroid so that their rounding scales with the patch, are written to contact_sort_key and
    contact_proj_v at their position in the ordering, and every contact is dropped in contact_keep until its support
    polygon keeps it. 'eps' bounds the relative rounding error of the float type. 'use_coop' shares the contacts over
    the 32 lanes of a warp, every lane receiving the results.

    Returns whether the patch is coplanar, whether its depth range lies within 'tol' of its smallest penetration, its
    squared in-plane radius, its smallest and largest penetrations, the contact farthest from the first one in the plane
    of the normal of the first one (-1 if all coincide), and the deepest contact, the first in order among equals.
    """
    n_lanes = qd.static(32 if use_coop else 1)
    n_cb = i_cb_end - i_cb_start
    i_pc0 = collider_state.contact_sort_idx[i_cb_start, i_b]
    normal_ref = collider_state.contact_data.normal[i_pc0, i_b]
    mean_normal = qd.Vector.zero(gs.qd_float, 3)
    centroid = qd.Vector.zero(gs.qd_float, 3)
    for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
        i_cb = i_cb_start + i_chunk_ * n_lanes + tid
        if i_cb < i_cb_end:
            i_pc = collider_state.contact_sort_idx[i_cb, i_b]
            normal = collider_state.contact_data.normal[i_pc, i_b]
            sign = gs.qd_float(1.0)
            if normal_ref.dot(normal) < 0.0:
                sign = -1.0
            mean_normal += sign * normal
            centroid += collider_state.contact_data.pos[i_pc, i_b]
    if qd.static(use_coop):
        # Every lane holds the sums the first lane does (see qd_block_sum in utils/simt.py), so that the lanes agree bit
        # for bit on every decision taken from them
        for k in qd.static(range(3)):
            mean_normal[k] = su.qd_block_sum(mean_normal[k])
            centroid[k] = su.qd_block_sum(centroid[k])
    centroid *= 1.0 / n_cb
    mean_normal_norm = mean_normal.norm()

    position_ref = collider_state.contact_data.pos[i_pc0, i_b]
    penetration_ref = collider_state.contact_data.penetration[i_pc0, i_b]
    penetration_min = penetration_ref
    penetration_max = penetration_ref
    max_in_plane_r2 = gs.qd_float(0.0)
    i_cb_u = -1
    i_ca = -1
    is_constant_depth = False
    is_coplanar = mean_normal_norm > eps
    if is_coplanar:
        mean_normal /= mean_normal_norm
        normal_unit = normal_ref.normalized()
        # A contact lies in the plane of the patch when either the midpoint of its witnesses or one of them does, so
        # that a tilted patch whose midpoint depth varies along one supporting surface still counts as planar. No
        # per-contact normal check applies: a contact whose normal is diagonal (an edge against an edge at a corner of
        # the patch) is still a vertex of its support polygon by its position.
        offset_ref = 0.5 * penetration_ref * normal_ref.dot(mean_normal)
        max_depth = gs.qd_float(0.0)
        witness_error_a = gs.qd_float(0.0)
        witness_error_b = gs.qd_float(0.0)
        length_sqr_u = gs.qd_float(0.0)
        penetration_apex = gs.qd_float(-1.0)
        for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
            i_cb = i_cb_start + i_chunk_ * n_lanes + tid
            if i_cb < i_cb_end:
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                pos = collider_state.contact_data.pos[i_pc, i_b]
                penetration = collider_state.contact_data.penetration[i_pc, i_b]
                delta = pos - centroid
                depth = qd.abs(delta.dot(mean_normal))
                max_depth = qd.max(max_depth, depth)
                normal = collider_state.contact_data.normal[i_pc, i_b]
                offset = 0.5 * penetration * qd.abs(normal.dot(mean_normal)) - offset_ref
                height = (pos - position_ref).dot(mean_normal)
                witness_error_a = qd.max(witness_error_a, qd.abs(height - offset))
                witness_error_b = qd.max(witness_error_b, qd.abs(height + offset))
                max_in_plane_r2 = qd.max(max_in_plane_r2, delta.norm_sqr() - depth * depth)
                penetration_min = qd.min(penetration_min, penetration)
                penetration_max = qd.max(penetration_max, penetration)
                delta_u = pos - position_ref
                delta_u -= delta_u.dot(normal_unit) * normal_unit
                if delta_u.norm_sqr() > length_sqr_u:
                    length_sqr_u = delta_u.norm_sqr()
                    i_cb_u = i_cb
                if penetration > penetration_apex:
                    penetration_apex = penetration
                    i_ca = i_cb
        if qd.static(use_coop):
            max_depth = su.qd_block_max(max_depth)
            witness_error_a = su.qd_block_max(witness_error_a)
            witness_error_b = su.qd_block_max(witness_error_b)
            max_in_plane_r2 = su.qd_block_max(max_in_plane_r2)
            penetration_min = su.qd_block_min(penetration_min)
            penetration_max = su.qd_block_max(penetration_max)
        length_sqr_u, i_cb_u = func_contact_lanes_argmax(i_cb_u, length_sqr_u, 0.0, use_coop)
        penetration_apex, i_ca = func_contact_lanes_argmax(i_ca, penetration_apex, -1.0, use_coop)
        is_constant_depth = penetration_max - penetration_min <= tol * penetration_min
        if qd.min(max_depth, witness_error_a, witness_error_b) > tol * qd.sqrt(max_in_plane_r2):
            is_coplanar = False

    if is_coplanar:
        # The first axis of the plane is the world axis least aligned with its normal, projected onto it, so that every
        # lane holding the same normal gets the same basis
        normal_abs = qd.abs(mean_normal)
        axis_u = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        if normal_abs[1] < normal_abs[0] and normal_abs[1] < normal_abs[2]:
            axis_u = qd.Vector([0.0, 1.0, 0.0])
        elif normal_abs[2] < normal_abs[0] and normal_abs[2] <= normal_abs[1]:
            axis_u = qd.Vector([0.0, 0.0, 1.0])
        axis_u -= axis_u.dot(mean_normal) * mean_normal
        axis_u /= axis_u.norm()
        axis_v = mean_normal.cross(axis_u)
        for i_chunk_ in range((n_cb + n_lanes - 1) // n_lanes):
            i_cb = i_cb_start + i_chunk_ * n_lanes + tid
            if i_cb < i_cb_end:
                i_pc = collider_state.contact_sort_idx[i_cb, i_b]
                delta = collider_state.contact_data.pos[i_pc, i_b] - centroid
                collider_state.contact_sort_key[i_cb, i_b] = delta.dot(axis_u)
                collider_state.contact_proj_v[i_cb, i_b] = delta.dot(axis_v)
                collider_state.contact_keep[i_pc, i_b] = 0
    if i_ca < 0:
        i_ca = i_cb_start
    return is_coplanar, is_constant_depth, max_in_plane_r2, penetration_min, penetration_max, i_cb_u, i_ca


@qd.func
def func_contact_sort_positions(
    tid: int, i_b: int, i_cb_start: int, n_cb: int, collider_state: array_class.ColliderState, use_coop: qd.template()
):
    """Sort the candidate vertices of the support polygon of a patch by their projected positions, lexicographically.

    The 'n_cb' positions held by contact_lex_idx from 'i_cb_start' on are sorted by the coordinates at those positions,
    in contact_sort_key then contact_proj_v, ties broken by position. 'use_coop' runs a bitonic network over the 32
    lanes of a warp (see qd_bitonic_pair in utils/simt.py), and a heapsort on a single thread otherwise.
    """
    if qd.static(use_coop):
        n_levels = su.qd_bitonic_n_levels(n_cb)
        for i_level in range(n_levels):
            for i_step in range(i_level + 1):
                for i_chunk_ in range(((1 << (n_levels - 1)) + 31) // 32):
                    i_pair = i_chunk_ * 32 + tid
                    i_s, i_t = su.qd_bitonic_pair(i_pair, i_level, i_step)
                    if i_t < n_cb:
                        i_p = collider_state.contact_lex_idx[i_cb_start + i_s, i_b]
                        i_q = collider_state.contact_lex_idx[i_cb_start + i_t, i_b]
                        pos_u_p = collider_state.contact_sort_key[i_p, i_b]
                        pos_v_p = collider_state.contact_proj_v[i_p, i_b]
                        pos_u_q = collider_state.contact_sort_key[i_q, i_b]
                        pos_v_q = collider_state.contact_proj_v[i_q, i_b]
                        if pos_u_q < pos_u_p or (
                            pos_u_q <= pos_u_p and (pos_v_q < pos_v_p or (pos_v_q <= pos_v_p and i_q < i_p))
                        ):
                            collider_state.contact_lex_idx[i_cb_start + i_s, i_b] = i_q
                            collider_state.contact_lex_idx[i_cb_start + i_t, i_b] = i_p
                qd.simt.subgroup.sync()
    else:
        func_contact_heapsort(i_b, i_cb_start, i_cb_start + n_cb, collider_state, CONTACT_ORDER.POSITION)


@qd.func
def func_contact_mark_hull(
    i_b: int,
    i_cb_start: int,
    i_cb_deepest: int,
    n_hull: int,
    max_in_plane_r2: float,
    collider_state: array_class.ColliderState,
    tol: float,
    eps: float,
):
    """Keep the vertices of the support polygon of a patch, and its deepest corner when deeper than every vertex.

    The 'n_hull' vertices are read from contact_hull_stack from 'i_cb_start' on (see func_contact_support_hull), and
    'i_cb_deepest' is the deepest corner of the polygon before its simplification (-1 if absent). The corner is kept
    when its penetration exceeds that of every vertex by the relative tolerance 'tol', the rounding being bounded from
    the position of the corner and the squared in-plane radius 'max_in_plane_r2' of the patch. 'eps' bounds the
    relative rounding error of the float type.

    Returns the largest penetration of the vertices floored at zero, and the deepest vertex, the first in polygon order
    among equals.
    """
    penetration_hull = gs.qd_float(0.0)
    penetration_vertex = gs.qd_float(-1.0)
    i_ch = i_cb_start
    for i_h in range(n_hull):
        i_cb = collider_state.contact_hull_stack[i_cb_start + i_h, i_b]
        i_pc = collider_state.contact_sort_idx[i_cb, i_b]
        collider_state.contact_keep[i_pc, i_b] = 1
        penetration = collider_state.contact_data.penetration[i_pc, i_b]
        penetration_hull = qd.max(penetration_hull, penetration)
        if penetration > penetration_vertex:
            penetration_vertex = penetration
            i_ch = i_cb
    if i_cb_deepest >= 0:
        i_pc = collider_state.contact_sort_idx[i_cb_deepest, i_b]
        penetration = collider_state.contact_data.penetration[i_pc, i_b]
        pos = collider_state.contact_data.pos[i_pc, i_b]
        rounding = eps * (2.0 * pos.norm() + penetration + 2.0 * qd.sqrt(max_in_plane_r2))
        if penetration > penetration_hull + tol * penetration + rounding:
            collider_state.contact_keep[i_pc, i_b] = 1
    return penetration_hull, i_ch


@qd.func
def func_contact_reduce_patches(
    tid: int,
    i_b: int,
    i_cb_start: int,
    n_con: int,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    tol: float,
    cos_tol: float,
    prune_deep_penetration_ratio: float,
    eps: float,
    use_coop: qd.template(),
):
    """Reduce every prunable patch of the ordering to its support polygon and the contacts deeper than it represents.

    The patches cover the positions from 'i_cb_start' up to 'n_con' of the ordering, grouped by
    func_contact_group_patches, their normals lying along the axis of their first contact within an angle of cosine
    'cos_tol'. A coplanar patch keeps the vertices of its support polygon and the contacts func_contact_restore_depth
    restores for the relative tolerance 'tol' and the ratio 'prune_deep_penetration_ratio', its other contacts being
    marked removed in contact_keep. 'eps' bounds the relative rounding error of the float type. 'use_coop' shares each
    patch over the 32 lanes of a warp, the first lane building the support polygon.
    """
    while i_cb_start < n_con:
        i_cb_end = func_contact_patch_end(
            tid, i_b, i_cb_start, n_con, collider_state, cos_tol, use_coop, split_by_normal=True
        )
        is_prunable = func_contact_patch_is_prunable(
            i_b, i_cb_start, i_cb_end, collider_state, dyn_info, collider_static_config, gjk_static_config
        )
        if is_prunable:
            (
                is_coplanar,
                is_constant_depth,
                max_in_plane_r2,
                penetration_min,
                penetration_max,
                i_cb_u,
                i_ca,
            ) = func_contact_patch_project(tid, i_b, i_cb_start, i_cb_end, collider_state, tol, eps, use_coop)
            if is_coplanar:
                n_cb = func_contact_filter_interior(
                    tid, i_b, i_cb_start, i_cb_end, max_in_plane_r2, collider_state, eps, use_coop
                )
                func_contact_sort_positions(tid, i_b, i_cb_start, n_cb, collider_state, use_coop)
                # The support polygon is a serial chain of reads, which the first lane walks
                n_hull, i_ch = 0, 0
                penetration_hull = gs.qd_float(0.0)
                if tid == 0:
                    n_hull, i_cb_deepest = func_contact_support_hull(
                        i_b,
                        i_cb_start,
                        i_cb_start + n_cb,
                        collider_state,
                        tol,
                        rounding=qd.sqrt(eps * max_in_plane_r2),
                        eps=eps,
                    )
                    penetration_hull, i_ch = func_contact_mark_hull(
                        i_b, i_cb_start, i_cb_deepest, n_hull, max_in_plane_r2, collider_state, tol, eps
                    )
                if qd.static(use_coop):
                    n_hull = su.qd_block_shuffle(n_hull, 0)
                    i_ch = su.qd_block_shuffle(i_ch, 0)
                    penetration_hull = su.qd_block_shuffle(penetration_hull, 0)
                    qd.simt.subgroup.sync()
                func_contact_restore_depth(
                    tid,
                    i_b,
                    i_cb_start,
                    i_cb_end,
                    i_cb_u,
                    i_ca,
                    i_ch,
                    n_hull,
                    penetration_min,
                    penetration_max,
                    penetration_hull,
                    collider_state,
                    prune_deep_penetration_ratio,
                    tol,
                    eps,
                    use_coop,
                    is_constant_depth=is_constant_depth,
                )
        if qd.static(use_coop):
            qd.simt.subgroup.sync()
        i_cb_start = i_cb_end


@qd.func
def func_clamp_prune_contacts(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    errno: qd.Tensor,
):
    """Reduce contact support polygons within link pairs and enforce contact-buffer limits.

    Sorts live contacts by link pair and geometry-relative contact data, then groups compatible normals and reduces
    approximately coplanar patches. Surviving contacts retain their normals. Depth restoration keeps supports whose
    penetration exceeds the depth represented by the retained polygon within the pruning tolerance.

    Compacts the logical-to-physical permutation in contact_sort_idx and updates n_contacts. Hibernated contacts keep
    their leading slots, and contact_data keeps its physical layout. Pruning runs in eligible scenes with positive
    tolerance outside the differentiable path. Contact-buffer overflow sets errno for the caller to report.
    """
    _B = collider_state.n_contacts.shape[0]
    qd.loop_config(name="clamp_prune_contacts", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        # The scalars are read inside the loop, which spares the launch of a serial kernel reading them beforehand
        max_contacts = collider_info.max_contacts[None]
        tol = collider_info.contact_pruning_tolerance[None]
        # The normals of a patch lie along the axis of its first contact within the angle 'tol' (see
        # func_contact_group_by_normal)
        cos_tol = 1.0 - 0.5 * tol * tol
        prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
        EPS = rigid_info.EPS[None]

        n_con = qd.min(collider_state.n_contacts[i_b], collider_info.max_candidate_contacts[None])
        collider_state.n_contacts[i_b] = n_con
        # The kept contacts of the sleepers lead the buffer in the order func_collider_clear_env gave them, and the
        # prune runs on the live contacts after them (see n_contacts_hibernated in array_class.py)
        n_hib = collider_state.n_contacts_hibernated[i_b]

        # Identity permutation of the live contacts. Required so downstream consumers can always indirect through
        # contact_sort_idx, even when pruning is inactive.
        for i_c in range(n_hib, n_con):
            collider_state.contact_sort_idx[i_c, i_b] = i_c

        # The pruning is compiled in where some link pair can hold 3 contacts (see has_prunable_contacts in
        # collider.py), outside the differentiable path, and runs for a positive tolerance
        if qd.static(collider_static_config.has_prunable_contacts and not rigid_config.requires_grad):
            if n_con - n_hib >= 3 and tol > 0.0:
                # Phase 1: one sort groups the contacts by link pair and orders them deterministically within each pair
                # (see func_contact_is_after), then the buckets of several patches are grouped by normal
                for i_c in range(n_hib, n_con):
                    i_gb = collider_state.contact_data.geom_b[i_c, i_b]
                    pos = collider_state.contact_data.pos[i_c, i_b] - dyn_state.geoms.pos[i_gb, i_b]
                    pos_local = gu.qd_inv_transform_by_quat(pos, dyn_state.geoms.quat[i_gb, i_b])
                    collider_state.contact_proj_v[i_c, i_b] = func_contact_order_key(pos_local)
                func_contact_heapsort(i_b, n_hib, n_con, collider_state, CONTACT_ORDER.LINK_PAIR)
                func_contact_group_patches(0, i_b, n_hib, n_con, collider_state, cos_tol, use_coop=False)

                # Every contact is kept unless the support polygon of its patch drops it
                for i_c in range(n_con):
                    collider_state.contact_keep[i_c, i_b] = 1

                # Phase 2: reduce the patches of each link pair (logical-contiguous after the sorts above)
                func_contact_reduce_patches(
                    0,
                    i_b,
                    n_hib,
                    n_con,
                    collider_state,
                    dyn_info,
                    collider_static_config,
                    gjk_static_config,
                    tol,
                    cos_tol,
                    prune_deep_penetration_ratio,
                    EPS,
                    use_coop=False,
                )

                # Phase 3: compaction of contact_sort_idx
                collider_state.n_contacts[i_b] = func_contact_compact(
                    0, i_b, n_hib, n_con, collider_state, use_coop=False
                )

        # The contact constraint buffers are sized to 4 * max_contacts, so any surviving contact beyond that budget
        # would write out of bounds. Clamp and flag the env: check_errno halts the simulation with a request to
        # increase 'max_contacts'.
        if collider_state.n_contacts[i_b] > max_contacts:
            collider_state.n_contacts[i_b] = max_contacts
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CONTACTS


@qd.func
def func_contact_sort_tile_coop(
    i_thread: int,
    i_b: int,
    i_tile_start: int,
    n_sort: int,
    i_s_start: int,
    sh_sort_geoms: qd.template(),
    sh_sort_v: qd.template(),
    sh_sort_idx: qd.template(),
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    n_threads: qd.template(),
):
    """Sort a tile of contacts in the shared memory of a block, in the order of phase 1 of the contact pruning.

    The 'n_sort' contacts from 'i_tile_start' on, in their physical order, get their frame order key written to
    contact_proj_v and staged with their geom pair in the shared arrays from 'i_s_start' on, where the 'n_threads'
    threads sort them by a bitonic network (see qd_bitonic_pair in utils/simt.py). Padding slots up to a power of two
    would sort after every contact and never move, so that the pairs reaching them are skipped. The intrinsic order of
    tied keys reads the contact data still. The sorted contacts are written to contact_sort_idx from 'i_tile_start' on.
    The threads synchronize as the lanes of a single warp when 'n_threads' is 32, and as a whole block otherwise.
    """
    for i_chunk_ in range((n_sort + n_threads - 1) // n_threads):
        i_s = i_chunk_ * n_threads + i_thread
        if i_s < n_sort:
            i_c = i_tile_start + i_s
            i_gb = collider_state.contact_data.geom_b[i_c, i_b]
            pos = collider_state.contact_data.pos[i_c, i_b] - dyn_state.geoms.pos[i_gb, i_b]
            frame_key = func_contact_order_key(gu.qd_inv_transform_by_quat(pos, dyn_state.geoms.quat[i_gb, i_b]))
            collider_state.contact_proj_v[i_c, i_b] = frame_key
            sh_sort_geoms[i_s_start + i_s, 0] = collider_state.contact_data.geom_a[i_c, i_b]
            sh_sort_geoms[i_s_start + i_s, 1] = i_gb
            sh_sort_v[i_s_start + i_s] = frame_key
            sh_sort_idx[i_s_start + i_s] = i_c
    if qd.static(n_threads == 32):
        qd.simt.subgroup.sync()
    else:
        qd.simt.block.sync()
    n_levels = su.qd_bitonic_n_levels(n_sort)
    for i_level in range(n_levels):
        for i_step in range(i_level + 1):
            for i_chunk_ in range(((1 << (n_levels - 1)) + n_threads - 1) // n_threads):
                i_pair = i_chunk_ * n_threads + i_thread
                i_s, i_t = su.qd_bitonic_pair(i_pair, i_level, i_step)
                if i_t < n_sort:
                    i_p = sh_sort_idx[i_s_start + i_s]
                    i_q = sh_sort_idx[i_s_start + i_t]
                    i_gpa, i_gpb = sh_sort_geoms[i_s_start + i_s, 0], sh_sort_geoms[i_s_start + i_s, 1]
                    i_gqa, i_gqb = sh_sort_geoms[i_s_start + i_t, 0], sh_sort_geoms[i_s_start + i_t, 1]
                    i_lpa, i_lpb = dyn_info.geoms.link_idx[i_gpa], dyn_info.geoms.link_idx[i_gpb]
                    i_lqa, i_lqb = dyn_info.geoms.link_idx[i_gqa], dyn_info.geoms.link_idx[i_gqb]
                    key_p = func_contact_sort_key(i_lpa, i_lpb, i_gpa, i_gpb, sh_sort_v[i_s_start + i_s])
                    key_q = func_contact_sort_key(i_lqa, i_lqb, i_gqa, i_gqb, sh_sort_v[i_s_start + i_t])
                    if func_contact_is_before(i_b, i_q, i_p, key_q, key_p, collider_state):
                        sh_sort_idx[i_s_start + i_s] = i_q
                        sh_sort_idx[i_s_start + i_t] = i_p
                        for i_k in qd.static(range(2)):
                            i_g = sh_sort_geoms[i_s_start + i_s, i_k]
                            sh_sort_geoms[i_s_start + i_s, i_k] = sh_sort_geoms[i_s_start + i_t, i_k]
                            sh_sort_geoms[i_s_start + i_t, i_k] = i_g
                        sh_sort_v[i_s_start + i_s] = key_q[4]
                        sh_sort_v[i_s_start + i_t] = key_p[4]
            if qd.static(n_threads == 32):
                qd.simt.subgroup.sync()
            else:
                qd.simt.block.sync()
    for i_chunk_ in range((n_sort + n_threads - 1) // n_threads):
        i_s = i_chunk_ * n_threads + i_thread
        if i_s < n_sort:
            collider_state.contact_sort_idx[i_tile_start + i_s, i_b] = sh_sort_idx[i_s_start + i_s]
    if qd.static(n_threads == 32):
        qd.simt.subgroup.sync()
    else:
        qd.simt.block.sync()


@qd.func
def func_contact_support_statistics(
    tid: int,
    i_b: int,
    i_c_owner: int,
    i_sg: int,
    n_con: int,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    cos_tol: float,
    use_coop: qd.template(),
):
    """Compute the statistics of the supporting contacts of a support group, from the first of them on.

    The supporting contacts of the support group 'i_sg' join one of its movable links to a link of another group (see
    links_support_group_idx in array_class.py), the first of them in contact order being 'i_c_owner'. Their centroid,
    their radius about it floored to keep it invertible for coincident contacts, and whether they span several link
    pairs or normals that leave the axis of the first one by an angle of cosine below 'cos_tol' are written to the
    support group fields of the collider state. 'use_coop' shares the contacts over the 32 lanes of a warp.
    """
    n_lanes = qd.static(32 if use_coop else 1)
    i_l_min, i_l_max = func_contact_link_pair(i_b, i_c_owner, collider_state)
    i_pc_owner = collider_state.contact_sort_idx[i_c_owner, i_b]
    normal = collider_state.contact_data.normal[i_pc_owner, i_b]
    pos_sum = qd.Vector.zero(gs.qd_float, 3)
    n_support = 0
    is_spread = False
    pos_mean = qd.Vector.zero(gs.qd_float, 3)
    radius = gs.qd_float(1e-12)
    for i_scan in range(2):
        for i_chunk_ in range((n_con - i_c_owner + n_lanes - 1) // n_lanes):
            i_c = i_c_owner + i_chunk_ * n_lanes + tid
            if i_c < n_con:
                i_pc = collider_state.contact_sort_idx[i_c, i_b]
                i_la = collider_state.contact_data.link_a[i_pc, i_b]
                i_lb = collider_state.contact_data.link_b[i_pc, i_b]
                i_sga = collider_info.links_support_group_idx[i_la]
                i_sgb = collider_info.links_support_group_idx[i_lb]
                i_l = i_la if i_sga == i_sg else i_lb
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                if i_sga != i_sgb and (i_sga == i_sg or i_sgb == i_sg) and not dyn_info.links.is_fixed[I_l]:
                    pos = collider_state.contact_data.pos[i_pc, i_b]
                    if i_scan == 0:
                        pos_sum += pos
                        n_support += 1
                        is_spread = (
                            is_spread
                            or qd.min(i_la, i_lb) != i_l_min
                            or qd.max(i_la, i_lb) != i_l_max
                            or qd.abs(collider_state.contact_data.normal[i_pc, i_b].dot(normal)) < cos_tol
                        )
                    else:
                        radius = qd.max(radius, (pos - pos_mean).norm())
        if i_scan == 0:
            if qd.static(use_coop):
                for k in qd.static(range(3)):
                    pos_sum[k] = su.qd_block_sum(pos_sum[k])
                n_support = su.qd_block_sum(n_support)
                is_spread = su.qd_block_any(is_spread)
            pos_mean = pos_sum / qd.max(n_support, 1)
    if qd.static(use_coop):
        radius = su.qd_block_max(radius)
    if tid == 0:
        collider_state.support_groups_pos[i_sg, i_b] = pos_mean
        collider_state.support_groups_radius[i_sg, i_b] = radius
        collider_state.support_groups_is_spread[i_sg, i_b] = is_spread


@qd.func
def func_reduce_contact_wrenches(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    use_coop: qd.template(),
):
    """Remove the contacts whose wrench one or two other contacts of their link pair reproduce within a bound.

    Runs on the contacts left by the support polygon reduction, sorted by link pair. A contact is removed when one or
    two other contacts of the same link pair, with its links in the same order and at least its friction coefficients,
    reproduce its penetration within the pruning tolerance and its normal force with a wrench error at most 5% of the
    wrench norm of a unit force. The error covers the force and torque on both links and the generalized force on the
    joints of one degree of freedom between them, with positions normalized by the radius of the supporting contacts of
    the support group of each link (see links_support_group_idx in array_class.py). The support group of both links must
    carry contacts of several link pairs or normals, so that the removal leaves them supported elsewhere. Proposals are
    accepted in contact order, the donors of an accepted removal being kept.

    The kept contacts are compacted in 'contact_sort_idx' and 'n_contacts' is updated. 'use_coop' runs every environment
    on the 32 lanes of a warp, which share out the support statistics and the proposals, instead of a single thread.
    The reduction runs in a kernel of its own, whose register footprint would slow down the support polygon reduction.
    """
    _B = collider_state.n_contacts.shape[0]
    n_threads = qd.static(32 if use_coop else 1)
    if qd.static(use_coop):
        # Every environment takes a warp, 4 of them sharing a block
        qd.loop_config(name="reduce_contact_wrenches_coop", block_dim=4 * n_threads)
    else:
        qd.loop_config(name="reduce_contact_wrenches", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_flat in range(_B * n_threads):
        i_thread = i_flat % n_threads
        i_b = i_flat // n_threads
        n_links = collider_info.links_support_group_idx.shape[0]
        tol = collider_info.contact_pruning_tolerance[None]
        cos_tol = 1.0 - 0.5 * tol * tol
        prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
        EPS = rigid_info.EPS[None]
        n_con = collider_state.n_contacts[i_b]
        n_hib = collider_state.n_contacts_hibernated[i_b]

        # Small contact sets keep their support without paying for the reduction, which needs a link pair holding
        # several contacts. Every thread finds the same answer from the same reads.
        is_reducible = False
        if n_con - n_hib > 8 and tol > 0.0:
            i_l_min_prev, i_l_max_prev = func_contact_link_pair(i_b, n_hib, collider_state)
            for i_c in range(n_hib + 1, n_con):
                i_l_min, i_l_max = func_contact_link_pair(i_b, i_c, collider_state)
                if i_l_min == i_l_min_prev and i_l_max == i_l_max_prev:
                    is_reducible = True
                    break
                i_l_min_prev, i_l_max_prev = i_l_min, i_l_max
        # Every environment takes the same passes, which keeps the synchronizations uniform over a 64-lane AMDGPU
        # wavefront running two of them
        n_reduce = n_con - n_hib if is_reducible else 0

        # The first supporting contact of a support group in contact order owns it, and computes its statistics
        for i_pass in range(3):
            for i_chunk_ in range((n_reduce + n_threads - 1) // n_threads):
                i_c = n_hib + i_chunk_ * n_threads + i_thread
                # Past the last contact, both links in one group make it support neither
                i_la, i_lb, i_sga, i_sgb = 0, 0, -1, -1
                if i_c < n_con:
                    i_pc = collider_state.contact_sort_idx[i_c, i_b]
                    i_la = collider_state.contact_data.link_a[i_pc, i_b]
                    i_lb = collider_state.contact_data.link_b[i_pc, i_b]
                    i_sga = collider_info.links_support_group_idx[i_la]
                    i_sgb = collider_info.links_support_group_idx[i_lb]
                for i_side in range(2):
                    i_l, i_sg = i_la, i_sga
                    if i_side == 1:
                        i_l, i_sg = i_lb, i_sgb
                    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                    is_owner = False
                    if i_sga != i_sgb and not dyn_info.links.is_fixed[I_l]:
                        if i_pass == 0:
                            collider_state.support_groups_contact_first[i_sg, i_b] = n_con
                        elif i_pass == 1:
                            qd.atomic_min(collider_state.support_groups_contact_first[i_sg, i_b], i_c)
                        else:
                            is_owner = collider_state.support_groups_contact_first[i_sg, i_b] == i_c
                    if i_pass == 2:
                        # The owners of the chunk take their turn, the lanes of the warp sharing their contacts. The
                        # owners make a lane mask of the type qd_block_ballot returns.
                        mask_owners = qd.cast(is_owner, qd.u32)
                        if qd.static(use_coop):
                            mask_owners = su.qd_block_ballot(is_owner)
                        for i_owner_ in range(n_threads):
                            if mask_owners != 0:
                                i_lane = qd.cast(qd.math.ffs(mask_owners), gs.qd_int) - 1
                                mask_owners = mask_owners & (mask_owners - 1)
                                i_sg_owner = i_sg
                                if qd.static(use_coop):
                                    i_sg_owner = su.qd_block_shuffle(i_sg, i_lane)
                                func_contact_support_statistics(
                                    i_thread,
                                    i_b,
                                    n_hib + i_chunk_ * n_threads + i_lane,
                                    i_sg_owner,
                                    n_con,
                                    collider_state,
                                    dyn_info,
                                    collider_info,
                                    rigid_config,
                                    cos_tol,
                                    use_coop,
                                )
            if qd.static(use_coop):
                qd.simt.subgroup.sync()

        tol_force = gs.qd_float(0.05)

        # Each contact proposes the donors that may replace it, at its position in contact_lex_idx and
        # contact_hull_stack, free once the support polygons are reduced, and contact_keep holds the decision on it at
        # its physical slot: 1 undecided, 0 removed, 2 donor of a removal.
        for i_chunk_ in range((n_reduce + n_threads - 1) // n_threads):
            i_c = n_hib + i_chunk_ * n_threads + i_thread
            if i_c >= n_con:
                continue
            collider_state.contact_lex_idx[i_c, i_b] = -1
            i_pc = collider_state.contact_sort_idx[i_c, i_b]
            collider_state.contact_keep[i_pc, i_b] = 1
            i_la = collider_state.contact_data.link_a[i_pc, i_b]
            i_lb = collider_state.contact_data.link_b[i_pc, i_b]
            i_sga = collider_info.links_support_group_idx[i_la]
            i_sgb = collider_info.links_support_group_idx[i_lb]
            I_la = [i_la, i_b] if qd.static(rigid_config.batch_links_info) else i_la
            I_lb = [i_lb, i_b] if qd.static(rigid_config.batch_links_info) else i_lb
            is_fixed_a = dyn_info.links.is_fixed[I_la]
            is_fixed_b = dyn_info.links.is_fixed[I_lb]
            if i_sga == i_sgb:
                continue
            if not (is_fixed_a or collider_state.support_groups_is_spread[i_sga, i_b]):
                continue
            if not (is_fixed_b or collider_state.support_groups_is_spread[i_sgb, i_b]):
                continue

            # The donors are taken from the contacts of the same link pair, contiguous in the ordering
            i_l_min, i_l_max = qd.min(i_la, i_lb), qd.max(i_la, i_lb)
            i_c_start = i_c
            while i_c_start > n_hib:
                i_l_min_prev, i_l_max_prev = func_contact_link_pair(i_b, i_c_start - 1, collider_state)
                if i_l_min_prev != i_l_min or i_l_max_prev != i_l_max:
                    break
                i_c_start -= 1
            i_c_end = i_c + 1
            while i_c_end < n_con:
                i_l_min_next, i_l_max_next = func_contact_link_pair(i_b, i_c_end, collider_state)
                if i_l_min_next != i_l_min or i_l_max_next != i_l_max:
                    break
                i_c_end += 1
            if i_c_end - i_c_start <= 1:
                continue

            pos_c = collider_state.contact_data.pos[i_pc, i_b]
            penetration = collider_state.contact_data.penetration[i_pc, i_b]
            origin = collider_state.support_groups_pos[i_sga, i_b]
            origin_other = collider_state.support_groups_pos[i_sgb, i_b]
            radius = collider_state.support_groups_radius[i_sga, i_b]
            radius_other = collider_state.support_groups_radius[i_sgb, i_b]
            if is_fixed_a:
                origin, radius = origin_other, radius_other
            if is_fixed_b:
                origin_other, radius_other = origin, radius
            # Squared wrench norm of a unit normal force, the reference of the relative error bound, increasing with
            # the normalized position
            position_norm = qd.min((pos_c - origin).norm() / radius, (pos_c - origin_other).norm() / radius_other)
            error_max_sqr = tol_force**2 * (0.5 * (qd.sqrt(position_norm**2 + 4) + position_norm) ** 2)
            # The residual is bounded at the smaller radius, about the nearer origin
            radius = qd.min(radius, radius_other)
            if (pos_c - origin_other).norm_sqr() < (pos_c - origin).norm_sqr():
                origin, origin_other = origin_other, origin
            position = (pos_c - origin) / radius
            normal = collider_state.contact_data.normal[i_pc, i_b].normalized()

            # Generalized force metric of the joints of one degree of freedom between both links, from the links up to
            # their common ancestor. The same-link residual cancels force and direct torque.
            covariance = qd.Matrix.zero(gs.qd_float, 3, 3)
            metric_linear = qd.Vector.zero(gs.qd_float, 3)
            metric_constant = gs.qd_float(0.0)
            # Descending link order stops both walks where they meet (see _add_friction_constraint in solver.py)
            i_la_walk, i_lb_walk = i_la, i_lb
            for i_walk_ in range(n_links):
                if i_la_walk == i_lb_walk:
                    break
                i_l = qd.max(i_la_walk, i_lb_walk)
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                incidence = gs.qd_float(1.0) if i_la_walk > i_lb_walk else gs.qd_float(-1.0)
                i_parent = dyn_info.links.parent_idx[I_l]
                if i_la_walk > i_lb_walk:
                    i_la_walk = i_parent
                else:
                    i_lb_walk = i_parent
                # Joint rows come from the links joined to their parent by one degree of freedom at most
                if i_parent < 0:
                    continue
                i_sg = collider_info.links_support_group_idx[i_l]
                if i_sg != collider_info.links_support_group_idx[i_parent]:
                    continue
                for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                    I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                    if dyn_info.joints.n_dofs[I_j] == 1:
                        axis = incidence * dyn_state.joints.xaxis[i_j, i_b]
                        metric_constant += axis.norm_sqr()
                        if dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                            row = axis.cross((pos_c - dyn_state.joints.xanchor[i_j, i_b]) / radius)
                            covariance += axis.outer_product(axis)
                            metric_linear += row.cross(axis)
                            metric_constant += row.norm_sqr()
            metric = covariance.trace() * qd.Matrix.identity(gs.qd_float, 3) - covariance

            # The candidate donors are the 8 nearest contacts in position and normal with the links of the removed
            # contact in the same order and at least its friction coefficients, a search budget bounding the pairs
            # tried below. The error bound decides every removal.
            donors = qd.Vector([-1] * 8, dt=gs.qd_int)
            distances = qd.Vector([1e30] * 8, dt=gs.qd_float)
            n_donors = 0
            for i_cd in range(i_c_start, i_c_end):
                i_pcd = collider_state.contact_sort_idx[i_cd, i_b]
                if i_cd == i_c or collider_state.contact_data.link_a[i_pcd, i_b] != i_la:
                    continue
                if (
                    collider_state.contact_data.friction[i_pcd, i_b] < collider_state.contact_data.friction[i_pc, i_b]
                    or collider_state.contact_data.friction_rolling[i_pcd, i_b]
                    < collider_state.contact_data.friction_rolling[i_pc, i_b]
                    or collider_state.contact_data.friction_torsional[i_pcd, i_b]
                    < collider_state.contact_data.friction_torsional[i_pc, i_b]
                ):
                    continue
                delta = (collider_state.contact_data.pos[i_pcd, i_b] - origin) / radius - position
                distance = delta.norm_sqr() + (collider_state.contact_data.normal[i_pcd, i_b] - normal).norm_sqr()
                i_insert = i_cd
                for i_k in range(8):
                    if distance < distances[i_k]:
                        distance, distances[i_k] = distances[i_k], distance
                        i_insert, donors[i_k] = donors[i_k], i_insert
                n_donors = qd.min(n_donors + 1, 8)

            # The removed contact goes onto the segment between two donors, or onto a single one. The weight of the
            # second donor is the one nearest the removed contact in the metric of the residual: four times the identity
            # for the moment arms on both links, joints added. The error decides acceptance whatever the weight.
            residual_metric = metric + 4.0 * qd.Matrix.identity(gs.qd_float, 3)
            torque_metric = covariance + 2.0 * qd.Matrix.identity(gs.qd_float, 3)
            depth_rounding = EPS * (2.0 * pos_c.norm() + penetration + 2.0 * radius)
            is_removed = False
            for i_k0 in range(n_donors):
                if is_removed:
                    break
                for i_k1 in range(i_k0, n_donors):
                    i_c0 = donors[i_k0]
                    i_c1 = donors[i_k1]
                    i_pc0 = collider_state.contact_sort_idx[i_c0, i_b]
                    i_pc1 = collider_state.contact_sort_idx[i_c1, i_b]
                    penetration_0 = collider_state.contact_data.penetration[i_pc0, i_b]
                    penetration_1 = collider_state.contact_data.penetration[i_pc1, i_b]
                    if penetration > prune_deep_penetration_ratio * qd.max(penetration_0, penetration_1):
                        continue
                    pos_0 = collider_state.contact_data.pos[i_pc0, i_b]
                    pos_1 = collider_state.contact_data.pos[i_pc1, i_b]
                    offset = (pos_c - pos_0) / radius
                    edge = (pos_1 - pos_0) / radius
                    weight = gs.qd_float(0.0)
                    length_sqr = edge.dot(residual_metric @ edge)
                    if length_sqr > 0.0:
                        weight = qd.min(1.0, qd.max(0.0, offset.dot(residual_metric @ edge) / length_sqr))
                    # The donors must reproduce the penetration of the removed contact too, the cheapest test of a pair
                    penetration_donor = (1.0 - weight) * penetration_0 + weight * penetration_1
                    if penetration > penetration_donor + tol * qd.max(penetration, penetration_donor) + depth_rounding:
                        continue
                    # Torque of the residual moment arm, by the largest eigenvalue of its rank-two Gram matrix. The
                    # torque metric holds twice the identity, so that the error is at least sqrt(2) times the moment
                    # arm, which rejects a pair beyond it before the matrices.
                    delta = offset - weight * edge
                    if 2.0 * delta.norm_sqr() > error_max_sqr:
                        continue
                    skew = gu.qd_vec_to_skew(delta)
                    gram_torque = skew.transpose() @ torque_metric @ skew
                    trace = gram_torque.trace()
                    spread = qd.sqrt(qd.max(0.0, 2.0 * (gram_torque * gram_torque).sum() - trace * trace))
                    error = qd.sqrt(qd.max(0.0, 0.5 * (trace + spread)))
                    # Rotation of the force direction between the removed contact and each donor, by the chord between
                    # their normals times the norm of the map of the donor, joints added. The map of one link
                    # [[I, 0], [skew(r), I]] has the squared Frobenius norm 6 + 2 |r|^2, so that both links bound it by
                    # 12 + 4 max(|r_a|^2, |r_b|^2).
                    for i_side in range(2):
                        i_pt, pos_t, fraction = i_pc0, pos_0, 1.0 - weight
                        if i_side == 1:
                            i_pt, pos_t, fraction = i_pc1, pos_1, weight
                        displacement = (pos_t - pos_c) / radius
                        norm_donor = metric_constant + 2.0 * displacement.dot(metric_linear)
                        norm_donor += displacement.dot(metric @ displacement)
                        position_other = (pos_t - origin_other) / radius
                        norm_donor += 12.0 + 4.0 * qd.max(
                            (position + displacement).norm_sqr(), position_other.norm_sqr()
                        )
                        normal_donor = collider_state.contact_data.normal[i_pt, i_b].normalized()
                        error += fraction * qd.sqrt(qd.max(0.0, norm_donor)) * (normal - normal_donor).norm()
                    if error * error > error_max_sqr:
                        continue
                    collider_state.contact_lex_idx[i_c, i_b] = i_c0
                    collider_state.contact_hull_stack[i_c, i_b] = i_c1
                    is_removed = True
                    break
        if qd.static(use_coop):
            qd.simt.subgroup.sync()

        # The lanes load the proposals of a chunk of contacts at once, which the first lane then decides in turn
        for i_chunk_ in range((n_reduce + n_threads - 1) // n_threads):
            i_c = n_hib + i_chunk_ * n_threads + i_thread
            i_pc, i_pc0, i_pc1 = -1, -1, -1
            if i_c < n_con:
                i_c0 = collider_state.contact_lex_idx[i_c, i_b]
                if i_c0 >= 0:
                    i_c1 = collider_state.contact_hull_stack[i_c, i_b]
                    i_pc = collider_state.contact_sort_idx[i_c, i_b]
                    i_pc0 = collider_state.contact_sort_idx[i_c0, i_b]
                    i_pc1 = collider_state.contact_sort_idx[i_c1, i_b]
            for i_lane in range(n_threads):
                i_pc_lane, i_pc0_lane, i_pc1_lane = i_pc, i_pc0, i_pc1
                if qd.static(use_coop):
                    i_pc_lane = su.qd_block_shuffle(i_pc, i_lane)
                    i_pc0_lane = su.qd_block_shuffle(i_pc0, i_lane)
                    i_pc1_lane = su.qd_block_shuffle(i_pc1, i_lane)
                if i_thread == 0 and i_pc_lane >= 0:
                    if (
                        collider_state.contact_keep[i_pc_lane, i_b] == 1
                        and collider_state.contact_keep[i_pc0_lane, i_b] > 0
                        and collider_state.contact_keep[i_pc1_lane, i_b] > 0
                    ):
                        collider_state.contact_keep[i_pc_lane, i_b] = 0
                        collider_state.contact_keep[i_pc0_lane, i_b] = 2
                        collider_state.contact_keep[i_pc1_lane, i_b] = 2
        if qd.static(use_coop):
            qd.simt.subgroup.sync()

        n_kept = func_contact_compact(i_thread, i_b, n_hib, n_hib + n_reduce, collider_state, use_coop)
        if i_thread == 0 and is_reducible:
            collider_state.n_contacts[i_b] = n_kept


@qd.func
def func_sort_contact_tiles_coop(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """Sort by tiles the live contacts of every environment holding 3 contacts or more, for the cooperative pruning.

    Every tile is sorted by a block of its own (see func_contact_sort_tile_coop), so that the tiles of an environment
    run side by side, then the upper levels of the sorting network merge them in global memory, one block per
    environment. The blocks of the other environments return at once.
    """
    _B = collider_state.n_contacts.shape[0]
    _K = qd.static(32)
    _W = qd.static(4)
    # Contacts of a tile, sorted in the shared memory of its block
    _N_TILE = qd.static(16 * _K * _W)
    n_tiles = (collider_state.contact_keep.shape[0] + _N_TILE - 1) // _N_TILE
    qd.loop_config(name="sort_contact_tiles_coop", block_dim=_K * _W)
    for i_flat in range(_B * _K * _W * n_tiles):
        i_thread = i_flat % (_K * _W)
        i_block = i_flat // (_K * _W)
        i_tile = i_block % n_tiles
        i_b = i_block // n_tiles
        sh_sort_geoms = qd.simt.block.SharedArray((_N_TILE, 2), gs.qd_int)
        sh_sort_v = qd.simt.block.SharedArray((_N_TILE,), gs.qd_float)
        sh_sort_idx = qd.simt.block.SharedArray((_N_TILE,), gs.qd_int)
        n_con = qd.min(collider_state.n_contacts[i_b], collider_info.max_candidate_contacts[None])
        n_hib = collider_state.n_contacts_hibernated[i_b]
        i_tile_start = n_hib + i_tile * _N_TILE
        if n_con - n_hib >= 3 and i_tile_start < n_con:
            func_contact_sort_tile_coop(
                i_thread,
                i_b,
                i_tile_start,
                qd.min(_N_TILE, n_con - i_tile_start),
                0,
                sh_sort_geoms,
                sh_sort_v,
                sh_sort_idx,
                dyn_state,
                collider_state,
                dyn_info,
                n_threads=_K * _W,
            )

    # The merge runs in a kernel of its own, free of the register footprint of the pruning kernel
    qd.loop_config(name="merge_contact_tiles_coop", block_dim=_K * _W)
    for i_flat in range(_B * _K * _W):
        i_thread = i_flat % (_K * _W)
        i_b = i_flat // (_K * _W)
        n_con = qd.min(collider_state.n_contacts[i_b], collider_info.max_candidate_contacts[None])
        n_hib = collider_state.n_contacts_hibernated[i_b]
        n_live = n_con - n_hib
        if n_live > 8 * _K * _W:
            n_levels = su.qd_bitonic_n_levels(n_live)
            for i_level in range(qd.static(_N_TILE.bit_length() - 1), n_levels):
                for i_step in range(i_level + 1):
                    for i_chunk_ in range(((1 << (n_levels - 1)) + _K * _W - 1) // (_K * _W)):
                        i_pair = i_chunk_ * _K * _W + i_thread
                        i_s, i_t = su.qd_bitonic_pair(i_pair, i_level, i_step)
                        if i_t < n_live:
                            i_p = collider_state.contact_sort_idx[n_hib + i_s, i_b]
                            i_q = collider_state.contact_sort_idx[n_hib + i_t, i_b]
                            key_p = func_contact_sort_key(
                                collider_state.contact_data.link_a[i_p, i_b],
                                collider_state.contact_data.link_b[i_p, i_b],
                                collider_state.contact_data.geom_a[i_p, i_b],
                                collider_state.contact_data.geom_b[i_p, i_b],
                                collider_state.contact_proj_v[i_p, i_b],
                            )
                            key_q = func_contact_sort_key(
                                collider_state.contact_data.link_a[i_q, i_b],
                                collider_state.contact_data.link_b[i_q, i_b],
                                collider_state.contact_data.geom_a[i_q, i_b],
                                collider_state.contact_data.geom_b[i_q, i_b],
                                collider_state.contact_proj_v[i_q, i_b],
                            )
                            if func_contact_is_before(i_b, i_q, i_p, key_q, key_p, collider_state):
                                collider_state.contact_sort_idx[n_hib + i_s, i_b] = i_q
                                collider_state.contact_sort_idx[n_hib + i_t, i_b] = i_p
                    qd.simt.block.sync()


@qd.func
def func_clamp_prune_contacts_coop(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    errno: qd.Tensor,
):
    """Reduce the contacts of every environment with one block of threads each (see func_clamp_prune_contacts).

    The contacts come sorted by func_sort_contact_tiles_coop. The threads of a block share out the patches of a round:
    each warp projects its patches with its own lanes, each thread builds the support polygon of one patch, then each
    warp restores the deep contacts of its patches. The contact budget is checked last.
    """
    _B = collider_state.n_contacts.shape[0]
    _K = qd.static(32)
    _W = qd.static(4)
    qd.loop_config(name="clamp_prune_contacts_coop", block_dim=_K * _W)
    for i_flat in range(_B * _K * _W):
        i_thread = i_flat % (_K * _W)
        tid = i_thread % _K
        i_warp = i_thread // _K
        i_b = i_flat // (_K * _W)
        # Bounds and statistics of the patches of a round, from one phase to the next
        sh_patch_start = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_patch_end = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_patch_hull_end = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_is_coplanar = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_is_constant_depth = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_max_in_plane_r2 = qd.simt.block.SharedArray((_K * _W,), gs.qd_float)
        sh_penetration_min = qd.simt.block.SharedArray((_K * _W,), gs.qd_float)
        sh_penetration_max = qd.simt.block.SharedArray((_K * _W,), gs.qd_float)
        sh_penetration_hull = qd.simt.block.SharedArray((_K * _W,), gs.qd_float)
        sh_i_cb_u = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_i_ca = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_i_ch = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_n_hull = qd.simt.block.SharedArray((_K * _W,), gs.qd_int)
        sh_round = qd.simt.block.SharedArray((2,), gs.qd_int)

        # See func_clamp_prune_contacts for the scalars read inside the loop
        max_contacts = collider_info.max_contacts[None]
        tol = collider_info.contact_pruning_tolerance[None]
        cos_tol = 1.0 - 0.5 * tol * tol
        prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
        EPS = rigid_info.EPS[None]

        n_con = qd.min(collider_state.n_contacts[i_b], collider_info.max_candidate_contacts[None])
        n_hib = collider_state.n_contacts_hibernated[i_b]
        n_live = n_con - n_hib
        qd.simt.block.sync()
        if i_thread == 0:
            collider_state.n_contacts[i_b] = n_con

        # Identity permutation and default keep of the live contacts, read by every consumer even without pruning. The
        # contacts of a set sorted by tiles already hold their permutation.
        is_sorted = n_live >= 3
        for i_chunk_ in range((n_live + _K * _W - 1) // (_K * _W)):
            i_c = n_hib + i_chunk_ * _K * _W + i_thread
            if i_c < n_con:
                collider_state.contact_keep[i_c, i_b] = 1
                if not is_sorted:
                    collider_state.contact_sort_idx[i_c, i_b] = i_c

        if n_live >= 3:
            # Only a link pair of 3 contacts or more holds a prunable patch, its contacts contiguous after the sort
            if i_warp == 0:
                has_bucket = False
                for i_chunk_ in range((n_live - 2 + _K - 1) // _K):
                    i_c = n_hib + i_chunk_ * _K + tid
                    if i_c + 2 < n_con:
                        i_l_min, i_l_max = func_contact_link_pair(i_b, i_c, collider_state)
                        i_l_min_next, i_l_max_next = func_contact_link_pair(i_b, i_c + 2, collider_state)
                        has_bucket = has_bucket or (i_l_min == i_l_min_next and i_l_max == i_l_max_next)
                has_bucket = su.qd_block_any(has_bucket)
                if tid == 0:
                    sh_round[0] = 1 if has_bucket else 0
            qd.simt.block.sync()

            if sh_round[0] != 0:
                # The lanes of the first warp group the buckets holding several patches together
                qd.simt.block.sync()
                if i_warp == 0:
                    func_contact_group_patches(tid, i_b, n_hib, n_con, collider_state, cos_tol, use_coop=True)
                qd.simt.block.sync()

                # Phase 2: the patches are taken in rounds of up to one per thread, which the first warp finds. Each
                # warp projects its share of them with its own lanes, each thread builds the support polygon of one
                # patch, a serial chain of reads that the patches of a round run side by side, then each warp restores
                # the deep contacts of its share. Each patch writes the scratch of its own range of contacts alone.
                i_cb_next = n_hib
                while i_cb_next < n_con:
                    if i_warp == 0:
                        i_cb_start = i_cb_next
                        n_patches_round = 0
                        while i_cb_start < n_con and n_patches_round < _K * _W:
                            i_cb_end = func_contact_patch_end(
                                tid,
                                i_b,
                                i_cb_start,
                                n_con,
                                collider_state,
                                cos_tol,
                                use_coop=True,
                                split_by_normal=True,
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
                            i_cb_start = sh_patch_start[i_patch]
                            i_cb_end = sh_patch_end[i_patch]
                            is_prunable = func_contact_patch_is_prunable(
                                i_b,
                                i_cb_start,
                                i_cb_end,
                                collider_state,
                                dyn_info,
                                collider_static_config,
                                gjk_static_config,
                            )
                            is_coplanar = False
                            if is_prunable:
                                (
                                    is_coplanar,
                                    is_constant_depth,
                                    max_in_plane_r2,
                                    penetration_min,
                                    penetration_max,
                                    i_cb_u,
                                    i_ca,
                                ) = func_contact_patch_project(
                                    tid, i_b, i_cb_start, i_cb_end, collider_state, tol, EPS, use_coop=True
                                )
                                if is_coplanar:
                                    n_cb = func_contact_filter_interior(
                                        tid,
                                        i_b,
                                        i_cb_start,
                                        i_cb_end,
                                        max_in_plane_r2,
                                        collider_state,
                                        EPS,
                                        use_coop=True,
                                    )
                                    func_contact_sort_positions(
                                        tid, i_b, i_cb_start, n_cb, collider_state, use_coop=True
                                    )
                                    if tid == 0:
                                        sh_patch_hull_end[i_patch] = i_cb_start + n_cb
                                        sh_is_constant_depth[i_patch] = 1 if is_constant_depth else 0
                                        sh_max_in_plane_r2[i_patch] = max_in_plane_r2
                                        sh_penetration_min[i_patch] = penetration_min
                                        sh_penetration_max[i_patch] = penetration_max
                                        sh_i_cb_u[i_patch] = i_cb_u
                                        sh_i_ca[i_patch] = i_ca
                            if tid == 0:
                                sh_is_coplanar[i_patch] = 1 if is_coplanar else 0
                    qd.simt.block.sync()

                    # Consecutive patches go to different warps, which step through their diverging hull builds
                    # independently, whereas the lanes of one warp step through them in turn
                    i_patch = tid * _W + i_warp
                    if i_patch < n_patches_round:
                        if sh_is_coplanar[i_patch] != 0:
                            i_cb_start = sh_patch_start[i_patch]
                            max_in_plane_r2 = sh_max_in_plane_r2[i_patch]
                            n_hull, i_cb_deepest = func_contact_support_hull(
                                i_b,
                                i_cb_start,
                                sh_patch_hull_end[i_patch],
                                collider_state,
                                tol,
                                rounding=qd.sqrt(EPS * max_in_plane_r2),
                                eps=EPS,
                            )
                            penetration_hull, i_ch = func_contact_mark_hull(
                                i_b, i_cb_start, i_cb_deepest, n_hull, max_in_plane_r2, collider_state, tol, EPS
                            )
                            sh_n_hull[i_patch] = n_hull
                            sh_i_ch[i_patch] = i_ch
                            sh_penetration_hull[i_patch] = penetration_hull
                    qd.simt.block.sync()

                    # The deep contacts of every reduced patch, the lanes of a warp sharing its contacts again
                    for i_patch_ in range((n_patches_round + _W - 1) // _W):
                        i_patch = i_patch_ * _W + i_warp
                        if i_patch < n_patches_round:
                            if sh_is_coplanar[i_patch] != 0:
                                func_contact_restore_depth(
                                    tid,
                                    i_b,
                                    sh_patch_start[i_patch],
                                    sh_patch_end[i_patch],
                                    sh_i_cb_u[i_patch],
                                    sh_i_ca[i_patch],
                                    sh_i_ch[i_patch],
                                    sh_n_hull[i_patch],
                                    sh_penetration_min[i_patch],
                                    sh_penetration_max[i_patch],
                                    sh_penetration_hull[i_patch],
                                    collider_state,
                                    prune_deep_penetration_ratio,
                                    tol,
                                    EPS,
                                    use_coop=True,
                                    is_constant_depth=sh_is_constant_depth[i_patch] != 0,
                                )
                    qd.simt.block.sync()

                # Phase 3: compaction of contact_sort_idx
                if i_warp == 0:
                    n_kept = func_contact_compact(tid, i_b, n_hib, n_con, collider_state, use_coop=True)
                    if tid == 0:
                        collider_state.n_contacts[i_b] = n_kept
                qd.simt.block.sync()

        # See func_clamp_prune_contacts for the contact budget
        if i_thread == 0:
            if collider_state.n_contacts[i_b] > max_contacts:
                collider_state.n_contacts[i_b] = max_contacts
                errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CONTACTS


@qd.func
def func_clamp_prune_contacts_warp(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    errno: qd.Tensor,
):
    """Reduce the contacts of every environment with one warp each, 4 environments sharing a block.

    The lanes of a warp sort the contacts of their environment, group them by link pair and normal, then reduce every
    prunable patch in turn: they project the patch, filter it and restore its deep contacts together, the first lane
    building the support polygon. The kept contacts are compacted and the contact budget is checked last. An
    environment holds a few contacts in the scenes this path serves, which a block per environment would leave idle
    (see func_clamp_prune_contacts_coop for the larger contact buffers).
    """
    _B = collider_state.n_contacts.shape[0]
    _K = qd.static(32)
    _N_ENVS_PER_BLOCK = qd.static(4)
    # Contacts the warp of an environment stages in shared memory to sort them. The rare environment holding more sorts
    # them in place by a heapsort on its first lane, in the same order (see func_contact_is_before).
    _N_STAGED = qd.static(256)

    qd.loop_config(name="clamp_prune_contacts_warp", block_dim=_K * _N_ENVS_PER_BLOCK)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K
        i_s_start = (i_b % _N_ENVS_PER_BLOCK) * _N_STAGED
        sh_sort_geoms = qd.simt.block.SharedArray((_N_ENVS_PER_BLOCK * _N_STAGED, 2), gs.qd_int)
        sh_sort_v = qd.simt.block.SharedArray((_N_ENVS_PER_BLOCK * _N_STAGED,), gs.qd_float)
        sh_sort_idx = qd.simt.block.SharedArray((_N_ENVS_PER_BLOCK * _N_STAGED,), gs.qd_int)
        tol = collider_info.contact_pruning_tolerance[None]
        cos_tol = 1.0 - 0.5 * tol * tol
        prune_deep_penetration_ratio = collider_info.prune_deep_penetration_ratio[None]
        EPS = rigid_info.EPS[None]
        max_contacts = collider_info.max_contacts[None]
        n_con = qd.min(collider_state.n_contacts[i_b], collider_info.max_candidate_contacts[None])
        n_hib = collider_state.n_contacts_hibernated[i_b]
        n_live = n_con - n_hib
        qd.simt.subgroup.sync()
        if tid == 0:
            collider_state.n_contacts[i_b] = n_con
        for i_chunk_ in range((n_live + _K - 1) // _K):
            i_c = n_hib + i_chunk_ * _K + tid
            if i_c < n_con:
                collider_state.contact_keep[i_c, i_b] = 1
                collider_state.contact_sort_idx[i_c, i_b] = i_c
        qd.simt.subgroup.sync()
        if n_live >= 3 and tol > 0.0:
            if n_live <= _N_STAGED:
                func_contact_sort_tile_coop(
                    tid,
                    i_b,
                    n_hib,
                    n_live,
                    i_s_start,
                    sh_sort_geoms,
                    sh_sort_v,
                    sh_sort_idx,
                    dyn_state,
                    collider_state,
                    dyn_info,
                    n_threads=_K,
                )
            else:
                for i_chunk_ in range((n_live + _K - 1) // _K):
                    i_c = n_hib + i_chunk_ * _K + tid
                    if i_c < n_con:
                        i_gb = collider_state.contact_data.geom_b[i_c, i_b]
                        pos = collider_state.contact_data.pos[i_c, i_b] - dyn_state.geoms.pos[i_gb, i_b]
                        pos_local = gu.qd_inv_transform_by_quat(pos, dyn_state.geoms.quat[i_gb, i_b])
                        collider_state.contact_proj_v[i_c, i_b] = func_contact_order_key(pos_local)
                qd.simt.subgroup.sync()
                if tid == 0:
                    func_contact_heapsort(i_b, n_hib, n_con, collider_state, CONTACT_ORDER.LINK_PAIR)
                qd.simt.subgroup.sync()
            # Only a link pair of 3 contacts or more holds a prunable patch, its contacts contiguous after the sort
            has_bucket = False
            for i_chunk_ in range((n_live - 2 + _K - 1) // _K):
                i_c = n_hib + i_chunk_ * _K + tid
                if i_c + 2 < n_con:
                    i_l_min, i_l_max = func_contact_link_pair(i_b, i_c, collider_state)
                    i_l_min_next, i_l_max_next = func_contact_link_pair(i_b, i_c + 2, collider_state)
                    has_bucket = has_bucket or (i_l_min == i_l_min_next and i_l_max == i_l_max_next)
            if su.qd_block_any(has_bucket):
                func_contact_group_patches(tid, i_b, n_hib, n_con, collider_state, cos_tol, use_coop=True)
                func_contact_reduce_patches(
                    tid,
                    i_b,
                    n_hib,
                    n_con,
                    collider_state,
                    dyn_info,
                    collider_static_config,
                    gjk_static_config,
                    tol,
                    cos_tol,
                    prune_deep_penetration_ratio,
                    EPS,
                    use_coop=True,
                )
        qd.simt.subgroup.sync()
        if n_live >= 3:
            n_kept = func_contact_compact(tid, i_b, n_hib, n_con, collider_state, use_coop=True)
            if tid == 0:
                collider_state.n_contacts[i_b] = n_kept
        # See func_clamp_prune_contacts for the contact budget
        if tid == 0:
            if collider_state.n_contacts[i_b] > max_contacts:
                collider_state.n_contacts[i_b] = max_contacts
                errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_CONTACTS
