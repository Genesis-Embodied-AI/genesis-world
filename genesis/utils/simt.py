import quadrants as qd


@qd.func
def qd_block_sum(value, log2_size: qd.template() = 5):
    """Sum value over the 2**log2_size lanes of a block, every lane receiving the bits lane 0 holds."""
    # The butterfly adds the same operands on the two lanes of every pair, but the compiler contracts the multiply that
    # produced a lane's own operand into that add, so the two lanes round differently, and a decision every lane takes
    # on the sum then diverges.
    return qd.simt.subgroup.broadcast(qd.simt.subgroup.reduce_all_add_tiled(value, log2_size), qd.u32(0))


@qd.func
def qd_block_scan(value):
    """Inclusive prefix sum of value over the 32 lanes of a block, returning this lane's prefix and the block total,
    the total the bits the last lane holds so every lane carries the same running count across chunks."""
    value_incl = qd.simt.subgroup.inclusive_add(value)
    return value_incl, qd.simt.subgroup.broadcast(value_incl, qd.u32(31))


@qd.func
def qd_block_min(value):
    """Minimum of value over the 32 lanes of a block, every lane receiving the bits lane 0 holds.

    See qd_block_sum for why lane 0's bits are broadcast.
    """
    return qd.simt.subgroup.broadcast(qd.simt.subgroup.reduce_all_min_tiled(value, 5), qd.u32(0))


@qd.func
def qd_block_max(value):
    """Maximum of value over the 32 lanes of a block, every lane receiving the bits lane 0 holds.

    See qd_block_sum for why lane 0's bits are broadcast.
    """
    return qd.simt.subgroup.broadcast(qd.simt.subgroup.reduce_all_max_tiled(value, 5), qd.u32(0))


@qd.func
def qd_slot_neighbors(tid, i_slot):
    """Slots of the two neighboring lanes of a 32-lane chunk, the inputs of qd_segment_bounds."""
    i_slot_prev = qd.simt.subgroup.shuffle_up(i_slot, qd.u32(1))
    i_slot_next = qd.simt.subgroup.shuffle_down(i_slot, qd.u32(1))
    return i_slot_prev, i_slot_next


@qd.func
def qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next):
    """Whether lane tid of a 32-lane chunk opens and whether it closes the run of lanes sharing slot i_slot.

    i_slot_prev and i_slot_next are the slots of the neighboring lanes, a lane holding no item carrying slot -1. The
    chunk boundary closes every run, and the tail lane alone writes a segment out so segments accumulate in chunk order.
    """
    WARP_SIZE = qd.static(32)
    is_head = 1
    if tid > 0 and i_slot_prev == i_slot:
        is_head = 0
    is_tail = i_slot >= 0 and (tid == WARP_SIZE - 1 or i_slot_next != i_slot)
    return is_head, is_tail


@qd.func
def qd_segmented_sum(tid, i_slot, i_slot_prev, i_slot_next, value):
    """Sum value over the lanes of a 32-lane chunk that share slot i_slot.

    Returns the total and whether this lane is the segment's tail (see qd_segment_bounds).
    """
    is_head, is_tail = qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next)
    total = qd.simt.subgroup.segmented_reduce_add_tiled(value, is_head, 5)
    return total, is_tail


@qd.func
def qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, value):
    """Minimum of value over the lanes of a 32-lane chunk that share slot i_slot.

    Returns the minimum and whether this lane is the segment's tail, as qd_segmented_sum does.
    """
    is_head, is_tail = qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next)
    total = qd.simt.subgroup.segmented_reduce_min_tiled(value, is_head, 5)
    return total, is_tail


@qd.func
def qd_segment_add(tid, i_slot, i_slot_prev, i_slot_next, i_row, value, sh_acc):
    """Add the segmented sum of value into entry i_row + i_slot of the shared array, from the tail lane of the segment.

    See qd_segmented_sum for the segments.
    """
    total, is_tail = qd_segmented_sum(tid, i_slot, i_slot_prev, i_slot_next, value)
    if is_tail:
        sh_acc[i_row + i_slot] = sh_acc[i_row + i_slot] + total


@qd.func
def qd_segment_min(tid, i_slot, i_slot_prev, i_slot_next, i_row, value, sh_min):
    """Fold the segmented minimum of value into entry i_row + i_slot of the shared array, from the tail lane of the
    segment.

    See qd_segmented_min for the segments.
    """
    total, is_tail = qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, value)
    if is_tail:
        sh_min[i_row + i_slot] = qd.min(sh_min[i_row + i_slot], total)
