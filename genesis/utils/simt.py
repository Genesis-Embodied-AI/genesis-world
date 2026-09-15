import quadrants as qd


@qd.func
def qd_block_sum(value):
    """Sum value over the 32 lanes of a block, every lane receiving the bits lane 0 holds."""
    # The butterfly adds the same operands on the two lanes of every pair, but the compiler contracts the multiply that
    # produced a lane's own operand into that add, so the two lanes round differently, and a decision every lane takes
    # on the sum then diverges.
    return qd.simt.subgroup.broadcast(qd.simt.subgroup.reduce_all_add_tiled(value, 5), qd.u32(0))


@qd.func
def qd_block_min(value):
    """Minimum of value over the 32 lanes of a block, every lane receiving the bits lane 0 holds (see qd_block_sum)."""
    return qd.simt.subgroup.broadcast(qd.simt.subgroup.reduce_all_min_tiled(value, 5), qd.u32(0))


@qd.func
def qd_segmented_sum(tid, i_slot, i_slot_prev, i_slot_next, value):
    """Sum value over the lanes of a 32-lane chunk that share slot i_slot.

    i_slot_prev and i_slot_next are the slots of the neighboring lanes; a lane holding no item carries slot -1. Returns
    the segment's total and whether this lane is the segment's tail, the last lane of the chunk holding the slot, which
    alone writes the total out so the accumulation order is the chunk order.
    """
    is_head = 1
    if tid > 0 and i_slot_prev == i_slot:
        is_head = 0
    total = qd.simt.subgroup.segmented_reduce_add_tiled(value, is_head, 5)
    is_tail = i_slot >= 0 and (tid == 31 or i_slot_next != i_slot)
    return total, is_tail


@qd.func
def qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, value):
    """Minimum of value over the lanes of a 32-lane chunk that share slot i_slot (see qd_segmented_sum)."""
    is_head = 1
    if tid > 0 and i_slot_prev == i_slot:
        is_head = 0
    total = qd.simt.subgroup.segmented_reduce_min_tiled(value, is_head, 5)
    is_tail = i_slot >= 0 and (tid == 31 or i_slot_next != i_slot)
    return total, is_tail


@qd.func
def qd_segment_add(tid, i_slot, i_slot_prev, i_slot_next, value, sh_acc, i_row):
    """Segmented sum of value (see qd_segmented_sum), the tail lane adding the segment's total into entry
    i_row + i_slot of the shared array."""
    total, is_tail = qd_segmented_sum(tid, i_slot, i_slot_prev, i_slot_next, value)
    if is_tail:
        sh_acc[i_row + i_slot] = sh_acc[i_row + i_slot] + total


@qd.func
def qd_segment_min(tid, i_slot, i_slot_prev, i_slot_next, value, sh_min, i_row):
    """Segmented minimum of value (see qd_segmented_min), the tail lane folding the segment's minimum into entry
    i_row + i_slot of the shared array."""
    total, is_tail = qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, value)
    if is_tail:
        sh_min[i_row + i_slot] = qd.min(sh_min[i_row + i_slot], total)
