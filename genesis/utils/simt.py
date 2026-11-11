import quadrants as qd

import genesis as gs
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.misc import qd_isfinite

# The base-2 logarithm of the lanes of a block, the unit of the subgroup helpers below
BLOCK_LOG2_LANES = 5


def requires_deterministic_accumulation() -> bool:
    """Return whether the accumulations of parallel threads must be independent of the order of the threads.

    These are the atomic adds, scatters and sums of terms from several threads. This holds with deterministic algorithms,
    except on a single thread of the CPU, which accumulates in a fixed order already.
    """
    return gs.use_deterministic_algorithms and not (gs.backend == gs.cpu and qd.cfg.cpu_max_num_threads == 1)


@qd.func
def qd_block_sum(value, log2_size: qd.template() = 5):
    """Sum a value over the lanes of a block, every lane receiving the bits the first lane of its block holds.

    A block is 2**log2_size consecutive lanes of the subgroup, so a subgroup wider than the block (64-lane AMDGPU
    wavefronts) holds several blocks, each summed on its own.
    """
    # The butterfly adds the same operands on the two lanes of every pair, but the compiler contracts the multiply that
    # produced a lane's own operand into that add, so the two lanes round differently, and a decision every lane takes
    # on the sum then diverges. The source lane differs between the blocks of a subgroup, which SPIR-V allows for a
    # shuffle alone, its broadcast requiring the same source lane on every lane of the subgroup.
    value_sum = qd.simt.subgroup.reduce_all_add_tiled(value, log2_size)
    return qd.simt.subgroup.shuffle(value_sum, qd.u32(qd.simt.subgroup.invocation_id() >> log2_size << log2_size))


@qd.func
def qd_block_scan(value):
    """Inclusive prefix sum of a value over the 32 lanes of a block, returning this lane's prefix and the block total.

    Every lane receives as total the bits the last lane of its block holds, so every lane carries the same running
    count across chunks. See qd_block_sum for the blocks and the source lane.
    """
    value_incl = qd.simt.subgroup.inclusive_add_tiled(value, 5)
    return value_incl, qd.simt.subgroup.shuffle(value_incl, qd.u32(qd.simt.subgroup.invocation_id() | 31))


@qd.func
def qd_block_min(value):
    """Minimum of a value over the 32 lanes of a block, every lane receiving the bits the first lane of its block holds.

    See qd_block_sum for the blocks and the source lane.
    """
    value_min = qd.simt.subgroup.reduce_all_min_tiled(value, 5)
    return qd.simt.subgroup.shuffle(value_min, qd.u32(qd.simt.subgroup.invocation_id() >> 5 << 5))


@qd.func
def qd_block_max(value):
    """Maximum of a value over the 32 lanes of a block, every lane receiving the bits the first lane of its block holds.

    See qd_block_sum for the blocks and the source lane.
    """
    value_max = qd.simt.subgroup.reduce_all_max_tiled(value, 5)
    return qd.simt.subgroup.shuffle(value_max, qd.u32(qd.simt.subgroup.invocation_id() >> 5 << 5))


@qd.func
def qd_block_shuffle(value, i_lane):
    """Read a value from lane i_lane of the 32 lanes of a block, every lane of the block receiving it.

    See qd_block_sum for the blocks and the source lane.
    """
    return qd.simt.subgroup.shuffle(value, qd.u32((qd.simt.subgroup.invocation_id() >> 5 << 5) + i_lane))


@qd.func
def qd_block_ballot(predicate):
    """Mask of the lanes of a block that a predicate holds on, every lane of the block receiving it.

    Bit i of the mask stands for lane i of the block, see qd_block_sum for the blocks. Every lane of the subgroup takes
    part in the vote.
    """
    lane_base = qd.simt.subgroup.invocation_id() >> 5 << 5
    ballot = qd.simt.subgroup.ballot(qd.cast(predicate, qd.i32))
    return qd.u32((ballot >> qd.u64(lane_base)) & qd.u64(0xFFFFFFFF))


@qd.func
def qd_block_all(predicate):
    """Whether a predicate holds on every one of the 32 lanes of a block, every lane of the block receiving the answer.

    See qd_block_sum for the blocks. A vote over the whole subgroup would also read the lanes beyond a block that no
    thread runs (the upper half of a 64-lane AMDGPU wavefront running a 32-lane block), whose values are undefined.
    """
    return qd.simt.subgroup.all_true_tiled(predicate, 5) != 0


@qd.func
def qd_block_any(predicate):
    """Whether a predicate holds on any of the 32 lanes of a block, every lane of the block receiving the answer.

    See qd_block_all for the lanes taking part in the vote.
    """
    return qd.simt.subgroup.any_true_tiled(predicate, 5) != 0


@qd.func
def qd_lane_vector_get(v, i, block_dim: qd.template()):
    """Read entry i of a vector spread over the lanes of a block, every lane receiving it.

    Lane tid holds entries tid + k * block_dim of the vector in v[k], so the entry is broadcast from lane
    i % block_dim, which reads it from its register v[i // block_dim].
    """
    k_owner = i // block_dim
    v_owner = v[0]
    for k in qd.static(range(1, v.n)):
        if k == k_owner:
            v_owner = v[k]
    return qd.simt.subgroup.broadcast(v_owner, qd.u32(i - k_owner * block_dim))


@qd.func
def qd_lane_local_get(values: qd.template(), i_k: int, block_dim: qd.template()):
    """Read an entry of the share of a buffer spread over the lanes of a block that this lane holds.

    The entry read is i_k, see qd_lane_vector_get for the layout. The entries of a vector are scalars, those of a matrix
    its rows. With several lanes, the entry is selected among the few entries of the share by a static unroll, which
    keeps the share in registers where a runtime index would place it in local memory. A single lane holds the whole
    buffer and indexes it. The buffer is read in place, as the share of a single lane is the whole buffer.
    """
    if qd.static(len(values.get_shape()) == 2):
        row = values[0, :]
        if qd.static(block_dim > 1):
            for k in qd.static(range(1, values.n)):
                if k == i_k:
                    row = values[k, :]
        else:
            row = qd.Vector([values[i_k, j] for j in qd.static(range(values.m))])
        return row
    else:
        value = values[0]
        if qd.static(block_dim > 1):
            for k in qd.static(range(1, values.n)):
                if k == i_k:
                    value = values[k]
        else:
            value = values[i_k]
        return value


@qd.func
def qd_lane_local_set(values: qd.template(), i_k: int, value, block_dim: qd.template()):
    """Replace in place an entry of the share of a buffer spread over the lanes of a block that this lane holds.

    Entry i_k takes value. See qd_lane_vector_get for the layout and qd_lane_local_get for the selection.
    """
    if qd.static(len(values.get_shape()) == 2):
        if qd.static(block_dim > 1):
            for k in qd.static(range(values.n)):
                if k == i_k:
                    values[k, :] = value
        else:
            for j in qd.static(range(values.m)):
                values[i_k, j] = value[j]
    elif qd.static(block_dim > 1):
        for k in qd.static(range(values.n)):
            if k == i_k:
                values[k] = value
    else:
        values[i_k] = value


@qd.func
def qd_lane_slot_read(values: qd.template(), k: int, block_dim: qd.template()):
    """Read a slot of a buffer spread over the lanes of a block, every lane of the block receiving it.

    The slot read is k, a scalar for a vector and a row for a matrix (see qd_lane_local_get). See qd_lane_vector_get for
    the layout. Every lane of the block must call it with the same k.
    """
    if qd.static(len(values.get_shape()) == 2):
        row = values[0, :]
        if qd.static(block_dim > 1):
            row = qd.Vector([qd_lane_vector_get(values[:, j], k, block_dim) for j in qd.static(range(values.m))])
        else:
            row = qd.Vector([values[k, j] for j in qd.static(range(values.m))])
        return row
    else:
        value = values[0]
        if qd.static(block_dim > 1):
            value = qd_lane_vector_get(values, k, block_dim)
        else:
            value = values[k]
        return value


@qd.func
def qd_lane_slot_write(values: qd.template(), k: int, value, tid: int, block_dim: qd.template()):
    """Replace in place a slot of a buffer spread over the lanes of a block, on the lane holding it.

    Slot k takes value. See qd_lane_vector_get for the layout.
    """
    if k % block_dim == tid:
        qd_lane_local_set(values, k // block_dim, value, block_dim)


@qd.func
def qd_lane_slot_min(k: int, n_slots: int, n_slot_rows: qd.template(), block_dim: qd.template()):
    """Lowest of the slots the lanes of a block name, every lane of the block receiving it.

    Each lane names one of the slots it holds, or n_slots for none (see qd_lane_vector_get for the layout). The slots
    are taken row by row, the lowest lane naming a slot of the lowest row holding the lowest slot, which a ballot of the
    lanes finds where a min-reduction would take a shuffle per halving.
    """
    k_min = n_slots
    for i_k in qd.static(range(n_slot_rows)):
        lanes = qd_block_ballot(k < n_slots and k // block_dim == i_k)
        if k_min == n_slots and lanes != 0:
            k_min = i_k * block_dim + qd.cast(qd.math.ffs(lanes), gs.qd_int) - 1
    return k_min


@qd.func
def qd_lane_mask_pop_first(masks: qd.template(), n_slots: int, block_dim: qd.template()):
    """Lowest slot a mask of lane-spread slots holds, cleared from the mask in place.

    Bit i of row r of the mask stands for slot r * block_dim + i, see qd_lane_vector_get for the layout. The slot is
    n_slots when the mask holds none.
    """
    k = n_slots
    for i_k in qd.static(range(masks.n)):
        if k == n_slots and masks[i_k] != 0:
            k = i_k * block_dim + qd.cast(qd.math.ffs(masks[i_k]), gs.qd_int) - 1
            masks[i_k] = masks[i_k] & (masks[i_k] - 1)
    return k


@qd.func
def qd_is_lane_slot_seen(values: qd.template(), n_values: int, value: int, tid: int, block_dim: qd.template()):
    """Whether a value is among the leading slots of a buffer spread over the lanes of a block.

    The first n_values slots are searched. See qd_lane_vector_get for the layout. Every lane of the block must call it
    with the same arguments but tid.
    """
    n_slots = qd.static(values.n)
    is_seen = False
    for i_k in range(n_slots):
        if i_k * block_dim + tid < n_values and qd_lane_local_get(values, i_k, block_dim) == value:
            is_seen = True
    if qd.static(block_dim > 1):
        is_seen = qd_block_any(is_seen)
    return is_seen


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


# The values summed exactly whatever the order of their terms, and the bins each of them is cut into (see ExactSums)
EXACT_SUM_SLOTS = Size("exact_sum_slots")
EXACT_SUM_BINS = Size("exact_sum_bins")


class ExactSums(Data, kind=DataKind.SCRATCH, shape=(EXACT_SUM_SLOTS, EXACT_SUM_BINS, ENVS)):
    """
    Sums of floating-point terms added from parallel threads, whose result is the same whatever the order of the terms.

    A term is cut exactly into integer multiples of the unit of each bin, from the highest bin down, and each bin adds
    its multiples as a 64-bit integer, which integer addition makes independent of the order (see func_exact_sum_add).
    With units 2^b apart, a term adds less than 2^b to every bin but the highest, so a bin holds 2^(63 - b) terms before
    it overflows, and the highest bin holds terms below 2^(63 - b) times its unit. The part of a term below the unit of
    the lowest bin is dropped, the same whatever the order.
    """

    # The low and high 32 bits of the 64-bit integer sum of each bin
    bins_lo: qd.Tensor = field(int)
    bins_hi: qd.Tensor = field(int)
    # The unit of each bin, a power of two, in increasing order
    bins_unit: qd.Tensor = field(float, shape=(EXACT_SUM_BINS,), kind=DataKind.CONSTANT)
    # The sum of the terms of each slot that are 'inf' or 'nan', which no bin can hold and which the sum takes, 0 without
    # any. It is the same whatever the order of the terms, 'nan' absorbing every term and the infinities of opposite
    # signs giving 'nan'.
    slots_nonfinite: qd.Tensor = field(float, shape=(EXACT_SUM_SLOTS, ENVS))


@qd.func
def func_exact_sum_add(i_s: int, i_b: int, value: float, exact_sums: ExactSums):
    """
    Add a term to a slot of exact sums.

    The cut of the term into bins is exact: from the highest bin down, the integer part of the term at the unit of a bin
    holds the leading bits of its mantissa, and the remainder the trailing ones. A bin adds its 64 bits as two 32-bit
    halves, the low half carrying into the high one.
    """
    if not qd_isfinite(value):
        qd.atomic_add(exact_sums.slots_nonfinite[i_s, i_b], value)
    remainder = qd.select(qd_isfinite(value), value, 0.0)
    for k_ in range(exact_sums.bins_unit.shape[0]):
        k = exact_sums.bins_unit.shape[0] - 1 - k_
        unit = exact_sums.bins_unit[k]
        count = qd.cast(remainder / unit, qd.i64)
        if count != 0:
            remainder = remainder - qd.cast(count, gs.qd_float) * unit
            count_lo = qd.cast(count & qd.i64(0xFFFFFFFF), qd.u32)
            lo_prev = qd.cast(qd.atomic_add(exact_sums.bins_lo[i_s, k, i_b], qd.cast(count_lo, gs.qd_int)), qd.u32)
            carry = qd.select(lo_prev + count_lo < lo_prev, 1, 0)
            qd.atomic_add(exact_sums.bins_hi[i_s, k, i_b], qd.cast(count >> 32, gs.qd_int) + carry)


@qd.func
def func_exact_sum_take(i_s: int, i_b: int, exact_sums: ExactSums):
    """
    Return the sum of a slot of exact sums, from the highest bin down, and empty the slot for the next sum.

    The sum is 'inf' or 'nan' if a term was, for the reader to detect.
    """
    total = gs.qd_float(0.0)
    for k_ in range(exact_sums.bins_unit.shape[0]):
        k = exact_sums.bins_unit.shape[0] - 1 - k_
        count = (qd.cast(exact_sums.bins_hi[i_s, k, i_b], qd.i64) << 32) | qd.cast(
            qd.cast(exact_sums.bins_lo[i_s, k, i_b], qd.u32), qd.i64
        )
        total = total + qd.cast(count, gs.qd_float) * exact_sums.bins_unit[k]
        exact_sums.bins_lo[i_s, k, i_b] = 0
        exact_sums.bins_hi[i_s, k, i_b] = 0
    total = total + exact_sums.slots_nonfinite[i_s, i_b]
    exact_sums.slots_nonfinite[i_s, i_b] = 0.0
    return total
