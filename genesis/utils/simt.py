import torch

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


def has_mirrored_shuffles() -> bool:
    """Return whether the subgroup shuffles of the active GPU hand the upper half of a wavefront the wrong half's lanes.

    On RDNA3 and RDNA4 AMD GPUs (gfx11, gfx12), whose wavefront of 64 lanes runs as two halves of 32, a shuffle hands
    lane 32 and above the value of the lane mirrored across the halves, lane i ^ 32 when asked for lane i. The block
    helpers of this module then request the mirrored lane on the upper half, which gets every lane its intended value.
    """
    # FIXME: quadrants#1095 - shuffles return lane target ^ 32 on the upper half of an RDNA wavefront. Every read takes
    # one more integer operation there.
    if gs.backend != gs.amdgpu:
        return False
    arch_name = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    return arch_name.startswith(("gfx11", "gfx12"))


@qd.func
def _qd_lane_read(value, i_lane, is_mirrored: qd.template()):
    """Read a value from lane i_lane of the subgroup, the lane mirrored across the halves on the upper half if asked."""
    i_lane_read = i_lane
    if qd.static(is_mirrored):
        i_lane_read = i_lane ^ (qd.simt.subgroup.invocation_id() & 32)
    return qd.simt.subgroup.shuffle(value, qd.u32(i_lane_read))


@qd.func
def _qd_add(a, b):
    return a + b


@qd.func
def _qd_block_reduce(value, op: qd.template(), log2_size: qd.template()):
    """Reduce a value over the lanes of a block by a butterfly of shuffles, every lane receiving the first lane's bits.

    The quadrants reductions read the lanes of the other half of an RDNA wavefront wrongly (see has_mirrored_shuffles),
    which this butterfly corrects. See qd_block_sum for the blocks.
    """
    lane = qd.simt.subgroup.invocation_id()
    for i in qd.static(range(log2_size)):
        value = op(value, _qd_lane_read(value, lane ^ (1 << i), is_mirrored=True))
    return _qd_lane_read(value, lane >> log2_size << log2_size, is_mirrored=True)


@qd.func
def qd_block_sum(value, log2_size: qd.template() = -1):
    """Sum a value over the lanes of a block, every lane receiving the bits the first lane of its block holds.

    A block is 2**log2_size consecutive lanes of the subgroup, the whole subgroup for a negative log2_size (32 lanes on
    CUDA and Metal, 64 on AMDGPU), so a subgroup wider than the block holds several blocks, each summed on its own.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    if qd.static(has_mirrored_shuffles()):
        return _qd_block_reduce(value, _qd_add, LOG2_SIZE)
    # The butterfly adds the same operands on the two lanes of every pair, but the compiler contracts the multiply that
    # produced a lane's own operand into that add, so the two lanes round differently, and a decision every lane takes
    # on the sum then diverges. The source lane differs between the blocks of a subgroup, which SPIR-V allows for a
    # shuffle alone, its broadcast requiring the same source lane on every lane of the subgroup.
    value_sum = qd.simt.subgroup.reduce_all_add_tiled(value, LOG2_SIZE)
    return qd.simt.subgroup.shuffle(value_sum, qd.u32(qd.simt.subgroup.invocation_id() >> LOG2_SIZE << LOG2_SIZE))


@qd.func
def qd_block_scan(value, log2_size: qd.template() = -1):
    """Inclusive prefix sum of a value over the lanes of a block, returning this lane's prefix and the block total.

    Every lane receives as total the bits the last lane of its block holds, so every lane carries the same running
    count across chunks. See qd_block_sum for the blocks and the source lane.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    IS_MIRRORED = qd.static(has_mirrored_shuffles())
    value_incl = value
    if qd.static(IS_MIRRORED):
        lane = qd.simt.subgroup.invocation_id()
        lane_in_block = lane & ((1 << LOG2_SIZE) - 1)
        for i in qd.static(range(LOG2_SIZE)):
            value_prev = _qd_lane_read(value_incl, lane - (1 << i), IS_MIRRORED)
            if lane_in_block >= (1 << i):
                value_incl = value_prev + value_incl
    else:
        value_incl = qd.simt.subgroup.inclusive_add_tiled(value, LOG2_SIZE)
    lane_last = qd.simt.subgroup.invocation_id() | ((1 << LOG2_SIZE) - 1)
    return value_incl, _qd_lane_read(value_incl, lane_last, IS_MIRRORED)


@qd.func
def qd_block_min(value, log2_size: qd.template() = -1):
    """Minimum of a value over the lanes of a block, every lane receiving the bits the first lane of its block holds.

    See qd_block_sum for the blocks and the source lane.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    if qd.static(has_mirrored_shuffles()):
        return _qd_block_reduce(value, qd.min, LOG2_SIZE)
    value_min = qd.simt.subgroup.reduce_all_min_tiled(value, LOG2_SIZE)
    return qd.simt.subgroup.shuffle(value_min, qd.u32(qd.simt.subgroup.invocation_id() >> LOG2_SIZE << LOG2_SIZE))


@qd.func
def qd_block_max(value, log2_size: qd.template() = -1):
    """Maximum of a value over the lanes of a block, every lane receiving the bits the first lane of its block holds.

    See qd_block_sum for the blocks and the source lane.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    if qd.static(has_mirrored_shuffles()):
        return _qd_block_reduce(value, qd.max, LOG2_SIZE)
    value_max = qd.simt.subgroup.reduce_all_max_tiled(value, LOG2_SIZE)
    return qd.simt.subgroup.shuffle(value_max, qd.u32(qd.simt.subgroup.invocation_id() >> LOG2_SIZE << LOG2_SIZE))


@qd.func
def qd_block_shuffle(value, i_lane, log2_size: qd.template() = -1):
    """Read a value from lane i_lane of the lanes of a block, every lane of the block receiving it.

    See qd_block_sum for the blocks and the source lane.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    lane = (qd.simt.subgroup.invocation_id() >> LOG2_SIZE << LOG2_SIZE) + i_lane
    return _qd_lane_read(value, lane, qd.static(has_mirrored_shuffles()))


@qd.func
def qd_subgroup_first(value):
    """Read a value from the first lane of the subgroup, every lane receiving it."""
    if qd.static(has_mirrored_shuffles()):
        return _qd_lane_read(value, 0, is_mirrored=True)
    return qd.simt.subgroup.broadcast(value, qd.u32(0))


@qd.func
def qd_block_ballot(predicate, log2_size: qd.template() = -1):
    """Mask of the lanes of a block that a predicate holds on, every lane of the block receiving it.

    Bit i of the mask stands for lane i of the block, see qd_block_sum for the blocks. The mask is a u32 for a block of
    at most 32 lanes, a u64 for a block of 64 lanes. Every lane of the subgroup takes part in the vote.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    MASK_DTYPE = qd.u32 if LOG2_SIZE <= 5 else qd.u64
    lane_base = qd.simt.subgroup.invocation_id() >> LOG2_SIZE << LOG2_SIZE
    ballot = qd.simt.subgroup.ballot(qd.cast(predicate, qd.i32)) >> qd.u64(lane_base)
    if qd.static(LOG2_SIZE <= 5):
        ballot = ballot & qd.u64((1 << (1 << LOG2_SIZE)) - 1)
    return qd.cast(ballot, MASK_DTYPE)


@qd.func
def qd_block_all(predicate, log2_size: qd.template() = -1):
    """Whether a predicate holds on every lane of a block, every lane of the block receiving the answer.

    See qd_block_sum for the blocks. A vote over the whole subgroup would also read the lanes beyond a block that no
    thread runs (the upper half of a 64-lane AMDGPU wavefront running a 32-lane block), whose values are undefined.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    if qd.static(has_mirrored_shuffles()):
        return qd_block_ballot(not predicate, LOG2_SIZE) == 0
    return qd.simt.subgroup.all_true_tiled(predicate, LOG2_SIZE) != 0


@qd.func
def qd_block_any(predicate, log2_size: qd.template() = -1):
    """Whether a predicate holds on any lane of a block, every lane of the block receiving the answer.

    See qd_block_all for the lanes taking part in the vote.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    if qd.static(has_mirrored_shuffles()):
        return qd_block_ballot(predicate, LOG2_SIZE) != 0
    return qd.simt.subgroup.any_true_tiled(predicate, LOG2_SIZE) != 0


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
    if qd.static(has_mirrored_shuffles()):
        return _qd_lane_read(v_owner, i - k_owner * block_dim, is_mirrored=True)
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
    LOG2_BLOCK_DIM = qd.static(block_dim.bit_length() - 1)
    k_min = n_slots
    for i_k in qd.static(range(n_slot_rows)):
        lanes = qd_block_ballot(k < n_slots and k // block_dim == i_k, LOG2_BLOCK_DIM)
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
        is_seen = qd_block_any(is_seen, qd.static(block_dim.bit_length() - 1))
    return is_seen


@qd.func
def qd_slot_neighbors(tid, i_slot):
    """Slots of the two neighboring lanes of a chunk of lanes, the inputs of qd_segment_bounds."""
    if qd.static(has_mirrored_shuffles()):
        lane = qd.simt.subgroup.invocation_id()
        return _qd_lane_read(i_slot, lane - 1, is_mirrored=True), _qd_lane_read(i_slot, lane + 1, is_mirrored=True)
    i_slot_prev = qd.simt.subgroup.shuffle_up(i_slot, qd.u32(1))
    i_slot_next = qd.simt.subgroup.shuffle_down(i_slot, qd.u32(1))
    return i_slot_prev, i_slot_next


@qd.func
def qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next, log2_size: qd.template() = -1):
    """Whether lane tid of a chunk of lanes opens and whether it closes the run of lanes sharing slot i_slot.

    A chunk is a block of lanes, see qd_block_sum. i_slot_prev and i_slot_next are the slots of the neighboring lanes, a
    lane holding no item carrying slot -1. The chunk boundary closes every run, and the tail lane alone writes a segment
    out so segments accumulate in chunk order.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    is_head = 1
    if tid > 0 and i_slot_prev == i_slot:
        is_head = 0
    is_tail = i_slot >= 0 and (tid == (1 << LOG2_SIZE) - 1 or i_slot_next != i_slot)
    return is_head, is_tail


@qd.func
def _qd_segmented_reduce(value, is_head, op: qd.template(), log2_size: qd.template()):
    """Reduce a value over the lanes of a block from the last head lane up to this lane, by shuffles.

    The first lane of the block counts as a head. The quadrants segmented reductions read the lanes of the other half of
    an RDNA wavefront wrongly (see has_mirrored_shuffles), which this scan corrects.
    """
    MASK_BITS = qd.static(32 if log2_size <= 5 else 64)
    MASK_DTYPE = qd.u32 if log2_size <= 5 else qd.u64
    lane = qd.simt.subgroup.invocation_id()
    lane_in_block = lane & ((1 << log2_size) - 1)
    heads = qd_block_ballot(is_head != 0, log2_size)
    heads_below = heads & ((MASK_DTYPE(2) << MASK_DTYPE(lane_in_block)) - MASK_DTYPE(1))
    distance = lane_in_block
    if heads_below != 0:
        distance = lane_in_block - (MASK_BITS - 1 - qd.cast(qd.math.clz(heads_below), qd.i32))
    for i in qd.static(range(log2_size)):
        value_prev = _qd_lane_read(value, lane - (1 << i), is_mirrored=True)
        if distance >= (1 << i):
            value = op(value, value_prev)
    return value


@qd.func
def qd_segmented_sum(tid, i_slot, i_slot_prev, i_slot_next, value, log2_size: qd.template() = -1):
    """Sum value over the lanes of a chunk of lanes that share slot i_slot.

    Returns the total and whether this lane is the segment's tail (see qd_segment_bounds).
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    is_head, is_tail = qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next, LOG2_SIZE)
    if qd.static(has_mirrored_shuffles()):
        return _qd_segmented_reduce(value, is_head, _qd_add, LOG2_SIZE), is_tail
    total = qd.simt.subgroup.segmented_reduce_add_tiled(value, is_head, LOG2_SIZE)
    return total, is_tail


@qd.func
def qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, value, log2_size: qd.template() = -1):
    """Minimum of value over the lanes of a chunk of lanes that share slot i_slot.

    Returns the minimum and whether this lane is the segment's tail, as qd_segmented_sum does.
    """
    LOG2_SIZE = qd.static(qd.simt.subgroup.log2_group_size() if log2_size < 0 else log2_size)
    is_head, is_tail = qd_segment_bounds(tid, i_slot, i_slot_prev, i_slot_next, LOG2_SIZE)
    if qd.static(has_mirrored_shuffles()):
        return _qd_segmented_reduce(value, is_head, qd.min, LOG2_SIZE), is_tail
    total = qd.simt.subgroup.segmented_reduce_min_tiled(value, is_head, LOG2_SIZE)
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
