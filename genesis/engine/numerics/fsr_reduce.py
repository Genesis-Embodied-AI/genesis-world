import quadrants as qd

import genesis as gs


@qd.func
def _segment_metadata(
    gid: int,
    n,
    seg_ids: qd.template(),
    sorted_keys: qd.template(),
    sentinel: qd.template(),
):
    lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
    is_valid = gid < n and sorted_keys[gid] != sentinel
    segment = qd.i32(0)
    previous_segment = qd.i32(-1)
    next_segment = qd.i32(-2)
    if is_valid:
        segment = qd.i32(seg_ids[gid])
        if lane > 0 and gid > 0:
            previous_segment = qd.i32(seg_ids[gid - 1])
        if lane < 31 and gid + 1 < n:
            next_segment = qd.i32(seg_ids[gid + 1])
    is_head = qd.i32(not is_valid or lane == 0 or previous_segment != segment)
    is_tail = is_valid and (lane == 31 or gid == n - 1 or next_segment != segment)
    return qd.Vector([qd.i32(is_valid), segment, is_head, qd.i32(is_tail)], dt=qd.i32)


@qd.func
def head_segmented_reduce_add(value, tail_flag):
    lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
    tail_mask = qd.u32(qd.simt.subgroup.ballot(qd.i32(tail_flag != 0))) | qd.u32(0x80000000)
    tails_at_or_above = tail_mask & (qd.u32(0xFFFFFFFF) << qd.u32(lane))
    lowest_tail = tails_at_or_above & (qd.u32(0) - tails_at_or_above)
    tail_lane = qd.i32(31) - qd.i32(qd.math.clz(lowest_tail))
    distance = tail_lane - lane
    for level in qd.static(range(5)):
        offset = qd.static(1 << level)
        partner = qd.simt.subgroup.shuffle_down(value, qd.u32(offset))
        if distance >= offset:
            value = value + partner
    return value


@qd.func(requires_top_level=True)
def func_fsr_reduce(
    seg_ids: qd.template(),
    sorted_perm: qd.template(),
    sorted_keys: qd.template(),
    values: qd.template(),
    output: qd.template(),
    n: qd.template(),
    padded_n: qd.template(),
    capacity: qd.template(),
    value_type: qd.template(),
    block_scalar_count: qd.template(),
):
    """Sum the flattened blocks of each run of equal sorted keys into the output entry of its segment.

    On a GPU, a warp reduces the runs inside its 32 consecutive sorted entries with a head-segmented shuffle tree, and
    the head of each run writes alone, so a run spanning several warps takes one atomic add per warp it spans. A CPU
    runs few threads and has no lanes to shuffle between, so each run is the work of one thread: the threads split the
    entries, and the thread holding the first entry of a run sums the blocks of every entry of the run in one ordered
    walk, as the only writer of its segment.
    """
    if qd.static(gs.backend == gs.cpu):
        qd.loop_config(name="fsr_reduce_cpu")
        for gid in range(n[()]):
            if sorted_keys[gid] != qd.u64(0xFFFFFFFFFFFFFFFF):
                segment = qd.i32(seg_ids[gid])
                if gid == 0 or qd.i32(seg_ids[gid - 1]) != segment:
                    total = qd.Vector.zero(value_type, block_scalar_count)
                    entry = gid
                    is_in_run = True
                    while is_in_run:
                        source = qd.i32(sorted_perm[entry]) * block_scalar_count
                        for component in qd.static(range(block_scalar_count)):
                            total[component] += values[source + component]
                        entry += 1
                        # Reading seg_ids only below n: a kernel 'and' does not short-circuit
                        is_in_run = False
                        if entry < n[()]:
                            is_in_run = qd.i32(seg_ids[entry]) == segment
                    for component in qd.static(range(block_scalar_count)):
                        output[segment * block_scalar_count + component] += total[component]
    else:
        qd.loop_config(name="fsr_reduce", block_dim=256)
        for gid in range(qd.min(((padded_n[()] + 31) // 32) * 32, capacity)):
            lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
            if gid - lane < padded_n[()]:
                metadata = _segment_metadata(gid, n[()], seg_ids, sorted_keys, qd.u64(0xFFFFFFFFFFFFFFFF))
                source = qd.i32(0)
                if metadata[0] != 0:
                    source = qd.i32(sorted_perm[gid])
                for component in qd.static(range(block_scalar_count)):
                    value = value_type(0.0)
                    if metadata[0] != 0:
                        value = values[source * block_scalar_count + component]
                    reduced = head_segmented_reduce_add(value, metadata[3])
                    if metadata[0] != 0 and metadata[2] != 0:
                        qd.atomic_add(output[metadata[1] * block_scalar_count + component], reduced)
