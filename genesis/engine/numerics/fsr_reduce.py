import quadrants as qd

import genesis as gs


@qd.func
def head_segmented_reduce_add(value, tail_flag, is_max: qd.template() = False):
    """Sum, or with `is_max` the maximum, of the values of the run of lanes of every head lane of a warp, a run ending
    at its lane of `tail_flag`."""
    lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
    tail_mask = qd.u32(qd.simt.subgroup.ballot(gs.qd_int(tail_flag != 0))) | qd.u32(0x80000000)
    tails_at_or_above = tail_mask & (qd.u32(0xFFFFFFFF) << qd.u32(lane))
    lowest_tail = tails_at_or_above & (qd.u32(0) - tails_at_or_above)
    tail_lane = gs.qd_int(31) - gs.qd_int(qd.math.clz(lowest_tail))
    distance = tail_lane - lane
    for level in qd.static(range(5)):
        offset = qd.static(1 << level)
        partner = qd.simt.subgroup.shuffle_down(value, qd.u32(offset))
        if distance >= offset:
            if qd.static(is_max):
                value = qd.max(value, partner)
            else:
                value = value + partner
    return value


@qd.func
def _key_runs(key):
    """Whether the lane starts and ends a run of equal keys in its warp."""
    lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
    previous_key = qd.simt.subgroup.shuffle(key, qd.u32(qd.max(lane - 1, 0)))
    next_key = qd.simt.subgroup.shuffle(key, qd.u32(qd.min(lane + 1, 31)))
    return lane == 0 or key != previous_key, lane == 31 or key != next_key


@qd.func
def add_by_key(value, key, out: qd.template()):
    """Add the value of an entry into the entry of its key in `out`, an entry of key -1 adding nothing.

    On a GPU, the lanes of a warp hold consecutive entries sorted by key, every lane of the warp takes part, and each
    run of one key adds its sum by one atomic. A CPU has no lanes, and every entry adds alone."""
    if qd.static(gs.backend == gs.cpu):
        if key >= 0:
            qd.atomic_add(out[key], value)
    else:
        is_head, is_tail = _key_runs(key)
        total = head_segmented_reduce_add(value, is_tail)
        if key >= 0 and is_head:
            qd.atomic_add(out[key], total)


@qd.func
def max_by_key(value, key, out: qd.template()):
    """Maximum counterpart of add_by_key."""
    if qd.static(gs.backend == gs.cpu):
        if key >= 0:
            qd.atomic_max(out[key], value)
    else:
        is_head, is_tail = _key_runs(key)
        total = head_segmented_reduce_add(value, is_tail, is_max=True)
        if key >= 0 and is_head:
            qd.atomic_max(out[key], total)


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
    """Sum the flattened blocks of each run of equal sorted keys into the output entry of its segment (see
    add_by_key)."""
    qd.loop_config(name="fsr_reduce", block_dim=256)
    for gid in range(qd.min(((padded_n[()] + 31) // 32) * 32, capacity)):
        segment = -1
        source = 0
        if gid < n[()]:
            if sorted_keys[gid] != qd.u64(0xFFFFFFFFFFFFFFFF):
                segment = gs.qd_int(seg_ids[gid])
                source = gs.qd_int(sorted_perm[gid]) * block_scalar_count
        for component in qd.static(range(block_scalar_count)):
            value = value_type(0.0)
            key = -1
            if segment >= 0:
                value = values[source + component]
                key = segment * block_scalar_count + component
            add_by_key(value, key, output)
