from __future__ import annotations

import quadrants as qd


@qd.func
def _segment_metadata(
    gid,
    n,
    seg_ids: qd.template(),
    sorted_keys: qd.template(),
    sentinel: qd.template(),
):
    lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
    valid = gid < n and sorted_keys[gid] != sentinel
    segment = qd.i32(0)
    previous = qd.i32(-1)
    following = qd.i32(-2)
    if valid:
        segment = qd.i32(seg_ids[gid])
        if lane > 0 and gid > 0:
            previous = qd.i32(seg_ids[gid - 1])
        if lane < 31 and gid + 1 < n:
            following = qd.i32(seg_ids[gid + 1])
    is_head = qd.i32(not valid or lane == 0 or previous != segment)
    is_tail = valid and (lane == 31 or gid == n - 1 or following != segment)
    return qd.Vector(
        [
            qd.i32(valid),
            segment,
            is_head,
            qd.i32(is_tail),
        ],
        dt=qd.i32,
    )


@qd.func
def _head_segmented_reduce_add(
    value,
    tail_flag,
):
    lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
    tail_mask = qd.u32(qd.simt.subgroup.ballot(qd.i32(tail_flag != 0))) | qd.u32(0x80000000)
    tails_at_or_above = tail_mask & (qd.u32(0xFFFFFFFF) << qd.u32(lane))
    lowest_tail = tails_at_or_above & (qd.u32(0) - tails_at_or_above)
    tail_lane = qd.i32(31) - qd.i32(qd.math.clz(lowest_tail))
    distance = tail_lane - lane
    for level in qd.static(range(5)):
        offset = qd.static(1 << level)
        partner = qd.simt.subgroup.shuffle_down(
            value,
            qd.u32(offset),
        )
        if distance >= offset:
            value = value + partner
    return value


@qd.func(requires_top_level=True)
def fast_segmented_reduce_doublet(
    seg_ids: qd.template(),
    sorted_perm: qd.template(),
    sorted_keys: qd.template(),
    values: qd.template(),
    output: qd.template(),
    n: qd.template(),
    padded_n: qd.template(),
    capacity: qd.template(),
):
    qd.loop_config(name="fast_segmented_reduce_doublet", block_dim=256)
    for gid in range(capacity):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
        if gid - lane < padded_n[()]:
            metadata = _segment_metadata(
                gid,
                n[()],
                seg_ids,
                sorted_keys,
                qd.u32(0xFFFFFFFF),
            )
            source = qd.i32(0)
            if metadata[0] != 0:
                source = qd.i32(sorted_perm[gid])
            for component in qd.static(range(3)):
                value = qd.f64(0.0)
                if metadata[0] != 0:
                    value = values[source, component]
                reduced = _head_segmented_reduce_add(
                    value,
                    metadata[3],
                )
                if metadata[0] != 0 and metadata[2] != 0:
                    qd.atomic_add(
                        output[metadata[1], component],
                        reduced,
                    )


@qd.func(requires_top_level=True)
def fast_segmented_reduce_triplet(
    seg_ids: qd.template(),
    sorted_perm: qd.template(),
    sorted_keys: qd.template(),
    values: qd.template(),
    output: qd.template(),
    n: qd.template(),
    padded_n: qd.template(),
    capacity: qd.template(),
):
    qd.loop_config(name="fast_segmented_reduce_triplet", block_dim=256)
    for gid in range(capacity):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
        if gid - lane < padded_n[()]:
            metadata = _segment_metadata(
                gid,
                n[()],
                seg_ids,
                sorted_keys,
                qd.u64(0xFFFFFFFFFFFFFFFF),
            )
            source = qd.i32(0)
            if metadata[0] != 0:
                source = qd.i32(sorted_perm[gid])
            for component in qd.static(range(9)):
                value = qd.f64(0.0)
                if metadata[0] != 0:
                    row = qd.static(component // 3)
                    column = qd.static(component % 3)
                    value = values[source, row, column]
                reduced = _head_segmented_reduce_add(
                    value,
                    metadata[3],
                )
                if metadata[0] != 0 and metadata[2] != 0:
                    row = qd.static(component // 3)
                    column = qd.static(component % 3)
                    qd.atomic_add(
                        output[
                            metadata[1],
                            row,
                            column,
                        ],
                        reduced,
                    )


@qd.func(requires_top_level=True)
def fast_segmented_reduce_body(
    seg_ids: qd.template(),
    sorted_perm: qd.template(),
    sorted_keys: qd.template(),
    values: qd.template(),
    output: qd.template(),
    n: qd.template(),
    padded_n: qd.template(),
    capacity: qd.template(),
):
    qd.loop_config(name="fast_segmented_reduce_body", block_dim=256)
    for gid in range(capacity):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
        if gid - lane < padded_n[()]:
            metadata = _segment_metadata(
                gid,
                n[()],
                seg_ids,
                sorted_keys,
                qd.u64(0xFFFFFFFFFFFFFFFF),
            )
            source = qd.i32(0)
            if metadata[0] != 0:
                source = qd.i32(sorted_perm[gid])
            for component in qd.static(range(9)):
                value = qd.f64(0.0)
                if metadata[0] != 0:
                    value = values[source * 9 + component]
                reduced = _head_segmented_reduce_add(
                    value,
                    metadata[3],
                )
                if metadata[0] != 0 and metadata[2] != 0:
                    qd.atomic_add(
                        output[metadata[1] * 9 + component],
                        reduced,
                    )
