from __future__ import annotations

import numpy as np
import quadrants as qd

_RADIX_DIGITS = 256
_BLOCK_THREADS = 256
_ITEMS_PER_THREAD = 24


@qd.func
def _rank_warp_striped_items(
    digits: qd.template(),
    ranks: qd.template(),
    block_prefixes: qd.template(),
):
    tid = qd.simt.block.thread_idx()
    lane = qd.i32(qd.simt.subgroup.invocation_id())
    subgroup = tid // 32
    warp_offsets = qd.simt.block.SharedArray(
        (8 * 256,),
        qd.i32,
    )
    match_masks = qd.simt.block.SharedArray(
        (8 * 256,),
        qd.i32,
    )

    for item in qd.static(range(8)):
        index = subgroup * 256 + lane + item * 32
        warp_offsets[index] = 0
        match_masks[index] = 0
    qd.simt.subgroup.sync()
    qd.simt.subgroup.mem_fence()

    for item in qd.static(range(24)):
        qd.atomic_add(
            warp_offsets[subgroup * 256 + digits[item]],
            1,
        )
    qd.simt.block.sync()

    block_count = qd.i32(0)
    if tid < 256:
        for warp in qd.static(range(8)):
            index = warp * 256 + tid
            warp_count = warp_offsets[index]
            warp_offsets[index] = block_count
            block_count = block_count + warp_count
    block_prefix = qd.simt.block.exclusive_add(
        block_count,
        256,
        qd.i32,
    )
    if tid < 256:
        block_prefixes[tid] = block_prefix
    qd.simt.block.sync()

    lane_mask_le = qd.simt.subgroup.lanemask_le(qd.u32(lane))
    lane_mask = qd.i32(qd.u32(1) << qd.u32(lane))
    for item in qd.static(range(24)):
        digit = digits[item]
        index = subgroup * 256 + digit
        qd.atomic_or(match_masks[index], lane_mask)
        qd.simt.subgroup.sync()
        qd.simt.subgroup.mem_fence()

        match = qd.u32(match_masks[index])
        leader = qd.i32(31) - qd.i32(qd.math.clz(match))
        count = qd.i32(qd.math.popcnt(match & lane_mask_le))
        offset = qd.i32(0)
        if lane == leader:
            offset = qd.atomic_add(
                warp_offsets[index],
                count,
            )
        offset = qd.simt.subgroup.shuffle(
            offset,
            qd.u32(leader),
        )
        if lane == leader:
            match_masks[index] = 0
        qd.simt.subgroup.sync()
        qd.simt.subgroup.mem_fence()
        ranks[item] = offset + count - 1
    return qd.Vector(
        [block_count, block_prefix],
        dt=qd.i32,
    )


@qd.data_oriented
class DynamicRadixSort:
    """CGQ OneSweep radix sort with a device-resident live count."""

    def __init__(self, key_dtype, capacity: int) -> None:
        if key_dtype not in (qd.u32, qd.u64):
            raise TypeError("DynamicRadixSort supports only u32 and u64 keys")
        if capacity < 1:
            raise ValueError("DynamicRadixSort capacity must be positive")

        self.key_dtype = key_dtype
        n_passes = 4 if key_dtype == qd.u32 else 8
        self.n_passes_host = n_passes
        n_blocks = max(
            (capacity + _BLOCK_THREADS * _ITEMS_PER_THREAD - 1) // (_BLOCK_THREADS * _ITEMS_PER_THREAD),
            1,
        )

        self.bins = qd.ndarray(
            qd.i32,
            shape=(n_passes, _RADIX_DIGITS),
        )
        self.lookback_status = qd.ndarray(
            qd.i32,
            shape=(n_blocks, _RADIX_DIGITS),
        )
        self.lookback_partial = qd.ndarray(
            qd.i32,
            shape=(n_blocks, _RADIX_DIGITS),
        )
        self.lookback_complete = qd.ndarray(
            qd.i32,
            shape=(n_blocks, _RADIX_DIGITS),
        )
        self.tile_counters = qd.ndarray(
            qd.i32,
            shape=(n_passes,),
        )

        self.bins.from_numpy(
            np.zeros(
                (n_passes, _RADIX_DIGITS),
                dtype=np.int32,
            )
        )
        lookback_shape = (n_blocks, _RADIX_DIGITS)
        self.lookback_status.from_numpy(np.zeros(lookback_shape, dtype=np.int32))
        self.lookback_partial.from_numpy(np.zeros(lookback_shape, dtype=np.int32))
        self.lookback_complete.from_numpy(np.zeros(lookback_shape, dtype=np.int32))
        self.tile_counters.from_numpy(np.zeros(n_passes, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def clear_bins(self):
        qd.loop_config(name="drs_memset_bins", block_dim=256)
        for index in range(self.n_passes_host * 256):
            radix_pass = index // 256
            digit = index - radix_pass * 256
            self.bins[radix_pass, digit] = 0

    @qd.func(requires_top_level=True)
    def histogram(self, keys: qd.template(), n: qd.template()):
        qd.loop_config(name="drs_histogram", block_dim=128)
        for thread in range(128 * 128):
            tid = qd.simt.block.thread_idx()
            block = thread // 128
            histogram = qd.simt.block.SharedArray(
                (self.n_passes_host * 256,),
                qd.i32,
            )
            for radix_pass in qd.static(range(self.n_passes_host)):
                for item in qd.static(range(2)):
                    histogram[radix_pass * 256 + item * 128 + tid] = 0
            qd.simt.block.sync()

            base = block * 128 * 16 + tid
            while base < n:
                for item in qd.static(range(16)):
                    index = base + item * 128
                    key32 = qd.u32(0)
                    key64 = qd.u64(0)
                    valid = index < n
                    if valid:
                        if qd.static(self.key_dtype == qd.u32):
                            key32 = keys[index]
                        else:
                            key64 = keys[index]
                    for radix_pass in qd.static(range(self.n_passes_host)):
                        digit = qd.i32(0)
                        if valid:
                            shift = qd.static(radix_pass * 8)
                            if qd.static(self.key_dtype == qd.u32):
                                digit = qd.i32((key32 >> shift) & qd.u32(255))
                            else:
                                digit = qd.i32((key64 >> shift) & qd.u64(255))
                            qd.atomic_add(
                                histogram[radix_pass * 256 + digit],
                                1,
                            )
                base = base + 128 * 128 * 16
            qd.simt.block.sync()
            for radix_pass in qd.static(range(self.n_passes_host)):
                for item in qd.static(range(2)):
                    digit = item * 128 + tid
                    qd.atomic_add(
                        self.bins[radix_pass, digit],
                        histogram[radix_pass * 256 + digit],
                    )
            qd.simt.block.sync()

    @qd.func(requires_top_level=True)
    def scan_bins(self):
        qd.loop_config(name="drs_exclusive_sum", block_dim=256)
        for index in range(self.n_passes_host * 256):
            radix_pass = index // 256
            digit = index - radix_pass * 256
            value = self.bins[radix_pass, digit]
            prefix = qd.simt.block.exclusive_add(
                value,
                256,
                qd.i32,
            )
            self.bins[radix_pass, digit] = prefix

    @qd.func(requires_top_level=True)
    def clear_pass(
        self,
        radix_pass: qd.template(),
        n: qd.template(),
    ):
        qd.loop_config(name="drs_memset_lookback", block_dim=256)
        needed_blocks = qd.max(
            (n + 6143) // 6144,
            1,
        )
        for index in range(needed_blocks * 256):
            block = index // 256
            digit = index - block * 256
            self.lookback_status[block, digit] = 0
            if index == 0:
                self.tile_counters[radix_pass] = 0

    @qd.func(requires_top_level=True)
    def onesweep_pass(
        self,
        keys_in: qd.template(),
        keys_out: qd.template(),
        perm_in: qd.template(),
        perm_out: qd.template(),
        n: qd.template(),
        radix_pass: qd.template(),
    ):
        qd.loop_config(name="drs_onesweep", block_dim=256)
        needed_blocks = qd.max(
            (n + 6143) // 6144,
            1,
        )
        for _ in range(needed_blocks * 256):
            qd.simt.block.sync()
            tid = qd.simt.block.thread_idx()
            count = n
            launch_block = _ // 256
            tile_slot = qd.simt.block.SharedArray((1,), qd.i32)
            if tid == 0:
                if launch_block < needed_blocks:
                    tile_slot[0] = qd.atomic_add(
                        self.tile_counters[radix_pass],
                        1,
                    )
                else:
                    tile_slot[0] = needed_blocks
            qd.simt.block.sync()

            tile = tile_slot[0]
            if tile < needed_blocks:
                block_prefixes = qd.simt.block.SharedArray(
                    (256,),
                    qd.i32,
                )
                tile_prefixes = qd.simt.block.SharedArray(
                    (256,),
                    qd.i32,
                )
                sorted_data = qd.simt.block.SharedArray(
                    (6144,),
                    qd.u64,
                )
                sorted_perm = qd.simt.block.SharedArray(
                    (6144,),
                    qd.i32,
                )

                keys32 = qd.Vector.zero(
                    qd.u32,
                    24,
                )
                keys64 = qd.Vector.zero(
                    qd.u64,
                    24,
                )
                digits = qd.Vector.zero(
                    qd.i32,
                    24,
                )
                local_ranks = qd.Vector.zero(
                    qd.i32,
                    24,
                )

                for item in qd.static(range(24)):
                    input_index = tile * 6144 + (tid // 32) * 24 * 32 + item * 32 + tid % 32
                    valid = input_index < count
                    key32 = qd.u32(0xFFFFFFFF)
                    key64 = qd.u64(0xFFFFFFFFFFFFFFFF)
                    if valid:
                        if qd.static(self.key_dtype == qd.u32):
                            key32 = keys_in[input_index]
                        else:
                            key64 = keys_in[input_index]

                    shift = qd.static(radix_pass * 8)
                    digit_key = qd.u32(0xFF)
                    if qd.static(self.key_dtype == qd.u32):
                        digit_key = (key32 >> shift) & qd.u32(0xFF)
                    else:
                        digit_key = qd.u32((key64 >> shift) & qd.u64(0xFF))
                    digit = qd.i32(digit_key)
                    keys32[item] = key32
                    keys64[item] = key64
                    digits[item] = digit

                rank_metadata = _rank_warp_striped_items(
                    digits,
                    local_ranks,
                    block_prefixes,
                )
                invalid_count = qd.min(
                    qd.max(
                        tile * 6144 + 6144 - count,
                        0,
                    ),
                    6144,
                )
                local_count = rank_metadata[0]
                if tid == 255:
                    local_count = local_count - invalid_count
                for item in qd.static(range(24)):
                    input_index = tile * 6144 + (tid // 32) * 24 * 32 + item * 32 + tid % 32
                    if input_index < count:
                        digit = digits[item]
                        sorted_index = block_prefixes[digit] + local_ranks[item]
                        if qd.static(self.key_dtype == qd.u32):
                            sorted_data[sorted_index] = qd.u64(keys32[item])
                        else:
                            sorted_data[sorted_index] = keys64[item]
                        sorted_perm[sorted_index] = perm_in[input_index]

                tile_prefix = qd.i32(0)
                if tid < 256:
                    predecessor = tile - 1
                    found_complete = tile == 0
                    if tile > 0:
                        self.lookback_partial[tile, tid] = local_count
                        qd.simt.grid.mem_fence()
                        qd.atomic_exchange(
                            self.lookback_status[tile, tid],
                            1,
                        )

                    while not found_complete:
                        if predecessor < 0:
                            found_complete = True
                        else:
                            state = qd.volatile_load(
                                self.lookback_status[
                                    predecessor,
                                    tid,
                                ]
                            )
                            if state != 0:
                                qd.simt.grid.mem_fence()
                                if state == 2:
                                    tile_prefix = tile_prefix + qd.volatile_load(
                                        self.lookback_complete[
                                            predecessor,
                                            tid,
                                        ]
                                    )
                                    found_complete = True
                                else:
                                    tile_prefix = tile_prefix + qd.volatile_load(
                                        self.lookback_partial[
                                            predecessor,
                                            tid,
                                        ]
                                    )
                                    predecessor = predecessor - 1

                    self.lookback_complete[
                        tile,
                        tid,
                    ] = tile_prefix + local_count
                    qd.simt.grid.mem_fence()
                    qd.atomic_exchange(
                        self.lookback_status[tile, tid],
                        2,
                    )
                    tile_prefixes[tid] = tile_prefix
                qd.simt.block.sync()

                valid_count = qd.min(
                    qd.max(
                        count - tile * 6144,
                        0,
                    ),
                    6144,
                )
                for item in qd.static(range(24)):
                    sorted_index = item * 256 + tid
                    if sorted_index < valid_count:
                        key = sorted_data[sorted_index]
                        shift = qd.static(radix_pass * 8)
                        digit = qd.i32((key >> shift) & qd.u64(0xFF))
                        output_index = (
                            self.bins[radix_pass, digit] + tile_prefixes[digit] + sorted_index - block_prefixes[digit]
                        )
                        if qd.static(self.key_dtype == qd.u32):
                            keys_out[output_index] = qd.u32(key)
                        else:
                            keys_out[output_index] = key
                        perm_out[output_index] = sorted_perm[sorted_index]
            qd.simt.block.sync()

    @qd.func(requires_top_level=True)
    def sort(
        self,
        keys_a: qd.template(),
        keys_b: qd.template(),
        perm_a: qd.template(),
        perm_b: qd.template(),
        n: qd.template(),
    ):
        self.clear_bins()
        self.histogram(keys_a, n)
        self.scan_bins()
        for radix_pass in qd.static(range(self.n_passes_host)):
            self.clear_pass(radix_pass, n)
            if qd.static(radix_pass % 2 == 0):
                self.onesweep_pass(
                    keys_a,
                    keys_b,
                    perm_a,
                    perm_b,
                    n,
                    radix_pass,
                )
            else:
                self.onesweep_pass(
                    keys_b,
                    keys_a,
                    perm_b,
                    perm_a,
                    n,
                    radix_pass,
                )


@qd.func(requires_top_level=True)
def dynamic_radix_sort(
    sorter: qd.template(),
    keys_a: qd.template(),
    keys_b: qd.template(),
    perm_a: qd.template(),
    perm_b: qd.template(),
    n: qd.template(),
):
    sorter.sort(
        keys_a,
        keys_b,
        perm_a,
        perm_b,
        n,
    )
