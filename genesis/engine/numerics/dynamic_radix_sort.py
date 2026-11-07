import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field

_RADIX_DIGITS = 256
_BLOCK_THREADS = 256
_ITEMS_PER_THREAD = 24
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
# Chunks of a radix pass on a CPU backend, each one thread's serial histogram and scatter of a contiguous range
CPU_CHUNKS = 256

RADIX_PASSES = Size("radix_passes")
RADIX_TILES = Size("radix_tiles")
RADIX_CHUNKS = Size("radix_chunks")


class DynamicRadixSortScratch(Data, kind=DataKind.SCRATCH):
    """Pass state of a stable radix sort of keys and their permutation, with a device-side count."""

    bins: qd.Tensor = field(int, shape=(RADIX_PASSES, _RADIX_DIGITS), ndarray=True)
    lookback_status: qd.Tensor = field(int, shape=(RADIX_TILES, _RADIX_DIGITS), ndarray=True)
    lookback_partial: qd.Tensor = field(int, shape=(RADIX_TILES, _RADIX_DIGITS), ndarray=True)
    lookback_complete: qd.Tensor = field(int, shape=(RADIX_TILES, _RADIX_DIGITS), ndarray=True)
    tile_counters: qd.Tensor = field(int, shape=(RADIX_PASSES,), ndarray=True)
    # Pass loop of kernel_radix_sort on the device: the first of the two passes of an iteration, and the condition
    pass_idx: qd.Tensor = field(int, ndarray=True)
    is_pass_running: qd.Tensor = field(int, ndarray=True)
    # Digit offsets of every chunk in the CPU passes
    chunk_offsets: qd.Tensor = field(int, shape=(RADIX_CHUNKS, _RADIX_DIGITS), ndarray=True)


def radix_passes(key_bits: int) -> int:
    """Return the number of radix passes of a sort over keys of `key_bits` bits, 32 or 64, one per byte."""
    assert key_bits in (32, 64)
    return key_bits // 8


def dynamic_radix_sort_sizes(key_bits: int, capacity: int) -> dict[Size, int]:
    """Return the sizes of the scratch of a sort over at most `capacity` keys of `key_bits` bits, 32 or 64."""
    return {
        RADIX_PASSES: radix_passes(key_bits),
        RADIX_TILES: max((capacity + _TILE_ITEMS - 1) // _TILE_ITEMS, 1),
        RADIX_CHUNKS: CPU_CHUNKS if gs.backend == gs.cpu else 1,
    }


@qd.func
def _rank_warp_striped_items(digits: qd.template(), ranks: qd.template(), block_prefixes: qd.template()):
    tid = qd.simt.block.thread_idx()
    i_lane = gs.qd_int(qd.simt.subgroup.invocation_id())
    i_warp = tid // 32
    sh_warp_offsets = qd.simt.block.SharedArray((8 * 256,), gs.qd_int)
    sh_match_masks = qd.simt.block.SharedArray((8 * 256,), gs.qd_int)

    for k in qd.static(range(8)):
        i_sh = i_warp * 256 + i_lane + k * 32
        sh_warp_offsets[i_sh] = 0
        sh_match_masks[i_sh] = 0
    qd.simt.subgroup.sync()
    qd.simt.subgroup.mem_fence()

    for k in qd.static(range(24)):
        qd.atomic_add(sh_warp_offsets[i_warp * 256 + digits[k]], 1)
    qd.simt.block.sync()

    block_count = gs.qd_int(0)
    if tid < 256:
        for j_warp in qd.static(range(8)):
            i_sh = j_warp * 256 + tid
            warp_count = sh_warp_offsets[i_sh]
            sh_warp_offsets[i_sh] = block_count
            block_count = block_count + warp_count
    block_prefix = qd.simt.block.exclusive_add(block_count, 256, gs.qd_int)
    if tid < 256:
        block_prefixes[tid] = block_prefix
    qd.simt.block.sync()

    lane_mask_le = qd.simt.subgroup.lanemask_le(qd.u32(i_lane))
    lane_mask = gs.qd_int(qd.u32(1) << qd.u32(i_lane))
    for k in qd.static(range(24)):
        i_digit = digits[k]
        i_sh = i_warp * 256 + i_digit
        qd.atomic_or(sh_match_masks[i_sh], lane_mask)
        qd.simt.subgroup.sync()
        qd.simt.subgroup.mem_fence()

        match = qd.u32(sh_match_masks[i_sh])
        j_lane = gs.qd_int(31) - gs.qd_int(qd.math.clz(match))
        count = gs.qd_int(qd.math.popcnt(match & lane_mask_le))
        offset = gs.qd_int(0)
        if i_lane == j_lane:
            offset = qd.atomic_add(sh_warp_offsets[i_sh], count)
        offset = qd.simt.subgroup.shuffle(offset, qd.u32(j_lane))
        if i_lane == j_lane:
            sh_match_masks[i_sh] = 0
        qd.simt.subgroup.sync()
        qd.simt.subgroup.mem_fence()
        ranks[k] = offset + count - 1
    return qd.Vector([block_count, block_prefix], dt=gs.qd_int)


@qd.func(requires_top_level=True)
def _func_clear_bins(sorter: DynamicRadixSortScratch, key_bits: qd.template()):
    n_passes = qd.static(radix_passes(key_bits))
    qd.loop_config(name="drs_memset_bins", block_dim=256)
    for i_bin in range(n_passes * 256):
        i_pass = i_bin // 256
        i_digit = i_bin - i_pass * 256
        sorter.bins[i_pass, i_digit] = 0


@qd.func(requires_top_level=True)
def _func_histogram(keys: qd.template(), n: qd.Tensor, sorter: DynamicRadixSortScratch, key_bits: qd.template()):
    n_passes = qd.static(radix_passes(key_bits))
    qd.loop_config(name="drs_histogram", block_dim=128)
    for i_th in range(128 * 128):
        tid = qd.simt.block.thread_idx()
        i_blk = i_th // 128
        sh_histogram = qd.simt.block.SharedArray((n_passes * 256,), gs.qd_int)
        for i_pass in qd.static(range(n_passes)):
            for k in qd.static(range(2)):
                sh_histogram[i_pass * 256 + k * 128 + tid] = 0
        qd.simt.block.sync()

        for i_chunk_ in range((n[()] + 128 * 128 * 16 - 1) // (128 * 128 * 16)):
            i_in0 = i_chunk_ * 128 * 128 * 16 + i_blk * 128 * 16 + tid
            if i_in0 < n[()]:
                for k in qd.static(range(16)):
                    i_in = i_in0 + k * 128
                    key32 = qd.u32(0)
                    key64 = qd.u64(0)
                    is_valid = i_in < n[()]
                    if is_valid:
                        if qd.static(key_bits == 32):
                            key32 = keys[i_in]
                        else:
                            key64 = keys[i_in]
                    for i_pass in qd.static(range(n_passes)):
                        i_digit = gs.qd_int(0)
                        if is_valid:
                            shift = qd.static(i_pass * 8)
                            if qd.static(key_bits == 32):
                                i_digit = gs.qd_int((key32 >> shift) & qd.u32(255))
                            else:
                                i_digit = gs.qd_int((key64 >> shift) & qd.u64(255))
                            qd.atomic_add(sh_histogram[i_pass * 256 + i_digit], 1)
        qd.simt.block.sync()
        for i_pass in qd.static(range(n_passes)):
            for k in qd.static(range(2)):
                i_digit = k * 128 + tid
                qd.atomic_add(sorter.bins[i_pass, i_digit], sh_histogram[i_pass * 256 + i_digit])
        qd.simt.block.sync()


@qd.func(requires_top_level=True)
def _func_scan_bins(sorter: DynamicRadixSortScratch, key_bits: qd.template()):
    n_passes = qd.static(radix_passes(key_bits))
    qd.loop_config(name="drs_exclusive_sum", block_dim=256)
    for i_bin in range(n_passes * 256):
        i_pass = i_bin // 256
        i_digit = i_bin - i_pass * 256
        value = sorter.bins[i_pass, i_digit]
        prefix = qd.simt.block.exclusive_add(value, 256, gs.qd_int)
        sorter.bins[i_pass, i_digit] = prefix


@qd.func(requires_top_level=True)
def _func_clear_pass(i_pass: int, n: qd.Tensor, sorter: DynamicRadixSortScratch):
    qd.loop_config(name="drs_memset_lookback", block_dim=256)
    n_tiles = qd.max((n[()] + 6143) // 6144, 1)
    for i_th in range(n_tiles * 256):
        i_tile = i_th // 256
        i_digit = i_th - i_tile * 256
        sorter.lookback_status[i_tile, i_digit] = 0
        if i_th == 0:
            sorter.tile_counters[i_pass] = 0


@qd.func(requires_top_level=True)
def _func_onesweep_pass(
    i_pass: int,
    keys_in: qd.template(),
    keys_out: qd.template(),
    perm_in: qd.template(),
    perm_out: qd.template(),
    n: qd.Tensor,
    sorter: DynamicRadixSortScratch,
    key_bits: qd.template(),
):
    qd.loop_config(name="drs_onesweep", block_dim=256)
    n_tiles = qd.max((n[()] + 6143) // 6144, 1)
    for i_th in range(n_tiles * 256):
        shift = qd.u32(i_pass * 8)
        qd.simt.block.sync()
        tid = qd.simt.block.thread_idx()
        count = n[()]
        i_blk = i_th // 256
        sh_tile_slot = qd.simt.block.SharedArray((1,), gs.qd_int)
        if tid == 0:
            if i_blk < n_tiles:
                sh_tile_slot[0] = qd.atomic_add(sorter.tile_counters[i_pass], 1)
            else:
                sh_tile_slot[0] = n_tiles
        qd.simt.block.sync()

        i_tile = sh_tile_slot[0]
        if i_tile < n_tiles:
            sh_block_prefixes = qd.simt.block.SharedArray((256,), gs.qd_int)
            sh_tile_prefixes = qd.simt.block.SharedArray((256,), gs.qd_int)
            sh_sorted_keys = qd.simt.block.SharedArray((6144,), qd.u64)
            sh_sorted_perm = qd.simt.block.SharedArray((6144,), gs.qd_int)

            keys32 = qd.Vector.zero(qd.u32, 24)
            keys64 = qd.Vector.zero(qd.u64, 24)
            digits = qd.Vector.zero(gs.qd_int, 24)
            local_ranks = qd.Vector.zero(gs.qd_int, 24)

            for k in qd.static(range(24)):
                i_in = i_tile * 6144 + (tid // 32) * 24 * 32 + k * 32 + tid % 32
                is_valid = i_in < count
                key32 = qd.u32(0xFFFFFFFF)
                key64 = qd.u64(0xFFFFFFFFFFFFFFFF)
                if is_valid:
                    if qd.static(key_bits == 32):
                        key32 = keys_in[i_in]
                    else:
                        key64 = keys_in[i_in]

                digit_key = qd.u32(0xFF)
                if qd.static(key_bits == 32):
                    digit_key = (key32 >> shift) & qd.u32(0xFF)
                else:
                    digit_key = qd.u32((key64 >> qd.u64(shift)) & qd.u64(0xFF))
                i_digit = gs.qd_int(digit_key)
                keys32[k] = key32
                keys64[k] = key64
                digits[k] = i_digit

            rank_metadata = _rank_warp_striped_items(digits, local_ranks, sh_block_prefixes)
            invalid_count = qd.min(qd.max(i_tile * 6144 + 6144 - count, 0), 6144)
            local_count = rank_metadata[0]
            if tid == 255:
                local_count = local_count - invalid_count
            for k in qd.static(range(24)):
                i_in = i_tile * 6144 + (tid // 32) * 24 * 32 + k * 32 + tid % 32
                if i_in < count:
                    i_digit = digits[k]
                    i_sorted = sh_block_prefixes[i_digit] + local_ranks[k]
                    if qd.static(key_bits == 32):
                        sh_sorted_keys[i_sorted] = qd.u64(keys32[k])
                    else:
                        sh_sorted_keys[i_sorted] = keys64[k]
                    sh_sorted_perm[i_sorted] = perm_in[i_in]

            tile_prefix = gs.qd_int(0)
            if tid < 256:
                j_tile = i_tile - 1
                is_prefix_complete = i_tile == 0
                if i_tile > 0:
                    sorter.lookback_partial[i_tile, tid] = local_count
                    qd.simt.grid.mem_fence()
                    qd.atomic_exchange(sorter.lookback_status[i_tile, tid], 1)

                while not is_prefix_complete:
                    if j_tile < 0:
                        is_prefix_complete = True
                    else:
                        state = qd.volatile_load(sorter.lookback_status[j_tile, tid])
                        if state != 0:
                            qd.simt.grid.mem_fence()
                            if state == 2:
                                tile_prefix = tile_prefix + qd.volatile_load(sorter.lookback_complete[j_tile, tid])
                                is_prefix_complete = True
                            else:
                                tile_prefix = tile_prefix + qd.volatile_load(sorter.lookback_partial[j_tile, tid])
                                j_tile = j_tile - 1

                sorter.lookback_complete[i_tile, tid] = tile_prefix + local_count
                qd.simt.grid.mem_fence()
                qd.atomic_exchange(sorter.lookback_status[i_tile, tid], 2)
                sh_tile_prefixes[tid] = tile_prefix
            qd.simt.block.sync()

            valid_count = qd.min(qd.max(count - i_tile * 6144, 0), 6144)
            for k in qd.static(range(24)):
                i_sorted = k * 256 + tid
                if i_sorted < valid_count:
                    key = sh_sorted_keys[i_sorted]
                    i_digit = gs.qd_int((key >> qd.u64(shift)) & qd.u64(0xFF))
                    i_out = (
                        sorter.bins[i_pass, i_digit] + sh_tile_prefixes[i_digit] + i_sorted - sh_block_prefixes[i_digit]
                    )
                    if qd.static(key_bits == 32):
                        keys_out[i_out] = qd.u32(key)
                    else:
                        keys_out[i_out] = key
                    perm_out[i_out] = sh_sorted_perm[i_sorted]
        qd.simt.block.sync()


@qd.func(requires_top_level=True)
def _func_start_passes(sorter: DynamicRadixSortScratch):
    for _ in range(1):
        sorter.pass_idx[()] = 0
        sorter.is_pass_running[()] = 1


@qd.func(requires_top_level=True)
def _func_advance_passes(sorter: DynamicRadixSortScratch, key_bits: qd.template()):
    n_passes = qd.static(radix_passes(key_bits))
    for _ in range(1):
        sorter.pass_idx[()] = sorter.pass_idx[()] + 2
        sorter.is_pass_running[()] = gs.qd_int(sorter.pass_idx[()] < n_passes)


@qd.func
def _func_digit(i_pass: qd.template(), key, key_bits: qd.template()):
    shift = qd.static(i_pass * 8)
    i_digit = gs.qd_int(0)
    if qd.static(key_bits == 32):
        i_digit = gs.qd_int((key >> qd.u32(shift)) & qd.u32(255))
    else:
        i_digit = gs.qd_int((key >> qd.u64(shift)) & qd.u64(255))
    return i_digit


@qd.func(requires_top_level=True)
def _func_cpu_pass(
    i_pass: qd.template(),
    keys_in: qd.template(),
    keys_out: qd.template(),
    perm_in: qd.template(),
    perm_out: qd.template(),
    n: qd.Tensor,
    sorter: DynamicRadixSortScratch,
    key_bits: qd.template(),
):
    qd.loop_config(name="drs_cpu_histogram")
    for i_chunk in range(CPU_CHUNKS):
        for i_digit in range(_RADIX_DIGITS):
            sorter.chunk_offsets[i_chunk, i_digit] = 0
        size = (n[()] + CPU_CHUNKS - 1) // CPU_CHUNKS
        for i_in in range(i_chunk * size, qd.min(i_chunk * size + size, n[()])):
            sorter.chunk_offsets[i_chunk, _func_digit(i_pass, keys_in[i_in], key_bits)] += 1
    for _ in range(1):
        running = gs.qd_int(0)
        for i_digit in range(_RADIX_DIGITS):
            for i_chunk in range(CPU_CHUNKS):
                count = sorter.chunk_offsets[i_chunk, i_digit]
                sorter.chunk_offsets[i_chunk, i_digit] = running
                running += count
    qd.loop_config(name="drs_cpu_scatter")
    for i_chunk in range(CPU_CHUNKS):
        size = (n[()] + CPU_CHUNKS - 1) // CPU_CHUNKS
        for i_in in range(i_chunk * size, qd.min(i_chunk * size + size, n[()])):
            key = keys_in[i_in]
            i_digit = _func_digit(i_pass, key, key_bits)
            i_out = sorter.chunk_offsets[i_chunk, i_digit]
            sorter.chunk_offsets[i_chunk, i_digit] = i_out + 1
            keys_out[i_out] = key
            perm_out[i_out] = perm_in[i_in]


@qd.func(requires_top_level=True)
def _func_cpu_sort(
    keys_a: qd.template(),
    keys_b: qd.template(),
    perm_a: qd.template(),
    perm_b: qd.template(),
    n: qd.Tensor,
    sorter: DynamicRadixSortScratch,
    key_bits: qd.template(),
):
    for i_pass in qd.static(range(radix_passes(key_bits))):
        if qd.static(i_pass % 2 == 0):
            _func_cpu_pass(i_pass, keys_a, keys_b, perm_a, perm_b, n, sorter, key_bits)
        else:
            _func_cpu_pass(i_pass, keys_b, keys_a, perm_b, perm_a, n, sorter, key_bits)


@qd.func(requires_top_level=True)
def func_dynamic_radix_sort(
    keys_a: qd.template(),
    keys_b: qd.template(),
    perm_a: qd.template(),
    perm_b: qd.template(),
    n: qd.Tensor,
    sorter: DynamicRadixSortScratch,
    key_bits: qd.template(),
):
    """Sort keys stably together with their permutation inside a graph kernel, the result landing in the input
    arrays."""
    if qd.static(gs.backend == gs.cpu):
        _func_cpu_sort(keys_a, keys_b, perm_a, perm_b, n, sorter, key_bits)
    else:
        _func_clear_bins(sorter, key_bits)
        _func_histogram(keys_a, n, sorter, key_bits)
        _func_scan_bins(sorter, key_bits)
        _func_start_passes(sorter)
        while qd.graph.do_while(sorter.is_pass_running):
            _func_clear_pass(sorter.pass_idx[()], n, sorter)
            _func_onesweep_pass(sorter.pass_idx[()], keys_a, keys_b, perm_a, perm_b, n, sorter, key_bits)
            _func_clear_pass(sorter.pass_idx[()] + 1, n, sorter)
            _func_onesweep_pass(sorter.pass_idx[()] + 1, keys_b, keys_a, perm_b, perm_a, n, sorter, key_bits)
            _func_advance_passes(sorter, key_bits)


@qd.kernel(graph=True)
def kernel_radix_sort(
    keys_a: qd.Tensor,
    keys_b: qd.Tensor,
    perm_a: qd.Tensor,
    perm_b: qd.Tensor,
    n: qd.Tensor,
    sorter: DynamicRadixSortScratch,
    key_bits: qd.template(),
):
    """Run func_dynamic_radix_sort as a graph of its own."""
    func_dynamic_radix_sort(keys_a, keys_b, perm_a, perm_b, n, sorter, key_bits)


def dynamic_radix_sort(sorter: DynamicRadixSortScratch, keys_a, keys_b, perm_a, perm_b, n, key_bits: int) -> None:
    """Sort the first `n[()]` keys of `keys_a` stably together with their permutation `perm_a` (see
    func_dynamic_radix_sort)."""
    kernel_radix_sort(keys_a, keys_b, perm_a, perm_b, n, sorter, key_bits)
