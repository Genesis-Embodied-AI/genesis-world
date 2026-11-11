"""Stable radix sort of unsigned keys together with their permutation, over a number of keys read on the device."""

import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field

# The digits of a pass, one byte of a key, and the keys a tile of a GPU pass ranks: 24 for each of its 256 threads
RADIX_DIGITS = 256
TILE_KEYS = 24 * 256
# The chunks of a CPU pass, each one counted and scattered by one thread
CPU_CHUNKS = 256

RADIX_PASSES = Size("radix_passes")
RADIX_TILES = Size("radix_tiles")
RADIX_CHUNKS = Size("radix_chunks")


class RadixSortScratch(Data, kind=DataKind.SCRATCH):
    """
    The workspace of a stable radix sort of keys and of their permutation.

    `bins` holds the first output slot of every digit of every pass. A GPU pass ranks the keys tile by tile, a tile
    taking its turn from `tile_counters` and publishing its digit counts in `lookback_partial`, then their sums over the
    tiles before it in `lookback_complete`, `lookback_status` telling which one it holds (decoupled look-back).
    `pass_idx` and `is_pass_running` run the passes in a loop of the graph. A CPU pass holds the next output slot of
    every digit of every chunk in `chunk_offsets`.
    """

    bins: qd.Tensor = field(int, shape=(RADIX_PASSES, RADIX_DIGITS), ndarray=True)
    lookback_status: qd.Tensor = field(int, shape=(RADIX_TILES, RADIX_DIGITS), ndarray=True)
    lookback_partial: qd.Tensor = field(int, shape=(RADIX_TILES, RADIX_DIGITS), ndarray=True)
    lookback_complete: qd.Tensor = field(int, shape=(RADIX_TILES, RADIX_DIGITS), ndarray=True)
    tile_counters: qd.Tensor = field(int, shape=(RADIX_PASSES,), ndarray=True)
    pass_idx: qd.Tensor = field(int, shape=(), ndarray=True)
    is_pass_running: qd.Tensor = field(int, shape=(), ndarray=True)
    chunk_offsets: qd.Tensor = field(int, shape=(RADIX_CHUNKS, RADIX_DIGITS), ndarray=True)


def radix_sort_sizes(key_bits: int, capacity: int) -> dict[Size, int]:
    """
    Return the sizes of the workspace of a sort over a bounded number of keys.

    The sort runs over at most `capacity` keys of `key_bits` bits, 32 or 64.
    """
    assert key_bits in (32, 64)
    return {
        RADIX_PASSES: key_bits // 8,
        RADIX_TILES: max((capacity + TILE_KEYS - 1) // TILE_KEYS, 1),
        RADIX_CHUNKS: CPU_CHUNKS if gs.backend == gs.cpu else 1,
    }


@qd.func
def func_key_digit(i_pass: int, key, key_bits: qd.template()):
    """
    Return the digit of a key that a pass sorts on.

    The digit is the byte of rank `i_pass` from the least significant one.
    """
    i_digit = gs.qd_int(0)
    if qd.static(key_bits == 32):
        i_digit = gs.qd_int((key >> qd.u32(8 * i_pass)) & qd.u32(255))
    else:
        i_digit = gs.qd_int((key >> qd.u64(8 * i_pass)) & qd.u64(255))
    return i_digit


@qd.func
def func_rank_warp_striped_keys(digits: qd.template(), ranks: qd.template(), block_prefixes: qd.template()):
    """
    Rank the keys of a tile by their digit, the 24 keys of every thread striped over its warp.

    Every thread gets the rank of each of its keys among the keys of the tile of the same digit, and the threads of the
    first 256 the number of keys of the tile of their digit, whose exclusive sum over the digits lands in
    `block_prefixes`. Returns the count and the exclusive sum of the digit of the thread.
    """
    WARPS = qd.static(8)
    KEYS_PER_THREAD = qd.static(24)
    tid = qd.simt.block.thread_idx()
    i_lane = gs.qd_int(qd.simt.subgroup.invocation_id())
    i_warp = tid // 32
    sh_warp_offsets = qd.simt.block.SharedArray((WARPS * 256,), gs.qd_int)
    sh_match_masks = qd.simt.block.SharedArray((WARPS * 256,), gs.qd_int)

    for k in qd.static(range(WARPS)):
        i_sh = i_warp * 256 + i_lane + k * 32
        sh_warp_offsets[i_sh] = 0
        sh_match_masks[i_sh] = 0
    qd.simt.subgroup.sync()
    qd.simt.subgroup.mem_fence()

    for k in qd.static(range(KEYS_PER_THREAD)):
        qd.atomic_add(sh_warp_offsets[i_warp * 256 + digits[k]], 1)
    qd.simt.block.sync()

    # The count of every digit over the warps, each warp starting at the count of the warps before it
    block_count = gs.qd_int(0)
    if tid < 256:
        for j_warp in qd.static(range(WARPS)):
            i_sh = j_warp * 256 + tid
            warp_count = sh_warp_offsets[i_sh]
            sh_warp_offsets[i_sh] = block_count
            block_count = block_count + warp_count
    block_prefix = qd.simt.block.exclusive_add(block_count, 256, gs.qd_int)
    if tid < 256:
        block_prefixes[tid] = block_prefix
    qd.simt.block.sync()

    # The lanes holding a key of the same digit agree on its slot through the lowest of them
    lane_mask_le = qd.simt.subgroup.lanemask_le(qd.u32(i_lane))
    lane_mask = gs.qd_int(qd.u32(1) << qd.u32(i_lane))
    for k in qd.static(range(KEYS_PER_THREAD)):
        i_sh = i_warp * 256 + digits[k]
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
def func_clear_bins(sorter: RadixSortScratch):
    qd.loop_config(name="radix_sort_clear_bins", block_dim=256)
    for i_bin in range(sorter.bins.shape[0] * 256):
        i_pass = i_bin // 256
        sorter.bins[i_pass, i_bin - i_pass * 256] = 0


@qd.func(requires_top_level=True)
def func_count_digits(sorter: RadixSortScratch, keys: qd.Tensor, n_keys: qd.Tensor, key_bits: qd.template()):
    """Count the digits of every pass over the keys, by blocks counting in shared memory."""
    N_PASSES = qd.static(key_bits // 8)
    KEYS_PER_THREAD = qd.static(16)
    qd.loop_config(name="radix_sort_count_digits", block_dim=128)
    for i_th in range(128 * 128):
        tid = qd.simt.block.thread_idx()
        i_blk = i_th // 128
        sh_counts = qd.simt.block.SharedArray((N_PASSES * 256,), gs.qd_int)
        for i_pass in qd.static(range(N_PASSES)):
            for k in qd.static(range(2)):
                sh_counts[i_pass * 256 + k * 128 + tid] = 0
        qd.simt.block.sync()

        n = n_keys[()]
        for i_chunk_ in range((n + 128 * 128 * KEYS_PER_THREAD - 1) // (128 * 128 * KEYS_PER_THREAD)):
            i_key_start = i_chunk_ * 128 * 128 * KEYS_PER_THREAD + i_blk * 128 * KEYS_PER_THREAD + tid
            if i_key_start < n:
                for k in qd.static(range(KEYS_PER_THREAD)):
                    i_key = i_key_start + k * 128
                    if i_key < n:
                        key = keys[i_key]
                        for i_pass in qd.static(range(N_PASSES)):
                            qd.atomic_add(sh_counts[i_pass * 256 + func_key_digit(i_pass, key, key_bits)], 1)
        qd.simt.block.sync()
        for i_pass in qd.static(range(N_PASSES)):
            for k in qd.static(range(2)):
                i_digit = k * 128 + tid
                qd.atomic_add(sorter.bins[i_pass, i_digit], sh_counts[i_pass * 256 + i_digit])
        qd.simt.block.sync()


@qd.func(requires_top_level=True)
def func_scan_bins(sorter: RadixSortScratch):
    """Replace the count of every digit of every pass by the first output slot of the digit."""
    qd.loop_config(name="radix_sort_scan_bins", block_dim=256)
    for i_bin in range(sorter.bins.shape[0] * 256):
        i_pass = i_bin // 256
        i_digit = i_bin - i_pass * 256
        sorter.bins[i_pass, i_digit] = qd.simt.block.exclusive_add(sorter.bins[i_pass, i_digit], 256, gs.qd_int)


@qd.func(requires_top_level=True)
def func_clear_lookback(sorter: RadixSortScratch, i_pass: int, n_keys: qd.Tensor):
    qd.loop_config(name="radix_sort_clear_lookback", block_dim=256)
    n_tiles = qd.max((n_keys[()] + 6143) // 6144, 1)
    for i_th in range(n_tiles * 256):
        i_tile = i_th // 256
        sorter.lookback_status[i_tile, i_th - i_tile * 256] = 0
        if i_th == 0:
            sorter.tile_counters[i_pass] = 0


@qd.func(requires_top_level=True)
def func_onesweep_pass(
    sorter: RadixSortScratch,
    i_pass: int,
    keys_in: qd.Tensor,
    keys_out: qd.Tensor,
    permutation_in: qd.Tensor,
    permutation_out: qd.Tensor,
    n_keys: qd.Tensor,
    key_bits: qd.template(),
):
    """Scatter the keys and the permutation by their digit of a pass, ranking the tiles in their order of arrival."""
    TILE = qd.static(6144)
    KEYS_PER_THREAD = qd.static(24)
    qd.loop_config(name="radix_sort_onesweep", block_dim=256)
    n_tiles = qd.max((n_keys[()] + TILE - 1) // TILE, 1)
    for i_th in range(n_tiles * 256):
        qd.simt.block.sync()
        tid = qd.simt.block.thread_idx()
        n = n_keys[()]
        i_blk = i_th // 256
        # The tiles take their turn in the order the blocks start, so that a tile only waits on tiles already running
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
            sh_sorted_keys = qd.simt.block.SharedArray((TILE,), qd.u64)
            sh_sorted_permutation = qd.simt.block.SharedArray((TILE,), gs.qd_int)

            keys32 = qd.Vector.zero(qd.u32, KEYS_PER_THREAD)
            keys64 = qd.Vector.zero(qd.u64, KEYS_PER_THREAD)
            digits = qd.Vector.zero(gs.qd_int, KEYS_PER_THREAD)
            local_ranks = qd.Vector.zero(gs.qd_int, KEYS_PER_THREAD)
            # A slot past the keys holds the largest key, which sorts last
            for k in qd.static(range(KEYS_PER_THREAD)):
                i_in = i_tile * TILE + (tid // 32) * KEYS_PER_THREAD * 32 + k * 32 + tid % 32
                if qd.static(key_bits == 32):
                    key32 = qd.u32(0xFFFFFFFF)
                    if i_in < n:
                        key32 = keys_in[i_in]
                    keys32[k] = key32
                    digits[k] = func_key_digit(i_pass, key32, key_bits)
                else:
                    key64 = qd.u64(0xFFFFFFFFFFFFFFFF)
                    if i_in < n:
                        key64 = keys_in[i_in]
                    keys64[k] = key64
                    digits[k] = func_key_digit(i_pass, key64, key_bits)

            rank_metadata = func_rank_warp_striped_keys(digits, local_ranks, sh_block_prefixes)
            local_count = rank_metadata[0]
            # The padding of the last tile counts as keys of the last digit, which no tile publishes
            if tid == 255:
                local_count = local_count - qd.min(qd.max(i_tile * TILE + TILE - n, 0), TILE)
            for k in qd.static(range(KEYS_PER_THREAD)):
                i_in = i_tile * TILE + (tid // 32) * KEYS_PER_THREAD * 32 + k * 32 + tid % 32
                if i_in < n:
                    i_sorted = sh_block_prefixes[digits[k]] + local_ranks[k]
                    if qd.static(key_bits == 32):
                        sh_sorted_keys[i_sorted] = qd.u64(keys32[k])
                    else:
                        sh_sorted_keys[i_sorted] = keys64[k]
                    sh_sorted_permutation[i_sorted] = permutation_in[i_in]

            # The keys of every digit in the tiles before this one, read back tile by tile until a tile holding them all
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
                        status = qd.volatile_load(sorter.lookback_status[j_tile, tid])
                        if status != 0:
                            qd.simt.grid.mem_fence()
                            if status == 2:
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

            n_tile_keys = qd.min(qd.max(n - i_tile * TILE, 0), TILE)
            for k in qd.static(range(KEYS_PER_THREAD)):
                i_sorted = k * 256 + tid
                if i_sorted < n_tile_keys:
                    key = sh_sorted_keys[i_sorted]
                    i_digit = func_key_digit(i_pass, key, key_bits=64)
                    i_out = (
                        sorter.bins[i_pass, i_digit] + sh_tile_prefixes[i_digit] + i_sorted - sh_block_prefixes[i_digit]
                    )
                    if qd.static(key_bits == 32):
                        keys_out[i_out] = qd.u32(key)
                    else:
                        keys_out[i_out] = key
                    permutation_out[i_out] = sh_sorted_permutation[i_sorted]
        qd.simt.block.sync()


@qd.func(requires_top_level=True)
def func_cpu_pass(
    sorter: RadixSortScratch,
    i_pass: qd.template(),
    keys_in: qd.Tensor,
    keys_out: qd.Tensor,
    permutation_in: qd.Tensor,
    permutation_out: qd.Tensor,
    n_keys: qd.Tensor,
    key_bits: qd.template(),
):
    """Scatter the keys and the permutation by their digit of a pass, chunk by chunk."""
    qd.loop_config(name="radix_sort_cpu_count")
    for i_chunk in range(sorter.chunk_offsets.shape[0]):
        for i_digit in range(256):
            sorter.chunk_offsets[i_chunk, i_digit] = 0
        chunk_size = (n_keys[()] + sorter.chunk_offsets.shape[0] - 1) // sorter.chunk_offsets.shape[0]
        for i_in in range(i_chunk * chunk_size, qd.min(i_chunk * chunk_size + chunk_size, n_keys[()])):
            i_digit = func_key_digit(i_pass, keys_in[i_in], key_bits)
            sorter.chunk_offsets[i_chunk, i_digit] = sorter.chunk_offsets[i_chunk, i_digit] + 1
    for _ in range(1):
        n_keys_before = gs.qd_int(0)
        for i_digit in range(256):
            for i_chunk in range(sorter.chunk_offsets.shape[0]):
                count = sorter.chunk_offsets[i_chunk, i_digit]
                sorter.chunk_offsets[i_chunk, i_digit] = n_keys_before
                n_keys_before += count
    qd.loop_config(name="radix_sort_cpu_scatter")
    for i_chunk in range(sorter.chunk_offsets.shape[0]):
        chunk_size = (n_keys[()] + sorter.chunk_offsets.shape[0] - 1) // sorter.chunk_offsets.shape[0]
        for i_in in range(i_chunk * chunk_size, qd.min(i_chunk * chunk_size + chunk_size, n_keys[()])):
            key = keys_in[i_in]
            i_digit = func_key_digit(i_pass, key, key_bits)
            i_out = sorter.chunk_offsets[i_chunk, i_digit]
            sorter.chunk_offsets[i_chunk, i_digit] = i_out + 1
            keys_out[i_out] = key
            permutation_out[i_out] = permutation_in[i_in]


@qd.func(requires_top_level=True)
def func_radix_sort(
    sorter: RadixSortScratch,
    keys: qd.Tensor,
    keys_buffer: qd.Tensor,
    permutation: qd.Tensor,
    permutation_buffer: qd.Tensor,
    n_keys: qd.Tensor,
    key_bits: qd.template(),
):
    """
    Sort the leading keys of an array stably together with their permutation, in place.

    The sort covers the first `n_keys[()]` keys. The buffers hold the output of every other pass. The passes are even in
    number, so the sorted keys land back in the input arrays. The GPU passes run in a loop of the graph, which the
    kernel of the caller must declare (see 'kernel_radix_sort').
    """
    if qd.static(gs.backend == gs.cpu):
        for i_pass in qd.static(range(key_bits // 8)):
            if qd.static(i_pass % 2 == 0):
                func_cpu_pass(sorter, i_pass, keys, keys_buffer, permutation, permutation_buffer, n_keys, key_bits)
            else:
                func_cpu_pass(sorter, i_pass, keys_buffer, keys, permutation_buffer, permutation, n_keys, key_bits)
    else:
        func_clear_bins(sorter)
        func_count_digits(sorter, keys, n_keys, key_bits)
        func_scan_bins(sorter)
        for _ in range(1):
            sorter.pass_idx[()] = 0
            sorter.is_pass_running[()] = 1
        while qd.graph.do_while(sorter.is_pass_running):
            func_clear_lookback(sorter, sorter.pass_idx[()], n_keys)
            func_onesweep_pass(
                sorter, sorter.pass_idx[()], keys, keys_buffer, permutation, permutation_buffer, n_keys, key_bits
            )
            func_clear_lookback(sorter, sorter.pass_idx[()] + 1, n_keys)
            func_onesweep_pass(
                sorter, sorter.pass_idx[()] + 1, keys_buffer, keys, permutation_buffer, permutation, n_keys, key_bits
            )
            for _ in range(1):
                sorter.pass_idx[()] = sorter.pass_idx[()] + 2
                sorter.is_pass_running[()] = gs.qd_int(sorter.pass_idx[()] < sorter.bins.shape[0])


@qd.kernel(graph=True)
def kernel_radix_sort(
    sorter: RadixSortScratch,
    keys: qd.Tensor,
    keys_buffer: qd.Tensor,
    permutation: qd.Tensor,
    permutation_buffer: qd.Tensor,
    n_keys: qd.Tensor,
    key_bits: qd.template(),
):
    """Sort keys stably together with their permutation in a launch of its own (see 'func_radix_sort')."""
    func_radix_sort(sorter, keys, keys_buffer, permutation, permutation_buffer, n_keys, key_bits)
