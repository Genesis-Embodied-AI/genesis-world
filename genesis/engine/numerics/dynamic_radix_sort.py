import numpy as np

import quadrants as qd

import genesis as gs

_RADIX_DIGITS = 256
_BLOCK_THREADS = 256
_ITEMS_PER_THREAD = 24
# Chunks of a radix pass on a CPU backend, each one thread's serial histogram and scatter of a contiguous range
CPU_CHUNKS = 256


@qd.func
def _rank_warp_striped_items(
    digits: qd.template(),  # qd.Vector
    ranks: qd.template(),  # qd.Vector
    block_prefixes: qd.template(),  # qd.simt.block.SharedArray
):
    tid = qd.simt.block.thread_idx()
    lane = qd.i32(qd.simt.subgroup.invocation_id())
    subgroup = tid // 32
    sh_warp_offsets = qd.simt.block.SharedArray((8 * 256,), qd.i32)
    sh_match_masks = qd.simt.block.SharedArray((8 * 256,), qd.i32)

    for item in qd.static(range(8)):
        index = subgroup * 256 + lane + item * 32
        sh_warp_offsets[index] = 0
        sh_match_masks[index] = 0
    qd.simt.subgroup.sync()
    qd.simt.subgroup.mem_fence()

    for item in qd.static(range(24)):
        qd.atomic_add(sh_warp_offsets[subgroup * 256 + digits[item]], 1)
    qd.simt.block.sync()

    block_count = qd.i32(0)
    if tid < 256:
        for warp in qd.static(range(8)):
            index = warp * 256 + tid
            warp_count = sh_warp_offsets[index]
            sh_warp_offsets[index] = block_count
            block_count = block_count + warp_count
    block_prefix = qd.simt.block.exclusive_add(block_count, 256, qd.i32)
    if tid < 256:
        block_prefixes[tid] = block_prefix
    qd.simt.block.sync()

    lane_mask_le = qd.simt.subgroup.lanemask_le(qd.u32(lane))
    lane_mask = qd.i32(qd.u32(1) << qd.u32(lane))
    for item in qd.static(range(24)):
        digit = digits[item]
        index = subgroup * 256 + digit
        qd.atomic_or(sh_match_masks[index], lane_mask)
        qd.simt.subgroup.sync()
        qd.simt.subgroup.mem_fence()

        match = qd.u32(sh_match_masks[index])
        leader = qd.i32(31) - qd.i32(qd.math.clz(match))
        count = qd.i32(qd.math.popcnt(match & lane_mask_le))
        offset = qd.i32(0)
        if lane == leader:
            offset = qd.atomic_add(sh_warp_offsets[index], count)
        offset = qd.simt.subgroup.shuffle(offset, qd.u32(leader))
        if lane == leader:
            sh_match_masks[index] = 0
        qd.simt.subgroup.sync()
        qd.simt.subgroup.mem_fence()
        ranks[item] = offset + count - 1
    return qd.Vector([block_count, block_prefix], dt=qd.i32)


@qd.data_oriented
class DynamicRadixSort:
    """Stable OneSweep radix sort (Adinets and Merrill, 2022) of keys and their permutation, with a device-side count.

    The count is a device value, so one compiled launch serves any count up to the capacity the sorter was built for.

    On a GPU, OneSweep ranks the keys of a tile by warp ballots and block-wide shared-memory histograms, and chains the
    tiles by a decoupled look-back. On a CPU backend, each pass is a chunked counting sort instead: the digit histogram
    of each of CPU_CHUNKS contiguous chunks in parallel, the digit-major exclusive sum of the histograms, then the
    stable scatter of each chunk in parallel from its offsets. The CPU backend of Quadrants lacks the warp operations of
    OneSweep, and a CPU thread ranks a whole chunk serially, which keeps the scatter stable.
    """

    def __init__(self, key_dtype, capacity: int) -> None:
        assert key_dtype in (qd.u32, qd.u64)
        assert capacity >= 1

        self.key_dtype = key_dtype
        n_passes = 4 if key_dtype == qd.u32 else 8
        self.n_passes = n_passes
        n_blocks = max((capacity + _BLOCK_THREADS * _ITEMS_PER_THREAD - 1) // (_BLOCK_THREADS * _ITEMS_PER_THREAD), 1)

        self.bins = qd.ndarray(qd.i32, shape=(n_passes, _RADIX_DIGITS))
        self.lookback_status = qd.ndarray(qd.i32, shape=(n_blocks, _RADIX_DIGITS))
        self.lookback_partial = qd.ndarray(qd.i32, shape=(n_blocks, _RADIX_DIGITS))
        self.lookback_complete = qd.ndarray(qd.i32, shape=(n_blocks, _RADIX_DIGITS))
        self.tile_counters = qd.ndarray(qd.i32, shape=(n_passes,))
        # Pass loop of kernel_radix_sort on the device: the first of the two passes of an iteration, and the condition
        self.pass_idx = qd.ndarray(qd.i32, shape=())
        self.pass_cond = qd.ndarray(qd.i32, shape=())

        # Digit offsets of every chunk in the CPU passes
        self.chunk_offsets = qd.ndarray(qd.i32, shape=(CPU_CHUNKS if gs.backend == gs.cpu else 1, _RADIX_DIGITS))

        self.bins.from_numpy(np.zeros((n_passes, _RADIX_DIGITS), dtype=np.int32))
        lookback_shape = (n_blocks, _RADIX_DIGITS)
        self.lookback_status.from_numpy(np.zeros(lookback_shape, dtype=np.int32))
        self.lookback_partial.from_numpy(np.zeros(lookback_shape, dtype=np.int32))
        self.lookback_complete.from_numpy(np.zeros(lookback_shape, dtype=np.int32))
        self.tile_counters.from_numpy(np.zeros(n_passes, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def clear_bins(self):
        qd.loop_config(name="drs_memset_bins", block_dim=256)
        for index in range(self.n_passes * 256):
            radix_pass = index // 256
            digit = index - radix_pass * 256
            self.bins[radix_pass, digit] = 0

    @qd.func(requires_top_level=True)
    def histogram(
        self,
        keys: qd.template(),  # qd.Ndarray
        n: qd.template(),  # int
    ):
        qd.loop_config(name="drs_histogram", block_dim=128)
        for thread in range(128 * 128):
            tid = qd.simt.block.thread_idx()
            block = thread // 128
            sh_histogram = qd.simt.block.SharedArray((self.n_passes * 256,), qd.i32)
            for radix_pass in qd.static(range(self.n_passes)):
                for item in qd.static(range(2)):
                    sh_histogram[radix_pass * 256 + item * 128 + tid] = 0
            qd.simt.block.sync()

            for i_chunk_ in range((n + 128 * 128 * 16 - 1) // (128 * 128 * 16)):
                base = i_chunk_ * 128 * 128 * 16 + block * 128 * 16 + tid
                if base < n:
                    for item in qd.static(range(16)):
                        index = base + item * 128
                        key32 = qd.u32(0)
                        key64 = qd.u64(0)
                        is_valid = index < n
                        if is_valid:
                            if qd.static(self.key_dtype == qd.u32):
                                key32 = keys[index]
                            else:
                                key64 = keys[index]
                        for radix_pass in qd.static(range(self.n_passes)):
                            digit = qd.i32(0)
                            if is_valid:
                                shift = qd.static(radix_pass * 8)
                                if qd.static(self.key_dtype == qd.u32):
                                    digit = qd.i32((key32 >> shift) & qd.u32(255))
                                else:
                                    digit = qd.i32((key64 >> shift) & qd.u64(255))
                                qd.atomic_add(sh_histogram[radix_pass * 256 + digit], 1)
            qd.simt.block.sync()
            for radix_pass in qd.static(range(self.n_passes)):
                for item in qd.static(range(2)):
                    digit = item * 128 + tid
                    qd.atomic_add(self.bins[radix_pass, digit], sh_histogram[radix_pass * 256 + digit])
            qd.simt.block.sync()

    @qd.func(requires_top_level=True)
    def scan_bins(self):
        qd.loop_config(name="drs_exclusive_sum", block_dim=256)
        for index in range(self.n_passes * 256):
            radix_pass = index // 256
            digit = index - radix_pass * 256
            value = self.bins[radix_pass, digit]
            prefix = qd.simt.block.exclusive_add(value, 256, qd.i32)
            self.bins[radix_pass, digit] = prefix

    @qd.func(requires_top_level=True)
    def clear_pass(
        self,
        radix_pass,
        n: qd.template(),  # int
    ):
        qd.loop_config(name="drs_memset_lookback", block_dim=256)
        needed_blocks = qd.max((n + 6143) // 6144, 1)
        for index in range(needed_blocks * 256):
            block = index // 256
            digit = index - block * 256
            self.lookback_status[block, digit] = 0
            if index == 0:
                self.tile_counters[radix_pass] = 0

    @qd.func(requires_top_level=True)
    def onesweep_pass(
        self,
        keys_in: qd.template(),  # qd.Ndarray
        keys_out: qd.template(),  # qd.Ndarray
        perm_in: qd.template(),  # qd.Ndarray
        perm_out: qd.template(),  # qd.Ndarray
        n: qd.template(),  # int
        radix_pass,
    ):
        qd.loop_config(name="drs_onesweep", block_dim=256)
        needed_blocks = qd.max((n + 6143) // 6144, 1)
        for thread in range(needed_blocks * 256):
            shift = qd.u32(radix_pass * 8)
            qd.simt.block.sync()
            tid = qd.simt.block.thread_idx()
            count = n
            launch_block = thread // 256
            sh_tile_slot = qd.simt.block.SharedArray((1,), qd.i32)
            if tid == 0:
                if launch_block < needed_blocks:
                    sh_tile_slot[0] = qd.atomic_add(self.tile_counters[radix_pass], 1)
                else:
                    sh_tile_slot[0] = needed_blocks
            qd.simt.block.sync()

            tile = sh_tile_slot[0]
            if tile < needed_blocks:
                sh_block_prefixes = qd.simt.block.SharedArray((256,), qd.i32)
                sh_tile_prefixes = qd.simt.block.SharedArray((256,), qd.i32)
                sh_sorted_keys = qd.simt.block.SharedArray((6144,), qd.u64)
                sh_sorted_perm = qd.simt.block.SharedArray((6144,), qd.i32)

                keys32 = qd.Vector.zero(qd.u32, 24)
                keys64 = qd.Vector.zero(qd.u64, 24)
                digits = qd.Vector.zero(qd.i32, 24)
                local_ranks = qd.Vector.zero(qd.i32, 24)

                for item in qd.static(range(24)):
                    input_index = tile * 6144 + (tid // 32) * 24 * 32 + item * 32 + tid % 32
                    is_valid = input_index < count
                    key32 = qd.u32(0xFFFFFFFF)
                    key64 = qd.u64(0xFFFFFFFFFFFFFFFF)
                    if is_valid:
                        if qd.static(self.key_dtype == qd.u32):
                            key32 = keys_in[input_index]
                        else:
                            key64 = keys_in[input_index]

                    digit_key = qd.u32(0xFF)
                    if qd.static(self.key_dtype == qd.u32):
                        digit_key = (key32 >> shift) & qd.u32(0xFF)
                    else:
                        digit_key = qd.u32((key64 >> qd.u64(shift)) & qd.u64(0xFF))
                    digit = qd.i32(digit_key)
                    keys32[item] = key32
                    keys64[item] = key64
                    digits[item] = digit

                rank_metadata = _rank_warp_striped_items(digits, local_ranks, sh_block_prefixes)
                invalid_count = qd.min(qd.max(tile * 6144 + 6144 - count, 0), 6144)
                local_count = rank_metadata[0]
                if tid == 255:
                    local_count = local_count - invalid_count
                for item in qd.static(range(24)):
                    input_index = tile * 6144 + (tid // 32) * 24 * 32 + item * 32 + tid % 32
                    if input_index < count:
                        digit = digits[item]
                        sorted_index = sh_block_prefixes[digit] + local_ranks[item]
                        if qd.static(self.key_dtype == qd.u32):
                            sh_sorted_keys[sorted_index] = qd.u64(keys32[item])
                        else:
                            sh_sorted_keys[sorted_index] = keys64[item]
                        sh_sorted_perm[sorted_index] = perm_in[input_index]

                tile_prefix = qd.i32(0)
                if tid < 256:
                    predecessor = tile - 1
                    is_prefix_complete = tile == 0
                    if tile > 0:
                        self.lookback_partial[tile, tid] = local_count
                        qd.simt.grid.mem_fence()
                        qd.atomic_exchange(self.lookback_status[tile, tid], 1)

                    while not is_prefix_complete:
                        if predecessor < 0:
                            is_prefix_complete = True
                        else:
                            state = qd.volatile_load(self.lookback_status[predecessor, tid])
                            if state != 0:
                                qd.simt.grid.mem_fence()
                                if state == 2:
                                    tile_prefix = tile_prefix + qd.volatile_load(
                                        self.lookback_complete[predecessor, tid]
                                    )
                                    is_prefix_complete = True
                                else:
                                    tile_prefix = tile_prefix + qd.volatile_load(
                                        self.lookback_partial[predecessor, tid]
                                    )
                                    predecessor = predecessor - 1

                    self.lookback_complete[tile, tid] = tile_prefix + local_count
                    qd.simt.grid.mem_fence()
                    qd.atomic_exchange(self.lookback_status[tile, tid], 2)
                    sh_tile_prefixes[tid] = tile_prefix
                qd.simt.block.sync()

                valid_count = qd.min(qd.max(count - tile * 6144, 0), 6144)
                for item in qd.static(range(24)):
                    sorted_index = item * 256 + tid
                    if sorted_index < valid_count:
                        key = sh_sorted_keys[sorted_index]
                        digit = qd.i32((key >> qd.u64(shift)) & qd.u64(0xFF))
                        output_index = (
                            self.bins[radix_pass, digit]
                            + sh_tile_prefixes[digit]
                            + sorted_index
                            - sh_block_prefixes[digit]
                        )
                        if qd.static(self.key_dtype == qd.u32):
                            keys_out[output_index] = qd.u32(key)
                        else:
                            keys_out[output_index] = key
                        perm_out[output_index] = sh_sorted_perm[sorted_index]
            qd.simt.block.sync()

    @qd.func(requires_top_level=True)
    def start_passes(self):
        for _ in range(1):
            self.pass_idx[()] = 0
            self.pass_cond[()] = 1

    @qd.func(requires_top_level=True)
    def advance_passes(self):
        for _ in range(1):
            self.pass_idx[()] = self.pass_idx[()] + 2
            self.pass_cond[()] = qd.i32(self.pass_idx[()] < self.n_passes)

    @qd.func
    def digit(self, key, radix_pass: qd.template()):
        shift = qd.static(radix_pass * 8)
        value = qd.i32(0)
        if qd.static(self.key_dtype == qd.u32):
            value = qd.i32((key >> qd.u32(shift)) & qd.u32(255))
        else:
            value = qd.i32((key >> qd.u64(shift)) & qd.u64(255))
        return value

    @qd.func(requires_top_level=True)
    def cpu_pass(
        self,
        keys_in: qd.template(),  # qd.Ndarray
        keys_out: qd.template(),  # qd.Ndarray
        perm_in: qd.template(),  # qd.Ndarray
        perm_out: qd.template(),  # qd.Ndarray
        n: qd.template(),  # int
        radix_pass: qd.template(),  # int
    ):
        """Run one stable counting-sort pass of the keys on one of their digits, over CPU_CHUNKS chunks.

        The digit is the byte of rank radix_pass, counted from the least significant one.
        """
        qd.loop_config(name="drs_cpu_histogram")
        for chunk in range(CPU_CHUNKS):
            for digit in range(_RADIX_DIGITS):
                self.chunk_offsets[chunk, digit] = 0
            size = (n + CPU_CHUNKS - 1) // CPU_CHUNKS
            for i in range(chunk * size, qd.min(chunk * size + size, n)):
                self.chunk_offsets[chunk, self.digit(keys_in[i], radix_pass)] += 1
        for _ in range(1):
            running = qd.i32(0)
            for digit in range(_RADIX_DIGITS):
                for chunk in range(CPU_CHUNKS):
                    count = self.chunk_offsets[chunk, digit]
                    self.chunk_offsets[chunk, digit] = running
                    running += count
        qd.loop_config(name="drs_cpu_scatter")
        for chunk in range(CPU_CHUNKS):
            size = (n + CPU_CHUNKS - 1) // CPU_CHUNKS
            for i in range(chunk * size, qd.min(chunk * size + size, n)):
                key = keys_in[i]
                digit = self.digit(key, radix_pass)
                output_index = self.chunk_offsets[chunk, digit]
                self.chunk_offsets[chunk, digit] = output_index + 1
                keys_out[output_index] = key
                perm_out[output_index] = perm_in[i]

    @qd.func(requires_top_level=True)
    def cpu_sort(
        self,
        keys_a: qd.template(),  # qd.Ndarray
        keys_b: qd.template(),  # qd.Ndarray
        perm_a: qd.template(),  # qd.Ndarray
        perm_b: qd.template(),  # qd.Ndarray
        n: qd.template(),  # int
    ):
        """Run every pass of func_dynamic_radix_sort on a CPU backend, unrolled, alternating between the two halves."""
        for radix_pass in qd.static(range(self.n_passes)):
            if qd.static(radix_pass % 2 == 0):
                self.cpu_pass(keys_a, keys_b, perm_a, perm_b, n, radix_pass)
            else:
                self.cpu_pass(keys_b, keys_a, perm_b, perm_a, n, radix_pass)


@qd.func(requires_top_level=True)
def func_dynamic_radix_sort(
    sorter: qd.template(),  # DynamicRadixSort
    keys_a: qd.template(),  # qd.Ndarray
    keys_b: qd.template(),  # qd.Ndarray
    perm_a: qd.template(),  # qd.Ndarray
    perm_b: qd.template(),  # qd.Ndarray
    n: qd.template(),  # int
):
    """Sort keys stably together with their permutation inside a graph kernel, the result landing in the input arrays.

    The first n entries of keys_a and perm_a are sorted. keys_b and perm_b are the scratch halves of the double buffer.
    Every pass count is even, so the result always lands back in the 'a' halves.

    On a GPU, the OneSweep passes run in a device loop, two per iteration (a to b, then b to a), with the pass index a
    device value. The graph holds the code of two passes whatever the key width, whereas unrolling the 8 passes of a u64
    key would inline 8 copies of a pass and its shared-memory tiles, and multiply the compile time. A CPU pass is three
    short loops (see cpu_pass), cheap to unroll, whose digit shift is a compile-time constant.
    """
    if qd.static(gs.backend == gs.cpu):
        sorter.cpu_sort(keys_a, keys_b, perm_a, perm_b, n)
    else:
        sorter.clear_bins()
        sorter.histogram(keys_a, n)
        sorter.scan_bins()
        sorter.start_passes()
        while qd.graph.do_while(sorter.pass_cond):
            sorter.clear_pass(sorter.pass_idx[()], n)
            sorter.onesweep_pass(keys_a, keys_b, perm_a, perm_b, n, sorter.pass_idx[()])
            sorter.clear_pass(sorter.pass_idx[()] + 1, n)
            sorter.onesweep_pass(keys_b, keys_a, perm_b, perm_a, n, sorter.pass_idx[()] + 1)
            sorter.advance_passes()


@qd.kernel(graph=True)
def kernel_radix_sort(
    sorter: qd.template(),  # DynamicRadixSort
    keys_a: qd.types.ndarray(),
    keys_b: qd.types.ndarray(),
    perm_a: qd.types.ndarray(),
    perm_b: qd.types.ndarray(),
    n: qd.types.ndarray(),  # i32 scalar
):
    """Run func_dynamic_radix_sort as a graph of its own, reading the count from a device scalar."""
    func_dynamic_radix_sort(sorter, keys_a, keys_b, perm_a, perm_b, n[()])


def dynamic_radix_sort(sorter, keys_a, keys_b, perm_a, perm_b, n) -> None:
    """Sort keys stably together with their permutation, the result landing in the input arrays.

    The first n entries of keys_a and perm_a are sorted, n being a device scalar (see func_dynamic_radix_sort).
    """
    kernel_radix_sort(sorter, keys_a, keys_b, perm_a, perm_b, n)
