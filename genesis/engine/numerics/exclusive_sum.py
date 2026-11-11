"""Exclusive prefix sum of 32-bit integers over a number of values read on the device."""

import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field

# The values a tile of the GPU scan sums: 24 for each of its 128 threads
TILE_VALUES = 24 * 128
# The chunks of the CPU scan, each one summed and scanned by one thread
CPU_CHUNKS = 256

SCAN_TILES = Size("scan_tiles")
SCAN_PARTIALS = Size("scan_partials")


class ExclusiveSumScratch(Data, kind=DataKind.SCRATCH):
    """
    The workspace of an exclusive prefix sum.

    A GPU tile takes its turn from `tile_counter` and publishes its sum in `partial`, then its sum and that of the tiles
    before it in `complete`, `status` telling which one it holds (decoupled look-back). The CPU scan holds the sum of
    the chunks before every chunk in `partial`.
    """

    status: qd.Tensor = field(int, shape=(SCAN_TILES,), ndarray=True)
    partial: qd.Tensor = field(int, shape=(SCAN_PARTIALS,), ndarray=True)
    complete: qd.Tensor = field(int, shape=(SCAN_TILES,), ndarray=True)
    tile_counter: qd.Tensor = field(int, shape=(), ndarray=True)


def exclusive_sum_sizes(capacity: int) -> dict[Size, int]:
    """
    Return the sizes of the workspace of a sum over a bounded number of values.

    The sum runs over at most `capacity` values.
    """
    n_tiles = max((capacity + TILE_VALUES - 1) // TILE_VALUES, 1)
    return {SCAN_TILES: n_tiles, SCAN_PARTIALS: max(n_tiles, CPU_CHUNKS)}


@qd.func(requires_top_level=True)
def func_cpu_exclusive_sum(scanner: ExclusiveSumScratch, values: qd.Tensor, output: qd.Tensor, n_values: qd.Tensor):
    N_CHUNKS = qd.static(256)
    qd.loop_config(name="exclusive_sum_cpu_chunks")
    for i_chunk in range(N_CHUNKS):
        chunk_size = (n_values[()] + N_CHUNKS - 1) // N_CHUNKS
        total = gs.qd_int(0)
        for i_val in range(i_chunk * chunk_size, qd.min(i_chunk * chunk_size + chunk_size, n_values[()])):
            total += values[i_val]
        scanner.partial[i_chunk] = total
    for _ in range(1):
        total_before = gs.qd_int(0)
        for i_chunk in range(N_CHUNKS):
            total = scanner.partial[i_chunk]
            scanner.partial[i_chunk] = total_before
            total_before += total
    qd.loop_config(name="exclusive_sum_cpu_scan")
    for i_chunk in range(N_CHUNKS):
        chunk_size = (n_values[()] + N_CHUNKS - 1) // N_CHUNKS
        total_before = scanner.partial[i_chunk]
        for i_val in range(i_chunk * chunk_size, qd.min(i_chunk * chunk_size + chunk_size, n_values[()])):
            value = values[i_val]
            output[i_val] = total_before
            total_before += value


@qd.func(requires_top_level=True)
def func_gpu_exclusive_sum(scanner: ExclusiveSumScratch, values: qd.Tensor, output: qd.Tensor, n_values: qd.Tensor):
    TILE = qd.static(3072)
    VALUES_PER_THREAD = qd.static(24)
    qd.loop_config(name="exclusive_sum_clear_tiles", block_dim=128)
    for i_tile in range(qd.max((n_values[()] + TILE - 1) // TILE, 1)):
        scanner.status[i_tile] = 0
        if i_tile == 0:
            scanner.tile_counter[()] = 0
    qd.loop_config(name="exclusive_sum_gpu_scan", block_dim=128)
    n_tiles = qd.max((n_values[()] + TILE - 1) // TILE, 1)
    for i_th in range(n_tiles * 128):
        qd.simt.block.sync()
        tid = qd.simt.block.thread_idx()
        i_blk = i_th // 128
        if i_blk < n_tiles:
            # The tiles take their turn in the order the blocks start, so that a tile only waits on tiles already
            # running
            sh_tile_slot = qd.simt.block.SharedArray((1,), gs.qd_int)
            if tid == 0:
                sh_tile_slot[0] = qd.atomic_add(scanner.tile_counter[()], 1)
            qd.simt.block.sync()
            i_tile = sh_tile_slot[0]
            local_prefixes = qd.Vector.zero(gs.qd_int, VALUES_PER_THREAD)
            sh_tile_values = qd.simt.block.SharedArray((TILE,), gs.qd_int)
            sh_tile_total = qd.simt.block.SharedArray((1,), gs.qd_int)

            for k in qd.static(range(VALUES_PER_THREAD)):
                i_val = i_tile * TILE + k * 128 + tid
                value = gs.qd_int(0)
                if i_val < n_values[()]:
                    value = values[i_val]
                sh_tile_values[k * 128 + tid] = value
            qd.simt.block.sync()

            thread_total = gs.qd_int(0)
            for k in qd.static(range(VALUES_PER_THREAD)):
                local_prefixes[k] = thread_total
                thread_total = thread_total + sh_tile_values[tid * VALUES_PER_THREAD + k]
            thread_prefix = qd.simt.block.exclusive_add(thread_total, 128, gs.qd_int)
            if tid == 127:
                sh_tile_total[0] = thread_prefix + thread_total
            qd.simt.block.sync()

            # The sum of the tiles before this one, read back tile by tile until a tile holding it whole
            sh_tile_prefix = qd.simt.block.SharedArray((1,), gs.qd_int)
            if tid == 0:
                tile_sum = sh_tile_total[0]
                tile_prefix = gs.qd_int(0)
                j_tile = i_tile - 1
                is_prefix_complete = i_tile == 0
                if i_tile > 0:
                    scanner.partial[i_tile] = tile_sum
                    qd.simt.grid.mem_fence()
                    qd.atomic_exchange(scanner.status[i_tile], 1)
                while not is_prefix_complete:
                    if j_tile < 0:
                        is_prefix_complete = True
                    else:
                        status = qd.volatile_load(scanner.status[j_tile])
                        if status != 0:
                            qd.simt.grid.mem_fence()
                            if status == 2:
                                tile_prefix = tile_prefix + qd.volatile_load(scanner.complete[j_tile])
                                is_prefix_complete = True
                            else:
                                tile_prefix = tile_prefix + qd.volatile_load(scanner.partial[j_tile])
                                j_tile = j_tile - 1
                scanner.complete[i_tile] = tile_prefix + tile_sum
                qd.simt.grid.mem_fence()
                qd.atomic_exchange(scanner.status[i_tile], 2)
                sh_tile_prefix[0] = tile_prefix
            qd.simt.block.sync()

            for k in qd.static(range(VALUES_PER_THREAD)):
                sh_tile_values[tid * VALUES_PER_THREAD + k] = sh_tile_prefix[0] + thread_prefix + local_prefixes[k]
            qd.simt.block.sync()

            for k in qd.static(range(VALUES_PER_THREAD)):
                i_val = i_tile * TILE + k * 128 + tid
                if i_val < n_values[()]:
                    output[i_val] = sh_tile_values[k * 128 + tid]
        qd.simt.block.sync()


@qd.func(requires_top_level=True)
def func_exclusive_sum(scanner: ExclusiveSumScratch, values: qd.Tensor, output: qd.Tensor, n_values: qd.Tensor):
    """
    Write the exclusive prefix sum of the leading values of an array into another.

    The sum covers the first `n_values[()]` values of `values` and lands in `output`.
    """
    if qd.static(gs.backend == gs.cpu):
        func_cpu_exclusive_sum(scanner, values, output, n_values)
    else:
        func_gpu_exclusive_sum(scanner, values, output, n_values)


@qd.kernel
def kernel_exclusive_sum(scanner: ExclusiveSumScratch, values: qd.Tensor, output: qd.Tensor, n_values: qd.Tensor):
    """Write the exclusive prefix sum of values in a launch of its own (see 'func_exclusive_sum')."""
    func_exclusive_sum(scanner, values, output, n_values)
