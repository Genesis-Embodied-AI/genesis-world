import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field

_BLOCK_THREADS = 128
_ITEMS_PER_THREAD = 24
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
# Chunks of the scan on a CPU backend, each one thread's serial scan of a contiguous range
CPU_CHUNKS = 256

# The tiles of a scan, and the entries of the partial sums of its tiles and of its CPU chunks
SCAN_TILES = Size("scan_tiles")
SCAN_PARTIALS = Size("scan_partials")


class DynamicExclusiveSumScratch(Data, kind=DataKind.SCRATCH):
    """Tile state of an exclusive sum of 32-bit integer (i32) values, with a device-side count."""

    # One look-back lane per tile, as the radix sort's look-back has one per digit
    status: qd.Tensor = field(int, shape=(SCAN_TILES, 1), ndarray=True)
    partial: qd.Tensor = field(int, shape=(SCAN_PARTIALS, 1), ndarray=True)
    complete: qd.Tensor = field(int, shape=(SCAN_TILES, 1), ndarray=True)
    tile_counter: qd.Tensor = field(int, ndarray=True)


def dynamic_exclusive_sum_sizes(capacity: int) -> dict[Size, int]:
    """Return the sizes of the scratch of an exclusive sum."""
    n_tiles = max((capacity + _TILE_ITEMS - 1) // _TILE_ITEMS, 1)
    return {SCAN_TILES: n_tiles, SCAN_PARTIALS: max(n_tiles, CPU_CHUNKS)}


@qd.func
def func_lookback_prefix(
    status: qd.template(),
    partial: qd.template(),
    complete: qd.template(),
    i_tile: int,
    i_lane: int,
    tile_sum: int,
):
    """Publish the sum of a tile in one lane and return the sum of the tiles before it (decoupled look-back)."""
    tile_prefix = gs.qd_int(0)
    j_tile = i_tile - 1
    is_prefix_complete = i_tile == 0
    if i_tile > 0:
        partial[i_tile, i_lane] = tile_sum
        qd.simt.grid.mem_fence()
        qd.atomic_exchange(status[i_tile, i_lane], 1)
    while not is_prefix_complete:
        if j_tile < 0:
            is_prefix_complete = True
        else:
            state = qd.volatile_load(status[j_tile, i_lane])
            if state != 0:
                qd.simt.grid.mem_fence()
                if state == 2:
                    tile_prefix = tile_prefix + qd.volatile_load(complete[j_tile, i_lane])
                    is_prefix_complete = True
                else:
                    tile_prefix = tile_prefix + qd.volatile_load(partial[j_tile, i_lane])
                    j_tile = j_tile - 1
    complete[i_tile, i_lane] = tile_prefix + tile_sum
    qd.simt.grid.mem_fence()
    qd.atomic_exchange(status[i_tile, i_lane], 2)
    return tile_prefix


@qd.func(requires_top_level=True)
def _func_clear_scan(n: qd.Tensor, scanner: DynamicExclusiveSumScratch):
    qd.loop_config(name="exclusive_sum_init_tile_state", block_dim=128)
    n_tiles = qd.max((n[()] + 3071) // 3072, 1)
    for i_tile in range(n_tiles):
        scanner.status[i_tile, 0] = 0
        if i_tile == 0:
            scanner.tile_counter[()] = 0


@qd.func(requires_top_level=True)
def _func_scan_cpu(values: qd.template(), output: qd.template(), n: qd.Tensor, scanner: DynamicExclusiveSumScratch):
    qd.loop_config(name="exclusive_sum_chunk_sums")
    for i_chunk in range(CPU_CHUNKS):
        size = (n[()] + CPU_CHUNKS - 1) // CPU_CHUNKS
        total = gs.qd_int(0)
        for i_val in range(i_chunk * size, qd.min(i_chunk * size + size, n[()])):
            total += values[i_val]
        scanner.partial[i_chunk, 0] = total
    for _ in range(1):
        running = gs.qd_int(0)
        for i_chunk in range(CPU_CHUNKS):
            total = scanner.partial[i_chunk, 0]
            scanner.partial[i_chunk, 0] = running
            running += total
    qd.loop_config(name="exclusive_sum_chunk_scans")
    for i_chunk in range(CPU_CHUNKS):
        size = (n[()] + CPU_CHUNKS - 1) // CPU_CHUNKS
        running = scanner.partial[i_chunk, 0]
        for i_val in range(i_chunk * size, qd.min(i_chunk * size + size, n[()])):
            value = values[i_val]
            output[i_val] = running
            running += value


@qd.func(requires_top_level=True)
def _func_scan_gpu(values: qd.template(), output: qd.template(), n: qd.Tensor, scanner: DynamicExclusiveSumScratch):
    _func_clear_scan(n, scanner)
    qd.loop_config(name="exclusive_sum_scan", block_dim=128)
    n_tiles = qd.max((n[()] + 3071) // 3072, 1)
    for i_th in range(n_tiles * 128):
        qd.simt.block.sync()
        tid = qd.simt.block.thread_idx()
        i_blk = i_th // 128
        if i_blk < n_tiles:
            sh_tile_slot = qd.simt.block.SharedArray((1,), gs.qd_int)
            if tid == 0:
                sh_tile_slot[0] = qd.atomic_add(scanner.tile_counter[()], 1)
            qd.simt.block.sync()
            i_tile = sh_tile_slot[0]
            local_prefixes = qd.Vector.zero(gs.qd_int, 24)
            sh_tile_values = qd.simt.block.SharedArray((3072,), gs.qd_int)
            sh_tile_total = qd.simt.block.SharedArray((1,), gs.qd_int)

            for k in qd.static(range(24)):
                i_val = i_tile * 3072 + k * 128 + tid
                value = gs.qd_int(0)
                if i_val < n[()]:
                    value = values[i_val]
                sh_tile_values[k * 128 + tid] = value
            qd.simt.block.sync()

            thread_total = gs.qd_int(0)
            for k in qd.static(range(24)):
                i_loc = tid * 24 + k
                local_prefixes[k] = thread_total
                thread_total = thread_total + sh_tile_values[i_loc]
            thread_prefix = qd.simt.block.exclusive_add(thread_total, 128, gs.qd_int)
            if tid == 127:
                sh_tile_total[0] = thread_prefix + thread_total
            qd.simt.block.sync()

            sh_tile_prefix = qd.simt.block.SharedArray((1,), gs.qd_int)
            if tid == 0:
                sh_tile_prefix[0] = func_lookback_prefix(
                    scanner.status, scanner.partial, scanner.complete, i_tile, 0, sh_tile_total[0]
                )
            qd.simt.block.sync()

            for k in qd.static(range(24)):
                i_loc = tid * 24 + k
                sh_tile_values[i_loc] = sh_tile_prefix[0] + thread_prefix + local_prefixes[k]
            qd.simt.block.sync()

            for k in qd.static(range(24)):
                i_val = i_tile * 3072 + k * 128 + tid
                if i_val < n[()]:
                    output[i_val] = sh_tile_values[k * 128 + tid]
        qd.simt.block.sync()


@qd.func(requires_top_level=True)
def func_dynamic_exclusive_sum(
    values: qd.template(), output: qd.template(), n: qd.Tensor, scanner: DynamicExclusiveSumScratch
):
    """Write the exclusive prefix sum of the first entries of an array into another."""
    if qd.static(gs.backend == gs.cpu):
        _func_scan_cpu(values, output, n, scanner)
    else:
        _func_scan_gpu(values, output, n, scanner)


@qd.kernel
def kernel_dynamic_exclusive_sum(
    values: qd.Tensor, output: qd.Tensor, n: qd.Tensor, scanner: DynamicExclusiveSumScratch
):
    """Run func_dynamic_exclusive_sum as a launch of its own."""
    func_dynamic_exclusive_sum(values, output, n, scanner)
