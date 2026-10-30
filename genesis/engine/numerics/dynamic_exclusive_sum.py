import numpy as np
import quadrants as qd

import genesis as gs

_BLOCK_THREADS = 128
_ITEMS_PER_THREAD = 24
# Chunks of the scan on a CPU backend, each one thread's serial scan of a contiguous range
CPU_CHUNKS = 256


@qd.data_oriented
class DynamicExclusiveSum:
    """Decoupled-lookback exclusive sum over i32 data, whose live count is read on the device.

    On a CPU backend, a three-phase scan over CPU_CHUNKS contiguous chunks instead: the chunk sums in parallel, their
    exclusive sum, then the scan of each chunk in parallel from its offset. The decoupled look-back scans a tile by
    warp and block operations and lets each tile wait on the status its predecessors publish, which the CPU backend of
    Quadrants has no warp operations for, and which a CPU thread that waits on another one may never see progress for;
    the three phases need neither, the ends of their parallel loops ordering them."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("DynamicExclusiveSum capacity must be positive")
        n_blocks = max(
            (capacity + _BLOCK_THREADS * _ITEMS_PER_THREAD - 1) // (_BLOCK_THREADS * _ITEMS_PER_THREAD),
            1,
        )

        self.status = qd.ndarray(qd.i32, shape=(n_blocks,))
        self.partial = qd.ndarray(qd.i32, shape=(max(n_blocks, CPU_CHUNKS),))
        self.complete = qd.ndarray(qd.i32, shape=(n_blocks,))
        self.tile_counter = qd.ndarray(qd.i32, shape=())
        self.status.from_numpy(np.zeros(n_blocks, dtype=np.int32))
        self.partial.from_numpy(np.zeros(self.partial.shape[0], dtype=np.int32))
        self.complete.from_numpy(np.zeros(n_blocks, dtype=np.int32))
        self.tile_counter.from_numpy(np.array(0, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def clear(
        self,
        n: qd.template(),  # int
    ):
        qd.loop_config(name="des_init_tile_state", block_dim=128)
        needed_blocks = qd.max((n + 3071) // 3072, 1)
        for block in range(needed_blocks):
            self.status[block] = 0
            if block == 0:
                self.tile_counter[()] = 0

    @qd.func(requires_top_level=True)
    def scan(
        self,
        values: qd.template(),  # qd.Ndarray
        output: qd.template(),  # qd.Ndarray
        n: qd.template(),  # int
    ):
        if qd.static(gs.backend == gs.cpu):
            self.scan_cpu(values, output, n)
        else:
            self.scan_gpu(values, output, n)

    @qd.func(requires_top_level=True)
    def scan_cpu(
        self,
        values: qd.template(),  # qd.Ndarray
        output: qd.template(),  # qd.Ndarray
        n: qd.template(),  # int
    ):
        qd.loop_config(name="des_chunk_sums")
        for chunk in range(CPU_CHUNKS):
            size = (n + CPU_CHUNKS - 1) // CPU_CHUNKS
            total = qd.i32(0)
            for i in range(chunk * size, qd.min(chunk * size + size, n)):
                total += values[i]
            self.partial[chunk] = total
        for _ in range(1):
            running = qd.i32(0)
            for chunk in range(CPU_CHUNKS):
                total = self.partial[chunk]
                self.partial[chunk] = running
                running += total
        qd.loop_config(name="des_chunk_scans")
        for chunk in range(CPU_CHUNKS):
            size = (n + CPU_CHUNKS - 1) // CPU_CHUNKS
            running = self.partial[chunk]
            for i in range(chunk * size, qd.min(chunk * size + size, n)):
                value = values[i]
                output[i] = running
                running += value

    @qd.func(requires_top_level=True)
    def scan_gpu(
        self,
        values: qd.template(),  # qd.Ndarray
        output: qd.template(),  # qd.Ndarray
        n: qd.template(),  # int
    ):
        self.clear(n)
        qd.loop_config(name="des_scan", block_dim=128)
        needed_blocks = qd.max((n + 3071) // 3072, 1)
        for thread in range(needed_blocks * 128):
            qd.simt.block.sync()
            tid = qd.simt.block.thread_idx()
            count = n
            launch_block = thread // 128
            if launch_block < needed_blocks:
                tile_slot = qd.simt.block.SharedArray((1,), qd.i32)
                if tid == 0:
                    tile_slot[0] = qd.atomic_add(self.tile_counter[()], 1)
                qd.simt.block.sync()
                tile = tile_slot[0]
                local_prefixes = qd.Vector.zero(qd.i32, 24)
                tile_values = qd.simt.block.SharedArray((3072,), qd.i32)
                tile_total = qd.simt.block.SharedArray((1,), qd.i32)

                for item in qd.static(range(24)):
                    index = tile * 3072 + item * 128 + tid
                    value = qd.i32(0)
                    if index < count:
                        value = values[index]
                    tile_values[item * 128 + tid] = value
                qd.simt.block.sync()

                thread_total = qd.i32(0)
                for item in qd.static(range(24)):
                    logical_index = tid * 24 + item
                    local_prefixes[item] = thread_total
                    thread_total = thread_total + tile_values[logical_index]
                thread_prefix = qd.simt.block.exclusive_add(thread_total, 128, qd.i32)
                if tid == 127:
                    tile_total[0] = thread_prefix + thread_total
                qd.simt.block.sync()

                tile_prefix_shared = qd.simt.block.SharedArray((1,), qd.i32)
                if tid == 0:
                    tile_sum = tile_total[0]
                    tile_prefix = qd.i32(0)
                    predecessor = tile - 1
                    found_complete = tile == 0
                    if tile > 0:
                        self.partial[tile] = tile_sum
                        qd.simt.grid.mem_fence()
                        qd.atomic_exchange(self.status[tile], 1)
                    while not found_complete:
                        if predecessor < 0:
                            found_complete = True
                        else:
                            state = qd.volatile_load(self.status[predecessor])
                            if state != 0:
                                qd.simt.grid.mem_fence()
                                if state == 2:
                                    tile_prefix = tile_prefix + qd.volatile_load(self.complete[predecessor])
                                    found_complete = True
                                else:
                                    tile_prefix = tile_prefix + qd.volatile_load(self.partial[predecessor])
                                    predecessor = predecessor - 1
                    self.complete[tile] = tile_prefix + tile_sum
                    qd.simt.grid.mem_fence()
                    qd.atomic_exchange(self.status[tile], 2)
                    tile_prefix_shared[0] = tile_prefix
                qd.simt.block.sync()

                for item in qd.static(range(24)):
                    logical_index = tid * 24 + item
                    tile_values[logical_index] = tile_prefix_shared[0] + thread_prefix + local_prefixes[item]
                qd.simt.block.sync()

                for item in qd.static(range(24)):
                    index = tile * 3072 + item * 128 + tid
                    if index < count:
                        output[index] = tile_values[item * 128 + tid]
            qd.simt.block.sync()


@qd.func(requires_top_level=True)
def func_dynamic_exclusive_sum(
    scanner: qd.template(),  # DynamicExclusiveSum
    values: qd.template(),  # qd.Ndarray
    output: qd.template(),  # qd.Ndarray
    n: qd.template(),  # int
):
    """Write the exclusive prefix sum of the first n values to output."""
    scanner.scan(values, output, n)


@qd.kernel
def kernel_dynamic_exclusive_sum(
    scanner: qd.template(),  # DynamicExclusiveSum
    values: qd.types.ndarray(),
    output: qd.types.ndarray(),
    n: qd.types.ndarray(),  # i32 scalar
):
    """func_dynamic_exclusive_sum as its own launch, with the count a device scalar."""
    scanner.scan(values, output, n[()])
