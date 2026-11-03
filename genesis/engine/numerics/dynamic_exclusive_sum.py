import numpy as np

import quadrants as qd

import genesis as gs

_BLOCK_THREADS = 128
_ITEMS_PER_THREAD = 24
# Chunks of the scan on a CPU backend, each one thread's serial scan of a contiguous range
CPU_CHUNKS = 256


@qd.data_oriented
class DynamicExclusiveSum:
    """Decoupled look-back exclusive sum of i32 values, whose live count is read on the device.

    On a GPU, warp and block operations scan each tile of the input, and each tile waits on the status that its
    predecessors publish. On a CPU backend, the scan runs in three phases over CPU_CHUNKS contiguous chunks instead: the
    chunk sums in parallel, their exclusive sum, then the scan of each chunk in parallel from its offset. The look-back
    needs warp operations and threads that progress concurrently, which the CPU backend of Quadrants lacks, while the
    end of each parallel loop is enough to order the three phases.
    """

    def __init__(self, capacity: int) -> None:
        assert capacity >= 1
        n_blocks = max((capacity + _BLOCK_THREADS * _ITEMS_PER_THREAD - 1) // (_BLOCK_THREADS * _ITEMS_PER_THREAD), 1)

        self.status = qd.ndarray(gs.qd_int, shape=(n_blocks,))
        self.partial = qd.ndarray(gs.qd_int, shape=(max(n_blocks, CPU_CHUNKS),))
        self.complete = qd.ndarray(gs.qd_int, shape=(n_blocks,))
        self.tile_counter = qd.ndarray(gs.qd_int, shape=())
        self.status.from_numpy(np.zeros(n_blocks, dtype=gs.np_int))
        self.partial.from_numpy(np.zeros(self.partial.shape[0], dtype=gs.np_int))
        self.complete.from_numpy(np.zeros(n_blocks, dtype=gs.np_int))
        self.tile_counter.from_numpy(np.array(0, dtype=gs.np_int))

    @qd.func(requires_top_level=True)
    def clear(self, n: qd.template()):
        qd.loop_config(name="des_init_tile_state", block_dim=128)
        needed_blocks = qd.max((n + 3071) // 3072, 1)
        for block in range(needed_blocks):
            self.status[block] = 0
            if block == 0:
                self.tile_counter[()] = 0

    @qd.func(requires_top_level=True)
    def scan(self, values: qd.template(), output: qd.template(), n: qd.template()):
        if qd.static(gs.backend == gs.cpu):
            self.scan_cpu(values, output, n)
        else:
            self.scan_gpu(values, output, n)

    @qd.func(requires_top_level=True)
    def scan_cpu(self, values: qd.template(), output: qd.template(), n: qd.template()):
        qd.loop_config(name="des_chunk_sums")
        for chunk in range(CPU_CHUNKS):
            size = (n + CPU_CHUNKS - 1) // CPU_CHUNKS
            total = gs.qd_int(0)
            for i in range(chunk * size, qd.min(chunk * size + size, n)):
                total += values[i]
            self.partial[chunk] = total
        for _ in range(1):
            running = gs.qd_int(0)
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
    def scan_gpu(self, values: qd.template(), output: qd.template(), n: qd.template()):
        self.clear(n)
        qd.loop_config(name="des_scan", block_dim=128)
        needed_blocks = qd.max((n + 3071) // 3072, 1)
        for thread in range(needed_blocks * 128):
            qd.simt.block.sync()
            tid = qd.simt.block.thread_idx()
            count = n
            launch_block = thread // 128
            if launch_block < needed_blocks:
                sh_tile_slot = qd.simt.block.SharedArray((1,), gs.qd_int)
                if tid == 0:
                    sh_tile_slot[0] = qd.atomic_add(self.tile_counter[()], 1)
                qd.simt.block.sync()
                tile = sh_tile_slot[0]
                local_prefixes = qd.Vector.zero(gs.qd_int, 24)
                sh_tile_values = qd.simt.block.SharedArray((3072,), gs.qd_int)
                sh_tile_total = qd.simt.block.SharedArray((1,), gs.qd_int)

                for item in qd.static(range(24)):
                    index = tile * 3072 + item * 128 + tid
                    value = gs.qd_int(0)
                    if index < count:
                        value = values[index]
                    sh_tile_values[item * 128 + tid] = value
                qd.simt.block.sync()

                thread_total = gs.qd_int(0)
                for item in qd.static(range(24)):
                    logical_index = tid * 24 + item
                    local_prefixes[item] = thread_total
                    thread_total = thread_total + sh_tile_values[logical_index]
                thread_prefix = qd.simt.block.exclusive_add(thread_total, 128, gs.qd_int)
                if tid == 127:
                    sh_tile_total[0] = thread_prefix + thread_total
                qd.simt.block.sync()

                sh_tile_prefix = qd.simt.block.SharedArray((1,), gs.qd_int)
                if tid == 0:
                    tile_sum = sh_tile_total[0]
                    tile_prefix = gs.qd_int(0)
                    predecessor = tile - 1
                    is_prefix_complete = tile == 0
                    if tile > 0:
                        self.partial[tile] = tile_sum
                        qd.simt.grid.mem_fence()
                        qd.atomic_exchange(self.status[tile], 1)
                    while not is_prefix_complete:
                        if predecessor < 0:
                            is_prefix_complete = True
                        else:
                            state = qd.volatile_load(self.status[predecessor])
                            if state != 0:
                                qd.simt.grid.mem_fence()
                                if state == 2:
                                    tile_prefix = tile_prefix + qd.volatile_load(self.complete[predecessor])
                                    is_prefix_complete = True
                                else:
                                    tile_prefix = tile_prefix + qd.volatile_load(self.partial[predecessor])
                                    predecessor = predecessor - 1
                    self.complete[tile] = tile_prefix + tile_sum
                    qd.simt.grid.mem_fence()
                    qd.atomic_exchange(self.status[tile], 2)
                    sh_tile_prefix[0] = tile_prefix
                qd.simt.block.sync()

                for item in qd.static(range(24)):
                    logical_index = tid * 24 + item
                    sh_tile_values[logical_index] = sh_tile_prefix[0] + thread_prefix + local_prefixes[item]
                qd.simt.block.sync()

                for item in qd.static(range(24)):
                    index = tile * 3072 + item * 128 + tid
                    if index < count:
                        output[index] = sh_tile_values[item * 128 + tid]
            qd.simt.block.sync()


@qd.func(requires_top_level=True)
def func_dynamic_exclusive_sum(values: qd.template(), output: qd.template(), scanner: qd.template(), n: qd.template()):
    """Write the exclusive prefix sum of the leading entries of an array into another array.

    The sum covers the first n entries of values and lands in output.
    """
    scanner.scan(values, output, n)


@qd.kernel
def kernel_dynamic_exclusive_sum(
    values: qd.types.ndarray(), output: qd.types.ndarray(), n: qd.types.ndarray(), scanner: qd.template()
):
    """Run func_dynamic_exclusive_sum as a launch of its own, reading the count from a device scalar."""
    scanner.scan(values, output, n[()])
