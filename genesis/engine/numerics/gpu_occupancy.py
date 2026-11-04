import torch

import genesis as gs

# Blocks of a resident-grid launch on the CPU backend, and on a GPU whose properties torch cannot query
_FALLBACK_BLOCKS = 64


def cuda_resident_blocks(block_size: int, shared_bytes: int, registers_per_thread: int) -> int:
    """Return the number of blocks that the CUDA device keeps resident at once, the grid of a grid-stride launch.

    Each block holds block_size threads, which use shared_bytes of shared memory per block and registers_per_thread
    registers each. The count is bounded by the threads, the shared memory and the registers of a multiprocessor. It is
    _FALLBACK_BLOCKS on the CPU backend and on a GPU without a torch.cuda device.
    """
    if gs.backend == gs.cpu or not torch.cuda.is_available():
        return _FALLBACK_BLOCKS
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    blocks_from_threads = max(properties.max_threads_per_multi_processor // block_size, 1)
    blocks_from_shared = (
        max(properties.shared_memory_per_multiprocessor // shared_bytes, 1) if shared_bytes > 0 else blocks_from_threads
    )
    blocks_from_registers = max(properties.regs_per_multiprocessor // (registers_per_thread * block_size), 1)
    blocks_per_multiprocessor = min(blocks_from_threads, blocks_from_shared, blocks_from_registers)
    return max(1, properties.multi_processor_count * blocks_per_multiprocessor)
