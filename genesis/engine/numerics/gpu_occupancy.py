import torch

# Blocks of a resident-grid launch where no CUDA device is present (the CPU backends run the loops serially)
_FALLBACK_BLOCKS = 64


def cuda_resident_blocks(block_size: int, shared_bytes: int, registers_per_thread: int) -> int:
    """Blocks of block_size threads the device keeps resident at once, bounded by threads, shared memory and
    registers per multiprocessor: the grid of a launch whose threads grid-stride over a device-sized extent."""
    if not torch.cuda.is_available():
        return _FALLBACK_BLOCKS
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    blocks_from_threads = max(properties.max_threads_per_multi_processor // block_size, 1)
    blocks_from_shared = (
        max(properties.shared_memory_per_multiprocessor // shared_bytes, 1) if shared_bytes > 0 else blocks_from_threads
    )
    blocks_from_registers = max(properties.regs_per_multiprocessor // (registers_per_thread * block_size), 1)
    blocks_per_sm = min(
        blocks_from_threads,
        blocks_from_shared,
        blocks_from_registers,
        getattr(properties, "max_blocks_per_multi_processor", blocks_from_threads),
    )
    return max(1, properties.multi_processor_count * blocks_per_sm)
