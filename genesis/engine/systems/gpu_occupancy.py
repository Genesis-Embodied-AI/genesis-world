from __future__ import annotations

import torch


def cuda_resident_blocks(
    block_size: int,
    shared_bytes: int,
    registers_per_thread: int,
) -> int:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    blocks_from_threads = max(properties.max_threads_per_multi_processor // block_size, 1)
    blocks_from_shared = (
        max(properties.shared_memory_per_multiprocessor // shared_bytes, 1) if shared_bytes > 0 else blocks_from_threads
    )
    blocks_from_registers = max(
        properties.regs_per_multiprocessor // (registers_per_thread * block_size),
        1,
    )
    blocks_per_sm = min(
        blocks_from_threads,
        blocks_from_shared,
        blocks_from_registers,
        getattr(properties, "max_blocks_per_multi_processor", blocks_from_threads),
    )
    return max(1, properties.multi_processor_count * blocks_per_sm)
