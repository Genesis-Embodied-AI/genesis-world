import quadrants as qd

REDUCE_BLOCK_DIM = 256


@qd.func
def write_block_partial(block: qd.template(), value, partials: qd.template(), value_type: qd.template()):
    """Sum one value per thread over a block of REDUCE_BLOCK_DIM threads and store the sum as the partial of the block.

    The sum lands at partials[block]. Called inside a top-level range loop launched with block_dim=REDUCE_BLOCK_DIM over
    a whole number of blocks.
    """
    total = qd.simt.block.reduce_add(value, REDUCE_BLOCK_DIM, value_type)
    if qd.simt.block.thread_idx() == 0:
        partials[block] = total
