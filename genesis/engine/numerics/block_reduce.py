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


@qd.func(requires_top_level=True)
def func_sum_block_partials(
    partials: qd.template(), out: qd.template(), n_partials: qd.template(), value_type: qd.template()
):
    """Sum the leading block partials into a scalar in a fixed order, which makes the result bitwise reproducible.

    The sum of the first n_partials entries of partials lands in out.
    """
    qd.loop_config(name="sum_block_partials", block_dim=REDUCE_BLOCK_DIM)
    for tid in range(REDUCE_BLOCK_DIM):
        value = value_type(0.0)
        for i_chunk_ in range((n_partials + REDUCE_BLOCK_DIM - 1) // REDUCE_BLOCK_DIM):
            i = i_chunk_ * REDUCE_BLOCK_DIM + tid
            if i < n_partials:
                value += partials[i]
        total = qd.simt.block.reduce_add(value, REDUCE_BLOCK_DIM, value_type)
        if tid == 0:
            out[()] = total
