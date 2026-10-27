import quadrants as qd

REDUCE_BLOCK_DIM = 256


@qd.func
def write_block_partial(
    value,
    partials: qd.template(),  # qd.Ndarray, one entry per launched block
    block: qd.template(),  # int expression, the index of the calling block
    value_type: qd.template(),  # Quadrants scalar data type
):
    """Reduce one value per thread of a REDUCE_BLOCK_DIM-wide block and store the block sum at partials[block].

    Called inside a top-level range loop launched with block_dim=REDUCE_BLOCK_DIM over a whole number of blocks.
    """
    total = qd.simt.block.reduce_add(value, REDUCE_BLOCK_DIM, value_type)
    if qd.simt.block.thread_idx() == 0:
        partials[block] = total


@qd.func(requires_top_level=True)
def func_sum_block_partials(
    partials: qd.template(),  # qd.Ndarray
    n_partials: qd.template(),  # int expression
    out: qd.template(),  # 0-d qd.Ndarray
    value_type: qd.template(),  # Quadrants scalar data type
):
    """out = sum of the first n_partials partials, in a fixed order, so the result is bitwise reproducible."""
    qd.loop_config(name="sum_block_partials", block_dim=REDUCE_BLOCK_DIM)
    for tid in range(REDUCE_BLOCK_DIM):
        value = value_type(0.0)
        i = tid
        while i < n_partials:
            value += partials[i]
            i += REDUCE_BLOCK_DIM
        total = qd.simt.block.reduce_add(value, REDUCE_BLOCK_DIM, value_type)
        if tid == 0:
            out[()] = total
