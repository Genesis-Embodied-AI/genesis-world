from __future__ import annotations

import quadrants as qd


@qd.func(requires_top_level=True)
def sym_bcoo_spmv_naive(matrix: qd.template(), x: qd.template(), y: qd.template()):
    """Naively accumulate a symmetric BCOO ``matrix @ x`` into ``y``.

    The matrix stores one triangle, so off-diagonal blocks are mirrored during
    multiplication. Output clearing is deliberately owned by the caller so
    this operation can participate in fused operator applications.
    """
    # TODO: Replace row-side per-block atomics with a warp head-segmented
    # reduction; keep mirrored-column atomics and benchmark the crossover.
    for entry in range(matrix.bcoo_nnz[()]):
        row = matrix.bcoo_row[entry]
        col = matrix.bcoo_col[entry]
        value_offset = entry * matrix.block_scalar_count

        for block_row in qd.static(range(matrix.block_shape[0])):
            result = matrix.value_type(0.0)
            for block_col in qd.static(range(matrix.block_shape[1])):
                value = matrix.bcoo_val[value_offset + block_row * matrix.block_shape[1] + block_col]
                result = result + value * x[col * matrix.block_shape[1] + block_col]
            qd.atomic_add(y[row * matrix.block_shape[0] + block_row], result)

        if row != col:
            for block_col in qd.static(range(matrix.block_shape[1])):
                result = matrix.value_type(0.0)
                for block_row in qd.static(range(matrix.block_shape[0])):
                    value = matrix.bcoo_val[value_offset + block_row * matrix.block_shape[1] + block_col]
                    result = result + value * x[row * matrix.block_shape[0] + block_row]
                qd.atomic_add(y[col * matrix.block_shape[1] + block_col], result)
