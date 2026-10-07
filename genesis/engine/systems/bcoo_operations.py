from __future__ import annotations

import quadrants as qd


@qd.func(requires_top_level=True)
def sym_bcoo_spmv_naive(
    matrix: qd.template(),  # BCOOMatrix
    x: qd.template(),  # qd.Ndarray
    y: qd.template(),  # qd.Ndarray
):
    """Naively accumulate a symmetric BCOO ``matrix @ x`` into ``y``.

    The matrix stores one triangle, so off-diagonal blocks are mirrored during
    multiplication. Output clearing is deliberately owned by the caller so
    this operation can participate in fused operator applications.
    """
    # TODO: Replace row-side per-block atomics with a warp head-segmented
    # reduction; keep mirrored-column atomics and benchmark the crossover.
    for entry in range(matrix.bcoo_nnz[()]):
        row, col, block = matrix.read_bcoo(entry)

        for block_row in qd.static(range(matrix.block_rows)):
            result = block[block_row, 0] * 0.0
            for block_col in qd.static(range(matrix.block_cols)):
                result = result + block[block_row, block_col] * x[col * matrix.block_cols + block_col]
            qd.atomic_add(y[row * matrix.block_rows + block_row], result)

        if row != col:
            for block_col in qd.static(range(matrix.block_cols)):
                result = block[0, block_col] * 0.0
                for block_row in qd.static(range(matrix.block_rows)):
                    result = result + block[block_row, block_col] * x[row * matrix.block_rows + block_row]
                qd.atomic_add(y[col * matrix.block_cols + block_col], result)
