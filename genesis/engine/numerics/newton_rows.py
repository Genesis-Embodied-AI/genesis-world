"""The rows of the linear system of a Newton solve, one block row of 3 unknowns per point, grouped by participant."""

from enum import IntEnum

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field, iter_state_arrays
from genesis.utils.misc import qd_destroy_data

from .bcoo import BCOOMatrixScratch, bcoo_matrix_sizes, func_add_by_key, func_read_entry, func_set_n_triplets


class NewtonSegment(IntEnum):
    """
    The participants of a Newton solve that own rows, in the order of their rows.

    The nodes of the sparse grid of the implicit MPM solver come first, so that the row of a node keeps its index. The
    vertices of the cloth follow, environment by environment.
    """

    MPM_NODES = 0
    CLOTH_VERTICES = 1


N_SEGMENTS = len(NewtonSegment)

ROWS = Size("rows")
# The smallest capacity of the rows, and the slack of a capacity grown past an overflow
MIN_ROWS = 64 * 64
ROWS_GROWTH_FACTOR = 1.2


class NewtonRows(Data, kind=DataKind.SCRATCH, shape=(ROWS,)):
    """
    The linear system of the Newton iterations, one row of 3 unknowns per point, and its solve.

    The rows of segment s (see NewtonSegment) span `segments_start[s]` to `segments_start[s + 1]`, the last entry
    counting every row, and its triplets span `segments_triplet_start[s]` to `segments_triplet_start[s + 1]`, both set
    once per substep. `gradient` is the gradient of the incremental potential at the current iterate, the right-hand
    side, `solution` the solution of the Hessian against it, and `r`, `z`, `p` and `Ap` the residual, the
    preconditioned residual, the search direction and its product with the Hessian of the preconditioned conjugate
    gradient (PCG). `jacobi` holds the inverses of the diagonal blocks of the Hessian, the block Jacobi preconditioner.
    `rows_env` is the environment of every row, the rows of an environment following one another within a segment. The
    matrix holds the upper triangle of the Hessian.
    """

    segments_start: qd.Tensor = field(int, shape=(N_SEGMENTS + 1,), ndarray=True)
    segments_triplet_start: qd.Tensor = field(int, shape=(N_SEGMENTS + 1,), ndarray=True)
    rows_env: qd.Tensor = field(int, ndarray=True)
    gradient: qd.Tensor = field((float, 3), ndarray=True)
    solution: qd.Tensor = field((float, 3), ndarray=True)
    r: qd.Tensor = field((float, 3), ndarray=True)
    z: qd.Tensor = field((float, 3), ndarray=True)
    p: qd.Tensor = field((float, 3), ndarray=True)
    Ap: qd.Tensor = field((float, 3), ndarray=True)
    jacobi: qd.Tensor = field((float, (3, 3)), ndarray=True)
    matrix: BCOOMatrixScratch


def grow_newton_rows(
    n_rows: int, n_triplets: int, newton_rows: NewtonRows | None, newton_rows_replaced: list[NewtonRows]
) -> NewtonRows:
    """
    Return rows holding the given numbers of rows and of triplets of their Hessian: the rows given if they hold them,
    larger ones otherwise.

    Rows replaced stay referenced in `newton_rows_replaced`, with their arrays freed.
    """
    if newton_rows is not None:
        if newton_rows.rows_env.shape[0] >= n_rows and newton_rows.matrix.triplets_row.shape[0] >= n_triplets:
            return newton_rows
        # FIXME: quadrants#1074 - a launch is cached under the identity of its struct arguments, which a new struct
        # could take over from a dead one, so the rows replaced stay referenced, with their arrays freed.
        for _, array in iter_state_arrays(newton_rows):
            qd_destroy_data(array)
        newton_rows_replaced.append(newton_rows)
    max_rows = max(int(np.ceil(n_rows * ROWS_GROWTH_FACTOR)), MIN_ROWS)
    max_triplets = max(int(np.ceil(n_triplets * ROWS_GROWTH_FACTOR)), MIN_ROWS)
    return NewtonRows(sizes={ROWS: max_rows, NewtonRows.matrix: bcoo_matrix_sizes(max_triplets, block_shape=(3, 3))})


@qd.func
def func_set_segment(i_seg: int, n_rows: int, n_triplets: int, newton_rows: NewtonRows):
    """
    Place a segment of the rows right after the previous one, from its numbers of rows and of triplets.

    The segments are set in their order from a single thread, the first one starting at zero, and the last one sets the
    triplets of the matrix.
    """
    row_start = gs.qd_int(0)
    triplet_start = gs.qd_int(0)
    if i_seg > 0:
        row_start = newton_rows.segments_start[i_seg]
        triplet_start = newton_rows.segments_triplet_start[i_seg]
    newton_rows.segments_start[i_seg] = row_start
    newton_rows.segments_triplet_start[i_seg] = triplet_start
    newton_rows.segments_start[i_seg + 1] = row_start + n_rows
    newton_rows.segments_triplet_start[i_seg + 1] = triplet_start + n_triplets
    if i_seg == newton_rows.segments_start.shape[0] - 2:
        func_set_n_triplets(newton_rows.matrix, triplet_start + n_triplets)


@qd.func(requires_top_level=True)
def func_rows_precond_invert(newton_rows: NewtonRows):
    """Invert the diagonal blocks of the rows into their block Jacobi preconditioner, every block equilibrated to a unit
    diagonal, which keeps the inverse accurate for the rows of very different stiffness."""
    for i_entry in range(newton_rows.matrix.n_entries[None]):
        i_r, j_r, block = func_read_entry(newton_rows.matrix, i_entry, block_rows=3, block_cols=3)
        if i_r == j_r:
            diagonal = qd.Vector([block[0, 0], block[1, 1], block[2, 2]], dt=gs.qd_float)
            scale = qd.select(diagonal > 0.0, 1.0 / qd.sqrt(qd.select(diagonal > 0.0, diagonal, 1.0)), 1.0)
            scaling = scale.outer_product(scale)
            inverse = qd.Matrix.identity(gs.qd_float, 3)
            if (block * scaling).determinant() > 0.0:
                inverse = (block * scaling).inverse()
            newton_rows.jacobi[i_r] = inverse * scaling


@qd.func(requires_top_level=True)
def func_rows_matvec(newton_rows: NewtonRows, envs_mask: qd.Tensor, p_dot_ap: qd.Tensor, has_p_dot_ap: qd.template()):
    """
    Add the product Ap of the matrix of the rows with the search direction of every environment selected, the upper
    triangle standing for the transposed blocks of the lower one.

    With `has_p_dot_ap`, p . Ap of every environment adds to its entry in `p_dot_ap` along, in the energy form.
    """
    n_entries = newton_rows.matrix.n_entries[None]
    qd.loop_config(block_dim=256)
    for i_entry in range((n_entries + 31) // 32 * 32):
        i_r_key = gs.qd_int(-1)
        i_b_key = gs.qd_int(-1)
        product = qd.Vector.zero(gs.qd_float, 3)
        energy = gs.qd_float(0.0)
        if i_entry < n_entries:
            i_r, j_r, block = func_read_entry(newton_rows.matrix, i_entry, block_rows=3, block_cols=3)
            i_b = newton_rows.rows_env[i_r]
            if envs_mask[i_b]:
                i_r_key = i_r
                i_b_key = i_b
                p_col = newton_rows.p[j_r]
                product = block @ p_col
                energy = p_col.dot(product)
                if i_r != j_r:
                    p_row = newton_rows.p[i_r]
                    qd.atomic_add(newton_rows.Ap[j_r], block.transpose() @ p_row)
                    energy = 2.0 * p_row.dot(product)
        func_add_by_key(i_r_key, product, newton_rows.Ap, n_components=3)
        if qd.static(has_p_dot_ap):
            func_add_by_key(i_b_key, energy, p_dot_ap)
