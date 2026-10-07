import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.bcoo_matrix import BCOOMatrix, sort_reduce_bcoo
from genesis.utils.misc import qd_to_numpy


@qd.kernel(graph=True, fastcache=True)
def assemble_nonsymmetric(
    matrix: qd.template(),
    triplet_rows: qd.types.ndarray(qd.i32, ndim=1),
    triplet_cols: qd.types.ndarray(qd.i32, ndim=1),
    triplet_values: qd.types.ndarray(qd.f64, ndim=3),
    bcoo_rows: qd.types.ndarray(qd.i32, ndim=1),
    bcoo_cols: qd.types.ndarray(qd.i32, ndim=1),
    bcoo_values: qd.types.ndarray(qd.f64, ndim=3),
):
    matrix.write_triplet(0, 1, 0, qd.Matrix([[1.0, 2.0], [3.0, 4.0]]))
    matrix.write_triplet(1, 0, 1, qd.Matrix([[5.0, 6.0], [7.0, 8.0]]))
    matrix.write_triplet(2, 1, 0, qd.Matrix([[9.0, 10.0], [11.0, 12.0]]))
    for slot in range(matrix.n_triplets[()]):
        row, col, block = matrix.read_triplet(slot)
        triplet_rows[slot] = row
        triplet_cols[slot] = col
        for i, j in qd.static(qd.ndrange(2, 2)):
            triplet_values[slot, i, j] = block[i, j]
    sort_reduce_bcoo(matrix)
    for entry in range(matrix.bcoo_nnz[()]):
        row, col, block = matrix.read_bcoo(entry)
        bcoo_rows[entry] = row
        bcoo_cols[entry] = col
        for i, j in qd.static(qd.ndrange(2, 2)):
            bcoo_values[entry, i, j] = block[i, j]


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("value_type", [qd.f32, qd.f64])
def test_general_bcoo_preserves_nonsymmetric_blocks_and_reduces_duplicates(value_type):
    matrix = BCOOMatrix(
        shape=(2, 2),
        block_shape=(2, 2),
        value_type=value_type,
        symmetric=False,
        initial_triplets=3,
        max_triplets=3,
    )
    triplet_rows = qd.ndarray(qd.i32, shape=(3,))
    triplet_cols = qd.ndarray(qd.i32, shape=(3,))
    triplet_values = qd.ndarray(qd.f64, shape=(3, 2, 2))
    bcoo_rows = qd.ndarray(qd.i32, shape=(3,))
    bcoo_cols = qd.ndarray(qd.i32, shape=(3,))
    bcoo_values = qd.ndarray(qd.f64, shape=(3, 2, 2))

    assemble_nonsymmetric(
        matrix,
        triplet_rows,
        triplet_cols,
        triplet_values,
        bcoo_rows,
        bcoo_cols,
        bcoo_values,
    )
    qd.sync()

    nnz = int(qd_to_numpy(matrix.bcoo_nnz))
    np.testing.assert_array_equal(qd_to_numpy(triplet_rows), [1, 0, 1])
    np.testing.assert_array_equal(qd_to_numpy(triplet_cols), [0, 1, 0])
    np.testing.assert_allclose(
        qd_to_numpy(triplet_values),
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
    )
    np.testing.assert_array_equal(qd_to_numpy(bcoo_rows)[:nnz], [0, 1])
    np.testing.assert_array_equal(qd_to_numpy(bcoo_cols)[:nnz], [1, 0])

    values = qd_to_numpy(bcoo_values)[:nnz]
    expected_blocks = np.array(
        [
            [[5.0, 6.0], [7.0, 8.0]],
            [[10.0, 12.0], [14.0, 16.0]],
        ]
    )
    np.testing.assert_allclose(values, expected_blocks)
