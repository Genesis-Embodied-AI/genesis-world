import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.bcoo_matrix import get_bcoo_matrix, sort_reduce_bcoo, write_bcoo_block
from genesis.utils.misc import qd_to_numpy


@qd.kernel(graph=True, fastcache=True)
def assemble_nonsymmetric(matrix: qd.template()):
    write_bcoo_block(matrix, 0, 1, 0, qd.Matrix([[1.0, 2.0], [3.0, 4.0]]))
    write_bcoo_block(matrix, 1, 0, 1, qd.Matrix([[5.0, 6.0], [7.0, 8.0]]))
    write_bcoo_block(matrix, 2, 1, 0, qd.Matrix([[9.0, 10.0], [11.0, 12.0]]))
    sort_reduce_bcoo(matrix)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_general_bcoo_preserves_nonsymmetric_blocks_and_reduces_duplicates():
    matrix = get_bcoo_matrix(
        shape=(2, 2),
        block_shape=(2, 2),
        value_type=qd.f64,
        symmetric=False,
        initial_triplets=3,
        max_triplets=3,
    )

    assemble_nonsymmetric(matrix)
    qd.sync()

    nnz = int(qd_to_numpy(matrix.bcoo_nnz))
    np.testing.assert_array_equal(qd_to_numpy(matrix.bcoo_row)[:nnz], [0, 1])
    np.testing.assert_array_equal(qd_to_numpy(matrix.bcoo_col)[:nnz], [1, 0])

    values = qd_to_numpy(matrix.bcoo_val)[: nnz * 4].reshape(nnz, 2, 2)
    expected_blocks = np.array(
        [
            [[5.0, 6.0], [7.0, 8.0]],
            [[10.0, 12.0], [14.0, 16.0]],
        ]
    )
    np.testing.assert_allclose(values, expected_blocks)
