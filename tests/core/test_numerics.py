import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.numerics.bcoo_matrix import BCOOMatrix, func_sort_reduce_bcoo, func_sym_bcoo_spmv
from genesis.engine.numerics.block_reduce import REDUCE_BLOCK_DIM, func_sum_block_partials, write_block_partial
from genesis.engine.numerics.dynamic_exclusive_sum import DynamicExclusiveSum, func_dynamic_exclusive_sum
from genesis.engine.numerics.dynamic_radix_sort import DynamicRadixSort, func_dynamic_radix_sort
from genesis.engine.numerics.linear_pcg import (
    PCGWorkspace,
    func_apply_block_jacobi,
    func_init_linear_pcg_direction,
    func_init_linear_pcg_residual,
    func_linear_pcg_direction_update,
    func_linear_pcg_matvec_update,
)
from genesis.engine.numerics.open_addressing_hash import OpenAddressingHash, func_reset_hash
from genesis.utils.misc import qd_to_numpy


@qd.kernel(graph=True)
def kernel_radix_sort(
    sorter: qd.template(),
    keys_a: qd.types.ndarray(),
    keys_b: qd.types.ndarray(),
    perm_a: qd.types.ndarray(),
    perm_b: qd.types.ndarray(),
    n: qd.types.ndarray(),
):
    func_dynamic_radix_sort(sorter, keys_a, keys_b, perm_a, perm_b, n[()])


@qd.kernel(graph=True)
def kernel_exclusive_sum(
    scanner: qd.template(), values: qd.types.ndarray(), output: qd.types.ndarray(), n: qd.types.ndarray()
):
    func_dynamic_exclusive_sum(scanner, values, output, n[()])


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize(("qd_dtype", "np_dtype"), ((qd.u32, np.uint32), (qd.u64, np.uint64)))
def test_dynamic_radix_sort_matches_stable_numpy(qd_dtype, np_dtype):
    capacity = 20_000
    rng = np.random.default_rng(7)
    sorter = DynamicRadixSort(qd_dtype, capacity)
    keys_a = qd.ndarray(qd_dtype, shape=(capacity,))
    keys_b = qd.ndarray(qd_dtype, shape=(capacity,))
    perm_a = qd.ndarray(qd.i32, shape=(capacity,))
    perm_b = qd.ndarray(qd.i32, shape=(capacity,))
    n = qd.ndarray(qd.i32, shape=())

    for count in (0, 1, 31, 257, 6144, 6145, capacity):
        keys = rng.integers(0, np.iinfo(np_dtype).max, size=capacity, dtype=np_dtype)
        keys[1::17] = keys[::17][: len(keys[1::17])]
        permutation = np.arange(capacity, dtype=np.int32)
        keys_a.from_numpy(keys)
        perm_a.from_numpy(permutation)
        n.from_numpy(np.array(count, dtype=np.int32))

        kernel_radix_sort(sorter, keys_a, keys_b, perm_a, perm_b, n)
        qd.sync()

        order = np.argsort(keys[:count], kind="stable")
        np.testing.assert_array_equal(qd_to_numpy(keys_a)[:count], keys[:count][order])
        np.testing.assert_array_equal(qd_to_numpy(perm_a)[:count], permutation[:count][order])


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_dynamic_exclusive_sum_matches_numpy():
    capacity = 100_000
    rng = np.random.default_rng(3)
    scanner = DynamicExclusiveSum(capacity)
    values = qd.ndarray(qd.i32, shape=(capacity,))
    output = qd.ndarray(qd.i32, shape=(capacity,))
    n = qd.ndarray(qd.i32, shape=())
    data = rng.integers(0, 8, size=capacity, dtype=np.int32)
    values.from_numpy(data)

    for count in (0, 1, 129, 3072, 3073, capacity):
        n.from_numpy(np.array(count, dtype=np.int32))
        kernel_exclusive_sum(scanner, values, output, n)
        qd.sync()
        expected = np.concatenate(([0], np.cumsum(data[: max(count - 1, 0)], dtype=np.int32)))[:count]
        np.testing.assert_array_equal(qd_to_numpy(output)[:count], expected)


@qd.kernel
def kernel_write_triplets(
    matrix: qd.template(), rows: qd.types.ndarray(), cols: qd.types.ndarray(), values: qd.types.ndarray()
):
    for slot in range(matrix.n_triplets[()]):
        block = qd.Matrix.zero(matrix.value_type, 3, 3)
        for i, j in qd.static(qd.ndrange(3, 3)):
            block[i, j] = values[slot, i, j]
        matrix.write_triplet(slot, rows[slot], cols[slot], block)


@qd.kernel(graph=True)
def kernel_sort_reduce_and_multiply(matrix: qd.template(), x: qd.types.ndarray(), y: qd.types.ndarray()):
    func_sort_reduce_bcoo(matrix)
    for i in range(y.shape[0]):
        y[i] = 0.0
    func_sym_bcoo_spmv(matrix, x, y)


def _random_symmetric_triplets(rng, n_rows, n_triplets):
    rows = rng.integers(0, n_rows, size=n_triplets).astype(np.int32)
    cols = rng.integers(0, n_rows, size=n_triplets).astype(np.int32)
    values = rng.standard_normal((n_triplets, 3, 3))
    values[rows == cols] = 0.5 * (values[rows == cols] + values[rows == cols].transpose(0, 2, 1))
    dense = np.zeros((3 * n_rows, 3 * n_rows))
    for row, col, block in zip(rows, cols, values):
        dense[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += block
        if row != col:
            dense[3 * col : 3 * col + 3, 3 * row : 3 * row + 3] += block.T
    return rows, cols, values, dense


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_symmetric_bcoo_reduces_duplicates_and_multiplies():
    rng = np.random.default_rng(11)
    n_rows, n_triplets = 40, 500
    rows, cols, values, dense = _random_symmetric_triplets(rng, n_rows, n_triplets)
    matrix = BCOOMatrix(
        shape=(n_rows, n_rows),
        block_shape=(3, 3),
        value_type=qd.f64,
        symmetric=True,
        initial_triplets=n_triplets,
        max_triplets=n_triplets,
    )
    rows_nd, cols_nd = qd.ndarray(qd.i32, shape=(n_triplets,)), qd.ndarray(qd.i32, shape=(n_triplets,))
    values_nd = qd.ndarray(qd.f64, shape=(n_triplets, 3, 3))
    rows_nd.from_numpy(rows)
    cols_nd.from_numpy(cols)
    values_nd.from_numpy(values)
    x = rng.standard_normal(3 * n_rows)
    x_nd, y_nd = qd.ndarray(qd.f64, shape=(3 * n_rows,)), qd.ndarray(qd.f64, shape=(3 * n_rows,))
    x_nd.from_numpy(x)

    kernel_write_triplets(matrix, rows_nd, cols_nd, values_nd)
    kernel_sort_reduce_and_multiply(matrix, x_nd, y_nd)
    qd.sync()

    nnz = int(qd_to_numpy(matrix.bcoo_nnz))
    bcoo_rows = qd_to_numpy(matrix.bcoo_row)[:nnz]
    bcoo_cols = qd_to_numpy(matrix.bcoo_col)[:nnz]
    assert np.all(bcoo_rows <= bcoo_cols)
    keys = bcoo_rows.astype(np.int64) * n_rows + bcoo_cols
    assert np.all(np.diff(keys) > 0)
    expected_keys = np.unique(np.minimum(rows, cols).astype(np.int64) * n_rows + np.maximum(rows, cols))
    np.testing.assert_array_equal(keys, expected_keys)
    np.testing.assert_allclose(qd_to_numpy(y_nd), dense @ x, rtol=1e-10, atol=1e-10)


@qd.kernel
def kernel_hash_insert(table: qd.template(), keys: qd.types.ndarray()):
    for i in range(keys.shape[0]):
        table.insert(keys[i], i)


@qd.kernel
def kernel_hash_lookup(table: qd.template(), keys: qd.types.ndarray(), out: qd.types.ndarray()):
    for i in range(keys.shape[0]):
        out[i] = table.lookup(keys[i])


@qd.kernel
def kernel_hash_reset(table: qd.template()):
    func_reset_hash(table)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_open_addressing_hash_inserts_and_looks_up():
    rng = np.random.default_rng(5)
    n_keys = 10_000
    keys = np.unique(rng.integers(0, np.iinfo(np.uint64).max - 1, size=2 * n_keys, dtype=np.uint64))[:n_keys]
    missing = keys + np.uint64(1)
    missing = missing[~np.isin(missing, keys)]
    table = OpenAddressingHash(n_keys)
    keys_nd, missing_nd = qd.ndarray(qd.u64, shape=(len(keys),)), qd.ndarray(qd.u64, shape=(len(missing),))
    found_nd, absent_nd = qd.ndarray(qd.i32, shape=(len(keys),)), qd.ndarray(qd.i32, shape=(len(missing),))
    keys_nd.from_numpy(keys)
    missing_nd.from_numpy(missing)

    for _ in range(2):
        kernel_hash_reset(table)
        kernel_hash_insert(table, keys_nd)
        kernel_hash_lookup(table, keys_nd, found_nd)
        kernel_hash_lookup(table, missing_nd, absent_nd)
        qd.sync()
        np.testing.assert_array_equal(qd_to_numpy(found_nd), np.arange(len(keys)))
        assert np.all(qd_to_numpy(absent_nd) == -1)


@qd.kernel
def kernel_invert_block_diagonal(matrix: qd.template(), precond_inv_diag: qd.types.ndarray()):
    for entry in range(matrix.bcoo_nnz[()]):
        row, col, block = matrix.read_bcoo(entry)
        if row == col:
            inverse = block.inverse()
            for i, j in qd.static(qd.ndrange(3, 3)):
                precond_inv_diag[row * 9 + i * 3 + j] = inverse[i, j]


@qd.kernel(graph=True)
def kernel_pcg_solve(
    ws: qd.template(),
    matrix: qd.template(),
    precond_inv_diag: qd.types.ndarray(),
    b: qd.types.ndarray(),
    x: qd.types.ndarray(),
):
    func_sort_reduce_bcoo(matrix)
    func_init_linear_pcg_residual(ws, b, x)
    func_apply_block_jacobi(ws, precond_inv_diag)
    func_init_linear_pcg_direction(ws)
    while qd.graph.do_while(ws.cond):
        func_linear_pcg_matvec_update(ws, matrix, x)
        func_apply_block_jacobi(ws, precond_inv_diag)
        func_linear_pcg_direction_update(ws)


def _pcg_block_system(off_diagonal_scale, check_curvature):
    """A symmetric block system with SPD diagonal blocks a^T a + c I, solved by PCG with the block-Jacobi
    preconditioner, which they keep SPD whatever the off-diagonal blocks. Returns the dense matrix, the right-hand side,
    the solution and the workspace."""
    rng = np.random.default_rng(17)
    n_rows, n_triplets = 60, 400
    rows, cols, values, _ = _random_symmetric_triplets(rng, n_rows, n_triplets)
    diag_values = np.stack([a.T @ a + 30.0 * np.eye(3) for a in rng.standard_normal((n_rows, 3, 3))])
    rows = np.concatenate((rows, np.arange(n_rows, dtype=np.int32)))
    cols = np.concatenate((cols, np.arange(n_rows, dtype=np.int32)))
    values = np.concatenate((off_diagonal_scale * values, diag_values))
    n_triplets = len(rows)
    dense = np.zeros((3 * n_rows, 3 * n_rows))
    for row, col, block in zip(rows, cols, values):
        dense[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += block
        if row != col:
            dense[3 * col : 3 * col + 3, 3 * row : 3 * row + 3] += block.T

    matrix = BCOOMatrix(
        shape=(n_rows, n_rows),
        block_shape=(3, 3),
        value_type=qd.f64,
        symmetric=True,
        initial_triplets=n_triplets,
        max_triplets=n_triplets,
    )
    rows_nd, cols_nd = qd.ndarray(qd.i32, shape=(n_triplets,)), qd.ndarray(qd.i32, shape=(n_triplets,))
    values_nd = qd.ndarray(qd.f64, shape=(n_triplets, 3, 3))
    rows_nd.from_numpy(rows)
    cols_nd.from_numpy(cols)
    values_nd.from_numpy(values)
    kernel_write_triplets(matrix, rows_nd, cols_nd, values_nd)

    ws = PCGWorkspace(n_rows, qd.f64)
    ws.n_rows.from_numpy(np.array(n_rows, dtype=np.int32))
    ws.tol_rate.from_numpy(np.array(1e-20))
    ws.check_curvature.from_numpy(np.array(int(check_curvature), dtype=np.int32))
    precond_inv_diag = qd.ndarray(qd.f64, shape=(9 * n_rows,))
    b = rng.standard_normal(3 * n_rows)
    b_nd, x_nd = qd.ndarray(qd.f64, shape=(3 * n_rows,)), qd.ndarray(qd.f64, shape=(3 * n_rows,))
    b_nd.from_numpy(b)

    kernel_sort_reduce_and_multiply(matrix, x_nd, b_nd)
    b_nd.from_numpy(b)
    kernel_invert_block_diagonal(matrix, precond_inv_diag)
    kernel_pcg_solve(ws, matrix, precond_inv_diag, b_nd, x_nd)
    qd.sync()
    return dense, b, qd_to_numpy(x_nd), ws


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_linear_pcg_solves_spd_block_system():
    # A dominant diagonal makes the matrix SPD, which the curvature check accepts
    dense, b, x, ws = _pcg_block_system(0.1, check_curvature=True)
    assert np.all(np.linalg.eigvalsh(dense) > 0)
    np.testing.assert_allclose(x, np.linalg.solve(dense, b), rtol=1e-7, atol=1e-8)
    assert 0 < int(qd_to_numpy(ws.iter_remaining)) < int(qd_to_numpy(ws.max_iter))
    assert int(qd_to_numpy(ws.breakdown)) == 0


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_linear_pcg_flags_non_positive_curvature():
    # Off-diagonal blocks large enough to make the matrix indefinite, its SPD diagonal keeping the preconditioner SPD
    dense, b, x, ws = _pcg_block_system(10.0, check_curvature=True)
    assert np.linalg.eigvalsh(dense)[0] < 0
    assert int(qd_to_numpy(ws.breakdown)) == 1


@qd.kernel(graph=True)
def kernel_sum(values: qd.types.ndarray(), partials: qd.types.ndarray(), out: qd.types.ndarray(), n: qd.i32):
    qd.loop_config(block_dim=REDUCE_BLOCK_DIM)
    for i in range(partials.shape[0] * REDUCE_BLOCK_DIM):
        value = 0.0
        if i < n:
            value = values[i]
        write_block_partial(value, partials, i // REDUCE_BLOCK_DIM, qd.f64)
    func_sum_block_partials(partials, (n + REDUCE_BLOCK_DIM - 1) // REDUCE_BLOCK_DIM, out, qd.f64)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_block_partial_sum_is_reproducible():
    rng = np.random.default_rng(23)
    n = 123_457
    values = rng.standard_normal(n)
    values_nd = qd.ndarray(qd.f64, shape=(n,))
    values_nd.from_numpy(values)
    partials = qd.ndarray(qd.f64, shape=((n + REDUCE_BLOCK_DIM - 1) // REDUCE_BLOCK_DIM,))
    out = qd.ndarray(qd.f64, shape=())

    results = []
    for _ in range(3):
        kernel_sum(values_nd, partials, out, n)
        qd.sync()
        results.append(float(qd_to_numpy(out)))
    assert results[0] == results[1] == results[2]
    np.testing.assert_allclose(results[0], values.sum(), rtol=1e-12)
