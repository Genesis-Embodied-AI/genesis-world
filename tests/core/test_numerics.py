import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.numerics.bcoo_matrix import BCOOMatrix, func_sort_reduce_bcoo, sort_reduce_bcoo
from genesis.engine.numerics.dynamic_exclusive_sum import DynamicExclusiveSum, func_dynamic_exclusive_sum
from genesis.engine.numerics.dynamic_radix_sort import DynamicRadixSort, func_dynamic_radix_sort
from genesis.engine.numerics.linear_pcg import (
    PCGScalars,
    PCGWorkspace,
    func_init_linear_pcg_direction,
    func_init_linear_pcg_residual,
    func_linear_pcg_direction,
    func_linear_pcg_dot_rz,
    func_linear_pcg_spmv,
    func_linear_pcg_update_xr,
    func_pcg_arm,
    func_pcg_decide,
    func_pcg_reset,
    func_pcg_select_envs,
    func_pcg_step,
)
from genesis.engine.numerics.newton_control import NewtonControl
from genesis.engine.numerics.open_addressing_hash import OpenAddressingHash, func_reset_hash
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose, assert_equal


@qd.kernel(graph=True)
def kernel_radix_sort(
    sorter: qd.template(),
    keys_a: qd.types.ndarray(),
    keys_b: qd.types.ndarray(),
    perm_a: qd.types.ndarray(),
    perm_b: qd.types.ndarray(),
    n: qd.types.ndarray(),
):
    func_dynamic_radix_sort(keys_a, keys_b, perm_a, perm_b, sorter, n[()])


@qd.kernel(graph=True)
def kernel_exclusive_sum(
    scanner: qd.template(), values: qd.types.ndarray(), output: qd.types.ndarray(), n: qd.types.ndarray()
):
    func_dynamic_exclusive_sum(values, output, scanner, n[()])


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(("qd_dtype", "np_dtype"), ((qd.u32, np.uint32), (qd.u64, np.uint64)))
def test_dynamic_radix_sort_matches_stable_numpy(qd_dtype, np_dtype):
    capacity = 20_000
    rng = np.random.default_rng(7)
    sorter = DynamicRadixSort(qd_dtype, capacity)
    keys_a = qd.ndarray(qd_dtype, shape=(capacity,))
    keys_b = qd.ndarray(qd_dtype, shape=(capacity,))
    perm_a = qd.ndarray(gs.qd_int, shape=(capacity,))
    perm_b = qd.ndarray(gs.qd_int, shape=(capacity,))
    n = qd.ndarray(gs.qd_int, shape=())

    for count in (0, 1, 31, 257, 6144, 6145, capacity):
        keys = rng.integers(0, np.iinfo(np_dtype).max, size=capacity, dtype=np_dtype)
        keys[1::17] = keys[::17][: len(keys[1::17])]
        permutation = np.arange(capacity, dtype=gs.np_int)
        keys_a.from_numpy(keys)
        perm_a.from_numpy(permutation)
        n.from_numpy(np.array(count, dtype=gs.np_int))

        kernel_radix_sort(sorter, keys_a, keys_b, perm_a, perm_b, n)
        qd.sync()

        order = np.argsort(keys[:count], kind="stable")
        assert_equal(qd_to_numpy(keys_a)[:count], keys[:count][order])
        assert_equal(qd_to_numpy(perm_a)[:count], permutation[:count][order])


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_dynamic_exclusive_sum_matches_numpy():
    capacity = 100_000
    rng = np.random.default_rng(3)
    scanner = DynamicExclusiveSum(capacity)
    values = qd.ndarray(gs.qd_int, shape=(capacity,))
    output = qd.ndarray(gs.qd_int, shape=(capacity,))
    n = qd.ndarray(gs.qd_int, shape=())
    data = rng.integers(0, 8, size=capacity, dtype=gs.np_int)
    values.from_numpy(data)

    for count in (0, 1, 129, 3072, 3073, capacity):
        n.from_numpy(np.array(count, dtype=gs.np_int))
        kernel_exclusive_sum(scanner, values, output, n)
        qd.sync()
        expected = np.concatenate(([0], np.cumsum(data[: max(count - 1, 0)], dtype=gs.np_int)))[:count]
        assert_equal(qd_to_numpy(output)[:count], expected)


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
def kernel_sort_reduce_and_multiply(pcg_workspace: qd.template(), pcg_scalars: qd.template(), matrix: qd.template()):
    func_sort_reduce_bcoo(matrix)
    func_linear_pcg_spmv(pcg_workspace, pcg_scalars, matrix)


def _random_symmetric_triplets(rng, n_rows, n_triplets):
    rows = rng.integers(0, n_rows, size=n_triplets).astype(gs.np_int)
    cols = rng.integers(0, n_rows, size=n_triplets).astype(gs.np_int)
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
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_symmetric_bcoo_reduces_duplicates_and_multiplies():
    rng = np.random.default_rng(11)
    n_rows, n_triplets = 40, 500
    rows, cols, values, dense = _random_symmetric_triplets(rng, n_rows, n_triplets)
    matrix = BCOOMatrix(
        shape=(n_rows, n_rows),
        block_shape=(3, 3),
        value_type=gs.qd_float,
        symmetric=True,
        initial_triplets=n_triplets,
        max_triplets=n_triplets,
    )
    rows_nd, cols_nd = qd.ndarray(gs.qd_int, shape=(n_triplets,)), qd.ndarray(gs.qd_int, shape=(n_triplets,))
    values_nd = qd.ndarray(gs.qd_float, shape=(n_triplets, 3, 3))
    rows_nd.from_numpy(rows)
    cols_nd.from_numpy(cols)
    values_nd.from_numpy(values)
    x = rng.standard_normal(3 * n_rows)
    pcg_workspace = PCGWorkspace(n_rows, gs.qd_float)
    pcg_workspace.n_rows.from_numpy(np.array(n_rows, dtype=gs.np_int))
    pcg_workspace.p.from_numpy(x)
    pcg_scalars = PCGScalars(gs.qd_float, max_iter=1, tol_rate=0.0)

    kernel_write_triplets(matrix, rows_nd, cols_nd, values_nd)
    kernel_sort_reduce_and_multiply(pcg_workspace, pcg_scalars, matrix)
    qd.sync()

    nnz = int(qd_to_numpy(matrix.bcoo_nnz))
    bcoo_rows = qd_to_numpy(matrix.bcoo_row)[:nnz]
    bcoo_cols = qd_to_numpy(matrix.bcoo_col)[:nnz]
    assert np.all(bcoo_rows <= bcoo_cols)
    keys = bcoo_rows * n_rows + bcoo_cols
    assert np.all(np.diff(keys) > 0)
    expected_keys = np.unique(np.minimum(rows, cols) * n_rows + np.maximum(rows, cols))
    assert_equal(keys, expected_keys)
    assert_allclose(qd_to_numpy(pcg_workspace.ap), dense @ x, tol=1e-10)
    assert_allclose(qd_to_numpy(pcg_scalars.p_dot_ap), x @ dense @ x, tol=1e-10)


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
def test_open_addressing_hash_inserts_and_looks_up():
    rng = np.random.default_rng(5)
    n_keys = 10_000
    keys = np.unique(rng.integers(0, np.iinfo(np.uint64).max - 1, size=2 * n_keys, dtype=np.uint64))[:n_keys]
    missing = keys + np.uint64(1)
    missing = missing[~np.isin(missing, keys)]
    table = OpenAddressingHash(n_keys)
    keys_nd, missing_nd = qd.ndarray(qd.u64, shape=(len(keys),)), qd.ndarray(qd.u64, shape=(len(missing),))
    found_nd, absent_nd = qd.ndarray(gs.qd_int, shape=(len(keys),)), qd.ndarray(gs.qd_int, shape=(len(missing),))
    keys_nd.from_numpy(keys)
    missing_nd.from_numpy(missing)

    for _ in range(2):
        kernel_hash_reset(table)
        kernel_hash_insert(table, keys_nd)
        kernel_hash_lookup(table, keys_nd, found_nd)
        kernel_hash_lookup(table, missing_nd, absent_nd)
        qd.sync()
        assert_equal(qd_to_numpy(found_nd), np.arange(len(keys)))
        assert_equal(qd_to_numpy(absent_nd), -1)


@qd.kernel
def kernel_invert_block_diagonal(matrix: qd.template(), precond_inv_diag: qd.types.ndarray()):
    for entry in range(matrix.bcoo_nnz[()]):
        row, col, block = matrix.read_bcoo(entry)
        if row == col:
            inverse = block.inverse()
            for i, j in qd.static(qd.ndrange(3, 3)):
                precond_inv_diag[row * 9 + i * 3 + j] = inverse[i, j]


@qd.func(requires_top_level=True)
def func_apply_block_jacobi(precond_inv_diag: qd.template(), pcg_workspace: qd.template()):
    for row in range(pcg_workspace.n_rows[()]):
        for i in qd.static(range(3)):
            value = pcg_workspace.value_type(0.0)
            for j in qd.static(range(3)):
                value += precond_inv_diag[row * 9 + i * 3 + j] * pcg_workspace.r[row * 3 + j]
            pcg_workspace.z[row * 3 + i] = value


@qd.kernel(graph=True)
def kernel_pcg_solve(
    pcg_workspace: qd.template(),
    pcg_scalars: qd.template(),
    newton_control: qd.template(),
    matrix: qd.template(),
    precond_inv_diag: qd.types.ndarray(),
    b: qd.types.ndarray(),
    x: qd.types.ndarray(),
    is_adjoint: qd.template(),
):
    func_pcg_select_envs(pcg_scalars, newton_control, is_adjoint)
    func_pcg_reset(pcg_scalars)
    func_init_linear_pcg_residual(b, x, pcg_workspace, pcg_scalars, is_adjoint)
    func_apply_block_jacobi(precond_inv_diag, pcg_workspace)
    func_init_linear_pcg_direction(pcg_workspace, pcg_scalars)
    func_pcg_arm(pcg_scalars)
    while qd.graph.do_while(pcg_scalars.is_running):
        func_linear_pcg_spmv(pcg_workspace, pcg_scalars, matrix)
        func_pcg_step(pcg_scalars)
        func_linear_pcg_update_xr(x, pcg_workspace, pcg_scalars)
        func_apply_block_jacobi(precond_inv_diag, pcg_workspace)
        func_linear_pcg_dot_rz(pcg_workspace, pcg_scalars)
        func_pcg_decide(pcg_scalars)
        func_linear_pcg_direction(pcg_workspace, pcg_scalars)


def _pcg_block_system(off_diagonal_scales, check_curvature, pcg_scalars=None, x_init=None):
    """Solve random symmetric block systems, one per environment, by block-Jacobi preconditioned conjugate gradient (PCG).

    The diagonal blocks a^T a + c I are symmetric positive definite (SPD), which keeps the preconditioner SPD whatever
    the off-diagonal blocks, scaled by the entry of 'off_diagonal_scales' of their environment. With 'check_curvature',
    the solve flags a non-positive curvature in its scalars. Return the dense matrix of every environment, the
    right-hand side, the solution and the scalars of the solve, whose rows run environment by environment.

    Without 'pcg_scalars' the solve is a forward one over every environment. With the scalars of an earlier solve and
    its solution 'x_init', it is an adjoint solve over the environments of 'is_env_adjoint', which leave the other
    environments at 'x_init'. The matrices of the two solves differ by their off-diagonal scales alone.
    """
    rng = np.random.default_rng(17)
    n_envs = len(off_diagonal_scales)
    n_env_rows, n_env_triplets = 60, 400
    n_rows = n_envs * n_env_rows
    rows, cols, values, denses = [], [], [], []
    for i_b, off_diagonal_scale in enumerate(off_diagonal_scales):
        env_rows, env_cols, env_values, _ = _random_symmetric_triplets(rng, n_env_rows, n_env_triplets)
        diag_values = np.stack([a.T @ a + 30.0 * np.eye(3) for a in rng.standard_normal((n_env_rows, 3, 3))])
        env_rows = np.concatenate((env_rows, np.arange(n_env_rows, dtype=gs.np_int)))
        env_cols = np.concatenate((env_cols, np.arange(n_env_rows, dtype=gs.np_int)))
        env_values = np.concatenate((off_diagonal_scale * env_values, diag_values))
        dense = np.zeros((3 * n_env_rows, 3 * n_env_rows))
        for row, col, block in zip(env_rows, env_cols, env_values):
            dense[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += block
            if row != col:
                dense[3 * col : 3 * col + 3, 3 * row : 3 * row + 3] += block.T
        rows.append(env_rows + i_b * n_env_rows)
        cols.append(env_cols + i_b * n_env_rows)
        values.append(env_values)
        denses.append(dense)
    rows, cols, values = np.concatenate(rows), np.concatenate(cols), np.concatenate(values)
    n_triplets = len(rows)

    matrix = BCOOMatrix(
        shape=(n_rows, n_rows),
        block_shape=(3, 3),
        value_type=gs.qd_float,
        symmetric=True,
        initial_triplets=n_triplets,
        max_triplets=n_triplets,
    )
    rows_nd, cols_nd = qd.ndarray(gs.qd_int, shape=(n_triplets,)), qd.ndarray(gs.qd_int, shape=(n_triplets,))
    values_nd = qd.ndarray(gs.qd_float, shape=(n_triplets, 3, 3))
    rows_nd.from_numpy(rows)
    cols_nd.from_numpy(cols)
    values_nd.from_numpy(values)
    kernel_write_triplets(matrix, rows_nd, cols_nd, values_nd)

    pcg_workspace = PCGWorkspace(n_rows, gs.qd_float)
    pcg_workspace.n_rows.from_numpy(np.array(n_rows, dtype=gs.np_int))
    pcg_workspace.row_env.from_numpy(np.repeat(np.arange(n_envs, dtype=gs.np_int), n_env_rows))
    is_adjoint = pcg_scalars is not None
    if not is_adjoint:
        pcg_scalars = PCGScalars(gs.qd_float, max_iter=1024, tol_rate=1e-20, n_envs=n_envs)
    pcg_scalars.is_curvature_checked.from_numpy(np.array(int(check_curvature), dtype=gs.np_int))
    newton_control = NewtonControl(gs.qd_float, 1, 1, 1.0, False, n_envs)
    newton_control.is_env_running.fill(1)
    precond_inv_diag = qd.ndarray(gs.qd_float, shape=(9 * n_rows,))
    b = rng.standard_normal(3 * n_rows)
    b_nd, x_nd = qd.ndarray(gs.qd_float, shape=(3 * n_rows,)), qd.ndarray(gs.qd_float, shape=(3 * n_rows,))
    b_nd.from_numpy(b)
    if is_adjoint:
        x_nd.from_numpy(x_init)

    sort_reduce_bcoo(matrix)
    kernel_invert_block_diagonal(matrix, precond_inv_diag)
    kernel_pcg_solve(
        pcg_workspace, pcg_scalars, newton_control, matrix, precond_inv_diag, b_nd, x_nd, is_adjoint=is_adjoint
    )
    qd.sync()
    return denses, b, qd_to_numpy(x_nd), pcg_scalars


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_linear_pcg_solves_spd_block_system():
    # A dominant diagonal makes the matrix symmetric positive definite (SPD), which the curvature check accepts
    (dense,), b, x, pcg_scalars = _pcg_block_system(off_diagonal_scales=(0.1,), check_curvature=True)
    assert np.all(np.linalg.eigvalsh(dense) > 0)
    assert_allclose(x, np.linalg.solve(dense, b), rtol=1e-7, atol=1e-8)
    assert 0 < int(qd_to_numpy(pcg_scalars.iter_remaining)) < int(qd_to_numpy(pcg_scalars.max_iter))
    assert_equal(qd_to_numpy(pcg_scalars.has_breakdown), 0)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_linear_pcg_flags_non_positive_curvature():
    # Off-diagonal blocks large enough to make the matrix of the second environment indefinite, its symmetric positive
    # definite (SPD) diagonal keeping the preconditioner SPD. The first environment solves as it does alone.
    (dense_spd, dense_indefinite), b, x, pcg_scalars = _pcg_block_system(
        off_diagonal_scales=(0.1, 10.0), check_curvature=True
    )
    assert np.linalg.eigvalsh(dense_spd)[0] > 0
    assert np.linalg.eigvalsh(dense_indefinite)[0] < 0
    assert_equal(qd_to_numpy(pcg_scalars.has_breakdown), (0, 1))
    n_env_dofs = dense_spd.shape[0]
    assert_allclose(x[:n_env_dofs], np.linalg.solve(dense_spd, b[:n_env_dofs]), rtol=1e-7, atol=1e-8)

    # The retry of the adjoint solve over the broken environment alone, on a matrix made SPD, leaves the solution of the
    # other environment as it was. This covers the keep guard of func_init_linear_pcg_residual, which the solutions of
    # the other participants (Rigid, FEM, PBD) repeat in their own func_pcg_init.
    pcg_scalars.is_env_adjoint.from_numpy(np.array((0, 1), dtype=gs.np_int))
    (_, dense_retry), b, x_retry, pcg_scalars = _pcg_block_system(
        off_diagonal_scales=(0.1, 0.1), check_curvature=False, pcg_scalars=pcg_scalars, x_init=x
    )
    assert np.linalg.eigvalsh(dense_retry)[0] > 0
    assert_equal(x_retry[:n_env_dofs], x[:n_env_dofs])
    assert_allclose(x_retry[n_env_dofs:], np.linalg.solve(dense_retry, b[n_env_dofs:]), rtol=1e-7, atol=1e-8)
