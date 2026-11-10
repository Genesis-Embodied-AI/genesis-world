import quadrants as qd

import genesis as gs
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.misc import qd_debug_asserts, qd_isfinite

from .bcoo_matrix import BCOOMatrixScratch, func_read_bcoo
from .fsr_reduce import add_by_key, warp_padded
from .newton_control import NewtonControl

# Threads of a block of the preconditioned conjugate gradient (PCG) launches
PCG_BLOCK = 256


class PCGScalars(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """Device-resident scalars of the preconditioned conjugate gradient (PCG) solves of the environments of a batch:
    the options of the forward and adjoint solves, the flags of the batch, then the state of every environment."""

    max_iterations: qd.Tensor = field(int, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    tol_rate: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    adjoint_tol_rate: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    is_running: qd.Tensor = field(int, shape=(), ndarray=True)
    n_iterations_left: qd.Tensor = field(int, shape=(), ndarray=True)
    is_curvature_checked: qd.Tensor = field(int, shape=(), ndarray=True)
    is_env_active: qd.Tensor = field(int, ndarray=True)
    is_env_adjoint: qd.Tensor = field(int, ndarray=True)
    # A direction of non-positive curvature (p.Ap <= 0) of an environment flags it, the operator being indefinite
    has_breakdown: qd.Tensor = field(int, ndarray=True)
    tol_sq: qd.Tensor = field(float, ndarray=True)
    rz: qd.Tensor = field(float, ndarray=True)
    rz_new: qd.Tensor = field(float, ndarray=True)
    p_dot_ap: qd.Tensor = field(float, ndarray=True)
    alpha: qd.Tensor = field(float, ndarray=True)
    beta: qd.Tensor = field(float, ndarray=True)
    is_converged: qd.Tensor = field(int, ndarray=True)


PCG_ROWS = Size("pcg_rows")


class PCGWorkspaceScratch(Data, kind=DataKind.SCRATCH):
    """Vectors of a preconditioned conjugate gradient (PCG), one 3-vector per block row.

    `rows_env` holds the environment of every block row, the rows being sorted by environment, by which the reductions
    add (see add_by_key)."""

    r: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    z: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    p: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    ap: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    rows_env: qd.Tensor = field(int, shape=(PCG_ROWS,), ndarray=True)
    n_rows: qd.Tensor = field(int, ndarray=True)


@qd.func(requires_top_level=True)
def func_linear_pcg_dot_rz(out: qd.template(), pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars):
    """Add r . z of the preconditioned residual of every running environment into its entry of an array of the
    environments."""
    qd.loop_config(name="pcg_dot_rz", block_dim=PCG_BLOCK)
    for i_r in range(warp_padded(pcg_workspace.n_rows[()])):
        env_key = gs.qd_int(-1)
        value = gs.qd_float(0.0)
        if i_r < pcg_workspace.n_rows[()]:
            i_b = pcg_workspace.rows_env[i_r]
            if pcg_scalars.is_converged[i_b] == 0:
                env_key = i_b
                value = pcg_workspace.r[i_r].dot(pcg_workspace.z[i_r])
        add_by_key(env_key, value, out)


@qd.func(requires_top_level=True)
def func_init_linear_pcg_residual(
    b: qd.template(),
    x: qd.template(),
    pcg_workspace: PCGWorkspaceScratch,
    pcg_scalars: PCGScalars,
    is_adjoint: qd.template(),
):
    """Start a linear solve from a zero solution, its residual being the right-hand side.

    With `is_adjoint`, an environment left out by `is_env_active` keeps the solution of an earlier adjoint solve. The
    preconditioner and func_init_linear_pcg_direction follow."""
    qd.loop_config(name="pcg_init_residual", block_dim=PCG_BLOCK)
    for i_r in range(pcg_workspace.n_rows[()]):
        is_kept = False
        if qd.static(is_adjoint):
            i_b = pcg_workspace.rows_env[i_r]
            is_kept = pcg_scalars.is_converged[i_b] != 0
        if not is_kept:
            x[i_r] = qd.Vector.zero(gs.qd_float, 3)
        pcg_workspace.r[i_r] = b[i_r]
        pcg_workspace.ap[i_r] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_init_linear_pcg_direction(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars):
    """Set the first search direction p = z, and add r . z to rz."""
    qd.loop_config(name="pcg_copy_z_to_p", block_dim=PCG_BLOCK)
    for i_r in range(pcg_workspace.n_rows[()]):
        pcg_workspace.p[i_r] = pcg_workspace.z[i_r]
    func_linear_pcg_dot_rz(out=pcg_scalars.rz, pcg_workspace=pcg_workspace, pcg_scalars=pcg_scalars)


@qd.func(requires_top_level=True)
def func_linear_pcg_spmv(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars, matrix: BCOOMatrixScratch):
    """Add the sparse matrix-vector product (SpMV) of the search direction to Ap, with p.Ap in energy form."""
    qd.loop_config(name="pcg_spmv_pap", block_dim=PCG_BLOCK)
    for i_nz in range(warp_padded(matrix.bcoo_nnz[()])):
        row_key = gs.qd_int(-1)
        env_key = gs.qd_int(-1)
        energy = gs.qd_float(0.0)
        product = qd.Vector.zero(gs.qd_float, 3)
        if i_nz < matrix.bcoo_nnz[()]:
            i_r = matrix.bcoo_row[i_nz]
            i_b = pcg_workspace.rows_env[i_r]
            if pcg_scalars.is_converged[i_b] == 0:
                row_key = i_r
                env_key = i_b
                i_c = matrix.bcoo_col[i_nz]
                block = func_read_bcoo(i_nz, matrix, block_rows=3, block_cols=3)[2]
                p_col = pcg_workspace.p[i_c]
                product = block @ p_col
                energy = p_col.dot(product)
                # The upper triangle stores a block off the diagonal for its transpose as well
                if i_r != i_c:
                    p_row = pcg_workspace.p[i_r]
                    qd.atomic_add(pcg_workspace.ap[i_c], block.transpose() @ p_row)
                    energy = 2.0 * p_row.dot(product)
        add_by_key(row_key, product, pcg_workspace.ap, n_components=3)
        add_by_key(env_key, energy, pcg_scalars.p_dot_ap)


@qd.func(requires_top_level=True)
def func_linear_pcg_update_xr(x: qd.template(), pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars):
    """Update the solution and the residual of every running environment along its search direction."""
    qd.loop_config(name="pcg_update_xr", block_dim=PCG_BLOCK)
    for i_r in range(pcg_workspace.n_rows[()]):
        i_b = pcg_workspace.rows_env[i_r]
        if pcg_scalars.is_converged[i_b] == 0:
            alpha = pcg_scalars.alpha[i_b]
            x[i_r] = x[i_r] + alpha * pcg_workspace.p[i_r]
            pcg_workspace.r[i_r] = pcg_workspace.r[i_r] - alpha * pcg_workspace.ap[i_r]


@qd.func(requires_top_level=True)
def func_linear_pcg_direction(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars):
    """Update the search direction of every running environment, p = z + beta p, and zero Ap for the product of the
    next iteration."""
    qd.loop_config(name="pcg_update_p", block_dim=PCG_BLOCK)
    for i_r in range(pcg_workspace.n_rows[()]):
        pcg_workspace.ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
        i_b = pcg_workspace.rows_env[i_r]
        if pcg_scalars.is_converged[i_b] == 0:
            pcg_workspace.p[i_r] = pcg_workspace.z[i_r] + pcg_scalars.beta[i_b] * pcg_workspace.p[i_r]


# ============================================ stages of the solve ============================================


@qd.func(requires_top_level=True)
def func_pcg_select_envs(pcg_scalars: PCGScalars, newton_control: NewtonControl, is_adjoint: qd.template()):
    """Take part in the next solve the environments whose Newton solve runs, or those of `is_env_adjoint` for an adjoint
    solve, which follows a converged forward solve."""
    for i_b in range(pcg_scalars.is_env_active.shape[0]):
        if qd.static(is_adjoint):
            pcg_scalars.is_env_active[i_b] = pcg_scalars.is_env_adjoint[i_b]
        else:
            pcg_scalars.is_env_active[i_b] = newton_control.is_env_running[i_b]


@qd.func(requires_top_level=True)
def func_pcg_reset(pcg_scalars: PCGScalars):
    """Reset the scalars of a solve before its owners start their rows, and re-arm the iteration budget."""
    for _ in range(1):
        pcg_scalars.n_iterations_left[()] = pcg_scalars.max_iterations[()]
    for i_b in range(pcg_scalars.rz.shape[0]):
        pcg_scalars.has_breakdown[i_b] = 0
        pcg_scalars.rz[i_b] = 0.0
        pcg_scalars.rz_new[i_b] = 0.0
        pcg_scalars.p_dot_ap[i_b] = 0.0
        pcg_scalars.is_converged[i_b] = 1 - pcg_scalars.is_env_active[i_b]


@qd.func(requires_top_level=True)
def func_pcg_arm(pcg_scalars: PCGScalars, is_adjoint: qd.template()):
    """Set the convergence threshold of every environment, the tolerance rate of the solve times |rz0|, once every owner
    set its first direction, and arm the iterations if any environment takes part."""
    for _ in range(1):
        pcg_scalars.is_running[()] = 0
    for i_b in range(pcg_scalars.rz.shape[0]):
        tol_rate = pcg_scalars.adjoint_tol_rate[()] if qd.static(is_adjoint) else pcg_scalars.tol_rate[()]
        pcg_scalars.tol_sq[i_b] = tol_rate * qd.abs(pcg_scalars.rz[i_b])
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_max(pcg_scalars.is_running[()], 1)


@qd.func(requires_top_level=True)
def func_pcg_step(pcg_scalars: PCGScalars):
    """Take the step alpha = rz / p.Ap of every running environment, once every owner added its p.Ap."""
    for i_b in range(pcg_scalars.rz.shape[0]):
        if pcg_scalars.is_converged[i_b] == 0:
            rz = pcg_scalars.rz[i_b]
            p_dot_ap = pcg_scalars.p_dot_ap[i_b]
            if pcg_scalars.is_curvature_checked[()] != 0 and p_dot_ap <= 0.0 and qd.abs(rz) > 0.0:
                pcg_scalars.has_breakdown[i_b] = 1
            pcg_scalars.alpha[i_b] = rz / p_dot_ap if qd.abs(p_dot_ap) > 0.0 else gs.qd_float(0.0)
        pcg_scalars.rz_new[i_b] = 0.0


@qd.func(requires_top_level=True)
def func_pcg_decide(pcg_scalars: PCGScalars):
    """Take beta = rz_new / rz and the convergence decision of every running environment once every owner added its r .
    z, then roll rz over and zero p.Ap."""
    for _ in range(1):
        pcg_scalars.n_iterations_left[()] = pcg_scalars.n_iterations_left[()] - 1
        pcg_scalars.is_running[()] = 0
    for i_b in range(pcg_scalars.rz.shape[0]):
        if pcg_scalars.is_converged[i_b] == 0:
            rz = pcg_scalars.rz[i_b]
            rz_new = pcg_scalars.rz_new[i_b]
            if qd.static(qd_debug_asserts()):
                assert qd_isfinite(rz_new), "PCG: r . z of environment %d is not finite" % i_b
            if qd.abs(rz_new) <= pcg_scalars.tol_sq[i_b] or not qd_isfinite(rz_new):
                pcg_scalars.is_converged[i_b] = 1
            elif pcg_scalars.n_iterations_left[()] > 0:
                qd.atomic_max(pcg_scalars.is_running[()], 1)
            pcg_scalars.beta[i_b] = rz_new / rz if qd.abs(rz) > 0.0 else gs.qd_float(0.0)
            pcg_scalars.rz[i_b] = rz_new
        pcg_scalars.p_dot_ap[i_b] = 0.0
