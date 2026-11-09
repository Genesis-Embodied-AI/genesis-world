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

    max_iter: qd.Tensor = field(int, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    tol_rate: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    adjoint_tol_rate: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    is_running: qd.Tensor = field(int, shape=(), ndarray=True)
    iter_remaining: qd.Tensor = field(int, shape=(), ndarray=True)
    is_curvature_checked: qd.Tensor = field(int, shape=(), ndarray=True)
    env_active: qd.Tensor = field(int, ndarray=True)
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
    """Vectors of a preconditioned conjugate gradient (PCG), one 3-vector per block row, and `row_env`, the environment
    of every block row, the rows sorted by environment. The reductions add by the environment of the rows (see
    add_by_key)."""

    r: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    z: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    p: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    ap: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    row_env: qd.Tensor = field(int, shape=(PCG_ROWS,), ndarray=True)
    n_rows: qd.Tensor = field(int, ndarray=True)


@qd.func(requires_top_level=True)
def _func_dot_into(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars, is_new: qd.template()):
    if qd.static(is_new):
        _func_dot(pcg_scalars.rz_new, pcg_workspace, pcg_scalars)
    else:
        _func_dot(pcg_scalars.rz, pcg_workspace, pcg_scalars)


@qd.func(requires_top_level=True)
def _func_dot(out: qd.template(), pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars):
    qd.loop_config(name="pcg_dot_rz", block_dim=PCG_BLOCK)
    for i_r in range(warp_padded(pcg_workspace.n_rows[()])):
        i_b = gs.qd_int(-1)
        value = gs.qd_float(0.0)
        if i_r < pcg_workspace.n_rows[()]:
            j_b = pcg_workspace.row_env[i_r]
            if pcg_scalars.is_converged[j_b] == 0:
                i_b = j_b
                value = pcg_workspace.r[i_r].dot(pcg_workspace.z[i_r])
        add_by_key(value, i_b, out)


@qd.func(requires_top_level=True)
def func_init_linear_pcg_residual(
    b: qd.template(),
    x: qd.template(),
    pcg_workspace: PCGWorkspaceScratch,
    pcg_scalars: PCGScalars,
    is_adjoint: qd.template(),
):
    """Start a solve of A x = b from x = 0, with r = b and Ap = 0. With `is_adjoint`, an environment left out by
    `env_active` keeps the solution of an earlier adjoint solve. The preconditioner and func_init_linear_pcg_direction
    follow."""
    qd.loop_config(name="pcg_init_residual", block_dim=PCG_BLOCK)
    for i_r in range(pcg_workspace.n_rows[()]):
        is_kept = False
        if qd.static(is_adjoint):
            is_kept = pcg_scalars.is_converged[pcg_workspace.row_env[i_r]] != 0
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
    _func_dot_into(pcg_workspace, pcg_scalars, False)


@qd.func(requires_top_level=True)
def func_linear_pcg_spmv(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars, matrix: BCOOMatrixScratch):
    """Add the sparse matrix-vector product (SpMV) of the search direction to Ap, with p.Ap in energy form."""
    qd.loop_config(name="pcg_spmv_pap", block_dim=PCG_BLOCK)
    for i_nz in range(warp_padded(matrix.bcoo_nnz[()])):
        i_r = gs.qd_int(-1)
        i_b = gs.qd_int(-1)
        energy = gs.qd_float(0.0)
        product = qd.Vector.zero(gs.qd_float, 3)
        if i_nz < matrix.bcoo_nnz[()]:
            j_r = matrix.bcoo_row[i_nz]
            j_b = pcg_workspace.row_env[j_r]
            if pcg_scalars.is_converged[j_b] == 0:
                i_r = j_r
                i_b = j_b
                product, energy = _spmv_entry(i_nz, i_r, pcg_workspace, matrix)
        add_by_key(product, i_r, pcg_workspace.ap, n_components=3)
        add_by_key(energy, i_b, pcg_scalars.p_dot_ap)


@qd.func
def _spmv_entry(i_nz: int, i_r: int, pcg_workspace: PCGWorkspaceScratch, matrix: BCOOMatrixScratch):
    i_c = matrix.bcoo_col[i_nz]
    block = func_read_bcoo(i_nz, matrix, block_rows=3, block_cols=3)[2]
    p_col = pcg_workspace.p[i_c]
    product = block @ p_col
    energy = p_col.dot(product)
    if i_r != i_c:
        p_row = pcg_workspace.p[i_r]
        qd.atomic_add(pcg_workspace.ap[i_c], block.transpose() @ p_row)
        energy = 2.0 * p_row.dot(product)
    return product, energy


@qd.func(requires_top_level=True)
def func_linear_pcg_update_xr(x: qd.template(), pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars):
    """Update the solution and the residual of every running environment along its search direction, x += alpha p and
    r -= alpha Ap."""
    qd.loop_config(name="pcg_update_xr", block_dim=PCG_BLOCK)
    for i_r in range(pcg_workspace.n_rows[()]):
        i_b = pcg_workspace.row_env[i_r]
        if pcg_scalars.is_converged[i_b] == 0:
            alpha = pcg_scalars.alpha[i_b]
            x[i_r] += alpha * pcg_workspace.p[i_r]
            pcg_workspace.r[i_r] -= alpha * pcg_workspace.ap[i_r]


@qd.func(requires_top_level=True)
def func_linear_pcg_dot_rz(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars):
    """Add r . z of the preconditioned residual to rz_new."""
    _func_dot_into(pcg_workspace, pcg_scalars, True)


@qd.func(requires_top_level=True)
def func_linear_pcg_direction(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: PCGScalars):
    """Update the search direction of every running environment, p = z + beta p, and zero Ap for the product of the
    next iteration."""
    qd.loop_config(name="pcg_update_p", block_dim=PCG_BLOCK)
    for i_r in range(pcg_workspace.n_rows[()]):
        pcg_workspace.ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
        i_b = pcg_workspace.row_env[i_r]
        if pcg_scalars.is_converged[i_b] == 0:
            pcg_workspace.p[i_r] = pcg_workspace.z[i_r] + pcg_scalars.beta[i_b] * pcg_workspace.p[i_r]


# ============================================ stages of the solve ============================================


@qd.func(requires_top_level=True)
def func_pcg_select_envs(pcg_scalars: PCGScalars, newton_control: NewtonControl, is_adjoint: qd.template()):
    """Take part in the next solve the environments whose Newton solve runs, or those of `is_env_adjoint` for an adjoint
    solve, which follows a converged forward solve."""
    for i_b in range(pcg_scalars.env_active.shape[0]):
        if qd.static(is_adjoint):
            pcg_scalars.env_active[i_b] = pcg_scalars.is_env_adjoint[i_b]
        else:
            pcg_scalars.env_active[i_b] = newton_control.is_env_running[i_b]


@qd.func(requires_top_level=True)
def func_pcg_reset(pcg_scalars: PCGScalars):
    """Reset the scalars of a solve before its owners start their rows, and re-arm the iteration budget."""
    for _ in range(1):
        pcg_scalars.iter_remaining[()] = pcg_scalars.max_iter[()]
    for i_b in range(pcg_scalars.rz.shape[0]):
        pcg_scalars.has_breakdown[i_b] = 0
        pcg_scalars.rz[i_b] = 0.0
        pcg_scalars.rz_new[i_b] = 0.0
        pcg_scalars.p_dot_ap[i_b] = 0.0
        pcg_scalars.is_converged[i_b] = 1 - pcg_scalars.env_active[i_b]


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
    """Take the step alpha = rz / p.Ap of every running environment once every owner added its p.Ap, and zero rz_new.
    With is_curvature_checked set, a non-positive p.Ap along a non-zero direction flags has_breakdown."""
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
        pcg_scalars.iter_remaining[()] -= 1
        pcg_scalars.is_running[()] = 0
    for i_b in range(pcg_scalars.rz.shape[0]):
        if pcg_scalars.is_converged[i_b] == 0:
            rz = pcg_scalars.rz[i_b]
            rz_new = pcg_scalars.rz_new[i_b]
            if qd.static(qd_debug_asserts()):
                assert qd_isfinite(rz_new), "PCG: r . z of environment %d is not finite" % i_b
            if qd.abs(rz_new) <= pcg_scalars.tol_sq[i_b] or not qd_isfinite(rz_new):
                pcg_scalars.is_converged[i_b] = 1
            elif pcg_scalars.iter_remaining[()] > 0:
                qd.atomic_max(pcg_scalars.is_running[()], 1)
            pcg_scalars.beta[i_b] = rz_new / rz if qd.abs(rz) > 0.0 else gs.qd_float(0.0)
            pcg_scalars.rz[i_b] = rz_new
        pcg_scalars.p_dot_ap[i_b] = 0.0
