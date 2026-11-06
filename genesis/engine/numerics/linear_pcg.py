import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field
from genesis.utils.misc import get_gpu_resident_blocks, qd_debug_asserts, qd_isfinite

from .bcoo_matrix import BCOOMatrixScratch, func_read_bcoo
from .fsr_reduce import add_by_key, head_segmented_reduce_add

# Threads of a block of the preconditioned conjugate gradient (PCG) launches
PCG_BLOCK = 256
# Chunks of the reductions on a CPU backend, each one thread's serial pass over a contiguous range
_CPU_CHUNKS = 256
# Registers per thread of the streaming loops and of the sparse matrix-vector product (SpMV) launches
_STREAM_REGISTERS = 32
_SPMV_REGISTERS = 64


@qd.data_oriented
class PCGScalars:
    """Device-resident scalars of one preconditioned conjugate gradient (PCG) solve over the rows of several owners.

    Every environment of the batch runs its own solve over its rows: every owner of rows adds its share of r . z and
    p . Ap into the scalars of the environment of each row. Between the stages of the owners, the solver running the
    iterations takes the step alpha (see func_pcg_step) and the direction factor beta with the convergence decision
    (see func_pcg_decide) of every environment, which every owner then reads. The iterations run while any environment
    has not converged (`is_running`), within one budget for the batch.

    A solve runs `max_iter` iterations at most, and an environment converges once its r . z falls to `tol_rate` times
    its start value. An environment left out by `env_active` (see func_pcg_reset) starts converged. An adjoint solve
    takes the environments of `is_env_adjoint` (see func_pcg_select_envs). With `is_curvature_checked` set, it flags in
    `has_breakdown` the environments whose operator is indefinite, which a solve on a matrix without symmetric positive
    definite (SPD) projection checks for.
    """

    def __init__(self, value_type, max_iter: int, tol_rate: float, n_envs: int = 1) -> None:
        self.value_type = value_type

        def scalar(dtype, value):
            array = qd.ndarray(dtype, shape=())
            array.fill(value)
            return array

        def per_env(dtype, value):
            array = qd.ndarray(dtype, shape=(n_envs,))
            array.fill(value)
            return array

        self.is_running = scalar(gs.qd_int, 0)
        self.max_iter = scalar(gs.qd_int, max_iter)
        self.iter_remaining = scalar(gs.qd_int, 0)
        self.tol_rate = scalar(value_type, tol_rate)
        self.is_curvature_checked = scalar(gs.qd_int, 0)

        self.env_active = per_env(gs.qd_int, 1)
        self.is_env_adjoint = per_env(gs.qd_int, 1)
        # A direction of non-positive curvature (p.Ap <= 0) of an environment flags it, the operator being indefinite
        self.has_breakdown = per_env(gs.qd_int, 0)
        self.tol_sq = per_env(value_type, 0.0)
        self.rz = per_env(value_type, 0.0)
        self.rz_new = per_env(value_type, 0.0)
        self.p_dot_ap = per_env(value_type, 0.0)
        self.alpha = per_env(value_type, 0.0)
        self.beta = per_env(value_type, 0.0)
        self.is_converged = per_env(gs.qd_int, 0)


PCG_ROWS = Size("pcg_rows")


class PCGWorkspaceScratch(Data, kind=DataKind.SCRATCH):
    """Vectors of a preconditioned conjugate gradient (PCG) over the 3x3 block rows of one owner.

    The vectors hold one 3-vector per block row. The owner runs each of its stages over its live block rows `n_rows`, by grid-striding loops on a resident grid, and
    reads and adds into the scalars of the solve (see PCGScalars). `row_env` holds the environment of every block row,
    which the owner writes, its rows sorted by environment.

    The reductions (the dots, the SpMV with p.Ap) have one implementation per backend, both reducing by the environment
    of the rows. On a GPU, they combine the values of each run of one environment in a warp by a segmented shuffle tree
    and add one atomic per run. The CPU backend of Quadrants lacks lanes: it runs a few dozen threads, each of which
    reduces a contiguous chunk of the rows serially and adds once per environment it meets. The elementwise loops are
    the same for both.
    """

    r: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    z: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    p: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    ap: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    row_env: qd.Tensor = field(int, shape=(PCG_ROWS,), ndarray=True)
    n_rows: qd.Tensor = field(int, ndarray=True)


@qd.func(requires_top_level=True)
def _func_dot_into(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template(), is_new: qd.template()):
    if qd.static(gs.backend == gs.cpu):
        if qd.static(is_new):
            _func_dot_into_cpu(pcg_scalars.rz_new, pcg_workspace, pcg_scalars)
        else:
            _func_dot_into_cpu(pcg_scalars.rz, pcg_workspace, pcg_scalars)
    else:
        if qd.static(is_new):
            _func_dot_into_gpu(pcg_scalars.rz_new, pcg_workspace, pcg_scalars)
        else:
            _func_dot_into_gpu(pcg_scalars.rz, pcg_workspace, pcg_scalars)


@qd.func(requires_top_level=True)
def _func_dot_into_cpu(out: qd.template(), pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template()):
    qd.loop_config(name="pcg_dot_rz_cpu")
    for i_chunk in range(_CPU_CHUNKS):
        n_rows = pcg_workspace.n_rows[()]
        size = (n_rows + _CPU_CHUNKS - 1) // _CPU_CHUNKS
        value = gs.qd_float(0.0)
        j_b = gs.qd_int(-1)
        for i_r in range(i_chunk * size, qd.min(i_chunk * size + size, n_rows)):
            i_b = pcg_workspace.row_env[i_r]
            if i_b != j_b:
                if j_b >= 0:
                    qd.atomic_add(out[j_b], value)
                j_b = i_b
                value = gs.qd_float(0.0)
            if pcg_scalars.is_converged[i_b] == 0:
                value += pcg_workspace.r[i_r].dot(pcg_workspace.z[i_r])
        if j_b >= 0:
            qd.atomic_add(out[j_b], value)


@qd.func(requires_top_level=True)
def _func_dot_into_gpu(out: qd.template(), pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template()):
    n_blocks = qd.static(get_gpu_resident_blocks(PCG_BLOCK, _STREAM_REGISTERS))
    qd.loop_config(name="pcg_dot_rz", block_dim=PCG_BLOCK)
    for i_th in range(n_blocks * PCG_BLOCK):
        n_rows = pcg_workspace.n_rows[()]
        padded = ((n_rows + 31) // 32) * 32
        for i_chunk_ in range((padded + n_blocks * PCG_BLOCK - 1) // (n_blocks * PCG_BLOCK)):
            i_r = i_chunk_ * n_blocks * PCG_BLOCK + i_th
            i_b = gs.qd_int(-1)
            value = gs.qd_float(0.0)
            if i_r < n_rows:
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
    pcg_scalars: qd.template(),
    is_adjoint: qd.template(),
):
    """Start a solve of the linear system from a zero solution, with the right-hand side as the residual.

    The solve of A x = b starts from x = 0 with r = b and Ap = 0. With `is_adjoint`, an environment left out by
    `env_active` keeps its solution, which an earlier adjoint solve left. Next, the owner applies its preconditioner
    (z = M^-1 r) and calls func_init_linear_pcg_direction to set the first direction.
    """
    n_blocks = qd.static(get_gpu_resident_blocks(PCG_BLOCK, _STREAM_REGISTERS))
    qd.loop_config(name="pcg_init_residual", block_dim=PCG_BLOCK)
    for i_th in range(n_blocks * PCG_BLOCK):
        for i_chunk_ in range((pcg_workspace.n_rows[()] + n_blocks * PCG_BLOCK - 1) // (n_blocks * PCG_BLOCK)):
            i_r = i_chunk_ * n_blocks * PCG_BLOCK + i_th
            if i_r < pcg_workspace.n_rows[()]:
                is_kept = False
                if qd.static(is_adjoint):
                    is_kept = pcg_scalars.is_converged[pcg_workspace.row_env[i_r]] != 0
                if not is_kept:
                    x[i_r] = qd.Vector.zero(gs.qd_float, 3)
                pcg_workspace.r[i_r] = b[i_r]
                pcg_workspace.ap[i_r] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_init_linear_pcg_direction(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template()):
    """Set the first search direction p = z, and add r . z to rz."""
    n_blocks = qd.static(get_gpu_resident_blocks(PCG_BLOCK, _STREAM_REGISTERS))
    qd.loop_config(name="pcg_copy_z_to_p", block_dim=PCG_BLOCK)
    for i_th in range(n_blocks * PCG_BLOCK):
        for i_chunk_ in range((pcg_workspace.n_rows[()] + n_blocks * PCG_BLOCK - 1) // (n_blocks * PCG_BLOCK)):
            i_r = i_chunk_ * n_blocks * PCG_BLOCK + i_th
            if i_r < pcg_workspace.n_rows[()]:
                pcg_workspace.p[i_r] = pcg_workspace.z[i_r]
    _func_dot_into(pcg_workspace, pcg_scalars, False)


@qd.func(requires_top_level=True)
def func_linear_pcg_spmv(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template(), matrix: BCOOMatrixScratch):
    """Add the sparse matrix-vector product (SpMV) of the search direction to Ap, with p.Ap in energy form.

    The energy form sums w p_r^T B p_c over the stored upper blocks (w = 1 on the diagonal, 2 off it, for the mirrored
    lower block), so every term is local to one entry and needs no separate dot. An owner whose operator has a term
    beyond the matrix adds it to Ap and p.Ap before the step of the solve (see func_pcg_step).

    The products of one row are summed before they reach Ap, which spares an atomic per entry. A GPU sums the runs of a
    row inside a warp by a segmented shuffle reduction. A CPU thread sums them along its chunk of the row-sorted
    entries, which a row leaves only at the chunk ends (see PCGWorkspace). p.Ap adds into the environment of each row,
    and the entries of an environment that converged are skipped.
    """
    if qd.static(gs.backend == gs.cpu):
        _func_linear_pcg_spmv_cpu(pcg_workspace, pcg_scalars, matrix)
    else:
        _func_linear_pcg_spmv_gpu(pcg_workspace, pcg_scalars, matrix)


@qd.func
def _spmv_entry(i_nz: int, i_r: int, pcg_workspace: PCGWorkspaceScratch, matrix: BCOOMatrixScratch):
    """Product B p_c of a stored upper block of row `i_r` for its row, its energy form, and off the diagonal the mirrored
    product B^T p_r added to Ap of its column."""
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
def _func_linear_pcg_spmv_cpu(
    pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template(), matrix: BCOOMatrixScratch
):
    qd.loop_config(name="pcg_spmv_pap_cpu")
    for i_chunk in range(_CPU_CHUNKS):
        nnz = matrix.bcoo_nnz[()]
        size = (nnz + _CPU_CHUNKS - 1) // _CPU_CHUNKS
        energy_sum = gs.qd_float(0.0)
        j_b = gs.qd_int(-1)
        row_sum = qd.Vector.zero(gs.qd_float, 3)
        j_r = gs.qd_int(-1)
        for i_nz in range(i_chunk * size, qd.min(i_chunk * size + size, nnz)):
            i_r = matrix.bcoo_row[i_nz]
            i_b = pcg_workspace.row_env[i_r]
            if i_b != j_b:
                if j_b >= 0:
                    qd.atomic_add(pcg_scalars.p_dot_ap[j_b], energy_sum)
                j_b = i_b
                energy_sum = gs.qd_float(0.0)
            if pcg_scalars.is_converged[i_b] == 0:
                if i_r != j_r:
                    if j_r >= 0:
                        qd.atomic_add(pcg_workspace.ap[j_r], row_sum)
                    j_r = i_r
                    row_sum = qd.Vector.zero(gs.qd_float, 3)
                product, energy = _spmv_entry(i_nz, i_r, pcg_workspace, matrix)
                row_sum += product
                energy_sum += energy
        if j_r >= 0:
            qd.atomic_add(pcg_workspace.ap[j_r], row_sum)
        if j_b >= 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[j_b], energy_sum)


@qd.func(requires_top_level=True)
def _func_linear_pcg_spmv_gpu(
    pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template(), matrix: BCOOMatrixScratch
):
    n_spmv_blocks = qd.static(get_gpu_resident_blocks(PCG_BLOCK, _SPMV_REGISTERS))
    qd.loop_config(name="pcg_spmv_pap", block_dim=PCG_BLOCK)
    for i_th in range(n_spmv_blocks * PCG_BLOCK):
        i_lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
        nnz = matrix.bcoo_nnz[()]
        padded = ((nnz + 31) // 32) * 32
        for i_chunk_ in range((padded + n_spmv_blocks * PCG_BLOCK - 1) // (n_spmv_blocks * PCG_BLOCK)):
            i_nz = i_chunk_ * n_spmv_blocks * PCG_BLOCK + i_th
            if i_nz < padded:
                i_r = gs.qd_int(-1)
                i_b = gs.qd_int(-1)
                energy = gs.qd_float(0.0)
                product = qd.Vector.zero(gs.qd_float, 3)
                if i_nz < nnz:
                    j_r = matrix.bcoo_row[i_nz]
                    j_b = pcg_workspace.row_env[j_r]
                    if pcg_scalars.is_converged[j_b] == 0:
                        i_r = j_r
                        i_b = j_b
                        product, energy = _spmv_entry(i_nz, i_r, pcg_workspace, matrix)
                i_r_prev = qd.simt.subgroup.shuffle(i_r, qd.u32(qd.max(i_lane - 1, 0)))
                i_r_next = qd.simt.subgroup.shuffle(i_r, qd.u32(qd.min(i_lane + 1, 31)))
                is_head = i_lane == 0 or i_r != i_r_prev
                is_tail = i_lane == 31 or i_r != i_r_next
                row_sum = qd.Vector.zero(gs.qd_float, 3)
                for d in qd.static(range(3)):
                    row_sum[d] = head_segmented_reduce_add(product[d], is_tail)
                if i_r >= 0 and is_head:
                    qd.atomic_add(pcg_workspace.ap[i_r], row_sum)
                add_by_key(energy, i_b, pcg_scalars.p_dot_ap)


@qd.func(requires_top_level=True)
def func_linear_pcg_update_xr(x: qd.template(), pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template()):
    """Update the solution and the residual of every running environment along its search direction, x += alpha p and
    r -= alpha Ap."""
    n_blocks = qd.static(get_gpu_resident_blocks(PCG_BLOCK, _STREAM_REGISTERS))
    qd.loop_config(name="pcg_update_xr", block_dim=PCG_BLOCK)
    for i_th in range(n_blocks * PCG_BLOCK):
        for i_chunk_ in range((pcg_workspace.n_rows[()] + n_blocks * PCG_BLOCK - 1) // (n_blocks * PCG_BLOCK)):
            i_r = i_chunk_ * n_blocks * PCG_BLOCK + i_th
            if i_r < pcg_workspace.n_rows[()]:
                i_b = pcg_workspace.row_env[i_r]
                if pcg_scalars.is_converged[i_b] == 0:
                    alpha = pcg_scalars.alpha[i_b]
                    x[i_r] += alpha * pcg_workspace.p[i_r]
                    pcg_workspace.r[i_r] -= alpha * pcg_workspace.ap[i_r]


@qd.func(requires_top_level=True)
def func_linear_pcg_dot_rz(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template()):
    """Add r . z of the preconditioned residual to rz_new."""
    _func_dot_into(pcg_workspace, pcg_scalars, True)


@qd.func(requires_top_level=True)
def func_linear_pcg_direction(pcg_workspace: PCGWorkspaceScratch, pcg_scalars: qd.template()):
    """Update the search direction of every running environment, p = z + beta p, and zero Ap for the product of the
    next iteration."""
    n_blocks = qd.static(get_gpu_resident_blocks(PCG_BLOCK, _STREAM_REGISTERS))
    qd.loop_config(name="pcg_update_p", block_dim=PCG_BLOCK)
    for i_th in range(n_blocks * PCG_BLOCK):
        for i_chunk_ in range((pcg_workspace.n_rows[()] + n_blocks * PCG_BLOCK - 1) // (n_blocks * PCG_BLOCK)):
            i_r = i_chunk_ * n_blocks * PCG_BLOCK + i_th
            if i_r < pcg_workspace.n_rows[()]:
                pcg_workspace.ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
                i_b = pcg_workspace.row_env[i_r]
                if pcg_scalars.is_converged[i_b] == 0:
                    pcg_workspace.p[i_r] = pcg_workspace.z[i_r] + pcg_scalars.beta[i_b] * pcg_workspace.p[i_r]


# ============================================ stages of the solve ============================================


@qd.func(requires_top_level=True)
def func_pcg_select_envs(pcg_scalars: qd.template(), newton_control: qd.template(), is_adjoint: qd.template()):
    """Take part in the next solve the environments whose Newton solve runs, or those of `is_env_adjoint` for an adjoint
    solve, which follows a converged forward solve."""
    for i_b in range(pcg_scalars.env_active.shape[0]):
        if qd.static(is_adjoint):
            pcg_scalars.env_active[i_b] = pcg_scalars.is_env_adjoint[i_b]
        else:
            pcg_scalars.env_active[i_b] = newton_control.is_env_running[i_b]


@qd.func(requires_top_level=True)
def func_pcg_reset(pcg_scalars: qd.template()):
    """Reset the scalars of a solve before its owners start their rows, and re-arm the iteration budget.

    An environment outside `env_active` starts converged, so that the owners leave its rows as they are."""
    for _ in range(1):
        pcg_scalars.iter_remaining[()] = pcg_scalars.max_iter[()]
    for i_b in range(pcg_scalars.rz.shape[0]):
        pcg_scalars.has_breakdown[i_b] = 0
        pcg_scalars.rz[i_b] = 0.0
        pcg_scalars.rz_new[i_b] = 0.0
        pcg_scalars.p_dot_ap[i_b] = 0.0
        pcg_scalars.is_converged[i_b] = 1 - pcg_scalars.env_active[i_b]


@qd.func(requires_top_level=True)
def func_pcg_arm(pcg_scalars: qd.template()):
    """Set the convergence threshold tol_rate |rz0| of every environment once every owner set its first direction, and
    arm the iterations if any environment takes part."""
    for _ in range(1):
        pcg_scalars.is_running[()] = 0
    for i_b in range(pcg_scalars.rz.shape[0]):
        pcg_scalars.tol_sq[i_b] = pcg_scalars.tol_rate[()] * qd.abs(pcg_scalars.rz[i_b])
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_max(pcg_scalars.is_running[()], 1)


@qd.func(requires_top_level=True)
def func_pcg_step(pcg_scalars: qd.template()):
    """Take the step alpha = rz / p.Ap of every running environment once every owner added its p.Ap, and zero rz_new
    for the dots that follow.

    With is_curvature_checked set, a non-positive p.Ap along a non-zero direction flags has_breakdown of its environment.
    """
    for i_b in range(pcg_scalars.rz.shape[0]):
        if pcg_scalars.is_converged[i_b] == 0:
            rz = pcg_scalars.rz[i_b]
            p_dot_ap = pcg_scalars.p_dot_ap[i_b]
            if pcg_scalars.is_curvature_checked[()] != 0 and p_dot_ap <= 0.0 and qd.abs(rz) > 0.0:
                pcg_scalars.has_breakdown[i_b] = 1
            pcg_scalars.alpha[i_b] = rz / p_dot_ap if qd.abs(p_dot_ap) > 0.0 else pcg_scalars.value_type(0.0)
        pcg_scalars.rz_new[i_b] = 0.0


@qd.func(requires_top_level=True)
def func_pcg_decide(pcg_scalars: qd.template()):
    """Take beta = rz_new / rz and the convergence decision of every running environment once every owner added its
    r . z, then roll rz over.

    An environment converges once its |rz_new| falls to its threshold, and the iterations stop once every environment
    converged or at the end of the budget. An rz_new that is not finite fails the assertion in debug mode and otherwise
    stops the environment unconverged, its Newton solve judging the non-finite solution. p.Ap is zeroed for the product
    of the next iteration.
    """
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
            pcg_scalars.beta[i_b] = rz_new / rz if qd.abs(rz) > 0.0 else pcg_scalars.value_type(0.0)
            pcg_scalars.rz[i_b] = rz_new
        pcg_scalars.p_dot_ap[i_b] = 0.0
