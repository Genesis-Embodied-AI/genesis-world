import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_debug_asserts, qd_isfinite

from .fsr_reduce import head_segmented_reduce_add
from .gpu_occupancy import cuda_resident_blocks

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

    Every owner of rows adds its share of r . z and p . Ap into them. Between the stages of the owners, the solver
    running the iterations takes the step alpha (see func_pcg_step) and the direction factor beta with the convergence
    decision (see func_pcg_decide), which every owner then reads.

    A solve runs `max_iter` iterations at most, and converges once r . z falls to `tol_rate` times its start value.
    """

    def __init__(self, value_type, max_iter: int, tol_rate: float) -> None:
        self.value_type = value_type
        self.is_running = qd.ndarray(gs.qd_int, shape=())
        self.is_running.fill(0)
        self.max_iter = qd.ndarray(gs.qd_int, shape=())
        self.max_iter.fill(max_iter)
        self.iter_remaining = qd.ndarray(gs.qd_int, shape=())
        self.iter_remaining.fill(0)
        self.tol_rate = qd.ndarray(value_type, shape=())
        self.tol_rate.fill(tol_rate)
        self.tol_sq = qd.ndarray(value_type, shape=())
        self.tol_sq.fill(0.0)
        self.rz = qd.ndarray(value_type, shape=())
        self.rz.fill(0.0)
        self.rz_new = qd.ndarray(value_type, shape=())
        self.rz_new.fill(0.0)
        self.p_dot_ap = qd.ndarray(value_type, shape=())
        self.p_dot_ap.fill(0.0)
        self.alpha = qd.ndarray(value_type, shape=())
        self.alpha.fill(0.0)
        self.beta = qd.ndarray(value_type, shape=())
        self.beta.fill(0.0)
        self.is_converged = qd.ndarray(gs.qd_int, shape=())
        self.is_converged.fill(0)
        # With is_curvature_checked set, a direction of non-positive curvature (p.Ap <= 0) sets has_breakdown: the
        # operator is then indefinite, which a solve on a matrix without symmetric positive definite (SPD) projection
        # checks for
        self.is_curvature_checked = qd.ndarray(gs.qd_int, shape=())
        self.is_curvature_checked.fill(0)
        self.has_breakdown = qd.ndarray(gs.qd_int, shape=())
        self.has_breakdown.fill(0)


@qd.data_oriented
class PCGWorkspace:
    """Vectors of a preconditioned conjugate gradient (PCG) over the 3x3 block rows of one owner.

    The owner runs each of its stages over its live block rows `n_rows`, by grid-striding loops on a resident grid, and
    reads and adds into the scalars of the solve (see PCGScalars).

    The reductions (the dots, the SpMV with p.Ap) have one implementation per backend. On a GPU, they combine the values
    of a warp by shuffles and add one atomic per warp. The CPU backend of Quadrants lacks lanes: it runs a few dozen
    threads, each of which reduces a contiguous chunk of the rows serially. The elementwise loops are the same for both.
    """

    def __init__(self, max_rows: int, value_type) -> None:
        self.value_type = value_type
        self.n_blocks = cuda_resident_blocks(PCG_BLOCK, 0, _STREAM_REGISTERS)
        self.n_spmv_blocks = cuda_resident_blocks(PCG_BLOCK, 0, _SPMV_REGISTERS)
        self.allocate(max_rows)
        self.n_rows = qd.ndarray(gs.qd_int, shape=())
        self.n_rows.fill(0)

    def allocate(self, max_rows: int) -> None:
        """Allocate the vectors for a capacity of block rows, replacing the current ones and keeping this object."""
        length = max(3 * max(max_rows, 1), 1)
        self.r = qd.ndarray(self.value_type, shape=(length,))
        self.z = qd.ndarray(self.value_type, shape=(length,))
        self.p = qd.ndarray(self.value_type, shape=(length,))
        self.ap = qd.ndarray(self.value_type, shape=(length,))


@qd.func
def _warp_sum(value):
    for level in qd.static(range(5)):
        value = value + qd.simt.subgroup.shuffle_xor(value, qd.u32(16 >> level))
    return value


@qd.func(requires_top_level=True)
def _func_dot_into(
    a: qd.template(), b: qd.template(), pcg_workspace: qd.template(), pcg_scalars: qd.template(), is_new: qd.template()
):
    """Add the dot product of two vectors over the live rows to rz, or to rz_new with `is_new`.

    As in every reduction of the rows, a GPU sums the products of a warp by shuffles and adds one atomic per warp, and a
    CPU sums each of _CPU_CHUNKS contiguous chunks in one thread and adds one atomic per chunk.
    """
    if qd.static(gs.backend == gs.cpu):
        _func_dot_into_cpu(a, b, pcg_workspace, pcg_scalars, is_new)
    else:
        _func_dot_into_gpu(a, b, pcg_workspace, pcg_scalars, is_new)


@qd.func(requires_top_level=True)
def _func_dot_into_cpu(
    a: qd.template(), b: qd.template(), pcg_workspace: qd.template(), pcg_scalars: qd.template(), is_new: qd.template()
):
    qd.loop_config(name="pcg_dot_rz_cpu")
    for chunk in range(_CPU_CHUNKS):
        n_rows = pcg_workspace.n_rows[()]
        size = (n_rows + _CPU_CHUNKS - 1) // _CPU_CHUNKS
        value = pcg_workspace.value_type(0.0)
        if pcg_scalars.is_converged[()] == 0:
            for v in range(chunk * size, qd.min(chunk * size + size, n_rows)):
                for d in qd.static(range(3)):
                    value += a[3 * v + d] * b[3 * v + d]
        if value != 0.0:
            if qd.static(is_new):
                qd.atomic_add(pcg_scalars.rz_new[()], value)
            else:
                qd.atomic_add(pcg_scalars.rz[()], value)


@qd.func(requires_top_level=True)
def _func_dot_into_gpu(
    a: qd.template(), b: qd.template(), pcg_workspace: qd.template(), pcg_scalars: qd.template(), is_new: qd.template()
):
    qd.loop_config(name="pcg_dot_rz", block_dim=PCG_BLOCK)
    for t in range(pcg_workspace.n_blocks * PCG_BLOCK):
        lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
        n_rows = pcg_workspace.n_rows[()]
        padded = ((n_rows + 31) // 32) * 32
        value = pcg_workspace.value_type(0.0)
        for i_chunk_ in range(
            (padded + pcg_workspace.n_blocks * PCG_BLOCK - 1) // (pcg_workspace.n_blocks * PCG_BLOCK)
        ):
            v = i_chunk_ * pcg_workspace.n_blocks * PCG_BLOCK + t
            if v < n_rows and pcg_scalars.is_converged[()] == 0:
                for d in qd.static(range(3)):
                    value += a[3 * v + d] * b[3 * v + d]
        value = _warp_sum(value)
        if lane == 0 and value != 0.0:
            if qd.static(is_new):
                qd.atomic_add(pcg_scalars.rz_new[()], value)
            else:
                qd.atomic_add(pcg_scalars.rz[()], value)


@qd.func(requires_top_level=True)
def func_init_linear_pcg_residual(b: qd.template(), x: qd.template(), pcg_workspace: qd.template()):
    """Start a solve of the linear system from a zero solution, with the right-hand side as the residual.

    The solve of A x = b starts from x = 0 with r = b and Ap = 0. Next, the owner applies its preconditioner
    (z = M^-1 r) and calls func_init_linear_pcg_direction to set the first direction.
    """
    qd.loop_config(name="pcg_init_residual", block_dim=PCG_BLOCK)
    for t in range(pcg_workspace.n_blocks * PCG_BLOCK):
        for i_chunk_ in range(
            (3 * pcg_workspace.n_rows[()] + pcg_workspace.n_blocks * PCG_BLOCK - 1)
            // (pcg_workspace.n_blocks * PCG_BLOCK)
        ):
            i = i_chunk_ * pcg_workspace.n_blocks * PCG_BLOCK + t
            if i < 3 * pcg_workspace.n_rows[()]:
                x[i] = 0.0
                pcg_workspace.r[i] = b[i]
                pcg_workspace.ap[i] = 0.0


@qd.func(requires_top_level=True)
def func_init_linear_pcg_direction(pcg_workspace: qd.template(), pcg_scalars: qd.template()):
    """Set the first search direction p = z, and add r . z to rz."""
    qd.loop_config(name="pcg_copy_z_to_p", block_dim=PCG_BLOCK)
    for t in range(pcg_workspace.n_blocks * PCG_BLOCK):
        for i_chunk_ in range(
            (3 * pcg_workspace.n_rows[()] + pcg_workspace.n_blocks * PCG_BLOCK - 1)
            // (pcg_workspace.n_blocks * PCG_BLOCK)
        ):
            i = i_chunk_ * pcg_workspace.n_blocks * PCG_BLOCK + t
            if i < 3 * pcg_workspace.n_rows[()]:
                pcg_workspace.p[i] = pcg_workspace.z[i]
    _func_dot_into(pcg_workspace.r, pcg_workspace.z, pcg_workspace, pcg_scalars, False)


@qd.func(requires_top_level=True)
def func_linear_pcg_spmv(pcg_workspace: qd.template(), pcg_scalars: qd.template(), matrix: qd.template()):
    """Add the sparse matrix-vector product (SpMV) of the search direction to Ap, with p.Ap in energy form.

    The energy form sums w p_r^T B p_c over the stored upper blocks (w = 1 on the diagonal, 2 off it, for the mirrored
    lower block), so every term is local to one entry and needs no separate dot. An owner whose operator has a term
    beyond the matrix adds it to Ap and p.Ap before the step of the solve (see func_pcg_step).

    The products of one row are summed before they reach Ap, which spares an atomic per entry. A GPU sums the runs of a
    row inside a warp by a segmented shuffle reduction. A CPU thread sums them along its chunk of the row-sorted
    entries, which a row leaves only at the chunk ends (see PCGWorkspace).
    """
    if qd.static(gs.backend == gs.cpu):
        _func_linear_pcg_spmv_cpu(pcg_workspace, pcg_scalars, matrix)
    else:
        _func_linear_pcg_spmv_gpu(pcg_workspace, pcg_scalars, matrix)


@qd.func(requires_top_level=True)
def _func_linear_pcg_spmv_cpu(pcg_workspace: qd.template(), pcg_scalars: qd.template(), matrix: qd.template()):
    """Run func_linear_pcg_spmv over contiguous chunks of the row-sorted entries, one thread per chunk.

    The products of a row accumulate in the thread and add once to Ap where the row ends, the mirrored products of the
    lower triangle add per entry, and p.Ap adds once per chunk.
    """
    qd.loop_config(name="pcg_spmv_pap_cpu")
    for chunk in range(_CPU_CHUNKS):
        if pcg_scalars.is_converged[()] == 0:
            nnz = matrix.bcoo_nnz[()]
            size = (nnz + _CPU_CHUNKS - 1) // _CPU_CHUNKS
            energy_sum = pcg_workspace.value_type(0.0)
            row_sum = qd.Vector.zero(pcg_workspace.value_type, 3)
            current_row = gs.qd_int(-1)
            for entry in range(chunk * size, qd.min(chunk * size + size, nnz)):
                row, col, block = matrix.read_bcoo(entry)
                if row != current_row:
                    if current_row >= 0:
                        for d in qd.static(range(3)):
                            qd.atomic_add(pcg_workspace.ap[3 * current_row + d], row_sum[d])
                    current_row = row
                    row_sum = qd.Vector.zero(pcg_workspace.value_type, 3)
                p_col = qd.Vector(
                    [pcg_workspace.p[3 * col + d] for d in qd.static(range(3))], dt=pcg_workspace.value_type
                )
                product = block @ p_col
                row_sum += product
                if row != col:
                    p_row = qd.Vector(
                        [pcg_workspace.p[3 * row + d] for d in qd.static(range(3))], dt=pcg_workspace.value_type
                    )
                    product_mirrored = block.transpose() @ p_row
                    for d in qd.static(range(3)):
                        qd.atomic_add(pcg_workspace.ap[3 * col + d], product_mirrored[d])
                    energy_sum += 2.0 * p_row.dot(product)
                else:
                    energy_sum += p_col.dot(product)
            if current_row >= 0:
                for d in qd.static(range(3)):
                    qd.atomic_add(pcg_workspace.ap[3 * current_row + d], row_sum[d])
            if energy_sum != 0.0:
                qd.atomic_add(pcg_scalars.p_dot_ap[()], energy_sum)


@qd.func(requires_top_level=True)
def _func_linear_pcg_spmv_gpu(pcg_workspace: qd.template(), pcg_scalars: qd.template(), matrix: qd.template()):
    qd.loop_config(name="pcg_spmv_pap", block_dim=PCG_BLOCK)
    for t in range(pcg_workspace.n_spmv_blocks * PCG_BLOCK):
        lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
        nnz = matrix.bcoo_nnz[()]
        padded = ((nnz + 31) // 32) * 32
        energy_sum = pcg_workspace.value_type(0.0)
        for i_chunk_ in range(
            (padded + pcg_workspace.n_spmv_blocks * PCG_BLOCK - 1) // (pcg_workspace.n_spmv_blocks * PCG_BLOCK)
        ):
            entry = i_chunk_ * pcg_workspace.n_spmv_blocks * PCG_BLOCK + t
            if entry < padded:
                row = gs.qd_int(-1)
                product = qd.Vector.zero(pcg_workspace.value_type, 3)
                if entry < nnz and pcg_scalars.is_converged[()] == 0:
                    row_, col, block = matrix.read_bcoo(entry)
                    row = row_
                    p_col = qd.Vector(
                        [pcg_workspace.p[3 * col + d] for d in qd.static(range(3))], dt=pcg_workspace.value_type
                    )
                    product = block @ p_col
                    if row != col:
                        p_row = qd.Vector(
                            [pcg_workspace.p[3 * row + d] for d in qd.static(range(3))], dt=pcg_workspace.value_type
                        )
                        product_mirrored = block.transpose() @ p_row
                        for d in qd.static(range(3)):
                            qd.atomic_add(pcg_workspace.ap[3 * col + d], product_mirrored[d])
                        energy_sum += 2.0 * p_row.dot(product)
                    else:
                        energy_sum += p_col.dot(product)
                previous_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.max(lane - 1, 0)))
                next_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.min(lane + 1, 31)))
                is_head = lane == 0 or row != previous_row
                is_tail = lane == 31 or row != next_row
                row_sum = qd.Vector.zero(pcg_workspace.value_type, 3)
                for d in qd.static(range(3)):
                    row_sum[d] = head_segmented_reduce_add(product[d], is_tail)
                if row >= 0 and is_head:
                    for d in qd.static(range(3)):
                        if row_sum[d] != 0.0:
                            qd.atomic_add(pcg_workspace.ap[3 * row + d], row_sum[d])
        energy_sum = _warp_sum(energy_sum)
        if lane == 0 and energy_sum != 0.0:
            qd.atomic_add(pcg_scalars.p_dot_ap[()], energy_sum)


@qd.func(requires_top_level=True)
def func_linear_pcg_update_xr(x: qd.template(), pcg_workspace: qd.template(), pcg_scalars: qd.template()):
    """Update the solution and the residual along the search direction, x += alpha p and r -= alpha Ap."""
    qd.loop_config(name="pcg_update_xr", block_dim=PCG_BLOCK)
    for t in range(pcg_workspace.n_blocks * PCG_BLOCK):
        if pcg_scalars.is_converged[()] == 0:
            alpha = pcg_scalars.alpha[()]
            for i_chunk_ in range(
                (3 * pcg_workspace.n_rows[()] + pcg_workspace.n_blocks * PCG_BLOCK - 1)
                // (pcg_workspace.n_blocks * PCG_BLOCK)
            ):
                i = i_chunk_ * pcg_workspace.n_blocks * PCG_BLOCK + t
                if i < 3 * pcg_workspace.n_rows[()]:
                    x[i] += alpha * pcg_workspace.p[i]
                    pcg_workspace.r[i] -= alpha * pcg_workspace.ap[i]


@qd.func(requires_top_level=True)
def func_linear_pcg_dot_rz(pcg_workspace: qd.template(), pcg_scalars: qd.template()):
    """Add r . z of the preconditioned residual to rz_new."""
    _func_dot_into(pcg_workspace.r, pcg_workspace.z, pcg_workspace, pcg_scalars, True)


@qd.func(requires_top_level=True)
def func_linear_pcg_direction(pcg_workspace: qd.template(), pcg_scalars: qd.template()):
    """Update the search direction p = z + beta p, and zero Ap for the product of the next iteration."""
    qd.loop_config(name="pcg_update_p", block_dim=PCG_BLOCK)
    for t in range(pcg_workspace.n_blocks * PCG_BLOCK):
        is_converged = pcg_scalars.is_converged[()] != 0
        beta = pcg_scalars.beta[()]
        for i_chunk_ in range(
            (3 * pcg_workspace.n_rows[()] + pcg_workspace.n_blocks * PCG_BLOCK - 1)
            // (pcg_workspace.n_blocks * PCG_BLOCK)
        ):
            i = i_chunk_ * pcg_workspace.n_blocks * PCG_BLOCK + t
            if i < 3 * pcg_workspace.n_rows[()]:
                pcg_workspace.ap[i] = 0.0
                if not is_converged:
                    pcg_workspace.p[i] = pcg_workspace.z[i] + beta * pcg_workspace.p[i]


@qd.func(requires_top_level=True)
def func_apply_block_jacobi(precond_inv_diag: qd.template(), pcg_workspace: qd.template()):
    """Apply the block Jacobi preconditioner to the residual.

    The result is z = M^-1 r, with M the 3x3 block diagonal whose inverse blocks the caller fills in precond_inv_diag.
    """
    qd.loop_config(name="pcg_apply_block_jacobi")
    for row in range(pcg_workspace.n_rows[()]):
        for i in qd.static(range(3)):
            value = pcg_workspace.value_type(0.0)
            for j in qd.static(range(3)):
                value += precond_inv_diag[row * 9 + i * 3 + j] * pcg_workspace.r[row * 3 + j]
            pcg_workspace.z[row * 3 + i] = value


# ============================================ stages of the solve ============================================


@qd.func(requires_top_level=True)
def func_pcg_reset(pcg_scalars: qd.template()):
    """Reset the scalars of a solve before its owners start their rows, and re-arm the iteration budget."""
    for _ in range(1):
        pcg_scalars.rz[()] = 0.0
        pcg_scalars.rz_new[()] = 0.0
        pcg_scalars.p_dot_ap[()] = 0.0
        pcg_scalars.is_converged[()] = 0
        pcg_scalars.has_breakdown[()] = 0
        pcg_scalars.iter_remaining[()] = pcg_scalars.max_iter[()]


@qd.func(requires_top_level=True)
def func_pcg_arm(pcg_scalars: qd.template()):
    """Set the convergence threshold tol_rate |rz0| once every owner set its first direction, and arm the iterations."""
    for _ in range(1):
        pcg_scalars.tol_sq[()] = pcg_scalars.tol_rate[()] * qd.abs(pcg_scalars.rz[()])
        pcg_scalars.is_running[()] = 1


@qd.func(requires_top_level=True)
def func_pcg_step(pcg_scalars: qd.template()):
    """Take the step alpha = rz / p.Ap once every owner added its p.Ap, and zero rz_new for the dots that follow.

    With is_curvature_checked set, a non-positive p.Ap along a non-zero direction sets has_breakdown.
    """
    for _ in range(1):
        if pcg_scalars.is_converged[()] == 0:
            rz = pcg_scalars.rz[()]
            p_dot_ap = pcg_scalars.p_dot_ap[()]
            if pcg_scalars.is_curvature_checked[()] != 0 and p_dot_ap <= 0.0 and rz != 0.0:
                pcg_scalars.has_breakdown[()] = 1
            pcg_scalars.alpha[()] = rz / p_dot_ap if p_dot_ap != 0.0 else pcg_scalars.value_type(0.0)
        pcg_scalars.rz_new[()] = 0.0


@qd.func(requires_top_level=True)
def func_pcg_decide(pcg_scalars: qd.template()):
    """Take beta = rz_new / rz and the convergence decision once every owner added its r . z, then roll rz over.

    The solve converges once |rz_new| falls to the threshold, and stops there or at the end of its iteration budget. An
    rz_new that is not finite fails the assertion in debug mode and otherwise ends the iterations unconverged, the
    caller judging the non-finite solution. p.Ap is zeroed for the product of the next iteration.
    """
    for _ in range(1):
        rz = pcg_scalars.rz[()]
        rz_new = pcg_scalars.rz_new[()]
        remaining = pcg_scalars.iter_remaining[()] - 1
        is_not_done = 0
        if pcg_scalars.is_converged[()] == 0 and remaining > 0:
            if qd.static(qd_debug_asserts()):
                assert qd_isfinite(rz_new), "PCG: r . z is not finite"
            if qd.abs(rz_new) <= pcg_scalars.tol_sq[()]:
                pcg_scalars.is_converged[()] = 1
            elif qd_isfinite(rz_new):
                is_not_done = 1
        pcg_scalars.beta[()] = rz_new / rz if rz != 0.0 else pcg_scalars.value_type(0.0)
        pcg_scalars.rz[()] = rz_new
        pcg_scalars.p_dot_ap[()] = 0.0
        pcg_scalars.iter_remaining[()] = remaining
        if remaining <= 0 or is_not_done == 0:
            pcg_scalars.is_running[()] = 0
