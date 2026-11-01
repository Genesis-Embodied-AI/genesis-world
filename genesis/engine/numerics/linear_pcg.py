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
class PCGWorkspace:
    """Vectors and device-resident scalars of a preconditioned conjugate gradient (PCG) over 3x3 block rows.

    An iteration takes five launches: the sparse matrix-vector product (SpMV) with p.Ap fused in energy form, the x/r
    update with alpha inlined, the preconditioner of the caller, the r.z dot, and a fused tail that updates p, zeroes
    the Ap and p.Ap of the next iteration, and lets the last block of the grid decide convergence and flip the parity of
    the rz/rz_new slots. Every loop grid-strides over the live block rows `n_rows` with a resident grid, whose block
    count the ticket of the tail needs.

    The reductions (the dots, the SpMV with p.Ap, the convergence decision of the tail) have one implementation per
    backend. On a GPU, they combine the values of a warp by shuffles and those of the grid by a ticket that the last
    block takes. The CPU backend of Quadrants lacks both: it runs a few dozen threads with no lanes, each of which
    reduces a contiguous chunk of the rows serially, and the end of a parallel loop orders what follows it. The
    elementwise loops (the copies and the x/r update) are the same for both.
    """

    def __init__(self, max_rows: int, value_type) -> None:
        self.value_type = value_type
        self.n_blocks = cuda_resident_blocks(PCG_BLOCK, 0, _STREAM_REGISTERS)
        self.n_spmv_blocks = cuda_resident_blocks(PCG_BLOCK, 0, _SPMV_REGISTERS)
        self.allocate(max_rows)

        self.n_rows = qd.ndarray(qd.i32, shape=())
        self.n_rows.fill(0)
        self.cond = qd.ndarray(qd.i32, shape=())
        self.cond.fill(0)
        self.max_iter = qd.ndarray(qd.i32, shape=())
        self.max_iter.fill(1024)
        self.iter_remaining = qd.ndarray(qd.i32, shape=())
        self.iter_remaining.fill(0)
        self.tol_rate = qd.ndarray(value_type, shape=())
        self.tol_rate.fill(1e-4)
        self.tol_sq = qd.ndarray(value_type, shape=())
        self.tol_sq.fill(0.0)
        # rz and rz_new are the two slots of one buffer, selected by the parity the tail flips every iteration
        self.rz_slots = qd.ndarray(value_type, shape=(2,))
        self.parity = qd.ndarray(qd.i32, shape=())
        self.parity.fill(0)
        self.p_dot_ap = qd.ndarray(value_type, shape=())
        self.p_dot_ap.fill(0.0)
        self.converged = qd.ndarray(qd.i32, shape=())
        self.converged.fill(0)
        # With check_curvature set, a direction of non-positive curvature (p.Ap <= 0) raises breakdown: the operator is
        # then indefinite, which a solve on a matrix without symmetric positive definite (SPD) projection checks for
        self.check_curvature = qd.ndarray(qd.i32, shape=())
        self.check_curvature.fill(0)
        self.breakdown = qd.ndarray(qd.i32, shape=())
        self.breakdown.fill(0)
        self.not_done = qd.ndarray(qd.i32, shape=())
        self.not_done.fill(0)
        self.ticket = qd.ndarray(qd.i32, shape=())
        self.ticket.fill(0)

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
def _func_dot_into_slot(
    a: qd.template(),
    b: qd.template(),
    ws: qd.template(),
    slot_offset: qd.template(),
):
    """Add the dot product of two vectors over the live rows to one of the two rz slots, zeroed by an earlier launch.

    The dot a . b lands in rz_slots[parity XOR slot_offset]. As in every reduction of PCGWorkspace, a GPU sums the
    products of a warp by shuffles and adds one atomic per warp, and a CPU sums each of _CPU_CHUNKS contiguous chunks in
    one thread and adds one atomic per chunk.
    """
    if qd.static(gs.backend == gs.cpu):
        _func_dot_into_slot_cpu(a, b, ws, slot_offset)
    else:
        _func_dot_into_slot_gpu(a, b, ws, slot_offset)


@qd.func(requires_top_level=True)
def _func_dot_into_slot_cpu(
    a: qd.template(),
    b: qd.template(),
    ws: qd.template(),
    slot_offset: qd.template(),
):
    qd.loop_config(name="pcg_dot_rz_cpu")
    for chunk in range(_CPU_CHUNKS):
        n_rows = ws.n_rows[()]
        size = (n_rows + _CPU_CHUNKS - 1) // _CPU_CHUNKS
        value = ws.value_type(0.0)
        if ws.converged[()] == 0:
            for v in range(chunk * size, qd.min(chunk * size + size, n_rows)):
                for d in qd.static(range(3)):
                    value += a[3 * v + d] * b[3 * v + d]
        if value != 0.0:
            slot = ws.parity[()]
            if qd.static(slot_offset != 0):
                slot = 1 - slot
            qd.atomic_add(ws.rz_slots[slot], value)


@qd.func(requires_top_level=True)
def _func_dot_into_slot_gpu(
    a: qd.template(),
    b: qd.template(),
    ws: qd.template(),
    slot_offset: qd.template(),
):
    qd.loop_config(name="pcg_dot_rz", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
        n_rows = ws.n_rows[()]
        padded = ((n_rows + 31) // 32) * 32
        value = ws.value_type(0.0)
        for i_chunk_ in range((padded + ws.n_blocks * PCG_BLOCK - 1) // (ws.n_blocks * PCG_BLOCK)):
            v = i_chunk_ * ws.n_blocks * PCG_BLOCK + t
            if v < n_rows and ws.converged[()] == 0:
                for d in qd.static(range(3)):
                    value += a[3 * v + d] * b[3 * v + d]
        value = _warp_sum(value)
        if lane == 0 and value != 0.0:
            slot = ws.parity[()]
            if qd.static(slot_offset != 0):
                slot = 1 - slot
            qd.atomic_add(ws.rz_slots[slot], value)


@qd.func(requires_top_level=True)
def func_init_linear_pcg_residual(
    b: qd.template(),
    x: qd.template(),
    ws: qd.template(),
):
    """Start a solve of the linear system from a zero solution, with the right-hand side as the residual.

    The solve of A x = b starts from x = 0 with r = b and Ap = p.Ap = 0, resets the convergence flag, the ticket and the
    parity, and re-arms the iteration budget. Next, the caller applies its preconditioner (z = M^-1 r) and calls
    func_init_linear_pcg_direction to set the first direction.
    """
    qd.loop_config(name="pcg_memset_x", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        for i_chunk_ in range((3 * ws.n_rows[()] + ws.n_blocks * PCG_BLOCK - 1) // (ws.n_blocks * PCG_BLOCK)):
            i = i_chunk_ * ws.n_blocks * PCG_BLOCK + t
            if i < 3 * ws.n_rows[()]:
                x[i] = 0.0
    qd.loop_config(name="pcg_copy_b_to_r", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        for i_chunk_ in range((3 * ws.n_rows[()] + ws.n_blocks * PCG_BLOCK - 1) // (ws.n_blocks * PCG_BLOCK)):
            i = i_chunk_ * ws.n_blocks * PCG_BLOCK + t
            if i < 3 * ws.n_rows[()]:
                ws.r[i] = b[i]
    qd.loop_config(name="pcg_memset_converged", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        for i_chunk_ in range((3 * ws.n_rows[()] + ws.n_blocks * PCG_BLOCK - 1) // (ws.n_blocks * PCG_BLOCK)):
            i = i_chunk_ * ws.n_blocks * PCG_BLOCK + t
            if i < 3 * ws.n_rows[()]:
                ws.ap[i] = 0.0
        if t == 0:
            ws.converged[()] = 0
            ws.breakdown[()] = 0
            ws.p_dot_ap[()] = 0.0
            ws.not_done[()] = 0
            ws.ticket[()] = 0
            ws.parity[()] = 0
            ws.iter_remaining[()] = ws.max_iter[()]


@qd.func(requires_top_level=True)
def func_init_linear_pcg_direction(
    ws: qd.template(),
):
    """Set the first search direction and the convergence threshold, and arm the iteration loop.

    The direction is p = z, with rz = r . z, and the threshold is tol_rate |rz0|.
    """
    qd.loop_config(name="pcg_copy_z_to_p", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        for i_chunk_ in range((3 * ws.n_rows[()] + ws.n_blocks * PCG_BLOCK - 1) // (ws.n_blocks * PCG_BLOCK)):
            i = i_chunk_ * ws.n_blocks * PCG_BLOCK + t
            if i < 3 * ws.n_rows[()]:
                ws.p[i] = ws.z[i]
        if t == 0:
            parity = ws.parity[()]
            ws.rz_slots[parity] = 0.0
    _func_dot_into_slot(ws.r, ws.z, ws, 0)
    for _ in range(1):
        parity = ws.parity[()]
        ws.tol_sq[()] = ws.tol_rate[()] * qd.abs(ws.rz_slots[parity])
        ws.cond[()] = 1


@qd.func(requires_top_level=True)
def func_linear_pcg_matvec_update(
    x: qd.template(),
    ws: qd.template(),
    matrix: qd.template(),
):
    """Run the first half of an iteration: the product of the matrix and the search direction, then the solution update.

    The sparse matrix-vector product (SpMV) Ap = A p accumulates p.Ap in energy form. Then alpha = rz / p.Ap, inlined,
    updates x += alpha p and r -= alpha Ap, and the rz_new slot is zeroed for the dot that follows. The energy form sums
    w p_r^T B p_c over the stored upper blocks (w = 1 on the diagonal, 2 off it, for the mirrored lower block), so every
    term is local to one entry and needs no separate dot or barrier. Ap and p.Ap are zero on entry (init, or the tail of
    the previous iteration). Next, the caller applies its preconditioner and calls func_linear_pcg_direction_update for
    the second half.
    """
    func_linear_pcg_spmv(ws, matrix)
    func_linear_pcg_update_xr(x, ws)


@qd.func(requires_top_level=True)
def func_linear_pcg_spmv(
    ws: qd.template(),
    matrix: qd.template(),
):
    """Add the sparse matrix-vector product (SpMV) of the search direction to Ap, with p.Ap in energy form.

    This is the first part of func_linear_pcg_matvec_update. A caller whose operator has a term beyond the matrix adds
    it to Ap and p.Ap before func_linear_pcg_update_xr.

    The products of one row are summed before they reach Ap, which spares an atomic per entry. A GPU sums the runs of a
    row inside a warp by a segmented shuffle reduction. A CPU thread sums them along its chunk of the row-sorted
    entries, which a row leaves only at the chunk ends (see PCGWorkspace).
    """
    if qd.static(gs.backend == gs.cpu):
        _func_linear_pcg_spmv_cpu(ws, matrix)
    else:
        _func_linear_pcg_spmv_gpu(ws, matrix)


@qd.func(requires_top_level=True)
def _func_linear_pcg_spmv_cpu(
    ws: qd.template(),
    matrix: qd.template(),
):
    """Run func_linear_pcg_spmv over contiguous chunks of the row-sorted entries, one thread per chunk.

    The products of a row accumulate in the thread and add once to Ap where the row ends, the mirrored products of the
    lower triangle add per entry, and p.Ap adds once per chunk.
    """
    qd.loop_config(name="pcg_spmv_pap_cpu")
    for chunk in range(_CPU_CHUNKS):
        if ws.converged[()] == 0:
            nnz = matrix.bcoo_nnz[()]
            size = (nnz + _CPU_CHUNKS - 1) // _CPU_CHUNKS
            energy_sum = ws.value_type(0.0)
            row_sum = qd.Vector.zero(ws.value_type, 3)
            current_row = qd.i32(-1)
            for entry in range(chunk * size, qd.min(chunk * size + size, nnz)):
                row, col, block = matrix.read_bcoo(entry)
                if row != current_row:
                    if current_row >= 0:
                        for d in qd.static(range(3)):
                            qd.atomic_add(ws.ap[3 * current_row + d], row_sum[d])
                    current_row = row
                    row_sum = qd.Vector.zero(ws.value_type, 3)
                p_col = qd.Vector([ws.p[3 * col + d] for d in qd.static(range(3))], dt=ws.value_type)
                product = block @ p_col
                row_sum += product
                if row != col:
                    p_row = qd.Vector([ws.p[3 * row + d] for d in qd.static(range(3))], dt=ws.value_type)
                    product_mirrored = block.transpose() @ p_row
                    for d in qd.static(range(3)):
                        qd.atomic_add(ws.ap[3 * col + d], product_mirrored[d])
                    energy_sum += 2.0 * p_row.dot(product)
                else:
                    energy_sum += p_col.dot(product)
            if current_row >= 0:
                for d in qd.static(range(3)):
                    qd.atomic_add(ws.ap[3 * current_row + d], row_sum[d])
            if energy_sum != 0.0:
                qd.atomic_add(ws.p_dot_ap[()], energy_sum)


@qd.func(requires_top_level=True)
def _func_linear_pcg_spmv_gpu(
    ws: qd.template(),
    matrix: qd.template(),
):
    qd.loop_config(name="pcg_spmv_pap", block_dim=PCG_BLOCK)
    for t in range(ws.n_spmv_blocks * PCG_BLOCK):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
        nnz = matrix.bcoo_nnz[()]
        padded = ((nnz + 31) // 32) * 32
        energy_sum = ws.value_type(0.0)
        for i_chunk_ in range((padded + ws.n_spmv_blocks * PCG_BLOCK - 1) // (ws.n_spmv_blocks * PCG_BLOCK)):
            entry = i_chunk_ * ws.n_spmv_blocks * PCG_BLOCK + t
            if entry < padded:
                row = qd.i32(-1)
                product = qd.Vector.zero(ws.value_type, 3)
                if entry < nnz and ws.converged[()] == 0:
                    row_, col, block = matrix.read_bcoo(entry)
                    row = row_
                    p_col = qd.Vector([ws.p[3 * col + d] for d in qd.static(range(3))], dt=ws.value_type)
                    product = block @ p_col
                    if row != col:
                        p_row = qd.Vector([ws.p[3 * row + d] for d in qd.static(range(3))], dt=ws.value_type)
                        product_mirrored = block.transpose() @ p_row
                        for d in qd.static(range(3)):
                            qd.atomic_add(ws.ap[3 * col + d], product_mirrored[d])
                        energy_sum += 2.0 * p_row.dot(product)
                    else:
                        energy_sum += p_col.dot(product)
                previous_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.max(lane - 1, 0)))
                next_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.min(lane + 1, 31)))
                is_head = lane == 0 or row != previous_row
                is_tail = lane == 31 or row != next_row
                row_sum = qd.Vector.zero(ws.value_type, 3)
                for d in qd.static(range(3)):
                    row_sum[d] = head_segmented_reduce_add(product[d], is_tail)
                if row >= 0 and is_head:
                    for d in qd.static(range(3)):
                        if row_sum[d] != 0.0:
                            qd.atomic_add(ws.ap[3 * row + d], row_sum[d])
        energy_sum = _warp_sum(energy_sum)
        if lane == 0 and energy_sum != 0.0:
            qd.atomic_add(ws.p_dot_ap[()], energy_sum)


@qd.func(requires_top_level=True)
def func_linear_pcg_update_xr(
    x: qd.template(),
    ws: qd.template(),
):
    """Update the solution and the residual along the search direction, and zero the rz_new slot for the next dot.

    The update is x += alpha p and r -= alpha Ap with alpha = rz / p.Ap. With check_curvature set, a non-positive p.Ap
    along a non-zero direction raises breakdown.
    """
    qd.loop_config(name="pcg_update_xr", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        parity = ws.parity[()]
        if ws.converged[()] == 0:
            p_dot_ap = ws.p_dot_ap[()]
            if t == 0 and ws.check_curvature[()] != 0:
                if p_dot_ap <= 0.0 and ws.rz_slots[parity] != 0.0:
                    ws.breakdown[()] = 1
            alpha = ws.rz_slots[parity] / p_dot_ap if p_dot_ap != 0.0 else ws.value_type(0.0)
            for i_chunk_ in range((3 * ws.n_rows[()] + ws.n_blocks * PCG_BLOCK - 1) // (ws.n_blocks * PCG_BLOCK)):
                i = i_chunk_ * ws.n_blocks * PCG_BLOCK + t
                if i < 3 * ws.n_rows[()]:
                    x[i] += alpha * ws.p[i]
                    ws.r[i] -= alpha * ws.ap[i]
        if t == 0:
            ws.rz_slots[1 - parity] = 0.0


@qd.func(requires_top_level=True)
def func_linear_pcg_direction_update(
    ws: qd.template(),
):
    """Run the second half of an iteration: the dot of the residual and its preconditioned value, then the fused tail.

    The dot is rz_new = r . z. The tail sets p = z + beta p with beta = rz_new / rz, zeroes Ap and p.Ap for the next
    iteration, and marks convergence once |rz_new| falls to the threshold. An rz_new that is not finite fails the
    assertion in debug mode and otherwise ends the iterations unconverged, the caller judging the non-finite solution.

    The decision needs the update of every block to be over. A GPU cannot wait for the grid inside a launch, so every
    block reads the iteration budget before its decrement, and the last block to take a ticket writes the decremented
    budget, clears the loop condition when the budget runs out or the solve converged, resets the ticket and flips the
    rz/rz_new parity. On a CPU (see PCGWorkspace), the end of the parallel update orders the decision of a single thread
    after it.
    """
    _func_dot_into_slot(ws.r, ws.z, ws, 1)
    if qd.static(gs.backend == gs.cpu):
        _func_direction_update_cpu(ws)
    else:
        _func_direction_update_gpu(ws)


@qd.func(requires_top_level=True)
def _func_direction_update_cpu(
    ws: qd.template(),
):
    """Run the fused tail of func_linear_pcg_direction_update on a CPU backend, from plain parallel loops.

    The direction update runs in parallel, then a single thread takes the convergence decision, which the end of the
    parallel loop orders after the update.
    """
    qd.loop_config(name="pcg_update_p_cpu")
    for i in range(3 * ws.n_rows[()]):
        parity = ws.parity[()]
        rz = ws.rz_slots[parity]
        beta = ws.rz_slots[1 - parity] / rz if rz != 0.0 else ws.value_type(0.0)
        ws.ap[i] = 0.0
        if ws.converged[()] == 0:
            ws.p[i] = ws.z[i] + beta * ws.p[i]
    for _ in range(1):
        parity = ws.parity[()]
        remaining = ws.iter_remaining[()] - 1
        not_done = 0
        ws.p_dot_ap[()] = 0.0
        if ws.converged[()] == 0 and remaining > 0:
            rz_new = ws.rz_slots[1 - parity]
            if qd.static(qd_debug_asserts()):
                assert qd_isfinite(rz_new), "PCG: r . z is not finite"
            if qd.abs(rz_new) <= ws.tol_sq[()]:
                ws.converged[()] = 1
            elif qd_isfinite(rz_new):
                not_done = 1
        ws.iter_remaining[()] = remaining
        if remaining <= 0 or not_done == 0:
            ws.cond[()] = 0
        ws.parity[()] = 1 - parity


@qd.func(requires_top_level=True)
def _func_direction_update_gpu(
    ws: qd.template(),
):
    qd.loop_config(name="pcg_update_p_converge", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        parity = ws.parity[()]
        rz = ws.rz_slots[parity]
        rz_new = ws.rz_slots[1 - parity]
        is_converged = ws.converged[()] != 0
        beta = rz_new / rz if rz != 0.0 else ws.value_type(0.0)
        for i_chunk_ in range((3 * ws.n_rows[()] + ws.n_blocks * PCG_BLOCK - 1) // (ws.n_blocks * PCG_BLOCK)):
            i = i_chunk_ * ws.n_blocks * PCG_BLOCK + t
            if i < 3 * ws.n_rows[()]:
                ws.ap[i] = 0.0
                if not is_converged:
                    ws.p[i] = ws.z[i] + beta * ws.p[i]

        remaining = ws.iter_remaining[()] - 1
        local_not_done = 0
        if t == 0:
            ws.p_dot_ap[()] = 0.0
            if not is_converged and remaining > 0:
                if qd.static(qd_debug_asserts()):
                    assert qd_isfinite(rz_new), "PCG: r . z is not finite"
                if qd.abs(rz_new) <= ws.tol_sq[()]:
                    ws.converged[()] = 1
                elif qd_isfinite(rz_new):
                    local_not_done = 1
        block_not_done = qd.simt.block.reduce_add(local_not_done, PCG_BLOCK, qd.i32)
        if qd.simt.block.thread_idx() == 0:
            if block_not_done != 0:
                qd.atomic_exchange(ws.not_done[()], 1)
            # Order the flag write before this block's ticket, so that the last block sees every flag
            qd.simt.grid.mem_fence()
            if qd.atomic_add(ws.ticket[()], 1) == ws.n_blocks - 1:
                ws.iter_remaining[()] = remaining
                not_done = qd.atomic_exchange(ws.not_done[()], 0)
                if remaining <= 0 or not_done == 0:
                    ws.cond[()] = 0
                ws.ticket[()] = 0
                ws.parity[()] = 1 - parity


@qd.func(requires_top_level=True)
def func_apply_block_jacobi(
    precond_inv_diag: qd.template(),
    ws: qd.template(),
):
    """Apply the block Jacobi preconditioner to the residual.

    The result is z = M^-1 r, with M the 3x3 block diagonal whose inverse blocks the caller fills in precond_inv_diag.
    """
    qd.loop_config(name="pcg_apply_block_jacobi")
    for row in range(ws.n_rows[()]):
        for i in qd.static(range(3)):
            value = ws.value_type(0.0)
            for j in qd.static(range(3)):
                value += precond_inv_diag[row * 9 + i * 3 + j] * ws.r[row * 3 + j]
            ws.z[row * 3 + i] = value
