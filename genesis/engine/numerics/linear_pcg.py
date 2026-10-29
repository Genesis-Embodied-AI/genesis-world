import numpy as np
import quadrants as qd

from .fsr_reduce import head_segmented_reduce_add
from .gpu_occupancy import cuda_resident_blocks

# Threads of a block of the PCG launches
PCG_BLOCK = 256
# Launch-resource metadata of the PCG kernels: registers per thread of the streaming loops and of the SpMV
_STREAM_REGISTERS = 32
_SPMV_REGISTERS = 64


@qd.data_oriented
class PCGWorkspace:
    """Vectors and device-resident scalars of a preconditioned conjugate gradient over 3x3 block rows.

    The structure is CGQ's masked PCG for one component: five launches per iteration (the SpMV with p.Ap fused in
    energy form, the x/r update with alpha inlined, the caller's preconditioner, the r.z dot, and a fused tail that
    updates p, zeroes the next iteration's Ap and p.Ap, and lets the last block of the grid decide convergence and
    flip the parity of the rz/rz_new slots). Every loop grid-strides over the live block rows `n_rows` with a resident
    grid, whose block count the tail's ticket needs.
    """

    def __init__(self, max_rows: int, value_type) -> None:
        self.value_type = value_type
        self.n_blocks = cuda_resident_blocks(PCG_BLOCK, 0, _STREAM_REGISTERS)
        self.n_spmv_blocks = cuda_resident_blocks(PCG_BLOCK, 0, _SPMV_REGISTERS)
        self.allocate(max_rows)
        np_float = np.float64 if value_type == qd.f64 else np.float32

        def scalar(dtype, value, np_dtype):
            result = qd.ndarray(dtype, shape=())
            result.from_numpy(np.array(value, dtype=np_dtype))
            return result

        self.n_rows = scalar(qd.i32, 0, np.int32)
        self.cond = scalar(qd.i32, 0, np.int32)
        self.max_iter = scalar(qd.i32, 1024, np.int32)
        self.iter_remaining = scalar(qd.i32, 0, np.int32)
        self.tol_rate = scalar(value_type, 1e-4, np_float)
        self.tol_sq = scalar(value_type, 0.0, np_float)
        # rz and rz_new are the two slots of one buffer, selected by the parity the tail flips every iteration
        self.rz_slots = qd.ndarray(value_type, shape=(2,))
        self.parity = scalar(qd.i32, 0, np.int32)
        self.p_dot_ap = scalar(value_type, 0.0, np_float)
        self.converged = scalar(qd.i32, 0, np.int32)
        self.not_done = scalar(qd.i32, 0, np.int32)
        self.ticket = scalar(qd.i32, 0, np.int32)

    def allocate(self, max_rows: int) -> None:
        """(Re)allocate the vectors for a capacity of block rows, keeping this object."""
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
    ws: qd.template(),  # PCGWorkspace
    a: qd.template(),  # qd.Ndarray
    b: qd.template(),  # qd.Ndarray
    slot_offset: qd.template(),  # int: 0 for the rz slot, 1 for rz_new
):
    """rz_slots[parity XOR slot_offset] += a . b over the live rows, one warp reduction and atomic per warp. The
    slot was zeroed by an earlier launch."""
    qd.loop_config(name="pcg_dot_rz", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
        n_rows = ws.n_rows[()]
        padded = ((n_rows + 31) // 32) * 32
        value = ws.value_type(0.0)
        v = t
        while v < padded:
            if v < n_rows and ws.converged[()] == 0:
                for d in qd.static(range(3)):
                    value += a[3 * v + d] * b[3 * v + d]
            v += ws.n_blocks * PCG_BLOCK
        value = _warp_sum(value)
        if lane == 0 and value != 0.0:
            slot = ws.parity[()]
            if qd.static(slot_offset != 0):
                slot = 1 - slot
            qd.atomic_add(ws.rz_slots[slot], value)


@qd.func(requires_top_level=True)
def func_init_linear_pcg_residual(
    ws: qd.template(),  # PCGWorkspace
    b: qd.template(),  # qd.Ndarray
    x: qd.template(),  # qd.Ndarray
):
    """Start solving A x = b from x = 0: x = 0, r = b, Ap = p.Ap = 0, the convergence flag, ticket and parity reset,
    and the iteration budget re-armed.

    The caller applies its preconditioner (z = M^-1 r) next, then calls func_init_linear_pcg_direction.
    """
    qd.loop_config(name="pcg_memset_x", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        i = t
        while i < 3 * ws.n_rows[()]:
            x[i] = 0.0
            i += ws.n_blocks * PCG_BLOCK
    qd.loop_config(name="pcg_copy_b_to_r", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        i = t
        while i < 3 * ws.n_rows[()]:
            ws.r[i] = b[i]
            i += ws.n_blocks * PCG_BLOCK
    qd.loop_config(name="pcg_memset_converged", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        i = t
        while i < 3 * ws.n_rows[()]:
            ws.ap[i] = 0.0
            i += ws.n_blocks * PCG_BLOCK
        if t == 0:
            ws.converged[()] = 0
            ws.p_dot_ap[()] = 0.0
            ws.not_done[()] = 0
            ws.ticket[()] = 0
            ws.parity[()] = 0
            ws.iter_remaining[()] = ws.max_iter[()]


@qd.func(requires_top_level=True)
def func_init_linear_pcg_direction(
    ws: qd.template(),  # PCGWorkspace
):
    """p = z, rz = r . z, the convergence threshold tol_rate |rz0|, and arm the iteration loop."""
    qd.loop_config(name="pcg_copy_z_to_p", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        i = t
        while i < 3 * ws.n_rows[()]:
            ws.p[i] = ws.z[i]
            i += ws.n_blocks * PCG_BLOCK
        if t == 0:
            ws.rz_slots[ws.parity[()]] = 0.0
    _func_dot_into_slot(ws, ws.r, ws.z, 0)
    for _ in range(1):
        ws.tol_sq[()] = ws.tol_rate[()] * qd.abs(ws.rz_slots[ws.parity[()]])
        ws.cond[()] = 1


@qd.func(requires_top_level=True)
def func_linear_pcg_matvec_update(
    ws: qd.template(),  # PCGWorkspace
    matrix: qd.template(),  # BCOOMatrix, symmetric 3x3 blocks
    x: qd.template(),  # qd.Ndarray
):
    """First half of an iteration: the SpMV Ap = A p with p.Ap accumulated in energy form, then x += alpha p and
    r -= alpha Ap with alpha = rz / p.Ap inlined, and the rz_new slot zeroed for the dot that follows.

    The energy form sums w p_r^T B p_c over the stored upper blocks (w = 1 on the diagonal, 2 off it, for the mirrored
    lower block), so every term is local to one entry and no separate dot or barrier is needed. Ap and p.Ap are zero
    on entry (init, or the previous iteration's tail). The caller applies its preconditioner next, then calls
    func_linear_pcg_direction_update.
    """
    func_linear_pcg_spmv(ws, matrix)
    func_linear_pcg_update_xr(ws, x)


@qd.func(requires_top_level=True)
def func_linear_pcg_spmv(
    ws: qd.template(),  # PCGWorkspace
    matrix: qd.template(),  # BCOOMatrix, symmetric 3x3 blocks
):
    """The SpMV half of func_linear_pcg_matvec_update: Ap += A p with p.Ap in energy form. A caller whose operator has
    a term beyond the matrix adds it to Ap and p.Ap before func_linear_pcg_update_xr."""
    qd.loop_config(name="pcg_spmv_pap", block_dim=PCG_BLOCK)
    for t in range(ws.n_spmv_blocks * PCG_BLOCK):
        lane = qd.i32(qd.simt.subgroup.invocation_id()) & 31
        nnz = matrix.bcoo_nnz[()]
        padded = ((nnz + 31) // 32) * 32
        energy_sum = ws.value_type(0.0)
        entry = t
        while entry < padded:
            row = qd.i32(-1)
            product = qd.Vector.zero(ws.value_type, 3)
            if entry < nnz and ws.converged[()] == 0:
                row_, col, block = matrix.read_bcoo(entry)
                row = row_
                p_col = qd.Vector([ws.p[3 * col + d] for d in qd.static(range(3))], dt=ws.value_type)
                product = block @ p_col
                if row != col:
                    p_row = qd.Vector([ws.p[3 * row + d] for d in qd.static(range(3))], dt=ws.value_type)
                    mirrored = block.transpose() @ p_row
                    for d in qd.static(range(3)):
                        qd.atomic_add(ws.ap[3 * col + d], mirrored[d])
                    energy_sum += 2.0 * p_row.dot(product)
                else:
                    energy_sum += p_col.dot(product)
            previous_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.max(lane - 1, 0)))
            next_row = qd.simt.subgroup.shuffle(row, qd.u32(qd.min(lane + 1, 31)))
            is_head = lane == 0 or row != previous_row
            is_tail = lane == 31 or row != next_row
            reduced = qd.Vector.zero(ws.value_type, 3)
            for d in qd.static(range(3)):
                reduced[d] = head_segmented_reduce_add(product[d], is_tail)
            if row >= 0 and is_head:
                for d in qd.static(range(3)):
                    if reduced[d] != 0.0:
                        qd.atomic_add(ws.ap[3 * row + d], reduced[d])
            entry += ws.n_spmv_blocks * PCG_BLOCK
        energy_sum = _warp_sum(energy_sum)
        if lane == 0 and energy_sum != 0.0:
            qd.atomic_add(ws.p_dot_ap[()], energy_sum)


@qd.func(requires_top_level=True)
def func_linear_pcg_update_xr(
    ws: qd.template(),  # PCGWorkspace
    x: qd.template(),  # qd.Ndarray
):
    """x += alpha p and r -= alpha Ap with alpha = rz / p.Ap, and the rz_new slot zeroed for the dot that follows."""
    qd.loop_config(name="pcg_update_xr", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        if ws.converged[()] == 0:
            p_dot_ap = ws.p_dot_ap[()]
            alpha = ws.rz_slots[ws.parity[()]] / p_dot_ap if p_dot_ap != 0.0 else ws.value_type(0.0)
            i = t
            while i < 3 * ws.n_rows[()]:
                x[i] += alpha * ws.p[i]
                ws.r[i] -= alpha * ws.ap[i]
                i += ws.n_blocks * PCG_BLOCK
        if t == 0:
            ws.rz_slots[1 - ws.parity[()]] = 0.0


@qd.func(requires_top_level=True)
def func_linear_pcg_direction_update(
    ws: qd.template(),  # PCGWorkspace
):
    """Second half of an iteration: rz_new = r . z, then the fused tail.

    The tail sets p = z + beta p with beta = rz_new / rz, zeroes Ap and p.Ap for the next iteration, and marks
    convergence once |rz_new| falls to the threshold. Every block reads the iteration budget before its decrement;
    the last block to take a ticket writes the decremented budget, clears the loop condition when the budget runs out
    or the component converged, resets the ticket and flips the rz/rz_new parity.
    """
    _func_dot_into_slot(ws, ws.r, ws.z, 1)
    qd.loop_config(name="pcg_update_p_converge", block_dim=PCG_BLOCK)
    for t in range(ws.n_blocks * PCG_BLOCK):
        parity = ws.parity[()]
        rz = ws.rz_slots[parity]
        rz_new = ws.rz_slots[1 - parity]
        frozen = ws.converged[()] != 0
        beta = rz_new / rz if rz != 0.0 else ws.value_type(0.0)
        i = t
        while i < 3 * ws.n_rows[()]:
            ws.ap[i] = 0.0
            if not frozen:
                ws.p[i] = ws.z[i] + beta * ws.p[i]
            i += ws.n_blocks * PCG_BLOCK

        remaining = ws.iter_remaining[()] - 1
        local_not_done = 0
        if t == 0:
            ws.p_dot_ap[()] = 0.0
            if not frozen and remaining > 0:
                if qd.abs(rz_new) <= ws.tol_sq[()]:
                    ws.converged[()] = 1
                else:
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
    ws: qd.template(),  # PCGWorkspace
    precond_inv_diag: qd.template(),  # qd.Ndarray of 9 scalars per block row
):
    """z = M^-1 r with M the 3x3 block diagonal whose inverse blocks the caller filled."""
    qd.loop_config(name="pcg_apply_block_jacobi")
    for row in range(ws.n_rows[()]):
        for i in qd.static(range(3)):
            value = ws.value_type(0.0)
            for j in qd.static(range(3)):
                value += precond_inv_diag[row * 9 + i * 3 + j] * ws.r[row * 3 + j]
            ws.z[row * 3 + i] = value
