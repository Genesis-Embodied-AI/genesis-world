# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""GPU direct solver of the MochiSolver: fused blocked Cholesky factorization and triangular solves of the dense Newton
matrix of every environment with cooperative TxT register tiles, the factor staged in shared memory (the same algorithm
as the rigid solver's constraint Hessian factorization)."""

import quadrants as qd

import genesis as gs

from .data import MochiInfo, MochiState
from .islands import MochiIslandState


@qd.func
def _cholesky_and_solve_fused_tiled_impl(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    island_state: MochiIslandState,
    mochi_config: qd.template(),
    TileCls: qd.template(),
):
    """Factorize H = L L^T (left-looking blocked Cholesky with TxT register tiles, completed tiles kept in shared
    memory), then solve L L^T dx = res in place. One thread block of T threads per environment; partial tiles beyond
    n_dofs are padded with the identity. The pivot is floored relative to the original diagonal like the scalar
    solver."""
    T = qd.static(mochi_config.cholesky_tile_size)
    LOG2_T = qd.static(T.bit_length() - 1)
    MAX_DOFS = qd.static(mochi_config.tiled_n_dofs)

    EPS = mochi_info.EPS[None]
    _B = mochi_state.is_active.shape[0]
    n_dofs = mochi_state.res.shape[0]
    N_BLOCKS = (n_dofs + T - 1) // T

    qd.loop_config(name="mochi_cholesky_and_solve_fused_tiled", block_dim=T)
    for i in range(_B * T):
        tid = i % T
        i_b = i // T
        if i_b >= _B:
            continue
        if not mochi_state.is_active[i_b] or not island_state.uses_dense[i_b]:
            continue

        # +1 padding avoids shared memory bank conflicts on column-wise access.
        L_sh = qd.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), gs.qd_float)
        v_sh = qd.simt.block.SharedArray((MAX_DOFS,), gs.qd_float)

        for kb in range(N_BLOCKS):
            k0 = kb * T
            k1 = qd.min(k0 + T, n_dofs)

            L_kk = TileCls.eye(dtype=gs.qd_float)
            L_kk[:] = mochi_state.H_dense[i_b, k0:k1, k0:k1]
            for jb in range(kb):
                j0 = jb * T
                for t in range(T):
                    v = L_sh[k0:k1, j0 + t]
                    L_kk -= qd.outer(v, v)
            d_row = k0 + tid
            diag_orig = gs.qd_float(1.0)
            if d_row < n_dofs:
                diag_orig = mochi_state.H_dense[i_b, d_row, d_row]
            L_kk.cholesky_(EPS * qd.max(diag_orig, EPS))

            for ib in range(kb + 1, N_BLOCKS):
                i0 = ib * T
                i1 = qd.min(i0 + T, n_dofs)
                L_ik = TileCls.zeros(dtype=gs.qd_float)
                L_ik[:] = mochi_state.H_dense[i_b, i0:i1, k0:k1]
                for jb in range(kb):
                    j0 = jb * T
                    for t in range(T):
                        v_own = L_sh[i0:i1, j0 + t]
                        v_diag = L_sh[k0:k1, j0 + t]
                        L_ik -= qd.outer(v_own, v_diag)
                L_kk.solve_triangular_(L_ik)
                L_sh[i0:i1, k0:k1] = L_ik
            L_sh[k0:k1, k0:k1] = L_kk

        # Triangular solves with the T threads striping each row's dot product.
        k = tid
        while k < n_dofs:
            v_sh[k] = mochi_state.res[k, i_b]
            k = k + T
        qd.simt.block.sync()

        for i_d in range(n_dofs):
            dot = gs.qd_float(0.0)
            j = tid
            while j < i_d:
                dot = dot + L_sh[i_d, j] * v_sh[j]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                v_sh[i_d] = (v_sh[i_d] - dot) / L_sh[i_d, i_d]
            qd.simt.block.sync()

        for i_d_ in range(n_dofs):
            i_d = n_dofs - 1 - i_d_
            dot = gs.qd_float(0.0)
            j = i_d + 1 + tid
            while j < n_dofs:
                dot = dot + L_sh[j, i_d] * v_sh[j]
                j = j + T
            dot = qd.simt.subgroup.reduce_all_add_tiled(dot, LOG2_T)
            if tid == 0:
                v_sh[i_d] = (v_sh[i_d] - dot) / L_sh[i_d, i_d]
            qd.simt.block.sync()

        k = tid
        while k < n_dofs:
            mochi_state.dx[k, i_b] = v_sh[k]
            k = k + T


@qd.kernel
def kernel_cholesky_solve_tiled(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    island_state: MochiIslandState,
    mochi_config: qd.template(),
):
    """Tile-size dispatcher of the fused tiled factorization and solve (16x16 or 32x32 tiles)."""
    _cholesky_and_solve_fused_tiled_impl(
        mochi_info,
        mochi_state,
        island_state,
        mochi_config,
        qd.simt.Tile32x32 if qd.static(mochi_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
    )
