from __future__ import annotations

import quadrants as qd


@qd.func
def ct_query(
    table: qd.template(),  # qd.Ndarray
    N,
    eid_a,
    eid_b,
):
    return table[eid_a * N + eid_b]


@qd.func
def ct_query_i(
    table: qd.template(),  # qd.Ndarray
    N,
    eid_a,
    eid_b,
):
    return table[eid_a * N + eid_b]


@qd.func
def ct_enabled(
    mask: qd.template(),  # qd.Ndarray
    N,
    eid_a,
    eid_b,
):
    return mask[eid_a * N + eid_b] != 0


@qd.func
def ct_query_pt(
    table: qd.template(),  # qd.Ndarray
    N,
    eid_p,
    eid_t0,
    eid_t1,
    eid_t2,
):
    return (table[eid_p * N + eid_t0] + table[eid_p * N + eid_t1] + table[eid_p * N + eid_t2]) * (1.0 / 3.0)


@qd.func
def ct_query_pt_enabled(
    table: qd.template(),  # qd.Ndarray
    N,
    eid_p,
    eid_t0,
    eid_t1,
    eid_t2,
):
    return table[eid_p * N + eid_t0] != 0 and table[eid_p * N + eid_t1] != 0 and table[eid_p * N + eid_t2] != 0


@qd.func
def ct_query_pt_flag(
    table: qd.template(),  # qd.Ndarray
    N,
    eid_p,
    eid_t0,
    eid_t1,
    eid_t2,
):
    return qd.i32(ct_query_pt_enabled(table, N, eid_p, eid_t0, eid_t1, eid_t2))


@qd.func
def ct_query_ee(
    table: qd.template(),  # qd.Ndarray
    N,
    eid_a0,
    eid_a1,
    eid_b0,
    eid_b1,
):
    return (
        table[eid_a0 * N + eid_b0]
        + table[eid_a0 * N + eid_b1]
        + table[eid_a1 * N + eid_b0]
        + table[eid_a1 * N + eid_b1]
    ) * 0.25


@qd.func
def ct_query_pe(
    table: qd.template(),  # qd.Ndarray
    N,
    eid_p,
    eid_e0,
    eid_e1,
):
    return (table[eid_p * N + eid_e0] + table[eid_p * N + eid_e1]) * 0.5


@qd.func
def ct_enabled_pt(
    mask: qd.template(),  # qd.Ndarray
    N,
    eid_p,
    eid_t0,
    eid_t1,
    eid_t2,
):
    return mask[eid_p * N + eid_t0] != 0 and mask[eid_p * N + eid_t1] != 0 and mask[eid_p * N + eid_t2] != 0


@qd.func
def ct_enabled_pe(
    mask: qd.template(),  # qd.Ndarray
    N,
    eid_p,
    eid_e0,
    eid_e1,
):
    return mask[eid_p * N + eid_e0] != 0 and mask[eid_p * N + eid_e1] != 0


@qd.func
def ct_enabled_ee(
    mask: qd.template(),  # qd.Ndarray
    N,
    eid_a0,
    eid_a1,
    eid_b0,
    eid_b1,
):
    return (
        mask[eid_a0 * N + eid_b0] != 0
        and mask[eid_a0 * N + eid_b1] != 0
        and mask[eid_a1 * N + eid_b0] != 0
        and mask[eid_a1 * N + eid_b1] != 0
    )
