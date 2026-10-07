from __future__ import annotations

import quadrants as qd


@qd.func
def pair_d_hat_pt(
    d_hats: qd.template(),  # qd.Ndarray
    global_d_hat,
    vP,
    vT0,
    vT1,
    vT2,
):
    return (d_hats[vP] + d_hats[vT0]) * 0.5


@qd.func
def pair_d_hat_ee(
    d_hats: qd.template(),  # qd.Ndarray
    global_d_hat,
    ea0,
    ea1,
    eb0,
    eb1,
):
    return (d_hats[ea0] + d_hats[eb0]) * 0.5


@qd.func
def pair_d_hat_pe(
    d_hats: qd.template(),  # qd.Ndarray
    global_d_hat,
    vP,
    e0,
    e1,
):
    return (d_hats[vP] + d_hats[e0]) * 0.5


@qd.func
def pair_d_hat_pp(
    d_hats: qd.template(),  # qd.Ndarray
    global_d_hat,
    v0,
    v1,
):
    return (d_hats[v0] + d_hats[v1]) * 0.5


@qd.func
def pair_d_hat_ph(
    d_hats: qd.template(),  # qd.Ndarray
    global_d_hat,
    vert_idx,
):
    return d_hats[vert_idx]
