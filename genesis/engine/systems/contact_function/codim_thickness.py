from __future__ import annotations

import quadrants as qd


@qd.func
def pair_thickness_pt(thicknesses: qd.template(), vP, vT0, vT1, vT2):
    xi_point = thicknesses[vP]
    xi_triangle = qd.max(thicknesses[vT0], qd.max(thicknesses[vT1], thicknesses[vT2]))
    return xi_point + xi_triangle


@qd.func
def pair_thickness_ee(thicknesses: qd.template(), ea0, ea1, eb0, eb1):
    xi_a = qd.max(thicknesses[ea0], thicknesses[ea1])
    xi_b = qd.max(thicknesses[eb0], thicknesses[eb1])
    return xi_a + xi_b


@qd.func
def pair_thickness_pe(thicknesses: qd.template(), vP, e0, e1):
    return thicknesses[vP] + qd.max(thicknesses[e0], thicknesses[e1])


@qd.func
def pair_thickness_pp(thicknesses: qd.template(), v0, v1):
    return thicknesses[v0] + thicknesses[v1]


@qd.func
def pair_thickness_ph(thicknesses: qd.template(), vert_idx):
    return thicknesses[vert_idx]
