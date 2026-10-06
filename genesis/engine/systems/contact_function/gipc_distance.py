from __future__ import annotations

import quadrants as qd

from .distance_primitives import ee_distance2, pe_distance2, pp_distance2, pt_distance2


@qd.func
def gipc_d_PP(v0x, v0y, v0z, v1x, v1y, v1z):
    return pp_distance2(v0x, v0y, v0z, v1x, v1y, v1z)


@qd.func
def gipc_d_PE(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z):
    return pe_distance2(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z)


@qd.func
def gipc_d_PT(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z):
    return pt_distance2(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z)


@qd.func
def gipc_d_EE(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z):
    return ee_distance2(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z)


@qd.func
def gipc_compute_eps_x(
    v0x,
    v0y,
    v0z,
    v1x,
    v1y,
    v1z,
    v2x,
    v2y,
    v2z,
    v3x,
    v3y,
    v3z,
):
    edge_a_x = v0x - v1x
    edge_a_y = v0y - v1y
    edge_a_z = v0z - v1z
    edge_b_x = v2x - v3x
    edge_b_y = v2y - v3y
    edge_b_z = v2z - v3z
    return (
        1.0e-3
        * (edge_a_x * edge_a_x + edge_a_y * edge_a_y + edge_a_z * edge_a_z)
        * (edge_b_x * edge_b_x + edge_b_y * edge_b_y + edge_b_z * edge_b_z)
    )
