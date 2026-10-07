from __future__ import annotations

import quadrants as qd

from .gipc_make_pd import gipc_make_pd
from .gipc_pfpx import (
    gipc_pfpx_ee,
    gipc_pfpx_ee_mollified,
    gipc_pfpx_pe,
    gipc_pfpx_pe_mollified,
    gipc_pfpx_pp,
    gipc_pfpx_pp_mollified,
    gipc_pfpx_pt,
)


@qd.func
def _barrier_lambda0(I5: qd.f64, dH_tilde: qd.f64, kappa: qd.f64, d_tilde: qd.f64) -> qd.f64:
    result = qd.f64(0.0)
    if d_tilde < 1.0e-6 * dH_tilde:
        g = qd.f64(1.0e-6)
        log_g = qd.f64(-13.815510557964274)
        A_g = log_g + (g - 1.0) / g
        result = 8.0 * kappa * (g * A_g * A_g + (g * g - 1.0) * log_g / g + (g - 1.0) * log_g * A_g * 0.5)
    else:
        log_I5 = qd.log(I5)
        A = log_I5 + (I5 - 1.0) / I5
        result = 8.0 * kappa * (I5 * A * A + (I5 * I5 - 1.0) * log_I5 / I5 + (I5 - 1.0) * log_I5 * A * 0.5)
    return result


@qd.func
def _write_rank1(
    values: qd.template(),  # qd.Vector
    size: qd.template(),  # int
    gradient_scale: qd.f64,
    hessian_scale: qd.f64,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    for row in qd.static(range(size)):
        gradient[row] = gradient_scale * values[row]
        for column in qd.static(range(size)):
            hessian[row * size + column] = hessian_scale * values[row] * values[column]


@qd.func
def gipc_barrier_grad_hess_pt(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    v2x: qd.f64,
    v2y: qd.f64,
    v2z: qd.f64,
    v3x: qd.f64,
    v3y: qd.f64,
    v3z: qd.f64,
    d_hat: qd.f64,
    xi: qd.f64,
    kappa: qd.f64,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    normal_x = (v2y - v1y) * (v3z - v1z) - (v2z - v1z) * (v3y - v1y)
    normal_y = (v2z - v1z) * (v3x - v1x) - (v2x - v1x) * (v3z - v1z)
    normal_z = (v2x - v1x) * (v3y - v1y) - (v2y - v1y) * (v3x - v1x)
    numerator = (v0x - v1x) * normal_x + (v0y - v1y) * normal_y + (v0z - v1z) * normal_z
    normal_squared = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
    distance_squared = numerator * numerator / normal_squared

    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    d_tilde = distance_squared - xi * xi
    d_hat_sqrt = qd.sqrt(dH_tilde)
    distance = qd.sqrt(distance_squared)
    I5 = d_tilde / dH_tilde
    log_I5 = qd.log(I5)
    A = log_I5 + (I5 - 1.0) / I5
    pk1 = 4.0 * kappa * (I5 - 1.0) * log_I5 * A

    values = qd.Vector.zero(qd.f64, 12)
    gipc_pfpx_pt(
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
        d_hat_sqrt,
        values,
    )
    lambda0 = _barrier_lambda0(I5, dH_tilde, kappa, d_tilde)
    _write_rank1(
        values,
        12,
        pk1 * distance / d_hat_sqrt,
        lambda0 * distance_squared / d_tilde,
        gradient,
        hessian,
    )


@qd.func
def gipc_barrier_grad_hess_ee(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    v2x: qd.f64,
    v2y: qd.f64,
    v2z: qd.f64,
    v3x: qd.f64,
    v3y: qd.f64,
    v3z: qd.f64,
    d_hat: qd.f64,
    xi: qd.f64,
    kappa: qd.f64,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    edge_a_x = v1x - v0x
    edge_a_y = v1y - v0y
    edge_a_z = v1z - v0z
    edge_b_x = v3x - v2x
    edge_b_y = v3y - v2y
    edge_b_z = v3z - v2z
    normal_x = edge_a_y * edge_b_z - edge_a_z * edge_b_y
    normal_y = edge_a_z * edge_b_x - edge_a_x * edge_b_z
    normal_z = edge_a_x * edge_b_y - edge_a_y * edge_b_x
    offset_x = v2x - v0x
    offset_y = v2y - v0y
    offset_z = v2z - v0z
    numerator = offset_x * normal_x + offset_y * normal_y + offset_z * normal_z
    normal_squared = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
    distance_squared = numerator * numerator / normal_squared

    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    d_tilde = distance_squared - xi * xi
    d_hat_sqrt = qd.sqrt(dH_tilde)
    distance = qd.sqrt(distance_squared)
    I5 = d_tilde / dH_tilde
    log_I5 = qd.log(I5)
    A = log_I5 + (I5 - 1.0) / I5
    pk1 = 4.0 * kappa * (I5 - 1.0) * log_I5 * A

    values = qd.Vector.zero(qd.f64, 12)
    gipc_pfpx_ee(
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
        d_hat_sqrt,
        values,
    )
    lambda0 = _barrier_lambda0(I5, dH_tilde, kappa, d_tilde)
    _write_rank1(
        values,
        12,
        pk1 * distance / d_hat_sqrt,
        lambda0 * distance_squared / d_tilde,
        gradient,
        hessian,
    )


@qd.func
def gipc_barrier_grad_hess_ee_rank1(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    v2x: qd.f64,
    v2y: qd.f64,
    v2z: qd.f64,
    v3x: qd.f64,
    v3y: qd.f64,
    v3z: qd.f64,
    d_hat: qd.f64,
    xi: qd.f64,
    kappa: qd.f64,
    v_out: qd.template(),  # qd.Vector
    grad_scale: qd.template(),  # qd.Vector
    hess_coef: qd.template(),  # qd.Vector
):
    edge_a_x = v1x - v0x
    edge_a_y = v1y - v0y
    edge_a_z = v1z - v0z
    edge_b_x = v3x - v2x
    edge_b_y = v3y - v2y
    edge_b_z = v3z - v2z
    normal_x = edge_a_y * edge_b_z - edge_a_z * edge_b_y
    normal_y = edge_a_z * edge_b_x - edge_a_x * edge_b_z
    normal_z = edge_a_x * edge_b_y - edge_a_y * edge_b_x
    offset_x = v2x - v0x
    offset_y = v2y - v0y
    offset_z = v2z - v0z
    numerator = offset_x * normal_x + offset_y * normal_y + offset_z * normal_z
    normal_squared = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
    distance_squared = numerator * numerator / normal_squared

    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    d_tilde = distance_squared - xi * xi
    distance = qd.sqrt(distance_squared)
    d_hat_sqrt = qd.sqrt(dH_tilde)
    gipc_pfpx_ee(
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
        d_hat_sqrt,
        v_out,
    )

    I5 = d_tilde / dH_tilde
    log_I5 = qd.log(I5)
    A = log_I5 + (I5 - 1.0) / I5
    pk1 = 4.0 * kappa * (I5 - 1.0) * log_I5 * A
    grad_scale[0] = pk1 * distance / d_hat_sqrt
    hess_coef[0] = _barrier_lambda0(I5, dH_tilde, kappa, d_tilde) * distance_squared / d_tilde


@qd.func
def gipc_barrier_grad_hess_pp(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    d_hat: qd.f64,
    xi: qd.f64,
    kappa: qd.f64,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    dx = v0x - v1x
    dy = v0y - v1y
    dz = v0z - v1z
    distance_squared = dx * dx + dy * dy + dz * dz
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    d_tilde = distance_squared - xi * xi
    d_hat_sqrt = qd.sqrt(dH_tilde)
    distance = qd.sqrt(distance_squared)
    I5 = d_tilde / dH_tilde
    log_I5 = qd.log(I5)
    A = log_I5 + (I5 - 1.0) / I5
    pk1 = 4.0 * kappa * (I5 - 1.0) * log_I5 * A

    values = qd.Vector.zero(qd.f64, 6)
    gipc_pfpx_pp(v0x, v0y, v0z, v1x, v1y, v1z, d_hat_sqrt, values)
    lambda0 = _barrier_lambda0(I5, dH_tilde, kappa, d_tilde)
    _write_rank1(
        values,
        6,
        pk1 * distance / d_hat_sqrt,
        lambda0 * distance_squared / d_tilde,
        gradient,
        hessian,
    )


@qd.func
def gipc_barrier_grad_hess_pe(
    v0x: qd.f64,
    v0y: qd.f64,
    v0z: qd.f64,
    v1x: qd.f64,
    v1y: qd.f64,
    v1z: qd.f64,
    v2x: qd.f64,
    v2y: qd.f64,
    v2z: qd.f64,
    d_hat: qd.f64,
    xi: qd.f64,
    kappa: qd.f64,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    cross_x = (v1y - v0y) * (v2z - v0z) - (v1z - v0z) * (v2y - v0y)
    cross_y = (v1z - v0z) * (v2x - v0x) - (v1x - v0x) * (v2z - v0z)
    cross_z = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x)
    edge_x = v2x - v1x
    edge_y = v2y - v1y
    edge_z = v2z - v1z
    distance_squared = (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z) / (
        edge_x * edge_x + edge_y * edge_y + edge_z * edge_z
    )

    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    d_tilde = distance_squared - xi * xi
    d_hat_sqrt = qd.sqrt(dH_tilde)
    distance = qd.sqrt(distance_squared)
    I5 = d_tilde / dH_tilde
    log_I5 = qd.log(I5)
    A = log_I5 + (I5 - 1.0) / I5
    pk1 = 4.0 * kappa * (I5 - 1.0) * log_I5 * A

    values = qd.Vector.zero(qd.f64, 9)
    gipc_pfpx_pe(
        v0x,
        v0y,
        v0z,
        v1x,
        v1y,
        v1z,
        v2x,
        v2y,
        v2z,
        d_hat_sqrt,
        values,
    )
    lambda0 = _barrier_lambda0(I5, dH_tilde, kappa, d_tilde)
    _write_rank1(
        values,
        9,
        pk1 * distance / d_hat_sqrt,
        lambda0 * distance_squared / d_tilde,
        gradient,
        hessian,
    )


@qd.func
def _barrier_grad_hess_mollified_impl(
    column4: qd.template(),  # qd.Vector
    column8: qd.template(),  # qd.Vector
    distance_squared,
    I1,
    eps_x,
    d_hat,
    xi,
    kappa,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    if I1 == 0.0:
        for component in qd.static(range(12)):
            gradient[component] = 0.0
        for component in range(144):
            hessian[component] = 0.0
    else:
        dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
        d_tilde = distance_squared - xi * xi
        I2 = d_tilde / dH_tilde
        c = qd.sqrt(I1)
        f22 = qd.sqrt(d_tilde) / qd.sqrt(dH_tilde)
        log_I2 = qd.log(I2)
        log_I2_squared = log_I2 * log_I2
        eps_x_squared = eps_x * eps_x
        I2_minus_one = I2 - 1.0
        A2 = log_I2 + I2_minus_one / I2

        p1 = 4.0 * kappa * (eps_x - I1) / eps_x_squared * I2_minus_one * I2_minus_one * log_I2_squared
        mollifier = I1 * (2.0 * eps_x - I1) / eps_x_squared
        p2 = 4.0 * kappa * mollifier * I2_minus_one * log_I2 * A2
        for component in qd.static(range(12)):
            gradient[component] = column4[component] * p1 * c + column8[component] * p2 * f22

        lambda10 = 4.0 * kappa * (eps_x - 3.0 * I1) / eps_x_squared * I2_minus_one * I2_minus_one * log_I2_squared
        lambda20 = mollifier * _barrier_lambda0(I2, dH_tilde, kappa, d_tilde)
        lambda_g1g = 16.0 * kappa * (eps_x - I1) * I2_minus_one * log_I2 * A2 * c * f22 / eps_x_squared
        projected = qd.Vector.zero(qd.f64, 3)
        gipc_make_pd(lambda10, lambda_g1g, lambda20, projected)
        for row in qd.static(range(12)):
            for column in qd.static(range(12)):
                hessian[row * 12 + column] = (
                    projected[0] * column4[row] * column4[column]
                    + projected[1] * (column4[row] * column8[column] + column8[row] * column4[column])
                    + projected[2] * column8[row] * column8[column]
                )


@qd.func
def gipc_barrier_grad_hess_ee_mollified(
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
    rest_v0x,
    rest_v0y,
    rest_v0z,
    rest_v1x,
    rest_v1y,
    rest_v1z,
    rest_v2x,
    rest_v2y,
    rest_v2z,
    rest_v3x,
    rest_v3y,
    rest_v3z,
    d_hat,
    xi,
    kappa,
    distance_squared,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    edge_a_x = v1x - v0x
    edge_a_y = v1y - v0y
    edge_a_z = v1z - v0z
    edge_b_x = v3x - v2x
    edge_b_y = v3y - v2y
    edge_b_z = v3z - v2z
    cross_x = edge_a_y * edge_b_z - edge_a_z * edge_b_y
    cross_y = edge_a_z * edge_b_x - edge_a_x * edge_b_z
    cross_z = edge_a_x * edge_b_y - edge_a_y * edge_b_x
    I1 = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
    rest_edge_a_squared = (rest_v0x - rest_v1x) ** 2 + (rest_v0y - rest_v1y) ** 2 + (rest_v0z - rest_v1z) ** 2
    rest_edge_b_squared = (rest_v2x - rest_v3x) ** 2 + (rest_v2y - rest_v3y) ** 2 + (rest_v2z - rest_v3z) ** 2
    eps_x = 1.0e-3 * rest_edge_a_squared * rest_edge_b_squared
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    column4 = qd.Vector.zero(qd.f64, 12)
    column8 = qd.Vector.zero(qd.f64, 12)
    gipc_pfpx_ee_mollified(
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
        qd.sqrt(dH_tilde),
        column4,
        column8,
    )
    scale = qd.sqrt(distance_squared) / qd.sqrt(distance_squared - xi * xi)
    for component in qd.static(range(12)):
        column8[component] = column8[component] * scale
    _barrier_grad_hess_mollified_impl(
        column4,
        column8,
        distance_squared,
        I1,
        eps_x,
        d_hat,
        xi,
        kappa,
        gradient,
        hessian,
    )


@qd.func
def gipc_barrier_grad_hess_pp_mollified(
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
    rest_v0x,
    rest_v0y,
    rest_v0z,
    rest_v1x,
    rest_v1y,
    rest_v1z,
    rest_v2x,
    rest_v2y,
    rest_v2z,
    rest_v3x,
    rest_v3y,
    rest_v3z,
    d_hat,
    xi,
    kappa,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    distance_x = v0x - v1x
    distance_y = v0y - v1y
    distance_z = v0z - v1z
    distance_squared = distance_x * distance_x + distance_y * distance_y + distance_z * distance_z
    edge_a_x = v2x - v0x
    edge_a_y = v2y - v0y
    edge_a_z = v2z - v0z
    edge_b_x = v3x - v1x
    edge_b_y = v3y - v1y
    edge_b_z = v3z - v1z
    cross_x = edge_a_y * edge_b_z - edge_a_z * edge_b_y
    cross_y = edge_a_z * edge_b_x - edge_a_x * edge_b_z
    cross_z = edge_a_x * edge_b_y - edge_a_y * edge_b_x
    I1 = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
    rest_edge_a_squared = (rest_v0x - rest_v2x) ** 2 + (rest_v0y - rest_v2y) ** 2 + (rest_v0z - rest_v2z) ** 2
    rest_edge_b_squared = (rest_v1x - rest_v3x) ** 2 + (rest_v1y - rest_v3y) ** 2 + (rest_v1z - rest_v3z) ** 2
    eps_x = 1.0e-3 * rest_edge_a_squared * rest_edge_b_squared
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    column4 = qd.Vector.zero(qd.f64, 12)
    column8 = qd.Vector.zero(qd.f64, 12)
    gipc_pfpx_pp_mollified(
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
        qd.sqrt(dH_tilde),
        column4,
        column8,
    )
    scale = qd.sqrt(distance_squared) / qd.sqrt(distance_squared - xi * xi)
    for component in qd.static(range(12)):
        column8[component] = column8[component] * scale
    _barrier_grad_hess_mollified_impl(
        column4,
        column8,
        distance_squared,
        I1,
        eps_x,
        d_hat,
        xi,
        kappa,
        gradient,
        hessian,
    )


@qd.func
def gipc_barrier_grad_hess_pe_mollified(
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
    rest_v0x,
    rest_v0y,
    rest_v0z,
    rest_v1x,
    rest_v1y,
    rest_v1z,
    rest_v2x,
    rest_v2y,
    rest_v2z,
    rest_v3x,
    rest_v3y,
    rest_v3z,
    d_hat,
    xi,
    kappa,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    edge_x = v2x - v1x
    edge_y = v2y - v1y
    edge_z = v2z - v1z
    point_x = v0x - v1x
    point_y = v0y - v1y
    point_z = v0z - v1z
    cross_distance_x = point_y * edge_z - point_z * edge_y
    cross_distance_y = point_z * edge_x - point_x * edge_z
    cross_distance_z = point_x * edge_y - point_y * edge_x
    distance_squared = (
        cross_distance_x * cross_distance_x + cross_distance_y * cross_distance_y + cross_distance_z * cross_distance_z
    ) / (edge_x * edge_x + edge_y * edge_y + edge_z * edge_z)
    mollifier_edge_a_x = v3x - v0x
    mollifier_edge_a_y = v3y - v0y
    mollifier_edge_a_z = v3z - v0z
    mollifier_edge_b_x = v2x - v1x
    mollifier_edge_b_y = v2y - v1y
    mollifier_edge_b_z = v2z - v1z
    cross_x = mollifier_edge_a_y * mollifier_edge_b_z - mollifier_edge_a_z * mollifier_edge_b_y
    cross_y = mollifier_edge_a_z * mollifier_edge_b_x - mollifier_edge_a_x * mollifier_edge_b_z
    cross_z = mollifier_edge_a_x * mollifier_edge_b_y - mollifier_edge_a_y * mollifier_edge_b_x
    I1 = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
    rest_edge_a_squared = (rest_v0x - rest_v3x) ** 2 + (rest_v0y - rest_v3y) ** 2 + (rest_v0z - rest_v3z) ** 2
    rest_edge_b_squared = (rest_v1x - rest_v2x) ** 2 + (rest_v1y - rest_v2y) ** 2 + (rest_v1z - rest_v2z) ** 2
    eps_x = 1.0e-3 * rest_edge_a_squared * rest_edge_b_squared
    dH_tilde = d_hat * d_hat + 2.0 * d_hat * xi
    column4 = qd.Vector.zero(qd.f64, 12)
    column8 = qd.Vector.zero(qd.f64, 12)
    gipc_pfpx_pe_mollified(
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
        qd.sqrt(dH_tilde),
        column4,
        column8,
    )
    scale = qd.sqrt(distance_squared) / qd.sqrt(distance_squared - xi * xi)
    for component in qd.static(range(12)):
        column8[component] = column8[component] * scale
    _barrier_grad_hess_mollified_impl(
        column4,
        column8,
        distance_squared,
        I1,
        eps_x,
        d_hat,
        xi,
        kappa,
        gradient,
        hessian,
    )
