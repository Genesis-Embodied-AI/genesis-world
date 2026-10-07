from __future__ import annotations

import quadrants as qd

from .distance_primitives import ee_distance2, pe_distance2, pp_distance2, pt_distance2
from .friction_utils import (
    ee_friction_closest_point,
    ee_friction_tangent_basis,
    friction_energy,
    friction_from_weights,
    pe_friction_closest_point,
    pe_friction_tangent_basis,
    pp_friction_tangent_basis,
    pt_friction_closest_point,
    pt_friction_tangent_basis,
)
from .gipc_barrier import gipc_normal_force


@qd.func
def _tangent_displacement(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    basis: qd.template(),  # qd.Matrix
    weights: qd.template(),  # qd.Vector
    count: qd.template(),  # int
    output: qd.template(),  # qd.Vector
):
    for point in qd.static(range(count)):
        for tangent_axis in qd.static(range(2)):
            for axis in qd.static(range(3)):
                output[tangent_axis] += (
                    weights[point] * basis[axis, tangent_axis] * (current[point, axis] - lagged[point, axis])
                )


@qd.func
def gipc_friction_grad_hess_pt(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    d_hat,
    kappa,
    mu,
    eps_vh,
    xi,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    distance_squared = pt_distance2(
        lagged[0, 0],
        lagged[0, 1],
        lagged[0, 2],
        lagged[1, 0],
        lagged[1, 1],
        lagged[1, 2],
        lagged[2, 0],
        lagged[2, 1],
        lagged[2, 2],
        lagged[3, 0],
        lagged[3, 1],
        lagged[3, 2],
    )
    normal_force = gipc_normal_force(distance_squared, d_hat, xi, kappa)
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    beta = qd.Vector.zero(qd.f64, 2)
    pt_friction_tangent_basis(lagged, basis)
    pt_friction_closest_point(lagged, beta)
    weights = qd.Vector([1.0, -1.0 + beta[0] + beta[1], -beta[0], -beta[1]])
    friction_from_weights(
        current,
        lagged,
        basis,
        weights,
        4,
        mu,
        normal_force,
        eps_vh,
        gradient,
        hessian,
    )


@qd.func
def gipc_friction_energy_pt(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    d_hat,
    kappa,
    mu,
    eps_vh,
    xi,
):
    distance_squared = pt_distance2(
        lagged[0, 0],
        lagged[0, 1],
        lagged[0, 2],
        lagged[1, 0],
        lagged[1, 1],
        lagged[1, 2],
        lagged[2, 0],
        lagged[2, 1],
        lagged[2, 2],
        lagged[3, 0],
        lagged[3, 1],
        lagged[3, 2],
    )
    normal_force = gipc_normal_force(distance_squared, d_hat, xi, kappa)
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    beta = qd.Vector.zero(qd.f64, 2)
    pt_friction_tangent_basis(lagged, basis)
    pt_friction_closest_point(lagged, beta)
    weights = qd.Vector([1.0, -1.0 + beta[0] + beta[1], -beta[0], -beta[1]])
    displacement = qd.Vector.zero(qd.f64, 2)
    _tangent_displacement(current, lagged, basis, weights, 4, displacement)
    return friction_energy(mu, normal_force, eps_vh, displacement)


@qd.func
def gipc_friction_grad_hess_ee(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    d_hat,
    kappa,
    mu,
    eps_vh,
    xi,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    distance_squared = ee_distance2(
        lagged[0, 0],
        lagged[0, 1],
        lagged[0, 2],
        lagged[1, 0],
        lagged[1, 1],
        lagged[1, 2],
        lagged[2, 0],
        lagged[2, 1],
        lagged[2, 2],
        lagged[3, 0],
        lagged[3, 1],
        lagged[3, 2],
    )
    normal_force = gipc_normal_force(distance_squared, d_hat, xi, kappa)
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    gamma = qd.Vector.zero(qd.f64, 2)
    ee_friction_tangent_basis(lagged, basis)
    ee_friction_closest_point(lagged, gamma)
    weights = qd.Vector([1.0 - gamma[0], gamma[0], gamma[1] - 1.0, -gamma[1]])
    friction_from_weights(
        current,
        lagged,
        basis,
        weights,
        4,
        mu,
        normal_force,
        eps_vh,
        gradient,
        hessian,
    )


@qd.func
def gipc_friction_energy_ee(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    d_hat,
    kappa,
    mu,
    eps_vh,
    xi,
):
    distance_squared = ee_distance2(
        lagged[0, 0],
        lagged[0, 1],
        lagged[0, 2],
        lagged[1, 0],
        lagged[1, 1],
        lagged[1, 2],
        lagged[2, 0],
        lagged[2, 1],
        lagged[2, 2],
        lagged[3, 0],
        lagged[3, 1],
        lagged[3, 2],
    )
    normal_force = gipc_normal_force(distance_squared, d_hat, xi, kappa)
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    gamma = qd.Vector.zero(qd.f64, 2)
    ee_friction_tangent_basis(lagged, basis)
    ee_friction_closest_point(lagged, gamma)
    weights = qd.Vector([1.0 - gamma[0], gamma[0], gamma[1] - 1.0, -gamma[1]])
    displacement = qd.Vector.zero(qd.f64, 2)
    _tangent_displacement(current, lagged, basis, weights, 4, displacement)
    return friction_energy(mu, normal_force, eps_vh, displacement)


@qd.func
def gipc_friction_grad_hess_pe(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    d_hat,
    kappa,
    mu,
    eps_vh,
    xi,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    distance_squared = pe_distance2(
        lagged[0, 0],
        lagged[0, 1],
        lagged[0, 2],
        lagged[1, 0],
        lagged[1, 1],
        lagged[1, 2],
        lagged[2, 0],
        lagged[2, 1],
        lagged[2, 2],
    )
    normal_force = gipc_normal_force(distance_squared, d_hat, xi, kappa)
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    pe_friction_tangent_basis(lagged, basis)
    eta = pe_friction_closest_point(lagged)
    weights = qd.Vector([1.0, eta - 1.0, -eta])
    friction_from_weights(
        current,
        lagged,
        basis,
        weights,
        3,
        mu,
        normal_force,
        eps_vh,
        gradient,
        hessian,
    )


@qd.func
def gipc_friction_energy_pe(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    d_hat,
    kappa,
    mu,
    eps_vh,
    xi,
):
    distance_squared = pe_distance2(
        lagged[0, 0],
        lagged[0, 1],
        lagged[0, 2],
        lagged[1, 0],
        lagged[1, 1],
        lagged[1, 2],
        lagged[2, 0],
        lagged[2, 1],
        lagged[2, 2],
    )
    normal_force = gipc_normal_force(distance_squared, d_hat, xi, kappa)
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    pe_friction_tangent_basis(lagged, basis)
    eta = pe_friction_closest_point(lagged)
    weights = qd.Vector([1.0, eta - 1.0, -eta])
    displacement = qd.Vector.zero(qd.f64, 2)
    _tangent_displacement(current, lagged, basis, weights, 3, displacement)
    return friction_energy(mu, normal_force, eps_vh, displacement)


@qd.func
def gipc_friction_grad_hess_pp(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    d_hat,
    kappa,
    mu,
    eps_vh,
    xi,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    distance_squared = pp_distance2(
        lagged[0, 0],
        lagged[0, 1],
        lagged[0, 2],
        lagged[1, 0],
        lagged[1, 1],
        lagged[1, 2],
    )
    normal_force = gipc_normal_force(distance_squared, d_hat, xi, kappa)
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    pp_friction_tangent_basis(lagged, basis)
    weights = qd.Vector([1.0, -1.0])
    friction_from_weights(
        current,
        lagged,
        basis,
        weights,
        2,
        mu,
        normal_force,
        eps_vh,
        gradient,
        hessian,
    )


@qd.func
def gipc_friction_energy_pp(
    current: qd.template(),  # qd.Matrix
    lagged: qd.template(),  # qd.Matrix
    d_hat,
    kappa,
    mu,
    eps_vh,
    xi,
):
    distance_squared = pp_distance2(
        lagged[0, 0],
        lagged[0, 1],
        lagged[0, 2],
        lagged[1, 0],
        lagged[1, 1],
        lagged[1, 2],
    )
    normal_force = gipc_normal_force(distance_squared, d_hat, xi, kappa)
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    pp_friction_tangent_basis(lagged, basis)
    weights = qd.Vector([1.0, -1.0])
    displacement = qd.Vector.zero(qd.f64, 2)
    _tangent_displacement(current, lagged, basis, weights, 2, displacement)
    return friction_energy(mu, normal_force, eps_vh, displacement)
