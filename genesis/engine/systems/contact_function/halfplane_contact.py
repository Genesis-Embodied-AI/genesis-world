from __future__ import annotations

import quadrants as qd

from .friction_utils import (
    _orthogonal_basis_from_normal,
    friction_energy,
    friction_gradient,
    friction_hessian,
)


@qd.func
def halfplane_signed_distance(
    xx: qd.f64,
    xy: qd.f64,
    xz: qd.f64,
    px: qd.f64,
    py: qd.f64,
    pz: qd.f64,
    nx: qd.f64,
    ny: qd.f64,
    nz: qd.f64,
) -> qd.f64:
    return nx * (xx - px) + ny * (xy - py) + nz * (xz - pz)


@qd.func
def halfplane_barrier_gradient(
    g_b: qd.f64,
    d: qd.f64,
    nx: qd.f64,
    ny: qd.f64,
    nz: qd.f64,
    out: qd.template(),  # qd.Vector
):
    scale = g_b * 2.0 * d
    out[0] = scale * nx
    out[1] = scale * ny
    out[2] = scale * nz


@qd.func
def halfplane_barrier_hessian(
    H_b: qd.f64,
    g_b: qd.f64,
    d_sq: qd.f64,
    nx: qd.f64,
    ny: qd.f64,
    nz: qd.f64,
    out: qd.template(),  # qd.Matrix
):
    coefficient = qd.max(4.0 * d_sq * H_b + 2.0 * g_b, 0.0)
    out[0] = coefficient * nx * nx
    out[1] = coefficient * nx * ny
    out[2] = coefficient * nx * nz
    out[3] = coefficient * ny * nx
    out[4] = coefficient * ny * ny
    out[5] = coefficient * ny * nz
    out[6] = coefficient * nz * nx
    out[7] = coefficient * nz * ny
    out[8] = coefficient * nz * nz


@qd.func
def halfplane_friction_tangent_basis(
    nx,
    ny,
    nz,
    basis: qd.template(),  # qd.Matrix
):
    _orthogonal_basis_from_normal(nx, ny, nz, basis)


@qd.func
def halfplane_friction_energy(
    x: qd.template(),  # qd.Vector
    lagged_x: qd.template(),  # qd.Vector
    nx,
    ny,
    nz,
    mu,
    normal_force,
    eps_vh,
):
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    halfplane_friction_tangent_basis(nx, ny, nz, basis)
    displacement = qd.Vector.zero(qd.f64, 2)
    for tangent_axis in qd.static(range(2)):
        for axis in qd.static(range(3)):
            displacement[tangent_axis] += basis[axis, tangent_axis] * (x[axis] - lagged_x[axis])
    return friction_energy(mu, normal_force, eps_vh, displacement)


@qd.func
def halfplane_friction_grad_hess(
    x: qd.template(),  # qd.Vector
    lagged_x: qd.template(),  # qd.Vector
    nx,
    ny,
    nz,
    mu,
    normal_force,
    eps_vh,
    gradient: qd.template(),  # qd.Vector
    hessian: qd.template(),  # qd.Matrix
):
    basis = qd.Matrix.zero(qd.f64, 3, 2)
    halfplane_friction_tangent_basis(nx, ny, nz, basis)
    displacement = qd.Vector.zero(qd.f64, 2)
    for tangent_axis in qd.static(range(2)):
        for axis in qd.static(range(3)):
            displacement[tangent_axis] += basis[axis, tangent_axis] * (x[axis] - lagged_x[axis])
    tangent_gradient = qd.Vector.zero(qd.f64, 2)
    tangent_hessian = qd.Vector.zero(qd.f64, 4)
    friction_gradient(tangent_gradient, mu, normal_force, eps_vh, displacement)
    friction_hessian(tangent_hessian, mu, normal_force, eps_vh, displacement)
    for row in qd.static(range(3)):
        gradient[row] = basis[row, 0] * tangent_gradient[0] + basis[row, 1] * tangent_gradient[1]
        for column in qd.static(range(3)):
            hessian[row * 3 + column] = basis[row, 0] * (
                tangent_hessian[0] * basis[column, 0] + tangent_hessian[1] * basis[column, 1]
            ) + basis[row, 1] * (tangent_hessian[2] * basis[column, 0] + tangent_hessian[3] * basis[column, 1])
