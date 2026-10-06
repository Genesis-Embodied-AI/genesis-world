from __future__ import annotations

import quadrants as qd


@qd.func
def f0(x2, epsvh):
    value = qd.sqrt(x2)
    if x2 < epsvh * epsvh:
        value = x2 * (-qd.sqrt(x2) / 3.0 + epsvh) / (epsvh * epsvh) + epsvh / 3.0
    return value


@qd.func
def f1_div_rel_dx_norm(x2, epsvh):
    value = 1.0 / qd.sqrt(x2)
    if x2 < epsvh * epsvh:
        value = (-qd.sqrt(x2) + 2.0 * epsvh) / (epsvh * epsvh)
    return value


@qd.func
def f2_term(epsvh):
    return -1.0 / (epsvh * epsvh)


@qd.func
def friction_energy(mu, normal_force, eps_vh, tangent_displacement: qd.template()):
    squared_norm = tangent_displacement[0] ** 2 + tangent_displacement[1] ** 2
    return mu * normal_force * f0(squared_norm, eps_vh)


@qd.func
def friction_gradient(
    output: qd.template(),
    mu,
    normal_force,
    eps_vh,
    tangent_displacement: qd.template(),
):
    squared_norm = tangent_displacement[0] ** 2 + tangent_displacement[1] ** 2
    scale = mu * normal_force * f1_div_rel_dx_norm(squared_norm, eps_vh)
    output[0] = scale * tangent_displacement[0]
    output[1] = scale * tangent_displacement[1]


@qd.func
def friction_hessian(
    output: qd.template(),
    mu,
    normal_force,
    eps_vh,
    tangent_displacement: qd.template(),
):
    u0 = tangent_displacement[0]
    u1 = tangent_displacement[1]
    squared_norm = u0 * u0 + u1 * u1
    eps_squared = eps_vh * eps_vh
    f1_div_norm = f1_div_rel_dx_norm(squared_norm, eps_vh)
    if squared_norm >= eps_squared:
        scale = mu * normal_force * f1_div_norm / squared_norm
        output[0] = scale * u1 * u1
        output[1] = -scale * u0 * u1
        output[2] = output[1]
        output[3] = scale * u0 * u0
    elif squared_norm == 0.0:
        scale = mu * normal_force * f1_div_norm
        output[0] = scale
        output[1] = 0.0
        output[2] = 0.0
        output[3] = scale
    else:
        relative_norm = qd.sqrt(squared_norm)
        outer_scale = f2_term(eps_vh) / relative_norm
        scale = mu * normal_force
        output[0] = scale * (outer_scale * u0 * u0 + f1_div_norm)
        output[1] = scale * outer_scale * u0 * u1
        output[2] = output[1]
        output[3] = scale * (outer_scale * u1 * u1 + f1_div_norm)


@qd.func
def _normalize3(x, y, z, output: qd.template()):
    inverse_norm = 1.0 / qd.sqrt(x * x + y * y + z * z)
    output[0] = x * inverse_norm
    output[1] = y * inverse_norm
    output[2] = z * inverse_norm


@qd.func
def _orthogonal_basis_from_normal(nx, ny, nz, basis: qd.template()):
    tangent_x = qd.f64(0.0)
    tangent_y = -nz
    tangent_z = ny
    if tangent_y * tangent_y + tangent_z * tangent_z <= nx * nx + nz * nz:
        tangent_x = nz
        tangent_y = 0.0
        tangent_z = -nx
    tangent = qd.Vector.zero(qd.f64, 3)
    _normalize3(tangent_x, tangent_y, tangent_z, tangent)
    basis[0, 0] = tangent[0]
    basis[1, 0] = tangent[1]
    basis[2, 0] = tangent[2]
    basis[0, 1] = ny * tangent[2] - nz * tangent[1]
    basis[1, 1] = nz * tangent[0] - nx * tangent[2]
    basis[2, 1] = nx * tangent[1] - ny * tangent[0]


@qd.func
def pt_friction_tangent_basis(
    positions: qd.template(),
    basis: qd.template(),
):
    tangent = qd.Vector.zero(qd.f64, 3)
    _normalize3(
        positions[2, 0] - positions[1, 0],
        positions[2, 1] - positions[1, 1],
        positions[2, 2] - positions[1, 2],
        tangent,
    )
    second_edge_x = positions[3, 0] - positions[1, 0]
    second_edge_y = positions[3, 1] - positions[1, 1]
    second_edge_z = positions[3, 2] - positions[1, 2]
    projection = tangent[0] * second_edge_x + tangent[1] * second_edge_y + tangent[2] * second_edge_z
    bitangent = qd.Vector.zero(qd.f64, 3)
    _normalize3(
        second_edge_x - projection * tangent[0],
        second_edge_y - projection * tangent[1],
        second_edge_z - projection * tangent[2],
        bitangent,
    )
    for axis in qd.static(range(3)):
        basis[axis, 0] = tangent[axis]
        basis[axis, 1] = bitangent[axis]


@qd.func
def pt_friction_closest_point(positions: qd.template(), beta: qd.template()):
    edge0 = qd.Vector(
        [
            positions[2, 0] - positions[1, 0],
            positions[2, 1] - positions[1, 1],
            positions[2, 2] - positions[1, 2],
        ]
    )
    edge1 = qd.Vector(
        [
            positions[3, 0] - positions[1, 0],
            positions[3, 1] - positions[1, 1],
            positions[3, 2] - positions[1, 2],
        ]
    )
    rhs = qd.Vector(
        [
            edge0.dot(
                qd.Vector(
                    [
                        positions[0, 0] - positions[1, 0],
                        positions[0, 1] - positions[1, 1],
                        positions[0, 2] - positions[1, 2],
                    ]
                )
            ),
            edge1.dot(
                qd.Vector(
                    [
                        positions[0, 0] - positions[1, 0],
                        positions[0, 1] - positions[1, 1],
                        positions[0, 2] - positions[1, 2],
                    ]
                )
            ),
        ]
    )
    a = edge0.dot(edge0)
    b = edge0.dot(edge1)
    c = edge1.dot(edge1)
    inverse_determinant = 1.0 / (a * c - b * b)
    beta[0] = (c * rhs[0] - b * rhs[1]) * inverse_determinant
    beta[1] = (-b * rhs[0] + a * rhs[1]) * inverse_determinant


@qd.func
def ee_friction_tangent_basis(positions: qd.template(), basis: qd.template()):
    tangent = qd.Vector.zero(qd.f64, 3)
    _normalize3(
        positions[1, 0] - positions[0, 0],
        positions[1, 1] - positions[0, 1],
        positions[1, 2] - positions[0, 2],
        tangent,
    )
    edge_b = qd.Vector(
        [
            positions[3, 0] - positions[2, 0],
            positions[3, 1] - positions[2, 1],
            positions[3, 2] - positions[2, 2],
        ]
    )
    projection = tangent.dot(edge_b)
    bitangent = qd.Vector.zero(qd.f64, 3)
    _normalize3(
        edge_b[0] - projection * tangent[0],
        edge_b[1] - projection * tangent[1],
        edge_b[2] - projection * tangent[2],
        bitangent,
    )
    for axis in qd.static(range(3)):
        basis[axis, 0] = tangent[axis]
        basis[axis, 1] = bitangent[axis]


@qd.func
def ee_friction_closest_point(positions: qd.template(), gamma: qd.template()):
    edge_a = qd.Vector(
        [
            positions[1, 0] - positions[0, 0],
            positions[1, 1] - positions[0, 1],
            positions[1, 2] - positions[0, 2],
        ]
    )
    edge_b = qd.Vector(
        [
            positions[3, 0] - positions[2, 0],
            positions[3, 1] - positions[2, 1],
            positions[3, 2] - positions[2, 2],
        ]
    )
    offset = qd.Vector(
        [
            positions[0, 0] - positions[2, 0],
            positions[0, 1] - positions[2, 1],
            positions[0, 2] - positions[2, 2],
        ]
    )
    a = edge_a.dot(edge_a)
    b = -edge_b.dot(edge_a)
    c = edge_b.dot(edge_b)
    rhs0 = -offset.dot(edge_a)
    rhs1 = offset.dot(edge_b)
    inverse_determinant = 1.0 / (a * c - b * b)
    gamma[0] = (c * rhs0 - b * rhs1) * inverse_determinant
    gamma[1] = (-b * rhs0 + a * rhs1) * inverse_determinant


@qd.func
def pe_friction_tangent_basis(positions: qd.template(), basis: qd.template()):
    tangent = qd.Vector.zero(qd.f64, 3)
    _normalize3(
        positions[2, 0] - positions[1, 0],
        positions[2, 1] - positions[1, 1],
        positions[2, 2] - positions[1, 2],
        tangent,
    )
    normal_x = tangent[1] * (positions[0, 2] - positions[1, 2]) - tangent[2] * (positions[0, 1] - positions[1, 1])
    normal_y = tangent[2] * (positions[0, 0] - positions[1, 0]) - tangent[0] * (positions[0, 2] - positions[1, 2])
    normal_z = tangent[0] * (positions[0, 1] - positions[1, 1]) - tangent[1] * (positions[0, 0] - positions[1, 0])
    bitangent = qd.Vector.zero(qd.f64, 3)
    _normalize3(normal_x, normal_y, normal_z, bitangent)
    for axis in qd.static(range(3)):
        basis[axis, 0] = tangent[axis]
        basis[axis, 1] = bitangent[axis]


@qd.func
def pe_friction_closest_point(positions: qd.template()):
    edge = qd.Vector(
        [
            positions[2, 0] - positions[1, 0],
            positions[2, 1] - positions[1, 1],
            positions[2, 2] - positions[1, 2],
        ]
    )
    point_offset = qd.Vector(
        [
            positions[0, 0] - positions[1, 0],
            positions[0, 1] - positions[1, 1],
            positions[0, 2] - positions[1, 2],
        ]
    )
    return point_offset.dot(edge) / edge.dot(edge)


@qd.func
def pp_friction_tangent_basis(positions: qd.template(), basis: qd.template()):
    direction = qd.Vector(
        [
            positions[1, 0] - positions[0, 0],
            positions[1, 1] - positions[0, 1],
            positions[1, 2] - positions[0, 2],
        ]
    )
    inverse_norm = 1.0 / qd.sqrt(direction.dot(direction))
    _orthogonal_basis_from_normal(
        direction[0] * inverse_norm,
        direction[1] * inverse_norm,
        direction[2] * inverse_norm,
        basis,
    )


@qd.func
def friction_from_weights(
    current: qd.template(),
    lagged: qd.template(),
    basis: qd.template(),
    weights: qd.template(),
    count: qd.template(),
    mu,
    normal_force,
    eps_vh,
    gradient: qd.template(),
    hessian: qd.template(),
):
    tangent_displacement = qd.Vector.zero(qd.f64, 2)
    for point in qd.static(range(count)):
        for tangent_axis in qd.static(range(2)):
            for axis in qd.static(range(3)):
                tangent_displacement[tangent_axis] += (
                    weights[point] * basis[axis, tangent_axis] * (current[point, axis] - lagged[point, axis])
                )

    tangent_gradient = qd.Vector.zero(qd.f64, 2)
    tangent_hessian = qd.Vector.zero(qd.f64, 4)
    friction_gradient(tangent_gradient, mu, normal_force, eps_vh, tangent_displacement)
    friction_hessian(tangent_hessian, mu, normal_force, eps_vh, tangent_displacement)
    for point in qd.static(range(count)):
        for axis in qd.static(range(3)):
            gradient[point * 3 + axis] = weights[point] * (
                basis[axis, 0] * tangent_gradient[0] + basis[axis, 1] * tangent_gradient[1]
            )
        for other in qd.static(range(count)):
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    hessian[(point * 3 + row) * (count * 3) + other * 3 + column] = (
                        weights[point]
                        * weights[other]
                        * (
                            basis[row, 0]
                            * (tangent_hessian[0] * basis[column, 0] + tangent_hessian[1] * basis[column, 1])
                            + basis[row, 1]
                            * (tangent_hessian[2] * basis[column, 0] + tangent_hessian[3] * basis[column, 1])
                        )
                    )
