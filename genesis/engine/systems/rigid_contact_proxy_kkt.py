from __future__ import annotations

import quadrants as qd

from genesis.utils import geom as gu


@qd.func
def rigid_contact_proxy_skew(
    value: qd.template(),  # qd.Vector
):
    result = qd.Matrix.zero(qd.f64, 3, 3)
    result[0, 1] = -value[2]
    result[0, 2] = value[1]
    result[1, 0] = value[2]
    result[1, 2] = -value[0]
    result[2, 0] = -value[1]
    result[2, 1] = value[0]
    return result


@qd.func
def so3_left_jacobian_inverse(
    phi: qd.template(),  # qd.Vector
):
    theta_sq = phi.dot(phi)
    hat = rigid_contact_proxy_skew(phi)
    coefficient = qd.f64(0.0)
    if theta_sq < 1.0e-10:
        coefficient = 1.0 / 12.0 + theta_sq / 720.0
    else:
        theta = qd.sqrt(theta_sq)
        half = 0.5 * theta
        coefficient = (1.0 - half * qd.cos(half) / qd.sin(half)) / theta_sq
    return qd.Matrix.identity(qd.f64, 3) - 0.5 * hat + coefficient * (hat @ hat)


@qd.func
def so3_right_jacobian_inverse(
    phi: qd.template(),  # qd.Vector
):
    return so3_left_jacobian_inverse(-phi)


@qd.func
def so3_left_jacobian(
    phi: qd.template(),  # qd.Vector
):
    theta_sq = phi.dot(phi)
    hat = rigid_contact_proxy_skew(phi)
    result = qd.Matrix.identity(qd.f64, 3)
    if theta_sq < 1.0e-10:
        result = result + 0.5 * hat + (1.0 / 6.0) * (hat @ hat)
    else:
        theta = qd.sqrt(theta_sq)
        a = (1.0 - qd.cos(theta)) / theta_sq
        b = (theta - qd.sin(theta)) / (theta_sq * theta)
        result = result + a * hat + b * (hat @ hat)
    return result


@qd.func
def fk_defect_prefix_cap(h0, h_slack, curvature, limit):
    delta = qd.max(0.0, limit - h0)
    linear = h_slack - h0
    root = qd.f64(0.0)
    if curvature <= 0.0:
        if linear <= 0.0:
            root = 1.0
        else:
            root = delta / linear
    else:
        discriminant = 2.0 * curvature * delta + linear * linear
        sqrt_discriminant = qd.sqrt(qd.max(0.0, discriminant))
        if linear >= 0.0:
            denominator = sqrt_discriminant + linear
            if denominator > 0.0:
                root = 2.0 * delta / denominator
        else:
            root = (sqrt_discriminant - linear) / curvature
    if not (root > 0.0):
        root = 0.0
    return qd.min(1.0, root)


@qd.func
def rigid_contact_proxy_constraint(
    mechanism_position: qd.template(),  # qd.Vector
    mechanism_quaternion: qd.template(),  # qd.Vector
    proxy_position: qd.template(),  # qd.Vector
    proxy_quaternion: qd.template(),  # qd.Vector
    translation: qd.template(),  # qd.Vector
    rotation: qd.template(),  # qd.Vector
):
    relative_quaternion = gu.qd_quat_mul(proxy_quaternion, gu.qd_inv_quat(mechanism_quaternion))
    relative_rotation = gu.qd_quat_to_rotvec(relative_quaternion, qd.f64(1.0e-12))
    for axis in qd.static(range(3)):
        translation[axis] = proxy_position[axis] - mechanism_position[axis]
        rotation[axis] = relative_rotation[axis]


@qd.func
def rigid_contact_proxy_prepare_maps(
    translation: qd.template(),  # qd.Vector
    rotation: qd.template(),  # qd.Vector
    constraint: qd.template(),  # qd.Vector
    tangent_map: qd.template(),  # qd.Matrix
    normal_map: qd.template(),  # qd.Matrix
    particular: qd.template(),  # qd.Vector
):
    left_jacobian = so3_left_jacobian(rotation)
    right_jacobian_inverse = so3_right_jacobian_inverse(rotation)
    tangent_rotation = left_jacobian @ right_jacobian_inverse
    particular_rotation = -(left_jacobian @ rotation)

    for row in qd.static(range(6)):
        constraint[row] = 0.0
        particular[row] = 0.0
        for column in qd.static(range(6)):
            tangent_map[row, column] = 0.0
            normal_map[row, column] = 0.0

    for axis in qd.static(range(3)):
        constraint[axis] = translation[axis]
        constraint[axis + 3] = rotation[axis]
        particular[axis] = -translation[axis]
        particular[axis + 3] = particular_rotation[axis]
        tangent_map[axis, axis] = 1.0
        normal_map[axis, axis] = 1.0
        for column in qd.static(range(3)):
            tangent_map[axis + 3, column + 3] = tangent_rotation[axis, column]
            normal_map[axis + 3, column + 3] = left_jacobian[axis, column]


@qd.func
def expand_proxy_twist(
    tangent_map: qd.template(),  # qd.Matrix
    link_twist: qd.template(),  # qd.Vector
):
    return tangent_map @ link_twist


@qd.func
def restrict_proxy_wrench(
    tangent_map: qd.template(),  # qd.Matrix
    proxy_wrench: qd.template(),  # qd.Vector
):
    return tangent_map.transpose() @ proxy_wrench


@qd.func
def expand_slack_twist(
    normal_map: qd.template(),  # qd.Matrix
    slack: qd.template(),  # qd.Vector
):
    return normal_map @ slack


@qd.func
def restrict_slack_wrench(
    normal_map: qd.template(),  # qd.Matrix
    proxy_wrench: qd.template(),  # qd.Vector
):
    return normal_map.transpose() @ proxy_wrench
