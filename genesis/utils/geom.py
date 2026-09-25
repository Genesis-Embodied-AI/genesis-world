import math
from typing import Literal

import numba as nb
import numpy as np
import quadrants as qd
import torch
import torch.nn.functional as F

import genesis as gs
from genesis.typing import Vec3FType

# ------------------------------------------------------------------------------------
# ------------------------------------- Quadrants ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def qd_i_cross_vec(vec):
    return qd.Vector([0.0, -vec[2], vec[1]], dt=gs.qd_float)


@qd.func
def qd_j_cross_vec(vec):
    return qd.Vector([vec[2], 0.0, -vec[0]], dt=gs.qd_float)


@qd.func
def qd_k_cross_vec(vec):
    return qd.Vector([-vec[1], vec[0], 0.0], dt=gs.qd_float)


@qd.func
def qd_transform_by_quat_fast(v, quat):
    """
    Rotate a vector by a quaternion that is already normalized.

    Prefer this over the general form whenever the quaternion is known to be a unit one, for exactness rather than for
    speed: the general form divides by the squared norm, which is a no-op for a unit quaternion in exact arithmetic
    yet leaves a component that should be untouched a few units in the last place off, by an amount that depends on
    the rotation. A body resting flat on a plane then reports a height that varies with its orientation about the
    plane's normal, and the sign of that variation decides whether it is in contact at all. The two cross products
    here contribute exactly zero to a component the rotation preserves, so it comes back bit for bit.

    Assumptions:
    - quat must be normalized
    """
    q_w, q_x, q_y, q_z = quat
    u = qd.Vector([q_x, q_y, q_z])
    t = 2.0 * u.cross(v)
    return v + q_w * t + u.cross(t)


@qd.func
def qd_transform_by_trans_quat_fast(pos, trans, quat):
    """
    Place a point by a translation and a quaternion that is already normalized. See qd_transform_by_quat_fast.
    """
    return qd_transform_by_quat_fast(pos, quat) + trans


@qd.func
def qd_xyz_to_quat(xyz):
    """
    Convert intrinsic x-y-z Euler angles to quaternion.
    """
    ai, aj, ak = 0.5 * xyz[2], -0.5 * xyz[1], 0.5 * xyz[0]
    si, sj, sk = qd.sin(ai), qd.sin(aj), qd.sin(ak)
    ci, cj, ck = qd.cos(ai), qd.cos(aj), qd.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = qd.Vector(
        [
            +cj * cc + sj * ss,
            +cj * cs - sj * sc,
            -cj * ss - sj * cc,
            +cj * sc - sj * cs,
        ],
        dt=gs.qd_float,
    )
    return quat


@qd.func
def qd_R_to_xyz(R, eps):
    """
    Convert a rotation matrix into intrinsic x-y-z Euler angles.
    """
    xyz = qd.Vector.zero(gs.qd_float, 3)

    cy = qd.sqrt(R[2, 2] ** 2 + R[1, 2] ** 2)
    if cy > eps:
        xyz[0] = -qd.atan2(R[1, 2], R[2, 2])
        xyz[1] = -qd.atan2(-R[0, 2], cy)
        xyz[2] = -qd.atan2(R[0, 1], R[0, 0])
    else:
        xyz[0] = 0.0
        xyz[1] = -qd.atan2(-R[0, 2], cy)
        xyz[2] = -qd.atan2(-R[1, 0], R[1, 1])
    return xyz


@qd.func
def qd_rotvec_to_R(rotvec, eps):
    R = qd.Matrix.identity(gs.qd_float, 3)

    angle = rotvec.norm()
    if angle > eps:
        c = qd.cos(angle)
        s = qd.sqrt(1.0 - c**2)
        t = 1.0 - c
        x, y, z = rotvec / angle

        R = qd.Matrix(
            [
                [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
            ],
            dt=gs.qd_float,
        )

    return R


@qd.func
def qd_rotvec_to_quat(rotvec, eps):
    quat = qd.Vector.zero(gs.qd_float, 4)

    # We need to use [norm_sqr] instead of [norm] to avoid nan gradients in the backward pass. Even when theta = 0,
    # the gradient of [norm] operation is computed and used (note that the gradient becomes NaN when theta = 0). This
    # is seemd to be a bug in Quadrants autodiff @TODO: change back after the bug is fixed.
    thetasq = rotvec.norm_sqr()
    if thetasq > (eps**2):
        theta = qd.sqrt(thetasq)
        theta_half = 0.5 * theta
        c, s = qd.cos(theta_half), qd.sin(theta_half)

        quat[0] = c
        xyz = s / theta * rotvec
        for i in qd.static(range(3)):
            quat[i + 1] = xyz[i]

        # First order quaternion normalization is accurate enough yet necessary
        quat *= 0.5 * (3.0 - quat.norm_sqr())
    else:
        quat[0] = 1.0

    return quat


@qd.func
def qd_rotvec_to_quat_grad_rotvec(rotvec, eps, out_grad):
    """Adjoint of qd_rotvec_to_quat(rotvec, eps) with respect to rotvec, given the upstream quaternion gradient.

    Differentiates the eps-regularized surrogate quat = (c, sinc * rotvec) with theta_reg = sqrt(|rotvec|^2 + eps^2),
    c = cos(theta_reg / 2) and sinc = sin(theta_reg / 2) / theta_reg. The surrogate deviates from the exact forward
    (which branches at |rotvec| = eps and applies a first-order renormalization) by O(eps^2) in the main branch and
    smooths over the constant-identity branch, trading exactness at the branch point for a gradient that stays
    finite and continuous through rotvec = 0. Chain rule through theta_reg gives, per component i:
        d(quat[0])/d(rotvec[i])   = -0.5 * sin(theta_reg / 2) * rotvec[i] / theta_reg
        d(quat[1+j])/d(rotvec[i]) = delta(i, j) * sinc + rotvec[j] * (0.5 * c - sinc) / theta_reg^2 * rotvec[i]
    """
    thetasq = rotvec.dot(rotvec)
    theta_reg = qd.sqrt(thetasq + eps * eps)
    theta_half = 0.5 * theta_reg
    sin_half = qd.sin(theta_half)
    cos_half = qd.cos(theta_half)
    sinc = sin_half / theta_reg
    d_sinc_d_theta = (0.5 * cos_half - sinc) / theta_reg

    out_grad_vec = gs.qd_vec3(out_grad[1], out_grad[2], out_grad[3])
    coeff = -0.5 * sin_half / theta_reg * out_grad[0] + d_sinc_d_theta / theta_reg * out_grad_vec.dot(rotvec)
    return coeff * rotvec + sinc * out_grad_vec


@qd.func
def qd_quat_to_R(quat, eps):
    """
    Converts quaternion to 3x3 rotation matrix.
    """
    R = qd.Matrix.identity(gs.qd_float, 3)

    d = quat.norm_sqr()
    if d > eps:
        s = 2.0 / d
        w, x, y, z = quat
        xs, ys, zs = x * s, y * s, z * s
        wx, wy, wz = w * xs, w * ys, w * zs
        xx, xy, xz = x * xs, x * ys, x * zs
        yy, yz, zz = y * ys, y * zs, z * zs

        R = qd.Matrix(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dt=gs.qd_float,
        )

    return R


@qd.func
def qd_quat_to_xyz(quat, eps):
    """
    Convert a quaternion into intrinsic x-y-z Euler angles.
    """
    roll = gs.qd_float(0.0)
    pitch = gs.qd_float(0.0)
    yaw = gs.qd_float(0.0)

    quat_norm_sqr = quat.norm_sqr()
    if quat_norm_sqr > eps:
        s = 2.0 / quat_norm_sqr
        q_w, q_x, q_y, q_z = quat
        q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
        q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
        q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
        q_yy, q_yz, q_zz = q_y * q_ys, q_y * q_zs, q_z * q_zs

        sinycosp = q_wz - q_xy
        cosycosp = 1.0 - (q_yy + q_zz)
        cosp = qd.sqrt(cosycosp**2 + sinycosp**2)

        pitch = qd.atan2(q_xz + q_wy, cosp)
        if cosp > eps:
            roll = qd.atan2(q_wx - q_yz, 1.0 - (q_xx + q_yy))
            yaw = qd.atan2(sinycosp, cosycosp)
        else:
            yaw = qd.atan2(q_wz + q_xy, 1.0 - (q_xx + q_zz))

    return qd.Vector([roll, pitch, yaw], dt=gs.qd_float)


@qd.func
def qd_quat_to_xyz_grad_quat(quat, eps, out_grad):
    """Adjoint of qd_quat_to_xyz(quat, eps) with respect to quat, given the upstream Euler-angle gradient.

    Mirrors the forward branch structure: each angle is an atan2 of quadratic forms u = s * P(quat) with
    s = 2 / |quat|^2, so du/dq = s * (dP/dq - 2 P q / |quat|^2) and datan2(y, x) = (x dy - y dx) / (x^2 + y^2).
    The gradient is zero in the degenerate |quat|^2 <= eps branch (constant forward), and the roll / yaw terms
    switch to the gimbal-lock forms when cosp <= eps, matching the forward exactly.
    """
    g_quat = qd.Vector.zero(gs.qd_float, 4)
    quat_norm_sqr = quat.norm_sqr()
    if quat_norm_sqr > eps:
        s = 2.0 / quat_norm_sqr
        q_w, q_x, q_y, q_z = quat

        # u = s * P with dP/dq per quadratic form; du/dq folds the ds/dq = -2 s q / |quat|^2 term
        p_siny = q_w * q_z - q_x * q_y
        p_cosy = q_y * q_y + q_z * q_z
        p_pitch = q_x * q_z + q_w * q_y
        p_roll_num = q_w * q_x - q_y * q_z
        p_roll_den = q_x * q_x + q_y * q_y
        p_gimbal_num = q_w * q_z + q_x * q_y
        p_gimbal_den = q_x * q_x + q_z * q_z
        u_siny = s * p_siny
        u_cosy = 1.0 - s * p_cosy
        u_pitch = s * p_pitch
        u_roll_num = s * p_roll_num
        u_roll_den = 1.0 - s * p_roll_den
        cosp = qd.sqrt(u_cosy * u_cosy + u_siny * u_siny)

        g_roll = out_grad[0]
        g_pitch = out_grad[1]
        g_yaw = out_grad[2]

        # pitch = atan2(u_pitch, cosp) with cosp = |(u_siny, u_cosy)|
        denom_pitch = u_pitch * u_pitch + cosp * cosp
        g_u_pitch = g_pitch * cosp / denom_pitch
        g_cosp = -g_pitch * u_pitch / denom_pitch
        g_u_siny = gs.qd_float(0.0)
        g_u_cosy = gs.qd_float(0.0)
        if cosp > eps:
            g_u_siny = g_cosp * u_siny / cosp
            g_u_cosy = g_cosp * u_cosy / cosp

        g_u_roll_num = gs.qd_float(0.0)
        g_u_roll_den = gs.qd_float(0.0)
        g_u_gimbal_num = gs.qd_float(0.0)
        g_u_gimbal_den = gs.qd_float(0.0)
        if cosp > eps:
            # roll = atan2(u_roll_num, u_roll_den); yaw = atan2(u_siny, u_cosy)
            denom_roll = u_roll_num * u_roll_num + u_roll_den * u_roll_den
            g_u_roll_num = g_roll * u_roll_den / denom_roll
            g_u_roll_den = -g_roll * u_roll_num / denom_roll
            denom_yaw = u_siny * u_siny + u_cosy * u_cosy
            g_u_siny = g_u_siny + g_yaw * u_cosy / denom_yaw
            g_u_cosy = g_u_cosy - g_yaw * u_siny / denom_yaw
        else:
            # Gimbal lock: roll = 0 and yaw = atan2(s * (wz + xy), 1 - s * (xx + zz))
            u_gimbal_num = s * p_gimbal_num
            u_gimbal_den = 1.0 - s * p_gimbal_den
            denom_gimbal = u_gimbal_num * u_gimbal_num + u_gimbal_den * u_gimbal_den
            g_u_gimbal_num = g_yaw * u_gimbal_den / denom_gimbal
            g_u_gimbal_den = -g_yaw * u_gimbal_num / denom_gimbal

        # Fold each u = s * P (or 1 - s * P) back to quat: du/dq_k = +-s * (dP/dq_k - 2 P q_k / |quat|^2)
        two_over_n = 2.0 / quat_norm_sqr
        d_p_siny = qd.Vector([q_z, -q_y, -q_x, q_w], dt=gs.qd_float)
        d_p_cosy = qd.Vector([0.0, 0.0, 2.0 * q_y, 2.0 * q_z], dt=gs.qd_float)
        d_p_pitch = qd.Vector([q_y, q_z, q_w, q_x], dt=gs.qd_float)
        d_p_roll_num = qd.Vector([q_x, q_w, -q_z, -q_y], dt=gs.qd_float)
        d_p_roll_den = qd.Vector([0.0, 2.0 * q_x, 2.0 * q_y, 0.0], dt=gs.qd_float)
        d_p_gimbal_num = qd.Vector([q_z, q_y, q_x, q_w], dt=gs.qd_float)
        d_p_gimbal_den = qd.Vector([0.0, 2.0 * q_x, 0.0, 2.0 * q_z], dt=gs.qd_float)
        for j in qd.static(range(4)):
            q_j = quat[j]
            g_quat[j] = (
                g_u_siny * s * (d_p_siny[j] - p_siny * two_over_n * q_j)
                - g_u_cosy * s * (d_p_cosy[j] - p_cosy * two_over_n * q_j)
                + g_u_pitch * s * (d_p_pitch[j] - p_pitch * two_over_n * q_j)
                + g_u_roll_num * s * (d_p_roll_num[j] - p_roll_num * two_over_n * q_j)
                - g_u_roll_den * s * (d_p_roll_den[j] - p_roll_den * two_over_n * q_j)
                + g_u_gimbal_num * s * (d_p_gimbal_num[j] - p_gimbal_num * two_over_n * q_j)
                - g_u_gimbal_den * s * (d_p_gimbal_den[j] - p_gimbal_den * two_over_n * q_j)
            )
    return g_quat


@qd.func
def qd_quat_to_rotvec(quat, eps):
    q_w, q_x, q_y, q_z = quat
    rotvec = qd.Vector([q_x, q_y, q_z], dt=gs.qd_float)

    s2 = rotvec.norm()
    if s2 > eps:
        angle = 2.0 * qd.atan2(s2, qd.abs(q_w))
        inv_sinc = angle / s2
        rotvec = (-1.0 if q_w < 0.0 else 1.0) * inv_sinc * rotvec

    return rotvec


@qd.func
def qd_trans_quat_to_T(trans, quat, eps):
    T = qd.Matrix.identity(gs.qd_float, 4)
    T[:3, :3] = qd_quat_to_R(quat, eps)
    T[:3, 3] = trans
    return T


@qd.func
def qd_inv_quat(quat):
    return qd.Vector([quat[0], -quat[1], -quat[2], -quat[3]], dt=gs.qd_float)


@qd.func
def qd_quat_mul_axis(q, axis):
    return qd.Vector(
        [
            -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
            +q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
            +q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
            +q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
        ]
    )


@qd.func
def qd_quat_mul(u, v):
    vu = u.outer_product(v)
    w = vu[0, 0] - vu[1, 1] - vu[2, 2] - vu[3, 3]
    x = vu[0, 1] + vu[1, 0] + vu[2, 3] - vu[3, 2]
    y = vu[0, 2] - vu[1, 3] + vu[2, 0] + vu[3, 1]
    z = vu[0, 3] + vu[1, 2] - vu[2, 1] + vu[3, 0]
    return qd.Vector([w, x, y, z], dt=gs.qd_float)


@qd.func
def qd_quat_mul_grad_lhs(u, v, out_grad):
    """Adjoint of qd_quat_mul(u, v) with respect to u, given the upstream gradient of its output."""
    return qd.Vector(
        [
            out_grad[0] * v[0] + out_grad[1] * v[1] + out_grad[2] * v[2] + out_grad[3] * v[3],
            -out_grad[0] * v[1] + out_grad[1] * v[0] - out_grad[2] * v[3] + out_grad[3] * v[2],
            -out_grad[0] * v[2] + out_grad[1] * v[3] + out_grad[2] * v[0] - out_grad[3] * v[1],
            -out_grad[0] * v[3] - out_grad[1] * v[2] + out_grad[2] * v[1] + out_grad[3] * v[0],
        ],
        dt=gs.qd_float,
    )


@qd.func
def qd_quat_mul_grad_rhs(u, v, out_grad):
    """Adjoint of qd_quat_mul(u, v) with respect to v, given the upstream gradient of its output."""
    return qd.Vector(
        [
            out_grad[0] * u[0] + out_grad[1] * u[1] + out_grad[2] * u[2] + out_grad[3] * u[3],
            -out_grad[0] * u[1] + out_grad[1] * u[0] + out_grad[2] * u[3] - out_grad[3] * u[2],
            -out_grad[0] * u[2] - out_grad[1] * u[3] + out_grad[2] * u[0] + out_grad[3] * u[1],
            -out_grad[0] * u[3] + out_grad[1] * u[2] - out_grad[2] * u[1] + out_grad[3] * u[0],
        ],
        dt=gs.qd_float,
    )


@qd.func
def qd_transform_quat_by_quat(v, u):
    """Transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    vec = qd_quat_mul(u, v)
    return vec.normalized()


@qd.func
def qd_transform_by_quat(v, quat):
    q_w, q_x, q_y, q_z = quat
    q_xx, q_xy, q_xz, q_wx = q_x * q_x, q_x * q_y, q_x * q_z, q_x * q_w
    q_yy, q_yz, q_wy = q_y * q_y, q_y * q_z, q_y * q_w
    q_zz, q_wz = q_z * q_z, q_z * q_w
    q_ww = q_w * q_w

    return qd.Vector(
        [
            v.x * (q_xx + q_ww - q_yy - q_zz) + v.y * (2.0 * q_xy - 2.0 * q_wz) + v.z * (2.0 * q_xz + 2.0 * q_wy),
            v.x * (2.0 * q_wz + 2.0 * q_xy) + v.y * (q_ww - q_xx + q_yy - q_zz) + v.z * (-2.0 * q_wx + 2.0 * q_yz),
            v.x * (-2.0 * q_wy + 2.0 * q_xz) + v.y * (2.0 * q_wx + 2.0 * q_yz) + v.z * (q_ww - q_xx - q_yy + q_zz),
        ],
        dt=gs.qd_float,
    ) / (q_ww + q_xx + q_yy + q_zz)


@qd.func
def qd_transform_by_quat_grad_quat(v, quat, out_grad):
    """Adjoint of qd_transform_by_quat(v, quat) with respect to quat, with v held constant.

    The forward is num(q) / |q|^2 where num is the quaternion sandwich q v q*. The full derivative applies the
    quotient rule: d(num/D) = dnum/D - num * dD/D^2 with D = |q|^2 and dD/dq = 2 q. Keeping only the numerator term
    (dnum) is correct only when q is a fixed unit quaternion; it is wrong whenever q is an optimization variable,
    even at unit length (the radial dD term is nonzero), so the denominator term must be included.
    """
    q_w, q_x, q_y, q_z = quat
    v_x, v_y, v_z = v

    d_out0_d_quat = 2.0 * qd.Vector(
        [
            q_w * v_x - q_z * v_y + q_y * v_z,
            q_x * v_x + q_y * v_y + q_z * v_z,
            -q_y * v_x + q_x * v_y + q_w * v_z,
            -q_z * v_x - q_w * v_y + q_x * v_z,
        ],
        dt=gs.qd_float,
    )
    d_out1_d_quat = 2.0 * qd.Vector(
        [
            q_z * v_x + q_w * v_y - q_x * v_z,
            q_y * v_x - q_x * v_y - q_w * v_z,
            q_x * v_x + q_y * v_y + q_z * v_z,
            q_w * v_x - q_z * v_y + q_y * v_z,
        ],
        dt=gs.qd_float,
    )
    d_out2_d_quat = 2.0 * qd.Vector(
        [
            -q_y * v_x + q_x * v_y + q_w * v_z,
            q_z * v_x + q_w * v_y - q_x * v_z,
            -q_w * v_x + q_z * v_y - q_y * v_z,
            q_x * v_x + q_y * v_y + q_z * v_z,
        ],
        dt=gs.qd_float,
    )
    d_num_d_quat = out_grad[0] * d_out0_d_quat + out_grad[1] * d_out1_d_quat + out_grad[2] * d_out2_d_quat

    # Quotient-rule denominator term: out = num / D with D = |q|^2, so d(out)/dq = dnum/dq / D - out * (2 q) / D.
    D = q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z
    out = qd_transform_by_quat(v, quat)
    return d_num_d_quat / D - (2.0 / D) * out_grad.dot(out) * quat


@qd.func
def qd_inv_transform_by_quat(v, quat):
    return qd_transform_by_quat(v, qd_inv_quat(quat))


@qd.func
def qd_transform_by_T(pos, T):
    return T[:3, :3] @ pos + T[:3, 3]


@qd.func
def qd_inv_transform_by_T(pos, T):
    return T[:3, :3].transpose() @ (pos - T[:3, 3])


@qd.func
def qd_transform_by_trans_quat(pos, trans, quat):
    return qd_transform_by_quat(pos, quat) + trans


@qd.func
def qd_inv_transform_by_trans_quat(pos, trans, quat):
    return qd_transform_by_quat(pos - trans, qd_inv_quat(quat))


@qd.func
def qd_transform_motion_by_trans_quat(m_ang, m_vel, trans, quat):
    quat_inv = qd_inv_quat(quat)
    ang = qd_transform_by_quat(m_ang, quat_inv)
    vel = qd_transform_by_quat(m_vel - trans.cross(m_ang), quat_inv)
    return ang, vel


@qd.func
def qd_inv_transform_motion_by_trans_quat(m_ang, m_vel, trans, quat):
    ang = qd_transform_by_quat(m_ang, quat)
    vel = qd_transform_by_quat(m_vel, quat) + trans.cross(ang)
    return ang, vel


@qd.func
def qd_transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    new_pos = t_trans + qd_transform_by_quat(pos, t_quat)
    new_quat = qd_transform_quat_by_quat(quat, t_quat)
    return new_pos, new_quat


@qd.func
def qd_transform_inertia_by_trans_quat(i_inertial, i_mass, trans, quat, eps):
    x, y, z = trans.x, trans.y, trans.z
    xx, xy, xz, yy, yz, zz = x * x, x * y, x * z, y * y, y * z, z * z
    hhT = qd.Matrix(
        [
            [yy + zz, -xy, -xz],
            [-xy, xx + zz, -yz],
            [-xz, -yz, xx + yy],
        ]
    )

    R = qd_quat_to_R(quat, eps)
    i = R @ i_inertial @ R.transpose() + hhT * i_mass
    trans = trans * i_mass

    return i, trans, quat, i_mass


@qd.func
def qd_normalize(v, eps):
    # The guard floors the norm, not the radicand: under the square root it would compare against a squared length and
    # bias every vector shorter than its own square root. Flooring the norm leaves any vector longer than 'eps' exactly
    # unit and still keeps a vanishing one finite.
    return v / qd.max(v.norm(), eps)


@qd.func
def qd_identity_quat():
    return qd.Vector([1.0, 0.0, 0.0, 0.0], dt=gs.qd_float)


@qd.func
def qd_vec3(val):
    return qd.Vector([val, val, val], dt=gs.qd_float)


@qd.func
def qd_nowhere():
    # let's inject a bit of humor here
    return qd.Vector([-2333333, -6666666, -5201314], dt=gs.qd_float)


@qd.func
def qd_tet_vol(p0, p1, p2, p3):
    return (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0


@qd.func
def inertial_mul(pos, i, mass, vel, ang):
    _ang = i @ ang + pos.cross(vel)
    _vel = mass * vel - pos.cross(ang)
    return _ang, _vel


@qd.func
def motion_cross_force(m_ang, m_vel, f_ang, f_vel):
    vel = m_ang.cross(f_vel)
    ang = m_ang.cross(f_ang) + m_vel.cross(f_vel)
    return ang, vel


@qd.func
def motion_cross_motion(s_ang, s_vel, m_ang, m_vel):
    vel = s_ang.cross(m_vel) + s_vel.cross(m_ang)
    ang = s_ang.cross(m_ang)
    return ang, vel


@qd.func
def motion_cross_motion_grad(s_ang, s_vel, m_ang, m_vel, ang_grad, vel_grad):
    """Adjoint of motion_cross_motion, returning the additive gradient deltas (s_ang, s_vel, m_ang, m_vel).

    Uses the cross-product adjoint of f = a x b: a.grad += b x f.grad, b.grad += f.grad x a.
    """
    return (
        m_ang.cross(ang_grad) + m_vel.cross(vel_grad),
        m_ang.cross(vel_grad),
        ang_grad.cross(s_ang) + vel_grad.cross(s_vel),
        vel_grad.cross(s_ang),
    )


@qd.func
def qd_orthogonals(a):
    """
    Returns orthogonal vectors `b` and `c`, given a normal vector `a`.

    Note that `a` is assumed to be normalized.
    """
    b = qd.Vector.zero(gs.qd_float, 3)
    if qd.abs(a[1]) < 0.5:
        b[0] = -a[0] * a[1]
        b[1] = 1.0 - a[1] ** 2
        b[2] = -a[2] * a[1]
    else:
        b[0] = -a[0] * a[2]
        b[1] = -a[1] * a[2]
        b[2] = 1.0 - a[2] ** 2
    b = b.normalized()
    return b.cross(a), b


@qd.func
def imp_aref(params, neg_penetration, vel, pos):
    timeconst, dampratio, dmin, dmax, width, mid, power = params

    imp_x = qd.abs(neg_penetration) / width
    imp_a = (1.0 / mid ** (power - 1)) * imp_x**power
    imp_b = 1.0 - (1.0 / (1.0 - mid) ** (power - 1)) * (1.0 - imp_x) ** power
    imp_y = imp_a if imp_x < mid else imp_b

    imp = dmin + imp_y * (dmax - dmin)
    imp = qd.math.clamp(imp, dmin, dmax)
    imp = dmax if imp_x > 1.0 else imp

    b = 2.0 / (dmax * timeconst)
    k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

    aref = -b * vel - k * imp * pos

    return imp, aref


@qd.func
def imp_aref_grad(params, neg_penetration):
    """Replay of imp_aref's impedance together with its derivative w.r.t. the impedance driver.

    Returns (imp, b, k, d_imp_d_x) with x = |neg_penetration| / width: the shared factors every manual constraint
    reverse needs to chain gradients through imp_aref, whose outputs are aref = -b * vel - k * imp * pos and the
    impedance imp feeding diag / efc_D. d_imp_d_x is zero outside the smooth band (imp_raw clamped at dmin / dmax or
    x > 1), matching the piecewise-flat regions of the forward.
    """
    timeconst, dampratio, dmin, dmax, width, mid, power = params

    imp_x = qd.abs(neg_penetration) / width
    imp_a_coef = 1.0 / mid ** (power - 1.0)
    imp_b_coef = 1.0 / (1.0 - mid) ** (power - 1.0)
    imp_a = imp_a_coef * imp_x**power
    imp_b = 1.0 - imp_b_coef * (1.0 - imp_x) ** power
    imp_y = imp_a if imp_x < mid else imp_b
    imp_raw = dmin + imp_y * (dmax - dmin)
    imp = qd.math.clamp(imp_raw, dmin, dmax)
    imp = dmax if imp_x > 1.0 else imp

    b = 2.0 / (dmax * timeconst)
    k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

    d_imp_y_d_x = power * imp_a_coef * imp_x ** (power - 1.0)
    if imp_x >= mid:
        d_imp_y_d_x = power * imp_b_coef * (1.0 - imp_x) ** (power - 1.0)
    d_imp_d_x = gs.qd_float(0.0)
    if imp_raw > dmin and imp_raw < dmax and imp_x <= 1.0:
        d_imp_d_x = (dmax - dmin) * d_imp_y_d_x
    return imp, b, k, d_imp_d_x


# ------------------------------------------------------------------------------------
# -------------------------------- torch and numpy -----------------------------------
# ------------------------------------------------------------------------------------


def inv_quat(quat):
    if isinstance(quat, torch.Tensor):
        _quat = quat.clone()
        _quat[..., 1:].neg_()
    elif isinstance(quat, np.ndarray):
        _quat = quat.copy()
        _quat[..., 1:] *= -1
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")
    return _quat


def inv_T(T):
    if isinstance(T, torch.Tensor):
        T_inv = torch.zeros_like(T)
    elif isinstance(T, np.ndarray):
        T_inv = np.zeros_like(T)
    else:
        gs.raise_exception(f"the input must be torch.Tensor or np.ndarray. got: {type(T)=}")

    trans, R = T[..., :3, 3], T[..., :3, :3]
    T_inv[..., 3, 3] = 1.0
    T_inv[..., :3, 3] = -R.T @ trans
    T_inv[..., :3, :3] = R.T

    return T_inv


def normalize(x, eps: float = 1e-12):
    if isinstance(x, torch.Tensor):
        return x / torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).clamp(min=eps, max=None)
    elif isinstance(x, np.ndarray):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(x)=}")


def rot6d_to_R(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] http://arxiv.org/abs/1812.07035
    """
    if isinstance(d6, torch.Tensor):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    elif isinstance(d6, np.ndarray):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        dot = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2 = a2 - dot * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2, axis=-1)
        return np.stack((b1, b2, b3), axis=-2)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(d6)=}")


def R_to_rot6d(R):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        R: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] http://arxiv.org/abs/1812.07035
    """
    if isinstance(R, torch.Tensor):
        return R[..., :2, :].flatten(start_dim=-2)
    elif isinstance(R, np.ndarray):
        return R[..., :2, :].reshape((*R.shape[:-2], 6))
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


@nb.jit(nopython=True, cache=True)
def _np_xyz_to_quat(xyz: np.ndarray, rpy: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the (qw, qx, qy, qz) Quaternion representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param xyz: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert xyz.ndim >= 1
    if out is None:
        out_ = np.empty((*xyz.shape[:-1], 4), dtype=xyz.dtype)
    else:
        assert out.shape == (*xyz.shape[:-1], 4)
        out_ = out

    rpy2 = 0.5 * xyz
    roll2, pitch2, yaw2 = rpy2[..., 0], rpy2[..., 1], rpy2[..., 2]
    cosr, sinr = np.cos(roll2), np.sin(roll2)
    cosp, sinp = np.cos(pitch2), np.sin(pitch2)
    cosy, siny = np.cos(yaw2), np.sin(yaw2)
    sign = 1.0 if rpy else -1.0

    out_[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out_[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out_[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out_[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out_


@torch.jit.script
def _tc_xyz_to_quat(xyz: torch.Tensor, rpy: bool = False, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty(xyz.shape[:-1] + (4,), dtype=xyz.dtype, device=xyz.device)

    roll2, pitch2, yaw2 = (0.5 * xyz).unbind(-1)
    cosr, sinr = roll2.cos(), roll2.sin()
    cosp, sinp = pitch2.cos(), pitch2.sin()
    cosy, siny = yaw2.cos(), yaw2.sin()
    sign = 1.0 if rpy else -1.0

    out[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out


def xyz_to_quat(xyz, rpy=False, degrees=False):
    if isinstance(xyz, torch.Tensor):
        if degrees:
            xyz = torch.deg2rad(xyz)
        return _tc_xyz_to_quat(xyz, rpy)
    elif isinstance(xyz, np.ndarray):
        if degrees:
            xyz = np.deg2rad(xyz)
        return _np_xyz_to_quat(xyz, rpy)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(xyz)=}")


@nb.jit(nopython=True, cache=True)
def _np_quat_to_R(quat: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a batch of quaternions.

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated and
                returned, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype)
    else:
        assert out.shape == (*quat.shape[:-1], 3, 3)
        out_ = out

    s = 2.0 / np.sum(np.square(quat), -1)
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
    q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
    q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
    q_yy, q_yz = q_y * q_ys, q_y * q_zs
    q_zz = q_z * q_zs

    out_[..., 0, 0] = 1.0 - (q_yy + q_zz)
    out_[..., 0, 1] = q_xy - q_wz
    out_[..., 0, 2] = q_xz + q_wy
    out_[..., 1, 0] = q_xy + q_wz
    out_[..., 1, 1] = 1.0 - (q_xx + q_zz)
    out_[..., 1, 2] = q_yz - q_wx
    out_[..., 2, 0] = q_xz - q_wy
    out_[..., 2, 1] = q_yz + q_wx
    out_[..., 2, 2] = 1.0 - (q_xx + q_yy)

    return out_


@torch.jit.script
def _tc_quat_to_R(quat, out: torch.Tensor | None = None):
    if out is None:
        R = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
    else:
        assert out.shape == quat.shape[:-1] + (3, 3)
        R = out

    s = 2 / (quat**2).sum(dim=-1, keepdim=True)
    q_vec_s = s * quat[..., 1:]

    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    q_sx, q_sy, q_sz = q_vec_s[..., 0], q_vec_s[..., 1], q_vec_s[..., 2]
    q_wx, q_wy, q_wz = q_w * q_sx, q_w * q_sy, q_w * q_sz
    q_xx, q_xy, q_xz = q_x * q_sx, q_x * q_sy, q_x * q_sz
    q_yy, q_yz = q_y * q_sy, q_y * q_sz
    q_zz = q_z * q_sz

    R[..., 0, 0] = 1.0 - (q_yy + q_zz)
    R[..., 0, 1] = q_xy - q_wz
    R[..., 0, 2] = q_xz + q_wy
    R[..., 1, 0] = q_xy + q_wz
    R[..., 1, 1] = 1.0 - (q_xx + q_zz)
    R[..., 1, 2] = q_yz - q_wx
    R[..., 2, 0] = q_xz - q_wy
    R[..., 2, 1] = q_yz + q_wx
    R[..., 2, 2] = 1.0 - (q_xx + q_yy)

    return R


def quat_to_R(quat, *, out=None):
    # NOTE: Ignore zero-norm quaternion for efficiency

    if all(isinstance(e, torch.Tensor) for e in (quat, out) if e is not None):
        return _tc_quat_to_R(quat, out=out)
    elif all(isinstance(e, np.ndarray) for e in (quat, out) if e is not None):
        return _np_quat_to_R(quat, out=out)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


@nb.jit(nopython=True, cache=True)
def _np_quat_to_xyz(quat, rpy=False, out=None):
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a batch of quaternions.

    The Roll, Pitch and Yaw angles are guaranteed to be within range [-pi,pi], [-pi/2,pi/2], [-pi,pi], respectively.

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((*quat.shape[:-1], 3), dtype=quat.dtype)
    else:
        assert out.shape == (*quat.shape[:-1], 3)
        out_ = out

    # Flatten batch dimensions
    quat_2d = quat.reshape((-1, 4))
    out_2d = out_.reshape((-1, 3))

    s = 2.0 / np.sum(np.square(quat_2d), -1)
    q_w, q_x, q_y, q_z = quat_2d[:, 0], quat_2d[:, 1], quat_2d[:, 2], quat_2d[:, 3]
    q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
    q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
    q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
    q_yy, q_yz, q_zz = q_y * q_ys, q_y * q_zs, q_z * q_zs

    if rpy:
        sinp = q_wy - q_xz
        sinrcosp = q_wx + q_yz
        sinycosp = q_wz + q_xy
    else:
        sinp = q_xz + q_wy
        sinrcosp = q_wx - q_yz
        sinycosp = q_wz - q_xy
    cosr_cosp = 1.0 - (q_xx + q_yy)
    cosycosp = 1.0 - (q_yy + q_zz)
    cosp = np.sqrt(cosycosp**2 + sinycosp**2)

    out_2d[:, 0] = np.arctan2(sinrcosp, cosr_cosp)
    out_2d[:, 1] = np.arctan2(sinp, cosp)
    out_2d[:, 2] = np.arctan2(sinycosp, cosycosp)

    cosp_mask = cosp < gs.EPS
    if rpy:
        sinycosp_sinrsinpcosy = q_wz[cosp_mask] - q_xy[cosp_mask]
    else:
        sinycosp_sinrsinpcosy = q_wz[cosp_mask] + q_xy[cosp_mask]
    cospcosy_sinrsinpsiny = 1.0 - (q_xx[cosp_mask] + q_zz[cosp_mask])
    out_2d[cosp_mask, 0] = 0.0
    out_2d[cosp_mask, 2] = np.arctan2(sinycosp_sinrsinpcosy, cospcosy_sinrsinpsiny)

    return out_


@torch.jit.script
def _tc_quat_to_xyz(quat, eps: float, rpy: bool = False):
    xyz = torch.empty(quat.shape[:-1] + (3,), dtype=quat.dtype, device=quat.device)
    x, y, z = xyz[..., :1], xyz[..., 1:2], xyz[..., 2:]

    q_w, q_x, q_y, q_z = quat[..., :1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:]
    q_ww, q_wx, q_wy, q_wz = q_w * q_w, q_w * q_x, q_w * q_y, q_w * q_z
    q_xx, q_xy, q_xz = q_x * q_x, q_x * q_y, q_x * q_z
    q_yy, q_yz = q_y * q_y, q_y * q_z
    q_zz = q_z**2

    # Compute some intermediary quantities.
    # Numerical robustness of 'cos(pitch)' could be improved using 'hypot' implementation from Eigen:
    # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/MathFunctionsImpl.h#L149
    if rpy:
        sinp = q_wy - q_xz
        sinrcosp = q_wx + q_yz
        sinycosp = q_wz + q_xy
    else:
        sinp = q_xz + q_wy
        sinrcosp = q_wx - q_yz
        sinycosp = q_wz - q_xy
    cosrcosp = (q_ww - q_xx - q_yy + q_zz) / 2
    cosycosp = (q_ww + q_xx - q_yy - q_zz) / 2
    cosp = torch.sqrt(cosycosp**2 + sinycosp**2)

    # Roll (x-axis rotation)
    torch.atan2(sinrcosp, cosrcosp, out=x)

    # Pitch (y-axis rotation)
    torch.atan2(sinp, cosp, out=y)

    # Yaw (z-axis rotation)
    torch.atan2(sinycosp, cosycosp, out=z)

    # Special treatment of nearly singular rotations
    cosp_mask = cosp < eps
    if rpy:
        sinycosp_sinrsinpcosy = q_wz - q_xy
    else:
        sinycosp_sinrsinpcosy = q_wz + q_xy
    cospcosy_sinrsinpsiny = (q_ww - q_xx + q_yy - q_zz) / 2
    x.masked_fill_(cosp_mask, 0.0)
    torch.where(cosp_mask, torch.arctan2(sinycosp_sinrsinpcosy, cospcosy_sinrsinpsiny), z, out=z)

    return xyz


def quat_to_xyz(quat, rpy=False, degrees=False):
    if isinstance(quat, torch.Tensor):
        rpy = _tc_quat_to_xyz(quat, gs.EPS, rpy)
        if degrees:
            rpy = torch.rad2deg(rpy)
    elif isinstance(quat, np.ndarray):
        rpy = _np_quat_to_xyz(quat, rpy)
        if degrees:
            rpy = np.rad2deg(rpy)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")
    return rpy


@nb.jit(nopython=True, cache=True)
def _np_R_to_quat(R, out=None):
    assert R.ndim >= 2

    if out is None:
        out_ = np.empty((*R.shape[:-2], 4), dtype=R.dtype)
    else:
        assert out.shape == (*R.shape[:-2], 4)
        out_ = out

    for i in np.ndindex(R.shape[:-2]):
        R_i = R[i]
        quat_i = out_[i]

        if R_i[2, 2] < 0.0:
            if R_i[0, 0] > R_i[1, 1]:
                t = 1.0 + R_i[0, 0] - R_i[1, 1] - R_i[2, 2]
                quat_i[0] = R_i[2, 1] - R_i[1, 2]
                quat_i[1] = t
                quat_i[2] = R_i[1, 0] + R_i[0, 1]
                quat_i[3] = R_i[0, 2] + R_i[2, 0]
            else:
                t = 1.0 - R_i[0, 0] + R_i[1, 1] - R_i[2, 2]
                quat_i[0] = R_i[0, 2] - R_i[2, 0]
                quat_i[1] = R_i[1, 0] + R_i[0, 1]
                quat_i[2] = t
                quat_i[3] = R_i[2, 1] + R_i[1, 2]
        else:
            if R_i[0, 0] < -R_i[1, 1]:
                t = 1.0 - R_i[0, 0] - R_i[1, 1] + R_i[2, 2]
                quat_i[0] = R_i[1, 0] - R_i[0, 1]
                quat_i[1] = R_i[0, 2] + R_i[2, 0]
                quat_i[2] = R_i[2, 1] + R_i[1, 2]
                quat_i[3] = t
            else:
                t = 1.0 + R_i[0, 0] + R_i[1, 1] + R_i[2, 2]
                quat_i[0] = t
                quat_i[1] = R_i[2, 1] - R_i[1, 2]
                quat_i[2] = R_i[0, 2] - R_i[2, 0]
                quat_i[3] = R_i[1, 0] - R_i[0, 1]
        quat_i /= 2.0 * np.sqrt(t)

    return out_


@torch.jit.script
def _tc_R_to_quat(R, out=None):
    if out is None:
        quat = torch.zeros(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)
    else:
        # assert out.shape == R.shape[:-2] + (4,)
        quat = out

    r11, r12, r13 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r21, r22, r23 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r31, r32, r33 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = r11 + r22 + r33
    c0 = trace > 0.0
    c1 = (r11 > r22) & (r11 > r33)
    c2 = r22 > r33
    s = torch.where(
        c0,
        torch.sqrt(trace + 1.0) * 2,
        torch.where(
            c1,
            torch.sqrt(1.0 + r11 - r22 - r33) * 2,
            torch.where(
                c2,
                torch.sqrt(1.0 + r22 - r11 - r33) * 2,
                torch.sqrt(1.0 + r33 - r11 - r22) * 2,
            ),
        ),
    )
    return torch.where(
        c0[..., None],
        torch.stack([0.25 * s, (r32 - r23) / s, (r13 - r31) / s, (r21 - r12) / s], dim=-1),
        torch.where(
            c1[..., None],
            torch.stack([(r32 - r23) / s, 0.25 * s, (r12 + r21) / s, (r13 + r31) / s], dim=-1),
            torch.where(
                c2[..., None],
                torch.stack([(r13 - r31) / s, (r12 + r21) / s, 0.25 * s, (r23 + r32) / s], dim=-1),
                torch.stack([(r21 - r12) / s, (r13 + r31) / s, (r23 + r32) / s, 0.25 * s], dim=-1),
            ),
        ),
    )

    return quat


def R_to_quat(R, *, out=None):
    if all(isinstance(e, torch.Tensor) for e in (R, out) if e is not None):
        return _tc_R_to_quat(R, out=out)
    elif all(isinstance(e, np.ndarray) for e in (R, out) if e is not None):
        return _np_R_to_quat(R, out=out)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


def R_to_xyz(R, rpy=False, degrees=False):
    return quat_to_xyz(R_to_quat(R), rpy=rpy, degrees=degrees)


def trans_R_to_T(trans=None, R=None, *, out=None):
    is_torch = all(isinstance(e, torch.Tensor) for e in (trans, R) if e is not None)
    is_numpy = not is_torch and all(isinstance(e, np.ndarray) for e in (trans, R) if e is not None)

    T = out
    B = () if trans is None and R is None else R.shape[:-2] if trans is None else trans.shape[:-1]
    dtype = R.dtype if trans is None else trans.dtype
    if is_torch:
        if T is None:
            device = R.device if trans is None else trans.device
            T = torch.zeros((*B, 4, 4), dtype=dtype, device=device)
    elif is_numpy:
        if T is None:
            T = np.zeros((*B, 4, 4), dtype=dtype)
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(R)=}")
    if B:
        assert T.shape == (*B, 4, 4)

    T[..., 3, 3] = 1.0
    if trans is not None:
        T[..., :3, 3] = trans
    if R is None:
        if is_torch:
            torch.diagonal(T, dim1=-2, dim2=-1).fill_(1.0)
        else:
            T[..., [0, 1, 2], [0, 1, 2]] = 1.0
    else:
        T[..., :3, :3] = R

    return T


def R_to_T(R, *, out=None):
    return trans_R_to_T(None, R, out=out)


def trans_to_T(trans, *, out=None):
    return trans_R_to_T(trans, None, out=out)


def trans_quat_to_T(trans=None, quat=None, *, out=None):
    is_torch = all(isinstance(e, torch.Tensor) for e in (trans, quat) if e is not None)
    is_numpy = not is_torch and all(isinstance(e, np.ndarray) for e in (trans, quat) if e is not None)

    T = out
    B = () if trans is None and quat is None else quat.shape[:-2] if trans is None else trans.shape[:-1]
    if is_torch:
        if T is None:
            T = torch.zeros((*B, 4, 4), dtype=trans.dtype, device=trans.device)
    elif is_numpy:
        dtype = np.result_type(trans, quat)
        trans, quat = trans.astype(dtype), quat.astype(dtype)
        if T is None:
            T = np.zeros((*B, 4, 4), dtype=dtype)
    else:
        gs.raise_exception(
            f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(quat)=}"
        )
    if B:
        assert T.shape == (*B, 4, 4)

    T[..., 3, 3] = 1.0
    if trans is not None:
        T[..., :3, 3] = trans
    if quat is not None:
        quat_to_R(quat, out=T[..., :3, :3])

    return T


def T_to_trans(T):
    return T[..., :3, 3]


def T_to_quat(T):
    return R_to_quat(T[..., :3, :3])


def T_to_trans_quat(T):
    return T_to_trans(T), T_to_quat(T)


@nb.jit(nopython=True, cache=True)
def _np_quat_mul(u, v, out=None):
    assert u.shape == v.shape
    u_2d = np.atleast_2d(u)
    v_2d = np.atleast_2d(v)

    w1, x1, y1, z1 = u_2d[..., 0], u_2d[..., 1], u_2d[..., 2], u_2d[..., 3]
    w2, x2, y2, z2 = v_2d[..., 0], v_2d[..., 1], v_2d[..., 2], v_2d[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    if out is None:
        # Use the dtype obtained by promoting the inputs (e.g. 'ww', a pure u/v product), not 'qq', whose float64
        # literal would otherwise upcast a float32 result.
        out_ = np.empty(u_2d.shape, dtype=ww.dtype)
    else:
        assert out.shape == u.shape
        out_ = out

    out_[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out_[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out_[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out_[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    out_ /= np.sqrt(np.sum(np.square(np.expand_dims(out_, -2)), -1))

    return out_.reshape(u.shape)


@torch.jit.script
def _tc_quat_mul(u, v):
    w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out = torch.empty(qq.shape + (4,), dtype=qq.dtype, device=qq.device)
    out[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    out /= torch.linalg.vector_norm(out, ord=2, dim=-1, keepdim=True)
    return out


def transform_quat_by_quat(v, u):
    """
    This method transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    assert u.ndim >= 1

    if all(isinstance(e, torch.Tensor) for e in (u, v)):
        quat = _tc_quat_mul(u, v)
    elif all(isinstance(e, np.ndarray) for e in (u, v)):
        quat = _np_quat_mul(u, v, out=None)
    else:
        gs.raise_exception(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")

    return quat


@nb.jit(nopython=True, cache=True)
def _np_transform_by_quat(v, quat, out=None):
    if out is None:
        out_ = np.empty(v.shape, dtype=v.dtype)
    else:
        assert out.shape == v.shape
        out_ = out

    v_T, quat_T, out_T = v.T, quat.T, out_.T
    v_x, v_y, v_z = v_T
    q_ww, q_wx, q_wy, q_wz = quat_T[0] * quat_T
    q_xx, q_xy, q_xz = quat_T[1] * quat_T[1:]
    q_yy, q_yz = quat_T[2] * quat_T[2:]
    q_zz = quat_T[3] * quat_T[3]

    out_T[0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
    out_T[1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
    out_T[2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

    out_T /= q_ww + q_xx + q_yy + q_zz

    return out_


@torch.jit.script
def _tc_transform_by_quat(v, quat, out: torch.Tensor | None = None):
    q_w, q_x, q_y, q_z = quat[..., :1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:]
    q_ww, q_wx, q_wy, q_wz = q_w * q_w, q_w * q_x, q_w * q_y, q_w * q_z
    q_xx, q_xy, q_xz = q_x * q_x, q_x * q_y, q_x * q_z
    q_yy, q_yz = q_y * q_y, q_y * q_z
    q_zz = q_z**2

    vs = v / (q_ww + q_xx + q_yy + q_zz)
    v_x, v_y, v_z = vs[..., :1], vs[..., 1:2], vs[..., 2:]

    if out is None:
        out = torch.empty(vs.shape, dtype=vs.dtype, device=vs.device)
    u_x, u_y, u_z = out[..., :1], out[..., 1:2], out[..., 2:]

    u_x.copy_(v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy))
    u_y.copy_(v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx))
    u_z.copy_(v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz))

    return out


def transform_by_quat(v, quat):
    """
    This method transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    assert v.ndim >= 1 and quat.ndim >= 1

    if all(isinstance(e, torch.Tensor) for e in (v, quat)):
        return _tc_transform_by_quat(v, quat)
    elif all(isinstance(e, np.ndarray) for e in (v, quat)):
        return _np_transform_by_quat(v, quat, out=None)
    else:
        gs.raise_exception(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")


def axis_angle_to_quat(angle, axis):
    if isinstance(angle, torch.Tensor) and isinstance(axis, torch.Tensor):
        theta = (0.5 * angle).unsqueeze(-1)
        xyz = normalize(axis) * theta.sin()
        w = theta.cos()
        return normalize(torch.cat([w, xyz], dim=-1))
    elif isinstance(angle, np.ndarray) and isinstance(axis, np.ndarray):
        theta = (0.5 * angle)[..., None]
        xyz = normalize(axis) * np.sin(theta)
        w = np.cos(theta)
        return normalize(np.concatenate([w, xyz], axis=-1))
    else:
        gs.raise_exception(
            f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(angle)=} and {type(axis)=}"
        )


def transform_by_xyz(pos, xyz):
    return transform_by_quat(pos, xyz_to_quat(xyz))


def transform_by_trans_quat(pos, trans, quat):
    return transform_by_quat(pos, quat) + trans


def inv_transform_by_quat(pos, quat):
    return transform_by_quat(pos, inv_quat(quat))


def inv_transform_by_trans_quat(pos, trans, quat):
    return inv_transform_by_quat(pos - trans, quat)


def transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    new_pos = t_trans + transform_by_quat(pos, t_quat)
    new_quat = transform_quat_by_quat(quat, t_quat)
    return new_pos, new_quat


def inv_transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    t_quat_inv = inv_quat(t_quat)
    new_pos = transform_by_quat(pos - t_trans, t_quat_inv)
    new_quat = transform_quat_by_quat(quat, t_quat_inv)
    return new_pos, new_quat


def transform_by_R(pos, R):
    """
    Transforms 3D points by a 3x3 rotation matrix or a batch of matrices, supporting both NumPy arrays and PyTorch
    tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point (3,), a batch of points (B, 3), or a batched
        batch of points (B, N, 3).
    R: np.ndarray | torch.Tensor
        The 3x3 rotation matrix or a batch of B rotation matrices of shape (B, 3, 3). Must be of the same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3 and R.shape[-2:] == (3, 3)
    assert R.ndim - pos.ndim in (0, 1)

    B = R.shape[:-2]
    N = pos.shape[-2] if pos.ndim == R.ndim else 1
    R = R.reshape((-1, 3, 3))
    pos_ = pos.reshape((math.prod(B), N, 3))

    if all(isinstance(e, torch.Tensor) for e in (pos, R) if e is not None):
        new_pos = torch.bmm(R, pos_.swapaxes(1, 2)).swapaxes(1, 2)
    elif all(isinstance(e, np.ndarray) for e in (pos, R) if e is not None):
        new_pos = np.matmul(R, pos_.swapaxes(1, 2)).swapaxes(1, 2)
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(pos)=} and {type(R)=}")

    new_pos = new_pos.reshape(pos.shape)

    return new_pos


def transform_by_trans_R(pos, trans, R):
    assert trans.shape[:-1] == R.shape[:-2]
    if trans.ndim < pos.ndim:
        trans = trans[..., None, :]
    return transform_by_R(pos, R) + trans


def transform_by_T(pos, T):
    """
    Transforms 3D points by a 4x4 transformation matrix or a batch of matrices, supporting both NumPy arrays and
    PyTorch tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point (3,), a batch of points (B, 3), or a
        batched batch of points (B, N, 3).
    T: np.ndarray | torch.Tensor
        The 4x4 transformation matrix or a batch of B transformation matrices of shape (B, 4, 4). Must be of the
        same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    return transform_by_trans_R(pos, T[..., :3, 3], T[..., :3, :3])


def inv_transform_by_T(pos, T):
    trans, R = T[..., :3, 3], T[..., :3, :3]

    R_inv = R.swapaxes(-1, -2)
    if pos.ndim == T.ndim:
        trans = trans.reshape((-1, 1, 3))

    return transform_by_R(pos - trans, R_inv)


def _tc_polar(A: torch.Tensor, pure_rotation: bool, side: Literal["right", "left"], eps: float):
    """Torch implementation of polar decomposition with batched support."""
    if A.ndim < 2:
        gs.raise_exception(f"Input must be at least 2D. got: {A.ndim=} dimensions")

    # Check if batched
    is_batched = A.ndim > 2
    M, N = A.shape[-2], A.shape[-1]

    # Perform SVD (supports batching automatically)
    U_svd, Sigma, Vt = torch.linalg.svd(A, full_matrices=False)

    # Normalize SVD signs for consistency: ensure the largest magnitude element in each column of U is positive
    # This resolves sign ambiguities that can differ between torch and numpy implementations
    if is_batched:
        # For batched case: max_indices shape is (*batch, N)
        max_indices = torch.argmax(torch.abs(U_svd), dim=-2)  # Shape: (*batch, N)
        # Use advanced indexing to get max values efficiently
        batch_dims = U_svd.shape[:-2]
        batch_size = math.prod(batch_dims) if batch_dims else 1
        U_flat = U_svd.reshape(batch_size, M, N)
        max_indices_flat = max_indices.reshape(batch_size, N)

        # Create batch indices for advanced indexing
        batch_idx = torch.arange(batch_size, device=U_svd.device).unsqueeze(1).expand(-1, N)  # (batch_size, N)
        col_idx = torch.arange(N, device=U_svd.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, N)
        max_vals = U_flat[batch_idx, max_indices_flat, col_idx]  # (batch_size, N)
        max_vals_abs = torch.abs(max_vals)
        signs = torch.where(max_vals_abs > eps, torch.sign(max_vals), torch.ones_like(max_vals))
        signs = signs.reshape(*batch_dims, N)
    else:
        # For single matrix case
        max_indices = torch.argmax(torch.abs(U_svd), dim=0)  # Shape: (N,)
        max_vals = U_svd[max_indices, torch.arange(N, device=U_svd.device)]  # (N,)
        max_vals_abs = torch.abs(max_vals)
        signs = torch.where(max_vals_abs > eps, torch.sign(max_vals), torch.ones_like(max_vals))

    U_svd = U_svd * signs.unsqueeze(-2) if is_batched else U_svd * signs
    Vt = Vt * signs.unsqueeze(-1) if is_batched else Vt * signs.unsqueeze(-1)

    # Handle pure_rotation: if det(U) < 0, flip signs to make it a pure rotation
    is_square = M == N
    if pure_rotation and is_square:  # Only for square matrices
        # Compute U first to check its determinant
        U_temp = U_svd @ Vt
        if is_batched:
            det_U = torch.linalg.det(U_temp)  # Shape: (*batch,)
            # Flip signs where det < 0
            flip_mask = det_U < 0
            if flip_mask.any():
                # Flip both the last column of U_svd and last row of Vt simultaneously
                U_svd[..., :, -1] = torch.where(flip_mask.unsqueeze(-1), -U_svd[..., :, -1], U_svd[..., :, -1])
                Vt[..., -1, :] = torch.where(flip_mask.unsqueeze(-1), -Vt[..., -1, :], Vt[..., -1, :])
        else:
            det_U = torch.linalg.det(U_temp)
            if det_U < 0:
                # Flip both the last column of U_svd and last row of Vt simultaneously
                U_svd[:, -1] *= -1
                Vt[-1, :] *= -1

    # Compute U
    U = U_svd @ Vt

    # Use absolute value to ensure P is positive semi-definite
    Sigma_abs = torch.abs(Sigma)

    if side == "right":
        # P = Vt.T @ diag(|Sigma|) @ Vt
        # For batched: Vt is (*batch, N, M), need (*batch, M, N) -> transpose last two dims
        # Create diagonal matrix using torch.diag_embed for batched support
        if is_batched:
            Sigma_diag = torch.diag_embed(Sigma_abs)  # Shape: (*batch, N, N)
            # Vt is (*batch, N, M), need Vt.T which is (*batch, M, N)
            Vt_T = Vt.transpose(-1, -2)  # Shape: (*batch, M, N)
            P = Vt_T @ Sigma_diag @ Vt  # Shape: (*batch, N, N)
        else:
            Sigma_diag = torch.diag(Sigma_abs)  # Shape: (N, N)
            P = Vt.T @ Sigma_diag @ Vt  # Shape: (N, N)
    else:  # "left"
        # P = U_svd @ diag(|Sigma|) @ U_svd.T (left polar: A = P @ U)
        if is_batched:
            Sigma_diag = torch.diag_embed(Sigma_abs)  # Shape: (*batch, M, M)
            U_svd_T = U_svd.transpose(-1, -2)  # Shape: (*batch, N, M)
            P = U_svd @ Sigma_diag @ U_svd_T  # Shape: (*batch, M, M)
        else:
            Sigma_diag = torch.diag(Sigma_abs)  # Shape: (M, M)
            P = U_svd @ Sigma_diag @ U_svd.T  # Shape: (M, M)

    return U, P


@nb.jit(nopython=True, cache=True)
def _np_polar_core_single(A, pure_rotation: bool, side_int: int):
    """
    Numba-accelerated core computation for polar decomposition of a single matrix.

    Parameters
    ----------
    A : np.ndarray
        The matrix to decompose. Must be a 2D matrix (M, N).
    pure_rotation : bool
        If True, ensure the unitary matrix U has det(U) = 1 (pure rotation).
    side_int : int
        0 for "right", 1 for "left".

    Returns
    -------
    U : np.ndarray
        Unitary matrix.
    P : np.ndarray
        Positive semi-definite matrix.
    """
    M, N = A.shape[0], A.shape[1]

    # Perform SVD
    U_svd, Sigma, Vt = np.linalg.svd(A, full_matrices=False)

    # Normalize SVD signs for consistency: ensure the largest magnitude element in each column of U is positive
    # This resolves sign ambiguities that can differ between torch and numpy implementations
    max_indices = np.argmax(np.abs(U_svd), axis=0)  # Shape: (N,)
    signs = np.empty(N, dtype=U_svd.dtype)
    for j in range(N):
        max_val = np.abs(U_svd[max_indices[j], j])
        if max_val > gs.EPS:
            signs[j] = np.sign(U_svd[max_indices[j], j])
        else:
            signs[j] = 1.0
    U_svd = U_svd * signs
    Vt = Vt * signs[:, None]

    # Handle pure_rotation: if det(U) < 0, flip signs to make it a pure rotation
    is_square = M == N
    if pure_rotation and is_square:  # Only for square matrices
        # Compute U first to check its determinant
        U_temp = U_svd @ Vt
        det_U = np.linalg.det(U_temp)
        if det_U < 0:
            # Flip both the last column of U_svd and last row of Vt simultaneously
            # This changes det(U) from -1 to 1 but maintains A = U_svd @ diag(Sigma) @ Vt
            # because the two sign flips cancel out in the product
            U_svd[:, -1] *= -1
            Vt[-1, :] *= -1

    # Compute U
    U = U_svd @ Vt

    # Use absolute value to ensure P is positive semi-definite
    Sigma_abs = np.abs(Sigma)

    if side_int == 0:  # "right"
        # P = Vt.T @ diag(|Sigma|) @ Vt
        # Create diagonal matrix manually for numba compatibility
        Sigma_diag = np.zeros((N, N), dtype=Sigma.dtype)
        for i in range(N):
            Sigma_diag[i, i] = Sigma_abs[i]
        P = Vt.T @ Sigma_diag @ Vt
    else:  # "left"
        # P = U_svd @ diag(|Sigma|) @ U_svd.T (left polar: A = P @ U)
        # Create diagonal matrix manually for numba compatibility
        Sigma_diag = np.zeros((M, M), dtype=Sigma.dtype)
        for i in range(M):
            Sigma_diag[i, i] = Sigma_abs[i]
        P = U_svd @ Sigma_diag @ U_svd.T

    return U, P


@nb.jit(nopython=True, cache=True)
def _np_polar_core_batched(A, pure_rotation: bool, side_int: int, U_out, P_out):
    """
    Numba-accelerated core computation for batched polar decomposition.

    Parameters
    ----------
    A : np.ndarray
        The batched matrices to decompose. Shape (*batch, M, N).
    pure_rotation : bool
        If True, ensure the unitary matrix U has det(U) = 1 (pure rotation).
    side_int : int
        0 for "right", 1 for "left".
    U_out : np.ndarray
        Output array for U, shape (*batch, M, N).
    P_out : np.ndarray
        Output array for P, shape (*batch, N, N) or (*batch, M, M) depending on side.

    Returns
    -------
    None (results written to U_out and P_out)
    """
    M, N = A.shape[-2], A.shape[-1]

    # Calculate batch size by flattening all batch dimensions
    batch_size = 1
    for i in range(A.ndim - 2):
        batch_size *= A.shape[i]

    # Flatten batch dimensions
    A_flat = A.reshape(batch_size, M, N)
    U_flat = U_out.reshape(batch_size, M, N)
    if side_int == 0:  # "right"
        P_flat = P_out.reshape(batch_size, N, N)
    else:  # "left"
        P_flat = P_out.reshape(batch_size, M, M)

    # Process each matrix in the batch
    for i in range(batch_size):
        U_i, P_i = _np_polar_core_single(A_flat[i], pure_rotation, side_int)
        U_flat[i] = U_i
        P_flat[i] = P_i


def _np_polar(A: np.ndarray, pure_rotation: bool, side: Literal["right", "left"]):
    """Numpy implementation of polar decomposition with numba acceleration and batched support."""
    if A.ndim < 2:
        gs.raise_exception(f"Input must be at least 2D. got: {A.ndim=} dimensions")

    # Convert side to int for numba compatibility
    side_int = 0 if side == "right" else 1

    # Check if batched
    is_batched = A.ndim > 2
    M, N = A.shape[-2], A.shape[-1]

    if is_batched:
        # Pre-allocate output arrays
        B = A.shape[:-2]
        U_out = np.empty((*B, M, N), dtype=A.dtype)
        if side == "right":
            P_out = np.empty((*B, N, N), dtype=A.dtype)
        else:
            P_out = np.empty((*B, M, M), dtype=A.dtype)

        # Call batched numba function
        _np_polar_core_batched(A, pure_rotation, side_int, U_out, P_out)
        return U_out, P_out
    else:
        # Call single matrix numba function
        return _np_polar_core_single(A, pure_rotation, side_int)


def polar(A, pure_rotation: bool = True, side: Literal["right", "left"] = "right"):
    """
    Compute the polar decomposition of a matrix or batch of matrices.

    Parameters
    ----------
    A : np.ndarray | torch.Tensor
        The matrix or batch of matrices to decompose. Can be:
        - Single matrix: shape (M, N)
        - Batched: shape (*batch, M, N)
    pure_rotation : bool, optional
        If True, ensure the unitary matrix U has det(U) = 1 (pure rotation).
        If False, U may have det(U) = -1 (contains reflection). Default is True.
    side : Literal['right', 'left'], optional
        The side of the decomposition. Either 'right' or 'left'. Default is 'right'.

    Returns
    -------
    tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]
        A tuple of (U, P) where:
        - U : The unitary matrix (rotation part), same shape as A (M, N) or (*batch, M, N).
        - P : The positive semi-definite matrix (scaling part). For 'right' decomposition,
          P has shape (N, N) or (*batch, N, N). For 'left' decomposition, P has shape (M, M) or (*batch, M, M).
    """
    if isinstance(A, np.ndarray):
        return _np_polar(A, pure_rotation, side)
    if isinstance(A, torch.Tensor):
        return _tc_polar(A, pure_rotation, side, gs.EPS)
    gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(A)=}")


@nb.jit(nopython=True, cache=True)
def _np_slerp(q0, q1, t):
    q0_norm = np.sqrt(np.sum(np.square(q0.reshape((-1, 4))), -1).reshape((*q0.shape[:-1], 1)))
    q0 = q0 / q0_norm
    q1_norm = np.sqrt(np.sum(np.square(q1.reshape((-1, 4))), -1).reshape((*q1.shape[:-1], 1)))
    q1 = q1 / q1_norm

    d = q0 * q1
    dot = np.sum(d.reshape((-1, 4)), -1).reshape((*d.shape[:-1], 1))
    dot_abs = np.abs(dot)
    t = t.reshape(dot.shape)

    theta = np.arccos(dot_abs)
    sin_theta_inv = 1.0 / np.sqrt(1.0 - dot_abs**2)

    is_theta_eps = dot_abs > 1.0 - gs.EPS
    s0 = np.where(is_theta_eps, 1.0 - t, np.sin((1.0 - t) * theta) * sin_theta_inv)
    s1 = np.where(is_theta_eps, t, np.sin(t * theta) * sin_theta_inv) * np.where(dot < 0.0, -1.0, 1.0)
    return s0 * q0 + s1 * q1


@torch.jit.script
def _tc_slerp(q0, q1, t, eps: float):
    q0 = q0 / torch.linalg.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dot_abs = dot.abs()
    t = t.reshape(dot.shape)

    theta = torch.acos(dot_abs)
    sin_theta_inv = 1.0 / torch.sqrt(1.0 - dot_abs**2)

    is_theta_eps = dot_abs > 1.0 - eps
    s0 = torch.where(is_theta_eps, 1.0 - t, torch.sin((1.0 - t) * theta) * sin_theta_inv)
    s1 = torch.where(is_theta_eps, t, torch.sin(t * theta) * sin_theta_inv) * torch.where(dot < 0.0, -1.0, 1.0)
    return s0 * q0 + s1 * q1


def slerp(q0, q1, t):
    """
    Perform spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q0 : numpy.array | torch.Tensor
        The start quaternion (w, x, y, z), can be batched.
    q1 : numpy.array | torch.Tensor
        The end quaternion (w, x, y, z), can be batched.
    t : numpy.array | torch.Tensor
        The interpolation parameter between 0 and 1.

    Returns
    -------
    numpy.array | torch.Tensor
        The interpolated quaternion (w, x, y, z).
    """
    if isinstance(q0, np.ndarray):
        return _np_slerp(q0, q1, t)
    if isinstance(q0, torch.Tensor):
        return _tc_slerp(q0, q1, torch.as_tensor(t, dtype=gs.tc_float, device=q0.device), gs.EPS)
    gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(q0)=}")


# ------------------------------------------------------------------------------------
# ------------------------------------- numpy ----------------------------------------
# ------------------------------------------------------------------------------------


def scale_to_T(scale):
    T = np.eye(4, dtype=scale.dtype)
    T[[0, 1, 2], [0, 1, 2]] = scale
    return T


@nb.jit(nopython=True, cache=True)
def _np_z_up_to_R(z, up=None, out=None):
    B = z.shape[:-1]
    if out is None:
        out_ = np.empty((*B, 3, 3), dtype=z.dtype)
    else:
        assert out.shape == (*B, 3, 3)
        out_ = out

    z_norm = np.sqrt(np.sum(np.square(z.reshape((-1, 3))), -1)).reshape(B)

    out_[..., 2] = z
    for i in np.ndindex(B):
        z_norm_i = z_norm[i]
        R = out_[i]
        x, y, z = R.T

        if z_norm_i > gs.EPS:
            z /= z_norm_i
        else:
            if up is None or abs(up[i][1]) < 0.5:
                z[:] = 0.0, 1.0, 0.0
            else:
                z[:] = 0.0, 0.0, 1.0

        if up is not None:
            x[:] = np.cross(up[i], z)
        else:
            if abs(z[2]) < 1.0 - gs.EPS:
                # up = (0.0, 0.0, 1.0)
                x[0] = z[1]
                x[1] = -z[0]
                x[2] = 0.0
            else:
                # up = (0.0, 1.0, 0.0)
                x[0] = z[2]
                x[1] = 0.0
                x[2] = -z[0]

        x_norm = np.linalg.norm(x)
        if x_norm > gs.EPS:
            x /= x_norm
            y[:] = np.cross(z, x)
        else:
            # This would only occurs if the user specified non-zero colinear z and up
            R[:] = np.eye(3, dtype=R.dtype)

    return out_


@torch.jit.script
def _tc_z_up_to_R(z, eps: float, up=None, out: torch.Tensor | None = None):
    if out is None:
        R = torch.empty(z.shape[:-1] + (3, 3), dtype=z.dtype, device=z.device)
    else:
        # assert out.shape == z.shape[:-1] + (3, 3)
        R = out

    # Set z as the third column of rotation matrix
    R[..., 2] = z

    # Handle batch dimension properly
    x, y, z = R[..., 0], R[..., 1], R[..., 2]

    # Normalize z vectors
    z_norm = torch.linalg.vector_norm(z, ord=2, dim=-1, keepdim=True)
    z /= z_norm.clamp(min=eps)

    # Handle zero norm cases
    zero_mask = z_norm < eps
    if up is None:
        torch.where(zero_mask, torch.tensor((0.0, 1.0, 0.0), device=z.device, dtype=z.dtype), z, out=z)
    else:
        up_mask = up[..., 1:2].abs() < 0.5
        torch.where(zero_mask & up_mask, torch.tensor((0.0, 1.0, 0.0), device=z.device, dtype=z.dtype), z, out=z)
        torch.where(zero_mask & ~up_mask, torch.tensor((0.0, 0.0, 1.0), device=z.device, dtype=z.dtype), z, out=z)

    # Compute x vectors (first column)
    if up is not None:
        x[:] = torch.cross(torch.broadcast_to(up, z.shape), z, dim=-1)
    else:
        up_mask = z[..., 2:].abs() < 1.0 - eps
        torch.where(up_mask, z[..., 1], z[..., 2], out=x[..., 0])
        torch.where(up_mask, -z[..., 0], 0.0, out=x[..., 1])
        torch.where(up_mask, 0.0, -z[..., 0], out=x[..., 2])

    # Normalize x vectors
    x_norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
    x /= x_norm.clamp(min=eps)

    # Handle zero x norm cases
    zero_x_mask = x_norm < eps
    # For zero x norm, set identity matrix
    torch.where(zero_x_mask[..., None], torch.eye(3, device=z.device, dtype=z.dtype), R, out=R)
    # Continue with non-zero cases
    torch.where(~zero_x_mask, torch.cross(z, x, dim=-1), y, out=y)

    return R


def z_up_to_R(z, up=None, out=None):
    if isinstance(z, torch.Tensor):
        return _tc_z_up_to_R(z, gs.EPS, up, out)
    else:
        return _np_z_up_to_R(z, up, out)


def pos_lookat_up_to_T(pos, lookat, up):
    if all(isinstance(e, torch.Tensor) for e in (pos, lookat, up) if e is not None):
        T = torch.zeros((*pos.shape[:-1], 4, 4), dtype=pos.dtype, device=pos.device)
        T[..., 3, 3] = 1.0
        T[..., :3, 3] = pos

        z = pos - lookat
        z_up_to_R(z, up=up, out=T[..., :3, :3])
        return T
    elif all(isinstance(e, np.ndarray) for e in (pos, lookat, up) if e is not None):
        T = np.zeros((*pos.shape[:-1], 4, 4), dtype=pos.dtype)
        T[..., 3, 3] = 1.0
        T[..., :3, 3] = pos

        z = pos - lookat
        z_up_to_R(z, up=up, out=T[..., :3, :3])
        return T
    else:
        gs.raise_exception(
            f"all of the inputs must be torch.Tensor or np.ndarray. got: {type(pos)=}, {type(lookat)=}, {type(up)=}"
        )


def T_to_pos_lookat_up(T):
    pos = T[..., :3, 3]
    lookat = T[..., :3, 3] - T[..., :3, 2]
    up = T[..., :3, 1]
    return pos, lookat, up


def euler_to_quat(euler_xyz):
    return xyz_to_quat(np.asarray(euler_xyz), rpy=True, degrees=True)


@nb.jit(nopython=True, cache=True)
def _np_euler_to_R(rpy: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose last dimension gathers the 3 Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated and
                returned, which is slower.
    """
    assert rpy.ndim >= 1
    if out is None:
        out_ = np.empty((*rpy.shape[:-1], 3, 3), dtype=rpy.dtype)
    else:
        assert out.shape == (*rpy.shape[:-1], 3, 3)
        out_ = out

    cos_rpy, sin_rpy = np.cos(rpy), np.sin(rpy)
    cos_roll, cos_pitch, cos_yaw = cos_rpy[..., 0], cos_rpy[..., 1], cos_rpy[..., 2]
    sin_roll, sin_pitch, sin_yaw = sin_rpy[..., 0], sin_rpy[..., 1], sin_rpy[..., 2]

    out_[..., 0, 0] = cos_pitch * cos_yaw
    out_[..., 0, 1] = -cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw
    out_[..., 0, 2] = sin_roll * sin_yaw + cos_roll * sin_pitch * cos_yaw
    out_[..., 1, 0] = cos_pitch * sin_yaw
    out_[..., 1, 1] = cos_roll * cos_yaw + sin_roll * sin_pitch * sin_yaw
    out_[..., 1, 2] = -sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw
    out_[..., 2, 0] = -sin_pitch
    out_[..., 2, 1] = sin_roll * cos_pitch
    out_[..., 2, 2] = cos_roll * cos_pitch

    return out_


def euler_to_R(euler_xyz):
    return _np_euler_to_R(np.deg2rad(euler_xyz))


@nb.jit(nopython=True, cache=True)
def quat_to_rotvec(quat: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the angle-axis representation of a single or a batch of quaternions (qw, qx, qy, qz).

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array into which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    assert quat.ndim >= 1
    B = quat.shape[:-1]
    if out is None:
        out_ = np.empty((*B, 3), dtype=quat.dtype)
    else:
        assert out.shape == (*B, 3)
        out_ = out

    # Split real (qw,) and imaginary (qx, qy, qz) quaternion parts
    q_w, q_vec = quat[..., :1], np.ascontiguousarray(quat[..., 1:])

    # Compute the angle-axis representation of the relative rotation
    s2 = np.sqrt(np.sum(np.square(q_vec.reshape((-1, 3))), -1)).reshape((*B, 1))
    angle = 2.0 * np.arctan2(s2, np.abs(q_w))
    # FIXME: Ideally, a taylor expansion should be used to handle angle ~ 0.0
    inv_sinc = angle / np.maximum(s2, gs.EPS)
    out_[:] = np.where(q_w < 0.0, -1.0, 1.0) * inv_sinc * q_vec

    if out is None:
        return out_
    return None


@nb.jit(nopython=True, cache=True)
def rotvec_to_quat(rotvec: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the quaternion representation (qw, qx, qy, qz) of a single or a batch of angle-axis vectors.

    :param rotvec: N-dimensional array whose last dimension gathers the 3 angle-axis components
                   angle * (ax, ay, az).
    :param out: Pre-allocated array into which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    assert rotvec.ndim >= 1
    B = rotvec.shape[:-1]
    if out is None:
        out_ = np.empty((*B, 4), dtype=rotvec.dtype)
    else:
        assert out.shape == (*B, 4)
        out_ = out

    # Split unit axis and positive angle
    angle = np.sqrt(np.sum(np.square(rotvec.reshape((-1, 3))), -1)).reshape(B)
    # FIXME: Taylor expansion should be used to handle angle ~ 0.0
    axis = rotvec / np.maximum(angle[..., None], gs.EPS)

    # Compute the quaternion representation
    out_[..., 0] = np.cos(0.5 * angle)
    out_[..., 1:] = np.sin(0.5 * angle[..., None]) * axis

    return out_


@nb.jit(nopython=True, cache=True)
def _np_axis_cos_angle_to_R(axis: np.ndarray, cos_theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if isinstance(cos_theta, (float, np.float32, np.float64)):
        assert axis.ndim == 1
    else:
        assert axis.ndim - 1 == cos_theta.ndim
    if out is None:
        out_ = np.empty((*axis.shape[:-1], 3, 3), dtype=axis.dtype)
    else:
        assert out.shape == (*axis.shape[:-1], 3, 3)
        out_ = out

    axis_norm = np.sqrt(np.sum(np.square(axis.reshape((-1, 3))), -1).reshape((*axis.shape[:-1], 1)))
    axis = axis / axis_norm
    if not isinstance(cos_theta, (float, np.float32, np.float64)):
        cos_theta = cos_theta[..., None]
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    cos1_axis = (1.0 - cos_theta) * axis
    sin_axis = sin_theta * axis

    tmp = cos1_axis[..., 0] * axis[..., 1]
    out_[..., 0, 1] = tmp - sin_axis[..., 2]
    out_[..., 1, 0] = tmp + sin_axis[..., 2]
    tmp = cos1_axis[..., 0] * axis[..., 2]
    out_[..., 0, 2] = tmp + sin_axis[..., 1]
    out_[..., 2, 0] = tmp - sin_axis[..., 1]
    tmp = cos1_axis[..., 1] * axis[..., 2]
    out_[..., 1, 2] = tmp - sin_axis[..., 0]
    out_[..., 2, 1] = tmp + sin_axis[..., 0]
    tmp = cos1_axis * axis + cos_theta
    out_[..., 0, 0] = tmp[..., 0]
    out_[..., 1, 1] = tmp[..., 1]
    out_[..., 2, 2] = tmp[..., 2]

    return out_


def axis_angle_to_R(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return _np_axis_cos_angle_to_R(axis, np.cos(theta))


def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    return axis_angle_to_R(rotvec, np.linalg.norm(rotvec, axis=-1))


@nb.jit(nopython=True, cache=True)
def R_to_rotvec(R: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the angle-axis representation of a single or a batch of 3D rotation matrices.

    :param R: N-dimensional array whose last 2 dimensions gathers individual 3D rotation matrices.
    :param out: Pre-allocated array into which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    # The quaternion path (Shepperd + arctan2) is preferred over direct Rodrigues formula theta from arccos of trace)
    # because the latter is catastrophically imprecise for near-identity rotations in float32 (the trace is ~3.0 and
    # arccos amplifies rounding).
    return quat_to_rotvec(_np_R_to_quat(R), out=out)


@nb.jit(nopython=True, cache=True)
def z_to_R(v_a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the "smallest" rotation transforming vector 'v_a' in 'e_z' as a rotation matrix.

    This operation is computed by rotating the world frame by moving the original z-axis to the given vector via the
    shortest path.
    """
    B = v_a.shape[:-1]

    v_a_norm = np.sqrt(np.sum(np.square(v_a.reshape((-1, 3))), -1).reshape((*B, 1)))
    v_a = v_a / v_a_norm

    axis = np.empty((*B, 3), dtype=v_a.dtype)
    cos_theta = np.empty(B, dtype=v_a.dtype)
    for i in np.ndindex(B):
        axis_i = axis[i]
        v_x, v_y, v_z = v_a[i]
        cos_theta[i] = v_z
        if abs(cos_theta[i]) < 1.0 - gs.EPS:
            axis_i[0] = -v_y
            axis_i[1] = v_x
            axis_i[2] = 0.0
        else:
            axis_i[:] = 0.0, 1.0, 0.0

    return _np_axis_cos_angle_to_R(axis, cos_theta, out)


@nb.jit(nopython=True, cache=True)
def z_to_quat(v_a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the "smallest" rotation transforming vector 'v_a' in 'e_z' as a quaternion (q_w, q_x, q_y, q_z).

    This method is (surprisingly) slower than `z_up_to_R`, w/ and w/o chaining `transform_by_(quat|R)`.
    """
    if out is None:
        out_ = np.empty((*v_a.shape[:-1], 4), dtype=v_a.dtype)
    else:
        assert out.shape == (*v_a.shape[:-1], 4)
        out_ = out

    v_a_norm = np.sqrt(np.sum(np.square(v_a.reshape((-1, 3))), -1).reshape((*v_a.shape[:-1], 1)))
    v_a = v_a / v_a_norm
    v_x, v_y, v_z = v_a[..., 0], v_a[..., 1], v_a[..., 2]

    for i in np.ndindex(v_a.shape[:-1]):
        v_x, v_y, v_z = v_a[i]
        quat = out_[i]

        if v_a_norm[i] < gs.EPS:
            quat[:] = 1.0, 0.0, 0.0, 0.0
        elif v_z > -1.0 + gs.EPS:
            s = np.sqrt(2.0 * (1.0 + v_z))
            s_inv = 1.0 / s
            quat[:] = 0.5 * s, -v_y * s_inv, v_x * s_inv, 0.0
        else:
            eps_thr = np.sqrt(gs.EPS)
            eps_x = abs(v_x) < gs.EPS
            eps_y = abs(v_y) < gs.EPS
            if not eps_y:
                ratio = v_x / v_y
                if eps_x:
                    esp_ratio = abs(ratio) < eps_thr
            elif eps_y and not eps_x:
                ratio = v_y / v_x
                esp_ratio = abs(ratio) < eps_thr
            w_2 = 0.5 * (1.0 + max(v_z, -1.0))
            quat[0] = np.sqrt(w_2)
            if eps_x and eps_y:
                # Both q_x and q_y would do fine. Picking q_y arbitrarily.
                quat[1] = 0.0
                quat[2] = 1.0
            elif esp_ratio:
                coef_abs = np.sqrt(1.0 - w_2) * (1.0 - 0.5 * ratio**2)
                if eps_x:
                    quat[1] = -np.sign(v_y) * coef_abs
                    quat[2] = -quat[1] * ratio
                else:
                    quat[2] = np.sign(v_x) * coef_abs
                    quat[1] = -quat[2] * ratio
            else:
                q_x_abs = np.sqrt((1.0 - w_2) / (1.0 + ratio**2))
                quat[1] = -np.sign(v_y) * q_x_abs
                quat[2] = +np.sign(v_x) * q_x_abs * ratio
            quat[3] = 0.0

            # First order quaternion normalization is accurate enough
            quat *= 0.5 * (3.0 - np.sum(np.square(quat), -1))

    return out_


def compute_camera_angle(camera_pos, camera_lookat):
    camera_dir = np.asarray(camera_lookat) - np.asarray(camera_pos)

    # rotation around vertical (y) axis
    angle_x = np.arctan2(-camera_dir[0], -camera_dir[2])

    # rotation w.r.t horizontal plane
    angle_y = np.arctan2(camera_dir[1], np.linalg.norm([camera_dir[0], camera_dir[2]]))

    angle_z = 0.0

    return np.array([angle_x, angle_y, angle_z])


def transform_inertia_by_T(inertia_tensor, T, mass):
    """
    Transform the inertia tensor to the new reference frame.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Parallel axis theorem
    translation_inertia = mass * (np.dot(t, t) * np.eye(3) - np.outer(t, t))

    return R @ inertia_tensor @ R.T + translation_inertia


def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    theta : torch.Tensor
        Horizontal angles in radians.
    phi : torch.Tensor
        Vertical angles in radians.

    Returns
    -------
    vectors : torch.Tensor
        Vectors in cartesian coordinates as tensor of shape (..., 3).
    """
    cos_phi = torch.cos(phi)

    x = torch.cos(theta) * cos_phi  # forward
    y = torch.sin(theta) * cos_phi  # left
    z = torch.sin(phi)  # up

    return torch.stack([x, y, z], dim=-1)


def random_quaternion(batch_size):
    # Generate three uniform random numbers for each quaternion in the batch
    u1, u2, u3 = np.random.rand(3, batch_size)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.stack((q1, q2, q3, q4), axis=1)


def generate_grid_points_on_plane(lo: Vec3FType, hi: Vec3FType, normal: Vec3FType, nx: int, ny: int) -> np.ndarray:
    """
    Build an nx-by-ny grid of points on the plane defined by the bounds and normal.

    Parameters
    ----------
    lo: array-like[float, float, float]
        Lower bound of the plane
    hi: array-like[float, float, float]
        Upper bound of the plane
    normal: array-like[float, float, float]
        Normal of the plane
    nx: int
        Number of grid points in x direction
    ny: int
        Number of grid points in y direction

    Returns
    -------
    grid: np.ndarray, shape (ny, nx, 3)
        Grid points on the plane
    """
    # Compute tangent axes
    normal = np.asarray(normal, dtype=gs.np_float)
    n_norm = np.linalg.norm(normal)
    if n_norm < gs.EPS:
        gs.raise_exception(f"normal must be non-zero, got: {normal}")
    normal = normal / n_norm
    t0, t1 = orthogonals(normal)

    # Compute lower and upper bounds in local basis
    rot = np.stack((t0, t1, normal), axis=0, dtype=gs.np_float)
    bounds = np.stack((lo, hi), axis=1, dtype=gs.np_float)
    (lo_u, hi_u), (lo_v, hi_v), (lo_w, hi_w) = rot @ bounds

    # Make sure that bounds are orthogonal to plane normal
    extent_w = abs(hi_w - lo_w)
    if extent_w > gs.EPS:
        gs.logger.warning(f"Bounds does not lie on a plane orthogonal to normal (normal-axis mismatch={extent_w:.6e}).")
    plane_w = 0.5 * (lo_w + hi_w)

    # Sample point grid on plane
    u_vals = np.linspace(lo_u, hi_u, num=nx, dtype=gs.np_float)
    v_vals = np.linspace(lo_v, hi_v, num=ny, dtype=gs.np_float)
    vv, uu = np.meshgrid(v_vals, u_vals, indexing="ij")
    grid = t0 * np.expand_dims(uu, axis=-1) + t1 * np.expand_dims(vv, axis=-1) + normal * plane_w

    return grid


def generate_ring_points_on_sphere(
    radius: float,
    cap_axis: tuple[float, float, float],
    n_rings: int,
    arc_spacing: float,
    return_normals: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Build ``n_rings`` concentric latitude rings on the upper spherical cap (pole at ``cap_axis``).

    Ring index 0 is the pole; ring ``n_rings - 1`` is the equatorial plane of the cap. The number of
    samples on each ring is chosen from the ring circumference and ``arc_spacing``.

    Parameters
    ----------
    radius : float
        Radius of points on the sphere, used to determine the number of points on each ring.
    cap_axis : (float, float, float)
        Unit-ish axis from sphere center toward the dome pole (probes lie on the cap around this axis).
    n_rings : int
        Number of latitude rings, including the pole as a degenerate ring.
    arc_spacing : float
        Target arc length between neighboring probes along each ring.
    return_normals : bool
        Whether to return the normal vectors of the points.

    Returns
    -------
    points : np.ndarray, shape (N, 3)
        Points on the sphere surface.
    normals : np.ndarray, shape (N, 3), optional
        Normal vectors of the points. Only returned if ``return_normals`` is True.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if n_rings < 1:
        raise ValueError(f"n_rings must be >= 1, got {n_rings}")
    if arc_spacing <= 0.0:
        raise ValueError(f"probe_arc_spacing must be positive, got {arc_spacing}")

    pole = np.asarray(cap_axis, dtype=gs.np_float)
    p_norm = float(np.linalg.norm(pole))
    if p_norm < gs.EPS:
        raise ValueError("cap_axis must be non-zero")
    pole = pole / p_norm
    t0, t1 = orthogonals(pole)

    pts: list[np.ndarray] = []
    denom = max(n_rings - 1, 1)
    for i in range(n_rings):
        theta = (i / denom) * (0.5 * np.pi)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        ring_r = radius * sin_t
        circ = 2.0 * np.pi * ring_r
        if ring_r <= radius * gs.EPS:
            n_pts = 1
        else:
            n_pts = max(3, int(np.ceil(circ / arc_spacing)))
        for j in range(n_pts):
            psi = (j / n_pts) * (2.0 * np.pi)
            direction = sin_t * (np.cos(psi) * t0 + np.sin(psi) * t1) + cos_t * pole
            pts.append(radius * direction)

    points = np.stack(pts, axis=0)
    if not return_normals:
        return points
    normals = points / radius
    return points, normals


# ------------------------------------------------------------------------------------
# ------------------------------------- misc ----------------------------------------
# ------------------------------------------------------------------------------------


def zero_pos():
    return np.zeros(3, dtype=gs.np_float)


def identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=gs.np_float)


def tc_zero_pos():
    return torch.zeros(3, dtype=gs.tc_float, device=gs.device)


def tc_identity_quat():
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)


def nowhere():
    # let's inject a bit of humor here
    return np.array([2333333, 6666666, 5201314], dtype=gs.np_float)


def default_solver_params():
    """
    Default constraint solver parameters ``(timeconst, dampratio, dmin, dmax, width, mid, power)``.

    The constraint reference acceleration is computed as ``aref = -b * vel - k * imp * pos``, where:

    - ``b = 2 / (dmax * timeconst)`` controls velocity damping (energy-dissipating). Depends only on ``timeconst``.
    - ``k = 1 / (dmax² * timeconst² * dampratio²)`` controls position stiffness (energy-conserving elastic spring).

    Despite its name, ``dampratio`` does **not** control damping — it scales the spring stiffness ``k``. Smaller
    ``dampratio`` yields a stiffer spring (more elastic bounce), while larger values yield a softer, more compliant
    contact. Setting ``dampratio`` to zero is forbidden as it removes the elastic restoring force entirely. Use a
    small positive value (e.g. ``0.01``) with a large ``timeconst`` (e.g. ``1.0``) for nearly undamped elastic contact.

    The impedance parameters ``(dmin, dmax, width, mid, power)`` control the depth-dependent constraint rigidity.

    Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    """
    return np.array([0.0, 1.0, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0])


def default_friction():
    return 1.0


def default_friction_torsional():
    return 0.005


def default_friction_rolling():
    return 0.0001


def default_dofs_kp(n=6):
    return np.full((n,), fill_value=100.0, dtype=gs.np_float)


def default_dofs_kv(n=6):
    return np.full((n,), fill_value=10.0, dtype=gs.np_float)


@qd.data_oriented
class SpatialHasher:
    def __init__(self, cell_size, grid_res, n_slots=None):
        self.cell_size = cell_size
        self.grid_res = grid_res

        if n_slots is None:
            self.n_slots = np.prod(grid_res)
        else:
            self.n_slots = n_slots

    def build(self, n_batch):
        self._B = n_batch
        # number of elements in each slot
        self.slot_size = qd.field(gs.qd_int, shape=(self.n_slots, self._B))
        # element index offset in each slot
        self.slot_start = qd.field(gs.qd_int, shape=(self.n_slots, self._B))
        self.cur_cnt = qd.field(gs.qd_int, shape=(self._B,))

    @qd.func
    def compute_reordered_idx(self, n, pos, active, reordered_idx):
        """
        Reordered element idx based on the given positions and active flags.

        Parameters:
            n (int)       : The number of elements in the positions and active arrays.
            pos           : The array of positions.
            active        : The array of active flags.
            reordered_idx : The array to store the computed reordered indices.

        Returns:
            None
        """

        self.slot_size.fill(0)
        self.slot_start.fill(0)
        self.cur_cnt.fill(0)

        for i_n, i_b in qd.ndrange(n, self._B):
            if active[i_n, i_b]:
                slot_idx = self.pos_to_slot(pos[i_n, i_b])
                qd.atomic_add(self.slot_size[slot_idx, i_b], 1)

        for i_n in range(self.n_slots):
            for i_b in range(self._B):
                self.slot_start[i_n, i_b] = qd.atomic_add(self.cur_cnt[i_b], self.slot_size[i_n, i_b])

        for i_n, i_b in qd.ndrange(n, self._B):
            if active[i_n, i_b]:
                slot_idx = self.pos_to_slot(pos[i_n, i_b])
                reordered_idx[i_n, i_b] = qd.atomic_add(self.slot_start[slot_idx, i_b], 1)

        # recover slot_start
        for i_s, i_b in qd.ndrange(self.n_slots, self._B):
            self.slot_start[i_s, i_b] -= self.slot_size[i_s, i_b]

    @qd.func
    def for_all_neighbors(self, i_p, pos, task_range, ret: qd.template(), task: qd.template(), i_b):
        """
        Iterates over all neighbors of a given position and performs a task on each neighbor.
        Elements are considered neighbors if they are within task_range.

        Parameters:
            i_p (int)  : Index of the querying particle.
            pos        : Template for the positions of all particles.
            task       : Template for the task to be performed on each neighbor of the querying particle.
            task_range : Range within which the task should be performed.
            ret        : Template for the return value of the task.

        Returns:
            None
        """
        base = self.pos_to_grid(pos[i_p, i_b])
        for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
            slot_idx = self.grid_to_slot(base + offset)
            for j_p in range(
                self.slot_start[slot_idx, i_b], self.slot_size[slot_idx, i_b] + self.slot_start[slot_idx, i_b]
            ):
                if i_p != j_p and (pos[i_p, i_b] - pos[j_p, i_b]).norm() < task_range:
                    task(i_p, j_p, ret, i_b)

    @qd.func
    def pos_to_grid(self, pos):
        return qd.floor(pos / self.cell_size, gs.qd_int)

    @qd.func
    def grid_to_pos(self, grid_id):
        return (grid_id + 0.5) * self.cell_size

    @qd.func
    def grid_to_slot(self, grid_id):
        return (
            grid_id[0] * self.grid_res[1] * self.grid_res[2] + grid_id[1] * self.grid_res[2] + grid_id[2]
        ) % self.n_slots

    @qd.func
    def pos_to_slot(self, pos):
        return self.grid_to_slot(self.pos_to_grid(pos))


@nb.jit(nopython=True, cache=True)
def cubic_spline_1d(x, y, xv):
    """
    Evaluate a 1D cubic spline at specified points.

    Constructs a cubic spline interpolation of the input data `(x, y)` and
    evaluates it at `xv`. The spline is C² continuous and uses **not-a-knot**
    boundary conditions, producing a smooth curve that passes through all data points.

    Parameters
    ----------
    x : array_like, shape (n,)
        Strictly increasing x-coordinates of the data points.
    y : array_like, shape (n,) or (n, m)
        Corresponding y-coordinates. Multiple columns can be provided for
        simultaneous interpolation of multiple datasets.
    xv : array_like
        Points at which to evaluate the spline.

    Returns
    -------
    yv : ndarray, shape (len(xv),) or (len(xv), m)
        Interpolated values at `xv`.
    """
    assert len(x) == len(y)
    y_2d = y[:, None] if y.ndim == 1 else y
    n, m = y_2d.shape
    h = np.diff(x)

    # Band storage: only store the non-zero diagonals
    # band[0] = 2nd lower diagonal (only element at (2, 0))
    # band[1] = 1st lower diagonal
    # band[2] = main diagonal
    # band[3] = 1st upper diagonal
    # band[4] = 2nd upper diagonal (only element at (n - 3, n - 1))
    band = np.zeros((5, n), dtype=gs.np_float)
    rhs = np.zeros((n, m), dtype=gs.np_float)

    # Not-a-knot boundary conditions
    band[2, 0], band[3, 0], band[4, 0] = h[1], -(h[0] + h[1]), h[0]
    band[0, n - 1], band[1, n - 1], band[2, n - 1] = h[-1], -(h[-2] + h[-1]), h[-2]

    # Interior points
    for i in range(1, n - 1):
        band[1, i], band[2, i], band[3, i] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 3 * ((y_2d[i + 1] - y_2d[i]) / h[i] - (y_2d[i] - y_2d[i - 1]) / h[i - 1])

    # Specialized Gaussian elimination with band storage
    for k in range(n - 1):
        pivot = band[2, k]
        if k + 1 < n:
            factor = band[1, k + 1] / pivot
            band[2, k + 1] -= factor * band[3, k]
            if k + 2 < n:
                band[3, k + 1] -= factor * band[4, k]
            rhs[k + 1] -= factor * rhs[k]
        if k == n - 3:
            factor = band[0, n - 1] / pivot
            band[1, n - 1] -= factor * band[3, k]
            band[2, n - 1] -= factor * band[4, k]
            rhs[n - 1] -= factor * rhs[k]

    # Back substitution
    c = np.zeros_like(rhs)
    c[n - 1] = rhs[n - 1] / band[2, n - 1]
    if n > 1:
        c[n - 2] = (rhs[n - 2] - band[3, n - 2] * c[n - 1]) / band[2, n - 2]
    for i in range(n - 3, -1, -1):
        c[i] = rhs[i] - band[3, i] * c[i + 1]
        if i + 2 < n and band[4, i] != 0:
            c[i] -= band[4, i] * c[i + 2]
        c[i] /= band[2, i]

    # Solve for b and d
    b = (y_2d[1:] - y_2d[:-1]) / h[:, None] - h[:, None] * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h[:, None])

    # Evaluate spline at xv
    xv = np.atleast_1d(xv)
    ix = np.clip(np.searchsorted(x[1:], xv), 0, n - 2)
    dx = (xv - x[ix])[:, None]
    return y_2d[ix] + b[ix] * dx + c[ix] * dx**2 + d[ix] * dx**3


@nb.jit(nopython=True, cache=True)
def orthogonals(a) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns orthogonal vectors `b` and `c`, given a normal vector `a`.

    Note that `a` is assumed to be normalized.
    """
    B = a.shape[:-1]

    a_1d = a.reshape((-1, 3))
    a_x, a_y, a_z = a_1d[:, 0], a_1d[:, 1], a_1d[:, 2]

    b1 = np.stack((-a_x * a_y, 1.0 - a_y**2, -a_z * a_y), axis=-1)
    b2 = np.stack((-a_x * a_z, -a_y * a_z, 1.0 - a_z**2), axis=-1)
    b_1d = np.where(np.abs(a_y).reshape((-1, 1)) < 0.5, b1, b2)
    b_1d /= np.sqrt(np.sum(np.square(b_1d), -1)).reshape((-1, 1))

    return np.cross(b_1d, a_1d).reshape((*B, 3)), b_1d.reshape((*B, 3))


@qd.func
def qd_hash01(k0, k1, k2, k3):
    """Counter-based hash to [0, 1): value depends only on the four keys, never on thread scheduling, so kernels
    drawing from it stay bitwise deterministic under any parallel execution."""
    h = qd.cast(k0, qd.u32) * qd.u32(0x9E3779B1)
    h = (h ^ qd.cast(k1, qd.u32)) * qd.u32(0x85EBCA77)
    h = (h ^ qd.cast(k2, qd.u32)) * qd.u32(0xC2B2AE3D)
    h = (h ^ qd.cast(k3, qd.u32)) * qd.u32(0x27D4EB2F)
    h = h ^ (h >> qd.u32(15))
    h = h * qd.u32(0x2C1B3C6D)
    h = h ^ (h >> qd.u32(12))
    return qd.cast(h >> qd.u32(8), gs.qd_float) * (1.0 / 16777216.0)
