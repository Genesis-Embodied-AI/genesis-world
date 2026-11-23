"""Pose constraint between a rigid contact proxy and the link it stands for, with its maps on SE(3).

A proxy is a massless copy of the pose of a link, the only pose the contact sees. Its pose (t_p, R_p) equals the pose
(t*, R*) of the link at convergence, through the constraint c = (t_p - t*, Log(R_p R*^T)), translation and rotation kept
apart. The increment of the proxy is its world-frame left twist (dt, dw), the rotation turning as Exp(dw) R_p. With
phi = Log(R_p R*^T), the linearized constraint is dc = A dg - B dq, A = diag(I, Jl(phi)^-1) and B the link Jacobian
whose rotation rows carry Jr(phi)^-1. The proxy then moves by S B dq + d_p, the tangent map S = A^-1 B (B with the link
Jacobian taken out) carrying the twist of the link to that of the proxy, and the particular step d_p = -A^-1 c removing
the constraint to first order. The normal map N = A^-1 lifts a slack of the constraint to a twist of the proxy, so that
C N = I.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu


@qd.func
def func_so3_left_jacobian(phi: qd.types.vector(3)):
    """Compute the left Jacobian Jl(phi) = I + (1 - cos t) / t^2 [phi] + (t - sin t) / t^3 [phi]^2 of SO(3), t = |phi|,
    by its Taylor series below t^2 = 1e-10."""
    theta_sq = phi.norm_sqr()
    hat = gu.qd_vec_to_skew(phi)
    a, b = gs.qd_float(0.5), gs.qd_float(1.0 / 6.0)
    if theta_sq >= 1e-10:
        theta = qd.sqrt(theta_sq)
        a = (1.0 - qd.cos(theta)) / theta_sq
        b = (theta - qd.sin(theta)) / (theta_sq * theta)
    return qd.Matrix.identity(gs.qd_float, 3) + a * hat + b * hat @ hat


@qd.func
def func_so3_left_jacobian_inverse(phi: qd.types.vector(3)):
    """Compute the inverse Jl(phi)^-1 = I - [phi] / 2 + (1 - (t / 2) cot(t / 2)) / t^2 [phi]^2 of the left Jacobian of
    SO(3), t = |phi|, by its Taylor series below t^2 = 1e-10. The right Jacobian is Jr(phi) = Jl(-phi)."""
    theta_sq = phi.norm_sqr()
    hat = gu.qd_vec_to_skew(phi)
    a = 1.0 / 12.0 + theta_sq / 720.0
    if theta_sq >= 1e-10:
        half = 0.5 * qd.sqrt(theta_sq)
        a = (1.0 - half * qd.cos(half) / qd.sin(half)) / theta_sq
    return qd.Matrix.identity(gs.qd_float, 3) - 0.5 * hat + a * hat @ hat


@qd.func
def func_proxy_constraint(
    proxy_pos: qd.types.vector(3),
    proxy_quat: qd.types.vector(4),
    link_pos: qd.types.vector(3),
    link_quat: qd.types.vector(4),
):
    """Compute the constraint c = (t_p - t*, Log(R_p R*^T)) between the pose of a proxy and that of its link, translation
    first, the rotation of angle at most pi."""
    translation = proxy_pos - link_pos
    rotation = gu.qd_quat_to_rotvec(gu.qd_quat_mul(proxy_quat, gu.qd_inv_quat(link_quat)))
    return qd.Vector(
        [translation[0], translation[1], translation[2], rotation[0], rotation[1], rotation[2]], dt=gs.qd_float
    )


@qd.func
def func_proxy_maps(constraint: qd.types.vector(6)):
    """
    Compute the maps of the pose constraint of a proxy (see the module docstring), the translation blocks of the tangent
    map S and of the normal map N being the identity.

    Returns the rotation block Jl(phi) Jr(phi)^-1 of the tangent map, the rotation block Jl(phi) of the normal map, and
    the particular step d_p = (-c_t, -Jl(phi) c_w).
    """
    phi = qd.Vector([constraint[3], constraint[4], constraint[5]], dt=gs.qd_float)
    left_jacobian = func_so3_left_jacobian(phi)
    tangent_rotation = left_jacobian @ func_so3_left_jacobian_inverse(-phi)
    particular_rotation = -(left_jacobian @ phi)
    particular = qd.Vector(
        [
            -constraint[0],
            -constraint[1],
            -constraint[2],
            particular_rotation[0],
            particular_rotation[1],
            particular_rotation[2],
        ],
        dt=gs.qd_float,
    )
    return tangent_rotation, left_jacobian, particular


@qd.func
def func_fk_defect_prefix_cap(h0: float, h_slack: float, curvature: float, limit: float) -> float:
    """
    Compute the largest step in [0, 1] along which the surface mismatch of a proxy with its link stays within `limit`.

    The mismatch along the step alpha is bounded by h0 + (h_slack - h0) alpha + curvature alpha^2 / 2, h0 the mismatch at
    the start and h_slack that of the slack of the constraint (zero outside the restoration). The bound being convex and
    starting below the limit, every shorter step also stays within it. The positive root takes the form that avoids
    cancellation for either sign of h_slack - h0.
    """
    delta = qd.max(0.0, limit - h0)
    b = h_slack - h0
    root = gs.qd_float(1.0)
    if curvature <= 0.0:
        if b > 0.0:
            root = delta / b
    else:
        sqrt_discriminant = qd.sqrt(qd.max(0.0, 2.0 * curvature * delta + b * b))
        if b >= 0.0:
            denominator = sqrt_discriminant + b
            if denominator > 0.0:
                root = 2.0 * delta / denominator
            else:
                root = 0.0
        else:
            root = (sqrt_discriminant - b) / curvature
    return qd.min(1.0, qd.max(root, 0.0))
