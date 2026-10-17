# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Smooth penalty contact response of one sample point against a collider, in the collider frame: energy, force per
unit area and its derivative with respect to the sample position."""

import quadrants as qd

import genesis as gs

from .data import FRICTION_MODEL


@qd.func
def func_mat3_to_sym6(m):
    """Upper triangle of a symmetric 3x3 as (m00, m11, m22, m01, m02, m12): the contact tangent dforce and the
    accumulated D and S D S sums are symmetric, so six floats carry them exactly."""
    return qd.Vector([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]])


@qd.func
def func_sym6_to_mat3(s):
    """The symmetric 3x3 stored by func_mat3_to_sym6."""
    return qd.Matrix([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]])


@qd.func
def polyrelu(x, t, shift):
    """C2 smoothing of max(0, x - shift) over (shift - t, shift + t): value, first and second derivative."""
    x = x - shift
    f = qd.max(0.0, x)
    df = gs.qd_float(1.0) if x > 0.0 else gs.qd_float(0.0)
    ddf = gs.qd_float(0.0)
    if t > 0.0 and qd.abs(x) <= t:
        x_t = x / t
        x2_t2 = x_t * x_t
        x3_t3 = x2_t2 * x_t
        f = qd.max(0.0, (3.0 / 16.0) * t + (0.5 + (3.0 / 8.0) * x_t - (1.0 / 16.0) * x3_t3) * x)
        df = qd.min(qd.max(0.5 + 0.75 * x_t - 0.25 * x3_t3, 0.0), 1.0)
        ddf = qd.max(0.0, 0.75 * (1.0 - x2_t2) / t)
    return f, df, ddf


@qd.func
def ipc_step_c1(x, t):
    """C1 regularization of |x| with compact support t: value, derivative, second derivative and derivative over x."""
    f = x - t / 3.0
    df = gs.qd_float(1.0)
    ddf = gs.qd_float(0.0)
    df_x = gs.qd_float(0.0)
    if x > 0.0:
        df_x = 1.0 / x
    if x < t:
        x_t = x / t
        f = x * x_t * (1.0 - x_t / 3.0)
        ddf = 2.0 * (1.0 - x_t) / t
        df_x = (2.0 - x_t) / t
        df = x * df_x
    return f, df, ddf, df_x


@qd.func
def cinf_regularized(x, eps, eps_machine):
    """Smooth regularization of |x| without compact support, scaled so that its curvature at zero matches the C1
    regularization of the same width: value, derivative, second derivative and derivative over x."""
    eps_clamped = qd.max(eps, eps_machine)
    eps2 = eps_clamped * eps_clamped / 4.0
    s = qd.sqrt(x * x + eps2)
    inv_s = 1.0 / s
    f = s - eps_clamped / 2.0
    df = x * inv_s
    ddf = eps2 * inv_s * inv_s * inv_s
    return f, df, ddf, inv_s


@qd.func
def penalty_force_norm(d, mask, k, thr, h):
    """Magnitude of the normal penalty force per unit area at signed distance d."""
    penalty, dpenalty, _ = polyrelu(-d, h, h - thr)
    return mask * k * penalty * dpenalty


@qd.func
def collision_response(
    d,
    grad,
    n_colliding,
    p_rel,
    d_stage_start,
    k,
    h,
    thr,
    mu,
    falloff_vel,
    c_visc,
    c_ndamp,
    max_align,
    dt_stage,
    eps,
    mochi_config: qd.template(),
):
    """Contact response of one sample at signed distance d with distance gradient grad (collider frame), outward
    colliding normal n_colliding (collider frame), stage displacement p_rel and stage-start signed distance
    d_stage_start.

    Returns the energy per unit area, the force per unit area on the sample and its derivative with respect to the
    sample position (a symmetric negative semi-definite matrix), and the normal force magnitude driving dissipation.
    The derivative drops the curvature of the distance field and the dependency of the dissipation on the normal
    force, which is what keeps it negative semi-definite without projection.
    """
    # Contacts whose distance gradient and colliding normal point the same way would pull the bodies together (a
    # sample embedded past the far side of a thin collider), so they are disabled and dissipation fades out first.
    align = grad.dot(n_colliding)
    mask = gs.qd_float(1.0) if align <= max_align else gs.qd_float(0.0)
    fade = gs.qd_float(1.0)
    if qd.static(mochi_config.fade_friction):  # noqa: SIM102
        if max_align > -1.0:
            fade = (max_align - align) / (max_align + 1.0)

    penalty, dpenalty, ddpenalty = polyrelu(-d, h, h - thr)
    penalty *= mask
    dpenalty *= mask
    ddpenalty *= mask

    energy = 0.5 * k * penalty * penalty
    force_norm = k * penalty * dpenalty
    force = force_norm * grad
    dforce_norm = -k * (dpenalty * dpenalty + penalty * ddpenalty)
    dforce = dforce_norm * grad.outer_product(grad)

    # Normal force driving friction and damping: the elastic force at the current iterate, or the one recovered at
    # the start of the stage from the current gradient.
    f_n = fade * force_norm
    if qd.static(not mochi_config.implicit_normal_force_for_dissipation):
        f_n = fade * penalty_force_norm(d_stage_start, mask, k, thr, h)

    has_dissipation = mu > 0.0 or c_visc > 0.0 or c_ndamp > 0.0
    if has_dissipation and f_n > 0.0:
        normal = -n_colliding
        if qd.static(mochi_config.friction_with_collider_normal):
            grad_norm = grad.norm()
            if grad_norm > 100.0 * eps:
                normal = grad / grad_norm
        p_rel_n_scalar = p_rel.dot(normal)
        p_rel_n = p_rel_n_scalar * normal
        p_rel_t = p_rel - p_rel_n
        p_rel_t_norm = p_rel_t.norm()

        eig_p = gs.qd_float(0.0)
        eig_n = gs.qd_float(0.0)
        eig_delta_t = gs.qd_float(0.0)
        tangent = qd.Vector.zero(gs.qd_float, 3)
        energy_factor = gs.qd_float(0.0)

        if mu > 0.0:
            falloff = falloff_vel * dt_stage
            smoother = gs.qd_float(0.0)
            ddsmoother = gs.qd_float(0.0)
            dsmoother_x = gs.qd_float(0.0)
            if qd.static(mochi_config.friction_model == FRICTION_MODEL.CINF):
                smoother, _, ddsmoother, dsmoother_x = cinf_regularized(p_rel_t_norm, falloff, eps)
            else:
                smoother, _, ddsmoother, dsmoother_x = ipc_step_c1(p_rel_t_norm, falloff)
            energy_factor += mu * smoother
            tmp = mu * f_n * dsmoother_x
            force -= tmp * p_rel_t
            eig_p = -tmp
            if qd.static(not mochi_config.use_fitted_friction_hessian):  # noqa: SIM102
                if p_rel_t_norm > 1e-11:
                    eig_t = -mu * f_n * ddsmoother
                    tangent = p_rel_t / p_rel_t_norm
                    eig_delta_t = eig_t - eig_p

        if c_visc > 0.0:
            visc = c_visc / dt_stage
            energy_factor += 0.5 * visc * p_rel_t_norm * p_rel_t_norm
            tmp = visc * f_n
            force -= tmp * p_rel_t
            eig_p -= tmp

        if c_ndamp > 0.0:
            ndamp = c_ndamp / dt_stage
            energy_factor += 0.5 * ndamp * p_rel_n_scalar * p_rel_n_scalar
            tmp = ndamp * f_n
            force -= tmp * p_rel_n
            eig_n = -tmp

        dforce += eig_p * qd.Matrix.identity(gs.qd_float, 3) + (eig_n - eig_p) * normal.outer_product(normal)
        if qd.static(not mochi_config.use_fitted_friction_hessian):
            dforce += eig_delta_t * tangent.outer_product(tangent)
        energy += f_n * energy_factor

    return energy, force, dforce, f_n
