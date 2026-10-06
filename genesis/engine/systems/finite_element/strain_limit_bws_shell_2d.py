from __future__ import annotations

import quadrants as qd


@qd.func
def Ds3x2(x0: qd.template(), x1: qd.template(), x2: qd.template()):
    result = qd.Matrix.zero(qd.f64, 3, 2)
    for axis in qd.static(range(3)):
        result[axis, 0] = x1[axis] - x0[axis]
        result[axis, 1] = x2[axis] - x0[axis]
    return result


@qd.func
def F3x2(Ds: qd.template(), Dm_inv: qd.template()):
    return Ds @ Dm_inv


@qd.func
def dFdX(Dm_inv: qd.template()):
    result = qd.Matrix.zero(qd.f64, 6, 9)
    d0 = Dm_inv[0, 0]
    d1 = Dm_inv[1, 0]
    d2 = Dm_inv[0, 1]
    d3 = Dm_inv[1, 1]
    for axis in qd.static(range(3)):
        result[axis, axis] = -(d0 + d1)
        result[axis + 3, axis] = -(d2 + d3)
        result[axis, axis + 3] = d0
        result[axis + 3, axis + 3] = d2
        result[axis, axis + 6] = d1
        result[axis + 3, axis + 6] = d3
    return result


@qd.func
def E(F: qd.template(), stretchS, shearS, strainLimitMultiplier):
    u = qd.Vector([F[0, 0], F[1, 0], F[2, 0]])
    v = qd.Vector([F[0, 1], F[1, 1], F[2, 1]])
    I6 = u.dot(v)
    I5u = u.norm()
    I5v = v.norm()
    stretch = (I5u - 1.0) ** 2 + (I5v - 1.0) ** 2
    if strainLimitMultiplier > 0.0:
        if I5u > 1.0:
            stretch = stretch + strainLimitMultiplier * (I5u - 1.0) ** 3
        if I5v > 1.0:
            stretch = stretch + strainLimitMultiplier * (I5v - 1.0) ** 3
    return stretchS * stretch + shearS * I6 * I6


@qd.func
def dEdF(F: qd.template(), stretchS, shearS, strainLimitMultiplier):
    u = qd.Vector([F[0, 0], F[1, 0], F[2, 0]])
    v = qd.Vector([F[0, 1], F[1, 1], F[2, 1]])
    I6 = u.dot(v)
    norm_u = u.norm()
    norm_v = v.norm()
    result = qd.Matrix.zero(qd.f64, 3, 2)
    if norm_u > 1e-30:
        coefficient = 1.0 - 1.0 / norm_u
        if strainLimitMultiplier > 0.0 and norm_u > 1.0:
            coefficient = coefficient + 1.5 * strainLimitMultiplier * (norm_u + 1.0 / norm_u - 2.0)
        for axis in qd.static(range(3)):
            result[axis, 0] = result[axis, 0] + stretchS * 2.0 * coefficient * u[axis]
    if norm_v > 1e-30:
        coefficient = 1.0 - 1.0 / norm_v
        if strainLimitMultiplier > 0.0 and norm_v > 1.0:
            coefficient = coefficient + 1.5 * strainLimitMultiplier * (norm_v + 1.0 / norm_v - 2.0)
        for axis in qd.static(range(3)):
            result[axis, 1] = result[axis, 1] + stretchS * 2.0 * coefficient * v[axis]
    for axis in qd.static(range(3)):
        result[axis, 0] = result[axis, 0] + shearS * 2.0 * I6 * v[axis]
        result[axis, 1] = result[axis, 1] + shearS * 2.0 * I6 * u[axis]
    return result


@qd.func
def ddEddF(F: qd.template(), stretchS, shearS, strainLimitMultiplier):
    u = qd.Vector([F[0, 0], F[1, 0], F[2, 0]])
    v = qd.Vector([F[0, 1], F[1, 1], F[2, 1]])
    norm_u = u.norm()
    norm_v = v.norm()
    I5u = u.dot(u)
    I5v = v.dot(v)

    stretch_hessian = qd.Matrix.zero(qd.f64, 6, 6)
    if norm_u > 1.0:
        value = 2.0 * (norm_u - 1.0) / norm_u
        if strainLimitMultiplier > 0.0:
            value = value + 3.0 * strainLimitMultiplier * (norm_u - 1.0) ** 2 / norm_u
        for axis in qd.static(range(3)):
            stretch_hessian[axis, axis] = value
    if norm_v > 1.0:
        value = 2.0 * (norm_v - 1.0) / norm_v
        if strainLimitMultiplier > 0.0:
            value = value + 3.0 * strainLimitMultiplier * (norm_v - 1.0) ** 2 / norm_v
        for axis in qd.static(range(3)):
            stretch_hessian[axis + 3, axis + 3] = value

    if norm_u > 1e-30:
        fu = u / norm_u
        coefficient = 2.0
        if norm_u > 1.0:
            coefficient = 2.0 / norm_u
            if strainLimitMultiplier > 0.0:
                coefficient = coefficient + 3.0 * (I5u - 1.0) * strainLimitMultiplier / norm_u
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                stretch_hessian[i, j] = stretch_hessian[i, j] + coefficient * fu[i] * fu[j]
    if norm_v > 1e-30:
        fv = v / norm_v
        coefficient = 2.0
        if norm_v > 1.0:
            coefficient = 2.0 / norm_v
            if strainLimitMultiplier > 0.0:
                coefficient = coefficient + 3.0 * (I5v - 1.0) * strainLimitMultiplier / norm_v
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                stretch_hessian[i + 3, j + 3] = stretch_hessian[i + 3, j + 3] + coefficient * fv[i] * fv[j]

    swap = qd.Matrix.zero(qd.f64, 6, 6)
    for axis in qd.static(range(3)):
        swap[axis, axis + 3] = 1.0
        swap[axis + 3, axis] = 1.0
    I6 = u.dot(v)
    sign_I6 = qd.f64(1.0)
    if I6 < 0.0:
        sign_I6 = qd.f64(-1.0)
    g = qd.Vector.zero(qd.f64, 6)
    for axis in qd.static(range(3)):
        g[axis] = v[axis]
        g[axis + 3] = u[axis]
    I2 = I5u + I5v
    lambda0 = 0.5 * (I2 + qd.sqrt(I2 * I2 + 12.0 * I6 * I6))
    q_raw = I6 * (swap @ g) + lambda0 * g
    q_norm = q_raw.norm()
    shear_hessian = qd.Matrix.zero(qd.f64, 6, 6)
    if q_norm > 1e-30:
        q = q_raw / q_norm
        projector = 0.5 * (qd.Matrix.identity(qd.f64, 6) + sign_I6 * swap)
        projected_q = projector @ q
        projected_norm_squared = projected_q.dot(projected_q)
        for i in qd.static(range(6)):
            for j in qd.static(range(6)):
                value = projector[i, j]
                if projected_norm_squared > 1e-30:
                    value = value - projected_q[i] * projected_q[j] / projected_norm_squared
                shear_hessian[i, j] = 2.0 * (qd.abs(I6) * value + lambda0 * q[i] * q[j])

    return stretchS * stretch_hessian + shearS * shear_hessian
