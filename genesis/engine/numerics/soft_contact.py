"""Soft contact rows of an incremental potential: a one-sided penalty on the displacement of a contact point along a
direction, its stiffness and target given by the impedance of the soft constraints of the rigid solver."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu

# The rows of the friction pyramid of a contact: the normal tilted toward and away from each of two tangents
PYRAMID_ROWS = 4


@qd.func
def func_soft_row(sol_params: qd.types.vector(7), phi: float, vel: float, diag_coef: float, dt: float):
    """
    Return the stiffness and the target of a soft row of a contact point.

    The row stands for the soft constraint of the rigid solver of solver parameters `sol_params` at the gap `phi` and
    the normal velocity `vel`, `diag_coef` being the inverse weight of the point along the row. Its energy is
    1/2 stiffness (n . u - target)^2 while the displacement u of the point over the substep `dt` leaves the gap short of
    the target.
    """
    imp, aref = gu.imp_aref(sol_params, phi, vel, phi)
    diag = qd.max(diag_coef * (1.0 - imp) / imp, gs.EPS)
    return 1.0 / diag, dt * vel + dt * dt * aref


@qd.func
def func_band_row(sol_params: qd.types.vector(7), phi: float, vel: float, diag_coef: float, dt: float):
    """
    Return the stiffness and the target of the soft row of a contact point within the band of its partner.

    A point at a negative gap takes the soft row of its penetration (see 'func_soft_row'). A point at a positive gap
    takes the stiffness of the row at rest, the edge of the range, and the gap as target, so that it reaches the
    surface by the end of the substep at most.
    """
    # The row at rest is chosen at runtime, so that the soft row is inlined once
    phi_row = gs.qd_float(0.0)
    vel_row = gs.qd_float(0.0)
    if phi <= 0.0:
        phi_row = phi
        vel_row = vel
    stiffness, target = func_soft_row(sol_params, phi_row, vel_row, diag_coef, dt)
    if phi > 0.0:
        target = -phi
    return stiffness, target


@qd.func
def func_contact_band_width(approach: float, dt: float, dx: float):
    """Return the gap within which a contact point gets a row at the start of a substep, from its approach speed."""
    return 2.0 * dt * qd.max(approach, 0.0) + dx


@qd.func
def func_contact_sol_params(sol_params: qd.types.vector(7), dt: float):
    """Return the solver parameters of the rows of a contact, the time constant floored at twice the substep as the
    rigid solver floors those of its contacts."""
    sol_params_row = sol_params
    sol_params_row[0] = qd.max(sol_params[0], 2.0 * dt)
    return sol_params_row


@qd.func
def func_contact_friction(friction: float, friction_ratio: float, min_friction: float):
    """Return the friction coefficient of a contact, floored so that the rows of its friction pyramid keep a positive
    inverse weight (see 'func_pyramid_coefficient')."""
    return qd.max(friction * friction_ratio, min_friction)


@qd.func
def func_pyramid_coefficient(friction: float, impratio: float, invweight: float):
    """
    Return the inverse weight of the rows of the friction pyramid of a contact, regularized as the pyramidal contacts of
    the rigid solver.

    The two sides of the contact weigh `invweight` in all, and `impratio` is the ratio of the regularization.
    """
    friction_sq_reg = friction * friction / impratio
    return invweight * (1.0 + friction_sq_reg) * 2.0 * friction_sq_reg


@qd.func
def func_pyramid_directions(normal: qd.types.vector(3), friction: float):
    """Return the directions of the rows of the friction pyramid of a contact, as the rows of a matrix."""
    tangent_a, tangent_b = gu.qd_orthogonals(normal)
    return qd.Matrix.rows(
        [
            normal - friction * tangent_a,
            normal + friction * tangent_a,
            normal - friction * tangent_b,
            normal + friction * tangent_b,
        ]
    )


@qd.func
def func_pyramid_rows(
    normal: qd.types.vector(3),
    friction: float,
    diag_coef: float,
    sol_params: qd.types.vector(7),
    phi: float,
    vel_rel: qd.types.vector(3),
    vel_partner: qd.types.vector(3),
    dt: float,
):
    """
    Return the stiffness and the target of the rows of the friction pyramid of a contact point within the band of its
    partner (see 'func_band_row').

    The point lies at the gap `phi` and moves at the velocity `vel_rel` relative to its partner. The targets follow the
    partner over the substep at its velocity `vel_partner`, so that the rows act on the displacement of the point alone.
    """
    directions = func_pyramid_directions(normal, friction)
    stiffness = qd.Vector.zero(gs.qd_float, directions.n)
    target = qd.Vector.zero(gs.qd_float, directions.n)
    # The rows are looped at runtime, so that the soft row is inlined once
    for k in range(directions.n):
        direction = qd.Vector([directions[k, d] for d in qd.static(range(3))], dt=gs.qd_float)
        stiffness_k, target_k = func_band_row(sol_params, phi, direction.dot(vel_rel), diag_coef, dt)
        stiffness[k] = stiffness_k
        target[k] = target_k + dt * direction.dot(vel_partner)
    return stiffness, target


@qd.func
def func_contact_invweight(
    U: qd.types.matrix(3, 3), V: qd.types.matrix(3, 3), invweight_rest: qd.types.matrix(3, 3), n: qd.types.vector(3)
):
    """Return the inverse weight of a particle along a direction, its matrix at rest rotated along with the material by
    the polar rotation U V^T of its deformation gradient."""
    n_rest = V @ (U.transpose() @ n)
    return n_rest.dot(invweight_rest @ n_rest)


@qd.func
def func_soft_row_terms(direction: qd.types.vector(3), stiffness: float, target: float, u_rel: qd.types.vector(3)):
    """Return the energy, the gradient and the Hessian of a soft row in the displacement of its contact point."""
    energy = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    gap = direction.dot(u_rel) - target
    if stiffness > 0.0 and gap < 0.0:
        energy = 0.5 * stiffness * gap * gap
        gradient = stiffness * gap * direction
        hessian = stiffness * direction.outer_product(direction)
    return energy, gradient, hessian


@qd.func
def func_pyramid_terms(
    normal: qd.types.vector(3),
    friction: float,
    stiffness: qd.types.vector(PYRAMID_ROWS),
    target: qd.types.vector(PYRAMID_ROWS),
    u_rel: qd.types.vector(3),
):
    """Return the energy, the gradient and the Hessian of the soft rows of the friction pyramid of a contact point in
    its displacement relative to its partner (see 'func_soft_row_terms')."""
    directions = func_pyramid_directions(normal, friction)
    energy = gs.qd_float(0.0)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    for k in qd.static(range(directions.n)):
        energy_k, gradient_k, hessian_k = func_soft_row_terms(directions[k, :], stiffness[k], target[k], u_rel)
        energy += energy_k
        gradient += gradient_k
        hessian += hessian_k
    return energy, gradient, hessian
