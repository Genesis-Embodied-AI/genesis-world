"""Soft-constraint rows of Rigid (its solref / solimp impedance) written on a displacement unknown."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu


@qd.func
def soft_row(params: qd.types.vector(7), phi: float, vel: float, diag_coef: float, dt: float):
    """Return the stiffness and the target of one soft row relative to its partner.

    The partner lies at signed distance `phi` and approaches at velocity `vel` along the row, whose solref / solimp
    parameters are `params`. `diag_coef` scales the regularized diagonal of the row, and `dt` is the substep."""
    imp, aref = gu.imp_aref(params, phi, vel, phi)
    diag = qd.max(diag_coef * (1.0 - imp) / imp, gs.EPS)
    return 1.0 / diag, dt * vel + dt * dt * aref
