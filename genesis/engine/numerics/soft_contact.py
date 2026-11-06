"""Soft-constraint rows of Rigid (its solref / solimp impedance) written on a displacement unknown.

Rigid's rows 1/2 D (J a - aref)^2, active where J a < aref, become 1/2 D min(0, J u - t)^2 on the displacement u = dt (v
+ dt a), with t = dt J v + dt^2 aref plus the displacement of the partner."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu


@qd.func
def soft_row(params: qd.types.vector(7), phi: float, vel: float, diag_coef: float, dt: float):
    """Return the stiffness D and the target t of one row relative to the partner at signed distance `phi`.

    `vel` is the velocity J v of the row relative to the partner at the step start, and `diag_coef` the diag of the rows
    of Rigid before the impedance scaling (1 - imp) / imp."""
    imp, aref = gu.imp_aref(params, phi, vel, phi)
    diag = qd.max(diag_coef * (1.0 - imp) / imp, gs.EPS)
    return 1.0 / diag, dt * vel + dt * dt * aref
