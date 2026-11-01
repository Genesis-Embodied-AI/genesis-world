"""Soft-constraint rows of Rigid (MuJoCo's solref / solimp impedance) written on a displacement unknown.

Rigid solves for the acceleration a, with rows 1/2 D (J a - aref)^2 active where J a < aref. The displacement unknown is
u = dt v_new, with v_new = v + dt a. Every row then becomes 1/2 D min(0, J u - t)^2, where t = dt J v + dt^2 aref plus
the displacement of the partner, on the scale of the kinetic term 1/2 m |u - dt v*|^2 of an implicit Newton step.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu


@qd.func
def soft_row(params, phi: float, vel: float, diag_coef: float, dt: float):
    """Return the stiffness D and the target t of one row relative to the partner, at a negative signed distance.

    phi is the signed distance, vel the velocity J v of the row relative to the partner at the step start, and the
    inverse weight factor diag_coef is the diag of the rows of Rigid before the impedance scaling (1 - imp) / imp.
    """
    imp, aref = gu.imp_aref(params, phi, vel, phi)
    diag = qd.max(diag_coef * (1.0 - imp) / imp, gs.EPS)
    return 1.0 / diag, dt * vel + dt * dt * aref
