"""Rigid's soft-constraint rows (MuJoCo solref / solimp impedance) on a displacement unknown.

Rigid solves for the acceleration a, with rows 1/2 D (J a - aref)^2 active where J a < aref. With the displacement
u = dt v_new and v_new = v + dt a, every row becomes 1/2 D min(0, J u - t)^2, on the scale of the kinetic term
1/2 m |u - dt v*|^2 of an implicit Newton step, with t = dt J v + dt^2 aref plus the partner's own displacement.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu


@qd.func
def soft_row(params, phi, vel, diag_coef, dt):
    """Stiffness D and target t of one row, relative to the partner, at signed distance phi < 0.

    `vel` is the row's velocity J v relative to the partner at the step start and `diag_coef` its inverse weight
    factor, the `diag` of Rigid's rows before the impedance scaling `(1 - imp) / imp`.
    """
    imp, aref = gu.imp_aref(params, phi, vel, phi)
    diag = qd.max(diag_coef * (1.0 - imp) / imp, gs.EPS)
    return 1.0 / diag, dt * vel + dt * dt * aref
