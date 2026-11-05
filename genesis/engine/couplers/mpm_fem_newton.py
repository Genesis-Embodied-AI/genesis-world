"""The contact rows of the material point method (MPM) with the implicit finite element method (FEM) as a coupling of
the NewtonSolver.

The stage functions are those of `mpm_fem_func`, and this module binds them to the data of one scene, on the
participants of MPM and FEM (see FEMSolver.build_newton_participant).
"""

import quadrants as qd

from genesis.engine.solvers.fem_newton import FemNewtonParticipant
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_solver import NewtonHost

from . import mpm_fem_func


@qd.data_oriented
class MpmFemContact:
    """Device stages of the MPM-FEM contact rows in the joint solve: their share of the FEM gradient, their cross terms
    in both blocks of the operator, and the contact point displacement at every iterate."""

    def __init__(self, mpm: MpmNewtonParticipant, fem: FemNewtonParticipant, query: mpm_fem_func.FemTriangleQuery):
        self.mpm = mpm
        self.fem = fem
        self.query = query

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f: int):
        mpm_fem_func.func_fem_contact_gradient(self.mpm.grid, self.mpm.newton, self.fem, self.mpm.params)

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f: int, pcg_scalars: qd.template()):
        mpm_fem_func.func_fem_contact_matvec(
            f,
            self.mpm.particles,
            self.mpm.grid,
            self.mpm.newton,
            self.mpm.linear,
            self.fem,
            pcg_scalars,
            self.mpm.params,
        )

    @qd.func(requires_top_level=True)
    def func_after_step(self, f: int):
        mpm_fem_func.func_fem_point_disp(f, self.mpm.grid, self.mpm.newton, self.fem, self.mpm.params)


@qd.kernel
def kernel_fem_rows(f: int, contact: qd.template()):
    mpm_fem_func.func_build_fem_rows(
        f,
        contact.mpm.params.substep_dt[()],
        contact.mpm.particles,
        contact.mpm.grid,
        contact.mpm.newton,
        contact.fem,
        contact.query,
        contact.mpm.particles_info,
        contact.mpm.params,
    )
    contact.func_after_step(f)


class MpmFemContactHost(NewtonHost):
    """Host stages of the MPM-FEM contact rows before the joint solve: the triangle trees, then the rows and their
    contact point displacement at the start point.

    The force of the rows on FEM acts in the joint solve itself."""

    def __init__(self, device: MpmFemContact):
        self._device = device

    def newton_prepare(self, f):
        self._device.query.build(f)
        kernel_fem_rows(f, self._device)
