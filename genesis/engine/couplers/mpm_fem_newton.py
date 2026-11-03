"""Implicit finite element method (FEM) as a participant of the NewtonSolver, and the contact rows of the material point
method (MPM) with FEM as a coupling.

The stage functions are those of `mpm_fem_func`, and this module binds them to the data of one scene.

The joint solve replaces FEM's own implicit substep (`FEMSolver.batch_solve`): FEM prepares its inertia and material
data, the joint Newton iterations move its end-of-substep positions together with the MPM grid, and FEM takes its
velocities from them and runs the rest of its coupling after the solve.
"""

import quadrants as qd

from genesis.engine.solvers.newton_solver import NewtonHost, kernel_assemble_participant

from . import mpm_fem_func


@qd.data_oriented
class FemNewtonParticipant:
    """Device stages of the FEM vertex positions in the joint solve."""

    def __init__(self, fem):
        self.fem = fem

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int):
        self.fem._func_compute_ele_hessian_gradient(f)
        self.fem._func_accumulate_vertex_force_preconditioner(f)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        mpm_fem_func.func_fem_factor(self.fem)

    @qd.func(requires_top_level=True)
    def func_pcg_init(self):
        mpm_fem_func.func_fem_pcg_init(self.fem.substep_dt, self.fem)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        mpm_fem_func.func_fem_precondition(self.fem.substep_dt, self.fem)

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, pcg_scalars: qd.template()):
        mpm_fem_func.func_fem_pcg_init_direction(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_matvec(self, pcg_scalars: qd.template()):
        mpm_fem_func.func_fem_matvec(self.fem.substep_dt, self.fem)

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, pcg_scalars: qd.template()):
        mpm_fem_func.func_fem_matvec_dot(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, pcg_scalars: qd.template()):
        mpm_fem_func.func_fem_pcg_update_xr(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, pcg_scalars: qd.template()):
        mpm_fem_func.func_fem_pcg_dot_rz(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, pcg_scalars: qd.template()):
        mpm_fem_func.func_fem_pcg_direction(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        mpm_fem_func.func_fem_start_line_search(f, self.fem, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        mpm_fem_func.func_fem_step_forward(f, self.fem, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f: int, is_trial: bool):
        mpm_fem_func.func_fem_energy(f, self.fem.substep_dt, self.fem, control, is_trial)


@qd.data_oriented
class MpmFemContact:
    """Device stages of the MPM-FEM contact rows in the joint solve: their share of the FEM gradient, their cross terms
    in both blocks of the operator, and the contact point displacement at every iterate."""

    def __init__(self, mpm, fem, query):
        self.particles = mpm.particles
        self.particles_info = mpm.particles_info
        self.grid = mpm.grid
        self.newton = mpm.newton
        self.linear = mpm.linear
        self.params = mpm.params
        self.fem = fem.fem
        self.query = query

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f: int):
        mpm_fem_func.func_fem_contact_gradient(self.grid, self.newton, self.fem, self.params)

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f: int, pcg_scalars: qd.template()):
        mpm_fem_func.func_fem_contact_matvec(
            f, self.particles, self.grid, self.newton, self.linear, self.fem, pcg_scalars, self.params
        )
        mpm_fem_func.func_fem_contact_matvec_post(
            f, self.particles, self.grid, self.newton, self.linear, self.fem, pcg_scalars, self.params
        )

    @qd.func(requires_top_level=True)
    def func_after_step(self, f: int):
        mpm_fem_func.func_fem_point_disp(f, self.grid, self.newton, self.fem, self.params)


@qd.kernel
def kernel_fem_begin(fem: qd.template()):
    for i_b in range(fem._B):
        fem.batch_active[i_b] = True
        fem.batch_pcg_active[i_b] = True
        fem.batch_linesearch_active[i_b] = True


@qd.kernel
def kernel_fem_rows(f: int, contact: qd.template()):
    mpm_fem_func.func_build_fem_rows(
        f,
        contact.params.substep_dt[()],
        contact.particles,
        contact.grid,
        contact.newton,
        contact.fem,
        contact.query,
        contact.particles_info,
        contact.params,
    )
    contact.func_after_step(f)


class FemNewtonHost(NewtonHost):
    """Host stages of FEM around the joint solve: the inertia and material data of its substep before, and its
    velocities from the solved positions after.

    Once FEM has its velocities, the host runs the part of the coupler's step on FEM that follows FEM's own solve in a
    substep FEM solves alone (see LegacyCoupler.couple_fem)."""

    def __init__(self, fem_solver, device: FemNewtonParticipant, coupler):
        self._fem_solver = fem_solver
        self._device = device
        self._coupler = coupler

    def newton_prepare(self, f):
        self._fem_solver.precompute_material_data(f)
        self._fem_solver.init_pos_and_inertia(f)
        kernel_fem_begin(self._fem_solver)

    def newton_assemble(self, f, is_exact=False):
        kernel_assemble_participant(f, self._device)

    def newton_finish(self, f):
        self._fem_solver.setup_pos_vel(f)

    def newton_after(self, f):
        self._coupler.couple_fem(f)


class MpmFemContactHost(NewtonHost):
    """Host stages of the MPM-FEM contact rows before the joint solve: the triangle trees, then the rows and their
    contact point displacement at the start point.

    The force of the rows on FEM acts in the joint solve itself."""

    def __init__(self, device: MpmFemContact):
        self._device = device

    def newton_prepare(self, f):
        self._device.query.build(f)
        kernel_fem_rows(f, self._device)
