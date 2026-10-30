"""Implicit FEM as a secondary participant of the NewtonSolver, and the MPM-FEM contact rows as a coupling. The stage
functions are those of `mpm_fem_func`; this module binds them to the data of one scene.

FEM's own implicit substep (`FEMSolver.batch_solve`) is replaced by the joint solve: FEM prepares its inertia and
material data, the joint Newton iterations move its end-of-substep positions together with the MPM grid, and FEM takes
its velocities from them and runs the rest of its coupling after the solve.
"""

import quadrants as qd

from . import mpm_fem_func


@qd.data_oriented
class FemNewtonParticipant:
    """Device stages of the FEM vertex positions in the joint solve."""

    def __init__(self, fem):
        self.fem = fem

    @qd.func(requires_top_level=True)
    def func_evaluate(self, f):
        self.fem._func_compute_ele_hessian_gradient(f)

    @qd.func(requires_top_level=True)
    def func_gradient(self, f):
        self.fem._func_accumulate_vertex_force_preconditioner(f)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        mpm_fem_func.func_fem_factor(self.fem)

    @qd.func(requires_top_level=True)
    def func_pcg_init(self):
        mpm_fem_func.func_fem_pcg_init(self.fem, self.fem.substep_dt)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        mpm_fem_func.func_fem_precondition(self.fem, self.fem.substep_dt)

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, ws: qd.template()):
        mpm_fem_func.func_fem_pcg_init_direction(self.fem, ws)

    @qd.func(requires_top_level=True)
    def func_matvec(self):
        mpm_fem_func.func_fem_matvec(self.fem, self.fem.substep_dt)

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, ws: qd.template()):
        mpm_fem_func.func_fem_matvec_dot(self.fem, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, ws: qd.template()):
        mpm_fem_func.func_fem_pcg_update_xr(self.fem, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, ws: qd.template()):
        mpm_fem_func.func_fem_pcg_dot_rz(self.fem, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, ws: qd.template()):
        mpm_fem_func.func_fem_pcg_direction(self.fem, ws)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f):
        mpm_fem_func.func_fem_start_line_search(self.fem, control, f)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f):
        mpm_fem_func.func_fem_step_forward(self.fem, control, f)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f, is_trial):
        mpm_fem_func.func_fem_energy(self.fem, control, f, is_trial, self.fem.substep_dt)


@qd.data_oriented
class MpmFemContact:
    """Device stages of the MPM-FEM contact rows in the joint solve: their share of the FEM gradient, their cross
    terms in both blocks of the operator, and the contact point displacement at every iterate."""

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
    def func_contact_gradient(self, f):
        mpm_fem_func.func_fem_contact_gradient(self.grid, self.newton, self.fem, self.params)

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f):
        mpm_fem_func.func_fem_contact_matvec(
            self.particles, self.grid, self.newton, self.linear, self.fem, self.params, f
        )

    @qd.func(requires_top_level=True)
    def func_contact_matvec_post(self, f):
        mpm_fem_func.func_fem_contact_matvec_post(
            self.particles, self.grid, self.newton, self.linear, self.fem, self.params, f
        )

    @qd.func(requires_top_level=True)
    def func_after_step(self, f):
        mpm_fem_func.func_fem_point_disp(self.grid, self.newton, self.fem, self.params, f)


@qd.kernel
def kernel_fem_begin(fem: qd.template()):
    for i_b in range(fem._B):
        fem.batch_active[i_b] = True
        fem.batch_pcg_active[i_b] = True
        fem.batch_linesearch_active[i_b] = True


@qd.kernel
def kernel_fem_rows(contact: qd.template(), f: qd.i32):
    mpm_fem_func.func_build_fem_rows(
        contact.particles,
        contact.particles_info,
        contact.grid,
        contact.newton,
        contact.fem,
        contact.params,
        contact.query,
        f,
        contact.params.substep_dt[()],
    )
    contact.func_after_step(f)


class FemNewtonHost:
    """Host stages of FEM around the joint solve: the inertia and material data of its substep before, its velocities
    from the solved positions after, and then the part of the coupler's step on FEM that FEM's own solve would have
    preceded."""

    def __init__(self, fem_solver, coupler):
        self._fem_solver = fem_solver
        self._coupler = coupler

    def newton_prepare(self, f):
        self._fem_solver.precompute_material_data(f)
        self._fem_solver.init_pos_and_inertia(f)
        kernel_fem_begin(self._fem_solver)

    def newton_finish(self, f):
        self._fem_solver.setup_pos_vel(f)

    def newton_after(self, f):
        self._coupler.couple_fem(f)

    def newton_end(self, f):
        pass


class MpmFemContactHost:
    """Host stages of the MPM-FEM contact rows before the joint solve: the triangle trees, then the rows and their
    contact point displacement at the start point. The rows' force on FEM is in the joint solve itself."""

    def __init__(self, device: MpmFemContact):
        self._device = device

    def newton_prepare(self, f):
        self._device.query.build(f)
        kernel_fem_rows(self._device, f)

    def newton_finish(self, f):
        pass

    def newton_after(self, f):
        pass

    def newton_end(self, f):
        pass
