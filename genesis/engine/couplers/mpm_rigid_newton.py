"""Rigid's dofs as a secondary participant of the NewtonSolver, and the MPM-Rigid contact rows as a coupling (design
§6.2.3). The stage functions are those of `mpm_rigid_func`; this module binds them to the data of one scene.

The Rigid unknown is x = dt^2 qacc per dof, so that Rigid's convex primal cost times dt^4 joins the joint energy in the
displacement units of the MPM energy, with Rigid's island Cholesky factor of M + J^T D J as its preconditioner. The
contact coupling acts on the displacement of each particle relative to the motion of its contact point on the link.
"""

import quadrants as qd

from . import mpm_rigid_func


@qd.data_oriented
class RigidNewtonParticipant:
    """Device stages of Rigid's dofs in the joint solve."""

    def __init__(self, rigid, coupling, config, params):
        self.coupling = coupling
        self.dyn_state = rigid.dyn_state
        self.constraint_state = rigid.constraint_solver.constraint_state
        self.dyn_info = rigid.dyn_info
        self.rigid_info = rigid.rigid_info
        self.rigid_config = rigid.rigid_config
        self.config = config
        self.params = params

    @qd.func(requires_top_level=True)
    def func_evaluate(self, f):
        mpm_rigid_func.func_rigid_evaluate(
            self.coupling,
            self.dyn_state,
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self.config,
            self.params,
        )

    @qd.func(requires_top_level=True)
    def func_gradient(self, f):
        mpm_rigid_func.func_rigid_gradient(
            self.coupling, self.dyn_state, self.constraint_state, self.config, self.params
        )

    @qd.func(requires_top_level=True)
    def func_factor(self):
        mpm_rigid_func.func_rigid_factor(
            self.coupling,
            self.dyn_state,
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self.config,
        )

    @qd.func(requires_top_level=True)
    def func_pcg_init(self):
        mpm_rigid_func.func_rigid_pcg_init(self.coupling)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        mpm_rigid_func.func_rigid_precondition(
            self.coupling, self.dyn_state, self.constraint_state, self.rigid_info, self.rigid_config, self.config
        )

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, ws: qd.template()):
        mpm_rigid_func.func_rigid_pcg_init_direction(self.coupling, ws)

    @qd.func(requires_top_level=True)
    def func_matvec(self):
        mpm_rigid_func.func_rigid_matvec(
            self.coupling, self.constraint_state, self.dyn_info, self.rigid_info, self.rigid_config, self.config
        )

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, ws: qd.template()):
        mpm_rigid_func.func_rigid_matvec_dot(self.coupling, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, ws: qd.template()):
        mpm_rigid_func.func_rigid_pcg_update_xr(self.coupling, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, ws: qd.template()):
        mpm_rigid_func.func_rigid_pcg_dot_rz(self.coupling, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, ws: qd.template()):
        mpm_rigid_func.func_rigid_pcg_direction(self.coupling, ws)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f):
        mpm_rigid_func.func_rigid_start_line_search(self.coupling, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f):
        mpm_rigid_func.func_rigid_step_forward(self.coupling, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f, is_trial):
        self.func_evaluate(f)
        mpm_rigid_func.func_rigid_energy(self.coupling, control, self.params, is_trial)


@qd.data_oriented
class MpmRigidContact:
    """Device stages of the MPM-Rigid contact rows in the joint solve: their cross terms in the Rigid gradient and in
    both blocks of the operator, and the contact point displacement at every iterate."""

    def __init__(self, mpm, rigid):
        self.particles = mpm.particles
        self.grid = mpm.grid
        self.newton = mpm.newton
        self.linear = mpm.linear
        self.coupling = rigid.coupling
        self.dyn_state = rigid.dyn_state
        self.constraint_state = rigid.constraint_state
        self.dyn_info = rigid.dyn_info
        self.config = rigid.config
        self.params = rigid.params

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f):
        mpm_rigid_func.func_rigid_contact_gradient(
            self.particles,
            self.grid,
            self.newton,
            self.coupling,
            self.dyn_state,
            self.dyn_info,
            self.config,
            self.params,
            f,
        )

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f):
        mpm_rigid_func.func_rigid_contact_matvec(
            self.particles,
            self.grid,
            self.newton,
            self.linear,
            self.coupling,
            self.dyn_state,
            self.dyn_info,
            self.config,
            self.params,
            f,
        )

    @qd.func(requires_top_level=True)
    def func_contact_matvec_post(self, f):
        mpm_rigid_func.func_rigid_contact_matvec_post(self.coupling, self.dyn_state, self.dyn_info, self.config)

    @qd.func(requires_top_level=True)
    def func_after_step(self, f):
        mpm_rigid_func.func_link_spatial_disp(
            self.coupling, self.coupling.x, self.dyn_info.links, self.dyn_state.dofs, self.config
        )
        mpm_rigid_func.func_contact_point_disp(
            self.particles, self.grid, self.newton, self.coupling, self.dyn_state.links, self.params, f
        )


@qd.kernel
def kernel_rigid_begin(rigid: qd.template()):
    mpm_rigid_func.func_rigid_begin(
        rigid.coupling, rigid.dyn_state.dofs, rigid.constraint_state, rigid.rigid_config, rigid.config, rigid.params
    )


@qd.kernel
def kernel_rigid_write_back(rigid: qd.template(), f: qd.i32):
    rigid.func_evaluate(f)
    mpm_rigid_func.func_rigid_write_back(rigid.coupling, rigid.constraint_state, rigid.config)


@qd.kernel
def kernel_contact_point_disp(contact: qd.template(), f: qd.i32):
    contact.func_after_step(f)


@qd.kernel
def kernel_mpm_reaction(contact: qd.template(), f: qd.i32):
    mpm_rigid_func.func_mpm_reaction(
        contact.particles,
        contact.grid,
        contact.newton,
        contact.coupling,
        contact.dyn_state,
        contact.constraint_state,
        contact.params,
        f,
    )


@qd.kernel
def kernel_add_mpm_contact_force(contact: qd.template()):
    mpm_rigid_func.func_add_mpm_contact_force(contact.coupling, contact.dyn_state.links, contact.params)


class RigidNewtonHost:
    """Host stages of Rigid's dofs around the joint solve: the start point from Rigid's warm start, and the hand-back
    of the converged qacc, after which Rigid finishes its deferred substep."""

    def __init__(self, rigid_solver, device: RigidNewtonParticipant):
        self._rigid_solver = rigid_solver
        self._device = device

    def newton_prepare(self, f):
        kernel_rigid_begin(self._device)

    def newton_finish(self, f):
        kernel_rigid_write_back(self._device, f)

    def newton_after(self, f):
        self._rigid_solver.finish_deferred_substep()


class MpmRigidContactHost:
    """Host stages of the MPM-Rigid contact rows around the joint solve: the contact point displacement at the start
    point, the MPM rows' force on Rigid at the converged solve, and its share of Rigid's reported contact forces once
    Rigid wrote its own."""

    def __init__(self, device: MpmRigidContact):
        self._device = device

    def newton_prepare(self, f):
        kernel_contact_point_disp(self._device, f)

    def newton_finish(self, f):
        kernel_mpm_reaction(self._device, f)

    def newton_after(self, f):
        kernel_add_mpm_contact_force(self._device)
