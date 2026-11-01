"""Rigid's dofs as a secondary participant of the NewtonSolver, and the contact rows between the material point method
(MPM) and Rigid as a coupling.

This module binds the stage functions of `mpm_rigid_func` to the data of one scene. The Rigid unknown, its share of the
joint energy, its preconditioner and the contact coupling are described there.
"""

import quadrants as qd

from . import mpm_rigid_adjoint_func, mpm_rigid_func


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
        self.sdf_info = rigid.collider.sdf.sdf_info
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
    """Device stages of the MPM-Rigid contact rows in the joint solve.

    They add the cross terms of the rows to the Rigid gradient and to both blocks of the operator, and update the
    contact point displacement at every iterate."""

    def __init__(self, mpm, rigid):
        self.particles = mpm.particles
        self.particles_info = mpm.particles_info
        self.grid = mpm.grid
        self.newton = mpm.newton
        self.linear = mpm.linear
        self.coupling = rigid.coupling
        self.dyn_state = rigid.dyn_state
        self.constraint_state = rigid.constraint_state
        self.dyn_info = rigid.dyn_info
        self.rigid_info = rigid.rigid_info
        self.sdf_info = rigid.sdf_info
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


@qd.kernel
def kernel_rigid_adjoint_rhs(rigid: qd.template()):
    mpm_rigid_func.func_rigid_adjoint_rhs(rigid.coupling, rigid.constraint_state, rigid.params)


@qd.kernel
def kernel_rigid_adjoint_finish(rigid: qd.template()):
    mpm_rigid_func.func_rigid_adjoint_finish(rigid.coupling, rigid.constraint_state, rigid.params)


@qd.kernel
def kernel_rigid_contact_adjoint(contact: qd.template(), f: qd.i32):
    mpm_rigid_adjoint_func.func_rigid_contact_adjoint(
        contact.particles,
        contact.particles_info,
        contact.grid,
        contact.newton,
        contact.linear,
        contact.coupling,
        contact.dyn_state.geoms,
        contact.dyn_info.geoms,
        contact.dyn_state.links,
        contact.dyn_info.links,
        contact.dyn_state.dofs,
        contact.rigid_info,
        contact.sdf_info,
        contact.config,
        contact.params,
        f,
    )


class RigidNewtonHost:
    """Host stages of Rigid's dofs around the joint solve.

    They start from Rigid's warm start and hand back the converged qacc, from which Rigid updates its accelerations and
    contact forces, then integrates its deferred substep once the solve is over. The backward pass is Rigid's reverse of
    its constraint solve, whose adjoint solve is the joint one."""

    is_differentiable = True

    def __init__(self, rigid_solver, device: RigidNewtonParticipant):
        self._rigid_solver = rigid_solver
        self._device = device

    def newton_prepare(self, f):
        kernel_rigid_begin(self._device)

    def newton_finish(self, f):
        kernel_rigid_write_back(self._device, f)

    def newton_after(self, f):
        self._rigid_solver.finish_deferred_solve()

    def newton_end(self, f):
        self._rigid_solver.integrate_deferred_substep()

    def newton_backward(self, f, solve):
        """Reverse Rigid's constraint solve, with its adjoint solve run jointly with the other participants.

        `solve` runs the joint adjoint solve."""
        self._rigid_solver._constraint_force_grad(solve_adjoint=solve)

    def newton_adjoint_rhs(self, f):
        kernel_rigid_adjoint_rhs(self._device)

    def newton_adjoint_finish(self, f):
        kernel_rigid_adjoint_finish(self._device)


class MpmRigidContactHost:
    """Host stages of the MPM-Rigid contact rows around the joint solve.

    They set the contact point displacement at the start point, apply the force of the MPM rows on Rigid at the
    converged solve, and add its share to Rigid's reported contact forces once Rigid wrote its own."""

    is_differentiable = True

    def __init__(self, device: MpmRigidContact):
        self._device = device

    def newton_prepare(self, f):
        kernel_contact_point_disp(self._device, f)

    def newton_finish(self, f):
        kernel_mpm_reaction(self._device, f)

    def newton_after(self, f):
        kernel_add_mpm_contact_force(self._device)

    def newton_end(self, f):
        pass

    def newton_adjoint_rhs(self, f):
        pass

    def newton_adjoint_finish(self, f):
        """Pull the contact rows back into the adjoints of the particles and of Rigid's state."""
        kernel_rigid_contact_adjoint(self._device, f)
