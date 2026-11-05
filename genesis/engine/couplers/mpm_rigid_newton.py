"""The contact rows between the material point method (MPM) and Rigid as a coupling of the NewtonSolver.

This module binds the stage functions of `mpm_rigid_func` to the data of one scene, on the participants of MPM and Rigid
(see RigidSolver.build_newton_participant). The coupling builds the rows of the particles on the coupled geoms every
substep, which MPM assembles with its own, and on the geoms of a Rigid without dofs they act on the particle
displacement alone.
"""

import quadrants as qd

from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_solver import NewtonDevice, NewtonHost
from genesis.engine.solvers.rigid.newton_func import func_link_spatial_disp
from genesis.engine.solvers.rigid.newton_participant import RigidNewtonParticipant

from . import mpm_rigid_adjoint_func, mpm_rigid_func


@qd.data_oriented
class MpmRigidContact(NewtonDevice):
    """Device stages of the contact rows between the material point method (MPM) and Rigid in the joint solve.

    They build the rows of the substep with `query`, the tree of the coupled geoms, add the cross terms of the rows to
    the Rigid gradient and to both blocks of the operator, and update the contact point displacement at every iterate.
    `rigid` is the participant of Rigid, whose dofs the rows act on. A Rigid without dofs gives a participant that is
    no member of the solve, and the rows then act on the particle displacement alone."""

    def __init__(self, mpm: MpmNewtonParticipant, rigid: RigidNewtonParticipant, query: mpm_rigid_func.RigidGeomQuery):
        self.mpm = mpm
        self.rigid = rigid
        self.query = query

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f: int):
        mpm_rigid_func.func_rigid_contact_gradient(
            f,
            self.mpm.particles,
            self.mpm.grid,
            self.mpm.newton,
            self.rigid.rigid_data,
            self.rigid.dyn_state,
            self.rigid.dyn_info,
            self.mpm.mpm_info,
            self.rigid.config,
        )

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f: int, pcg_scalars: qd.template()):
        mpm_rigid_func.func_rigid_contact_matvec(
            f,
            self.mpm.particles,
            self.mpm.grid,
            self.mpm.newton,
            self.mpm.linear,
            self.rigid.rigid_data,
            pcg_scalars,
            self.rigid.dyn_state,
            self.rigid.dyn_info,
            self.mpm.mpm_info,
            self.rigid.config,
        )
        mpm_rigid_func.func_rigid_contact_matvec_post(
            self.rigid.rigid_data, self.rigid.dyn_state, self.rigid.dyn_info, self.rigid.config
        )

    @qd.func(requires_top_level=True)
    def func_after_step(self, f: int):
        func_link_spatial_disp(
            src=self.rigid.rigid_data.x,
            rigid_data=self.rigid.rigid_data,
            dofs_state=self.rigid.dyn_state.dofs,
            links_info=self.rigid.dyn_info.links,
            config=self.rigid.config,
        )
        mpm_rigid_func.func_contact_point_disp(
            f,
            self.mpm.particles,
            self.mpm.grid,
            self.mpm.newton,
            self.rigid.rigid_data,
            self.rigid.dyn_state.links,
            self.mpm.mpm_info,
        )


@qd.kernel
def kernel_rigid_rows(f: int, contact: qd.template()):
    mpm_rigid_func.func_build_rigid_rows(
        f,
        contact.mpm.mpm_info.substep_dt[()],
        contact.mpm.particles,
        contact.mpm.grid,
        contact.mpm.newton,
        contact.rigid.dyn_state.geoms,
        contact.rigid.dyn_state.links,
        contact.query,
        contact.mpm.particles_info,
        contact.rigid.dyn_info.geoms,
        contact.rigid.dyn_info.links,
        contact.rigid.rigid_info,
        contact.rigid.sdf_info,
        contact.mpm.mpm_info,
        contact.rigid.config,
    )


@qd.kernel
def kernel_contact_point_disp(f: int, contact: qd.template()):
    contact.func_after_step(f)


@qd.kernel
def kernel_mpm_reaction(f: int, contact: qd.template()):
    mpm_rigid_func.func_mpm_reaction(
        f,
        contact.mpm.particles,
        contact.mpm.grid,
        contact.mpm.newton,
        contact.rigid.rigid_data,
        contact.rigid.dyn_state,
        contact.rigid.constraint_state,
        contact.mpm.mpm_info,
    )


@qd.kernel
def kernel_add_mpm_contact_force(contact: qd.template()):
    mpm_rigid_func.func_add_mpm_contact_force(
        contact.rigid.rigid_data, contact.rigid.dyn_state.links, contact.mpm.mpm_info
    )


@qd.kernel
def kernel_rigid_contact_adjoint(f: int, contact: qd.template()):
    mpm_rigid_adjoint_func.func_rigid_contact_adjoint(
        f,
        contact.mpm.particles,
        contact.mpm.grid,
        contact.mpm.newton,
        contact.mpm.linear,
        contact.rigid.rigid_data,
        contact.rigid.dyn_state.geoms,
        contact.rigid.dyn_state.links,
        contact.rigid.dyn_state.dofs,
        contact.mpm.particles_info,
        contact.rigid.dyn_info.geoms,
        contact.rigid.dyn_info.links,
        contact.rigid.rigid_info,
        contact.rigid.sdf_info,
        contact.mpm.mpm_info,
        contact.rigid.config,
    )


class MpmRigidContactHost(NewtonHost):
    """Host stages of the MPM-Rigid contact rows around the joint solve.

    They build the rows of the substep, and the backward pass pulls them back. With `has_rigid_dofs`, the rows act on
    Rigid's dofs in the joint solve as well: they set the contact point displacement at the start point, apply the force
    of the MPM rows on Rigid at the converged solve, and add its share to Rigid's reported contact forces once Rigid
    wrote its own."""

    is_differentiable = True

    def __init__(self, device: MpmRigidContact, has_rigid_dofs: bool):
        self._device = device
        self._has_rigid_dofs = has_rigid_dofs

    def newton_prepare(self, f):
        self._device.query.build(self._device.mpm.mpm_info)
        kernel_rigid_rows(f, self._device)
        if self._has_rigid_dofs:
            kernel_contact_point_disp(f, self._device)

    def newton_finish(self, f, control):
        if self._has_rigid_dofs:
            kernel_mpm_reaction(f, self._device)

    def newton_after(self, f):
        if self._has_rigid_dofs:
            kernel_add_mpm_contact_force(self._device)

    def newton_adjoint_finish(self, f):
        """Pull the contact rows back into the adjoints of the particles and of Rigid's state."""
        kernel_rigid_contact_adjoint(f, self._device)
