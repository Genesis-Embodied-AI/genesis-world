"""The contact rows between the material point method (MPM) and Rigid as a coupling of the NewtonSolver.

This module binds the stage functions of `mpm_rigid_func` to the data of one scene, on the participants of MPM and Rigid
(see RigidSolver.build_newton_participant). The coupling builds the rows of the particles on the coupled geoms every
substep, which MPM assembles with its own, and on the geoms of a Rigid without dofs they act on the particle
displacement alone.
"""

import quadrants as qd

from genesis.engine.solvers.newton_solver import NewtonHost
from genesis.engine.solvers.rigid.newton_func import func_link_spatial_disp
from genesis.engine.solvers.rigid.newton_participant import RigidNewtonData, has_coupled_terrain
from genesis.utils.data import StaticConfig

from . import mpm_rigid_adjoint_func, mpm_rigid_func


class MpmRigidContactConfig(StaticConfig):
    """Compile-time switches of the stages of the MPM-Rigid contact rows."""

    # The links info of Rigid is batched over the environments
    rigid_batch_links_info: bool
    # Rigid has constraint rows, which join the solve
    has_rigid_rows: bool
    # A coupled geom is a terrain (see has_coupled_terrain). This config stands in for Rigid's collider configuration in
    # the signed distance field (SDF) queries, which read this one switch of it.
    has_terrain: bool


@qd.data_oriented
class MpmRigidContact:
    """Device stages of the MPM-Rigid contact rows in the joint solve.

    They build the rows of the substep with `query`, the tree of the coupled geoms, add the cross terms of the rows to
    the Rigid gradient and to both blocks of the operator, and update the contact point displacement at every iterate.
    `rigid_data` is the data of Rigid's participant, whose dofs the rows act on. A Rigid without dofs gives a data with
    no dof of any row, the rows then acting on the particle displacement alone."""

    def __init__(self, mpm, rigid_solver, rigid_data: RigidNewtonData, n_envs: int):
        self.particles = mpm.particles
        self.particles_info = mpm.particles_info
        self.grid = mpm.grid
        self.newton = mpm.newton
        self.linear = mpm.linear
        self.params = mpm.params
        self.rigid_data = rigid_data
        self.dyn_state = rigid_solver.dyn_state
        self.constraint_state = rigid_solver.constraint_solver.constraint_state
        self.dyn_info = rigid_solver.dyn_info
        self.rigid_info = rigid_solver.rigid_info
        self.sdf_info = rigid_solver.collider.collider_info.sdf
        self.config = MpmRigidContactConfig(
            rigid_batch_links_info=rigid_solver._options.batch_links_info,
            has_rigid_rows=not rigid_solver._disable_constraint,
            has_terrain=has_coupled_terrain(rigid_solver),
        )
        self.query = mpm_rigid_func.RigidGeomQuery(rigid_solver, mpm.newton, n_envs, True)

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f: int):
        mpm_rigid_func.func_rigid_contact_gradient(
            f,
            self.particles,
            self.grid,
            self.newton,
            self.rigid_data,
            self.dyn_state,
            self.dyn_info,
            self.params,
            self.config,
        )

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f: int, pcg_scalars: qd.template()):
        mpm_rigid_func.func_rigid_contact_matvec(
            f,
            self.particles,
            self.grid,
            self.newton,
            self.linear,
            self.rigid_data,
            pcg_scalars,
            self.dyn_state,
            self.dyn_info,
            self.params,
            self.config,
        )
        mpm_rigid_func.func_rigid_contact_matvec_post(self.rigid_data, self.dyn_state, self.dyn_info, self.config)

    @qd.func(requires_top_level=True)
    def func_after_step(self, f: int):
        func_link_spatial_disp(
            self.rigid_data.x, self.rigid_data, self.dyn_state.dofs, self.dyn_info.links, self.config
        )
        mpm_rigid_func.func_contact_point_disp(
            f, self.particles, self.grid, self.newton, self.rigid_data, self.dyn_state.links, self.params
        )


@qd.kernel
def kernel_rigid_rows(f: int, contact: qd.template()):
    mpm_rigid_func.func_build_rigid_rows(
        f,
        contact.params.substep_dt[()],
        contact.particles,
        contact.grid,
        contact.newton,
        contact.dyn_state.geoms,
        contact.dyn_state.links,
        contact.query,
        contact.particles_info,
        contact.dyn_info.geoms,
        contact.dyn_info.links,
        contact.rigid_info,
        contact.sdf_info,
        contact.params,
        contact.config,
    )


@qd.kernel
def kernel_contact_point_disp(f: int, contact: qd.template()):
    contact.func_after_step(f)


@qd.kernel
def kernel_mpm_reaction(f: int, contact: qd.template()):
    mpm_rigid_func.func_mpm_reaction(
        f,
        contact.particles,
        contact.grid,
        contact.newton,
        contact.rigid_data,
        contact.dyn_state,
        contact.constraint_state,
        contact.params,
    )


@qd.kernel
def kernel_add_mpm_contact_force(contact: qd.template()):
    mpm_rigid_func.func_add_mpm_contact_force(contact.rigid_data, contact.dyn_state.links, contact.params)


@qd.kernel
def kernel_rigid_contact_adjoint(f: int, contact: qd.template()):
    mpm_rigid_adjoint_func.func_rigid_contact_adjoint(
        f,
        contact.particles,
        contact.grid,
        contact.newton,
        contact.linear,
        contact.rigid_data,
        contact.dyn_state.geoms,
        contact.dyn_state.links,
        contact.dyn_state.dofs,
        contact.particles_info,
        contact.dyn_info.geoms,
        contact.dyn_info.links,
        contact.rigid_info,
        contact.sdf_info,
        contact.params,
        contact.config,
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
        self._device.query.build(self._device.params)
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
