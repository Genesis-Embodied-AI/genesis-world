"""Rigid's dofs as a participant of the NewtonSolver, and the contact rows between the material point method (MPM) and
Rigid as a coupling.

This module binds the stage functions of `mpm_rigid_func` to the data of one scene. The Rigid unknown, its share of the
joint energy, its preconditioner and the contact coupling are described there. The coupling builds the rows of the
particles on the coupled geoms every substep, which MPM assembles with its own, and on the geoms of a Rigid without
dofs they act on the particle displacement alone.
"""

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.solvers.newton_solver import NewtonHost, kernel_assemble_participant

from . import mpm_rigid_adjoint_func, mpm_rigid_func

# Links in contact with MPM per environment whose contact stiffness the Rigid preconditioner takes in. The stiffness on
# any further link is left to the conjugate gradient iterations.
MAX_PRECONDITIONED_LINKS = 16


class RigidNewtonConfig(metaclass=array_class.AutoInitMeta):
    """Compile-time switches of Rigid's stages in the joint solve: the code paths a scene takes, never its sizes."""

    # The links info of Rigid is batched over the environments
    rigid_batch_links_info: bool
    # Rigid has constraint rows, which join the solve
    has_rigid_rows: bool


class MpmRigidContactConfig(metaclass=array_class.AutoInitMeta):
    """Compile-time switches of the stages of the MPM-Rigid contact rows."""

    # The links info of Rigid is batched over the environments
    rigid_batch_links_info: bool
    # Rigid has constraint rows, which join the solve
    has_rigid_rows: bool
    # The one switch of Rigid's collider configuration that the signed distance field (SDF) queries read. This config
    # stands in for it there, so that the kernels compile independently of the rest of that configuration.
    has_terrain: bool


@qd.data_oriented
class RigidNewtonParams:
    """Scene values of Rigid's stages in the joint solve, as device scalars."""

    def __init__(self, substep_dt: float) -> None:
        self.substep_dt = qd.ndarray(gs.qd_float, shape=())
        self.substep_dt.from_numpy(np.array(substep_dt, dtype=gs.np_float))


@qd.data_oriented
class RigidNewtonData:
    """Rigid unknowns of the joint Newton solve and the Rigid part of its conjugate gradient vectors.

    The Rigid unknown is x = dt^2 qacc per dof and environment, with Rigid's own cost scaled by dt^4 to the displacement
    units of the MPM energy. `qacc`, `Ma` and `cost` hold Rigid's evaluation of x. `r`, `z`, `p`, `ap` and `sol` are the
    Rigid rows of the conjugate gradient residual, preconditioned residual, direction, operator image and solution. Per
    link, `link_V` holds the spatial displacement of a dof vector and `link_F` accumulates a spatial force, linear then
    angular about the root center of mass, for the couplings acting on the links.
    """

    def __init__(self, n_dofs: int, n_links: int, n_envs: int) -> None:
        shape = (max(n_dofs, 1), n_envs)
        self.x = qd.ndarray(gs.qd_float, shape=shape)
        self.x_temp = qd.ndarray(gs.qd_float, shape=shape)
        self.dx = qd.ndarray(gs.qd_float, shape=shape)
        self.grad = qd.ndarray(gs.qd_float, shape=shape)
        self.qacc = qd.ndarray(gs.qd_float, shape=shape)
        self.Ma = qd.ndarray(gs.qd_float, shape=shape)
        self.r = qd.ndarray(gs.qd_float, shape=shape)
        self.z = qd.ndarray(gs.qd_float, shape=shape)
        self.p = qd.ndarray(gs.qd_float, shape=shape)
        self.ap = qd.ndarray(gs.qd_float, shape=shape)
        self.sol = qd.ndarray(gs.qd_float, shape=shape)
        self.cost = qd.ndarray(gs.qd_float, shape=(n_envs,))
        link_shape = (max(n_links, 1), n_envs)
        self.link_F = qd.ndarray(gs.qd_float, shape=(*link_shape, 6))
        self.link_V = qd.ndarray(gs.qd_float, shape=(*link_shape, 6))
        # The contact stiffness the couplings put on the links, which the preconditioner takes in by the Woodbury
        # identity on Rigid's factor A. Per link, its 6 x 6 spatial stiffness K_l, row-major. Per environment, the links
        # in contact (at most `n_contact_links`, the second dimension of `contact_links` that the kernels read), the
        # cdof map U of their chains with one column per spatial component, W = A^-1 U, and the lower-upper (LU) factor
        # of S = I + K U^T W with its row pivots. `solve_rhs` / `solve_out` stage one column of a solve.
        n_contact_links = min(max(n_links, 1), MAX_PRECONDITIONED_LINKS)
        n_cols = 6 * n_contact_links
        self.link_K = qd.ndarray(gs.qd_float, shape=(*link_shape, 36))
        self.contact_links = qd.ndarray(gs.qd_int, shape=(n_envs, n_contact_links))
        self.n_links_in_contact = qd.ndarray(gs.qd_int, shape=(n_envs,))
        self.wb_U = qd.ndarray(gs.qd_float, shape=(max(n_dofs, 1), n_envs, n_cols))
        self.wb_W = qd.ndarray(gs.qd_float, shape=(max(n_dofs, 1), n_envs, n_cols))
        self.wb_S = qd.ndarray(gs.qd_float, shape=(n_envs, n_cols, n_cols))
        self.wb_pivot = qd.ndarray(gs.qd_int, shape=(n_envs, n_cols))
        self.wb_y = qd.ndarray(gs.qd_float, shape=(n_envs, n_cols))
        self.solve_rhs = qd.ndarray(gs.qd_float, shape=shape)
        self.solve_out = qd.ndarray(gs.qd_float, shape=shape)


@qd.data_oriented
class RigidNewtonParticipant:
    """Device stages of Rigid's dofs in the joint solve, which own the Rigid data of the solve."""

    def __init__(self, rigid, n_envs: int):
        self.rigid_data = RigidNewtonData(rigid.n_dofs_, rigid.n_links, n_envs)
        self.dyn_state = rigid.dyn_state
        self.constraint_state = rigid.constraint_solver.constraint_state
        self.dyn_info = rigid.dyn_info
        self.rigid_info = rigid.rigid_info
        self.rigid_config = rigid.rigid_config
        self.sdf_info = rigid.collider.sdf.sdf_info
        self.config = RigidNewtonConfig(
            rigid_batch_links_info=rigid._options.batch_links_info,
            has_rigid_rows=not rigid._disable_constraint,
        )
        self.params = RigidNewtonParams(rigid.substep_dt)

    @qd.func(requires_top_level=True)
    def func_evaluate(self, f: int):
        mpm_rigid_func.func_rigid_evaluate(
            self.rigid_data,
            self.dyn_state,
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.params,
            self.rigid_config,
            self.config,
        )

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int):
        self.func_evaluate(f)
        mpm_rigid_func.func_rigid_gradient(
            self.rigid_data, self.dyn_state, self.constraint_state, self.params, self.config
        )

    @qd.func(requires_top_level=True)
    def func_factor(self):
        mpm_rigid_func.func_rigid_factor(
            self.rigid_data,
            self.dyn_state,
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self.config,
        )

    @qd.func(requires_top_level=True)
    def func_pcg_init(self):
        mpm_rigid_func.func_rigid_pcg_init(self.rigid_data)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        mpm_rigid_func.func_rigid_precondition(
            self.rigid_data, self.dyn_state, self.constraint_state, self.rigid_info, self.rigid_config, self.config
        )

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, pcg_scalars: qd.template()):
        mpm_rigid_func.func_rigid_pcg_init_direction(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_matvec(self, pcg_scalars: qd.template()):
        mpm_rigid_func.func_rigid_matvec(
            self.rigid_data, self.constraint_state, self.dyn_info, self.rigid_info, self.rigid_config, self.config
        )

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, pcg_scalars: qd.template()):
        mpm_rigid_func.func_rigid_matvec_dot(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, pcg_scalars: qd.template()):
        mpm_rigid_func.func_rigid_pcg_update_xr(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, pcg_scalars: qd.template()):
        mpm_rigid_func.func_rigid_pcg_dot_rz(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, pcg_scalars: qd.template()):
        mpm_rigid_func.func_rigid_pcg_direction(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        mpm_rigid_func.func_rigid_start_line_search(self.rigid_data, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        mpm_rigid_func.func_rigid_step_forward(self.rigid_data, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f: int, is_trial: bool):
        self.func_evaluate(f)
        mpm_rigid_func.func_rigid_energy(self.rigid_data, control, self.params, is_trial)


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
        self.sdf_info = rigid_solver.collider.sdf.sdf_info
        self.config = MpmRigidContactConfig(
            rigid_batch_links_info=rigid_solver._options.batch_links_info,
            has_rigid_rows=not rigid_solver._disable_constraint,
            has_terrain=rigid_solver.collider.collider_config.has_terrain,
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
        mpm_rigid_func.func_link_spatial_disp(
            self.rigid_data.x, self.rigid_data, self.dyn_state.dofs, self.dyn_info.links, self.config
        )
        mpm_rigid_func.func_contact_point_disp(
            f, self.particles, self.grid, self.newton, self.rigid_data, self.dyn_state.links, self.params
        )


@qd.kernel
def kernel_rigid_begin(rigid: qd.template()):
    mpm_rigid_func.func_rigid_begin(
        rigid.rigid_data, rigid.dyn_state.dofs, rigid.constraint_state, rigid.params, rigid.rigid_config, rigid.config
    )


@qd.kernel
def kernel_rigid_write_back(f: int, rigid: qd.template()):
    rigid.func_evaluate(f)
    mpm_rigid_func.func_rigid_write_back(rigid.rigid_data, rigid.constraint_state, rigid.config)


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
def kernel_rigid_adjoint_rhs(rigid: qd.template()):
    mpm_rigid_func.func_rigid_adjoint_rhs(rigid.rigid_data, rigid.constraint_state, rigid.params)


@qd.kernel
def kernel_rigid_adjoint_finish(rigid: qd.template()):
    mpm_rigid_func.func_rigid_adjoint_finish(rigid.rigid_data, rigid.constraint_state, rigid.params)


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


class RigidNewtonHost(NewtonHost):
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

    def newton_assemble(self, f, is_exact=False):
        kernel_assemble_participant(f, self._device)

    def newton_finish(self, f):
        kernel_rigid_write_back(f, self._device)

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

    def newton_finish(self, f):
        if self._has_rigid_dofs:
            kernel_mpm_reaction(f, self._device)

    def newton_after(self, f):
        if self._has_rigid_dofs:
            kernel_add_mpm_contact_force(self._device)

    def newton_adjoint_finish(self, f):
        """Pull the contact rows back into the adjoints of the particles and of Rigid's state."""
        kernel_rigid_contact_adjoint(f, self._device)
