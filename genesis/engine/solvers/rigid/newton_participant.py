"""Rigid's dofs as a participant of the NewtonSolver.

This module binds the stage functions of `newton_func` to the data of one scene. The Rigid unknown, its share of the
joint energy and its preconditioner are described there. The host stages start the solve from Rigid's warm start, hand
the converged qacc back to Rigid and integrate its deferred substep (see RigidSolver.build_newton_participant).
"""

from collections.abc import Iterator

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.newton_solver import NewtonDevice, NewtonHost, kernel_assemble_participant
from genesis.grad.bridge import no_grad
from genesis.utils.data import DataItem, StaticConfig, iter_data

from . import newton_func

# Links in contact with MPM per environment whose contact stiffness the Rigid preconditioner takes in. The stiffness on
# any further link is left to the conjugate gradient iterations.
MAX_PRECONDITIONED_LINKS = 16


class RigidNewtonConfig(StaticConfig):
    """Compile-time switches of Rigid's stages in the joint solve: the code paths a scene takes, never its sizes."""

    # The links info of Rigid is batched over the environments
    rigid_batch_links_info: bool
    # Rigid has constraint rows, which join the solve
    has_rigid_rows: bool
    # A coupled geom is a terrain (see RigidSolver.has_coupled_terrain). This config stands in for Rigid's collider
    # configuration in the signed distance field (SDF) queries, which read this one switch of it.
    has_terrain: bool


@qd.data_oriented
class RigidNewtonParticipant(NewtonDevice):
    """Device stages of Rigid's dofs in the joint solve, which own the Rigid data of the solve.

    It is the one place that selects the Rigid arrays the couplings read, and `coupled_geoms_idx` lists the geoms the
    couplings act on."""

    def __init__(self, rigid, n_envs: int):
        self.rigid_data = newton_func.RigidNewtonScratch(
            sizes=dict(
                dofs=rigid.n_dofs_,
                links=rigid.n_links,
                envs=n_envs,
                contact_links=min(max(rigid.n_links, 1), MAX_PRECONDITIONED_LINKS),
            )
        )
        self.coupled_geoms_idx = np.array([geom.idx for geom in rigid.geoms if geom.needs_coup], dtype=gs.np_int)
        self.dyn_state = rigid.dyn_state
        self.constraint_state = rigid.constraint_solver.constraint_state
        self.dyn_info = rigid.dyn_info
        self.rigid_info = rigid.rigid_info
        self.rigid_config = rigid.rigid_config
        self.sdf_info = rigid.collider.collider_info.sdf
        self.config = RigidNewtonConfig(
            rigid_batch_links_info=rigid._options.batch_links_info,
            has_rigid_rows=not rigid._disable_constraint,
            has_terrain=rigid.has_coupled_terrain,
        )
        self.newton_info = newton_func.RigidNewtonInfo(values=dict(substep_dt=rigid.substep_dt))

    @property
    def data(self) -> Iterator[DataItem]:
        yield from iter_data(self, "rigid_data")
        yield from iter_data(self, "newton_info")

    @property
    def geoms_init_AABB(self):
        return self.rigid_info.geoms_init_AABB

    @qd.func(requires_top_level=True)
    def func_evaluate(self, f: int):
        newton_func.func_rigid_evaluate(
            self.rigid_data,
            self.dyn_state,
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.newton_info,
            self.rigid_config,
            self.config,
        )

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int):
        self.func_evaluate(f)
        newton_func.func_rigid_gradient(
            self.rigid_data, self.dyn_state, self.constraint_state, self.newton_info, self.config
        )

    @qd.func(requires_top_level=True)
    def func_factor(self):
        newton_func.func_rigid_factor(
            self.rigid_data,
            self.dyn_state,
            self.constraint_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self.config,
        )

    @qd.func(requires_top_level=True)
    def func_pcg_init(self, pcg_scalars: qd.template(), is_adjoint: qd.template()):
        newton_func.func_rigid_pcg_init(self.rigid_data, pcg_scalars, is_adjoint)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        newton_func.func_rigid_precondition(
            self.rigid_data, self.dyn_state, self.constraint_state, self.rigid_info, self.rigid_config, self.config
        )

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, pcg_scalars: qd.template()):
        newton_func.func_rigid_pcg_init_direction(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_matvec(self, pcg_scalars: qd.template()):
        newton_func.func_rigid_matvec(
            self.rigid_data, self.constraint_state, self.dyn_info, self.rigid_info, self.rigid_config, self.config
        )

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, pcg_scalars: qd.template()):
        newton_func.func_rigid_matvec_dot(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, pcg_scalars: qd.template()):
        newton_func.func_rigid_pcg_update_xr(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, pcg_scalars: qd.template()):
        newton_func.func_rigid_pcg_dot_rz(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, pcg_scalars: qd.template()):
        newton_func.func_rigid_pcg_direction(self.rigid_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        newton_func.func_rigid_start_line_search(self.rigid_data, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        newton_func.func_rigid_step_forward(self.rigid_data, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f: int, is_trial: qd.template()):
        self.func_evaluate(f)
        newton_func.func_rigid_energy(self.rigid_data, control, self.newton_info, is_trial)


@qd.kernel
def kernel_rigid_begin(rigid: qd.template()):
    newton_func.func_rigid_begin(
        rigid.rigid_data,
        rigid.dyn_state.dofs,
        rigid.constraint_state,
        rigid.newton_info,
        rigid.rigid_config,
        rigid.config,
    )


@qd.kernel
def kernel_rigid_write_back(f: int, rigid: qd.template()):
    rigid.func_evaluate(f)
    newton_func.func_rigid_write_back(rigid.rigid_data, rigid.constraint_state, rigid.config)


@no_grad
@qd.kernel
def kernel_rigid_freeze(rigid: qd.template(), control: qd.template()):
    for i_b in range(rigid.rigid_info.is_env_frozen.shape[0]):
        rigid.rigid_info.is_env_frozen[i_b] = control.has_failed_non_finite[i_b]


@qd.kernel
def kernel_rigid_adjoint_rhs(rigid: qd.template()):
    newton_func.func_rigid_adjoint_rhs(rigid.rigid_data, rigid.constraint_state, rigid.newton_info)


@qd.kernel
def kernel_rigid_adjoint_finish(rigid: qd.template()):
    newton_func.func_rigid_adjoint_finish(rigid.rigid_data, rigid.constraint_state, rigid.newton_info)


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

    def newton_finish(self, f, control):
        kernel_rigid_write_back(f, self._device)

    def newton_after(self, f):
        self._rigid_solver.finish_deferred_solve()

    def newton_end(self, f, control):
        # The integration keeps the state of the environments the mask names (see func_copy_next_to_curr)
        kernel_rigid_freeze(self._device, control)
        self._rigid_solver.integrate_deferred_substep()

    def newton_backward(self, f, solve):
        """Reverse Rigid's constraint solve, with its adjoint solve run jointly with the other participants.

        `solve` runs the joint adjoint solve."""
        self._rigid_solver._constraint_force_grad(solve_adjoint=solve)

    def newton_adjoint_rhs(self, f):
        kernel_rigid_adjoint_rhs(self._device)

    def newton_adjoint_finish(self, f):
        kernel_rigid_adjoint_finish(self._device)
