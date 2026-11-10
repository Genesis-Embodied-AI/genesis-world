"""Rigid's dofs as a participant of the NewtonSolver, binding the stage functions of `newton_func` to the data of one
scene (see RigidSolver.build_newton_participant)."""

from collections.abc import Iterator

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.linear_pcg import PCGVectors
from genesis.engine.numerics.newton_control import NewtonControl
from genesis.engine.solvers.newton_solver import NewtonDevice, NewtonHost
from genesis.engine.solvers.rigid.constraint.solver import func_hessian_and_cholesky_factor_direct
from genesis.grad.bridge import no_grad
from genesis.utils.array_class import ConstraintState
from genesis.utils.data import DataItem, StaticConfig, iter_data

from . import newton_func


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
class RigidNewtonParticipant(PCGVectors, NewtonDevice):
    """Device stages of Rigid's dofs in the joint solve, which select the Rigid arrays the couplings read, the geoms
    they act on being `coupled_geoms_idx`."""

    def __init__(self, rigid, n_envs: int):
        self.rigid_data = newton_func.RigidNewtonScratch(
            sizes=dict(
                dofs=rigid.n_dofs_,
                links=rigid.n_links,
                envs=n_envs,
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
    def func_assemble(self, f: int, is_exact: qd.template()):
        self.func_evaluate(f)
        newton_func.func_rigid_gradient(
            self.rigid_data, self.dyn_state, self.constraint_state, self.newton_info, self.config
        )

    @qd.func(requires_top_level=True)
    def func_factor_begin(self):
        newton_func.func_clear_links_stiffness(self.rigid_data)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        # Rigid's own factor of its Hessian with the contact stiffness of the links, or its mass factor without rows
        if qd.static(self.config.has_rigid_rows):
            func_hessian_and_cholesky_factor_direct(
                self.constraint_state, self.rigid_info, self.rigid_config, compute_envelope=True, extra_hessian=self
            )

    # The contact stiffness of the links joins the factor of Rigid (see func_hessian_and_cholesky_factor_direct)
    has_extra_hessian = True

    @qd.func
    def func_add_island_hessian_batch(
        self, i_b: int, i_island: int, constraint_state: ConstraintState, rigid_config: qd.template()
    ):
        newton_func.func_add_links_stiffness_island_batch(
            i_b,
            i_island,
            self.rigid_data,
            self.dyn_state.dofs,
            constraint_state,
            self.dyn_info.links,
            rigid_config,
            self.config,
        )

    # The dofs as vectors of the joint PCG (see PCGVectors), numbered environment by environment so that the entries of
    # an environment are consecutive

    @property
    def pcg_vectors(self):
        return self.rigid_data

    @property
    def pcg_solution(self):
        return self.rigid_data.solution

    @property
    def pcg_rhs(self):
        return self.rigid_data.grad

    @qd.func
    def pcg_count(self):
        return self.rigid_data.x.shape[0] * self.rigid_data.x.shape[1]

    @qd.func
    def pcg_entry(self, i: int):
        n_dofs = self.rigid_data.x.shape[0]
        i_b = i // n_dofs
        return qd.Vector([i - i_b * n_dofs, i_b], dt=gs.qd_int), i_b

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        newton_func.func_rigid_precondition(
            self.rigid_data, self.dyn_state, self.constraint_state, self.rigid_info, self.rigid_config, self.config
        )

    @qd.func(requires_top_level=True)
    def func_matvec(self):
        newton_func.func_rigid_matvec(
            self.rigid_data, self.constraint_state, self.dyn_info, self.rigid_info, self.rigid_config, self.config
        )

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, f: int, control: NewtonControl):
        newton_func.func_rigid_start_line_search(self.rigid_data, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, f: int, control: NewtonControl):
        newton_func.func_rigid_step_forward(self.rigid_data, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, f: int, energies: qd.template()):
        self.func_evaluate(f)
        newton_func.func_rigid_energy(energies, self.rigid_data, self.newton_info)


@qd.kernel
def kernel_rigid_write_back(f: int, rigid: qd.template()):
    rigid.func_evaluate(f)
    newton_func.func_rigid_write_back(rigid.rigid_data, rigid.constraint_state, rigid.config)


@no_grad
@qd.kernel
def kernel_rigid_freeze(rigid: qd.template(), control: NewtonControl):
    for i_b in range(rigid.rigid_info.is_env_frozen.shape[0]):
        rigid.rigid_info.is_env_frozen[i_b] = control.has_failed_non_finite[i_b]


class RigidNewtonHost(NewtonHost):
    """Host stages of Rigid's dofs around the joint solve."""

    is_differentiable = True

    def __init__(self, rigid_solver, device: RigidNewtonParticipant):
        self._rigid_solver = rigid_solver
        self._device = device

    def newton_prepare(self, f):
        newton_func.kernel_rigid_begin(
            self._device.rigid_data,
            self._device.dyn_state,
            self._device.constraint_state,
            self._device.newton_info,
            self._device.rigid_config,
            self._device.config,
        )

    def newton_finish(self, f, control):
        kernel_rigid_write_back(f, self._device)

    def newton_after(self, f):
        self._rigid_solver.finish_deferred_solve()

    def newton_end(self, f, control):
        # The integration keeps the state of the environments the mask names (see func_copy_next_to_curr)
        kernel_rigid_freeze(self._device, control)
        self._rigid_solver.integrate_deferred_substep()

    def newton_backward(self, f, solve):
        """Reverse Rigid's constraint solve, with its adjoint solve run jointly with the other participants."""
        self._rigid_solver._constraint_force_grad(solve_adjoint=solve)

    def newton_adjoint_rhs(self, f):
        newton_func.kernel_rigid_adjoint_rhs(
            self._device.rigid_data, self._device.constraint_state, self._device.newton_info
        )

    def newton_adjoint_finish(self, f):
        newton_func.kernel_rigid_adjoint_finish(
            self._device.rigid_data, self._device.constraint_state, self._device.newton_info
        )
