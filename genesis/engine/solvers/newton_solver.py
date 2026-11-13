"""
Newton solver of the substeps of the implicit material point method (MPM) solver.

The solver runs the Newton iterations, the preconditioned conjugate gradient and the line search. The terms of the
energy and their solve come from the solvers that own them: the MPM solver for the nodes of its grid (see
'func_mpm_assemble' in mpm_solver.py), the rigid solver for its dofs (see rigid/global_newton.py), and the coupling for
the contacts between them (see couplers/mpm_rigid_global_newton.py).
"""

from collections.abc import Iterator
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.engine.couplers.mpm_rigid_global_newton import (
    func_add_contact_link_terms,
    func_assemble_mpm_rigid_coupling,
    func_contact_matvec,
    func_contact_terms,
    func_mark_constrained_links,
)
from genesis.engine.numerics.bcoo import assemble_bcoo_matrix, func_add_by_key
from genesis.engine.solvers.rigid.constraint.data import ConstraintState
from genesis.engine.solvers.rigid.data import DynInfo, DynState, RigidInfo
from genesis.engine.solvers.rigid.global_newton import (
    NewtonRigidDofs,
    NewtonRigidLinks,
    RigidContactStiffness,
    func_link_force_to_dofs,
    func_link_spatial_disp,
    func_rigid_assemble_gradient,
    func_rigid_factor,
    func_rigid_finish,
    func_rigid_init_line_search,
    func_rigid_init_substep,
    func_rigid_matvec,
    func_rigid_solve_batch,
    func_rigid_step_forward,
    func_rigid_update_constraint,
)
from genesis.options.solvers import LegacyCouplerOptions, NewtonOptions
from genesis.utils.data import ENVS, Config, Data, DataItem, DataKind, field, iter_data
from genesis.utils.misc import qd_isfinite, qd_to_numpy

from .base_solver import ErrorCode, Solver
from .mpm_solver import (
    MPMInfo,
    MPMSparseGrid,
    MPMState,
    NewtonRows,
    func_mpm_assemble,
    func_mpm_init_line_search,
    func_mpm_kinetic_energy,
    func_mpm_matvec,
    func_mpm_precond_invert,
    func_mpm_step_forward,
    func_particle_energy,
    grow_linear_system,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# The arrays of the dofs and of the links of the rigid solver in the Newton solve, which a solve without them leaves out
RIGID = "rigid"


class NewtonErrorCode(ErrorCode):
    """The error flags the Newton solver writes in its errno (see 'Solver.register_errors')."""

    INVALID_SOLVE_NAN = auto()


class NewtonFailure(IntEnum):
    """The reason the Newton solve of an environment stopped short of convergence."""

    NONE = 0
    ITERATIONS = 1
    NON_FINITE = 2


class NewtonEnvs(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The progress and the scalars of the Newton solve of every environment.

    `is_active` says whether its Newton iterations go on, `is_converged` whether its last direction moved every particle
    and every dof by less than the tolerance, and `failure` why it stopped short of convergence (see NewtonFailure).
    `energy_start` is the energy at the start point of the line search, `energy_trial` at its trial point of step
    `alpha`, and `max_displacement` the largest update of the displacement of a particle or of a dof along the
    direction. The scalars of the preconditioned conjugate gradient (PCG) follow: `rz` is r . z, `rz_tol` the value
    below which it stops, and `pcg_alpha` and `pcg_beta` its steps. `rigid_cost` is the cost of the constraint solve of
    the rigid solver at the current iterate of its dofs (see 'func_rigid_update_constraint' in rigid/global_newton.py).
    """

    is_active: qd.Tensor = field(bool)
    is_converged: qd.Tensor = field(bool)
    failure: qd.Tensor = field(int)
    n_iterations: qd.Tensor = field(int)
    energy_start: qd.Tensor = field(float)
    energy_trial: qd.Tensor = field(float)
    max_displacement: qd.Tensor = field(float)
    alpha: qd.Tensor = field(float)
    is_line_search_active: qd.Tensor = field(bool)
    n_line_search_iterations: qd.Tensor = field(int)
    is_pcg_active: qd.Tensor = field(bool)
    rz: qd.Tensor = field(float)
    rz_new: qd.Tensor = field(float)
    rz_tol: qd.Tensor = field(float)
    p_dot_ap: qd.Tensor = field(float)
    pcg_alpha: qd.Tensor = field(float)
    pcg_beta: qd.Tensor = field(float)
    rigid_cost: qd.Tensor = field(float, labels=RIGID)


class NewtonLoops(Data, kind=DataKind.SCRATCH, shape=()):
    """
    The flags that keep the loops of the Newton solve running as long as an environment iterates on, with the iterations
    left to the preconditioned conjugate gradient (PCG).

    The flags are ndarrays on every backend, since a loop of a graph tests the same ndarray on every call. Between two
    checks (see 'NewtonOptions.failure_check_interval'), `has_failed_iterations` latches a solve that reached its
    iteration cap.
    """

    is_newton_running: qd.Tensor = field(qd.i32, ndarray=True)
    is_line_search_running: qd.Tensor = field(qd.i32, ndarray=True)
    is_pcg_running: qd.Tensor = field(qd.i32, ndarray=True)
    n_pcg_iterations_left: qd.Tensor = field(int, ndarray=True)
    has_failed_iterations: qd.Tensor = field(int, ndarray=True)


class NewtonInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """The constants of the Newton solve (see NewtonOptions), the tolerance on the displacement of a particle or a dof
    being the tolerance on its velocity over a substep."""

    max_iterations: qd.Tensor = field(int)
    line_search_max_iterations: qd.Tensor = field(int)
    line_search_shrink: qd.Tensor = field(float)
    displacement_tol: qd.Tensor = field(float)
    pcg_max_iterations: qd.Tensor = field(int)
    pcg_tol: qd.Tensor = field(float)


class NewtonConfig(Config):
    """The compile-time configuration of the kernels of the Newton solver."""

    # Whether the dofs of the rigid solver join the unknowns (see 'RigidSolver.is_solved_in_global_newton')
    has_rigid_dofs: bool
    # Whether the rigid solver has constraint rows. Their cost then joins the energy and the factor of the Hessian of
    # its Newton constraint solver preconditions its dofs, its Gauss cost and the factor of its mass matrix otherwise.
    has_rigid_rows: bool


class NewtonData(Data):
    """The arrays of the Newton solver of a fixed size, allocated once its scene is built."""

    newton_envs: NewtonEnvs
    newton_loops: NewtonLoops
    newton_info: NewtonInfo
    rigid_dofs: NewtonRigidDofs = field(labels=RIGID)
    rigid_links: NewtonRigidLinks = field(labels=RIGID)
    errno: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)


class NewtonSolver(Solver):
    """
    Solver of the substeps of the implicit material point method (MPM) solver by Newton iterations on their incremental
    potential.

    The unknowns of a substep are the displacement of the nodes of the sparse grid of the MPM solver over it (see
    MPMSparseGrid), and when the rigid geoms couple with the particles, the accelerations of the dofs of the rigid
    solver, whose constraint solve then joins the energy (see 'RigidSolver.is_solved_in_global_newton'). Every iteration
    solves the projected Hessian against the gradient by a preconditioned conjugate gradient (PCG) and searches a step
    along the direction that lowers the energy, every environment iterating until its own convergence. The solve runs
    once every solver has prepared its substep and the coupler has run, and before the solvers finish it, since the MPM
    solver moves its particles by the displacement solved and the rigid solver integrates the accelerations solved.
    """

    # Its whole state lives in the arrays 'data' yields (see 'CheckpointMixin')
    is_checkpointable = True

    def __init__(self, scene: "Scene", sim: "Simulator", options: NewtonOptions) -> None:
        super().__init__(scene, sim, options)
        self.data_manager: NewtonData | None = None
        self.newton_envs: NewtonEnvs | None = None
        self.newton_loops: NewtonLoops | None = None
        self.newton_info: NewtonInfo | None = None
        self.newton_config: NewtonConfig | None = None
        self.rigid_dofs: NewtonRigidDofs | None = None
        self.rigid_links: NewtonRigidLinks | None = None
        # The stiffness of the contacts that the factor of the rigid solver adds, made once the rigid solver is built
        self._rigid_contact_stiffness: RigidContactStiffness | None = None
        # The rows of the linear system, rebuilt with a larger capacity when the grid of the MPM solver outgrows them,
        # and the rows they replaced, kept referenced with their arrays freed (see 'grow_linear_system')
        self.newton_rows: NewtonRows | None = None
        self._newton_rows_replaced: list[NewtonRows] = []

    @property
    def is_active(self) -> bool:
        mpm_solver = self._sim.mpm_solver
        return mpm_solver.is_active and mpm_solver._use_implicit_solver

    def build(self) -> None:
        super().build()
        if not self.is_active:
            return
        if self._requires_grad:
            gs.raise_exception(
                "Differentiable simulation ('requires_grad=True') is not supported by the implicit MPM solver: set "
                "MPMOptions(use_implicit_solver=False)."
            )
        if not isinstance(self._sim.coupler.options, LegacyCouplerOptions):
            gs.raise_exception(
                "The implicit MPM solver requires LegacyCouplerOptions, got "
                f"{type(self._sim.coupler.options).__name__}."
            )
        rigid_solver = self._sim.rigid_solver
        rigid_options = self._scene.options.rigid
        has_rigid_dofs = rigid_solver.is_solved_in_global_newton
        if has_rigid_dofs:
            # The Newton solve preconditions the dofs of the rigid solver by the factor of the Hessian of its Newton
            # constraint solver, over the rows of the pyramidal friction cone, and its solution is final for every dof,
            # which a sleeping body would discard and a noslip pass after it would change
            for is_supported, requirement in (
                (rigid_options.friction_cone == gs.friction_cone.pyramidal, "friction_cone=gs.friction_cone.pyramidal"),
                (
                    rigid_options.constraint_solver == gs.constraint_solver.Newton,
                    "constraint_solver=gs.constraint_solver.Newton",
                ),
                (not rigid_options.use_hibernation, "use_hibernation=False"),
                (rigid_options.noslip_iterations == 0, "noslip_iterations=0"),
            ):
                if not is_supported:
                    gs.raise_exception(
                        f"Coupling the rigid solver with the implicit MPM solver requires RigidOptions({requirement}), "
                        "or else LegacyCouplerOptions(rigid_mpm=False)."
                    )
        self.newton_config = NewtonConfig(
            has_rigid_dofs=has_rigid_dofs,
            has_rigid_rows=not rigid_options.disable_constraint,
        )
        self.data_manager = NewtonData(
            sizes={"envs": self._B, "dofs": max(rigid_solver.n_dofs, 1), "links": max(rigid_solver.n_links, 1)},
            values=dict(
                newton_info=dict(
                    max_iterations=self._options.max_iterations,
                    line_search_max_iterations=self._options.line_search_max_iterations,
                    line_search_shrink=self._options.line_search_shrink,
                    displacement_tol=self._options.velocity_tol * self._sim.substep_dt,
                    pcg_max_iterations=self._options.pcg_max_iterations,
                    pcg_tol=self._options.pcg_tol,
                )
            ),
            unused={RIGID: not has_rigid_dofs},
        )
        # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
        self.register_errors(
            self.data_manager.errno,
            {
                NewtonErrorCode.INVALID_SOLVE_NAN: lambda: (
                    "The Newton solve of the implicit MPM solver produced values that are not finite. Please decrease "
                    "the simulation timestep or adjust the material parameters."
                ),
            },
        )
        self.newton_envs = self.data_manager.newton_envs
        self.newton_loops = self.data_manager.newton_loops
        self.newton_info = self.data_manager.newton_info
        self.rigid_dofs = self.data_manager.rigid_dofs
        self.rigid_links = self.data_manager.rigid_links

    @property
    def data(self) -> Iterator[DataItem]:
        if self.data_manager is not None:
            yield from iter_data(self.data_manager, "newton_info")
            yield from iter_data(self.data_manager, "newton_envs")
            yield from iter_data(self.data_manager, "newton_loops")
            if self.newton_config.has_rigid_dofs:
                yield from iter_data(self.data_manager, "rigid_dofs")
                yield from iter_data(self.data_manager, "rigid_links")
            yield from iter_data(self.data_manager, "errno", DataKind.STATE)
        if self.newton_rows is not None:
            yield from iter_data(self, "newton_rows")

    def substep_pre_coupling(self) -> None:
        pass

    def substep_post_coupling(self) -> None:
        """
        Solve the substep of the implicit material point method (MPM) solver before the solvers finish it, with that of
        the rigid solver when it is solved in the global Newton solve (see 'RigidSolver.is_solved_in_global_newton').
        """
        if not self.is_active:
            return
        mpm_solver = self._sim.mpm_solver
        rigid_solver = self._sim.rigid_solver
        mpm_state, mpm_info, sparse_grid = mpm_solver.mpm_state, mpm_solver.mpm_info, mpm_solver.sparse_grid
        dyn_state, dyn_info, rigid_info = rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_info
        constraint_state = rigid_solver.constraint_solver.constraint_state
        errno = self._errno
        if self.newton_config.has_rigid_dofs and self._rigid_contact_stiffness is None:
            self._rigid_contact_stiffness = RigidContactStiffness(self.rigid_links, dyn_state.dofs, dyn_info.links)
        self.newton_rows = grow_linear_system(
            int(qd_to_numpy(sparse_grid.n_blocks)) * 64,
            int(qd_to_numpy(mpm_state.sorted_particles.n_tasks)),
            self.newton_rows,
            self._newton_rows_replaced,
        )
        args_state = (mpm_state, sparse_grid, self.rigid_dofs, self.rigid_links, dyn_state, constraint_state)
        args_info = (mpm_info, dyn_info, rigid_info)
        args_config = (mpm_solver.mpm_config, rigid_solver.rigid_config, self.newton_config)
        kernel_init_substep(self.newton_envs, self.newton_loops, *args_state, *args_info, *args_config, errno)
        for _ in range(self._options.max_iterations):
            if mpm_solver.mpm_config.has_rigid_contact:
                kernel_assemble_mpm_rigid_coupling(
                    mpm_state,
                    sparse_grid,
                    self.rigid_dofs,
                    self.rigid_links,
                    dyn_state,
                    mpm_info,
                    dyn_info,
                    rigid_solver.rigid_config,
                    self.newton_config,
                )
            kernel_assemble_mpm(self.newton_rows, mpm_state, sparse_grid, mpm_info, mpm_solver.mpm_config)
            assemble_bcoo_matrix(self.newton_rows.matrix, block_scalars=9)
            if self.newton_config.has_rigid_dofs:
                kernel_assemble_rigid(
                    self.newton_envs,
                    *args_state,
                    *args_info,
                    *args_config,
                    self._rigid_contact_stiffness,
                )
            kernel_pcg_start(
                self.newton_rows,
                self.newton_envs,
                self.newton_loops,
                *args_state,
                self.newton_info,
                *args_info,
                *args_config,
            )
            kernel_pcg_loop(
                self.newton_rows, self.newton_envs, self.newton_loops, *args_state, *args_info, *args_config
            )
            kernel_start_line_search(
                self.newton_rows,
                self.newton_envs,
                self.newton_loops,
                *args_state,
                self.newton_info,
                *args_info,
                *args_config,
            )
            while True:
                kernel_line_search_trial(
                    self.newton_envs, self.newton_loops, *args_state, self.newton_info, *args_info, *args_config
                )
                if not qd_to_numpy(self.newton_loops.is_line_search_running):
                    break
            kernel_end_iteration(self.newton_envs, self.newton_loops, self.newton_info)
            if not qd_to_numpy(self.newton_loops.is_newton_running):
                break
        kernel_flag_failures(self.newton_envs, self.newton_loops, errno)
        if self.newton_config.has_rigid_dofs:
            kernel_finish_rigid(self.newton_envs, *args_state, *args_info, *args_config, errno)
        if self._sim.cur_substep_global % self._options.failure_check_interval == 0:
            if qd_to_numpy(self.newton_loops.has_failed_iterations):
                kernel_clear_failed_iterations(self.newton_loops)
                gs.logger.warning(
                    "The Newton solve of the implicit MPM solver stopped before converging: raise "
                    "NewtonOptions(max_iterations) or loosen NewtonOptions(velocity_tol)."
                )


# ------------------------------------------------------------------------------------
# ------------------------------------- energy ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def func_add_energies(
    energies: qd.Tensor,
    envs_mask: qd.Tensor,
    rigid_cost: qd.Tensor,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Add the energy of the Newton solve at the current iterate into the entry of every environment selected.

    `envs_mask` selects the environments, and `energies` holds their entries. The energy is the incremental potential of
    the particles, of their contacts with the rigid geoms and of the nodes, the particles and the nodes of an
    environment following one another, and when the dofs of the rigid solver join the unknowns, the cost of the rigid
    solver at their iterate, scaled by dt^4 to the units of the incremental potential.
    """
    if qd.static(newton_config.has_rigid_dofs):
        func_link_spatial_disp(rigid_dofs.x, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
    n_valid = mpm_state.sorted_particles.n_valid[None]
    qd.loop_config(block_dim=256)
    for i_s in range((n_valid + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        energy = gs.qd_float(0.0)
        if i_s < n_valid:
            energy_contact = gs.qd_float(0.0)
            if qd.static(mpm_config.has_rigid_contact):
                energy_contact, gradient_contact, hessian_contact = func_contact_terms(
                    i_s,
                    mpm_state,
                    sparse_grid,
                    rigid_links,
                    dyn_state,
                    mpm_info,
                    dyn_info,
                    newton_config.has_rigid_dofs,
                )
            i_b, energy_particle = func_particle_energy(i_s, energy_contact, mpm_state, sparse_grid, mpm_info)
            if envs_mask[i_b]:
                i_b_key = i_b
                energy = energy_particle
        func_add_by_key(i_b_key, energy, energies)
    func_mpm_kinetic_energy(energies, envs_mask, sparse_grid, mpm_info)
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_update_constraint(
            rigid_cost, rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, newton_config.has_rigid_rows
        )
        dt = rigid_info.substep_dt[None]
        for i_b in range(energies.shape[0]):
            if envs_mask[i_b]:
                energies[i_b] = energies[i_b] + dt * dt * dt * dt * rigid_cost[i_b]


@qd.kernel
def kernel_init_substep(
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
    errno: qd.Tensor,
):
    """
    Initialize the Newton solve of the substep in every environment at zero displacement of the nodes, but those whose
    solve failed earlier in the step, and evaluate its energy there.

    The dofs of the rigid solver start as its constraint solve starts (see 'func_rigid_init_substep' in
    rigid/global_newton.py), and the links that the particles touch are constrained over the substep (see
    'func_mark_constrained_links' in couplers/mpm_rigid_global_newton.py).
    """
    for _ in range(1):
        newton_loops.is_newton_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        is_active = (errno[i_b] & NewtonErrorCode.INVALID_SOLVE_NAN) == 0
        newton_envs.is_active[i_b] = is_active
        newton_envs.is_converged[i_b] = False
        newton_envs.failure[i_b] = NewtonFailure.NONE
        newton_envs.n_iterations[i_b] = 0
        newton_envs.energy_start[i_b] = 0.0
        if is_active:
            qd.atomic_max(newton_loops.is_newton_running[None], 1)
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_init_substep(
            rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, newton_config.has_rigid_rows
        )
        func_mark_constrained_links(mpm_state, dyn_state, dyn_info)
    func_add_energies(
        newton_envs.energy_start,
        newton_envs.is_active,
        newton_envs.rigid_cost,
        mpm_state,
        sparse_grid,
        rigid_dofs,
        rigid_links,
        dyn_state,
        constraint_state,
        mpm_info,
        dyn_info,
        rigid_info,
        mpm_config,
        rigid_config,
        newton_config,
    )


# ------------------------------------------------------------------------------------
# ------------------------------------ assembly --------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_assemble_mpm_rigid_coupling(
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """Stage the terms of the contacts of the particles with the rigid geoms at the current iterate (see
    'func_assemble_mpm_rigid_coupling' in couplers/mpm_rigid_global_newton.py)."""
    func_assemble_mpm_rigid_coupling(
        mpm_state,
        sparse_grid,
        rigid_dofs,
        rigid_links,
        dyn_state,
        mpm_info,
        dyn_info,
        rigid_config,
        newton_config.has_rigid_dofs,
    )


@qd.kernel
def kernel_assemble_mpm(
    newton_rows: NewtonRows,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    mpm_info: MPMInfo,
    mpm_config: qd.template(),
):
    """Assemble the rows of the nodes of the MPM solver at the current iterate (see 'func_mpm_assemble' in
    mpm_solver.py)."""
    func_mpm_assemble(newton_rows, mpm_state, sparse_grid, mpm_info, mpm_config)


@qd.kernel
def kernel_assemble_rigid(
    newton_envs: NewtonEnvs,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
    rigid_contact_stiffness: qd.template(),
):
    """
    Assemble the gradient of the energy in the dofs of the rigid solver at the current iterate, and factor their block
    of the Hessian.

    The assembly takes the terms of the contacts of the particles that the coupling staged (see
    'kernel_assemble_mpm_rigid_coupling'), and the factor preconditions the dofs in the linear solve. The gradient holds
    that of the cost of the rigid solver and the forces of the contacts of the particles on the links (see
    'func_add_contact_link_terms' in couplers/mpm_rigid_global_newton.py). The block holds the Hessian of the rigid
    solver and the stiffness of these contacts (see RigidContactStiffness).
    """
    has_rigid_rows = qd.static(newton_config.has_rigid_rows)
    func_rigid_update_constraint(
        newton_envs.rigid_cost, rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, has_rigid_rows
    )
    func_rigid_assemble_gradient(rigid_dofs, dyn_state, constraint_state, rigid_info, has_rigid_rows)
    func_add_contact_link_terms(mpm_state, rigid_links, dyn_state, dyn_info)
    func_link_force_to_dofs(rigid_dofs.gradient, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
    func_rigid_factor(constraint_state, rigid_info, rigid_config, rigid_contact_stiffness, has_rigid_rows)


# ------------------------------------------------------------------------------------
# -------------------------- preconditioned conjugate gradient -----------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_pcg_start(
    newton_rows: NewtonRows,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: NewtonInfo,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Start the preconditioned conjugate gradient (PCG) of the environments still iterating from a zero solution.

    The preconditioner inverts the diagonal blocks of the rows (see 'func_mpm_precond_invert' in mpm_solver.py) and the
    block of the dofs by its factor. An environment stops once r . z falls below `pcg_tol` times its value at the start,
    and fails its Newton solve once r . z is not finite, which a value that is not finite in its rows or its dofs makes.
    """
    func_mpm_precond_invert(newton_rows)
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.is_pcg_active[i_b] = newton_envs.is_active[i_b]
        newton_envs.rz[i_b] = 0.0
        newton_envs.p_dot_ap[i_b] = 0.0
    n_rows = newton_rows.n_rows[None]
    qd.loop_config(block_dim=256)
    for i_r in range((n_rows + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        rz = gs.qd_float(0.0)
        if i_r < n_rows:
            i_b = newton_rows.rows_env[i_r]
            r = newton_rows.gradient[i_r]
            z = newton_rows.jacobi[i_r] @ r
            newton_rows.solution[i_r] = qd.Vector.zero(gs.qd_float, 3)
            newton_rows.r[i_r] = r
            newton_rows.z[i_r] = z
            newton_rows.p[i_r] = z
            newton_rows.Ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
            if newton_envs.is_pcg_active[i_b]:
                i_b_key = i_b
                rz = r.dot(z)
        func_add_by_key(i_b_key, rz, newton_envs.rz)
    if qd.static(newton_config.has_rigid_dofs):
        for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
            rigid_dofs.r[i_d, i_b] = rigid_dofs.gradient[i_d, i_b]
            rigid_dofs.solution[i_d, i_b] = 0.0
            rigid_dofs.Ap[i_d, i_b] = 0.0
        for i_b in range(rigid_dofs.x.shape[1]):
            func_rigid_solve_batch(
                i_b,
                rigid_dofs.r,
                rigid_dofs.z,
                dyn_state,
                constraint_state,
                rigid_info,
                rigid_config,
                newton_config.has_rigid_rows,
            )
        for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
            rigid_dofs.p[i_d, i_b] = rigid_dofs.z[i_d, i_b]
            if newton_envs.is_pcg_active[i_b]:
                qd.atomic_add(newton_envs.rz[i_b], rigid_dofs.r[i_d, i_b] * rigid_dofs.z[i_d, i_b])
    for _ in range(1):
        newton_loops.is_pcg_running[None] = 0
        newton_loops.n_pcg_iterations_left[None] = newton_info.pcg_max_iterations[None]
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.rz_tol[i_b] = newton_info.pcg_tol[None] * qd.abs(newton_envs.rz[i_b])
        if newton_envs.is_pcg_active[i_b]:
            if not qd_isfinite(newton_envs.rz[i_b]):
                newton_envs.is_pcg_active[i_b] = False
                newton_envs.failure[i_b] = NewtonFailure.NON_FINITE
            elif newton_envs.rz[i_b] > 0.0:
                qd.atomic_max(newton_loops.is_pcg_running[None], 1)
            else:
                newton_envs.is_pcg_active[i_b] = False


@qd.func(requires_top_level=True)
def func_pcg_iteration(
    newton_rows: NewtonRows,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Run one iteration of the preconditioned conjugate gradient (PCG) of the environments still solving.

    The product of the Hessian with the search direction is that of the rows (see 'func_mpm_matvec' in mpm_solver.py),
    in the energy form p . Ap of every environment without the dofs. When the dofs of the rigid solver join the
    unknowns, it adds their block (see 'func_rigid_matvec' in rigid/global_newton.py) and the coupling of the contacts
    of the particles with the links (see 'func_contact_matvec' in couplers/mpm_rigid_global_newton.py).
    """
    func_mpm_matvec(
        newton_rows, newton_envs.is_pcg_active, newton_envs.p_dot_ap, qd.static(not newton_config.has_rigid_dofs)
    )
    n_rows = newton_rows.n_rows[None]
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_matvec(rigid_dofs, constraint_state, rigid_info, rigid_config, newton_config.has_rigid_rows)
        func_link_spatial_disp(rigid_dofs.p, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
        func_contact_matvec(
            newton_rows,
            newton_envs.is_pcg_active,
            mpm_state,
            sparse_grid,
            rigid_links,
            dyn_state,
            mpm_info,
            dyn_info,
        )
        func_link_force_to_dofs(rigid_dofs.Ap, rigid_links, dyn_state.dofs, dyn_info.links, rigid_config)
        # p . Ap over the rows and over the dofs, once Ap holds the whole product
        qd.loop_config(block_dim=256)
        for i_r in range((n_rows + 31) // 32 * 32):
            i_b_key = gs.qd_int(-1)
            energy = gs.qd_float(0.0)
            if i_r < n_rows:
                i_b = newton_rows.rows_env[i_r]
                if newton_envs.is_pcg_active[i_b]:
                    i_b_key = i_b
                    energy = newton_rows.p[i_r].dot(newton_rows.Ap[i_r])
            func_add_by_key(i_b_key, energy, newton_envs.p_dot_ap)
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            if newton_envs.is_pcg_active[i_b]:
                qd.atomic_add(newton_envs.p_dot_ap[i_b], rigid_dofs.p[i_d, i_b] * rigid_dofs.Ap[i_d, i_b])
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_pcg_active[i_b]:
            p_dot_ap = newton_envs.p_dot_ap[i_b]
            pcg_alpha = gs.qd_float(0.0)
            if not qd_isfinite(p_dot_ap):
                newton_envs.is_pcg_active[i_b] = False
                newton_envs.failure[i_b] = NewtonFailure.NON_FINITE
            elif qd.abs(p_dot_ap) > 0.0:
                pcg_alpha = newton_envs.rz[i_b] / p_dot_ap
            newton_envs.pcg_alpha[i_b] = pcg_alpha
        newton_envs.rz_new[i_b] = 0.0
        newton_envs.p_dot_ap[i_b] = 0.0
    qd.loop_config(block_dim=256)
    for i_r in range((n_rows + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        rz = gs.qd_float(0.0)
        if i_r < n_rows:
            i_b = newton_rows.rows_env[i_r]
            if newton_envs.is_pcg_active[i_b]:
                pcg_alpha = newton_envs.pcg_alpha[i_b]
                r = newton_rows.r[i_r] - pcg_alpha * newton_rows.Ap[i_r]
                z = newton_rows.jacobi[i_r] @ r
                newton_rows.solution[i_r] = newton_rows.solution[i_r] + pcg_alpha * newton_rows.p[i_r]
                newton_rows.r[i_r] = r
                newton_rows.z[i_r] = z
                i_b_key = i_b
                rz = r.dot(z)
        func_add_by_key(i_b_key, rz, newton_envs.rz_new)
    if qd.static(newton_config.has_rigid_dofs):
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            if newton_envs.is_pcg_active[i_b]:
                pcg_alpha = newton_envs.pcg_alpha[i_b]
                rigid_dofs.solution[i_d, i_b] = rigid_dofs.solution[i_d, i_b] + pcg_alpha * rigid_dofs.p[i_d, i_b]
                rigid_dofs.r[i_d, i_b] = rigid_dofs.r[i_d, i_b] - pcg_alpha * rigid_dofs.Ap[i_d, i_b]
        for i_b in range(rigid_dofs.p.shape[1]):
            if newton_envs.is_pcg_active[i_b]:
                func_rigid_solve_batch(
                    i_b,
                    rigid_dofs.r,
                    rigid_dofs.z,
                    dyn_state,
                    constraint_state,
                    rigid_info,
                    rigid_config,
                    newton_config.has_rigid_rows,
                )
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            if newton_envs.is_pcg_active[i_b]:
                qd.atomic_add(newton_envs.rz_new[i_b], rigid_dofs.r[i_d, i_b] * rigid_dofs.z[i_d, i_b])
    for _ in range(1):
        newton_loops.n_pcg_iterations_left[None] = newton_loops.n_pcg_iterations_left[None] - 1
        newton_loops.is_pcg_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_pcg_active[i_b]:
            rz = newton_envs.rz[i_b]
            rz_new = newton_envs.rz_new[i_b]
            newton_envs.pcg_beta[i_b] = rz_new / rz
            newton_envs.rz[i_b] = rz_new
            if not qd_isfinite(rz_new):
                newton_envs.is_pcg_active[i_b] = False
                newton_envs.failure[i_b] = NewtonFailure.NON_FINITE
            elif qd.abs(rz_new) <= newton_envs.rz_tol[i_b]:
                newton_envs.is_pcg_active[i_b] = False
            elif newton_loops.n_pcg_iterations_left[None] > 0:
                qd.atomic_max(newton_loops.is_pcg_running[None], 1)
    for i_r in range(n_rows):
        newton_rows.Ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
        i_b = newton_rows.rows_env[i_r]
        if newton_envs.is_pcg_active[i_b]:
            newton_rows.p[i_r] = newton_rows.z[i_r] + newton_envs.pcg_beta[i_b] * newton_rows.p[i_r]
    if qd.static(newton_config.has_rigid_dofs):
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            rigid_dofs.Ap[i_d, i_b] = 0.0
            if newton_envs.is_pcg_active[i_b]:
                rigid_dofs.p[i_d, i_b] = rigid_dofs.z[i_d, i_b] + newton_envs.pcg_beta[i_b] * rigid_dofs.p[i_d, i_b]


@qd.kernel(graph=True)
def kernel_pcg_loop(
    newton_rows: NewtonRows,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Run the iterations of the preconditioned conjugate gradient (PCG) as one graph, until every environment converges or
    the iterations run out.
    """
    while qd.graph.do_while(newton_loops.is_pcg_running):
        func_pcg_iteration(
            newton_rows,
            newton_envs,
            newton_loops,
            mpm_state,
            sparse_grid,
            rigid_dofs,
            rigid_links,
            dyn_state,
            constraint_state,
            mpm_info,
            dyn_info,
            rigid_info,
            rigid_config,
            newton_config,
        )


# ------------------------------------------------------------------------------------
# ----------------------------------- line search ------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_start_line_search(
    newton_rows: NewtonRows,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: NewtonInfo,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Set the Newton direction of every environment still iterating, minus the solution of the preconditioned conjugate
    gradient (PCG), and start its line search from the full step.

    An environment converges once its direction moves every particle and every dof by less than the tolerance, the step
    being taken still. The first step stays short of the first zero volume of a liquid particle (see
    'func_mpm_init_line_search' in mpm_solver.py).
    """
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.max_displacement[i_b] = 0.0
        newton_envs.alpha[i_b] = 1.0
        newton_envs.n_line_search_iterations[i_b] = 0
    func_mpm_init_line_search(
        newton_rows,
        newton_envs.is_active,
        newton_envs.max_displacement,
        newton_envs.alpha,
        mpm_state,
        sparse_grid,
        mpm_info,
    )
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_init_line_search(rigid_dofs, newton_envs.is_active, newton_envs.max_displacement)
    for _ in range(1):
        newton_loops.is_line_search_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        is_active = newton_envs.is_active[i_b]
        if is_active:
            max_displacement = newton_envs.max_displacement[i_b]
            newton_envs.is_converged[i_b] = max_displacement < newton_info.displacement_tol[None]
            if not (qd_isfinite(max_displacement) and qd_isfinite(newton_envs.energy_start[i_b])):
                newton_envs.failure[i_b] = NewtonFailure.NON_FINITE
            qd.atomic_max(newton_loops.is_line_search_running[None], 1)
        newton_envs.is_line_search_active[i_b] = is_active


@qd.kernel
def kernel_line_search_trial(
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: NewtonInfo,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Move the nodes and the dofs of every environment still searching to the trial point of its step, and accept the step
    once the energy there is at most the energy at the start point, or shrink it.

    A converged or failed environment takes its step at once, and an environment out of trials keeps its last one.
    """
    func_mpm_step_forward(sparse_grid, newton_envs.is_line_search_active, newton_envs.alpha)
    if qd.static(newton_config.has_rigid_dofs):
        func_rigid_step_forward(rigid_dofs, newton_envs.is_line_search_active, newton_envs.alpha)
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.energy_trial[i_b] = 0.0
    func_add_energies(
        newton_envs.energy_trial,
        newton_envs.is_line_search_active,
        newton_envs.rigid_cost,
        mpm_state,
        sparse_grid,
        rigid_dofs,
        rigid_links,
        dyn_state,
        constraint_state,
        mpm_info,
        dyn_info,
        rigid_info,
        mpm_config,
        rigid_config,
        newton_config,
    )
    for _ in range(1):
        newton_loops.is_line_search_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_line_search_active[i_b]:
            newton_envs.n_line_search_iterations[i_b] = newton_envs.n_line_search_iterations[i_b] + 1
            is_accepted = newton_envs.is_converged[i_b] or newton_envs.failure[i_b] != NewtonFailure.NONE
            if newton_envs.energy_trial[i_b] <= newton_envs.energy_start[i_b]:
                is_accepted = True
            if newton_envs.n_line_search_iterations[i_b] >= newton_info.line_search_max_iterations[None]:
                is_accepted = True
            if is_accepted:
                newton_envs.is_line_search_active[i_b] = False
                newton_envs.energy_start[i_b] = newton_envs.energy_trial[i_b]
            else:
                newton_envs.alpha[i_b] = newton_info.line_search_shrink[None] * newton_envs.alpha[i_b]
                qd.atomic_max(newton_loops.is_line_search_running[None], 1)


@qd.kernel
def kernel_end_iteration(newton_envs: NewtonEnvs, newton_loops: NewtonLoops, newton_info: NewtonInfo):
    """Stop every environment that converged or failed, or reached the iteration cap, which it latches as a failure."""
    for _ in range(1):
        newton_loops.is_newton_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_active[i_b]:
            newton_envs.n_iterations[i_b] = newton_envs.n_iterations[i_b] + 1
            if newton_envs.is_converged[i_b] or newton_envs.failure[i_b] != NewtonFailure.NONE:
                newton_envs.is_active[i_b] = False
            elif newton_envs.n_iterations[i_b] >= newton_info.max_iterations[None]:
                newton_envs.is_active[i_b] = False
                newton_envs.failure[i_b] = NewtonFailure.ITERATIONS
                qd.atomic_max(newton_loops.has_failed_iterations[None], 1)
            else:
                qd.atomic_max(newton_loops.is_newton_running[None], 1)


@qd.kernel
def kernel_finish_rigid(
    newton_envs: NewtonEnvs,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    rigid_dofs: NewtonRigidDofs,
    rigid_links: NewtonRigidLinks,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    mpm_info: MPMInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    rigid_config: qd.template(),
    newton_config: qd.template(),
    errno: qd.Tensor,
):
    """
    Hand the accelerations of the dofs solved to the rigid solver (see 'func_rigid_finish' in rigid/global_newton.py).

    An environment whose Newton solve failed keeps the state of the rigid solver until the error raises (see
    'RigidInfo.is_env_frozen').
    """
    for i_b in range(rigid_info.is_env_frozen.shape[0]):
        rigid_info.is_env_frozen[i_b] = (errno[i_b] & NewtonErrorCode.INVALID_SOLVE_NAN) != 0
    func_rigid_finish(
        newton_envs.rigid_cost,
        rigid_dofs,
        dyn_state,
        constraint_state,
        rigid_info,
        rigid_config,
        newton_config.has_rigid_rows,
    )


@qd.kernel
def kernel_flag_failures(newton_envs: NewtonEnvs, newton_loops: NewtonLoops, errno: qd.Tensor):
    """Flag the environments whose Newton solve met a value that is not finite, which keep the state of their particles
    and of the rigid solver until the error raises (see 'kernel_finish_rigid' and 'kernel_mpm_g2p' in mpm_solver.py)."""
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.failure[i_b] == NewtonFailure.NON_FINITE:
            qd.atomic_or(errno[i_b], NewtonErrorCode.INVALID_SOLVE_NAN)


@qd.kernel
def kernel_clear_failed_iterations(newton_loops: NewtonLoops):
    for _ in range(1):
        newton_loops.has_failed_iterations[None] = 0
