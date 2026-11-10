"""Newton solver of the substeps of the implicit material point method (MPM) solver."""

from collections.abc import Iterator
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.bcoo import (
    BCOOMatrixScratch,
    assemble_bcoo_matrix,
    bcoo_matrix_sizes,
    func_add_by_key,
    func_add_triplet,
    func_read_entry,
    func_report_triplet_demand,
    func_write_triplet,
)
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_solve_mass_batch
from genesis.engine.solvers.rigid.constraint.data import ConstraintState
from genesis.engine.solvers.rigid.constraint.island import func_group_constraints_by_island
from genesis.engine.solvers.rigid.constraint.solver import (
    func_cholesky_solve_batch,
    func_hessian_and_cholesky_factor_direct,
    func_update_constraint,
    initialize_Jaref,
    initialize_Ma,
)
from genesis.engine.solvers.rigid.data import DOFS, LINKS, DofsState, DynInfo, DynState, LinksInfo, RigidInfo
from genesis.options.solvers import LegacyCouplerOptions, NewtonOptions
from genesis.utils.data import ENVS, Config, Data, DataItem, DataKind, Size, field, iter_data, iter_state_arrays
from genesis.utils.misc import qd_destroy_data, qd_isfinite, qd_to_numpy
from genesis.utils.simt import qd_block_sum, qd_segmented_sum, qd_slot_neighbors

from .base_solver import Solver
from .mpm_solver import (
    MPMErrorCode,
    MPMInfo,
    MPMSparseGrid,
    MPMState,
    func_block_of_key,
    func_cell_particles,
    func_local_cell,
    func_node_offset,
    func_particle_displacement_update,
    func_particle_energy,
    func_particle_inversion_step_cap,
    func_particle_stencil_terms,
    func_q1_node_weight,
    func_q1_stencil,
    func_sorted_particle,
    func_stencil_pair_block,
    func_stencil_slots,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# The rows of the linear system of the Newton iterations, one per node of the sparse grid of the implicit MPM solver
ROWS = Size("rows")
# The smallest capacity of the rows, and the slack of a capacity grown past an overflow
MIN_ROWS = 64 * 64
ROWS_GROWTH_FACTOR = 1.2
# The arrays of the dofs and of the links of the rigid solver in the Newton solve, which a solve without them leaves out
RIGID = "rigid"


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
    the rigid solver at the current iterate of its dofs (see 'func_rigid_evaluate').
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
    The flags that keep the loops of the Newton solve running as long as an environment iterates on, with the
    iterations left to the preconditioned conjugate gradient (PCG).

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

    # Whether the dofs of the rigid solver join the unknowns (see 'NewtonSolver.is_rigid_deferred')
    is_rigid_deferred: bool
    # Whether the rigid solver has constraint rows. Their cost then joins the energy and the factor of the Hessian of its
    # Newton constraint solver preconditions its dofs, its Gauss cost and the factor of its mass matrix otherwise.
    has_rigid_rows: bool
    # Whether the rigid links hold their info per environment (see 'RigidOptions.batch_links_info')
    is_links_info_batched: bool


class NewtonRigidDofs(Data, kind=DataKind.SCRATCH, shape=(DOFS, ENVS)):
    """
    The unknowns of the dofs of the rigid solver in the Newton solve, x = dt^2 qacc, in the units of the displacement of
    the nodes, with the vectors of their solve.

    `qacc` is the acceleration of x and `Ma` its product with the mass matrix, at which the rigid solver updates its
    constraint state (see 'func_rigid_evaluate'). The other vectors are those of the rows (see NewtonRows): the
    gradient, the solution, the residual, the preconditioned residual, the search direction and its product with the
    Hessian of the preconditioned conjugate gradient (PCG), and the start point and the direction of the line search.
    """

    x: qd.Tensor = field(float)
    x_start: qd.Tensor = field(float)
    dx: qd.Tensor = field(float)
    qacc: qd.Tensor = field(float)
    Ma: qd.Tensor = field(float)
    gradient: qd.Tensor = field(float)
    solution: qd.Tensor = field(float)
    r: qd.Tensor = field(float)
    z: qd.Tensor = field(float)
    p: qd.Tensor = field(float)
    Ap: qd.Tensor = field(float)


class NewtonRigidLinks(Data, kind=DataKind.SCRATCH, shape=(LINKS, ENVS)):
    """
    The links of the rigid solver in the Newton solve.

    `motion_vel` and `motion_ang` are the spatial displacement of a link from a vector over the dofs, linear and angular
    about the root center of mass of the link (see 'func_link_spatial_disp'), and `force_vel` and `force_ang` the
    spatial force the contacts with the particles put on it, which 'func_link_force_to_dofs' maps to the dofs.
    `stiffness` is the 6 x 6 spatial stiffness of these contacts, which the factor of the rigid solver adds to its
    Hessian (see RigidContactStiffness).
    """

    motion_vel: qd.Tensor = field((float, 3))
    motion_ang: qd.Tensor = field((float, 3))
    force_vel: qd.Tensor = field((float, 3))
    force_ang: qd.Tensor = field((float, 3))
    stiffness: qd.Tensor = field((float, (6, 6)))


class NewtonData(Data):
    """The arrays of the Newton solver of a fixed size, allocated once its scene is built."""

    newton_envs: NewtonEnvs
    newton_loops: NewtonLoops
    newton_info: NewtonInfo
    rigid_dofs: NewtonRigidDofs = field(labels=RIGID)
    rigid_links: NewtonRigidLinks = field(labels=RIGID)


class NewtonRows(Data, kind=DataKind.SCRATCH, shape=(ROWS,)):
    """
    The linear system of the Newton iterations, one row of 3 unknowns per node of the sparse grid, and its solve.

    `gradient` is the gradient of the incremental potential at the current iterate, the right-hand side, `solution` the
    solution of the Hessian against it, and `r`, `z`, `p` and `Ap` the residual, the preconditioned residual, the search
    direction and its product with the Hessian of the preconditioned conjugate gradient (PCG). `jacobi` holds the
    inverses of the diagonal blocks of the Hessian, the block Jacobi preconditioner. `rows_env` is the environment of
    every row, the rows of an environment following one another. The matrix holds the upper triangle of the Hessian.
    """

    n_rows: qd.Tensor = field(int, shape=(), ndarray=True)
    rows_env: qd.Tensor = field(int, ndarray=True)
    gradient: qd.Tensor = field((float, 3), ndarray=True)
    solution: qd.Tensor = field((float, 3), ndarray=True)
    r: qd.Tensor = field((float, 3), ndarray=True)
    z: qd.Tensor = field((float, 3), ndarray=True)
    p: qd.Tensor = field((float, 3), ndarray=True)
    Ap: qd.Tensor = field((float, 3), ndarray=True)
    jacobi: qd.Tensor = field((float, (3, 3)), ndarray=True)
    matrix: BCOOMatrixScratch


@qd.data_oriented
class RigidContactStiffness:
    """
    The extra Hessian of the factor of the rigid solver: the stiffness of the contacts of the particles with the links.

    The extra Hessian is J_l^T K_l J_l summed over the links l, J_l mapping the dofs of the chain of a link to its
    spatial displacement and K_l being its spatial stiffness (see NewtonRigidLinks). The factor of the rigid solver adds
    it to its own Hessian (see 'func_hessian_and_cholesky_factor_direct').
    """

    has_extra_hessian = True

    def __init__(
        self,
        rigid_links: NewtonRigidLinks,
        dofs_state: DofsState,
        links_info: LinksInfo,
        newton_config: NewtonConfig,
    ) -> None:
        self.rigid_links = rigid_links
        self.dofs_state = dofs_state
        self.links_info = links_info
        self.newton_config = newton_config

    @qd.func
    def func_add_island_hessian_batch(
        self, i_b: int, i_island: int, constraint_state: ConstraintState, rigid_config: qd.template()
    ):
        for i_l in range(self.rigid_links.stiffness.shape[0]):
            stiffness = self.rigid_links.stiffness[i_l, i_b]
            if stiffness.norm_sqr() > 0.0:
                i_l1 = i_l
                while i_l1 >= 0:
                    I_l1 = [i_l1, i_b] if qd.static(self.newton_config.is_links_info_batched) else i_l1
                    for i_d1 in range(self.links_info.dof_start[I_l1], self.links_info.dof_end[I_l1]):
                        if constraint_state.island.dofs_island_idx[i_d1, i_b] == i_island:
                            stiffness_cdof1 = stiffness @ func_spatial_vector(
                                self.dofs_state.cdof_vel[i_d1, i_b], self.dofs_state.cdof_ang[i_d1, i_b]
                            )
                            pos1 = constraint_state.island.dof_local_pos[i_d1, i_b]
                            i_l2 = i_l
                            while i_l2 >= 0:
                                I_l2 = [i_l2, i_b] if qd.static(self.newton_config.is_links_info_batched) else i_l2
                                for i_d2 in range(self.links_info.dof_start[I_l2], self.links_info.dof_end[I_l2]):
                                    # The lower triangle of the block, in the island-local order of the dofs
                                    if pos1 >= constraint_state.island.dof_local_pos[i_d2, i_b]:
                                        value = stiffness_cdof1.dot(
                                            func_spatial_vector(
                                                self.dofs_state.cdof_vel[i_d2, i_b], self.dofs_state.cdof_ang[i_d2, i_b]
                                            )
                                        )
                                        if qd.static(rigid_config.enable_jacobi_equilibration):
                                            value = (
                                                value
                                                * constraint_state.nt_jacobi[i_d1, i_b]
                                                * constraint_state.nt_jacobi[i_d2, i_b]
                                            )
                                        # The thread of the island alone writes its block
                                        constraint_state.nt_H[i_b, i_d1, i_d2] = (
                                            constraint_state.nt_H[i_b, i_d1, i_d2] + value
                                        )
                                i_l2 = self.links_info.parent_idx[I_l2]
                    i_l1 = self.links_info.parent_idx[I_l1]


class NewtonSolver(Solver):
    """
    Solver of the substeps of the implicit material point method (MPM) solver by Newton iterations on their incremental
    potential.

    The unknowns of a substep are the displacement of the nodes of the sparse grid of the MPM solver over it (see
    MPMSparseGrid), and when the rigid geoms couple with the particles, the accelerations of the dofs of the rigid
    solver, whose constraint solve then joins the energy (see 'is_rigid_deferred'). Every iteration solves the projected
    Hessian against the gradient by a preconditioned conjugate gradient (PCG) and searches a step along the direction
    that lowers the energy, every environment iterating until its own convergence. The solve runs once every solver has
    prepared its substep and the coupler has run, and before the solvers finish it, since the MPM solver moves its
    particles by the displacement solved and the rigid solver integrates the accelerations solved.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: NewtonOptions) -> None:
        super().__init__(scene, sim, options)
        self.data_manager: NewtonData | None = None
        self.newton_envs: NewtonEnvs | None = None
        self.newton_loops: NewtonLoops | None = None
        self.newton_info: NewtonInfo | None = None
        self.newton_config: NewtonConfig | None = None
        self.rigid_dofs: NewtonRigidDofs | None = None
        self.rigid_links: NewtonRigidLinks | None = None
        self._is_rigid_deferred = False
        # The stiffness of the contacts that the factor of the rigid solver adds, made once the rigid solver is built
        self._rigid_contact_stiffness: RigidContactStiffness | None = None
        # The rows of the linear system, rebuilt with a larger capacity when the grid of the MPM solver outgrows them,
        # and the rows they replaced, kept referenced with their arrays freed (see 'update_newton_rows')
        self.newton_rows: NewtonRows | None = None
        self._newton_rows_replaced: list[NewtonRows] = []

    @property
    def is_active(self) -> bool:
        mpm_solver = self._sim.mpm_solver
        return mpm_solver.is_active and mpm_solver._use_implicit_solver

    @property
    def is_rigid_deferred(self) -> bool:
        """
        Whether the rigid solver solves its substeps within the Newton solve, the accelerations of its dofs joining the
        unknowns.

        It does once a rigid geom that couples with the particles moves (see 'LegacyCouplerOptions.rigid_mpm' and
        'needs_coup' of gs.materials.Rigid), the geoms fixed in the world acting on the particles alone.
        """
        return self._is_rigid_deferred

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
        self._is_rigid_deferred = self._sim.coupler.rigid_couplings[0] and any(
            geom.needs_coup and not geom.is_fixed for geom in rigid_solver.geoms
        )
        if self._is_rigid_deferred:
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
            is_rigid_deferred=self._is_rigid_deferred,
            has_rigid_rows=not rigid_options.disable_constraint,
            is_links_info_batched=rigid_options.batch_links_info,
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
            unused={RIGID: not self._is_rigid_deferred},
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
            if self._is_rigid_deferred:
                yield from iter_data(self.data_manager, "rigid_dofs")
                yield from iter_data(self.data_manager, "rigid_links")
        if self.newton_rows is not None:
            yield from iter_data(self, "newton_rows")

    def substep_pre_coupling(self, f: int) -> None:
        pass

    def substep_post_coupling(self, f: int) -> None:
        """
        Solve the substep of the implicit material point method (MPM) solver, and that of the rigid solver with it when
        deferred, before the solvers finish them.
        """
        if not self.is_active:
            return
        mpm_solver = self._sim.mpm_solver
        rigid_solver = self._sim.rigid_solver
        mpm_state, mpm_info, sparse_grid = mpm_solver.mpm_state, mpm_solver.mpm_info, mpm_solver.sparse_grid
        dyn_state, dyn_info, rigid_info = rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_info
        constraint_state = rigid_solver.constraint_solver.constraint_state
        errno = mpm_solver._errno
        if self._is_rigid_deferred and self._rigid_contact_stiffness is None:
            self._rigid_contact_stiffness = RigidContactStiffness(
                self.rigid_links, dyn_state.dofs, dyn_info.links, self.newton_config
            )
        self.newton_rows = update_newton_rows(
            int(qd_to_numpy(sparse_grid.n_blocks)) * 64,
            int(qd_to_numpy(mpm_state.sorted_particles.n_tasks)),
            self.newton_rows,
            self._newton_rows_replaced,
        )
        args_state = (mpm_state, sparse_grid, self.rigid_dofs, self.rigid_links, dyn_state, constraint_state)
        args_info = (mpm_info, dyn_info, rigid_info)
        args_config = (mpm_solver.mpm_config, rigid_solver.rigid_config, self.newton_config)
        kernel_start_solve(f, self.newton_envs, self.newton_loops, *args_state, *args_info, *args_config, errno)
        for _ in range(self._options.max_iterations):
            kernel_assemble(f, self.newton_rows, self.newton_envs, *args_state, *args_info, *args_config)
            assemble_bcoo_matrix(self.newton_rows.matrix, block_scalars=9)
            if self._is_rigid_deferred:
                kernel_assemble_rigid(
                    f,
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
                f, self.newton_rows, self.newton_envs, self.newton_loops, *args_state, *args_info, *args_config
            )
            kernel_start_line_search(
                f,
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
                    f, self.newton_envs, self.newton_loops, *args_state, self.newton_info, *args_info, *args_config
                )
                if not qd_to_numpy(self.newton_loops.is_line_search_running):
                    break
            kernel_end_iteration(self.newton_envs, self.newton_loops, self.newton_info)
            if not qd_to_numpy(self.newton_loops.is_newton_running):
                break
        kernel_flag_failures(self.newton_envs, self.newton_loops, errno)
        if self._is_rigid_deferred:
            kernel_finish_rigid(self.newton_envs, *args_state, *args_info, *args_config, errno)
        if self._sim.cur_substep_global % self._options.failure_check_interval == 0:
            if qd_to_numpy(self.newton_loops.has_failed_iterations):
                kernel_clear_failed_iterations(self.newton_loops)
                gs.logger.warning(
                    "The Newton solve of the implicit MPM solver stopped before converging: raise "
                    "NewtonOptions(max_iterations) or loosen NewtonOptions(velocity_tol)."
                )


def update_newton_rows(
    n_rows: int, n_tasks: int, newton_rows: NewtonRows | None, newton_rows_replaced: list[NewtonRows]
) -> NewtonRows:
    """
    Return rows of the linear system holding the nodes of the sparse grid and the triplets of their Hessian, the rows
    given if they do, larger ones otherwise.

    The Hessian holds the kinetic block of every node and the upper triangle of the 8 x 8 blocks of the stencil of every
    task of the assembly (see 'kernel_list_cells' in mpm_solver.py). Rows replaced stay referenced in
    `newton_rows_replaced`, with their arrays freed.
    """
    n_triplets = n_rows + 36 * n_tasks
    if newton_rows is not None:
        if newton_rows.rows_env.shape[0] >= n_rows and newton_rows.matrix.triplets_row.shape[0] >= n_triplets:
            return newton_rows
        # FIXME: quadrants#1074 - a launch is cached under the identity of its struct arguments, which a new struct
        # could take over from a dead one, so the rows replaced stay referenced, with their arrays freed.
        for _, array in iter_state_arrays(newton_rows):
            qd_destroy_data(array)
        newton_rows_replaced.append(newton_rows)
    max_rows = max(int(np.ceil(n_rows * ROWS_GROWTH_FACTOR)), MIN_ROWS)
    max_triplets = max(int(np.ceil(n_triplets * ROWS_GROWTH_FACTOR)), MIN_ROWS)
    return NewtonRows(sizes={ROWS: max_rows, NewtonRows.matrix: bcoo_matrix_sizes(max_triplets, block_shape=(3, 3))})


# ------------------------------------------------------------------------------------
# ----------------------------------- rigid terms ------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_spatial_vector(lin: qd.types.vector(3), ang: qd.types.vector(3)):
    """Return the 6-vector of a spatial quantity, its linear part first."""
    return qd.Vector([lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]], dt=gs.qd_float)


@qd.func
def func_point_motion(motion_vel: qd.types.vector(3), motion_ang: qd.types.vector(3), r: qd.types.vector(3)):
    """Return the displacement of the point of a link at the arm r from its root center of mass, from the spatial
    displacement of the link."""
    return motion_vel + motion_ang.cross(r)


@qd.func
def func_point_stiffness(hessian: qd.types.matrix(3, 3), r: qd.types.vector(3)):
    """Return the spatial stiffness P^T H P of a link from the stiffness H of its point at the arm r from its root
    center of mass, P = [I, -[r]x] mapping the spatial displacement of the link to the displacement of the point."""
    P = qd.Matrix(
        [
            [1.0, 0.0, 0.0, 0.0, r[2], -r[1]],
            [0.0, 1.0, 0.0, -r[2], 0.0, r[0]],
            [0.0, 0.0, 1.0, r[1], -r[0], 0.0],
        ],
        dt=gs.qd_float,
    )
    return P.transpose() @ hessian @ P


@qd.func
def func_gauss_cost_term(Ma: float, qf_smooth: float, qacc: float, acc_smooth: float):
    """Return the term of a dof in the Gauss cost 1/2 (a - a_smooth)^T M (a - a_smooth) of the rigid solver."""
    return 0.5 * (Ma - qf_smooth) * (qacc - acc_smooth)


@qd.func
def func_rigid_gradient(Ma: float, qf_smooth: float, qfrc_constraint: float, dt: float):
    """Return the gradient of the cost of the rigid solver, scaled by dt^4 to the units of the incremental potential, in
    the unknown x = dt^2 qacc of a dof."""
    return dt * dt * (Ma - qf_smooth - qfrc_constraint)


@qd.func(requires_top_level=True)
def func_link_spatial_disp(
    src: qd.Tensor,
    rigid_links: NewtonRigidLinks,
    dofs_state: DofsState,
    links_info: LinksInfo,
    newton_config: qd.template(),
):
    """Set the spatial displacement of every link from a vector over the dofs, summed over the dofs of its chain."""
    qd.loop_config(name="newton_link_spatial_disp")
    for i_l, i_b in qd.ndrange(rigid_links.motion_vel.shape[0], rigid_links.motion_vel.shape[1]):
        motion_vel = qd.Vector.zero(gs.qd_float, 3)
        motion_ang = qd.Vector.zero(gs.qd_float, 3)
        j_l = i_l
        while j_l >= 0:
            J_l = [j_l, i_b] if qd.static(newton_config.is_links_info_batched) else j_l
            for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                motion_vel += dofs_state.cdof_vel[i_d, i_b] * src[i_d, i_b]
                motion_ang += dofs_state.cdof_ang[i_d, i_b] * src[i_d, i_b]
            j_l = links_info.parent_idx[J_l]
        rigid_links.motion_vel[i_l, i_b] = motion_vel
        rigid_links.motion_ang[i_l, i_b] = motion_ang


@qd.func(requires_top_level=True)
def func_link_force_to_dofs(
    out: qd.Tensor,
    rigid_links: NewtonRigidLinks,
    dofs_state: DofsState,
    links_info: LinksInfo,
    newton_config: qd.template(),
):
    """Add to a vector over the dofs the generalized force of the spatial force on every link, over the dofs of its
    chain."""
    qd.loop_config(name="newton_link_force_to_dofs")
    for i_l, i_b in qd.ndrange(rigid_links.force_vel.shape[0], rigid_links.force_vel.shape[1]):
        force_vel = rigid_links.force_vel[i_l, i_b]
        force_ang = rigid_links.force_ang[i_l, i_b]
        if force_vel.norm_sqr() + force_ang.norm_sqr() > 0.0:
            j_l = i_l
            while j_l >= 0:
                J_l = [j_l, i_b] if qd.static(newton_config.is_links_info_batched) else j_l
                for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                    qd.atomic_add(
                        out[i_d, i_b],
                        dofs_state.cdof_vel[i_d, i_b].dot(force_vel) + dofs_state.cdof_ang[i_d, i_b].dot(force_ang),
                    )
                j_l = links_info.parent_idx[J_l]


@qd.func(requires_top_level=True)
def func_clear_link_forces(rigid_links: NewtonRigidLinks):
    qd.loop_config(name="newton_clear_link_forces")
    for i_l, i_b in qd.ndrange(rigid_links.force_vel.shape[0], rigid_links.force_vel.shape[1]):
        rigid_links.force_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        rigid_links.force_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_contact_arm(f: int, i_s: int, i_g: int, mpm_state: MPMState, dyn_state: DynState, dyn_info: DynInfo):
    """Return the link of the geom a sorted particle touches, the environment and the arm of the particle from the root
    center of mass of the link."""
    i_p, i_b, cell = func_sorted_particle(i_s, mpm_state.particles.pos.shape[1], mpm_state.sorted_particles)
    i_l = dyn_info.geoms.link_idx[i_g]
    return i_l, i_b, mpm_state.particles.pos[f, i_p, i_b] - dyn_state.links.root_COM[i_l, i_b]


@qd.func
def func_contact_partner_disp(
    f: int, i_s: int, mpm_state: MPMState, rigid_links: NewtonRigidLinks, dyn_state: DynState, dyn_info: DynInfo
):
    """
    Return the displacement of the point of the rigid link a sorted particle touches, from the spatial displacement of
    the link (see 'func_link_spatial_disp'), zero for a particle in no contact.
    """
    disp = qd.Vector.zero(gs.qd_float, 3)
    i_g = mpm_state.rigid_contacts.geom_idx[i_s]
    if i_g >= 0:
        i_l, i_b, r = func_contact_arm(f, i_s, i_g, mpm_state, dyn_state, dyn_info)
        disp = func_point_motion(rigid_links.motion_vel[i_l, i_b], rigid_links.motion_ang[i_l, i_b], r)
    return disp


@qd.func(requires_top_level=True)
def func_rigid_evaluate(
    rigid_cost: qd.Tensor,
    rigid_dofs: NewtonRigidDofs,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """Evaluate the cost of the constraint solve of the rigid solver at qacc = x / dt^2, by the update of its constraint
    state, or its Gauss cost without constraint rows."""
    dt = rigid_info.substep_dt[None]
    qd.loop_config(name="newton_rigid_qacc")
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        rigid_dofs.qacc[i_d, i_b] = rigid_dofs.x[i_d, i_b] / (dt * dt)
    initialize_Ma(rigid_dofs.Ma, rigid_dofs.qacc, rigid_info, rigid_config)
    if qd.static(newton_config.has_rigid_rows):
        initialize_Jaref(qacc=rigid_dofs.qacc, constraint_state=constraint_state, rigid_config=rigid_config)
        func_update_constraint(
            qacc=rigid_dofs.qacc,
            Ma=rigid_dofs.Ma,
            cost=rigid_cost,
            dyn_state=dyn_state,
            constraint_state=constraint_state,
            rigid_config=rigid_config,
        )
    else:
        qd.loop_config(name="newton_rigid_gauss_cost")
        for i_b in range(rigid_dofs.x.shape[1]):
            cost = gs.qd_float(0.0)
            for i_d in range(rigid_dofs.x.shape[0]):
                cost += func_gauss_cost_term(
                    rigid_dofs.Ma[i_d, i_b],
                    dyn_state.dofs.qf_smooth[i_d, i_b],
                    rigid_dofs.qacc[i_d, i_b],
                    dyn_state.dofs.acc_smooth[i_d, i_b],
                )
            rigid_cost[i_b] = cost


@qd.func
def func_rigid_solve_batch(
    i_b: int,
    rhs: qd.Tensor,
    out: qd.Tensor,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """
    Solve the block of the dofs of the rigid solver of an environment against a vector over them.

    The solve of `rhs` into `out` goes through the factor of the rigid solver, of its block of the Hessian (see
    'kernel_assemble_rigid') or of its mass matrix without constraint rows (see 'NewtonConfig.has_rigid_rows').
    """
    if qd.static(newton_config.has_rigid_rows):
        for i_island in range(constraint_state.island.n_islands[i_b]):
            func_cholesky_solve_batch(
                i_b, i_island, rhs=rhs, out=out, constraint_state=constraint_state, rigid_config=rigid_config
            )
    else:
        func_solve_mass_batch(i_b, rhs, out, dyn_state, rigid_info, rigid_config)


@qd.func(requires_top_level=True)
def func_rigid_matvec(
    rigid_dofs: NewtonRigidDofs,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    newton_config: qd.template(),
):
    """Set Ap = (M + J^T D J) p over the dofs of the rigid solver, J and D those of its active constraint rows."""
    initialize_Ma(rigid_dofs.Ap, rigid_dofs.p, rigid_info, rigid_config)
    if qd.static(newton_config.has_rigid_rows):
        qd.loop_config(name="newton_rigid_matvec_rows")
        for i_b in range(rigid_dofs.p.shape[1]):
            for i_c in range(constraint_state.n_constraints[i_b]):
                if constraint_state.active[i_c, i_b]:
                    jac_p = gs.qd_float(0.0)
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        jac_p += constraint_state.jac[i_c, i_d, i_b] * rigid_dofs.p[i_d, i_b]
                    efc_D_jac_p = constraint_state.efc_D[i_c, i_b] * jac_p
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        rigid_dofs.Ap[i_d, i_b] = (
                            rigid_dofs.Ap[i_d, i_b] + constraint_state.jac[i_c, i_d, i_b] * efc_D_jac_p
                        )


# ------------------------------------------------------------------------------------
# ------------------------------------- energy ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_node_kinetic_energy(
    mass: float, is_fixed: int, u: qd.types.vector(3), vel_free: qd.types.vector(3), dt: float
):
    """Return the kinetic energy of a node in the incremental potential, 1/2 m |u - dt v_free|^2, zero for a fixed node
    or a node without mass."""
    energy = gs.qd_float(0.0)
    if mass > 0.0 and is_fixed == 0:
        energy = 0.5 * mass * (u - dt * vel_free).norm_sqr()
    return energy


@qd.func(requires_top_level=True)
def func_add_energies(
    f: int,
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

    `envs_mask` selects the environments, and `energies` holds their entries. The energy is the incremental potential
    of the particles and of the nodes, the particles and the nodes of an environment following one another, and when
    the dofs of the rigid solver join the unknowns, the cost of the rigid solver at their iterate, scaled by dt^4 to
    the units of the incremental potential.
    """
    if qd.static(newton_config.is_rigid_deferred):
        func_link_spatial_disp(rigid_dofs.x, rigid_links, dyn_state.dofs, dyn_info.links, newton_config)
    n_valid = mpm_state.sorted_particles.n_valid[None]
    qd.loop_config(block_dim=256)
    for i_s in range((n_valid + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        energy = gs.qd_float(0.0)
        if i_s < n_valid:
            disp_partner = qd.Vector.zero(gs.qd_float, 3)
            if qd.static(newton_config.is_rigid_deferred):
                disp_partner = func_contact_partner_disp(f, i_s, mpm_state, rigid_links, dyn_state, dyn_info)
            i_b, energy_particle = func_particle_energy(
                f, i_s, disp_partner, mpm_state, sparse_grid, mpm_info, mpm_config
            )
            if envs_mask[i_b]:
                i_b_key = i_b
                energy = energy_particle
        func_add_by_key(i_b_key, energy, energies)
    n_nodes = sparse_grid.n_blocks[None] * 64
    qd.loop_config(block_dim=256)
    for i_n in range((n_nodes + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        energy = gs.qd_float(0.0)
        if i_n < n_nodes:
            i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
            if envs_mask[i_b]:
                i_b_key = i_b
                energy = func_node_kinetic_energy(
                    sparse_grid.nodes_mass[i_n],
                    sparse_grid.nodes_is_fixed[i_n],
                    sparse_grid.nodes_u[i_n],
                    sparse_grid.nodes_vel_free[i_n],
                    mpm_info.substep_dt[None],
                )
        func_add_by_key(i_b_key, energy, energies)
    if qd.static(newton_config.is_rigid_deferred):
        func_rigid_evaluate(
            rigid_cost, rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, newton_config
        )
        dt = rigid_info.substep_dt[None]
        for i_b in range(energies.shape[0]):
            if envs_mask[i_b]:
                energies[i_b] = energies[i_b] + dt * dt * dt * dt * rigid_cost[i_b]


@qd.kernel
def kernel_start_solve(
    f: int,
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
    Start the Newton solve of every environment at zero displacement of the nodes, but those whose solve failed earlier
    in the step, and evaluate its energy there.

    The dofs of the rigid solver start from the warm start of its constraint solve, or from their smooth acceleration.
    Its constraint rows are grouped by island (see 'func_solve_init' in rigid/constraint/solver.py), as its own
    constraint solve starts. A link whose geom holds the rows of the contact of a particle is constrained over the
    substep (see 'LinksState.is_constrained' in rigid/data.py), as the contacts of the rigid solver mark theirs.
    """
    for _ in range(1):
        newton_loops.is_newton_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        is_active = (errno[i_b] & MPMErrorCode.INVALID_NEWTON_NAN) == 0
        newton_envs.is_active[i_b] = is_active
        newton_envs.is_converged[i_b] = False
        newton_envs.failure[i_b] = NewtonFailure.NONE
        newton_envs.n_iterations[i_b] = 0
        newton_envs.energy_start[i_b] = 0.0
        if is_active:
            qd.atomic_max(newton_loops.is_newton_running[None], 1)
    if qd.static(newton_config.is_rigid_deferred):
        dt = rigid_info.substep_dt[None]
        for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
            qacc = dyn_state.dofs.acc_smooth[i_d, i_b]
            if qd.static(newton_config.has_rigid_rows):
                if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
                    qacc = constraint_state.qacc_ws[i_d, i_b]
            rigid_dofs.x[i_d, i_b] = dt * dt * qacc
        # The constraint assembly of the rigid solver resets the links it marks constrained, but without constraint rows
        if qd.static(not newton_config.has_rigid_rows):
            for i_l, i_b in qd.ndrange(*dyn_state.links.is_constrained.shape):
                dyn_state.links.is_constrained[i_l, i_b] = False
        for i_s in range(mpm_state.sorted_particles.n_valid[None]):
            i_g = mpm_state.rigid_contacts.geom_idx[i_s]
            if i_g >= 0:
                i_l, i_b, _ = func_contact_arm(f, i_s, i_g, mpm_state, dyn_state, dyn_info)
                dyn_state.links.is_constrained[i_l, i_b] = True
        if qd.static(newton_config.has_rigid_rows):
            if qd.static(rigid_config.enable_signorini_contact):
                for i_cone, i_b in qd.ndrange(*constraint_state.cone_latch.shape):
                    constraint_state.cone_latch[i_cone, i_b] = -1.0
            for i_b in range(rigid_dofs.x.shape[1]):
                func_group_constraints_by_island(i_b, constraint_state, rigid_config)
                for i_island in range(constraint_state.island.n_islands[i_b]):
                    constraint_state.island.improved[i_island, i_b] = True
    func_add_energies(
        f,
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
def kernel_assemble(
    f: int,
    newton_rows: NewtonRows,
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
):
    """
    Assemble the gradient and the projected Hessian of the incremental potential at the displacement of the nodes.

    Every node takes its kinetic block, an identity on a fixed node, whose unknown stays zero, and on a node without
    mass, whose unknown the elastic terms of the particles around it set.
    Every task of the sorted particles (see 'kernel_list_cells' in mpm_solver.py) writes the 36 blocks of the upper
    triangle of the stencil of its cell, summed over its particles: the lanes of a warp sum them on GPU, and one thread
    walks the particles of a cell on CPU. The gradient of a node then gathers the factors its particles staged. The
    contact of a particle with a link acts on its displacement relative to the point of the link it touches, which the
    iterate of the dofs of the rigid solver moves when they join the unknowns.
    """
    if qd.static(newton_config.is_rigid_deferred):
        func_link_spatial_disp(rigid_dofs.x, rigid_links, dyn_state.dofs, dyn_info.links, newton_config)
    n_nodes = sparse_grid.n_blocks[None] * 64
    n_valid = mpm_state.sorted_particles.n_valid[None]
    for _ in range(1):
        newton_rows.n_rows[None] = n_nodes
        func_report_triplet_demand(n_nodes + 36 * mpm_state.sorted_particles.n_tasks[None], newton_rows.matrix)
    for i_n in range(n_nodes):
        i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
        newton_rows.rows_env[i_n] = i_b
        mass = sparse_grid.nodes_mass[i_n]
        if mass <= 0.0 or sparse_grid.nodes_is_fixed[i_n] != 0:
            mass = 1.0
        func_write_triplet(
            i_n, i_n, i_n, mass * qd.Matrix.identity(gs.qd_float, 3), newton_rows.matrix, is_symmetric=True
        )

    # The pairs of nodes (a, b) of the upper triangle of a stencil, a <= b, in the order of the triplets of a task
    pairs_a = qd.Vector([a for a in range(8) for b in range(a, 8)], dt=gs.qd_int)
    pairs_b = qd.Vector([b for a in range(8) for b in range(a, 8)], dt=gs.qd_int)
    if qd.static(gs.backend == gs.cpu):
        for i_cell in range(mpm_state.sorted_particles.n_cells[None]):
            i_s_start = mpm_state.sorted_particles.cells_start[i_cell]
            i_t_start = n_nodes + 36 * mpm_state.sorted_particles.tasks_idx[i_s_start]
            for i_s in range(i_s_start, mpm_state.sorted_particles.cells_start[i_cell + 1]):
                disp_partner = qd.Vector.zero(gs.qd_float, 3)
                if qd.static(newton_config.is_rigid_deferred):
                    disp_partner = func_contact_partner_disp(f, i_s, mpm_state, rigid_links, dyn_state, dyn_info)
                slots, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian = (
                    func_particle_stencil_terms(f, i_s, disp_partner, mpm_state, sparse_grid, mpm_info, mpm_config)
                )
                for i_pair in range(36):
                    a = pairs_a[i_pair]
                    b = pairs_b[i_pair]
                    block_ab = func_stencil_pair_block(
                        a, b, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian
                    )
                    if i_s == i_s_start:
                        func_write_triplet(
                            i_t_start + i_pair, slots[a], slots[b], block_ab, newton_rows.matrix, is_symmetric=True
                        )
                    else:
                        func_add_triplet(
                            i_t_start + i_pair, slots[a], slots[b], block_ab, newton_rows.matrix, is_symmetric=True
                        )
    else:
        qd.loop_config(block_dim=128)
        for i_s in range((n_valid + 31) // 32 * 32):
            tid = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
            is_valid = i_s < n_valid
            # A lane past the particles takes the terms of the last one and adds nothing, every lane taking part in the
            # sums of the warp
            i_s_clamped = qd.min(i_s, n_valid - 1)
            i_task = gs.qd_int(-1)
            if is_valid:
                i_task = mpm_state.sorted_particles.tasks_idx[i_s]
            i_task_prev, i_task_next = qd_slot_neighbors(tid, i_task)
            disp_partner = qd.Vector.zero(gs.qd_float, 3)
            if qd.static(newton_config.is_rigid_deferred):
                disp_partner = func_contact_partner_disp(f, i_s_clamped, mpm_state, rigid_links, dyn_state, dyn_info)
            slots, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian = (
                func_particle_stencil_terms(f, i_s_clamped, disp_partner, mpm_state, sparse_grid, mpm_info, mpm_config)
            )
            for i_pair in range(36):
                a = pairs_a[i_pair]
                b = pairs_b[i_pair]
                block_ab = gs.qd_float(is_valid) * func_stencil_pair_block(
                    a, b, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian
                )
                is_tail = False
                for j, k in qd.static(qd.ndrange(3, 3)):
                    block_jk, is_tail = qd_segmented_sum(tid, i_task, i_task_prev, i_task_next, block_ab[j, k])
                    block_ab[j, k] = block_jk
                if is_tail:
                    func_write_triplet(
                        n_nodes + 36 * i_task + i_pair,
                        slots[a],
                        slots[b],
                        block_ab,
                        newton_rows.matrix,
                        is_symmetric=True,
                    )

    # The gradient of every node: its kinetic term and the factors staged by the particles of the cells around it
    N_LANES = qd.static(1 if gs.backend == gs.cpu else 32)
    n_particles = mpm_state.particles.pos.shape[1]
    inv_dx = mpm_info.inv_dx[None]
    dt = mpm_info.substep_dt[None]
    qd.loop_config(block_dim=128)
    for i_th in range(n_nodes * N_LANES):
        i_lane = gs.qd_int(0)
        if qd.static(N_LANES > 1):
            i_lane = gs.qd_int(qd.simt.subgroup.invocation_id()) & 31
        i_n = i_th // N_LANES
        i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
        node = block * 4 + func_local_cell(i_n % 64)
        gradient = qd.Vector.zero(gs.qd_float, 3)
        for a in range(8):
            offset = func_node_offset(a)
            cell = node - offset
            i_s_start, i_s_end = func_cell_particles(i_b, cell, i_lane, sparse_grid, is_lanes=N_LANES > 1)
            for i_chunk_ in range((i_s_end - i_s_start + N_LANES - 1) // N_LANES):
                i_s = i_s_start + i_chunk_ * N_LANES + i_lane
                if i_s < i_s_end:
                    i_p = mpm_state.sorted_particles.particles_idx[i_s] - i_b * n_particles
                    x_p = mpm_state.particles.pos[f, i_p, i_b]
                    w, grad_w = func_q1_node_weight(offset, x_p * inv_dx - cell.cast(gs.qd_float), inv_dx)
                    gradient += (
                        w * mpm_state.sorted_particles.point_gradient[i_s]
                        + mpm_state.sorted_particles.stress_factor[i_s] @ grad_w
                    )
        if qd.static(N_LANES > 1):
            for d in qd.static(range(3)):
                gradient[d] = qd_block_sum(gradient[d])
        if i_lane == 0:
            mass = sparse_grid.nodes_mass[i_n]
            gradient_node = qd.Vector.zero(gs.qd_float, 3)
            if sparse_grid.nodes_is_fixed[i_n] == 0:
                u = sparse_grid.nodes_u[i_n]
                gradient_node = gradient + mass * (u - dt * sparse_grid.nodes_vel_free[i_n])
            newton_rows.gradient[i_n] = gradient_node


@qd.kernel
def kernel_assemble_rigid(
    f: int,
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
    Assemble the gradient of the energy in the dofs of the rigid solver at the current iterate, after the rows (see
    'kernel_assemble'), and factor their block of the Hessian, which preconditions them in the linear solve.

    The gradient holds that of the cost of the rigid solver and the forces of the contacts of the particles on the links
    (see 'MPMRigidContacts.gradient'). The block holds the Hessian of the rigid solver and the stiffness of these
    contacts (see RigidContactStiffness), or else the mass matrix, whose factor the rigid solver holds already (see
    'NewtonConfig.has_rigid_rows').
    """
    func_rigid_evaluate(
        newton_envs.rigid_cost, rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, newton_config
    )
    dt = rigid_info.substep_dt[None]
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        qfrc_constraint = gs.qd_float(0.0)
        if qd.static(newton_config.has_rigid_rows):
            qfrc_constraint = constraint_state.qfrc_constraint[i_d, i_b]
        rigid_dofs.gradient[i_d, i_b] = func_rigid_gradient(
            rigid_dofs.Ma[i_d, i_b], dyn_state.dofs.qf_smooth[i_d, i_b], qfrc_constraint, dt
        )
    func_clear_link_forces(rigid_links)
    for i_l, i_b in qd.ndrange(rigid_links.stiffness.shape[0], rigid_links.stiffness.shape[1]):
        rigid_links.stiffness[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_g = mpm_state.rigid_contacts.geom_idx[i_s]
        if i_g >= 0:
            i_l, i_b, r = func_contact_arm(f, i_s, i_g, mpm_state, dyn_state, dyn_info)
            force = -mpm_state.rigid_contacts.gradient[i_s]
            qd.atomic_add(rigid_links.force_vel[i_l, i_b], force)
            qd.atomic_add(rigid_links.force_ang[i_l, i_b], r.cross(force))
            qd.atomic_add(
                rigid_links.stiffness[i_l, i_b], func_point_stiffness(mpm_state.rigid_contacts.hessian[i_s], r)
            )
    func_link_force_to_dofs(rigid_dofs.gradient, rigid_links, dyn_state.dofs, dyn_info.links, newton_config)
    if qd.static(newton_config.has_rigid_rows):
        func_hessian_and_cholesky_factor_direct(
            constraint_state, rigid_info, rigid_config, compute_envelope=True, extra_hessian=rigid_contact_stiffness
        )


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
    Start the preconditioned conjugate gradient (PCG) of the environments still iterating from a zero solution, its
    preconditioner inverting the diagonal blocks of the rows and the block of the dofs by its factor.

    A diagonal block is inverted equilibrated to a unit diagonal, which keeps the inverse accurate for the rows of very
    different stiffness. An environment stops once r . z falls below `pcg_tol` times its value at the start, and fails
    its Newton solve once r . z is not finite, which a value that is not finite in its rows or its dofs makes.
    """
    for i_entry in range(newton_rows.matrix.n_entries[None]):
        i_r, j_r, block = func_read_entry(i_entry, newton_rows.matrix, block_rows=3, block_cols=3)
        if i_r == j_r:
            diagonal = qd.Vector([block[0, 0], block[1, 1], block[2, 2]], dt=gs.qd_float)
            scale = qd.select(diagonal > 0.0, 1.0 / qd.sqrt(qd.select(diagonal > 0.0, diagonal, 1.0)), 1.0)
            scaling = scale.outer_product(scale)
            inverse = qd.Matrix.identity(gs.qd_float, 3)
            if (block * scaling).determinant() > 0.0:
                inverse = (block * scaling).inverse()
            newton_rows.jacobi[i_r] = inverse * scaling
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
    if qd.static(newton_config.is_rigid_deferred):
        for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
            rigid_dofs.r[i_d, i_b] = rigid_dofs.gradient[i_d, i_b]
            rigid_dofs.solution[i_d, i_b] = 0.0
            rigid_dofs.Ap[i_d, i_b] = 0.0
        for i_b in range(rigid_dofs.x.shape[1]):
            func_rigid_solve_batch(
                i_b, rigid_dofs.r, rigid_dofs.z, dyn_state, constraint_state, rigid_info, rigid_config, newton_config
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
    f: int,
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

    When the dofs of the rigid solver join the unknowns, the product of the Hessian with the search direction adds their
    block, the Hessian of the rigid solver, and the coupling of the contacts of the particles with the links, whose
    blocks between the nodes of a particle the matrix holds already (see 'MPMRigidContacts.hessian').
    """
    # The product Ap of the matrix with the search direction, in the energy form p . Ap of every environment without the
    # dofs, the upper triangle standing for the transposed blocks of the lower one
    n_entries = newton_rows.matrix.n_entries[None]
    qd.loop_config(block_dim=256)
    for i_entry in range((n_entries + 31) // 32 * 32):
        i_r_key = gs.qd_int(-1)
        i_b_key = gs.qd_int(-1)
        product = qd.Vector.zero(gs.qd_float, 3)
        energy = gs.qd_float(0.0)
        if i_entry < n_entries:
            i_r, j_r, block = func_read_entry(i_entry, newton_rows.matrix, block_rows=3, block_cols=3)
            i_b = newton_rows.rows_env[i_r]
            if newton_envs.is_pcg_active[i_b]:
                i_r_key = i_r
                i_b_key = i_b
                p_col = newton_rows.p[j_r]
                product = block @ p_col
                energy = p_col.dot(product)
                if i_r != j_r:
                    p_row = newton_rows.p[i_r]
                    qd.atomic_add(newton_rows.Ap[j_r], block.transpose() @ p_row)
                    energy = 2.0 * p_row.dot(product)
        func_add_by_key(i_r_key, product, newton_rows.Ap, n_components=3)
        if qd.static(not newton_config.is_rigid_deferred):
            func_add_by_key(i_b_key, energy, newton_envs.p_dot_ap)
    n_rows = newton_rows.n_rows[None]
    if qd.static(newton_config.is_rigid_deferred):
        func_rigid_matvec(rigid_dofs, constraint_state, rigid_info, rigid_config, newton_config)
        func_link_spatial_disp(rigid_dofs.p, rigid_links, dyn_state.dofs, dyn_info.links, newton_config)
        func_clear_link_forces(rigid_links)
        # The coupling of a contact: its Hessian H in the displacement of the particle relative to the point of the
        # link, between the nodes of the particle and the link, and between the link and itself
        n_particles = mpm_state.particles.pos.shape[1]
        inv_dx = mpm_info.inv_dx[None]
        for i_s in range(mpm_state.sorted_particles.n_valid[None]):
            i_g = mpm_state.rigid_contacts.geom_idx[i_s]
            if i_g >= 0:
                i_p, i_b, cell = func_sorted_particle(i_s, n_particles, mpm_state.sorted_particles)
                if newton_envs.is_pcg_active[i_b]:
                    x_p = mpm_state.particles.pos[f, i_p, i_b]
                    w, grad_w = func_q1_stencil(x_p * inv_dx - cell.cast(gs.qd_float), inv_dx)
                    slots = func_stencil_slots(i_b, cell, sparse_grid)
                    weights = qd.Vector(
                        [w[a] * gs.qd_float(sparse_grid.nodes_is_fixed[slots[a]] == 0) for a in qd.static(range(8))]
                    )
                    p_particle = qd.Vector.zero(gs.qd_float, 3)
                    for a in qd.static(range(8)):
                        p_particle += weights[a] * newton_rows.p[slots[a]]
                    i_l = dyn_info.geoms.link_idx[i_g]
                    r = x_p - dyn_state.links.root_COM[i_l, i_b]
                    p_partner = func_point_motion(rigid_links.motion_vel[i_l, i_b], rigid_links.motion_ang[i_l, i_b], r)
                    hessian = mpm_state.rigid_contacts.hessian[i_s]
                    product_partner = hessian @ p_partner
                    for a in qd.static(range(8)):
                        qd.atomic_add(newton_rows.Ap[slots[a]], -weights[a] * product_partner)
                    force = product_partner - hessian @ p_particle
                    qd.atomic_add(rigid_links.force_vel[i_l, i_b], force)
                    qd.atomic_add(rigid_links.force_ang[i_l, i_b], r.cross(force))
        func_link_force_to_dofs(rigid_dofs.Ap, rigid_links, dyn_state.dofs, dyn_info.links, newton_config)
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
            if qd.abs(p_dot_ap) > 0.0:
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
    if qd.static(newton_config.is_rigid_deferred):
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
                    newton_config,
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
    if qd.static(newton_config.is_rigid_deferred):
        for i_d, i_b in qd.ndrange(rigid_dofs.p.shape[0], rigid_dofs.p.shape[1]):
            rigid_dofs.Ap[i_d, i_b] = 0.0
            if newton_envs.is_pcg_active[i_b]:
                rigid_dofs.p[i_d, i_b] = rigid_dofs.z[i_d, i_b] + newton_envs.pcg_beta[i_b] * rigid_dofs.p[i_d, i_b]


@qd.kernel(graph=True)
def kernel_pcg_loop(
    f: int,
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
            f,
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
    f: int,
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
    'func_particle_inversion_step_cap' in mpm_solver.py).
    """
    for i_n in range(newton_rows.n_rows[None]):
        i_b = newton_rows.rows_env[i_n]
        if newton_envs.is_active[i_b]:
            sparse_grid.nodes_du[i_n] = -newton_rows.solution[i_n]
            sparse_grid.nodes_u_start[i_n] = sparse_grid.nodes_u[i_n]
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.max_displacement[i_b] = 0.0
        newton_envs.alpha[i_b] = 1.0
        newton_envs.n_line_search_iterations[i_b] = 0
    for i_s in range(mpm_state.sorted_particles.n_valid[None]):
        i_b, du = func_particle_displacement_update(f, i_s, mpm_state, sparse_grid, mpm_info)
        if newton_envs.is_active[i_b]:
            qd.atomic_max(newton_envs.max_displacement[i_b], qd.max(qd.abs(du).max(), 0.0))
            is_capped, alpha_cap = func_particle_inversion_step_cap(f, i_s, mpm_state, sparse_grid, mpm_info)
            if is_capped:
                qd.atomic_min(newton_envs.alpha[i_b], alpha_cap)
    if qd.static(newton_config.is_rigid_deferred):
        for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
            if newton_envs.is_active[i_b]:
                dx = -rigid_dofs.solution[i_d, i_b]
                rigid_dofs.dx[i_d, i_b] = dx
                rigid_dofs.x_start[i_d, i_b] = rigid_dofs.x[i_d, i_b]
                qd.atomic_max(newton_envs.max_displacement[i_b], qd.abs(dx))
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
    f: int,
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
    n_rows = sparse_grid.n_blocks[None] * 64
    for i_n in range(n_rows):
        i_b, block = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
        if newton_envs.is_line_search_active[i_b]:
            sparse_grid.nodes_u[i_n] = (
                sparse_grid.nodes_u_start[i_n] + newton_envs.alpha[i_b] * sparse_grid.nodes_du[i_n]
            )
    if qd.static(newton_config.is_rigid_deferred):
        for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
            if newton_envs.is_line_search_active[i_b]:
                rigid_dofs.x[i_d, i_b] = rigid_dofs.x_start[i_d, i_b] + newton_envs.alpha[i_b] * rigid_dofs.dx[i_d, i_b]
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.energy_trial[i_b] = 0.0
    func_add_energies(
        f,
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
    Hand the accelerations of the dofs solved to the rigid solver, which finishes its substep from them (see
    'RigidSolver.substep_post_coupling').

    The constraint state of the rigid solver takes their values, and its constraint force the whole force beyond the
    smooth one, the forces of the contacts of the particles included. Without constraint rows, the dofs take their
    acceleration and their force at once, which the constraint solve sets otherwise (see 'func_update_qacc' in the
    constraint solver of the rigid solver). An environment whose Newton solve failed keeps the state of the rigid
    solver until the error raises (see 'RigidInfo.is_env_frozen'), its dofs taking their smooth acceleration, which
    keeps every value the rigid solver computes from them finite.
    """
    dt = rigid_info.substep_dt[None]
    for i_b in range(rigid_info.is_env_frozen.shape[0]):
        rigid_info.is_env_frozen[i_b] = (errno[i_b] & MPMErrorCode.INVALID_NEWTON_NAN) != 0
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        if rigid_info.is_env_frozen[i_b]:
            rigid_dofs.x[i_d, i_b] = dt * dt * dyn_state.dofs.acc_smooth[i_d, i_b]
    func_rigid_evaluate(
        newton_envs.rigid_cost, rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, newton_config
    )
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        if qd.static(newton_config.has_rigid_rows):
            constraint_state.qacc[i_d, i_b] = rigid_dofs.qacc[i_d, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = rigid_dofs.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b]
        else:
            dyn_state.dofs.acc[i_d, i_b] = rigid_dofs.qacc[i_d, i_b]
            dyn_state.dofs.force[i_d, i_b] = rigid_dofs.Ma[i_d, i_b]


@qd.kernel
def kernel_flag_failures(newton_envs: NewtonEnvs, newton_loops: NewtonLoops, errno: qd.Tensor):
    """Flag the environments whose Newton solve met a value that is not finite, which keep the state of their particles
    (see 'kernel_transfer_from_sparse_grid' in mpm_solver.py) and of the rigid solver (see 'kernel_finish_rigid') until
    the error raises."""
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.failure[i_b] == NewtonFailure.NON_FINITE:
            qd.atomic_or(errno[i_b], MPMErrorCode.INVALID_NEWTON_NAN)


@qd.kernel
def kernel_clear_failed_iterations(newton_loops: NewtonLoops):
    for _ in range(1):
        newton_loops.has_failed_iterations[None] = 0
