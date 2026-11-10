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
from genesis.options.solvers import NewtonOptions
from genesis.utils.data import ENVS, Data, DataItem, DataKind, Size, field, iter_data, iter_state_arrays
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
    func_stencil_pair_block,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# The rows of the linear system of the Newton iterations, one per node of the sparse grid of the implicit MPM solver
ROWS = Size("rows")
# The smallest capacity of the rows, and the slack of a capacity grown past an overflow
MIN_ROWS = 64 * 64
ROWS_GROWTH_FACTOR = 1.2


class NewtonFailure(IntEnum):
    """The reason the Newton solve of an environment stopped short of convergence."""

    NONE = 0
    ITERATIONS = 1
    NON_FINITE = 2


class NewtonEnvs(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The progress and the scalars of the Newton solve of every environment.

    `is_active` says whether its Newton iterations go on, `is_converged` whether its last direction moved every particle
    by less than the tolerance, and `failure` why it stopped short of convergence (see NewtonFailure). `energy_start` is
    the energy at the start point of the line search, `energy_trial` at its trial point of step `alpha`, and
    `max_displacement` the largest update of the displacement of a particle along the direction. The scalars of the
    preconditioned conjugate gradient (PCG) follow: `rz` is r . z, `rz_tol` the value below which it stops, and
    `pcg_alpha` and `pcg_beta` its steps.
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


class NewtonLoops(Data, kind=DataKind.SCRATCH, shape=()):
    """
    The flags that keep the loops of the Newton solve running as long as an environment iterates on, with the
    iterations left to the PCG.

    The flags are ndarrays on every backend, since a loop of a graph tests the same ndarray on every call.
    `has_failed_iterations` latches a solve that reached its iteration cap since the last check (see
    'NewtonOptions.failure_check_interval').
    """

    is_newton_running: qd.Tensor = field(qd.i32, ndarray=True)
    is_line_search_running: qd.Tensor = field(qd.i32, ndarray=True)
    is_pcg_running: qd.Tensor = field(qd.i32, ndarray=True)
    n_pcg_iterations_left: qd.Tensor = field(int, ndarray=True)
    has_failed_iterations: qd.Tensor = field(int, ndarray=True)


class NewtonInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """The constants of the Newton solve (see NewtonOptions), the tolerance on the displacement of a particle being the
    tolerance on its velocity over a substep."""

    max_iterations: qd.Tensor = field(int)
    line_search_max_iterations: qd.Tensor = field(int)
    line_search_shrink: qd.Tensor = field(float)
    displacement_tol: qd.Tensor = field(float)
    pcg_max_iterations: qd.Tensor = field(int)
    pcg_tol: qd.Tensor = field(float)


class NewtonData(Data):
    """The arrays of the Newton solver of a fixed size, allocated once its scene is built."""

    newton_envs: NewtonEnvs
    newton_loops: NewtonLoops
    newton_info: NewtonInfo


class NewtonRows(Data, kind=DataKind.SCRATCH, shape=(ROWS,)):
    """
    The linear system of the Newton iterations, one row of 3 unknowns per node of the sparse grid, and its solve.

    `gradient` is the gradient of the incremental potential at the current iterate, the right-hand side, `solution` the
    solution of the Hessian against it, and `r`, `z`, `p` and `Ap` the residual, the preconditioned residual, the search
    direction and its product with the Hessian of the PCG. `jacobi` holds the inverses of the diagonal blocks of the
    Hessian, the block Jacobi preconditioner. `rows_env` is the environment of every row, the rows of an environment
    following one another. The matrix holds the upper triangle of the Hessian.
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


class NewtonSolver(Solver):
    """
    Solver of the substeps of the implicit MPM solver by Newton iterations on their incremental potential.

    The unknown of a substep is the displacement of the nodes of the sparse grid of the MPM solver over it (see
    MPMSparseGrid). Every iteration solves the projected Hessian against the gradient by a preconditioned conjugate
    gradient (PCG) and searches a step along the direction that lowers the energy, every environment iterating until
    its own convergence. The solve runs once every solver has prepared its substep and the coupler has run, and before
    the solvers finish it, since the MPM solver moves its particles by the displacement solved.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: NewtonOptions) -> None:
        super().__init__(scene, sim, options)
        self.data_manager: NewtonData | None = None
        self.newton_envs: NewtonEnvs | None = None
        self.newton_loops: NewtonLoops | None = None
        self.newton_info: NewtonInfo | None = None
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
        """Whether the rigid solver solves its substeps within the Newton solve."""
        return False

    def build(self) -> None:
        super().build()
        if not self.is_active:
            return
        self.data_manager = NewtonData(
            sizes={"envs": self._B},
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
        )
        self.newton_envs = self.data_manager.newton_envs
        self.newton_loops = self.data_manager.newton_loops
        self.newton_info = self.data_manager.newton_info

    @property
    def data(self) -> Iterator[DataItem]:
        if self.data_manager is not None:
            yield from iter_data(self.data_manager, "newton_info")
            yield from iter_data(self.data_manager, "newton_envs")
            yield from iter_data(self.data_manager, "newton_loops")
        if self.newton_rows is not None:
            yield from iter_data(self, "newton_rows")

    def substep_pre_coupling(self, f: int) -> None:
        pass

    def substep_post_coupling(self, f: int) -> None:
        """Solve the substep of the implicit MPM solver, before the solvers finish it."""
        if not self.is_active:
            return
        mpm_solver = self._sim.mpm_solver
        mpm_state, mpm_info, sparse_grid = mpm_solver.mpm_state, mpm_solver.mpm_info, mpm_solver.sparse_grid
        errno = mpm_solver._errno
        self.newton_rows = update_newton_rows(
            int(qd_to_numpy(sparse_grid.n_blocks)) * 64,
            int(qd_to_numpy(mpm_state.sorted_particles.n_tasks)),
            self.newton_rows,
            self._newton_rows_replaced,
        )
        args_mpm = (mpm_state, sparse_grid)
        kernel_start_solve(f, self.newton_envs, self.newton_loops, *args_mpm, mpm_info, errno)
        for _ in range(self._options.max_iterations):
            kernel_assemble(f, self.newton_rows, self.newton_envs, *args_mpm, mpm_info)
            assemble_bcoo_matrix(self.newton_rows.matrix, block_scalars=9)
            kernel_pcg_start(self.newton_rows, self.newton_envs, self.newton_loops, self.newton_info)
            kernel_pcg_loop(self.newton_rows, self.newton_envs, self.newton_loops)
            kernel_start_line_search(
                f, self.newton_rows, self.newton_envs, self.newton_loops, *args_mpm, self.newton_info, mpm_info
            )
            while True:
                kernel_line_search_trial(f, self.newton_envs, self.newton_loops, *args_mpm, self.newton_info, mpm_info)
                if not qd_to_numpy(self.newton_loops.is_line_search_running):
                    break
            kernel_end_iteration(self.newton_envs, self.newton_loops, self.newton_info)
            if not qd_to_numpy(self.newton_loops.is_newton_running):
                break
        kernel_flag_failures(self.newton_envs, self.newton_loops, errno)
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
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    mpm_info: MPMInfo,
):
    """Add the incremental potential at the displacement of the nodes into the entry of every environment that
    `envs_mask` selects, the particles and the nodes of an environment following one another."""
    n_valid = mpm_state.sorted_particles.n_valid[None]
    qd.loop_config(block_dim=256)
    for i_s in range((n_valid + 31) // 32 * 32):
        i_b_key = gs.qd_int(-1)
        energy = gs.qd_float(0.0)
        if i_s < n_valid:
            i_b, energy_particle = func_particle_energy(f, i_s, mpm_state, sparse_grid, mpm_info)
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
            i_b, _ = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
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


@qd.kernel
def kernel_start_solve(
    f: int,
    newton_envs: NewtonEnvs,
    newton_loops: NewtonLoops,
    mpm_state: MPMState,
    sparse_grid: MPMSparseGrid,
    mpm_info: MPMInfo,
    errno: qd.Tensor,
):
    """Start the Newton solve of every environment at zero displacement, but those whose solve failed earlier in the
    step, and evaluate its energy there."""
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
    func_add_energies(f, newton_envs.energy_start, newton_envs.is_active, mpm_state, sparse_grid, mpm_info)


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
    mpm_info: MPMInfo,
):
    """
    Assemble the gradient and the projected Hessian of the incremental potential at the displacement of the nodes.

    Every node takes its kinetic block, an identity on a fixed node or a node without mass, whose unknown stays zero.
    Every task of the sorted particles (see 'kernel_list_cells' in mpm_solver.py) writes the 36 blocks of the upper
    triangle of the stencil of its cell, summed over its particles: the lanes of a warp sum them on GPU, and one thread
    walks the particles of a cell on CPU. The gradient of a node then gathers the factors its particles staged.
    """
    n_nodes = sparse_grid.n_blocks[None] * 64
    n_valid = mpm_state.sorted_particles.n_valid[None]
    for _ in range(1):
        newton_rows.n_rows[None] = n_nodes
        func_report_triplet_demand(n_nodes + 36 * mpm_state.sorted_particles.n_tasks[None], newton_rows.matrix)
    for i_n in range(n_nodes):
        i_b, _ = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
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
        for i_c in range(mpm_state.sorted_particles.n_cells[None]):
            i_s_start = mpm_state.sorted_particles.cells_start[i_c]
            i_t_start = n_nodes + 36 * mpm_state.sorted_particles.tasks_idx[i_s_start]
            for i_s in range(i_s_start, mpm_state.sorted_particles.cells_start[i_c + 1]):
                slots, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian = (
                    func_particle_stencil_terms(f, i_s, mpm_state, sparse_grid, mpm_info)
                )
                for i_pair in range(36):
                    a = pairs_a[i_pair]
                    b = pairs_b[i_pair]
                    block = func_stencil_pair_block(
                        a, b, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian
                    )
                    if i_s == i_s_start:
                        func_write_triplet(
                            i_t_start + i_pair, slots[a], slots[b], block, newton_rows.matrix, is_symmetric=True
                        )
                    else:
                        func_add_triplet(
                            i_t_start + i_pair, slots[a], slots[b], block, newton_rows.matrix, is_symmetric=True
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
            slots, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian = (
                func_particle_stencil_terms(f, i_s_clamped, mpm_state, sparse_grid, mpm_info)
            )
            for i_pair in range(36):
                a = pairs_a[i_pair]
                b = pairs_b[i_pair]
                block = gs.qd_float(is_valid) * func_stencil_pair_block(
                    a, b, r_frame, U, S_block, alpha, beta, fibre_weights, point_weights, point_hessian
                )
                is_tail = False
                for j, k in qd.static(qd.ndrange(3, 3)):
                    block_jk, is_tail = qd_segmented_sum(tid, i_task, i_task_prev, i_task_next, block[j, k])
                    block[j, k] = block_jk
                if is_tail:
                    func_write_triplet(
                        n_nodes + 36 * i_task + i_pair, slots[a], slots[b], block, newton_rows.matrix, is_symmetric=True
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
            if mass > 0.0 and sparse_grid.nodes_is_fixed[i_n] == 0:
                u = sparse_grid.nodes_u[i_n]
                gradient_node = gradient + mass * (u - dt * sparse_grid.nodes_vel_free[i_n])
            newton_rows.gradient[i_n] = gradient_node


# ------------------------------------------------------------------------------------
# -------------------------- preconditioned conjugate gradient -----------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_pcg_start(
    newton_rows: NewtonRows, newton_envs: NewtonEnvs, newton_loops: NewtonLoops, newton_info: NewtonInfo
):
    """
    Start the PCG of the environments still iterating from a zero solution, its preconditioner inverting the diagonal
    blocks of the Hessian.

    A diagonal block is inverted equilibrated to a unit diagonal, which keeps the inverse accurate for the rows of very
    different stiffness. An environment stops once r . z falls below `pcg_tol` times its value at the start.
    """
    for i_e in range(newton_rows.matrix.n_entries[None]):
        i_r, i_c, block = func_read_entry(i_e, newton_rows.matrix, block_rows=3, block_cols=3)
        if i_r == i_c:
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
    for _ in range(1):
        newton_loops.is_pcg_running[None] = 0
        newton_loops.n_pcg_iterations_left[None] = newton_info.pcg_max_iterations[None]
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.rz_tol[i_b] = newton_info.pcg_tol[None] * qd.abs(newton_envs.rz[i_b])
        if newton_envs.is_pcg_active[i_b]:
            if newton_envs.rz[i_b] > 0.0:
                qd.atomic_max(newton_loops.is_pcg_running[None], 1)
            else:
                newton_envs.is_pcg_active[i_b] = False


@qd.func(requires_top_level=True)
def func_pcg_iteration(newton_rows: NewtonRows, newton_envs: NewtonEnvs, newton_loops: NewtonLoops):
    """Run one iteration of the PCG of the environments still solving."""
    # The product Ap of the Hessian with the search direction, in the energy form p . Ap of every environment, the upper
    # triangle standing for the transposed blocks of the lower one
    n_entries = newton_rows.matrix.n_entries[None]
    qd.loop_config(block_dim=256)
    for i_e in range((n_entries + 31) // 32 * 32):
        i_r_key = gs.qd_int(-1)
        i_b_key = gs.qd_int(-1)
        product = qd.Vector.zero(gs.qd_float, 3)
        energy = gs.qd_float(0.0)
        if i_e < n_entries:
            i_r, i_c, block = func_read_entry(i_e, newton_rows.matrix, block_rows=3, block_cols=3)
            i_b = newton_rows.rows_env[i_r]
            if newton_envs.is_pcg_active[i_b]:
                i_r_key = i_r
                i_b_key = i_b
                p_col = newton_rows.p[i_c]
                product = block @ p_col
                energy = p_col.dot(product)
                if i_r != i_c:
                    p_row = newton_rows.p[i_r]
                    qd.atomic_add(newton_rows.Ap[i_c], block.transpose() @ p_row)
                    energy = 2.0 * p_row.dot(product)
        func_add_by_key(i_r_key, product, newton_rows.Ap, n_components=3)
        func_add_by_key(i_b_key, energy, newton_envs.p_dot_ap)
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_pcg_active[i_b]:
            p_dot_ap = newton_envs.p_dot_ap[i_b]
            pcg_alpha = gs.qd_float(0.0)
            if qd.abs(p_dot_ap) > 0.0:
                pcg_alpha = newton_envs.rz[i_b] / p_dot_ap
            newton_envs.pcg_alpha[i_b] = pcg_alpha
        newton_envs.rz_new[i_b] = 0.0
        newton_envs.p_dot_ap[i_b] = 0.0
    n_rows = newton_rows.n_rows[None]
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
    for _ in range(1):
        newton_loops.n_pcg_iterations_left[None] = newton_loops.n_pcg_iterations_left[None] - 1
        newton_loops.is_pcg_running[None] = 0
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.is_pcg_active[i_b]:
            rz = newton_envs.rz[i_b]
            rz_new = newton_envs.rz_new[i_b]
            newton_envs.pcg_beta[i_b] = rz_new / rz
            newton_envs.rz[i_b] = rz_new
            if qd.abs(rz_new) <= newton_envs.rz_tol[i_b] or not qd_isfinite(rz_new):
                newton_envs.is_pcg_active[i_b] = False
            elif newton_loops.n_pcg_iterations_left[None] > 0:
                qd.atomic_max(newton_loops.is_pcg_running[None], 1)
    for i_r in range(n_rows):
        newton_rows.Ap[i_r] = qd.Vector.zero(gs.qd_float, 3)
        i_b = newton_rows.rows_env[i_r]
        if newton_envs.is_pcg_active[i_b]:
            newton_rows.p[i_r] = newton_rows.z[i_r] + newton_envs.pcg_beta[i_b] * newton_rows.p[i_r]


@qd.kernel(graph=True)
def kernel_pcg_loop(newton_rows: NewtonRows, newton_envs: NewtonEnvs, newton_loops: NewtonLoops):
    """Run the PCG iterations as one graph, until every environment converges or the iterations run out."""
    while qd.graph.do_while(newton_loops.is_pcg_running):
        func_pcg_iteration(newton_rows, newton_envs, newton_loops)


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
    newton_info: NewtonInfo,
    mpm_info: MPMInfo,
):
    """
    Set the Newton direction of every environment still iterating, minus the solution of the PCG, and start its line
    search from the full step.

    An environment converges once its direction moves every particle by less than the tolerance, the step being taken
    still. The first step stays short of the first zero volume of a liquid particle (see
    'func_particle_inversion_step_cap' in mpm_solver.py).
    """
    for i_n in range(newton_rows.n_rows[None]):
        if newton_envs.is_active[newton_rows.rows_env[i_n]]:
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
    newton_info: NewtonInfo,
    mpm_info: MPMInfo,
):
    """
    Move the nodes of every environment still searching to the trial point of its step, and accept the step once the
    energy there is at most the energy at the start point, or shrink it.

    A converged or failed environment takes its step at once, and an environment out of trials keeps its last one.
    """
    n_rows = sparse_grid.n_blocks[None] * 64
    for i_n in range(n_rows):
        i_b, _ = func_block_of_key(sparse_grid.blocks_key[i_n // 64])
        if newton_envs.is_line_search_active[i_b]:
            sparse_grid.nodes_u[i_n] = (
                sparse_grid.nodes_u_start[i_n] + newton_envs.alpha[i_b] * sparse_grid.nodes_du[i_n]
            )
    for i_b in range(newton_envs.is_active.shape[0]):
        newton_envs.energy_trial[i_b] = 0.0
    func_add_energies(f, newton_envs.energy_trial, newton_envs.is_line_search_active, mpm_state, sparse_grid, mpm_info)
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
def kernel_flag_failures(newton_envs: NewtonEnvs, newton_loops: NewtonLoops, errno: qd.Tensor):
    """Flag the environments whose Newton solve met a value that is not finite, which keep their state (see
    'kernel_transfer_from_sparse_grid' in mpm_solver.py) until the error raises."""
    for i_b in range(newton_envs.is_active.shape[0]):
        if newton_envs.failure[i_b] == NewtonFailure.NON_FINITE:
            qd.atomic_or(errno[i_b], MPMErrorCode.INVALID_NEWTON_NAN)


@qd.kernel
def kernel_clear_failed_iterations(newton_loops: NewtonLoops):
    for _ in range(1):
        newton_loops.has_failed_iterations[None] = 0
