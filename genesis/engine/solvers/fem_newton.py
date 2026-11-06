"""Implicit finite element method (FEM) as a participant of the NewtonSolver.

The FEM unknown is the end-of-substep vertex position x, starting at the substep start. Its share of the joint energy is
FEM's own incremental potential (inertia, elasticity and the linearized damping of its implicit solver) times dt^2, in
the displacement units of the material point method (MPM) energy. FEM owns the rows of its vertices in the linear system
of the solve (see newton_linear.py): it assembles the mass block of every vertex and the 10 blocks of the upper triangle
of every element, and preconditions them by their diagonal blocks. The solution is minus the Newton direction. A
vertex FEM holds at its target is an eliminated row (see newton_linear.py and is_free).

The joint solve replaces FEM's own implicit substep (`FEMSolver.batch_solve`): FEM prepares its inertia and material
data, the joint Newton iterations move its end-of-substep positions together with the other participants, and FEM takes
its velocities from them and runs the rest of its coupling after the solve (see FEMSolver.build_newton_participant).
"""

import quadrants as qd

import genesis as gs
from genesis.utils.data import StaticConfig, iter_data

from .newton_linear import (
    BlockJacobiScratch,
    LinearBlock,
    NewtonLinearScratch,
    block_jacobi_visit,
    build_block_jacobi,
    build_linear_block,
    discard_block,
    func_block_jacobi_apply,
    func_block_jacobi_invert,
    func_block_jacobi_reset,
    row_vector,
    write_block,
)
from .newton_solver import NewtonDevice, NewtonHost

# Blocks of the upper triangle of the 4 x 4 vertex blocks of an element
ELEMENT_BLOCKS = 10


# ============================================ Layer 1: the FEM block ============================================


@qd.func
def is_free(i_b: int, i_v: int, fem: qd.template()):
    """Whether a vertex is an unknown of the solve, not held at its target by a vertex constraint."""
    is_constrained = False
    if qd.static(fem.config.has_vertex_constraints):
        is_constrained = fem.vertex_constraints.is_constrained[i_v, i_b]
    return not is_constrained


@qd.func(requires_top_level=True)
def func_fem_assemble(f: int, dt: float, fem: qd.template(), linear: NewtonLinearScratch, rows: LinearBlock):
    """Write the gradient -dt^2 force of every vertex, its mass block and the blocks of every element, all scaled to
    the joint energy.

    The block of vertices (k, l) of an element is V (1 + beta / dt) sum_ij S_ki S_lj H_ij, H being the Hessian of the
    energy density in the deformation gradient, and S the mapping matrix of the element (see FEMSolver.compute_Ap). A
    constrained vertex takes an identity block and a zero gradient, and the blocks of the elements on it are
    discarded."""
    fem.solver.func_compute_ele_hessian_gradient(f)
    fem.solver.func_accumulate_vertex_force_preconditioner(f)
    dt_sq = dt * dt
    qd.loop_config(name="mpm_fem_assemble_vertices")
    for i_b, i_v in qd.ndrange(fem.n_envs, fem.n_vertices):
        i_r = i_b * fem.n_vertices + i_v
        row = rows.row_start[()] + i_r
        mass = gs.qd_float(1.0)
        g = qd.Vector.zero(gs.qd_float, 3)
        if is_free(i_b, i_v, fem):
            mass = dt_sq * fem.elements_v_info[i_v].mass_over_dt2 * (1.0 + fem.damping_alpha_dt)
            g = -dt_sq * fem.elements_v_energy[i_b, i_v].force
        write_block(rows.triplet_start[()] + i_r, row, row, mass * qd.Matrix.identity(gs.qd_float, 3), linear)
        for d in qd.static(range(3)):
            linear.b_rhs[3 * row + d] = g[d]
        linear.pcg.row_env[row] = i_b
    qd.loop_config(name="mpm_fem_assemble_elements")
    for i_b, i_e in qd.ndrange(fem.n_envs, fem.n_elements):
        i_vs = fem.elements_i[i_e].el2v
        S = fem.solver._func_compute_element_mapping_matrix(i_vs, fem.elements_i[i_e].B, i_b)
        scale = dt_sq * fem.elements_i[i_e].V * fem.damping_beta_factor
        i_t = rows.triplet_start[()] + fem.n_envs * fem.n_vertices + ELEMENT_BLOCKS * (i_b * fem.n_elements + i_e)
        row_start = rows.row_start[()] + i_b * fem.n_vertices
        for k_r in qd.static(range(4)):
            for k_c in qd.static(range(k_r, 4)):
                i_block = qd.static(4 * k_r - k_r * (k_r - 1) // 2 + k_c - k_r)
                if is_free(i_b, i_vs[k_r], fem) and is_free(i_b, i_vs[k_c], fem):
                    block = qd.Matrix.zero(gs.qd_float, 3, 3)
                    for i, j in qd.static(qd.ndrange(3, 3)):
                        block += (S[k_r, i] * S[k_c, j]) * fem.elements_el_hessian[i_b, i, j, i_e]
                    write_block(i_t + i_block, row_start + i_vs[k_r], row_start + i_vs[k_c], scale * block, linear)
                else:
                    discard_block(i_t + i_block, linear)


@qd.func
def fem_solution(i_b: int, i_v: int, fem: qd.template()):
    """The solution of the linear system on a vertex, minus its Newton direction."""
    return row_vector(fem.linear_block.row_start[()] + i_b * fem.n_vertices + i_v, fem.linear.solution)


@qd.func(requires_top_level=True)
def func_fem_start_line_search(f: int, fem: qd.template(), control: qd.template()):
    """The line-search start point, and the largest component of the step into the Newton measure."""
    qd.loop_config(name="mpm_fem_start_line_search")
    for i_b, i_v in qd.ndrange(fem.n_envs, fem.n_vertices):
        fem.linesearch_state_v[i_b, i_v].x_prev = fem.elements_v[f + 1, i_v, i_b].pos
        qd.atomic_max(control.max_disp[i_b], qd.abs(fem_solution(i_b, i_v, fem)).max())


@qd.func(requires_top_level=True)
def func_fem_step_forward(f: int, fem: qd.template(), control: qd.template()):
    qd.loop_config(name="mpm_fem_step_forward")
    for i_b, i_v in qd.ndrange(fem.n_envs, fem.n_vertices):
        fem.elements_v[f + 1, i_v, i_b].pos = fem.linesearch_state_v[i_b, i_v].x_prev - control.alpha[
            i_b
        ] * fem_solution(i_b, i_v, fem)


@qd.func(requires_top_level=True)
def func_fem_energy(f: int, dt: float, fem: qd.template(), control: qd.template(), is_trial: qd.template()):
    """Add FEM's incremental potential at the end-of-substep positions, times dt^2, to the start or trial energy.

    The potential holds the inertia and the linearized damping of the vertices, and the elastic energy and damping of
    the elements, at the positions of frame f + 1."""
    dt_sq = dt * dt
    damping_alpha_dt = fem.damping_alpha_dt
    qd.loop_config(name="mpm_fem_energy_vertices")
    for i_b, i_v in qd.ndrange(fem.n_envs, fem.n_vertices):
        diff = fem.elements_v[f + 1, i_v, i_b].pos - fem.elements_v_energy[i_b, i_v].inertia
        energy = 0.5 * fem.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
        if qd.static(fem.config.has_damping):
            step = fem.elements_v[f + 1, i_v, i_b].pos - fem.elements_v[f, i_v, i_b].pos
            energy += 0.5 * fem.elements_v_info[i_v].mass_over_dt2 * step.dot(step) * damping_alpha_dt
        if is_trial:
            qd.atomic_add(control.energy_trial[i_b], dt_sq * energy)
        else:
            qd.atomic_add(control.energy_start[i_b], dt_sq * energy)
    fem.solver.func_compute_ele_energy(f)
    qd.loop_config(name="mpm_fem_energy_elements")
    for i_b, i_e in qd.ndrange(fem.n_envs, fem.n_elements):
        energy = fem.elements_el_energy[i_b, i_e].energy * fem.elements_i[i_e].V
        if is_trial:
            qd.atomic_add(control.energy_trial[i_b], dt_sq * energy)
        else:
            qd.atomic_add(control.energy_start[i_b], dt_sq * energy)


# ============================================ Layer 2: participant =============================================


class FemNewtonConfig(StaticConfig):
    """Compile-time switches of FEM's stages in the joint solve: the code paths a scene takes, never its sizes."""

    # Some vertices are constrained to a target, which the preconditioner leaves in place
    has_vertex_constraints: bool
    # The implicit solver damps the vertex velocity
    has_damping: bool


@qd.data_oriented
class FemNewtonParticipant(NewtonDevice):
    """Device stages of the FEM vertex positions in the joint solve.

    It is the one place that selects the FEM arrays and scalars the stages and the couplings read, and `solver` gives
    the stages access to the functions of FEM's own implicit solver. The step sizes are plain floats, compiled into the
    stages as constants. The arrays are properties of the solver, so that
    the walk over the data of the NewtonSolver (see iter_data_oriented) stops at the solver, which owns them."""

    # The rows of a vertex are its own
    n_vertex_rows = 1

    def __init__(self, fem, n_envs: int):
        self.solver = fem
        self.n_envs = n_envs
        self.n_vertices = fem.n_vertices
        # The vertices of every environment are contact vertices (see newton_contact.py)
        self.vertex_offset = qd.ndarray(gs.qd_int, shape=())
        self.vertex_offset.fill(0)
        self.vertex_count = qd.ndarray(gs.qd_int, shape=())
        self.vertex_count.fill(n_envs * fem.n_vertices)
        self.n_elements = fem.n_elements
        self.n_surfaces = fem.n_surfaces
        self.config = FemNewtonConfig(
            has_vertex_constraints=fem._enable_vertex_constraints, has_damping=fem._damping_alpha > 0.0
        )
        self.substep_dt = fem.substep_dt
        self.damping_alpha_dt = fem._damping_alpha * fem._substep_dt
        self.damping_beta_factor = 1.0 + fem._damping_beta / fem._substep_dt
        # The linear system of the solve, bound by the NewtonSolver, and the rows of the vertices in it
        self.linear: NewtonLinearScratch | None = None
        n_rows = n_envs * fem.n_vertices
        self.linear_block: LinearBlock = build_linear_block(n_rows, n_rows + ELEMENT_BLOCKS * n_envs * fem.n_elements)
        self.jacobi: BlockJacobiScratch = build_block_jacobi(n_rows)
        self.max_linear_rows = n_rows

    @property
    def data(self):
        yield from iter_data(self, "linear_block")
        yield from iter_data(self, "jacobi")

    @property
    def vertex_constraints(self):
        """The constrained vertices, which exist only when the solver enables vertex constraints."""
        return self.solver.vertex_constraints

    @property
    def surface(self):
        return self.solver.surface

    @property
    def batch_active(self):
        return self.solver.batch_active

    @property
    def batch_pcg_active(self):
        return self.solver.batch_pcg_active

    @property
    def batch_linesearch_active(self):
        return self.solver.batch_linesearch_active

    @property
    def elements_v(self):
        return self.solver.elements_v

    @property
    def elements_i(self):
        return self.solver.elements_i

    @property
    def elements_v_info(self):
        return self.solver.elements_v_info

    @property
    def elements_v_energy(self):
        return self.solver.elements_v_energy

    @property
    def elements_el_energy(self):
        return self.solver.elements_el_energy

    @property
    def elements_el_hessian(self):
        return self.solver.elements_el_hessian

    @property
    def linesearch_state_v(self):
        return self.solver.linesearch_state_v

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int, is_exact: qd.template()):
        func_fem_assemble(f, self.substep_dt, self, self.linear, self.linear_block)

    @qd.func(requires_top_level=True)
    def func_factor_begin(self):
        func_block_jacobi_reset(self.linear_block.n_rows[()], self.jacobi)

    @qd.func
    def func_linear_visit_block(self, i_r: int, i_c: int, H: qd.types.matrix(3, 3)):
        block_jacobi_visit(i_r, i_c, H, self.jacobi)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        func_block_jacobi_invert(self.linear_block.n_rows[()], self.jacobi)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        func_block_jacobi_apply(self.linear_block, self.linear, self.jacobi)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        func_fem_start_line_search(f, self, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        func_fem_step_forward(f, self, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f: int, is_trial: qd.template()):
        func_fem_energy(f, self.substep_dt, self, control, is_trial)

    # ------------------------------------------- vertex map -------------------------------------------

    @qd.func
    def func_vertex_value(self, f: int, i_v: int):
        i_b = i_v // self.n_vertices
        i_fv = i_v - i_b * self.n_vertices
        return self.elements_v[f + 1, i_fv, i_b].pos - self.elements_v[f, i_fv, i_b].pos

    @qd.func
    def func_vertex_rows(self, i_v: int):
        i_b = i_v // self.n_vertices
        rows = qd.Vector([self.linear_block.row_start[()] + i_v], dt=gs.qd_int)
        weights = qd.Vector([gs.qd_float(is_free(i_b, i_v - i_b * self.n_vertices, self))], dt=gs.qd_float)
        return 1, rows, weights


@qd.kernel
def kernel_fem_begin(fem: qd.template()):
    for i_b in range(fem.n_envs):
        fem.batch_active[i_b] = True
        fem.batch_pcg_active[i_b] = True
        fem.batch_linesearch_active[i_b] = True


@qd.kernel
def kernel_fem_freeze(f: int, fem: qd.template(), control: qd.template()):
    for i_v, i_b in qd.ndrange(fem.n_vertices, fem.n_envs):
        if control.has_failed_non_finite[i_b] != 0:
            fem.elements_v[f + 1, i_v, i_b].pos = fem.elements_v[f, i_v, i_b].pos
            fem.elements_v[f + 1, i_v, i_b].vel = fem.elements_v[f, i_v, i_b].vel


class FemNewtonHost(NewtonHost):
    """Host stages of FEM around the joint solve: the inertia and material data of its substep before, and its
    velocities from the solved positions after.

    Once FEM has its velocities, the host runs the part of the coupler's step on FEM that follows FEM's own solve in a
    substep FEM solves alone (see LegacyCoupler.couple_fem)."""

    def __init__(self, fem_solver, device: FemNewtonParticipant):
        self._fem_solver = fem_solver
        self._device = device

    def newton_prepare(self, f):
        self._fem_solver.precompute_material_data(f)
        self._fem_solver.init_pos_and_inertia(f, self._fem_solver.sim.rigid_solver.dyn_state.links)
        kernel_fem_begin(self._device)

    def newton_finish(self, f, control):
        self._fem_solver.setup_pos_vel(f)
        kernel_fem_freeze(f, self._device, control)

    def newton_after(self, f):
        self._fem_solver.sim.coupler.couple_fem(f)

    def newton_end(self, f, control):
        # FEM takes the vertex positions of the substep from the velocities here, in place of its post-coupling phase,
        # so that the mask of the solve still holds
        self._fem_solver.compute_pos(f)
        kernel_fem_freeze(f, self._device, control)
