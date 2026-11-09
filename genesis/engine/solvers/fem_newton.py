"""Implicit finite element method (FEM) as a participant of the NewtonSolver."""

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.newton_control import NewtonControl
from genesis.utils.data import StaticConfig, iter_data

from .newton_contact import VertexRange
from .newton_linear import (
    BlockJacobiOwner,
    BlockJacobiScratch,
    LinearBlock,
    NewtonLinearScratch,
    build_block_jacobi,
    build_linear_block,
    discard_block,
    write_block,
)
from .newton_solver import NewtonDevice, NewtonHost

# Blocks of the upper triangle of the 4 x 4 vertex blocks of an element
ELEMENT_BLOCKS = 10


# ============================================ Layer 2: participant =============================================


class FemNewtonConfig(StaticConfig):
    """Compile-time switches of FEM's stages in the joint solve: the code paths a scene takes, never its sizes."""

    # Some vertices are constrained to a target, which the preconditioner leaves in place
    has_vertex_constraints: bool
    # The implicit solver damps the vertex velocity
    has_damping: bool


@qd.data_oriented
class FemNewtonParticipant(BlockJacobiOwner, NewtonDevice):
    """Device stages of the FEM vertex positions in the joint solve."""

    # The rows of a vertex are its own
    n_vertex_rows = 1

    def __init__(self, fem, n_envs: int):
        self.solver = fem
        self.n_envs = n_envs
        self.n_vertices = fem.n_vertices
        # The vertices of every environment are contact vertices (see newton_contact.py)
        self.vertex_range = VertexRange(values={"offset": 0, "count": n_envs * fem.n_vertices})
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
        yield from iter_data(self, "vertex_range")
        yield from iter_data(self, "linear_block")
        yield from iter_data(self, "jacobi")

    @qd.func
    def is_free(self, i_b: int, i_v: int):
        """Whether a vertex is an unknown of the solve, not held at its target by a vertex constraint."""
        is_constrained = False
        if qd.static(self.config.has_vertex_constraints):
            is_constrained = self.solver.vertex_constraints.is_constrained[i_v, i_b]
        return not is_constrained

    @qd.func
    def solution(self, i_b: int, i_v: int):
        """The solution of the linear system on a vertex, minus its Newton direction."""
        return self.linear.solution[self.linear_block.row_start[()] + i_b * self.n_vertices + i_v]

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int, is_exact: qd.template()):
        """Write the gradient -dt^2 force of every vertex, its mass block and the blocks of every element, all scaled to
        the joint energy."""
        self.solver.func_compute_ele_energy_gradient(f)
        self.solver._func_accumulate_vertices(f, self.solver.batch_active, gather_Ap=False)
        dt_sq = self.substep_dt * self.substep_dt
        qd.loop_config(name="mpm_fem_assemble_vertices")
        for i_b, i_v in qd.ndrange(self.n_envs, self.n_vertices):
            i_r = i_b * self.n_vertices + i_v
            row = self.linear_block.row_start[()] + i_r
            mass = gs.qd_float(1.0)
            g = qd.Vector.zero(gs.qd_float, 3)
            if self.is_free(i_b, i_v):
                mass = dt_sq * self.solver.elements_v_info[i_v].mass_over_dt2 * (1.0 + self.damping_alpha_dt)
                g = -dt_sq * self.solver.elements_v_energy[i_b, i_v].force
            i_t = self.linear_block.triplet_start[()] + i_r
            write_block(i_t, row, row, mass * qd.Matrix.identity(gs.qd_float, 3), self.linear)
            self.linear.b_rhs[row] = g
            self.linear.pcg.rows_env[row] = i_b
        qd.loop_config(name="mpm_fem_assemble_elements")
        for i_b, i_e in qd.ndrange(self.n_envs, self.n_elements):
            i_vs = self.solver.elements_i[i_e].el2v
            S = self.solver._func_compute_element_mapping_matrix(i_vs, self.solver.elements_i[i_e].B, i_b)
            scale = dt_sq * self.solver.elements_i[i_e].V * self.damping_beta_factor
            i_t = self.linear_block.triplet_start[()] + self.n_envs * self.n_vertices
            i_t += ELEMENT_BLOCKS * (i_b * self.n_elements + i_e)
            row_start = self.linear_block.row_start[()] + i_b * self.n_vertices
            for k_c in qd.static(range(4)):
                vertex_unit = qd.Vector([gs.qd_float(k_c == i) for i in qd.static(range(4))])
                # Row 3 k_r + a, column c: the force on vertex k_r along a of a unit displacement of vertex k_c along c
                columns = qd.Matrix.zero(gs.qd_float, 12, 3)
                # A runtime loop over the axes keeps a single inlined instance of the product per vertex
                for c in range(3):
                    axis_unit = qd.Vector([gs.qd_float(c == i) for i in qd.static(range(3))])
                    _, dP = self.solver.compute_elastic_product(i_e, i_b, S, axis_unit.outer_product(vertex_unit))
                    forces = dP @ S.transpose()
                    column = qd.Vector([forces[a, k_r] for k_r in qd.static(range(4)) for a in qd.static(range(3))])
                    columns = columns + column.outer_product(axis_unit)
                for k_r in qd.static(range(k_c + 1)):
                    i_block = qd.static(4 * k_r - k_r * (k_r - 1) // 2 + k_c - k_r)
                    if self.is_free(i_b, i_vs[k_r]) and self.is_free(i_b, i_vs[k_c]):
                        block = qd.Matrix(
                            [[columns[3 * k_r + a, c] for c in qd.static(range(3))] for a in qd.static(range(3))]
                        )
                        rows = row_start + i_vs
                        write_block(i_t + i_block, rows[k_r], rows[k_c], scale * block, self.linear)
                    else:
                        discard_block(i_t + i_block, self.linear)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: NewtonControl, f: int):
        """Record the line-search start point, and add the largest component of the step into the Newton measure."""
        qd.loop_config(name="mpm_fem_start_line_search")
        for i_b, i_v in qd.ndrange(self.n_envs, self.n_vertices):
            self.solver.linesearch_state_v[i_b, i_v].x_prev = self.solver.elements_v[f + 1, i_v, i_b].pos
            qd.atomic_max(control.max_displacement[i_b], qd.abs(self.solution(i_b, i_v)).max())

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: NewtonControl, f: int):
        qd.loop_config(name="mpm_fem_step_forward")
        for i_b, i_v in qd.ndrange(self.n_envs, self.n_vertices):
            x_prev = self.solver.linesearch_state_v[i_b, i_v].x_prev
            self.solver.elements_v[f + 1, i_v, i_b].pos = x_prev - control.alpha[i_b] * self.solution(i_b, i_v)

    @qd.func(requires_top_level=True)
    def func_energy(self, f: int, energies: qd.template()):
        """Add FEM's incremental potential times dt^2 into the entry of every environment.

        The potential sums the inertia and the linearized damping of the vertices and the elastic energy and damping of
        the elements, at the end-of-substep positions of frame f + 1, into `energies`."""
        dt_sq = self.substep_dt * self.substep_dt
        qd.loop_config(name="mpm_fem_energy_vertices")
        for i_b, i_v in qd.ndrange(self.n_envs, self.n_vertices):
            mass_over_dt2 = self.solver.elements_v_info[i_v].mass_over_dt2
            diff = self.solver.elements_v[f + 1, i_v, i_b].pos - self.solver.elements_v_energy[i_b, i_v].inertia
            energy = 0.5 * mass_over_dt2 * diff.dot(diff)
            if qd.static(self.config.has_damping):
                step = self.solver.elements_v[f + 1, i_v, i_b].pos - self.solver.elements_v[f, i_v, i_b].pos
                energy += 0.5 * mass_over_dt2 * step.dot(step) * self.damping_alpha_dt
            qd.atomic_add(energies[i_b], dt_sq * energy)
        self.solver._func_compute_ele_energy(f)
        qd.loop_config(name="mpm_fem_energy_elements")
        for i_b, i_e in qd.ndrange(self.n_envs, self.n_elements):
            energy = self.solver.elements_el_energy[i_b, i_e].energy * self.solver.elements_i[i_e].V
            qd.atomic_add(energies[i_b], dt_sq * energy)

    # ------------------------------------------- vertex map -------------------------------------------

    @qd.func
    def func_vertex_value(self, f: int, i_v: int):
        i_b = i_v // self.n_vertices
        i_v_local = i_v - i_b * self.n_vertices
        return self.solver.elements_v[f + 1, i_v_local, i_b].pos - self.solver.elements_v[f, i_v_local, i_b].pos

    @qd.func
    def func_vertex_rows(self, i_v: int):
        i_b = i_v // self.n_vertices
        rows = qd.Vector([self.linear_block.row_start[()] + i_v], dt=gs.qd_int)
        weights = qd.Vector([gs.qd_float(self.is_free(i_b, i_v - i_b * self.n_vertices))], dt=gs.qd_float)
        return 1, rows, weights


@qd.kernel
def kernel_fem_begin(fem: qd.template()):
    for i_b in range(fem.n_envs):
        fem.solver.batch_active[i_b] = True
        fem.solver.batch_pcg_active[i_b] = True
        fem.solver.batch_linesearch_active[i_b] = True


@qd.kernel
def kernel_fem_freeze(f: int, fem: qd.template(), control: NewtonControl):
    for i_v, i_b in qd.ndrange(fem.n_vertices, fem.n_envs):
        if control.has_failed_non_finite[i_b] != 0:
            fem.solver.elements_v[f + 1, i_v, i_b].pos = fem.solver.elements_v[f, i_v, i_b].pos
            fem.solver.elements_v[f + 1, i_v, i_b].vel = fem.solver.elements_v[f, i_v, i_b].vel


class FemNewtonHost(NewtonHost):
    """Host stages of FEM around the joint solve.

    Before the solve, they write the inertia and material data of its substep. After it, they derive its velocities
    from the solved positions and run the part of the coupler's step on FEM (see LegacyCoupler.couple_fem)."""

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
