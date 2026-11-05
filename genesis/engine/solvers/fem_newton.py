"""Implicit finite element method (FEM) as a participant of the NewtonSolver.

The FEM unknown is the end-of-substep vertex position x, starting at the substep start. Its share of the joint energy is
FEM's own incremental potential (inertia, elasticity and the linearized damping of its implicit solver) times dt^2, in
the displacement units of the material point method (MPM) energy. The joint solve replaces FEM's own solve and takes its
preconditioned conjugate gradient (PCG) vectors: pcg_state_v.r, z, p, Ap and x, the solution, whose negative is the
Newton direction.

The joint solve replaces FEM's own implicit substep (`FEMSolver.batch_solve`): FEM prepares its inertia and material
data, the joint Newton iterations move its end-of-substep positions together with the other participants, and FEM takes
its velocities from them and runs the rest of its coupling after the solve (see FEMSolver.build_newton_participant).
"""

import quadrants as qd

import genesis as gs

from .newton_solver import NewtonHost, kernel_assemble_participant


# ============================================ Layer 1: the FEM block ============================================


@qd.func(requires_top_level=True)
def func_fem_factor(fem: qd.template()):
    """FEM's preconditioner, the inverse of every vertex's diagonal block, once the contact rows added theirs."""
    qd.loop_config(name="mpm_fem_factor")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].prec = fem.pcg_state_v[i_b, i_v].diag3x3.inverse()


@qd.func(requires_top_level=True)
def func_fem_pcg_init(dt: float, fem: qd.template(), pcg_scalars: qd.template(), is_adjoint: qd.template()):
    """The FEM part of the preconditioned conjugate gradient (PCG) start from 0: r = grad = -dt^2 force, solution and Ap
    cleared.

    With `is_adjoint`, an environment left out of the solve keeps its solution (see func_init_linear_pcg_residual)."""
    qd.loop_config(name="mpm_fem_pcg_init")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].r = -dt * dt * fem.elements_v_energy[i_b, i_v].force
        is_kept = False
        if qd.static(is_adjoint):
            is_kept = pcg_scalars.is_converged[i_b] != 0
        if not is_kept:
            fem.pcg_state_v[i_b, i_v].x = qd.Vector.zero(gs.qd_float, 3)
        fem.pcg_state_v[i_b, i_v].Ap = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_fem_precondition(dt: float, fem: qd.template()):
    """z = M^-1 r by FEM's inverse 3x3 diagonal blocks, scaled to the joint energy.

    A vertex FEM holds at its target gets z = 0, so that its direction stays 0 and the contact rows leave it in
    place."""
    qd.loop_config(name="mpm_fem_precondition")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        z = (fem.pcg_state_v[i_b, i_v].prec @ fem.pcg_state_v[i_b, i_v].r) / (dt * dt)
        if qd.static(fem._enable_vertex_constraints):
            if fem.vertex_constraints.is_constrained[i_v, i_b]:
                z = qd.Vector.zero(gs.qd_float, 3)
        fem.pcg_state_v[i_b, i_v].z = z


@qd.func(requires_top_level=True)
def func_fem_pcg_init_direction(fem: qd.template(), pcg_scalars: qd.template()):
    """Set the first search direction p = z over the FEM vertices, and add r . z to rz."""
    qd.loop_config(name="mpm_fem_pcg_init_direction")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].p = fem.pcg_state_v[i_b, i_v].z
        qd.atomic_add(pcg_scalars.rz[i_b], fem.pcg_state_v[i_b, i_v].r.dot(fem.pcg_state_v[i_b, i_v].z))


@qd.func(requires_top_level=True)
def func_fem_matvec(dt: float, fem: qd.template()):
    """Ap = dt^2 H p over the FEM vertices with FEM's own operator, before the FEM rows add their term."""
    fem.compute_Ap()
    qd.loop_config(name="mpm_fem_matvec_scale")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].Ap = dt * dt * fem.pcg_state_v[i_b, i_v].Ap


@qd.func(requires_top_level=True)
def func_fem_matvec_dot(fem: qd.template(), pcg_scalars: qd.template()):
    """Add p . Ap over the FEM vertices to p . Ap, once Ap holds the whole joint operator."""
    qd.loop_config(name="mpm_fem_matvec_dot")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[i_b], fem.pcg_state_v[i_b, i_v].p.dot(fem.pcg_state_v[i_b, i_v].Ap))


@qd.func(requires_top_level=True)
def func_fem_pcg_update_xr(fem: qd.template(), pcg_scalars: qd.template()):
    """Update the solution and the residual over the FEM vertices, x += alpha p and r -= alpha Ap."""
    qd.loop_config(name="mpm_fem_pcg_update_xr")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if pcg_scalars.is_converged[i_b] == 0:
            alpha = pcg_scalars.alpha[i_b]
            fem.pcg_state_v[i_b, i_v].x = fem.pcg_state_v[i_b, i_v].x + alpha * fem.pcg_state_v[i_b, i_v].p
            fem.pcg_state_v[i_b, i_v].r = fem.pcg_state_v[i_b, i_v].r - alpha * fem.pcg_state_v[i_b, i_v].Ap


@qd.func(requires_top_level=True)
def func_fem_pcg_dot_rz(fem: qd.template(), pcg_scalars: qd.template()):
    """Add r . z over the FEM vertices to rz_new."""
    qd.loop_config(name="mpm_fem_pcg_dot_rz")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.rz_new[i_b], fem.pcg_state_v[i_b, i_v].r.dot(fem.pcg_state_v[i_b, i_v].z))


@qd.func(requires_top_level=True)
def func_fem_pcg_direction(fem: qd.template(), pcg_scalars: qd.template()):
    """Update the search direction p = z + beta p over the FEM vertices, and clear Ap."""
    qd.loop_config(name="mpm_fem_pcg_direction")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].Ap = qd.Vector.zero(gs.qd_float, 3)
        if pcg_scalars.is_converged[i_b] == 0:
            beta = pcg_scalars.beta[i_b]
            fem.pcg_state_v[i_b, i_v].p = fem.pcg_state_v[i_b, i_v].z + beta * fem.pcg_state_v[i_b, i_v].p


@qd.func(requires_top_level=True)
def func_fem_start_line_search(f: int, fem: qd.template(), control: qd.template()):
    """The line-search start point, and the largest component of the step -x into the Newton measure."""
    qd.loop_config(name="mpm_fem_start_line_search")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.linesearch_state_v[i_b, i_v].x_prev = fem.elements_v[f + 1, i_v, i_b].pos
        step = fem.pcg_state_v[i_b, i_v].x
        qd.atomic_max(control.max_disp[i_b], qd.max(qd.abs(step[0]), qd.abs(step[1]), qd.abs(step[2])))


@qd.func(requires_top_level=True)
def func_fem_step_forward(f: int, fem: qd.template(), control: qd.template()):
    qd.loop_config(name="mpm_fem_step_forward")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.elements_v[f + 1, i_v, i_b].pos = (
            fem.linesearch_state_v[i_b, i_v].x_prev - control.alpha[i_b] * fem.pcg_state_v[i_b, i_v].x
        )


@qd.func(requires_top_level=True)
def func_fem_energy(f: int, dt: float, fem: qd.template(), control: qd.template(), is_trial: qd.template()):
    """Add FEM's incremental potential at the end-of-substep positions, times dt^2, to the start or trial energy.

    The potential holds the inertia and the linearized damping of the vertices, and the elastic energy and damping of
    the elements, at the positions of frame f + 1."""
    dt_sq = dt * dt
    damping_alpha_dt = fem._damping_alpha * fem._substep_dt
    qd.loop_config(name="mpm_fem_energy_vertices")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        diff = fem.elements_v[f + 1, i_v, i_b].pos - fem.elements_v_energy[i_b, i_v].inertia
        energy = 0.5 * fem.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
        if qd.static(fem._damping_alpha > 0.0):
            step = fem.elements_v[f + 1, i_v, i_b].pos - fem.elements_v[f, i_v, i_b].pos
            energy += 0.5 * fem.elements_v_info[i_v].mass_over_dt2 * step.dot(step) * damping_alpha_dt
        if is_trial:
            qd.atomic_add(control.energy_trial[i_b], dt_sq * energy)
        else:
            qd.atomic_add(control.energy_start[i_b], dt_sq * energy)
    fem._func_compute_ele_energy(f)
    qd.loop_config(name="mpm_fem_energy_elements")
    for i_b, i_e in qd.ndrange(fem._B, fem.n_elements):
        energy = fem.elements_el_energy[i_b, i_e].energy * fem.elements_i[i_e].V
        if is_trial:
            qd.atomic_add(control.energy_trial[i_b], dt_sq * energy)
        else:
            qd.atomic_add(control.energy_start[i_b], dt_sq * energy)


# ============================================ Layer 2: participant =============================================


@qd.data_oriented
class FemNewtonParticipant:
    """Device stages of the FEM vertex positions in the joint solve."""

    def __init__(self, fem):
        self.fem = fem

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int):
        self.fem._func_compute_ele_hessian_gradient(f)
        self.fem._func_accumulate_vertex_force_preconditioner(f)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        func_fem_factor(self.fem)

    @qd.func(requires_top_level=True)
    def func_pcg_init(self, pcg_scalars: qd.template(), is_adjoint: qd.template()):
        func_fem_pcg_init(self.fem.substep_dt, self.fem, pcg_scalars, is_adjoint)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        func_fem_precondition(self.fem.substep_dt, self.fem)

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, pcg_scalars: qd.template()):
        func_fem_pcg_init_direction(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_matvec(self, pcg_scalars: qd.template()):
        func_fem_matvec(self.fem.substep_dt, self.fem)

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, pcg_scalars: qd.template()):
        func_fem_matvec_dot(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, pcg_scalars: qd.template()):
        func_fem_pcg_update_xr(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, pcg_scalars: qd.template()):
        func_fem_pcg_dot_rz(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, pcg_scalars: qd.template()):
        func_fem_pcg_direction(self.fem, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        func_fem_start_line_search(f, self.fem, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        func_fem_step_forward(f, self.fem, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f: int, is_trial: qd.template()):
        func_fem_energy(f, self.fem.substep_dt, self.fem, control, is_trial)


@qd.kernel
def kernel_fem_begin(fem: qd.template()):
    for i_b in range(fem._B):
        fem.batch_active[i_b] = True
        fem.batch_pcg_active[i_b] = True
        fem.batch_linesearch_active[i_b] = True


@qd.kernel
def kernel_fem_freeze(f: int, fem: qd.template(), control: qd.template()):
    for i_v, i_b in qd.ndrange(fem.n_vertices, fem._B):
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
        kernel_fem_begin(self._fem_solver)

    def newton_assemble(self, f, is_exact=False):
        kernel_assemble_participant(f, self._device)

    def newton_finish(self, f, control):
        self._fem_solver.setup_pos_vel(f)
        kernel_fem_freeze(f, self._fem_solver, control)

    def newton_after(self, f):
        self._fem_solver.sim.coupler.couple_fem(f)

    def newton_end(self, f, control):
        # FEM takes the vertex positions of the substep from the velocities here, in place of its post-coupling phase,
        # so that the mask of the solve still holds
        self._fem_solver.compute_pos(f)
        kernel_fem_freeze(f, self._fem_solver, control)
