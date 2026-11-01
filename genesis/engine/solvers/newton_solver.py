"""Global Newton solver of the implicit solvers that couple strongly.

One Newton solve per substep minimizes the sum of the participants' incremental potentials and of the couplings
between them. The linear system of every Newton iteration is solved by one preconditioned conjugate gradient (PCG) over
the rows of all participants, preconditioned block-diagonally by each participant over its own rows, and the line
search runs on the total energy.

A participant owns its unknowns, its energy, its share of the gradient and of the operator, and a preconditioner over
its own rows. The primary participant also owns the block coordinate (BCOO) matrix and the PCG workspace whose device
scalars (r.z, p.Ap, the threshold, the convergence flag) every participant reduces into. A secondary one keeps its
vectors and adds its partial dot products at fixed points of the iteration. A coupling owns only the knowledge of both
sides and adds its cross terms to their gradients and operators.

The participant and coupling objects enter the kernels as templates, in a fixed number of slots: an absent one is a
`NullParticipant`. Quadrants registers the arrays of a template only through @qd.data_oriented attributes and frozen
dataclasses, so the slots are kernel arguments rather than a list.
"""

import functools
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.newton_control import (
    NewtonControl,
    func_accept_or_backtrack,
    func_check_convergence,
    func_init_line_search,
    func_update_convergence,
)
from genesis.grad.bridge import custom_backward
from genesis.utils.misc import qd_to_numpy

from .base_solver import Solver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# Run the Newton iterations as one graph (`kernel_newton_loop`), or drive them from the host with only the
# preconditioned conjugate gradient iterations as a graph. The graph runs faster, the host loop compiles faster.
# FIXME: The compile time of a graph grows faster than its size, so the Newton graph compiles much slower than its
# stages as separate kernels, and a change of any stage in it recompiles it whole.
USE_NEWTON_GRAPH = False

# Slots of the composed kernels
N_SECONDARY_SLOTS = 3
N_COUPLING_SLOTS = 3

# Relative reduction of r . z at which the conjugate gradient of the adjoint solve stops, tighter than the forward
# solve's, whose error the adjoint carries into every gradient
ADJOINT_PCG_TOL_RATE = {64: 1e-24, 32: 1e-12}


@qd.data_oriented
class NullParticipant:
    """Stands in for an absent secondary participant or coupling of the composed kernels: every stage does nothing."""

    @qd.func(requires_top_level=True)
    def func_evaluate(self, f):
        pass

    @qd.func(requires_top_level=True)
    def func_gradient(self, f):
        pass

    @qd.func(requires_top_level=True)
    def func_factor(self):
        pass

    @qd.func(requires_top_level=True)
    def func_pcg_init(self):
        pass

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        pass

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, ws: qd.template()):
        pass

    @qd.func(requires_top_level=True)
    def func_matvec(self):
        pass

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, ws: qd.template()):
        pass

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, ws: qd.template()):
        pass

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, ws: qd.template()):
        pass

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, ws: qd.template()):
        pass

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f):
        pass

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f):
        pass

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f, is_trial):
        pass

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f):
        pass

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f):
        pass

    @qd.func(requires_top_level=True)
    def func_contact_matvec_post(self, f):
        pass

    @qd.func(requires_top_level=True)
    def func_after_step(self, f):
        pass


# ============================================== composed stages ==============================================


@qd.func(requires_top_level=True)
def func_assemble_secondary(
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    c2: qd.template(),
    f,
):
    """Assemble the gradients and factors of the secondary participants and couplings at the current iterate.

    Runs after the assembly of the primary, which stages what the couplings read of it."""
    s0.func_evaluate(f)
    s0.func_gradient(f)
    s1.func_evaluate(f)
    s2.func_evaluate(f)
    s1.func_gradient(f)
    s2.func_gradient(f)
    c0.func_contact_gradient(f)
    c1.func_contact_gradient(f)
    c2.func_contact_gradient(f)
    s0.func_factor()
    s1.func_factor()
    s2.func_factor()


@qd.func(requires_top_level=True)
def func_pcg_start(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
):
    """Build the preconditioner of every participant and start the preconditioned conjugate gradient from 0.

    The start sets r = -grad, z = M^-1 r, p = z and r . z."""
    p.func_pcg_prepare()
    p.func_pcg_init_residual()
    s0.func_pcg_init()
    s1.func_pcg_init()
    s2.func_pcg_init()
    p.func_precondition()
    s0.func_precondition()
    s1.func_precondition()
    s2.func_precondition()
    p.func_pcg_init_direction()
    s0.func_pcg_init_direction(p.ws)
    s1.func_pcg_init_direction(p.ws)
    s2.func_pcg_init_direction(p.ws)


@qd.func(requires_top_level=True)
def func_pcg_iteration(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    c2: qd.template(),
    f,
):
    """Run one preconditioned conjugate gradient iteration over the rows of every participant.

    The operator includes the cross terms of the couplings."""
    p.func_pcg_spmv()
    c0.func_contact_matvec(f)
    c1.func_contact_matvec(f)
    c2.func_contact_matvec(f)
    s0.func_matvec()
    s1.func_matvec()
    s2.func_matvec()
    c0.func_contact_matvec_post(f)
    c1.func_contact_matvec_post(f)
    c2.func_contact_matvec_post(f)
    s0.func_matvec_dot(p.ws)
    s1.func_matvec_dot(p.ws)
    s2.func_matvec_dot(p.ws)
    s0.func_pcg_update_xr(p.ws)
    s1.func_pcg_update_xr(p.ws)
    s2.func_pcg_update_xr(p.ws)
    p.func_pcg_update_xr()
    p.func_precondition()
    s0.func_precondition()
    s1.func_precondition()
    s2.func_precondition()
    s0.func_pcg_dot_rz(p.ws)
    s1.func_pcg_dot_rz(p.ws)
    s2.func_pcg_dot_rz(p.ws)
    p.func_pcg_direction_update()
    s0.func_pcg_direction(p.ws)
    s1.func_pcg_direction(p.ws)
    s2.func_pcg_direction(p.ws)


@qd.func(requires_top_level=True)
def func_start_line_search(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    f,
):
    """Set the Newton direction of every participant from the conjugate gradient solution, then start the line search.

    The participants report the measure of the direction, on which the convergence is checked first."""
    s0.func_start_line_search(control, f)
    s1.func_start_line_search(control, f)
    s2.func_start_line_search(control, f)
    p.func_start_line_search(control, f)
    func_check_convergence(control)
    func_init_line_search(control)


@qd.func(requires_top_level=True)
def func_step_forward(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    c2: qd.template(),
    f,
):
    """Move every participant to the trial point of the line-search step and update what the couplings derive of it."""
    p.func_step_forward(control)
    s0.func_step_forward(control, f)
    s1.func_step_forward(control, f)
    s2.func_step_forward(control, f)
    c0.func_after_step(f)
    c1.func_after_step(f)
    c2.func_after_step(f)


@qd.func(requires_top_level=True)
def func_energy(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    f,
    is_trial,
):
    """Evaluate the total energy at the start or trial point.

    The primary writes the energy and the secondary participants add theirs."""
    p.func_energy(control, f, is_trial)
    s0.func_energy(control, f, is_trial)
    s1.func_energy(control, f, is_trial)
    s2.func_energy(control, f, is_trial)


# ================================================== kernels ==================================================


@qd.kernel
def kernel_assemble_secondary(
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    c2: qd.template(),
    f: qd.i32,
):
    func_assemble_secondary(s0, s1, s2, c0, c1, c2, f)


@qd.kernel
def kernel_pcg_start(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
):
    func_pcg_start(p, s0, s1, s2)


@qd.kernel(graph=True)
def kernel_pcg_loop(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    c2: qd.template(),
    f: qd.i32,
):
    """Run the conjugate gradient iterations after kernel_pcg_start as one graph, until convergence or the budget."""
    while qd.graph.do_while(p.ws.cond):
        func_pcg_iteration(p, s0, s1, s2, c0, c1, c2, f)


@qd.kernel
def kernel_pcg_iteration(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    c2: qd.template(),
    f: qd.i32,
):
    func_pcg_iteration(p, s0, s1, s2, c0, c1, c2, f)


@qd.kernel
def kernel_start_line_search(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    f: qd.i32,
):
    func_start_line_search(control, p, s0, s1, s2, f)


@qd.kernel
def kernel_step_forward(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    c2: qd.template(),
    f: qd.i32,
):
    func_step_forward(control, p, s0, s1, s2, c0, c1, c2, f)


@qd.kernel
def kernel_energy(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    f: qd.i32,
    is_trial: qd.i32,
):
    func_energy(control, p, s0, s1, s2, f, is_trial)


@qd.kernel
def kernel_accept_or_backtrack(control: qd.template()):
    func_accept_or_backtrack(control)


@qd.kernel
def kernel_update_convergence(control: qd.template()):
    func_update_convergence(control)


@qd.kernel(graph=True)
def kernel_newton_loop(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    s2: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    c2: qd.template(),
    f: qd.i32,
):
    """Run the Newton iterations of a substep as one graph, after the evaluation of the start energy.

    The assembly of the primary must need no storage growth inside the loop. The MPM triplet demand depends on the grid
    topology alone, which is fixed over the substep, so the storage sized before the loop holds every assembly in it.
    """
    while qd.graph.do_while(control.newton_cond):
        p.func_assemble(control, f)
        func_assemble_secondary(s0, s1, s2, c0, c1, c2, f)
        func_pcg_start(p, s0, s1, s2)
        while qd.graph.do_while(p.ws.cond):
            func_pcg_iteration(p, s0, s1, s2, c0, c1, c2, f)
        func_start_line_search(control, p, s0, s1, s2, f)
        while qd.graph.do_while(control.ls_cond):
            func_step_forward(control, p, s0, s1, s2, c0, c1, c2, f)
            func_energy(control, p, s0, s1, s2, f, 1)
            func_accept_or_backtrack(control)
        func_update_convergence(control)


# =================================================== solver ===================================================


class NewtonSolver(Solver):
    """Solver running one Newton solve per substep over the participants registered into it (see the module docstring).

    The simulator holds it apart from its solvers, since it owns no physical state, and runs its substep at the start of
    the post-coupling phase, after every solver's pre-coupling phase and the coupler, and before every solver's own
    post-coupling phase.

    Each participant and coupling registers a host object, whose methods run the host-side stages around the solve
    (`newton_prepare`, `newton_finish`, `newton_after`, `newton_end`, and `newton_assemble` for the primary), and a
    device object for the composed kernels.
    """

    def __init__(self, scene: "Scene", sim: "Simulator"):
        super().__init__(scene, sim, None)
        self.control = NewtonControl(gs.qd_float)
        self._null = NullParticipant()
        self._primary = None
        self._secondaries = []
        self._couplings = []

    def register_primary(self, host, device):
        self._primary = (host, device)

    def register_secondary(self, host, device):
        if len(self._secondaries) == N_SECONDARY_SLOTS:
            gs.raise_exception(f"The Newton solver holds {N_SECONDARY_SLOTS} secondary participants at most.")
        self._secondaries.append((host, device))

    def register_coupling(self, host, device):
        if len(self._couplings) == N_COUPLING_SLOTS:
            gs.raise_exception(f"The Newton solver holds {N_COUPLING_SLOTS} couplings at most.")
        self._couplings.append((host, device))

    @property
    def is_active(self):
        return self._primary is not None

    def _slots(self):
        secondary = [device for _, device in self._secondaries]
        secondary += [self._null] * (N_SECONDARY_SLOTS - len(secondary))
        coupling = [device for _, device in self._couplings]
        coupling += [self._null] * (N_COUPLING_SLOTS - len(coupling))
        return self._primary[1], secondary, coupling

    def step(self, f):
        """Run the Newton solve of a substep: the preparation of the participants, the start energy, the Newton
        iterations and the hand-back of the solution.

        The launches of the solve of substep `f` form one entry of the step recorded for a backward pass, which
        `backward` reverses: the solve is differentiated by the implicit function theorem rather than launch by launch.
        """
        if not self.is_active or self._primary[0].runs_own_substep():
            return
        with custom_backward(self.backward, f):
            self._step(f)
        # What follows the solve is reversed by its own autodiff, such as the integration of Rigid's deferred substep
        for host in [self._primary[0]] + [host for host, _ in self._secondaries + self._couplings]:
            host.newton_end(f)

    def backward(self, f):
        """Reverse the Newton solve of a substep, from the state left by the replay of its forward pass.

        The primary reverses the solve of substep `f`. Each secondary participant wraps it in its own reverse of the
        part of its substep that the joint solve takes over, sharing the joint adjoint solve (see 'stage_adjoint')."""
        hosts = [host for host, _ in self._secondaries + self._couplings]
        if not all(host.is_differentiable for host in hosts):
            gs.raise_exception(
                "The backward pass of the joint Newton solve supports the participants of Rigid alone besides MPM."
            )
        solve = functools.partial(self._primary[0].newton_backward, f)
        for host, _ in self._secondaries:
            solve = functools.partial(host.newton_backward, f, solve)
        solve()

    def stage_adjoint(self, f):
        """Assemble the rows of the secondary participants and couplings and their right-hand side of the adjoint solve.

        The rows are taken at the solution of substep `f`, after the primary assembled its Hessian there and staged its
        own right-hand side."""
        if not (self._secondaries or self._couplings):
            return
        primary, secondary, coupling = self._slots()
        kernel_assemble_secondary(*secondary, *coupling, f)
        for host, _ in self._secondaries + self._couplings:
            host.newton_adjoint_rhs(f)

    def finish_adjoint(self, f):
        """Hand the solution of the adjoint solve of a substep to the secondary participants and couplings."""
        for host, _ in self._secondaries + self._couplings:
            host.newton_adjoint_finish(f)

    @property
    def has_secondaries(self):
        return bool(self._secondaries or self._couplings)

    def solve_adjoint(self, f, check_curvature=False) -> bool:
        """Solve the adjoint system H lambda = b on the rows of the primary, at the tighter tolerance of the adjoint.

        The preconditioned conjugate gradient of the forward solve of substep `f` runs from the right-hand side and the
        matrix staged by the primary. With `check_curvature`, meant for a Hessian without positive definite projection,
        returns whether the solve met no direction of non-positive curvature. Otherwise returns True."""
        primary, secondary, coupling = self._slots()
        ws = primary.ws
        tol_rate = qd_to_numpy(ws.tol_rate, copy=True)
        bits = 64 if gs.qd_float == qd.f64 else 32
        ws.tol_rate.from_numpy(np.array(ADJOINT_PCG_TOL_RATE[bits], dtype=gs.np_float))
        ws.check_curvature.from_numpy(np.array(int(check_curvature), dtype=np.int32))
        self._pcg(f)
        ws.tol_rate.from_numpy(tol_rate)
        if not check_curvature:
            return True
        ws.check_curvature.from_numpy(np.array(0, dtype=np.int32))
        return not int(qd_to_numpy(ws.breakdown))

    def _step(self, f):
        control = self.control
        primary, secondary, coupling = self._slots()
        hosts = [self._primary[0]] + [host for host, _ in self._secondaries]
        coupling_hosts = [host for host, _ in self._couplings]

        for host in hosts + coupling_hosts:
            host.newton_prepare(f)
        kernel_energy(control, primary, *secondary, f, is_trial=0)

        if USE_NEWTON_GRAPH and not self._has_grad_participants():
            kernel_newton_loop(control, primary, *secondary, *coupling, f)
        else:
            self._newton_loop_launches(f)

        for host in hosts + coupling_hosts:
            host.newton_finish(f)
        for host in hosts + coupling_hosts:
            host.newton_after(f)

    def _has_grad_participants(self):
        """Whether a participant other than the primary takes part in a scene tracking gradients, its ndarrays then
        owning a gradient."""
        return self._requires_grad and bool(self._secondaries or self._couplings)

    def _pcg(self, f):
        """Run the conjugate gradient iterations of a substep from its staged right-hand side, as one graph."""
        primary, secondary, coupling = self._slots()
        kernel_pcg_start(primary, *secondary)
        # FIXME: quadrants#946 - a graph refuses the ndarrays that own a gradient, so the iterations launch one by one,
        # with a device read each, when other participants take part in a scene tracking gradients
        if not self._has_grad_participants():
            kernel_pcg_loop(primary, *secondary, *coupling, f)
            return
        while True:
            kernel_pcg_iteration(primary, *secondary, *coupling, f)
            if not int(qd_to_numpy(primary.ws.cond)):
                break

    def _newton_loop_launches(self, f):
        """Drive the Newton iterations from the host, the conjugate gradient iterations of each running as one graph."""
        control = self.control
        primary, secondary, coupling = self._slots()
        has_secondary = bool(self._secondaries or self._couplings)
        while True:
            self._primary[0].newton_assemble(f)
            if has_secondary:
                kernel_assemble_secondary(*secondary, *coupling, f)
            self._pcg(f)
            kernel_start_line_search(control, primary, *secondary, f)
            while True:
                kernel_step_forward(control, primary, *secondary, *coupling, f)
                kernel_energy(control, primary, *secondary, f, 1)
                kernel_accept_or_backtrack(control)
                if not int(qd_to_numpy(control.ls_cond)):
                    break
            kernel_update_convergence(control)
            if not int(qd_to_numpy(control.newton_cond)):
                break
