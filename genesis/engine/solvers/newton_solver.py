"""Global Newton solver of the implicit solvers that couple strongly.

One Newton solve per substep minimizes the sum of the participants' incremental potentials and of the couplings
between them. The linear system of every Newton iteration is solved by one preconditioned conjugate gradient (PCG) over
the rows of all participants, preconditioned block-diagonally by each participant over its own rows, and the line
search runs on the total energy.

Every participant runs the same stages. It owns its unknowns, its energy, its share of the gradient and of the operator,
the PCG vectors over its own rows and a preconditioner over them, and adds its share of r . z and p . Ap into the
scalars of the solve, which the Newton solver owns (see PCGScalars) and from which it takes the steps and the
convergence decision between the stages of the participants. A coupling owns only the knowledge of both sides and adds
its cross terms to their gradients and operators.

The participants enter the kernels as one template argument, `NewtonParticipants`, which holds the participants and the
couplings in @qd.data_oriented NamedTuples that the composed stages iterate with qd.static.
"""

import collections
import functools
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.linear_pcg import (
    PCGScalars,
    func_pcg_arm,
    func_pcg_decide,
    func_pcg_reset,
    func_pcg_select_envs,
    func_pcg_step,
)
from genesis.engine.numerics.newton_control import (
    NEWTON_FAILED_LINE_SEARCH,
    NewtonControl,
    func_accept_or_backtrack,
    func_check_convergence,
    func_clear_non_finite,
    func_init_line_search,
    func_reset_energy,
    func_reset_newton_control,
    func_reset_newton_measures,
    func_update_convergence,
)
from genesis.grad.bridge import custom_backward
from genesis.utils.data import DataKind, iter_data_oriented
from genesis.utils.misc import qd_to_numpy

from .base_solver import Solver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator
    from genesis.options.solvers import NewtonOptions

# Default of NewtonOptions.adjoint_pcg_tol by the bits of the floating-point precision
DEFAULT_ADJOINT_PCG_TOL = {64: 1e-24, 32: 1e-12}


@qd.data_oriented
class NewtonParticipants:
    """Participants of a Newton solve and the couplings between them, gathered into one kernel argument.

    `members` and `couplings` are @qd.data_oriented NamedTuples (see `participant_group`) of their device objects. They
    stay members of this object: a tuple passed as the kernel argument itself cannot be weakly referenced, so the kernel
    would re-derive its specialization on every launch.
    """

    def __init__(self, members, couplings):
        self.members = members
        self.couplings = couplings


def participant_group(name: str, devices: list):
    """Gather the device objects of several participants into a @qd.data_oriented NamedTuple, which qd.static iterates.

    Quadrants declares the arrays a kernel reaches through a template argument by walking its @qd.data_oriented members.
    """
    fields = [f"member_{i}" for i in range(len(devices))]
    # FIXME: quadrants#1040 - the walk of the arguments reads the members of a data-oriented object through vars(),
    # which a NamedTuple has none of. A subclass without __slots__ carries a __dict__, which holds the members under
    # their field names as well, so that the walk declares their arrays.
    group_type = qd.data_oriented(type(name, (collections.namedtuple(name, fields),), {}))
    group = group_type(*devices)
    group.__dict__.update(zip(fields, devices))
    return group


class NewtonHost:
    """Host stages of a participant or a coupling around the Newton solve of a substep, which do nothing by default.

    A participant also assembles its share of the Newton system from the host (`newton_assemble`), reverses the part of
    its substep that the solve takes over around the joint adjoint solve (`newton_backward`), and flags a solve stopped
    on a non-finite measure for the environments its solver reports errors on (`newton_flag_non_finite`)."""

    is_differentiable = False

    def newton_prepare(self, f):
        pass

    def newton_flag_non_finite(self, control):
        pass

    def newton_finish(self, f):
        pass

    def newton_after(self, f):
        pass

    def newton_end(self, f):
        pass

    def newton_adjoint_rhs(self, f):
        pass

    def newton_adjoint_finish(self, f):
        pass


# ============================================== composed stages ==============================================


@qd.func(requires_top_level=True)
def func_assemble(f: int, participants: qd.template()):
    """Assemble the Newton system of every participant at the current iterate, then the share of the couplings."""
    for member in qd.static(participants.members):
        member.func_assemble(f)
    func_contact_gradient(f, participants)


@qd.func(requires_top_level=True)
def func_contact_gradient(f: int, participants: qd.template()):
    """Add the share of the couplings to the gradients and preconditioners of both sides, once they all assembled."""
    for coupling in qd.static(participants.couplings):
        coupling.func_contact_gradient(f)


@qd.func(requires_top_level=True)
def func_pcg_start(
    newton_control: qd.template(), pcg_scalars: qd.template(), participants: qd.template(), is_adjoint: qd.template()
):
    """Build the preconditioner of every participant and start the preconditioned conjugate gradient from 0.

    The start sets r = -grad, z = M^-1 r and p = z over the rows of every participant, then the convergence threshold
    from the r . z they all added. The environments whose Newton solve stopped take no part, except in an adjoint
    solve."""
    func_pcg_select_envs(pcg_scalars, newton_control, is_adjoint)
    func_pcg_reset(pcg_scalars)
    for member in qd.static(participants.members):
        member.func_factor()
        member.func_pcg_init(pcg_scalars, is_adjoint)
        member.func_precondition()
        member.func_pcg_init_direction(pcg_scalars)
    func_pcg_arm(pcg_scalars)


@qd.func(requires_top_level=True)
def func_pcg_iteration(f: int, pcg_scalars: qd.template(), participants: qd.template()):
    """Run one preconditioned conjugate gradient iteration over the rows of every participant.

    The couplings add their cross terms once every participant set the product of its own operator. The step alpha and
    the direction factor beta, with the convergence decision, are taken once every participant added its p . Ap and its
    r . z (see PCGScalars)."""
    for member in qd.static(participants.members):
        member.func_matvec(pcg_scalars)
    for coupling in qd.static(participants.couplings):
        coupling.func_contact_matvec(f, pcg_scalars)
    for member in qd.static(participants.members):
        member.func_matvec_dot(pcg_scalars)
    func_pcg_step(pcg_scalars)
    for member in qd.static(participants.members):
        member.func_pcg_update_xr(pcg_scalars)
        member.func_precondition()
        member.func_pcg_dot_rz(pcg_scalars)
    func_pcg_decide(pcg_scalars)
    for member in qd.static(participants.members):
        member.func_pcg_direction(pcg_scalars)


@qd.func(requires_top_level=True)
def func_start_line_search(f: int, newton_control: qd.template(), participants: qd.template()):
    """Start the line search and set the Newton direction of every participant from the conjugate gradient solution.

    The participants report the measure of the direction, on which the convergence is checked, and may lower the start
    step of the line search below its full value."""
    func_reset_newton_measures(newton_control)
    func_init_line_search(newton_control)
    for member in qd.static(participants.members):
        member.func_start_line_search(newton_control, f)
    func_check_convergence(newton_control)


@qd.func(requires_top_level=True)
def func_step_forward(f: int, newton_control: qd.template(), participants: qd.template()):
    """Move every participant to the trial point of the line-search step and update what the couplings derive of it."""
    for member in qd.static(participants.members):
        member.func_step_forward(newton_control, f)
    for coupling in qd.static(participants.couplings):
        coupling.func_after_step(f)


@qd.func(requires_top_level=True)
def func_energy(f: int, newton_control: qd.template(), participants: qd.template(), is_trial: qd.template()):
    """Evaluate the total energy at the start or trial point, which every participant adds its own to."""
    func_reset_energy(newton_control, is_trial)
    for member in qd.static(participants.members):
        member.func_energy(newton_control, f, is_trial)


# ================================================== kernels ==================================================


@qd.kernel
def kernel_assemble_participant(f: int, participant: qd.template()):
    """Assemble the Newton system of one participant, for a participant needing no storage growth from the host."""
    participant.func_assemble(f)


@qd.kernel
def kernel_contact_gradient(f: int, participants: qd.template()):
    func_contact_gradient(f, participants)


@qd.kernel
def kernel_pcg_start(
    newton_control: qd.template(), pcg_scalars: qd.template(), participants: qd.template(), is_adjoint: qd.template()
):
    func_pcg_start(newton_control, pcg_scalars, participants, is_adjoint)


@qd.kernel(graph=True)
def kernel_pcg_loop(f: int, pcg_scalars: qd.template(), participants: qd.template()):
    """Run the conjugate gradient iterations after kernel_pcg_start as one graph, until convergence or the budget."""
    while qd.graph.do_while(pcg_scalars.is_running):
        func_pcg_iteration(f, pcg_scalars, participants)


@qd.kernel
def kernel_pcg_iteration(f: int, pcg_scalars: qd.template(), participants: qd.template()):
    func_pcg_iteration(f, pcg_scalars, participants)


@qd.kernel
def kernel_start_line_search(f: int, newton_control: qd.template(), participants: qd.template()):
    func_start_line_search(f, newton_control, participants)


@qd.kernel
def kernel_energy(f: int, newton_control: qd.template(), participants: qd.template(), is_trial: qd.template()):
    func_energy(f, newton_control, participants, is_trial)


@qd.kernel
def kernel_line_search_trial(f: int, newton_control: qd.template(), participants: qd.template()):
    """Step every participant to the trial point, evaluate its energy, and accept it or halve the step."""
    func_step_forward(f, newton_control, participants)
    func_energy(f, newton_control, participants, is_trial=True)
    func_accept_or_backtrack(newton_control)


@qd.kernel
def kernel_update_convergence(newton_control: qd.template()):
    func_update_convergence(newton_control)


@qd.kernel
def kernel_reset_newton_control(newton_control: qd.template()):
    func_reset_newton_control(newton_control)


@qd.kernel
def kernel_clear_non_finite(newton_control: qd.template()):
    func_clear_non_finite(newton_control)


@qd.kernel(graph=True)
def kernel_newton_loop(f: int, newton_control: qd.template(), pcg_scalars: qd.template(), participants: qd.template()):
    """Run the Newton iterations of a substep as one graph, after the evaluation of the start energy.

    The assembly of every participant must need no storage growth inside the loop. The MPM triplet demand depends on the
    grid topology alone, which is fixed over the substep, so the storage sized before the loop holds every assembly in
    it.
    """
    while qd.graph.do_while(newton_control.is_newton_running):
        func_assemble(f, participants)
        func_pcg_start(newton_control, pcg_scalars, participants, False)
        while qd.graph.do_while(pcg_scalars.is_running):
            func_pcg_iteration(f, pcg_scalars, participants)
        func_start_line_search(f, newton_control, participants)
        while qd.graph.do_while(newton_control.is_line_search_running):
            func_step_forward(f, newton_control, participants)
            func_energy(f, newton_control, participants, is_trial=True)
            func_accept_or_backtrack(newton_control)
        func_update_convergence(newton_control)


# =================================================== solver ===================================================


class NewtonSolver(Solver):
    """Solver running one Newton solve per substep over the participants registered into it (see the module docstring).

    The simulator holds it apart from its solvers, since it owns no physical state, and runs its substep at the start of
    the post-coupling phase, after every solver's pre-coupling phase and the coupler, and before every solver's own
    post-coupling phase.

    The coupler registers every participant and coupling of the scene (see LegacyCoupler), each as a host object (see
    NewtonHost), whose methods run the host-side stages around the solve, and a device object for the composed kernels.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: "NewtonOptions"):
        super().__init__(scene, sim, options)
        # The scalars of the solve, one per environment, allocated once the participants register (see `register`)
        self.control = None
        self.pcg_scalars = None
        bits = 64 if gs.qd_float == qd.f64 else 32
        self._adjoint_pcg_tol = options.adjoint_pcg_tol or DEFAULT_ADJOINT_PCG_TOL[bits]
        self._participant_hosts = []
        self._coupling_hosts = []
        self._participants = None
        self._is_loop_graph = options.graph == "newton"
        self._has_coupling_devices = False
        self._has_warned_adjoint_projection = False

    def register(self, participants, couplings):
        """Register the participants of the Newton solve and the couplings between them, each as a pair of its host
        object and its device object, which activates the solver.

        A coupling acting on one participant alone, whose cross terms vanish, gives None as its device object and runs
        its host stages only."""
        options = self._options
        self.control = NewtonControl(
            gs.qd_float,
            options.max_iterations,
            options.line_search_max_iterations,
            options.line_search_shrink,
            options.strict,
            self._sim._B,
        )
        self.pcg_scalars = PCGScalars(gs.qd_float, options.pcg_max_iterations, options.pcg_tol, self._sim._B)
        coupling_devices = [device for _, device in couplings if device is not None]
        # FIXME: quadrants#946 - a graph refuses the ndarrays that own a gradient, which those of the participants
        # besides MPM do in a scene tracking gradients
        if self._is_loop_graph and self._requires_grad and (len(participants) > 1 or coupling_devices):
            gs.raise_exception(
                "NewtonOptions(graph='newton') does not support a differentiable scene coupling MPM with another "
                "solver."
            )
        self.control.disp_tol.from_numpy(
            np.array(self._options.velocity_tol * self._sim._substep_dt, dtype=gs.np_float)
        )
        self._participant_hosts = [host for host, _ in participants]
        self._coupling_hosts = [host for host, _ in couplings]
        self._participants = NewtonParticipants(
            participant_group("NewtonMembers", [device for _, device in participants]),
            participant_group("NewtonCouplings", coupling_devices),
        )
        self._has_coupling_devices = bool(coupling_devices)

    @property
    def is_active(self):
        return self._participants is not None

    @property
    def data(self):
        """Yield the scalars of the solve and the arrays of its participants and couplings, which every substep rebuilds
        before reading them (see 'Solver.data').

        The solver owns no physical state, so no checkpoint reads them: the simulator frees them with its solvers (see
        'Simulator.destroy')."""
        if self._participants is not None:
            yield from iter_data_oriented("control", self.control, DataKind.SCRATCH)
            yield from iter_data_oriented("pcg_scalars", self.pcg_scalars, DataKind.SCRATCH)
            yield from iter_data_oriented("participants", self._participants, DataKind.SCRATCH)

    @property
    def _hosts(self):
        return [*self._participant_hosts, *self._coupling_hosts]

    @property
    def _is_joint(self):
        """Whether the device stages of the solve span several participants."""
        return len(self._participant_hosts) > 1 or self._has_coupling_devices

    def step(self, f):
        """Run the Newton solve of a substep: the preparation of the participants, the start energy, the Newton
        iterations and the hand-back of the solution.

        The launches of the solve of substep `f` form one entry of the step recorded for a backward pass, which
        `backward` reverses: the solve is differentiated by the implicit function theorem rather than launch by launch.

        An environment whose solve stopped on a non-finite measure keeps its state for the rest of the step, each
        participant leaving it as it was, and the step raises for it at its end (see `_check_failures`), while the other
        environments step on.
        """
        if not self.is_active:
            return
        with custom_backward(self.backward, f):
            self._step(f)
        if self._options.strict:
            self._check_line_search(f)
        # What follows the solve is reversed by its own autodiff, such as the integration of Rigid's deferred substep
        for host in self._hosts:
            host.newton_end(f)
        self._check_failures()

    def _check_line_search(self, f):
        """Raise at once on a solve whose line search found no lower energy, in strict mode."""
        control = self.control
        failed = qd_to_numpy(control.failed)
        for i_b in np.flatnonzero(failed == NEWTON_FAILED_LINE_SEARCH):
            gs.raise_exception(
                f"Strict Newton solve: the line search of Newton iteration {int(qd_to_numpy(control.newton_iter)[i_b])} "
                f"of substep {self._sim.cur_substep_global} in environment {i_b} found no energy below "
                f"{float(qd_to_numpy(control.energy_start)[i_b])} within {self._options.line_search_max_iterations} "
                f"trials, the last one giving {float(qd_to_numpy(control.energy_trial)[i_b])}. The energy disagrees "
                "with its gradient or with the curvature of its Newton system."
            )

    def _check_failures(self):
        """Report the solves that failed since the last check.

        The environments whose solve stopped on a non-finite measure are handed to every participant once a step is
        over, which flags them for the environments it reports errors on, an error being raised at the beginning of the
        next step. A solve that reached `max_iterations` is warned of every `failure_check_interval` substeps, each
        check waiting for the device."""
        control = self.control
        sim = self._sim
        if (sim.cur_substep_global + 1) % sim._substeps == 0:
            for host in self._participant_hosts:
                host.newton_flag_non_finite(control)
            kernel_clear_non_finite(control)
        if sim.cur_substep_global % self._options.failure_check_interval == 0:
            if int(qd_to_numpy(control.has_failed_iterations)):
                control.has_failed_iterations.from_numpy(np.array(0, dtype=gs.np_int))
                gs.logger.warning(
                    "The implicit Newton solve stopped before converging: raise NewtonOptions(max_iterations) or "
                    "loosen NewtonOptions(velocity_tol)."
                )

    def backward(self, f):
        """Reverse the Newton solve of a substep, from the state left by the replay of its forward pass.

        Each participant wraps the joint adjoint solve in its own reverse of the part of its substep that the solve
        takes over: the reverse of what follows the solve into the right-hand side before, and the pullback of the
        adjoint solution after."""
        if not all(host.is_differentiable for host in self._hosts):
            gs.raise_exception(
                "The backward pass of the joint Newton solve supports the participants of Rigid alone besides MPM."
            )
        solve = functools.partial(self._solve_adjoint, f)
        for host in self._participant_hosts:
            solve = functools.partial(host.newton_backward, f, solve)
        solve()

    def _solve_adjoint(self, f):
        """Solve the adjoint system of substep `f` on the Hessian at its solution by the conjugate gradient of the
        forward solve, and hand the solution to every participant and coupling.

        The Hessian is exact or projected to positive definite, as `adjoint_hessian` selects. With 'auto', an environment
        whose solve met a direction of non-positive curvature on the exact Hessian runs again on the projected one, the
        other environments keeping their solution of the exact one."""
        mode = self._options.adjoint_hessian
        is_exact = mode in ("exact", "auto")
        self._stage_adjoint(f, is_exact)
        envs_breakdown = self._pcg_adjoint(f, check_curvature=is_exact)
        # At a converged solution the Hessian of the incremental potential is positive definite
        if is_exact and envs_breakdown.any():
            if mode == "exact":
                gs.raise_exception(
                    "The exact Hessian of an implicit MPM substep is not positive definite at its solution in "
                    f"environments {np.flatnonzero(envs_breakdown).tolist()}, which the forward Newton solve most "
                    "likely stopped short of: tighten NewtonOptions(velocity_tol), or use "
                    "NewtonOptions(adjoint_hessian='auto') or 'projected'."
                )
            if not self._has_warned_adjoint_projection:
                gs.logger.warning(
                    "The exact Hessian of an implicit MPM substep is not positive definite at its solution, which the "
                    "forward Newton solve most likely stopped short of: its adjoint solves on the projected Hessian, "
                    "whose gradients are approximate. Tighten NewtonOptions(velocity_tol) for exact ones."
                )
        pcg_scalars = self.pcg_scalars
        pcg_scalars.is_env_adjoint.fill(1)
                self._has_warned_adjoint_projection = True
            self._stage_adjoint(f, is_exact=False)
            self._pcg_adjoint(f)
        for host in self._hosts:
            host.newton_adjoint_finish(f)

    def _stage_adjoint(self, f, is_exact):
        """Assemble the Newton system of every participant at the solution of substep `f`, the Hessian exact with
        `is_exact`, then stage the right-hand side of the adjoint solve in place of the gradient."""
        self._assemble(f, is_exact)
        for host in self._hosts:
            host.newton_adjoint_rhs(f)

    def _pcg_adjoint(self, f, check_curvature=False):
        """Solve the staged adjoint system H lambda = b by the conjugate gradient of the forward solve, at the tighter
        tolerance of the adjoint, over the environments of `is_env_adjoint` (see PCGScalars).
            pcg_scalars.is_env_adjoint.from_numpy(envs_breakdown)

        With `check_curvature`, meant for a Hessian without positive definite projection, returns for every environment
        whether its solve met a direction of non-positive curvature."""
        pcg_scalars = self.pcg_scalars
        pcg_scalars.tol_rate.from_numpy(np.array(self._adjoint_pcg_tol, dtype=gs.np_float))
        pcg_scalars.is_curvature_checked.from_numpy(np.array(int(check_curvature), dtype=gs.np_int))
        self._pcg(f, is_adjoint=True)
        pcg_scalars.tol_rate.from_numpy(np.array(self._options.pcg_tol, dtype=gs.np_float))
        if not check_curvature:
            return None
        pcg_scalars.is_curvature_checked.from_numpy(np.array(0, dtype=gs.np_int))
        return qd_to_numpy(pcg_scalars.has_breakdown)

    def _step(self, f):
        control = self.control
        participants = self._participants

        for host in self._hosts:
            host.newton_prepare(f)
        kernel_reset_newton_control(control)
        kernel_energy(f, control, participants, is_trial=False)

        # FIXME: The compile time of a graph grows faster than its size, so the Newton graph compiles much slower than
        # its stages as separate kernels, and a change of any stage in it recompiles it whole
        if self._is_loop_graph:
            kernel_newton_loop(f, control, self.pcg_scalars, participants)
        else:
            self._newton_loop_launches(f)

        for host in self._hosts:
            host.newton_finish(f, control)
        for host in self._hosts:
            host.newton_after(f)

    def _assemble(self, f, is_exact=False):
        """Assemble the Newton system of every participant from the host, then the share of the couplings."""
        for host in self._participant_hosts:
            host.newton_assemble(f, is_exact)
        if self._has_coupling_devices:
            kernel_contact_gradient(f, self._participants)

    def _pcg(self, f, is_adjoint=False):
        """Run the conjugate gradient iterations of a substep from its staged right-hand side, as one graph, over the
        environments whose Newton solve runs, or every environment with `is_adjoint`."""
        participants = self._participants
        pcg_scalars = self.pcg_scalars
        kernel_pcg_start(self.control, pcg_scalars, participants, is_adjoint)
        # FIXME: quadrants#946 - a graph refuses the ndarrays that own a gradient, so the iterations launch one by one,
        # with a device read each, when other participants take part in a scene tracking gradients
        if not (self._requires_grad and self._is_joint):
            kernel_pcg_loop(f, pcg_scalars, participants)
            return
        while True:
            kernel_pcg_iteration(f, pcg_scalars, participants)
            if not int(qd_to_numpy(pcg_scalars.is_running)):
                break

    def _newton_loop_launches(self, f):
        """Drive the Newton iterations from the host, the conjugate gradient iterations of each running as one graph."""
        control = self.control
        participants = self._participants
        while True:
            self._assemble(f)
            self._pcg(f)
            kernel_start_line_search(f, control, participants)
            while True:
                kernel_line_search_trial(f, control, participants)
                if not int(qd_to_numpy(control.is_line_search_running)):
                    break
            kernel_update_convergence(control)
            if not int(qd_to_numpy(control.is_newton_running)):
                break
