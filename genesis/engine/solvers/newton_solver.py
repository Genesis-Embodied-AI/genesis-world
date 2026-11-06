"""Global Newton solver of the implicit solvers that couple strongly.

One Newton solve per substep minimizes the sum of the participants' incremental potentials and of their contacts, with a
line search on the total energy. The participants that assemble their operator (MPM, FEM, PBD) own a range of the rows
of one linear system (see newton_linear.py), to which their contacts add their blocks (see newton_contact.py). Rigid
solves its own rows within the same preconditioned conjugate gradient (PCG), matrix-free. The participants enter the
kernels as one template argument, `NewtonParticipants`."""

import collections
import functools
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.bcoo_matrix import func_sort_reduce_bcoo, sort_reduce_bcoo
from genesis.engine.numerics.linear_pcg import (
    PCGScalars,
    func_init_linear_pcg_direction,
    func_init_linear_pcg_residual,
    func_linear_pcg_direction,
    func_linear_pcg_dot_rz,
    func_linear_pcg_spmv,
    func_linear_pcg_update_xr,
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
from genesis.utils.data import DataItem, DataKind, iter_data, iter_data_oriented
from genesis.utils.misc import qd_to_numpy

from .base_solver import Solver
from .newton_linear import (
    MIN_SLOTS,
    NewtonLinearScratch,
    build_newton_linear,
    func_linear_layout,
    func_linear_visit,
    grown_capacity,
    retire_scratch,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator
    from genesis.options.solvers import NewtonOptions

# Default of NewtonOptions.adjoint_pcg_tol by the bits of the floating-point precision
DEFAULT_ADJOINT_PCG_TOL = {64: 1e-24, 32: 1e-12}


@qd.data_oriented
class NewtonParticipants:
    """Participants of a Newton solve, their contacts and their linear system, gathered into one kernel argument.

    `owners` is a @qd.data_oriented NamedTuple (see `participant_group`) of the participants that assemble their share
    of `linear`, `rigid` the participant of Rigid or None, and `members` all of them. `contact` is the NewtonContact of
    the solve or None. They stay members of this object, which the kernel cache can weakly reference."""

    def __init__(self, owners: list, rigid, contact, linear: NewtonLinearScratch):
        self.owners = participant_group("NewtonOwners", owners)
        self.rigid = rigid
        self.has_rigid = rigid is not None
        self.members = participant_group("NewtonMembers", [*owners, *([] if rigid is None else [rigid])])
        self.contact = contact
        self.has_contact = contact is not None
        self.linear = linear


class NewtonDevice:
    """Base of the device objects of the participants and the couplings of a Newton solve, whose `data` yields the
    declared structs (see Data) that the Newton solver frees with its own."""

    @property
    def data(self) -> Iterator[DataItem]:
        yield from ()


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

    `newton_begin` runs before the pre-coupling phase of every solver. A participant reverses the part of its substep
    that the solve takes over around the adjoint solve (`newton_backward`) and flags the solves stopped on a non-finite
    measure (`newton_flag_non_finite`). `newton_finish` and `newton_end` leave the environments of
    `control.has_failed_non_finite` as they were at the substep start."""

    is_differentiable = False

    def newton_begin(self, f):
        pass

    def newton_prepare(self, f):
        pass

    def newton_flag_non_finite(self, control):
        pass

    def newton_finish(self, f, control):
        pass

    def newton_after(self, f):
        pass

    def newton_end(self, f, control):
        pass

    def newton_adjoint_rhs(self, f):
        pass

    def newton_adjoint_finish(self, f):
        pass


# ============================================== composed stages ==============================================


@qd.func(requires_top_level=True)
def func_assemble_terms(f: int, participants: qd.template(), is_exact: qd.template()):
    """Assemble the Newton system at the current iterate but for the reduction of the matrix: every owner of rows
    writes its blocks and its gradient into the linear system and Rigid assembles its own, then the contacts add
    theirs."""
    for member in qd.static(participants.members):
        member.func_assemble(f, is_exact)
    if qd.static(participants.has_contact):
        participants.contact.func_contact_assemble(f, participants.linear)


@qd.func(requires_top_level=True)
def func_assemble(f: int, participants: qd.template(), is_exact: qd.template()):
    func_assemble_terms(f, participants, is_exact)
    func_sort_reduce_bcoo(participants.linear.matrix, block_scalars=9)


@qd.func(requires_top_level=True)
def func_pcg_start(
    f: int,
    newton_control: qd.template(),
    pcg_scalars: qd.template(),
    participants: qd.template(),
    is_adjoint: qd.template(),
):
    """Build the preconditioner and start the preconditioned conjugate gradient from 0 over all the rows.

    The owners take their diagonal blocks from one traverse of the matrix (see func_linear_visit), and the contacts add
    theirs on Rigid, before they factor them. The environments whose Newton solve stopped take no part, except in an
    adjoint solve."""
    func_pcg_select_envs(pcg_scalars, newton_control, is_adjoint)
    func_pcg_reset(pcg_scalars)
    for member in qd.static(participants.members):
        member.func_factor_begin()
    func_linear_visit(participants.linear, participants.owners)
    if qd.static(participants.has_rigid):
        participants.contact.func_contact_visit_rigid(f)
    for member in qd.static(participants.members):
        member.func_factor()
    func_init_linear_pcg_residual(
        participants.linear.b_rhs, participants.linear.solution, participants.linear.pcg, pcg_scalars, is_adjoint
    )
    for owner in qd.static(participants.owners):
        owner.func_precondition()
    func_init_linear_pcg_direction(participants.linear.pcg, pcg_scalars)
    if qd.static(participants.has_rigid):
        participants.rigid.func_pcg_init(pcg_scalars, is_adjoint)
        participants.rigid.func_precondition()
        participants.rigid.func_pcg_init_direction(pcg_scalars)
    func_pcg_arm(pcg_scalars)


@qd.func(requires_top_level=True)
def func_pcg_iteration(f: int, pcg_scalars: qd.template(), participants: qd.template()):
    """Run one preconditioned conjugate gradient iteration over all the rows: the product of the matrix and of Rigid
    with p . Ap, the step, the preconditioner and the direction (see PCGScalars)."""
    func_linear_pcg_spmv(participants.linear.pcg, pcg_scalars, participants.linear.matrix)
    if qd.static(participants.has_rigid):
        participants.rigid.func_matvec(pcg_scalars)
        participants.contact.func_contact_rigid_product(f, participants.linear, pcg_scalars)
        participants.rigid.func_matvec_dot(pcg_scalars)
    func_pcg_step(pcg_scalars)
    func_linear_pcg_update_xr(participants.linear.solution, participants.linear.pcg, pcg_scalars)
    for owner in qd.static(participants.owners):
        owner.func_precondition()
    func_linear_pcg_dot_rz(participants.linear.pcg, pcg_scalars)
    if qd.static(participants.has_rigid):
        participants.rigid.func_pcg_update_xr(pcg_scalars)
        participants.rigid.func_precondition()
        participants.rigid.func_pcg_dot_rz(pcg_scalars)
    func_pcg_decide(pcg_scalars)
    func_linear_pcg_direction(participants.linear.pcg, pcg_scalars)
    if qd.static(participants.has_rigid):
        participants.rigid.func_pcg_direction(pcg_scalars)


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
    """Move every participant to the trial point of the line-search step."""
    for member in qd.static(participants.members):
        member.func_step_forward(newton_control, f)


@qd.func(requires_top_level=True)
def func_energy(f: int, newton_control: qd.template(), participants: qd.template(), is_trial: qd.template()):
    """Evaluate the total energy at the start or trial point, which every participant and the contacts add theirs
    to."""
    func_reset_energy(newton_control, is_trial)
    if qd.static(is_trial):
        func_add_energies(f, participants, newton_control.energy_trial)
    else:
        func_add_energies(f, participants, newton_control.energy_start)


@qd.func(requires_top_level=True)
def func_add_energies(f: int, participants: qd.template(), energies: qd.template()):
    """Add the energy of every participant and of the contacts into the entry of every environment in `energies`."""
    for member in qd.static(participants.members):
        member.func_energy(f, energies)
    if qd.static(participants.has_contact):
        participants.contact.func_contact_energy(f, energies)


# ================================================== kernels ==================================================


@qd.kernel
def kernel_assemble_terms(f: int, participants: qd.template(), is_exact: qd.template()):
    # The host launches the reduction apart (see sort_reduce_bcoo)
    func_assemble_terms(f, participants, is_exact)


@qd.kernel
def kernel_linear_layout(participants: qd.template()):
    func_linear_layout(participants)


@qd.kernel
def kernel_pcg_start(
    f: int,
    newton_control: qd.template(),
    pcg_scalars: qd.template(),
    participants: qd.template(),
    is_adjoint: qd.template(),
):
    func_pcg_start(f, newton_control, pcg_scalars, participants, is_adjoint)


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
    """Run the Newton iterations of a substep as one graph, after the evaluation of the start energy, the storage sized
    before the loop holding every assembly in it."""
    while qd.graph.do_while(newton_control.is_newton_running):
        func_assemble(f, participants, False)
        func_pcg_start(f, newton_control, pcg_scalars, participants, False)
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

    It owns no physical state, and runs its substep at the start of the post-coupling phase. The coupler registers every
    participant and coupling (see LegacyCoupler) as a host object (see NewtonHost) and a device object, with the
    contacts of the solve (see NewtonContact)."""

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
        self._owner_devices = []
        self._contact = None
        self._linears_replaced = []
        self._host_only_devices = ()
        self._devices = []
        self._is_loop_graph = options.graph == "newton"
        self._has_warned_adjoint_projection = False

    def register(self, owners, rigid, couplings, contact, host_only_devices=()):
        """Register the participants of the Newton solve and the couplings between them, each as a pair of its host
        object and its device object, and the contacts of the solve, which activates the solver.

        `owners` assemble their share of the linear system, and `rigid`, the participant of Rigid or None, solves its
        own rows. The device of a coupling is a pair kind of `contact`. `host_only_devices` lists the devices that the
        solve reads without being participants of it."""
        participants = [*owners, *([] if rigid is None else [rigid])]
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
        # FIXME: quadrants#946 - a graph refuses the ndarrays that own a gradient, which those of the participants
        # besides MPM do in a scene tracking gradients
        if self._is_loop_graph and self._requires_grad and (len(participants) > 1 or couplings):
            gs.raise_exception(
                "NewtonOptions(graph='newton') does not support a differentiable scene coupling MPM with another "
                "solver."
            )
        self.control.disp_tol.from_numpy(
            np.array(self._options.velocity_tol * self._sim._substep_dt, dtype=gs.np_float)
        )
        self._participant_hosts = [host for host, _ in participants]
        self._coupling_hosts = [host for host, _ in couplings]
        self._owner_devices = [device for _, device in owners]
        self._participants = NewtonParticipants(self._owner_devices, None if rigid is None else rigid[1], contact, None)
        self._bind_linear(build_newton_linear(self._max_linear_rows, MIN_SLOTS))
        self._contact = contact
        self._host_only_devices = tuple(host_only_devices)
        self._devices.clear()
        self._devices.extend(device for _, device in participants)
        self._devices.extend(device for _, device in couplings)
        self._devices.extend(host_only_devices)
        if contact is not None:
            self._devices.append(contact)

    @property
    def is_active(self):
        return self._participants is not None

    @property
    def data(self):
        """Yield the scalars of the solve and the arrays of its participants and couplings (see 'Solver.data'), which
        every substep rebuilds before reading them, so that no checkpoint reads them."""
        if self._participants is not None:
            yield from iter_data(self._participants, "linear")
            yield from iter_data_oriented("control", self.control, DataKind.SCRATCH)
            yield from iter_data_oriented("pcg_scalars", self.pcg_scalars, DataKind.SCRATCH)
            # One walk over both, so that an array the host-only devices share with the participants is yielded once
            yield from iter_data_oriented(
                "participants", (self._participants, self._host_only_devices), DataKind.SCRATCH
            )
            for device in self._devices:
                yield from device.data

    @property
    def _max_linear_rows(self):
        """Block rows that the owners of the linear system can hold at their capacity."""
        return sum(device.max_linear_rows for device in self._owner_devices)

    def _prepare_linear(self):
        """Lay the linear system out for the substep, once every owner and the contacts reported their counts, and grow
        it to them.

        The rows grow with the capacity of the owners, and the triplets with the demand of the layout."""
        participants = self._participants
        kernel_linear_layout(participants)
        linear = participants.linear
        max_rows = self._max_linear_rows
        is_overflow = bool(int(qd_to_numpy(linear.matrix.has_triplet_overflow)))
        if is_overflow or max_rows > linear.pcg.row_env.shape[0]:
            max_triplets = linear.matrix.triplet_row.shape[0]
            if is_overflow:
                max_triplets = max(grown_capacity(int(qd_to_numpy(linear.matrix.triplet_demand))), max_triplets)
            retire_scratch(linear, self._linears_replaced)
            self._bind_linear(build_newton_linear(max_rows, max_triplets))
            kernel_linear_layout(participants)

    def _bind_linear(self, linear: NewtonLinearScratch):
        """Hand the linear system to the solve and to every owner of its rows, which reads it as `linear`."""
        self._participants.linear = linear
        for device in self._owner_devices:
            device.linear = linear

    @property
    def _hosts(self):
        return [*self._participant_hosts, *self._coupling_hosts]

    @property
    def _is_joint(self):
        """Whether the device stages of the solve read the data of another solver than MPM, a participant or the partner
        of a coupling."""
        return len(self._participant_hosts) > 1 or bool(self._coupling_hosts)

    def begin(self, f):
        """Run the host stages that precede the pre-coupling phase of the solvers in substep `f`."""
        if not self.is_active:
            return
        for host in self._hosts:
            host.newton_begin(f)

    def step(self, f):
        """Run the Newton solve of substep `f`: the preparation of the participants, the start energy, the Newton
        iterations and the hand-back of the solution.

        The launches form one entry of the step recorded for `backward`, which differentiates the solve by the implicit
        function theorem. An environment whose solve stopped on a non-finite measure keeps its state for the rest of the
        step, which raises for it at its end (see `_check_failures`)."""
        if not self.is_active:
            return
        with custom_backward(self.backward, f):
            self._step(f)
        if self._options.strict:
            self._check_line_search(f)
        # What follows the solve is reversed by its own autodiff, such as the integration of Rigid's deferred substep
        for host in self._hosts:
            host.newton_end(f, self.control)
        self._check_failures()

    def _check_line_search(self, f):
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
        """Report the solves that failed since the last check: the non-finite ones to every participant, an error being
        raised at the beginning of the next step, and those that reached `max_iterations` as a warning, every
        `failure_check_interval` substeps."""
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
        """Reverse the Newton solve of a substep, from the state left by the replay of its forward pass, each
        participant wrapping the adjoint solve in its own reverse of the part of its substep that the solve takes
        over."""
        if not all(host.is_differentiable for host in self._hosts):
            gs.raise_exception(
                "The backward pass of the joint Newton solve supports the participants of Rigid alone besides MPM."
            )
        solve = functools.partial(self._solve_adjoint, f)
        for host in self._participant_hosts:
            solve = functools.partial(host.newton_backward, f, solve)
        solve()

    def _solve_adjoint(self, f):
        mode = self._options.adjoint_hessian
        is_exact = mode in ("exact", "auto")
        pcg_scalars = self.pcg_scalars
        pcg_scalars.is_env_adjoint.fill(1)
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
                self._has_warned_adjoint_projection = True
            pcg_scalars.is_env_adjoint.from_numpy(envs_breakdown)
            self._stage_adjoint(f, is_exact=False)
            self._pcg_adjoint(f)
        for host in self._hosts:
            host.newton_adjoint_finish(f)

    def _stage_adjoint(self, f, is_exact):
        self._assemble(f, is_exact)
        for host in self._hosts:
            host.newton_adjoint_rhs(f)

    def _pcg_adjoint(self, f, check_curvature=False):
        """Solve the staged adjoint system by the conjugate gradient, over the environments of `is_env_adjoint`.

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
        if self._contact is not None:
            self._contact.prepare(f)
        self._prepare_linear()
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
        kernel_assemble_terms(f, self._participants, is_exact)
        sort_reduce_bcoo(self._participants.linear.matrix, block_scalars=9)

    def _pcg(self, f, is_adjoint=False):
        """Run the conjugate gradient iterations of a substep from its staged right-hand side, as one graph, over the
        environments whose Newton solve runs, or every environment with `is_adjoint`."""
        participants = self._participants
        pcg_scalars = self.pcg_scalars
        kernel_pcg_start(f, self.control, pcg_scalars, participants, is_adjoint)
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
