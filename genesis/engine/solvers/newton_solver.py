"""Global Newton solver of the implicit solvers that couple strongly.

One Newton solve per substep minimizes the sum of the participants' incremental potentials and of the couplings
between them. The linear system of every Newton iteration is solved by one preconditioned conjugate gradient over the
rows of all participants, preconditioned block-diagonally by each participant over its own rows, and the line search
runs on the total energy.

A participant owns its unknowns, its energy, its share of the gradient and of the operator, and a preconditioner over
its own rows. The primary participant also owns the BCOO matrix and the PCG workspace whose device scalars (r.z, p.Ap,
the threshold, the convergence flag) every participant reduces into; a secondary one keeps its vectors and adds its
partial dot products at fixed points of the iteration. A coupling owns only the knowledge of both sides and adds its
cross terms to their gradients and operators.

The participant and coupling objects enter the kernels as templates, in a fixed number of slots: an absent one is a
`NullParticipant`. Quadrants registers the arrays of a template only through @qd.data_oriented attributes and frozen
dataclasses, so the slots are kernel arguments rather than a list.
"""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.newton_control import (
    NewtonControl,
    func_accept_or_backtrack,
    func_check_convergence,
    func_init_line_search,
    func_update_convergence,
)

from .base_solver import Solver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# Run the Newton iterations as one graph (`kernel_newton_loop`), or drive them from the host with only the PCG
# iterations as a graph. The graph runs faster, the host loop compiles faster.
# FIXME: The Newton graph compiles cold in about 47 s against about 15 s for its stages as separate kernels (the
# compile time of a graph grows faster than its size), and a change of any stage in it recompiles it whole.
USE_NEWTON_GRAPH = False

# Slots of the composed kernels
N_SECONDARY_SLOTS = 2
N_COUPLING_SLOTS = 2


@qd.data_oriented
class NullParticipant:
    """Stands in for an absent secondary participant or coupling of the composed kernels: every stage does nothing."""

    @qd.func(requires_top_level=True)
    def func_evaluate(self):
        pass

    @qd.func(requires_top_level=True)
    def func_gradient(self):
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
    def func_start_line_search(self, control: qd.template()):
        pass

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template()):
        pass

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), is_trial):
        pass

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f):
        pass

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f):
        pass

    @qd.func(requires_top_level=True)
    def func_contact_matvec_post(self):
        pass

    @qd.func(requires_top_level=True)
    def func_after_step(self, f):
        pass


# ============================================== composed stages ==============================================


@qd.func(requires_top_level=True)
def func_assemble_secondary(
    s0: qd.template(),
    s1: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    f,
):
    """The secondary participants' gradients and factors at the current iterate, after the primary's assembly staged
    what the couplings read of it."""
    s0.func_evaluate()
    s0.func_gradient()
    s1.func_evaluate()
    s1.func_gradient()
    c0.func_contact_gradient(f)
    c1.func_contact_gradient(f)
    s0.func_factor()
    s1.func_factor()


@qd.func(requires_top_level=True)
def func_pcg_start(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
):
    """Each participant's preconditioner, and the PCG start from 0: r = -grad, z = M^-1 r, p = z, rz."""
    p.func_pcg_prepare()
    p.func_pcg_init_residual()
    s0.func_pcg_init()
    s1.func_pcg_init()
    p.func_precondition()
    s0.func_precondition()
    s1.func_precondition()
    p.func_pcg_init_direction()
    s0.func_pcg_init_direction(p.ws)
    s1.func_pcg_init_direction(p.ws)


@qd.func(requires_top_level=True)
def func_pcg_iteration(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    f,
):
    """One PCG iteration over the rows of every participant, with the couplings' cross terms in the operator."""
    p.func_pcg_spmv()
    c0.func_contact_matvec(f)
    c1.func_contact_matvec(f)
    s0.func_matvec()
    s1.func_matvec()
    c0.func_contact_matvec_post()
    c1.func_contact_matvec_post()
    s0.func_matvec_dot(p.ws)
    s1.func_matvec_dot(p.ws)
    s0.func_pcg_update_xr(p.ws)
    s1.func_pcg_update_xr(p.ws)
    p.func_pcg_update_xr()
    p.func_precondition()
    s0.func_precondition()
    s1.func_precondition()
    s0.func_pcg_dot_rz(p.ws)
    s1.func_pcg_dot_rz(p.ws)
    p.func_pcg_direction_update()
    s0.func_pcg_direction(p.ws)
    s1.func_pcg_direction(p.ws)


@qd.func(requires_top_level=True)
def func_start_line_search(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    f,
):
    """The Newton direction of every participant from the PCG solution, its measure, and the line search armed."""
    s0.func_start_line_search(control)
    s1.func_start_line_search(control)
    p.func_start_line_search(control, f)
    func_check_convergence(control)
    func_init_line_search(control)


@qd.func(requires_top_level=True)
def func_step_forward(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    f,
):
    """The trial point of every participant at the line-search step, and what the couplings derive from it."""
    p.func_step_forward(control)
    s0.func_step_forward(control)
    s1.func_step_forward(control)
    c0.func_after_step(f)
    c1.func_after_step(f)


@qd.func(requires_top_level=True)
def func_energy(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    f,
    is_trial,
):
    """The total energy at the start or trial point: the primary writes it, the secondary participants add theirs."""
    p.func_energy(control, f, is_trial)
    s0.func_energy(control, is_trial)
    s1.func_energy(control, is_trial)


# ================================================== kernels ==================================================


@qd.kernel
def kernel_assemble_secondary(
    s0: qd.template(),
    s1: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    f: qd.i32,
):
    func_assemble_secondary(s0, s1, c0, c1, f)


@qd.kernel
def kernel_pcg_start(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
):
    func_pcg_start(p, s0, s1)


@qd.kernel(graph=True)
def kernel_pcg_loop(
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    f: qd.i32,
):
    """The PCG iterations after kernel_pcg_start, until convergence or the iteration budget, as one graph."""
    while qd.graph.do_while(p.ws.cond):
        func_pcg_iteration(p, s0, s1, c0, c1, f)


@qd.kernel
def kernel_start_line_search(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    f: qd.i32,
):
    func_start_line_search(control, p, s0, s1, f)


@qd.kernel
def kernel_step_forward(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    c0: qd.template(),
    c1: qd.template(),
    f: qd.i32,
):
    func_step_forward(control, p, s0, s1, c0, c1, f)


@qd.kernel
def kernel_energy(
    control: qd.template(),
    p: qd.template(),
    s0: qd.template(),
    s1: qd.template(),
    f: qd.i32,
    is_trial: qd.i32,
):
    func_energy(control, p, s0, s1, f, is_trial)


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
    c0: qd.template(),
    c1: qd.template(),
    f: qd.i32,
):
    """The Newton iterations of a substep as one graph, after the start energy armed the loop.

    The primary's assembly must not need storage growth inside the loop: the MPM triplet demand depends on the grid
    topology only, fixed for the substep, so the assembly that sized the storage before the loop covers every one in it.
    """
    while qd.graph.do_while(control.newton_cond):
        p.func_assemble(control, f)
        func_assemble_secondary(s0, s1, c0, c1, f)
        func_pcg_start(p, s0, s1)
        while qd.graph.do_while(p.ws.cond):
            func_pcg_iteration(p, s0, s1, c0, c1, f)
        func_start_line_search(control, p, s0, s1, f)
        while qd.graph.do_while(control.ls_cond):
            func_step_forward(control, p, s0, s1, c0, c1, f)
            func_energy(control, p, s0, s1, f, 1)
            func_accept_or_backtrack(control)
        func_update_convergence(control)


# =================================================== solver ===================================================


class NewtonSolver(Solver):
    """One Newton solve per substep over the participants that register into it (see the module docstring).

    The simulator holds it apart from its solvers, since it owns no physical state, and runs its substep at the start
    of the post-coupling phase, after every solver's pre-coupling phase and the coupler, and before every solver's own
    post-coupling phase.

    Each participant and coupling registers a host object, whose methods run the host-side stages around the solve
    (`newton_prepare`, `newton_finish`, `newton_after`, and `newton_assemble` for the primary), and a device object
    for the composed kernels.
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
        """The Newton solve of substep f: the participants' preparation, the start energy, the Newton iterations and
        the hand-back of the solution."""
        if not self.is_active or self._primary[0].runs_own_substep():
            return
        control = self.control
        primary, secondary, coupling = self._slots()
        hosts = [self._primary[0]] + [host for host, _ in self._secondaries]
        coupling_hosts = [host for host, _ in self._couplings]

        for host in hosts + coupling_hosts:
            host.newton_prepare(f)
        kernel_energy(control, primary, *secondary, f, 0)

        if USE_NEWTON_GRAPH:
            kernel_newton_loop(control, primary, *secondary, *coupling, f)
        else:
            self._newton_loop_launches(f)

        for host in hosts + coupling_hosts:
            host.newton_finish(f)
        for host in hosts + coupling_hosts:
            host.newton_after(f)

    def _newton_loop_launches(self, f):
        """The Newton iterations driven from the host, the PCG iterations of each as one graph."""
        control = self.control
        primary, secondary, coupling = self._slots()
        has_secondary = bool(self._secondaries or self._couplings)
        while True:
            self._primary[0].newton_assemble(f)
            if has_secondary:
                kernel_assemble_secondary(*secondary, *coupling, f)
            kernel_pcg_start(primary, *secondary)
            kernel_pcg_loop(primary, *secondary, *coupling, f)
            kernel_start_line_search(control, primary, *secondary, f)
            while True:
                kernel_step_forward(control, primary, *secondary, *coupling, f)
                kernel_energy(control, primary, *secondary, f, 1)
                kernel_accept_or_backtrack(control)
                if not int(control.ls_cond.to_numpy()):
                    break
            kernel_update_convergence(control)
            if not int(control.newton_cond.to_numpy()):
                break
