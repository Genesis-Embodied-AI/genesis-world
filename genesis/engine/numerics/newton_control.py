import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_debug_asserts, qd_isfinite

# Values of NewtonControl.failed
NEWTON_FAILED_ITERATIONS = 1
NEWTON_FAILED_NON_FINITE = 2
NEWTON_FAILED_LINE_SEARCH = 3


@qd.data_oriented
class NewtonControl:
    """Device-resident scalars steering a Newton solve with a backtracking line search inside one graph.

    The participants of the solve write the measures: the energy at the start point and at the trial point, and the
    largest displacement update. The phases below turn them into the loop conditions.

    `failed` says how the last solve stopped short: NEWTON_FAILED_ITERATIONS at the iteration cap,
    NEWTON_FAILED_NON_FINITE at once on a measure that is not finite, which no further iteration can recover, or
    NEWTON_FAILED_LINE_SEARCH in strict mode at a line search that found no lower energy.
    `has_failed_iterations` and `has_failed_non_finite` latch the first two over the solves until their reader clears
    them.

    A solve runs `max_newton_iter` iterations at most, each with `max_ls_iter` trial steps at most, the step shrinking
    by `ls_shrink` between two trials. With `is_strict`, a line search exhausting its trials without lowering the energy
    fails the solve instead of keeping its last trial.
    """

    def __init__(
        self, value_type, max_newton_iter: int, max_ls_iter: int, ls_shrink: float, is_strict: bool = False
    ) -> None:
        self.value_type = value_type

        # FIXME: The environments of a batch share these scalars, so they backtrack, converge and fail together, and the
        # result of one environment depends on the others. Scalars per environment need reductions per environment.
        self.is_newton_running = qd.ndarray(gs.qd_int, shape=())
        self.is_newton_running.fill(0)
        self.is_line_search_running = qd.ndarray(gs.qd_int, shape=())
        self.is_line_search_running.fill(0)
        self.is_converged = qd.ndarray(gs.qd_int, shape=())
        self.is_converged.fill(0)
        self.failed = qd.ndarray(gs.qd_int, shape=())
        self.failed.fill(0)
        self.has_failed_iterations = qd.ndarray(gs.qd_int, shape=())
        self.has_failed_iterations.fill(0)
        self.has_failed_non_finite = qd.ndarray(gs.qd_int, shape=())
        self.has_failed_non_finite.fill(0)
        self.newton_iter = qd.ndarray(gs.qd_int, shape=())
        self.newton_iter.fill(0)
        self.ls_iter = qd.ndarray(gs.qd_int, shape=())
        self.ls_iter.fill(0)
        self.max_newton_iter = qd.ndarray(gs.qd_int, shape=())
        self.max_newton_iter.fill(max_newton_iter)
        self.max_ls_iter = qd.ndarray(gs.qd_int, shape=())
        self.max_ls_iter.fill(max_ls_iter)
        self.ls_shrink = qd.ndarray(value_type, shape=())
        self.ls_shrink.fill(ls_shrink)
        self.is_strict = qd.ndarray(gs.qd_int, shape=())
        self.is_strict.fill(int(is_strict))
        self.disp_tol = qd.ndarray(value_type, shape=())
        self.disp_tol.fill(0.0)
        self.alpha = qd.ndarray(value_type, shape=())
        self.alpha.fill(1.0)
        self.energy_start = qd.ndarray(value_type, shape=())
        self.energy_start.fill(0.0)
        self.energy_trial = qd.ndarray(value_type, shape=())
        self.energy_trial.fill(0.0)
        self.max_disp = qd.ndarray(value_type, shape=())
        self.max_disp.fill(0.0)


@qd.func(requires_top_level=True)
def func_reset_newton_control(control: qd.template()):
    for _ in range(1):
        control.newton_iter[()] = 0
        control.failed[()] = 0
        control.is_converged[()] = 0
        control.is_newton_running[()] = 1


@qd.func(requires_top_level=True)
def func_clear_non_finite(control: qd.template()):
    """Clear the latch of the solves stopped on a non-finite measure, once the participants read it."""
    for _ in range(1):
        control.has_failed_non_finite[()] = 0


@qd.func(requires_top_level=True)
def func_reset_newton_measures(control: qd.template()):
    """Clear the largest displacement update, which the participants accumulate with their Newton direction."""
    for _ in range(1):
        control.max_disp[()] = 0.0


@qd.func(requires_top_level=True)
def func_reset_energy(control: qd.template(), is_trial: bool):
    """Clear the start or the trial energy, which the participants accumulate."""
    for _ in range(1):
        if is_trial:
            control.energy_trial[()] = 0.0
        else:
            control.energy_start[()] = 0.0


@qd.func(requires_top_level=True)
def func_check_convergence(control: qd.template()):
    """Judge the convergence on the Newton direction itself, before the line search.

    The solve converges once the largest displacement update falls under disp_tol. A direction or a start energy that is
    not finite fails the solve at once: the assertion fails in debug mode, and otherwise `failed` becomes
    NEWTON_FAILED_NON_FINITE, which skips the line search and ends the loop.
    """
    for _ in range(1):
        control.is_converged[()] = gs.qd_int(control.max_disp[()] < control.disp_tol[()])
        is_finite = qd_isfinite(control.max_disp[()]) and qd_isfinite(control.energy_start[()])
        if qd.static(qd_debug_asserts()):
            assert is_finite, (
                "Newton solve: the direction or the energy is not finite at iteration %d" % (control.newton_iter[()])
            )
        if not is_finite:
            control.failed[()] = NEWTON_FAILED_NON_FINITE


@qd.func(requires_top_level=True)
def func_init_line_search(control: qd.template()):
    for _ in range(1):
        control.alpha[()] = 1.0
        control.ls_iter[()] = 0
        control.is_line_search_running[()] = 1


@qd.func(requires_top_level=True)
def func_accept_or_backtrack(control: qd.template()):
    """Accept the trial step once its energy is at most the start energy, or shrink alpha for the next trial.

    A converged direction is accepted at its first trial without the energy test. Exhausting the trials keeps the last
    one, or in strict mode fails the solve with NEWTON_FAILED_LINE_SEARCH. The accepted trial energy is the start
    energy of the next Newton iteration, so the participants evaluate the start energy only once per solve, before the
    first iteration.
    """
    for _ in range(1):
        control.ls_iter[()] += 1
        is_accepted = (
            control.is_converged[()] != 0
            or control.failed[()] != 0
            or control.energy_trial[()] <= control.energy_start[()]
        )
        is_exhausted = control.ls_iter[()] >= control.max_ls_iter[()]
        if is_accepted or (is_exhausted and control.is_strict[()] == 0):
            control.is_line_search_running[()] = 0
            control.energy_start[()] = control.energy_trial[()]
        elif is_exhausted:
            control.is_line_search_running[()] = 0
            control.failed[()] = NEWTON_FAILED_LINE_SEARCH
        else:
            control.alpha[()] = control.ls_shrink[()] * control.alpha[()]


@qd.func(requires_top_level=True)
def func_update_convergence(control: qd.template()):
    """Stop the loop after a converged, a non-finite or a failed line-search iteration, and raise the failure flag at
    the iteration cap."""
    for _ in range(1):
        control.newton_iter[()] += 1
        if control.failed[()] == NEWTON_FAILED_NON_FINITE:
            control.is_newton_running[()] = 0
            control.has_failed_non_finite[()] = 1
        elif control.failed[()] == NEWTON_FAILED_LINE_SEARCH or control.is_converged[()] != 0:
            control.is_newton_running[()] = 0
        elif control.newton_iter[()] >= control.max_newton_iter[()]:
            control.is_newton_running[()] = 0
            control.failed[()] = NEWTON_FAILED_ITERATIONS
            control.has_failed_iterations[()] = 1
