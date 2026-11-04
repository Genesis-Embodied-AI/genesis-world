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

    Every environment of the batch runs its own solve: the participants write its measures, the energy at its start
    point and at its trial point and its largest displacement update, and the phases below turn them into its own loop
    conditions, so that environments backtrack, converge and fail apart. The loops of the graph run while any
    environment runs (`is_newton_running`, `is_line_search_running`), an environment that stopped keeping its iterate.

    `failed` says how the last solve of an environment stopped short: NEWTON_FAILED_ITERATIONS at the iteration cap,
    NEWTON_FAILED_NON_FINITE at once on a measure that is not finite, which no further iteration can recover, or
    NEWTON_FAILED_LINE_SEARCH in strict mode at a line search that found no lower energy. `has_failed_non_finite` latches
    the environments stopped on a non-finite measure, which skip the solves of the rest of the step, until their reader
    clears it, and `has_failed_iterations` latches a solve of any environment at the iteration cap.

    A solve runs `max_newton_iter` iterations at most, each with `max_ls_iter` trial steps at most, the step shrinking
    by `ls_shrink` between two trials. With `is_strict`, a line search exhausting its trials without lowering the energy
    fails the solve instead of keeping its last trial.
    """

    def __init__(
        self,
        value_type,
        max_newton_iter: int,
        max_ls_iter: int,
        ls_shrink: float,
        is_strict: bool = False,
        n_envs: int = 1,
    ) -> None:
        self.value_type = value_type

        def scalar(dtype, value):
            array = qd.ndarray(dtype, shape=())
            array.fill(value)
            return array

        def per_env(dtype, value):
            array = qd.ndarray(dtype, shape=(n_envs,))
            array.fill(value)
            return array

        self.is_newton_running = scalar(gs.qd_int, 0)
        self.is_line_search_running = scalar(gs.qd_int, 0)
        self.has_failed_iterations = scalar(gs.qd_int, 0)
        self.max_newton_iter = scalar(gs.qd_int, max_newton_iter)
        self.max_ls_iter = scalar(gs.qd_int, max_ls_iter)
        self.ls_shrink = scalar(value_type, ls_shrink)
        self.is_strict = scalar(gs.qd_int, int(is_strict))
        self.disp_tol = scalar(value_type, 0.0)

        self.is_env_running = per_env(gs.qd_int, 0)
        self.is_env_line_search_running = per_env(gs.qd_int, 0)
        self.is_converged = per_env(gs.qd_int, 0)
        self.failed = per_env(gs.qd_int, 0)
        self.has_failed_non_finite = per_env(gs.qd_int, 0)
        self.newton_iter = per_env(gs.qd_int, 0)
        self.ls_iter = per_env(gs.qd_int, 0)
        self.alpha = per_env(value_type, 1.0)
        self.energy_start = per_env(value_type, 0.0)
        self.energy_trial = per_env(value_type, 0.0)
        self.max_disp = per_env(value_type, 0.0)


@qd.func(requires_top_level=True)
def func_reset_newton_control(control: qd.template()):
    """Start the solve of every environment but those stopped on a non-finite measure earlier in the step."""
    for _ in range(1):
        control.is_newton_running[()] = 0
    for i_b in range(control.failed.shape[0]):
        control.newton_iter[i_b] = 0
        control.failed[i_b] = 0
        control.is_converged[i_b] = 0
        control.is_env_running[i_b] = 1 - control.has_failed_non_finite[i_b]
        if control.is_env_running[i_b] != 0:
            qd.atomic_max(control.is_newton_running[()], 1)


@qd.func(requires_top_level=True)
def func_clear_non_finite(control: qd.template()):
    """Clear the latch of the solves stopped on a non-finite measure, once the participants read it."""
    for i_b in range(control.has_failed_non_finite.shape[0]):
        control.has_failed_non_finite[i_b] = 0


@qd.func(requires_top_level=True)
def func_reset_newton_measures(control: qd.template()):
    """Clear the largest displacement update, which the participants accumulate with their Newton direction."""
    for i_b in range(control.max_disp.shape[0]):
        control.max_disp[i_b] = 0.0


@qd.func(requires_top_level=True)
def func_reset_energy(control: qd.template(), is_trial: qd.template()):
    """Clear the start or the trial energy, which the participants accumulate."""
    for i_b in range(control.energy_start.shape[0]):
        if is_trial:
            control.energy_trial[i_b] = 0.0
        else:
            control.energy_start[i_b] = 0.0


@qd.func(requires_top_level=True)
def func_check_convergence(control: qd.template()):
    """Judge the convergence of every running environment on its Newton direction itself, before the line search.

    An environment converges once its largest displacement update falls under disp_tol. A direction or a start energy
    that is not finite fails its solve at once: the assertion fails in debug mode, and otherwise `failed` becomes
    NEWTON_FAILED_NON_FINITE, which skips its line search and stops its iterations.
    """
    for i_b in range(control.failed.shape[0]):
        if control.is_env_running[i_b] != 0:
            control.is_converged[i_b] = gs.qd_int(control.max_disp[i_b] < control.disp_tol[()])
            is_finite = qd_isfinite(control.max_disp[i_b]) and qd_isfinite(control.energy_start[i_b])
            if qd.static(qd_debug_asserts()):
                assert is_finite, (
                    "Newton solve: the direction or the energy of environment %d is not finite at iteration %d"
                    % (i_b, control.newton_iter[i_b])
                )
            if not is_finite:
                control.failed[i_b] = NEWTON_FAILED_NON_FINITE


@qd.func(requires_top_level=True)
def func_init_line_search(control: qd.template()):
    for _ in range(1):
        control.is_line_search_running[()] = 0
    for i_b in range(control.failed.shape[0]):
        control.alpha[i_b] = 1.0
        control.ls_iter[i_b] = 0
        control.is_env_line_search_running[i_b] = control.is_env_running[i_b]
        if control.is_env_running[i_b] != 0:
            qd.atomic_max(control.is_line_search_running[()], 1)


@qd.func(requires_top_level=True)
def func_accept_or_backtrack(control: qd.template()):
    """Accept the trial step of every environment in its line search once its energy is at most the start energy, or
    shrink its alpha for the next trial.

    A converged direction is accepted at its first trial without the energy test. Exhausting the trials keeps the last
    one, or in strict mode fails the solve with NEWTON_FAILED_LINE_SEARCH. The accepted trial energy is the start
    energy of the next Newton iteration, so the participants evaluate the start energy only once per solve, before the
    first iteration.
    """
    for _ in range(1):
        control.is_line_search_running[()] = 0
    for i_b in range(control.failed.shape[0]):
        if control.is_env_line_search_running[i_b] != 0:
            control.ls_iter[i_b] += 1
            is_accepted = (
                control.is_converged[i_b] != 0
                or control.failed[i_b] != 0
                or control.energy_trial[i_b] <= control.energy_start[i_b]
            )
            is_exhausted = control.ls_iter[i_b] >= control.max_ls_iter[()]
            if is_accepted or (is_exhausted and control.is_strict[()] == 0):
                control.is_env_line_search_running[i_b] = 0
                control.energy_start[i_b] = control.energy_trial[i_b]
            elif is_exhausted:
                control.is_env_line_search_running[i_b] = 0
                control.failed[i_b] = NEWTON_FAILED_LINE_SEARCH
            else:
                control.alpha[i_b] = control.ls_shrink[()] * control.alpha[i_b]
                qd.atomic_max(control.is_line_search_running[()], 1)


@qd.func(requires_top_level=True)
def func_update_convergence(control: qd.template()):
    """Stop every environment after a converged, a non-finite or a failed line-search iteration, and raise the failure
    flag at the iteration cap."""
    for _ in range(1):
        control.is_newton_running[()] = 0
    for i_b in range(control.failed.shape[0]):
        if control.is_env_running[i_b] != 0:
            control.newton_iter[i_b] += 1
            if control.failed[i_b] == NEWTON_FAILED_NON_FINITE:
                control.is_env_running[i_b] = 0
                control.has_failed_non_finite[i_b] = 1
            elif control.failed[i_b] == NEWTON_FAILED_LINE_SEARCH or control.is_converged[i_b] != 0:
                control.is_env_running[i_b] = 0
            elif control.newton_iter[i_b] >= control.max_newton_iter[()]:
                control.is_env_running[i_b] = 0
                control.failed[i_b] = NEWTON_FAILED_ITERATIONS
                qd.atomic_max(control.has_failed_iterations[()], 1)
            else:
                qd.atomic_max(control.is_newton_running[()], 1)
