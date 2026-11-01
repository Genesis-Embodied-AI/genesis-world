import quadrants as qd

from genesis.utils.misc import qd_debug_asserts, qd_isfinite

# Values of NewtonControl.failed
NEWTON_FAILED_ITERATIONS = 1
NEWTON_FAILED_NON_FINITE = 2


@qd.data_oriented
class NewtonControl:
    """Device-resident scalars steering a Newton solve with a backtracking line search inside one graph.

    The participants of the solve write the measures: the energy at the start point and at the trial point, and the
    largest displacement update. The phases below turn them into the loop conditions.

    `failed` says how the last solve stopped short: NEWTON_FAILED_ITERATIONS at the iteration cap, or
    NEWTON_FAILED_NON_FINITE at once on a measure that is not finite, which no further iteration can recover.
    `failed_iterations` and `failed_non_finite` latch them over the solves until their reader clears them.
    """

    def __init__(self, value_type) -> None:
        self.value_type = value_type

        self.newton_cond = qd.ndarray(qd.i32, shape=())
        self.newton_cond.fill(0)
        self.ls_cond = qd.ndarray(qd.i32, shape=())
        self.ls_cond.fill(0)
        self.converged = qd.ndarray(qd.i32, shape=())
        self.converged.fill(0)
        # Yield flag of the graph checkpoints that only gate a resume, so that it stays zero
        self.never_yield = qd.ndarray(qd.i32, shape=())
        self.never_yield.fill(0)
        self.failed = qd.ndarray(qd.i32, shape=())
        self.failed.fill(0)
        self.failed_iterations = qd.ndarray(qd.i32, shape=())
        self.failed_iterations.fill(0)
        self.failed_non_finite = qd.ndarray(qd.i32, shape=())
        self.failed_non_finite.fill(0)
        self.newton_iter = qd.ndarray(qd.i32, shape=())
        self.newton_iter.fill(0)
        self.ls_iter = qd.ndarray(qd.i32, shape=())
        self.ls_iter.fill(0)
        self.max_newton_iter = qd.ndarray(qd.i32, shape=())
        self.max_newton_iter.fill(1024)
        self.max_ls_iter = qd.ndarray(qd.i32, shape=())
        self.max_ls_iter.fill(12)
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
def func_reset_newton_control(
    control: qd.template(),  # NewtonControl
):
    for _ in range(1):
        control.newton_iter[()] = 0
        control.failed[()] = 0
        control.converged[()] = 0
        control.newton_cond[()] = 1


@qd.func(requires_top_level=True)
def func_reset_newton_measures(
    control: qd.template(),  # NewtonControl
):
    """Clear the measures the participants accumulate during one Newton iteration."""
    for _ in range(1):
        control.max_disp[()] = 0.0


@qd.func(requires_top_level=True)
def func_check_convergence(
    control: qd.template(),  # NewtonControl
):
    """Judge the convergence on the Newton direction itself, before the line search.

    The solve converges once the largest displacement update falls under disp_tol. A direction or a start energy that is
    not finite fails the solve at once: the assertion fails in debug mode, and otherwise `failed` becomes
    NEWTON_FAILED_NON_FINITE, which skips the line search and ends the loop.
    """
    for _ in range(1):
        control.converged[()] = qd.i32(control.max_disp[()] < control.disp_tol[()])
        is_finite = qd_isfinite(control.max_disp[()]) and qd_isfinite(control.energy_start[()])
        if qd.static(qd_debug_asserts()):
            assert is_finite, (
                "Newton solve: the direction or the energy is not finite at iteration %d" % (control.newton_iter[()])
            )
        if not is_finite:
            control.failed[()] = NEWTON_FAILED_NON_FINITE


@qd.func(requires_top_level=True)
def func_init_line_search(
    control: qd.template(),  # NewtonControl
):
    for _ in range(1):
        control.alpha[()] = 1.0
        control.ls_iter[()] = 0
        control.ls_cond[()] = 1


@qd.func(requires_top_level=True)
def func_accept_or_backtrack(
    control: qd.template(),  # NewtonControl
):
    """Accept the trial step once its energy is at most the start energy, or halve alpha for the next trial.

    A converged direction is accepted at its first trial without the energy test. Exhausting the trials keeps the last
    one. The accepted trial energy is the start energy of the next Newton iteration, so the participants evaluate the
    start energy only once per solve, before the first iteration.
    """
    for _ in range(1):
        control.ls_iter[()] += 1
        is_done = (
            control.converged[()] != 0
            or control.failed[()] != 0
            or control.energy_trial[()] <= control.energy_start[()]
            or control.ls_iter[()] >= control.max_ls_iter[()]
        )
        if is_done:
            control.ls_cond[()] = 0
            control.energy_start[()] = control.energy_trial[()]
        else:
            control.alpha[()] = 0.5 * control.alpha[()]


@qd.func(requires_top_level=True)
def func_update_convergence(
    control: qd.template(),  # NewtonControl
):
    """Stop the loop after a converged or a non-finite iteration, and raise the failure flag at the iteration cap."""
    for _ in range(1):
        control.newton_iter[()] += 1
        if control.failed[()] == NEWTON_FAILED_NON_FINITE:
            control.newton_cond[()] = 0
            control.failed_non_finite[()] = 1
        elif control.converged[()] != 0:
            control.newton_cond[()] = 0
        elif control.newton_iter[()] >= control.max_newton_iter[()]:
            control.newton_cond[()] = 0
            control.failed[()] = NEWTON_FAILED_ITERATIONS
            control.failed_iterations[()] = 1
