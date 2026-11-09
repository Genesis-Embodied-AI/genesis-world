import quadrants as qd

import genesis as gs
from genesis.utils.data import ENVS, Data, DataKind, field
from genesis.utils.misc import qd_debug_asserts, qd_isfinite

# Values of NewtonControl.failed
NEWTON_FAILED_ITERATIONS = 1
NEWTON_FAILED_NON_FINITE = 2
NEWTON_FAILED_LINE_SEARCH = 3


class NewtonControl(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """Device-resident scalars steering the Newton solves of the environments of a batch with a backtracking line
    search inside one graph: the options of the solve, the flags of the batch, then the state of every environment."""

    max_newton_iter: qd.Tensor = field(int, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    max_ls_iter: qd.Tensor = field(int, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    ls_shrink: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    is_strict: qd.Tensor = field(int, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    disp_tol: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    is_newton_running: qd.Tensor = field(int, shape=(), ndarray=True)
    is_line_search_running: qd.Tensor = field(int, shape=(), ndarray=True)
    has_failed_iterations: qd.Tensor = field(int, shape=(), ndarray=True)
    is_env_running: qd.Tensor = field(int, ndarray=True)
    is_env_line_search_running: qd.Tensor = field(int, ndarray=True)
    is_converged: qd.Tensor = field(int, ndarray=True)
    failed: qd.Tensor = field(int, ndarray=True)
    has_failed_non_finite: qd.Tensor = field(int, ndarray=True)
    newton_iter: qd.Tensor = field(int, ndarray=True)
    ls_iter: qd.Tensor = field(int, ndarray=True)
    alpha: qd.Tensor = field(float, ndarray=True)
    energy_start: qd.Tensor = field(float, ndarray=True)
    energy_trial: qd.Tensor = field(float, ndarray=True)
    max_disp: qd.Tensor = field(float, ndarray=True)


@qd.func(requires_top_level=True)
def func_reset_newton_control(control: NewtonControl):
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
def func_clear_non_finite(control: NewtonControl):
    """Clear the latch of the solves stopped on a non-finite measure, once the participants read it."""
    for i_b in range(control.has_failed_non_finite.shape[0]):
        control.has_failed_non_finite[i_b] = 0


@qd.func(requires_top_level=True)
def func_reset_newton_measures(control: NewtonControl):
    """Clear the largest displacement update, which the participants accumulate with their Newton direction."""
    for i_b in range(control.max_disp.shape[0]):
        control.max_disp[i_b] = 0.0


@qd.func(requires_top_level=True)
def func_reset_energy(control: NewtonControl, is_trial: qd.template()):
    """Clear the start or the trial energy, which the participants accumulate."""
    for i_b in range(control.energy_start.shape[0]):
        if is_trial:
            control.energy_trial[i_b] = 0.0
        else:
            control.energy_start[i_b] = 0.0


@qd.func(requires_top_level=True)
def func_check_convergence(control: NewtonControl):
    """Judge the convergence of every running environment on its Newton direction, before the line search."""
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
def func_init_line_search(control: NewtonControl):
    for _ in range(1):
        control.is_line_search_running[()] = 0
    for i_b in range(control.failed.shape[0]):
        control.alpha[i_b] = 1.0
        control.ls_iter[i_b] = 0
        control.is_env_line_search_running[i_b] = control.is_env_running[i_b]
        if control.is_env_running[i_b] != 0:
            qd.atomic_max(control.is_line_search_running[()], 1)


@qd.func(requires_top_level=True)
def func_accept_or_backtrack(control: NewtonControl):
    """Accept the trial step of every environment once its energy is at most the start energy, or shrink its alpha."""
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
def func_update_convergence(control: NewtonControl):
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
