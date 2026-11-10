import quadrants as qd

import genesis as gs
from genesis.utils.data import ENVS, Data, DataKind, field
from genesis.utils.misc import qd_debug_asserts, qd_isfinite

# Values of NewtonControl.failure_code
NEWTON_FAILED_ITERATIONS = 1
NEWTON_FAILED_NON_FINITE = 2
NEWTON_FAILED_LINE_SEARCH = 3


class NewtonControl(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """Device-resident scalars steering the Newton solves of the environments of a batch with a backtracking line
    search inside one graph: the options of the solve, the flags of the batch, then the state of every environment."""

    max_iterations: qd.Tensor = field(int, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    line_search_max_iterations: qd.Tensor = field(int, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    line_search_shrink: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    is_strict: qd.Tensor = field(int, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    displacement_tol: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)
    is_newton_running: qd.Tensor = field(int, shape=(), ndarray=True)
    is_line_search_running: qd.Tensor = field(int, shape=(), ndarray=True)
    has_failed_iterations: qd.Tensor = field(int, shape=(), ndarray=True)
    is_env_running: qd.Tensor = field(int, ndarray=True)
    is_env_line_search_running: qd.Tensor = field(int, ndarray=True)
    is_converged: qd.Tensor = field(int, ndarray=True)
    failure_code: qd.Tensor = field(int, ndarray=True)
    has_failed_non_finite: qd.Tensor = field(int, ndarray=True)
    n_iterations: qd.Tensor = field(int, ndarray=True)
    n_line_search_iterations: qd.Tensor = field(int, ndarray=True)
    alpha: qd.Tensor = field(float, ndarray=True)
    energy_start: qd.Tensor = field(float, ndarray=True)
    energy_trial: qd.Tensor = field(float, ndarray=True)
    max_displacement: qd.Tensor = field(float, ndarray=True)


# =========================================== Layer 0: Newton step and inertia ===========================================


@qd.func
def newton_direction(solution):
    """Newton direction of an unknown from the solution of its linear system, the step that lowers the energy."""
    return -solution


@qd.func
def line_search_point(start, alpha: float, direction):
    return start + alpha * direction


@qd.func
def step_measure(step):
    """Largest component of a Newton step, the measure of its convergence (see func_check_convergence)."""
    measure = gs.qd_float(0.0)
    if qd.static(step.is_tensor()):
        measure = qd.abs(step).max()
    else:
        measure = qd.abs(step)
    return measure


@qd.func
def inertia_energy(mass: float, x, target):
    return 0.5 * mass * (x - target).dot(x - target)


@qd.func
def inertia_gradient(mass: float, x, target):
    return mass * (x - target)


@qd.func
def inertia_block(mass: float):
    """Hessian block m I of the inertia of a vertex."""
    return mass * qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def step_displacement(x, x_start):
    """Displacement of an unknown over the substep, from its position at the substep start."""
    return x - x_start


@qd.func
def has_converged(max_displacement: float, displacement_tol: float):
    return max_displacement < displacement_tol


@qd.func
def is_trial_accepted(energy_trial: float, energy_start: float):
    """Whether a line-search trial lowers the energy, or keeps it."""
    return energy_trial <= energy_start


@qd.func
def backtracked_step(alpha: float, shrink: float):
    return shrink * alpha


# ================================================== Layer 1: stages ==================================================


@qd.kernel
def kernel_reset_newton_control(control: NewtonControl):
    """Start the solve of every environment but those stopped on a non-finite measure earlier in the step."""
    for _ in range(1):
        control.is_newton_running[()] = 0
    for i_b in range(control.failure_code.shape[0]):
        control.n_iterations[i_b] = 0
        control.failure_code[i_b] = 0
        control.is_converged[i_b] = 0
        control.is_env_running[i_b] = 1 - control.has_failed_non_finite[i_b]
        if control.is_env_running[i_b] != 0:
            qd.atomic_max(control.is_newton_running[()], 1)


@qd.kernel
def kernel_clear_non_finite(control: NewtonControl):
    """Clear the latch of the solves stopped on a non-finite measure, once the participants read it."""
    for i_b in range(control.has_failed_non_finite.shape[0]):
        control.has_failed_non_finite[i_b] = 0


@qd.func(requires_top_level=True)
def func_reset_newton_measures(control: NewtonControl):
    """Clear the largest displacement update, which the participants accumulate with their Newton direction."""
    for i_b in range(control.max_displacement.shape[0]):
        control.max_displacement[i_b] = 0.0


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
    for i_b in range(control.failure_code.shape[0]):
        if control.is_env_running[i_b] != 0:
            control.is_converged[i_b] = gs.qd_int(
                has_converged(control.max_displacement[i_b], control.displacement_tol[()])
            )
            is_finite = qd_isfinite(control.max_displacement[i_b]) and qd_isfinite(control.energy_start[i_b])
            if qd.static(qd_debug_asserts()):
                assert is_finite, (
                    "Newton solve: the direction or the energy of environment %d is not finite at iteration %d"
                    % (i_b, control.n_iterations[i_b])
                )
            if not is_finite:
                control.failure_code[i_b] = NEWTON_FAILED_NON_FINITE


@qd.func(requires_top_level=True)
def func_init_line_search(control: NewtonControl):
    for _ in range(1):
        control.is_line_search_running[()] = 0
    for i_b in range(control.failure_code.shape[0]):
        control.alpha[i_b] = 1.0
        control.n_line_search_iterations[i_b] = 0
        control.is_env_line_search_running[i_b] = control.is_env_running[i_b]
        if control.is_env_running[i_b] != 0:
            qd.atomic_max(control.is_line_search_running[()], 1)


@qd.func(requires_top_level=True)
def func_accept_or_backtrack(control: NewtonControl):
    """Accept the trial step of every environment once its energy is at most the start energy, or shrink its alpha."""
    for _ in range(1):
        control.is_line_search_running[()] = 0
    for i_b in range(control.failure_code.shape[0]):
        if control.is_env_line_search_running[i_b] != 0:
            control.n_line_search_iterations[i_b] = control.n_line_search_iterations[i_b] + 1
            is_accepted = (
                control.is_converged[i_b] != 0
                or control.failure_code[i_b] != 0
                or is_trial_accepted(control.energy_trial[i_b], control.energy_start[i_b])
            )
            is_exhausted = control.n_line_search_iterations[i_b] >= control.line_search_max_iterations[()]
            if is_accepted or (is_exhausted and control.is_strict[()] == 0):
                control.is_env_line_search_running[i_b] = 0
                control.energy_start[i_b] = control.energy_trial[i_b]
            elif is_exhausted:
                control.is_env_line_search_running[i_b] = 0
                control.failure_code[i_b] = NEWTON_FAILED_LINE_SEARCH
            else:
                control.alpha[i_b] = backtracked_step(control.alpha[i_b], control.line_search_shrink[()])
                qd.atomic_max(control.is_line_search_running[()], 1)


@qd.func(requires_top_level=True)
def func_update_convergence(control: NewtonControl):
    """Stop every environment after a converged, a non-finite or a failed line-search iteration, and raise the failure
    flag at the iteration cap."""
    for _ in range(1):
        control.is_newton_running[()] = 0
    for i_b in range(control.failure_code.shape[0]):
        if control.is_env_running[i_b] != 0:
            control.n_iterations[i_b] = control.n_iterations[i_b] + 1
            if control.failure_code[i_b] == NEWTON_FAILED_NON_FINITE:
                control.is_env_running[i_b] = 0
                control.has_failed_non_finite[i_b] = 1
            elif control.failure_code[i_b] == NEWTON_FAILED_LINE_SEARCH or control.is_converged[i_b] != 0:
                control.is_env_running[i_b] = 0
            elif control.n_iterations[i_b] >= control.max_iterations[()]:
                control.is_env_running[i_b] = 0
                control.failure_code[i_b] = NEWTON_FAILED_ITERATIONS
                qd.atomic_max(control.has_failed_iterations[()], 1)
            else:
                qd.atomic_max(control.is_newton_running[()], 1)
