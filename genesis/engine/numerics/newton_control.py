import numpy as np
import quadrants as qd


@qd.data_oriented
class NewtonControl:
    """Device-resident scalars steering a Newton solve with a backtracking line search inside one graph.

    The participants of the solve write the measures (energy at the start point and at the trial point and the largest
    displacement update); the phases below turn them into the loop conditions.
    """

    def __init__(self, value_type) -> None:
        self.value_type = value_type
        np_float = np.float64 if value_type == qd.f64 else np.float32

        def scalar(dtype, value, np_dtype):
            result = qd.ndarray(dtype, shape=())
            result.from_numpy(np.array(value, dtype=np_dtype))
            return result

        self.newton_cond = scalar(qd.i32, 0, np.int32)
        self.ls_cond = scalar(qd.i32, 0, np.int32)
        self.converged = scalar(qd.i32, 0, np.int32)
        self.failed = scalar(qd.i32, 0, np.int32)
        self.newton_iter = scalar(qd.i32, 0, np.int32)
        self.ls_iter = scalar(qd.i32, 0, np.int32)
        self.max_newton_iter = scalar(qd.i32, 1024, np.int32)
        self.max_ls_iter = scalar(qd.i32, 12, np.int32)
        self.disp_tol = scalar(value_type, 0.0, np_float)
        self.alpha = scalar(value_type, 1.0, np_float)
        self.energy_start = scalar(value_type, 0.0, np_float)
        self.energy_trial = scalar(value_type, 0.0, np_float)
        self.max_disp = scalar(value_type, 0.0, np_float)


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
    """Judge convergence on the Newton direction itself, before the line search: the largest displacement update
    under disp_tol."""
    for _ in range(1):
        control.converged[()] = qd.i32(control.max_disp[()] < control.disp_tol[()])


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
    """Accept the trial step once it does not raise the energy, or halve alpha for the next trial.

    A converged direction is accepted at its first trial without the energy test. Exhausting the trials keeps the
    last one. The accepted trial energy is the start energy of the next Newton iteration, so the participants
    evaluate the start energy only once per solve, before the first iteration.
    """
    for _ in range(1):
        control.ls_iter[()] += 1
        is_done = (
            control.converged[()] != 0
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
    """Stop after a converged iteration; reaching the iteration cap raises the failure flag."""
    for _ in range(1):
        control.newton_iter[()] += 1
        if control.converged[()] != 0:
            control.newton_cond[()] = 0
        elif control.newton_iter[()] >= control.max_newton_iter[()]:
            control.newton_cond[()] = 0
            control.failed[()] = 1
