from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import ActionInvocation, SimAction, SimData, SimSystem


def _same_action_data(lhs: tuple[object, ...], rhs: tuple[object, ...]) -> bool:
    return len(lhs) == len(rhs) and all(left is right for left, right in zip(lhs, rhs))


class PCGSolver(SimSystem):
    """PCG interface and lifecycle-action owner."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible PCG configuration and diagnostics."""

        n_iterations: qd.Ndarray
        is_failed: qd.Ndarray
        preconditioned_residual: qd.Ndarray
        n_block_rows: qd.Ndarray
        pcg_tol_rate: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data: PCGSolver.Data | None = None
        self.is_initialized_host = False
        self.primary_operators = self.create_action_collection()
        self.operators = self.create_action_collection()
        self.reduced_primary_operators = self.create_action_collection()
        self.reduced_operators = self.create_action_collection()
        self.preconditioners = self.create_action_collection()
        self.reduced_preconditioners = self.create_action_collection()
        self.shared_preconditioners = self.create_action_collection()
        self.solver_initializers = self.create_action_collection()
        self.solvers = self.create_action_collection()

    def build(self) -> None:
        pass

    def register_solver_actions(
        self,
        initializer: SimAction,
        solver: SimAction,
    ) -> None:
        """Declare one complete concrete PCG implementation protocol.

        ``initializer`` initializes already allocated implementation state.
        ``solver`` executes one complete PCG solve through the operators and
        preconditioners selected by this interface.

        Both actions are mandatory and must have the same owning ``SimSystem``.
        Exactly one implementation protocol must exist when the engine is
        initialized; absence, duplication, or mixed ownership is a fatal engine
        construction error.
        """
        if initializer.owner is not solver.owner:
            raise RuntimeError("PCG solver actions must have exactly one owning SimSystem")
        if not _same_action_data(initializer.data, solver.data):
            raise RuntimeError("PCG solver initializer and solve action must share the same ordered SimData inputs")
        self.solver_initializers.register(initializer)
        self.solvers.register(solver)

    def register_primary_operator(self, action: SimAction) -> None:
        """Declare the complete primary operator for the standard PCG route.

        The action applies the assembled base matrix. Additional physical
        systems may accumulate operator contributions through
        ``register_system_actions``. Exactly one standard primary operator must
        exist when the engine is initialized; any other count is a fatal engine
        construction error.
        """
        self.primary_operators.register(action)

    def register_system_actions(
        self,
        operator: SimAction,
        reduced_operator: SimAction,
        preconditioner: SimAction,
    ) -> None:
        """Declare one physical system's complete PCG participation protocol.

        A participating system must provide its operator contribution in both
        the standard and reduced coordinate routes, together with the
        preconditioner contribution it owns in the standard route. The reduced
        route's primary operator and matching preconditioner are declared by
        ``register_reduced_system_actions``.

        All three actions must have the same owning ``SimSystem``. A missing
        action or mixed ownership is a fatal engine construction error; partial
        system participation is not a supported state.
        """
        owners = {
            operator.owner,
            reduced_operator.owner,
            preconditioner.owner,
        }
        if len(owners) != 1:
            raise RuntimeError("PCG system actions must have exactly one owning SimSystem")
        self.operators.register(operator)
        self.reduced_operators.register(reduced_operator)
        self.preconditioners.register(preconditioner)

    def register_reduced_system_actions(
        self,
        primary_operator: SimAction,
        preconditioner: SimAction,
    ) -> None:
        """Declare the complete reduced-coordinate PCG route protocol.

        ``primary_operator`` replaces the standard primary operator and applies
        the assembled matrix through the reduced coordinate mapping.
        ``preconditioner`` must match that same reduced DOF layout. Reduced
        operator contributions declared by physical systems and route-invariant
        shared preconditioners are composed on top of this pair.

        Both actions must have the same owning ``SimSystem``. Omitting either
        action or mixing owners is a fatal engine construction error; there is
        no partially registered reduced route.
        """
        if primary_operator.owner is not preconditioner.owner:
            raise RuntimeError("Reduced PCG system actions must have exactly one owning SimSystem")
        self.reduced_primary_operators.register(primary_operator)
        self.reduced_preconditioners.register(preconditioner)

    def register_shared_preconditioner(self, action: SimAction) -> None:
        """Declare one complete route-invariant preconditioner contribution.

        The action must apply the preconditioner for the DOFs owned by its
        system in both standard and reduced layouts. It is composed with the
        preconditioner selected for either route. Missing preconditioning for a
        selected route is a fatal engine initialization error.
        """
        self.shared_preconditioners.register(action)

    def _resolve_actions(
        self,
    ) -> tuple[
        SimAction,
        tuple[ActionInvocation, ...],
        tuple[ActionInvocation, ...],
        tuple[ActionInvocation, ...],
    ]:
        solver_initializers = self.solver_initializers.actions
        solvers = self.solvers.actions
        primary_operators = self.primary_operators.actions
        operators = self.operators.actions
        reduced_primary_operators = self.reduced_primary_operators.actions
        reduced_operators = self.reduced_operators.actions
        preconditioners = self.preconditioners.actions
        reduced_preconditioners = self.reduced_preconditioners.actions
        shared_preconditioners = self.shared_preconditioners.actions
        if len(solver_initializers) != 1 or len(solvers) != 1:
            raise RuntimeError(
                "PCGSolver requires exactly one complete solver protocol, "
                f"got {len(solver_initializers)} initializers and {len(solvers)} solvers"
            )
        if solver_initializers[0].owner is not solvers[0].owner:
            raise RuntimeError("PCG solver initializer and solve action must have the same owner")
        if len(primary_operators) != 1:
            raise RuntimeError(f"PCGSolver requires exactly one primary operator, got {len(primary_operators)}")
        if len(reduced_primary_operators) > 1:
            raise RuntimeError("PCGSolver accepts at most one reduced primary operator")
        if not preconditioners and not shared_preconditioners:
            raise RuntimeError("PCGSolver requires at least one preconditioner contributor")
        if not reduced_primary_operators:
            operator_actions = (
                primary_operators[0].invocation,
                *(action.invocation for action in operators),
            )
            selected_preconditioners = preconditioners
        else:
            operator_actions = (
                reduced_primary_operators[0].invocation,
                *(action.invocation for action in reduced_operators),
            )
            selected_preconditioners = reduced_preconditioners
        preconditioner_actions = (
            *(action.invocation for action in selected_preconditioners),
            *(action.invocation for action in shared_preconditioners),
        )
        if not preconditioner_actions:
            raise RuntimeError("Selected PCG path has no preconditioner actions")
        return (
            solver_initializers[0],
            (solvers[0].invocation,),
            operator_actions,
            preconditioner_actions,
        )

    def init(self, total_dof: int, n_block_rows: int, pcg_tol_rate: float) -> None:
        if self.is_initialized_host:
            raise RuntimeError("PCGSolver is already initialized")
        if min(total_dof, n_block_rows) < 0:
            raise ValueError("PCGSolver dimensions must be non-negative")
        (
            solver_initializer,
            solver_invocations,
            operator_actions,
            preconditioner_actions,
        ) = self._resolve_actions()
        if len(solver_initializer.data) != 1:
            raise RuntimeError("The selected PCG implementation requires exactly one primary Data input")
        solver_data = solver_initializer.data[0]
        if solver_data.dof_capacity != total_dof:
            raise RuntimeError(
                f"PCG implementation capacity {solver_data.dof_capacity} does not match total DOFs {total_dof}"
            )
        self.solver_invocations = solver_invocations
        self.operator_actions = operator_actions
        self.preconditioner_actions = preconditioner_actions
        self.data = get_pcg_solver_data(
            n_iterations=solver_data.n_iterations,
            is_failed=solver_data.is_failed,
            preconditioned_residual=solver_data.preconditioned_residual,
            n_block_rows=n_block_rows,
            pcg_tol_rate=pcg_tol_rate,
        )
        self.is_initialized_host = True


def get_pcg_solver_data(
    *,
    n_iterations,
    is_failed,
    preconditioned_residual,
    n_block_rows: int,
    pcg_tol_rate: float,
) -> PCGSolver.Data:
    n_block_rows_data = qd.ndarray(qd.i32, shape=())
    pcg_tol_rate_data = qd.ndarray(qd.f64, shape=())
    n_block_rows_data.from_numpy(np.array(n_block_rows, dtype=np.int32))
    pcg_tol_rate_data.from_numpy(np.array(pcg_tol_rate, dtype=np.float64))
    data = PCGSolver.Data()
    data.n_iterations = n_iterations
    data.is_failed = is_failed
    data.preconditioned_residual = preconditioned_residual
    data.n_block_rows = n_block_rows_data
    data.pcg_tol_rate = pcg_tol_rate_data
    return data


@qd.func(requires_top_level=True)
def solve_pcg(
    data: qd.template(),
    linear_system_data: qd.template(),
    action_provider: qd.template(),
    max_iterations,
    max_pcg_iterations: qd.template(),
    total_pcg_iterations: qd.template(),
):
    for solver_invocation in qd.static(action_provider.pcg_solver_invocations):
        solver_invocation.kernel(
            *(
                solver_invocation.data
                + (
                    linear_system_data,
                    action_provider,
                    data.pcg_tol_rate[()],
                    max_iterations,
                    max_pcg_iterations,
                    total_pcg_iterations,
                )
            )
        )
