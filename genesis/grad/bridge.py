"""
Torch autograd graph of a differentiable scene.

While torch records gradients, every operation on the state of a differentiable scene (a step, a setter, a state query)
is one node of the torch autograd graph. The adjoint of the state lives in the `.grad` fields of the solvers. Each node
saves what its backward needs, so a backward pass reaching part of a rollout runs the backward of that part alone.

The nodes are ordered by the state handle, a zero-dim tensor standing for the state of the scene: each operation takes
the current one and outputs the next, so autograd runs their backward passes in the reverse order of the operations.
Without gradient recording, a change of the state bumps the version of the current state handle instead, and a change
of the model or a reset bumps the version of the model handle every node saved, so a later backward through a node
they invalidate raises. Both handles live on `gs.device`: autograd runs a node on the worker thread of the device of
the gradients it receives, which is then the thread issuing the torch operations around it.
"""

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.autograd.function import once_differentiable
from torch.autograd.graph import increment_version

import genesis as gs

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver, StateChange


class Operation:
    """
    An operation on the state of a differentiable scene, recorded as one node of its autograd graph.

    `forward` runs the operation on the inputs of the node, which track no gradient inside it, and returns its outputs.
    `backward` receives the gradients of the outputs, None for those the loss does not reach, and returns those of the
    inputs, in order, the adjoint of the state flowing through the `.grad` fields of the solvers.
    """

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def backward(self, *outputs_grad: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        raise NotImplementedError


class OperationNode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph: "RolloutGraph", op: Operation, state_handle, model_handle, *inputs: torch.Tensor):
        # The simulator owns the graph, so a strong reference would make a cycle that crosses the C++ autograd graph,
        # which the garbage collector cannot traverse: the scene would never be freed.
        ctx.graph = weakref.ref(graph)
        ctx.op = op
        ctx.set_materialize_grads(False)
        outputs = graph._run(op.forward, *inputs)
        next_state_handle = state_handle.detach().clone()
        ctx.save_for_backward(next_state_handle, model_handle)
        return next_state_handle, *outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, state_handle_grad: torch.Tensor | None, *outputs_grad: torch.Tensor | None):
        graph = ctx.graph()
        if graph is None:
            gs.raise_exception("Calling backward on a scene that has been destroyed since the rollout.")
        try:
            _ = ctx.saved_tensors
        except RuntimeError as e:
            gs.raise_exception_from(
                "The scene changed since this operation was recorded: its state was set while torch recorded no "
                "gradient, its model was modified, or it was reset. Call backward before any of them.",
                e,
            )
        graph._enter_pass()
        inputs_grad = graph._run(ctx.op.backward, *outputs_grad)
        # The previous node runs its backward only if it receives a gradient for the state handle it returned
        if state_handle_grad is None:
            state_handle_grad = torch.zeros((), dtype=gs.tc_float, device=gs.device)
        return None, None, state_handle_grad, None, *inputs_grad


class RolloutGraph:
    """
    The autograd graph of the rollout of a differentiable scene.

    It holds the state handle the last recorded operation returned and the model handle every operation of the rollout
    saved, and bounds the backward passes through them: the first node a pass reaches calls `on_pass_start`, and
    `on_pass_end` runs once the pass is over.
    """

    def __init__(self, requires_grad: bool, on_pass_start: Callable[[], None], on_pass_end: Callable[[], None]):
        self._requires_grad = requires_grad
        self._on_pass_start = on_pass_start
        self._on_pass_end = on_pass_end
        self._state_handle: torch.Tensor | None = None
        self._model_handle: torch.Tensor | None = None
        # The backward pass running, identified by the graph task of the autograd engine executing it
        self._pass_id = -1
        self._is_running_operation = False
        self.restart()

    def restart(self):
        """Begin a new rollout, invalidating every operation of the previous one."""
        if self._requires_grad:
            if self._model_handle is not None:
                increment_version(self._model_handle)
            self._state_handle = torch.zeros((), dtype=gs.tc_float, device=gs.device, requires_grad=True)
            self._model_handle = torch.zeros((), dtype=gs.tc_float, device=gs.device)

    @property
    def is_recording(self) -> bool:
        """
        Whether the operations on the state are recorded.

        A differentiable scene records them while torch records gradients, except those an operation being recorded
        runs itself, which it reverses as part of its own.
        """
        return self._requires_grad and torch.is_grad_enabled() and not self._is_running_operation

    def record(self, op: Operation, *inputs: torch.Tensor) -> list[torch.Tensor]:
        """Run an operation on the state as one node of the graph, and return its outputs."""
        state_handle, *outputs = OperationNode.apply(self, op, self._state_handle, self._model_handle, *inputs)
        self._state_handle = state_handle
        return outputs

    def mark_state_changed(self):
        """Invalidate the last operation recorded, whose output a change of the state that is not recorded overwrote."""
        if self._requires_grad and not self._is_running_operation:
            increment_version(self._state_handle)

    def invalidate(self, change: "StateChange | None" = None, envs_idx=None):
        """
        Invalidate every operation of the rollout, for a change of the scene the backward pass of a step cannot replay.

        A change of the model alters the replay of every recorded step, and a state set whole replaces the one they
        lead to. The arguments are those of a solver Subscriber callback, which this method is for the changes of the
        model (see 'Simulator.build').
        """
        if self._requires_grad and not self._is_running_operation:
            increment_version(self._model_handle)

    def _enter_pass(self):
        """Start a backward pass from zero adjoints the first time one of its nodes runs."""
        # FIXME: private torch API, the same graph task identity 'register_multi_grad_hook' relies on.
        pass_id = torch._C._current_graph_task_id()
        if pass_id != self._pass_id:
            self._pass_id = pass_id
            self._on_pass_start()
            # FIXME: 'torch.autograd.graph.queue_callback', the public alias of this call, once the minimum supported
            # torch has it.
            torch.autograd.Variable._execution_engine.queue_callback(self._on_pass_end)

    def _run(self, op_method, *args):
        """Run the forward or backward pass of a recorded operation, inside which no operation is recorded."""
        self._is_running_operation = True
        try:
            return op_method(*args)
        finally:
            self._is_running_operation = False


class KernelLaunch(NamedTuple):
    """
    One launch of a kernel writing state, with its arguments in order.

    The values a setter writes are torch tensors among the arguments. An `InputOperation` runs the launch forward with
    them detached, and backward through the autodiff of the kernel, which gives the gradient of each value and clears
    the adjoint of the slots it overwrote.
    """

    kernel: Callable
    args: tuple


class InputOperation(Operation):
    """
    A setter of the state, or any input given to the simulation between two steps, as one operation of the graph.

    Its inputs are the values among the arguments of its launches that track gradients. A launch whose kernel reads
    quantities derived from the state, such as a wrench applied at a point of a link, runs backward at the state its
    solver saved (`saved_state`), which folds the adjoint it leaves on derived quantities afterwards.
    """

    def __init__(self, solver: "Solver", launches: tuple[KernelLaunch, ...], saved_state=None):
        # See OperationNode for the weak reference
        self._solver = weakref.ref(solver)
        self._launches = launches
        self._saved_state = saved_state
        self.inputs = tuple(
            {
                id(arg): arg
                for launch in launches
                for arg in launch.args
                if isinstance(arg, torch.Tensor) and arg.requires_grad
            }.values()
        )

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        for launch in self._launches:
            launch.kernel(*(arg.detach() if isinstance(arg, torch.Tensor) else arg for arg in launch.args))
        return ()

    def backward(self) -> tuple[torch.Tensor | None, ...]:
        solver = self._solver()
        if self._saved_state is not None:
            solver.restore_saved_state(self._saved_state)
        # Each value gets a fresh copy tracking gradients, into which the autodiff of the kernels accumulates
        values_grad = {id(value): value.detach().clone().requires_grad_() for value in self.inputs}
        for launch in reversed(self._launches):
            launch.kernel.grad(
                *(values_grad.get(id(arg), arg) if isinstance(arg, torch.Tensor) else arg for arg in launch.args)
            )
        if self._saved_state is not None:
            solver.fold_derived_grad(self._saved_state)
        return tuple(values_grad[id(value)].grad for value in self.inputs)


class ReadOperation(Operation):
    """
    A query of the state, whose outputs are its tensors tracking gradients.

    The backward pass hands their gradients, None for those the loss does not reach, to the `add_grad_from_state` of
    the solver that read them, with the selection of the state they cover, which only the solver interprets, and the
    state it saved for tensors of quantities derived from it, None otherwise. A state is always queried between two
    steps, where its solver holds it in frame 0, which is where the gradient goes.
    """

    def __init__(self, solver: "Solver", tensors: tuple[torch.Tensor, ...], selection, saved_state=None):
        # See OperationNode for the weak reference
        self._solver = weakref.ref(solver)
        self._tensors = tensors
        self._selection = selection
        self._saved_state = saved_state

    def forward(self) -> tuple[torch.Tensor, ...]:
        tensors, self._tensors = self._tensors, None
        return tensors

    def backward(self, *outputs_grad: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        if all(grad is None for grad in outputs_grad):
            return ()
        grads = tuple(None if grad is None else grad.contiguous() for grad in outputs_grad)
        self._solver().add_grad_from_state(grads, self._selection, self._saved_state)
        return ()
