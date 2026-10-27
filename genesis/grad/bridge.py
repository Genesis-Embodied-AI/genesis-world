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

import contextlib
import dataclasses
import warnings
import weakref
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.autograd.function import once_differentiable
from torch.autograd.graph import increment_version

from quadrants.lang import impl

import genesis as gs
from genesis.grad.kernel_writes import get_written_paths
from genesis.utils.array_class import iter_state_arrays
from genesis.utils.cache import OwnedCache
from genesis.utils.misc import qd_from_torch, qd_to_torch

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
    def forward(ctx, graph: "RolloutGraph", operation: Operation, state_handle, model_handle, *inputs: torch.Tensor):
        # The simulator owns the graph, so a strong reference would make a cycle that crosses the C++ autograd graph,
        # which the garbage collector cannot traverse: the scene would never be freed.
        ctx.graph = weakref.ref(graph)
        ctx.operation = operation
        ctx.set_materialize_grads(False)
        outputs = graph._run(operation.forward, *inputs)
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
        inputs_grad = graph._run(ctx.operation.backward, *outputs_grad)
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

    def record(self, operation: Operation, *inputs: torch.Tensor) -> list[torch.Tensor]:
        """Run an operation on the state as one node of the graph, and return its outputs."""
        state_handle, *outputs = OperationNode.apply(self, operation, self._state_handle, self._model_handle, *inputs)
        self._state_handle = state_handle
        return outputs

    def mark_state_changed(self):
        """Invalidate the last operation recorded, for a change of the state torch records no gradient of."""
        if self._requires_grad and not self._is_running_operation:
            increment_version(self._state_handle)

    def invalidate(self, change: "StateChange | None" = None, envs_idx=None):
        """
        Invalidate every operation of the rollout, for a change of the model, which alters the replay of every step.

        The simulator subscribes this method to the changes of the model of the differentiable solvers (see
        'Simulator.build'), whose Subscriber callbacks take these arguments.
        """
        if self._requires_grad and not self._is_running_operation:
            increment_version(self._model_handle)

    def _enter_pass(self):
        """Start a backward pass from zero adjoints the first time one of its nodes runs."""
        # The identity of the graph task running the pass, private torch API on which 'register_multi_grad_hook' relies
        pass_id = torch._C._current_graph_task_id()
        if pass_id != self._pass_id:
            self._pass_id = pass_id
            self._on_pass_start()
            # FIXME: torch - 'torch.autograd.graph.queue_callback', the public alias of this call, is missing from the
            # releases of torch Genesis supports.
            torch.autograd.Variable._execution_engine.queue_callback(self._on_pass_end)

    def _run(self, operation_method, *args):
        """Run the forward or backward pass of a recorded operation, inside which no operation is recorded."""
        self._is_running_operation = True
        try:
            return operation_method(*args)
        finally:
            self._is_running_operation = False


class LaunchRecorder:
    """
    Record the kernels launched in its context, each with its arguments, in the order of the launches.

    It stands for the autodiff tape of Quadrants, whose kernels hand it every forward launch before running it.
    """

    def __init__(self):
        self.launches: list[tuple[Callable, tuple]] = []
        self._tensors_without_grad: list[torch.Tensor] = []
        self._previous_tape = None
        self._warnings_context = warnings.catch_warnings()

    def __enter__(self) -> "LaunchRecorder":
        # FIXME: quadrants - a forward launch reads and allocates the gradient of a torch tensor tracking gradients. The
        # warning of torch is silenced, and the allocated gradients are taken back in '__exit__'.
        self._warnings_context.__enter__()
        warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor that is not a leaf Tensor")
        runtime = impl.get_runtime()
        self._previous_tape, runtime.target_tape = runtime.target_tape, self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        impl.get_runtime().target_tape = self._previous_tape
        for tensor in self._tensors_without_grad:
            tensor.grad = None
        self._warnings_context.__exit__(exc_type, exc_value, traceback)

    def insert(self, kernel: Callable, args: tuple) -> None:
        """Record the launch of a kernel, before it runs (the hook of the autodiff tape of Quadrants)."""
        self.launches.append((kernel, args))
        self._tensors_without_grad += [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.requires_grad and arg.grad is None
        ]


# The arrays of the state each launch uses and those it may write, by compiled kernel and struct arguments, each entry
# living as long as the structs
_LAUNCH_ARRAYS_CACHE = OwnedCache()


class StepRecorder(LaunchRecorder):
    """
    Record the kernels launched in its context as entries it can reverse.

    Every launch is one entry, reversed by the autodiff of its kernel, unless it runs in a block opened by 'entry',
    whose launches form one entry. An entry is reversed at the values it left, as the autodiff of Quadrants expects,
    then the arrays it overwrote are restored, so that every entry before it is reversed at the values it left too.
    Before each launch, the recorder saves the arrays of the state the launch writes that an earlier entry uses: the
    writes come from the source of the kernel (see 'get_written_paths'), narrowed to the struct members its compiled
    form uses, and the uses of the earlier entries from their compiled form. A backward pass follows the convention of
    autodiff, consuming the adjoint of the values a launch wrote and clearing it.
    """

    def __init__(self):
        super().__init__()
        self.entries: list[RecordedEntry] = []
        self._is_grouping = False
        # The arrays the closed entries use, and those the open one does, by identity
        self._arrays_used: set[int] = set()
        self._arrays_used_by_entry: set[int] = set()

    def insert(self, kernel: Callable, args: tuple) -> None:
        if not self._is_grouping:
            self._arrays_used |= self._arrays_used_by_entry
            self._arrays_used_by_entry.clear()
            self.entries.append(RecordedEntry([], True, None, (), [], None))
        launches, _, _, _, arrays_saved, _ = self.entries[-1]
        launches.append((kernel, args))
        # The arrays of the state the launch uses and those it may write are those its struct arguments hold, INFO and
        # CONSTANT apart, which a step never writes (see 'DataKind'). They depend on the compiled form of the kernel and
        # on the structs alone.
        key = kernel.ensure_compiled(*args)
        structs = [arg for arg in args if dataclasses.is_dataclass(arg)]
        launch_key = (key, *(id(struct) for struct in structs))
        launch_arrays = _LAUNCH_ARRAYS_CACHE.find(launch_key)
        if launch_arrays is None:
            flat_names_used = kernel.used_py_dataclass_parameters_by_key_enforcing.get(key)
            paths_written = get_written_paths(kernel.func, args[0] if kernel.is_classkernel else None)
            arrays_used, arrays_written = set(), []
            for arg_meta, arg in zip(kernel.arg_metas, args):
                if dataclasses.is_dataclass(arg):
                    for path, array in iter_state_arrays(arg, (arg_meta.name,)):
                        if flat_names_used is None or "__qd_" + "__qd_".join(path) in flat_names_used:
                            arrays_used.add(id(array))
                            if path in paths_written:
                                arrays_written.append(array)
            launch_arrays = (arrays_used, arrays_written)
            for struct in structs:
                _LAUNCH_ARRAYS_CACHE.own(struct, launch_key, launch_arrays)
        arrays_used, arrays_written = launch_arrays
        for array in arrays_written:
            if id(array) in self._arrays_used and all(array is not saved for saved, _ in arrays_saved):
                arrays_saved.append((array, qd_to_torch(array, copy=True)))
        self._arrays_used_by_entry |= arrays_used

    @contextlib.contextmanager
    def entry(
        self,
        backward: Callable[..., None] | None = None,
        *args,
        restore: Callable[[], None] | None = None,
        is_autodiff: bool = False,
    ) -> Iterator[None]:
        """
        Record the kernels launched in the block as one entry.

        The launches are reversed by their autodiff when `is_autodiff` holds, by `backward` called with `args` otherwise,
        None for launches without gradient. `restore` writes back, after the reversal, what the block overwrites in
        arrays the recorder cannot see, such as the frames of a solver.
        """
        assert not self._is_grouping, "A block recording one entry cannot open inside another."
        self._arrays_used |= self._arrays_used_by_entry
        self._arrays_used_by_entry.clear()
        self.entries.append(RecordedEntry([], is_autodiff, backward, args, [], restore))
        self._is_grouping = True
        try:
            yield
        finally:
            self._is_grouping = False

    def reverse(self) -> None:
        """Run the backward pass of the entries in reverse order, which leaves the arrays as they were before them."""
        for launches, is_autodiff, backward, backward_args, arrays_saved, restore in reversed(self.entries):
            if is_autodiff:
                for kernel, args in reversed(launches):
                    kernel.grad(*args)
            elif backward is not None:
                backward(*backward_args)
            for array, value in arrays_saved:
                qd_from_torch(array, value)
            if restore is not None:
                restore()
            if arrays_saved and gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()


class RecordedEntry(NamedTuple):
    """
    An entry of a recorded step: its launches, whether autodiff reverses them, the backward reversing them otherwise
    (None without gradient) with its arguments, the arrays it overwrites with their values before it, and the function
    restoring what it overwrites that the recorder cannot save.
    """

    launches: list[tuple[Callable, tuple]]
    is_autodiff: bool
    backward: Callable[..., None] | None
    backward_args: tuple
    arrays_saved: list[tuple]
    restore: Callable[[], None] | None


@contextlib.contextmanager
def custom_backward(backward: Callable[..., None] | None, *args) -> Iterator[None]:
    """
    Record the kernels launched in the block as one entry of the step being recorded, reversed by the given backward.

    `backward` is called with `args`, and is None for launches without gradient. It does nothing outside of the
    recording of a step (see StepRecorder).
    """
    recorder = impl.get_runtime().target_tape
    if isinstance(recorder, StepRecorder):
        with recorder.entry(backward, *args):
            yield
    else:
        yield


def is_recording_step() -> bool:
    """
    Whether the kernels being launched are recorded for the backward pass of a step (see StepRecorder).

    The recorder sees kernel launches alone, so a step writes its state through kernels while it is recorded, never
    through a zero-copy view.
    """
    return isinstance(impl.get_runtime().target_tape, StepRecorder)


class InputOperation(Operation):
    """
    A setter of the state, or any input given to the simulation between two steps, as one operation of the graph.

    The kernels of the setter ran as it was recorded (see 'LaunchRecorder'), so the forward pass does nothing. Its
    inputs are the torch tensors among the arguments of the launches that track gradients. The backward pass runs the
    autodiff of the kernels in reverse order, which gives the gradient of each value and clears the adjoint of the slots
    it overwrote. Launches whose kernels read quantities derived from the state, such as a wrench applied at a point of
    a link, run backward at the state their solver saved (`saved_state`), and the solver folds the adjoint they leave on
    derived quantities into the state afterwards.
    """

    def __init__(self, solver: "Solver", launches: list[tuple[Callable, tuple]], saved_state=None):
        # See OperationNode for the weak reference
        self._solver = weakref.ref(solver)
        self._launches = launches
        self._saved_state = saved_state
        self.inputs = tuple(
            {
                id(arg): arg
                for _, args in launches
                for arg in args
                if isinstance(arg, torch.Tensor) and arg.requires_grad
            }.values()
        )

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return ()

    def backward(self) -> tuple[torch.Tensor | None, ...]:
        solver = self._solver()
        if self._saved_state is not None:
            solver.restore_saved_state(self._saved_state)
        # Each value gets a fresh copy tracking gradients, into which the autodiff of the kernels accumulates
        values_grad = {id(value): value.detach().clone().requires_grad_() for value in self.inputs}
        for kernel, args in reversed(self._launches):
            kernel.grad(*(values_grad.get(id(arg), arg) for arg in args))
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
        # A tensor a node returns without taking it as input becomes an output of the node itself, through which torch
        # routes the gradient of the loss. The node keeps no reference to it, which would outlive the rollout.
        tensors, self._tensors = self._tensors, None
        return tensors

    def backward(self, *outputs_grad: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        if all(grad is None for grad in outputs_grad):
            return ()
        grads = tuple(None if grad is None else grad.contiguous() for grad in outputs_grad)
        self._solver().add_grad_from_state(grads, self._selection, self._saved_state)
        return ()
