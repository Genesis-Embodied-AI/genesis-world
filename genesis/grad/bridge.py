"""
Torch autograd graph of a differentiable scene.

While torch records gradients, every operation on the state of a differentiable scene is one node of the torch autograd
graph. The adjoint of the state lives in the `.grad` fields of the solvers. Each node saves what its backward needs, so
a backward pass reaching part of a rollout runs the backward of that part alone.

A step saves the state it starts from, and its backward pass replays it from there while recording the kernels it
launches, then reverses them (see LaunchRecorder). The calls of the methods writing the state between two steps (see
'mutates') record the kernels they launch as they run, which their backward pass reverses (see 'RolloutGraph.run_call').
The queries of the state are nodes whose outputs track gradients, which the backward pass adds to the adjoint of what
they read.

The nodes are ordered by the state handle, a zero-dim tensor standing for the state of the scene: each operation takes
the current one and outputs the next, so autograd runs their backward passes in the reverse order of the operations.
Without gradient recording, a change of the state bumps the version of the current state handle instead, and a change
of the model or a reset bumps the version of the model handle every node saved, so a later backward through a node
they invalidate raises. Both handles live on `gs.device`: autograd runs a node on the worker thread of the device of
the gradients it receives, which is then the thread issuing the torch operations around it.
"""

import contextlib
import dataclasses
import inspect
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
    from genesis.engine.solvers.base_solver import Solver
    from genesis.engine.state_change import StateChange


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
        # The recorder of the calls since the last step, whether one of them is running, and the first entry of the
        # segment of it being recorded (see 'run_call')
        self._recorder: LaunchRecorder | None = None
        self._is_calling = False
        self._segment_start = 0
        self.restart()

    def restart(self):
        """Begin a new rollout, invalidating every operation of the previous one."""
        if self._requires_grad:
            if self._model_handle is not None:
                increment_version(self._model_handle)
            self._state_handle = torch.zeros((), dtype=gs.tc_float, device=gs.device, requires_grad=True)
            self._model_handle = torch.zeros((), dtype=gs.tc_float, device=gs.device)
            self._recorder = None

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

    def record_step(self, operation: Operation) -> None:
        """Run a step as one node of the graph, which starts the recording of the calls after it anew."""
        self._recorder = None
        self.record(operation)

    def run_call(self, owner, method: Callable, args: tuple, kwargs: dict, *, differentiable: bool = True):
        """
        Run a call of a method writing the state, recording the kernels it launches for the backward pass.

        The view path and the kernel path of an accessor are equivalent by contract, so the call takes its kernel path,
        which the recorder sees (see 'ZeroCopyMode'). One recorder spans the calls between two steps, so that the
        arrays a call overwrites that an earlier one used are saved (see LaunchRecorder). The queries of the state the
        call makes cut it into segments, each one node of the graph between the nodes of the queries, which takes the
        torch tensors its launches used as inputs. An input tracking gradients that is not `differentiable` is refused.
        A call made by another one is part of it.
        """
        if self._is_calling:
            return method(owner, *args, **kwargs)
        if not differentiable and any(
            isinstance(value, torch.Tensor) and value.requires_grad for value in (*args, *kwargs.values())
        ):
            gs.raise_exception("Gradients with respect to this input are not supported.")
        if self._recorder is None:
            self._recorder = LaunchRecorder()
        self._is_calling = True
        self._segment_start = len(self._recorder.entries)
        try:
            with gs.use_zerocopy.disabled(), self._recorder:
                result = method(owner, *args, **kwargs)
        finally:
            self._is_calling = False
        self._record_segment()
        return result

    def record_read(self, solver: "Solver", tensors: tuple[torch.Tensor, ...], selection) -> None:
        """Record a query of the state as one node of the graph, which ends the segment of the call making it."""
        if self._is_calling:
            self._record_segment()
        self.record(ReadOperation(solver, tensors, selection))

    def _record_segment(self) -> None:
        """Record the launches of the call running since its last query of the state as one node of the graph."""
        start, stop = self._segment_start, len(self._recorder.entries)
        operation = SegmentOperation(self._recorder, start, stop)
        self.record(operation, *operation.inputs)
        self._segment_start = stop

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


# The backward reversing the launches of a kernel, None for a kernel whose launches carry no gradient, by the Python
# function of the kernel (see 'no_grad' and 'register_custom_backward')
_KERNELS_BACKWARD: dict[Callable, Callable[..., None] | None] = {}


def no_grad(kernel: Callable) -> Callable:
    """
    Declare that the launches of a kernel carry no gradient, such as the reset of a warm start or of error flags.

    A recorded launch of it is reversed by nothing, its overwrites being restored all the same (see LaunchRecorder). The
    tag goes above the kernel decorator.
    """
    _KERNELS_BACKWARD[kernel.fn] = None
    return kernel


def register_custom_backward(backward: Callable[..., None]) -> Callable[[Callable], Callable]:
    """
    Declare the backward reversing the launches of a kernel, which autodiff cannot derive.

    The backward is called with the arguments of the launch, from the values it left (see LaunchRecorder). The tag goes
    above the kernel decorator.
    """

    def decorator(kernel: Callable) -> Callable:
        _KERNELS_BACKWARD[kernel.fn] = backward
        return kernel

    return decorator


def reversed_with(**overwrites: bool | int | float) -> Callable[[Callable], Callable]:
    """
    Declare that the launches of a kernel are reversed by its own autodiff, with some of their arguments overwritten.

    The arguments `overwrites` names select a variant of the kernel computing the same values from the same arrays,
    which autodiff can reverse where the variant the launch runs syncs the lanes of a block. The reverse reads the
    values the launch left (see LaunchRecorder), so the variant must write no array the launch leaves untouched. The
    values are native Python scalars, as static arguments take. The tag goes above the kernel decorator.
    """
    assert all(isinstance(value, (bool, int, float)) for value in overwrites.values())

    def decorator(kernel: Callable) -> Callable:
        names = list(inspect.signature(kernel.fn).parameters)
        overwrites_by_position = {names.index(name): value for name, value in overwrites.items()}

        def backward(*args):
            kernel.grad(*[overwrites_by_position.get(i, arg) for i, arg in enumerate(args)])

        _KERNELS_BACKWARD[kernel.fn] = backward
        return kernel

    return decorator


# The arrays of the state each launch uses and those it may write, by compiled kernel and struct arguments, each entry
# living as long as the structs
_LAUNCH_ARRAYS_CACHE = OwnedCache()


class LaunchRecorder:
    """
    Record the kernels launched in its context as entries it can reverse, a step or the calls between two steps.

    It stands for the autodiff tape of Quadrants, whose kernels hand it every forward launch before running it.
    Every launch is one entry, reversed by the autodiff of its kernel, unless it runs in a block opened by 'entry',
    whose launches form one entry. An entry is reversed at the values it left, as the autodiff of Quadrants expects,
    then the arrays it overwrote are restored, so that every entry before it is reversed at the values it left too.
    Before each launch, the recorder saves the arrays of the state the launch writes that an earlier entry uses: the
    writes come from the source of the kernel (see 'get_written_paths'), narrowed to the struct members its compiled
    form uses, and the uses of the earlier entries from their compiled form. A backward pass follows the convention of
    autodiff, consuming the adjoint of the values a launch wrote and clearing it.
    """

    def __init__(self):
        self.entries: list[RecordedEntry] = []
        self._previous_tape = None
        self._warnings_context: warnings.catch_warnings | None = None
        # The torch tensors the launches took that tracked gradients without holding any (see '__enter__')
        self._tensors_without_grad: list[torch.Tensor] = []
        self._is_grouping = False
        # Whether the recorder is saving arrays, whose copy is a kernel launch of its own without zero-copy views
        self._is_saving = False
        # The arrays the closed entries use, and those the open one does, by identity
        self._arrays_used: set[int] = set()
        self._arrays_used_by_entry: set[int] = set()

    def __enter__(self) -> "LaunchRecorder":
        # The recorder of the calls between two steps is entered once per call, which a warnings context allows once
        self._warnings_context = warnings.catch_warnings()
        self._warnings_context.__enter__()
        # FIXME: quadrants#997 - a forward launch allocates the gradient of a torch tensor tracking gradients, which
        # torch warns about for a tensor derived from another one. The warning is silenced, the gradients taken back.
        warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor that is not a leaf Tensor")
        # FIXME: quadrants#1000 - the launch hook of Quadrants is reachable only through this private runtime slot
        runtime = impl.get_runtime()
        self._previous_tape, runtime.target_tape = runtime.target_tape, self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        impl.get_runtime().target_tape = self._previous_tape
        for tensor in self._tensors_without_grad:
            tensor.grad = None
        self._tensors_without_grad.clear()
        self._warnings_context.__exit__(exc_type, exc_value, traceback)

    def insert(self, kernel: Callable, args: tuple) -> None:
        """Record the launch of a kernel, before it runs (the hook of the autodiff tape of Quadrants)."""
        if self._is_saving:
            return
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad and arg.grad is None:
                self._tensors_without_grad.append(arg)
        if not self._is_grouping:
            self._arrays_used |= self._arrays_used_by_entry
            self._arrays_used_by_entry.clear()
            if kernel.func in _KERNELS_BACKWARD:
                entry = RecordedEntry(
                    launches=[],
                    is_autodiff=False,
                    backward=_KERNELS_BACKWARD[kernel.func],
                    backward_args=args,
                    arrays_saved=[],
                    restore=None,
                )
            else:
                entry = RecordedEntry(
                    launches=[], is_autodiff=True, backward=None, backward_args=(), arrays_saved=[], restore=None
                )
            self.entries.append(entry)
        entry = self.entries[-1]
        entry.launches.append((kernel, args))
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
        self._is_saving = True
        for array in arrays_written:
            if id(array) in self._arrays_used and all([array is not saved for saved, _ in entry.arrays_saved]):
                entry.arrays_saved.append((array, qd_to_torch(array, copy=True)))
        self._is_saving = False
        self._arrays_used_by_entry |= arrays_used

    @contextlib.contextmanager
    def entry(
        self,
        backward: Callable[..., None] | None = None,
        *args,
        restore: Callable[[], None] | None = None,
        autodiff: bool = False,
    ) -> Iterator[None]:
        """
        Record the kernels launched in the block as one entry.

        The launches are reversed by their autodiff when `autodiff` holds, by `backward` called with `args` otherwise,
        None for launches without gradient. `restore` writes back, after the reversal, what the block overwrites in
        arrays the recorder cannot see, such as the frames of a solver.
        """
        assert not self._is_grouping, "A block recording one entry cannot open inside another."
        self._arrays_used |= self._arrays_used_by_entry
        self._arrays_used_by_entry.clear()
        self.entries.append(
            RecordedEntry(
                launches=[],
                is_autodiff=autodiff,
                backward=backward,
                backward_args=args,
                arrays_saved=[],
                restore=restore,
            )
        )
        self._is_grouping = True
        try:
            yield
        finally:
            self._is_grouping = False

    def reverse(self, start: int = 0, stop: int | None = None) -> dict[int, torch.Tensor]:
        """
        Run the backward pass of the entries in reverse order, which leaves the arrays as they were before them.

        `start` and `stop` bound the entries reversed, all of them by default. Returns the gradients the reversal gives
        the torch tensors tracking gradients the launches take, by identity of the tensor.
        """
        tensors_grad = {}
        # The zero-copy restores of consecutive entries flush once, before the next launch reading them
        is_sync_pending = False
        for launches, is_autodiff, backward, backward_args, arrays_saved, restore in reversed(self.entries[start:stop]):
            if is_sync_pending and (is_autodiff or backward is not None):
                torch.mps.synchronize()
                is_sync_pending = False
            if is_autodiff:
                for kernel, args in reversed(launches):
                    # A torch tensor tracking gradients is reversed through a leaf standing for it, which collects its
                    # gradient
                    leaves = {
                        id(arg): arg.detach().requires_grad_()
                        for arg in args
                        if isinstance(arg, torch.Tensor) and arg.requires_grad
                    }
                    kernel.grad(*(leaves.get(id(arg), arg) for arg in args))
                    for tensor_id, leaf in leaves.items():
                        if tensor_id in tensors_grad:
                            tensors_grad[tensor_id] = tensors_grad[tensor_id] + leaf.grad
                        else:
                            tensors_grad[tensor_id] = leaf.grad
            elif backward is not None:
                backward(*backward_args)
            for array, value in arrays_saved:
                qd_from_torch(array, value)
            if restore is not None:
                restore()
            is_sync_pending |= bool(arrays_saved) and gs.use_zerocopy.is_supported and gs.backend == gs.metal
        if is_sync_pending:
            torch.mps.synchronize()
        return tensors_grad


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
    recording of a step (see LaunchRecorder).
    """
    recorder = impl.get_runtime().target_tape
    if isinstance(recorder, LaunchRecorder):
        with recorder.entry(backward, *args):
            yield
    else:
        yield


def is_recording_launches() -> bool:
    """
    Whether the kernels being launched are recorded for the backward pass of a step (see LaunchRecorder).

    The recorder sees kernel launches alone, so a step writes its state through kernels while it is recorded, never
    through a zero-copy view.
    """
    return isinstance(impl.get_runtime().target_tape, LaunchRecorder)


class SegmentOperation(Operation):
    """
    The launches of a call between two of its queries of the state, as one operation of the graph (see 'run_call').

    The launches ran as they were recorded, so the forward pass does nothing. The inputs are the torch tensors tracking
    gradients the launches took, whose gradients the backward pass returns, torch carrying them back to the values of
    the call and to the queries the tensors derive from.
    """

    def __init__(self, recorder: LaunchRecorder, start: int, stop: int):
        self._recorder = recorder
        self._start = start
        self._stop = stop
        self.inputs = tuple(
            {
                id(arg): arg
                for entry in recorder.entries[start:stop]
                for _, args in entry.launches
                for arg in args
                if isinstance(arg, torch.Tensor) and arg.requires_grad
            }.values()
        )

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return ()

    def backward(self) -> tuple[torch.Tensor | None, ...]:
        tensors_grad = self._recorder.reverse(self._start, self._stop)
        return tuple([tensors_grad.get(id(tensor)) for tensor in self.inputs])


class ReadOperation(Operation):
    """
    A query of the state, whose outputs are its tensors tracking gradients.

    The backward pass hands their gradients, None for those the loss does not reach, to the `add_grad_from_state` of
    the solver that read them, with the selection of the state they cover, which only the solver interprets. A state is
    always queried between two steps, where its solver holds it in frame 0, which is where the gradient goes.
    """

    def __init__(self, solver: "Solver", tensors: tuple[torch.Tensor, ...], selection):
        # See OperationNode for the weak reference
        self._solver = weakref.proxy(solver)
        self._tensors = tensors
        self._selection = selection

    def forward(self) -> tuple[torch.Tensor, ...]:
        # A tensor a node returns without taking it as input becomes an output of the node itself, through which torch
        # routes the gradient of the loss. The node keeps no reference to it, which would outlive the rollout.
        tensors, self._tensors = self._tensors, None
        return tensors

    def backward(self, *outputs_grad: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        if all([grad is None for grad in outputs_grad]):
            return ()
        grads = tuple([None if grad is None else grad.contiguous() for grad in outputs_grad])
        self._solver.add_grad_from_state(grads, self._selection)
        return ()
