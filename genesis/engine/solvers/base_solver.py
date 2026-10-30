from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, NamedTuple, overload

import numpy as np
import torch

import numba as nb

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.entities.base_entity import Entity
from genesis.engine.materials.base import Material
from genesis.engine.state_change import StateChange, Subscriber, mutates
from genesis.engine.states import SolverCheckpoint
from genesis.grad.bridge import RolloutGraph
from genesis.repr_base import RBC
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    data_to_array,
    drop_repeated_index,
    indices_to_mask,
    qd_add_grad,
    qd_from_torch,
    qd_to_numpy,
    qd_to_torch,
    sanitize_index,
    sanitize_indexed_tensor,
    tensor_to_array,
)


if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class TimeBasedMixin:
    """Integration interval of one solver, and the number of substeps it implies per scene step.

    Mixed into every solver whose options carry a `dt`.
    """

    _substeps: int
    _substep_dt: float

    def __init__(self, scene, sim, options):
        """Compute how many substeps this solver runs per scene step, from the interval it integrates over.

        The count is `SimOptions.dt` divided by the interval the options of this solver ask for. A solver whose `dt`
        was left unset takes `SimOptions.substeps` instead. Since the substep loop advances every solver together,
        `Simulator.build` then checks that the active solvers all end up on the same count.
        """
        super().__init__(scene, sim, options)
        self._substeps = max(1, round(sim.dt / options.dt)) if "dt" in options.model_fields_set else sim.substeps
        self._substep_dt = sim.dt / self._substeps

    @property
    def dt(self):
        """Duration of one scene step, in seconds, which this solver covers in `substeps` substeps."""
        return self._sim.dt

    @property
    def substep_dt(self):
        return self._substep_dt

    @property
    def substeps(self):
        """The number of times this solver integrates per scene step."""
        return self._substeps

    @property
    def n_frames(self):
        """
        The number of frames of substeps the solver holds.

        A solver holding its state in frames keeps those of one step with gradients, whose backward pass replays it
        whole, and those of one substep otherwise (see 'Solver.has_frames').
        """
        return self._substeps + 1 if self._requires_grad else 2


@nb.jit(nopython=True, cache=True)
def merge_errno(errno_out: np.ndarray, errno_in: np.ndarray) -> int:
    """
    Add the error flags of each env in `errno_in` to `errno_out` in place, and return the union of `errno_out`.

    It merges the flags of a solver on CPU at the end of every step in one pass (see 'Solver.update_errno').
    """
    flags = 0
    for i_b in range(errno_out.shape[0]):
        errno_out[i_b] |= errno_in[i_b]
        flags |= errno_out[i_b]
    return flags


class BoundaryItems(NamedTuple):
    """The items a boundary checkpoint holds, listed once by the solver (see 'Solver.save_checkpoint')."""

    # Every item the checkpoint holds
    items: list[array_class.DataItem]
    # The names of the arrays copied at once, in the order of their views
    names: list[str]
    # The zero-copy views of the arrays copied at once
    views: list[torch.Tensor]
    # The items copied one by one: the scalars, and the arrays without a view
    items_copied: list[array_class.DataItem]
    # The zero-copy views of the gradients of the arrays, which a backward pass checks at every step, the environment
    # first with a trailing axis that gives every view one to reduce over
    grads: list[torch.Tensor]
    # The items whose gradient has no view, copied at every check
    items_grad_copied: list[array_class.ArrayItem]


class GravityMixin:
    """Gravity of one solver: the buffer storing it, and the accessors reading and writing it.

    Mixed into every solver whose options carry a gravity vector.
    """

    _gravity: "qd.Tensor | None" = None

    def _build_gravity(self, gravity=None, *, as_field: bool = False):
        """Allocate the buffer storing this solver's gravity and fill it with the value from the options.

        Called once, at build time, every environment getting the same value. Pass `gravity` to reuse a buffer the
        solver already declares among its own arrays, as the rigid solver does with `RigidInfo.gravity`, so that
        `set_gravity` and the kernels read the same memory. Leave it out to have one allocated here. Pass
        `as_field=True` for a solver whose kernels access gravity as an attribute of the solver object, which quadrants
        supports for a field only. A solver holding no entity gets no buffer, and its accessors then raise.
        """
        # FIXME: a field is what a kernel taking the solver as a template can read, since Quadrants resolves no
        # ndarray attribute of a @qd.data_oriented class in kernel scope. Two ways out, either of which retires the
        # as_field flag: adding ndarray support to that resolution, or migrating the solver to a frozen dataclass so
        # its ndarrays are predeclared. Until then the buffer is only allocated for a solver with an entity, since a
        # field costs an SNode tree that Quadrants never gives back short of qd.reset().
        if not self.is_active:
            return
        shape = (self._B,)
        if gravity is None:
            if as_field:
                gravity = qd.tensor(gs.qd_vec3, shape, backend=qd.Backend.FIELD)
            else:
                gravity = array_class.V(gs.qd_vec3, shape)
        self._gravity = gravity
        self._gravity.from_numpy(np.tile(np.array(self._options.gravity, dtype=gs.np_float), (self._B, 1)))

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_gravity(self, gravity, envs_idx=None):
        """
        Set the gravity acting on the bodies of this solver, in m/s^2.

        Parameters
        ----------
        gravity : array_like, shape (3,) or (n_envs, 3)
            The gravity vector, one per environment when several are named.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if self._gravity is None:
            gs.raise_exception(f"{type(self).__name__} holds no body for a gravity to act on.")
        # Indices are not checked on purpose: deciding to raise means reading a comparison back from the device, which
        # syncs, and this may run every step.
        if gs.use_zerocopy:
            envs_mask = indices_to_mask(envs_idx)
            gravity_t = qd_to_torch(self._gravity, transpose=True, copy=False)
            assign_indexed_tensor(
                gravity_t, envs_mask, broadcast_tensor(gravity, gs.tc_float, gravity_t[envs_mask].shape)
            )
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        gravity = broadcast_tensor(gravity, gs.tc_float, (len(envs_idx), 3), ("envs_idx", ""))
        kernel_set_gravity(gravity.contiguous(), envs_idx, self._gravity)

    def get_gravity(self, envs_idx=None):
        """
        Get the gravity acting on the bodies of this solver, in m/s^2.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        gravity : torch.Tensor, shape (3,) or (n_envs, 3)
            The gravity vector of each environment.
        """
        if self._gravity is None:
            gs.raise_exception(f"{type(self).__name__} holds no body for a gravity to act on.")
        tensor = qd_to_torch(self._gravity, envs_idx, transpose=True, copy=True)
        return tensor[0] if self._sim.n_envs == 0 else tensor


class Solver(RBC):
    # The material of the entities this solver simulates, None for a solver holding no entity of its own
    material_cls: type[Material] | None = None
    # Whether the solver supports a scene with `requires_grad`, which every active solver of such a scene must
    is_differentiable: bool = False
    # Whether the solver holds its state in frames, one per substep of a window of them (see 'n_frames'), frame 0
    # holding the state between two steps. The simulator copies the last frame into frame 0 at the end of each
    # window ('copy_frame'), and a boundary checkpoint holds frame 0 alone.
    has_frames: bool = False

    def __init__(self, scene: "Scene", sim: "Simulator", options):
        self._uid = gs.UID()
        self._sim = sim
        self._scene = scene
        self._options = options
        self._entities: list[Entity] = gs.List()

        # Whether the state of the solver tracks gradients, and the autograd graph of the simulator recording them
        self._requires_grad: bool = sim.options.requires_grad
        self._rollout_graph: RolloutGraph = sim._rollout_graph

        # The items and views of the boundary checkpoints, built by the first one, which comes after the build, and
        # valid from then on since the arrays are never reallocated, which spares walking the structs at every step
        self._boundary_items: BoundaryItems | None = None

        # force fields
        self._ffs = list()

        # Registered Subscribers, notified after @mutates-tagged methods run; see subscribe(). The re-entrancy guard
        # below defers notification to the outermost tagged call, accumulating every change that occurred in between,
        # so a setter calling other tagged setters notifies once rather than per nested call.
        self._subscribers: set[Subscriber] = set()
        self._is_mutating = False
        self._mutation_changes: set[StateChange] = set()
        self._mutation_links_idx_list: list[np.ndarray] | None = None
        # Global indices of the links whose pose is a function of the configuration, resolving
        # MutatedLinks.ARTICULATED notifications, and the per-link parent indices (-1 for roots) closing named-links
        # reaches over kinematic subtrees; None keeps the corresponding notifications unbounded. Set at build by
        # solvers that expose a kinematic structure (see KinematicSolver).
        self._articulated_links_idx: np.ndarray | None = None
        self._links_parent_idx: np.ndarray | None = None

        # The error flags the kernels of the solver write, the message raised for each, and the flags accumulated from
        # them and from the backward pass since the last reset of each env (see 'register_errors')
        self._errno: "qd.Tensor | None" = None
        self._error_messages: dict[int, str] = {}
        self._errno_latched: torch.Tensor | None = None
        # The page-locked host copy of '_errno_latched' a non-blocking check reads, and the event recorded once it
        # landed (see 'check_errno')
        self._errno_host: torch.Tensor | None = None
        self._errno_event: "torch.cuda.Event | None" = None
        # The context of the check that enqueued the copy 'check_errno' reads next
        self._errno_host_context: str | None = None

    def _add_force_field(self, force_field):
        self._ffs.append(force_field)

    def subscribe(self, subscriber: Subscriber):
        """Register a Subscriber to be notified after any @mutates-tagged method whose change is in its filter."""
        if subscriber.links_filter is not None:
            subscriber.links_filter = tensor_to_array(
                sanitize_index(subscriber.links_filter, -1, self.n_links, 0, "links_filter")
            )
        self._subscribers.add(subscriber)

    def register_errors(self, errno: qd.Tensor, messages: dict[int, str]) -> None:
        """
        Declare the error flags the kernels of the solver write per env, one bit per error, and the message of each.

        `errno` is an array of the solver, which the base solver references. 'check_errno' raises the message of the
        first flagged error, in the order of `messages`. It reads `_errno_latched`, the union of every flag `errno` held
        since the last reset of each env, which no restore of the state rewinds (see 'update_errno'), so a solver clears
        the flags of some envs through 'clear_errno'. The flags of the backward pass come after those of `messages`
        (see 'StepOperation.backward').
        """
        # Each error owns its bit, those of the backward pass included
        assert not messages.keys() & self._error_messages.keys(), f"{type(self).__name__} reuses a reserved error bit."
        self._errno = errno
        self._error_messages = {**messages, **self._error_messages}

    def update_errno(self) -> None:
        """
        Add the error flags `errno` holds to those the solver raises, '_errno_latched'.

        A step calls this once it is over, so that 'check_errno' and 'get_error_envs_mask' read '_errno_latched' alone.
        A backward pass restores `errno` with the state before reversing each step, so it calls this once before its
        first restore and after reversing each step.
        """
        if self._errno is None:
            return
        # On CPU, numpy views of both share their memory, which spares a torch operation
        if gs.backend == gs.cpu and gs.use_zerocopy.is_supported:
            merge_errno(self._errno_latched.numpy(), qd_to_numpy(self._errno, copy=False))
        else:
            self._errno_latched |= qd_to_torch(self._errno)

    def get_error_envs_mask(self) -> torch.Tensor:
        """Return whether each env flagged an error since its last reset (see 'register_errors')."""
        return self._errno_latched > 0

    def flag_errno(self, code: int, envs_idx=None) -> None:
        """Flag the given error for the given envs, all of them if None, which 'check_errno' then raises."""
        # The flags of every env are updated through a mask, rather than those of the selected envs alone: indexing by a
        # boolean mask makes torch count the envs it selects to size the result, which waits for the device, and the
        # backward pass flags envs after reversing every step (see 'StepOperation.backward')
        envs_mask = torch.zeros_like(self._errno_latched, dtype=torch.bool)
        envs_mask[indices_to_mask(envs_idx)] = True
        self._errno_latched |= torch.where(envs_mask, code, 0)

    def clear_errno(self, envs_idx=None) -> None:
        """Clear the errors the solver flagged for the given envs, all of them if None (see 'register_errors')."""
        envs_mask = indices_to_mask(envs_idx)
        if self._errno is not None:
            if gs.use_zerocopy:
                errno = qd_to_torch(self._errno, copy=False)
                errno[envs_mask] = 0
                if gs.backend == gs.metal:
                    torch.mps.synchronize()
            else:
                errno = qd_to_torch(self._errno, copy=True)
                errno[envs_mask] = 0
                qd_from_torch(self._errno, errno)
        self._errno_latched[envs_mask] = 0

    def check_errno(self, blocking: bool = True, *, context: str | None = None) -> None:
        """
        Raise on the first error the solver flagged since the last reset of each env (see 'register_errors').

        Without `blocking` on CUDA, the flags are copied to page-locked host memory without waiting for the device, and
        the copy the previous call enqueued is checked instead: an error raises one call later, with the context of the
        call that enqueued it, and the host only waits for the device when it runs ahead of it by more than the interval
        between two calls. A blocking check raises what no copy has reported yet with the context of the last one.
        `context` says where the check runs, ahead of the message.
        """
        if not self._error_messages:
            return
        if not blocking and self._errno_host is not None:
            self._errno_event.synchronize()
            errno = np.bitwise_or.reduce(self._errno_host.numpy())
            context, self._errno_host_context = self._errno_host_context, context
            # The flags stay raised on the device until a reset, so a copy that raises is cleared rather than renewed,
            # which spares raising it again once a valid state is set
            if errno:
                self._errno_host.zero_()
            else:
                self._errno_host.copy_(self._errno_latched, non_blocking=True)
                self._errno_event.record()
        else:
            errno = np.bitwise_or.reduce(tensor_to_array(self._errno_latched))
            if context is None:
                context = self._errno_host_context
            self._errno_host_context = None
        if not errno:
            return
        for code, message in self._error_messages.items():
            if errno & code:
                gs.raise_exception(message if context is None else f"{context}: {message}")

    def _sanitize_io_variables(
        self,
        tensor,
        inputs_idx,
        input_size,
        idx_name,
        envs_idx=None,
        element_shape=(),
        *,
        dtype=None,
        batched=True,
        batched_idx=False,
        skip_allocation=False,
        accumulate=False,
    ):
        """
        Sanitize the indices of the entries of a state a getter reads or a setter writes, and the values written.

        The values are floating unless `dtype` says otherwise. `inputs_idx` is shared by every env. With `batched_idx`,
        for kernels indexing them per env, the indices are returned with a leading env dimension, whatever the number
        of envs: indices shared by every env are repeated for each of them (`[3, 7]` becomes `[[3, 7], [3, 7]]` for two
        envs, `3` becomes `[[3], [3]]`), while a nested sequence holds one row of indices per env (`[[3, 7], [1, 2]]`
        selects the entries 3 and 7 in the first env, 1 and 2 in the second). The values are returned the same way.

        A write keeps the last occurrence of an index given several times and its values, which is the state writing
        them in turn leaves (see 'drop_repeated_index'): in a row of indices per env, an earlier occurrence becomes -1,
        which the kernels skip. A write adding its values to the state (`accumulate`) keeps every occurrence, so that
        the values of a repeated index add up.
        """
        if dtype is None:
            dtype = gs.tc_float
        n_envs = self._scene.n_envs
        is_overwrite = tensor is not None and not accumulate
        envs_idx_ = self._scene._sanitize_envs_idx(envs_idx) if batched else self._scene._envs_idx[:0]

        if batched and batched_idx and np.ndim(inputs_idx) == 2:
            dim_names = ("envs_idx", idx_name, *["" for _ in element_shape])
            inputs_idx_ = broadcast_tensor(inputs_idx, gs.tc_int, (len(envs_idx_), -1), dim_names[:2]).contiguous()
            # Negative indices wrap as in sanitize_index, before the repeated ones are dropped
            is_valid_negative = (-input_size <= inputs_idx_) & (inputs_idx_ < 0)
            inputs_idx_ = torch.where(is_valid_negative, inputs_idx_ + input_size, inputs_idx_)
            tensor_ = None
            if tensor is not None or not skip_allocation:
                tensor_shape = (*inputs_idx_.shape, *element_shape)
                tensor_ = broadcast_tensor(tensor, dtype, tensor_shape, dim_names).contiguous()
            if is_overwrite:
                envs_idx_, (tensor_, inputs_idx_) = drop_repeated_index(
                    envs_idx, envs_idx_, (tensor_, inputs_idx_), axis=0, max_size=n_envs
                )
                inputs_idx_, (tensor_,) = drop_repeated_index(inputs_idx, inputs_idx_, (tensor_,), 1, input_size)
            return tensor_, inputs_idx_, envs_idx_

        if n_envs == 0 or not batched:
            tensor_, (inputs_idx_,) = sanitize_indexed_tensor(
                tensor,
                dtype,
                (inputs_idx,),
                (-1, *element_shape),
                (input_size, *element_shape),
                (idx_name, *["" for _ in element_shape]),
                skip_allocation=skip_allocation,
            )
        else:
            tensor_, (envs_idx_, inputs_idx_) = sanitize_indexed_tensor(
                tensor,
                dtype,
                (envs_idx_, inputs_idx),
                (-1, -1, *element_shape),
                (n_envs, input_size, *element_shape),
                ("envs_idx", idx_name, *["" for _ in element_shape]),
                skip_allocation=skip_allocation,
            )
            if is_overwrite:
                envs_idx_, (tensor_,) = drop_repeated_index(envs_idx, envs_idx_, (tensor_,), axis=0, max_size=n_envs)
        if is_overwrite:
            axis = 1 if n_envs > 0 and batched else 0
            inputs_idx_, (tensor_,) = drop_repeated_index(inputs_idx, inputs_idx_, (tensor_,), axis, input_size)

        if batched_idx:
            inputs_idx_ = inputs_idx_.expand(len(envs_idx_), -1).contiguous()
            if tensor_ is not None and n_envs == 0:
                tensor_ = tensor_[None]
        return tensor_, inputs_idx_, envs_idx_

    def build(self):
        self._B = self._sim._B
        self._errno_latched = torch.zeros((self._B,), dtype=gs.tc_int, device=gs.device)
        if gs.device.type == "cuda":
            self._errno_host = torch.zeros((self._B,), dtype=gs.tc_int, pin_memory=True)
            self._errno_event = torch.cuda.Event()
        if self._requires_grad:
            self._error_messages[array_class.ErrorCode.INVALID_GRAD_NAN] = (
                "The backward pass produced a 'nan' or 'inf' gradient of the state. The rollout is likely too stiff to "
                "differentiate over this horizon: shorten it, or decrease the simulation timestep."
            )

    # ------------------------------------------------------------------------------------
    # ------------------------------ differentiable simulation ---------------------------
    # ------------------------------------------------------------------------------------

    # The protocol a solver implements to take part in a differentiable scene. The simulator calls it on every active
    # solver alone, so a solver of any origin becomes differentiable by implementing it, and the defaults below fit a
    # solver holding no differentiable state.

    @property
    def requires_grad(self) -> bool:
        """
        Whether the state of the solver tracks gradients.

        In a scene with `requires_grad`, the state of every solver does, but those whose state stays outside the
        autograd graph (see 'KinematicSolver').
        """
        return self._requires_grad

    @property
    def is_recording(self) -> bool:
        """
        Whether the operations on the state of the solver are recorded in the autograd graph of the rollout right now.

        They are while the solver tracks gradients ('requires_grad') and the graph records, which it stops doing under
        `torch.no_grad` and while a step replays its kernels for its backward pass (see 'RolloutGraph.is_recording').
        """
        return self._requires_grad and self._rollout_graph.is_recording

    def record_read(self, tensors: tuple[torch.Tensor, ...], selection) -> None:
        """
        Record the query of the given tensors of a state as one operation of the rollout, while it is recorded (see
        ReadOperation).

        `selection` names the state the tensors cover, which 'add_grad_from_state' interprets.
        """
        if self.is_recording:
            self._rollout_graph.record_read(self, tensors, selection)

    @overload
    def _read_data(
        self, data: qd.Tensor | qd.Field | qd.Ndarray, envs_idx, idx, *, copy: bool | None = None
    ) -> torch.Tensor: ...

    @overload
    def _read_data(
        self, data: Sequence[qd.Tensor | qd.Field | qd.Ndarray], envs_idx, idx, *, copy: bool | None = None
    ) -> list[torch.Tensor]: ...

    def _read_data(self, data, envs_idx, idx, *, copy=None):
        """
        Read the given slots of one array of the solver, or of several, in the layout of the getters ('qd_to_torch').

        `data` is one array, read as one tensor, or a sequence of them, read as a list of tensors, whatever their kind
        (state, info). A solver holding its state in frames is given arrays held in frames, read from frame 0, which
        holds the state between two steps (see 'has_frames'). `copy` is
        passed on to 'qd_to_torch': a getter returning what it reads asks for a copy, since a getter returns by value.
        While the operations of the solver are recorded, the read is one operation of the rollout whose outputs are
        copies of the whole arrays, which a later step leaves untouched, and from which torch selects the slots (see
        'record_read').
        """
        is_single = isinstance(data, (qd.Tensor, qd.Field, qd.Ndarray))
        fields = (data,) if is_single else tuple(data)
        copy = True if self.is_recording else copy
        if self.has_frames:
            tensors = [qd_to_torch(field, row_mask=0, keepdim=False).movedim(1, 0) for field in fields]
            if copy:
                tensors = [tensor.clone(memory_format=torch.contiguous_format) for tensor in tensors]
        elif self.is_recording:
            tensors = [qd_to_torch(field, transpose=True, copy=True) for field in fields]
        else:
            tensors = [qd_to_torch(field, envs_idx, idx, transpose=True, copy=copy) for field in fields]
        if self.is_recording:
            self.record_read(tuple(tensors), fields)
        if self.has_frames or self.is_recording:
            mask = indices_to_mask(envs_idx, idx)
            tensors = [tensor[mask] for tensor in tensors]
        return tensors[0] if is_single else tensors

    def add_grad_from_state(self, grads, selection) -> None:
        """
        Add the gradients of the tensors of a queried state to the adjoint of the state of the solver.

        By default, `selection` holds the fields the tensors are, each read whole in the layout of the getters, from
        frame 0 for a solver holding its state in frames (see '_read_data').
        """
        for field, grad in zip(selection, grads):
            # A field with no entry keeps one padding slot, to which no gradient goes
            if grad is not None and grad.numel() > 0:
                if self.has_frames:
                    qd_add_grad(field, grad.movedim(0, 1), row=0)
                else:
                    qd_add_grad(field, grad)
        if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
            torch.mps.synchronize()

    def reset_grad(self) -> None:
        """Zero the adjoint of the state of the solver, from which a backward pass starts."""

    @property
    def data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        """Yield every array, scalar and static config the solver holds, under the dotted name a checkpoint and a
        trajectory frame use for it (see 'iter_data').
        """
        gs.raise_exception(f"{type(self).__name__} cannot be checkpointed yet.")

    def save_checkpoint(
        self, kinds: frozenset[array_class.DataKind], *, boundary_only: bool = False
    ) -> SolverCheckpoint:
        """
        Return a checkpoint of the arrays of the given kinds this solver holds, for 'load_checkpoint' to restore.

        The arrays are what 'data' yields of those kinds. A full checkpoint copies them with their gradients where they
        live: on the device as torch tensors when zero-copy views exist, which keeps a save and a restore within one
        process off the host, and as numpy arrays otherwise. A boundary checkpoint holds the state at the boundary of
        two steps alone, from which the backward pass of a recorded step replays it: no gradient, frame 0 of the arrays
        holding frames, on the device whatever the backend.
        """
        arrays = {}
        if boundary_only:
            if self._boundary_items is None:
                # The gradients ('iter_data' names each one after its array) and the placeholders of the features
                # switched off (see 'ArrayItem') hold nothing a replay reads
                self._boundary_items = BoundaryItems(
                    items=[], names=[], views=[], items_copied=[], grads=[], items_grad_copied=[]
                )
                grad_names = set()
                for item in self.data:
                    if isinstance(item, array_class.ConfigItem) or item.kind not in kinds:
                        continue
                    if isinstance(item, array_class.ArrayItem):
                        is_placeholder = not isinstance(item.array, torch.Tensor) and not item.array.shape
                        if item.name in grad_names or is_placeholder:
                            continue
                        if not isinstance(item.array, torch.Tensor) and item.array.has_grad():
                            grad_names.add(f"{item.name}.grad")
                            grad = None
                            if gs.use_zerocopy.is_supported:
                                try:
                                    grad = qd_to_torch(item.array.grad, transpose=True, copy=False)
                                except ValueError:
                                    # See 'fill_data'
                                    pass
                            if grad is None:
                                self._boundary_items.items_grad_copied.append(item)
                            else:
                                self._boundary_items.grads.append(grad[..., None])
                    self._boundary_items.items.append(item)
                    if isinstance(item, array_class.ScalarItem) or not gs.use_zerocopy.is_supported:
                        self._boundary_items.items_copied.append(item)
                        continue
                    if isinstance(item.array, torch.Tensor):
                        view = item.array
                    else:
                        try:
                            if self.has_frames:
                                view = qd_to_torch(item.array, row_mask=0, keepdim=False, copy=False)
                            else:
                                view = qd_to_torch(item.array, copy=False)
                        except ValueError:
                            # A member of a struct field may have no zero-copy view (see 'fill_data')
                            self._boundary_items.items_copied.append(item)
                            continue
                    self._boundary_items.names.append(item.name)
                    self._boundary_items.views.append(view)
            if self._boundary_items.views:
                # One fused copy of every array with a view, rather than one launch each
                copies = [torch.empty_like(view) for view in self._boundary_items.views]
                # The views read the arrays as the last kernel left them once it is done (see qd_to_torch)
                if gs.backend == gs.metal:
                    qd.sync()
                torch._foreach_copy_(copies, self._boundary_items.views)
                # The copy reads the arrays on the torch stream, before the next kernel writes them (see qd_zero_grad)
                if gs.backend == gs.metal:
                    torch.mps.synchronize()
                arrays.update(zip(self._boundary_items.names, copies))
            items = self._boundary_items.items_copied
        else:
            items = [item for item in self.data if not isinstance(item, array_class.ConfigItem) and item.kind in kinds]
        for item in items:
            if isinstance(item, array_class.ScalarItem):
                arrays[item.name] = np.array(getattr(item.owner, item.attr))
            elif boundary_only and self.has_frames:
                arrays[item.name] = qd_to_torch(item.array, row_mask=0, keepdim=False, copy=True)
            else:
                arrays[item.name] = data_to_array(
                    item.array, to_torch=boundary_only or gs.use_zerocopy.is_supported, copy=True
                )
        # The static configs are provenance, which a boundary checkpoint never leaving the process does without
        configs = {}
        if not boundary_only:
            configs = {item.name: item.config for item in self.data if isinstance(item, array_class.ConfigItem)}
        return SolverCheckpoint(arrays=arrays, configs=configs, kinds=kinds)

    def load_checkpoint(self, checkpoint: SolverCheckpoint, *, boundary_only: bool = False) -> None:
        """Write back the arrays of a checkpoint 'save_checkpoint' returned, rejecting a record read from another build.

        `boundary_only` states that 'save_checkpoint' saved a boundary checkpoint.

        Nothing is notified to the subscribers (see 'mutates'). A solver whose step reads a DERIVED array before
        recomputing it overrides this to recompute or invalidate that array when the record lacks DERIVED (see
        'DataKind'). On Metal, the caller synchronizes the torch stream before the next kernel (see 'fill_data').
        """
        if boundary_only:
            # The arrays with a view are written back in one fused copy, from the copies 'save_checkpoint' made of them
            # in this process, which spares checking them against the scene
            if self._boundary_items.views:
                # The writes wait for the last kernel to be done with the arrays (see qd_to_torch)
                if gs.backend == gs.metal:
                    qd.sync()
                values = [checkpoint.arrays[name] for name in self._boundary_items.names]
                torch._foreach_copy_(self._boundary_items.views, values)
            items = self._boundary_items.items_copied
            arrays = {item.name: checkpoint.arrays[item.name] for item in items}
        else:
            items = [
                item
                for item in self.data
                if not isinstance(item, array_class.ConfigItem) and item.kind in checkpoint.kinds
            ]
            arrays = checkpoint.arrays
        array_class.fill_data(items, arrays, frame_only=boundary_only and self.has_frames)

    def __getstate__(self) -> SolverCheckpoint:
        """Return a SolverCheckpoint of this solver, for '__setstate__' to restore (see 'CHECKPOINT_KINDS')."""
        return self.save_checkpoint(array_class.CHECKPOINT_KINDS)

    def __setstate__(self, state: SolverCheckpoint) -> None:
        """Write back the arrays of the kinds a record holds, rejecting a record read from another build."""
        self.load_checkpoint(state)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def scene(self):
        return self._scene

    @property
    def sim(self):
        return self._sim

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def entities(self) -> list[Entity]:
        return self._entities

    @property
    def n_entities(self):
        return len(self._entities)

    @property
    def force_fields(self) -> list:
        return self._ffs

    def _repr_brief(self):
        repr_str = f"{self.__repr_name__()}: {self._uid}, n_entities: {self.n_entities}"
        return repr_str


@qd.kernel
def kernel_set_gravity(tensor: qd.types.ndarray(), envs_idx: qd.types.ndarray(), gravity: qd.Tensor):
    # The qd.Tensor annotation accepts both forms the buffer is allocated in: a field, for a solver whose kernels
    # reach gravity through the solver itself, and an ndarray for every other one (see _build_gravity).
    for i_b_ in range(envs_idx.shape[0]):
        for j in qd.static(range(3)):
            gravity[envs_idx[i_b_]][j] = tensor[i_b_, j]
