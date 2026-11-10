from collections.abc import Callable, Iterator, Sequence
from enum import IntFlag
from typing import TYPE_CHECKING, NamedTuple, overload

import numpy as np
import torch

import numba as nb

import quadrants as qd

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.engine.force_fields import ForceField, ForceFields
from genesis.engine.state_change import StateChange, Subscriber, mutates
from genesis.engine.states import SolverCheckpoint
from genesis.grad.bridge import RolloutGraph, no_grad
from genesis.repr_base import RBC
from genesis.utils.data import (
    CHECKPOINT_KINDS,
    ENVS,
    ArrayItem,
    ConfigItem,
    Data,
    DataItem,
    DataKind,
    ScalarItem,
    Size,
    copy_data,
    eoa,
    field,
    fill_data,
)
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


# The bits of an errno: those of an int32 without its sign bit, the flags being int32 kernel literals
ERRNO_BITS = 31


class BackwardErrorCode(IntFlag):
    """The error flags every solver tracking gradients holds in its errno beside its own, which the backward pass writes.

    They take the lowest bits, those of each solver starting above them (see ErrorCode).
    """

    INVALID_GRAD_NAN = 0b00000000000000000000000000000001


class ErrorCode(IntFlag):
    """Base of the error flags of one solver, whose members take successive bits above those of BackwardErrorCode.

    Each solver declares its own flags in a subclass with auto(), in declaration order, the subclass raising at
    creation past the bits of an errno (see ERRNO_BITS). Python refuses to subclass an enum that holds members, so the
    flags of the backward pass stay apart in BackwardErrorCode rather than in this base.
    """

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[int]) -> int:
        bit = max(BackwardErrorCode).bit_length() + count
        if bit >= ERRNO_BITS:
            raise TypeError(f"'{name}' exceeds the {ERRNO_BITS} bits of an errno.")
        return 1 << bit


# The visual vertices and faces a solver skins for rendering
VVERTS = Size("vverts")
VFACES = Size("vfaces")


class SimulatorState(Data, kind=DataKind.SCRATCH):
    """
    Runtime state of the simulator that the kernels of the solvers read.

    `t` is the time of the substep being run in each environment, in seconds, written at the start of every substep from
    the steps the environment counted (see 'Simulator.get_time'), since the environments step and reset independently.
    """

    t: qd.Tensor = field(float, shape=(ENVS,))


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


class ErrnoMixin:
    """The error flags a component writes per env from its kernels, raised at the beginning of the next step.

    Mixed into every solver, and into the couplers whose kernels flag errors of their own.
    """

    # The error flags the kernels write, the report building the message raised for each, and the flags accumulated
    # from them and from the backward pass since the last reset of each env (see 'register_errors').
    _errno: "qd.Tensor | None" = None
    _error_reports: dict[int, Callable[[], str]] = {}
    _errno_latched: torch.Tensor | None = None
    # The page-locked host copy of '_errno_latched' a non-blocking check reads, and the event recorded once it landed
    # (see 'check_errno').
    _errno_host: torch.Tensor | None = None
    _errno_event: "torch.cuda.Event | None" = None
    # The context of the check that enqueued the copy 'check_errno' reads next
    _errno_host_context: str | None = None

    def build_errno(self, n_envs: int) -> None:
        """Allocate the flags accumulated for each env, which the build of the component calls."""
        self._errno_latched = torch.zeros((n_envs,), dtype=gs.tc_int, device=gs.device)
        if gs.device.type == "cuda":
            self._errno_host = torch.zeros((n_envs,), dtype=gs.tc_int, pin_memory=True)
            self._errno_event = torch.cuda.Event()

    def register_errors(self, errno: qd.Tensor, reports: dict[int, Callable[[], str]]) -> None:
        """
        Declare the error flags the kernels of the component write per env, one bit per error, and the report of each.

        `errno` is an array of the component, which the mixin references. 'check_errno' raises the message returned by
        the report of the first flagged error, in the order of `reports`. A report runs when its error raises, so that
        its message can name what the flags alone cannot tell apart, such as the environment past a capacity. The check
        reads `_errno_latched`, the union of every flag `errno` held since the last reset of each env, which no restore
        of the state rewinds (see 'update_errno'), so a component clears the flags of some envs through 'clear_errno'.
        The flags of the backward pass come after those of `reports`, as 'StepOperation.backward' sets them.
        """
        # Each error owns its bit, those of the backward pass included
        assert not reports.keys() & self._error_reports.keys(), f"{type(self).__name__} reuses a reserved error bit."
        self._errno = errno
        self._error_reports = {**reports, **self._error_reports}

    def update_errno(self) -> None:
        """
        Add the error flags `errno` holds to those the component raises, '_errno_latched'.

        A step calls this once it is over, so that 'check_errno' and 'get_error_envs_mask' read '_errno_latched' alone.
        A backward pass restores `errno` with the state before reversing each step, so it calls this once before its
        first restore and after reversing each step.
        """
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
        # backward pass flags envs after reversing every step (see 'StepOperation.backward').
        envs_mask = torch.zeros_like(self._errno_latched, dtype=torch.bool)
        envs_mask[indices_to_mask(envs_idx)] = True
        self._errno_latched |= torch.where(envs_mask, code, 0)

    def clear_errno(self, envs_idx=None) -> None:
        """Clear the errors the component flagged for the given envs, all of them if None (see 'register_errors')."""
        envs_mask = indices_to_mask(envs_idx)
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
        Raise on the first error the component flagged since the last reset of each env (see 'register_errors').

        Without `blocking` on CUDA, the flags are copied to page-locked host memory without waiting for the device, and
        the copy the previous call enqueued is checked instead: an error raises one call later, with the context of the
        call that enqueued it, and the host only waits for the device when it runs ahead of it by more than the interval
        between two calls. A blocking check raises what no copy has reported yet with the context of the last one.
        `context` says where the check runs, ahead of the message.
        """
        if not self._error_reports:
            return
        if not blocking and self._errno_host is not None:
            self._errno_event.synchronize()
            errno = np.bitwise_or.reduce(self._errno_host.numpy())
            context, self._errno_host_context = self._errno_host_context, context
            # The flags stay raised on the device until a reset, so a copy that raises is cleared rather than renewed,
            # which spares raising it again once a valid state is set.
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
        for code, report in self._error_reports.items():
            if errno & code:
                message = report()
                gs.raise_exception(message if context is None else f"{context}: {message}")


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
    items: list[DataItem]
    # The items copied at once, in the order of their views
    items_viewed: list[ArrayItem]
    # The zero-copy views of the arrays copied at once
    views: list[torch.Tensor]
    # The items copied one by one: the scalars, and the arrays without a view
    items_copied: list[DataItem]
    # The zero-copy views of the gradients of the arrays, which a backward pass checks at every step, the environment
    # first with a trailing axis that gives every view one to reduce over
    grads: list[torch.Tensor]
    # The items whose gradient has no view, copied at every check
    items_grad_copied: list[ArrayItem]


class CheckpointMixin:
    """The checkpoint of a component whose whole state lives in its declared data, which it walks through 'data'.

    A component sets `is_checkpointable` to declare that its 'data' holds its whole state, every entry point of the
    protocol raising otherwise. Mixed into every solver and every coupler. The activation of the force fields of the
    scene, shared by every solver, is a control of the scene outside the state of any of them.
    """

    # Whether the arrays 'data' yields hold the whole state of the component, which a checkpoint then restores exactly
    is_checkpointable: bool = False
    # The items and views of the boundary checkpoints, built by the first one, which comes after the build, and valid
    # from then on since the arrays are never reallocated, which spares walking the structs at every step
    _boundary_items: BoundaryItems | None = None

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        """Yield every array, scalar and static config the component holds, under the dotted name a checkpoint and a
        trajectory frame use for it (see 'iter_data').
        """
        gs.raise_exception(f"{type(self).__name__} cannot be checkpointed yet: its state lives outside its data.")

    def save_checkpoint(self, *, boundary_only: bool = False) -> SolverCheckpoint:
        """
        Return a checkpoint of the arrays this component holds, for 'load_checkpoint' to restore.

        A full checkpoint holds the arrays of every kind a step reads (see 'CHECKPOINT_KINDS'), a boundary checkpoint
        those of the state alone (see 'DataKind.STATE'). A full checkpoint copies them with their gradients where they
        live: on the device as torch tensors when zero-copy views exist, which keeps a save and a restore within one
        process off the host, and as numpy arrays otherwise. A boundary checkpoint holds the state at the boundary of
        two steps alone, from which the backward pass of a recorded step replays it: no gradient, on the device whatever
        the backend.
        """
        if not self.is_checkpointable:
            gs.raise_exception(f"{type(self).__name__} cannot be checkpointed yet: its state lives outside its data.")
        arrays = {}
        kinds = frozenset((DataKind.STATE,)) if boundary_only else CHECKPOINT_KINDS
        if boundary_only:
            if self._boundary_items is None:
                # The gradients ('iter_data' names each one after its array) and the placeholders of the features
                # switched off (see 'ArrayItem') hold nothing a replay reads
                self._boundary_items = BoundaryItems(
                    items=[], items_viewed=[], views=[], items_copied=[], grads=[], items_grad_copied=[]
                )
                grad_names = set()
                for item in self.data:
                    if isinstance(item, ConfigItem) or item.kind != DataKind.STATE:
                        continue
                    if isinstance(item, ArrayItem):
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
                    if isinstance(item, ScalarItem) or not gs.use_zerocopy.is_supported:
                        self._boundary_items.items_copied.append(item)
                        continue
                    if isinstance(item.array, torch.Tensor):
                        view = item.array
                    else:
                        try:
                            view = qd_to_torch(item.array, copy=False)
                        except ValueError:
                            # A member of a struct field may have no zero-copy view (see 'fill_data')
                            self._boundary_items.items_copied.append(item)
                            continue
                    self._boundary_items.items_viewed.append(item)
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
                arrays.update(zip([item.name for item in self._boundary_items.items_viewed], copies))
            items = self._boundary_items.items_copied
        else:
            items = [item for item in self.data if not isinstance(item, ConfigItem) and item.kind in kinds]
        arrays.update(copy_data(items, to_torch=boundary_only or gs.use_zerocopy.is_supported))
        # The static configs are provenance, which a boundary checkpoint never leaving the process does without
        configs = {}
        if not boundary_only:
            configs = {item.name: item.config for item in self.data if isinstance(item, ConfigItem)}
        return SolverCheckpoint(arrays=arrays, configs=configs, kinds=kinds)

    def load_checkpoint(
        self,
        checkpoint: SolverCheckpoint,
        *,
        boundary_only: bool = False,
        envs_idx: int | Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """Write back the arrays of a checkpoint 'save_checkpoint' returned, rejecting a record read from another build.

        `boundary_only` states that 'save_checkpoint' saved a boundary checkpoint, of which `envs_idx` selects the
        environments to write back, all of them if None, along the last index of every array of the state.

        Nothing is notified to the subscribers (see 'mutates'). A component whose step reads a DERIVED array before
        recomputing it overrides this to recompute or invalidate that array when the record lacks DERIVED (see
        'DataKind'). On Metal, the caller synchronizes the torch stream before the next kernel (see 'fill_data').
        """
        if not self.is_checkpointable:
            gs.raise_exception(f"{type(self).__name__} cannot be checkpointed yet: its state lives outside its data.")
        assert boundary_only or envs_idx is None
        if boundary_only:
            # The arrays with a view are written back in one fused copy, from the copies 'save_checkpoint' made of them
            # in this process, which spares checking them against the scene
            items_mask = {}
            if envs_idx is not None:
                # The saved arrays are written back whole through a mask of the selected environments, which spares
                # reading their count back from the device (see 'assign_indexed_tensor').
                envs_mask = torch.zeros((self.sim._B,), dtype=torch.bool, device=gs.device)
                assign_indexed_tensor(envs_mask, indices_to_mask(envs_idx), value=True)
                for item in self._boundary_items.items:
                    if isinstance(item, ArrayItem):
                        items_mask[item.name] = (*[slice(None)] * (len(item.array.shape) - 1), envs_mask)
            if self._boundary_items.views:
                # The writes wait for the last kernel to be done with the arrays (see qd_to_torch)
                if gs.backend == gs.metal:
                    qd.sync()
                values = [checkpoint.arrays[item.name] for item in self._boundary_items.items_viewed]
                if envs_idx is None:
                    torch._foreach_copy_(self._boundary_items.views, values)
                else:
                    for item, view, value in zip(self._boundary_items.items_viewed, self._boundary_items.views, values):
                        assign_indexed_tensor(view, items_mask[item.name], value)
            items = self._boundary_items.items_copied
            arrays = {item.name: checkpoint.arrays[item.name] for item in items}
            if envs_idx is not None:
                # The arrays without a view are written whole, their other environments as they stand
                for item in items:
                    if isinstance(item, ArrayItem):
                        array = data_to_array(item.array, to_torch=True, copy=True)
                        assign_indexed_tensor(array, items_mask[item.name], arrays[item.name])
                        arrays[item.name] = array
        else:
            items = [item for item in self.data if not isinstance(item, ConfigItem) and item.kind in checkpoint.kinds]
            arrays = checkpoint.arrays
        fill_data(items, arrays)

    def get_state(self) -> SolverCheckpoint:
        """Return the state the component holds between two steps, for 'set_state' to restore (see 'DataKind.STATE')."""
        return self.save_checkpoint(boundary_only=True)

    def set_state(
        self, state: SolverCheckpoint, envs_idx: int | Sequence[int] | np.ndarray | torch.Tensor | None = None
    ) -> None:
        """
        Write back the state 'get_state' returned, for the given environments, all of them if None.

        The state is restored exactly as saved, and the errors the component flagged for these environments are cleared.
        """
        # The selection reaches the zero-copy views of the state as it came (see 'load_checkpoint')
        n_envs = self.sim.scene.n_envs
        if n_envs == 0 and envs_idx is not None:
            gs.raise_exception("`envs_idx` is not supported for non-parallelized scene.")
        if isinstance(envs_idx, (int, np.integer)) and not -n_envs <= envs_idx < n_envs:
            gs.raise_exception(f"`envs_idx` out of range: {envs_idx} not in [{-n_envs}, {n_envs}).")
        self.load_checkpoint(state, boundary_only=True, envs_idx=envs_idx)
        if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
            torch.mps.synchronize()
        self.clear_errno(envs_idx)

    def __getstate__(self) -> SolverCheckpoint:
        """Return a SolverCheckpoint of this component, for '__setstate__' to restore (see 'CHECKPOINT_KINDS')."""
        return self.save_checkpoint()

    def __setstate__(self, state: SolverCheckpoint) -> None:
        """Write back the arrays of the kinds a record holds, rejecting a record read from another build."""
        self.load_checkpoint(state)


class GravityMixin:
    """Gravity of one solver: the buffer storing it, and the accessors reading and writing it.

    Mixed into every solver whose options carry a gravity vector.
    """

    _gravity: "qd.Tensor | None" = None

    def _build_gravity(self, gravity=None):
        """Allocate the buffer storing this solver's gravity and fill it with the value from the options.

        Called once, at build time, every environment getting the same value. Pass `gravity` to reuse a buffer the
        solver already declares among its own arrays, as the rigid solver does with `RigidInfo.gravity`, so that
        `set_gravity` and the kernels read the same memory. Leave it out to have a field allocated here, for a
        solver whose kernels read gravity as an attribute of the solver object. A solver holding no entity gets no
        buffer, and its accessors then raise.
        """
        # FIXME: a field is what a kernel taking the solver as a template can read, since Quadrants resolves no
        # ndarray attribute of a @qd.data_oriented class in kernel scope. Two ways out, either of which lets the solver
        # declare its buffer: adding ndarray support to that resolution, or migrating the solver to a frozen dataclass
        # so its ndarrays are predeclared. Until then the buffer is only allocated for a solver with an entity, since a
        # field costs an SNode tree that Quadrants never gives back short of qd.reset().
        if not self.is_active:
            return
        if gravity is None:
            gravity = qd.tensor(gs.qd_vec3, (self._B,), backend=qd.Backend.FIELD)
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
            assign_indexed_tensor(gravity_t, envs_mask, gravity)
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


class Solver(ErrnoMixin, CheckpointMixin, RBC):
    # Whether the solver supports a scene with `requires_grad`, which every active solver of such a scene must
    is_differentiable: bool = False

    def __init__(self, scene: "Scene", sim: "Simulator", options):
        self._uid = gs.UID()
        self._sim = sim
        self._scene = scene
        self._options = options
        self._entities: list[Entity] = gs.List()

        # Whether the state of the solver tracks gradients, and the autograd graph of the simulator recording them
        self._requires_grad: bool = sim.options.requires_grad
        self._rollout_graph: RolloutGraph = sim._rollout_graph

        # The force fields acting on the entities of the solver, which its kernels take as a template
        self._force_fields = ForceFields()

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

    def _add_force_field(self, force_field: ForceField) -> None:
        self._force_fields.fields.append(force_field)

    def subscribe(self, subscriber: Subscriber):
        """Register a Subscriber to be notified after any @mutates-tagged method whose change is in its filter."""
        if subscriber.links_filter is not None:
            subscriber.links_filter = tensor_to_array(
                sanitize_index(subscriber.links_filter, -1, self.n_links, 0, "links_filter")
            )
        self._subscribers.add(subscriber)

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
        self.build_errno(self._B)
        if self._requires_grad:
            self._error_reports = {
                **self._error_reports,
                BackwardErrorCode.INVALID_GRAD_NAN: lambda: (
                    "The backward pass produced a 'nan' or 'inf' gradient of the state. The rollout is likely too stiff "
                    "to differentiate over this horizon: shorten it, or decrease the simulation timestep."
                ),
            }

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
        self,
        data: qd.Tensor | qd.Field | qd.Ndarray,
        envs_idx,
        idx,
        *,
        copy: bool | None = None,
        is_batched: bool = True,
    ) -> torch.Tensor: ...

    @overload
    def _read_data(
        self,
        data: Sequence[qd.Tensor | qd.Field | qd.Ndarray],
        envs_idx,
        idx,
        *,
        copy: bool | None = None,
        is_batched: bool = True,
    ) -> list[torch.Tensor]: ...

    def _read_data(self, data, envs_idx, idx, *, copy=None, is_batched=True):
        """
        Read the given slots of one array of the solver, or of several, in the layout of the getters ('qd_to_torch').

        `data` is one array, read as one tensor, or a sequence of them, read as a list of tensors, whatever their kind
        (state, info). `is_batched` states that the arrays hold the environments as their last axis. `idx` selects the
        items once for every environment, or one row per environment (see '_sanitize_io_variables'). `copy` is passed on
        to 'qd_to_torch': a getter returning what it reads asks for a copy, since a getter returns by value. While the
        operations of the solver are recorded, the read is one operation of the rollout whose outputs are copies of the
        whole arrays, which a later step leaves untouched, and from which torch selects the slots (see 'record_read').
        """
        is_single = isinstance(data, (qd.Tensor, qd.Field, qd.Ndarray))
        fields = (data,) if is_single else tuple(data)
        copy = True if self.is_recording else copy
        tensors = [qd_to_torch(field, transpose=is_batched, copy=copy) for field in fields]
        if self.is_recording:
            self.record_read(tuple(tensors), fields)
        mask = indices_to_mask(envs_idx, idx, pair_rows=True) if is_batched else indices_to_mask(idx)
        tensors = [tensor[mask] for tensor in tensors]
        return tensors[0] if is_single else tensors

    def _write_data(
        self,
        data: qd.Tensor | qd.Field | qd.Ndarray,
        values: float | Sequence | np.ndarray | torch.Tensor,
        idx: int | Sequence | np.ndarray | torch.Tensor | None,
        n_items: int,
        idx_name: str,
        envs_idx: int | Sequence[int] | np.ndarray | torch.Tensor | None = None,
        *,
        element_shape: tuple[int, ...] = (),
        dtype: torch.dtype | None = None,
        is_batched: bool = True,
    ) -> None:
        """
        Write the given slots of one array of the items of the solver, from values in the layout of the setters.

        `data` holds the items as its first index, followed by the environments where `is_batched`, and by nothing for a
        parameter shared by every environment, which `envs_idx` then leaves as None. `idx` selects the items once for
        every environment, or one row per environment (see '_sanitize_io_variables'). On Metal, the caller synchronizes
        the torch stream before the next kernel.
        """
        n_envs = self._scene.n_envs
        if gs.use_zerocopy:
            if n_envs == 0 and envs_idx is not None:
                gs.raise_exception("`envs_idx` is not supported for non-parallelized scene.")
            if isinstance(envs_idx, (int, np.integer)) and not -n_envs <= envs_idx < n_envs:
                gs.raise_exception(f"`envs_idx` out of range: {envs_idx} not in [{-n_envs}, {n_envs}).")
            view = qd_to_torch(data, transpose=is_batched, copy=False)
            if not is_batched:
                mask = indices_to_mask(idx)
            elif n_envs == 0:
                mask = (0, *indices_to_mask(idx))
            else:
                mask = indices_to_mask(envs_idx, idx, pair_rows=True)
            assign_indexed_tensor(view, mask, values)
            return
        values, idx, envs_idx = self._sanitize_io_variables(
            values,
            idx,
            n_items,
            idx_name,
            envs_idx,
            element_shape,
            dtype=dtype,
            batched=is_batched,
            batched_idx=is_batched,
        )
        if is_batched:
            kernel_set_values(idx, envs_idx, values, data)
        else:
            kernel_set_info(idx, values, data)

    def add_grad_from_state(self, grads, selection) -> None:
        """
        Add the gradients of the tensors of a queried state to the adjoint of the state of the solver.

        By default, `selection` holds the fields the tensors are, each read whole in the layout of the getters (see
        '_read_data').
        """
        for array, grad in zip(selection, grads):
            # A field with no entry keeps one padding slot, to which no gradient goes
            if grad is not None and grad.numel() > 0:
                qd_add_grad(array, grad)
        if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
            torch.mps.synchronize()

    def reset_grad(self) -> None:
        """Zero the adjoint of the state of the solver, from which a backward pass starts."""

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(
        self, state: SolverCheckpoint, envs_idx: int | Sequence[int] | np.ndarray | torch.Tensor | None = None
    ) -> None:
        """Write back the state 'get_state' returned, notifying the subscribers (see 'CheckpointMixin.set_state')."""
        super().set_state(state, envs_idx)

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
    def force_fields(self) -> ForceFields:
        """The force fields acting on the entities of the solver, in the container its kernels take as a template."""
        return self._force_fields

    def _repr_brief(self):
        repr_str = f"{self.__repr_name__()}: {self._uid}, n_entities: {self.n_entities}"
        return repr_str


@qd.kernel(fastcache=True)
def kernel_set_gravity(tensor: qd.types.ndarray(), envs_idx: qd.types.ndarray(), gravity: qd.Tensor):
    # The qd.Tensor annotation accepts both forms the buffer is allocated in: a field, for a solver whose kernels
    # reach gravity through the solver itself, and an ndarray for every other one (see _build_gravity).
    for i_b_ in range(envs_idx.shape[0]):
        for j in qd.static(range(3)):
            gravity[envs_idx[i_b_]][j] = tensor[i_b_, j]


# The accessors of a solver tell a vector from a scalar by 'is_tensor', a compile-time query of the element type of the
# array they access, which quadrants exposes alike for fields and ndarrays only on an entry.


@qd.kernel(fastcache=True)
def kernel_set_values(
    items_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    items_values: qd.Tensor,
):
    """Set a 3D vector or a scalar of some items of a solver, such as the velocity of its particles or vertices."""
    for i_i_, i_b_ in qd.ndrange(items_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(items_values))):
        i_i = items_idx[i_b_, i_i_]
        i_b = envs_idx[i_b_]
        # The earlier occurrences of a repeated item are -1 (see 'drop_repeated_index')
        if i_i >= 0:
            if qd.static(items_values[i_i, i_b].is_tensor()):
                items_values[i_i, i_b] = qd.Vector(
                    [values[i_b_, i_i_, 0], values[i_b_, i_i_, 1], values[i_b_, i_i_, 2]]
                )
            else:
                items_values[i_i, i_b] = values[i_b_, i_i_]


@qd.kernel(fastcache=True)
def kernel_get_values(
    items_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    items_values: qd.Tensor,
):
    """Get a 3D vector or a scalar of some items of a solver (see 'kernel_set_values')."""
    for i_i_, i_b_ in qd.ndrange(items_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(items_values))):
        i_i = items_idx[i_b_, i_i_]
        i_b = envs_idx[i_b_]
        if qd.static(items_values[i_i, i_b].is_tensor()):
            for k in qd.static(range(3)):
                values[i_b_, i_i_, k] = items_values[i_i, i_b][k]
        else:
            values[i_b_, i_i_] = items_values[i_i, i_b]


@qd.kernel(fastcache=True)
def kernel_set_actu(
    items_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actus: qd.types.ndarray(),
    items_muscle_group: qd.Tensor,
    items_actu: qd.Tensor,
):
    """Set the actuation of some items of a solver between two steps, each one taking that of its muscle group."""
    for i_i_, i_g, i_b_ in qd.ndrange(items_idx.shape[1], actus.shape[2], envs_idx.shape[0]):
        i_i = items_idx[i_b_, i_i_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_values'
        if i_i >= 0:
            if items_muscle_group[i_i] == i_g:
                items_actu[i_i, i_b] = actus[i_b_, i_i_, i_g]


@qd.kernel(fastcache=True)
def kernel_set_info(items_idx: qd.types.ndarray(), values: qd.types.ndarray(), items_values: qd.Tensor):
    """Set a 3D vector or a scalar parameter of some items of a solver, shared by every environment."""
    for i_i_ in range(items_idx.shape[0]):
        i_i = items_idx[i_i_]
        # See 'kernel_set_values'
        if i_i >= 0:
            if qd.static(items_values[i_i].is_tensor()):
                items_values[i_i] = qd.Vector([values[i_i_, 0], values[i_i_, 1], values[i_i_, 2]])
            else:
                items_values[i_i] = values[i_i_]


@qd.kernel(fastcache=True)
def kernel_get_info(items_idx: qd.types.ndarray(), values: qd.types.ndarray(), items_values: qd.Tensor):
    """Get a 3D vector or a scalar parameter of some items of a solver (see 'kernel_set_info')."""
    for i_i_ in range(items_idx.shape[0]):
        i_i = items_idx[i_i_]
        if qd.static(items_values[i_i].is_tensor()):
            for k in qd.static(range(3)):
                values[i_i_, k] = items_values[i_i][k]
        else:
            values[i_i_] = items_values[i_i]


@no_grad
@qd.kernel(fastcache=True)
def kernel_set_time(
    i_substep: int, steps: qd.types.ndarray(), sim_state: SimulatorState, substeps: int, substep_dt: float
):
    """Set the time of the given substep of the step being run in each environment, from the steps it counted.

    `substeps` is the number of substeps of a step and `substep_dt` their interval.
    """
    for i_b in range(steps.shape[0]):
        sim_state.t[i_b] = ((steps[i_b] - 1) * substeps + i_substep) * substep_dt
