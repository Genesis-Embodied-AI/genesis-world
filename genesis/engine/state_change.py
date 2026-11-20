"""
Changes of the state of a trackable, the trackers and subscribers following them, and the tags of the methods making
them (see 'mutates') and of the getters depending on them (see 'depends_on').

The module stands outside of the solvers, so that the entities tag their own methods.
"""

import functools
import inspect
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import torch

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.typing import IndexType
from genesis.utils.misc import indices_to_mask

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


class StateChange(gs.IntEnum):
    """Category of a change of the state of a trackable.

    It names what kind of state changed, never an index space (links, dofs, particles, ...), which differs from one
    trackable to the next. GEOMETRY is the kinematic configuration that places or deforms the world surface (link poses,
    qpos, vertices). DYNAMICS is the velocity state. MODEL is what every later step reads besides the state: the model
    parameters (mass, inertia, friction, gains, limits, gravity) and the constraints attached to the state (particle and
    vertex constraints). VISUAL is the appearance alone, which the physics never reads, such as the visual vertices of a
    visual geom: a scoped render context records it, never a solver.
    """

    GEOMETRY = 0
    DYNAMICS = 1
    MODEL = 2
    VISUAL = 3


# The categories are handled as plain integers below the public interface, which number them from zero so that a change
# log indexes its versions by category, and each integer gives back its category for the consumers.
_ALL_CHANGES = tuple(range(len(StateChange)))
_STATE_CHANGES = tuple(StateChange)

# The selections an axis of a change log keeps per category before counting as changed everywhere (see '_Counters')
_MAX_SELECTIONS = 32


class ChangeAxis(NamedTuple):
    """An axis along which a trackable records where its changes happened, such as its environments or its links."""

    size: int
    """The number of entries of the axis."""
    indices_maps: dict[str, torch.Tensor | None]
    """The entry of each index of the index spaces naming entries of the axis, by name of the argument holding the
    indices, the indices being the entries themselves for None (see 'mutates')."""
    reach: Callable[[torch.Tensor], torch.Tensor] | None = None
    """The entries a change of some entries reaches, set in place in their mask, the entries themselves for None."""


class EnvsAxes(NamedTuple):
    """Where the changes of a trackable recording the environments alone happened (see 'ChangeRecord')."""

    envs: torch.Tensor | None
    """Whether the changes reached each parallel environment of the scene, a single one for a scene without them, None
    for every environment."""


class ChangeRecord(NamedTuple):
    """The categories of the changes of a trackable since a tracker last caught up, and where they happened."""

    changes: frozenset[StateChange]
    """The categories of the changes."""
    axes_is_changed: tuple[torch.Tensor | None, ...]
    """Whether the changes reached each entry of each axis, on the device, None for every entry, as the named tuple
    'axes_cls' of the change log (see 'ChangeLog')."""


class _Counters:
    """The changes of one category per axis since the last one reaching every entry of the axis.

    A change of every entry bumps the host count of the axis and drops the selections recorded before it. Any other
    change appends its selection as given, which a tracker resolves into a mask on the device as it catches up, so
    that a setter does no work on the device. Past a few selections, the axis counts as changed everywhere.
    """

    __slots__ = ("all_entries", "selections")

    def __init__(self, n_axes: int) -> None:
        self.all_entries = [0] * n_axes
        self.selections: list[list[tuple[torch.Tensor | None, IndexType]]] = [[] for _ in range(n_axes)]


class Tracker:
    """The changes of some categories a trackable reports to a consumer, since the consumer last caught up."""

    def __init__(self, log: "ChangeLog", changes: frozenset[int]) -> None:
        self._log = log
        # The version of each category, and per axis the count of changes of every entry and the number of selections,
        # as the tracker last caught up with them (see '_Counters')
        self._seen_versions = {change: log._versions[change] for change in changes}
        self._seen_all_entries = {change: log._counters[change].all_entries.copy() for change in changes}
        self._seen_n_selections = {
            change: [len(selections) for selections in log._counters[change].selections] for change in changes
        }

    def take(self) -> ChangeRecord | None:
        """Return the changes since the tracker last caught up, None if none, and catch up."""
        log = self._log
        n_axes = len(log.axes_size)
        changes: list[int] = []
        # Whether the changes reached every entry of each axis, and the selections they reached otherwise
        axes_is_all = [False] * n_axes
        axes_selections: list[list[tuple[torch.Tensor | None, IndexType]]] = [[] for _ in range(n_axes)]
        for change, seen_version in self._seen_versions.items():
            version = log._versions[change]
            if version == seen_version:
                continue
            self._seen_versions[change] = version
            changes.append(change)
            counters = log._counters[change]
            seen_all_entries, seen_n_selections = self._seen_all_entries[change], self._seen_n_selections[change]
            for i_axis in range(n_axes):
                selections = counters.selections[i_axis]
                if counters.all_entries[i_axis] != seen_all_entries[i_axis]:
                    seen_all_entries[i_axis] = counters.all_entries[i_axis]
                    axes_is_all[i_axis] = True
                elif not axes_is_all[i_axis]:
                    axes_selections[i_axis] += selections[seen_n_selections[i_axis] :]
                seen_n_selections[i_axis] = len(selections)
        if not changes:
            return None
        axes_is_changed = []
        for axis, size, is_all, selections in zip(log.axes, log.axes_size, axes_is_all, axes_selections):
            if is_all:
                axes_is_changed.append(None)
                continue
            # The entries are set in place, a write of a scalar through an index tensor synchronizing the device
            entries_is_changed = torch.zeros(size, dtype=gs.tc_bool, device=gs.device)
            for entries_map, indices_idx in selections:
                (entries_mask,) = indices_to_mask(indices_idx)
                if entries_map is not None:
                    entries_mask = entries_map[entries_mask]
                if not isinstance(entries_mask, torch.Tensor):
                    entries_is_changed[entries_mask] = True
                elif entries_mask.dtype == torch.bool:
                    entries_is_changed |= entries_mask
                else:
                    entries_is_changed.index_put_((entries_mask,), log._true)
            axes_is_changed.append(entries_is_changed if axis.reach is None else axis.reach(entries_is_changed))
        return ChangeRecord(frozenset(_STATE_CHANGES[change] for change in changes), log._axes_cls(*axes_is_changed))


class ChangeLog:
    """The changes of the state of a trackable, which it reports to its consumers.

    A change bumps the versions of its categories, and records where it happened along each axis: every entry as a host
    count, some of them as the selection given, which a tracker resolves on the device as it catches up (see
    '_Counters'). The memory is bounded whatever the number of changes recorded in between.

    Parameters
    ----------
    n_envs : int
        The number of parallel environments of the scene, one for a scene without them, along which every trackable
        records where its changes happened, by the indices of the argument 'envs_idx'.
    axes_cls : type[tuple], optional
        The named tuple holding where the changes of a record happened along each axis, the environments first under
        'envs' (see 'ChangeRecord'). Defaults to 'EnvsAxes'.
    **items_axes : ChangeAxis
        The axes of the items of the trackable, such as its links, by name of their field in 'axes_cls'.
    """

    def __init__(self, n_envs: int, axes_cls: type[tuple] = EnvsAxes, **items_axes: ChangeAxis) -> None:
        self._axes_cls = axes_cls
        axes = dict(envs=ChangeAxis(n_envs, {"envs_idx": None}), **items_axes)
        self.axes = tuple(axes[name] for name in axes_cls._fields)
        self.axes_size = tuple(axis.size for axis in self.axes)
        # The axis and the entry map of each index argument naming entries of an axis
        self._indices_axes = {
            name: (i_axis, entries_map)
            for i_axis, axis in enumerate(self.axes)
            for name, entries_map in axis.indices_maps.items()
        }
        # The version of each category, a host count every change of it bumps
        self._versions = [0] * len(_ALL_CHANGES)
        # The value the trackers set the entries some changes reached with (see 'Tracker.take')
        self._true = torch.ones((), dtype=gs.tc_bool, device=gs.device)
        # The changes of each category some tracker follows, made with its first tracker
        self._counters: list[_Counters | None] = [None] * len(_ALL_CHANGES)
        # The eager subscribers, called back inside the outermost tagged call, whose nested tagged calls accumulate
        # their changes here
        self._subscribers: list[Subscriber] = []
        self._is_mutating = False
        self._mutation_changes: set[int] = set()
        # The versions as each key last marked them clean, none before its first time (see 'is_dirty_since')
        self._keys_versions: dict[Hashable, list[int]] = {}

    def subscribe(self, subscriber: "Subscriber") -> None:
        """Register an eager subscriber, called back after the subscribers registered before it.

        Parameters
        ----------
        subscriber : Subscriber
            The subscriber, called back inside each tagged method recording a change of its categories.
        """
        self._subscribers.append(subscriber)

    def track(self, *changes: StateChange) -> Tracker:
        """Return a tracker of the changes of some categories, caught up with the trackable.

        Parameters
        ----------
        *changes : StateChange
            The categories the tracker reports, all of them if none.
        """
        changes_int = frozenset(map(int, changes)) or frozenset(_ALL_CHANGES)
        for change in changes_int:
            if self._counters[change] is None:
                self._counters[change] = _Counters(len(self.axes))
        return Tracker(self, changes_int)

    def is_dirty_since(self, key: Hashable, *changes: StateChange) -> bool:
        """Whether the trackable changed in some categories since some key last marked it clean.

        It is a host check reading nothing from the device (see 'mark_clean').

        Parameters
        ----------
        key : Hashable
            What was made current, such as the refresh of a gated getter (see 'depends_on'), None for the results of
            the trackable as a whole. A key that never marked the trackable clean makes it dirty.
        *changes : StateChange
            The categories checked, all of them if none.
        """
        key_versions = self._keys_versions.get(key)
        if key_versions is None:
            return True
        return any([self._versions[change] != key_versions[change] for change in changes or _ALL_CHANGES])

    def mark_clean(self, key: Hashable = None) -> None:
        """Record the versions of the trackable as current under some key.

        Parameters
        ----------
        key : Hashable, optional
            What was made current (see 'is_dirty_since'), None for the results of the trackable as a whole, which the
            component computing them marks clean once they are computed. Defaults to None.
        """
        self._keys_versions[key] = self._versions.copy()

    def record(self, *changes: StateChange, **indices: IndexType | None) -> None:
        """Record a change of some categories at some entries of each axis, all of them along an axis no index names.

        A category no tracker follows only bumps its version, its indices left unresolved.

        Parameters
        ----------
        *changes : StateChange
            The categories of the change.
        **indices : IndexType | None
            The entries the change reached, by name of the index space they belong to (see 'ChangeAxis.indices_maps'),
            None naming every entry. They are kept as given, without a copy, and read only when a tracker takes the
            change. The caller must never write them in place afterward: such a write corrupts the record.
        """
        for change in changes:
            self._versions[change] += 1
        tracked_changes = [change for change in changes if self._counters[change] is not None]
        if not tracked_changes:
            return
        axes_selection: list[tuple[torch.Tensor | None, IndexType] | None] = [None] * len(self.axes)
        for name, indices_idx in indices.items():
            if indices_idx is not None:
                i_axis, entries_map = self._indices_axes[name]
                axes_selection[i_axis] = (entries_map, indices_idx)
        for change in tracked_changes:
            counters = self._counters[change]
            for i_axis, selection in enumerate(axes_selection):
                selections = counters.selections[i_axis]
                if selection is None or len(selections) >= _MAX_SELECTIONS:
                    counters.all_entries[i_axis] += 1
                    selections.clear()
                else:
                    selections.append(selection)


class SubscriberCallback(Protocol):
    """The function a subscriber calls back with the category of a change and the indices it happened at.

    The indices are named as the change log records them (see 'ChangeLog.record').
    """

    def __call__(self, change: StateChange, **indices: IndexType | None) -> None: ...


class Subscriber:
    """A callback that a trackable calls inside a tagged method, as a change happens.

    A consumer registers it through 'ChangeLog.subscribe'. It is reserved for a consumer that must react before the
    tagged method returns, such as the interactive viewer showing a setter at once. Any other consumer reads a tracker
    when it chooses (see 'Tracker'), since a callback runs inside every call of its categories and adds its cost to each
    of them.

    Parameters
    ----------
    to : frozenset[StateChange]
        The categories the subscriber is called back for.
    callback : SubscriberCallback
        The function called back with the category of each change and the indices it happened at.
    """

    def __init__(self, to: frozenset[StateChange], callback: SubscriberCallback):
        self.to = to
        self.callback = callback

    def notify(self, change: StateChange, **indices: IndexType | None) -> None:
        """Call back the subscriber for a change at some indices, ignoring any change outside its categories.

        Parameters
        ----------
        change : StateChange
            The category of the change.
        **indices : IndexType | None
            The entries the change reached, named as the change log records them (see 'ChangeLog.record').
        """
        if change in self.to:
            self.callback(_STATE_CHANGES[change], **indices)


class TrackableMixin:
    """A component whose state changes, which it records in its change log for its consumers."""

    # The changes of the state of the trackable, made as it builds
    changelog: ChangeLog | None = None


# What a gate reads the change logs of, from the object its getter belongs to, the object itself if None
TrackablesOf = Callable[[Any], "TrackableMixin | tuple[TrackableMixin, ...]"] | None


class _HookedGetter:
    """A gated getter waiting for its class to exist, to hook the refreshes it waits for.

    Each refresh comes with the trackable whose change log the gate reads (see 'depends_on').
    """

    def __init__(
        self,
        getter: Callable[..., Any] | property,
        refreshes: list[tuple[Callable[[Any], Callable[..., Any]], TrackablesOf]],
    ) -> None:
        self.getter = getter
        self.refreshes = refreshes

    def __set_name__(self, owner: type, name: str) -> None:
        for refresh, on in self.refreshes:
            # Evaluated on the class, the refresh resolves the methods of the class alone, the attributes of an instance
            # (the components it holds) being unknown yet
            method = refresh(owner)
            if method not in _HOOKED_REFRESHES:
                # The refresh is hooked on the class defining it, so that every class inheriting it shares the hook
                (defining_cls,) = (cls for cls in owner.__mro__ if cls.__dict__.get(method.__name__) is method)
                setattr(defining_cls, method.__name__, _hook_refresh(method, on))
        setattr(owner, name, self.getter)


# The refreshes some getter waits for, hooked once whatever the number of getters waiting for them
_HOOKED_REFRESHES: set[Callable[..., Any]] = set()


def _trackables_of(obj: Any, on: TrackablesOf) -> tuple[TrackableMixin, ...]:
    trackables = obj if on is None else on(obj)
    return trackables if isinstance(trackables, tuple) else (trackables,)


def _hook_refresh(method: Callable[..., Any], on: TrackablesOf) -> Callable[..., Any]:
    @functools.wraps(method)
    def hooked(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The versions are read as the refresh starts, so that a change it records itself is not taken as refreshed
        for trackable in _trackables_of(self, on):
            trackable.changelog.mark_clean(hooked)
        return method(self, *args, **kwargs)

    _HOOKED_REFRESHES.add(hooked)
    return hooked


def depends_on(
    *changes: StateChange,
    on: TrackablesOf = None,
    refresh: "Callable[[Any], Callable[[], Any]] | None" = None,
    recover: "bool | Callable[[Any], Any]" = True,
) -> Callable[[Callable[..., Any]], Any]:
    """Gate a getter on the changes of some trackables, so that it delivers its output only while it is current.

    The output is stale after a change of the given categories since it was last made current, and a getter that is not
    stale delivers its output as it is. A getter may stack several gates, the outermost checked first. A gate placed
    above '@property' gates the property.

    Parameters
    ----------
    *changes : StateChange
        The categories whose changes make the output stale, all of them if none.
    on : Callable[[Any], TrackableMixin | tuple[TrackableMixin, ...]] | None, optional
        Return, from the object the getter belongs to, the trackables whose change logs the gate reads, the object
        itself if None. A gate reading the object itself also takes the output as current while nothing changed since
        the object last marked itself clean without a key (see 'ChangeLog.mark_clean'), its results including the
        output, unlike the results of another trackable. Defaults to None.
    refresh : Callable[[Any], Callable[[], Any]] | None, optional
        Return, from the object the getter belongs to or from its class, the method of that class making the output
        current. The output is current while nothing changed since that method last started, whoever called it. None
        for an output no method of the class makes current. Defaults to None.
    recover : bool | Callable[[Any], Any], optional
        How a stale getter recovers. True calls 'refresh' first, then delivers the output, which needs a 'refresh'.
        False delivers None. A function delivers the function applied to the output, which keeps what still holds of
        it, an output withheld whole by an inner gate staying None. Defaults to True.
    """
    # TODO: A refresh returning False recovers nothing, for a refresh that can fail.
    assert recover is not True or refresh is not None, "A getter recovering fully needs a refresh."
    changes_int = tuple(map(int, changes))

    def decorator(method: "Callable[..., Any] | property | _HookedGetter") -> Any:
        refreshes = []
        if isinstance(method, _HookedGetter):
            method, refreshes = method.getter, method.refreshes
        getter = method.fget if isinstance(method, property) else method

        @functools.wraps(getter)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            key = None if refresh is None else refresh(type(self))
            trackables = _trackables_of(self, on)
            is_dirty = any([trackable.changelog.is_dirty_since(key, *changes_int) for trackable in trackables])
            if is_dirty and on is None:
                is_dirty = any([trackable.changelog.is_dirty_since(None, *changes_int) for trackable in trackables])
            if not is_dirty:
                return getter(self, *args, **kwargs)
            if recover is True:
                refresh(self)()
                return getter(self, *args, **kwargs)
            if not callable(recover):
                return None
            # An output an inner gate withheld whole has nothing left to recover
            output = getter(self, *args, **kwargs)
            return output if output is None else recover(output)

        gated = property(wrapper) if isinstance(method, property) else wrapper
        if refresh is not None:
            refreshes = [*refreshes, (refresh, on)]
        return _HookedGetter(gated, refreshes) if refreshes else gated

    return decorator


def mutates(
    *changes: StateChange,
    differentiable: bool = True,
    derived: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a state-mutating method of a solver or of one of its entities with the StateChange categories it produces.

    The outermost tagged call records the changes of every tagged call it makes in the change log of its solver, where
    the arguments naming entries of an axis of the log point (see 'ChangeAxis.indices_maps'), and calls back the eager
    subscribers once (see 'Subscriber'). Untagged methods, reads included, record nothing.

    Parameters
    ----------
    *changes : StateChange
        The categories the method produces. One method may produce several (set_state changes both GEOMETRY and
        DYNAMICS), and an input of the next step such as a control target produces none.
    differentiable : bool, optional
        Whether the inputs of the method carry a gradient. Defaults to True.
    derived : bool, optional
        Whether the method only refreshes quantities derived from the state. Defaults to False.
    """
    triggered = frozenset(map(int, changes))
    is_recorded = StateChange.MODEL not in triggered

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        # The position after 'self' and the default of each argument, which a call fetches from its arguments directly,
        # a binding of the whole signature costing more than the record itself.
        arguments_position_default = {
            parameter.name: (
                i if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD) else None,
                None if parameter.default is parameter.empty else parameter.default,
            )
            for i, parameter in enumerate(tuple(inspect.signature(method).parameters.values())[1:])
        }

        def get_argument(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if name in kwargs:
                return kwargs[name]
            position, default = arguments_position_default[name]
            return args[position] if position is not None and position < len(args) else default

        def run_recording(self: "Solver", *args: Any, **kwargs: Any) -> Any:
            if not triggered:
                return method(self, *args, **kwargs)
            changelog = self.changelog
            if changelog._is_mutating:
                # Nested tagged call: the outermost call states the reach of the whole operation
                changelog._mutation_changes |= triggered
                return method(self, *args, **kwargs)
            changelog._is_mutating = True
            changelog._mutation_changes.clear()
            changelog._mutation_changes |= triggered
            try:
                result = method(self, *args, **kwargs)
            finally:
                changelog._is_mutating = False
            mutation_changes = frozenset(changelog._mutation_changes)
            indices = {
                name: get_argument(name, args, kwargs)
                for name in arguments_position_default
                if name in changelog._indices_axes
            }
            changelog.record(*mutation_changes, **indices)
            for subscriber in changelog._subscribers:
                for changed in mutation_changes & subscriber.to:
                    subscriber.notify(changed, **indices)
            return result

        @functools.wraps(method)
        def wrapper(self: "Solver | Entity", *args: Any, **kwargs: Any) -> Any:
            solver = self._solver if isinstance(self, Entity) else self
            run = run_recording if solver is self else method
            if not is_recorded:
                return run(self, *args, **kwargs)
            # A differentiable rollout records every call writing the state as one operation of its graph, so a
            # differentiable scene takes no shortcut around it. Its inputs are the tensors tracking gradients it takes.
            if solver.is_recording:
                return solver._rollout_graph.run_call(self, run, args, kwargs, differentiable=differentiable)
            result = run(self, *args, **kwargs)
            # A call left unrecorded overwrites the output of the last recorded operation, except a refresh of derived
            # quantities, which leaves them consistent with the state
            if solver.requires_grad and not derived:
                solver._rollout_graph.mark_state_changed()
            return result

        return wrapper

    return decorator
