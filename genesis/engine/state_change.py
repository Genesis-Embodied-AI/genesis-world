"""
Changes of the state of a trackable, recorded per array and per category, the validity of the arrays derived from it,
and the tags of the methods writing it (see 'mutates') and of the getters reading what a step alone computes (see
'depends_on').

Every write of a tracked array is recorded where it happens: a kernel launch through the arrays its source may write
(see 'get_accessed_paths'), a tagged method through the kernels it names, a step at once for every array of its solvers.
An array a launch writes is current by contract, computed from the arrays the launch read. A derived array is stale once
an array of the state it was computed from is written again, and a read of it refreshes it first, whether it is a launch
or a copy to the host, if a refresh of its solver computes it.

The module stands outside of the solvers, so that the entities tag their own methods.
"""

import contextlib
import dataclasses
import functools
import inspect
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Any, NamedTuple, Protocol

import torch

import quadrants as qd
from quadrants.lang import impl

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.typing import IndexType
from genesis.utils.data import ArrayItem, DataItem, DataKind
from genesis.utils.kernel_access import get_accessed_paths
from genesis.utils.misc import indices_to_mask, set_array_read_hook


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
# log indexes its versions by category, and each integer gives back its category for the consumers. The entry after
# them is recorded by every write of the state, whatever its category, and the tracked arrays follow, one entry each.
_ALL_CHANGES = tuple(range(len(StateChange)))
_STATE_CHANGES = tuple(StateChange)
_ANY_STATE = len(StateChange)
_N_FIXED_ENTRIES = _ANY_STATE + 1

# The selections an axis of a change log keeps per category before counting as changed everywhere (see '_Counters')
_MAX_SELECTIONS = 32

# The kinds of the arrays a change log tracks: the state, the model and what is computed from them, the derived arrays
# and the scratch ones (see 'scoped_kind')
_COMPUTED_KINDS = (DataKind.DERIVED, DataKind.SCRATCH)
_TRACKED_KINDS = (DataKind.INFO, DataKind.STATE, *_COMPUTED_KINDS)


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
    """The categories of the changes, empty for a change of the state of no category."""
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

    def is_dirty(self) -> bool:
        """Whether the trackable changed in the categories of the tracker since it last caught up, read on the host."""
        versions = self._log._versions
        return any([versions[change] != version for change, version in self._seen_versions.items()])

    def take(self, *, catch_up: bool = True) -> ChangeRecord | None:
        """Return the changes since the tracker last caught up, None if none.

        Parameters
        ----------
        catch_up : bool, optional
            Whether the tracker catches up, the changes returned being reported no more. Defaults to True.
        """
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
            changes.append(change)
            counters = log._counters[change]
            seen_all_entries, seen_n_selections = self._seen_all_entries[change], self._seen_n_selections[change]
            for i_axis in range(n_axes):
                selections = counters.selections[i_axis]
                if counters.all_entries[i_axis] != seen_all_entries[i_axis]:
                    axes_is_all[i_axis] = True
                elif not axes_is_all[i_axis]:
                    axes_selections[i_axis] += selections[seen_n_selections[i_axis] :]
            if catch_up:
                self._seen_versions[change] = version
                self._seen_all_entries[change] = counters.all_entries.copy()
                self._seen_n_selections[change] = [len(selections) for selections in counters.selections]
        if not changes:
            return None
        axes_is_changed = []
        for axis, size, is_all, selections in zip(log.axes, log.axes_size, axes_is_all, axes_selections):
            if is_all:
                axes_is_changed.append(None)
                continue
            entries_is_changed = torch.zeros(size, dtype=torch.bool, device=gs.device)
            for entries_map, indices_idx in selections:
                entries_is_changed |= _entries_mask(size, indices_idx, entries_map, log._true)
            axes_is_changed.append(entries_is_changed if axis.reach is None else axis.reach(entries_is_changed))
        changes_state = frozenset(_STATE_CHANGES[change] for change in changes if change < _ANY_STATE)
        return ChangeRecord(changes_state, log._axes_cls(*axes_is_changed))


def _entries_mask(size: int, indices_idx: IndexType, entries_map: torch.Tensor | None, true: torch.Tensor):
    """Return the mask of the entries of an axis some indices name, mapped onto the axis by 'entries_map' if any.

    The mask is of torch booleans, which the refreshes tell from indices by its dtype.
    """
    entries_is_changed = torch.zeros(size, dtype=torch.bool, device=gs.device)
    (entries_mask,) = indices_to_mask(indices_idx)
    if entries_map is not None:
        entries_mask = entries_map[entries_mask]
    # The entries are set in place, a write of a scalar through an index tensor synchronizing the device
    if not isinstance(entries_mask, torch.Tensor):
        entries_is_changed[entries_mask] = True
    elif entries_mask.dtype == torch.bool:
        entries_is_changed |= entries_mask
    else:
        entries_is_changed.index_put_((entries_mask,), true)
    return entries_is_changed


class _Production:
    """The derived arrays a launch, a refresh or a step last wrote together, and the state they were computed from.

    The roots are the arrays of the state and of the model the writer read, directly or through the derived arrays it
    read. The production is stale once a root is written again, or while some of its environments were left stale as
    it was made (see 'pending').
    """

    __slots__ = ("roots", "seen_versions", "seen_bulk", "seen_state_total", "trackers", "pending")

    def __init__(self) -> None:
        # The entries of the roots, by change log
        self.roots: dict[ChangeLog, tuple[int, ...]] = {}
        # The versions of the entries of each change log, the count of its steps and of its records of the state as the
        # production was made (see 'ChangeLog._bulk')
        self.seen_versions: dict[ChangeLog, list[int]] = {}
        self.seen_bulk: dict[ChangeLog, int] = {}
        self.seen_state_total: dict[ChangeLog, int] = {}
        # The environments the state was written in since then, one tracker per change log
        self.trackers: dict[ChangeLog, Tracker] = {}
        # The environments the writer left stale, those it skipped while they were stale, None for none
        self.pending: torch.Tensor | None = None

    def snapshot(self, roots: dict["ChangeLog", tuple[int, ...]]) -> None:
        self.roots = roots
        for log in roots:
            self.seen_versions[log] = log._versions.copy()
            self.seen_bulk[log] = log._bulk
            self.seen_state_total[log] = log._state_total
            self.trackers[log] = log.track_state()

    def is_stale(self, excluded: tuple[int, ...] = (), changes: tuple[int, ...] | None = None) -> bool:
        """Whether a root changed since the production was made, counting no write of the 'excluded' entries, and the
        writes of the roots of the given categories alone, all of them for None."""
        if self.pending is not None:
            return True
        for log, roots in self.roots.items():
            if log._state_total == self.seen_state_total[log]:
                continue
            versions, seen_versions, is_bulk = log._versions, self.seen_versions[log], log._is_bulk
            entries_change = log._entries_change
            is_bulk_changed = log._bulk != self.seen_bulk[log]
            if any(
                [
                    (versions[root] != seen_versions[root] or (is_bulk_changed and is_bulk[root]))
                    and root not in excluded
                    and (changes is None or entries_change[root] in changes)
                    for root in roots
                ]
            ):
                return True
        return False

    def stale_envs(self) -> torch.Tensor | None:
        """Return the mask of the environments the production is stale in, on the device, None for all of them."""
        envs_is_stale = self.pending
        for tracker in self.trackers.values():
            record = tracker.take(catch_up=False)
            if record is None:
                continue
            envs_is_changed = record.axes_is_changed[0]
            if envs_is_changed is None:
                return None
            envs_is_stale = envs_is_changed if envs_is_stale is None else envs_is_stale | envs_is_changed
        return envs_is_stale


class ChangeLog:
    """The changes of the state of a trackable, which it reports to its consumers.

    A change bumps the versions of its categories, and records where it happened along each axis: every entry as a host
    count, some of them as the selection given, which a tracker resolves on the device as it catches up (see
    '_Counters'). The memory is bounded whatever the number of changes recorded in between.

    The arrays the trackable registers are tracked one by one (see 'register'): each write of one of them bumps its
    version and records the category of the array, and each derived array knows the production that last wrote it (see
    '_Production'), so that a read refreshes it while stale (see 'refresh').

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
        # The version of each entry, a host count every change of it bumps
        self._versions = [0] * _N_FIXED_ENTRIES
        # The value the trackers set the entries some changes reached with (see 'Tracker.take')
        self._true = torch.ones((), dtype=torch.bool, device=gs.device)
        # The changes of each category some tracker follows, made with its first tracker. The arrays have none.
        self._counters: list[_Counters | None] = [None] * _N_FIXED_ENTRIES
        self._counters[_ANY_STATE] = _Counters(len(self.axes))
        # The eager subscribers, called back inside the outermost tagged call, whose nested tagged calls accumulate
        # their changes here
        self._subscribers: list[Subscriber] = []
        self._is_mutating = False
        self._mutation_changes: set[int] = set()
        # The versions as each key last marked them clean, none before its first time (see 'is_dirty_since')
        self._keys_versions: dict[Hashable, list[int]] = {}
        # The number of writes of the tracked arrays of the state and of the model, and of steps. A step writes every
        # array of the state and every derived one at once, which counts in the version of each of them, 'is_bulk'
        # telling which (see 'record_step').
        self._state_total = 0
        self._bulk = 0
        self._is_bulk = [False] * _N_FIXED_ENTRIES
        for change in (StateChange.GEOMETRY, StateChange.DYNAMICS, _ANY_STATE):
            self._is_bulk[change] = True
        # The kind and the category of each tracked array, by entry, the entries of the arrays of the state and of the
        # model, and the production of each derived array as of the step it last wrote it in (see 'owner')
        self._entries_kind: list[DataKind | None] = [None] * _N_FIXED_ENTRIES
        self._entries_change: list[int | None] = [None] * _N_FIXED_ENTRIES
        self._root_entries: tuple[int, ...] = ()
        self._entries_owner: list[_Production | None] = [None] * _N_FIXED_ENTRIES
        self._entries_owner_bulk: list[int] = [-1] * _N_FIXED_ENTRIES
        # The production of every derived array as a step leaves it, None before the first step, depending on every
        # array of the state and of the model, and those of the derived arrays a refresh computes, depending on what the
        # refresh reads alone, by their roots (see 'register_refreshes')
        self._step_production: _Production | None = None
        self._step_productions: dict[tuple[int, ...], _Production] = {}
        self._entries_step_production: dict[int, _Production] = {}
        # The index tensors naming every environment (see 'register')
        self._envs_idx_all: tuple[torch.Tensor, ...] = ()
        # The number of launches whose writes the change log recorded (see 'mutates')
        self._n_launches_records = 0
        # The production of the derived arrays each launch writes, by key of the launch (see '_produce'), owned by the
        # change log so that the module-level records of the launches reach no change log but weakly
        self._productions: dict[object, _Production] = {}
        # The method of the trackable refreshing each derived array, held weakly so that the change log keeps no
        # trackable alive, and the entries each such method writes
        self._entries_refresh: dict[int, weakref.WeakMethod] = {}
        self._refresh_entries: dict[weakref.WeakMethod, tuple[int, ...]] = {}

    # ------------------------------------------------------------------------------------
    # ------------------------------------- arrays ---------------------------------------
    # ------------------------------------------------------------------------------------

    def register(self, items: Iterable[DataItem], *, envs_idx: Iterable[torch.Tensor] = ()) -> None:
        """Track the arrays of the state, of the model and derived from them among some data items.

        Parameters
        ----------
        items : Iterable[DataItem]
            The data of the trackable, such as 'Solver.data', whose quadrants arrays of kind INFO, DERIVED or SCRATCH
            are tracked, and those of kind STATE declaring the category of their changes, gradients aside, a write of
            an INFO array changing the model. The other arrays of the state, such as the control inputs, the warm
            starts or the error flags, change nothing a derived array is computed from.
        envs_idx : Iterable[torch.Tensor], optional
            The index tensors naming every environment, which a launch reaching them all takes, so that it leaves none
            of them stale whatever they were before, read on the host. Defaults to ().
        """
        self._envs_idx_all = (*self._envs_idx_all, *envs_idx)
        log_ref = weakref.ref(self)
        _install_launch_tracker()
        _LAUNCH_TRACKER.clear()
        _METHODS_WRITES.clear()
        for item in items:
            if not isinstance(item, ArrayItem) or item.kind not in _TRACKED_KINDS or item.name.endswith(".grad"):
                continue
            if item.kind == DataKind.STATE and item.change is None:
                continue
            # A placeholder of data allocated on first use is tracked too, so that the methods allocating it are known to
            # write the state or the model before they do
            if isinstance(item.array, torch.Tensor) or item.array in _TRACKED_ARRAYS:
                continue
            entry = len(self._versions)
            change = StateChange.MODEL if item.kind == DataKind.INFO else item.change
            self._versions.append(0)
            self._counters.append(None)
            self._is_bulk.append(item.kind != DataKind.INFO)
            self._entries_kind.append(item.kind)
            self._entries_change.append(None if change is None else int(change))
            self._entries_owner.append(None)
            self._entries_owner_bulk.append(-1)
            _TRACKED_ARRAYS[item.array] = (log_ref, entry)
            # A kernel taking the array alone receives it unwrapped
            if isinstance(item.array, qd.Tensor):
                _TRACKED_ARRAYS[item.array._unwrap()] = (log_ref, entry)
        self._root_entries = tuple(
            entry for entry, kind in enumerate(self._entries_kind) if kind in (DataKind.INFO, DataKind.STATE)
        )

    def register_refreshes(self, owner: Any) -> None:
        """Make the methods of an object tagged as refreshing derived arrays refresh them as they are read stale.

        Parameters
        ----------
        owner : Any
            The object, whose methods tagged 'mutates(derived=True)' take the environments to refresh, None for all of
            them, and write the derived or scratch arrays the kernels they name may write (see 'get_accessed_paths'). A
            scratch array a refresh computes is refreshed only where a block makes it derived (see 'scoped_kind').
        """
        # The roots each refresh reads, and the derived arrays it reads that another refresh computes
        refreshes_roots, refreshes_reads = {}, {}
        # A method a subclass overrides is registered once, as the subclass defines it
        names = set()
        for cls in type(owner).__mro__:
            for name, attr_value in vars(cls).items():
                method = _DERIVED_METHODS.get(attr_value) if callable(attr_value) else None
                if method is None or name in names:
                    continue
                names.add(name)
                refresh = weakref.WeakMethod(getattr(owner, method.__name__))
                access = get_accessed_paths(method, owner)
                writes = _resolve_paths(access.paths_written, {"self": owner})
                entries = tuple(
                    entry for log, entry in writes if log is self and self._entries_kind[entry] in _COMPUTED_KINDS
                )
                self._refresh_entries[refresh] = entries
                for entry in entries:
                    assert entry not in self._entries_refresh, "Two refreshes compute the same array."
                    self._entries_refresh[entry] = refresh
                used = [entry for log, entry in _resolve_paths(access.paths_used, {"self": owner}) if log is self]
                refreshes_roots[refresh] = {entry for entry in used if self._entries_kind[entry] not in _COMPUTED_KINDS}
                refreshes_reads[refresh] = {entry for entry in used if self._entries_kind[entry] in _COMPUTED_KINDS}
        # A derived array a refresh computes depends on the roots of the refresh, through the derived arrays it reads
        # as well, which a step computing it leaves current until one of them is written (see 'record_step')
        for refresh, entries in self._refresh_entries.items():
            roots, pending, seen = set(), [refresh], set()
            while pending:
                refresh_read = pending.pop()
                if refresh_read in seen:
                    continue
                seen.add(refresh_read)
                roots |= refreshes_roots[refresh_read]
                for entry in refreshes_reads[refresh_read]:
                    if entry in self._entries_refresh:
                        pending.append(self._entries_refresh[entry])
            production = self._step_productions.setdefault(tuple(sorted(roots)), _Production())
            for entry in entries:
                self._entries_step_production[entry] = production

    def owner(self, entry: int) -> _Production | None:
        """Return the production that last wrote a derived or scratch array, None if none did.

        A step leaves its scratch arrays stale, whatever it wrote in them.
        """
        if self._entries_owner_bulk[entry] != self._bulk:
            if self._step_production is None or self._entries_kind[entry] == DataKind.SCRATCH:
                return None
            return self._entries_step_production.get(entry, self._step_production)
        return self._entries_owner[entry]

    def is_stale(
        self, array: Any, *, excluded: tuple[Any, ...] = (), changes: tuple[StateChange, ...] | None = None
    ) -> bool:
        """Whether a tracked array is stale in some environment, read on the host.

        A derived array is stale once an array of the state or of the model it was computed from changed since it was
        computed. An array of the state counts as the last step left it, such as the accelerations the step computes,
        stale once any array of the state or of the model changed since that step.

        Parameters
        ----------
        array : qd.Tensor
            An array of the trackable, stale until the trackable tracks it.
        excluded : tuple[qd.Tensor, ...], optional
            The arrays whose writes do not count, such as the configuration for a quantity of the static links alone.
            Defaults to ().
        changes : tuple[StateChange, ...] | None, optional
            The categories of the writes that count, all of them for None, for a quantity that depends on some of what
            it was computed from alone, such as contact points whatever the velocities. Defaults to None.
        """
        tracked = _tracked_entry(array)
        if tracked is None:
            return True
        entry = tracked[1]
        production = self.owner(entry) if self._entries_kind[entry] in _COMPUTED_KINDS else self._step_production
        if production is None:
            return True
        excluded_entries = tuple(_tracked_entry(excluded_array)[1] for excluded_array in excluded)
        return production.is_stale(excluded_entries, None if changes is None else tuple(map(int, changes)))

    def is_written_since_step(self, *arrays: Any) -> bool:
        """Whether a launch or a refresh wrote some tracked derived or scratch arrays since the last step, on the host."""
        bulk, entries_owner_bulk = self._bulk, self._entries_owner_bulk
        return any([entries_owner_bulk[_tracked_entry(array)[1]] == bulk for array in arrays])

    def refresh(self, array: Any) -> None:
        """Refresh a derived array in the environments it is stale in, through the method computing it (see
        'register_refreshes').

        Parameters
        ----------
        array : qd.Tensor
            A derived array that a refresh computes, left as it is until the trackable tracks it.
        """
        tracked = _tracked_entry(array)
        if tracked is not None:
            self._refresh(self._entries_refresh[tracked[1]])

    def _refresh(self, refresh: weakref.WeakMethod) -> None:
        global _refresh_depth
        envs_is_stale, is_stale = None, False
        # The scratch arrays the refresh writes along the way count only where a block makes them derived
        entries = [
            entry for entry in self._refresh_entries[refresh] if _effective_kind(self, entry) == DataKind.DERIVED
        ]
        for production in {self.owner(entry) for entry in entries}:
            if production is None:
                is_stale, envs_is_stale = True, None
                break
            if production.is_stale():
                envs_is_changed = production.stale_envs()
                if envs_is_changed is None:
                    is_stale, envs_is_stale = True, None
                    break
                envs_is_stale = envs_is_changed if not is_stale else envs_is_stale | envs_is_changed
                is_stale = True
        if not is_stale:
            return
        # A launch another tape records, such as the recorder of a differentiable rollout, is reversed by the autodiff
        # of its kernel, which only the kernels running over every environment allow
        if impl.get_runtime().target_tape not in (None, _LAUNCH_TRACKER):
            envs_is_stale = None
        # The mask is computed on the torch stream, which Metal does not share with the kernels reading it
        if envs_is_stale is not None and gs.backend == gs.metal:
            torch.mps.synchronize()
        # The productions the refresh makes leave no environment stale, since it refreshes every stale one
        _refresh_depth += 1
        try:
            refresh()(envs_is_stale)
        finally:
            _refresh_depth -= 1

    def _produce(
        self,
        production_key: object,
        entries: tuple[int, ...],
        roots: dict["ChangeLog", tuple[int, ...]],
        envs_idx: IndexType | None,
    ) -> None:
        """Make derived arrays current in some environments, all of them for None, as computed from some roots.

        The production is the one of the given key, that of a launch, made on its first use.
        """
        production = self._productions.get(production_key)
        if production is None:
            production = self._productions[production_key] = _Production()
        if envs_idx is None or _refresh_depth > 0 or any([envs_idx is envs_all for envs_all in self._envs_idx_all]):
            production.pending = None
        else:
            # The environments left out keep the staleness they had, whichever production they came from
            envs_is_stale = torch.zeros(self.axes_size[0], dtype=torch.bool, device=gs.device)
            for previous in {self.owner(entry) for entry in entries}:
                if previous is None:
                    envs_is_stale[:] = True
                    break
                if previous.is_stale():
                    envs_is_changed = previous.stale_envs()
                    if envs_is_changed is None:
                        envs_is_stale[:] = True
                        break
                    envs_is_stale |= envs_is_changed
            envs_is_stale &= ~_entries_mask(self.axes_size[0], envs_idx, None, self._true)
            production.pending = envs_is_stale
        production.snapshot(roots)
        for entry in entries:
            self._entries_owner[entry] = production
            self._entries_owner_bulk[entry] = self._bulk

    def invalidate(self) -> None:
        """Make every derived array stale in every environment, as a restore of a state recorded with them stale."""
        self._bulk += 1
        self._state_total += 1
        self._step_production = None

    def _invalidate_entries(self, entries: tuple[int, ...]) -> None:
        """Make some derived arrays stale in every environment, until a refresh or a step computes them."""
        for entry in entries:
            self._entries_owner[entry] = None
            self._entries_owner_bulk[entry] = self._bulk

    def record_step(self) -> None:
        """Record that a step wrote every array of the state and computed every derived one, in every environment."""
        self._bulk += 1
        self._state_total += 1
        for change in (StateChange.GEOMETRY, StateChange.DYNAMICS, _ANY_STATE):
            self._versions[change] += 1
            counters = self._counters[change]
            if counters is not None:
                for i_axis in range(len(self.axes)):
                    counters.all_entries[i_axis] += 1
                    counters.selections[i_axis].clear()
        if self._step_production is None:
            self._step_production = _Production()
        for production, roots in (
            (self._step_production, self._root_entries),
            *[(production, roots) for roots, production in self._step_productions.items()],
        ):
            production.pending = None
            production.snapshot({self: roots})

    def record_arrays(self, arrays: Iterable[Any], envs_idx: IndexType | None = None) -> None:
        """Record writes of some tracked arrays outside any launch or tagged method, such as the restore of a record.

        The derived arrays written are current in the environments written, as computed from the state.

        Parameters
        ----------
        arrays : Iterable[qd.Tensor]
            The arrays written, those the trackable does not track left out.
        envs_idx : IndexType | None, optional
            The environments written, all of them for None. Defaults to None.
        """
        entries = tuple(
            entry for log, entry in [_tracked_entry(array) or (None, None) for array in arrays] if log is self
        )
        entries_derived = tuple(entry for entry in entries if self._entries_kind[entry] == DataKind.DERIVED)
        self._record_entries(entries, envs_idx=envs_idx)
        if entries_derived:
            self._produce(object(), entries_derived, {self: self._root_entries}, envs_idx)

    def _record_entries(self, entries: tuple[int, ...], **indices: IndexType | None) -> None:
        """Bump the versions of some tracked arrays and record the categories of those of the state and of the model.

        A refresh writes the state only to normalize it, such as the quaternions of the configuration, which counts as
        no change of it.
        """
        versions, entries_kind, entries_change = self._versions, self._entries_kind, self._entries_change
        changes = set()
        for entry in entries:
            is_root = entries_kind[entry] not in _COMPUTED_KINDS
            if _refresh_depth > 0 and is_root:
                continue
            versions[entry] += 1
            if is_root:
                changes.add(_ANY_STATE)
                if entries_change[entry] is not None:
                    changes.add(entries_change[entry])
        if changes:
            self._state_total += 1
            self.record(*changes, **indices)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- categories -------------------------------------
    # ------------------------------------------------------------------------------------

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

    def track_state(self) -> Tracker:
        """Return a tracker of every change of the state, whatever its category, caught up with the trackable."""
        return Tracker(self, frozenset((_ANY_STATE,)))

    def is_dirty_since(self, key: Hashable, *changes: StateChange) -> bool:
        """Whether the trackable changed in some categories since some key last marked it clean.

        It is a host check reading nothing from the device (see 'mark_clean').

        Parameters
        ----------
        key : Hashable
            What was made current, such as the readings of a sensor, None for the results of the trackable as a whole.
            A key that never marked the trackable clean makes it dirty.
        *changes : StateChange
            The categories checked, any change of the state if none.
        """
        key_versions = self._keys_versions.get(key)
        if key_versions is None:
            return True
        return any(
            [self._versions[change] != key_versions[change] for change in changes or (_ANY_STATE, *_ALL_CHANGES)]
        )

    def mark_clean(self, key: Hashable = None) -> None:
        """Record the versions of the trackable as current under some key.

        Parameters
        ----------
        key : Hashable, optional
            What was made current (see 'is_dirty_since'), None for the results of the trackable as a whole, which the
            component computing them marks clean once they are computed. Defaults to None.
        """
        self._keys_versions[key] = self._versions[:_N_FIXED_ENTRIES]

    def record(self, *changes: int, **indices: IndexType | None) -> None:
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
        # The categories a tagged method changed, which its outermost call notifies to the subscribers (see 'mutates')
        if self._is_mutating:
            self._mutation_changes.update([change for change in changes if change < _ANY_STATE])
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


# ------------------------------------------------------------------------------------
# ------------------------------ launches and reads ----------------------------------
# ------------------------------------------------------------------------------------

# The change log and the entry of each tracked array, as long as the array lives. The module-level records reach the
# change logs weakly, since a change log reaches its trackable and what it holds through its subscribers.
_TRACKED_ARRAYS: "weakref.WeakKeyDictionary[Any, tuple[weakref.ref[ChangeLog], int]]" = weakref.WeakKeyDictionary()

# The depth of the refreshes running, whose productions leave no environment stale (see 'ChangeLog._refresh')
_refresh_depth = 0

# The kinds the blocks running give to some computed arrays, innermost last, each with the entries it names by change
# log, None naming every computed array (see 'scoped_kind')
_KIND_SCOPES: list[tuple[DataKind, frozenset[tuple["ChangeLog", int]] | None]] = []


def _effective_kind(log: "ChangeLog", entry: int) -> DataKind:
    """Return the kind of a computed array, as the innermost block naming it gives it, else as declared."""
    for kind, entries in reversed(_KIND_SCOPES):
        if entries is None or (log, entry) in entries:
            return kind
    return log._entries_kind[entry]


def _resolve_paths(paths: Iterable[tuple[str, ...]], roots: dict[str, Any]) -> list[tuple[ChangeLog, int]]:
    """Return the tracked arrays some paths reach from the objects their first names stand for.

    A path ending on a struct reaches every array of it, and the names past an array are those of its elements.
    """
    arrays = {}
    for path in paths:
        value = roots.get(path[0])
        for name in path[1:]:
            if _tracked_entry(value) is not None:
                break
            if dataclasses.is_dataclass(value):
                value = getattr(value, name, None)
            else:
                # An attribute of a plain object, such as a struct a solver holds, its properties left unevaluated
                value = getattr(value, "__dict__", {}).get(name)
            if value is None:
                break
        # The arrays of a struct, walked through its members
        pending = [value]
        while pending:
            value = pending.pop()
            entry = _tracked_entry(value)
            if entry is not None:
                arrays[entry] = None
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                pending += [getattr(value, field.name) for field in dataclasses.fields(value)]
    return list(arrays)


def _tracked_entry(value: Any) -> tuple[ChangeLog, int] | None:
    """Return the change log and the entry of a tracked array, None for any other value."""
    try:
        tracked = _TRACKED_ARRAYS.get(value)
    except TypeError:
        # A value no weak reference may point to, such as a number, is no array
        return None
    if tracked is None:
        return None
    log = tracked[0]()
    return None if log is None else (log, tracked[1])


class _LogWrite(NamedTuple):
    """The arrays of one change log a launch may write, and the positions of the arguments naming its indices."""

    log: "weakref.ref[ChangeLog]"
    entries: tuple[int, ...]
    entries_computed: tuple[int, ...]
    indices_positions: tuple[tuple[str, int], ...]


class _LaunchEffect(NamedTuple):
    """What a launch of a kernel on given structs reads and writes among the tracked arrays."""

    writes: tuple[_LogWrite, ...]
    # The derived and scratch arrays the launch reads without writing them
    reads_computed: tuple[tuple["weakref.ref[ChangeLog]", int], ...]
    # The arrays of the state and of the model the launch reads, by change log
    roots: dict["weakref.ref[ChangeLog]", tuple[int, ...]]
    # The key of the production of the computed arrays the launch writes (see 'ChangeLog._produce'), and the position
    # of its argument 'envs_idx'
    production_key: object | None
    i_envs: int | None


class _LaunchTracker:
    """The hook Quadrants calls before every launch of a forward kernel, recording what the launch reads and writes.

    It refreshes the stale derived arrays the launch reads, records the arrays it writes, and makes the derived ones
    current (see 'ChangeLog'). What a launch accesses depends on the kernel and on the structs it takes, so it is
    computed once per kernel and structs, the structs being compared by identity at each launch.
    """

    # The autograd tape of Quadrants also stands for a gradient checker, which this tracker is not
    grad_checker = None

    def __init__(self) -> None:
        # The positions of the struct arguments of each kernel, and the effect of each set of structs it took, the
        # structs held through weak references, the first of which drops its entry once collected
        self._kernels_effects: dict[Any, tuple[tuple[int, ...], list[tuple[tuple[weakref.ref, ...], Any]]]] = {}
        # Whether the launches go untracked, those another tape forwards included, and whether they still refresh the
        # stale derived arrays they read first (see 'untracked_launches')
        self.is_suspended = False
        self.is_refreshing = False

    def clear(self) -> None:
        """Forget the effects computed, which a registration of arrays makes out of date."""
        self._kernels_effects.clear()

    def insert(self, kernel: Any, args: tuple) -> None:
        """Track the launch of a kernel with some arguments, before it runs (the hook of the autograd tape)."""
        if self.is_suspended and not self.is_refreshing:
            return
        kernel_effects = self._kernels_effects.get(kernel)
        if kernel_effects is None:
            # The structs and the arrays the kernel takes, which tell its launches apart
            positions = tuple(
                i
                for i, arg in enumerate(args)
                if dataclasses.is_dataclass(arg) or isinstance(arg, (qd.Field, qd.Ndarray))
            )
            kernel_effects = self._kernels_effects[kernel] = (positions, [])
        positions, structs_effects = kernel_effects
        if not positions:
            return
        for structs_refs, structs_effect in structs_effects:
            for i, struct_ref in zip(positions, structs_refs):
                if args[i] is not struct_ref():
                    break
            else:
                effect = structs_effect
                break
        else:
            effect = _compute_launch_effect(kernel, args, positions)
            structs_entry = [(), effect]
            structs_entry[0] = tuple(
                weakref.ref(args[i], functools.partial(_drop_structs_entry, structs_effects, structs_entry))
                if j == 0
                else weakref.ref(args[i])
                for j, i in enumerate(positions)
            )
            structs_effects.append(structs_entry)
        if effect is None:
            return

        for log_ref, entry in effect.reads_computed:
            log = log_ref()
            refresh = None if log is None else log._entries_refresh.get(entry)
            if refresh is not None and _effective_kind(log, entry) == DataKind.DERIVED:
                production = log.owner(entry)
                if production is None or production.is_stale():
                    log._refresh(refresh)
        # The launches of a refresh are tracked whatever the block running it
        if self.is_suspended and _refresh_depth == 0:
            return
        for log_ref, entries, _, indices_positions in effect.writes:
            log = log_ref()
            if log is not None:
                # A refresh the call triggers writes no array the call itself does
                log._n_launches_records += _refresh_depth == 0
                log._record_entries(entries, **{name: args[i] for name, i in indices_positions})
        if effect.production_key is not None:
            envs_idx = None if effect.i_envs is None else args[effect.i_envs]
            # The roots of the computed arrays read are roots of the arrays computed from them. An array read stale
            # leaves them current all the same, since the analysis may report reads a static branch never makes (see
            # 'get_accessed_paths'), which only a step computes in practice.
            roots = {log_ref(): entries for log_ref, entries in effect.roots.items() if log_ref() is not None}
            for log_ref, entry in effect.reads_computed:
                log = log_ref()
                production = None if log is None else log.owner(entry)
                if production is not None:
                    for root_log, root_entries in production.roots.items():
                        roots[root_log] = tuple(sorted(set(roots.get(root_log, ())) | set(root_entries)))
            for log_ref, _, entries_computed, _ in effect.writes:
                log = log_ref()
                if entries_computed and log is not None:
                    # What a launch leaves in a scratch array is no result, unlike what a refresh computes in it (see
                    # 'scoped_kind')
                    entries_scratch = ()
                    if _refresh_depth == 0:
                        entries_scratch = tuple(
                            entry for entry in entries_computed if _effective_kind(log, entry) == DataKind.SCRATCH
                        )
                    log._invalidate_entries(entries_scratch)
                    entries_produced = tuple(entry for entry in entries_computed if entry not in entries_scratch)
                    if entries_produced:
                        log._produce(effect.production_key, entries_produced, roots, envs_idx)


def _drop_structs_entry(structs_effects: list, structs_entry: list, _: weakref.ref) -> None:
    for i, entry in enumerate(structs_effects):
        if entry is structs_entry:
            del structs_effects[i]
            break


def _compute_launch_effect(kernel: Any, args: tuple, positions: tuple[int, ...]) -> _LaunchEffect | None:
    """Return what a launch of a kernel reads and writes among the tracked arrays, None if it reaches none of them."""
    names = [arg_meta.name for arg_meta in kernel.arg_metas]
    # A kernel of Quadrants itself moves data, such as the copy of an array to the host, which reads it as it stands
    is_moving_data = kernel.func.__module__.startswith("quadrants.")
    owner = args[0] if kernel.is_classkernel else None
    access = get_accessed_paths(kernel.func, owner)
    roots_objects = {names[i]: args[i] for i in positions}
    if owner is not None:
        roots_objects[names[0]] = owner
    used = _resolve_paths(access.paths_used, roots_objects)
    written = set(_resolve_paths(access.paths_written, roots_objects))
    if not used:
        return None
    writes = []
    logs = {log: None for log, _ in used}
    for log in logs:
        entries = tuple(entry for entry_log, entry in written if entry_log is log)
        if not entries:
            continue
        entries_computed = tuple(entry for entry in entries if log._entries_kind[entry] in _COMPUTED_KINDS)
        indices_positions = tuple((name, i) for i, name in enumerate(names) if name in log._indices_axes)
        writes.append(_LogWrite(weakref.ref(log), entries, entries_computed, indices_positions))
    reads_computed = tuple(
        (weakref.ref(log), entry)
        for log, entry in used
        if (log, entry) not in written and log._entries_kind[entry] in _COMPUTED_KINDS and not is_moving_data
    )
    roots = {}
    for log, entry in used:
        if log._entries_kind[entry] not in _COMPUTED_KINDS:
            roots.setdefault(log, []).append(entry)
    roots = {weakref.ref(log): tuple(entries) for log, entries in roots.items()}
    is_producing = any([write.entries_computed for write in writes])
    i_envs = names.index("envs_idx") if "envs_idx" in names else None
    return _LaunchEffect(tuple(writes), reads_computed, roots, object() if is_producing else None, i_envs)


# The tracker of every launch, installed once the first change log tracks arrays
_LAUNCH_TRACKER = _LaunchTracker()


def _install_launch_tracker() -> None:
    runtime = impl.get_runtime()
    # FIXME: quadrants#1000 - the launch hook of Quadrants is reachable only through this private runtime slot
    if runtime.target_tape is None:
        runtime.target_tape = _LAUNCH_TRACKER
        set_array_read_hook(_on_array_read)


@contextlib.contextmanager
def untracked_launches(*, refresh: bool = False) -> Iterator[None]:
    """Run the block without recording what its launches write, for a caller recording it itself, such as a step.

    A step is one operation, whose results are current once it is over, while its launches compute the derived arrays
    the next ones read, which a launch tracked alone would refresh first, the analysis reporting more reads than the
    kernels make (see 'get_accessed_paths'). The hook of another tape, such as the recorder of a differentiable
    rollout, stays in place, its launches forwarded to the tracker going untracked all the same.

    Parameters
    ----------
    refresh : bool, optional
        Whether a launch still refreshes the stale derived arrays it reads first, as a coupler reading the poses of the
        links at the start of a step, the launches of the refresh being tracked. Defaults to False.
    """
    runtime = impl.get_runtime()
    is_installed = runtime.target_tape is _LAUNCH_TRACKER
    is_suspended, is_refreshing = _LAUNCH_TRACKER.is_suspended, _LAUNCH_TRACKER.is_refreshing
    if is_installed and not refresh:
        runtime.target_tape = None
    _LAUNCH_TRACKER.is_suspended, _LAUNCH_TRACKER.is_refreshing = True, refresh
    try:
        yield
    finally:
        _LAUNCH_TRACKER.is_suspended, _LAUNCH_TRACKER.is_refreshing = is_suspended, is_refreshing
        if is_installed:
            runtime.target_tape = _LAUNCH_TRACKER


def _on_array_read(array: Any) -> None:
    """Refresh a derived array a copy to the host or a view reads, if stale and refreshed by its trackable."""
    entry = _tracked_entry(array)
    if entry is None:
        return
    log, i_entry = entry
    refresh = log._entries_refresh.get(i_entry)
    if refresh is not None and _effective_kind(log, i_entry) == DataKind.DERIVED:
        production = log.owner(i_entry)
        if production is None or production.is_stale():
            log._refresh(refresh)


# ------------------------------------------------------------------------------------
# ------------------------------------ the tags --------------------------------------
# ------------------------------------------------------------------------------------

# The methods tagged as refreshing derived arrays (see 'mutates'), by the function the class holds
_DERIVED_METHODS: dict[Callable[..., Any], Callable[..., Any]] = {}

# The arrays each tagged method may write, by change log, by method and bound object (see '_method_writes')
_METHODS_WRITES: "weakref.WeakKeyDictionary[Any, dict[Callable[..., Any], tuple[_LogWrite, ...]]]"
_METHODS_WRITES = weakref.WeakKeyDictionary()


def _method_writes(method: Callable[..., Any], owner: Any) -> tuple[_LogWrite, ...]:
    """Return the tracked arrays a method may write through the kernels it names, by change log."""
    owner_writes = _METHODS_WRITES.setdefault(owner, {})
    writes = owner_writes.get(method)
    if writes is None:
        names = list(inspect.signature(method).parameters)
        written = _resolve_paths(get_accessed_paths(method, owner).paths_written, {names[0]: owner})
        writes = []
        for log in {log: None for log, _ in written}:
            entries = tuple(entry for entry_log, entry in written if entry_log is log)
            indices_positions = tuple((name, i - 1) for i, name in enumerate(names) if name in log._indices_axes)
            writes.append(_LogWrite(weakref.ref(log), entries, (), indices_positions))
        writes = owner_writes[method] = tuple(writes)
    return writes


class scoped_kind:
    """Give some computed arrays another kind, SCRATCH or DERIVED, for the duration of a block.

    What a launch writes in a derived array is current by contract, and a stale derived array is refreshed as it is
    read, while what a launch writes in a scratch array leaves it stale and a read takes it as it stands. A block
    reusing a derived array for something else, such as contacts cleared, makes it scratch, and a getter reading a
    scratch array a refresh computes makes it derived, such as the mass matrix a step uses as its working buffer. What
    a refresh writes is current whatever the kind of the array, and a step leaves every scratch array stale. Blocks
    nest and compose, the innermost one naming an array deciding its kind. As a context manager it takes the arrays,
    or structs standing for every array of them, and as a decorator of a method, the functions returning them from the
    object the method belongs to, the whole method running in the block.

    Parameters
    ----------
    kind : DataKind
        The kind the arrays take, SCRATCH or DERIVED.
    *arrays : qd.Tensor | Data | Callable[[Any], qd.Tensor | Data]
        The arrays or structs, or as a decorator the functions returning them from the object of the method, every
        derived and scratch array if none.
    """

    def __init__(self, kind: DataKind, *arrays: Any) -> None:
        assert kind in _COMPUTED_KINDS
        self._kind = kind
        self._arrays = arrays

    def __enter__(self) -> "scoped_kind":
        entries = None
        if self._arrays:
            entries = frozenset(
                (log, entry)
                for array in self._arrays
                for log, entry in _resolve_paths([("array",)], {"array": array})
                if log._entries_kind[entry] in _COMPUTED_KINDS
            )
        _KIND_SCOPES.append((self._kind, entries))
        return self

    def __exit__(self, *exc_info: Any) -> None:
        _KIND_SCOPES.pop()

    def __call__(self, method: Callable[..., Any]) -> Callable[..., Any]:
        kind, arrays_of = self._kind, self._arrays

        @functools.wraps(method)
        def wrapper(owner: Any, *args: Any, **kwargs: Any) -> Any:
            with scoped_kind(kind, *[array_of(owner) for array_of in arrays_of]):
                return method(owner, *args, **kwargs)

        return wrapper


def depends_on(
    *arrays: Callable[[Any], Any],
    changes: tuple[StateChange, ...] | None = None,
    recover: Callable[[Any], Any] | None = None,
) -> Callable[[Callable[..., Any]], Any]:
    """Gate a getter on derived arrays only a step computes, so that it delivers its output only while they are current.

    A derived array a refresh computes needs no gate, its read refreshing it (see 'ChangeLog.refresh'). A getter may
    stack several gates, the outermost checked first. A gate placed above '@property' gates the property.

    Parameters
    ----------
    *arrays : Callable[[Any], qd.Tensor | tuple[qd.Tensor, ...]]
        Return, from the object the getter belongs to, the tracked arrays its output depends on, an array of the state
        counting as the last step left it (see 'ChangeLog.is_stale').
    changes : tuple[StateChange, ...] | None, optional
        The categories of the writes the output depends on, all of them for None (see 'ChangeLog.is_stale'). Defaults
        to None.
    recover : Callable[[Any], Any] | None, optional
        How a stale getter recovers: None delivers None, and a function delivers the function applied to the output,
        which keeps what still holds of it, an output withheld whole by an inner gate staying None. Defaults to None.
    """

    def decorator(method: "Callable[..., Any] | property") -> Any:
        getter = method.fget if isinstance(method, property) else method

        @functools.wraps(getter)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            is_stale = False
            for array_of in arrays:
                arrays_gated = array_of(self)
                for array in arrays_gated if isinstance(arrays_gated, tuple) else (arrays_gated,):
                    is_stale |= _tracked_entry(array)[0].is_stale(array, changes=changes)
            if not is_stale:
                return getter(self, *args, **kwargs)
            if recover is None:
                return None
            # An output an inner gate withheld whole has nothing left to recover
            output = getter(self, *args, **kwargs)
            return output if output is None else recover(output)

        return property(wrapper) if isinstance(method, property) else wrapper

    return decorator


def mutates(
    *, differentiable: bool = True, derived: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a method of a solver or of one of its entities writing its state, or refreshing what derives from it.

    A method writing the state records the arrays every kernel it names may write, whether it launched them or wrote
    their arrays through zero-copy views instead, so that a zero-copy write lives in a tagged method next to the kernel
    it stands for. The arguments naming entries of an axis of the change log of the arrays say where (see
    'ChangeAxis.indices_maps'). The outermost tagged call notifies the eager subscribers once (see 'Subscriber').

    Parameters
    ----------
    differentiable : bool, optional
        Whether the inputs of the method carry a gradient. Defaults to True.
    derived : bool, optional
        Whether the method only refreshes arrays derived from the state, taking the environments to refresh, None for
        all of them (see 'ChangeLog.register_refreshes'). Defaults to False.
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        names = list(inspect.signature(method).parameters)[1:]

        def run_recording(self: Any, *args: Any, **kwargs: Any) -> Any:
            writes = _method_writes(method, self)
            # The arrays the launches of the call write are recorded as they launch, so that the derived arrays a later
            # launch of the call computes stay current. A call launching no kernel writing a change log took the path
            # writing its arrays through zero-copy views, whose writes the kernels it names stand for as it returns.
            writes = [(log_ref(), *write) for log_ref, *write in writes if log_ref() is not None]
            n_launches_records = [log._n_launches_records for log, *_ in writes]
            # The change logs written and that of the trackable, whose subscribers the outermost tagged call of each
            # calls back once with every category recorded in between, by launches and restores included
            owner_log = (self._solver if isinstance(self, Entity) else self).changelog
            logs = {owner_log: None, **{log: None for log, *_ in writes}}
            logs_outermost = [log for log in logs if not log._is_mutating]
            for log in logs_outermost:
                log._is_mutating = True
                log._mutation_changes.clear()
            arguments = dict(zip(names, args)) | kwargs
            try:
                result = method(self, *args, **kwargs)
                for (log, entries, _, indices_positions), n_records in zip(writes, n_launches_records):
                    if log._n_launches_records == n_records:
                        log._record_entries(entries, **{name: arguments.get(name) for name, _ in indices_positions})
            finally:
                for log in logs_outermost:
                    log._is_mutating = False
            for log in logs_outermost:
                indices = {name: arguments.get(name) for name in log._indices_axes if name in names}
                for subscriber in log._subscribers:
                    for changed in log._mutation_changes & subscriber.to:
                        subscriber.notify(changed, **indices)
            return result

        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            solver = self._solver if isinstance(self, Entity) else self
            run = method if derived else run_recording
            writes = () if derived else _method_writes(method, self)
            is_model = any(
                [
                    log_ref()._entries_kind[entry] == DataKind.INFO
                    for log_ref, entries, _, _ in writes
                    if log_ref() is not None
                    for entry in entries
                ]
            )
            if is_model:
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

        if derived:
            _DERIVED_METHODS[wrapper] = method
        return wrapper

    return decorator
