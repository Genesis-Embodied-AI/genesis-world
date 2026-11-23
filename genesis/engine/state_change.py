"""
Changes of the state of a trackable, recorded per array and per category, the freshness of the arrays derived from it,
and the tags of the methods writing it (see 'mutates').

Every write of a tracked array is recorded where it happens: a kernel launch through the arrays its source may write
(see 'get_accessed_paths'), a tagged method through the kernels it names, a commit at once for every array of a change
log (see 'ChangeLog.commit').

An array a launch writes is fresh by contract, computed from the arrays the launch read, until one of them is written
again (see 'ChangeLog.is_fresh'). An array that a refresh of its trackable computes is refreshable: a read refreshes it
first, whether it is a launch or a copy to the host. A copy to the host of an array that is neither fresh nor
refreshable returns None in its place, and so does a read of a cache of values computed from one (see 'TrackedCache').

The consumers follow the changes through trackers, which report what was modified since they last pulled it (see
'ChangeLog.track').

The module stands outside of the solvers, so that the entities tag their own methods.
"""

import dataclasses
import functools
import inspect
import weakref
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, Protocol

import torch

import quadrants as qd
from quadrants.lang import impl

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.typing import IndexType
from genesis.utils.data import ArrayItem, DataItem, DataKind
from genesis.utils.kernel_access import get_accessed_paths
from genesis.utils.misc import indices_to_mask, qd_to_torch


class StateChange(gs.IntEnum):
    """Category of a change of the state of a trackable.

    It names what kind of state changed, never an index space (links, dofs, particles, ...), which differs from one
    trackable to the next. GEOMETRY is what places or deforms the world surface as it stands: the kinematic
    configuration (qpos, vertices), and the model parameters placing or shaping the surface (the pose of a link in its
    parent, the anchor of a joint, the pose of a geom on its link and its size). DYNAMICS is the velocity state. MODEL
    is what changes the future trajectory alone, leaving the world as it stands: the other model parameters (mass,
    inertia, friction, gains, limits, gravity) and the constraints attached to the state (particle and vertex
    constraints). VISUAL is the appearance alone, which the physics never reads, such as the visual vertices of a visual
    geom or the image each environment shows of a dynamic texture, which a scoped render context records.
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
# The categories a commit changes in every environment, every array of the state being written (see 'ChangeLog.commit')
_COMMIT_CHANGES = frozenset((StateChange.GEOMETRY, StateChange.DYNAMICS, _ANY_STATE))

# The selections of environments a category keeps before counting as changed in every environment (see '_Counters')
_MAX_SELECTIONS = 32

# The kinds of the arrays a change log tracks: the state, the model and what is computed from them, the derived arrays
# and the scratch ones (see 'scoped_kind').
_COMPUTED_KINDS = (DataKind.DERIVED, DataKind.SCRATCH)
_TRACKED_KINDS = (DataKind.INFO, DataKind.STATE, *_COMPUTED_KINDS)


class ChangeRecord:
    """The categories of the changes a tracker pulled, and the environments they reached.

    The environments are resolved on the device as they are first read, so that a consumer reading the categories alone
    does no work on the device.
    """

    __slots__ = ("changes", "_n_envs", "_selections", "_true", "_envs")

    def __init__(
        self,
        changes: frozenset[StateChange],
        n_envs: int,
        selections: list[IndexType] | None,
        true: torch.Tensor,
    ) -> None:
        # The categories of the changes, empty for a change of the state of no category
        self.changes = changes
        # The selections of environments the changes reached, None for every environment (see '_Counters')
        self._n_envs, self._selections, self._true = n_envs, selections, true
        self._envs: torch.Tensor | None = None

    @property
    def envs(self) -> torch.Tensor | None:
        """Whether the changes reached each environment, on the device, None for every environment.

        A scene without parallel environments counts as a single one.
        """
        if self._envs is None and self._selections is not None:
            self._envs = _selections_mask(self._n_envs, self._selections, self._true)
        return self._envs


class _Counters:
    """The changes of one category since the last one reaching every environment.

    A change of every environment stamps the counters with the clock of the change log and drops the selections recorded
    before it. Any other change appends its selection as given, which a tracker resolves into a mask on the device as it
    is read, so that a setter does no work on the device. Past a few selections, every environment counts as changed.
    """

    __slots__ = ("all_clock", "selections")

    def __init__(self) -> None:
        self.all_clock = 0
        self.selections: list[IndexType] = []


class Tracker:
    """The changes of some categories and arrays of a trackable since its consumer last pulled them.

    A tracker reports everything it follows as modified in every environment until its first pull, so that a consumer
    computes its results once before it relies on the tracker.
    """

    def __init__(self, log: "ChangeLog", changes: frozenset[int]) -> None:
        self._log = log
        self._changes = tuple(changes)
        # Whether a commit changes each category in every environment (see 'ChangeLog.commit')
        self._changes_is_committed = tuple(change in _COMMIT_CHANGES for change in self._changes)
        # The clock of the change log, and per category the number of selections, as the tracker last pulled them (see
        # '_Counters'), before any clock value for a tracker never pulled.
        self._seen_clock = -1
        self._seen_n_selections = (0,) * len(self._changes)

    def is_modified(self, *changes: StateChange) -> bool:
        """Whether the trackable changed since the tracker was last pulled, read on the host.

        Parameters
        ----------
        *changes : StateChange
            The categories checked among those of the tracker, all of them if none.
        """
        log, seen_clock = self._log, self._seen_clock
        if log._clock == seen_clock:
            return False
        versions, has_commit_since = log._versions, log._commit_clock > seen_clock
        return any(
            [
                versions[change] > seen_clock or (is_committed and has_commit_since)
                for change, is_committed in zip(self._changes, self._changes_is_committed)
                if not changes or change in changes
            ]
        )

    def pull(self) -> ChangeRecord | None:
        """Return the changes since the tracker was last pulled, None if none, which it reports no more from then on."""
        log, seen_clock = self._log, self._seen_clock
        if log._clock == seen_clock:
            return None
        versions, has_commit_since = log._versions, log._commit_clock > seen_clock
        changes: list[int] = []
        # Whether the changes reached every environment, and the selections they reached otherwise
        is_all, selections = False, []
        for change, is_committed, seen_n_selections in zip(
            self._changes, self._changes_is_committed, self._seen_n_selections
        ):
            is_change_all = is_committed and has_commit_since
            if not is_change_all and versions[change] <= seen_clock:
                continue
            changes.append(change)
            counters = log._counters[change]
            if is_change_all or counters.all_clock > seen_clock:
                is_all = True
            elif not is_all:
                selections += counters.selections[seen_n_selections:]
        self._seen_clock = log._clock
        self._seen_n_selections = tuple(len(log._counters[change].selections) for change in self._changes)
        if not changes:
            return None
        changes_state = frozenset(_STATE_CHANGES[change] for change in changes if change < _ANY_STATE)
        return ChangeRecord(changes_state, log.n_envs, None if is_all else selections, log._true)


class TrackedCache:
    """The zero-copy torch views of some tracked arrays, made once and read only while every one of them is readable.

    A read refreshes first the arrays that are refreshable, and returns None in place of the views while one of them is
    neither fresh nor refreshable, as a copy to the host does (see 'qd_to_torch'). It returns the arrays themselves
    where the backend offers no zero-copy view, for a caller computing from them with a kernel.

    Parameters
    ----------
    *arrays : qd.Tensor
        The arrays, viewed environment first.
    """

    def __init__(self, *arrays: Any) -> None:
        self._arrays = arrays
        self._views: tuple[torch.Tensor, ...] | None = None

    def get(self) -> tuple[torch.Tensor, ...] | tuple[Any, ...] | None:
        """Return the views of the arrays, None while one of them is neither fresh nor refreshable."""
        for array in self._arrays:
            read = _ARRAY_READERS.get(array)
            if read is not None and not read():
                return None
        if not gs.use_zerocopy:
            return self._arrays
        if self._views is None:
            self._views = tuple(qd_to_torch(array, transpose=True, copy=False) for array in self._arrays)
        # The views read the arrays as the last kernel left them once it is done (see 'qd_to_torch')
        elif gs.backend == gs.metal:
            qd.sync()
        return self._views


def _selections_mask(n_envs: int, selections: Iterable[IndexType], true: torch.Tensor) -> torch.Tensor:
    """Return the mask of the environments some selections name (see '_Counters').

    The mask is of torch booleans, which the refreshes tell from indices by its dtype.
    """
    envs_is_changed = torch.zeros(n_envs, dtype=torch.bool, device=gs.device)
    for envs_idx in selections:
        (envs_mask,) = indices_to_mask(envs_idx)
        # The environments are set in place, a write of a scalar through an index tensor synchronizing the device
        if not isinstance(envs_mask, torch.Tensor):
            envs_is_changed[envs_mask] = True
        elif envs_mask.dtype == torch.bool:
            envs_is_changed |= envs_mask
        else:
            envs_is_changed.index_put_((envs_mask,), true)
    return envs_is_changed


class _LogMark:
    """A change log as a production saw it.

    It holds the clock of the change log, its counts of commits and of records of the state (see
    'ChangeLog._n_commits'), and the number of selections of the changes of its state (see 'envs_to_refresh').
    """

    __slots__ = ("clock", "n_commits", "state_total", "n_selections")

    def __init__(self, log: "ChangeLog") -> None:
        self.update(log)

    def update(self, log: "ChangeLog") -> None:
        self.clock, self.n_commits, self.state_total = log._clock, log._n_commits, log._state_total
        self.n_selections = len(log._counters[_ANY_STATE].selections)


class _RootSet:
    """Some roots of a change log, with the clock as their latest write left it (see 'ChangeLog._root_set')."""

    __slots__ = ("clock", "has_committed")

    def __init__(self, clock: int, has_committed: bool) -> None:
        self.clock = clock
        # Whether a commit writes one of the roots, which no clock records (see 'ChangeLog.commit')
        self.has_committed = has_committed


class _Production:
    """The derived arrays a launch, a refresh or a commit last wrote together, and the state they were computed from.

    The roots are the arrays of the state and of the model the writer read, directly or through the derived arrays it
    read. The production is fresh until a root is written again, and in the environments the writer refreshed alone
    while it left some of them pending (see 'pending').
    """

    __slots__ = ("roots", "logs_mark", "pending", "checks")

    def __init__(self) -> None:
        # The entries of the roots, by change log
        self.roots: dict[ChangeLog, tuple[int, ...]] = {}
        # Each change log as the production was made
        self.logs_mark: dict[ChangeLog, _LogMark] = {}
        # The environments to refresh that the writer skipped, None for none
        self.pending: torch.Tensor | None = None
        # The check of each selection of the roots, by excluded entries and categories (see 'check')
        self.checks: dict[tuple, Callable[[], bool]] = {}

    def bind(self, roots: dict["ChangeLog", tuple[int, ...]], logs_mark: dict["ChangeLog", _LogMark]) -> None:
        # A writer reads the same arrays from one production to the next and updates its marks in place, so the checks
        # resolved for them are kept until they differ (see 'check').
        if roots != self.roots or logs_mark is not self.logs_mark:
            self.roots, self.logs_mark = roots, logs_mark
            self.checks.clear()

    def check(self, excluded: tuple[int, ...] = (), changes: tuple[int, ...] | None = None) -> Callable[[], bool]:
        """Return the function telling whether a root changed since the production was made (see 'ChangeLog.is_fresh').

        The writes of the excluded entries do not count, nor those of the roots outside the given categories, all of
        them counting for None. The selection of the roots, the root sets it reaches and the marks of their change logs
        are resolved once, since the structure of the change logs stays fixed between two registrations of arrays, so
        that the function reads a few counters alone.
        """
        key = (excluded, changes)
        check = self.checks.get(key)
        if check is not None:
            return check
        logs_root_set_mark = tuple(
            (
                log,
                log._root_set(
                    tuple(
                        root
                        for root in roots
                        if root not in excluded and (changes is None or log._entries_change[root] in changes)
                    )
                ),
                self.logs_mark[log],
            )
            for log, roots in self.roots.items()
        )
        # A production of a single change log, the common case, reads its counters without the loop, every substep
        # asking it several times
        if len(logs_root_set_mark) == 1:
            ((log, root_set, log_mark),) = logs_root_set_mark

            def is_changed() -> bool:
                if self.pending is not None:
                    return True
                if log._state_total == log_mark.state_total:
                    return False
                return root_set.clock > log_mark.clock or (
                    root_set.has_committed and log._n_commits != log_mark.n_commits
                )

        else:

            def is_changed() -> bool:
                if self.pending is not None:
                    return True
                for log, root_set, log_mark in logs_root_set_mark:
                    if log._state_total != log_mark.state_total and (
                        root_set.clock > log_mark.clock
                        or (root_set.has_committed and log._n_commits != log_mark.n_commits)
                    ):
                        return True
                return False

        self.checks[key] = is_changed
        return is_changed

    def envs_to_refresh(self) -> torch.Tensor | None:
        """Return the mask of the environments to refresh, on the device, None for all of them.

        They are those where a root changed since the production was made, and those the writer left pending.
        """
        envs_to_refresh = self.pending
        for log, log_mark in self.logs_mark.items():
            if log._state_total == log_mark.state_total:
                continue
            counters = log._counters[_ANY_STATE]
            if log._commit_clock > log_mark.clock or counters.all_clock > log_mark.clock:
                return None
            selections = counters.selections[log_mark.n_selections :]
            if not selections:
                continue
            envs_is_changed = _selections_mask(log.n_envs, selections, log._true)
            envs_to_refresh = envs_is_changed if envs_to_refresh is None else envs_to_refresh | envs_is_changed
        return envs_to_refresh


class ChangeLog:
    """The changes of the state of a trackable, which it reports to its consumers.

    A change bumps the versions of its categories, and records the environments it reached: every environment as a host
    count, some of them as the selection given, which a tracker resolves on the device as it is read (see '_Counters').
    The memory is bounded whatever the number of changes recorded in between.

    The arrays the trackable registers are tracked one by one (see 'register_arrays'): each write of one of them bumps
    its version and records the category of the array, and each derived array knows the production that last wrote it
    (see '_Production'), so that a read refreshes it when it is not fresh (see 'refresh').

    Parameters
    ----------
    n_envs : int
        The number of parallel environments of the scene, one for a scene without them, which the indices of the
        argument 'envs_idx' of a write name.
    envs_idx : Iterable[torch.Tensor], optional
        The index tensors naming every environment, which a launch reaching them all takes, so that it leaves every one
        of them fresh whatever they were before, read on the host. Defaults to ().
    run_refresh : Callable[[Callable[[torch.Tensor | None], None], torch.Tensor | None], None] | None, optional
        The function running a refresh on the environments to refresh, such as the trackable recording it as one
        operation of a differentiable rollout. None runs it as it is. Defaults to None.
    """

    def __init__(
        self,
        n_envs: int,
        *,
        envs_idx: Iterable[torch.Tensor] = (),
        run_refresh: Callable[[Callable[[torch.Tensor | None], None], torch.Tensor | None], None] | None = None,
    ) -> None:
        self.n_envs = n_envs
        self._run_refresh = run_refresh
        # The version of each entry, the value of the clock of the change log as it last changed, which every change
        # advances, so that a production compares the versions of its roots with the clock as it was made. A commit
        # stamps its own clock alone, which stands for the versions of the categories it changes (see 'commit').
        self._versions = [0] * _N_FIXED_ENTRIES
        self._clock = 0
        self._commit_clock = 0
        # The value the trackers set the environments some changes reached with (see 'ChangeRecord.envs')
        self._true = torch.ones((), dtype=torch.bool, device=gs.device)
        # The changes of each category some tracker follows, made with its first tracker. The arrays have none.
        self._counters: list[_Counters | None] = [None] * _N_FIXED_ENTRIES
        self._counters[_ANY_STATE] = _Counters()
        # The selections of the categories a commit changes, which it drops (see 'commit')
        self._commit_selections = [self._counters[_ANY_STATE].selections]
        # The eager subscribers, called back inside the outermost tagged call, whose nested tagged calls accumulate
        # their changes here.
        self._subscribers: list[Subscriber] = []
        # The function telling whether each tracked array is fresh, by array and by the arrays whose writes do not
        # count, resolved as it is first asked until the next registration (see 'is_fresh').
        self._arrays_is_fresh: dict[tuple[Any, tuple[Any, ...]], Callable[[], bool]] = {}
        self._is_mutating = False
        self._mutation_changes: set[int] = set()
        # The number of writes of the tracked arrays of the state and of the model, and of commits. A commit writes
        # every array of the state and every derived one at once, which counts in the version of each of them,
        # '_entries_is_committed' telling which (see 'commit').
        self._state_total = 0
        self._n_commits = 0
        self._entries_is_committed = [False] * _N_FIXED_ENTRIES
        for change in _COMMIT_CHANGES:
            self._entries_is_committed[change] = True
        # The kind and the category of each tracked array by entry, the entries of the arrays of the state and of the
        # model, and the production of each derived array with the count of commits at its write (see '_production_of').
        self._entries_kind: list[DataKind | None] = [None] * _N_FIXED_ENTRIES
        self._entries_change: list[int | None] = [None] * _N_FIXED_ENTRIES
        self._entries_invalidated_by: list[tuple[int, ...] | None] = [None] * _N_FIXED_ENTRIES
        self._root_entries: tuple[int, ...] = ()
        self._entries_owner: list[_Production | None] = [None] * _N_FIXED_ENTRIES
        self._entries_owner_commit: list[int] = [-1] * _N_FIXED_ENTRIES
        # The root sets of the productions by their entries, and the root sets each entry belongs to (see '_root_set')
        self._root_sets: dict[tuple[int, ...], _RootSet] = {}
        self._entries_root_sets: list[list[_RootSet]] = [[] for _ in range(_N_FIXED_ENTRIES)]
        # The production of every derived array as a commit leaves it, None before the first commit, depending on every
        # array of the state and of the model, and those of the derived arrays a refresh computes, depending on what the
        # refresh reads alone, by their roots (see 'register_refresh').
        self._commit_production: _Production | None = None
        self._commit_productions: dict[tuple[int, ...], _Production] = {}
        # The change log as the last commit left it, which every production of a commit shares, updated in place
        self._commit_mark = _LogMark(self)
        self._commit_logs_mark: dict[ChangeLog, _LogMark] = {self: self._commit_mark}
        self._entries_commit_production: dict[int, _Production] = {}
        # The index tensors naming every environment, which a launch over all of them takes (see '_produce')
        self._envs_idx_all = tuple(envs_idx)
        # The number of launches whose writes the change log recorded (see 'mutates')
        self._n_launches_records = 0
        # The production of the derived arrays each launch writes, by key of the launch (see '_produce'), owned by the
        # change log so that the module-level records of the launches reach no change log but weakly.
        self._productions: dict[object, _Production] = {}
        # The refresh computing each derived array, the entries each refresh writes, and the roots and the computed
        # arrays it reads (see 'register_refresh'). Only the change log holds the refreshes, which capture the
        # trackable, so that the module-level records of the launches and the reads reach no trackable.
        self._entries_refresh: dict[int, Callable[[torch.Tensor | None], None]] = {}
        self._refresh_entries: dict[Callable[[torch.Tensor | None], None], tuple[int, ...]] = {}
        self._refreshes_roots: dict[Callable[[torch.Tensor | None], None], set[int]] = {}
        self._refreshes_reads: dict[Callable[[torch.Tensor | None], None], set[int]] = {}
        # The arrays the change log tracks, which leave the module-level records as it is collected, and those of each
        # entry (see 'register_arrays').
        self._arrays: list[Any] = []
        self._entries_arrays: dict[int, tuple[Any, ...]] = {}
        weakref.finalize(self, _untrack_arrays, self._arrays)

    # ------------------------------------------------------------------------------------
    # ------------------------------------- arrays ---------------------------------------
    # ------------------------------------------------------------------------------------

    def register_arrays(self, items: Iterable[DataItem]) -> None:
        """Track the arrays of the state, of the model and derived from them among some data items.

        An array already tracked is skipped, so that a component tracking its arrays as it builds them leaves the
        registration of all the data of its trackable valid.

        Parameters
        ----------
        items : Iterable[DataItem]
            The data of the trackable, such as 'Solver.data', whose quadrants arrays of kind INFO, DERIVED or SCRATCH
            are tracked, and those of kind STATE declaring the category of their changes, gradients aside, a write of
            an INFO array changing the model. The other arrays of the state, such as the control inputs, the warm
            starts or the error flags, are left untracked: no refresh reads them, and what a commit delivers from them
            stays delivered after a write of them, until the next commit.
        """
        self._arrays_is_fresh.clear()
        log_ref = weakref.ref(self)
        _install_launch_tracker()
        _LAUNCH_TRACKER.clear()
        for owners_writes in _METHODS_WRITES:
            owners_writes.clear()
        for item in items:
            if not isinstance(item, ArrayItem) or item.kind not in _TRACKED_KINDS or item.name.endswith(".grad"):
                continue
            if item.kind == DataKind.STATE and item.change is None:
                continue
            # A torch tensor, such as the step counts of the simulator, is no quadrants array, and an array several
            # items reach is tracked once.
            if isinstance(item.array, torch.Tensor) or item.array in _TRACKED_ARRAYS:
                continue
            # A placeholder of data allocated on first use is tracked too, so that the methods allocating it are known
            # to write the state or the model before they do.
            entry = len(self._versions)
            # A write of the model changes it, unless its field places or shapes the surface (see 'StateChange')
            change = StateChange.MODEL if item.kind == DataKind.INFO and item.change is None else item.change
            self._versions.append(0)
            self._counters.append(None)
            self._entries_is_committed.append(item.kind != DataKind.INFO)
            self._entries_kind.append(item.kind)
            self._entries_change.append(None if change is None else int(change))
            self._entries_invalidated_by.append(
                None if item.invalidated_by is None else tuple(map(int, item.invalidated_by))
            )
            self._entries_owner.append(None)
            self._entries_owner_commit.append(-1)
            self._entries_root_sets.append([])
            # A kernel taking the array alone receives it unwrapped
            arrays = (item.array, item.array._unwrap()) if isinstance(item.array, qd.Tensor) else (item.array,)
            for array in arrays:
                _TRACKED_ARRAYS[array] = (log_ref, entry)
            self._arrays += arrays
            self._entries_arrays[entry] = arrays
        self._root_entries = tuple(
            entry for entry, kind in enumerate(self._entries_kind) if kind in (DataKind.INFO, DataKind.STATE)
        )
        if self._commit_production is not None:
            self._commit_production.bind({self: self._root_entries}, self._commit_logs_mark)
        self._set_readers()

    def register_refresh(self, refresh: Callable[[torch.Tensor | None], None]) -> None:
        """Make a function computing derived arrays refresh them as they are read.

        Parameters
        ----------
        refresh : Callable[[torch.Tensor | None], None]
            The function, taking the environments to refresh as indices or as a mask, None for all of them. It computes
            the derived or scratch arrays its kernels may write through the objects it captures or a partial binds to
            its leading parameters (see 'get_accessed_paths'), a wrapper counting as the function it wraps. It refreshes
            a scratch array only where a block makes it derived (see 'scoped_kind').
        """
        fn, args = inspect.unwrap(refresh), ()
        if isinstance(fn, functools.partial):
            fn, args = fn.func, fn.args
        objects = dict(zip(fn.__code__.co_varnames, args))
        objects.update(zip(fn.__code__.co_freevars, [cell.cell_contents for cell in fn.__closure__ or ()]))
        access = get_accessed_paths(fn)
        entries = tuple(
            entry
            for log, entry in _resolve_paths(access.paths_written, objects)
            if log is self and self._entries_kind[entry] in _COMPUTED_KINDS
        )
        self._refresh_entries[refresh] = entries
        for entry in entries:
            assert entry not in self._entries_refresh, "Two refreshes compute the same array."
            self._entries_refresh[entry] = refresh
        used = [entry for log, entry in _resolve_paths(access.paths_used, objects) if log is self]
        self._refreshes_roots[refresh] = {entry for entry in used if self._entries_kind[entry] not in _COMPUTED_KINDS}
        self._refreshes_reads[refresh] = {entry for entry in used if self._entries_kind[entry] in _COMPUTED_KINDS}
        # A derived array a refresh computes depends on the roots of the refresh, through the derived arrays it reads
        # as well, computed by the refreshes registered so far, which a commit leaves fresh until one of them is written
        # (see 'commit').
        for refresh, entries in self._refresh_entries.items():
            roots, pending, refreshes_visited = set(), [refresh], set()
            while pending:
                refresh_read = pending.pop()
                if refresh_read in refreshes_visited:
                    continue
                refreshes_visited.add(refresh_read)
                roots |= self._refreshes_roots[refresh_read]
                for entry in self._refreshes_reads[refresh_read]:
                    if entry in self._entries_refresh:
                        pending.append(self._entries_refresh[entry])
            roots = tuple(sorted(roots))
            production = self._commit_productions.get(roots)
            if production is None:
                production = self._commit_productions[roots] = _Production()
                production.bind({self: roots}, self._commit_logs_mark)
            for entry in entries:
                self._entries_commit_production[entry] = production
        self._set_readers()

    def _set_readers(self) -> None:
        """Resolve the reader of every derived and scratch array, anew at each registration of arrays or refreshes."""
        for entry, arrays in self._entries_arrays.items():
            if self._entries_kind[entry] in _COMPUTED_KINDS:
                read = self._reader(entry)
                for array in arrays:
                    _ARRAY_READERS[array] = read

    def _reader(self, entry: int) -> Callable[[], bool]:
        """Return the function a read of a derived or scratch array calls first (see '_ARRAY_READERS').

        It refreshes the array if it is not fresh and a refresh of its trackable computes it, and tells whether the read
        goes on, which it does for a refreshable array alone: fresh, or computed by a refresh. A read within a block
        whose launches go untracked finds the arrays as the block left them (see 'untracked_launches').
        """
        log_ref, launch_tracker = weakref.ref(self), _LAUNCH_TRACKER
        is_scratch = self._entries_kind[entry] == DataKind.SCRATCH
        has_refresh, invalidated_by = entry in self._entries_refresh, self._entries_invalidated_by[entry]

        def read() -> bool:
            if launch_tracker.is_suspended and not launch_tracker.is_refreshing:
                return True
            log = log_ref()
            if log is None:
                return True
            if _KIND_SCOPES:
                if _effective_kind(log, entry) != DataKind.DERIVED:
                    return True
            elif is_scratch:
                return True
            production = log._production_of(entry)
            if has_refresh:
                if production is None or production.check()():
                    log._refresh(log._entries_refresh[entry])
                return True
            return production is not None and not production.check((), invalidated_by)()

        return read

    def _root_set(self, roots: tuple[int, ...]) -> _RootSet:
        """Return the root set of some roots, made on its first request and kept up to date by every write since."""
        root_set = self._root_sets.get(roots)
        if root_set is None:
            root_set = self._root_sets[roots] = _RootSet(
                max([self._versions[root] for root in roots], default=0),
                any([self._entries_is_committed[root] for root in roots]),
            )
            for root in roots:
                self._entries_root_sets[root].append(root_set)
        return root_set

    def _production_of(self, entry: int) -> _Production | None:
        """Return the production that last wrote a derived or scratch array, None if none did.

        A commit leaves no scratch array fresh, whatever was written in them.
        """
        if self._entries_owner_commit[entry] != self._n_commits:
            if self._commit_production is None or self._entries_kind[entry] == DataKind.SCRATCH:
                return None
            return self._entries_commit_production.get(entry, self._commit_production)
        return self._entries_owner[entry]

    def is_fresh(self, array: Any, *, excluded: tuple[Any, ...] = ()) -> bool:
        """Whether a tracked array is fresh in every environment, read on the host.

        A derived array is fresh until a write of a category it is invalidated by (see 'field'), every category by
        default, reaches an array of the state or of the model it was computed from. A derived array that a refresh
        computes follows every such array the refresh reads, whatever its category. An array of the state counts as
        the last commit left it, such as the accelerations, fresh until any array of the state or of the model is
        written. The answer is resolved as it is first asked, until another registration of arrays changes the
        structure of the change log (see 'register_arrays'), so that a caller asking often, such as a substep, reads a
        few counters alone.

        Parameters
        ----------
        array : qd.Tensor
            An array of the trackable, never fresh until the trackable tracks it.
        excluded : tuple[qd.Tensor, ...], optional
            The arrays whose writes do not count, such as the configuration for a quantity of the static links alone.
            Defaults to ().
        """
        key = (array, excluded)
        is_array_fresh = self._arrays_is_fresh.get(key)
        if is_array_fresh is not None:
            return is_array_fresh()
        tracked = _tracked_entry(array)
        if tracked is None:
            return False
        entry = tracked[1]
        excluded_entries = tuple([_tracked_entry(excluded_array)[1] for excluded_array in excluded])
        changes = None if entry in self._entries_refresh else self._entries_invalidated_by[entry]
        owners, owners_commit = self._entries_owner, self._entries_owner_commit
        commit_production = self._entries_commit_production.get(entry)
        is_scratch = self._entries_kind[entry] == DataKind.SCRATCH
        if self._entries_kind[entry] not in _COMPUTED_KINDS:

            def is_array_fresh() -> bool:
                production = self._commit_production
                return production is not None and not production.check(excluded_entries, changes)()

        else:
            # The function inlines '_production_of' and the lookup of the cached check of the production, since a
            # substep asks it several times.
            def is_array_fresh() -> bool:
                if owners_commit[entry] != self._n_commits:
                    production = self._commit_production
                    if production is None or is_scratch:
                        return False
                    if commit_production is not None:
                        production = commit_production
                else:
                    production = owners[entry]
                    if production is None:
                        return False
                check = production.checks.get((excluded_entries, changes))
                if check is None:
                    check = production.check(excluded_entries, changes)
                return not check()

        self._arrays_is_fresh[key] = is_array_fresh
        return is_array_fresh()

    def refresh(self, array: Any) -> None:
        """Refresh a derived array where it is not fresh, through the function computing it (see 'register_refresh').

        Parameters
        ----------
        array : qd.Tensor
            A derived array that a refresh computes, left as it is until the trackable tracks it.
        """
        tracked = _tracked_entry(array)
        if tracked is not None:
            self._refresh(self._entries_refresh[tracked[1]])

    def _refresh(self, refresh: Callable[[torch.Tensor | None], None]) -> None:
        global _refresh_depth
        envs_to_refresh, is_refreshing = None, False
        # The scratch arrays the refresh writes along the way count only where a block makes them derived
        entries = [
            entry for entry in self._refresh_entries[refresh] if _effective_kind(self, entry) == DataKind.DERIVED
        ]
        for production in {self._production_of(entry) for entry in entries}:
            if production is None:
                is_refreshing, envs_to_refresh = True, None
                break
            if production.check()():
                envs_is_changed = production.envs_to_refresh()
                if envs_is_changed is None:
                    is_refreshing, envs_to_refresh = True, None
                    break
                envs_to_refresh = envs_is_changed if not is_refreshing else envs_to_refresh | envs_is_changed
                is_refreshing = True
        if not is_refreshing:
            return
        # A launch another tape records, such as the recorder of a differentiable rollout, is reversed by the autodiff
        # of its kernel, which only the kernels running over every environment allow.
        if impl.get_runtime().target_tape not in (None, _LAUNCH_TRACKER):
            envs_to_refresh = None
        # The mask is computed on the torch stream, which Metal does not share with the kernels reading it
        if envs_to_refresh is not None and gs.backend == gs.metal:
            torch.mps.synchronize()
        # The productions the refresh makes leave no environment pending, since it refreshes every one of them it must
        _refresh_depth += 1
        try:
            if self._run_refresh is None:
                refresh(envs_to_refresh)
            else:
                self._run_refresh(refresh, envs_to_refresh)
        finally:
            _refresh_depth -= 1

    def _produce(
        self,
        production_key: object,
        entries: tuple[int, ...],
        roots: dict["ChangeLog", tuple[int, ...]],
        envs_idx: IndexType | None,
    ) -> None:
        """Make derived arrays fresh in some environments, all of them for None, as computed from some roots.

        The production is the one of the given key, that of a launch, made on its first use.
        """
        production = self._productions.get(production_key)
        if production is None:
            production = self._productions[production_key] = _Production()
        if envs_idx is None or _refresh_depth > 0 or any([envs_idx is envs_all for envs_all in self._envs_idx_all]):
            production.pending = None
        else:
            # The environments left out stay to refresh if they were, whichever production they came from
            envs_to_refresh = torch.zeros(self.n_envs, dtype=torch.bool, device=gs.device)
            for previous in {self._production_of(entry) for entry in entries}:
                if previous is None:
                    envs_to_refresh[:] = True
                    break
                if previous.check()():
                    envs_is_changed = previous.envs_to_refresh()
                    if envs_is_changed is None:
                        envs_to_refresh[:] = True
                        break
                    envs_to_refresh |= envs_is_changed
            envs_to_refresh &= ~_selections_mask(self.n_envs, (envs_idx,), self._true)
            production.pending = envs_to_refresh
        # The marks of a production are updated in place, which keeps its checks resolved (see '_Production.bind')
        logs_mark = production.logs_mark
        if logs_mark.keys() == roots.keys():
            for log, log_mark in logs_mark.items():
                log_mark.update(log)
        else:
            logs_mark = {log: _LogMark(log) for log in roots}
        production.bind(roots, logs_mark)
        for entry in entries:
            self._entries_owner[entry] = production
            self._entries_owner_commit[entry] = self._n_commits

    def restore(self, is_modified: bool) -> None:
        """Record the restore of a state, with the derived arrays a commit delivers as they were recorded.

        The scratch arrays hold what the run before the restore left in them, until a refresh computes them anew.

        Parameters
        ----------
        is_modified : bool
            Whether the state was recorded modified since its last commit (see 'is_modified'), its derived arrays being
            neither fresh nor its last commit standing again then.
        """
        if is_modified:
            self._n_commits += 1
            self._state_total += 1
            self._commit_production = None
        else:
            self._mark_committed()
        # A scratch array holds what the run before the restore left in it, such as the mass matrix of a sleeping tree,
        # which a tracker of the array reports as a write.
        entries_scratch = tuple(entry for entry, kind in enumerate(self._entries_kind) if kind == DataKind.SCRATCH)
        self._invalidate_entries(entries_scratch)
        self._record(*entries_scratch)

    def _invalidate_entries(self, entries: tuple[int, ...]) -> None:
        """Leave some computed arrays in need of a refresh in every environment, until a refresh or a commit."""
        for entry in entries:
            self._entries_owner[entry] = None
            self._entries_owner_commit[entry] = self._n_commits

    @property
    def is_committed(self) -> bool:
        """Whether nothing was recorded since the last commit, which leaves every derived array fresh."""
        return self._clock == self._commit_clock and self._commit_production is not None

    def is_modified(self, *changes: StateChange) -> bool:
        """Whether a change of some categories was recorded since the last commit, read on the host.

        A restore of a state recorded unmodified counts as a commit (see 'restore'), and a refresh changes no category.

        Parameters
        ----------
        *changes : StateChange
            The categories checked, all of them if none.
        """
        clock, versions = self._commit_mark.clock, self._versions
        return any([versions[change] > clock for change in changes or _ALL_CHANGES])

    def commit(self) -> None:
        """Record that every array of the state was written and every derived one computed, in every environment.

        The derived arrays are fresh from then on, until an array they are computed from is written. The trackers of
        the geometry and of the dynamics report the commit as a change of every environment.
        """
        # A step commits at every substep, so the commit writes the fields of the change log alone. Its clock stands for
        # the versions of the categories it changes, read by the trackers, and the selections of these categories are
        # dropped, which leaves its mark with none.
        self._clock += 1
        self._n_commits += 1
        self._state_total += 1
        self._commit_clock = clock = self._clock
        for selections in self._commit_selections:
            selections.clear()
        commit_mark = self._commit_mark
        commit_mark.clock, commit_mark.n_commits, commit_mark.state_total = clock, self._n_commits, self._state_total
        commit_mark.n_selections = 0
        # A restore of a state recorded modified drops the production of the commit (see 'restore')
        if self._commit_production is None:
            self._mark_committed()

    def _mark_committed(self) -> None:
        self._commit_mark.update(self)
        # The productions of a commit share its mark of the change log, fixing their roots (see 'register_refresh').
        if self._commit_production is None:
            self._commit_production = _Production()
            self._commit_production.bind({self: self._root_entries}, self._commit_logs_mark)

    def add(self, *items: StateChange | Any, envs_idx: IndexType | None = None) -> None:
        """Record changes of some categories and writes of some tracked arrays outside any tagged method or launch.

        It records a change of a component that holds no tracked array, such as the parameters of a sensor, and a write
        through a view or a restore, which no launch records. The derived arrays written are fresh in the environments
        written, as computed from the state.

        Parameters
        ----------
        *items : StateChange | qd.Tensor
            The categories changed and the arrays written, those the trackable does not track left out.
        envs_idx : IndexType | None, optional
            The environments changed, all of them for None. They are kept as given, without a copy, and read only when
            a tracker reads the change: the caller must never write them in place afterward, which corrupts the
            record. Defaults to None.
        """
        changes = [int(item) for item in items if isinstance(item, StateChange)]
        if changes:
            self._record(*changes, envs_idx=envs_idx)
        tracked = [_tracked_entry(item) for item in items if not isinstance(item, StateChange)]
        entries = tuple(entry for log, entry in [entry or (None, None) for entry in tracked] if log is self)
        if not entries:
            return
        self._record_write(self._log_write(entries), envs_idx=envs_idx)
        entries_derived = tuple(entry for entry in entries if self._entries_kind[entry] == DataKind.DERIVED)
        if entries_derived:
            self._produce(_ADD_PRODUCTION_KEY, entries_derived, {self: self._root_entries}, envs_idx)

    def _log_write(self, entries: tuple[int, ...]) -> "_LogWrite":
        """Return the record of a write of some tracked arrays, which resolves once what such a write records."""
        entries_kind, entries_change = self._entries_kind, self._entries_change
        entries_roots = tuple(entry for entry in entries if entries_kind[entry] not in _COMPUTED_KINDS)
        changes = {entries_change[entry] for entry in entries_roots if entries_change[entry] is not None}
        if entries_roots:
            changes.add(_ANY_STATE)
        entries_computed = tuple(entry for entry in entries if entries_kind[entry] in _COMPUTED_KINDS)
        return _LogWrite(weakref.ref(self), entries, entries_computed, tuple(changes), entries_roots, _WriteRootSets())

    def _record_write(self, write: "_LogWrite", envs_idx: IndexType | None) -> None:
        """Bump the versions of the arrays a write reaches and record the categories of those of the state and model.

        A refresh writes the state only to normalize it, such as the quaternions of the configuration, which counts as
        no change of it.
        """
        if _refresh_depth > 0:
            entries, changes = write.entries_computed, ()
        else:
            entries, changes = write.entries, write.changes
        if not entries:
            return
        if changes:
            self._state_total += 1
        self._record(*changes, *entries, envs_idx=envs_idx)
        if changes:
            write_root_sets = write.root_sets
            if write_root_sets.n_root_sets != len(self._root_sets):
                write_root_sets.n_root_sets = len(self._root_sets)
                write_root_sets.root_sets = tuple(
                    root_set for entry in write.entries_roots for root_set in self._entries_root_sets[entry]
                )
            clock = self._clock
            for root_set in write_root_sets.root_sets:
                root_set.clock = clock

    def _record(self, *changes: int, envs_idx: IndexType | None = None) -> None:
        """Record a change of some categories and tracked arrays in some environments, all of them for None.

        A category no tracker follows only bumps its version, its environments left unresolved.
        """
        self._clock += 1
        clock = self._clock
        for change in changes:
            self._versions[change] = clock
        # The categories a tagged method changed, which its outermost call notifies to the subscribers (see 'mutates')
        if self._is_mutating:
            self._mutation_changes.update([change for change in changes if change < _ANY_STATE])
        # A launch over every environment takes their indices, without repeat once sanitized, which name them all as
        # None does.
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype != torch.bool and envs_idx.numel() == self.n_envs:
            envs_idx = None
        for change in changes:
            counters = self._counters[change]
            if counters is None:
                continue
            selections = counters.selections
            if envs_idx is None or len(selections) >= _MAX_SELECTIONS:
                counters.all_clock = clock
                selections.clear()
            else:
                selections.append(envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- consumers --------------------------------------
    # ------------------------------------------------------------------------------------

    def subscribe(self, subscriber: "Subscriber") -> None:
        """Register an eager subscriber, called back after the subscribers registered before it.

        Parameters
        ----------
        subscriber : Subscriber
            The subscriber, called back inside each tagged method recording a change of its categories.
        """
        self._subscribers.append(subscriber)

    def track(self, *items: StateChange | Any) -> Tracker:
        """Return a tracker of the changes of some categories and of the writes of some tracked arrays.

        Parameters
        ----------
        *items : StateChange | qd.Tensor
            The categories and the arrays the tracker reports, every category if none. An array counts as written by
            the launches, the refreshes, the additions and the restores writing it, a commit leaving it as it is.
        """
        changes_int = frozenset(
            [int(item) if isinstance(item, StateChange) else _tracked_entry(item)[1] for item in items]
        ) or frozenset(_ALL_CHANGES)
        for change in changes_int:
            if self._counters[change] is None:
                self._counters[change] = _Counters()
                if change in _COMMIT_CHANGES:
                    self._commit_selections.append(self._counters[change].selections)
        return Tracker(self, changes_int)


class SubscriberCallback(Protocol):
    """The function a subscriber calls back with the category of a change and the indices it happened at.

    The indices name the environments the change reached by 'envs_idx', None for all of them.
    """

    def __call__(self, change: StateChange, **indices: IndexType | None) -> None: ...


class Subscriber:
    """A callback that a trackable calls inside a tagged method, as a change happens.

    A consumer registers it through 'ChangeLog.subscribe'. It is reserved for a consumer that must react before the
    tagged method returns, such as the interactive viewer showing a setter at once. Any other consumer reads a tracker
    when it chooses (see 'Tracker'), since a callback runs inside every call of its categories and costs time in each.

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
            The indices the change happened at (see 'SubscriberCallback').
        """
        if change in self.to:
            self.callback(_STATE_CHANGES[change], **indices)


class TrackableMixin:
    """A component whose state changes, which it records in its change log for its consumers.

    A trackable is identified by itself, whatever the equality of a class it is mixed in, so that it keys what its
    consumers follow (see 'Shape.invalidated_by').
    """

    # The changes of the state of the trackable, made as it builds
    changelog: ChangeLog | None = None

    __eq__ = object.__eq__
    __hash__ = object.__hash__


# ------------------------------------------------------------------------------------
# ------------------------------ launches and reads ----------------------------------
# ------------------------------------------------------------------------------------

# The change log and the entry of each tracked array, as long as its change log lives, a plain dictionary being the
# fastest lookup of the read hook. The module-level records reach the change logs weakly, since a change log reaches its
# trackable and what it holds through its subscribers.
_TRACKED_ARRAYS: dict[Any, tuple["weakref.ref[ChangeLog]", int]] = {}

# The function each read of a derived or scratch array calls first, which refreshes it if a refresh of its trackable
# computes it and tells whether the read goes on, resolved once per array (see 'ChangeLog._reader'). It reaches its
# change log weakly, as the record above does.
_ARRAY_READERS: dict[Any, Callable[[], bool]] = {}


def _untrack_arrays(arrays: list[Any]) -> None:
    for array in arrays:
        _TRACKED_ARRAYS.pop(array, None)
        _ARRAY_READERS.pop(array, None)


# The key of the production of the derived arrays an addition writes (see 'ChangeLog.add'), one per change log
_ADD_PRODUCTION_KEY = object()

# The depth of the refreshes running, whose productions leave no environment pending (see 'ChangeLog._refresh')
_refresh_depth = 0

# The depth of the guarded methods running, whose launches are skipped once they read an array neither fresh nor
# refreshable (see 'guard_reads')
_guard_depth = 0

# The kinds the blocks running give to some computed arrays, innermost last, each with the entries it names by change
# log, None naming every computed array (see 'scoped_kind').
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
        # An unhashable value, such as a list, is no array
        return None
    if tracked is None:
        return None
    log = tracked[0]()
    return None if log is None else (log, tracked[1])


class _WriteRootSets:
    """The root sets the roots of a write belong to, with the number of root sets of the change log they were resolved
    at, which grows as productions ask for new ones (see 'ChangeLog._root_set')."""

    __slots__ = ("n_root_sets", "root_sets")

    def __init__(self) -> None:
        self.n_root_sets = -1
        self.root_sets: tuple[_RootSet, ...] = ()


class _LogWrite(NamedTuple):
    """The arrays of one change log a launch or a tagged method may write, resolved once into what a write records."""

    log: "weakref.ref[ChangeLog]"
    entries: tuple[int, ...]
    entries_computed: tuple[int, ...]
    """The derived and scratch arrays among them."""
    changes: tuple[int, ...]
    """The categories a write of the roots records, every change of the state included."""
    entries_roots: tuple[int, ...]
    """The arrays of the state and of the model among them."""
    root_sets: _WriteRootSets


class _LaunchEffect(NamedTuple):
    """What a launch of a kernel on given structs reads and writes among the tracked arrays."""

    writes: tuple[_LogWrite, ...]
    # The derived and scratch arrays the launch reads without writing them, and those a refresh computes with it
    reads_computed: tuple[tuple["weakref.ref[ChangeLog]", int], ...]
    reads_refreshed: tuple[tuple["weakref.ref[ChangeLog]", int], ...]
    # The arrays of the state and of the model the launch reads, by change log
    roots: dict["weakref.ref[ChangeLog]", tuple[int, ...]]
    # The key of the production of the computed arrays the launch writes (see 'ChangeLog._produce'), and the position
    # of its argument 'envs_idx'.
    production_key: object | None
    i_envs: int | None


class _LaunchTracker:
    """The hook Quadrants calls before every launch of a forward kernel, recording what the launch reads and writes.

    It refreshes the derived arrays the launch reads, records the arrays it writes, and makes the derived ones fresh
    (see 'ChangeLog'). What a launch accesses depends on the kernel and on the structs it takes, so it is
    computed once per kernel and structs, the structs being compared by identity at each launch.
    """

    # The autograd tape of Quadrants also stands for a gradient checker, which this tracker is not
    grad_checker = None

    def __init__(self) -> None:
        # The positions of the struct arguments of each kernel, and the effect of each set of structs it took, the
        # structs held through weak references, the first of which drops its entry once collected.
        self._kernels_effects: dict[Any, tuple[tuple[int, ...], list[tuple[tuple[weakref.ref, ...], Any]]]] = {}
        # Whether the launches go untracked, those another tape forwards included, and whether they still refresh the
        # derived arrays they read first (see 'untracked_launches').
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

        for log_ref, entry in effect.reads_refreshed:
            log = log_ref()
            if log is not None and _effective_kind(log, entry) == DataKind.DERIVED:
                production = log._production_of(entry)
                if production is None or production.check()():
                    log._refresh(log._entries_refresh[entry])
        # A guarded method returns None in place of a launch reading what no refresh can compute (see 'guard_reads')
        if _guard_depth > 0 and _refresh_depth == 0:
            for log_ref, entry in effect.reads_computed:
                log = log_ref()
                if log is not None and not _ARRAY_READERS[log._entries_arrays[entry][0]]():
                    raise _UnreadableArray
        # The launches of a refresh are tracked whatever the block running it
        if self.is_suspended and _refresh_depth == 0:
            return
        envs_idx = None if effect.i_envs is None else args[effect.i_envs]
        for write in effect.writes:
            log = write.log()
            if log is not None:
                # A refresh the call triggers writes no array the call itself does
                log._n_launches_records += _refresh_depth == 0
                log._record_write(write, envs_idx)
        if effect.production_key is not None:
            # The roots of the computed arrays read are roots of the arrays computed from them. An array read while it
            # is not fresh leaves them fresh all the same, since the analysis may report reads a static branch never
            # makes (see 'get_accessed_paths'), of arrays only a commit delivers in practice.
            roots = {log_ref(): entries for log_ref, entries in effect.roots.items() if log_ref() is not None}
            for log_ref, entry in effect.reads_computed:
                log = log_ref()
                production = None if log is None else log._production_of(entry)
                if production is not None:
                    for root_log, root_entries in production.roots.items():
                        roots[root_log] = tuple(sorted(set(roots.get(root_log, ())) | set(root_entries)))
            for write in effect.writes:
                log, entries_computed = write.log(), write.entries_computed
                if entries_computed and log is not None:
                    # A launch leaves no result in a scratch array, whereas a refresh computes one (see 'scoped_kind').
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
        writes.append(log._log_write(entries))
    reads_computed = tuple(
        (log, entry)
        for log, entry in used
        if (log, entry) not in written and log._entries_kind[entry] in _COMPUTED_KINDS and not is_moving_data
    )
    reads_refreshed = tuple((weakref.ref(log), entry) for log, entry in reads_computed if entry in log._entries_refresh)
    reads_computed = tuple((weakref.ref(log), entry) for log, entry in reads_computed)
    roots = {}
    for log, entry in used:
        if log._entries_kind[entry] not in _COMPUTED_KINDS:
            roots.setdefault(log, []).append(entry)
    roots = {weakref.ref(log): tuple(entries) for log, entries in roots.items()}
    is_producing = any([write.entries_computed for write in writes])
    i_envs = names.index("envs_idx") if "envs_idx" in names else None
    return _LaunchEffect(
        tuple(writes), reads_computed, reads_refreshed, roots, object() if is_producing else None, i_envs
    )


# The tracker of every launch, installed once the first change log tracks arrays
_LAUNCH_TRACKER = _LaunchTracker()


def _install_launch_tracker() -> None:
    runtime = impl.get_runtime()
    # FIXME: quadrants#1000 - the launch hook of Quadrants is reachable only through this private runtime slot, which a
    # Quadrants autograd tape entered after the build empties, leaving every launch untracked.
    if runtime.target_tape is None:
        runtime.target_tape = _LAUNCH_TRACKER
        qd_to_torch.register_readers(_ARRAY_READERS)


class untracked_launches:
    """Run a block or a method without recording what its launches write, for a caller recording it through a commit.

    A commit records the block as one write, whose results are fresh once it is over, while its launches compute the
    derived arrays the next ones read, which a launch tracked alone would refresh first, the analysis reporting more
    reads than the kernels make (see 'get_accessed_paths'). The hook of another tape, such as the recorder of a
    differentiable rollout, stays in place, its launches forwarded to the tracker going untracked all the same. Used as
    the decorator of a method, it runs every call of the method this way.

    Parameters
    ----------
    refresh : bool, optional
        Whether a launch still refreshes the derived arrays it reads first, as a coupler reading the poses of the links
        written since the last commit, the launches of the refresh being tracked. Defaults to False.
    """

    # A class, cheaper to enter than a generator, the block running on the hot path
    __slots__ = ("_refresh", "_runtime", "_is_installed", "_is_suspended", "_is_refreshing")

    def __init__(self, *, refresh: bool = False) -> None:
        self._refresh = refresh

    def __enter__(self) -> None:
        self._runtime = runtime = impl.get_runtime()
        self._is_installed = runtime.target_tape is _LAUNCH_TRACKER
        self._is_suspended, self._is_refreshing = _LAUNCH_TRACKER.is_suspended, _LAUNCH_TRACKER.is_refreshing
        if self._is_installed and not self._refresh:
            runtime.target_tape = None
        _LAUNCH_TRACKER.is_suspended, _LAUNCH_TRACKER.is_refreshing = True, self._refresh

    def __exit__(self, *exc_info: Any) -> None:
        _LAUNCH_TRACKER.is_suspended, _LAUNCH_TRACKER.is_refreshing = self._is_suspended, self._is_refreshing
        if self._is_installed:
            self._runtime.target_tape = _LAUNCH_TRACKER

    def __call__(self, method: Callable[..., Any]) -> Callable[..., Any]:
        # Each call enters a context of its own, which holds what it restores as it exits
        refresh = self._refresh

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with untracked_launches(refresh=refresh):
                return method(*args, **kwargs)

        return wrapper


# ------------------------------------------------------------------------------------
# ------------------------------------ the tags --------------------------------------
# ------------------------------------------------------------------------------------


class _MethodWrites(NamedTuple):
    """What a call of a tagged method on one object writes, resolved once per object (see 'mutates')."""

    writes: tuple[_LogWrite, ...]
    """The arrays the kernels the method names may write, by change log."""
    is_model: bool
    """Whether one of them is an array of the model."""


# What each tagged method writes, by object, cleared by a registration of arrays (see 'ChangeLog.register_arrays').
_METHODS_WRITES: list["weakref.WeakKeyDictionary[Any, _MethodWrites]"] = []


def _method_writes(method: Callable[..., Any], owner: Any) -> _MethodWrites:
    """Return what a call of a tagged method on an object writes."""
    names = list(inspect.signature(method).parameters)
    written = _resolve_paths(get_accessed_paths(method, owner).paths_written, {names[0]: owner})
    writes = []
    for log in {log: None for log, _ in written}:
        entries = tuple(entry for entry_log, entry in written if entry_log is log)
        writes.append(log._log_write(entries))
    is_model = any([written_log._entries_kind[entry] == DataKind.INFO for written_log, entry in written])
    return _MethodWrites(tuple(writes), is_model)


class scoped_kind:
    """Give some computed arrays another kind, SCRATCH or DERIVED, for the duration of a block.

    What a launch writes in a derived array is fresh by contract, and a read refreshes a derived array that is not,
    while what a launch writes in a scratch array leaves it to refresh and a read takes it as it stands. A block reusing
    a derived array for something else, such as contacts cleared, makes it scratch, and a getter reading a scratch
    array a refresh computes makes it derived, such as the mass matrix a solver uses as its working buffer. What a
    refresh writes is fresh whatever the kind of the array, and a commit leaves no scratch array fresh. Blocks
    nest and compose, the innermost one naming an array deciding its kind. As a context manager it takes the arrays, or
    structs standing for every array of them, and as a decorator of a method, the functions returning them from the
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


def mutates(method: Callable[..., Any] | None = None, *, differentiable: bool = True) -> Callable[..., Any]:
    """Tag a method of a solver or of one of its entities writing its state.

    A method writing the state records the arrays every kernel it names may write, whether it launched them or wrote
    their arrays through zero-copy views instead, so that a zero-copy write lives in a tagged method next to the kernel
    it stands for. The argument 'envs_idx' of the method names the environments written, all of them if it has none.
    The outermost tagged call notifies the eager subscribers once (see 'Subscriber').

    Parameters
    ----------
    method : Callable[..., Any] | None, optional
        The method, for the bare decorator '@mutates', which takes the default options. None returns the decorator
        taking the options given, '@mutates(differentiable=False)'. Defaults to None.
    differentiable : bool, optional
        Whether the inputs of the method carry a gradient. Defaults to True.
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        names = list(inspect.signature(method).parameters)[1:]
        owners_writes: weakref.WeakKeyDictionary[Any, _MethodWrites] = weakref.WeakKeyDictionary()
        _METHODS_WRITES.append(owners_writes)

        def run_recording(self: Any, *args: Any, **kwargs: Any) -> Any:
            writes = owners_writes[self].writes
            # The arrays the launches of the call write are recorded as they launch, so that the derived arrays a later
            # launch of the call computes stay fresh. A call launching no kernel writing a change log took the path
            # writing its arrays through zero-copy views, whose writes the kernels it names stand for as it returns.
            writes = [(write.log(), write) for write in writes if write.log() is not None]
            n_launches_records = [log._n_launches_records for log, _ in writes]
            # The change logs written and that of the trackable, whose subscribers the outermost tagged call of each
            # calls back once with every category recorded in between, by launches and restores included.
            owner_log = (self._solver if isinstance(self, Entity) else self).changelog
            logs = {owner_log: None, **{log: None for log, _ in writes}}
            logs_outermost = [log for log in logs if not log._is_mutating]
            for log in logs_outermost:
                log._is_mutating = True
                log._mutation_changes.clear()
            envs_idx = (dict(zip(names, args)) | kwargs).get("envs_idx")
            try:
                result = method(self, *args, **kwargs)
                for (log, write), n_records in zip(writes, n_launches_records):
                    if log._n_launches_records == n_records:
                        log._record_write(write, envs_idx)
            finally:
                for log in logs_outermost:
                    log._is_mutating = False
            for log in logs_outermost:
                for subscriber in log._subscribers:
                    for changed in log._mutation_changes & subscriber.to:
                        subscriber.notify(changed, envs_idx=envs_idx)
            return result

        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # The table holds the objects weakly, and the solver stays out of their records, so that a destroyed scene
            # can be released.
            method_writes = owners_writes.get(self)
            if method_writes is None:
                method_writes = owners_writes[self] = _method_writes(method, self)
            solver = self._solver if isinstance(self, Entity) else self
            if method_writes.is_model:
                return run_recording(self, *args, **kwargs)
            # A call writing no tracked array has nothing to record, nor to notify where no subscriber listens
            run = run_recording if method_writes.writes or solver.changelog._subscribers else method
            # A differentiable rollout records every call writing the state as one operation of its graph, so a
            # differentiable scene takes no shortcut around it. Its inputs are the tensors tracking gradients it takes.
            if solver.is_recording:
                return solver._rollout_graph.run_call(self, run, args, kwargs, differentiable=differentiable)
            result = run(self, *args, **kwargs)
            # A call left unrecorded overwrites the output of the last recorded operation
            if solver.requires_grad:
                solver._rollout_graph.mark_state_changed()
            return result

        return wrapper

    return decorator if method is None else decorator(method)


class _UnreadableArray(Exception):
    """Raised before a launch within a guarded method that reads an array neither fresh nor refreshable."""


def guard_reads(method: Callable[..., Any]) -> Callable[..., Any]:
    """Tag a method returning None once a kernel it launches reads an array neither fresh nor refreshable.

    The launch is skipped, as a copy to the host of such an array returns None (see 'qd_to_torch'), so that a getter
    computing its output with a kernel, such as the accelerations of the links in their own frame, returns None until
    the next step computes what it reads. The launches of a refresh are never skipped.
    """

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _guard_depth
        _guard_depth += 1
        try:
            return method(*args, **kwargs)
        except _UnreadableArray:
            return None
        finally:
            _guard_depth -= 1

    return wrapper
