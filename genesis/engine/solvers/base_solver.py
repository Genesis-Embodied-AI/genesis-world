import enum
import functools
import inspect
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.entities.base_entity import Entity
from genesis.engine.materials.base import Material
from genesis.engine.states import QueriedStates, SolverCheckpoint
from genesis.repr_base import RBC
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    sanitize_index,
    tensor_to_array,
)


if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class StateChange(enum.Enum):
    """Category of solver scene-state mutation broadcast to subscribers (see `Solver.subscribe`).

    Solver-agnostic: it names what kind of state changed, never an index space (links, dofs, particles, ...), which
    differs from one solver to the next. GEOMETRY is the kinematic configuration that places or deforms the world
    surface (link poses, qpos, vertices); DYNAMICS is the velocity state. Model parameters (mass, inertia, friction,
    gains, limits) are not scene state and are never broadcast.
    """

    GEOMETRY = enum.auto()
    DYNAMICS = enum.auto()


class MutatedLinks(enum.Enum):
    """Which links a @mutates-tagged method can affect, consumed by link-filtered subscribers (see Subscriber).

    ALL: any link (the conservative default). ARTICULATED: only links whose pose is a function of the configuration
    - a configuration-space setter (qpos, dofs) can never displace a fixed link, which carries no degree of freedom
    (fixed links move only through explicit base-pose setters, whose reach is their link selection). A method whose
    reach is an explicit link selection instead declares the name of its links-index argument.
    """

    ALL = enum.auto()
    ARTICULATED = enum.auto()


def _expand_links_subtree(links_idx: np.ndarray, links_parent_idx: np.ndarray) -> np.ndarray:
    """Close a set of global link indices over their kinematic subtrees: every descendant of a member is a member.

    Iterates one tree level per pass (a link joins when its parent is a member), so it terminates after at most the
    tree depth."""
    is_affected = np.zeros(links_parent_idx.shape[0], dtype=bool)
    is_affected[links_idx] = True
    has_parent = links_parent_idx >= 0
    while True:
        is_expanded = is_affected.copy()
        is_expanded[has_parent] |= is_affected[links_parent_idx[has_parent]]
        if (is_expanded == is_affected).all():
            return np.flatnonzero(is_affected)
        is_affected = is_expanded


def mutates(
    *changes: StateChange, links: str | MutatedLinks = MutatedLinks.ALL
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a solver state-mutating method with the StateChange categories it produces (one method may produce several,
    e.g. set_state changes both GEOMETRY and DYNAMICS).

    Notification is deferred to the outermost tagged call: a setter that internally calls other tagged setters fires a
    single notification per StateChange that occurred, when the outermost call returns, rather than one per nested
    call. The outermost call's `envs_idx` (None meaning all envs) is forwarded. Untagged methods, reads included, never
    notify, so a subscriber only ever wakes on a genuine mutation.

    `links` declares which links the method can affect, so link-filtered subscribers can drop notifications that
    cannot concern them: the name of the argument holding the targeted global link indices (e.g. "links_idx") - the
    reach then closes over their kinematic subtrees, since forward kinematics carries a new pose to every descendant,
    fixed children included - MutatedLinks.ARTICULATED for configuration-space setters, or MutatedLinks.ALL when any
    link may change. Nested tagged calls union their mutated links, an unbounded reach absorbing the rest.
    """
    triggered = frozenset(changes)

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(method)

        def resolve_mutated_links_idx(self: "Solver", bound_arguments) -> np.ndarray | None:
            if links is MutatedLinks.ARTICULATED:
                return self._articulated_links_idx
            if links is MutatedLinks.ALL:
                return None
            links_idx = bound_arguments.get(links)
            if links_idx is None or self._links_parent_idx is None:
                # The setter substitutes its own default selection (potentially reaching any link), or the solver
                # exposes no kinematic structure to close the reach over: stay unbounded.
                return None
            # sanitize_index covers every index form the setters accept (int, sequence, slice, bool mask, tensor);
            # the reach set lives host-side, where the subscriber filters compare.
            links_idx = tensor_to_array(sanitize_index(links_idx, -1, self.n_links, 0, links))
            return _expand_links_subtree(links_idx, self._links_parent_idx)

        @functools.wraps(method)
        def wrapper(self: "Solver", *args: Any, **kwargs: Any) -> Any:
            if not self._subscribers:
                return method(self, *args, **kwargs)
            bound_arguments = signature.bind(self, *args, **kwargs).arguments
            mutated_links_idx = resolve_mutated_links_idx(self, bound_arguments)
            if self._is_mutating:
                # Nested tagged call: record the changes and mutated links, let the outermost call do the single
                # notification.
                self._mutation_changes |= triggered
                if self._mutation_links_idx_list is not None:
                    if mutated_links_idx is None:
                        self._mutation_links_idx_list = None
                    else:
                        self._mutation_links_idx_list.append(mutated_links_idx)
                return method(self, *args, **kwargs)
            self._is_mutating = True
            self._mutation_changes = set(triggered)
            self._mutation_links_idx_list = None if mutated_links_idx is None else [mutated_links_idx]
            try:
                result = method(self, *args, **kwargs)
            finally:
                self._is_mutating = False
            envs_idx = bound_arguments.get("envs_idx")
            links_idx = None
            if self._mutation_links_idx_list is not None:
                links_idx = np.concatenate(self._mutation_links_idx_list)
            for subscriber in self._subscribers:
                if not subscriber.is_watching(links_idx):
                    continue
                for changed in self._mutation_changes & subscriber.to:
                    if subscriber.callback is None:
                        subscriber._pending.add(changed)
                    else:
                        subscriber.callback(changed, envs_idx)
            return result

        return wrapper

    return decorator


class Subscriber:
    """A unique handle for the solver state changes whose category is in `to`.

    A consumer constructs a Subscriber and registers it with a solver via Solver.subscribe. The mode is fixed at
    construction by whether a callback is given:
      - eager (callback given): each matching change immediately calls callback(change, envs_idx);
      - lazy (no callback): matching changes accumulate into `pending` until the owner calls clear() - e.g. a sensor
        that rebuilds a cache on its next update rather than on every set_pos.

    `links_filter` (global link indices, normalized by Solver.subscribe) restricts the subscription to changes that
    can affect those links: a notification whose mutated links (see mutates) are disjoint from the filter is dropped;
    a notification with an unbounded reach always passes.
    """

    def __init__(
        self,
        to: frozenset[StateChange],
        callback: Callable[[StateChange, object], None] | None = None,
        links_filter: "np.typing.ArrayLike | None" = None,
    ):
        self.to = to
        self.callback = callback
        self.links_filter = links_filter
        self._pending: set[StateChange] = set()

    def is_watching(self, links_idx: np.ndarray | None) -> bool:
        """Whether a change affecting `links_idx` (None meaning possibly any link) concerns this subscriber."""
        if self.links_filter is None or links_idx is None:
            return True
        return bool(np.isin(links_idx, self.links_filter).any())

    @property
    def pending(self) -> frozenset[StateChange]:
        """Categories accumulated since the last clear() (always empty in eager mode)."""
        return frozenset(self._pending)

    def clear(self):
        """Drop the accumulated changes, once they have been handled."""
        self._pending.clear()


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

    def __init__(self, scene: "Scene", sim: "Simulator", options):
        self._uid = gs.UID()
        self._sim = sim
        self._scene = scene
        self._options = options
        self._entities: list[Entity] = gs.List()

        # Queue of solver-level states queried during the current backward window. Solvers that surface solver-state
        # (kinematic, rigid) push into it from `get_state`; others leave it empty. `Simulator.get_state` calls `discard`
        # here to lift entries owned by a `SimState`, preventing `collect_output_grads` from accumulating adjoints twice
        # through both the simulator-level and the per-solver loop.
        self._queried_states = QueriedStates()

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

    def _add_force_field(self, force_field):
        self._ffs.append(force_field)

    def subscribe(self, subscriber: Subscriber):
        """Register a Subscriber to be notified after any @mutates-tagged method whose change is in its filter."""
        if subscriber.links_filter is not None:
            subscriber.links_filter = tensor_to_array(
                sanitize_index(subscriber.links_filter, -1, self.n_links, 0, "links_filter")
            )
        self._subscribers.add(subscriber)

    def build(self):
        self._B = self._sim._B

    @property
    def data(self) -> Iterator[array_class.DataItem]:
        """Yield every array and static config the solver holds, tagged by kind (see 'DataKind'), under the dotted name
        a checkpoint and a trajectory frame use for it.
        """
        gs.raise_exception(f"{type(self).__name__} cannot be checkpointed yet.")

    def __getstate__(self) -> SolverCheckpoint:
        """Return a SolverCheckpoint of this solver, for '__setstate__' to restore.

        The state is what 'data' yields of the kinds a checkpoint carries (see 'CHECKPOINT_KINDS'), copied where it
        lives: on the device as torch tensors when zero-copy views exist, which keeps a save and a restore within one
        process off the host, and as numpy arrays otherwise.
        """
        read = qd_to_torch if gs.use_zerocopy else qd_to_numpy
        arrays = {}
        configs = {}
        for name, value, kind in self.data:
            if kind == array_class.DataKind.CONFIG:
                configs[name] = value
            elif kind in array_class.CHECKPOINT_KINDS:
                arrays[name] = read(value, copy=True)
        return SolverCheckpoint(arrays=arrays, configs=configs, kinds=array_class.CHECKPOINT_KINDS)

    def __setstate__(self, state: SolverCheckpoint) -> None:
        """Write back the arrays of the kinds a record holds, rejecting a record read from another build."""
        array_class.fill_data((item for item in self.data if item.kind in state.kinds), state.arrays)

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
