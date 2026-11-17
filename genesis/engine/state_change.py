"""
Changes of the state of the solvers: their categories, and the tag of the methods making them.

A tagged method notifies the subscribers of its solver (see 'Subscriber'), and is a call the autograd
graph of a differentiable rollout records (see 'RolloutGraph.run_call'). The module stands outside of the solvers, so
that the entities tag their own methods.
"""

import enum
import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from genesis.engine.entities.base_entity import Entity
from genesis.utils.misc import tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


class StateChange(enum.Enum):
    """Category of scene-state mutation broadcast to subscribers (see `Solver.subscribe` and `RenderContext.subscribe`).

    Solver-agnostic: it names what kind of state changed, never an index space (links, dofs, particles, ...), which
    differs from one solver to the next. GEOMETRY is the kinematic configuration that places or deforms the world
    surface (link poses, qpos, vertices). DYNAMICS is the velocity state. MODEL is what every later step reads besides
    the state: the model parameters (mass, inertia, friction, gains, limits, gravity) and the constraints attached to
    the state (particle and vertex constraints). VISUAL is the appearance alone, which the physics never reads, such as
    the visual vertices of a visual geom or the variant each environment shows of a dynamic texture: the render context
    reports it, never a solver.
    """

    GEOMETRY = enum.auto()
    DYNAMICS = enum.auto()
    MODEL = enum.auto()
    VISUAL = enum.auto()


class Subscriber:
    """A unique handle for the state changes whose category is in `to`.

    A consumer constructs a Subscriber and passes it to the 'subscribe' method of whatever it follows the changes of.
    The mode is fixed at construction by whether a callback is given:
      - eager (callback given): each matching change immediately calls callback(change, envs_idx);
      - lazy (no callback): matching changes accumulate into `pending` until the owner calls clear() - e.g. a sensor
        that rebuilds a cache on its next update rather than on every set_pos.

    `links_filter` (global link indices) restricts the subscription to changes that can affect those links: a
    notification whose mutated links (see mutates) cannot move a filtered link is dropped, while a notification with an
    unbounded reach always passes. The solver resolves the filter as the subscriber subscribes (see 'Solver.subscribe').
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

    @property
    def pending(self) -> frozenset[StateChange]:
        """Categories accumulated since the last clear() (always empty in eager mode)."""
        return frozenset(self._pending)

    def clear(self):
        """Drop the accumulated changes, once they have been handled."""
        self._pending.clear()

    def notify(self, change: StateChange, envs_idx=None):
        """Notify the subscriber of a change in some environments.

        A change outside 'to' is ignored. An eager subscriber is called back right away, while a lazy one accumulates
        the change until 'clear'.

        Parameters
        ----------
        change : StateChange
            The kind of change.
        envs_idx : None | int | array_like, optional
            The environments changed, all of them if None.
        """
        if change not in self.to:
            return
        if self.callback is None:
            self._pending.add(change)
        else:
            self.callback(change, envs_idx)


def mutates(
    *changes: StateChange,
    links: str | None = None,
    differentiable: bool = True,
    derived: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a state-mutating method of a solver or of one of its entities with the StateChange categories it produces.

    Notification is deferred to the outermost tagged call: a setter that internally calls other tagged setters fires a
    single notification per StateChange that occurred, when the outermost call returns, rather than one per nested
    call. The outermost call's `envs_idx` (None meaning all envs) is forwarded. Untagged methods, reads included, never
    notify, so a subscriber only ever wakes on a genuine mutation.

    Parameters
    ----------
    *changes : StateChange
        The categories the method produces. One method may produce several (set_state changes both GEOMETRY and
        DYNAMICS), and an input of the next step such as a control target produces none.
    links : str | None, optional
        The name of the argument holding the targeted global indices of links ("links_idx"), of degrees of freedom
        ("dofs_idx") or of configuration entries ("qs_idx"), the latter two reaching the links holding them, or None
        when any link may change. Link-filtered subscribers drop the notifications that cannot concern them. A selection
        of links left to None reaches any link, and one of degrees of freedom or configuration entries every link
        holding one. A change of a link moves its whole kinematic subtree, which the filters of the subscribers account
        for (see 'Solver.subscribe'). Nested tagged calls union their mutated links, an unbounded reach absorbing the
        rest. The reach is resolved only for a subscriber filtering on links and listening to the change, an unbounded
        reach standing for it otherwise. Defaults to None.
    differentiable : bool, optional
        Whether the inputs of the method carry a gradient. Defaults to True.
    derived : bool, optional
        Whether the method only refreshes quantities derived from the state. Defaults to False.
    """
    triggered = frozenset(changes)
    is_recorded = StateChange.MODEL not in triggered

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(method)

        def resolve_mutated_links_idx(self: "Solver", bound_arguments) -> np.ndarray | None:
            if links is None or self._links_parent_idx is None:
                return None
            indices = bound_arguments.get(links)
            # The links holding each degree of freedom or configuration entry, None for indices of links
            indices_link_idx = self._indices_link_idx.get(links)
            if indices is None:
                return None if indices_link_idx is None else self._articulated_links_idx
            # The indices are read on the host as they come: only a selection on the device syncs it
            n_indices = len(self._links_parent_idx) if indices_link_idx is None else len(indices_link_idx)
            if isinstance(indices, torch.Tensor):
                indices = tensor_to_array(indices)
            if isinstance(indices, (slice, range)):
                indices = np.arange(n_indices)[indices]
            indices = np.atleast_1d(np.asarray(indices))
            if indices.dtype == np.bool_:
                indices = np.flatnonzero(indices)
            return indices if indices_link_idx is None else indices_link_idx[indices]

        def run_notifying(self: "Solver", *args: Any, **kwargs: Any) -> Any:
            if not triggered or not self._subscribers:
                return method(self, *args, **kwargs)
            bound_arguments = signature.bind(self, *args, **kwargs).arguments
            mutated_links_idx = None
            if triggered & self._links_watched_changes:
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
                links_watched = self._subscribers_links_watched.get(subscriber)
                if links_watched is not None and links_idx is not None and not links_watched[links_idx].any():
                    continue
                for changed in self._mutation_changes & subscriber.to:
                    subscriber.notify(changed, envs_idx)
            return result

        @functools.wraps(method)
        def wrapper(self: "Solver | Entity", *args: Any, **kwargs: Any) -> Any:
            solver = self._solver if isinstance(self, Entity) else self
            run = run_notifying if solver is self else method
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
