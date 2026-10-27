"""
Changes of the state of the solvers: their categories, and the tag of the methods making them.

A tagged method notifies the subscribers of its solver (see 'Subscriber' in base_solver.py), and is a call the autograd
graph of a differentiable rollout records (see 'RolloutGraph.run_call'). The module stands outside of the solvers, so
that the entities tag their own methods.
"""

import enum
import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from genesis.engine.entities.base_entity import Entity
from genesis.utils.misc import sanitize_index, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


class StateChange(enum.Enum):
    """Category of solver scene-state mutation broadcast to subscribers (see `Solver.subscribe`).

    Solver-agnostic: it names what kind of state changed, never an index space (links, dofs, particles, ...), which
    differs from one solver to the next. GEOMETRY is the kinematic configuration that places or deforms the world
    surface (link poses, qpos, vertices). DYNAMICS is the velocity state. MODEL is the model parameters (mass,
    inertia, friction, gains, limits, gravity).
    """

    GEOMETRY = enum.auto()
    DYNAMICS = enum.auto()
    MODEL = enum.auto()


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
    *changes: StateChange,
    links: str | MutatedLinks = MutatedLinks.ALL,
    differentiable: bool = True,
    derived: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a state-mutating method of a solver or of one of its entities with the StateChange categories it produces
    (one method may produce several, e.g. set_state changes both GEOMETRY and DYNAMICS, and an input of the next step
    such as a control target produces none).

    Every tagged method writing the state, all but those changing the model, is a call the autograd graph of a
    differentiable rollout records and replays (see 'RolloutGraph.run_call'), so a differentiable scene takes no
    shortcut around it. Its inputs are its tensors tracking gradients, which an input that is not `differentiable`
    refuses. A call the graph does not record overwrites the output of the last operation recorded, unless it only
    refreshes quantities `derived` from the state, which it leaves consistent with it.

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
    is_recorded = StateChange.MODEL not in triggered

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

        def run_notifying(self: "Solver", *args: Any, **kwargs: Any) -> Any:
            if not triggered or not self._subscribers:
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

        @functools.wraps(method)
        def wrapper(self: "Solver | Entity", *args: Any, **kwargs: Any) -> Any:
            solver = self._solver if isinstance(self, Entity) else self
            run = run_notifying if solver is self else method
            if not is_recorded:
                return run(self, *args, **kwargs)
            if solver.is_recording:
                return solver._rollout_graph.run_call(self, run, args, kwargs, differentiable=differentiable)
            result = run(self, *args, **kwargs)
            if solver.requires_grad and not derived:
                solver._rollout_graph.mark_state_changed()
            return result

        return wrapper

    return decorator
