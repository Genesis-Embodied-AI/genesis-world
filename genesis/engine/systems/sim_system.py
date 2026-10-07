from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum, auto
import inspect
from typing import TYPE_CHECKING, Callable, TypeVar

import quadrants as qd

if TYPE_CHECKING:
    from .sim_engine import SimEngine


T = TypeVar("T", bound="SimSystem")


class SimData:
    """Marker for fully constructed mutable Quadrants data objects."""


class ActionKind(Enum):
    """Actual Quadrants callable category derived from compiler metadata."""

    HOST = auto()
    INLINE_FUNC = auto()
    TOP_LEVEL_FUNC = auto()
    REAL_FUNC = auto()


def _classify_qd_callable(kernel: Callable) -> ActionKind:
    if not getattr(kernel, "_is_quadrants_function", False):
        raise TypeError(f"{kernel!r} is not a Quadrants @qd.func")
    if getattr(kernel, "_is_wrapped_kernel", False):
        raise TypeError("Quadrants @qd.kernel objects are not SimAction callables")
    if getattr(kernel, "_is_real_function", False):
        return ActionKind.REAL_FUNC
    if getattr(kernel, "_qd_requires_top_level", False):
        return ActionKind.TOP_LEVEL_FUNC
    return ActionKind.INLINE_FUNC


def _transient_arity(kernel: Callable, closed_data: tuple[object, ...]) -> int:
    fn = getattr(kernel, "fn", None)
    if fn is None:
        raise TypeError(f"{kernel!r} does not expose its underlying Quadrants function")
    signature = inspect.signature(fn)
    if any(
        parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    ):
        raise TypeError(f"{fn.__qualname__} uses unsupported variable arguments")
    arity = len(signature.parameters) - len(closed_data)
    if arity < 0:
        raise TypeError(
            f"{fn.__qualname__} has {len(signature.parameters)} parameters but "
            f"{len(closed_data)} persistent arguments were closed"
        )
    return arity


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class SimAction:
    """Schedulable ``(kernel, *data)`` closure owned by a ``SimSystem``.

    ``owner`` exists only for lifecycle validation and diagnostics.
    """

    def __init__(
        self,
        *,
        owner: "SimSystem",
        data: tuple[object, ...],
        kernel: Callable,
        kind: ActionKind,
        transient_arity: int,
        rank: int = -1,
    ) -> None:
        self.owner = owner
        self.data = data
        self.kernel = kernel
        self.kind = kind
        self.transient_arity = transient_arity
        self.rank = int(rank)

    # Quadrants does not recognize a data-oriented @qd.func __call__ through
    # its public callable API. Keep an explicit method until that is supported.
    @qd.pyfunc
    def invoke(
        self,
        transient_args: qd.template() = (),  # tuple[object, ...]
    ):
        self.kernel(*(self.data + transient_args))


def validate_action_protocol(
    action: SimAction,
    *,
    protocol: str,
    expected_kind: ActionKind,
    transient_arity: int,
) -> None:
    """Validate a registered Action against its actual Quadrants callable."""
    if action.kind is ActionKind.HOST:
        actual_kind = ActionKind.HOST
        actual_arity = len(inspect.signature(action.kernel).parameters) - len(action.data)
    else:
        actual_kind = _classify_qd_callable(action.kernel)
        actual_arity = _transient_arity(action.kernel, action.data)
    if actual_kind is not expected_kind:
        raise TypeError(
            f"{protocol} requires {expected_kind.name}, but {action.kernel.fn.__qualname__} is {actual_kind.name}"
        )
    if actual_arity != transient_arity:
        raise TypeError(
            f"{protocol} requires {transient_arity} transient arguments, but "
            f"{action.kernel.fn.__qualname__} requires {actual_arity}"
        )


class SimPipeline:
    """Host runtime for a bound graph and its checkpoint-yield callbacks."""

    def __init__(
        self,
        graph: Callable,
        *,
        yield_callbacks: Mapping[int, Callable] | None = None,
    ) -> None:
        self.graph = graph
        self.yield_callbacks = {
            int(checkpoint): callback
            for checkpoint, callback in (() if yield_callbacks is None else yield_callbacks.items())
        }

    def run(self, *args, **kwargs):
        """Launch the graph and handle every yield until it completes."""
        status = self.graph(*args, **kwargs)
        while status.yielded:
            checkpoint = int(status.checkpoint)
            callback = self.yield_callbacks.get(checkpoint)
            if callback is None:
                raise RuntimeError(f"SimPipeline has no yield callback for checkpoint {checkpoint}")
            resume_from = callback(status)
            if resume_from is None:
                resume_from = checkpoint
            status = self.graph.resume(
                *args,
                from_checkpoint=int(resume_from),
                **kwargs,
            )
        return status


class ActionCollection:
    """Build-time action registry resolved into a ``SimPipeline`` before use."""

    __slots__ = ("_actions", "_owner")

    def __init__(self, owner: "SimSystem") -> None:
        self._owner = owner
        self._actions: list[SimAction] = []

    def register(self, action: SimAction) -> None:
        if not self._owner.is_building:
            raise RuntimeError("SimAction objects may be registered only during SimSystem.build()")
        if any(
            registered.owner is action.owner
            and len(registered.data) == len(action.data)
            and all(
                registered_item is action_item for registered_item, action_item in zip(registered.data, action.data)
            )
            and registered.kernel is action.kernel
            for registered in self._actions
        ):
            raise RuntimeError("SimAction is already registered")
        if action.rank >= 0 and any(registered.rank == action.rank for registered in self._actions):
            raise RuntimeError(f"SimAction rank {action.rank} is already registered")
        self._actions.append(action)

    @property
    def actions(self) -> tuple[SimAction, ...]:
        if self._owner.is_building:
            raise RuntimeError("ActionCollection is available only after build")
        ranked = [(index, action) for index, action in enumerate(self._actions) if action.rank >= 0]
        unranked = [action for action in self._actions if action.rank < 0]
        ranked.sort(key=lambda item: item[1].rank)
        return tuple(action for _, action in ranked) + tuple(unranked)


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class SimSystem(ABC):
    """Python organizational node for simulation data, dependencies, and actions.

    A system collects ``SimData``, declares dependencies on other systems, and
    publishes ``SimAction`` objects during build, then populates the canonical
    nested ``Data`` object during init. Device execution is expressed
    by composed ``SimPipeline`` instances rather than by scheduling the system
    object itself.

    Concrete Systems must be default-constructible. Scene values, capacities,
    solver references, and configuration enter through explicit wiring or
    dependencies and populate the already-created ``Data`` object during
    ``init()``. Constructor arguments would bypass dependency discovery and the
    build/init lifecycle, hide ownership in call sites, and prevent uniform
    Engine construction.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__bases__ != (SimSystem,):
            raise TypeError(f"{cls.__name__} must inherit directly and only from SimSystem")
        initializer = cls.__dict__.get("__init__")
        if initializer is not None:
            parameters = tuple(inspect.signature(initializer).parameters.values())
            if len(parameters) != 1 or parameters[0].name != "self":
                raise TypeError(
                    f"{cls.__name__}.__init__ must accept only self; "
                    "wire scene/configuration inputs explicitly and populate the stable Data object in init()"
                )

    def __init__(self) -> None:
        self._engine: SimEngine | None = None
        self._dependencies: dict[SimSystem, bool] = {}
        self._is_valid = True
        self._build_phase_open = False

    @property
    def engine(self) -> SimEngine:
        engine = self._engine
        if engine is None:
            raise RuntimeError(f"{type(self).__name__} is not registered with a SimEngine")
        return engine

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def is_building(self) -> bool:
        return self._build_phase_open

    def find(self, system_type: type[T]) -> T | None:
        """Return a registered system of the requested type, if one is active."""
        system = self.engine.find(system_type)
        if system is not None:
            self._dependencies.setdefault(system, False)
        return system

    def require(self, system_type: type[T]) -> T:
        """Return a registered system of the requested type."""
        system = self.engine.find(system_type)
        if system is None:
            raise RuntimeError(f"Required system {system_type.__name__} is not registered")
        self._dependencies[system] = True
        return system

    def dependencies(self) -> dict["SimSystem", bool]:
        """Return dependency systems mapped to ``True`` for require and ``False`` for find."""
        return dict(self._dependencies)

    @staticmethod
    def _resolve_bound_callable(value: Callable) -> tuple[object, Callable] | None:
        data = getattr(value, "instance", None)
        kernel = getattr(value, "quadrants_callable", None)
        if data is None or kernel is None:
            data = getattr(value, "__self__", None)
            kernel = getattr(value, "__func__", None)
        if data is None or kernel is None:
            return None
        return data, kernel

    def create_action(self, kernel: Callable, *data: object, rank: int = -1) -> SimAction:
        """Create a schedulable ``(kernel, *data)`` closure during build."""
        if not self._build_phase_open:
            raise RuntimeError("SimAction objects may be created only during SimSystem.build()")
        resolved = self._resolve_bound_callable(kernel)
        if resolved is not None:
            bound_target, resolved_kernel = resolved
            resolved_data = (bound_target, *data)
        else:
            resolved_kernel = kernel
            resolved_data = tuple(data)
            if not resolved_data:
                raise TypeError("Pure action kernels require one or more explicit ordered Data arguments")
        is_qd_callable = getattr(resolved_kernel, "_is_quadrants_function", False)
        kind = _classify_qd_callable(resolved_kernel) if is_qd_callable else ActionKind.HOST
        transient_arity = (
            _transient_arity(resolved_kernel, resolved_data)
            if is_qd_callable
            else len(inspect.signature(resolved_kernel).parameters) - len(resolved_data)
        )
        if transient_arity < 0:
            raise TypeError(f"{resolved_kernel.__qualname__} has fewer parameters than closed objects")
        return SimAction(
            owner=self,
            data=resolved_data,
            kernel=resolved_kernel,
            kind=kind,
            transient_arity=transient_arity,
            rank=rank,
        )

    def create_action_collection(self) -> ActionCollection:
        """Create a lifecycle collection owned by this system."""
        return ActionCollection(self)

    def create_pipeline(self, graph: Callable) -> SimPipeline:
        """Instantiate one graph runtime from a bound graph method."""
        if self._resolve_bound_callable(graph) is None:
            raise TypeError("Expected a bound graph method")
        return SimPipeline(graph)

    @abstractmethod
    def build(self) -> None:
        """Resolve dependencies and publish actions while registration is open."""

    def init(self) -> None:
        """Allocate and populate fields on the canonical nested Data object."""

    def _begin_build(self) -> None:
        if self._build_phase_open:
            raise RuntimeError(f"{type(self).__name__} build phase is already open")
        self._build_phase_open = True

    def _end_build(self) -> None:
        if not self._build_phase_open:
            raise RuntimeError(f"{type(self).__name__} build phase is not open")
        self._build_phase_open = False

    def _set_engine(self, engine: SimEngine) -> None:
        if not type(self).__dict__.get("_data_oriented", False):
            raise TypeError(f"{type(self).__name__} must be explicitly decorated with @qd.data_oriented")
        self._engine = engine

    def _invalidate(self) -> None:
        self._is_valid = False
