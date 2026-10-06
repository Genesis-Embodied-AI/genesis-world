from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from .sim_engine import SimEngine


T = TypeVar("T", bound="SimSystem")


class SimData:
    """Marker for fully constructed mutable Quadrants data objects."""


@dataclass(frozen=True)
class ActionInvocation:
    """Quadrants-visible implementation of one scheduled ``SimAction``."""

    data: tuple[object, ...]
    kernel: Callable

    def __call__(self, *args, **kwargs):
        return self.kernel(*(self.data + args), **kwargs)


@dataclass(frozen=True)
class SimAction:
    """Schedulable ``(kernel, *data)`` closure owned by a ``SimSystem``.

    ``owner`` exists only for lifecycle validation and diagnostics. Pipelines
    pass only the ordered ``data`` and ``kernel`` closure into Quadrants graphs.
    """

    owner: "SimSystem"
    data: tuple[object, ...]
    kernel: Callable

    @property
    def invocation(self) -> ActionInvocation:
        return ActionInvocation(
            data=self.data,
            kernel=self.kernel,
        )


class SimPipeline:
    """Instantiated graph runtime with checkpoint-yield callbacks."""

    def __init__(self, data: object, graph: Callable) -> None:
        self.data = data
        self.graph = graph
        self._yield_callbacks: dict[int, Callable] = {}
        self._action_data: list[object] = []
        self._has_launched = False

    def bind_actions(self, *actions: SimAction | ActionInvocation) -> None:
        """Expose every scheduled Data object directly from the graph root."""
        if self._has_launched:
            raise RuntimeError("Actions must be bound before SimPipeline.run()")
        for action in actions:
            for data in action.data:
                if any(registered is data for registered in self._action_data):
                    continue
                index = len(self._action_data)
                self._action_data.append(data)
                setattr(self.data, f"_sim_pipeline_action_data_{index}", data)

    def register_yield_callback(self, checkpoint: int, callback: Callable) -> None:
        """Register the host callback required to resume one graph checkpoint."""
        if self._has_launched:
            raise RuntimeError("Yield callbacks must be registered before SimPipeline.run()")
        checkpoint = int(checkpoint)
        if checkpoint in self._yield_callbacks:
            raise RuntimeError(f"Checkpoint {checkpoint} already has a yield callback")
        self._yield_callbacks[checkpoint] = callback

    def run(self, *args, **kwargs):
        """Launch the graph and handle every yield until it completes."""
        self._has_launched = True
        status = self.graph(self.data, *args, **kwargs)
        while status.yielded:
            checkpoint = int(status.checkpoint)
            callback = self._yield_callbacks.get(checkpoint)
            if callback is None:
                raise RuntimeError(f"SimPipeline has no yield callback for checkpoint {checkpoint}")
            resume_from = callback(status)
            if resume_from is None:
                resume_from = checkpoint
            status = self.graph.resume(
                self.data,
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
        self._actions.append(action)

    @property
    def actions(self) -> tuple[SimAction, ...]:
        if self._owner.is_building:
            raise RuntimeError("ActionCollection is available only after build")
        return tuple(self._actions)


class SimSystem(ABC):
    """Python organizational node for simulation data, dependencies, and actions.

    A system collects ``SimData``, declares dependencies on other systems, and
    publishes ``SimAction`` objects during build. Device execution is expressed
    by composed ``SimPipeline`` instances rather than by scheduling the system
    object itself.
    """

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

    def create_action(self, kernel: Callable, *data: object) -> SimAction:
        """Create a schedulable ``(kernel, *data)`` closure during build."""
        if not self._build_phase_open:
            raise RuntimeError("SimAction objects may be created only during SimSystem.build()")
        if data:
            resolved_kernel = kernel
            resolved_data = tuple(data)
        else:
            resolved = self._resolve_bound_callable(kernel)
            if resolved is None:
                raise TypeError("Pure action kernels require one or more explicit ordered Data arguments")
            bound_target, resolved_kernel = resolved
            resolved_data = (bound_target,)
        return SimAction(
            owner=self,
            data=resolved_data,
            kernel=resolved_kernel,
        )

    def create_action_collection(self) -> ActionCollection:
        """Create a lifecycle collection owned by this system."""
        return ActionCollection(self)

    def create_pipeline(self, graph: Callable | tuple[object, Callable]) -> SimPipeline:
        """Instantiate one graph runtime from ``(data, graph)``."""
        if isinstance(graph, tuple):
            if len(graph) != 2:
                raise TypeError("Expected a (data, callable) pair")
            data, graph_callable = graph
        else:
            resolved = self._resolve_bound_callable(graph)
            if resolved is None:
                raise TypeError("Expected a (data, callable) pair or a bound method")
            data, graph_callable = resolved
        return SimPipeline(
            data=data,
            graph=graph_callable,
        )

    @abstractmethod
    def build(self) -> None:
        """Resolve dependencies and initialize device-visible runtime data."""

    def _begin_build(self) -> None:
        if self._build_phase_open:
            raise RuntimeError(f"{type(self).__name__} build phase is already open")
        self._build_phase_open = True

    def _end_build(self) -> None:
        if not self._build_phase_open:
            raise RuntimeError(f"{type(self).__name__} build phase is not open")
        self._build_phase_open = False

    def _set_engine(self, engine: SimEngine) -> None:
        self._engine = engine

    def _invalidate(self) -> None:
        self._is_valid = False
