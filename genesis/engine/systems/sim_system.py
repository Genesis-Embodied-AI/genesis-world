from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .sim_engine import SimEngine


T = TypeVar("T", bound="SimSystem")


class SimSystem(ABC):
    """Provide build-time registration and dependency lookup for one numerical system."""

    def __init__(self) -> None:
        self._engine: SimEngine | None = None
        self._is_valid = True

    @property
    def engine(self) -> SimEngine:
        engine = self._engine
        if engine is None:
            raise RuntimeError(f"{type(self).__name__} is not registered with a SimEngine")
        return engine

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    def find(self, system_type: type[T]) -> T | None:
        """Return a registered system of the requested type, if one is active."""
        return self.engine.find(system_type)

    def require(self, system_type: type[T]) -> T:
        """Return a registered system of the requested type."""
        system = self.find(system_type)
        if system is None:
            raise RuntimeError(f"Required system {system_type.__name__} is not registered")
        return system

    @abstractmethod
    def do_build(self) -> None:
        """Resolve dependencies and initialize device-visible runtime data."""

    def _set_engine(self, engine: SimEngine) -> None:
        self._engine = engine

    def _invalidate(self) -> None:
        self._is_valid = False
