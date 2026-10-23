"""Typed descriptions and internal runtime data shared by solvers and couplers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class SystemData:
    """A domain record identified by its type, owner, and index.

    The index is 0 for a solver's aggregate record, the scene index of the entity for per-entity records and the solver
    index of the geometry for per-geometry records.
    """

    owner: "Solver"
    idx: int = 0


DataT = TypeVar("DataT", bound=SystemData)


class SystemDataDescription(ABC, Generic[DataT]):
    """Resolved inputs and dimensions sufficient to allocate a solver's shared data."""

    @abstractmethod
    def allocate(self, owner: "Solver") -> DataT:
        raise NotImplementedError


class SystemDataArray:
    """Scene-owned runtime records, resolved by type and owner during build."""

    def __init__(self):
        self._index: dict[tuple[type[SystemData], "Solver", int], SystemData] = {}
        self._is_built = False

    def allocate(self, owner: "Solver", description: SystemDataDescription[DataT]) -> DataT:
        assert not self._is_built
        return self.add(description.allocate(owner))

    def add(self, data: DataT) -> DataT:
        assert not self._is_built
        key = (type(data), data.owner, data.idx)
        assert key not in self._index, (
            f"Duplicate {type(data).__name__} at index {data.idx} for {type(data.owner).__name__}."
        )
        self._index[key] = data
        return data

    def get(self, data_type: type[DataT], owner: "Solver", idx: int = 0) -> DataT:
        data = self._index.get((data_type, owner, idx))
        assert data is not None, f"No {data_type.__name__} at index {idx} for {type(owner).__name__}."
        return data

    def build(self):
        self._is_built = True

    def clear(self):
        self._index.clear()
        self._is_built = False
