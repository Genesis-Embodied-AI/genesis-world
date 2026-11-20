import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch

from genesis.repr_base import RBC
from genesis.utils.data import DataKind

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


@dataclasses.dataclass
class SolverCheckpoint:
    """The arrays of one solver of the given kinds (see 'DataKind'), for 'Solver.__setstate__' to write back.

    A checkpoint '__getstate__' reads holds every kind a step reads from one step to the next (see 'CHECKPOINT_KINDS'),
    so stepping on from it lands where the original went. A trajectory frame may hold the state alone, and writing it
    back recomputes what it leaves out. Each array stands under the name 'Solver.data' gives it, copied where it lives:
    on the device as a torch tensor when zero-copy views exist, as a numpy array otherwise. The static configs come
    along as provenance and are never written back.
    """

    arrays: dict[str, np.ndarray | torch.Tensor]
    configs: dict[str, object]
    kinds: frozenset[DataKind]


@dataclasses.dataclass
class SimulatorCheckpoint:
    """The state a simulation stands in.

    How many steps each environment has run, and what every active solver reads from one step to the next, under the
    name of its class.
    """

    steps: np.ndarray | torch.Tensor
    solvers: dict[str, SolverCheckpoint]


class SimState(RBC):
    """
    The state a simulation stands in between two steps, which 'Scene.reset' restores.

    It holds the state of every active solver under the name of its class (see 'Solver.get_state'), and the number of
    steps the scene ran when it was taken.
    """

    def __init__(self, scene: "Scene", s_global: int, solvers: dict[str, SolverCheckpoint]) -> None:
        self._scene = scene
        self._s_global = s_global
        self._solvers = solvers

    @property
    def scene(self) -> "Scene":
        return self._scene

    @property
    def s_global(self) -> int:
        return self._s_global

    @property
    def solvers(self) -> dict[str, SolverCheckpoint]:
        """The state of every active solver, under the name of its class."""
        return self._solvers
