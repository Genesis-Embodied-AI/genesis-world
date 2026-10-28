"""Grid state for the stable fluid (SF) solver."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs

from . import SystemData, SystemDataDescription

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class SFData(SystemData):
    """The grid holds the velocity, pressure and jet concentrations of every cell, shared by all environments."""

    grid: qd.Field


@dataclass(frozen=True, kw_only=True)
class SFDescription(SystemDataDescription[SFData]):
    res: tuple[int, int, int]
    n_jets: int

    def allocate(self, owner: "Solver") -> SFData:
        cell_state = qd.types.struct(
            v=gs.qd_vec3,
            v_tmp=gs.qd_vec3,
            div=gs.qd_float,
            p=gs.qd_float,
            q=qd.types.vector(self.n_jets, gs.qd_float),
        )
        return SFData(owner=owner, grid=cell_state.field(shape=self.res, layout=qd.Layout.SOA))
