from __future__ import annotations

import quadrants as qd

from .sim_system import SimSystem


@qd.data_oriented
class SimConfig(SimSystem):
    """Store the numerical controls shared by the global Newton pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.dt = qd.ndarray(qd.f64, shape=())
        self.tol = qd.ndarray(qd.f64, shape=())
        self.max_newton_iter = qd.ndarray(qd.i64, shape=())
        self.max_pcg_iter = qd.ndarray(qd.i64, shape=())
        self.max_ls_iter = qd.ndarray(qd.i64, shape=())

    def build(self) -> None:
        pass
