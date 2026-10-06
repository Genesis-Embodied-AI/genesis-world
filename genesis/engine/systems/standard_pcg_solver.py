from __future__ import annotations

import numpy as np
import quadrants as qd

from .linear_pcg import LinearPCG
from .sim_system import SimSystem


@qd.data_oriented
class StandardPCGSolver(SimSystem):
    """CGQ standard PCG solver system."""

    def __init__(self) -> None:
        super().__init__()
        self.is_initialized_host = False

    def build(self) -> None:
        pass

    def init(self, total_dof: int, n_block_rows: int, pcg_tol_rate: float) -> None:
        if self.is_initialized_host:
            raise RuntimeError("StandardPCGSolver is already initialized")
        self.linear_pcg = LinearPCG(total_dof)
        self.n_block_rows = qd.ndarray(qd.i32, shape=())
        self.pcg_tol_rate = qd.ndarray(qd.f64, shape=())
        self.n_block_rows.from_numpy(np.array(n_block_rows, dtype=np.int32))
        self.pcg_tol_rate.from_numpy(np.array(pcg_tol_rate, dtype=np.float64))
        self.is_initialized_host = True

    @qd.func(requires_top_level=True)
    def solve(
        self,
        global_linear_system: qd.template(),
        rigid: qd.template(),
        rigid_forest: qd.template(),
        fem_preconditioner: qd.template(),
        has_rigid: qd.template(),
        has_rigid_forest: qd.template(),
        has_fem: qd.template(),
        max_iter,
    ):
        self.linear_pcg.initialize(
            global_linear_system,
            rigid,
            rigid_forest,
            fem_preconditioner,
            has_rigid,
            has_rigid_forest,
            has_fem,
        )
        while qd.graph.do_while(self.linear_pcg.condition):
            self.linear_pcg.iteration(
                global_linear_system,
                rigid,
                rigid_forest,
                fem_preconditioner,
                has_rigid,
                has_rigid_forest,
                has_fem,
                self.pcg_tol_rate[()],
                max_iter,
            )
