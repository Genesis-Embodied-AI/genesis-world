from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimData, SimSystem


class SimConfig(SimSystem):
    """Organize numerical controls shared by the global Newton pipeline."""

    @qd.data_oriented
    class Data(SimData):
        dt: qd.Ndarray
        tol: qd.Ndarray
        max_newton_iter: qd.Ndarray
        max_pcg_iter: qd.Ndarray
        max_ls_iter: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = get_sim_config_data()

    def build(self) -> None:
        pass


def get_sim_config_data() -> SimConfig.Data:
    dt = qd.ndarray(qd.f64, shape=())
    tol = qd.ndarray(qd.f64, shape=())
    max_newton_iter = qd.ndarray(qd.i64, shape=())
    max_pcg_iter = qd.ndarray(qd.i64, shape=())
    max_ls_iter = qd.ndarray(qd.i64, shape=())
    dt.from_numpy(np.array(0.0, dtype=np.float64))
    tol.from_numpy(np.array(0.0, dtype=np.float64))
    max_newton_iter.from_numpy(np.array(0, dtype=np.int64))
    max_pcg_iter.from_numpy(np.array(0, dtype=np.int64))
    max_ls_iter.from_numpy(np.array(0, dtype=np.int64))
    data = SimConfig.Data()
    data.dt = dt
    data.tol = tol
    data.max_newton_iter = max_newton_iter
    data.max_pcg_iter = max_pcg_iter
    data.max_ls_iter = max_ls_iter
    return data
