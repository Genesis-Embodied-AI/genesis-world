from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimData, SimSystem


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
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
        self.data = self.Data()
        self._dt: float | None = None
        self._tol: float | None = None
        self._max_newton_iter: int | None = None
        self._max_pcg_iter: int | None = None
        self._max_ls_iter: int | None = None

    def build(self) -> None:
        pass

    def wire(
        self,
        *,
        dt: float,
        tol: float,
        max_newton_iter: int,
        max_pcg_iter: int,
        max_ls_iter: int,
    ) -> None:
        self._dt = float(dt)
        self._tol = float(tol)
        self._max_newton_iter = int(max_newton_iter)
        self._max_pcg_iter = int(max_pcg_iter)
        self._max_ls_iter = int(max_ls_iter)

    def init(self) -> None:
        if (
            self._dt is None
            or self._tol is None
            or self._max_newton_iter is None
            or self._max_pcg_iter is None
            or self._max_ls_iter is None
        ):
            raise RuntimeError("SimConfig parameters have not been wired")
        dt = self._dt
        tol = self._tol
        max_newton_iter = self._max_newton_iter
        max_pcg_iter = self._max_pcg_iter
        max_ls_iter = self._max_ls_iter
        data = self.data
        data.dt = qd.ndarray(qd.f64, shape=())
        data.tol = qd.ndarray(qd.f64, shape=())
        data.max_newton_iter = qd.ndarray(qd.i64, shape=())
        data.max_pcg_iter = qd.ndarray(qd.i64, shape=())
        data.max_ls_iter = qd.ndarray(qd.i64, shape=())
        data.dt.from_numpy(np.array(dt, dtype=np.float64))
        data.tol.from_numpy(np.array(tol, dtype=np.float64))
        data.max_newton_iter.from_numpy(np.array(max_newton_iter, dtype=np.int64))
        data.max_pcg_iter.from_numpy(np.array(max_pcg_iter, dtype=np.int64))
        data.max_ls_iter.from_numpy(np.array(max_ls_iter, dtype=np.int64))
        self._dt = None
        self._tol = None
        self._max_newton_iter = None
        self._max_pcg_iter = None
        self._max_ls_iter = None
