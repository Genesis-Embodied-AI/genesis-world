from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimData, SimSystem


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class GlobalVertexManager(SimSystem):
    """Organize global vertex data and dependency relationships."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible global vertex index and contact attributes."""

        n_verts: qd.Ndarray
        positions: qd.Ndarray
        safe_positions: qd.Ndarray
        trajectory_end_positions: qd.Ndarray
        x_bar: qd.Ndarray
        body_id: qd.Ndarray
        geometry_id: qd.Ndarray
        geometry_source: qd.Ndarray
        source_geometry_id: qd.Ndarray
        geometry_environment: qd.Ndarray
        thicknesses: qd.Ndarray
        d_hats: qd.Ndarray
        is_fixed: qd.Ndarray
        in_contact: qd.Ndarray
        path_rot: qd.Ndarray
        path_pivot: qd.Ndarray
        path_pivot_disp: qd.Ndarray
        path_inflation: qd.Ndarray
        path_kind: qd.Ndarray
        path_speed: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._wire_args = None

    def wire_data(
        self,
        n_verts: int,
        *,
        thicknesses: np.ndarray | None = None,
        d_hats: np.ndarray | None = None,
        is_fixed: np.ndarray | None = None,
        geometry_ids: np.ndarray | None = None,
        geometry_sources: np.ndarray | None = None,
        source_geometry_ids: np.ndarray | None = None,
        geometry_environments: np.ndarray | None = None,
    ) -> None:
        self._wire_args = (
            n_verts,
            thicknesses,
            d_hats,
            is_fixed,
            geometry_ids,
            geometry_sources,
            source_geometry_ids,
            geometry_environments,
        )

    def build(self) -> None:
        from .finite_element import FiniteElementMethod

        self.fem_system = self.require(FiniteElementMethod)

    def init(self) -> None:
        if self._wire_args is None:
            raise RuntimeError("GlobalVertexManager data has not been wired")
        (
            n_verts,
            thicknesses,
            d_hats,
            is_fixed,
            geometry_ids,
            geometry_sources,
            source_geometry_ids,
            geometry_environments,
        ) = self._wire_args
        if n_verts < 0:
            raise ValueError("Global vertex count must be non-negative")
        capacity = max(n_verts, 1)

        def values_or_default(values, dtype, default):
            if values is None:
                return np.full(capacity, default, dtype=dtype)
            result = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
            if len(result) != n_verts:
                raise ValueError("Global vertex input length must match n_verts")
            if n_verts == 0:
                return np.full(capacity, default, dtype=dtype)
            return result

        thickness_values = values_or_default(thicknesses, np.float64, 0.0)
        d_hat_values = values_or_default(d_hats, np.float64, 0.0)
        fixed_values = values_or_default(is_fixed, np.int32, 0)
        geometry_id_values = values_or_default(geometry_ids, np.int32, -1)
        geometry_source_values = values_or_default(geometry_sources, np.int32, -1)
        source_geometry_id_values = values_or_default(source_geometry_ids, np.int32, -1)
        geometry_environment_values = values_or_default(geometry_environments, np.int32, -1)
        if np.any(thickness_values[:n_verts] < 0.0):
            raise ValueError("Global vertex thicknesses must be non-negative")
        if d_hats is not None and np.any(d_hat_values[:n_verts] <= 0.0):
            raise ValueError("Global vertex d_hats must be positive")
        if np.any((fixed_values[:n_verts] != 0) & (fixed_values[:n_verts] != 1)):
            raise ValueError("Global vertex fixed flags must be zero or one")
        if geometry_ids is not None and np.any(geometry_id_values[:n_verts] < 0):
            raise ValueError("Global vertex geometry IDs must be non-negative")
        if geometry_sources is not None and np.any(
            (geometry_source_values[:n_verts] != 0) & (geometry_source_values[:n_verts] != 1)
        ):
            raise ValueError("Global vertex geometry sources must be zero or one")
        if source_geometry_ids is not None and np.any(source_geometry_id_values[:n_verts] < 0):
            raise ValueError("Global vertex source geometry IDs must be non-negative")
        if geometry_environments is not None and np.any(geometry_environment_values[:n_verts] < 0):
            raise ValueError("Global vertex geometry environments must be non-negative")

        def array(dtype, shape, values):
            result = qd.ndarray(dtype, shape=shape)
            result.from_numpy(values)
            return result

        zeros_3 = np.zeros((capacity, 3), dtype=np.float64)
        self.data.n_verts = array(qd.i32, (), np.array(n_verts, dtype=np.int32))
        self.data.positions = array(qd.f64, (capacity, 3), zeros_3)
        self.data.safe_positions = array(qd.f64, (capacity, 3), zeros_3)
        self.data.trajectory_end_positions = array(qd.f64, (capacity, 3), zeros_3)
        self.data.x_bar = array(qd.f64, (capacity, 3), zeros_3)
        self.data.body_id = array(qd.i32, (capacity,), np.full(capacity, -1, dtype=np.int32))
        self.data.geometry_id = array(qd.i32, (capacity,), geometry_id_values)
        self.data.geometry_source = array(qd.i32, (capacity,), geometry_source_values)
        self.data.source_geometry_id = array(qd.i32, (capacity,), source_geometry_id_values)
        self.data.geometry_environment = array(qd.i32, (capacity,), geometry_environment_values)
        self.data.thicknesses = array(qd.f64, (capacity,), thickness_values)
        self.data.d_hats = array(qd.f64, (capacity,), d_hat_values)
        self.data.is_fixed = array(qd.i32, (capacity,), fixed_values)
        self.data.in_contact = array(qd.i32, (capacity,), np.zeros(capacity, dtype=np.int32))
        self.data.path_rot = array(qd.f64, (capacity, 3), zeros_3)
        self.data.path_pivot = array(qd.f64, (capacity, 3), zeros_3)
        self.data.path_pivot_disp = array(qd.f64, (capacity, 3), zeros_3)
        self.data.path_inflation = array(qd.f64, (capacity,), np.zeros(capacity, dtype=np.float64))
        self.data.path_kind = array(qd.i32, (capacity,), np.zeros(capacity, dtype=np.int32))
        self.data.path_speed = array(qd.f64, (capacity,), np.zeros(capacity, dtype=np.float64))
        self._wire_args = None


@qd.func(requires_top_level=True)
def record_safe_positions(data: qd.template()):
    for i_vertex in range(data.n_verts[()]):
        for axis in qd.static(range(3)):
            data.safe_positions[i_vertex, axis] = data.positions[i_vertex, axis]


@qd.func(requires_top_level=True)
def reset_trajectory(data: qd.template()):
    for i_vertex in range(data.n_verts[()]):
        for axis in qd.static(range(3)):
            value = data.positions[i_vertex, axis]
            data.safe_positions[i_vertex, axis] = value
            data.trajectory_end_positions[i_vertex, axis] = value
            data.path_rot[i_vertex, axis] = 0.0
            data.path_pivot[i_vertex, axis] = 0.0
            data.path_pivot_disp[i_vertex, axis] = 0.0
        data.path_inflation[i_vertex] = 0.0
        data.path_kind[i_vertex] = 0
        data.path_speed[i_vertex] = 0.0


@qd.func(requires_top_level=True)
def zero_in_contact(data: qd.template()):
    for i_vertex in range(data.n_verts[()]):
        data.in_contact[i_vertex] = 0
