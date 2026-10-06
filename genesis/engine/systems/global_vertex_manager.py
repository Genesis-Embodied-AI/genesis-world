from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimSystem


@qd.data_oriented
class GlobalVertexManager(SimSystem):
    """Own the CGQ global world-space vertex index and contact attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.is_initialized_host = False

    def build(self) -> None:
        from .finite_element import FiniteElementMethod

        self.fem = self.require(FiniteElementMethod)

    def init(self, n_verts: int) -> None:
        if self.is_initialized_host:
            raise RuntimeError("GlobalVertexManager is already initialized")
        if n_verts < 0:
            raise ValueError("GlobalVertexManager vertex count must be non-negative")

        capacity = max(n_verts, 1)
        self.n_verts = qd.ndarray(qd.i32, shape=())
        self.positions = qd.ndarray(qd.f64, shape=(capacity, 3))
        self.safe_positions = qd.ndarray(qd.f64, shape=(capacity, 3))
        self.trajectory_end_positions = qd.ndarray(qd.f64, shape=(capacity, 3))
        self.x_bar = qd.ndarray(qd.f64, shape=(capacity, 3))
        self.body_id = qd.ndarray(qd.i32, shape=(capacity,))
        self.geometry_id = qd.ndarray(qd.i32, shape=(capacity,))
        self.geometry_source = qd.ndarray(qd.i32, shape=(capacity,))
        self.source_geometry_id = qd.ndarray(qd.i32, shape=(capacity,))
        self.geometry_environment = qd.ndarray(qd.i32, shape=(capacity,))
        self.thicknesses = qd.ndarray(qd.f64, shape=(capacity,))
        self.d_hats = qd.ndarray(qd.f64, shape=(capacity,))
        self.is_fixed = qd.ndarray(qd.i32, shape=(capacity,))
        self.in_contact = qd.ndarray(qd.i32, shape=(capacity,))
        self.path_rot = qd.ndarray(qd.f64, shape=(capacity, 3))
        self.path_pivot = qd.ndarray(qd.f64, shape=(capacity, 3))
        self.path_pivot_disp = qd.ndarray(qd.f64, shape=(capacity, 3))
        self.path_inflation = qd.ndarray(qd.f64, shape=(capacity,))
        self.path_kind = qd.ndarray(qd.i32, shape=(capacity,))
        self.path_speed = qd.ndarray(qd.f64, shape=(capacity,))

        self.n_verts.from_numpy(np.array(n_verts, dtype=np.int32))
        self.positions.from_numpy(np.zeros((capacity, 3), dtype=np.float64))
        self.safe_positions.from_numpy(np.zeros((capacity, 3), dtype=np.float64))
        self.trajectory_end_positions.from_numpy(np.zeros((capacity, 3), dtype=np.float64))
        self.x_bar.from_numpy(np.zeros((capacity, 3), dtype=np.float64))
        self.body_id.from_numpy(np.full(capacity, -1, dtype=np.int32))
        self.geometry_id.from_numpy(np.full(capacity, -1, dtype=np.int32))
        self.geometry_source.from_numpy(np.full(capacity, -1, dtype=np.int32))
        self.source_geometry_id.from_numpy(np.full(capacity, -1, dtype=np.int32))
        self.geometry_environment.from_numpy(np.full(capacity, -1, dtype=np.int32))
        self.thicknesses.from_numpy(np.zeros(capacity, dtype=np.float64))
        self.d_hats.from_numpy(np.zeros(capacity, dtype=np.float64))
        self.is_fixed.from_numpy(np.zeros(capacity, dtype=np.int32))
        self.in_contact.from_numpy(np.zeros(capacity, dtype=np.int32))
        self.path_rot.from_numpy(np.zeros((capacity, 3), dtype=np.float64))
        self.path_pivot.from_numpy(np.zeros((capacity, 3), dtype=np.float64))
        self.path_pivot_disp.from_numpy(np.zeros((capacity, 3), dtype=np.float64))
        self.path_inflation.from_numpy(np.zeros(capacity, dtype=np.float64))
        self.path_kind.from_numpy(np.zeros(capacity, dtype=np.int32))
        self.path_speed.from_numpy(np.zeros(capacity, dtype=np.float64))
        self.is_initialized_host = True

    def wire_thickness_data(self, thicknesses: np.ndarray) -> None:
        values = np.ascontiguousarray(thicknesses, dtype=np.float64).reshape(-1)
        if len(values) != self.positions.shape[0]:
            raise ValueError("GlobalVertexManager thickness count must match n_verts")
        if np.any(values < 0.0):
            raise ValueError("GlobalVertexManager thicknesses must be non-negative")
        self.thicknesses.from_numpy(values)

    def wire_d_hat_data(self, d_hats: np.ndarray) -> None:
        values = np.ascontiguousarray(d_hats, dtype=np.float64).reshape(-1)
        if len(values) != self.positions.shape[0]:
            raise ValueError("GlobalVertexManager d_hat count must match n_verts")
        if np.any(values <= 0.0):
            raise ValueError("GlobalVertexManager d_hats must be positive")
        self.d_hats.from_numpy(values)

    def wire_is_fixed_data(self, is_fixed: np.ndarray) -> None:
        values = np.ascontiguousarray(is_fixed, dtype=np.int32).reshape(-1)
        if len(values) != self.positions.shape[0]:
            raise ValueError("GlobalVertexManager fixed-flag count must match n_verts")
        if np.any((values != 0) & (values != 1)):
            raise ValueError("GlobalVertexManager fixed flags must be zero or one")
        self.is_fixed.from_numpy(values)

    def wire_geometry_id_data(self, geometry_ids: np.ndarray) -> None:
        values = np.ascontiguousarray(geometry_ids, dtype=np.int32).reshape(-1)
        if len(values) != self.positions.shape[0]:
            raise ValueError("GlobalVertexManager geometry ID count must match n_verts")
        if np.any(values < 0):
            raise ValueError("GlobalVertexManager geometry IDs must be non-negative")
        self.geometry_id.from_numpy(values)

    def wire_geometry_source_data(
        self,
        geometry_sources: np.ndarray,
        source_geometry_ids: np.ndarray,
        geometry_environments: np.ndarray,
    ) -> None:
        sources = np.ascontiguousarray(
            geometry_sources,
            dtype=np.int32,
        ).reshape(-1)
        source_ids = np.ascontiguousarray(
            source_geometry_ids,
            dtype=np.int32,
        ).reshape(-1)
        environments = np.ascontiguousarray(
            geometry_environments,
            dtype=np.int32,
        ).reshape(-1)
        if not (len(sources) == len(source_ids) == len(environments) == self.positions.shape[0]):
            raise ValueError("GlobalVertexManager geometry source data must match n_verts")
        if np.any((sources != 0) & (sources != 1)) or np.any(source_ids < 0) or np.any(environments < 0):
            raise ValueError("GlobalVertexManager geometry source data is invalid")
        self.geometry_source.from_numpy(sources)
        self.source_geometry_id.from_numpy(source_ids)
        self.geometry_environment.from_numpy(environments)

    @qd.func(requires_top_level=True)
    def record_safe_positions(self):
        for i_vertex in range(self.n_verts[()]):
            for axis in qd.static(range(3)):
                self.safe_positions[i_vertex, axis] = self.positions[i_vertex, axis]

    @qd.func(requires_top_level=True)
    def reset_trajectory(self):
        for i_vertex in range(self.n_verts[()]):
            for axis in qd.static(range(3)):
                value = self.positions[i_vertex, axis]
                self.safe_positions[i_vertex, axis] = value
                self.trajectory_end_positions[i_vertex, axis] = value
                self.path_rot[i_vertex, axis] = 0.0
                self.path_pivot[i_vertex, axis] = 0.0
                self.path_pivot_disp[i_vertex, axis] = 0.0
            self.path_inflation[i_vertex] = 0.0
            self.path_kind[i_vertex] = 0
            self.path_speed[i_vertex] = 0.0

    @qd.func(requires_top_level=True)
    def zero_in_contact(self):
        for i_vertex in range(self.n_verts[()]):
            self.in_contact[i_vertex] = 0
