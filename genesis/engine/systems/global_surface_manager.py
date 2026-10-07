from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimData, SimSystem


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class GlobalSurfaceManager(SimSystem):
    """Organize global surface data and host wiring."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible collision surface topology and area weights."""

        n_surf_triangles: qd.Ndarray
        n_surf_edges: qd.Ndarray
        n_surf_verts: qd.Ndarray
        n_codim_verts: qd.Ndarray
        surf_triangles: qd.Ndarray
        surf_edges: qd.Ndarray
        surf_verts: qd.Ndarray
        vert_dimensions: qd.Ndarray
        vert_area_weights: qd.Ndarray
        edge_area_weights: qd.Ndarray
        face_area_weights: qd.Ndarray
        codim_verts: qd.Ndarray
        codim_vert_area_weights: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._surf_triangles: np.ndarray | None = None
        self._surf_edges: np.ndarray | None = None
        self._surf_verts: np.ndarray | None = None
        self._vert_dimensions: np.ndarray | None = None
        self._surf_vert_area_weights: np.ndarray | None = None
        self._surf_edge_area_weights: np.ndarray | None = None
        self._surf_face_area_weights: np.ndarray | None = None

    def wire_data(
        self,
        surf_triangles: np.ndarray,
        surf_edges: np.ndarray,
        surf_verts: np.ndarray,
        *,
        vert_dimensions: np.ndarray | None = None,
        surf_vert_area_weights: np.ndarray | None = None,
        surf_edge_area_weights: np.ndarray | None = None,
        surf_face_area_weights: np.ndarray | None = None,
    ) -> None:
        self._surf_triangles = surf_triangles
        self._surf_edges = surf_edges
        self._surf_verts = surf_verts
        self._vert_dimensions = vert_dimensions
        self._surf_vert_area_weights = surf_vert_area_weights
        self._surf_edge_area_weights = surf_edge_area_weights
        self._surf_face_area_weights = surf_face_area_weights

    def build(self) -> None:
        from .finite_element import FiniteElementMethod
        from .global_vertex_manager import GlobalVertexManager

        self.fem_system = self.require(FiniteElementMethod)
        self.vertex_system = self.require(GlobalVertexManager)

    def init(self) -> None:
        if self._surf_triangles is None or self._surf_edges is None or self._surf_verts is None:
            raise RuntimeError("GlobalSurfaceManager data has not been wired")
        surf_triangles = self._surf_triangles
        surf_edges = self._surf_edges
        surf_verts = self._surf_verts
        vert_dimensions = self._vert_dimensions
        surf_vert_area_weights = self._surf_vert_area_weights
        surf_edge_area_weights = self._surf_edge_area_weights
        surf_face_area_weights = self._surf_face_area_weights
        triangles = np.ascontiguousarray(surf_triangles, dtype=np.int32).reshape(-1, 3)
        edges = np.ascontiguousarray(surf_edges, dtype=np.int32).reshape(-1, 2)
        vertices = np.ascontiguousarray(surf_verts, dtype=np.int32).reshape(-1)
        triangle_capacity = max(len(triangles), 1)
        edge_capacity = max(len(edges), 1)
        vertex_capacity = max(len(vertices), 1)

        def padded(values, count, capacity, dtype, default):
            if values is None:
                return np.full(capacity, default, dtype=dtype)
            result = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
            if len(result) not in ({count} if count else {0, 1}):
                raise ValueError("Global surface attribute length must match its topology")
            if count == 0:
                return np.full(capacity, default, dtype=dtype)
            return result

        dimension_values = padded(vert_dimensions, len(vertices), vertex_capacity, np.int32, 2)
        vertex_weights = padded(surf_vert_area_weights, len(vertices), vertex_capacity, np.float64, 0.0)
        edge_weights = padded(surf_edge_area_weights, len(edges), edge_capacity, np.float64, 0.0)
        face_weights = padded(surf_face_area_weights, len(triangles), triangle_capacity, np.float64, 0.0)

        def array(dtype, shape, values):
            result = qd.ndarray(dtype, shape=shape)
            result.from_numpy(values)
            return result

        triangle_storage = triangles if len(triangles) else np.zeros((1, 3), dtype=np.int32)
        edge_storage = edges if len(edges) else np.zeros((1, 2), dtype=np.int32)
        vertex_storage = vertices if len(vertices) else np.zeros(1, dtype=np.int32)
        self.data.n_surf_triangles = array(qd.i32, (), np.array(len(triangles), dtype=np.int32))
        self.data.n_surf_edges = array(qd.i32, (), np.array(len(edges), dtype=np.int32))
        self.data.n_surf_verts = array(qd.i32, (), np.array(len(vertices), dtype=np.int32))
        self.data.n_codim_verts = array(qd.i32, (), np.array(0, dtype=np.int32))
        self.data.surf_triangles = array(qd.i32, (triangle_capacity, 3), triangle_storage)
        self.data.surf_edges = array(qd.i32, (edge_capacity, 2), edge_storage)
        self.data.surf_verts = array(qd.i32, (vertex_capacity,), vertex_storage)
        self.data.vert_dimensions = array(qd.i32, (vertex_capacity,), dimension_values)
        self.data.vert_area_weights = array(qd.f64, (vertex_capacity,), vertex_weights)
        self.data.edge_area_weights = array(qd.f64, (edge_capacity,), edge_weights)
        self.data.face_area_weights = array(qd.f64, (triangle_capacity,), face_weights)
        self.data.codim_verts = array(qd.u32, (1,), np.zeros(1, dtype=np.uint32))
        self.data.codim_vert_area_weights = array(qd.f64, (1,), np.zeros(1, dtype=np.float64))
        self._surf_triangles = None
        self._surf_edges = None
        self._surf_verts = None
        self._vert_dimensions = None
        self._surf_vert_area_weights = None
        self._surf_edge_area_weights = None
        self._surf_face_area_weights = None
