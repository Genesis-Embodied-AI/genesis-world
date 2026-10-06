from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimSystem


@qd.data_oriented
class GlobalSurfaceManager(SimSystem):
    """Own the CGQ global collision surface topology and area weights."""

    def __init__(self) -> None:
        super().__init__()
        self.is_wired_host = False

    def do_build(self) -> None:
        from .finite_element import FiniteElementMethod
        from .global_vertex_manager import GlobalVertexManager

        self.fem = self.require(FiniteElementMethod)
        self.vertex = self.require(GlobalVertexManager)

    def wire_surface_data(
        self,
        surf_triangles: np.ndarray,
        surf_edges: np.ndarray,
        surf_verts: np.ndarray,
    ) -> None:
        if self.is_wired_host:
            raise RuntimeError("GlobalSurfaceManager data is already wired")

        triangles = np.ascontiguousarray(surf_triangles, dtype=np.int32).reshape(-1, 3)
        edges = np.ascontiguousarray(surf_edges, dtype=np.int32).reshape(-1, 2)
        vertices = np.ascontiguousarray(surf_verts, dtype=np.int32).reshape(-1)

        self.n_surf_triangles = qd.ndarray(qd.i32, shape=())
        self.n_surf_edges = qd.ndarray(qd.i32, shape=())
        self.n_surf_verts = qd.ndarray(qd.i32, shape=())
        self.n_codim_verts = qd.ndarray(qd.i32, shape=())
        self.surf_triangles = qd.ndarray(qd.i32, shape=(max(len(triangles), 1), 3))
        self.surf_edges = qd.ndarray(qd.i32, shape=(max(len(edges), 1), 2))
        self.surf_verts = qd.ndarray(qd.i32, shape=(max(len(vertices), 1),))
        self.vert_dimensions = qd.ndarray(qd.i32, shape=(max(len(vertices), 1),))
        self.vert_area_weights = qd.ndarray(qd.f64, shape=(max(len(vertices), 1),))
        self.edge_area_weights = qd.ndarray(qd.f64, shape=(max(len(edges), 1),))
        self.face_area_weights = qd.ndarray(qd.f64, shape=(max(len(triangles), 1),))
        self.codim_verts = qd.ndarray(qd.u32, shape=(1,))
        self.codim_vert_area_weights = qd.ndarray(qd.f64, shape=(1,))

        self.n_surf_triangles.from_numpy(np.array(len(triangles), dtype=np.int32))
        self.n_surf_edges.from_numpy(np.array(len(edges), dtype=np.int32))
        self.n_surf_verts.from_numpy(np.array(len(vertices), dtype=np.int32))
        self.n_codim_verts.from_numpy(np.array(0, dtype=np.int32))
        self.surf_triangles.from_numpy(triangles if len(triangles) else np.zeros((1, 3), dtype=np.int32))
        self.surf_edges.from_numpy(edges if len(edges) else np.zeros((1, 2), dtype=np.int32))
        self.surf_verts.from_numpy(vertices if len(vertices) else np.zeros(1, dtype=np.int32))
        self.vert_dimensions.from_numpy(np.full(max(len(vertices), 1), 2, dtype=np.int32))
        self.vert_area_weights.from_numpy(np.zeros(max(len(vertices), 1), dtype=np.float64))
        self.edge_area_weights.from_numpy(np.zeros(max(len(edges), 1), dtype=np.float64))
        self.face_area_weights.from_numpy(np.zeros(max(len(triangles), 1), dtype=np.float64))
        self.codim_verts.from_numpy(np.zeros(1, dtype=np.uint32))
        self.codim_vert_area_weights.from_numpy(np.zeros(1, dtype=np.float64))
        self.is_wired_host = True

    def wire_vert_dimensions(self, vert_dimensions: np.ndarray) -> None:
        values = np.ascontiguousarray(vert_dimensions, dtype=np.int32).reshape(-1)
        if len(values) != self.vert_dimensions.shape[0]:
            raise ValueError("GlobalSurfaceManager vertex dimensions must match surf_verts")
        self.vert_dimensions.from_numpy(values)

    def wire_area_weights(
        self,
        surf_vert_area_weights: np.ndarray,
        surf_edge_area_weights: np.ndarray,
        surf_face_area_weights: np.ndarray,
    ) -> None:
        vertex_values = np.ascontiguousarray(surf_vert_area_weights, dtype=np.float64).reshape(-1)
        edge_values = np.ascontiguousarray(surf_edge_area_weights, dtype=np.float64).reshape(-1)
        face_values = np.ascontiguousarray(surf_face_area_weights, dtype=np.float64).reshape(-1)
        if len(vertex_values) != self.vert_area_weights.shape[0]:
            raise ValueError("GlobalSurfaceManager vertex area weights must match surf_verts")
        if len(edge_values) not in (0, self.edge_area_weights.shape[0]):
            raise ValueError("GlobalSurfaceManager edge area weights must match surf_edges")
        if len(face_values) not in (0, self.face_area_weights.shape[0]):
            raise ValueError("GlobalSurfaceManager face area weights must match surf_triangles")
        self.vert_area_weights.from_numpy(vertex_values)
        self.edge_area_weights.from_numpy(
            edge_values if len(edge_values) else np.zeros(self.edge_area_weights.shape[0], dtype=np.float64)
        )
        self.face_area_weights.from_numpy(
            face_values if len(face_values) else np.zeros(self.face_area_weights.shape[0], dtype=np.float64)
        )
