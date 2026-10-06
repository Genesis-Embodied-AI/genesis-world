from __future__ import annotations

from abc import abstractmethod

from .sim_system import SimSystem


class BroadPhaseSystem(SimSystem):
    """CGQ broad-phase system contract."""

    def build(self) -> None:
        from .contact_system import ContactSystem
        from .global_body_manager import GlobalBodyManager
        from .global_surface_manager import GlobalSurfaceManager
        from .global_vertex_manager import GlobalVertexManager

        self.contact = self.require(ContactSystem)
        self.body = self.require(GlobalBodyManager)
        self.surface = self.require(GlobalSurfaceManager)
        self.vertex = self.require(GlobalVertexManager)

    @abstractmethod
    def init_bvh(self, n_triangles: int, n_edges: int, n_codim_verts: int) -> None:
        pass

    @abstractmethod
    def triangle_build(self):
        pass

    @abstractmethod
    def edge_build(self):
        pass

    @abstractmethod
    def trajectory_query(self):
        pass
