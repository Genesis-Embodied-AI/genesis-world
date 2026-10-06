from __future__ import annotations

from .sim_system import SimSystem


class BroadPhaseSystem(SimSystem):
    """Broad-phase system contract."""

    def build(self) -> None:
        from .contact_system import ContactSystem
        from .global_body_manager import GlobalBodyManager
        from .global_surface_manager import GlobalSurfaceManager
        from .global_vertex_manager import GlobalVertexManager

        self.contact_system = self.require(ContactSystem)
        self.body_system = self.require(GlobalBodyManager)
        self.surface_system = self.require(GlobalSurfaceManager)
        self.vertex_system = self.require(GlobalVertexManager)
