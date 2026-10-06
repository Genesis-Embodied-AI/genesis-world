from __future__ import annotations

from abc import abstractmethod

from .sim_system import SimSystem


class ContactConstitution(SimSystem):
    """Base contract for graph-native contact constitutions."""

    def build(self) -> None:
        from .contact_system import ContactSystem
        from .global_surface_manager import GlobalSurfaceManager
        from .global_vertex_manager import GlobalVertexManager

        self.contact_system = self.require(ContactSystem)
        self.surface_system = self.require(GlobalSurfaceManager)
        self.vertex_system = self.require(GlobalVertexManager)
        self.actions = self.create_action_protocol()

    def resolve_actions(self) -> dict[str, object]:
        if self.is_building:
            raise RuntimeError("Contact constitution actions are available only after build")
        return {name: action.invocation for name, action in self.actions.items()}

    @abstractmethod
    def create_action_protocol(self) -> dict[str, object]:
        """Create the complete stateless numerical action protocol."""
