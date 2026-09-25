"""ImGui overlay package. Re-exports :class:`ImGuiOverlayPlugin` so callers can keep importing
``from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin``."""

from genesis.ext.pyrender.overlay.plugin import ImGuiOverlayPlugin

__all__ = ["ImGuiOverlayPlugin"]
