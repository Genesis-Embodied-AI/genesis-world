from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node
    from genesis.ext.pyrender.viewer import Viewer


EVENT_HANDLE_STATE = Literal[True] | None
EVENT_HANDLED: Literal[True] = True


class ViewerPlugin:
    """
    Base class for handling pyglet.window.Window events.
    """

    def __init__(self):
        self.viewer: "Viewer | None" = None
        self.camera: "Node | None" = None
        self.scene: "Scene | None" = None

    def build(self, viewer: "Viewer", camera: "Node", scene: "Scene"):
        """Build and initialize the plugin with pyrender viewer context."""

        self.viewer = viewer
        self.camera = camera
        self.scene = scene

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        pass

    def on_key_press(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_key_release(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_resize(self, width: int, height: int) -> EVENT_HANDLE_STATE:
        pass

    def update_on_sim_step(self) -> None:
        pass

    def on_draw(self) -> None:
        pass

    def on_close(self) -> None:
        pass
