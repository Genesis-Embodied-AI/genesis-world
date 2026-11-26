from typing import TYPE_CHECKING

from genesis.ext.pyrender.constants import WireframeMode
from genesis.vis.keybindings import Key, Keybind

from ..base import ViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class DefaultControlsPlugin(ViewerPlugin):
    """
    Default keyboard controls for the Genesis viewer.

    This plugin handles the standard viewer keyboard shortcuts for recording, changing render modes, etc.
    """

    def __init__(self):
        super().__init__()

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        keybinds = []

        # Both shortcuts prompt for a destination through 'imgui-bundle', an optional dependency, and come first to
        # keep the on-screen help order.
        try:
            import imgui_bundle  # noqa: F401

            keybinds += [
                Keybind("record_video", Key.R, callback=self._toggle_record_video, allow_overload=True),
                Keybind("save_image", Key.S, callback=self._save_image, allow_overload=True),
            ]
        except ImportError:
            pass

        self.viewer.register_keybinds(
            *keybinds,
            Keybind("reset_camera", Key.Z, callback=self._reset_camera, allow_overload=True),
            Keybind("camera_rotation", Key.A, callback=self._toggle_cam_rotation, allow_overload=True),
            Keybind("shadow", Key.H, callback=self._toggle_shadow, allow_overload=True),
            Keybind("face_normals", Key.F, callback=self._toggle_face_normals, allow_overload=True),
            Keybind("vertex_normals", Key.V, callback=self._toggle_vertex_normals, allow_overload=True),
            Keybind("world_frame", Key.W, callback=self._toggle_world_frame, allow_overload=True),
            Keybind("link_frame", Key.L, callback=self._toggle_link_frame, allow_overload=True),
            Keybind("wireframe", Key.D, callback=self._toggle_wireframe, allow_overload=True),
            Keybind("camera_mode", Key.O, callback=self._toggle_projection_mode, allow_overload=True),
            Keybind("camera_frustum", Key.C, callback=self._toggle_camera_frustum, allow_overload=True),
            Keybind("reload_shader", Key.P, callback=self._reload_shader, allow_overload=True),
            Keybind("fullscreen_mode", Key.F11, callback=self._toggle_fullscreen, allow_overload=True),
        )

    def _toggle_cam_rotation(self):
        self.viewer.is_rotating = not self.viewer.is_rotating
        if self.viewer.is_rotating:
            self.viewer.set_message_text("Rotation On")
        else:
            self.viewer.set_message_text("Rotation Off")

    def _toggle_fullscreen(self):
        self.viewer.set_fullscreen(not self.viewer.fullscreen)
        self.viewer.activate()
        if self.viewer.fullscreen:
            self.viewer.set_message_text("Fullscreen On")
        else:
            self.viewer.set_message_text("Fullscreen Off")

    def _toggle_shadow(self):
        self.viewer.shadow = not self.viewer.shadow
        if self.viewer.shadow:
            self.viewer.set_message_text("Shadows On")
        else:
            self.viewer.set_message_text("Shadows Off")

    def _toggle_world_frame(self):
        if not self.viewer.gs_context.world_frame_shown:
            self.viewer.gs_context.on_world_frame()
            self.viewer.set_message_text("World Frame On")
        else:
            self.viewer.gs_context.off_world_frame()
            self.viewer.set_message_text("World Frame Off")

    def _toggle_link_frame(self):
        # The frames of every kinematic and rigid entity are drawn unless all of them already are
        visualizer = self.scene.visualizer
        if not visualizer.link_frames:
            visualizer.set_link_frames(link_frames=True)
            self.viewer.set_message_text("Link Frame On")
        else:
            visualizer.set_link_frames(link_frames=False)
            self.viewer.set_message_text("Link Frame Off")

    def _toggle_camera_frustum(self):
        if not self.viewer.gs_context.camera_frustum_shown:
            self.viewer.gs_context.on_camera_frustum()
            self.viewer.set_message_text("Camera Frustum On")
        else:
            self.viewer.gs_context.off_camera_frustum()
            self.viewer.set_message_text("Camera Frustum Off")

    def _toggle_face_normals(self):
        self.viewer.show_face_normals = not self.viewer.show_face_normals
        if self.viewer.show_face_normals:
            self.viewer.set_message_text("Face Normals On")
        else:
            self.viewer.set_message_text("Face Normals Off")

    def _toggle_vertex_normals(self):
        self.viewer.show_vertex_normals = not self.viewer.show_vertex_normals
        if self.viewer.show_vertex_normals:
            self.viewer.set_message_text("Vert Normals On")
        else:
            self.viewer.set_message_text("Vert Normals Off")

    def _toggle_record_video(self):
        self.viewer.toggle_recording()

    def _save_image(self):
        self.viewer.save_image()

    def _toggle_projection_mode(self):
        self.viewer.is_perspective = not self.viewer.is_perspective
        self.viewer.set_message_text("Perspective camera" if self.viewer.is_perspective else "Orthographic camera")

    def _toggle_wireframe(self):
        self.viewer.wireframe_mode = WireframeMode((self.viewer.wireframe_mode + 1) % len(WireframeMode))
        self.viewer.set_message_text(
            ("Default Wireframe", "Flip Wireframe", "All Wireframe", "All Solid")[self.viewer.wireframe_mode]
        )

    def _reset_camera(self):
        self.viewer._reset_view()

    def _reload_shader(self):
        self.viewer._renderer.reload_program()
