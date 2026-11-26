"""A pyglet-based interactive 3D scene viewer."""

import copy
import os
import shutil
import sys
import threading
import time
from contextlib import nullcontext
from threading import Event, RLock, Semaphore, Thread
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

import OpenGL
import pyglet

import genesis as gs
from genesis.utils.cache import get_cache_dir
from genesis.utils.misc import get_default_screen
from genesis.utils.video_encoder import VideoEncoder
from genesis.vis.keybindings import Key, KeyAction, Keybind, Keybindings, KeyMod
from genesis.vis.viewer_plugins import EVENT_HANDLE_STATE, EVENT_HANDLED, ViewerPlugin

from .camera import OrthographicCamera, PerspectiveCamera
from .constants import (
    DEFAULT_SCENE_SCALE,
    DEFAULT_Z_FAR,
    FONT_COLOR_DARKMODE,
    FONT_COLOR_LIGHTMODE,
    FONT_SIZE,
    MIN_OPEN_GL_MAJOR,
    MIN_OPEN_GL_MINOR,
    SOFTWARE_RENDERERS,
    TARGET_OPEN_GL_MAJOR,
    TARGET_OPEN_GL_MINOR,
    TEXT_PADDING,
    RenderFlags,
    TextAlign,
    WireframeMode,
)
from .node import Node
from .renderer import Renderer
from .trackball import Trackball, TrackballMode

if TYPE_CHECKING:
    from genesis.vis.rasterizer_context import RasterizerContext

HELP_TEXT_KEY = Key.I
HELP_TEXT_KEYBIND_NAME = "toggle_instructions"


pyglet.options["shadow_window"] = False
if pyglet.options.get("dpi_scaling") != "real":
    pyglet.options["dpi_scaling"] = "real"


class _OffscreenRequest(NamedTuple):
    """A render that another thread asks of the thread owning the GL context (see 'render_offscreen')."""

    camera_node: Node
    render_target: Renderer
    rgb: bool
    depth: bool
    seg: bool
    normal: bool
    split_envs: bool
    env_idx: int | None
    debug_view: bool


class Viewer(pyglet.window.Window):
    """An interactive viewer for 3D scenes.

    The viewer's camera is separate from the scene's, but will take on
    the parameters of the scene's main view camera and start in the same pose.
    If the scene does not have a camera, a suitable default will be provided.

    Parameters
    ----------
    context : RasterizerContext
        The rasterizer context whose scene the viewer shows.
    viewport_size : (2,) int, optional
        The width and height of the window, in pixels. Defaults to (640, 480).
    run_in_thread : bool, optional
        Whether the viewer runs its event loop in a thread of its own. Defaults to False.
    view_center : (3,) float, optional
        The point that the camera looks at and rotates about. Defaults to the centroid of the scene.
    shadow : bool, optional
        Whether the lights cast shadows. Defaults to False.
    plane_reflection : bool, optional
        Whether the floor reflects the scene. Defaults to False.
    plugins : list of ViewerPlugin, optional
        The plugins of the viewer. Defaults to none.
    enable_help_text : bool, optional
        Whether the window shows the keyboard instructions and the messages of the plugins. Defaults to True.
    debug_view : bool, optional
        Whether the window draws the markers and the debug representation of the entities. Defaults to False.
    window_title : str, optional
        The title of the window. Defaults to "Scene Viewer".
    refresh_rate : float, optional
        The rate at which the window is drawn, in Hertz. Defaults to 30.0.

    Note
    ----
    The basic commands for moving about the scene are given as follows:

    - **Rotating about the scene**: Hold the left mouse button and drag the cursor.
    - **Panning**: Hold ALT or SHIFT, then hold the left mouse button and drag the cursor, or hold the middle mouse
      button and drag the cursor.
    - **Zooming**: Scroll the mouse wheel, or hold CTRL and the left mouse button, or the right mouse button, and drag
      the cursor.

    Keyboard shortcuts are registered by ``DefaultControlsPlugin`` (see
    ``genesis/vis/viewer_plugins/plugins/default_controls.py``) and surfaced in the on-screen help overlay. Press the
    help key to toggle it.
    """

    def __init__(
        self,
        context: "RasterizerContext",
        viewport_size=None,
        run_in_thread=False,
        view_center=None,
        shadow=False,
        plane_reflection=False,
        plugins=(),
        enable_help_text=True,
        debug_view=False,
        window_title="Scene Viewer",
        refresh_rate=30.0,
    ):
        #######################################################################
        # Save attributes and flags
        #######################################################################
        if viewport_size is None:
            viewport_size = (640, 480)
        self.gs_context = context
        self._scene = context._scene
        # ``_viewport_size`` tracks the current window content area and follows ``on_resize`` (the OS may
        # clamp the window to a smaller area than requested); ``_offscreen_viewport_size`` keeps the size the
        # caller asked for so the offscreen renderer can always honor it regardless of window clamping.
        self._viewport_size = viewport_size
        self._offscreen_viewport_size = viewport_size
        self._render_lock = RLock()
        self._initialized_event = Event()
        self._is_active = False
        self._is_software = False
        self._is_refreshing = False
        self._exception = None
        self._thread: Optional[Thread] = None
        self._run_in_thread = run_in_thread
        self._seg_node_map = context.seg_node_map

        self._offscreen_event = Event()
        self._offscreen_pending_render = None
        self._offscreen_pending_close = None
        # Renderers retired by rebind() on the stepping thread, waiting to be deleted on the
        # thread that owns the GL context (see _flush_retired_renderers).
        self._retired_renderers = []
        self._offscreen_semaphore = Semaphore(0)
        self._offscreen_result = None

        self._video_recorder = None
        # Step counter of the last frame written to the video, so a paused (non-advancing) simulation does not fill
        # the recording with duplicate frozen frames.
        self._last_recorded_t = -1

        self._view_center = view_center
        self._plane_reflection = plane_reflection
        self._debug_view = debug_view
        self._window_title = window_title
        self._refresh_rate = refresh_rate
        self._is_recording = False
        self._is_mouse_pressed = False
        self._save_directory = None

        # The rendering state that the keyboard shortcuts and the plugins toggle
        self.wireframe_mode = WireframeMode.MATERIAL
        self.shadow = shadow
        self.show_vertex_normals = False
        self.show_face_normals = False
        self.is_rotating = False
        self._is_perspective = True

        self._keybindings: Keybindings = Keybindings()
        self._held_keys: dict[tuple[int, int], bool] = {}

        #######################################################################
        # Set up camera node
        #######################################################################
        self._setup_main_camera()

        # Setup help text functionality
        is_dark_mode = np.mean(context.background_color[0:3]) < 0.5
        self._font_color = FONT_COLOR_DARKMODE if is_dark_mode else FONT_COLOR_LIGHTMODE
        self._enable_help_text = enable_help_text
        if self._enable_help_text:
            self._collapse_instructions = True
            instr_key_str = str(Key(HELP_TEXT_KEY))
            self._instr_texts: tuple[list[str], list[str]] = (
                [f"> [{instr_key_str}]: show keyboard instructions"],
                [f"< [{instr_key_str}]: hide keyboard instructions"],
            )
            self._key_instr_texts: list[str] = []
            self._message_text = None
            self._ticks_till_fade = 2.0 / 3.0 * self._refresh_rate
            self._message_opac = 1.0 + self._ticks_till_fade
            self.register_keybinds(
                Keybind(
                    HELP_TEXT_KEYBIND_NAME,
                    HELP_TEXT_KEY,
                    callback=self._toggle_instructions,
                    protected=True,
                    allow_overload=True,
                )
            )

        # Setup viewer plugins
        self.plugins: list[ViewerPlugin] = []
        for plugin in plugins:
            self.register_plugin(plugin)

        #######################################################################
        # Initialize OpenGL context and renderer
        #######################################################################
        self._renderer = Renderer(*self._viewport_size, context.jit)
        self._is_active = True

        # Starting the viewer would raise an exception if the OpenGL context is invalid for some reason. This exception
        # must be caught in order to implement some fallback mechanism. One may want to start the viewer from the main
        # thread while the running loop would be running on a background thread. However, this approach is not possible
        # because all access to the OpenGL context must be done from the thread that created it in the first place. As
        # a result, the logic for catching an invalid OpenGL context must be implemented at the thread-level.
        if self._run_in_thread:
            self._initialized_event.clear()
            self._thread = Thread(target=self.start, daemon=True)
            self._thread.start()
            self._initialized_event.wait()
            if not self._is_active:
                if self._exception:
                    raise RuntimeError("Unable to initialize an OpenGL 3+ context.") from self._exception
                raise OpenGL.error.Error("Invalid OpenGL context (unknown exception).")

    @property
    def scene(self):
        """:class:`.Scene` : The scene being visualized."""
        return self._scene

    @property
    def viewport_size(self):
        """(2,) int : The width and height of the viewing window."""
        return self._viewport_size

    @property
    def render_lock(self):
        """:class:`threading.RLock` : If acquired, prevents the viewer from
        rendering until released.

        Run :meth:`.Viewer.render_lock.acquire` before making updates to
        the scene in a different thread, and run
        :meth:`.Viewer.render_lock.release` once you're done to let the viewer
        continue.
        """
        return self._render_lock

    @property
    def is_perspective(self):
        """bool : Whether the camera of the window projects in perspective, orthographically otherwise."""
        return self._is_perspective

    @is_perspective.setter
    def is_perspective(self, value):
        self._is_perspective = value
        self._camera_node.camera = self._default_persp_cam if value else self._default_orth_cam

    @property
    def trackball(self):
        """Trackball : The camera of the window, which the mouse moves."""
        return self._trackball

    @property
    def is_software(self):
        """bool : Whether the OpenGL context of the window is rendered on the CPU, which makes rendering slow."""
        return self._is_software

    @property
    def is_active(self):
        """bool : `True` if the viewer is active, or `False` if it has
        been closed.
        """
        return self._is_active and (not self._run_in_thread or self._thread.is_alive())

    @property
    def run_in_thread(self):
        """bool : Whether the viewer was run in a separate thread."""
        return self._run_in_thread

    @property
    def is_recording(self):
        """bool : Whether the window is being recorded to a video file (see 'toggle_recording')."""
        return self._is_recording

    def _setup_main_camera(self):
        # The window renders from a camera node of its own, a copy of the main camera of the scene, which it restores as
        # the main camera once it closes. A rebind to a rebuilt scene graph sets it up again.
        main_camera_node = self.scene.main_camera_node
        assert isinstance(main_camera_node.camera, PerspectiveCamera)
        self._prior_main_camera_node = main_camera_node
        self._default_persp_cam = copy.copy(main_camera_node.camera)
        self._default_camera_pose = self.scene.get_pose(main_camera_node)
        # The orthographic projection frames the scene, within the depth range of the perspective one
        scale = self.scene.scale if self.scene.scale > 1e-6 else 1.0
        zfar = self._default_persp_cam.zfar
        if zfar is None:
            zfar = max(10.0 * self.scene.scale, DEFAULT_Z_FAR)
        self._default_orth_cam = OrthographicCamera(
            xmag=scale, ymag=scale, znear=self._default_persp_cam.znear, zfar=zfar
        )
        self._orth_cam_reset_mags = (scale, scale)

        camera = self._default_persp_cam if self.is_perspective else self._default_orth_cam
        self._camera_node = Node(camera=camera, matrix=self._default_camera_pose)
        self.scene.add_node(self._camera_node)
        self.scene.main_camera_node = self._camera_node
        self._reset_view()

    def rebind(self, context, plugins):
        """Show a rebuilt scene graph in the window already open, keeping its OpenGL context.

        The scene graph, the segmentation map, the renderer, the camera node and the plugins follow the new context. No
        render may run on the old scene graph meanwhile.

        Parameters
        ----------
        context : RasterizerContext
            The rasterizer context of the rebuilt scene.
        plugins : list of ViewerPlugin
            The plugins of the viewer.
        """
        with self._render_lock:
            self.gs_context = context
            self._scene = context._scene
            self._seg_node_map = context.seg_node_map
            if self._renderer is not None:
                # rebind() runs on the stepping thread, which has no current GL context: deleting GL objects
                # here segfaults. Retire the old renderer instead; the render thread deletes it right before
                # its next draw, so shared GL objects (e.g. reused textures) are released - and re-uploaded -
                # before the new renderer can record them as live (see _flush_retired_renderers).
                self._retired_renderers.append(self._renderer)
            self._renderer = Renderer(*self._viewport_size, context.jit)
            self._setup_main_camera()
            self.plugins = []
            for plugin in plugins:
                self.register_plugin(plugin)

    def register_plugin(self, plugin: ViewerPlugin) -> None:
        """
        Register a viewer plugin.

        Parameters
        ----------
        plugin : :class:`.ViewerPlugin`
            The viewer plugin to add.
        """
        plugin.build(self, self._camera_node, self.gs_context.scene)
        # Register pyglet.window event handlers from the plugin
        self.push_handlers(plugin)
        # Append via copy-on-write so a dispatch loop already iterating self.plugins (on_draw / sim-step) keeps walking
        # its own snapshot. This lets a plugin register another one from inside its on_draw callback.
        self.plugins = self.plugins + [plugin]

    def register_keybinds(self, /, *keybinds: Keybind, overwrite: bool = False) -> None:
        """Register key bindings, which call their function when their key is pressed.

        Parameters
        ----------
        keybinds : Keybind
            The key bindings.
        overwrite : bool, optional
            Whether a key binding replaces the one already bound to its key. Defaults to False.
        """
        for keybind in keybinds:
            self._keybindings.register(keybind, overwrite)
        self._update_instr_texts()

    def remap_keybind(
        self,
        keybind_name: str,
        new_key_code: Key,
        new_key_mods: tuple[KeyMod] | None,
        new_key_action: KeyAction = KeyAction.PRESS,
    ) -> None:
        """
        Remap an existing keybind to a new key combination.

        Parameters
        ----------
        keybind_name : str
            The name of the keybind to remap.
        new_key_code : int
            The new key code from pyglet.
        new_key_mods : tuple[KeyMod] | None
            The new modifier keys pressed.
        new_key_action : KeyAction, optional
            The new type of key action. If not provided, the key action of the old keybind is used.
        """
        self._keybindings.rebind(
            keybind_name,
            new_key_code,
            new_key_mods,
            new_key_action,
        )
        self._update_instr_texts()

    def remove_keybind(self, keybind_name: str) -> None:
        """
        Remove an existing keybind.

        Parameters
        ----------
        keybind_name : str
            The name of the keybind to remove.
        """
        self._keybindings.remove(keybind_name)
        self._update_instr_texts()

    def close(self):
        """Close the window, and wait until it is closed, so that the caller may change the scene right after."""
        if self._run_in_thread:
            self._is_active = False
            while self._thread.is_alive():
                time.sleep(1.0 / self._refresh_rate)
        else:
            viewer_thread = self._thread or threading.main_thread()
            if viewer_thread != threading.current_thread():
                gs.raise_exception("'Viewer.close' can only be called from the thread that started the viewer.")

            self.on_close()

    def save_video(self):
        """Close the recording, and move its video to a file that a dialog asks for, or delete it if none is given."""
        self._video_recorder.close()
        if not self._video_recorder.has_file:
            return
        filename = self._get_save_filename(["mp4"])
        if filename is None:
            os.remove(self._video_recorder.filename)
        else:
            shutil.move(self._video_recorder.filename, filename)

    def toggle_recording(self) -> bool:
        """Start or stop recording the window to a video file, and tell whether it is recording.

        Starting opens a video writer and marks the title of the window. Stopping closes it and asks for the file to
        save the video to (see 'save_video').
        """
        if self._is_recording:
            # This runs in event handlers, where a failed encoding must not escape the event loop and leave recording on
            try:
                self.save_video()
            except gs.GenesisException as e:
                gs.logger.warning(f"{e}")
            self.set_caption(self._window_title)
        else:
            self._video_recorder = VideoEncoder(
                os.path.join(get_cache_dir(), "tmp_video.mp4"), fps=self._refresh_rate, is_threaded=True
            )
            # Sentinel so the first frame is always captured regardless of the current step counter.
            self._last_recorded_t = -1
            self.set_caption(f"{self._window_title} (RECORDING)")
        self._is_recording = not self._is_recording
        return self._is_recording

    def on_close(self):
        """Exit the event loop when the window is closed."""
        # Always consider the viewer initialized at this point to avoid being stuck if starting fails
        if not self._initialized_event.is_set():
            self._initialized_event.set()

        # Do not consider the viewer as active anymore
        self._is_active = False

        for plugin in self.plugins:
            plugin.on_close()

        # Remove the camera of the window and restore the main camera of the scene
        if self._camera_node is not None and self.scene.has_node(self._camera_node):
            self.scene.remove_node(self._camera_node)
        self._camera_node = None
        if self._prior_main_camera_node is not None:
            self.scene.main_camera_node = self._prior_main_camera_node
        self._prior_main_camera_node = None

        # Delete renderer, along with any renderer retired by a rebind() that never drew again
        self._flush_retired_renderers()
        if self._renderer is not None:
            try:
                self._renderer.delete()
                self.gs_context.jit.delete()
            except (OpenGL.error.GLError, OpenGL.error.NullFunctionError):
                pass
        self._renderer = None

        # Delete video recorder
        if self._is_recording:
            try:
                self._video_recorder.close()
            except gs.GenesisException as e:
                gs.logger.warning(f"{e}")
            if self._video_recorder.has_file:
                os.remove(self._video_recorder.filename)

        # Force clean-up of OpenGL context data
        try:
            OpenGL.contextdata.cleanupContext()
            super().close()
        except Exception:
            pass
        try:
            super().on_close()
        except Exception:
            pass
        try:
            pyglet.app.exit()
        except Exception:
            pass
        try:
            pyglet.app.platform_event_loop.stop()
        except Exception:
            pass

        self._offscreen_semaphore.release()

    def close_offscreen(self, render_target):
        """Free a renderer of a camera, in the thread owning the OpenGL context.

        Parameters
        ----------
        render_target : Renderer
            The renderer of the camera.
        """
        if not self.is_active:
            return

        self._offscreen_pending_close = render_target
        if self._run_in_thread:
            # Send offscreen request
            self._offscreen_event.set()
            # Wait for offscreen
            self._offscreen_semaphore.acquire()
        else:
            # Force close renderer synchronously
            self._event_loop_step_offscreen()

    def render_offscreen(
        self,
        camera_node,
        render_target,
        rgb=True,
        depth=False,
        seg=False,
        normal=False,
        split_envs=False,
        env_idx=None,
        debug_view=True,
    ):
        """Render the images of a camera, in the thread owning the OpenGL context.

        Parameters
        ----------
        camera_node : Node
            The node of the camera.
        render_target : Renderer
            The renderer of the camera.
        rgb : bool, optional
            Whether to render the color image. Defaults to True.
        depth : bool, optional
            Whether to render the depth image. Defaults to False.
        seg : bool, optional
            Whether to render the segmentation image, which excludes the color image. Defaults to False.
        normal : bool, optional
            Whether to render the image of the normals of the surfaces. Defaults to False.
        split_envs : bool, optional
            Whether the environments are rendered one by one. Defaults to False.
        env_idx : int, optional
            The environment to render, all of them when None. Defaults to None.
        debug_view : bool, optional
            Whether to draw the markers and the debug representation of the entities. Defaults to True.

        Returns
        -------
        images : tuple
            The images asked for (see 'Renderer.render_images').
        """
        if not self.is_active:
            # A viewer in its own thread stores what ended it rather than raising it where nobody waits, so the call
            # that finds the viewer gone is where that exception surfaces.
            gs.raise_exception_from("Viewer already closed.", self._exception)

        if rgb and seg:
            gs.raise_exception("RGB and segmentation map cannot be rendered in the same forward pass.")
        self._offscreen_pending_render = _OffscreenRequest(
            camera_node, render_target, rgb, depth, seg, normal, split_envs, env_idx, debug_view
        )
        if self._run_in_thread:
            # Send offscreen request
            self._offscreen_event.set()
            # Wait for offscreen
            self._offscreen_semaphore.acquire()
        else:
            # Force offscreen rendering synchronously
            self._event_loop_step_offscreen()
        return self._offscreen_result

    def wait_until_initialized(self):
        """Wait until the window is open, or failed to open."""
        self._initialized_event.wait()

    def _event_loop_step_offscreen(self):
        if self._offscreen_pending_render is None and self._offscreen_pending_close is None:
            return

        with self.render_lock if self._run_in_thread else nullcontext():
            # Make OpenGL context current
            self.switch_to()

            self._flush_retired_renderers()

            if self._offscreen_pending_close is not None:
                # Extract request right away
                target = self._offscreen_pending_close
                self._offscreen_pending_close = None

                # Delete renderer.
                # Note that it must be done here, because calling this method involve OpenGL routines that cannot cross
                # thread boundaries, otherwise it will cause segmentation fault.
                target.delete()

            if self._offscreen_pending_render is not None:
                # Extract request right away
                request = self._offscreen_pending_render
                self._offscreen_pending_render = None
                target = request.render_target

                # Update context, just in case is not already done before
                self.gs_context.update()

                if target is self._renderer:
                    # The interactive window's own renderer tracks the OS window content area, which the OS may clamp
                    # below the requested resolution (e.g. a viewport larger than a macOS runner can allocate). Force
                    # it to the requested viewport size so the offscreen result matches what the caller asked for, then
                    # restore the live window size so on-screen drawing keeps following the window.
                    saved_viewport = (target.viewport_width, target.viewport_height)
                    target.viewport_width, target.viewport_height = self._offscreen_viewport_size
                    try:
                        self.clear()
                        retval = self._render(target, request)
                    finally:
                        target.viewport_width, target.viewport_height = saved_viewport
                else:
                    # A per-camera offscreen FBO is already sized to that camera's own resolution and is independent of
                    # the window, so its viewport is authoritative and must not be overridden with the window size.
                    self.clear()
                    retval = self._render(target, request)
                self._offscreen_result = retval if retval else (None, None)

            if self._run_in_thread:
                self._offscreen_semaphore.release()

    def _flush_retired_renderers(self):
        """Delete renderers retired by rebind(), along with the meshes and textures of the context each was bound to.
        Must run with the GL context current, before the replacement renderer draws (see rebind)."""
        while self._retired_renderers:
            renderer = self._retired_renderers.pop()
            try:
                renderer.delete()
                renderer.jit.delete()
            except (OpenGL.error.GLError, OpenGL.error.NullFunctionError):
                pass

    def on_draw(self):
        """Redraw the scene into the viewing window."""
        if self._renderer is None:
            return

        with self.render_lock:
            # Make OpenGL context current
            self.switch_to()

            self._flush_retired_renderers()

            # Render the scene
            self.clear()
            self._render(self._renderer)

            # Capture the recording frame right after the scene render, before any on-screen overlay (captions, help
            # text, and the plugins' ImGui panel / gizmo) is drawn, so the video shows only the rendered scene.
            record_frame = self._read_record_frame() if self._is_recording else None

        # The frame is encoded once the lock is released, so that a slow encoder does not hold up the simulation.
        if record_frame is not None:
            self._video_recorder.write(record_frame)

        # Render help text
        self._render_help_text()

        # Drive plugins only once the viewer is committed and initialized. start() renders the scene to probe GL
        # configs before that point, and a rejected config must not leave a plugin's GL/ImGui state dangling for the
        # next attempt (e.g. an ImGui frame opened but never ended). _initialized_event is set at the end of start().
        if self._initialized_event.is_set():
            # The plugins read the scene, so a threaded render loop draws them only while the scene is not updated
            with self.render_lock:
                for plugin in self.plugins:
                    plugin.on_draw()

    def on_resize(self, width: int, height: int) -> EVENT_HANDLE_STATE:
        """Resize the camera and trackball when the window is resized."""
        if self._renderer is None:
            return

        # Every frame of a video has the size of its first one, so a resize ends the recording
        if self._is_recording and (width, height) != self._viewport_size:
            gs.logger.warning("Viewer window resized while recording. Stopping the recording.")
            self.toggle_recording()

        self._renderer._delete_shadow_framebuffer()
        self._renderer._delete_floor_framebuffer()

        self._viewport_size = (width, height)
        self._trackball.resize(self._viewport_size)
        self._renderer.viewport_width = width
        self._renderer.viewport_height = height
        self.on_draw()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        """Record an initial mouse press."""
        # Stop animating while using the mouse
        self._is_mouse_pressed = True

        mode = TrackballMode.ROTATE
        if button == pyglet.window.mouse.LEFT:
            if modifiers & pyglet.window.key.MOD_CTRL:
                mode = TrackballMode.ZOOM
            elif modifiers & (pyglet.window.key.MOD_ALT | pyglet.window.key.MOD_SHIFT):
                mode = TrackballMode.PAN
        elif button == pyglet.window.mouse.MIDDLE:
            mode = TrackballMode.PAN
        elif button == pyglet.window.mouse.RIGHT:
            mode = TrackballMode.ZOOM
        self._trackball.set_mode(mode)

        self._trackball.down(np.array([x, y]))

        return EVENT_HANDLED

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        """The mouse was moved with one or more buttons held down."""
        result = self._trackball.drag(np.array([x, y]))
        return result

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        """Record a mouse release."""
        self._is_mouse_pressed = False
        return EVENT_HANDLED

    def on_mouse_scroll(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        """Record a mouse scroll."""
        if self.is_perspective:
            self._trackball.scroll(dy)
        else:
            spfc = 0.95
            dy_f = float(dy)
            if abs(dy_f) < 1e-8:
                return EVENT_HANDLED
            sf = float(spfc**dy_f)

            c = self._camera_node.camera
            xmag = max(c.xmag * sf, 1e-8)
            ymag = max(c.ymag * sf, 1e-8 * c.ymag / c.xmag)
            c.xmag = xmag
            c.ymag = ymag

        return EVENT_HANDLED

    def _call_keybind_callback(self, symbol: int, modifiers: int, action: KeyAction) -> None:
        """Call registered keybind callbacks for the given key event."""
        keybind: Keybind = self._keybindings.get(symbol, modifiers, action)
        if keybind is not None and keybind.callback is not None:
            keybind.callback(*keybind.args, **keybind.kwargs)

    def on_key_press(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        """Record a key press."""
        if (symbol, modifiers) not in self._held_keys:
            self._call_keybind_callback(symbol, modifiers, KeyAction.PRESS)

        self._held_keys[(symbol, modifiers)] = True

    def on_key_release(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        """Record a key release."""
        self._held_keys.pop((symbol, modifiers), None)

        self._call_keybind_callback(symbol, modifiers, KeyAction.RELEASE)

    def on_deactivate(self) -> EVENT_HANDLE_STATE:
        """Clear held keys when window loses focus."""
        self._held_keys.clear()

    @staticmethod
    def _time_event(dt, self):
        """The timer callback."""
        # Don't run old dead events after we've already closed
        if not self._is_active:
            return

        # The camera turns about the vertical axis once every six seconds
        if self.is_rotating and not self._is_mouse_pressed:
            self._trackball.rotate(np.pi / 3.0 / self._refresh_rate, np.array([0.0, 0.0, 1.0]))

        self.on_draw()

    def _reset_view(self):
        """Reset the view to a good initial state.

        The view is initially along the positive x-axis at a
        sufficient distance from the scene.
        """
        scale = DEFAULT_SCENE_SCALE
        centroid = self.scene.centroid if self._view_center is None else self._view_center

        self._camera_node.matrix = self._default_camera_pose
        oc = self._default_orth_cam
        oc.xmag, oc.ymag = self._orth_cam_reset_mags
        self._trackball = Trackball(self._default_camera_pose, self.viewport_size, scale, centroid)

    def _get_save_filename(self, file_exts):
        file_types = {
            "mp4": ("video files", "*.mp4"),
            "png": ("png files", "*.png"),
            "jpg": ("jpeg files", "*.jpg"),
            "gif": ("gif files", "*.gif"),
            "all": ("all files", "*"),
        }
        save_dir = os.getcwd() if self._save_directory is None else self._save_directory

        try:
            # Imported on demand because 'imgui-bundle' is an optional dependency, unavailable on some platforms.
            from imgui_bundle import portable_file_dialogs

            # The dialog is the native one of the platform and runs out of process, leaving pyglet sole owner of the
            # windowing system. Filters are a flat sequence alternating label and patterns.
            filters = [field for file_ext in file_exts for field in file_types[file_ext]]
            filename = portable_file_dialogs.save_file("Select file save location", save_dir, filters).result()
        except Exception as e:
            gs.logger.warning(f"Failed to open file save location dialog: {e}")
            return None

        if not filename:
            return None
        # The dialog hands the name back as typed, and a missing extension would leave the writer downstream with
        # no format to select, so fall back on the first one offered.
        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.{file_exts[0]}"
        return os.path.normpath(filename)

    def save_image(self):
        """Save the image of the window to a file that a dialog asks for."""
        # Postpone import of OpenCV at runtime to reduce hard system dependencies
        import cv2

        filename = self._get_save_filename(["png", "jpg", "gif", "all"])
        if filename is not None:
            self._save_directory = os.path.dirname(filename)
            data = self._renderer.jit.read_color_buf(*self._viewport_size)
            cv2.imwrite(filename, np.flip(data, axis=-1))

    def _read_record_frame(self):
        """Read the current frame to append to the video, or None if the simulation has not advanced since the last
        recorded frame (e.g. while paused) so the recording does not accumulate duplicate frozen frames."""
        t = self.gs_context.scene.sim.cur_step_global
        if t == self._last_recorded_t:
            return None
        self._last_recorded_t = t
        data = self._renderer.jit.read_color_buf(*self._viewport_size)
        if np.all(data == 0.0):
            return None
        return data

    def _render(self, renderer, request=None):
        """Render the scene into the window, or offscreen from the camera of a request.

        Returns
        -------
        images : tuple
            The images rendered offscreen (see 'Renderer.render_images').
        """
        self._camera_node.matrix = self._trackball.pose.copy()

        flags = (RenderFlags.NONE, RenderFlags.FLIP_WIREFRAME, RenderFlags.ALL_WIREFRAME, RenderFlags.ALL_SOLID)[
            self.wireframe_mode
        ]
        if self.show_vertex_normals:
            flags |= RenderFlags.VERTEX_NORMALS
        if self.show_face_normals:
            flags |= RenderFlags.FACE_NORMALS
        if request is None:
            return renderer.render_images(
                self.scene,
                flags,
                shadow=self.shadow,
                plane_reflection=self._plane_reflection,
                debug_view=self._debug_view,
            )
        return renderer.render_images(
            self.scene,
            flags | RenderFlags.OFFSCREEN,
            request.camera_node,
            rgb=request.rgb,
            depth=request.depth,
            seg_node_map=self._seg_node_map if request.seg else None,
            normal=request.normal,
            shadow=self.shadow,
            plane_reflection=self._plane_reflection,
            split_envs=request.split_envs,
            env_idx=request.env_idx,
            debug_view=request.debug_view,
        )

    def start(self, auto_refresh=True):
        """Open the window with the first OpenGL configuration that works, then run its event loop.

        Parameters
        ----------
        auto_refresh : bool, optional
            Whether to run the event loop until the window closes, rather than drawing a single frame. Defaults to True.
        """
        # The imports of the xlib modules below make 'pyglet' a local name, which must hold the module if they fail
        import pyglet
        import pyglet.app

        try:
            import pyglet.display.xlib
            import pyglet.window.xlib

            xlib_exceptions = (pyglet.window.xlib.XlibException, pyglet.display.xlib.NoSuchDisplayException)
        except ImportError:
            xlib_exceptions = ()

        # Pyglet's Win32EventLoop captures the thread that first instantiates it and refuses ``dispatch_events``
        # from any other thread. Mixing ``run_in_thread=True`` and ``run_in_thread=False`` viewers in the same
        # Python process (typical in unit tests) leaves a stale thread id behind. Recreate the platform event
        # loop here so its constructor rebinds the dispatch thread to whoever is about to call us.
        pyglet.app.platform_event_loop = pyglet.app.PlatformEventLoop()

        # Try multiple configs starting with target OpenGL version and multisampling enabled, then removing these
        # options if not supported.
        confs = [
            pyglet.gl.Config(
                depth_size=24,
                alpha_size=8,  # This parameter is essential to ensure proper pixel matching across platforms
                double_buffer=True,  # Double buffering to avoid flickering
                major_version=TARGET_OPEN_GL_MAJOR,
                minor_version=TARGET_OPEN_GL_MINOR,
            ),
            pyglet.gl.Config(
                depth_size=24,
                alpha_size=8,
                double_buffer=True,
                major_version=MIN_OPEN_GL_MAJOR,
                minor_version=MIN_OPEN_GL_MINOR,
            ),
        ]
        if "PYTEST_VERSION" not in os.environ:
            # MSAA must be disabled in headless mode for consistency across all platform because it behaves differently
            # depending on the rendering driver and there is no reliable way to control it. Although MSAAx2 is supported
            # by all drivers (incl. CPU-based), CPU-based Apple Cocoa using bilinear interpolation for rescaling instead
            # of nearest neighbors, and there is no way to tweak this behavior.
            confs = [
                pyglet.gl.Config(
                    sample_buffers=1,  # Enable multi-sampling (MSAA)
                    samples=2,
                    depth_size=24,
                    double_buffer=True,
                    major_version=TARGET_OPEN_GL_MAJOR,
                    minor_version=TARGET_OPEN_GL_MINOR,
                ),
                confs[0],
                pyglet.gl.Config(
                    sample_buffers=1,
                    samples=2,
                    depth_size=24,
                    double_buffer=True,
                    major_version=MIN_OPEN_GL_MAJOR,
                    minor_version=MIN_OPEN_GL_MINOR,
                ),
                confs[1],
            ]
        while confs:
            conf = confs.pop(0)

            # Close any existing context and window
            try:
                OpenGL.contextdata.cleanupContext()
                self.set_visible(False)
            except Exception:
                pass
            try:
                super().close()
            except Exception:
                pass
            # pyglet's headless backend leaves self._egl_surface set after close(), so re-__init__-ing this window for
            # the next GL config skips HeadlessWindow._create's context.attach() and the new context crashes in
            # switch_to() with no surface. Reset it (like the platform_event_loop reset above) so each retry attaches
            # its own; harmless on other backends, where the attribute is unused. When headless, destroy the orphaned
            # pbuffer first (close() does not, and pyglet created it) to avoid leaking one per retry.
            if pyglet.options.get("headless") and self._egl_surface is not None:
                from pyglet.libs.egl import egl

                egl.eglDestroySurface(self._egl_display_connection, self._egl_surface)
            self._egl_surface = None

            try:
                # Keep the window invisible for now. It will be displayed only if everything is working fine.
                # This approach avoids "flickering" when creating and closing an invalid context. Besides, it avoids
                # "frozen" graphical window during compilation that would be interpreted as as bug by the end-user.
                try:
                    # Resolving the screen rather than letting pyglet do it keeps the viewer available while every
                    # display is asleep, which pyglet reports as no screen at all. See 'get_default_screen'. It
                    # belongs inside this block, whose failures are reported to the thread waiting on startup.
                    screen = get_default_screen()

                    super().__init__(
                        config=conf,
                        visible=False,
                        resizable=True,
                        width=self._viewport_size[0],
                        height=self._viewport_size[1],
                        screen=screen,
                        # Enable vsync only when the viewer owns a render thread. In main-thread mode (e.g. macOS),
                        # a vsync-locked flip() would block the simulation loop for up to a display frame on every
                        # redraw, periodically overrunning the step budget and stuttering; redraws are already
                        # capped by the refresh_rate gate, so vsync is unnecessary there.
                        vsync=bool(self._run_in_thread),
                    )
                except xlib_exceptions:
                    # Trying again without UTF8 support as a fallback.
                    # See: https://github.com/pyglet/pyglet/issues/1024
                    if pyglet.window.xlib._have_utf8:
                        pyglet.window.xlib._have_utf8 = False
                        confs.insert(0, conf)
                    raise

                # Determine if software emulation is being used
                glinfo = self.context.get_info()
                renderer = glinfo.get_renderer()
                self._is_software = any(name in renderer for name in SOFTWARE_RENDERERS)

                # Run the entire rendering pipeline first without window, to make sure that all kernels are compiled
                self.refresh()

                # At this point, we are all set to display the graphical window, unless asked to keep the screen
                # free of any. The request is read here rather than at import, this module being imported before
                # the caller gets a chance to make it. pyglet's own option only covers its EGL-backed backend.
                if not (pyglet.options["headless"] or os.environ.get("GS_HEADLESS")):
                    self.set_visible(True)

                # Run the entire rendering pipeline once again, as a final validation that everything is fine
                self.refresh()

                break
            except Exception as e:
                # Every failure moves on to the next configuration, and the last one reaches the thread waiting for the
                # window, which would wait forever otherwise
                if not confs:
                    # It is essential to set the exception before closing the viewer, otherwise the main thread preempt
                    # execution of this thread and wrongly report unknown exception.
                    if self._run_in_thread:
                        self._exception = e

                    # The viewer closes, which leaves no OpenGL context or window dangling
                    try:
                        self.on_close()
                    except Exception:
                        pass

                    if self._run_in_thread:
                        # Reporting the exception for the main thread to raise it
                        return
                    else:
                        # Raise the exception right away
                        raise RuntimeError("Unable to initialize an OpenGL 3+ context.") from e

        if self._run_in_thread:
            pyglet.clock.schedule_interval(Viewer._time_event, 1.0 / self._refresh_rate, self)
        else:
            # Run as fast as possible if not running in thread
            pyglet.clock.schedule(Viewer._time_event, self)

        # Update window title
        self.set_caption(self._window_title)

        # Bringing the window to the front pulls the whole application forward on MacOS, which puts it on screen even
        # though it was never mapped, so it is skipped whenever the screen must be left alone.
        if not (pyglet.options["headless"] or os.environ.get("GS_HEADLESS")):
            self.activate()

        # The viewer can be considered as fully initialized at this point
        if not self._initialized_event.is_set():
            self._initialized_event.set()

        gs.logger.debug(f"Using interactive viewer OpenGL device: {renderer}")
        if self._is_software:
            gs.logger.info("Software rendering context detected. Beware rendering will be slow.")

        if auto_refresh:
            is_invalid = False
            try:
                while self._is_active:
                    try:
                        self.refresh()
                        is_invalid = False
                    except AttributeError:
                        # The graphical window has been closed manually
                        pass
                    except pyglet.gl.GLException as e:
                        # Refresh may fail in rare occurrences due to what looks like a race condition.
                        # Trying once in such a case is usually sufficent to succeed without risking deadlock.
                        if is_invalid or (f"(0x{pyglet.gl.GL_INVALID_OPERATION})" not in str(e)):
                            raise
                        is_invalid = True
            except Exception as e:
                if self._exception is None:
                    self._exception = e
            self.on_close()
        else:
            self.refresh()

    def run(self):
        """Run the event loop of a window started without one, until the window closes."""
        if self._run_in_thread:
            gs.raise_exception("'Viewer.run' cannot be called manually if the viewer is already running in thread.")
        elif threading.main_thread() != threading.current_thread():
            gs.raise_exception("'Viewer.run' can only be called manually from main thread on MacOS.")

        while self._is_active:
            try:
                self.refresh()
            except AttributeError:
                # The graphical window has been closed manually
                pass
        self.on_close()

    def refresh(self):
        """Process the events of the window and draw a frame, in the thread that started the viewer."""
        viewer_thread = self._thread or threading.main_thread()
        if viewer_thread != threading.current_thread():
            gs.raise_exception("'Viewer.refresh' can only be called from the thread that started the viewer.")

        # Dispatching the window events can run a GUI callback that calls a setter, which asks for a refresh in turn.
        # That nested refresh is skipped, since it would start a new GUI frame inside the one being drawn.
        if self._is_refreshing:
            return
        self._is_refreshing = True
        try:
            if self._run_in_thread:
                time_next_frame = time.time() + 1.0 / self._refresh_rate
                while self._offscreen_event.wait(time_next_frame - time.time()):
                    self._event_loop_step_offscreen()
                    self._offscreen_event.clear()

            self.switch_to()

            # Dispatch input events before drawing, so the frame (and the ImGui overlay / gizmo it renders) reacts to
            # the current mouse and keyboard state. Drawing first and dispatching afterwards leaves interactive controls
            # one frame behind the cursor, which makes dragging the gizmo feel stuttery.
            if sys.platform == "win32":
                # even changing `platform_event_loop.step(0.0)` to 0.001 causes the viewer to hang on Windows
                # this is a workaround on Windows. not sure if it's correct
                time.sleep(0.001)
            else:
                pyglet.app.platform_event_loop.step(0.0)

            self.dispatch_pending_events()
            if self._is_active:
                self.dispatch_events()

            pyglet.clock.tick()

            if self._is_active:
                self.flip()
        finally:
            self._is_refreshing = False

    def update_on_sim_step(self):
        """Call the bindings of the keys held down and the plugins, once per step of the simulation."""
        for symbol, modifiers in list(self._held_keys.keys()):
            self._call_keybind_callback(symbol, modifiers, KeyAction.HOLD)
        for plugin in self.plugins:
            plugin.update_on_sim_step()

    def _update_instr_texts(self):
        """Update the instruction text based on current keybindings."""
        if not self._enable_help_text:
            return

        self._key_instr_texts = self._instr_texts[1] + [
            f"{'[' + str(kb.key):>{7}}]: " + kb.name.replace("_", " ")
            for kb in self._keybindings.keybinds
            if kb.name != HELP_TEXT_KEYBIND_NAME
        ]

    def _toggle_instructions(self):
        """Toggle the display of keyboard instructions."""
        assert self._enable_help_text
        self._collapse_instructions = not self._collapse_instructions

    def set_message_text(self, text: str):
        """Show a message in the corner of the window for a short while, when the window shows help text."""
        if not self._enable_help_text:
            return
        self._message_text = text
        self._message_opac = 1.0 + self._ticks_till_fade

    def _render_help_text(self):
        """Render help text and messages on the viewer."""
        if not self._enable_help_text:
            return

        # Render temporary message
        if self._message_text is not None:
            self._renderer.render_text(
                [self._message_text],
                self._viewport_size[0] - TEXT_PADDING,
                TEXT_PADDING,
                font_name="OpenSans-Regular",
                font_pt=FONT_SIZE,
                color=np.array([0.1, 0.7, 0.2, np.clip(self._message_opac, 0.0, 1.0)]),
                align=TextAlign.BOTTOM_RIGHT,
            )

            if self._message_opac > 1.0:
                self._message_opac -= 1.0
            else:
                self._message_opac *= 0.90

            if self._message_opac < 0.05:
                self._message_opac = 1.0 + self._ticks_till_fade
                self._message_text = None

        # Render keyboard instructions
        self._renderer.render_text(
            self._instr_texts[0] if self._collapse_instructions else self._key_instr_texts,
            TEXT_PADDING,
            self._viewport_size[1] - TEXT_PADDING,
            font_name="UbuntuMono-Regular",
            font_pt=FONT_SIZE,
            color=self._font_color,
        )


__all__ = ["Viewer"]
