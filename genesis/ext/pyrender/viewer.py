"""A pyglet-based interactive 3D scene viewer."""

import copy
import os
import shutil
import sys
import threading
import time
from contextlib import nullcontext
from threading import Event, RLock, Semaphore, Thread
from typing import TYPE_CHECKING, Optional

import numpy as np

import OpenGL
from OpenGL.GL import *
import pyglet

import genesis as gs
from genesis.utils.misc import get_default_screen
from genesis.vis.keybindings import Key, KeyAction, Keybind, Keybindings, KeyMod
from genesis.vis.viewer_plugins import EVENT_HANDLE_STATE, EVENT_HANDLED, ViewerPlugin

from .camera import IntrinsicsCamera, OrthographicCamera, PerspectiveCamera
from .constants import (
    DEFAULT_SCENE_SCALE,
    DEFAULT_Z_FAR,
    DEFAULT_Z_NEAR,
    FONT_COLOR_DARKMODE,
    FONT_COLOR_LIGHTMODE,
    FONT_SIZE,
    MIN_OPEN_GL_MAJOR,
    MIN_OPEN_GL_MINOR,
    TARGET_OPEN_GL_MAJOR,
    TARGET_OPEN_GL_MINOR,
    TEXT_PADDING,
    RenderFlags,
    TextAlign,
)
from .light import DirectionalLight
from .node import Node
from .renderer import Renderer
from .trackball import Trackball

if TYPE_CHECKING:
    from genesis.vis.rasterizer_context import RasterizerContext

HELP_TEXT_KEY = Key.I
HELP_TEXT_KEYBIND_NAME = "toggle_instructions"


pyglet.options["shadow_window"] = False
if pyglet.options.get("dpi_scaling") != "real":
    pyglet.options["dpi_scaling"] = "real"


class Viewer(pyglet.window.Window):
    """An interactive viewer for 3D scenes.

    The viewer's camera is separate from the scene's, but will take on
    the parameters of the scene's main view camera and start in the same pose.
    If the scene does not have a camera, a suitable default will be provided.

    Parameters
    ----------
    scene : :class:`Scene`
        The scene to visualize.
    viewport_size : (2,) int
        The width and height of the initial viewing window.
    render_flags : dict
        A set of flags for rendering the scene. Described in the note below.
    viewer_flags : dict
        A set of flags for controlling the viewer's behavior.
        Described in the note below.
    **kwargs : dict
        Any keyword arguments left over will be interpreted as belonging to
        either the :attr:`.Viewer.render_flags` or :attr:`.Viewer.viewer_flags`
        dictionaries. Those flag sets will be updated appropriately.

    Note
    ----
    The basic commands for moving about the scene are given as follows:

    - **Rotating about the scene**: Hold the left mouse button and
      drag the cursor.
    - **Panning**:

      - Hold ALT, then hold the left mouse button and drag the cursor, or
      - Hold the middle mouse button and drag the cursor.

    - **Zooming**:

      - Scroll the mouse wheel, or
      - Hold the right mouse button and drag the cursor.

    Keyboard shortcuts are registered by ``DefaultControlsPlugin`` (see
    ``genesis/vis/viewer_plugins/plugins/default_controls.py``) and surfaced in the on-screen help overlay; press the
    help key to toggle it.

    Note
    ----
    The valid keys for ``render_flags`` are as follows:

    - ``flip_wireframe``: `bool`, If `True`, all objects will have their
      wireframe modes flipped from what their material indicates.
      Defaults to `False`.
    - ``all_wireframe``: `bool`, If `True`, all objects will be rendered
      in wireframe mode. Defaults to `False`.
    - ``all_solid``: `bool`, If `True`, all objects will be rendered in
      solid mode. Defaults to `False`.
    - ``shadows``: `bool`, If `True`, shadows will be rendered.
      Defaults to `False`.
    - ``vertex_normals``: `bool`, If `True`, vertex normals will be
      rendered as blue lines. Defaults to `False`.
    - ``face_normals``: `bool`, If `True`, face normals will be rendered as
      blue lines. Defaults to `False`.
    - ``cull_faces``: `bool`, If `True`, backfaces will be culled.
      Defaults to `True`.
    - ``point_size`` : float, The point size in pixels. Defaults to 1px.

    Note
    ----
    The valid keys for ``viewer_flags`` are as follows:

    - ``rotate``: `bool`, If `True`, the scene's camera will rotate
      about an axis. Defaults to `False`.
    - ``rotate_rate``: `float`, The rate of rotation in radians per second.
      Defaults to `PI / 3.0`.
    - ``rotate_axis``: `(3,) float`, The axis in world coordinates to rotate
      about. Defaults to ``[0,0,1]``.
    - ``view_center``: `(3,) float`, The position to rotate the scene about.
      Defaults to the scene's centroid.
    - ``use_raymond_lighting``: `bool`, If `True`, an additional set of three
      directional lights that move with the camera will be added to the scene.
      Defaults to `False`.
    - ``use_direct_lighting``: `bool`, If `True`, an additional directional
      light that moves with the camera and points out of it will be added to
      the scene. Defaults to `False`.
    - ``lighting_intensity``: `float`, The overall intensity of the
      viewer's additional lights (when they're in use). Defaults to 3.0.
    - ``use_perspective_cam``: `bool`, If `True`, a perspective camera will
      be used. Otherwise, an orthographic camera is used. Defaults to `True`.
    - ``save_directory``: `str`, A directory to open the file dialogs in.
      Defaults to `None`.
    - ``window_title``: `str`, A title for the viewer's application window.
      Defaults to `"Scene Viewer"`.
    - ``refresh_rate``: `float`, A refresh rate for rendering, in Hertz.
      Defaults to `30.0`.
    - ``fullscreen``: `bool`, Whether to make viewer fullscreen.
      Defaults to `False`.
    - ``caption``: `list of dict`, Text caption(s) to display on the viewer.
      Defaults to `None`.

    Note
    ----
    Animation can be accomplished by running the viewer with ``run_in_thread``
    enabled. Then, just run a loop in your main thread, updating the scene as
    needed. Before updating the scene, be sure to acquire the
    :attr:`.Viewer.render_lock`, and release it  when your update is done.
    """

    def __init__(
        self,
        context: "RasterizerContext",
        viewport_size=None,
        render_flags=None,
        viewer_flags=None,
        run_in_thread=False,
        auto_start=True,
        shadow=False,
        plane_reflection=False,
        plugins=None,
        enable_help_text=True,
        **kwargs,
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

        self._default_render_flags = {
            "flip_wireframe": False,
            "all_wireframe": False,
            "all_solid": False,
            "shadows": shadow,
            "plane_reflection": plane_reflection,
            "vertex_normals": False,
            "face_normals": False,
            "cull_faces": True,
            "offscreen": False,
            "point_size": 1.0,
            "rgb": True,
            "seg": False,
            "depth": False,
        }
        self._default_viewer_flags = {
            "mouse_pressed": False,
            "rotate": False,
            "rotate_rate": np.pi / 3.0,
            "rotate_axis": np.array([0.0, 0.0, 1.0]),
            "view_center": None,
            "record": False,
            "use_raymond_lighting": False,
            "use_direct_lighting": False,
            "lighting_intensity": 3.0,
            "use_perspective_cam": True,
            "save_directory": None,
            "window_title": "Scene Viewer",
            "refresh_rate": 30.0,
            "fullscreen": False,
            "caption": None,
        }
        self._render_flags = self._default_render_flags.copy()
        self._viewer_flags = self._default_viewer_flags.copy()
        self._viewer_flags["rotate_axis"] = self._default_viewer_flags["rotate_axis"].copy()

        if render_flags is not None:
            self._render_flags.update(render_flags)
        if viewer_flags is not None:
            self._viewer_flags.update(viewer_flags)

        for key in kwargs:
            if key in self.render_flags:
                self._render_flags[key] = kwargs[key]
            elif key in self.viewer_flags:
                self._viewer_flags[key] = kwargs[key]

        self._keybindings: Keybindings = Keybindings()
        self._held_keys: dict[tuple[int, int], bool] = {}

        #######################################################################
        # Save internal settings
        #######################################################################

        # Set up raymond lights and direct lights
        self._raymond_lights = self._create_raymond_lights()
        self._direct_light = self._create_direct_light()

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
            self._ticks_till_fade = 2.0 / 3.0 * self.viewer_flags["refresh_rate"]
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
        self._renderer = Renderer(*self._viewport_size, context.jit, self.render_flags["point_size"])
        self._is_active = True

        # Starting the viewer would raise an exception if the OpenGL context is invalid for some reason. This exception
        # must be caught in order to implement some fallback mechanism. One may want to start the viewer from the main
        # thread while the running loop would be running on a background thread. However, this approach is not possible
        # because all access to the OpenGL context must be done from the thread that created it in the first place. As
        # a result, the logic for catching an invalid OpenGL context must be implemented at the thread-level.
        self.auto_start = auto_start
        if self._run_in_thread:
            self._initialized_event.clear()
            self._thread = Thread(target=self.start, daemon=True)
            self._thread.start()
            self._initialized_event.wait()
            if not self._is_active:
                if self._exception:
                    raise RuntimeError("Unable to initialize an OpenGL 3+ context.") from self._exception
                raise OpenGL.error.Error("Invalid OpenGL context (unknown exception).")
        else:
            if self.auto_start:
                self.start()

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
    def render_flags(self):
        """dict : Flags for controlling the renderer's behavior.

        - ``flip_wireframe``: `bool`, If `True`, all objects will have their
          wireframe modes flipped from what their material indicates.
          Defaults to `False`.
        - ``all_wireframe``: `bool`, If `True`, all objects will be rendered
          in wireframe mode. Defaults to `False`.
        - ``all_solid``: `bool`, If `True`, all objects will be rendered in
          solid mode. Defaults to `False`.
        - ``shadows``: `bool`, If `True`, shadows will be rendered.
          Defaults to `False`.
        - ``vertex_normals``: `bool`, If `True`, vertex normals will be
          rendered as blue lines. Defaults to `False`.
        - ``face_normals``: `bool`, If `True`, face normals will be rendered as
          blue lines. Defaults to `False`.
        - ``cull_faces``: `bool`, If `True`, backfaces will be culled.
          Defaults to `True`.
        - ``point_size`` : float, The point size in pixels. Defaults to 1px.

        """
        return self._render_flags

    @render_flags.setter
    def render_flags(self, value):
        self._render_flags = value

    @property
    def viewer_flags(self):
        """dict : Flags for controlling the viewer's behavior.

        The valid keys for ``viewer_flags`` are as follows:

        - ``rotate``: `bool`, If `True`, the scene's camera will rotate
          about an axis. Defaults to `False`.
        - ``rotate_rate``: `float`, The rate of rotation in radians per second.
          Defaults to `PI / 3.0`.
        - ``rotate_axis``: `(3,) float`, The axis in world coordinates to
          rotate about. Defaults to ``[0,0,1]``.
        - ``view_center``: `(3,) float`, The position to rotate the scene
          about. Defaults to the scene's centroid.
        - ``use_raymond_lighting``: `bool`, If `True`, an additional set of
          three directional lights that move with the camera will be added to
          the scene. Defaults to `False`.
        - ``use_direct_lighting``: `bool`, If `True`, an additional directional
          light that moves with the camera and points out of it will be
          added to the scene. Defaults to `False`.
        - ``lighting_intensity``: `float`, The overall intensity of the
          viewer's additional lights (when they're in use). Defaults to 3.0.
        - ``use_perspective_cam``: `bool`, If `True`, a perspective camera will
          be used. Otherwise, an orthographic camera is used. Defaults to
          `True`.
        - ``save_directory``: `str`, A directory to open the file dialogs in.
          Defaults to `None`.
        - ``window_title``: `str`, A title for the viewer's application window.
          Defaults to `"Scene Viewer"`.
        - ``refresh_rate``: `float`, A refresh rate for rendering, in Hertz.
          Defaults to `30.0`.
        - ``fullscreen``: `bool`, Whether to make viewer fullscreen.
          Defaults to `False`.
        - ``caption``: `list of dict`, Text caption(s) to display on
          the viewer. Defaults to `None`.

        """
        return self._viewer_flags

    @viewer_flags.setter
    def viewer_flags(self, value):
        self._viewer_flags = value

    def _setup_main_camera(self):
        # Extract main camera from the current scene and set up our mirrored copy. Re-runnable so a rebind
        # to a rebuilt scene graph re-creates the camera node on the new scene.
        self._camera_node = None
        self._prior_main_camera_node = None
        self._default_camera_pose = None
        self._default_persp_cam = None
        self._default_orth_cam = None
        self._trackball = None

        znear = None
        zfar = None
        if self.scene.main_camera_node is not None:
            n = self.scene.main_camera_node
            camera = copy.copy(n.camera)
            if isinstance(camera, (PerspectiveCamera, IntrinsicsCamera)):
                self._default_persp_cam = camera
                znear = camera.znear
                zfar = camera.zfar
            elif isinstance(camera, OrthographicCamera):
                self._default_orth_cam = camera
                znear = camera.znear
                zfar = camera.zfar
            self._default_camera_pose = self.scene.get_pose(self.scene.main_camera_node)
            self._prior_main_camera_node = n

        # Set defaults as needed
        if zfar is None:
            zfar = max(self.scene.scale * 10.0, DEFAULT_Z_FAR)
        if znear is None or znear < 1e-6:
            if self.scene.scale < 1e-6:
                znear = DEFAULT_Z_NEAR
            else:
                znear = min(self.scene.scale / 10.0, DEFAULT_Z_NEAR)

        if self._default_persp_cam is None:
            self._default_persp_cam = PerspectiveCamera(yfov=np.pi / 3.0, znear=znear, zfar=zfar)
        if self._default_orth_cam is None:
            xmag = ymag = self.scene.scale
            if self.scene.scale < 1e-6:
                xmag = ymag = 1.0
            self._default_orth_cam = OrthographicCamera(xmag=xmag, ymag=ymag, znear=znear, zfar=zfar)
        self._orth_cam_reset_mags = (self._default_orth_cam.xmag, self._default_orth_cam.ymag)
        if self._default_camera_pose is None:
            self._default_camera_pose = self._compute_initial_camera_pose()

        # Pick camera
        if self.viewer_flags["use_perspective_cam"]:
            camera = self._default_persp_cam
        else:
            camera = self._default_orth_cam

        self._camera_node = Node(matrix=self._default_camera_pose, camera=camera)
        self.scene.add_node(self._camera_node)
        self.scene.main_camera_node = self._camera_node
        self._reset_view()

    def rebind(self, context, plugins):
        """Re-point this live viewer/window at a rebuilt scene graph (InteractiveScene rebuild) instead of
        opening a new window. Reuses the OS window and GL context, refreshing the context-bound state:
        scene graph, segmentation map, renderer, camera node and plugins. Rough on purpose - assumes no
        concurrent render thread is touching the old scene during the rebuild."""
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
            self._renderer = Renderer(*self._viewport_size, context.jit, self.render_flags["point_size"])
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
        """
        Add a key handler to call a function when the given key is pressed.

        Parameters
        ----------
        keybinds : Keybind
            One or more Keybind objects to register.
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
        """Close the viewer.

        This function will wait for the actual close, so you immediately
        manipulate the scene afterwards.
        """
        if self._run_in_thread:
            self._is_active = False
            while self._thread.is_alive():
                time.sleep(1.0 / self.viewer_flags["refresh_rate"])
        else:
            viewer_thread = self._thread or threading.main_thread()
            if viewer_thread != threading.current_thread():
                raise RuntimeError("'Viewer.close' can only be called from the thread that started the viewer.")

            self.on_close()

    def save_video(self, filename=None):
        """Save the stored frames to a video file.

        To use this asynchronously, run the viewer with the ``record``
        flag and the ``run_in_thread`` flags set.
        Kill the viewer after your desired time with
        :meth:`.Viewer.close_external`, and then call :meth:`.Viewer.save_video`.

        Parameters
        ----------
        filename : str
            The file to save the video to. If not specified,
            a file dialog will be opened to ask the user where
            to save the video file.
        """
        self._video_recorder.close()
        if filename is None:
            filename = self._get_save_filename(["mp4"])
        if filename is None:
            os.remove(self._video_recorder.filename)
        else:
            shutil.move(self._video_recorder.filename, filename)

    def toggle_recording(self) -> bool:
        """Start or stop recording the on-screen viewer to a video file, returning the resulting record state.

        Starting opens a fresh video writer and marks the window title; stopping closes it and prompts (via
        save_video) for a destination. Both the 'R' keybind and the overlay record button drive this one path."""
        if self.viewer_flags["record"]:
            self.save_video()
            self.set_caption(self.viewer_flags["window_title"])
        else:
            # Importing moviepy is very slow and rarely needed, so defer it to the first recording.
            from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

            self._video_recorder = FFMPEG_VideoWriter(
                filename=os.path.join(gs.utils.misc.get_cache_dir(), "tmp_video.mp4"),
                fps=self.viewer_flags["refresh_rate"],
                size=self.viewport_size,
            )
            # Sentinel so the first frame is always captured regardless of the current step counter.
            self._last_recorded_t = -1
            self.set_caption("{} (RECORDING)".format(self.viewer_flags["window_title"]))
        self.viewer_flags["record"] = not self.viewer_flags["record"]
        return self.viewer_flags["record"]

    def on_close(self):
        """Exit the event loop when the window is closed."""
        # Always consider the viewer initialized at this point to avoid being stuck if starting fails
        if not self._initialized_event.is_set():
            self._initialized_event.set()

        # Do not consider the viewer as active anymore
        self._is_active = False

        for plugin in self.plugins:
            plugin.on_close()

        # Remove our camera and restore the prior one
        try:
            if self._camera_node is not None:
                self.scene.remove_node(self._camera_node)
            self._camera_node = None
        except Exception:
            pass
        if self._prior_main_camera_node is not None:
            self.scene.main_camera_node = self._prior_main_camera_node
        self._prior_main_camera_node = None

        # Delete any lighting nodes that we've attached
        if self.viewer_flags["use_raymond_lighting"]:
            for n in self._raymond_lights:
                if self.scene.has_node(n):
                    self.scene.remove_node(n)
        if self.viewer_flags["use_direct_lighting"]:
            if self.scene.has_node(self._direct_light):
                self.scene.remove_node(self._direct_light)

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
        if self.viewer_flags["record"]:
            self._video_recorder.close()
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
        if not self.is_active:
            return

        self._offscreen_pending_close = (render_target,)
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
        skip_markers=False,
        split_envs=False,
    ):
        if not self.is_active:
            # A viewer in its own thread stores what ended it rather than raising it where nobody waits, so the call
            # that finds the viewer gone is where that exception surfaces.
            gs.raise_exception_from("Viewer already closed.", self._exception)

        if rgb and seg:
            gs.raise_exception("RGB and segmentation map cannot be rendered in the same forward pass.")
        self.render_flags["rgb"] = rgb
        self.render_flags["seg"] = seg
        self.render_flags["depth"] = depth
        self._offscreen_pending_render = (camera_node, render_target, normal, skip_markers, split_envs)
        if self._run_in_thread:
            # Send offscreen request
            self._offscreen_event.set()
            # Wait for offscreen
            self._offscreen_semaphore.acquire()
        else:
            # Force offscreen rendering synchronously
            self._event_loop_step_offscreen()
        self.render_flags["rgb"] = True
        self.render_flags["seg"] = False
        self.render_flags["depth"] = False
        return self._offscreen_result

    def wait_until_initialized(self):
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
                (target,) = self._offscreen_pending_close
                self._offscreen_pending_close = None

                # Delete renderer.
                # Note that it must be done here, because calling this method involve OpenGL routines that cannot cross
                # thread boundaries, otherwise it will cause segmentation fault.
                target.delete()

            if self._offscreen_pending_render is not None:
                # Extract request right away
                camera, target, normal, skip_markers, split_envs = self._offscreen_pending_render
                self._offscreen_pending_render = None

                # Update context, just in case is not already done before
                self.gs_context.update()

                self._offscreen_results = []
                self.render_flags["offscreen"] = True
                self.render_flags["skip_markers"] = skip_markers
                self.render_flags["split_envs"] = split_envs
                if target is self._renderer:
                    # The interactive window's own renderer tracks the OS window content area, which the OS may clamp
                    # below the requested resolution (e.g. a viewport larger than a macOS runner can allocate). Force
                    # it to the requested viewport size so the offscreen result matches what the caller asked for, then
                    # restore the live window size so on-screen drawing keeps following the window.
                    saved_viewport = (target.viewport_width, target.viewport_height)
                    target.viewport_width, target.viewport_height = self._offscreen_viewport_size
                    try:
                        self.clear()
                        retval = self._render(camera, target, normal)
                    finally:
                        target.viewport_width, target.viewport_height = saved_viewport
                else:
                    # A per-camera offscreen FBO is already sized to that camera's own resolution and is independent of
                    # the window, so its viewport is authoritative and must not be overridden with the window size.
                    self.clear()
                    retval = self._render(camera, target, normal)
                self._offscreen_result = retval if retval else (None, None)
                self.render_flags["offscreen"] = False
                self.render_flags["skip_markers"] = False
                self.render_flags["split_envs"] = False

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

        with self.render_lock if self._run_in_thread or not self.auto_start else nullcontext():
            # Make OpenGL context current
            self.switch_to()

            self._flush_retired_renderers()

            # Render the scene
            self.clear()
            self._render()

        # Capture the recording frame right after the scene render, before any on-screen overlay (captions, help
        # text, and the plugins' ImGui panel / gizmo) is drawn, so the video shows only the rendered scene.
        if self.viewer_flags["record"]:
            self._record()

        if self.viewer_flags["caption"] is not None:
            for caption in self.viewer_flags["caption"]:
                xpos, ypos = self._location_to_x_y(caption["location"])
                self._renderer.render_text(
                    caption["text"],
                    xpos,
                    ypos,
                    font_name=caption["font_name"],
                    font_pt=caption["font_pt"],
                    color=caption["color"],
                    scale=caption["scale"],
                    align=caption["location"],
                )

        # Render help text
        self._render_help_text()

        # Drive plugins only once the viewer is committed and initialized. start() renders the scene to probe GL
        # configs before that point, and a rejected config must not leave a plugin's GL/ImGui state dangling for the
        # next attempt (e.g. an ImGui frame opened but never ended). _initialized_event is set at the end of start().
        if self._initialized_event.is_set():
            for plugin in self.plugins:
                plugin.on_draw()

    def on_resize(self, width: int, height: int) -> EVENT_HANDLE_STATE:
        """Resize the camera and trackball when the window is resized."""
        if self._renderer is None:
            return

        self._renderer._delete_shadow_framebuffer()
        self._renderer._delete_floor_framebuffer()

        self._viewport_size = (width, height)
        self._trackball.resize(self._viewport_size)
        self._renderer.viewport_width = width
        self._renderer.viewport_height = height
        self.on_draw()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        """The mouse was moved with no buttons held down."""
        pass

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        """Record an initial mouse press."""
        # Stop animating while using the mouse
        self.viewer_flags["mouse_pressed"] = True

        self._trackball.set_state(Trackball.STATE_ROTATE)
        if button == pyglet.window.mouse.LEFT:
            ctrl = modifiers & pyglet.window.key.MOD_CTRL
            shift = modifiers & pyglet.window.key.MOD_SHIFT
            alt = modifiers & pyglet.window.key.MOD_ALT
            if ctrl:
                self._trackball.set_state(Trackball.STATE_ZOOM)
            elif alt or shift:
                self._trackball.set_state(Trackball.STATE_PAN)
        elif button == pyglet.window.mouse.MIDDLE:
            self._trackball.set_state(Trackball.STATE_PAN)
        elif button == pyglet.window.mouse.RIGHT:
            self._trackball.set_state(Trackball.STATE_ZOOM)

        self._trackball.down(np.array([x, y]))

        return EVENT_HANDLED

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        """The mouse was moved with one or more buttons held down."""
        result = self._trackball.drag(np.array([x, y]))
        return result

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        """Record a mouse release."""
        self.viewer_flags["mouse_pressed"] = False
        return EVENT_HANDLED

    def on_mouse_scroll(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        """Record a mouse scroll."""
        if self.viewer_flags["use_perspective_cam"]:
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

        if self.viewer_flags["rotate"] and not self.viewer_flags["mouse_pressed"]:
            self._rotate()

        self.on_draw()

    def _reset_view(self):
        """Reset the view to a good initial state.

        The view is initially along the positive x-axis at a
        sufficient distance from the scene.
        """
        scale = DEFAULT_SCENE_SCALE
        centroid = self.scene.centroid

        if self.viewer_flags["view_center"] is not None:
            centroid = self.viewer_flags["view_center"]

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
        save_dir = self.viewer_flags["save_directory"]
        if save_dir is None:
            save_dir = os.getcwd()

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

    def _save_image(self):
        # Postpone import of OpenCV at runtime to reduce hard system dependencies
        import cv2

        filename = self._get_save_filename(["png", "jpg", "gif", "all"])
        if filename is not None:
            self.viewer_flags["save_directory"] = os.path.dirname(filename)
            data = self._renderer.jit.read_color_buf(*self._viewport_size, rgba=False)
            cv2.imwrite(filename, np.flip(data, axis=-1))

    def _record(self):
        """Append the current frame to the video, unless the simulation has not advanced since the last recorded
        frame (e.g. while paused) so the recording does not accumulate duplicate frozen frames."""
        t = self.gs_context.scene.sim.cur_step_global
        if t == self._last_recorded_t:
            return
        self._last_recorded_t = t
        data = self._renderer.jit.read_color_buf(*self._viewport_size, rgba=False)
        if not np.all(data == 0.0):
            self._video_recorder.write_frame(data)

    def _rotate(self):
        """Animate the scene by rotating the camera."""
        az = self.viewer_flags["rotate_rate"] / self.viewer_flags["refresh_rate"]
        self._trackball.rotate(az, self.viewer_flags["rotate_axis"])

    def _render(self, camera_node=None, renderer=None, normal=False):
        """Render the scene into the framebuffer and flip."""
        scene = self.scene
        self._camera_node.matrix = self._trackball.pose.copy()

        if renderer is None:
            renderer = self._renderer

        if camera_node is not None:
            saved_camera_node = self.scene.main_camera_node
            self.scene.main_camera_node = camera_node

        # Set lighting
        vli = self.viewer_flags["lighting_intensity"]
        if self.viewer_flags["use_raymond_lighting"]:
            for n in self._raymond_lights:
                n.light.intensity = vli / 3.0
                if not self.scene.has_node(n):
                    scene.add_node(n, parent_node=self._camera_node)
        else:
            self._direct_light.light.intensity = vli
            for n in self._raymond_lights:
                if self.scene.has_node(n):
                    self.scene.remove_node(n)

        if self.viewer_flags["use_direct_lighting"]:
            if not self.scene.has_node(self._direct_light):
                scene.add_node(self._direct_light, parent_node=self._camera_node)
        elif self.scene.has_node(self._direct_light):
            self.scene.remove_node(self._direct_light)

        flags = RenderFlags.NONE
        if self.render_flags["flip_wireframe"]:
            flags |= RenderFlags.FLIP_WIREFRAME
        elif self.render_flags["all_wireframe"]:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_flags["all_solid"]:
            flags |= RenderFlags.ALL_SOLID

        if self.render_flags["shadows"] and not self._is_software:
            flags |= RenderFlags.SHADOWS_ALL
        if self.render_flags["plane_reflection"] and not self._is_software:
            flags |= RenderFlags.REFLECTIVE_FLOOR
        if self.render_flags.get("split_envs", False):
            flags |= RenderFlags.ENV_SEPARATE
        if self.render_flags["vertex_normals"]:
            flags |= RenderFlags.VERTEX_NORMALS
        if self.render_flags["face_normals"]:
            flags |= RenderFlags.FACE_NORMALS
        if not self.render_flags["cull_faces"]:
            flags |= RenderFlags.SKIP_CULL_FACES

        if self.render_flags["offscreen"]:
            flags |= RenderFlags.OFFSCREEN
        if self.render_flags.get("skip_markers", False):
            flags |= RenderFlags.SKIP_MARKERS
        else:
            flags |= RenderFlags.MARKER_XRAY

        seg_node_map = None
        if self.render_flags["seg"]:
            flags |= RenderFlags.SEG
            seg_node_map = self._seg_node_map

        if self.render_flags["depth"]:
            flags |= RenderFlags.RET_DEPTH
            if not (self.render_flags["rgb"] or self.render_flags["seg"]):
                flags |= RenderFlags.DEPTH_ONLY

        first_pass_done = False
        if self.render_flags["rgb"] or self.render_flags["depth"] or self.render_flags["seg"]:
            retval = renderer.render(self.scene, flags, seg_node_map=seg_node_map)
            first_pass_done = True
        else:
            retval = ()

        if normal:
            old_cache = renderer._program_cache
            renderer._program_cache = renderer._normal_program_cache

            flags = RenderFlags.FLAT | RenderFlags.OFFSCREEN
            if self.render_flags.get("split_envs", False):
                flags |= RenderFlags.ENV_SEPARATE
            if self.render_flags.get("skip_markers", False):
                flags |= RenderFlags.SKIP_MARKERS
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            normal_arr, *_ = renderer.render(scene, flags, is_first_pass=not first_pass_done)
            retval = (*retval, normal_arr)

            renderer._program_cache = old_cache

        if camera_node is not None:
            self.scene.main_camera_node = saved_camera_node

        return retval

    def start(self, auto_refresh=True):
        import pyglet  # For some reason, this is necessary if 'pyglet.window.xlib' fails to import...
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
                except xlib_exceptions as e:
                    # Trying again without UTF8 support as a fallback.
                    # See: https://github.com/pyglet/pyglet/issues/1024
                    if pyglet.window.xlib._have_utf8:
                        pyglet.window.xlib._have_utf8 = False
                        confs.insert(0, conf)
                    raise

                # Determine if software emulation is being used
                glinfo = self.context.get_info()
                renderer = glinfo.get_renderer()
                self._is_software = any(e in renderer for e in ("llvmpipe", "Apple Software Renderer"))

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
            except (
                pyglet.window.NoSuchConfigException,
                pyglet.gl.ContextException,
                pyglet.gl.GLException,
                OpenGL.error.Error,
                AttributeError,
                ArgumentError,
                RuntimeError,
                TypeError,  # Race conditions wheno accessing OpenGL resources not binded to context yet
                Exception,  # Just in case, to avoid deadlock when running in thread
            ) as e:
                if not confs:
                    # It is essential to set the exception before closing the viewer, otherwise the main thread preempt
                    # execution of this thread and wrongly report unknown exception.
                    if self._run_in_thread:
                        self._exception = e

                    # Now the viewer can be safely cause to avoid leaving any global OpenGL context or window dangling
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
            pyglet.clock.schedule_interval(Viewer._time_event, 1.0 / self.viewer_flags["refresh_rate"], self)
        else:
            # Run as fast as possible if not running in thread
            pyglet.clock.schedule(Viewer._time_event, self)

        # Update window title
        self.set_caption(self.viewer_flags["window_title"])

        # Bringing the window to the front pulls the whole application forward on MacOS, which puts it on screen even
        # though it was never mapped, so it is skipped whenever the screen must be left alone.
        if not (pyglet.options["headless"] or os.environ.get("GS_HEADLESS")):
            self.activate()

        # The viewer can be considered as fully initialized at this point
        if not self._initialized_event.is_set():
            self._initialized_event.set()

        gs.logger.debug(f"Using interactive viewer OpenGL device: {renderer}")
        if self._is_software:
            gs.logger.info(
                "Software rendering context detected. Shadows and plane reflection not supported. Beware rendering "
                "will be extremely slow."
            )

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
        if self._run_in_thread:
            raise RuntimeError("'Viewer.run' cannot be called manually if the viewer is already running in thread.")
        elif threading.main_thread() != threading.current_thread():
            raise RuntimeError("'Viewer.run' can only be called manually from main thread on MacOS.")

        while self._is_active:
            try:
                self.refresh()
            except AttributeError:
                # The graphical window has been closed manually
                pass
        self.on_close()

    def refresh(self):
        viewer_thread = self._thread or threading.main_thread()
        if viewer_thread != threading.current_thread():
            raise RuntimeError("'Viewer.refresh' can only be called from the thread that started the viewer.")

        if self._run_in_thread:
            time_next_frame = time.time() + 1.0 / self.viewer_flags["refresh_rate"]
            while self._offscreen_event.wait(time_next_frame - time.time()):
                self._event_loop_step_offscreen()
                self._offscreen_event.clear()

        self.switch_to()

        # Dispatch input events before drawing, so the frame (and the ImGui overlay / gizmo it renders) reacts to the
        # current mouse and keyboard state. Drawing first and dispatching afterwards leaves interactive controls one
        # frame behind the cursor, which makes dragging the gizmo feel stuttery.
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

    def update_on_sim_step(self):
        # Call HOLD callbacks for all currently held keys
        for symbol, modifiers in list(self._held_keys.keys()):
            self._call_keybind_callback(symbol, modifiers, KeyAction.HOLD)
        for plugin in self.plugins:
            plugin.update_on_sim_step()

    def _compute_initial_camera_pose(self):
        centroid = self.scene.centroid
        if self.viewer_flags["view_center"] is not None:
            centroid = self.viewer_flags["view_center"]
        scale = self.scene.scale
        if scale == 0.0:
            scale = DEFAULT_SCENE_SCALE

        s2 = 1.0 / np.sqrt(2.0)
        cp = np.eye(4)
        cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
        hfov = np.pi / 6.0
        dist = scale / (2.0 * np.tan(hfov))
        cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid

        return cp

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(Node(light=DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix))

        return nodes

    def _create_direct_light(self):
        light = DirectionalLight(color=np.ones(3), intensity=1.0)
        n = Node(light=light, matrix=np.eye(4))
        return n

    def _location_to_x_y(self, location):
        if location == TextAlign.CENTER:
            return (self.viewport_size[0] / 2.0, self.viewport_size[1] / 2.0)
        elif location == TextAlign.CENTER_LEFT:
            return (TEXT_PADDING, self.viewport_size[1] / 2.0)
        elif location == TextAlign.CENTER_RIGHT:
            return (self.viewport_size[0] - TEXT_PADDING, self.viewport_size[1] / 2.0)
        elif location == TextAlign.BOTTOM_LEFT:
            return (TEXT_PADDING, TEXT_PADDING)
        elif location == TextAlign.BOTTOM_RIGHT:
            return (self.viewport_size[0] - TEXT_PADDING, TEXT_PADDING)
        elif location == TextAlign.BOTTOM_CENTER:
            return (self.viewport_size[0] / 2.0, TEXT_PADDING)
        elif location == TextAlign.TOP_LEFT:
            return (TEXT_PADDING, self.viewport_size[1] - TEXT_PADDING)
        elif location == TextAlign.TOP_RIGHT:
            return (self.viewport_size[0] - TEXT_PADDING, self.viewport_size[1] - TEXT_PADDING)
        elif location == TextAlign.TOP_CENTER:
            return (self.viewport_size[0] / 2.0, self.viewport_size[1] - TEXT_PADDING)

    def _update_instr_texts(self):
        """Update the instruction text based on current keybindings."""
        if not self._enable_help_text:
            return

        self._key_instr_texts = self._instr_texts[0] + [
            # f"{'[' + get_keycode_string(kb.key_code):>{7}}]: " + kb.name.replace("_", " ")
            f"{'[' + str(kb.key):>{7}}]: " + kb.name.replace("_", " ")
            for kb in self._keybindings.keybinds
            if kb.name != HELP_TEXT_KEYBIND_NAME
        ]

    def _toggle_instructions(self):
        """Toggle the display of keyboard instructions."""
        if not self._enable_help_text:
            raise RuntimeError("Instructions display is disabled.")
        self._collapse_instructions = not self._collapse_instructions

    def set_message_text(self, text: str):
        """Set a temporary message to display on the viewer."""
        self._message_text = text
        self._message_opac = 1.0 + self._ticks_till_fade

    def _render_help_text(self):
        """Render help text and messages on the viewer."""
        if not self._enable_help_text:
            return

        # Render temporary message
        if self._message_text is not None:
            self._renderer.render_text(
                self._message_text,
                self._viewport_size[0] - TEXT_PADDING,
                TEXT_PADDING,
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
        if self._collapse_instructions:
            self._renderer.render_texts(
                self._instr_texts[0],
                TEXT_PADDING,
                self._viewport_size[1] - TEXT_PADDING,
                font_pt=FONT_SIZE,
                color=self._font_color,
            )
        else:
            self._renderer.render_texts(
                self._key_instr_texts,
                TEXT_PADDING,
                self._viewport_size[1] - TEXT_PADDING,
                font_pt=FONT_SIZE,
                color=self._font_color,
            )


__all__ = ["Viewer"]
