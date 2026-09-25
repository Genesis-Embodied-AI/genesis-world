import importlib
import os
import sys
import threading
import time
from traceback import TracebackException
from typing import TYPE_CHECKING

import numpy as np
import OpenGL.error
import OpenGL.platform
import pyglet

import genesis as gs
import genesis.utils.geom as gu
from genesis.ext import pyrender
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
from genesis.repr_base import RBC
from genesis.utils.misc import redirect_libc_stderr, tensor_to_array
from genesis.utils.tools import Rate
from genesis.vis.keybindings import Key, KeyAction, Keybind, KeyMod
from genesis.vis.viewer_plugins import DefaultControlsPlugin

if TYPE_CHECKING:
    from genesis.options.vis import ViewerOptions
    from genesis.vis.viewer_plugins import ViewerPlugin


class ViewerLock:
    def __init__(self, pyrender_viewer):
        self._pyrender_viewer = pyrender_viewer

    def __enter__(self):
        self._pyrender_viewer.render_lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self._pyrender_viewer.render_lock.release()


class Viewer(RBC):
    def __init__(self, options: "ViewerOptions", context):
        self._is_built = False
        self._res = options.res
        self._run_in_thread = options.run_in_thread
        self._refresh_rate = options.refresh_rate
        self._realtime_factor = options.realtime_factor
        self._camera_init_pos = np.asarray(options.camera_pos, dtype=gs.np_float)
        self._camera_init_lookat = np.asarray(options.camera_lookat, dtype=gs.np_float)
        self._camera_up = np.asarray(options.camera_up, dtype=gs.np_float)
        self._camera_fov = options.camera_fov

        self._enable_help_text = options.enable_help_text
        self._plugins: list["ViewerPlugin"] = []
        if options.enable_default_keybinds:
            self._plugins.append(DefaultControlsPlugin())
        if options.enable_gui:
            self._plugins.append(ImGuiOverlayPlugin())

        # Validate viewer options
        if any(e.shape != (3,) for e in (self._camera_init_pos, self._camera_init_lookat, self._camera_up)):
            gs.raise_exception("ViewerOptions.camera_(pos|lookat|up) must be sequences of length 3.")

        self._pyrender_viewer = None
        self.context = context
        self.scene = None

        self._followed_entity = None
        self._follow_fixed_axis = None
        self._follow_smoothing = None
        self._follow_fix_orientation = None
        self._follow_lookat = None

        # Wall-clock time of the last on-screen redraw, used to cap single-thread redraws at refresh_rate (the
        # background-thread viewer paces its own redraws). Real-time pacing waits until the scene is built so the
        # physics dt is known.
        self._last_refresh_time = None
        self._realtime_pacer = None

    def build(self, scene):
        self.scene = scene

        # When the viewer is shown, hold the stepping loop to realtime_factor x wall-clock real time: each step advances
        # dt seconds of simulation, so it should take dt / realtime_factor seconds of wall time. Rate skips sleeping
        # when already behind, so a sim that cannot keep up simply runs as fast as it can.
        self._realtime_pacer = None if self._realtime_factor is None else Rate(self._realtime_factor / scene.sim.dt)

        # set viewer camera
        self.setup_camera()

        # Reuse an existing window across an InteractiveScene rebuild instead of opening a new one (which
        # would close and reopen the OS window). The preserved pyrender viewer is re-pointed at the rebuilt
        # scene graph in place.
        if self._pyrender_viewer is not None:
            self._pyrender_viewer.rebind(self.context, self._plugins)
            self.lock = ViewerLock(self._pyrender_viewer)
            self._is_built = True
            return

        # Try all candidate onscreen OpenGL "platforms" if none is specifically requested. OSMesa is ruled out for the
        # viewer: it binds PyOpenGL to its own private copy of Mesa for the whole process (see OSMesaPlatform), which
        # the window-system context of the viewer cannot be driven through.
        opengl_platform_orig = os.environ.get("PYOPENGL_PLATFORM")
        if opengl_platform_orig == "osmesa":
            gs.raise_exception(
                "PYOPENGL_PLATFORM='osmesa' only supports offscreen rendering. Unset it or disable the interactive "
                "viewer."
            )
        if opengl_platform_orig is None:
            if sys.platform == "win32":
                all_opengl_platforms = ("wgl",)  # same as "native"
            elif sys.platform == "linux":
                if pyglet.options.get("headless"):
                    # pyglet's headless windowing creates an EGL pbuffer context, so only the matching PyOpenGL EGL
                    # platform can share it; native/glx query a different context and fail with "no valid context",
                    # churning GL state on the way out.
                    all_opengl_platforms = ("egl",)
                else:
                    # "native" is platform-specific ("egl" or "glx")
                    all_opengl_platforms = ("native", "egl", "glx")
            else:
                all_opengl_platforms = ("native",)
        else:
            all_opengl_platforms = (opengl_platform_orig,)

        for i, platform in enumerate(all_opengl_platforms):
            # Force re-import OpenGL platform
            os.environ["PYOPENGL_PLATFORM"] = platform
            importlib.reload(OpenGL.platform)

            try:
                gs.logger.debug(f"Trying to create OpenGL Context for PYOPENGL_PLATFORM='{platform}'...")
                with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
                    self._pyrender_viewer = pyrender.Viewer(
                        context=self.context,
                        viewport_size=self._res,
                        run_in_thread=self._run_in_thread,
                        auto_start=False,
                        view_center=self._camera_init_lookat,
                        shadow=self.context.shadow,
                        plane_reflection=self.context.plane_reflection,
                        enable_help_text=self._enable_help_text,
                        plugins=self._plugins,
                        viewer_flags={
                            "window_title": f"Genesis {gs.__version__}",
                            "refresh_rate": self._refresh_rate,
                        },
                    )
                    if not self._run_in_thread:
                        self._pyrender_viewer.start(auto_refresh=False)
                    self._pyrender_viewer.wait_until_initialized()
                break
            except (OpenGL.error.Error, RuntimeError) as e:
                # Invalid OpenGL context. Trying another platform if any...
                traceback = TracebackException.from_exception(e)
                gs.logger.debug("".join(traceback.format()))

                # Clear broken OpenGL context if it went this far
                if self._pyrender_viewer is not None:
                    self._pyrender_viewer.close()
                    self._pyrender_viewer = None

                if i == len(all_opengl_platforms) - 1:
                    raise
            finally:
                # Restore original platform systematically
                del os.environ["PYOPENGL_PLATFORM"]
                if opengl_platform_orig is not None:
                    os.environ["PYOPENGL_PLATFORM"] = opengl_platform_orig

        self.lock = ViewerLock(self._pyrender_viewer)

        dt = self.scene.sim.dt
        # Real-time target step rate the pacer aims for; compare against the reported "Running at X FPS" to tell
        # whether the loop is real-time-bounded (X ~ target) or compute-bounded (X < target).
        target = "uncapped" if self._realtime_factor is None else f"{self._realtime_factor / dt:.1f} FPS"
        gs.logger.info(
            f"Viewer created. Resolution: ~<{self._res[0]}×{self._res[1]}>~, refresh_rate: ~<{self._refresh_rate}>~, "
            f"dt: ~<{dt}>~s, realtime_factor: ~<{self._realtime_factor}>~ (real-time target: ~<{target}>~)."
        )

        self._is_built = True

    def run(self):
        if self._pyrender_viewer is None:
            gs.raise_exception("Viewer must be built successfully before calling this method.")
        self._pyrender_viewer.run()

    def stop(self):
        if self._pyrender_viewer is not None and self._pyrender_viewer.is_active:
            self._pyrender_viewer.close()

    def is_alive(self):
        if self._pyrender_viewer is None:
            return False
        if self._pyrender_viewer._exception is not None:
            if self._pyrender_viewer.is_active:
                try:
                    self._pyrender_viewer.close()
                except Exception:
                    pass
            gs.raise_exception_from("Unexpected viewer error.", self._pyrender_viewer._exception)
        return self._pyrender_viewer.is_active

    def setup_camera(self):
        yfov = self._camera_fov / 180.0 * np.pi
        pose = gu.pos_lookat_up_to_T(self._camera_init_pos, self._camera_init_lookat, self._camera_up)
        self._camera_up = pose[:3, 1].copy()
        self._camera_node = self.context.add_node(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)

    def update(self, auto_refresh=None, force=False):
        if not self.is_alive():
            gs.raise_exception("Viewer closed.")

        if self._followed_entity is not None:
            self.update_following()

        self._pyrender_viewer.update_on_sim_step()

        # Sync the drawn scene with the simulation at most refresh_rate times per second, independently of how often
        # the simulation steps: a redraw is the only consumer of the sync, and stepping runs far more often than the
        # screen repaints. A camera rendering in between syncs the scene itself.
        now = time.perf_counter()
        is_refresh_due = self._last_refresh_time is None or now - self._last_refresh_time >= 1.0 / self._refresh_rate
        if force or is_refresh_due:
            self._last_refresh_time = now
            with self.lock:
                self.context.update(force)

                # Refresh viewer by default if and if this is possible
                if auto_refresh is None:
                    viewer_thread = self._pyrender_viewer._thread or threading.main_thread()
                    auto_refresh = viewer_thread == threading.current_thread()

                if auto_refresh and not self._pyrender_viewer.run_in_thread:
                    self._pyrender_viewer.refresh()

        # Pace the stepping loop to real time when a factor is set (no effect once the sim falls behind). Read the
        # pacer once: the realtime_factor setter may swap it from the viewer thread between the check and the call.
        realtime_pacer = self._realtime_pacer
        if realtime_pacer is not None:
            realtime_pacer.sleep()

    def close_offscreen(self, render_target):
        return self._pyrender_viewer.close_offscreen(render_target)

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
        return self._pyrender_viewer.render_offscreen(
            camera_node,
            render_target,
            rgb,
            depth,
            seg,
            normal,
            skip_markers=skip_markers,
            split_envs=split_envs,
        )

    def set_camera_pose(self, pose=None, pos=None, lookat=None):
        """
        Set viewer camera pose.

        Parameters
        ----------
        pose : array-like, shape (4, 4), optional
            Camera-to-world pose. If provided, `pos` and `lookat` will be ignored.
        pos : array-like, shape (3,), optional
            Camera position.
        lookat : array-like, shape (3,), optional
            Camera lookat point.
        """
        if pose is None:
            pos = self._camera_init_pos if pos is None else tensor_to_array(pos, dtype=gs.np_float)
            lookat = self._camera_init_lookat if lookat is None else tensor_to_array(lookat, dtype=gs.np_float)
            up = self._camera_up

            pose = gu.pos_lookat_up_to_T(pos, lookat, up)
            self._camera_up = pose[:3, 1].copy()
        else:
            pose = tensor_to_array(pose, dtype=gs.np_float)
            if pose.shape != (4, 4):
                gs.raise_exception("pose should be a 4x4 matrix.")

        self._pyrender_viewer._trackball.set_camera_pose(pose)

    def follow_entity(self, entity, fixed_axis=(None, None, None), smoothing=None, fix_orientation=False):
        """
        Set the viewer to follow a specified entity.
        Parameters
        ----------
        entity : genesis.Entity
            The entity to follow.
        fixed_axis : (float, float, float), optional
            The fixed axis for the viewer's movement. For each axis, if None, the viewer will move freely. If a float, the viewer will be fixed on at that value.
            For example, [None, None, None] will allow the viewer to move freely while following, [None, None, 0.5] will fix the viewer's z-axis at 0.5.
        smoothing : float, optional
            The smoothing factor in ]0,1[ for the viewer's movement. If None, no smoothing will be applied.
        fix_orientation : bool, optional
            If True, the viewer will maintain its orientation relative to the world. If False, the viewer will look at the base link of the entity.
        """
        self._followed_entity = entity
        self._follow_fixed_axis = fixed_axis
        self._follow_smoothing = smoothing
        self._follow_fix_orientation = fix_orientation
        self._follow_lookat = self._camera_init_lookat

    def update_following(self):
        """
        Update the viewer position to follow the specified entity.
        """
        entity_pos = tensor_to_array(self._followed_entity.get_pos(relative=False))
        if entity_pos.ndim > 1:  # check for multiple envs
            entity_pos = entity_pos[0]
        # numpy < 2.0 doesn't support the copy keyword argument in np.asarray()
        camera_transform = np.array(self._pyrender_viewer._trackball.pose, copy=True)
        camera_pos = np.array(self._pyrender_viewer._trackball.pose[:3, 3])

        if self._follow_smoothing is not None:
            # Smooth viewer movement with a low-pass filter
            camera_pos = self._follow_smoothing * camera_pos + (1 - self._follow_smoothing) * (
                entity_pos + self._camera_init_pos
            )
            self._follow_lookat = (
                self._follow_smoothing * self._follow_lookat + (1 - self._follow_smoothing) * entity_pos
            )
        else:
            camera_pos = entity_pos + self._camera_init_pos
            self._follow_lookat = entity_pos

        for i, fixed_axis in enumerate(self._follow_fixed_axis):
            # Fix the camera's position along the specified axis
            if fixed_axis is not None:
                camera_pos[i] = fixed_axis

        if self._follow_fix_orientation:
            # Keep the camera orientation fixed by overriding the lookat point
            camera_transform[:3, 3] = camera_pos
            self.set_camera_pose(pose=camera_transform)
        else:
            self.set_camera_pose(pos=camera_pos, lookat=self._follow_lookat)

    @gs.assert_built
    def register_keybinds(self, /, *keybinds: Keybind, overwrite: bool = False) -> None:
        """
        Register a callback function to be called when a key is pressed.

        Parameters
        ----------
        keybinds : Keybind
            One or more Keybind objects to register. See Keybind documentation for usage.
        """
        self._pyrender_viewer.register_keybinds(*keybinds, overwrite=overwrite)

    @gs.assert_built
    def remap_keybind(
        self,
        keybind_name: str,
        new_key: Key,
        new_key_mods: tuple[KeyMod] | None,
        new_key_action: KeyAction = KeyAction.PRESS,
    ) -> None:
        """
        Remap an existing keybind by name to a new key combination.

        Parameters
        ----------
        keybind_name : str
            The name of the keybind to remap.
        new_key : int
            The new key code from pyglet.
        new_key_mods : tuple[KeyMod] | None
            The new modifier keys pressed.
        new_key_action : KeyAction, optional
            The new type of key action. If not provided, the key action of the old keybind is used.
        """
        self._pyrender_viewer.remap_keybind(
            keybind_name,
            new_key,
            new_key_mods,
            new_key_action,
        )

    @gs.assert_built
    def remove_keybind(self, keybind_name: str) -> None:
        """
        Remove an existing keybind by name.

        Parameters
        ----------
        keybind_name : str
            The name of the keybind to remove.
        """
        self._pyrender_viewer.remove_keybind(keybind_name)

    def add_plugin(self, plugin: "ViewerPlugin") -> "ViewerPlugin":
        """
        Add a viewer plugin to the viewer.

        Parameters
        ----------
        plugin : ViewerPlugin
            The viewer plugin to add.
        """
        # Register first so a failure in the plugin's build() leaves the viewer's plugin set unchanged; the plugin
        # is only recorded once it has successfully attached. The render lock serializes the attach with the viewer
        # thread when running in a thread; it is reentrant, so this is a no-op when called from a render callback
        # (e.g. the overlay Plugins tab) that already holds it.
        if self.is_built:
            with self._pyrender_viewer.render_lock:
                self._pyrender_viewer.register_plugin(plugin)
        self._plugins.append(plugin)
        return plugin

    def remove_plugin(self, plugin: "ViewerPlugin") -> None:
        """
        Remove a viewer plugin from the viewer, detaching it from the live render loop if already built.

        Parameters
        ----------
        plugin : ViewerPlugin
            The viewer plugin to remove.
        """
        if plugin in self._plugins:
            self._plugins.remove(plugin)
        if self.is_built and plugin in self._pyrender_viewer.plugins:
            # Hold the (reentrant) render lock so the detach is serialized with the viewer thread, then drop the
            # plugin via copy-on-write so a dispatch loop already iterating self.plugins finishes on its snapshot.
            with self._pyrender_viewer.render_lock:
                self._pyrender_viewer.plugins = [p for p in self._pyrender_viewer.plugins if p is not plugin]
                self._pyrender_viewer.remove_handlers(plugin)
                plugin.on_close()

    @gs.assert_built
    def toggle_recording(self) -> bool:
        """
        Start or stop recording the on-screen viewer to a video file, returning the resulting record state.

        Stopping prompts for a destination file. This is the same on-screen capture toggled by the 'R' shortcut.
        """
        return self._pyrender_viewer.toggle_recording()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def recording(self) -> bool:
        """Whether the viewer is currently recording its on-screen output to a video file."""
        return self._pyrender_viewer.viewer_flags["record"]

    @property
    def plugins(self):
        """The registered viewer plugins, read-only; use ``add_plugin`` to register one."""
        return tuple(self._plugins)

    @property
    def is_built(self):
        return self._is_built

    @property
    def res(self):
        return self._res

    @property
    def refresh_rate(self):
        return self._refresh_rate

    @property
    def realtime_factor(self):
        return self._realtime_factor

    @realtime_factor.setter
    def realtime_factor(self, value):
        # Rebuild the pacer for the new factor (None -> uncapped). Reassigning the pacer reference is atomic, so
        # update() reading it on the stepping thread always sees a consistent object.
        self._realtime_factor = value
        if value is None or self.scene is None:
            self._realtime_pacer = None
        else:
            self._realtime_pacer = Rate(value / self.scene.sim.dt)

    @property
    def camera_pos(self):
        """
        Get the camera's current position.
        """
        return np.array(self._pyrender_viewer._trackball._n_pose[:3, 3])

    @property
    def camera_lookat(self):
        """
        Get the camera's current lookat point.
        """
        pos = np.array(self._pyrender_viewer._trackball._n_pose[:3, 3])
        z = self._pyrender_viewer._trackball._n_pose[:3, 2]
        return pos - z

    @property
    def camera_pose(self):
        """
        Get the camera's current pose represented by a 4x4 matrix.
        """
        return np.array(self._pyrender_viewer._trackball._n_pose)

    @property
    def camera_up(self):
        return self._camera_up

    @property
    def camera_fov(self):
        return self._camera_fov
