"""Wrapper for offscreen rendering.

Author: Matthew Matl
"""

import os
import sys

from OpenGL.GL import (
    GL_RENDERER,
    glGetString,
)

import genesis as gs

from .constants import RenderFlags


class OffscreenRenderer(object):
    """A wrapper for offscreen rendering.

    Parameters
    ----------
    point_size : float
        The size of screen-space points in pixels.
    """

    def __init__(self, point_size=1.0, seg_node_map=None):
        self.point_size = point_size
        self._platform = None
        self._is_software = False
        self._has_valid_context = False
        self._create()
        self._seg_node_map = seg_node_map

    @property
    def point_size(self):
        """float : The pixel size of points in point clouds."""
        return self._point_size

    @point_size.setter
    def point_size(self, value):
        self._point_size = float(value)

    def make_current(self):
        """This function sets the current context and must be called before all rendering and GPU upload operations."""
        if self._has_valid_context:
            gs.raise_exception(
                "The method was called while having an other context current. Please call 'make_uncurrent' first."
            )

        self._platform.make_current()
        self._has_valid_context = True

    def make_uncurrent(self):
        """This function unsets the current context and must be called after all rendering and GPU upload operations
        are done.
        """
        if not self._has_valid_context:
            gs.raise_exception("The method was called before making a context current.")
        self._platform.make_uncurrent()
        self._has_valid_context = False

    def save_current_context(self):
        """Capture the current GL context as a restore callable (see 'Platform.save_current_context')."""
        return self._platform.save_current_context()

    def render(
        self,
        scene,
        renderer,
        rgb=True,
        seg=False,
        normal=False,
        depth=False,
        flags=RenderFlags.NONE,
        camera_node=None,
        shadow=False,
        plane_reflection=False,
        split_envs=False,
        env_idx=None,
        debug_view=False,
    ):
        """Render a scene with the given set of flags.

        Parameters
        ----------
        scene : :class:`Scene`
            A scene to render.
        flags : int
            A bitwise or of one or more flags from :class:`.RenderFlags`.

        Returns
        -------
        color_im : (h, w, 3) uint8 or (h, w, 4) uint8
            The color buffer in RGB format, or in RGBA format if
            :attr:`.RenderFlags.RGBA` is set.
            Not returned if flags includes :attr:`.RenderFlags.DEPTH_ONLY`.
        depth_im : (h, w) float32
            The depth buffer in linear units.
        """
        if seg and rgb:
            gs.raise_exception("RGB and segmentation map cannot be rendered in the same forward pass.")

        if not self._has_valid_context:
            gs.raise_exception(
                "Ensure that the right context is set before rendering. Please call the method 'make_current'."
            )

        flags |= RenderFlags.OFFSCREEN
        if shadow:
            flags |= RenderFlags.SHADOWS_ALL
        if plane_reflection:
            flags |= RenderFlags.REFLECTIVE_FLOOR
        if split_envs:
            flags |= RenderFlags.ENV_SEPARATE
        return renderer.render_images(
            scene,
            flags,
            camera_node,
            rgb=rgb,
            depth=depth,
            seg_node_map=self._seg_node_map if seg else None,
            normal=normal,
            env_idx=env_idx,
            debug_view=debug_view,
        )

    def delete(self):
        """Free all OpenGL resources."""
        # Do not force this context current before deleting it. The platforms' current-context state is
        # process/thread-global, so making it current here would clobber the context another renderer may be
        # using (e.g. while it is mid-render, when this renderer is being torn down by garbage collection).
        # 'delete_context' makes itself current only when the platform requires it.
        self._platform.delete_context()
        del self._platform
        self._platform = None

    def _create(self):
        # The PyOpenGL platform requested through PYOPENGL_PLATFORM, defaulting to the window-less one of the operating
        # system.
        platform = os.environ.get("PYOPENGL_PLATFORM", {"linux": "egl", "darwin": "cgl"}.get(sys.platform, "pyglet"))
        if platform not in ("osmesa", "pyglet", "egl", "cgl"):
            gs.logger.warning(f"PYOPENGL_PLATFORM='{platform}' not supported. Falling back to 'pyglet'.")
            platform = "pyglet"
        if sys.platform != "darwin" and platform == "cgl":
            gs.raise_exception("PYOPENGL_PLATFORM='cgl' is only supported on MacOS.")

        if platform == "pyglet":
            from .platforms.pyglet_platform import PygletPlatform

            self._platform = PygletPlatform()
        elif platform == "egl":
            from .platforms import egl

            if "EGL_DEVICE_ID" in os.environ:
                device_id = int(os.environ["EGL_DEVICE_ID"])
            else:
                device_id = None
            self._platform = egl.EGLPlatform(device_id)
        elif platform == "osmesa":
            from .platforms.osmesa import OSMesaPlatform

            self._platform = OSMesaPlatform()
        else:
            from .platforms.cgl import CGLPlatform

            self._platform = CGLPlatform()
        self._platform.init_context()

        self._platform.make_current()
        try:
            renderer = glGetString(GL_RENDERER).decode()
            gs.logger.debug(f"Using offscreen rendering OpenGL device: {renderer}")
            self._is_software = any(e in renderer for e in ("llvmpipe", "Apple Software Renderer"))
        except Exception:
            pass
        if self._is_software:
            gs.logger.info("Software rendering context detected. Beware rendering will be slow.")
        self._platform.make_uncurrent()

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass


__all__ = ["OffscreenRenderer"]
