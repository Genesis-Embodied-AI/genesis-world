"""OpenGL contexts without window, which the cameras render into."""

import os
import sys

from OpenGL.GL import (
    GL_RENDERER,
    glGetString,
)

import genesis as gs

from .constants import SOFTWARE_RENDERERS, RenderFlags


class OffscreenRenderer:
    """An OpenGL context without window, which the cameras render into.

    Parameters
    ----------
    seg_node_map : dict, optional
        The color of each node in a segmentation image. Defaults to None.
    """

    def __init__(self, seg_node_map=None):
        self._platform = None
        self._is_software = False
        self._has_valid_context = False
        self._create()
        self._seg_node_map = seg_node_map

    @property
    def is_software(self):
        """bool : Whether the OpenGL context is rendered on the CPU, which makes rendering slow."""
        return self._is_software

    def make_current(self):
        """Make the context current, which every render and upload requires."""
        if self._has_valid_context:
            gs.raise_exception("The context is current already. Please call 'make_uncurrent' first.")

        self._platform.make_current()
        self._has_valid_context = True

    def make_uncurrent(self):
        """Release the context, once the renders and uploads are done."""
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
        camera_node,
        rgb=True,
        depth=False,
        seg=False,
        normal=False,
        shadow=False,
        plane_reflection=False,
        split_envs=False,
        env_idx=None,
        debug_view=False,
    ):
        """Render the images asked for, from a camera of a scene, in the current context.

        Parameters
        ----------
        scene : Scene
            The scene to render.
        renderer : Renderer
            The renderer drawing the images, which holds the framebuffer of the camera.
        camera_node : Node
            The node of the camera to render from.
        rgb : bool, optional
            Whether to render the color image. Defaults to True.
        depth : bool, optional
            Whether to render the depth image. Defaults to False.
        seg : bool, optional
            Whether to render the segmentation image, which excludes the color image. Defaults to False.
        normal : bool, optional
            Whether to render the image of the normals of the surfaces. Defaults to False.
        shadow : bool, optional
            Whether the lights cast shadows. Defaults to False.
        plane_reflection : bool, optional
            Whether the floor reflects the scene. Defaults to False.
        split_envs : bool, optional
            Whether the environments are rendered one by one, rather than side by side in one image. Defaults to False.
        env_idx : int, optional
            The environment to render, all of them when None. Defaults to None.
        debug_view : bool, optional
            Whether to draw the markers and the debug representation of the entities. Defaults to False.

        Returns
        -------
        images : tuple
            The color or segmentation image, the depth image and the normal image, those asked for alone and in this
            order.
        """
        if seg and rgb:
            gs.raise_exception("RGB and segmentation map cannot be rendered in the same forward pass.")
        if not self._has_valid_context:
            gs.raise_exception(
                "Ensure that the right context is set before rendering. Please call the method 'make_current'."
            )
        return renderer.render_images(
            scene,
            RenderFlags.OFFSCREEN,
            camera_node,
            rgb=rgb,
            depth=depth,
            seg_node_map=self._seg_node_map if seg else None,
            normal=normal,
            shadow=shadow,
            plane_reflection=plane_reflection,
            split_envs=split_envs,
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
        renderer = glGetString(GL_RENDERER).decode()
        gs.logger.debug(f"Using offscreen rendering OpenGL device: {renderer}")
        self._is_software = any(name in renderer for name in SOFTWARE_RENDERERS)
        if self._is_software:
            gs.logger.info("Software rendering context detected. Beware rendering will be slow.")
        self._platform.make_uncurrent()

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass


__all__ = ["OffscreenRenderer"]
