import ctypes.util
import os
import sys

import numpy as np

import OpenGL

import genesis as gs
from genesis.repr_base import RBC
from genesis.utils.misc import has_display


# PyOpenGL binds every OpenGL function to the library of the platform selected when pyrender is first imported (see
# OSMesaPlatform), so the platform must be settled here. Offscreen rendering on Linux goes through EGL, and a headless
# machine lacking it can still render through OSMesa, a self-contained software rasterizer: select it upfront, leaving
# an explicit choice untouched. A machine with a display keeps its window-system platform, which the viewer relies on.
if sys.platform == "linux" and "PYOPENGL_PLATFORM" not in os.environ:
    if ctypes.util.find_library("EGL") is None and ctypes.util.find_library("OSMesa") and not has_display():
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
from genesis.ext import pyrender
from genesis.vis.camera import Camera


class Rasterizer(RBC):
    def __init__(self, viewer, context):
        self._viewer = viewer
        self._context = context
        self._camera_nodes = dict()
        self._camera_targets = dict()
        self._offscreen = self._viewer is None
        self._renderer = None
        self._buffer_updates = None

    def build(self):
        if self._context is None:
            return

        if self._offscreen:
            self._renderer = pyrender.OffscreenRenderer(seg_node_map=self._context.seg_node_map)

        self.visualizer = self._context.visualizer

    def add_camera(self, camera):
        self._camera_nodes[camera.uid] = self._context.add_node(
            pyrender.PerspectiveCamera(
                yfov=np.deg2rad(camera.fov),
                znear=camera.near,
                zfar=camera.far,
                aspectRatio=camera.aspect_ratio,
            ),
        )
        self._camera_targets[camera.uid] = pyrender.Renderer(camera.res[0], camera.res[1], self._context.jit)

    def update_camera(self, camera):
        self._camera_nodes[camera.uid].camera.yfov = np.deg2rad(camera.fov)
        self._context.set_node_pose(self._camera_nodes[camera.uid], camera.transform)
        self._context.update_camera_frustum(camera)

    def remove_camera(self, camera):
        self._context.remove_node(self._camera_nodes[camera.uid])
        del self._camera_nodes[camera.uid]
        if self._offscreen:
            self._camera_targets[camera.uid].delete()
        else:
            self._viewer.close_offscreen(self._camera_targets[camera.uid])
        del self._camera_targets[camera.uid]

    def render_camera(self, camera, rgb=True, depth=False, segmentation=False, normal=False, *, split_envs):
        """Render a camera. With 'split_envs', the environments are rendered one by one from the pose the camera
        holds in each, and stacked; otherwise one image is rendered of the environments laid out side by side."""
        # Update camera
        self.update_camera(camera)

        rgb_arr, depth_arr, seg_idxc_arr, normal_arr = None, None, None, None
        skip_markers = not camera.debug if isinstance(camera, Camera) else True
        if self._offscreen:
            # Set the context
            self._renderer.make_current()

            try:
                if rgb or depth or normal:
                    retval = self._renderer.render(
                        self._context._scene,
                        self._camera_targets[camera.uid],
                        camera_node=self._camera_nodes[camera.uid],
                        split_envs=split_envs,
                        rgb=rgb,
                        normal=normal,
                        seg=False,
                        depth=depth,
                        plane_reflection=rgb and self._context.plane_reflection,
                        shadow=rgb and self._context.shadow,
                        skip_markers=skip_markers,
                    )

                if segmentation:
                    seg_idxc_rgb_arr, *_ = self._renderer.render(
                        self._context._scene,
                        self._camera_targets[camera.uid],
                        camera_node=self._camera_nodes[camera.uid],
                        split_envs=split_envs,
                        rgb=False,
                        normal=False,
                        seg=True,
                        depth=False,
                        plane_reflection=False,
                        shadow=False,
                        skip_markers=skip_markers,
                    )
            finally:
                # Unset the context
                self._renderer.make_uncurrent()
        else:
            # Render
            if rgb or depth or normal:
                retval = self._viewer.render_offscreen(
                    self._camera_nodes[camera.uid],
                    self._camera_targets[camera.uid],
                    rgb=rgb,
                    depth=depth,
                    normal=normal,
                    seg=False,
                    skip_markers=skip_markers,
                    split_envs=split_envs,
                )

            if segmentation:
                seg_idxc_rgb_arr, *_ = self._viewer.render_offscreen(
                    self._camera_nodes[camera.uid],
                    self._camera_targets[camera.uid],
                    rgb=False,
                    depth=False,
                    normal=False,
                    seg=True,
                    skip_markers=skip_markers,
                    split_envs=split_envs,
                )

        if segmentation:
            seg_idxc_arr = self._context.seg_idxc_rgb_arr_to_idxc_arr(seg_idxc_rgb_arr)

        if rgb:
            rgb_arr = retval[0]
        if depth:
            depth_arr = retval[int(rgb)]
        if normal:
            normal_arr = retval[int(rgb + depth)]
        return rgb_arr, depth_arr, seg_idxc_arr, normal_arr

    def update_scene(self, force_render: bool = False):
        self._context.update(force_render)

    def destroy(self):
        for node in self._camera_nodes.values():
            self._context.remove_node(node)
        self._camera_nodes.clear()

        if self._offscreen:
            if self._renderer is not None:
                # Freeing this renderer's GL resources (its camera targets and its own context) requires its
                # context to be current, but the current context is process/thread-global and may belong to
                # another renderer (e.g. one mid-render). Save it, free on our context, then restore it after
                # destroying our context so the other renderer is left current rather than stranded. The saved
                # restore callable is self-contained, so it still works once our own platform is gone.
                try:
                    restore_context = self._renderer.save_current_context()
                    self._renderer.make_current()
                    for camera_target in self._camera_targets.values():
                        try:
                            camera_target.delete()
                        except (OpenGL.error.NullFunctionError, OSError):
                            pass
                    self._context.jit.delete()
                    self._renderer.delete()
                    if restore_context is not None:
                        restore_context()
                except (OpenGL.error.GLError, OpenGL.error.NullFunctionError, ImportError):
                    pass
                del self._renderer
                self._renderer = None
        else:
            for camera_target in self._camera_targets.values():
                try:
                    if self._viewer is not None:
                        self._viewer.close_offscreen(camera_target)
                except (OpenGL.error.NullFunctionError, OSError):
                    pass
        self._camera_targets.clear()

    @property
    def viewer(self):
        return self._viewer

    @property
    def offscreen(self):
        return self._offscreen
