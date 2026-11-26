"""PBR renderer for Python.

Author: Matthew Matl
"""

import os
import sys

import numpy as np

import PIL
import pyglet
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_TEST,
    GL_DRAW_FRAMEBUFFER,
    GL_DST_COLOR,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FUNC_ADD,
    GL_GREATER,
    GL_HALF_FLOAT,
    GL_LESS,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_MAX_SAMPLES,
    GL_MIN,
    GL_MULTISAMPLE,
    GL_NEAREST,
    GL_NONE,
    GL_ONE,
    GL_READ_FRAMEBUFFER,
    GL_RENDERBUFFER,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_2D_ARRAY,
    GL_TRUE,
    GL_UNSIGNED_BYTE,
    GL_ZERO,
    glBindFramebuffer,
    glBindRenderbuffer,
    glBindTexture,
    glBlendEquation,
    glBlendEquationSeparate,
    glBlendFuncSeparate,
    glBlitFramebuffer,
    glClear,
    glClearColor,
    glDeleteFramebuffers,
    glDeleteRenderbuffers,
    glDepthFunc,
    glDepthMask,
    glDepthRange,
    glDisable,
    glDrawBuffer,
    glEnable,
    glFramebufferRenderbuffer,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGenerateMipmap,
    glGetIntegerv,
    glReadBuffer,
    glRenderbufferStorage,
    glRenderbufferStorageMultisample,
    glViewport,
)

from .constants import (
    DEFAULT_Z_FAR,
    DEFAULT_Z_NEAR,
    GLTF,
    BufFlags,
    LightType,
    ProgramFlags,
    RenderFlags,
    TexFlags,
    TextAlign,
    ViewFlags,
)
from .font import FontCache
from .jit_render import is_env_pass
from .light import DirectionalLight, PointLight, SpotLight
from .sampler import Sampler
from .shader_program import ShaderProgramCache
from .texture import Texture
from .utils import format_color_vector


MARKER_XRAY_DIM_FACTOR = 0.3
MARKER_XRAY_ALPHA = 0.4


class Renderer(object):
    """Class for handling all rendering operations on a scene.

    Note
    ----
    This renderer relies on the existence of an OpenGL context and
    does not create one on its own.

    Parameters
    ----------
    viewport_width : int
        Width of the viewport in pixels.
    viewport_height : int
        Width of the viewport height in pixels.
    point_size : float, optional
        Size of points in pixels. Defaults to 1.0.
    """

    def __init__(self, viewport_width, viewport_height, jit, point_size=1.0):
        self.dpscale = 1

        # Scaling needed on retina displays for old pyglet releases
        if sys.platform == "darwin" and pyglet.version < "2.0":
            self.dpscale = 2

        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.point_size = point_size

        # Optional framebuffer for offscreen renders
        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_ms = None
        self._main_cb_ms = None
        self._main_db_ms = None
        self._main_fb_dims = (None, None)
        self._shadow_fb = None
        self._transmittance_fb = None
        self._floor_fb = None
        self._latest_znear = DEFAULT_Z_NEAR
        self._latest_zfar = DEFAULT_Z_FAR

        # Shader Program Cache
        self._program_cache = ShaderProgramCache()
        self._font_cache = FontCache()

        self._floor_texture_color = None
        self._floor_texture_depth = None

        self.jit = jit

    @property
    def viewport_width(self):
        """int : The width of the main viewport, in pixels."""
        return self._viewport_width

    @viewport_width.setter
    def viewport_width(self, value):
        self._viewport_width = self.dpscale * value

    @property
    def viewport_height(self):
        """int : The height of the main viewport, in pixels."""
        return self._viewport_height

    @viewport_height.setter
    def viewport_height(self, value):
        self._viewport_height = self.dpscale * value

    @property
    def point_size(self):
        """float : The size of screen-space points, in pixels."""
        return self._point_size

    @point_size.setter
    def point_size(self, value):
        self._point_size = float(value)

    def render(self, scene, flags, seg_node_map=None, *, is_first_pass=True, force_skip_shadows=False, env_idx=None):
        """Render a scene with the given set of flags.

        Parameters
        ----------
        scene : :class:`Scene`
            A scene to render.
        flags : int
            A specification from :class:`.RenderFlags`.
        seg_node_map : dict
            A map from :class:`.Node` objects to (3,) colors for each.
            If specified along with flags set to :attr:`.RenderFlags.SEG`,
            the color image will be a segmentation image.

        Returns
        -------
        color_im : (h, w, 3) uint8 or (h, w, 4) uint8
            If :attr:`RenderFlags.OFFSCREEN` is set, the color buffer. This is
            normally an RGB buffer, but if :attr:`.RenderFlags.RGBA` is set,
            the buffer will be a full RGBA buffer.
        depth_im : (h, w) float32
            If :attr:`RenderFlags.OFFSCREEN` is set, the depth buffer
            in linear units.
        """
        # A pass with the environments drawn side by side moves every environment-instanced primitive by its
        # environment offset (see 'JITRenderer.env_offset_buffer'). A pass rendering the environments one at a time
        # draws them where they are, so their lighting and shadows are the same in every environment.
        is_grid = not is_env_pass(flags)

        # Update context with meshes and textures
        if is_first_pass:
            self.jit.update_context(scene, flags)
            for texture in scene.updated_textures:
                texture.upload_layers()
            scene.updated_textures.clear()
            self.jit.update(scene)
            all_ready = self.jit.set_lighting(scene, is_grid)

            # Flush queued buffer updates AFTER jit.update(scene) so that new
            # nodes created by set_primitive -> _add_to_context already have
            # their GPU buffers allocated and can receive the data.
            self.jit.flush_buffer()

            if not all_ready:
                # Shadow textures not yet initialized - skip this frame to avoid
                # flickering. The caller should display the previous frame.
                return ()

        if flags & RenderFlags.SEG or flags & RenderFlags.DEPTH_ONLY or flags & RenderFlags.NORMAL:
            flags &= ~RenderFlags.REFLECTIVE_FLOOR

        # The normals are drawn by a pass of their own, which the flags tell apart from the others (see 'forward_pass'
        # in jit_render.py)
        normal_flags = flags & (RenderFlags.VERTEX_NORMALS | RenderFlags.FACE_NORMALS)
        flags &= ~normal_flags

        if env_idx is not None:
            envs_idx = (env_idx,)
            use_env_idx = False
        elif flags & RenderFlags.ENV_SEPARATE and flags & RenderFlags.OFFSCREEN and scene.n_envs > 1:
            envs_idx = range(scene.n_envs)
            use_env_idx = True
        else:
            envs_idx = (-1,)
            use_env_idx = False

        retval_list = None
        for env_idx in envs_idx:
            # Render necessary shadow maps
            if not (force_skip_shadows or flags & RenderFlags.SEG or flags & RenderFlags.DEPTH_ONLY):
                for ln in scene.light_nodes:
                    take_pass = False
                    if isinstance(ln.light, DirectionalLight) and flags & RenderFlags.SHADOWS_DIRECTIONAL:
                        take_pass = True
                    elif isinstance(ln.light, SpotLight) and flags & RenderFlags.SHADOWS_SPOT:
                        take_pass = True
                    elif isinstance(ln.light, PointLight) and flags & RenderFlags.SHADOWS_POINT:
                        take_pass = True
                    # The shadow texture lives on the light, so one map serves every renderer of the scene (the
                    # viewer window and each camera): it is rendered again once the scene changed or another
                    # environment is drawn. The pass draws every opaque mesh in fill mode whatever the flags, so the
                    # key is the scene revision and the environment alone.
                    if take_pass and ln.light.shadow_map_revision != (scene.revision, env_idx):
                        if isinstance(ln.light, PointLight):
                            self._point_shadow_mapping_pass(scene, ln, flags, env_idx=env_idx)
                        else:
                            self._shadow_mapping_pass(scene, ln, flags, env_idx=env_idx)
                        glBindFramebuffer(GL_FRAMEBUFFER, 0)
                        ln.light.shadow_map_revision = (scene.revision, env_idx)

            if flags & RenderFlags.REFLECTIVE_FLOOR:
                self._floor_pass(scene, flags, env_idx=env_idx)

            retval = self._forward_pass(scene, flags, seg_node_map=seg_node_map, env_idx=env_idx)
            if retval is not None:
                if retval_list is None:
                    retval_list = tuple([val] for val in retval)
                else:
                    for idx, val in enumerate(retval):
                        retval_list[idx].append(val)

            # The normals are drawn over the image after it is read, so they show on screen alone
            if normal_flags:
                self._normal_pass(scene, (flags & ~RenderFlags.REFLECTIVE_FLOOR) | normal_flags, env_idx=env_idx)

        # Update camera settings for retrieving depth buffers
        self._latest_znear = scene.main_camera_node.camera.znear
        self._latest_zfar = scene.main_camera_node.camera.zfar

        if retval_list is None:
            return

        if use_env_idx:
            retval_list = tuple(np.stack(val_list, axis=0) for val_list in retval_list)
        else:
            retval_list = tuple(val_list[0] for val_list in retval_list)
        return retval_list

    def render_images(
        self,
        scene,
        flags,
        camera_node=None,
        *,
        rgb=True,
        depth=False,
        seg_node_map=None,
        normal=False,
        env_idx=None,
        debug_view=False,
    ):
        """Render the images asked for, from a camera of a scene.

        Parameters
        ----------
        scene : Scene
            The scene to render.
        flags : int
            The flags of the render (see RenderFlags), on top of those that the other arguments set.
        camera_node : Node, optional
            The node of the camera to render from. Defaults to the main camera of the scene.
        rgb : bool, optional
            Whether to render the color image. Defaults to True.
        depth : bool, optional
            Whether to render the depth image. Defaults to False.
        seg_node_map : dict, optional
            The color of each node in a segmentation image, rendered in place of the color image when given. Defaults
            to None.
        normal : bool, optional
            Whether to render the image of the normals of the surfaces. Defaults to False.
        env_idx : int, optional
            The environment to render, all of them when None. Defaults to None.
        debug_view : bool, optional
            Whether to draw the markers, seen through what hides them, and the debug representation of the entities.
            Defaults to False.

        Returns
        -------
        images : tuple
            The color or segmentation image, the depth image and the normal image, those rendered offscreen alone and
            in this order.
        """
        if camera_node is not None:
            saved_camera_node = scene.main_camera_node
            scene.main_camera_node = camera_node

        flags |= RenderFlags.MARKER_XRAY if debug_view else RenderFlags.SKIP_MARKERS
        self.jit.view = ViewFlags.DEBUG if debug_view else ViewFlags.NORMAL
        if seg_node_map is not None:
            flags |= RenderFlags.SEG
        if depth:
            flags |= RenderFlags.RET_DEPTH
            if not (rgb or seg_node_map is not None):
                flags |= RenderFlags.DEPTH_ONLY

        images = ()
        is_first_pass = True
        if rgb or depth or seg_node_map is not None:
            images = self.render(scene, flags, seg_node_map, env_idx=env_idx)
            is_first_pass = False
        if normal:
            normal_flags = RenderFlags.NORMAL | RenderFlags.OFFSCREEN
            normal_flags |= flags & (RenderFlags.ENV_SEPARATE | RenderFlags.SKIP_MARKERS)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            normal_image, *_ = self.render(
                scene, normal_flags, is_first_pass=is_first_pass, force_skip_shadows=True, env_idx=env_idx
            )
            images = (*images, normal_image)

        if camera_node is not None:
            scene.main_camera_node = saved_camera_node
        return images

    def render_text(
        self, text, x, y, font_name="OpenSans-Regular", font_pt=40, color=None, scale=1.0, align=TextAlign.BOTTOM_LEFT
    ):
        """Render text into the current viewport.

        Note
        ----
        This cannot be done into an offscreen buffer.

        Parameters
        ----------
        text : str
            The text to render.
        x : int
            Horizontal pixel location of text.
        y : int
            Vertical pixel location of text.
        font_name : str
            Name of font, from the ``pyrender/fonts`` folder, or
            a path to a ``.ttf`` file.
        font_pt : int
            Height of the text, in font points.
        color : (4,) float
            The color of the text. Default is black.
        scale : int
            Scaling factor for text.
        align : int
            One of the :class:`TextAlign` options which specifies where the
            ``x`` and ``y`` parameters lie on the text. For example,
            :attr:`TextAlign.BOTTOM_LEFT` means that ``x`` and ``y`` indicate
            the position of the bottom-left corner of the textbox.
        """
        x *= self.dpscale
        y *= self.dpscale
        font_pt *= self.dpscale

        if color is None:
            color = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            color = format_color_vector(color, 4)

        # Set up viewport for render
        self._configure_forward_pass_viewport(0)

        # Load font
        font = self._font_cache.get_font(font_name, font_pt)
        if not font._in_context():
            font._add_to_context()

        # Load program
        program = self._get_text_program()
        program._bind()

        # Set uniforms
        p = np.eye(4)
        p[0, 0] = 2.0 / self.viewport_width
        p[0, 3] = -1.0
        p[1, 1] = 2.0 / self.viewport_height
        p[1, 3] = -1.0
        program.set_uniform("projection", p)
        program.set_uniform("text_color", color)

        # Draw text
        font.render_string(text, x, y, scale, align)

    def render_texts(self, texts, x, y, font_name="UbuntuMono-Regular", font_pt=40, color=None, scale=1.0):
        x *= self.dpscale
        y *= self.dpscale
        font_pt *= self.dpscale

        if color is None:
            color = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            color = format_color_vector(color, 4)

        # Set up viewport for render
        self._configure_forward_pass_viewport(0)

        # Load font
        font = self._font_cache.get_font(font_name, font_pt)
        if not font._in_context():
            font._add_to_context()

        # Load program
        program = self._get_text_program()
        program._bind()

        # Set uniforms
        p = np.eye(4)
        p[0, 0] = 2.0 / self.viewport_width
        p[0, 3] = -1.0
        p[1, 1] = 2.0 / self.viewport_height
        p[1, 3] = -1.0
        program.set_uniform("projection", p)
        program.set_uniform("text_color", color)

        # Draw text
        for i, text in enumerate(texts):
            font.render_string(text, x, int(y - i * font_pt * 1.1), scale, TextAlign.TOP_LEFT)

    def delete(self):
        """Free all allocated OpenGL resources."""
        # Free shaders
        self._program_cache.clear()

        # Free fonts
        self._font_cache.clear()

        self._delete_main_framebuffer()
        self._delete_shadow_framebuffer()
        self._delete_floor_framebuffer()

    ###########################################################################
    # Rendering passes
    ###########################################################################

    def _floor_pass(self, scene, flags, seg_node_map=None, env_idx=-1):
        self._configure_floor_pass_viewport(flags)
        # The alpha of the mirror image marks where it holds mirrored geometry (see floor_tex in mesh.frag)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        V, P = self._get_camera_matrices(scene, env_idx)
        cam_pos = self._get_camera_pose(scene, env_idx)[:3, 3]

        screen_size = np.array([self.viewport_width, self.viewport_height], np.float32)

        self.jit.forward_pass(
            self,
            V,
            P,
            cam_pos,
            flags | RenderFlags.SKIP_FLOOR,
            ProgramFlags.USE_MATERIAL,
            screen_size,
            reflection_mat=self.jit.reflection_mat,
            env_idx=env_idx,
        )

        # A rough floor samples a coarser level of the mirror image, which blurs it
        glBindTexture(GL_TEXTURE_2D, self._floor_texture_color._texid)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _forward_pass(self, scene, flags, seg_node_map=None, env_idx=-1):
        # Set up viewport for render
        self._configure_forward_pass_viewport(flags)

        # Clear it
        if flags & RenderFlags.SEG:
            glClearColor(0.0, 0.0, 0.0, 1.0)
            if seg_node_map is None:
                seg_node_map = {}
        else:
            glClearColor(*scene.bg_color)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if flags & RenderFlags.SEG or flags & RenderFlags.DEPTH_ONLY:
            glDisable(GL_MULTISAMPLE)
        else:
            glEnable(GL_MULTISAMPLE)

        # Set up camera matrices
        V, P = self._get_camera_matrices(scene, env_idx)
        cam_pos = self._get_camera_pose(scene, env_idx)[:3, 3]

        floor_tex = self._floor_texture_color._texid if flags & RenderFlags.REFLECTIVE_FLOOR else 0
        screen_size = np.array([self.viewport_width, self.viewport_height], np.float32)
        color_list = None
        if flags & RenderFlags.SEG:
            # A primitive missing from the map takes a color below -1, which the pass draws nothing for
            color_list = np.full((len(self.jit.node_list), 3), -2.0, np.float32)
            for i, node in enumerate(self.jit.node_list):
                if node in seg_node_map:
                    color_list[i] = seg_node_map[node] / 255.0

        if flags & (RenderFlags.SEG | RenderFlags.DEPTH_ONLY | RenderFlags.SKIP_MARKERS):
            self.jit.forward_pass(
                self,
                V,
                P,
                cam_pos,
                flags,
                ProgramFlags.USE_MATERIAL,
                screen_size,
                color_list,
                floor_tex=floor_tex,
                env_idx=env_idx,
            )
        else:
            # The markers are drawn last, so that the depth buffer holds the rest of the scene alone when the X-ray pass
            # tests it, which ghosts the markers that real objects hide and none that another marker hides (dark rings
            # at the joints of chained debug cylinders otherwise)
            for markers_only in (False, True):
                if markers_only and flags & RenderFlags.MARKER_XRAY:
                    self._marker_xray_pass(V, P, cam_pos, flags, screen_size, env_idx)
                self.jit.forward_pass(
                    self,
                    V,
                    P,
                    cam_pos,
                    flags if markers_only else flags | RenderFlags.SKIP_MARKERS,
                    ProgramFlags.USE_MATERIAL,
                    screen_size,
                    floor_tex=floor_tex,
                    env_idx=env_idx,
                    markers_only=markers_only,
                )

        # If doing offscreen render, copy result from framebuffer and return
        if flags & RenderFlags.OFFSCREEN:
            return self._read_main_framebuffer(scene, flags)

    def _marker_xray_pass(self, V, P, cam_pos, flags, screen_size, env_idx):
        """Render markers behind geometry with darkened transparency (X-ray effect)."""
        marker_mask = self.jit.is_marker
        if not marker_mask.any():
            return

        # Save and dim marker colors, force alpha blending
        saved_base_color_factor = self.jit.base_color_factor[marker_mask].copy()
        saved_is_blended = self.jit.is_blended[marker_mask].copy()
        self.jit.base_color_factor[marker_mask, :3] *= MARKER_XRAY_DIM_FACTOR
        self.jit.base_color_factor[marker_mask, 3] = MARKER_XRAY_ALPHA
        self.jit.is_blended[marker_mask] = True

        # Draw only occluded fragments, without touching the depth buffer.
        glDepthFunc(GL_GREATER)
        glDepthMask(GL_FALSE)

        xray_flags = flags & ~(RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT | RenderFlags.SHADOWS_POINT)
        self.jit.forward_pass(
            self, V, P, cam_pos, xray_flags, ProgramFlags.USE_MATERIAL, screen_size, env_idx=env_idx, markers_only=True
        )

        # Restore GL state and marker data
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        self.jit.base_color_factor[marker_mask] = saved_base_color_factor
        self.jit.is_blended[marker_mask] = saved_is_blended

    def _point_shadow_mapping_pass(self, scene, light_node, flags, env_idx=-1):
        light = light_node.light
        position = scene.get_pose(light_node)[:3, 3]
        camera = light._get_shadow_camera(scene.scale)
        projection = camera.get_projection_matrix()
        view = light._get_view_matrices(position)
        light_matrix = projection @ view

        self._configure_point_shadow_mapping_viewport(light, flags)

        self.jit.shadow_mapping_pass(
            self,
            None,
            None,
            flags,
            ProgramFlags.POINT_SHADOW,
            env_idx=env_idx,
            light_matrix=light_matrix,
            light_pos=position,
        )
        self._transmittance_pass(
            light, flags, ProgramFlags.POINT_SHADOW, env_idx, light_matrix=light_matrix, light_pos=position
        )

    def _shadow_mapping_pass(self, scene, light_node, flags, env_idx=-1):
        light = light_node.light

        # Set up viewport for render
        self._configure_shadow_mapping_viewport(light, flags)

        # Set up camera matrices
        V, P = self._get_light_cam_matrices(scene, light_node, flags)

        self.jit.shadow_mapping_pass(self, V, P, flags, ProgramFlags.SHADOW, env_idx=env_idx)
        self._transmittance_pass(light, flags, ProgramFlags.SHADOW, env_idx, V=V, P=P)

    def _transmittance_pass(
        self, light, flags, program_flags, env_idx, V=None, P=None, light_matrix=None, light_pos=None
    ):
        """Render what transparent surfaces let through of a light into its transmittance map (see mesh_depth.frag)."""
        if self._transmittance_fb is None:
            self._transmittance_fb = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._transmittance_fb)
        light.transmittance_texture._bind_as_color_attachment()
        glDrawBuffer(GL_COLOR_ATTACHMENT0)
        glViewport(0, 0, light.transmittance_texture.width, light.transmittance_texture.height)
        # White lets the light through untinted, behind a transparent surface at the far end of the depth range
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        if not self.jit.is_transparent.any():
            return

        # The colors multiply over the transparent surfaces that a ray of the light crosses, whatever their order, and
        # the alpha keeps the depth of the nearest one
        glDisable(GL_DEPTH_TEST)
        glBlendFuncSeparate(GL_DST_COLOR, GL_ZERO, GL_ONE, GL_ONE)
        glBlendEquationSeparate(GL_FUNC_ADD, GL_MIN)
        self.jit.shadow_mapping_pass(
            self,
            V,
            P,
            flags,
            program_flags | ProgramFlags.TRANSMITTANCE,
            env_idx=env_idx,
            light_matrix=light_matrix,
            light_pos=light_pos,
            is_transmittance=True,
        )
        glBlendEquation(GL_FUNC_ADD)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    def _normal_pass(self, scene, flags, env_idx=-1):
        self._configure_forward_pass_viewport(flags)
        V, P = self._get_camera_matrices(scene, env_idx)
        cam_pos = self._get_camera_pose(scene, env_idx)[:3, 3]
        screen_size = np.array([self.viewport_width, self.viewport_height], np.float32)
        program_flags = ProgramFlags.NONE
        if flags & RenderFlags.VERTEX_NORMALS:
            program_flags |= ProgramFlags.VERTEX_NORMALS
        if flags & RenderFlags.FACE_NORMALS:
            program_flags |= ProgramFlags.FACE_NORMALS
        self.jit.forward_pass(self, V, P, cam_pos, flags, program_flags, screen_size, env_idx=env_idx)

    ###########################################################################
    # Camera Matrix Management
    ###########################################################################

    def _get_camera_matrices(self, scene, env_idx):
        main_camera_node = scene.main_camera_node
        if main_camera_node is None:
            raise ValueError("Cannot render scene without a camera")
        P = main_camera_node.camera.get_projection_matrix(width=self.viewport_width, height=self.viewport_height)
        pose = self._get_camera_pose(scene, env_idx)
        V = np.linalg.inv(pose)  # V maps from world to camera
        return V, P

    def _get_camera_pose(self, scene, env_idx):
        cam_pos = scene.get_pose(scene.main_camera_node)
        if len(cam_pos.shape) == 3:
            if cam_pos.shape[0] != 1:
                assert env_idx != -1, "We have a multiple camera pose scene, we should be rendering per env"
                cam_pos = cam_pos[env_idx]
            else:
                cam_pos = cam_pos[0]
        return cam_pos

    def _get_light_cam_matrices(self, scene, light_node, flags):
        light = light_node.light
        pose = scene.get_pose(light_node)
        camera = light._get_shadow_camera(scene.scale)
        P = camera.get_projection_matrix()
        if isinstance(light, DirectionalLight):
            direction = -pose[:3, 2]
            c = scene.centroid
            loc = c - direction * scene.scale
            pose[:3, 3] = loc
        V = np.linalg.inv(pose)  # V maps from world to camera
        return V, P

    ###########################################################################
    # Shader Program Management
    ###########################################################################

    def _get_text_program(self):
        program = self._program_cache.get_program(vertex_shader="text.vert", fragment_shader="text.frag")

        if not program._in_context():
            program._add_to_context()

        return program

    def _get_primitive_program(self, primitive, flags, program_flags):
        vertex_shader = None
        fragment_shader = None
        geometry_shader = None
        defines = {}

        if program_flags & ProgramFlags.POINT_SHADOW:
            vertex_shader = "point_shadow.vert"
            fragment_shader = "mesh_depth.frag"
            geometry_shader = "point_shadow.geom"
            defines["POINT_SHADOW"] = 1
        elif (
            program_flags & ProgramFlags.USE_MATERIAL
            and not flags & RenderFlags.DEPTH_ONLY
            and not flags & RenderFlags.NORMAL
            and not flags & RenderFlags.SEG
        ):
            vertex_shader = "mesh.vert"
            fragment_shader = "mesh.frag"
            # The floor alone samples the mirror image of the scene (see REFLECTIVE_FLOOR in mesh.frag)
            if flags & RenderFlags.REFLECTIVE_FLOOR and primitive.is_floor:
                defines["REFLECTIVE_FLOOR"] = 1
            if primitive.double_sided:
                geometry_shader = "mesh_double_sided.geom"
                defines["DOUBLE_SIDED"] = 1
        # A primitive without normals is left out of the pass drawing them, and takes the program of a depth pass
        elif (
            program_flags & (ProgramFlags.VERTEX_NORMALS | ProgramFlags.FACE_NORMALS)
            and primitive.buf_flags & BufFlags.NORMAL
        ):
            vertex_shader = "vertex_normals.vert"
            if primitive.mode == GLTF.POINTS:
                geometry_shader = "vertex_normals_pc.geom"
            else:
                geometry_shader = "vertex_normals.geom"
            fragment_shader = "vertex_normals.frag"
        elif flags & RenderFlags.NORMAL:
            vertex_shader = "mesh_normal.vert"
            fragment_shader = "mesh_normal.frag"
            if primitive.double_sided:
                geometry_shader = "mesh_normal_double_sided.geom"
                defines["DOUBLE_SIDED"] = 1
        elif flags & RenderFlags.SEG:
            vertex_shader = "segmentation.vert"
            fragment_shader = "segmentation.frag"
            if primitive.double_sided:
                geometry_shader = "segmentation_double_sided.geom"
                defines["DOUBLE_SIDED"] = 1
        else:
            vertex_shader = "mesh_depth.vert"
            fragment_shader = "mesh_depth.frag"

        # A transparent surface casts a shadow as dark as it is opaque, for every kind of light (see mesh_depth.frag).
        # The layer that a texture array shows may change after the program is built, so a texture array always
        # samples its alpha.
        material = primitive.material
        is_shadow_program = program_flags & (ProgramFlags.SHADOW | ProgramFlags.POINT_SHADOW)
        if is_shadow_program:
            is_base_color_array = (
                material.tex_flags & TexFlags.BASE_COLOR and material.baseColorTexture.tex_type == GL_TEXTURE_2D_ARRAY
            )
            if primitive.is_transparent or is_base_color_array:
                defines["TRANSPARENT_SHADOW"] = 1
                if material.tex_flags & TexFlags.BASE_COLOR:
                    defines["HAS_BASE_COLOR_TEX"] = 1
                if is_base_color_array:
                    defines["BASE_COLOR_TEX_ARRAY"] = 1
                    if material.baseColorTexture.data_format == GL_HALF_FLOAT:
                        defines["BASE_COLOR_TEX_LINEAR"] = 1
                if program_flags & ProgramFlags.TRANSMITTANCE:
                    defines["TRANSMITTANCE"] = 1

        # Set up vertex buffer DEFINES
        bf = primitive.buf_flags
        buf_idx = 1
        if bf & BufFlags.NORMAL:
            defines["NORMAL_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TANGENT:
            defines["TANGENT_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TEXCOORD_0:
            defines["TEXCOORD_0_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.TEXCOORD_1:
            defines["TEXCOORD_1_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.COLOR_0:
            defines["COLOR_0_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.JOINTS_0:
            defines["JOINTS_0_LOC"] = buf_idx
            buf_idx += 1
        if bf & BufFlags.WEIGHTS_0:
            defines["WEIGHTS_0_LOC"] = buf_idx
            buf_idx += 1
        defines["INST_M_LOC"] = buf_idx
        defines["INST_ENV_OFFSET_LOC"] = buf_idx + 4
        if primitive.inst_tints is not None:
            defines["INST_TINT_LOC"] = buf_idx + 5

        # Set up shadow mapping defines
        if flags & RenderFlags.SHADOWS_DIRECTIONAL:
            defines["DIRECTIONAL_LIGHT_SHADOWS"] = 1
        if flags & RenderFlags.SHADOWS_SPOT:
            defines["SPOT_LIGHT_SHADOWS"] = 1
        if flags & RenderFlags.SHADOWS_POINT:
            defines["POINT_LIGHT_SHADOWS"] = 1
        # The light arrays hold the lights of the scene, with one slot at least, so that the samplers of their shadow
        # and transmittance maps fit in the texture units of every driver along with those of the material
        for name, light_type in (
            ("MAX_DIRECTIONAL_LIGHTS", LightType.DIRECTIONAL),
            ("MAX_POINT_LIGHTS", LightType.POINT),
            ("MAX_SPOT_LIGHTS", LightType.SPOT),
        ):
            defines[name] = max(np.count_nonzero(self.jit.lights_type == light_type), 1)

        # Set up vertex normal defines
        if program_flags & ProgramFlags.VERTEX_NORMALS:
            defines["VERTEX_NORMALS"] = 1
        if program_flags & ProgramFlags.FACE_NORMALS:
            defines["FACE_NORMALS"] = 1

        # Set up material texture defines
        if program_flags & ProgramFlags.USE_MATERIAL:
            tf = primitive.material.tex_flags
            if tf & TexFlags.NORMAL:
                defines["HAS_NORMAL_TEX"] = 1
            if tf & TexFlags.OCCLUSION:
                defines["HAS_OCCLUSION_TEX"] = 1
            if tf & TexFlags.EMISSIVE:
                defines["HAS_EMISSIVE_TEX"] = 1
                if primitive.material.emissiveTexture.tex_type == GL_TEXTURE_2D_ARRAY:
                    defines["EMISSIVE_TEX_ARRAY"] = 1
            if tf & TexFlags.BASE_COLOR:
                defines["HAS_BASE_COLOR_TEX"] = 1
                if primitive.material.baseColorTexture.tex_type == GL_TEXTURE_2D_ARRAY:
                    defines["BASE_COLOR_TEX_ARRAY"] = 1
                    if primitive.material.baseColorTexture.data_format == GL_HALF_FLOAT:
                        defines["BASE_COLOR_TEX_LINEAR"] = 1
            if tf & TexFlags.METALLIC_ROUGHNESS:
                defines["HAS_METALLIC_ROUGHNESS_TEX"] = 1

        program = self._program_cache.get_program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            defines=defines,
        )

        if not program._in_context():
            program._add_to_context()

        return program

    ###########################################################################
    # Viewport Management
    ###########################################################################

    def _configure_forward_pass_viewport(self, flags):
        # If using offscreen render, bind main framebuffer
        if flags & RenderFlags.OFFSCREEN:
            self._configure_main_framebuffer()
            if flags & RenderFlags.SEG or flags & RenderFlags.DEPTH_ONLY:
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            else:
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb_ms)
        else:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

        glViewport(0, 0, self.viewport_width, self.viewport_height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

    def _configure_floor_pass_viewport(self, flags):
        self._configure_floor_framebuffer()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._floor_fb)

        glViewport(0, 0, self.viewport_width, self.viewport_height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

    def _configure_point_shadow_mapping_viewport(self, light, flags):
        self._configure_shadow_framebuffer()
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fb)
        light.shadow_texture._bind_as_depth_attachment()
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, light.shadow_texture.width, light.shadow_texture.height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)

    def _configure_shadow_mapping_viewport(self, light, flags):
        self._configure_shadow_framebuffer()
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fb)
        light.shadow_texture._bind_as_depth_attachment()
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, light.shadow_texture.width, light.shadow_texture.height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)

    ###########################################################################
    # Framebuffer Management
    ###########################################################################

    def _configure_floor_framebuffer(self):
        if self._floor_texture_depth is None:
            self._floor_texture_depth = Texture(
                width=self.viewport_width, height=self.viewport_height, source_channels="D", data_format=GL_FLOAT
            )
            self._floor_texture_depth._add_to_context()
        if self._floor_texture_color is None:
            # The mirror image holds sRGB values within [0, 1], which 8 bits per channel store. A floating-point texture
            # takes no filtering of its levels on some renderers, which would leave the floor without its reflection.
            self._floor_texture_color = Texture(
                width=self.viewport_width,
                height=self.viewport_height,
                source_channels="RGBA",
                data_format=GL_UNSIGNED_BYTE,
                sampler=Sampler(
                    magFilter=GL_LINEAR,
                    minFilter=GL_LINEAR_MIPMAP_LINEAR,
                    wrapS=GLTF.CLAMP_TO_EDGE,
                    wrapT=GLTF.CLAMP_TO_EDGE,
                ),
            )
            self._floor_texture_color._add_to_context()
        if self._floor_fb is None:
            self._floor_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._floor_fb)
            self._floor_texture_color._bind_as_color_attachment()
            self._floor_texture_depth._bind_as_depth_attachment()

    def _delete_floor_framebuffer(self):
        if self._floor_fb is not None:
            glDeleteFramebuffers(1, [self._floor_fb])
            self._floor_fb = None

        if self._floor_texture_color is not None:
            self._floor_texture_color.delete()
            self._floor_texture_color = None

        if self._floor_texture_depth is not None:
            self._floor_texture_depth.delete()
            self._floor_texture_depth = None

    def _configure_shadow_framebuffer(self):
        if self._shadow_fb is None:
            self._shadow_fb = glGenFramebuffers(1)

    def _delete_shadow_framebuffer(self):
        if self._shadow_fb is not None:
            glDeleteFramebuffers(1, [self._shadow_fb])
            self._shadow_fb = None
        if self._transmittance_fb is not None:
            glDeleteFramebuffers(1, [self._transmittance_fb])
            self._transmittance_fb = None

    def _configure_main_framebuffer(self):
        # If mismatch with prior framebuffer, delete it
        if (
            self._main_fb is not None
            and self.viewport_width != self._main_fb_dims[0]
            or self.viewport_height != self._main_fb_dims[1]
        ):
            self._delete_main_framebuffer()

        # If framebuffer doesn't exist, create it
        if self._main_fb is None:
            # Generate standard buffer
            self._main_cb, self._main_db = glGenRenderbuffers(2)

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self.viewport_width, self.viewport_height)

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.viewport_width, self.viewport_height)

            self._main_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._main_cb)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._main_db)

            # Generate multisample buffer. Each driver picks its own sample positions within a pixel, so pixels on
            # edges differ between drivers. Tests use no multisampling, so that images match across platforms, as in the
            # viewer window (see 'PYTEST_VERSION' in viewer.py).
            num_samples = 0 if "PYTEST_VERSION" in os.environ else min(glGetIntegerv(GL_MAX_SAMPLES), 4)
            self._main_cb_ms, self._main_db_ms = glGenRenderbuffers(2)
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb_ms)
            glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, num_samples, GL_RGBA, self.viewport_width, self.viewport_height
            )
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db_ms)
            glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, num_samples, GL_DEPTH_COMPONENT24, self.viewport_width, self.viewport_height
            )
            self._main_fb_ms = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb_ms)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._main_cb_ms)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._main_db_ms)

            self._main_fb_dims = (self.viewport_width, self.viewport_height)

    def _delete_main_framebuffer(self):
        if self._main_fb is not None:
            glDeleteFramebuffers(2, [self._main_fb, self._main_fb_ms])
        if self._main_cb is not None:
            glDeleteRenderbuffers(2, [self._main_cb, self._main_cb_ms])
        if self._main_db is not None:
            glDeleteRenderbuffers(2, [self._main_db, self._main_db_ms])

        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_ms = None
        self._main_cb_ms = None
        self._main_db_ms = None
        self._main_fb_dims = (None, None)

    def _read_main_framebuffer(self, scene, flags):
        width, height = self._main_fb_dims

        if not (flags & RenderFlags.SEG or flags & RenderFlags.DEPTH_ONLY):
            # Bind framebuffer and blit buffers
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR)
            if flags & RenderFlags.RET_DEPTH:
                glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)

        # Read depth if requested
        if flags & RenderFlags.RET_DEPTH:
            z_near = scene.main_camera_node.camera.znear
            z_far = scene.main_camera_node.camera.zfar
            if z_far is None:
                z_far = -1.0
            depth_im = self.jit.read_depth_buf(width, height, z_near, z_far)

            # Resize
            depth_im = self._resize_image(depth_im, antialias=False)

        if flags & RenderFlags.DEPTH_ONLY:
            return (depth_im,)

        # Read color
        color_im = self.jit.read_color_buf(width, height, flags & RenderFlags.RGBA)

        # Resize
        color_im = self._resize_image(color_im, antialias=not flags & RenderFlags.SEG)

        if flags & RenderFlags.RET_DEPTH:
            return color_im, depth_im
        return (color_im,)

    def _resize_image(self, value, antialias):
        """Rescale the generated image if necessary."""
        if self.dpscale == 1:
            return value

        img = PIL.Image.fromarray(value)
        size = (self.viewport_width // self.dpscale, self.viewport_height // self.dpscale)
        resample = PIL.Image.BILINEAR if antialias else PIL.Image.NEAREST
        img = img.resize(size, resample=resample)
        return np.array(img, copy=False)

    def reload_program(self):
        self._program_cache.clear()
        self.jit.program_id.clear()
