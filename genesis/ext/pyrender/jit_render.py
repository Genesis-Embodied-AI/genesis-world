import contextlib
import ctypes
import os
from typing import NamedTuple

import numpy as np

import numba as nb
import OpenGL.GL as GL
import OpenGL.constant as GL_constant

from .camera import OrthographicCamera, PerspectiveCamera
from .constants import (
    BufFlags,
    FLOAT_SZ,
    LIGHT_FIELDS,
    LightType,
    MAX_N_LIGHTS,
    ProgramFlags,
    RenderFlags,
    TexChannels,
    TexFlags,
    UNIFORMS_NAME,
    ViewFlags,
)
from .light import PointLight, SpotLight
from .numba_gl_wrapper import GLWrapper
from .sampler import Sampler
from .scene import CORNER_INDICES
from .texture import CubeMapTexture, Texture
from .utils import transform_box


_DISABLE_OFFSCREEN_MARKERS = "GS_DISABLE_OFFSCREEN_MARKERS" in os.environ

# The rotation from the world frame to the camera frame of each face of a cube map, in the order of the faces of OpenGL
# (+x, -x, +y, -y, +z, -z). The camera looks along its -z axis, and its y axis follows the orientation of the faces that
# OpenGL cube maps define.
_CUBE_MAP_FACES_DIR = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float64)
_CUBE_MAP_FACES_UP = np.array([[0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [0, -1, 0]], np.float64)
CUBE_MAP_FACES_ROT = np.stack(
    (np.cross(_CUBE_MAP_FACES_UP, -_CUBE_MAP_FACES_DIR), _CUBE_MAP_FACES_UP, -_CUBE_MAP_FACES_DIR), axis=1
)


class PassPrograms(NamedTuple):
    """The shader programs drawing each primitive in a pass."""

    ids: np.ndarray
    """The program of each primitive."""
    uniforms_loc: np.ndarray
    """The location of each uniform of 'UNIFORMS_NAME' in the program of each primitive, one row per primitive."""


class PlacedGroup(NamedTuple):
    """Primitives placed through instance transforms that share an instance count, so that their poses stack."""

    idx: np.ndarray
    """The indices of the primitives of the group."""
    n_instances: int
    """The number of instances of each primitive of the group."""
    rows: np.ndarray
    """The rows of the transforms of the instances of each primitive of the group (see 'JITRenderer.instances_pose')."""


def load_const(const_name):
    c = GL.__getattribute__(const_name)
    if isinstance(c, GL_constant.FloatConstant):
        return float(c)
    if isinstance(c, GL_constant.IntConstant) or isinstance(c, GL_constant.LongConstant):
        return int(c)
    assert isinstance(c, GL_constant.StringConstant)
    return str(c, "utf-8")


GL_TRUE = load_const("GL_TRUE")
GL_TEXTURE0 = load_const("GL_TEXTURE0")
GL_TEXTURE_2D = load_const("GL_TEXTURE_2D")
GL_TEXTURE_2D_ARRAY = load_const("GL_TEXTURE_2D_ARRAY")
GL_TEXTURE_CUBE_MAP = load_const("GL_TEXTURE_CUBE_MAP")
GL_BLEND = load_const("GL_BLEND")
GL_SRC_ALPHA = load_const("GL_SRC_ALPHA")
GL_ONE_MINUS_SRC_ALPHA = load_const("GL_ONE_MINUS_SRC_ALPHA")
GL_ONE = load_const("GL_ONE")
GL_SRC1_COLOR = load_const("GL_SRC1_COLOR")
GL_FRONT_AND_BACK = load_const("GL_FRONT_AND_BACK")
GL_LINE = load_const("GL_LINE")
GL_FILL = load_const("GL_FILL")
GL_CULL_FACE = load_const("GL_CULL_FACE")
GL_BACK = load_const("GL_BACK")
GL_FRONT = load_const("GL_FRONT")
GL_PROGRAM_POINT_SIZE = load_const("GL_PROGRAM_POINT_SIZE")
GL_UNSIGNED_INT = load_const("GL_UNSIGNED_INT")
GL_RGBA = load_const("GL_RGBA")
GL_DEPTH_COMPONENT = load_const("GL_DEPTH_COMPONENT")
GL_UNSIGNED_BYTE = load_const("GL_UNSIGNED_BYTE")
GL_FLOAT = load_const("GL_FLOAT")
GL_ARRAY_BUFFER = load_const("GL_ARRAY_BUFFER")
GL_STREAM_DRAW = load_const("GL_STREAM_DRAW")

RenderFlags_DEPTH_ONLY = RenderFlags.DEPTH_ONLY
RenderFlags_SEG = RenderFlags.SEG
RenderFlags_FLIP_WIREFRAME = RenderFlags.FLIP_WIREFRAME
RenderFlags_ALL_WIREFRAME = RenderFlags.ALL_WIREFRAME
RenderFlags_ALL_SOLID = RenderFlags.ALL_SOLID
RenderFlags_SHADOWS_DIRECTIONAL = RenderFlags.SHADOWS_DIRECTIONAL
RenderFlags_SHADOWS_SPOT = RenderFlags.SHADOWS_SPOT
RenderFlags_SHADOWS_POINT = RenderFlags.SHADOWS_POINT
RenderFlags_SHADOWS_ALL = RenderFlags.SHADOWS_ALL
RenderFlags_SKIP_FLOOR = RenderFlags.SKIP_FLOOR
RenderFlags_OFFSCREEN = RenderFlags.OFFSCREEN
RenderFlags_REFLECTIVE_FLOOR = RenderFlags.REFLECTIVE_FLOOR
RenderFlags_NORMAL = RenderFlags.NORMAL
RenderFlags_NORMALS = RenderFlags.VERTEX_NORMALS | RenderFlags.FACE_NORMALS
LightType_POINT = LightType.POINT
TexFlags_EMISSIVE = TexFlags.EMISSIVE
TexFlags_BASE_COLOR = TexFlags.BASE_COLOR
LightType_SPOT = LightType.SPOT
LightType_DIRECTIONAL = LightType.DIRECTIONAL
# The flag casting the shadows of the lights of each kind, in the order of the values of 'LightType'
LIGHTS_SHADOW_FLAG = (RenderFlags_SHADOWS_DIRECTIONAL, RenderFlags_SHADOWS_POINT, RenderFlags_SHADOWS_SPOT)

# The index of each uniform in the table of locations of a program (see 'UNIFORMS_NAME')
U_M = UNIFORMS_NAME.index("M")
U_V = UNIFORMS_NAME.index("V")
U_P = UNIFORMS_NAME.index("P")
U_CAM_POS = UNIFORMS_NAME.index("cam_pos")
U_AMBIENT_LIGHT = UNIFORMS_NAME.index("ambient_light")
U_REFLECTION_MAT = UNIFORMS_NAME.index("reflection_mat")
U_DEBUG_VIEW = UNIFORMS_NAME.index("debug_view")
U_ENV_OFFSET_SCALE = UNIFORMS_NAME.index("env_offset_scale")
U_FLOOR_TEX = UNIFORMS_NAME.index("floor_tex")
U_SCREEN_SIZE = UNIFORMS_NAME.index("screen_size")
U_NORMAL_MAGNITUDE = UNIFORMS_NAME.index("normal_magnitude")
U_COLOR = UNIFORMS_NAME.index("color")
U_N_DIRECTIONAL_LIGHTS = UNIFORMS_NAME.index("n_directional_lights")
U_N_POINT_LIGHTS = UNIFORMS_NAME.index("n_point_lights")
U_N_SPOT_LIGHTS = UNIFORMS_NAME.index("n_spot_lights")
U_MATERIAL_BASE_COLOR_FACTOR = UNIFORMS_NAME.index("material.base_color_factor")
U_MATERIAL_METALLIC_FACTOR = UNIFORMS_NAME.index("material.metallic_factor")
U_MATERIAL_ROUGHNESS_FACTOR = UNIFORMS_NAME.index("material.roughness_factor")
U_MATERIAL_EMISSIVE_FACTOR = UNIFORMS_NAME.index("material.emissive_factor")
U_MATERIAL_TRANSMISSION_FACTOR = UNIFORMS_NAME.index("material.transmission_factor")
U_MATERIAL_IOR = UNIFORMS_NAME.index("material.ior")
U_MATERIAL_BASE_COLOR_LAYER = UNIFORMS_NAME.index("material.base_color_layer")
U_MATERIAL_EMISSIVE_LAYER = UNIFORMS_NAME.index("material.emissive_layer")
U_MATERIAL_TEXTURES = UNIFORMS_NAME.index("material.normal_texture")
U_LIGHT_MATRIX = UNIFORMS_NAME.index("light_matrix")
U_LIGHT_POS = UNIFORMS_NAME.index("light_pos")
U_BASE_COLOR_FACTOR = UNIFORMS_NAME.index("base_color_factor")
U_TRANSMISSION_FACTOR = UNIFORMS_NAME.index("transmission_factor")
U_IOR = UNIFORMS_NAME.index("ior")
U_BASE_COLOR_TEXTURE = UNIFORMS_NAME.index("base_color_texture")
U_BASE_COLOR_LAYER = UNIFORMS_NAME.index("base_color_layer")
U_LIGHTS = UNIFORMS_NAME.index("directional_lights[0].color")
N_LIGHT_FIELDS = len(LIGHT_FIELDS)
U_LIGHT_COLOR = LIGHT_FIELDS.index("color")
U_LIGHT_INTENSITY = LIGHT_FIELDS.index("intensity")
U_LIGHT_POSITION = LIGHT_FIELDS.index("position")
U_LIGHT_DIRECTION = LIGHT_FIELDS.index("direction")
U_LIGHT_ANGLE_SCALE = LIGHT_FIELDS.index("light_angle_scale")
U_LIGHT_ANGLE_OFFSET = LIGHT_FIELDS.index("light_angle_offset")
U_LIGHT_SHADOW_MAP = LIGHT_FIELDS.index("shadow_map")
U_LIGHT_TRANSMITTANCE_MAP = LIGHT_FIELDS.index("transmittance_map")
U_LIGHT_LIGHT_MATRIX = LIGHT_FIELDS.index("light_matrix")


@nb.jit(nopython=True, cache=True)
def set_uniform_matrix_4fv(loc, value, gl):
    gl.glUniformMatrix4fv(loc, 1, GL_TRUE, address_to_ptr(value.ctypes.data))


@nb.jit(nopython=True, cache=True)
def set_uniform_3fv(loc, value, gl):
    gl.glUniform3fv(loc, 1, address_to_ptr(value.ctypes.data))


@nb.jit(nopython=True, cache=True)
def set_uniform_4fv(loc, value, gl):
    gl.glUniform4fv(loc, 1, address_to_ptr(value.ctypes.data))


@nb.jit(nopython=True, cache=True)
def set_texture_layers(uniforms_loc, tex_flags, layers, gl):
    """Set the layers that the base color (0) and the emissive (1) texture arrays of a material are sampled at."""
    if tex_flags & (TexFlags_BASE_COLOR << 8):
        gl.glUniform1f(uniforms_loc[U_MATERIAL_BASE_COLOR_LAYER], layers[0])
    if tex_flags & (TexFlags_EMISSIVE << 8):
        gl.glUniform1f(uniforms_loc[U_MATERIAL_EMISSIVE_LAYER], layers[1])


@nb.jit(nopython=True, cache=True)
def bind_lighting(
    uniforms_loc,
    flags,
    lights_type,
    lights_color,
    lights_intensity,
    lights_pos,
    lights_dir,
    lights_angle_scale,
    lights_angle_offset,
    shadow_map,
    transmittance_map,
    light_matrix,
    ambient_light,
    placeholder_tex,
    gl,
):
    set_uniform_3fv(uniforms_loc[U_AMBIENT_LIGHT], ambient_light, gl)
    # The number of lights of each kind set so far, which is the slot of the next one in the array of its kind
    n_lights = np.zeros(len(LIGHTS_SHADOW_FLAG), np.int32)
    active_texture = 0
    depth_tex = -1
    cube_tex = -1
    transmittance_tex = -1
    transmittance_cube_tex = -1

    for i in range(len(lights_type)):
        light_type = lights_type[i]
        u = U_LIGHTS + (light_type * MAX_N_LIGHTS + n_lights[light_type]) * N_LIGHT_FIELDS
        n_lights[light_type] += 1
        if light_type != LightType_DIRECTIONAL:
            set_uniform_3fv(uniforms_loc[u + U_LIGHT_POSITION], lights_pos[i], gl)
        if light_type != LightType_POINT:
            set_uniform_3fv(uniforms_loc[u + U_LIGHT_DIRECTION], lights_dir[i], gl)
        if light_type == LightType_SPOT:
            gl.glUniform1f(uniforms_loc[u + U_LIGHT_ANGLE_SCALE], lights_angle_scale[i])
            gl.glUniform1f(uniforms_loc[u + U_LIGHT_ANGLE_OFFSET], lights_angle_offset[i])
        set_uniform_3fv(uniforms_loc[u + U_LIGHT_COLOR], lights_color[i], gl)
        gl.glUniform1f(uniforms_loc[u + U_LIGHT_INTENSITY], lights_intensity[i])
        if flags & LIGHTS_SHADOW_FLAG[light_type]:
            # Unit 0 holds the material textures (see the comment below)
            if active_texture == 0:
                active_texture += 1
            tex_type = GL_TEXTURE_CUBE_MAP if light_type == LightType_POINT else GL_TEXTURE_2D
            gl.glActiveTexture(GL_TEXTURE0 + active_texture)
            gl.glBindTexture(tex_type, shadow_map[i])
            gl.glUniform1i(uniforms_loc[u + U_LIGHT_SHADOW_MAP], active_texture)
            gl.glActiveTexture(GL_TEXTURE0 + active_texture + 1)
            gl.glBindTexture(tex_type, transmittance_map[i])
            gl.glUniform1i(uniforms_loc[u + U_LIGHT_TRANSMITTANCE_MAP], active_texture + 1)
            if light_type == LightType_POINT:
                cube_tex, transmittance_cube_tex = active_texture, active_texture + 1
            else:
                set_uniform_matrix_4fv(uniforms_loc[u + U_LIGHT_LIGHT_MATRIX], light_matrix[i, 0], gl)
                depth_tex, transmittance_tex = active_texture, active_texture + 1
            active_texture += 2

    gl.glUniform1i(uniforms_loc[U_N_DIRECTIONAL_LIGHTS], n_lights[LightType_DIRECTIONAL])
    gl.glUniform1i(uniforms_loc[U_N_SPOT_LIGHTS], n_lights[LightType_SPOT])
    gl.glUniform1i(uniforms_loc[U_N_POINT_LIGHTS], n_lights[LightType_POINT])

    # A draw fails when sampler uniforms of two types point at the same texture unit, and every sampler left unset
    # points at unit 0 along with the material textures (sampler2D). So the shadow samplers (sampler2DShadow,
    # samplerCube) take units of their own, and the light slots beyond the lights present point at one of those. A
    # unit that no light fills holds a placeholder of its type, since a sampler reading no complete texture makes the
    # draw invalid on some drivers.
    if flags & (RenderFlags_SHADOWS_DIRECTIONAL | RenderFlags_SHADOWS_SPOT) and depth_tex < 0:
        if active_texture == 0:
            active_texture += 1
        depth_tex = active_texture
        transmittance_tex = active_texture + 1
        active_texture += 2
        gl.glActiveTexture(GL_TEXTURE0 + depth_tex)
        gl.glBindTexture(GL_TEXTURE_2D, placeholder_tex[1])
        gl.glActiveTexture(GL_TEXTURE0 + transmittance_tex)
        gl.glBindTexture(GL_TEXTURE_2D, placeholder_tex[0])
    if flags & RenderFlags_SHADOWS_POINT and cube_tex < 0:
        if active_texture == 0:
            active_texture += 1
        cube_tex = active_texture
        transmittance_cube_tex = active_texture + 1
        active_texture += 2
        gl.glActiveTexture(GL_TEXTURE0 + cube_tex)
        gl.glBindTexture(GL_TEXTURE_CUBE_MAP, placeholder_tex[2])
        gl.glActiveTexture(GL_TEXTURE0 + transmittance_cube_tex)
        gl.glBindTexture(GL_TEXTURE_CUBE_MAP, placeholder_tex[3])
    for light_type in range(len(LIGHTS_SHADOW_FLAG)):
        if flags & LIGHTS_SHADOW_FLAG[light_type]:
            is_point = light_type == LightType_POINT
            for i in range(n_lights[light_type], max(n_lights[light_type], 1)):
                u = U_LIGHTS + (light_type * MAX_N_LIGHTS + i) * N_LIGHT_FIELDS
                gl.glUniform1i(uniforms_loc[u + U_LIGHT_SHADOW_MAP], cube_tex if is_point else depth_tex)
                gl.glUniform1i(
                    uniforms_loc[u + U_LIGHT_TRANSMITTANCE_MAP],
                    transmittance_cube_tex if is_point else transmittance_tex,
                )

    return active_texture


@nb.jit(nopython=True, cache=True)
def compute_bounds(pose, instance_bounds, grid_offset_bounds, is_floor, is_marker, corners, scene_bounds):
    """Compute the world corners of the bounds of each primitive, and the bounds of the scene.

    The grid pass moves an instance by its offset after the node pose (see 'mesh.vert'), so its bounds widen the world
    bounds by the range of the offsets. The scene bounds follow 'Scene.bounds': a floor counts by its centre alone, and
    a marker for nothing.

    Parameters
    ----------
    pose : np.ndarray
        The pose of the node of each primitive.
    instance_bounds : np.ndarray
        The bounds the instances of each primitive cover in the frame of its node.
    grid_offset_bounds : np.ndarray
        The range of the offsets the grid pass moves the instances of each primitive by.
    is_floor : np.ndarray
        Whether each primitive is a floor.
    is_marker : np.ndarray
        Whether each primitive is a marker.
    corners : np.ndarray
        The output corners of each primitive, for a pass drawing one environment (0) and for the grid pass (1).
    scene_bounds : np.ndarray
        The output lower and upper corners of the scene, for a pass drawing one environment (0) and for the grid pass
        (1).

    Returns
    -------
    is_bounded : bool
        Whether some primitive bounds the scene, without which 'scene_bounds' is left undefined.
    """
    is_bounded = False
    scene_bounds[:, 0] = np.inf
    scene_bounds[:, 1] = -np.inf
    grid_box = np.empty((2, 3), corners.dtype)
    centre = np.empty(3, corners.dtype)
    for i in range(len(pose)):
        grid_box[0] = np.inf
        grid_box[1] = -np.inf
        for c in range(8):
            for a in range(3):
                value = pose[i, a, 3]
                for b in range(3):
                    i_bound = CORNER_INDICES[c, b]
                    value += pose[i, a, b] * instance_bounds[i, i_bound // 3, i_bound % 3]
                corners[0, i, c, a] = value
                grid_box[0, a] = min(grid_box[0, a], value)
                grid_box[1, a] = max(grid_box[1, a], value)
        for c in range(8):
            for a in range(3):
                i_corner = CORNER_INDICES[c, a]
                corners[1, i, c, a] = grid_box[i_corner // 3, a] + grid_offset_bounds[i, i_corner // 3, a]
        if is_marker[i]:
            continue
        is_bounded = True
        for i_pass in range(2):
            if is_floor[i]:
                centre[:] = 0.0
                for c in range(8):
                    centre += corners[i_pass, i, c]
                centre /= 8
                for a in range(3):
                    scene_bounds[i_pass, 0, a] = min(scene_bounds[i_pass, 0, a], centre[a])
                    scene_bounds[i_pass, 1, a] = max(scene_bounds[i_pass, 1, a], centre[a])
            else:
                for c in range(8):
                    for a in range(3):
                        scene_bounds[i_pass, 0, a] = min(scene_bounds[i_pass, 0, a], corners[i_pass, i, c, a])
                        scene_bounds[i_pass, 1, a] = max(scene_bounds[i_pass, 1, a], corners[i_pass, i, c, a])
    return is_bounded


@nb.jit(nopython=True, cache=True)
def is_outside_view(corners, view_proj):
    """Tell whether the box of each primitive lies outside the view volume of a camera.

    A box lies outside when its 8 corners lie outside one of the planes of the volume. A point inside has clip
    coordinates with -w <= x, y, z <= w.

    Parameters
    ----------
    corners : np.ndarray
        The world corners of the box of each primitive.
    view_proj : np.ndarray
        The view-projection matrix of the camera.
    """
    is_outside = np.zeros(len(corners), np.bool_)
    clip = np.empty(4, view_proj.dtype)
    n_above = np.empty(3, np.int32)
    n_below = np.empty(3, np.int32)
    for i in range(len(corners)):
        n_above[:] = 0
        n_below[:] = 0
        for c in range(8):
            for a in range(4):
                clip[a] = view_proj[a, 3]
                for b in range(3):
                    clip[a] += view_proj[a, b] * corners[i, c, b]
            for a in range(3):
                n_above[a] += clip[a] > clip[3]
                n_below[a] += clip[a] < -clip[3]
        is_outside[i] = (n_above == 8).any() or (n_below == 8).any()
    return is_outside


@nb.extending.intrinsic
def address_to_ptr(typingctx, src):
    """returns a void pointer from a given memory address"""
    sig = nb.core.types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], nb.core.cgutils.voidptr_t)

    return sig, codegen


def is_env_pass(flags):
    """Tell whether a pass draws one environment where it is, rather than every environment side by side on the grid.

    A camera bound to an environment draws it alone. The viewer window draws the grid whatever the separation setting,
    which picks how its cameras render (see 'JITRenderer.env_offset_buffer').
    """
    return bool(flags & RenderFlags.ENV_SEPARATE and flags & RenderFlags.OFFSCREEN)


class JITRenderer:
    def __init__(self, scene, envs_offset):
        self.gl = None
        self._forward_pass = None
        self._shadow_mapping_pass = None
        self._read_depth_buf = None
        self._read_color_buf = None
        self._update_buffer_fn = None
        self._buffer = dict()
        self._scene = scene
        # The kind of view the next passes draw, whose meshes alone they draw (see 'Mesh.views'), which the caller of a
        # render sets.
        self.view = ViewFlags.NORMAL
        # Offset of each rendered environment in the pass drawing them side by side. Instance k of an
        # environment-instanced primitive (see Primitive.is_env_instanced) reads row k through an instance attribute
        # fed by 'env_offset_buffer', so the poses stay the true ones and a pass rendering one environment at a time
        # leaves the offsets out by scaling the attribute to zero.
        self.envs_offset = np.ascontiguousarray(envs_offset, dtype=np.float32)
        self.env_offset_buffer = None
        # The GL buffer of the transforms of the instances of every primitive (see 'instances_pose')
        self.instances_buffer = None
        # Scene revision the node poses and the bounds were last gathered at, and nodes revision the node poses were
        # last read at (see 'Scene.revision' and 'Scene.nodes_revision')
        self._scene_revision = None
        self._nodes_revision = None
        # (scene revision, grid pass) the light setup was last done for, and whether it found every shadow texture in
        # the context. See 'set_lighting'.
        self._lights_revision = None
        self._is_lighting_ready = False
        # Meshes and textures held in the GL context, and the (scene revision, shadow flags) they were last synced for.
        # The context is shared by every renderer of the scene, so the bookkeeping lives here rather than per renderer.
        self._meshes = set()
        self._mesh_textures = set()
        self._shadow_textures = set()
        self._placeholder_textures = None
        self.placeholder_tex = None
        self._context_revision = None
        self.set_primitive([], [])
        self.set_light()
        self.reflection_mat = np.identity(4, np.float32)

    def _update_instances(self):
        """Refresh what follows the instance transforms: the centre of each placed primitive per environment (see
        'set_primitive') and the bounds its instances cover."""
        for idx, n_instances, rows in self._placed_groups:
            poses = self.instances_pose[rows].swapaxes(-1, -2)
            centres = np.einsum("kenj,kj->ken", poses[:, :, :3, :3], self.centroid_geom[idx]) + poses[:, :, :3, 3]
            if n_instances == 1:
                # A lone instance places the primitive in every environment, so its centre is the centre for all of them
                self.centroid_local[idx] = centres
            else:
                self.centroid_local[idx, :n_instances] = centres
            self.instance_bounds[idx] = transform_box(self.bounds_geom[idx], poses)

    def update(self):
        """Gather the primitives and the poses of the scene that changed since the last call."""
        scene = self._scene
        if scene.has_meshes_changed:
            node_list, primitive_list = [], []
            for node in scene.sorted_mesh_nodes():
                for primitive in node.mesh.primitives:
                    node_list.append(node)
                    primitive_list.append(primitive)
            self.set_primitive(node_list, primitive_list)
            scene.has_meshes_changed = False
        elif self._scene_revision != scene.revision:
            # A node moves through Scene.set_pose, which advances the nodes revision. Its own transform setters leave it
            # alone.
            if self._nodes_revision != scene.nodes_revision:
                for i, node in enumerate(self.node_list):
                    self.pose[i] = scene.get_pose(node)
            # Instances move with the simulation, so where they place their primitive has to be refreshed as well
            self._update_instances()
        if self._scene_revision != scene.revision:
            # A scene of markers keeps the bounds of 'Scene.bounds'. The light setup holds the bounds it was given, so
            # every update writes fresh ones.
            scene_bounds = np.empty((2, 2, 3), np.float32)
            is_bounded = compute_bounds(
                self.pose,
                self.instance_bounds,
                self.grid_offset_bounds,
                self.is_floor,
                self.is_marker,
                self.corners,
                scene_bounds,
            )
            if is_bounded:
                self.scene_bounds = {False: scene_bounds[0], True: scene_bounds[1]}
        self._scene_revision = scene.revision
        self._nodes_revision = scene.nodes_revision

    def set_lighting(self, is_grid):
        """Set up the lights for a pass, and tell whether every shadow texture was found in the context.

        The setup follows the scene: the light poses and the bounds the directional shadow frustum is fitted to, which
        the grid pass has its own of. It stays unsettled while a shadow texture is missing from the context, so the next
        render tries again.

        Parameters
        ----------
        is_grid : bool
            Whether the pass draws the environments side by side (see 'is_env_pass').

        Returns
        -------
        is_lighting_ready : bool
            Whether every shadow texture was found in the context.
        """
        scene = self._scene
        if self.scene_bounds[is_grid] is not None:
            scene.bounds = self.scene_bounds[is_grid]
        lights_revision = (scene.revision, is_grid)
        if self._lights_revision != lights_revision:
            self._is_lighting_ready = self.set_light()
            self._lights_revision = lights_revision if self._is_lighting_ready else None
        return self._is_lighting_ready

    def _add_primitive_to_context(self, primitive):
        """Upload a primitive to the GL context, one instanced per environment or placed in one reading the offsets."""
        if not (primitive.is_env_instanced or primitive.env_idx is not None):
            primitive._add_to_context()
            return
        if self.env_offset_buffer is None:
            self.env_offset_buffer = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.env_offset_buffer)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.envs_offset.nbytes, self.envs_offset, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        primitive._add_to_context(self.env_offset_buffer)

    def update_context(self, flags):
        """Add to the GL context the meshes and textures the scene holds and free the ones it dropped.

        Runs once per nodes revision and set of shadow flags, which decide the lights given a shadow texture. The
        meshes, their textures and the lights change with the nodes alone (see 'Scene.nodes_revision').

        Parameters
        ----------
        flags : int
            The render flags of the pass (see RenderFlags).
        """
        scene = self._scene
        shadow_flags = flags & RenderFlags.SHADOWS_ALL
        if self._context_revision == (scene.nodes_revision, shadow_flags):
            return
        self._context_revision = (scene.nodes_revision, shadow_flags)

        # Get existing and new meshes
        scene_meshes_new = scene.meshes.copy()
        scene_meshes_old = self._meshes

        # Remove from context old meshes that are now irrelevant
        for mesh in scene_meshes_old - scene_meshes_new:
            for p in mesh.primitives:
                p.delete()

        # Update set of meshes right away, so that the context can be cleaned up correctly in case of failure
        self._meshes = scene_meshes_new

        # Add new meshes to context
        for mesh in scene_meshes_new - scene_meshes_old:
            for p in mesh.primitives:
                self._add_primitive_to_context(p)

        # Update mesh textures
        mesh_textures = set()
        for m in scene_meshes_new:
            for p in m.primitives:
                mesh_textures |= p.material.textures

        # Add new textures to context
        for texture in mesh_textures - self._mesh_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._mesh_textures - mesh_textures:
            texture.delete()

        self._mesh_textures = mesh_textures.copy()

        shadow_textures = set()
        for light in scene.lights:
            if flags & light.shadow_flag and light.shadow_texture is None:
                light.generate_shadow_texture()
            if light.shadow_texture is not None:
                shadow_textures.add(light.shadow_texture)
                shadow_textures.add(light.transmittance_texture)

        # Add new textures to context
        for texture in shadow_textures - self._shadow_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._shadow_textures - shadow_textures:
            texture.delete()

        self._shadow_textures = shadow_textures.copy()

    def delete(self):
        """Free the GL resources of the meshes and textures held in the context."""
        for mesh in self._meshes:
            for p in mesh.primitives:
                try:
                    p.delete()
                except (GL.error.GLError, GL.error.NullFunctionError):
                    pass
        self._meshes.clear()

        for texture in (*self._mesh_textures, *self._shadow_textures, *(self._placeholder_textures or ())):
            try:
                texture.delete()
            except (GL.error.GLError, GL.error.NullFunctionError):
                pass
        self._mesh_textures.clear()
        self._shadow_textures.clear()
        self._placeholder_textures = None
        self._context_revision = None
        for buffer in (self.env_offset_buffer, self.instances_buffer):
            if buffer is not None:
                try:
                    GL.glDeleteBuffers(1, [buffer])
                except (GL.error.GLError, GL.error.NullFunctionError):
                    pass
        self.env_offset_buffer = None
        self.instances_buffer = None

    def set_light(self):
        """Gather the lights of the scene, and tell whether every shadow texture was found in the context."""
        scene = self._scene
        light_nodes = scene.light_nodes
        n = len(light_nodes)
        # The shaders bind the maps of every light along with the material textures, within the units of every driver
        assert n <= MAX_N_LIGHTS

        self.lights_type = np.zeros(n, np.int32)
        self.lights_color = np.zeros((n, 3), np.float32)
        self.lights_intensity = np.zeros(n, np.float32)
        self.lights_pos = np.zeros((n, 3), np.float32)
        self.lights_dir = np.zeros((n, 3), np.float32)
        # The falloff of a spot light across its cone, as glTF defines it (see 'spot_lights' in mesh.frag)
        self.lights_angle_scale = np.zeros(n, np.float32)
        self.lights_angle_offset = np.zeros(n, np.float32)
        self.shadow_map = np.zeros(n, np.int32)
        self.transmittance_map = np.zeros(n, np.int32)
        # The view-projection matrix of the shadow camera of each light, one per face of the cube map of a point light
        self.lights_matrix = np.zeros((n, 6, 4, 4), np.float32)
        self.lights_idx_by_node = {}

        self.ambient_light = np.array(scene.ambient_light, np.float32)

        is_textures_ready = True

        for i, node in enumerate(light_nodes):
            light = node.light
            pose = scene.get_pose(node)
            self.lights_idx_by_node[node] = i
            self.lights_color[i] = light.color
            self.lights_intensity[i] = light.intensity
            self.lights_pos[i] = pose[:3, 3]
            self.lights_dir[i] = -pose[:3, 2]

            if light.shadow_texture is not None:
                if light.shadow_texture._in_context() and light.transmittance_texture._in_context():
                    self.shadow_map[i] = light.shadow_texture._texid
                    self.transmittance_map[i] = light.transmittance_texture._texid
                else:
                    is_textures_ready = False

            self.lights_type[i] = light.light_type
            if isinstance(light, PointLight):
                # Each face of the cube map sees a right angle of the scene from the light
                P = PerspectiveCamera(yfov=np.pi / 2, znear=0.1, zfar=25.0, aspect_ratio=1.0).get_projection_matrix()
                faces_view = np.tile(np.eye(4), (6, 1, 1))
                faces_view[:, :3, :3] = CUBE_MAP_FACES_ROT
                faces_view[:, :3, 3] = -CUBE_MAP_FACES_ROT @ self.lights_pos[i]
                self.lights_matrix[i] = P @ faces_view
                continue
            scene_scale = scene.scale if scene.scale > 1e-6 else 1.0
            if isinstance(light, SpotLight):
                cos_outer = np.cos(light.cone_angle)
                self.lights_angle_scale[i] = 1.0 / max(0.001, 1.0 - cos_outer)
                self.lights_angle_offset[i] = -cos_outer * self.lights_angle_scale[i]
                camera = PerspectiveCamera(
                    yfov=min(2 * light.cone_angle + np.pi / 16.0, np.pi),
                    znear=0.01 * scene_scale,
                    zfar=10 * scene_scale,
                    aspect_ratio=1.0,
                )
            else:
                # The shadow camera of a directional light looks at the scene from outside its bounds. Its map spans the
                # bounding sphere of the scene, whose radius is half the diagonal. Every shadow lies inside it, since a
                # shadow falls where the light meets a caster first, and every caster lies inside the sphere.
                pose = pose.copy()
                pose[:3, 3] = scene.centroid - self.lights_dir[i] * scene_scale
                radius = 0.5 * scene_scale
                camera = OrthographicCamera(xmag=radius, ymag=radius, znear=0.01 * scene_scale, zfar=10 * scene_scale)
            self.lights_matrix[i, 0] = camera.get_projection_matrix() @ np.linalg.inv(pose)

        return is_textures_ready

    def set_primitive(self, node_list, primitive_list):
        self.node_list = node_list
        self.primitive_list = primitive_list

        n = len(primitive_list)
        self.vao_id = np.zeros(n, np.int32)
        # The programs of each pass, by its render flags and program flags (see 'load_programs')
        self.programs = {}
        self.pose = np.zeros((n, 4, 4), np.float32)
        # The textures of each slot of the material, in the order of the bits of 'TexFlags', and the flags of the slots
        # filled, the bit of slot i shifted by 8 marking a texture array
        self.textures = np.zeros((n, 5), np.int32)
        self.tex_flags = np.zeros(n, np.int32)
        # Row i of texture_layers holds, for each rendered environment, the layer of the base color (0) and emissive (1)
        # texture arrays that a layered primitive samples (see 'Primitive.texture_layers').
        # texture_layers_idx maps each primitive to its row, or -1 for a primitive sampling no texture array.
        is_layered = np.array([primitive.texture_layers is not None for primitive in primitive_list], np.bool_)
        self.texture_layers_idx = np.full(n, -1, np.int32)
        self.texture_layers_idx[is_layered] = np.arange(np.count_nonzero(is_layered))
        self.texture_layers = np.zeros((np.count_nonzero(is_layered), len(self.envs_offset), 2), np.float32)
        self.base_color_factor = np.zeros((n, 4), np.float32)
        self.metallic_factor = np.zeros(n, np.float32)
        self.roughness_factor = np.zeros(n, np.float32)
        self.emissive_factor = np.zeros((n, 3), np.float32)
        self.transmission_factor = np.zeros(n, np.float32)
        self.ior = np.zeros(n, np.float32)
        self.is_blended = np.zeros(n, np.bool_)
        self.is_wireframe = np.zeros(n, np.bool_)
        # The one floor the reflection is set up for (see 'reflection_mat')
        self.is_reflective_floor = np.zeros(n, np.bool_)
        self.is_transparent = np.zeros(n, np.bool_)
        self.is_marker = np.zeros(n, np.bool_)
        # A heterogeneous variant, drawn in the environments it is active in alone (see 'env_active')
        self.is_env_filtered = np.zeros(n, np.bool_)
        self.is_transmissive = np.zeros(n, np.bool_)
        self.mode = np.zeros(n, np.int32)
        self.n_instances = np.zeros(n, np.int32)
        # The visibility of each primitive in each environment, which an env filtered primitive alone reads (see
        # 'is_env_filtered'). Its environments are those of the instance poses.
        n_env = max((len(p.poses) for p in primitive_list if p.poses is not None), default=1)
        self.env_active = np.ones((n, n_env), np.bool_)
        # The number of indices of each primitive, or minus its number of positions for a primitive without indices
        self.n_indices = np.zeros(n, np.int32)
        # The views drawing each primitive (see 'Mesh.views')
        self.views = np.zeros(n, np.int8)
        # GL buffer of the instance tints of a primitive carrying them, 0 otherwise (see Primitive.inst_tints)
        self.inst_tint_buffer_id = np.zeros(n, np.int32)
        self.inst_attr_start = np.zeros(n, np.int32)
        # Centre of each primitive in the frame of its node, per environment. Sorting on it rather than on the node
        # origin places a mesh where its geometry actually is, which is what decides whether it occludes another one.
        # Sized by the primitives drawn one instance per environment, since those are the only ones whose centre
        # depends on the environment. 'n_env' spans every instance of every primitive, particles included, which for
        # three coordinates apiece would be far more than what tracking a centre per environment needs.
        # An instance set may be empty, and such a primitive is still drawn, so it may not shrink the axis to nothing
        n_env_instanced = max(
            (len(p.poses) for p in primitive_list if p.poses is not None and p.is_env_instanced and len(p.poses)),
            default=1,
        )
        self.centroid_local = np.zeros((n, n_env_instanced, 3), np.float32)
        # Centre of the bare geometry, which the instance transforms are applied to whenever they move
        self.centroid_geom = np.zeros((n, 3), np.float32)
        self.is_instance_placed = np.zeros(n, np.bool_)
        # Per primitive: the bounds of the bare geometry, the bounds its instances cover, the range of the offsets the
        # grid pass moves them by, and whether it is a floor. The scene bounds count a floor by its centre alone
        # (see 'update').
        self.bounds_geom = np.zeros((n, 2, 3), np.float32)
        self.instance_bounds = np.zeros((n, 2, 3), np.float32)
        self.grid_offset_bounds = np.zeros((n, 2, 3), np.float32)
        self.is_floor = np.zeros(n, np.bool_)
        self.is_env_instanced = np.zeros(n, np.bool_)
        # Index among the rendered environments of the one a primitive is placed in, -1 for none (see Primitive.env_idx)
        self.env_row = np.full(n, -1, np.int32)
        # Axis-aligned bounds of what is drawn, for a pass rendering one environment at a time (False) and for the
        # grid pass (True), None until a bounded primitive exists (see 'update')
        self.scene_bounds = {False: None, True: None}
        # World corners of the bounds of each primitive, for a pass rendering one environment at a time (0) and for the
        # grid pass (1) (see 'update')
        self.corners = np.zeros((2, n, 8, 3), np.float32)
        self._primitive_index = {primitive: i for i, primitive in enumerate(primitive_list)}

        has_floor = False

        for i, primitive in enumerate(primitive_list):
            if primitive._vaid is None:
                self._add_primitive_to_context(primitive)
            self.vao_id[i] = primitive._vaid
            self.pose[i] = self._scene.get_pose(node_list[i])
            # An instance transform is what places a primitive, and it moves with the simulation, so the centre has to
            # go through it and be refreshed. That holds for the lone instance shared by every environment just as much
            # as for one instance per environment. A cloud of instances sharing a primitive - particles - has no single
            # centre to be sorted on, so it keeps the centre of the mesh itself.
            self.is_instance_placed[i] = primitive.poses is not None and (
                len(primitive.poses) == 1 or primitive.is_env_instanced
            )
            # A primitive without a single vertex keeps the zero centre and bounds it was allocated with, rather than
            # reducing over an empty axis
            positions = primitive.positions
            if len(positions):
                self.bounds_geom[i, 0] = positions.min(axis=0)
                self.bounds_geom[i, 1] = positions.max(axis=0)
                self.centroid_geom[i] = 0.5 * (self.bounds_geom[i, 0] + self.bounds_geom[i, 1])
            self.is_floor[i] = primitive.is_floor
            self.is_env_instanced[i] = primitive.is_env_instanced
            if primitive.env_idx is not None:
                self.env_row[i] = primitive.env_idx
            if not self.is_instance_placed[i]:
                self.centroid_local[i] = primitive.centroid
            # The bounds the instances cover, refreshed with the instance transforms afterwards (see 'update_poses')
            if primitive.poses is not None and len(primitive.poses):
                poses = primitive.poses
            else:
                poses = np.eye(4, dtype=np.float32)[np.newaxis]
            self.instance_bounds[i] = transform_box(self.bounds_geom[i], poses)

            material = primitive.material
            for i_slot, texture in enumerate(material.slots_texture):
                if texture is not None:
                    self.textures[i, i_slot] = texture._texid
                    self.tex_flags[i] |= 1 << i_slot
                    if texture.tex_type == GL_TEXTURE_2D_ARRAY:
                        self.tex_flags[i] |= 1 << (i_slot + 8)

            self.base_color_factor[i] = material.base_color_factor
            self.metallic_factor[i] = material.metallic_factor
            self.roughness_factor[i] = material.roughness_factor
            self.emissive_factor[i] = material.emissive_factor
            self.transmission_factor[i] = material.transmission_factor
            self.ior[i] = material.ior

            self.is_blended[i] = material.is_blended
            self.is_wireframe[i] = material.wireframe
            self.is_reflective_floor[i] = primitive.is_floor and not has_floor
            self.is_transparent[i] = node_list[i].mesh.is_transparent
            self.is_marker[i] = node_list[i].mesh.is_marker
            self.is_transmissive[i] = material.is_transmissive
            self.views[i] = node_list[i].mesh.views
            # A pass drawing one environment draws the instance of that environment of a per-environment instanced
            # primitive, and every instance of any other, which the culling below keeps to its environment
            if primitive.is_env_instanced and not primitive.envs.all():
                self.is_env_filtered[i] = True
                self.env_active[i, : len(primitive.envs)] = primitive.envs

            if primitive.is_floor:
                has_floor = True

            self.mode[i] = primitive.mode
            self.n_instances[i] = len(primitive.poses) if primitive.poses is not None else 1
            self.n_indices[i] = primitive.indices.size if primitive.indices is not None else -len(primitive.positions)
            self.inst_tint_buffer_id[i] = primitive.buffers_id.get("inst_tint", 0)
            self.inst_attr_start[i] = primitive._inst_attr_start

        # The transforms of the instances of every primitive, those of primitive i in rows 'instances_start[i]' to
        # 'instances_start[i + 1]', transposed as the shaders read them (see 'INST_M_LOC' in renderer.py). They move
        # together in one GL buffer, which the vertex array of each primitive reads at its rows, so that the poses that
        # changed upload at once (see 'update_poses').
        self.instances_start = np.zeros(n + 1, np.int64)
        np.cumsum(self.n_instances, out=self.instances_start[1:])
        self.instances_pose = np.empty((self.instances_start[-1], 4, 4), np.float32)
        for i, primitive in enumerate(primitive_list):
            poses = np.eye(4)[np.newaxis] if primitive.poses is None else primitive.poses
            self.instances_pose[self.instances_start[i] : self.instances_start[i + 1]] = poses.swapaxes(1, 2)
        # The primitives whose transforms changed since the last upload
        self.is_pose_changed = np.ones(n, np.bool_)
        if self.instances_buffer is not None:
            GL.glDeleteBuffers(1, [self.instances_buffer])
            self.instances_buffer = None
        if len(self.instances_pose):
            self.instances_buffer = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.instances_buffer)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.instances_pose.nbytes, None, GL.GL_STREAM_DRAW)
            # Each transform spans 4 attribute locations, one column apiece
            for i in range(n):
                GL.glBindVertexArray(self.vao_id[i])
                for j in range(4):
                    location = self.inst_attr_start[i] + j
                    offset = ctypes.c_void_p(16 * FLOAT_SZ * int(self.instances_start[i]) + 4 * FLOAT_SZ * j)
                    GL.glEnableVertexAttribArray(location)
                    GL.glVertexAttribPointer(location, 4, GL.GL_FLOAT, GL.GL_FALSE, 16 * FLOAT_SZ, offset)
                    GL.glVertexAttribDivisor(location, 1)
            GL.glBindVertexArray(0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        # The grid pass moves an instance placed in one environment by the offset of that environment, and instance k of
        # an environment-instanced primitive by the offset of environment k (see 'mesh.vert')
        is_env_placed = self.env_row >= 0
        self.grid_offset_bounds[is_env_placed] = self.envs_offset[self.env_row[is_env_placed], np.newaxis]
        self.grid_offset_bounds[self.is_env_instanced] = (self.envs_offset.min(axis=0), self.envs_offset.max(axis=0))

        self.update_texture_layers([primitive_list[i] for i in np.flatnonzero(is_layered)])

        # Gate the per-env culling to scenes with heterogeneous variants or primitives standing in one environment
        self._has_env_filtered = bool(self.is_env_filtered.any() or (self.env_row >= 0).any())

        # The placed primitives are refreshed together, in groups of equal instance count so that their poses stack.
        # Their instance count is fixed by the buffers sized on it. A group without an instance has nothing to reduce.
        n_instances_placed = {}
        for i in np.flatnonzero(self.is_instance_placed):
            n_instances_placed.setdefault(len(primitive_list[i].poses), []).append(i)
        self._placed_groups = [
            PlacedGroup(np.array(idx), n_instances, self.instances_start[idx, np.newaxis] + np.arange(n_instances))
            for n_instances, idx in n_instances_placed.items()
            if n_instances > 0
        ]
        self._update_instances()

    @contextlib.contextmanager
    def _culled(self, env_idx, view_proj=None, is_grid=False):
        """Zero the index count of the primitives that a pass leaves out while it runs, so that their draw is empty.

        These are the primitives absent from the environment drawn (heterogeneous variants, and those standing in
        another environment), those that the kind of view drawn leaves out (see 'view'), and, given the view-projection
        matrix of a camera, those whose bounds lie outside its view volume.
        """
        is_culled = (self.views & self.view) == 0
        if view_proj is not None:
            is_culled |= is_outside_view(self.corners[int(is_grid)], view_proj)
        if env_idx >= 0 and self._has_env_filtered:
            is_culled |= (self.env_row >= 0) & (self.env_row != env_idx)
            # Only the env filtered primitives carry one visibility per environment. Primitives standing in a single
            # environment (env_row) alone may leave 'env_active' narrower than the rendered environments.
            if self.is_env_filtered.any():
                is_culled[self.is_env_filtered] |= ~self.env_active[self.is_env_filtered, env_idx]
        if not is_culled.any():
            yield
            return
        n_indices_culled = self.n_indices[is_culled].copy()
        self.n_indices[is_culled] = 0
        try:
            yield
        finally:
            self.n_indices[is_culled] = n_indices_culled

    def load_programs(self, renderer, flags, program_flags):
        if (flags, program_flags) not in self.programs:
            program_ids = np.zeros_like(self.vao_id)
            uniforms_loc = np.zeros((len(self.vao_id), len(UNIFORMS_NAME)), np.int32)
            for i, primitive in enumerate(self.primitive_list):
                prim_flags = flags
                # Markers are excluded from scene bounds, so shadow maps may not cover them. Disable shadow
                # reception while keeping full lighting.
                if self.is_marker[i]:
                    prim_flags &= ~RenderFlags.SHADOWS_ALL
                # A single floor mirrors the scene, the one the reflection is set up for (see 'reflection_mat')
                if not self.is_reflective_floor[i]:
                    prim_flags &= ~RenderFlags.REFLECTIVE_FLOOR
                program = renderer._get_primitive_program(primitive, prim_flags, program_flags)
                program_ids[i] = program._program_id
                uniforms_loc[i] = program.uniforms_loc
            self.programs[(flags, program_flags)] = PassPrograms(program_ids, uniforms_loc)

    def gen_func_ptr(self):
        """Compile the passes of the renderer, once, with the OpenGL functions of the current context."""
        if self.gl is not None:
            return
        self.gl = GLWrapper()

        IS_OPENGL_42_AVAILABLE = "glDrawElementsInstancedBaseInstance" in self.gl.gl_funcs

        @nb.jit(
            nb.none(
                nb.int32,
                nb.int32,
                nb.int32,
                nb.int32,
                nb.int64,
                nb.int32,
                nb.int32,
                nb.int32,
                nb.int32,
                nb.boolean,
                self.gl.wrapper_type,
            ),
            cache=True,
        )
        def draw_instances(
            mode,
            n_indices,
            n_instances,
            k,
            instance_start,
            inst_attr_start,
            instances_buffer,
            env_offset_buffer,
            inst_tint_buffer_id,
            is_env_instanced,
            gl,
        ):
            """Draw the instance k of the primitive whose vertex array is bound, or all of them for k = -1.

            A positive index count draws the indices, a negative one that many vertices in order.
            """
            if k < 0:
                if n_indices > 0:
                    gl.glDrawElementsInstanced(mode, n_indices, GL_UNSIGNED_INT, address_to_ptr(0), n_instances)
                else:
                    gl.glDrawArraysInstanced(mode, 0, -n_indices, n_instances)
            elif IS_OPENGL_42_AVAILABLE:
                if n_indices > 0:
                    gl.glDrawElementsInstancedBaseInstance(mode, n_indices, GL_UNSIGNED_INT, address_to_ptr(0), 1, k)
                else:
                    gl.glDrawArraysInstancedBaseInstance(mode, 0, -n_indices, 1, k)
            else:
                # OpenGL 4.1 draws no instance but the first, so the instance attributes start at instance k for the
                # draw, and at the first one again afterwards, the state of the vertex array being shared by every draw
                for i_rebase in range(2):
                    k_attr = k if i_rebase == 0 else 0
                    gl.glBindBuffer(GL_ARRAY_BUFFER, instances_buffer)
                    for j in range(4):
                        offset = (instance_start + k_attr) * 64 + j * 16
                        gl.glVertexAttribPointer(inst_attr_start + j, 4, GL_FLOAT, 0, 64, address_to_ptr(offset))
                    if is_env_instanced:
                        gl.glBindBuffer(GL_ARRAY_BUFFER, env_offset_buffer)
                        gl.glVertexAttribPointer(inst_attr_start + 4, 3, GL_FLOAT, 0, 12, address_to_ptr(k_attr * 12))
                    if inst_tint_buffer_id:
                        gl.glBindBuffer(GL_ARRAY_BUFFER, inst_tint_buffer_id)
                        gl.glVertexAttribPointer(inst_attr_start + 5, 4, GL_FLOAT, 0, 16, address_to_ptr(k_attr * 16))
                    if i_rebase == 0:
                        if n_indices > 0:
                            gl.glDrawElementsInstanced(mode, n_indices, GL_UNSIGNED_INT, address_to_ptr(0), 1)
                        else:
                            gl.glDrawArraysInstanced(mode, 0, -n_indices, 1)

        @nb.jit(
            nb.none(
                nb.int32[:],
                nb.int32[:],
                nb.int32[:, :],
                nb.float32[:, :, :],
                nb.int32[:],
                nb.int32[:, :],
                nb.float32[:, :],
                nb.float32[:],
                nb.float32[:],
                nb.float32[:, :],
                nb.float32[:],
                nb.float32[:],
                nb.boolean[:],
                nb.boolean[:],
                nb.boolean[:],
                nb.boolean[:],
                nb.boolean[:],
                nb.boolean[:],
                nb.int32[:],
                nb.int32[:],
                nb.int32[:],
                nb.int32[:],
                nb.float32[:, :],
                nb.float32[:],
                nb.float32[:, :],
                nb.float32[:, :],
                nb.float32[:],
                nb.float32[:],
                nb.int32[:],
                nb.int32[:],
                nb.float32[:, :, :, :],
                nb.float32[:],
                nb.float32[:, :],
                nb.float32[:, :],
                nb.float32[:],
                nb.int32,
                nb.float32[:, :],
                nb.float32[:, :],
                nb.int32,
                nb.int32[:],
                nb.float32[:],
                nb.int32,
                nb.int64[:],
                nb.int32[:],
                nb.int32[:],
                nb.boolean[:, :],
                nb.int32[:],
                nb.float32,
                nb.int32,
                nb.boolean[:],
                nb.int32,
                nb.int32,
                nb.float32[:, :, :],
                nb.int32[:],
                nb.float32[:],
                self.gl.wrapper_type,
            ),
            cache=True,
        )
        def forward_pass(
            vao_id,
            program_id,
            uniforms_loc,
            pose,
            tex_flags,
            textures,
            base_color_factor,
            metallic_factor,
            roughness_factor,
            emissive_factor,
            transmission_factor,
            ior,
            is_blended,
            is_wireframe,
            is_reflective_floor,
            is_marker,
            is_env_filtered,
            is_transmissive,
            mode,
            n_instances,
            n_indices,
            lights_type,
            lights_color,
            lights_intensity,
            lights_pos,
            lights_dir,
            lights_angle_scale,
            lights_angle_offset,
            shadow_map,
            transmittance_map,
            light_matrix,
            ambient_light,
            mat_V,
            mat_P,
            cam_pos,
            flags,
            color_list,
            reflection_mat,
            floor_tex,
            placeholder_tex,
            screen_size,
            env_idx,
            instances_start,
            inst_tint_buffer_id,
            inst_attr_start,
            env_active,
            draw_order,
            env_offset_scale,
            debug_view,
            is_env_instanced,
            instances_buffer,
            env_offset_buffer,
            texture_layers,
            texture_layers_idx,
            normal_magnitude,
            gl,
        ):
            # A pass drawing the normals draws them alone, as lines of a single color (see vertex_normals.geom)
            is_normal_pass = flags & RenderFlags_NORMALS
            is_rgba = not (flags & RenderFlags_DEPTH_ONLY or flags & RenderFlags_SEG or is_normal_pass)
            # A pass of the color image shades the surfaces with their material and the lights
            is_shaded = is_rgba and not flags & RenderFlags_NORMAL

            # Every sampler left unset reads unit 0, which holds a placeholder until a material binds a texture there
            # (see bind_lighting). This also unbinds the mirror image that a floor pass draws into.
            gl.glActiveTexture(GL_TEXTURE0)
            gl.glBindTexture(GL_TEXTURE_2D, placeholder_tex[0])

            det_reflection = np.linalg.det(reflection_mat)
            last_pid = -1
            lighting_texture = 0

            for i_p in draw_order:
                # Only render markers on the main graphical window, while skipping plane-reflection. A primitive this
                # pass hides has no index to draw, and its state setup costs the driver as much as a draw.
                if (
                    n_indices[i_p] == 0
                    or ((is_reflective_floor[i_p] or is_marker[i_p]) and flags & RenderFlags_SKIP_FLOOR)
                    or (
                        is_marker[i_p]
                        and (not is_rgba or (flags & RenderFlags_OFFSCREEN and _DISABLE_OFFSCREEN_MARKERS))
                    )
                ):
                    continue

                pid = program_id[i_p]
                if pid != last_pid:
                    gl.glUseProgram(pid)
                    if is_shaded:
                        # The programs of the markers receive no shadow (see 'load_programs')
                        light_flags = flags
                        if is_marker[i_p]:
                            light_flags &= ~RenderFlags_SHADOWS_ALL
                        lighting_texture = bind_lighting(
                            uniforms_loc[i_p],
                            light_flags,
                            lights_type,
                            lights_color,
                            lights_intensity,
                            lights_pos,
                            lights_dir,
                            lights_angle_scale,
                            lights_angle_offset,
                            shadow_map,
                            transmittance_map,
                            light_matrix,
                            ambient_light,
                            placeholder_tex,
                            gl,
                        )
                        set_uniform_3fv(uniforms_loc[i_p, U_CAM_POS], cam_pos, gl)
                        set_uniform_matrix_4fv(uniforms_loc[i_p, U_REFLECTION_MAT], reflection_mat, gl)
                        # The instance tints show in the debug views alone (see 'Primitive.inst_tints')
                        if inst_tint_buffer_id[i_p]:
                            gl.glUniform1i(uniforms_loc[i_p, U_DEBUG_VIEW], debug_view)

                    set_uniform_matrix_4fv(uniforms_loc[i_p, U_V], mat_V, gl)
                    set_uniform_matrix_4fv(uniforms_loc[i_p, U_P], mat_P, gl)
                    gl.glUniform1f(uniforms_loc[i_p, U_ENV_OFFSET_SCALE], env_offset_scale)

                    last_pid = pid

                active_texture = lighting_texture

                # The program of a reflective floor samples the mirror image (see REFLECTIVE_FLOOR in renderer.py)
                if flags & RenderFlags_REFLECTIVE_FLOOR and is_reflective_floor[i_p]:
                    gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                    gl.glBindTexture(GL_TEXTURE_2D, floor_tex)
                    gl.glUniform1i(uniforms_loc[i_p, U_FLOOR_TEX], active_texture)
                    gl.glUniform2f(uniforms_loc[i_p, U_SCREEN_SIZE], screen_size[0], screen_size[1])
                    active_texture += 1

                set_uniform_matrix_4fv(uniforms_loc[i_p, U_M], pose[i_p], gl)
                if is_normal_pass:
                    gl.glUniform1f(uniforms_loc[i_p, U_NORMAL_MAGNITUDE], normal_magnitude[i_p])
                gl.glBindVertexArray(vao_id[i_p])

                if is_shaded:
                    tf = tex_flags[i_p]
                    # An unset sampler reads unit 0. Keep texture arrays off unit 0, since a sampler2D and a
                    # sampler2DArray reading one unit make the draw invalid.
                    if tf >> 8 and active_texture == 0:
                        active_texture = 1
                    for i in range(textures.shape[1]):
                        if tf & (1 << i):
                            gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                            is_array = tf & (1 << (i + 8))
                            gl.glBindTexture(GL_TEXTURE_2D_ARRAY if is_array else GL_TEXTURE_2D, textures[i_p, i])
                            gl.glUniform1i(uniforms_loc[i_p, U_MATERIAL_TEXTURES + i], active_texture)
                            active_texture += 1

                    set_uniform_4fv(uniforms_loc[i_p, U_MATERIAL_BASE_COLOR_FACTOR], base_color_factor[i_p], gl)
                    gl.glUniform1f(uniforms_loc[i_p, U_MATERIAL_METALLIC_FACTOR], metallic_factor[i_p])
                    gl.glUniform1f(uniforms_loc[i_p, U_MATERIAL_ROUGHNESS_FACTOR], roughness_factor[i_p])
                    set_uniform_3fv(uniforms_loc[i_p, U_MATERIAL_EMISSIVE_FACTOR], emissive_factor[i_p], gl)
                    gl.glUniform1f(uniforms_loc[i_p, U_MATERIAL_TRANSMISSION_FACTOR], transmission_factor[i_p])
                    gl.glUniform1f(uniforms_loc[i_p, U_MATERIAL_IOR], ior[i_p])

                    if is_transmissive[i_p]:
                        # A transmissive material adds the light it reflects to the background, which its second output
                        # attenuates by the light it lets through (see 'frag_transmittance' in mesh.frag)
                        gl.glEnable(GL_BLEND)
                        gl.glBlendFunc(GL_ONE, GL_SRC1_COLOR)
                    elif is_blended[i_p]:
                        # Blending is enabled where needed alone, since it makes the results of Nvidia GPUs
                        # non-deterministic
                        gl.glEnable(GL_BLEND)
                        # The alpha accumulates the coverage of the fragments, which the mirror image of a floor reads
                        # (see REFLECTIVE_FLOOR in mesh.frag)
                        gl.glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
                    else:
                        gl.glDisable(GL_BLEND)

                    is_wireframe_drawn = is_wireframe[i_p] and not flags & RenderFlags_ALL_SOLID
                    if flags & RenderFlags_FLIP_WIREFRAME:
                        is_wireframe_drawn = not is_wireframe_drawn
                    if is_wireframe_drawn or flags & RenderFlags_ALL_WIREFRAME:
                        gl.glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    else:
                        gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                else:
                    gl.glDisable(GL_BLEND)
                    gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                gl.glEnable(GL_CULL_FACE)
                gl.glCullFace(GL_BACK if det_reflection > 0 else GL_FRONT)

                gl.glDisable(GL_PROGRAM_POINT_SIZE)

                if flags & RenderFlags_SEG:
                    if color_list[i_p, 0] < -1:
                        gl.glBindVertexArray(0)
                        continue
                    set_uniform_3fv(uniforms_loc[i_p, U_COLOR], color_list[i_p], gl)

                # A texture array is sampled at the layer of the environment drawn
                i_layers = texture_layers_idx[i_p]
                is_layer_set = i_layers >= 0 and is_shaded
                if is_layer_set and env_idx >= 0:
                    set_texture_layers(uniforms_loc[i_p], tex_flags[i_p], texture_layers[i_layers, env_idx], gl)

                # A single-sided transmissive surface is drawn twice, back faces first and front faces over them, so
                # that the far side of a glass or of a body of water shows through its near side
                n_face_passes = 2 if is_rgba and is_transmissive[i_p] else 1
                for i_face_pass in range(n_face_passes):
                    if n_face_passes == 2:
                        is_front_culled = (i_face_pass == 0) == (det_reflection > 0)
                        gl.glCullFace(GL_FRONT if is_front_culled else GL_BACK)
                    if (is_env_filtered[i_p] or is_layer_set) and env_idx == -1:
                        # Combined draw-all of a heterogeneous variant: draw only the environments it is active in,
                        # otherwise the variant would be drawn in every environment's instance. A primitive sampling a
                        # texture array is drawn one environment at a time too, each at its own layer.
                        for k in range(n_instances[i_p]):
                            if not env_active[i_p, k]:
                                continue
                            if is_layer_set:
                                set_texture_layers(uniforms_loc[i_p], tex_flags[i_p], texture_layers[i_layers, k], gl)
                            draw_instances(
                                mode[i_p],
                                n_indices[i_p],
                                n_instances[i_p],
                                k,
                                instances_start[i_p],
                                inst_attr_start[i_p],
                                instances_buffer,
                                env_offset_buffer,
                                inst_tint_buffer_id[i_p],
                                is_env_instanced[i_p],
                                gl,
                            )
                    else:
                        draw_instances(
                            mode[i_p],
                            n_indices[i_p],
                            n_instances[i_p],
                            env_idx if is_env_instanced[i_p] else -1,
                            instances_start[i_p],
                            inst_attr_start[i_p],
                            instances_buffer,
                            env_offset_buffer,
                            inst_tint_buffer_id[i_p],
                            is_env_instanced[i_p],
                            gl,
                        )

                gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glFlush()

        @nb.jit(
            nb.none(
                nb.int32[:],
                nb.int32[:],
                nb.int32[:, :],
                nb.float32[:, :, :],
                nb.int32[:],
                nb.int32[:],
                nb.int32[:],
                nb.float32[:, :, :],
                nb.float32[:],
                nb.boolean,
                nb.boolean,
                nb.boolean[:],
                nb.boolean[:],
                nb.boolean[:],
                nb.int32,
                nb.int64[:],
                nb.int32[:],
                nb.int32[:],
                nb.boolean[:, :],
                nb.float32,
                nb.boolean[:],
                nb.int32,
                nb.int32,
                nb.int32[:],
                nb.int32[:, :],
                nb.float32[:, :],
                nb.float32[:],
                nb.float32[:],
                nb.float32[:, :, :],
                nb.int32[:],
                self.gl.wrapper_type,
            ),
            cache=True,
        )
        def shadow_mapping_pass(
            vao_id,
            program_id,
            uniforms_loc,
            pose,
            mode,
            n_instances,
            n_indices,
            light_matrix,
            light_pos,
            is_point,
            is_transmittance,
            is_marker,
            is_transparent,
            is_env_filtered,
            env_idx,
            instances_start,
            inst_attr_start,
            inst_tint_buffer_id,
            env_active,
            env_offset_scale,
            is_env_instanced,
            instances_buffer,
            env_offset_buffer,
            tex_flags,
            textures,
            base_color_factor,
            transmission_factor,
            ior,
            texture_layers,
            texture_layers_idx,
            gl,
        ):
            last_pid = -1
            for i_p in range(len(vao_id)):
                # Markers cast no shadow
                if is_marker[i_p]:
                    continue
                # The transmittance map holds the transparent surfaces alone (see '_shadow_mapping_pass' in renderer.py)
                if is_transmittance and not is_transparent[i_p]:
                    continue

                pid = program_id[i_p]
                if pid != last_pid:
                    gl.glUseProgram(pid)

                    # A point light renders the six faces of its cube map at once (see point_shadow.geom)
                    if is_point:
                        gl.glUniformMatrix4fv(
                            uniforms_loc[i_p, U_LIGHT_MATRIX], 6, GL_TRUE, address_to_ptr(light_matrix.ctypes.data)
                        )
                        set_uniform_3fv(uniforms_loc[i_p, U_LIGHT_POS], light_pos, gl)
                    else:
                        set_uniform_matrix_4fv(uniforms_loc[i_p, U_LIGHT_MATRIX], light_matrix[0], gl)
                    gl.glUniform1f(uniforms_loc[i_p, U_ENV_OFFSET_SCALE], env_offset_scale)

                    last_pid = pid

                set_uniform_matrix_4fv(uniforms_loc[i_p, U_M], pose[i_p], gl)
                gl.glBindVertexArray(vao_id[i_p])

                # A transparent surface casts a shadow as dark as it is opaque (see mesh_depth.frag), and so does a
                # texture array whatever the layer shown (see TRANSPARENT_SHADOW in renderer.py). A texture array is
                # sampled at the layer of the environment drawn.
                i_layers = texture_layers_idx[i_p]
                is_layer_set = False
                tf = tex_flags[i_p]
                if is_transparent[i_p] or tf & (TexFlags_BASE_COLOR << 8):
                    set_uniform_4fv(uniforms_loc[i_p, U_BASE_COLOR_FACTOR], base_color_factor[i_p], gl)
                    gl.glUniform1f(uniforms_loc[i_p, U_TRANSMISSION_FACTOR], transmission_factor[i_p])
                    if is_transmittance:
                        gl.glUniform1f(uniforms_loc[i_p, U_IOR], ior[i_p])
                    if tf & TexFlags_BASE_COLOR:
                        gl.glActiveTexture(GL_TEXTURE0)
                        gl.glBindTexture(
                            GL_TEXTURE_2D_ARRAY if tf & (TexFlags_BASE_COLOR << 8) else GL_TEXTURE_2D, textures[i_p, 2]
                        )
                        gl.glUniform1i(uniforms_loc[i_p, U_BASE_COLOR_TEXTURE], 0)
                        is_layer_set = i_layers >= 0
                        if is_layer_set and env_idx >= 0:
                            gl.glUniform1f(uniforms_loc[i_p, U_BASE_COLOR_LAYER], texture_layers[i_layers, env_idx, 0])

                # Both faces of a transparent surface filter the light, and blend into the transmittance map
                if is_transmittance:
                    gl.glDisable(GL_CULL_FACE)
                    gl.glEnable(GL_BLEND)
                else:
                    gl.glEnable(GL_CULL_FACE)
                    gl.glCullFace(GL_BACK)
                    gl.glDisable(GL_BLEND)
                gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                gl.glDisable(GL_PROGRAM_POINT_SIZE)

                if (is_env_filtered[i_p] or is_layer_set) and env_idx == -1:
                    # Combined draw-all of a heterogeneous variant: shadow only the environments it is active in,
                    # otherwise the variant would cast a shadow in every environment's instance. A primitive sampling a
                    # texture array casts the shadow of one environment at a time, each at its own layer.
                    for k in range(n_instances[i_p]):
                        if not env_active[i_p, k]:
                            continue
                        if is_layer_set:
                            gl.glUniform1f(uniforms_loc[i_p, U_BASE_COLOR_LAYER], texture_layers[i_layers, k, 0])
                        draw_instances(
                            mode[i_p],
                            n_indices[i_p],
                            n_instances[i_p],
                            k,
                            instances_start[i_p],
                            inst_attr_start[i_p],
                            instances_buffer,
                            env_offset_buffer,
                            inst_tint_buffer_id[i_p],
                            is_env_instanced[i_p],
                            gl,
                        )
                else:
                    draw_instances(
                        mode[i_p],
                        n_indices[i_p],
                        n_instances[i_p],
                        env_idx if is_env_instanced[i_p] else -1,
                        instances_start[i_p],
                        inst_attr_start[i_p],
                        instances_buffer,
                        env_offset_buffer,
                        inst_tint_buffer_id[i_p],
                        is_env_instanced[i_p],
                        gl,
                    )

                gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glFlush()

        @nb.jit(nb.float32[:, :](nb.int32, nb.int32, nb.float32, nb.float32, self.gl.wrapper_type), cache=True)
        def read_depth_buf(width, height, z_near, z_far, gl):
            buf = np.zeros((height, width), np.float32)
            gl.glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, address_to_ptr(buf.ctypes.data))
            depth_im = buf[::-1, :] * 2 - 1
            if z_far < 0:
                depth_im = z_near / (1 - depth_im) * 2
            else:
                depth_im = (z_near * z_far) / (z_far + z_near - depth_im * (z_far - z_near)) * 2
            return depth_im

        @nb.jit(nb.uint8[:, :, :](nb.int32, nb.int32, self.gl.wrapper_type), cache=True)
        def read_color_buf(width, height, gl):
            # Drivers copy RGBA pixels as they are stored, while reading RGB converts each pixel
            buf = np.empty((height, width, 4), np.uint8)
            gl.glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, address_to_ptr(buf.ctypes.data))
            return buf[::-1, :, :3]

        @nb.jit(nb.none(nb.int64[:, :], self.gl.wrapper_type), cache=True)
        def update_buffer(updates, gl):
            """Write ranges of GL buffers, given per row as the buffer, the byte offset and size of the range, the
            address of the data and the byte size of the buffer, -1 for a buffer to skip.

            A write of a whole buffer allocates it anew first, so that the driver does not wait for the draws still
            reading its previous content.
            """
            for i in range(updates.shape[0]):
                buffer_id = updates[i, 0]
                offset = updates[i, 1]
                size = updates[i, 2]
                buffer_size = updates[i, 4]
                if buffer_id >= 0:
                    gl.glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
                    if offset == 0 and size == buffer_size:
                        gl.glBufferData(GL_ARRAY_BUFFER, buffer_size, address_to_ptr(0), GL_STREAM_DRAW)
                    gl.glBufferSubData(GL_ARRAY_BUFFER, offset, size, address_to_ptr(updates[i, 3]))
            gl.glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._forward_pass = forward_pass
        self._shadow_mapping_pass = shadow_mapping_pass
        self._read_depth_buf = read_depth_buf
        self._read_color_buf = read_color_buf
        self._update_buffer_fn = update_buffer

    def forward_pass(
        self,
        renderer,
        V,
        P,
        cam_pos,
        flags,
        program_flags,
        screen_size,
        color_list=None,
        reflection_mat=np.identity(4, np.float32),
        floor_tex=0,
        env_idx=-1,
        markers_only=False,
    ):
        self.load_programs(renderer, flags, program_flags)
        self.gen_func_ptr()
        if self._placeholder_textures is None:
            # A white 2D color texture, a 2D depth texture compared as a shadow map, a depth cube map and a white color
            # cube map, the colors letting a light through untinted (see bind_lighting)
            self._placeholder_textures = (
                Texture(source=np.full((1, 1, 4), 255, np.uint8), source_channels=TexChannels.RGBA),
                Texture(width=1, height=1, source_channels=TexChannels.DEPTH, data_format=GL.GL_UNSIGNED_SHORT),
                CubeMapTexture(
                    1, TexChannels.DEPTH, GL.GL_FLOAT, Sampler(mag_filter=GL.GL_NEAREST, min_filter=GL.GL_NEAREST)
                ),
                CubeMapTexture(
                    1, TexChannels.RGBA, GL.GL_HALF_FLOAT, Sampler(mag_filter=GL.GL_LINEAR, min_filter=GL.GL_LINEAR)
                ),
            )
            for texture in self._placeholder_textures:
                texture._add_to_context()
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self._placeholder_textures[3]._texid)
            for i_face in range(6):
                GL.glTexSubImage2D(
                    GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i_face,
                    0,
                    0,
                    0,
                    1,
                    1,
                    GL.GL_RGBA,
                    GL.GL_HALF_FLOAT,
                    np.ones((1, 1, 4), np.float16),
                )
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
            self.placeholder_tex = np.array([texture._texid for texture in self._placeholder_textures], np.int32)
        # The primitives a pass leaves out have their index count set to 0 for the pass, so that no draw call renders
        # them: the markers for the cameras that skip them, everything but the markers for the X-ray pass, and the
        # primitives without normals for the pass drawing the normals.
        is_hidden = np.zeros_like(self.is_marker)
        if flags & RenderFlags.SKIP_MARKERS:
            is_hidden |= self.is_marker
        if markers_only:
            is_hidden |= ~self.is_marker
        # The normals drawn span 5% of the diagonal of the bounding box of their primitive
        normal_magnitude = np.empty(0, np.float32)
        if flags & RenderFlags_NORMALS:
            is_hidden |= [not primitive.buf_flags & BufFlags.NORMAL for primitive in self.primitive_list]
            normal_magnitude = np.array([0.05 * primitive.scale for primitive in self.primitive_list], np.float32)
        n_indices_hidden = self.n_indices[is_hidden].copy()
        self.n_indices[is_hidden] = 0

        # Opaque primitives are drawn nearest first, so that the depth test rejects what they hide as early as
        # possible, and transparent ones farthest first, the only order their blending is correct in. Both are keyed on
        # the depth of the primitive's centre along the view axis, which is what decides which lies behind the other.
        # Computed here rather than once and for all, so it follows the camera of this pass and of this environment.
        centres_local = self.centroid_local[:, min(max(env_idx, 0), self.centroid_local.shape[1] - 1)]
        centroids = np.einsum("nij,nj->ni", self.pose[:, :3, :3], centres_local) + self.pose[:, :3, 3]
        # The floor pass mirrors the scene through the view matrix itself, so that every pass positions its vertices
        # with the same expression (see 'invariant' in mesh.vert). The mirror reverses which of two meshes lies behind
        # the other, so the depth row carries it too.
        V = V @ reflection_mat
        view_row = V[2]
        depths = centroids @ view_row[:3] + view_row[3]
        # Depth along the view axis grows towards the viewer, so ascending depth is farthest first
        draw_order = np.lexsort((np.where(self.is_transparent, depths, -depths), self.is_transparent)).astype(np.int32)

        with self._culled(env_idx, P @ V, not is_env_pass(flags)):
            self._forward_pass(
                self.vao_id,
                *self.programs[(flags, program_flags)],
                self.pose,
                self.tex_flags,
                self.textures,
                self.base_color_factor,
                self.metallic_factor,
                self.roughness_factor,
                self.emissive_factor,
                self.transmission_factor,
                self.ior,
                self.is_blended,
                self.is_wireframe,
                self.is_reflective_floor,
                self.is_marker,
                self.is_env_filtered,
                self.is_transmissive,
                self.mode,
                self.n_instances,
                self.n_indices,
                self.lights_type,
                self.lights_color,
                self.lights_intensity,
                self.lights_pos,
                self.lights_dir,
                self.lights_angle_scale,
                self.lights_angle_offset,
                self.shadow_map,
                self.transmittance_map,
                self.lights_matrix,
                self.ambient_light,
                np.ascontiguousarray(V, dtype=np.float32),
                np.ascontiguousarray(P, dtype=np.float32),
                np.ascontiguousarray(cam_pos, dtype=np.float32),
                flags,
                color_list if flags & RenderFlags.SEG else np.empty((0, 3), np.float32),
                reflection_mat,
                floor_tex,
                self.placeholder_tex,
                screen_size,
                env_idx,
                self.instances_start,
                self.inst_tint_buffer_id,
                self.inst_attr_start,
                self.env_active,
                draw_order,
                0.0 if is_env_pass(flags) else 1.0,
                int(self.view == ViewFlags.DEBUG),
                self.is_env_instanced,
                self.instances_buffer if self.instances_buffer is not None else 0,
                self.env_offset_buffer if self.env_offset_buffer is not None else 0,
                self.texture_layers,
                self.texture_layers_idx,
                normal_magnitude,
                self.gl.wrapper_instance,
            )
        self.n_indices[is_hidden] = n_indices_hidden

    def shadow_mapping_pass(self, renderer, light_node, flags, env_idx=-1, is_transmittance=False):
        """Render the shadow map of a light, the depth seen from it or for a point light its distance to the scene.

        Parameters
        ----------
        renderer : Renderer
            The renderer whose shader programs draw the pass.
        light_node : Node
            The node of the light, among the lights of the last 'set_light'.
        flags : int
            The render flags of the pass.
        env_idx : int, optional
            The environment drawn, -1 for all of them. Defaults to -1.
        is_transmittance : bool, optional
            Whether the pass renders the transmittance map of the light rather than its shadow map. Defaults to False.
        """
        i_light = self.lights_idx_by_node[light_node]
        is_point = self.lights_type[i_light] == LightType.POINT
        program_flags = ProgramFlags.POINT_SHADOW if is_point else ProgramFlags.SHADOW
        if is_transmittance:
            program_flags |= ProgramFlags.TRANSMITTANCE
        self.load_programs(renderer, flags, program_flags)
        self.gen_func_ptr()
        with self._culled(env_idx):
            self._shadow_mapping_pass(
                self.vao_id,
                *self.programs[(flags, program_flags)],
                self.pose,
                self.mode,
                self.n_instances,
                self.n_indices,
                self.lights_matrix[i_light],
                self.lights_pos[i_light],
                is_point,
                is_transmittance,
                self.is_marker,
                self.is_transparent,
                self.is_env_filtered,
                env_idx,
                self.instances_start,
                self.inst_attr_start,
                self.inst_tint_buffer_id,
                self.env_active,
                0.0 if is_env_pass(flags) else 1.0,
                self.is_env_instanced,
                self.instances_buffer if self.instances_buffer is not None else 0,
                self.env_offset_buffer if self.env_offset_buffer is not None else 0,
                self.tex_flags,
                self.textures,
                self.base_color_factor,
                self.transmission_factor,
                self.ior,
                self.texture_layers,
                self.texture_layers_idx,
                self.gl.wrapper_instance,
            )

    def update_texture_layers(self, primitives) -> None:
        """Copy the texture array layers each rendered environment of some primitives samples, for the next render pass.

        Parameters
        ----------
        primitives : iterable of Primitive
            The primitives whose layers changed. Those that this renderer does not draw are skipped.
        """
        meshes = set()
        is_shadow_changed = False
        for primitive in primitives:
            i = self._primitive_index.get(primitive)
            if i is not None:
                self.texture_layers[self.texture_layers_idx[i]] = primitive.texture_layers
                meshes.add(self.node_list[i].mesh)
                if self.tex_flags[i] & (TexFlags.BASE_COLOR << 8):
                    is_shadow_changed = True
        # The shadow of a base color texture array follows the layers shown (see TRANSPARENT_SHADOW in renderer.py),
        # while a shadow map is rendered again only once the scene changed
        if is_shadow_changed:
            for light_node in self._scene.light_nodes:
                light_node.light.shadow_map_revision = None
        # The alpha of the layers shown decides whether a mesh is drawn with the transparent ones, a flag that every
        # primitive of the mesh carries. It is evaluated once per mesh, since it reads every primitive of the mesh.
        for mesh in meshes:
            is_transparent = mesh.is_transparent
            for mesh_primitive in mesh.primitives:
                i_p = self._primitive_index.get(mesh_primitive)
                if i_p is not None:
                    self.is_transparent[i_p] = is_transparent

    def update_buffer(self, primitives, buffer_name, data):
        """Queue the update of one vertex buffer of several primitives, flushed during the next render pass.

        New vertex positions update the bounds of the bare geometry and of the instances. A placed primitive refreshes
        the latter in bulk instead (see '_update_instances').

        Parameters
        ----------
        primitives : Sequence[Primitive]
            The primitives.
        buffer_name : str
            The buffer, such as 'pos', 'normal' or 'inst_tint'.
        data : Sequence[np.ndarray]
            The new content of the buffer of each primitive.
        """
        # The GL buffer is looked up as the queue is flushed, since a primitive enters the GL context at its first
        # render pass, after which the updates queued before it still apply
        for primitive, primitive_data in zip(primitives, data):
            self._buffer[(primitive, buffer_name)] = primitive_data
            i = self._primitive_index.get(primitive)
            if i is not None and len(primitive_data) and buffer_name == "pos":
                self.bounds_geom[i, 0] = primitive_data.min(axis=0)
                self.bounds_geom[i, 1] = primitive_data.max(axis=0)
                # The bounds of the instances follow the new box. A placed primitive refreshes them in bulk.
                if not self.is_instance_placed[i]:
                    instance_poses = primitive.poses
                    if instance_poses is None or not len(instance_poses):
                        instance_poses = np.eye(4, dtype=np.float32)[np.newaxis]
                    self.instance_bounds[i] = transform_box(self.bounds_geom[i], instance_poses)
        self._scene.bump_revision()

    def update_poses(self, primitives, poses):
        """Set the transforms of the instances of several primitives, uploaded during the next render pass.

        The bounds of the instances follow, except for a placed primitive, which refreshes them in bulk (see
        '_update_instances'). A primitive that this renderer starts drawing at its next render pass reads its transforms
        from 'Primitive.poses' then.

        Parameters
        ----------
        primitives : Sequence[Primitive]
            The primitives, each with the number of instances it was drawn with so far.
        poses : np.ndarray
            The transform of each instance of each primitive, one row per primitive.
        """
        idx = np.fromiter((self._primitive_index.get(primitive, -1) for primitive in primitives), int, len(primitives))
        is_drawn = idx >= 0
        idx, poses = idx[is_drawn], poses[is_drawn]
        assert (self.n_instances[idx] == poses.shape[1]).all()
        rows = self.instances_start[idx, np.newaxis] + np.arange(poses.shape[1])
        self.instances_pose[rows] = poses.swapaxes(-1, -2)
        self.is_pose_changed[idx] = True
        is_placed = self.is_instance_placed[idx]
        if poses.shape[1] and not is_placed.all():
            self.instance_bounds[idx[~is_placed]] = transform_box(self.bounds_geom[idx[~is_placed]], poses[~is_placed])
        self._scene.bump_revision()

    def flush_buffer(self):
        """Upload the transforms of the instances that changed and the queued buffer updates, and clear the queue."""
        # One row per write (see 'update_buffer' in 'gen_func_ptr')
        updates = []
        buffers = []
        for (primitive, buffer_name), data in self._buffer.items():
            buffer = np.ascontiguousarray(data, dtype=np.float32)
            buffers.append(buffer)
            # A primitive removed from the scene since the update was queued holds no buffer, which the upload skips
            buffer_id = primitive.buffers_id.get(buffer_name, -1)
            updates.append((buffer_id, 0, buffer.nbytes, buffer.ctypes.data, buffer.nbytes))
        if self.instances_buffer is not None and self.is_pose_changed.any():
            # The transforms upload by runs of neighbouring primitives, a single run when every primitive moved
            is_changed = np.concatenate(([False], self.is_pose_changed, [False]))
            runs = np.flatnonzero(is_changed[1:] != is_changed[:-1]).reshape((-1, 2))
            stride = self.instances_pose.strides[0]
            address = self.instances_pose.ctypes.data
            for start, end in self.instances_start[runs] * stride:
                updates.append((self.instances_buffer, start, end - start, address + start, self.instances_pose.nbytes))
            self.is_pose_changed[:] = False
        if not updates:
            return

        self.gen_func_ptr()
        self._update_buffer_fn(np.array(updates, dtype=np.int64), self.gl.wrapper_instance)
        self._buffer.clear()

    def read_depth_buf(self, width, height, z_near, z_far):
        """Read the depth buffer of the current framebuffer as the distance to the camera plane, rows from the top.

        Parameters
        ----------
        width : int
            The width of the buffer, in pixels.
        height : int
            The height of the buffer, in pixels.
        z_near : float
            The near plane of the camera.
        z_far : float
            The far plane of the camera, negative for an infinite projection.
        """
        self.gen_func_ptr()
        return self._read_depth_buf(width, height, z_near, z_far, self.gl.wrapper_instance)

    def read_color_buf(self, width, height):
        """Read the RGB color buffer of the current framebuffer, rows from the top.

        Parameters
        ----------
        width : int
            The width of the buffer, in pixels.
        height : int
            The height of the buffer, in pixels.
        """
        self.gen_func_ptr()
        return self._read_color_buf(width, height, self.gl.wrapper_instance)
