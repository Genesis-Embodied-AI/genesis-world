import contextlib
import os
from typing import NamedTuple

import numpy as np
import numba as nb

import OpenGL.GL as GL
import OpenGL.constant as GL_constant

from .constants import RenderFlags, ViewFlags, MAX_N_LIGHTS
from .light import DirectionalLight, PointLight, SpotLight
from .material import MetallicRoughnessMaterial, SpecularGlossinessMaterial
from .numba_gl_wrapper import GLWrapper
from .scene import CORNER_INDICES
from .texture import CubeMapTexture, Texture
from .utils import transform_box


_DISABLE_OFFSCREEN_MARKERS = "GS_DISABLE_OFFSCREEN_MARKERS" in os.environ


class PlacedGroup(NamedTuple):
    """Primitives placed through instance transforms that share an instance count, so that their poses stack."""

    idx: np.ndarray
    n_instances: int


def load_const(const_name):
    c = GL.__getattribute__(const_name)
    if isinstance(c, GL_constant.FloatConstant):
        return float(c)
    if isinstance(c, GL_constant.IntConstant) or isinstance(c, GL_constant.LongConstant):
        return int(c)
    if isinstance(c, GL_constant.StringConstant):
        return str(c, "utf-8")
    raise TypeError("Unknown OpenGL constant type")


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
GL_ZERO = load_const("GL_ZERO")
GL_FRONT_AND_BACK = load_const("GL_FRONT_AND_BACK")
GL_LINE = load_const("GL_LINE")
GL_FILL = load_const("GL_FILL")
GL_CULL_FACE = load_const("GL_CULL_FACE")
GL_BACK = load_const("GL_BACK")
GL_FRONT = load_const("GL_FRONT")
GL_PROGRAM_POINT_SIZE = load_const("GL_PROGRAM_POINT_SIZE")
GL_UNSIGNED_INT = load_const("GL_UNSIGNED_INT")
GL_RGBA = load_const("GL_RGBA")
GL_RGB = load_const("GL_RGB")
GL_DEPTH_COMPONENT = load_const("GL_DEPTH_COMPONENT")
GL_UNSIGNED_BYTE = load_const("GL_UNSIGNED_BYTE")
GL_FLOAT = load_const("GL_FLOAT")
GL_ARRAY_BUFFER = load_const("GL_ARRAY_BUFFER")
GL_STREAM_DRAW = load_const("GL_STREAM_DRAW")

RenderFlags_DEPTH_ONLY = RenderFlags.DEPTH_ONLY
RenderFlags_SEG = RenderFlags.SEG
RenderFlags_FLIP_WIREFRAME = RenderFlags.FLIP_WIREFRAME
RenderFlags_ALL_WIREFRAME = RenderFlags.ALL_WIREFRAME
RenderFlags_SKIP_CULL_FACES = RenderFlags.SKIP_CULL_FACES
RenderFlags_SHADOWS_DIRECTIONAL = RenderFlags.SHADOWS_DIRECTIONAL
RenderFlags_SHADOWS_SPOT = RenderFlags.SHADOWS_SPOT
RenderFlags_SHADOWS_POINT = RenderFlags.SHADOWS_POINT
RenderFlags_SKIP_FLOOR = RenderFlags.SKIP_FLOOR
RenderFlags_OFFSCREEN = RenderFlags.OFFSCREEN
RenderFlags_REFLECTIVE_FLOOR = RenderFlags.REFLECTIVE_FLOOR
RenderFlags_FLAT = RenderFlags.FLAT


@nb.jit(nopython=True, cache=True)
def get_uniform_location(pid, name, gl):
    n = len(name)
    arr = np.zeros(n + 1, np.uint8)
    for i in range(n):
        arr[i] = ord(name[i])
    return gl.glGetUniformLocation(pid, arr.ctypes.data)


@nb.jit(nopython=True, cache=True)
def set_uniform_matrix_4fv(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniformMatrix4fv(loc, 1, GL_TRUE, address_to_ptr(value.ctypes.data))
    else:
        print("uniform not found:", name)


@nb.jit(nopython=True, cache=True)
def set_uniform_1i(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform1i(loc, value)
    else:
        print("uniform not found:", name)


@nb.jit(nopython=True, cache=True)
def set_uniform_1f(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform1f(loc, value)
    else:
        print("uniform not found:", name)


@nb.jit(nopython=True, cache=True)
def set_texture_layers(pid, tex_flags, layers, gl):
    """Set the layers that the base color (0) and the emissive (1) texture arrays of a material are sampled at."""
    if tex_flags & (8 << 8):
        set_uniform_1f(pid, "material.base_color_layer", layers[0], gl)
    if tex_flags & (4 << 8):
        set_uniform_1f(pid, "material.emissive_layer", layers[1], gl)


@nb.jit(nopython=True, cache=True)
def set_uniform_2f(pid, name, value1, value2, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform2f(loc, value1, value2)
    else:
        print("uniform not found:", name)


@nb.jit(nopython=True, cache=True)
def set_uniform_3fv(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform3fv(loc, 1, address_to_ptr(value.ctypes.data))
    else:
        print("uniform not found:", name)


@nb.jit(nopython=True, cache=True)
def set_uniform_4fv(pid, name, value, gl):
    loc = get_uniform_location(pid, name, gl)
    if loc >= 0:
        gl.glUniform4fv(loc, 1, address_to_ptr(value.ctypes.data))
    else:
        print("uniform not found:", name)


@nb.jit(nopython=True, cache=True)
def bind_lighting(pid, flags, light, shadow_map, light_matrix, ambient_light, placeholder_tex, gl):
    n = len(light)
    set_uniform_3fv(pid, "ambient_light", ambient_light, gl)
    n_dir, n_pt, n_spot = 0, 0, 0
    active_texture = 0
    depth_tex = -1
    cube_tex = -1

    for i in range(n):
        if abs(light[i, 7] - 0.0) < 0.1:
            b = "directional_lights[" + str(n_dir) + "]."
            set_uniform_3fv(pid, b + "color", light[i, :3], gl)
            set_uniform_3fv(pid, b + "direction", light[i, 3:6], gl)
            set_uniform_1f(pid, b + "intensity", light[i, 6], gl)
            if flags & RenderFlags_SHADOWS_DIRECTIONAL:
                if active_texture == 0:
                    active_texture += 1
                gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                gl.glBindTexture(GL_TEXTURE_2D, shadow_map[i])
                set_uniform_1i(pid, b + "shadow_map", active_texture, gl)
                set_uniform_matrix_4fv(pid, b + "light_matrix", light_matrix[i, 0], gl)
                depth_tex = active_texture
                active_texture += 1
            n_dir += 1
        elif abs(light[i, 7] - 1.0) < 0.1:
            b = "point_lights[" + str(n_pt) + "]."
            set_uniform_3fv(pid, b + "color", light[i, :3], gl)
            set_uniform_3fv(pid, b + "position", light[i, 3:6], gl)
            set_uniform_1f(pid, b + "intensity", light[i, 6], gl)
            if flags & RenderFlags_SHADOWS_POINT:
                if active_texture == 0:
                    active_texture += 1
                gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                gl.glBindTexture(GL_TEXTURE_CUBE_MAP, shadow_map[i])
                set_uniform_1i(pid, b + "shadow_map", active_texture, gl)
                cube_tex = active_texture
                active_texture += 1
            n_pt += 1

    set_uniform_1i(pid, "n_directional_lights", n_dir, gl)
    set_uniform_1i(pid, "n_spot_lights", n_spot, gl)
    set_uniform_1i(pid, "n_point_lights", n_pt, gl)

    # A draw fails when sampler uniforms of two types point at the same texture unit, and every sampler left unset
    # points at unit 0 along with the material textures (sampler2D). So the shadow samplers (sampler2DShadow,
    # samplerCube) take units of their own, and the light slots beyond the lights present point at one of those. A
    # unit that no light fills holds a placeholder of its type, since a sampler reading no complete texture makes the
    # draw invalid on some drivers.
    if flags & RenderFlags_SHADOWS_DIRECTIONAL:
        if depth_tex < 0:
            if active_texture == 0:
                active_texture += 1
            depth_tex = active_texture
            active_texture += 1
            gl.glActiveTexture(GL_TEXTURE0 + depth_tex)
            gl.glBindTexture(GL_TEXTURE_2D, placeholder_tex[1])

        for i in range(n_dir, MAX_N_LIGHTS):
            b = "directional_lights[" + str(i) + "]."
            set_uniform_1i(pid, b + "shadow_map", depth_tex, gl)

    if flags & RenderFlags_SHADOWS_POINT:
        if cube_tex < 0:
            if active_texture == 0:
                active_texture += 1
            cube_tex = active_texture
            active_texture += 1
            gl.glActiveTexture(GL_TEXTURE0 + cube_tex)
            gl.glBindTexture(GL_TEXTURE_CUBE_MAP, placeholder_tex[2])

        for i in range(n_pt, MAX_N_LIGHTS):
            b = "point_lights[" + str(i) + "]."
            set_uniform_1i(pid, b + "shadow_map", cube_tex, gl)

    return active_texture


@nb.extending.intrinsic
def address_to_ptr(typingctx, src):
    """returns a void pointer from a given memory address"""
    sig = nb.core.types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], nb.core.cgutils.voidptr_t)

    return sig, codegen


def is_env_pass(flags):
    """Whether a pass draws one environment where it is, for a camera bound to it, rather than every environment side
    by side on the grid as the viewer window does. The window draws the grid whatever the separation setting: the
    setting picks how its cameras render (see 'JITRenderer.env_offset_buffer')."""
    return bool(flags & RenderFlags.ENV_SEPARATE and flags & RenderFlags.OFFSCREEN)


class JITRenderer:
    def __init__(self, scene, node_list, primitive_list, envs_offset):
        self._forward_pass = None
        self._shadow_mapping_pass = None
        self._read_depth_buf = None
        self._read_color_buf = None
        self._update_normal_flat = None
        self._update_normal_smooth = None
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
        # Scene revision the node poses were last gathered at (see Scene.revision)
        self._scene_revision = None
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
        self.set_primitive(scene, node_list, primitive_list)
        self.set_light(scene, scene.light_nodes, scene.ambient_light)
        self.reflection_mat = np.identity(4, np.float32)

    def _update_instances(self):
        """Refresh what follows the instance transforms: the centre of each placed primitive per environment (see
        'set_primitive') and the bounds its instances cover."""
        for idx, n_instances in self._placed_groups:
            poses = np.stack([self.primitive_list[i].poses for i in idx])
            centres = np.einsum("kenj,kj->ken", poses[:, :, :3, :3], self.centroid_geom[idx]) + poses[:, :, :3, 3]
            if n_instances == 1:
                # A lone instance places the primitive in every environment, so its centre is the centre for all of them
                self.centroid_local[idx] = centres
            else:
                self.centroid_local[idx, :n_instances] = centres
            self._set_instance_bounds(idx, poses)

    def _set_instance_bounds(self, idx, poses):
        """Record the bounds the instances of some primitives cover in the frame of their node.

        The range of the offsets the grid pass moves the instances by is recorded as well. An instance placed in one
        environment moves by the offset of that environment, and instance k of an environment-instanced primitive by
        the offset of environment k.

        Parameters
        ----------
        idx : np.ndarray
            The indices of the primitives.
        poses : np.ndarray
            The transforms of the instances of each primitive, one row per primitive.
        """
        self.instance_bounds[idx] = transform_box(self.bounds_geom[idx], poses)
        offsets = np.zeros((*poses.shape[:-2], 3), dtype=np.float32)
        is_env_placed = self.env_row[idx] >= 0
        offsets[is_env_placed] = self.envs_offset[self.env_row[idx][is_env_placed], np.newaxis]
        if self.is_env_instanced[idx].any():
            offsets[self.is_env_instanced[idx]] = self.envs_offset
        self.grid_offset_bounds[idx, 0] = offsets.min(axis=1)
        self.grid_offset_bounds[idx, 1] = offsets.max(axis=1)

    def update(self, scene):
        if scene.meshes_updated:
            node_list, primitive_list = [], []
            for node in scene.sorted_mesh_nodes():
                mesh = node.mesh
                if not mesh.is_visible:
                    continue
                for primitive in mesh.primitives:
                    node_list.append(node)
                    primitive_list.append(primitive)
            self.set_primitive(scene, node_list, primitive_list)
            scene.reset_meshes_updated()
        elif self._scene_revision != scene.revision:
            # A node moves through Scene.set_pose, which advances the revision. Its own transform setters leave it
            # alone.
            for i, node in enumerate(self.node_list):
                self.pose[i] = scene.get_pose(node)
            # Instances move with the simulation, so where they place their primitive has to be refreshed as well
            self._update_instances()
        if self._scene_revision != scene.revision:
            corners = self.instance_bounds.reshape((-1, 6))[:, CORNER_INDICES]
            corners = np.einsum("nij,ncj->nci", self.pose[:, :3, :3], corners) + self.pose[:, np.newaxis, :3, 3]
            self.corners[False] = corners
            # The grid pass moves an instance by its offset after the node pose (see 'mesh.vert'), so its bounds widen
            # the world bounds by the range of the offsets
            grid_bounds_min = corners.min(axis=1) + self.grid_offset_bounds[:, 0]
            grid_bounds_max = corners.max(axis=1) + self.grid_offset_bounds[:, 1]
            grid_bounds = np.stack((grid_bounds_min, grid_bounds_max), axis=1)
            self.corners[True] = grid_bounds.reshape((-1, 6))[:, CORNER_INDICES]
            # The scene bounds as 'Scene.bounds' defines them, over every primitive at once. A floor counts by its
            # centre alone. A marker counts for nothing, so a scene of markers keeps the bounds of 'Scene.bounds'.
            is_bounded = self.render_flags[:, 6] == 0
            if is_bounded.any():
                for is_grid, corners in self.corners.items():
                    corners = corners.copy()
                    corners[self.is_floor] = corners[self.is_floor].mean(axis=1, keepdims=True)
                    corners = corners[is_bounded].reshape((-1, 3))
                    self.scene_bounds[is_grid] = np.stack((corners.min(axis=0), corners.max(axis=0)), axis=0)
        self._scene_revision = scene.revision

    def set_lighting(self, scene, is_grid):
        """Set up the lights for a pass, and tell whether every shadow texture was found in the context.

        The setup follows the scene: the light poses and the bounds the directional shadow frustum is fitted to, which
        the grid pass has its own of. It stays unsettled while a shadow texture is missing from the context, so the next
        render tries again.
        """
        if self.scene_bounds[is_grid] is not None:
            scene.bounds = self.scene_bounds[is_grid]
        lights_revision = (scene.revision, is_grid)
        if self._lights_revision != lights_revision:
            self._is_lighting_ready = self.set_light(scene, scene.light_nodes, scene.ambient_light)
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

    def update_context(self, scene, flags):
        """Add to the GL context the meshes and textures the scene holds and free the ones it dropped.

        Runs once per scene revision. The shadow flags decide which lights get a shadow texture, so they are part of
        the key.
        """
        shadow_flags = flags & (RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_POINT | RenderFlags.SHADOWS_SPOT)
        if self._context_revision == (scene.revision, shadow_flags):
            return
        self._context_revision = (scene.revision, shadow_flags)

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
            # Create if needed
            active = False
            if isinstance(light, DirectionalLight) and flags & RenderFlags.SHADOWS_DIRECTIONAL:
                active = True
            elif isinstance(light, PointLight) and flags & RenderFlags.SHADOWS_POINT:
                active = True
            elif isinstance(light, SpotLight) and flags & RenderFlags.SHADOWS_SPOT:
                active = True

            if active and light.shadow_texture is None:
                light._generate_shadow_texture()
            if light.shadow_texture is not None:
                shadow_textures.add(light.shadow_texture)

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
        if self.env_offset_buffer is not None:
            try:
                GL.glDeleteBuffers(1, [self.env_offset_buffer])
            except (GL.error.GLError, GL.error.NullFunctionError):
                pass
            self.env_offset_buffer = None

    def set_light(self, scene, light_nodes, ambient_light):
        self.light_list = light_nodes

        n = len(light_nodes)
        if n > MAX_N_LIGHTS:
            raise ValueError("Number of lights exceeds limit.")

        # directional: color <- 3, direction <- 3, intensity, 0
        # point:       color <- 3, position <- 3, intensity, 1
        self.light = np.zeros((n, 8), np.float32)
        self.shadow_map = np.zeros(n, np.int32)
        self.light_matrix = np.zeros((n, 6, 4, 4), np.float32)

        self.ambient_light = np.array(ambient_light, np.float32)

        all_textures_ready = True

        for i, node in enumerate(light_nodes):
            light = node.light
            pose = scene.get_pose(node)
            position = pose[:3, 3]
            direction = -pose[:3, 2]

            if isinstance(light, DirectionalLight):
                self.light[i, :3] = light.color
                self.light[i, 3:6] = direction
                self.light[i, 6] = light.intensity
                self.light[i, 7] = 0

                if light.shadow_texture:
                    if light.shadow_texture._in_context():
                        self.shadow_map[i] = light.shadow_texture._texid
                    else:
                        all_textures_ready = False

                pose = pose.copy()
                camera = light._get_shadow_camera(scene.scale)
                P = camera.get_projection_matrix()
                c = scene.centroid
                loc = c - direction * scene.scale
                pose[:3, 3] = loc
                V = np.linalg.inv(pose)  # V maps from world to camera
                self.light_matrix[i, 0] = P.dot(V)
            elif isinstance(light, PointLight):
                self.light[i, :3] = light.color
                self.light[i, 3:6] = position
                self.light[i, 6] = light.intensity
                self.light[i, 7] = 1

                if light.shadow_texture:
                    if light.shadow_texture._in_context():
                        self.shadow_map[i] = light.shadow_texture._texid
                    else:
                        all_textures_ready = False

                camera = light._get_shadow_camera(scene.scale)
                projection = camera.get_projection_matrix()
                view = light._get_view_matrices(position)
                self.light_matrix[i] = projection @ view
            else:
                raise TypeError("Light type not supported yet.")

        return all_textures_ready

    def set_primitive(self, scene, node_list, primitive_list):
        self.node_list = node_list
        self.primitive_list = primitive_list

        n = len(primitive_list)
        self.vao_id = np.zeros(n, np.int32)
        self.program_id = {}
        self.pose = np.zeros((n, 4, 4), np.float32)
        # 0: flags, bit i set for texture slot i, and bit i + 8 for one that is a texture array, 1-7: texture id
        self.textures = np.zeros((n, 8), np.int32)
        # Row i of texture_layers holds, for each rendered environment, the layer of the base color (0) and emissive (1)
        # texture arrays that a layered primitive samples (see 'Primitive.texture_layers').
        # texture_layers_idx maps each primitive to its row, or -1 for a primitive sampling no texture array.
        is_layered = np.array([primitive.texture_layers is not None for primitive in primitive_list], np.bool_)
        self.texture_layers_idx = np.full(n, -1, np.int32)
        self.texture_layers_idx[is_layered] = np.arange(np.count_nonzero(is_layered))
        self.texture_layers = np.zeros((np.count_nonzero(is_layered), len(self.envs_offset), 2), np.float32)
        # base_color <- 4, metallic <- 1, roughness <- 1, emissive <- 3, transmission <- 1, ior <- 1
        self.pbr_mat = np.zeros((n, 11), np.float32)
        self.spec_mat = np.zeros((n, 11), np.float32)  # diffuse <- 4, specular <- 3, glossiness <- 1, emissive <- 3
        # (blend, wireframe, double sided, pbr texture, reflective floor, transparent, marker, env shared, env filtered,
        # translucent color, transmissive)
        self.render_flags = np.zeros((n, 11), np.int8)
        self.mode = np.zeros(n, np.int32)
        self.n_instances = np.zeros(n, np.int32)
        # Per-(primitive, env) visibility used only when a primitive is "env filtered" (render_flags column 8), i.e. a
        # heterogeneous variant present in a subset of environments. n_env spans the per-env instance poses.
        n_env = max((len(p.poses) for p in primitive_list if p.poses is not None), default=1)
        self.env_active = np.ones((n, n_env), np.bool_)
        self.n_indices = np.zeros(n, np.int32)  # positive: indices, negative: positions
        # The views drawing each primitive (see 'Mesh.views')
        self.views = np.zeros(n, np.int8)
        self.model_buffer_id = np.zeros(n, np.int32)
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
        # World corners of the bounds of each primitive, for a pass rendering one environment at a time (False) and for
        # the grid pass (True) (see 'update')
        self.corners = {False: None, True: None}
        self._primitive_index = {primitive: i for i, primitive in enumerate(primitive_list)}

        floor_existed = False

        for i, primitive in enumerate(primitive_list):
            if primitive._vaid is None:
                self._add_primitive_to_context(primitive)
            self.vao_id[i] = primitive._vaid
            self.pose[i] = scene.get_pose(node_list[i])
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
            # The bounds the instances cover, refreshed with the instance transforms afterwards (see 'update_buffer')
            if primitive.poses is not None and len(primitive.poses):
                poses = primitive.poses
            else:
                poses = np.eye(4, dtype=np.float32)[np.newaxis]
            self._set_instance_bounds(np.array([i]), poses[np.newaxis])

            material = primitive.material
            tf = material.tex_flags
            self.textures[i, 0] = tf
            if tf & 1:
                self.textures[i, 1] = material.normalTexture._texid
            if tf & 2:
                self.textures[i, 2] = material.occlusiontexture._texid
            if tf & 4:
                self.textures[i, 3] = material.emissiveTexture._texid
            if tf & 8:
                self.textures[i, 4] = material.baseColorTexture._texid
            if tf & 16:
                self.textures[i, 5] = material.metallicRoughnessTexture._texid
            if tf & 32:
                self.textures[i, 6] = material.diffuseTexture._texid
            if tf & 64:
                self.textures[i, 7] = material.specularGlossinessTexture._texid
            if tf & 4 and material.emissiveTexture.tex_type == GL_TEXTURE_2D_ARRAY:
                self.textures[i, 0] |= 4 << 8
            if tf & 8 and material.baseColorTexture.tex_type == GL_TEXTURE_2D_ARRAY:
                self.textures[i, 0] |= 8 << 8

            if isinstance(material, MetallicRoughnessMaterial):
                self.pbr_mat[i, :4] = material.baseColorFactor
                self.pbr_mat[i, 4] = material.metallicFactor
                self.pbr_mat[i, 5] = material.roughnessFactor
                self.pbr_mat[i, 6:9] = material.emissiveFactor
                self.pbr_mat[i, 9] = material.transmissionFactor
                self.pbr_mat[i, 10] = material.ior
            elif isinstance(material, SpecularGlossinessMaterial):
                self.spec_mat[i, :4] = material.diffuseFactor
                self.spec_mat[i, 4:7] = material.specularFactor
                self.spec_mat[i, 7] = material.glossinessFactor
                self.spec_mat[i, 8:11] = material.emissiveFactor

            self.render_flags[i, 0] = material.alphaMode == "BLEND"
            self.render_flags[i, 1] = material.wireframe
            self.render_flags[i, 2] = material.doubleSided
            self.render_flags[i, 3] = isinstance(material, MetallicRoughnessMaterial)
            self.render_flags[i, 4] = primitive.is_floor and not floor_existed
            self.render_flags[i, 5] = node_list[i].mesh.is_transparent
            self.render_flags[i, 6] = node_list[i].mesh.is_marker
            # A mesh of translucent color casts a partial shadow through the transparent shadow programs alone (see
            # TRANSPARENT_SHADOW in renderer.py)
            self.render_flags[i, 9] = node_list[i].mesh.has_translucent_color
            self.render_flags[i, 10] = isinstance(material, MetallicRoughnessMaterial) and material.is_transmissive
            self.views[i] = node_list[i].mesh.views
            # A pass drawing one environment draws the instance of that environment of a per-environment instanced
            # primitive, and every instance of any other, which the culling below keeps to its environment
            self.render_flags[i, 7] = not primitive.is_env_instanced
            if primitive.is_env_instanced and not primitive.envs.all():
                self.render_flags[i, 8] = 1
                self.env_active[i, : len(primitive.envs)] = primitive.envs

            if primitive.is_floor:
                floor_existed = True

            self.mode[i] = primitive.mode
            self.n_instances[i] = len(primitive.poses) if primitive.poses is not None else 1
            self.n_indices[i] = primitive.indices.size if primitive.indices is not None else -len(primitive.positions)
            self.model_buffer_id[i] = primitive._buffers.get("model", 0)
            self.inst_tint_buffer_id[i] = primitive._buffers.get("inst_tint", 0)
            self.inst_attr_start[i] = getattr(primitive, "_inst_attr_start", 0)

        self.update_texture_layers([primitive_list[i] for i in np.flatnonzero(is_layered)])

        # Gate the per-env culling to scenes with heterogeneous variants or primitives standing in one environment
        self._has_env_filtered = bool(self.render_flags[:, 8].any() or (self.env_row >= 0).any())

        # The placed primitives are refreshed together, in groups of equal instance count so that their poses stack.
        # Their instance count is fixed by the buffers sized on it. A group without an instance has nothing to reduce.
        n_instances_placed = {}
        for i in np.flatnonzero(self.is_instance_placed):
            n_instances_placed.setdefault(len(primitive_list[i].poses), []).append(i)
        self._placed_groups = [
            PlacedGroup(np.array(idx), n_instances)
            for n_instances, idx in n_instances_placed.items()
            if n_instances > 0
        ]
        self._update_instances()

    @contextlib.contextmanager
    def _env_filtered_culling(self, env_idx, view_proj=None, is_grid=False):
        # Temporarily zero the index count of the primitives this pass does not draw, so their draw renders nothing.
        # These are the primitives absent from env_idx (heterogeneous variants, and those standing in another
        # environment) and those the kind of view drawn leaves out (see 'view'). Given the view-projection matrix of a
        # camera, they also include the primitives whose bounds lie outside its view volume. Mirrors the SKIP_MARKERS
        # approach used in forward_pass.
        inactive = (self.views & self.view) == 0
        corners = self.corners[is_grid]
        if view_proj is not None and corners is not None:
            # A box lies outside the view volume when its 8 corners lie outside one of its planes. A point inside has
            # clip coordinates with -w <= x, y, z <= w.
            clip = corners @ view_proj[:, :3].T + view_proj[:, 3]
            coords, w = clip[..., :3], clip[..., 3:]
            inactive |= ((coords > w).all(axis=1) | (coords < -w).all(axis=1)).any(axis=-1)
        if env_idx >= 0 and self._has_env_filtered:
            inactive |= (self.env_row >= 0) & (self.env_row != env_idx)
            # Only the env filtered primitives carry one visibility per environment. Primitives standing in a single
            # environment (env_row) alone may leave 'env_active' narrower than the rendered environments.
            is_env_filtered = self.render_flags[:, 8].astype(bool)
            if is_env_filtered.any():
                inactive[is_env_filtered] |= ~self.env_active[is_env_filtered, env_idx]
        if not inactive.any():
            yield
            return
        saved = self.n_indices[inactive].copy()
        self.n_indices[inactive] = 0
        try:
            yield
        finally:
            self.n_indices[inactive] = saved

    def load_programs(self, renderer, flags, program_flags):
        if (flags, program_flags) not in self.program_id:
            program_id = np.zeros_like(self.vao_id)
            for i, primitive in enumerate(self.primitive_list):
                prim_flags = flags
                # Markers are excluded from scene bounds, so shadow maps may not cover them. Disable shadow
                # reception while keeping full lighting.
                if self.render_flags[i, 6]:
                    prim_flags &= ~(
                        RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT | RenderFlags.SHADOWS_POINT
                    )
                # A single floor mirrors the scene, the one the reflection is set up for (see 'reflection_mat')
                if not self.render_flags[i, 4]:
                    prim_flags &= ~RenderFlags.REFLECTIVE_FLOOR
                program = renderer._get_primitive_program(primitive, prim_flags, program_flags)
                program_id[i] = program._program_id
            self.program_id[(flags, program_flags)] = program_id

    def gen_func_ptr(self):
        self.gl = GLWrapper()

        IS_OPENGL_42_AVAILABLE = hasattr(self.gl.wrapper_instance, "glDrawElementsInstancedBaseInstance")

        @nb.jit(
            nb.none(
                nb.int32[:],
                nb.int32[:],
                nb.float32[:, :, :],
                nb.int32[:, :],
                nb.float32[:, :],
                nb.float32[:, :],
                nb.int8[:, :],
                nb.int32[:],
                nb.int32[:],
                nb.int32[:],
                nb.float32[:, :],
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
                nb.int32[:],
                nb.int32[:],
                nb.int32[:],
                nb.boolean[:, :],
                nb.int32[:],
                nb.float32,
                nb.int32,
                nb.boolean[:],
                nb.int32,
                nb.float32[:, :, :],
                nb.int32[:],
                self.gl.wrapper_type,
            ),
            cache=True,
        )
        def forward_pass(
            vao_id,
            program_id,
            pose,
            textures,
            pbr_mat,
            spec_mat,
            render_flags,
            mode,
            n_instances,
            n_indices,
            light,
            shadow_map,
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
            model_buffer_id,
            inst_tint_buffer_id,
            inst_attr_start,
            env_active,
            draw_order,
            env_offset_scale,
            debug_view,
            is_env_instanced,
            env_offset_buffer,
            texture_layers,
            texture_layers_idx,
            gl,
        ):
            is_rgba = not (flags & RenderFlags_DEPTH_ONLY or flags & RenderFlags_SEG)

            # Every sampler left unset reads unit 0, which holds a placeholder until a material binds a texture there
            # (see bind_lighting). This also unbinds the mirror image that a floor pass draws into.
            gl.glActiveTexture(GL_TEXTURE0)
            gl.glBindTexture(GL_TEXTURE_2D, placeholder_tex[0])

            det_reflection = np.linalg.det(reflection_mat)
            last_pid = -1
            lighting_texture = 0

            for id in draw_order:
                # Only render markers on the main graphical window, while skipping plane-reflection. A primitive this
                # pass hides has no index to draw, and its state setup costs the driver as much as a draw.
                if (
                    n_indices[id] == 0
                    or ((render_flags[id, 4] or render_flags[id, 6]) and flags & RenderFlags_SKIP_FLOOR)
                    or (
                        render_flags[id, 6]
                        and (not is_rgba or (flags & RenderFlags_OFFSCREEN and _DISABLE_OFFSCREEN_MARKERS))
                    )
                ):
                    continue

                pid = program_id[id]
                if pid != last_pid:
                    gl.glUseProgram(pid)
                    if is_rgba and not flags & RenderFlags_FLAT:
                        # Strip shadow flags for markers — their programs have no shadow uniforms
                        light_flags = flags
                        if render_flags[id, 6]:
                            light_flags &= ~(
                                RenderFlags_SHADOWS_DIRECTIONAL | RenderFlags_SHADOWS_SPOT | RenderFlags_SHADOWS_POINT
                            )
                        lighting_texture = bind_lighting(
                            pid, light_flags, light, shadow_map, light_matrix, ambient_light, placeholder_tex, gl
                        )
                        set_uniform_3fv(pid, "cam_pos", cam_pos, gl)
                        set_uniform_matrix_4fv(pid, "reflection_mat", reflection_mat, gl)
                        # The instance tints show in the debug views alone (see 'Primitive.inst_tints')
                        if inst_tint_buffer_id[id]:
                            set_uniform_1i(pid, "debug_view", debug_view, gl)

                    set_uniform_matrix_4fv(pid, "V", mat_V, gl)
                    set_uniform_matrix_4fv(pid, "P", mat_P, gl)
                    set_uniform_1f(pid, "env_offset_scale", env_offset_scale, gl)

                    last_pid = pid

                active_texture = lighting_texture

                # The program of a reflective floor samples the mirror image (see REFLECTIVE_FLOOR in renderer.py)
                if flags & RenderFlags_REFLECTIVE_FLOOR and render_flags[id, 4]:
                    gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                    gl.glBindTexture(GL_TEXTURE_2D, floor_tex)
                    set_uniform_1i(pid, "floor_tex", active_texture, gl)
                    set_uniform_2f(pid, "screen_size", screen_size[0], screen_size[1], gl)
                    active_texture += 1

                set_uniform_matrix_4fv(pid, "M", pose[id], gl)
                gl.glBindVertexArray(vao_id[id])

                is_culled = not (render_flags[id, 2] or flags & RenderFlags_SKIP_CULL_FACES)
                if is_rgba and not flags & RenderFlags_FLAT:
                    tf = textures[id, 0]
                    texture_list = [
                        "normal_texture",
                        "occlusion_texture",
                        "emissive_texture",
                        "base_color_texture",
                        "metallic_roughness_texture",
                        "diffuse_texture",
                        "specular_glossiness_texture",
                    ]
                    # An unset sampler reads unit 0. Keep texture arrays off unit 0, since a sampler2D and a
                    # sampler2DArray reading one unit make the draw invalid.
                    if tf >> 8 and active_texture == 0:
                        active_texture = 1
                    for i in range(7):
                        if tf & (1 << i):
                            gl.glActiveTexture(GL_TEXTURE0 + active_texture)
                            is_array = tf & (1 << (i + 8))
                            gl.glBindTexture(GL_TEXTURE_2D_ARRAY if is_array else GL_TEXTURE_2D, textures[id, i + 1])
                            set_uniform_1i(pid, "material." + texture_list[i], active_texture, gl)
                            active_texture += 1

                    if render_flags[id, 3]:
                        set_uniform_4fv(pid, "material.base_color_factor", pbr_mat[id, :4], gl)
                        set_uniform_1f(pid, "material.metallic_factor", pbr_mat[id, 4], gl)
                        set_uniform_1f(pid, "material.roughness_factor", pbr_mat[id, 5], gl)
                        set_uniform_3fv(pid, "material.emissive_factor", pbr_mat[id, 6:9], gl)
                        set_uniform_1f(pid, "material.transmission_factor", pbr_mat[id, 9], gl)
                        set_uniform_1f(pid, "material.ior", pbr_mat[id, 10], gl)
                    else:
                        set_uniform_4fv(pid, "material.diffuse_factor", spec_mat[id, :4], gl)
                        set_uniform_3fv(pid, "material.specular_factor", spec_mat[id, 4:7], gl)
                        set_uniform_1f(pid, "material.roughness_factor", spec_mat[id, 7], gl)
                        set_uniform_3fv(pid, "material.glossiness_factor", spec_mat[id, 8:11], gl)

                    if render_flags[id, 10]:
                        # A transmissive material adds the light it reflects to the background, which its second output
                        # attenuates by the light it lets through (see 'frag_transmittance' in mesh.frag)
                        gl.glEnable(GL_BLEND)
                        gl.glBlendFunc(GL_ONE, GL_SRC1_COLOR)
                    elif render_flags[id, 0]:
                        # FIXME: Note that enabling GL_BLEND yield non-deterministic results on Nvidia GPU.
                        # As a result, it must only be enabled if absolutely necessary rather than systematically.
                        gl.glEnable(GL_BLEND)
                        # The alpha accumulates the coverage of the fragments, which the mirror image of a floor reads
                        # (see REFLECTIVE_FLOOR in mesh.frag)
                        gl.glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
                    else:
                        gl.glDisable(GL_BLEND)

                    wf = render_flags[id, 1]
                    if flags & RenderFlags_FLIP_WIREFRAME:
                        wf = not wf
                    if wf or flags & RenderFlags_ALL_WIREFRAME:
                        gl.glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    else:
                        gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                    if is_culled:
                        gl.glEnable(GL_CULL_FACE)
                        gl.glCullFace(GL_BACK if det_reflection > 0 else GL_FRONT)
                    else:
                        gl.glDisable(GL_CULL_FACE)
                else:
                    gl.glEnable(GL_CULL_FACE)
                    gl.glCullFace(GL_BACK if det_reflection > 0 else GL_FRONT)
                    gl.glDisable(GL_BLEND)
                    gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                gl.glDisable(GL_PROGRAM_POINT_SIZE)

                if flags & RenderFlags_SEG:
                    if color_list[id, 0] < -1:
                        gl.glBindVertexArray(0)
                        continue
                    set_uniform_3fv(pid, "color", color_list[id], gl)

                # A texture array is sampled at the layer of the environment drawn
                i_layers = texture_layers_idx[id]
                is_layer_set = i_layers >= 0 and is_rgba and not flags & RenderFlags_FLAT
                if is_layer_set and env_idx >= 0:
                    set_texture_layers(pid, textures[id, 0], texture_layers[i_layers, env_idx], gl)

                # A single-sided transmissive surface is drawn twice, back faces first and front faces over them, so
                # that the far side of a glass or of a body of water shows through its near side
                n_face_passes = 2 if is_rgba and render_flags[id, 10] and is_culled else 1
                for i_face_pass in range(n_face_passes):
                    if n_face_passes == 2:
                        is_front_culled = (i_face_pass == 0) == (det_reflection > 0)
                        gl.glCullFace(GL_FRONT if is_front_culled else GL_BACK)
                    if render_flags[id, 7] or env_idx == -1:
                        if (render_flags[id, 8] or is_layer_set) and env_idx == -1:
                            # Combined draw-all of a heterogeneous variant: draw only the environments it is active
                            # in, otherwise the variant would be drawn in every environment's instance. A primitive
                            # sampling a texture array is drawn one environment at a time too, each at its own layer.
                            for k in range(n_instances[id]):
                                if not env_active[id, k]:
                                    continue
                                if is_layer_set:
                                    set_texture_layers(pid, textures[id, 0], texture_layers[i_layers, k], gl)
                                if IS_OPENGL_42_AVAILABLE:
                                    if n_indices[id] > 0:
                                        gl.glDrawElementsInstancedBaseInstance(
                                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1, k
                                        )
                                    else:
                                        gl.glDrawArraysInstancedBaseInstance(mode[id], 0, -n_indices[id], 1, k)
                                else:
                                    gl.glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id[id])
                                    for j in range(4):
                                        gl.glVertexAttribPointer(
                                            inst_attr_start[id] + j, 4, GL_FLOAT, 0, 64, address_to_ptr(k * 64 + j * 16)
                                        )
                                    if is_env_instanced[id]:
                                        gl.glBindBuffer(GL_ARRAY_BUFFER, env_offset_buffer)
                                        gl.glVertexAttribPointer(
                                            inst_attr_start[id] + 4, 3, GL_FLOAT, 0, 12, address_to_ptr(k * 12)
                                        )
                                    if inst_tint_buffer_id[id]:
                                        gl.glBindBuffer(GL_ARRAY_BUFFER, inst_tint_buffer_id[id])
                                        gl.glVertexAttribPointer(
                                            inst_attr_start[id] + 5, 4, GL_FLOAT, 0, 16, address_to_ptr(k * 16)
                                        )
                                    if n_indices[id] > 0:
                                        gl.glDrawElementsInstanced(
                                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1
                                        )
                                    else:
                                        gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], 1)
                                    if inst_tint_buffer_id[id]:
                                        gl.glVertexAttribPointer(
                                            inst_attr_start[id] + 5, 4, GL_FLOAT, 0, 16, address_to_ptr(0)
                                        )
                                    if is_env_instanced[id]:
                                        gl.glBindBuffer(GL_ARRAY_BUFFER, env_offset_buffer)
                                        gl.glVertexAttribPointer(
                                            inst_attr_start[id] + 4, 3, GL_FLOAT, 0, 12, address_to_ptr(0)
                                        )
                                    gl.glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id[id])
                                    for j in range(4):
                                        gl.glVertexAttribPointer(
                                            inst_attr_start[id] + j, 4, GL_FLOAT, 0, 64, address_to_ptr(j * 16)
                                        )
                        elif n_indices[id] > 0:
                            gl.glDrawElementsInstanced(
                                mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), n_instances[id]
                            )
                        else:
                            gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], n_instances[id])
                    elif IS_OPENGL_42_AVAILABLE:
                        if n_indices[id] > 0:
                            gl.glDrawElementsInstancedBaseInstance(
                                mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1, env_idx
                            )
                        else:
                            gl.glDrawArraysInstancedBaseInstance(mode[id], 0, -n_indices[id], 1, env_idx)
                    else:
                        # OpenGL 4.1 fallback: rebind instance attributes with offset
                        gl.glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id[id])
                        for j in range(4):
                            gl.glVertexAttribPointer(
                                inst_attr_start[id] + j, 4, GL_FLOAT, 0, 64, address_to_ptr(env_idx * 64 + j * 16)
                            )
                        if inst_tint_buffer_id[id]:
                            gl.glBindBuffer(GL_ARRAY_BUFFER, inst_tint_buffer_id[id])
                            gl.glVertexAttribPointer(
                                inst_attr_start[id] + 5, 4, GL_FLOAT, 0, 16, address_to_ptr(env_idx * 16)
                            )
                        if n_indices[id] > 0:
                            gl.glDrawElementsInstanced(mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1)
                        else:
                            gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], 1)
                        # Restore default attribute pointer (offset 0) to avoid corrupting VAO state
                        if inst_tint_buffer_id[id]:
                            gl.glVertexAttribPointer(inst_attr_start[id] + 5, 4, GL_FLOAT, 0, 16, address_to_ptr(0))
                            gl.glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id[id])
                        for j in range(4):
                            gl.glVertexAttribPointer(
                                inst_attr_start[id] + j, 4, GL_FLOAT, 0, 64, address_to_ptr(j * 16)
                            )

                gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glFlush()

        @nb.jit(
            nb.none(
                nb.int32[:],
                nb.int32[:],
                nb.float32[:, :, :],
                nb.int32[:],
                nb.int32[:],
                nb.int32[:],
                nb.float32[:, :],
                nb.float32[:, :],
                nb.float32[:, :, :],
                nb.float32[:],
                nb.boolean,
                nb.int8[:, :],
                nb.int32,
                nb.int32[:],
                nb.int32[:],
                nb.boolean[:, :],
                nb.float32,
                nb.boolean[:],
                nb.int32,
                nb.int32[:, :],
                nb.float32[:, :],
                nb.float32[:, :, :],
                nb.int32[:],
                self.gl.wrapper_type,
            ),
            cache=True,
        )
        def shadow_mapping_pass(
            vao_id,
            program_id,
            pose,
            mode,
            n_instances,
            n_indices,
            mat_V,
            mat_P,
            light_matrix,
            light_pos,
            is_point,
            render_flags,
            env_idx,
            model_buffer_id,
            inst_attr_start,
            env_active,
            env_offset_scale,
            is_env_instanced,
            env_offset_buffer,
            textures,
            pbr_mat,
            texture_layers,
            texture_layers_idx,
            gl,
        ):
            last_pid = -1
            for id in range(len(vao_id)):
                # Markers cast no shadow, and neither does a translucent color drawn without a transparent shadow (see
                # TRANSPARENT_SHADOW in renderer.py)
                if render_flags[id, 6] or (render_flags[id, 9] and not render_flags[id, 3]):
                    continue

                pid = program_id[id]
                if pid != last_pid:
                    gl.glUseProgram(pid)

                    # A point light renders the six faces of its cube map at once (see point_shadow.geom)
                    if is_point:
                        for i in range(6):
                            set_uniform_matrix_4fv(pid, "light_matrix[" + str(i) + "]", light_matrix[i], gl)
                        set_uniform_3fv(pid, "light_pos", light_pos, gl)
                    else:
                        set_uniform_matrix_4fv(pid, "V", mat_V, gl)
                        set_uniform_matrix_4fv(pid, "P", mat_P, gl)
                    set_uniform_1f(pid, "env_offset_scale", env_offset_scale, gl)

                    last_pid = pid

                set_uniform_matrix_4fv(pid, "M", pose[id], gl)
                gl.glBindVertexArray(vao_id[id])

                # A transparent surface casts a shadow as dark as it is opaque (see mesh_depth.frag), and so does a
                # texture array whatever the layer shown (see TRANSPARENT_SHADOW in renderer.py). A texture array is
                # sampled at the layer of the environment drawn.
                i_layers = texture_layers_idx[id]
                is_layer_set = False
                tf = textures[id, 0]
                if render_flags[id, 5] or tf & (8 << 8):
                    set_uniform_4fv(pid, "base_color_factor", pbr_mat[id, :4], gl)
                    set_uniform_1f(pid, "transmission_factor", pbr_mat[id, 9], gl)
                    if tf & 8:
                        gl.glActiveTexture(GL_TEXTURE0)
                        gl.glBindTexture(GL_TEXTURE_2D_ARRAY if tf & (8 << 8) else GL_TEXTURE_2D, textures[id, 4])
                        set_uniform_1i(pid, "base_color_texture", 0, gl)
                        is_layer_set = i_layers >= 0
                        if is_layer_set and env_idx >= 0:
                            set_uniform_1f(pid, "base_color_layer", texture_layers[i_layers, env_idx, 0], gl)

                gl.glEnable(GL_CULL_FACE)
                gl.glCullFace(GL_BACK)
                gl.glDisable(GL_BLEND)
                gl.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                gl.glDisable(GL_PROGRAM_POINT_SIZE)

                if render_flags[id, 7] or env_idx == -1:
                    if (render_flags[id, 8] or is_layer_set) and env_idx == -1:
                        # Combined draw-all of a heterogeneous variant: shadow only the environments it is active in,
                        # otherwise the variant would cast a shadow in every environment's instance. A primitive
                        # sampling a texture array casts the shadow of one environment at a time, each at its own layer.
                        for k in range(n_instances[id]):
                            if not env_active[id, k]:
                                continue
                            if is_layer_set:
                                set_uniform_1f(pid, "base_color_layer", texture_layers[i_layers, k, 0], gl)
                            if IS_OPENGL_42_AVAILABLE:
                                if n_indices[id] > 0:
                                    gl.glDrawElementsInstancedBaseInstance(
                                        mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1, k
                                    )
                                else:
                                    gl.glDrawArraysInstancedBaseInstance(mode[id], 0, -n_indices[id], 1, k)
                            else:
                                gl.glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id[id])
                                for j in range(4):
                                    gl.glVertexAttribPointer(
                                        inst_attr_start[id] + j, 4, GL_FLOAT, 0, 64, address_to_ptr(k * 64 + j * 16)
                                    )
                                if is_env_instanced[id]:
                                    gl.glBindBuffer(GL_ARRAY_BUFFER, env_offset_buffer)
                                    gl.glVertexAttribPointer(
                                        inst_attr_start[id] + 4, 3, GL_FLOAT, 0, 12, address_to_ptr(k * 12)
                                    )
                                if n_indices[id] > 0:
                                    gl.glDrawElementsInstanced(
                                        mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1
                                    )
                                else:
                                    gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], 1)
                                if is_env_instanced[id]:
                                    gl.glVertexAttribPointer(
                                        inst_attr_start[id] + 4, 3, GL_FLOAT, 0, 12, address_to_ptr(0)
                                    )
                                    gl.glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id[id])
                                for j in range(4):
                                    gl.glVertexAttribPointer(
                                        inst_attr_start[id] + j, 4, GL_FLOAT, 0, 64, address_to_ptr(j * 16)
                                    )
                    elif n_indices[id] > 0:
                        gl.glDrawElementsInstanced(
                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), n_instances[id]
                        )
                    else:
                        gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], n_instances[id])
                elif IS_OPENGL_42_AVAILABLE:
                    if n_indices[id] > 0:
                        gl.glDrawElementsInstancedBaseInstance(
                            mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1, env_idx
                        )
                    else:
                        gl.glDrawArraysInstancedBaseInstance(mode[id], 0, -n_indices[id], 1, env_idx)
                else:
                    gl.glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id[id])
                    for j in range(4):
                        gl.glVertexAttribPointer(
                            inst_attr_start[id] + j, 4, GL_FLOAT, 0, 64, address_to_ptr(env_idx * 64 + j * 16)
                        )
                    if n_indices[id] > 0:
                        gl.glDrawElementsInstanced(mode[id], n_indices[id], GL_UNSIGNED_INT, address_to_ptr(0), 1)
                    else:
                        gl.glDrawArraysInstanced(mode[id], 0, -n_indices[id], 1)
                    # Restore default attribute pointer (offset 0) to avoid corrupting VAO state
                    for j in range(4):
                        gl.glVertexAttribPointer(inst_attr_start[id] + j, 4, GL_FLOAT, 0, 64, address_to_ptr(j * 16))

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

        @nb.jit(nb.uint8[:, :, :](nb.int32, nb.int32, nb.int32, self.gl.wrapper_type), cache=True)
        def read_color_buf(width, height, rgba, gl):
            if rgba:
                buf = np.zeros((height, width, 4), np.uint8)
                gl.glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, address_to_ptr(buf.ctypes.data))
            else:
                buf = np.zeros((height, width, 3), np.uint8)
                gl.glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, address_to_ptr(buf.ctypes.data))
            return buf[::-1, :, :]

        @nb.jit(nb.float32[:, :](nb.float32[:, :, :]), cache=True)
        def update_normal_flat(p):
            face_normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            vertex_normal = np.zeros((p.shape[0] * 3, 3), p.dtype)
            for f in range(face_normal.shape[0]):
                n = face_normal[f]
                n /= np.linalg.norm(n)
                vertex_normal[f * 3 + 0] = n
                vertex_normal[f * 3 + 1] = n
                vertex_normal[f * 3 + 2] = n
            return vertex_normal

        @nb.jit(nb.float32[:, :](nb.float32[:, :], nb.int32[:, :]), cache=True)
        def update_normal_smooth(p, idx):
            face_normal = np.cross(p[idx[:, 1]] - p[idx[:, 0]], p[idx[:, 2]] - p[idx[:, 0]])
            vertex_normal = np.zeros_like(p)
            for f in range(face_normal.shape[0]):
                vertex_normal[idx[f, 0]] += face_normal[f]
                vertex_normal[idx[f, 1]] += face_normal[f]
                vertex_normal[idx[f, 2]] += face_normal[f]
            for v in range(vertex_normal.shape[0]):
                vertex_normal[v] /= np.linalg.norm(vertex_normal[v])
            return vertex_normal

        @nb.jit(nb.none(nb.int64[:, :], self.gl.wrapper_type), cache=True)
        def update_buffer(updates, gl):
            for i in range(updates.shape[0]):
                buffer_id = updates[i, 0]
                buffer_size = updates[i, 1]
                buffer_addr = updates[i, 2]
                if buffer_id >= 0:
                    gl.glBindBuffer(GL_ARRAY_BUFFER, buffer_id)

                    gl.glBufferData(GL_ARRAY_BUFFER, buffer_size, address_to_ptr(0), GL_STREAM_DRAW)
                    gl.glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_size, address_to_ptr(buffer_addr))

                    gl.glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._forward_pass = forward_pass
        self._shadow_mapping_pass = shadow_mapping_pass
        self._read_depth_buf = read_depth_buf
        self._read_color_buf = read_color_buf
        self._update_normal_flat = update_normal_flat
        self._update_normal_smooth = update_normal_smooth
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
        if self._forward_pass is None:
            self.gen_func_ptr()
        if self._placeholder_textures is None:
            # A 2D color texture, a 2D depth texture compared as a shadow map, and a depth cube map (see bind_lighting)
            self._placeholder_textures = (
                Texture(source=np.full((1, 1, 4), 255, np.uint8), source_channels="RGBA"),
                Texture(width=1, height=1, source_channels="D", data_format=GL.GL_UNSIGNED_SHORT),
                CubeMapTexture(1, 1),
            )
            for texture in self._placeholder_textures:
                texture._add_to_context()
            self.placeholder_tex = np.array([texture._texid for texture in self._placeholder_textures], np.int32)
        # Temporarily hide markers for non-debug offscreen cameras by setting their
        # index count to 0, so draw calls render nothing for these nodes.
        if flags & RenderFlags.SKIP_MARKERS:
            marker_mask = self.render_flags[:, 6].astype(bool)
            saved_n_indices = self.n_indices[marker_mask].copy()
            self.n_indices[marker_mask] = 0
        # Hide everything except markers for the X-ray pass
        if markers_only:
            non_marker_mask = ~self.render_flags[:, 6].astype(bool)
            saved_non_marker_indices = self.n_indices[non_marker_mask].copy()
            self.n_indices[non_marker_mask] = 0

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
        is_transparent = self.render_flags[:, 5].astype(bool)
        # Depth along the view axis grows towards the viewer, so ascending depth is farthest first
        draw_order = np.lexsort((np.where(is_transparent, depths, -depths), is_transparent)).astype(np.int32)

        with self._env_filtered_culling(env_idx, P @ V, not is_env_pass(flags)):
            self._forward_pass(
                self.vao_id,
                self.program_id[(flags, program_flags)],
                self.pose,
                self.textures,
                self.pbr_mat,
                self.spec_mat,
                self.render_flags,
                self.mode,
                self.n_instances,
                self.n_indices,
                self.light,
                self.shadow_map,
                self.light_matrix,
                self.ambient_light,
                np.ascontiguousarray(V, dtype=np.float32),
                np.ascontiguousarray(P, dtype=np.float32),
                np.ascontiguousarray(cam_pos, dtype=np.float32),
                flags,
                color_list if flags & RenderFlags.SEG else self.pbr_mat,
                reflection_mat,
                floor_tex,
                self.placeholder_tex,
                screen_size,
                env_idx,
                self.model_buffer_id,
                self.inst_tint_buffer_id,
                self.inst_attr_start,
                self.env_active,
                draw_order,
                0.0 if is_env_pass(flags) else 1.0,
                int(self.view == ViewFlags.DEBUG),
                self.is_env_instanced,
                self.env_offset_buffer if self.env_offset_buffer is not None else 0,
                self.texture_layers,
                self.texture_layers_idx,
                self.gl.wrapper_instance,
            )
        if flags & RenderFlags.SKIP_MARKERS:
            self.n_indices[marker_mask] = saved_n_indices
        if markers_only:
            self.n_indices[non_marker_mask] = saved_non_marker_indices

    def shadow_mapping_pass(self, renderer, V, P, flags, program_flags, env_idx=-1, light_matrix=None, light_pos=None):
        """Render the shadow map of a light, the depth seen from it or for a point light its distance to the scene.

        Parameters
        ----------
        renderer : Renderer
            The renderer whose shader programs draw the pass.
        V : (4,4) float | None
            The view matrix of the light, None for a point light.
        P : (4,4) float | None
            The projection matrix of the light, None for a point light.
        flags : int
            The render flags of the pass.
        program_flags : int
            The program flags of the pass.
        env_idx : int, optional
            The environment drawn, -1 for all of them. Defaults to -1.
        light_matrix : (6,4,4) float | None, optional
            The view-projection matrix of each face of the cube map of a point light. Defaults to None.
        light_pos : (3,) float | None, optional
            The position of a point light, None for any other light. Defaults to None.
        """
        self.load_programs(renderer, flags, program_flags)
        if self._shadow_mapping_pass is None:
            self.gen_func_ptr()
        is_point = light_pos is not None
        identity = np.identity(4, dtype=np.float32)
        with self._env_filtered_culling(env_idx):
            self._shadow_mapping_pass(
                self.vao_id,
                self.program_id[(flags, program_flags)],
                self.pose,
                self.mode,
                self.n_instances,
                self.n_indices,
                identity if is_point else np.ascontiguousarray(V, dtype=np.float32),
                identity if is_point else np.ascontiguousarray(P, dtype=np.float32),
                np.ascontiguousarray(light_matrix, dtype=np.float32) if is_point else identity[None],
                np.ascontiguousarray(light_pos, dtype=np.float32) if is_point else identity[0, :3],
                is_point,
                self.render_flags,
                env_idx,
                self.model_buffer_id,
                self.inst_attr_start,
                self.env_active,
                0.0 if is_env_pass(flags) else 1.0,
                self.is_env_instanced,
                self.env_offset_buffer if self.env_offset_buffer is not None else 0,
                self.textures,
                self.pbr_mat,
                self.texture_layers,
                self.texture_layers_idx,
                self.gl.wrapper_instance,
            )

    def update_normal(self, node, vertices):
        primitive = node.mesh.primitives[0]
        if primitive.normals is None:
            return None
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        if primitive.indices is not None:
            if self._update_normal_smooth is None:
                self.gen_func_ptr()
            return self._update_normal_smooth(vertices, primitive.indices)
        else:
            if self._update_normal_flat is None:
                self.gen_func_ptr()
            return self._update_normal_flat(vertices.reshape((-1, 3, 3)))

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
                if self.textures[i, 0] & (8 << 8):
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
                    self.render_flags[i_p, 5] = is_transparent

    def update_buffer(self, primitives, buffer_name, data):
        """Queue the update of one buffer of several primitives, flushed during the next render pass.

        New vertex positions update the bounds of the bare geometry, and new instance transforms update the bounds of
        the instances. A placed primitive refreshes the latter from its poses in bulk instead (see '_update_instances').

        The GL buffer id is resolved at flush time rather than here: a primitive is only added to the GL context during
        its first render pass, so an id resolved at enqueue time would be -1 for any update queued before that pass and
        the update would be silently dropped.

        Parameters
        ----------
        primitives : Sequence[Primitive]
            The primitives.
        buffer_name : str
            The buffer, such as 'pos', 'normal', or 'model' for the instance transforms, uploaded transposed, the
            translation in the last row.
        data : Sequence[np.ndarray]
            The new content of the buffer of each primitive.
        """
        primitives_idx, poses = [], []
        for primitive, primitive_data in zip(primitives, data):
            self._buffer[(primitive, buffer_name)] = primitive_data
            i = self._primitive_index.get(primitive)
            if i is not None and len(primitive_data):
                if buffer_name == "pos":
                    self.bounds_geom[i, 0] = primitive_data.min(axis=0)
                    self.bounds_geom[i, 1] = primitive_data.max(axis=0)
                    # The bounds of the instances follow the new box. A placed primitive refreshes them in bulk.
                    if not self.is_instance_placed[i]:
                        instance_poses = primitive.poses
                        if instance_poses is None or not len(instance_poses):
                            instance_poses = np.eye(4, dtype=np.float32)[np.newaxis]
                        self._set_instance_bounds(np.array([i]), instance_poses[np.newaxis])
                elif buffer_name == "model" and not self.is_instance_placed[i]:
                    primitives_idx.append(i)
                    poses.append(primitive_data.swapaxes(-1, -2))
        if primitives_idx:
            self._set_instance_bounds(np.array(primitives_idx), np.stack(poses))
        self._scene.bump_revision()

    def flush_buffer(self):
        """Upload all queued buffer updates to the GPU and clear the queue."""
        if not self._buffer:
            return

        updates = np.zeros((len(self._buffer), 3), dtype=np.int64)
        buffers = []
        for idx, ((primitive, buffer_name), data) in enumerate(self._buffer.items()):
            buffer = np.ascontiguousarray(data, dtype=np.float32)
            buffers.append(buffer)

            # Still -1 if the primitive was removed from the scene since the update was queued. The upload kernel
            # skips negative ids.
            updates[idx, 0] = primitive.get_buffer_id(buffer_name)
            updates[idx, 1] = 4 * buffer.size
            updates[idx, 2] = buffer.ctypes.data

        if self._update_buffer_fn is None:
            self.gen_func_ptr()
        self._update_buffer_fn(updates, self.gl.wrapper_instance)
        self._buffer.clear()

    def read_depth_buf(self, weight, height, z_near, z_far):
        if self._read_depth_buf is None:
            self.gen_func_ptr()
        return self._read_depth_buf(weight, height, z_near, z_far, self.gl.wrapper_instance)

    def read_color_buf(self, weight, height, rgba):
        if self._read_color_buf is None:
            self.gen_func_ptr()
        return self._read_color_buf(weight, height, rgba, self.gl.wrapper_instance)
