"""Primitives of the rasterizer, as glTF 2.0 defines them."""

import numba as nb
import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_POINTS,
    GL_STATIC_DRAW,
    GL_STREAM_DRAW,
    GL_TRIANGLES,
    OpenGL,
    ctypes,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glVertexAttribDivisor,
    glVertexAttribPointer,
)

from .constants import FLOAT_SZ, UINT_SZ, BufFlags
from .material import Material
from .utils import format_color_array, transform_box


@nb.jit(nopython=True, cache=True)
def _compute_bounds(positions):
    bounds = np.zeros((2, 3))
    if len(positions) > 0:
        for i in range(3):
            bounds[0, i] = np.min(positions[:, i])
            bounds[1, i] = np.max(positions[:, i])
    return bounds


@nb.jit(nopython=True, cache=True)
def _compute_flat_normals(triangles):
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    vertex_normals = np.zeros((3 * len(triangles), 3), triangles.dtype)
    for i_f in range(len(triangles)):
        normal = face_normals[i_f] / max(np.linalg.norm(face_normals[i_f]), 1e-10)
        for i_v in range(3):
            vertex_normals[3 * i_f + i_v] = normal
    return vertex_normals


@nb.jit(nopython=True, cache=True)
def _compute_smooth_normals(positions, faces):
    # A vertex shared by several faces takes the sum of their normals, each weighted by the area of its face
    face_normals = np.cross(
        positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]]
    )
    vertex_normals = np.zeros_like(positions)
    for i_f in range(len(faces)):
        for i_v in range(3):
            vertex_normals[faces[i_f, i_v]] += face_normals[i_f]
    for i_v in range(len(positions)):
        vertex_normals[i_v] /= max(np.linalg.norm(vertex_normals[i_v]), 1e-10)
    return vertex_normals


class Primitive:
    """A set of vertices and the shapes they form, drawn with one material, possibly in several instances.

    Parameters
    ----------
    positions : (n, 3) float
        The positions of the vertices.
    normals : (n, 3) float, optional
        The normals of the vertices, None for a primitive drawn without lighting from its normals. The renderer draws
        the normals that the faces give, flat or smooth as the indices share the vertices. Defaults to None.
    texcoord_0 : (n, 2) float, optional
        The texture coordinates of the vertices. Defaults to None.
    color_0 : (n, 4) float, optional
        The RGBA colors of the vertices, scaling the base color of the material. Defaults to None.
    indices : (m, 3) int, optional
        The vertices of each face, None for vertices that form the faces in order. Defaults to None.
    material : Material, optional
        The material of the primitive. Defaults to a default Material.
    mode : int, optional
        The kind of shapes the vertices form, GL_POINTS or GL_TRIANGLES. Defaults to GL_TRIANGLES.
    poses : (k, 4, 4) float, optional
        The pose of each instance, None for a single instance at the pose of the node. Defaults to None.
    vertex_mapping : (n,) int, optional
        The vertex of the source mesh that each vertex copies, the vertices of a flat-shaded mesh being split per
        face. Defaults to None.
    double_sided : bool, optional
        Whether both faces of each triangle are drawn and lit. Defaults to False.
    is_floor : bool, optional
        Whether the primitive is the floor, which the plane reflection mirrors the scene through. Defaults to False.
    envs : int or (n_envs,) bool, optional
        The environments the instances belong to, among the rendered ones: None for a primitive drawn as is in every
        environment, an int for a primitive whose instances all stand in that environment, and a mask over the rendered
        environments for a primitive with one instance per environment, marking the environments it exists in. A pass
        drawing the environments side by side moves each instance by the offset of its environment (see
        'JITRenderer.env_offset_buffer'), and a pass drawing one environment draws the instances standing in it.
        Defaults to None.
    inst_tints : (k, 4) float, optional
        The RGBA tint of every instance, which the debug views alone show (see ViewFlags). The base color of the
        material, textures included, takes the hue of the rgb of the tint and keeps its own brightness, the tint giving
        its own color to a mid-grey material. The alpha weighs the tint, a zero alpha drawing the instance as is.
        Streamed like the poses. Defaults to None.
    """

    def __init__(
        self,
        positions,
        normals=None,
        texcoord_0=None,
        color_0=None,
        indices=None,
        material=None,
        mode=GL_TRIANGLES,
        poses=None,
        vertex_mapping=None,
        double_sided=False,
        is_floor=False,
        envs=None,
        inst_tints=None,
    ):
        assert mode in (GL_POINTS, GL_TRIANGLES)
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)
        self.normals = None if normals is None else np.ascontiguousarray(normals, dtype=np.float32)
        assert self.normals is None or self.normals.shape == self.positions.shape
        self.texcoord_0 = None if texcoord_0 is None else np.ascontiguousarray(texcoord_0[:, :2], dtype=np.float32)
        self.color_0 = None if color_0 is None else format_color_array(color_0, shape=(len(self.positions), 4))
        self._has_translucent_colors = self.color_0 is not None and bool((self.color_0[:, 3] < 1.0).any())
        self.indices = None if indices is None else np.ascontiguousarray(indices, dtype=np.int32)
        self.material = Material() if material is None else material
        self.mode = mode
        self.vertex_mapping = vertex_mapping
        self.double_sided = double_sided
        self.is_floor = is_floor
        self.envs = envs
        self.texture_layers: np.ndarray | None = None
        """(n_envs, 2) float : For a material whose base color or emissive texture is a texture array, the layers of the
        base color (column 0) and of the emissive (column 1) array that each rendered environment samples, else None."""

        self._bounds = None
        self._bounds_0 = None
        self.poses = poses
        self.inst_tints = inst_tints

        self._vaid = None
        # The first attribute location of the instance transforms, set as the primitive is added to the context
        self._inst_attr_start = None
        self._buffers = {}

    @property
    def poses(self):
        """(k, 4, 4) float : The pose of each instance, None for a single instance at the pose of the node."""
        return self._poses

    @poses.setter
    def poses(self, value):
        if value is not None:
            value = np.ascontiguousarray(value, dtype=np.float32)
            if value.ndim == 2:
                value = value[np.newaxis]
            assert value.shape[1:] == (4, 4)
        self._poses = value
        self._bounds = None

    @property
    def inst_tints(self):
        """(k, 4) float : The RGBA tint of every instance, None to draw them as they are."""
        return self._inst_tints

    @inst_tints.setter
    def inst_tints(self, value):
        if value is not None:
            value = np.ascontiguousarray(value, dtype=np.float32)
            assert value.ndim == 2 and value.shape[1] == 4
        self._inst_tints = value

    @property
    def bounds(self):
        """(2, 3) float : The axis-aligned bounds of the primitive, over its instances."""
        if self._bounds is None:
            if self._bounds_0 is None:
                self._bounds_0 = _compute_bounds(self.positions)
            self._bounds = self._bounds_0 if self.poses is None else transform_box(self._bounds_0, self.poses)
        return self._bounds

    @property
    def centroid(self):
        """(3,) float : The centre of the axis-aligned bounds of the primitive."""
        return np.mean(self.bounds, axis=0)

    @property
    def scale(self):
        """float : The length of the diagonal of the axis-aligned bounds of the primitive."""
        return max(np.linalg.norm(self.bounds[1] - self.bounds[0]), 1e-7)

    @property
    def buf_flags(self):
        """int : The vertex buffers of the primitive (see BufFlags)."""
        buf_flags = BufFlags.POSITION
        for data, flag in (
            (self.normals, BufFlags.NORMAL),
            (self.texcoord_0, BufFlags.TEXCOORD_0),
            (self.color_0, BufFlags.COLOR_0),
        ):
            if data is not None:
                buf_flags |= flag
        return buf_flags

    @property
    def is_transparent(self):
        """bool : Whether the primitive shows part of what lies behind it."""
        return self.material.is_transparent or self._has_translucent_colors

    @property
    def is_env_instanced(self):
        """bool : Whether instance k is the copy of environment k (see 'envs')."""
        return isinstance(self.envs, np.ndarray)

    @property
    def env_idx(self):
        """int : The environment every instance stands in, None when they are shared or one per environment."""
        return self.envs if isinstance(self.envs, (int, np.integer)) else None

    @property
    def buffers_id(self):
        """dict of str to int : The GL buffer holding each attribute of the primitive in the GL context, by name."""
        return self._buffers

    def compute_normals(self, positions):
        """Return the normals that the faces give to the vertices at some positions.

        Parameters
        ----------
        positions : (n, 3) float
            The positions of the vertices of the primitive.

        Returns
        -------
        normals : (n, 3) float
            The normal of the face of each vertex when the vertices form the faces in order, the normals of the faces
            sharing each vertex averaged by their area otherwise.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if self.indices is None:
            return _compute_flat_normals(positions.reshape((-1, 3, 3)))
        return _compute_smooth_normals(positions, self.indices.reshape((-1, 3)))

    def delete(self):
        """Remove the primitive from the OpenGL context."""
        glBindVertexArray(0)
        self._remove_from_context()

    def _add_to_context(self, env_offset_buffer=None):
        """Upload the primitive to the GL context.

        'env_offset_buffer' is the buffer of environment offsets to read the instance offset attribute from. Left
        unbound, the attribute holds the zero offset.
        """
        if self._vaid is not None:
            return

        self._vaid = glGenVertexArrays(1)
        glBindVertexArray(self._vaid)

        # The positions and the normals stream, since they follow the simulation, while the other attributes share one
        # static buffer. The attribute locations follow the order of 'BufFlags' (see the defines of the programs in
        # renderer.py).
        for location, (buffer_name, data) in enumerate(
            (
                ("pos", self.positions),
                ("normal", None if self.normals is None else self.compute_normals(self.positions)),
            )
        ):
            if data is None:
                continue
            buffer = glGenBuffers(1)
            self._buffers[buffer_name] = buffer
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, np.ascontiguousarray(data, dtype=np.float32), GL_STREAM_DRAW)
            glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, FLOAT_SZ * 3, ctypes.c_void_p(0))
            glEnableVertexAttribArray(location)

        static_attrs = [data for data in (self.texcoord_0, self.color_0) if data is not None]
        location = 1 if self.normals is None else 2
        if static_attrs:
            vertex_data = np.ascontiguousarray(np.hstack(static_attrs), dtype=np.float32)
            vertex_buffer = glGenBuffers(1)
            self._buffers["vertex"] = vertex_buffer
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
            glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
            stride = FLOAT_SZ * vertex_data.shape[1]
            offset = 0
            for data in static_attrs:
                size = data.shape[1]
                glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(FLOAT_SZ * offset))
                glEnableVertexAttribArray(location)
                location += 1
                offset += size

        # The transforms of the instances take the next 4 attribute locations, which the renderer drawing the primitive
        # points at its rows of the instance buffer (see 'JITRenderer.instances_pose')
        self._inst_attr_start = location

        if env_offset_buffer is not None:
            glBindBuffer(GL_ARRAY_BUFFER, env_offset_buffer)
            glEnableVertexAttribArray(location + 4)
            if self.is_env_instanced:
                glVertexAttribPointer(location + 4, 3, GL_FLOAT, GL_FALSE, FLOAT_SZ * 3, ctypes.c_void_p(0))
                glVertexAttribDivisor(location + 4, 1)
            else:
                # Every instance is in one environment: the attribute starts at its row and advances past the last
                # instance only, so every instance reads that row
                n_instances = 1 if self.poses is None else len(self.poses)
                env_offset = ctypes.c_void_p(FLOAT_SZ * 3 * self.env_idx)
                glVertexAttribPointer(location + 4, 3, GL_FLOAT, GL_FALSE, FLOAT_SZ * 3, env_offset)
                glVertexAttribDivisor(location + 4, max(n_instances, 1))

        if self.inst_tints is not None:
            tint_buffer = glGenBuffers(1)
            self._buffers["inst_tint"] = tint_buffer
            glBindBuffer(GL_ARRAY_BUFFER, tint_buffer)
            glBufferData(GL_ARRAY_BUFFER, self.inst_tints.nbytes, self.inst_tints, GL_STREAM_DRAW)
            glEnableVertexAttribArray(location + 5)
            glVertexAttribPointer(location + 5, 4, GL_FLOAT, GL_FALSE, FLOAT_SZ * 4, ctypes.c_void_p(0))
            glVertexAttribDivisor(location + 5, 1)

        if self.indices is not None:
            element_buffer = glGenBuffers(1)
            self._buffers["element"] = element_buffer
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
            indices = np.ascontiguousarray(self.indices, dtype=np.uint32)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, UINT_SZ * indices.size, indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def _remove_from_context(self):
        if self._vaid is not None:
            try:
                glDeleteVertexArrays(1, [self._vaid])
                glDeleteBuffers(len(self._buffers), list(self._buffers.values()))
            except OpenGL.error.Error:
                pass
            self._vaid = None
            self._buffers.clear()

    def _in_context(self):
        return self._vaid is not None
