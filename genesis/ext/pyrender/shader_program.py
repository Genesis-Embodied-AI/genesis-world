"""Shader programs of the rasterizer."""

import os
import re

import numpy as np

import OpenGL
from OpenGL.GL import (
    GL_FRAGMENT_SHADER,
    GL_GEOMETRY_SHADER,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glBindVertexArray,
    glDeleteProgram,
    glDeleteVertexArrays,
    glGenVertexArrays,
    glGetUniformLocation,
    glUniform1i,
    glUniform4fv,
    glUniformMatrix4fv,
    glUseProgram,
    shaders,
)

from .constants import UNIFORMS_NAME

SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")


class ShaderProgramCache:
    """The shader programs built in each OpenGL context, for each set of shaders and defines."""

    def __init__(self):
        self._program_cache = {}

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        """Return the program built from shaders of the 'shaders' folder, in the current OpenGL context.

        Parameters
        ----------
        vertex_shader : str
            The file of the vertex shader.
        fragment_shader : str
            The file of the fragment shader.
        geometry_shader : str, optional
            The file of the geometry shader. Defaults to none.
        defines : dict of str to int, optional
            The macros of the shaders and their values. Defaults to none.

        Returns
        -------
        program : ShaderProgram
            The program, which the caller adds to the context.
        """
        if defines is None:
            defines = {}
        key = (
            OpenGL.contextdata.getContext(),
            vertex_shader,
            fragment_shader,
            geometry_shader,
            *sorted(defines.items()),
        )
        if key not in self._program_cache:
            self._program_cache[key] = ShaderProgram(
                vertex_shader=os.path.join(SHADER_DIR, vertex_shader),
                fragment_shader=os.path.join(SHADER_DIR, fragment_shader),
                geometry_shader=None if geometry_shader is None else os.path.join(SHADER_DIR, geometry_shader),
                defines=defines,
            )
        return self._program_cache[key]

    def clear(self):
        """Free the programs of every context."""
        for program in self._program_cache.values():
            try:
                program.delete()
            except OpenGL.error.GLError:
                pass
        self._program_cache.clear()


class ShaderProgram:
    """An OpenGL shader program built from shader files.

    The shaders take each define as a macro: the preprocessor directives '#ifdef' and '#ifndef' test whether it is
    given, and its name is replaced by its value everywhere.

    Parameters
    ----------
    vertex_shader : str
        The file of the vertex shader.
    fragment_shader : str
        The file of the fragment shader.
    geometry_shader : str, optional
        The file of the geometry shader. Defaults to none.
    defines : dict of str to int, optional
        The macros of the shaders and their values. Defaults to none.
    """

    def __init__(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.geometry_shader = geometry_shader
        self.defines = {} if defines is None else defines
        self._program_id = None
        self._vao_id = None
        # The location of each uniform of 'UNIFORMS_NAME', -1 for one the program lacks or leaves unused. The compiled
        # passes set the uniforms by location, since a lookup by name on every draw costs more than setting the value.
        self.uniforms_loc = None

    def _add_to_context(self):
        assert self._program_id is None
        shader_ids = []

        # Load vert shader
        shader_ids.append(shaders.compileShader(self._load(self.vertex_shader), GL_VERTEX_SHADER))
        # Load frag shader
        shader_ids.append(shaders.compileShader(self._load(self.fragment_shader), GL_FRAGMENT_SHADER))
        # Load geometry shader
        if self.geometry_shader is not None:
            shader_ids.append(shaders.compileShader(self._load(self.geometry_shader), GL_GEOMETRY_SHADER))

        # PyOpenGL checks the program against the bound vertex array, which a core profile requires to exist
        if self._vao_id is None:
            self._vao_id = glGenVertexArrays(1)
        glBindVertexArray(self._vao_id)

        self._program_id = shaders.compileProgram(*shader_ids, validate=False)
        glBindVertexArray(0)
        self.uniforms_loc = np.array(
            [glGetUniformLocation(self._program_id, name) for name in UNIFORMS_NAME], dtype=np.int32
        )

    def _in_context(self):
        return self._program_id is not None

    def _remove_from_context(self):
        if self._program_id is not None:
            try:
                glDeleteProgram(self._program_id)
                glDeleteVertexArrays(1, [self._vao_id])
            except OpenGL.error.Error:
                pass
            self._program_id = None
            self._vao_id = None
            self.uniforms_loc = None

    def _load(self, shader_filename):
        with open(shader_filename) as f:
            text = f.read()
        # The directives testing a macro are resolved here, which leaves the macros free of any '#define'
        text = re.sub(
            r"#(ifdef|ifndef)\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*$",
            lambda match: "#if 1" if (match.group(2) in self.defines) == (match.group(1) == "ifdef") else "#if 0",
            text,
            flags=re.MULTILINE,
        )
        for define, value in self.defines.items():
            text = text.replace(define, str(value))
        return text

    def _bind(self):
        """Bind this shader program to the current OpenGL context."""
        assert self._program_id is not None
        glUseProgram(self._program_id)

    def _unbind(self):
        """Unbind this shader program from the current OpenGL context."""
        glUseProgram(0)

    def delete(self):
        """Delete this shader program from the current OpenGL context."""
        self._remove_from_context()

    def set_uniform(self, name, value):
        """Set a uniform of the program, which the caller binds.

        Parameters
        ----------
        name : str
            The name of the uniform. A uniform that the program lacks or leaves unused is ignored.
        value : int or (4,) or (4, 4) float
            The value of the uniform, an integer such as the texture unit of a sampler, a vector or a matrix.
        """
        loc = glGetUniformLocation(self._program_id, name)
        if isinstance(value, int):
            glUniform1i(loc, value)
            return
        value = np.ascontiguousarray(value, dtype=np.float32)
        if value.ndim == 1:
            glUniform4fv(loc, 1, value)
        else:
            glUniformMatrix4fv(loc, 1, GL_TRUE, value)
