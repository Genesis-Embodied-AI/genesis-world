"""OpenGL shader program wrapper."""

import numbers
import os
import re

import numpy as np

import OpenGL
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shader_utils

func = None


class ShaderProgramCache(object):
    """A cache for shader programs."""

    def __init__(self, shader_dir=None):
        self._program_cache = {}
        self.shader_dir = shader_dir
        if self.shader_dir is None:
            base_dir, _ = os.path.split(os.path.realpath(__file__))
            self.shader_dir = os.path.join(base_dir, "shaders")

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        """Get a program via a list of shader files to include in the program.

        Parameters
        ----------
        vertex_shader : str
            The vertex shader filename.
        fragment_shader : str
            The fragment shader filename.
        geometry_shader : str
            The geometry shader filename.
        defines : dict
            Defines and their values for the shader.

        Returns
        -------
        program : :class:`.ShaderProgram`
            The program.
        """
        shader_names = []
        if defines is None:
            defines = {}
        shader_filenames = [x for x in [vertex_shader, fragment_shader, geometry_shader] if x is not None]
        for fn in shader_filenames:
            if fn is None:
                continue
            _, name = os.path.split(fn)
            shader_names.append(name)
        cid = OpenGL.contextdata.getContext()
        key = tuple([cid] + sorted([(s, 1) for s in shader_names] + [(d, defines[d]) for d in defines]))

        if key not in self._program_cache:
            shader_filenames = [os.path.join(self.shader_dir, fn) for fn in shader_filenames]
            if len(shader_filenames) == 2:
                shader_filenames.append(None)
            vs, fs, gs = shader_filenames
            self._program_cache[key] = ShaderProgram(
                vertex_shader=vs, fragment_shader=fs, geometry_shader=gs, defines=defines
            )
        return self._program_cache[key]

    def clear(self):
        for key in self._program_cache:
            try:
                self._program_cache[key].delete()
            except OpenGL.error.GLError:
                pass
        self._program_cache.clear()


class ShaderProgram(object):
    """A thin wrapper about OpenGL shader programs that supports easy creation,
    binding, and uniform-setting.

    Parameters
    ----------
    vertex_shader : str
        The vertex shader filename.
    fragment_shader : str
        The fragment shader filename.
    geometry_shader : str
        The geometry shader filename.
    defines : dict
        Defines and their values for the shader.
    """

    def __init__(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.geometry_shader = geometry_shader

        self.defines = defines
        if self.defines is None:
            self.defines = {}

        self._program_id = None
        self._vao_id = None  # PYOPENGL BUG

        # DEBUG
        # self._unif_map = {}

    def _add_to_context(self):
        if self._program_id is not None:
            raise ValueError("Shader program already in context")
        shader_ids = []

        # Load vert shader
        shader_ids.append(gl_shader_utils.compileShader(self._load(self.vertex_shader), GL_VERTEX_SHADER))
        # Load frag shader
        shader_ids.append(gl_shader_utils.compileShader(self._load(self.fragment_shader), GL_FRAGMENT_SHADER))
        # Load geometry shader
        if self.geometry_shader is not None:
            shader_ids.append(gl_shader_utils.compileShader(self._load(self.geometry_shader), GL_GEOMETRY_SHADER))

        # Bind empty VAO PYOPENGL BUG
        if self._vao_id is None:
            self._vao_id = glGenVertexArrays(1)
        glBindVertexArray(self._vao_id)

        # Compile program
        self._program_id = gl_shader_utils.compileProgram(*shader_ids, validate=False)

        # Unbind empty VAO PYOPENGL BUG
        glBindVertexArray(0)

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

    def _load(self, shader_filename):
        path, _ = os.path.split(shader_filename)

        with open(shader_filename) as f:
            text = f.read()

        def ifdef(matchobj):
            if matchobj.group(1) in self.defines:
                return "#if 1"
            else:
                return "#if 0"

        def ifndef(matchobj):
            if matchobj.group(1) in self.defines:
                return "#if 0"
            else:
                return "#if 1"

        ifdef_regex = re.compile("#ifdef\\s+([a-zA-Z_][a-zA-Z_0-9]*)\\s*$", re.MULTILINE)
        ifndef_regex = re.compile("#ifndef\\s+([a-zA-Z_][a-zA-Z_0-9]*)\\s*$", re.MULTILINE)
        text = re.sub(ifdef_regex, ifdef, text)
        text = re.sub(ifndef_regex, ifndef, text)

        for define in self.defines:
            value = str(self.defines[define])
            text = text.replace(define, value)

        return text

    def _bind(self):
        """Bind this shader program to the current OpenGL context."""
        if self._program_id is None:
            raise ValueError("Cannot bind program that is not in context")
        # glBindVertexArray(self._vao_id)
        glUseProgram(self._program_id)

    def _unbind(self):
        """Unbind this shader program from the current OpenGL context."""
        glUseProgram(0)

    def delete(self):
        """Delete this shader program from the current OpenGL context."""
        self._remove_from_context()

    def set_uniform(self, name, value, unsigned=False):
        """Set a uniform value in the current shader program.

        Parameters
        ----------
        name : str
            Name of the uniform to set.
        value : int, float, or ndarray
            Value to set the uniform to.
        unsigned : bool
            If True, ints will be treated as unsigned values.
        """
        # DEBUG
        # self._unif_map[name] = 1, (1,)
        loc = glGetUniformLocation(self._program_id, name)

        if loc == -1:
            raise ValueError("Invalid shader variable: {}".format(name))

        if isinstance(value, np.ndarray):
            # DEBUG
            # self._unif_map[name] = value.size, value.shape
            if value.ndim == 1:
                if np.issubdtype(value.dtype, np.unsignedinteger) or unsigned:
                    value = np.ascontiguousarray(value, np.uint32)
                elif np.issubdtype(value.dtype, np.integer):
                    value = np.ascontiguousarray(value, dtype=np.int32)
                else:
                    value = np.ascontiguousarray(value, dtype=np.float32)
                func = self._FUNC_MAP[(len(value), value.dtype.kind)]
                func(loc, 1, value)
            else:
                value = np.ascontiguousarray(value, dtype=np.float32)
                func = self._FUNC_MAP[tuple(value.shape[:2])]
                func(loc, 1, GL_TRUE, value)

        # Call correct uniform function
        elif isinstance(value, (numbers.Real, np.floating)):
            glUniform1f(loc, float(value))
        elif isinstance(value, (numbers.Integral, np.integer)):
            if unsigned:
                glUniform1ui(loc, int(value))
            else:
                glUniform1i(loc, int(value))
        else:
            raise ValueError("Invalid data type")

    _FUNC_MAP = {
        (1, "u"): glUniform1uiv,
        (2, "u"): glUniform2uiv,
        (3, "u"): glUniform3uiv,
        (4, "u"): glUniform4uiv,
        (1, "i"): glUniform1iv,
        (2, "i"): glUniform2iv,
        (3, "i"): glUniform3iv,
        (4, "i"): glUniform4iv,
        (1, "f"): glUniform1fv,
        (2, "f"): glUniform2fv,
        (3, "f"): glUniform3fv,
        (4, "f"): glUniform4fv,
        (2, 2): glUniformMatrix2fv,
        (2, 3): glUniformMatrix2x3fv,
        (2, 4): glUniformMatrix2x4fv,
        (3, 2): glUniformMatrix3x2fv,
        (3, 3): glUniformMatrix4fv,
        (3, 4): glUniformMatrix3x4fv,
        (4, 2): glUniformMatrix4x2fv,
        (4, 3): glUniformMatrix4x3fv,
        (4, 4): glUniformMatrix4fv,
    }
