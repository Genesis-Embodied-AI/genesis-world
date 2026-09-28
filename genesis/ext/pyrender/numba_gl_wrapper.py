import ctypes

import numpy as np
from numba import *
from numba import types
from numba.extending import (
    models,
    register_model,
    make_attribute_wrapper,
    typeof_impl,
    as_numba_type,
    unbox,
    NativeValue,
)
from numba.core import cgutils

import OpenGL.GL as GL
from OpenGL._bytes import as_8_bit
from OpenGL.GL import (
    GLboolean,
    GLenum,
    GLfloat,
    GLint,
    GLintptr,
    GLsizei,
    GLsizeiptr,
    GLuint,
    GLvoid,
    GLvoidp,
)

import genesis as gs


class GLWrapper:
    def __init__(self):
        self.gl_funcs = {}
        self._wrapper_type = None
        self._wrapper_instance = None

    def load_func(self, func_name, *signature):
        try:
            dll = GL.platform.PLATFORM.GL
            func_ptr = GL.platform.ctypesloader.buildFunction(
                GL.platform.PLATFORM.functionTypeFor(dll)(*signature),
                func_name,
                dll,
            )
        except AttributeError:
            pointer = GL.platform.PLATFORM.getExtensionProcedure(as_8_bit(func_name))
            func_ptr = GL.platform.PLATFORM.functionTypeFor(dll)(*signature)(pointer)
        self.gl_funcs[func_name] = func_ptr

    @property
    def wrapper_type(self):
        if self._wrapper_type is None:
            self.build_wrapper()
        return self._wrapper_type

    @property
    def wrapper_instance(self):
        if self._wrapper_instance is None:
            self.build_wrapper()
        return self._wrapper_instance

    def build_wrapper(self):
        load_func = self.load_func
        for name, signature in (
            ("glGetUniformLocation", (GLint, GLuint, GLvoidp)),
            ("glUniformMatrix4fv", (GLvoid, GLint, GLsizei, GLboolean, GLvoidp)),
            ("glUniform1i", (GLvoid, GLint, GLint)),
            ("glUniform1f", (GLvoid, GLint, GLfloat)),
            ("glUniform2f", (GLvoid, GLint, GLfloat, GLfloat)),
            ("glUniform3fv", (GLvoid, GLint, GLsizei, GLvoidp)),
            ("glUniform4fv", (GLvoid, GLint, GLsizei, GLvoidp)),
            ("glBindVertexArray", (GLvoid, GLuint)),
            ("glActiveTexture", (GLvoid, GLenum)),
            ("glBindTexture", (GLvoid, GLenum, GLuint)),
            ("glEnable", (GLvoid, GLenum)),
            ("glDisable", (GLvoid, GLenum)),
            ("glBlendFunc", (GLvoid, GLenum, GLenum)),
            ("glPolygonMode", (GLvoid, GLenum, GLenum)),
            ("glCullFace", (GLvoid, GLenum)),
            ("glDrawElementsInstanced", (GLvoid, GLenum, GLsizei, GLenum, GLvoidp, GLsizei)),
            ("glDrawArraysInstanced", (GLvoid, GLenum, GLint, GLsizei, GLsizei)),
            ("glDrawElementsInstancedBaseInstance", (GLvoid, GLenum, GLsizei, GLenum, GLvoidp, GLsizei, GLuint)),
            ("glDrawArraysInstancedBaseInstance", (GLvoid, GLenum, GLint, GLsizei, GLsizei, GLuint)),
            ("glUseProgram", (GLvoid, GLuint)),
            ("glFlush", (GLvoid,)),
            ("glReadPixels", (GLvoid, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, GLvoidp)),
            ("glBindBuffer", (GLvoid, GLenum, GLuint)),
            ("glBufferData", (GLvoid, GLenum, GLsizeiptr, GLvoidp, GLenum)),
            ("glBufferSubData", (GLvoid, GLenum, GLintptr, GLsizeiptr, GLvoidp)),
            ("glVertexAttribPointer", (GLvoid, GLuint, GLint, GLenum, GLboolean, GLsizei, GLvoidp)),
        ):
            try:
                load_func(name, *signature)
            except AttributeError:
                # OpenGL function not available, probably because the installed version does not support it (too old).
                # Moving to the next one without raising an exception since it is not blocking at this point.
                gs.logger.debug(f"OpenGL function '{name}' not available on this machine.")

        funcs = self.gl_funcs
        func_types = {}
        for func_name in funcs:
            func_types[func_name] = typeof(funcs[func_name])

        class GLFunc:
            def __init__(self):
                for func_name in funcs:
                    setattr(self, func_name, funcs[func_name])
                # Unboxing this object into a kernel argument reads the function addresses from this array in one go.
                # Reading them from the ctypes attributes above costs the kernel call a Python attribute lookup and
                # a pointer extraction per function, which adds up to more than a small kernel does.
                self.addresses = np.fromiter(
                    (ctypes.cast(funcs[func_name], ctypes.c_void_p).value for func_name in funcs), np.uint64, len(funcs)
                )

        class GLFuncType(types.Type):
            def __init__(self):
                super().__init__("GLFunc")

            def __eq__(self, other):
                return hasattr(other, "name") and other.name == "GLFunc"

            def __hash__(self):
                return hash("GLFuncType")

        glfunc_type = GLFuncType()

        @typeof_impl.register(GLFunc)
        def typeof_index(val, c):
            return glfunc_type

        as_numba_type.register(GLFunc, glfunc_type)

        @register_model(GLFuncType)
        class GLFuncModel(models.StructModel):
            def __init__(self, dmm, fe_type):
                members = list(func_types.items())
                super().__init__(dmm, fe_type, members)

        addresses_type = types.Array(types.uint64, 1, "C")

        @unbox(GLFuncType)
        def unbox_glfunc(typ, obj, c):
            addresses_obj = c.pyapi.object_getattr_string(obj, "addresses")
            addresses = c.unbox(addresses_type, addresses_obj)
            c.pyapi.decref(addresses_obj)
            data = c.context.make_array(addresses_type)(c.context, c.builder, addresses.value).data

            gl_func = cgutils.create_struct_proxy(typ)(c.context, c.builder)
            for i, func_name in enumerate(funcs):
                address = c.builder.load(cgutils.gep(c.builder, data, i))
                func_ptr_type = c.context.get_value_type(func_types[func_name])
                setattr(gl_func, func_name, c.builder.inttoptr(address, func_ptr_type))

            return NativeValue(gl_func._getvalue(), is_error=addresses.is_error)

        for func_name in funcs:
            make_attribute_wrapper(GLFuncType, func_name, func_name)

        self._wrapper_type = glfunc_type
        self._wrapper_instance = GLFunc()
