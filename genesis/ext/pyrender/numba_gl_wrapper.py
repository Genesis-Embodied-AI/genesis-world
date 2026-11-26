import ctypes

import numpy as np

from numba import typeof, types
from numba.core import cgutils
from numba.extending import (
    NativeValue,
    as_numba_type,
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
    unbox,
)
import OpenGL.GL as GL
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
    """The OpenGL functions that the compiled passes call, as a numba type and an instance of it.

    A function missing from the OpenGL version of the machine is left out of both, the passes checking 'gl_funcs'
    before calling it.
    """

    def __init__(self):
        self.gl_funcs = {}
        for func_name, signature in (
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
            ("glBlendFuncSeparate", (GLvoid, GLenum, GLenum, GLenum, GLenum)),
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
            dll = GL.platform.PLATFORM.GL
            func_type = GL.platform.PLATFORM.functionTypeFor(dll)(*signature)
            try:
                try:
                    self.gl_funcs[func_name] = GL.platform.ctypesloader.buildFunction(func_type, func_name, dll)
                except AttributeError:
                    pointer = GL.platform.PLATFORM.getExtensionProcedure(func_name.encode())
                    self.gl_funcs[func_name] = func_type(pointer)
            except AttributeError:
                # An OpenGL version older than the one introducing the function, which the passes do without
                gs.logger.debug(f"OpenGL function '{func_name}' not available on this machine.")

        func_types = {func_name: typeof(func) for func_name, func in self.gl_funcs.items()}

        # Unboxing the instance into a kernel argument reads the function addresses from this array in one go. Reading
        # them from the ctypes functions costs the kernel call a Python lookup and a pointer extraction per function,
        # which adds up to more than a small kernel does.
        addresses = np.fromiter(
            (ctypes.cast(func, ctypes.c_void_p).value for func in self.gl_funcs.values()), np.uint64, len(self.gl_funcs)
        )

        class GLFunc:
            def __init__(self):
                self.addresses = addresses

        class GLFuncType(types.Type):
            def __init__(self):
                super().__init__("GLFunc")

            def __eq__(self, other):
                return isinstance(other, types.Type) and other.name == "GLFunc"

            def __hash__(self):
                return hash("GLFuncType")

        glfunc_type = GLFuncType()

        @typeof_impl.register(GLFunc)
        def typeof_glfunc(val, c):
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
            for i, func_name in enumerate(self.gl_funcs):
                address = c.builder.load(cgutils.gep(c.builder, data, i))
                func_ptr_type = c.context.get_value_type(func_types[func_name])
                setattr(gl_func, func_name, c.builder.inttoptr(address, func_ptr_type))

            return NativeValue(gl_func._getvalue(), is_error=addresses.is_error)

        for func_name in self.gl_funcs:
            make_attribute_wrapper(GLFuncType, func_name, func_name)

        self.wrapper_type = glfunc_type
        self.wrapper_instance = GLFunc()
