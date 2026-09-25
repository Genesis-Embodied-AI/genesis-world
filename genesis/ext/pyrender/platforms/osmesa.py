import ctypes
import functools

import OpenGL.platform
import OpenGL.platform.osmesa

import genesis as gs

from .base import Platform


# PyOpenGL commits to the OSMesa library at import time (see 'OSMesaPlatform'), so the bindings below only exist when
# it was selected upfront.
if not isinstance(OpenGL.platform.PLATFORM, OpenGL.platform.osmesa.OSMesaPlatform):
    gs.raise_exception("PYOPENGL_PLATFORM='osmesa' must be set before importing genesis.")
from OpenGL import GL as gl
from OpenGL import arrays
from OpenGL.osmesa import (
    OSMESA_CONTEXT_MAJOR_VERSION,
    OSMESA_CONTEXT_MINOR_VERSION,
    OSMESA_CORE_PROFILE,
    OSMESA_DEPTH_BITS,
    OSMESA_FORMAT,
    OSMESA_PROFILE,
    OSMESA_RGBA,
    OSMesaCreateContextAttribs,
    OSMesaDestroyContext,
    OSMesaGetColorBuffer,
    OSMesaGetCurrentContext,
    OSMesaMakeCurrent,
)


__all__ = ["OSMesaPlatform"]


class OSMesaPlatform(Platform):
    """Renders offscreen using OSMesa, the window-system-independent software rasterizer of Mesa.

    OSMesa is a self-contained copy of Mesa, and PyOpenGL binds every OpenGL function to the library of the platform
    selected when it is first imported. The platform is therefore committed for the whole process: PYOPENGL_PLATFORM
    must be set to 'osmesa' before importing genesis, and every OpenGL context of the process is then an OSMesa one.
    Genesis selects it by itself on a headless Linux machine lacking EGL (see rasterizer.py).

    Making the context current requires a color buffer serving as default framebuffer. A 1x1 buffer suffices, like the
    hidden 1x1 window of the pyglet platform, since rendering targets framebuffer objects (see Platform).
    """

    def __init__(self):
        self._context = None
        self._buffer = None

    def init_context(self):
        attrs = arrays.GLintArray.asArray(
            [
                OSMESA_FORMAT,
                OSMESA_RGBA,
                OSMESA_DEPTH_BITS,
                24,
                OSMESA_PROFILE,
                OSMESA_CORE_PROFILE,
                OSMESA_CONTEXT_MAJOR_VERSION,
                3,
                OSMESA_CONTEXT_MINOR_VERSION,
                3,
                0,
            ]
        )
        self._context = OSMesaCreateContextAttribs(attrs, None)
        self._buffer = arrays.GLubyteArray.zeros((1, 1, 4))

    def make_current(self):
        assert OSMesaMakeCurrent(self._context, self._buffer, gl.GL_UNSIGNED_BYTE, 1, 1)

    def make_uncurrent(self):
        OSMesaMakeCurrent(None, None, gl.GL_UNSIGNED_BYTE, 0, 0)

    def save_current_context(self):
        # 'make_uncurrent' releases the context and every render path ends there, so this platform's own context is
        # never the one captured here, only an external one (e.g. another renderer mid-render) or none. A context is
        # made current together with its color buffer, which is queried back from the context to restore both.
        context = OSMesaGetCurrentContext()
        if not context:
            return None
        width, height, _, buffer = OSMesaGetColorBuffer(context)
        return functools.partial(OSMesaMakeCurrent, context, buffer, gl.GL_UNSIGNED_BYTE, width, height)

    def delete_context(self):
        if self._context is not None:
            # The current context is process-global and shared across renderers, so release it only when it is ours,
            # otherwise a renderer that is mid-render would be stranded. Handles are compared by address since PyOpenGL
            # returns a fresh pointer object on every query.
            current_addr = ctypes.cast(OSMesaGetCurrentContext(), ctypes.c_void_p).value
            own_addr = ctypes.cast(self._context, ctypes.c_void_p).value
            if current_addr == own_addr:
                self.make_uncurrent()
            OSMesaDestroyContext(self._context)
            self._context = None
            self._buffer = None
