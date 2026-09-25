import ctypes
import functools

import OpenGL.contextdata
import OpenGL.platform
import pyglet

import genesis as gs

from .base import Platform


# CGL pixel format attributes and values, transliterated from the OpenGL framework headers, as PyOpenGL ships no CGL
# bindings.
CGL_PFA_DEPTH_SIZE = 12
CGL_PFA_ACCELERATED = 73
CGL_PFA_OPENGL_PROFILE = 99
CGL_OPENGL_PROFILE_VERSION_3_2_CORE = 0x3200

# A private handle to the OpenGL framework, opened from the path PyOpenGL already resolved. The entry points below need
# explicit ctypes signatures, and annotating PyOpenGL's own handle in place would change process-wide behavior, since
# 'CGLGetCurrentContext' is what backs its context bookkeeping.
_framework = ctypes.CDLL(OpenGL.platform.PLATFORM.CGL._name)

_CGLChoosePixelFormat = _framework.CGLChoosePixelFormat
_CGLChoosePixelFormat.argtypes = (
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int),
)
_CGLChoosePixelFormat.restype = ctypes.c_int

_CGLDestroyPixelFormat = _framework.CGLDestroyPixelFormat
_CGLDestroyPixelFormat.argtypes = (ctypes.c_void_p,)
_CGLDestroyPixelFormat.restype = ctypes.c_int

_CGLCreateContext = _framework.CGLCreateContext
_CGLCreateContext.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
_CGLCreateContext.restype = ctypes.c_int

_CGLDestroyContext = _framework.CGLDestroyContext
_CGLDestroyContext.argtypes = (ctypes.c_void_p,)
_CGLDestroyContext.restype = ctypes.c_int

_CGLSetCurrentContext = _framework.CGLSetCurrentContext
_CGLSetCurrentContext.argtypes = (ctypes.c_void_p,)
_CGLSetCurrentContext.restype = ctypes.c_int

_CGLGetCurrentContext = _framework.CGLGetCurrentContext
_CGLGetCurrentContext.argtypes = ()
_CGLGetCurrentContext.restype = ctypes.c_void_p


class CGLPlatform(Platform):
    """Renders offscreen using CGL, the window-system-independent OpenGL interface of MacOS.

    The context requires neither a window nor an active display, rendering exclusively into framebuffer objects it
    allocates itself. This is what sets it apart from the pyglet platform, whose context comes attached to an invisible
    window and therefore to a screen: rendering here survives a locked screen, where every display is asleep, as well as
    a virtual machine exposing no screen at all.
    """

    def __init__(self):
        self._pixel_format = None
        self._context = None

    def init_context(self):
        # A core profile is mandatory: the compatibility profile caps the software renderer at OpenGL 2.1 / GLSL 1.20,
        # which cannot compile the shaders. Hardware acceleration in contrast is only a preference, so it is requested
        # first and dropped on retry, letting a machine with no usable GPU fall back to the software renderer. The
        # sample count is left out because rendering targets framebuffer objects that carry their own.
        attributes = [CGL_PFA_OPENGL_PROFILE, CGL_OPENGL_PROFILE_VERSION_3_2_CORE, CGL_PFA_DEPTH_SIZE, 24]
        pixel_format, num_pixel_formats = ctypes.c_void_p(), ctypes.c_int()
        for candidate in ([*attributes, CGL_PFA_ACCELERATED], attributes):
            attributes_array = (ctypes.c_int * (len(candidate) + 1))(*candidate, 0)
            error = _CGLChoosePixelFormat(attributes_array, ctypes.byref(pixel_format), ctypes.byref(num_pixel_formats))
            if not error and num_pixel_formats.value:
                break
        else:
            gs.raise_exception(
                "Failed to find an OpenGL 3.2+ core profile pixel format. No OpenGL renderer is available on this "
                "machine."
            )

        context = ctypes.c_void_p()
        error = _CGLCreateContext(pixel_format, None, ctypes.byref(context))
        if error:
            _CGLDestroyPixelFormat(pixel_format)
            gs.raise_exception(f"Failed to create an offscreen CGL context (error {error}).")

        self._pixel_format = pixel_format
        self._context = context

    def make_current(self):
        _CGLSetCurrentContext(self._context)
        # pyglet tracks process-wide which of its contexts is current and short-circuits 'Context.set_current' when it
        # matches, so taking the current context over behind its back would leave that belief stale and a later viewer
        # rebind would be skipped, sending its GL calls here. Clear it in lockstep with 'gl_info', exactly as pyglet's
        # own 'Context.set_current' and 'Context.destroy' do, so the next rebind actually happens.
        pyglet.gl.current_context = None
        pyglet.gl.gl_info.remove_active_context()

    def make_uncurrent(self):
        _CGLSetCurrentContext(None)

    def save_current_context(self):
        # 'make_uncurrent' clears the current context and every render path ends there, so this platform's own context
        # is never the one captured here, only an external one (e.g. another renderer mid-render) or none. Restoring a
        # context this platform is about to destroy would otherwise leave the caller with a dangling one.
        context = _CGLGetCurrentContext()
        if not context:
            return None
        return functools.partial(_CGLSetCurrentContext, ctypes.c_void_p(context))

    def delete_context(self):
        if self._context is not None:
            # The current context is per-thread and shared across renderers, so release it only when it is ours:
            # binding this one here would strand a renderer that is mid-render, and CGL destroys a context that is
            # not current all the same.
            if _CGLGetCurrentContext() == self._context.value:
                OpenGL.contextdata.cleanupContext(OpenGL.contextdata.getContext())
                _CGLSetCurrentContext(None)
            _CGLDestroyContext(self._context)
            self._context = None
        if self._pixel_format is not None:
            _CGLDestroyPixelFormat(self._pixel_format)
            self._pixel_format = None


__all__ = ["CGLPlatform"]
