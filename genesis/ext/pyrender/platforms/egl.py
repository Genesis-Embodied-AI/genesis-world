import ctypes
import functools
import os

import OpenGL.platform

import genesis as gs

from .base import Platform


EGL_PLATFORM_DEVICE_EXT = 0x313F

_EGL_LOAD_ERROR = "EGL is required to create an offscreen OpenGL context, and it is unavailable on this system."


def _ensure_egl_loaded():
    plugin = OpenGL.platform.PlatformPlugin.by_name("egl")
    if plugin is None:
        raise RuntimeError("EGL platform plugin is not available.")

    plugin_class = plugin.load()
    plugin.loaded = True
    # create instance of this platform implementation
    plugin = plugin_class()

    # Probing the library here names EGL as the missing piece. A failed dlopen appears either as an ImportError or
    # as a None library, the shape `install` below tolerates, and either one otherwise reaches the user from deep
    # inside PyOpenGL, where the message mentions neither EGL nor Genesis.
    try:
        egl_library = plugin.EGL
    except ImportError as e:
        gs.raise_exception_from(_EGL_LOAD_ERROR, e)
    if egl_library is None:
        gs.raise_exception(_EGL_LOAD_ERROR)

    plugin.install(vars(OpenGL.platform))


_ensure_egl_loaded()
from OpenGL import EGL as egl, arrays
from OpenGL._opaque import opaque_pointer_cls


def _get_egl_func(func_name, res_type, *arg_types):
    address = egl.eglGetProcAddress(func_name)
    if address is None:
        return None

    proto = ctypes.CFUNCTYPE(res_type)
    proto.argtypes = arg_types
    func = proto(address)
    return func


# The device extensions of EGL, which PyOpenGL leaves out
_EGLDeviceEXT = opaque_pointer_cls("EGLDeviceEXT")
_eglGetPlatformDisplayEXT = _get_egl_func("eglGetPlatformDisplayEXT", egl.EGLDisplay)
_eglQueryDevicesEXT = _get_egl_func("eglQueryDevicesEXT", egl.EGLBoolean)


def query_devices():
    if _eglQueryDevicesEXT is None:
        raise RuntimeError("EGL query extension is not loaded or is not supported.")

    num_devices = egl.EGLint()
    success = _eglQueryDevicesEXT(0, None, ctypes.pointer(num_devices))
    if not success or num_devices.value < 1:
        return []

    devices = (_EGLDeviceEXT * num_devices.value)()  # array of size num_devices
    success = _eglQueryDevicesEXT(num_devices.value, devices, ctypes.pointer(num_devices))
    if not success or num_devices.value < 1:
        return []

    return [EGLDevice(devices[i]) for i in range(num_devices.value)]


def get_default_device():
    # Fall back to not using query extension.
    if _eglQueryDevicesEXT is None:
        return EGLDevice(None)

    return query_devices()[0]


def get_device_by_index(device_id):
    if _eglQueryDevicesEXT is None and device_id == 0:
        return get_default_device()

    devices = query_devices()
    if device_id >= len(devices):
        raise ValueError("Invalid device ID ({}) with {} devices".format(device_id, len(devices)))
    return devices[device_id]


class EGLDevice:
    def __init__(self, display=None):
        self._display = display

    def get_display(self):
        if self._display is None:
            return egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)

        return _eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, self._display, None)


class EGLPlatform(Platform):
    """Renders using EGL."""

    def __init__(self, device_id: int | None = None):
        if _eglQueryDevicesEXT is None and device_id not in (0, None):
            raise RuntimeError("EGL platform plugin is not available. Enforcing specific EGL device not supported.")
        self._egl_device_id = device_id
        self._egl_display = None
        self._egl_context = None

    def init_context(self):
        config_attributes = arrays.GLintArray.asArray(
            [
                egl.EGL_SURFACE_TYPE,
                egl.EGL_PBUFFER_BIT,
                egl.EGL_BLUE_SIZE,
                8,
                egl.EGL_RED_SIZE,
                8,
                egl.EGL_GREEN_SIZE,
                8,
                egl.EGL_DEPTH_SIZE,
                24,
                egl.EGL_COLOR_BUFFER_TYPE,
                egl.EGL_RGB_BUFFER,
                egl.EGL_RENDERABLE_TYPE,
                egl.EGL_OPENGL_BIT,
                egl.EGL_CONFORMANT,
                egl.EGL_OPENGL_BIT,
                egl.EGL_NONE,
            ]
        )
        context_attributes = arrays.GLintArray.asArray(
            [
                egl.EGL_CONTEXT_MAJOR_VERSION,
                4,
                egl.EGL_CONTEXT_MINOR_VERSION,
                1,
                egl.EGL_CONTEXT_OPENGL_PROFILE_MASK,
                egl.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                egl.EGL_NONE,
            ]
        )
        major, minor = ctypes.c_long(), ctypes.c_long()
        num_configs = ctypes.c_long()
        configs = (egl.EGLConfig * 1)()

        # The devices to try, the default display alone without the device extensions
        if _eglQueryDevicesEXT is None:
            devices = (EGLDevice(None),)
        elif self._egl_device_id is None:
            devices = query_devices()
        else:
            devices = (get_device_by_index(self._egl_device_id),)

        # Get the first EGL device that is working
        error = None
        for i, device in enumerate(devices):
            gs.logger.debug(f"Trying to create EGL Context for EGL_DEVICE_ID='{self._egl_device_id or i}'...")

            # Cache DISPLAY if necessary and get an off-screen EGL display
            orig_dpy = None
            if "DISPLAY" in os.environ:
                orig_dpy = os.environ["DISPLAY"]
                del os.environ["DISPLAY"]

            egl_display = device.get_display()
            if orig_dpy is not None:
                os.environ["DISPLAY"] = orig_dpy

            # Initialize EGL, pick a configuration, bind the OpenGL API, then create a context and select it
            try:
                if not egl.eglInitialize(egl_display, major, minor):
                    raise egl.EGLError("Unable to initialize the EGL display.")
                if not egl.eglChooseConfig(egl_display, config_attributes, configs, 1, num_configs):
                    raise egl.EGLError("No EGL configuration matches the attributes requested.")
                if not egl.eglBindAPI(egl.EGL_OPENGL_API):
                    raise egl.EGLError("Unable to bind EGL to the OpenGL API.")
                egl_context = egl.eglCreateContext(egl_display, configs[0], egl.EGL_NO_CONTEXT, context_attributes)
                if not egl.eglMakeCurrent(egl_display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl_context):
                    raise egl.EGLError("Unable to make the EGL context current.")
                break
            except egl.EGLError as e:
                error = e
        else:
            if self._egl_device_id is None:
                err_msg = "No EGL context could be initialized."
            else:
                err_msg = f"No EGL context could not be initialized for EGL_DEVICE_ID='{self._egl_device_id or i}'."
            raise egl.EGLError(err_msg) from error

        self._egl_display = egl_display
        self._egl_context = egl_context

        # Make it current
        self.make_current()

    def make_current(self):
        if not egl.eglMakeCurrent(self._egl_display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, self._egl_context):
            gs.raise_exception("Unable to make the EGL context current.")

    def make_uncurrent(self):
        """Make the OpenGL context uncurrent."""
        if self._egl_display is not None:
            egl.eglMakeCurrent(self._egl_display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT)

    def save_current_context(self):
        # 'make_uncurrent' releases the context of this platform, so the context captured is that of another renderer,
        # if any. The module-level 'egl' alias keeps the returned callable valid during teardown, interpreter shutdown
        # included. The contexts of Genesis are surfaceless, so the display and the context restore one (see
        # 'make_current').
        egl_display = egl.eglGetCurrentDisplay()
        egl_context = egl.eglGetCurrentContext()
        if not egl_context:
            return None
        return functools.partial(egl.eglMakeCurrent, egl_display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl_context)

    def delete_context(self):
        if self._egl_display is not None:
            if self._egl_context is not None:
                # The EGL current context is per-thread and shared across renderers. Only release it if it is
                # ours; otherwise another renderer (possibly mid-render) is current and tearing down this context
                # must not strand it with no current context. egl.eglGetCurrentContext returns a fresh pointer object
                # each call that never compares equal to the stored handle by identity, so compare the underlying
                # addresses instead of the pointer objects.
                current_addr = ctypes.cast(egl.eglGetCurrentContext(), ctypes.c_void_p).value
                own_addr = ctypes.cast(self._egl_context, ctypes.c_void_p).value
                if current_addr == own_addr:
                    self.make_uncurrent()
                egl.eglDestroyContext(self._egl_display, self._egl_context)
                self._egl_context = None
            # The display stays initialized, since the NVIDIA driver fails to initialize a display terminated before
            # (EGL_BAD_ACCESS), which the next scene would do
            self._egl_display = None


__all__ = ["EGLPlatform"]
