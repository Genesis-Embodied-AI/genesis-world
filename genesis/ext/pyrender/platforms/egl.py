import ctypes
import functools
import os

import OpenGL.platform

import genesis as gs

from .base import Platform


EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233

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
from OpenGL import EGL as egl


def _get_egl_func(func_name, res_type, *arg_types):
    address = egl.eglGetProcAddress(func_name)
    if address is None:
        return None

    proto = ctypes.CFUNCTYPE(res_type)
    proto.argtypes = arg_types
    func = proto(address)
    return func


def _get_egl_struct(struct_name):
    from OpenGL._opaque import opaque_pointer_cls

    return opaque_pointer_cls(struct_name)


# These are not defined in PyOpenGL by default.
_EGLDeviceEXT = _get_egl_struct("EGLDeviceEXT")
_eglGetPlatformDisplayEXT = _get_egl_func("eglGetPlatformDisplayEXT", egl.EGLDisplay)
_eglQueryDevicesEXT = _get_egl_func("eglQueryDevicesEXT", egl.EGLBoolean)
_eglQueryDeviceStringEXT = _get_egl_func("eglQueryDeviceStringEXT", ctypes.c_char_p)


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

    @property
    def name(self):
        if self._display is None:
            return "default"

        name = _eglQueryDeviceStringEXT(self._display, EGL_DRM_DEVICE_FILE_EXT)
        if name is None:
            return None

        return name.decode("ascii")

    def __repr__(self):
        return "<EGLDevice(name={})>".format(self.name)


class EGLPlatform(Platform):
    """Renders using EGL."""

    def __init__(self, device_id: int | None = None):
        if _eglQueryDevicesEXT is None and device_id not in (0, None):
            raise RuntimeError("EGL platform plugin is not available. Enforcing specific EGL device not supported.")
        self._egl_device_id = device_id
        self._egl_device = None
        self._egl_display = None
        self._egl_context = None

    def init_context(self):
        _ensure_egl_loaded()
        from OpenGL.EGL import (
            EGL_SURFACE_TYPE,
            EGL_NO_SURFACE,
            EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE,
            EGL_RED_SIZE,
            EGL_GREEN_SIZE,
            EGL_DEPTH_SIZE,
            EGL_COLOR_BUFFER_TYPE,
            EGL_RGB_BUFFER,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_CONFORMANT,
            EGL_NONE,
            EGL_NO_CONTEXT,
            EGL_OPENGL_API,
            EGL_CONTEXT_MAJOR_VERSION,
            EGL_CONTEXT_MINOR_VERSION,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            eglInitialize,
            eglChooseConfig,
            eglBindAPI,
            eglCreateContext,
            eglMakeCurrent,
            EGLConfig,
            EGLError,
        )
        from OpenGL import arrays

        config_attributes = arrays.GLintArray.asArray(
            [
                EGL_SURFACE_TYPE,
                EGL_PBUFFER_BIT,
                EGL_BLUE_SIZE,
                8,
                EGL_RED_SIZE,
                8,
                EGL_GREEN_SIZE,
                8,
                EGL_DEPTH_SIZE,
                24,
                EGL_COLOR_BUFFER_TYPE,
                EGL_RGB_BUFFER,
                EGL_RENDERABLE_TYPE,
                EGL_OPENGL_BIT,
                EGL_CONFORMANT,
                EGL_OPENGL_BIT,
                EGL_NONE,
            ]
        )
        context_attributes = arrays.GLintArray.asArray(
            [
                EGL_CONTEXT_MAJOR_VERSION,
                4,
                EGL_CONTEXT_MINOR_VERSION,
                1,
                EGL_CONTEXT_OPENGL_PROFILE_MASK,
                EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE,
            ]
        )
        major, minor = ctypes.c_long(), ctypes.c_long()
        num_configs = ctypes.c_long()
        configs = (EGLConfig * 1)()

        # Get the list of devices to try on
        if _eglQueryDevicesEXT is None:
            all_devices = (EGLDevice(None),)
        else:
            all_devices = query_devices()
        if _eglQueryDevicesEXT is None or self._egl_device_id is None:
            devices = all_devices
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

            try:
                # Initialize EGL
                assert eglInitialize(egl_display, major, minor)

                # Configure EGL
                assert eglChooseConfig(egl_display, config_attributes, configs, 1, num_configs)

                # Bind EGL to the OpenGL API
                assert eglBindAPI(EGL_OPENGL_API)

                # Create an EGL context
                egl_context = eglCreateContext(egl_display, configs[0], EGL_NO_CONTEXT, context_attributes)

                # Select EGL context
                assert eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, egl_context)

                break
            except (AssertionError, EGLError) as e:
                error = e
        else:
            if self._egl_device_id is None:
                err_msg = "No EGL context could be initialized."
            else:
                err_msg = f"No EGL context could not be initialized for EGL_DEVICE_ID='{self._egl_device_id or i}'."
            raise EGLError(err_msg) from error

        # Backup the device and display that will be used
        self._egl_device = device
        self._egl_display = egl_display
        self._egl_context = egl_context

        # Make it current
        self.make_current()

    def make_current(self):
        from OpenGL.EGL import eglMakeCurrent, EGL_NO_SURFACE

        assert eglMakeCurrent(self._egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, self._egl_context)

    def make_uncurrent(self):
        """Make the OpenGL context uncurrent."""
        from OpenGL.EGL import eglMakeCurrent, EGL_NO_SURFACE, EGL_NO_CONTEXT

        if self._egl_display is not None:
            eglMakeCurrent(self._egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)

    def save_current_context(self):
        # Capture the current context as a restore callable, returning None when nothing is current. EGL's
        # 'make_uncurrent' releases the context, so our own context is never current here - only an external context
        # (e.g. another renderer mid-render) or none. Use the module-level 'egl' alias rather than a local import so
        # the returned callable stays valid during teardown, including at interpreter shutdown. Genesis EGL contexts
        # are surfaceless, so only the display and context are needed to restore (matching 'make_current').
        egl_display = egl.eglGetCurrentDisplay()
        egl_context = egl.eglGetCurrentContext()
        if not egl_context:
            return None
        return functools.partial(egl.eglMakeCurrent, egl_display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl_context)

    def delete_context(self):
        from OpenGL.EGL import eglDestroyContext, eglGetCurrentContext

        if self._egl_display is not None:
            if self._egl_context is not None:
                # The EGL current context is per-thread and shared across renderers. Only release it if it is
                # ours; otherwise another renderer (possibly mid-render) is current and tearing down this context
                # must not strand it with no current context. eglGetCurrentContext returns a fresh pointer object
                # each call that never compares equal to the stored handle by identity, so compare the underlying
                # addresses instead of the pointer objects.
                current_addr = ctypes.cast(eglGetCurrentContext(), ctypes.c_void_p).value
                own_addr = ctypes.cast(self._egl_context, ctypes.c_void_p).value
                if current_addr == own_addr:
                    self.make_uncurrent()
                eglDestroyContext(self._egl_display, self._egl_context)
                self._egl_context = None
            # NOTE: intentionally not calling eglTerminate here. The NVIDIA EGL
            # driver does not support terminate/re-initialize cycles on the same
            # device (returns EGL_BAD_ACCESS), which breaks multi-scene workflows.
            self._egl_display = None


__all__ = ["EGLPlatform"]
