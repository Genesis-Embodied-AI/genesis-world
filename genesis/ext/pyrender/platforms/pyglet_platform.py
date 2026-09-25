from ..constants import TARGET_OPEN_GL_MAJOR, TARGET_OPEN_GL_MINOR, MIN_OPEN_GL_MAJOR, MIN_OPEN_GL_MINOR
from .base import Platform

import OpenGL
import pyglet


__all__ = ["PygletPlatform"]


class PygletPlatform(Platform):
    """Renders on-screen using a 1x1 hidden Pyglet window for getting
    an OpenGL context.
    """

    def __init__(self):
        self._window = None

    def init_context(self):
        pyglet.options["shadow_window"] = False

        try:
            pyglet.lib.x11.xlib.XInitThreads()
        except Exception:
            pass

        self._window = None
        confs = [
            pyglet.gl.Config(
                sample_buffers=1,
                samples=4,
                depth_size=24,
                double_buffer=True,
                major_version=TARGET_OPEN_GL_MAJOR,
                minor_version=TARGET_OPEN_GL_MINOR,
            ),
            pyglet.gl.Config(
                depth_size=24,
                double_buffer=True,
                major_version=TARGET_OPEN_GL_MAJOR,
                minor_version=TARGET_OPEN_GL_MINOR,
            ),
            pyglet.gl.Config(
                sample_buffers=1,
                samples=4,
                depth_size=24,
                double_buffer=True,
                major_version=MIN_OPEN_GL_MAJOR,
                minor_version=MIN_OPEN_GL_MINOR,
            ),
            pyglet.gl.Config(
                depth_size=24, double_buffer=True, major_version=MIN_OPEN_GL_MAJOR, minor_version=MIN_OPEN_GL_MINOR
            ),
        ]
        while confs:
            conf = confs.pop(0)
            try:
                self._window = pyglet.window.Window(config=conf, visible=False, resizable=False, width=1, height=1)
                break
            except (pyglet.window.NoSuchConfigException, pyglet.gl.ContextException) as e:
                if not confs:
                    raise ValueError(
                        "Failed to initialize Pyglet window with an OpenGL >= 3+ "
                        "context. If you're logged in via SSH, ensure that you're "
                        "running your script with vglrun (i.e. VirtualGL)."
                    ) from e

    def make_current(self):
        if self._window:
            self._window.switch_to()

    def make_uncurrent(self):
        try:
            pyglet.gl.xlib.glx.glXMakeContextCurrent(self._window.context.x_display, 0, 0, None)
        except Exception:
            pass
        # The glx call above is a no-op off X11, so explicitly clear pyglet's current-context bookkeeping too. This
        # keeps it accurate on every platform (e.g. Cocoa), so 'save_current_context' never mistakes this renderer's
        # own context for an external one to restore, and the next 'make_current' rebinds rather than short-circuiting.
        # Clear gl_info in lockstep, exactly as pyglet's own Context.set_current and Context.destroy do. Otherwise
        # gl_info.have_context() stays True while no context is current, and pyglet's Windows pixel-format matching
        # then takes the WGL ARB path, whose extension functions resolve only with a context current and otherwise
        # raise "wglChoosePixelFormatARB before GL context created".
        pyglet.gl.current_context = None
        pyglet.gl.gl_info.remove_active_context()

    def save_current_context(self):
        # 'set_current' is a bound method of the current Context, i.e. a self-contained zero-argument restore callable.
        context = pyglet.gl.current_context
        return context.set_current if context is not None else None

    def delete_context(self):
        if self._window is not None:
            self.make_current()
            cid = OpenGL.contextdata.getContext()
            try:
                self._window.context.destroy()
                self._window.close()
            except Exception:
                pass
            self._window = None
            OpenGL.contextdata.cleanupContext(cid)
            del cid
