import abc
from abc import ABCMeta


class Platform(metaclass=ABCMeta):
    """Base class for all OpenGL platforms.

    Rendering targets framebuffer objects allocated by the renderer itself, so a platform only has to provide a current
    OpenGL context, whatever the size of its default framebuffer.
    """

    @abc.abstractmethod
    def init_context(self):
        """Create an OpenGL context."""
        pass

    @abc.abstractmethod
    def make_current(self):
        """Make the OpenGL context current."""
        pass

    @abc.abstractmethod
    def make_uncurrent(self):
        """Make the OpenGL context uncurrent."""
        pass

    def save_current_context(self):
        """Capture the GL context currently current, returning a zero-argument callable that restores it, or None.

        Returns None when nothing is current. The current context is process- or thread-global and shared across all
        renderers, so tearing one down - which makes its own context current - must restore whatever was current
        before, otherwise a renderer that was current (e.g. one mid-render) is left stranded. The callable is
        self-contained, so it can be invoked after this renderer's own context has been destroyed. The base platform
        has no global current context to capture.
        """
        return None

    @abc.abstractmethod
    def delete_context(self):
        """Delete the OpenGL context."""
        pass

    def __del__(self):
        try:
            self.delete_context()
        except Exception:
            pass
