"""Samplers of the textures of the rasterizer."""

from typing import NamedTuple

from OpenGL.GL import GL_REPEAT


class Sampler(NamedTuple):
    """The filtering and the wrapping that a texture is read with, as OpenGL enums."""

    mag_filter: int | None = None
    """The magnification filter, None to pick it from the content of the texture."""
    min_filter: int | None = None
    """The minification filter, None to pick it from the content of the texture."""
    wrap_s: int = GL_REPEAT
    """The wrapping along the first texture coordinate."""
    wrap_t: int = GL_REPEAT
    """The wrapping along the second texture coordinate."""
