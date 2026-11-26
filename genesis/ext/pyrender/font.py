"""Fonts of the rasterizer, which draws text from textures of glyphs."""

import os

import numpy as np

import freetype
import OpenGL
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_DEPTH_TEST,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FILL,
    GL_FLOAT,
    GL_FRONT_AND_BACK,
    GL_LINEAR,
    GL_NEAREST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT,
    ctypes,
    glActiveTexture,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDisable,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glPixelStorei,
    glPolygonMode,
    glVertexAttribPointer,
)

from .constants import FLOAT_SZ, TexChannels, TextAlign
from .texture import Texture
from .sampler import Sampler


class FontCache:
    """The fonts loaded in each OpenGL context, at each size."""

    def __init__(self):
        self._font_cache = {}

    def get_font(self, font_name, font_pt):
        """Return a font, loaded in the current OpenGL context at a size.

        Parameters
        ----------
        font_name : str
            The name of the font, a file of the 'fonts' folder without its extension.
        font_pt : int
            The height of the font, in pixels.
        """
        key = (OpenGL.contextdata.getContext(), font_name, font_pt)
        if key not in self._font_cache:
            font_file = os.path.join(os.path.dirname(__file__), "fonts", f"{font_name}.ttf")
            self._font_cache[key] = Font(font_file, font_pt)
        return self._font_cache[key]

    def clear(self):
        """Free the fonts of every context."""
        for key in self._font_cache:
            try:
                self._font_cache[key].delete()
            except OpenGL.error.GLError:
                pass
        self._font_cache.clear()


class Font:
    """A font at one size, its glyphs packed into one texture.

    Parameters
    ----------
    font_file : str
        The TrueType file of the font.
    font_pt : int
        The height of the font, in pixels.
    """

    def __init__(self, font_file, font_pt):
        self._face = freetype.Face(font_file)
        self._face.set_pixel_sizes(0, font_pt)
        # The glyphs of the 128 ASCII characters are packed into one texture, so that a string is drawn in a single
        # call whatever its length. Each glyph takes one cell of a grid of 16 columns, which keeps the texture small
        # enough for OpenGL at any font size. A glyph is framed by a copy of its own border, so that linear filtering
        # past its edge reads the edge, as a texture of its own clamped to its edge would.
        bitmaps = []
        self._glyph_size = np.zeros((128, 2), dtype=np.int32)
        self._glyph_bearing = np.zeros((128, 2), dtype=np.int32)
        self._glyph_advance = np.zeros(128, dtype=np.int32)
        for i in range(0, 128):
            face = self._face
            if "PYTEST_VERSION" in os.environ:
                # Bit-exact, ugly but deterministic
                face.load_char(
                    chr(i),
                    freetype.FT_LOAD_RENDER
                    | freetype.FT_LOAD_MONOCHROME
                    | freetype.FT_LOAD_NO_HINTING
                    | freetype.FT_LOAD_NO_AUTOHINT,
                )
                # FreeType mono bitmaps are 1 bit per pixel, packed
                bitmap = face.glyph.bitmap
                buf = np.asarray(bitmap.buffer, dtype=np.uint8)
                bits = np.unpackbits(buf).reshape(bitmap.rows, bitmap.pitch * 8)
                src = bits[:, : bitmap.width].astype(np.float32)
            else:
                # Normal, pretty rendering
                face.load_char(chr(i))
                bitmap = face.glyph.bitmap
                src = np.asarray(bitmap.buffer, dtype=np.float32) / 255.0
                src = src.reshape((bitmap.rows, bitmap.width))
            bitmaps.append(src)
            self._glyph_size[i] = (bitmap.width, bitmap.rows)
            self._glyph_bearing[i] = (face.glyph.bitmap_left, face.glyph.bitmap_top)
            self._glyph_advance[i] = face.glyph.advance.x >> 6
        n_columns = 16
        cell_height = max(bitmap.shape[0] for bitmap in bitmaps) + 2
        cell_width = max(bitmap.shape[1] for bitmap in bitmaps) + 2
        atlas_height = (len(bitmaps) // n_columns) * cell_height
        atlas_width = n_columns * cell_width
        atlas = np.zeros((atlas_height, atlas_width), dtype=np.float32)
        # The texture coordinates of each glyph in the atlas: left, bottom, right and top. The texture flips the atlas
        # upright (see 'Texture'), so that its row r lies at v = 1 - r / atlas_height.
        self._glyph_uv = np.zeros((128, 4), dtype=np.float32)
        for i, bitmap in enumerate(bitmaps):
            rows, width = bitmap.shape
            top = (i // n_columns) * cell_height + 1
            left = (i % n_columns) * cell_width + 1
            if bitmap.size:
                atlas[top - 1 : top + rows + 1, left - 1 : left + width + 1] = np.pad(bitmap, 1, mode="edge")
            bottom_v = 1.0 - (top + rows) / atlas_height
            self._glyph_uv[i] = (left / atlas_width, bottom_v, (left + width) / atlas_width, 1.0 - top / atlas_height)
        filter_mode = GL_NEAREST if "PYTEST_VERSION" in os.environ else GL_LINEAR
        sampler = Sampler(
            mag_filter=filter_mode, min_filter=filter_mode, wrap_s=GL_CLAMP_TO_EDGE, wrap_t=GL_CLAMP_TO_EDGE
        )
        self._atlas = Texture(sampler=sampler, source=atlas, source_channels=TexChannels.R)

        self._vbo = None
        self._vao = None

    def _add_to_context(self):
        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)
        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ * 6 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * FLOAT_SZ, ctypes.c_void_p(0))
        glBindVertexArray(0)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        if not self._atlas._in_context():
            self._atlas._add_to_context()

    def _remove_from_context(self):
        self._atlas.delete()
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = None
            self._vbo = None

    def _in_context(self):
        return self._vao is not None

    def _bind(self):
        glBindVertexArray(self._vao)

    def _unbind(self):
        glBindVertexArray(0)

    def delete(self):
        self._unbind()
        self._remove_from_context()

    def render_string(self, text, x, y, align):
        """Draw a line of text with the text program, which the caller binds with its uniforms.

        Parameters
        ----------
        text : str
            The line of text, in ASCII.
        x : int
            The horizontal position of the text, in pixels from the left.
        y : int
            The vertical position of the text, in pixels from the bottom.
        align : TextAlign
            The corner of the text that the position gives.
        """
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self._bind()

        codes = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        if not len(codes):
            self._unbind()
            return
        size = self._glyph_size[codes]
        bearing = self._glyph_bearing[codes]
        advance = self._glyph_advance[codes]

        # Determine width and height of text relative to x, y
        width = advance.sum()
        height = max(bearing[:, 1].max(), 0.0)

        # The text starts at the corner it is aligned on
        if align == TextAlign.BOTTOM_RIGHT:
            x -= width
        else:
            y -= height

        # Two triangles per glyph, each vertex holding its position and its texture coordinates in the atlas
        xpos = x + np.concatenate(((0.0,), np.cumsum(advance[:-1]))) + bearing[:, 0]
        ypos = y - (size[:, 1] - bearing[:, 1])
        u_left, v_bottom, u_right, v_top = self._glyph_uv[codes].T
        left, right, bottom, top = xpos, xpos + size[:, 0], ypos, ypos + size[:, 1]
        vertices = np.stack(
            (
                np.stack((left, bottom, u_left, v_bottom), axis=-1),
                np.stack((right, bottom, u_right, v_bottom), axis=-1),
                np.stack((right, top, u_right, v_top), axis=-1),
                np.stack((right, top, u_right, v_top), axis=-1),
                np.stack((left, top, u_left, v_top), axis=-1),
                np.stack((left, bottom, u_left, v_bottom), axis=-1),
            ),
            axis=1,
        ).astype(np.float32)

        self._atlas._bind()
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, 6 * len(codes))

        self._unbind()
        self._atlas._unbind()
