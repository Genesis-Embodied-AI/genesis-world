"""Font texture loader and processor.

Author: Matthew Matl
"""

import freetype
import numpy as np
import os

import OpenGL
from OpenGL.GL import *

from .constants import TextAlign, FLOAT_SZ
from .texture import Texture
from .sampler import Sampler


class FontCache(object):
    """A cache for fonts."""

    def __init__(self, font_dir=None):
        self._font_cache = {}
        self.font_dir = font_dir
        if self.font_dir is None:
            base_dir, _ = os.path.split(os.path.realpath(__file__))
            self.font_dir = os.path.join(base_dir, "fonts")

    def get_font(self, font_name, font_pt):
        # If it's a file, load it directly, else, try to load from font dir.
        if os.path.isfile(font_name):
            font_filename = font_name
            _, font_name = os.path.split(font_name)
            font_name, _ = os.path.split(font_name)
        else:
            font_filename = os.path.join(self.font_dir, font_name) + ".ttf"

        cid = OpenGL.contextdata.getContext()
        key = (cid, font_name, int(font_pt))

        if key not in self._font_cache:
            self._font_cache[key] = Font(font_filename, font_pt)
        return self._font_cache[key]

    def clear(self):
        for key in self._font_cache:
            try:
                self._font_cache[key].delete()
            except OpenGL.error.GLError:
                pass
        self._font_cache.clear()


class Font(object):
    """A font object.

    Parameters
    ----------
    font_file : str
        The file to load the font from.
    font_pt : int
        The height of the font in pixels.
    """

    def __init__(self, font_file, font_pt=40):
        self.font_file = font_file
        self.font_pt = int(font_pt)
        self._face = freetype.Face(font_file)
        self._face.set_pixel_sizes(0, font_pt)
        # The glyphs of the 128 ASCII characters are packed side by side into one texture, so that a string is drawn
        # in a single call whatever its length. Each glyph is bottom-aligned in the atlas, which the texture flips
        # upright (see 'Texture'), so that the bottom of a glyph lies at v = 0. A blank column separates two glyphs, so
        # that linear filtering never blends one into its neighbour.
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
        atlas_height = max(max(bitmap.shape[0] for bitmap in bitmaps), 1)
        atlas_width = sum(bitmap.shape[1] + 1 for bitmap in bitmaps)
        atlas = np.zeros((atlas_height, atlas_width), dtype=np.float32)
        # The texture coordinates of each glyph in the atlas: left, bottom, right and top
        self._glyph_uv = np.zeros((128, 4), dtype=np.float32)
        x = 0
        for i, bitmap in enumerate(bitmaps):
            rows, width = bitmap.shape
            atlas[atlas_height - rows :, x : x + width] = bitmap
            self._glyph_uv[i] = (x / atlas_width, 0.0, (x + width) / atlas_width, rows / atlas_height)
            x += width + 1
        filter_mode = GL_NEAREST if "PYTEST_VERSION" in os.environ else GL_LINEAR
        sampler = Sampler(magFilter=filter_mode, minFilter=filter_mode, wrapS=GL_CLAMP_TO_EDGE, wrapT=GL_CLAMP_TO_EDGE)
        self._atlas = Texture(sampler=sampler, source=atlas, source_channels="R")

        self._vbo = None
        self._vao = None

    @property
    def font_file(self):
        """str : The file the font was loaded from."""
        return self._font_file

    @font_file.setter
    def font_file(self, value):
        self._font_file = value

    @property
    def font_pt(self):
        """int : The height of the font in pixels."""
        return self._font_pt

    @font_pt.setter
    def font_pt(self, value):
        self._font_pt = int(value)

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

    def render_string(self, text, x, y, scale=1.0, align=TextAlign.BOTTOM_LEFT):
        """Render a string to the current view buffer.

        Note
        ----
        Assumes correct shader program already bound w/ uniforms set.

        Parameters
        ----------
        text : str
            The text to render.
        x : int
            Horizontal pixel location of text.
        y : int
            Vertical pixel location of text.
        scale : int
            Scaling factor for text.
        align : int
            One of the TextAlign options which specifies where the ``x``
            and ``y`` parameters lie on the text. For example,
            :attr:`.TextAlign.BOTTOM_LEFT` means that ``x`` and ``y`` indicate
            the position of the bottom-left corner of the textbox.
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
        size = self._glyph_size[codes] * scale
        bearing = self._glyph_bearing[codes] * scale
        advance = self._glyph_advance[codes] * scale

        # Determine width and height of text relative to x, y
        width = advance.sum()
        height = max(bearing[:, 1].max(), 0.0)

        # Determine offsets based on alignments
        xoff = 0
        yoff = 0
        if align == TextAlign.BOTTOM_RIGHT:
            xoff = -width
        elif align == TextAlign.BOTTOM_CENTER:
            xoff = -width / 2.0
        elif align == TextAlign.TOP_LEFT:
            yoff = -height
        elif align == TextAlign.TOP_RIGHT:
            yoff = -height
            xoff = -width
        elif align == TextAlign.TOP_CENTER:
            yoff = -height
            xoff = -width / 2.0
        elif align == TextAlign.CENTER:
            xoff = -width / 2.0
            yoff = -height / 2.0
        elif align == TextAlign.CENTER_LEFT:
            yoff = -height / 2.0
        elif align == TextAlign.CENTER_RIGHT:
            xoff = -width
            yoff = -height / 2.0

        x += xoff
        y += yoff

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
