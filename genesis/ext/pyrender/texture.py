"""Textures, conforming to the glTF 2.0 standards as specified in
https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-texture

Author: Matthew Matl
"""

import numpy as np

from OpenGL.GL import *
from OpenGL.GL.EXT import texture_filter_anisotropic

import genesis as gs

from .utils import format_texture_source
from .sampler import Sampler


class Texture(object):
    """A texture and its sampler.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    sampler : :class:`Sampler`
        The sampler used by this texture.
    source : (h,w,c) uint8 or (h,w,c) float or :class:`PIL.Image.Image`
        The image used by this texture. If None, the texture is created
        empty and width and height must be specified. A texture array takes
        its first (n,h,w,c) uint8 or float16 layers, the others being written
        by 'set_layers'.
    source_channels : str
        Either `D`, `R`, `RG`, `GB`, `RGB`, or `RGBA`. Indicates the
        channels to extract from `source`. Any missing channels will be filled
        with `1.0`.
    width : int, optional
        For empty textures, the width of the texture buffer.
    height : int, optional
        For empty textures, the height of the texture buffer.
    tex_type : int
        Either GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY or GL_TEXTURE_CUBE.
    data_format : int
        GL_UNSIGNED_BYTE, GL_HALF_FLOAT or GL_FLOAT.
    """

    def __init__(
        self,
        name=None,
        sampler=None,
        source=None,
        source_channels=None,
        width=None,
        height=None,
        tex_type=GL_TEXTURE_2D,
        data_format=GL_UNSIGNED_BYTE,
    ):
        self.source_channels = source_channels
        self.name = name
        self.sampler = sampler
        self.source = source
        self.width = width
        self.height = height
        self.tex_type = tex_type
        self.data_format = data_format

        self._texid = None
        self._is_transparent = False
        # The layers written by 'set_layers' and not uploaded yet, held only on the host until the next upload
        self._pending_layers: dict[int, np.ndarray] = {}

    @property
    def name(self):
        """str : The user-defined name of this object."""
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def sampler(self):
        """:class:`Sampler` : The sampler used by this texture."""
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        if value is None:
            value = Sampler()
        self._sampler = value

    @property
    def source(self):
        """(h,w,c) uint8 or float or :class:`PIL.Image.Image` : The image
        used in this texture.
        """
        return self._source

    @source.setter
    def source(self, value):
        if value is None:
            self._source = None
        elif isinstance(value, np.ndarray) and value.ndim == 4:
            self._source = np.ascontiguousarray(value[..., : len(self.source_channels)])
        else:
            self._source = format_texture_source(value, self.source_channels)
        self._is_transparent = False

    @property
    def source_channels(self):
        """str : The channels that were extracted from the original source."""
        return self._source_channels

    @source_channels.setter
    def source_channels(self, value):
        self._source_channels = value

    @property
    def width(self):
        """int : The width of the texture buffer."""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        """int : The height of the texture buffer."""
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def tex_type(self):
        """int : The type of the texture."""
        return self._tex_type

    @tex_type.setter
    def tex_type(self, value):
        self._tex_type = value

    @property
    def data_format(self):
        """int : The format of the texture data."""
        return self._data_format

    @data_format.setter
    def data_format(self, value):
        self._data_format = value

    def is_transparent(self, cutoff=1.0):
        """bool : If True, the texture is partially transparent."""
        if self._is_transparent is None:
            self._is_transparent = False
            if self.source_channels == "RGBA" and self.source is not None:
                if np.any(self.source[..., 3] < cutoff):
                    self._is_transparent = True
        return self._is_transparent

    def delete(self):
        """Remove this texture from the OpenGL context."""
        self._unbind()
        self._remove_from_context()

    ##################
    # OpenGL code
    ##################

    @property
    def _gl_format(self) -> int:
        """The OpenGL format of the channels of the texture."""
        if self.source_channels == "R":
            return GL_RED
        if self.source_channels in ("RG", "GB"):
            return GL_RG
        if self.source_channels == "RGB":
            return GL_RGB
        if self.source_channels == "RGBA":
            return GL_RGBA
        return GL_DEPTH_COMPONENT

    def set_layers(self, layers_idx: np.ndarray, images: np.ndarray) -> None:
        """Write (len(layers_idx),h,w,c) images into layers of a texture array, uploaded by 'upload_source'."""
        for layer, image in zip(layers_idx.tolist(), images[..., : len(self.source_channels)]):
            self._pending_layers[layer] = np.ascontiguousarray(image[::-1])

    def upload_source(self) -> None:
        """Upload the layers of a texture array written since its last upload, then drop them from the host.

        A texture not yet in the OpenGL context uploads them once it is added, so this does nothing then.
        """
        if self._texid is None:
            return
        glBindTexture(self.tex_type, self._texid)
        height, width = self.source.shape[1:3]
        # The layers are tightly packed (see '_add_to_context')
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        for layer, data in self._pending_layers.items():
            glTexSubImage3D(self.tex_type, 0, 0, 0, layer, width, height, 1, self._gl_format, self.data_format, data)
        self._pending_layers.clear()
        glBindTexture(self.tex_type, 0)

    def _add_to_context(self):
        if self._texid is not None:
            return
            # raise ValueError('Texture already loaded into OpenGL context')

        fmt = self._gl_format

        # Generate the OpenGL texture
        self._texid = glGenTextures(1)
        glBindTexture(self.tex_type, self._texid)

        # Flip data for OpenGL buffer
        data = None
        width = self.width
        height = self.height
        is_array = self.tex_type == GL_TEXTURE_2D_ARRAY
        if self.source is not None:
            data = np.flip(self.source, axis=1 if is_array else 0).reshape((-1,))
            height, width = self.source.shape[1:3] if is_array else self.source.shape[:2]

        # A 16-bit depth texture halves what sampling a shadow map reads, which is what bounds the cost of the shadow
        # comparisons at the default map size, and its quantization stays below the depth bias of those comparisons.
        internal_fmt = fmt
        if fmt == GL_DEPTH_COMPONENT and self.data_format == GL_UNSIGNED_SHORT:
            internal_fmt = GL_DEPTH_COMPONENT16
        # An unsized format would let the driver store half floats at 8 bits per channel
        if self.data_format == GL_HALF_FLOAT:
            internal_fmt = GL_RGBA16F if fmt == GL_RGBA else GL_RGB16F

        # The source rows are tightly packed, while OpenGL reads each row from a 4-byte boundary by default, which shifts
        # every row after the first where a row is no multiple of 4 bytes, as one of an odd width in RGB half floats
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # Bind texture and generate mipmaps
        if is_array:
            glTexImage3D(
                self.tex_type, 0, internal_fmt, width, height, len(self.source), 0, fmt, self.data_format, data
            )
        else:
            glTexImage2D(self.tex_type, 0, internal_fmt, width, height, 0, fmt, self.data_format, data)
        # A texture array has no mipmaps, since generating them covers every layer, which would make the upload of a
        # single layer cost as much as the whole array
        if self.source is not None and not is_array:
            glGenerateMipmap(self.tex_type)

        if self.sampler.magFilter is not None:
            glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, self.sampler.magFilter)
        else:
            if self.source is not None:
                glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            else:
                glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        if self.sampler.minFilter is not None:
            glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, self.sampler.minFilter)
        else:
            if is_array:
                glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            elif self.source is not None:
                glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            else:
                glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glTexParameteri(self.tex_type, GL_TEXTURE_WRAP_S, self.sampler.wrapS)
        glTexParameteri(self.tex_type, GL_TEXTURE_WRAP_T, self.sampler.wrapT)
        border_color = np.full((4,), fill_value=255, dtype=np.uint8)
        if fmt == GL_DEPTH_COMPONENT or self.data_format in (GL_FLOAT, GL_HALF_FLOAT):
            border_color = np.ones((4,), dtype=np.float32)
        glTexParameterfv(self.tex_type, GL_TEXTURE_BORDER_COLOR, border_color)

        if fmt == GL_DEPTH_COMPONENT:
            # A depth texture is a shadow map read through a sampler2DShadow: a fetch compares the fragment depth against
            # the texels and blends the comparisons bilinearly, further from the light counting as shadowed. The border
            # depth of 1.0 leaves fragments beyond the map lit.
            glTexParameteri(self.tex_type, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)
            glTexParameteri(self.tex_type, GL_TEXTURE_COMPARE_FUNC, GL_GREATER)
            glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(self.tex_type, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(self.tex_type, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        if texture_filter_anisotropic.glInitTextureFilterAnisotropicEXT():
            max_aniso = glGetFloat(texture_filter_anisotropic.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
            glTexParameterf(GL_TEXTURE_2D, texture_filter_anisotropic.GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso)
        else:
            gs.logger.debug("Current OpenGL context does not support anisotropic filtering. Disabling it...")

        # Unbind texture
        glBindTexture(self.tex_type, 0)
        if is_array:
            self.upload_source()

    def _remove_from_context(self):
        if self._texid is not None:
            try:
                # TODO OPENGL BUG?
                # glDeleteTextures(1, [self._texid])
                glDeleteTextures([self._texid])
            except OpenGL.error.Error:
                pass
            self._texid = None

    def _in_context(self):
        return self._texid is not None

    def _bind(self):
        # TODO HANDLE INDEXING INTO OTHER UV's
        glBindTexture(self.tex_type, self._texid)

    def _unbind(self):
        glBindTexture(self.tex_type, 0)

    def _bind_as_depth_attachment(self):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.tex_type, self._texid, 0)

    def _bind_as_color_attachment(self):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.tex_type, self._texid, 0)


class CubeMapTexture(object):
    """
    Cube map depth texture for point light shadow.
    """

    def __init__(self, width=None, height=None):
        self.width = width
        self.height = height

        self.tex_type = GL_TEXTURE_CUBE_MAP
        self.data_format = GL_FLOAT
        self._texid = None

    @property
    def width(self):
        """int : The width of the texture buffer."""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        """int : The height of the texture buffer."""
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    def delete(self):
        """Remove this texture from the OpenGL context."""
        self._unbind()
        self._remove_from_context()

    ##################
    # OpenGL code
    ##################
    def _add_to_context(self):
        if self._texid is not None:
            return
            # raise ValueError('Texture already loaded into OpenGL context')

        fmt = GL_DEPTH_COMPONENT

        # Generate the OpenGL texture
        self._texid = glGenTextures(1)
        glBindTexture(self.tex_type, self._texid)

        # Flip data for OpenGL buffer
        width = self.width
        height = self.height

        # Bind texture and generate mipmaps
        for i in range(6):
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, fmt, width, height, 0, fmt, self.data_format, None)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        # Unbind texture
        glBindTexture(self.tex_type, 0)

    def _remove_from_context(self):
        if self._texid is not None:
            try:
                # TODO OPENGL BUG?
                # glDeleteTextures(1, [self._texid])
                glDeleteTextures([self._texid])
            except OpenGL.error.Error:
                pass
            self._texid = None

    def _in_context(self):
        return self._texid is not None

    def _bind(self):
        glBindTexture(self.tex_type, self._texid)

    def _unbind(self):
        glBindTexture(self.tex_type, 0)

    def _bind_as_depth_attachment(self):
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self._texid, 0)
