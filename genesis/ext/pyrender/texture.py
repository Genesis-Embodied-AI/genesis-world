"""Textures of the rasterizer, as glTF 2.0 defines them."""

import numpy as np

from OpenGL.GL import (
    GL_CLAMP_TO_BORDER,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_COMPARE_REF_TO_TEXTURE,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT16,
    GL_DEPTH_COMPONENT32F,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_GREATER,
    GL_HALF_FLOAT,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_MAX_ARRAY_TEXTURE_LAYERS,
    GL_NEAREST,
    GL_RED,
    GL_RG,
    GL_RGB,
    GL_RGB16F,
    GL_RGB32F,
    GL_RGBA,
    GL_RGBA16F,
    GL_RGBA32F,
    GL_TEXTURE_2D,
    GL_TEXTURE_2D_ARRAY,
    GL_TEXTURE_BORDER_COLOR,
    GL_TEXTURE_COMPARE_FUNC,
    GL_TEXTURE_COMPARE_MODE,
    GL_TEXTURE_CUBE_MAP,
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_R,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_SHORT,
    OpenGL,
    glBindTexture,
    glDeleteTextures,
    glFramebufferTexture,
    glFramebufferTexture2D,
    glGenTextures,
    glGenerateMipmap,
    glGetFloat,
    glGetIntegerv,
    glPixelStorei,
    glTexImage2D,
    glTexImage3D,
    glTexParameterf,
    glTexParameterfv,
    glTexParameteri,
    glTexSubImage3D,
)
from OpenGL.GL.EXT import texture_filter_anisotropic

import genesis as gs

from .constants import TexChannels
from .sampler import Sampler
from .utils import format_texture_source


class Texture:
    """A texture and its sampler.

    Parameters
    ----------
    sampler : Sampler, optional
        The sampler of the texture. Defaults to a default Sampler.
    source : (h, w, c) uint8 or float or PIL.Image.Image, optional
        The image of the texture, None for an empty texture of the given size. A texture array takes (n, h, w, c) uint8
        or float16 layers, which 'set_layers' rewrites. Defaults to None.
    source_channels : TexChannels
        The channels that the texture holds, taken from the source, a missing channel filled with 1.0.
    width : int, optional
        The width of an empty texture. Defaults to None.
    height : int, optional
        The height of an empty texture. Defaults to None.
    tex_type : int, optional
        GL_TEXTURE_2D or GL_TEXTURE_2D_ARRAY. Defaults to GL_TEXTURE_2D.
    data_format : int, optional
        GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_HALF_FLOAT or GL_FLOAT. Defaults to GL_UNSIGNED_BYTE.
    """

    def __init__(
        self,
        sampler=None,
        source=None,
        source_channels=None,
        width=None,
        height=None,
        tex_type=GL_TEXTURE_2D,
        data_format=GL_UNSIGNED_BYTE,
    ):
        self.source_channels = source_channels
        self.sampler = Sampler() if sampler is None else sampler
        self.source = source
        self.width = width
        self.height = height
        self.tex_type = tex_type
        self.data_format = data_format

        self._texid = None
        # The layers written by 'set_layers' and not uploaded yet, held only on the host until the next upload
        self._pending_layers: dict[int, np.ndarray] = {}
        # The layers of a texture array that some primitive samples, None standing for every layer. 'is_transparent'
        # reads the alpha of these alone, since a layer left unshown may hold any image, such as the blank batched
        # variant of a dynamic texture before its first write.
        self.layers_shown: np.ndarray | None = None

    @property
    def source(self):
        """(h, w, c) uint8 or float : The image of the texture."""
        return self._source

    @source.setter
    def source(self, value):
        if value is None:
            self._source = None
        elif isinstance(value, np.ndarray) and value.ndim == 4:
            self._source = np.ascontiguousarray(value[..., : self.source_channels.n_channels])
        else:
            self._source = format_texture_source(value, self.source_channels)
        # The lowest alpha of each layer, in the scale of the source, which 'set_layers' keeps up to date since the host
        # drops the layers it writes once they are uploaded. Texels of zero alpha count, since filtering blends them
        # with their neighbours into partial alpha.
        self._layers_alpha_min = None
        if self.source_channels == TexChannels.RGBA and self._source is not None:
            self._layers_alpha_min = self._source[..., 3].min(axis=(-2, -1))

    @property
    def is_transparent(self):
        """bool : Whether some texels shown are translucent, to be blended with what lies behind them."""
        if self._layers_alpha_min is None:
            return False
        layers_alpha_min = (
            self._layers_alpha_min if self.layers_shown is None else self._layers_alpha_min[self.layers_shown]
        )
        # An 8-bit source stores an alpha of 1 as 255
        alpha_opaque = 255 if self._source.dtype == np.uint8 else 1.0
        return (layers_alpha_min < alpha_opaque).any()

    def delete(self):
        """Remove this texture from the OpenGL context."""
        self._unbind()
        self._remove_from_context()

    ##################
    # OpenGL code
    ##################

    @property
    def _gl_format(self) -> int:
        return (GL_RED, GL_RG, GL_RGB, GL_RGBA, GL_DEPTH_COMPONENT)[self.source_channels]

    def set_layers(self, layers_idx: np.ndarray, images: np.ndarray) -> None:
        """Write (len(layers_idx),h,w,c) images into layers of a texture array, uploaded by 'upload_layers'."""
        for layer, image in zip(layers_idx.tolist(), images[..., : self.source_channels.n_channels]):
            self._pending_layers[layer] = np.ascontiguousarray(image[::-1])
        if self._layers_alpha_min is not None:
            self._layers_alpha_min[layers_idx] = images[..., 3].min(axis=(-2, -1))

    def upload_layers(self) -> None:
        """Upload the layers of a texture array written since its last upload, then drop them from the host."""
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
        # comparisons at the default map size, and its quantization stays below the depth bias of those comparisons
        # wherever the depth is linear. A float depth texture serves a perspective projection, which packs the depth of
        # whatever lies far from its near plane close to 1.
        internal_fmt = fmt
        if fmt == GL_DEPTH_COMPONENT and self.data_format == GL_UNSIGNED_SHORT:
            internal_fmt = GL_DEPTH_COMPONENT16
        if fmt == GL_DEPTH_COMPONENT and self.data_format == GL_FLOAT:
            internal_fmt = GL_DEPTH_COMPONENT32F
        # An unsized format would let the driver store floats at 8 bits per channel
        if self.data_format == GL_HALF_FLOAT:
            internal_fmt = GL_RGBA16F if fmt == GL_RGBA else GL_RGB16F
        if self.data_format == GL_FLOAT and fmt in (GL_RGBA, GL_RGB):
            internal_fmt = GL_RGBA32F if fmt == GL_RGBA else GL_RGB32F

        # The source rows are tightly packed, while OpenGL reads each row from a 4-byte boundary by default, shifting
        # every row after the first where a row is no multiple of 4 bytes, as an odd width of RGB half floats does.
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # Bind texture and generate mipmaps
        if is_array:
            # A dynamic texture takes a layer per variant, and one per rendered environment for its batched variant
            max_layers = glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS)
            if len(self.source) > max_layers:
                gs.raise_exception(
                    f"A dynamic texture takes {len(self.source)} layers, more than the {max_layers} this GPU holds. "
                    "Use fewer variants, or for a texture with a batched variant, render fewer environments with "
                    "'rendered_envs_idx' of the visualization options."
                )
            glTexImage3D(
                self.tex_type, 0, internal_fmt, width, height, len(self.source), 0, fmt, self.data_format, data
            )
        else:
            glTexImage2D(self.tex_type, 0, internal_fmt, width, height, 0, fmt, self.data_format, data)
        # A texture array has no mipmaps, since generating them covers every layer, which would make the upload of a
        # single layer cost as much as the whole array.
        if self.source is not None and not is_array:
            glGenerateMipmap(self.tex_type)

        if self.sampler.mag_filter is not None:
            glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, self.sampler.mag_filter)
        else:
            if self.source is not None:
                glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            else:
                glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        if self.sampler.min_filter is not None:
            glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, self.sampler.min_filter)
        else:
            if is_array:
                glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            elif self.source is not None:
                glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            else:
                glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glTexParameteri(self.tex_type, GL_TEXTURE_WRAP_S, self.sampler.wrap_s)
        glTexParameteri(self.tex_type, GL_TEXTURE_WRAP_T, self.sampler.wrap_t)
        border_color = np.full((4,), fill_value=255, dtype=np.uint8)
        # A texture of floating-point values reads its border as given, while the others scale it to [0, 1]
        if fmt == GL_DEPTH_COMPONENT or self.data_format in (GL_FLOAT, GL_HALF_FLOAT):
            border_color = np.ones((4,), dtype=np.float32)
        glTexParameterfv(self.tex_type, GL_TEXTURE_BORDER_COLOR, border_color)

        if fmt == GL_DEPTH_COMPONENT:
            # A depth texture is a shadow map read through a sampler2DShadow: a fetch compares the fragment depth
            # against the texels and blends the comparisons bilinearly, further from the light counting as shadowed.
            # The border depth of 1.0 leaves fragments beyond the map lit.
            glTexParameteri(self.tex_type, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)
            glTexParameteri(self.tex_type, GL_TEXTURE_COMPARE_FUNC, GL_GREATER)
            glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(self.tex_type, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(self.tex_type, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        # Anisotropic filtering serves the images of surfaces seen at grazing angles. A texture that a pass renders into
        # is read in light or screen space, where anisotropic lookups differ between GPUs and software renderers.
        if self.source is not None:
            if texture_filter_anisotropic.glInitTextureFilterAnisotropicEXT():
                max_aniso = glGetFloat(texture_filter_anisotropic.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                glTexParameterf(self.tex_type, texture_filter_anisotropic.GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso)
            else:
                gs.logger.debug("Current OpenGL context does not support anisotropic filtering. Disabling it...")

        # Unbind texture
        glBindTexture(self.tex_type, 0)

    def _remove_from_context(self):
        if self._texid is not None:
            try:
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
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.tex_type, self._texid, 0)

    def _bind_as_color_attachment(self):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.tex_type, self._texid, 0)


class CubeMapTexture(Texture):
    """A cube map rendered into, the depth seen from a point light, or the color of the light passing through.

    Parameters
    ----------
    width : int
        The width of each face, in texels.
    height : int
        The height of each face, in texels.
    is_color : bool, optional
        Whether the cube map holds a color rather than a depth. Defaults to False.
    """

    def __init__(self, width, height, is_color=False):
        super().__init__(
            source_channels=TexChannels.RGBA if is_color else TexChannels.DEPTH,
            width=width,
            height=height,
            tex_type=GL_TEXTURE_CUBE_MAP,
            data_format=GL_HALF_FLOAT if is_color else GL_FLOAT,
        )
        self.is_color = is_color

    def _add_to_context(self):
        if self._texid is not None:
            return

        self._texid = glGenTextures(1)
        glBindTexture(self.tex_type, self._texid)
        internal_fmt = GL_RGBA16F if self.is_color else GL_DEPTH_COMPONENT
        for i_face in range(6):
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i_face,
                0,
                internal_fmt,
                self.width,
                self.height,
                0,
                self._gl_format,
                self.data_format,
                None,
            )

        # A color cube map is read filtered, the depth one compared texel by texel
        tex_filter = GL_LINEAR if self.is_color else GL_NEAREST
        glTexParameteri(self.tex_type, GL_TEXTURE_MAG_FILTER, tex_filter)
        glTexParameteri(self.tex_type, GL_TEXTURE_MIN_FILTER, tex_filter)
        for wrap_axis in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R):
            glTexParameteri(self.tex_type, wrap_axis, GL_CLAMP_TO_EDGE)
        glBindTexture(self.tex_type, 0)

    def _bind_as_depth_attachment(self):
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self._texid, 0)

    def _bind_as_color_attachment(self):
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self._texid, 0)
