"""Material of the primitives, following the metallic-roughness model of glTF 2.0 and its transmission and index of
refraction extensions."""

from .constants import TexFlags
from .texture import Texture
from .utils import format_color_vector, format_texture_source


class Material:
    """Physically based material of a primitive, following the metallic-roughness model of glTF 2.0.

    A texture is given as a :class:`Texture`, or as an image or an array that is turned into one. A factor left to None
    takes its default, as a glTF material that does not specify it.

    Parameters
    ----------
    normalTexture : (n, n, 3) float or Texture, optional
        The normal map in tangent space, in linear space, OpenGL conventions (+X right, +Y up, +Z towards the viewer).
        Defaults to None.
    occlusionTexture : (n, n, 1) float or Texture, optional
        The ambient occlusion map, in linear space, read from its red channel. Defaults to None.
    emissiveTexture : (n, n, 3) float or Texture, optional
        The emission map, in sRGB space, scaled by 'emissiveFactor'. Defaults to None.
    emissiveFactor : (3,) float, optional
        The linear RGB color that the material emits. Defaults to black.
    alphaMode : str, optional
        "OPAQUE" draws the material opaque whatever its alpha, and "BLEND" composites it over what lies behind it by
        its alpha. Defaults to "OPAQUE".
    wireframe : bool, optional
        Whether the primitive is drawn as a wireframe. Defaults to False.
    baseColorFactor : (4,) float, optional
        The linear RGBA base color, scaling 'baseColorTexture'. Its alpha is the coverage of the material. Defaults to
        white.
    baseColorTexture : (n, n, 4) float or Texture, optional
        The base color map, in sRGB space, with the coverage of the material in its alpha channel, not premultiplied.
        Defaults to None.
    metallicFactor : float, optional
        The metalness, 1.0 for a metal and 0.0 for a dielectric, scaling the blue channel of
        'metallicRoughnessTexture'. Defaults to 1.0.
    roughnessFactor : float, optional
        The roughness, from 0.0 for a polished surface to 1.0 for a rough one, scaling the green channel of
        'metallicRoughnessTexture'. Defaults to 1.0.
    metallicRoughnessTexture : (n, n, 2) float or Texture, optional
        The metalness and roughness map, in linear space. Defaults to None.
    transmissionFactor : float, optional
        The share of the light that the material lets through, tinted by its base color, the share it reflects growing
        at grazing angles. Defaults to 0.0.
    ior : float, optional
        The index of refraction, which sets the share of the light that a transmissive material reflects. Defaults to
        1.5.
    """

    def __init__(
        self,
        normalTexture=None,
        occlusionTexture=None,
        emissiveTexture=None,
        emissiveFactor=None,
        alphaMode="OPAQUE",
        wireframe=False,
        baseColorFactor=None,
        baseColorTexture=None,
        metallicFactor=None,
        roughnessFactor=None,
        metallicRoughnessTexture=None,
        transmissionFactor=0.0,
        ior=1.5,
    ):
        assert alphaMode in ("OPAQUE", "BLEND")
        self.normalTexture = normalTexture
        self.occlusionTexture = occlusionTexture
        self.emissiveTexture = emissiveTexture
        self.emissiveFactor = (0.0, 0.0, 0.0) if emissiveFactor is None else emissiveFactor
        self.alphaMode = alphaMode
        self.wireframe = wireframe
        self.baseColorFactor = (1.0, 1.0, 1.0, 1.0) if baseColorFactor is None else baseColorFactor
        self.baseColorTexture = baseColorTexture
        self.metallicFactor = 1.0 if metallicFactor is None else metallicFactor
        self.roughnessFactor = 1.0 if roughnessFactor is None else roughnessFactor
        self.metallicRoughnessTexture = metallicRoughnessTexture
        self.transmissionFactor = transmissionFactor
        self.ior = ior

    @property
    def normalTexture(self):
        """Texture : The normal map in tangent space."""
        return self._normalTexture

    @normalTexture.setter
    def normalTexture(self, value):
        self._normalTexture = _format_texture(value, "RGB")

    @property
    def occlusionTexture(self):
        """Texture : The ambient occlusion map."""
        return self._occlusionTexture

    @occlusionTexture.setter
    def occlusionTexture(self, value):
        self._occlusionTexture = _format_texture(value, "R")

    @property
    def emissiveTexture(self):
        """Texture : The emission map."""
        return self._emissiveTexture

    @emissiveTexture.setter
    def emissiveTexture(self, value):
        self._emissiveTexture = _format_texture(value, "RGB")

    @property
    def emissiveFactor(self):
        """(3,) float : The linear RGB color that the material emits."""
        return self._emissiveFactor

    @emissiveFactor.setter
    def emissiveFactor(self, value):
        self._emissiveFactor = format_color_vector(value, 3)

    @property
    def baseColorFactor(self):
        """(4,) float : The linear RGBA base color."""
        return self._baseColorFactor

    @baseColorFactor.setter
    def baseColorFactor(self, value):
        self._baseColorFactor = format_color_vector(value, 4)

    @property
    def baseColorTexture(self):
        """Texture : The base color map."""
        return self._baseColorTexture

    @baseColorTexture.setter
    def baseColorTexture(self, value):
        self._baseColorTexture = _format_texture(value, "RGBA")

    @property
    def metallicRoughnessTexture(self):
        """Texture : The metalness and roughness map."""
        return self._metallicRoughnessTexture

    @metallicRoughnessTexture.setter
    def metallicRoughnessTexture(self, value):
        self._metallicRoughnessTexture = _format_texture(value, "GB")

    @property
    def is_transmissive(self):
        """bool : Whether the material lets part of the light through."""
        return self.transmissionFactor > 0.0

    @property
    def is_transparent(self):
        """bool : Whether the material shows part of what lies behind it."""
        # The background shows through a transmissive material, so it is drawn after the opaque geometry, and its shadow
        # is that of a translucent color (see mesh_depth.frag)
        if self.is_transmissive:
            return True
        if self.alphaMode == "OPAQUE":
            return False
        return self.baseColorFactor[3] < 1.0 or (
            self.baseColorTexture is not None and self.baseColorTexture.is_transparent
        )

    @property
    def tex_flags(self):
        """int : The flags of the textures that the material holds (see TexFlags)."""
        tex_flags = TexFlags.NONE
        for texture, flag in (
            (self.normalTexture, TexFlags.NORMAL),
            (self.occlusionTexture, TexFlags.OCCLUSION),
            (self.emissiveTexture, TexFlags.EMISSIVE),
            (self.baseColorTexture, TexFlags.BASE_COLOR),
            (self.metallicRoughnessTexture, TexFlags.METALLIC_ROUGHNESS),
        ):
            if texture is not None:
                tex_flags |= flag
        return tex_flags

    @property
    def textures(self):
        """set of Texture : The textures that the material holds."""
        textures = (
            self.normalTexture,
            self.occlusionTexture,
            self.emissiveTexture,
            self.baseColorTexture,
            self.metallicRoughnessTexture,
        )
        return {texture for texture in textures if texture is not None}


def _format_texture(texture, target_channels):
    if texture is None or isinstance(texture, Texture):
        return texture
    return Texture(source=format_texture_source(texture, target_channels), source_channels=target_channels)
