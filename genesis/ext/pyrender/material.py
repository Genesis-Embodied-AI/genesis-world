"""Material of the primitives, following the metallic-roughness model of glTF 2.0 and its transmission and index of
refraction extensions."""

from .constants import TexChannels, TexFlags
from .texture import Texture
from .utils import format_color_vector, format_texture_source


class Material:
    """Physically based material of a primitive, following the metallic-roughness model of glTF 2.0.

    A texture is given as a :class:`Texture`, or as an image or an array that is turned into one. A factor left to None
    takes its default, as a glTF material that does not specify it.

    Parameters
    ----------
    normal_texture : (n, n, 3) float or Texture, optional
        The normal map in tangent space, in linear space, OpenGL conventions (+X right, +Y up, +Z towards the viewer).
        Defaults to None.
    occlusion_texture : (n, n) float or Texture, optional
        The ambient occlusion map, in linear space, the share of the ambient light and of the environment that reaches
        each point. Defaults to None.
    emissive_texture : (n, n, 3) float or Texture, optional
        The emission map, in sRGB space, scaled by 'emissive_factor'. Defaults to None.
    emissive_factor : (3,) float, optional
        The linear RGB color that the material emits. Defaults to black.
    is_blended : bool, optional
        Whether the material composites over what lies behind it by its alpha, rather than being drawn opaque whatever
        its alpha. Defaults to False.
    wireframe : bool, optional
        Whether the primitive is drawn as a wireframe. Defaults to False.
    base_color_factor : (4,) float, optional
        The linear RGBA base color, scaling 'base_color_texture'. Its alpha is the coverage of the material. Defaults to
        white.
    base_color_texture : (n, n, 4) float or Texture, optional
        The base color map, in sRGB space, with the coverage of the material in its alpha channel, not premultiplied.
        Defaults to None.
    metallic_factor : float, optional
        The metalness, 1.0 for a metal and 0.0 for a dielectric, scaling the blue channel of
        'metallic_roughness_texture'. Defaults to 1.0.
    roughness_factor : float, optional
        The roughness, from 0.0 for a polished surface to 1.0 for a rough one, scaling the green channel of
        'metallic_roughness_texture'. Defaults to 1.0.
    metallic_roughness_texture : (n, n, 2) float or Texture, optional
        The metalness and roughness map, in linear space. Defaults to None.
    transmission_factor : float, optional
        The share of the light that the material lets through, tinted by its base color, the share it reflects growing
        at grazing angles. Defaults to 0.0.
    ior : float, optional
        The index of refraction, which sets the share of the light that a transmissive material reflects. Defaults to
        1.5.
    """

    def __init__(
        self,
        normal_texture=None,
        occlusion_texture=None,
        emissive_texture=None,
        emissive_factor=None,
        is_blended=False,
        wireframe=False,
        base_color_factor=None,
        base_color_texture=None,
        metallic_factor=None,
        roughness_factor=None,
        metallic_roughness_texture=None,
        transmission_factor=0.0,
        ior=1.5,
    ):
        self.normal_texture = normal_texture
        self.occlusion_texture = occlusion_texture
        self.emissive_texture = emissive_texture
        self.emissive_factor = (0.0, 0.0, 0.0) if emissive_factor is None else emissive_factor
        self.is_blended = is_blended
        self.wireframe = wireframe
        self.base_color_factor = (1.0, 1.0, 1.0, 1.0) if base_color_factor is None else base_color_factor
        self.base_color_texture = base_color_texture
        self.metallic_factor = 1.0 if metallic_factor is None else metallic_factor
        self.roughness_factor = 1.0 if roughness_factor is None else roughness_factor
        self.metallic_roughness_texture = metallic_roughness_texture
        self.transmission_factor = transmission_factor
        self.ior = ior

    @property
    def normal_texture(self):
        """Texture : The normal map in tangent space."""
        return self._normal_texture

    @normal_texture.setter
    def normal_texture(self, value):
        self._normal_texture = _format_texture(value, TexChannels.RGB)

    @property
    def occlusion_texture(self):
        """Texture : The ambient occlusion map."""
        return self._occlusion_texture

    @occlusion_texture.setter
    def occlusion_texture(self, value):
        self._occlusion_texture = _format_texture(value, TexChannels.R)

    @property
    def emissive_texture(self):
        """Texture : The emission map."""
        return self._emissive_texture

    @emissive_texture.setter
    def emissive_texture(self, value):
        self._emissive_texture = _format_texture(value, TexChannels.RGB)

    @property
    def emissive_factor(self):
        """(3,) float : The linear RGB color that the material emits."""
        return self._emissive_factor

    @emissive_factor.setter
    def emissive_factor(self, value):
        self._emissive_factor = format_color_vector(value, 3)

    @property
    def base_color_factor(self):
        """(4,) float : The linear RGBA base color."""
        return self._base_color_factor

    @base_color_factor.setter
    def base_color_factor(self, value):
        self._base_color_factor = format_color_vector(value, 4)

    @property
    def base_color_texture(self):
        """Texture : The base color map."""
        return self._base_color_texture

    @base_color_texture.setter
    def base_color_texture(self, value):
        self._base_color_texture = _format_texture(value, TexChannels.RGBA)

    @property
    def metallic_roughness_texture(self):
        """Texture : The metalness and roughness map."""
        return self._metallic_roughness_texture

    @metallic_roughness_texture.setter
    def metallic_roughness_texture(self, value):
        self._metallic_roughness_texture = _format_texture(value, TexChannels.GB)

    @property
    def is_transmissive(self):
        """bool : Whether the material lets part of the light through."""
        return self.transmission_factor > 0.0

    @property
    def is_transparent(self):
        """bool : Whether the material shows part of what lies behind it."""
        # The background shows through a transmissive material, so it is drawn after the opaque geometry, and its shadow
        # is that of a translucent color (see mesh_depth.frag)
        if self.is_transmissive:
            return True
        if not self.is_blended:
            return False
        return self.base_color_factor[3] < 1.0 or (
            self.base_color_texture is not None and self.base_color_texture.is_transparent
        )

    @property
    def slots_texture(self):
        """tuple of Texture : The texture of each slot of the material, in the order of TexFlags, None if empty."""
        return (
            self.normal_texture,
            self.emissive_texture,
            self.base_color_texture,
            self.metallic_roughness_texture,
            self.occlusion_texture,
        )

    @property
    def tex_flags(self):
        """int : The flags of the textures that the material holds (see TexFlags)."""
        tex_flags = TexFlags.NONE
        for i_slot, texture in enumerate(self.slots_texture):
            if texture is not None:
                tex_flags |= 1 << i_slot
        return tex_flags

    @property
    def textures(self):
        """set of Texture : The textures that the material holds."""
        return {texture for texture in self.slots_texture if texture is not None}


def _format_texture(texture, target_channels):
    if texture is None or isinstance(texture, Texture):
        return texture
    return Texture(source=format_texture_source(texture, target_channels), source_channels=target_channels)
