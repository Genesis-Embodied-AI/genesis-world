"""Material properties, conforming to the glTF 2.0 standards as specified in
https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-material
and
https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness

Author: Matthew Matl
"""

from abc import ABCMeta

import numpy as np

from .constants import TexFlags
from .utils import format_color_vector, format_texture_source
from .texture import Texture


class Material(metaclass=ABCMeta):
    """Base for standard glTF 2.0 materials.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    normalTexture : (n,n,3) float or :class:`Texture`, optional
        A tangent space normal map. The texture contains RGB components in
        linear space. Each texel represents the XYZ components of a normal
        vector in tangent space. Red [0 to 255] maps to X [-1 to 1]. Green
        [0 to 255] maps to Y [-1 to 1]. Blue [128 to 255] maps to Z
        [1/255 to 1]. The normal vectors use OpenGL conventions where +X is
        right and +Y is up. +Z points toward the viewer.
    occlusionTexture : (n,n,1) float or :class:`Texture`, optional
        The occlusion map texture. The occlusion values are sampled from the R
        channel. Higher values indicate areas that should receive full indirect
        lighting and lower values indicate no indirect lighting. These values
        are linear. If other channels are present (GBA), they are ignored for
        occlusion calculations.
    emissiveTexture : (n,n,3) float or :class:`Texture`, optional
        The emissive map controls the color and intensity of the light being
        emitted by the material. This texture contains RGB components in sRGB
        color space. If a fourth component (A) is present, it is ignored.
    emissiveFactor : (3,) float, optional
        The RGB components of the emissive color of the material. These values
        are linear. If an emissiveTexture is specified, this value is
        multiplied with the texel values.
    alphaMode : str, optional
        The material's alpha rendering mode enumeration specifying the
        interpretation of the alpha value of the main factor and texture.
        Allowed Values:

        - `"OPAQUE"` The alpha value is ignored and the rendered output is
          fully opaque.
        - `"MASK"` The rendered output is either fully opaque or fully
          transparent depending on the alpha value and the specified alpha
          cutoff value.
        - `"BLEND"` The alpha value is used to composite the source and
          destination areas. The rendered output is combined with the
          background using the normal painting operation (i.e. the Porter
          and Duff over operator).

    alphaCutoff : float, optional
        Specifies the cutoff threshold when in MASK mode. If the alpha value is
        greater than or equal to this value then it is rendered as fully
        opaque, otherwise, it is rendered as fully transparent.
        A value greater than 1.0 will render the entire material as fully
        transparent. This value is ignored for other modes.
    doubleSided : bool, optional
        Specifies whether the material is double sided. When this value is
        false, back-face culling is enabled. When this value is true,
        back-face culling is disabled and double sided lighting is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal
        per vertex and face indexing.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    """

    def __init__(
        self,
        name=None,
        normalTexture=None,
        occlusionTexture=None,
        emissiveTexture=None,
        emissiveFactor=None,
        alphaMode=None,
        alphaCutoff=None,
        doubleSided=False,
        smooth=True,
        wireframe=False,
    ):
        # Set defaults
        if alphaMode is None:
            alphaMode = "OPAQUE"

        if alphaCutoff is None:
            alphaCutoff = 0.5

        if emissiveFactor is None:
            emissiveFactor = np.zeros((3,), dtype=np.float32)

        self.name = name
        self.normalTexture = normalTexture
        self.occlusionTexture = occlusionTexture
        self.emissiveTexture = emissiveTexture
        self.emissiveFactor = emissiveFactor
        self.alphaMode = alphaMode
        self.alphaCutoff = alphaCutoff
        self.doubleSided = doubleSided
        self.smooth = smooth
        self.wireframe = wireframe

        self._tex_flags = None

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
    def normalTexture(self):
        """(n,n,3) float or :class:`Texture` : The tangent-space normal map."""
        return self._normalTexture

    @normalTexture.setter
    def normalTexture(self, value):
        # TODO TMP
        self._normalTexture = self._format_texture(value, "RGB")
        self._tex_flags = None

    @property
    def occlusionTexture(self):
        """(n,n,1) float or :class:`Texture` : The ambient occlusion map."""
        return self._occlusionTexture

    @occlusionTexture.setter
    def occlusionTexture(self, value):
        self._occlusionTexture = self._format_texture(value, "R")
        self._tex_flags = None

    @property
    def emissiveTexture(self):
        """(n,n,3) float or :class:`Texture` : The emission map."""
        return self._emissiveTexture

    @emissiveTexture.setter
    def emissiveTexture(self, value):
        self._emissiveTexture = self._format_texture(value, "RGB")
        self._tex_flags = None

    @property
    def emissiveFactor(self):
        """(3,) float : Base multiplier for emission colors."""
        return self._emissiveFactor

    @emissiveFactor.setter
    def emissiveFactor(self, value):
        if value is None:
            value = np.zeros(3)
        self._emissiveFactor = format_color_vector(value, 3)

    @property
    def alphaMode(self):
        """str : The mode for blending."""
        return self._alphaMode

    @alphaMode.setter
    def alphaMode(self, value):
        if value not in set(["OPAQUE", "MASK", "BLEND"]):
            raise ValueError("Invalid alpha mode {}".format(value))
        self._alphaMode = value

    @property
    def alphaCutoff(self):
        """float : The cutoff threshold in MASK mode."""
        return self._alphaCutoff

    @alphaCutoff.setter
    def alphaCutoff(self, value):
        if value < 0 or value > 1:
            raise ValueError("Alpha cutoff must be in range [0,1]")
        self._alphaCutoff = float(value)

    @property
    def doubleSided(self):
        """bool : Whether the material is double-sided."""
        return self._doubleSided

    @doubleSided.setter
    def doubleSided(self, value):
        if not isinstance(value, bool):
            raise TypeError("Double sided must be a boolean value")
        self._doubleSided = value

    @property
    def smooth(self):
        """bool : Whether to render the mesh smoothly by
        interpolating vertex normals.
        """
        return self._smooth

    @smooth.setter
    def smooth(self, value):
        if not isinstance(value, bool):
            raise TypeError("Double sided must be a boolean value")
        self._smooth = value

    @property
    def wireframe(self):
        """bool : Whether to render the mesh in wireframe mode."""
        return self._wireframe

    @wireframe.setter
    def wireframe(self, value):
        if not isinstance(value, bool):
            raise TypeError("Wireframe must be a boolean value")
        self._wireframe = value

    @property
    def is_transparent(self):
        """bool : If True, the object is partially transparent."""
        return self._compute_transparency()

    @property
    def has_translucent_color(self):
        """bool : If True, the color factor is partially transparent, whatever the alpha of the textures."""
        return self._compute_transparency(include_textures=False)

    @property
    def tex_flags(self):
        """int : Texture availability flags."""
        if self._tex_flags is None:
            self._tex_flags = self._compute_tex_flags()
        return self._tex_flags

    @property
    def textures(self):
        """list of :class:`Texture` : The textures associated with this
        material.
        """
        return self._compute_textures()

    def _compute_transparency(self, include_textures=True):
        return False

    def _compute_tex_flags(self):
        tex_flags = TexFlags.NONE
        if self.normalTexture is not None:
            tex_flags |= TexFlags.NORMAL
        if self.occlusionTexture is not None:
            tex_flags |= TexFlags.OCCLUSION
        if self.emissiveTexture is not None:
            tex_flags |= TexFlags.EMISSIVE
        return tex_flags

    def _compute_textures(self):
        all_textures = [self.normalTexture, self.occlusionTexture, self.emissiveTexture]
        textures = set([t for t in all_textures if t is not None])
        return textures

    def _format_texture(self, texture, target_channels="RGB"):
        """Format a texture as a float32 np array."""
        if isinstance(texture, Texture) or texture is None:
            return texture
        else:
            source = format_texture_source(texture, target_channels)
            return Texture(source=source, source_channels=target_channels)


class MetallicRoughnessMaterial(Material):
    """A material based on the metallic-roughness material model from
    Physically-Based Rendering (PBR) methodology.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    normalTexture : (n,n,3) float or :class:`Texture`, optional
        A tangent space normal map. The texture contains RGB components in
        linear space. Each texel represents the XYZ components of a normal
        vector in tangent space. Red [0 to 255] maps to X [-1 to 1]. Green
        [0 to 255] maps to Y [-1 to 1]. Blue [128 to 255] maps to Z
        [1/255 to 1]. The normal vectors use OpenGL conventions where +X is
        right and +Y is up. +Z points toward the viewer.
    occlusionTexture : (n,n,1) float or :class:`Texture`, optional
        The occlusion map texture. The occlusion values are sampled from the R
        channel. Higher values indicate areas that should receive full indirect
        lighting and lower values indicate no indirect lighting. These values
        are linear. If other channels are present (GBA), they are ignored for
        occlusion calculations.
    emissiveTexture : (n,n,3) float or :class:`Texture`, optional
        The emissive map controls the color and intensity of the light being
        emitted by the material. This texture contains RGB components in sRGB
        color space. If a fourth component (A) is present, it is ignored.
    emissiveFactor : (3,) float, optional
        The RGB components of the emissive color of the material. These values
        are linear. If an emissiveTexture is specified, this value is
        multiplied with the texel values.
    alphaMode : str, optional
        The material's alpha rendering mode enumeration specifying the
        interpretation of the alpha value of the main factor and texture.
        Allowed Values:

        - `"OPAQUE"` The alpha value is ignored and the rendered output is
          fully opaque.
        - `"MASK"` The rendered output is either fully opaque or fully
          transparent depending on the alpha value and the specified alpha
          cutoff value.
        - `"BLEND"` The alpha value is used to composite the source and
          destination areas. The rendered output is combined with the
          background using the normal painting operation (i.e. the Porter
          and Duff over operator).

    alphaCutoff : float, optional
        Specifies the cutoff threshold when in MASK mode. If the alpha value is
        greater than or equal to this value then it is rendered as fully
        opaque, otherwise, it is rendered as fully transparent.
        A value greater than 1.0 will render the entire material as fully
        transparent. This value is ignored for other modes.
    doubleSided : bool, optional
        Specifies whether the material is double sided. When this value is
        false, back-face culling is enabled. When this value is true,
        back-face culling is disabled and double sided lighting is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal
        per vertex and face indexing.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    baseColorFactor : (4,) float, optional
        The RGBA components of the base color of the material. The fourth
        component (A) is the alpha coverage of the material. The alphaMode
        property specifies how alpha is interpreted. These values are linear.
        If a baseColorTexture is specified, this value is multiplied with the
        texel values.
    baseColorTexture : (n,n,4) float or :class:`Texture`, optional
        The base color texture. This texture contains RGB(A) components in sRGB
        color space. The first three components (RGB) specify the base color of
        the material. If the fourth component (A) is present, it represents the
        alpha coverage of the material. Otherwise, an alpha of 1.0 is assumed.
        The alphaMode property specifies how alpha is interpreted.
        The stored texels must not be premultiplied.
    metallicFactor : float
        The metalness of the material. A value of 1.0 means the material is a
        metal. A value of 0.0 means the material is a dielectric. Values in
        between are for blending between metals and dielectrics such as dirty
        metallic surfaces. This value is linear. If a metallicRoughnessTexture
        is specified, this value is multiplied with the metallic texel values.
    roughnessFactor : float
        The roughness of the material. A value of 1.0 means the material is
        completely rough. A value of 0.0 means the material is completely
        smooth. This value is linear. If a metallicRoughnessTexture is
        specified, this value is multiplied with the roughness texel values.
    metallicRoughnessTexture : (n,n,2) float or :class:`Texture`, optional
        The metallic-roughness texture. The metalness values are sampled from
        the B channel. The roughness values are sampled from the G channel.
        These values are linear. If other channels are present (R or A), they
        are ignored for metallic-roughness calculations.
    transmissionFactor : float, optional
        The share of the light that the material lets through, as glTF's transmission extension defines it. The light
        seen through the material is tinted by its base color, and the share it reflects grows at grazing angles.
        Defaults to 0.0.
    ior : float, optional
        The index of refraction of the material, as glTF's index of refraction extension defines it, which sets the
        share of the light a transmissive material reflects. Defaults to 1.5.
    """

    def __init__(
        self,
        name=None,
        normalTexture=None,
        occlusionTexture=None,
        emissiveTexture=None,
        emissiveFactor=None,
        alphaMode=None,
        alphaCutoff=None,
        doubleSided=False,
        smooth=True,
        wireframe=False,
        baseColorFactor=None,
        baseColorTexture=None,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        metallicRoughnessTexture=None,
        transmissionFactor=0.0,
        ior=1.5,
    ):
        super().__init__(
            name=name,
            normalTexture=normalTexture,
            occlusionTexture=occlusionTexture,
            emissiveTexture=emissiveTexture,
            emissiveFactor=emissiveFactor,
            alphaMode=alphaMode,
            alphaCutoff=alphaCutoff,
            doubleSided=doubleSided,
            smooth=smooth,
            wireframe=wireframe,
        )

        # Set defaults
        if baseColorFactor is None:
            baseColorFactor = np.ones((4,), dtype=np.float32)

        self.baseColorFactor = baseColorFactor
        self.baseColorTexture = baseColorTexture
        self.metallicFactor = metallicFactor
        self.roughnessFactor = roughnessFactor
        self.metallicRoughnessTexture = metallicRoughnessTexture
        self.transmissionFactor = transmissionFactor
        self.ior = ior

    @property
    def baseColorFactor(self):
        """(4,) float or :class:`Texture` : The RGBA base color multiplier."""
        return self._baseColorFactor

    @baseColorFactor.setter
    def baseColorFactor(self, value):
        if value is None:
            value = np.ones(4)
        self._baseColorFactor = format_color_vector(value, 4)

    @property
    def baseColorTexture(self):
        """(n,n,4) float or :class:`Texture` : The diffuse texture."""
        return self._baseColorTexture

    @baseColorTexture.setter
    def baseColorTexture(self, value):
        self._baseColorTexture = self._format_texture(value, "RGBA")
        self._tex_flags = None

    @property
    def metallicFactor(self):
        """float : The metalness of the material."""
        return self._metallicFactor

    @metallicFactor.setter
    def metallicFactor(self, value):
        if value is None:
            value = 1.0
        if value < 0 or value > 1:
            raise ValueError("Metallic factor must be in range [0,1]")
        self._metallicFactor = float(value)

    @property
    def roughnessFactor(self):
        """float : The roughness of the material."""
        return self.RoughnessFactor

    @roughnessFactor.setter
    def roughnessFactor(self, value):
        if value is None:
            value = 1.0
        if value < 0 or value > 1:
            raise ValueError("Roughness factor must be in range [0,1]")
        self.RoughnessFactor = float(value)

    @property
    def metallicRoughnessTexture(self):
        """(n,n,2) float or :class:`Texture` : The metallic-roughness texture."""
        return self._metallicRoughnessTexture

    @metallicRoughnessTexture.setter
    def metallicRoughnessTexture(self, value):
        self._metallicRoughnessTexture = self._format_texture(value, "GB")
        self._tex_flags = None

    @property
    def transmissionFactor(self):
        """float : The share of the light that the material lets through."""
        return self._transmissionFactor

    @transmissionFactor.setter
    def transmissionFactor(self, value):
        if value is None:
            value = 0.0
        if value < 0 or value > 1:
            raise ValueError("Transmission factor must be in range [0,1]")
        self._transmissionFactor = float(value)

    @property
    def ior(self):
        """float : The index of refraction of the material."""
        return self._ior

    @ior.setter
    def ior(self, value):
        if value is None:
            value = 1.5
        if value < 1:
            raise ValueError("Index of refraction must be greater than or equal to 1")
        self._ior = float(value)

    @property
    def is_transmissive(self):
        """bool : If True, the material lets part of the light through."""
        return self._transmissionFactor > 0.0

    def _compute_tex_flags(self):
        tex_flags = super(MetallicRoughnessMaterial, self)._compute_tex_flags()
        if self.baseColorTexture is not None:
            tex_flags |= TexFlags.BASE_COLOR
        if self.metallicRoughnessTexture is not None:
            tex_flags |= TexFlags.METALLIC_ROUGHNESS
        return tex_flags

    def _compute_transparency(self, include_textures=True):
        # The background shows through a transmissive material, so it is drawn after the opaque geometry, and its shadow
        # is that of a translucent color (see mesh_depth.frag)
        if self.is_transmissive:
            return True
        if self.alphaMode == "OPAQUE":
            return False
        cutoff = self.alphaCutoff
        if self.alphaMode == "BLEND":
            cutoff = 1.0
        if self.baseColorFactor[3] < cutoff:
            return True
        if include_textures and self.baseColorTexture is not None and self.baseColorTexture.is_transparent(cutoff):
            return True
        return False

    def _compute_textures(self):
        textures = super(MetallicRoughnessMaterial, self)._compute_textures()
        all_textures = [self.baseColorTexture, self.metallicRoughnessTexture]
        all_textures = {t for t in all_textures if t is not None}
        textures |= all_textures
        return textures


class SpecularGlossinessMaterial(Material):
    """A material based on the specular-glossiness material model from
    Physically-Based Rendering (PBR) methodology.

    Parameters
    ----------
    name : str, optional
        The user-defined name of this object.
    normalTexture : (n,n,3) float or :class:`Texture`, optional
        A tangent space normal map. The texture contains RGB components in
        linear space. Each texel represents the XYZ components of a normal
        vector in tangent space. Red [0 to 255] maps to X [-1 to 1]. Green
        [0 to 255] maps to Y [-1 to 1]. Blue [128 to 255] maps to Z
        [1/255 to 1]. The normal vectors use OpenGL conventions where +X is
        right and +Y is up. +Z points toward the viewer.
    occlusionTexture : (n,n,1) float or :class:`Texture`, optional
        The occlusion map texture. The occlusion values are sampled from the R
        channel. Higher values indicate areas that should receive full indirect
        lighting and lower values indicate no indirect lighting. These values
        are linear. If other channels are present (GBA), they are ignored for
        occlusion calculations.
    emissiveTexture : (n,n,3) float or :class:`Texture`, optional
        The emissive map controls the color and intensity of the light being
        emitted by the material. This texture contains RGB components in sRGB
        color space. If a fourth component (A) is present, it is ignored.
    emissiveFactor : (3,) float, optional
        The RGB components of the emissive color of the material. These values
        are linear. If an emissiveTexture is specified, this value is
        multiplied with the texel values.
    alphaMode : str, optional
        The material's alpha rendering mode enumeration specifying the
        interpretation of the alpha value of the main factor and texture.
        Allowed Values:

        - `"OPAQUE"` The alpha value is ignored and the rendered output is
          fully opaque.
        - `"MASK"` The rendered output is either fully opaque or fully
          transparent depending on the alpha value and the specified alpha
          cutoff value.
        - `"BLEND"` The alpha value is used to composite the source and
          destination areas. The rendered output is combined with the
          background using the normal painting operation (i.e. the Porter
          and Duff over operator).

    alphaCutoff : float, optional
        Specifies the cutoff threshold when in MASK mode. If the alpha value is
        greater than or equal to this value then it is rendered as fully
        opaque, otherwise, it is rendered as fully transparent.
        A value greater than 1.0 will render the entire material as fully
        transparent. This value is ignored for other modes.
    doubleSided : bool, optional
        Specifies whether the material is double sided. When this value is
        false, back-face culling is enabled. When this value is true,
        back-face culling is disabled and double sided lighting is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal
        per vertex and face indexing.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    diffuseFactor : (4,) float
        The RGBA components of the reflected diffuse color of the material.
        Metals have a diffuse value of [0.0, 0.0, 0.0]. The fourth component
        (A) is the opacity of the material. The values are linear.
    diffuseTexture : (n,n,4) float or :class:`Texture`, optional
        The diffuse texture. This texture contains RGB(A) components of the
        reflected diffuse color of the material in sRGB color space. If the
        fourth component (A) is present, it represents the alpha coverage of
        the material. Otherwise, an alpha of 1.0 is assumed.
        The alphaMode property specifies how alpha is interpreted.
        The stored texels must not be premultiplied.
    specularFactor : (3,) float
        The specular RGB color of the material. This value is linear.
    glossinessFactor : float
        The glossiness or smoothness of the material. A value of 1.0 means the
        material has full glossiness or is perfectly smooth. A value of 0.0
        means the material has no glossiness or is perfectly rough. This value
        is linear.
    specularGlossinessTexture : (n,n,4) or :class:`Texture`, optional
        The specular-glossiness texture is a RGBA texture, containing the
        specular color (RGB) in sRGB space and the glossiness value (A) in
        linear space.
    """

    def __init__(
        self,
        name=None,
        normalTexture=None,
        occlusionTexture=None,
        emissiveTexture=None,
        emissiveFactor=None,
        alphaMode=None,
        alphaCutoff=None,
        doubleSided=False,
        smooth=True,
        wireframe=False,
        diffuseFactor=None,
        diffuseTexture=None,
        specularFactor=None,
        glossinessFactor=1.0,
        specularGlossinessTexture=None,
    ):
        super().__init__(
            name=name,
            normalTexture=normalTexture,
            occlusionTexture=occlusionTexture,
            emissiveTexture=emissiveTexture,
            emissiveFactor=emissiveFactor,
            alphaMode=alphaMode,
            alphaCutoff=alphaCutoff,
            doubleSided=doubleSided,
            smooth=smooth,
            wireframe=wireframe,
        )

        # Set defaults
        if diffuseFactor is None:
            diffuseFactor = np.ones((4,), dtype=np.float32)
        if specularFactor is None:
            specularFactor = np.ones((3,), dtype=np.float32)

        self.diffuseFactor = diffuseFactor
        self.diffuseTexture = diffuseTexture
        self.specularFactor = specularFactor
        self.glossinessFactor = glossinessFactor
        self.specularGlossinessTexture = specularGlossinessTexture

    @property
    def diffuseFactor(self):
        """(4,) float : The diffuse base color."""
        return self._diffuseFactor

    @diffuseFactor.setter
    def diffuseFactor(self, value):
        self._diffuseFactor = format_color_vector(value, 4)

    @property
    def diffuseTexture(self):
        """(n,n,4) float or :class:`Texture` : The diffuse map."""
        return self._diffuseTexture

    @diffuseTexture.setter
    def diffuseTexture(self, value):
        self._diffuseTexture = self._format_texture(value, "RGBA")
        self._tex_flags = None

    @property
    def specularFactor(self):
        """(3,) float : The specular color of the material."""
        return self._specularFactor

    @specularFactor.setter
    def specularFactor(self, value):
        self._specularFactor = format_color_vector(value, 3)

    @property
    def glossinessFactor(self):
        """float : The glossiness of the material."""
        return self.glossinessFactor

    @glossinessFactor.setter
    def glossinessFactor(self, value):
        if value < 0 or value > 1:
            raise ValueError("glossiness factor must be in range [0,1]")
        self._glossinessFactor = float(value)

    @property
    def specularGlossinessTexture(self):
        """(n,n,4) or :class:`Texture` : The specular-glossiness texture."""
        return self._specularGlossinessTexture

    @specularGlossinessTexture.setter
    def specularGlossinessTexture(self, value):
        self._specularGlossinessTexture = self._format_texture(value, "GB")
        self._tex_flags = None

    def _compute_tex_flags(self):
        flags = super(SpecularGlossinessMaterial, self)._compute_tex_flags()
        if self.diffuseTexture is not None:
            flags |= TexFlags.DIFFUSE
        if self.specularGlossinessTexture is not None:
            flags |= TexFlags.SPECULAR_GLOSSINESS
        return flags

    def _compute_transparency(self, include_textures=True):
        if self.alphaMode == "OPAQUE":
            return False
        cutoff = self.alphaCutoff
        if self.alphaMode == "BLEND":
            cutoff = 1.0
        if self.diffuseFactor[3] < cutoff:
            return True
        if include_textures and self.diffuseTexture is not None and self.diffuseTexture.is_transparent(cutoff):
            return True
        return False

    def _compute_textures(self):
        textures = super(SpecularGlossinessMaterial, self)._compute_textures()
        all_textures = [self.diffuseTexture, self.specularGlossinessTexture]
        all_textures = {t for t in all_textures if t is not None}
        textures |= all_textures
        return textures
