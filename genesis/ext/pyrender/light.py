"""Punctual lights of the rasterizer, as the KHR_lights_punctual extension of glTF 2.0 defines them."""

from abc import ABC, abstractmethod

from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_FLOAT, GL_HALF_FLOAT, GL_LINEAR, GL_UNSIGNED_SHORT

from .constants import (
    DIRECTIONAL_SHADOW_TEX_SZ,
    POINT_SHADOW_TEX_SZ,
    POINT_TRANSMITTANCE_TEX_SZ,
    SPOT_SHADOW_TEX_SZ,
    TRANSMITTANCE_TEX_SZ,
    LightType,
    RenderFlags,
    TexChannels,
)
from .sampler import Sampler
from .texture import CubeMapTexture, Texture
from .utils import format_color_vector


class Light(ABC):
    """A light, which shines along the -z axis of its node when it has a direction.

    Parameters
    ----------
    color : (3,) float
        The RGB color of the light, in linear space.
    intensity : float
        The brightness of the light, in lux for a directional light and in candela for a point or a spot light.
    """

    light_type: int
    """The kind of the light (see LightType)."""
    shadow_flag: int
    """The render flag asking for the shadows of this kind of light (see RenderFlags)."""

    def __init__(self, color, intensity):
        self.color = format_color_vector(color, 3)
        self.intensity = intensity
        self._shadow_texture = None
        self._transmittance_texture = None
        # (scene revision, environment index) the shadow texture holds a map for, None until one is rendered. See
        # Scene.revision.
        self.shadow_map_revision = None

    @property
    def shadow_texture(self):
        """Texture or CubeMapTexture : The shadow map of the light, None until 'generate_shadow_texture'."""
        return self._shadow_texture

    @shadow_texture.setter
    def shadow_texture(self, value):
        if self._shadow_texture is not None and self._shadow_texture._in_context():
            self._shadow_texture.delete()
        self._shadow_texture = value

    @property
    def transmittance_texture(self):
        """Texture or CubeMapTexture : The share of each color of the light that transparent surfaces let through, and
        the depth of the nearest of them, None until 'generate_shadow_texture'."""
        return self._transmittance_texture

    @transmittance_texture.setter
    def transmittance_texture(self, value):
        if self._transmittance_texture is not None and self._transmittance_texture._in_context():
            self._transmittance_texture.delete()
        self._transmittance_texture = value

    @abstractmethod
    def generate_shadow_texture(self):
        """Create the shadow map and the transmittance map of the light, which the renderer then uploads."""


def _make_transmittance_texture(data_format):
    # The tint of the light varies slowly across a transparent surface, which a coarser map holds. Beyond the map, its
    # white border lets the light through untinted. Its alpha holds the depth of the nearest transparent surface, as the
    # shadow map does (see the shadow texture of each light for the precision it needs).
    return Texture(
        width=TRANSMITTANCE_TEX_SZ,
        height=TRANSMITTANCE_TEX_SZ,
        source_channels=TexChannels.RGBA,
        data_format=data_format,
        sampler=Sampler(
            mag_filter=GL_LINEAR, min_filter=GL_LINEAR, wrap_s=GL_CLAMP_TO_BORDER, wrap_t=GL_CLAMP_TO_BORDER
        ),
    )


class DirectionalLight(Light):
    """A light infinitely far away, whose rays are parallel and keep their intensity whatever the distance.

    Parameters
    ----------
    color : (3,) float
        The RGB color of the light, in linear space.
    intensity : float
        The illuminance of the light, in lux.
    """

    light_type = LightType.DIRECTIONAL
    shadow_flag = RenderFlags.SHADOWS_DIRECTIONAL

    def generate_shadow_texture(self):
        self.shadow_texture = Texture(
            width=DIRECTIONAL_SHADOW_TEX_SZ,
            height=DIRECTIONAL_SHADOW_TEX_SZ,
            source_channels=TexChannels.DEPTH,
            data_format=GL_UNSIGNED_SHORT,
        )
        self.transmittance_texture = _make_transmittance_texture(GL_HALF_FLOAT)


class PointLight(Light):
    """A light shining in every direction from its position, whose intensity decays as the inverse square distance.

    Parameters
    ----------
    color : (3,) float
        The RGB color of the light, in linear space.
    intensity : float
        The luminous intensity of the light, in candela.
    """

    light_type = LightType.POINT
    shadow_flag = RenderFlags.SHADOWS_POINT

    def generate_shadow_texture(self):
        self.shadow_texture = CubeMapTexture(POINT_SHADOW_TEX_SZ, POINT_SHADOW_TEX_SZ)
        # The tint of the light varies slowly across a transparent surface, which a coarser map holds
        self.transmittance_texture = CubeMapTexture(
            POINT_TRANSMITTANCE_TEX_SZ, POINT_TRANSMITTANCE_TEX_SZ, is_color=True
        )


class SpotLight(Light):
    """A light shining in a cone from its position, whose intensity decays as the inverse square distance.

    The intensity fades from the axis of the cone to its edge.

    Parameters
    ----------
    color : (3,) float
        The RGB color of the light, in linear space.
    intensity : float
        The luminous intensity of the light on the axis of its cone, in candela.
    cone_angle : float
        The angle between the axis of the cone and its edge, in radians, in (0, pi / 2).
    """

    light_type = LightType.SPOT
    shadow_flag = RenderFlags.SHADOWS_SPOT

    def __init__(self, color, intensity, cone_angle):
        super().__init__(color, intensity)
        self.cone_angle = cone_angle

    def generate_shadow_texture(self):
        # The shadow camera of a spot light projects in perspective from a near plane close to the light (see 'Texture')
        self.shadow_texture = Texture(
            width=SPOT_SHADOW_TEX_SZ,
            height=SPOT_SHADOW_TEX_SZ,
            source_channels=TexChannels.DEPTH,
            data_format=GL_FLOAT,
        )
        self.transmittance_texture = _make_transmittance_texture(GL_FLOAT)
