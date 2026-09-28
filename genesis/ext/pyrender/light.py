"""Punctual light sources as defined by the glTF 2.0 KHR extension at
https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual

Author: Matthew Matl
"""

import abc
from abc import ABCMeta

import numpy as np

from OpenGL.GL import *

from .utils import format_color_vector
from .texture import Texture, CubeMapTexture
from .constants import SHADOW_TEX_SZ
from .camera import OrthographicCamera, PerspectiveCamera


class Light(metaclass=ABCMeta):
    """Base class for all light objects.

    Parameters
    ----------
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend on the
        type of light. Point and spot lights use luminous intensity in candela
        (lm/sr), while directional lights use illuminance in lux (lm/m2).
    name : str, optional
        Name of the light.
    """

    def __init__(self, color=None, intensity=None, name=None):
        if color is None:
            color = np.ones(3)
        if intensity is None:
            intensity = 1.0

        self.name = name
        self.color = color
        self.intensity = intensity
        self._shadow_camera = None
        self._shadow_texture = None
        # (scene revision, environment index) the shadow texture holds a map for, None until one is rendered. See
        # Scene.revision.
        self.shadow_map_revision = None

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
    def color(self):
        """(3,) float : The light's color."""
        return self._color

    @color.setter
    def color(self, value):
        self._color = format_color_vector(value, 3)

    @property
    def intensity(self):
        """float : The light's intensity in candela or lux."""
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = float(value)

    @property
    def shadow_texture(self):
        """:class:`.Texture` : A texture used to hold shadow maps for this light."""
        return self._shadow_texture

    @shadow_texture.setter
    def shadow_texture(self, value):
        if self._shadow_texture is not None:
            if self._shadow_texture._in_context():
                self._shadow_texture.delete()
        self._shadow_texture = value

    @abc.abstractmethod
    def _generate_shadow_texture(self, size=None):
        """Generate a shadow texture for this light.

        Parameters
        ----------
        size : int, optional
            Size of texture map. Must be a positive power of two.
        """
        pass

    @abc.abstractmethod
    def _get_shadow_camera(self, scene_scale):
        """Generate and return a shadow mapping camera for this light.

        Parameters
        ----------
        scene_scale : float
            Length of scene's bounding box diagonal.

        Returns
        -------
        camera : :class:`.Camera`
            The camera used to render shadowmaps for this light.
        """
        pass


class DirectionalLight(Light):
    """Directional lights are light sources that act as though they are
    infinitely far away and emit light in the direction of the local -z axis.
    This light type inherits the orientation of the node that it belongs to;
    position and scale are ignored except for their effect on the inherited
    node orientation. Because it is at an infinite distance, the light is
    not attenuated. Its intensity is defined in lumens per metre squared,
    or lux (lm/m2).

    Parameters
    ----------
    color : (3,) float, optional
        RGB value for the light's color in linear space. Defaults to white
        (i.e. [1.0, 1.0, 1.0]).
    intensity : float, optional
        Brightness of light, in lux (lm/m^2). Defaults to 1.0
    name : str, optional
        Name of the light.
    """

    def __init__(self, color=None, intensity=None, name=None):
        super().__init__(color=color, intensity=intensity, name=name)

    def _generate_shadow_texture(self, size=None):
        """Generate a shadow texture for this light.

        Parameters
        ----------
        size : int, optional
            Size of texture map. Must be a positive power of two.
        """
        if size is None:
            size = SHADOW_TEX_SZ
        self.shadow_texture = Texture(width=size, height=size, source_channels="D", data_format=GL_UNSIGNED_SHORT)

    def _get_shadow_camera(self, scene_scale):
        """Generate and return a shadow mapping camera for this light.

        Parameters
        ----------
        scene_scale : float
            Length of scene's bounding box diagonal.

        Returns
        -------
        camera : :class:`.Camera`
            The camera used to render shadowmaps for this light.
        """
        if scene_scale < 1e-6:
            scene_scale = 1.0
        return OrthographicCamera(znear=0.01 * scene_scale, zfar=10 * scene_scale, xmag=scene_scale, ymag=scene_scale)


class PointLight(Light):
    """Point lights emit light in all directions from their position in space;
    rotation and scale are ignored except for their effect on the inherited
    node position. The brightness of the light attenuates in a physically
    correct manner as distance increases from the light's position (i.e.
    brightness goes like the inverse square of the distance). Point light
    intensity is defined in candela, which is lumens per square radian (lm/sr).

    Parameters
    ----------
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of light in candela (lm/sr).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. If None, the range is assumed to be infinite.
    name : str, optional
        Name of the light.
    """

    def __init__(self, color=None, intensity=None, range=None, name=None):
        super().__init__(color=color, intensity=intensity, name=name)
        self.range = range

    @property
    def range(self):
        """float : The cutoff distance for the light."""
        return self._range

    @range.setter
    def range(self, value):
        if value is not None:
            value = float(value)
            if value <= 0:
                raise ValueError("Range must be > 0")
            self._range = value
        self._range = value

    def _generate_shadow_texture(self, size=None):
        """Generate a shadow texture for this light.

        Parameters
        ----------
        size : int, optional
            Size of texture map. Must be a positive power of two.
        """
        if size is None:
            size = SHADOW_TEX_SZ
        self.shadow_texture = CubeMapTexture(size, size)

    def _get_shadow_camera(self, scene_scale):
        """Generate and return a shadow mapping camera for this light.

        Parameters
        ----------
        scene_scale : float
            Length of scene's bounding box diagonal.

        Returns
        -------
        camera : :class:`.Camera`
            The camera used to render shadowmaps for this light.
        """
        return PerspectiveCamera(znear=0.1, zfar=25.0, yfov=np.clip(np.pi / 2, 0.0, np.pi), aspectRatio=1.0)

    def _get_view_matrices(self, position):
        def look_at(pos, target, up):
            z = pos - target
            z /= np.linalg.norm(z)
            x = np.cross(up, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            return np.array(
                [
                    [x[0], x[1], x[2], -np.dot(pos, x)],
                    [y[0], y[1], y[2], -np.dot(pos, y)],
                    [z[0], z[1], z[2], -np.dot(pos, z)],
                    [0, 0, 0, 1],
                ]
            )

        return np.stack(
            [
                look_at(position, position + np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
                look_at(position, position + np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
                look_at(position, position + np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
                look_at(position, position + np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
                look_at(position, position + np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])),
                look_at(position, position + np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0])),
            ],
            axis=0,
        )


class SpotLight(Light):
    """Spot lights emit light in a cone in the direction of the local -z axis.
    The angle and falloff of the cone is defined using two numbers, the
    ``innerConeAngle`` and ``outerConeAngle``.
    As with point lights, the brightness
    also attenuates in a physically correct manner as distance increases from
    the light's position (i.e. brightness goes like the inverse square of the
    distance). Spot light intensity refers to the brightness inside the
    ``innerConeAngle`` (and at the location of the light) and is defined in
    candela, which is lumens per square radian (lm/sr). A spot light's position
    and orientation are inherited from its node transform. Inherited scale does
    not affect cone shape, and is ignored except for its effect on position
    and orientation.

    Parameters
    ----------
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of light in candela (lm/sr).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. If None, the range is assumed to be infinite.
    innerConeAngle : float
        Angle, in radians, from centre of spotlight where falloff begins.
        Must be greater than or equal to ``0`` and less
        than ``outerConeAngle``. Defaults to ``0``.
    outerConeAngle : float
        Angle, in radians, from centre of spotlight where falloff ends.
        Must be greater than ``innerConeAngle`` and less than or equal to
        ``PI / 2.0``. Defaults to ``PI / 4.0``.
    name : str, optional
        Name of the light.
    """

    def __init__(
        self, color=None, intensity=None, range=None, innerConeAngle=0.0, outerConeAngle=(np.pi / 4.0), name=None
    ):
        super().__init__(name=name, color=color, intensity=intensity)
        self.outerConeAngle = outerConeAngle
        self.innerConeAngle = innerConeAngle
        self.range = range

    @property
    def innerConeAngle(self):
        """float : The inner cone angle in radians."""
        return self._innerConeAngle

    @innerConeAngle.setter
    def innerConeAngle(self, value):
        if value < 0.0 or value > self.outerConeAngle:
            raise ValueError("Invalid value for inner cone angle")
        self._innerConeAngle = float(value)

    @property
    def outerConeAngle(self):
        """float : The outer cone angle in radians."""
        return self._outerConeAngle

    @outerConeAngle.setter
    def outerConeAngle(self, value):
        if value < 0.0 or value > np.pi / 2.0 + 1e-9:
            raise ValueError("Invalid value for outer cone angle")
        self._outerConeAngle = float(value)

    @property
    def range(self):
        """float : The cutoff distance for the light."""
        return self._range

    @range.setter
    def range(self, value):
        if value is not None:
            value = float(value)
            if value <= 0:
                raise ValueError("Range must be > 0")
            self._range = value
        self._range = value

    def _generate_shadow_texture(self, size=None):
        """Generate a shadow texture for this light.

        Parameters
        ----------
        size : int, optional
            Size of texture map. Must be a positive power of two.
        """
        if size is None:
            size = SHADOW_TEX_SZ
        self.shadow_texture = Texture(width=size, height=size, source_channels="D", data_format=GL_UNSIGNED_SHORT)

    def _get_shadow_camera(self, scene_scale):
        """Generate and return a shadow mapping camera for this light.

        Parameters
        ----------
        scene_scale : float
            Length of scene's bounding box diagonal.

        Returns
        -------
        camera : :class:`.Camera`
            The camera used to render shadowmaps for this light.
        """
        return PerspectiveCamera(
            znear=0.01 * scene_scale,
            zfar=10 * scene_scale,
            yfov=np.clip(2 * self.outerConeAngle + np.pi / 16.0, 0.0, np.pi),
            aspectRatio=1.0,
        )


__all__ = ["Light", "DirectionalLight", "SpotLight", "PointLight"]
