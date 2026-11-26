"""Cameras of the rasterizer, as glTF 2.0 defines them.

The pose of a camera follows the OpenGL convention: the camera looks along its -z axis, its x and y axes pointing to the
right and up in the image."""

import abc
from abc import ABCMeta

import numpy as np

from .constants import DEFAULT_Z_FAR, DEFAULT_Z_NEAR


class Camera(metaclass=ABCMeta):
    """Base class of the cameras of the rasterizer.

    Parameters
    ----------
    znear : float, optional
        The distance to the near clipping plane. Defaults to DEFAULT_Z_NEAR.
    zfar : float, optional
        The distance to the far clipping plane, beyond 'znear'. Defaults to DEFAULT_Z_FAR.
    """

    def __init__(self, znear=DEFAULT_Z_NEAR, zfar=DEFAULT_Z_FAR):
        assert znear >= 0.0 and (zfar is None or zfar > znear)
        self.znear = znear
        self.zfar = zfar

    @abc.abstractmethod
    def get_projection_matrix(self, width=None, height=None):
        """Return the OpenGL projection matrix of the camera.

        Parameters
        ----------
        width : int, optional
            The width of the viewport in pixels. Defaults to None.
        height : int, optional
            The height of the viewport in pixels. Defaults to None.
        """


class PerspectiveCamera(Camera):
    """Camera projecting in perspective.

    Parameters
    ----------
    yfov : float
        The vertical field of view in radians.
    znear : float, optional
        The distance to the near clipping plane. Defaults to DEFAULT_Z_NEAR.
    zfar : float, optional
        The distance to the far clipping plane, beyond 'znear', None for a projection reaching to infinity. Defaults to
        None.
    aspect_ratio : float, optional
        The ratio of the width to the height of the field of view, None for that of the viewport. Defaults to None.
    """

    def __init__(self, yfov, znear=DEFAULT_Z_NEAR, zfar=None, aspect_ratio=None):
        super().__init__(znear=znear, zfar=zfar)
        assert yfov > 0.0 and (aspect_ratio is None or aspect_ratio > 0.0)
        self.yfov = yfov
        self.aspect_ratio = aspect_ratio

    def get_projection_matrix(self, width=None, height=None):
        aspect_ratio = self.aspect_ratio
        if aspect_ratio is None:
            aspect_ratio = width / height
        t = np.tan(0.5 * self.yfov)
        n, f = self.znear, self.zfar

        P = np.zeros((4, 4))
        P[0, 0] = 1.0 / (aspect_ratio * t)
        P[1, 1] = 1.0 / t
        P[3, 2] = -1.0
        if f is None:
            P[2, 2] = -1.0
            P[2, 3] = -2.0 * n
        else:
            P[2, 2] = (f + n) / (n - f)
            P[2, 3] = (2.0 * f * n) / (n - f)
        return P


class OrthographicCamera(Camera):
    """Camera projecting orthographically.

    Parameters
    ----------
    xmag : float
        The half width of the view, replaced by the one that the aspect ratio of the viewport gives when it is known.
    ymag : float
        The half height of the view.
    znear : float, optional
        The distance to the near clipping plane, above zero. Defaults to DEFAULT_Z_NEAR.
    zfar : float, optional
        The distance to the far clipping plane, beyond 'znear'. Defaults to DEFAULT_Z_FAR.
    """

    def __init__(self, xmag, ymag, znear=DEFAULT_Z_NEAR, zfar=DEFAULT_Z_FAR):
        super().__init__(znear=znear, zfar=zfar)
        assert xmag > 0.0 and ymag > 0.0 and znear > 0.0
        self.xmag = xmag
        self.ymag = ymag

    def get_projection_matrix(self, width=None, height=None):
        xmag = self.xmag if width is None or height is None else width / height * self.ymag
        n, f = self.znear, self.zfar

        P = np.zeros((4, 4))
        P[0, 0] = 1.0 / xmag
        P[1, 1] = 1.0 / self.ymag
        P[2, 2] = 2.0 / (n - f)
        P[2, 3] = (f + n) / (n - f)
        P[3, 3] = 1.0
        return P


__all__ = ["Camera", "PerspectiveCamera", "OrthographicCamera"]
