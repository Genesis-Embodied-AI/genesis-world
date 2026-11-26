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


class FisheyeCamera(Camera):
    """Camera projecting as a fisheye lens, following the Kannala-Brandt model of the fisheye module of OpenCV.

    A ray making an angle theta with the line of sight reaches the image at the distance
    theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) from the principal point, in units of the focal
    lengths. The model holds beyond a right angle, up to a ray pointing backwards. The rasterizer renders the faces of a
    cube map around the camera and resamples them. The depth of a pixel is its distance to the centre of projection,
    which the clipping distances bound.

    Parameters
    ----------
    intrinsics : (4,) float
        The focal lengths and the principal point (fx, fy, cx, cy), in pixels from the top-left corner of the image.
    distortion : (4,) float
        The distortion coefficients (k1, k2, k3, k4), all zero for an equidistant projection.
    znear : float, optional
        The distance to the near clipping plane of each face of the cube map. Defaults to DEFAULT_Z_NEAR.
    zfar : float, optional
        The largest distance to the centre of projection drawn, beyond 'znear'. Defaults to DEFAULT_Z_FAR.
    """

    def __init__(self, intrinsics, distortion, znear=DEFAULT_Z_NEAR, zfar=DEFAULT_Z_FAR):
        super().__init__(znear=znear, zfar=zfar)
        assert zfar is not None
        self.intrinsics = intrinsics
        self.distortion = distortion

    @property
    def face_zfar(self):
        """float : The far plane of the faces of the cube map, at the far clipping distance in their corners."""
        return np.sqrt(3.0) * self.zfar

    def get_projection_matrix(self, width=None, height=None):
        # The projection of each face of the cube map, a right angle wide, the resampling clipping the depth at the far
        # clipping distance (see fisheye.frag)
        face_camera = PerspectiveCamera(yfov=0.5 * np.pi, znear=self.znear, zfar=self.face_zfar, aspect_ratio=1.0)
        return face_camera.get_projection_matrix()


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


__all__ = ["Camera", "PerspectiveCamera", "FisheyeCamera", "OrthographicCamera"]
