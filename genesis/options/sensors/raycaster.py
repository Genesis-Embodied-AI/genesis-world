import math
from dataclasses import dataclass
from typing import Sequence

import torch

import genesis as gs
from genesis.utils.geom import spherical_to_cartesian


@dataclass
class RaycastPattern:
    """
    Base class for raycast patterns.
    """

    def __init__(self):
        self._return_shape: tuple[int, ...] = self._get_return_shape()
        self._ray_dirs: torch.Tensor = torch.empty((*self._return_shape, 3), dtype=gs.tc_float, device=gs.device)
        self._ray_starts: torch.Tensor = torch.empty((*self._return_shape, 3), dtype=gs.tc_float, device=gs.device)
        self.compute_ray_dirs()
        self.compute_ray_starts()

    def _get_return_shape(self) -> tuple[int, ...]:
        """Get the shape of the ray vectors, e.g. (n_scan_lines, n_points_per_line) or (n_rays,)"""
        raise NotImplementedError(f"{type(self).__name__} must implement `get_return_shape()`.")

    def compute_ray_dirs(self):
        """
        Update ray_dirs, the local direction vectors of the rays.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement `compute_ray_dirs()`.")

    def compute_ray_starts(self):
        """
        Update ray_starts, the local start positions of the rays.

        As a default, all rays will start at the local origin.
        """
        self._ray_starts.fill_(0.0)

    @property
    def return_shape(self) -> tuple[int, ...]:
        return self._return_shape

    @property
    def ray_dirs(self) -> torch.Tensor:
        return self._ray_dirs

    @property
    def ray_starts(self) -> torch.Tensor:
        return self._ray_starts


# ============================== Generic Patterns ==============================


class GridPattern(RaycastPattern):
    """
    Configuration for grid-based ray casting.

    Defines a 2D grid of rays in the sensor coordinate system.

    Parameters
    ----------
    resolution : float
        Grid spacing in meters.
    size : tuple[float, float]
        Grid dimensions (length, width) in meters.
    direction : tuple[float, float, float]
        Ray direction vector.
    """

    def __init__(
        self,
        resolution: float = 0.1,
        size: tuple[float, float] = (2.0, 2.0),
        direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
    ):
        if resolution < 1e-3:
            gs.raise_exception(f"Resolution should be at least 1e-3 (1mm). Got `{resolution}`.")
        self.coords = [
            torch.arange(-size / 2, size / 2 + gs.EPS, resolution, dtype=gs.tc_float, device=gs.device) for size in size
        ]
        self.direction = torch.tensor(direction, dtype=gs.tc_float, device=gs.device)

        super().__init__()

    def _get_return_shape(self) -> tuple[int, ...]:
        return (len(self.coords[0]), len(self.coords[1]))

    def compute_ray_dirs(self):
        self._ray_dirs[:] = self.direction.expand((*self._return_shape, 3))

    def compute_ray_starts(self):
        grid_x, grid_y = torch.meshgrid(*self.coords, indexing="ij")
        self._ray_starts[..., 0] = grid_x
        self._ray_starts[..., 1] = grid_y
        self._ray_starts[..., 2] = 0.0


def _generate_uniform_angles(
    n_points: tuple[int, int],
    fov: tuple[float | tuple[float, float] | None, float | tuple[float, float] | None],
    res: tuple[float | None, float | None],
    angles: tuple[Sequence[float] | None, Sequence[float] | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to generate uniform angles given various formats (n and fov, res and fov, or angles).
    """
    return_angles = []

    for n_points_i, fov_i, res_i, angles_i in zip(n_points, fov, res, angles):
        if angles_i is None:
            assert fov_i is not None, "FOV should be provided if angles not given."

            if isinstance(fov_i, Sequence):
                f_min, f_max = fov_i
                fov_size = f_max - f_min
            else:
                f_max = fov_i / 2.0
                f_min = -f_max
                fov_size = fov_i

            assert fov_size <= 360.0 + gs.EPS, "FOV should not be larger than a full rotation."

            # Avoid duplicate angle at 0/360 degrees
            is_full_rotation = fov_size >= 360.0 - gs.EPS
            if res_i is not None:
                n_points_i = math.ceil(fov_size / res_i) + (0 if is_full_rotation else 1)

            assert n_points_i is not None

            if is_full_rotation:
                f_max -= fov_size / n_points_i

            angles_i = torch.linspace(f_min, f_max, n_points_i, dtype=gs.tc_float, device=gs.device)
        else:
            angles_i = torch.tensor(angles_i, dtype=gs.tc_float, device=gs.device)

        return_angles.append(torch.deg2rad(angles_i))

    return tuple(return_angles)


class SphericalPattern(RaycastPattern):
    """
    Configuration for spherical ray pattern.

    Either specify:
    - (`n_points`, `fov`) for uniform spacing by count.
    - (`angular_resolution`, `fov`) for uniform spacing by resolution.
    - `angles` for custom angles.


    Parameters
    ----------
    fov: tuple[float | tuple[float, float], float | tuple[float, float]]
        Field of view in degrees for horizontal and vertical directions. Defaults to (360.0, 30.0).
        If a single float is provided, the FOV is centered around 0 degrees.
        If a tuple is provided, it specifies the (min, max) angles.
    n_points: tuple[int, int]
        Number of horizontal/azimuth and vertical/elevation scan lines. Defaults to (64, 128).
    angular_resolution: tuple[float, float], optional
        Horizontal and vertical angular resolution in degrees. Overrides n_points if provided.
    angles: tuple[Sequence[float], Sequence[float]], optional
        Array of horizontal/vertical angles. Overrides the other options if provided.
    """

    def __init__(
        self,
        fov: tuple[float | tuple[float, float], float | tuple[float, float]] = (360.0, 60.0),
        n_points: tuple[int, int] = (128, 64),
        angular_resolution: tuple[float | None, float | None] = (None, None),
        angles: tuple[Sequence[float] | None, Sequence[float] | None] = (None, None),
    ):
        for fov_i in fov:
            if (isinstance(fov_i, float) and (fov_i < 0 or fov_i > 360.0 + gs.EPS)) or (
                isinstance(fov_i, tuple) and (fov_i[1] - fov_i[0] > 360.0 + gs.EPS)
            ):
                gs.raise_exception(f"[{type(self).__name__}] FOV should be between 0 and 360. Got: {fov}.")

        self.angles = _generate_uniform_angles(n_points, fov, angular_resolution, angles)

        super().__init__()

    def _get_return_shape(self) -> tuple[int, ...]:
        return tuple(len(a) for a in self.angles)

    def compute_ray_dirs(self):
        meshgrid = torch.meshgrid(*self.angles, indexing="ij")
        self._ray_dirs[:] = spherical_to_cartesian(*meshgrid)


# ============================== Camera Patterns ==============================


class DepthCameraPattern(RaycastPattern):
    """
    Configuration for pinhole depth camera ray casting.

    You can configure the camera intrinsics in several ways:
    1. Provide fx and fy directly (and optionally cx, cy)
    2. Provide fov_horizontal only (fy computed to maintain aspect ratio)
    3. Provide fov_vertical only (fx computed to maintain aspect ratio)
    4. Provide both fov_horizontal and fov_vertical
    5. Provide fx or fy only (the other computed to maintain aspect ratio)

    A focal length given directly takes priority over its field of view. With no focal length and no field of view,
    fov_horizontal defaults to 90 degrees. If cx or cy are not provided, they default to the image center.

    Parameters
    ----------
    res: tuple[int, int]
        The resolution of the camera, specified as a tuple (width, height).
    fx : float | None
        Focal length in x direction in pixels. Computed from fov_horizontal if None.
    fy : float | None
        Focal length in y direction in pixels. Computed from fov_vertical if None.
    cx : float | None
        Principal point x coordinate in pixels. Defaults to image center if None.
    cy : float | None
        Principal point y coordinate in pixels. Defaults to image center if None.
    fov_horizontal : float | None
        Horizontal field of view in degrees. Used to compute fx if fx is None.
    fov_vertical : float | None
        Vertical field of view in degrees. Used to compute fy if fy is None.
    """

    def __init__(
        self,
        res: tuple[int, int] = (128, 96),
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        fov_horizontal: float | None = None,
        fov_vertical: float | None = None,
    ):
        self.width, self.height = res

        if self.width <= 0 or self.height <= 0:
            gs.raise_exception(f"[{type(self).__name__}] Image dimensions must be positive. Got: {res}")

        if fx is None and fy is None and fov_horizontal is None and fov_vertical is None:
            fov_horizontal = 90.0
        if fx is None and fov_horizontal is not None:
            fx = self.width / (2.0 * math.tan(math.radians(fov_horizontal) / 2.0))
        if fy is None and fov_vertical is not None:
            fy = self.height / (2.0 * math.tan(math.radians(fov_vertical) / 2.0))
        # Square pixels maintain the aspect ratio of the field of view
        if fx is None:
            fx = fy
        if fy is None:
            fy = fx
        if cx is None:
            cx = self.width * 0.5
        if cy is None:
            cy = self.height * 0.5

        self.fx: float = fx
        self.fy: float = fy
        self.cx: float = cx
        self.cy: float = cy

        super().__init__()

    def _get_return_shape(self) -> tuple[int, ...]:
        return (self.height, self.width)

    def compute_ray_dirs(self):
        u = torch.arange(0, self.width, dtype=gs.tc_float, device=gs.device) + 0.5
        v = torch.arange(0, self.height, dtype=gs.tc_float, device=gs.device) + 0.5
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        # standard camera frame coordinates
        x_c = (uu - self.cx) / self.fx
        y_c = (vv - self.cy) / self.fy
        z_c = torch.ones_like(x_c)

        # transform to robotics camera frame
        dirs = torch.stack([z_c, -x_c, -y_c], dim=-1)
        dirs /= torch.linalg.norm(dirs, dim=-1, keepdim=True)

        self._ray_dirs[:] = dirs
