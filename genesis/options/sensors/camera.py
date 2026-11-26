"""
Camera sensor options for Rasterizer, Raytracer, and Batch Renderer backends.
"""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import (
    Matrix4x4Type,
    PositiveFloat,
    PositiveInt,
    PositiveVec2IType,
    UnitVec4FType,
    ValidFloat,
    Vec3FType,
)

from .options import KinematicSensorOptionsMixin

if TYPE_CHECKING:
    from genesis.engine.sensors.base_sensor import Sensor
    from genesis.engine.sensors.camera import BatchRendererCameraSensor, RasterizerCameraSensor, RaytracerCameraSensor


class BaseCameraOptions[SensorT: "Sensor"](KinematicSensorOptionsMixin[SensorT]):
    """
    Base class for camera sensor options containing common properties.

    Parameters
    ----------
    res : tuple[int, int]
        Resolution as (width, height). Default is (512, 512).
    pos : array-like[float, float, float]
        Camera position offset. If attached to a link, this is relative to the link frame.
        If not attached, this is relative to the world origin. Default is (3.5, 0.0, 1.5).
    lookat : array-like[float, float, float]
        Point the camera looks at in world frame.
    up : array-like[float, float, float]
        Up vector for camera orientation. Default is (0, 0, 1).
    fov : float
        Vertical field of view in degrees. Default is 60.0.
    lights : list[dict], optional
        List of lights to add for this camera backend. Each light is a dict with
        backend-specific parameters. Default is empty list.
    offset_T : array-like, shape (4, 4), optional
        4x4 transformation matrix specifying the camera's pose relative to the attached link.
        If provided, this takes priority over pos_offset and euler_offset. Default is None.
    link : KinematicLink | None, optional
        The link to which this sensor is attached, or None for a static sensor. It sets 'entity_idx' and
        'link_idx_local', and takes precedence over them when both are given. Defaults to None.
    entity_idx : Entity | int | None, optional
        Deprecated, use 'link'. The entity to which this sensor is attached, or its index in the scene, or None for a
        static sensor. Ignored if 'link' is also given. Defaults to None.
    link_idx_local : int, optional
        Deprecated, use 'link'. The index in 'entity_idx' of the link to which this sensor is attached. Ignored if
        'link' is also given. Defaults to 0, the base link.
    """

    res: PositiveVec2IType = (512, 512)
    pos: Vec3FType = (3.5, 0.0, 1.5)
    lookat: Vec3FType = (0.0, 0.0, 0.0)
    up: Vec3FType = (0.0, 0.0, 1.0)
    fov: ValidFloat = Field(default=60.0, gt=0, lt=180)
    lights: list[dict[str, Any]] = []
    offset_T: Matrix4x4Type | None = None

    def model_post_init(self, context: Any) -> None:
        if self.history_length > 0:
            gs.raise_exception(
                "Camera sensors do not support `history_length`. The camera read path renders lazily on read() "
                "and bypasses the shared sensor cache that backs the history buffer."
            )


class RasterizerCameraOptions(BaseCameraOptions["RasterizerCameraSensor"]):
    """
    Options for Rasterizer camera sensor (OpenGL-based rendering).

    Parameters
    ----------
    near : float
        Near clipping plane distance. Default is 0.01.
    far : float
        Far clipping plane distance. Default is 100.0.
    """

    near: PositiveFloat = 0.01
    far: PositiveFloat = 100.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.far <= self.near:
            gs.raise_exception(f"far must be greater than near, got near={self.near}, far={self.far}")


class RaytracerCameraOptions(BaseCameraOptions["RaytracerCameraSensor"]):
    """
    Options for Raytracer camera sensor (LuisaRender path tracing).

    Parameters
    ----------
    model : str
        Camera model: "pinhole" or "thinlens". Default is "pinhole".
    spp : int
        Samples per pixel for path tracing. Default is 256.
    denoise : bool
        Whether to apply denoising. Default is False.
    aperture : float
        The f-number of the lens of a thin-lens camera: its focal length over the diameter of its opening. A smaller
        f-number widens the opening and blurs more what lies away from the focus distance. Default is 2.8.
    focal_len : float
        The focal length of the lens of a thin-lens camera in meters. At the same f-number, a longer focal length
        widens the opening and blurs more what lies away from the focus distance, while `fov` alone sets what the
        camera sees. Default is 0.05.
    focus_dist : float
        Focus distance in meters for thinlens camera. Default is 3.0.
    env_surface : gs.surfaces.Surface | None
        Environment surface for skybox. Default is None.
    env_radius : float
        Environment sphere radius. Default is 15.0.
    env_pos : tuple[float, float, float]
        Environment sphere position. Default is (0, 0, 0).
    env_quat : tuple[float, float, float, float]
        Environment sphere quaternion (w, x, y, z). Default is (1, 0, 0, 0).
    """

    model: Literal["pinhole", "thinlens"] = "pinhole"
    spp: PositiveInt = 256
    denoise: StrictBool = False
    aperture: PositiveFloat = 2.8
    focal_len: PositiveFloat = 0.05
    focus_dist: PositiveFloat = 3.0
    env_surface: Any = None  # gs.surfaces.Surface
    env_radius: PositiveFloat = 15.0
    env_pos: Vec3FType = (0.0, 0.0, 0.0)
    env_quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)


class BatchRendererCameraOptions(BaseCameraOptions["BatchRendererCameraSensor"]):
    """
    Options for Batch Renderer camera sensor (Madrona GPU batch rendering).

    Note: All batch renderer cameras must have the same resolution.

    Parameters
    ----------
    use_rasterizer : bool
        Whether to use rasterizer mode. Default is True.
    """

    model: Literal["pinhole", "thinlens", "fisheye"] = "pinhole"
    near: PositiveFloat = 0.01
    far: PositiveFloat = 100.0
    use_rasterizer: StrictBool = True

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.far <= self.near:
            gs.raise_exception(f"far must be greater than near, got near={self.near}, far={self.far}")
