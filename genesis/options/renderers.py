from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np

from pydantic import Field, StrictBool, StrictInt, model_validator

import genesis as gs
from genesis.datatypes import List
from genesis.typing import PositiveFloat, UnitVec4FType, Vec3FType

from .options import Options
from .surfaces import Surface


if TYPE_CHECKING:
    LightArray = Sequence[Mapping[str, Any] | "SphereLight"]
else:
    LightArray = Annotated[List["SphereLight"], Field(strict=False)]


class SphereLight(Options):
    """
    Sphere light for the ray tracer.

    Parameters
    ----------
    pos : tuple of float
        Position of the light. Defaults to (0.0, 0.0, 10.0).
    color : tuple of float
        Color of the light. Values are not restricted to [0, 1] to allow HDR lighting.
    intensity : float
        Intensity multiplier for the light color. Defaults to 1.0.
    radius : float
        Radius of the light sphere. Defaults to 4.0.
    """

    pos: Vec3FType = (0.0, 0.0, 10.0)
    color: Vec3FType = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    radius: float = 4.0


class RendererOptions(Options):
    """
    This is the base class for all `gs.renderers.*` classes.
    Note that this is not an actual renderer, but rather a renderer configuration specifying which renderer to use and its parameters.
    """

    pass


class Rasterizer(RendererOptions):
    """
    Rasterizer renderer. This has no parameter to be configured.

    Note
    ----
    You can set which renderer to use for cameras, but the interactive viewer always uses the rasterizer rendering backend. If you want to configure properties like shadow, lights, etc., you should use `gs.options.VisOptions` instead.
    """

    pass


class RayTracer(RendererOptions):
    """
    RayTracer renderer.

    Note
    ----
    We use a environmental sphere wrapped around the scene to render the environment map (i.e. skybox).

    Parameters
    ----------
    device_index : int, optional
        Device ID used for the raytracer. None for Genesis' device. Defaults to None.
    logging_level : str, optional
        Logging level. Should be one of "debug", "info", "warning". Defaults to "warning".
    state_limit : int, optional
        State limit for raytracer integrator. Defaults to 2 ** 25.
    tracing_depth : int, optional
        Tracing depth. Defaults to 32.
    rr_depth : int, optional
        Russian Roulette depth. Defaults to 0.
    rr_threshold : float, optional
        Russian Roulette threshold. Defaults to 0.95.
    env_surface : Surface | None, optional
        Environment surface. Defaults to None.
    env_radius : float, optional
        Environment radius. Defaults to 1000.0.
    env_pos : tuple of float, optional
        Environment position. Defaults to (0.0, 0.0, 0.0).
    env_euler : tuple of float, optional
        Environment Euler angles in degrees. Shortcut for `env_quat`. Defaults to (0.0, 0.0, 0.0).
    env_quat : tuple of float | None, optional
        Environment quaternion. Defaults to None.
    lights : list of SphereLight, optional
        List of sphere lights.
    normal_diff_clamp : float, optional
        Lower bound for direct face normal vs vertex normal for face normal interpolation. Range is [0, 180]. Defaults
        to 180.
    """

    device_index: StrictInt | None = None
    logging_level: Literal["debug", "info", "warning"] = "warning"
    state_limit: StrictInt = 2**25
    tracing_depth: StrictInt = 32
    rr_depth: StrictInt = 0
    rr_threshold: PositiveFloat = 0.95

    # environment texture
    env_surface: Surface | None = None
    env_radius: float = 1000.0
    env_pos: Vec3FType = (0.0, 0.0, 0.0)
    env_euler: Vec3FType | None = Field(default=None, exclude=True, repr=False)
    env_quat: UnitVec4FType | None = None

    # sphere lights
    lights: LightArray = List((SphereLight(pos=(0.0, 0.0, 10.0), color=(1.0, 1.0, 1.0), intensity=10.0, radius=4.0),))

    # lower bound for direct face normal vs vertex normal for face normal interpolation
    normal_diff_clamp: float = Field(default=180.0, ge=0.0, le=180.0)

    @model_validator(mode="before")
    @classmethod
    def _resolve_env_euler(cls, data: dict) -> dict:
        env_euler = data.get("env_euler")
        env_quat = data.get("env_quat")
        if env_euler is not None and env_quat is not None:
            gs.raise_exception("'env_euler' and 'env_quat' cannot both be set.")
        if env_quat is None:
            if env_euler is None:
                env_euler = (0.0, 0.0, 0.0)
            data["env_quat"] = tuple(gs.utils.geom.xyz_to_quat(np.array(env_euler), rpy=True, degrees=True))
        return data

    def model_post_init(self, context: Any) -> None:
        if self.env_surface is not None:
            self.env_surface.update_texture()


class BatchRenderer(RendererOptions):
    """
    BatchRenderer renderer.

    Note
    ----
    This renderer is used to render the scene in a batch.

    Parameters
    ----------
    use_rasterizer : bool, optional
        Whether to use the rasterizer renderer. Defaults to False.
    """

    use_rasterizer: StrictBool = False
