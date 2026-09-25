import sys
from typing import TYPE_CHECKING, Annotated, Any, Literal, Mapping, Sequence, Union

from pydantic import Field, StrictBool, StrictInt, model_validator

import genesis as gs
from genesis.datatypes import List
from genesis.typing import IArrayType, PositiveFloat, PositiveInt, PositiveVec2IType, UnitIntervalVec3Type, Vec3FType
from genesis.utils.serialization import ignore_invalid_load

from .options import Options


if TYPE_CHECKING:
    LightType = Union[Mapping[str, Any], "DirectionalLight", "PointLight", "AmbientLight"]
    LightArray = Sequence[LightType]
else:
    LightType = Annotated["DirectionalLight | PointLight | AmbientLight", Field(discriminator="type")]
    LightArray = Annotated[List[LightType], Field(strict=False)]


class ViewerOptions(Options):
    """
    Options configuring preperties of the interactive viewer.

    Note
    ----
    The viewer's camera uses the `Rasterizer` backend regardless of `gs.renderers.*` when creating the scene.

    Parameters
    ----------
    res : tuple, shape (2,), optional
        The resolution of the viewer. If not set, will auto-compute using resolution of the connected display.
    run_in_thread : bool
        Whether to run the viewer in a background thread, where it stays responsive while the simulation is idle. If
        None, it runs in a background thread wherever the platform supports it. MacOS only supports the main thread,
        so asking for a background thread there raises an error. A scene file recording such a request is loaded
        as asking for the platform default, so it opens on every platform.
    refresh_rate : int
        The rate (in frames per second) at which the viewer repaints on screen, and the framerate the video recorded
        from the viewer window is encoded at. Independent of the physics timestep, and of the framerate passed to
        `camera.start_recording`.
    realtime_factor : float | None
        Multiple of wall-clock real time that one second of playback stands for (1.0 is real time, 2.0 is twice as
        fast). When the viewer is shown, the simulation is paced to it, falling behind gracefully when it cannot keep
        up. It also sets the speed of the videos recorded by `camera.start_recording`, viewer or not. Set to None to
        run as fast as possible, which recordings treat as real time since there is no pace to follow. Defaults
        to 1.0.
    camera_pos : tuple of float, shape (3,)
        The position of the viewer's camera.
    camera_lookat : tuple of float, shape (3,)
        The lookat position that the camera.
    camera_up : tuple of float, shape (3,)
        The up vector of the camera's extrinsic pose.
    camera_fov : float
        The field of view (in degrees) of the camera.
    enable_help_text : bool
        Whether to enable the rendering of instructions text in the viewer.
    enable_default_keybinds : bool
        Whether to enable the default keyboard controls in the viewer.
    enable_gui : bool
        Whether to automatically attach the ImGui overlay panel when the viewer is constructed. Defaults
        to False. The overlay renders its own controls and captures keyboard input, so it requires
        enable_help_text and enable_default_keybinds to be False; they default to False when enable_gui is
        True, and explicitly setting either to True alongside enable_gui raises an error. Scene editing
        controls (Rebuild Scene, Add Entity, per-entity remove) are visible but disabled unless the scene
        is managed by a gs.InteractiveScene that supports them.
    """

    res: PositiveVec2IType | None = None
    # MacOS serves the viewer window from its main thread only (see Visualizer).
    run_in_thread: Annotated[
        StrictBool | None, ignore_invalid_load((None, False) if sys.platform == "darwin" else (None, False, True))
    ] = None
    refresh_rate: PositiveInt = 60
    realtime_factor: PositiveFloat | None = 1.0
    camera_pos: Vec3FType = (3.5, 0.5, 2.5)
    camera_lookat: Vec3FType = (0.0, 0.0, 0.5)
    camera_up: Vec3FType = (0.0, 0.0, 1.0)
    camera_fov: float = 40
    enable_help_text: StrictBool = True
    enable_default_keybinds: StrictBool = True
    enable_gui: StrictBool = False
    # Deprecated alias for refresh_rate, resolved in model_post_init.
    max_FPS: PositiveInt | None = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # 'max_FPS' is deprecated in favor of 'refresh_rate'; map it over when set, but refuse to guess which the user
        # meant when they also set 'refresh_rate' explicitly.
        if self.max_FPS is not None:
            if "refresh_rate" in self.model_fields_set:
                gs.raise_exception("'max_FPS' is deprecated and replaced by 'refresh_rate'; set only one of them.")
            gs.logger.warning("'max_FPS' is deprecated and will be removed; it now maps to 'refresh_rate'.")
            self.refresh_rate = self.max_FPS
        if not self.enable_gui:
            return
        # The GUI overlay renders its own controls and captures keyboard input, so the help-text overlay and default
        # keybind plugin must be off. Default them off when the user left them implicit, but raise if the user
        # explicitly requested a conflicting value.
        if self.enable_help_text:
            if "enable_help_text" in self.model_fields_set:
                gs.raise_exception("'enable_help_text' must be False when 'enable_gui' is True.")
            self.enable_help_text = False
        if self.enable_default_keybinds:
            if "enable_default_keybinds" in self.model_fields_set:
                gs.raise_exception("'enable_default_keybinds' must be False when 'enable_gui' is True.")
            self.enable_default_keybinds = False


class DirectionalLight(Options):
    type: Literal["directional"] = "directional"
    dir: Vec3FType
    color: UnitIntervalVec3Type
    intensity: float


class PointLight(Options):
    type: Literal["point"] = "point"
    pos: Vec3FType
    color: UnitIntervalVec3Type
    intensity: float


class AmbientLight(Options):
    type: Literal["ambient"] = "ambient"
    color: UnitIntervalVec3Type
    intensity: float


class VisOptions(Options):
    """
    This configures visualization-related properties that are independent of the viewer or camera.

    Parameters
    ----------
    show_world_frame : bool
        Whether to visualize the world frame. Default to False.
    world_frame_size : float
        The length (in meters) of the world frame's axes.
    show_link_frame : bool
        Whether to visualize the frames of each RigidLink. Default to False.
    link_frame_size : float
        The length (in meters) of the link frames' axes.
    show_cameras : bool
        Whether to render the cameras added to the scene, together with their frustums. Default to False.
    shadow : bool
        Whether to render shadow. Defaults to True.
    plane_reflection : bool
        Whether to render plane reflection. Defaults to False.
    split_envs : bool
        Whether the cameras render each environment in its own world, split from the others, or all the environments
        as if they shared one scene, laid out side by side as the interactive viewer shows them. Split, a camera returns
        one image per rendered environment, stacked along a leading dimension, each drawn from the same viewpoint with
        lighting and shadows identical in every environment, at the cost of one render per environment. Merged, a
        camera returns one image of the whole grid from a viewpoint bound to one environment, which is cheaper and
        shows the environments next to each other. This is only an option for Rasterizer, and applies to the scene
        cameras, debug cameras included. Camera sensors and BatchRender always render the environments split. Defaults
        to False.
    background_color : tuple of float, shape (3,)
        The color of the scene background.
    ambient_light : tuple of float, shape (3,)
        The color of the scene's ambient light.
    visualize_mpm_boundary : bool
        Whether to visualize the boundary of the MPM Solver.
    visualize_sph_boundary : bool
        Whether to visualize the boundary of the SPH Solver.
    visualize_pbd_boundary : bool
        Whether to visualize the boundary of the PBD Solver.
    segmentation_level : str
        The segmentation level used for segmentation mask rendering. Should be one of ['entity', 'link', 'geom'].
        Defaults to 'link'.
    render_particle_as : str
        How particles in the scene should be rendered. Should be one of ['sphere', 'tet']. Defaults to 'sphere'.
    particle_size_scale : float
        Scale applied to actual particle size for rendering. Defaults to 1.0.
    contact_force_scale : float = 0.02
        Scale in m.N^{-1} for contact arrow visualization, e.g. the force arrow representing 10N will be 0.2m long if
        scale is 0.02. Defaults to 0.01.
    n_support_neighbors : int
        Number of supporting neighbor particles used to compute vertex position of the visual mesh. Used for rendering
        deformable bodies. Defaults to 12.
    rendered_envs_idx : list, optional
        Indices of the environments that will be rendered. If not provided, all the environments will be considered.
        Defaults to None.
    n_rendered_envs : int, optional
        This option is deprecated. Please use `rendered_envs_idx` instead.
    lights : list of dict.
        Lights added to the scene.
    """

    show_world_frame: StrictBool = False
    world_frame_size: float = 1.0
    show_link_frame: StrictBool = False
    link_frame_size: float = 0.2
    show_cameras: StrictBool = False
    shadow: StrictBool = True
    plane_reflection: StrictBool = False
    split_envs: StrictBool = False
    background_color: UnitIntervalVec3Type = (0.04, 0.08, 0.12)
    ambient_light: UnitIntervalVec3Type = (0.1, 0.1, 0.1)
    visualize_mpm_boundary: StrictBool = False
    visualize_sph_boundary: StrictBool = False
    visualize_pbd_boundary: StrictBool = False
    segmentation_level: Literal["entity", "link", "geom"] = "link"
    render_particle_as: Literal["sphere", "tet"] = "sphere"
    particle_size_scale: PositiveFloat = 1.0
    contact_force_scale: PositiveFloat = 0.01
    n_support_neighbors: StrictInt = 12
    rendered_envs_idx: IArrayType | None = None
    lights: LightArray = List((DirectionalLight(dir=(-1, -1, -1), color=(1.0, 1.0, 1.0), intensity=5.0),))

    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_n_rendered_envs(cls, data: dict) -> dict:
        n_rendered_envs = data.pop("n_rendered_envs", None)
        if n_rendered_envs is not None:
            gs.logger.warning(
                "Viewer option 'n_rendered_envs' is deprecated and will be removed in a future release. "
                "Please use 'rendered_envs_idx' instead."
            )
            if data.get("rendered_envs_idx") is not None:
                raise ValueError("Cannot specify both 'n_rendered_envs' and 'rendered_envs_idx'.")
            data["rendered_envs_idx"] = tuple(range(n_rendered_envs))
        env_separate_rigid = data.pop("env_separate_rigid", None)
        if env_separate_rigid is not None:
            gs.logger.warning(
                "Viewer option 'env_separate_rigid' is deprecated and will be removed in a future release. "
                "Please use 'split_envs' instead."
            )
            if data.get("split_envs") is not None:
                raise ValueError("Cannot specify both 'env_separate_rigid' and 'split_envs'.")
            data["split_envs"] = env_separate_rigid
        return data
