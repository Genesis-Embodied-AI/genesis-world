"""
Camera sensors for rendering: Rasterizer, Raytracer, and Batch Renderer.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np
import torch

import trimesh

import genesis as gs
from genesis.engine.render_context import ArrayRef, MeshGeometry, Shape, StaticMeshBatch
from genesis.options.renderers import BatchRenderer as BatchRendererOptions
from genesis.options.sensors import (
    BatchRendererCameraOptions,
    RasterizerCameraOptions,
    RaytracerCameraOptions,
    SensorOptions,
)
from genesis.options.vis import ThinLens, VisOptions
from genesis.utils.geom import (
    T_to_quat,
    T_to_trans,
    pos_lookat_up_to_T,
    trans_quat_to_T,
    transform_by_quat,
    transform_by_trans_quat,
    transform_quat_by_quat,
)
from genesis.utils.misc import tensor_to_array
from genesis.vis.batch_renderer import BatchRenderer
from genesis.vis.rasterizer import Rasterizer
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import KinematicSensorMetadataMixin, KinematicSensorMixin, OptionsT, Sensor, SharedSensorMetadata

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.batch_renderer import BatchRenderer
    from genesis.vis.rasterizer import Rasterizer
    from genesis.vis.rasterizer_context import RasterizerContext
    from genesis.vis.raytracer import Raytracer

    from .sensor_manager import SensorManager


# ========================== Data Class ==========================


class CameraReturnType(NamedTuple):
    """Camera sensor return data."""

    rgb: torch.Tensor


class CameraView(NamedTuple):
    """The volume a camera sees: its rays between two depths."""

    depths: tuple[float, float]
    """The depths bounding the view, the near one zero for a view reaching the center of projection."""
    is_fisheye: bool
    """Whether the camera is an equidistant fisheye, a pinhole otherwise: a ray makes an angle with the line of sight
    proportional to its distance to the center of the image, the shorter half side spanning half the field of view."""
    is_disk: bool
    """Whether the image holds its inscribed disk alone, the image as a whole otherwise."""
    is_radial: bool
    """Whether the depths are distances along each ray, planes orthogonal to the line of sight otherwise."""


class MinimalVisualizerWrapper:
    """
    Minimal visualizer wrapper for BatchRenderer camera sensors.

    BatchRenderer requires a visualizer-like object to provide camera information and context, but camera sensors don't
    need the full visualizer functionality (viewer, UI, etc.). This wrapper provides just the minimal interface expected
    by BatchRenderer while avoiding the overhead of creating a full visualizer instance.
    """

    def __init__(self, scene, sensors, vis_options):
        self.scene = scene
        self._cameras = []  # Will be populated with camera wrappers
        self._sensors = sensors  # Keep reference to sensors

        # Create a minimal rasterizer context for camera frustum visualization (required by BatchRenderer even though
        # cameras don't render frustums)
        self._context = RasterizerContext(vis_options)
        self._context.build(scene)
        self._context.reset()


class BaseCameraWrapper:
    """Base class for camera wrappers to reduce code duplication."""

    def __init__(self, sensor):
        self.sensor = sensor
        self.uid = sensor._idx
        self.res = sensor._options.res
        self.fov = sensor._options.fov
        self.near = sensor._options.near
        self.far = sensor._options.far


class RasterizerCameraWrapper(BaseCameraWrapper):
    """Lightweight wrapper object used by the rasterizer backend."""

    def __init__(self, sensor: "RasterizerCameraSensor"):
        super().__init__(sensor)
        self.aspect_ratio = self.res[0] / self.res[1]


class BatchRendererCameraWrapper(BaseCameraWrapper):
    """Wrapper object used by the batch renderer backend."""

    def __init__(self, sensor: "BatchRendererCameraSensor"):
        super().__init__(sensor)
        self.idx = len(sensor._shared_metadata.sensors)  # Camera index in batch
        self.model = sensor._options.model

        # Initial pose
        pos = torch.tensor(sensor._options.pos, dtype=gs.tc_float, device=gs.device)
        lookat = torch.tensor(sensor._options.lookat, dtype=gs.tc_float, device=gs.device)
        up = torch.tensor(sensor._options.up, dtype=gs.tc_float, device=gs.device)

        # Store pos/lookat/up for later updates
        self._pos = pos
        self._lookat = lookat
        self._up = up
        self.transform = pos_lookat_up_to_T(pos, lookat, up)

    def get_pos(self):
        """Get camera position (for batch renderer)."""
        n_envs = self.sensor._manager.n_envs
        if self._pos.ndim > 1 or n_envs == 0:
            return self._pos
        return self._pos[None].expand((n_envs, -1))

    def get_quat(self):
        """Get camera quaternion (for batch renderer)."""
        quat = T_to_quat(self.transform)
        n_envs = self.sensor._manager.n_envs
        if quat.ndim > 1 or n_envs == 0:
            return quat
        return quat[None].expand((n_envs, -1))


# ========================== Shared Metadata ==========================


@dataclass
class RasterizerCameraSharedMetadata(KinematicSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Rasterizer cameras."""

    # Rasterizer instance
    renderer: "Rasterizer | None" = None
    # RasterizerContext instance
    context: "RasterizerContext | None" = None
    # List of light dictionaries
    lights: list[dict[str, Any]] | None = None
    # List of RasterizerCameraSensor instances
    sensors: list["RasterizerCameraSensor"] | None = None
    # {sensor_idx: np.ndarray with shape (B, H, W, 3)}
    image_cache: dict[int, np.ndarray] | None = None
    # Track when rasterizer cameras were last updated
    last_render_timestep: int = -1

    def destroy(self):
        super().destroy()

        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
        if self.context is not None:
            self.context.destroy()
            self.context = None
        self.lights = None
        self.image_cache = None
        self.sensors = None


@dataclass
class RaytracerCameraSharedMetadata(KinematicSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Raytracer cameras."""

    # Raytracer instance
    renderer: "Raytracer | None" = None
    # List of light objects
    lights: list[Any] | None = None
    # List of RaytracerCameraSensor instances
    sensors: list["RaytracerCameraSensor"] | None = None
    # {sensor_idx: np.ndarray with shape (B, H, W, 3)}
    image_cache: dict[int, np.ndarray] | None = None
    # Track when raytracer cameras were last updated
    last_render_timestep: int = -1

    def destroy(self):
        super().destroy()

        self.renderer = None
        self.sensors = None
        self.image_cache = None


@dataclass
class BatchRendererCameraSharedMetadata(KinematicSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Batch Renderer cameras."""

    # BatchRenderer instance
    renderer: "BatchRenderer | None" = None
    # gs.List of lights
    lights: Any | None = None
    # List of BatchRendererCameraSensor instances
    sensors: list["BatchRendererCameraSensor"] | None = None
    # {sensor_idx: np.ndarray with shape (B, H, W, 3)}
    image_cache: dict[int, np.ndarray] | None = None
    # Track when batch was last rendered
    last_render_timestep: int = -1
    # MinimalVisualizerWrapper instance
    visualizer_wrapper: "MinimalVisualizerWrapper | None" = None

    def destroy(self):
        super().destroy()

        self.renderer = None
        self.sensors = None
        self.image_cache = None
        self.visualizer_wrapper = None


# ========================== Base Camera Sensor ==========================


class BaseCameraSensor(KinematicSensorMixin, Sensor[OptionsT, None, SharedSensorMetadata, CameraReturnType]):
    """
    Base class for camera sensors that render RGB images into an internal image_cache.

    This class centralizes:
    - Attachment handling via KinematicSensorMixin
    - The _stale flag used for auto-render-on-read
    - Common Sensor cache integration (shape/dtype)
    - Shared read() method returning torch tensors
    """

    uses_ring_pipeline: ClassVar[bool] = False

    def __init__(
        self,
        options: "SensorOptions",
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        # `uses_ring_pipeline = False` triggers the generic delay / jitter / history rejection in `Sensor.__init__`.
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._stale: bool = True

    # ========================== Cache Integration (shared) ==========================

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        w, h = self._options.res
        return ((h, w, 3),)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return torch.uint8

    @classmethod
    def _update_shared_cache(
        cls,
        shared_context: None,
        shared_metadata: SharedSensorMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer | None",
        intermediate_cache: torch.Tensor,
    ):
        # No per-step cache update for cameras (handled lazily on read()). `BaseCameraSensor` declares
        # `uses_ring_pipeline = False`, so the manager passes both timeline rings as ``None`` here.
        pass

    @classmethod
    def reset(cls, shared_metadata: SharedSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        # Reset can restore a different state at the last rendered timestep, so force the next read to rerender.
        # FIXME: the frame cache keys on a single timestep for the whole batch, so a partial-env reset invalidates
        # every environment. Extend the caching mechanism to be env-aware so envs_idx only refreshes the reset envs.
        shared_metadata.last_render_timestep = -1

    def _update_markers(self, markers: tuple[Shape, ...], envs_idx: torch.Tensor | None) -> tuple[StaticMeshBatch]:
        if not markers:
            # A solid sphere at the center of projection
            center = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
            center.visual.vertex_colors = (190, 190, 190, 255)
            meshes = [center]
            # The view, translucent, apart so that it blends over the scene: the rays of the camera between two depths,
            # through a grid of rings of the image, its outer ring sampling the boundary of the image.
            view = self._get_view()
            if view is not None:
                N_SIDE, N_RINGS = 16, 8
                aspect_ratio = self._options.res[0] / self._options.res[1]
                half_fov = np.deg2rad(self._options.fov / 2.0)
                # The image boundary, in a frame of the image whose vertical half side spans one
                rho_scale = min(aspect_ratio, 1.0)
                if view.is_disk:
                    angles = np.linspace(0.0, 2.0 * np.pi, 4 * N_SIDE, endpoint=False)
                    image_boundary = rho_scale * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
                else:
                    sides_pos = np.linspace(-1.0, 1.0, N_SIDE, endpoint=False)
                    sides_end = np.ones_like(sides_pos)
                    image_boundary = np.concatenate(
                        (
                            np.stack((aspect_ratio * sides_pos, -sides_end), axis=-1),
                            np.stack((aspect_ratio * sides_end, sides_pos), axis=-1),
                            np.stack((-aspect_ratio * sides_pos, sides_end), axis=-1),
                            np.stack((-aspect_ratio * sides_end, -sides_pos), axis=-1),
                        )
                    )
                n_boundary = len(image_boundary)
                rings_pos = np.arange(1, N_RINGS + 1)[:, None, None] / N_RINGS * image_boundary
                image_pos = np.concatenate((np.zeros((1, 2)), rings_pos.reshape((-1, 2))))
                if view.is_fisheye:
                    image_rho = np.linalg.norm(image_pos, axis=-1, keepdims=True)
                    rays_angle = image_rho / rho_scale * half_fov
                    rays_radial = np.divide(image_pos, image_rho, out=np.zeros_like(image_pos), where=image_rho > 0.0)
                    rays_dir = np.concatenate((rays_radial * np.sin(rays_angle), -np.cos(rays_angle)), axis=-1)
                else:
                    rays_dir = np.concatenate((np.tan(half_fov) * image_pos, -np.ones((len(image_pos), 1))), axis=-1)
                rays_dir /= np.linalg.norm(rays_dir, axis=-1, keepdims=True) if view.is_radial else -rays_dir[:, 2:]
                # A cap at each depth, a fan around the line of sight then a strip between consecutive rings, and the
                # strip between the outer rings of the caps.
                i_v = np.arange(n_boundary)
                i_v_next = (i_v + 1) % n_boundary
                cap_faces = [np.stack((np.zeros_like(i_v), 1 + i_v, 1 + i_v_next), axis=-1)]
                for ring_start in range(1, 1 + (N_RINGS - 1) * n_boundary, n_boundary):
                    ring_v, ring_v_next = ring_start + i_v, ring_start + i_v_next
                    cap_faces.append(np.stack((ring_v, ring_v + n_boundary, ring_v_next), axis=-1))
                    cap_faces.append(np.stack((ring_v_next, ring_v + n_boundary, ring_v_next + n_boundary), axis=-1))
                cap_faces = np.concatenate(cap_faces)
                n_cap = len(image_pos)
                outer_v, outer_v_next = n_cap - n_boundary + i_v, n_cap - n_boundary + i_v_next
                side_faces = np.concatenate(
                    (
                        np.stack((outer_v, outer_v + n_cap, outer_v_next), axis=-1),
                        np.stack((outer_v_next, outer_v + n_cap, outer_v_next + n_cap), axis=-1),
                    )
                )
                # The winding is made outward for the renderers culling back faces once validation has merged the near
                # cap into the center of projection, for a depth of zero.
                view_mesh = trimesh.Trimesh(
                    np.concatenate([depth * rays_dir for depth in view.depths]),
                    np.concatenate((cap_faces[:, ::-1], cap_faces + n_cap, side_faces)),
                    validate=True,
                )
                view_mesh.fix_normals()
                view_mesh.visual.vertex_colors = (255, 255, 255, 38)
                meshes.append(view_mesh)
            n_envs_ = self._manager.n_envs_
            markers = (
                StaticMeshBatch(
                    geometries=tuple(
                        MeshGeometry(
                            surface=None,
                            visual=mesh.visual,
                            verts_ref=ArrayRef(mesh.vertices),
                            normals_ref=ArrayRef(mesh.vertex_normals),
                            faces_ref=ArrayRef(mesh.faces),
                        )
                        for mesh in meshes
                    ),
                    pos=torch.zeros((n_envs_, len(meshes), 3), dtype=gs.tc_float, device=gs.device),
                    quat=torch.zeros((n_envs_, len(meshes), 4), dtype=gs.tc_float, device=gs.device),
                ),
            )
        pos, quat = markers[0].pos, markers[0].quat
        pos[envs_idx] = self.get_pos(envs_idx)[..., None, :]
        quat[envs_idx] = self.get_quat(envs_idx)[..., None, :]
        return markers

    def _get_view(self) -> CameraView | None:
        """Return the volume the camera sees, None for a camera whose view its options do not state, such as one whose
        lens a profile defines, which then draws itself alone."""
        return None

    @gs.assert_built
    def get_pos(self, envs_idx=None) -> torch.Tensor:
        """Return the position of the camera in the world.

        `envs_idx` selects the environments, all of them if None. A camera attached to a link follows it at the offset
        its options state, and a fixed one stands at `pos`.
        """
        if self._link is None:
            return torch.tensor(self._options.pos, dtype=gs.tc_float, device=gs.device)
        if self._options.offset_T is None:
            offset_pos = torch.tensor(self._options.pos, dtype=gs.tc_float, device=gs.device)
        else:
            offset_pos = torch.tensor(self._options.offset_T, dtype=gs.tc_float, device=gs.device)[:3, 3]
        link_pos = self._link.get_pos(envs_idx, relative=False)
        link_quat = self._link.get_quat(envs_idx, relative=False)
        return transform_by_trans_quat(offset_pos.expand(link_pos.shape), link_pos, link_quat)

    @gs.assert_built
    def get_quat(self, envs_idx=None) -> torch.Tensor:
        """Return the orientation of the camera in the world, the camera looking down its -z axis (see 'get_pos')."""
        if self._link is None or self._options.offset_T is None:
            pos, lookat, up = (
                torch.tensor(value, dtype=gs.tc_float, device=gs.device)
                for value in (self._options.pos, self._options.lookat, self._options.up)
            )
            offset_quat = T_to_quat(pos_lookat_up_to_T(pos, lookat, up))
        else:
            offset_quat = T_to_quat(torch.tensor(self._options.offset_T, dtype=gs.tc_float, device=gs.device))
        if self._link is None:
            return offset_quat
        link_quat = self._link.get_quat(envs_idx, relative=False)
        return transform_quat_by_quat(offset_quat.expand(link_quat.shape), link_quat)

    # ========================== Attachment handling ==========================

    @gs.assert_built
    def move_to_attach(self):
        """
        Move the camera to follow the currently attached rigid link.

        Uses a shared transform computation and delegates to _apply_camera_transform().
        """
        if self._link is None:
            gs.raise_exception("Camera not attached to any rigid link.")
        self._apply_camera_transform(trans_quat_to_T(self.get_pos(), self.get_quat()))

    # ========================== Hooks for subclasses ==========================

    def _apply_camera_transform(self, camera_T: torch.Tensor):
        """Apply the computed camera transform to the backend-specific camera representation."""
        raise NotImplementedError

    def _render_current_state(self):
        """Perform the actual render for the current state; subclasses must implement."""
        raise NotImplementedError

    # ========================== Shared read() ==========================

    def _get_image_cache_entry(self):
        """Return this sensor's entry in the shared image cache."""
        return self._shared_metadata.image_cache[self._idx]

    def _ensure_rendered_for_current_state(self):
        """Ensure this camera has an up-to-date render before reading.
        Base handles staleness and timestamps; subclasses implement _render_current_state().
        """
        scene = self._manager.sim.scene

        # If the scene time advanced, mark all cameras as stale
        if self._shared_metadata.last_render_timestep != scene.sim.cur_step_global:
            if self._shared_metadata.sensors is not None:
                for sensor in self._shared_metadata.sensors:
                    sensor._stale = True
            self._shared_metadata.last_render_timestep = scene.sim.cur_step_global

        # If this camera is not stale, cache is considered fresh
        if not self._stale:
            return

        # Update camera pose only when attached; detached cameras keep their last world pose
        if self._link is not None:
            self.move_to_attach()

        # Call subclass-specific render
        self._render_current_state()

        # Mark as fresh
        self._stale = False

    def _sanitize_envs_idx(self, envs_idx):
        """Sanitize envs_idx to valid indices."""
        if envs_idx is None:
            return None
        if isinstance(envs_idx, (int, np.integer)):
            return envs_idx
        return np.asarray(envs_idx)

    @gs.assert_built
    def read(self, envs_idx=None) -> CameraReturnType:
        """Render if needed, then read the cached image from the backend-specific cache."""
        self._ensure_rendered_for_current_state()
        cached_image = self._get_image_cache_entry()
        return _camera_read_from_image_cache(self, cached_image, envs_idx, to_numpy=False)


# ========================== Camera Sensor Helpers ==========================
def _camera_read_from_image_cache(sensor, cached_image, envs_idx, *, to_numpy: bool) -> CameraReturnType:
    """
    Shared helper to convert a cached RGB image array into CameraReturnType with correct env handling.

    Parameters
    ----------
    sensor : any camera sensor with _manager and _return_data_cls
    cached_image : np.ndarray | torch.Tensor
        Image cache for this camera, shaped (B, H, W, 3) or (H, W, 3) depending on n_envs.
    envs_idx : None | int | sequence
        Environment index/indices to select.
    to_numpy : bool
        If True and cached_image is a torch Tensor, convert to numpy first.
    """
    if to_numpy and isinstance(cached_image, torch.Tensor):
        cached_image = tensor_to_array(cached_image)

    if envs_idx is None:
        if sensor._manager.n_envs == 0:
            return sensor._return_data_cls(rgb=cached_image[0])
        return sensor._return_data_cls(rgb=cached_image)
    if isinstance(envs_idx, (int, np.integer)):
        return sensor._return_data_cls(rgb=cached_image[envs_idx])
    return sensor._return_data_cls(rgb=cached_image[envs_idx])


# ========================== Rasterizer Camera Sensor ==========================


class RasterizerCameraSensor(
    BaseCameraSensor, Sensor[RasterizerCameraOptions, None, RasterizerCameraSharedMetadata, CameraReturnType]
):
    """
    Rasterizer camera sensor using OpenGL-based rendering.

    This sensor renders RGB images using the existing Rasterizer backend, but operates independently from the scene
    visualizer.
    """

    def __init__(
        self,
        options: RasterizerCameraOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._options: RasterizerCameraOptions
        self._camera_node = None
        self._camera_target = None
        self._camera_wrapper = None
        self._is_camera_registered = False

    # ========================== Sensor Lifecycle ==========================

    @gs.assert_unbuilt
    def build(self):
        """Initialize the rasterizer and register this camera."""
        super().build()

        scene = self._manager.sim.scene

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = gs.List()
            self._shared_metadata.image_cache = {}

            # If a viewer is active, reuse its windowed OpenGL context for both offscreen and onscreen rendering, rather
            # than creating a separate headless context which is fragile.
            if scene.viewer is not None:
                self._shared_metadata.context = scene.visualizer.context
                self._shared_metadata.renderer = scene.visualizer.rasterizer
            else:
                # No viewer - create standalone rasterizer with offscreen context
                self._shared_metadata.context = self._create_standalone_context(scene)
                self._shared_metadata.renderer = Rasterizer(viewer=None, context=self._shared_metadata.context)
                self._shared_metadata.renderer.build()

        self._shared_metadata.sensors.append(self)

        # Register camera now if standalone (offscreen), or defer to first render if using visualizer's rasterizer
        # (visualizer isn't built yet at sensor.build() time)
        if self._shared_metadata.renderer.offscreen:
            self._ensure_camera_registered()

        w, h = self._options.res
        self._shared_metadata.image_cache[self._idx] = torch.zeros(
            (self._manager.n_envs_, h, w, 3), dtype=torch.uint8, device=gs.device
        )

    @classmethod
    def reset(cls, shared_metadata: SharedSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        # The context syncs with the simulation once per step, so a reset at the last synced step must clear that
        shared_metadata.context.reset()

    def _ensure_camera_registered(self):
        """Register this camera with the renderer (no-op if already registered)."""
        if self._is_camera_registered:
            return

        # Add lights from options to the context
        for light_config in self._options.lights:
            if self._shared_metadata.lights is not None:
                light_dict = self._convert_light_config_to_rasterizer(light_config)
                self._shared_metadata.context.add_light(light_dict)

        if self._camera_wrapper is None:
            self._camera_wrapper = RasterizerCameraWrapper(self)
        self._shared_metadata.renderer.add_camera(self._camera_wrapper)
        self._update_camera_pose()
        self._is_camera_registered = True

    def _create_standalone_context(self, scene):
        """Create a simplified RasterizerContext for camera sensors."""
        if not scene.sim.rigid_only and scene.n_envs > 1:
            gs.raise_exception("Rasterizer with n_envs > 1, does not work when using non rigid simulation")
        if scene.n_envs > 1:
            gs.logger.warning(
                "Rasterizer with n_envs > 1 is slow as it doesn't do batched rendering consider using BatchRenderer instead."
            )
        vis_options = VisOptions(
            show_world_frame=False,
            show_cameras=False,
            rendered_envs_idx=range(max(self._manager.n_envs_, 1)),
        )

        context = RasterizerContext(vis_options)
        context.build(scene)
        context.reset()
        return context

    @staticmethod
    def _convert_light_config_to_rasterizer(light_config):
        """Convert a light config dict to a typed light options object for the rasterizer."""
        from genesis.options.vis import DirectionalLight, PointLight

        light_type = light_config.get("type", "directional")
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)

        if light_type == "point":
            pos = light_config.get("pos", (0.0, 0.0, 5.0))
            return PointLight(pos=pos, color=color, intensity=intensity)
        else:
            dir = light_config.get("dir", (0.0, 0.0, -1.0))
            return DirectionalLight(dir=dir, color=color, intensity=intensity)

    def _update_camera_pose(self) -> None:
        """Update camera pose based on options."""
        pos = torch.tensor(self._options.pos, dtype=gs.tc_float, device=gs.device)
        lookat = torch.tensor(self._options.lookat, dtype=gs.tc_float, device=gs.device)
        up = torch.tensor(self._options.up, dtype=gs.tc_float, device=gs.device)

        # If attached to a link, pos is relative to link frame
        if self._link is not None:
            # Convert pos from link-relative to world coordinates
            link_pos = self._link.get_pos(relative=False)
            link_quat = self._link.get_quat(relative=False)

            # Apply pos directly as offset from link
            pos_world = transform_by_quat(pos, link_quat) + link_pos
            pos = pos_world

        transform = pos_lookat_up_to_T(pos, lookat, up)
        self._camera_wrapper.transform = tensor_to_array(transform)
        self._shared_metadata.renderer.update_camera(self._camera_wrapper)

    def _get_view(self) -> CameraView:
        """Return the view of the camera between its clipping planes (see 'BaseCameraSensor._get_view')."""
        return CameraView(
            depths=(self._options.near, self._options.far), is_fisheye=False, is_disk=False, is_radial=False
        )

    def _apply_camera_transform(self, camera_T: torch.Tensor):
        """Update rasterizer camera wrapper from a world transform."""
        self._ensure_camera_registered()
        self._camera_wrapper.transform = tensor_to_array(camera_T)
        self._shared_metadata.renderer.update_camera(self._camera_wrapper)

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        self._ensure_camera_registered()

        self._shared_metadata.renderer.update_scene()
        rgb_arr, _, _, _ = self._shared_metadata.renderer.render_camera(
            self._camera_wrapper, rgb=True, depth=False, segmentation=False, normal=False, split_envs=True
        )

        # Ensure contiguous layout because the rendered array may have negative strides.
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_arr)).to(dtype=torch.uint8, device=gs.device)

        if len(rgb_tensor.shape) == 3:
            # Single environment rendered - add batch dimension.
            rgb_tensor = rgb_tensor.unsqueeze(0)
        self._shared_metadata.image_cache[self._idx][:] = rgb_tensor


# ========================== Raytracer Camera Sensor ==========================
class RaytracerCameraSensor(
    BaseCameraSensor, Sensor[RaytracerCameraOptions, None, RaytracerCameraSharedMetadata, CameraReturnType]
):
    """
    Raytracer camera sensor using LuisaRender path tracing.
    """

    def __init__(
        self,
        options: RaytracerCameraOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._options: RaytracerCameraOptions
        self._camera_obj = None

    def _get_view(self) -> CameraView:
        """Return the view of the camera up to its plane in focus (see 'BaseCameraSensor._get_view')."""
        return CameraView(depths=(0.0, self._options.focus_dist), is_fisheye=False, is_disk=False, is_radial=False)

    @gs.assert_unbuilt
    def build(self) -> None:
        """Register a raytracer camera that reuses the visualizer pipeline."""
        super().build()

        scene = self._manager.sim.scene
        visualizer = scene.visualizer

        renderer = getattr(visualizer, "raytracer", None)
        if renderer is None:
            gs.raise_exception(
                "RaytracerCameraSensor requires the scene to be created with `renderer=gs.renderers.RayTracer(...)`."
            )

        # Multi-environment rendering is not yet supported for Raytracer cameras
        n_envs = self._manager.n_envs
        if n_envs > 1:
            gs.raise_exception(
                f"Raytracer camera sensors do not support multi-environment rendering (n_envs={n_envs}). "
                "Use BatchRenderer camera sensors for batched rendering."
            )

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = []
            self._shared_metadata.image_cache = {}
            self._shared_metadata.renderer = renderer

        self._shared_metadata.sensors.append(self)

        # Add lights from options as mesh lights to the scene
        scene = self._manager.sim.scene
        for light_config in self._options.lights:
            self._add_light_as_mesh_light(scene, light_config)

        # Compute world pose for the camera
        pos = torch.tensor(self._options.pos, dtype=gs.tc_float, device=gs.device)
        lookat = torch.tensor(self._options.lookat, dtype=gs.tc_float, device=gs.device)
        up = torch.tensor(self._options.up, dtype=gs.tc_float, device=gs.device)

        # If attached to a link, transform pos to world coordinates
        if self._link is not None:
            link_pos = self._link.get_pos(relative=False).squeeze(0)
            link_quat = self._link.get_quat(relative=False).squeeze(0)

            # Apply pos directly as offset from link
            pos = transform_by_trans_quat(pos, link_pos, link_quat)

            # Transform lookat and up (no rotation offset since rotation is defined by lookat/up)
            lookat = transform_by_trans_quat(lookat, link_pos, link_quat)
            up = transform_by_quat(up, link_quat)

        lens = None
        if self._options.model == "thinlens":
            lens = ThinLens(
                aperture=self._options.aperture,
                focal_len=self._options.focal_len,
                focus_dist=self._options.focus_dist,
            )
        self._camera_obj = visualizer.add_camera(
            res=self._options.res,
            pos=pos,
            lookat=lookat,
            up=up,
            model="pinhole",
            fov=self._options.fov,
            lens=lens,
            GUI=False,
            spp=self._options.spp,
            denoise=self._options.denoise,
            near=0.05,
            far=100.0,
            env_idx=None if n_envs == 0 else 0,
            debug=False,
        )

        # Attach the visualizer camera to the link if this sensor is attached
        if self._link is not None:
            if self._options.offset_T is not None:
                offset_T = torch.tensor(self._options.offset_T, dtype=gs.tc_float, device=gs.device)
            else:
                pos = torch.tensor(self._options.pos, dtype=gs.tc_float, device=gs.device)
                lookat = torch.tensor(self._options.lookat, dtype=gs.tc_float, device=gs.device)
                up = torch.tensor(self._options.up, dtype=gs.tc_float, device=gs.device)
                offset_T = pos_lookat_up_to_T(pos, lookat, up)
            self._camera_obj.attach(self._link, offset_T)

        w, h = self._options.res
        self._shared_metadata.image_cache[self._idx] = torch.zeros(
            (self._manager.n_envs_, h, w, 3), dtype=torch.uint8, device=gs.device
        )

    @gs.assert_built
    def move_to_attach(self):
        # Bypass original implementation since it will be handled by visualizer
        pass

    def _add_light_as_mesh_light(self, scene, light_config):
        """Add a light as a mesh light to the scene."""
        # Default values for raytracer mesh lights
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)
        radius = light_config.get("radius", 0.5)
        pos = light_config.get("pos", (0.0, 0.0, 5.0))
        revert_dir = light_config.get("revert_dir", False)
        double_sided = light_config.get("double_sided", False)
        cutoff = light_config.get("cutoff", 180.0)

        morph = gs.morphs.Sphere(pos=pos, radius=radius)
        scene.add_mesh_light(
            morph=morph,
            color=(*color, 1.0),
            intensity=intensity,
            revert_dir=revert_dir,
            double_sided=double_sided,
            cutoff=cutoff,
        )

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        if self._link is not None:
            self._camera_obj.move_to_attach()

        rgb_arr, _, _, _ = self._camera_obj.render(
            rgb=True,
            depth=False,
            segmentation=False,
            colorize_seg=False,
            normal=False,
            antialiasing=False,
            force_render=True,
        )
        # Ensure contiguous layout because the rendered array may have negative strides.
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_arr)).to(dtype=torch.uint8, device=gs.device)

        self._shared_metadata.image_cache[self._idx][0] = rgb_tensor


# ========================== Batch Renderer Camera Sensor ==========================


class BatchRendererCameraSensor(
    BaseCameraSensor, Sensor[BatchRendererCameraOptions, None, BatchRendererCameraSharedMetadata, CameraReturnType]
):
    """
    Batch renderer camera sensor using Madrona GPU batch rendering.

    Note: All batch renderer cameras must have the same resolution.
    """

    def __init__(
        self,
        options: BatchRendererCameraOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._options: BatchRendererCameraOptions
        self._camera_obj = None

    @gs.assert_unbuilt
    def build(self):
        """Initialize the batch renderer and register this camera."""
        super().build()

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRendererCameraSensor requires CUDA backend.")

        scene = self._manager.sim.scene

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = gs.List()
            self._shared_metadata.image_cache = {}
            self._shared_metadata.last_render_timestep = -1

            all_sensors = self._manager._sensors_by_type[type(self)]
            resolutions = [s._options.res for s in all_sensors]
            if len(set(resolutions)) > 1:
                gs.raise_exception(
                    f"All BatchRendererCameraSensor instances must have the same resolution. Found: {set(resolutions)}"
                )

            br_options = BatchRendererOptions(use_rasterizer=self._options.use_rasterizer)

            vis_options = VisOptions(
                show_world_frame=False,
                show_cameras=False,
                rendered_envs_idx=range(max(self._manager.n_envs_, 1)),
            )

            self._shared_metadata.visualizer_wrapper = MinimalVisualizerWrapper(scene, all_sensors, vis_options)
            self._shared_metadata.renderer = BatchRenderer(
                self._shared_metadata.visualizer_wrapper, br_options, vis_options
            )

        self._shared_metadata.sensors.append(self)

        # Add lights from options to the renderer
        for light_config in self._options.lights:
            if self._shared_metadata.renderer is not None:
                self._add_light_to_batch_renderer(light_config)

        self._camera_obj = BatchRendererCameraWrapper(self)

        if len(self._shared_metadata.sensors) == len(self._manager._sensors_by_type[type(self)]):
            self._shared_metadata.visualizer_wrapper._cameras = [s._camera_obj for s in self._shared_metadata.sensors]
            self._shared_metadata.renderer.build()

        w, h = self._options.res
        self._shared_metadata.image_cache[self._idx] = torch.zeros(
            (self._manager.n_envs_, h, w, 3), dtype=torch.uint8, device=gs.device
        )

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        sensors = self._shared_metadata.sensors or [self]

        for sensor in sensors:
            if sensor._link is not None:
                sensor.move_to_attach()

        self._shared_metadata.renderer.update_scene(force_render=True)

        rgb_arr, *_ = self._shared_metadata.renderer.render(
            rgb=True, depth=False, segmentation=False, normal=False, antialiasing=False, force_render=True
        )

        # rgb_arr might be a tuple of arrays (one per camera) or a single array
        if isinstance(rgb_arr, (tuple, list)):
            rgb_arrs = [torch.as_tensor(arr).to(dtype=torch.uint8, device=gs.device) for arr in rgb_arr]
        else:
            rgb_arrs = torch.as_tensor(rgb_arr).to(dtype=torch.uint8, device=gs.device)

        for sensor, rgb_arr in zip(sensors, rgb_arrs):
            sensor._shared_metadata.image_cache[sensor._idx][:] = rgb_arr
            sensor._stale = False

        self._shared_metadata.last_render_timestep = self._manager.sim.cur_step_global

    def _get_view(self) -> CameraView:
        """Return the view of the camera between its near and far depths (see 'BaseCameraSensor._get_view').

        The rasterizer clips the rays at planes and draws a fisheye by resampling a pinhole image, which holds its
        inscribed disk alone. The ray tracer clips each ray at distances from the center of projection.
        """
        is_fisheye = self._options.model == "fisheye"
        return CameraView(
            depths=(self._options.near, self._options.far),
            is_fisheye=is_fisheye,
            is_disk=is_fisheye and self._options.use_rasterizer,
            is_radial=not self._options.use_rasterizer,
        )

    def _apply_camera_transform(self, camera_T: torch.Tensor):
        """Update batch renderer camera from a world transform."""
        # Note: BatchRenderer will pick up the updated transform on next render
        self._camera_obj.transform = camera_T
        self._camera_obj._pos = T_to_trans(camera_T)

    def _add_light_to_batch_renderer(self, light_config):
        """Add a light to the batch renderer."""
        # Default values for batch renderer
        pos = light_config.get("pos", (0.0, 0.0, 5.0))
        dir = light_config.get("dir", (0.0, 0.0, -1.0))
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)
        directional = light_config.get("directional", True)
        castshadow = light_config.get("castshadow", True)
        cutoff = light_config.get("cutoff", 45.0)
        attenuation = light_config.get("attenuation", (1.0, 0.0, 0.0))

        self._shared_metadata.renderer.add_light(
            pos=pos,
            dir=dir,
            color=color,
            intensity=intensity,
            directional=directional,
            castshadow=castshadow,
            cutoff=cutoff,
            attenuation=attenuation,
        )
