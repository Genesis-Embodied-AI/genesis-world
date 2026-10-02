"""
Camera sensors for rendering: Rasterizer, Raytracer, and Batch Renderer.
"""

from typing import TYPE_CHECKING, ClassVar, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.options.renderers import BatchRenderer as BatchRendererOptions
from genesis.options.sensors import BatchRendererCameraOptions, RasterizerCameraOptions, RaytracerCameraOptions
from genesis.options.vis import DirectionalLight, PointLight, VisOptions
from genesis.typing import EnvsIdxType
from genesis.utils.geom import (
    T_to_quat,
    T_to_trans,
    pos_lookat_up_to_T,
    trans_quat_to_T,
    transform_by_quat,
    transform_by_trans_quat,
)
from genesis.utils.misc import indices_to_mask, tensor_to_array
from genesis.vis.batch_renderer import BatchRenderer
from genesis.vis.rasterizer import Rasterizer
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import ArrayT, LinkAttachedSensorArrayMixin, OptionsT, Sensor, SensorArray

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

    from .sensor_manager import SensorManager


# ========================== Data Class ==========================


class CameraReturnType(NamedTuple):
    """Camera sensor return data."""

    rgb: torch.Tensor


class MinimalVisualizerWrapper:
    """A stand-in for the visualizer that BatchRenderer reads its cameras and rasterizer context from."""

    def __init__(self, scene: "Scene", cameras: list["BatchRendererCameraWrapper"], vis_options: VisOptions):
        self.scene = scene
        self._cameras = cameras

        # Create a minimal rasterizer context for camera frustum visualization (required by BatchRenderer even though
        # cameras don't render frustums)
        self._context = RasterizerContext(vis_options)
        self._context.build(scene)
        self._context.reset()


class BaseCameraWrapper:
    """
    The record of one camera a renderer backend reads: its intrinsics and its index among the cameras of its type, both
    fixed at build.
    """

    def __init__(self, sensor: "CameraSensor"):
        self.sensor = sensor
        self.uid = sensor.idx
        self.res = sensor.options.res
        self.fov = sensor.options.fov
        self.near = sensor.options.near
        self.far = sensor.options.far


class RasterizerCameraWrapper(BaseCameraWrapper):
    """Lightweight wrapper object used by the rasterizer backend."""

    def __init__(self, sensor: "RasterizerCameraSensor"):
        super().__init__(sensor)
        self.aspect_ratio = self.res[0] / self.res[1]


class BatchRendererCameraWrapper(BaseCameraWrapper):
    """Wrapper object used by the batch renderer backend."""

    def __init__(self, sensor: "BatchRendererCameraSensor"):
        super().__init__(sensor)
        self.idx = sensor.idx
        self.model = sensor.options.model

        pos = torch.tensor(sensor.options.pos, dtype=gs.tc_float, device=gs.device)
        lookat = torch.tensor(sensor.options.lookat, dtype=gs.tc_float, device=gs.device)
        up = torch.tensor(sensor.options.up, dtype=gs.tc_float, device=gs.device)
        self._pos = pos
        self._lookat = lookat
        self._up = up
        self.transform = pos_lookat_up_to_T(pos, lookat, up)

    def set_transform(self, camera_T: torch.Tensor) -> None:
        """Set the world pose of the camera, which the batch renderer reads at its next render."""
        self.transform = camera_T
        self._pos = T_to_trans(camera_T)

    def get_pos(self) -> torch.Tensor:
        """Return the camera position the batch renderer reads, per environment."""
        n_envs = self.sensor._array._sim.n_envs
        if self._pos.ndim > 1 or n_envs == 0:
            return self._pos
        return self._pos[None].expand((n_envs, -1))

    def get_quat(self) -> torch.Tensor:
        """Return the camera quaternion the batch renderer reads, per environment."""
        quat = T_to_quat(self.transform)
        n_envs = self.sensor._array._sim.n_envs
        if quat.ndim > 1 or n_envs == 0:
            return quat
        return quat[None].expand((n_envs, -1))


# ========================== Base Camera Array and Handle ==========================


class CameraSensorArray(LinkAttachedSensorArrayMixin, SensorArray[OptionsT, CameraReturnType]):
    """
    Array of the cameras of one backend, rendering an RGB frame per environment on read and keeping the frames between
    reads.

    A read renders a camera again once the scene stepped after its last render, and a reset drops every frame. A
    backend implements `_apply_camera_transform` and `_render_current_state`.
    """

    # Cameras render on read (see read), so the per-step ring pipeline and the options it implements are refused
    has_ring_pipeline: ClassVar[bool] = False
    # Whether the backend renders every camera of the type in one pass (see _render_current_state)
    has_batch_render: ClassVar[bool] = False

    def __init__(self, sim: "Simulator", manager: "SensorManager"):
        super().__init__(sim, manager)
        # The frame of each camera, (B, h, w, 3) uint8, released by destroy
        self.images: list[torch.Tensor] = []

    def _get_return_format(self, options: OptionsT) -> tuple[tuple[int, ...], ...]:
        w, h = options.res
        return ((h, w, 3),)

    def _get_cache_dtype(self) -> torch.dtype:
        return torch.uint8

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        """A camera draws nothing."""

    def build(self) -> None:
        super().build()

        _B = self._sim._B
        # The frames of each camera, written by the renders and served by the reads
        self.images.clear()
        # The pose of each camera in the frame of its link: the authored transform, or the one its position, target
        # and up vector describe
        self.offsets_T: list[torch.Tensor] = []
        for sensor in self._sensors:
            options = sensor.options
            w, h = options.res
            self.images.append(torch.zeros((_B, h, w, 3), dtype=torch.uint8, device=gs.device))
            if options.offset_T is not None:
                self.offsets_T.append(torch.tensor(options.offset_T, dtype=gs.tc_float, device=gs.device))
            else:
                pos = torch.tensor(options.pos, dtype=gs.tc_float, device=gs.device)
                lookat = torch.tensor(options.lookat, dtype=gs.tc_float, device=gs.device)
                up = torch.tensor(options.up, dtype=gs.tc_float, device=gs.device)
                self.offsets_T.append(pos_lookat_up_to_T(pos, lookat, up))

        # The step of the scene at the last render of each camera, -1 for a frame never rendered
        self._rendered_steps = [-1] * self.n_sensors

    def destroy(self) -> None:
        super().destroy()
        self.images.clear()

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        """Drop the frames of every camera."""
        super().reset(envs_idx)
        # FIXME: the frames key on the step of the whole scene, which a reset rewinds, so a reset of some environments
        # drops the frames of all of them. A per-environment key would keep the frames of the others.
        self._forget_frames()

    # ========================== Attachment handling ==========================

    def move_to_attach(self, i_s: int) -> None:
        """Move camera ``i_s`` to the pose of the link it is attached to, composed with its offset."""
        link = self._links[i_s]
        if link is None:
            gs.raise_exception("Camera not attached to any rigid link.")
        link_T = trans_quat_to_T(link.get_pos(relative=False), link.get_quat(relative=False))
        self._apply_camera_transform(i_s, torch.matmul(link_T, self.offsets_T[i_s]))

    # ========================== Hooks for subclasses ==========================

    def _apply_camera_transform(self, i_s: int, camera_T: torch.Tensor) -> None:
        """Set the world pose of camera ``i_s`` in the backend."""
        raise NotImplementedError(f"{type(self).__name__} has not implemented `_apply_camera_transform()`.")

    def _render_current_state(self, i_s: int) -> None:
        """
        Render the current state of the scene for camera ``i_s`` into ``self.images[i_s]``, in every environment.

        A backend with `has_batch_render` also writes the frames of the other cameras of the type into their images.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `_render_current_state()`.")

    # ========================== Reads ==========================

    def _forget_frames(self) -> None:
        """
        Drop the frames of every camera.

        The next read renders them again. A reset calls it, and so does a backend whose camera moved by itself.
        """
        self._rendered_steps[:] = [-1] * self.n_sensors

    def _render(self, i_s: int, envs_idx: EnvsIdxType | None) -> torch.Tensor:
        """
        Return the frames of camera ``i_s`` in the selected environments (all of them when None), after rendering
        them again if the scene stepped after the last render.
        """
        step = self._sim.cur_step_global
        if self._rendered_steps[i_s] != step:
            cameras_idx = range(self.n_sensors) if self.has_batch_render else (i_s,)
            # Only an attached camera moves here. A detached one keeps its last world pose
            for j_s in cameras_idx:
                if self._links[j_s] is not None:
                    self.move_to_attach(j_s)
            self._render_current_state(i_s)
            for j_s in cameras_idx:
                self._rendered_steps[j_s] = step
        images = self.images[i_s]
        frames = images[indices_to_mask(envs_idx)]
        # A read is a snapshot the caller keeps and may write, so a slice selection, which views the images, is copied
        if frames.untyped_storage().data_ptr() == images.untyped_storage().data_ptr():
            frames = frames.clone()
        return frames

    def read(self, i_s: int, envs_idx: EnvsIdxType | None = None, is_ground_truth: bool = False) -> CameraReturnType:
        """
        Return the RGB frame of camera ``i_s`` in the selected environments (all of them by default), from the current
        state of the scene.

        The ground truth of a camera is its frame.
        """
        frames = self._render(i_s, envs_idx)
        if isinstance(envs_idx, (int, np.integer)) or (envs_idx is None and self._sim.n_envs == 0):
            frames = frames[0]
        return CameraReturnType(rgb=frames)

    def read_all(
        self, entity_idx: int | None = None, envs_idx: EnvsIdxType | None = None, is_ground_truth: bool = False
    ) -> torch.Tensor:
        """
        Return the frames of every camera of the type, or of those attached to an entity, laid end to end per
        environment in one flat uint8 tensor.

        The result is None when no camera is attached to the requested entity.
        """
        target_entity_idx = None if entity_idx is None else (-1 if entity_idx < 0 else entity_idx)
        cameras_idx = [
            i_s
            for i_s, sensor in enumerate(self._sensors)
            if target_entity_idx is None or sensor.options.entity_idx == target_entity_idx
        ]
        if not cameras_idx:
            return None
        frames = [self._render(i_s, envs_idx) for i_s in cameras_idx]
        tensor = torch.cat([camera_frames.reshape((camera_frames.shape[0], -1)) for camera_frames in frames], dim=1)
        if self._sim.n_envs == 0:
            tensor = tensor[0]
        return tensor


class CameraSensor(Sensor[OptionsT, ArrayT]):
    """Handle of one camera: `read` renders its RGB frame from the current state of the scene."""

    @gs.assert_built
    def move_to_attach(self) -> None:
        """Move the camera to the pose of the link it is attached to, composed with its offset."""
        self._array.move_to_attach(self._idx)


# ========================== Rasterizer Camera Sensor ==========================


class RasterizerCameraSensorArray(CameraSensorArray[RasterizerCameraOptions]):
    """
    Array of the rasterizer cameras, rendering with OpenGL.

    With a viewer they render through its rasterizer and context. Without one they render through an offscreen
    rasterizer and context of their own.
    """

    def __init__(self, sim: "Simulator", manager: "SensorManager"):
        super().__init__(sim, manager)
        self.renderer: Rasterizer | None = None
        self.context: RasterizerContext | None = None

    def build(self) -> None:
        super().build()

        scene = self._sim.scene
        if scene.viewer is not None:
            self.context = scene.visualizer.context
            self.renderer = scene.visualizer.rasterizer
        else:
            if not scene.sim._rigid_only and scene.n_envs > 1:
                gs.raise_exception("Rasterizer with n_envs > 1, does not work when using non rigid simulation")
            if scene.n_envs > 1:
                gs.logger.warning(
                    "Rasterizer with n_envs > 1 is slow as it doesn't do batched rendering consider using "
                    "BatchRenderer instead."
                )
            vis_options = VisOptions(
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
                rendered_envs_idx=range(scene.sim._B),
            )
            self.context = RasterizerContext(vis_options)
            self.context.build(scene)
            self.context.reset()
            self.renderer = Rasterizer(viewer=None, context=self.context)
            self.renderer.build()

        self._camera_wrappers = [RasterizerCameraWrapper(sensor) for sensor in self._sensors]
        self._is_camera_registered = [False] * self.n_sensors
        # The viewer's rasterizer takes the cameras at the first render, its visualizer being built after the sensors
        if self.renderer.offscreen:
            for i_s in range(self.n_sensors):
                self._ensure_camera_registered(i_s)

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        super().reset(envs_idx)
        # The context syncs with the simulation once per step, so a reset at the last synced step must clear that
        self.context.reset()

    def destroy(self) -> None:
        super().destroy()

        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
        if self.context is not None:
            self.context.destroy()
            self.context = None

    def _ensure_camera_registered(self, i_s: int) -> None:
        """Register camera ``i_s`` and its lights with the renderer, once."""
        if self._is_camera_registered[i_s]:
            return

        options = self._sensors[i_s].options
        for light_config in options.lights:
            color = light_config.get("color", (1.0, 1.0, 1.0))
            intensity = light_config.get("intensity", 1.0)
            if light_config.get("type", "directional") == "point":
                pos = light_config.get("pos", (0.0, 0.0, 5.0))
                self.context.add_light(PointLight(pos=pos, color=color, intensity=intensity))
            else:
                direction = light_config.get("dir", (0.0, 0.0, -1.0))
                self.context.add_light(DirectionalLight(dir=direction, color=color, intensity=intensity))

        camera_wrapper = self._camera_wrappers[i_s]
        self.renderer.add_camera(camera_wrapper)

        # The configured position is relative to the link once attached to a built one (move_to_attach corrects the
        # pose at the first render otherwise)
        pos = torch.tensor(options.pos, dtype=gs.tc_float, device=gs.device)
        lookat = torch.tensor(options.lookat, dtype=gs.tc_float, device=gs.device)
        up = torch.tensor(options.up, dtype=gs.tc_float, device=gs.device)
        link = self._links[i_s]
        if link is not None and link.is_built:
            pos = transform_by_quat(pos, link.get_quat(relative=False)) + link.get_pos(relative=False)
        camera_wrapper.transform = tensor_to_array(pos_lookat_up_to_T(pos, lookat, up))
        self.renderer.update_camera(camera_wrapper)
        self._is_camera_registered[i_s] = True

    def _apply_camera_transform(self, i_s: int, camera_T: torch.Tensor) -> None:
        self._ensure_camera_registered(i_s)
        camera_wrapper = self._camera_wrappers[i_s]
        camera_wrapper.transform = tensor_to_array(camera_T)
        self.renderer.update_camera(camera_wrapper)

    def _render_current_state(self, i_s: int) -> None:
        self._ensure_camera_registered(i_s)

        self.renderer.update_scene()
        rgb_arr, _, _, _ = self.renderer.render_camera(
            self._camera_wrappers[i_s], rgb=True, depth=False, segmentation=False, normal=False, split_envs=True
        )

        # Ensure contiguous layout because the rendered array may have negative strides.
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_arr)).to(dtype=torch.uint8, device=gs.device)

        if rgb_tensor.ndim == 3:
            # Single environment rendered - add batch dimension.
            rgb_tensor = rgb_tensor.unsqueeze(0)
        self.images[i_s].copy_(rgb_tensor)


class RasterizerCameraSensor(CameraSensor[RasterizerCameraOptions, RasterizerCameraSensorArray]):
    """
    Rasterizer camera sensor using OpenGL-based rendering.

    This sensor renders RGB images using the existing Rasterizer backend, but operates independently from the scene
    visualizer.
    """


# ========================== Raytracer Camera Sensor ==========================


class RaytracerCameraSensorArray(CameraSensorArray[RaytracerCameraOptions]):
    """Array of the raytracer cameras, each a camera of the visualizer's LuisaRender path tracer."""

    def build(self) -> None:
        super().build()

        scene = self._sim.scene
        visualizer = scene.visualizer
        if visualizer.raytracer is None:
            gs.raise_exception(
                "RaytracerCameraSensor requires the scene to be created with `renderer=gs.renderers.RayTracer(...)`."
            )
        n_envs = self._sim.n_envs
        if n_envs > 1:
            gs.raise_exception(
                f"Raytracer camera sensors do not support multi-environment rendering (n_envs={n_envs}). "
                "Use BatchRenderer camera sensors for batched rendering."
            )

        self._camera_objs = []
        for i_s, (sensor, link) in enumerate(zip(self._sensors, self._links)):
            options = sensor.options
            # Add lights from options as mesh lights to the scene
            for light_config in options.lights:
                if not scene.is_built:
                    color = light_config.get("color", (1.0, 1.0, 1.0))
                    morph = gs.morphs.Sphere(
                        pos=light_config.get("pos", (0.0, 0.0, 5.0)), radius=light_config.get("radius", 0.5)
                    )
                    scene.add_mesh_light(
                        morph=morph,
                        color=(*color, 1.0),
                        intensity=light_config.get("intensity", 1.0),
                        revert_dir=light_config.get("revert_dir", False),
                        double_sided=light_config.get("double_sided", False),
                        cutoff=light_config.get("cutoff", 180.0),
                    )

            # The configured pose is relative to the link once attached to a built one (the visualizer camera corrects
            # the pose at the first render otherwise)
            pos = torch.tensor(options.pos, dtype=gs.tc_float, device=gs.device)
            lookat = torch.tensor(options.lookat, dtype=gs.tc_float, device=gs.device)
            up = torch.tensor(options.up, dtype=gs.tc_float, device=gs.device)
            if link is not None and link.is_built:
                link_pos = link.get_pos(relative=False).squeeze(0)
                link_quat = link.get_quat(relative=False).squeeze(0)
                pos = transform_by_trans_quat(pos, link_pos, link_quat)
                lookat = transform_by_trans_quat(lookat, link_pos, link_quat)
                up = transform_by_quat(up, link_quat)

            camera_obj = visualizer.add_camera(
                res=options.res,
                pos=pos,
                lookat=lookat,
                up=up,
                model=options.model,
                fov=options.fov,
                aperture=options.aperture,
                focus_dist=options.focus_dist,
                GUI=False,
                spp=options.spp,
                denoise=options.denoise,
                near=0.05,
                far=100.0,
                env_idx=None if n_envs == 0 else 0,
                debug=False,
            )
            if link is not None:
                camera_obj.attach(link, self.offsets_T[i_s])
            self._camera_objs.append(camera_obj)

    def move_to_attach(self, i_s: int) -> None:
        """The visualizer camera follows its link by itself (see _render_current_state)."""

    def _render_current_state(self, i_s: int) -> None:
        camera_obj = self._camera_objs[i_s]
        if self._links[i_s] is not None:
            camera_obj.move_to_attach()

        rgb_arr, _, _, _ = camera_obj.render(
            rgb=True,
            depth=False,
            segmentation=False,
            colorize_seg=False,
            normal=False,
            antialiasing=False,
            force_render=True,
        )
        # Ensure contiguous layout because the rendered array may have negative strides.
        self.images[i_s][0] = torch.from_numpy(np.ascontiguousarray(rgb_arr)).to(dtype=torch.uint8, device=gs.device)


class RaytracerCameraSensor(CameraSensor[RaytracerCameraOptions, RaytracerCameraSensorArray]):
    """
    Raytracer camera sensor using LuisaRender path tracing.
    """


# ========================== Batch Renderer Camera Sensor ==========================


class BatchRendererCameraSensorArray(CameraSensorArray[BatchRendererCameraOptions]):
    """Array of the batch renderer cameras, all rendered in one pass by one Madrona GPU batch renderer.

    The renderer takes every camera at build with the lights of all of them. The cameras share one resolution.
    """

    has_batch_render: ClassVar[bool] = True

    def __init__(self, sim: "Simulator", manager: "SensorManager"):
        super().__init__(sim, manager)
        self.renderer: BatchRenderer | None = None
        self.visualizer_wrapper: MinimalVisualizerWrapper | None = None

    def build(self) -> None:
        super().build()

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRendererCameraSensor requires CUDA backend.")
        resolutions = {sensor.options.res for sensor in self._sensors}
        if len(resolutions) > 1:
            gs.raise_exception(
                f"All BatchRendererCameraSensor instances must have the same resolution. Found: {resolutions}"
            )

        self._camera_wrappers = [BatchRendererCameraWrapper(sensor) for sensor in self._sensors]
        br_options = BatchRendererOptions(use_rasterizer=self._sensors[0].options.use_rasterizer)
        vis_options = VisOptions(
            show_world_frame=False,
            show_link_frame=False,
            show_cameras=False,
            rendered_envs_idx=range(self._sim._B),
        )
        self.visualizer_wrapper = MinimalVisualizerWrapper(self._sim.scene, self._camera_wrappers, vis_options)
        self.renderer = BatchRenderer(self.visualizer_wrapper, br_options, vis_options)

        for sensor in self._sensors:
            for light_config in sensor.options.lights:
                self.renderer.add_light(
                    pos=light_config.get("pos", (0.0, 0.0, 5.0)),
                    dir=light_config.get("dir", (0.0, 0.0, -1.0)),
                    color=light_config.get("color", (1.0, 1.0, 1.0)),
                    intensity=light_config.get("intensity", 1.0),
                    directional=light_config.get("directional", True),
                    castshadow=light_config.get("castshadow", True),
                    cutoff=light_config.get("cutoff", 45.0),
                    attenuation=light_config.get("attenuation", (1.0, 0.0, 0.0)),
                )

        self.renderer.build()

    def destroy(self) -> None:
        super().destroy()

        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
        self.visualizer_wrapper = None

    def _render_current_state(self, i_s: int) -> None:
        # The renderer draws every camera in one pass and syncs the scene itself. The array keys the frames on the
        # step (see _rendered_steps), so the renderer's own step cache is bypassed: a reset rewinds the step while
        # that cache keeps the last frame
        rgb_arr, *_ = self.renderer.render(
            rgb=True, depth=False, segmentation=False, normal=False, antialiasing=False, force_render=True
        )

        # The frames of every camera come out of the one render
        for images, rgb_arr in zip(self.images, rgb_arr):
            rgb_tensor = torch.as_tensor(rgb_arr).to(dtype=torch.uint8, device=gs.device)
            # A scene without parallel environments comes back without the batch axis
            if rgb_tensor.ndim == 3:
                rgb_tensor = rgb_tensor[None]
            images.copy_(rgb_tensor)

    def _apply_camera_transform(self, i_s: int, camera_T: torch.Tensor) -> None:
        self._camera_wrappers[i_s].set_transform(camera_T)


class BatchRendererCameraSensor(CameraSensor[BatchRendererCameraOptions, BatchRendererCameraSensorArray]):
    """
    Batch renderer camera sensor using Madrona GPU batch rendering.

    Note: All batch renderer cameras must have the same resolution.
    """
