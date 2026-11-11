import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.options.sensors import RaycastPattern, Raycaster as RaycasterOptions
from genesis.utils.geom import normalize, transform_by_quat, transform_by_trans_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field
from genesis.utils.raycast_qd import RaycastTarget, get_raycast_targets, kernel_cast_rays
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import (
    KinematicSensorMetadataMixin,
    KinematicSensorMixin,
    SharedSensorContext,
    SimpleSensor,
    SimpleSensorMetadata,
)

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh

    from .sensor_manager import SensorManager


class RaycastContext(SharedSensorContext):
    """
    Per-simulator raycast targets, shared across sensor types that cast rays.

    Holds the targets of each active solver: the collision geoms of a rigid solver, and the visual geoms opted into
    ``material.use_visual_raycasting`` (see get_raycast_targets).
    """

    def __init__(self, sim):
        super().__init__(sim)
        self._targets: list[RaycastTarget] = []

    @property
    def targets(self) -> list[RaycastTarget]:
        """The targets of the active solvers.

        Raises if inactive: only a consumer that activated it may read them.
        """
        if not self._active:
            raise gs.GenesisException("RaycastContext queried before activation: no sensor declared a raycast need.")
        return self._targets

    @property
    def collision_target(self) -> RaycastTarget | None:
        """The target of the collision geoms of the rigid solver, None without collision face."""
        return next(iter([target for target in self.targets if not target.is_visual]), None)

    def activate(self):
        """Build the targets of the active solvers at the first activation, and nothing at the next ones."""
        if self._active:
            return
        self._active = True
        for solver in (self._sim.rigid_solver, self._sim.kinematic_solver):
            if not solver.is_active:
                continue
            # The collision faces are those of the rigid solver alone
            if isinstance(solver, RigidSolver):
                self._targets.extend(get_raycast_targets(solver, is_visual=False))
            self._targets.extend(get_raycast_targets(solver, is_visual=True))
        self.update()

    def update(self):
        """Fit the top-level tree of every target to the current poses of its geoms."""
        if not self._active:
            return
        for target in self._targets:
            target.update()

    def reset(self, envs_idx):
        self.update()

    def destroy(self):
        self._targets.clear()


@dataclass
class RaycasterSharedMetadata(KinematicSensorMetadataMixin, SimpleSensorMetadata):
    # The targets cast against each frame live on the shared RaycastContext (see get_raycast_targets), so a Raycaster
    # and a DepthCamera share one set of trees. The casts chain into the output cache (see kernel_cast_rays). The link
    # poses of the sensors are gathered per solver (see KinematicSensorMetadataMixin.solver_groups).

    # Per-step scratch tensors for sensor link poses, lazily allocated on the first cast (B and n_sensors known).
    links_pos: torch.Tensor | None = None
    links_quat: torch.Tensor | None = None

    sensors_ray_start_idx: list[int] = field(default_factory=list)
    total_n_rays: int = 0
    total_cache_size: int = 0

    min_ranges: torch.Tensor = make_tensor_field((0,))
    max_ranges: torch.Tensor = make_tensor_field((0,))
    no_hit_values: torch.Tensor = make_tensor_field((0,))
    return_world_frame: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)

    patterns: list[RaycastPattern] = field(default_factory=list)
    ray_dirs: torch.Tensor = make_tensor_field((0, 3))
    ray_starts: torch.Tensor = make_tensor_field((0, 3))
    ray_starts_world: torch.Tensor = make_tensor_field((0, 3))
    ray_dirs_world: torch.Tensor = make_tensor_field((0, 3))

    points_to_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_cache_offsets: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_point_offsets: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_point_counts: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_return_points: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)


class RaycasterReturnType(NamedTuple):
    points: torch.Tensor | None
    distances: torch.Tensor


class RaycasterSensor(
    KinematicSensorMixin, SimpleSensor[RaycasterOptions, RaycastContext, RaycasterSharedMetadata, RaycasterReturnType]
):
    def __init__(self, options: RaycasterOptions, idx: int, shared_context, shared_metadata, manager: "SensorManager"):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self.ray_starts: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)
        self.ray_dirs: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)

    def build(self):
        super().build()

        # A raycaster always casts, so activate the shared ``RaycastContext`` now: the first consumer's activation
        # builds the targets. Every raycaster then validates there is geometry to cast against.
        self._shared_context.activate()
        # The first raycaster seeds the leading boundary (0) of the per-sensor offsets into the shared cache tensor.
        if self._idx == 0:
            self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
                self._shared_metadata.sensor_cache_offsets, 0
            )
        if not self._shared_context.targets:
            gs.raise_exception(
                "Raycaster sensor has no geometry to raycast against: rigid_solver is inactive and no entity "
                "has material.use_visual_raycasting=True."
            )

        self._shared_metadata.patterns.append(self._options.pattern)

        ray_starts = self._options.pattern.ray_starts.reshape(-1, 3)
        self.ray_starts = transform_by_trans_quat(
            ray_starts, self._shared_metadata.offsets_pos[0, -1, :], self._shared_metadata.offsets_quat[0, -1, :]
        )
        self._shared_metadata.ray_starts = torch.cat([self._shared_metadata.ray_starts, self.ray_starts])

        ray_dirs = self._options.pattern.ray_dirs.reshape(-1, 3)
        self.ray_dirs = transform_by_quat(ray_dirs, self._shared_metadata.offsets_quat[0, -1, :])
        self._shared_metadata.ray_dirs = torch.cat([self._shared_metadata.ray_dirs, self.ray_dirs])

        num_rays = math.prod(self._options.pattern.return_shape)
        self._shared_metadata.sensors_ray_start_idx.append(self._shared_metadata.total_n_rays)

        # Cache offsets are a running cumulative sum of the per-sensor cache sizes, so sensors with different sizes
        # (e.g. a points lidar next to a distances-only depth camera) pack without gaps or overlap.
        self._shared_metadata.total_cache_size += self._cache_size
        self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
            self._shared_metadata.sensor_cache_offsets, self._shared_metadata.total_cache_size
        )
        self._shared_metadata.sensor_point_offsets = concat_with_tensor(
            self._shared_metadata.sensor_point_offsets, self._shared_metadata.total_n_rays
        )
        self._shared_metadata.sensor_point_counts = concat_with_tensor(
            self._shared_metadata.sensor_point_counts, num_rays
        )
        self._shared_metadata.sensor_return_points = concat_with_tensor(
            self._shared_metadata.sensor_return_points, self._options.return_points
        )
        self._shared_metadata.total_n_rays += num_rays

        self._shared_metadata.points_to_sensor_idx = concat_with_tensor(
            self._shared_metadata.points_to_sensor_idx, [self._idx] * num_rays, flatten=True
        )
        self._shared_metadata.return_world_frame = concat_with_tensor(
            self._shared_metadata.return_world_frame, self._options.return_world_frame
        )
        self._shared_metadata.min_ranges = concat_with_tensor(self._shared_metadata.min_ranges, self._options.min_range)
        self._shared_metadata.max_ranges = concat_with_tensor(self._shared_metadata.max_ranges, self._options.max_range)
        self._shared_metadata.no_hit_values = concat_with_tensor(
            self._shared_metadata.no_hit_values, self._options.no_hit_value
        )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        shape = self._options.pattern.return_shape
        # Distances-only: drop the (*shape, 3) points field so the cache holds just the distances.
        if not self._options.return_points:
            return (shape,)
        return ((*shape, 3), shape)

    def _get_formatted_data(self, tensor: torch.Tensor, envs_idx=None) -> RaycasterReturnType:
        # With points disabled the base class returns a bare distances tensor; re-wrap it as RaycasterReturnType so
        # the (points, distances) NamedTuple contract holds, with points=None.
        data = super()._get_formatted_data(tensor, envs_idx)
        if self._options.return_points:
            return data
        return RaycasterReturnType(points=None, distances=data)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_raw_data(
        cls, shared_context: RaycastContext, shared_metadata: RaycasterSharedMetadata, raw_data_T: torch.Tensor
    ):
        # The targets were already refreshed once this step by SensorManager (see RaycastContext.update)
        targets = shared_context.targets

        # Allocate the link-pose scratch buffers on first cast (B and n_sensors are known here). Identity quat is baked
        # into the initial allocation so static sensors (entity_idx<0) leave their rows at identity, letting the cast
        # kernel apply pos_offset / euler_offset in world frame.
        if shared_metadata.links_pos is None:
            B = targets[0].solver._B
            shared_metadata.links_pos = torch.zeros(
                B, shared_metadata.n_sensors, 3, device=gs.device, dtype=gs.tc_float
            )
            shared_metadata.links_quat = torch.zeros(
                B, shared_metadata.n_sensors, 4, device=gs.device, dtype=gs.tc_float
            )
            shared_metadata.links_quat[:, :, 0] = 1.0

        # Gather link poses per sensor. Sensors are pre-bucketed into shared_metadata.solver_groups at build time so
        # this loop issues one bulk get_links_pos / get_links_quat per solver with already-tensor-typed indices.
        links_pos = shared_metadata.links_pos
        links_quat = shared_metadata.links_quat
        for group in shared_metadata.solver_groups:
            pos = group.solver.get_links_pos(links_idx=group.links_idx)
            quat = group.solver.get_links_quat(links_idx=group.links_idx)
            if group.solver.n_envs == 0:
                pos = pos[None]
                quat = quat[None]
            links_pos[:, group.sensor_cols, :] = pos
            links_quat[:, group.sensor_cols, :] = quat

        # The targets cast in one launch each, chained into one output buffer (see kernel_cast_rays)
        for i_target, target in enumerate(targets):
            kernel_cast_rays(
                shared_metadata.points_to_sensor_idx,
                links_pos,
                links_quat,
                shared_metadata.ray_starts,
                shared_metadata.ray_dirs,
                shared_metadata.max_ranges,
                shared_metadata.no_hit_values,
                shared_metadata.return_world_frame,
                shared_metadata.sensor_cache_offsets,
                shared_metadata.sensor_point_offsets,
                shared_metadata.sensor_point_counts,
                shared_metadata.sensor_return_points,
                raw_data_T,
                target,
                eps=gs.EPS,
                is_merge=i_target > 0,
                is_last=i_target == len(targets) - 1,
            )

    def _draw_debug(self, context: "RasterizerContext", envs_idx: list[int]) -> list[tuple[None, "Mesh", None]]:
        """Make spheres at the start and at the hit point of every ray, for the first rendered environment."""
        if 0 not in envs_idx:
            return []
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        data = self.read(env_idx)

        pos = self._link.get_pos(env_idx, relative=False)
        quat = self._link.get_quat(env_idx, relative=False)
        if pos.ndim == 2:
            pos, quat = pos[0], quat[0]

        ray_starts = transform_by_trans_quat(self.ray_starts, pos, quat)

        if self._options.return_points:
            points = data.points.reshape((-1, 3))
            if not self._options.return_world_frame:
                points = transform_by_trans_quat(points + self.ray_starts, pos, quat)
        else:
            # Reconstruct the local-frame hit points as distance * unit ray_dir. Missed rays carry exactly
            # no_hit_value as distance (which may lie below max_range, so an ordering test cannot discriminate) and
            # collapse onto the ray start, matching the (0, 0, 0) stored for them when points are enabled.
            distances = data.distances.reshape((-1, 1))
            hit_points_local = torch.where(
                distances != self._options.no_hit_value, distances * normalize(self.ray_dirs), 0.0
            )
            points = transform_by_trans_quat(hit_points_local + self.ray_starts, pos, quat)

        ray_starts_spheres = context.make_debug_spheres(
            ray_starts, radius=self._options.debug_sphere_radius, color=self._options.debug_ray_start_color
        )
        hit_points_spheres = context.make_debug_spheres(
            points, radius=self._options.debug_sphere_radius, color=self._options.debug_ray_hit_color
        )
        return [(None, ray_starts_spheres, None), (None, hit_points_spheres, None)]
