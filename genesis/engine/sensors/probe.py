import itertools
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.options.sensors.options import ProbesWithNormalSensorOptionsMixin
from genesis.options.sensors.tactile import TactileProbeSensorOptionsMixin
from genesis.typing import EnvsIdxType, LaxFArrayType
from genesis.utils.misc import assign_indexed_tensor, indices_to_mask, tensor_to_array

from .tactile_shared import normalize_grid_probe_layout

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.options.sensors.options import ProbeSensorOptionsMixin
    from genesis.vis.rasterizer_context import RasterizerContext


@qd.func
def func_noised_probe_radius(probe_radius: float, probe_radius_noise: float, eps: float) -> float:
    radius = probe_radius
    if probe_radius_noise > eps:
        radius = qd.max(
            gs.qd_float(0.0),
            probe_radius + (qd.random(gs.qd_float) * gs.qd_float(2.0) - gs.qd_float(1.0)) * probe_radius_noise,
        )
    return radius


def probe_local_pos_of(options: "ProbeSensorOptionsMixin") -> torch.Tensor:
    """
    Return the probe positions of a probe sensor in its link frame as a flat ``(n_probes, 3)`` tensor, whether the
    options lay them out as a list or as a grid.
    """
    return torch.tensor(options.probe_local_pos, dtype=gs.tc_float, device=gs.device).reshape(-1, 3).contiguous()


class ProbeSensorArrayMixin:
    """
    Array of sensors that carry several probes each, laid end to end in fused per-probe tables.

    The probes of every sensor are laid one sensor after the other; ``sensor_probe_start`` gives the first row of each
    sensor. The per-probe parameters of the tactile options (gain, resample range, dead taxels) take their neutral value
    for the sensors whose options do not carry them.
    """

    # How many channel groups a sensor's cache columns hold, in the (group, probe, component) order, for the mapping
    # of the cache columns back to their probe (see cache_col_probe_idx in build)
    _taxel_channel_groups: ClassVar[int] = 1

    def build(self) -> None:
        super().build()

        _B = self._sim._B
        sensors_options = [sensor.options for sensor in self._sensors]
        self._sensors_probe_local_pos = [probe_local_pos_of(sensor_options) for sensor_options in sensors_options]
        self._sensors_probe_layout_shape = [
            np.shape(sensor_options.probe_local_pos)[:-1] for sensor_options in sensors_options
        ]
        # The grid frame of each sensor (see normalize_grid_probe_layout), read by the FFT dilation and the spatial
        # crosstalk; a sensor carrying no per-probe normal takes the plane normal of its grid
        self._sensors_grid_frame = [
            normalize_grid_probe_layout(
                np.asarray(sensor_options.probe_local_pos, dtype=gs.np_float),
                np.asarray(sensor_options.probe_local_normal, dtype=gs.np_float)
                if isinstance(sensor_options, ProbesWithNormalSensorOptionsMixin)
                else None,
                len(layout_shape) == 2,
            )
            for sensor_options, layout_shape in zip(sensors_options, self._sensors_probe_layout_shape)
        ]
        n_probes = [pos.shape[0] for pos in self._sensors_probe_local_pos]
        self.n_probes = n_probes
        self.total_n_probes = sum(n_probes)
        self.n_probes_per_sensor = torch.tensor(n_probes, dtype=gs.tc_int, device=gs.device)
        self.probe_starts = list(itertools.accumulate(n_probes, initial=0))[:-1]
        self.sensor_probe_start = torch.tensor(self.probe_starts, dtype=gs.tc_int, device=gs.device)
        self.probe_sensor_idx = torch.repeat_interleave(
            torch.arange(len(sensors_options), dtype=gs.tc_int, device=gs.device), self.n_probes_per_sensor
        )
        self.probe_positions = torch.cat(self._sensors_probe_local_pos)
        self.probe_radii = torch.cat(
            [
                torch.full((n,), sensor_options.probe_radius, dtype=gs.tc_float, device=gs.device)
                if isinstance(sensor_options.probe_radius, float)
                else torch.tensor(sensor_options.probe_radius, dtype=gs.tc_float, device=gs.device).reshape(n)
                for sensor_options, n in zip(sensors_options, n_probes)
            ]
        )
        self.has_any_probe_radius_noise = any(
            sensor_options.probe_radius_noise > 0.0 for sensor_options in sensors_options
        )

        sensors_is_tactile = [
            isinstance(sensor_options, TactileProbeSensorOptionsMixin) for sensor_options in sensors_options
        ]
        gains = [
            sensor_options.probe_gain if is_tactile_sensor else 1.0
            for sensor_options, is_tactile_sensor in zip(sensors_options, sensors_is_tactile)
        ]
        probe_gains = torch.cat(
            [
                torch.full((n,), float(gain), dtype=gs.tc_float, device=gs.device)
                if isinstance(gain, (int, float))
                else torch.tensor(gain, dtype=gs.tc_float, device=gs.device).reshape(n)
                for gain, n in zip(gains, n_probes)
            ]
        )
        self.probe_gains = torch.stack([probe_gains] * _B)
        resample_ranges = [
            sensor_options.probe_gain_resample_range if is_tactile_sensor else None
            for sensor_options, is_tactile_sensor in zip(sensors_options, sensors_is_tactile)
        ]
        has_resample = [resample_range is not None for resample_range in resample_ranges]
        # The mask tensors live on the torch side only, as conditions for torch.where, which requires torch.bool;
        # gs.tc_bool maps to torch.int32 on Apple Metal for quadrants interop
        self.probe_has_gain_resample = torch.repeat_interleave(
            torch.tensor(has_resample, dtype=torch.bool, device=gs.device), self.n_probes_per_sensor
        )
        self.has_any_gain_resample = any(has_resample)
        # A resampled gain is generally not 1, so the measured branch cannot be assumed equal to the ground truth
        self.has_any_probe_gain = self.has_any_gain_resample or bool(((probe_gains - 1.0).abs() > gs.EPS).any().item())

        # The scalar parameters of each sensor spread over its probes in one pass: radius noise, gain resample range,
        # dead taxel probability and value range
        dead_probabilities = [
            sensor_options.dead_taxel_probability if is_tactile_sensor else 0.0
            for sensor_options, is_tactile_sensor in zip(sensors_options, sensors_is_tactile)
        ]
        dead_ranges = [
            sensor_options.dead_taxel_value_range if is_tactile_sensor else (0.0, 0.0)
            for sensor_options, is_tactile_sensor in zip(sensors_options, sensors_is_tactile)
        ]
        sensors_params = torch.tensor(
            [
                (sensor_options.probe_radius_noise, *((0.0, 0.0) if r is None else r), p, *dead_range)
                for sensor_options, r, p, dead_range in zip(
                    sensors_options, resample_ranges, dead_probabilities, dead_ranges
                )
            ],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # One contiguous row per parameter, as the kernels take them
        probes_params = torch.repeat_interleave(sensors_params, self.n_probes_per_sensor, dim=0).T.contiguous()
        (
            self.probe_radii_noise,
            self.probe_gain_resample_low,
            self.probe_gain_resample_high,
            self.dead_taxel_probability,
            self.dead_taxel_value_low,
            self.dead_taxel_value_high,
        ) = probes_params.unbind(dim=0)

        # The probe of each cache column: a sensor's columns are ordered (group, probe, component), so the probe axis
        # is a strided arange repeated per component and tiled over the groups
        sensors_col_probe_idx = []
        for cache_size, n_p, probe_start in zip(self.cache_sizes, n_probes, self.probe_starts):
            components_per_group = cache_size // (self._taxel_channel_groups * n_p)
            cols = torch.arange(n_p, device=gs.device).repeat_interleave(components_per_group)
            sensors_col_probe_idx.append(cols.repeat(self._taxel_channel_groups) + probe_start)
        self.cache_col_probe_idx = torch.cat(sensors_col_probe_idx)
        self.has_any_dead_taxel = any(probability > 0.0 for probability in dead_probabilities)
        self.dead_taxel_mask = torch.zeros((_B, self.total_n_probes), dtype=torch.bool, device=gs.device)
        self.dead_taxel_values = torch.zeros((_B, self.total_n_probes), dtype=gs.tc_float, device=gs.device)
        # The dead state broadcast to the cache columns, rebuilt on the next `_apply_hardware_imperfections` after a
        # reset resampled it, rather than gathered every step
        self.dead_mask_per_col = self.dead_taxel_mask[:, self.cache_col_probe_idx]
        self.dead_values_per_col = self.dead_taxel_values[:, self.cache_col_probe_idx]
        self.has_stale_dead_columns = False
        self._debug_objects: list[list] = [[] for _ in sensors_options]

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        super().reset(envs_idx)

        # Resample per-(env, probe) gain for probes whose sensor configured a resample range.
        if self.has_any_gain_resample and self.probe_gains.numel() > 0:
            mask = self.probe_has_gain_resample.unsqueeze(0)  # (1, total_n_probes)
            low = self.probe_gain_resample_low.unsqueeze(0)
            high = self.probe_gain_resample_high.unsqueeze(0)
            sub = self.probe_gains[envs_idx]
            new_gain = torch.rand_like(sub) * (high - low) + low
            self.probe_gains[envs_idx] = torch.where(mask, new_gain, sub)
        # Resample dead mask + values per env for affected probes.
        if self.has_any_dead_taxel and self.dead_taxel_mask.numel() > 0:
            prob = self.dead_taxel_probability.unsqueeze(0)  # (1, total_n_probes)
            n_envs = self.dead_taxel_mask[envs_idx].shape[0]
            rolls = torch.rand((n_envs, self.total_n_probes), device=gs.device, dtype=gs.tc_float)
            self.dead_taxel_mask[envs_idx] = rolls < prob
            low = self.dead_taxel_value_low.unsqueeze(0)
            high = self.dead_taxel_value_high.unsqueeze(0)
            uniforms = torch.rand((n_envs, self.total_n_probes), device=gs.device, dtype=gs.tc_float)
            self.dead_taxel_values[envs_idx] = uniforms * (high - low) + low
            # The per-cache-column broadcast is stale until the next `_apply_hardware_imperfections` rebuilds it.
            self.has_stale_dead_columns = True

    def _probe_slice(self, i_s: int) -> slice:
        """The rows of sensor ``i_s`` in the fused per-probe tables."""
        return slice(self.probe_starts[i_s], self.probe_starts[i_s] + self.n_probes[i_s])

    def set_probe_gain(self, i_s: int, value: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        """
        Set the gain of each probe of sensor ``i_s`` on the measured branch, for the given environments.

        ``value`` is a scalar, broadcast to every probe of the sensor, or an array of length ``n_probes``.
        """
        assign_indexed_tensor(
            self.probe_gains, indices_to_mask(envs_idx, self._probe_slice(i_s)), value, ("envs_idx", "probes_idx")
        )
        # Conservatively mark gain in use (a user-set gain may be non-unit); never reset to False.
        self.has_any_probe_gain = True

    def _apply_hardware_imperfections(self, measured_slot_0: torch.Tensor) -> None:
        super()._apply_hardware_imperfections(measured_slot_0)
        if not self.has_any_dead_taxel:
            return
        if self.has_stale_dead_columns:
            self.dead_mask_per_col = self.dead_taxel_mask[:, self.cache_col_probe_idx]
            self.dead_values_per_col = self.dead_taxel_values[:, self.cache_col_probe_idx]
            self.has_stale_dead_columns = False
        torch.where(self.dead_mask_per_col, self.dead_values_per_col, measured_slot_0, out=measured_slot_0)

    def _compute_probes_world_pos(
        self, i_s: int, context: "RasterizerContext"
    ) -> tuple[list[int] | None, int, np.ndarray | None, np.ndarray]:
        """
        Transform the probe positions of sensor ``i_s`` from link-local to world frame for debug drawing.

        Returns ``(envs_idx, n_debug_envs, env_offsets, probe_world_flat)``. ``probe_world_flat`` is ``(n_debug_envs *
        n_probes, 3)`` with env-offset already added.
        """
        link = self._links[i_s]
        probe_local_pos = self._sensors_probe_local_pos[i_s]
        if self._sim.n_envs > 0:
            envs_idx = list(context.rendered_envs_idx)
            n_debug_envs = len(envs_idx)
            env_offsets = context.scene.envs_offset[np.asarray(envs_idx, dtype=gs.np_int)]
            link_pos = link.get_pos(envs_idx, relative=False)[:, None, :]
            link_quat = link.get_quat(envs_idx, relative=False)[:, None, :]
            probe_world = gu.transform_by_trans_quat(probe_local_pos[None, :, :], link_pos, link_quat)
            probe_world = tensor_to_array(probe_world) + env_offsets[:, None, :]
        else:
            envs_idx = None
            n_debug_envs = 1
            env_offsets = None
            link_pos = link.get_pos(envs_idx, relative=False).reshape(3)
            link_quat = link.get_quat(envs_idx, relative=False).reshape(4)
            probe_world = tensor_to_array(gu.transform_by_trans_quat(probe_local_pos, link_pos, link_quat))
        return envs_idx, n_debug_envs, env_offsets, probe_world.reshape(-1, 3)

    def _draw_probe_spheres(
        self,
        i_s: int,
        context: "RasterizerContext",
        probe_world: np.ndarray,
        rgb: tuple[float, float, float],
        probe_radii: np.ndarray | None = None,
        probe_radii_noise: np.ndarray | None = None,
    ) -> list["Mesh"]:
        """
        Draw a small opaque center sphere and a translucent outer sensing sphere at each ``probe_world`` position.

        ``probe_world`` is ``(N, 3)`` (already tiled over rendered envs). ``probe_radii`` and ``probe_radii_noise``
        are the matching ``(N,)`` per-position nominal sensing radius and additive uniform noise; both default to
        the per-probe values of the sensor, tiled to match ``probe_world``. When noise is positive, each outer sphere
        is drawn at a fresh sample ``clip(r + U(-noise, +noise), 0, inf)`` rounded to the nearest ``noise`` magnitude
        so the unique-radius batches stay small. Returns the created debug objects.
        """
        options = self._sensors[i_s].options
        rgb = tuple(float(c) for c in rgb)
        center_color = (*rgb, 1.0)
        objs = [
            context.draw_debug_spheres(
                poss=probe_world,
                radius=float(options.debug_probe_center_radius),
                color=center_color,
            )
        ]
        if options.debug_probe_sphere_opacity <= 0.0:
            return objs
        outer_color = (*rgb, float(options.debug_probe_sphere_opacity))
        probe_slice = self._probe_slice(i_s)
        n_probes = self.n_probes[i_s]
        n_tile = probe_world.shape[0] // n_probes if n_probes > 0 else 0
        n_tile = max(n_tile, 1)
        if probe_radii is None:
            probe_radii = np.tile(tensor_to_array(self.probe_radii[probe_slice]).reshape(-1), n_tile)
        if probe_radii_noise is None:
            probe_radii_noise = np.tile(tensor_to_array(self.probe_radii_noise[probe_slice]).reshape(-1), n_tile)
        nz = probe_radii_noise > 0.0
        if nz.any():
            jitter = np.random.uniform(-1.0, 1.0, size=probe_radii.shape) * probe_radii_noise
            noisy = np.maximum(0.0, probe_radii + jitter)
            rounded = probe_radii.astype(float, copy=True)
            rounded[nz] = np.round(noisy[nz] / probe_radii_noise[nz]) * probe_radii_noise[nz]
            probe_radii = rounded
        for r in np.unique(probe_radii):
            if r <= 0.0:
                continue
            mask = probe_radii == r
            objs.append(
                context.draw_debug_spheres(
                    poss=probe_world[mask],
                    radius=float(r),
                    color=outer_color,
                )
            )
        return objs

    def _draw_debug_probes(
        self,
        i_s: int,
        context: "RasterizerContext",
        color_groups_fn: Callable[[list[int] | None], list[tuple]] | None = None,
    ) -> tuple[list[int] | None, int, np.ndarray | None]:
        """
        Draw the debug markers of the probes of sensor ``i_s``.

        It clears the previous debug objects, then draws the two-sphere marker (a small opaque center and a translucent
        outer sensing sphere) on the probe positions of each color group.

        ``color_groups_fn(envs_idx)`` returns a list of ``(rgb, mask)`` pairs: ``rgb`` is a length-3 sequence and
        ``mask`` a flat ``(n_debug_envs * n_probes,)`` bool array (or a tensor castable to bool) selecting the probe
        positions that take the color. ``None`` draws every probe in the ``debug_probe_color`` of the sensor, which
        suits any probe sensor.

        It returns ``(envs_idx, n_debug_envs, env_offsets)``, so a subclass extends the drawing with more debug geometry
        from the same layout of the environments.
        """
        debug_objects = self._debug_objects[i_s]
        for obj in debug_objects:
            context.clear_debug_object(obj)
        debug_objects.clear()

        envs_idx, n_debug_envs, env_offsets, probe_world = self._compute_probes_world_pos(i_s, context)
        probe_slice = self._probe_slice(i_s)
        n_tile = max(n_debug_envs, 1)
        radii_tiled = np.tile(tensor_to_array(self.probe_radii[probe_slice]).reshape(-1), n_tile)
        noise_tiled = np.tile(tensor_to_array(self.probe_radii_noise[probe_slice]).reshape(-1), n_tile)
        if color_groups_fn is None:
            groups = [(self._sensors[i_s].options.debug_probe_color, np.ones(probe_world.shape[0], dtype=bool))]
        else:
            groups = color_groups_fn(envs_idx)
        for rgb, mask in groups:
            mask_arr = tensor_to_array(mask, dtype=bool).reshape(-1)
            (probes_idx,) = np.nonzero(mask_arr)
            if probes_idx.size == 0:
                continue
            debug_objects.extend(
                self._draw_probe_spheres(
                    i_s, context, probe_world[probes_idx], rgb, radii_tiled[probes_idx], noise_tiled[probes_idx]
                )
            )
        return envs_idx, n_debug_envs, env_offsets

    def _tactile_color_groups_fn(
        self, i_s: int, get_is_contact_flat: Callable[[list[int] | None], object]
    ) -> Callable[[list[int] | None], list[tuple]]:
        """
        Build a ``color_groups_fn`` for the usual tactile split of sensor ``i_s``: the probes out of contact take
        ``debug_probe_color`` and the probes in contact take ``debug_contact_color``.

        The options of the sensor must carry ``debug_contact_color`` (``TactileProbeSensorOptionsMixin``).
        """
        options = self._sensors[i_s].options

        def fn(envs_idx: EnvsIdxType | None) -> list[tuple]:
            is_contact = tensor_to_array(get_is_contact_flat(envs_idx), dtype=bool).reshape(-1)
            return [
                (options.debug_probe_color, ~is_contact),
                (options.debug_contact_color, is_contact),
            ]

        return fn


class ProbesWithNormalSensorArrayMixin(ProbeSensorArrayMixin):
    """Array of probe sensors whose probes also carry a per-probe outward normal in link-local frame."""

    def build(self) -> None:
        super().build()

        normals = []
        for sensor, n_probes in zip(self._sensors, self.n_probes):
            raw_normal = torch.tensor(sensor.options.probe_local_normal, dtype=gs.tc_float, device=gs.device)
            if raw_normal.ndim == 1:
                normals.append(raw_normal.expand(n_probes, 3).contiguous())
            else:
                normals.append(raw_normal.reshape(n_probes, 3).contiguous())
        self.probe_local_normal = torch.cat(normals)


class ProbeSensorMixin:
    """Handle of a sensor carrying several probes: its probe layout and the setter of its probe gains."""

    @property
    def probe_local_pos(self) -> torch.Tensor:
        return probe_local_pos_of(self._options)

    @property
    def n_probes(self) -> int:
        return self.probe_local_pos.shape[0]

    @gs.assert_built
    def set_probe_gain(self, value: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        """
        Set the gain of each probe of this sensor on the measured branch, for the given environments.

        ``value`` is a scalar, broadcast to every probe of the sensor, or an array of length ``n_probes``.
        """
        self._array.set_probe_gain(self._idx, value, envs_idx)
