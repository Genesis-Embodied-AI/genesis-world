from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.render_context import Points, Shape
from genesis.engine.state_change import StateChange, TrackableMixin
from genesis.options.sensors.tactile import TactileProbeSensorOptionsMixin
from genesis.utils.misc import BuildableMixin, concat_with_tensor, make_tensor_field, tensor_to_array

if TYPE_CHECKING:
    from genesis.options.sensors.options import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


@qd.func
def func_noised_probe_radius(probe_radius: float, probe_radius_noise: float, eps: float) -> float:
    radius = probe_radius
    if probe_radius_noise > eps:
        radius = qd.max(
            gs.qd_float(0.0),
            probe_radius + (qd.random(gs.qd_float) * gs.qd_float(2.0) - gs.qd_float(1.0)) * probe_radius_noise,
        )
    return radius


@dataclass
class ProbeSensorMetadataMixin:
    """Shared metadata for sensors that register multiple probes in a fused layout."""

    total_n_probes: int = 0
    probe_positions: torch.Tensor = make_tensor_field((0, 3))
    probe_radii: torch.Tensor = make_tensor_field((0,))
    probe_radii_noise: torch.Tensor = make_tensor_field((0,))
    has_any_probe_radius_noise: bool = False
    has_any_probe_gain: bool = False
    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_probe_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    measured_scratch_T: torch.Tensor = make_tensor_field((0, 0))

    probe_gains: torch.Tensor = make_tensor_field((0, 0))
    probe_gain_resample_low: torch.Tensor = make_tensor_field((0,))
    probe_gain_resample_high: torch.Tensor = make_tensor_field((0,))
    # The mask tensors below live on the torch side only, as conditions for torch.where, which requires torch.bool;
    # gs.tc_bool maps to torch.int32 on Apple Metal for quadrants interop.
    probe_has_gain_resample: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.bool)
    any_gain_resample: bool = False

    dead_taxel_mask: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: torch.bool)
    dead_taxel_values: torch.Tensor = make_tensor_field((0, 0))
    dead_taxel_probability: torch.Tensor = make_tensor_field((0,))
    dead_taxel_value_low: torch.Tensor = make_tensor_field((0,))
    dead_taxel_value_high: torch.Tensor = make_tensor_field((0,))
    any_dead_taxel: bool = False
    dead_mask_per_col: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: torch.bool)
    dead_values_per_col: torch.Tensor = make_tensor_field((0, 0))
    dead_dirty: bool = True
    cache_col_probe_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.long)
    cache_col_n_channel_groups: list[int] = field(default_factory=list)


def get_measured_bufs(
    shared_metadata: "ProbeSensorMetadataMixin",
    current_ground_truth_data_T: torch.Tensor,
    measured_data_timeline: "TensorRingBuffer",
) -> tuple[torch.Tensor, torch.Tensor]:
    current_ground_truth_data_T.zero_()
    measured_slot = measured_data_timeline.at(0, copy=False)
    measured_slot.zero_()
    if shared_metadata.measured_scratch_T.shape != current_ground_truth_data_T.shape:
        shared_metadata.measured_scratch_T = torch.empty_like(current_ground_truth_data_T)
    return measured_slot, shared_metadata.measured_scratch_T


class ProbeSensorMixin[ProbeSensorSharedMetadataT: ProbeSensorMetadataMixin](BuildableMixin):
    """Shared logic for registering this sensor's probes in ``ProbeSensorMetadataMixin`` fields."""

    # Number of channel groups per probe in the cache layout. Used by the per-cache-col probe-index builder.
    _taxel_channel_groups: int = 1

    def __init__(
        self,
        options: "SensorOptions",
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        # `_get_return_format` runs inside `super().__init__`, so the probe layout fields must be set first.
        raw_pos = torch.tensor(options.probe_local_pos, dtype=gs.tc_float, device=gs.device)
        self._probe_layout_shape = raw_pos.shape[:-1]
        self._n_probes = int(np.prod(self._probe_layout_shape))
        self._probe_local_pos = raw_pos.reshape(self._n_probes, 3).contiguous()
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._probe_start_idx = -1
        # The bounds between the radii the outer spheres of the probes are drawn with, and whether each radius is drawn
        self._debug_radii_bounds: torch.Tensor | None = None
        self._debug_radii_is_drawn: list[bool] = []

    @gs.assert_unbuilt
    def build(self) -> None:
        super().build()
        self._probe_start_idx = self._shared_metadata.total_n_probes
        self._shared_metadata.sensor_probe_start = concat_with_tensor(
            self._shared_metadata.sensor_probe_start, self._shared_metadata.total_n_probes, expand=(1,)
        )
        self._shared_metadata.total_n_probes += self._n_probes
        self._shared_metadata.n_probes_per_sensor = concat_with_tensor(
            self._shared_metadata.n_probes_per_sensor, self._n_probes, expand=(1,)
        )
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start,
            sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0,
            expand=(1,),
        )
        self._shared_metadata.probe_sensor_idx = concat_with_tensor(
            self._shared_metadata.probe_sensor_idx,
            torch.full((self._n_probes,), self._idx, dtype=gs.tc_int, device=gs.device),
            expand=(self._n_probes,),
        )
        self._shared_metadata.probe_positions = concat_with_tensor(
            self._shared_metadata.probe_positions, self._probe_local_pos, expand=(self._n_probes, 3)
        )
        if isinstance(self._options.probe_radius, float):
            probe_radii = torch.full((self._n_probes,), self._options.probe_radius, dtype=gs.tc_float, device=gs.device)
        else:
            probe_radii = torch.tensor(self._options.probe_radius, dtype=gs.tc_float, device=gs.device).reshape(
                self._n_probes
            )
        self._shared_metadata.probe_radii = concat_with_tensor(
            self._shared_metadata.probe_radii, probe_radii, expand=(self._n_probes,)
        )
        self._shared_metadata.probe_radii_noise = concat_with_tensor(
            self._shared_metadata.probe_radii_noise,
            torch.full((self._n_probes,), self._options.probe_radius_noise, dtype=gs.tc_float, device=gs.device),
            expand=(self._n_probes,),
        )
        if self._options.probe_radius_noise > 0.0:
            self._shared_metadata.has_any_probe_radius_noise = True

        # Tactile-specific options (probe_gain, dead_taxel_*) live on ``TactileProbeSensorOptionsMixin``; generic
        # probe sensors (e.g. SurfaceDistanceProbe) don't carry them and register defaults (gain 1, no dead).
        B = self._manager.n_envs_
        opts = self._options
        is_tactile = isinstance(opts, TactileProbeSensorOptionsMixin)
        # Initial per-probe gain (probe_gain may be scalar or per-probe array).
        gain_value = opts.probe_gain if is_tactile else 1.0
        if isinstance(gain_value, (int, float)):
            init_gain = torch.full((B, self._n_probes), float(gain_value), dtype=gs.tc_float, device=gs.device)
            if float(gain_value) != 1.0:
                self._shared_metadata.has_any_probe_gain = True
        else:
            init_gain = (
                torch.tensor(gain_value, dtype=gs.tc_float, device=gs.device)
                .reshape(self._n_probes)
                .unsqueeze(0)
                .expand(B, self._n_probes)
                .contiguous()
            )
            if not bool((init_gain == 1.0).all().item()):
                self._shared_metadata.has_any_probe_gain = True
        self._shared_metadata.probe_gains = concat_with_tensor(
            self._shared_metadata.probe_gains, init_gain, expand=(B, self._n_probes), dim=1
        )

        # Per-probe gain resample range (constant across envs). When option is None, write zeros + has_resample=False;
        # the reset hook gates on ``has_gain_resample`` per probe.
        resample_range = opts.probe_gain_resample_range if is_tactile else None
        if resample_range is None:
            low, high = 0.0, 0.0
            has_resample = False
        else:
            low, high = float(resample_range[0]), float(resample_range[1])
            has_resample = True
            self._shared_metadata.any_gain_resample = True
            # Resampled gain is generally != 1, so the measured branch can't be assumed equal to GT.
            self._shared_metadata.has_any_probe_gain = True
        self._shared_metadata.probe_gain_resample_low = concat_with_tensor(
            self._shared_metadata.probe_gain_resample_low,
            torch.full((self._n_probes,), low, dtype=gs.tc_float, device=gs.device),
            expand=(self._n_probes,),
        )
        self._shared_metadata.probe_gain_resample_high = concat_with_tensor(
            self._shared_metadata.probe_gain_resample_high,
            torch.full((self._n_probes,), high, dtype=gs.tc_float, device=gs.device),
            expand=(self._n_probes,),
        )
        self._shared_metadata.probe_has_gain_resample = concat_with_tensor(
            self._shared_metadata.probe_has_gain_resample,
            torch.full((self._n_probes,), has_resample, dtype=torch.bool, device=gs.device),
            expand=(self._n_probes,),
        )

        # Per-probe dead taxel configuration (constant across envs).
        dead_prob = float(opts.dead_taxel_probability) if is_tactile else 0.0
        dead_range = opts.dead_taxel_value_range if is_tactile else (0.0, 0.0)
        s_low, s_high = float(dead_range[0]), float(dead_range[1])
        self._shared_metadata.dead_taxel_probability = concat_with_tensor(
            self._shared_metadata.dead_taxel_probability,
            torch.full((self._n_probes,), dead_prob, dtype=gs.tc_float, device=gs.device),
            expand=(self._n_probes,),
        )
        self._shared_metadata.dead_taxel_value_low = concat_with_tensor(
            self._shared_metadata.dead_taxel_value_low,
            torch.full((self._n_probes,), s_low, dtype=gs.tc_float, device=gs.device),
            expand=(self._n_probes,),
        )
        self._shared_metadata.dead_taxel_value_high = concat_with_tensor(
            self._shared_metadata.dead_taxel_value_high,
            torch.full((self._n_probes,), s_high, dtype=gs.tc_float, device=gs.device),
            expand=(self._n_probes,),
        )
        if dead_prob > 0.0:
            self._shared_metadata.any_dead_taxel = True
        self._shared_metadata.dead_taxel_mask = torch.zeros(
            (B, self._shared_metadata.total_n_probes), dtype=torch.bool, device=gs.device
        )
        self._shared_metadata.dead_taxel_values = torch.zeros(
            (B, self._shared_metadata.total_n_probes), dtype=gs.tc_float, device=gs.device
        )
        # Invalidate the lazy cache-col probe index; rebuilt on next dead apply.
        self._shared_metadata.cache_col_probe_idx = torch.empty((0,), dtype=torch.long, device=gs.device)
        self._shared_metadata.cache_col_n_channel_groups.append(self._taxel_channel_groups)

    @classmethod
    def reset(cls, shared_metadata, shared_ground_truth_cache, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        # Resample per-(env, probe) gain for probes whose sensor configured a resample range.
        if shared_metadata.any_gain_resample and shared_metadata.probe_gains.numel() > 0:
            mask = shared_metadata.probe_has_gain_resample.unsqueeze(0)  # (1, total_n_probes)
            low = shared_metadata.probe_gain_resample_low.unsqueeze(0)
            high = shared_metadata.probe_gain_resample_high.unsqueeze(0)
            sub = shared_metadata.probe_gains[envs_idx]
            new_gain = torch.rand_like(sub) * (high - low) + low
            shared_metadata.probe_gains[envs_idx] = torch.where(mask, new_gain, sub)
        # Resample dead mask + values per env for affected probes.
        if shared_metadata.any_dead_taxel and shared_metadata.dead_taxel_mask.numel() > 0:
            prob = shared_metadata.dead_taxel_probability.unsqueeze(0)  # (1, total_n_probes)
            n_envs = shared_metadata.dead_taxel_mask[envs_idx].shape[0]
            rolls = torch.rand((n_envs, shared_metadata.total_n_probes), device=gs.device, dtype=gs.tc_float)
            new_mask = rolls < prob
            shared_metadata.dead_taxel_mask[envs_idx] = new_mask
            low = shared_metadata.dead_taxel_value_low.unsqueeze(0)
            high = shared_metadata.dead_taxel_value_high.unsqueeze(0)
            uniforms = torch.rand((n_envs, shared_metadata.total_n_probes), device=gs.device, dtype=gs.tc_float)
            shared_metadata.dead_taxel_values[envs_idx] = uniforms * (high - low) + low
            # The per-cache-column broadcast is now stale; rebuilt on the next `_apply_hardware_imperfections`.
            shared_metadata.dead_dirty = True

    @gs.assert_built
    def set_probe_gain(self, value, envs_idx=None):
        """Set the per-probe measured-branch contact-depth gain for the given envs.

        ``value`` may be a scalar (broadcast to all probes of this sensor), or an array of length ``n_probes``.
        Affects only the probes registered by this sensor instance.
        """
        envs_idx = self._sanitize_envs_idx(envs_idx)
        probe_start = int(self._shared_metadata.sensor_probe_start[self._idx].item())
        probe_slice = slice(probe_start, probe_start + self._n_probes)
        if isinstance(value, (int, float)):
            row = torch.full((len(envs_idx), self._n_probes), float(value), dtype=gs.tc_float, device=gs.device)
        else:
            t = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device).reshape(-1)
            if t.numel() != self._n_probes:
                gs.raise_exception(f"set_probe_gain expected {self._n_probes} values, got {t.numel()}.")
            row = t.unsqueeze(0).expand(len(envs_idx), self._n_probes).contiguous()
        self._shared_metadata.probe_gains[envs_idx, probe_slice] = row
        # Conservatively mark gain in use (a user-set gain may be non-unit); never reset to False.
        self._shared_metadata.has_any_probe_gain = True

    @classmethod
    def _apply_hardware_imperfections(cls, shared_metadata, measured_slot_0):
        super()._apply_hardware_imperfections(shared_metadata, measured_slot_0)
        if not shared_metadata.any_dead_taxel:
            return
        cls._maybe_build_cache_col_probe_idx(shared_metadata, measured_slot_0)
        # The per-(env, probe) dead state only changes on reset; broadcast it to per-(env, cache_col) layout once
        # (when dirty) instead of gathering every step.
        if shared_metadata.dead_dirty or shared_metadata.dead_mask_per_col.shape != measured_slot_0.shape:
            idx = shared_metadata.cache_col_probe_idx  # (total_cache_size,)
            shared_metadata.dead_mask_per_col = shared_metadata.dead_taxel_mask[:, idx]
            shared_metadata.dead_values_per_col = shared_metadata.dead_taxel_values[:, idx].to(
                dtype=measured_slot_0.dtype
            )
            shared_metadata.dead_dirty = False
        torch.where(
            shared_metadata.dead_mask_per_col,
            shared_metadata.dead_values_per_col,
            measured_slot_0,
            out=measured_slot_0,
        )

    @classmethod
    def _maybe_build_cache_col_probe_idx(cls, shared_metadata, tensor):
        n_cols = tensor.shape[1]
        if shared_metadata.cache_col_probe_idx.shape == (n_cols,):
            return
        sizes = shared_metadata.cache_sizes
        n_probes_per = shared_metadata.n_probes_per_sensor.tolist()
        probe_starts = shared_metadata.sensor_probe_start.tolist()
        groups = shared_metadata.cache_col_n_channel_groups
        # Each sensor's cache columns are ordered (group, probe, component); only the probe axis indexes a probe,
        # so its slice is a strided arange: arange(n_p) repeated per-component, tiled over the k groups.
        per_sensor = []
        for i_s, cache_size in enumerate(sizes):
            n_p = n_probes_per[i_s]
            if n_p == 0:
                continue
            k = groups[i_s] if i_s < len(groups) else 1
            components_per_group = cache_size // (k * n_p)
            cols = torch.arange(n_p, dtype=torch.long, device=gs.device)
            cols = cols.repeat_interleave(components_per_group).repeat(k)
            per_sensor.append(cols + probe_starts[i_s])
        shared_metadata.cache_col_probe_idx = (
            torch.cat(per_sensor) if per_sensor else torch.empty((0,), dtype=torch.long, device=gs.device)
        )

    @property
    def probe_local_pos(self) -> torch.Tensor:
        return self._probe_local_pos

    @property
    def n_probes(self) -> int:
        return self._n_probes

    def _update_probes_markers(
        self,
        markers: tuple[Shape, ...],
        envs_idx: torch.Tensor | None,
        is_contact: torch.Tensor | None,
        invalidated_by: dict[TrackableMixin, tuple[StateChange, ...]],
    ) -> tuple[Points, ...]:
        """Return the markers of the probes, written in the environments `envs_idx` (see 'Sensor._update_markers').

        `markers` are those of the probes the latest call returned. `is_contact` tells whether each probe of a tactile
        sensor is in contact, laid out as the readings of the sensor, whose last entry of history is drawn. It is None
        for any other sensor. `invalidated_by` states the changes the markers follow (see 'Shape.invalidated_by').
        """
        options = self._options
        probes_radii = self._shared_metadata.probe_radii[self._probe_start_idx : self._probe_start_idx + self._n_probes]
        if not markers:
            # A small opaque center sphere at each probe, those of a tactile sensor in contact taking its contact color
            groups_rgb = (options.debug_probe_color,)
            if isinstance(options, TactileProbeSensorOptionsMixin):
                groups_rgb += (options.debug_contact_color,)
            styles = [(options.debug_probe_center_radius, (*rgb, 1.0)) for rgb in groups_rgb]
            if options.debug_probe_sphere_opacity > 0.0:
                # A translucent outer sphere at each probe, of its sensing radius, or of a sample of its noised radius
                # at each refresh rounded to the nearest multiple of the noise, so that four multiples around the
                # nominal radius cover every sample, one marker per radius.
                radii, noise = probes_radii, options.probe_radius_noise
                if noise > 0.0:
                    multiples = torch.floor((radii - noise).clamp(min=0.0) / noise)
                    radii = (multiples[:, None] + torch.arange(4, dtype=gs.tc_float, device=gs.device)) * noise
                radii = torch.unique(radii)
                self._debug_radii_bounds = 0.5 * (radii[1:] + radii[:-1])
                self._debug_radii_is_drawn = (radii > 0.0).tolist()
                styles += [
                    (radius, (*rgb, options.debug_probe_sphere_opacity))
                    for rgb in groups_rgb
                    for radius in radii[radii > 0.0].tolist()
                ]
            # Every marker draws a sphere at each probe, showing those of its group alone
            pos = torch.zeros((self._manager.n_envs_, self._n_probes, 3), dtype=gs.tc_float, device=gs.device)
            styles_is_active = torch.zeros((len(styles), *pos.shape[:-1]), dtype=gs.tc_bool, device=gs.device)
            markers = tuple(
                Points(pos, radius, color, is_active=is_active, invalidated_by=invalidated_by)
                for (radius, color), is_active in zip(styles, styles_is_active)
            )

        if self._link is None:
            # A sensor attached to no entity sits in the world frame, its probes at their local position, with the
            # environments a link pose would hold.
            n_envs = self._manager.n_envs
            envs_shape = (len(envs_idx),) if envs_idx is not None else ((n_envs,) if n_envs > 0 else ())
            probes_pos = self._probe_local_pos.expand((*envs_shape, *self._probe_local_pos.shape))
        else:
            link_pos = self._link.get_pos(envs_idx, relative=False)[..., None, :]
            link_quat = self._link.get_quat(envs_idx, relative=False)[..., None, :]
            probes_pos = gu.transform_by_trans_quat(self._probe_local_pos, link_pos, link_quat)
        if is_contact is None:
            groups_active = torch.ones((1, *probes_pos.shape[:-1]), dtype=gs.tc_bool, device=gs.device)
        else:
            is_contact = is_contact.reshape((*probes_pos.shape[:-2], -1, self._n_probes))[..., -1, :]
            groups_active = torch.stack((torch.logical_not(is_contact), is_contact))
        markers_is_active = list(groups_active)
        if self._debug_radii_bounds is not None:
            radii, noise = probes_radii, options.probe_radius_noise
            if noise > 0.0:
                # The samples are drawn from the generator of numpy, group after group over the probes of each
                # environment in turn, so that a seeded scene draws the same radii.
                jitter = np.zeros(probes_pos.shape[:-1])
                for group_active in tensor_to_array(groups_active):
                    jitter[group_active] = np.random.uniform(-1.0, 1.0, size=int(group_active.sum()))
                jitter = torch.as_tensor(jitter, dtype=gs.tc_float, device=gs.device)
                radii = torch.round((radii + jitter * noise).clamp(min=0.0) / noise) * noise
            radii_idx = torch.bucketize(radii, self._debug_radii_bounds)
            for group_active in groups_active:
                for i_r, is_drawn in enumerate(self._debug_radii_is_drawn):
                    if is_drawn:
                        markers_is_active.append(group_active & (radii_idx == i_r))
        # The markers share the positions of the probes
        pos = markers[0].pos
        pos[envs_idx] = probes_pos
        for marker, marker_is_active in zip(markers, markers_is_active):
            is_active = marker.is_active
            is_active[envs_idx] = marker_is_active
        return markers


@dataclass
class ProbesWithNormalSensorMetadataMixin(ProbeSensorMetadataMixin):
    """Shared metadata for probe sensors that also carry a per-probe outward normal."""

    probe_local_normal: torch.Tensor = make_tensor_field((0, 3))


class ProbesWithNormalSensorMixin[ProbesWithNormalSensorSharedMetadataT: ProbesWithNormalSensorMetadataMixin](
    ProbeSensorMixin[ProbesWithNormalSensorSharedMetadataT]
):
    """Probe sensor whose probes carry a per-probe outward normal in link-local frame."""

    def __init__(
        self,
        options: "SensorOptions",
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        raw_normal = torch.tensor(self._options.probe_local_normal, dtype=gs.tc_float, device=gs.device)
        if raw_normal.ndim == 1:
            self._probe_local_normal = raw_normal.expand(self._n_probes, 3).contiguous()
        else:
            self._probe_local_normal = raw_normal.reshape(self._n_probes, 3).contiguous()

    @gs.assert_unbuilt
    def build(self) -> None:
        super().build()
        self._shared_metadata.probe_local_normal = concat_with_tensor(
            self._shared_metadata.probe_local_normal, self._probe_local_normal, expand=(self._n_probes, 3)
        )

    @property
    def probe_local_normal(self) -> torch.Tensor:
        return self._probe_local_normal
