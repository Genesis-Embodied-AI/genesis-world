import importlib
import pkgutil
import sys
from typing import TYPE_CHECKING, ForwardRef, get_args, get_origin

import torch

import genesis as gs
from genesis.options.sensors import types as _sensor_types_namespace
from genesis.options.sensors.options import SensorOptions
from genesis.utils.ring_buffer import TensorRingBuffer

from .base_sensor import Sensor, SharedSensorContext, SharedSensorMetadata

if TYPE_CHECKING:
    from genesis.vis.rasterizer_context import RasterizerContext


class SensorManager:
    # Maps sensor options class -> sensor class for runtime dispatch.
    SENSOR_TYPES_MAP: dict[type[SensorOptions], type["Sensor"]] = {}

    def __init__(self, sim):
        self._sim = sim
        self._sensors_by_type: dict[type["Sensor"], list["Sensor"]] = {}
        self._sensors_metadata: dict[type["Sensor"], SharedSensorMetadata | None] = {}
        # Cross-type shared contexts, keyed by context class so every sensor type declaring the same context resolves
        # to one instance. Built/updated/reset/destroyed by this manager; see ``SharedSensorContext``.
        self._shared_contexts: dict[type, SharedSensorContext] = {}
        # Per-dtype intermediate caches: pre-`_post_process` storage in intermediate space. The transposed GT cache is
        # `(cols, B)` for C-contiguous per-class row slices required by kernel writes.
        self._ground_truth_intermediate_cache: dict[type[torch.dtype], torch.Tensor] = {}
        self._intermediate_cache: dict[type[torch.dtype], torch.Tensor] = {}
        # Per-class return caches in return space - what `read()` and `read_ground_truth()` slice into. Separate buffers
        # when a per-class return-space ring is allocated (the orchestrator delay-samples the ring into the cache);
        # alias-views into the per-dtype intermediate cache otherwise (identity `_post_process`, no delay, no history -
        # the per-step write inside `_update_shared_cache` is then directly visible to `read()`).
        self._return_cache: dict[type["Sensor"], torch.Tensor] = {}
        self._ground_truth_return_cache: dict[type["Sensor"], torch.Tensor] = {}
        # Paired GT and measured timeline rings (post-transform, PRE-hardware-imperfections data). Allocated together
        # per dtype when any sensor in the dtype declares `uses_ring_pipeline = True`. They share the same rotation idx
        # so a single `rotate()` per step advances both.
        self._ground_truth_timeline_ring: dict[type[torch.dtype], TensorRingBuffer] = {}
        self._measured_timeline_ring: dict[type[torch.dtype], TensorRingBuffer] = {}
        # Per-class return-space rings (post-everything: post-hardware-imperfections, post-`_post_process`,
        # pre-delay-sample). Allocated together when any sensor in the class has `delay > 0`, OR `history_length > 0`,
        # OR the class overrides `_post_process`. Each step the post-everything snapshot is written to slot 0; the
        # source for delay sampling (into the per-class return cache) and for history reads. GT and measured rings share
        # their rotation idx so a single `rotate()` per step advances both.
        self._ground_truth_return_timeline_ring: dict[type["Sensor"], TensorRingBuffer] = {}
        self._measured_return_timeline_ring: dict[type["Sensor"], TensorRingBuffer] = {}
        # Per-class precomputed history index tensor [0, 1, ..., max_history-1]. Used to fancy-index the rings on
        # history reads.
        self._hist_idx_by_class: dict[type["Sensor"], torch.Tensor] = {}
        self._cache_slices_by_type: dict[type["Sensor"], slice] = {}
        # (sensor class, entity_idx) -> slice within the class cache. entity_idx == -1 means static sensors.
        self._entity_slice_in_class: dict[type["Sensor"], dict[int, slice]] = {}
        self._max_history_by_class: dict[type["Sensor"], int] = {}

    def create_sensor(self, sensor_options: "SensorOptions") -> "Sensor":
        sensor_options.validate_scene(self._sim.scene)
        sensor_cls = SensorManager._resolve_sensor_cls(type(sensor_options))
        self._sensors_by_type.setdefault(sensor_cls, [])
        if sensor_cls not in self._sensors_metadata:
            self._sensors_metadata[sensor_cls] = sensor_cls._metadata_cls()
        # Create the shared context before the sensor, so the instance exists to hand to it. ``NoneType`` marks
        # "no context"; the sensor then receives ``None``.
        context_cls = sensor_cls._shared_context_cls
        if context_cls is not type(None) and context_cls not in self._shared_contexts:
            self._shared_contexts[context_cls] = context_cls(self._sim)
        sensor = sensor_cls(
            sensor_options,
            len(self._sensors_by_type[sensor_cls]),
            self._shared_contexts.get(context_cls),
            self._sensors_metadata[sensor_cls],
            self,
        )
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    @staticmethod
    def _resolve_sensor_cls(options_cls: type) -> type["Sensor"]:
        """Resolve the sensor class for the given options class, triggering lazy discovery if needed."""
        sensor_cls = SensorManager.SENSOR_TYPES_MAP.get(options_cls)
        if sensor_cls is not None:
            return sensor_cls

        # Not registered yet — check that the options class specifies its sensor type, then try to discover it. The
        # sensor class name is extracted from the generic metadata on the options class bases.
        is_parameterized = False
        for base in options_cls.__bases__:
            meta = base.__pydantic_generic_metadata__
            if meta["origin"] is not None and issubclass(meta["origin"], SensorOptions):
                is_parameterized = bool(meta["args"]) and isinstance(meta["args"][0], str)
                break
        # Fallback: typing introspection on __orig_bases__ (for pydantic versions that flatten bases)
        if not is_parameterized:
            for base in options_cls.__orig_bases__:
                origin = get_origin(base)
                if origin is not None and issubclass(origin, SensorOptions):
                    args = get_args(base)
                    is_parameterized = bool(args) and isinstance(args[0], (str, ForwardRef))
                    break

        if not is_parameterized:
            gs.raise_exception(
                f"{options_cls.__name__} must parameterize its SensorOptions base with a sensor class, "
                f"e.g. `class {options_cls.__name__}(SensorOptions['MySensor']): ...`"
            )

        # Try to discover the sensor module from sibling modules of the options package.
        options_module = options_cls.__module__
        if "." in options_module:
            pkg_name = options_module.rsplit(".", 1)[0]
            pkg = sys.modules.get(pkg_name)
            if pkg is not None:
                pkg_path = pkg.__dict__.get("__path__")
                if pkg_path is not None:
                    for _, modname, _ in pkgutil.iter_modules(pkg_path, pkg.__name__ + "."):
                        if modname not in sys.modules:
                            try:
                                importlib.import_module(modname)
                            except Exception:
                                continue
                        if options_cls in SensorManager.SENSOR_TYPES_MAP:
                            return SensorManager.SENSOR_TYPES_MAP[options_cls]

        gs.raise_exception(
            f"No sensor class registered for {options_cls.__name__}. Ensure the sensor module is in the same "
            "package as the options module, or import the sensor class manually before calling add_sensor()."
        )

    def build(self):
        # Sort each class by entity_idx so sensors attached to the same entity occupy a contiguous slice of the class
        # cache. Static sensors have entity_idx=-1 and group together. Python's sort is stable, so registration order is
        # preserved within each entity bucket.
        for sensors in self._sensors_by_type.values():
            sensors.sort(key=lambda s: s._options.entity_idx)
            for new_idx, sensor in enumerate(sensors):
                sensor._idx = new_idx

        # Per-class intermediate / return dtypes come from `_get_intermediate_dtype` / `_get_cache_dtype`. Dtype is
        # class-uniform by design (the per-class slice into the per-dtype intermediate buffer must be contiguous, so all
        # instances of a class share one dtype). Shape is per-instance via `_get_intermediate_format` /
        # `_get_return_format` and contributes to the class slice size below.
        cache_size_per_dtype: dict[torch.dtype, int] = {}
        max_history_per_dtype: dict[torch.dtype, int] = {}
        intermediate_dtype_by_class: dict[type["Sensor"], torch.dtype] = {}
        return_dtype_by_class: dict[type["Sensor"], torch.dtype] = {}
        # Deepest return-space slot delay sampling can reach, plus one.
        delay_depth_by_class: dict[type["Sensor"], int] = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            intermediate_dtype = sensor_cls._get_intermediate_dtype()
            return_dtype = sensor_cls._get_cache_dtype()
            intermediate_dtype_by_class[sensor_cls] = intermediate_dtype
            return_dtype_by_class[sensor_cls] = return_dtype

            cache_size_per_dtype.setdefault(intermediate_dtype, 0)
            cls_cache_start_idx = cache_size_per_dtype[intermediate_dtype]
            entity_offsets: dict[int, list[int]] = {}
            cls_offset = 0
            cls_max_history = 0
            cls_delay_depth = 1
            for sensor in sensors:
                sensor._cache_offset = cls_offset
                cache_size_per_dtype[intermediate_dtype] += sensor._cache_size
                # A delay reserves one slot past `_delay_ts` for the jitter shift, which `set_jitter` can raise at any
                # time; without that slot, `at()` wraps modulo the depth and returns the newest frame as the oldest.
                cls_delay_depth = max(cls_delay_depth, sensor._delay_ts + (2 if sensor._options.delay > 0.0 else 1))
                hist = sensor._options.history_length
                if hist > 0:
                    max_history_per_dtype[intermediate_dtype] = max(
                        max_history_per_dtype.get(intermediate_dtype, 0), hist
                    )
                    cls_max_history = max(cls_max_history, hist)
                eid = sensor._options.entity_idx
                if eid in entity_offsets:
                    entity_offsets[eid][1] = cls_offset + sensor._cache_size
                else:
                    entity_offsets[eid] = [cls_offset, cls_offset + sensor._cache_size]
                cls_offset += sensor._cache_size

            cls_cache_end_idx = cache_size_per_dtype[intermediate_dtype]
            self._cache_slices_by_type[sensor_cls] = slice(cls_cache_start_idx, cls_cache_end_idx)
            self._entity_slice_in_class[sensor_cls] = {
                eid: slice(start, stop) for eid, (start, stop) in entity_offsets.items()
            }
            self._max_history_by_class[sensor_cls] = cls_max_history
            delay_depth_by_class[sensor_cls] = cls_delay_depth

        self._ground_truth_timeline_ring.clear()
        self._measured_timeline_ring.clear()
        self._return_cache.clear()
        self._ground_truth_return_cache.clear()
        self._ground_truth_return_timeline_ring.clear()
        self._measured_return_timeline_ring.clear()
        self._hist_idx_by_class.clear()

        # Per-dtype flag: at least one class in this dtype uses the ring-based per-step pipeline. Drives allocation of
        # the paired GT + measured timeline rings.
        dtype_uses_rings: dict[torch.dtype, bool] = {}
        for sensor_cls in self._sensors_by_type:
            dtype = intermediate_dtype_by_class[sensor_cls]
            dtype_uses_rings[dtype] = dtype_uses_rings.get(dtype, False) or sensor_cls.uses_ring_pipeline

        for dtype, total_cols in cache_size_per_dtype.items():
            cache_shape = (self._sim._B, total_cols)
            # Ground truth cache is stored transposed (cols, B) so that per-class row slices are C-contiguous, which is
            # required for kernel writes. The cache and ring buffer stay (B, cols) since they only receive data via
            # .copy_() / torch.lerp which handle non-contiguous targets.
            gt_cache_shape = (total_cols, self._sim._B)
            self._ground_truth_intermediate_cache[dtype] = torch.zeros(gt_cache_shape, dtype=dtype, device=gs.device)
            self._intermediate_cache[dtype] = torch.zeros(cache_shape, dtype=dtype, device=gs.device)
            if dtype_uses_rings[dtype]:
                # Timeline rings serve `_apply_transform` recurrence. Two slots cover the canonical one-step recurrence;
                # the ring is grown to `max_history` when any sensor in the dtype requests history, so a multi-tap
                # stateful filter inside `_apply_transform` can read deeper without keeping its own state.
                ring_n = max(2, max_history_per_dtype.get(dtype, 0))
                self._measured_timeline_ring[dtype] = TensorRingBuffer(ring_n, cache_shape, dtype=dtype)
                self._ground_truth_timeline_ring[dtype] = TensorRingBuffer(
                    ring_n, cache_shape, dtype=dtype, idx=self._measured_timeline_ring[dtype]._idx
                )

        # Per-class return-space caches + rings. The return-space ring is the single per-class buffer that records each
        # step's post-`_post_process` snapshot; it is the source for delay sampling and history reads, and provides the
        # `timeline` argument that stateful `_post_process` overrides see. Allocated whenever any sensor in the class
        # has delay > 0, OR history > 0, OR the class overrides `_post_process`. Sized to fit the deepest demand. When
        # no return-space ring is needed (no delay, no history, identity `_post_process`), the return cache is a
        # zero-copy alias-view of the intermediate cache so per-step writes propagate without extra work.
        for sensor_cls, sensors in self._sensors_by_type.items():
            intermediate_dtype = intermediate_dtype_by_class[sensor_cls]
            return_dtype = return_dtype_by_class[sensor_cls]
            cls_slice = self._cache_slices_by_type[sensor_cls]
            cls_size = cls_slice.stop - cls_slice.start
            cls_max_history = self._max_history_by_class[sensor_cls]
            cls_delay_depth = delay_depth_by_class[sensor_cls]
            pp_overridden = sensor_cls._post_process.__func__ is not Sensor._post_process.__func__
            needs_ring = cls_delay_depth > 1 or cls_max_history > 0 or pp_overridden
            if needs_ring:
                ring_n = max(cls_delay_depth, cls_max_history, 2 if pp_overridden else 1)
                ring_shape = (self._sim._B, cls_size)
                self._ground_truth_return_timeline_ring[sensor_cls] = TensorRingBuffer(
                    ring_n, ring_shape, dtype=return_dtype
                )
                self._measured_return_timeline_ring[sensor_cls] = TensorRingBuffer(
                    ring_n, ring_shape, dtype=return_dtype, idx=self._ground_truth_return_timeline_ring[sensor_cls]._idx
                )
                self._return_cache[sensor_cls] = torch.zeros(
                    (self._sim._B, cls_size), dtype=return_dtype, device=gs.device
                )
                self._ground_truth_return_cache[sensor_cls] = torch.zeros(
                    (self._sim._B, cls_size), dtype=return_dtype, device=gs.device
                )
            else:
                self._return_cache[sensor_cls] = self._intermediate_cache[intermediate_dtype][:, cls_slice]
                self._ground_truth_return_cache[sensor_cls] = self._ground_truth_intermediate_cache[intermediate_dtype][
                    cls_slice, :
                ].T
            if cls_max_history > 0:
                self._hist_idx_by_class[sensor_cls] = torch.arange(cls_max_history, device=gs.device, dtype=torch.int32)

        for sensor_cls, sensors in self._sensors_by_type.items():
            for sensor in sensors:
                sensor.build()
                sensor._is_built = True

    def destroy(self):
        for context in self._shared_contexts.values():
            context.destroy()
        self._shared_contexts.clear()
        for sensors_metadata in self._sensors_metadata.values():
            if sensors_metadata is not None:
                sensors_metadata.destroy()
        self._sensors_metadata.clear()
        self._sensors_by_type.clear()

    def reset(self, envs_idx=None):
        if not self._sensors_by_type:
            return

        envs_idx = self._sim._scene._sanitize_envs_idx(envs_idx)

        for dtype in self._ground_truth_intermediate_cache.keys():
            self._ground_truth_intermediate_cache[dtype][:, envs_idx] = 0.0
            self._intermediate_cache[dtype][envs_idx] = 0.0
            if dtype in self._ground_truth_timeline_ring:
                self._ground_truth_timeline_ring[dtype].buffer[:, envs_idx] = 0.0
            if dtype in self._measured_timeline_ring:
                self._measured_timeline_ring[dtype].buffer[:, envs_idx] = 0.0

        # Reset per-class return caches. When the return cache is an alias-view of the intermediate cache the clear is
        # redundant (the intermediate clear above already wrote zeros to the same memory) but harmless. Per-class
        # return-space rings are always distinct buffers.
        for sensor_cls in self._return_cache:
            self._return_cache[sensor_cls][envs_idx] = 0
            self._ground_truth_return_cache[sensor_cls][envs_idx] = 0
        for ring in self._ground_truth_return_timeline_ring.values():
            ring.buffer[:, envs_idx] = 0
        for ring in self._measured_return_timeline_ring.values():
            ring.buffer[:, envs_idx] = 0

        # Reset shared contexts before the per-type sensor reset (a reset may change otherwise-static geometry, so the
        # context must rebuild before any sensor reads it again).
        for context in self._shared_contexts.values():
            context.reset(envs_idx)

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_intermediate_dtype()
            cache_slice = self._cache_slices_by_type[sensor_cls]
            sensor_cls.reset(
                self._sensors_metadata[sensor_cls], self._ground_truth_intermediate_cache[dtype][cache_slice], envs_idx
            )

    def step(self):
        # Timeline rings must rotate before `_update_shared_cache` because `_apply_transform` mutates `at(0)` of the
        # timeline ring and needs a fresh write slot. Return-space rings, by contrast, are read during `_post_process`
        # (past post-output values) and written afterward; their rotation is deferred to inside the per-class loop so
        # `at(0)` during `_post_process` is the previous step's post-output (a meaningful "previous value") rather than
        # stale data from the slot about to be overwritten.
        for ring in self._measured_timeline_ring.values():
            ring.rotate()

        # Refresh each shared context once per step, before the per-type loop reads it, so multiple consuming sensor
        # types (e.g. Raycaster + DepthCamera) rebuild the shared resource at most once rather than once each.
        for context in self._shared_contexts.values():
            context.update()

        fps_tracker = self._sim.fps_tracker
        for sensor_cls, sensors in self._sensors_by_type.items():
            fps_tracker.start_phase(f"sensors/{sensor_cls.__name__}")
            dtype = sensor_cls._get_intermediate_dtype()
            cache_slice = self._cache_slices_by_type[sensor_cls]
            ground_truth_slice = self._ground_truth_intermediate_cache[dtype][cache_slice]
            intermediate = self._intermediate_cache[dtype][:, cache_slice]
            ground_truth_data_timeline = (
                self._ground_truth_timeline_ring[dtype][:, cache_slice]
                if dtype in self._ground_truth_timeline_ring
                else None
            )
            measured_data_timeline = (
                self._measured_timeline_ring[dtype][:, cache_slice] if dtype in self._measured_timeline_ring else None
            )
            metadata = self._sensors_metadata[sensor_cls]
            sensor_cls._update_shared_cache(
                self._shared_contexts.get(sensor_cls._shared_context_cls),
                metadata,
                ground_truth_slice,
                ground_truth_data_timeline,
                measured_data_timeline,
                intermediate,
            )

            gt_return_ring = self._ground_truth_return_timeline_ring.get(sensor_cls)
            if gt_return_ring is None:
                # No return-space ring: identity `_post_process`, no delay, no history. Return cache aliases
                # intermediate (the per-step write inside `_update_shared_cache` is already visible to `read()`).
                continue
            measured_return_ring = self._measured_return_timeline_ring[sensor_cls]

            # Project both branches into the return-space ring slot 0. `_post_process` returns the post-cast tensor. The
            # ring has not yet been rotated this step, so during the override `timeline.at(0)` is the previous step's
            # post-output (the most recent valid value) and `timeline.at(1)`, `at(2)`, etc. are older. `is_measured`
            # lets the override apply readout-stage contributions on only one branch.
            measured_projected = sensor_cls._post_process(
                metadata, intermediate, measured_return_ring, is_measured=True
            )
            gt_projected = sensor_cls._post_process(metadata, ground_truth_slice.T, gt_return_ring, is_measured=False)

            # Rotate now, after `_post_process` reads finished and before writing this step's projections into slot 0.
            # Only one rotate per pair since GT and measured return rings share idx.
            gt_return_ring.rotate()
            measured_return_ring.set(measured_projected)
            gt_return_ring.set(gt_projected)

            # GT has no readout delay (delay is a measured-only effect), so the GT read is just the current ring slot.
            self._ground_truth_return_cache[sensor_cls].copy_(gt_return_ring.at(0, copy=False))
            # Measured: per-sensor delay + jitter sampling from the return-space ring into the per-class return cache.
            # `_apply_delay` is an overrideable classmethod on `Sensor` whose default ZOH implementation is dtype-safe
            # for any return space (bool, uint8, quantized float, ...).
            sensor_cls._apply_delay(metadata, measured_return_ring, self._return_cache[sensor_cls])

        fps_tracker.stop_phase()

    def draw_debug(self, context: "RasterizerContext"):
        for sensor in self.sensors:
            if sensor._options.draw_debug:
                sensor._draw_debug(context)

    def get_cloned_from_cache(self, sensor: "Sensor", is_ground_truth: bool = False) -> torch.Tensor:
        sensor_cls = type(sensor)
        rel_start = sensor._cache_offset
        history_length = sensor._options.history_length

        if history_length > 0:
            sensor_hist = self._gather_history(sensor_cls, history_length, is_ground_truth)
            sensor_slice = slice(rel_start, rel_start + sensor._cache_size)
            sensor_hist = sensor_hist[:, :, sensor_slice]
            blocks = [sensor_hist[..., rel_slice].flatten(1, 2) for rel_slice in sensor._cache_slices]
            if len(blocks) == 1:
                return blocks[0]
            return torch.cat(blocks, dim=1)

        # Pure view into the per-class return cache. Eager `_post_process` already populated it during step().
        return_cache = (
            self._ground_truth_return_cache[sensor_cls] if is_ground_truth else self._return_cache[sensor_cls]
        )
        return return_cache[:, rel_start : rel_start + sensor._cache_size]

    def _gather_history(self, sensor_cls: type["Sensor"], history_length: int, is_ground_truth: bool) -> torch.Tensor:
        # Gather the last `history_length` snapshots for the whole class into a fresh `(B, H, cls_size)` tensor. Always
        # reads from the per-class return-space ring: it records the post-everything snapshot at each step, so history
        # reads return the final measured (or GT) values observed in the past. The intermediate ring is in
        # pre-hardware-imperfection space and would yield wrong history.
        hist_idx = self._hist_idx_by_class[sensor_cls][:history_length]
        ring = (
            self._ground_truth_return_timeline_ring[sensor_cls]
            if is_ground_truth
            else self._measured_return_timeline_ring[sensor_cls]
        )
        return ring.at(hist_idx).transpose(0, 1)

    def read_sensors(
        self, entity_idx: int | None = None, envs_idx=None, is_ground_truth: bool = False
    ) -> dict[int, torch.Tensor]:
        """
        Read the latest data of every sensor class in scope as a single tensor per class.

        Always returns a fresh tensor per class, independent of the internal sensor storage; the caller is free to
        mutate the result.

        Parameters
        ----------
        entity_idx : int | None
            - None (default): include every sensor in the scene.
            - k >= 0: include only sensors whose `entity_idx == k`.
            - -1: include only static sensors (those not attached to any entity).
        envs_idx : array-like | int | slice | None
            Environment selection. Defaults to all environments.
        is_ground_truth : bool
            When True, return ground-truth tensors instead of measured tensors.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from sensor-type tag (`gs.sensors.types.<Name>`) to a tensor of shape
            (B, [history,] class_or_entity_cache_size). For sensors without history, the history
            dimension is omitted.
        """
        # Sanitize envs_idx to a 1D tensor so fancy-indexing the batch axis always allocates a fresh tensor; this is
        # what gives the function its mutation-safe contract.
        env_index = self._sim._scene._sanitize_envs_idx(envs_idx)

        result: dict[int, torch.Tensor] = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            entity_slice_map = self._entity_slice_in_class.get(sensor_cls, {})
            if entity_idx is None:
                cls_slice = self._cache_slices_by_type[sensor_cls]
                within_cls_slice = slice(0, cls_slice.stop - cls_slice.start)
            else:
                eid = -1 if entity_idx < 0 else entity_idx
                if eid not in entity_slice_map:
                    continue
                within_cls_slice = entity_slice_map[eid]

            cls_max_history = self._max_history_by_class[sensor_cls]
            if cls_max_history > 0:
                sensor_hist = self._gather_history(sensor_cls, cls_max_history, is_ground_truth)
                tensor = sensor_hist[env_index, :, within_cls_slice]
            else:
                return_cache = (
                    self._ground_truth_return_cache[sensor_cls] if is_ground_truth else self._return_cache[sensor_cls]
                )
                tensor = return_cache[env_index, within_cls_slice]

            if self._sim.n_envs == 0:
                tensor = tensor[0]
            options_cls = type(sensors[0]._options)
            type_id = getattr(_sensor_types_namespace, options_cls.__name__)
            result[type_id] = tensor
        return result

    def get_sensors_by_entity(self, entity_idx: int) -> "gs.List[Sensor]":
        """List of all sensors attached to the given entity (or static sensors for entity_idx == -1)."""
        target_eid = -1 if entity_idx < 0 else entity_idx
        return gs.List(
            sensor
            for sensor_list in self._sensors_by_type.values()
            for sensor in sensor_list
            if sensor._options.entity_idx == target_eid
        )

    @property
    def sensors(self):
        return gs.List([sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list])
