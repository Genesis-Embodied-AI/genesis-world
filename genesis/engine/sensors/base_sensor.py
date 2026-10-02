from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, NamedTuple, TypeVar, get_args, get_origin

import numpy as np
import torch

import typing_extensions

import genesis as gs
from genesis.repr_base import RBC
from genesis.typing import EnvsIdxType, LaxFArrayType, NumArrayType, NumericType
from genesis.utils.geom import euler_to_quat
from genesis.utils.misc import assign_indexed_tensor, indices_to_mask
from genesis.utils.ring_buffer import TensorRingBuffer

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import KinematicLink
    from genesis.engine.simulator import Simulator
    from genesis.engine.solvers import RigidSolver
    from genesis.engine.solvers.kinematic_solver import KinematicSolver
    from genesis.options.sensors.options import SensorOptions
    from genesis.recorders.base_recorder import Recorder, RecorderOptions
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


def _to_tuple(*values: NumArrayType, length_per_value: int = 3) -> tuple[NumericType, ...]:
    """Concatenate the input values into one flat tuple, each scalar repeated ``length_per_value`` times."""
    full_tuple = ()
    for value in values:
        if isinstance(value, NumericType):
            value = (value,) * length_per_value
        elif isinstance(value, torch.Tensor):
            value = value.reshape((-1,))
        full_tuple += tuple(value)
    return full_tuple


class SharedSensorContext(ABC):
    """
    A resource shared by several sensor arrays, such as the raycast bounding volume hierarchies (BVHs) that a raycaster
    and a tactile array both cast against.

    An array fetches the context it reads with `SensorManager.get_context` in its `build`, and every array asking for
    the same class gets the one instance SensorManager owns. An array aggregates the state of every sensor of one type, and one kernel runs
    over them. A context is a single resource that several types read. The results of an array are identical whether or
    not the resource is shared, and SensorManager keeps them consistent.

    SensorManager constructs the context with the sim when the first array declaring it is constructed. The context
    stays an empty shell until a consumer activates it:

    - ``activate``: a consuming array calls it from its own ``build``, when the scene geometry is available. The first
      call constructs the resource, and later calls have no effect.
    - ``update``: the manager calls it once per step before the arrays step. An inactive context skips it.
    - ``reset`` and ``destroy``: the manager calls them on ``scene.reset()`` and at teardown.

    Reading the resource of an inactive context raises. Subclasses implement every lifecycle method and guard
    ``update``, ``reset`` and ``destroy`` on ``is_active``.
    """

    def __init__(self, sim: "Simulator"):
        self._sim = sim
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @abstractmethod
    def activate(self) -> None:
        """
        Mark the context active and construct the resource.

        A consuming array calls it from its ``build``, when the scene geometry is available. Later calls have no effect.
        """

    @abstractmethod
    def update(self) -> None:
        """
        Refresh the resource for the current step.

        SensorManager calls it once per step, before the arrays step. An inactive context skips the refresh.
        """

    @abstractmethod
    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        """
        Reset the resource.

        SensorManager calls it on ``scene.reset()``. An inactive context skips the reset.
        """

    @abstractmethod
    def destroy(self) -> None:
        """Release the resources the context holds when the scene is torn down."""


OptionsT = TypeVar("OptionsT", bound="SensorOptions")
ArrayT = TypeVar("ArrayT", bound="SensorArray")
DataT = typing_extensions.TypeVar("DataT", default=tuple, covariant=True)


class Sensor(RBC, Generic[OptionsT, ArrayT]):
    """
    Handle of one sensor: its options, its index among the sensors of its type, and the array of that type.

    Every sensor of a type belongs to one `SensorArray`, which owns every tensor and runs every kernel over all of them
    at once. The handle only names one of them: `read` and every setter delegate to the array with the index. Users
    obtain handles from `scene.add_sensor(options)`.

    A concrete sensor class declares its options and its array as the generic parameters::

        class MySensor(Sensor[MyOptions, MySensorArray]): ...
    """

    _options_cls: ClassVar[type]
    _array_cls: ClassVar[type]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is not None and issubclass(origin, Sensor):
                args = get_args(base)
                if len(args) >= 1 and not isinstance(args[0], TypeVar):
                    cls._options_cls = args[0]
                if len(args) >= 2 and not isinstance(args[1], TypeVar):
                    cls._array_cls = args[1]
                break
        # A class naming its own options is a concrete sensor type: it registers itself and needs its array
        if "_options_cls" in cls.__dict__:
            if "_array_cls" not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must specify Sensor[OptionsT, ArrayT].")
            from .sensor_manager import SensorManager

            SensorManager.SENSOR_TYPES_MAP[cls._options_cls] = cls

    def __init__(self, options: OptionsT, array: ArrayT):
        self._options: OptionsT = options
        self._array: ArrayT = array
        # Index among the sensors of the type, fixed at build once the array has ordered them
        self._idx: int = -1
        self._is_built = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def options(self) -> OptionsT:
        """The options this sensor was added with."""
        return self._options

    @property
    def idx(self) -> int:
        """Index of the sensor among the sensors of its type, fixed at build."""
        return self._idx

    @gs.assert_built
    def read(self, envs_idx: EnvsIdxType | None = None) -> "NamedTuple | torch.Tensor":
        """
        Read the latest measured data of this sensor, as of the last `scene.step()`.

        Parameters
        ----------
        envs_idx : array_like, optional
            The indices of the environments to read. If None, read all environments.

        Returns
        -------
        data
            The measured sensor data, formatted per the return type of the array.
        """
        return self._array.read(self._idx, envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx: EnvsIdxType | None = None) -> "NamedTuple | torch.Tensor":
        """
        Read the latest ground-truth data of this sensor, as of the last `scene.step()`.

        Parameters
        ----------
        envs_idx : array_like, optional
            The indices of the environments to read. If None, read all environments.

        Returns
        -------
        data
            The ground-truth sensor data, formatted per the return type of the array.
        """
        return self._array.read(self._idx, envs_idx, is_ground_truth=True)

    @gs.assert_unbuilt
    def start_recording(self, rec_options: "RecorderOptions") -> "Recorder":
        """
        Record the measured data of this sensor at every step, as the recorder options describe.

        Parameters
        ----------
        rec_options : RecorderOptions
            The options of the recorder to attach.

        Returns
        -------
        recorder : Recorder
            The recorder attached to this sensor.
        """
        return self._array._sim._scene._recorder_manager.add_recorder(self.read, rec_options)


class SensorArray(RBC, Generic[OptionsT, DataT]):
    """
    Array of every sensor of one type in the scene: it owns their tensors and simulates all of them at once.

    It relates to its sensors as a solver to its entities. SensorManager constructs it with the sim when the first
    sensor of the type is added, hands it every sensor handle through `add_sensor`, then drives it like a solver:
    `build` lays out the caches and computes every table from all the sensors at once, `step` computes one step of data
    for all of them, `reset` and `destroy` follow the scene. Handles read and write their sensor's slice of the tensors
    through the array, with their index.

    An array reading a resource shared with other types fetches it with `SensorManager.get_context` in its `build` (see
    SharedSensorContext); SensorManager constructs the context on the first request and refreshes it before the arrays
    step.

    Every attribute is created in `build`. The constructor holds only what `destroy` releases after an aborted build.
    """

    # Whether the sensors of this type go through the per-step ring pipeline (caches, delay sampling, transform
    # recurrence, history snapshots). A type computing its data on read (cameras) opts out: it owns its storage,
    # overrides `read` and `read_all`, and refuses the options the pipeline implements.
    has_ring_pipeline: ClassVar[bool] = True

    # The return type of the sensors of this type, taken from the generic parameter; a bare tuple by default
    _return_data_cls: ClassVar[type] = tuple

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is not None and issubclass(origin, SensorArray):
                args = get_args(base)
                if len(args) >= 2 and not isinstance(args[1], TypeVar):
                    cls._return_data_cls = args[1]
                break

    def __init__(self, sim: "Simulator", manager: "SensorManager"):
        self._sim = sim
        self._manager = manager
        self._sensors: list[Sensor] = []
        self._is_built = False

    # =============================== registration ===============================

    def add_sensor(self, sensor: Sensor) -> None:
        """Register a sensor handle before the build."""
        options = sensor.options
        # The ring pipeline is what implements delay, jitter and history, so a type opting out refuses them rather than
        # silently ignoring them
        if not self.has_ring_pipeline:
            for name, value in (
                ("delay", options.delay),
                ("jitter", options.jitter),
                ("history_length", options.history_length),
            ):
                if value > 0:
                    gs.raise_exception(f"{type(sensor).__name__} does not support `{name}`; got {name}={value}.")
        self._sensors.append(sensor)

    @property
    def sensors(self) -> list[Sensor]:
        return self._sensors

    @property
    def n_sensors(self) -> int:
        return len(self._sensors)

    @property
    def is_built(self) -> bool:
        return self._is_built

    # =============================== methods to implement ===============================

    def _get_return_format(self, options: OptionsT) -> tuple[int | tuple[int, ...], ...]:
        """
        Return the shape of each tensor a sensor with these options reads.

        The options are free to shape the return (the pattern of a raycaster, the resolution of a camera, the probe
        positions of a proximity sensor). A single tensor is described by one tuple such as ``(N,)``, several tensors by
        a tuple of tuples such as ``((3,), (3,), (3,))`` for the ``NamedTuple(lin_acc, ang_vel, mag)`` of the IMU.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `_get_return_format()`.")

    def _get_cache_dtype(self) -> torch.dtype:
        """Return the dtype of the tensors the sensors read, one dtype for the whole type."""
        raise NotImplementedError(f"{type(self).__name__} has not implemented `_get_cache_dtype()`.")

    def _get_intermediate_dtype(self) -> torch.dtype:
        """
        Return the dtype of the pipeline-internal cache, which defaults to the return dtype.

        An array whose ``_post_process`` changes the dtype overrides it too, as ContactSensor does with a float
        intermediate and a bool return.
        """
        return self._get_cache_dtype()

    def _update_cache(self) -> None:
        """
        Compute one step of sensor data into the caches, up to the per-step working buffer.

        The hook writes four buffers of shape ``(B, cols)``:

        - ``_ground_truth_cache``, the ground truth (GT) after the transform.
        - slot 0 of ``_ground_truth_timeline``, the GT write slot of the step.
        - slot 0 of ``_measured_timeline``, the measured write slot of the step, after the physics imperfections and the
          transform and before the hardware imperfections.
        - ``_intermediate_cache``, the measured value after the hardware imperfections and before ``_post_process`` and
          the delay sampling.

        `step` runs ``_post_process``, the return-space rings and the delay sampling after this hook returns.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `_update_cache()`.")

    def _apply_delay(self, return_ring: TensorRingBuffer, return_cache: torch.Tensor) -> None:
        """
        Sample stale slots of the measured return-space ring into the user-visible measured return cache.

        The default is a per-sensor zero-order hold (ZOH) at ``delay + jitter`` steps back, which holds for every return
        dtype (bool, int, uint8, quantized float). An override may sample a float return space more smoothly, for
        instance by a linear interpolation between adjacent slots.

        ``return_ring`` is the measured return-space ring: slot 0 holds the value of the current step and slot ``k`` the
        value ``k`` steps back. ``return_cache`` is the measured return cache to populate, of the same shape and dtype
        as the ring.
        """
        if not self.has_any_delay and not self.has_any_jitter:
            # Fast path: no per-sensor delay loop, just copy the most recent slot type-wide.
            return_cache.copy_(return_ring.at(0, copy=False))
            return

        if self.has_any_jitter:
            # Uniform jitter in [0, jitter_ts) per env per sensor. Combined with the `jitter <= delay` and `jitter < dt`
            # option constraints, the effective per-step shift cannot wrap the ring.
            cur_jitter_ts = torch.rand_like(self.jitter_ts).mul_(self.jitter_ts)
        else:
            cur_jitter_ts = None

        tensor_start = 0
        for i_s, tensor_size in enumerate(self.cache_sizes):
            cur_delay_ts = self.delays_ts[:, i_s]
            if cur_jitter_ts is not None:
                # Probabilistic rounding of the continuous-time delay onto integer ring slots: with `jitter < dt` (one
                # slot), the realized jitter sample `j` is in `[0, 1)`; adding `uniform[0, 1)` then flooring picks the
                # next slot (`D + 1`) with probability `j`, preserving the expected jitter shift while staying
                # dtype-safe (no interpolation between adjacent slots).
                cur_delay_ts = cur_delay_ts + cur_jitter_ts[:, i_s] + torch.rand_like(cur_jitter_ts[:, i_s])
            # The per-row gather of the ring takes an int64 index
            cur_delay_ts_int = cur_delay_ts.to(dtype=torch.int64)
            tensor_slice = slice(tensor_start, tensor_start + tensor_size)
            return_cache[:, tensor_slice].copy_(return_ring.at(cur_delay_ts_int, tensor_slice, per_row=True))
            tensor_start += tensor_size

    def _post_process(self, tensor: torch.Tensor, timeline: TensorRingBuffer, *, is_measured: bool) -> torch.Tensor:
        """
        Project the intermediate cache into return space, once per branch per step.

        ``tensor`` is the whole intermediate cache ``[B, total_cache_size]``: on the measured branch the value after the
        physics imperfections, the transform and the hardware imperfections, on the ground truth (GT) branch the value
        after the transform. The hook returns the projected value in the return dtype. `step` writes it into slot 0 of
        the return-space ring, and the read the user sees comes from the delay sampling of that ring.

        ``timeline`` is that ring, holding the projected snapshots. `step` rotates it after this call returns, so
        ``timeline.at(0)`` is the projection of the previous step, ``timeline.at(1)`` the one before, and so on.
        ``is_measured`` is True on the measured branch and False on the GT branch, so an override can apply a
        readout-stage contribution to one side.
        """
        return tensor

    def _draw_debug(self, i_s: int, context: "RasterizerContext") -> None:
        """Draw the debug visualization of sensor ``i_s`` in the rasterizer context."""
        raise NotImplementedError(f"{type(self).__name__} has not implemented `_draw_debug()`.")

    # =============================== lifecycle ===============================

    def build(self) -> None:
        """
        Lay out the caches of every sensor of the type and compute the type-wide tables, all sensors added.

        A subclass or mixin holding tables overrides it, calls super first and stacks its own. SensorManager calls it
        once per array once the scene is built.
        """
        _B = self._sim._B
        self._sensors.sort(key=lambda sensor: sensor.options.entity_idx)
        for i_s, sensor in enumerate(self._sensors):
            sensor._idx = i_s

        # The flat cache of the type lays the elements of each sensor end to end, one span per returned tensor. A
        # history read stacks the snapshots per returned tensor, so the flat read spans grow with the history.
        self.cache_sizes: list[int] = []
        self._sensors_cache_offset: list[int] = []
        self._sensors_cache_slices: list[list[slice]] = []
        self._sensors_read_slices: list[list[slice]] = []
        self._sensors_return_shapes: list[tuple[tuple[int, ...], ...]] = []
        entity_spans: dict[int, list[int]] = {}
        for sensor in self._sensors:
            return_format = self._get_return_format(sensor.options)
            assert len(return_format) > 0
            intrinsic_shapes: tuple[tuple[int, ...], ...] = (
                (return_format,) if isinstance(return_format[0], int) else return_format
            )
            history_length = sensor.options.history_length
            cache_size = 0
            cache_slices = []
            read_slices = []
            read_offset = 0
            for shape in intrinsic_shapes:
                data_size = np.prod(shape)
                cache_slices.append(slice(cache_size, cache_size + data_size))
                cache_size += data_size
                span = data_size * history_length if history_length > 0 else data_size
                read_slices.append(slice(read_offset, read_offset + span))
                read_offset += span
            cache_offset = sum(self.cache_sizes)
            self.cache_sizes.append(cache_size)
            self._sensors_cache_offset.append(cache_offset)
            self._sensors_cache_slices.append(cache_slices)
            self._sensors_read_slices.append(read_slices)
            if history_length > 0:
                self._sensors_return_shapes.append(tuple((history_length, *shape) for shape in intrinsic_shapes))
            else:
                self._sensors_return_shapes.append(intrinsic_shapes)
            span = entity_spans.setdefault(sensor.options.entity_idx, [cache_offset, cache_offset])
            span[1] = cache_offset + cache_size
        self._entity_slices = {entity_idx: slice(start, stop) for entity_idx, (start, stop) in entity_spans.items()}
        # The first column of each sensor, for the kernels (the Python list serves the reads)
        self.sensors_cache_start = torch.tensor(self._sensors_cache_offset, dtype=gs.tc_int, device=gs.device)
        n_cols = sum(self.cache_sizes)

        # A type computing its data on read owns its storage (see has_ring_pipeline)
        if not self.has_ring_pipeline:
            return

        # The read delay of each sensor in steps and its jitter, per environment for the setters
        self._dt = self._sim.dt
        delays_ts = [round(sensor.options.delay / self._dt) for sensor in self._sensors]
        jitters_ts = [sensor.options.jitter / self._dt for sensor in self._sensors]
        self.delays_ts = torch.stack([torch.as_tensor(delays_ts, dtype=gs.tc_int, device=gs.device)] * _B)
        self.jitter_ts = torch.stack([torch.as_tensor(jitters_ts, dtype=gs.tc_float, device=gs.device)] * _B)
        # Python flags gating the per-step delay sampling without a GPU sync. Latched True by `set_jitter`.
        self.has_any_delay = any(delay_ts > 0 for delay_ts in delays_ts)
        self.has_any_jitter = any(sensor.options.jitter > gs.EPS for sensor in self._sensors)
        self.history_lengths = [sensor.options.history_length for sensor in self._sensors]
        max_history = max(self.history_lengths, default=0)
        # A delay reserves one slot past itself for the jitter shift, which `set_jitter` can raise at any time; without
        # that slot, `at()` wraps modulo the depth and returns the newest frame as the oldest
        delay_depth = max((delay_ts + (2 if delay_ts > 0 else 1) for delay_ts in delays_ts), default=1)

        # The working buffers: the ground truth and the measured value of the step, in intermediate space
        intermediate_dtype = self._get_intermediate_dtype()
        return_dtype = self._get_cache_dtype()
        cache_shape = (_B, n_cols)
        self._ground_truth_cache = torch.zeros(cache_shape, dtype=intermediate_dtype, device=gs.device)
        self._intermediate_cache = torch.zeros(cache_shape, dtype=intermediate_dtype, device=gs.device)
        # The paired ground truth (GT) and measured timeline rings (post-transform, PRE-hardware-imperfections data)
        # share one rotation index so a single `rotate()` per step advances both
        ring_n = max(2, max_history)
        self._measured_timeline = TensorRingBuffer(ring_n, cache_shape, dtype=intermediate_dtype)
        self._ground_truth_timeline = TensorRingBuffer(
            ring_n, cache_shape, dtype=intermediate_dtype, idx=self._measured_timeline._idx
        )
        # The return-space rings (post-everything, pre-delay-sample) exist when a delay, a history or a projection
        # asks for them; `step` then writes the post-everything snapshot to slot 0 and samples the delay from there.
        # Otherwise the return caches alias the working buffers, whose per-step write is directly visible to `read`.
        is_post_process_overridden = type(self)._post_process is not SensorArray._post_process
        if delay_depth > 1 or max_history > 0 or is_post_process_overridden:
            ring_n = max(delay_depth, max_history, 2 if is_post_process_overridden else 1)
            self._ground_truth_return_timeline = TensorRingBuffer(ring_n, cache_shape, dtype=return_dtype)
            self._measured_return_timeline = TensorRingBuffer(
                ring_n, cache_shape, dtype=return_dtype, idx=self._ground_truth_return_timeline._idx
            )
            self._return_cache = torch.zeros(cache_shape, dtype=return_dtype, device=gs.device)
            self._ground_truth_return_cache = torch.zeros(cache_shape, dtype=return_dtype, device=gs.device)
        else:
            self._ground_truth_return_timeline = None
            self._measured_return_timeline = None
            self._return_cache = self._intermediate_cache
            self._ground_truth_return_cache = self._ground_truth_cache
        self._history_idx = torch.arange(max_history, device=gs.device, dtype=gs.tc_int)

    def step(self) -> None:
        """Compute one step of data for every sensor of the type.

        The caches come first, then the projection, the return-space rings and the delay sampling. A type computing its
        data on read has nothing to step.
        """
        if not self.has_ring_pipeline:
            return
        self._measured_timeline.rotate()
        self._update_cache()
        if self._measured_return_timeline is None:
            return
        measured_projected = self._post_process(
            self._intermediate_cache, self._measured_return_timeline, is_measured=True
        )
        ground_truth_projected = self._post_process(
            self._ground_truth_cache, self._ground_truth_return_timeline, is_measured=False
        )
        self._ground_truth_return_timeline.rotate()
        self._measured_return_timeline.set(measured_projected)
        self._ground_truth_return_timeline.set(ground_truth_projected)
        self._ground_truth_return_cache.copy_(self._ground_truth_return_timeline.at(0, copy=False))
        self._apply_delay(self._measured_return_timeline, self._return_cache)

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        """
        Reset the sensors of the type in the given environments, clearing every cache and ring.

        SensorManager calls it on `scene.reset()` with the environments as a mask, as indices_to_mask returns it, over
        every environment on a whole reset. A subclass holding state overrides it, calls super first and clears its own.
        A type computing its data on read has no cache.
        """
        if not self.has_ring_pipeline:
            return
        self._ground_truth_cache[envs_idx] = 0
        self._intermediate_cache[envs_idx] = 0
        for ring in (
            self._ground_truth_timeline,
            self._measured_timeline,
            self._ground_truth_return_timeline,
            self._measured_return_timeline,
        ):
            if ring is not None:
                ring.buffer[(slice(None), *envs_idx)] = 0
        if self._return_cache is not self._intermediate_cache:
            self._return_cache[envs_idx] = 0
            self._ground_truth_return_cache[envs_idx] = 0

    def destroy(self) -> None:
        """Release the resources of the type, when the scene is destroyed."""

    # =============================== reads and writes ===============================

    def read(self, i_s: int, envs_idx: EnvsIdxType | None = None, is_ground_truth: bool = False) -> DataT:
        """Read sensor ``i_s``, as of the last step, formatted per the return type (with the history stacked in front
        of each returned tensor when the sensor keeps one)."""
        cache_slice = self._cache_slice(i_s)
        history_length = self.history_lengths[i_s]
        if history_length > 0:
            history = self._gather_history(history_length, is_ground_truth)[:, :, cache_slice]
            blocks = [
                history[..., rel_slice].reshape((history.shape[0], -1)) for rel_slice in self._sensors_cache_slices[i_s]
            ]
            tensor = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)
        else:
            # The return cache `step` populated; _get_formatted_data selects the environments from it
            return_cache = self._ground_truth_return_cache if is_ground_truth else self._return_cache
            tensor = return_cache[:, cache_slice]
        return self._get_formatted_data(i_s, tensor, envs_idx)

    def read_all(
        self, entity_idx: int | None = None, envs_idx: EnvsIdxType | None = None, is_ground_truth: bool = False
    ) -> torch.Tensor:
        """
        Read every sensor of the type as one tensor, or only those attached to an entity.

        Returns a fresh tensor of shape ``(B, [history,] cache_size)``, the history dimension present when a sensor of
        the type keeps one, or None when no sensor of the type is attached to the requested entity. ``entity_idx``
        None selects every sensor, ``-1`` the static ones.
        """
        if entity_idx is None:
            cols = slice(0, sum(self.cache_sizes))
        else:
            cols = self._entity_slices.get(-1 if entity_idx < 0 else entity_idx)
            if cols is None:
                return None
        max_history = max(self.history_lengths, default=0)
        if max_history > 0:
            tensor = self._gather_history(max_history, is_ground_truth)[indices_to_mask(envs_idx, None, cols)]
        else:
            return_cache = self._ground_truth_return_cache if is_ground_truth else self._return_cache
            tensor = return_cache[indices_to_mask(envs_idx, cols)]
            # A slice selection views the cache, and the caller is free to mutate what read_all returns
            if tensor.untyped_storage().data_ptr() == return_cache.untyped_storage().data_ptr():
                tensor = tensor.clone()
        if self._sim.n_envs == 0:
            tensor = tensor[0]
        return tensor

    def draw_debug(self, context: "RasterizerContext") -> None:
        for i_s, sensor in enumerate(self._sensors):
            if sensor.options.draw_debug:
                self._draw_debug(i_s, context)

    def _cache_slice(self, i_s: int) -> slice:
        """The columns of sensor ``i_s`` in the flat cache of the type."""
        start = self._sensors_cache_offset[i_s]
        return slice(start, start + self.cache_sizes[i_s])

    def _gather_history(self, history_length: int, is_ground_truth: bool) -> torch.Tensor:
        """The last ``history_length`` post-everything snapshots of the type, as a fresh ``(B, H, cols)`` tensor."""
        # The return-space ring records the final value observed at each step; the intermediate ring holds
        # pre-hardware-imperfection values and would yield a wrong history
        ring = self._ground_truth_return_timeline if is_ground_truth else self._measured_return_timeline
        return ring.at(self._history_idx[:history_length]).transpose(0, 1)

    def _get_formatted_data(self, i_s: int, tensor: torch.Tensor, envs_idx: EnvsIdxType | None = None) -> DataT:
        """Split the flat data of sensor ``i_s`` into fresh tensors of its return type, for the given environments (the
        environment axis dropped when the scene has none)."""
        tensor_chunk = tensor[indices_to_mask(envs_idx)]
        # A read is a snapshot the caller keeps across steps, so a slice selection, which views the cache, is copied
        if tensor_chunk.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
            tensor_chunk = tensor_chunk.clone()
        n_selected = tensor_chunk.shape[0]
        tensor_chunk = tensor_chunk.reshape((n_selected, -1))
        return_values = []
        for shape, read_slice in zip(self._sensors_return_shapes[i_s], self._sensors_read_slices[i_s]):
            field_data = tensor_chunk[..., read_slice].reshape((n_selected, *shape))
            if self._sim.n_envs == 0:
                field_data = field_data[0]
            return_values.append(field_data)
        if len(return_values) == 1:
            return return_values[0]
        return self._return_data_cls(*return_values)

    def _set_field(
        self,
        value: LaxFArrayType,
        field: torch.Tensor,
        field_start: int,
        field_size: int,
        envs_idx: EnvsIdxType | None = None,
    ) -> None:
        """Write ``value`` into the columns ``field_start:field_start + field_size`` of a per-environment table (or the
        column ``field_start`` of a per-sensor one), for the given environments."""
        if field.ndim == 2:
            # Flat field structure: per-sensor spans may differ in size (e.g. cache-sized imperfection fields), so the
            # caller provides this sensor's start rather than a uniform stride.
            index_slice = slice(field_start, field_start + field_size)
        else:
            index_slice = field_start
        assign_indexed_tensor(field, indices_to_mask(envs_idx, index_slice, keepdim=False), value, ("envs_idx", ""))

    def set_jitter(self, i_s: int, jitter: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        """Set the read jitter of sensor ``i_s``, in seconds, for the given environments."""
        jitter_np = np.asarray(jitter, dtype=gs.np_float)
        if np.any(jitter_np < 0):
            gs.raise_exception(f"Sensor jitter must be non-negative; got jitter={jitter_np.tolist()}.")
        if np.any(jitter_np >= self._dt + gs.EPS):
            gs.raise_exception(
                f"Sensor jitter must not exceed the simulation step dt={self._dt}; got jitter={jitter_np.tolist()}."
            )
        # Same bound as `SensorOptions.model_post_init`, enforced here because only a sensor declaring a delay at build
        # time gets the ring slot a jittered read reaches (see the delay depth in `build`)
        delay = self._sensors[i_s].options.delay
        if np.any(jitter_np > delay):
            gs.raise_exception(
                f"Sensor jitter must not exceed the read delay={delay}; got jitter={jitter_np.tolist()}."
            )
        self._set_field(jitter_np / self._dt, self.jitter_ts, i_s, 1, envs_idx)
        # Recompute the slow-path flag from the freshly-written table. One GPU->CPU sync at setter call time; setters
        # are not hot path. The check covers partial envs_idx writes and other sensors.
        self.has_any_jitter = bool((self.jitter_ts > gs.EPS).any().item())


class _SolverLinkGroup(NamedTuple):
    """The sensors attached to one kinematic solver: their solver-local link indices and their columns in the tables
    of the type."""

    solver: "KinematicSolver"
    # Solver-local link index of each sensor of the group
    links_idx: torch.Tensor
    # Column of the type's tables each link pose lands in
    sensor_cols: torch.Tensor


class LinkAttachedSensorArrayMixin:
    """Array of sensors attached to a link, holding the link of each sensor and its pose offsets per environment.

    A sensor whose options name no entity is static, with no link.
    """

    def build(self) -> None:
        super().build()

        self._links: list["KinematicLink | None"] = []
        self._links_idx: list[int] = []
        for sensor in self._sensors:
            options = sensor.options
            if options.entity_idx >= 0:
                entity = self._sim.entities[options.entity_idx]
                self._links.append(entity.links[options.link_idx_local])
                self._links_idx.append(options.link_idx_local + entity.link_start)
            else:
                self._links.append(None)
                self._links_idx.append(-1)
        _B = self._sim._B
        offsets_pos = [sensor.options.pos_offset for sensor in self._sensors]
        offsets_quat = euler_to_quat([sensor.options.euler_offset for sensor in self._sensors])
        self.offsets_pos = torch.stack([torch.as_tensor(offsets_pos, dtype=gs.tc_float, device=gs.device)] * _B)
        self.offsets_quat = torch.stack([torch.as_tensor(offsets_quat, dtype=gs.tc_float, device=gs.device)] * _B)

    def set_pos_offset(self, i_s: int, pos_offset: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._set_field(pos_offset, self.offsets_pos, i_s, 3, envs_idx)

    def set_quat_offset(self, i_s: int, quat_offset: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._set_field(quat_offset, self.offsets_quat, i_s, 4, envs_idx)


class KinematicSensorArrayMixin(LinkAttachedSensorArrayMixin):
    """
    Array of sensors attached to a KinematicEntity (or any subclass, including RigidEntity).

    The attached sensors are bucketed at build into per-solver ``_SolverLinkGroup`` entries so the per-step gather is
    one bulk read per solver. Static sensors keep an identity link pose, leaving the kernel to apply ``pos_offset`` /
    ``euler_offset`` in world frame.
    """

    def build(self) -> None:
        super().build()

        # One bucket per solver, holding the link of each attached sensor and the per-type column its pose lands in
        groups: dict["KinematicSolver", tuple[list[int], list[int]]] = {}
        for i_s, (link, link_idx) in enumerate(zip(self._links, self._links_idx)):
            if link is not None:
                links_idx, sensor_cols = groups.setdefault(link.entity.solver, ([], []))
                links_idx.append(link_idx)
                sensor_cols.append(i_s)
        self.solver_groups = [
            _SolverLinkGroup(
                solver=solver,
                links_idx=torch.tensor(links_idx, dtype=gs.tc_int, device=gs.device),
                sensor_cols=torch.tensor(sensor_cols, dtype=gs.tc_int, device=gs.device),
            )
            for solver, (links_idx, sensor_cols) in groups.items()
        ]


class RigidSensorArrayMixin(LinkAttachedSensorArrayMixin):
    """Array of sensors attached to a RigidEntity: the rigid solver and the global link index of each sensor."""

    def build(self) -> None:
        super().build()

        self.solver: "RigidSolver" = self._sim.rigid_solver
        self.links_idx = torch.tensor(
            [link_idx for link_idx in self._links_idx if link_idx >= 0], dtype=gs.tc_int, device=gs.device
        )


class LinkAttachedSensorMixin:
    """Handle of a sensor attached to a link: the setters of its pose offset."""

    @gs.assert_built
    def set_pos_offset(self, pos_offset: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._array.set_pos_offset(self._idx, pos_offset, envs_idx)

    @gs.assert_built
    def set_quat_offset(self, quat_offset: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._array.set_quat_offset(self._idx, quat_offset, envs_idx)


class SimpleSensorArray(SensorArray[OptionsT, DataT]):
    """
    Array of the sensor types that go through the standard per-step pipeline.

    Pipeline (per branch, in execution order):

    - Ground truth (GT) branch: ``raw -> _apply_transform(is_measured=False) -> _post_process(is_measured=False) ->
      ground truth``.
    - Measured branch: ``raw -> _apply_physics_imperfections -> _apply_transform(is_measured=True) ->
      _apply_hardware_imperfections -> _post_process(is_measured=True) -> delay sampling -> measured``.

    ``_update_raw_data`` and ``_apply_physics_imperfections`` are packaged inside ``_update_current_timestep_data``;
    override the latter to fuse them in a single kernel pass.

    Both branches keep their own intermediate-space timeline ring (``_ground_truth_timeline`` /
    ``_measured_timeline``). The timeline rings store post-transform, PRE-hardware-imperfections data, so
    ``_apply_transform`` recurrence reads clean previous slots and stateful filters (e.g. thermal dissipation) are not
    contaminated by hardware noise. Hardware imperfections mutate a per-step working buffer (the intermediate cache),
    never a timeline ring. The post-``_post_process`` snapshot of that working buffer is then frozen into slot 0 of the
    return-space ring, and delay sampling reads stale slots of the return-space ring to produce the user-visible value,
    so each delayed read returns the post-everything signal observed at the step of capture.

    Concrete arrays override the hooks (``_update_raw_data``, ``_update_current_timestep_data``,
    ``_apply_physics_imperfections``, ``_apply_transform``, ``_apply_hardware_imperfections``, ``_post_process``)
    rather than ``_update_cache`` itself. History reads gather post-everything snapshots from the return-space ring,
    so ``read`` with a history returns the final measured values observed at each past step.
    """

    def build(self) -> None:
        """Stack the imperfection parameters of every sensor, one span per sensor over the elements of its cache."""
        super().build()

        _B = self._sim._B
        # The bound set_jitter enforces at runtime, checked on the authored options once dt is known
        for sensor in self._sensors:
            if sensor.options.jitter >= self._dt + gs.EPS:
                gs.raise_exception(
                    f"Sensor jitter must not exceed the simulation step dt={self._dt}; got "
                    f"jitter={sensor.options.jitter}."
                )
        # The imperfections apply elementwise to the flat cache, so each value spreads over its sensor's span
        spans = [(sensor.options, cache_size) for sensor, cache_size in zip(self._sensors, self.cache_sizes)]
        resolution = sum((_to_tuple(sensor_options.resolution, length_per_value=n) for sensor_options, n in spans), ())
        bias = sum((_to_tuple(sensor_options.bias, length_per_value=n) for sensor_options, n in spans), ())
        random_walk = sum(
            (_to_tuple(sensor_options.random_walk, length_per_value=n) for sensor_options, n in spans), ()
        )
        noise = sum((_to_tuple(sensor_options.noise, length_per_value=n) for sensor_options, n in spans), ())
        self.resolution = torch.stack([torch.as_tensor(resolution, dtype=gs.tc_float, device=gs.device)] * _B)
        self.bias = torch.stack([torch.as_tensor(bias, dtype=gs.tc_float, device=gs.device)] * _B)
        self.random_walk = torch.stack([torch.as_tensor(random_walk, dtype=gs.tc_float, device=gs.device)] * _B)
        self._cur_random_walk = torch.zeros_like(self.random_walk)
        self.noise = torch.stack([torch.as_tensor(noise, dtype=gs.tc_float, device=gs.device)] * _B)
        sensors_options = [sensor_options for sensor_options, _ in spans]
        # Python flags gating the per-step imperfection work without a GPU sync. The setters recompute them from the
        # whole table (see set_noise).
        self.has_any_noise = any(
            np.any(np.asarray(sensor_options.noise, dtype=gs.np_float) > gs.EPS) for sensor_options in sensors_options
        )
        self.has_any_random_walk = any(
            np.any(np.asarray(sensor_options.random_walk, dtype=gs.np_float) > gs.EPS)
            for sensor_options in sensors_options
        )
        self.has_any_bias = any(
            np.any(np.abs(np.asarray(sensor_options.bias, dtype=gs.np_float)) > gs.EPS)
            for sensor_options in sensors_options
        )
        self.has_any_resolution = any(
            np.any(np.asarray(sensor_options.resolution, dtype=gs.np_float) > gs.EPS)
            for sensor_options in sensors_options
        )

    def reset(self, envs_idx: EnvsIdxType | None) -> None:
        super().reset(envs_idx)

        self._cur_random_walk[envs_idx] = 0.0

    def set_resolution(self, i_s: int, resolution: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._set_field(resolution, self.resolution, self._sensors_cache_offset[i_s], self.cache_sizes[i_s], envs_idx)
        self.has_any_resolution = bool((self.resolution > gs.EPS).any().item())

    def set_bias(self, i_s: int, bias: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._set_field(bias, self.bias, self._sensors_cache_offset[i_s], self.cache_sizes[i_s], envs_idx)
        self.has_any_bias = bool((self.bias.abs() > gs.EPS).any().item())

    def set_random_walk(self, i_s: int, random_walk: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._set_field(random_walk, self.random_walk, self._sensors_cache_offset[i_s], self.cache_sizes[i_s], envs_idx)
        self.has_any_random_walk = bool((self.random_walk > gs.EPS).any().item())

    def set_noise(self, i_s: int, noise: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._set_field(noise, self.noise, self._sensors_cache_offset[i_s], self.cache_sizes[i_s], envs_idx)
        self.has_any_noise = bool((self.noise > gs.EPS).any().item())

    def _update_cache(self) -> None:
        # Both branches start from the same raw signal; the ring contract is in the class docstring
        ground_truth_slot_0 = self._ground_truth_timeline.at(0, copy=False)
        measured_slot_0 = self._measured_timeline.at(0, copy=False)
        self._update_current_timestep_data(ground_truth_slot_0, measured_slot_0)
        # Ground-truth branch
        self._apply_transform(ground_truth_slot_0, self._ground_truth_timeline, is_measured=False)
        self._ground_truth_cache.copy_(ground_truth_slot_0)
        # Measured branch, then the hardware imperfections on the working buffer so the ring stays clean
        self._apply_transform(measured_slot_0, self._measured_timeline, is_measured=True)
        self._intermediate_cache.copy_(measured_slot_0)
        self._apply_hardware_imperfections(self._intermediate_cache)

    def _update_current_timestep_data(self, ground_truth_slot_0: torch.Tensor, measured_slot_0: torch.Tensor) -> None:
        """
        Compute the raw signal of the step and the measured signal with its physics imperfections.

        The default computes the raw signal into the ground-truth cache, copies it into both slots and perturbs the
        measured one. A sensor integrating its own state finds the previous step in the ground-truth cache. An override
        fuses both computations in one kernel pass.

        Parameters
        ----------
        ground_truth_slot_0 : torch.Tensor
            Slot 0 of the ground-truth timeline ring, to fill with the raw signal.
        measured_slot_0 : torch.Tensor
            Slot 0 of the measured timeline ring, to fill with the perturbed signal.
        """
        self._update_raw_data(self._ground_truth_cache)
        ground_truth_slot_0.copy_(self._ground_truth_cache)
        measured_slot_0.copy_(self._ground_truth_cache)
        self._apply_physics_imperfections(measured_slot_0, self._measured_timeline)

    def _update_raw_data(self, raw_data: torch.Tensor) -> None:
        """
        Compute the raw signal of every sensor of the type.

        Each sensor type implements this hook with its kernel.

        Parameters
        ----------
        raw_data : torch.Tensor
            The ground-truth cache to fill, holding the previous step's raw signal on entry.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `_update_raw_data()`.")

    def _apply_physics_imperfections(self, measured_slot_0: torch.Tensor, timeline: TensorRingBuffer) -> None:
        """
        Perturb the measured signal with the imperfections of the physical phenomenon, before the sensor element
        transduces it.

        ``measured_slot_0`` is slot 0 of the measured timeline ring and holds the raw signal on entry, to mutate
        in place. ``timeline`` is the measured ring, whose previous slots (``timeline.at(1)``, ...) serve
        stateful perturbations.
        """

    def _apply_transform(self, data: torch.Tensor, timeline: TensorRingBuffer, *, is_measured: bool) -> None:
        """
        Transform the signal of one branch, by a coordinate change or a filter of the sensor element.

        ``data`` is slot 0 of the timeline ring of the branch, to mutate in place. ``timeline`` is that ring: the ground
        truth (GT) ring on the GT branch, the measured ring on the measured branch. A stateful filter reads the previous
        slots with ``timeline.at(1)``, ``timeline.at(2)`` and so on. The rings hold the signal before the hardware
        imperfections, so a recurrence accumulates no hardware noise.

        ``is_measured`` names the branch. The hook runs on both branches, and a branch-symmetric effect such as a frame
        change applies to both. An effect of the sensor element that belongs to the measured signal alone, such as a
        resistor-capacitor (RC) time constant or a mechanical bandwidth, is gated on ``is_measured``.
        """

    def _apply_hardware_imperfections(self, measured_slot_0: torch.Tensor) -> None:
        """
        Apply the perturbations of the embedded sampling layer at the sensor output: random walk, noise, bias and
        quantization.

        A precomputed Python flag (``has_any_*``) gates each contribution, and a type with all-zero values pays no GPU
        work. ``measured_slot_0`` is the working buffer ``_post_process`` projects next, so the mutations stay local
        to the current step and out of the ``_apply_transform`` recurrence. An effect with memory across the output,
        such as a gain with memory, belongs in ``_post_process``, which sees the return-space ring and reads its
        previous slots.
        """
        if self.has_any_random_walk:
            self._cur_random_walk += torch.normal(0.0, self.random_walk)
            measured_slot_0 += self._cur_random_walk
        if self.has_any_noise:
            measured_slot_0 += torch.normal(0.0, self.noise)
        if self.has_any_bias:
            measured_slot_0 += self.bias
        if self.has_any_resolution:
            resolution = self.resolution
            mask = resolution > gs.EPS
            measured_slot_0[mask] = torch.round(measured_slot_0[mask] / resolution[mask]) * resolution[mask]


class SimpleSensor(Sensor[OptionsT, ArrayT]):
    """Handle of a sensor going through the standard per-step pipeline: the setters of its imperfections."""

    @gs.assert_built
    def set_resolution(self, resolution: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._array.set_resolution(self._idx, resolution, envs_idx)

    @gs.assert_built
    def set_bias(self, bias: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._array.set_bias(self._idx, bias, envs_idx)

    @gs.assert_built
    def set_random_walk(self, random_walk: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._array.set_random_walk(self._idx, random_walk, envs_idx)

    @gs.assert_built
    def set_noise(self, noise: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._array.set_noise(self._idx, noise, envs_idx)

    @gs.assert_built
    def set_jitter(self, jitter: LaxFArrayType, envs_idx: EnvsIdxType | None = None) -> None:
        self._array.set_jitter(self._idx, jitter, envs_idx)
