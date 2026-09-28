from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Generic, NamedTuple, TypeVar, get_args, get_origin

import numpy as np
import torch
from typing_extensions import TypeVar as TypeVarWithDefault

import genesis as gs
from genesis.repr_base import RBC
from genesis.typing import NumArrayType, NumericType
from genesis.utils.geom import euler_to_quat
from genesis.utils.misc import broadcast_tensor, concat_with_tensor, make_tensor_field

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.solvers import RigidSolver
    from genesis.engine.solvers.kinematic_solver import KinematicSolver
    from genesis.options.sensors.options import SensorOptions
    from genesis.recorders.base_recorder import Recorder, RecorderOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


def _to_tuple(*values: NumArrayType, length_per_value: int = 3) -> tuple[NumericType, ...]:
    """
    Convert all input values to one flattened tuple, where each value is ensured to be a tuple of length_per_value.
    """
    full_tuple = ()
    for value in values:
        if isinstance(value, NumericType):
            value = (value,) * length_per_value
        elif isinstance(value, torch.Tensor):
            value = value.reshape((-1,))
        full_tuple += tuple(value)
    return full_tuple


# Note: dataclass is used as opposed to pydantic.BaseModel since torch.Tensors are not supported by default
@dataclass
class SharedSensorMetadata:
    """
    Shared metadata between all sensors of the same class. Time-related state only - visible to SensorManager.
    """

    cache_sizes: list[int] = field(default_factory=list)
    delays_ts: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    history_lengths: list[int] = field(default_factory=list)
    jitter_ts: torch.Tensor = make_tensor_field((0, 0))
    # True iff at least one sensor in the class has a nonzero read delay. Precomputed at build so the per-step fast path
    # can avoid a GPU-syncing reduction.
    has_any_delay: bool = False
    # True iff at least one sensor in the class has a nonzero jitter. Latched True by `set_jitter`; same
    # precompute-and-latch contract as `has_any_delay`.
    has_any_jitter: bool = False

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def destroy(self):
        """
        Destroy shared metadata.

        This method is called by SensorManager when the scene is destroyed. This should remove any references to the
        sensors from the shared metadata, and clean up any resources associated with the sensors.
        """


@dataclass
class SimpleSensorMetadata(SharedSensorMetadata):
    """
    SimpleSensor's per-class state for the imperfection parameters (noise/bias/random_walk/resolution).

    Opaque to SensorManager (which only uses ``SharedSensorMetadata`` fields). Per-sensor-class metadata subclasses
    inherit this when the sensor derives from ``SimpleSensor``; sensors deriving from ``Sensor`` directly (Camera)
    inherit ``SharedSensorMetadata`` instead.
    """

    resolution: torch.Tensor = make_tensor_field((0, 0))
    bias: torch.Tensor = make_tensor_field((0, 0))
    _cur_random_walk: torch.Tensor = make_tensor_field((0, 0))
    random_walk: torch.Tensor = make_tensor_field((0, 0))
    noise: torch.Tensor = make_tensor_field((0, 0))
    # Precomputed Python bool flags gate the per-step noise/bias/quantize work without GPU sync. Set at build from
    # options and refreshed by the corresponding setters. Conservatively True once any sensor has nonzero value; never
    # flipped back to False (avoids tracking per-sensor state).
    has_any_noise: bool = False
    has_any_random_walk: bool = False
    has_any_bias: bool = False
    has_any_resolution: bool = False


class SharedSensorContext(ABC):
    """
    Abstract base for a resource shared across *different* sensor types, owned by ``SensorManager``. A sensor type
    declares the context it consumes as the second ``Sensor[Options, Context, Metadata, Data]`` parameter (``None``
    when it has none); every type declaring the same context class resolves to the one instance the manager owns.

    Distinct from ``SharedSensorMetadata``: metadata aggregates the per-sensor state of all sensors of a *single* type
    so one kernel can run over them (a batching optimization that grows with the number of sensors); a context is a
    *single* resource read by *several* sensor types, O(1) in the number of sensors (a sharing optimization). A context
    is purely an optimization: a sensor must produce identical results whether or not it is shared, so consistency stays
    ``SensorManager``'s responsibility, never the context's.

    The manager constructs the context with the sim at sensor-creation time, since the context must already exist to be
    handed to consuming sensors, but it stays an empty shell until a consumer activates it.

    - ``activate`` - a consuming sensor calls this from its own ``build`` (must be idempotent). The first call
      constructs the resource on the spot; the scene geometry is available by then. Inactive contexts stay empty shells
      and pay nothing. There is no separate manager-driven build: activation does the construction.
    - ``update`` - the manager calls this once per step before the per-type update loop; a no-op when inactive.
    - ``reset`` / ``destroy`` - manager-driven on ``scene.reset()`` and teardown.

    Querying an inactive context (e.g. reading its resource) must raise: only consumers that activated it may read it.
    Subclasses must implement every lifecycle method; ``update`` / ``reset`` / ``destroy`` guard themselves on
    ``is_active``.
    """

    def __init__(self, sim):
        self._sim = sim
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @abstractmethod
    def activate(self) -> None:
        """Declare the context active (a consumer needs it) and construct the resource; must be idempotent. Called from
        a consuming sensor's ``build``, when the scene geometry is available."""

    @abstractmethod
    def update(self) -> None:
        """Refresh the resource for the current step; manager-driven once per step. Must no-op when inactive."""

    @abstractmethod
    def reset(self, envs_idx) -> None:
        """Reset the resource; manager-driven on ``scene.reset()``. Must no-op when inactive."""

    @abstractmethod
    def destroy(self) -> None:
        """Release any resources held by the context; manager-driven on teardown."""


SharedSensorMetadataT = TypeVar("SharedSensorMetadataT", bound=SharedSensorMetadata)
OptionsT = TypeVar("OptionsT", bound="SensorOptions")
DataT = TypeVarWithDefault("DataT", default=tuple, covariant=True)
# Second ``Sensor[...]`` parameter: the cross-type shared context. No default - declare ``None`` explicitly when the
# sensor type has no shared context (``SensorManager`` then passes ``None`` to the per-step hooks for that type).
SharedSensorContextT = TypeVar("SharedSensorContextT")


class Sensor(RBC, Generic[OptionsT, SharedSensorContextT, SharedSensorMetadataT, DataT]):
    """
    Base class for all types of sensors.

    To create a sensor, prefer using `scene.add_sensor(options)` instead of instantiating this class directly.

    Each concrete sensor class declares its associated options, metadata, and data types via Generic type parameters::

        class MySensor(Sensor[MyOptions, MyContext, MyMetadata, MyData]):
            ...  # 2nd param is the shared context (use ``None`` if none); DataT defaults to tuple

    Note
    -----
    The Sensor system is designed to be performant. All sensors of the same type are updated at once and stored
    in a cache in SensorManager. Cache size is inferred from the return format and cache length of each sensor.
    `read()` and `read_ground_truth()`, the public-facing methods of every Sensor, automatically handles indexing into
    the shared cache to return the correct data.
    """

    _options_cls: ClassVar[type]
    _metadata_cls: ClassVar[type]
    _return_data_cls: ClassVar[type] = tuple
    # Cross-type shared context class declared as the second ``Sensor[...]`` parameter; ``NoneType`` (declared as
    # ``None``) means this sensor type consumes no shared context.
    _shared_context_cls: ClassVar[type] = type(None)
    # Whether instances of this class participate in the ring-based per-step pipeline (delay sampling, transform
    # recurrence, history snapshots). Drives allocation of the GT + measured timeline rings in `SensorManager.build`.
    # Subclasses whose `_update_shared_cache` bypasses the rings (e.g. cameras handling rendering lazily) explicitly set
    # this to ``False``.
    uses_ring_pipeline: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is not None and issubclass(origin, Sensor):
                args = get_args(base)
                if len(args) >= 1 and not isinstance(args[0], TypeVar):
                    cls._options_cls = args[0]
                if len(args) >= 2 and not isinstance(args[1], TypeVar):
                    cls._shared_context_cls = args[1]
                if len(args) >= 3 and not isinstance(args[2], TypeVar):
                    cls._metadata_cls = args[2]
                if len(args) >= 4 and not isinstance(args[3], TypeVar):
                    cls._return_data_cls = args[3]
                break
        # Strict contract: overriding `_post_process` requires overriding `_get_intermediate_format` and/or
        # `_get_intermediate_dtype`. The intermediate buffer must be a distinct buffer regardless of whether its
        # shape/dtype happen to coincide with the return space (the timeline ring is in intermediate space; mixing data
        # spaces breaks `_apply_transform` filter overrides that read previous slots). When the intermediate shape and
        # dtype both coincide with return, override one method as a no-op to make the structural distinction explicit.
        if "_post_process" in cls.__dict__ and not (
            "_get_intermediate_format" in cls.__dict__ or "_get_intermediate_dtype" in cls.__dict__
        ):
            raise TypeError(
                f"{cls.__name__} overrides `_post_process` but neither `_get_intermediate_format` nor "
                f"`_get_intermediate_dtype`; declare the intermediate buffer explicitly (no-op override returning "
                f"the return-space value is acceptable when they coincide)."
            )
        # Auto-register if this class defines its own options (not inherited). Enforce that concrete sensor classes also
        # specify the metadata type parameter.
        if "_options_cls" in cls.__dict__:
            if "_metadata_cls" not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must specify Sensor[OptionsT, ContextT, MetadataT, DataT=tuple].")
            from .sensor_manager import SensorManager

            SensorManager.SENSOR_TYPES_MAP[cls._options_cls] = cls

    def __init__(
        self,
        options: "SensorOptions",
        idx: int,
        shared_context: SharedSensorContextT,
        shared_metadata: SharedSensorMetadataT,
        manager: "SensorManager",
    ):
        self._options: "SensorOptions" = options
        self._idx: int = idx
        # The per-type metadata and cross-type context a sensor needs are passed in explicitly so sensors never
        # introspect the manager's registries. The manager itself comes last as it is only a backdoor for debug / fast
        # prototyping (and so it can be dropped from the signature later).
        self._manager: "SensorManager" = manager
        # The cross-type shared context instance, or ``None`` when this sensor type declares no context. Reachable from
        # instance methods (build / debug); the per-step classmethod hooks receive it as the ``shared_context`` argument.
        self._shared_context: SharedSensorContextT = shared_context
        self._shared_metadata: SharedSensorMetadataT = shared_metadata
        self._is_built = False

        # Classes that opt out of the ring pipeline (e.g. cameras handling rendering lazily on read) cannot honor delay
        # / jitter / history because those features depend on the per-class return-space ring. Reject the inputs at
        # construction so the user picks a different sensor or drops the option rather than silently getting no-ops.
        if not self.uses_ring_pipeline:
            if options.delay > 0.0:
                gs.raise_exception(f"{type(self).__name__} does not support `delay`; got delay={options.delay}.")
            if options.jitter > 0.0:
                gs.raise_exception(f"{type(self).__name__} does not support `jitter`; got jitter={options.jitter}.")
            if options.history_length > 0:
                gs.raise_exception(
                    f"{type(self).__name__} does not support `history_length`; got "
                    f"history_length={options.history_length}."
                )

        self._dt = self._manager._sim.dt
        self._delay_ts = round(self._options.delay / self._dt)

        self._cache_slices: list[slice] = []
        return_format = self._get_return_format()
        assert len(return_format) > 0
        intrinsic_shapes: tuple[tuple[int, ...], ...] = (
            (return_format,) if isinstance(return_format[0], int) else return_format
        )

        history_length = self._options.history_length
        self._cache_size = 0
        self._read_flat_slices: list[slice] = []
        read_off = 0
        for shape in intrinsic_shapes:
            data_size = np.prod(shape)
            self._cache_slices.append(slice(self._cache_size, self._cache_size + data_size))
            self._cache_size += data_size

            span = data_size * history_length if history_length > 0 else data_size
            self._read_flat_slices.append(slice(read_off, read_off + span))
            read_off += span

        if history_length > 0:
            self._return_shapes = tuple((history_length, *s) for s in intrinsic_shapes)
        else:
            self._return_shapes = intrinsic_shapes

        # Element offset within the per-class cache; initialized by SensorManager during build
        self._cache_offset: int = -1

    # =============================== methods to implement ===============================

    def build(self):
        """
        Build the sensor.

        This method is called by SensorManager during the scene build phase. This is where any shared metadata should be
        initialized.
        """
        self._shared_metadata.delays_ts = concat_with_tensor(
            self._shared_metadata.delays_ts, self._delay_ts, expand=(self._manager._sim._B, 1), dim=1
        )
        self._shared_metadata.cache_sizes.append(self._cache_size)
        self._shared_metadata.history_lengths.append(self._options.history_length)
        if self._delay_ts > 0:
            self._shared_metadata.has_any_delay = True

    @classmethod
    def reset(cls, shared_metadata: SharedSensorMetadataT, shared_ground_truth_cache: torch.Tensor, envs_idx):
        """
        Reset the sensor.

        This method is called by SensorManager when the scene is reset by `scene.reset()`.

        Parameters
        ----------
        shared_metadata : SharedSensorMetadata
            The shared metadata for the sensor class.
        shared_ground_truth_cache : torch.Tensor
            The shared ground truth cache for the sensor class.
        envs_idx: array_like
            The indices of the environments to reset. The envs_idx should already be sanitized by SensorManager.
        """
        pass

    def _get_return_format(self) -> tuple[int | tuple[int, ...], ...]:
        """
        Shape(s) of what ``read()`` returns; instance method because the shape may depend on options.

        Sensor options are free to affect the returned shape (Raycaster's pattern, Camera's resolution, Proximity's
        probe positions, etc.) - this is supported by design. Returns a single tuple ``(N,)`` for a single-tensor
        return, or a tuple-of-tuples ``((3,), (3,), (3,))`` for a multi-tensor return (e.g. IMU's ``NamedTuple(lin_acc,
        ang_vel, mag)``).
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `_get_return_format()`.")

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        """
        Dtype of what ``read()`` returns; classmethod because the dtype is class-uniform across all instances.

        The manager allocates one per-dtype intermediate cache buffer and uses a per-class slice within it; if instances
        of the same class returned different dtypes, the per-class slice would no longer be a single contiguous range,
        breaking the per-class batched ``_update_shared_cache`` and ``_apply_transform`` contract. Dtype is therefore
        class-uniform by design.
        """
        raise NotImplementedError(f"{cls.__name__} has not implemented `_get_cache_dtype()`.")

    def _get_intermediate_format(self) -> tuple[int | tuple[int, ...], ...]:
        """
        Shape(s) of the pipeline-internal cache; defaults to ``_get_return_format()``.

        Override together with ``_post_process`` when the projection changes shape. Same instance-method semantics as
        ``_get_return_format``: the shape may depend on options.
        """
        return self._get_return_format()

    @classmethod
    def _get_intermediate_dtype(cls) -> torch.dtype:
        """
        Dtype of the pipeline-internal cache; defaults to ``_get_cache_dtype()``.

        Override together with ``_post_process`` when the projection changes dtype (e.g. ContactSensor's float
        intermediate vs. bool return). Same class-uniform semantics as ``_get_cache_dtype``.
        """
        return cls._get_cache_dtype()

    @classmethod
    def _update_shared_cache(
        cls,
        shared_context: SharedSensorContextT,
        shared_metadata: SharedSensorMetadataT,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer | None",
        intermediate_cache: torch.Tensor,
    ):
        """
        Compute one step of sensor data into the shared caches up to the per-step working buffer.

        Updates the shared ground-truth cache slice (shape ``(cols, B)``, C-contiguous rows), the GT timeline ring
        (``ground_truth_data_timeline.at(0)`` is the current GT write slot, post-transform), the measured timeline ring
        (``measured_data_timeline.at(0)`` is the current measured write slot, post-physics-imperfections /
        post-transform / PRE-hardware-imperfections), and the per-dtype ``intermediate_cache`` (shape ``(B, cols)``, the
        per-step measured working buffer in intermediate space: post-HW-imperfections, pre-``_post_process``,
        pre-delay-sample). When the sensor opts out of the ring pipeline (e.g. Camera), both timeline rings are ``None``
        and the implementation writes directly to ``intermediate_cache``. The manager handles ``_post_process``
        projection, return-space ring writes, and delay sampling after this hook returns.
        """
        raise NotImplementedError(f"{cls.__name__} has not implemented `update_shared_cache()`.")

    @classmethod
    def _apply_delay(
        cls, shared_metadata: SharedSensorMetadataT, return_ring: "TensorRingBuffer", return_cache: torch.Tensor
    ):
        """
        Sample stale slots of the measured return-space ring into the user-visible measured return cache.

        Default implementation: per-sensor zero-order-hold (ZOH) lookup at ``delay + jitter`` steps back. ZOH is the
        only sampling strategy that is dtype-safe for arbitrary return types (bool, int, uint8, quantized float), which
        is why it is the default. Override on the sensor class if you have a return space where a smoother sampling rule
        is appropriate (e.g. linear interpolation between adjacent slots for a continuous-valued sensor whose return
        dtype is float).

        ``return_ring`` is the per-class measured return-space ring (slot 0 = current step's post-everything value;
        slots 1.. are previous steps in increasing age). ``return_cache`` is the per-class measured return cache to
        populate; it is in return space (same shape and dtype as the ring).
        """
        if not shared_metadata.has_any_delay and not shared_metadata.has_any_jitter:
            # Fast path: no per-sensor delay loop, just copy the most recent slot class-wide.
            return_cache.copy_(return_ring.at(0, copy=False))
            return

        if shared_metadata.has_any_jitter:
            # Uniform jitter in [0, jitter_ts) per env per sensor. Combined with the `jitter <= delay` and `jitter < dt`
            # option constraints, the effective per-step shift cannot wrap the ring.
            cur_jitter_ts = torch.rand_like(shared_metadata.jitter_ts).mul_(shared_metadata.jitter_ts)
        else:
            cur_jitter_ts = None

        tensor_start = 0
        for sensor_idx, tensor_size in enumerate(shared_metadata.cache_sizes):
            cur_delay_ts = shared_metadata.delays_ts[:, sensor_idx]
            if cur_jitter_ts is not None:
                # Probabilistic rounding of the continuous-time delay onto integer ring slots: with `jitter < dt` (one
                # slot), the realized jitter sample `j` is in `[0, 1)`; adding `uniform[0, 1)` then flooring picks the
                # next slot (`D + 1`) with probability `j`, preserving the expected jitter shift while staying
                # dtype-safe (no interpolation between adjacent slots).
                cur_delay_ts = (
                    cur_delay_ts + cur_jitter_ts[:, sensor_idx] + torch.rand_like(cur_jitter_ts[:, sensor_idx])
                )
            cur_delay_ts_int = cur_delay_ts.to(dtype=torch.int64)
            tensor_slice = slice(tensor_start, tensor_start + tensor_size)
            return_cache[:, tensor_slice].copy_(return_ring.at(cur_delay_ts_int, tensor_slice, per_row=True))
            tensor_start += tensor_size

    @classmethod
    def _post_process(
        cls,
        shared_metadata: SharedSensorMetadataT,
        tensor: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ) -> torch.Tensor:
        """
        Project from intermediate space to return space. Applied once per branch per step.

        ``tensor`` is the full per-class intermediate cache ``[B, total_cache_size]`` (post-physics / post-transform /
        post-hardware for measured; post-transform for GT). Return the post-cast value (any return-space dtype); the
        orchestrator writes the return into slot 0 of the per-class return-space ring, and the user-visible read is then
        produced by delay-sampling the ring. ``timeline`` is that ring (post-``_post_process`` snapshots); the return
        ring is rotated AFTER this call returns, so during the override ``timeline.at(0)`` is the previous step's
        post-output (the most recent valid value), ``timeline.at(1)`` is the step before that, and so on.
        ``is_measured`` is ``True`` on the measured branch call and ``False`` on the GT branch call, so an override can
        apply readout-stage contributions on only one side.

        Designed for cast / clamp / threshold / mask / deadband / simple reductions and (optionally) stateful HW
        responses that should not contaminate ``_apply_transform`` recurrence. Default: identity (return ``tensor``
        unchanged - valid because the strict-override contract enforces matching intermediate / return dtypes when
        ``_post_process`` is not overridden).
        """
        return tensor

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw debug shapes for the sensor in the scene.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `draw_debug()`.")

    # =============================== public shared methods ===============================

    @gs.assert_built
    def read(self, envs_idx=None) -> DataT:
        """
        Read the sensor data (with noise applied if applicable).

        Pure view into the per-class return cache (post-``_post_process``); ``_post_process`` was applied eagerly once
        per step by the orchestrator.
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self), envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx=None) -> DataT:
        """
        Read the ground truth sensor data (without noise). Pure view into the per-class ground-truth return cache.
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self, is_ground_truth=True), envs_idx)

    @gs.assert_unbuilt
    def start_recording(self, rec_options: "RecorderOptions") -> "Recorder":
        """
        Automatically read and process sensor data. See RecorderOptions for more details.

        Data from `sensor.read()` is used. If the sensor data needs to be preprocessed before passing to the recorder,
        consider using `scene.add_recorder()` instead with a custom data function.

        Parameters
        ----------
        rec_options : RecorderOptions
            The options for the recording.
        """
        return self._manager._sim._scene._recorder_manager.add_recorder(self.read, rec_options)

    @property
    def is_built(self) -> bool:
        return self._is_built

    # =============================== private shared methods ===============================

    def _get_formatted_data(self, tensor: torch.Tensor, envs_idx=None) -> torch.Tensor:
        """
        Returns tensor(s) matching the return format.

        Note that this method does not clone the data tensor, it should have been cloned by the caller.
        """
        envs_idx = self._sanitize_envs_idx(envs_idx)

        return_values = []
        tensor_chunk = tensor[envs_idx].reshape((len(envs_idx), -1))

        for i, shape in enumerate(self._return_shapes):
            sl = self._read_flat_slices[i]
            field_data = tensor_chunk[..., sl].reshape((len(envs_idx), *shape))
            if self._manager._sim.n_envs == 0:
                field_data = field_data[0]
            return_values.append(field_data)

        if len(return_values) == 1:
            return return_values[0]
        return self._return_data_cls(*return_values)

    def _sanitize_envs_idx(self, envs_idx) -> torch.Tensor:
        return self._manager._sim._scene._sanitize_envs_idx(envs_idx)

    def _set_metadata_field(self, value, field, field_start, field_size, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        if field.ndim == 2:
            # Flat field structure: per-sensor spans may differ in size (e.g. cache-sized imperfection fields), so the
            # caller provides this sensor's start rather than a uniform stride.
            index_slice = slice(field_start, field_start + field_size)
        else:
            # per sensor field structure
            index_slice = field_start

        field[:, index_slice] = broadcast_tensor(value, field.dtype, (len(envs_idx), field_size), ("envs_idx", ""))


class _SolverLinkGroup(NamedTuple):
    """Per-solver bucket: (solver, in-solver link indices, sensor columns)."""

    solver: "KinematicSolver"
    links_idx: torch.Tensor  # solver-local link indices, one per sensor in this group
    sensor_cols: torch.Tensor  # which per-class sensor column each link pose lands in


@dataclass
class KinematicSensorMetadataMixin:
    """
    Shared metadata for sensors attached to a KinematicEntity (or any subclass, including RigidEntity).

    Sensors are bucketed at build time into per-solver ``_SolverLinkGroup`` entries so the per-step gather is one bulk
    read per solver. Static sensors (``entity_idx<0``) are not bucketed and keep an identity link pose, leaving the
    kernel to apply ``pos_offset`` / ``euler_offset`` in world frame.
    """

    offsets_pos: torch.Tensor = make_tensor_field((0, 0, 3))
    offsets_quat: torch.Tensor = make_tensor_field((0, 0, 4))
    solver_groups: list[_SolverLinkGroup] = field(default_factory=list)

    @property
    def n_sensors(self) -> int:
        return self.offsets_pos.shape[1]


@dataclass
class RigidSensorMetadataMixin:
    """
    Base shared metadata class for sensors that are attached to a RigidEntity.
    """

    solver: "RigidSolver | None" = None
    links_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    offsets_pos: torch.Tensor = make_tensor_field((0, 0, 3))
    offsets_quat: torch.Tensor = make_tensor_field((0, 0, 4))


RigidSensorMetadataMixinT = TypeVar("RigidSensorMetadataMixinT", bound=RigidSensorMetadataMixin)
KinematicSensorMetadataMixinT = TypeVar("KinematicSensorMetadataMixinT", bound=KinematicSensorMetadataMixin)


class _LinkAttachedSensorMixin:
    """
    Common boilerplate for sensors attached to a link.

    Holds the python-side ``_link`` reference, concatenates per-sensor pos/euler offsets into shared metadata at build
    time, and exposes ``set_{pos,quat}_offset``. Subclasses implement ``_register_link`` to record the link mapping in
    solver-specific shared-metadata shape (single tensor for ``RigidSensorMixin``, per-solver buckets for
    ``KinematicSensorMixin``).
    """

    _link: "RigidLink | None" = None

    def build(self):
        super().build()

        batch_size = self._manager._sim._B
        if self._options.entity_idx >= 0:
            entity = self._manager._sim.entities[self._options.entity_idx]
            self._link = entity.links[self._options.link_idx_local]
            link_idx = self._options.link_idx_local + entity.link_start
            self._register_link(entity, link_idx)

        self._shared_metadata.offsets_pos = concat_with_tensor(
            self._shared_metadata.offsets_pos, self._options.pos_offset, expand=(batch_size, 1, 3), dim=1
        )
        self._shared_metadata.offsets_quat = concat_with_tensor(
            self._shared_metadata.offsets_quat,
            euler_to_quat([self._options.euler_offset]),
            expand=(batch_size, 1, 4),
            dim=1,
        )

    def _register_link(self, entity, link_idx: int):
        raise NotImplementedError

    @gs.assert_built
    def set_pos_offset(self, pos_offset, envs_idx=None):
        self._set_metadata_field(pos_offset, self._shared_metadata.offsets_pos, self._idx, 3, envs_idx)

    @gs.assert_built
    def set_quat_offset(self, quat_offset, envs_idx=None):
        self._set_metadata_field(quat_offset, self._shared_metadata.offsets_quat, self._idx, 4, envs_idx)


class RigidSensorMixin(_LinkAttachedSensorMixin, Generic[RigidSensorMetadataMixinT]):
    """Base sensor class for sensors that are attached to a RigidEntity."""

    def build(self):
        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver
        super().build()

    def _register_link(self, entity, link_idx: int):
        self._shared_metadata.links_idx = concat_with_tensor(self._shared_metadata.links_idx, link_idx)


class KinematicSensorMixin(_LinkAttachedSensorMixin, Generic[KinematicSensorMetadataMixinT]):
    """
    Base sensor class for sensors that may attach to entities across solvers (rigid or kinematic).

    Bucketing into ``shared_metadata.solver_groups`` happens at build time so the per-step gather is one bulk read per
    solver.
    """

    def _register_link(self, entity, link_idx: int):
        sensor_col = self._shared_metadata.n_sensors
        groups = self._shared_metadata.solver_groups
        existing = next((i for i, g in enumerate(groups) if g.solver is entity.solver), None)
        if existing is None:
            groups.append(
                _SolverLinkGroup(
                    solver=entity.solver,
                    links_idx=concat_with_tensor(torch.empty(0, device=gs.device, dtype=gs.tc_int), link_idx),
                    sensor_cols=concat_with_tensor(torch.empty(0, device=gs.device, dtype=gs.tc_int), sensor_col),
                )
            )
        else:
            group = groups[existing]
            groups[existing] = _SolverLinkGroup(
                solver=group.solver,
                links_idx=concat_with_tensor(group.links_idx, link_idx),
                sensor_cols=concat_with_tensor(group.sensor_cols, sensor_col),
            )


class SimpleSensor(Sensor[OptionsT, SharedSensorContextT, SharedSensorMetadataT, DataT]):
    """
    Base class for sensors that use the standard per-step pipeline.

    Pipeline (per branch, in execution order):

    - GT branch: ``raw -> _apply_transform(is_measured=False) -> _post_process(is_measured=False) -> ground truth``.
    - Measured branch: ``raw -> _apply_physics_imperfections -> _apply_transform(is_measured=True) ->
      _apply_hardware_imperfections -> _post_process(is_measured=True) -> delay sampling -> measured``.

    ``_update_raw_data`` and ``_apply_physics_imperfections`` are packaged inside ``_update_current_timestep_data``;
    override the latter to fuse them in a single kernel pass.

    Both branches keep their own intermediate-space timeline ring (``ground_truth_data_timeline`` /
    ``measured_data_timeline``). The timeline rings store post-transform, PRE-hardware-imperfections data, so
    ``_apply_transform`` recurrence reads clean previous slots and stateful filters (e.g. thermal dissipation) are not
    contaminated by hardware noise. Hardware imperfections mutate a per-step working buffer (the intermediate cache),
    never a timeline ring. The post-``_post_process`` snapshot of that working buffer is then frozen into slot 0 of the
    per-class return-space ring, and delay sampling reads stale slots of the return-space ring to produce the
    user-visible value - so each delayed read returns the post-everything signal observed at the step of capture.

    Concrete sensors override hooks (``_update_raw_data``, ``_update_current_timestep_data``,
    ``_apply_physics_imperfections``, ``_apply_transform``, ``_apply_hardware_imperfections``, ``_post_process``) rather
    than ``_update_shared_cache`` itself.

    History reads gather post-everything snapshots from the per-class return-space ring, so ``read(history_length=N)``
    returns the final measured values that were observed at each past step. ``_post_process`` is eager (applied once per
    branch per step by the orchestrator).
    """

    uses_ring_pipeline: ClassVar[bool] = True

    @gs.assert_built
    def set_resolution(self, resolution, envs_idx=None):
        self._set_metadata_field(
            resolution, self._shared_metadata.resolution, self._cache_offset, self._cache_size, envs_idx
        )
        self._shared_metadata.has_any_resolution = bool((self._shared_metadata.resolution > gs.EPS).any().item())

    @gs.assert_built
    def set_bias(self, bias, envs_idx=None):
        self._set_metadata_field(bias, self._shared_metadata.bias, self._cache_offset, self._cache_size, envs_idx)
        self._shared_metadata.has_any_bias = bool((self._shared_metadata.bias != 0).any().item())

    @gs.assert_built
    def set_random_walk(self, random_walk, envs_idx=None):
        self._set_metadata_field(
            random_walk, self._shared_metadata.random_walk, self._cache_offset, self._cache_size, envs_idx
        )
        self._shared_metadata.has_any_random_walk = bool((self._shared_metadata.random_walk > gs.EPS).any().item())

    @gs.assert_built
    def set_noise(self, noise, envs_idx=None):
        self._set_metadata_field(noise, self._shared_metadata.noise, self._cache_offset, self._cache_size, envs_idx)
        self._shared_metadata.has_any_noise = bool((self._shared_metadata.noise > gs.EPS).any().item())

    @gs.assert_built
    def set_jitter(self, jitter, envs_idx=None):
        jitter_np = np.asarray(jitter, dtype=gs.np_float)
        if np.any(jitter_np < 0):
            gs.raise_exception(f"Sensor jitter must be non-negative; got jitter={tuple(jitter_np.ravel())}.")
        if np.any(jitter_np >= self._dt + gs.EPS):
            gs.raise_exception(
                f"Sensor jitter must not exceed the simulation step dt={self._dt}; got "
                f"jitter={tuple(jitter_np.ravel())}."
            )
        # Same bound as `SensorOptions.model_post_init`, enforced here because only a sensor declaring a delay at build
        # time gets the ring slot a jittered read reaches (see `cls_delay_depth` in sensor_manager.py).
        if np.any(jitter_np > self._options.delay):
            gs.raise_exception(
                f"Sensor jitter must not exceed the read delay={self._options.delay}; got "
                f"jitter={tuple(jitter_np.ravel())}."
            )
        self._set_metadata_field(jitter_np / self._dt, self._shared_metadata.jitter_ts, self._idx, 1, envs_idx)
        # Recompute the slow-path flag from the freshly-written class metadata. One GPU->CPU sync at setter call time;
        # setters are not hot path. The check covers partial envs_idx writes and other sensors.
        self._shared_metadata.has_any_jitter = bool((self._shared_metadata.jitter_ts > gs.EPS).any().item())

    def build(self):
        """
        Initialize all shared metadata needed to update all noisy sensors.

        Time-related state (``delays_ts``, ``jitter_ts``) is pushed by ``Sensor.build()``; this method adds the
        imperfection-parameter state.
        """
        super().build()
        to_tuple = partial(_to_tuple, length_per_value=self._cache_size)

        batch_size = self._manager._sim._B

        # Jitter must not exceed the step, so a read shifts by at most one extra ring slot - the margin the return-space
        # ring is sized for (see `cls_delay_depth` in sensor_manager.py). An EPS slack lets `jitter == dt` pass cleanly
        # despite float quantization.
        jitter_np = np.asarray(self._options.jitter, dtype=gs.np_float)
        if np.any(jitter_np >= self._dt + gs.EPS):
            gs.raise_exception(
                f"Sensor jitter must not exceed the simulation step dt={self._dt}; got "
                f"jitter={tuple(jitter_np.ravel())}."
            )

        self._shared_metadata.resolution = concat_with_tensor(
            self._shared_metadata.resolution, to_tuple(self._options.resolution), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.bias = concat_with_tensor(
            self._shared_metadata.bias, to_tuple(self._options.bias), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.random_walk = concat_with_tensor(
            self._shared_metadata.random_walk, to_tuple(self._options.random_walk), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata._cur_random_walk = torch.zeros_like(self._shared_metadata.random_walk)
        self._shared_metadata.noise = concat_with_tensor(
            self._shared_metadata.noise, to_tuple(self._options.noise), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.jitter_ts = concat_with_tensor(
            self._shared_metadata.jitter_ts, to_tuple(self._options.jitter / self._dt), expand=(batch_size, -1), dim=-1
        )
        if np.any(jitter_np > gs.EPS):
            self._shared_metadata.has_any_jitter = True
        if np.any(np.asarray(self._options.noise, dtype=gs.np_float) > gs.EPS):
            self._shared_metadata.has_any_noise = True
        if np.any(np.asarray(self._options.random_walk, dtype=gs.np_float) > gs.EPS):
            self._shared_metadata.has_any_random_walk = True
        if np.any(np.asarray(self._options.bias, dtype=gs.np_float) != 0):
            self._shared_metadata.has_any_bias = True
        if np.any(np.asarray(self._options.resolution, dtype=gs.np_float) > gs.EPS):
            self._shared_metadata.has_any_resolution = True

    @classmethod
    def reset(cls, shared_metadata: SharedSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        shared_metadata._cur_random_walk[envs_idx, ...].fill_(0.0)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_context: SharedSensorContextT,
        shared_metadata: SharedSensorMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer | None",
        intermediate_cache: torch.Tensor,
    ):
        # Both branches share the same raw signal. The GT and measured timeline rings (paired, same size, shared
        # rotation idx) store post-transform, PRE-hardware-imperfections data; `_apply_transform` reads previous ring
        # slots cleanly and hardware imperfections never write back to the ring, so transform recurrence stays clean.

        if measured_data_timeline is None:
            # No measured pipeline for this dtype (only non-SimpleSensor classes); shouldn't happen for SimpleSensor
            # instances but keep the path correct: raw GT -> intermediate cache.
            cls._update_raw_data(shared_context, shared_metadata, current_ground_truth_data_T)
            intermediate_cache.copy_(current_ground_truth_data_T.T)
        else:
            gt_slot_0 = ground_truth_data_timeline.at(0, copy=False)
            measured_slot_0 = measured_data_timeline.at(0, copy=False)

            # Raw signal and measured-only physics imperfections in one hook so an override can fuse them in a single
            # kernel pass. Default writes raw GT to slot 0 of both rings and then applies `_apply_physics_imperfections`
            # in place on the measured ring slot only - GT keeps the raw simulated phenomenon, measured carries the
            # noised value.
            cls._update_current_timestep_data(
                shared_context,
                shared_metadata,
                current_ground_truth_data_T,
                ground_truth_data_timeline,
                measured_data_timeline,
            )

            # GT branch transform. `is_measured=False` lets sensor-element-specific effects (RC filter, mechanical
            # bandwidth) skip on the GT path while branch-symmetric coordinate transforms still run.
            cls._apply_transform(shared_metadata, gt_slot_0, ground_truth_data_timeline, is_measured=False)
            current_ground_truth_data_T.copy_(gt_slot_0.T)

            # Measured branch transform - same hook, on the measured ring with `is_measured=True`. Recurrence is
            # independent of the GT branch because each branch has its own timeline ring.
            cls._apply_transform(shared_metadata, measured_slot_0, measured_data_timeline, is_measured=True)

            # Copy post-transform value from the measured ring slot 0 into the per-step intermediate cache (the working
            # buffer), then apply hardware imperfections in place. The ring stays clean of HW noise so
            # `_apply_transform` recurrence next step sees uncontaminated previous slots; the working buffer holds the
            # per-step post-HW value that the orchestrator will project via `_post_process` and write into the
            # return-space ring slot 0. Delay sampling reads from the return-space ring, so each delayed slot carries
            # its own frozen noise sample (embedded-sampler semantics).
            intermediate_cache.copy_(measured_slot_0)
            cls._apply_hardware_imperfections(shared_metadata, intermediate_cache)

        # `_post_process`, write to return ring slot 0, and delay sampling are handled by the manager after this hook
        # returns.

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_context: SharedSensorContextT,
        shared_metadata: SharedSensorMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        """
        Pack the raw signal and measured-only physics imperfections into one hook.

        Default behavior: compute raw GT into ``current_ground_truth_data_T`` (shape ``(cols, B)``, C-contiguous, the
        kernel-friendly target) via ``_update_raw_data``, mirror it into slot 0 of the GT and measured timeline rings,
        then call ``_apply_physics_imperfections`` in place on the measured ring slot. Override this method to fuse
        ``_update_raw_data`` and ``_apply_physics_imperfections`` in a single kernel pass: write the raw GT to
        ``current_ground_truth_data_T`` and to the GT ring slot, and write the noised value directly to the measured
        ring slot.
        """
        cls._update_raw_data(shared_context, shared_metadata, current_ground_truth_data_T)
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured_slot_0 = measured_data_timeline.at(0, copy=False)
        measured_slot_0.copy_(current_ground_truth_data_T.T)
        cls._apply_physics_imperfections(shared_metadata, measured_slot_0, measured_data_timeline)

    @classmethod
    def _apply_physics_imperfections(
        cls, shared_metadata: SharedSensorMetadata, data: torch.Tensor, timeline: "TensorRingBuffer"
    ):
        """
        Apply physics-level perturbations in place on the current measured-timeline slot, BEFORE ``_apply_transform``.

        Physics-level means random fluctuations of the underlying physical phenomenon the simulator does not model
        (genuine drift, random walk of the quantity, fine-scale turbulence on top of the deterministic field, etc.).
        These shape what the sensor *sees* beyond the simulated GT, but they are NOT the sensor element's response
        (thermal mass / RC time constant, mechanical bandwidth -> ``_apply_transform`` with ``is_measured=True``) and
        they are NOT the sensor's electronics (ADC, ethercat, embedded buffering -> ``_apply_hardware_imperfections``).
        Measured-only by construction (GT keeps the raw simulated phenomenon).

        ``data IS timeline.at(0)`` (the measured ring's slot 0). Stateful overrides read previous slots with
        ``timeline.at(1)``, etc. Default: no-op. Sensors that fuse this with ``_update_raw_data`` in a single kernel
        should override ``_update_current_timestep_data`` instead of this hook.
        """

    @classmethod
    def _update_raw_data(
        cls, shared_context: SharedSensorContextT, shared_metadata: SharedSensorMetadata, raw_data_T: torch.Tensor
    ):
        """Sensor-specific kernel computing raw data into ``raw_data_T`` (shape ``(cols, B)``)."""
        raise NotImplementedError(f"{cls.__name__} has not implemented `_update_raw_data()`.")

    @classmethod
    def _apply_transform(
        cls,
        shared_metadata: SharedSensorMetadata,
        data: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ):
        """
        Pre-acquisition transform + optional stateful filter; mutates ``data`` in place.

        Receives ``data`` as a batch-first view ``[B, cache_size, ...]`` (the current slot 0 of ``timeline``) and must
        mutate it in place. ``timeline`` is the branch's ring - the GT ring on the GT branch call, the measured ring on
        the measured branch call - and is always non-``None``. Read previous slots with ``timeline.at(1)``,
        ``timeline.at(2)``, etc. for stateful filters. Ring contents are clean of hardware imperfections, so recurrence
        state never accumulates hardware noise.

        ``is_measured`` indicates which branch is currently active. The hook runs on both branches by default so
        branch-symmetric effects (coordinate transforms, frame change) happen uniformly. Gate on ``is_measured`` for
        sensor-element-specific pre-acquisition effects that must NOT appear in GT (RC time constant, mechanical
        bandwidth, etc.).
        """

    @classmethod
    def _apply_hardware_imperfections(cls, shared_metadata: SimpleSensorMetadata, measured_slot_0: torch.Tensor):
        """
        Apply SimpleSensor's imperfection model in-place on the per-step measured working buffer.

        Opinionated interpretation of the imperfection parameters (noise, bias, random_walk, resolution) as the
        perturbations introduced by the embedded sampling layer at the sensor output. Each contribution is gated by a
        precomputed Python bool flag (``has_any_*``) so sensor classes with all-zero values pay no GPU work.

        ``measured_slot_0`` is the per-dtype intermediate cache (the working buffer about to be projected by
        ``_post_process``), not a ring slot - mutations here are local to the current step and never bleed into
        ``_apply_transform`` recurrence. The post-projection result is written by the orchestrator into the return-space
        ring slot 0, so each delayed read picks up a frozen noise sample captured at that step.

        Designed for stateless per-step perturbations. Stateful HW responses (sensor-element bandwidth, signal-dependent
        gain with memory) belong in ``_post_process``, which sees the return-space ring and can read its previous slots.
        """
        if shared_metadata.has_any_random_walk:
            shared_metadata._cur_random_walk += torch.normal(0.0, shared_metadata.random_walk)
            measured_slot_0 += shared_metadata._cur_random_walk
        if shared_metadata.has_any_noise:
            measured_slot_0 += torch.normal(0.0, shared_metadata.noise)
        if shared_metadata.has_any_bias:
            measured_slot_0 += shared_metadata.bias
        if shared_metadata.has_any_resolution:
            resolution = shared_metadata.resolution
            mask = resolution > gs.EPS
            measured_slot_0[mask] = torch.round(measured_slot_0[mask] / resolution[mask]) * resolution[mask]
