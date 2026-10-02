import importlib
import pkgutil
import sys
from typing import TYPE_CHECKING, ForwardRef, get_args, get_origin

import torch

import genesis as gs
from genesis.options.sensors import types as _sensor_types_namespace
from genesis.options.sensors.options import SensorOptions
from genesis.typing import EnvsIdxType
from genesis.utils.misc import indices_to_mask

from .base_sensor import Sensor, SensorArray, SharedSensorContext

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator
    from genesis.vis.rasterizer_context import RasterizerContext


class SensorManager:
    """
    The sensor arrays of a scene, one per sensor type, driven like the solvers of the simulator.

    `create_sensor` resolves the sensor class of the options, constructs the array of that type on its first sensor and
    hands the array the new handle. `get_context` constructs a shared context on its first request; the contexts are
    refreshed before the arrays at every step, so an array reads them current.
    """

    # Maps sensor options class -> sensor class for runtime dispatch.
    SENSOR_TYPES_MAP: dict[type[SensorOptions], type["Sensor"]] = {}

    def __init__(self, sim: "Simulator"):
        self._sim = sim
        self._arrays: dict[type[Sensor], SensorArray] = {}
        # Cross-type shared contexts, keyed by class so every array asking for the same context gets the one instance
        # (see SharedSensorContext)
        self._shared_contexts: dict[type[SharedSensorContext], SharedSensorContext] = {}

    def create_sensor(self, sensor_options: "SensorOptions") -> "Sensor":
        sensor_options.validate_scene(self._sim.scene)
        sensor_cls = SensorManager._resolve_sensor_cls(type(sensor_options))
        # One array per sensor type, constructed on its first sensor: two types built on the same array class get one
        # array each, and a bulk read keeps one entry per type
        array = self._arrays.get(sensor_cls)
        if array is None:
            array = sensor_cls._array_cls(self._sim, self)
            self._arrays[sensor_cls] = array
        sensor = sensor_cls(sensor_options, array)
        array.add_sensor(sensor)
        return sensor

    def get_context(self, context_cls: type[SharedSensorContext]) -> SharedSensorContext:
        """The shared context of the given class, constructed on its first request and shared by every array asking."""
        context = self._shared_contexts.get(context_cls)
        if context is None:
            context = context_cls(self._sim)
            self._shared_contexts[context_cls] = context
        return context

    @staticmethod
    def _resolve_sensor_cls(options_cls: type["SensorOptions"]) -> type["Sensor"]:
        """
        Resolve the sensor class of an options class.

        The registry is consulted first. On a miss, the options class must name its sensor type through its generic
        parameter, and the sibling modules of the options package are imported, which registers the sensor class.
        """
        sensor_cls = SensorManager.SENSOR_TYPES_MAP.get(options_cls)
        if sensor_cls is not None:
            return sensor_cls

        # Not registered yet: check that the options class specifies its sensor type, then try to discover it. The
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
                            except ImportError:
                                continue
                        if options_cls in SensorManager.SENSOR_TYPES_MAP:
                            return SensorManager.SENSOR_TYPES_MAP[options_cls]

        gs.raise_exception(
            f"No sensor class registered for {options_cls.__name__}. Ensure the sensor module is in the same "
            "package as the options module, or import the sensor class manually before calling add_sensor()."
        )

    def build(self) -> None:
        for array in self._arrays.values():
            array.build()
            array._is_built = True
            for sensor in array.sensors:
                sensor._is_built = True

    def destroy(self) -> None:
        for context in self._shared_contexts.values():
            context.destroy()
        self._shared_contexts.clear()
        for array in self._arrays.values():
            array.destroy()
        self._arrays.clear()

    def reset(self, envs_idx: EnvsIdxType | None = None) -> None:
        # A reset runs torch writes alone, so the selection travels as a mask (see indices_to_mask), one over every
        # environment when None
        envs_mask = indices_to_mask(slice(None) if envs_idx is None else envs_idx)
        # A reset may change otherwise-static geometry, so a context rebuilds before any array reads it again
        for context in self._shared_contexts.values():
            context.reset(envs_mask)
        for array in self._arrays.values():
            array.reset(envs_mask)

    def step(self) -> None:
        # Each shared context is refreshed once per step, before the arrays read it
        for context in self._shared_contexts.values():
            context.update()
        fps_tracker = self._sim.fps_tracker
        for array in self._arrays.values():
            fps_tracker.start_phase(f"sensors/{type(array.sensors[0]).__name__}")
            array.step()
        fps_tracker.stop_phase()

    def draw_debug(self, context: "RasterizerContext") -> None:
        for array in self._arrays.values():
            array.draw_debug(context)

    def read_sensors(
        self, entity_idx: int | None = None, envs_idx: EnvsIdxType | None = None, is_ground_truth: bool = False
    ) -> dict[int, torch.Tensor]:
        """
        Read every sensor of the scene, one fresh tensor per sensor type.

        Every tensor is fresh and independent of the internal sensor storage, so the caller is free to mutate it.

        Parameters
        ----------
        entity_idx : int | None
            - None (default): include every sensor in the scene.
            - k >= 0: include only sensors whose `entity_idx == k`.
            - -1: include only static sensors (those not attached to any entity).
        envs_idx : array-like | int | slice | None
            The environments to read, all of them when None.
        is_ground_truth : bool
            When True, return ground-truth tensors instead of measured tensors.

        Returns
        -------
        dict[int, torch.Tensor]
            For each sensor type present (keyed by its `gs.sensors.types` id), a tensor of shape
            (B, [history,] type_or_entity_cache_size). For sensors without history, the history dimension is omitted.
        """
        result: dict[int, torch.Tensor] = {}
        for array in self._arrays.values():
            tensor = array.read_all(entity_idx, envs_idx, is_ground_truth)
            if tensor is None:
                continue
            options_cls = type(array.sensors[0].options)
            type_id = getattr(_sensor_types_namespace, options_cls.__name__)
            result[type_id] = tensor
        return result

    def get_sensors_by_entity(self, entity_idx: int) -> "gs.List[Sensor]":
        """List of all sensors attached to the given entity (or static sensors for entity_idx == -1)."""
        target_eid = -1 if entity_idx < 0 else entity_idx
        return gs.List(sensor for sensor in self.sensors if sensor.options.entity_idx == target_eid)

    @property
    def sensors(self) -> list["Sensor"]:
        return gs.List([sensor for array in self._arrays.values() for sensor in array.sensors])
