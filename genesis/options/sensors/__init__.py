from .camera import *
from .options import *
from .options import Raycaster as Lidar
from .raycaster import *
from .tactile import *

from .options import SensorOptions


class _SensorTypesNamespace:
    """Lazy mapping from sensor-options class names to opaque integer tags. Use the tag returned by
    `gs.sensors.types.<Name>` as the key into `scene.read_sensors()` / `entity.read_sensors()`."""

    def __init__(self):
        self._ids: dict[str, int] = {}

    def __getattr__(self, name: str) -> int:
        if name.startswith("_"):
            raise AttributeError(name)
        cached = self._ids.get(name)
        if cached is not None:
            return cached
        import sys

        obj = vars(sys.modules[__name__]).get(name)
        if not isinstance(obj, type) or not issubclass(obj, SensorOptions):
            raise AttributeError(name)
        tid = len(self._ids)
        self._ids[name] = tid
        return tid


types = _SensorTypesNamespace()
