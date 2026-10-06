"""Read-only localhost inspector for graph-native simulation engines.

The module has no side effects: call :func:`start_server` explicitly to serve
an engine, then call ``stop()`` on the returned handle or use it in a ``with``
statement.
"""

from .server import VisualizerServer, start_server, start_visualizer
from .snapshot import SCHEMA_VERSION, SnapshotBuilder, build_snapshot

__all__ = [
    "SCHEMA_VERSION",
    "SnapshotBuilder",
    "VisualizerServer",
    "build_snapshot",
    "start_server",
    "start_visualizer",
]
