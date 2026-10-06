from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def _load_visualizer():
    package_dir = Path(__file__).parents[2] / "genesis" / "engine" / "systems" / "visualizer"
    spec = importlib.util.spec_from_file_location(
        "_standalone_system_visualizer",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the system visualizer package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


visualizer = _load_visualizer()


class FakeBuffer:
    __slots__ = ("dtype", "shape")

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def to_numpy(self):
        raise AssertionError("Device values must not be read")


class NestedData:
    def __init__(self):
        self.mask = FakeBuffer((8,), "i32")
        self.parent = None


class FakeData:
    enabled: bool
    positions: FakeBuffer
    pending: FakeBuffer

    def __init__(self):
        self.enabled = True
        self.positions = FakeBuffer((8, 3), "f64")
        self.nested = NestedData()
        self.nested.parent = self
        self._private_value = "must not be exposed"

    @property
    def unavailable(self):
        raise RuntimeError("Partially built attribute")


class Data(FakeData):
    engine_state: FakeBuffer


def integrate_kernel():
    return None


def collision_kernel():
    return None


def step_graph():
    return None


def resume_collision(_status):
    return None


class FakeAction:
    def __init__(self, owner, data, kernel):
        self.owner = owner
        self.data = data if isinstance(data, tuple) else (data,)
        self.kernel = kernel


class FakeActionCollection:
    __slots__ = ("_actions", "_owner")

    def __init__(self, owner, actions):
        self._owner = owner
        self._actions = list(actions)

    @property
    def actions(self):
        raise RuntimeError("Collection is not available during build")


class FakePipeline:
    def __init__(self):
        self.data = object()
        self.graph = step_graph
        self._yield_callbacks = {7: resume_collision}
        self._has_launched = False


class IntegratorSystem:
    def __init__(self):
        self.data = FakeData()
        self.is_valid = True
        self.is_building = False
        self.iterations = 4
        self.scratch = FakeBuffer((2,), "f32")
        self._private_value = "must not be exposed"
        self.advance_action = FakeAction(self, self.data, integrate_kernel)
        self.actions = FakeActionCollection(self, [self.advance_action])

    def build(self):
        return None


class CollisionSystem:
    def __init__(self, integrator):
        self.data = FakeData()
        self.is_valid = True
        self.is_building = False
        self.integrator_system = integrator
        self.collision_action = FakeAction(self, (self.data, integrator.data), collision_kernel)

    def build(self):
        return None

    def dependencies(self):
        return {self.integrator_system: True}


class FakeEngine:
    def __init__(self):
        integrator = IntegratorSystem()
        collision = CollisionSystem(integrator)
        self.systems = {
            CollisionSystem: collision,
            IntegratorSystem: integrator,
        }
        self.is_built_host = True
        self.data = Data()
        self.data.engine_state = FakeBuffer((), "i32")
        self.monitor = FakeBuffer((), "i32")
        self.step_pipeline = FakePipeline()
        self.step_pipeline._action_data = [integrator.data, collision.data]
        self._private_value = "must not be exposed"


class SystemVisualizerSnapshotTest(unittest.TestCase):
    def test_snapshot_is_deterministic_cycle_safe_and_metadata_only(self):
        engine = FakeEngine()

        first = visualizer.build_snapshot(engine)
        second = visualizer.build_snapshot(engine)
        builder = visualizer.SnapshotBuilder(engine)

        self.assertEqual(first, second)
        self.assertEqual(builder.build(), builder.build())
        self.assertEqual([item["name"] for item in first["systems"]], ["CollisionSystem", "IntegratorSystem"])
        self.assertEqual(len(first["dependencies"]), 1)
        self.assertEqual(first["dependencies"][0]["attribute"], "require")
        self.assertTrue(first["dependencies"][0]["required"])
        self.assertEqual(first["pipelines"][0]["yield_callbacks"][0]["checkpoint"], 7)
        self.assertEqual(len(first["pipelines"][0]["action_data_refs"]), 2)
        self.assertIn("step_graph", first["pipelines"][0]["graph"])
        self.assertTrue(any(entry["kind"] == "action" for entry in first["graph_entries"]))
        self.assertEqual(first["engine"]["data"][0]["name"], "data")
        self.assertEqual(first["engine"]["data"][0]["buffers"][0]["name"], "engine_state")
        self.assertEqual(first["engine"]["buffers"][0]["name"], "monitor")
        self.assertTrue(any(system["buffers"] for system in first["systems"]))
        self.assertTrue(all("data_refs" in action for action in first["actions"]))
        declared = first["systems"][0]["data"][0]["declared_fields"]
        self.assertTrue(any(field["name"] == "positions" and field["initialized"] for field in declared))
        self.assertTrue(any(field["name"] == "pending" and not field["initialized"] for field in declared))
        self.assertTrue(any(len(action["data_types"]) == 2 for action in first["actions"]))

        encoded = json.dumps(first, sort_keys=True)
        self.assertNotIn("must not be exposed", encoded)
        self.assertNotIn("unavailable", encoded)
        self.assertIn('"shape": [8, 3]', encoded)
        self.assertIn('"dtype": "f64"', encoded)
        self.assertIn('"ref": "data-', encoded)

    def test_partially_built_engine_is_supported(self):
        snapshot = visualizer.build_snapshot(object())

        self.assertEqual(snapshot["systems"], [])
        self.assertEqual(snapshot["dependencies"], [])
        self.assertEqual(snapshot["pipelines"], [])


class SystemVisualizerServerTest(unittest.TestCase):
    def test_html_json_and_read_only_endpoints(self):
        with visualizer.start_server(FakeEngine()) as server:
            self.assertEqual(server.host, "127.0.0.1")
            self.assertTrue(server.is_running)

            with urllib.request.urlopen(f"{server.url}/", timeout=3) as response:
                html = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
                self.assertIn("Simulation Architecture Inspector", html)
                self.assertIn("Actions and collections", html)
                self.assertIn("/api/snapshot", html)
                self.assertIn("cytoscape@3.30.4", html)
                self.assertIn("graph-unlock", html)
                self.assertIn('name: "grid"', html)
                self.assertIn("require", html)
                self.assertIn("graph-inspector", html)
                self.assertIn("openGraphInspector", html)

            with urllib.request.urlopen(f"{server.url}/api/snapshot", timeout=3) as response:
                payload = json.load(response)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers.get_content_type(), "application/json")
                self.assertEqual(len(payload["systems"]), 2)

            request = urllib.request.Request(f"{server.url}/api/snapshot", method="POST")
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=3)
            self.assertEqual(raised.exception.code, 405)

        self.assertFalse(server.is_running)
        server.stop()

    def test_explicit_start_and_stop(self):
        server = visualizer.start_server(FakeEngine())
        self.assertTrue(server.is_running)
        self.assertGreater(server.port, 0)

        server.stop()

        self.assertFalse(server.is_running)


if __name__ == "__main__":
    unittest.main()
