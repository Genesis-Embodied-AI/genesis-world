"""Cycle-safe, metadata-only snapshots of the graph-native simulation runtime."""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1
_MISSING = object()
_STRUCTURAL_PRIVATE_NAMES = frozenset({"_action_data", "_actions", "_owner", "_yield_callbacks"})


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _class_name(value: object) -> str:
    return type(value).__name__


def _callable_name(value: object) -> str:
    function = getattr(value, "__func__", value)
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None) or getattr(function, "__name__", None)
    if qualname:
        return f"{module}.{qualname}" if module else str(qualname)
    return _type_name(value)


def _safe_getattr(value: object, name: str, default: Any = _MISSING) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return default


def _members(value: object, *, include_structural_private: bool = False) -> list[tuple[str, object]]:
    """Read stored attributes without enumerating or evaluating arbitrary properties."""
    result: dict[str, object] = {}
    try:
        namespace = dict(vars(value))
    except (TypeError, AttributeError, RuntimeError):
        namespace = {}
    for name, member in namespace.items():
        if isinstance(name, str) and (
            not name.startswith("_") or (include_structural_private and name in _STRUCTURAL_PRIVATE_NAMES)
        ):
            result[name] = member

    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if not isinstance(name, str) or name in result:
                continue
            if name.startswith("_") and not (include_structural_private and name in _STRUCTURAL_PRIVATE_NAMES):
                continue
            member = _safe_getattr(value, name)
            if member is not _MISSING:
                result[name] = member
    return sorted(result.items())


def _safe_primitive(value: object, *, depth: int = 0) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return value.name
    if depth >= 2:
        return _MISSING
    if isinstance(value, (tuple, list)) and len(value) <= 32:
        converted = [_safe_primitive(item, depth=depth + 1) for item in value]
        if all(item is not _MISSING for item in converted):
            return converted
    if isinstance(value, Mapping) and len(value) <= 32:
        converted_mapping: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, bool, Enum)):
                return _MISSING
            converted = _safe_primitive(item, depth=depth + 1)
            if converted is _MISSING:
                return _MISSING
            converted_mapping[str(key.name if isinstance(key, Enum) else key)] = converted
        return {key: converted_mapping[key] for key in sorted(converted_mapping)}
    return _MISSING


def _metadata_text(value: object) -> str | None:
    primitive = _safe_primitive(value)
    if primitive is not _MISSING and not isinstance(primitive, (list, dict)):
        return str(primitive)
    name = _safe_getattr(value, "name", None)
    if isinstance(name, str):
        return name
    name = _safe_getattr(value, "__name__", None)
    if isinstance(name, str):
        return name
    return _class_name(value) if value is not None else None


def _shape(value: object) -> list[object] | None:
    shape = _safe_getattr(value, "shape")
    if shape is _MISSING:
        return None
    if isinstance(shape, int):
        return [shape]
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes, bytearray)):
        dimensions: list[object] = []
        for dimension in shape:
            converted = _safe_primitive(dimension)
            if not isinstance(converted, (int, str)):
                return None
            dimensions.append(converted)
        return dimensions
    return None


def _is_buffer(value: object) -> bool:
    if value is None or isinstance(value, (str, bytes, bytearray, list, tuple, dict, set)):
        return False
    try:
        inspect.getattr_static(value, "shape")
    except (AttributeError, TypeError):
        return False
    return _shape(value) is not None


def _buffer_snapshot(name: str, value: object) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "type": _type_name(value),
        "shape": _shape(value) or [],
    }
    dtype = _safe_getattr(value, "dtype")
    if dtype is not _MISSING:
        dtype_name = _metadata_text(dtype)
        if dtype_name is not None:
            result["dtype"] = dtype_name
    return result


def _looks_like_system(value: object) -> bool:
    name = _class_name(value)
    return name.endswith("System") or (
        _safe_getattr(value, "data") is not _MISSING and callable(_safe_getattr(value, "build", None))
    )


def _looks_like_data(value: object) -> bool:
    return _class_name(value) == "Data" or any(cls.__name__ == "SimData" for cls in type(value).__mro__)


def _looks_like_action(value: object) -> bool:
    return _class_name(value) == "SimAction" or (
        _safe_getattr(value, "owner") is not _MISSING
        and _safe_getattr(value, "data") is not _MISSING
        and callable(_safe_getattr(value, "kernel", None))
    )


def _looks_like_action_collection(value: object) -> bool:
    return _class_name(value) == "ActionCollection" or (
        _safe_getattr(value, "_owner") is not _MISSING and isinstance(_safe_getattr(value, "_actions"), (list, tuple))
    )


def _looks_like_pipeline(value: object) -> bool:
    callbacks = _safe_getattr(value, "_yield_callbacks")
    return _class_name(value) == "SimPipeline" or (
        callable(_safe_getattr(value, "graph", None)) and isinstance(callbacks, Mapping)
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


class SnapshotBuilder:
    """Build deterministic snapshots without reading device-backed values."""

    def __init__(self, engine: object) -> None:
        self.engine = engine
        self._reset()

    def _reset(self) -> None:
        self._system_ids: dict[int, str] = {}
        self._data_ids: dict[int, str] = {}
        self._data_counter = 0
        self._actions: list[dict[str, object]] = []
        self._action_ids: dict[int, str] = {}
        self._collections: list[dict[str, object]] = []
        self._collection_ids: dict[int, str] = {}
        self._pipelines: list[dict[str, object]] = []
        self._pipeline_ids: dict[int, str] = {}
        self._graph_entries: dict[tuple[str, str, str], dict[str, object]] = {}

    def build(self) -> dict[str, object]:
        self._reset()
        systems = self._registered_systems()
        self._assign_system_ids(systems)

        system_snapshots = [self._system_snapshot(name, system) for name, system in systems]
        engine_data = _safe_getattr(self.engine, "data")
        engine_data_snapshots = []
        if engine_data is not _MISSING and engine_data is not None and _looks_like_data(engine_data):
            engine_data_snapshots.append(self._data_snapshot("data", engine_data, "engine.data"))
        self._discover_engine_runtime_objects()

        return {
            "schema_version": SCHEMA_VERSION,
            "engine": {
                "type": _type_name(self.engine),
                "config": self._configuration(self.engine, excluded={"data", "systems"}),
                "buffers": self._buffers(self.engine),
                "data": engine_data_snapshots,
            },
            "systems": system_snapshots,
            "dependencies": self._dependencies(systems),
            "actions": sorted(self._actions, key=lambda item: str(item["id"])),
            "action_collections": sorted(self._collections, key=lambda item: str(item["id"])),
            "pipelines": sorted(self._pipelines, key=lambda item: str(item["id"])),
            "graph_entries": [
                self._graph_entries[key]
                for key in sorted(self._graph_entries, key=lambda item: (item[0], item[1], item[2]))
            ],
        }

    def _registered_systems(self) -> list[tuple[str, object]]:
        registry = _safe_getattr(self.engine, "systems")
        candidates: list[tuple[str, object]] = []
        if isinstance(registry, Mapping):
            try:
                registry_items = list(registry.items())
            except RuntimeError:
                registry_items = []
            for key, value in registry_items:
                key_name = key if isinstance(key, str) else getattr(key, "__qualname__", _type_name(key))
                candidates.append((str(key_name), value))
        elif isinstance(registry, Sequence) and not isinstance(registry, (str, bytes, bytearray)):
            candidates.extend((_class_name(value), value) for value in registry)
        else:
            for name, value in _members(self.engine):
                if _looks_like_system(value):
                    candidates.append((name, value))
        return sorted(candidates, key=lambda item: (item[0], _type_name(item[1])))

    def _assign_system_ids(self, systems: list[tuple[str, object]]) -> None:
        counts: dict[str, int] = {}
        for registry_name, system in systems:
            base = f"system-{_slug(registry_name or _class_name(system))}"
            ordinal = counts.get(base, 0)
            counts[base] = ordinal + 1
            self._system_ids[id(system)] = base if ordinal == 0 else f"{base}-{ordinal + 1}"

    def _configuration(self, value: object, *, excluded: set[str] | None = None) -> dict[str, object]:
        excluded = excluded or set()
        result: dict[str, object] = {}
        for name, member in _members(value):
            if name in excluded or callable(member):
                continue
            primitive = _safe_primitive(member)
            if primitive is not _MISSING:
                result[name] = primitive
        return result

    def _buffers(self, value: object) -> list[dict[str, object]]:
        return [_buffer_snapshot(name, member) for name, member in _members(value) if _is_buffer(member)]

    def _system_snapshot(self, registry_name: str, system: object) -> dict[str, object]:
        system_id = self._system_ids[id(system)]
        data_objects: list[dict[str, object]] = []
        action_ids: list[str] = []
        collection_ids: list[str] = []
        pipeline_ids: list[str] = []
        members = _members(system)

        for name, value in members:
            if name == "data" or name.endswith("_data") or _looks_like_data(value):
                if value is not None and not _is_buffer(value) and _safe_primitive(value) is _MISSING:
                    data_objects.append(self._data_snapshot(name, value, f"{system_id}.{name}"))

        for name, value in members:
            if _looks_like_action(value):
                action_ids.append(self._action_snapshot(name, value, system_id))
            elif _looks_like_action_collection(value):
                collection_ids.append(self._collection_snapshot(name, value, system_id))
            elif _looks_like_pipeline(value):
                pipeline_ids.append(self._pipeline_snapshot(name, value, system_id))

        valid = _safe_getattr(system, "is_valid")
        building = _safe_getattr(system, "is_building")
        return {
            "id": system_id,
            "name": registry_name or _class_name(system),
            "type": _type_name(system),
            "valid": valid if isinstance(valid, bool) else None,
            "building": building if isinstance(building, bool) else None,
            "config": self._configuration(system, excluded={"data"}),
            "buffers": self._buffers(system),
            "data": data_objects,
            "actions": sorted(action_ids),
            "action_collections": sorted(collection_ids),
            "pipelines": sorted(pipeline_ids),
        }

    def _data_snapshot(self, name: str, value: object, path: str, *, depth: int = 0) -> dict[str, object]:
        existing = self._data_ids.get(id(value))
        if existing is not None:
            return {"name": name, "ref": existing}

        self._data_counter += 1
        data_id = f"data-{self._data_counter}"
        self._data_ids[id(value)] = data_id
        result: dict[str, object] = {
            "id": data_id,
            "name": name,
            "type": _type_name(value),
            "config": {},
            "buffers": [],
            "children": [],
            "declared_fields": [],
        }
        if depth >= 8:
            result["truncated"] = True
            return result

        config = result["config"]
        buffers = result["buffers"]
        children = result["children"]
        declared_fields = result["declared_fields"]
        assert isinstance(config, dict)
        assert isinstance(buffers, list)
        assert isinstance(children, list)
        assert isinstance(declared_fields, list)
        declarations: dict[str, object] = {}
        for cls in reversed(type(value).__mro__):
            declarations.update(cls.__dict__.get("__annotations__", {}))
        for field_name, annotation in sorted(declarations.items()):
            member = _safe_getattr(value, field_name)
            declared_fields.append(
                {
                    "name": field_name,
                    "type": _metadata_text(annotation) or str(annotation),
                    "initialized": member is not _MISSING,
                }
            )
        for member_name, member in _members(value):
            if callable(member):
                continue
            primitive = _safe_primitive(member)
            if primitive is not _MISSING:
                config[member_name] = primitive
            elif _is_buffer(member):
                buffers.append(_buffer_snapshot(member_name, member))
            elif _looks_like_action(member):
                continue
            elif _looks_like_action_collection(member):
                continue
            elif _looks_like_pipeline(member):
                continue
            elif _looks_like_system(member):
                system_ref = self._system_ids.get(id(member))
                if system_ref is not None:
                    children.append({"name": member_name, "system_ref": system_ref})
            elif member is not None and _members(member):
                children.append(self._data_snapshot(member_name, member, f"{path}.{member_name}", depth=depth + 1))
        buffers.sort(key=lambda item: str(item["name"]))
        children.sort(key=lambda item: str(item["name"]))
        return result

    def _action_snapshot(self, name: str, action: object, fallback_owner: str | None = None) -> str:
        existing = self._action_ids.get(id(action))
        if existing is not None:
            return existing
        action_id = f"action-{len(self._action_ids) + 1}"
        self._action_ids[id(action)] = action_id
        owner = _safe_getattr(action, "owner")
        owner_id = self._system_ids.get(id(owner), fallback_owner)
        kernel = _safe_getattr(action, "kernel")
        kernel_name = _callable_name(kernel) if callable(kernel) else None
        data = _safe_getattr(action, "data")
        data_items = data if isinstance(data, tuple) else (() if data is _MISSING else (data,))
        data_refs = [self._data_ids.get(id(item)) for item in data_items]
        snapshot: dict[str, object] = {
            "id": action_id,
            "name": name,
            "type": _type_name(action),
            "owner": owner_id,
            "kernel": kernel_name,
            "data_types": [_type_name(item) for item in data_items],
            "data_refs": data_refs,
        }
        if len(data_items) == 1:
            snapshot["data_type"] = snapshot["data_types"][0]
            if data_refs[0] is not None:
                snapshot["data_ref"] = data_refs[0]
        self._actions.append(snapshot)
        if kernel_name:
            self._add_graph_entry(kernel_name, "action", owner_id or "")
        return action_id

    def _collection_snapshot(self, name: str, collection: object, fallback_owner: str | None = None) -> str:
        existing = self._collection_ids.get(id(collection))
        if existing is not None:
            return existing
        collection_id = f"collection-{len(self._collection_ids) + 1}"
        self._collection_ids[id(collection)] = collection_id
        owner = _safe_getattr(collection, "_owner")
        owner_id = self._system_ids.get(id(owner), fallback_owner)
        actions = _safe_getattr(collection, "_actions", ())
        action_ids = []
        if isinstance(actions, (list, tuple)):
            for index, action in enumerate(actions):
                if _looks_like_action(action):
                    action_ids.append(self._action_snapshot(f"{name}[{index}]", action, owner_id))
        self._collections.append(
            {
                "id": collection_id,
                "name": name,
                "type": _type_name(collection),
                "owner": owner_id,
                "actions": action_ids,
            }
        )
        return collection_id

    def _pipeline_snapshot(self, name: str, pipeline: object, owner: str) -> str:
        existing = self._pipeline_ids.get(id(pipeline))
        if existing is not None:
            return existing
        pipeline_id = f"pipeline-{len(self._pipeline_ids) + 1}"
        self._pipeline_ids[id(pipeline)] = pipeline_id
        graph = _safe_getattr(pipeline, "graph")
        graph_name = _callable_name(graph) if callable(graph) else None
        callbacks = _safe_getattr(pipeline, "_yield_callbacks", {})
        action_data = _safe_getattr(pipeline, "_action_data", ())
        action_data_items = action_data if isinstance(action_data, (list, tuple)) else ()
        callback_snapshots: list[dict[str, object]] = []
        if isinstance(callbacks, Mapping):
            for checkpoint, callback in sorted(callbacks.items(), key=lambda item: str(item[0])):
                callback_snapshots.append(
                    {
                        "checkpoint": _safe_primitive(checkpoint)
                        if _safe_primitive(checkpoint) is not _MISSING
                        else _metadata_text(checkpoint),
                        "callback": _callable_name(callback) if callable(callback) else _type_name(callback),
                    }
                )
        launched = _safe_getattr(pipeline, "_has_launched")
        snapshot = {
            "id": pipeline_id,
            "name": name,
            "type": _type_name(pipeline),
            "owner": owner,
            "graph": graph_name,
            "stages": ([] if graph_name is None else [{"kind": "graph", "entry": graph_name}]),
            "yield_callbacks": callback_snapshots,
            "action_data_types": [_type_name(item) for item in action_data_items],
            "action_data_refs": [self._data_ids.get(id(item)) for item in action_data_items],
            "has_launched": launched if isinstance(launched, bool) else None,
        }
        self._pipelines.append(snapshot)
        if graph_name:
            self._add_graph_entry(graph_name, "pipeline", pipeline_id)
        return pipeline_id

    def _discover_engine_runtime_objects(self) -> None:
        for name, value in _members(self.engine):
            if _looks_like_pipeline(value):
                self._pipeline_snapshot(name, value, "engine")
            elif _looks_like_action(value):
                self._action_snapshot(name, value)
            elif _looks_like_action_collection(value):
                self._collection_snapshot(name, value)

    def _add_graph_entry(self, name: str, kind: str, owner: str) -> None:
        key = (name, kind, owner)
        self._graph_entries[key] = {"name": name, "kind": kind, "owner": owner}

    def _dependencies(self, systems: list[tuple[str, object]]) -> list[dict[str, object]]:
        edges: dict[tuple[str, str], tuple[str, bool]] = {}

        def record(source: str, target: object, attribute: str, required: bool) -> None:
            target_id = self._system_ids.get(id(target))
            if target_id is None or source == target_id:
                return
            key = (source, target_id)
            previous = edges.get(key)
            if previous is None or (required and not previous[1]):
                edges[key] = (attribute, required)

        engine_dependencies = _safe_getattr(self.engine, "dependencies")
        if callable(engine_dependencies):
            try:
                for target, required in engine_dependencies().items():
                    record("engine", target, "engine", bool(required))
            except Exception:
                pass

        for _, system in systems:
            source = self._system_ids[id(system)]
            declared_dependencies = _safe_getattr(system, "dependencies")
            if callable(declared_dependencies):
                try:
                    for target, required in declared_dependencies().items():
                        record(source, target, "require" if required else "find", bool(required))
                except Exception:
                    pass
            for name, member in _members(system):
                targets: list[object] = []
                if id(member) in self._system_ids:
                    targets.append(member)
                elif isinstance(member, Mapping):
                    targets.extend(value for value in member.values() if id(value) in self._system_ids)
                elif isinstance(member, (list, tuple, set, frozenset)):
                    targets.extend(value for value in member if id(value) in self._system_ids)
                for target in targets:
                    record(source, target, name, False)
        return [
            {
                "source": source,
                "target": target,
                "attribute": attribute,
                "required": required,
            }
            for (source, target), (attribute, required) in sorted(edges.items())
        ]


def build_snapshot(engine: object) -> dict[str, object]:
    """Return a deterministic structural snapshot of an engine-like object."""
    return SnapshotBuilder(engine).build()


__all__ = ["SCHEMA_VERSION", "SnapshotBuilder", "build_snapshot"]
