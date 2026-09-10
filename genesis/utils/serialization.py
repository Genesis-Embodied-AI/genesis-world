"""Export what Genesis holds to a file that anyone can load back, and load one back.

A file holds JavaScript Object Notation (JSON) and arrays, and what each value in it is comes from the type the
loader expects: the caller says what it exported and what it is loading, and the declared types of the descriptions
and options take it from there. So a file holds no module path, no import, and no code, and a class name appears only
where a declared type has subclasses and the file must say which one.
"""

import dataclasses
import hashlib
import io
import json
import os
import pathlib
import sys
import types
import typing
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Mapping, Sequence, Set
from enum import IntEnum
from functools import lru_cache, partial
from typing import Callable, NamedTuple

import numpy as np
from frozendict import frozendict
from pydantic import AfterValidator, TypeAdapter, ValidationError, ValidationInfo

import genesis as gs
from genesis.options.options import Options

# What every member of an archive is stamped with, so a file is the bytes its scene comes to and nothing else.
EPOCH = (1980, 1, 1, 0, 0, 0)
_SOURCE_DIGEST: str | None = None
MANIFEST_NAME = "scene.json"
ARRAY_MEMBER = "arrays.npz"

# This dict records how to write a class from another library to a file and read it back. Such a class cannot
# declare this itself. '_class_codec' walks the method resolution order, so a subclass falls back to its base's pair.
_REGISTERED: dict[type, tuple[Callable, Callable]] = {}


class Exporting(NamedTuple):
    """Gives a class exporting itself somewhere to put an array and the export of any value it holds."""

    array: Callable[[np.ndarray], int]
    value: Callable[..., object]


class Exported(NamedTuple):
    """What one export writes: its arrays, the slot each set of bytes already written stands at, those arrays, and
    every class whose values it holds.
    """

    values: list[np.ndarray]
    slots: dict[tuple, int]
    stored: list[np.ndarray]
    classes: set[type]


class Loading(NamedTuple):
    """Gives a class loading itself back the arrays a file holds and the load of any value it holds.

    'shared' holds what has already been built from a given part of a file, so a class handed the same bytes twice
    can share with the first instance what deriving it again would cost.
    """

    array: Callable[[int], np.ndarray]
    value: Callable[..., object]
    shared: dict


class Loaded(NamedTuple):
    """The arrays one load reads, and what the classes loading themselves have already built from them."""

    values: list[np.ndarray]
    shared: dict


class SerializationMixin:
    """Declares a class serialisable and lets the class itself say how it travels in a file.

    A class whose declared fields hold its values needs none of this, since a file carries those fields. Fields fall
    short for geometry belonging in an array, or a constructor consuming rather than keeping its inputs. A separate
    registry does the same for a class another library owns.
    """

    def export(self, exporting: Exporting):
        """Return everything this instance holds in a form JSON can carry.

        The 'exporting' handle sets arrays aside.

        Must be overridden by the class declaring itself serialisable.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, raw, loading: Loading) -> "SerializationMixin":
        """Recreate the instance 'export' encoded, reading back through 'loading' the arrays it put aside.

        Must be overridden by the class declaring itself serialisable.
        """
        raise NotImplementedError


def register(cls: type, export: Callable, load: Callable) -> None:
    """Record how a class from another library travels in a file, which lets a file hold a class Genesis does not own.

    'export(value, exporting)' returns what JSON can hold and 'load(raw, loading)' gives the value back, both of them
    handed an 'Exporting' or a 'Loading' to put arrays aside and to export or load the values the class holds. A class
    Genesis declares mixes in 'SerializationMixin' and states the same two things as its own methods.
    """
    _REGISTERED[cls] = (export, load)


def _class_codec(cls) -> tuple[Callable, Callable] | None:
    """Return the export and load pair for a class.

    A self-serialisable class supplies its own methods; otherwise its own or its base's registered pair applies.
    """
    if not isinstance(cls, type):
        return None
    if issubclass(cls, SerializationMixin):
        return (cls.export, cls.load)
    for parent in cls.__mro__:
        if parent in _REGISTERED:
            return _REGISTERED[parent]
    return None


# Validators receive it as 'info.context' while a value read from a file is checked (see 'ignore_invalid_load').
_FROM_FILE = frozendict({"from_file": True})


def ignore_invalid_load(accepted: Iterable) -> AfterValidator:
    """Returns a validator loading a recorded value outside 'accepted' as None, the unset state of an optional field.

    A value a file records holds wherever the file was written. A field whose values hold there and nowhere else
    declares beside its type, through this, which of them hold on this machine, so the file opens here whatever it
    asked for. What a caller gives is left as given, to be rejected where it is used.
    """
    accepted = tuple(accepted)

    def settle(value, info: ValidationInfo):
        if value in accepted or info.context is None or not info.context.get("from_file", False):
            return value
        return None

    return AfterValidator(settle)


@lru_cache(maxsize=None)
def _field_check(cls: type, name: str) -> TypeAdapter:
    """Returns what one field of an option admits, so a value read from a file is put through it.

    What a field constrains stands beside its type rather than in it, so the two are put back together here.
    """
    info = cls.model_fields[name]
    admitted = typing.Annotated[tuple([info.annotation, *info.metadata])] if info.metadata else info.annotation
    return TypeAdapter(admitted)


def _resolve_quoted(hint, namespace: dict):
    """Returns a declaration holding the classes it names, where Python 3.10 left a quoted name unresolved.

    Until Python 3.11, a name quoted inside a generic - 'list["Joint"]' - stays the string it was written as, and a
    declaration read as a string states nothing about what it holds.
    """
    if not isinstance(hint, types.GenericAlias):
        return hint
    args = typing.get_args(hint)
    return typing.get_origin(hint)[
        tuple(namespace[arg] if isinstance(arg, str) else _resolve_quoted(arg, namespace) for arg in args)
    ]


def _declared_fields(cls: type) -> dict[str, type]:
    """Return the declared type of each field of a class. Those types drive how a value loads back."""
    if issubclass(cls, Options):
        return {name: info.annotation for name, info in cls.model_fields.items()}
    hints = typing.get_type_hints(cls)
    if sys.version_info < (3, 11):
        namespace = vars(sys.modules[cls.__module__])
        return {name: _resolve_quoted(hint, namespace) for name, hint in hints.items()}
    return hints


def _unwrap_annotation(expect):
    """Returns what a declaration asks for, stripped of wrappers that leave its meaning intact.

    A file says by holding null whether an optional value is there, so 'X | None' asks for X. The payload inside
    'Annotated' serves validation and leaves the value untouched.
    """
    if typing.get_origin(expect) is typing.Annotated:
        expect = typing.get_args(expect)[0]
    if typing.get_origin(expect) in (types.UnionType, typing.Union):
        args = [arg for arg in typing.get_args(expect) if arg is not type(None)]
        return _unwrap_annotation(args[0]) if len(args) == 1 else expect
    return expect


def _accepted_classes(expect) -> tuple:
    """Returns the classes a declaration accepts, one for 'Mesh' and two for 'Box | Sphere', with 'None' left out.

    A single-class declaration needs no class name in the file. A declaration accepting several makes the file name
    which class it holds.
    """
    if typing.get_origin(expect) in (types.UnionType, typing.Union):
        return tuple(arg for arg in typing.get_args(expect) if arg is not type(None))
    return (expect,)


def _classes_by_name(expect) -> dict[str, type]:
    """Collects every class a file may name where a declaration stands, keyed by class name.

    A declaration accepting 'Morph' accepts every kind of morph, so 'Morph' and everything extending it may stand
    there. A name read back is looked up here, so a file naming a class outside the declaration is rejected.
    """
    by_name = {}
    classes = [_unwrap_annotation(arg) for arg in _accepted_classes(expect)]
    while classes:
        cls = classes.pop()
        by_name[cls.__name__] = cls
        classes.extend(cls.__subclasses__())
    return by_name


def _deduce_union(expect, value):
    """Returns the argument of a union declaration that holds this value, or the declaration itself.

    A scale states one factor or one per axis, so only the argument the value stands in states what its items are.
    """
    if typing.get_origin(expect) not in (types.UnionType, typing.Union):
        return expect
    args = [_unwrap_annotation(arg) for arg in _accepted_classes(expect)]
    for arg in args:
        if _is_sequence(typing.get_origin(arg) or arg) == _is_sequence(type(value)):
            return arg
    return args[0]


def _is_sequence(cls) -> bool:
    """Tells whether instances of 'cls' travel as a JSON list. Sequences and sets qualify, and strings stay scalars."""
    return isinstance(cls, type) and issubclass(cls, (Sequence, Set)) and not issubclass(cls, str)


def _is_mapping(cls) -> bool:
    """Tells whether instances of 'cls' travel as a JSON list of key and value pairs."""
    return isinstance(cls, type) and issubclass(cls, Mapping)


def _has_declared_fields(expect) -> bool:
    """Tells whether every class a declaration accepts travels field by field, as dataclasses and options do."""
    args = [_unwrap_annotation(arg) for arg in _accepted_classes(expect)]
    return bool(args) and all(
        isinstance(arg, type) and (dataclasses.is_dataclass(arg) or issubclass(arg, Options)) for arg in args
    )


def _layout(cls: type) -> list[str] | dict[str, int]:
    """What one class is made of: the value of every member where it enumerates, its field names otherwise."""
    if issubclass(cls, IntEnum):
        return {member.name: member.value for member in cls}
    return sorted(_declared_fields(cls))


def schema(classes: set[type]) -> dict:
    """Records what each of these classes is made of, under the name each is written as.

    A field renamed or an enumeration member renumbered leaves every value in a file intact while changing what it
    means. A file therefore carries this of the classes it holds, and a load compares it class by class. A class
    that gained a field since keeps loading while one whose fields moved is rejected by name. Sorted by name, so what
    a file holds follows from the classes alone rather than from the order they are met in.
    """
    return {cls.__name__: _layout(cls) for cls in sorted(classes, key=lambda cls: cls.__name__)}


def declared_schema(expect: dict) -> dict:
    """Records the layout of every class reachable from a declaration, the reference a load checks the file against.

    A load holds only the declared types until it reads a value, so it takes every class a declaration admits. The
    comparison looks up only the names the file holds, so the extra entries cost nothing.
    """
    layout = {}
    for annotation in expect.values():
        seen = set()
        declarations = [annotation]
        while declarations:
            declared = _unwrap_annotation(declarations.pop())
            # A declaration is reached through the arguments it is written with, so a list of morphs reaches Morph.
            declarations.extend(typing.get_args(declared))
            if not isinstance(declared, type) or declared in seen:
                continue
            seen.add(declared)
            if issubclass(declared, IntEnum):
                layout[declared.__name__] = _layout(declared)
            elif _has_declared_fields(declared):
                for cls in _classes_by_name(declared).values():
                    if cls in seen and cls is not declared:
                        continue
                    seen.add(cls)
                    layout[cls.__name__] = _layout(cls)
                    declarations.extend(_declared_fields(cls).values())
    return layout


def _surface_textures(surface: Options) -> Iterable[tuple[str, "gs.textures.ImageTexture"]]:
    """Yield every image texture a surface holds, under the name of the field it stands in.

    A batch texture holds textures of its own, so a batch created from images yields each of them.
    """
    textures = [(name, value) for name, value in dict(surface).items() if isinstance(value, gs.textures.Texture)]
    while textures:
        name, texture = textures.pop()
        if isinstance(texture, gs.textures.ImageTexture):
            yield name, texture
        elif isinstance(texture, gs.textures.BatchTexture):
            textures.extend((name, held) for held in texture.textures if held is not None)


def pixel_less_textures(surface: Options) -> tuple[str, ...]:
    """Names each field of a surface holding a texture no file can carry, which is one read from an HDR or EXR file."""
    return tuple(name for name, texture in _surface_textures(surface) if texture.image_array is None)


def source_digest() -> str:
    """Digest the Genesis sources this process runs, so a file says which code wrote it.

    The release version alone answers for nothing between two releases, which is where a branch and a local edit both
    live, and either changes what a scene simulates. Reading the sources answers for both, and the digest is taken
    once per process.
    """
    global _SOURCE_DIGEST
    if _SOURCE_DIGEST is None:
        root = pathlib.Path(gs.__file__).parent
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        _SOURCE_DIGEST = digest.hexdigest()[:16]
    return _SOURCE_DIGEST


def _store_array(value, exported: "Exported") -> int:
    """Store one array with the arrays the file holds and return its index.

    Bytes already stored keep the index they were given, so entities built from one asset share the geometry in the
    file as they share it in memory. Two views onto one buffer read the same bytes, so the memory a view reads keys
    the lookup rather than the view itself, and every keyed array is held until the export ends so no address is
    reused meanwhile. The copy is made contiguous in row order, so the file holds the same bytes however numpy laid
    the values out, and keeps the shape it was given: an array holding one value and no axis loads back as such.
    """
    layout = value.__array_interface__
    key = (layout["data"][0], layout["shape"], layout["strides"], layout["typestr"])
    slot = exported.slots.get(key)
    if slot is None:
        slot = len(exported.values)
        exported.slots[key] = slot
        exported.stored.append(value)
        exported.values.append(np.require(value, requirements="C"))
    return slot


def _exporting_into(exported: "Exported") -> Exporting:
    """Builds the handle a class receives to export itself into this file."""
    return Exporting(array=partial(_store_array, exported=exported), value=partial(_export_value, exported=exported))


def _loading_from(loaded: "Loaded") -> Loading:
    """Builds the handle a class receives to load itself back out of this file."""
    return Loading(array=loaded.values.__getitem__, value=partial(_load_value, loaded=loaded), shared=loaded.shared)


def _export_value(value, expect, exported: "Exported"):
    """Export one value as what JSON can hold, against the type it is declared to be.

    What the declaration settles is left unsaid: a field declared to hold one class says only its values, and only a
    field whose declared type has subclasses says which one it holds.
    """
    expect = _unwrap_annotation(expect)
    if value is None:
        return None
    # A field declared to hold anything tells nothing about its content, so the value records its own class.
    if expect is typing.Any or expect is object:
        return _export_any(value, exported)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, IntEnum):
        exported.classes.add(type(value))
        return value.name
    codec = _class_codec(type(value))
    if codec is not None:
        held = codec[0](value, _exporting_into(exported))
        # A class saying how it travels may stand where its base is declared, so the file names which one it is.
        if held is None or _accepted_classes(expect) == (type(value),):
            return held
        return {"@class": type(value).__name__, "raw": held}
    if _has_declared_fields(expect):
        exported.classes.add(type(value))
        given = vars(value)
        values = {
            name: _export_value(given[name], annotation, exported)
            for name, annotation in _declared_fields(type(value)).items()
        }
        if issubclass(type(value), Options):
            # An option object carries the names of the fields the user gave, since rebuilding one needs them. They
            # are sorted, so exporting the same scene twice gives the same file.
            values = {"values": values, "given": sorted(value.model_fields_set)}
        return values if _accepted_classes(expect) == (type(value),) else {"@": type(value).__name__, **values}
    expect = _deduce_union(expect, value)
    origin = typing.get_origin(expect)
    if _is_sequence(origin):
        args = typing.get_args(expect)
        if issubclass(origin, tuple) and len(args) > 1 and args[1] is not Ellipsis:
            return [_export_value(item, arg, exported) for item, arg in zip(value, args)]
        item_type = args[0] if args else None
        return [_export_value(item, item_type, exported) for item in value]
    if _is_mapping(origin):
        key, item = typing.get_args(expect) or (None, None)
        return [[_export_value(k, key, exported), _export_value(v, item, exported)] for k, v in value.items()]
    if isinstance(value, (bool, int, float, str)):
        return value
    if _is_mapping(type(value)):
        return _export_any(value, exported)
    gs.raise_exception(f"A Genesis file cannot hold {type(value).__name__} where {expect} was declared.")


def _export_any(value, exported: "Exported"):
    """Export a value declared as 'Any', which states nothing about it, such as the entries of an open mapping.

    The value records its JSON kind, a named class records how it travels, and it loads back as exported.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if _is_mapping(type(value)):
        return {"@dict": [[_export_any(k, exported), _export_any(v, exported)] for k, v in value.items()]}
    if _is_sequence(type(value)):
        return {"@list": [_export_any(item, exported) for item in value]}
    # A value whose declaration says nothing about it records its own class name. This lets a morph carry the meshes
    # it received rather than a path to read them from.
    codec = _class_codec(type(value))
    if codec is not None:
        return {"@class": type(value).__name__, "raw": codec[0](value, _exporting_into(exported))}
    gs.raise_exception(f"A Genesis file cannot hold {type(value).__name__} where 'Any' was declared.")


def _load_any(raw, loaded: "Loaded"):
    """Load back one value declared as 'Any', as what it was exported as."""
    if not isinstance(raw, dict):
        return raw
    if "@list" in raw:
        return [_load_any(item, loaded) for item in raw["@list"]]
    if "@dict" in raw:
        return {_load_any(k, loaded): _load_any(v, loaded) for k, v in raw["@dict"]}
    if "@class" in raw:
        classes = [SerializationMixin, *_REGISTERED]
        while classes:
            cls = classes.pop()
            classes.extend(cls.__subclasses__())
            if cls.__name__ == raw["@class"]:
                return _class_codec(cls)[1](raw["raw"], _loading_from(loaded))
        gs.raise_exception(f"A Genesis file holds a '{raw['@class']}', which no class says how it is loaded.")
    gs.raise_exception(f"A Genesis file holds a value this version cannot load where 'Any' stands: {sorted(raw)}.")


def _load_value(raw, expect, loaded: "Loaded"):
    """Load back one value as the type it is declared to be, creating only what that declaration allows."""
    if raw is None:
        if typing.get_origin(expect) is typing.Annotated:
            expect = typing.get_args(expect)[0]
        if type(None) not in typing.get_args(expect) and expect is not typing.Any and expect is not object:
            gs.raise_exception(f"A Genesis file holds nothing where a {expect} was declared.")
        return None
    expect = _unwrap_annotation(expect)
    if expect is typing.Any or expect is object:
        return _load_any(raw, loaded)
    if isinstance(expect, type) and issubclass(expect, IntEnum):
        return expect[raw]
    if isinstance(raw, dict) and "@class" in raw:
        cls = _classes_by_name(expect).get(raw["@class"])
        codec = None if cls is None else _class_codec(cls)
        if codec is None:
            gs.raise_exception(f"A Genesis file holds a '{raw['@class']}' where {expect} was declared.")
        return codec[1](raw["raw"], _loading_from(loaded))
    codec = _class_codec(expect)
    if codec is not None:
        return codec[1](raw, _loading_from(loaded))
    if _has_declared_fields(expect):
        # A single-class declaration settles which class stands there, so only a wider one makes the file name it.
        if "@" in raw:
            kinds = _classes_by_name(expect)
            cls = kinds.get(raw["@"])
            if cls is None:
                gs.raise_exception(
                    f"A Genesis file holds a '{raw['@']}' where {expect} was declared, and Genesis declares no such "
                    f"class there (it declares {', '.join(sorted(kinds))})."
                )
        else:
            cls = _accepted_classes(expect)[0]
        declared = _declared_fields(cls)
        if issubclass(cls, Options):
            values = {name: _load_value(raw["values"][name], declared[name], loaded) for name in raw["values"]}
            # A file arrives from wherever it was written, so each value it states is put through what its field
            # admits. The class itself is built past its own validation: an option resolves what it was authored
            # with, and its validators reject the resolved form. An option standing in a field is left to the pass
            # its own load makes. The fields marked as given are the ones the user gave then, since one option
            # inherits from another only what the receiver left unset.
            for name, value in values.items():
                if isinstance(value, Options) or (
                    isinstance(value, (list, tuple)) and any(isinstance(item, Options) for item in value)
                ):
                    continue
                try:
                    values[name] = _field_check(cls, name).validate_python(value, context=_FROM_FILE)
                except ValidationError as e:
                    gs.raise_exception_from(f"A Genesis file states a '{name}' that {cls.__name__} rejects.", e)
            return cls.model_construct(_fields_set=set(raw["given"]), **values)
        return cls(**{name: _load_value(raw[name], declared[name], loaded) for name in declared if name in raw})
    expect = _deduce_union(expect, raw)
    origin = typing.get_origin(expect)
    if _is_sequence(origin):
        args = typing.get_args(expect)
        if issubclass(origin, tuple) and len(args) > 1 and args[1] is not Ellipsis:
            return tuple(_load_value(item, arg, loaded) for item, arg in zip(raw, args))
        item_type = args[0] if args else None
        items = [_load_value(item, item_type, loaded) for item in raw]
        # An abstract sequence declaration loads as a plain list, a concrete one as the class it names.
        return items if origin in (list, Sequence) else origin(items)
    if _is_mapping(origin):
        key, item = typing.get_args(expect) or (None, None)
        pairs = {_load_value(k, key, loaded): _load_value(v, item, loaded) for k, v in raw}
        # An abstract mapping declaration loads as a plain dict, a concrete one as the class it names.
        return pairs if origin in (dict, Mapping) else origin(pairs)
    if isinstance(raw, (bool, int, float, str)):
        # JSON holds one number kind, so a whole number stands for a float where one is declared.
        if isinstance(expect, type) and not isinstance(raw, expect) and not (expect is float and isinstance(raw, int)):
            gs.raise_exception(f"A Genesis file holds a {type(raw).__name__} where a {expect.__name__} was declared.")
        return raw
    # A value carrying its own kind loads back as that kind, and a mapping declared without key and item types
    # travels that way. Every other declaration is settled above, so a value standing here contradicts its field.
    if isinstance(raw, dict) and not {"@dict", "@list", "@class"}.isdisjoint(raw):
        return _load_any(raw, loaded)
    gs.raise_exception(f"A Genesis file holds a {type(raw).__name__} where {expect} was declared.")


def _redact_paths(value, key=None):
    """Return the value with every path it holds reduced to the name of what the path points at.

    A directory belongs to the author's filesystem, so a file keeps the name alone. A path stands where a field or a
    key names one, 'file' or a name ending in '_path' (a morph's file, a mesh's or a texture's path), and inside an
    inline document as the value of an attribute. Every other string, a prim path or a joint name written with slashes
    among them, is left as it is.
    """
    if isinstance(value, str):
        try:
            document = ET.fromstring(value)
        except ET.ParseError:
            is_path = key is not None and (key == "file" or key.endswith("_path")) and os.path.isabs(value)
            return os.path.basename(value) if is_path else value
        for element in document.iter():
            for stated in element.attrib.values():
                if os.path.isabs(stated):
                    value = value.replace(stated, os.path.basename(stated))
        return value
    if isinstance(value, list):
        return [_redact_paths(item) for item in value]
    if isinstance(value, dict):
        if "@dict" in value:
            return {"@dict": [[k, _redact_paths(v, k if isinstance(k, str) else None)] for k, v in value["@dict"]]}
        return {name: _redact_paths(item, name) for name, item in value.items()}
    return value


def export(path: str | os.PathLike, values: dict) -> None:
    """Export what a caller holds to a file that loads back as data alone, holding no code and reading no asset.

    'values' names each value the file carries, and loading it asks for those names back. The file is a zip archive of
    one manifest and one member holding every array, and what each value is comes from the declared type of what
    holds it, so loading one creates the descriptions, options and meshes those declarations allow and nothing else.
    Every path the values name is reduced here to the name of what it points at (see '_redact_paths'), so no other
    code has to.
    """
    exported = Exported(values=[], slots={}, stored=[], classes=set())
    # Every class the file holds is known once its values are exported, so the schema follows them.
    held = _redact_paths({name: _export_value(value, type(value), exported) for name, value in values.items()})
    manifest = {
        "genesis": gs.__version__,
        "source": source_digest(),
        "schema": schema(exported.classes),
        "held": held,
    }
    text = json.dumps(manifest, indent=1)
    member = io.BytesIO()
    # Every array is compressed where it is written: the member holding them is itself an archive, so an archive
    # around it would find nothing left to compress.
    np.savez_compressed(member, **{str(index): array for index, array in enumerate(exported.values)})
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in ((MANIFEST_NAME, text.encode()), (ARRAY_MEMBER, member.getvalue())):
            # Zip stamps each member with the write clock by default. A fixed timestamp makes two exports of one
            # scene byte-identical.
            entry = zipfile.ZipInfo(name, date_time=EPOCH)
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)


def load(path: str | os.PathLike, expect: dict) -> dict:
    """Load back what 'export' wrote, each value as the type the caller says it expects.

    A file arrives from wherever it was written, so anything about it may be wrong: it may be truncated, hold neither
    of the two members a file is written as, or have been exported when a class meant something else. Each failure is
    rejected with a message naming what was expected, so a reader knows whether to ask for the file again or for
    another version.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read(MANIFEST_NAME))
            member = np.load(io.BytesIO(archive.read(ARRAY_MEMBER)))
            loaded = Loaded(values=[member[str(index)] for index in range(len(member.files))], shared={})
    except zipfile.BadZipFile as e:
        gs.raise_exception_from(f"'{path}' is not a Genesis file, or was truncated on its way here.", e)
    except KeyError as e:
        gs.raise_exception_from(
            f"'{path}' is missing what a Genesis file is written as: a '{MANIFEST_NAME}' manifest and an "
            f"'{ARRAY_MEMBER}' member holding its arrays.",
            e,
        )
    except (json.JSONDecodeError, ValueError) as e:
        gs.raise_exception_from(f"'{path}' holds a '{MANIFEST_NAME}' or an '{ARRAY_MEMBER}' Genesis cannot read.", e)
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("held"), dict)
        or not isinstance(manifest.get("schema"), dict)
        or not all(
            isinstance(layout, dict) or (isinstance(layout, list) and all(isinstance(name, str) for name in layout))
            for layout in manifest["schema"].values()
        )
    ):
        gs.raise_exception(f"'{path}' holds a '{MANIFEST_NAME}' that is not the manifest Genesis writes.")
    # Three differences change what the values in the file mean. A class the file holds may no longer exist, a
    # field it holds may no longer be declared, and an enumeration member's value may have moved. A field or a
    # member added since means nothing new: an option states which of its fields were given, and a description
    # field left out stands at its declared default.
    layout = declared_schema(expect)
    for name, written_layout in manifest["schema"].items():
        current = layout.get(name)
        if current is None:
            gs.raise_exception(f"'{path}' holds a '{name}', which this version of Genesis no longer declares.")
        if isinstance(written_layout, dict):
            moved = sorted(
                f"{member}={value}" for member, value in written_layout.items() if current.get(member) != value
            )
            if moved:
                gs.raise_exception(f"'{path}' was written where '{name}' stated {', '.join(moved)}, which has moved.")
        else:
            gone = sorted(set(written_layout) - set(current))
            if gone:
                gs.raise_exception(f"'{path}' holds {gone} of '{name}', which this version of Genesis no longer has.")
    if manifest.get("source") != source_digest():
        gs.logger.warning(
            f"'{path}' was exported by Genesis {manifest.get('genesis')} ({manifest.get('source')}) and is being "
            f"loaded by {gs.__version__} ({source_digest()}). What the file holds still means what it says, and the "
            "simulation it describes may run differently."
        )
    missing = sorted(set(expect) - set(manifest["held"]))
    if missing:
        gs.raise_exception(f"'{path}' holds {sorted(manifest['held'])} rather than the {missing} asked for.")
    # A file written elsewhere may hold anything, so a value that contradicts its declaration is rejected here
    # rather than reaching a constructor.
    try:
        return {name: _load_value(manifest["held"][name], annotation, loaded) for name, annotation in expect.items()}
    except (TypeError, KeyError, IndexError, ValueError) as e:
        gs.raise_exception_from(f"'{path}' holds values a Genesis scene is not made of.", e)


def _exported_array(value: np.ndarray, exporting: Exporting) -> int:
    """Store the array in the file's array member and return its index, which the values carry in its place."""
    return exporting.array(value)


def _loaded_array(raw: int, loading: Loading) -> np.ndarray:
    """Return the stored array the exported index points at."""
    return loading.array(raw)


register(np.ndarray, _exported_array, _loaded_array)
