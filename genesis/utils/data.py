"""Declarative solver data: the structs kernels take as arguments, their sizes, and the records of their arrays."""

import contextlib
import dataclasses
import math
import operator
import sys
import types
import typing
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextvars import ContextVar
from enum import IntEnum
from typing import Annotated, Any, ClassVar, NamedTuple

import numpy as np
import torch

import pydantic
from pydantic import TypeAdapter
from typing_extensions import dataclass_transform  # Made it into standard lib from Python 3.12

import quadrants as qd
from quadrants.lang import impl
from quadrants.lang.util import to_numpy_type

import genesis as gs
from genesis.typing import StrictBool, StrictInt, ValidFloat
from genesis.utils.misc import data_to_array, get_own_annotations


def _tensor_backend():
    return qd.Backend.NDARRAY if gs.use_ndarray else qd.Backend.FIELD


# FIXME: quadrants#936 - a vector or matrix tensor takes no layout, so those read environment by environment are
# allocated and indexed with the environment first instead of last, against the convention of every other tensor.
# FIXME: quadrants#935 - indexing a laid-out tensor by a list or tuple skips its axis permutation, so laid-out tensors
# are indexed by plain subscripts alone.
def V(*args, **kwargs):
    return qd.tensor(*args, backend=_tensor_backend(), **kwargs)


@dataclass_transform(eq_default=True, kw_only_default=True, frozen_default=True)
class _StaticConfigMeta(type):
    """Generate the keyword-only constructor, the equality and the hash of a static config from its annotations."""

    def __new__(cls, name, bases, namespace):
        config_cls = super().__new__(cls, name, bases, namespace)
        annotations = get_own_annotations(config_cls)
        validators = {}
        for key, annotation in annotations.items():
            annotation = _resolve_type(annotation, config_cls)
            if annotation is bool:
                validators[key] = TypeAdapter(StrictBool)
            elif annotation is int:
                validators[key] = TypeAdapter(StrictInt)
            elif isinstance(annotation, type) and issubclass(annotation, IntEnum):
                validators[key] = TypeAdapter(annotation)
            elif isinstance(annotation, _StaticConfigMeta):
                validators[key] = TypeAdapter(annotation, config=pydantic.ConfigDict(arbitrary_types_allowed=True))
            else:
                raise TypeError(f"{name}.{key}: a static config holds bool, int, IntEnum or static configs.")
        defaults = {key: namespace[key] for key in annotations if key in namespace}

        def __init__(self, **kwargs):
            unexpected = kwargs.keys() - validators.keys()
            missing = validators.keys() - kwargs.keys() - defaults.keys()
            if unexpected or missing:
                raise TypeError(f"{name}() got unexpected {sorted(unexpected)} and misses {sorted(missing)}.")
            for key, validator in validators.items():
                # Native types alone, since the configs key the compiled kernels by value
                object.__setattr__(self, key, validator.validate_python(kwargs.get(key, defaults.get(key))))

        def __setattr__(self, key, value):
            raise AttributeError(f"{name} is immutable.")

        def __eq__(self, other):
            return type(other) is type(self) and vars(other) == vars(self)

        def __hash__(self):
            return hash((type(self), *[vars(self)[key] for key in validators]))

        if annotations:
            for method in (__init__, __setattr__, __eq__, __hash__):
                setattr(config_cls, method.__name__, method)
        return config_cls


class StaticConfig(metaclass=_StaticConfigMeta):
    """
    Base of the static configs, the compile-time switches kernels take as `qd.template()`.

    A subclass declares its fields as annotations of type bool, int, IntEnum or another static config, an optional
    default as their value.
    Kernels key their compiled instances on the configs they take, by equality, so a rebuilt scene with the same
    values reuses its kernels. A config is immutable, since a write would change the key its kernels are stored under.
    """


class _SizeArithmetic:
    """The arithmetic of sizes and integers (see Size)."""

    __slots__ = ()

    def __add__(self, other):
        return SizeExpr(operator.add, "+", self, other)

    def __radd__(self, other):
        return SizeExpr(operator.add, "+", other, self)

    def __sub__(self, other):
        return SizeExpr(operator.sub, "-", self, other)

    def __rsub__(self, other):
        return SizeExpr(operator.sub, "-", other, self)

    def __mul__(self, other):
        return SizeExpr(operator.mul, "*", self, other)

    def __rmul__(self, other):
        return SizeExpr(operator.mul, "*", other, self)

    def __floordiv__(self, other):
        return SizeExpr(operator.floordiv, "//", self, other)

    def __rfloordiv__(self, other):
        return SizeExpr(operator.floordiv, "//", other, self)


class Size(_SizeArithmetic):
    """
    A named size of solver data, given under its name in the `sizes` of the constructor of a struct (see Data).

    It is padded to one, or `()` to drop the dimension it stands for. Sizes combine with each other and with integers
    through `+`, `-`, `*` and `//`, into the size of a dimension.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str):
        self._name = name

    def size(self, sizes: Mapping[str, Any]) -> int:
        """Return the size, padded to one."""
        size = sizes.get(self._name)
        if size is None or isinstance(size, tuple):
            raise TypeError(f"The size '{self._name}' is missing.")
        return max(int(size), 1)

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"Size({self._name!r})"


class SizeExpr(_SizeArithmetic):
    """An arithmetic of sizes and integers, the size of a dimension computed from them (see Size)."""

    __slots__ = ("_function", "_symbol", "_operands")

    def __init__(self, function, symbol: str, *operands):
        self._function, self._symbol, self._operands = function, symbol, operands

    def size(self, sizes: Mapping[str, Any]) -> int:
        """Return the size of the dimension, padded to one."""
        values = [
            operand.size(sizes) if isinstance(operand, _SizeArithmetic) else operand for operand in self._operands
        ]
        return max(self._function(*values), 1)

    def __str__(self):
        return f"({self._operands[0]} {self._symbol} {self._operands[1]})"

    def __repr__(self):
        return f"SizeExpr{self}"


# The environments, the last axis of every per-environment array in canonical order
ENVS = Size("envs")


class Label:
    """
    A family of arrays sharing runtime properties, which the constructor of a struct sets at once (see Data).

    A label equals its name, so the property tables of a constructor take either.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def __eq__(self, other):
        return str(self) == str(other) if isinstance(other, (Label, str)) else NotImplemented

    def __hash__(self):
        return hash(self._value)

    def __str__(self):
        return self._value

    def __repr__(self):
        return f"Label({self._value!r})"


class FieldRef:
    """
    A reference to a field of a struct, through the class attribute of the same name (see Data).

    A reference to a member reaches its fields as attributes in turn, `ConstraintState.island.link_id`, and selects
    every array of the member in the property tables of the constructor.
    """

    __slots__ = ("_owner", "_path", "_root")

    def __init__(self, owner: type, path: tuple[str, ...], root: "type | None" = None):
        # The struct holding the last field of the path, and the one the path starts from
        self._owner, self._path, self._root = owner, path, owner if root is None else root

    def __getattr__(self, name):
        member_cls = {data_field.name: data_field for data_field in _data_fields(self._owner)}[self._path[-1]].type
        if (
            not (isinstance(member_cls, type) and issubclass(member_cls, Data))
            or name not in member_cls.__dataclass_fields__
        ):
            raise AttributeError(f"'{'.'.join(self._path)}' has no field '{name}'.")
        return FieldRef(member_cls, (*self._path, name), self._root)

    def __repr__(self):
        return f"FieldRef({'.'.join(self._path)})"


# Metadata key under which 'field' stores its spec
_SPEC = "spec"


class _FieldSpec(NamedTuple):
    dtype: Any = None
    shape: tuple | None = None
    kind: "DataKind | None" = None
    labels: tuple[Label, ...] = ()
    grad: bool = False
    ndarray: bool = False
    lazy: bool = False
    opt_in: bool = False
    oversize_hint: str = ""
    is_required: bool = False


def field(
    dtype: Any = None,
    *,
    shape: "tuple[Size | int, ...] | None" = None,
    kind: "DataKind | None" = None,
    labels: "Label | str | tuple[Label | str, ...]" = (),
    grad: bool = False,
    ndarray: bool = False,
    lazy: bool = False,
    opt_in: bool = False,
    oversize_hint: str = "",
):
    """
    Declare a field of a struct (see Data).

    `dtype` is the element type of an array: `float`, `int` or `bool` in the precision of the simulation, a Quadrants
    type, a function returning one, or such a type paired with an element shape, `(float, 3)` for a vector and
    `(float, (3, 3))` for a matrix, whose dimensions may be sizes. `shape` lists the sizes of its dimensions (see Size),
    and a field of shape () and kind CONSTANT or INFO is a constant the constructor fills from its values, a vector
    constant from any sequence of its length. `kind` and `shape` replace the defaults of the struct, for every array of
    a member. `labels` select the array, or every array of a member, in the property tables of the constructor. `grad`
    gives the array gradients when the constructor requires them. `ndarray` allocates an ndarray on every backend. A
    `lazy` member is allocated when first read, from the sizes and the gradient requirement of its struct. An `opt_in`
    array or member is unused until another component opts it in (see 'opt_in'). An array too large for a backend to
    address raises at allocation, with `oversize_hint` appended to the message, for instance the options shrinking it.
    """
    labels = tuple([Label(str(label)) for label in (labels if isinstance(labels, tuple) else (labels,))])
    spec = _FieldSpec(dtype, shape, kind, labels, grad, ndarray, lazy, opt_in, oversize_hint)
    return dataclasses.field(metadata={_SPEC: spec})


def requires(*, opt_in: bool = False):
    """Declare a field holding a struct built and owned elsewhere, which the struct binds when built (see Data).

    An `opt_in` requirement is unused until another component opts it in (see 'opt_in'): the struct then allocates the
    0-d placeholders of the required struct itself when none is built, and raises when it is opted in.
    """
    return dataclasses.field(metadata={_SPEC: _FieldSpec(opt_in=opt_in, is_required=True)})


def attach(target: "type[Data]", **structs: "type[Data]") -> None:
    """Add to a struct owned elsewhere the requirements of the given structs, by field name, each declared `opt_in`.

    A module calls it when imported, `attach(ColliderInfo, sdf=SDFInfo)`, so that a kernel reaches the struct through the
    target without the owner of the target declaring it, and opts the field in where it uses it (see 'requires'). A
    field attached again with the same struct is left as is. The fields of a struct are final once an instance is built,
    since a kernel taking it compiles its layout in.
    """
    for field_name, struct_cls in structs.items():
        target_field = target.__dataclass_fields__.get(field_name)
        if target_field is not None:
            spec = target_field.metadata.get(_SPEC, _FieldSpec())
            if not (spec.is_required and spec.opt_in and _resolve_type(target_field.type, target) is struct_cls):
                raise TypeError(f"attach({target.__name__}, {field_name}={struct_cls.__name__}) names another field.")
            continue
        if target._is_sealed:
            raise TypeError(
                f"attach({target.__name__}, {field_name}={struct_cls.__name__}) comes too late, an instance being already "
                f"built: import the module calling it before building any scene."
            )
        # The field of the requirement, taken from a dataclass declaring it alone
        target.__dataclass_fields__[field_name] = dataclasses.fields(
            dataclasses.make_dataclass(field_name, [(field_name, struct_cls, requires(opt_in=True))])
        )[0]
        setattr(target, field_name, FieldRef(target, (field_name,)))
        target._requires[field_name] = struct_cls


def opt_in(target: "FieldRef | type[Data]", *, cond: "Callable[[], bool] | None" = None, **sizes: Any) -> None:
    """Opt in a field declared `opt_in` (see 'field'), or a whole struct, for those built in the entered namespace.

    The target is a field by reference from the struct holding it, `opt_in(GJKScratch.contact0, envs=n_threads)`, with
    the sizes of its arrays, which add to those its struct is built with, or a struct class, which its builder checks
    with 'is_opted_in'. Opting in a requirement (see 'requires') opts in the struct it requires as well. `cond` defers the decision to the moment the struct is built, the opt-in holding if it returns
    true then, so that a component asks before it knows. It holds in the children of the namespace too, and must come
    before any instance of the struct is built in it.
    """
    root_cls, path = (target._root, target._path) if isinstance(target, FieldRef) else (target, ())
    entered_namespace = _ACTIVE_NAMESPACE.get()
    if entered_namespace is None:
        raise TypeError(f"opt_in({target!r}) needs an entered namespace (see 'namespace').")
    if entered_namespace._structs.get(root_cls):
        raise TypeError(
            f"opt_in({target!r}) comes too late: a {root_cls.__name__} is already built in {entered_namespace!r}."
        )
    entered_namespace._opt_ins.setdefault((root_cls, path), []).append((cond, sizes))
    if len(path) == 1 and path[0] in root_cls._requires:
        required_cls = _resolve_type(root_cls._requires[path[0]], root_cls)
        entered_namespace._opt_ins.setdefault((required_cls, ()), []).append((cond, {}))


def is_opted_in(target: "FieldRef | type[Data]") -> bool:
    """Return whether a field or a struct is opted in for those built in the entered namespace (see 'opt_in')."""
    root_cls, path = (target._root, target._path) if isinstance(target, FieldRef) else (target, ())
    return path in _opt_in_sizes(root_cls)


def _opt_in_sizes(cls: type) -> dict[tuple[str, ...], dict[str, Any]]:
    """Return the sizes of the fields of a struct opted in within the entered namespace or its ancestors, by path."""
    opted_in = {}
    opting_namespace = _ACTIVE_NAMESPACE.get()
    while opting_namespace is not None:
        for (root_cls, path), requests in opting_namespace._opt_ins.items():
            if root_cls is not cls or path in opted_in:
                continue
            for cond, sizes in requests:
                if cond is None or cond():
                    if opted_in.setdefault(path, sizes) != sizes:
                        raise TypeError(f"{cls.__name__}.{'.'.join(path)} is opted in with different sizes.")
        opting_namespace = opting_namespace.parent
    return opted_in


# The namespace structs register in and find the structs they require from, set while one is entered (see Namespace)
_ACTIVE_NAMESPACE: ContextVar["Namespace | None"] = ContextVar("data_namespace", default=None)


class Namespace:
    """
    A named namespace of the structs built within it (see Data and 'namespace').

    Namespaces form a tree, each child named under its parent and kept, so that a struct registered once serves every
    struct built later requiring it. The structs built within a namespace register in it and find the structs they
    require in it, else in its ancestors, so a child shadows the structs of the same type of its ancestors and several
    instances of a struct serve distinct structs requiring it.
    """

    def __init__(self, name: str, parent: "Namespace | None" = None):
        self.name, self.parent = name, parent
        self._structs: dict[type, weakref.WeakSet] = {}
        self._children: dict[str, Namespace] = {}
        # The requests opting in fields or structs, by the struct they start from and their path (see 'opt_in')
        self._opt_ins: dict[tuple[type, tuple[str, ...]], list[tuple[Callable[[], bool] | None, dict[str, Any]]]] = {}

    def __repr__(self):
        return self.name if self.parent is None else f"{self.parent!r}.{self.name}"


@contextlib.contextmanager
def namespace(name: "str | Namespace", *, root: bool = False):
    """Enter a namespace for the structs built within the block (see Namespace).

    The namespace is the child of the given name of the namespace entered, created the first time it is asked for, with
    `root` a new namespace, whatever namespace is entered, or a namespace given itself, kept by its owner to enter it
    again. The namespace entered is local to the thread.
    """
    parent = _ACTIVE_NAMESPACE.get()
    if isinstance(name, Namespace):
        entered_namespace = name
    elif root:
        entered_namespace = Namespace(name)
    elif parent is None:
        raise TypeError(f"namespace({name!r}) opens a child of the entered namespace, but none is entered.")
    else:
        if name not in parent._children:
            parent._children[name] = Namespace(name, parent)
        entered_namespace = parent._children[name]
    token = _ACTIVE_NAMESPACE.set(entered_namespace)
    try:
        yield
    finally:
        _ACTIVE_NAMESPACE.reset(token)


class _Node(NamedTuple):
    """A field of a struct tree: an array, or a member struct when 'member' is set."""

    path: tuple[str, ...]
    member: "type[Data] | None"
    spec: _FieldSpec
    kind: "DataKind"
    shape: tuple
    labels: tuple[Label, ...]

    @property
    def is_constant(self) -> bool:
        return self.member is None and self.shape == () and self.kind in (DataKind.CONSTANT, DataKind.INFO)


class Data:
    """
    Base of the structs of solver data, which kernels take as arguments.

    A subclass declares its arrays as fields annotated `qd.Tensor` (see 'field'), its members as fields annotated with
    another struct, and the structs it uses without owning them (see 'requires' and 'attach'). The class keywords `kind`
    and `shape` set the defaults of its fields (see DataKind).

    The constructor allocates every array of the tree. `sizes` maps each size (see Size) to its value and `values` each
    constant to its value, either one also keyed by a field (see FieldRef) for the arrays of that field alone.
    `requires_grad` gives gradients to the arrays declared with `grad`. The property tables `env_first`, `layout` and
    `unused` map a selection of arrays (a label, a field reference, a tuple of them, or None for all) to whether the
    environment dimension comes first, the order the dimensions are stored in, and whether the array is a 0-d
    placeholder no kernel reads. Each required struct is bound to the one of its type built in the entered namespace or
    its nearest ancestor holding one (see Namespace).
    """

    kind: ClassVar["DataKind | None"] = None
    _shape: ClassVar[tuple | None] = None
    _lazy: ClassVar[dict[str, type]] = {}
    # The pydantic model validating the values of the struct, by the paths of its unused constants
    _values_models: ClassVar[dict[frozenset, type]] = {}
    # The type of each struct the struct requires by its field, those attached to it among them, its fields being
    # final once an instance is built
    _requires: ClassVar[dict[str, type]] = {}
    _is_sealed: ClassVar[bool] = False

    def __init_subclass__(cls, kind: "DataKind | None" = None, shape: tuple | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if kind is not None:
            cls.kind = kind
        if shape is not None:
            cls._shape = shape
        cls._lazy, cls._values_models, cls._requires = dict(cls._lazy), {}, {}
        annotations = get_own_annotations(cls)
        lazy_names = [
            name
            for name in annotations
            if isinstance(cls.__dict__.get(name), dataclasses.Field) and cls.__dict__[name].metadata[_SPEC].lazy
        ]
        for name in lazy_names:
            cls._lazy[name] = annotations.pop(name)

            def get_lazy(self, name=name):
                member = self.__dict__.get(f"_lazy_{name}")
                if member is None:
                    sizes, requires_grad = self._context
                    member_cls = _resolve_type(type(self)._lazy[name], type(self))
                    member_sizes = {key: size for key, size in sizes.items() if key in _size_names(member_cls)}
                    member = member_cls(sizes=member_sizes, requires_grad=requires_grad)
                    object.__setattr__(self, f"_lazy_{name}", member)
                return member

            setattr(cls, name, property(get_lazy))
        # The lazy members are left out of the fields
        if lazy_names:
            cls.__annotations__ = annotations
        dataclasses.dataclass(init=False, eq=True, frozen=True)(cls)
        for data_field in dataclasses.fields(cls):
            # An instance shadows them with its arrays and members
            setattr(cls, data_field.name, FieldRef(cls, (data_field.name,)))
            spec = data_field.metadata.get(_SPEC, _FieldSpec())
            if spec.is_required:
                cls._requires[data_field.name] = data_field.type

    def __init__(
        self,
        *,
        sizes: Mapping[str, Any] | None = None,
        values: Mapping[str, Any] | None = None,
        env_first: Mapping[Any, bool] | bool | None = None,
        layout: "Mapping[Any, tuple[int, ...] | None] | tuple[int, ...] | None" = None,
        unused: Mapping[Any, bool] | bool | None = None,
        requires_grad: bool = False,
        _binds: "Mapping[str | FieldRef, Data | qd.Tensor] | None" = None,
    ):
        cls = type(self)
        name = cls.__name__
        # The first instance resolves the annotations of the required structs and checks them, its fields being final
        if not cls._is_sealed:
            cls._requires = {
                field_name: _resolve_type(annotation, cls) for field_name, annotation in cls._requires.items()
            }
            for field_name, required_cls in cls._requires.items():
                if not (isinstance(required_cls, type) and issubclass(required_cls, Data)):
                    raise TypeError(f"{name}.{field_name}: a required field holds a struct.")
            # Two fields bound to one struct would alias it, which Quadrants cannot detect
            if len(set(cls._requires.values())) < len(cls._requires):
                raise TypeError(f"{name}: a struct requires one struct per type.")
            cls._is_sealed = True

        # The fields opted in within the entered namespace or its ancestors (see 'opt_in'), the struct itself aside
        opted_in = {path: path_sizes for path, path_sizes in _opt_in_sizes(cls).items() if path}
        # The required structs are bound in their fields, from the entered namespace, else from the nearest ancestor
        binds = {}
        for field_name, required_cls in cls._requires.items():
            holding_namespace = _ACTIVE_NAMESPACE.get()
            while holding_namespace is not None and not holding_namespace._structs.get(required_cls):
                holding_namespace = holding_namespace.parent
            if holding_namespace is None:
                # An opt-in requirement built nowhere and opted in by nobody holds placeholders of its own, its arrays
                # being unused as those of any opt-in field
                if cls.__dataclass_fields__[field_name].metadata[_SPEC].opt_in and (field_name,) not in opted_in:
                    continue
                raise TypeError(f"{name}() finds no {required_cls.__name__} in {_ACTIVE_NAMESPACE.get()!r}.")
            structs = list(holding_namespace._structs[required_cls])
            if len(structs) > 1:
                raise TypeError(
                    f"{name}() finds {len(structs)} {required_cls.__name__} in {holding_namespace!r}, not one."
                )
            binds[(field_name,)] = structs[0]

        # The fields of the tree in declaration order, a member before its own fields, walking the holders with the
        # fields each has left
        nodes = []
        pending = [((), (), None, () if cls._shape is None else cls._shape, cls, iter(_data_fields(cls)))]
        while pending:
            path, labels, kind, shape, holder, data_fields = pending[-1]
            data_field = next(data_fields, None)
            if data_field is None:
                pending.pop()
                continue
            spec = data_field.metadata.get(_SPEC, _FieldSpec())
            node_path, node_labels = (*path, data_field.name), labels + spec.labels
            node_kind = spec.kind if spec.kind is not None else kind
            member = data_field.type
            if not (isinstance(member, type) and issubclass(member, Data)):
                node_shape = spec.shape if spec.shape is not None else shape
                node_kind = holder.kind if node_kind is None else node_kind
                nodes.append(_Node(node_path, None, spec, node_kind, node_shape, node_labels))
                continue
            # A member takes the shape of its field, else the default of its class, else that of its holder. Its kind,
            # that of its field else that of its holder, falls back on the default of its class for its arrays.
            node_shape = next(iter([s for s in (spec.shape, member._shape) if s is not None]), shape)
            nodes.append(_Node(node_path, member, spec, node_kind, node_shape, node_labels))
            if node_path not in binds:
                pending.append((node_path, node_labels, node_kind, node_shape, member, iter(_data_fields(member))))
        paths = {node.path for node in nodes}
        leaves = [node for node in nodes if node.member is None]

        # A field bound to an array or a struct allocated elsewhere, which no kernel may access through both, since
        # Quadrants assumes distinct kernel arguments never alias
        for key, bound in (_binds or {}).items():
            key_path = key._path if isinstance(key, FieldRef) else (key,)
            if key_path not in paths:
                raise TypeError(f"{name}() '_binds' names {key}, which is no field of it.")
            binds[key_path] = bound
        sizes = _by_path(name, "sizes", sizes or {}, paths)
        if unknown := sorted(opted_in.keys() - {node.path for node in nodes if node.spec.opt_in}):
            raise TypeError(f"{name}() is opted in at {unknown}, which no field declares opt_in.")
        for path, path_sizes in opted_in.items():
            sizes.setdefault(path, {}).update(path_sizes)
        for entries in sizes.values():
            if unknown := sorted(entries.keys() - _size_names(cls)):
                raise TypeError(f"{name}() 'sizes' gives {unknown}, which size no array of it.")
        properties = {leaf.path: {} for leaf in leaves}
        for table_name, table in (("env_first", env_first), ("layout", layout), ("unused", unused)):
            # The entries of the table with the path of the member they apply to, a member taking a nested table that
            # selects within it as in its own constructor, and a plain value applying to every array
            entries = []
            pending = [((), table if isinstance(table, Mapping) else {} if table is None else {None: table})]
            while pending:
                scope, level = pending.pop()
                for key, value in level.items():
                    items = key if isinstance(key, tuple) else (key,)
                    for item in items:
                        if isinstance(item, FieldRef):
                            if (*scope, *item._path) not in paths:
                                raise TypeError(f"{name}() '{table_name}' selects {item}, which is no field of it.")
                        elif item is not None and not any(
                            [Label(str(item)) in leaf.labels for leaf in leaves if leaf.path[: len(scope)] == scope]
                        ):
                            raise TypeError(
                                f"{name}() '{table_name}' selects the label {item}, which no array carries."
                            )
                    if isinstance(value, Mapping):
                        key_path = (*scope, *key._path) if isinstance(key, FieldRef) else (*scope, key)
                        if key_path not in paths:
                            raise TypeError(f"{name}() '{table_name}' names {key}, which is no field of it.")
                        pending.append((key_path, value))
                    else:
                        entries.append((scope, items, value))
            # The depth of the key setting the property of each array, the deepest field reference winning, None
            # selecting the struct or member it applies to, and a tie or a label selecting it twice raising
            selected_depths = {}
            for scope, items, value in entries:
                for leaf in leaves:
                    depth = _selection_depth(scope, items, leaf)
                    if depth is None:
                        continue
                    depth_prev = selected_depths.get(leaf.path)
                    if depth_prev is not None:
                        if depth == depth_prev or min(depth, depth_prev) < 0:
                            raise TypeError(f"{name}() '{table_name}' selects {'.'.join(leaf.path)} twice.")
                        if depth < depth_prev:
                            continue
                    if table_name == "layout" and "env_first" in properties[leaf.path]:
                        raise TypeError(f"{name}() sets both 'env_first' and 'layout' of {'.'.join(leaf.path)}.")
                    selected_depths[leaf.path] = depth
                    properties[leaf.path][table_name] = value

        # An opt-in field nobody opted in is unused
        for node in nodes:
            if node.spec.opt_in and node.path not in opted_in:
                for leaf in leaves:
                    if leaf.path[: len(node.path)] == node.path:
                        properties[leaf.path]["unused"] = True

        constants = [leaf for leaf in leaves if leaf.is_constant]
        unused_paths = frozenset([leaf.path for leaf in constants if properties[leaf.path].get("unused")])
        values_model = cls._values_models.get(unused_paths)
        if values_model is None:
            values_model = cls._values_models[unused_paths] = _values_model(name, constants, unused_paths)
        values_nested = {}
        for path, path_values in _by_path(name, "values", values or {}, paths).items():
            level = values_nested
            for path_name in path:
                level = level.setdefault(path_name, {})
            level.update(path_values)
        values = values_model.model_validate(values_nested).model_dump()

        # The members first, then the arrays in declaration order, under the struct holding each
        structs = {(): self}
        for node in nodes:
            is_bound = any([node.path[: len(path)] == path for path in binds])
            if node.member is not None and not is_bound:
                structs[node.path] = object.__new__(node.member)
        for struct in structs.values():
            object.__setattr__(struct, "_context", (sizes.get((), {}), requires_grad))
        for node in nodes:
            holder = structs.get(node.path[:-1])
            if holder is None:
                continue
            if node.path in binds:
                value = binds[node.path]
            elif node.member is not None:
                value = structs[node.path]
            else:
                # The sizes given to the fields holding the array, or to the array itself, override those of the struct
                node_sizes = {}
                for depth in range(len(node.path) + 1):
                    node_sizes.update(sizes.get(node.path[:depth], {}))
                value = _allocate(name, node, properties[node.path], node_sizes, values, requires_grad)
            object.__setattr__(holder, node.path[-1], value)

        # Quadrants assumes the arrays of a kernel argument distinct, so the structs and arrays bound may share none
        held, pending = set(), [((), self)]
        while pending:
            path, struct = pending.pop()
            for data_field in dataclasses.fields(struct):
                value = getattr(struct, data_field.name)
                if id(value) in held:
                    raise TypeError(f"{name}() holds {'.'.join((*path, data_field.name))} twice.")
                held.add(id(value))
                if isinstance(value, Data):
                    pending.append(((*path, data_field.name), value))

        entered_namespace = _ACTIVE_NAMESPACE.get()
        if entered_namespace is not None:
            entered_namespace._structs.setdefault(cls, weakref.WeakSet()).add(self)


def _resolve_type(annotation: Any, owner: type) -> Any:
    """Return what an annotation of a class names, a string or a forward reference evaluated where the class lives.

    A string comes from a quoted annotation or from `from __future__ import annotations`, and resolves in the namespace
    of the module of the class alone. A forward reference comes from Python 3.14, which records a name not defined yet
    as one, and resolves within the scope defining the class, functions included.
    """
    try:
        if isinstance(annotation, str):
            return eval(annotation, vars(sys.modules[owner.__module__]))
        if isinstance(annotation, typing.ForwardRef):
            return annotation.evaluate()
    except NameError as err:
        raise TypeError(f"{owner.__qualname__}: the annotation {annotation!r} names nothing defined.") from err
    return annotation


def _data_fields(cls: type) -> tuple[dataclasses.Field, ...]:
    """Return the fields of a struct, the annotations naming a struct defined later resolved in place (see Data)."""
    data_fields = dataclasses.fields(cls)
    for data_field in data_fields:
        data_field.type = _resolve_type(data_field.type, cls)
    return data_fields


def _size_names(cls: type) -> frozenset[str]:
    """Return the names of the sizes of a struct tree, its lazy members included and its required structs excluded."""
    if "_tree_size_names" not in cls.__dict__:
        size_names, dims, holders = set(), [], [cls]
        while holders:
            holder = holders.pop()
            dims += holder._shape or ()
            holders += [_resolve_type(lazy_type, holder) for lazy_type in holder._lazy.values()]
            for data_field in _data_fields(holder):
                if data_field.name in holder._requires:
                    continue
                if isinstance(data_field.type, type) and issubclass(data_field.type, Data):
                    holders.append(data_field.type)
                spec = data_field.metadata.get(_SPEC, _FieldSpec())
                element_shape = spec.dtype[1] if isinstance(spec.dtype, tuple) else ()
                dims += [
                    *(spec.shape or ()),
                    *(element_shape if isinstance(element_shape, tuple) else (element_shape,)),
                ]
        while dims:
            dim = dims.pop()
            if isinstance(dim, Size):
                size_names.add(str(dim))
            elif isinstance(dim, SizeExpr):
                dims += dim._operands
        cls._tree_size_names = frozenset(size_names)
    return cls._tree_size_names


def _by_path(struct_name, arg_name, table, paths):
    """Return the entries of a nested table of a struct constructor by the path of the field they belong to.

    A key is the name of an entry, the name of a member whose nested table follows, or a reference to a field (see
    FieldRef) from the member the table applies to, whose value is its own entry or its nested table.
    """
    entries = {}
    pending = [((), table)]
    while pending:
        path, level = pending.pop()
        for key, item in level.items():
            key_path = (*path, *key._path) if isinstance(key, FieldRef) else (*path, str(key))
            if isinstance(key, FieldRef) and key_path not in paths:
                raise TypeError(f"{struct_name}() '{arg_name}' names {key}, which is no field of it.")
            if isinstance(item, Mapping):
                pending.append((key_path, item))
                continue
            path_entries = entries.setdefault(key_path[:-1], {})
            if key_path[-1] in path_entries:
                raise TypeError(f"{struct_name}() '{arg_name}' gives {'.'.join(key_path)} twice.")
            path_entries[key_path[-1]] = item
    return entries


def _selection_depth(scope, items, leaf) -> int | None:
    """Return the depth of the field the items of a key of a property table select an array by (see Data).

    The key applies to the member at the path 'scope'. None is that member, and a label is of depth -1. It is None when
    the key leaves the array out.
    """
    if leaf.path[: len(scope)] != scope:
        return None
    depths = []
    for item in items:
        if item is None:
            depths.append(len(scope))
        elif isinstance(item, FieldRef):
            if leaf.path[len(scope) : len(scope) + len(item._path)] == item._path:
                depths.append(len(scope) + len(item._path))
        elif Label(str(item)) in leaf.labels:
            depths.append(-1)
    return max(depths, default=None)


# FIXME: quadrants#941 - a Python-scope fill of a field still being declared closes its SNode tree, one tree per filled
# constant. A constant waits for the next materialization instead, at the cost of one synchronization there.
_pending_fields_fill: list[tuple[qd.Tensor, Any]] = []
_materialize = impl.PyQuadrants.materialize
_clear = impl.PyQuadrants.clear


def _materialize_then_fill(self):
    _materialize(self)
    # The list is emptied before replaying it, since each fill launches a kernel that materializes again. The replay
    # ends with a synchronization because a DLPack export materializes right before handing out its buffer, which
    # leaves its caller no chance to synchronize between the fill and the first read.
    fields_fill = _pending_fields_fill.copy()
    _pending_fields_fill.clear()
    for tensor, value in fields_fill:
        tensor.fill(value)
    if fields_fill:
        self.prog.synchronize()


def _clear_then_drop_fills(self):
    _clear(self)
    _pending_fields_fill.clear()


impl.PyQuadrants.materialize = _materialize_then_fill
impl.PyQuadrants.clear = _clear_then_drop_fills


def _allocate(struct_name, leaf, properties, sizes, values, requires_grad):
    """Allocate an array of a struct tree (see Data)."""
    # The element type, its base resolved to the precision of the simulation and its element shape to the sizes
    dtype, element_shape = leaf.spec.dtype if isinstance(leaf.spec.dtype, tuple) else (leaf.spec.dtype, ())
    if isinstance(dtype, types.FunctionType):
        dtype = dtype()
    dtype = base_dtype = {float: gs.qd_float, int: gs.qd_int, bool: gs.qd_bool}.get(dtype, dtype)
    element_dims = element_shape if isinstance(element_shape, tuple) else (element_shape,)
    element_sizes = [str(dim) for dim in element_dims if isinstance(dim, Size)]
    is_element_sized = all([sizes.get(name) is not None for name in element_sizes])
    # An unused array whose element shape is sized by no given size is a placeholder of the element base type
    if is_element_sized or not properties.get("unused"):
        element_dims = [dim.size(sizes) if isinstance(dim, _SizeArithmetic) else dim for dim in element_dims]
        if len(element_dims) == 1:
            dtype = qd.types.vector(*element_dims, dtype)
        elif len(element_dims) == 2:
            dtype = qd.types.matrix(*element_dims, dtype)

    if properties.get("unused"):
        return V(dtype=dtype, shape=())
    if leaf.is_constant:
        value = values
        for name in leaf.path:
            value = value[name]
        array = V(dtype=dtype, shape=())
        if isinstance(value, tuple):
            value = qd.Vector(value, dt=dtype.dtype)
        # Filling a field now would close the SNode tree still collecting fields, so the fill waits for the next
        # materialization (see quadrants#941 above). An ndarray is filled right away: its zero-copy export runs no
        # materialization on CPU and CUDA, so a deferred value could be read before its fill.
        if _tensor_backend() == qd.Backend.FIELD:
            _pending_fields_fill.append((array, value))
        else:
            array.fill(value)
        return array

    dims, axes = [], []
    for dim in leaf.shape:
        # A size given as () drops its dimension
        if isinstance(dim, Size) and isinstance(sizes.get(str(dim)), tuple):
            continue
        try:
            dims.append(dim.size(sizes) if isinstance(dim, _SizeArithmetic) else dim)
        except TypeError as err:
            raise TypeError(f"{struct_name}.{'.'.join(leaf.path)}: {err}") from None
        axes.append(dim)
    layout = properties.get("layout")
    if properties.get("env_first") is not None:
        if ENVS not in axes:
            raise TypeError(f"'env_first' selects {'.'.join(leaf.path)}, which spans no environment dimension.")
        order = [i for i, dim in enumerate(axes) if dim != ENVS]
        i_env = axes.index(ENVS)
        order = [i_env, *order] if properties["env_first"] else [*order, i_env]
        layout = tuple(order) if order != list(range(len(axes))) else None
    # A tensor holds at most 2^31 - 1 elements, and at most 2 GiB on Metal
    # FIXME: quadrants#963 - Metal silently mis-addresses the tensors above 2 GiB instead of raising.
    n_elements = math.prod([*dims, *[dim for dim in element_dims if isinstance(dim, int)]])
    if gs.backend == gs.metal:
        is_addressable = n_elements * np.dtype(to_numpy_type(base_dtype)).itemsize <= 2**31
    else:
        is_addressable = n_elements <= np.iinfo(np.int32).max
    if not is_addressable:
        gs.raise_exception(
            f"{struct_name}.{'.'.join(leaf.path)} shape {tuple(dims)} is too large. {leaf.spec.oversize_hint}".rstrip()
        )
    if leaf.spec.ndarray:
        return qd.tensor(dtype, shape=tuple(dims), backend=qd.Backend.NDARRAY)
    return V(dtype=dtype, shape=tuple(dims), layout=layout, needs_grad=leaf.spec.grad and requires_grad)


def _values_model(struct_name, constants, unused_paths):
    """Return the pydantic model validating the constants of a struct, nested like it, the unused ones optional."""
    # The fields of the model of each struct holding constants, by path
    structs_fields = {(): {}}
    for leaf in constants:
        for depth in range(1, len(leaf.path)):
            structs_fields.setdefault(leaf.path[:depth], {})
        base = leaf.spec.dtype[0] if isinstance(leaf.spec.dtype, tuple) else leaf.spec.dtype
        if base is bool or base == qd.u1:
            value_type = StrictBool
        elif base is int or (not isinstance(base, type) and base in qd.types.primitive_types.integer_types):
            value_type = StrictInt
        else:
            value_type = ValidFloat
        # A vector constant takes any sequence of its length
        if isinstance(leaf.spec.dtype, tuple):
            value_type = Annotated[tuple[(value_type,) * leaf.spec.dtype[1]], pydantic.Field(strict=False)]
        is_optional = leaf.path in unused_paths
        structs_fields[leaf.path[:-1]][leaf.path[-1]] = (value_type | None, None) if is_optional else (value_type, ...)
    # The members before their holders, so that each model nests the models of its members
    for path in sorted(structs_fields, key=len, reverse=True):
        model_name = ".".join((struct_name, *path)) + "Values"
        config = pydantic.ConfigDict(extra="forbid")
        model = pydantic.create_model(model_name, __config__=config, **structs_fields[path])
        if path:
            is_required = any([info.is_required() for info in model.model_fields.values()])
            structs_fields[path[:-1]][path[-1]] = (model, ...) if is_required else (model, model())
    return model


class DataKind(IntEnum):
    """The kind of every array or scalar a solver yields (see 'Solver.data'), which checkpoints and trajectories
    select by.

    An array the next step reads before writing it is STATE or INFO, and a DERIVED array is recomputed before it is
    read, by the next step or by the restore of a record without it (see 'Solver.__setstate__').
    """

    CONSTANT = 0
    """Computed from the scene description once, at build."""

    INFO = 1
    """A model parameter the setters write, such as mass, friction, gains or limits."""

    STATE = 2
    """What specifies the scene with INFO: the configuration, velocities, accelerations, control inputs and the warm
    starts of the collision detection and the constraint solve, which change the result of the next step."""

    DERIVED = 3
    """Recomputed from STATE and INFO by the next step or forward kinematics, such as the link poses, the contacts and
    the constraint forces, recomputing it changing the result at the last bits at most."""

    SCRATCH = 4
    """Written by a step or a solve before it reads it."""


# The kinds a checkpoint carries: INFO, STATE and DERIVED. The static configs travel in a separate field of the
# checkpoint as provenance.
CHECKPOINT_KINDS = frozenset((DataKind.INFO, DataKind.STATE, DataKind.DERIVED))


class ArrayItem(NamedTuple):
    """An array of a solver or of the simulator, under the dotted name of the slot and fields reaching it.

    It is a quadrants array, or a torch tensor such as the step counts of the simulator, which a restore fills in place
    (see 'fill_data'). An unused array is a 0-d placeholder (see 'unused' in Data), which a restore leaves as built.
    """

    name: str
    kind: DataKind
    array: qd.Tensor | qd.Field | qd.Ndarray | torch.Tensor


class ScalarItem(NamedTuple):
    """A Python native scalar of a solver, such as a flag, under the name of its attribute (see 'iter_data').

    It is the attribute 'attr' of 'owner', a property with a setter included, which a restore sets back.
    """

    name: str
    kind: DataKind
    owner: object
    attr: str


# What a solver yields among its data, which a checkpoint or a trajectory reads and writes back (see 'iter_data')
DataItem = ArrayItem | ScalarItem


class ConfigItem(NamedTuple):
    """A static config of plain Python natives that keys the compiled kernels of a solver, under its name.

    A checkpoint and a trajectory record it as provenance alone, and never write it back.
    """

    name: str
    config: object


def iter_data(owner: object, attr: str, kind: DataKind | None = None) -> Iterator[DataItem | ConfigItem]:
    """Yield the data of an attribute, one item per array if it is a struct, named by the fields reaching it.

    The attribute is 'attr' of 'owner', and an item of a struct is named by 'attr' and the fields reaching it, joined
    with dots. Structs are dataclasses and named tuples, nested to any depth, whose arrays take the kind of their field
    (see 'field') else that of their struct. A quadrants array carrying gradients yields them as one more item named
    'grad'. A Python native scalar of an object outside any struct, such as a flag, yields an item of the given kind,
    which a restore sets back. A static config yields a ConfigItem. Anything else raises.
    """
    value = getattr(owner, attr)
    if isinstance(value, StaticConfig):
        yield ConfigItem(attr, value)
    elif isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray, torch.Tensor)):
        if kind is None:
            gs.raise_exception(f"'{attr}' holds an array that no struct or annotation gives a kind.")
        yield ArrayItem(attr, kind, value)
        if not isinstance(value, torch.Tensor) and value.has_grad():
            yield ArrayItem(f"{attr}.grad", kind, value.grad)
    elif dataclasses.is_dataclass(value) or isinstance(value, tuple):
        if dataclasses.is_dataclass(value):
            members = [
                (field.name, field.metadata.get(_SPEC, _FieldSpec()).kind) for field in dataclasses.fields(value)
            ]
        else:
            members = [(member, None) for member in value._fields]
        for member, stated_kind in members:
            is_member_array = isinstance(getattr(value, member), (qd.Tensor, qd.Field, qd.Ndarray, torch.Tensor))
            if stated_kind is None and kind is None and is_member_array:
                stated_kind = type(value).kind
            for item in iter_data(value, member, kind if stated_kind is None else stated_kind):
                yield item._replace(name=f"{attr}.{item.name}")
    elif isinstance(value, (bool, int, float)) and not (dataclasses.is_dataclass(owner) or isinstance(owner, tuple)):
        if kind is None:
            gs.raise_exception(f"'{attr}' holds a {type(value).__name__} that no annotation gives a kind.")
        yield ScalarItem(attr, kind, owner, attr)
    else:
        gs.raise_exception(f"'{attr}' holds a {type(value).__name__} where an array, a struct or a scalar stands.")


def iter_state_arrays(
    struct, path: tuple[str, ...], kind: DataKind | None = None
) -> Iterator[tuple[tuple[str, ...], qd.Tensor | qd.Field | qd.Ndarray]]:
    """Yield the quadrants arrays of a struct a step may write, those of no kind INFO or CONSTANT, with their paths.

    A path is 'path' followed by the fields reaching the array. Kinds resolve as in 'iter_data'.
    """
    for field in dataclasses.fields(struct):
        value = getattr(struct, field.name)
        field_kind = field.metadata.get(_SPEC, _FieldSpec()).kind
        field_kind = kind if field_kind is None else field_kind
        if isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
            if field_kind is None:
                field_kind = type(struct).kind
            if field_kind not in (DataKind.INFO, DataKind.CONSTANT):
                yield (*path, field.name), value
        elif dataclasses.is_dataclass(value):
            yield from iter_state_arrays(value, (*path, field.name), field_kind)


def check_data(
    items: Iterable[DataItem], values: Mapping[str, np.ndarray | torch.Tensor], *, frame_only: bool = False
) -> list[DataItem]:
    """Return the items 'fill_data' writes, rejecting a record made for another scene.

    A value read from a record has the shape of its item, and is floating exactly when its item is. Its dtype may
    differ otherwise, as it is cast when written: a one-byte boolean fills an int32 flag. Any other mismatch raises,
    and so does a name on one side only, since the record comes from another build or version. An unused array, of
    shape () on either side, stays as built. With
    `frame_only`, every array holds frames and is recorded for its frame 0 alone (see 'Solver.has_frames').
    """
    items = tuple(items)
    unshared = sorted({item.name for item in items} ^ set(values))
    if unshared:
        gs.raise_exception(f"The record and this scene do not share {unshared}: another build or version wrote it.")
    writes = []
    for item in items:
        value = values[item.name]
        if isinstance(item, ArrayItem) and isinstance(item.array, torch.Tensor):
            shape, dtype = tuple(item.array.shape), item.array.dtype
            is_item_real = dtype.is_floating_point
        elif isinstance(item, ArrayItem):
            array = item.array
            if isinstance(array, (qd.VectorTensor, qd.MatrixTensor)):
                element_shape = array.element_shape
            elif isinstance(array, qd.MatrixField):
                element_shape = (array.n, array.m)[: array.ndim]
            else:
                element_shape = ()
            if tuple(array.shape) == () or tuple(value.shape) == element_shape:
                continue
            shape, dtype = (*array.shape, *element_shape), array.dtype
            if frame_only:
                shape = shape[1:]
            is_item_real = dtype in qd.types.primitive_types.real_types
        else:
            scalar = getattr(item.owner, item.attr)
            shape, dtype = (), type(scalar).__name__
            is_item_real = isinstance(scalar, float)
        if tuple(value.shape) != shape:
            gs.raise_exception(f"'{item.name}' is shaped {shape} here and {value.shape} in the record: another build.")
        if isinstance(value, torch.Tensor):
            is_real = value.dtype.is_floating_point
        else:
            is_real = np.issubdtype(value.dtype, np.floating)
        if is_real != is_item_real:
            gs.raise_exception(f"'{item.name}' holds {dtype} values here where {value.dtype} ones were recorded.")
        writes.append(item)
    return writes


def fill_data(
    items: Iterable[DataItem], values: Mapping[str, np.ndarray | torch.Tensor], *, frame_only: bool = False
) -> None:
    """Write back into each item the value recorded under its name, once every value passed 'check_data', so that a
    record made for another scene leaves the scene as it was.

    An array is filled in place, through its zero-copy view where one exists, so that a device-resident record stays on
    the device. On Metal these writes stay on the torch stream, so the caller must call 'torch.mps.synchronize' once
    before the next kernel. A scalar is set back as the Python native the host code reading it expects. With
    `frame_only`, the frame 0 alone of each array is written (see 'check_data').
    """
    for item in check_data(items, values, frame_only=frame_only):
        value = values[item.name]
        if isinstance(item, ScalarItem):
            # 'check_data' admits a boolean record for an integer, so the value takes the native type of the attribute
            setattr(item.owner, item.attr, type(getattr(item.owner, item.attr))(value.item()))
            continue
        view = None
        if isinstance(item.array, torch.Tensor):
            view = item.array
        elif gs.use_zerocopy.is_supported:
            try:
                view = data_to_array(item.array, to_torch=True, copy=False)
            except ValueError:
                # A member of a struct field may have no zero-copy view (see 'qd_zero_grad')
                pass
        if view is not None:
            # A value is cast to the dtype of its array as it reaches the device, which may support no other precision
            (view[0] if frame_only else view).copy_(torch.as_tensor(value, dtype=view.dtype, device=view.device))
        elif frame_only:
            # Without zero-copy views, the array is written whole, its other frames as they are
            frames = data_to_array(item.array, to_torch=True, copy=True)
            frames[0] = value
            item.array.from_torch(frames)
        elif isinstance(value, torch.Tensor):
            item.array.from_torch(value)
        else:
            item.array.from_numpy(value)
