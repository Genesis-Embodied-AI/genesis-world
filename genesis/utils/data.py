"""Declarative solver data: the structs kernels take as arguments, their sizes, and the records of their arrays."""

import dataclasses
import math
import operator
import types
from collections.abc import Iterable, Iterator, Mapping
from enum import IntEnum
from typing import Annotated, Any, ClassVar, NamedTuple

import numpy as np
import torch

import pydantic
from pydantic import TypeAdapter
from typing_extensions import dataclass_transform  # Made it into standard lib from Python 3.12

import quadrants as qd
from quadrants.lang import impl

import genesis as gs
from genesis.typing import StrictBool, StrictInt, ValidFloat
from genesis.utils.misc import data_to_array


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
        annotations = namespace.get("__annotations__", {})
        validators = {}
        for key, annotation in annotations.items():
            if annotation is bool:
                validators[key] = TypeAdapter(StrictBool)
            elif annotation is int:
                validators[key] = TypeAdapter(StrictInt)
            elif isinstance(annotation, type) and issubclass(annotation, IntEnum):
                validators[key] = TypeAdapter(annotation)
            else:
                raise TypeError(f"{name}.{key}: a static config holds bool, int or IntEnum, not {annotation}.")
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
            namespace.update(__init__=__init__, __setattr__=__setattr__, __eq__=__eq__, __hash__=__hash__)
        return super().__new__(cls, name, bases, namespace)


class StaticConfig(metaclass=_StaticConfigMeta):
    """
    Base of the static configs, the compile-time switches kernels take as `qd.template()`.

    A subclass declares its fields as annotations of type bool, int or IntEnum, an optional default as their value.
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

    __slots__ = ("_owner", "_path")

    def __init__(self, owner: type, path: tuple[str, ...]):
        self._owner, self._path = owner, path

    def __getattr__(self, name):
        member_cls = self._owner.__dataclass_fields__[self._path[-1]].type
        if (
            not (isinstance(member_cls, type) and issubclass(member_cls, Data))
            or name not in member_cls.__dataclass_fields__
        ):
            raise AttributeError(f"'{'.'.join(self._path)}' has no field '{name}'.")
        return FieldRef(member_cls, (*self._path, name))

    def __repr__(self):
        return f"FieldRef({'.'.join(self._path)})"


# Metadata key under which 'field' stores its spec
SPEC = "spec"


class FieldSpec(NamedTuple):
    dtype: Any = None
    shape: tuple | None = None
    kind: "DataKind | None" = None
    labels: tuple[Label, ...] = ()
    grad: bool = False
    ndarray: bool = False
    lazy: bool = False


def field(
    dtype: Any = None,
    *,
    shape: "tuple[Size | int, ...] | None" = None,
    kind: "DataKind | None" = None,
    labels: "Label | str | tuple[Label | str, ...]" = (),
    grad: bool = False,
    ndarray: bool = False,
    lazy: bool = False,
):
    """
    Declare a field of a struct (see Data).

    `dtype` is the element type of an array: `float`, `int` or `bool` in the precision of the simulation, a Quadrants
    type, a function returning one, or such a type paired with an element shape, `(float, 3)` for a vector and
    `(float, (3, 3))` for a matrix, whose dimensions may be sizes. `shape` lists the sizes of its dimensions (see Size),
    and a field of shape () and kind CONSTANT or INFO is a constant the constructor fills from its values, a vector
    constant from any sequence of its length. `kind` and `shape` replace the defaults of the
    struct, for every array of a member. `labels` select the array, or every array of a member, in the property tables
    of the constructor. `grad` gives the array gradients when the constructor requires them. `ndarray` allocates an
    ndarray on every backend. A `lazy` member is allocated when first read, from the sizes and the gradient requirement
    of its struct.
    """
    labels = tuple([Label(str(label)) for label in (labels if isinstance(labels, tuple) else (labels,))])
    return dataclasses.field(metadata={SPEC: FieldSpec(dtype, shape, kind, labels, grad, ndarray, lazy)})


def _resolve_dtype(dtype, sizes):
    """Return the Quadrants type of an element type given to 'field', its element shape sized by 'sizes'."""
    if isinstance(dtype, tuple):
        base, element_shape = dtype
        element_shape = element_shape if isinstance(element_shape, tuple) else (element_shape,)
        dims = [dim.size(sizes) if isinstance(dim, _SizeArithmetic) else dim for dim in element_shape]
        if len(dims) == 2:
            return qd.types.matrix(*dims, _resolve_dtype(base, sizes))
        return qd.types.vector(*dims, _resolve_dtype(base, sizes))
    if isinstance(dtype, types.FunctionType):
        return dtype()
    return {float: gs.qd_float, int: gs.qd_int, bool: gs.qd_bool}.get(dtype, dtype)


class _Node(NamedTuple):
    """A field of a struct tree: an array, or a member struct when 'member' is set."""

    path: tuple[str, ...]
    member: "type[Data] | None"
    spec: FieldSpec
    kind: "DataKind"
    shape: tuple
    labels: tuple[Label, ...]

    @property
    def is_constant(self) -> bool:
        return self.member is None and self.shape == () and self.kind in (DataKind.CONSTANT, DataKind.INFO)


class Data:
    """
    Base of the structs of solver data, which kernels take as arguments.

    A subclass declares its arrays as fields annotated `qd.Tensor` (see 'field'), and its members as fields annotated
    with another struct. The class keywords `kind` and `shape` set the defaults of its fields (see DataKind). A member
    takes the shape of its field, else the default of its class, else that of its holder, and the kind of its field,
    else that of its holder, else the default of its class.

    The constructor allocates every array of the tree in declaration order. `sizes` maps the name of each size to its
    value, those no array uses being ignored. `values` holds the constants nested like the struct, validated strictly.
    `requires_grad` gives gradients to the arrays declared with `grad`. The property tables map a selection of arrays to
    a value: `env_first` moves the environment dimension first or last, `layout` stores the dimensions in the given
    permutation of their declared order, and `unused` allocates the arrays no kernel reads as 0-d placeholders. A
    selection is a label or its name, matching any number of arrays, a reference to a field of the struct (see FieldRef),
    a tuple of them, or None for every array. No two keys of a table may select the same array. `_binds` maps the path
    of a field to an array or a struct allocated elsewhere, bound in its place: no kernel may access both, since
    Quadrants assumes distinct kernel arguments never alias.
    """

    kind: ClassVar["DataKind | None"] = None
    _shape: ClassVar[tuple | None] = None
    _lazy: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, kind: "DataKind | None" = None, shape: tuple | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if kind is not None:
            cls.kind = kind
        if shape is not None:
            cls._shape = shape
        cls._lazy = dict(cls._lazy)
        annotations = cls.__dict__.get("__annotations__", {})
        for name in list(annotations):
            default = cls.__dict__.get(name)
            if isinstance(default, dataclasses.Field) and default.metadata[SPEC].lazy:
                cls._lazy[name] = annotations.pop(name)
                setattr(cls, name, property(lambda self, name=name: self._get_lazy(name)))
        dataclasses.dataclass(init=False, eq=True, frozen=True)(cls)
        # An instance shadows them with its arrays and members
        for data_field in dataclasses.fields(cls):
            setattr(cls, data_field.name, FieldRef(cls, (data_field.name,)))

    def __init__(
        self,
        *,
        sizes: Mapping[str, Any] | None = None,
        values: Mapping[str, Any] | None = None,
        env_first: Mapping[Any, bool] | None = None,
        layout: Mapping[Any, "tuple[int, ...] | None"] | None = None,
        unused: Mapping[Any, bool] | None = None,
        requires_grad: bool = False,
        _binds: "Mapping[tuple[str, ...], Data | qd.Tensor] | None" = None,
    ):
        cls, sizes, binds = type(self), sizes or {}, _binds or {}
        nodes = list(cls._nodes((), (), None, () if cls._shape is None else cls._shape))
        leaves = [node for node in nodes if node.member is None]
        properties = {leaf.path: {} for leaf in leaves}
        for name, table in (("env_first", env_first), ("layout", layout), ("unused", unused)):
            for path, value in _select(cls.__name__, name, table or {}, leaves).items():
                if name == "layout" and "env_first" in properties[path]:
                    raise TypeError(f"{cls.__name__}() sets both 'env_first' and 'layout' of {'.'.join(path)}.")
                properties[path][name] = value
        constants = [leaf for leaf in leaves if leaf.is_constant]
        is_optional = {leaf.path: properties[leaf.path].get("unused", False) for leaf in constants}
        values = _values_model(cls.__name__, constants, is_optional).model_validate(values or {}).model_dump()

        # The members first, then the arrays in declaration order, under the struct holding each
        structs = {(): self}
        for node in nodes:
            is_bound = any([node.path[: len(path)] == path for path in binds])
            if node.member is not None and not is_bound:
                structs[node.path] = object.__new__(node.member)
        for struct in structs.values():
            object.__setattr__(struct, "_context", (sizes, requires_grad))
        for node in nodes:
            holder = structs.get(node.path[:-1])
            if holder is None:
                continue
            if node.path in binds:
                value = binds[node.path]
            elif node.member is not None:
                value = structs[node.path]
            else:
                value = _allocate(cls.__name__, node, properties[node.path], sizes, values, requires_grad)
            object.__setattr__(holder, node.path[-1], value)

    @classmethod
    def _nodes(cls, path, labels, kind, shape):
        """Yield the fields of the struct tree in declaration order, a member before its own fields."""
        for data_field in dataclasses.fields(cls):
            spec = data_field.metadata.get(SPEC, FieldSpec())
            node_path, node_labels = (*path, data_field.name), labels + spec.labels
            node_kind = spec.kind if spec.kind is not None else kind
            member = (
                data_field.type if isinstance(data_field.type, type) and issubclass(data_field.type, Data) else None
            )
            if member is None:
                node_shape = spec.shape if spec.shape is not None else shape
                yield _Node(
                    node_path, None, spec, cls.kind if node_kind is None else node_kind, node_shape, node_labels
                )
                continue
            node_shape = next(iter([s for s in (spec.shape, member._shape) if s is not None]), shape)
            yield _Node(node_path, member, spec, node_kind, node_shape, node_labels)
            yield from member._nodes(node_path, node_labels, node_kind, node_shape)

    def _get_lazy(self, name):
        member = self.__dict__.get(f"_lazy_{name}")
        if member is None:
            sizes, requires_grad = self._context
            member = type(self)._lazy[name](sizes=sizes, requires_grad=requires_grad)
            object.__setattr__(self, f"_lazy_{name}", member)
        return member


def _select(struct_name, table_name, table, leaves):
    """Return the value a property table of a struct constructor gives each array it selects, by path (see Data)."""
    selected = {}
    for key, value in table.items():
        refs = [item for item in (key if isinstance(key, tuple) else (key,)) if isinstance(item, FieldRef)]
        for ref in refs:
            if not any([_is_selected(ref, leaf) for leaf in leaves]):
                raise TypeError(f"{struct_name}() '{table_name}' selects {ref}, which is no field of it.")
        for leaf in leaves:
            if _is_selected(key, leaf):
                if leaf.path in selected:
                    raise TypeError(f"{struct_name}() '{table_name}' selects {'.'.join(leaf.path)} twice.")
                selected[leaf.path] = value
    return selected


def _is_selected(key, leaf) -> bool:
    """Whether a key of a property table of a struct constructor selects an array (see Data)."""
    if key is None:
        return True
    if isinstance(key, tuple):
        return any([_is_selected(item, leaf) for item in key])
    if isinstance(key, FieldRef):
        return leaf.path[: len(key._path)] == key._path
    return Label(str(key)) in leaf.labels


def _allocate(struct_name, leaf, properties, sizes, values, requires_grad):
    """Allocate an array of a struct tree (see Data)."""
    dtype = _resolve_dtype(leaf.spec.dtype, sizes)
    if properties.get("unused"):
        return V(dtype=dtype, shape=())
    if leaf.is_constant:
        value = values
        for name in leaf.path:
            value = value[name]
        return _scalar_from(dtype, value)
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
    if leaf.spec.ndarray:
        return qd.tensor(dtype, shape=tuple(dims), backend=qd.Backend.NDARRAY)
    return V(dtype=dtype, shape=tuple(dims), layout=layout, needs_grad=leaf.spec.grad and requires_grad)


_VALUES_MODELS = {}


def _values_model(struct_name, constants, is_optional, depth=0):
    """Return the pydantic model validating the constants of a struct, nested like it, those left unused optional."""
    model_key = (struct_name, depth, tuple([(leaf.path, is_optional[leaf.path]) for leaf in constants]))
    model = _VALUES_MODELS.get(model_key)
    if model is not None:
        return model
    fields = {}
    for name in dict.fromkeys([leaf.path[depth] for leaf in constants]):
        children = [leaf for leaf in constants if leaf.path[depth] == name]
        if len(children[0].path) > depth + 1:
            member_model = _values_model(f"{struct_name}.{name}", children, is_optional, depth + 1)
            is_required = any([info.is_required() for info in member_model.model_fields.values()])
            fields[name] = (member_model, ...) if is_required else (member_model, member_model())
            continue
        (leaf,) = children
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
        fields[name] = (value_type | None, None) if is_optional[leaf.path] else (value_type, ...)
    model = pydantic.create_model(f"{struct_name}Values", __config__=pydantic.ConfigDict(extra="forbid"), **fields)
    _VALUES_MODELS[model_key] = model
    return model


# FIXME: quadrants#941 - a Python-scope fill of a field still being declared closes its SNode tree, one tree per filled
# constant. '_scalar_from' defers it to the next materialization instead, at the cost of one synchronization there.
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


def _scalar_from(dtype, value):
    data = V(dtype=dtype, shape=())
    if isinstance(value, tuple):
        value = qd.Vector(value, dt=dtype.dtype)
    # Filling a field now would close the SNode tree still collecting fields, so the fill waits for the next
    # materialization (see quadrants#941 above). An ndarray is filled right away: its zero-copy export runs no
    # materialization on CPU and CUDA, so a deferred value could be read before its fill.
    if _tensor_backend() == qd.Backend.FIELD:
        _pending_fields_fill.append((data, value))
    else:
        data.fill(value)
    return data


class DataKind(IntEnum):
    """The kind of every array or scalar a solver yields (see 'Solver.data'), which checkpoints and trajectories
    select by.

    CONSTANT is an array the build computes from the scene description once. INFO is a model parameter the setters
    write, such as mass, friction, gains or limits. STATE is what specifies the scene with INFO: the reduced-space
    configuration, velocities, accelerations, control inputs and the warm starts of the collision detection and the
    constraint solve, which change the result of the next step. DERIVED is what the next step or forward kinematics
    recomputes from STATE and INFO, such as the link poses, the contacts and the constraint forces, and recomputing it
    changes the result at the last bits at most. A step or a solve writes SCRATCH before it reads it.

    The kinds decide what a minimal trajectory records (see TrajectoryFile), so every solver classifies its arrays by
    one rule: an array the next step reads before writing it is STATE or INFO, and a DERIVED array is recomputed
    before it is read, by the next step or by the restore of a record without it (see 'Solver.__setstate__').
    """

    CONSTANT = 0
    INFO = 1
    STATE = 2
    DERIVED = 3
    SCRATCH = 4


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
            members = [(field.name, field.metadata.get(SPEC, FieldSpec()).kind) for field in dataclasses.fields(value)]
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
        field_kind = field.metadata.get(SPEC, FieldSpec()).kind
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


def is_tensor_addressable(shape, dtype) -> bool:
    """
    Whether every backend can address all the elements of a solver tensor of this shape and dtype.

    A tensor holds at most 2^31 - 1 elements, and at most 2 GiB on Metal.
    """
    # FIXME: quadrants#963 - Metal silently mis-addresses the tensors above 2 GiB instead of raising.
    if gs.backend == gs.metal:
        return math.prod(shape) * np.dtype(dtype).itemsize <= 2**31
    return math.prod(shape) <= np.iinfo(np.int32).max
