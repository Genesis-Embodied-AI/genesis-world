import dataclasses
import math
import types
from collections.abc import Iterable, Iterator, Mapping
from enum import IntEnum
from typing import Any, ClassVar, NamedTuple

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


def V_VEC(*args, **kwargs):
    return qd.Vector.tensor(*args, backend=_tensor_backend(), **kwargs)


def V_MAT(*args, **kwargs):
    return qd.Matrix.tensor(*args, backend=_tensor_backend(), **kwargs)


def maybe_shape(shape, is_on):
    return shape if is_on else ()


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


class Axis:
    """
    A dimension of solver data, sized when a struct spanning it is allocated (see Data).

    Its size is given under its name in the `shape` of the constructor, padded to one, or `()` to drop the dimension.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str):
        self._name = name

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"Axis({self._name!r})"


# The environments, the last axis of every per-environment array in canonical order
ENVS = Axis("envs")


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


class FieldSpec(NamedTuple):
    dtype: Any = None
    shape: tuple | None = None
    labels: tuple[Label, ...] = ()
    grad: bool = False
    ndarray: bool = False
    lazy: bool = False


# Metadata keys under which 'field' stores the kind and the spec of a struct field
KIND = "kind"
SPEC = "spec"


def field(
    dtype: Any = None,
    *,
    shape: "tuple[Axis | int, ...] | None" = None,
    kind: "DataKind | None" = None,
    labels: "Label | str | tuple[Label | str, ...]" = (),
    grad: bool = False,
    ndarray: bool = False,
    lazy: bool = False,
):
    """
    Declare a field of a struct (see Data).

    `dtype` is the element type of an array: `float`, `int` or `bool` in the precision of the simulation, a Quadrants
    type, or such a type paired with an element shape, `(float, 3)` for a vector and `(float, (3, 3))` for a matrix.
    `shape` lists the axes and the fixed sizes of the array, replacing the default shape of its struct, and a field of
    shape () and kind CONSTANT or INFO is a constant the constructor fills. `kind` replaces the kind of its struct (see
    DataKind). `labels` are the families of the array, or of every array of a member, for the property tables of the
    constructor. `grad` gives it gradients when the constructor requires them. An
    `ndarray` array is an ndarray whatever the backend of the solver data. A `lazy` member is allocated the first time
    it is read, from the sizes and the gradients of its struct alone, for a scratch most scenes never need.
    """
    labels = tuple([Label(str(label)) for label in (labels if isinstance(labels, tuple) else (labels,))])
    return dataclasses.field(metadata={KIND: kind, SPEC: FieldSpec(dtype, shape, labels, grad, ndarray, lazy)})


def _resolve_dtype(dtype):
    """Return the Quadrants type of an element type given to 'field'."""
    if isinstance(dtype, tuple):
        base, element_shape = dtype
        if isinstance(element_shape, tuple):
            return qd.types.matrix(*element_shape, _resolve_dtype(base))
        return qd.types.vector(element_shape, _resolve_dtype(base))
    if isinstance(dtype, types.FunctionType):
        return dtype()
    return {float: gs.qd_float, int: gs.qd_int, bool: gs.qd_bool}.get(dtype, dtype)


class _Leaf(NamedTuple):
    path: tuple[str, ...]
    spec: FieldSpec
    kind: "DataKind"
    shape: tuple
    labels: tuple[Label, ...]


class Data:
    """
    Base of the structs of solver data, which kernels take as arguments.

    A subclass declares its arrays as fields annotated `qd.Tensor` (see 'field'), and its members as fields annotated
    with another struct. The class keywords `kind` and `shape` set the default kind and shape of its fields (see
    DataKind), a member taking the shape of its field, else the default shape of its own class, else that of the
    struct holding it.

    The constructor allocates every array of the tree in declaration order, once and for all. `shape` maps the name of
    each axis to its size, the axes no array spans being ignored. `values` holds the constants nested like the struct,
    validated strictly. `requires_grad` gives gradients to the arrays declared with `grad`. The property tables map a
    selection of arrays to a value: `env_first` stores the environment axis first or last, `layout` stores the axes in
    the given permutation of their canonical order, and `unused` leaves the arrays no kernel reads as 0-d placeholders,
    to spare their memory. A selection is a label or its name, matching any number of arrays, a reference to a field of
    the struct (see FieldRef), which must exist, a tuple of them, or None for every array, and no two keys may select
    the same array. `_binds` maps the path of a field to an array or a struct allocated elsewhere,
    bound in its place, an alias no kernel may access along with its original, since Quadrants assumes distinct kernel
    arguments never alias.
    """

    kind: ClassVar["DataKind"]
    _shape: ClassVar[tuple | None] = None
    _lazy: ClassVar[dict[str, tuple[type, FieldSpec]]] = {}

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
                cls._lazy[name] = (annotations.pop(name), default.metadata[SPEC])
                delattr(cls, name)
                setattr(cls, name, property(lambda self, name=name: self._get_lazy(name)))
        dataclasses.dataclass(init=False, eq=True, frozen=True)(cls)
        # An instance shadows them with its arrays and members
        for data_field in dataclasses.fields(cls):
            setattr(cls, data_field.name, FieldRef(cls, (data_field.name,)))

    def __init__(
        self,
        *,
        shape: Mapping[str, Any] | None = None,
        values: Mapping[str, Any] | None = None,
        env_first: Mapping[Any, bool] | None = None,
        layout: Mapping[Any, "tuple[int, ...] | None"] | None = None,
        unused: Mapping[Any, bool] | None = None,
        requires_grad: bool = False,
        _binds: "Mapping[tuple[str, ...], Data | qd.Tensor] | None" = None,
    ):
        cls = type(self)
        context = (shape or {}, values or {}, env_first or {}, layout or {}, unused or {}, requires_grad)
        object.__setattr__(self, "_context", context)
        cls._build(self, cls._shape or (), (), (), None, context, _binds or {})

    @classmethod
    def _leaves(cls, shape, labels, path, kind):
        """Yield the arrays of the struct, with the shape, the kind and the labels they take within it."""
        for data_field in dataclasses.fields(cls):
            spec = data_field.metadata.get(SPEC, FieldSpec())
            field_kind = data_field.metadata.get(KIND)
            field_kind = kind if field_kind is None else field_kind
            field_labels = labels + spec.labels
            if isinstance(data_field.type, type) and issubclass(data_field.type, Data):
                member_shape = spec.shape if spec.shape is not None else data_field.type._shape or shape
                yield from data_field.type._leaves(member_shape, field_labels, (*path, data_field.name), field_kind)
            else:
                field_shape = spec.shape if spec.shape is not None else shape
                field_kind = cls.kind if field_kind is None else field_kind
                yield _Leaf((*path, data_field.name), spec, field_kind, field_shape, field_labels)

    @classmethod
    def _properties(cls, leaves, context):
        """Return the runtime properties of the arrays, by path, from the property tables of the constructor."""
        _, _, env_first, layout, unused, _ = context
        properties = {leaf.path: {} for leaf in leaves}
        for name, table in (("env_first", env_first), ("layout", layout), ("unused", unused)):
            selected = {}
            for key, value in table.items():
                for ref in [item for item in (key if isinstance(key, tuple) else (key,)) if isinstance(item, FieldRef)]:
                    if not any(_is_selected(ref, leaf) for leaf in leaves):
                        raise TypeError(f"{cls.__name__}() '{name}' selects {ref}, which is no field of it.")
                for leaf in leaves:
                    if _is_selected(key, leaf):
                        if leaf.path in selected:
                            raise TypeError(f"{cls.__name__}() '{name}' selects {'.'.join(leaf.path)} twice.")
                        selected[leaf.path] = value
            for path, value in selected.items():
                if name == "layout" and "env_first" in properties[path]:
                    raise TypeError(f"{cls.__name__}() sets both 'env_first' and 'layout' of {'.'.join(path)}.")
                properties[path][name] = value
        return properties

    @classmethod
    def _build(cls, self, shape, labels, path, kind, context, binds, leaves=None):
        """Allocate the arrays of the struct in declaration order, then bind them and its members to it."""
        sizes, values, _, _, _, requires_grad = context
        if leaves is None:
            leaves = list(cls._leaves(shape, labels, path, kind))
            properties = cls._properties(leaves, context)
            unused_paths = frozenset([leaf.path for leaf in leaves if properties[leaf.path].get("unused")])
            values = _values_model(cls, shape, path, kind, unused_paths).model_validate(values).model_dump()
            context = (sizes, values, *context[2:])
            object.__setattr__(self, "_properties", properties)
        properties = self._properties
        for data_field in dataclasses.fields(cls):
            field_path = (*path, data_field.name)
            spec = data_field.metadata.get(SPEC, FieldSpec())
            if field_path in binds:
                object.__setattr__(self, data_field.name, binds[field_path])
                continue
            if isinstance(data_field.type, type) and issubclass(data_field.type, Data):
                member = object.__new__(data_field.type)
                object.__setattr__(member, "_properties", properties)
                object.__setattr__(member, "_context", context)
                member_shape = spec.shape if spec.shape is not None else data_field.type._shape or shape
                field_kind = data_field.metadata.get(KIND)
                field_kind = kind if field_kind is None else field_kind
                data_field.type._build(
                    member, member_shape, labels + spec.labels, field_path, field_kind, context, binds, leaves=()
                )
                object.__setattr__(self, data_field.name, member)
                continue
            leaf_properties = properties[field_path]
            field_shape = spec.shape if spec.shape is not None else shape
            field_kind = data_field.metadata.get(KIND)
            field_kind = (cls.kind if kind is None else kind) if field_kind is None else field_kind
            dtype = _resolve_dtype(spec.dtype)
            if leaf_properties.get("unused"):
                array = V(dtype=dtype, shape=())
            elif field_shape == () and field_kind in (DataKind.CONSTANT, DataKind.INFO):
                value = values
                for name in field_path:
                    value = value[name]
                array = V_SCALAR_FROM(dtype=dtype, value=value)
            else:
                dims, axes = [], []
                for dim in field_shape:
                    if isinstance(dim, Axis):
                        if str(dim) not in sizes:
                            raise TypeError(f"{cls.__name__}.{data_field.name} needs the size of axis '{dim}'.")
                        size = sizes[str(dim)]
                        if isinstance(size, tuple):
                            continue
                        dims.append(max(int(size), 1))
                    else:
                        dims.append(dim)
                    axes.append(dim)
                array_layout = leaf_properties.get("layout")
                if leaf_properties.get("env_first") is not None:
                    if ENVS not in axes:
                        raise TypeError(f"'env_first' selects {'.'.join(field_path)}, which spans no environment axis.")
                    order = [i for i, dim in enumerate(axes) if dim != ENVS]
                    i_env = axes.index(ENVS)
                    order = [i_env, *order] if leaf_properties["env_first"] else [*order, i_env]
                    array_layout = tuple(order) if order != list(range(len(axes))) else None
                if spec.ndarray:
                    array = qd.tensor(dtype, shape=tuple(dims), backend=qd.Backend.NDARRAY)
                else:
                    array = V(
                        dtype=dtype, shape=tuple(dims), layout=array_layout, needs_grad=spec.grad and requires_grad
                    )
            object.__setattr__(self, data_field.name, array)

    def _get_lazy(self, name):
        member = self.__dict__.get(f"_lazy_{name}")
        if member is None:
            member_cls, spec = type(self)._lazy[name]
            sizes, *_, requires_grad = self._context
            member = member_cls(shape=sizes, requires_grad=requires_grad)
            object.__setattr__(self, f"_lazy_{name}", member)
        return member


def _is_selected(key, leaf) -> bool:
    """Whether a key of a property table of a struct constructor selects an array (see Data)."""
    if key is None:
        return True
    if isinstance(key, tuple):
        return any(_is_selected(item, leaf) for item in key)
    if isinstance(key, FieldRef):
        return leaf.path[: len(key._path)] == key._path
    return Label(str(key)) in leaf.labels


_VALUES_MODELS = {}


def _values_model(cls, shape, path, kind, unused_paths):
    """Return the pydantic model validating the constants of a struct, nested like it, those left unused optional."""
    model_key = (cls, unused_paths)
    model = _VALUES_MODELS.get(model_key)
    if model is not None:
        return model
    fields = {}
    for data_field in dataclasses.fields(cls):
        spec = data_field.metadata.get(SPEC, FieldSpec())
        field_path = (*path, data_field.name)
        field_kind = data_field.metadata.get(KIND)
        field_kind = kind if field_kind is None else field_kind
        if isinstance(data_field.type, type) and issubclass(data_field.type, Data):
            member_shape = spec.shape if spec.shape is not None else data_field.type._shape or shape
            member_model = _values_model(data_field.type, member_shape, field_path, field_kind, unused_paths)
            if member_model.model_fields:
                is_required = any(info.is_required() for info in member_model.model_fields.values())
                fields[data_field.name] = (member_model, ...) if is_required else (member_model, member_model())
            continue
        field_shape = spec.shape if spec.shape is not None else shape
        field_kind = cls.kind if field_kind is None else field_kind
        if field_shape != () or field_kind not in (DataKind.CONSTANT, DataKind.INFO):
            continue
        base = spec.dtype if not isinstance(spec.dtype, tuple) else spec.dtype[0]
        if base is bool or base == qd.u1:
            value_type = StrictBool
        elif base is int or (not isinstance(base, type) and base in qd.types.primitive_types.integer_types):
            value_type = StrictInt
        else:
            value_type = ValidFloat
        fields[data_field.name] = (value_type | None, None) if field_path in unused_paths else (value_type, ...)
    model = pydantic.create_model(f"{cls.__name__}Values", __config__=pydantic.ConfigDict(extra="forbid"), **fields)
    _VALUES_MODELS[model_key] = model
    return model


# FIXME: quadrants#941 - a Python-scope fill of a field still being declared closes its SNode tree, one tree per filled
# constant. 'V_SCALAR_FROM' defers it to the next materialization instead, at the cost of one synchronization there.
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


def V_SCALAR_FROM(dtype, value):
    data = V(dtype=dtype, shape=())
    # Filling a field now would close the SNode tree still collecting fields, so the fill waits for the next
    # materialization (see quadrants#941 above). An ndarray is filled right away: its zero-copy export runs no
    # materialization on CPU and CUDA, so a deferred value could be read before its fill.
    if _tensor_backend() == qd.Backend.FIELD:
        _pending_fields_fill.append((data, value))
    else:
        data.fill(value)
    return data


def _timer_dtype():
    """Return the type of the profiling timers, 64-bit where the backend supports it."""
    return qd.i32 if gs.backend == gs.metal else qd.i64


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


def of_kind(kind: DataKind):
    """Declare a struct field whose kind differs from the default its struct states (see DataKind)."""
    return dataclasses.field(metadata={KIND: kind})


class ArrayItem(NamedTuple):
    """An array of a solver or of the simulator, under the dotted name of the slot and fields reaching it.

    It lives on the device of the backend, and a restore fills it in place (see 'iter_data'). It is a quadrants array,
    or a torch tensor such as the step counts of the simulator. An unused quadrants array is the 0-d placeholder of a
    feature switched off (see 'maybe_shape'), which a restore leaves as built (see 'check_data').
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

    The attribute is 'attr' of 'owner', and an item of a struct is named by 'attr' plus field names joined with dots.
    Structs are the frozen dataclasses and named tuples of this module, nested to any depth. The class attribute 'kind'
    gives the default kind of a struct's fields, and 'of_kind' on a field overrides it. An override on a struct-valued
    field applies to every array of that struct. An array is a quadrants array or a torch tensor, and a quadrants array
    carrying gradients yields them as one more item of its kind, named 'grad'. A Python native scalar of an object that
    is no struct, such as a flag, yields an item of the given kind, which a restore sets back as the attribute. A static
    config yields a ConfigItem. Anything else raises and names the offending attribute.
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
            members = [(field.name, field.metadata.get(KIND)) for field in dataclasses.fields(value)]
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
        field_kind = field.metadata.get(KIND, kind)
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
    and so does a name on one side only, since the record comes from another build or version. 'maybe_shape' shapes an
    unused array as (), so an array with that shape on either side stays as built, and is left out. With
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


# =========================================== ErrorCode ===========================================


class ErrorCode(IntEnum):
    SUCCESS = 0b000000000000000000000000000000000
    OVERFLOW_COLLISION_PAIRS = 0b00000000000000000000000000000001
    OVERFLOW_CANDIDATE_CONTACTS = 0b00000000000000000000000000000010
    OVERFLOW_CONTACTS = 0b00000000000000000000000000000100
    UNSUPPORTED_COLLISION_PAIR = 0b00000000000000000000000000001000
    INVALID_CONTACT_NAN = 0b00000000000000000000000000010000
    INVALID_FORCE_NAN = 0b00000000000000000000000000100000
    INVALID_ACC_NAN = 0b00000000000000000000000001000000
    INVALID_GRAD_NAN = 0b00000000000000000000000010000000
    INVALID_PARTICLE_NAN = 0b00000000000000000000000100000000


# =========================================== RigidInfo ===========================================


# Axes of the rigid solver data, sized by the solver counts padded to one (see the trailing-underscore counts of
# KinematicSolver)
ENVS = Axis("envs")
QS = Axis("qs")
DOFS = Axis("dofs")
LINKS = Axis("links")
JOINTS = Axis("joints")
GEOMS = Axis("geoms")
ROOTS = Axis("roots")
TREES = Axis("trees")
# The links of every tree level by level, and the children of every link (see trees_levels_links_idx in RigidInfo)
TREES_LEVELS_LINKS = Axis("trees_levels_links")
CHILD_LINKS = Axis("child_links")
LINKS_BOUNDS = Axis("links_bounds")

# The arrays of the dynamics, which a kinematic-only solver never reads
DYNAMICS = Label("dynamics")
# The tables of the level sweep (see trees_levels_links_idx)
LEVEL_SWEEP = Label("level_sweep")
# The scratch of the register-tiled mass factor (see mass_mat_tiled_scratch)
TILED_MASS = Label("tiled_mass")


class RigidInfo(Data, kind=DataKind.CONSTANT):
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO, labels=DYNAMICS)
    meaninertia: qd.Tensor = field(float, shape=(ENVS,), kind=DataKind.INFO, labels=DYNAMICS)
    # *_bw: Cache for backward pass
    # Awake dofs per env under hibernation, counted down where an island sleeps and up where one wakes. An env whose
    # count is zero has every body asleep and skips the passes that gate on it whole. An env whose count is the dof
    # count has no sleeper and skips the passes that look for one (the chain edges and the wake pass of the island
    # build, the pair filter of the broad phase, the contact advection, the inert rows), so hibernation costs it one
    # read per pass until something sleeps in it. A dof-less scene pads its dof buffers to one slot and takes the slow
    # path.
    n_awake_dofs: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)
    qpos0: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.INFO)
    qpos: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.STATE, grad=True)
    qpos_next: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.SCRATCH, grad=True)
    links_T: qd.Tensor = field((float, (4, 4)), shape=(LINKS,), kind=DataKind.DERIVED)
    geoms_init_AABB: qd.Tensor = field((float, 3), shape=(GEOMS, 8), labels=DYNAMICS)
    # Flipped from canonical (n_dofs(i_d1), n_dofs(i_d2), _B) to physical (_B, n_dofs(i_d2), n_dofs(i_d1)) under the
    # cooperative constraint kernels: i_d1 becomes innermost / stride-1, which coalesces consumer
    # kernels whose lanes stride i_d1 with a serial inner i_d2 loop. The trade-off is regression on writer-side kernels
    # that pair with cooperative rewrites to recover under the same enable_cooperative_constraint_kernels flag.
    mass_mat: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), kind=DataKind.DERIVED, labels=DYNAMICS, grad=True)
    # Canonical. Its dominant consumer is a serial Cholesky-style back-substitution that is already coalesced under
    # (n_dofs, n_dofs, _B) with lanes varying i_b, so flipping L would regress that path more than the corresponding
    # writer-side win on the tiled factor_mass.
    mass_mat_L: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), kind=DataKind.DERIVED, grad=True, labels=DYNAMICS)
    mass_mat_D_inv: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.DERIVED, grad=True, labels=DYNAMICS)
    # Batch-first scratch for the register-tiled mass factor (qd.simt tile ops are batch-first, so the factorization
    # cannot run in place on the batch-last mass_mat_L), with the constraint Hessian's shape so nt_H can alias it (see
    # ConstraintState) instead of allocating a second buffer. The factor only touches it before the constraint solve
    # repopulates it in the same step.
    mass_mat_tiled_scratch: qd.Tensor = field(float, shape=(ENVS, DOFS, DOFS), kind=DataKind.SCRATCH, labels=TILED_MASS)
    # Kinematic roots: the links sharing a root link (links.root_idx), static ones included. A root spans the links
    # [root, links_root_end[root]) whose root it is (a span may interleave links of other roots, so the walks gate each
    # link on its root). The composite inertia and the center of mass are per root.
    roots_link_idx: qd.Tensor = field(int, shape=(ROOTS,))
    links_root_end: qd.Tensor = field(int, shape=(LINKS,))
    # Kinematic trees: the links a chain of moving joints connects, so a static link belongs to none (links_tree_idx
    # -1) and each branch of a fixed base is a tree. Tree i_t is rooted at trees_root_idx[i_t], spans the links
    # [trees_root_idx[i_t], trees_link_end[i_t]) mapped to it and the contiguous dofs [trees_dof_start[i_t],
    # trees_dof_start[i_t] + trees_n_dofs[i_t]), in ascending dof order. The islands are built on the trees. A scene
    # without any tree keeps one padded slot at root 0 and link end 0, so every tree walk is empty.
    trees_root_idx: qd.Tensor = field(int, shape=(TREES,))
    trees_link_end: qd.Tensor = field(int, shape=(TREES,))
    trees_n_links: qd.Tensor = field(int, shape=(TREES,))
    trees_dof_start: qd.Tensor = field(int, shape=(TREES,))
    trees_n_dofs: qd.Tensor = field(int, shape=(TREES,))
    links_tree_idx: qd.Tensor = field(int, shape=(LINKS,))
    # trees_levels_links_idx lists the links of every tree level by level for the level sweep (see
    # func_sweep_links_by_level and KinematicSolver._init_tree_fields). The levels of tree i_t start at entry
    # trees_level_start[i_t] and number trees_n_levels[i_t], and the level of entry i_k ends before entry
    # trees_levels_links_end[i_k]. The root tables list all the links of every root the same way, and the level of entry
    # i_k starts at entry roots_levels_links_start[i_k]. Empty where the level sweep does not run.
    trees_level_start: qd.Tensor = field(int, shape=(TREES,), labels=LEVEL_SWEEP)
    trees_n_levels: qd.Tensor = field(int, shape=(TREES,), labels=LEVEL_SWEEP)
    trees_levels_links_idx: qd.Tensor = field(int, shape=(TREES_LEVELS_LINKS,), labels=LEVEL_SWEEP)
    trees_levels_links_end: qd.Tensor = field(int, shape=(TREES_LEVELS_LINKS,), labels=LEVEL_SWEEP)
    roots_level_start: qd.Tensor = field(int, shape=(ROOTS,), labels=LEVEL_SWEEP)
    roots_n_levels: qd.Tensor = field(int, shape=(ROOTS,), labels=LEVEL_SWEEP)
    roots_levels_links_idx: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    roots_levels_links_start: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    roots_levels_links_end: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    # The children of link i_l are links_child_idx[links_child_start[i_l]:links_child_start[i_l + 1]], in descending
    # order: the leaf-to-root folds of the level sweep add them into their parent in the order of the serial folds.
    links_child_start: qd.Tensor = field(int, shape=(LINKS_BOUNDS,), labels=LEVEL_SWEEP)
    links_child_idx: qd.Tensor = field(int, shape=(CHILD_LINKS,), labels=LEVEL_SWEEP)
    # Per-DOF bounds of the mass block the DOF belongs to: the DOFs of its branch rooted where the fixed structure ends
    # (deeper branches stay mass-coupled to their chain and belong to the enclosing block), merged across entities and
    # kept contiguous by attach(). A block lies within one kinematic tree, whose dof range the blocks partition (an
    # aligned free body splits into one block per dof), so the assemble/factor/solve walk the blocks of a tree and
    # restrict to these bounds.
    dofs_mass_block_start: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    dofs_mass_block_end: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    dofs_mass_envelope_start: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    # Mass parent of each dof, -1 for none: the previous dof of its link, else the last dof of its nearest ancestor link
    # holding one, when it lies in the same mass block.
    dofs_mass_parent: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    # Mask of the (dof, dof) pairs the mass matrix couples: a dof with its ancestors along the kinematic chain and
    # the dofs of its own link, within its mass block.
    mass_parent_mask: qd.Tensor = field(float, shape=(DOFS, DOFS), labels=DYNAMICS)
    # Runtime constants
    substep_dt: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    ls_iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    ls_tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    noslip_iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    noslip_tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    impratio: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    n_equalities: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    n_candidate_equalities: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    hibernation_thresh_vel: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    EPS: qd.Tensor = field(float, shape=())


def is_tensor_addressable(shape, dtype) -> bool:
    """
    Whether every backend can address all the elements of a solver tensor of this shape and dtype.

    A tensor holds at most 2^31 - 1 elements, and at most 2 GiB on Metal.
    """
    # FIXME: quadrants#963 - Metal silently mis-addresses the tensors above 2 GiB instead of raising.
    if gs.backend == gs.metal:
        return math.prod(shape) * np.dtype(dtype).itemsize <= 2**31
    return math.prod(shape) <= np.iinfo(np.int32).max


# =========================================== Constraint ===========================================


# Axes of the constraint solve
CONSTRAINTS = Axis("constraints")
CONTACTS = Axis("contacts")
# Rows of the elliptic cones, and the cones themselves (see n_constraints_cone)
CONE_ROWS = Axis("cone_rows")
CONES = Axis("cones")
# Rows of the Newton update vectors, one per dof and slot (see nt_vec)
NT_VEC_ROWS = Axis("nt_vec_rows")
# Dofs a Jacobian row reaches at most (see jac_dofs_idx)
JAC_DOFS = Axis("jac_dofs")
# Columns of the noslip scratch, two per environment, or per lane of the cooperative sweep (see noslip_MinvJT)
NOSLIP_COLUMNS = Axis("noslip_columns")
# Slots of the work-lists of the tiled island seed, and their size classes (see factor_worklist_i_b)
WORKLIST = Axis("worklist")
TILE_CLASSES = Axis("tile_classes")

# The rows of the constraint state and of the islands the cooperative kernels read, one env per block, which they
# read from consecutive addresses once stored environment-first
COOPERATIVE_ROWS = Label("cooperative_rows")
# The Jacobian rows, which the cooperative kernels and serialized execution read whole for one env (see jac)
JACOBIAN_ROWS = Label("jacobian_rows")
# The rows serialized execution reads env by env, in the outermost loop of every constraint kernel
SERIAL_ROWS = Label("serial_rows")
NEWTON = Label("newton")
CONE_INCREMENTAL = Label("cone_incremental")
SIGNORINI = Label("signorini")
CONE_FREE_REUSE = Label("cone_free_reuse")
NOSLIP = Label("noslip")
NOSLIP_COOPERATIVE = Label("noslip_cooperative")
HIBERNATION = Label("hibernation")
# The reverse Cuthill-McKee tree reorder of the CPU per-island skyline path (see rcm_tree_pos)
RCM = Label("rcm")
# The contact lists of the islands, of the skyline reorder or of the cooperative contact sort (see contact_id)
ISLAND_CONTACTS = Label("island_contacts")
WORKLISTS = Label("worklists")
# The buffers of the backward pass of the constraint solve
BACKWARD = Label("backward")


class IslandSlices(Data, kind=DataKind.SCRATCH, shape=(TREES, ENVS)):
    # Per-(island, env) slices into a packed id array: island i_island's items are id[start[i_island, i_b] : start +
    # n[i_island, i_b]]. curr is the write cursor the partition build advances while filling each slice; once built,
    # curr == start + n. Indexed [n_trees, B] since an island holds at least one tree (see trees_root_idx in RigidInfo).
    curr: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)
    n: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)
    start: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)


class IslandState(Data, kind=DataKind.SCRATCH):
    # Partition of the dof-carrying kinematic trees (see trees_root_idx in RigidInfo) into islands, the connected
    # components of the trees under the contact, equality and hibernation couplings, rebuilt every step by island.py.
    # trees_parent_idx is the union-find forest over the trees, trees_island_idx the island of each tree and
    # links_island_idx that of each link, -1 for a static link and for a dof-less tree and its links. link_slices maps
    # an island to its slice of link_id, dof_slices to its slice of dof_id (island-local dof -> global dof, ascending
    # unless the CPU skyline path reorders it by contact adjacency).
    trees_parent_idx: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    trees_island_idx: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    links_island_idx: qd.Tensor = field(int, shape=(LINKS, ENVS), labels=COOPERATIVE_ROWS)
    n_islands: qd.Tensor = field(int, shape=(ENVS,))
    link_slices: IslandSlices = field(labels=HIBERNATION)
    link_id: qd.Tensor = field(int, shape=(LINKS, ENVS), labels=(COOPERATIVE_ROWS, HIBERNATION))
    dof_slices: IslandSlices
    dof_id: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Inverse of dof_id: dof_local_pos[d] is the island-local position of global dof d
    dof_local_pos: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    dofs_island_idx: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # First dof of an island whose dof list holds consecutive dofs in ascending order, -1 otherwise (see func_list_item
    # in linesearch.py).
    dof_range_start: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    # Skyline envelope of the CPU per-island path: dof_env_start_local[dof_slices.start[i] + ld] is the smallest
    # island-local column structurally nonzero in local row ld of island i's Hessian block, 0 (dense) when uncomputed.
    # dof_env_col_end is its transpose, the largest local row whose envelope reaches column ld, which bounds the
    # column-oriented factor and solve sweeps to the band: a tall island of a few hundred dofs otherwise tests every
    # row below the diagonal against its envelope, an O(n^2) scan per sweep. It has no safe uncomputed default (0
    # truncates), so only the CPU per-island path reads it.
    dof_env_start_local: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    dof_env_col_end: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Per-island contact lists of the CPU per-island skyline reorder (see rcm_tree_pos below, empty otherwise).
    # contact_id also carries the input order of the cooperative contact sort (see func_sort_contacts_coop in
    # island.py).
    contact_slices: IslandSlices = field(labels=RCM)
    contact_id: qd.Tensor = field(int, shape=(CONTACTS, ENVS), labels=(COOPERATIVE_ROWS, ISLAND_CONTACTS))
    constraint_slices: IslandSlices
    constraint_id: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Island of each constraint, -1 for one touching no island
    constraint_island_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Hibernation (empty unless use_hibernation): is_hibernated marks an island whose every link is asleep, and
    # hibernated_next_link is the per-link daisy chain the partition build re-unions before labeling, which keeps a
    # sleeping component, whose bodies generate no live contacts, one island across steps.
    is_hibernated: qd.Tensor = field(int, shape=(TREES, ENVS), kind=DataKind.STATE, labels=HIBERNATION)
    hibernated_next_link: qd.Tensor = field(int, shape=(LINKS, ENVS), kind=DataKind.STATE, labels=HIBERNATION)
    # (env, island) work-lists of the cooperative per-island factor+solve, one per island size class (see
    # island_tile_caps): class c holds its factor_worklist_size[c] awake islands in its own region of the two index
    # lists, of n_trees * _B slots each, in the order of the atomic reservation of the partition pass. Read where an env
    # can hold several islands (see func_island_tiled_factor_solve_all in solver.py).
    factor_worklist_i_b: qd.Tensor = field(int, shape=(WORKLIST,), labels=WORKLISTS)
    factor_worklist_i_island: qd.Tensor = field(int, shape=(WORKLIST,), labels=WORKLISTS)
    factor_worklist_size: qd.Tensor = field(int, shape=(TILE_CLASSES,), labels=WORKLISTS)
    # Scratch of the reverse Cuthill-McKee tree reorder of the CPU per-island skyline path: a tree's island-local slot,
    # contact degree and ordered flag, and the resulting visit order. Distinct buffers, since quadrants assumes
    # distinct kernel arguments never alias.
    rcm_tree_pos: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_degree: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_is_ordered: qd.Tensor = field(bool, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_order: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    # Per-island Newton / CG iteration state (see linesearch.py), indexed [i_island, i_b]: inertia is the trace of the
    # mass matrix over the island's dofs, the scale of its convergence tests, and improved says whether the island
    # still iterates, every pass of an iteration being confined to the islands that do. ls_improvement carries
    # -cost(alpha_accepted) in shifted form, cost(alpha) - cost(0), which stays resolvable in float32 near convergence.
    inertia: qd.Tensor = field(float, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    improved: qd.Tensor = field(bool, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    ls_improvement: qd.Tensor = field(float, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)


# =========================================== IKState ===========================================


# Axes of the inverse-kinematics scratch, sized by the largest entity of the solver and the target cap
IK_QS = Axis("ik_qs")
IK_DOFS = Axis("ik_dofs")
IK_LINKS = Axis("ik_links")
IK_JOINTS = Axis("ik_joints")
IK_TARGETS = Axis("ik_targets")
# The pose residuals of the targets stacked, six per target
IK_ERRORS = Axis("ik_errors")


class IKState(Data, kind=DataKind.SCRATCH):
    """Damped-least-squares inverse-kinematics scratch, one column per parallel solve.

    A column is an environment for the entity API or a candidate restart for the planner.
    """

    # Single-target spatial Jacobian (6 x n_dofs), rebuilt per target link and copied into the stacked block.
    jacobian: qd.Tensor = field(float, shape=(6, IK_DOFS, ENVS))
    # Stacked multi-target Jacobian and its transpose, feeding the damped normal-equations solve.
    jacobian_stacked: qd.Tensor = field(float, shape=(IK_ERRORS, IK_DOFS, ENVS))
    jacobian_stacked_t: qd.Tensor = field(float, shape=(IK_DOFS, IK_ERRORS, ENVS))
    # Damped normal matrix J J^T + damping^2 I, its lower / upper LU factors and forward-substitution scratch, and
    # its inverse.
    mat: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    lu_lower: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    lu_upper: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    lu_y: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    inv: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    # Stacked pose residual, the best residual seen across restarts, and inv @ residual.
    err_pose: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    err_pose_best: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    vec: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    # Joint-space step and the best configuration found across restarts.
    delta_qpos: qd.Tensor = field(float, shape=(IK_DOFS, ENVS))
    qpos_best: qd.Tensor = field(float, shape=(IK_QS, ENVS))


class IKScratchFK(Data, kind=DataKind.SCRATCH):
    """Per-column forward-kinematics scratch for the entity inverse-kinematics solve.

    Holds the working configuration and the link / joint frames it produces, so each Gauss-Newton step evaluates
    and integrates poses on caller-owned buffers without mutating live solver state. A column is an environment.
    """

    # Entity-local working configuration iterated by the solve.
    qpos: qd.Tensor = field(float, shape=(IK_QS, ENVS))
    # Link poses and joint frames placed by func_forward_kinematics_scratch, feeding the error and the Jacobian.
    links_pos: qd.Tensor = field((float, 3), shape=(IK_LINKS, ENVS))
    links_quat: qd.Tensor = field((float, 4), shape=(IK_LINKS, ENVS))
    joints_xanchor: qd.Tensor = field((float, 3), shape=(IK_JOINTS, ENVS))
    joints_xaxis: qd.Tensor = field((float, 3), shape=(IK_JOINTS, ENVS))


class IKTargets(Data, kind=DataKind.SCRATCH):
    """Inverse-kinematics targets and DOF selection for one solve batch, one column per parallel solve.

    A column is an environment for the entity API or a candidate restart for the planner. Positions and
    orientations are vec-typed so the solver reads a target pose per (link, column) directly; the host populates
    every field before the solve.
    """

    # Global indices of the target links, the effective DOFs to move, and the columns (solver envs) to solve.
    links_idx: qd.Tensor = field(int, shape=(IK_TARGETS,))
    dofs_idx: qd.Tensor = field(int, shape=(IK_DOFS,))
    envs_idx: qd.Tensor = field(int, shape=(ENVS,))
    # Target pose per (link, column) and the link-local point whose pose is driven, in the world frame.
    pos: qd.Tensor = field((float, 3), shape=(IK_TARGETS, ENVS))
    quat: qd.Tensor = field((float, 4), shape=(IK_TARGETS, ENVS))
    local_point: qd.Tensor = field((float, 3), shape=(IK_TARGETS,))
    # Optional custom initial configuration per (column, q); used only when custom_init_qpos is set.
    init_qpos: qd.Tensor = field(float, shape=(ENVS, IK_QS))
    # Which position / rotation axes to solve (shared across links), and per-link position / rotation enables.
    pos_mask: qd.Tensor = field(float, shape=(3,))
    rot_mask: qd.Tensor = field(float, shape=(3,))
    link_pos_mask: qd.Tensor = field(int, shape=(IK_TARGETS,))
    link_rot_mask: qd.Tensor = field(int, shape=(IK_TARGETS,))


class IKScratch(Data, kind=DataKind.SCRATCH):
    """The three solve-time buffers an inverse-kinematics call needs, shared by every entity of one solver.

    Sized by the largest entity the solver holds and by the target cap, so each solve writes the leading rows its own
    entity spans; a column is an environment.
    """

    state: IKState
    fk: IKScratchFK
    targets: IKTargets


# The dofs of the widest entity of the solver, and the scratch of its kinematics queries
ENTITY_DOFS = Axis("entity_dofs")
KINEMATICS = Label("kinematics")


class KinematicsScratch(Data, kind=DataKind.SCRATCH):
    """Buffers backing the kinematics queries of every entity of one solver.

    A column is an environment. The Jacobian spans the widest entity, and the configuration cache spans the solver
    configuration so the global q indices callers already work in index it directly.
    """

    jacobian: qd.Tensor = field(float, shape=(6, ENTITY_DOFS, ENVS), labels=KINEMATICS)
    qpos_cache: qd.Tensor = field(float, shape=(QS, ENVS), labels=KINEMATICS)


class WeightScratch(Data, kind=DataKind.SCRATCH, shape=(DOFS, ENVS)):
    """Buffers used to compute the constraint inverse weights of one solver.

    One weight needs one row of a link Jacobian and the mass-matrix solve against that row.
    """

    jac_row: qd.Tensor = field(float)
    solve_out: qd.Tensor = field(float)


class ConstraintState(Data, kind=DataKind.SCRATCH):
    # /!\ The declaration order is the allocation order, which changing may reduce runtime speed by >10% /!\
    n_constraints: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    qd_n_equalities: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)
    n_constraints_equality: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    n_constraints_frictionloss: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    # Number of elliptic-cone contact rows (rows_per_contact per contact), laid out contiguously at the start of the
    # collision segment. Zero for the pyramidal cone. The cone rows occupy
    # [ne + n_frictionloss, ne + n_frictionloss + n_cone).
    n_constraints_cone: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    is_warmstart: qd.Tensor = field(bool, shape=(ENVS,), kind=DataKind.STATE)
    improved: qd.Tensor = field(bool, shape=(ENVS,))
    cost_ws: qd.Tensor = field(float, shape=(ENVS,))
    cost: qd.Tensor = field(float, shape=(ENVS,))
    # DOF-vec family flip: canonical (n_dofs_, _B) -> physical (_B, n_dofs_). Adjacent-lane reads striding i_d in
    # cooperative kernels become stride-1; the regression on 1T-per-(i_d, i_b) writers is patched on a per-consumer
    # basis under the same enable_cooperative_constraint_kernels flag.
    Ma: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    Ma_ws: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    grad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    Mgrad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Previous-iteration cone-row residuals, kept only for the CPU elliptic cone so the incremental factor can downdate
    # the prior coupled cone block by a per-iteration rank-3 update (Jaref is overwritten by the linesearch apply).
    cone_prev_jaref: qd.Tensor = field(float, shape=(CONE_ROWS, ENVS), labels=(SERIAL_ROWS, CONE_INCREMENTAL))
    # Per contact under 'signorini': the latched normal force scaling its friction discs (negative until seeded, see
    # func_relatch_cone in solver.py), and the gap between the iterate's normal force and the latch at the last update.
    cone_latch: qd.Tensor = field(float, shape=(CONES, ENVS), labels=SIGNORINI)
    cone_latch_gap: qd.Tensor = field(float, shape=(CONES, ENVS), labels=SIGNORINI)
    search: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    qfrc_constraint: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    qacc: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    qacc_ws: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    mv: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Optional CG fields
    cg_prev_grad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    cg_prev_Mgrad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Optional Newton fields
    # Rank-1 working vectors of the incremental Cholesky update, flattened slot-minor as [i_d * n_slots + i_u]: one slot
    # per fused update of the per-island factor (func_rank_batch_update_island), a single slot for the whole-env factor
    # of a single-island scene (indexing then reduces to [i_d]). Flat 2D to keep the DOF-vec rank and layout everywhere.
    nt_vec: qd.Tensor = field(float, shape=(NT_VEC_ROWS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    # Hessian matrix of the optimization problem as a dense 2D tensor.
    # Note that only the lower triangular part is updated for efficiency because this matrix is symmetric by definition.
    # As a result, the values of the strictly upper triangular part is undefined - except under
    # enable_cone_free_hessian_reuse, where each used lower-triangle slot's mirror persists the cone-free assembled
    # Hessian (M + J^T D J of the active rows only), maintained across a step's Newton iterations by signed flip
    # scatters; its diagonal lives in nt_H_cone_free_diag since the lower diagonal belongs to the factor. A rebuild
    # restores the mirror into the lower triangle and bakes the current cone blocks on top, skipping the full
    # J^T D J reassembly.
    # In practice, this variable is re-purposed to store the Cholesky factor L st H = L @ L.T to spare memory resources.
    # Under the register-tiled mass factor it is the scratch of the factor (see the constraint solver build).
    # TODO: Optimize storage to only allocate memory half of the Hessian matrix to sparse memory resources.
    nt_H: qd.Tensor = field(float, shape=(ENVS, DOFS, DOFS), labels=NEWTON)
    # Per-DOF Jacobi scale s_i = 1/sqrt(diag(M + J^T D J)_i), at natural DOF id, refreshed ahead of every direct
    # rebuild (func_jacobi_scale, 1 on an empty diagonal). Assembly writes H already scaled, nt_H holds L of S H S,
    # update vectors are scaled at construction, and the batch solve wraps grad/Mgrad, so Mgrad = H^-1 grad exactly. A
    # unit diagonal keeps the Hessian's mixed-unit spread within float32's conditioning capability at any geometry
    # scale and makes the bare-EPS pivot floor scale-relative. Only meaningful with enable_jacobi_equilibration.
    nt_jacobi: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    # Compacted list of constraints whose active state changed, used by incremental Cholesky update
    # to reduce GPU thread divergence by iterating only over constraints that need processing.
    incr_changed_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    incr_n_changed: qd.Tensor = field(int, shape=(ENVS,))
    # Layout-flippable constraint-state tensors: canonical shape (len_constraints_, _B), physically stored as
    # (_B, len_constraints_) under BATCH_FIRST, kernel-body indexing ``Jaref[i_c, i_b]`` being rewritten accordingly.
    active: qd.Tensor = field(bool, shape=(CONSTRAINTS, ENVS), kind=DataKind.DERIVED, labels=COOPERATIVE_ROWS)
    prev_active: qd.Tensor = field(bool, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    diag: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    aref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    Jaref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Frictionloss rows store their friction loss; elliptic-cone head (normal) rows reuse the field to carry the
    # contact sliding friction coefficient read by the cone solver, and with torsional friction the spin row carries
    # the torsional coefficient the same way (the tangent rows hold 0).
    efc_frictionloss: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    efc_force: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), kind=DataKind.DERIVED, grad=True, labels=COOPERATIVE_ROWS)
    efc_D: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    jv: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Canonical (len_constraints_, n_dofs_, _B), physical (_B, len_constraints_, n_dofs_) under JAC_ENV_FIRST, each row
    # of an env stored whole. The cooperative kernels then read the per-iteration J^T @ efc_force and the Hessian
    # staging with lanes on consecutive dofs of one row, and a row's support spans a cache line or two. Serialized
    # execution reads and writes the rows dof by dof for a fixed env.
    jac: qd.Tensor = field(float, shape=(CONSTRAINTS, DOFS, ENVS), labels=JACOBIAN_ROWS)
    # The support of each row: a row lists the dofs of the kinematic chains of the two links it couples, or at most two
    # dofs for a joint equality.
    jac_dofs_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, JAC_DOFS, ENVS), labels=JACOBIAN_ROWS)
    jac_n_dofs: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Backward gradients
    dL_dqacc: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    dL_dM: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), labels=BACKWARD)
    dL_djac: qd.Tensor = field(float, shape=(CONSTRAINTS, DOFS, ENVS), labels=BACKWARD)
    dL_daref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    dL_defc_D: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    dL_dforce: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    # Backward buffers for linear system solver
    bw_u: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_r: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_p: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_Ap: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_Ju: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    bw_y: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    bw_w: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    # Timers for profiling
    timers: qd.Tensor = field(_timer_dtype, shape=(10, ENVS))
    # Solver loop iteration counter (0-indexed, increments each iteration in the graph loop)
    solver_iter_counter: qd.Tensor = field(qd.i32, shape=())
    # Always ndarray (not field): graph.do_while requires the same physical ndarray on every call.
    graph_counter: qd.Tensor = field(qd.i32, shape=(), ndarray=True)
    # Iterations left to the graph loop past the current one, which it resumes with once an env iterates on
    graph_counter_next: qd.Tensor = field(qd.i32, shape=())
    # Diagonal of the persisted cone-free Hessian packed in nt_H's mirror slots (see nt_H). Only meaningful with
    # enable_cone_free_hessian_reuse.
    nt_H_cone_free_diag: qd.Tensor = field(float, shape=(ENVS, DOFS), labels=CONE_FREE_REUSE)
    # Scratch of the noslip sweep (empty when noslip is off): M^{-1} J^T of the two rows of the pair being updated, in
    # the two columns of the env, or of the lane of the cooperative sweep at [i_d, 2 * (i_b * 32 + tid) + i_r] (see
    # func_noslip_update_row in noslip.py).
    noslip_MinvJT: qd.Tensor = field(float, shape=(DOFS, NOSLIP_COLUMNS), labels=NOSLIP)
    # Row coloring of the cooperative noslip sweep (empty otherwise, see func_color_rows_batch in noslip.py): the color
    # of each row, the color count of each island and the next free color of the mass block starting at each dof.
    noslip_rows_color: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    noslip_islands_n_colors: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    noslip_blocks_n_colors: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    # Union-find partition of links into contact islands, read by the per-island Newton solve and hibernation.
    island: IslandState


# =========================================== Collider ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactData:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    geom_a: qd.Tensor
    geom_b: qd.Tensor
    penetration: qd.Tensor
    normal: qd.Tensor
    pos: qd.Tensor
    friction: qd.Tensor
    friction_torsional: qd.Tensor
    friction_rolling: qd.Tensor
    sol_params: qd.Tensor
    force: qd.Tensor
    link_a: qd.Tensor
    link_b: qd.Tensor
    pair_idx: qd.Tensor


def get_contact_data(solver, max_candidate_contacts, requires_grad):
    _B = solver._B
    max_candidate_contacts_ = max(max_candidate_contacts, 1)

    return ContactData(
        geom_a=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
        geom_b=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
        normal=V(dtype=gs.qd_vec3, shape=(max_candidate_contacts_, _B), needs_grad=requires_grad),
        pos=V(dtype=gs.qd_vec3, shape=(max_candidate_contacts_, _B), needs_grad=requires_grad),
        penetration=V(dtype=gs.qd_float, shape=(max_candidate_contacts_, _B), needs_grad=requires_grad),
        friction=V(dtype=gs.qd_float, shape=(max_candidate_contacts_, _B)),
        friction_torsional=V(dtype=gs.qd_float, shape=(max_candidate_contacts_, _B)),
        friction_rolling=V(dtype=gs.qd_float, shape=(max_candidate_contacts_, _B)),
        sol_params=V_VEC(7, dtype=gs.qd_float, shape=(max_candidate_contacts_, _B)),
        force=V(dtype=gs.qd_vec3, shape=(max_candidate_contacts_, _B)),
        link_a=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
        link_b=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
        pair_idx=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DiffContactInput:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    ### Non-differentiable input data
    # Geom id of the two geometries
    geom_a: qd.Tensor
    geom_b: qd.Tensor
    # Local positions of the 3 vertices from the two geometries that define the face on the Minkowski difference
    local_pos1_a: qd.Tensor
    local_pos1_b: qd.Tensor
    local_pos1_c: qd.Tensor
    local_pos2_a: qd.Tensor
    local_pos2_b: qd.Tensor
    local_pos2_c: qd.Tensor
    # Local positions of the 1 vertex from the two geometries that define the support point for the face above
    w_local_pos1: qd.Tensor
    w_local_pos2: qd.Tensor
    # Plane-convex contacts only: the convex support "core" (box vertex / sphere center / capsule nearest endpoint)
    # in the convex geom's local frame.
    core_local: qd.Tensor
    # Reference id of the contact point, which is needed for the backward pass
    ref_id: qd.Tensor
    # Flag whether the contact data can be computed in numerically stable way in both the forward and backward passes
    valid: qd.Tensor
    ### Differentiable input data
    # Reference penetration depth, which is needed for computing the weight of the contact point
    ref_penetration: qd.Tensor


def get_diff_contact_input(_B, max_contacts_per_pair, is_active, requires_grad=False):
    shape = maybe_shape((_B, max_contacts_per_pair), is_active and requires_grad)
    return DiffContactInput(
        geom_a=V(dtype=gs.qd_int, shape=shape),
        geom_b=V(dtype=gs.qd_int, shape=shape),
        local_pos1_a=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos1_b=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos1_c=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_a=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_b=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_c=V_VEC(3, dtype=gs.qd_float, shape=shape),
        w_local_pos1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        w_local_pos2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        core_local=V_VEC(3, dtype=gs.qd_float, shape=shape),
        ref_id=V(dtype=gs.qd_int, shape=shape),
        valid=V(dtype=gs.qd_int, shape=shape),
        ref_penetration=V(dtype=gs.qd_float, shape=shape, needs_grad=True),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SortBuffer:
    kind: ClassVar[DataKind] = DataKind.STATE

    value: qd.Tensor
    i_l: qd.Tensor
    is_max: qd.Tensor


def get_sort_buffer(solver):
    _B = solver._B

    return SortBuffer(
        value=V(dtype=gs.qd_float, shape=(2 * solver.n_links_, _B)),
        i_l=V(dtype=gs.qd_int, shape=(2 * solver.n_links_, _B)),
        is_max=V(dtype=gs.qd_bool, shape=(2 * solver.n_links_, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactCache:
    kind: ClassVar[DataKind] = DataKind.STATE

    normal: qd.Tensor
    # Previous-step penetration per pair (reset to 0 when out of contact), the warm-start for the MPR->GJK gate.
    penetration: qd.Tensor


def get_contact_cache(solver, n_possible_pairs, active):
    # Only the convex-convex detection reads the cache, so the other scenes keep it empty
    shape = maybe_shape((max(n_possible_pairs, 1), solver._B), active)
    return ContactCache(
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        penetration=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class NarrowphaseWorkQueue:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # The pair of every entry (see collision_pair_idx in ColliderInfo), queued for the multicontact pass, then for the
    # terrain pass, which runs after it
    i_b: qd.Tensor
    i_ga: qd.Tensor
    i_gb: qd.Tensor
    i_pair: qd.Tensor
    # Candidate contacts of the multicontact pass, one slot per contact a pair can hold. The contact0 kernel stores the
    # first contact of the pair in slot 0, the detections of the pass store theirs in their own slot, and the gather
    # accepts them in slot order according to their status (see MULTICONTACT_SLOT in collider/constants.py). Under the
    # contact patch an entry holds slot 0 alone and no status.
    contact_pos: qd.Tensor
    contact_normal: qd.Tensor
    contact_penetration: qd.Tensor
    contact_status: qd.Tensor
    # Where contact 0 of the entry stands (see MULTICONTACT_ENTRY in collider/constants.py). The contact0 kernel leaves
    # it to Gilbert-Johnson-Keerthi (GJK) when the per-pair gate from Minkowski portal refinement (MPR) to GJK fired,
    # and each perturbed contact otherwise falls back from MPR to GJK.
    contact0_status: qd.Tensor
    # The number of pairs queued and claimed so far
    n_pairs: qd.Tensor
    n_claimed: qd.Tensor


def get_narrowphase_work_queue(max_entries, n_slots, split_narrowphase, has_terrain):
    is_queue_active = split_narrowphase or has_terrain
    entries_shape = maybe_shape((max_entries,), is_queue_active)
    slots_shape = maybe_shape((max_entries, n_slots), split_narrowphase)
    return NarrowphaseWorkQueue(
        i_b=V(dtype=gs.qd_int, shape=entries_shape),
        i_ga=V(dtype=gs.qd_int, shape=entries_shape),
        i_gb=V(dtype=gs.qd_int, shape=entries_shape),
        i_pair=V(dtype=gs.qd_int, shape=entries_shape),
        contact_pos=V_VEC(3, dtype=gs.qd_float, shape=slots_shape),
        contact_normal=V_VEC(3, dtype=gs.qd_float, shape=slots_shape),
        contact_penetration=V(dtype=gs.qd_float, shape=slots_shape),
        contact_status=V(dtype=gs.qd_int, shape=maybe_shape((max_entries, n_slots), split_narrowphase and n_slots > 1)),
        contact0_status=V(dtype=gs.qd_int, shape=maybe_shape((max_entries,), split_narrowphase)),
        n_pairs=V(dtype=gs.qd_int, shape=maybe_shape((1,), is_queue_active)),
        n_claimed=V(dtype=gs.qd_int, shape=maybe_shape((1,), is_queue_active)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ColliderState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    sort_buffer: SortBuffer
    contact_data: ContactData
    active_buffer: qd.Tensor
    # Bounding box of every link, the box of its frame moved with it (see func_collision_clear in broadphase.py)
    links_aabb_min: qd.Tensor
    links_aabb_max: qd.Tensor
    # Valid link pairs whose boxes overlap, by index into ColliderInfo.valid_link_pairs
    n_broad_link_pairs: qd.Tensor
    broad_link_pairs: qd.Tensor
    n_broad_pairs: qd.Tensor
    broad_collision_pairs: qd.Tensor
    box_depth: qd.Tensor
    box_points: qd.Tensor
    box_pts: qd.Tensor
    box_lines: qd.Tensor
    box_linesu: qd.Tensor
    box_axi: qd.Tensor
    box_ppts2: qd.Tensor
    box_pu: qd.Tensor
    # The prism under the triangle every thread of the terrain pass tests, which reads its thread count from its shape
    prism: qd.Tensor
    n_contacts: qd.Tensor = of_kind(DataKind.DERIVED)
    # Kept contacts of the sleepers, at the front of the contact buffer with the identity permutation: a hibernated
    # link's contacts against fixed links and other sleepers are carried from step to step with the force of the last
    # solve they took part in, for the contact getters and the per-link contact force alone. Every per-step pass
    # (prune, sort, island edges, constraint rows, noslip) walks the live contacts after them,
    # [n_contacts_hibernated, n_contacts).
    n_contacts_hibernated: qd.Tensor = of_kind(DataKind.DERIVED)
    first_time: qd.Tensor = of_kind(DataKind.STATE)
    contact_cache: ContactCache
    # Input data for differentiable contact detection used in the backward pass
    diff_contact_input: DiffContactInput
    work_queue: NarrowphaseWorkQueue
    contact_sort_key: qd.Tensor
    contact_sort_idx: qd.Tensor = of_kind(DataKind.DERIVED)
    contact_proj_v: qd.Tensor
    contact_keep: qd.Tensor
    contact_hull_stack: qd.Tensor
    # Per-bucket lex sort permutation used by the cooperative dedup kernel
    # (func_clamp_prune_contacts_coop) for the phase-3 (u, v) lex sort. Sized to max_candidate_contacts because
    # each env writes its own permutation.
    contact_lex_idx: qd.Tensor


def get_collider_state(
    solver,
    rigid_config,
    n_possible_pairs,
    n_terrain_threads,
    collider_info,
    collider_static_config,
    split_narrowphase,
):
    _B = solver._B
    n_links = solver.n_links_
    n_valid_link_pairs = collider_info.n_valid_link_pairs[None]
    # Collider._init_max_contacts, which runs before this, sizes the contacts per regime (convex vs nonconvex) and the
    # broad phase candidates within the possible pairs.
    max_collision_pairs_broad = collider_info.max_collision_pairs_broad[None]
    max_candidate_contacts = max(collider_info.max_candidate_contacts[None], 1)
    requires_grad = rigid_config.requires_grad

    box_depth_shape = maybe_shape(
        (collider_static_config.n_contacts_per_nonconvex_pair, _B), rigid_config.box_box_detection
    )
    box_points_shape = maybe_shape(
        (collider_static_config.n_contacts_per_nonconvex_pair, _B), rigid_config.box_box_detection
    )
    box_pts_shape = maybe_shape((6, _B), rigid_config.box_box_detection)
    box_lines_shape = maybe_shape((4, _B), rigid_config.box_box_detection)
    box_linesu_shape = maybe_shape((4, _B), rigid_config.box_box_detection)
    box_axi_shape = maybe_shape((3, _B), rigid_config.box_box_detection)
    box_ppts2_shape = maybe_shape((4, 2, _B), rigid_config.box_box_detection)
    box_pu_shape = maybe_shape((4, _B), rigid_config.box_box_detection)
    # The pruning scratch is laid out environment-first, so that the threads of the block running an environment read
    # neighbouring slots from the same cache lines (see func_clamp_prune_contacts_coop)
    prune_shape = maybe_shape((max(max_candidate_contacts, 1), _B), collider_static_config.has_prunable_contacts)
    prune_layout = (1, 0) if collider_static_config.has_prunable_contacts else None

    return ColliderState(
        sort_buffer=get_sort_buffer(solver),
        active_buffer=V(dtype=gs.qd_int, shape=(n_links, _B)),
        links_aabb_min=V(dtype=gs.qd_vec3, shape=(n_links, _B)),
        links_aabb_max=V(dtype=gs.qd_vec3, shape=(n_links, _B)),
        n_broad_link_pairs=V(dtype=gs.qd_int, shape=(_B,)),
        broad_link_pairs=V(dtype=gs.qd_int, shape=(max(n_valid_link_pairs, 1), _B)),
        n_broad_pairs=V(dtype=gs.qd_int, shape=(_B,)),
        box_depth=V(dtype=gs.qd_float, shape=box_depth_shape),
        box_points=V_VEC(3, dtype=gs.qd_float, shape=box_points_shape),
        box_pts=V_VEC(3, dtype=gs.qd_float, shape=box_pts_shape),
        box_lines=V_VEC(6, dtype=gs.qd_float, shape=box_lines_shape),
        box_linesu=V_VEC(6, dtype=gs.qd_float, shape=box_linesu_shape),
        box_axi=V_VEC(3, dtype=gs.qd_float, shape=box_axi_shape),
        box_ppts2=V(dtype=gs.qd_float, shape=box_ppts2_shape),
        box_pu=V_VEC(3, dtype=gs.qd_float, shape=box_pu_shape),
        prism=V_VEC(3, dtype=gs.qd_float, shape=(6, n_terrain_threads)),
        n_contacts=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts_hibernated=V(dtype=gs.qd_int, shape=(_B,)),
        first_time=V(dtype=gs.qd_bool, shape=(_B,)),
        contact_cache=get_contact_cache(
            solver, n_possible_pairs, collider_static_config.has_non_box_plane_convex_convex
        ),
        broad_collision_pairs=V_VEC(2, dtype=gs.qd_int, shape=(max(max_collision_pairs_broad, 1), _B)),
        contact_data=get_contact_data(solver, max_candidate_contacts, requires_grad),
        diff_contact_input=get_diff_contact_input(_B, max(max_candidate_contacts, 1), True, requires_grad),
        # A pair holds its first contact and its four perturbed ones (see N_PERTURBATIONS in narrowphase.py), or its
        # first contact alone under the contact patch (see NarrowphaseWorkQueue).
        work_queue=get_narrowphase_work_queue(
            max(max_collision_pairs_broad, 1) * _B,
            1 if solver._options.enable_contact_patch else 5,
            split_narrowphase,
            collider_static_config.has_terrain,
        ),
        contact_sort_key=V(dtype=gs.qd_float, shape=(max(max_candidate_contacts, 1), _B), layout=(1, 0)),
        contact_sort_idx=V(dtype=gs.qd_int, shape=(max(max_candidate_contacts, 1), _B)),
        contact_proj_v=V(dtype=gs.qd_float, shape=prune_shape, layout=prune_layout),
        contact_keep=V(dtype=gs.qd_int, shape=prune_shape, layout=prune_layout),
        contact_hull_stack=V(dtype=gs.qd_int, shape=prune_shape, layout=prune_layout),
        contact_lex_idx=V(dtype=gs.qd_int, shape=prune_shape, layout=prune_layout),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VertsSpatialGrid:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    # Per-geom 8x8x8 grid over collision verts in the local AABB: a permutation of vert indices sorted by grid
    # cell (z fastest), the matching vert positions duplicated in that order for sequential streaming, and per-cell
    # vert ranges (8^3 + 1 entries per geom), so a scan visits only the cells overlapping a query box. The cell
    # mapping is anchored by geoms_origin / geoms_inv_cell_size in the geom frame.
    verts_idx: qd.Tensor
    verts_pos: qd.Tensor
    cells_vert_start: qd.Tensor
    geoms_origin: qd.Tensor
    geoms_inv_cell_size: qd.Tensor


def get_verts_spatial_grid(solver):
    return VertsSpatialGrid(
        verts_idx=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        verts_pos=V_VEC(3, dtype=gs.qd_float, shape=(solver.n_verts_,)),
        cells_vert_start=V(dtype=gs.qd_int, shape=(max(solver.n_geoms * (8**3 + 1), 1),)),
        geoms_origin=V_VEC(3, dtype=gs.qd_float, shape=(solver.n_geoms_,)),
        geoms_inv_cell_size=V_VEC(3, dtype=gs.qd_float, shape=(solver.n_geoms_,)),
    )


@qd.data_oriented
class ColliderStaticConfig(StaticConfig):
    has_terrain: bool
    # True when a convex-convex pair escapes func_narrow_phase_convex_specializations (box-box, plane-box) and reaches
    # the general narrowphase: the split narrowphase on GPU (contact0 + multicontact + sort), the monolithic
    # func_narrow_phase_convex_vs_convex on CPU. Also gates the deterministic contact sort (see func_sort_contacts),
    # which orders the contacts independently of the racy narrowphase layout. See its resolution in collider.py.
    has_non_box_plane_convex_convex: bool
    has_convex_specialization: bool
    has_nonconvex_nonterrain: bool
    # Whether the link-pair contact pruning pass is compiled in, see its resolution in collider.py
    has_prunable_contacts: bool
    # maximum number of contact pairs per collision pair
    n_contacts_per_convex_pair: int
    # maximum number of contact pairs per nonconvex (vertex-vs-SDF) collision pair; >= n_contacts_per_convex_pair
    n_contacts_per_nonconvex_pair: int
    # ccd algorithm
    ccd_algorithm: int
    # GPU core count and cores per compute unit, sizing the split narrow phase launches and scratch states
    gpu_cores: int
    gpu_cores_per_unit: int


# =========================================== MPR ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRSimplexSupport:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    v1: qd.Tensor
    v2: qd.Tensor
    v: qd.Tensor


def get_mpr_simplex_support(B_):
    return MPRSimplexSupport(
        v1=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
        v2=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
        v=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    simplex_support: MPRSimplexSupport
    simplex_size: qd.Tensor
    # What the depth of the contact in simplex_support[1..3] is worth, a PORTAL_STATUS value. Zero means no portal.
    # Only VALID portals are reused (perturbation reconstruction, EPA seeding); INVALID forces a GJK refine.
    portal_status: qd.Tensor


def get_mpr_state(B_):
    return MPRState(
        simplex_support=get_mpr_simplex_support(B_),
        simplex_size=V(dtype=gs.qd_int, shape=(B_,)),
        portal_status=V(dtype=gs.qd_int, shape=(B_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    CCD_EPS: qd.Tensor
    CCD_TOLERANCE: qd.Tensor
    CCD_ITERATIONS: qd.Tensor


def get_mpr_info(**kwargs):
    return MPRInfo(
        CCD_EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_EPS"]),
        CCD_TOLERANCE=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_TOLERANCE"]),
        CCD_ITERATIONS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_ITERATIONS"]),
    )


# =========================================== GJK ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MDVertex:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # Vertex of the Minkowski difference
    obj1: qd.Tensor
    obj2: qd.Tensor
    local_obj1: qd.Tensor
    local_obj2: qd.Tensor
    id1: qd.Tensor
    id2: qd.Tensor
    mink: qd.Tensor


def get_gjk_simplex_vertex(_B):
    shape = (_B, 4)
    return MDVertex(
        obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
        mink=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


def get_epa_polytope_vertex(_B, gjk_info):
    max_num_polytope_verts = 5 + gjk_info.epa_max_iterations[None]
    shape = (_B, max_num_polytope_verts)
    return MDVertex(
        obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
        mink=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKSimplex:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    nverts: qd.Tensor


def get_gjk_simplex(_B):
    shape = (_B,)
    return GJKSimplex(nverts=V(dtype=gs.qd_int, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKSimplexBuffer:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    normal: qd.Tensor
    sdist: qd.Tensor


def get_gjk_simplex_buffer(_B):
    shape = (_B, 4)
    return GJKSimplexBuffer(normal=V_VEC(3, dtype=gs.qd_float, shape=shape), sdist=V(dtype=gs.qd_float, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytope:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    nverts: qd.Tensor
    nfaces: qd.Tensor
    nfaces_map: qd.Tensor
    horizon_nedges: qd.Tensor
    horizon_w: qd.Tensor


def get_epa_polytope(_B):
    shape = (_B,)
    return EPAPolytope(
        nverts=V(dtype=gs.qd_int, shape=shape),
        nfaces=V(dtype=gs.qd_int, shape=shape),
        nfaces_map=V(dtype=gs.qd_int, shape=shape),
        horizon_nedges=V(dtype=gs.qd_int, shape=shape),
        horizon_w=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytopeFace:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    verts_idx: qd.Tensor
    adj_idx: qd.Tensor
    normal: qd.Tensor
    dist2: qd.Tensor
    map_idx: qd.Tensor
    visited: qd.Tensor


def get_epa_polytope_face(_B, polytope_max_faces):
    shape = (_B, polytope_max_faces)
    return EPAPolytopeFace(
        verts_idx=V_VEC(3, dtype=gs.qd_int, shape=shape),
        adj_idx=V_VEC(3, dtype=gs.qd_int, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        dist2=V(dtype=gs.qd_float, shape=shape),
        map_idx=V(dtype=gs.qd_int, shape=shape),
        visited=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytopeHorizonData:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    face_idx: qd.Tensor
    edge_idx: qd.Tensor


def get_epa_polytope_horizon_data(_B, polytope_max_horizons):
    shape = (_B, polytope_max_horizons)
    return EPAPolytopeHorizonData(face_idx=V(dtype=gs.qd_int, shape=shape), edge_idx=V(dtype=gs.qd_int, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactFace:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    vert1: qd.Tensor
    vert2: qd.Tensor
    endverts: qd.Tensor
    normal1: qd.Tensor
    normal2: qd.Tensor
    id1: qd.Tensor
    id2: qd.Tensor


def get_contact_face(_B, max_contact_polygon_verts):
    shape = (_B, max_contact_polygon_verts)
    return ContactFace(
        vert1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vert2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        endverts=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactNormal:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    endverts: qd.Tensor
    normal: qd.Tensor
    id: qd.Tensor


def get_contact_normal(_B, max_contact_polygon_verts):
    shape = (_B, max_contact_polygon_verts)
    return ContactNormal(
        endverts=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactHalfspace:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    normal: qd.Tensor
    dist: qd.Tensor


def get_contact_halfspace(_B, max_contact_polygon_verts):
    shape = (_B, max_contact_polygon_verts)
    return ContactHalfspace(normal=V_VEC(3, dtype=gs.qd_float, shape=shape), dist=V(dtype=gs.qd_float, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class Witness:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    point_obj1: qd.Tensor
    point_obj2: qd.Tensor


def get_witness(_B, max_contacts_per_pair):
    shape = (_B, max_contacts_per_pair)
    return Witness(
        point_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape), point_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape)
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    support_mesh_prev_vertex_id: qd.Tensor
    simplex_vertex: MDVertex
    simplex_buffer: GJKSimplexBuffer
    simplex: GJKSimplex
    simplex_vertex_intersect: MDVertex
    simplex_buffer_intersect: GJKSimplexBuffer
    nsimplex: qd.Tensor
    last_searched_simplex_vertex_id: qd.Tensor
    polytope: EPAPolytope
    polytope_verts: MDVertex
    polytope_faces: EPAPolytopeFace
    polytope_faces_map: qd.Tensor
    polytope_horizon_data: EPAPolytopeHorizonData
    polytope_horizon_stack: EPAPolytopeHorizonData
    contact_faces: ContactFace
    contact_normals: ContactNormal
    contact_halfspaces: ContactHalfspace
    contact_clipped_polygons: qd.Tensor
    multi_contact_flag: qd.Tensor
    witness: Witness
    n_witness: qd.Tensor
    n_contacts: qd.Tensor
    contact_pos: qd.Tensor
    normal: qd.Tensor
    is_col: qd.Tensor
    penetration: qd.Tensor
    distance: qd.Tensor
    # Index of the EPA polytope face nearest to the origin (the penetration face), or -1 when EPA did not run. Lets the
    # multi-contact perturbation reconstruct the exact unperturbed normal from that face's un-rotated support simplex.
    nearest_face: qd.Tensor
    # Differentiable contact detection
    diff_contact_input: DiffContactInput
    n_diff_contact_input: qd.Tensor
    diff_penetration: qd.Tensor


def get_gjk_state(_B, rigid_config, gjk_info, requires_grad=False):
    enable_mujoco_compatibility = rigid_config.enable_mujoco_compatibility
    polytope_max_faces = gjk_info.polytope_max_faces[None]
    max_contacts_per_pair = gjk_info.max_contacts_per_pair[None]
    max_contact_polygon_verts = gjk_info.max_contact_polygon_verts[None]

    # FIXME: Define GJKState and MujocoCompatGJKState that derives from the former but defines additional attributes
    return GJKState(
        # GJK simplex
        support_mesh_prev_vertex_id=V(dtype=gs.qd_int, shape=(_B, 2)),
        simplex_vertex=get_gjk_simplex_vertex(_B),
        simplex_buffer=get_gjk_simplex_buffer(_B),
        simplex=get_gjk_simplex(_B),
        last_searched_simplex_vertex_id=V(dtype=gs.qd_int, shape=(_B,)),
        simplex_vertex_intersect=get_gjk_simplex_vertex(_B),
        simplex_buffer_intersect=get_gjk_simplex_buffer(_B),
        nsimplex=V(dtype=gs.qd_int, shape=(_B,)),
        # EPA polytope
        polytope=get_epa_polytope(_B),
        polytope_verts=get_epa_polytope_vertex(_B, gjk_info),
        polytope_faces=get_epa_polytope_face(_B, polytope_max_faces),
        polytope_faces_map=V(dtype=gs.qd_int, shape=(_B, polytope_max_faces)),
        polytope_horizon_data=get_epa_polytope_horizon_data(_B, 6 + gjk_info.epa_max_iterations[None]),
        polytope_horizon_stack=get_epa_polytope_horizon_data(_B, polytope_max_faces * 3),
        # Multi-contact detection (MuJoCo compatibility)
        contact_faces=get_contact_face(_B, max_contact_polygon_verts),
        contact_normals=get_contact_normal(_B, max_contact_polygon_verts),
        contact_halfspaces=get_contact_halfspace(_B, max_contact_polygon_verts),
        contact_clipped_polygons=V_VEC(3, dtype=gs.qd_float, shape=(_B, 2, max_contact_polygon_verts)),
        multi_contact_flag=V(dtype=gs.qd_bool, shape=(_B,)),
        # Final results
        witness=get_witness(_B, max_contacts_per_pair),
        n_witness=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts=V(dtype=gs.qd_int, shape=(_B,)),
        contact_pos=V_VEC(3, dtype=gs.qd_float, shape=(_B, max_contacts_per_pair)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(_B, max_contacts_per_pair)),
        is_col=V(dtype=gs.qd_bool, shape=(_B,)),
        penetration=V(dtype=gs.qd_float, shape=(_B,)),
        distance=V(dtype=gs.qd_float, shape=(_B,)),
        nearest_face=V(dtype=gs.qd_int, shape=(_B,)),
        diff_contact_input=get_diff_contact_input(
            _B, max(max_contacts_per_pair, 1), is_active=True, requires_grad=requires_grad
        ),
        n_diff_contact_input=V(dtype=gs.qd_int, shape=(_B,)),
        diff_penetration=V(dtype=gs.qd_float, shape=maybe_shape((_B, max_contacts_per_pair), requires_grad)),
    )


def get_gjk_state_contact_only(_B):
    """Minimal GJK state for contact detection only (no EPA, no multi-contact).

    Used by kernel 1 to run func_gjk as a boolean overlap test. All EPA polytope,
    multi-contact, and differentiable fields are allocated at dummy size (1,) since
    func_gjk never accesses them.
    """
    _dummy_B = 1

    return GJKState(
        support_mesh_prev_vertex_id=V(dtype=gs.qd_int, shape=(_B, 2)),
        simplex_vertex=get_gjk_simplex_vertex(_B),
        simplex_buffer=get_gjk_simplex_buffer(_B),
        simplex=get_gjk_simplex(_B),
        last_searched_simplex_vertex_id=V(dtype=gs.qd_int, shape=(_B,)),
        simplex_vertex_intersect=get_gjk_simplex_vertex(_B),
        simplex_buffer_intersect=get_gjk_simplex_buffer(_B),
        nsimplex=V(dtype=gs.qd_int, shape=(_B,)),
        # EPA - dummy allocations, never accessed by func_gjk
        polytope=get_epa_polytope(_dummy_B),
        polytope_verts=MDVertex(
            obj1=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            obj2=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            local_obj1=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            local_obj2=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            id1=V(dtype=gs.qd_int, shape=(1, 1)),
            id2=V(dtype=gs.qd_int, shape=(1, 1)),
            mink=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        ),
        polytope_faces=get_epa_polytope_face(_dummy_B, 1),
        polytope_faces_map=V(dtype=gs.qd_int, shape=(1, 1)),
        polytope_horizon_data=get_epa_polytope_horizon_data(_dummy_B, 1),
        polytope_horizon_stack=get_epa_polytope_horizon_data(_dummy_B, 1),
        # Multi-contact - dummy
        contact_faces=get_contact_face(_dummy_B, 1),
        contact_normals=get_contact_normal(_dummy_B, 1),
        contact_halfspaces=get_contact_halfspace(_dummy_B, 1),
        contact_clipped_polygons=V_VEC(3, dtype=gs.qd_float, shape=(1, 2, 1)),
        multi_contact_flag=V(dtype=gs.qd_bool, shape=(_B,)),
        # Results - full _B for fields func_gjk writes; dummy for EPA-only fields
        witness=get_witness(_B, 1),
        n_witness=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts=V(dtype=gs.qd_int, shape=(1,)),
        contact_pos=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        is_col=V(dtype=gs.qd_bool, shape=(1,)),
        penetration=V(dtype=gs.qd_float, shape=(1,)),
        distance=V(dtype=gs.qd_float, shape=(_B,)),
        nearest_face=V(dtype=gs.qd_int, shape=(_B,)),
        diff_contact_input=get_diff_contact_input(_dummy_B, 1, is_active=False),
        n_diff_contact_input=V(dtype=gs.qd_int, shape=(1,)),
        diff_penetration=V(dtype=gs.qd_float, shape=()),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    max_contacts_per_pair: qd.Tensor
    max_contact_polygon_verts: qd.Tensor
    # Maximum number of iterations for GJK and EPA algorithms
    gjk_max_iterations: qd.Tensor
    epa_max_iterations: qd.Tensor
    FLOAT_MIN: qd.Tensor
    FLOAT_MIN_SQ: qd.Tensor
    FLOAT_MAX: qd.Tensor
    FLOAT_MAX_SQ: qd.Tensor
    # Tolerance for stopping GJK and EPA algorithms when they converge (only for non-discrete geometries).
    tolerance: qd.Tensor
    # If the distance between two objects is smaller than this value, we consider them colliding.
    collision_eps: qd.Tensor
    # In safe GJK, we do not allow degenerate simplex to happen, because it becomes the main reason of EPA errors.
    # To prevent degeneracy, we throw away the simplex that has smaller degeneracy measure (e.g. colinearity,
    # coplanarity) than this threshold.
    simplex_max_degeneracy_sq: qd.Tensor
    polytope_max_faces: qd.Tensor
    # Threshold for reprojection error when we compute the witness points from the polytope. In computing the
    # witness points, we project the origin onto the polytope faces and compute the barycentric coordinates of the
    # projected point. To confirm the projection is valid, we compute the projected point using the barycentric
    # coordinates and compare it with the original projected point. If the difference is larger than this threshold,
    # we consider the projection invalid, because it means numerical errors are too large.
    # We check both relative and absolute errors: the relative error catches numerically degenerate faces,
    # while the absolute error prevents false rejections on smooth geometries (e.g. spheres) where
    # polytope faces become extremely small near convergence, amplifying the relative error.
    polytope_max_rel_reprojection_error: qd.Tensor
    polytope_max_abs_reprojection_error: qd.Tensor
    # Tolerance for normal alignment between (face-face) or (edge-face). The normals should align within this
    # tolerance to be considered as a valid parallel contact.
    contact_face_tol: qd.Tensor
    contact_edge_tol: qd.Tensor
    # Epsilon values for differentiable contact. [eps_boundary] denotes the maximum distance between the face
    # and the support point in the direction of the face normal. If this distance is 0, the face is on the
    # boundary of the Minkowski difference. For [eps_distance], the distance between the origin and the face
    # should not exceed this eps value plus the default EPA depth. For [eps_affine], the affine coordinates
    # of the origin's projection onto the face should not violate [0, 1] range by this eps value.
    # FIXME: Adjust these values based on the case study.
    diff_contact_eps_boundary: qd.Tensor
    diff_contact_eps_distance: qd.Tensor
    diff_contact_eps_affine: qd.Tensor
    # The minimum norm of the normal to be considered as a valid normal in the differentiable formulation.
    diff_contact_min_normal_norm: qd.Tensor
    # The minimum penetration depth to be considered as a valid contact in the differentiable formulation.
    # The contact with penetration depth smaller than this value is ignored in the differentiable formulation.
    # This should be large enough to be safe from numerical errors, because in the backward pass, the computed
    # penetration depth could be different from the forward pass due to the numerical errors. If this value is
    # too small, the non-zero penetration depth could be falsely computed to 0 in the backward pass and thus
    # produce nan values for the contact normal.
    diff_contact_min_penetration: qd.Tensor


def get_gjk_info(**kwargs):
    return GJKInfo(
        max_contacts_per_pair=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["max_contacts_per_pair"]),
        max_contact_polygon_verts=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["max_contact_polygon_verts"]),
        gjk_max_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["gjk_max_iterations"]),
        epa_max_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["epa_max_iterations"]),
        FLOAT_MIN=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MIN"]),
        FLOAT_MIN_SQ=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MIN"] ** 2),
        FLOAT_MAX=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MAX"]),
        FLOAT_MAX_SQ=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MAX"] ** 2),
        tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["tolerance"]),
        collision_eps=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["collision_eps"]),
        simplex_max_degeneracy_sq=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["simplex_max_degeneracy_sq"]),
        polytope_max_faces=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["polytope_max_faces"]),
        polytope_max_rel_reprojection_error=V_SCALAR_FROM(
            dtype=gs.qd_float, value=kwargs["polytope_max_rel_reprojection_error"]
        ),
        polytope_max_abs_reprojection_error=V_SCALAR_FROM(
            dtype=gs.qd_float, value=kwargs["polytope_max_abs_reprojection_error"]
        ),
        contact_face_tol=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["contact_face_tol"]),
        contact_edge_tol=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["contact_edge_tol"]),
        diff_contact_eps_boundary=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_boundary"]),
        diff_contact_eps_distance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_distance"]),
        diff_contact_eps_affine=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_affine"]),
        diff_contact_min_normal_norm=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_min_normal_norm"]),
        diff_contact_min_penetration=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_min_penetration"]),
    )


@qd.data_oriented
class GJKStaticConfig(StaticConfig):
    enable_contact_patch: bool


# =========================================== SupportField ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SupportFieldInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    # First cell of the table of each geom, the geoms of one trimesh sharing one table
    support_cell_start: qd.Tensor
    # First extra candidate of each cell, past those of its record, its extras ending where those of the next start
    support_list_start: qd.Tensor
    # Mesh-frame position and index of each candidate vertex, among the vertices of its geom: the record of every cell
    # at twice the index of the cell, holding the first two candidates of the cell (a single candidate twice), then the
    # extra candidates of all cells
    support_v: qd.Tensor
    support_vid: qd.Tensor
    support_res: qd.Tensor


def get_support_field_info(n_geoms, n_support_cells, n_support_candidates, support_res):
    return SupportFieldInfo(
        support_cell_start=V(dtype=gs.qd_int, shape=(max(n_geoms, 1),)),
        support_list_start=V(dtype=gs.qd_int, shape=(n_support_cells + 1,)),
        support_v=V_VEC(3, dtype=gs.qd_float, shape=(max(n_support_candidates, 1),)),
        support_vid=V(dtype=gs.qd_int, shape=(max(n_support_candidates, 1),)),
        support_res=V_SCALAR_FROM(dtype=gs.qd_int, value=support_res),
    )


# =========================================== SDF ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SDFGeomInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    T_mesh_to_sdf: qd.Tensor
    sdf_res: qd.Tensor
    sdf_max: qd.Tensor
    sdf_cell_size: qd.Tensor
    sdf_cell_start: qd.Tensor
    # Coarse min-grid companion: per-block minima over grid nodes, a certified lower bound of the trilinear sd.
    sdf_coarse_res: qd.Tensor
    sdf_coarse_cell_start: qd.Tensor


def get_sdf_geom_info(n_geoms):
    return SDFGeomInfo(
        T_mesh_to_sdf=V_MAT(n=4, m=4, dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_res=V_VEC(3, dtype=gs.qd_int, shape=(n_geoms,)),
        sdf_max=V(dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_cell_size=V_VEC(3, dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_cell_start=V(dtype=gs.qd_int, shape=(n_geoms,)),
        sdf_coarse_res=V_VEC(3, dtype=gs.qd_int, shape=(n_geoms,)),
        sdf_coarse_cell_start=V(dtype=gs.qd_int, shape=(n_geoms,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SDFInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    geoms_info: SDFGeomInfo
    geoms_sdf_val: qd.Tensor
    geoms_sdf_grad: qd.Tensor
    geoms_sdf_closest_vert: qd.Tensor
    geoms_sdf_coarse_val: qd.Tensor


def get_sdf_info(n_geoms, n_cells, n_coarse_cells):
    if not is_tensor_addressable((n_cells, 3), gs.np_float):
        gs.raise_exception(
            f"SDF Gradient shape (n_cells={n_cells}, 3) is too large. Consider setting a larger 'sdf_cell_size', or a "
            "smaller 'sdf_max_res', in 'gs.materials.Rigid' options."
        )

    return SDFInfo(
        geoms_info=get_sdf_geom_info(max(n_geoms, 1)),
        geoms_sdf_val=V(dtype=gs.qd_float, shape=(max(n_cells, 1),)),
        geoms_sdf_grad=V_VEC(3, dtype=gs.qd_float, shape=(max(n_cells, 1),)),
        geoms_sdf_closest_vert=V(dtype=gs.qd_int, shape=(max(n_cells, 1),)),
        geoms_sdf_coarse_val=V(dtype=gs.qd_float, shape=(max(n_coarse_cells, 1),)),
    )


# =========================================== ColliderInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ColliderInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    # Narrowphase sub-component descriptions, owned by their respective builders (MPR, GJK, SupportField, SDF) and
    # embedded here so kernels receive the whole collider description as one argument. They are built at their final
    # size before this struct is.
    mpr: MPRInfo
    gjk: GJKInfo
    support_field: SupportFieldInfo
    sdf: SDFInfo
    vert_neighbors: qd.Tensor
    vert_neighbor_start: qd.Tensor
    vert_n_neighbors: qd.Tensor
    verts_spatial_grid: VertsSpatialGrid
    # (i_ga, i_gb) -> dense pair index, or -1 if invalid. Used by the broadphase, the narrowphase and the contact cache.
    collision_pair_idx: qd.Tensor
    # (i_la, i_lb) -> index into valid_link_pairs, or -1 if no geom pair between the two links is valid
    link_pair_idx: qd.Tensor
    # Lower and upper corners of the box of every link in its own frame, bounding the boxes of its geoms
    links_init_aabb_min: qd.Tensor
    links_init_aabb_max: qd.Tensor
    max_possible_pairs: qd.Tensor
    max_collision_pairs: qd.Tensor
    max_candidate_contacts: qd.Tensor
    max_collision_pairs_broad: qd.Tensor
    # Post-pruning contact-point budget per environment, which sizes the contact constraint buffers (4 constraints per
    # contact point). Smaller than max_candidate_contacts when contact pruning is enabled or 'max_contacts' is set.
    max_contacts: qd.Tensor
    # Link pairs holding at least one valid geom pair, the link carrying most geoms first (see func_broad_phase_geoms)
    n_valid_link_pairs: qd.Tensor
    valid_link_pairs: qd.Tensor
    # Terrain fields
    terrain_hf: qd.Tensor
    terrain_rc: qd.Tensor
    terrain_scale: qd.Tensor
    terrain_xyz_maxmin: qd.Tensor
    # multi contact perturbation and tolerance
    mc_perturbation: qd.Tensor
    mc_tolerance: qd.Tensor
    # differentiable contact tolerance
    diff_pos_tolerance: qd.Tensor
    diff_normal_tolerance: qd.Tensor
    # link-pair contact pruning
    contact_pruning_tolerance: qd.Tensor
    prune_deep_penetration_ratio: qd.Tensor


def get_collider_info(
    solver,
    n_vert_neighbors,
    n_valid_link_pairs,
    collider_static_config,
    mpr_info,
    gjk_info,
    support_field_info,
    sdf_info,
    **kwargs,
):
    for geom in solver.geoms:
        if geom.type == gs.GEOM_TYPE.TERRAIN:
            terrain_hf_shape = geom.entity.terrain_hf.shape
            break
    else:
        terrain_hf_shape = 1

    return ColliderInfo(
        mpr=mpr_info,
        gjk=gjk_info,
        support_field=support_field_info,
        sdf=sdf_info,
        vert_neighbors=V(dtype=gs.qd_int, shape=(max(n_vert_neighbors, 1),)),
        vert_neighbor_start=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        vert_n_neighbors=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        verts_spatial_grid=get_verts_spatial_grid(solver),
        collision_pair_idx=V(dtype=gs.qd_int, shape=(solver.n_geoms_, solver.n_geoms_)),
        link_pair_idx=V(dtype=gs.qd_int, shape=(solver.n_links_, solver.n_links_)),
        links_init_aabb_min=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        links_init_aabb_max=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        max_possible_pairs=V(dtype=gs.qd_int, shape=()),
        max_collision_pairs=V(dtype=gs.qd_int, shape=()),
        max_candidate_contacts=V(dtype=gs.qd_int, shape=()),
        max_collision_pairs_broad=V(dtype=gs.qd_int, shape=()),
        max_contacts=V(dtype=gs.qd_int, shape=()),
        n_valid_link_pairs=V_SCALAR_FROM(dtype=gs.qd_int, value=n_valid_link_pairs),
        valid_link_pairs=V(dtype=gs.qd_ivec2, shape=(max(n_valid_link_pairs, 1),)),
        terrain_hf=V(dtype=gs.qd_float, shape=terrain_hf_shape),
        terrain_rc=V(dtype=gs.qd_int, shape=(2,)),
        terrain_scale=V(dtype=gs.qd_float, shape=(2,)),
        terrain_xyz_maxmin=V(dtype=gs.qd_float, shape=(6,)),
        mc_perturbation=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mc_perturbation"]),
        mc_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mc_tolerance"]),
        diff_pos_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_pos_tolerance"]),
        diff_normal_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_normal_tolerance"]),
        contact_pruning_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["contact_pruning_tolerance"]),
        prune_deep_penetration_ratio=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["prune_deep_penetration_ratio"]),
    )


# =========================================== DofsInfo and DofsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DofsInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    entity_idx: qd.Tensor
    stiffness: qd.Tensor
    invweight: qd.Tensor
    armature: qd.Tensor
    damping: qd.Tensor
    frictionloss: qd.Tensor
    motion_ang: qd.Tensor
    motion_vel: qd.Tensor
    limit: qd.Tensor
    act_gain: qd.Tensor
    act_bias: qd.Tensor
    force_range: qd.Tensor
    dof_length: qd.Tensor


def get_dofs_info(solver):
    shape = (solver.n_dofs_, solver._B) if solver._options.batch_dofs_info else (solver.n_dofs_,)

    return DofsInfo(
        entity_idx=V(dtype=gs.qd_int, shape=shape),
        stiffness=V(dtype=gs.qd_float, shape=shape),
        invweight=V(dtype=gs.qd_float, shape=shape),
        armature=V(dtype=gs.qd_float, shape=shape),
        damping=V(dtype=gs.qd_float, shape=shape),
        frictionloss=V(dtype=gs.qd_float, shape=shape),
        motion_ang=V(dtype=gs.qd_vec3, shape=shape),
        motion_vel=V(dtype=gs.qd_vec3, shape=shape),
        limit=V(dtype=gs.qd_vec2, shape=shape),
        act_gain=V(dtype=gs.qd_float, shape=shape),
        act_bias=V(dtype=gs.qd_vec3, shape=shape),
        force_range=V(dtype=gs.qd_vec2, shape=shape),
        dof_length=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DofsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    # *_bw: Cache to avoid overwriting for backward pass
    force: qd.Tensor
    qf_bias: qd.Tensor
    qf_passive: qd.Tensor
    qf_applied: qd.Tensor
    pos: qd.Tensor
    vel: qd.Tensor = of_kind(DataKind.STATE)
    vel_prev: qd.Tensor = of_kind(DataKind.SCRATCH)
    vel_next: qd.Tensor = of_kind(DataKind.SCRATCH)
    acc: qd.Tensor = of_kind(DataKind.STATE)
    acc_smooth: qd.Tensor
    acc_smooth_bw: qd.Tensor
    qf_smooth: qd.Tensor
    qf_constraint: qd.Tensor
    # Implicit damping correction (see func_implicit_damping in forward_dynamics.py): qf_damping_implicit holds the
    # implicit share of the damping force, h D qacc for the diagonal D the damped mass factor adds, and
    # qacc_damping_implicit the correction (M + hD)^-1 h D qacc subtracted from the acceleration.
    qf_damping_implicit: qd.Tensor = of_kind(DataKind.SCRATCH)
    qacc_damping_implicit: qd.Tensor = of_kind(DataKind.SCRATCH)
    cdof_ang: qd.Tensor
    cdof_vel: qd.Tensor
    cdofd_ang: qd.Tensor
    cdofd_vel: qd.Tensor
    f_vel: qd.Tensor
    f_ang: qd.Tensor
    ctrl_force: qd.Tensor = of_kind(DataKind.STATE)
    ctrl_pos: qd.Tensor = of_kind(DataKind.STATE)
    ctrl_vel: qd.Tensor = of_kind(DataKind.STATE)
    ctrl_mode: qd.Tensor = of_kind(DataKind.STATE)
    is_hibernated: qd.Tensor = of_kind(DataKind.STATE)


def get_dofs_state(solver):
    shape = (solver.n_dofs_, solver._B)
    requires_grad = solver._requires_grad
    shape_bw = maybe_shape((2, *shape), requires_grad)

    return DofsState(
        force=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_bias=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_passive=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_applied=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        pos=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel_prev=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel_next=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc_smooth=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc_smooth_bw=V(dtype=gs.qd_float, shape=shape_bw, needs_grad=requires_grad),
        qf_smooth=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_constraint=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_damping_implicit=V(dtype=gs.qd_float, shape=shape),
        qacc_damping_implicit=V(dtype=gs.qd_float, shape=shape),
        cdof_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdof_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        f_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        f_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        ctrl_force=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_pos=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_vel=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_mode=V(dtype=gs.qd_int, shape=shape),
        is_hibernated=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== LinksState and LinksInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class LinksState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    # *_bw: Cache to avoid overwriting for backward pass
    cinr_inertial: qd.Tensor
    cinr_pos: qd.Tensor
    cinr_quat: qd.Tensor
    cinr_mass: qd.Tensor
    crb_inertial: qd.Tensor
    crb_pos: qd.Tensor
    crb_quat: qd.Tensor
    crb_mass: qd.Tensor
    cdd_vel: qd.Tensor
    cdd_ang: qd.Tensor
    # The pose of a fixed root link is state, which forward kinematics leaves as set (see
    # func_forward_kinematics_link_bw), the poses of the other links being recomputed from the positions of the dofs.
    pos: qd.Tensor = of_kind(DataKind.STATE)
    quat: qd.Tensor = of_kind(DataKind.STATE)
    pos_bw: qd.Tensor
    quat_bw: qd.Tensor
    i_pos: qd.Tensor
    i_pos_bw: qd.Tensor
    i_quat: qd.Tensor
    cd_ang: qd.Tensor
    cd_vel: qd.Tensor
    # Whether any contact or connect/weld equality row involves the link this step. Written by the constraint
    # assembly (the deterministic post-sort view of the contacts), consumed by the midpoint-integration eligibility;
    # the raw collider contact buffer is only valid through the contact_sort_idx indirection.
    is_constrained: qd.Tensor
    # Midpoint angular velocity a free body eligible for midpoint integration converged to this substep, in its inertial
    # frame, and the residual of the midpoint equation there. The integration the backward pass reverses resumes from
    # them, since autodiff cannot reverse the Newton iteration (see func_midpoint_free_body). Allocated for the dynamic
    # solvers integrating with the midpoint rule only.
    midpoint_ang: qd.Tensor = of_kind(DataKind.SCRATCH)
    midpoint_res: qd.Tensor = of_kind(DataKind.SCRATCH)
    cd_ang_bw: qd.Tensor
    cd_vel_bw: qd.Tensor
    mass_sum: qd.Tensor
    root_COM: qd.Tensor  # COM of the kinematic tree
    root_COM_bw: qd.Tensor
    cacc_ang: qd.Tensor
    cacc_lin: qd.Tensor
    cfrc_ang: qd.Tensor
    cfrc_vel: qd.Tensor
    cfrc_applied_ang: qd.Tensor = of_kind(DataKind.STATE)
    cfrc_applied_vel: qd.Tensor = of_kind(DataKind.STATE)
    # The wrench the couplers accumulate during a substep, which the next substep consumes
    cfrc_coupling_ang: qd.Tensor = of_kind(DataKind.STATE)
    cfrc_coupling_vel: qd.Tensor = of_kind(DataKind.STATE)
    contact_force: qd.Tensor
    # Hibernation: is_hibernated flags a sleeping link, awake_steps counts the consecutive substeps an awake link has
    # spent below the hibernation speed tolerance, up to hibernation_min_steps (see func_count_settled_step), zero
    # again the step it exceeds the tolerance or wakes.
    is_hibernated: qd.Tensor = of_kind(DataKind.STATE)
    awake_steps: qd.Tensor = of_kind(DataKind.STATE)


def get_links_state(solver, kinematic_only):
    shape = (solver.n_links_, solver._B)
    requires_grad = solver._requires_grad
    is_midpoint = not kinematic_only and solver.rigid_config.integrator != gs.integrator.Euler
    # The backward joint buffers hold one slot per joint of a link plus the link itself; collapsed when grad is off.
    n_joints_bw = (max(link.n_joints for link in solver.links) + 1) if requires_grad and solver.n_links else 1
    shape_bw = (solver.n_links_, n_joints_bw, solver._B)

    return LinksState(
        cinr_inertial=V(dtype=gs.qd_mat3, shape=shape, needs_grad=requires_grad),
        cinr_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cinr_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        cinr_mass=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        crb_inertial=V(dtype=gs.qd_mat3, shape=shape, needs_grad=requires_grad),
        crb_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        crb_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        crb_mass=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        cdd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        pos_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        quat_bw=V(dtype=gs.qd_vec4, shape=shape_bw, needs_grad=requires_grad),
        i_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        i_pos_bw=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        i_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        cd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        is_constrained=V(dtype=gs.qd_bool, shape=shape),
        midpoint_ang=V(dtype=gs.qd_vec3, shape=maybe_shape(shape, is_midpoint)),
        midpoint_res=V(dtype=gs.qd_vec3, shape=maybe_shape(shape, is_midpoint)),
        cd_ang_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        cd_vel_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        mass_sum=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        root_COM=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        root_COM_bw=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cacc_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cacc_lin=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_applied_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_applied_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_coupling_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_coupling_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        contact_force=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        is_hibernated=V(dtype=gs.qd_int, shape=shape),
        awake_steps=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class LinksInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    parent_idx: qd.Tensor
    root_idx: qd.Tensor
    q_start: qd.Tensor
    dof_start: qd.Tensor
    joint_start: qd.Tensor
    q_end: qd.Tensor
    dof_end: qd.Tensor
    joint_end: qd.Tensor
    n_dofs: qd.Tensor
    pos: qd.Tensor
    quat: qd.Tensor
    invweight: qd.Tensor
    is_fixed: qd.Tensor
    inertial_pos: qd.Tensor
    inertial_quat: qd.Tensor
    inertial_i: qd.Tensor
    inertial_mass: qd.Tensor
    entity_idx: qd.Tensor
    # Heterogeneous simulation support: per-link geom/vgeom index ranges
    geom_start: qd.Tensor
    geom_end: qd.Tensor
    vgeom_start: qd.Tensor
    vgeom_end: qd.Tensor


def get_links_info(solver):
    links_info_shape = (solver.n_links_, solver._B) if solver._options.batch_links_info else solver.n_links_

    return LinksInfo(
        parent_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        root_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        q_start=V(dtype=gs.qd_int, shape=links_info_shape),
        dof_start=V(dtype=gs.qd_int, shape=links_info_shape),
        joint_start=V(dtype=gs.qd_int, shape=links_info_shape),
        q_end=V(dtype=gs.qd_int, shape=links_info_shape),
        dof_end=V(dtype=gs.qd_int, shape=links_info_shape),
        joint_end=V(dtype=gs.qd_int, shape=links_info_shape),
        n_dofs=V(dtype=gs.qd_int, shape=links_info_shape),
        pos=V(dtype=gs.qd_vec3, shape=links_info_shape),
        quat=V(dtype=gs.qd_vec4, shape=links_info_shape),
        invweight=V(dtype=gs.qd_vec2, shape=links_info_shape),
        is_fixed=V(dtype=gs.qd_bool, shape=links_info_shape),
        inertial_pos=V(dtype=gs.qd_vec3, shape=links_info_shape),
        inertial_quat=V(dtype=gs.qd_vec4, shape=links_info_shape),
        inertial_i=V(dtype=gs.qd_mat3, shape=links_info_shape),
        inertial_mass=V(dtype=gs.qd_float, shape=links_info_shape),
        entity_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        # Heterogeneous simulation support: per-link geom/vgeom index ranges
        geom_start=V(dtype=gs.qd_int, shape=links_info_shape),
        geom_end=V(dtype=gs.qd_int, shape=links_info_shape),
        vgeom_start=V(dtype=gs.qd_int, shape=links_info_shape),
        vgeom_end=V(dtype=gs.qd_int, shape=links_info_shape),
    )


# =========================================== JointsInfo and JointsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class JointsInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    type: qd.Tensor
    sol_params: qd.Tensor
    q_start: qd.Tensor
    dof_start: qd.Tensor
    q_end: qd.Tensor
    dof_end: qd.Tensor
    n_dofs: qd.Tensor
    pos: qd.Tensor


def get_joints_info(solver):
    shape = (solver.n_joints_, solver._B) if solver._options.batch_joints_info else (solver.n_joints_,)

    return JointsInfo(
        type=V(dtype=gs.qd_int, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
        q_start=V(dtype=gs.qd_int, shape=shape),
        dof_start=V(dtype=gs.qd_int, shape=shape),
        q_end=V(dtype=gs.qd_int, shape=shape),
        dof_end=V(dtype=gs.qd_int, shape=shape),
        n_dofs=V(dtype=gs.qd_int, shape=shape),
        pos=V(dtype=gs.qd_vec3, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class JointsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    xanchor: qd.Tensor
    xaxis: qd.Tensor


def get_joints_state(solver):
    shape = (solver.n_joints_, solver._B)
    requires_grad = solver._requires_grad

    return JointsState(
        xanchor=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        xaxis=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
    )


# =========================================== GeomsInfo and GeomsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GeomsInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    pos: qd.Tensor
    center: qd.Tensor
    quat: qd.Tensor
    data: qd.Tensor
    link_idx: qd.Tensor
    type: qd.Tensor
    friction: qd.Tensor
    friction_torsional: qd.Tensor
    friction_rolling: qd.Tensor
    sol_params: qd.Tensor
    vert_num: qd.Tensor
    vert_start: qd.Tensor
    vert_end: qd.Tensor
    verts_state_start: qd.Tensor
    verts_state_end: qd.Tensor
    face_num: qd.Tensor
    face_start: qd.Tensor
    face_end: qd.Tensor
    edge_num: qd.Tensor
    edge_start: qd.Tensor
    edge_end: qd.Tensor
    is_convex: qd.Tensor
    contype: qd.Tensor
    conaffinity: qd.Tensor
    is_fixed: qd.Tensor
    is_decomposed: qd.Tensor
    is_hollow: qd.Tensor
    needs_coup: qd.Tensor
    coup_friction: qd.Tensor
    coup_softness: qd.Tensor
    coup_restitution: qd.Tensor


def get_geoms_info(solver, is_active=True):
    shape = (solver.n_geoms_,) if is_active else ()

    return GeomsInfo(
        pos=V(dtype=gs.qd_vec3, shape=shape),
        center=V(dtype=gs.qd_vec3, shape=shape),
        quat=V(dtype=gs.qd_vec4, shape=shape),
        data=V(dtype=gs.qd_vec7, shape=shape),
        link_idx=V(dtype=gs.qd_int, shape=shape),
        type=V(dtype=gs.qd_int, shape=shape),
        friction=V(dtype=gs.qd_float, shape=shape),
        friction_torsional=V(dtype=gs.qd_float, shape=shape),
        friction_rolling=V(dtype=gs.qd_float, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
        vert_num=V(dtype=gs.qd_int, shape=shape),
        vert_start=V(dtype=gs.qd_int, shape=shape),
        vert_end=V(dtype=gs.qd_int, shape=shape),
        verts_state_start=V(dtype=gs.qd_int, shape=shape),
        verts_state_end=V(dtype=gs.qd_int, shape=shape),
        face_num=V(dtype=gs.qd_int, shape=shape),
        face_start=V(dtype=gs.qd_int, shape=shape),
        face_end=V(dtype=gs.qd_int, shape=shape),
        edge_num=V(dtype=gs.qd_int, shape=shape),
        edge_start=V(dtype=gs.qd_int, shape=shape),
        edge_end=V(dtype=gs.qd_int, shape=shape),
        is_convex=V(dtype=gs.qd_bool, shape=shape),
        contype=V(dtype=gs.qd_int, shape=shape),
        conaffinity=V(dtype=gs.qd_int, shape=shape),
        is_fixed=V(dtype=gs.qd_bool, shape=shape),
        is_decomposed=V(dtype=gs.qd_bool, shape=shape),
        is_hollow=V(dtype=gs.qd_bool, shape=shape),
        needs_coup=V(dtype=gs.qd_int, shape=shape),
        coup_friction=V(dtype=gs.qd_float, shape=shape),
        coup_softness=V(dtype=gs.qd_float, shape=shape),
        coup_restitution=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GeomsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    quat: qd.Tensor
    aabb_min: qd.Tensor
    aabb_max: qd.Tensor
    verts_updated: qd.Tensor
    is_hibernated: qd.Tensor = of_kind(DataKind.STATE)
    friction_ratio: qd.Tensor = of_kind(DataKind.INFO)


def get_geoms_state(solver, is_active=True):
    shape = (solver.n_geoms_, solver._B) if is_active else ()
    requires_grad = solver.rigid_config.requires_grad

    return GeomsState(
        pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        aabb_min=V(dtype=gs.qd_vec3, shape=shape),
        aabb_max=V(dtype=gs.qd_vec3, shape=shape),
        verts_updated=V(dtype=gs.qd_bool, shape=shape),
        is_hibernated=V(dtype=gs.qd_int, shape=shape),
        friction_ratio=V(dtype=gs.qd_float, shape=shape),
    )


# =========================================== VertsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VertsInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    init_pos: qd.Tensor
    init_normal: qd.Tensor
    geom_idx: qd.Tensor
    init_center_pos: qd.Tensor
    verts_state_idx: qd.Tensor
    is_fixed: qd.Tensor


def get_verts_info(solver, is_active=True):
    shape = (solver.n_verts_,) if is_active else ()

    return VertsInfo(
        init_pos=V(dtype=gs.qd_vec3, shape=shape),
        init_normal=V(dtype=gs.qd_vec3, shape=shape),
        geom_idx=V(dtype=gs.qd_int, shape=shape),
        init_center_pos=V(dtype=gs.qd_vec3, shape=shape),
        verts_state_idx=V(dtype=gs.qd_int, shape=shape),
        is_fixed=V(dtype=gs.qd_bool, shape=shape),
    )


# =========================================== FacesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FacesInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    verts_idx: qd.Tensor
    geom_idx: qd.Tensor


def get_faces_info(solver, is_active=True):
    shape = (solver.n_faces_,) if is_active else ()

    return FacesInfo(verts_idx=V(dtype=gs.qd_ivec3, shape=shape), geom_idx=V(dtype=gs.qd_int, shape=shape))


# =========================================== EdgesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EdgesInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    v0: qd.Tensor
    v1: qd.Tensor
    length: qd.Tensor


def get_edges_info(solver, is_active=True):
    shape = (solver.n_edges_,) if is_active else ()

    return EdgesInfo(
        v0=V(dtype=gs.qd_int, shape=shape), v1=V(dtype=gs.qd_int, shape=shape), length=V(dtype=gs.qd_float, shape=shape)
    )


# =========================================== VertsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VertsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor


def get_free_verts_state(solver, is_active=True):
    return VertsState(pos=V(dtype=gs.qd_vec3, shape=(solver.n_free_verts_, solver._B) if is_active else ()))


def get_fixed_verts_state(solver, is_active=True):
    return VertsState(pos=V(dtype=gs.qd_vec3, shape=(solver.n_fixed_verts_,) if is_active else ()))


# =========================================== VvertsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VVertsInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    init_pos: qd.Tensor
    init_vnormal: qd.Tensor
    vgeom_idx: qd.Tensor
    vverts_state_idx: qd.Tensor


def get_vverts_info(solver):
    shape = (solver.n_vverts_,)

    return VVertsInfo(
        init_pos=V(dtype=gs.qd_vec3, shape=shape),
        init_vnormal=V(dtype=gs.qd_vec3, shape=shape),
        vgeom_idx=V(dtype=gs.qd_int, shape=shape),
        vverts_state_idx=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== VVertsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VVertsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor


def get_vverts_state(solver, is_active=True):
    if not is_active:
        return VVertsState(pos=V(dtype=gs.qd_vec3, shape=()))
    if not is_tensor_addressable((solver.n_custom_vverts_, solver._B, 3), gs.np_float):
        gs.raise_exception(
            f"Custom-vverts state shape (n_custom_vverts={solver.n_custom_vverts_}, B={solver._B}, 3) is too large. "
            "Consider opting fewer kinematic entities into 'enable_custom_vverts=True', or reducing 'n_envs'."
        )
    return VVertsState(pos=V(dtype=gs.qd_vec3, shape=(solver.n_custom_vverts_, solver._B)))


# =========================================== VfacesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VFacesInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    vverts_idx: qd.Tensor
    vgeom_idx: qd.Tensor


def get_vfaces_info(solver):
    shape = (solver.n_vfaces_,)

    return VFacesInfo(vverts_idx=V(dtype=gs.qd_ivec3, shape=shape), vgeom_idx=V(dtype=gs.qd_int, shape=shape))


# =========================================== VgeomsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VGeomsInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    pos: qd.Tensor
    quat: qd.Tensor
    link_idx: qd.Tensor
    vvert_num: qd.Tensor
    vvert_start: qd.Tensor
    vvert_end: qd.Tensor
    vface_num: qd.Tensor
    vface_start: qd.Tensor
    vface_end: qd.Tensor
    color: qd.Tensor


def get_vgeoms_info(solver):
    shape = (solver.n_vgeoms_,)

    return VGeomsInfo(
        pos=V(dtype=gs.qd_vec3, shape=shape),
        quat=V(dtype=gs.qd_vec4, shape=shape),
        link_idx=V(dtype=gs.qd_int, shape=shape),
        vvert_num=V(dtype=gs.qd_int, shape=shape),
        vvert_start=V(dtype=gs.qd_int, shape=shape),
        vvert_end=V(dtype=gs.qd_int, shape=shape),
        vface_num=V(dtype=gs.qd_int, shape=shape),
        vface_start=V(dtype=gs.qd_int, shape=shape),
        vface_end=V(dtype=gs.qd_int, shape=shape),
        color=V(dtype=gs.qd_vec4, shape=shape),
    )


# =========================================== VGeomsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VGeomsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    quat: qd.Tensor


def get_vgeoms_state(solver, is_active=True):
    shape = (solver.n_vgeoms_, solver._B) if is_active else ()

    return VGeomsState(pos=V(dtype=gs.qd_vec3, shape=shape), quat=V(dtype=gs.qd_vec4, shape=shape))


# =========================================== EqualitiesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EqualitiesInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    eq_obj1id: qd.Tensor
    eq_obj2id: qd.Tensor
    # The data of a screw constraint, the largest one, sizes the entries of all equalities (see func_screw_frame in
    # constraint/solver.py)
    eq_data: qd.Tensor
    eq_type: qd.Tensor
    sol_params: qd.Tensor


def get_equalities_info(solver, is_active=True):
    shape = (solver.n_candidate_equalities_, solver._B) if is_active else ()

    return EqualitiesInfo(
        eq_obj1id=V(dtype=gs.qd_int, shape=shape),
        eq_obj2id=V(dtype=gs.qd_int, shape=shape),
        eq_data=V(dtype=gs.qd_vec18, shape=shape),
        eq_type=V(dtype=gs.qd_int, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
    )


# =========================================== EntitiesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EntitiesInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    dof_start: qd.Tensor
    dof_end: qd.Tensor
    n_dofs: qd.Tensor
    link_start: qd.Tensor
    link_end: qd.Tensor
    n_links: qd.Tensor
    geom_start: qd.Tensor
    geom_end: qd.Tensor
    n_geoms: qd.Tensor
    gravity_compensation: qd.Tensor
    is_local_collision_mask: qd.Tensor


def get_entities_info(solver):
    shape = (solver.n_entities_,)

    return EntitiesInfo(
        dof_start=V(dtype=gs.qd_int, shape=shape),
        dof_end=V(dtype=gs.qd_int, shape=shape),
        n_dofs=V(dtype=gs.qd_int, shape=shape),
        link_start=V(dtype=gs.qd_int, shape=shape),
        link_end=V(dtype=gs.qd_int, shape=shape),
        n_links=V(dtype=gs.qd_int, shape=shape),
        geom_start=V(dtype=gs.qd_int, shape=shape),
        geom_end=V(dtype=gs.qd_int, shape=shape),
        n_geoms=V(dtype=gs.qd_int, shape=shape),
        gravity_compensation=V(dtype=gs.qd_float, shape=shape),
        is_local_collision_mask=V(dtype=gs.qd_bool, shape=shape),
    )


# =========================================== EntitiesState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EntitiesState:
    kind: ClassVar[DataKind] = DataKind.STATE

    is_hibernated: qd.Tensor


def get_entities_state(solver, is_active=True):
    return EntitiesState(is_hibernated=V(dtype=gs.qd_int, shape=(solver.n_entities_, solver._B) if is_active else ()))


# =========================================== DynInfo and DynState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DynInfo:
    """Build-time model description of the kinematics+dynamics component, one leaf struct per element type.

    Kinematic-only solvers allocate the collision-related leaves (geoms, verts, faces, edges, equalities) as scalar
    dummies: the fields must exist for kernel argument flattening, while unused ones are pruned per compiled kernel.
    """

    entities: EntitiesInfo
    links: LinksInfo
    joints: JointsInfo
    dofs: DofsInfo
    geoms: GeomsInfo
    verts: VertsInfo
    faces: FacesInfo
    edges: EdgesInfo
    vverts: VVertsInfo
    vfaces: VFacesInfo
    vgeoms: VGeomsInfo
    equalities: EqualitiesInfo


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DynState:
    """Runtime state of the kinematics+dynamics component, one leaf struct per element type.

    Dummy-leaf allocation follows DynInfo. The adjoint cache is a second instance holding copies of the leaves
    overwritten during a forward substep (dofs, links, joints, geoms), with every other field a scalar dummy.
    """

    entities: EntitiesState
    links: LinksState
    joints: JointsState
    dofs: DofsState
    geoms: GeomsState
    free_verts: VertsState
    fixed_verts: VertsState
    vverts: VVertsState
    vgeoms: VGeomsState


# =================================== RigidSimStaticConfig ===================================


@qd.data_oriented
class RigidSimStaticConfig(StaticConfig):
    backend: int
    para_level: int
    enable_collision: bool
    use_hibernation: bool
    batch_links_info: bool
    batch_dofs_info: bool
    batch_joints_info: bool
    enable_mujoco_compatibility: bool
    enable_elliptic_friction: bool
    # Whether contact friction is bounded by the disc of radius mu * f_n taken from the normal force developed at the
    # current iterate, instead of jointly with it through the coupled cone (see gs.contact_resolution). Implies
    # enable_elliptic_friction: the pyramidal rows do not separate into a normal row and a friction disc.
    enable_signorini_contact: bool
    enable_multi_contact: bool
    enable_joint_limit: bool
    box_box_detection: bool
    sparse_solve: bool
    integrator: int
    solver_type: int
    requires_grad: bool
    prefer_decomposed_solver: int = -1  # -1 = None (auto), 0 = False, 1 = True
    # Whether the cone-free assembled Hessian is persisted in nt_H's mirror slots (diagonal in nt_H_cone_free_diag);
    # see the nt_H declaration for the packed-storage mechanics and the rigid solver's resolution for the gating.
    enable_cone_free_hessian_reuse: bool = False
    # Whether contacts carry torsional friction rows resisting relative spin about the contact normal: one extra
    # opposing pyramid pair per contact with the pyramidal cone, one extra cone row with the elliptic cone.
    enable_torsional_friction: bool = False
    # Whether contacts also carry rolling friction rows resisting relative rotation about the two tangent axes: two
    # extra opposing pyramid pairs per contact with the pyramidal cone, two extra cone rows with the elliptic cone.
    # Requires enable_torsional_friction (the rolling rows sit after the spin row in the contact row layout).
    enable_rolling_friction: bool = False
    # Consecutive sub-tolerance steps a body's max DOF velocity must hold before it is ready to hibernate. Guards
    # against a body that is only momentarily slow (e.g. at the apex of a toss) sleeping prematurely.
    hibernation_min_steps: int = 10
    parallel_init: bool = False  # parallelize init over (constraints, envs) when GPU is not saturated by envs alone
    # Whether the walks of the kinematic trees and roots sweep them level by level (see func_sweep_links_by_level)
    enable_level_sweep: bool = False
    # Lanes of the level sweep per env, a power of two dividing its blocks of 32 lanes. Four envs per block balance the
    # lanes a narrow level leaves idle against the links a wide level serializes.
    level_sweep_n_lanes_per_env: int = 8
    broadphase_traversal: int = 0
    enable_tiled_cholesky_mass_matrix: bool = False
    mass_matrix_fits_shared: bool = False
    # Register-tile width of the mass-matrix Cholesky kernels: 16 (Tile16x16) or 32 (Tile32x32), see
    # cholesky_tile_size_for.
    cholesky_tile_size: int = 32
    # Register-streaming tiled per-entity mass factor for the >shared-cap branch of func_factor_mass (GPU forward
    # only). When True, each entity's single-mass-block submatrix factors in registers via the same TileNxN Cholesky
    # primitive as the Hessian, instead of the shared-pivot cooperative LDL^T. Only enabled when every entity is a
    # single mass block (the common case: one kinematic tree). The tile width is always 32: the path is only taken
    # when the per-entity block exceeds shared memory, which on any real GPU means well over 48 DOFs.
    enable_register_tiled_mass: bool = False
    # When True, the shared-memory mass factor also solves the smooth acceleration from the forces against the factor it
    # holds, which runs it after the force passes, and func_compute_qacc skips its own solve. See the rigid solver's
    # resolution for the gating.
    enable_fused_smooth_acc_solve: bool = False
    # When True, func_solve_init seeds every island's factor with the tiled per-island kernels at any env count. The
    # solve is seeded one env per thread otherwise (see has_per_env_solve_seed). See the rigid solver's resolution for
    # the gating.
    enable_tiled_island_seed: bool = False
    # When True, the scene holds one dof-carrying kinematic tree, so an env forms at most one island and the
    # per-island passes of the solve read the env's plain dof and row ranges. See the rigid solver's resolution.
    is_single_island: bool = False
    # When True, the seed kernel assembles the env's one Hessian block and the monolith factors it with the scalar
    # dense Cholesky (see _kernel_solve_monolith), a single-island scene above the cooperative bound.
    has_scalar_seed_factor: bool = False
    # When True, the constraint solver uses the GPU subgroup-cooperative kernel variants (warp-cooperative linesearch
    # refinement, per-friction constraint builder, cooperative mass-matrix assembly), together with the batch-first
    # tensor layouts they expect, eg (_B, len_constraints_) for Jaref / efc_D / ... which unlocks coalesced cross-lane
    # reads.
    enable_cooperative_constraint_kernels: bool = False
    # When True, the noslip sweep of an island runs on a block of 32 lanes: the island's rows are colored so that the
    # rows of a color touch disjoint mass blocks, the lanes update the rows of a color in parallel and the colors are
    # swept in order (see func_noslip in noslip.py). The rows are visited in another order than by the one-thread
    # sweep, so the two sweeps give different iterates. See the rigid solver's resolution for the gating.
    enable_cooperative_noslip: bool = False
    # Purely descriptive layout flag: True whenever the layout-flippable constraint-state tensors are physically
    # batch-first, i.e. enable_cooperative_constraint_kernels or serialized execution (env loop outermost, so per-env
    # rows must be contiguous). Consumers that only need iteration order to follow the physical layout (ndrange axes,
    # flattened index decompositions) key on this flag, while algorithm selection (warp-cooperative vs serial
    # reductions) keys on enable_cooperative_constraint_kernels alone.
    constraint_layout_batch_first: bool = False
    # Whether the Newton system is Jacobi-equilibrated: see nt_jacobi in ConstraintState for the mechanics, the rigid
    # solver's resolution for the gating.
    enable_jacobi_equilibration: bool = False
    tiled_n_dofs_per_block: int = -1
    # Dof caps of the shared tiles of the cooperative per-island factor+solve, one launch per cap (see island_tile_caps
    # below): the doublings of the first cap, the tile size doubled up to the smallest dof-carrying tree, below
    # the last one, the largest tile-size multiple that fits GPU shared memory. Islands above the last cap factor in
    # global memory.
    island_tile_cap_first: int = 0
    island_tile_cap_last: int = 0
    # Whether an island can hold more dofs than the last cap, which compiles the factor paths above it.
    has_island_above_tile_cap: bool = False

    @property
    def rows_per_contact(self) -> int:
        """Constraint rows per contact.

        The pyramidal cone carries 2 opposing friction-mixed edges per friction axis, the elliptic cone the normal
        row plus one row per friction axis; torsional friction adds the spin axis and rolling friction the two
        tangent axes.
        """
        n_extra_axes = int(self.enable_torsional_friction) + 2 * int(self.enable_rolling_friction)
        if self.enable_elliptic_friction:
            return 3 + n_extra_axes
        return 4 + 2 * n_extra_axes

    @property
    def hessian_rank_update_batch(self) -> int:
        """Number of rank-1 Cholesky updates fused into one column sweep by the per-island incremental factor.

        Sizes the nt_vec slots and the static per-column unroll of func_rank_batch_update_island: 8 amortizes the
        active-set flip batching, widened when the coupled elliptic-cone update must stage 2 slots per cone row
        (see func_cone_rank_update_island).
        """
        return max(8, 2 * self.rows_per_contact)

    @property
    def has_per_env_solve_seed(self) -> bool:
        """Whether func_solve_init seeds the Newton factor, the gradient and the convergence certificate one env per
        thread, for both solve arms.

        This holds on GPU wherever the tiled factor seed is off or leaves the factor to the scalar dense Cholesky (see
        has_scalar_seed_factor). The decomposed arm then factors the Hessian of every env one env per thread at each
        iteration too, since its tiled per-iteration factor maintains the assembled Hessian that only the tiled seed
        provides.
        """
        return (
            self.solver_type == gs.constraint_solver.Newton
            and self.backend != gs.cpu
            and (not self.enable_tiled_island_seed or self.has_scalar_seed_factor)
        )


# =========================================== DataManager ===========================================


def cholesky_tile_size_for(n_dofs):
    """Tile size of the register-streaming Cholesky kernels for a system of n_dofs dofs: 16 where one 16-lane tile
    holds it or where a 32-lane tile would pad it past its lane utilization (33 to 48 dofs), 32 otherwise."""
    return 16 if (n_dofs <= 16 or 32 < n_dofs <= 48) else 32


def cholesky_packed_tile_shape(max_dofs):
    """Shape of the shared tile of the tiled factor of a block of up to max_dofs dofs.

    The tile holds the blocks of the lower triangle alone, one below the other, each a square of the tile size (see
    cholesky_tile_size_for) padded by one column against shared-memory bank conflicts, a block of a single tile being
    the tile itself (see func_packed_tile_pos in solver.py). The sizes are native integers, as the shared array takes
    them.
    """
    tile_size = cholesky_tile_size_for(max_dofs)
    n_blocks = (max_dofs + tile_size - 1) // tile_size
    n_rows = min(tile_size, max_dofs)
    return int(n_blocks * (n_blocks + 1) // 2 * n_rows), int(n_rows + 1)


def island_tile_caps(cap_first, cap_last):
    """Ascending dof caps of the shared tiles of the cooperative per-island factor+solve, see island_tile_cap_first:
    the doublings of the first cap below the last one, then the last one."""
    caps = []
    cap = cap_first
    while cap < cap_last:
        caps.append(cap)
        cap *= 2
    caps.append(cap_last)
    return tuple(caps)


@qd.data_oriented
class DataManager:
    def __init__(self, solver, kinematic_only):
        is_dynamic = not kinematic_only
        mass_mat_shape = (solver.n_dofs_, solver.n_dofs_, solver._B)
        if not is_tensor_addressable(mass_mat_shape, gs.np_float):
            gs.raise_exception(
                f"Mass matrix shape (n_dofs={solver.n_dofs_}, n_dofs={solver.n_dofs_}, n_envs={solver._B}) is too large."
            )
        config, options = solver.rigid_config, solver._options
        shape = dict(
            envs=solver._B,
            qs=solver.n_qs_,
            dofs=solver.n_dofs_,
            links=solver.n_links_,
            roots=solver.n_roots_,
            trees=solver.n_trees_,
        )
        values = dict(EPS=gs.EPS)
        if is_dynamic:
            shape.update(
                geoms=solver.n_geoms_,
                trees_levels_links=np.count_nonzero(solver._links_tree_root_idx >= 0),
                child_links=np.count_nonzero(solver._links_parent_idx >= 0),
                links_bounds=solver.n_links_ + 1,
            )
            values.update(
                substep_dt=solver._substep_dt,
                iterations=options.iterations,
                tolerance=options.tolerance,
                ls_iterations=options.ls_iterations,
                ls_tolerance=options.ls_tolerance,
                noslip_iterations=options.noslip_iterations,
                noslip_tolerance=options.noslip_tolerance,
                impratio=options.impratio,
                n_equalities=solver._n_equalities,
                n_candidate_equalities=solver.n_candidate_equalities_,
                hibernation_thresh_vel=solver._hibernation_thresh_vel,
            )
        self.rigid_info = RigidInfo(
            shape=shape,
            values=values,
            layout={
                RigidInfo.mass_mat: (2, 1, 0) if is_dynamic and config.enable_cooperative_constraint_kernels else None
            },
            unused={
                DYNAMICS: kinematic_only,
                LEVEL_SWEEP: not (is_dynamic and config.enable_level_sweep),
                TILED_MASS: not (is_dynamic and config.enable_register_tiled_mass),
            },
            requires_grad=is_dynamic and solver._requires_grad,
        )

        # Leaf structs are allocated one by one before assembling the per-component aggregates, to control the
        # allocation order of the underlying buffers (see the warning in get_constraint_state).
        is_dynamic = not kinematic_only
        dofs_info = get_dofs_info(solver)
        dofs_state = get_dofs_state(solver)
        links_info = get_links_info(solver)
        links_state = get_links_state(solver, kinematic_only)
        joints_info = get_joints_info(solver)
        joints_state = get_joints_state(solver)

        entities_info = get_entities_info(solver)
        entities_state = get_entities_state(solver)

        vverts_info = get_vverts_info(solver)
        vverts_state = get_vverts_state(solver)
        vfaces_info = get_vfaces_info(solver)

        vgeoms_info = get_vgeoms_info(solver)
        vgeoms_state = get_vgeoms_state(solver)

        geoms_info = get_geoms_info(solver, is_dynamic)
        geoms_state = get_geoms_state(solver, is_dynamic)

        verts_info = get_verts_info(solver, is_dynamic)
        faces_info = get_faces_info(solver, is_dynamic)
        edges_info = get_edges_info(solver, is_dynamic)

        free_verts_state = get_free_verts_state(solver, is_dynamic)
        fixed_verts_state = get_fixed_verts_state(solver, is_dynamic)

        equalities_info = get_equalities_info(solver, is_dynamic)

        entity_dofs = max((entity.n_dofs for entity in solver.entities), default=0)
        self.kinematics_scratch = KinematicsScratch(
            shape=dict(envs=solver._B, qs=solver.n_qs, entity_dofs=entity_dofs), unused={KINEMATICS: entity_dofs == 0}
        )

        self.dyn_info = DynInfo(
            entities=entities_info,
            links=links_info,
            joints=joints_info,
            dofs=dofs_info,
            geoms=geoms_info,
            verts=verts_info,
            faces=faces_info,
            edges=edges_info,
            vverts=vverts_info,
            vfaces=vfaces_info,
            vgeoms=vgeoms_info,
            equalities=equalities_info,
        )
        self.dyn_state = DynState(
            entities=entities_state,
            links=links_state,
            joints=joints_state,
            dofs=dofs_state,
            geoms=geoms_state,
            free_verts=free_verts_state,
            fixed_verts=fixed_verts_state,
            vverts=vverts_state,
            vgeoms=vgeoms_state,
        )

        self.errno = V(dtype=gs.qd_int, shape=(solver._B,))

        self._solver = solver
        self._ik_scratch = None
        self._weight_scratch = None

    @property
    def weight_scratch(self) -> WeightScratch:
        """Scratch buffers for the inverse-weight computation, allocated the first time a setter triggers one.

        At build time the computation borrows the buffers of the constraint solver, so a scene that never writes a
        mass, a center of mass or an inertia allocates none of this.
        """
        if self._weight_scratch is None:
            self._weight_scratch = WeightScratch(shape=dict(envs=self._solver._B, dofs=self._solver.n_dofs_))
        return self._weight_scratch

    @property
    def ik_scratch(self) -> IKScratch:
        """Inverse-kinematics scratch, allocated the first time a solve asks for it.

        The damped normal matrix it holds is quadratic in RigidOptions.IK_max_targets, so a scene that never solves
        inverse kinematics carries none of it.
        """
        if self._ik_scratch is None:
            entities = self._solver.entities
            n_targets = self._solver._options.IK_max_targets
            self._ik_scratch = IKScratch(
                shape=dict(
                    envs=self._solver._B,
                    ik_qs=max([entity.n_qs for entity in entities]),
                    ik_dofs=max([entity.n_dofs for entity in entities]),
                    ik_links=max([entity.n_links for entity in entities]),
                    ik_joints=max([entity.n_joints for entity in entities]),
                    ik_targets=n_targets,
                    ik_errors=6 * n_targets,
                )
            )
        return self._ik_scratch


# =========================================== RaycastResult ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class RaycastResult:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    distance: qd.Tensor
    geom_idx: qd.Tensor
    hit_point: qd.Tensor
    normal: qd.Tensor


def get_raycast_result(n_envs: int):
    return RaycastResult(
        distance=V(dtype=gs.qd_float, shape=(n_envs,)),
        geom_idx=V(dtype=gs.qd_int, shape=(n_envs,)),
        hit_point=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
    )


GeomsInitAABB = qd.Tensor


# =========================================== BVH ===========================================


class BVH_SORT_KIND(IntEnum):
    """The sort ordering the leaf keys of a BVH set.

    See genesis.engine.bvh.
    """

    # The device-wide radix sort of quadrants over every key of the set (GPU, large trees)
    DEVICE_RADIX = 0
    # The rank of each leaf among the keys of its tree (small trees)
    RANK = 1
    # A radix sort per tree, its chunks in parallel (CPU, large trees)
    PER_TREE_RADIX = 2


class BVH_FIT_KIND(IntEnum):
    """The bottom-up fit of the node boxes of a BVH set.

    See genesis.engine.bvh.
    """

    # One thread per leaf walking to the root, the arrival counters sequentially consistent (CPU)
    LEAF_WALK = 0
    # The leaf walk with a device fence on each side of the relaxed arrival atomic (GPU)
    FENCED_LEAF_WALK = 1
    # One block per tree sweeping the nodes level by level, every barrier uniform (Metal, see quadrants#925)
    BLOCK_SWEEP = 2


@qd.data_oriented
class BVHStaticConfig(StaticConfig):
    """The compile-time shape of a set of linear bounding volume hierarchies (LBVH).

    See genesis.engine.bvh. The leaf keys pack the tree, the morton code and the leaf index in that order, so one flat
    sort orders every tree.
    """

    # The bits of a morton code, three interleaved coordinates of the same width
    morton_bits: int
    # The bits a sort must order, a multiple of 16 up to 64
    end_bit: int
    # Scan depth of the device-wide radix sort (see quadrants.algorithms.sort), 1 to 8
    log256_max_n: int
    # BVH_SORT_KIND
    sort_kind: int
    # BVH_FIT_KIND
    fit_kind: int


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class BVHTreeState:
    """What a traversal reads of a set of linear bounding volume hierarchies (LBVH).

    See genesis.engine.bvh for the build. The internal nodes come first, n_leaves - 1 of them, then the leaves, so
    node n_leaves - 1 + i holds the i-th sorted leaf, whose leaf index is leaves_idx[i_t, i]. A leaf node holds -1 in
    nodes_left.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    nodes_left: qd.Tensor
    nodes_right: qd.Tensor
    nodes_min: qd.Tensor
    nodes_max: qd.Tensor
    leaves_idx: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class BVHLeaves:
    """The boxes of the leaves of a set of linear bounding volume hierarchies (LBVH).

    See genesis.engine.bvh. The caller writes them before a build, and a box query tests them against a tree.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    aabbs_min: qd.Tensor
    aabbs_max: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class BVHState:
    """The linear bounding volume hierarchies (LBVH) of a set of trees.

    A build (see genesis.engine.bvh) takes the boxes of the leaves (see BVHLeaves) and produces the tree a traversal
    reads (see BVHTreeState), plus the sorted leaf keys, the parent of each node and the scratch of the sort, the extent
    and the fit.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    leaves: BVHLeaves
    tree: BVHTreeState
    leaves_keys: qd.Tensor
    keys_scratch: qd.Tensor
    sort_scratch: qd.Tensor
    sort_hist: qd.Tensor
    n_keys: qd.Tensor
    nodes_parent: qd.Tensor
    nodes_fitted: qd.Tensor
    fit_frontier: qd.Tensor
    lanes_min: qd.Tensor
    lanes_max: qd.Tensor
    trees_min: qd.Tensor
    trees_max: qd.Tensor


def get_bvh_state(
    n_trees: int,
    n_leaves: int,
    n_sort_scratch: int,
    n_sort_chunks: int,
    n_extent_lanes: int,
    bvh_config: BVHStaticConfig,
    is_active: bool,
) -> BVHState:
    n_nodes = 2 * n_leaves - 1
    is_device_radix = is_active and bvh_config.sort_kind == BVH_SORT_KIND.DEVICE_RADIX
    is_per_tree_radix = is_active and bvh_config.sort_kind == BVH_SORT_KIND.PER_TREE_RADIX
    is_block_sweep = is_active and bvh_config.fit_kind == BVH_FIT_KIND.BLOCK_SWEEP
    return BVHState(
        tree=BVHTreeState(
            nodes_left=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, n_nodes), is_active)),
            nodes_right=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, n_nodes), is_active)),
            nodes_min=V(dtype=gs.qd_vec3, shape=maybe_shape((n_trees, n_nodes), is_active)),
            nodes_max=V(dtype=gs.qd_vec3, shape=maybe_shape((n_trees, n_nodes), is_active)),
            leaves_idx=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, n_leaves), is_active)),
        ),
        leaves=BVHLeaves(
            aabbs_min=V(dtype=gs.qd_vec3, shape=maybe_shape((n_trees, n_leaves), is_active)),
            aabbs_max=V(dtype=gs.qd_vec3, shape=maybe_shape((n_trees, n_leaves), is_active)),
        ),
        leaves_keys=V(dtype=qd.u64, shape=maybe_shape((n_trees * n_leaves,), is_active)),
        keys_scratch=V(dtype=qd.u64, shape=maybe_shape((n_trees * n_leaves,), is_active)),
        sort_scratch=V(dtype=qd.u32, shape=maybe_shape((n_sort_scratch,), is_device_radix)),
        sort_hist=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, n_sort_chunks, 256), is_per_tree_radix)),
        n_keys=V_SCALAR_FROM(gs.qd_int, n_trees * n_leaves),
        nodes_parent=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, n_nodes), is_active)),
        nodes_fitted=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, max(n_leaves - 1, 1)), is_active)),
        fit_frontier=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, 2, max(n_leaves - 1, 1)), is_block_sweep)),
        lanes_min=V(dtype=gs.qd_vec3, shape=maybe_shape((n_trees, n_extent_lanes), is_active)),
        lanes_max=V(dtype=gs.qd_vec3, shape=maybe_shape((n_trees, n_extent_lanes), is_active)),
        trees_min=V(dtype=gs.qd_vec3, shape=maybe_shape((n_trees,), is_active)),
        trees_max=V(dtype=gs.qd_vec3, shape=maybe_shape((n_trees,), is_active)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class BVHQueryResults:
    """The (tree, leaf, query) triplets a box query of a BVH collects, and their count, above the capacity on overflow.

    See func_bvh_query_aabb in genesis.engine.bvh.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    triplets: qd.Tensor
    count: qd.Tensor


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class BVHQueryState:
    """A box query of a set of trees: the leaves whose boxes are tested, the trees traversed and the pairs found.

    Leaf ``i_q`` of tree ``i_t`` of ``leaves`` is tested against tree ``i_t`` of ``tree`` (see func_bvh_query_leaves
    in genesis.engine.bvh).
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    leaves: BVHLeaves
    tree: BVHTreeState
    results: BVHQueryResults


def get_bvh_query_results(max_results: int, is_active: bool = True) -> BVHQueryResults:
    return BVHQueryResults(
        triplets=V(dtype=gs.qd_ivec3, shape=maybe_shape((max_results,), is_active)),
        count=V(dtype=gs.qd_int, shape=maybe_shape((1,), is_active)),
    )


# =========================================== SAPCoupler ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SAPContactQueriesState:
    """The box queries of the SAP coupler, one per contact handler traversing a tree.

    See SAPCoupler._init_bvh.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    fem_self: BVHQueryState
    rigid_tri: BVHQueryState
    rigid_tet: BVHQueryState
