"""Composition core of the simulation engine: Systems own Data, Actions bind functions to it, pipelines run them.

Data follows the conventions of genesis.utils.array_class: a `@qd.data_oriented` class with `qd.Tensor` fields
allocated by `V()`, named with a `State` / `Info` suffix, created empty with its System and filled by its init Action.

A System is a backend unit the user never names: it is constructed from the scene and declares on its class
the Systems it depends on (Require / Find), the Actions it hands out (@System.action) and the ActionCollections it
receives (@System.action_collection, filled through a @System.protocol). An Engine adds the Systems added by hand, the
Systems the functions registered for its class with @register_system return for the scene, and every System they
require, then resolves the dependencies, runs every build() and freezes the collections. A Pipeline runs one entry
point: a host function or a kernel.
"""

from collections import deque
from collections.abc import Iterable
import functools
import inspect
import types
import typing
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Literal, TypeVar, overload

import quadrants as qd

import genesis as gs
from genesis.utils.misc import check_inheritance_depth

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# ------------------------------------------------------------------------------------
# ------------------------------------- Actions --------------------------------------
# ------------------------------------------------------------------------------------

# Where an Action may be invoked, derived from the wrapped function:
#   "host"   - Python function, invoked from host code
#   "inline" - @qd.func, invoked inside the device loop of the manager
#   "stage"  - @qd.func(requires_top_level=True), invoked at the top level of a kernel or graph body
ActionKind = Literal["host", "inline", "stage"]


def kind_of(function: Callable[..., Any]) -> ActionKind:
    """Derive the kind of an Action from the markers Quadrants sets on its callables."""
    if getattr(function, "_is_wrapped_kernel", False):
        gs.raise_exception(f"{function} is a @qd.kernel, and an Action wraps a function.")
    if not getattr(function, "_is_quadrants_function", False):
        return "host"
    return "stage" if getattr(function, "_qd_requires_top_level", False) else "inline"


# Maps each ActionKind -> the one Action class of that kind, filled when the class is defined
ACTION_TYPES_MAP: dict[ActionKind, type["Action"]] = {}


# WORKAROUND: @qd.data_oriented on the core classes lets Quadrants walk from a kernel parameter through collections and
# Actions to the bound Data, and lets invoke() run in kernel scope. They hold no kernels.
@qd.data_oriented
class Action:
    """A function bound to its leading arguments, which a manager System invokes with the remaining call arguments.

    `bound_args` (Data, or the System of a bound method) are fixed at creation, and `invoke(call_args)` calls
    `function(*bound_args, *call_args)`. The concrete type (HostAction, InlineAction or StageAction) states where the
    manager invokes it, so a protocol annotates which kind each parameter requires.
    """

    kind: ClassVar[ActionKind]

    # Each ActionKind has exactly one Action class, HostAction, InlineAction or StageAction, and each of them is final,
    # so a protocol annotation names the kind it requires without ambiguity
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        check_inheritance_depth(cls, Action, max_depth=1)
        kind = vars(cls).get("kind")
        action_kinds = typing.get_args(ActionKind)
        if kind not in action_kinds:
            gs.raise_exception(f"{cls.__name__} must set `kind` to one of {action_kinds}, got {kind!r}.")
        if kind in ACTION_TYPES_MAP:
            gs.raise_exception(
                f"{cls.__name__} declares kind {kind!r}, which {ACTION_TYPES_MAP[kind].__name__} already holds."
            )
        ACTION_TYPES_MAP[kind] = cls

    def __init__(self, function: Callable[..., Any], bound_args: tuple[object, ...]) -> None:
        self.function = function
        self.bound_args = bound_args

    @qd.pyfunc
    def invoke(self, call_args: qd.template() = ()):
        return self.function(*(self.bound_args + call_args))


@qd.data_oriented
class HostAction(Action):
    kind = "host"


@qd.data_oriented
class InlineAction(Action):
    kind = "inline"


@qd.data_oriented
class StageAction(Action):
    kind = "stage"


@qd.data_oriented
class ActionCollection:
    """Protocol storage owned by a manager System, filled by its protocols and frozen by Engine.

    `collection.add(action, rank=r)` registers with an explicit position, and `collection += action` registers
    unranked. The frozen order puts ranked entries first, sorted by rank, then unranked entries in registration order.
    The rank belongs to this registration, so one System can take different positions in different managers.
    """

    def __init__(self, owner: "System", name: str, kind: ActionKind, call_args: tuple[str, ...]) -> None:
        self._owner = owner
        self._name = name
        self._kind: ActionKind = kind
        self._call_args = call_args
        self._pending: list[tuple[int | None, Action]] = []
        self._frozen: tuple[Action, ...] | None = None

    @property
    def actions(self) -> tuple[Action, ...]:
        """The frozen schedule, available once Engine has finished build()."""
        if self._frozen is None:
            gs.raise_exception(f"{type(self._owner).__name__}.{self._name} is frozen only after build().")
        return self._frozen

    def __iadd__(self, action: Action) -> "ActionCollection":
        self.add(action)
        return self

    def add(self, action: Action, *, rank: int | None = None) -> None:
        """Register one Action, optionally at a fixed rank, from inside a protocol of the owner System."""
        where = f"{type(self._owner).__name__}.{self._name}"
        if not isinstance(action, Action):
            gs.raise_exception(f"{where}: expected an Action, got {type(action).__name__}.")
        # The manager invokes every Action of this collection from one kind of call site. A host function fails in
        # tracing, a stage inside a loop breaks top-level ordering, and an inline function at top level runs serially.
        if action.kind != self._kind:
            gs.raise_exception(f"{where}: expects {self._kind!r} Actions, got {type(action).__name__}.")
        # invoke() calls function(*bound_args, *call_args), so the signature ends with exactly the call arguments of
        # this collection, and a mismatch fails here instead of during graph compilation
        function = getattr(action.function, "fn", action.function)
        parameters = list(inspect.signature(function).parameters.values())
        if any(parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD) for parameter in parameters):
            gs.raise_exception(f"{where}: {function.__qualname__} must not use *args or **kwargs.")
        n_bound = len(action.bound_args)
        call_args = tuple(parameter.name for parameter in parameters[n_bound:])
        if call_args != self._call_args:
            gs.raise_exception(
                f"{where}: {function.__qualname__} binds {n_bound} argument(s) and then takes {call_args}, but the "
                f"protocol calls it with {self._call_args}."
            )
        # None keeps registration order, and an int >= 0 pins a position that one entry at most can hold
        if rank is not None:
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
                gs.raise_exception(f"{where}: rank must be None or an int >= 0, got {rank!r}.")
            if any(rank_taken == rank for rank_taken, _ in self._pending):
                gs.raise_exception(f"{where}: rank {rank} is already taken.")
        self._pending.append((rank, action))

    def freeze(self) -> None:
        """Fix the schedule once Engine has finished build(), since graphs unroll it at compile time."""
        ranked_entries = sorted(
            ((rank, action) for rank, action in self._pending if rank is not None), key=lambda entry: entry[0]
        )
        unranked_entries = [(rank, action) for rank, action in self._pending if rank is None]
        self._frozen = tuple(action for _, action in (*ranked_entries, *unranked_entries))


# ------------------------------------------------------------------------------------
# ----------------------------------- Declarations -----------------------------------
# ------------------------------------------------------------------------------------

T = TypeVar("T")
A = TypeVar("A", bound=Action)
F = TypeVar("F", bound=Callable[..., Any])

# "required" - Require: a missing System fails Engine.build()
# "optional" - Find: a missing System resolves to None
DepLabel = Literal["required", "optional"]


class Dep(Generic[T]):
    """Static dependency declaration, resolved into the instance dict by Engine before build()."""

    label: ClassVar[DepLabel]

    def __init__(self, system_type: type[T]) -> None:
        self.system_type = system_type

    def __set_name__(self, owner, name):
        self._name = name


class Require(Dep[T]):
    """Mandatory dependency, with `self.<name>` typed as T."""

    label = "required"

    @overload
    def __get__(self, instance: None, owner: type) -> "Require[T]": ...
    @overload
    def __get__(self, instance: object, owner: type) -> T: ...
    def __get__(self, instance: object | None, owner: type) -> "Require[T] | T":
        # Non-data descriptor: after resolution the System lives in the instance dict and shadows this
        if instance is None:
            return self
        gs.raise_exception(f"{self._name} is resolved by Engine before build().")


class Find(Dep[T]):
    """Optional dependency, with `self.<name>` typed as T | None."""

    label = "optional"

    @overload
    def __get__(self, instance: None, owner: type) -> "Find[T]": ...
    @overload
    def __get__(self, instance: object, owner: type) -> T | None: ...
    def __get__(self, instance: object | None, owner: type) -> "Find[T] | T | None":
        # Non-data descriptor: after resolution the System lives in the instance dict and shadows this
        if instance is None:
            return self
        gs.raise_exception(f"{self._name} is resolved by Engine before build().")


class ActionCollectionDeclaration:
    """Class-level declaration created by @System.action_collection, whose method body documents the protocol.

    System.__init__ creates the ActionCollection in the instance dict, which shadows this descriptor.
    """

    def __init__(self, function: Callable[..., object], kind: ActionKind, call_args: tuple[str, ...]) -> None:
        self.__doc__ = function.__doc__
        self.kind: ActionKind = kind
        self.call_args = tuple(call_args)

    @overload
    def __get__(self, instance: None, owner: type) -> "ActionCollectionDeclaration": ...
    @overload
    def __get__(self, instance: object, owner: type) -> ActionCollection: ...
    def __get__(self, instance: Any, owner: type) -> "ActionCollectionDeclaration | ActionCollection":
        return self


class ActionDeclaration(Generic[A]):
    """Class-level declaration created by @System.action(kind=...), whose method returns `function, *bound_args`."""

    def __init__(self, function: Callable[..., object], action_type: type[A]) -> None:
        self.__doc__ = function.__doc__
        self.func = function
        self.action_type = action_type

    def __set_name__(self, owner, name):
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type) -> "ActionDeclaration[A]": ...
    @overload
    def __get__(self, instance: object, owner: type) -> A: ...
    def __get__(self, instance: Any, owner: type) -> "ActionDeclaration[A] | A":
        if instance is None:
            return self
        where = f"{type(instance).__name__}.{self._name}"
        # Reached on the first access only. Actions are created during build(), once dependencies are resolved.
        if not instance.is_building:
            gs.raise_exception(f"{where}: Actions are created only in build().")
        declared_result: Any = self.func(instance)
        function, *bound_args = declared_result if isinstance(declared_result, tuple) else (declared_result,)
        # A bound method binds its System as the first argument
        if isinstance(function, types.MethodType):
            function, bound_args = function.__func__, [function.__self__, *bound_args]
        function_kind = kind_of(function)
        if function_kind != self.action_type.kind:
            gs.raise_exception(f"{where}: declared {self.action_type.kind!r}, but the function is {function_kind!r}.")
        action = self.action_type(function, tuple(bound_args))
        # The instance dict shadows this descriptor afterwards, so each System and name holds one Action
        instance.__dict__[self._name] = action
        return action


def wrap_protocol(method, doc: str | None, references: tuple[str, ...]) -> Any:
    """Wrap a protocol method with its build-time checks, and attach its guide and references."""
    signature = inspect.signature(method)
    hints: dict[str, Any] | None = None

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        nonlocal hints
        where = f"{type(self).__name__}.{method.__name__}"
        # A protocol changes the schedule that build() freezes
        if not self.is_building:
            gs.raise_exception(f"{where} is a protocol, valid only during build().")
        # Resolved on the first call, since annotations may name classes defined after the decorated method
        if hints is None:
            hints = typing.get_type_hints(method)
        for name, value in signature.bind(self, *args, **kwargs).arguments.items():
            expected_type = hints.get(name)
            if (
                isinstance(expected_type, type)
                and issubclass(expected_type, Action)
                and not isinstance(value, expected_type)
            ):
                gs.raise_exception(f"{where}: `{name}` requires {expected_type.__name__}, got {type(value).__name__}.")
        return method(self, *args, **kwargs)

    if doc is not None or references:
        reference_lines = [f"Guide: {doc}"] if doc is not None else []
        reference_lines += [f"Reference: {reference}" for reference in references]
        wrapper.__doc__ = "\n\n".join(
            filter(None, [inspect.cleandoc(wrapper.__doc__ or ""), "\n".join(reference_lines)])
        )
    return wrapper


# ------------------------------------------------------------------------------------
# -------------------------------------- Systems -------------------------------------
# ------------------------------------------------------------------------------------


class System:
    """One backend unit of the simulation, constructed from the scene, that owns its Data and composes with
    other Systems.

    Engine adds a System instance when it is added by hand, when a function registered with `register_system` returns
    it for the scene, or when another added System or the Engine requires its class, which Engine then constructs from
    the scene. Behavior is shared by composition: System accepts one level of inheritance only.
    """

    # Read-only mapping from attribute name to its Require / Find declaration
    deps: ClassVar[MappingProxyType[str, Dep]] = MappingProxyType({})
    # Names of the declared ActionCollections
    action_collections: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Single-level inheritance with no intermediate bases or mixins, so behavior is composed
        check_inheritance_depth(cls, System, max_depth=1)
        cls.deps = MappingProxyType({name: dep for name, dep in vars(cls).items() if isinstance(dep, Dep)})
        cls.action_collections = tuple(
            name for name, value in vars(cls).items() if isinstance(value, ActionCollectionDeclaration)
        )

    def __init__(self, scene: "Scene") -> None:
        self._scene = scene
        self.is_building = False
        for name in type(self).action_collections:
            declaration = vars(type(self))[name]
            self.__dict__[name] = ActionCollection(self, name, declaration.kind, declaration.call_args)

    @overload
    @staticmethod
    def protocol(method: F, /) -> F: ...
    @overload
    @staticmethod
    def protocol(*, doc: str | None = None, references: tuple[str, ...] = ()) -> Callable[[F], F]: ...
    @staticmethod
    def protocol(method=None, /, *, doc=None, references=()):
        """Mark an `on_*` method that receives Actions, valid only during build().

        Parameters annotated with HostAction, InlineAction or StageAction state which kind of Action they require, and
        every call is checked against those annotations.

        A protocol is the contract other Systems implement, so it defines every Action parameter:
            Signature  function(*bound_args, <call arguments>), matching the `call_args` of the collection
            Invoked    when and where the manager invokes it, and its order relative to the other Actions
            Must       what the Action has to do, including any count or range it must respect
            Must not   what it may not touch at that point
        A short protocol does this in its docstring (bare `@System.protocol`). A protocol whose contract needs
        derivations or a paper keeps a short docstring and moves the contract into a companion Markdown guide:

            @System.protocol(doc="integrator_protocol.md", references=("https://...",))

        `doc` is a path relative to the file that defines the protocol.
        `references` are links to papers or specifications. Both are appended to the docstring, so they show on hover.
        """
        if method is not None:
            return wrap_protocol(method, None, ())
        return lambda method: wrap_protocol(method, doc, tuple(references))

    @staticmethod
    def action_collection(
        *, kind: ActionKind, call_args: tuple[str, ...] = ()
    ) -> Callable[[Callable[..., object]], ActionCollectionDeclaration]:
        """Declare one ActionCollection owned by a manager System.

        kind       The kind of every Action it accepts: "host", "inline" or "stage" (see System.action). It matches
                   where the manager invokes them: host code, inside its own loop, or at the top level.
        call_args  The names of the arguments the manager supplies after the bound arguments of each Action. A function
                   whose remaining parameters differ in count, order or name is rejected on `+=`.

        The decorated method body is never executed, and its docstring documents the protocol.
        """
        return lambda function: ActionCollectionDeclaration(function, kind, call_args)

    @overload
    @staticmethod
    def action(*, kind: Literal["host"]) -> Callable[[Callable[..., Any]], ActionDeclaration[HostAction]]: ...
    @overload
    @staticmethod
    def action(*, kind: Literal["inline"]) -> Callable[[Callable[..., Any]], ActionDeclaration[InlineAction]]: ...
    @overload
    @staticmethod
    def action(*, kind: Literal["stage"]) -> Callable[[Callable[..., Any]], ActionDeclaration[StageAction]]: ...
    @staticmethod
    def action(*, kind: ActionKind) -> Callable[[Callable[..., Any]], ActionDeclaration[Any]]:
        """Declare one Action of the given kind.

        The decorated method returns `function, *bound_args`. Its first access, which happens during build(), binds
        them into one Action of the declared kind and caches it, and every later access returns that object. A bound
        method binds its own System, so `return self.initialize` is a complete Action.

        `kind` states where the manager invokes the Action. It is required, and it matches the function:

        kind="host"
            Function:  a plain Python function or bound method.
            Invoked:   from host code, e.g. the host function of an init pipeline.
            Runs:      on the host, and may allocate, call from_numpy, or launch kernels itself.
            Use for:   one-time setup that requires the host. Uncommon, since device-capable bookkeeping such as
                       counting or offsets belongs in a kernel.
            Typed as:  HostAction.

        kind="inline"
            Function:  a @qd.func without requires_top_level.
            Invoked:   inside the device loop of the manager, once per iteration or once per participant.
            Runs:      inlined into that loop body, as part of the parallel task of the loop.
            Use for:   small per-element or per-participant work, such as reporting a count or returning the
                       contribution of one element to a reduction.
            Typed as:  InlineAction.

        kind="stage"
            Function:  a @qd.func(requires_top_level=True).
            Invoked:   at the top level of a kernel or graph body.
            Runs:      each of its top-level for-loops becomes a parallel task (a graph node), i.e. one pipeline stage.
            Use for:   work that is parallel over the data of the participant, such as a loop over all its DOFs.
            Typed as:  StageAction.

        Each kind has one valid call site: a host function fails when traced into device code, a stage called inside a
        loop has its top-level loops demoted and loses stage ordering, and an inline function called at top level runs
        as one serial step. The declared kind is therefore checked three times: against the function when the Action is
        created (a @qd.kernel is rejected), against the collection on `+=`, and against the protocol parameter
        annotation (HostAction / InlineAction / StageAction) when the protocol is called.
        """
        return lambda function: ActionDeclaration(function, ACTION_TYPES_MAP[kind])

    def build(self):
        pass


def get_required_system_types(owner_types: Iterable[type[Any]]) -> tuple[type[System], ...]:
    """Return the breadth-first, ordered unique closure of required System types absent from the initial owner types."""
    pending_types = deque(owner_types)
    seen_types = set(pending_types)
    required_system_types: list[type[System]] = []
    while pending_types:
        owner_type = pending_types.popleft()
        for dep in owner_type.deps.values():
            if dep.label == "required" and dep.system_type not in seen_types:
                seen_types.add(dep.system_type)
                pending_types.append(dep.system_type)
                required_system_types.append(dep.system_type)
    return tuple(required_system_types)


# ------------------------------------------------------------------------------------
# ------------------------------- Pipelines and engine -------------------------------
# ------------------------------------------------------------------------------------

# "host"   - entry is a Python function
# "device" - entry is a @qd.kernel, which may yield and resume when declared with graph=True and checkpoints=True
PipelineKind = Literal["host", "device"]


class Pipeline:
    """One entry point with its bound arguments, recognized from what it wraps: a Python function or a @qd.kernel.

    `bound_args` are passed to the entry on every launch and on every resume: `Pipeline(kernel_step, engine)` runs
    `kernel_step(engine)`. Pipeline is final.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        check_inheritance_depth(cls, Pipeline, max_depth=0)

    def __init__(
        self,
        entry: Callable[..., Any],
        *bound_args: object,
        yield_callbacks: dict[int, Callable[..., Any]] | None = None,
    ):
        # A Quadrants callable keeps the decorated Python function as `fn`
        entry_name = getattr(getattr(entry, "fn", entry), "__qualname__", repr(entry))
        if getattr(entry, "_is_wrapped_kernel", False):
            kind: PipelineKind = "device"
        elif getattr(entry, "_is_quadrants_function", False):
            gs.raise_exception(f"{entry_name} is a @qd.func, and a device entry is a @qd.kernel that calls it.")
        else:
            kind = "host"
        # Only a checkpointed graph kernel yields, so callbacks anywhere else would never run
        primal = getattr(getattr(entry, "quadrants_callable", entry), "_primal", None)
        has_checkpoints = kind == "device" and getattr(primal, "use_checkpoints", False)
        if yield_callbacks and not has_checkpoints:
            gs.raise_exception(
                f"{entry_name} has no checkpoints, so it takes no yield callbacks. Declare it "
                "@qd.kernel(graph=True, checkpoints=True)."
            )
        # A Quadrants kernel exposes resume(), which Callable cannot express
        self.entry: Any = entry
        self.bound_args = bound_args
        self.kind: PipelineKind = kind
        self.yield_callbacks = dict(yield_callbacks or {})

    def run(self) -> None:
        """Run the entry once, dispatching yield callbacks and resuming until a graph kernel completes."""
        status = self.entry(*self.bound_args)
        # A host entry and a plain kernel return None, and a graph kernel returns a GraphStatus that may have yielded
        while status is not None and status.yielded:
            checkpoint = int(status.checkpoint)
            callback = self.yield_callbacks.get(checkpoint)
            if callback is None:
                gs.raise_exception(f"No yield callback for checkpoint {checkpoint}.")
            resume_from = callback(status)
            status = self.entry.resume(
                *self.bound_args, from_checkpoint=checkpoint if resume_from is None else int(resume_from)
            )


SystemT = TypeVar("SystemT", bound=System)
CreateFunT = TypeVar("CreateFunT", bound=Callable[["Scene"], System | None])

# Lists rather than sets, since the registration order fixes the order in which an Engine adds the Systems
_system_registry: dict[type["Engine"], list[Callable[["Scene"], System | None]]] = {}


def register_system(engine_cls: type["Engine"]) -> Callable[[CreateFunT], CreateFunT]:
    """Register, for one final Engine class, a function that receives the scene and returns the System it calls for, or
    None for no System.

    Used as `@register_system(MyEngine)`, once per Engine class the function serves. The build of an Engine calls every
    function registered for its exact class once, in registration order, and adds every System returned.
    """
    # Engine.__init_subclass__ makes every proper subclass of Engine final
    if not isinstance(engine_cls, type) or not issubclass(engine_cls, Engine) or engine_cls is Engine:
        gs.raise_exception(f"Systems are registered for a final Engine subclass, got {engine_cls!r}.")

    def register(create_fun: CreateFunT) -> CreateFunT:
        assert isinstance(create_fun, Callable)
        _system_registry.setdefault(engine_cls, []).append(create_fun)
        return create_fun

    return register


def unregister_system(engine_cls: type["Engine"], create_fun: Callable[["Scene"], System | None]) -> None:
    assert isinstance(create_fun, Callable)
    _system_registry[engine_cls].remove(create_fun)


class Engine:
    """The owner of all Systems of one simulation, which adds and builds them and holds the pipelines.

    A concrete Engine inherits directly from Engine and is final: subclassing it again is rejected.
    """

    # Read-only mapping from attribute name to its Require / Find declaration
    deps: ClassVar[MappingProxyType[str, Dep]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Every Engine subclass is final, so the Systems registered for it apply to exactly one class
        check_inheritance_depth(cls, Engine, max_depth=1)
        cls.deps = MappingProxyType({name: dep for name, dep in vars(cls).items() if isinstance(dep, Dep)})

    def __init__(self, scene: "Scene") -> None:
        self.scene = scene
        self.systems: dict[type[System], System] = {}

    def add_system(self, system: SystemT) -> SystemT:
        """Add a System instance, at most one per class, and return it."""
        if type(system) in self.systems:
            gs.raise_exception(f"{type(system).__name__} is already added.")
        self.systems[type(system)] = system
        # WORKAROUND: expose each System as an attribute so a kernel taking the engine reaches its Data
        setattr(self, f"_system_{len(self.systems)}", system)
        return system

    def build(self) -> None:
        """Add the registered and required Systems, resolve every dependency, run build() on every System in add order,
        then freeze every ActionCollection.

        Systems added by hand come first, then the Systems the functions registered for this Engine class return for
        the scene (see register_system), then every System an added System or the Engine requires, constructed from
        the scene, breadth first. A Find adds nothing: it resolves to the System when
        something else added it, and to None otherwise.
        """
        for create_fun in _system_registry.get(type(self), ()):
            system = create_fun(self.scene)
            if system is not None:
                self.add_system(system)
        # Compute the ordered unique closure of required System types before constructing any of them
        for system_type in get_required_system_types((type(self), *self.systems)):
            self.add_system(system_type(self.scene))
        # Resolve required and optional dependencies once the complete System set is fixed
        owners: list[Engine | System] = [self, *self.systems.values()]
        for owner in owners:
            for name, dep in owner.deps.items():
                owner.__dict__[name] = self.systems.get(dep.system_type)
        # Actions are created and protocols called only while the build window is open
        for system in self.systems.values():
            system.is_building = True
        for system in self.systems.values():
            system.build()
        # Closing the window freezes every declared collection, so the graph topology is fixed from here on
        for system in self.systems.values():
            system.is_building = False
            for name in type(system).action_collections:
                system.__dict__[name].freeze()
