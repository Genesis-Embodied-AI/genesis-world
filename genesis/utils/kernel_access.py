"""
Arrays a Quadrants kernel, or a method launching kernels, may read and write, from its source.

Quadrants records the struct members a compiled kernel uses, but not which of them it writes, and only once the kernel
is compiled, so both are read from the Python source of the function and of the funcs and kernels it calls. The backward
pass of a recorded substep saves the values a launch overwrites that an earlier launch reads (see LaunchRecorder), and
the change logs follow the arrays every launch writes and the arrays the derived ones are computed from (see
'genesis.engine.state_change'). The analysis over-approximates: both branches of a static condition count, so an access
it reports may never happen, which costs a copy or a refresh, never a gradient or a stale value.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import NamedTuple

from quadrants.lang._quadrants_callable import QuadrantsCallable

from genesis.utils.cache import OwnedCache


class Callee(NamedTuple):
    """A func a call reaches: its Python function, the object its first parameter is bound to and the path of it."""

    fn: Callable
    owner: object | None
    owner_path: tuple[str, ...]


class KernelAccess(NamedTuple):
    """The arrays a function may access, each as the path of the parameter reaching it (see 'get_accessed_paths')."""

    paths_used: frozenset[tuple[str, ...]]
    """The paths the function may read or write, the written ones included. A path ending on a struct stands for every
    array of it."""
    paths_written: frozenset[tuple[str, ...]]
    """The paths the function may write."""


# The paths each function may access, by function and bound object, each entry living as long as the bound object (the
# function itself for a free function).
_ACCESSED_PATHS_CACHE = OwnedCache()


# FIXME: quadrants#1001 - Quadrants exposes no write set of a kernel, so it is derived from the source here, which
# duplicates the analysis of its frontend and counts both branches of a static condition.
def get_accessed_paths(fn: Callable, owner: object | None = None) -> KernelAccess:
    """
    Return the paths of the parameters of a kernel, a func or a method launching kernels that it may read and write.

    A path is the name of a parameter followed by the struct fields reaching the array, such as
    `("dyn_state", "dofs", "vel")`. A write is an assignment to an element, an augmented assignment, the first argument
    of an atomic operation or a `fill`, in the function itself or in a func or kernel it calls with the array as
    argument, a kernel picked through a conditional expression counting both of its branches.

    Parameters
    ----------
    fn : Callable
        The kernel, func or method.
    owner : object | None, optional
        The object the first parameter of a method is bound to, through which it reaches the funcs of other objects.
        Defaults to None.

    Returns
    -------
    access : KernelAccess
        The paths the function may read and write.
    """
    # A method under decorators is analyzed as written, in the scope it was written in
    fn = inspect.unwrap(fn)
    cache_key = (fn, id(owner))
    access_cached = _ACCESSED_PATHS_CACHE.find(cache_key)
    if access_cached is not None:
        return access_cached
    func_def = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    params = [arg.arg for arg in (*func_def.args.posonlyargs, *func_def.args.args, *func_def.args.kwonlyargs)]
    owner_name = params[0] if owner is not None else None
    aliases = {param: {(param,)} for param in params}
    # Local names bound to a parameter or to fields of one, until no binding adds a path. The expressions binding them
    # name a struct without reading it, as do the arguments a call binds to the parameters of a func.
    assigns = [node for node in ast.walk(func_def) if isinstance(node, ast.Assign)]
    nodes_binding = set()
    is_growing = True
    while is_growing:
        is_growing = False
        for node in assigns:
            for target in node.targets:
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    bindings = zip(target.elts, node.value.elts)
                else:
                    bindings = ((target, node.value),)
                for name, value in bindings:
                    paths = _resolve(value, aliases)
                    if isinstance(name, ast.Name) and paths:
                        nodes_binding.add(value)
                        if not paths <= aliases.get(name.id, set()):
                            aliases.setdefault(name.id, set()).update(paths)
                            is_growing = True

    # Local names bound to funcs, directly, through 'qd.static', through either branch of a conditional expression or
    # as the elements of a tuple.
    funcs: dict[str, list[Callee]] = {}
    for node in assigns:
        for target in node.targets:
            if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                bindings = zip(target.elts, node.value.elts)
            else:
                bindings = ((target, node.value),)
            for name, value in bindings:
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "static"
                ):
                    value = value.args[0]
                values = (value.body, value.orelse) if isinstance(value, ast.IfExp) else (value,)
                callees = [callee for callee in [_find_callee(v, fn, owner_name, owner) for v in values] if callee]
                if isinstance(name, ast.Name) and callees:
                    funcs.setdefault(name.id, []).extend(callees)

    paths_used, paths_written = set(), set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                paths_written |= _resolve_store(target, aliases)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            paths_written |= _resolve_store(node.target, aliases)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                name = ""
            if name.startswith("atomic_") and node.args:
                paths_written |= _resolve(node.args[0], aliases)
            elif name == "fill" and isinstance(node.func, ast.Attribute):
                paths_written |= _resolve(node.func.value, aliases)
            else:
                callees = funcs.get(node.func.id, []) if isinstance(node.func, ast.Name) else []
                if not callees:
                    callee = _find_callee(node.func, fn, owner_name, owner)
                    callees = [] if callee is None else [callee]
                for callee in callees:
                    # The arrays the func accesses through its parameters, bound to the arguments of the call and, for
                    # a method, to the object it is bound to.
                    callee_params = list(inspect.signature(callee.fn).parameters)
                    bindings = []
                    if callee.owner is not None:
                        bindings.append((callee_params.pop(0), {callee.owner_path}))
                    bindings += [(param, _resolve(arg, aliases)) for param, arg in zip(callee_params, node.args)]
                    bindings += [
                        (keyword.arg, _resolve(keyword.value, aliases))
                        for keyword in node.keywords
                        if keyword.arg is not None
                    ]
                    nodes_binding.update(node.args)
                    nodes_binding.update([keyword.value for keyword in node.keywords])
                    callee_access = get_accessed_paths(callee.fn, callee.owner)
                    for callee_param, arg_paths in bindings:
                        for callee_paths, paths in (
                            (callee_access.paths_used, paths_used),
                            (callee_access.paths_written, paths_written),
                        ):
                            for callee_path in callee_paths:
                                if callee_path[0] == callee_param:
                                    paths |= {(*path, *callee_path[1:]) for path in arg_paths}

    # The arrays read in place: each outermost expression naming a parameter or fields of one, the names a binding
    # stores to and the expressions it binds aside.
    nodes_inner = {child.value for child in ast.walk(func_def) if isinstance(child, ast.Attribute)}
    for node in ast.walk(func_def):
        if not isinstance(node, (ast.Name, ast.Attribute)) or node in nodes_inner or node in nodes_binding:
            continue
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            continue
        paths_used |= _resolve(node, aliases)
    paths_used |= paths_written

    access = KernelAccess(
        frozenset([path for path in paths_used if path[0] in params]),
        frozenset([path for path in paths_written if path[0] in params]),
    )
    return _ACCESSED_PATHS_CACHE.own(fn if owner is None else owner, cache_key, access)


def _resolve(node: ast.expr, aliases: dict[str, set[tuple[str, ...]]]) -> set[tuple[str, ...]]:
    """Return the paths an expression may designate: a name bound to a path, its fields, and elements of them."""
    if isinstance(node, ast.Name):
        return set(aliases.get(node.id, ()))
    if isinstance(node, ast.Attribute):
        return {(*path, node.attr) for path in _resolve(node.value, aliases)}
    if isinstance(node, ast.Subscript):
        return _resolve(node.value, aliases)
    return set()


def _resolve_store(target: ast.expr, aliases: dict[str, set[tuple[str, ...]]]) -> set[tuple[str, ...]]:
    """Return the paths of the arrays an assignment target writes, the element of an array or a field of one."""
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*[_resolve_store(elt, aliases) for elt in target.elts])
    if isinstance(target, ast.Subscript):
        return _resolve(target.value, aliases)
    if isinstance(target, ast.Attribute) and any([isinstance(node, ast.Subscript) for node in ast.walk(target.value)]):
        return _resolve(target.value, aliases)
    return set()


def _find_callee(node: ast.expr, fn: Callable, owner_name: str | None, owner: object | None) -> Callee | None:
    """
    Return the Quadrants func an expression names, None for anything else.

    `fn` is the function whose source holds the expression, and `owner` the object its first parameter, named
    `owner_name`, is bound to.
    """
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    names.reverse()
    if node.id == owner_name:
        # A method of the owner or of an object it holds, bound to that object
        callee_owner = owner
        for name in names[:-1]:
            callee_owner = getattr(callee_owner, name, None)
        value = getattr(type(callee_owner), names[-1], None) if names else None
        owner_path = (owner_name, *names[:-1])
        if value is None and names:
            # A func the object holds as an attribute of its own, unbound
            value, callee_owner, owner_path = getattr(callee_owner, names[-1], None), None, ()
    else:
        closure = zip(fn.__code__.co_freevars, [cell.cell_contents for cell in fn.__closure__ or ()])
        scope = {**fn.__globals__, **dict(closure)}
        if node.id not in scope:
            return None
        value = scope[node.id]
        for name in names:
            value = getattr(value, name, None)
        callee_owner, owner_path = None, ()
    if not isinstance(value, QuadrantsCallable) or value.fn is fn:
        return None
    return Callee(value.fn, callee_owner, owner_path)
