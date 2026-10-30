"""
Arrays a Quadrants kernel may write, from its source.

The backward pass of a recorded substep saves the values a launch overwrites that an earlier launch reads (see
LaunchRecorder). Quadrants records the struct members a compiled kernel uses, but not which of them it writes, so the
writes are read from the Python source of the kernel and of the funcs it calls. The analysis over-approximates: both
branches of a static condition count, so a write it reports may never happen, which costs a copy, never a gradient.
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


# The paths each kernel or func may write, by function and bound object, each entry living as long as the bound object
# (the function itself for a free function)
_WRITTEN_PATHS_CACHE = OwnedCache()


# FIXME: quadrants#1001 - Quadrants exposes no write set of a kernel, so it is derived from the source here, which
# duplicates the analysis of its frontend and counts both branches of a static condition.
def get_written_paths(fn: Callable, owner: object | None = None) -> frozenset[tuple[str, ...]]:
    """
    Return the paths of the parameters of a kernel or a func that it may write.

    A path is the name of a parameter followed by the struct fields reaching the array, such as
    `("dyn_state", "dofs", "vel")`. A write is an assignment to an element, an augmented assignment, the first argument
    of an atomic operation or a `fill`, in the function itself or in a func it calls with the array as argument. `owner`
    is the object the first parameter of a method is bound to, through which it reaches the funcs of other objects.
    """
    cache_key = (fn, id(owner))
    paths_cached = _WRITTEN_PATHS_CACHE.find(cache_key)
    if paths_cached is not None:
        return paths_cached
    func_def = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    params = [arg.arg for arg in (*func_def.args.posonlyargs, *func_def.args.args, *func_def.args.kwonlyargs)]
    owner_name = params[0] if owner is not None else None
    aliases = {param: {(param,)} for param in params}
    # Local names bound to a parameter or to fields of one, until no binding adds a path
    assigns = [node for node in ast.walk(func_def) if isinstance(node, ast.Assign)]
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
                    if isinstance(name, ast.Name) and paths and not paths <= aliases.get(name.id, set()):
                        aliases.setdefault(name.id, set()).update(paths)
                        is_growing = True

    # Local names bound to a func, directly or through 'qd.static'
    funcs = {}
    for node in assigns:
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "static":
            value = value.args[0]
        callee = _find_callee(value, fn, {}, owner_name, owner)
        for target in node.targets:
            if isinstance(target, ast.Name) and callee is not None:
                funcs[target.id] = callee

    paths_written = set()
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
                callee = _find_callee(node.func, fn, funcs, owner_name, owner)
                if callee is None:
                    continue
                # The arrays the func writes through its parameters, bound to the arguments of the call and, for a
                # method, to the object it is bound to
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
                callee_paths_written = get_written_paths(callee.fn, callee.owner)
                for callee_param, arg_paths in bindings:
                    for callee_path in callee_paths_written:
                        if callee_path[0] == callee_param:
                            paths_written |= {(*path, *callee_path[1:]) for path in arg_paths}
    params_paths_written = frozenset([path for path in paths_written if path[0] in params])
    return _WRITTEN_PATHS_CACHE.own(fn if owner is None else owner, cache_key, params_paths_written)


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


def _find_callee(
    node: ast.expr, fn: Callable, funcs: dict[str, Callee], owner_name: str | None, owner: object | None
) -> Callee | None:
    """
    Return the Quadrants func an expression names, None for anything else.

    `fn` is the function whose source holds the expression, `funcs` the local names it binds to a func, and `owner` the
    object its first parameter, named `owner_name`, is bound to.
    """
    if isinstance(node, ast.Name) and node.id in funcs:
        return funcs[node.id]
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
