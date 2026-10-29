"""Shared helpers for translating a desired collision matrix into contype/conaffinity bitmasks."""

from functools import lru_cache
from time import perf_counter

import z3

from .cache import register_cache_clear

# A fallback search keeps refuting one width per bit below its answer, so it is abandoned after this delay in total,
# and every check is bounded by what remains of it. Callers fall back on their default masks when nothing is found.
Z3_SEARCH_TIMEOUT_S = 30


# Geoms of one entity are parsed once per scene it is added to, which asks for the same assignment every time.
@lru_cache(maxsize=64)
def solve_contype_conaffinity(
    n: int, invalid_pairs: frozenset[frozenset[int]], max_bits: int = 31
) -> tuple[tuple[int, int], ...] | None:
    """
    Synthesize per-geom ``(contype, conaffinity)`` bitmasks realizing a desired collision matrix.

    Genesis (like MuJoCo) decides whether two geoms ``i`` and ``j`` may collide with the rule
    ``(contype[i] & conaffinity[j]) | (contype[j] & conaffinity[i]) != 0``. Given the set of pairs that must **not**
    collide, this finds bitmasks such that every excluded pair is disabled and every other pair is enabled. Used by
    both the MJCF importer (``<contact><exclude>``) and the USD importer (CollisionGroup / FilteredPairsAPI).

    Geoms excluding exactly the same partners are interchangeable, so one bit per group of such geoms discriminates
    them against every other group, in O(n^2) time. Each geom takes the bit of its group as ``contype``, and as
    ``conaffinity`` the bits of every group it may collide with, its own group included. A collision matrix whose
    groups do not fit in ``max_bits`` bits falls back on an ascending z3 search over the bit widths, which gives up
    after ``Z3_SEARCH_TIMEOUT_S`` seconds. The result is memoized per input.

    Every geom keeps at least one bit set across its two masks: a geom whose masks are both zero would be demoted to
    a visual-only geom downstream (``contype or conaffinity`` is the collision-geom discriminator), silently
    disabling its collision against every other entity as well.

    Parameters
    ----------
    n : int
        Number of collision geoms (indices ``0..n-1``).
    invalid_pairs : frozenset[frozenset[int]]
        Pairs ``frozenset({i, j})`` that must be prevented from colliding.
    max_bits : int
        Maximum number of bits the masks may use (default 31, keeping masks within a signed 32-bit int).

    Returns
    -------
    tuple[tuple[int, int], ...] | None
        ``(contype, conaffinity)`` per geom index, or ``None`` if no assignment up to ``max_bits`` bits was found.
    """
    excluded_geoms = [set() for _ in range(n)]
    for i_ga in range(n):
        for i_gb in range(i_ga + 1, n):
            if frozenset((i_ga, i_gb)) in invalid_pairs:
                excluded_geoms[i_ga].add(i_gb)
                excluded_geoms[i_gb].add(i_ga)

    group_of_exclusion = {}
    group_of_geom = [
        group_of_exclusion.setdefault(frozenset(exclusion), len(group_of_exclusion)) for exclusion in excluded_geoms
    ]
    num_groups = len(group_of_exclusion)
    if num_groups <= max_bits:
        # A geom never excludes its own group, since a geom excluding a partner of its group would exclude itself.
        conaffinity_of_group = [(1 << num_groups) - 1] * num_groups
        for exclusion, i_group in group_of_exclusion.items():
            for i_g in exclusion:
                conaffinity_of_group[i_group] &= ~(1 << group_of_geom[i_g])
        return tuple((1 << i_group, conaffinity_of_group[i_group]) for i_group in group_of_geom)

    deadline = perf_counter() + Z3_SEARCH_TIMEOUT_S
    for num_bits in range(1, max_bits + 1):
        solver = z3.Solver()
        contype_bits = [[z3.Bool(f"contype_{i}_{b}") for b in range(num_bits)] for i in range(n)]
        conaffinity_bits = [[z3.Bool(f"conaffinity_{i}_{b}") for b in range(num_bits)] for i in range(n)]
        for i in range(n):
            solver.add(z3.Or([*contype_bits[i], *conaffinity_bits[i]]))
            for j in range(i + 1, n):
                cond1 = z3.Or([z3.And(contype_bits[i][b], conaffinity_bits[j][b]) for b in range(num_bits)])
                cond2 = z3.Or([z3.And(contype_bits[j][b], conaffinity_bits[i][b]) for b in range(num_bits)])
                if frozenset((i, j)) in invalid_pairs:
                    solver.add(z3.Not(cond1), z3.Not(cond2))
                else:
                    solver.add(z3.Or(cond1, cond2))
        # Building the constraints of a width takes time on its own, so the delay left is read once they are built.
        remaining_ms = int((deadline - perf_counter()) * 1000)
        if remaining_ms <= 0:
            return None
        solver.set("timeout", remaining_ms)
        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return tuple(
                tuple(
                    sum((1 << b) if z3.is_true(model[e]) else 0 for b, e in enumerate(bits))
                    for bits in (contype_bits[i], conaffinity_bits[i])
                )
                for i in range(n)
            )
        # A width the remaining delay cannot refute ends the search, since the wider widths cost more to refute.
        if result != z3.unsat:
            return None
    return None


register_cache_clear(solve_contype_conaffinity.cache_clear)
