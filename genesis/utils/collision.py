"""Shared helpers for translating a desired collision matrix into contype/conaffinity bitmasks."""

from functools import lru_cache
from time import perf_counter

import z3

from .misc import register_cache_clear

# A fallback search keeps refuting one width per bit below its answer, so it is abandoned after this delay in total,
# and every check is bounded by what remains of it. Callers fall back on their default masks when nothing is found.
Z3_SEARCH_TIMEOUT_S = 30


def _synthesize_bitmasks_constructively(
    n: int, invalid_pairs: frozenset[frozenset[int]], max_bits: int
) -> list[tuple[int, int]] | None:
    """
    Assign one bit per group of geoms that forbid exactly the same partners, or None if that needs more than max_bits.

    Geoms of one group forbid the same partners, so they are interchangeable: a single bit discriminates all of them
    against every other group, and the forbidden partners of one of them name the groups it must not enable. Each geom
    takes the bit of its own group as ``contype``, and as ``conaffinity`` the bits of every group it may collide with,
    its own group included.

    Parameters
    ----------
    n : int
        Number of collision geoms (indices ``0..n-1``).
    invalid_pairs : frozenset[frozenset[int]]
        Pairs ``frozenset({i, j})`` that must be prevented from colliding.
    max_bits : int
        Maximum number of bits the masks may use.

    Returns
    -------
    list[tuple[int, int]] | None
        ``(contype, conaffinity)`` per geom index, or ``None`` if the grouping needs more than ``max_bits`` bits.
    """
    forbidden_geoms_of_geom = [set() for _ in range(n)]
    for i_ga in range(n):
        for i_gb in range(i_ga + 1, n):
            if frozenset((i_ga, i_gb)) in invalid_pairs:
                forbidden_geoms_of_geom[i_ga].add(i_gb)
                forbidden_geoms_of_geom[i_gb].add(i_ga)

    group_of_geom = [0] * n
    group_of_forbidden_set = {}
    for i_g in range(n):
        forbidden_set = frozenset(forbidden_geoms_of_geom[i_g])
        group_of_geom[i_g] = group_of_forbidden_set.setdefault(forbidden_set, len(group_of_forbidden_set))
    num_groups = len(group_of_forbidden_set)
    if num_groups > max_bits:
        return None

    forbidden_groups_of_group = [set() for _ in range(num_groups)]
    for i_g in range(n):
        i_group = group_of_geom[i_g]
        for i_gb in forbidden_geoms_of_geom[i_g]:
            forbidden_groups_of_group[i_group].add(group_of_geom[i_gb])

    conaffinity_of_group = []
    for i_group in range(num_groups):
        conaffinity = 1 << i_group
        for i_partner in range(num_groups):
            if i_partner not in forbidden_groups_of_group[i_group]:
                conaffinity |= 1 << i_partner
        conaffinity_of_group.append(conaffinity)

    return [(1 << group_of_geom[i_g], conaffinity_of_group[group_of_geom[i_g]]) for i_g in range(n)]


def _synthesize_bitmasks_with_z3(
    n: int, invalid_pairs: frozenset[frozenset[int]], max_bits: int
) -> list[tuple[int, int]] | None:
    """
    Search for the narrowest bitmask assignment with z3, giving up after ``Z3_SEARCH_TIMEOUT_S`` seconds.

    Only reached for collision matrices whose grouping needs more than ``max_bits`` bits, which the constructive
    assignment cannot express. Every check is bounded by what remains of the delay, since refuting a width may be
    arbitrarily expensive.

    Parameters
    ----------
    n : int
        Number of collision geoms (indices ``0..n-1``).
    invalid_pairs : frozenset[frozenset[int]]
        Pairs ``frozenset({i, j})`` that must be prevented from colliding.
    max_bits : int
        Maximum number of bits to try.

    Returns
    -------
    list[tuple[int, int]] | None
        ``(contype, conaffinity)`` per geom index, or ``None`` if no assignment up to ``max_bits`` bits was found in
        time.
    """
    deadline = perf_counter() + Z3_SEARCH_TIMEOUT_S
    for num_bits in range(1, max_bits + 1):
        remaining_ms = int((deadline - perf_counter()) * 1000)
        if remaining_ms <= 0:
            return None
        solver = z3.Solver()
        solver.set("timeout", remaining_ms)
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
        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return [
                tuple(
                    sum((1 << b) if z3.is_true(model[e]) else 0 for b, e in enumerate(bits))
                    for bits in (contype_bits[i], conaffinity_bits[i])
                )
                for i in range(n)
            ]
        if result != z3.unsat:
            # The width could not be refuted within the remaining delay, and the wider widths are the expensive ones
            # to refute, so the search stops here instead of exhausting the whole delay. Callers fall back on their
            # default masks.
            return None
    return None


# Geoms of one entity are parsed once per scene it is added to, which asks for the same assignment every time.
@lru_cache(maxsize=64)
def _synthesize_bitmasks(
    n: int, invalid_pairs: frozenset[frozenset[int]], max_bits: int
) -> tuple[tuple[int, int], ...] | None:
    """
    Return the ``(contype, conaffinity)`` assignment realizing a desired collision matrix, memoized per input.

    The constructive assignment covers every collision matrix whose geoms fall into at most ``max_bits`` groups of
    identical forbidden-partner sets, and the ascending z3 search covers the rest.

    Parameters
    ----------
    n : int
        Number of collision geoms (indices ``0..n-1``).
    invalid_pairs : frozenset[frozenset[int]]
        Pairs ``frozenset({i, j})`` that must be prevented from colliding.
    max_bits : int
        Maximum number of bits the masks may use.

    Returns
    -------
    tuple[tuple[int, int], ...] | None
        ``(contype, conaffinity)`` per geom index, or ``None`` if no assignment up to ``max_bits`` bits satisfies the
        constraints.
    """
    masks = _synthesize_bitmasks_constructively(n, invalid_pairs, max_bits)
    if masks is None:
        masks = _synthesize_bitmasks_with_z3(n, invalid_pairs, max_bits)
    return None if masks is None else tuple(masks)


register_cache_clear(_synthesize_bitmasks.cache_clear)


def solve_contype_conaffinity(
    n: int, invalid_pairs: set[frozenset[int]], max_bits: int = 31
) -> list[tuple[int, int]] | None:
    """
    Synthesize per-geom ``(contype, conaffinity)`` bitmasks realizing a desired collision matrix.

    Genesis (like MuJoCo) decides whether two geoms ``i`` and ``j`` may collide with the rule
    ``(contype[i] & conaffinity[j]) | (contype[j] & conaffinity[i]) != 0``. Given the set of pairs that must **not**
    collide, this finds bitmasks such that every excluded pair is disabled and every other pair is enabled. Used by
    both the MJCF importer (``<contact><exclude>``) and the USD importer (CollisionGroup / FilteredPairsAPI).

    Geoms forbidding exactly the same partners share a bit, one per group of such geoms, which realizes the requested
    matrix in O(n^2) time. A collision matrix whose groups do not fit in ``max_bits`` bits falls back on an ascending
    z3 search over the bit widths, which gives up after ``Z3_SEARCH_TIMEOUT_S`` seconds.

    Every geom keeps at least one bit set across its two masks: a geom whose masks are both zero would be demoted to
    a visual-only geom downstream (``contype or conaffinity`` is the collision-geom discriminator), silently
    disabling its collision against every other entity as well.

    Parameters
    ----------
    n : int
        Number of collision geoms (indices ``0..n-1``).
    invalid_pairs : set[frozenset[int]]
        Pairs ``frozenset({i, j})`` that must be prevented from colliding.
    max_bits : int
        Maximum number of bits the masks may use (default 31, keeping masks within a signed 32-bit int).

    Returns
    -------
    list[tuple[int, int]] | None
        ``(contype, conaffinity)`` per geom index, or ``None`` if no assignment up to ``max_bits`` bits satisfies the
        constraints.
    """
    masks = _synthesize_bitmasks(n, frozenset(invalid_pairs), max_bits)
    return None if masks is None else list(masks)
