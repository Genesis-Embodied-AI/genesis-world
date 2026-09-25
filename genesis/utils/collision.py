"""Shared helpers for translating a desired collision matrix into contype/conaffinity bitmasks."""

import z3


def solve_contype_conaffinity(
    n: int, invalid_pairs: set[frozenset[int]], max_bits: int = 31
) -> list[tuple[int, int]] | None:
    """
    Synthesize per-geom ``(contype, conaffinity)`` bitmasks realizing a desired collision matrix.

    Genesis (like MuJoCo) decides whether two geoms ``i`` and ``j`` may collide with the rule
    ``(contype[i] & conaffinity[j]) | (contype[j] & conaffinity[i]) != 0``. Given the set of pairs
    that must **not** collide, this finds bitmasks (using as few bits as possible, hence the ascending
    ``K``) such that every excluded pair is disabled and every other pair is enabled. Used by both the
    MJCF importer (``<contact><exclude>``) and the USD importer (CollisionGroup / FilteredPairsAPI).

    Every geom is constrained to keep at least one bit set across its two masks: a geom whose masks
    are both zero would be demoted to a visual-only geom downstream (``contype or conaffinity`` is the
    collision-geom discriminator), silently disabling its collision against every other entity as well.

    Parameters
    ----------
    n : int
        Number of collision geoms (indices ``0..n-1``).
    invalid_pairs : set[frozenset[int]]
        Pairs ``frozenset({i, j})`` that must be prevented from colliding.
    max_bits : int
        Maximum number of bits to try (default 31, keeping masks within a signed 32-bit int).

    Returns
    -------
    list[tuple[int, int]] | None
        ``(contype, conaffinity)`` per geom index, or ``None`` if no assignment up to ``max_bits``
        bits satisfies the constraints.
    """
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
        if solver.check() == z3.sat:
            model = solver.model()
            return [
                tuple(
                    sum((1 << b) if z3.is_true(model[e]) else 0 for b, e in enumerate(bits))
                    for bits in (contype_bits[i], conaffinity_bits[i])
                )
                for i in range(n)
            ]
    return None
