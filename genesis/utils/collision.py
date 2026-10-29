"""Shared helpers for translating a desired collision matrix into contype/conaffinity bitmasks."""

from functools import lru_cache

from .cache import register_cache_clear


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

    Finding the fewest bits is NP-hard, so two deterministic polynomial constructions are tried in turn. Geoms
    excluding exactly the same partners are interchangeable, so the first one assigns one bit per group of such geoms:
    each geom takes the bit of its group as ``contype``, and as ``conaffinity`` the bits of every group it may collide
    with, its own group included. When there are more groups than ``max_bits``, the second one covers the allowed
    pairs with cliques greedily, one bit per clique set in both masks of its members. The result is memoized per input.

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
        ``(contype, conaffinity)`` per geom index, or ``None`` if neither construction fits in ``max_bits`` bits.
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

    # Each clique grows from a geom with an allowed pair left uncovered, adding the candidate that covers the most
    # uncovered pairs with the clique. That count is updated as the clique grows, keeping each addition linear in n.
    allowed_geoms = [set(range(n)) - excluded_geoms[i_g] - {i_g} for i_g in range(n)]
    uncovered_geoms = [set(geoms) for geoms in allowed_geoms]
    mask_of_geom = [0] * n
    num_cliques = 0
    for i_ga in range(n):
        while uncovered_geoms[i_ga]:
            if num_cliques == max_bits:
                return None
            clique = {i_ga}
            candidate_geoms = set(allowed_geoms[i_ga])
            gain_of_geom = dict.fromkeys(candidate_geoms, 0)
            for i_g in uncovered_geoms[i_ga]:
                gain_of_geom[i_g] = 1
            while candidate_geoms:
                i_gb = max(candidate_geoms, key=lambda i_g: (gain_of_geom[i_g], -i_g))
                clique.add(i_gb)
                candidate_geoms &= allowed_geoms[i_gb]
                for i_g in uncovered_geoms[i_gb] & candidate_geoms:
                    gain_of_geom[i_g] += 1
            for i_g in clique:
                uncovered_geoms[i_g] -= clique
                mask_of_geom[i_g] |= 1 << num_cliques
            num_cliques += 1

    # A geom allowed to collide with no other one still needs a bit, set in a 'conaffinity' that no 'contype' matches.
    if 0 in mask_of_geom and num_cliques == max_bits:
        return None
    return tuple((mask, mask) if mask else (0, 1 << num_cliques) for mask in mask_of_geom)


register_cache_clear(solve_contype_conaffinity.cache_clear)
