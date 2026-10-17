# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Bounding-box hierarchy over the collider tetrahedra of the deformable solids.

A sample point in contact with a deformable solid lies inside one of its deformed tetrahedra, so the contact query is a
point location. The hierarchy is built once over the rest shape (the topology never changes) and its boxes are refit to
the deformed vertices at every assembly, the way mochi refits the hierarchy of its tetrahedral maps; a query descends
from the root and visits only the nodes whose box contains the point, a few dozen node tests for thousands of
tetrahedra, where a uniform hash over a fine mesh walks chains of hundreds of entries near the body.

Nodes are stored in depth-first order with an escape index, so that a kernel traverses the tree with a single loop and
no stack (see sample_tree.py): the children of an inner node `i` are `i + 1` and `escape[i + 1]`. The refit runs level
by level from the deepest nodes to the root (a level's boxes only depend on the level below), so the nodes are also
listed by depth.
"""

from typing import NamedTuple

import numpy as np

import genesis as gs

# Tetrahedra per leaf: one keeps the box test as tight as the tetrahedron's own bounds.
LEAF_SIZE = 1


class TetTree(NamedTuple):
    """Bounding-box hierarchy over items, nodes in depth-first order (see `build_tet_tree`)."""

    order: np.ndarray
    first: np.ndarray
    count: np.ndarray
    escape: np.ndarray
    is_leaf: np.ndarray
    level_nodes: np.ndarray
    level_start: np.ndarray
    n_levels: int


def build_tet_tree(aabb_min, aabb_max, leaf_size=LEAF_SIZE):
    """Build the hierarchy over items given by their bounds.

    Returns a `TetTree`: `order` (the permutation that makes the items of every leaf contiguous), per-node `first`,
    `count`, `escape`, `is_leaf`, and `level_nodes` / `level_start` listing the nodes from the deepest level up
    (`level_start` has one more entry than there are levels).
    """
    aabb_min = np.asarray(aabb_min, dtype=np.float64).reshape((-1, 3))
    aabb_max = np.asarray(aabb_max, dtype=np.float64).reshape((-1, 3))
    n = len(aabb_min)
    centers = 0.5 * (aabb_min + aabb_max)
    order = np.arange(n, dtype=np.int64)
    first, count, escape, is_leaf, depth = [], [], [], [], []

    def visit(lo, hi, d):
        i_node = len(first)
        first.append(lo)
        count.append(hi - lo)
        escape.append(-1)
        is_leaf.append(hi - lo <= leaf_size)
        depth.append(d)
        if not is_leaf[i_node]:
            items = order[lo:hi]
            extent = aabb_max[items].max(axis=0) - aabb_min[items].min(axis=0)
            axis = int(np.argmax(extent))
            coords = centers[items, axis]
            mask = coords <= coords.mean()
            # A degenerate split (all items on one side) falls back to a median split so the recursion terminates.
            if mask.all() or not mask.any():
                mask = np.zeros(hi - lo, dtype=bool)
                mask[np.argsort(coords, kind="stable")[: (hi - lo) // 2]] = True
            order[lo:hi] = np.concatenate([items[mask], items[~mask]])
            mid = lo + int(mask.sum())
            visit(lo, mid, d + 1)
            visit(mid, hi, d + 1)
        escape[i_node] = len(first)

    if n > 0:
        visit(0, n, 0)
    depth = np.asarray(depth, dtype=np.int64)
    n_levels = int(depth.max()) + 1 if n > 0 else 0
    # deepest level first
    level_nodes = np.argsort(-depth, kind="stable")
    level_start = np.zeros((n_levels + 1,), dtype=np.int64)
    for i_level in range(n_levels):
        level_start[i_level + 1] = level_start[i_level] + int((depth == n_levels - 1 - i_level).sum())
    return TetTree(
        order=order,
        first=np.asarray(first, dtype=gs.np_int),
        count=np.asarray(count, dtype=gs.np_int),
        escape=np.asarray(escape, dtype=gs.np_int),
        is_leaf=np.asarray(is_leaf, dtype=gs.np_int),
        level_nodes=level_nodes.astype(gs.np_int),
        level_start=level_start.astype(gs.np_int),
        n_levels=n_levels,
    )
