# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Bounding-sphere hierarchy over the contact samples of a link, built once at scene construction in the link frame.

Contact detection only needs the samples that can be within the penalty band of a collider, which for a resting or
sliding body is a small fraction of its surface. The hierarchy lets the detection kernel prune whole regions of the
sample cloud by one distance evaluation per node (the collider's signed distance at the node center, minus the node
radius, is a lower bound of the distance of every sample it bounds). Nodes are stored in depth-first order with an
escape index, so that a kernel traverses the tree with a single loop and no stack: a node that cannot contain a
contact jumps to its escape node (the next node outside its subtree), a node that can either descends (next node in
the array) or, when it is a leaf, evaluates its samples, which are contiguous in the sample arrays.
"""

import numpy as np

import genesis as gs

# Samples per leaf: small enough to prune tightly, large enough that the leaf test does not dominate.
LEAF_SIZE = 8


def build_sample_tree(pos, leaf_size=LEAF_SIZE):
    """Build the hierarchy of a point cloud.

    Returns (order, centers, radii, first, count, escape, is_leaf): `order` is the permutation that makes the samples
    of every leaf contiguous (apply it to the sample arrays before storing them), `first/count` index the permuted
    samples of each node, `escape[i]` is the depth-first index of the next node outside the subtree of node `i` (the
    number of nodes for the last one on each path).
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    order = np.arange(n, dtype=np.int64)
    centers, radii, first, count, escape, is_leaf = [], [], [], [], [], []

    def visit(lo, hi):
        i_node = len(centers)
        points = pos[order[lo:hi]]
        center = points.mean(axis=0)
        radius = float(np.sqrt(((points - center) ** 2).sum(axis=1).max()))
        centers.append(center)
        radii.append(radius)
        first.append(lo)
        count.append(hi - lo)
        escape.append(-1)
        is_leaf.append(hi - lo <= leaf_size)
        if not is_leaf[i_node]:
            axis = int(np.argmax(points.max(axis=0) - points.min(axis=0)))
            coords = points[:, axis]
            mask = coords <= coords.mean()
            # A degenerate split (all points on one side) falls back to a median split so the recursion terminates.
            if mask.all() or not mask.any():
                mask = np.zeros(hi - lo, dtype=bool)
                mask[np.argsort(coords)[: (hi - lo) // 2]] = True
            order[lo:hi] = np.concatenate([order[lo:hi][mask], order[lo:hi][~mask]])
            mid = lo + int(mask.sum())
            visit(lo, mid)
            visit(mid, hi)
        escape[i_node] = len(centers)

    if n > 0:
        visit(0, n)
    return (
        order,
        np.asarray(centers, dtype=gs.np_float).reshape((-1, 3)),
        np.asarray(radii, dtype=gs.np_float),
        np.asarray(first, dtype=gs.np_int),
        np.asarray(count, dtype=gs.np_int),
        np.asarray(escape, dtype=gs.np_int),
        np.asarray(is_leaf, dtype=gs.np_int),
    )
