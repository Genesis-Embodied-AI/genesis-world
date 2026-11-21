import numpy as np
import pytest

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import (
    BoxQuery,
    BVHLeaves,
    BVHPackedTrees,
    BVHQueryResults,
    BVHTreeState,
    build_bvh,
    build_packed_bvh,
    func_bvh_query_leaves,
    func_bvh_traverse,
    get_bvh_data,
)
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_equal


@pytest.fixture(scope="function")
def bvh(n_leaves, n_trees, boxes_layout):
    bvh_state, bvh_config = get_bvh_data(n_trees, n_leaves)
    leaves_min = np.random.rand(n_trees, n_leaves, 3).astype(gs.np_float) * 20.0
    leaves_max = leaves_min + np.random.rand(n_trees, n_leaves, 3).astype(gs.np_float)
    if boxes_layout == "flat":
        leaves_min[..., 2] = 0.0
        leaves_max[..., 2] = 1.0
    elif boxes_layout == "identical":
        leaves_min[:] = leaves_min[:, :1]
        leaves_max[:] = leaves_max[:, :1]
    bvh_state.leaves.aabbs_min.from_numpy(leaves_min)
    bvh_state.leaves.aabbs_max.from_numpy(leaves_max)
    build_bvh(bvh_state, bvh_config, eps=gs.EPS, node_fit=None)
    return bvh_state, bvh_config


@pytest.mark.required
@pytest.mark.parametrize(
    "n_leaves, n_trees, boxes_layout",
    [
        (1, 3, "random"),
        (5, 1, "random"),
        (64, 4, "random"),
        (255, 2, "random"),
        (16384, 2, "random"),
        (30, 512, "random"),
    ],
)
def test_build_tree(bvh):
    bvh_state, bvh_config = bvh
    n_trees, n_leaves = bvh_state.leaves.aabbs_min.shape
    n_internal = n_leaves - 1
    leaves_idx = qd_to_numpy(bvh_state.tree.leaves_idx)
    left = qd_to_numpy(bvh_state.tree.nodes_left)
    right = qd_to_numpy(bvh_state.tree.nodes_right)

    # The leaf nodes hold a permutation of the leaves
    for i_t in range(n_trees):
        assert_equal(np.sort(leaves_idx[i_t]), np.arange(n_leaves))
        assert (left[i_t, n_internal:] == -1).all() and (right[i_t, n_internal:] == -1).all()

    # A leaf node holds the box of its leaf and an internal node the box of its children
    leaves_min = qd_to_numpy(bvh_state.leaves.aabbs_min)
    leaves_max = qd_to_numpy(bvh_state.leaves.aabbs_max)
    nodes_min = qd_to_numpy(bvh_state.tree.nodes_min)
    nodes_max = qd_to_numpy(bvh_state.tree.nodes_max)
    for i_t in range(n_trees):
        assert_equal(nodes_min[i_t, n_internal:], leaves_min[i_t, leaves_idx[i_t]])
        assert_equal(nodes_max[i_t, n_internal:], leaves_max[i_t, leaves_idx[i_t]])
        i_left, i_right = left[i_t, :n_internal], right[i_t, :n_internal]
        assert_equal(nodes_min[i_t, :n_internal], np.minimum(nodes_min[i_t, i_left], nodes_min[i_t, i_right]))
        assert_equal(nodes_max[i_t, :n_internal], np.maximum(nodes_max[i_t, i_left], nodes_max[i_t, i_right]))


@pytest.mark.required
@pytest.mark.parametrize(
    "n_leaves, n_trees, boxes_layout",
    [
        (500, 10, "random"),
        (5, 1, "random"),
        (1, 1, "random"),
        (1, 3, "random"),
        (100, 40, "random"),
        (200, 3, "flat"),
        (200, 3, "identical"),
    ],
)
def test_query(bvh):
    bvh_state, bvh_config = bvh
    n_trees, n_leaves = bvh_state.leaves.aabbs_min.shape
    box_query = BoxQuery(BVHQueryResults(sizes=dict(query_results=n_trees * n_leaves * n_leaves, bvh_trees=n_trees)))

    @qd.kernel
    def query_leaves(leaves: BVHLeaves, tree: BVHTreeState, visitor: qd.template()):
        visitor.results.count[0] = 0
        for i_t in range(visitor.results.trees_count.shape[0]):
            visitor.results.trees_count[i_t] = 0
        func_bvh_query_leaves(leaves, tree, visitor)

    leaves_min = qd_to_numpy(bvh_state.leaves.aabbs_min)
    leaves_max = qd_to_numpy(bvh_state.leaves.aabbs_max)
    is_hit = (leaves_min[:, :, None] <= leaves_max[:, None]).all(axis=-1)
    is_hit &= (leaves_max[:, :, None] >= leaves_min[:, None]).all(axis=-1)

    # Every leaf queried against its own tree finds exactly the leaves its box meets, itself included
    query_leaves(bvh_state.leaves, bvh_state.tree, box_query)
    assert_equal(qd_to_numpy(box_query.results.trees_count), is_hit.sum(axis=(1, 2)))
    count = qd_to_numpy(box_query.results.count)[0]
    triplets = qd_to_numpy(box_query.results.triplets)[:count]
    is_found = np.zeros((n_trees, n_leaves, n_leaves), dtype=bool)
    is_found[triplets[:, 0], triplets[:, 1], triplets[:, 2]] = True
    assert_equal(is_found, is_hit)

    # Trees of unequal leaf counts packed in one set, over the first leaves of each tree, each of these leaves queried
    # against the packed tree of its own tree, find the same leaves among them
    trees_n_leaves = np.array([max(n_leaves - i_t, 1) for i_t in range(n_trees)], dtype=gs.np_int)
    trees_leaf_start = np.cumsum(trees_n_leaves) - trees_n_leaves
    trees = build_packed_bvh(
        [leaves_min[i_t, :n] for i_t, n in enumerate(trees_n_leaves)],
        [leaves_max[i_t, :n] for i_t, n in enumerate(trees_n_leaves)],
    )

    @qd.kernel
    def query_packed_leaves(
        trees_n_leaves: qd.types.ndarray(),
        trees_leaf_start: qd.types.ndarray(),
        leaves: BVHLeaves,
        trees: BVHPackedTrees,
        visitor: qd.template(),
    ):
        visitor.results.count[0] = 0
        for i_t in range(visitor.results.trees_count.shape[0]):
            visitor.results.trees_count[i_t] = 0
        for i_t, i_q in qd.ndrange(leaves.aabbs_min.shape[0], leaves.aabbs_min.shape[1]):
            if i_q < trees_n_leaves[i_t]:
                i_root = trees.trees_root[i_t]
                leaf_offset = trees.trees_leaf_offset[i_t]
                func_bvh_traverse(
                    i_t,
                    i_q,
                    i_root,
                    leaf_offset,
                    leaf_base=-trees_leaf_start[i_t],
                    query=visitor.Query(aabb_min=leaves.aabbs_min[i_t, i_q], aabb_max=leaves.aabbs_max[i_t, i_q]),
                    bound=0.0,
                    tree=trees.tree,
                    visitor=visitor,
                )

    is_packed = np.arange(n_leaves) < trees_n_leaves[:, None]
    is_hit_packed = is_hit & is_packed[:, :, None] & is_packed[:, None]
    query_packed_leaves(trees_n_leaves, trees_leaf_start, bvh_state.leaves, trees, box_query)
    assert_equal(qd_to_numpy(box_query.results.trees_count), is_hit_packed.sum(axis=(1, 2)))
    count = qd_to_numpy(box_query.results.count)[0]
    triplets = qd_to_numpy(box_query.results.triplets)[:count]
    is_found[:] = False
    is_found[triplets[:, 0], triplets[:, 1], triplets[:, 2]] = True
    assert_equal(is_found, is_hit_packed)

    # A result list holding one pair per tree keeps the first pair of every tree, a genuine one, and reports the
    # overflow of each tree with more.
    short_box_query = BoxQuery(BVHQueryResults(sizes=dict(query_results=n_trees, bvh_trees=n_trees)))
    query_leaves(bvh_state.leaves, bvh_state.tree, short_box_query)
    assert_equal(qd_to_numpy(short_box_query.results.trees_count), is_hit.sum(axis=(1, 2)))
    triplets = qd_to_numpy(short_box_query.results.triplets)
    assert_equal(np.sort(triplets[:, 0]), np.arange(n_trees))
    assert is_hit[triplets[:, 0], triplets[:, 1], triplets[:, 2]].all()
