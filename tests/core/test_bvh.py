import numpy as np
import pytest

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.bvh import build_bvh, func_bvh_query_leaves, func_no_filter, get_bvh_data
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
    build_bvh(bvh_state, bvh_config, eps=gs.EPS)
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
    query_state = array_class.BVHQueryState(
        leaves=bvh_state.leaves,
        tree=bvh_state.tree,
        results=array_class.get_bvh_query_results(n_trees * n_leaves * n_leaves),
    )

    @qd.kernel
    def query_leaves(query_state: array_class.BVHQueryState) -> bool:
        query_state.results.count[0] = 0
        func_bvh_query_leaves(query_state, func_no_filter, 0)
        return query_state.results.count[0] > query_state.results.triplets.shape[0]

    leaves_min = qd_to_numpy(bvh_state.leaves.aabbs_min)
    leaves_max = qd_to_numpy(bvh_state.leaves.aabbs_max)
    is_hit = (leaves_min[:, :, None] <= leaves_max[:, None]).all(axis=-1)
    is_hit &= (leaves_max[:, :, None] >= leaves_min[:, None]).all(axis=-1)

    # Every leaf queried against its own tree finds exactly the leaves its box meets, itself included
    assert not query_leaves(query_state)
    count = qd_to_numpy(query_state.results.count)[0]
    triplets = qd_to_numpy(query_state.results.triplets)[:count]
    is_found = np.zeros((n_trees, n_leaves, n_leaves), dtype=bool)
    is_found[triplets[:, 0], triplets[:, 1], triplets[:, 2]] = True
    assert_equal(is_found, is_hit)

    # A result list holding one pair reports the overflow exactly when there are more, and the pair it holds is genuine
    short_query_state = array_class.BVHQueryState(
        leaves=bvh_state.leaves, tree=bvh_state.tree, results=array_class.get_bvh_query_results(1)
    )
    assert query_leaves(short_query_state) == (is_hit.sum() > 1)
    assert is_hit[tuple(qd_to_numpy(short_query_state.results.triplets)[0])]
