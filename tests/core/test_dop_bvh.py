import numpy as np
import pytest

import quadrants as qd

import genesis as gs
from genesis.engine.ipc.dop_bvh import (
    DOPBVHConfig,
    DOPBVHHits,
    DOPBVHLeaves,
    DOPBVHQueries,
    func_dop_add_point,
    func_dop_empty,
    func_dop_expand,
    get_dop_bvh_data,
    kernel_build_dop_bvh,
    kernel_query_dop_bvh,
)
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_equal


@pytest.fixture(scope="function")
def dop_bvh(n_leaves, n_trees):
    @qd.kernel
    def kernel_set_leaves(
        points: qd.types.ndarray(), radii: qd.types.ndarray(), tags: qd.types.ndarray(), leaves: DOPBVHLeaves
    ):
        for i_a, i_t in qd.ndrange(leaves.lower.shape[0], leaves.lower.shape[1]):
            lower, upper = func_dop_empty()
            for k in qd.static(range(3)):
                point = qd.Vector([points[i_t, i_a, k, j] for j in qd.static(range(3))])
                lower, upper = func_dop_add_point(lower, upper, point)
            lower, upper = func_dop_expand(lower, upper, radii[i_t, i_a])
            leaves.lower[i_a, i_t] = lower
            leaves.upper[i_a, i_t] = upper
            leaves.tag[i_a, i_t] = tags[i_t, i_a]

    state, config = get_dop_bvh_data(n_trees, n_leaves)
    points = np.random.rand(n_trees, n_leaves, 3, 3) * 0.2 + np.random.rand(n_trees, n_leaves, 1, 3) * 5.0
    points = points.astype(gs.np_float)
    radii = (np.random.rand(n_trees, n_leaves) * 0.1).astype(gs.np_float)
    tags = np.random.randint(-1, 3, size=(n_trees, n_leaves))
    kernel_set_leaves(points, radii, tags.astype(gs.np_int), state.leaves)
    kernel_build_dop_bvh(state, config)
    return state, config, points, radii


@pytest.mark.required
@pytest.mark.parametrize("n_leaves, n_trees", [(1, 2), (7, 1), (300, 2)])
def test_build_tree(dop_bvh):
    state, config, points, radii = dop_bvh
    n_leaves = state.leaves.lower.shape[0]
    n_internal = n_leaves - 1
    leaves_lower = qd_to_numpy(state.leaves.lower, transpose=True)
    leaves_upper = qd_to_numpy(state.leaves.upper, transpose=True)
    leaves_tag = qd_to_numpy(state.leaves.tag, transpose=True)
    leaves_idx = qd_to_numpy(state.tree.leaves_idx, transpose=True)
    left = qd_to_numpy(state.tree.nodes_left, transpose=True)
    right = qd_to_numpy(state.tree.nodes_right, transpose=True)
    nodes_lower = qd_to_numpy(state.tree.nodes_lower, transpose=True)
    nodes_upper = qd_to_numpy(state.tree.nodes_upper, transpose=True)
    nodes_tag = qd_to_numpy(state.tree.nodes_tag, transpose=True)

    # A leaf covers its points grown by its radius along every axis, the diagonals being sqrt(3) long
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]], dtype=np.float64)
    projections = points.astype(np.float64) @ axes.T
    margins = radii[..., None].astype(np.float64) * np.linalg.norm(axes, axis=-1)
    # The growth by the radius rounds to the nearest 32-bit float, which may land half a unit inside
    slack = 2.0 * np.spacing(np.abs(leaves_lower))
    assert (leaves_lower <= projections.min(axis=2) - margins + slack).all()
    slack = 2.0 * np.spacing(np.abs(leaves_upper))
    assert (leaves_upper >= projections.max(axis=2) + margins - slack).all()

    for i_t in range(leaves_lower.shape[0]):
        # The leaf nodes hold a permutation of the leaves, with their bounds and tags
        assert_equal(np.sort(leaves_idx[i_t]), np.arange(n_leaves))
        assert (left[i_t, n_internal:] == -1).all() and (right[i_t, n_internal:] == -1).all()
        assert_equal(nodes_lower[i_t, n_internal:], leaves_lower[i_t, leaves_idx[i_t]])
        assert_equal(nodes_upper[i_t, n_internal:], leaves_upper[i_t, leaves_idx[i_t]])
        assert_equal(nodes_tag[i_t, n_internal:], leaves_tag[i_t, leaves_idx[i_t]])
        # An internal node holds the bounds of its children, and their tag when they share it, -1 otherwise
        i_left, i_right = left[i_t, :n_internal], right[i_t, :n_internal]
        assert_equal(nodes_lower[i_t, :n_internal], np.minimum(nodes_lower[i_t, i_left], nodes_lower[i_t, i_right]))
        assert_equal(nodes_upper[i_t, :n_internal], np.maximum(nodes_upper[i_t, i_left], nodes_upper[i_t, i_right]))
        tag_left, tag_right = nodes_tag[i_t, i_left], nodes_tag[i_t, i_right]
        assert_equal(nodes_tag[i_t, :n_internal], np.where(tag_left == tag_right, tag_left, -1))


@pytest.mark.required
@pytest.mark.parametrize("n_leaves, n_trees", [(1, 1), (5, 2), (300, 2)])
@pytest.mark.parametrize("is_block_query", [False, True])
def test_query(dop_bvh, is_block_query):
    state, config, _, _ = dop_bvh
    if is_block_query and gs.backend not in (gs.cuda, gs.amdgpu):
        pytest.skip("The block schedule of a query runs on the backends whose lanes vote.")
    config = DOPBVHConfig(
        **{key: vars(config)[key] for key in DOPBVHConfig.__annotations__} | {"is_block_query": is_block_query}
    )
    n_leaves, n_trees = state.leaves.lower.shape[:2]
    leaves_lower = qd_to_numpy(state.leaves.lower, transpose=True)
    leaves_upper = qd_to_numpy(state.leaves.upper, transpose=True)
    leaves_tag = qd_to_numpy(state.leaves.tag, transpose=True)

    # Every leaf queries its own tree within a random gap and with a random tag, the even queries alone running
    gaps = np.random.rand(n_trees, n_leaves) * 0.3
    query_tags = np.random.randint(-1, 3, size=(n_trees, n_leaves))
    queries = DOPBVHQueries(sizes=dict(dop_queries=n_leaves, dop_trees=n_trees))
    queries.lower.from_numpy(np.ascontiguousarray(leaves_lower.swapaxes(0, 1)))
    queries.upper.from_numpy(np.ascontiguousarray(leaves_upper.swapaxes(0, 1)))
    queries.gap.from_numpy(np.ascontiguousarray(gaps.T).astype(gs.np_float))
    queries.tag.from_numpy(np.ascontiguousarray(query_tags.T).astype(gs.np_int))
    is_active = np.arange(n_leaves) % 2 == 0
    active = np.flatnonzero(np.broadcast_to(is_active, (n_trees, n_leaves)).reshape(-1))
    active_buffer = np.zeros(n_trees * n_leaves, dtype=gs.np_int)
    active_buffer[: len(active)] = active
    queries.active.from_numpy(active_buffer)
    queries.n_active.from_numpy(np.array(len(active), dtype=gs.np_int))

    # The gaps rounded up to 32-bit floats along the coordinate axes and along the diagonals, as the query rounds them
    gaps_axes = np.stack((gaps, gaps * np.sqrt(3.0)), axis=-1)
    gaps_f32 = gaps_axes.astype(np.float32)
    gaps_f32 = np.where(gaps_f32 < gaps_axes, np.nextafter(gaps_f32, np.float32(np.inf)), gaps_f32)
    gaps_f32 = gaps_f32[..., [0, 0, 0, 1, 1, 1, 1]]
    is_apart = (leaves_lower[:, None] - leaves_upper[:, :, None] >= gaps_f32[:, :, None]).any(axis=-1)
    is_apart |= (leaves_lower[:, :, None] - leaves_upper[:, None] >= gaps_f32[:, :, None]).any(axis=-1)
    is_culled = (leaves_tag[:, None] >= 0) & (leaves_tag[:, None] == query_tags[:, :, None])
    is_hit = ~is_apart & ~is_culled & is_active[:, None]
    n_hits = int(is_hit.sum())

    # A list too short for every hit keeps genuine hits and counts them all, and a list grown to that count holds them
    for capacity in (max(n_hits // 2, 1), n_hits):
        hits = DOPBVHHits(sizes=dict(dop_hits=max(capacity, 1)))
        kernel_query_dop_bvh(queries, state.tree, hits, config)
        assert qd_to_numpy(hits.count) == n_hits
        found = qd_to_numpy(hits.hits)[: min(n_hits, capacity)]
        assert is_hit[found[:, 0], found[:, 1], found[:, 2]].all()
    is_found = np.zeros_like(is_hit)
    is_found[found[:, 0], found[:, 1], found[:, 2]] = True
    assert_equal(is_found, is_hit)
