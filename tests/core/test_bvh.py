import numpy as np
import pytest

import genesis as gs
from genesis.engine.bvh import AABB, LBVH
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose, assert_equal


@pytest.fixture(scope="function")
def lbvh(n_aabbs, n_batches):
    """Fixture for a LBVH tree"""
    aabb = AABB(n_batches=n_batches, n_aabbs=n_aabbs)
    aabbs_min = np.random.rand(n_batches, n_aabbs, 3).astype(gs.np_float) * 20.0
    aabbs_max = aabbs_min + np.random.rand(n_batches, n_aabbs, 3).astype(gs.np_float)

    aabb.aabbs.min.from_numpy(aabbs_min)
    aabb.aabbs.max.from_numpy(aabbs_max)

    lbvh = LBVH(aabb, max_n_query_result_per_aabb=32)
    lbvh.build()

    return lbvh


@pytest.mark.required
@pytest.mark.parametrize("n_aabbs, n_batches", [(1, 3), (5, 1), (64, 4), (255, 2), (16384, 2)])
def test_morton_code_normalization(lbvh, tol):
    aabbs_min = qd_to_numpy(lbvh.aabbs.min)
    aabbs_max = qd_to_numpy(lbvh.aabbs.max)
    scene_min = aabbs_min.min(axis=1)
    scene_max = aabbs_max.max(axis=1)
    extent = scene_max - scene_min

    # The reduction selects an input value, so it must land on it exactly whichever fold order the backend picks.
    assert_equal(qd_to_numpy(lbvh.aabb_min), scene_min)
    assert_equal(qd_to_numpy(lbvh.aabb_max), scene_max)
    assert_allclose(qd_to_numpy(lbvh.scale), np.where(extent > gs.EPS, 1.0 / extent, 1.0), tol=tol)

    # Two centers may share a cell of the 1024^3 grid the codes quantize to, so equal codes are legitimate.
    morton_codes = qd_to_numpy(lbvh.morton_codes)[..., 0]
    assert (morton_codes[:, 1:] >= morton_codes[:, :-1]).all(), "Morton codes are not sorted"


@pytest.mark.required
def test_expand_bits():
    # expand_bits turns a 10-bit integer into a 30-bit one by inserting two zeros before each bit.
    import quadrants as qd

    @qd.kernel
    def expand_bits(lbvh: qd.template(), x: qd.template(), expanded_x: qd.template()):
        n_x = x.shape[0]
        for i in range(n_x):
            expanded_x[i] = lbvh.expand_bits(x[i])

    # random integer
    x_np = np.random.randint(0, 1024, (10,), dtype=np.uint32)
    x_qd = qd.field(qd.uint32, shape=x_np.shape)
    x_qd.from_numpy(x_np)
    expanded_x_qd = qd.field(qd.uint32, shape=x_np.shape)
    # expand bits
    n_aabbs = 10
    n_batches = 1
    aabb = AABB(n_aabbs=n_aabbs, n_batches=n_batches)
    lbvh = LBVH(aabb)
    expand_bits(lbvh, x_qd, expanded_x_qd)
    expanded_x_np = expanded_x_qd.to_numpy()
    for i in range(x_np.shape[0]):
        str_x = f"{x_np[i]:010b}"
        str_expanded_x = f"{expanded_x_np[i]:030b}"
        # check that the expanded bits are correct
        assert str_expanded_x == "".join(f"00{bit}" for bit in str_x), (
            f"Expected {str_expanded_x}, got {''.join(f'00{bit}' for bit in str_x)}"
        )


@pytest.mark.required
@pytest.mark.parametrize("n_aabbs, n_batches", [(500, 10), (5, 1), (1, 1), (1, 3)])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_build_tree(lbvh):
    nodes = lbvh.nodes.to_numpy()
    n_aabbs = lbvh.n_aabbs
    n_batches = lbvh.n_batches

    # Check parent-child relationships
    for j in range(n_batches):
        for i in range(n_aabbs * 2 - 1):
            parent = nodes["parent"][j, i]
            left = nodes["left"][j, i]
            right = nodes["right"][j, i]

            # Check that the parent is correct
            if i == 0:  # root node has no parent
                assert parent == -1

            else:
                assert nodes["left"][j, parent] == i or nodes["right"][j, parent] == i, (
                    f"Node {i} in batch {j} has incorrect parent: {parent}"
                )

            # Check that left and right children are correct
            if left != -1:
                assert nodes["parent"][j, left] == i, (
                    f"Left child {left} of node {i} in batch {j} has incorrect parent: {nodes['parent'][j, left]}, expected {i}"
                )
            if right != -1:
                assert nodes["parent"][j, right] == i, (
                    f"Right child {right} of node {i} in batch {j} has incorrect parent: {nodes['parent'][j, right]}, expected {i}"
                )

            if left != -1 and right != -1:
                # Check that the AABBs of the children are within the AABB of the parent
                parent_min = nodes["bound"]["min"][j, i]
                parent_max = nodes["bound"]["max"][j, i]
                left_min = nodes["bound"]["min"][j, left]
                left_max = nodes["bound"]["max"][j, left]
                right_min = nodes["bound"]["min"][j, right]
                right_max = nodes["bound"]["max"][j, right]

                parent_min_expected = np.minimum(left_min, right_min)
                parent_max_expected = np.maximum(left_max, right_max)
                assert_allclose(parent_min, parent_min_expected, atol=1e-6, rtol=1e-5)
                assert_allclose(parent_max, parent_max_expected, atol=1e-6, rtol=1e-5)


@pytest.mark.required
@pytest.mark.parametrize("n_aabbs, n_batches", [(500, 10), (5, 1), (1, 1), (1, 3)])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_query(lbvh):
    import quadrants as qd

    @qd.kernel
    def query_kernel(lbvh: qd.template(), aabbs: qd.template()):
        lbvh.query(aabbs)

    aabbs = lbvh.aabbs

    # Query the tree
    query_kernel(lbvh, aabbs)

    query_result_count = lbvh.query_result_count.to_numpy()
    if query_result_count > lbvh.max_query_results:
        raise ValueError(f"Query result count {query_result_count} exceeds max_query_results {lbvh.max_query_results}")
    query_result = lbvh.query_result.to_numpy()

    n_aabbs = lbvh.n_aabbs
    n_batches = lbvh.n_batches

    # Check that the query results are correct
    intersect = np.zeros((n_batches, n_aabbs, n_aabbs), dtype=bool)
    for j in range(query_result_count):
        i_b, i_a, j_a = query_result[j]
        intersect[i_b, i_a, j_a] = True

    for i_b in range(n_batches):
        for i_a in range(n_aabbs):
            for j_a in range(n_aabbs):
                if i_a == j_a:
                    assert intersect[i_b, i_a, j_a] == True, f"AABB {i_a} should intersect with itself"
                else:
                    assert intersect[i_b, i_a, j_a] == intersect[i_b, j_a, i_a], (
                        f"AABBs {i_a} and {j_a} should have the same intersection result"
                    )
