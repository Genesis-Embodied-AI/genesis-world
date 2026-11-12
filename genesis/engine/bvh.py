"""Linear bounding volume hierarchies (LBVH) over sets of axis-aligned boxes: build and box queries.

A set holds one tree per batch over the same number of leaves. The caller writes the box of each leaf into the state,
builds the set, and traverses the trees it gets back (see BVHTreeState). The build follows
Karras (2012, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees").

A leaf carries two indices: i_a names it by its box, in the order the caller wrote them, and i_leaf by its slot in the
sorted order the tree is laid out in, the one leaves_idx maps back to i_a.
"""

import math
from enum import IntEnum
from typing import NamedTuple

import numpy as np

import quadrants as qd
from quadrants.algorithms import sort as radix_sort
from quadrants.algorithms import sort_scratch_slots

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.data import Config, Data, DataKind, Size, field
from genesis.utils.misc import get_gpu_core_count, qd_to_numpy


class BVH_SORT_KIND(IntEnum):
    """The sort ordering the leaf keys of a set of BVH trees."""

    # The device-wide radix sort of quadrants over every key of the set (GPU, large trees)
    DEVICE_RADIX = 0
    # The rank of each leaf among the keys of its tree (small trees)
    RANK = 1
    # A radix sort per tree, its chunks in parallel (CPU, large trees)
    PER_TREE_RADIX = 2


class BVH_FIT_KIND(IntEnum):
    """The bottom-up fit of the node boxes of a set of BVH trees."""

    # One thread per leaf walking to the root, the arrival counters sequentially consistent (CPU)
    LEAF_WALK = 0
    # The leaf walk with a device fence on each side of the relaxed arrival atomic (GPU)
    FENCED_LEAF_WALK = 1
    # One block per tree sweeping the nodes level by level, every barrier uniform (Metal, see quadrants#925)
    BLOCK_SWEEP = 2


class BVHConfig(Config):
    """The compile-time shape of a set of linear bounding volume hierarchies (LBVH).

    A leaf key packs the tree, the morton code and the leaf index in that order, so one flat sort orders every tree.
    """

    # The bits of a morton code, three interleaved coordinates of the same width
    morton_bits: int
    # The bits a sort must order, a multiple of 16 up to 64
    end_bit: int
    # Scan depth of the device-wide radix sort (see quadrants.algorithms.sort), 1 to 8
    log256_max_n: int
    # BVH_SORT_KIND
    sort_kind: int
    # BVH_FIT_KIND
    fit_kind: int
    # SORT_MAX_CHUNKS capped by the CPU threads of quadrants
    sort_max_chunks: int


# Sizes of a set of linear bounding volume hierarchies (LBVH): its trees, the leaves of each tree, the slots of its
# device-wide radix sort, the rows of chunk histograms of its host radix sort and the lanes of its extent reduction.
BVH_TREES = Size("bvh_trees")
BVH_LEAVES = Size("bvh_leaves")
# The nodes of each tree, 2 * n_leaves - 1 for a set and the sum of those of its trees for packed trees
BVH_NODES = Size("bvh_nodes")
# The trees packed in one row (see BVHPackedTrees), and the instances of packed trees (see PackedInstances)
BVH_PACKED_TREES = Size("bvh_packed_trees")
BVH_INSTANCES = Size("bvh_instances")
SORT_SLOTS = Size("sort_slots")
SORT_CHUNKS = Size("sort_chunks")
EXTENT_LANES = Size("extent_lanes")
# The (tree, leaf, query) triplets a box query of a BVH holds
QUERY_RESULTS = Size("query_results")
# The scratch of each sort and fit (see BVH_SORT_KIND and BVH_FIT_KIND)
DEVICE_RADIX_SORT = "device_radix_sort"
PER_TREE_RADIX_SORT = "per_tree_radix_sort"
BLOCK_SWEEP_FIT = "block_sweep_fit"


class BVHTreeState(Data, kind=DataKind.SCRATCH, shape=(BVH_TREES, BVH_NODES)):
    """What a traversal reads of a set of linear bounding volume hierarchies (LBVH).

    A tree of row i_t spans the nodes from its root, its internal nodes first and then its leaves, the leaf node i_n
    holding the leaf leaves_idx[i_t, i_n - leaf_offset]. A tree of a set roots at node 0 with a leaf offset of
    n_leaves - 1, and a packed tree at its own root and offset (see BVHPackedTrees). A leaf node holds -1 in nodes_left.
    """

    nodes_left: qd.Tensor = field(int)
    nodes_right: qd.Tensor = field(int)
    nodes_min: qd.Tensor = field((float, 3))
    nodes_max: qd.Tensor = field((float, 3))
    leaves_idx: qd.Tensor = field(int, shape=(BVH_TREES, BVH_LEAVES))


class BVHLeaves(Data, kind=DataKind.SCRATCH, shape=(BVH_TREES, BVH_LEAVES)):
    """The boxes of the leaves of a set of linear bounding volume hierarchies (LBVH), written before a build."""

    aabbs_min: qd.Tensor = field((float, 3))
    aabbs_max: qd.Tensor = field((float, 3))


class BVHState(Data, kind=DataKind.SCRATCH):
    """The linear bounding volume hierarchies (LBVH) of a set of trees, built by func_build_bvh."""

    tree: BVHTreeState
    leaves: BVHLeaves
    leaves_keys: qd.Tensor = field(qd.u64, shape=(BVH_TREES * BVH_LEAVES,))
    keys_scratch: qd.Tensor = field(qd.u64, shape=(BVH_TREES * BVH_LEAVES,))
    sort_scratch: qd.Tensor = field(qd.u32, shape=(SORT_SLOTS,), labels=DEVICE_RADIX_SORT)
    sort_hist: qd.Tensor = field(int, shape=(SORT_CHUNKS, 256), labels=PER_TREE_RADIX_SORT)
    n_keys: qd.Tensor = field(int, shape=())
    nodes_parent: qd.Tensor = field(int, shape=(BVH_TREES, BVH_NODES))
    nodes_fitted: qd.Tensor = field(int, shape=(BVH_TREES, BVH_LEAVES - 1))
    fit_frontier: qd.Tensor = field(int, shape=(BVH_TREES, 2, BVH_LEAVES - 1), labels=BLOCK_SWEEP_FIT)
    lanes_min: qd.Tensor = field((float, 3), shape=(BVH_TREES, EXTENT_LANES))
    lanes_max: qd.Tensor = field((float, 3), shape=(BVH_TREES, EXTENT_LANES))
    trees_min: qd.Tensor = field((float, 3), shape=(BVH_TREES,))
    trees_max: qd.Tensor = field((float, 3), shape=(BVH_TREES,))


class BVHQueryResults(Data, kind=DataKind.SCRATCH):
    """The (tree, leaf, query) triplets a box query of a BVH collects, their count, and the count of each tree.

    Each tree holds an equal share of the triplets, and its count grows above that share on overflow (see BoxQuery).
    """

    triplets: qd.Tensor = field((int, 3), shape=(QUERY_RESULTS,))
    count: qd.Tensor = field(int, shape=(1,))
    trees_count: qd.Tensor = field(int, shape=(BVH_TREES,))


class BVHPackedTrees(Data, kind=DataKind.CONSTANT):
    """Trees over different numbers of leaves, built once and packed in the single row of a tree set."""

    tree: BVHTreeState
    # Root node of each tree
    trees_root: qd.Tensor = field(int, shape=(BVH_PACKED_TREES,))
    # Offset from a leaf node of each tree to its slot in leaves_idx
    trees_leaf_offset: qd.Tensor = field(int, shape=(BVH_PACKED_TREES,))


# A subclass rather than a member: a struct is frozen once built, so it cannot take trees built apart at their exact
# size (see build_packed_bvh).
class PackedInstances(BVHPackedTrees):
    """Packed trees and their instances, each tied to the tree of its elements, -1 for none, the instances of one mesh
    sharing it."""

    instances_tree_idx: qd.Tensor = field(int, shape=(BVH_INSTANCES,))
    # Offset from a leaf of the tree of each instance to the element of the instance
    instances_leaf_base: qd.Tensor = field(int, shape=(BVH_INSTANCES,))


# Bits of a morton code: three interleaved 10-bit coordinates.
MORTON_BITS = 30
# The rank sort, quadratic per tree, beats the radix sort up to a leaf count. On the CPU the trees sort serially, so
# the bound holds at every tree count. On the GPU the passes of every tree run together and the crossover sits near
# one leaf per core for one tree, shrinking with the cube root of the tree count: n_trees * n_leaves**3 stays under
# the cube of this fraction of the core count, set under the crossover of every device measured.
RANK_SORT_MAX_LEAVES_CPU = 256
RANK_SORT_LEAVES_PER_CORE = 0.75
# Lanes reducing the extent of one tree, each over a strided slice of its leaves, folded by one thread per tree.
TREE_EXTENT_LANES = 64
# Most chunks of the leaves of one tree the host radix sort counts and scatters in parallel (see func_build_bvh). The
# chunks bring parallelism to a build of few trees, a build of many trees being parallel over its trees already, and
# every chunk costs a histogram of 256 bins per pass.
SORT_MAX_CHUNKS = 64


class BVHData(NamedTuple):
    """A set of linear bounding volume hierarchies (LBVH): its state and the static config of its kernels."""

    state: BVHState
    config: BVHConfig


def is_rank_sort(n_trees_built: int, n_leaves: int) -> bool:
    """Return whether a build of BVH trees sorts its keys by rank.

    The build covers n_trees_built trees over n_leaves leaves each (see RANK_SORT_LEAVES_PER_CORE for the bound).
    """
    if gs.backend == gs.cpu:
        return n_leaves <= RANK_SORT_MAX_LEAVES_CPU
    max_leaves = RANK_SORT_LEAVES_PER_CORE * get_gpu_core_count()
    return n_trees_built * n_leaves**3 <= max_leaves**3


def get_bvh_config(n_trees: int, n_leaves: int, n_trees_built: int | None = None) -> BVHConfig:
    """Return the static config of a build of the first trees of a set of BVH trees.

    The set holds n_trees trees over n_leaves leaves each, and the build covers its first n_trees_built trees. If None,
    n_trees_built resolves to n_trees. Defaults to None. The bits of the keys and the scan depth of the radix sort cover
    the whole set, while the sort kind follows the trees built, so the builds of one set differ by their sort alone.
    """
    if n_trees_built is None:
        n_trees_built = n_trees
    n_keys = n_trees * n_leaves
    leaf_bits = max(1, (n_leaves - 1).bit_length())
    tree_bits = (n_trees - 1).bit_length()
    key_bits = tree_bits + MORTON_BITS + leaf_bits
    # The key of a leaf packs the tree, the morton code and the leaf in 64 bits
    if key_bits > 64:
        gs.raise_exception(f"A BVH set of {n_trees} trees over {n_leaves} leaves needs {key_bits} key bits, above 64.")
    # The radix sort orders a positive even multiple of 8 bits
    end_bit = min(64, 16 * math.ceil(key_bits / 16))
    log256_max_n = 1
    while 256**log256_max_n < n_keys:
        log256_max_n += 1
    # Outside performance mode the radix sort serves every set, with the widest keys and a scan depth reaching any key
    # count a device holds. The rank sort is still chosen per build, its single pass being much faster on small trees.
    if gs.use_ndarray:
        end_bit = 64
        log256_max_n = max(log256_max_n, 4)
    if is_rank_sort(n_trees_built, n_leaves):
        sort_kind = BVH_SORT_KIND.RANK
    elif gs.backend == gs.cpu:
        sort_kind = BVH_SORT_KIND.PER_TREE_RADIX
    else:
        sort_kind = BVH_SORT_KIND.DEVICE_RADIX
    # FIXME: quadrants#925 - Metal lowers a fence to a control barrier, so the leaf walk, whose fences sit in divergent
    # flow, reads stale boxes there. One block per tree sweeps the nodes level by level instead, every barrier uniform.
    if gs.backend == gs.cpu:
        fit_kind = BVH_FIT_KIND.LEAF_WALK
    elif gs.backend == gs.metal:
        fit_kind = BVH_FIT_KIND.BLOCK_SWEEP
    else:
        fit_kind = BVH_FIT_KIND.FENCED_LEAF_WALK
    return BVHConfig(
        morton_bits=MORTON_BITS,
        end_bit=end_bit,
        log256_max_n=log256_max_n,
        sort_kind=sort_kind,
        fit_kind=fit_kind,
        sort_max_chunks=min(SORT_MAX_CHUNKS, qd.cfg.cpu_max_num_threads),
    )


def get_bvh_data(n_trees: int, n_leaves: int, is_active: bool = True) -> BVHData:
    """Return a set of trees, one per batch, over the same number of leaves each, and the config building all of them.

    A set switched off holds placeholders alone (see 'unused' in Data), for the tree a kernel argument carries but a
    static configuration skips. The state serves a build of any number of its first trees, each build taking the config
    that get_bvh_config returns for the number of trees it covers.
    """
    if is_active and n_leaves < 1:
        gs.raise_exception(f"A BVH set needs at least one leaf per tree, got {n_leaves}.")
    bvh_config = get_bvh_config(n_trees, n_leaves)

    # The state: the scratch of the device-wide radix sort is sized by quadrants, the histograms of the host radix sort
    # hold the chunks of every tree built (see func_build_bvh), and the lanes of the extent are fixed.
    n_sort_scratch = 0
    if bvh_config.sort_kind == BVH_SORT_KIND.DEVICE_RADIX:
        n_sort_scratch = sort_scratch_slots(n_trees * n_leaves, bvh_config.log256_max_n)
    bvh_state = BVHState(
        sizes=dict(
            bvh_trees=n_trees,
            bvh_leaves=n_leaves,
            bvh_nodes=2 * n_leaves - 1,
            sort_slots=n_sort_scratch,
            sort_chunks=n_trees * max(1, bvh_config.sort_max_chunks // n_trees),
            extent_lanes=TREE_EXTENT_LANES,
        ),
        # A set switched off leaves every array unused, else the scratch of the sort and the fit it runs alone
        unused=dict(
            device_radix_sort=bvh_config.sort_kind != BVH_SORT_KIND.DEVICE_RADIX,
            per_tree_radix_sort=bvh_config.sort_kind != BVH_SORT_KIND.PER_TREE_RADIX,
            block_sweep_fit=bvh_config.fit_kind != BVH_FIT_KIND.BLOCK_SWEEP,
        )
        if is_active
        else True,
    )

    return BVHData(bvh_state, bvh_config)


@qd.func
def _func_expand_bits(v: qd.u32) -> qd.u32:
    """Spread the low 10 bits of a value two bits apart."""
    v = (v * qd.u32(0x00010001)) & qd.u32(0xFF0000FF)
    v = (v | ((v & qd.u32(0x00FFFFFF)) << 8)) & qd.u32(0x0F00F00F)
    v = (v * qd.u32(0x00000011)) & qd.u32(0xC30C30C3)
    v = (v * qd.u32(0x00000005)) & qd.u32(0x49249249)
    return v


@qd.func
def _func_delta(i_t: int, i_leaf: int, j_leaf: int, bvh_state: BVHState) -> int:
    """Return the common prefix length of the keys of two sorted leaves, -1 when the second one is out of range."""
    n_leaves = bvh_state.leaves.aabbs_min.shape[1]
    delta = -1
    if j_leaf >= 0 and j_leaf < n_leaves:
        key_i = bvh_state.leaves_keys[i_t * n_leaves + i_leaf]
        key_j = bvh_state.leaves_keys[i_t * n_leaves + j_leaf]
        # The keys of one tree differ at least in their leaf bits, so the prefix is shorter than the key
        delta = qd.math.clz(key_i ^ key_j)
    return delta


@qd.func
def _func_fit_ancestors(i_t: int, i_leaf: int, bvh_state: BVHState, bvh_config: qd.template(), node_fit: qd.template()):
    """Walk from a sorted leaf to the root, fitting each node whose two children are fitted.

    The leaves of a tree walk at once. The second child to arrive at a node fits it and goes on, the first one stops,
    so every node is fitted once after both children.
    """
    n_leaves = bvh_state.leaves.aabbs_min.shape[1]
    i_n = n_leaves - 1 + i_leaf
    i_a = bvh_state.tree.leaves_idx[i_t, i_leaf]
    bvh_state.tree.nodes_min[i_t, i_n] = bvh_state.leaves.aabbs_min[i_t, i_a]
    bvh_state.tree.nodes_max[i_t, i_n] = bvh_state.leaves.aabbs_max[i_t, i_a]
    if qd.static(bool(node_fit)):
        node_fit.fit_leaf(i_t, i_n, i_a)
    for _ in range(n_leaves):
        i_parent = bvh_state.nodes_parent[i_t, i_n]
        if i_parent < 0:
            break
        # The box of this node must be visible to the sibling's thread before the arrival it may win: a device fence
        # on each side of the relaxed atomic of the GPU, the sequentially consistent atomic alone on the CPU
        if qd.static(bvh_config.fit_kind == BVH_FIT_KIND.FENCED_LEAF_WALK):
            qd.simt.grid.mem_fence()
        n_arrived = qd.atomic_add(bvh_state.nodes_fitted[i_t, i_parent], 1)
        if qd.static(bvh_config.fit_kind == BVH_FIT_KIND.FENCED_LEAF_WALK):
            qd.simt.grid.mem_fence()
        if n_arrived == 0:
            break
        i_left = bvh_state.tree.nodes_left[i_t, i_parent]
        i_right = bvh_state.tree.nodes_right[i_t, i_parent]
        bvh_state.tree.nodes_min[i_t, i_parent] = qd.min(
            bvh_state.tree.nodes_min[i_t, i_left], bvh_state.tree.nodes_min[i_t, i_right]
        )
        bvh_state.tree.nodes_max[i_t, i_parent] = qd.max(
            bvh_state.tree.nodes_max[i_t, i_left], bvh_state.tree.nodes_max[i_t, i_right]
        )
        if qd.static(bool(node_fit)):
            node_fit.fit_node(i_t, i_parent, i_left, i_right)
        i_n = i_parent


@qd.func
def _func_push_fit_parent(i_t: int, i_n: int, i_next: int, is_active: bool, sh_frontier_count, bvh_state: BVHState):
    """Count the arrival of a node at its parent, and push the parent into the next frontier on the second arrival.

    An inactive lane takes part with a zero increment, so that every lane of the block runs the same atomics.
    """
    # FIXME: quadrants#925 - an atomic carries a memory barrier that Metal lowers to a control barrier, so the atomics
    # stay in uniform control flow: an inactive lane adds 0 to a valid slot instead of skipping the call.
    i_parent = -1
    if is_active:
        i_parent = bvh_state.nodes_parent[i_t, i_n]
    n_arrival = qd.i32(i_parent >= 0)
    n_arrived = qd.atomic_add(bvh_state.nodes_fitted[i_t, qd.max(i_parent, 0)], n_arrival)
    n_push = qd.i32(n_arrival == 1 and n_arrived == 1)
    i_slot = qd.atomic_add(sh_frontier_count[i_next], n_push)
    if n_push == 1:
        bvh_state.fit_frontier[i_t, i_next, i_slot] = i_parent


@qd.func
def _func_fit_tree_block(i_t: int, i_lane: int, bvh_state: BVHState, block_dim: int, node_fit: qd.template()):
    """Fit the node boxes of a tree by one block of threads, sweeping a frontier of ready nodes up the tree.

    A round fits the nodes both children of which were fitted by earlier rounds and pushes their parents, on their
    second child, into the frontier of the next round. The block synchronizes between rounds, so every box a round
    reads was written before a barrier and the sweep depends on no memory ordering across threads.
    """
    # The block barrier of the SPIR-V backends orders shared memory alone, so a device-scope fence precedes each one
    # to publish the boxes, the arrival counters and the frontier, which live in device memory
    n_leaves = bvh_state.leaves.aabbs_min.shape[1]
    sh_frontier_count = qd.simt.block.SharedArray((2,), gs.qd_int)
    if i_lane == 0:
        sh_frontier_count[0] = 0
        sh_frontier_count[1] = 0
    qd.simt.block.sync()
    for i_chunk in range((n_leaves + block_dim - 1) // block_dim):
        i_leaf = i_chunk * block_dim + i_lane
        is_active = i_leaf < n_leaves
        i_n = n_leaves - 1 + i_leaf
        if is_active:
            i_a = bvh_state.tree.leaves_idx[i_t, i_leaf]
            bvh_state.tree.nodes_min[i_t, i_n] = bvh_state.leaves.aabbs_min[i_t, i_a]
            bvh_state.tree.nodes_max[i_t, i_n] = bvh_state.leaves.aabbs_max[i_t, i_a]
            if qd.static(bool(node_fit)):
                node_fit.fit_leaf(i_t, i_n, i_a)
        _func_push_fit_parent(i_t, i_n, 0, is_active, sh_frontier_count, bvh_state)
    qd.simt.grid.mem_fence()
    qd.simt.block.sync()
    for i_round in range(n_leaves):
        i_cur = i_round % 2
        n_frontier = sh_frontier_count[i_cur]
        if n_frontier == 0:
            break
        for i_chunk in range((n_frontier + block_dim - 1) // block_dim):
            i_slot = i_chunk * block_dim + i_lane
            is_active = i_slot < n_frontier
            i_n = bvh_state.fit_frontier[i_t, i_cur, qd.min(i_slot, n_frontier - 1)]
            if is_active:
                i_left = bvh_state.tree.nodes_left[i_t, i_n]
                i_right = bvh_state.tree.nodes_right[i_t, i_n]
                bvh_state.tree.nodes_min[i_t, i_n] = qd.min(
                    bvh_state.tree.nodes_min[i_t, i_left], bvh_state.tree.nodes_min[i_t, i_right]
                )
                bvh_state.tree.nodes_max[i_t, i_n] = qd.max(
                    bvh_state.tree.nodes_max[i_t, i_left], bvh_state.tree.nodes_max[i_t, i_right]
                )
                if qd.static(bool(node_fit)):
                    node_fit.fit_node(i_t, i_n, i_left, i_right)
            _func_push_fit_parent(i_t, i_n, 1 - i_cur, is_active, sh_frontier_count, bvh_state)
        qd.simt.grid.mem_fence()
        qd.simt.block.sync()
        # The list swept this round takes the pushes of the next one, so its count is cleared behind a barrier
        if i_lane == 0:
            sh_frontier_count[i_cur] = 0
        qd.simt.block.sync()


@qd.func
def func_build_bvh(
    n_trees: int, bvh_state: BVHState, bvh_config: qd.template(), eps: float, node_fit: qd.template() = None
):
    """Build the first trees of the set from the boxes of their leaves, at the top level of a kernel.

    The passes run in order: the extent of each tree, the morton key of each leaf, the sort of the keys, the leaf index
    of each sorted slot, the radix tree and the node boxes. The build covers n_trees trees, under the config
    get_bvh_config returns for that number of trees built.

    The closure `node_fit`, if any, fits data of its own per node along the node boxes, which a visitor reads to prune
    the nodes (see func_bvh_traverse): `fit_leaf(i_t, i_n, i_a)` at the leaf node i_n of leaf i_a, and
    `fit_node(i_t, i_n, i_left, i_right)` at the internal node i_n once both its children are fitted.
    """
    n_leaves = bvh_state.leaves.aabbs_min.shape[1]
    # The device-wide radix sort reads its key count on the device
    bvh_state.n_keys[()] = n_trees * n_leaves

    # Extent of each tree over its leaf centers: each lane folds a strided slice of the leaves, then one thread per
    # tree folds the lanes, so the reduction needs no floating-point atomic
    n_extent_lanes = bvh_state.lanes_min.shape[1]
    for i_t, i_lane in qd.ndrange(n_trees, n_extent_lanes):
        lane_min = (bvh_state.leaves.aabbs_min[i_t, 0] + bvh_state.leaves.aabbs_max[i_t, 0]) * 0.5
        lane_max = lane_min
        for i_chunk in range((n_leaves + n_extent_lanes - 1) // n_extent_lanes):
            i_a = i_chunk * n_extent_lanes + i_lane
            if i_a < n_leaves:
                center = (bvh_state.leaves.aabbs_min[i_t, i_a] + bvh_state.leaves.aabbs_max[i_t, i_a]) * 0.5
                lane_min = qd.min(lane_min, center)
                lane_max = qd.max(lane_max, center)
        bvh_state.lanes_min[i_t, i_lane] = lane_min
        bvh_state.lanes_max[i_t, i_lane] = lane_max
    for i_t in range(n_trees):
        tree_min = bvh_state.lanes_min[i_t, 0]
        tree_max = bvh_state.lanes_max[i_t, 0]
        for i_lane in range(1, n_extent_lanes):
            tree_min = qd.min(tree_min, bvh_state.lanes_min[i_t, i_lane])
            tree_max = qd.max(tree_max, bvh_state.lanes_max[i_t, i_lane])
        bvh_state.trees_min[i_t] = tree_min
        bvh_state.trees_max[i_t] = tree_max

    # Key of each leaf: the tree, then the morton code of the leaf center in the extent of its tree scaled to the unit
    # cube (a flat extent maps to 0), then the leaf index
    for i_t, i_a in qd.ndrange(n_trees, n_leaves):
        extent = bvh_state.trees_max[i_t] - bvh_state.trees_min[i_t]
        tree_scale = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(3)):
            if extent[j] > eps:
                tree_scale[j] = 1.0 / extent[j]
        center = (bvh_state.leaves.aabbs_min[i_t, i_a] + bvh_state.leaves.aabbs_max[i_t, i_a]) * 0.5
        scaled = (center - bvh_state.trees_min[i_t]) * tree_scale
        code_x = _func_expand_bits(qd.u32(qd.floor(scaled[0] * 1023.0)))
        code_y = _func_expand_bits(qd.u32(qd.floor(scaled[1] * 1023.0)))
        code_z = _func_expand_bits(qd.u32(qd.floor(scaled[2] * 1023.0)))
        code = (code_x << 2) | (code_y << 1) | code_z
        # The bits a leaf index takes, at least one
        leaf_bits = qd.max(1, 32 - qd.math.clz(qd.u32(n_leaves - 1)))
        key = (qd.u64(code) << leaf_bits) | qd.u64(i_a)
        bvh_state.leaves_keys[i_t * n_leaves + i_a] = (
            qd.u64(i_t) << (qd.static(bvh_config.morton_bits) + leaf_bits)
        ) | key

    # Sort of the keys, every key unique through its leaf bits so the order is total
    if qd.static(bvh_config.sort_kind == BVH_SORT_KIND.DEVICE_RADIX):
        radix_sort(
            bvh_state.leaves_keys,
            bvh_state.keys_scratch,
            bvh_state.leaves_keys,
            bvh_state.keys_scratch,
            bvh_state.sort_scratch,
            bvh_state.n_keys,
            key_dtype=qd.u64,
            has_values=False,
            end_bit=qd.static(bvh_config.end_bit),
            log256_max_n=qd.static(bvh_config.log256_max_n),
        )
    elif qd.static(bvh_config.sort_kind == BVH_SORT_KIND.RANK):
        # Each leaf ranks itself among the keys of its tree
        for i_t, i_a in qd.ndrange(n_trees, n_leaves):
            key = bvh_state.leaves_keys[i_t * n_leaves + i_a]
            rank = 0
            for i_o in range(n_leaves):
                if bvh_state.leaves_keys[i_t * n_leaves + i_o] < key:
                    rank += 1
            bvh_state.keys_scratch[i_t * n_leaves + rank] = key
        for i_k in range(n_trees * n_leaves):
            bvh_state.leaves_keys[i_k] = bvh_state.keys_scratch[i_k]
    else:
        # A least-significant-digit radix sort of eight bits per pass on the host backend, the leaves of each tree
        # split over chunks: the chunks count their digits and scatter their keys in parallel, one thread per tree
        # turning the counts into offsets in between. The passes ping-pong between the keys and the scratch and end
        # in the keys. The passes unroll statically: a loop of the kernel is parallel at its top level alone, so a
        # runtime loop over the passes would serialize the chunks. Row i_h of sort_hist holds the histogram of chunk i_c
        # of tree i_t, and a build of fewer trees than the set splits the same rows into more chunks.
        n_sort_chunks = qd.max(
            1, qd.min(qd.static(bvh_config.sort_max_chunks), bvh_state.sort_hist.shape[0] // n_trees)
        )
        for i_pass in qd.static(range(qd.static(bvh_config.end_bit) // 8)):
            shift = 8 * i_pass
            for i_t, i_c in qd.ndrange(n_trees, n_sort_chunks):
                i_h = i_t * n_sort_chunks + i_c
                for i_d in range(256):
                    bvh_state.sort_hist[i_h, i_d] = 0
                for i_a in range(i_c * n_leaves // n_sort_chunks, (i_c + 1) * n_leaves // n_sort_chunks):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = bvh_state.leaves_keys[i_t * n_leaves + i_a]
                    else:
                        key = bvh_state.keys_scratch[i_t * n_leaves + i_a]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    bvh_state.sort_hist[i_h, digit] = bvh_state.sort_hist[i_h, digit] + 1
            for i_t in range(n_trees):
                total = 0
                for i_d in range(256):
                    for i_c in range(n_sort_chunks):
                        i_h = i_t * n_sort_chunks + i_c
                        count = bvh_state.sort_hist[i_h, i_d]
                        bvh_state.sort_hist[i_h, i_d] = total
                        total += count
            for i_t, i_c in qd.ndrange(n_trees, n_sort_chunks):
                i_h = i_t * n_sort_chunks + i_c
                for i_a in range(i_c * n_leaves // n_sort_chunks, (i_c + 1) * n_leaves // n_sort_chunks):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = bvh_state.leaves_keys[i_t * n_leaves + i_a]
                    else:
                        key = bvh_state.keys_scratch[i_t * n_leaves + i_a]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    slot = bvh_state.sort_hist[i_h, digit]
                    bvh_state.sort_hist[i_h, digit] = slot + 1
                    if qd.static(i_pass % 2 == 0):
                        bvh_state.keys_scratch[i_t * n_leaves + slot] = key
                    else:
                        bvh_state.leaves_keys[i_t * n_leaves + slot] = key

    # Leaf of each sorted slot, the links of the leaf nodes and the root, and the arrival counters of the fit
    for i_t, i_leaf in qd.ndrange(n_trees, n_leaves):
        leaf_bits = qd.max(1, 32 - qd.math.clz(qd.u32(n_leaves - 1)))
        leaf_mask = (qd.u64(1) << leaf_bits) - qd.u64(1)
        bvh_state.tree.leaves_idx[i_t, i_leaf] = qd.i32(bvh_state.leaves_keys[i_t * n_leaves + i_leaf] & leaf_mask)
        bvh_state.tree.nodes_left[i_t, n_leaves - 1 + i_leaf] = -1
        bvh_state.tree.nodes_right[i_t, n_leaves - 1 + i_leaf] = -1
    for i_t, i_n in qd.ndrange(n_trees, bvh_state.nodes_fitted.shape[1]):
        bvh_state.nodes_fitted[i_t, i_n] = 0
    for i_t in range(n_trees):
        bvh_state.nodes_parent[i_t, 0] = -1

    # Radix tree: internal node i covers the sorted leaves sharing the longest key prefix around split i (Karras 2012)
    # Internal node i_n owns a range of sorted leaves that starts at leaf i_n and extends toward the longer prefix
    for i_t, i_n in qd.ndrange(n_trees, n_leaves - 1):
        delta_next = _func_delta(i_t, i_n, i_n + 1, bvh_state)
        delta_prev = _func_delta(i_t, i_n, i_n - 1, bvh_state)
        direction = 1 if delta_next > delta_prev else -1
        delta_min = _func_delta(i_t, i_n, i_n - direction, bvh_state)
        # The range of the node, by a doubling then a binary search over the leaves sharing more than delta_min,
        # each bounded by the 64 bits of a key
        length_max = 2
        for _ in range(64):
            if _func_delta(i_t, i_n, i_n + length_max * direction, bvh_state) <= delta_min:
                break
            length_max *= 2
        length = 0
        step = length_max // 2
        for _ in range(64):
            if step == 0:
                break
            if _func_delta(i_t, i_n, i_n + (length + step) * direction, bvh_state) > delta_min:
                length += step
            step //= 2
        j_leaf = i_n + length * direction
        # The split of the range, where the prefix shared by the whole node stops
        delta_node = _func_delta(i_t, i_n, j_leaf, bvh_state)
        split = 0
        step = (length + 1) // 2
        for _ in range(64):
            if step == 0:
                break
            if _func_delta(i_t, i_n, i_n + (split + step) * direction, bvh_state) > delta_node:
                split += step
            step = (step + 1) // 2 if step > 1 else 0
        gamma = i_n + split * direction + qd.min(direction, 0)
        i_left = gamma + n_leaves - 1 if qd.min(i_n, j_leaf) == gamma else gamma
        i_right = gamma + n_leaves if qd.max(i_n, j_leaf) == gamma + 1 else gamma + 1
        bvh_state.tree.nodes_left[i_t, i_n] = i_left
        bvh_state.tree.nodes_right[i_t, i_n] = i_right
        bvh_state.nodes_parent[i_t, i_left] = i_n
        bvh_state.nodes_parent[i_t, i_right] = i_n

    # Node boxes, bottom-up from the leaves (see BVH_FIT_KIND for the regimes)
    if qd.static(bvh_config.fit_kind == BVH_FIT_KIND.BLOCK_SWEEP):
        FIT_BLOCK_DIM = qd.static(128)
        qd.loop_config(block_dim=FIT_BLOCK_DIM)
        for i_flat in range(n_trees * FIT_BLOCK_DIM):
            _func_fit_tree_block(i_flat // FIT_BLOCK_DIM, i_flat % FIT_BLOCK_DIM, bvh_state, FIT_BLOCK_DIM, node_fit)
    else:
        for i_t, i_leaf in qd.ndrange(n_trees, n_leaves):
            _func_fit_ancestors(i_t, i_leaf, bvh_state, bvh_config, node_fit)


@qd.kernel(fastcache=True)
def build_bvh(bvh_state: BVHState, bvh_config: qd.template(), eps: float, node_fit: qd.template()):
    """Build every tree of the set from the boxes of its leaves, and the data of its nodes that `node_fit` fits.

    The kernel runs func_build_bvh for a caller outside a kernel.
    """
    func_build_bvh(bvh_state.leaves.aabbs_min.shape[0], bvh_state, bvh_config, eps, node_fit)


def build_packed_bvh(
    trees_aabb_min: list[np.ndarray],
    trees_aabb_max: list[np.ndarray],
    packed_cls: type[BVHPackedTrees] = BVHPackedTrees,
    **sizes: int,
) -> BVHPackedTrees:
    """Build one tree over each set of boxes, once, and pack the trees in one row.

    Tree i_t spans the boxes [trees_aabb_min[i_t], trees_aabb_max[i_t]], at least one, and its leaves hold the indices
    of its boxes among the boxes of every tree in order. The trees of one leaf count are built together, each at its
    exact size. Without tree, the trees are placeholders, which no query reaches. The packed set is an instance of
    `packed_cls`, a subclass whose other sizes `sizes` gives.
    """
    trees_n_leaves = np.array([len(aabb_min) for aabb_min in trees_aabb_min], dtype=gs.np_int)
    n_trees, n_leaves = len(trees_n_leaves), trees_n_leaves.sum()
    if n_trees == 0:
        return packed_cls(
            sizes=dict(bvh_trees=1, bvh_nodes=0, bvh_leaves=0, bvh_packed_trees=0, **sizes),
            unused={(BVHPackedTrees.tree, BVHPackedTrees.trees_root, BVHPackedTrees.trees_leaf_offset): True},
        )
    trees_leaf_start = np.cumsum(trees_n_leaves) - trees_n_leaves
    trees_root = np.cumsum(2 * trees_n_leaves - 1) - (2 * trees_n_leaves - 1)
    trees_leaf_offset = trees_root + trees_n_leaves - 1 - trees_leaf_start
    nodes_left = np.empty(2 * n_leaves - n_trees, dtype=gs.np_int)
    nodes_right = np.empty_like(nodes_left)
    nodes_min = np.empty((len(nodes_left), 3), dtype=gs.np_float)
    nodes_max = np.empty_like(nodes_min)
    leaves_idx = np.empty(n_leaves, dtype=gs.np_int)
    for n_tree_leaves in np.unique(trees_n_leaves).tolist():
        trees_idx = np.flatnonzero(trees_n_leaves == n_tree_leaves)
        bvh_state, bvh_config = get_bvh_data(len(trees_idx), n_tree_leaves)
        # The boxes come in the dtype of their source, which a device without double precision cannot copy from
        bvh_state.leaves.aabbs_min.from_numpy(np.stack([trees_aabb_min[i_t] for i_t in trees_idx], dtype=gs.np_float))
        bvh_state.leaves.aabbs_max.from_numpy(np.stack([trees_aabb_max[i_t] for i_t in trees_idx], dtype=gs.np_float))
        build_bvh(bvh_state, bvh_config, eps=gs.EPS, node_fit=None)
        # The nodes and the leaves of each tree, its rows first, the children of a packed node being absolute
        tree_left = qd_to_numpy(bvh_state.tree.nodes_left, transpose=True).T
        tree_right = qd_to_numpy(bvh_state.tree.nodes_right, transpose=True).T
        tree_min = qd_to_numpy(bvh_state.tree.nodes_min, transpose=True).swapaxes(0, 1)
        tree_max = qd_to_numpy(bvh_state.tree.nodes_max, transpose=True).swapaxes(0, 1)
        tree_leaves_idx = qd_to_numpy(bvh_state.tree.leaves_idx, transpose=True).T
        nodes_idx = trees_root[trees_idx, None] + np.arange(2 * n_tree_leaves - 1)
        nodes_left[nodes_idx] = np.where(tree_left >= 0, tree_left + trees_root[trees_idx, None], -1)
        nodes_right[nodes_idx] = np.where(tree_right >= 0, tree_right + trees_root[trees_idx, None], -1)
        nodes_min[nodes_idx] = tree_min
        nodes_max[nodes_idx] = tree_max
        leaves_idx[trees_leaf_start[trees_idx, None] + np.arange(n_tree_leaves)] = (
            tree_leaves_idx + trees_leaf_start[trees_idx, None]
        )
    packed_trees = packed_cls(
        sizes=dict(bvh_trees=1, bvh_nodes=len(nodes_left), bvh_leaves=n_leaves, bvh_packed_trees=n_trees, **sizes)
    )
    packed_trees.tree.nodes_left.from_numpy(nodes_left[None])
    packed_trees.tree.nodes_right.from_numpy(nodes_right[None])
    packed_trees.tree.nodes_min.from_numpy(nodes_min[None])
    packed_trees.tree.nodes_max.from_numpy(nodes_max[None])
    packed_trees.tree.leaves_idx.from_numpy(leaves_idx[None])
    packed_trees.trees_root.from_numpy(trees_root)
    packed_trees.trees_leaf_offset.from_numpy(trees_leaf_offset)
    return packed_trees


def build_packed_instances(
    instances_elem_start: list[int], instances_aabb_min: list[np.ndarray], instances_aabb_max: list[np.ndarray]
) -> PackedInstances:
    """Build the trees of the elements of a set of instances once, the instances of identical elements sharing one.

    The elements of instance i_k are numbered from instances_elem_start[i_k] and span the boxes
    [instances_aabb_min[i_k], instances_aabb_max[i_k]] in the frame of the instance, an instance without element
    getting no tree.
    """
    trees_key, trees_aabb_min, trees_aabb_max, trees_leaf_start = {}, [], [], [0]
    instances_tree_idx, instances_leaf_base = [], []
    for elem_start, aabb_min, aabb_max in zip(instances_elem_start, instances_aabb_min, instances_aabb_max):
        if len(aabb_min) == 0:
            instances_tree_idx.append(-1)
            instances_leaf_base.append(0)
            continue
        key = aabb_min.tobytes() + aabb_max.tobytes()
        if key not in trees_key:
            trees_key[key] = len(trees_aabb_min)
            trees_aabb_min.append(aabb_min)
            trees_aabb_max.append(aabb_max)
            trees_leaf_start.append(trees_leaf_start[-1] + len(aabb_min))
        instances_tree_idx.append(trees_key[key])
        instances_leaf_base.append(elem_start - trees_leaf_start[trees_key[key]])
    packed_instances = build_packed_bvh(
        trees_aabb_min, trees_aabb_max, PackedInstances, bvh_instances=len(instances_tree_idx)
    )
    packed_instances.instances_tree_idx.from_numpy(np.array(instances_tree_idx, dtype=gs.np_int))
    packed_instances.instances_leaf_base.from_numpy(np.array(instances_leaf_base, dtype=gs.np_int))
    return packed_instances


@qd.kernel(fastcache=True)
def kernel_build_instances_tree(
    i_k_excluded: int,
    instances_aabb_min: qd.Tensor,
    instances_aabb_max: qd.Tensor,
    bvh_state: BVHState,
    packed_instances: PackedInstances,
    bvh_config: qd.template(),
    eps: float,
):
    """Build the tree of the instances of each environment from their boxes in the world, an instance without tree and
    instance i_k_excluded getting an empty box, which no query reaches."""
    for i_k, i_b in qd.ndrange(bvh_state.leaves.aabbs_min.shape[1], bvh_state.leaves.aabbs_min.shape[0]):
        aabb_min = qd.Vector([qd.math.inf, qd.math.inf, qd.math.inf], dt=gs.qd_float)
        aabb_max = -aabb_min
        if packed_instances.instances_tree_idx[i_k] >= 0 and i_k != i_k_excluded:
            aabb_min = instances_aabb_min[i_k, i_b]
            aabb_max = instances_aabb_max[i_k, i_b]
        bvh_state.leaves.aabbs_min[i_b, i_k] = aabb_min
        bvh_state.leaves.aabbs_max[i_b, i_k] = aabb_max
    func_build_bvh(bvh_state.leaves.aabbs_min.shape[0], bvh_state, bvh_config, eps)


@qd.func
def func_bvh_traverse(
    i_t: int,
    i_q: int,
    i_root: int,
    leaf_offset: int,
    leaf_base: int,
    query,
    bound: float,
    tree: BVHTreeState,
    visitor: qd.template(),
) -> float:
    """Visit the elements of a tree that a query of environment i_t reaches, depth first, and return the bound the
    visits leave.

    The tree is row i_t of a tree set, or the single row of a set of one row, from node i_root, the leaf node i_n
    holding the leaf leaves_idx[row, i_n - leaf_offset], visited as the element leaf_base + leaf (see BVHTreeState).
    The visitor is a closure over the arrays it reads and writes, its funcs handed the environment i_t, the query
    index i_q, the query, a value of the struct type `visitor.Query`, and the bound: `is_node_hit(i_t, i_q, i_n,
    node_min, node_max, query, bound)` accepts the nodes to descend into, i_n indexing the node in its tree, and
    `visit_leaf(i_t, i_q, i_k, i_a, query, bound)` visits element i_a and returns the bound, which a closest-hit search
    lowers to prune the nodes beyond its best hit. A visitor writes its other results into its own arrays.

    A visitor whose `is_instance_level` holds walks a top-level tree whose leaves are instances of its packed trees
    `instances`: `enter_instance(i_t, i_q, i_k, query, bound)` returns the packed tree of instance i_k, -1 for none,
    with the query in its frame and the offset of its elements, and its `inner` visitor tests the nodes of both levels
    and visits the elements, i_k handed to it. A single-level visitor gets -1 for i_k.
    """
    # One pending sibling per level of each tree: a tree splits one key bit per level below the bits every key of the
    # set shares, so its depth is at most the morton bits plus the leaf bits, well under 64.
    STACK_SIZE = qd.static(128 if visitor.is_instance_level else 64)
    i_row = i_t if tree.nodes_left.shape[0] > 1 else 0
    stack = qd.Vector.zero(gs.qd_int, STACK_SIZE)
    stack[0] = i_root
    n_stack = 1
    # The walk of an instance ends before the next node of the top-level tree is popped, so the instance, its query and
    # the offsets that entering it sets hold over its whole walk.
    i_k = -1
    query_inner = query
    inner_leaf_offset = 0
    inner_leaf_base = 0
    while n_stack > 0:
        n_stack -= 1
        i_n = stack[n_stack]
        # A node of a bottom-level tree is stacked as -2 - i_n
        is_top = i_n >= 0
        i_n_node, query_node = i_n, query
        node_min, node_max = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        if is_top:
            node_min, node_max = tree.nodes_min[i_row, i_n], tree.nodes_max[i_row, i_n]
        if qd.static(visitor.is_instance_level):
            if not is_top:
                i_n_node = -2 - i_n
                node_min = visitor.instances.tree.nodes_min[0, i_n_node]
                node_max = visitor.instances.tree.nodes_max[0, i_n_node]
                query_node = query_inner
        # The inner visitor of an instance-level one tests the nodes of both levels
        is_node_hit = False
        if qd.static(visitor.is_instance_level):
            is_node_hit = visitor.inner.is_node_hit(i_t, i_q, i_n_node, node_min, node_max, query_node, bound)
        else:
            is_node_hit = visitor.is_node_hit(i_t, i_q, i_n_node, node_min, node_max, query_node, bound)
        if is_node_hit:
            if is_top:
                i_left = tree.nodes_left[i_row, i_n]
                if i_left < 0:
                    i_leaf = tree.leaves_idx[i_row, i_n - leaf_offset]
                    if qd.static(not visitor.is_instance_level):
                        bound = visitor.visit_leaf(i_t, i_q, -1, leaf_base + i_leaf, query, bound)
                    else:
                        i_tree, query_entered, leaf_base_entered = visitor.enter_instance(
                            i_t, i_q, i_leaf, query, bound
                        )
                        if i_tree >= 0:
                            i_k = i_leaf
                            query_inner = query_entered
                            inner_leaf_base = leaf_base_entered
                            inner_leaf_offset = visitor.instances.trees_leaf_offset[i_tree]
                            stack[n_stack] = -2 - visitor.instances.trees_root[i_tree]
                            n_stack += 1
                else:
                    stack[n_stack] = tree.nodes_right[i_row, i_n]
                    stack[n_stack + 1] = i_left
                    n_stack += 2
            if qd.static(visitor.is_instance_level):
                if not is_top:
                    i_left = visitor.instances.tree.nodes_left[0, i_n_node]
                    if i_left < 0:
                        i_leaf = visitor.instances.tree.leaves_idx[0, i_n_node - inner_leaf_offset]
                        bound = visitor.inner.visit_leaf(i_t, i_q, i_k, inner_leaf_base + i_leaf, query_inner, bound)
                    else:
                        stack[n_stack] = -2 - visitor.instances.tree.nodes_right[0, i_n_node]
                        stack[n_stack + 1] = -2 - i_left
                        n_stack += 2
    return bound


@qd.func
def func_aabb_intersects_aabb(
    a_min: qd.types.vector(3), a_max: qd.types.vector(3), b_min: qd.types.vector(3), b_max: qd.types.vector(3)
) -> bool:
    """Return whether two boxes overlap, touching ones included."""
    is_hit = True
    for j in qd.static(range(3)):
        if a_min[j] > b_max[j] or a_max[j] < b_min[j]:
            is_hit = False
    return is_hit


@qd.data_oriented
class BoxQuery:
    """The visitor of a box query, recording the (environment, element, query) triplets of the elements its box meets.

    The query is the box (see Query). Each environment holds an equal share of the results, its count growing past
    that share, the pairs beyond it dropped, so that a count above the share reports the overflow of that environment
    while the others keep theirs. The caller zeroes `count` and `trees_count` before every query. A subclass drops some
    elements or instances through `is_kept` and `is_instance_kept`.
    """

    is_instance_level = False

    def __init__(self, results: BVHQueryResults) -> None:
        self.results = results
        self.Query = qd.types.struct(aabb_min=qd.types.vector(3, gs.qd_float), aabb_max=qd.types.vector(3, gs.qd_float))

    @qd.func
    def is_node_hit(
        self,
        i_t: int,
        i_q: int,
        i_n: int,
        node_min: qd.types.vector(3),
        node_max: qd.types.vector(3),
        query,
        bound: float,
    ) -> bool:
        return func_aabb_intersects_aabb(node_min, node_max, query.aabb_min, query.aabb_max)

    @qd.func
    def is_kept(self, i_t: int, i_q: int, i_k: int, i_a: int) -> bool:
        return True

    @qd.func
    def is_instance_kept(self, i_t: int, i_q: int, i_k: int) -> bool:
        return True

    @qd.func
    def to_frame(self, query, pos: qd.types.vector(3), quat: qd.types.vector(4)):
        quat_inv = gu.qd_inv_quat(quat)
        aabb_min, aabb_max = gu.qd_transform_aabb(
            query.aabb_min, query.aabb_max, trans=-gu.qd_transform_by_quat(pos, quat_inv), quat=quat_inv
        )
        return self.Query(aabb_min=aabb_min, aabb_max=aabb_max)

    @qd.func
    def visit_leaf(self, i_t: int, i_q: int, i_k: int, i_a: int, query, bound: float) -> float:
        if self.is_kept(i_t, i_q, i_k, i_a):
            n_tree_results = qd.atomic_add(self.results.trees_count[i_t], 1)
            if n_tree_results < self.results.triplets.shape[0] // self.results.trees_count.shape[0]:
                i_r = qd.atomic_add(self.results.count[0], 1)
                self.results.triplets[i_r] = gs.qd_ivec3(i_t, i_a, i_q)
        return bound


@qd.data_oriented
class InstancesQuery:
    """The visitor of the instances of packed trees posed in each environment that a query reaches, walking the tree of
    each with its inner visitor, whose query it takes.

    The inner visitor tests the nodes of both levels, gates the instances through `is_instance_kept(i_t, i_q, i_k)`,
    returns the query in the frame of an instance through `to_frame(query, pos, quat)`, and visits the elements. It
    reads the instances (see PackedInstances) and the state holding their poses, `pos` and `quat` per instance and
    environment.
    """

    is_instance_level = True

    def __init__(self, inner, instances: PackedInstances, instances_state) -> None:
        self.inner = inner
        self.instances = instances
        self.instances_state = instances_state
        self.Query = inner.Query

    @qd.func
    def enter_instance(self, i_t: int, i_q: int, i_k: int, query, bound: float):
        i_tree = -1
        query_local = query
        if self.inner.is_instance_kept(i_t, i_q, i_k):
            i_tree = self.instances.instances_tree_idx[i_k]
            query_local = self.inner.to_frame(
                query, self.instances_state.pos[i_k, i_t], self.instances_state.quat[i_k, i_t]
            )
        return i_tree, query_local, self.instances.instances_leaf_base[i_k]


@qd.func
def func_bvh_query_leaves(leaves: BVHLeaves, tree: BVHTreeState, visitor: qd.template()):
    """Query every leaf box of a set against the tree of the same index in another set (see BoxQuery)."""
    for i_t, i_q in qd.ndrange(leaves.aabbs_min.shape[0], leaves.aabbs_min.shape[1]):
        func_bvh_traverse(
            i_t,
            i_q,
            i_root=0,
            leaf_offset=tree.leaves_idx.shape[1] - 1,
            leaf_base=0,
            query=visitor.Query(aabb_min=leaves.aabbs_min[i_t, i_q], aabb_max=leaves.aabbs_max[i_t, i_q]),
            bound=0.0,
            tree=tree,
            visitor=visitor,
        )
