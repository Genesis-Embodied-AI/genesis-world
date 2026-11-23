"""Linear bounding volume hierarchies (LBVH) over 14-sided discrete oriented polytopes (14-DOP): build and box queries
that append their hits.

A 14-DOP bounds a set by its extent along 7 axes: the 3 coordinate axes and the 4 diagonals (1, 1, 1), (1, 1, -1),
(1, -1, 1) and (1, -1, -1), whose projections are kept unnormalized (a diagonal is sqrt(3) long). Its bounds are 32-bit
floats rounded outward from the coordinates of the points they cover, so a bound never excludes a point it was built
from. Two bounds overlap within a gap unless they lie apart by at least the gap along one axis, the gap scaling by
sqrt(3) along the diagonals.

A set holds one tree per batch over the same number of leaves. The caller writes the bound and the tag of each leaf,
builds the set (Karras 2012, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"), and queries
it with one bound per query: every leaf a query meets is appended to a list of hits (batch, query, leaf), its count
growing past the capacity of the list on overflow, the hits beyond it dropped. A node carries the tag its leaves share,
-1 when they differ, and a query skips the subtrees whose leaves all carry its own tag.

A query runs on one thread, walking the tree depth first, or on the 32 lanes of a block, which pop up to 32 nodes of a
frontier shared by the block at a time. Both schedules append the same hits, in different orders, unless the frontier
of a block fills its shared memory, which the hit list counts (see DOPBVHHits).

Arrays of a batch take the batch as their last index.
"""

import math
from enum import IntEnum
from typing import NamedTuple

import quadrants as qd
from quadrants.algorithms import sort as radix_sort
from quadrants.algorithms import sort_scratch_slots

import genesis as gs
from genesis.utils.data import Config, Data, DataKind, Size, field
from genesis.utils.misc import get_gpu_core_count
from genesis.utils.simt import qd_block_ballot, qd_block_shuffle

# The axes of a 14-DOP, the coordinate axes first. The funcs declare the same count as a local.
DOP_AXES = 7

DOP_TREES = Size("dop_trees")
DOP_LEAVES = Size("dop_leaves")
# The nodes of each tree, 2 * n_leaves - 1, its internal nodes first and then its leaves
DOP_NODES = Size("dop_nodes")
DOP_QUERIES = Size("dop_queries")
DOP_HITS = Size("dop_hits")
SORT_SLOTS = Size("dop_sort_slots")
SORT_CHUNKS = Size("dop_sort_chunks")
EXTENT_LANES = Size("dop_extent_lanes")
# The scratch of each sort and fit (see DOP_SORT_KIND and DOP_FIT_KIND)
DEVICE_RADIX_SORT = "device_radix_sort"
PER_TREE_RADIX_SORT = "per_tree_radix_sort"
BLOCK_SWEEP_FIT = "block_sweep_fit"

# Bits of a morton code: three interleaved 10-bit coordinates
MORTON_BITS = 30
# The rank sort, quadratic per tree, beats the radix sort up to a leaf count. On the CPU the trees sort serially, so the
# bound holds at every tree count. On the GPU the passes of every tree run together and the crossover sits near one leaf
# per core for one tree, shrinking with the cube root of the tree count.
RANK_SORT_MAX_LEAVES_CPU = 256
RANK_SORT_LEAVES_PER_CORE = 0.75
# Most chunks of the leaves of one tree the host radix sort counts and scatters in parallel
SORT_MAX_CHUNKS = 64
# Lanes reducing the extent of one tree, each over a strided slice of its leaves, folded by one thread per tree
TREE_EXTENT_LANES = 64


class DOP_SORT_KIND(IntEnum):
    """The sort ordering the leaf keys of a set of trees."""

    # The device-wide radix sort of quadrants over every key of the set (GPU, large trees)
    DEVICE_RADIX = 0
    # The rank of each leaf among the keys of its tree (small trees)
    RANK = 1
    # A radix sort per tree, its chunks in parallel (CPU, large trees)
    PER_TREE_RADIX = 2


class DOP_FIT_KIND(IntEnum):
    """The bottom-up fit of the node bounds of a set of trees."""

    # One thread per leaf walking to the root, the arrival counters sequentially consistent (CPU)
    LEAF_WALK = 0
    # The leaf walk with a device fence on each side of the relaxed arrival atomic (GPU)
    FENCED_LEAF_WALK = 1
    # One block per tree sweeping the nodes level by level, every barrier uniform (Metal, see quadrants#925)
    BLOCK_SWEEP = 2


class DOPBVHConfig(Config):
    """The compile-time shape of a set of 14-DOP trees and of its queries.

    A leaf key packs the tree, the morton code and the leaf index in that order, so one flat sort orders every tree.
    """

    # The bits of a morton code, three interleaved coordinates of the same width
    morton_bits: int
    # The bits a sort must order, a multiple of 16 up to 64
    end_bit: int
    # Scan depth of the device-wide radix sort (see quadrants.algorithms.sort), 1 to 8
    log256_max_n: int
    # DOP_SORT_KIND
    sort_kind: int
    # DOP_FIT_KIND
    fit_kind: int
    # SORT_MAX_CHUNKS capped by the CPU threads of quadrants
    sort_max_chunks: int
    # Whether a query runs on the 32 lanes of a block rather than on one thread
    is_block_query: bool


class DOPBVHLeaves(Data, kind=DataKind.SCRATCH, shape=(DOP_LEAVES, DOP_TREES)):
    """The bounds and tags of the leaves of a set of trees, written before a build, leaf i_a of tree i_t at [i_a, i_t]."""

    lower: qd.Tensor = field((qd.f32, DOP_AXES))
    upper: qd.Tensor = field((qd.f32, DOP_AXES))
    tag: qd.Tensor = field(int)


class DOPBVHTree(Data, kind=DataKind.SCRATCH, shape=(DOP_NODES, DOP_TREES)):
    """
    What a query reads of a set of trees.

    A tree roots at node 0, its internal nodes first and then its leaves, the leaf node n_leaves - 1 + i_leaf holding
    the leaf leaves_idx[i_leaf]. A leaf node holds -1 in nodes_left. The tag of a node is the tag of its leaves, -1 when
    they differ.
    """

    nodes_left: qd.Tensor = field(int)
    nodes_right: qd.Tensor = field(int)
    nodes_lower: qd.Tensor = field((qd.f32, DOP_AXES))
    nodes_upper: qd.Tensor = field((qd.f32, DOP_AXES))
    nodes_tag: qd.Tensor = field(int)
    leaves_idx: qd.Tensor = field(int, shape=(DOP_LEAVES, DOP_TREES))


class DOPBVHState(Data, kind=DataKind.SCRATCH):
    """A set of trees and the scratch of its build (see func_build_dop_bvh)."""

    tree: DOPBVHTree
    leaves: DOPBVHLeaves
    leaves_keys: qd.Tensor = field(qd.u64, shape=(DOP_TREES * DOP_LEAVES,))
    keys_scratch: qd.Tensor = field(qd.u64, shape=(DOP_TREES * DOP_LEAVES,))
    sort_scratch: qd.Tensor = field(qd.u32, shape=(SORT_SLOTS,), labels=DEVICE_RADIX_SORT)
    sort_hist: qd.Tensor = field(int, shape=(SORT_CHUNKS, 256), labels=PER_TREE_RADIX_SORT)
    n_keys: qd.Tensor = field(int, shape=())
    nodes_parent: qd.Tensor = field(int, shape=(DOP_NODES, DOP_TREES))
    nodes_fitted: qd.Tensor = field(int, shape=(DOP_LEAVES - 1, DOP_TREES))
    fit_frontier: qd.Tensor = field(int, shape=(2, DOP_LEAVES - 1, DOP_TREES), labels=BLOCK_SWEEP_FIT)
    # The extent of the leaf bounds of each tree along the coordinate axes, which scales its morton codes, folded from
    # the extents of strided slices of its leaves
    lanes_lower: qd.Tensor = field((qd.f32, 3), shape=(EXTENT_LANES, DOP_TREES))
    lanes_upper: qd.Tensor = field((qd.f32, 3), shape=(EXTENT_LANES, DOP_TREES))
    trees_lower: qd.Tensor = field((qd.f32, 3), shape=(DOP_TREES,))
    trees_upper: qd.Tensor = field((qd.f32, 3), shape=(DOP_TREES,))


class DOPBVHQueries(Data, kind=DataKind.SCRATCH, shape=(DOP_QUERIES, DOP_TREES)):
    """
    The queries of a set of trees: query i_q of tree i_t at [i_q, i_t], its bound, its gap and its tag (see the module
    docstring), and the queries that run.

    `active` lists the queries that run as flat indices i_t * n_queries + i_q, its first `n_active` entries alone, so that
    a caller drops the queries it knows to meet nothing.
    """

    lower: qd.Tensor = field((qd.f32, DOP_AXES))
    upper: qd.Tensor = field((qd.f32, DOP_AXES))
    gap: qd.Tensor = field(float)
    tag: qd.Tensor = field(int)
    active: qd.Tensor = field(int, shape=(DOP_TREES * DOP_QUERIES,))
    n_active: qd.Tensor = field(int, shape=())


class DOPBVHHits(Data, kind=DataKind.SCRATCH):
    """
    The hits (tree, query, leaf) a query of a set of trees appends, their count, and the count of truncated queries.

    The count grows past the capacity on overflow, the hits beyond it dropped, so that a caller reading it knows the
    capacity a complete query needs. A query on the lanes of a block stops short when its frontier fills the shared
    memory of its block (see func_dop_bvh_query_block): it counts in `n_truncated`, and the caller reruns the queries
    on one thread each, which is complete.
    """

    hits: qd.Tensor = field((int, 3), shape=(DOP_HITS,))
    count: qd.Tensor = field(int, shape=())
    n_truncated: qd.Tensor = field(int, shape=())


class DOPBVHData(NamedTuple):
    """A set of 14-DOP trees: its state and the static config of its kernels."""

    state: DOPBVHState
    config: DOPBVHConfig


def get_dop_bvh_config(n_trees: int, n_leaves: int, is_block_query: bool | None = None) -> DOPBVHConfig:
    """
    Return the static config of a set of n_trees 14-DOP trees over n_leaves leaves each.

    `is_block_query` selects the schedule of a query, on the 32 lanes of a block or on one thread. By default the
    queries run on the lanes of a block where the lanes vote (CUDA, AMDGPU). The thread schedule is the one that never
    stops short (see DOPBVHHits).
    """
    if is_block_query is None:
        is_block_query = gs.backend in (gs.cuda, gs.amdgpu)
    elif is_block_query and gs.backend not in (gs.cuda, gs.amdgpu):
        gs.raise_exception("The block schedule of a 14-DOP query runs on the backends whose lanes vote (CUDA, AMDGPU).")
    n_keys = n_trees * n_leaves
    leaf_bits = max(1, (n_leaves - 1).bit_length())
    key_bits = (n_trees - 1).bit_length() + MORTON_BITS + leaf_bits
    if key_bits > 64:
        gs.raise_exception(f"A set of {n_trees} trees over {n_leaves} leaves needs {key_bits} key bits, above 64.")
    log256_max_n = 1
    while 256**log256_max_n < n_keys:
        log256_max_n += 1
    if gs.backend == gs.cpu:
        is_rank = n_leaves <= RANK_SORT_MAX_LEAVES_CPU
    else:
        is_rank = n_trees * n_leaves**3 <= (RANK_SORT_LEAVES_PER_CORE * get_gpu_core_count()) ** 3
    if is_rank:
        sort_kind = DOP_SORT_KIND.RANK
    elif gs.backend == gs.cpu:
        sort_kind = DOP_SORT_KIND.PER_TREE_RADIX
    else:
        sort_kind = DOP_SORT_KIND.DEVICE_RADIX
    # FIXME: quadrants#925 - Metal lowers a fence to a control barrier, so the leaf walk, whose fences sit in divergent
    # flow, reads stale bounds there. One block per tree sweeps the nodes level by level instead, every barrier uniform.
    if gs.backend == gs.cpu:
        fit_kind = DOP_FIT_KIND.LEAF_WALK
    elif gs.backend == gs.metal:
        fit_kind = DOP_FIT_KIND.BLOCK_SWEEP
    else:
        fit_kind = DOP_FIT_KIND.FENCED_LEAF_WALK
    return DOPBVHConfig(
        morton_bits=MORTON_BITS,
        end_bit=min(64, 16 * math.ceil(key_bits / 16)),
        log256_max_n=log256_max_n,
        sort_kind=sort_kind,
        fit_kind=fit_kind,
        sort_max_chunks=min(SORT_MAX_CHUNKS, qd.cfg.cpu_max_num_threads),
        is_block_query=is_block_query,
    )


def get_dop_bvh_data(n_trees: int, n_leaves: int) -> DOPBVHData:
    """Return a set of 14-DOP trees, one per batch, over the same number of leaves each, and the config of its kernels."""
    if n_leaves < 1:
        gs.raise_exception(f"A set of 14-DOP trees needs at least one leaf per tree, got {n_leaves}.")
    config = get_dop_bvh_config(n_trees, n_leaves)
    n_sort_scratch = 0
    if config.sort_kind == DOP_SORT_KIND.DEVICE_RADIX:
        n_sort_scratch = sort_scratch_slots(n_trees * n_leaves, config.log256_max_n)
    state = DOPBVHState(
        sizes={
            DOP_TREES: n_trees,
            DOP_LEAVES: n_leaves,
            DOP_NODES: 2 * n_leaves - 1,
            SORT_SLOTS: n_sort_scratch,
            SORT_CHUNKS: n_trees * max(1, config.sort_max_chunks // n_trees),
            EXTENT_LANES: TREE_EXTENT_LANES,
        },
        unused={
            DEVICE_RADIX_SORT: config.sort_kind != DOP_SORT_KIND.DEVICE_RADIX,
            PER_TREE_RADIX_SORT: config.sort_kind != DOP_SORT_KIND.PER_TREE_RADIX,
            BLOCK_SWEEP_FIT: config.fit_kind != DOP_FIT_KIND.BLOCK_SWEEP,
        },
    )
    return DOPBVHData(state, config)


# ------------------------------------------------------------------------------------
# ------------------------------------- bounds ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def _func_next_f32(value: qd.f32, is_up: qd.template()) -> qd.f32:
    """Return the 32-bit float next to a value, above it or below it."""
    bits = qd.bit_cast(value, qd.i32)
    # Away from zero the magnitude grows by one unit, toward zero it shrinks by one
    is_away = value > 0.0
    if qd.static(not is_up):
        is_away = value < 0.0
    result = value
    if value == 0.0:
        result = qd.bit_cast(qd.i32(1), qd.f32)
        if qd.static(not is_up):
            result = -result
    elif is_away:
        result = qd.bit_cast(bits + 1, qd.f32)
    else:
        result = qd.bit_cast(bits - 1, qd.f32)
    return result


@qd.func
def func_round_f32(value, is_up: qd.template()) -> qd.f32:
    """Round a float to the 32-bit float at or above it, or at or below it."""
    result = qd.cast(value, qd.f32)
    if qd.static(is_up):
        if qd.cast(result, gs.qd_float) < value:
            result = _func_next_f32(result, True)
    elif qd.cast(result, gs.qd_float) > value:
        result = _func_next_f32(result, False)
    return result


@qd.func
def func_dop_empty():
    """Return the bounds of the empty 14-DOP, which meets nothing and leaves unchanged a 14-DOP merged with it."""
    N_AXES = qd.static(7)
    return qd.Vector([3e38] * N_AXES, dt=qd.f32), qd.Vector([-3e38] * N_AXES, dt=qd.f32)


@qd.func
def func_dop_add_point(lower: qd.types.vector(7, qd.f32), upper: qd.types.vector(7, qd.f32), point: qd.types.vector(3)):
    """Return the bounds of a 14-DOP grown to cover a point, its projections rounded outward."""
    N_AXES = qd.static(7)
    x, y, z = point[0], point[1], point[2]
    projections = qd.Vector([x, y, z, x + y + z, x + y - z, x - y + z, x - y - z], dt=gs.qd_float)
    for k in qd.static(range(N_AXES)):
        low = func_round_f32(projections[k], is_up=False)
        high = func_round_f32(projections[k], is_up=True)
        # In 32-bit precision a diagonal projection rounds twice, each time by half a unit of a partial sum, which
        # exceeds a unit of the projection under cancellation: 2^-22 of the sum of the magnitudes covers both roundings
        if qd.static(k >= 3 and gs.qd_float == qd.f32):
            slack = 2.4e-7 * (qd.abs(x) + qd.abs(y) + qd.abs(z))
            low = _func_next_f32(low - slack, False)
            high = _func_next_f32(high + slack, True)
        lower[k] = qd.min(lower[k], low)
        upper[k] = qd.max(upper[k], high)
    return lower, upper


@qd.func
def func_dop_merge(
    lower_a: qd.types.vector(7, qd.f32),
    upper_a: qd.types.vector(7, qd.f32),
    lower_b: qd.types.vector(7, qd.f32),
    upper_b: qd.types.vector(7, qd.f32),
):
    """Return the bounds of the smallest 14-DOP covering two others."""
    return qd.min(lower_a, lower_b), qd.max(upper_a, upper_b)


@qd.func
def func_dop_gaps(gap) -> qd.types.vector(2, qd.f32):
    """Return a Euclidean gap along the coordinate axes and along the diagonals, rounded up."""
    SQRT3 = qd.static(math.sqrt(3.0))
    return qd.Vector([func_round_f32(gap, is_up=True), func_round_f32(gap * SQRT3, is_up=True)], dt=qd.f32)


@qd.func
def func_dop_expand(lower: qd.types.vector(7, qd.f32), upper: qd.types.vector(7, qd.f32), radius):
    """Return the bounds of a 14-DOP grown by a Euclidean radius, sqrt(3) times as much along the diagonals."""
    N_AXES = qd.static(7)
    gaps = func_dop_gaps(radius)
    for k in qd.static(range(N_AXES)):
        gap = gaps[1]
        if qd.static(k < 3):
            gap = gaps[0]
        lower[k] = lower[k] - gap
        upper[k] = upper[k] + gap
    return lower, upper


@qd.func
def func_dop_overlap(
    lower_a: qd.types.vector(7, qd.f32),
    upper_a: qd.types.vector(7, qd.f32),
    lower_b: qd.types.vector(7, qd.f32),
    upper_b: qd.types.vector(7, qd.f32),
    gaps: qd.types.vector(2, qd.f32),
) -> bool:
    """Return whether two 14-DOP lie within a gap of each other along every axis (see func_dop_gaps)."""
    N_AXES = qd.static(7)
    is_apart = False
    for k in qd.static(range(N_AXES)):
        gap = gaps[1]
        if qd.static(k < 3):
            gap = gaps[0]
        is_apart = is_apart or (lower_b[k] - upper_a[k] >= gap) or (lower_a[k] - upper_b[k] >= gap)
    return not is_apart


# ------------------------------------------------------------------------------------
# -------------------------------------- build ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def _func_expand_bits(v: qd.u32) -> qd.u32:
    """Spread the low 10 bits of a value two bits apart."""
    v = (v * qd.u32(0x00010001)) & qd.u32(0xFF0000FF)
    v = (v | ((v & qd.u32(0x00FFFFFF)) << 8)) & qd.u32(0x0F00F00F)
    v = (v * qd.u32(0x00000011)) & qd.u32(0xC30C30C3)
    v = (v * qd.u32(0x00000005)) & qd.u32(0x49249249)
    return v


@qd.func
def _func_delta(i_t: int, i_leaf: int, j_leaf: int, state: DOPBVHState) -> int:
    """Return the common prefix length of the keys of two sorted leaves, -1 when the second one is out of range."""
    n_leaves = state.leaves.lower.shape[0]
    delta = -1
    if j_leaf >= 0 and j_leaf < n_leaves:
        key_i = state.leaves_keys[i_t * n_leaves + i_leaf]
        key_j = state.leaves_keys[i_t * n_leaves + j_leaf]
        # The keys of one tree differ at least in their leaf bits, so the prefix is shorter than the key
        delta = qd.math.clz(key_i ^ key_j)
    return delta


@qd.func
def _func_fit_node(i_t: int, i_n: int, state: DOPBVHState):
    """Fit the bounds and the tag of an internal node to those of its two children."""
    i_left = state.tree.nodes_left[i_n, i_t]
    i_right = state.tree.nodes_right[i_n, i_t]
    state.tree.nodes_lower[i_n, i_t] = qd.min(state.tree.nodes_lower[i_left, i_t], state.tree.nodes_lower[i_right, i_t])
    state.tree.nodes_upper[i_n, i_t] = qd.max(state.tree.nodes_upper[i_left, i_t], state.tree.nodes_upper[i_right, i_t])
    tag_left = state.tree.nodes_tag[i_left, i_t]
    tag_right = state.tree.nodes_tag[i_right, i_t]
    state.tree.nodes_tag[i_n, i_t] = qd.select(tag_left == tag_right, tag_left, -1)


@qd.func
def _func_set_leaf_node(i_t: int, i_leaf: int, state: DOPBVHState):
    """Copy the bounds and the tag of a sorted leaf into its leaf node."""
    n_leaves = state.leaves.lower.shape[0]
    i_n = n_leaves - 1 + i_leaf
    i_a = state.tree.leaves_idx[i_leaf, i_t]
    state.tree.nodes_lower[i_n, i_t] = state.leaves.lower[i_a, i_t]
    state.tree.nodes_upper[i_n, i_t] = state.leaves.upper[i_a, i_t]
    state.tree.nodes_tag[i_n, i_t] = state.leaves.tag[i_a, i_t]


@qd.func
def _func_fit_ancestors(i_t: int, i_leaf: int, state: DOPBVHState, config: qd.template()):
    """Walk from a sorted leaf to the root, fitting each node whose two children are fitted.

    The leaves of a tree walk at once. The second child to arrive at a node fits it and goes on, the first one stops, so
    every node is fitted once after both children.
    """
    n_leaves = state.leaves.lower.shape[0]
    i_n = n_leaves - 1 + i_leaf
    _func_set_leaf_node(i_t, i_leaf, state)
    for _ in range(n_leaves):
        i_parent = state.nodes_parent[i_n, i_t]
        if i_parent < 0:
            break
        # The bounds of this node must be visible to the sibling's thread before the arrival it may win: a device fence
        # on each side of the relaxed atomic of the GPU, the sequentially consistent atomic alone on the CPU
        if qd.static(config.fit_kind == DOP_FIT_KIND.FENCED_LEAF_WALK):
            qd.simt.grid.mem_fence()
        n_arrived = qd.atomic_add(state.nodes_fitted[i_parent, i_t], 1)
        if qd.static(config.fit_kind == DOP_FIT_KIND.FENCED_LEAF_WALK):
            qd.simt.grid.mem_fence()
        if n_arrived == 0:
            break
        _func_fit_node(i_t, i_parent, state)
        i_n = i_parent


@qd.func
def _func_push_fit_parent(i_t: int, i_n: int, i_next: int, is_active: bool, sh_frontier_count, state: DOPBVHState):
    """Count the arrival of a node at its parent, and push the parent into the next frontier on the second arrival.

    An inactive lane takes part with a zero increment, so that every lane of the block runs the same atomics.
    """
    # FIXME: quadrants#925 - an atomic carries a memory barrier that Metal lowers to a control barrier, so the atomics
    # stay in uniform control flow: an inactive lane adds 0 to a valid slot instead of skipping the call.
    i_parent = -1
    if is_active:
        i_parent = state.nodes_parent[i_n, i_t]
    n_arrival = qd.i32(i_parent >= 0)
    n_arrived = qd.atomic_add(state.nodes_fitted[qd.max(i_parent, 0), i_t], n_arrival)
    n_push = qd.i32(n_arrival == 1 and n_arrived == 1)
    i_slot = qd.atomic_add(sh_frontier_count[i_next], n_push)
    if n_push == 1:
        state.fit_frontier[i_next, i_slot, i_t] = i_parent


@qd.func
def _func_fit_tree_block(i_t: int, i_lane: int, state: DOPBVHState, block_dim: int):
    """Fit the node bounds of a tree by one block of threads, sweeping a frontier of ready nodes up the tree.

    A round fits the nodes both children of which were fitted by earlier rounds and pushes their parents, on their second
    child, into the frontier of the next round. The block synchronizes between rounds, so every bound a round reads was
    written before a barrier and the sweep depends on no memory ordering across threads.
    """
    # The block barrier of the SPIR-V backends orders shared memory alone, so a device-scope fence precedes each one to
    # publish the bounds, the arrival counters and the frontier, which live in device memory
    n_leaves = state.leaves.lower.shape[0]
    sh_frontier_count = qd.simt.block.SharedArray((2,), gs.qd_int)
    if i_lane == 0:
        sh_frontier_count[0] = 0
        sh_frontier_count[1] = 0
    qd.simt.block.sync()
    for i_chunk in range((n_leaves + block_dim - 1) // block_dim):
        i_leaf = i_chunk * block_dim + i_lane
        is_active = i_leaf < n_leaves
        if is_active:
            _func_set_leaf_node(i_t, i_leaf, state)
        _func_push_fit_parent(i_t, n_leaves - 1 + i_leaf, 0, is_active, sh_frontier_count, state)
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
            i_n = state.fit_frontier[i_cur, qd.min(i_slot, n_frontier - 1), i_t]
            if is_active:
                _func_fit_node(i_t, i_n, state)
            _func_push_fit_parent(i_t, i_n, 1 - i_cur, is_active, sh_frontier_count, state)
        qd.simt.grid.mem_fence()
        qd.simt.block.sync()
        # The list swept this round takes the pushes of the next one, so its count is cleared behind a barrier
        if i_lane == 0:
            sh_frontier_count[i_cur] = 0
        qd.simt.block.sync()


@qd.func(requires_top_level=True)
def func_build_dop_bvh(state: DOPBVHState, config: qd.template()):
    """Build every tree of the set from the bounds and tags of its leaves, at the top level of a kernel.

    The passes run in order: the extent of each tree, the morton key of each leaf, the sort of the keys, the leaf index of
    each sorted slot, the radix tree, and the node bounds and tags.
    """
    n_leaves = state.leaves.lower.shape[0]
    n_trees = state.leaves.lower.shape[1]
    # The device-wide radix sort reads its key count on the device
    state.n_keys[()] = n_trees * n_leaves

    # Extent of each tree over the bounds of its leaves along the coordinate axes: each lane folds a strided slice of the
    # leaves, then one thread per tree folds the lanes, so the reduction needs no floating-point atomic
    n_lanes = state.lanes_lower.shape[0]
    for i_lane, i_t in qd.ndrange(n_lanes, n_trees):
        lane_lower = qd.Vector([3e38] * 3, dt=qd.f32)
        lane_upper = qd.Vector([-3e38] * 3, dt=qd.f32)
        for i_chunk in range((n_leaves + n_lanes - 1) // n_lanes):
            i_a = i_chunk * n_lanes + i_lane
            if i_a < n_leaves:
                leaf_lower = state.leaves.lower[i_a, i_t]
                leaf_upper = state.leaves.upper[i_a, i_t]
                for k in qd.static(range(3)):
                    lane_lower[k] = qd.min(lane_lower[k], leaf_lower[k])
                    lane_upper[k] = qd.max(lane_upper[k], leaf_upper[k])
        state.lanes_lower[i_lane, i_t] = lane_lower
        state.lanes_upper[i_lane, i_t] = lane_upper
    for i_t in range(n_trees):
        tree_lower = state.lanes_lower[0, i_t]
        tree_upper = state.lanes_upper[0, i_t]
        for i_lane in range(1, n_lanes):
            tree_lower = qd.min(tree_lower, state.lanes_lower[i_lane, i_t])
            tree_upper = qd.max(tree_upper, state.lanes_upper[i_lane, i_t])
        state.trees_lower[i_t] = tree_lower
        state.trees_upper[i_t] = tree_upper

    # Key of each leaf: the tree, then the morton code of the leaf center in the extent of its tree scaled to the unit
    # cube, then the leaf index
    for i_a, i_t in qd.ndrange(n_leaves, n_trees):
        extent = qd.cast(state.trees_upper[i_t] - state.trees_lower[i_t], gs.qd_float)
        code = qd.u32(0)
        for k in qd.static(range(3)):
            center = 0.5 * (
                qd.cast(state.leaves.lower[i_a, i_t][k], gs.qd_float)
                + qd.cast(state.leaves.upper[i_a, i_t][k], gs.qd_float)
            )
            scaled = (center - qd.cast(state.trees_lower[i_t][k], gs.qd_float)) / qd.max(extent[k], 1e-30)
            cell = qd.min(qd.max(scaled * 1024.0, 0.0), 1023.0)
            code = code | (_func_expand_bits(qd.u32(cell)) << qd.u32(2 - k))
        # The bits a leaf index takes, at least one
        leaf_bits = qd.max(1, 32 - qd.math.clz(qd.u32(n_leaves - 1)))
        key = (qd.u64(code) << leaf_bits) | qd.u64(i_a)
        state.leaves_keys[i_t * n_leaves + i_a] = (qd.u64(i_t) << (qd.static(config.morton_bits) + leaf_bits)) | key

    # Sort of the keys, every key unique through its leaf bits so the order is total
    if qd.static(config.sort_kind == DOP_SORT_KIND.DEVICE_RADIX):
        radix_sort(
            state.leaves_keys,
            state.keys_scratch,
            state.leaves_keys,
            state.keys_scratch,
            state.sort_scratch,
            state.n_keys,
            key_dtype=qd.u64,
            has_values=False,
            end_bit=qd.static(config.end_bit),
            log256_max_n=qd.static(config.log256_max_n),
        )
    elif qd.static(config.sort_kind == DOP_SORT_KIND.RANK):
        # Each leaf ranks itself among the keys of its tree
        for i_t, i_a in qd.ndrange(n_trees, n_leaves):
            key = state.leaves_keys[i_t * n_leaves + i_a]
            rank = 0
            for i_o in range(n_leaves):
                if state.leaves_keys[i_t * n_leaves + i_o] < key:
                    rank += 1
            state.keys_scratch[i_t * n_leaves + rank] = key
        for i_k in range(n_trees * n_leaves):
            state.leaves_keys[i_k] = state.keys_scratch[i_k]
    else:
        # A least-significant-digit radix sort of eight bits per pass on the host backend, the leaves of each tree split
        # over chunks: the chunks count their digits and scatter their keys in parallel, one thread per tree turning the
        # counts into offsets in between. The passes ping-pong between the keys and the scratch and end in the keys. The
        # passes unroll statically: a loop of the kernel is parallel at its top level alone, so a runtime loop over the
        # passes would serialize the chunks.
        n_sort_chunks = qd.max(1, qd.min(qd.static(config.sort_max_chunks), state.sort_hist.shape[0] // n_trees))
        for i_pass in qd.static(range(qd.static(config.end_bit) // 8)):
            shift = 8 * i_pass
            for i_t, i_c in qd.ndrange(n_trees, n_sort_chunks):
                i_h = i_t * n_sort_chunks + i_c
                for i_d in range(256):
                    state.sort_hist[i_h, i_d] = 0
                for i_a in range(i_c * n_leaves // n_sort_chunks, (i_c + 1) * n_leaves // n_sort_chunks):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = state.leaves_keys[i_t * n_leaves + i_a]
                    else:
                        key = state.keys_scratch[i_t * n_leaves + i_a]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    state.sort_hist[i_h, digit] = state.sort_hist[i_h, digit] + 1
            for i_t in range(n_trees):
                total = 0
                for i_d in range(256):
                    for i_c in range(n_sort_chunks):
                        i_h = i_t * n_sort_chunks + i_c
                        count = state.sort_hist[i_h, i_d]
                        state.sort_hist[i_h, i_d] = total
                        total += count
            for i_t, i_c in qd.ndrange(n_trees, n_sort_chunks):
                i_h = i_t * n_sort_chunks + i_c
                for i_a in range(i_c * n_leaves // n_sort_chunks, (i_c + 1) * n_leaves // n_sort_chunks):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = state.leaves_keys[i_t * n_leaves + i_a]
                    else:
                        key = state.keys_scratch[i_t * n_leaves + i_a]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    slot = state.sort_hist[i_h, digit]
                    state.sort_hist[i_h, digit] = slot + 1
                    if qd.static(i_pass % 2 == 0):
                        state.keys_scratch[i_t * n_leaves + slot] = key
                    else:
                        state.leaves_keys[i_t * n_leaves + slot] = key

    # Leaf of each sorted slot, the links of the leaf nodes and the root, and the arrival counters of the fit
    for i_leaf, i_t in qd.ndrange(n_leaves, n_trees):
        leaf_bits = qd.max(1, 32 - qd.math.clz(qd.u32(n_leaves - 1)))
        leaf_mask = (qd.u64(1) << leaf_bits) - qd.u64(1)
        state.tree.leaves_idx[i_leaf, i_t] = qd.i32(state.leaves_keys[i_t * n_leaves + i_leaf] & leaf_mask)
        state.tree.nodes_left[n_leaves - 1 + i_leaf, i_t] = -1
        state.tree.nodes_right[n_leaves - 1 + i_leaf, i_t] = -1
    for i_n, i_t in qd.ndrange(state.nodes_fitted.shape[0], n_trees):
        state.nodes_fitted[i_n, i_t] = 0
    for i_t in range(n_trees):
        state.nodes_parent[0, i_t] = -1

    # Radix tree: internal node i_n owns a range of sorted leaves that starts at leaf i_n and extends toward the longer
    # key prefix, split where the prefix shared by the whole range stops (Karras 2012)
    for i_n, i_t in qd.ndrange(n_leaves - 1, n_trees):
        delta_next = _func_delta(i_t, i_n, i_n + 1, state)
        delta_prev = _func_delta(i_t, i_n, i_n - 1, state)
        direction = 1 if delta_next > delta_prev else -1
        delta_min = _func_delta(i_t, i_n, i_n - direction, state)
        # The range of the node, by a doubling then a binary search over the leaves sharing more than delta_min, each
        # bounded by the 64 bits of a key
        length_max = 2
        for _ in range(64):
            if _func_delta(i_t, i_n, i_n + length_max * direction, state) <= delta_min:
                break
            length_max *= 2
        length = 0
        step = length_max // 2
        for _ in range(64):
            if step == 0:
                break
            if _func_delta(i_t, i_n, i_n + (length + step) * direction, state) > delta_min:
                length += step
            step //= 2
        j_leaf = i_n + length * direction
        delta_node = _func_delta(i_t, i_n, j_leaf, state)
        split = 0
        step = (length + 1) // 2
        for _ in range(64):
            if step == 0:
                break
            if _func_delta(i_t, i_n, i_n + (split + step) * direction, state) > delta_node:
                split += step
            step = (step + 1) // 2 if step > 1 else 0
        gamma = i_n + split * direction + qd.min(direction, 0)
        i_left = gamma + n_leaves - 1 if qd.min(i_n, j_leaf) == gamma else gamma
        i_right = gamma + n_leaves if qd.max(i_n, j_leaf) == gamma + 1 else gamma + 1
        state.tree.nodes_left[i_n, i_t] = i_left
        state.tree.nodes_right[i_n, i_t] = i_right
        state.nodes_parent[i_left, i_t] = i_n
        state.nodes_parent[i_right, i_t] = i_n

    # Node bounds and tags, bottom-up from the leaves (see DOP_FIT_KIND for the regimes)
    if qd.static(config.fit_kind == DOP_FIT_KIND.BLOCK_SWEEP):
        FIT_BLOCK_DIM = qd.static(128)
        qd.loop_config(block_dim=FIT_BLOCK_DIM)
        for i_flat in range(n_trees * FIT_BLOCK_DIM):
            _func_fit_tree_block(i_flat // FIT_BLOCK_DIM, i_flat % FIT_BLOCK_DIM, state, FIT_BLOCK_DIM)
    else:
        for i_leaf, i_t in qd.ndrange(n_leaves, n_trees):
            _func_fit_ancestors(i_t, i_leaf, state, config)


@qd.kernel(fastcache=True)
def kernel_build_dop_bvh(state: DOPBVHState, config: qd.template()):
    """Build every tree of a set from the bounds and tags of its leaves (see func_build_dop_bvh)."""
    func_build_dop_bvh(state, config)


# ------------------------------------------------------------------------------------
# ------------------------------------- query ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def _func_is_node_met(i_t: int, i_n: int, lower, upper, gaps, tag: int, tree: DOPBVHTree) -> bool:
    """Return whether a query of bound (lower, upper), gaps and tag descends into a node (see the module docstring)."""
    node_tag = tree.nodes_tag[i_n, i_t]
    is_met = not (node_tag >= 0 and node_tag == tag)
    if is_met:
        is_met = func_dop_overlap(lower, upper, tree.nodes_lower[i_n, i_t], tree.nodes_upper[i_n, i_t], gaps)
    return is_met


@qd.func
def _func_append_hit(i_t: int, i_q: int, i_a: int, hits: DOPBVHHits):
    """Append a hit to the list, counting it whatever the capacity."""
    i_h = qd.atomic_add(hits.count[()], 1)
    if i_h < hits.hits.shape[0]:
        hits.hits[i_h] = qd.Vector([i_t, i_q, i_a], dt=gs.qd_int)


@qd.func
def func_dop_bvh_query_thread(i_t: int, i_q: int, queries: DOPBVHQueries, tree: DOPBVHTree, hits: DOPBVHHits):
    """Append the leaves of tree i_t that query i_q meets, walking the tree depth first on one thread."""
    # One pending sibling per level of each tree: a tree splits one key bit per level below the bits every key of the set
    # shares, so its depth is at most the morton bits plus the leaf bits, well under 64
    STACK_SIZE = qd.static(64)
    n_leaves = tree.leaves_idx.shape[0]
    lower = queries.lower[i_q, i_t]
    upper = queries.upper[i_q, i_t]
    gaps = func_dop_gaps(queries.gap[i_q, i_t])
    tag = queries.tag[i_q, i_t]
    stack = qd.Vector.zero(gs.qd_int, STACK_SIZE)
    n_stack = 0
    if _func_is_node_met(i_t, 0, lower, upper, gaps, tag, tree):
        stack[0] = 0
        n_stack = 1
    for _ in range(2 * n_leaves):
        if n_stack == 0:
            break
        n_stack -= 1
        i_n = stack[n_stack]
        i_left = tree.nodes_left[i_n, i_t]
        if i_left < 0:
            _func_append_hit(i_t, i_q, tree.leaves_idx[i_n - (n_leaves - 1), i_t], hits)
        else:
            i_right = tree.nodes_right[i_n, i_t]
            if _func_is_node_met(i_t, i_right, lower, upper, gaps, tag, tree):
                stack[n_stack] = i_right
                n_stack += 1
            if _func_is_node_met(i_t, i_left, lower, upper, gaps, tag, tree):
                stack[n_stack] = i_left
                n_stack += 1


@qd.func
def func_dop_bvh_query_block(
    i_t: int,
    i_q: int,
    i_warp: int,
    lane: int,
    sh_stack,
    queries: DOPBVHQueries,
    tree: DOPBVHTree,
    hits: DOPBVHHits,
):
    """Append the leaves of tree i_t that query i_q meets, on the 32 lanes of a block sharing a frontier of nodes.

    Every lane of the block calls it with the same query, row i_warp of the shared array `sh_stack` being the frontier
    of the block. Each round pops up to 32 nodes, one per lane, tests their two children, pushes the internal ones the
    query meets (their slots taken by a ballot of the lanes) and appends the leaves it meets, one atomic per block and
    round. A query whose frontier fills `sh_stack` stops short and counts in `hits.n_truncated`.
    """
    WARP_SIZE = qd.static(32)
    n_leaves = tree.leaves_idx.shape[0]
    stack_size = sh_stack.shape[1]
    lower = queries.lower[i_q, i_t]
    upper = queries.upper[i_q, i_t]
    gaps = func_dop_gaps(queries.gap[i_q, i_t])
    tag = queries.tag[i_q, i_t]
    lanes_below = (qd.u32(1) << qd.u32(lane)) - qd.u32(1)
    # The root as the frontier of the first round, or as the lone leaf of a tree of one leaf
    n_top = 0
    if n_leaves == 1:
        is_hit = lane == 0 and _func_is_node_met(i_t, 0, lower, upper, gaps, tag, tree)
        if is_hit:
            _func_append_hit(i_t, i_q, tree.leaves_idx[0, i_t], hits)
    else:
        if lane == 0:
            sh_stack[i_warp, 0] = 0
        n_top = 1
    qd.simt.subgroup.sync()
    for _ in range(2 * n_leaves):
        # A round pops at most as many nodes as the frontier has free slots, since each one may push two children
        n_pop = qd.min(qd.min(n_top, WARP_SIZE), stack_size - n_top)
        if n_pop == 0:
            break
        is_active = lane < n_pop
        i_n = 0
        if is_active:
            i_n = sh_stack[i_warp, n_top - 1 - lane]
        qd.simt.subgroup.sync()
        n_top -= n_pop
        children = qd.Vector([0, 0], dt=gs.qd_int)
        if is_active:
            children = qd.Vector([tree.nodes_left[i_n, i_t], tree.nodes_right[i_n, i_t]], dt=gs.qd_int)
        for r in qd.static(range(2)):
            i_child = children[r]
            is_met = is_active and _func_is_node_met(i_t, i_child, lower, upper, gaps, tag, tree)
            is_leaf = is_met and tree.nodes_left[i_child, i_t] < 0
            is_push = is_met and not is_leaf
            pushes = qd_block_ballot(is_push)
            if is_push:
                sh_stack[i_warp, n_top + qd.cast(qd.math.popcnt(pushes & lanes_below), gs.qd_int)] = i_child
            n_top += qd.cast(qd.math.popcnt(pushes), gs.qd_int)
            # The leaves the lanes meet, appended at slots one atomic of the first lane reserves
            leaves = qd_block_ballot(is_leaf)
            i_base = 0
            if lane == 0 and leaves != 0:
                i_base = qd.atomic_add(hits.count[()], qd.cast(qd.math.popcnt(leaves), gs.qd_int))
            i_base = qd_block_shuffle(i_base, 0)
            if is_leaf:
                i_h = i_base + qd.cast(qd.math.popcnt(leaves & lanes_below), gs.qd_int)
                if i_h < hits.hits.shape[0]:
                    i_a = tree.leaves_idx[i_child - (n_leaves - 1), i_t]
                    hits.hits[i_h] = qd.Vector([i_t, i_q, i_a], dt=gs.qd_int)
            qd.simt.subgroup.sync()
    # A full frontier leaves nodes unvisited
    if n_top > 0 and lane == 0:
        qd.atomic_add(hits.n_truncated[()], 1)


@qd.func(requires_top_level=True)
def func_query_dop_bvh(queries: DOPBVHQueries, tree: DOPBVHTree, hits: DOPBVHHits, config: qd.template()):
    """Append the leaves every active query meets in the tree of its batch, at the top level of a kernel (see
    DOPBVHQueries and DOPBVHHits)."""
    # A block of 8 warps, each one holding a frontier of 1024 nodes
    BLOCK_DIM = qd.static(256)
    WARP_SIZE = qd.static(32)
    STACK_SIZE = qd.static(1024)
    n_queries = queries.lower.shape[0]
    for _ in range(1):
        hits.count[()] = 0
        hits.n_truncated[()] = 0
    if qd.static(config.is_block_query):
        qd.loop_config(block_dim=BLOCK_DIM)
        for i_flat in range(queries.n_active[()] * WARP_SIZE):
            sh_stack = qd.simt.block.SharedArray((BLOCK_DIM // WARP_SIZE, STACK_SIZE), gs.qd_int)
            i_flat_query = queries.active[i_flat // WARP_SIZE]
            func_dop_bvh_query_block(
                i_flat_query // n_queries,
                i_flat_query % n_queries,
                i_flat % BLOCK_DIM // WARP_SIZE,
                i_flat % WARP_SIZE,
                sh_stack,
                queries,
                tree,
                hits,
            )
    else:
        for i_active in range(queries.n_active[()]):
            i_flat_query = queries.active[i_active]
            func_dop_bvh_query_thread(i_flat_query // n_queries, i_flat_query % n_queries, queries, tree, hits)


@qd.kernel(fastcache=True)
def kernel_query_dop_bvh(queries: DOPBVHQueries, tree: DOPBVHTree, hits: DOPBVHHits, config: qd.template()):
    """Append the leaves every active query meets in the tree of its batch (see func_query_dop_bvh)."""
    func_query_dop_bvh(queries, tree, hits, config)
