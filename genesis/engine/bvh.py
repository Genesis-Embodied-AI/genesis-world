"""Linear bounding volume hierarchies (LBVH) over sets of axis-aligned boxes: build and box queries.

A set holds one tree per batch over the same number of leaves. The caller writes the box of each leaf into the state,
builds the set, and traverses the trees it gets back (see BVHTreeState in array_class). The build follows
Karras (2012, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees").

A leaf carries two indices: i_a names it by its box, in the order the caller wrote them, and i_leaf by its slot in the
sorted order the tree is laid out in, the one leaves_idx maps back to i_a.
"""

import math
import os
from typing import NamedTuple

import quadrants as qd
from quadrants.algorithms import sort as radix_sort
from quadrants.algorithms import sort_scratch_slots

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.constants import backend as gs_backend

# Bits of a morton code: three interleaved 10-bit coordinates.
MORTON_BITS = 30
# Up to this leaf count every leaf ranks itself among the keys of its tree in one pass, quadratic per tree; above it
# a radix sort takes over, the device-wide one of quadrants on the GPU, one per tree on the CPU. The crossover sits
# where the quadratic pass stops hiding behind the fixed cost of the radix passes, measured per backend.
RANK_SORT_MAX_LEAVES = {gs_backend.cpu: 256, gs_backend.metal: 1024, gs_backend.cuda: 2048}
# Lanes reducing the extent of one tree, each over a strided slice of its leaves, folded by one thread per tree.
EXTENT_LANES = 64
# Most chunks of the leaves of one tree the host radix sort counts and scatters in parallel (see func_build_bvh). The
# chunks bring parallelism to a set of few trees; a set of many trees is parallel over its trees already, and every
# chunk costs a histogram of 256 bins per pass.
SORT_MAX_CHUNKS = 64


class BVHData(NamedTuple):
    """A set of linear bounding volume hierarchies (LBVH): its state and the static config of its kernels."""

    state: array_class.BVHState
    config: array_class.BVHStaticConfig


def get_bvh_data(n_trees: int, n_leaves: int) -> BVHData:
    """Return a set of trees, one per batch, over the same number of leaves each."""
    if n_leaves < 1:
        gs.raise_exception(f"A BVH set needs at least one leaf per tree, got {n_leaves}.")

    # The config: the bits of the leaf keys a sort orders, and the sort the backend and the tree size take
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
    if n_leaves <= RANK_SORT_MAX_LEAVES.get(gs.backend, RANK_SORT_MAX_LEAVES[gs_backend.cuda]):
        sort_kind = array_class.BVH_SORT_KIND.RANK
    elif gs.backend == gs.cpu:
        sort_kind = array_class.BVH_SORT_KIND.PER_TREE_RADIX
    else:
        sort_kind = array_class.BVH_SORT_KIND.DEVICE_RADIX
    # FIXME: quadrants#925 - Metal lowers a fence to a control barrier, so the leaf walk, whose fences sit in divergent
    # flow, reads stale boxes there. One block per tree sweeps the nodes level by level instead, every barrier uniform.
    if gs.backend == gs.cpu:
        fit_kind = array_class.BVH_FIT_KIND.LEAF_WALK
    elif gs.backend == gs.metal:
        fit_kind = array_class.BVH_FIT_KIND.BLOCK_SWEEP
    else:
        fit_kind = array_class.BVH_FIT_KIND.FENCED_LEAF_WALK
    bvh_config = array_class.BVHStaticConfig(
        morton_bits=MORTON_BITS, end_bit=end_bit, log256_max_n=log256_max_n, sort_kind=sort_kind, fit_kind=fit_kind
    )

    # The state: the scratch of the device-wide radix sort is sized by quadrants, the lanes of the extent are fixed
    n_sort_scratch = (
        sort_scratch_slots(n_keys, log256_max_n) if sort_kind == array_class.BVH_SORT_KIND.DEVICE_RADIX else 0
    )
    n_sort_chunks = max(1, min(SORT_MAX_CHUNKS, (os.cpu_count() or 1) // n_trees))
    bvh_state = array_class.get_bvh_state(n_trees, n_leaves, n_sort_scratch, n_sort_chunks, EXTENT_LANES, bvh_config)

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
def _func_delta(i_t: int, i_leaf: int, j_leaf: int, bvh_state: array_class.BVHState) -> int:
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
def _func_fit_ancestors(i_t: int, i_leaf: int, bvh_state: array_class.BVHState, bvh_config: qd.template()):
    """Walk from a sorted leaf to the root, fitting each node whose two children are fitted.

    The leaves of a tree walk at once. The second child to arrive at a node fits it and goes on, the first one stops,
    so every node is fitted once after both children.
    """
    n_leaves = bvh_state.leaves.aabbs_min.shape[1]
    i_n = n_leaves - 1 + i_leaf
    i_a = bvh_state.tree.leaves_idx[i_t, i_leaf]
    bvh_state.tree.nodes_min[i_t, i_n] = bvh_state.leaves.aabbs_min[i_t, i_a]
    bvh_state.tree.nodes_max[i_t, i_n] = bvh_state.leaves.aabbs_max[i_t, i_a]
    for _ in range(n_leaves):
        i_parent = bvh_state.nodes_parent[i_t, i_n]
        if i_parent < 0:
            break
        # The box of this node must be visible to the sibling's thread before the arrival it may win: a device fence
        # on each side of the relaxed atomic of the GPU, the sequentially consistent atomic alone on the CPU
        if qd.static(bvh_config.fit_kind == array_class.BVH_FIT_KIND.FENCED_LEAF_WALK):
            qd.simt.grid.mem_fence()
        n_arrived = qd.atomic_add(bvh_state.nodes_fitted[i_t, i_parent], 1)
        if qd.static(bvh_config.fit_kind == array_class.BVH_FIT_KIND.FENCED_LEAF_WALK):
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
        i_n = i_parent


@qd.func
def _func_push_fit_parent(
    i_t: int, i_n: int, i_next: int, is_active: bool, sh_frontier_count, bvh_state: array_class.BVHState
):
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
def _func_fit_tree_block(i_t: int, i_lane: int, bvh_state: array_class.BVHState, block_dim: int):
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
            _func_push_fit_parent(i_t, i_n, 1 - i_cur, is_active, sh_frontier_count, bvh_state)
        qd.simt.grid.mem_fence()
        qd.simt.block.sync()
        # The list swept this round takes the pushes of the next one, so its count is cleared behind a barrier
        if i_lane == 0:
            sh_frontier_count[i_cur] = 0
        qd.simt.block.sync()


@qd.func
def func_build_bvh(bvh_state: array_class.BVHState, bvh_config: qd.template(), eps: float):
    """Build every tree of the set from the boxes of its leaves, at the top level of a kernel.

    The passes run in order: the extent of each tree, the morton key of each leaf, the sort of the keys, the leaf index
    of each sorted slot, the radix tree and the node boxes.
    """
    n_trees, n_leaves = bvh_state.leaves.aabbs_min.shape

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
    if qd.static(bvh_config.sort_kind == array_class.BVH_SORT_KIND.DEVICE_RADIX):
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
    elif qd.static(bvh_config.sort_kind == array_class.BVH_SORT_KIND.RANK):
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
        # runtime loop over the passes would serialize the chunks.
        n_sort_chunks = bvh_state.sort_hist.shape[1]
        for i_pass in qd.static(range(qd.static(bvh_config.end_bit) // 8)):
            shift = 8 * i_pass
            for i_t, i_c in qd.ndrange(n_trees, n_sort_chunks):
                for i_d in range(256):
                    bvh_state.sort_hist[i_t, i_c, i_d] = 0
                for i_a in range(i_c * n_leaves // n_sort_chunks, (i_c + 1) * n_leaves // n_sort_chunks):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = bvh_state.leaves_keys[i_t * n_leaves + i_a]
                    else:
                        key = bvh_state.keys_scratch[i_t * n_leaves + i_a]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    bvh_state.sort_hist[i_t, i_c, digit] += 1
            for i_t in range(n_trees):
                total = 0
                for i_d in range(256):
                    for i_c in range(n_sort_chunks):
                        count = bvh_state.sort_hist[i_t, i_c, i_d]
                        bvh_state.sort_hist[i_t, i_c, i_d] = total
                        total += count
            for i_t, i_c in qd.ndrange(n_trees, n_sort_chunks):
                for i_a in range(i_c * n_leaves // n_sort_chunks, (i_c + 1) * n_leaves // n_sort_chunks):
                    key = qd.u64(0)
                    if qd.static(i_pass % 2 == 0):
                        key = bvh_state.leaves_keys[i_t * n_leaves + i_a]
                    else:
                        key = bvh_state.keys_scratch[i_t * n_leaves + i_a]
                    digit = qd.i32((key >> shift) & qd.u64(0xFF))
                    slot = bvh_state.sort_hist[i_t, i_c, digit]
                    bvh_state.sort_hist[i_t, i_c, digit] = slot + 1
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
    if qd.static(bvh_config.fit_kind == array_class.BVH_FIT_KIND.BLOCK_SWEEP):
        FIT_BLOCK_DIM = qd.static(128)
        qd.loop_config(block_dim=FIT_BLOCK_DIM)
        for i_flat in range(n_trees * FIT_BLOCK_DIM):
            _func_fit_tree_block(i_flat // FIT_BLOCK_DIM, i_flat % FIT_BLOCK_DIM, bvh_state, FIT_BLOCK_DIM)
    else:
        for i_t, i_leaf in qd.ndrange(n_trees, n_leaves):
            _func_fit_ancestors(i_t, i_leaf, bvh_state, bvh_config)


@qd.kernel(fastcache=True)
def build_bvh(bvh_state: array_class.BVHState, bvh_config: qd.template(), eps: float):
    """Build every tree of the set from the boxes of its leaves.

    The kernel runs func_build_bvh for a caller outside a kernel.
    """
    func_build_bvh(bvh_state, bvh_config, eps)


@qd.func
def func_no_filter(i_t: int, i_a: int, i_q: int, filter_ctx: qd.template()) -> bool:
    """Keep every pair a box query finds."""
    return False


@qd.func
def func_bvh_query_aabb(
    i_t: int,
    i_q: int,
    query_min,
    query_max,
    bvh_tree_state: array_class.BVHTreeState,
    query_results: array_class.BVHQueryResults,
    filter: qd.template(),
    filter_ctx: qd.template(),
):
    """Collect the leaves of a tree whose box meets a query box, as (tree, leaf, query) triplets.

    ``filter(i_t, i_a, i_q, filter_ctx)`` drops a found leaf when it returns True. The count keeps growing past the
    capacity of the result list, the pairs beyond it dropped, so a count above the capacity reports the overflow.
    """
    n_leaves = bvh_tree_state.leaves_idx.shape[1]
    # One pending sibling per level: a tree splits one key bit per level below the bits every key of the set shares,
    # so its depth is at most the morton bits plus the leaf bits, well under this
    STACK_SIZE = qd.static(64)
    stack = qd.Vector.zero(gs.qd_int, STACK_SIZE)
    n_stack = 1
    for _ in range(2 * n_leaves):
        if n_stack == 0:
            break
        n_stack -= 1
        i_n = stack[n_stack]
        node_min = bvh_tree_state.nodes_min[i_t, i_n]
        node_max = bvh_tree_state.nodes_max[i_t, i_n]
        is_hit = True
        for j in qd.static(range(3)):
            if query_min[j] > node_max[j] or query_max[j] < node_min[j]:
                is_hit = False
        if is_hit:
            i_left = bvh_tree_state.nodes_left[i_t, i_n]
            if i_left < 0:
                i_a = bvh_tree_state.leaves_idx[i_t, i_n - (n_leaves - 1)]
                if not filter(i_t, i_a, i_q, filter_ctx):
                    i_r = qd.atomic_add(query_results.count[0], 1)
                    if i_r < query_results.triplets.shape[0]:
                        query_results.triplets[i_r] = gs.qd_ivec3(i_t, i_a, i_q)
            else:
                stack[n_stack] = bvh_tree_state.nodes_right[i_t, i_n]
                n_stack += 1
                stack[n_stack] = i_left
                n_stack += 1


@qd.func
def func_bvh_query_leaves(query_state: array_class.BVHQueryState, filter: qd.template(), filter_ctx: qd.template()):
    """Query every leaf box of a set against the tree of the same index in another set.

    See func_bvh_query_aabb for the results and their overflow.
    """
    for i_t, i_q in qd.ndrange(query_state.leaves.aabbs_min.shape[0], query_state.leaves.aabbs_min.shape[1]):
        query_min = query_state.leaves.aabbs_min[i_t, i_q]
        query_max = query_state.leaves.aabbs_max[i_t, i_q]
        func_bvh_query_aabb(i_t, i_q, query_min, query_max, query_state.tree, query_state.results, filter, filter_ctx)
