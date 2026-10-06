from __future__ import annotations

import math

import numpy as np
import quadrants as qd
from quadrants.lang.simt import block as qd_block, subgroup as qd_subgroup

from .bvh_math import aabb_overlap
from .bvh_predicate import _ee_emit_pair, _ee_pair_enabled, _node_pair_enabled
from .gpu_occupancy import cuda_resident_blocks


@qd.func
def _dual_ee_node_area(bvh: qd.template(), node):
    half = bvh.bounds_width // 2
    x = qd.f64(bvh.aabbs[node, half]) - qd.f64(bvh.aabbs[node, 0])
    y = qd.f64(bvh.aabbs[node, half + 1]) - qd.f64(bvh.aabbs[node, 1])
    z = qd.f64(bvh.aabbs[node, half + 2]) - qd.f64(bvh.aabbs[node, 2])
    return x * y + x * z + y * z


@qd.func
def _dual_ee_canonical_node_pair(left, right, output: qd.template()):
    output[0] = left
    output[1] = right
    if left > right:
        output[0] = right
        output[1] = left


@qd.func
def _dual_ee_expand_pair(
    bvh: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    left_node,
    right_node,
    children: qd.template(),
    result: qd.template(),
):
    result[0] = 0
    result[1] = 0
    result[2] = 0
    result[3] = 0
    for child in qd.static(range(3)):
        children[child, 0] = 0
        children[child, 1] = 0

    if left_node > right_node:
        swap = left_node
        left_node = right_node
        right_node = swap

    if _node_pair_enabled(body, bvh.node_body_id[left_node], bvh.node_body_id[right_node]) and aabb_overlap(
        bvh.aabbs,
        left_node,
        bvh.aabbs,
        right_node,
        bvh.bounds_width // 2,
    ):
        left_element = bvh.nodes_element[left_node]
        right_element = bvh.nodes_element[right_node]
        left_leaf = left_element != qd.u32(bvh.sentinel)
        right_leaf = right_element != qd.u32(bvh.sentinel)

        if left_leaf and right_leaf:
            if left_element != right_element:
                edge_a = qd.i32(qd.min(left_element, right_element))
                edge_b = qd.i32(qd.max(left_element, right_element))
                if _ee_pair_enabled(surface, vertex, body, contact, edge_a, edge_b) != 0:
                    result[1] = 1
                    result[2] = edge_a
                    result[3] = edge_b
        elif left_node == right_node:
            left_child = qd.i32(bvh.nodes_left[left_node])
            right_child = qd.i32(bvh.nodes_right[left_node])
            children[0, 0] = left_child
            children[0, 1] = left_child
            children[1, 0] = left_child
            children[1, 1] = right_child
            children[2, 0] = right_child
            children[2, 1] = right_child
            result[0] = 3
        else:
            split_left = not left_leaf and (
                right_leaf or _dual_ee_node_area(bvh, left_node) >= _dual_ee_node_area(bvh, right_node)
            )
            pair = qd.Vector.zero(qd.i32, 2)
            if split_left:
                _dual_ee_canonical_node_pair(qd.i32(bvh.nodes_left[left_node]), right_node, pair)
                children[0, 0] = pair[0]
                children[0, 1] = pair[1]
                _dual_ee_canonical_node_pair(qd.i32(bvh.nodes_right[left_node]), right_node, pair)
                children[1, 0] = pair[0]
                children[1, 1] = pair[1]
            else:
                _dual_ee_canonical_node_pair(left_node, qd.i32(bvh.nodes_left[right_node]), pair)
                children[0, 0] = pair[0]
                children[0, 1] = pair[1]
                _dual_ee_canonical_node_pair(left_node, qd.i32(bvh.nodes_right[right_node]), pair)
                children[1, 0] = pair[0]
                children[1, 1] = pair[1]
            result[0] = 2


@qd.data_oriented
class DualEEQueryState:
    def __init__(
        self,
        n_edges: int,
        frontier_levels: int,
        target_waves: float,
        max_levels: int,
    ) -> None:
        self.block_size = 256
        self.stack_capacity = 512
        self.max_levels_limit = 18
        subgroup_size = qd_subgroup.group_size()
        self.subgroups_per_block = self.block_size // subgroup_size
        # Pinned Quadrants 459a3eb57 CUDA profile for dual_ee_dfs_range_for.
        # This is launch-resource metadata, not a scene/config parameter.
        self.frontier_registers_per_thread = 64
        self.dfs_registers_per_thread = 64
        self.frontier_blocks = cuda_resident_blocks(
            self.block_size,
            0,
            self.frontier_registers_per_thread,
        )
        self.frontier_threads = self.frontier_blocks * self.block_size
        dfs_shared_bytes = self.subgroups_per_block * self.stack_capacity * 2 * np.dtype(np.uint32).itemsize
        self.dfs_blocks = cuda_resident_blocks(
            self.block_size,
            dfs_shared_bytes,
            self.dfs_registers_per_thread,
        )
        self.resident_dfs_warps_value = self.dfs_blocks * self.subgroups_per_block
        self.dfs_threads = self.resident_dfs_warps_value * subgroup_size
        self.overflow_frontier_bit = 1
        self.overflow_stack_bit = 2
        if not 0 <= frontier_levels <= self.max_levels_limit:
            raise ValueError(f"bvh/dual/frontier_levels must be in [0, {self.max_levels_limit}]")
        if target_waves < 1.0:
            raise ValueError("bvh/dual/target_waves must be >= 1")
        if not 1 <= max_levels <= self.max_levels_limit:
            raise ValueError(f"bvh/dual/max_levels must be in [1, {self.max_levels_limit}]")

        capacity = max(4 * n_edges, 1)
        self.frontier_a = qd.ndarray(qd.u32, shape=(capacity, 2))
        self.frontier_b = qd.ndarray(qd.u32, shape=(capacity, 2))
        self.frontier_count_a = qd.ndarray(qd.i32, shape=())
        self.frontier_count_b = qd.ndarray(qd.i32, shape=())
        self.next_task = qd.ndarray(qd.i32, shape=())
        self.overflow_bits = qd.ndarray(qd.i32, shape=())
        self.required_frontier = qd.ndarray(qd.i32, shape=())
        self.current_level = qd.ndarray(qd.i32, shape=())
        self.current_parity = qd.ndarray(qd.i32, shape=())
        self.loop_cond = qd.ndarray(qd.i32, shape=())
        self.target_frontier = qd.ndarray(qd.i32, shape=())
        self.selected_count = qd.ndarray(qd.i32, shape=())
        self.selected_parity = qd.ndarray(qd.i32, shape=())
        self.config_frontier_levels = qd.ndarray(qd.i32, shape=())
        self.config_target_waves = qd.ndarray(qd.f64, shape=())
        self.config_max_levels = qd.ndarray(qd.i32, shape=())
        self.frontier_capacity = qd.ndarray(qd.i32, shape=())
        self.resident_dfs_warps = qd.ndarray(qd.i32, shape=())

        self.config_frontier_levels.from_numpy(np.array(frontier_levels, dtype=np.int32))
        self.config_target_waves.from_numpy(np.array(target_waves, dtype=np.float64))
        self.config_max_levels.from_numpy(np.array(max_levels, dtype=np.int32))
        self.frontier_capacity.from_numpy(np.array(capacity, dtype=np.int32))
        self.resident_dfs_warps.from_numpy(np.array(self.resident_dfs_warps_value, dtype=np.int32))

    def handle_overflow(self) -> bool:
        bits = int(self.overflow_bits.to_numpy())
        if bits == 0:
            return False
        if bits & self.overflow_stack_bit:
            raise RuntimeError("dual EE warp-local DFS stack exhausted; increase stack_capacity")

        required = int(self.required_frontier.to_numpy())
        current = self.frontier_a.shape[0]
        capacity = max(required, math.ceil(current * 1.2))
        self.frontier_a = qd.ndarray(qd.u32, shape=(capacity, 2))
        self.frontier_b = qd.ndarray(qd.u32, shape=(capacity, 2))
        self.frontier_capacity.from_numpy(np.array(capacity, dtype=np.int32))
        self.overflow_bits.from_numpy(np.array(0, dtype=np.int32))
        self.required_frontier.from_numpy(np.array(0, dtype=np.int32))
        return True

    @qd.func(requires_top_level=True)
    def reset(self, bvh: qd.template(), n_pairs: qd.template()):
        qd.loop_config(name="dual_ee_reset_query")
        for _ in range(1):
            n_pairs[()] = 0
            self.frontier_count_a[()] = 0
            self.frontier_count_b[()] = 0
            self.next_task[()] = 0
            self.overflow_bits[()] = 0
            self.required_frontier[()] = 0
            self.current_level[()] = 0
            self.current_parity[()] = 0
            self.target_frontier[()] = 0
            self.selected_count[()] = 0
            self.selected_parity[()] = 0

            tree_n = bvh.n_prims[()]
            max_levels = qd.min(qd.max(self.config_max_levels[()], 1), self.max_levels_limit)
            if self.config_frontier_levels[()] <= 0:
                target_from_waves = qd.i32(
                    qd.ceil(qd.max(self.config_target_waves[()], 1.0) * qd.f64(qd.max(self.resident_dfs_warps[()], 1)))
                )
                self.target_frontier[()] = qd.min(tree_n, target_from_waves)
            if tree_n <= 1 or max_levels <= 0:
                self.loop_cond[()] = 0
            else:
                self.frontier_count_a[()] = 1
                self.frontier_a[0, 0] = qd.u32(0)
                self.frontier_a[0, 1] = qd.u32(0)
                self.loop_cond[()] = 1

    @qd.func(requires_top_level=True)
    def expand_frontier(
        self,
        bvh: qd.template(),
        surface: qd.template(),
        vertex: qd.template(),
        body: qd.template(),
        contact: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs,
        overflow_flag: qd.template(),
        input_a: qd.template(),
    ):
        group_size = qd_subgroup.group_size()
        n = qd.i32(0)
        if qd.static(input_a):
            n = self.frontier_count_a[()]
        else:
            n = self.frontier_count_b[()]
        padded = ((n + group_size - 1) // group_size) * group_size
        qd.loop_config(name="dual_ee_frontier_expand", block_dim=self.block_size)
        for worker in range(self.frontier_threads):
            index = worker
            while index < padded:
                lane = qd_subgroup.invocation_id()
                active = self.loop_cond[()] != 0 and index < n
                children = qd.Matrix.zero(qd.i32, 3, 2)
                result = qd.Vector.zero(qd.i32, 4)
                left_node = qd.i32(0)
                right_node = qd.i32(0)
                if active:
                    if qd.static(input_a):
                        left_node = qd.i32(self.frontier_a[index, 0])
                        right_node = qd.i32(self.frontier_a[index, 1])
                    else:
                        left_node = qd.i32(self.frontier_b[index, 0])
                        right_node = qd.i32(self.frontier_b[index, 1])
                    _dual_ee_expand_pair(
                        bvh,
                        surface,
                        vertex,
                        body,
                        contact,
                        left_node,
                        right_node,
                        children,
                        result,
                    )

                prefix = qd_subgroup.exclusive_add(result[0])
                total = qd_subgroup.reduce_all_add(result[0])
                base = qd.i32(0)
                if lane == 0 and total > 0:
                    if qd.static(input_a):
                        base = qd.atomic_add(self.frontier_count_b[()], total)
                    else:
                        base = qd.atomic_add(self.frontier_count_a[()], total)
                base = qd_subgroup.broadcast_first(base)
                for child in qd.static(range(3)):
                    if active and child < result[0]:
                        output = base + prefix + child
                        if output < self.frontier_capacity[()]:
                            if qd.static(input_a):
                                self.frontier_b[output, 0] = qd.u32(children[child, 0])
                                self.frontier_b[output, 1] = qd.u32(children[child, 1])
                            else:
                                self.frontier_a[output, 0] = qd.u32(children[child, 0])
                                self.frontier_a[output, 1] = qd.u32(children[child, 1])

                _ee_emit_pair(
                    active and result[1] != 0,
                    result[2],
                    result[3],
                    pairs,
                    n_pairs,
                    max_pairs,
                    overflow_flag,
                )
                index = index + self.frontier_threads

    @qd.func(requires_top_level=True)
    def prepare_frontier(self, overflow_flag: qd.template(), frontier_in_a: qd.template()):
        qd.loop_config(name="dual_ee_prepare_frontier")
        for _ in range(1):
            if self.loop_cond[()] != 0:
                produced = qd.i32(0)
                if qd.static(frontier_in_a):
                    produced = self.frontier_count_a[()]
                    self.frontier_count_b[()] = 0
                    self.current_parity[()] = 0
                else:
                    produced = self.frontier_count_b[()]
                    self.frontier_count_a[()] = 0
                    self.current_parity[()] = 1

                if produced > self.frontier_capacity[()]:
                    self.required_frontier[()] = produced
                    self.overflow_bits[()] = self.overflow_bits[()] | self.overflow_frontier_bit
                    overflow_flag[()] = 1

                level = self.current_level[()] + 1
                self.current_level[()] = level
                configured_levels = self.config_frontier_levels[()]
                max_levels = qd.min(qd.max(self.config_max_levels[()], 1), self.max_levels_limit)
                reached_target = False
                if configured_levels > 0:
                    reached_target = level >= qd.min(configured_levels, max_levels)
                else:
                    reached_target = produced >= self.target_frontier[()]
                stop = reached_target or produced <= 0 or level >= max_levels or self.overflow_bits[()] != 0
                if stop:
                    self.selected_count[()] = produced
                    if qd.static(frontier_in_a):
                        self.selected_parity[()] = 0
                        self.frontier_count_a[()] = 0
                    else:
                        self.selected_parity[()] = 1
                        self.frontier_count_b[()] = 0
                self.loop_cond[()] = qd.i32(not stop)

    @qd.func(requires_top_level=True)
    def dfs(
        self,
        bvh: qd.template(),
        surface: qd.template(),
        vertex: qd.template(),
        body: qd.template(),
        contact: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs,
        overflow_flag: qd.template(),
        use_a: qd.template(),
    ):
        group_size = qd_subgroup.group_size()
        qd.loop_config(name="dual_ee_dfs", block_dim=self.block_size)
        for thread in range(self.dfs_threads):
            lane = qd_subgroup.invocation_id()
            subgroup_in_block = (thread % self.block_size) // group_size
            stack_left = qd_block.SharedArray((self.subgroups_per_block, self.stack_capacity), qd.u32)
            stack_right = qd_block.SharedArray((self.subgroups_per_block, self.stack_capacity), qd.u32)
            expected_parity = qd.i32(0 if qd.static(use_a) else 1)
            if self.selected_parity[()] == expected_parity and self.overflow_bits[()] == 0:
                while True:
                    task = qd.i32(0)
                    if lane == 0:
                        task = qd.atomic_add(self.next_task[()], 1)
                    task = qd_subgroup.broadcast_first(task)
                    if task >= self.selected_count[()]:
                        break

                    if lane == 0:
                        if qd.static(use_a):
                            stack_left[subgroup_in_block, 0] = self.frontier_a[task, 0]
                            stack_right[subgroup_in_block, 0] = self.frontier_a[task, 1]
                        else:
                            stack_left[subgroup_in_block, 0] = self.frontier_b[task, 0]
                            stack_right[subgroup_in_block, 0] = self.frontier_b[task, 1]
                    qd_subgroup.mem_fence()
                    qd_subgroup.sync()
                    top = qd.i32(1)

                    while top > 0:
                        n_pop = qd.min(top, group_size)
                        active = lane < n_pop
                        left_node = qd.i32(0)
                        right_node = qd.i32(0)
                        if active:
                            left_node = qd.i32(stack_left[subgroup_in_block, top - 1 - lane])
                            right_node = qd.i32(stack_right[subgroup_in_block, top - 1 - lane])
                        qd_subgroup.sync()
                        top = top - n_pop

                        children = qd.Matrix.zero(qd.i32, 3, 2)
                        result = qd.Vector.zero(qd.i32, 4)
                        if active:
                            _dual_ee_expand_pair(
                                bvh,
                                surface,
                                vertex,
                                body,
                                contact,
                                left_node,
                                right_node,
                                children,
                                result,
                            )

                        for child in qd.static(range(3)):
                            push = active and child < result[0]
                            push_mask = qd_subgroup.ballot(qd.i32(push))
                            push_count = qd.i32(qd.math.popcnt(push_mask))
                            if top + push_count > self.stack_capacity:
                                if lane == 0:
                                    qd.atomic_or(self.overflow_bits[()], self.overflow_stack_bit)
                                    qd.atomic_or(overflow_flag[()], 1)
                            else:
                                if push:
                                    lane_lt = (qd.u64(1) << qd.u64(lane)) - qd.u64(1)
                                    output = top + qd.i32(qd.math.popcnt(push_mask & lane_lt))
                                    stack_left[subgroup_in_block, output] = qd.u32(children[child, 0])
                                    stack_right[subgroup_in_block, output] = qd.u32(children[child, 1])
                                top = top + push_count
                            qd_subgroup.mem_fence()
                            qd_subgroup.sync()

                        _ee_emit_pair(
                            active and result[1] != 0,
                            result[2],
                            result[3],
                            pairs,
                            n_pairs,
                            max_pairs,
                            overflow_flag,
                        )

    @qd.func(requires_top_level=True)
    def query(
        self,
        bvh: qd.template(),
        surface: qd.template(),
        vertex: qd.template(),
        body: qd.template(),
        contact: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs,
        overflow_flag: qd.template(),
    ):
        self.reset(bvh, n_pairs)
        for level in qd.static(range(18)):
            if qd.static(level % 2 == 0):
                self.expand_frontier(
                    bvh,
                    surface,
                    vertex,
                    body,
                    contact,
                    pairs,
                    n_pairs,
                    max_pairs,
                    overflow_flag,
                    True,
                )
                self.prepare_frontier(overflow_flag, False)
            else:
                self.expand_frontier(
                    bvh,
                    surface,
                    vertex,
                    body,
                    contact,
                    pairs,
                    n_pairs,
                    max_pairs,
                    overflow_flag,
                    False,
                )
                self.prepare_frontier(overflow_flag, True)
        self.dfs(
            bvh,
            surface,
            vertex,
            body,
            contact,
            pairs,
            n_pairs,
            max_pairs,
            overflow_flag,
            True,
        )
        self.dfs(
            bvh,
            surface,
            vertex,
            body,
            contact,
            pairs,
            n_pairs,
            max_pairs,
            overflow_flag,
            False,
        )
