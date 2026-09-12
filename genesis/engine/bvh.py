import quadrants as qd

import genesis as gs
from genesis.repr_base import RBC

# A constant stack size should be sufficient for BVH traversal.
# https://madmann91.github.io/2021/01/06/bvhs-part-2.html
# https://forums.developer.nvidia.com/t/thinking-parallel-part-ii-tree-traversal-on-the-gpu/148342
STACK_SIZE = 64

# Launch geometry of the scene-extent reduction in `LBVH.compute_aabb_centers_and_scales`, widest first. Every width
# is a multiple of 64, hence of any subgroup size `block.reduce_*` can be given (see `qd.simt.subgroup.group_size`).
# A block reduction costs the same whatever its slice is worth, hence the floor on the workload of a single lane.
EXTENT_REDUCE_BLOCK_DIMS = (256, 128, 64)
EXTENT_REDUCE_MIN_AABBS_PER_LANE = 32


@qd.data_oriented
class AABB(RBC):
    """
    AABB (Axis-Aligned Bounding Box) class for managing collections of bounding boxes in batches.

    This class defines an axis-aligned bounding box (AABB) structure and provides a Quadrants dataclass
    for efficient computation and intersection testing on the GPU. Each AABB is represented by its
    minimum and maximum 3D coordinates. The class supports batch processing of multiple AABBs.

    Attributes:
        n_batches (int): Number of batches of AABBs.
        n_aabbs (int): Number of AABBs per batch.
        qd_aabb (quadrants.dataclass): Quadrants dataclass representing an individual AABB with min and max vectors.
        aabbs (quadrants.field): Quadrants field storing all AABBs in the specified batches.

    Args:
        n_batches (int): Number of batches to allocate.
        n_aabbs (int): Number of AABBs per batch.

    Example:
        aabb_manager = AABB(n_batches=4, n_aabbs=128)
    """

    def __init__(self, n_batches, n_aabbs):
        self.n_batches = n_batches
        self.n_aabbs = n_aabbs

        @qd.dataclass
        class qd_aabb:
            min: gs.qd_vec3
            max: gs.qd_vec3

            @qd.func
            def intersects(self, other) -> bool:
                """
                Check if this AABB intersects with another AABB.
                """
                return (
                    self.min[0] <= other.max[0]
                    and self.max[0] >= other.min[0]
                    and self.min[1] <= other.max[1]
                    and self.max[1] >= other.min[1]
                    and self.min[2] <= other.max[2]
                    and self.max[2] >= other.min[2]
                )

        self.qd_aabb = qd_aabb

        self.aabbs = qd_aabb.field(
            shape=(n_batches, n_aabbs),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )


@qd.data_oriented
class LBVH(RBC):
    """
    Linear BVH is a simple BVH that is used to accelerate collision detection. It supports parallel building and
    querying of the BVH tree. Only supports axis-aligned bounding boxes (AABBs).

    Parameters
    -----
    aabbs : qd.field
        The input AABBs to be organized in the BVH, shape (n_batches, n_aabbs).
    max_n_query_result_per_aabb : int
        Maximum number of query results per AABB per batch (n_batches * n_aabbs * max_n_query_result_per_aabb).
        Defaults to 0, which means the max number of query results is 1 regardless of n_batches or n_aabbs.
    n_radix_sort_groups : int
        Number of groups to use for radix sort. More groups may improve performance but will use more memory.
    max_stack_depth : int
        Maximum stack depth for BVH traversal. Defaults to STACK_SIZE.

    Notes
    ------
        For algorithmic details, see:
        https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
    """

    def __init__(
        self,
        aabb: AABB,
        max_n_query_result_per_aabb: int = 8,
        n_radix_sort_groups: int = 256,
        max_stack_depth: int = STACK_SIZE,
    ):
        if aabb.n_aabbs < 1:
            gs.raise_exception("The number of AABBs must be at least 1.")
        n_radix_sort_groups = min(aabb.n_aabbs, n_radix_sort_groups)

        self.aabbs = aabb.aabbs
        self.n_aabbs = aabb.n_aabbs
        self.n_batches = aabb.n_batches

        self.max_query_results = max(1, min(self.n_aabbs * max_n_query_result_per_aabb * self.n_batches, 0x7FFFFFFF))
        self.max_stack_depth = max_stack_depth

        self.aabb_centers = qd.field(gs.qd_vec3, shape=(self.n_batches, self.n_aabbs))
        self.aabb_min = qd.field(gs.qd_vec3, shape=(self.n_batches,))
        self.aabb_max = qd.field(gs.qd_vec3, shape=(self.n_batches,))
        self.scale = qd.field(gs.qd_vec3, shape=(self.n_batches,))
        # Block-cooperative scene-extent reduction: it goes through subgroup shuffles, so it is GPU-only, and it
        # only pays off for a tree wide enough to fill the narrowest block. Anything smaller keeps the flat atomic
        # reduction below, whose contention is bounded by the AABB count at that size anyway.
        narrowest_block_dim = EXTENT_REDUCE_BLOCK_DIMS[-1]
        self._use_cooperative_extent_reduce = gs.backend != gs.cpu and self.n_aabbs >= narrowest_block_dim
        block_dim = next((width for width in EXTENT_REDUCE_BLOCK_DIMS if width <= self.n_aabbs), narrowest_block_dim)
        n_blocks = max(1, self.n_aabbs // (block_dim * EXTENT_REDUCE_MIN_AABBS_PER_LANE))
        self._extent_reduce_block_dim = block_dim
        self._n_extent_reduce_threads = n_blocks * block_dim
        self.morton_codes = qd.field(qd.types.vector(2, qd.u32), shape=(self.n_batches, self.n_aabbs))

        # Histogram for radix sort
        self.hist = qd.field(qd.u32, shape=(self.n_batches, 256))
        # Prefix sum for histogram
        self.prefix_sum = qd.field(qd.u32, shape=(self.n_batches, 256 + 1))
        # Offset for radix sort
        self.offset = qd.field(qd.u32, shape=(self.n_batches, self.n_aabbs))
        # Temporary storage for radix sort
        self.tmp_morton_codes = qd.field(qd.types.vector(2, qd.u32), shape=(self.n_batches, self.n_aabbs))

        self.n_radix_sort_groups = n_radix_sort_groups
        self.hist_group = qd.field(qd.u32, shape=(self.n_batches, self.n_radix_sort_groups, 256 + 1))
        self.prefix_sum_group = qd.field(qd.u32, shape=(self.n_batches, self.n_radix_sort_groups + 1, 256))
        self.group_size = self.n_aabbs // self.n_radix_sort_groups
        self.visited = qd.field(qd.u8, shape=(self.n_aabbs,))

        @qd.dataclass
        class Node:
            """
            Node structure for the BVH tree.

            Attributes:
                left (int): Index of the left child node.
                right (int): Index of the right child node.
                parent (int): Index of the parent node.
                bound (qd_aabb): Bounding box of the node, represented as an AABB.
            """

            left: qd.i32
            right: qd.i32
            parent: qd.i32
            bound: aabb.qd_aabb

        self.Node = Node

        # Nodes of the BVH, first n_aabbs - 1 are internal nodes, last n_aabbs are leaf nodes
        self.nodes = self.Node.field(shape=(self.n_batches, self.n_aabbs * 2 - 1))
        # Whether an internal node has been visited during traversal. A single-leaf tree (n_aabbs == 1) has no
        # internal nodes; allocate at least one slot so the field is never zero-sized. The build and bounds kernels
        # range over n_aabbs - 1, so the padding slot stays unused.
        n_internal_nodes = max(1, self.n_aabbs - 1)
        self.internal_node_active = qd.field(gs.qd_bool, shape=(self.n_batches, n_internal_nodes))
        self.internal_node_ready = qd.field(gs.qd_bool, shape=(self.n_batches, n_internal_nodes))

        # Query results, vec3 of batch id, self id, query id
        self.query_result = qd.field(gs.qd_ivec3, shape=(self.max_query_results,))
        # Count of query results
        self.query_result_count = qd.field(qd.i32, shape=())

    def build(self):
        """
        Build the BVH from the axis-aligned bounding boxes (AABBs).
        """
        self.compute_aabb_centers_and_scales()
        self.compute_morton_codes()
        self.radix_sort_morton_codes()
        self.build_radix_tree()
        self.compute_bounds()

    @qd.func
    def filter(self, i_a, i_q):
        """
        Filter function that always returns False.

        This function does not filter out any AABB by default.
        It can be overridden in subclasses to implement custom filtering logic.

        i_a: index of the found AABB
        i_q: index of the query AABB
        """
        return False

    @qd.kernel
    def compute_aabb_centers_and_scales(self):
        for i_b, i_a in qd.ndrange(self.n_batches, self.n_aabbs):
            self.aabb_centers[i_b, i_a] = (self.aabbs[i_b, i_a].min + self.aabbs[i_b, i_a].max) / 2

        for i_b in qd.ndrange(self.n_batches):
            self.aabb_min[i_b] = self.aabb_centers[i_b, 0]
            self.aabb_max[i_b] = self.aabb_centers[i_b, 0]

        if qd.static(self._use_cooperative_extent_reduce):
            # One atomic per block instead of one per AABB: each lane folds a strided (hence coalesced) slice in
            # registers and `block.reduce_*` collapses the block. Bit-identical to the flat reduction below, min /
            # max being exact, associative and commutative.
            _BLOCK_DIM = qd.static(self._extent_reduce_block_dim)
            _N_THREADS_PER_BATCH = qd.static(self._n_extent_reduce_threads)
            qd.loop_config(name="reduce_aabb_extent", block_dim=_BLOCK_DIM)
            for i_flat in range(self.n_batches * _N_THREADS_PER_BATCH):
                i_b = i_flat // _N_THREADS_PER_BATCH
                i_lane = i_flat % _N_THREADS_PER_BATCH
                # A real AABB of the batch as seed keeps lanes that run out of work neutral, so no identity value
                # is needed and no lane is left uninitialized.
                lane_min = self.aabbs[i_b, 0].min
                lane_max = self.aabbs[i_b, 0].max
                i_a = i_lane
                while i_a < self.n_aabbs:
                    lane_min = qd.min(lane_min, self.aabbs[i_b, i_a].min)
                    lane_max = qd.max(lane_max, self.aabbs[i_b, i_a].max)
                    i_a += _N_THREADS_PER_BATCH

                # `block.reduce_*` is scalar, so fold the vec3 component-wise. Result is valid on thread 0 only.
                block_min = lane_min
                block_max = lane_max
                for i in qd.static(range(3)):
                    block_min[i] = qd.simt.block.reduce_min(lane_min[i], _BLOCK_DIM, gs.qd_float)
                    block_max[i] = qd.simt.block.reduce_max(lane_max[i], _BLOCK_DIM, gs.qd_float)

                if qd.simt.block.thread_idx() == 0:
                    qd.atomic_min(self.aabb_min[i_b], block_min)
                    qd.atomic_max(self.aabb_max[i_b], block_max)
        else:
            for i_b, i_a in qd.ndrange(self.n_batches, self.n_aabbs):
                qd.atomic_min(self.aabb_min[i_b], self.aabbs[i_b, i_a].min)
                qd.atomic_max(self.aabb_max[i_b], self.aabbs[i_b, i_a].max)

        for i_b in qd.ndrange(self.n_batches):
            scale = self.aabb_max[i_b] - self.aabb_min[i_b]
            for i in qd.static(range(3)):
                self.scale[i_b][i] = qd.select(scale[i] > gs.EPS, 1.0 / scale[i], 1.0)

    @qd.kernel
    def compute_morton_codes(self):
        """
        Compute the Morton codes for each AABB.

        The first 32 bits is the Morton code for the x, y, z coordinates, and the last 32 bits is the index of the AABB
        in the original array. The x, y, z coordinates are scaled to a 10-bit integer range [0, 1024) and interleaved to
        form the Morton code.
        """
        for i_b, i_a in qd.ndrange(self.n_batches, self.n_aabbs):
            center = self.aabb_centers[i_b, i_a] - self.aabb_min[i_b]
            scaled_center = center * self.scale[i_b]
            morton_code_x = qd.floor(scaled_center[0] * 1023.0, dtype=qd.u32)
            morton_code_y = qd.floor(scaled_center[1] * 1023.0, dtype=qd.u32)
            morton_code_z = qd.floor(scaled_center[2] * 1023.0, dtype=qd.u32)
            morton_code_x = self.expand_bits(morton_code_x)
            morton_code_y = self.expand_bits(morton_code_y)
            morton_code_z = self.expand_bits(morton_code_z)
            morton_code = (morton_code_x << 2) | (morton_code_y << 1) | (morton_code_z)
            self.morton_codes[i_b, i_a] = qd.Vector([morton_code, i_a], dt=qd.u32)

    @qd.func
    def expand_bits(self, v: qd.u32) -> qd.u32:
        """
        Expands a 10-bit integer into 30 bits by inserting 2 zeros before each bit.
        """
        v = (v * qd.u32(0x00010001)) & qd.u32(0xFF0000FF)
        # This is to silence Quadrants debug warning of overflow
        # Has the same result as v = (v * qd.u32(0x00000101)) & qd.u32(0x0F00F00F)
        # Performance difference is negligible
        # See https://github.com/Genesis-Embodied-AI/Genesis/pull/1560 for details
        v = (v | ((v & 0x00FFFFFF) << 8)) & 0x0F00F00F
        v = (v * qd.u32(0x00000011)) & qd.u32(0xC30C30C3)
        v = (v * qd.u32(0x00000005)) & qd.u32(0x49249249)
        return v

    def radix_sort_morton_codes(self):
        """
        Radix sort the morton codes, using 8 bits at a time.
        """
        # The last 32 bits are the index of the AABB which are already sorted, no need to sort
        for i in range(4, 8):
            if self.n_radix_sort_groups == 1:
                self._kernel_radix_sort_morton_codes_one_round(i)
            else:
                self._kernel_radix_sort_morton_codes_one_round_group(i)

    @qd.kernel
    def _kernel_radix_sort_morton_codes_one_round(self, i: int):
        # Clear histogram
        self.hist.fill(0)

        # Fill histogram
        for i_b in range(self.n_batches):
            # This is now sequential
            # TODO Parallelize, need to use groups to handle data to remain stable, could be not worth it
            for i_a in range(self.n_aabbs):
                code = (self.morton_codes[i_b, i_a][1 - (i // 4)] >> ((i % 4) * 8)) & 0xFF
                self.offset[i_b, i_a] = qd.atomic_add(self.hist[i_b, qd.i32(code)], 1)

        # Compute prefix sum
        for i_b in qd.ndrange(self.n_batches):
            self.prefix_sum[i_b, 0] = 0
            for j in range(1, 256):  # sequential prefix sum
                self.prefix_sum[i_b, j] = self.prefix_sum[i_b, j - 1] + self.hist[i_b, j - 1]

        # Reorder morton codes
        for i_b, i_a in qd.ndrange(self.n_batches, self.n_aabbs):
            code = qd.i32((self.morton_codes[i_b, i_a][1 - (i // 4)] >> ((i % 4) * 8)) & 0xFF)
            idx = qd.i32(self.offset[i_b, i_a] + self.prefix_sum[i_b, code])
            self.tmp_morton_codes[i_b, idx] = self.morton_codes[i_b, i_a]

        # Swap the temporary and original morton codes
        for i_b, i_a in qd.ndrange(self.n_batches, self.n_aabbs):
            self.morton_codes[i_b, i_a] = self.tmp_morton_codes[i_b, i_a]

    @qd.kernel
    def _kernel_radix_sort_morton_codes_one_round_group(self, i: int):
        # Clear histogram
        self.hist_group.fill(0)

        # Fill histogram
        for i_b, i_g in qd.ndrange(self.n_batches, self.n_radix_sort_groups):
            start = i_g * self.group_size
            end = qd.select(i_g == self.n_radix_sort_groups - 1, self.n_aabbs, (i_g + 1) * self.group_size)
            for i_a in range(start, end):
                code = qd.i32((self.morton_codes[i_b, i_a][1 - (i // 4)] >> ((i % 4) * 8)) & 0xFF)
                self.offset[i_b, i_a] = self.hist_group[i_b, i_g, code]
                self.hist_group[i_b, i_g, code] = self.hist_group[i_b, i_g, code] + 1

        # Compute prefix sum
        for i_b, i_c in qd.ndrange(self.n_batches, 256):
            self.prefix_sum_group[i_b, 0, i_c] = 0
            for i_g in range(1, self.n_radix_sort_groups + 1):  # sequential prefix sum
                self.prefix_sum_group[i_b, i_g, i_c] = (
                    self.prefix_sum_group[i_b, i_g - 1, i_c] + self.hist_group[i_b, i_g - 1, i_c]
                )
        for i_b in range(self.n_batches):
            self.prefix_sum[i_b, 0] = 0
            for i_c in range(1, 256 + 1):  # sequential prefix sum
                self.prefix_sum[i_b, i_c] = (
                    self.prefix_sum[i_b, i_c - 1] + self.prefix_sum_group[i_b, self.n_radix_sort_groups, i_c - 1]
                )

        # Reorder morton codes
        for i_b, i_a in qd.ndrange(self.n_batches, self.n_aabbs):
            code = qd.i32((self.morton_codes[i_b, i_a][1 - (i // 4)] >> ((i % 4) * 8)) & 0xFF)
            i_g = qd.min(i_a // self.group_size, self.n_radix_sort_groups - 1)
            idx = qd.i32(self.prefix_sum[i_b, code] + self.prefix_sum_group[i_b, i_g, code] + self.offset[i_b, i_a])
            # Use the group prefix sum to find the correct index
            self.tmp_morton_codes[i_b, idx] = self.morton_codes[i_b, i_a]

        # Swap the temporary and original morton codes
        for i_b, i_a in qd.ndrange(self.n_batches, self.n_aabbs):
            self.morton_codes[i_b, i_a] = self.tmp_morton_codes[i_b, i_a]

    @qd.kernel
    def build_radix_tree(self):
        """
        Build the radix tree from the sorted morton codes.

        The tree is built in parallel for every internal node.
        """
        # Initialize the first node
        for i_b in qd.ndrange(self.n_batches):
            self.nodes[i_b, 0].parent = -1

        # Initialize the leaf nodes
        for i_b, i in qd.ndrange(self.n_batches, self.n_aabbs):
            self.nodes[i_b, i + self.n_aabbs - 1].left = -1
            self.nodes[i_b, i + self.n_aabbs - 1].right = -1

        # Parallel build for every internal node
        for i_b, i in qd.ndrange(self.n_batches, self.n_aabbs - 1):
            d = qd.select(self.delta(i, i + 1, i_b) > self.delta(i, i - 1, i_b), 1, -1)

            delta_min = self.delta(i, i - d, i_b)
            l_max = qd.u32(2)
            while self.delta(i, i + qd.i32(l_max) * d, i_b) > delta_min:
                l_max *= 2

            l = qd.u32(0)
            t = l_max // 2
            while t > 0:
                if self.delta(i, i + qd.i32(l + t) * d, i_b) > delta_min:
                    l += t
                t //= 2

            j = i + qd.i32(l) * d
            delta_node = self.delta(i, j, i_b)
            s = qd.u32(0)
            t = (l + 1) // 2
            while t > 0:
                if self.delta(i, i + qd.i32(s + t) * d, i_b) > delta_node:
                    s += t
                t = qd.select(t > 1, (t + 1) // 2, 0)

            gamma = i + qd.i32(s) * d + qd.min(d, 0)
            left = qd.select(qd.min(i, j) == gamma, gamma + self.n_aabbs - 1, gamma)
            right = qd.select(qd.max(i, j) == gamma + 1, gamma + self.n_aabbs, gamma + 1)
            self.nodes[i_b, i].left = qd.i32(left)
            self.nodes[i_b, i].right = qd.i32(right)
            self.nodes[i_b, qd.i32(left)].parent = i
            self.nodes[i_b, qd.i32(right)].parent = i

    @qd.func
    def delta(self, i: qd.i32, j: qd.i32, i_b: qd.i32):
        """
        Compute the longest common prefix (LCP) of the morton codes of two AABBs.
        """
        result = -1
        if j >= 0 and j < self.n_aabbs:
            result = 64
            for i_bit in range(2):
                x = self.morton_codes[i_b, i][i_bit] ^ self.morton_codes[i_b, j][i_bit]
                for b in range(32):
                    if x & (qd.u32(1) << (31 - b)):
                        result = b + 32 * i_bit
                        break
                if result != 64:
                    break
        return result

    def compute_bounds(self):
        """
        Compute the bounds of the BVH nodes.

        Starts from the leaf nodes and works upwards layer by layer.
        """
        self._kernel_compute_bounds_init()
        is_done = False
        while not is_done:
            is_done = self._kernel_compute_bounds_one_layer()

    @qd.kernel
    def _kernel_compute_bounds_init(self):
        self.internal_node_active.fill(False)
        self.internal_node_ready.fill(False)

        for i_b, i in qd.ndrange(self.n_batches, self.n_aabbs):
            idx = qd.i32(self.morton_codes[i_b, i][1])
            self.nodes[i_b, i + self.n_aabbs - 1].bound.min = self.aabbs[i_b, idx].min
            self.nodes[i_b, i + self.n_aabbs - 1].bound.max = self.aabbs[i_b, idx].max
            parent_idx = self.nodes[i_b, i + self.n_aabbs - 1].parent
            if parent_idx != -1:
                self.internal_node_active[i_b, parent_idx] = True

    @qd.kernel
    def _kernel_compute_bounds_one_layer(self) -> qd.i32:
        for i_b, i in qd.ndrange(self.n_batches, self.n_aabbs - 1):
            if self.internal_node_active[i_b, i]:
                left_bound = self.nodes[i_b, self.nodes[i_b, i].left].bound
                right_bound = self.nodes[i_b, self.nodes[i_b, i].right].bound
                self.nodes[i_b, i].bound.min = qd.min(left_bound.min, right_bound.min)
                self.nodes[i_b, i].bound.max = qd.max(left_bound.max, right_bound.max)
                parent_idx = self.nodes[i_b, i].parent
                if parent_idx != -1:
                    self.internal_node_ready[i_b, parent_idx] = True
                self.internal_node_active[i_b, i] = False

        is_done = True
        for i_b, i in qd.ndrange(self.n_batches, self.n_aabbs - 1):
            if self.internal_node_ready[i_b, i]:
                self.internal_node_active[i_b, i] = True
                is_done = False
        self.internal_node_ready.fill(False)

        return is_done

    @qd.func
    def query(self, aabbs: qd.template()):
        """
        Query the BVH for intersections with the given AABBs.

        The results are stored in the query_result field.
        """
        self.query_result_count[None] = 0
        overflow = False

        n_querys = aabbs.shape[1]
        for i_b, i_q in qd.ndrange(self.n_batches, n_querys):
            query_stack = qd.Vector.zero(qd.i32, self.max_stack_depth)
            stack_depth = 1

            while stack_depth > 0:
                stack_depth -= 1
                node_idx = query_stack[stack_depth]
                node = self.nodes[i_b, node_idx]
                # Check if the AABB intersects with the node's bounding box
                if aabbs[i_b, i_q].intersects(node.bound):
                    # If it's a leaf node, add the AABB index to the query results
                    if node.left == -1 and node.right == -1:
                        i_a = qd.i32(self.morton_codes[i_b, node_idx - (self.n_aabbs - 1)][1])
                        # Check if the filter condition is met
                        if self.filter(i_a, i_q):
                            continue
                        idx = qd.atomic_add(self.query_result_count[None], 1)
                        if idx < self.max_query_results:
                            self.query_result[idx] = gs.qd_ivec3(i_b, i_a, i_q)  # Store the AABB index
                        else:
                            overflow = True
                    else:
                        # Push children onto the stack
                        if node.right != -1:
                            query_stack[stack_depth] = node.right
                            stack_depth += 1
                        if node.left != -1:
                            query_stack[stack_depth] = node.left
                            stack_depth += 1

        return overflow


@qd.data_oriented
class FEMSurfaceTetLBVH(LBVH):
    """
    FEMSurfaceTetLBVH is a specialized Linear BVH for FEM surface tetrahedrals.

    It extends the LBVH class to support filtering based on FEM surface tetrahedral elements.
    """

    def __init__(self, fem_solver, aabb: AABB, max_n_query_result_per_aabb: int = 8, n_radix_sort_groups: int = 256):
        super().__init__(aabb, max_n_query_result_per_aabb, n_radix_sort_groups)
        self.fem_solver = fem_solver

    @qd.func
    def filter(self, i_a, i_q):
        """
        Filter function for FEM surface tets. Filter out tet that share vertices.

        This is used to avoid self-collisions in FEM surface tets.

        Parameters
        ----------
        i_a:
            index of the found AABB
        i_q:
            index of the query AABB
        """
        result = i_a >= i_q
        i_av = self.fem_solver.elements_i[self.fem_solver.surface_elements[i_a]].el2v
        i_qv = self.fem_solver.elements_i[self.fem_solver.surface_elements[i_q]].el2v
        for i, j in qd.static(qd.ndrange(4, 4)):
            if i_av[i] == i_qv[j]:
                result = True
        return result


@qd.data_oriented
class RigidTetLBVH(LBVH):
    """
    RigidTetLBVH is a specialized Linear BVH for rigid tetrahedrals.
    It extends the LBVH class to support filtering based on rigid tetrahedral elements.
    """

    def __init__(self, coupler, aabb: AABB, max_n_query_result_per_aabb: int = 8, n_radix_sort_groups: int = 256):
        super().__init__(aabb, max_n_query_result_per_aabb, n_radix_sort_groups)
        self.coupler = coupler
        self.rigid_solver = coupler.rigid_solver

    @qd.func
    def filter(self, i_a, i_q):
        """
        Filter function for Rigid tets. Filter out tet that belong to the same link

        i_a: index of the found AABB
        i_q: index of the query AABB
        """
        i_ag = self.coupler.rigid_volume_elems_geom_idx[i_a]
        i_qg = self.coupler.rigid_volume_elems_geom_idx[i_q]
        return self.coupler.rigid_collision_pair_idx[i_ag, i_qg] == -1
