"""LBVH -- Linear Bounding Volume Hierarchy (Karras 2012).

One ``LBVH`` instance per primitive type (triangles, edges).
All buffers are ``qd.ndarray`` to enable fastcache
kernel sharing across instances with different sizes.  Build and query methods
are ``@qd.func(requires_top_level=True)`` for use inside graph kernels.

Faithfully ports cgq ``BVHContext`` + ``lbvh_kernels.cu`` + ``bvh_subgraph.h``.
"""

# ruff: noqa: SIM102

from __future__ import annotations

import numpy as np
import quadrants as qd
from quadrants.algorithms import sort, sort_scratch_slots
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as qd_block, subgroup as qd_subgroup

from .bvh_math import (
    aabb_combine_aabb,
    aabb_combine_point,
    aabb_expand,
    aabb_init,
    aabb_overlap,
    determine_range,
    f64_to_f32_rd,
    f64_to_f32_ru,
    find_split,
    morton_code_30bit,
)
from .bvh_predicate import (
    _ee_emit_pair,
    _ee_pair_enabled,
    _node_pair_enabled,
    _pt_emit_pair,
    _pt_pair_enabled,
)
from .contact_function.contact_table_query import ct_enabled_pt
from .contact_function.edge_triangle_intersection import (
    triangle_edge_intersect,
)
from .dynamic_radix_sort import DynamicRadixSort
from .gpu_occupancy import cuda_resident_blocks

# structural, host-only: traversal stack depth. Read once to shape `stack_pool`
# and never from device code, so it may stay a module constant.
_BVH_STACK_CAPACITY = 64


@qd.func
def _edge_node_overlap(
    aabbs: qd.template(),
    node,
    edge_a: qd.template(),
    edge_b: qd.template(),
    half: qd.template(),
):
    projection_a = qd.Vector(
        [
            edge_a[0],
            edge_a[1],
            edge_a[2],
            edge_a[0] + edge_a[1] + edge_a[2],
            edge_a[0] + edge_a[1] - edge_a[2],
            edge_a[0] - edge_a[1] + edge_a[2],
            edge_a[0] - edge_a[1] - edge_a[2],
        ]
    )
    projection_b = qd.Vector(
        [
            edge_b[0],
            edge_b[1],
            edge_b[2],
            edge_b[0] + edge_b[1] + edge_b[2],
            edge_b[0] + edge_b[1] - edge_b[2],
            edge_b[0] - edge_b[1] + edge_b[2],
            edge_b[0] - edge_b[1] - edge_b[2],
        ]
    )
    overlaps = True
    for axis in qd.static(range(half)):
        lower = qd.min(projection_a[axis], projection_b[axis])
        upper = qd.max(projection_a[axis], projection_b[axis])
        if aabbs[node, axis] > upper or lower > aabbs[node, half + axis]:
            overlaps = False
    return overlaps


@qd.func
def _edge_triangle_intersects(
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
    edge,
    face,
):
    edge_a = surface.surf_edges[edge, 0]
    edge_b = surface.surf_edges[edge, 1]
    triangle_a = surface.surf_triangles[face, 0]
    triangle_b = surface.surf_triangles[face, 1]
    triangle_c = surface.surf_triangles[face, 2]
    enabled = not (
        edge_a == triangle_a
        or edge_a == triangle_b
        or edge_a == triangle_c
        or edge_b == triangle_a
        or edge_b == triangle_b
        or edge_b == triangle_c
    )

    edge_body = vertex.body_id[edge_a]
    face_body = vertex.body_id[triangle_a]
    if edge_body == face_body and edge_body >= 0 and body.self_collision[edge_body] == 0:
        enabled = False
    if (
        edge_body >= 0
        and face_body >= 0
        and edge_body != face_body
        and body.is_body_contact_ignored(edge_body, face_body)
    ):
        enabled = False

    n_elements = contact.n_contact_elements[()]
    if n_elements > 0:
        edge_a_enabled = ct_enabled_pt(
            contact.enable_table,
            n_elements,
            contact.vert_contact_element_ids[edge_a],
            contact.vert_contact_element_ids[triangle_a],
            contact.vert_contact_element_ids[triangle_b],
            contact.vert_contact_element_ids[triangle_c],
        )
        edge_b_enabled = ct_enabled_pt(
            contact.enable_table,
            n_elements,
            contact.vert_contact_element_ids[edge_b],
            contact.vert_contact_element_ids[triangle_a],
            contact.vert_contact_element_ids[triangle_b],
            contact.vert_contact_element_ids[triangle_c],
        )
        enabled = enabled and edge_a_enabled and edge_b_enabled

    intersects = False
    if enabled:
        edge_position_a = qd.Vector(
            [
                vertex.positions[edge_a, 0],
                vertex.positions[edge_a, 1],
                vertex.positions[edge_a, 2],
            ]
        )
        edge_position_b = qd.Vector(
            [
                vertex.positions[edge_b, 0],
                vertex.positions[edge_b, 1],
                vertex.positions[edge_b, 2],
            ]
        )
        triangle_position_a = qd.Vector(
            [
                vertex.positions[triangle_a, 0],
                vertex.positions[triangle_a, 1],
                vertex.positions[triangle_a, 2],
            ]
        )
        triangle_position_b = qd.Vector(
            [
                vertex.positions[triangle_b, 0],
                vertex.positions[triangle_b, 1],
                vertex.positions[triangle_b, 2],
            ]
        )
        triangle_position_c = qd.Vector(
            [
                vertex.positions[triangle_c, 0],
                vertex.positions[triangle_c, 1],
                vertex.positions[triangle_c, 2],
            ]
        )
        intersects = triangle_edge_intersect(
            triangle_position_a,
            triangle_position_b,
            triangle_position_c,
            edge_position_a,
            edge_position_b,
        )
    return intersects


@qd.func
def _swept_point_overlap(
    aabbs: qd.template(),
    node,
    start: qd.template(),
    endpoint: qd.template(),
    gap,
    half: qd.template(),
    use_dop14f: qd.template(),
):
    result = qd.i32(1)
    start_projection = qd.Vector(
        [
            start[0],
            start[1],
            start[2],
            start[0] + start[1] + start[2],
            start[0] + start[1] - start[2],
            start[0] - start[1] + start[2],
            start[0] - start[1] - start[2],
        ]
    )
    end_projection = qd.Vector(
        [
            endpoint[0],
            endpoint[1],
            endpoint[2],
            endpoint[0] + endpoint[1] + endpoint[2],
            endpoint[0] + endpoint[1] - endpoint[2],
            endpoint[0] - endpoint[1] + endpoint[2],
            endpoint[0] - endpoint[1] - endpoint[2],
        ]
    )
    for axis in qd.static(range(half)):
        scale = qd.f64(1.0)
        if qd.static(axis >= 3):
            scale = qd.f64(1.7320508075688772)
        if qd.static(use_dop14f):
            query_lower = f64_to_f32_rd(qd.min(start_projection[axis], end_projection[axis]))
            query_upper = f64_to_f32_ru(qd.max(start_projection[axis], end_projection[axis]))
            axis_gap = gap * scale
            if (qd.f64(aabbs[node, axis]) - qd.f64(query_upper)) >= axis_gap or (
                qd.f64(query_lower) - qd.f64(aabbs[node, half + axis])
            ) >= axis_gap:
                result = 0
        else:
            axis_gap = gap * scale
            query_lower = qd.min(start_projection[axis], end_projection[axis])
            query_upper = qd.max(start_projection[axis], end_projection[axis])
            if (aabbs[node, axis] - query_upper) >= axis_gap or (query_lower - aabbs[node, half + axis]) >= axis_gap:
                result = 0
    return result


# ---------------------------------------------------------------------------
# Two-phase scene-bound merge reduction
# ---------------------------------------------------------------------------


@qd.func
def _bin_min_f64(a: qd.f64, b: qd.f64):
    return qd.min(a, b)


@qd.func
def _bin_max_f64(a: qd.f64, b: qd.f64):
    return qd.max(a, b)


@qd.func
def _bin_min_f32(a: qd.f32, b: qd.f32):
    return qd.min(a, b)


@qd.func
def _bin_max_f32(a: qd.f32, b: qd.f32):
    return qd.max(a, b)


@qd.func
def _aabb_reduce_phase1(
    aabbs: qd.template(),
    partials: qd.template(),
    n_rt: qd.template(),
    total_threads: qd.i32,
    block: qd.template(),
    half: qd.template(),
):
    """Phase 1: each block reduces its tile of leaf AABBs into per-block partials.

    One kernel launch; each block processes ``block`` leaves and writes one
    component-wise fp64 partial bound.
    """
    loop_config(name="bvh_scene_reduce_phase1", block_dim=block)
    for i in range(total_threads):
        qd_block.sync()
        n = n_rt[()]
        tid = i % block
        block_id = i // block
        leaf = n - 1 + i

        for comp in qd.static(range(half)):
            v = qd.f64(1e32)
            if i < n:
                v = aabbs[leaf, comp]
            agg = qd_block.reduce(v, block, _bin_min_f64, qd.f64)
            if tid == 0:
                partials[block_id, comp] = agg
        for comp in qd.static(range(half, half * 2)):
            v = qd.f64(-1e32)
            if i < n:
                v = aabbs[leaf, comp]
            agg = qd_block.reduce(v, block, _bin_max_f64, qd.f64)
            if tid == 0:
                partials[block_id, comp] = agg


@qd.func
def _aabb_reduce_phase2(
    partials: qd.template(),
    aabbs: qd.template(),
    n_blocks_rt: qd.template(),
    block: qd.template(),
    half: qd.template(),
):
    """Phase 2: single block reduces ALL per-block partials into aabbs[0].

    Uses grid-stride accumulation so a single block of ``block`` threads
    can handle arbitrary n_blocks (not limited to one tile).  Each thread
    accumulates its stripe of partials, then block-reduces the accumulated
    values.
    """
    loop_config(name="bvh_scene_reduce_phase2", block_dim=block)
    for i in range(block):
        qd_block.sync()
        nb = n_blocks_rt[()]

        for comp in qd.static(range(half)):
            v = qd.f64(1e32)
            j = i
            while j < nb:
                v = qd.min(v, partials[j, comp])
                j = j + block
            agg = qd_block.reduce(v, block, _bin_min_f64, qd.f64)
            if i == 0:
                aabbs[0, comp] = agg
        for comp in qd.static(range(half, half * 2)):
            v = qd.f64(-1e32)
            j = i
            while j < nb:
                v = qd.max(v, partials[j, comp])
                j = j + block
            agg = qd_block.reduce(v, block, _bin_max_f64, qd.f64)
            if i == 0:
                aabbs[0, comp] = agg


@qd.func
def _aabb_reduce_phase1_f32(
    aabbs: qd.template(),
    partials: qd.template(),
    n_rt: qd.template(),
    total_threads: qd.i32,
    block: qd.template(),
    half: qd.template(),
):
    """DOP14f phase 1: reduce leaf bounds without widening back to f64."""
    loop_config(name="bvh_scene_reduce_phase1_dop14f", block_dim=block)
    for i in range(total_threads):
        qd_block.sync()
        n = n_rt[()]
        tid = i % block
        block_id = i // block
        leaf = n - 1 + i

        for comp in qd.static(range(half)):
            value = qd.f32(3e38)
            if i < n:
                value = aabbs[leaf, comp]
            aggregate = qd_block.reduce(value, block, _bin_min_f32, qd.f32)
            if tid == 0:
                partials[block_id, comp] = aggregate
        for comp in qd.static(range(half, half * 2)):
            value = qd.f32(-3e38)
            if i < n:
                value = aabbs[leaf, comp]
            aggregate = qd_block.reduce(value, block, _bin_max_f32, qd.f32)
            if tid == 0:
                partials[block_id, comp] = aggregate


@qd.func
def _aabb_reduce_phase2_f32(
    partials: qd.template(),
    aabbs: qd.template(),
    n_blocks_rt: qd.template(),
    block: qd.template(),
    half: qd.template(),
):
    """DOP14f phase 2: merge block partials into the scene bound."""
    loop_config(name="bvh_scene_reduce_phase2_dop14f", block_dim=block)
    for i in range(block):
        qd_block.sync()
        n_blocks = n_blocks_rt[()]

        for comp in qd.static(range(half)):
            value = qd.f32(3e38)
            j = i
            while j < n_blocks:
                value = qd.min(value, partials[j, comp])
                j = j + block
            aggregate = qd_block.reduce(value, block, _bin_min_f32, qd.f32)
            if i == 0:
                aabbs[0, comp] = aggregate
        for comp in qd.static(range(half, half * 2)):
            value = qd.f32(-3e38)
            j = i
            while j < n_blocks:
                value = qd.max(value, partials[j, comp])
                j = j + block
            aggregate = qd_block.reduce(value, block, _bin_max_f32, qd.f32)
            if i == 0:
                aabbs[0, comp] = aggregate


@qd.data_oriented
class LBVH:
    """Self-contained LBVH for *n* primitives (faces or edges).

    Tree layout (Karras 2012):
    - ``2n - 1`` nodes total; internals ``[0, n-1)``, leaves ``[n-1, 2n-1)``.
    - Root at index 0.
    - Leaf for sorted primitive *i* is at ``n - 1 + i``.
    - ``element_idx == 0xFFFFFFFF`` for internal nodes.
    """

    def __init__(
        self,
        n_prims: int,
        max_queries: int = 0,
        bound_type: str = "aabb",
        genesis_legacy_sort_reduce: bool = False,
        genesis_legacy_fp64_bounds: bool = False,
        genesis_legacy_refit: bool = False,
    ) -> None:
        """Create LBVH for *n_prims* primitives.

        Args:
            n_prims: Number of primitives (faces or edges).
            max_queries: Max query count for stack_pool sizing.  For PT query
                this is n_surf_verts; for EE query this equals n_prims.
                If 0, defaults to n_prims.
            genesis_legacy_fp64_bounds: Retain the former fourteen-f64 DOP14
                storage for explicit A/B profiling.
            genesis_legacy_refit: Retain the former initialize-and-two-merge
                internal-node refit for explicit A/B profiling.
        """
        assert n_prims > 0, "LBVH requires n_prims > 0"
        # Radix-sort pass geometry, the u32 "no node" marker and the reduction
        # block width. All fix unroll counts or block shapes, so none can be a
        # runtime device scalar. They are instance attributes rather than module
        # constants so their values reach device code as compile-time template
        # arguments and enter the fastcache key.
        self.sort_end_bit = 64
        self.sort_log256_max_n = 4
        self.genesis_legacy_sort_reduce_host = bool(genesis_legacy_sort_reduce)
        self.sentinel = 0xFFFFFFFF
        self.stack_capacity = _BVH_STACK_CAPACITY
        self.bvh_block = 256
        self.pt_warp_block = 256
        self.pt_warp_stack_capacity = 1024
        self.ee_warp_block = 256
        self.ee_warp_stack_capacity = 1024
        subgroup_size = qd_subgroup.group_size()
        self.pt_warp_subgroups_per_block = self.pt_warp_block // subgroup_size
        # Pinned Quadrants 459a3eb57 Nsight profile for
        # query_pt_warp_range_for.
        self.pt_warp_registers_per_thread = 128
        pt_warp_shared_bytes = (
            self.pt_warp_subgroups_per_block * self.pt_warp_stack_capacity * np.dtype(np.uint32).itemsize
        )
        self.pt_warp_blocks = cuda_resident_blocks(
            self.pt_warp_block,
            pt_warp_shared_bytes,
            self.pt_warp_registers_per_thread,
        )
        self.pt_warp_workers = self.pt_warp_blocks * self.pt_warp_subgroups_per_block
        self.pt_warp_threads = self.pt_warp_workers * subgroup_size
        self.ee_warp_subgroups_per_block = self.ee_warp_block // subgroup_size
        # Pinned Quadrants 459a3eb57 CUDA profile for query_ee_warp_range_for.
        self.ee_warp_registers_per_thread = 72
        ee_warp_shared_bytes = (
            self.ee_warp_subgroups_per_block * self.ee_warp_stack_capacity * np.dtype(np.uint32).itemsize
        )
        self.ee_warp_blocks = cuda_resident_blocks(
            self.ee_warp_block,
            ee_warp_shared_bytes,
            self.ee_warp_registers_per_thread,
        )
        self.ee_warp_workers = self.ee_warp_blocks * self.ee_warp_subgroups_per_block
        self.ee_warp_threads = self.ee_warp_workers * subgroup_size
        if bound_type not in ("aabb", "dop14"):
            raise ValueError(f"Unsupported LBVH bound type {bound_type!r}")
        self.bound_type = bound_type
        self.bounds_width = 14 if bound_type == "dop14" else 6
        self.use_dop14f_host = bound_type == "dop14" and not genesis_legacy_fp64_bounds
        self.genesis_legacy_refit_host = bool(genesis_legacy_refit)
        self.bounds_storage_width = 16 if self.use_dop14f_host else self.bounds_width
        bounds_dtype = qd.f32 if self.use_dop14f_host else qd.f64

        if max_queries <= 0:
            max_queries = n_prims
        n_nodes = 2 * n_prims - 1
        padded = ((n_prims + 63) // 64) * 64
        self.n_prims = qd.ndarray(qd.i32, shape=())
        self.ee_warp_stack_overflow = qd.ndarray(qd.i32, shape=())

        # --- Tree buffers ---
        self.aabbs = qd.ndarray(bounds_dtype, (n_nodes, self.bounds_storage_width))
        self.temp_aabbs = qd.ndarray(bounds_dtype, (n_prims, self.bounds_storage_width))
        self.temp_node_body_id = qd.ndarray(qd.i32, (n_prims,))
        self.indices = qd.ndarray(qd.u32, (n_prims,))
        self.nodes_parent = qd.ndarray(qd.u32, (n_nodes,))
        self.nodes_left = qd.ndarray(qd.u32, (n_nodes,))
        self.nodes_right = qd.ndarray(qd.u32, (n_nodes,))
        self.nodes_element = qd.ndarray(qd.u32, (n_nodes,))
        self.node_body_id = qd.ndarray(qd.i32, (n_nodes,))
        self.flags = qd.ndarray(qd.u32, (max(n_prims - 1, 1),))

        # --- Sort workspace ---
        sort_scratch = max(sort_scratch_slots(padded, self.sort_log256_max_n), 1)
        self.morton = qd.ndarray(qd.u64, (padded,))
        self.morton_tmp = qd.ndarray(qd.u64, (padded,))
        self.srt_perm = qd.ndarray(qd.i32, (padded,))
        self.srt_tmp_perm = qd.ndarray(qd.i32, (padded,))
        self.srt_scratch = qd.ndarray(qd.u32, (sort_scratch,))
        self.srt_n = qd.ndarray(qd.i32, shape=())
        self.morton_sort = DynamicRadixSort(qd.u64, padded)

        # --- Scene AABB reduce workspace ---
        n_blocks = (n_prims + self.bvh_block - 1) // self.bvh_block
        self.red_partials = qd.ndarray(bounds_dtype, (max(n_blocks, 1), self.bounds_storage_width))
        self.n_reduce_blocks = qd.ndarray(qd.i32, shape=())

        # --- BVH query stack workspace (per-thread) ---
        stack_rows = max(n_prims, max_queries)
        self.stack_pool = qd.ndarray(qd.u32, (max(stack_rows, 1), _BVH_STACK_CAPACITY))

        self.n_prims.from_numpy(np.array(n_prims, dtype=np.int32))
        self.ee_warp_stack_overflow.from_numpy(np.array(0, dtype=np.int32))
        self.srt_n.from_numpy(np.array(padded, dtype=np.int32))
        self.n_reduce_blocks.from_numpy(np.array(n_blocks, dtype=np.int32))

    # ======================================================================
    # BUILD pipeline
    # ======================================================================

    @qd.func(requires_top_level=True)
    def calc_leaf_aabb_tri(self, surf_mgr: qd.template(), vtx_mgr: qd.template()):
        """Compute swept leaf AABBs for triangles (stride=3).

        Matches cgq ``calc_leaf_aabb`` with stride=3.
        """
        n = self.n_prims[()]
        loop_config(name="bvh_calc_leaf_aabb_tri")
        for idx in range(n):
            leaf = n - 1 + idx
            aabb_init(self.aabbs, leaf, self.bounds_width // 2, self.use_dop14f_host)
            for k in qd.static(range(3)):
                vi = surf_mgr.surf_triangles[idx, k]
                px = vtx_mgr.positions[vi, 0]
                py = vtx_mgr.positions[vi, 1]
                pz = vtx_mgr.positions[vi, 2]
                aabb_combine_point(
                    self.aabbs,
                    leaf,
                    px,
                    py,
                    pz,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
                endpoint_x = vtx_mgr.trajectory_end_positions[vi, 0]
                endpoint_y = vtx_mgr.trajectory_end_positions[vi, 1]
                endpoint_z = vtx_mgr.trajectory_end_positions[vi, 2]
                aabb_combine_point(
                    self.aabbs,
                    leaf,
                    endpoint_x,
                    endpoint_y,
                    endpoint_z,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
            max_thickness = qd.f64(0.0)
            max_path_inflation = qd.f64(0.0)
            for k in qd.static(range(3)):
                vi = surf_mgr.surf_triangles[idx, k]
                max_thickness = qd.max(max_thickness, vtx_mgr.thicknesses[vi])
                max_path_inflation = qd.max(max_path_inflation, vtx_mgr.path_inflation[vi])
            if max_thickness > 0.0:
                aabb_expand(
                    self.aabbs,
                    leaf,
                    max_thickness,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
            if max_path_inflation > 0.0:
                aabb_expand(
                    self.aabbs,
                    leaf,
                    max_path_inflation,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
            aabb_expand(
                self.aabbs,
                leaf,
                vtx_mgr.d_hats[surf_mgr.surf_triangles[idx, 0]],
                self.bounds_width // 2,
                self.use_dop14f_host,
            )
            body_id = vtx_mgr.body_id[surf_mgr.surf_triangles[idx, 0]]
            for k in qd.static(range(1, 3)):
                if vtx_mgr.body_id[surf_mgr.surf_triangles[idx, k]] != body_id:
                    body_id = -1
            self.node_body_id[leaf] = body_id

    @qd.func(requires_top_level=True)
    def calc_leaf_aabb_edge(self, surf_mgr: qd.template(), vtx_mgr: qd.template()):
        """Compute swept leaf AABBs for edges (stride=2).

        Matches cgq ``calc_leaf_aabb`` with stride=2.
        """
        n = self.n_prims[()]
        loop_config(name="bvh_calc_leaf_aabb_edge")
        for idx in range(n):
            leaf = n - 1 + idx
            aabb_init(self.aabbs, leaf, self.bounds_width // 2, self.use_dop14f_host)
            for k in qd.static(range(2)):
                vi = surf_mgr.surf_edges[idx, k]
                px = vtx_mgr.positions[vi, 0]
                py = vtx_mgr.positions[vi, 1]
                pz = vtx_mgr.positions[vi, 2]
                aabb_combine_point(
                    self.aabbs,
                    leaf,
                    px,
                    py,
                    pz,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
                endpoint_x = vtx_mgr.trajectory_end_positions[vi, 0]
                endpoint_y = vtx_mgr.trajectory_end_positions[vi, 1]
                endpoint_z = vtx_mgr.trajectory_end_positions[vi, 2]
                aabb_combine_point(
                    self.aabbs,
                    leaf,
                    endpoint_x,
                    endpoint_y,
                    endpoint_z,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
            max_thickness = qd.f64(0.0)
            max_path_inflation = qd.f64(0.0)
            for k in qd.static(range(2)):
                vi = surf_mgr.surf_edges[idx, k]
                max_thickness = qd.max(max_thickness, vtx_mgr.thicknesses[vi])
                max_path_inflation = qd.max(max_path_inflation, vtx_mgr.path_inflation[vi])
            if max_thickness > 0.0:
                aabb_expand(
                    self.aabbs,
                    leaf,
                    max_thickness,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
            if max_path_inflation > 0.0:
                aabb_expand(
                    self.aabbs,
                    leaf,
                    max_path_inflation,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
            aabb_expand(
                self.aabbs,
                leaf,
                vtx_mgr.d_hats[surf_mgr.surf_edges[idx, 0]],
                self.bounds_width // 2,
                self.use_dop14f_host,
            )
            body_id = vtx_mgr.body_id[surf_mgr.surf_edges[idx, 0]]
            if vtx_mgr.body_id[surf_mgr.surf_edges[idx, 1]] != body_id:
                body_id = -1
            self.node_body_id[leaf] = body_id

    # --- Scene AABB reduce (cgq: cub::DeviceReduce::Reduce + AABBReduceOp) ---
    # Decomposed into 6 scalar reduce_min/reduce_max via qd.static unrolling.

    @qd.func(requires_top_level=True)
    def reduce_scene_aabb(self):
        """Parallel reduce of leaf AABBs into aabbs[0] (scene bounding box).

        Matches cgq ``cub::DeviceReduce::Reduce(AABBReduceOp)``.
        Two-phase block reduce: phase 1 (multi-block) writes per-block
        partials, phase 2 (single-block) reduces partials into aabbs[0].
        Total: 2 kernel launches (vs cgq's 1 CUB launch).
        """
        n_blocks = self.n_reduce_blocks[()]
        total_threads = n_blocks * self.bvh_block
        if qd.static(self.use_dop14f_host):
            _aabb_reduce_phase1_f32(
                self.aabbs,
                self.red_partials,
                self.n_prims,
                total_threads,
                self.bvh_block,
                self.bounds_width // 2,
            )
            _aabb_reduce_phase2_f32(
                self.red_partials,
                self.aabbs,
                self.n_reduce_blocks,
                self.bvh_block,
                self.bounds_width // 2,
            )
        if qd.static(not self.use_dop14f_host):
            _aabb_reduce_phase1(
                self.aabbs,
                self.red_partials,
                self.n_prims,
                total_threads,
                self.bvh_block,
                self.bounds_width // 2,
            )
            _aabb_reduce_phase2(
                self.red_partials,
                self.aabbs,
                self.n_reduce_blocks,
                self.bvh_block,
                self.bounds_width // 2,
            )

    @qd.func(requires_top_level=True)
    def calc_morton(self):
        """Compute Morton codes from leaf AABB centers, normalized to scene AABB.

        Matches cgq ``calc_morton``: each thread reads scene AABB independently.
        The legacy generic-sort fallback additionally fills its padded tail.
        """
        n = self.n_prims[()]
        sort_count = n
        if qd.static(self.genesis_legacy_sort_reduce_host):
            sort_count = self.morton.shape[0]
        loop_config(name="bvh_calc_morton")
        for idx in range(sort_count):
            if idx < n:
                half = qd.static(self.bounds_width // 2)
                scene_lx = qd.f64(self.aabbs[0, 0])
                scene_ly = qd.f64(self.aabbs[0, 1])
                scene_lz = qd.f64(self.aabbs[0, 2])
                scene_sx = qd.f64(self.aabbs[0, half]) - scene_lx
                scene_sy = qd.f64(self.aabbs[0, half + 1]) - scene_ly
                scene_sz = qd.f64(self.aabbs[0, half + 2]) - scene_lz
                scene_sx = max(scene_sx, 1e-30)
                scene_sy = max(scene_sy, 1e-30)
                scene_sz = max(scene_sz, 1e-30)

                leaf = n - 1 + idx
                cx = (qd.f64(self.aabbs[leaf, 0]) + qd.f64(self.aabbs[leaf, half])) * 0.5
                cy = (qd.f64(self.aabbs[leaf, 1]) + qd.f64(self.aabbs[leaf, half + 1])) * 0.5
                cz = (qd.f64(self.aabbs[leaf, 2]) + qd.f64(self.aabbs[leaf, half + 2])) * 0.5
                nx = (cx - scene_lx) / scene_sx
                ny = (cy - scene_ly) / scene_sy
                nz = (cz - scene_lz) / scene_sz
                mc32 = morton_code_30bit(nx, ny, nz)
                self.morton[idx] = (qd.u64(mc32) << qd.u64(32)) | qd.u64(qd.u32(idx))
            else:
                self.morton[idx] = qd.u64(0xFFFFFFFFFFFFFFFF)
            self.srt_perm[idx] = idx

    @qd.func(requires_top_level=True)
    def sort_morton(self):
        """Sort Morton keys with CGQ's dynamic OneSweep u64 path.

        The retained generic Quadrants radix sort is an explicit A/B fallback.
        """
        if qd.static(self.genesis_legacy_sort_reduce_host):
            sort(
                self.morton,
                self.morton_tmp,
                self.srt_perm,
                self.srt_tmp_perm,
                self.srt_scratch,
                self.srt_n,
                qd.u64,
                True,
                self.sort_end_bit,
                self.sort_log256_max_n,
            )
        else:
            self.morton_sort.sort(
                self.morton,
                self.morton_tmp,
                self.srt_perm,
                self.srt_tmp_perm,
                self.n_prims[()],
            )

    @qd.func(requires_top_level=True)
    def extract_indices(self):
        """Extract original primitive index from lower 32 bits of sorted Morton keys.

        Matches cgq ``extract_indices``.
        """
        n = self.n_prims[()]
        loop_config(name="bvh_extract_indices")
        for idx in range(n):
            self.indices[idx] = qd.u32(self.morton[idx] & qd.u64(0xFFFFFFFF))

    @qd.func(requires_top_level=True)
    def copy_leaf_aabb_to_temp(self):
        """Copy leaf AABBs to temp before reorder.

        Matches cgq ``bvh_copy_leaf_aabb_kernel``.
        """
        n = self.n_prims[()]
        loop_config(name="bvh_copy_leaf_aabb")
        for idx in range(n):
            leaf = n - 1 + idx
            for k in qd.static(range(self.bounds_storage_width)):
                self.temp_aabbs[idx, k] = self.aabbs[leaf, k]
            self.temp_node_body_id[idx] = self.node_body_id[leaf]

    @qd.func(requires_top_level=True)
    def reorder_leaf_aabb(self):
        """Reorder leaf AABBs to Morton-sorted order.

        Matches cgq ``reorder_leaf_aabb``.
        """
        n = self.n_prims[()]
        loop_config(name="bvh_reorder_leaf_aabb")
        for idx in range(n):
            leaf = n - 1 + idx
            src = qd.i32(self.indices[idx])
            for k in qd.static(range(self.bounds_storage_width)):
                self.aabbs[leaf, k] = self.temp_aabbs[src, k]
            self.node_body_id[leaf] = self.temp_node_body_id[src]

    @qd.func(requires_top_level=True)
    def calc_leaf_nodes(self):
        """Initialize all BVH nodes: internals get sentinel, leaves get element_idx.

        Matches cgq ``calc_leaf_nodes``.
        """
        n = self.n_prims[()]
        n_nodes = 2 * n - 1
        loop_config(name="bvh_calc_leaf_nodes")
        for idx in range(n_nodes):
            self.nodes_parent[idx] = qd.u32(self.sentinel)
            self.nodes_left[idx] = qd.u32(self.sentinel)
            self.nodes_right[idx] = qd.u32(self.sentinel)
            if idx < n - 1:
                self.nodes_element[idx] = qd.u32(self.sentinel)
            else:
                leaf_i = idx - (n - 1)
                self.nodes_element[idx] = self.indices[leaf_i]

    @qd.func(requires_top_level=True)
    def calc_internal_nodes(self):
        """Karras 2012 internal node construction.

        Matches cgq ``calc_internal_nodes``.
        """
        n = self.n_prims[()]
        loop_config(name="bvh_calc_internal_nodes")
        for idx in range(n - 1):
            ij = determine_range(self.morton, n, idx)
            first = ij[0]
            last = ij[1]
            gamma = find_split(self.morton, n, first, last)

            left_child = gamma
            right_child = gamma + 1
            ij_min = qd.min(first, last)
            ij_max = qd.max(first, last)
            if ij_min == gamma:
                left_child = left_child + n - 1
            if ij_max == gamma + 1:
                right_child = right_child + n - 1

            self.nodes_left[idx] = qd.u32(left_child)
            self.nodes_right[idx] = qd.u32(right_child)
            self.nodes_parent[left_child] = qd.u32(idx)
            self.nodes_parent[right_child] = qd.u32(idx)

    @qd.func(requires_top_level=True)
    def memset_flags(self):
        """Reset flags to 0xFFFFFFFF sentinel for bottom-up AABB refit.

        Matches cgq ``bvh_memset_flags_kernel``.
        """
        n = self.n_prims[()]
        loop_config(name="bvh_memset_flags")
        for idx in range(n - 1):
            self.flags[idx] = qd.u32(self.sentinel)

    @qd.func(requires_top_level=True)
    def calc_internal_aabb(self):
        """Bottom-up parallel AABB refit using atomicCAS on flags.

        Matches cgq ``calc_internal_aabb``:
        - flags initialized to 0xFFFFFFFF
        - atomicCAS(flags[parent], 0xFFFFFFFF, 0): first child returns, second merges
        - production writes the direct min/max merge once; the former
          initialize-and-two-merge traffic is config-gated for A/B profiling
        - qd.simt.grid.mem_fence() after merge (= __threadfence())
        - walk terminates when parent == 0xFFFFFFFF (root's parent)
        """
        n = self.n_prims[()]
        loop_config(name="bvh_calc_internal_aabb")
        for idx in range(n):
            node_idx = idx + n - 1
            parent = self.nodes_parent[node_idx]
            while parent != qd.u32(self.sentinel):
                old = qd.atomic_cas(self.flags[qd.i32(parent)], qd.u32(self.sentinel), qd.u32(0))
                if old == qd.u32(self.sentinel):
                    parent = qd.u32(self.sentinel)
                else:
                    lidx = qd.i32(self.nodes_left[qd.i32(parent)])
                    ridx = qd.i32(self.nodes_right[qd.i32(parent)])
                    if qd.static(self.genesis_legacy_refit_host):
                        aabb_init(
                            self.aabbs,
                            qd.i32(parent),
                            self.bounds_width // 2,
                            self.use_dop14f_host,
                        )
                        aabb_combine_aabb(
                            self.aabbs,
                            qd.i32(parent),
                            self.aabbs,
                            lidx,
                            self.bounds_width // 2,
                        )
                        aabb_combine_aabb(
                            self.aabbs,
                            qd.i32(parent),
                            self.aabbs,
                            ridx,
                            self.bounds_width // 2,
                        )
                    else:
                        for component in qd.static(range(self.bounds_width // 2)):
                            self.aabbs[qd.i32(parent), component] = qd.min(
                                self.aabbs[lidx, component],
                                self.aabbs[ridx, component],
                            )
                            upper_component = component + self.bounds_width // 2
                            self.aabbs[qd.i32(parent), upper_component] = qd.max(
                                self.aabbs[lidx, upper_component],
                                self.aabbs[ridx, upper_component],
                            )
                    left_body = self.node_body_id[lidx]
                    right_body = self.node_body_id[ridx]
                    self.node_body_id[qd.i32(parent)] = qd.select(
                        left_body == right_body,
                        left_body,
                        qd.i32(-1),
                    )
                    qd.simt.grid.mem_fence()
                    parent = self.nodes_parent[qd.i32(parent)]

    # ======================================================================
    # TOY-MODE leaf AABB + query (matches cgq toy/lbvh.cu for ipctk tests)
    # Leaf AABBs expanded by d_hat; query AABB expanded by d_hat; plain aabb_overlap.
    # ======================================================================

    @qd.func(requires_top_level=True)
    def calc_leaf_aabb_tri_toy(self, surf_mgr: qd.template(), vtx_mgr: qd.template(), d_hat: qd.f64):
        """Toy-mode: leaf AABBs for triangles with d_hat expansion.

        Matches cgq ``toy/lbvh.cu::calc_leaf_aabb_face_kernel``.
        """
        n = self.n_prims[()]
        for idx in range(n):
            leaf = n - 1 + idx
            aabb_init(self.aabbs, leaf, self.bounds_width // 2, self.use_dop14f_host)
            for k in qd.static(range(3)):
                vi = surf_mgr.surf_triangles[idx, k]
                px = vtx_mgr.positions[vi, 0]
                py = vtx_mgr.positions[vi, 1]
                pz = vtx_mgr.positions[vi, 2]
                aabb_combine_point(
                    self.aabbs,
                    leaf,
                    px,
                    py,
                    pz,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
            aabb_expand(
                self.aabbs,
                leaf,
                d_hat,
                self.bounds_width // 2,
                self.use_dop14f_host,
            )

    @qd.func(requires_top_level=True)
    def calc_leaf_aabb_edge_toy(self, surf_mgr: qd.template(), vtx_mgr: qd.template(), d_hat: qd.f64):
        """Toy-mode: leaf AABBs for edges with d_hat expansion.

        Matches cgq ``toy/lbvh.cu::calc_leaf_aabb_edge_kernel``.
        """
        n = self.n_prims[()]
        for idx in range(n):
            leaf = n - 1 + idx
            aabb_init(self.aabbs, leaf, self.bounds_width // 2, self.use_dop14f_host)
            for k in qd.static(range(2)):
                vi = surf_mgr.surf_edges[idx, k]
                px = vtx_mgr.positions[vi, 0]
                py = vtx_mgr.positions[vi, 1]
                pz = vtx_mgr.positions[vi, 2]
                aabb_combine_point(
                    self.aabbs,
                    leaf,
                    px,
                    py,
                    pz,
                    self.bounds_width // 2,
                    self.use_dop14f_host,
                )
            aabb_expand(
                self.aabbs,
                leaf,
                d_hat,
                self.bounds_width // 2,
                self.use_dop14f_host,
            )

    @qd.func(requires_top_level=True)
    def query_pt_toy(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
    ):
        """Toy-mode PT query: query AABB expanded by d_hat + plain aabb_overlap.

        Matches cgq ``toy/lbvh.cu::query_pt_kernel``.
        """
        n_queries = surf_mgr.n_surf_verts[()]

        for idx in range(n_queries):
            vidx = surf_mgr.surf_verts[idx]
            vx = vtx_mgr.positions[vidx, 0]
            vy = vtx_mgr.positions[vidx, 1]
            vz = vtx_mgr.positions[vidx, 2]

            q_lx = vx - d_hat
            q_ly = vy - d_hat
            q_lz = vz - d_hat
            q_ux = vx + d_hat
            q_uy = vy + d_hat
            q_uz = vz + d_hat

            stack_top = qd.i32(0)
            self.stack_pool[idx, 0] = qd.u32(0)
            stack_top = 1

            while stack_top > 0:
                stack_top = stack_top - 1
                node_id = qd.i32(self.stack_pool[idx, stack_top])
                L_idx = qd.i32(self.nodes_left[node_id])
                R_idx = qd.i32(self.nodes_right[node_id])

                # Process left child — plain aabb_overlap
                L_overlap = qd.i32(1)
                if self.aabbs[L_idx, 0] > q_ux:
                    L_overlap = 0
                if q_lx > self.aabbs[L_idx, 3]:
                    L_overlap = 0
                if self.aabbs[L_idx, 1] > q_uy:
                    L_overlap = 0
                if q_ly > self.aabbs[L_idx, 4]:
                    L_overlap = 0
                if self.aabbs[L_idx, 2] > q_uz:
                    L_overlap = 0
                if q_lz > self.aabbs[L_idx, 5]:
                    L_overlap = 0
                if L_overlap != 0:
                    L_elem = self.nodes_element[L_idx]
                    if L_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(L_idx)
                        stack_top = stack_top + 1
                    else:
                        face_idx = qd.i32(L_elem)
                        bi = vtx_mgr.body_id[vidx]
                        fv0 = surf_mgr.surf_triangles[face_idx, 0]
                        fv1 = surf_mgr.surf_triangles[face_idx, 1]
                        fv2 = surf_mgr.surf_triangles[face_idx, 2]
                        bj = vtx_mgr.body_id[fv0]
                        accept = qd.i32(1)
                        if bi == bj:
                            if bi >= 0:
                                accept = 0
                        if vidx == fv0:
                            accept = 0
                        if vidx == fv1:
                            accept = 0
                        if vidx == fv2:
                            accept = 0
                        if accept != 0:
                            cp_idx = qd.atomic_add(n_pairs[()], 1)
                            if cp_idx < max_pairs_val:
                                pairs[cp_idx, 0] = idx
                                pairs[cp_idx, 1] = face_idx

                # Process right child — plain aabb_overlap
                R_overlap = qd.i32(1)
                if self.aabbs[R_idx, 0] > q_ux:
                    R_overlap = 0
                if q_lx > self.aabbs[R_idx, 3]:
                    R_overlap = 0
                if self.aabbs[R_idx, 1] > q_uy:
                    R_overlap = 0
                if q_ly > self.aabbs[R_idx, 4]:
                    R_overlap = 0
                if self.aabbs[R_idx, 2] > q_uz:
                    R_overlap = 0
                if q_lz > self.aabbs[R_idx, 5]:
                    R_overlap = 0
                if R_overlap != 0:
                    R_elem = self.nodes_element[R_idx]
                    if R_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(R_idx)
                        stack_top = stack_top + 1
                    else:
                        face_idx2 = qd.i32(R_elem)
                        bi2 = vtx_mgr.body_id[vidx]
                        fv02 = surf_mgr.surf_triangles[face_idx2, 0]
                        fv12 = surf_mgr.surf_triangles[face_idx2, 1]
                        fv22 = surf_mgr.surf_triangles[face_idx2, 2]
                        bj2 = vtx_mgr.body_id[fv02]
                        accept2 = qd.i32(1)
                        if bi2 == bj2:
                            if bi2 >= 0:
                                accept2 = 0
                        if vidx == fv02:
                            accept2 = 0
                        if vidx == fv12:
                            accept2 = 0
                        if vidx == fv22:
                            accept2 = 0
                        if accept2 != 0:
                            cp_idx2 = qd.atomic_add(n_pairs[()], 1)
                            if cp_idx2 < max_pairs_val:
                                pairs[cp_idx2, 0] = idx
                                pairs[cp_idx2, 1] = face_idx2

    @qd.func(requires_top_level=True)
    def query_ee_toy(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
    ):
        """Toy-mode EE self-query: leaf AABB already expanded, plain aabb_overlap.

        Matches cgq ``toy/lbvh.cu::query_ee_kernel``.
        """
        n = self.n_prims[()]

        for idx in range(n):
            leaf_idx = idx + n - 1
            self_eid = self.nodes_element[leaf_idx]

            q_lx = self.aabbs[leaf_idx, 0]
            q_ly = self.aabbs[leaf_idx, 1]
            q_lz = self.aabbs[leaf_idx, 2]
            q_ux = self.aabbs[leaf_idx, 3]
            q_uy = self.aabbs[leaf_idx, 4]
            q_uz = self.aabbs[leaf_idx, 5]

            stack_top = qd.i32(0)
            self.stack_pool[idx, 0] = qd.u32(0)
            stack_top = 1

            while stack_top > 0:
                stack_top = stack_top - 1
                node_id = qd.i32(self.stack_pool[idx, stack_top])
                L_idx = qd.i32(self.nodes_left[node_id])
                R_idx = qd.i32(self.nodes_right[node_id])

                # Process left child — plain aabb_overlap
                L_overlap = qd.i32(1)
                if self.aabbs[L_idx, 0] > q_ux:
                    L_overlap = 0
                if q_lx > self.aabbs[L_idx, 3]:
                    L_overlap = 0
                if self.aabbs[L_idx, 1] > q_uy:
                    L_overlap = 0
                if q_ly > self.aabbs[L_idx, 4]:
                    L_overlap = 0
                if self.aabbs[L_idx, 2] > q_uz:
                    L_overlap = 0
                if q_lz > self.aabbs[L_idx, 5]:
                    L_overlap = 0
                if L_overlap != 0:
                    L_elem = self.nodes_element[L_idx]
                    if L_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(L_idx)
                        stack_top = stack_top + 1
                    else:
                        obj_idx = L_elem
                        if obj_idx > self_eid:
                            ea0 = surf_mgr.surf_edges[qd.i32(self_eid), 0]
                            ea1 = surf_mgr.surf_edges[qd.i32(self_eid), 1]
                            eb0 = surf_mgr.surf_edges[qd.i32(obj_idx), 0]
                            eb1 = surf_mgr.surf_edges[qd.i32(obj_idx), 1]
                            bi = vtx_mgr.body_id[ea0]
                            bj = vtx_mgr.body_id[eb0]
                            accept = qd.i32(1)
                            if bi == bj:
                                if bi >= 0:
                                    accept = 0
                            if ea0 == eb0:
                                accept = 0
                            if ea0 == eb1:
                                accept = 0
                            if ea1 == eb0:
                                accept = 0
                            if ea1 == eb1:
                                accept = 0
                            if accept != 0:
                                cp_idx = qd.atomic_add(n_pairs[()], 1)
                                if cp_idx < max_pairs_val:
                                    pairs[cp_idx, 0] = qd.i32(self_eid)
                                    pairs[cp_idx, 1] = qd.i32(obj_idx)

                # Process right child — plain aabb_overlap
                R_overlap = qd.i32(1)
                if self.aabbs[R_idx, 0] > q_ux:
                    R_overlap = 0
                if q_lx > self.aabbs[R_idx, 3]:
                    R_overlap = 0
                if self.aabbs[R_idx, 1] > q_uy:
                    R_overlap = 0
                if q_ly > self.aabbs[R_idx, 4]:
                    R_overlap = 0
                if self.aabbs[R_idx, 2] > q_uz:
                    R_overlap = 0
                if q_lz > self.aabbs[R_idx, 5]:
                    R_overlap = 0
                if R_overlap != 0:
                    R_elem = self.nodes_element[R_idx]
                    if R_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(R_idx)
                        stack_top = stack_top + 1
                    else:
                        obj_idx2 = R_elem
                        if obj_idx2 > self_eid:
                            ea02 = surf_mgr.surf_edges[qd.i32(self_eid), 0]
                            ea12 = surf_mgr.surf_edges[qd.i32(self_eid), 1]
                            eb02 = surf_mgr.surf_edges[qd.i32(obj_idx2), 0]
                            eb12 = surf_mgr.surf_edges[qd.i32(obj_idx2), 1]
                            bi2 = vtx_mgr.body_id[ea02]
                            bj2 = vtx_mgr.body_id[eb02]
                            accept2 = qd.i32(1)
                            if bi2 == bj2:
                                if bi2 >= 0:
                                    accept2 = 0
                            if ea02 == eb02:
                                accept2 = 0
                            if ea02 == eb12:
                                accept2 = 0
                            if ea12 == eb02:
                                accept2 = 0
                            if ea12 == eb12:
                                accept2 = 0
                            if accept2 != 0:
                                cp_idx2 = qd.atomic_add(n_pairs[()], 1)
                                if cp_idx2 < max_pairs_val:
                                    pairs[cp_idx2, 0] = qd.i32(self_eid)
                                    pairs[cp_idx2, 1] = qd.i32(obj_idx2)

    # ======================================================================
    # Full build sequence (convenience, calls all steps in order)
    # Note: query_ee_toy above has no overflow_flag (harness-only path)
    # ======================================================================

    def build_tri(self, surf_mgr, vtx_mgr):
        """Full BVH build for triangles (call from ``_step_kernel`` top level)."""
        self.calc_leaf_aabb_tri(surf_mgr, vtx_mgr)
        self.reduce_scene_aabb()
        self.calc_morton()
        self.sort_morton()
        self.extract_indices()
        self.copy_leaf_aabb_to_temp()
        self.reorder_leaf_aabb()
        self.calc_leaf_nodes()
        self.calc_internal_nodes()
        self.memset_flags()
        self.calc_internal_aabb()

    def build_edge(self, surf_mgr, vtx_mgr):
        """Full BVH build for edges (call from ``_step_kernel`` top level)."""
        self.calc_leaf_aabb_edge(surf_mgr, vtx_mgr)
        self.reduce_scene_aabb()
        self.calc_morton()
        self.sort_morton()
        self.extract_indices()
        self.copy_leaf_aabb_to_temp()
        self.reorder_leaf_aabb()
        self.calc_leaf_nodes()
        self.calc_internal_nodes()
        self.memset_flags()
        self.calc_internal_aabb()

    # ======================================================================
    # QUERY: PT broadphase (surface vertices vs triangle BVH)
    # ======================================================================

    @qd.func(requires_top_level=True)
    def query_pt_batched(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        body_mgr: qd.template(),
        contact: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
        overflow_flag: qd.template(),
    ):
        """Retained per-thread PT swept broadphase.

        Matches CGQ ``pt_query_batched`` and remains available as the explicit
        comparison path. Sets ``overflow_flag`` to 1 when the candidate count
        exceeds ``max_pairs_val``.
        Output pairs: ``(surf_vert_idx, face_idx)``.
        """
        n_queries = surf_mgr.n_surf_verts[()]

        for idx in range(n_queries):
            vidx = surf_mgr.surf_verts[idx]
            vx = vtx_mgr.positions[vidx, 0]
            vy = vtx_mgr.positions[vidx, 1]
            vz = vtx_mgr.positions[vidx, 2]
            endpoint_x = vtx_mgr.trajectory_end_positions[vidx, 0]
            endpoint_y = vtx_mgr.trajectory_end_positions[vidx, 1]
            endpoint_z = vtx_mgr.trajectory_end_positions[vidx, 2]

            q_lx = qd.min(vx, endpoint_x)
            q_ly = qd.min(vy, endpoint_y)
            q_lz = qd.min(vz, endpoint_z)
            q_ux = qd.max(vx, endpoint_x)
            q_uy = qd.max(vy, endpoint_y)
            q_uz = qd.max(vz, endpoint_z)
            query_gap = vtx_mgr.d_hats[vidx] + vtx_mgr.thicknesses[vidx] + vtx_mgr.path_inflation[vidx]
            query_start = qd.Vector([vx, vy, vz])
            query_endpoint = qd.Vector([endpoint_x, endpoint_y, endpoint_z])

            stack_top = qd.i32(0)
            if self.n_prims[()] == 1:
                root_overlap = qd.i32(1)
                if qd.static(self.bounds_width == 14):
                    root_overlap = _swept_point_overlap(
                        self.aabbs,
                        0,
                        query_start,
                        query_endpoint,
                        query_gap,
                        self.bounds_width // 2,
                        self.use_dop14f_host,
                    )
                else:
                    for axis in qd.static(range(3)):
                        if (
                            self.aabbs[0, axis]
                            - qd.max(
                                query_start[axis],
                                query_endpoint[axis],
                            )
                        ) >= query_gap or (
                            qd.min(
                                query_start[axis],
                                query_endpoint[axis],
                            )
                            - self.aabbs[0, axis + 3]
                        ) >= query_gap:
                            root_overlap = 0
                if not _node_pair_enabled(
                    body_mgr,
                    vtx_mgr.body_id[vidx],
                    self.node_body_id[0],
                ):
                    root_overlap = 0
                root_face = qd.i32(self.nodes_element[0])
                if root_overlap != 0 and _pt_pair_enabled(
                    surf_mgr,
                    vtx_mgr,
                    body_mgr,
                    contact,
                    vidx,
                    root_face,
                ):
                    root_output = qd.atomic_add(n_pairs[()], 1)
                    if root_output < max_pairs_val:
                        pairs[root_output, 0] = idx
                        pairs[root_output, 1] = root_face
                    else:
                        overflow_flag[()] = 1
            else:
                self.stack_pool[idx, 0] = qd.u32(0)
                stack_top = 1

            while stack_top > 0:
                stack_top = stack_top - 1
                node_id = qd.i32(self.stack_pool[idx, stack_top])
                L_idx = qd.i32(self.nodes_left[node_id])
                R_idx = qd.i32(self.nodes_right[node_id])

                # Process left child
                L_lo_x = self.aabbs[L_idx, 0]
                L_lo_y = self.aabbs[L_idx, 1]
                L_lo_z = self.aabbs[L_idx, 2]
                L_hi_x = self.aabbs[L_idx, 3]
                L_hi_y = self.aabbs[L_idx, 4]
                L_hi_z = self.aabbs[L_idx, 5]
                L_overlap = qd.i32(1)
                if (L_lo_x - q_ux) >= query_gap:
                    L_overlap = 0
                if (q_lx - L_hi_x) >= query_gap:
                    L_overlap = 0
                if (L_lo_y - q_uy) >= query_gap:
                    L_overlap = 0
                if (q_ly - L_hi_y) >= query_gap:
                    L_overlap = 0
                if (L_lo_z - q_uz) >= query_gap:
                    L_overlap = 0
                if (q_lz - L_hi_z) >= query_gap:
                    L_overlap = 0
                if qd.static(self.bounds_width == 14):
                    L_overlap = _swept_point_overlap(
                        self.aabbs,
                        L_idx,
                        query_start,
                        query_endpoint,
                        query_gap,
                        self.bounds_width // 2,
                        self.use_dop14f_host,
                    )
                if not _node_pair_enabled(body_mgr, vtx_mgr.body_id[vidx], self.node_body_id[L_idx]):
                    L_overlap = 0
                if L_overlap != 0:
                    L_elem = self.nodes_element[L_idx]
                    if L_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(L_idx)
                        stack_top = stack_top + 1
                    else:
                        face_idx = qd.i32(L_elem)
                        accept = _pt_pair_enabled(
                            surf_mgr,
                            vtx_mgr,
                            body_mgr,
                            contact,
                            vidx,
                            face_idx,
                        )
                        if accept != 0:
                            cp_idx = qd.atomic_add(n_pairs[()], 1)
                            if cp_idx < max_pairs_val:
                                pairs[cp_idx, 0] = idx
                                pairs[cp_idx, 1] = face_idx
                            else:
                                overflow_flag[()] = 1

                # Process right child
                R_lo_x = self.aabbs[R_idx, 0]
                R_lo_y = self.aabbs[R_idx, 1]
                R_lo_z = self.aabbs[R_idx, 2]
                R_hi_x = self.aabbs[R_idx, 3]
                R_hi_y = self.aabbs[R_idx, 4]
                R_hi_z = self.aabbs[R_idx, 5]
                R_overlap = qd.i32(1)
                if (R_lo_x - q_ux) >= query_gap:
                    R_overlap = 0
                if (q_lx - R_hi_x) >= query_gap:
                    R_overlap = 0
                if (R_lo_y - q_uy) >= query_gap:
                    R_overlap = 0
                if (q_ly - R_hi_y) >= query_gap:
                    R_overlap = 0
                if (R_lo_z - q_uz) >= query_gap:
                    R_overlap = 0
                if (q_lz - R_hi_z) >= query_gap:
                    R_overlap = 0
                if qd.static(self.bounds_width == 14):
                    R_overlap = _swept_point_overlap(
                        self.aabbs,
                        R_idx,
                        query_start,
                        query_endpoint,
                        query_gap,
                        self.bounds_width // 2,
                        self.use_dop14f_host,
                    )
                if not _node_pair_enabled(body_mgr, vtx_mgr.body_id[vidx], self.node_body_id[R_idx]):
                    R_overlap = 0
                if R_overlap != 0:
                    R_elem = self.nodes_element[R_idx]
                    if R_elem == qd.u32(self.sentinel):
                        self.stack_pool[idx, stack_top] = qd.u32(R_idx)
                        stack_top = stack_top + 1
                    else:
                        face_idx2 = qd.i32(R_elem)
                        accept2 = _pt_pair_enabled(
                            surf_mgr,
                            vtx_mgr,
                            body_mgr,
                            contact,
                            vidx,
                            face_idx2,
                        )
                        if accept2 != 0:
                            cp_idx2 = qd.atomic_add(n_pairs[()], 1)
                            if cp_idx2 < max_pairs_val:
                                pairs[cp_idx2, 0] = idx
                                pairs[cp_idx2, 1] = face_idx2
                            else:
                                overflow_flag[()] = 1

    @qd.func(requires_top_level=True)
    def query_pt_warp(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        body_mgr: qd.template(),
        contact: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
        overflow_flag: qd.template(),
    ):
        """CGQ warp-per-query swept PT traversal."""
        n_queries = surf_mgr.n_surf_verts[()]
        n = self.n_prims[()]
        group_size = qd_subgroup.group_size()
        qd.loop_config(name="query_pt_warp", block_dim=self.pt_warp_block)
        for thread in range(self.pt_warp_threads):
            query = thread // group_size
            lane = qd_subgroup.invocation_id()
            subgroup_in_block = (thread % self.pt_warp_block) // group_size
            stack = qd_block.SharedArray(
                (
                    self.pt_warp_subgroups_per_block,
                    self.pt_warp_stack_capacity,
                ),
                qd.u32,
            )

            while query < n_queries:
                vertex_index = surf_mgr.surf_verts[query]
                start = qd.Vector(
                    [
                        vtx_mgr.positions[vertex_index, 0],
                        vtx_mgr.positions[vertex_index, 1],
                        vtx_mgr.positions[vertex_index, 2],
                    ]
                )
                endpoint = qd.Vector(
                    [
                        vtx_mgr.trajectory_end_positions[vertex_index, 0],
                        vtx_mgr.trajectory_end_positions[vertex_index, 1],
                        vtx_mgr.trajectory_end_positions[vertex_index, 2],
                    ]
                )
                query_lower = qd.Vector(
                    [
                        qd.min(start[0], endpoint[0]),
                        qd.min(start[1], endpoint[1]),
                        qd.min(start[2], endpoint[2]),
                    ]
                )
                query_upper = qd.Vector(
                    [
                        qd.max(start[0], endpoint[0]),
                        qd.max(start[1], endpoint[1]),
                        qd.max(start[2], endpoint[2]),
                    ]
                )
                query_gap = (
                    vtx_mgr.d_hats[vertex_index]
                    + vtx_mgr.thicknesses[vertex_index]
                    + vtx_mgr.path_inflation[vertex_index]
                )
                query_projection_lower = qd.Vector.zero(qd.f64, 7)
                query_projection_upper = qd.Vector.zero(qd.f64, 7)
                query_projection_lower_gap_f32 = qd.Vector.zero(qd.f32, 7)
                query_projection_upper_gap_f32 = qd.Vector.zero(qd.f32, 7)
                if qd.static(self.bounds_width == 14):
                    start_projection = qd.Vector(
                        [
                            start[0],
                            start[1],
                            start[2],
                            start[0] + start[1] + start[2],
                            start[0] + start[1] - start[2],
                            start[0] - start[1] + start[2],
                            start[0] - start[1] - start[2],
                        ]
                    )
                    endpoint_projection = qd.Vector(
                        [
                            endpoint[0],
                            endpoint[1],
                            endpoint[2],
                            endpoint[0] + endpoint[1] + endpoint[2],
                            endpoint[0] + endpoint[1] - endpoint[2],
                            endpoint[0] - endpoint[1] + endpoint[2],
                            endpoint[0] - endpoint[1] - endpoint[2],
                        ]
                    )
                    for axis in qd.static(range(7)):
                        projection_lower = qd.min(
                            start_projection[axis],
                            endpoint_projection[axis],
                        )
                        projection_upper = qd.max(
                            start_projection[axis],
                            endpoint_projection[axis],
                        )
                        if qd.static(self.use_dop14f_host):
                            scale = qd.f64(1.0)
                            if qd.static(axis >= 3):
                                scale = qd.f64(1.7320508075688772)
                            axis_gap = query_gap * scale
                            query_projection_lower_gap_f32[axis] = f64_to_f32_rd(projection_lower - axis_gap)
                            query_projection_upper_gap_f32[axis] = f64_to_f32_ru(projection_upper + axis_gap)
                        else:
                            query_projection_lower[axis] = projection_lower
                            query_projection_upper[axis] = projection_upper
                query_body = vtx_mgr.body_id[vertex_index]

                if n == 1:
                    emit = False
                    face = qd.i32(0)
                    if lane == 0:
                        overlap = qd.i32(1)
                        if qd.static(self.bounds_width == 14):
                            for axis in qd.static(range(7)):
                                if qd.static(self.use_dop14f_host):
                                    if self.aabbs[0, axis] >= query_projection_upper_gap_f32[axis] or (
                                        query_projection_lower_gap_f32[axis] >= self.aabbs[0, axis + 7]
                                    ):
                                        overlap = 0
                                else:
                                    axis_gap = query_gap
                                    if qd.static(axis >= 3):
                                        axis_gap = query_gap * 1.7320508075688772
                                    if (self.aabbs[0, axis] - query_projection_upper[axis]) >= axis_gap or (
                                        query_projection_lower[axis] - self.aabbs[0, axis + 7]
                                    ) >= axis_gap:
                                        overlap = 0
                        else:
                            for axis in qd.static(range(3)):
                                if (self.aabbs[0, axis] - query_upper[axis]) >= query_gap or (
                                    query_lower[axis] - self.aabbs[0, axis + 3]
                                ) >= query_gap:
                                    overlap = 0
                        if not _node_pair_enabled(
                            body_mgr,
                            query_body,
                            self.node_body_id[0],
                        ):
                            overlap = 0
                        face = qd.i32(self.nodes_element[0])
                        if overlap != 0 and _pt_pair_enabled(
                            surf_mgr,
                            vtx_mgr,
                            body_mgr,
                            contact,
                            vertex_index,
                            face,
                        ):
                            emit = True
                    _pt_emit_pair(
                        emit,
                        query,
                        face,
                        pairs,
                        n_pairs,
                        max_pairs_val,
                        overflow_flag,
                    )
                elif n > 1:
                    if lane == 0:
                        stack[subgroup_in_block, 0] = qd.u32(0)
                    qd_subgroup.mem_fence()
                    qd_subgroup.sync()
                    top = qd.i32(1)

                    while top > 0:
                        n_pop = qd.min(top, group_size)
                        room = self.pt_warp_stack_capacity - top
                        n_pop = qd.min(n_pop, room)
                        assert n_pop >= 1, "query_pt_warp: frontier stack full"
                        if n_pop > 0:
                            active = lane < n_pop
                            node = qd.i32(0)
                            if active:
                                node = qd.i32(
                                    stack[
                                        subgroup_in_block,
                                        top - 1 - lane,
                                    ]
                                )
                            qd_subgroup.sync()
                            top = top - n_pop

                            children = qd.Vector.zero(qd.i32, 2)
                            if active:
                                children[0] = qd.i32(self.nodes_left[node])
                                children[1] = qd.i32(self.nodes_right[node])

                            for child_slot in qd.static(range(2)):
                                push = False
                                emit = False
                                push_node = qd.i32(0)
                                emit_face = qd.i32(0)
                                if active:
                                    child = children[child_slot]
                                    overlap = qd.i32(1)
                                    if qd.static(self.bounds_width == 14):
                                        for axis in qd.static(range(7)):
                                            if qd.static(self.use_dop14f_host):
                                                if (
                                                    self.aabbs[
                                                        child,
                                                        axis,
                                                    ]
                                                    >= query_projection_upper_gap_f32[axis]
                                                    or query_projection_lower_gap_f32[axis]
                                                    >= self.aabbs[
                                                        child,
                                                        axis + 7,
                                                    ]
                                                ):
                                                    overlap = 0
                                            else:
                                                axis_gap = query_gap
                                                if qd.static(axis >= 3):
                                                    axis_gap = query_gap * 1.7320508075688772
                                                if (
                                                    self.aabbs[child, axis] - query_projection_upper[axis]
                                                ) >= axis_gap or (
                                                    query_projection_lower[axis] - self.aabbs[child, axis + 7]
                                                ) >= axis_gap:
                                                    overlap = 0
                                    else:
                                        for axis in qd.static(range(3)):
                                            if (self.aabbs[child, axis] - query_upper[axis]) >= query_gap or (
                                                query_lower[axis] - self.aabbs[child, axis + 3]
                                            ) >= query_gap:
                                                overlap = 0
                                    if not _node_pair_enabled(
                                        body_mgr,
                                        query_body,
                                        self.node_body_id[child],
                                    ):
                                        overlap = 0

                                    if overlap != 0:
                                        element = self.nodes_element[child]
                                        if element == qd.u32(self.sentinel):
                                            push = True
                                            push_node = child
                                        else:
                                            face = qd.i32(element)
                                            if _pt_pair_enabled(
                                                surf_mgr,
                                                vtx_mgr,
                                                body_mgr,
                                                contact,
                                                vertex_index,
                                                face,
                                            ):
                                                emit = True
                                                emit_face = face

                                push_mask = qd_subgroup.ballot(qd.i32(push))
                                push_count = qd.i32(qd.math.popcnt(push_mask))
                                if push:
                                    lane_lt = (qd.u64(1) << qd.u64(lane)) - qd.u64(1)
                                    output = top + qd.i32(qd.math.popcnt(push_mask & lane_lt))
                                    stack[subgroup_in_block, output] = qd.u32(push_node)
                                top = top + push_count

                                _pt_emit_pair(
                                    emit,
                                    query,
                                    emit_face,
                                    pairs,
                                    n_pairs,
                                    max_pairs_val,
                                    overflow_flag,
                                )
                                qd_subgroup.mem_fence()
                                qd_subgroup.sync()
                query = query + self.pt_warp_workers

    @qd.func
    def _query_et_child(
        self,
        child,
        edge,
        edge_position_a: qd.template(),
        edge_position_b: qd.template(),
        stack_top,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        body_mgr: qd.template(),
        contact: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs,
        overflow_flag: qd.template(),
    ):
        if _edge_node_overlap(
            self.aabbs,
            child,
            edge_position_a,
            edge_position_b,
            self.bounds_width // 2,
        ):
            element = self.nodes_element[child]
            if element == qd.u32(self.sentinel):
                if stack_top < self.stack_capacity:
                    self.stack_pool[edge, stack_top] = qd.u32(child)
                    stack_top = stack_top + 1
            else:
                face = qd.i32(element)
                if _edge_triangle_intersects(
                    surf_mgr,
                    vtx_mgr,
                    body_mgr,
                    contact,
                    edge,
                    face,
                ):
                    output = qd.atomic_add(n_pairs[()], 1)
                    if output < max_pairs:
                        pairs[output, 0] = edge
                        pairs[output, 1] = face
                    else:
                        overflow_flag[()] = 1
        return stack_top

    @qd.func(requires_top_level=True)
    def query_et(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        body_mgr: qd.template(),
        contact: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs,
        overflow_flag: qd.template(),
    ):
        for edge in range(surf_mgr.n_surf_edges[()]):
            if contact.intersection_check[()] != 0 and self.n_prims[()] > 1:
                edge_a = surf_mgr.surf_edges[edge, 0]
                edge_b = surf_mgr.surf_edges[edge, 1]
                edge_position_a = qd.Vector(
                    [
                        vtx_mgr.positions[edge_a, 0],
                        vtx_mgr.positions[edge_a, 1],
                        vtx_mgr.positions[edge_a, 2],
                    ]
                )
                edge_position_b = qd.Vector(
                    [
                        vtx_mgr.positions[edge_b, 0],
                        vtx_mgr.positions[edge_b, 1],
                        vtx_mgr.positions[edge_b, 2],
                    ]
                )
                stack_top = qd.i32(1)
                self.stack_pool[edge, 0] = qd.u32(0)
                while stack_top > 0:
                    stack_top = stack_top - 1
                    node = qd.i32(self.stack_pool[edge, stack_top])
                    stack_top = self._query_et_child(
                        qd.i32(self.nodes_left[node]),
                        edge,
                        edge_position_a,
                        edge_position_b,
                        stack_top,
                        surf_mgr,
                        vtx_mgr,
                        body_mgr,
                        contact,
                        pairs,
                        n_pairs,
                        max_pairs,
                        overflow_flag,
                    )
                    stack_top = self._query_et_child(
                        qd.i32(self.nodes_right[node]),
                        edge,
                        edge_position_a,
                        edge_position_b,
                        stack_top,
                        surf_mgr,
                        vtx_mgr,
                        body_mgr,
                        contact,
                        pairs,
                        n_pairs,
                        max_pairs,
                        overflow_flag,
                    )

    # ======================================================================
    # QUERY: warp-cooperative EE self-query (edge vs edge BVH)
    # ======================================================================

    @qd.func(requires_top_level=True)
    def query_ee_warp(
        self,
        surf_mgr: qd.template(),
        vtx_mgr: qd.template(),
        body_mgr: qd.template(),
        contact: qd.template(),
        pairs: qd.template(),
        n_pairs: qd.template(),
        max_pairs_val: qd.i32,
        d_hat: qd.f64,
        overflow_flag: qd.template(),
    ):
        qd.loop_config(name="query_ee_warp_reset")
        for _ in range(1):
            self.ee_warp_stack_overflow[()] = 0

        n = self.n_prims[()]
        group_size = qd_subgroup.group_size()
        qd.loop_config(name="query_ee_warp", block_dim=self.ee_warp_block)
        for thread in range(self.ee_warp_threads):
            query = thread // group_size
            lane = qd_subgroup.invocation_id()
            subgroup_in_block = (thread % self.ee_warp_block) // group_size
            stack = qd_block.SharedArray((self.ee_warp_subgroups_per_block, self.ee_warp_stack_capacity), qd.u32)

            while query < n:
                if n > 1:
                    leaf = query + n - 1
                    self_eid = qd.i32(self.nodes_element[leaf])
                    query_body = vtx_mgr.body_id[surf_mgr.surf_edges[self_eid, 0]]
                    if lane == 0:
                        stack[subgroup_in_block, 0] = qd.u32(0)
                    qd_subgroup.mem_fence()
                    qd_subgroup.sync()
                    top = qd.i32(1)

                    while top > 0:
                        n_pop = qd.min(top, group_size)
                        room = self.ee_warp_stack_capacity - top
                        n_pop = qd.min(n_pop, room)
                        if n_pop <= 0:
                            if lane == 0:
                                self.ee_warp_stack_overflow[()] = 1
                                qd.atomic_or(overflow_flag[()], 1)
                            top = 0
                        else:
                            active = lane < n_pop
                            node = qd.i32(0)
                            if active:
                                node = qd.i32(stack[subgroup_in_block, top - 1 - lane])
                            qd_subgroup.sync()
                            top = top - n_pop

                            children = qd.Vector.zero(qd.i32, 2)
                            if active:
                                children[0] = qd.i32(self.nodes_left[node])
                                children[1] = qd.i32(self.nodes_right[node])

                            for child_slot in qd.static(range(2)):
                                push = False
                                emit = False
                                push_node = qd.i32(0)
                                emit_edge = qd.i32(0)
                                if active:
                                    child = children[child_slot]
                                    overlap = aabb_overlap(
                                        self.aabbs,
                                        leaf,
                                        self.aabbs,
                                        child,
                                        self.bounds_width // 2,
                                    )
                                    if d_hat != 0.0 and qd.static(self.bounds_width == 6):
                                        for axis in qd.static(range(3)):
                                            if (self.aabbs[child, axis] - self.aabbs[leaf, axis + 3]) >= d_hat:
                                                overlap = 0
                                            if (self.aabbs[leaf, axis] - self.aabbs[child, axis + 3]) >= d_hat:
                                                overlap = 0
                                    if overlap != 0 and _node_pair_enabled(
                                        body_mgr,
                                        query_body,
                                        self.node_body_id[child],
                                    ):
                                        element = self.nodes_element[child]
                                        if element == qd.u32(self.sentinel):
                                            push = True
                                            push_node = child
                                        elif element > qd.u32(self_eid):
                                            other_edge = qd.i32(element)
                                            if _ee_pair_enabled(
                                                surf_mgr,
                                                vtx_mgr,
                                                body_mgr,
                                                contact,
                                                self_eid,
                                                other_edge,
                                            ):
                                                emit = True
                                                emit_edge = other_edge

                                push_mask = qd_subgroup.ballot(qd.i32(push))
                                push_count = qd.i32(qd.math.popcnt(push_mask))
                                if push:
                                    lane_lt = (qd.u64(1) << qd.u64(lane)) - qd.u64(1)
                                    output = top + qd.i32(qd.math.popcnt(push_mask & lane_lt))
                                    stack[subgroup_in_block, output] = qd.u32(push_node)
                                top = top + push_count

                                _ee_emit_pair(
                                    emit,
                                    self_eid,
                                    emit_edge,
                                    pairs,
                                    n_pairs,
                                    max_pairs_val,
                                    overflow_flag,
                                )
                                qd_subgroup.mem_fence()
                                qd_subgroup.sync()
                query = query + self.ee_warp_workers
