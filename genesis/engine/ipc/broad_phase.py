"""
Edge-face (EF) broad phase of a surface of triangles, edges and vertices moving over a sweep, on 14-DOP trees (see
genesis/engine/ipc/dop_bvh.py): the point-triangle (PT) and edge-edge (EE) pairs whose bounds come within their gaps.

Every edge owns the vertices whose lowest-index edge it is, and every triangle the edges whose lowest-index triangle it
is, so every vertex has one owner edge and every edge one owner triangle (a surface without edge-free vertices and
without edges outside triangles). A triangle stands in its tree as its proxy, the union of its bound with the bounds of
the edges it owns. One query per edge, its bound grown by the largest gap of the vertices it owns, collects the proxies
it meets, then expands each hit into sub-pairs: the vertices the edge owns against the triangle (PT), and the edge
against each edge the triangle owns of lower index (EE), each kept when the bounds of the pair come within its gap. An EE
pair is found once, from its edge of higher index, and a PT pair once, from the owner edge of its vertex.

The vertices fall into groups. Before every query the groups bound the sweep of their vertices, and a small tree of
these bounds finds the groups that may meet a group owning triangles; the edges of the other groups query nothing.

Every primitive carries a tag, and two primitives whose tags are both 1 (rigid) form no pair: a query skips the subtrees
whose triangles are all rigid when its edge is rigid, and the sub-pairs drop the rigid pairs. A pair sharing a vertex is
dropped too. The bounds of a vertex, an edge and a triangle are those of the positions of their vertices at the start and
at the end of the sweep, an edge and a triangle grown by the largest thickness of their vertices, by the bulge of their
screw paths and by the activation distance of their first vertex. The gap of a vertex is its activation distance plus its
thickness plus the bulge of its path, and that of an edge-edge pair the larger activation distance of its two edges.
"""

import math
from typing import Callable

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.exclusive_sum import ExclusiveSumScratch, exclusive_sum_sizes, func_exclusive_sum
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.misc import qd_to_numpy
from genesis.utils.simt import qd_segmented_min, qd_segmented_sum, qd_slot_neighbors

from .dop_bvh import (
    DOPBVHHits,
    DOPBVHQueries,
    DOPBVHState,
    func_build_dop_bvh,
    func_dop_add_point,
    func_dop_empty,
    func_dop_expand,
    func_dop_gaps,
    func_dop_merge,
    func_dop_overlap,
    func_query_dop_bvh,
    get_dop_bvh_config,
    get_dop_bvh_data,
)

EF_VERTICES = Size("ef_vertices")
EF_EDGES = Size("ef_edges")
EF_TRIANGLES = Size("ef_triangles")
EF_GROUPS = Size("ef_groups")
EF_SUB_PT = Size("ef_sub_pt")
EF_SUB_EE = Size("ef_sub_ee")

# The factor the hit and sub-pair lists grow by past the count that overflowed them
EF_GROW_FACTOR = 1.5


class EFTopology(Data, kind=DataKind.CONSTANT):
    """
    The ownership of the primitives of the surface, their tags and their groups (see the module docstring).

    `edges_owned_verts` holds the vertices an edge owns, -1 for none, `triangles_owned_edges` the edges a triangle owns.
    A primitive takes the group its vertices share, -1 when they differ, and a group is rigid when all its vertices are.
    """

    vertices_group: qd.Tensor = field(int, shape=(EF_VERTICES,))
    vertices_tag: qd.Tensor = field(int, shape=(EF_VERTICES,))
    edges_owned_verts: qd.Tensor = field((int, 2), shape=(EF_EDGES,))
    edges_group: qd.Tensor = field(int, shape=(EF_EDGES,))
    edges_tag: qd.Tensor = field(int, shape=(EF_EDGES,))
    triangles_owned_edges: qd.Tensor = field((int, 3), shape=(EF_TRIANGLES,))
    triangles_group: qd.Tensor = field(int, shape=(EF_TRIANGLES,))
    triangles_tag: qd.Tensor = field(int, shape=(EF_TRIANGLES,))
    groups_is_rigid: qd.Tensor = field(bool, shape=(EF_GROUPS,))


class EFBounds(Data, kind=DataKind.SCRATCH):
    """The bounds of the vertices, edges and triangles of every environment over the sweep, and the gaps of the vertices
    and of the edge queries (see the module docstring)."""

    vertices_lower: qd.Tensor = field((qd.f32, 7), shape=(EF_VERTICES, ENVS))
    vertices_upper: qd.Tensor = field((qd.f32, 7), shape=(EF_VERTICES, ENVS))
    vertices_gap: qd.Tensor = field(float, shape=(EF_VERTICES, ENVS))
    edges_lower: qd.Tensor = field((qd.f32, 7), shape=(EF_EDGES, ENVS))
    edges_upper: qd.Tensor = field((qd.f32, 7), shape=(EF_EDGES, ENVS))
    triangles_lower: qd.Tensor = field((qd.f32, 7), shape=(EF_TRIANGLES, ENVS))
    triangles_upper: qd.Tensor = field((qd.f32, 7), shape=(EF_TRIANGLES, ENVS))


class EFGroups(Data, kind=DataKind.SCRATCH, shape=(EF_GROUPS, ENVS)):
    """
    The groups of every environment over the sweep: the box of the positions of their vertices, the largest gap of
    these vertices, the counts of their vertices and of the triangles with a vertex in them, and whether their edges
    query (see the module docstring).
    """

    lower: qd.Tensor = field((float, 3))
    upper: qd.Tensor = field((float, 3))
    gap: qd.Tensor = field(float)
    n_vertices: qd.Tensor = field(int)
    n_triangles: qd.Tensor = field(int)
    is_active: qd.Tensor = field(bool)


class EFSubPairs(Data, kind=DataKind.SCRATCH):
    """
    The sub-pairs of the hits of the edge queries: PT pairs (environment, vertex, triangle) and EE pairs (environment,
    edge, edge), the first edge of lower index, and their counts, which grow past the capacities on overflow.

    `summary` gathers the counts a caller reads back at once after the pairs are filtered: the hits of the groups and of
    the edge queries, the PT and EE sub-pairs, the PT and EE counts of the filtered pairs, and the queries that stopped
    short (see DOPBVHHits).
    """

    pt: qd.Tensor = field((int, 3), shape=(EF_SUB_PT,), ndarray=True)
    ee: qd.Tensor = field((int, 3), shape=(EF_SUB_EE,), ndarray=True)
    counts: qd.Tensor = field(int, shape=(2,), ndarray=True)
    summary: qd.Tensor = field(int, shape=(7,), ndarray=True)


class EFQueryList(Data, kind=DataKind.SCRATCH):
    """The flags of the edge queries of every environment that run, as flat indices i_b * n_edges + i_e, and their
    exclusive sum, which places them in the list of active queries."""

    flags: qd.Tensor = field(int, shape=(ENVS * EF_EDGES,), ndarray=True)
    offsets: qd.Tensor = field(int, shape=(ENVS * EF_EDGES,), ndarray=True)
    n_flags: qd.Tensor = field(int, shape=(), ndarray=True)
    scanner: ExclusiveSumScratch


def get_ef_topology(
    edges_verts_idx: np.ndarray,
    triangles_verts_idx: np.ndarray,
    vertices_group: np.ndarray,
    vertices_is_rigid: np.ndarray,
):
    """
    Return the topology of a surface for its EF broad phase, and its group count.

    The edges are given each once, and every triangle side is one of them. The group of every vertex is a non-negative
    index.
    """
    n_vertices, n_edges, n_triangles = len(vertices_group), len(edges_verts_idx), len(triangles_verts_idx)
    # The lowest-index edge of every vertex owns it
    vertices_owner = np.full(n_vertices, n_edges, dtype=gs.np_int)
    np.minimum.at(vertices_owner, edges_verts_idx, np.arange(n_edges)[:, None])
    if (vertices_owner == n_edges).any():
        gs.raise_exception("The EF broad phase takes a surface whose every vertex is on an edge.")
    edges_owned_verts = np.where(vertices_owner[edges_verts_idx] == np.arange(n_edges)[:, None], edges_verts_idx, -1)
    # The edge of every side of every triangle, by the key of its sorted vertices
    edges_sorted = np.sort(edges_verts_idx, axis=-1)
    edges_key = edges_sorted[:, 0].astype(np.int64) * n_vertices + edges_sorted[:, 1]
    edges_order = np.argsort(edges_key)
    sides = np.sort(triangles_verts_idx[:, ((0, 1), (1, 2), (2, 0))], axis=-1)
    sides_key = sides[..., 0].astype(np.int64) * n_vertices + sides[..., 1]
    triangles_edges = edges_order[np.searchsorted(edges_key[edges_order], sides_key)]
    assert (edges_key[triangles_edges] == sides_key).all()
    edges_owner = np.full(n_edges, n_triangles, dtype=gs.np_int)
    np.minimum.at(edges_owner, triangles_edges, np.arange(n_triangles)[:, None])
    if (edges_owner == n_triangles).any():
        gs.raise_exception("The EF broad phase takes a surface whose every edge is on a triangle.")
    triangles_owned_edges = np.where(
        edges_owner[triangles_edges] == np.arange(n_triangles)[:, None], triangles_edges, -1
    )
    # The group a primitive shares with its vertices, -1 across groups, and its tag, 1 when all its vertices are rigid
    edges_group = np.where(
        vertices_group[edges_verts_idx[:, 0]] == vertices_group[edges_verts_idx[:, 1]],
        vertices_group[edges_verts_idx[:, 0]],
        -1,
    )
    triangles_vertices_group = vertices_group[triangles_verts_idx]
    triangles_group = np.where(
        (triangles_vertices_group == triangles_vertices_group[:, :1]).all(axis=-1), triangles_vertices_group[:, 0], -1
    )
    n_groups = int(vertices_group.max()) + 1
    groups_is_rigid = np.ones(n_groups, dtype=bool)
    np.logical_and.at(groups_is_rigid, vertices_group, vertices_is_rigid)
    topology = EFTopology(
        sizes={EF_VERTICES: n_vertices, EF_EDGES: n_edges, EF_TRIANGLES: n_triangles, EF_GROUPS: n_groups},
    )
    topology.vertices_group.from_numpy(vertices_group.astype(gs.np_int))
    topology.vertices_tag.from_numpy(vertices_is_rigid.astype(gs.np_int))
    topology.edges_owned_verts.from_numpy(edges_owned_verts.astype(gs.np_int))
    topology.edges_group.from_numpy(edges_group.astype(gs.np_int))
    topology.edges_tag.from_numpy(vertices_is_rigid[edges_verts_idx].all(axis=-1).astype(gs.np_int))
    topology.triangles_owned_edges.from_numpy(triangles_owned_edges.astype(gs.np_int))
    topology.triangles_group.from_numpy(triangles_group.astype(gs.np_int))
    topology.triangles_tag.from_numpy(vertices_is_rigid[triangles_verts_idx].all(axis=-1).astype(gs.np_int))
    topology.groups_is_rigid.from_numpy(groups_is_rigid)
    return topology, n_groups


# ------------------------------------------------------------------------------------
# ------------------------------------- bounds ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_vertex_sweep_end(i_v: int, i_b: int, positions: qd.Tensor, displacements: qd.Tensor, is_swept: qd.template()):
    """Return the position of a vertex at the end of the sweep, its position without a sweep."""
    end = positions[i_v, i_b]
    if qd.static(is_swept):
        end = end + displacements[i_v, i_b]
    return end


@qd.func
def func_vertex_bulge(
    i_v: int, i_b: int, positions: qd.Tensor, pivots: qd.Tensor, rotations: qd.Tensor, is_screw: qd.template()
) -> float:
    """Return how far the screw path of a vertex bulges out of its chord over the sweep, r |w|^2 / 8 for a lever arm r
    and a rotation w (the sagitta of the arc), zero for a vertex moving straight."""
    bulge = gs.qd_float(0.0)
    if qd.static(is_screw):
        bulge = 0.125 * (positions[i_v, i_b] - pivots[i_v, i_b]).norm() * rotations[i_v, i_b].norm_sqr()
    return bulge


@qd.func
def func_primitive_bounds(
    i_b: int,
    verts_idx,
    positions: qd.Tensor,
    displacements: qd.Tensor,
    pivots: qd.Tensor,
    rotations: qd.Tensor,
    vertices_thickness: qd.Tensor,
    vertices_d_hat: qd.Tensor,
    is_screw: qd.template(),
    is_swept: qd.template(),
):
    """Return the bound of an edge or a triangle over the sweep (see the module docstring)."""
    lower, upper = func_dop_empty()
    thickness = gs.qd_float(0.0)
    bulge = gs.qd_float(0.0)
    for k in qd.static(range(verts_idx.n)):
        i_v = verts_idx[k]
        lower, upper = func_dop_add_point(lower, upper, positions[i_v, i_b])
        lower, upper = func_dop_add_point(
            lower, upper, func_vertex_sweep_end(i_v, i_b, positions, displacements, is_swept)
        )
        thickness = qd.max(thickness, vertices_thickness[i_v])
        bulge = qd.max(bulge, func_vertex_bulge(i_v, i_b, positions, pivots, rotations, is_screw))
    if thickness > 0.0:
        lower, upper = func_dop_expand(lower, upper, thickness)
    if bulge > 0.0:
        lower, upper = func_dop_expand(lower, upper, bulge)
    i_v0 = verts_idx[0]
    return func_dop_expand(lower, upper, vertices_d_hat[i_v0])


@qd.func(requires_top_level=True)
def func_ef_bounds_build(
    positions: qd.Tensor,
    displacements: qd.Tensor,
    pivots: qd.Tensor,
    rotations: qd.Tensor,
    vertices_thickness: qd.Tensor,
    vertices_d_hat: qd.Tensor,
    edges_verts_idx: qd.Tensor,
    triangles_verts_idx: qd.Tensor,
    bounds: EFBounds,
    groups: EFGroups,
    queries: DOPBVHQueries,
    tree: DOPBVHState,
    groups_queries: DOPBVHQueries,
    groups_tree: DOPBVHState,
    topology: EFTopology,
    tree_config: qd.template(),
    groups_tree_config: qd.template(),
    is_screw: qd.template(),
    is_swept: qd.template(),
):
    """
    Bound the vertices, edges, triangles and groups of every environment over the sweep, write the edge queries and the
    triangle proxies, and build the tree of the proxies and the tree of the groups of every environment.
    """
    n_envs = positions.shape[1]
    n_groups = groups.n_vertices.shape[0]
    for i_g, i_b in qd.ndrange(n_groups, n_envs):
        groups.lower[i_g, i_b] = qd.Vector([1e300] * 3, dt=gs.qd_float)
        groups.upper[i_g, i_b] = qd.Vector([-1e300] * 3, dt=gs.qd_float)
        groups.gap[i_g, i_b] = 0.0
        groups.n_vertices[i_g, i_b] = 0
        groups.n_triangles[i_g, i_b] = 0

    # The vertices, and the boxes of their groups. Few groups hold many vertices, so where the lanes vote, the 32 lanes
    # of a chunk of consecutive vertices of one environment fold each run of lanes sharing a group, and the tail lane of
    # the run alone updates the group, the contention of the atomics costing most of the bounds otherwise.
    CHUNK = qd.static(32 if tree_config.is_block_query else 1)
    n_vertices = positions.shape[0]
    n_flat = n_vertices * n_envs
    # Blocks of whole chunks, so that the lane of a vertex in its chunk is i_flat % CHUNK
    qd.loop_config(block_dim=128)
    for i_flat in range((n_flat + CHUNK - 1) // CHUNK * CHUNK):
        is_valid = i_flat < n_flat
        i_b = qd.min(i_flat, n_flat - 1) // n_vertices
        i_v = qd.min(i_flat, n_flat - 1) % n_vertices
        start = positions[i_v, i_b]
        end = func_vertex_sweep_end(i_v, i_b, positions, displacements, is_swept)
        lower, upper = func_dop_empty()
        lower, upper = func_dop_add_point(lower, upper, start)
        lower, upper = func_dop_add_point(lower, upper, end)
        gap = vertices_d_hat[i_v] + vertices_thickness[i_v]
        gap = gap + func_vertex_bulge(i_v, i_b, positions, pivots, rotations, is_screw)
        if is_valid:
            bounds.vertices_lower[i_v, i_b] = lower
            bounds.vertices_upper[i_v, i_b] = upper
            bounds.vertices_gap[i_v, i_b] = gap
        i_g = topology.vertices_group[i_v]
        box_lower = qd.min(start, end)
        box_upper = qd.max(start, end)
        n_run = 1
        is_tail = is_valid
        if qd.static(tree_config.is_block_query):
            tid = i_flat % CHUNK
            i_slot = qd.select(is_valid, i_b * n_groups + i_g, -1)
            i_slot_prev, i_slot_next = qd_slot_neighbors(tid, i_slot)
            for k in qd.static(range(3)):
                run_lower, is_tail = qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, box_lower[k])
                run_upper_neg, _ = qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, -box_upper[k])
                box_lower[k] = run_lower
                box_upper[k] = -run_upper_neg
            run_gap_neg, _ = qd_segmented_min(tid, i_slot, i_slot_prev, i_slot_next, -gap)
            gap = -run_gap_neg
            n_run, _ = qd_segmented_sum(tid, i_slot, i_slot_prev, i_slot_next, 1)
        if is_tail:
            qd.atomic_min(groups.lower[i_g, i_b], box_lower)
            qd.atomic_max(groups.upper[i_g, i_b], box_upper)
            qd.atomic_max(groups.gap[i_g, i_b], gap)
            qd.atomic_add(groups.n_vertices[i_g, i_b], n_run)

    # The edges and their queries, grown by the largest gap of the vertices they own
    for i_e, i_b in qd.ndrange(edges_verts_idx.shape[0], n_envs):
        edge = edges_verts_idx[i_e]
        lower, upper = func_primitive_bounds(
            i_b,
            edge,
            positions,
            displacements,
            pivots,
            rotations,
            vertices_thickness,
            vertices_d_hat,
            is_screw,
            is_swept,
        )
        bounds.edges_lower[i_e, i_b] = lower
        bounds.edges_upper[i_e, i_b] = upper
        i_v0 = edge[0]
        gap = vertices_d_hat[i_v0]
        owned = topology.edges_owned_verts[i_e]
        for k in qd.static(range(2)):
            i_v = owned[k]
            if i_v >= 0:
                gap = qd.max(gap, bounds.vertices_gap[i_v, i_b])
        queries.lower[i_e, i_b] = lower
        queries.upper[i_e, i_b] = upper
        queries.gap[i_e, i_b] = gap
        queries.tag[i_e, i_b] = qd.select(topology.edges_tag[i_e] == 1, 1, -1)

    # The triangles, their proxies, and their counts in the groups of their vertices, a triangle across groups adding
    # the box of the sweep of its vertices to each of them
    for i_t, i_b in qd.ndrange(triangles_verts_idx.shape[0], n_envs):
        tri = triangles_verts_idx[i_t]
        lower, upper = func_primitive_bounds(
            i_b,
            tri,
            positions,
            displacements,
            pivots,
            rotations,
            vertices_thickness,
            vertices_d_hat,
            is_screw,
            is_swept,
        )
        bounds.triangles_lower[i_t, i_b] = lower
        bounds.triangles_upper[i_t, i_b] = upper
        owned = topology.triangles_owned_edges[i_t]
        for k in qd.static(range(3)):
            i_e = owned[k]
            if i_e >= 0:
                lower, upper = func_dop_merge(lower, upper, bounds.edges_lower[i_e, i_b], bounds.edges_upper[i_e, i_b])
        tree.leaves.lower[i_t, i_b] = lower
        tree.leaves.upper[i_t, i_b] = upper
        tree.leaves.tag[i_t, i_b] = topology.triangles_tag[i_t]
        if topology.triangles_group[i_t] >= 0:
            i_g = topology.triangles_group[i_t]
            qd.atomic_add(groups.n_triangles[i_g, i_b], 1)
        else:
            box_lower = qd.Vector([1e300] * 3, dt=gs.qd_float)
            box_upper = qd.Vector([-1e300] * 3, dt=gs.qd_float)
            gap = gs.qd_float(0.0)
            for k in qd.static(range(3)):
                i_v = tri[k]
                end = func_vertex_sweep_end(i_v, i_b, positions, displacements, is_swept)
                box_lower = qd.min(box_lower, qd.min(positions[i_v, i_b], end))
                box_upper = qd.max(box_upper, qd.max(positions[i_v, i_b], end))
                gap = qd.max(gap, bounds.vertices_gap[i_v, i_b])
            for k in qd.static(range(3)):
                i_g = topology.vertices_group[tri[k]]
                i_v_prev_a = tri[0]
                i_v_prev_b = tri[1]
                is_first = True
                if qd.static(k >= 1):
                    is_first = i_g != topology.vertices_group[i_v_prev_a]
                if qd.static(k == 2):
                    is_first = is_first and i_g != topology.vertices_group[i_v_prev_b]
                if is_first:
                    qd.atomic_add(groups.n_triangles[i_g, i_b], 1)
                    qd.atomic_min(groups.lower[i_g, i_b], box_lower)
                    qd.atomic_max(groups.upper[i_g, i_b], box_upper)
                    qd.atomic_max(groups.gap[i_g, i_b], gap)

    # The groups as the leaves and the queries of their tree, grown by twice their gap, the gap of a query over the
    # growth of a leaf, plus an allowance for the rounding of the bounds, an empty group being a point at the origin
    for i_g, i_b in qd.ndrange(n_groups, n_envs):
        lower, upper = func_dop_empty()
        if groups.n_vertices[i_g, i_b] > 0:
            box_lower = groups.lower[i_g, i_b]
            box_upper = groups.upper[i_g, i_b]
            extent = qd.max(qd.abs(box_lower).max(), qd.abs(box_upper).max())
            for i_c in qd.static(range(8)):
                corner = box_lower
                for k in qd.static(range(3)):
                    if qd.static((i_c >> k) & 1):
                        corner[k] = box_upper[k]
                lower, upper = func_dop_add_point(lower, upper, corner)
            lower, upper = func_dop_expand(lower, upper, 2.0 * groups.gap[i_g, i_b] + 1e-6 * (1.0 + extent))
        else:
            lower, upper = func_dop_add_point(lower, upper, qd.Vector.zero(gs.qd_float, 3))
        groups_tree.leaves.lower[i_g, i_b] = lower
        groups_tree.leaves.upper[i_g, i_b] = upper
        groups_tree.leaves.tag[i_g, i_b] = -1
        groups_queries.lower[i_g, i_b] = lower
        groups_queries.upper[i_g, i_b] = upper
        groups_queries.gap[i_g, i_b] = 0.0
        groups_queries.tag[i_g, i_b] = -1
        groups_queries.active[i_b * n_groups + i_g] = i_b * n_groups + i_g
    for _ in range(1):
        groups_queries.n_active[()] = n_groups * n_envs

    func_build_dop_bvh(tree, tree_config)
    func_build_dop_bvh(groups_tree, groups_tree_config)


@qd.kernel(fastcache=True)
def kernel_ef_bounds_build(
    positions: qd.Tensor,
    displacements: qd.Tensor,
    pivots: qd.Tensor,
    rotations: qd.Tensor,
    vertices_thickness: qd.Tensor,
    vertices_d_hat: qd.Tensor,
    edges_verts_idx: qd.Tensor,
    triangles_verts_idx: qd.Tensor,
    bounds: EFBounds,
    groups: EFGroups,
    queries: DOPBVHQueries,
    tree: DOPBVHState,
    groups_queries: DOPBVHQueries,
    groups_tree: DOPBVHState,
    topology: EFTopology,
    tree_config: qd.template(),
    groups_tree_config: qd.template(),
    is_screw: qd.template(),
    is_swept: qd.template(),
):
    """Bound the surface over the sweep and build its trees (see func_ef_bounds_build)."""
    func_ef_bounds_build(
        positions,
        displacements,
        pivots,
        rotations,
        vertices_thickness,
        vertices_d_hat,
        edges_verts_idx,
        triangles_verts_idx,
        bounds,
        groups,
        queries,
        tree,
        groups_queries,
        groups_tree,
        topology,
        tree_config,
        groups_tree_config,
        is_screw,
        is_swept,
    )


# ------------------------------------------------------------------------------------
# ------------------------------------- query ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_is_pt_leaf_kept(
    i_b: int, i_v: int, i_t: int, triangles_verts_idx: qd.Tensor, bounds: EFBounds, topology: EFTopology
) -> bool:
    """Return whether a vertex and a triangle form a PT sub-pair: the bound of the vertex within its gap of the bound of
    the triangle, the vertex off the triangle, the two not both rigid."""
    tri = triangles_verts_idx[i_t]
    is_kept = i_v != tri[0] and i_v != tri[1] and i_v != tri[2]
    if is_kept:
        is_kept = not (topology.triangles_tag[i_t] == 1 and topology.vertices_tag[i_v] == 1)
    if is_kept:
        is_kept = func_dop_overlap(
            bounds.vertices_lower[i_v, i_b],
            bounds.vertices_upper[i_v, i_b],
            bounds.triangles_lower[i_t, i_b],
            bounds.triangles_upper[i_t, i_b],
            func_dop_gaps(bounds.vertices_gap[i_v, i_b]),
        )
    return is_kept


@qd.func
def func_is_ee_leaf_kept(
    i_b: int,
    i_e_lo: int,
    i_e_hi: int,
    vertices_d_hat: qd.Tensor,
    edges_verts_idx: qd.Tensor,
    bounds: EFBounds,
    topology: EFTopology,
) -> bool:
    """Return whether two edges form an EE sub-pair: their bounds within the activation distance of each other, no shared
    vertex, the two not both rigid."""
    edge_lo = edges_verts_idx[i_e_lo]
    edge_hi = edges_verts_idx[i_e_hi]
    is_kept = True
    for i, j in qd.static(qd.ndrange(2, 2)):
        if edge_lo[i] == edge_hi[j]:
            is_kept = False
    if is_kept:
        is_kept = not (topology.edges_tag[i_e_lo] == 1 and topology.edges_tag[i_e_hi] == 1)
    if is_kept:
        i_v_lo = edge_lo[0]
        i_v_hi = edge_hi[0]
        d_hat = qd.max(vertices_d_hat[i_v_lo], vertices_d_hat[i_v_hi])
        is_kept = func_dop_overlap(
            bounds.edges_lower[i_e_lo, i_b],
            bounds.edges_upper[i_e_lo, i_b],
            bounds.edges_lower[i_e_hi, i_b],
            bounds.edges_upper[i_e_hi, i_b],
            func_dop_gaps(d_hat),
        )
    return is_kept


@qd.func(requires_top_level=True)
def func_ef_query(
    vertices_d_hat: qd.Tensor,
    edges_verts_idx: qd.Tensor,
    triangles_verts_idx: qd.Tensor,
    bounds: EFBounds,
    groups: EFGroups,
    query_list: EFQueryList,
    queries: DOPBVHQueries,
    tree: DOPBVHState,
    hits: DOPBVHHits,
    groups_queries: DOPBVHQueries,
    groups_tree: DOPBVHState,
    groups_hits: DOPBVHHits,
    sub_pairs: EFSubPairs,
    topology: EFTopology,
    tree_config: qd.template(),
    groups_tree_config: qd.template(),
    is_expanded: qd.template(),
):
    """
    Find the groups whose edges query, run their edge queries against the tree of the proxies, and with `is_expanded`,
    expand the hits into PT and EE sub-pairs (see the module docstring). Without it the hits stay as they are, edges
    against proxies.
    """
    n_envs = groups.n_vertices.shape[1]
    n_groups = groups.n_vertices.shape[0]
    n_edges = edges_verts_idx.shape[0]

    # A group queries when its box meets that of a group owning triangles that it may pair with, itself included
    func_query_dop_bvh(groups_queries, groups_tree.tree, groups_hits, groups_tree_config)
    for i_g, i_b in qd.ndrange(n_groups, n_envs):
        groups.is_active[i_g, i_b] = False
    for i_h in range(qd.min(groups_hits.count[()], groups_hits.hits.shape[0])):
        hit = groups_hits.hits[i_h]
        i_b, i_g, i_g_other = hit[0], hit[1], hit[2]
        is_pair = groups.n_vertices[i_g, i_b] > 0 and groups.n_triangles[i_g_other, i_b] > 0
        if is_pair:
            is_pair = not (topology.groups_is_rigid[i_g] and topology.groups_is_rigid[i_g_other])
        if is_pair:
            groups.is_active[i_g, i_b] = True

    # The list of the edge queries that run: those of the active groups, and the edges across groups
    for i_e, i_b in qd.ndrange(n_edges, n_envs):
        i_g = topology.edges_group[i_e]
        is_running = i_g < 0
        if not is_running:
            is_running = groups.is_active[i_g, i_b]
        query_list.flags[i_b * n_edges + i_e] = qd.select(is_running, 1, 0)
    for _ in range(1):
        query_list.n_flags[()] = n_edges * n_envs
    func_exclusive_sum(query_list.scanner, query_list.flags, query_list.offsets, query_list.n_flags)
    for i_flat in range(n_edges * n_envs):
        if query_list.flags[i_flat] == 1:
            queries.active[query_list.offsets[i_flat]] = i_flat
        if i_flat == n_edges * n_envs - 1:
            queries.n_active[()] = query_list.offsets[i_flat] + query_list.flags[i_flat]

    func_query_dop_bvh(queries, tree.tree, hits, tree_config)

    if qd.static(is_expanded):
        for _ in range(1):
            sub_pairs.counts[0] = 0
            sub_pairs.counts[1] = 0
        for i_h in range(qd.min(hits.count[()], hits.hits.shape[0])):
            hit = hits.hits[i_h]
            i_b, i_e, i_t = hit[0], hit[1], hit[2]
            owned_verts = topology.edges_owned_verts[i_e]
            for k in qd.static(range(2)):
                i_v = owned_verts[k]
                if i_v >= 0:
                    if func_is_pt_leaf_kept(i_b, i_v, i_t, triangles_verts_idx, bounds, topology):
                        i_p = qd.atomic_add(sub_pairs.counts[0], 1)
                        if i_p < sub_pairs.pt.shape[0]:
                            sub_pairs.pt[i_p] = qd.Vector([i_b, i_v, i_t], dt=gs.qd_int)
            owned_edges = topology.triangles_owned_edges[i_t]
            for k in qd.static(range(3)):
                i_e_other = owned_edges[k]
                if i_e_other >= 0 and i_e_other < i_e:
                    if func_is_ee_leaf_kept(i_b, i_e_other, i_e, vertices_d_hat, edges_verts_idx, bounds, topology):
                        i_p = qd.atomic_add(sub_pairs.counts[1], 1)
                        if i_p < sub_pairs.ee.shape[0]:
                            sub_pairs.ee[i_p] = qd.Vector([i_b, i_e_other, i_e], dt=gs.qd_int)


@qd.kernel(fastcache=True)
def kernel_ef_query(
    vertices_d_hat: qd.Tensor,
    edges_verts_idx: qd.Tensor,
    triangles_verts_idx: qd.Tensor,
    bounds: EFBounds,
    groups: EFGroups,
    query_list: EFQueryList,
    queries: DOPBVHQueries,
    tree: DOPBVHState,
    hits: DOPBVHHits,
    groups_queries: DOPBVHQueries,
    groups_tree: DOPBVHState,
    groups_hits: DOPBVHHits,
    sub_pairs: EFSubPairs,
    topology: EFTopology,
    tree_config: qd.template(),
    groups_tree_config: qd.template(),
    is_expanded: qd.template(),
):
    """Run the edge queries of the surface (see func_ef_query)."""
    func_ef_query(
        vertices_d_hat,
        edges_verts_idx,
        triangles_verts_idx,
        bounds,
        groups,
        query_list,
        queries,
        tree,
        hits,
        groups_queries,
        groups_tree,
        groups_hits,
        sub_pairs,
        topology,
        tree_config,
        groups_tree_config,
        is_expanded,
    )


class EFBroadPhase:
    """
    The edge-face broad phase of a surface over the environments of a scene (see the module docstring): its topology,
    bounds, trees, query lists and the hit and sub-pair lists, grown on overflow.

    The surface is made of the edges `edges_verts_idx` and the triangles `triangles_verts_idx`, its vertices have the
    thicknesses `vertices_thickness` and the activation distances `vertices_d_hat`, fall into the groups `vertices_group`
    and are rigid where `vertices_is_rigid` holds. `run` bounds the surface and queries it. The caller filters the
    sub-pairs, then reads back `sub_pairs.summary` and hands it to `grow`, which replaces the lists that overflowed, frees
    the old ones through `retire`, and tells whether to run again.
    """

    def __init__(
        self,
        n_envs: int,
        edges_verts_idx: qd.Tensor,
        triangles_verts_idx: qd.Tensor,
        vertices_thickness: qd.Tensor,
        vertices_d_hat: qd.Tensor,
        vertices_group: np.ndarray,
        vertices_is_rigid: qd.Tensor,
        pair_capacity: int,
        retire: Callable[[Data], None],
    ) -> None:
        self._edges_verts_idx = edges_verts_idx
        self._triangles_verts_idx = triangles_verts_idx
        self._vertices_thickness = vertices_thickness
        self._vertices_d_hat = vertices_d_hat
        self._retire = retire
        self.topology, n_groups = get_ef_topology(
            qd_to_numpy(edges_verts_idx),
            qd_to_numpy(triangles_verts_idx),
            vertices_group,
            qd_to_numpy(vertices_is_rigid),
        )
        n_vertices, n_edges, n_triangles = len(vertices_group), edges_verts_idx.shape[0], triangles_verts_idx.shape[0]
        self.bounds = EFBounds(
            sizes={EF_VERTICES: n_vertices, EF_EDGES: n_edges, EF_TRIANGLES: n_triangles, ENVS: n_envs}
        )
        self.groups = EFGroups(sizes={EF_GROUPS: n_groups, ENVS: n_envs})
        self.query_list = EFQueryList(
            sizes={EF_EDGES: n_edges, ENVS: n_envs, EFQueryList.scanner: exclusive_sum_sizes(n_envs * n_edges)}
        )
        self.tree, self.tree_config = get_dop_bvh_data(n_envs, n_triangles)
        self.groups_tree, self.groups_tree_config = get_dop_bvh_data(n_envs, n_groups)
        # The queries on one thread each, for the rerun of a search whose queries on the lanes of a block stopped short
        self._query_configs_thread = (
            get_dop_bvh_config(n_envs, n_triangles, is_block_query=False),
            get_dop_bvh_config(n_envs, n_groups, is_block_query=False),
        )
        self._is_thread_query = False
        self.queries = DOPBVHQueries(sizes=dict(dop_queries=n_edges, dop_trees=n_envs))
        self.groups_queries = DOPBVHQueries(sizes=dict(dop_queries=n_groups, dop_trees=n_envs))
        # Every group meets itself
        self.groups_hits = DOPBVHHits(sizes=dict(dop_hits=n_envs * n_groups))
        # The hits and sub-pairs include the pairs the tests downstream reject, so they start above the pairs
        self.hits = DOPBVHHits(sizes=dict(dop_hits=4 * pair_capacity))
        self.sub_pairs = EFSubPairs(sizes={EF_SUB_PT: 2 * pair_capacity, EF_SUB_EE: 2 * pair_capacity})

    def run(
        self,
        positions: qd.Tensor,
        displacements: qd.Tensor,
        pivots: qd.Tensor,
        rotations: qd.Tensor,
        is_screw: bool,
        is_swept: bool,
        is_expanded: bool,
    ) -> None:
        """
        Bound the surface over the sweep and run the edge queries, expanding their hits into sub-pairs with
        `is_expanded` (see func_ef_query).

        `positions` and `displacements` are the positions of the vertices of every environment and their sweep, which
        ends at the positions without `is_swept`. With `is_screw` the vertices sweep along screws of origins `pivots` and
        rotation vectors `rotations`. The queries run on one thread each after a run whose queries stopped short (see
        `grow`).
        """
        tree_config, groups_tree_config = self.tree_config, self.groups_tree_config
        if self._is_thread_query:
            tree_config, groups_tree_config = self._query_configs_thread
            self._is_thread_query = False
        kernel_ef_bounds_build(
            positions,
            displacements,
            pivots,
            rotations,
            self._vertices_thickness,
            self._vertices_d_hat,
            self._edges_verts_idx,
            self._triangles_verts_idx,
            self.bounds,
            self.groups,
            self.queries,
            self.tree,
            self.groups_queries,
            self.groups_tree,
            self.topology,
            self.tree_config,
            self.groups_tree_config,
            is_screw,
            is_swept,
        )
        kernel_ef_query(
            self._vertices_d_hat,
            self._edges_verts_idx,
            self._triangles_verts_idx,
            self.bounds,
            self.groups,
            self.query_list,
            self.queries,
            self.tree,
            self.hits,
            self.groups_queries,
            self.groups_tree,
            self.groups_hits,
            self.sub_pairs,
            self.topology,
            tree_config,
            groups_tree_config,
            is_expanded,
        )

    def grow(self, n_groups_hits: int, n_hits: int, n_sub_pt: int, n_sub_ee: int, n_truncated: int) -> bool:
        """
        Grow the lists whose counts overflowed them, and return whether one did or a query stopped short.

        The counts are those of `sub_pairs.summary`, the hits of the groups and of the edge queries, then the PT and EE
        sub-pairs, then the queries that stopped short (see DOPBVHHits). A list grows to the larger of its count and
        `EF_GROW_FACTOR` times its capacity. After queries stopped short, the next run queries on one thread each.
        """
        self._is_thread_query = n_truncated > 0
        counts = (n_groups_hits, n_hits, n_sub_pt, n_sub_ee)
        capacities = (
            self.groups_hits.hits.shape[0],
            self.hits.hits.shape[0],
            self.sub_pairs.pt.shape[0],
            self.sub_pairs.ee.shape[0],
        )
        capacities_grown = tuple(
            max(count, math.ceil(capacity * EF_GROW_FACTOR)) if count > capacity else capacity
            for count, capacity in zip(counts, capacities)
        )
        if capacities_grown[0] > capacities[0]:
            self._retire(self.groups_hits)
            self.groups_hits = DOPBVHHits(sizes=dict(dop_hits=capacities_grown[0]))
        if capacities_grown[1] > capacities[1]:
            self._retire(self.hits)
            self.hits = DOPBVHHits(sizes=dict(dop_hits=capacities_grown[1]))
        if capacities_grown[2:] != capacities[2:]:
            self._retire(self.sub_pairs)
            self.sub_pairs = EFSubPairs(sizes={EF_SUB_PT: capacities_grown[2], EF_SUB_EE: capacities_grown[3]})
        return capacities_grown != capacities or self._is_thread_query
