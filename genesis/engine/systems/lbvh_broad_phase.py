from __future__ import annotations

import quadrants as qd

from .broad_phase_system import BroadPhaseSystem
from .dual_ee_query import (
    DualEEQueryState,
    get_dual_ee_query_data,
    handle_overflow as handle_dual_ee_overflow,
    query as dual_ee_query,
)
from .lbvh import (
    LBVH,
    build_edge,
    build_tri,
    get_lbvh_data,
    query_ee_warp,
    query_et,
    query_pt_batched,
    query_pt_warp,
)
from .sim_system import SimData


class LBVHBroadPhase(BroadPhaseSystem):
    """Baseline LBVH broad phase with explicit mutable Data."""

    @qd.data_oriented
    class Data(SimData):
        bound_type: str
        use_warp_pt: bool
        use_dual_ee: bool
        genesis_legacy_sort_reduce: bool
        genesis_legacy_fp64_bounds: bool
        genesis_legacy_refit: bool
        has_triangle_bvh: bool
        has_edge_bvh: bool
        has_codim_point_bvh: bool
        triangle_bvh: LBVH.Data | None
        edge_bvh: LBVH.Data | None
        ee_dual_state: DualEEQueryState.Data | None

    def __init__(self, data: Data) -> None:
        super().__init__()
        self.data = data
        self.actions: dict[str, object] = {}

    def build(self) -> None:
        super().build()
        data = self.data
        surface = self.surface_system.data
        vertex = self.vertex_system.data
        body = self.body_system.data
        contact = self.contact_system.data
        protocol = {
            "triangle_build": (triangle_build, (data, surface, vertex)),
            "edge_build": (edge_build, (data, surface, vertex)),
            "pt_query": (pt_query, (data, surface, vertex, body, contact)),
            "ee_query": (ee_query, (data, surface, vertex, body, contact)),
            "trajectory_query": (trajectory_query, (data, surface, vertex, body, contact)),
            "detect_initial_intersections": (
                detect_initial_intersections,
                (data, surface, vertex, body, contact),
            ),
        }
        self.actions = {
            name: self.create_action(kernel, *action_data) for name, (kernel, action_data) in protocol.items()
        }

    def resolve_actions(self) -> dict[str, object]:
        if self.is_building:
            raise RuntimeError("Broad-phase actions are available only after build")
        return {name: action.invocation for name, action in self.actions.items()}

    def handle_ee_query_overflow(self) -> bool:
        data = self.data
        if not data.has_edge_bvh:
            return False
        if data.use_dual_ee:
            return handle_dual_ee_overflow(data.ee_dual_state)
        if int(data.edge_bvh.ee_warp_stack_overflow.to_numpy()):
            raise RuntimeError("warp EE shared frontier stack exhausted; increase ee_warp_stack_capacity")
        return False


class InfoLBVHBatchedBroadPhaseDop14(LBVHBroadPhase):
    """Default DOP14 broad phase with homogeneous-body culling and dual EE traversal."""


def get_lbvh_broad_phase_data(
    *,
    n_triangles: int,
    n_edges: int,
    n_surface_vertices: int,
    n_codim_verts: int = 0,
    bound_type: str = "aabb",
    pt_query: str = "warp",
    ee_query: str = "dual",
    dual_frontier_levels: int = 0,
    dual_target_waves: float = 24.0,
    dual_max_levels: int = 18,
    genesis_legacy_sort_reduce: bool = False,
    genesis_legacy_fp64_bounds: bool = False,
    genesis_legacy_refit: bool = False,
) -> LBVHBroadPhase.Data:
    """Construct all LBVH-owned Data before broad-phase action registration."""
    if pt_query not in ("warp", "batched"):
        raise ValueError(f"Unsupported bvh/pt_query {pt_query!r}")
    if ee_query not in ("dual", "warp"):
        raise ValueError(f"Unsupported bvh/ee_query {ee_query!r}")
    if n_codim_verts > 0:
        raise NotImplementedError("Explicit codimensional PE/PP broad phase is outside the cloth milestone")

    data = LBVHBroadPhase.Data()
    data.bound_type = bound_type
    data.use_warp_pt = pt_query == "warp"
    data.use_dual_ee = ee_query == "dual"
    data.genesis_legacy_sort_reduce = bool(genesis_legacy_sort_reduce)
    data.genesis_legacy_fp64_bounds = bool(genesis_legacy_fp64_bounds)
    data.genesis_legacy_refit = bool(genesis_legacy_refit)
    data.has_triangle_bvh = n_triangles > 0
    data.has_edge_bvh = n_edges > 0
    data.has_codim_point_bvh = False
    data.triangle_bvh = (
        get_lbvh_data(
            n_triangles,
            max(n_surface_vertices, n_edges, 1),
            bound_type,
            genesis_legacy_sort_reduce,
            genesis_legacy_fp64_bounds,
            genesis_legacy_refit,
        )
        if data.has_triangle_bvh
        else None
    )
    data.edge_bvh = (
        get_lbvh_data(
            n_edges,
            n_edges,
            bound_type,
            genesis_legacy_sort_reduce,
            genesis_legacy_fp64_bounds,
            genesis_legacy_refit,
        )
        if data.has_edge_bvh
        else None
    )
    data.ee_dual_state = (
        get_dual_ee_query_data(
            n_edges,
            dual_frontier_levels,
            dual_target_waves,
            dual_max_levels,
        )
        if data.has_edge_bvh and data.use_dual_ee
        else None
    )
    return data


def get_info_lbvh_batched_broad_phase_dop14_data(**kwargs) -> LBVHBroadPhase.Data:
    return get_lbvh_broad_phase_data(bound_type="dop14", **kwargs)


@qd.func(requires_top_level=True)
def triangle_build(data: qd.template(), surface: qd.template(), vertex: qd.template()):
    if qd.static(data.has_triangle_bvh):
        build_tri(data.triangle_bvh, surface, vertex)


@qd.func(requires_top_level=True)
def edge_build(data: qd.template(), surface: qd.template(), vertex: qd.template()):
    if qd.static(data.has_edge_bvh):
        build_edge(data.edge_bvh, surface, vertex)


@qd.func(requires_top_level=True)
def pt_query(
    data: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
):
    if qd.static(data.has_triangle_bvh):
        if qd.static(data.use_warp_pt):
            query_pt_warp(
                data.triangle_bvh,
                surface,
                vertex,
                body,
                contact,
                contact.pairs_pt,
                contact.n_pairs_pt,
                contact.max_pairs_pt[()],
                contact.d_hat[()],
                contact.overflow_flag,
            )
        else:
            query_pt_batched(
                data.triangle_bvh,
                surface,
                vertex,
                body,
                contact,
                contact.pairs_pt,
                contact.n_pairs_pt,
                contact.max_pairs_pt[()],
                contact.d_hat[()],
                contact.overflow_flag,
            )


@qd.func(requires_top_level=True)
def ee_query(
    data: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
):
    if qd.static(data.has_edge_bvh):
        if qd.static(data.use_dual_ee):
            dual_ee_query(
                data.ee_dual_state,
                data.edge_bvh,
                surface,
                vertex,
                body,
                contact,
                contact.pairs_ee,
                contact.n_pairs_ee,
                contact.max_pairs_ee[()],
                contact.overflow_flag,
            )
        else:
            query_ee_warp(
                data.edge_bvh,
                surface,
                vertex,
                body,
                contact,
                contact.pairs_ee,
                contact.n_pairs_ee,
                contact.max_pairs_ee[()],
                qd.f64(0.0),
                contact.overflow_flag,
            )


@qd.func(requires_top_level=True)
def trajectory_query(
    data: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
):
    pt_query(data, surface, vertex, body, contact)
    ee_query(data, surface, vertex, body, contact)


@qd.func(requires_top_level=True)
def detect_initial_intersections(
    data: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
):
    if qd.static(data.has_triangle_bvh):
        query_et(
            data.triangle_bvh,
            surface,
            vertex,
            body,
            contact,
            contact.et_pairs,
            contact.n_et_pairs,
            contact.max_et_pairs[()],
            contact.et_overflow_flag,
        )
