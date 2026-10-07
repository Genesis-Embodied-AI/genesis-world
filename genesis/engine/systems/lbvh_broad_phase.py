from __future__ import annotations

import quadrants as qd

from .sim_system import SimData, SimSystem
from .dual_ee_query import (
    DualEEQueryState,
    handle_overflow as handle_dual_ee_overflow,
    initialize_dual_ee_query_data,
    query as dual_ee_query,
)
from .lbvh import (
    LBVH,
    build_edge,
    build_tri,
    initialize_lbvh_data,
    query_ee_warp,
    query_et,
    query_pt_batched,
    query_pt_warp,
)


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class LBVHBroadPhase(SimSystem):
    """Baseline LBVH broad phase with explicit mutable Data."""

    @qd.data_oriented
    class Data(SimData):
        triangle_bvh: LBVH.Data | None
        edge_bvh: LBVH.Data | None
        ee_dual_state: DualEEQueryState.Data | None

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._wire_args = None
        self.bound_type = "aabb"
        self.use_warp_pt = True
        self.use_dual_ee = True
        self.has_triangle_bvh = False
        self.has_edge_bvh = False
        self.genesis_legacy_sort_reduce = False
        self.genesis_legacy_fp64_bounds = False
        self.genesis_legacy_refit = False

    def wire_data(self, **kwargs) -> None:
        self._wire_args = dict(kwargs)
        self.bound_type = str(kwargs["bound_type"])
        self.use_warp_pt = kwargs["pt_query"] == "warp"
        self.use_dual_ee = kwargs["ee_query"] == "dual"
        self.has_triangle_bvh = kwargs["n_triangles"] > 0
        self.has_edge_bvh = kwargs["n_edges"] > 0
        self.genesis_legacy_sort_reduce = bool(kwargs["genesis_legacy_sort_reduce"])
        self.genesis_legacy_fp64_bounds = bool(kwargs["genesis_legacy_fp64_bounds"])
        self.genesis_legacy_refit = bool(kwargs["genesis_legacy_refit"])

    def build(self) -> None:
        from .contact_system import ContactSystem
        from .global_body_manager import GlobalBodyManager
        from .global_surface_manager import GlobalSurfaceManager
        from .global_vertex_manager import GlobalVertexManager

        self.contact_system = self.require(ContactSystem)
        self.body_system = self.require(GlobalBodyManager)
        self.surface_system = self.require(GlobalSurfaceManager)
        self.vertex_system = self.require(GlobalVertexManager)
        self.init_action = self.create_action(self.init)
        self.contact_system.on_broad_phase(self.init_action)
        data = self.data
        surface = self.surface_system.data
        vertex = self.vertex_system.data
        body = self.body_system.data
        contact = self.contact_system.data
        self.triangle_build_action = self.create_action(triangle_build, self, surface, vertex)
        self.edge_build_action = self.create_action(edge_build, self, surface, vertex)
        self.pt_query_action = self.create_action(pt_query, self, surface, vertex, body, contact)
        self.ee_query_action = self.create_action(ee_query, self, surface, vertex, body, contact)
        self.trajectory_query_action = self.create_action(trajectory_query, self, surface, vertex, body, contact)
        self.detect_initial_intersections_action = self.create_action(
            detect_initial_intersections,
            self,
            surface,
            vertex,
            body,
            contact,
        )

    @qd.func(requires_top_level=True)
    def on_build_triangles(self):
        triangle_build(
            self,
            self.surface_system.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_build_edges(self):
        edge_build(
            self,
            self.surface_system.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_query_pt(self):
        pt_query(
            self,
            self.surface_system.data,
            self.vertex_system.data,
            self.body_system.data,
            self.contact_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_query_ee(self):
        ee_query(
            self,
            self.surface_system.data,
            self.vertex_system.data,
            self.body_system.data,
            self.contact_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_query_trajectory(self):
        trajectory_query(
            self,
            self.surface_system.data,
            self.vertex_system.data,
            self.body_system.data,
            self.contact_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_detect_initial_intersections(self):
        detect_initial_intersections(
            self,
            self.surface_system.data,
            self.vertex_system.data,
            self.body_system.data,
            self.contact_system.data,
        )

    def handle_ee_query_overflow(self) -> bool:
        data = self.data
        if not self.has_edge_bvh:
            return False
        if self.use_dual_ee:
            return handle_dual_ee_overflow(data.ee_dual_state)
        if int(data.edge_bvh.ee_warp_stack_overflow.to_numpy()):
            raise RuntimeError("warp EE shared frontier stack exhausted; increase ee_warp_stack_capacity")
        return False

    def init(self) -> None:
        if self._wire_args is None:
            raise RuntimeError("LBVHBroadPhase data has not been wired")
        args = self._wire_args
        pt_query = args.pop("pt_query")
        ee_query = args.pop("ee_query")
        bound_type = args.pop("bound_type")
        n_triangles = args.pop("n_triangles")
        n_edges = args.pop("n_edges")
        n_surface_vertices = args.pop("n_surface_vertices")
        n_codim_verts = args.pop("n_codim_verts", 0)
        if pt_query not in ("warp", "batched"):
            raise ValueError(f"Unsupported bvh/pt_query {pt_query!r}")
        if ee_query not in ("dual", "warp"):
            raise ValueError(f"Unsupported bvh/ee_query {ee_query!r}")
        if n_codim_verts > 0:
            raise NotImplementedError("Explicit codimensional PE/PP broad phase is outside the cloth milestone")
        self.bound_type = bound_type
        self.use_warp_pt = pt_query == "warp"
        self.use_dual_ee = ee_query == "dual"
        self.has_triangle_bvh = n_triangles > 0
        self.has_edge_bvh = n_edges > 0
        legacy_sort = bool(args.pop("genesis_legacy_sort_reduce"))
        legacy_fp64 = bool(args.pop("genesis_legacy_fp64_bounds"))
        legacy_refit = bool(args.pop("genesis_legacy_refit"))
        dual_frontier_levels = args.pop("dual_frontier_levels")
        dual_target_waves = args.pop("dual_target_waves")
        dual_max_levels = args.pop("dual_max_levels")
        self.data.triangle_bvh = LBVH.Data() if self.has_triangle_bvh else None
        if self.has_triangle_bvh:
            initialize_lbvh_data(
                self.data.triangle_bvh,
                n_triangles,
                max(n_surface_vertices, n_edges, 1),
                bound_type,
                legacy_sort,
                legacy_fp64,
                legacy_refit,
            )
        self.data.edge_bvh = LBVH.Data() if self.has_edge_bvh else None
        if self.has_edge_bvh:
            initialize_lbvh_data(
                self.data.edge_bvh,
                n_edges,
                n_edges,
                bound_type,
                legacy_sort,
                legacy_fp64,
                legacy_refit,
            )
        self.data.ee_dual_state = DualEEQueryState.Data() if self.has_edge_bvh and self.use_dual_ee else None
        if self.data.ee_dual_state is not None:
            initialize_dual_ee_query_data(
                self.data.ee_dual_state,
                n_edges,
                dual_frontier_levels,
                dual_target_waves,
                dual_max_levels,
            )
        self._wire_args = None


@qd.func(requires_top_level=True)
def triangle_build(system: qd.template(), surface: qd.template(), vertex: qd.template()):
    if qd.static(system.has_triangle_bvh):
        build_tri(system.data.triangle_bvh, surface, vertex)


@qd.func(requires_top_level=True)
def edge_build(system: qd.template(), surface: qd.template(), vertex: qd.template()):
    if qd.static(system.has_edge_bvh):
        build_edge(system.data.edge_bvh, surface, vertex)


@qd.func(requires_top_level=True)
def pt_query(
    system: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
):
    if qd.static(system.has_triangle_bvh):
        if qd.static(system.use_warp_pt):
            query_pt_warp(
                system.data.triangle_bvh,
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
                system.data.triangle_bvh,
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
    system: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
):
    if qd.static(system.has_edge_bvh):
        if qd.static(system.use_dual_ee):
            dual_ee_query(
                system.data.ee_dual_state,
                system.data.edge_bvh,
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
                system.data.edge_bvh,
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
    system: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
):
    pt_query(system, surface, vertex, body, contact)
    ee_query(system, surface, vertex, body, contact)


@qd.func(requires_top_level=True)
def detect_initial_intersections(
    system: qd.template(),
    surface: qd.template(),
    vertex: qd.template(),
    body: qd.template(),
    contact: qd.template(),
):
    if qd.static(system.has_triangle_bvh):
        query_et(
            system.data.triangle_bvh,
            surface,
            vertex,
            body,
            contact,
            contact.et_pairs,
            contact.n_et_pairs,
            contact.max_et_pairs[()],
            contact.et_overflow_flag,
        )
