from __future__ import annotations

import quadrants as qd

from .broad_phase_system import BroadPhaseSystem
from .dual_ee_query import DualEEQueryState
from .lbvh import LBVH


@qd.data_oriented
class LBVHBroadPhase(BroadPhaseSystem):
    """CGQ baseline fp64-AABB LBVH broad phase."""

    def __init__(
        self,
        bound_type: str = "aabb",
        pt_query: str = "warp",
        ee_query: str = "dual",
        dual_frontier_levels: int = 0,
        dual_target_waves: float = 24.0,
        dual_max_levels: int = 18,
        genesis_legacy_sort_reduce: bool = False,
        genesis_legacy_fp64_bounds: bool = False,
        genesis_legacy_refit: bool = False,
    ) -> None:
        super().__init__()
        if pt_query not in ("warp", "batched"):
            raise ValueError(f"Unsupported bvh/pt_query {pt_query!r}")
        if ee_query not in ("dual", "warp"):
            raise ValueError(f"Unsupported bvh/ee_query {ee_query!r}")
        self.bound_type = bound_type
        self.use_warp_pt = pt_query == "warp"
        self.use_dual_ee = ee_query == "dual"
        self.dual_frontier_levels = dual_frontier_levels
        self.dual_target_waves = dual_target_waves
        self.dual_max_levels = dual_max_levels
        self.genesis_legacy_sort_reduce = bool(genesis_legacy_sort_reduce)
        self.genesis_legacy_fp64_bounds = bool(genesis_legacy_fp64_bounds)
        self.genesis_legacy_refit = bool(genesis_legacy_refit)
        self.has_triangle_bvh = False
        self.has_edge_bvh = False
        self.has_codim_point_bvh = False

    def init_bvh(self, n_triangles: int, n_edges: int, n_codim_verts: int) -> None:
        if n_triangles > 0:
            self.triangle_bvh = LBVH(
                n_triangles,
                max(
                    self.surface.surf_verts.shape[0],
                    self.surface.surf_edges.shape[0],
                    1,
                ),
                self.bound_type,
                self.genesis_legacy_sort_reduce,
                self.genesis_legacy_fp64_bounds,
                self.genesis_legacy_refit,
            )
            self.has_triangle_bvh = True
        if n_edges > 0:
            self.edge_bvh = LBVH(
                n_edges,
                n_edges,
                self.bound_type,
                self.genesis_legacy_sort_reduce,
                self.genesis_legacy_fp64_bounds,
                self.genesis_legacy_refit,
            )
            if self.use_dual_ee:
                self.ee_dual_state = DualEEQueryState(
                    n_edges,
                    self.dual_frontier_levels,
                    self.dual_target_waves,
                    self.dual_max_levels,
                )
            self.has_edge_bvh = True
        self.has_codim_point_bvh = n_codim_verts > 0
        if self.has_codim_point_bvh:
            raise NotImplementedError("Explicit codimensional PE/PP broad phase is outside the cloth milestone")

    @qd.func(requires_top_level=True)
    def triangle_build(self):
        if qd.static(self.has_triangle_bvh):
            self.triangle_bvh.calc_leaf_aabb_tri(self.surface, self.vertex)
            self.triangle_bvh.reduce_scene_aabb()
            self.triangle_bvh.calc_morton()
            self.triangle_bvh.sort_morton()
            self.triangle_bvh.extract_indices()
            self.triangle_bvh.copy_leaf_aabb_to_temp()
            self.triangle_bvh.reorder_leaf_aabb()
            self.triangle_bvh.calc_leaf_nodes()
            self.triangle_bvh.calc_internal_nodes()
            self.triangle_bvh.memset_flags()
            self.triangle_bvh.calc_internal_aabb()

    @qd.func(requires_top_level=True)
    def edge_build(self):
        if qd.static(self.has_edge_bvh):
            self.edge_bvh.calc_leaf_aabb_edge(self.surface, self.vertex)
            self.edge_bvh.reduce_scene_aabb()
            self.edge_bvh.calc_morton()
            self.edge_bvh.sort_morton()
            self.edge_bvh.extract_indices()
            self.edge_bvh.copy_leaf_aabb_to_temp()
            self.edge_bvh.reorder_leaf_aabb()
            self.edge_bvh.calc_leaf_nodes()
            self.edge_bvh.calc_internal_nodes()
            self.edge_bvh.memset_flags()
            self.edge_bvh.calc_internal_aabb()

    @qd.func(requires_top_level=True)
    def pt_query(self):
        if qd.static(self.has_triangle_bvh):
            if qd.static(self.use_warp_pt):
                self.triangle_bvh.query_pt_warp(
                    self.surface,
                    self.vertex,
                    self.body,
                    self.contact,
                    self.contact.pairs_pt,
                    self.contact.n_pairs_pt,
                    self.contact.max_pairs_pt[()],
                    self.contact.d_hat[()],
                    self.contact.overflow_flag,
                )
            else:
                self.triangle_bvh.query_pt_batched(
                    self.surface,
                    self.vertex,
                    self.body,
                    self.contact,
                    self.contact.pairs_pt,
                    self.contact.n_pairs_pt,
                    self.contact.max_pairs_pt[()],
                    self.contact.d_hat[()],
                    self.contact.overflow_flag,
                )

    @qd.func(requires_top_level=True)
    def ee_query(self):
        if qd.static(self.has_edge_bvh):
            if qd.static(self.use_dual_ee):
                self.ee_dual_state.query(
                    self.edge_bvh,
                    self.surface,
                    self.vertex,
                    self.body,
                    self.contact,
                    self.contact.pairs_ee,
                    self.contact.n_pairs_ee,
                    self.contact.max_pairs_ee[()],
                    self.contact.overflow_flag,
                )
            else:
                self.edge_bvh.query_ee_warp(
                    self.surface,
                    self.vertex,
                    self.body,
                    self.contact,
                    self.contact.pairs_ee,
                    self.contact.n_pairs_ee,
                    self.contact.max_pairs_ee[()],
                    qd.f64(0.0),
                    self.contact.overflow_flag,
                )

    @qd.func(requires_top_level=True)
    def trajectory_query(self):
        self.pt_query()
        self.ee_query()

    def handle_ee_query_overflow(self) -> bool:
        if not self.has_edge_bvh:
            return False
        if self.use_dual_ee:
            return self.ee_dual_state.handle_overflow()
        if int(self.edge_bvh.ee_warp_stack_overflow.to_numpy()):
            raise RuntimeError("warp EE shared frontier stack exhausted; increase ee_warp_stack_capacity")
        return False


@qd.data_oriented
class InfoLBVHBatchedBroadPhaseDop14(LBVHBroadPhase):
    """CGQ default DOP14 broad phase with homogeneous-body culling and dual EE traversal."""

    def __init__(
        self,
        *,
        pt_query: str = "warp",
        ee_query: str = "dual",
        dual_frontier_levels: int = 0,
        dual_target_waves: float = 24.0,
        dual_max_levels: int = 18,
        genesis_legacy_sort_reduce: bool = False,
        genesis_legacy_fp64_bounds: bool = False,
        genesis_legacy_refit: bool = False,
    ) -> None:
        super().__init__(
            bound_type="dop14",
            pt_query=pt_query,
            ee_query=ee_query,
            dual_frontier_levels=dual_frontier_levels,
            dual_target_waves=dual_target_waves,
            dual_max_levels=dual_max_levels,
            genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
            genesis_legacy_fp64_bounds=genesis_legacy_fp64_bounds,
            genesis_legacy_refit=genesis_legacy_refit,
        )
