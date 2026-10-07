from __future__ import annotations

import math

import numpy as np
import quadrants as qd
from quadrants.algorithms import (
    exclusive_scan_add,
    exclusive_scan_scratch_slots,
    sort,
    sort_scratch_slots,
)

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .contact import CONTACT_CONFIG_DEFAULTS, ContactTabular
from .contact_function.codim_thickness import pair_thickness_ee, pair_thickness_ph, pair_thickness_pt
from .contact_function.halfplane_contact import halfplane_signed_distance
from .contact_function.pair_d_hat import pair_d_hat_ph
from .contact_function.screw_ccd import (
    screw_edge_edge_ccd,
    screw_halfplane_ccd,
    screw_point_triangle_ccd,
)
from .dynamic_exclusive_sum import (
    DynamicExclusiveSum,
    dynamic_exclusive_sum,
)
from .dynamic_radix_sort import (
    DynamicRadixSort,
    dynamic_radix_sort,
)
from .fsr_reduce import (
    fast_segmented_reduce_doublet as fsr_reduce_doublet,
    fast_segmented_reduce_triplet as fsr_reduce_triplet,
)
from .sim_system import ActionKind, SimAction, SimData, SimSystem, validate_action_protocol

_CONTACT_SORT_MIN_CAPACITY = 4_865
_CONTACT_SORT_LOG256_MAX_N = 4
_CCD_MAX_ITERS = 50_000
_CCD_ETA = 0.2


def _padded64(value: int) -> int:
    return max(((value + 63) // 64) * 64, 64)


def get_contact_assembly_capacity() -> int:
    return _CONTACT_SORT_MIN_CAPACITY


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class ContactSystem(SimSystem):
    """Organize contact data, dependencies, actions, and capacity growth."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible mutable contact state."""

        adaptive_calm_time: qd.Ndarray
        adaptive_gap_ratio: qd.Ndarray
        adaptive_grow: qd.Ndarray
        adaptive_hysteresis: qd.Ndarray
        adaptive_kappa_grew: qd.Ndarray
        adaptive_kappa_mode: qd.Ndarray
        adaptive_kappa_tick: qd.Ndarray
        adaptive_max_scale: qd.Ndarray
        adaptive_relax_time: qd.Ndarray
        barrier_energy: qd.Ndarray
        body_calm_streak: qd.Ndarray
        body_kappa_scale: qd.Ndarray
        body_min_gap: qd.Ndarray
        capacity_grow_factor: qd.Ndarray
        capacity_shrink_threshold: qd.Ndarray
        ccd_alpha: qd.Ndarray
        ccd_alpha_ee: qd.Ndarray
        ccd_alpha_pe: qd.Ndarray
        ccd_alpha_ph: qd.Ndarray
        ccd_alpha_pp: qd.Ndarray
        ccd_alpha_pt: qd.Ndarray
        ccd_eta: qd.Ndarray
        contact_doublet_gradients: qd.Ndarray
        contact_doublet_vertices: qd.Ndarray
        contact_energy_value: qd.Ndarray
        contact_kappa_scale: qd.Ndarray
        contact_padding_overflow: qd.Ndarray
        contact_triplet_cols: qd.Ndarray
        contact_triplet_rows: qd.Ndarray
        contact_triplet_values: qd.Ndarray
        count_overflow_flag: qd.Ndarray
        d_hat: qd.Ndarray
        default_friction_rate: qd.Ndarray
        doublet_scan_scratch: qd.Ndarray
        doublet_scanner: DynamicExclusiveSum
        doublet_seg_flags: qd.Ndarray
        doublet_seg_ids: qd.Ndarray
        doublet_sort_keys: qd.Ndarray
        doublet_sort_keys_out: qd.Ndarray
        doublet_sort_perm: qd.Ndarray
        doublet_sort_perm_out: qd.Ndarray
        doublet_sort_scratch: qd.Ndarray
        doublet_sort_size: qd.Ndarray
        doublet_sorter: DynamicRadixSort
        dt_sq: qd.Ndarray
        enable_ee_table: qd.Ndarray
        enable_table: qd.Ndarray
        et_overflow_flag: qd.Ndarray
        et_pairs: qd.Ndarray
        et_yield_flag: qd.Ndarray
        frame_ccd_alpha: qd.Ndarray
        friction_energy: qd.Ndarray
        friction_eps_v: qd.Ndarray
        friction_flags_ee: qd.Ndarray
        friction_flags_pe: qd.Ndarray
        friction_flags_ph: qd.Ndarray
        friction_flags_pp: qd.Ndarray
        friction_flags_pt: qd.Ndarray
        friction_overflow_flag: qd.Ndarray
        friction_pair_reach_scale: qd.Ndarray
        friction_pairs_ee: qd.Ndarray
        friction_pairs_pe: qd.Ndarray
        friction_pairs_ph: qd.Ndarray
        friction_pairs_pp: qd.Ndarray
        friction_pairs_pt: qd.Ndarray
        global_calm_frames: qd.Ndarray
        halfplane_contact_element_ids: qd.Ndarray
        halfplane_normals: qd.Ndarray
        halfplane_positions: qd.Ndarray
        init_pair_capacity: qd.Ndarray
        intersection_check: qd.Ndarray
        intersection_flag: qd.Ndarray
        iter_body_min_gap: qd.Ndarray
        iter_min_gap_ratio: qd.Ndarray
        kappa: qd.Ndarray
        kappa_table: qd.Ndarray
        lagged_positions: qd.Ndarray
        max_accd_iters: qd.Ndarray
        max_contact_doublets: qd.Ndarray
        max_contact_triplets: qd.Ndarray
        max_et_pairs: qd.Ndarray
        max_friction_pairs_ee: qd.Ndarray
        max_friction_pairs_pe: qd.Ndarray
        max_friction_pairs_ph: qd.Ndarray
        max_friction_pairs_pp: qd.Ndarray
        max_friction_pairs_pt: qd.Ndarray
        max_pairs_ee: qd.Ndarray
        max_pairs_pe: qd.Ndarray
        max_pairs_ph: qd.Ndarray
        max_pairs_pp: qd.Ndarray
        max_pairs_pt: qd.Ndarray
        max_step_in_d_hat: qd.Ndarray
        min_gap_ratio: qd.Ndarray
        mu_table: qd.Ndarray
        n_active_pairs: qd.Ndarray
        n_contact_doublets: qd.Ndarray
        n_contact_elements: qd.Ndarray
        n_contact_triplets: qd.Ndarray
        n_counted_doublets: qd.Ndarray
        n_counted_triplets: qd.Ndarray
        n_et_pairs: qd.Ndarray
        n_friction_demand_doublets: qd.Ndarray
        n_friction_demand_triplets: qd.Ndarray
        n_friction_pairs_ee: qd.Ndarray
        n_friction_pairs_pe: qd.Ndarray
        n_friction_pairs_ph: qd.Ndarray
        n_friction_pairs_pp: qd.Ndarray
        n_friction_pairs_pt: qd.Ndarray
        n_halfplanes: qd.Ndarray
        n_pairs_ee: qd.Ndarray
        n_pairs_pe: qd.Ndarray
        n_pairs_ph: qd.Ndarray
        n_pairs_pp: qd.Ndarray
        n_pairs_pt: qd.Ndarray
        n_unique_doublets: qd.Ndarray
        n_unique_triplets: qd.Ndarray
        n_verts: qd.Ndarray
        overflow_flag: qd.Ndarray
        padded_contact_doublets: qd.Ndarray
        padded_contact_triplets: qd.Ndarray
        pairs_ee: qd.Ndarray
        pairs_pe: qd.Ndarray
        pairs_ph: qd.Ndarray
        pairs_pp: qd.Ndarray
        pairs_pt: qd.Ndarray
        triplet_scan_scratch: qd.Ndarray
        triplet_scanner: DynamicExclusiveSum
        triplet_seg_flags: qd.Ndarray
        triplet_seg_ids: qd.Ndarray
        triplet_sort_keys: qd.Ndarray
        triplet_sort_keys_out: qd.Ndarray
        triplet_sort_perm: qd.Ndarray
        triplet_sort_perm_out: qd.Ndarray
        triplet_sort_scratch: qd.Ndarray
        triplet_sort_size: qd.Ndarray
        triplet_sorter: DynamicRadixSort
        unique_doublet_gradients: qd.Ndarray
        unique_doublet_vertices: qd.Ndarray
        unique_triplet_cols: qd.Ndarray
        unique_triplet_rows: qd.Ndarray
        unique_triplet_values: qd.Ndarray
        vert_contact_element_ids: qd.Ndarray
        vertex_calm_streak: qd.Ndarray
        vertex_kappa_scale: qd.Ndarray
        vertex_min_gap: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._wire_args = None
        self.broad_phase_init_actions = self.create_action_collection()
        self.contact_assemble_init_actions = self.create_action_collection()
        self.intersection_check = False
        self.genesis_legacy_sort_reduce = False
        self.has_friction = False
        self.has_halfplanes = False
        self.has_codim = False
        self.is_initialized = False

    def wire_data(
        self,
        *,
        n_verts: int,
        n_bodies: int,
        d_hat: float,
        kappa: float,
        dt_sq: float,
        init_pair_capacity: int,
        contact_tabular: ContactTabular,
        friction_mu: float,
        friction_eps_v: float,
        halfplane_positions: np.ndarray,
        halfplane_normals: np.ndarray,
        adaptive_kappa_mode: str,
        adaptive_kappa_tick: str,
        contact_element_ids: np.ndarray | None = None,
        halfplane_contact_element_ids: np.ndarray | None = None,
        intersection_check: bool = False,
        intersection_check_capacity: int = 1_024,
        genesis_legacy_sort_reduce: bool = False,
    ) -> None:
        self._wire_args = {
            "n_verts": n_verts,
            "n_bodies": n_bodies,
            "d_hat": d_hat,
            "kappa": kappa,
            "dt_sq": dt_sq,
            "init_pair_capacity": init_pair_capacity,
            "contact_tabular": contact_tabular,
            "friction_mu": friction_mu,
            "friction_eps_v": friction_eps_v,
            "halfplane_positions": halfplane_positions,
            "halfplane_normals": halfplane_normals,
            "adaptive_kappa_mode": adaptive_kappa_mode,
            "adaptive_kappa_tick": adaptive_kappa_tick,
            "contact_element_ids": contact_element_ids,
            "halfplane_contact_element_ids": halfplane_contact_element_ids,
            "intersection_check": intersection_check,
            "intersection_check_capacity": intersection_check_capacity,
            "genesis_legacy_sort_reduce": genesis_legacy_sort_reduce,
        }
        self.intersection_check = bool(intersection_check)
        self.genesis_legacy_sort_reduce = bool(genesis_legacy_sort_reduce)
        self.has_friction = friction_mu > 0.0
        self.has_halfplanes = len(halfplane_positions) != 0

    def build(self) -> None:
        from .global_body_manager import GlobalBodyManager
        from .global_linear_system import GlobalLinearSystem
        from .global_surface_manager import GlobalSurfaceManager
        from .global_vertex_manager import GlobalVertexManager
        from .lbvh_broad_phase import LBVHBroadPhase
        from .rigid_contact_assemble import RigidContactAssemble

        self.body_system = self.require(GlobalBodyManager)
        self.vertex_system = self.require(GlobalVertexManager)
        self.surface_system = self.require(GlobalSurfaceManager)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.broad_phase_system = self.require(LBVHBroadPhase)
        self.rigid_contact_assemble_system = self.find(RigidContactAssemble)

        data = self.data
        surface = self.surface_system.data
        vertex = self.vertex_system.data
        self.reset_initial_intersections_action = self.create_action(reset_initial_intersections, data)
        self.flag_et_intersections_action = self.create_action(flag_et_intersections, data)
        self.reset_counted_demand_action = self.create_action(reset_counted_demand, data)
        self.adaptive_kappa_update_action = self.create_action(adaptive_kappa_update, data)
        self.adaptive_kappa_newton_tick_action = self.create_action(adaptive_kappa_newton_tick, data)
        self.reset_collision_counts_action = self.create_action(reset_collision_counts, data)
        self.halfplane_query_action = self.create_action(halfplane_query, data, surface, vertex)
        self.init_ccd_action = self.create_action(init_ccd, data)
        self.reset_frame_ccd_action = self.create_action(reset_frame_ccd, data)
        self.ccd_alpha_pt_action = self.create_action(ccd_alpha_pt_kernel, data, surface, vertex)
        self.ccd_alpha_ee_action = self.create_action(ccd_alpha_ee_kernel, data, surface, vertex)
        self.ccd_alpha_ph_action = self.create_action(halfplane_ccd_alpha_kernel, data, surface, vertex)
        self.reduce_ccd_alpha_action = self.create_action(reduce_ccd_alpha_final_kernel, data)
        self.ccd_action = self.create_action(ccd, data, surface, vertex)
        self.reset_contact_energy_action = self.create_action(reset_contact_energy, data)
        self.sum_contact_energy_action = self.create_action(sum_contact_energy, data)
        self.check_assembly_capacity_action = self.create_action(check_assembly_capacity, data)
        self.check_assembly_padding_action = self.create_action(check_assembly_padding, data)
        self.shrink_assembly_padding_action = self.create_action(shrink_assembly_padding, data)
        self.reset_assembly_counts_action = self.create_action(reset_assembly_counts, data)
        self.sort_reduce_action = self.create_action(sort_reduce, data)

    def on_broad_phase(self, init_action: SimAction) -> None:
        validate_action_protocol(
            init_action,
            protocol="ContactSystem.on_broad_phase.init",
            expected_kind=ActionKind.HOST,
            transient_arity=0,
        )
        if self.broad_phase_init_actions._actions:
            raise RuntimeError("ContactSystem already has a broad-phase initializer")
        self.broad_phase_init_actions.register(init_action)

    def on_contact_assemble(self, init_action: SimAction) -> None:
        validate_action_protocol(
            init_action,
            protocol="ContactSystem.on_contact_assemble.init",
            expected_kind=ActionKind.HOST,
            transient_arity=0,
        )
        self.contact_assemble_init_actions.register(init_action)

    def init(self) -> None:
        if self.is_initialized:
            raise RuntimeError("ContactSystem is already initialized")
        if self._wire_args is None:
            raise RuntimeError("ContactSystem data has not been wired")
        args = self._wire_args
        n_verts = args["n_verts"]
        n_bodies = args["n_bodies"]
        if n_verts < 0:
            raise ValueError("ContactSystem n_verts must be non-negative")
        if n_bodies < 0:
            raise ValueError("ContactSystem n_bodies must be non-negative")
        data = self.data
        _wire_contact_params(
            data,
            d_hat=args["d_hat"],
            kappa=args["kappa"],
            init_pair_capacity=args["init_pair_capacity"],
            intersection_check=args["intersection_check"],
            intersection_check_capacity=args["intersection_check_capacity"],
        )
        set_contact_dt_sq(data, args["dt_sq"])
        _wire_contact_friction_params(data, mu=args["friction_mu"], eps_v=args["friction_eps_v"])
        _wire_contact_tabular(data, args["contact_tabular"])
        _wire_contact_halfplanes(
            data,
            args["halfplane_positions"],
            args["halfplane_normals"],
            args["halfplane_contact_element_ids"],
        )
        _set_contact_adaptive_kappa(
            data,
            args["adaptive_kappa_mode"],
            args["adaptive_kappa_tick"],
            n_bodies,
            n_verts,
        )
        _initialize_contact_data(data, n_verts)
        if args["contact_element_ids"] is not None:
            set_contact_element_ids(data, args["contact_element_ids"])
        if not self.broad_phase_init_actions.actions:
            raise RuntimeError("ContactSystem requires a broad-phase initializer")
        for action in self.broad_phase_init_actions.actions:
            action.invoke()
        for action in self.contact_assemble_init_actions.actions:
            action.invoke()
        self._wire_args = None
        self.is_initialized = True

    @qd.func(requires_top_level=True)
    def on_reset_initial_intersections(self):
        reset_initial_intersections(self.data)

    @qd.func(requires_top_level=True)
    def on_flag_et_intersections(self):
        flag_et_intersections(self.data)

    @qd.func(requires_top_level=True)
    def on_reset_counted_demand(self):
        reset_counted_demand(self.data)

    @qd.func(requires_top_level=True)
    def on_update_adaptive_kappa(self):
        adaptive_kappa_update(self.data)

    @qd.func(requires_top_level=True)
    def on_tick_adaptive_kappa_newton(self):
        adaptive_kappa_newton_tick(self.data)

    @qd.func(requires_top_level=True)
    def on_reset_collision_counts(self):
        reset_collision_counts(self.data)

    @qd.func(requires_top_level=True)
    def on_query_halfplanes(self):
        halfplane_query(
            self.data,
            self.surface_system.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_initialize_ccd(self):
        init_ccd(self.data)

    @qd.func(requires_top_level=True)
    def on_reset_frame(self):
        reset_frame_ccd(self.data)

    @qd.func(requires_top_level=True)
    def on_compute_ccd_alpha_pt(self):
        ccd_alpha_pt_kernel(
            self.data,
            self.surface_system.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_compute_ccd_alpha_ee(self):
        ccd_alpha_ee_kernel(
            self.data,
            self.surface_system.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_compute_ccd_alpha_ph(self):
        halfplane_ccd_alpha_kernel(
            self.data,
            self.surface_system.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_reduce_ccd_alpha(self):
        reduce_ccd_alpha_final_kernel(self.data)

    @qd.func(requires_top_level=True)
    def on_ccd(self):
        ccd(
            self.data,
            self.surface_system.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_reset_contact_energy(self):
        reset_contact_energy(self.data)

    @qd.func(requires_top_level=True)
    def on_sum_contact_energy(self):
        sum_contact_energy(self.data)

    @qd.func(requires_top_level=True)
    def on_check_assembly_capacity(self):
        check_assembly_capacity(self.data)

    @qd.func(requires_top_level=True)
    def on_check_assembly_padding(self):
        check_assembly_padding(self.data)

    @qd.func(requires_top_level=True)
    def on_shrink_assembly_padding(self):
        shrink_assembly_padding(self.data)

    @qd.func(requires_top_level=True)
    def on_reset_assembly_counts(self):
        reset_assembly_counts(self.data)

    @qd.func(requires_top_level=True)
    def on_sort_reduce(self):
        sort_reduce(self.data, self.genesis_legacy_sort_reduce)

    def _handle_pair_overflow(self) -> None:
        data = self.data
        self.handle_broad_phase_overflow()
        self.realloc_pair_buffers(
            pt=int(qd_to_numpy(data.n_pairs_pt)),
            ee=int(qd_to_numpy(data.n_pairs_ee)),
            pe=int(qd_to_numpy(data.n_pairs_pe)),
            pp=int(qd_to_numpy(data.n_pairs_pp)),
            ph=int(qd_to_numpy(data.n_pairs_ph)),
        )

    def on_initial_intersection_yield(self, _status):
        from .sim_engine import ContactCheckpoint

        required = int(qd_to_numpy(self.data.n_et_pairs))
        self.realloc_et_pairs(required)
        return ContactCheckpoint.INITIAL_INTERSECTION

    def on_query_yield(self, _status):
        from .sim_engine import ContactCheckpoint

        self._handle_pair_overflow()
        return ContactCheckpoint.QUERY

    def on_friction_yield(self, _status):
        from .sim_engine import ContactCheckpoint

        data = self.data
        required = {
            channel: int(qd_to_numpy(getattr(data, f"n_friction_pairs_{channel}")))
            for channel in ("pt", "ee", "pe", "pp", "ph")
        }
        self.realloc_friction_pair_buffers(required)
        data.friction_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
        return ContactCheckpoint.FRICTION

    def on_count_yield(self, _status):
        from .sim_engine import ContactCheckpoint

        data = self.data
        required_doublets = int(qd_to_numpy(data.n_counted_doublets)) + int(
            qd_to_numpy(data.n_friction_demand_doublets)
        )
        required_triplets = int(qd_to_numpy(data.n_counted_triplets)) + int(
            qd_to_numpy(data.n_friction_demand_triplets)
        )
        self.realloc_assembly_buffers(required_doublets, required_triplets)
        if self.rigid_contact_assemble_system is not None:
            self.rigid_contact_assemble_system.realloc_assembly_buffers(data)
        data.count_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
        return ContactCheckpoint.FILTER

    def on_filter_yield(self, _status):
        from .sim_engine import ContactCheckpoint

        data = self.data
        grow_factor = CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"]
        n_doublets = int(qd_to_numpy(data.n_contact_doublets))
        n_triplets = int(qd_to_numpy(data.n_contact_triplets))
        padded_doublets = max(
            int(qd_to_numpy(data.padded_contact_doublets)),
            min(
                int(np.ceil(n_doublets * grow_factor)),
                data.contact_doublet_vertices.shape[0],
            ),
        )
        padded_triplets = max(
            int(qd_to_numpy(data.padded_contact_triplets)),
            min(
                int(np.ceil(n_triplets * grow_factor)),
                data.contact_triplet_rows.shape[0],
            ),
        )
        self.set_assembly_padding(padded_doublets, padded_triplets)
        data.contact_padding_overflow.from_numpy(np.array(0, dtype=np.int32))
        return ContactCheckpoint.SORT

    def on_et_overflow_yield(self, _status):
        from .sim_engine import ContactCheckpoint

        required = int(qd_to_numpy(self.data.n_et_pairs))
        self.realloc_et_pairs(required)
        return ContactCheckpoint.ET_OVERFLOW

    def on_et_failure_yield(self, _status):
        message = self.et_report_message("step")
        gs.logger.error(message)
        raise RuntimeError(message)

    def raise_if_initial_intersection(self) -> None:
        if int(qd_to_numpy(self.data.n_et_pairs)) == 0:
            return
        message = self.et_report_message("initial state")
        gs.logger.error(message)
        raise RuntimeError(message)

    def et_report_message(self, stage: str) -> str:
        data = self.data
        surface = self.surface_system.data
        vertex = self.vertex_system.data
        count = int(qd_to_numpy(data.n_et_pairs))
        pairs = qd_to_numpy(data.et_pairs)[:count]
        edges = qd_to_numpy(surface.surf_edges)
        faces = qd_to_numpy(surface.surf_triangles)
        positions = qd_to_numpy(vertex.positions)
        body_ids = qd_to_numpy(vertex.body_id)
        geometry_ids = qd_to_numpy(vertex.geometry_id)
        geometry_sources = qd_to_numpy(vertex.geometry_source)
        source_geometry_ids = qd_to_numpy(vertex.source_geometry_id)
        geometry_environments = qd_to_numpy(vertex.geometry_environment)
        reports = []
        for edge, face in pairs[:8]:
            edge_vertex = int(edges[edge, 0])
            edge_vertex_b = int(edges[edge, 1])
            face_vertex = int(faces[face, 0])
            face_vertex_b = int(faces[face, 1])
            face_vertex_c = int(faces[face, 2])
            edge_source = int(geometry_sources[edge_vertex])
            face_source = int(geometry_sources[face_vertex])
            edge_source_name = "FEM" if edge_source == 0 else "RIGID"
            face_source_name = "FEM" if face_source == 0 else "RIGID"
            edge_source_id = int(source_geometry_ids[edge_vertex])
            face_source_id = int(source_geometry_ids[face_vertex])
            edge_environment = int(geometry_environments[edge_vertex])
            face_environment = int(geometry_environments[face_vertex])
            edge_lookup = (
                f"fem_solver.entities[{edge_source_id}]"
                if edge_source == 0
                else f"rigid_solver.geoms[{edge_source_id}]"
            )
            face_lookup = (
                f"fem_solver.entities[{face_source_id}]"
                if face_source == 0
                else f"rigid_solver.geoms[{face_source_id}]"
            )
            reports.append(
                f"(edge {int(edge)}, face {int(face)}, "
                f"edge global_geometry_id {int(geometry_ids[edge_vertex])}, "
                f"face global_geometry_id {int(geometry_ids[face_vertex])}, "
                f"edge source {edge_source_name}, "
                f"edge geo_id {edge_source_id}, "
                f"edge env {edge_environment}, "
                f"edge lookup {edge_lookup}, "
                f"face source {face_source_name}, "
                f"face geo_id {face_source_id}, "
                f"face env {face_environment}, "
                f"face lookup {face_lookup}, "
                f"edge body_id {int(body_ids[edge_vertex])}, "
                f"face body_id {int(body_ids[face_vertex])}, "
                f"edge_positions "
                f"{positions[[edge_vertex, edge_vertex_b]].tolist()}, "
                f"face_positions "
                f"{positions[[face_vertex, face_vertex_b, face_vertex_c]].tolist()})"
            )
        more = f", ... +{count - 8} more" if count > 8 else ""
        contact_state = (
            f"pt_pairs={int(qd_to_numpy(data.n_pairs_pt))}, "
            f"ee_pairs={int(qd_to_numpy(data.n_pairs_ee))}, "
            f"active_pairs={int(qd_to_numpy(data.n_active_pairs))}, "
            f"ccd_alpha={float(qd_to_numpy(data.ccd_alpha)):.9g}, "
            "frame_ccd_alpha="
            f"{float(qd_to_numpy(data.frame_ccd_alpha)):.9g}"
        )
        if self.rigid_contact_assemble_system is not None:
            contact_state += (
                f", proxy_doublets={int(qd_to_numpy(self.rigid_contact_assemble_system.data.rigid_doublet_total))}"
            )
        broad_phase = self.broad_phase_system.data
        if broad_phase.use_dual_ee:
            dual = broad_phase.ee_dual_state
            contact_state += (
                ", dual_selected="
                f"{int(qd_to_numpy(dual.selected_count))}, "
                "dual_parity="
                f"{int(qd_to_numpy(dual.selected_parity))}, "
                "dual_level="
                f"{int(qd_to_numpy(dual.current_level))}, "
                "dual_overflow="
                f"{int(qd_to_numpy(dual.overflow_bits))}, "
                "dual_next_task="
                f"{int(qd_to_numpy(dual.next_task))}"
            )
        return (
            f"ET check: {stage} detected "
            f"{count} edge-triangle intersection pair(s). "
            f"{contact_state}. "
            f"Pairs: {', '.join(reports)}{more}"
        )

    def set_dt_sq(self, dt_sq: float) -> None:
        set_contact_dt_sq(self.data, dt_sq)

    def wire_contact_element_ids(self, contact_element_ids: np.ndarray) -> None:
        set_contact_element_ids(self.data, contact_element_ids)

    def realloc_pair_buffers(self, **required: int) -> None:
        realloc_contact_pair_buffers(self.data, **required)

    def realloc_et_pairs(self, required: int) -> None:
        realloc_contact_et_pairs(self.data, required)

    def handle_broad_phase_overflow(self) -> bool:
        return self.broad_phase_system.handle_ee_query_overflow()

    def set_assembly_padding(self, padded_doublets: int, padded_triplets: int) -> None:
        set_contact_assembly_padding(self.data, padded_doublets, padded_triplets)

    def realloc_assembly_buffers(self, required_doublets: int, required_triplets: int) -> None:
        realloc_contact_assembly_buffers(self.data, required_doublets, required_triplets)

    def realloc_friction_pair_buffers(self, required: dict[str, int]) -> None:
        realloc_contact_friction_pair_buffers(self.data, required)


def _wire_contact_params(
    data,
    *,
    d_hat: float,
    kappa: float,
    init_pair_capacity: int,
    intersection_check: bool = False,
    intersection_check_capacity: int = 1_024,
) -> None:
    if d_hat <= 0.0:
        raise ValueError("contact/d_hat must be positive")
    if kappa <= 0.0:
        raise ValueError("ContactTabular resistance must be positive")
    if init_pair_capacity < 1:
        raise ValueError("contact/init_collision_pair_capacity must be at least one")
    if intersection_check_capacity < 1:
        raise ValueError("contact/intersection_check_capacity must be at least one")

    data.d_hat = qd.ndarray(qd.f64, shape=())
    data.kappa = qd.ndarray(qd.f64, shape=())
    data.init_pair_capacity = qd.ndarray(qd.i32, shape=())
    data.dt_sq = qd.ndarray(qd.f64, shape=())
    data.ccd_eta = qd.ndarray(qd.f64, shape=())
    data.max_step_in_d_hat = qd.ndarray(qd.f64, shape=())
    data.capacity_grow_factor = qd.ndarray(qd.f64, shape=())
    data.capacity_shrink_threshold = qd.ndarray(qd.f64, shape=())
    data.friction_eps_v = qd.ndarray(qd.f64, shape=())
    data.intersection_check = qd.ndarray(qd.i32, shape=())
    data.max_et_pairs = qd.ndarray(qd.i32, shape=())
    data.n_et_pairs = qd.ndarray(qd.i32, shape=())
    data.et_overflow_flag = qd.ndarray(qd.i32, shape=())
    data.et_yield_flag = qd.ndarray(qd.i32, shape=())
    data.et_pairs = qd.ndarray(
        qd.i32,
        shape=(intersection_check_capacity, 2),
    )

    data.d_hat.from_numpy(np.array(d_hat, dtype=np.float64))
    data.kappa.from_numpy(np.array(kappa, dtype=np.float64))
    data.init_pair_capacity.from_numpy(np.array(init_pair_capacity, dtype=np.int32))
    data.dt_sq.from_numpy(np.array(0.0, dtype=np.float64))
    data.ccd_eta.from_numpy(np.array(_CCD_ETA, dtype=np.float64))
    data.max_step_in_d_hat.from_numpy(np.array(CONTACT_CONFIG_DEFAULTS["contact/max_step_in_d_hat"], dtype=np.float64))
    data.capacity_grow_factor.from_numpy(
        np.array(CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"], dtype=np.float64)
    )
    data.capacity_shrink_threshold.from_numpy(
        np.array(CONTACT_CONFIG_DEFAULTS["extras/capacity_shrink_threshold"], dtype=np.float64)
    )
    data.friction_eps_v.from_numpy(np.array(CONTACT_CONFIG_DEFAULTS["friction/eps_v"], dtype=np.float64))
    data.intersection_check.from_numpy(np.array(int(intersection_check), dtype=np.int32))
    data.max_et_pairs.from_numpy(np.array(intersection_check_capacity, dtype=np.int32))
    data.n_et_pairs.from_numpy(np.array(0, dtype=np.int32))
    data.et_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
    data.et_yield_flag.from_numpy(np.array(0, dtype=np.int32))
    data.et_pairs.from_numpy(np.zeros((intersection_check_capacity, 2), dtype=np.int32))
    for channel in ("pt", "ee", "pe", "pp", "ph"):
        setattr(data, f"pairs_{channel}", qd.ndarray(qd.i32, shape=(init_pair_capacity, 2)))
        setattr(data, f"ccd_alpha_{channel}", qd.ndarray(qd.f64, shape=(init_pair_capacity,)))
        setattr(data, f"n_pairs_{channel}", qd.ndarray(qd.i32, shape=()))
        setattr(data, f"max_pairs_{channel}", qd.ndarray(qd.i32, shape=()))
        getattr(data, f"n_pairs_{channel}").from_numpy(np.array(0, dtype=np.int32))
        getattr(data, f"max_pairs_{channel}").from_numpy(np.array(init_pair_capacity, dtype=np.int32))
        getattr(data, f"ccd_alpha_{channel}").from_numpy(np.ones(init_pair_capacity, dtype=np.float64))


def set_contact_dt_sq(data, dt_sq: float) -> None:
    if dt_sq <= 0.0:
        raise ValueError("ContactSystem dt_sq must be positive")
    data.dt_sq.from_numpy(np.array(dt_sq, dtype=np.float64))


def _wire_contact_friction_params(data, *, mu: float, eps_v: float) -> None:
    if mu < 0.0:
        raise ValueError("ContactTabular friction_rate must be non-negative")
    if eps_v <= 0.0:
        raise ValueError("friction/eps_v must be positive")
    data.default_friction_rate = qd.ndarray(qd.f64, shape=())
    data.default_friction_rate.from_numpy(np.array(mu, dtype=np.float64))
    data.friction_eps_v.from_numpy(np.array(eps_v, dtype=np.float64))


def _wire_contact_tabular(data, tabular: ContactTabular) -> None:
    n_elements = len(tabular._elements)
    kappa_table = np.empty(n_elements * n_elements, dtype=np.float64)
    mu_table = np.empty(n_elements * n_elements, dtype=np.float64)
    enable_table = np.empty(n_elements * n_elements, dtype=np.int32)
    enable_ee_table = np.empty(n_elements * n_elements, dtype=np.int32)
    for left in range(n_elements):
        for right in range(n_elements):
            model = tabular.at(left, right)
            index = left * n_elements + right
            kappa_table[index] = model.resistance
            mu_table[index] = model.friction_rate
            enable_table[index] = int(model.enable)
            enable_ee_table[index] = int(model.enable_ee)

    data.n_contact_elements = qd.ndarray(qd.i32, shape=())
    data.kappa_table = qd.ndarray(qd.f64, shape=(max(len(kappa_table), 1),))
    data.mu_table = qd.ndarray(qd.f64, shape=(max(len(mu_table), 1),))
    data.enable_table = qd.ndarray(qd.i32, shape=(max(len(enable_table), 1),))
    data.enable_ee_table = qd.ndarray(qd.i32, shape=(max(len(enable_ee_table), 1),))
    data.n_contact_elements.from_numpy(np.array(n_elements, dtype=np.int32))
    data.kappa_table.from_numpy(kappa_table)
    data.mu_table.from_numpy(mu_table)
    data.enable_table.from_numpy(enable_table)
    data.enable_ee_table.from_numpy(enable_ee_table)


def set_contact_element_ids(data, contact_element_ids: np.ndarray) -> None:
    values = np.ascontiguousarray(contact_element_ids, dtype=np.int32).reshape(-1)
    if len(values) != data.vert_contact_element_ids.shape[0]:
        raise ValueError("ContactSystem contact element IDs must match n_verts")
    if np.any(values < 0):
        raise ValueError("ContactSystem contact element IDs must be non-negative")
    data.vert_contact_element_ids.from_numpy(values)


def _wire_contact_halfplanes(
    data,
    positions: np.ndarray,
    normals: np.ndarray,
    contact_element_ids: np.ndarray | None = None,
) -> None:
    plane_positions = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
    plane_normals = np.ascontiguousarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(plane_positions) != len(plane_normals):
        raise ValueError("ContactSystem halfplane positions and normals must have equal length")
    norms = np.linalg.norm(plane_normals, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-12):
        raise ValueError("ContactSystem halfplane normals must be unit length")
    if contact_element_ids is None:
        plane_contact_element_ids = np.zeros(
            len(plane_positions),
            dtype=np.int32,
        )
    else:
        plane_contact_element_ids = np.ascontiguousarray(
            contact_element_ids,
            dtype=np.int32,
        ).reshape(-1)
        if len(plane_contact_element_ids) != len(plane_positions):
            raise ValueError("ContactSystem halfplane contact element IDs must match the number of halfplanes")
        if np.any(plane_contact_element_ids < 0):
            raise ValueError("ContactSystem halfplane contact element IDs must be non-negative")

    data.n_halfplanes = qd.ndarray(qd.i32, shape=())
    data.halfplane_positions = qd.ndarray(qd.f64, shape=(max(len(plane_positions), 1), 3))
    data.halfplane_normals = qd.ndarray(qd.f64, shape=(max(len(plane_normals), 1), 3))
    data.halfplane_contact_element_ids = qd.ndarray(
        qd.i32,
        shape=(max(len(plane_positions), 1),),
    )
    data.n_halfplanes.from_numpy(np.array(len(plane_positions), dtype=np.int32))
    data.halfplane_positions.from_numpy(plane_positions if len(plane_positions) else np.zeros((1, 3), dtype=np.float64))
    data.halfplane_normals.from_numpy(plane_normals if len(plane_normals) else np.zeros((1, 3), dtype=np.float64))
    data.halfplane_contact_element_ids.from_numpy(
        plane_contact_element_ids if len(plane_contact_element_ids) else np.zeros(1, dtype=np.int32)
    )


def _set_contact_adaptive_kappa(data, mode: str, tick: str, n_bodies: int, n_verts: int) -> None:
    mode_values = {"off": 0, "global": 1, "per-vertex": 2, "per-body": 3}
    tick_values = {"frame": 0, "newton": 1}
    if mode not in mode_values:
        raise ValueError(f"Unsupported contact/adaptive_kappa_mode {mode!r}")
    if tick not in tick_values:
        raise ValueError(f"Unsupported contact/adaptive_kappa_tick {tick!r}")
    if mode == "per-vertex" and tick == "newton":
        raise ValueError("The Newton contact system does not support per-vertex adaptive kappa with the Newton tick")

    data.adaptive_kappa_mode = qd.ndarray(qd.i32, shape=())
    data.adaptive_kappa_tick = qd.ndarray(qd.i32, shape=())
    data.contact_kappa_scale = qd.ndarray(qd.f64, shape=())
    data.adaptive_gap_ratio = qd.ndarray(qd.f64, shape=())
    data.adaptive_grow = qd.ndarray(qd.f64, shape=())
    data.adaptive_max_scale = qd.ndarray(qd.f64, shape=())
    data.adaptive_calm_time = qd.ndarray(qd.f64, shape=())
    data.adaptive_relax_time = qd.ndarray(qd.f64, shape=())
    data.adaptive_hysteresis = qd.ndarray(qd.f64, shape=())
    data.global_calm_frames = qd.ndarray(qd.i32, shape=())
    data.body_kappa_scale = qd.ndarray(qd.f64, shape=(max(n_bodies, 1),))
    data.body_min_gap = qd.ndarray(qd.f64, shape=(max(n_bodies, 1),))
    data.iter_body_min_gap = qd.ndarray(qd.f64, shape=(max(n_bodies, 1),))
    data.body_calm_streak = qd.ndarray(qd.i32, shape=(max(n_bodies, 1),))
    data.vertex_kappa_scale = qd.ndarray(qd.f64, shape=(max(n_verts, 1),))
    data.vertex_min_gap = qd.ndarray(qd.f64, shape=(max(n_verts, 1),))
    data.vertex_calm_streak = qd.ndarray(qd.i32, shape=(max(n_verts, 1),))
    data.adaptive_kappa_mode.from_numpy(np.array(mode_values[mode], dtype=np.int32))
    data.adaptive_kappa_tick.from_numpy(np.array(tick_values[tick], dtype=np.int32))
    data.contact_kappa_scale.from_numpy(np.array(1.0, dtype=np.float64))
    data.adaptive_gap_ratio.from_numpy(np.array(0.01, dtype=np.float64))
    data.adaptive_grow.from_numpy(np.array(2.0, dtype=np.float64))
    data.adaptive_max_scale.from_numpy(np.array(128.0, dtype=np.float64))
    data.adaptive_calm_time.from_numpy(np.array(0.3, dtype=np.float64))
    data.adaptive_relax_time.from_numpy(np.array(0.4, dtype=np.float64))
    data.adaptive_hysteresis.from_numpy(np.array(2.0, dtype=np.float64))
    data.global_calm_frames.from_numpy(np.array(0, dtype=np.int32))
    data.body_kappa_scale.from_numpy(np.ones(max(n_bodies, 1), dtype=np.float64))
    data.body_min_gap.from_numpy(np.full(max(n_bodies, 1), 1e300, dtype=np.float64))
    data.iter_body_min_gap.from_numpy(np.full(max(n_bodies, 1), 1e300, dtype=np.float64))
    data.body_calm_streak.from_numpy(np.zeros(max(n_bodies, 1), dtype=np.int32))
    data.vertex_kappa_scale.from_numpy(np.ones(max(n_verts, 1), dtype=np.float64))
    data.vertex_min_gap.from_numpy(np.full(max(n_verts, 1), 1e300, dtype=np.float64))
    data.vertex_calm_streak.from_numpy(np.zeros(max(n_verts, 1), dtype=np.int32))


def _initialize_contact_data(data, n_verts: int) -> None:
    pair_capacity = data.pairs_pt.shape[0]
    doublet_capacity = _CONTACT_SORT_MIN_CAPACITY
    triplet_capacity = _CONTACT_SORT_MIN_CAPACITY

    data.n_verts = qd.ndarray(qd.i32, shape=())
    data.n_verts.from_numpy(np.array(n_verts, dtype=np.int32))
    data.vert_contact_element_ids = qd.ndarray(qd.i32, shape=(max(n_verts, 1),))
    data.vert_contact_element_ids.from_numpy(np.zeros(max(n_verts, 1), dtype=np.int32))

    _allocate_contact_assembly_buffers(data, doublet_capacity, triplet_capacity)
    _allocate_contact_friction_buffers(data, pair_capacity, n_verts)

    data.n_counted_doublets = qd.ndarray(qd.i32, shape=())
    data.n_counted_triplets = qd.ndarray(qd.i32, shape=())
    data.n_friction_demand_doublets = qd.ndarray(qd.i32, shape=())
    data.n_friction_demand_triplets = qd.ndarray(qd.i32, shape=())
    data.n_active_pairs = qd.ndarray(qd.i32, shape=())
    data.overflow_flag = qd.ndarray(qd.i32, shape=())
    data.count_overflow_flag = qd.ndarray(qd.i32, shape=())
    data.contact_padding_overflow = qd.ndarray(qd.i32, shape=())
    data.intersection_flag = qd.ndarray(qd.i32, shape=())
    data.friction_overflow_flag = qd.ndarray(qd.i32, shape=())
    data.barrier_energy = qd.ndarray(qd.f64, shape=())
    data.friction_energy = qd.ndarray(qd.f64, shape=())
    data.contact_energy_value = qd.ndarray(qd.f64, shape=())
    data.ccd_alpha = qd.ndarray(qd.f64, shape=())
    data.frame_ccd_alpha = qd.ndarray(qd.f64, shape=())
    data.max_accd_iters = qd.ndarray(qd.i32, shape=())
    data.min_gap_ratio = qd.ndarray(qd.f64, shape=())
    data.iter_min_gap_ratio = qd.ndarray(qd.f64, shape=())
    data.adaptive_kappa_grew = qd.ndarray(qd.i32, shape=())

    for scalar in (
        data.n_counted_doublets,
        data.n_counted_triplets,
        data.n_friction_demand_doublets,
        data.n_friction_demand_triplets,
        data.n_active_pairs,
        data.overflow_flag,
        data.count_overflow_flag,
        data.contact_padding_overflow,
        data.intersection_flag,
        data.friction_overflow_flag,
        data.max_accd_iters,
        data.adaptive_kappa_grew,
    ):
        scalar.from_numpy(np.array(0, dtype=np.int32))
    data.barrier_energy.from_numpy(np.array(0.0, dtype=np.float64))
    data.friction_energy.from_numpy(np.array(0.0, dtype=np.float64))
    data.contact_energy_value.from_numpy(np.array(0.0, dtype=np.float64))
    data.ccd_alpha.from_numpy(np.array(1.0, dtype=np.float64))
    data.frame_ccd_alpha.from_numpy(np.array(1.0, dtype=np.float64))
    data.min_gap_ratio.from_numpy(np.array(1e300, dtype=np.float64))
    data.iter_min_gap_ratio.from_numpy(np.array(1e300, dtype=np.float64))


def _allocate_contact_assembly_buffers(data, doublet_capacity: int, triplet_capacity: int) -> None:
    padded_doublets = _padded64(doublet_capacity)
    padded_triplets = _padded64(triplet_capacity)
    data.max_contact_doublets = qd.ndarray(qd.i32, shape=())
    data.max_contact_triplets = qd.ndarray(qd.i32, shape=())
    data.padded_contact_doublets = qd.ndarray(qd.i32, shape=())
    data.padded_contact_triplets = qd.ndarray(qd.i32, shape=())
    data.n_contact_doublets = qd.ndarray(qd.i32, shape=())
    data.n_contact_triplets = qd.ndarray(qd.i32, shape=())
    data.n_unique_doublets = qd.ndarray(qd.i32, shape=())
    data.n_unique_triplets = qd.ndarray(qd.i32, shape=())

    data.max_contact_doublets.from_numpy(np.array(doublet_capacity, dtype=np.int32))
    data.max_contact_triplets.from_numpy(np.array(triplet_capacity, dtype=np.int32))
    data.padded_contact_doublets.from_numpy(np.array(doublet_capacity, dtype=np.int32))
    data.padded_contact_triplets.from_numpy(np.array(triplet_capacity, dtype=np.int32))
    for scalar in (
        data.n_contact_doublets,
        data.n_contact_triplets,
        data.n_unique_doublets,
        data.n_unique_triplets,
    ):
        scalar.from_numpy(np.array(0, dtype=np.int32))

    data.contact_doublet_vertices = qd.ndarray(qd.i32, shape=(doublet_capacity,))
    data.contact_doublet_gradients = qd.ndarray(qd.f64, shape=(doublet_capacity, 3))
    data.contact_triplet_rows = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.contact_triplet_cols = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.contact_triplet_values = qd.ndarray(qd.f64, shape=(triplet_capacity, 3, 3))
    data.unique_doublet_vertices = qd.ndarray(qd.i32, shape=(doublet_capacity,))
    data.unique_doublet_gradients = qd.ndarray(qd.f64, shape=(doublet_capacity, 3))
    data.unique_triplet_rows = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.unique_triplet_cols = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.unique_triplet_values = qd.ndarray(qd.f64, shape=(triplet_capacity, 3, 3))

    data.doublet_sort_keys = qd.ndarray(qd.u32, shape=(padded_doublets,))
    data.doublet_sort_keys_out = qd.ndarray(qd.u32, shape=(padded_doublets,))
    data.doublet_sort_perm = qd.ndarray(qd.i32, shape=(padded_doublets,))
    data.doublet_sort_perm_out = qd.ndarray(qd.i32, shape=(padded_doublets,))
    data.doublet_sort_size = qd.ndarray(qd.i32, shape=())
    data.doublet_sorter = DynamicRadixSort(
        qd.u32,
        padded_doublets,
    )
    data.doublet_sort_scratch = qd.ndarray(
        qd.u32,
        shape=(max(sort_scratch_slots(padded_doublets, _CONTACT_SORT_LOG256_MAX_N), 1),),
    )
    data.doublet_seg_flags = qd.ndarray(qd.i32, shape=(padded_doublets,))
    data.doublet_seg_ids = qd.ndarray(qd.i32, shape=(padded_doublets,))
    data.doublet_scanner = DynamicExclusiveSum(
        padded_doublets,
    )
    data.doublet_scan_scratch = qd.ndarray(
        qd.i32,
        shape=(max(exclusive_scan_scratch_slots(padded_doublets, _CONTACT_SORT_LOG256_MAX_N), 1),),
    )

    data.triplet_sort_keys = qd.ndarray(qd.u64, shape=(padded_triplets,))
    data.triplet_sort_keys_out = qd.ndarray(qd.u64, shape=(padded_triplets,))
    data.triplet_sort_perm = qd.ndarray(qd.i32, shape=(padded_triplets,))
    data.triplet_sort_perm_out = qd.ndarray(qd.i32, shape=(padded_triplets,))
    data.triplet_sort_size = qd.ndarray(qd.i32, shape=())
    data.triplet_sorter = DynamicRadixSort(
        qd.u64,
        padded_triplets,
    )
    data.triplet_sort_scratch = qd.ndarray(
        qd.u32,
        shape=(max(sort_scratch_slots(padded_triplets, _CONTACT_SORT_LOG256_MAX_N), 1),),
    )
    data.triplet_seg_flags = qd.ndarray(qd.i32, shape=(padded_triplets,))
    data.triplet_seg_ids = qd.ndarray(qd.i32, shape=(padded_triplets,))
    data.triplet_scanner = DynamicExclusiveSum(
        padded_triplets,
    )
    data.triplet_scan_scratch = qd.ndarray(
        qd.i32,
        shape=(max(exclusive_scan_scratch_slots(padded_triplets, _CONTACT_SORT_LOG256_MAX_N), 1),),
    )
    data.doublet_sort_size.from_numpy(np.array(0, dtype=np.int32))
    data.triplet_sort_size.from_numpy(np.array(0, dtype=np.int32))


def _allocate_contact_friction_buffers(data, pair_capacity: int, n_verts: int) -> None:
    for channel in ("pt", "ee", "pe", "pp", "ph"):
        setattr(data, f"friction_pairs_{channel}", qd.ndarray(qd.i32, shape=(pair_capacity, 2)))
        setattr(data, f"friction_flags_{channel}", qd.ndarray(qd.i32, shape=(pair_capacity,)))
        setattr(data, f"n_friction_pairs_{channel}", qd.ndarray(qd.i32, shape=()))
        setattr(data, f"max_friction_pairs_{channel}", qd.ndarray(qd.i32, shape=()))
        getattr(data, f"n_friction_pairs_{channel}").from_numpy(np.array(0, dtype=np.int32))
        getattr(data, f"max_friction_pairs_{channel}").from_numpy(np.array(pair_capacity, dtype=np.int32))

    data.lagged_positions = qd.ndarray(qd.f64, shape=(max(n_verts, 1), 3))
    data.friction_pair_reach_scale = qd.ndarray(qd.f64, shape=(pair_capacity * 5,))


def _grown_contact_capacity(required: int, current: int) -> int:
    grow_factor = CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"]
    return max(math.ceil(required * grow_factor), current + 1)


def _grow_dynamic_radix_sort(sorter, key_dtype, capacity: int) -> None:
    grown = DynamicRadixSort(key_dtype, capacity)
    sorter.lookback_status = grown.lookback_status
    sorter.lookback_partial = grown.lookback_partial
    sorter.lookback_complete = grown.lookback_complete


def _grow_dynamic_exclusive_sum(scanner, capacity: int) -> None:
    grown = DynamicExclusiveSum(capacity)
    scanner.status = grown.status
    scanner.partial = grown.partial
    scanner.complete = grown.complete


def realloc_contact_pair_buffers(
    data,
    *,
    pt: int,
    ee: int,
    pe: int,
    pp: int,
    ph: int,
) -> None:
    for channel, required in (("pt", pt), ("ee", ee), ("pe", pe), ("pp", pp), ("ph", ph)):
        current = getattr(data, f"pairs_{channel}").shape[0]
        if required <= current:
            continue
        capacity = _grown_contact_capacity(required, current)
        setattr(data, f"pairs_{channel}", qd.ndarray(qd.i32, shape=(capacity, 2)))
        setattr(data, f"ccd_alpha_{channel}", qd.ndarray(qd.f64, shape=(capacity,)))
        getattr(data, f"ccd_alpha_{channel}").from_numpy(np.ones(capacity, dtype=np.float64))
        getattr(data, f"max_pairs_{channel}").from_numpy(np.array(capacity, dtype=np.int32))
    data.overflow_flag.from_numpy(np.array(0, dtype=np.int32))


def realloc_contact_et_pairs(data, required: int) -> None:
    current = data.et_pairs.shape[0]
    if required <= current:
        return
    capacity = _grown_contact_capacity(required, current)
    data.et_pairs = qd.ndarray(qd.i32, shape=(capacity, 2))
    data.et_pairs.from_numpy(np.zeros((capacity, 2), dtype=np.int32))
    data.max_et_pairs.from_numpy(np.array(capacity, dtype=np.int32))
    data.n_et_pairs.from_numpy(np.array(0, dtype=np.int32))
    data.et_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
    data.et_yield_flag.from_numpy(np.array(0, dtype=np.int32))


def set_contact_assembly_padding(
    data,
    padded_doublets: int,
    padded_triplets: int,
) -> None:
    doublet_capacity = data.contact_doublet_vertices.shape[0]
    triplet_capacity = data.contact_triplet_rows.shape[0]
    if not 0 <= padded_doublets <= doublet_capacity:
        raise ValueError("Contact doublet padding must fit its allocation")
    if not 0 <= padded_triplets <= triplet_capacity:
        raise ValueError("Contact triplet padding must fit its allocation")
    data.padded_contact_doublets.from_numpy(np.array(padded_doublets, dtype=np.int32))
    data.padded_contact_triplets.from_numpy(np.array(padded_triplets, dtype=np.int32))


def realloc_contact_assembly_buffers(data, required_doublets: int, required_triplets: int) -> None:
    current_doublets = data.contact_doublet_vertices.shape[0]
    current_triplets = data.contact_triplet_rows.shape[0]
    if required_doublets <= current_doublets and required_triplets <= current_triplets:
        data.max_contact_doublets.from_numpy(np.array(current_doublets, dtype=np.int32))
        data.max_contact_triplets.from_numpy(np.array(current_triplets, dtype=np.int32))
        data.count_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
        return
    if required_doublets > current_doublets:
        doublet_capacity = _grown_contact_capacity(required_doublets, current_doublets)
        padded_doublets = _padded64(doublet_capacity)
        data.contact_doublet_vertices = qd.ndarray(qd.i32, shape=(doublet_capacity,))
        data.contact_doublet_gradients = qd.ndarray(qd.f64, shape=(doublet_capacity, 3))
        data.unique_doublet_vertices = qd.ndarray(qd.i32, shape=(doublet_capacity,))
        data.unique_doublet_gradients = qd.ndarray(qd.f64, shape=(doublet_capacity, 3))
        data.doublet_sort_keys = qd.ndarray(qd.u32, shape=(padded_doublets,))
        data.doublet_sort_keys_out = qd.ndarray(qd.u32, shape=(padded_doublets,))
        data.doublet_sort_perm = qd.ndarray(qd.i32, shape=(padded_doublets,))
        data.doublet_sort_perm_out = qd.ndarray(qd.i32, shape=(padded_doublets,))
        data.doublet_sort_scratch = qd.ndarray(
            qd.u32,
            shape=(max(sort_scratch_slots(padded_doublets, _CONTACT_SORT_LOG256_MAX_N), 1),),
        )
        data.doublet_seg_flags = qd.ndarray(qd.i32, shape=(padded_doublets,))
        data.doublet_seg_ids = qd.ndarray(qd.i32, shape=(padded_doublets,))
        data.doublet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(max(exclusive_scan_scratch_slots(padded_doublets, _CONTACT_SORT_LOG256_MAX_N), 1),),
        )
        _grow_dynamic_radix_sort(data.doublet_sorter, qd.u32, padded_doublets)
        _grow_dynamic_exclusive_sum(data.doublet_scanner, padded_doublets)
        data.max_contact_doublets.from_numpy(np.array(doublet_capacity, dtype=np.int32))
        data.padded_contact_doublets.from_numpy(np.array(doublet_capacity, dtype=np.int32))
    if required_triplets > current_triplets:
        triplet_capacity = _grown_contact_capacity(required_triplets, current_triplets)
        padded_triplets = _padded64(triplet_capacity)
        data.contact_triplet_rows = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        data.contact_triplet_cols = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        data.contact_triplet_values = qd.ndarray(qd.f64, shape=(triplet_capacity, 3, 3))
        data.unique_triplet_rows = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        data.unique_triplet_cols = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        data.unique_triplet_values = qd.ndarray(qd.f64, shape=(triplet_capacity, 3, 3))
        data.triplet_sort_keys = qd.ndarray(qd.u64, shape=(padded_triplets,))
        data.triplet_sort_keys_out = qd.ndarray(qd.u64, shape=(padded_triplets,))
        data.triplet_sort_perm = qd.ndarray(qd.i32, shape=(padded_triplets,))
        data.triplet_sort_perm_out = qd.ndarray(qd.i32, shape=(padded_triplets,))
        data.triplet_sort_scratch = qd.ndarray(
            qd.u32,
            shape=(max(sort_scratch_slots(padded_triplets, _CONTACT_SORT_LOG256_MAX_N), 1),),
        )
        data.triplet_seg_flags = qd.ndarray(qd.i32, shape=(padded_triplets,))
        data.triplet_seg_ids = qd.ndarray(qd.i32, shape=(padded_triplets,))
        data.triplet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(max(exclusive_scan_scratch_slots(padded_triplets, _CONTACT_SORT_LOG256_MAX_N), 1),),
        )
        _grow_dynamic_radix_sort(data.triplet_sorter, qd.u64, padded_triplets)
        _grow_dynamic_exclusive_sum(data.triplet_scanner, padded_triplets)
        data.max_contact_triplets.from_numpy(np.array(triplet_capacity, dtype=np.int32))
        data.padded_contact_triplets.from_numpy(np.array(triplet_capacity, dtype=np.int32))
    data.count_overflow_flag.from_numpy(np.array(0, dtype=np.int32))


def realloc_contact_friction_pair_buffers(data, required: dict[str, int]) -> None:
    for channel in ("pt", "ee", "pe", "pp", "ph"):
        current = getattr(data, f"friction_pairs_{channel}").shape[0]
        demand = required.get(channel, 0)
        if demand <= current:
            continue
        capacity = _grown_contact_capacity(demand, current)
        setattr(data, f"friction_pairs_{channel}", qd.ndarray(qd.i32, shape=(capacity, 2)))
        setattr(data, f"friction_flags_{channel}", qd.ndarray(qd.i32, shape=(capacity,)))
        getattr(data, f"max_friction_pairs_{channel}").from_numpy(np.array(capacity, dtype=np.int32))


@qd.func(requires_top_level=True)
def reset_initial_intersections(data):
    for _ in range(1):
        data.n_et_pairs[()] = 0
        data.et_overflow_flag[()] = 0
        data.et_yield_flag[()] = 0


@qd.func(requires_top_level=True)
def flag_et_intersections(data):
    for _ in range(1):
        data.et_yield_flag[()] = qd.i32(data.n_et_pairs[()] > 0)


@qd.func(requires_top_level=True)
def reset_counted_demand(data):
    for _ in range(1):
        data.n_counted_doublets[()] = 0
        data.n_counted_triplets[()] = 0
        data.count_overflow_flag[()] = 0


@qd.func
def adaptive_kappa_step(data, scale, gap, calm):
    next_scale = scale
    next_calm = calm
    if gap < data.adaptive_gap_ratio[()]:
        next_calm = 0
        next_scale = qd.min(scale * data.adaptive_grow[()], data.adaptive_max_scale[()])
    elif gap > data.adaptive_hysteresis[()] * data.adaptive_gap_ratio[()]:
        dt = qd.sqrt(data.dt_sq[()])
        hold = qd.i32(qd.ceil(data.adaptive_calm_time[()] / dt))
        if next_calm < qd.max(hold, 1):
            next_calm = next_calm + 1
        else:
            next_scale = qd.max(scale * qd.exp(-dt / data.adaptive_relax_time[()]), 1.0)
    return qd.Vector([next_scale, qd.f64(next_calm)])


@qd.func
def adaptive_kappa_frame_step(data, scale, gap, calm):
    result = qd.Vector([scale, qd.f64(calm)])
    if data.adaptive_kappa_tick[()] == 1 and gap < data.adaptive_gap_ratio[()]:
        result[1] = 0.0
    else:
        result = adaptive_kappa_step(data, scale, gap, calm)
    return result


@qd.func(requires_top_level=True)
def adaptive_kappa_update(data):
    for _ in range(1):
        mode = data.adaptive_kappa_mode[()]
        if mode == 1:
            result = adaptive_kappa_frame_step(
                data,
                data.contact_kappa_scale[()],
                data.min_gap_ratio[()],
                data.global_calm_frames[()],
            )
            data.contact_kappa_scale[()] = result[0]
            data.global_calm_frames[()] = qd.i32(result[1])
            data.min_gap_ratio[()] = qd.f64(1e300)
        data.adaptive_kappa_grew[()] = 0
    for body in range(data.body_kappa_scale.shape[0]):
        if data.adaptive_kappa_mode[()] == 3:
            result = adaptive_kappa_frame_step(
                data,
                data.body_kappa_scale[body],
                data.body_min_gap[body],
                data.body_calm_streak[body],
            )
            data.body_kappa_scale[body] = result[0]
            data.body_calm_streak[body] = qd.i32(result[1])
            data.body_min_gap[body] = qd.f64(1e300)
    for vertex in range(data.vertex_kappa_scale.shape[0]):
        if data.adaptive_kappa_mode[()] == 2:
            result = adaptive_kappa_step(
                data,
                data.vertex_kappa_scale[vertex],
                data.vertex_min_gap[vertex],
                data.vertex_calm_streak[vertex],
            )
            data.vertex_kappa_scale[vertex] = result[0]
            data.vertex_calm_streak[vertex] = qd.i32(result[1])
            data.vertex_min_gap[vertex] = qd.f64(1e300)


@qd.func(requires_top_level=True)
def adaptive_kappa_newton_tick(data):
    for _ in range(1):
        data.adaptive_kappa_grew[()] = 0
        if data.adaptive_kappa_tick[()] == 1 and data.adaptive_kappa_mode[()] == 1:
            if data.iter_min_gap_ratio[()] < data.adaptive_gap_ratio[()]:
                old_scale = data.contact_kappa_scale[()]
                new_scale = qd.min(old_scale * data.adaptive_grow[()], data.adaptive_max_scale[()])
                data.contact_kappa_scale[()] = new_scale
                data.adaptive_kappa_grew[()] = qd.i32(new_scale > old_scale)
            data.iter_min_gap_ratio[()] = qd.f64(1e300)
    for body in range(data.body_kappa_scale.shape[0]):
        if data.adaptive_kappa_tick[()] == 1 and data.adaptive_kappa_mode[()] == 3:
            if data.iter_body_min_gap[body] < data.adaptive_gap_ratio[()]:
                old_scale = data.body_kappa_scale[body]
                new_scale = qd.min(old_scale * data.adaptive_grow[()], data.adaptive_max_scale[()])
                data.body_kappa_scale[body] = new_scale
                if new_scale > old_scale:
                    data.adaptive_kappa_grew[()] = 1
            data.iter_body_min_gap[body] = qd.f64(1e300)


@qd.func(requires_top_level=True)
def reset_collision_counts(data):
    for _ in range(1):
        data.n_pairs_pt[()] = 0
        data.n_pairs_ee[()] = 0
        data.n_pairs_pe[()] = 0
        data.n_pairs_pp[()] = 0
        data.n_pairs_ph[()] = 0
        data.overflow_flag[()] = 0
        data.intersection_flag[()] = 0


@qd.func(requires_top_level=True)
def halfplane_query(data, surface: qd.template(), vertex: qd.template()):
    for pair_index in range(surface.n_surf_verts[()] * data.n_halfplanes[()]):
        surface_vertex = pair_index // data.n_halfplanes[()]
        plane = pair_index - surface_vertex * data.n_halfplanes[()]
        vertex_id = surface.surf_verts[surface_vertex]
        vertex_element = data.vert_contact_element_ids[vertex_id]
        plane_element = data.halfplane_contact_element_ids[plane]
        if data.enable_table[vertex_element * data.n_contact_elements[()] + plane_element] == 0:
            continue
        current_distance = halfplane_signed_distance(
            vertex.positions[vertex_id, 0],
            vertex.positions[vertex_id, 1],
            vertex.positions[vertex_id, 2],
            data.halfplane_positions[plane, 0],
            data.halfplane_positions[plane, 1],
            data.halfplane_positions[plane, 2],
            data.halfplane_normals[plane, 0],
            data.halfplane_normals[plane, 1],
            data.halfplane_normals[plane, 2],
        )
        endpoint_distance = halfplane_signed_distance(
            vertex.trajectory_end_positions[vertex_id, 0],
            vertex.trajectory_end_positions[vertex_id, 1],
            vertex.trajectory_end_positions[vertex_id, 2],
            data.halfplane_positions[plane, 0],
            data.halfplane_positions[plane, 1],
            data.halfplane_positions[plane, 2],
            data.halfplane_normals[plane, 0],
            data.halfplane_normals[plane, 1],
            data.halfplane_normals[plane, 2],
        )
        d_hat = pair_d_hat_ph(vertex.d_hats, data.d_hat[()], vertex_id)
        xi = pair_thickness_ph(vertex.thicknesses, vertex_id)
        if current_distance <= xi:
            data.intersection_flag[()] = 1
        elif qd.min(current_distance, endpoint_distance) < d_hat + xi:
            output = qd.atomic_add(data.n_pairs_ph[()], 1)
            if output < data.max_pairs_ph[()]:
                data.pairs_ph[output, 0] = surface_vertex
                data.pairs_ph[output, 1] = plane
            else:
                data.overflow_flag[()] = 1


@qd.func(requires_top_level=True)
def init_ccd(data):
    for _ in range(1):
        data.ccd_alpha[()] = 1.0
        data.max_accd_iters[()] = 0


@qd.func(requires_top_level=True)
def reset_frame_ccd(data):
    for _ in range(1):
        data.frame_ccd_alpha[()] = 1.0


@qd.func(requires_top_level=True)
def ccd_alpha_pt_kernel(data, surface: qd.template(), vertex: qd.template()):
    for pair_index in range(data.n_pairs_pt[()]):
        surface_vertex = data.pairs_pt[pair_index, 0]
        face = data.pairs_pt[pair_index, 1]
        ids = qd.Vector(
            [
                surface.surf_verts[surface_vertex],
                surface.surf_triangles[face, 0],
                surface.surf_triangles[face, 1],
                surface.surf_triangles[face, 2],
            ]
        )
        result = qd.Vector.zero(qd.f64, 1)
        screw_point_triangle_ccd(
            vertex,
            ids[0],
            ids[1],
            ids[2],
            ids[3],
            data.ccd_eta[()],
            pair_thickness_pt(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3]),
            50_000,
            result,
        )
        data.ccd_alpha_pt[pair_index] = result[0]
        qd.atomic_min(data.ccd_alpha[()], result[0])


@qd.func(requires_top_level=True)
def ccd_alpha_ee_kernel(data, surface: qd.template(), vertex: qd.template()):
    for pair_index in range(data.n_pairs_ee[()]):
        edge_a = data.pairs_ee[pair_index, 0]
        edge_b = data.pairs_ee[pair_index, 1]
        ids = qd.Vector(
            [
                surface.surf_edges[edge_a, 0],
                surface.surf_edges[edge_a, 1],
                surface.surf_edges[edge_b, 0],
                surface.surf_edges[edge_b, 1],
            ]
        )
        result = qd.Vector.zero(qd.f64, 1)
        screw_edge_edge_ccd(
            vertex,
            ids[0],
            ids[1],
            ids[2],
            ids[3],
            data.ccd_eta[()],
            pair_thickness_ee(vertex.thicknesses, ids[0], ids[1], ids[2], ids[3]),
            50_000,
            result,
        )
        data.ccd_alpha_ee[pair_index] = result[0]
        qd.atomic_min(data.ccd_alpha[()], result[0])


@qd.func(requires_top_level=True)
def halfplane_ccd_alpha_kernel(data, surface: qd.template(), vertex: qd.template()):
    for pair_index in range(data.n_pairs_ph[()]):
        surface_vertex = data.pairs_ph[pair_index, 0]
        plane = data.pairs_ph[pair_index, 1]
        vertex_id = surface.surf_verts[surface_vertex]
        result = qd.Vector.zero(qd.f64, 1)
        normal = qd.Vector(
            [
                data.halfplane_normals[plane, 0],
                data.halfplane_normals[plane, 1],
                data.halfplane_normals[plane, 2],
            ]
        )
        plane_position = qd.Vector(
            [
                data.halfplane_positions[plane, 0],
                data.halfplane_positions[plane, 1],
                data.halfplane_positions[plane, 2],
            ]
        )
        screw_halfplane_ccd(
            vertex,
            vertex_id,
            normal,
            normal.dot(plane_position),
            data.ccd_eta[()],
            pair_thickness_ph(vertex.thicknesses, vertex_id),
            50_000,
            result,
        )
        data.ccd_alpha_ph[pair_index] = result[0]
        qd.atomic_min(data.ccd_alpha[()], result[0])


@qd.func(requires_top_level=True)
def reduce_ccd_alpha_final_kernel(data):
    for _ in range(1):
        data.frame_ccd_alpha[()] = qd.min(
            data.frame_ccd_alpha[()],
            data.ccd_alpha[()],
        )


@qd.func(requires_top_level=True)
def ccd(data, surface: qd.template(), vertex: qd.template()):
    ccd_alpha_pt_kernel(data, surface, vertex)
    ccd_alpha_ee_kernel(data, surface, vertex)
    halfplane_ccd_alpha_kernel(data, surface, vertex)
    reduce_ccd_alpha_final_kernel(data)


@qd.func(requires_top_level=True)
def reset_contact_energy(data):
    for _ in range(1):
        data.barrier_energy[()] = 0.0
        data.friction_energy[()] = 0.0
        data.contact_energy_value[()] = 0.0


@qd.func(requires_top_level=True)
def sum_contact_energy(data):
    for _ in range(1):
        data.contact_energy_value[()] = data.barrier_energy[()] + data.friction_energy[()]


@qd.func(requires_top_level=True)
def check_assembly_capacity(data):
    for _ in range(1):
        required_doublets = data.n_counted_doublets[()] + data.n_friction_demand_doublets[()]
        required_triplets = data.n_counted_triplets[()] + data.n_friction_demand_triplets[()]
        overflow = (
            required_doublets > data.max_contact_doublets[()] or required_triplets > data.max_contact_triplets[()]
        )
        data.count_overflow_flag[()] = qd.i32(overflow)


@qd.func(requires_top_level=True)
def check_assembly_padding(data):
    for _ in range(1):
        data.contact_padding_overflow[()] = qd.i32(
            data.n_contact_doublets[()] > data.padded_contact_doublets[()]
            or data.n_contact_triplets[()] > data.padded_contact_triplets[()]
        )


@qd.func(requires_top_level=True)
def shrink_assembly_padding(data):
    for _ in range(1):
        n_doublets = data.n_contact_doublets[()]
        n_triplets = data.n_contact_triplets[()]
        if n_doublets > 0 or n_triplets > 0:
            target_doublets = qd.i32(qd.ceil(qd.f64(n_doublets) * data.capacity_grow_factor[()]))
            target_triplets = qd.i32(qd.ceil(qd.f64(n_triplets) * data.capacity_grow_factor[()]))
            target_doublets = qd.min(
                qd.max(target_doublets, 4865),
                data.max_contact_doublets[()],
            )
            target_triplets = qd.min(
                qd.max(target_triplets, 4865),
                data.max_contact_triplets[()],
            )
            if qd.f64(target_doublets) < (
                qd.f64(data.padded_contact_doublets[()]) * data.capacity_shrink_threshold[()]
            ):
                data.padded_contact_doublets[()] = target_doublets
            if qd.f64(target_triplets) < (
                qd.f64(data.padded_contact_triplets[()]) * data.capacity_shrink_threshold[()]
            ):
                data.padded_contact_triplets[()] = target_triplets


@qd.func(requires_top_level=True)
def reset_assembly_counts(data):
    for _ in range(1):
        data.n_contact_doublets[()] = 0
        data.n_contact_triplets[()] = 0
        data.n_unique_doublets[()] = 0
        data.n_unique_triplets[()] = 0
        data.n_active_pairs[()] = 0
        data.contact_padding_overflow[()] = 0


@qd.func(requires_top_level=True)
def doublet_sort_seed(data):
    qd.loop_config(name="contact_doublet_sort_seed")
    for index in range(data.doublet_sort_keys.shape[0]):
        if index < data.padded_contact_doublets[()]:
            if index < data.n_contact_doublets[()]:
                data.doublet_sort_keys[index] = qd.u32(data.contact_doublet_vertices[index])
                data.doublet_sort_perm[index] = index
            else:
                data.doublet_sort_keys[index] = qd.u32(0xFFFFFFFF)
                data.doublet_sort_perm[index] = index
    for _ in range(1):
        data.doublet_sort_size[()] = data.n_contact_doublets[()]


@qd.func(requires_top_level=True)
def doublet_sort_radix(data, genesis_legacy_sort_reduce: qd.template()):
    if qd.static(genesis_legacy_sort_reduce):
        sort(
            data.doublet_sort_keys,
            data.doublet_sort_keys_out,
            data.doublet_sort_perm,
            data.doublet_sort_perm_out,
            data.doublet_sort_scratch,
            data.doublet_sort_size,
            qd.u32,
            True,
            32,
            4,
        )
    else:
        dynamic_radix_sort(
            data.doublet_sorter,
            data.doublet_sort_keys,
            data.doublet_sort_keys_out,
            data.doublet_sort_perm,
            data.doublet_sort_perm_out,
            data.padded_contact_doublets[()],
        )


@qd.func(requires_top_level=True)
def doublet_segment_flags(data):
    qd.loop_config(name="contact_doublet_segment_flags")
    for index in range(data.doublet_sort_keys.shape[0]):
        if index < data.padded_contact_doublets[()]:
            flag = qd.i32(0)
            if index < data.n_contact_doublets[()] and (
                index == data.n_contact_doublets[()] - 1
                or data.doublet_sort_keys[index] != data.doublet_sort_keys[index + 1]
            ):
                flag = qd.i32(1)
            data.doublet_seg_flags[index] = flag


@qd.func(requires_top_level=True)
def doublet_scan(data, genesis_legacy_sort_reduce: qd.template()):
    if qd.static(genesis_legacy_sort_reduce):
        exclusive_scan_add(
            data.doublet_seg_flags,
            data.doublet_seg_ids,
            data.doublet_scan_scratch,
            data.n_contact_doublets[()],
            qd.i32,
            4,
        )
    else:
        dynamic_exclusive_sum(
            data.doublet_scanner,
            data.doublet_seg_flags,
            data.doublet_seg_ids,
            data.padded_contact_doublets[()],
        )


@qd.func(requires_top_level=True)
def doublet_zero_unique(data):
    qd.loop_config(name="contact_doublet_zero_unique")
    for index in range(data.unique_doublet_gradients.shape[0]):
        if index < data.padded_contact_doublets[()]:
            for axis in qd.static(range(3)):
                data.unique_doublet_gradients[index, axis] = 0.0


@qd.func(requires_top_level=True)
def doublet_fsr_merge(data, genesis_legacy_sort_reduce: qd.template()):
    if qd.static(genesis_legacy_sort_reduce):
        qd.loop_config(name="contact_doublet_fsr_merge_legacy")
        for index in range(data.n_contact_doublets[()]):
            source = qd.i32(data.doublet_sort_perm[index])
            segment = data.doublet_seg_ids[index]
            for axis in qd.static(range(3)):
                qd.atomic_add(
                    data.unique_doublet_gradients[
                        segment,
                        axis,
                    ],
                    data.contact_doublet_gradients[
                        source,
                        axis,
                    ],
                )
    else:
        fsr_reduce_doublet(
            data.doublet_seg_ids,
            data.doublet_sort_perm,
            data.doublet_sort_keys,
            data.contact_doublet_gradients,
            data.unique_doublet_gradients,
            data.n_contact_doublets,
            data.padded_contact_doublets,
            data.doublet_sort_keys.shape[0],
        )


@qd.func(requires_top_level=True)
def doublet_extract_unique(data):
    qd.loop_config(name="contact_doublet_extract_unique")
    for index in range(data.doublet_sort_keys.shape[0]):
        if (
            index < data.padded_contact_doublets[()]
            and index < data.n_contact_doublets[()]
            and data.doublet_seg_flags[index] != 0
        ):
            segment = qd.i32(data.doublet_seg_ids[index])
            data.unique_doublet_vertices[segment] = qd.i32(data.doublet_sort_keys[index])
            if index == data.n_contact_doublets[()] - 1:
                data.n_unique_doublets[()] = segment + 1


@qd.func(requires_top_level=True)
def triplet_sort_seed(data):
    qd.loop_config(name="contact_triplet_sort_seed")
    for index in range(data.triplet_sort_keys.shape[0]):
        if index < data.padded_contact_triplets[()]:
            if index < data.n_contact_triplets[()]:
                row = qd.u64(data.contact_triplet_rows[index])
                column = qd.u64(data.contact_triplet_cols[index])
                data.triplet_sort_keys[index] = (row << 32) | column
                data.triplet_sort_perm[index] = index
            else:
                data.triplet_sort_keys[index] = qd.u64(0xFFFFFFFFFFFFFFFF)
                data.triplet_sort_perm[index] = index
    for _ in range(1):
        data.triplet_sort_size[()] = data.n_contact_triplets[()]


@qd.func(requires_top_level=True)
def triplet_sort_radix(data, genesis_legacy_sort_reduce: qd.template()):
    if qd.static(genesis_legacy_sort_reduce):
        sort(
            data.triplet_sort_keys,
            data.triplet_sort_keys_out,
            data.triplet_sort_perm,
            data.triplet_sort_perm_out,
            data.triplet_sort_scratch,
            data.triplet_sort_size,
            qd.u64,
            True,
            64,
            4,
        )
    else:
        dynamic_radix_sort(
            data.triplet_sorter,
            data.triplet_sort_keys,
            data.triplet_sort_keys_out,
            data.triplet_sort_perm,
            data.triplet_sort_perm_out,
            data.padded_contact_triplets[()],
        )


@qd.func(requires_top_level=True)
def triplet_segment_flags(data):
    qd.loop_config(name="contact_triplet_segment_flags")
    for index in range(data.triplet_sort_keys.shape[0]):
        if index < data.padded_contact_triplets[()]:
            flag = qd.i32(0)
            if index < data.n_contact_triplets[()] and (
                index == data.n_contact_triplets[()] - 1
                or data.triplet_sort_keys[index] != data.triplet_sort_keys[index + 1]
            ):
                flag = qd.i32(1)
            data.triplet_seg_flags[index] = flag


@qd.func(requires_top_level=True)
def triplet_scan(data, genesis_legacy_sort_reduce: qd.template()):
    if qd.static(genesis_legacy_sort_reduce):
        exclusive_scan_add(
            data.triplet_seg_flags,
            data.triplet_seg_ids,
            data.triplet_scan_scratch,
            data.n_contact_triplets[()],
            qd.i32,
            4,
        )
    else:
        dynamic_exclusive_sum(
            data.triplet_scanner,
            data.triplet_seg_flags,
            data.triplet_seg_ids,
            data.padded_contact_triplets[()],
        )


@qd.func(requires_top_level=True)
def triplet_zero_unique(data):
    qd.loop_config(name="contact_triplet_zero_unique")
    for index in range(data.unique_triplet_values.shape[0]):
        if index < data.padded_contact_triplets[()]:
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    data.unique_triplet_values[index, row, column] = 0.0


@qd.func(requires_top_level=True)
def triplet_fsr_merge(data, genesis_legacy_sort_reduce: qd.template()):
    if qd.static(genesis_legacy_sort_reduce):
        qd.loop_config(name="contact_triplet_fsr_merge_legacy")
        for index in range(data.n_contact_triplets[()]):
            source = qd.i32(data.triplet_sort_perm[index])
            segment = data.triplet_seg_ids[index]
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    qd.atomic_add(
                        data.unique_triplet_values[
                            segment,
                            row,
                            column,
                        ],
                        data.contact_triplet_values[
                            source,
                            row,
                            column,
                        ],
                    )
    else:
        fsr_reduce_triplet(
            data.triplet_seg_ids,
            data.triplet_sort_perm,
            data.triplet_sort_keys,
            data.contact_triplet_values,
            data.unique_triplet_values,
            data.n_contact_triplets,
            data.padded_contact_triplets,
            data.triplet_sort_keys.shape[0],
        )


@qd.func(requires_top_level=True)
def triplet_extract_unique(data):
    qd.loop_config(name="contact_triplet_extract_unique")
    for index in range(data.triplet_sort_keys.shape[0]):
        if (
            index < data.padded_contact_triplets[()]
            and index < data.n_contact_triplets[()]
            and data.triplet_seg_flags[index] != 0
        ):
            segment = qd.i32(data.triplet_seg_ids[index])
            key = data.triplet_sort_keys[index]
            data.unique_triplet_rows[segment] = qd.i32(key >> 32)
            data.unique_triplet_cols[segment] = qd.i32(key & qd.u64(0xFFFFFFFF))
            if index == data.n_contact_triplets[()] - 1:
                data.n_unique_triplets[()] = segment + 1


@qd.func(requires_top_level=True)
def sort_reduce(data, genesis_legacy_sort_reduce: qd.template()):
    doublet_sort_seed(data)
    doublet_sort_radix(data, genesis_legacy_sort_reduce)
    doublet_segment_flags(data)
    doublet_scan(data, genesis_legacy_sort_reduce)
    doublet_zero_unique(data)
    doublet_fsr_merge(data, genesis_legacy_sort_reduce)
    doublet_extract_unique(data)
    triplet_sort_seed(data)
    triplet_sort_radix(data, genesis_legacy_sort_reduce)
    triplet_segment_flags(data)
    triplet_scan(data, genesis_legacy_sort_reduce)
    triplet_zero_unique(data)
    triplet_fsr_merge(data, genesis_legacy_sort_reduce)
    triplet_extract_unique(data)
