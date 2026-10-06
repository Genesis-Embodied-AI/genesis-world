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
from .sim_system import SimData, SimSystem

_CONTACT_SORT_MIN_CAPACITY = 4_865
_CONTACT_SORT_LOG256_MAX_N = 4
_CCD_MAX_ITERS = 50_000
_CCD_ETA = 0.2


def _padded64(value: int) -> int:
    return max(((value + 63) // 64) * 64, 64)


def get_contact_assembly_capacity() -> int:
    return _CONTACT_SORT_MIN_CAPACITY


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
        ccd_max_iters: int
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
        doublet_scanner: object
        doublet_seg_flags: qd.Ndarray
        doublet_seg_ids: qd.Ndarray
        doublet_sort_keys: qd.Ndarray
        doublet_sort_keys_out: qd.Ndarray
        doublet_sort_perm: qd.Ndarray
        doublet_sort_perm_out: qd.Ndarray
        doublet_sort_scratch: qd.Ndarray
        doublet_sort_size: qd.Ndarray
        doublet_sorter: object
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
        genesis_legacy_sort_reduce_host: bool
        global_calm_frames: qd.Ndarray
        halfplane_contact_element_ids: qd.Ndarray
        halfplane_normals: qd.Ndarray
        halfplane_positions: qd.Ndarray
        has_codim: bool
        has_friction: bool
        has_halfplanes: bool
        init_pair_capacity: qd.Ndarray
        intersection_check: qd.Ndarray
        intersection_check_host: bool
        intersection_flag: qd.Ndarray
        is_initialized_host: bool
        is_wired_host: bool
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
        sort_log256_max_n: int
        triplet_scan_scratch: qd.Ndarray
        triplet_scanner: object
        triplet_seg_flags: qd.Ndarray
        triplet_seg_ids: qd.Ndarray
        triplet_sort_keys: qd.Ndarray
        triplet_sort_keys_out: qd.Ndarray
        triplet_sort_perm: qd.Ndarray
        triplet_sort_perm_out: qd.Ndarray
        triplet_sort_scratch: qd.Ndarray
        triplet_sort_size: qd.Ndarray
        triplet_sorter: object
        unique_doublet_gradients: qd.Ndarray
        unique_doublet_vertices: qd.Ndarray
        unique_triplet_cols: qd.Ndarray
        unique_triplet_rows: qd.Ndarray
        unique_triplet_values: qd.Ndarray
        vert_contact_element_ids: qd.Ndarray
        vertex_calm_streak: qd.Ndarray
        vertex_kappa_scale: qd.Ndarray
        vertex_min_gap: qd.Ndarray

    def __init__(self, data: Data) -> None:
        super().__init__()
        self.data = data
        self.actions: dict[str, object] = {}

    def build(self) -> None:
        from .global_body_manager import GlobalBodyManager
        from .global_linear_system import GlobalLinearSystem
        from .global_surface_manager import GlobalSurfaceManager
        from .global_vertex_manager import GlobalVertexManager
        from .lbvh_broad_phase import InfoLBVHBatchedBroadPhaseDop14, LBVHBroadPhase

        self.body_system = self.require(GlobalBodyManager)
        self.vertex_system = self.require(GlobalVertexManager)
        self.surface_system = self.require(GlobalSurfaceManager)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.broad_phase_system = self.find(InfoLBVHBatchedBroadPhaseDop14)
        if self.broad_phase_system is None:
            self.broad_phase_system = self.require(LBVHBroadPhase)

        data = self.data
        surface = self.surface_system.data
        vertex = self.vertex_system.data
        own_actions = {
            "reset_initial_intersections": (reset_initial_intersections, (data,)),
            "flag_et_intersections": (flag_et_intersections, (data,)),
            "reset_counted_demand": (reset_counted_demand, (data,)),
            "adaptive_kappa_update": (adaptive_kappa_update, (data,)),
            "adaptive_kappa_newton_tick": (adaptive_kappa_newton_tick, (data,)),
            "reset_collision_counts": (reset_collision_counts, (data,)),
            "halfplane_query": (halfplane_query, (data, surface, vertex)),
            "init_ccd": (init_ccd, (data,)),
            "reset_frame_ccd": (reset_frame_ccd, (data,)),
            "ccd_alpha_pt": (ccd_alpha_pt_kernel, (data, surface, vertex)),
            "ccd_alpha_ee": (ccd_alpha_ee_kernel, (data, surface, vertex)),
            "ccd_alpha_ph": (halfplane_ccd_alpha_kernel, (data, surface, vertex)),
            "reduce_ccd_alpha": (reduce_ccd_alpha_final_kernel, (data,)),
            "ccd": (ccd, (data, surface, vertex)),
            "reset_contact_energy": (reset_contact_energy, (data,)),
            "sum_contact_energy": (sum_contact_energy, (data,)),
            "check_assembly_capacity": (check_assembly_capacity, (data,)),
            "check_assembly_padding": (check_assembly_padding, (data,)),
            "shrink_assembly_padding": (shrink_assembly_padding, (data,)),
            "reset_assembly_counts": (reset_assembly_counts, (data,)),
            "sort_reduce": (sort_reduce, (data,)),
        }
        self.actions = {
            name: self.create_action(kernel, *action_data) for name, (kernel, action_data) in own_actions.items()
        }

    def resolve_actions(self) -> dict[str, object]:
        if self.is_building:
            raise RuntimeError("Contact actions are available only after build")
        return {name: action.invocation for name, action in self.actions.items()}

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
    if data.is_wired_host:
        raise RuntimeError("ContactSystem parameters are already wired")
    if d_hat <= 0.0:
        raise ValueError("contact/d_hat must be positive")
    if kappa <= 0.0:
        raise ValueError("ContactTabular resistance must be positive")
    if init_pair_capacity < 1:
        raise ValueError("contact/init_collision_pair_capacity must be at least one")
    if intersection_check_capacity < 1:
        raise ValueError("contact/intersection_check_capacity must be at least one")
    if bool(intersection_check) != data.intersection_check_host:
        raise ValueError("ContactSystem intersection_check must be fixed before build")

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
    data.is_wired_host = True


def set_contact_dt_sq(data, dt_sq: float) -> None:
    if dt_sq <= 0.0:
        raise ValueError("ContactSystem dt_sq must be positive")
    data.dt_sq.from_numpy(np.array(dt_sq, dtype=np.float64))


def _wire_contact_friction_params(data, *, mu: float, eps_v: float) -> None:
    if mu < 0.0:
        raise ValueError("ContactTabular friction_rate must be non-negative")
    if eps_v <= 0.0:
        raise ValueError("friction/eps_v must be positive")
    data.has_friction = mu > 0.0
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
    data.has_halfplanes = len(plane_positions) != 0


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
    if data.is_initialized_host:
        raise RuntimeError("ContactSystem is already initialized")
    if not data.is_wired_host:
        raise RuntimeError("ContactSystem.wire_params() must run before init()")
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
    data.is_initialized_host = True


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
        shape=(max(sort_scratch_slots(padded_doublets, data.sort_log256_max_n), 1),),
    )
    data.doublet_seg_flags = qd.ndarray(qd.i32, shape=(padded_doublets,))
    data.doublet_seg_ids = qd.ndarray(qd.i32, shape=(padded_doublets,))
    data.doublet_scanner = DynamicExclusiveSum(
        padded_doublets,
    )
    data.doublet_scan_scratch = qd.ndarray(
        qd.i32,
        shape=(max(exclusive_scan_scratch_slots(padded_doublets, data.sort_log256_max_n), 1),),
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
        shape=(max(sort_scratch_slots(padded_triplets, data.sort_log256_max_n), 1),),
    )
    data.triplet_seg_flags = qd.ndarray(qd.i32, shape=(padded_triplets,))
    data.triplet_seg_ids = qd.ndarray(qd.i32, shape=(padded_triplets,))
    data.triplet_scanner = DynamicExclusiveSum(
        padded_triplets,
    )
    data.triplet_scan_scratch = qd.ndarray(
        qd.i32,
        shape=(max(exclusive_scan_scratch_slots(padded_triplets, data.sort_log256_max_n), 1),),
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
            shape=(max(sort_scratch_slots(padded_doublets, data.sort_log256_max_n), 1),),
        )
        data.doublet_seg_flags = qd.ndarray(qd.i32, shape=(padded_doublets,))
        data.doublet_seg_ids = qd.ndarray(qd.i32, shape=(padded_doublets,))
        data.doublet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(max(exclusive_scan_scratch_slots(padded_doublets, data.sort_log256_max_n), 1),),
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
            shape=(max(sort_scratch_slots(padded_triplets, data.sort_log256_max_n), 1),),
        )
        data.triplet_seg_flags = qd.ndarray(qd.i32, shape=(padded_triplets,))
        data.triplet_seg_ids = qd.ndarray(qd.i32, shape=(padded_triplets,))
        data.triplet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(max(exclusive_scan_scratch_slots(padded_triplets, data.sort_log256_max_n), 1),),
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


def get_contact_system_data(
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
) -> ContactSystem.Data:
    """Construct complete contact data before system build and action registration."""
    if n_verts < 0:
        raise ValueError("ContactSystem n_verts must be non-negative")
    if n_bodies < 0:
        raise ValueError("ContactSystem n_bodies must be non-negative")

    data = ContactSystem.Data()
    data.is_wired_host = False
    data.is_initialized_host = False
    data.has_friction = False
    data.has_halfplanes = False
    data.has_codim = False
    data.sort_log256_max_n = _CONTACT_SORT_LOG256_MAX_N
    data.ccd_max_iters = _CCD_MAX_ITERS
    data.intersection_check_host = bool(intersection_check)
    data.genesis_legacy_sort_reduce_host = bool(genesis_legacy_sort_reduce)

    _wire_contact_params(
        data,
        d_hat=d_hat,
        kappa=kappa,
        init_pair_capacity=init_pair_capacity,
        intersection_check=intersection_check,
        intersection_check_capacity=intersection_check_capacity,
    )
    set_contact_dt_sq(data, dt_sq)
    _wire_contact_friction_params(data, mu=friction_mu, eps_v=friction_eps_v)
    _wire_contact_tabular(data, contact_tabular)
    _wire_contact_halfplanes(
        data,
        halfplane_positions,
        halfplane_normals,
        halfplane_contact_element_ids,
    )
    _set_contact_adaptive_kappa(
        data,
        adaptive_kappa_mode,
        adaptive_kappa_tick,
        n_bodies,
        n_verts,
    )
    _initialize_contact_data(data, n_verts)
    if contact_element_ids is not None:
        set_contact_element_ids(data, contact_element_ids)
    return data


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
            data.ccd_max_iters,
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
            data.ccd_max_iters,
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
            data.ccd_max_iters,
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
def doublet_sort_radix(data):
    if qd.static(data.genesis_legacy_sort_reduce_host):
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
            data.sort_log256_max_n,
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
def doublet_scan(data):
    if qd.static(data.genesis_legacy_sort_reduce_host):
        exclusive_scan_add(
            data.doublet_seg_flags,
            data.doublet_seg_ids,
            data.doublet_scan_scratch,
            data.n_contact_doublets[()],
            qd.i32,
            data.sort_log256_max_n,
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
def doublet_fsr_merge(data):
    if qd.static(data.genesis_legacy_sort_reduce_host):
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
def triplet_sort_radix(data):
    if qd.static(data.genesis_legacy_sort_reduce_host):
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
            data.sort_log256_max_n,
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
def triplet_scan(data):
    if qd.static(data.genesis_legacy_sort_reduce_host):
        exclusive_scan_add(
            data.triplet_seg_flags,
            data.triplet_seg_ids,
            data.triplet_scan_scratch,
            data.n_contact_triplets[()],
            qd.i32,
            data.sort_log256_max_n,
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
def triplet_fsr_merge(data):
    if qd.static(data.genesis_legacy_sort_reduce_host):
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
def sort_reduce(data):
    doublet_sort_seed(data)
    doublet_sort_radix(data)
    doublet_segment_flags(data)
    doublet_scan(data)
    doublet_zero_unique(data)
    doublet_fsr_merge(data)
    doublet_extract_unique(data)
    triplet_sort_seed(data)
    triplet_sort_radix(data)
    triplet_segment_flags(data)
    triplet_scan(data)
    triplet_zero_unique(data)
    triplet_fsr_merge(data)
    triplet_extract_unique(data)
