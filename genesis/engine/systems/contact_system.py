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
from .sim_system import SimSystem

_CONTACT_SORT_MIN_CAPACITY = 4_865
_CONTACT_SORT_LOG256_MAX_N = 4
_CCD_MAX_ITERS = 50_000
_CCD_ETA = 0.2


def _padded64(value: int) -> int:
    return max(((value + 63) // 64) * 64, 64)


@qd.data_oriented
class ContactSystem(SimSystem):
    """Own CGQ collision pairs, contact assembly, friction, energy, and CCD state."""

    def __init__(
        self,
        *,
        intersection_check: bool = False,
        genesis_legacy_sort_reduce: bool = False,
    ) -> None:
        super().__init__()
        self._contact_constitution = None
        self.is_wired_host = False
        self.is_initialized_host = False
        self.has_friction = False
        self.has_halfplanes = False
        self.has_codim = False
        self.sort_log256_max_n = _CONTACT_SORT_LOG256_MAX_N
        self.ccd_max_iters = _CCD_MAX_ITERS
        self.intersection_check_host = bool(intersection_check)
        self.genesis_legacy_sort_reduce_host = bool(genesis_legacy_sort_reduce)

    def do_build(self) -> None:
        from .global_body_manager import GlobalBodyManager
        from .global_linear_system import GlobalLinearSystem
        from .global_surface_manager import GlobalSurfaceManager
        from .global_vertex_manager import GlobalVertexManager
        from .lbvh_broad_phase import InfoLBVHBatchedBroadPhaseDop14, LBVHBroadPhase

        self.body = self.require(GlobalBodyManager)
        self.vertex = self.require(GlobalVertexManager)
        self.surface = self.require(GlobalSurfaceManager)
        self.global_linear_system = self.require(GlobalLinearSystem)
        self.broad_phase = self.find(InfoLBVHBatchedBroadPhaseDop14)
        if self.broad_phase is None:
            self.broad_phase = self.require(LBVHBroadPhase)

    def set_contact_constitution(self, constitution) -> None:
        if self._contact_constitution is not None:
            raise RuntimeError("ContactSystem already has a ContactConstitution")
        self._contact_constitution = constitution

    def wire_params(
        self,
        *,
        d_hat: float,
        kappa: float,
        init_pair_capacity: int,
        intersection_check: bool = False,
        intersection_check_capacity: int = 1_024,
    ) -> None:
        if self.is_wired_host:
            raise RuntimeError("ContactSystem parameters are already wired")
        if d_hat <= 0.0:
            raise ValueError("contact/d_hat must be positive")
        if kappa <= 0.0:
            raise ValueError("ContactTabular resistance must be positive")
        if init_pair_capacity < 1:
            raise ValueError("contact/init_collision_pair_capacity must be at least one")
        if intersection_check_capacity < 1:
            raise ValueError("contact/intersection_check_capacity must be at least one")
        if bool(intersection_check) != self.intersection_check_host:
            raise ValueError("ContactSystem intersection_check must be fixed before build")

        self.d_hat = qd.ndarray(qd.f64, shape=())
        self.kappa = qd.ndarray(qd.f64, shape=())
        self.init_pair_capacity = qd.ndarray(qd.i32, shape=())
        self.dt_sq = qd.ndarray(qd.f64, shape=())
        self.ccd_eta = qd.ndarray(qd.f64, shape=())
        self.max_step_in_d_hat = qd.ndarray(qd.f64, shape=())
        self.capacity_grow_factor = qd.ndarray(qd.f64, shape=())
        self.capacity_shrink_threshold = qd.ndarray(qd.f64, shape=())
        self.friction_eps_v = qd.ndarray(qd.f64, shape=())
        self.intersection_check = qd.ndarray(qd.i32, shape=())
        self.max_et_pairs = qd.ndarray(qd.i32, shape=())
        self.n_et_pairs = qd.ndarray(qd.i32, shape=())
        self.et_overflow_flag = qd.ndarray(qd.i32, shape=())
        self.et_yield_flag = qd.ndarray(qd.i32, shape=())
        self.et_pairs = qd.ndarray(
            qd.i32,
            shape=(intersection_check_capacity, 2),
        )

        self.d_hat.from_numpy(np.array(d_hat, dtype=np.float64))
        self.kappa.from_numpy(np.array(kappa, dtype=np.float64))
        self.init_pair_capacity.from_numpy(np.array(init_pair_capacity, dtype=np.int32))
        self.dt_sq.from_numpy(np.array(0.0, dtype=np.float64))
        self.ccd_eta.from_numpy(np.array(_CCD_ETA, dtype=np.float64))
        self.max_step_in_d_hat.from_numpy(
            np.array(CONTACT_CONFIG_DEFAULTS["contact/max_step_in_d_hat"], dtype=np.float64)
        )
        self.capacity_grow_factor.from_numpy(
            np.array(CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"], dtype=np.float64)
        )
        self.capacity_shrink_threshold.from_numpy(
            np.array(CONTACT_CONFIG_DEFAULTS["extras/capacity_shrink_threshold"], dtype=np.float64)
        )
        self.friction_eps_v.from_numpy(np.array(CONTACT_CONFIG_DEFAULTS["friction/eps_v"], dtype=np.float64))
        self.intersection_check.from_numpy(np.array(int(intersection_check), dtype=np.int32))
        self.max_et_pairs.from_numpy(np.array(intersection_check_capacity, dtype=np.int32))
        self.n_et_pairs.from_numpy(np.array(0, dtype=np.int32))
        self.et_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
        self.et_yield_flag.from_numpy(np.array(0, dtype=np.int32))
        self.et_pairs.from_numpy(np.zeros((intersection_check_capacity, 2), dtype=np.int32))
        for channel in ("pt", "ee", "pe", "pp", "ph"):
            setattr(self, f"pairs_{channel}", qd.ndarray(qd.i32, shape=(init_pair_capacity, 2)))
            setattr(self, f"ccd_alpha_{channel}", qd.ndarray(qd.f64, shape=(init_pair_capacity,)))
            setattr(self, f"n_pairs_{channel}", qd.ndarray(qd.i32, shape=()))
            setattr(self, f"max_pairs_{channel}", qd.ndarray(qd.i32, shape=()))
            getattr(self, f"n_pairs_{channel}").from_numpy(np.array(0, dtype=np.int32))
            getattr(self, f"max_pairs_{channel}").from_numpy(np.array(init_pair_capacity, dtype=np.int32))
            getattr(self, f"ccd_alpha_{channel}").from_numpy(np.ones(init_pair_capacity, dtype=np.float64))
        self.is_wired_host = True

    def set_dt_sq(self, dt_sq: float) -> None:
        if dt_sq <= 0.0:
            raise ValueError("ContactSystem dt_sq must be positive")
        self.dt_sq.from_numpy(np.array(dt_sq, dtype=np.float64))

    def wire_friction_params(self, *, mu: float, eps_v: float) -> None:
        if mu < 0.0:
            raise ValueError("ContactTabular friction_rate must be non-negative")
        if eps_v <= 0.0:
            raise ValueError("friction/eps_v must be positive")
        self.has_friction = mu > 0.0
        self.default_friction_rate = qd.ndarray(qd.f64, shape=())
        self.default_friction_rate.from_numpy(np.array(mu, dtype=np.float64))
        self.friction_eps_v.from_numpy(np.array(eps_v, dtype=np.float64))

    def wire_contact_tabular(self, tabular: ContactTabular) -> None:
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

        self.n_contact_elements = qd.ndarray(qd.i32, shape=())
        self.kappa_table = qd.ndarray(qd.f64, shape=(max(len(kappa_table), 1),))
        self.mu_table = qd.ndarray(qd.f64, shape=(max(len(mu_table), 1),))
        self.enable_table = qd.ndarray(qd.i32, shape=(max(len(enable_table), 1),))
        self.enable_ee_table = qd.ndarray(qd.i32, shape=(max(len(enable_ee_table), 1),))
        self.n_contact_elements.from_numpy(np.array(n_elements, dtype=np.int32))
        self.kappa_table.from_numpy(kappa_table)
        self.mu_table.from_numpy(mu_table)
        self.enable_table.from_numpy(enable_table)
        self.enable_ee_table.from_numpy(enable_ee_table)

    def wire_contact_element_ids(self, contact_element_ids: np.ndarray) -> None:
        values = np.ascontiguousarray(contact_element_ids, dtype=np.int32).reshape(-1)
        if len(values) != self.vert_contact_element_ids.shape[0]:
            raise ValueError("ContactSystem contact element IDs must match n_verts")
        if np.any(values < 0):
            raise ValueError("ContactSystem contact element IDs must be non-negative")
        self.vert_contact_element_ids.from_numpy(values)

    def wire_halfplanes(
        self,
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

        self.n_halfplanes = qd.ndarray(qd.i32, shape=())
        self.halfplane_positions = qd.ndarray(qd.f64, shape=(max(len(plane_positions), 1), 3))
        self.halfplane_normals = qd.ndarray(qd.f64, shape=(max(len(plane_normals), 1), 3))
        self.halfplane_contact_element_ids = qd.ndarray(
            qd.i32,
            shape=(max(len(plane_positions), 1),),
        )
        self.n_halfplanes.from_numpy(np.array(len(plane_positions), dtype=np.int32))
        self.halfplane_positions.from_numpy(
            plane_positions if len(plane_positions) else np.zeros((1, 3), dtype=np.float64)
        )
        self.halfplane_normals.from_numpy(plane_normals if len(plane_normals) else np.zeros((1, 3), dtype=np.float64))
        self.halfplane_contact_element_ids.from_numpy(
            plane_contact_element_ids if len(plane_contact_element_ids) else np.zeros(1, dtype=np.int32)
        )
        self.has_halfplanes = len(plane_positions) != 0

    def set_adaptive_kappa(self, mode: str, tick: str, n_bodies: int) -> None:
        mode_values = {"off": 0, "global": 1, "per-vertex": 2, "per-body": 3}
        tick_values = {"frame": 0, "newton": 1}
        if mode not in mode_values:
            raise ValueError(f"Unsupported contact/adaptive_kappa_mode {mode!r}")
        if tick not in tick_values:
            raise ValueError(f"Unsupported contact/adaptive_kappa_tick {tick!r}")
        if mode == "per-vertex" and tick == "newton":
            raise ValueError("CGQ does not support per-vertex adaptive kappa with the Newton tick")

        self.adaptive_kappa_mode = qd.ndarray(qd.i32, shape=())
        self.adaptive_kappa_tick = qd.ndarray(qd.i32, shape=())
        self.contact_kappa_scale = qd.ndarray(qd.f64, shape=())
        self.adaptive_gap_ratio = qd.ndarray(qd.f64, shape=())
        self.adaptive_grow = qd.ndarray(qd.f64, shape=())
        self.adaptive_max_scale = qd.ndarray(qd.f64, shape=())
        self.adaptive_calm_time = qd.ndarray(qd.f64, shape=())
        self.adaptive_relax_time = qd.ndarray(qd.f64, shape=())
        self.adaptive_hysteresis = qd.ndarray(qd.f64, shape=())
        self.global_calm_frames = qd.ndarray(qd.i32, shape=())
        self.body_kappa_scale = qd.ndarray(qd.f64, shape=(max(n_bodies, 1),))
        self.body_min_gap = qd.ndarray(qd.f64, shape=(max(n_bodies, 1),))
        self.iter_body_min_gap = qd.ndarray(qd.f64, shape=(max(n_bodies, 1),))
        self.body_calm_streak = qd.ndarray(qd.i32, shape=(max(n_bodies, 1),))
        self.vertex_kappa_scale = qd.ndarray(qd.f64, shape=(max(self.vertex.positions.shape[0], 1),))
        self.vertex_min_gap = qd.ndarray(qd.f64, shape=(max(self.vertex.positions.shape[0], 1),))
        self.vertex_calm_streak = qd.ndarray(qd.i32, shape=(max(self.vertex.positions.shape[0], 1),))
        self.adaptive_kappa_mode.from_numpy(np.array(mode_values[mode], dtype=np.int32))
        self.adaptive_kappa_tick.from_numpy(np.array(tick_values[tick], dtype=np.int32))
        self.contact_kappa_scale.from_numpy(np.array(1.0, dtype=np.float64))
        self.adaptive_gap_ratio.from_numpy(np.array(0.01, dtype=np.float64))
        self.adaptive_grow.from_numpy(np.array(2.0, dtype=np.float64))
        self.adaptive_max_scale.from_numpy(np.array(128.0, dtype=np.float64))
        self.adaptive_calm_time.from_numpy(np.array(0.3, dtype=np.float64))
        self.adaptive_relax_time.from_numpy(np.array(0.4, dtype=np.float64))
        self.adaptive_hysteresis.from_numpy(np.array(2.0, dtype=np.float64))
        self.global_calm_frames.from_numpy(np.array(0, dtype=np.int32))
        self.body_kappa_scale.from_numpy(np.ones(max(n_bodies, 1), dtype=np.float64))
        self.body_min_gap.from_numpy(np.full(max(n_bodies, 1), 1e300, dtype=np.float64))
        self.iter_body_min_gap.from_numpy(np.full(max(n_bodies, 1), 1e300, dtype=np.float64))
        self.body_calm_streak.from_numpy(np.zeros(max(n_bodies, 1), dtype=np.int32))
        self.vertex_kappa_scale.from_numpy(np.ones(max(self.vertex.positions.shape[0], 1), dtype=np.float64))
        self.vertex_min_gap.from_numpy(np.full(max(self.vertex.positions.shape[0], 1), 1e300, dtype=np.float64))
        self.vertex_calm_streak.from_numpy(np.zeros(max(self.vertex.positions.shape[0], 1), dtype=np.int32))

    def init(self, n_verts: int) -> None:
        if self.is_initialized_host:
            raise RuntimeError("ContactSystem is already initialized")
        if not self.is_wired_host:
            raise RuntimeError("ContactSystem.wire_params() must run before init()")
        if self._contact_constitution is None:
            raise RuntimeError("ContactSystem requires a ContactConstitution")

        pair_capacity = self.pairs_pt.shape[0]
        doublet_capacity = _CONTACT_SORT_MIN_CAPACITY
        triplet_capacity = _CONTACT_SORT_MIN_CAPACITY

        self.n_verts = qd.ndarray(qd.i32, shape=())
        self.n_verts.from_numpy(np.array(n_verts, dtype=np.int32))
        self.vert_contact_element_ids = qd.ndarray(qd.i32, shape=(max(n_verts, 1),))
        self.vert_contact_element_ids.from_numpy(np.zeros(max(n_verts, 1), dtype=np.int32))

        self._allocate_assembly_buffers(doublet_capacity, triplet_capacity)
        self._allocate_friction_buffers(pair_capacity)

        self.n_counted_doublets = qd.ndarray(qd.i32, shape=())
        self.n_counted_triplets = qd.ndarray(qd.i32, shape=())
        self.n_friction_demand_doublets = qd.ndarray(qd.i32, shape=())
        self.n_friction_demand_triplets = qd.ndarray(qd.i32, shape=())
        self.n_active_pairs = qd.ndarray(qd.i32, shape=())
        self.overflow_flag = qd.ndarray(qd.i32, shape=())
        self.count_overflow_flag = qd.ndarray(qd.i32, shape=())
        self.contact_padding_overflow = qd.ndarray(qd.i32, shape=())
        self.intersection_flag = qd.ndarray(qd.i32, shape=())
        self.friction_overflow_flag = qd.ndarray(qd.i32, shape=())
        self.barrier_energy = qd.ndarray(qd.f64, shape=())
        self.friction_energy = qd.ndarray(qd.f64, shape=())
        self.contact_energy_value = qd.ndarray(qd.f64, shape=())
        self.ccd_alpha = qd.ndarray(qd.f64, shape=())
        self.frame_ccd_alpha = qd.ndarray(qd.f64, shape=())
        self.max_accd_iters = qd.ndarray(qd.i32, shape=())
        self.min_gap_ratio = qd.ndarray(qd.f64, shape=())
        self.iter_min_gap_ratio = qd.ndarray(qd.f64, shape=())
        self.adaptive_kappa_grew = qd.ndarray(qd.i32, shape=())

        for scalar in (
            self.n_counted_doublets,
            self.n_counted_triplets,
            self.n_friction_demand_doublets,
            self.n_friction_demand_triplets,
            self.n_active_pairs,
            self.overflow_flag,
            self.count_overflow_flag,
            self.contact_padding_overflow,
            self.intersection_flag,
            self.friction_overflow_flag,
            self.max_accd_iters,
            self.adaptive_kappa_grew,
        ):
            scalar.from_numpy(np.array(0, dtype=np.int32))
        self.barrier_energy.from_numpy(np.array(0.0, dtype=np.float64))
        self.friction_energy.from_numpy(np.array(0.0, dtype=np.float64))
        self.contact_energy_value.from_numpy(np.array(0.0, dtype=np.float64))
        self.ccd_alpha.from_numpy(np.array(1.0, dtype=np.float64))
        self.frame_ccd_alpha.from_numpy(np.array(1.0, dtype=np.float64))
        self.min_gap_ratio.from_numpy(np.array(1e300, dtype=np.float64))
        self.iter_min_gap_ratio.from_numpy(np.array(1e300, dtype=np.float64))
        self.contact_constitution = self._contact_constitution
        del self._contact_constitution
        self.is_initialized_host = True

    def _allocate_assembly_buffers(self, doublet_capacity: int, triplet_capacity: int) -> None:
        padded_doublets = _padded64(doublet_capacity)
        padded_triplets = _padded64(triplet_capacity)
        self.max_contact_doublets = qd.ndarray(qd.i32, shape=())
        self.max_contact_triplets = qd.ndarray(qd.i32, shape=())
        self.padded_contact_doublets = qd.ndarray(qd.i32, shape=())
        self.padded_contact_triplets = qd.ndarray(qd.i32, shape=())
        self.n_contact_doublets = qd.ndarray(qd.i32, shape=())
        self.n_contact_triplets = qd.ndarray(qd.i32, shape=())
        self.n_unique_doublets = qd.ndarray(qd.i32, shape=())
        self.n_unique_triplets = qd.ndarray(qd.i32, shape=())

        self.max_contact_doublets.from_numpy(np.array(doublet_capacity, dtype=np.int32))
        self.max_contact_triplets.from_numpy(np.array(triplet_capacity, dtype=np.int32))
        self.padded_contact_doublets.from_numpy(np.array(doublet_capacity, dtype=np.int32))
        self.padded_contact_triplets.from_numpy(np.array(triplet_capacity, dtype=np.int32))
        for scalar in (
            self.n_contact_doublets,
            self.n_contact_triplets,
            self.n_unique_doublets,
            self.n_unique_triplets,
        ):
            scalar.from_numpy(np.array(0, dtype=np.int32))

        self.contact_doublet_vertices = qd.ndarray(qd.i32, shape=(doublet_capacity,))
        self.contact_doublet_gradients = qd.ndarray(qd.f64, shape=(doublet_capacity, 3))
        self.contact_triplet_rows = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.contact_triplet_cols = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.contact_triplet_values = qd.ndarray(qd.f64, shape=(triplet_capacity, 3, 3))
        self.unique_doublet_vertices = qd.ndarray(qd.i32, shape=(doublet_capacity,))
        self.unique_doublet_gradients = qd.ndarray(qd.f64, shape=(doublet_capacity, 3))
        self.unique_triplet_rows = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.unique_triplet_cols = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.unique_triplet_values = qd.ndarray(qd.f64, shape=(triplet_capacity, 3, 3))

        self.doublet_sort_keys = qd.ndarray(qd.u32, shape=(padded_doublets,))
        self.doublet_sort_keys_out = qd.ndarray(qd.u32, shape=(padded_doublets,))
        self.doublet_sort_perm = qd.ndarray(qd.i32, shape=(padded_doublets,))
        self.doublet_sort_perm_out = qd.ndarray(qd.i32, shape=(padded_doublets,))
        self.doublet_sort_size = qd.ndarray(qd.i32, shape=())
        self.doublet_sorter = DynamicRadixSort(
            qd.u32,
            padded_doublets,
        )
        self.doublet_sort_scratch = qd.ndarray(
            qd.u32,
            shape=(max(sort_scratch_slots(padded_doublets, self.sort_log256_max_n), 1),),
        )
        self.doublet_seg_flags = qd.ndarray(qd.i32, shape=(padded_doublets,))
        self.doublet_seg_ids = qd.ndarray(qd.i32, shape=(padded_doublets,))
        self.doublet_scanner = DynamicExclusiveSum(
            padded_doublets,
        )
        self.doublet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(max(exclusive_scan_scratch_slots(padded_doublets, self.sort_log256_max_n), 1),),
        )

        self.triplet_sort_keys = qd.ndarray(qd.u64, shape=(padded_triplets,))
        self.triplet_sort_keys_out = qd.ndarray(qd.u64, shape=(padded_triplets,))
        self.triplet_sort_perm = qd.ndarray(qd.i32, shape=(padded_triplets,))
        self.triplet_sort_perm_out = qd.ndarray(qd.i32, shape=(padded_triplets,))
        self.triplet_sort_size = qd.ndarray(qd.i32, shape=())
        self.triplet_sorter = DynamicRadixSort(
            qd.u64,
            padded_triplets,
        )
        self.triplet_sort_scratch = qd.ndarray(
            qd.u32,
            shape=(max(sort_scratch_slots(padded_triplets, self.sort_log256_max_n), 1),),
        )
        self.triplet_seg_flags = qd.ndarray(qd.i32, shape=(padded_triplets,))
        self.triplet_seg_ids = qd.ndarray(qd.i32, shape=(padded_triplets,))
        self.triplet_scanner = DynamicExclusiveSum(
            padded_triplets,
        )
        self.triplet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(max(exclusive_scan_scratch_slots(padded_triplets, self.sort_log256_max_n), 1),),
        )
        self.doublet_sort_size.from_numpy(np.array(0, dtype=np.int32))
        self.triplet_sort_size.from_numpy(np.array(0, dtype=np.int32))

    def _allocate_friction_buffers(self, pair_capacity: int) -> None:
        for channel in ("pt", "ee", "pe", "pp", "ph"):
            setattr(self, f"friction_pairs_{channel}", qd.ndarray(qd.i32, shape=(pair_capacity, 2)))
            setattr(self, f"friction_flags_{channel}", qd.ndarray(qd.i32, shape=(pair_capacity,)))
            setattr(self, f"n_friction_pairs_{channel}", qd.ndarray(qd.i32, shape=()))
            setattr(self, f"max_friction_pairs_{channel}", qd.ndarray(qd.i32, shape=()))
            getattr(self, f"n_friction_pairs_{channel}").from_numpy(np.array(0, dtype=np.int32))
            getattr(self, f"max_friction_pairs_{channel}").from_numpy(np.array(pair_capacity, dtype=np.int32))

        self.lagged_positions = qd.ndarray(qd.f64, shape=self.vertex.positions.shape)
        self.friction_pair_reach_scale = qd.ndarray(qd.f64, shape=(pair_capacity * 5,))

    def _grown_capacity(self, required: int, current: int) -> int:
        grow_factor = CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"]
        return max(math.ceil(required * grow_factor), current + 1)

    def realloc_pair_buffers(
        self,
        *,
        pt: int,
        ee: int,
        pe: int,
        pp: int,
        ph: int,
    ) -> None:
        for channel, required in (("pt", pt), ("ee", ee), ("pe", pe), ("pp", pp), ("ph", ph)):
            current = getattr(self, f"pairs_{channel}").shape[0]
            if required <= current:
                continue
            capacity = self._grown_capacity(required, current)
            setattr(self, f"pairs_{channel}", qd.ndarray(qd.i32, shape=(capacity, 2)))
            setattr(self, f"ccd_alpha_{channel}", qd.ndarray(qd.f64, shape=(capacity,)))
            getattr(self, f"ccd_alpha_{channel}").from_numpy(np.ones(capacity, dtype=np.float64))
            getattr(self, f"max_pairs_{channel}").from_numpy(np.array(capacity, dtype=np.int32))
        self.overflow_flag.from_numpy(np.array(0, dtype=np.int32))

    def realloc_et_pairs(self, required: int) -> None:
        current = self.et_pairs.shape[0]
        if required <= current:
            return
        capacity = self._grown_capacity(required, current)
        self.et_pairs = qd.ndarray(qd.i32, shape=(capacity, 2))
        self.et_pairs.from_numpy(np.zeros((capacity, 2), dtype=np.int32))
        self.max_et_pairs.from_numpy(np.array(capacity, dtype=np.int32))
        self.n_et_pairs.from_numpy(np.array(0, dtype=np.int32))
        self.et_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
        self.et_yield_flag.from_numpy(np.array(0, dtype=np.int32))

    def handle_broad_phase_overflow(self) -> bool:
        return self.broad_phase.handle_ee_query_overflow()

    def set_assembly_padding(
        self,
        padded_doublets: int,
        padded_triplets: int,
    ) -> None:
        doublet_capacity = self.contact_doublet_vertices.shape[0]
        triplet_capacity = self.contact_triplet_rows.shape[0]
        if not 0 <= padded_doublets <= doublet_capacity:
            raise ValueError("Contact doublet padding must fit its allocation")
        if not 0 <= padded_triplets <= triplet_capacity:
            raise ValueError("Contact triplet padding must fit its allocation")
        self.padded_contact_doublets.from_numpy(np.array(padded_doublets, dtype=np.int32))
        self.padded_contact_triplets.from_numpy(np.array(padded_triplets, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def reset_initial_intersections(self):
        for _ in range(1):
            self.n_et_pairs[()] = 0
            self.et_overflow_flag[()] = 0
            self.et_yield_flag[()] = 0

    @qd.func(requires_top_level=True)
    def detect_initial_intersections(self):
        if qd.static(self.broad_phase.has_triangle_bvh):
            self.broad_phase.triangle_bvh.query_et(
                self.surface,
                self.vertex,
                self.body,
                self,
                self.et_pairs,
                self.n_et_pairs,
                self.max_et_pairs[()],
                self.et_overflow_flag,
            )

    @qd.func(requires_top_level=True)
    def flag_et_intersections(self):
        for _ in range(1):
            self.et_yield_flag[()] = qd.i32(self.n_et_pairs[()] > 0)

    def realloc_assembly_buffers(self, required_doublets: int, required_triplets: int) -> None:
        current_doublets = self.contact_doublet_vertices.shape[0]
        current_triplets = self.contact_triplet_rows.shape[0]
        if required_doublets <= current_doublets and required_triplets <= current_triplets:
            self.max_contact_doublets.from_numpy(np.array(current_doublets, dtype=np.int32))
            self.max_contact_triplets.from_numpy(np.array(current_triplets, dtype=np.int32))
            self.count_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
            return
        doublet_capacity = (
            self._grown_capacity(required_doublets, current_doublets)
            if required_doublets > current_doublets
            else current_doublets
        )
        triplet_capacity = (
            self._grown_capacity(required_triplets, current_triplets)
            if required_triplets > current_triplets
            else current_triplets
        )
        self._allocate_assembly_buffers(doublet_capacity, triplet_capacity)
        self.count_overflow_flag.from_numpy(np.array(0, dtype=np.int32))

    def realloc_friction_pair_buffers(self, required: dict[str, int]) -> None:
        for channel in ("pt", "ee", "pe", "pp", "ph"):
            current = getattr(self, f"friction_pairs_{channel}").shape[0]
            demand = required.get(channel, 0)
            if demand <= current:
                continue
            capacity = self._grown_capacity(demand, current)
            setattr(self, f"friction_pairs_{channel}", qd.ndarray(qd.i32, shape=(capacity, 2)))
            setattr(self, f"friction_flags_{channel}", qd.ndarray(qd.i32, shape=(capacity,)))
            getattr(self, f"max_friction_pairs_{channel}").from_numpy(np.array(capacity, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def reset_counted_demand(self):
        for _ in range(1):
            self.n_counted_doublets[()] = 0
            self.n_counted_triplets[()] = 0
            self.count_overflow_flag[()] = 0

    @qd.func(requires_top_level=True)
    def friction_snapshot(self):
        self.contact_constitution.friction_snapshot(self, self.surface, self.vertex)

    @qd.func(requires_top_level=True)
    def count_active(self):
        self.reset_counted_demand()
        self.contact_constitution.count_active(self, self.surface, self.vertex)
        self.check_assembly_capacity()

    @qd.func(requires_top_level=True)
    def filter_assemble(self):
        self.vertex.zero_in_contact()
        self.reset_assembly_counts()
        self.contact_constitution.filter_assemble(self, self.surface, self.vertex)

    @qd.func(requires_top_level=True)
    def contact_energy(self):
        self.reset_contact_energy()
        self.contact_constitution.contact_energy(self, self.surface, self.vertex)
        self.sum_contact_energy()

    @qd.func(requires_top_level=True)
    def bvh_triangle_build(self):
        self.broad_phase.triangle_build()

    @qd.func(requires_top_level=True)
    def bvh_edge_build(self):
        self.broad_phase.edge_build()

    @qd.func(requires_top_level=True)
    def trajectory_query(self):
        self.broad_phase.trajectory_query()
        if qd.static(self.has_halfplanes):
            self.halfplane_query()

    @qd.func
    def adaptive_kappa_step(self, scale, gap, calm):
        next_scale = scale
        next_calm = calm
        if gap < self.adaptive_gap_ratio[()]:
            next_calm = 0
            next_scale = qd.min(scale * self.adaptive_grow[()], self.adaptive_max_scale[()])
        elif gap > self.adaptive_hysteresis[()] * self.adaptive_gap_ratio[()]:
            dt = qd.sqrt(self.dt_sq[()])
            hold = qd.i32(qd.ceil(self.adaptive_calm_time[()] / dt))
            if next_calm < qd.max(hold, 1):
                next_calm = next_calm + 1
            else:
                next_scale = qd.max(scale * qd.exp(-dt / self.adaptive_relax_time[()]), 1.0)
        return qd.Vector([next_scale, qd.f64(next_calm)])

    @qd.func
    def adaptive_kappa_frame_step(self, scale, gap, calm):
        result = qd.Vector([scale, qd.f64(calm)])
        if self.adaptive_kappa_tick[()] == 1 and gap < self.adaptive_gap_ratio[()]:
            result[1] = 0.0
        else:
            result = self.adaptive_kappa_step(scale, gap, calm)
        return result

    @qd.func(requires_top_level=True)
    def adaptive_kappa_update(self):
        for _ in range(1):
            mode = self.adaptive_kappa_mode[()]
            if mode == 1:
                result = self.adaptive_kappa_frame_step(
                    self.contact_kappa_scale[()],
                    self.min_gap_ratio[()],
                    self.global_calm_frames[()],
                )
                self.contact_kappa_scale[()] = result[0]
                self.global_calm_frames[()] = qd.i32(result[1])
                self.min_gap_ratio[()] = qd.f64(1e300)
            self.adaptive_kappa_grew[()] = 0
        for body in range(self.body_kappa_scale.shape[0]):
            if self.adaptive_kappa_mode[()] == 3:
                result = self.adaptive_kappa_frame_step(
                    self.body_kappa_scale[body],
                    self.body_min_gap[body],
                    self.body_calm_streak[body],
                )
                self.body_kappa_scale[body] = result[0]
                self.body_calm_streak[body] = qd.i32(result[1])
                self.body_min_gap[body] = qd.f64(1e300)
        for vertex in range(self.vertex_kappa_scale.shape[0]):
            if self.adaptive_kappa_mode[()] == 2:
                result = self.adaptive_kappa_step(
                    self.vertex_kappa_scale[vertex],
                    self.vertex_min_gap[vertex],
                    self.vertex_calm_streak[vertex],
                )
                self.vertex_kappa_scale[vertex] = result[0]
                self.vertex_calm_streak[vertex] = qd.i32(result[1])
                self.vertex_min_gap[vertex] = qd.f64(1e300)

    @qd.func(requires_top_level=True)
    def adaptive_kappa_newton_tick(self):
        for _ in range(1):
            self.adaptive_kappa_grew[()] = 0
            if self.adaptive_kappa_tick[()] == 1 and self.adaptive_kappa_mode[()] == 1:
                if self.iter_min_gap_ratio[()] < self.adaptive_gap_ratio[()]:
                    old_scale = self.contact_kappa_scale[()]
                    new_scale = qd.min(old_scale * self.adaptive_grow[()], self.adaptive_max_scale[()])
                    self.contact_kappa_scale[()] = new_scale
                    self.adaptive_kappa_grew[()] = qd.i32(new_scale > old_scale)
                self.iter_min_gap_ratio[()] = qd.f64(1e300)
        for body in range(self.body_kappa_scale.shape[0]):
            if self.adaptive_kappa_tick[()] == 1 and self.adaptive_kappa_mode[()] == 3:
                if self.iter_body_min_gap[body] < self.adaptive_gap_ratio[()]:
                    old_scale = self.body_kappa_scale[body]
                    new_scale = qd.min(old_scale * self.adaptive_grow[()], self.adaptive_max_scale[()])
                    self.body_kappa_scale[body] = new_scale
                    if new_scale > old_scale:
                        self.adaptive_kappa_grew[()] = 1
                self.iter_body_min_gap[body] = qd.f64(1e300)

    @qd.func(requires_top_level=True)
    def reset_collision_counts(self):
        for _ in range(1):
            self.n_pairs_pt[()] = 0
            self.n_pairs_ee[()] = 0
            self.n_pairs_pe[()] = 0
            self.n_pairs_pp[()] = 0
            self.n_pairs_ph[()] = 0
            self.overflow_flag[()] = 0
            self.intersection_flag[()] = 0

    @qd.func(requires_top_level=True)
    def halfplane_query(self):
        for pair_index in range(self.surface.n_surf_verts[()] * self.n_halfplanes[()]):
            surface_vertex = pair_index // self.n_halfplanes[()]
            plane = pair_index - surface_vertex * self.n_halfplanes[()]
            vertex_id = self.surface.surf_verts[surface_vertex]
            vertex_element = self.vert_contact_element_ids[vertex_id]
            plane_element = self.halfplane_contact_element_ids[plane]
            if self.enable_table[vertex_element * self.n_contact_elements[()] + plane_element] == 0:
                continue
            current_distance = halfplane_signed_distance(
                self.vertex.positions[vertex_id, 0],
                self.vertex.positions[vertex_id, 1],
                self.vertex.positions[vertex_id, 2],
                self.halfplane_positions[plane, 0],
                self.halfplane_positions[plane, 1],
                self.halfplane_positions[plane, 2],
                self.halfplane_normals[plane, 0],
                self.halfplane_normals[plane, 1],
                self.halfplane_normals[plane, 2],
            )
            endpoint_distance = halfplane_signed_distance(
                self.vertex.trajectory_end_positions[vertex_id, 0],
                self.vertex.trajectory_end_positions[vertex_id, 1],
                self.vertex.trajectory_end_positions[vertex_id, 2],
                self.halfplane_positions[plane, 0],
                self.halfplane_positions[plane, 1],
                self.halfplane_positions[plane, 2],
                self.halfplane_normals[plane, 0],
                self.halfplane_normals[plane, 1],
                self.halfplane_normals[plane, 2],
            )
            d_hat = pair_d_hat_ph(self.vertex.d_hats, self.d_hat[()], vertex_id)
            xi = pair_thickness_ph(self.vertex.thicknesses, vertex_id)
            if current_distance <= xi:
                self.intersection_flag[()] = 1
            elif qd.min(current_distance, endpoint_distance) < d_hat + xi:
                output = qd.atomic_add(self.n_pairs_ph[()], 1)
                if output < self.max_pairs_ph[()]:
                    self.pairs_ph[output, 0] = surface_vertex
                    self.pairs_ph[output, 1] = plane
                else:
                    self.overflow_flag[()] = 1

    @qd.func(requires_top_level=True)
    def init_ccd(self):
        for _ in range(1):
            self.ccd_alpha[()] = 1.0
            self.max_accd_iters[()] = 0

    @qd.func(requires_top_level=True)
    def reset_frame_ccd(self):
        for _ in range(1):
            self.frame_ccd_alpha[()] = 1.0

    @qd.func(requires_top_level=True)
    def ccd_alpha_pt_kernel(self):
        for pair_index in range(self.n_pairs_pt[()]):
            surface_vertex = self.pairs_pt[pair_index, 0]
            face = self.pairs_pt[pair_index, 1]
            ids = qd.Vector(
                [
                    self.surface.surf_verts[surface_vertex],
                    self.surface.surf_triangles[face, 0],
                    self.surface.surf_triangles[face, 1],
                    self.surface.surf_triangles[face, 2],
                ]
            )
            result = qd.Vector.zero(qd.f64, 1)
            screw_point_triangle_ccd(
                self.vertex,
                ids[0],
                ids[1],
                ids[2],
                ids[3],
                self.ccd_eta[()],
                pair_thickness_pt(self.vertex.thicknesses, ids[0], ids[1], ids[2], ids[3]),
                self.ccd_max_iters,
                result,
            )
            self.ccd_alpha_pt[pair_index] = result[0]
            qd.atomic_min(self.ccd_alpha[()], result[0])

    @qd.func(requires_top_level=True)
    def ccd_alpha_ee_kernel(self):
        for pair_index in range(self.n_pairs_ee[()]):
            edge_a = self.pairs_ee[pair_index, 0]
            edge_b = self.pairs_ee[pair_index, 1]
            ids = qd.Vector(
                [
                    self.surface.surf_edges[edge_a, 0],
                    self.surface.surf_edges[edge_a, 1],
                    self.surface.surf_edges[edge_b, 0],
                    self.surface.surf_edges[edge_b, 1],
                ]
            )
            result = qd.Vector.zero(qd.f64, 1)
            screw_edge_edge_ccd(
                self.vertex,
                ids[0],
                ids[1],
                ids[2],
                ids[3],
                self.ccd_eta[()],
                pair_thickness_ee(self.vertex.thicknesses, ids[0], ids[1], ids[2], ids[3]),
                self.ccd_max_iters,
                result,
            )
            self.ccd_alpha_ee[pair_index] = result[0]
            qd.atomic_min(self.ccd_alpha[()], result[0])

    @qd.func(requires_top_level=True)
    def halfplane_ccd_alpha_kernel(self):
        for pair_index in range(self.n_pairs_ph[()]):
            surface_vertex = self.pairs_ph[pair_index, 0]
            plane = self.pairs_ph[pair_index, 1]
            vertex_id = self.surface.surf_verts[surface_vertex]
            result = qd.Vector.zero(qd.f64, 1)
            normal = qd.Vector(
                [
                    self.halfplane_normals[plane, 0],
                    self.halfplane_normals[plane, 1],
                    self.halfplane_normals[plane, 2],
                ]
            )
            plane_position = qd.Vector(
                [
                    self.halfplane_positions[plane, 0],
                    self.halfplane_positions[plane, 1],
                    self.halfplane_positions[plane, 2],
                ]
            )
            screw_halfplane_ccd(
                self.vertex,
                vertex_id,
                normal,
                normal.dot(plane_position),
                self.ccd_eta[()],
                pair_thickness_ph(self.vertex.thicknesses, vertex_id),
                self.ccd_max_iters,
                result,
            )
            self.ccd_alpha_ph[pair_index] = result[0]
            qd.atomic_min(self.ccd_alpha[()], result[0])

    @qd.func(requires_top_level=True)
    def reduce_ccd_alpha_final_kernel(self):
        for _ in range(1):
            self.frame_ccd_alpha[()] = qd.min(
                self.frame_ccd_alpha[()],
                self.ccd_alpha[()],
            )

    @qd.func(requires_top_level=True)
    def ccd(self):
        self.ccd_alpha_pt_kernel()
        self.ccd_alpha_ee_kernel()
        self.halfplane_ccd_alpha_kernel()
        self.reduce_ccd_alpha_final_kernel()

    @qd.func(requires_top_level=True)
    def reset_contact_energy(self):
        for _ in range(1):
            self.barrier_energy[()] = 0.0
            self.friction_energy[()] = 0.0
            self.contact_energy_value[()] = 0.0

    @qd.func(requires_top_level=True)
    def sum_contact_energy(self):
        for _ in range(1):
            self.contact_energy_value[()] = self.barrier_energy[()] + self.friction_energy[()]

    @qd.func(requires_top_level=True)
    def check_assembly_capacity(self):
        for _ in range(1):
            required_doublets = self.n_counted_doublets[()] + self.n_friction_demand_doublets[()]
            required_triplets = self.n_counted_triplets[()] + self.n_friction_demand_triplets[()]
            overflow = (
                required_doublets > self.max_contact_doublets[()] or required_triplets > self.max_contact_triplets[()]
            )
            self.count_overflow_flag[()] = qd.i32(overflow)

    @qd.func(requires_top_level=True)
    def check_assembly_padding(self):
        for _ in range(1):
            self.contact_padding_overflow[()] = qd.i32(
                self.n_contact_doublets[()] > self.padded_contact_doublets[()]
                or self.n_contact_triplets[()] > self.padded_contact_triplets[()]
            )

    @qd.func(requires_top_level=True)
    def shrink_assembly_padding(self):
        for _ in range(1):
            n_doublets = self.n_contact_doublets[()]
            n_triplets = self.n_contact_triplets[()]
            if n_doublets > 0 or n_triplets > 0:
                target_doublets = qd.i32(qd.ceil(qd.f64(n_doublets) * self.capacity_grow_factor[()]))
                target_triplets = qd.i32(qd.ceil(qd.f64(n_triplets) * self.capacity_grow_factor[()]))
                target_doublets = qd.min(
                    qd.max(target_doublets, 4865),
                    self.max_contact_doublets[()],
                )
                target_triplets = qd.min(
                    qd.max(target_triplets, 4865),
                    self.max_contact_triplets[()],
                )
                if qd.f64(target_doublets) < (
                    qd.f64(self.padded_contact_doublets[()]) * self.capacity_shrink_threshold[()]
                ):
                    self.padded_contact_doublets[()] = target_doublets
                if qd.f64(target_triplets) < (
                    qd.f64(self.padded_contact_triplets[()]) * self.capacity_shrink_threshold[()]
                ):
                    self.padded_contact_triplets[()] = target_triplets

    @qd.func(requires_top_level=True)
    def reset_assembly_counts(self):
        for _ in range(1):
            self.n_contact_doublets[()] = 0
            self.n_contact_triplets[()] = 0
            self.n_unique_doublets[()] = 0
            self.n_unique_triplets[()] = 0
            self.n_active_pairs[()] = 0
            self.contact_padding_overflow[()] = 0

    @qd.func(requires_top_level=True)
    def doublet_sort_seed(self):
        qd.loop_config(name="contact_doublet_sort_seed")
        for index in range(self.doublet_sort_keys.shape[0]):
            if index < self.padded_contact_doublets[()]:
                if index < self.n_contact_doublets[()]:
                    self.doublet_sort_keys[index] = qd.u32(self.contact_doublet_vertices[index])
                    self.doublet_sort_perm[index] = index
                else:
                    self.doublet_sort_keys[index] = qd.u32(0xFFFFFFFF)
                    self.doublet_sort_perm[index] = index
        for _ in range(1):
            self.doublet_sort_size[()] = self.n_contact_doublets[()]

    @qd.func(requires_top_level=True)
    def doublet_sort_radix(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            sort(
                self.doublet_sort_keys,
                self.doublet_sort_keys_out,
                self.doublet_sort_perm,
                self.doublet_sort_perm_out,
                self.doublet_sort_scratch,
                self.doublet_sort_size,
                qd.u32,
                True,
                32,
                self.sort_log256_max_n,
            )
        else:
            dynamic_radix_sort(
                self.doublet_sorter,
                self.doublet_sort_keys,
                self.doublet_sort_keys_out,
                self.doublet_sort_perm,
                self.doublet_sort_perm_out,
                self.padded_contact_doublets[()],
            )

    @qd.func(requires_top_level=True)
    def doublet_segment_flags(self):
        qd.loop_config(name="contact_doublet_segment_flags")
        for index in range(self.doublet_sort_keys.shape[0]):
            if index < self.padded_contact_doublets[()]:
                flag = qd.i32(0)
                if index < self.n_contact_doublets[()] and (
                    index == self.n_contact_doublets[()] - 1
                    or self.doublet_sort_keys[index] != self.doublet_sort_keys[index + 1]
                ):
                    flag = qd.i32(1)
                self.doublet_seg_flags[index] = flag

    @qd.func(requires_top_level=True)
    def doublet_scan(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            exclusive_scan_add(
                self.doublet_seg_flags,
                self.doublet_seg_ids,
                self.doublet_scan_scratch,
                self.n_contact_doublets[()],
                qd.i32,
                self.sort_log256_max_n,
            )
        else:
            dynamic_exclusive_sum(
                self.doublet_scanner,
                self.doublet_seg_flags,
                self.doublet_seg_ids,
                self.padded_contact_doublets[()],
            )

    @qd.func(requires_top_level=True)
    def doublet_zero_unique(self):
        qd.loop_config(name="contact_doublet_zero_unique")
        for index in range(self.unique_doublet_gradients.shape[0]):
            if index < self.padded_contact_doublets[()]:
                for axis in qd.static(range(3)):
                    self.unique_doublet_gradients[index, axis] = 0.0

    @qd.func(requires_top_level=True)
    def doublet_fsr_merge(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            qd.loop_config(name="contact_doublet_fsr_merge_legacy")
            for index in range(self.n_contact_doublets[()]):
                source = qd.i32(self.doublet_sort_perm[index])
                segment = self.doublet_seg_ids[index]
                for axis in qd.static(range(3)):
                    qd.atomic_add(
                        self.unique_doublet_gradients[
                            segment,
                            axis,
                        ],
                        self.contact_doublet_gradients[
                            source,
                            axis,
                        ],
                    )
        else:
            fsr_reduce_doublet(
                self.doublet_seg_ids,
                self.doublet_sort_perm,
                self.doublet_sort_keys,
                self.contact_doublet_gradients,
                self.unique_doublet_gradients,
                self.n_contact_doublets,
                self.padded_contact_doublets,
                self.doublet_sort_keys.shape[0],
            )

    @qd.func(requires_top_level=True)
    def doublet_extract_unique(self):
        qd.loop_config(name="contact_doublet_extract_unique")
        for index in range(self.doublet_sort_keys.shape[0]):
            if (
                index < self.padded_contact_doublets[()]
                and index < self.n_contact_doublets[()]
                and self.doublet_seg_flags[index] != 0
            ):
                segment = qd.i32(self.doublet_seg_ids[index])
                self.unique_doublet_vertices[segment] = qd.i32(self.doublet_sort_keys[index])
                if index == self.n_contact_doublets[()] - 1:
                    self.n_unique_doublets[()] = segment + 1

    @qd.func(requires_top_level=True)
    def triplet_sort_seed(self):
        qd.loop_config(name="contact_triplet_sort_seed")
        for index in range(self.triplet_sort_keys.shape[0]):
            if index < self.padded_contact_triplets[()]:
                if index < self.n_contact_triplets[()]:
                    row = qd.u64(self.contact_triplet_rows[index])
                    column = qd.u64(self.contact_triplet_cols[index])
                    self.triplet_sort_keys[index] = (row << 32) | column
                    self.triplet_sort_perm[index] = index
                else:
                    self.triplet_sort_keys[index] = qd.u64(0xFFFFFFFFFFFFFFFF)
                    self.triplet_sort_perm[index] = index
        for _ in range(1):
            self.triplet_sort_size[()] = self.n_contact_triplets[()]

    @qd.func(requires_top_level=True)
    def triplet_sort_radix(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            sort(
                self.triplet_sort_keys,
                self.triplet_sort_keys_out,
                self.triplet_sort_perm,
                self.triplet_sort_perm_out,
                self.triplet_sort_scratch,
                self.triplet_sort_size,
                qd.u64,
                True,
                64,
                self.sort_log256_max_n,
            )
        else:
            dynamic_radix_sort(
                self.triplet_sorter,
                self.triplet_sort_keys,
                self.triplet_sort_keys_out,
                self.triplet_sort_perm,
                self.triplet_sort_perm_out,
                self.padded_contact_triplets[()],
            )

    @qd.func(requires_top_level=True)
    def triplet_segment_flags(self):
        qd.loop_config(name="contact_triplet_segment_flags")
        for index in range(self.triplet_sort_keys.shape[0]):
            if index < self.padded_contact_triplets[()]:
                flag = qd.i32(0)
                if index < self.n_contact_triplets[()] and (
                    index == self.n_contact_triplets[()] - 1
                    or self.triplet_sort_keys[index] != self.triplet_sort_keys[index + 1]
                ):
                    flag = qd.i32(1)
                self.triplet_seg_flags[index] = flag

    @qd.func(requires_top_level=True)
    def triplet_scan(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            exclusive_scan_add(
                self.triplet_seg_flags,
                self.triplet_seg_ids,
                self.triplet_scan_scratch,
                self.n_contact_triplets[()],
                qd.i32,
                self.sort_log256_max_n,
            )
        else:
            dynamic_exclusive_sum(
                self.triplet_scanner,
                self.triplet_seg_flags,
                self.triplet_seg_ids,
                self.padded_contact_triplets[()],
            )

    @qd.func(requires_top_level=True)
    def triplet_zero_unique(self):
        qd.loop_config(name="contact_triplet_zero_unique")
        for index in range(self.unique_triplet_values.shape[0]):
            if index < self.padded_contact_triplets[()]:
                for row in qd.static(range(3)):
                    for column in qd.static(range(3)):
                        self.unique_triplet_values[index, row, column] = 0.0

    @qd.func(requires_top_level=True)
    def triplet_fsr_merge(self):
        if qd.static(self.genesis_legacy_sort_reduce_host):
            qd.loop_config(name="contact_triplet_fsr_merge_legacy")
            for index in range(self.n_contact_triplets[()]):
                source = qd.i32(self.triplet_sort_perm[index])
                segment = self.triplet_seg_ids[index]
                for row in qd.static(range(3)):
                    for column in qd.static(range(3)):
                        qd.atomic_add(
                            self.unique_triplet_values[
                                segment,
                                row,
                                column,
                            ],
                            self.contact_triplet_values[
                                source,
                                row,
                                column,
                            ],
                        )
        else:
            fsr_reduce_triplet(
                self.triplet_seg_ids,
                self.triplet_sort_perm,
                self.triplet_sort_keys,
                self.contact_triplet_values,
                self.unique_triplet_values,
                self.n_contact_triplets,
                self.padded_contact_triplets,
                self.triplet_sort_keys.shape[0],
            )

    @qd.func(requires_top_level=True)
    def triplet_extract_unique(self):
        qd.loop_config(name="contact_triplet_extract_unique")
        for index in range(self.triplet_sort_keys.shape[0]):
            if (
                index < self.padded_contact_triplets[()]
                and index < self.n_contact_triplets[()]
                and self.triplet_seg_flags[index] != 0
            ):
                segment = qd.i32(self.triplet_seg_ids[index])
                key = self.triplet_sort_keys[index]
                self.unique_triplet_rows[segment] = qd.i32(key >> 32)
                self.unique_triplet_cols[segment] = qd.i32(key & qd.u64(0xFFFFFFFF))
                if index == self.n_contact_triplets[()] - 1:
                    self.n_unique_triplets[()] = segment + 1

    @qd.func(requires_top_level=True)
    def sort_reduce(self):
        self.doublet_sort_seed()
        self.doublet_sort_radix()
        self.doublet_segment_flags()
        self.doublet_scan()
        self.doublet_zero_unique()
        self.doublet_fsr_merge()
        self.doublet_extract_unique()
        self.triplet_sort_seed()
        self.triplet_sort_radix()
        self.triplet_segment_flags()
        self.triplet_scan()
        self.triplet_zero_unique()
        self.triplet_fsr_merge()
        self.triplet_extract_unique()
