from __future__ import annotations

import math

import numpy as np
import quadrants as qd

from genesis.utils import geom as gu

from .finite_element.finite_element import _surface_area_weights, _surface_edges
from .rigid_contact_proxy_kkt import (
    fk_defect_prefix_cap,
    rigid_contact_proxy_constraint,
    rigid_contact_proxy_prepare_maps,
    so3_left_jacobian,
)
from .rigid_system import RigidSystem
from .sim_system import SimSystem


class RigidContactProxyGeometry:
    """Build-time staging for collision meshes delegated to rigid proxies."""

    def init(self, scene, d_hat: float) -> bool:
        rigid_solver = scene.rigid_solver
        if not rigid_solver.is_active:
            return False
        if d_hat <= 0.0 or not math.isfinite(d_hat):
            raise ValueError("RigidContactProxyGeometry d_hat must be finite and positive")

        mechanism_body: list[int] = []
        proxy_body: list[int] = []
        surface_radius: list[float] = []
        local_positions: list[np.ndarray] = []
        vertex_pair: list[np.ndarray] = []
        geometry_ids: list[np.ndarray] = []
        source_geometry_ids: list[np.ndarray] = []
        geometry_environments: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        n_mechanism_bodies = rigid_solver.n_links * rigid_solver._B
        vertex_offset = 0

        for environment in range(rigid_solver._B):
            for link in rigid_solver.links:
                geoms = []
                for geom in link.geoms:
                    if geom.contype == 0 or geom.n_verts == 0 or geom.n_faces == 0:
                        continue
                    if geom.active_envs_idx is not None and environment not in geom.active_envs_idx:
                        continue
                    geoms.append(geom)
                if not geoms:
                    continue

                inertial_position = np.asarray(link.desc.inertial_pos, dtype=np.float64)
                inertial_rotation = gu.quat_to_R(np.asarray(link.desc.inertial_quat, dtype=np.float64))
                pair_positions = []
                pair_triangles = []
                pair_geometry_ids = []
                pair_source_geometry_ids = []
                pair_geometry_environments = []
                pair_vertex_offset = 0
                for geom in geoms:
                    geom_rotation = gu.quat_to_R(np.asarray(geom.init_quat, dtype=np.float64))
                    positions_link = np.asarray(geom.init_verts, dtype=np.float64) @ geom_rotation.T + np.asarray(
                        geom.init_pos, dtype=np.float64
                    )
                    positions_inertial = (positions_link - inertial_position) @ inertial_rotation
                    pair_positions.append(positions_inertial)
                    pair_triangles.append(np.asarray(geom.init_faces, dtype=np.int32) + pair_vertex_offset)
                    pair_geometry_ids.append(
                        np.full(
                            geom.n_verts,
                            environment * rigid_solver.n_geoms + geom.idx,
                            dtype=np.int32,
                        )
                    )
                    pair_source_geometry_ids.append(np.full(geom.n_verts, geom.idx, dtype=np.int32))
                    pair_geometry_environments.append(np.full(geom.n_verts, environment, dtype=np.int32))
                    pair_vertex_offset += geom.n_verts

                pair = len(mechanism_body)
                positions = np.ascontiguousarray(np.concatenate(pair_positions), dtype=np.float64)
                faces = np.ascontiguousarray(np.concatenate(pair_triangles), dtype=np.int32)
                mechanism_body.append(environment * rigid_solver.n_links + link.idx)
                proxy_body.append(n_mechanism_bodies + pair)
                surface_radius.append(float(np.linalg.norm(positions, axis=1).max(initial=0.0)))
                local_positions.append(positions)
                vertex_pair.append(np.full(len(positions), pair, dtype=np.int32))
                geometry_ids.append(np.concatenate(pair_geometry_ids))
                source_geometry_ids.append(np.concatenate(pair_source_geometry_ids))
                geometry_environments.append(np.concatenate(pair_geometry_environments))
                triangles.append(faces + vertex_offset)
                vertex_offset += len(positions)

        if not mechanism_body:
            return False

        self.n_mechanism_bodies = n_mechanism_bodies
        self.n_pairs = len(mechanism_body)
        self.n_rigid_bodies = n_mechanism_bodies + self.n_pairs
        self.mechanism_body = np.ascontiguousarray(mechanism_body, dtype=np.int32)
        self.proxy_body = np.ascontiguousarray(proxy_body, dtype=np.int32)
        self.surface_radius = np.ascontiguousarray(surface_radius, dtype=np.float64)
        self.local_positions = np.ascontiguousarray(np.concatenate(local_positions), dtype=np.float64)
        self.vertex_pair = np.ascontiguousarray(np.concatenate(vertex_pair), dtype=np.int32)
        self.geometry_ids = np.ascontiguousarray(
            np.concatenate(geometry_ids),
            dtype=np.int32,
        )
        self.source_geometry_ids = np.ascontiguousarray(
            np.concatenate(source_geometry_ids),
            dtype=np.int32,
        )
        self.geometry_environments = np.ascontiguousarray(
            np.concatenate(geometry_environments),
            dtype=np.int32,
        )
        self.surf_triangles = np.ascontiguousarray(np.concatenate(triangles), dtype=np.int32)
        self.surf_edges = np.ascontiguousarray(_surface_edges(self.surf_triangles), dtype=np.int32)
        self.surf_verts = np.arange(len(self.local_positions), dtype=np.int32)
        self.vert_dimensions = np.full(len(self.local_positions), 2, dtype=np.int32)
        (
            self.vert_area_weights,
            self.edge_area_weights,
            self.face_area_weights,
        ) = _surface_area_weights(self.local_positions, self.surf_triangles, self.surf_edges)
        self.thicknesses = np.zeros(len(self.local_positions), dtype=np.float64)
        self.d_hats = np.full(len(self.local_positions), d_hat, dtype=np.float64)
        self.is_fixed = np.zeros(len(self.local_positions), dtype=np.int32)
        for vertex, pair in enumerate(self.vertex_pair):
            mechanism = self.mechanism_body[pair]
            link = rigid_solver.links[mechanism % rigid_solver.n_links]
            self.is_fixed[vertex] = int(link.is_fixed)
        return True


@qd.data_oriented
class RigidContactProxySystem(SimSystem):
    globalization_watchdog = 0
    globalization_merit = 1
    merit_dot_blocks = 64
    merit_dot_warps_per_block = 8
    merit_dot_partials = merit_dot_blocks * merit_dot_warps_per_block
    filter_capacity_value = 1024
    correction_fraction = 0.1
    penalty_ratio = 1.0 / (correction_fraction * correction_fraction)

    def __init__(self) -> None:
        super().__init__()
        self.is_initialized_host = False
        self.globalization_mode_host = self.globalization_merit
        self.restoration_enabled_host = True
        self.test_merit_energy_bias_host = 0.0
        self.ls_forensics_test_energy_bias_host = 0.0
        self.has_forest = False

    def build(self) -> None:
        from .rigid_joint_forest import RigidJointForestSystem

        self.rigid = self.require(RigidSystem)
        self.forest = self.require(RigidJointForestSystem)
        self.has_forest = True
        self.n_links_host = self.rigid.dyn_state.links.pos.shape[0]
        self.n_instances_host = self.rigid.n_instances_host

    def configure(
        self,
        globalization: str = "merit",
        restoration: bool = True,
        test_merit_energy_bias: float = 0.0,
        ls_forensics_test_energy_bias: float = 0.0,
    ) -> None:
        if globalization == "watchdog":
            mode = self.globalization_watchdog
        elif globalization == "merit":
            mode = self.globalization_merit
        else:
            raise ValueError("rigid_proxy/globalization must be 'watchdog' or 'merit'")
        if not math.isfinite(test_merit_energy_bias) or test_merit_energy_bias < 0.0:
            raise ValueError("rigid_proxy/test_merit_energy_bias must be finite and nonnegative")
        if not math.isfinite(ls_forensics_test_energy_bias) or ls_forensics_test_energy_bias < 0.0:
            raise ValueError("extras/ls_forensics/test_energy_bias must be finite and nonnegative")
        self.globalization_mode_host = mode
        self.restoration_enabled_host = bool(restoration)
        self.test_merit_energy_bias_host = float(test_merit_energy_bias)
        self.ls_forensics_test_energy_bias_host = float(ls_forensics_test_energy_bias)

    def wire_data(
        self,
        n_rigid_bodies: int,
        mechanism_body: np.ndarray,
        proxy_body: np.ndarray,
        surface_radius: np.ndarray,
    ) -> None:
        if self.is_initialized_host:
            raise RuntimeError("RigidContactProxySystem data is already wired")

        mechanism = np.ascontiguousarray(mechanism_body, dtype=np.int32).reshape(-1)
        proxies = np.ascontiguousarray(proxy_body, dtype=np.int32).reshape(-1)
        radii = np.ascontiguousarray(surface_radius, dtype=np.float64).reshape(-1)
        if len(proxies) != len(mechanism) or len(radii) != len(mechanism):
            raise ValueError("RigidContactProxySystem mapping arrays must have equal length")
        if n_rigid_bodies < 0:
            raise ValueError("RigidContactProxySystem body count must be non-negative")
        if np.any(mechanism < 0) or np.any(mechanism >= self.n_links_host * self.n_instances_host):
            raise ValueError("RigidContactProxySystem mechanism body is out of range")
        if np.any(proxies < 0) or np.any(proxies >= n_rigid_bodies):
            raise ValueError("RigidContactProxySystem proxy body is out of range")
        if np.any(mechanism >= proxies):
            raise ValueError("RigidContactProxySystem proxies must be appended after mechanism bodies")
        if len(np.unique(proxies)) != len(proxies):
            raise ValueError("RigidContactProxySystem proxy body appears more than once")
        if np.any(~np.isfinite(radii)) or np.any(radii < 0.0):
            raise ValueError("RigidContactProxySystem surface radius must be finite and non-negative")

        n_pairs = len(mechanism)
        pair_capacity = max(n_pairs, 1)
        body_capacity = max(n_rigid_bodies, 1)
        self.n_bodies_host = n_rigid_bodies
        self.n_pairs_host = n_pairs

        self.n_bodies = qd.ndarray(qd.i32, shape=())
        self.n_pairs = qd.ndarray(qd.i32, shape=())
        self.n_joint_edges = qd.ndarray(qd.i32, shape=())
        self.n_energy_partial = qd.ndarray(qd.i32, shape=())
        self.n_residual_partial = qd.ndarray(qd.i32, shape=())
        self.filter_capacity = qd.ndarray(qd.i32, shape=())
        self.merit_gradient_capacity = qd.ndarray(qd.i32, shape=())
        self.globalization_mode = qd.ndarray(qd.i32, shape=())
        self.restoration_enabled = qd.ndarray(qd.i32, shape=())
        self.test_merit_energy_bias = qd.ndarray(qd.f64, shape=())
        self.ls_forensics_test_energy_bias = qd.ndarray(qd.f64, shape=())

        self.mechanism_body = qd.ndarray(qd.i32, shape=(pair_capacity,))
        self.proxy_body = qd.ndarray(qd.i32, shape=(pair_capacity,))
        self.pair_of_body = qd.ndarray(qd.i32, shape=(body_capacity,))
        self.surface_radius = qd.ndarray(qd.f64, shape=(pair_capacity,))

        self.t = qd.ndarray(qd.f64, shape=(pair_capacity, 3))
        self.quat = qd.ndarray(qd.f64, shape=(pair_capacity, 4))
        self.t_prev = qd.ndarray(qd.f64, shape=(pair_capacity, 3))
        self.quat_prev = qd.ndarray(qd.f64, shape=(pair_capacity, 4))
        self.t_temp = qd.ndarray(qd.f64, shape=(pair_capacity, 3))
        self.quat_temp = qd.ndarray(qd.f64, shape=(pair_capacity, 4))
        self.dq = qd.ndarray(qd.f64, shape=(pair_capacity, 6))

        self.lambda_ = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
        self.metric = qd.ndarray(qd.f64, shape=(pair_capacity, 6, 6))
        self.constraint = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
        self.trial_constraint = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
        self.tangent_map = qd.ndarray(qd.f64, shape=(pair_capacity, 6, 6))
        self.normal_map = qd.ndarray(qd.f64, shape=(pair_capacity, 6, 6))
        self.particular = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
        self.slack = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
        self.reaction = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
        self.path_limit = qd.ndarray(qd.f64, shape=(pair_capacity,))
        self.filter_h = qd.ndarray(qd.f64, shape=(self.filter_capacity_value,))
        self.filter_energy = qd.ndarray(qd.f64, shape=(self.filter_capacity_value,))
        self.merit_gradient = qd.ndarray(qd.f64, shape=(1,))
        self.merit_dot_partial = qd.ndarray(qd.f64, shape=(self.merit_dot_partials,))
        self.merit_control_gradient = qd.ndarray(qd.f64, shape=(1,))
        self.energy_partial = qd.ndarray(qd.f64, shape=(1,))
        self.residual_partial = qd.ndarray(qd.f64, shape=(1,))

        self.max_surface_residual = qd.ndarray(qd.f64, shape=())
        self.dual_update_flag = qd.ndarray(qd.i32, shape=())
        self.fk_alpha = qd.ndarray(qd.f64, shape=())
        self.trial_max_residual = qd.ndarray(qd.f64, shape=())
        self.solve_tolerance = qd.ndarray(qd.f64, shape=())
        self.physical_converged = qd.ndarray(qd.i32, shape=())
        self.filter_size = qd.ndarray(qd.i32, shape=())
        self.filter_retry = qd.ndarray(qd.i32, shape=())
        self.merit_active = qd.ndarray(qd.i32, shape=())
        self.merit_probe = qd.ndarray(qd.i32, shape=())
        self.merit_evaluated = qd.ndarray(qd.i32, shape=())
        self.merit_gtd = qd.ndarray(qd.f64, shape=())
        self.merit_rho = qd.ndarray(qd.f64, shape=())
        self.merit_slope = qd.ndarray(qd.f64, shape=())
        self.merit_entries = qd.ndarray(qd.i32, shape=())
        self.merit_probes = qd.ndarray(qd.i32, shape=())
        self.merit_accepts = qd.ndarray(qd.i32, shape=())
        self.merit_trials = qd.ndarray(qd.i32, shape=())
        self.filter_retries = qd.ndarray(qd.i32, shape=())
        self.merit_max_rho = qd.ndarray(qd.f64, shape=())
        self.merit_max_gtd = qd.ndarray(qd.f64, shape=())
        self.ls_exhaust_accept_count = qd.ndarray(qd.i32, shape=())
        self.frame_failed = qd.ndarray(qd.i32, shape=())
        self.restoration_active = qd.ndarray(qd.i32, shape=())
        self.restoration_triggered = qd.ndarray(qd.i32, shape=())
        self.restoration_hard_probe = qd.ndarray(qd.i32, shape=())
        self.restoration_reprice_flag = qd.ndarray(qd.i32, shape=())
        self.restoration_entries = qd.ndarray(qd.i32, shape=())
        self.restoration_newton_epochs = qd.ndarray(qd.i32, shape=())
        self.restoration_dual_epochs = qd.ndarray(qd.i32, shape=())
        self.restoration_hard_probes = qd.ndarray(qd.i32, shape=())
        self.restoration_max_slack = qd.ndarray(qd.f64, shape=())
        self.restoration_energy = qd.ndarray(qd.f64, shape=())

        pair_of_body = np.full(body_capacity, -1, dtype=np.int32)
        for pair, (mechanism_id, proxy_id) in enumerate(zip(mechanism, proxies, strict=True)):
            pair_of_body[mechanism_id] = pair
            pair_of_body[proxy_id] = pair

        self.n_bodies.from_numpy(np.array(n_rigid_bodies, dtype=np.int32))
        self.n_pairs.from_numpy(np.array(n_pairs, dtype=np.int32))
        self.n_joint_edges.from_numpy(np.array(0, dtype=np.int32))
        self.n_energy_partial.from_numpy(np.array(0, dtype=np.int32))
        self.n_residual_partial.from_numpy(np.array(0, dtype=np.int32))
        self.filter_capacity.from_numpy(np.array(self.filter_capacity_value, dtype=np.int32))
        self.merit_gradient_capacity.from_numpy(np.array(0, dtype=np.int32))
        self.globalization_mode.from_numpy(np.array(self.globalization_mode_host, dtype=np.int32))
        self.restoration_enabled.from_numpy(np.array(self.restoration_enabled_host, dtype=np.int32))
        self.test_merit_energy_bias.from_numpy(np.array(self.test_merit_energy_bias_host, dtype=np.float64))
        self.ls_forensics_test_energy_bias.from_numpy(
            np.array(
                self.ls_forensics_test_energy_bias_host,
                dtype=np.float64,
            )
        )
        self.mechanism_body.from_numpy(mechanism if n_pairs else np.zeros(pair_capacity, dtype=np.int32))
        self.proxy_body.from_numpy(proxies if n_pairs else np.zeros(pair_capacity, dtype=np.int32))
        self.pair_of_body.from_numpy(pair_of_body)
        self.surface_radius.from_numpy(radii if n_pairs else np.zeros(pair_capacity, dtype=np.float64))

        zero3 = np.zeros((pair_capacity, 3), dtype=np.float64)
        identity_quat = np.zeros((pair_capacity, 4), dtype=np.float64)
        identity_quat[:, 0] = 1.0
        zero6 = np.zeros((pair_capacity, 6), dtype=np.float64)
        zero66 = np.zeros((pair_capacity, 6, 6), dtype=np.float64)
        self.t.from_numpy(zero3)
        self.quat.from_numpy(identity_quat)
        self.t_prev.from_numpy(zero3)
        self.quat_prev.from_numpy(identity_quat)
        self.t_temp.from_numpy(zero3)
        self.quat_temp.from_numpy(identity_quat)
        self.dq.from_numpy(zero6)
        self.lambda_.from_numpy(zero6)
        self.metric.from_numpy(zero66)
        self.constraint.from_numpy(zero6)
        self.trial_constraint.from_numpy(zero6)
        self.tangent_map.from_numpy(zero66)
        self.normal_map.from_numpy(zero66)
        self.particular.from_numpy(zero6)
        self.slack.from_numpy(zero6)
        self.reaction.from_numpy(zero6)
        self.path_limit.from_numpy(np.full(pair_capacity, np.inf, dtype=np.float64))
        self.filter_h.from_numpy(np.zeros(self.filter_capacity_value, dtype=np.float64))
        self.filter_energy.from_numpy(np.zeros(self.filter_capacity_value, dtype=np.float64))
        self.merit_gradient.from_numpy(np.zeros(1, dtype=np.float64))
        self.merit_dot_partial.from_numpy(np.zeros(self.merit_dot_partials, dtype=np.float64))
        self.merit_control_gradient.from_numpy(np.zeros(1, dtype=np.float64))
        self.energy_partial.from_numpy(np.zeros(1, dtype=np.float64))
        self.residual_partial.from_numpy(np.zeros(1, dtype=np.float64))
        self._reset_scalars_host()
        self.is_initialized_host = True

    def _reset_scalars_host(self) -> None:
        for field in (
            "max_surface_residual",
            "trial_max_residual",
            "solve_tolerance",
            "merit_gtd",
            "merit_rho",
            "merit_slope",
            "merit_max_rho",
            "merit_max_gtd",
            "restoration_max_slack",
            "restoration_energy",
        ):
            getattr(self, field).from_numpy(np.array(0.0, dtype=np.float64))
        self.fk_alpha.from_numpy(np.array(1.0, dtype=np.float64))
        for field in (
            "dual_update_flag",
            "physical_converged",
            "filter_size",
            "filter_retry",
            "merit_active",
            "merit_probe",
            "merit_evaluated",
            "merit_entries",
            "merit_probes",
            "merit_accepts",
            "merit_trials",
            "filter_retries",
            "ls_exhaust_accept_count",
            "frame_failed",
            "restoration_active",
            "restoration_triggered",
            "restoration_hard_probe",
            "restoration_reprice_flag",
            "restoration_entries",
            "restoration_newton_epochs",
            "restoration_dual_epochs",
            "restoration_hard_probes",
        ):
            getattr(self, field).from_numpy(np.array(0, dtype=np.int32))

    def ensure_merit_gradient_capacity(self, capacity: int) -> None:
        if self.globalization_mode_host != self.globalization_merit:
            return
        current = self.merit_gradient.shape[0]
        if capacity <= current:
            self.merit_gradient_capacity.from_numpy(np.array(capacity, dtype=np.int32))
            return
        self.merit_gradient = qd.ndarray(qd.f64, shape=(capacity,))
        self.merit_gradient.from_numpy(np.zeros(capacity, dtype=np.float64))
        self.merit_gradient_capacity.from_numpy(np.array(capacity, dtype=np.int32))

    @qd.func(requires_top_level=True)
    def capture_physical_gradient(self, linear_system: qd.template()):
        for dof in range(self.merit_gradient_capacity[()]):
            self.merit_gradient[dof] = linear_system.b_rhs[dof]

    @qd.func(requires_top_level=True)
    def initialize_merit(self):
        for _ in range(1):
            self.merit_active[()] = 0
            self.merit_probe[()] = 0
            self.merit_evaluated[()] = 0
            self.merit_rho[()] = 0.0
            self.merit_slope[()] = 0.0

    @qd.func
    def check_line_search(
        self,
        energy0,
        trial_energy,
        alpha,
        step,
        max_steps,
        exhausted,
        converged: qd.template(),
    ):
        energy_roundoff = 1.0e-12 * (1.0 + qd.abs(energy0))
        energy_ok = trial_energy <= energy0 + energy_roundoff
        accepted_by_rule = energy_ok
        decision_elastic_restoration = self.restoration_active[()] != 0
        hard_probe = self.restoration_hard_probe[()] != 0
        restoration_trigger_bias = self.ls_forensics_test_energy_bias[()] > 0.0 and self.restoration_triggered[()] == 0
        residual = self.trial_max_residual[()]
        residual0 = self.max_surface_residual[()]
        tolerance = self.solve_tolerance[()]
        merit_ok = False
        feasibility_mode = False
        anchor_h_ok = False
        anchor_energy_ok = False

        if not decision_elastic_restoration and not restoration_trigger_bias:
            residual_floor = qd.max(1.0e-12, 1.0e-6 * tolerance)
            request_merit_probe = (
                self.globalization_mode[()] == self.globalization_merit
                and not energy_ok
                and self.merit_active[()] == 0
                and self.merit_evaluated[()] == 0
                and self.filter_retry[()] == 0
                and not hard_probe
                and residual0 > residual_floor
                and residual0 <= tolerance
                and residual < residual0
            )
            if request_merit_probe:
                self.merit_probe[()] = 1
                self.merit_evaluated[()] = 1
                self.merit_probes[()] = self.merit_probes[()] + 1
                derivative = self.merit_gtd[()]
                self.merit_max_gtd[()] = qd.max(
                    self.merit_max_gtd[()],
                    qd.abs(derivative),
                )
                derivative_floor = 1.0e-12 * (1.0 + qd.abs(energy0))
                residual_rate = alpha * residual0
                if residual_rate > residual_floor and derivative > derivative_floor:
                    margin = qd.max(derivative_floor, qd.abs(derivative))
                    rho = (derivative + margin) / residual_rate
                    self.merit_active[()] = 1
                    self.merit_rho[()] = rho
                    self.merit_slope[()] = derivative - rho * residual_rate
                    self.merit_entries[()] = self.merit_entries[()] + 1
                    self.merit_max_rho[()] = qd.max(
                        self.merit_max_rho[()],
                        rho,
                    )
                self.merit_probe[()] = 0

            if self.merit_active[()] != 0 and not energy_ok:
                self.merit_trials[()] = self.merit_trials[()] + 1
                merit0 = energy0 + self.merit_rho[()] * residual0
                merit = trial_energy + self.merit_rho[()] * residual
                merit_rhs = merit0 + 1.0e-4 * alpha * self.merit_slope[()]
                merit_roundoff = 64.0 * 2.220446049250313e-16 * (1.0 + qd.abs(merit0))
                merit_ok = merit <= merit_rhs + merit_roundoff
                accepted_by_rule = merit_ok

            feasibility_mode = residual0 > tolerance or self.filter_retry[()] != 0
            if feasibility_mode:
                anchor_h_ok = residual <= (1.0 - 1.0e-4) * residual0
                anchor_energy_ok = trial_energy <= energy0 - 1.0e-4 * residual0
                accepted_by_rule = anchor_h_ok or anchor_energy_ok
                for entry in range(self.filter_size[()]):
                    filter_ok = (
                        residual <= (1.0 - 1.0e-4) * self.filter_h[entry]
                        or trial_energy <= self.filter_energy[entry] - 1.0e-4 * self.filter_h[entry]
                    )
                    accepted_by_rule = accepted_by_rule and filter_ok

        exhausted_fallback = exhausted and not accepted_by_rule
        accepted = accepted_by_rule or exhausted_fallback
        if exhausted_fallback:
            self.ls_exhaust_accept_count[()] = self.ls_exhaust_accept_count[()] + 1
            if restoration_trigger_bias:
                self.restoration_triggered[()] = 1
            self.trigger_restoration()

        if accepted and not decision_elastic_restoration:
            if self.merit_active[()] != 0 and not energy_ok and merit_ok:
                self.merit_accepts[()] = self.merit_accepts[()] + 1

            deep_backtracking = (
                accepted_by_rule and not feasibility_mode and not merit_ok and step >= 3 and step + 3 >= max_steps
            )
            will_retry_filter = (
                not feasibility_mode
                and residual0 > qd.max(1.0e-12, 1.0e-6 * tolerance)
                and residual < residual0
                and (exhausted_fallback or deep_backtracking)
            )
            if will_retry_filter:
                self.filter_retry[()] = 1
                self.filter_retries[()] = self.filter_retries[()] + 1

            if feasibility_mode:
                output = 0
                for entry in range(self.filter_size[()]):
                    dominated = residual <= self.filter_h[entry] and trial_energy <= self.filter_energy[entry]
                    if not dominated:
                        self.filter_h[output] = self.filter_h[entry]
                        self.filter_energy[output] = self.filter_energy[entry]
                        output = output + 1
                if output < self.filter_capacity[()]:
                    self.filter_h[output] = residual
                    self.filter_energy[output] = trial_energy
                    output = output + 1
                self.filter_size[()] = output
                self.filter_retry[()] = 0
            if exhausted_fallback:
                stagnation_tolerance = 1.0e-10 * (1.0 + qd.abs(energy0))
                if (
                    self.physical_converged[()] != 0
                    and residual <= tolerance
                    and qd.abs(trial_energy - energy0) <= stagnation_tolerance
                ):
                    converged[()] = 1
        return accepted

    def wire_geometry(
        self,
        global_vert_offset: int,
        geometry: RigidContactProxyGeometry,
        global_body_offset: int = 0,
    ) -> None:
        if not self.is_initialized_host:
            raise RuntimeError("RigidContactProxySystem mappings must be wired before geometry")
        local_positions = np.ascontiguousarray(geometry.local_positions, dtype=np.float64).reshape(-1, 3)
        vertex_pair = np.ascontiguousarray(geometry.vertex_pair, dtype=np.int32).reshape(-1)
        if len(local_positions) != len(vertex_pair):
            raise ValueError("RigidContactProxySystem local position and pair counts must match")
        if np.any(vertex_pair < 0) or np.any(vertex_pair >= int(self.n_pairs.to_numpy())):
            raise ValueError("RigidContactProxySystem vertex pair is out of range")
        if global_vert_offset < 0 or global_body_offset < 0:
            raise ValueError("RigidContactProxySystem global offsets must be non-negative")

        capacity = max(len(local_positions), 1)
        self.n_verts = qd.ndarray(qd.i32, shape=())
        self.global_vert_offset = qd.ndarray(qd.i32, shape=())
        self.global_body_offset = qd.ndarray(qd.i32, shape=())
        self.local_positions = qd.ndarray(qd.f64, shape=(capacity, 3))
        self.vertex_pair = qd.ndarray(qd.i32, shape=(capacity,))
        self.n_verts.from_numpy(np.array(len(local_positions), dtype=np.int32))
        self.global_vert_offset.from_numpy(np.array(global_vert_offset, dtype=np.int32))
        self.global_body_offset.from_numpy(np.array(global_body_offset, dtype=np.int32))
        self.local_positions.from_numpy(
            local_positions if len(local_positions) else np.zeros((capacity, 3), dtype=np.float64)
        )
        self.vertex_pair.from_numpy(vertex_pair if len(vertex_pair) else np.zeros(capacity, dtype=np.int32))

    @qd.func
    def _current_mechanism_pose(self, link, environment):
        link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
        link_position = self.rigid.dyn_state.links.pos[link, environment]
        link_quaternion = self.rigid.dyn_state.links.quat[link, environment]
        inertial_position = self.rigid.dyn_info.links.inertial_pos[link_index]
        inertial_quaternion = self.rigid.dyn_info.links.inertial_quat[link_index]
        position = link_position + gu.qd_transform_by_quat(
            inertial_position,
            link_quaternion,
        )
        quaternion = gu.qd_transform_quat_by_quat(
            inertial_quaternion,
            link_quaternion,
        )
        return position, quaternion

    @qd.func(requires_top_level=True)
    def initialize_proxy_state(self):
        for pair in range(self.n_pairs[()]):
            mechanism = self.mechanism_body[pair]
            link = mechanism % self.rigid.n_links[()]
            environment = mechanism // self.rigid.n_links[()]
            position, quaternion = self._current_mechanism_pose(
                link,
                environment,
            )
            for axis in qd.static(range(3)):
                value = position[axis]
                self.t[pair, axis] = value
                self.t_prev[pair, axis] = value
                self.t_temp[pair, axis] = value
            for axis in qd.static(range(4)):
                value = quaternion[axis]
                self.quat[pair, axis] = value
                self.quat_prev[pair, axis] = value
                self.quat_temp[pair, axis] = value

    @qd.func(requires_top_level=True)
    def prepare_metric(self):
        for pair in range(self.n_pairs[()]):
            mechanism = self.mechanism_body[pair]
            link = mechanism % self.rigid.n_links[()]
            environment = mechanism // self.rigid.n_links[()]
            link_index = [link, environment] if qd.static(self.rigid.rigid_config.batch_links_info) else link
            mass = self.rigid.dyn_info.links.inertial_mass[link_index]
            inertia = self.rigid.dyn_info.links.inertial_i[link_index]
            _, quaternion = self._current_mechanism_pose(link, environment)
            rotation = gu.qd_quat_to_R(quaternion, qd.f64(1.0e-12))
            world_inertia = rotation @ inertia @ rotation.transpose()
            for row in qd.static(range(6)):
                for column in qd.static(range(6)):
                    value = qd.f64(0.0)
                    if qd.static(row < 3 and column < 3):
                        if qd.static(row == column):
                            value = mass
                    elif qd.static(row >= 3 and column >= 3):
                        value = world_inertia[row - 3, column - 3]
                    self.metric[pair, row, column] = self.penalty_ratio * value

    @qd.func(requires_top_level=True)
    def reset_frame(self):
        for _ in range(1):
            self.filter_size[()] = 0
            self.filter_retry[()] = 0
            self.merit_active[()] = 0
            self.merit_probe[()] = 0
            self.merit_evaluated[()] = 0
            self.merit_gtd[()] = 0.0
            self.merit_rho[()] = 0.0
            self.merit_slope[()] = 0.0
            self.merit_entries[()] = 0
            self.merit_probes[()] = 0
            self.merit_accepts[()] = 0
            self.merit_trials[()] = 0
            self.filter_retries[()] = 0
            self.merit_max_rho[()] = 0.0
            self.merit_max_gtd[()] = 0.0
            self.ls_exhaust_accept_count[()] = 0
            self.frame_failed[()] = 0
            self.restoration_active[()] = 0
            self.restoration_hard_probe[()] = 0
            self.restoration_reprice_flag[()] = 0
            self.restoration_entries[()] = 0
            self.restoration_newton_epochs[()] = 0
            self.restoration_dual_epochs[()] = 0
            self.restoration_hard_probes[()] = 0
            self.restoration_max_slack[()] = 0.0
        for pair in range(self.n_pairs[()]):
            for component in qd.static(range(6)):
                self.lambda_[pair, component] = 0.0
                self.slack[pair, component] = 0.0

    @qd.func(requires_top_level=True)
    def mark_mechanism_constrained(self):
        for pair in range(self.n_pairs[()]):
            mechanism = self.mechanism_body[pair]
            link = mechanism % self.rigid.n_links[()]
            environment = mechanism // self.rigid.n_links[()]
            self.rigid.dyn_state.links.is_constrained[link, environment] = True

    @qd.func(requires_top_level=True)
    def prepare_tolerance(self, sim_config: qd.template(), contact: qd.template()):
        for _ in range(1):
            self.solve_tolerance[()] = qd.min(
                sim_config.tol[()],
                0.01 * contact.d_hat[()],
            )
            self.fk_alpha[()] = 1.0

    @qd.func(requires_top_level=True)
    def initialize_newton(self):
        for _ in range(1):
            if self.restoration_active[()] != 0:
                self.restoration_newton_epochs[()] = self.restoration_newton_epochs[()] + 1
            if self.restoration_hard_probe[()] != 0:
                self.restoration_hard_probes[()] = self.restoration_hard_probes[()] + 1
            if self.dual_update_flag[()] != 0 or self.restoration_reprice_flag[()] != 0:
                self.dual_update_flag[()] = 0
                self.restoration_reprice_flag[()] = 0

    @qd.func(requires_top_level=True)
    def apply_convergence(self, converged: qd.template()):
        for _ in range(1):
            self.physical_converged[()] = converged[()]
            if self.restoration_active[()] != 0:
                converged[()] = 0
            elif self.restoration_hard_probe[()] != 0:
                if converged[()] != 0 and self.max_surface_residual[()] <= self.solve_tolerance[()]:
                    self.restoration_hard_probe[()] = 0
                else:
                    converged[()] = 0
            elif converged[()] != 0 and self.max_surface_residual[()] > self.solve_tolerance[()]:
                converged[()] = 0
            if self.fk_alpha[()] < 1.0:
                converged[()] = 0

    @qd.func(requires_top_level=True)
    def prepare_path_limit(self, contact: qd.template()):
        for _ in range(1):
            self.fk_alpha[()] = 1.0
        for pair in range(self.n_pairs[()]):
            translation = qd.Vector(
                [
                    self.constraint[pair, 0],
                    self.constraint[pair, 1],
                    self.constraint[pair, 2],
                ]
            )
            rotation = qd.Vector(
                [
                    self.constraint[pair, 3],
                    self.constraint[pair, 4],
                    self.constraint[pair, 5],
                ]
            )
            residual = translation.norm() + self.surface_radius[pair] * rotation.norm()
            limit = qd.max(0.1 * contact.d_hat[()], residual)
            self.path_limit[pair] = limit
            proxy_rotation = qd.Vector.zero(qd.f64, 3)
            slack_translation = qd.Vector.zero(qd.f64, 3)
            slack_rotation = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                proxy_rotation[axis] = self.dq[pair, axis + 3]
                slack_translation[axis] = self.slack[pair, axis]
                slack_rotation[axis] = self.slack[pair, axis + 3]
            proxy_curvature = proxy_rotation.dot(proxy_rotation) * self.surface_radius[pair]
            forest_curvature = self.forest.curvature_bound(
                self.mechanism_body[pair],
                self.surface_radius[pair],
            )
            slack_residual = slack_translation.norm() + self.surface_radius[pair] * slack_rotation.norm()
            alpha = fk_defect_prefix_cap(
                residual,
                slack_residual,
                proxy_curvature + forest_curvature,
                limit,
            )
            if alpha < 1.0:
                qd.atomic_min(self.fk_alpha[()], alpha)

    @qd.func(requires_top_level=True)
    def evaluate_trial_guard(self):
        for _ in range(1):
            self.trial_max_residual[()] = 0.0
        for pair in range(self.n_pairs[()]):
            mechanism = self.mechanism_body[pair]
            mechanism_position = qd.Vector.zero(qd.f64, 3)
            mechanism_quaternion = qd.Vector.zero(qd.f64, 4)
            proxy_position = qd.Vector.zero(qd.f64, 3)
            proxy_quaternion = qd.Vector.zero(qd.f64, 4)
            for axis in qd.static(range(3)):
                mechanism_position[axis] = self.forest.endpoint_t[mechanism][axis]
                proxy_position[axis] = self.t[pair, axis]
            for axis in qd.static(range(4)):
                mechanism_quaternion[axis] = self.forest.endpoint_quat[mechanism][axis]
                proxy_quaternion[axis] = self.quat[pair, axis]
            translation = qd.Vector.zero(qd.f64, 3)
            rotation = qd.Vector.zero(qd.f64, 3)
            rigid_contact_proxy_constraint(
                mechanism_position,
                mechanism_quaternion,
                proxy_position,
                proxy_quaternion,
                translation,
                rotation,
            )
            for component in qd.static(range(3)):
                self.trial_constraint[pair, component] = translation[component]
                self.trial_constraint[pair, component + 3] = rotation[component]
            residual = translation.norm() + self.surface_radius[pair] * rotation.norm()
            qd.atomic_max(self.trial_max_residual[()], residual)
            tolerance = 1.0e-10 * (1.0 + self.path_limit[pair])
            if residual > self.path_limit[pair] + tolerance:
                self.frame_failed[()] = 1

    @qd.func(requires_top_level=True)
    def compute_restoration_energy(self, use_trial: qd.template()):
        for _ in range(1):
            self.restoration_energy[()] = 0.0
        for pair in range(self.n_pairs[()]):
            if self.restoration_active[()] != 0:
                constraint = qd.Vector.zero(qd.f64, 6)
                dual = qd.Vector.zero(qd.f64, 6)
                metric = qd.Matrix.zero(qd.f64, 6, 6)
                for row in qd.static(range(6)):
                    if qd.static(use_trial):
                        constraint[row] = self.trial_constraint[pair, row]
                    else:
                        constraint[row] = self.constraint[pair, row]
                    dual[row] = self.lambda_[pair, row]
                    for column in qd.static(range(6)):
                        metric[row, column] = self.metric[pair, row, column]
                value = dual.dot(constraint) + 0.5 * constraint.dot(metric @ constraint)
                qd.atomic_add(self.restoration_energy[()], value)

    @qd.func
    def trigger_restoration(self):
        triggered = False
        if (
            self.restoration_enabled[()] != 0
            and self.restoration_active[()] == 0
            and self.max_surface_residual[()] > self.solve_tolerance[()]
        ):
            self.restoration_active[()] = 1
            self.restoration_triggered[()] = 1
            self.restoration_hard_probe[()] = 0
            self.restoration_entries[()] = self.restoration_entries[()] + 1
            self.restoration_reprice_flag[()] = 1
            self.physical_converged[()] = 0
            triggered = True
        return triggered

    @qd.func(requires_top_level=True)
    def finalize_restoration_step(self, alpha):
        for _ in range(1):
            if self.restoration_active[()] != 0 and self.restoration_reprice_flag[()] == 0 and alpha > 0.0:
                self.dual_update_flag[()] = qd.i32(self.trial_max_residual[()] > self.solve_tolerance[()])
                if self.dual_update_flag[()] != 0:
                    self.restoration_dual_epochs[()] = self.restoration_dual_epochs[()] + 1
        for pair in range(self.n_pairs[()]):
            if (
                self.restoration_active[()] != 0
                and self.restoration_reprice_flag[()] == 0
                and alpha > 0.0
                and self.dual_update_flag[()] != 0
            ):
                constraint = qd.Vector.zero(qd.f64, 6)
                metric = qd.Matrix.zero(qd.f64, 6, 6)
                for row in qd.static(range(6)):
                    constraint[row] = self.trial_constraint[pair, row]
                    for column in qd.static(range(6)):
                        metric[row, column] = self.metric[pair, row, column]
                update = metric @ constraint
                for component in qd.static(range(6)):
                    self.lambda_[pair, component] = self.lambda_[pair, component] + update[component]
        for _ in range(1):
            if self.restoration_active[()] != 0 and self.restoration_reprice_flag[()] == 0 and alpha > 0.0:
                self.restoration_active[()] = 0
                self.restoration_hard_probe[()] = 1
                self.restoration_reprice_flag[()] = 1

    @qd.func(requires_top_level=True)
    def recover_reaction(self):
        for pair in range(self.n_pairs[()]):
            offset = self.forest.proxy_dof_offset[()] + pair * 6
            rotation = qd.Vector.zero(qd.f64, 3)
            gradient_translation = qd.Vector.zero(qd.f64, 3)
            gradient_rotation = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                rotation[axis] = self.constraint[pair, axis + 3]
                gradient_translation[axis] = self.merit_gradient[offset + axis]
                gradient_rotation[axis] = self.merit_gradient[offset + axis + 3]
            reaction_rotation = -(so3_left_jacobian(rotation).transpose() @ gradient_rotation)
            for axis in qd.static(range(3)):
                self.reaction[pair, axis] = -gradient_translation[axis]
                self.reaction[pair, axis + 3] = reaction_rotation[axis]

    @qd.func(requires_top_level=True)
    def prepare_constraint(self):
        for _ in range(1):
            self.max_surface_residual[()] = 0.0
        for pair in range(self.n_pairs[()]):
            mechanism = self.mechanism_body[pair]
            link = mechanism % self.rigid.n_links[()]
            environment = mechanism // self.rigid.n_links[()]
            mechanism_position = qd.Vector.zero(qd.f64, 3)
            mechanism_quaternion = qd.Vector.zero(qd.f64, 4)
            proxy_position = qd.Vector.zero(qd.f64, 3)
            proxy_quaternion = qd.Vector.zero(qd.f64, 4)
            current_position, current_quaternion = self._current_mechanism_pose(link, environment)
            for axis in qd.static(range(3)):
                if qd.static(self.has_forest):
                    mechanism_position[axis] = self.forest.endpoint_t[mechanism][axis]
                else:
                    mechanism_position[axis] = current_position[axis]
                proxy_position[axis] = self.t[pair, axis]
            for axis in qd.static(range(4)):
                if qd.static(self.has_forest):
                    mechanism_quaternion[axis] = self.forest.endpoint_quat[mechanism][axis]
                else:
                    mechanism_quaternion[axis] = current_quaternion[axis]
                proxy_quaternion[axis] = self.quat[pair, axis]

            translation = qd.Vector.zero(qd.f64, 3)
            rotation = qd.Vector.zero(qd.f64, 3)
            rigid_contact_proxy_constraint(
                mechanism_position,
                mechanism_quaternion,
                proxy_position,
                proxy_quaternion,
                translation,
                rotation,
            )
            constraint = qd.Vector.zero(qd.f64, 6)
            tangent_map = qd.Matrix.zero(qd.f64, 6, 6)
            normal_map = qd.Matrix.zero(qd.f64, 6, 6)
            particular = qd.Vector.zero(qd.f64, 6)
            rigid_contact_proxy_prepare_maps(
                translation,
                rotation,
                constraint,
                tangent_map,
                normal_map,
                particular,
            )
            for row in qd.static(range(6)):
                self.constraint[pair, row] = constraint[row]
                self.particular[pair, row] = particular[row]
                for column in qd.static(range(6)):
                    self.tangent_map[pair, row, column] = tangent_map[row, column]
                    self.normal_map[pair, row, column] = normal_map[row, column]

            residual = translation.norm() + self.surface_radius[pair] * rotation.norm()
            qd.atomic_max(self.max_surface_residual[()], residual)

    @qd.func(requires_top_level=True)
    def initialize_global_vertices(self, vertex: qd.template()):
        for local_vertex in range(self.n_verts[()]):
            pair = self.vertex_pair[local_vertex]
            local_position = qd.Vector.zero(qd.f64, 3)
            quaternion = qd.Vector.zero(qd.f64, 4)
            translation = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                local_position[axis] = self.local_positions[local_vertex, axis]
                translation[axis] = self.t[pair, axis]
            for axis in qd.static(range(4)):
                quaternion[axis] = self.quat[pair, axis]
            world_position = translation + gu.qd_transform_by_quat(local_position, quaternion)
            global_vertex = self.global_vert_offset[()] + local_vertex
            vertex.body_id[global_vertex] = self.global_body_offset[()] + self.proxy_body[pair]
            for axis in qd.static(range(3)):
                vertex.positions[global_vertex, axis] = world_position[axis]
                vertex.safe_positions[global_vertex, axis] = world_position[axis]
                vertex.trajectory_end_positions[global_vertex, axis] = world_position[axis]
                vertex.x_bar[global_vertex, axis] = world_position[axis]

    @qd.func(requires_top_level=True)
    def forward_global_vertices(self, vertex: qd.template()):
        for local_vertex in range(self.n_verts[()]):
            pair = self.vertex_pair[local_vertex]
            local_position = qd.Vector.zero(qd.f64, 3)
            quaternion = qd.Vector.zero(qd.f64, 4)
            translation = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                local_position[axis] = self.local_positions[local_vertex, axis]
                translation[axis] = self.t[pair, axis]
            for axis in qd.static(range(4)):
                quaternion[axis] = self.quat[pair, axis]
            world_position = translation + gu.qd_transform_by_quat(local_position, quaternion)
            global_vertex = self.global_vert_offset[()] + local_vertex
            for axis in qd.static(range(3)):
                vertex.positions[global_vertex, axis] = world_position[axis]

    @qd.func(requires_top_level=True)
    def publish_trajectory_end_positions(self, vertex: qd.template()):
        for local_vertex in range(self.n_verts[()]):
            pair = self.vertex_pair[local_vertex]
            global_vertex = self.global_vert_offset[()] + local_vertex
            pivot = qd.Vector.zero(qd.f64, 3)
            pivot_displacement = qd.Vector.zero(qd.f64, 3)
            rotation = qd.Vector.zero(qd.f64, 3)
            start = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                pivot[axis] = self.t[pair, axis]
                pivot_displacement[axis] = self.dq[pair, axis]
                rotation[axis] = self.dq[pair, axis + 3]
                start[axis] = vertex.positions[global_vertex, axis]

            lever = start - pivot
            delta_quaternion = gu.qd_rotvec_to_quat(rotation, qd.f64(1.0e-12))
            endpoint = start + pivot_displacement + gu.qd_transform_by_quat(lever, delta_quaternion) - lever
            for axis in qd.static(range(3)):
                vertex.trajectory_end_positions[global_vertex, axis] = endpoint[axis]
                vertex.path_rot[global_vertex, axis] = rotation[axis]
                vertex.path_pivot[global_vertex, axis] = pivot[axis]
                vertex.path_pivot_disp[global_vertex, axis] = pivot_displacement[axis]
            lever_length = lever.norm()
            vertex.path_inflation[global_vertex] = (
                qd.min(
                    rotation.dot(rotation) * 0.125,
                    2.0,
                )
                * lever_length
            )
            vertex.path_kind[global_vertex] = 0
            vertex.path_speed[global_vertex] = pivot_displacement.norm() + rotation.norm() * lever_length

    @qd.func(requires_top_level=True)
    def contribute_newton_max_disp(
        self,
        vertex: qd.template(),
        max_disp: qd.template(),
    ):
        for local_vertex in range(self.n_verts[()]):
            pair = self.vertex_pair[local_vertex]
            global_vertex = self.global_vert_offset[()] + local_vertex
            pivot = qd.Vector.zero(qd.f64, 3)
            pivot_displacement = qd.Vector.zero(qd.f64, 3)
            rotation = qd.Vector.zero(qd.f64, 3)
            start = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                pivot[axis] = self.t[pair, axis]
                pivot_displacement[axis] = self.dq[pair, axis]
                rotation[axis] = self.dq[pair, axis + 3]
                start[axis] = vertex.positions[global_vertex, axis]
            lever = start - pivot
            delta_quaternion = gu.qd_rotvec_to_quat(
                rotation,
                qd.f64(1.0e-12),
            )
            displacement = pivot_displacement + gu.qd_transform_by_quat(lever, delta_quaternion) - lever
            for axis in qd.static(range(3)):
                qd.atomic_max(max_disp[()], qd.abs(displacement[axis]))

    @qd.func(requires_top_level=True)
    def record_start_point(self):
        for pair in range(self.n_pairs[()]):
            for axis in qd.static(range(3)):
                self.t_temp[pair, axis] = self.t[pair, axis]
            for axis in qd.static(range(4)):
                self.quat_temp[pair, axis] = self.quat[pair, axis]

    @qd.func(requires_top_level=True)
    def step_forward(self, alpha):
        for pair in range(self.n_pairs[()]):
            angular = qd.Vector.zero(qd.f64, 3)
            for axis in qd.static(range(3)):
                self.t[pair, axis] = self.t_temp[pair, axis] + alpha * self.dq[pair, axis]
                angular[axis] = alpha * self.dq[pair, axis + 3]
            delta_quaternion = gu.qd_rotvec_to_quat(angular, qd.f64(1.0e-12))
            start_quaternion = qd.Vector.zero(qd.f64, 4)
            for axis in qd.static(range(4)):
                start_quaternion[axis] = self.quat_temp[pair, axis]
            trial_quaternion = gu.qd_quat_mul(delta_quaternion, start_quaternion)
            for axis in qd.static(range(4)):
                self.quat[pair, axis] = trial_quaternion[axis]

    @qd.func(requires_top_level=True)
    def copy_previous_state(self):
        for pair in range(self.n_pairs[()]):
            for axis in qd.static(range(3)):
                self.t_prev[pair, axis] = self.t[pair, axis]
            for axis in qd.static(range(4)):
                self.quat_prev[pair, axis] = self.quat[pair, axis]
