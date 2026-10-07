from __future__ import annotations

import math

import numpy as np
import quadrants as qd

from genesis.utils import geom as gu
from genesis.utils.misc import qd_to_numpy

from .finite_element.finite_element import _surface_area_weights, _surface_edges
from .rigid_contact_proxy_kkt import (
    fk_defect_prefix_cap,
    rigid_contact_proxy_constraint,
    rigid_contact_proxy_prepare_maps,
    so3_left_jacobian,
)
from .rigid_joint_forest import curvature_bound
from .rigid_system import RigidSystem
from .sim_system import SimData, SimSystem

_MERIT_DOT_BLOCKS = 64
_MERIT_DOT_WARPS_PER_BLOCK = 8
_MERIT_DOT_PARTIALS = _MERIT_DOT_BLOCKS * _MERIT_DOT_WARPS_PER_BLOCK
_FILTER_CAPACITY = 1024
_GLOBALIZATION_WATCHDOG = 0
_GLOBALIZATION_MERIT = 1


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


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class RigidContactProxySystem(SimSystem):
    """Organize rigid contact proxy state and typed dependencies."""

    globalization_watchdog = _GLOBALIZATION_WATCHDOG
    globalization_merit = _GLOBALIZATION_MERIT

    @qd.data_oriented
    class Data(SimData):
        """Complete mutable rigid proxy state."""

        n_bodies: qd.Ndarray
        n_pairs: qd.Ndarray
        n_joint_edges: qd.Ndarray
        n_energy_partial: qd.Ndarray
        n_residual_partial: qd.Ndarray
        filter_capacity: qd.Ndarray
        merit_gradient_capacity: qd.Ndarray
        globalization_mode: qd.Ndarray
        restoration_enabled: qd.Ndarray
        test_merit_energy_bias: qd.Ndarray
        ls_forensics_test_energy_bias: qd.Ndarray
        mechanism_body: qd.Ndarray
        proxy_body: qd.Ndarray
        pair_of_body: qd.Ndarray
        surface_radius: qd.Ndarray
        t: qd.Ndarray
        quat: qd.Ndarray
        t_prev: qd.Ndarray
        quat_prev: qd.Ndarray
        t_temp: qd.Ndarray
        quat_temp: qd.Ndarray
        dq: qd.Ndarray
        lambda_: qd.Ndarray
        metric: qd.Ndarray
        constraint: qd.Ndarray
        trial_constraint: qd.Ndarray
        tangent_map: qd.Ndarray
        normal_map: qd.Ndarray
        particular: qd.Ndarray
        slack: qd.Ndarray
        reaction: qd.Ndarray
        path_limit: qd.Ndarray
        filter_h: qd.Ndarray
        filter_energy: qd.Ndarray
        merit_gradient: qd.Ndarray
        merit_dot_partial: qd.Ndarray
        merit_control_gradient: qd.Ndarray
        energy_partial: qd.Ndarray
        residual_partial: qd.Ndarray
        max_surface_residual: qd.Ndarray
        dual_update_flag: qd.Ndarray
        fk_alpha: qd.Ndarray
        trial_max_residual: qd.Ndarray
        solve_tolerance: qd.Ndarray
        physical_converged: qd.Ndarray
        filter_size: qd.Ndarray
        filter_retry: qd.Ndarray
        merit_active: qd.Ndarray
        merit_probe: qd.Ndarray
        merit_evaluated: qd.Ndarray
        merit_gtd: qd.Ndarray
        merit_rho: qd.Ndarray
        merit_slope: qd.Ndarray
        merit_entries: qd.Ndarray
        merit_probes: qd.Ndarray
        merit_accepts: qd.Ndarray
        merit_trials: qd.Ndarray
        filter_retries: qd.Ndarray
        merit_max_rho: qd.Ndarray
        merit_max_gtd: qd.Ndarray
        ls_exhaust_accept_count: qd.Ndarray
        frame_failed: qd.Ndarray
        restoration_active: qd.Ndarray
        restoration_triggered: qd.Ndarray
        restoration_hard_probe: qd.Ndarray
        restoration_reprice_flag: qd.Ndarray
        restoration_entries: qd.Ndarray
        restoration_newton_epochs: qd.Ndarray
        restoration_dual_epochs: qd.Ndarray
        restoration_hard_probes: qd.Ndarray
        restoration_max_slack: qd.Ndarray
        restoration_energy: qd.Ndarray
        n_verts: qd.Ndarray
        global_vert_offset: qd.Ndarray
        global_body_offset: qd.Ndarray
        local_positions: qd.Ndarray
        vertex_pair: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._mechanism_body: np.ndarray | None = None
        self._proxy_body: np.ndarray | None = None
        self._surface_radius: np.ndarray | None = None
        self._geometry: RigidContactProxyGeometry | None = None
        self._global_vert_offset: int | None = None
        self._global_body_offset: int | None = None
        self._merit_gradient_capacity: int | None = None
        self._globalization: str | None = None
        self._restoration: bool | None = None
        self._test_merit_energy_bias: float | None = None
        self._ls_forensics_test_energy_bias: float | None = None
        self.n_links: int = 0
        self.n_instances: int = 0
        self.n_bodies: int = 0
        self.n_pairs: int = 0

    def wire_data(
        self,
        *,
        n_links: int,
        n_instances: int,
        n_rigid_bodies: int,
        mechanism_body: np.ndarray,
        proxy_body: np.ndarray,
        surface_radius: np.ndarray,
        geometry: RigidContactProxyGeometry,
        global_vert_offset: int,
        global_body_offset: int,
        merit_gradient_capacity: int,
        globalization: str = "merit",
        restoration: bool = True,
        test_merit_energy_bias: float = 0.0,
        ls_forensics_test_energy_bias: float = 0.0,
    ) -> None:
        self.n_links = int(n_links)
        self.n_instances = int(n_instances)
        self.n_bodies = int(n_rigid_bodies)
        self.n_pairs = len(np.ascontiguousarray(mechanism_body).reshape(-1))
        self._mechanism_body = mechanism_body
        self._proxy_body = proxy_body
        self._surface_radius = surface_radius
        self._geometry = geometry
        self._global_vert_offset = global_vert_offset
        self._global_body_offset = global_body_offset
        self._merit_gradient_capacity = merit_gradient_capacity
        self._globalization = globalization
        self._restoration = restoration
        self._test_merit_energy_bias = test_merit_energy_bias
        self._ls_forensics_test_energy_bias = ls_forensics_test_energy_bias

    def init(self) -> None:
        if (
            self._mechanism_body is None
            or self._proxy_body is None
            or self._surface_radius is None
            or self._geometry is None
            or self._global_vert_offset is None
            or self._global_body_offset is None
            or self._merit_gradient_capacity is None
            or self._globalization is None
            or self._restoration is None
            or self._test_merit_energy_bias is None
            or self._ls_forensics_test_energy_bias is None
        ):
            raise RuntimeError("RigidContactProxySystem data has not been wired")
        _populate_rigid_contact_proxy_data(
            self.data,
            n_links=self.n_links,
            n_instances=self.n_instances,
            n_rigid_bodies=self.n_bodies,
            mechanism_body=self._mechanism_body,
            proxy_body=self._proxy_body,
            surface_radius=self._surface_radius,
            geometry=self._geometry,
            global_vert_offset=self._global_vert_offset,
            global_body_offset=self._global_body_offset,
            merit_gradient_capacity=self._merit_gradient_capacity,
            globalization=self._globalization,
            restoration=self._restoration,
            test_merit_energy_bias=self._test_merit_energy_bias,
            ls_forensics_test_energy_bias=self._ls_forensics_test_energy_bias,
        )
        self._mechanism_body = None
        self._proxy_body = None
        self._surface_radius = None
        self._geometry = None
        self._global_vert_offset = None
        self._global_body_offset = None
        self._merit_gradient_capacity = None
        self._globalization = None
        self._restoration = None
        self._test_merit_energy_bias = None
        self._ls_forensics_test_energy_bias = None

    def build(self) -> None:
        from .contact_system import ContactSystem
        from .global_linear_system import GlobalLinearSystem
        from .global_vertex_manager import GlobalVertexManager
        from .pcg_solver import PCGSolver
        from .rigid_joint_forest import RigidJointForestSystem
        from .sim_config import SimConfig

        self.rigid = self.require(RigidSystem)
        self.forest = self.require(RigidJointForestSystem)
        self.contact_system = self.require(ContactSystem)
        self.vertex_system = self.require(GlobalVertexManager)
        self.sim_config_system = self.require(SimConfig)
        self.linear_system_system = self.require(GlobalLinearSystem)
        self.pcg_solver_system = self.require(PCGSolver)

    def on_kkt_failure_yield(self, _status):
        """Raise the reduced-KKT failure diagnostic for this proxy."""
        runtime = self.engine
        proxy = self.data
        linear = self.linear_system_system.data
        pcg = self.pcg_solver_system.data
        rigid = self.rigid.data
        forest = self.forest.data
        rigid_dofs = self.rigid.dof_count
        rigid_rhs = qd_to_numpy(linear.b_rhs)[:rigid_dofs]
        rigid_solution = qd_to_numpy(linear.x_sol)[:rigid_dofs]
        rigid_preconditioned = qd_to_numpy(pcg.preconditioned_residual)[:rigid_dofs]
        rigid_search = qd_to_numpy(rigid.constraint_state.search).reshape(-1)[:rigid_dofs]
        edge_dofs = qd_to_numpy(forest.edge_dof_index)[: int(qd_to_numpy(forest.n_edges))]
        parent_edges = qd_to_numpy(forest.parent_edge)[: int(qd_to_numpy(forest.n_mechanism_bodies))]
        edge_pivots = qd_to_numpy(forest.edge_d)[: len(edge_dofs)]
        edge_rhs = qd_to_numpy(forest.precond_a)[: len(edge_dofs)]
        edge_children = qd_to_numpy(forest.edge_child)[: len(edge_dofs)]
        forest_depth = qd_to_numpy(forest.depth)
        details = (
            f"newton={int(qd_to_numpy(runtime.newton_iter))}, "
            f"pcg={int(qd_to_numpy(pcg.n_iterations))}, "
            f"line_search={int(qd_to_numpy(runtime.ls_iter))}, "
            f"alpha={float(qd_to_numpy(runtime.alpha)):.6g}, "
            f"rigid_gradient_squared={float(qd_to_numpy(rigid.gradient_squared)):.6g}, "
            f"rigid_rhs_norm={float(np.linalg.norm(rigid_rhs)):.6g}, "
            f"rigid_solution_norm={float(np.linalg.norm(rigid_solution)):.6g}, "
            f"rigid_preconditioned_norm={float(np.linalg.norm(rigid_preconditioned)):.6g}, "
            f"rigid_search_norm={float(np.linalg.norm(rigid_search)):.6g}, "
            f"edge_dofs={edge_dofs.tolist()}, "
            f"edge_pivots={edge_pivots.tolist()}, "
            f"edge_rhs={edge_rhs.tolist()}, "
            f"edge_depths={forest_depth[edge_children].tolist()}, "
            f"active_parent_edges={int(np.count_nonzero(parent_edges >= 0))}, "
            f"forest_levels={int(qd_to_numpy(forest.n_levels))}, "
            f"max_disp={float(qd_to_numpy(runtime.max_disp)):.6g}, "
            f"residual={float(qd_to_numpy(proxy.max_surface_residual)):.6g}, "
            f"tolerance={float(qd_to_numpy(proxy.solve_tolerance)):.6g}, "
            f"fk_alpha={float(qd_to_numpy(proxy.fk_alpha)):.6g}, "
            f"restoration_active={int(qd_to_numpy(proxy.restoration_active))}, "
            f"hard_probe={int(qd_to_numpy(proxy.restoration_hard_probe))}, "
            f"restoration_entries={int(qd_to_numpy(proxy.restoration_entries))}, "
            f"restoration_epochs={int(qd_to_numpy(proxy.restoration_newton_epochs))}, "
            f"hard_probes={int(qd_to_numpy(proxy.restoration_hard_probes))}, "
            f"pcg_failed={int(qd_to_numpy(pcg.is_failed))}, "
            f"proxy_failed={int(qd_to_numpy(proxy.frame_failed))}"
        )
        raise RuntimeError(
            f"KKT rigid proxy solve exhausted the Newton budget before stationarity/feasibility ({details})"
        )

    @qd.func(requires_top_level=True)
    def on_initialize_state(self):
        initialize_proxy_state(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_prepare_metric(self):
        prepare_metric(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_reset_frame(self):
        reset_frame(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_mark_mechanism_constrained(self):
        mark_mechanism_constrained(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_prepare_tolerance(self):
        prepare_tolerance(
            self.data,
            self.rigid.data,
            self.forest.data,
            self.sim_config_system.data,
            self.contact_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_initialize_newton(self):
        initialize_newton(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_prepare_constraint(self):
        prepare_constraint(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_capture_physical_gradient(self):
        capture_physical_gradient(
            self.data,
            self.rigid.data,
            self.forest.data,
            self.linear_system_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_prepare_path_limit(self):
        prepare_path_limit(
            self.data,
            self.rigid.data,
            self.forest.data,
            self.contact_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_contribute_newton_max_displacement(
        self,
        max_displacement: qd.template(),  # qd.Ndarray
    ):
        contribute_newton_max_disp(
            self.data,
            self.rigid.data,
            self.forest.data,
            self.vertex_system.data,
            max_displacement,
        )

    @qd.func(requires_top_level=True)
    def on_apply_convergence(
        self,
        converged: qd.template(),  # qd.Ndarray
    ):
        apply_convergence(self.data, self.rigid.data, self.forest.data, converged)

    @qd.func(requires_top_level=True)
    def on_record_start_point(self):
        record_start_point(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_forward_global_vertices(self):
        forward_global_vertices(
            self.data,
            self.rigid.data,
            self.forest.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_publish_trajectory_end_positions(self):
        publish_trajectory_end_positions(
            self.data,
            self.rigid.data,
            self.forest.data,
            self.vertex_system.data,
        )

    @qd.func(requires_top_level=True)
    def on_compute_restoration_energy(
        self,
        use_trial: qd.template(),  # bool
    ):
        compute_restoration_energy(
            self.data,
            self.rigid.data,
            self.forest.data,
            use_trial,
        )

    @qd.func(requires_top_level=True)
    def on_initialize_merit(self):
        initialize_merit(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_step_forward(self, alpha):
        step_forward(self.data, self.rigid.data, self.forest.data, alpha)

    @qd.func(requires_top_level=True)
    def on_evaluate_trial_guard(self):
        evaluate_trial_guard(self.data, self.rigid.data, self.forest.data)

    @qd.func
    def on_check_line_search(
        self,
        energy0,
        trial_energy,
        alpha,
        step,
        max_steps,
        exhausted,
        converged: qd.template(),  # qd.Ndarray
    ):
        return check_line_search(
            self.data,
            self.rigid.data,
            self.forest.data,
            energy0,
            trial_energy,
            alpha,
            step,
            max_steps,
            exhausted,
            converged,
        )

    @qd.func(requires_top_level=True)
    def on_finalize_restoration_step(self, alpha):
        finalize_restoration_step(self.data, self.rigid.data, self.forest.data, alpha)

    @qd.func(requires_top_level=True)
    def on_recover_reaction(self):
        recover_reaction(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_copy_previous_state(self):
        copy_previous_state(self.data, self.rigid.data, self.forest.data)

    @qd.func(requires_top_level=True)
    def on_initialize_global_vertices(self):
        initialize_global_vertices(
            self.data,
            self.rigid.data,
            self.forest.data,
            self.vertex_system.data,
        )


def _reset_rigid_contact_proxy_scalars(data: RigidContactProxySystem.Data) -> None:
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
        getattr(data, field).from_numpy(np.array(0.0, dtype=np.float64))
    data.fk_alpha.from_numpy(np.array(1.0, dtype=np.float64))
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
        getattr(data, field).from_numpy(np.array(0, dtype=np.int32))


def _initialize_merit_gradient(data: RigidContactProxySystem.Data, capacity: int, mode: int) -> None:
    if mode != RigidContactProxySystem.globalization_merit:
        return
    current = data.merit_gradient.shape[0]
    if capacity <= current:
        data.merit_gradient_capacity.from_numpy(np.array(capacity, dtype=np.int32))
        return
    data.merit_gradient = qd.ndarray(qd.f64, shape=(capacity,))
    data.merit_gradient.from_numpy(np.zeros(capacity, dtype=np.float64))
    data.merit_gradient_capacity.from_numpy(np.array(capacity, dtype=np.int32))


def _populate_rigid_contact_proxy_data(
    data: RigidContactProxySystem.Data,
    *,
    n_links: int,
    n_instances: int,
    n_rigid_bodies: int,
    mechanism_body: np.ndarray,
    proxy_body: np.ndarray,
    surface_radius: np.ndarray,
    geometry: RigidContactProxyGeometry,
    global_vert_offset: int,
    global_body_offset: int,
    merit_gradient_capacity: int,
    globalization: str = "merit",
    restoration: bool = True,
    test_merit_energy_bias: float = 0.0,
    ls_forensics_test_energy_bias: float = 0.0,
) -> None:
    if globalization == "watchdog":
        mode = RigidContactProxySystem.globalization_watchdog
    elif globalization == "merit":
        mode = RigidContactProxySystem.globalization_merit
    else:
        raise ValueError("rigid_proxy/globalization must be 'watchdog' or 'merit'")
    if not math.isfinite(test_merit_energy_bias) or test_merit_energy_bias < 0.0:
        raise ValueError("rigid_proxy/test_merit_energy_bias must be finite and nonnegative")
    if not math.isfinite(ls_forensics_test_energy_bias) or ls_forensics_test_energy_bias < 0.0:
        raise ValueError("extras/ls_forensics/test_energy_bias must be finite and nonnegative")
    mechanism = np.ascontiguousarray(mechanism_body, dtype=np.int32).reshape(-1)
    proxies = np.ascontiguousarray(proxy_body, dtype=np.int32).reshape(-1)
    radii = np.ascontiguousarray(surface_radius, dtype=np.float64).reshape(-1)
    if len(proxies) != len(mechanism) or len(radii) != len(mechanism):
        raise ValueError("RigidContactProxySystem mapping arrays must have equal length")
    if n_rigid_bodies < 0:
        raise ValueError("RigidContactProxySystem body count must be non-negative")
    if np.any(mechanism < 0) or np.any(mechanism >= n_links * n_instances):
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
    data.n_bodies = qd.ndarray(qd.i32, shape=())
    data.n_pairs = qd.ndarray(qd.i32, shape=())
    data.n_joint_edges = qd.ndarray(qd.i32, shape=())
    data.n_energy_partial = qd.ndarray(qd.i32, shape=())
    data.n_residual_partial = qd.ndarray(qd.i32, shape=())
    data.filter_capacity = qd.ndarray(qd.i32, shape=())
    data.merit_gradient_capacity = qd.ndarray(qd.i32, shape=())
    data.globalization_mode = qd.ndarray(qd.i32, shape=())
    data.restoration_enabled = qd.ndarray(qd.i32, shape=())
    data.test_merit_energy_bias = qd.ndarray(qd.f64, shape=())
    data.ls_forensics_test_energy_bias = qd.ndarray(qd.f64, shape=())

    data.mechanism_body = qd.ndarray(qd.i32, shape=(pair_capacity,))
    data.proxy_body = qd.ndarray(qd.i32, shape=(pair_capacity,))
    data.pair_of_body = qd.ndarray(qd.i32, shape=(body_capacity,))
    data.surface_radius = qd.ndarray(qd.f64, shape=(pair_capacity,))

    data.t = qd.ndarray(qd.f64, shape=(pair_capacity, 3))
    data.quat = qd.ndarray(qd.f64, shape=(pair_capacity, 4))
    data.t_prev = qd.ndarray(qd.f64, shape=(pair_capacity, 3))
    data.quat_prev = qd.ndarray(qd.f64, shape=(pair_capacity, 4))
    data.t_temp = qd.ndarray(qd.f64, shape=(pair_capacity, 3))
    data.quat_temp = qd.ndarray(qd.f64, shape=(pair_capacity, 4))
    data.dq = qd.ndarray(qd.f64, shape=(pair_capacity, 6))

    data.lambda_ = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
    data.metric = qd.ndarray(qd.f64, shape=(pair_capacity, 6, 6))
    data.constraint = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
    data.trial_constraint = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
    data.tangent_map = qd.ndarray(qd.f64, shape=(pair_capacity, 6, 6))
    data.normal_map = qd.ndarray(qd.f64, shape=(pair_capacity, 6, 6))
    data.particular = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
    data.slack = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
    data.reaction = qd.ndarray(qd.f64, shape=(pair_capacity, 6))
    data.path_limit = qd.ndarray(qd.f64, shape=(pair_capacity,))
    data.filter_h = qd.ndarray(qd.f64, shape=(_FILTER_CAPACITY,))
    data.filter_energy = qd.ndarray(qd.f64, shape=(_FILTER_CAPACITY,))
    data.merit_gradient = qd.ndarray(qd.f64, shape=(1,))
    data.merit_dot_partial = qd.ndarray(qd.f64, shape=(_MERIT_DOT_PARTIALS,))
    data.merit_control_gradient = qd.ndarray(qd.f64, shape=(1,))
    data.energy_partial = qd.ndarray(qd.f64, shape=(1,))
    data.residual_partial = qd.ndarray(qd.f64, shape=(1,))

    data.max_surface_residual = qd.ndarray(qd.f64, shape=())
    data.dual_update_flag = qd.ndarray(qd.i32, shape=())
    data.fk_alpha = qd.ndarray(qd.f64, shape=())
    data.trial_max_residual = qd.ndarray(qd.f64, shape=())
    data.solve_tolerance = qd.ndarray(qd.f64, shape=())
    data.physical_converged = qd.ndarray(qd.i32, shape=())
    data.filter_size = qd.ndarray(qd.i32, shape=())
    data.filter_retry = qd.ndarray(qd.i32, shape=())
    data.merit_active = qd.ndarray(qd.i32, shape=())
    data.merit_probe = qd.ndarray(qd.i32, shape=())
    data.merit_evaluated = qd.ndarray(qd.i32, shape=())
    data.merit_gtd = qd.ndarray(qd.f64, shape=())
    data.merit_rho = qd.ndarray(qd.f64, shape=())
    data.merit_slope = qd.ndarray(qd.f64, shape=())
    data.merit_entries = qd.ndarray(qd.i32, shape=())
    data.merit_probes = qd.ndarray(qd.i32, shape=())
    data.merit_accepts = qd.ndarray(qd.i32, shape=())
    data.merit_trials = qd.ndarray(qd.i32, shape=())
    data.filter_retries = qd.ndarray(qd.i32, shape=())
    data.merit_max_rho = qd.ndarray(qd.f64, shape=())
    data.merit_max_gtd = qd.ndarray(qd.f64, shape=())
    data.ls_exhaust_accept_count = qd.ndarray(qd.i32, shape=())
    data.frame_failed = qd.ndarray(qd.i32, shape=())
    data.restoration_active = qd.ndarray(qd.i32, shape=())
    data.restoration_triggered = qd.ndarray(qd.i32, shape=())
    data.restoration_hard_probe = qd.ndarray(qd.i32, shape=())
    data.restoration_reprice_flag = qd.ndarray(qd.i32, shape=())
    data.restoration_entries = qd.ndarray(qd.i32, shape=())
    data.restoration_newton_epochs = qd.ndarray(qd.i32, shape=())
    data.restoration_dual_epochs = qd.ndarray(qd.i32, shape=())
    data.restoration_hard_probes = qd.ndarray(qd.i32, shape=())
    data.restoration_max_slack = qd.ndarray(qd.f64, shape=())
    data.restoration_energy = qd.ndarray(qd.f64, shape=())

    pair_of_body = np.full(body_capacity, -1, dtype=np.int32)
    for pair, (mechanism_id, proxy_id) in enumerate(zip(mechanism, proxies, strict=True)):
        pair_of_body[mechanism_id] = pair
        pair_of_body[proxy_id] = pair

    data.n_bodies.from_numpy(np.array(n_rigid_bodies, dtype=np.int32))
    data.n_pairs.from_numpy(np.array(n_pairs, dtype=np.int32))
    data.n_joint_edges.from_numpy(np.array(0, dtype=np.int32))
    data.n_energy_partial.from_numpy(np.array(0, dtype=np.int32))
    data.n_residual_partial.from_numpy(np.array(0, dtype=np.int32))
    data.filter_capacity.from_numpy(np.array(_FILTER_CAPACITY, dtype=np.int32))
    data.merit_gradient_capacity.from_numpy(np.array(0, dtype=np.int32))
    data.globalization_mode.from_numpy(np.array(mode, dtype=np.int32))
    data.restoration_enabled.from_numpy(np.array(restoration, dtype=np.int32))
    data.test_merit_energy_bias.from_numpy(np.array(test_merit_energy_bias, dtype=np.float64))
    data.ls_forensics_test_energy_bias.from_numpy(np.array(ls_forensics_test_energy_bias, dtype=np.float64))
    data.mechanism_body.from_numpy(mechanism if n_pairs else np.zeros(pair_capacity, dtype=np.int32))
    data.proxy_body.from_numpy(proxies if n_pairs else np.zeros(pair_capacity, dtype=np.int32))
    data.pair_of_body.from_numpy(pair_of_body)
    data.surface_radius.from_numpy(radii if n_pairs else np.zeros(pair_capacity, dtype=np.float64))

    zero3 = np.zeros((pair_capacity, 3), dtype=np.float64)
    identity_quat = np.zeros((pair_capacity, 4), dtype=np.float64)
    identity_quat[:, 0] = 1.0
    zero6 = np.zeros((pair_capacity, 6), dtype=np.float64)
    zero66 = np.zeros((pair_capacity, 6, 6), dtype=np.float64)
    data.t.from_numpy(zero3)
    data.quat.from_numpy(identity_quat)
    data.t_prev.from_numpy(zero3)
    data.quat_prev.from_numpy(identity_quat)
    data.t_temp.from_numpy(zero3)
    data.quat_temp.from_numpy(identity_quat)
    data.dq.from_numpy(zero6)
    data.lambda_.from_numpy(zero6)
    data.metric.from_numpy(zero66)
    data.constraint.from_numpy(zero6)
    data.trial_constraint.from_numpy(zero6)
    data.tangent_map.from_numpy(zero66)
    data.normal_map.from_numpy(zero66)
    data.particular.from_numpy(zero6)
    data.slack.from_numpy(zero6)
    data.reaction.from_numpy(zero6)
    data.path_limit.from_numpy(np.full(pair_capacity, np.inf, dtype=np.float64))
    data.filter_h.from_numpy(np.zeros(_FILTER_CAPACITY, dtype=np.float64))
    data.filter_energy.from_numpy(np.zeros(_FILTER_CAPACITY, dtype=np.float64))
    data.merit_gradient.from_numpy(np.zeros(1, dtype=np.float64))
    data.merit_dot_partial.from_numpy(np.zeros(_MERIT_DOT_PARTIALS, dtype=np.float64))
    data.merit_control_gradient.from_numpy(np.zeros(1, dtype=np.float64))
    data.energy_partial.from_numpy(np.zeros(1, dtype=np.float64))
    data.residual_partial.from_numpy(np.zeros(1, dtype=np.float64))
    _reset_rigid_contact_proxy_scalars(data)
    local_positions = np.ascontiguousarray(geometry.local_positions, dtype=np.float64).reshape(-1, 3)
    vertex_pair = np.ascontiguousarray(geometry.vertex_pair, dtype=np.int32).reshape(-1)
    if len(local_positions) != len(vertex_pair):
        raise ValueError("RigidContactProxySystem local position and pair counts must match")
    if np.any(vertex_pair < 0) or np.any(vertex_pair >= n_pairs):
        raise ValueError("RigidContactProxySystem vertex pair is out of range")
    if global_vert_offset < 0 or global_body_offset < 0:
        raise ValueError("RigidContactProxySystem global offsets must be non-negative")

    capacity = max(len(local_positions), 1)
    data.n_verts = qd.ndarray(qd.i32, shape=())
    data.global_vert_offset = qd.ndarray(qd.i32, shape=())
    data.global_body_offset = qd.ndarray(qd.i32, shape=())
    data.local_positions = qd.ndarray(qd.f64, shape=(capacity, 3))
    data.vertex_pair = qd.ndarray(qd.i32, shape=(capacity,))
    data.n_verts.from_numpy(np.array(len(local_positions), dtype=np.int32))
    data.global_vert_offset.from_numpy(np.array(global_vert_offset, dtype=np.int32))
    data.global_body_offset.from_numpy(np.array(global_body_offset, dtype=np.int32))
    data.local_positions.from_numpy(
        local_positions if len(local_positions) else np.zeros((capacity, 3), dtype=np.float64)
    )
    data.vertex_pair.from_numpy(vertex_pair if len(vertex_pair) else np.zeros(capacity, dtype=np.int32))
    _initialize_merit_gradient(data, merit_gradient_capacity, mode)


@qd.func(requires_top_level=True)
def capture_physical_gradient(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    linear_system_data: qd.template(),  # GlobalLinearSystem.Data
):
    for dof in range(data.merit_gradient_capacity[()]):
        data.merit_gradient[dof] = linear_system_data.read_rhs(dof)


@qd.func(requires_top_level=True)
def initialize_merit(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for _ in range(1):
        data.merit_active[()] = 0
        data.merit_probe[()] = 0
        data.merit_evaluated[()] = 0
        data.merit_rho[()] = 0.0
        data.merit_slope[()] = 0.0


@qd.func
def check_line_search(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    energy0,
    trial_energy,
    alpha,
    step,
    max_steps,
    exhausted,
    converged: qd.template(),  # qd.Ndarray
):
    energy_roundoff = 1.0e-12 * (1.0 + qd.abs(energy0))
    energy_ok = trial_energy <= energy0 + energy_roundoff
    accepted_by_rule = energy_ok
    decision_elastic_restoration = data.restoration_active[()] != 0
    hard_probe = data.restoration_hard_probe[()] != 0
    restoration_trigger_bias = data.ls_forensics_test_energy_bias[()] > 0.0 and data.restoration_triggered[()] == 0
    residual = data.trial_max_residual[()]
    residual0 = data.max_surface_residual[()]
    tolerance = data.solve_tolerance[()]
    merit_ok = False
    feasibility_mode = False
    anchor_h_ok = False
    anchor_energy_ok = False

    if not decision_elastic_restoration and not restoration_trigger_bias:
        residual_floor = qd.max(1.0e-12, 1.0e-6 * tolerance)
        request_merit_probe = (
            data.globalization_mode[()] == 1
            and not energy_ok
            and data.merit_active[()] == 0
            and data.merit_evaluated[()] == 0
            and data.filter_retry[()] == 0
            and not hard_probe
            and residual0 > residual_floor
            and residual0 <= tolerance
            and residual < residual0
        )
        if request_merit_probe:
            data.merit_probe[()] = 1
            data.merit_evaluated[()] = 1
            data.merit_probes[()] = data.merit_probes[()] + 1
            derivative = data.merit_gtd[()]
            data.merit_max_gtd[()] = qd.max(
                data.merit_max_gtd[()],
                qd.abs(derivative),
            )
            derivative_floor = 1.0e-12 * (1.0 + qd.abs(energy0))
            residual_rate = alpha * residual0
            if residual_rate > residual_floor and derivative > derivative_floor:
                margin = qd.max(derivative_floor, qd.abs(derivative))
                rho = (derivative + margin) / residual_rate
                data.merit_active[()] = 1
                data.merit_rho[()] = rho
                data.merit_slope[()] = derivative - rho * residual_rate
                data.merit_entries[()] = data.merit_entries[()] + 1
                data.merit_max_rho[()] = qd.max(
                    data.merit_max_rho[()],
                    rho,
                )
            data.merit_probe[()] = 0

        if data.merit_active[()] != 0 and not energy_ok:
            data.merit_trials[()] = data.merit_trials[()] + 1
            merit0 = energy0 + data.merit_rho[()] * residual0
            merit = trial_energy + data.merit_rho[()] * residual
            merit_rhs = merit0 + 1.0e-4 * alpha * data.merit_slope[()]
            merit_roundoff = 64.0 * 2.220446049250313e-16 * (1.0 + qd.abs(merit0))
            merit_ok = merit <= merit_rhs + merit_roundoff
            accepted_by_rule = merit_ok

        feasibility_mode = residual0 > tolerance or data.filter_retry[()] != 0
        if feasibility_mode:
            anchor_h_ok = residual <= (1.0 - 1.0e-4) * residual0
            anchor_energy_ok = trial_energy <= energy0 - 1.0e-4 * residual0
            accepted_by_rule = anchor_h_ok or anchor_energy_ok
            for entry in range(data.filter_size[()]):
                filter_ok = (
                    residual <= (1.0 - 1.0e-4) * data.filter_h[entry]
                    or trial_energy <= data.filter_energy[entry] - 1.0e-4 * data.filter_h[entry]
                )
                accepted_by_rule = accepted_by_rule and filter_ok

    exhausted_fallback = exhausted and not accepted_by_rule
    accepted = accepted_by_rule or exhausted_fallback
    if exhausted_fallback:
        data.ls_exhaust_accept_count[()] = data.ls_exhaust_accept_count[()] + 1
        if restoration_trigger_bias:
            data.restoration_triggered[()] = 1
        trigger_restoration(data, rigid, forest)

    if accepted and not decision_elastic_restoration:
        if data.merit_active[()] != 0 and not energy_ok and merit_ok:
            data.merit_accepts[()] = data.merit_accepts[()] + 1

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
            data.filter_retry[()] = 1
            data.filter_retries[()] = data.filter_retries[()] + 1

        if feasibility_mode:
            output = 0
            for entry in range(data.filter_size[()]):
                dominated = residual <= data.filter_h[entry] and trial_energy <= data.filter_energy[entry]
                if not dominated:
                    data.filter_h[output] = data.filter_h[entry]
                    data.filter_energy[output] = data.filter_energy[entry]
                    output = output + 1
            if output < data.filter_capacity[()]:
                data.filter_h[output] = residual
                data.filter_energy[output] = trial_energy
                output = output + 1
            data.filter_size[()] = output
            data.filter_retry[()] = 0
        if exhausted_fallback:
            stagnation_tolerance = 1.0e-10 * (1.0 + qd.abs(energy0))
            if (
                data.physical_converged[()] != 0
                and residual <= tolerance
                and qd.abs(trial_energy - energy0) <= stagnation_tolerance
            ):
                converged[()] = 1
    return accepted


@qd.func
def _current_mechanism_pose(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    link,
    environment,
):
    link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
    link_position = rigid.dyn_state.links.pos[link, environment]
    link_quaternion = rigid.dyn_state.links.quat[link, environment]
    inertial_position = rigid.dyn_info.links.inertial_pos[link_index]
    inertial_quaternion = rigid.dyn_info.links.inertial_quat[link_index]
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
def initialize_proxy_state(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for pair in range(data.n_pairs[()]):
        mechanism = data.mechanism_body[pair]
        link = mechanism % rigid.n_links[()]
        environment = mechanism // rigid.n_links[()]
        position, quaternion = _current_mechanism_pose(
            data,
            rigid,
            forest,
            link,
            environment,
        )
        for axis in qd.static(range(3)):
            value = position[axis]
            data.t[pair, axis] = value
            data.t_prev[pair, axis] = value
            data.t_temp[pair, axis] = value
        for axis in qd.static(range(4)):
            value = quaternion[axis]
            data.quat[pair, axis] = value
            data.quat_prev[pair, axis] = value
            data.quat_temp[pair, axis] = value


@qd.func(requires_top_level=True)
def prepare_metric(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for pair in range(data.n_pairs[()]):
        mechanism = data.mechanism_body[pair]
        link = mechanism % rigid.n_links[()]
        environment = mechanism // rigid.n_links[()]
        link_index = [link, environment] if qd.static(rigid.rigid_config.batch_links_info) else link
        mass = rigid.dyn_info.links.inertial_mass[link_index]
        inertia = rigid.dyn_info.links.inertial_i[link_index]
        _, quaternion = _current_mechanism_pose(data, rigid, forest, link, environment)
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
                data.metric[pair, row, column] = 100.0 * value


@qd.func(requires_top_level=True)
def reset_frame(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for _ in range(1):
        data.filter_size[()] = 0
        data.filter_retry[()] = 0
        data.merit_active[()] = 0
        data.merit_probe[()] = 0
        data.merit_evaluated[()] = 0
        data.merit_gtd[()] = 0.0
        data.merit_rho[()] = 0.0
        data.merit_slope[()] = 0.0
        data.merit_entries[()] = 0
        data.merit_probes[()] = 0
        data.merit_accepts[()] = 0
        data.merit_trials[()] = 0
        data.filter_retries[()] = 0
        data.merit_max_rho[()] = 0.0
        data.merit_max_gtd[()] = 0.0
        data.ls_exhaust_accept_count[()] = 0
        data.frame_failed[()] = 0
        data.restoration_active[()] = 0
        data.restoration_hard_probe[()] = 0
        data.restoration_reprice_flag[()] = 0
        data.restoration_entries[()] = 0
        data.restoration_newton_epochs[()] = 0
        data.restoration_dual_epochs[()] = 0
        data.restoration_hard_probes[()] = 0
        data.restoration_max_slack[()] = 0.0
    for pair in range(data.n_pairs[()]):
        for component in qd.static(range(6)):
            data.lambda_[pair, component] = 0.0
            data.slack[pair, component] = 0.0


@qd.func(requires_top_level=True)
def mark_mechanism_constrained(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for pair in range(data.n_pairs[()]):
        mechanism = data.mechanism_body[pair]
        link = mechanism % rigid.n_links[()]
        environment = mechanism // rigid.n_links[()]
        rigid.dyn_state.links.is_constrained[link, environment] = True


@qd.func(requires_top_level=True)
def prepare_tolerance(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    sim_config: qd.template(),  # SimConfig.Data
    contact: qd.template(),  # ContactSystem.Data
):
    for _ in range(1):
        data.solve_tolerance[()] = qd.min(
            sim_config.tol[()],
            0.01 * contact.d_hat[()],
        )
        data.fk_alpha[()] = 1.0


@qd.func(requires_top_level=True)
def initialize_newton(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for _ in range(1):
        if data.restoration_active[()] != 0:
            data.restoration_newton_epochs[()] = data.restoration_newton_epochs[()] + 1
        if data.restoration_hard_probe[()] != 0:
            data.restoration_hard_probes[()] = data.restoration_hard_probes[()] + 1
        if data.dual_update_flag[()] != 0 or data.restoration_reprice_flag[()] != 0:
            data.dual_update_flag[()] = 0
            data.restoration_reprice_flag[()] = 0


@qd.func(requires_top_level=True)
def apply_convergence(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    converged: qd.template(),  # qd.Ndarray
):
    for _ in range(1):
        data.physical_converged[()] = converged[()]
        if data.restoration_active[()] != 0:
            converged[()] = 0
        elif data.restoration_hard_probe[()] != 0:
            if converged[()] != 0 and data.max_surface_residual[()] <= data.solve_tolerance[()]:
                data.restoration_hard_probe[()] = 0
            else:
                converged[()] = 0
        elif converged[()] != 0 and data.max_surface_residual[()] > data.solve_tolerance[()]:
            converged[()] = 0
        if data.fk_alpha[()] < 1.0:
            converged[()] = 0


@qd.func(requires_top_level=True)
def prepare_path_limit(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    contact: qd.template(),  # ContactSystem.Data
):
    for _ in range(1):
        data.fk_alpha[()] = 1.0
    for pair in range(data.n_pairs[()]):
        translation = qd.Vector(
            [
                data.constraint[pair, 0],
                data.constraint[pair, 1],
                data.constraint[pair, 2],
            ]
        )
        rotation = qd.Vector(
            [
                data.constraint[pair, 3],
                data.constraint[pair, 4],
                data.constraint[pair, 5],
            ]
        )
        residual = translation.norm() + data.surface_radius[pair] * rotation.norm()
        limit = qd.max(0.1 * contact.d_hat[()], residual)
        data.path_limit[pair] = limit
        proxy_rotation = qd.Vector.zero(qd.f64, 3)
        slack_translation = qd.Vector.zero(qd.f64, 3)
        slack_rotation = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            proxy_rotation[axis] = data.dq[pair, axis + 3]
            slack_translation[axis] = data.slack[pair, axis]
            slack_rotation[axis] = data.slack[pair, axis + 3]
        proxy_curvature = proxy_rotation.dot(proxy_rotation) * data.surface_radius[pair]
        forest_curvature = curvature_bound(
            forest,
            rigid,
            data,
            data.mechanism_body[pair],
            data.surface_radius[pair],
        )
        slack_residual = slack_translation.norm() + data.surface_radius[pair] * slack_rotation.norm()
        alpha = fk_defect_prefix_cap(
            residual,
            slack_residual,
            proxy_curvature + forest_curvature,
            limit,
        )
        if alpha < 1.0:
            qd.atomic_min(data.fk_alpha[()], alpha)


@qd.func(requires_top_level=True)
def evaluate_trial_guard(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for _ in range(1):
        data.trial_max_residual[()] = 0.0
    for pair in range(data.n_pairs[()]):
        mechanism = data.mechanism_body[pair]
        mechanism_position = qd.Vector.zero(qd.f64, 3)
        mechanism_quaternion = qd.Vector.zero(qd.f64, 4)
        proxy_position = qd.Vector.zero(qd.f64, 3)
        proxy_quaternion = qd.Vector.zero(qd.f64, 4)
        for axis in qd.static(range(3)):
            mechanism_position[axis] = forest.endpoint_t[mechanism][axis]
            proxy_position[axis] = data.t[pair, axis]
        for axis in qd.static(range(4)):
            mechanism_quaternion[axis] = forest.endpoint_quat[mechanism][axis]
            proxy_quaternion[axis] = data.quat[pair, axis]
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
            data.trial_constraint[pair, component] = translation[component]
            data.trial_constraint[pair, component + 3] = rotation[component]
        residual = translation.norm() + data.surface_radius[pair] * rotation.norm()
        qd.atomic_max(data.trial_max_residual[()], residual)
        tolerance = 1.0e-10 * (1.0 + data.path_limit[pair])
        if residual > data.path_limit[pair] + tolerance:
            data.frame_failed[()] = 1


@qd.func(requires_top_level=True)
def compute_restoration_energy(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    use_trial: qd.template(),  # bool
):
    for _ in range(1):
        data.restoration_energy[()] = 0.0
    for pair in range(data.n_pairs[()]):
        if data.restoration_active[()] != 0:
            constraint = qd.Vector.zero(qd.f64, 6)
            dual = qd.Vector.zero(qd.f64, 6)
            metric = qd.Matrix.zero(qd.f64, 6, 6)
            for row in qd.static(range(6)):
                if qd.static(use_trial):
                    constraint[row] = data.trial_constraint[pair, row]
                else:
                    constraint[row] = data.constraint[pair, row]
                dual[row] = data.lambda_[pair, row]
                for column in qd.static(range(6)):
                    metric[row, column] = data.metric[pair, row, column]
            value = dual.dot(constraint) + 0.5 * constraint.dot(metric @ constraint)
            qd.atomic_add(data.restoration_energy[()], value)


@qd.func
def trigger_restoration(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    triggered = False
    if (
        data.restoration_enabled[()] != 0
        and data.restoration_active[()] == 0
        and data.max_surface_residual[()] > data.solve_tolerance[()]
    ):
        data.restoration_active[()] = 1
        data.restoration_triggered[()] = 1
        data.restoration_hard_probe[()] = 0
        data.restoration_entries[()] = data.restoration_entries[()] + 1
        data.restoration_reprice_flag[()] = 1
        data.physical_converged[()] = 0
        triggered = True
    return triggered


@qd.func(requires_top_level=True)
def finalize_restoration_step(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    alpha,
):
    for _ in range(1):
        if data.restoration_active[()] != 0 and data.restoration_reprice_flag[()] == 0 and alpha > 0.0:
            data.dual_update_flag[()] = qd.i32(data.trial_max_residual[()] > data.solve_tolerance[()])
            if data.dual_update_flag[()] != 0:
                data.restoration_dual_epochs[()] = data.restoration_dual_epochs[()] + 1
    for pair in range(data.n_pairs[()]):
        if (
            data.restoration_active[()] != 0
            and data.restoration_reprice_flag[()] == 0
            and alpha > 0.0
            and data.dual_update_flag[()] != 0
        ):
            constraint = qd.Vector.zero(qd.f64, 6)
            metric = qd.Matrix.zero(qd.f64, 6, 6)
            for row in qd.static(range(6)):
                constraint[row] = data.trial_constraint[pair, row]
                for column in qd.static(range(6)):
                    metric[row, column] = data.metric[pair, row, column]
            update = metric @ constraint
            for component in qd.static(range(6)):
                data.lambda_[pair, component] = data.lambda_[pair, component] + update[component]
    for _ in range(1):
        if data.restoration_active[()] != 0 and data.restoration_reprice_flag[()] == 0 and alpha > 0.0:
            data.restoration_active[()] = 0
            data.restoration_hard_probe[()] = 1
            data.restoration_reprice_flag[()] = 1


@qd.func(requires_top_level=True)
def recover_reaction(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for pair in range(data.n_pairs[()]):
        offset = forest.proxy_dof_offset[()] + pair * 6
        rotation = qd.Vector.zero(qd.f64, 3)
        gradient_translation = qd.Vector.zero(qd.f64, 3)
        gradient_rotation = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            rotation[axis] = data.constraint[pair, axis + 3]
            gradient_translation[axis] = data.merit_gradient[offset + axis]
            gradient_rotation[axis] = data.merit_gradient[offset + axis + 3]
        reaction_rotation = -(so3_left_jacobian(rotation).transpose() @ gradient_rotation)
        for axis in qd.static(range(3)):
            data.reaction[pair, axis] = -gradient_translation[axis]
            data.reaction[pair, axis + 3] = reaction_rotation[axis]


@qd.func(requires_top_level=True)
def prepare_constraint(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for _ in range(1):
        data.max_surface_residual[()] = 0.0
    for pair in range(data.n_pairs[()]):
        mechanism = data.mechanism_body[pair]
        mechanism_position = qd.Vector.zero(qd.f64, 3)
        mechanism_quaternion = qd.Vector.zero(qd.f64, 4)
        proxy_position = qd.Vector.zero(qd.f64, 3)
        proxy_quaternion = qd.Vector.zero(qd.f64, 4)
        for axis in qd.static(range(3)):
            mechanism_position[axis] = forest.endpoint_t[mechanism][axis]
            proxy_position[axis] = data.t[pair, axis]
        for axis in qd.static(range(4)):
            mechanism_quaternion[axis] = forest.endpoint_quat[mechanism][axis]
            proxy_quaternion[axis] = data.quat[pair, axis]

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
            data.constraint[pair, row] = constraint[row]
            data.particular[pair, row] = particular[row]
            for column in qd.static(range(6)):
                data.tangent_map[pair, row, column] = tangent_map[row, column]
                data.normal_map[pair, row, column] = normal_map[row, column]

        residual = translation.norm() + data.surface_radius[pair] * rotation.norm()
        qd.atomic_max(data.max_surface_residual[()], residual)


@qd.func(requires_top_level=True)
def initialize_global_vertices(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    vertex: qd.template(),  # GlobalVertexManager.Data
):
    for local_vertex in range(data.n_verts[()]):
        pair = data.vertex_pair[local_vertex]
        local_position = qd.Vector.zero(qd.f64, 3)
        quaternion = qd.Vector.zero(qd.f64, 4)
        translation = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            local_position[axis] = data.local_positions[local_vertex, axis]
            translation[axis] = data.t[pair, axis]
        for axis in qd.static(range(4)):
            quaternion[axis] = data.quat[pair, axis]
        world_position = translation + gu.qd_transform_by_quat(local_position, quaternion)
        global_vertex = data.global_vert_offset[()] + local_vertex
        vertex.body_id[global_vertex] = data.global_body_offset[()] + data.proxy_body[pair]
        for axis in qd.static(range(3)):
            vertex.positions[global_vertex, axis] = world_position[axis]
            vertex.safe_positions[global_vertex, axis] = world_position[axis]
            vertex.trajectory_end_positions[global_vertex, axis] = world_position[axis]
            vertex.x_bar[global_vertex, axis] = world_position[axis]


@qd.func(requires_top_level=True)
def forward_global_vertices(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    vertex: qd.template(),  # GlobalVertexManager.Data
):
    for local_vertex in range(data.n_verts[()]):
        pair = data.vertex_pair[local_vertex]
        local_position = qd.Vector.zero(qd.f64, 3)
        quaternion = qd.Vector.zero(qd.f64, 4)
        translation = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            local_position[axis] = data.local_positions[local_vertex, axis]
            translation[axis] = data.t[pair, axis]
        for axis in qd.static(range(4)):
            quaternion[axis] = data.quat[pair, axis]
        world_position = translation + gu.qd_transform_by_quat(local_position, quaternion)
        global_vertex = data.global_vert_offset[()] + local_vertex
        for axis in qd.static(range(3)):
            vertex.positions[global_vertex, axis] = world_position[axis]


@qd.func(requires_top_level=True)
def publish_trajectory_end_positions(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    vertex: qd.template(),  # GlobalVertexManager.Data
):
    for local_vertex in range(data.n_verts[()]):
        pair = data.vertex_pair[local_vertex]
        global_vertex = data.global_vert_offset[()] + local_vertex
        pivot = qd.Vector.zero(qd.f64, 3)
        pivot_displacement = qd.Vector.zero(qd.f64, 3)
        rotation = qd.Vector.zero(qd.f64, 3)
        start = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            pivot[axis] = data.t[pair, axis]
            pivot_displacement[axis] = data.dq[pair, axis]
            rotation[axis] = data.dq[pair, axis + 3]
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
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    vertex: qd.template(),  # GlobalVertexManager.Data
    max_disp: qd.template(),  # qd.Ndarray
):
    for local_vertex in range(data.n_verts[()]):
        pair = data.vertex_pair[local_vertex]
        global_vertex = data.global_vert_offset[()] + local_vertex
        pivot = qd.Vector.zero(qd.f64, 3)
        pivot_displacement = qd.Vector.zero(qd.f64, 3)
        rotation = qd.Vector.zero(qd.f64, 3)
        start = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            pivot[axis] = data.t[pair, axis]
            pivot_displacement[axis] = data.dq[pair, axis]
            rotation[axis] = data.dq[pair, axis + 3]
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
def record_start_point(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for pair in range(data.n_pairs[()]):
        for axis in qd.static(range(3)):
            data.t_temp[pair, axis] = data.t[pair, axis]
        for axis in qd.static(range(4)):
            data.quat_temp[pair, axis] = data.quat[pair, axis]


@qd.func(requires_top_level=True)
def step_forward(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
    alpha,
):
    for pair in range(data.n_pairs[()]):
        angular = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            data.t[pair, axis] = data.t_temp[pair, axis] + alpha * data.dq[pair, axis]
            angular[axis] = alpha * data.dq[pair, axis + 3]
        delta_quaternion = gu.qd_rotvec_to_quat(angular, qd.f64(1.0e-12))
        start_quaternion = qd.Vector.zero(qd.f64, 4)
        for axis in qd.static(range(4)):
            start_quaternion[axis] = data.quat_temp[pair, axis]
        trial_quaternion = gu.qd_quat_mul(delta_quaternion, start_quaternion)
        for axis in qd.static(range(4)):
            data.quat[pair, axis] = trial_quaternion[axis]


@qd.func(requires_top_level=True)
def copy_previous_state(
    data: qd.template(),  # RigidContactProxySystem.Data
    rigid: qd.template(),  # RigidSystem.Data
    forest: qd.template(),  # RigidJointForestSystem.Data
):
    for pair in range(data.n_pairs[()]):
        for axis in qd.static(range(3)):
            data.t_prev[pair, axis] = data.t[pair, axis]
        for axis in qd.static(range(4)):
            data.quat_prev[pair, axis] = data.quat[pair, axis]
