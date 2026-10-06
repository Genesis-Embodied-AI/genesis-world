from __future__ import annotations

from enum import IntEnum
from typing import TypeVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .bcoo_matrix import grow_bcoo_matrix, sort_reduce_bcoo, zero_bcoo_triplets
from .consistent_ipc_contact import ConsistentIPCContactConstitution
from .contact import CONTACT_CONFIG_DEFAULTS
from .contact_system import ContactSystem
from .finite_element import FEMDiagPreconditioner, FiniteElementMethod
from .finite_element.fem_diag_preconditioner import (
    gather_fem_diag_preconditioner,
    initialize_fem_diag_preconditioner,
    invert_fem_diag_preconditioner,
)
from .finite_element.fem_contact_assemble import (
    distribute_fem_fem_kernel,
    distribute_fem_gradient_kernel,
)
from .finite_element.finite_element_method import (
    contribute_fem_newton_max_disp,
    copy_fem_x_prev,
    forward_fem_global_vertices,
    forward_fem_scene_vertices,
    initialize_fem_global_vertices,
    negate_fem_dx,
    publish_fem_trajectory_end_positions,
    record_fem_start_point,
    reset_fem_energy,
    step_fem_forward,
    sync_fem_from_scene,
    update_fem_velocity,
)
from .global_body_manager import GlobalBodyManager, compute_vertex_offsets
from .global_linear_system import (
    GlobalLinearSystem,
    compute_n_triplets,
    derive_extents,
    zero_rhs,
)
from .global_surface_manager import GlobalSurfaceManager
from .global_vertex_manager import (
    GlobalVertexManager,
    record_safe_positions,
    reset_trajectory,
    zero_in_contact,
)
from .lbvh_broad_phase import InfoLBVHBatchedBroadPhaseDop14, LBVHBroadPhase
from .rigid_contact_assemble import (
    RigidContactAssemble,
    classify as classify_rigid_contact,
    distribute as distribute_rigid_contact,
)
from .rigid_contact_proxy import (
    RigidContactProxySystem,
    apply_convergence as apply_proxy_convergence,
    capture_physical_gradient,
    check_line_search as check_proxy_line_search,
    compute_restoration_energy,
    contribute_newton_max_disp as contribute_proxy_newton_max_disp,
    copy_previous_state as copy_proxy_previous_state,
    evaluate_trial_guard,
    finalize_restoration_step,
    forward_global_vertices as forward_proxy_global_vertices,
    initialize_global_vertices as initialize_proxy_global_vertices,
    initialize_merit,
    initialize_newton as initialize_proxy_newton,
    initialize_proxy_state,
    mark_mechanism_constrained,
    prepare_constraint,
    prepare_metric,
    prepare_path_limit,
    prepare_tolerance,
    publish_trajectory_end_positions as publish_proxy_trajectory_end_positions,
    record_start_point as record_proxy_start_point,
    recover_reaction,
    reset_frame as reset_proxy_frame,
    step_forward as step_proxy_forward,
)
from .rigid_joint_forest import (
    RigidJointForestSystem,
    build_preconditioner as build_forest_preconditioner,
    compute_endpoint_fk,
    compute_merit_directional_derivative,
    expand_solution,
    particular_spmv,
    prepare_particular,
    project_physical_rhs,
)
from .rigid_system import (
    RigidSystem,
    assemble as assemble_rigid,
    assemble_candidate_rows,
    build_preconditioner as build_rigid_preconditioner,
    energy as compute_rigid_energy,
    initialize_newton as initialize_rigid_newton,
    negate_dq,
    predict as predict_rigid,
    record_start_point as record_rigid_start_point,
    set_newton_active,
    step_forward as step_rigid_forward,
    update_velocity as update_rigid_velocity,
)
from .pcg_solver import PCGSolver, solve_pcg
from .sim_config import SimConfig
from .sim_system import ActionInvocation, SimData, SimPipeline, SimSystem

T = TypeVar("T", bound=SimSystem)


class ContactCheckpoint(IntEnum):
    FRAME = 0
    FRICTION = 1
    COUNT = 2
    FILTER = 3
    SORT = 4
    SOLVE = 5
    QUERY = 6
    CCD = 7
    LINE_SEARCH = 8
    KKT_FAILURE = 9
    FINALIZE = 10
    INITIAL_INTERSECTION = 11
    ET_OVERFLOW = 12
    ET_FAILURE = 13
    SOLVE_POST = 14
    LINE_SEARCH_TRIAL = 15
    PCG_SOLVE = 16
    LINE_SEARCH_POST = 19


class SimEngine:
    """Python host coordinator for the graph-native Newton runtime."""

    @qd.data_oriented
    class Data(SimData):
        """Complete mutable graph root for one initialized engine."""

        newton_cond: qd.Ndarray
        ls_cond: qd.Ndarray
        converged: qd.Ndarray
        frame_failed: qd.Ndarray
        newton_iter: qd.Ndarray
        max_pcg_iters: qd.Ndarray
        total_pcg_iters: qd.Ndarray
        ls_iter: qd.Ndarray
        alpha: qd.Ndarray
        max_disp: qd.Ndarray
        energy_delta: qd.Ndarray
        checkpoint_never_yield: qd.Ndarray
        energy_buf: qd.Ndarray
        sim_config_data: SimConfig.Data
        rigid_data: RigidSystem.Data | None
        fem_data: FiniteElementMethod.Data | None
        global_body_data: GlobalBodyManager.Data | None
        global_vertex_data: GlobalVertexManager.Data | None
        global_surface_data: GlobalSurfaceManager.Data | None
        contact_data: ContactSystem.Data | None
        rigid_contact_proxy_data: RigidContactProxySystem.Data | None
        rigid_contact_assemble_data: RigidContactAssemble.Data | None
        rigid_forest_data: RigidJointForestSystem.Data | None
        global_linear_system_data: GlobalLinearSystem.Data
        fem_preconditioner_data: FEMDiagPreconditioner.Data | None
        pcg_solver_data: PCGSolver.Data
        has_rigid: bool
        has_fem: bool
        has_contact: bool
        has_rigid_contact_proxy: bool
        has_rigid_contact_assemble: bool
        has_rigid_forest: bool
        has_et_check: bool
        genesis_serial_pipeline: bool
        fem_predict_actions: tuple[ActionInvocation, ...]
        fem_extent_actions: tuple[ActionInvocation, ...]
        fem_assemble_actions: tuple[ActionInvocation, ...]
        fem_energy_actions: tuple[ActionInvocation, ...]
        pcg_solver_invocations: tuple[ActionInvocation, ...]
        pcg_operator_actions: tuple[ActionInvocation, ...]
        pcg_preconditioner_actions: tuple[ActionInvocation, ...]
        contact_reset_initial_intersections_action: ActionInvocation
        contact_flag_et_intersections_action: ActionInvocation
        contact_reset_counted_demand_action: ActionInvocation
        contact_adaptive_kappa_update_action: ActionInvocation
        contact_adaptive_kappa_newton_tick_action: ActionInvocation
        contact_reset_collision_counts_action: ActionInvocation
        contact_halfplane_query_action: ActionInvocation
        contact_init_ccd_action: ActionInvocation
        contact_reset_frame_ccd_action: ActionInvocation
        contact_ccd_alpha_pt_action: ActionInvocation
        contact_ccd_alpha_ee_action: ActionInvocation
        contact_ccd_alpha_ph_action: ActionInvocation
        contact_reduce_ccd_alpha_action: ActionInvocation
        contact_ccd_action: ActionInvocation
        contact_reset_contact_energy_action: ActionInvocation
        contact_sum_contact_energy_action: ActionInvocation
        contact_check_assembly_capacity_action: ActionInvocation
        contact_check_assembly_padding_action: ActionInvocation
        contact_shrink_assembly_padding_action: ActionInvocation
        contact_reset_assembly_counts_action: ActionInvocation
        contact_sort_reduce_action: ActionInvocation
        broad_phase_triangle_build_action: ActionInvocation
        broad_phase_edge_build_action: ActionInvocation
        broad_phase_pt_query_action: ActionInvocation
        broad_phase_ee_query_action: ActionInvocation
        broad_phase_trajectory_query_action: ActionInvocation
        broad_phase_detect_initial_intersections_action: ActionInvocation
        contact_constitution_snapshot_lagged_positions_action: ActionInvocation
        contact_constitution_friction_pair_filter_pt_action: ActionInvocation
        contact_constitution_friction_pair_filter_ee_action: ActionInvocation
        contact_constitution_friction_pair_filter_ph_action: ActionInvocation
        contact_constitution_friction_snapshot_action: ActionInvocation
        contact_constitution_count_active_pt_action: ActionInvocation
        contact_constitution_count_active_ee_action: ActionInvocation
        contact_constitution_count_active_ph_action: ActionInvocation
        contact_constitution_count_active_action: ActionInvocation
        contact_constitution_filter_assemble_pt_action: ActionInvocation
        contact_constitution_filter_assemble_ee_action: ActionInvocation
        contact_constitution_filter_assemble_ph_action: ActionInvocation
        contact_constitution_friction_assemble_pt_action: ActionInvocation
        contact_constitution_friction_assemble_ee_action: ActionInvocation
        contact_constitution_friction_assemble_ph_action: ActionInvocation
        contact_constitution_filter_assemble_action: ActionInvocation
        contact_constitution_filter_energy_pt_action: ActionInvocation
        contact_constitution_filter_energy_ee_action: ActionInvocation
        contact_constitution_filter_energy_ph_action: ActionInvocation
        contact_constitution_friction_energy_pt_action: ActionInvocation
        contact_constitution_friction_energy_ee_action: ActionInvocation
        contact_constitution_friction_energy_ph_action: ActionInvocation
        contact_constitution_contact_energy_action: ActionInvocation

    def __init__(self) -> None:
        if gs.backend == gs.cpu:
            raise RuntimeError("SimEngine does not support the CPU backend")
        if not gs.use_ndarray:
            raise RuntimeError("SimEngine requires the ndarray backend")

        self.systems: dict[type, SimSystem] = {}
        self._dependencies: dict[SimSystem, bool] = {}
        self.is_built_host = False
        self.is_initialized_host = False
        self.params_wired_host = False
        self.genesis_serial_pipeline = False
        self._visualizer_server = None
        self.data: SimEngine.Data | None = None
        self.init_contact_pipeline: SimPipeline | None = None
        self.step_pipeline: SimPipeline | None = None
        self.rigid = None
        self.fem_system = None
        self.global_body_system = None
        self.global_vertex_system = None
        self.global_surface_system = None
        self.contact_system = None
        self.broad_phase_system = None
        self.contact_constitution_system = None
        self.rigid_contact_proxy = None
        self.rigid_contact_assemble = None
        self.rigid_forest = None
        self.global_linear_system_system = None
        self.pcg_solver_system = None
        self.fem_preconditioner_system = None

        self.sim_config_system = SimConfig()
        self.add_system(self.sim_config_system)

    def configure_genesis_serial_pipeline(self, enabled: bool) -> None:
        if self.is_built_host:
            raise RuntimeError("Pipeline scheduling must be configured before build_systems()")
        self.genesis_serial_pipeline = bool(enabled)

    def start_visualizer(self, host: str = "127.0.0.1", port: int = 0):
        """Start the read-only system architecture website."""
        from .visualizer import start_server

        server = self._visualizer_server
        if server is not None and server.is_running:
            return server
        self._visualizer_server = start_server(self, host=host, port=port)
        return self._visualizer_server

    def stop_visualizer(self) -> None:
        """Stop the architecture website if it is running."""
        server = self._visualizer_server
        if server is not None:
            server.stop()
            self._visualizer_server = None

    def add_system(self, system: SimSystem) -> None:
        if self.is_built_host:
            raise RuntimeError("add_system() is only valid before build_systems()")
        system_type = type(system)
        if system_type in self.systems:
            raise RuntimeError(f"SimSystem {system_type.__name__} is already registered")
        system._set_engine(self)
        self.systems[system_type] = system

    @property
    def fem(self) -> FiniteElementMethod | None:
        """Compatibility view of the organizational FEM system."""
        return self.fem_system

    def _require_runtime_data(self) -> Data:
        data = self.data
        if data is None:
            raise RuntimeError("SimEngine graph data is not initialized")
        return data

    @property
    def sim_config_data(self):
        system = self.sim_config_system
        return None if system is None else system.data

    @property
    def rigid_data(self):
        system = self.rigid
        return None if system is None else system.data

    @property
    def fem_data(self):
        system = self.fem_system
        return None if system is None else system.data

    @property
    def global_body_data(self):
        system = self.global_body_system
        return None if system is None else system.data

    @property
    def global_vertex_data(self):
        system = self.global_vertex_system
        return None if system is None else system.data

    @property
    def global_surface_data(self):
        system = self.global_surface_system
        return None if system is None else system.data

    @property
    def contact_data(self):
        system = self.contact_system
        return None if system is None else system.data

    @property
    def rigid_contact_proxy_data(self):
        system = self.rigid_contact_proxy
        return None if system is None else system.data

    @property
    def rigid_contact_assemble_data(self):
        system = self.rigid_contact_assemble
        return None if system is None else system.data

    @property
    def rigid_forest_data(self):
        system = self.rigid_forest
        return None if system is None else system.data

    @property
    def global_linear_system_data(self):
        system = self.global_linear_system_system
        return None if system is None else system.data

    @property
    def fem_preconditioner_data(self):
        system = self.fem_preconditioner_system
        return None if system is None else system.data

    @property
    def pcg_solver_data(self):
        system = self.pcg_solver_system
        return None if system is None else system.data

    @property
    def newton_cond(self):
        return self._require_runtime_data().newton_cond

    @property
    def ls_cond(self):
        return self._require_runtime_data().ls_cond

    @property
    def converged(self):
        return self._require_runtime_data().converged

    @property
    def frame_failed(self):
        return self._require_runtime_data().frame_failed

    @property
    def newton_iter(self):
        return self._require_runtime_data().newton_iter

    @property
    def max_pcg_iters(self):
        return self._require_runtime_data().max_pcg_iters

    @property
    def total_pcg_iters(self):
        return self._require_runtime_data().total_pcg_iters

    @property
    def ls_iter(self):
        return self._require_runtime_data().ls_iter

    @property
    def alpha(self):
        return self._require_runtime_data().alpha

    @property
    def max_disp(self):
        return self._require_runtime_data().max_disp

    @property
    def energy_delta(self):
        return self._require_runtime_data().energy_delta

    @property
    def checkpoint_never_yield(self):
        return self._require_runtime_data().checkpoint_never_yield

    @property
    def energy_buf(self):
        return self._require_runtime_data().energy_buf

    def find(self, system_type: type[T]) -> T | None:
        system = self.systems.get(system_type)
        if system is None or not system.is_valid:
            return None
        self._dependencies.setdefault(system, False)
        return system

    def require(self, system_type: type[T]) -> T:
        system = self.systems.get(system_type)
        if system is not None and not system.is_valid:
            system = None
        if system is None:
            raise RuntimeError(f"Required system {system_type.__name__} is not registered")
        self._dependencies[system] = True
        return system

    def dependencies(self) -> dict[SimSystem, bool]:
        """Return engine dependencies mapped to require/find strength."""
        return dict(self._dependencies)

    def build_systems(self) -> None:
        if self.is_built_host:
            raise RuntimeError("SimEngine systems are already built")
        systems = tuple(self.systems.values())
        for system in systems:
            system._begin_build()
        for system in systems:
            system.build()
        self.build()
        for system in systems:
            system._end_build()

    def build(self) -> None:
        if self.is_built_host:
            raise RuntimeError("SimEngine is already built")
        self.rigid = self.find(RigidSystem)
        self.fem_system = self.find(FiniteElementMethod)
        self.global_body_system = self.find(GlobalBodyManager)
        self.global_vertex_system = self.find(GlobalVertexManager)
        self.global_surface_system = self.find(GlobalSurfaceManager)
        self.contact_system = self.find(ContactSystem)
        self.broad_phase_system = self.find(InfoLBVHBatchedBroadPhaseDop14)
        if self.broad_phase_system is None:
            self.broad_phase_system = self.find(LBVHBroadPhase)
        self.contact_constitution_system = self.find(ConsistentIPCContactConstitution)
        self.rigid_contact_proxy = self.find(RigidContactProxySystem)
        self.rigid_contact_assemble = self.find(RigidContactAssemble)
        self.rigid_forest = self.find(RigidJointForestSystem)
        self.global_linear_system_system = self.require(GlobalLinearSystem)
        self.pcg_solver_system = self.require(PCGSolver)
        self.fem_preconditioner_system = self.find(FEMDiagPreconditioner)
        if self.fem_system is not None and self.fem_preconditioner_system is None:
            raise RuntimeError("FiniteElementMethod requires FEMDiagPreconditioner")
        if self.fem_system is not None and (
            self.global_body_data is None or self.global_vertex_data is None or self.global_surface_data is None
        ):
            raise RuntimeError("FiniteElementMethod requires global body, vertex, and surface managers")

        self.has_rigid = self.rigid is not None
        self.has_fem = self.fem_system is not None
        self.has_contact = self.contact_system is not None
        self.has_rigid_contact_proxy = self.rigid_contact_proxy is not None
        self.has_rigid_contact_assemble = self.rigid_contact_assemble is not None
        self.has_rigid_forest = self.rigid_forest is not None
        self.has_et_check = self.contact_data is not None and self.contact_data.intersection_check_host
        if self.has_rigid_contact_proxy != self.has_rigid_forest:
            raise RuntimeError("RigidContactProxySystem and RigidJointForestSystem must be registered together")
        if self.has_rigid_contact_proxy != self.has_rigid_contact_assemble:
            raise RuntimeError("RigidContactProxySystem and RigidContactAssemble must be registered together")
        if self.has_contact and not self.has_fem:
            raise RuntimeError("The current ContactSystem milestone requires FiniteElementMethod")
        if self.has_contact and (self.broad_phase_system is None or self.contact_constitution_system is None):
            raise RuntimeError("ContactSystem requires broad-phase and contact constitution systems")
        if not self.has_rigid and not self.has_fem:
            raise RuntimeError("SimEngine requires RigidSystem or FiniteElementMethod")
        self.is_built_host = True

    def _make_graph_fastcache_key(self) -> int:
        """Encode Python-static graph topology that Quadrants cannot infer from ndarray arguments."""
        contact = self.contact_data
        broad_phase = None if self.broad_phase_system is None else self.broad_phase_system.data
        flags = (
            self.has_rigid,
            self.has_fem,
            self.has_contact,
            self.has_rigid_contact_proxy,
            self.has_rigid_contact_assemble,
            self.has_rigid_forest,
            self.has_et_check,
            self.genesis_serial_pipeline,
            contact is not None and contact.has_halfplanes,
            contact is not None and contact.has_friction,
            self.rigid is not None and self.rigid.has_constraints,
            self.rigid is not None and self.rigid.has_collision,
            broad_phase is not None and broad_phase.use_warp_pt,
            broad_phase is not None and broad_phase.use_dual_ee,
            broad_phase is not None and broad_phase.bound_type == "dop14",
            broad_phase is not None and broad_phase.genesis_legacy_sort_reduce,
            broad_phase is not None and broad_phase.genesis_legacy_fp64_bounds,
            broad_phase is not None and broad_phase.genesis_legacy_refit,
            self.global_linear_system_data.matrix.genesis_legacy_sort_reduce_host,
            contact is not None and contact.genesis_legacy_sort_reduce_host,
            self.rigid_forest is not None and self.rigid_forest.fused_enabled,
            self.rigid_forest is not None and self.rigid_forest.genesis_legacy_enabled,
        )
        return sum(int(flag) << index for index, flag in enumerate(flags))

    def wire_solver_params(
        self,
        dt: float,
        tol: float,
        max_newton_iter: int,
        max_pcg_iter: int,
        max_ls_iter: int,
        pcg_tol_rate: float,
    ) -> None:
        if not self.is_built_host:
            raise RuntimeError("build_systems() must run before wire_solver_params()")
        self.sim_config_data.dt.from_numpy(np.array(dt, dtype=np.float64))
        self.sim_config_data.tol.from_numpy(np.array(tol, dtype=np.float64))
        self.sim_config_data.max_newton_iter.from_numpy(np.array(max_newton_iter, dtype=np.int64))
        self.sim_config_data.max_pcg_iter.from_numpy(np.array(max_pcg_iter, dtype=np.int64))
        self.sim_config_data.max_ls_iter.from_numpy(np.array(max_ls_iter, dtype=np.int64))
        self.max_ls_iter_host = max_ls_iter
        self.pcg_tol_rate_host = pcg_tol_rate
        self.params_wired_host = True

    def init(self) -> None:
        if self.is_initialized_host:
            raise RuntimeError("SimEngine is already initialized")
        if not self.params_wired_host:
            raise RuntimeError("wire_solver_params() must run before init()")

        dof_offset = 0
        if self.rigid is not None:
            self.rigid.init(dof_offset)
            dof_offset += self.rigid.storage_dof_count_host
        dof_block_base = dof_offset // 3

        n_elastic_triplets = 0
        if self.fem_system is not None:
            self.fem_system.init(dof_offset)
            dof_offset += self.fem_data.vert_capacity_host * 3
            n_elastic_triplets = self.fem_system.n_elastic_triplets()
            (
                fem_predict_actions,
                fem_extent_actions,
                fem_assemble_actions,
                fem_energy_actions,
            ) = self.fem_system.resolve_actions()
        else:
            fem_predict_actions = ()
            fem_extent_actions = ()
            fem_assemble_actions = ()
            fem_energy_actions = ()

        if self.rigid_contact_proxy is not None:
            dof_offset += self.rigid_contact_proxy.n_pairs_host * 6

        n_block_rows = dof_offset // 3
        max_contact_body_triplets = (
            self.contact_data.unique_triplet_rows.shape[0] if self.contact_data is not None else 0
        )
        if self.rigid_contact_assemble is not None:
            max_contact_body_triplets = (
                self.contact_data.unique_triplet_rows.shape[0] * 4 + self.contact_data.unique_doublet_vertices.shape[0]
            )
        linear_data = self.global_linear_system_data
        if (
            linear_data.n_block_rows_host != n_block_rows
            or linear_data.total_dof_host != dof_offset
            or linear_data.matrix.triplet_row.shape[0] < n_elastic_triplets + max_contact_body_triplets
        ):
            raise RuntimeError("GlobalLinearSystem data was not sized for the built simulation")
        self.pcg_solver_system.init(dof_offset, n_block_rows, self.pcg_tol_rate_host)
        pcg_solver_invocations = self.pcg_solver_system.solver_invocations
        pcg_operator_actions = self.pcg_solver_system.operator_actions
        pcg_preconditioner_actions = self.pcg_solver_system.preconditioner_actions
        scheduled_actions = (
            *fem_predict_actions,
            *fem_extent_actions,
            *fem_assemble_actions,
            *fem_energy_actions,
            *pcg_solver_invocations,
            *pcg_operator_actions,
            *pcg_preconditioner_actions,
        )
        contact_actions: dict[str, ActionInvocation] = {}
        broad_phase_actions: dict[str, ActionInvocation] = {}
        constitution_actions: dict[str, ActionInvocation] = {}
        if self.contact_system is not None:
            contact_actions = self.contact_system.resolve_actions()
            broad_phase_actions = self.broad_phase_system.resolve_actions()
            constitution_actions = self.contact_constitution_system.resolve_actions()
            contact_invocations = (
                *contact_actions.values(),
                *broad_phase_actions.values(),
                *constitution_actions.values(),
            )
            scheduled_actions = (*scheduled_actions, *contact_invocations)

        self.data = get_sim_engine_data(
            max_ls_iter=self.max_ls_iter_host,
            sim_config_data=self.sim_config_data,
            rigid_data=self.rigid_data,
            fem_data=self.fem_data,
            global_body_data=self.global_body_data,
            global_vertex_data=self.global_vertex_data,
            global_surface_data=self.global_surface_data,
            contact_data=self.contact_data,
            rigid_contact_proxy_data=self.rigid_contact_proxy_data,
            rigid_contact_assemble_data=self.rigid_contact_assemble_data,
            rigid_forest_data=self.rigid_forest_data,
            global_linear_system_data=self.global_linear_system_data,
            fem_preconditioner_data=self.fem_preconditioner_data,
            pcg_solver_data=self.pcg_solver_data,
            has_rigid=self.has_rigid,
            has_fem=self.has_fem,
            has_contact=self.has_contact,
            has_rigid_contact_proxy=self.has_rigid_contact_proxy,
            has_rigid_contact_assemble=self.has_rigid_contact_assemble,
            has_rigid_forest=self.has_rigid_forest,
            has_et_check=self.has_et_check,
            genesis_serial_pipeline=self.genesis_serial_pipeline,
            fem_predict_actions=fem_predict_actions,
            fem_extent_actions=fem_extent_actions,
            fem_assemble_actions=fem_assemble_actions,
            fem_energy_actions=fem_energy_actions,
            pcg_solver_invocations=pcg_solver_invocations,
            pcg_operator_actions=pcg_operator_actions,
            pcg_preconditioner_actions=pcg_preconditioner_actions,
            contact_actions=contact_actions,
            broad_phase_actions=broad_phase_actions,
            constitution_actions=constitution_actions,
        )
        init_contact_pipeline = SimPipeline(self.data, _init_contact_kernel)
        init_contact_pipeline.register_yield_callback(
            ContactCheckpoint.INITIAL_INTERSECTION,
            self._handle_initial_intersection_yield,
        )
        init_contact_pipeline.register_yield_callback(
            ContactCheckpoint.QUERY,
            self._handle_init_query_yield,
        )
        step_pipeline = SimPipeline(self.data, _step_kernel)
        for checkpoint in (
            ContactCheckpoint.FRICTION,
            ContactCheckpoint.COUNT,
            ContactCheckpoint.FILTER,
            ContactCheckpoint.SORT,
            ContactCheckpoint.QUERY,
            ContactCheckpoint.ET_OVERFLOW,
            ContactCheckpoint.ET_FAILURE,
            ContactCheckpoint.KKT_FAILURE,
        ):
            step_pipeline.register_yield_callback(checkpoint, self._handle_step_yield)
        init_contact_pipeline.bind_actions(*scheduled_actions)
        step_pipeline.bind_actions(*scheduled_actions)
        self.init_contact_pipeline = init_contact_pipeline
        self.step_pipeline = step_pipeline
        self.graph_fastcache_key = self._make_graph_fastcache_key()
        _initialize_global_resources(self.data, self.graph_fastcache_key)
        if self.contact_system is not None:
            self._initialize_contact()
        self.is_initialized_host = True

    def sync_from_solvers(self) -> None:
        """Synchronize externally authored Scene state without rebuilding graph resources."""
        _sync_from_solvers_kernel(self._require_runtime_data(), self.graph_fastcache_key)
        if self.contact_system is not None:
            self._initialize_contact()

    def _handle_pair_overflow(self) -> None:
        self.contact_system.handle_broad_phase_overflow()
        self.contact_system.realloc_pair_buffers(
            pt=int(qd_to_numpy(self.contact_data.n_pairs_pt)),
            ee=int(qd_to_numpy(self.contact_data.n_pairs_ee)),
            pe=int(qd_to_numpy(self.contact_data.n_pairs_pe)),
            pp=int(qd_to_numpy(self.contact_data.n_pairs_pp)),
            ph=int(qd_to_numpy(self.contact_data.n_pairs_ph)),
        )

    def _handle_initial_intersection_yield(self, status) -> ContactCheckpoint:
        required = int(qd_to_numpy(self.contact_data.n_et_pairs))
        self.contact_system.realloc_et_pairs(required)
        return ContactCheckpoint.INITIAL_INTERSECTION

    def _handle_init_query_yield(self, status) -> ContactCheckpoint:
        self._handle_pair_overflow()
        return ContactCheckpoint.QUERY

    def _initialize_contact(self) -> None:
        pair_overflow = self.contact_data.overflow_flag
        et_overflow = self.contact_data.et_overflow_flag
        self.init_contact_pipeline.run(pair_overflow, et_overflow, self.graph_fastcache_key)
        if int(qd_to_numpy(self.contact_data.n_et_pairs)) > 0:
            message = self._et_report_message("initial state")
            gs.logger.error(message)
            raise RuntimeError(message)

    def _et_report_message(self, stage: str) -> str:
        count = int(qd_to_numpy(self.contact_data.n_et_pairs))
        pairs = qd_to_numpy(self.contact_data.et_pairs)[:count]
        edges = qd_to_numpy(self.global_surface_data.surf_edges)
        faces = qd_to_numpy(self.global_surface_data.surf_triangles)
        positions = qd_to_numpy(self.global_vertex_data.positions)
        body_ids = qd_to_numpy(self.global_vertex_data.body_id)
        geometry_ids = qd_to_numpy(self.global_vertex_data.geometry_id)
        geometry_sources = qd_to_numpy(self.global_vertex_data.geometry_source)
        source_geometry_ids = qd_to_numpy(self.global_vertex_data.source_geometry_id)
        geometry_environments = qd_to_numpy(self.global_vertex_data.geometry_environment)
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
            f"pt_pairs={int(qd_to_numpy(self.contact_data.n_pairs_pt))}, "
            f"ee_pairs={int(qd_to_numpy(self.contact_data.n_pairs_ee))}, "
            f"active_pairs={int(qd_to_numpy(self.contact_data.n_active_pairs))}, "
            f"ccd_alpha={float(qd_to_numpy(self.contact_data.ccd_alpha)):.9g}, "
            "frame_ccd_alpha="
            f"{float(qd_to_numpy(self.contact_data.frame_ccd_alpha)):.9g}"
        )
        if self.rigid_contact_assemble is not None:
            contact_state += (
                f", proxy_doublets={int(qd_to_numpy(self.rigid_contact_assemble_data.rigid_doublet_total))}"
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

    def _handle_step_yield(self, status) -> ContactCheckpoint:
        checkpoint = status.checkpoint
        if self.contact_system is not None and checkpoint == ContactCheckpoint.FRICTION:
            required = {
                channel: int(qd_to_numpy(getattr(self.contact_data, f"n_friction_pairs_{channel}")))
                for channel in ("pt", "ee", "pe", "pp", "ph")
            }
            self.contact_system.realloc_friction_pair_buffers(required)
            self.contact_data.friction_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
            return ContactCheckpoint.FRICTION
        if self.contact_system is not None and checkpoint == ContactCheckpoint.COUNT:
            required_doublets = int(qd_to_numpy(self.contact_data.n_counted_doublets)) + int(
                qd_to_numpy(self.contact_data.n_friction_demand_doublets)
            )
            required_triplets = int(qd_to_numpy(self.contact_data.n_counted_triplets)) + int(
                qd_to_numpy(self.contact_data.n_friction_demand_triplets)
            )
            self.contact_system.realloc_assembly_buffers(required_doublets, required_triplets)
            if self.rigid_contact_assemble is not None:
                self.rigid_contact_assemble.realloc_assembly_buffers(self.contact_data)
            self.contact_data.count_overflow_flag.from_numpy(np.array(0, dtype=np.int32))
            return ContactCheckpoint.FILTER
        if self.contact_system is not None and checkpoint == ContactCheckpoint.FILTER:
            n_doublets = int(qd_to_numpy(self.contact_data.n_contact_doublets))
            n_triplets = int(qd_to_numpy(self.contact_data.n_contact_triplets))
            grow_factor = CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"]
            padded_doublets = max(
                int(qd_to_numpy(self.contact_data.padded_contact_doublets)),
                min(
                    int(np.ceil(n_doublets * grow_factor)),
                    self.contact_data.contact_doublet_vertices.shape[0],
                ),
            )
            padded_triplets = max(
                int(qd_to_numpy(self.contact_data.padded_contact_triplets)),
                min(
                    int(np.ceil(n_triplets * grow_factor)),
                    self.contact_data.contact_triplet_rows.shape[0],
                ),
            )
            self.contact_system.set_assembly_padding(
                padded_doublets,
                padded_triplets,
            )
            self.contact_data.contact_padding_overflow.from_numpy(np.array(0, dtype=np.int32))
            return ContactCheckpoint.SORT
        if checkpoint == ContactCheckpoint.SORT:
            required = int(qd_to_numpy(self.global_linear_system_data.matrix.n_triplets))
            grow_factor = CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"]
            grow_bcoo_matrix(
                self.global_linear_system_data.matrix,
                max(
                    int(np.ceil(required * grow_factor)),
                    self.global_linear_system_data.matrix.triplet_row.shape[0] + 1,
                ),
                live_size=required,
            )
            self.global_linear_system_data.matrix.triplet_overflow.from_numpy(np.array(0, dtype=np.int32))
            return ContactCheckpoint.SOLVE
        if self.contact_system is not None and checkpoint == ContactCheckpoint.QUERY:
            self._handle_pair_overflow()
            return ContactCheckpoint.QUERY
        if self.contact_system is not None and checkpoint == ContactCheckpoint.ET_OVERFLOW:
            required = int(qd_to_numpy(self.contact_data.n_et_pairs))
            self.contact_system.realloc_et_pairs(required)
            return ContactCheckpoint.ET_OVERFLOW
        if self.contact_system is not None and checkpoint == ContactCheckpoint.ET_FAILURE:
            message = self._et_report_message("step")
            gs.logger.error(message)
            raise RuntimeError(message)
        if checkpoint == ContactCheckpoint.KKT_FAILURE:
            proxy = self.rigid_contact_proxy_data
            rigid_dofs = self.rigid.dof_count_host
            rigid_rhs = qd_to_numpy(self.global_linear_system_data.b_rhs)[:rigid_dofs]
            rigid_solution = qd_to_numpy(self.global_linear_system_data.x_sol)[:rigid_dofs]
            rigid_preconditioned = qd_to_numpy(self.pcg_solver_data.preconditioned_residual)[:rigid_dofs]
            rigid_search = qd_to_numpy(self.rigid_data.constraint_state.search).reshape(-1)[:rigid_dofs]
            edge_dofs = qd_to_numpy(self.rigid_forest_data.edge_dof_index)[
                : int(qd_to_numpy(self.rigid_forest_data.n_edges))
            ]
            parent_edges = qd_to_numpy(self.rigid_forest_data.parent_edge)[
                : int(qd_to_numpy(self.rigid_forest_data.n_mechanism_bodies))
            ]
            edge_pivots = qd_to_numpy(self.rigid_forest_data.edge_d)[: len(edge_dofs)]
            edge_rhs = qd_to_numpy(self.rigid_forest_data.precond_a)[: len(edge_dofs)]
            edge_children = qd_to_numpy(self.rigid_forest_data.edge_child)[: len(edge_dofs)]
            forest_depth = qd_to_numpy(self.rigid_forest_data.depth)
            details = (
                f"newton={int(qd_to_numpy(self.newton_iter))}, "
                f"pcg={int(qd_to_numpy(self.pcg_solver_data.n_iterations))}, "
                f"line_search={int(qd_to_numpy(self.ls_iter))}, "
                f"alpha={float(qd_to_numpy(self.alpha)):.6g}, "
                f"rigid_gradient_squared={float(qd_to_numpy(self.rigid_data.gradient_squared)):.6g}, "
                f"rigid_rhs_norm={float(np.linalg.norm(rigid_rhs)):.6g}, "
                f"rigid_solution_norm={float(np.linalg.norm(rigid_solution)):.6g}, "
                f"rigid_preconditioned_norm={float(np.linalg.norm(rigid_preconditioned)):.6g}, "
                f"rigid_search_norm={float(np.linalg.norm(rigid_search)):.6g}, "
                f"edge_dofs={edge_dofs.tolist()}, "
                f"edge_pivots={edge_pivots.tolist()}, "
                f"edge_rhs={edge_rhs.tolist()}, "
                f"edge_depths={forest_depth[edge_children].tolist()}, "
                f"active_parent_edges={int(np.count_nonzero(parent_edges >= 0))}, "
                f"forest_levels={int(qd_to_numpy(self.rigid_forest_data.n_levels))}, "
                f"max_disp={float(qd_to_numpy(self.max_disp)):.6g}, "
                f"residual={float(qd_to_numpy(proxy.max_surface_residual)):.6g}, "
                f"tolerance={float(qd_to_numpy(proxy.solve_tolerance)):.6g}, "
                f"fk_alpha={float(qd_to_numpy(proxy.fk_alpha)):.6g}, "
                f"restoration_active={int(qd_to_numpy(proxy.restoration_active))}, "
                f"hard_probe={int(qd_to_numpy(proxy.restoration_hard_probe))}, "
                f"restoration_entries={int(qd_to_numpy(proxy.restoration_entries))}, "
                f"restoration_epochs={int(qd_to_numpy(proxy.restoration_newton_epochs))}, "
                f"hard_probes={int(qd_to_numpy(proxy.restoration_hard_probes))}, "
                f"pcg_failed={int(qd_to_numpy(self.pcg_solver_data.is_failed))}, "
                f"proxy_failed={int(qd_to_numpy(proxy.frame_failed))}"
            )
            raise RuntimeError(
                f"KKT rigid proxy solve exhausted the Newton budget before stationarity/feasibility ({details})"
            )
        raise RuntimeError(f"Unexpected timestep checkpoint {checkpoint}")

    def step(self) -> None:
        if not self.is_initialized_host:
            raise RuntimeError("SimEngine.init() must run before step()")
        if self.contact_system is None:
            pair_overflow = self.checkpoint_never_yield
            assembly_overflow = self.checkpoint_never_yield
            padding_overflow = self.checkpoint_never_yield
            friction_overflow = self.checkpoint_never_yield
        else:
            pair_overflow = self.contact_data.overflow_flag
            assembly_overflow = self.contact_data.count_overflow_flag
            padding_overflow = self.contact_data.contact_padding_overflow
            friction_overflow = self.contact_data.friction_overflow_flag
        et_overflow = self.checkpoint_never_yield if self.contact_system is None else self.contact_data.et_overflow_flag
        triplet_overflow = self.global_linear_system_data.matrix.triplet_overflow

        self.step_pipeline.run(
            pair_overflow,
            assembly_overflow,
            padding_overflow,
            triplet_overflow,
            friction_overflow,
            et_overflow,
            self.graph_fastcache_key,
        )
        if self.fem_system is not None:
            forward_fem_scene_vertices(self.fem_data)
        if bool(qd_to_numpy(self.frame_failed)):
            details = [
                f"newton={self.get_newton_iters()}",
                f"pcg={self.get_max_pcg_iters()}",
                f"line_search={self.get_max_ls_iters()}",
                f"triplet_overflow={int(qd_to_numpy(self.global_linear_system_data.matrix.triplet_overflow))}",
                f"bcoo_valid={int(qd_to_numpy(self.global_linear_system_data.matrix.bcoo_valid))}",
            ]
            if self.contact_system is not None:
                details.extend(
                    (
                        f"intersection={int(qd_to_numpy(self.contact_data.intersection_flag))}",
                        f"pairs_pt={int(qd_to_numpy(self.contact_data.n_pairs_pt))}",
                        f"pairs_ee={int(qd_to_numpy(self.contact_data.n_pairs_ee))}",
                        f"pairs_ph={int(qd_to_numpy(self.contact_data.n_pairs_ph))}",
                        f"active_pairs={int(qd_to_numpy(self.contact_data.n_active_pairs))}",
                        f"ccd_alpha={float(qd_to_numpy(self.contact_data.ccd_alpha)):.6g}",
                    )
                )
                for channel in ("pt", "ee", "ph"):
                    count = int(qd_to_numpy(getattr(self.contact_data, f"n_pairs_{channel}")))
                    if count:
                        alphas = qd_to_numpy(getattr(self.contact_data, f"ccd_alpha_{channel}"))[:count]
                        minimum_index = int(np.argmin(alphas))
                        pair = qd_to_numpy(getattr(self.contact_data, f"pairs_{channel}"))[minimum_index]
                        details.append(
                            f"ccd_{channel}=({float(alphas[minimum_index]):.6g},"
                            f" pair={tuple(int(value) for value in pair)})"
                        )
                        if channel == "ph":
                            surface_vertex = int(pair[0])
                            vertex_id = int(qd_to_numpy(self.global_surface_data.surf_verts)[surface_vertex])
                            current = qd_to_numpy(self.global_vertex_data.positions)[vertex_id]
                            endpoint = qd_to_numpy(self.global_vertex_data.trajectory_end_positions)[vertex_id]
                            thickness = float(qd_to_numpy(self.global_vertex_data.thicknesses)[vertex_id])
                            details.append(
                                f"ph_path=(current={tuple(float(value) for value in current)},"
                                f" endpoint={tuple(float(value) for value in endpoint)},"
                                f" thickness={thickness:.6g})"
                            )
            raise RuntimeError(f"SimEngine Newton solve failed ({', '.join(details)})")

    def get_newton_iters(self) -> int:
        return int(qd_to_numpy(self.newton_iter))

    def get_max_pcg_iters(self) -> int:
        return int(qd_to_numpy(self.max_pcg_iters))

    def get_total_pcg_iters(self) -> int:
        return int(qd_to_numpy(self.total_pcg_iters))

    def get_max_ls_iters(self) -> int:
        return int(qd_to_numpy(self.ls_iter))


def get_sim_engine_data(
    *,
    max_ls_iter: int,
    sim_config_data: SimConfig.Data,
    rigid_data: RigidSystem.Data | None,
    fem_data: FiniteElementMethod.Data | None,
    global_body_data: GlobalBodyManager.Data | None,
    global_vertex_data: GlobalVertexManager.Data | None,
    global_surface_data: GlobalSurfaceManager.Data | None,
    contact_data: ContactSystem.Data | None,
    rigid_contact_proxy_data: RigidContactProxySystem.Data | None,
    rigid_contact_assemble_data: RigidContactAssemble.Data | None,
    rigid_forest_data: RigidJointForestSystem.Data | None,
    global_linear_system_data: GlobalLinearSystem.Data,
    fem_preconditioner_data: FEMDiagPreconditioner.Data | None,
    pcg_solver_data: PCGSolver.Data,
    has_rigid: bool,
    has_fem: bool,
    has_contact: bool,
    has_rigid_contact_proxy: bool,
    has_rigid_contact_assemble: bool,
    has_rigid_forest: bool,
    has_et_check: bool,
    genesis_serial_pipeline: bool,
    fem_predict_actions: tuple[ActionInvocation, ...],
    fem_extent_actions: tuple[ActionInvocation, ...],
    fem_assemble_actions: tuple[ActionInvocation, ...],
    fem_energy_actions: tuple[ActionInvocation, ...],
    pcg_solver_invocations: tuple[ActionInvocation, ...],
    pcg_operator_actions: tuple[ActionInvocation, ...],
    pcg_preconditioner_actions: tuple[ActionInvocation, ...],
    contact_actions: dict[str, ActionInvocation],
    broad_phase_actions: dict[str, ActionInvocation],
    constitution_actions: dict[str, ActionInvocation],
) -> SimEngine.Data:
    """Construct the complete graph-visible engine root before pipeline creation."""
    data = SimEngine.Data()
    data.newton_cond = qd.ndarray(qd.i32, shape=())
    data.ls_cond = qd.ndarray(qd.i32, shape=())
    data.converged = qd.ndarray(qd.i32, shape=())
    data.frame_failed = qd.ndarray(qd.i32, shape=())
    data.newton_iter = qd.ndarray(qd.i32, shape=())
    data.max_pcg_iters = qd.ndarray(qd.i32, shape=())
    data.total_pcg_iters = qd.ndarray(qd.i32, shape=())
    data.ls_iter = qd.ndarray(qd.i32, shape=())
    data.alpha = qd.ndarray(qd.f64, shape=())
    data.max_disp = qd.ndarray(qd.f64, shape=())
    data.energy_delta = qd.ndarray(qd.f64, shape=())
    data.checkpoint_never_yield = qd.ndarray(qd.i32, shape=())
    data.energy_buf = qd.ndarray(qd.f64, shape=(max_ls_iter + 1,))
    data.max_pcg_iters.from_numpy(np.array(0, dtype=np.int32))
    data.total_pcg_iters.from_numpy(np.array(0, dtype=np.int32))
    data.checkpoint_never_yield.from_numpy(np.array(0, dtype=np.int32))
    data.sim_config_data = sim_config_data
    data.rigid_data = rigid_data
    data.fem_data = fem_data
    data.global_body_data = global_body_data
    data.global_vertex_data = global_vertex_data
    data.global_surface_data = global_surface_data
    data.contact_data = contact_data
    data.rigid_contact_proxy_data = rigid_contact_proxy_data
    data.rigid_contact_assemble_data = rigid_contact_assemble_data
    data.rigid_forest_data = rigid_forest_data
    data.global_linear_system_data = global_linear_system_data
    data.fem_preconditioner_data = fem_preconditioner_data
    data.pcg_solver_data = pcg_solver_data
    data.has_rigid = has_rigid
    data.has_fem = has_fem
    data.has_contact = has_contact
    data.has_rigid_contact_proxy = has_rigid_contact_proxy
    data.has_rigid_contact_assemble = has_rigid_contact_assemble
    data.has_rigid_forest = has_rigid_forest
    data.has_et_check = has_et_check
    data.genesis_serial_pipeline = genesis_serial_pipeline
    data.fem_predict_actions = fem_predict_actions
    data.fem_extent_actions = fem_extent_actions
    data.fem_assemble_actions = fem_assemble_actions
    data.fem_energy_actions = fem_energy_actions
    data.pcg_solver_invocations = pcg_solver_invocations
    data.pcg_operator_actions = pcg_operator_actions
    data.pcg_preconditioner_actions = pcg_preconditioner_actions
    data.contact_reset_initial_intersections_action = contact_actions.get("reset_initial_intersections")
    data.contact_flag_et_intersections_action = contact_actions.get("flag_et_intersections")
    data.contact_reset_counted_demand_action = contact_actions.get("reset_counted_demand")
    data.contact_adaptive_kappa_update_action = contact_actions.get("adaptive_kappa_update")
    data.contact_adaptive_kappa_newton_tick_action = contact_actions.get("adaptive_kappa_newton_tick")
    data.contact_reset_collision_counts_action = contact_actions.get("reset_collision_counts")
    data.contact_halfplane_query_action = contact_actions.get("halfplane_query")
    data.contact_init_ccd_action = contact_actions.get("init_ccd")
    data.contact_reset_frame_ccd_action = contact_actions.get("reset_frame_ccd")
    data.contact_ccd_alpha_pt_action = contact_actions.get("ccd_alpha_pt")
    data.contact_ccd_alpha_ee_action = contact_actions.get("ccd_alpha_ee")
    data.contact_ccd_alpha_ph_action = contact_actions.get("ccd_alpha_ph")
    data.contact_reduce_ccd_alpha_action = contact_actions.get("reduce_ccd_alpha")
    data.contact_ccd_action = contact_actions.get("ccd")
    data.contact_reset_contact_energy_action = contact_actions.get("reset_contact_energy")
    data.contact_sum_contact_energy_action = contact_actions.get("sum_contact_energy")
    data.contact_check_assembly_capacity_action = contact_actions.get("check_assembly_capacity")
    data.contact_check_assembly_padding_action = contact_actions.get("check_assembly_padding")
    data.contact_shrink_assembly_padding_action = contact_actions.get("shrink_assembly_padding")
    data.contact_reset_assembly_counts_action = contact_actions.get("reset_assembly_counts")
    data.contact_sort_reduce_action = contact_actions.get("sort_reduce")
    data.broad_phase_triangle_build_action = broad_phase_actions.get("triangle_build")
    data.broad_phase_edge_build_action = broad_phase_actions.get("edge_build")
    data.broad_phase_pt_query_action = broad_phase_actions.get("pt_query")
    data.broad_phase_ee_query_action = broad_phase_actions.get("ee_query")
    data.broad_phase_trajectory_query_action = broad_phase_actions.get("trajectory_query")
    data.broad_phase_detect_initial_intersections_action = broad_phase_actions.get("detect_initial_intersections")
    data.contact_constitution_snapshot_lagged_positions_action = constitution_actions.get("snapshot_lagged_positions")
    data.contact_constitution_friction_pair_filter_pt_action = constitution_actions.get("friction_pair_filter_pt")
    data.contact_constitution_friction_pair_filter_ee_action = constitution_actions.get("friction_pair_filter_ee")
    data.contact_constitution_friction_pair_filter_ph_action = constitution_actions.get("friction_pair_filter_ph")
    data.contact_constitution_friction_snapshot_action = constitution_actions.get("friction_snapshot")
    data.contact_constitution_count_active_pt_action = constitution_actions.get("count_active_pt")
    data.contact_constitution_count_active_ee_action = constitution_actions.get("count_active_ee")
    data.contact_constitution_count_active_ph_action = constitution_actions.get("count_active_ph")
    data.contact_constitution_count_active_action = constitution_actions.get("count_active")
    data.contact_constitution_filter_assemble_pt_action = constitution_actions.get("filter_assemble_pt")
    data.contact_constitution_filter_assemble_ee_action = constitution_actions.get("filter_assemble_ee")
    data.contact_constitution_filter_assemble_ph_action = constitution_actions.get("filter_assemble_ph")
    data.contact_constitution_friction_assemble_pt_action = constitution_actions.get("friction_assemble_pt")
    data.contact_constitution_friction_assemble_ee_action = constitution_actions.get("friction_assemble_ee")
    data.contact_constitution_friction_assemble_ph_action = constitution_actions.get("friction_assemble_ph")
    data.contact_constitution_filter_assemble_action = constitution_actions.get("filter_assemble")
    data.contact_constitution_filter_energy_pt_action = constitution_actions.get("filter_energy_pt")
    data.contact_constitution_filter_energy_ee_action = constitution_actions.get("filter_energy_ee")
    data.contact_constitution_filter_energy_ph_action = constitution_actions.get("filter_energy_ph")
    data.contact_constitution_friction_energy_pt_action = constitution_actions.get("friction_energy_pt")
    data.contact_constitution_friction_energy_ee_action = constitution_actions.get("friction_energy_ee")
    data.contact_constitution_friction_energy_ph_action = constitution_actions.get("friction_energy_ph")
    data.contact_constitution_contact_energy_action = constitution_actions.get("contact_energy")
    return data


@qd.kernel(fastcache=True)
def _initialize_global_resources(data: qd.template(), graph_fastcache_key: qd.template()):
    if qd.static(graph_fastcache_key < 0):
        data.frame_failed[()] = 0
    if qd.static(data.has_fem):
        initialize_fem_global_vertices(data.fem_data, data.global_vertex_data)
        compute_vertex_offsets(data.global_body_data, data.fem_data)
    if qd.static(data.has_rigid_contact_proxy):
        initialize_proxy_state(
            data.rigid_contact_proxy_data,
            data.rigid_data,
            data.rigid_forest_data,
        )
        prepare_metric(
            data.rigid_contact_proxy_data,
            data.rigid_data,
            data.rigid_forest_data,
        )
        initialize_proxy_global_vertices(
            data.rigid_contact_proxy_data,
            data.rigid_data,
            data.rigid_forest_data,
            data.global_vertex_data,
        )


@qd.kernel(fastcache=True)
def _sync_from_solvers_kernel(data: qd.template(), graph_fastcache_key: qd.template()):
    if qd.static(graph_fastcache_key < 0):
        data.frame_failed[()] = 0
    if qd.static(data.has_fem):
        sync_fem_from_scene(data.fem_data, data.global_vertex_data)
    if qd.static(data.has_rigid_contact_proxy):
        initialize_proxy_state(
            data.rigid_contact_proxy_data,
            data.rigid_data,
            data.rigid_forest_data,
        )
        reset_proxy_frame(
            data.rigid_contact_proxy_data,
            data.rigid_data,
            data.rigid_forest_data,
        )
        prepare_metric(
            data.rigid_contact_proxy_data,
            data.rigid_data,
            data.rigid_forest_data,
        )
        initialize_proxy_global_vertices(
            data.rigid_contact_proxy_data,
            data.rigid_data,
            data.rigid_forest_data,
            data.global_vertex_data,
        )


@qd.kernel(graph=True, checkpoints=True, fastcache=True)
def _init_contact_kernel(
    data: qd.template(),
    pair_overflow: qd.types.ndarray(qd.i32, ndim=0),
    et_overflow: qd.types.ndarray(qd.i32, ndim=0),
    graph_fastcache_key: qd.template(),
):
    if qd.static(graph_fastcache_key < 0):
        data.frame_failed[()] = 0
    with qd.checkpoint(ContactCheckpoint.FRAME, yield_on=data.checkpoint_never_yield):
        if qd.static(True):
            forward_fem_global_vertices(data.fem_data, data.global_vertex_data)
            reset_trajectory(data.global_vertex_data)
            if qd.static(data.genesis_serial_pipeline):
                data.broad_phase_triangle_build_action.kernel(*data.broad_phase_triangle_build_action.data)
                data.broad_phase_edge_build_action.kernel(*data.broad_phase_edge_build_action.data)
            else:
                with qd.graph.parallel_context():
                    with qd.graph.parallel():
                        data.broad_phase_triangle_build_action.kernel(*data.broad_phase_triangle_build_action.data)
                    with qd.graph.parallel():
                        data.broad_phase_edge_build_action.kernel(*data.broad_phase_edge_build_action.data)
    if qd.static(data.has_et_check):
        with qd.checkpoint(
            ContactCheckpoint.INITIAL_INTERSECTION,
            yield_on=et_overflow,
        ):
            if qd.static(True):
                data.contact_reset_initial_intersections_action.kernel(
                    *data.contact_reset_initial_intersections_action.data
                )
                data.broad_phase_detect_initial_intersections_action.kernel(
                    *data.broad_phase_detect_initial_intersections_action.data
                )
    with qd.checkpoint(ContactCheckpoint.QUERY, yield_on=pair_overflow):
        if qd.static(True):
            data.contact_reset_collision_counts_action.kernel(*data.contact_reset_collision_counts_action.data)
            if qd.static(data.genesis_serial_pipeline):
                data.broad_phase_trajectory_query_action.kernel(*data.broad_phase_trajectory_query_action.data)
                if qd.static(data.contact_data.has_halfplanes):
                    data.contact_halfplane_query_action.kernel(*data.contact_halfplane_query_action.data)
            else:
                with qd.graph.parallel_context():
                    with qd.graph.parallel():
                        data.broad_phase_pt_query_action.kernel(*data.broad_phase_pt_query_action.data)
                    with qd.graph.parallel():
                        data.broad_phase_ee_query_action.kernel(*data.broad_phase_ee_query_action.data)
                    with qd.graph.parallel():
                        if qd.static(data.contact_data.has_halfplanes):
                            data.contact_halfplane_query_action.kernel(*data.contact_halfplane_query_action.data)
    with qd.checkpoint(ContactCheckpoint.COUNT, yield_on=data.checkpoint_never_yield):
        if qd.static(True):
            data.contact_reset_counted_demand_action.kernel(*data.contact_reset_counted_demand_action.data)
            if qd.static(data.genesis_serial_pipeline):
                data.contact_constitution_count_active_action.kernel(
                    *data.contact_constitution_count_active_action.data
                )
            else:
                with qd.graph.parallel_context():
                    with qd.graph.parallel():
                        data.contact_constitution_count_active_pt_action.kernel(
                            *data.contact_constitution_count_active_pt_action.data
                        )
                    with qd.graph.parallel():
                        data.contact_constitution_count_active_ee_action.kernel(
                            *data.contact_constitution_count_active_ee_action.data
                        )
                    with qd.graph.parallel():
                        if qd.static(data.contact_data.has_halfplanes):
                            data.contact_constitution_count_active_ph_action.kernel(
                                *data.contact_constitution_count_active_ph_action.data
                            )
            data.contact_check_assembly_capacity_action.kernel(*data.contact_check_assembly_capacity_action.data)


@qd.kernel(graph=True, checkpoints=True, fastcache=True)
def _step_kernel(
    data: qd.template(),
    pair_overflow: qd.types.ndarray(qd.i32, ndim=0),
    assembly_overflow: qd.types.ndarray(qd.i32, ndim=0),
    padding_overflow: qd.types.ndarray(qd.i32, ndim=0),
    triplet_overflow: qd.types.ndarray(qd.i32, ndim=0),
    friction_overflow: qd.types.ndarray(qd.i32, ndim=0),
    et_overflow: qd.types.ndarray(qd.i32, ndim=0),
    graph_fastcache_key: qd.template(),
):
    if qd.static(graph_fastcache_key < 0):
        data.frame_failed[()] = 0
    with qd.checkpoint(ContactCheckpoint.FRAME, yield_on=data.checkpoint_never_yield):
        if qd.static(data.has_contact):
            data.contact_adaptive_kappa_update_action.kernel(*data.contact_adaptive_kappa_update_action.data)
            data.contact_reset_frame_ccd_action.kernel(*data.contact_reset_frame_ccd_action.data)
        if qd.static(data.has_rigid):
            predict_rigid(data.rigid_data)
            assemble_candidate_rows(data.rigid_data)
        if qd.static(data.has_rigid_contact_proxy):
            mark_mechanism_constrained(
                data.rigid_contact_proxy_data,
                data.rigid_data,
                data.rigid_forest_data,
            )
        if qd.static(data.has_rigid):
            initialize_rigid_newton(data.rigid_data)
        if qd.static(data.has_rigid_contact_proxy):
            reset_proxy_frame(
                data.rigid_contact_proxy_data,
                data.rigid_data,
                data.rigid_forest_data,
            )
            prepare_metric(
                data.rigid_contact_proxy_data,
                data.rigid_data,
                data.rigid_forest_data,
            )
            prepare_tolerance(
                data.rigid_contact_proxy_data,
                data.rigid_data,
                data.rigid_forest_data,
                data.sim_config_data,
                data.contact_data,
            )
        if qd.static(data.has_fem):
            for action in qd.static(data.fem_predict_actions):
                action.kernel(*(action.data + (data.sim_config_data,)))
            forward_fem_global_vertices(data.fem_data, data.global_vertex_data)
        for _ in range(1):
            data.newton_iter[()] = 0
            data.max_pcg_iters[()] = 0
            data.total_pcg_iters[()] = 0
            data.ls_iter[()] = 0
            data.frame_failed[()] = 0
            data.newton_cond[()] = 1

    with qd.checkpoint(ContactCheckpoint.FRICTION, yield_on=friction_overflow):
        if qd.static(data.has_contact):  # noqa: SIM102
            if qd.static(data.contact_data.has_friction):
                if qd.static(data.genesis_serial_pipeline):
                    data.contact_constitution_friction_snapshot_action.kernel(
                        *data.contact_constitution_friction_snapshot_action.data
                    )
                else:
                    data.contact_constitution_snapshot_lagged_positions_action.kernel(
                        *data.contact_constitution_snapshot_lagged_positions_action.data
                    )
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            data.contact_constitution_friction_pair_filter_pt_action.kernel(
                                *data.contact_constitution_friction_pair_filter_pt_action.data
                            )
                        with qd.graph.parallel():
                            data.contact_constitution_friction_pair_filter_ee_action.kernel(
                                *data.contact_constitution_friction_pair_filter_ee_action.data
                            )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_halfplanes):
                                data.contact_constitution_friction_pair_filter_ph_action.kernel(
                                    *data.contact_constitution_friction_pair_filter_ph_action.data
                                )

    while qd.graph.do_while(data.newton_cond):
        with qd.checkpoint(ContactCheckpoint.COUNT, yield_on=assembly_overflow):
            if qd.static(data.has_rigid_contact_proxy):
                initialize_proxy_newton(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                )
            if qd.static(data.has_rigid_forest):
                compute_endpoint_fk(
                    data.rigid_forest_data,
                    data.rigid_data,
                    data.rigid_contact_proxy_data,
                )
            if qd.static(data.has_rigid_contact_proxy):
                prepare_constraint(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                )
            if qd.static(data.has_contact):
                data.contact_reset_counted_demand_action.kernel(*data.contact_reset_counted_demand_action.data)
                if qd.static(data.genesis_serial_pipeline):
                    data.contact_constitution_count_active_action.kernel(
                        *data.contact_constitution_count_active_action.data
                    )
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            data.contact_constitution_count_active_pt_action.kernel(
                                *data.contact_constitution_count_active_pt_action.data
                            )
                        with qd.graph.parallel():
                            data.contact_constitution_count_active_ee_action.kernel(
                                *data.contact_constitution_count_active_ee_action.data
                            )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_halfplanes):
                                data.contact_constitution_count_active_ph_action.kernel(
                                    *data.contact_constitution_count_active_ph_action.data
                                )
                data.contact_check_assembly_capacity_action.kernel(*data.contact_check_assembly_capacity_action.data)

        with qd.checkpoint(ContactCheckpoint.FILTER, yield_on=padding_overflow):
            for _ in range(1):
                data.newton_iter[()] = data.newton_iter[()] + 1
            if qd.static(data.has_contact):
                data.contact_adaptive_kappa_newton_tick_action.kernel(
                    *data.contact_adaptive_kappa_newton_tick_action.data
                )
                zero_in_contact(data.global_vertex_data)
                data.contact_reset_assembly_counts_action.kernel(*data.contact_reset_assembly_counts_action.data)
                if qd.static(data.genesis_serial_pipeline):
                    data.contact_constitution_filter_assemble_action.kernel(
                        *data.contact_constitution_filter_assemble_action.data
                    )
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            data.contact_constitution_filter_assemble_pt_action.kernel(
                                *data.contact_constitution_filter_assemble_pt_action.data
                            )
                        with qd.graph.parallel():
                            data.contact_constitution_filter_assemble_ee_action.kernel(
                                *data.contact_constitution_filter_assemble_ee_action.data
                            )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_halfplanes):
                                data.contact_constitution_filter_assemble_ph_action.kernel(
                                    *data.contact_constitution_filter_assemble_ph_action.data
                                )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_friction):
                                data.contact_constitution_friction_assemble_pt_action.kernel(
                                    *data.contact_constitution_friction_assemble_pt_action.data
                                )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_friction):
                                data.contact_constitution_friction_assemble_ee_action.kernel(
                                    *data.contact_constitution_friction_assemble_ee_action.data
                                )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_friction and data.contact_data.has_halfplanes):
                                data.contact_constitution_friction_assemble_ph_action.kernel(
                                    *data.contact_constitution_friction_assemble_ph_action.data
                                )
                data.contact_check_assembly_padding_action.kernel(*data.contact_check_assembly_padding_action.data)

        with qd.checkpoint(ContactCheckpoint.SORT, yield_on=triplet_overflow):
            if qd.static(data.has_rigid_contact_assemble):
                data.contact_sort_reduce_action.kernel(*data.contact_sort_reduce_action.data)
                for action in qd.static(data.fem_extent_actions):
                    action.kernel(*(action.data + (data.global_linear_system_data,)))
                classify_rigid_contact(
                    data.rigid_contact_assemble_data,
                    data.rigid_contact_proxy_data,
                    data.rigid_forest_data,
                    data.global_vertex_data,
                    data.contact_data,
                    data.global_linear_system_data,
                )
                derive_extents(data.global_linear_system_data)
            else:
                if qd.static(data.has_fem):
                    for action in qd.static(data.fem_extent_actions):
                        action.kernel(*(action.data + (data.global_linear_system_data,)))
                if qd.static(True):
                    derive_extents(data.global_linear_system_data)
                if qd.static(data.has_contact):
                    data.contact_sort_reduce_action.kernel(*data.contact_sort_reduce_action.data)
                    compute_n_triplets(data.global_linear_system_data, data.contact_data)

        with qd.checkpoint(ContactCheckpoint.SOLVE, yield_on=data.checkpoint_never_yield):
            if qd.static(True):
                zero_rhs(data.global_linear_system_data)
                zero_bcoo_triplets(data.global_linear_system_data.matrix)

            if qd.static(data.has_rigid):
                assemble_rigid(
                    data.rigid_data,
                    data.sim_config_data,
                    data.global_linear_system_data,
                    data.has_rigid_forest,
                )
            if qd.static(data.has_fem):
                for action in qd.static(data.fem_assemble_actions):
                    action.kernel(*(action.data + (data.sim_config_data, data.global_linear_system_data)))
            if qd.static(data.has_rigid_contact_assemble):
                distribute_rigid_contact(
                    data.rigid_contact_assemble_data,
                    data.rigid_contact_proxy_data,
                    data.rigid_forest_data,
                    data.global_vertex_data,
                    data.contact_data,
                    data.fem_data,
                    data.global_linear_system_data,
                )
            elif qd.static(data.has_contact):
                distribute_fem_gradient_kernel(data.contact_data, data.fem_data, data.global_linear_system_data)
                distribute_fem_fem_kernel(data.contact_data, data.fem_data, data.global_linear_system_data)

            if qd.static(data.has_fem):
                sort_reduce_bcoo(data.global_linear_system_data.matrix)
                initialize_fem_diag_preconditioner(data.fem_preconditioner_data)
                gather_fem_diag_preconditioner(
                    data.fem_preconditioner_data,
                    data.global_linear_system_data,
                )
                invert_fem_diag_preconditioner(data.fem_preconditioner_data)
            if qd.static(data.has_rigid_contact_proxy):
                capture_physical_gradient(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.global_linear_system_data,
                )
            if qd.static(data.has_rigid_forest):
                prepare_particular(
                    data.rigid_forest_data,
                    data.rigid_data,
                    data.rigid_contact_proxy_data,
                )
                particular_spmv(
                    data.rigid_forest_data,
                    data.rigid_data,
                    data.rigid_contact_proxy_data,
                    data.global_linear_system_data,
                )
                project_physical_rhs(
                    data.rigid_forest_data,
                    data.rigid_data,
                    data.rigid_contact_proxy_data,
                    data.global_linear_system_data,
                )
                build_forest_preconditioner(
                    data.rigid_forest_data,
                    data.rigid_data,
                    data.rigid_contact_proxy_data,
                    data.global_linear_system_data,
                )

        # Work around Quadrants #956: a top-level qd.func after a yielding
        # checkpoint may execute once on the yielding launch. The explicit
        # no-yield gate keeps the complete PCG child loop behind the resume
        # boundary. Remove it after the compiler issue is fixed.
        # https://github.com/Genesis-Embodied-AI/quadrants/issues/956
        with qd.checkpoint(ContactCheckpoint.PCG_SOLVE, yield_on=data.checkpoint_never_yield):
            if qd.static(True):
                solve_pcg(
                    data.pcg_solver_data,
                    data.global_linear_system_data,
                    data,
                    data.sim_config_data.max_pcg_iter[()],
                    data.max_pcg_iters,
                    data.total_pcg_iters,
                )

        # Keep all post-PCG qd.func calls under a flat gate as well. Otherwise an earlier capacity yield can skip
        # direct kernel tasks but still enter one of these inlined functions.
        with qd.checkpoint(ContactCheckpoint.SOLVE_POST, yield_on=data.checkpoint_never_yield):
            if qd.static(data.has_rigid):
                negate_dq(
                    data.rigid_data,
                    data.global_linear_system_data,
                    data.has_rigid_forest,
                )
            if qd.static(data.has_fem):
                negate_fem_dx(data.fem_data, data.global_linear_system_data)
            if qd.static(data.has_rigid_forest):
                expand_solution(
                    data.rigid_forest_data,
                    data.rigid_data,
                    data.rigid_contact_proxy_data,
                    data.global_linear_system_data.x_sol,
                )
            if qd.static(data.has_rigid_contact_proxy):
                prepare_path_limit(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.contact_data,
                )

            for _ in range(1):
                data.max_disp[()] = qd.f64(0.0)
            if qd.static(data.has_fem):
                contribute_fem_newton_max_disp(data.fem_data, data.max_disp)
            if qd.static(data.has_rigid_contact_proxy):
                contribute_proxy_newton_max_disp(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.global_vertex_data,
                    data.max_disp,
                )

            for _ in range(1):
                rigid_converged = True
                fem_converged = True
                if qd.static(data.has_rigid_contact_proxy):
                    rigid_converged = data.max_disp[()] <= data.sim_config_data.tol[()] * data.sim_config_data.dt[()]
                elif qd.static(data.has_rigid):
                    rigid_converged = data.rigid_data.gradient_squared[()] <= data.sim_config_data.tol[()] ** 2
                if qd.static(data.has_fem and not data.has_rigid_contact_proxy):
                    fem_converged = data.max_disp[()] <= data.sim_config_data.tol[()] * data.sim_config_data.dt[()]
                data.converged[()] = qd.i32(rigid_converged and fem_converged)
            if qd.static(data.has_rigid_contact_proxy):
                apply_proxy_convergence(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.converged,
                )

            if qd.static(data.has_rigid):
                record_rigid_start_point(data.rigid_data)
                compute_rigid_energy(data.rigid_data, data.sim_config_data)
            if qd.static(data.has_fem):
                record_fem_start_point(data.fem_data)
                reset_fem_energy(data.fem_data)
                for action in qd.static(data.fem_energy_actions):
                    action.kernel(*(action.data + (data.sim_config_data,)))
                record_safe_positions(data.global_vertex_data)
                publish_fem_trajectory_end_positions(data.fem_data, data.global_vertex_data)
            if qd.static(data.has_rigid_contact_proxy):
                record_proxy_start_point(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                )
                forward_proxy_global_vertices(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.global_vertex_data,
                )
                publish_proxy_trajectory_end_positions(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.global_vertex_data,
                )
            if qd.static(data.has_contact):
                if qd.static(data.genesis_serial_pipeline):
                    data.contact_reset_contact_energy_action.kernel(*data.contact_reset_contact_energy_action.data)
                    data.contact_constitution_contact_energy_action.kernel(
                        *data.contact_constitution_contact_energy_action.data
                    )
                    data.contact_sum_contact_energy_action.kernel(*data.contact_sum_contact_energy_action.data)
                else:
                    data.contact_reset_contact_energy_action.kernel(*data.contact_reset_contact_energy_action.data)
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            data.contact_constitution_filter_energy_pt_action.kernel(
                                *data.contact_constitution_filter_energy_pt_action.data
                            )
                        with qd.graph.parallel():
                            data.contact_constitution_filter_energy_ee_action.kernel(
                                *data.contact_constitution_filter_energy_ee_action.data
                            )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_halfplanes):
                                data.contact_constitution_filter_energy_ph_action.kernel(
                                    *data.contact_constitution_filter_energy_ph_action.data
                                )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_friction):
                                data.contact_constitution_friction_energy_pt_action.kernel(
                                    *data.contact_constitution_friction_energy_pt_action.data
                                )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_friction):
                                data.contact_constitution_friction_energy_ee_action.kernel(
                                    *data.contact_constitution_friction_energy_ee_action.data
                                )
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_friction and data.contact_data.has_halfplanes):
                                data.contact_constitution_friction_energy_ph_action.kernel(
                                    *data.contact_constitution_friction_energy_ph_action.data
                                )
                    data.contact_sum_contact_energy_action.kernel(*data.contact_sum_contact_energy_action.data)
            if qd.static(data.has_rigid_contact_proxy):
                compute_restoration_energy(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    False,
                )

            for _ in range(1):
                baseline = qd.f64(0.0)
                if qd.static(data.has_rigid):
                    baseline = baseline + data.rigid_data.rigid_energy[()]
                if qd.static(data.has_fem):
                    baseline = baseline + data.fem_data.fem_energy[()]
                if qd.static(data.has_contact):
                    baseline = baseline + data.contact_data.contact_energy_value[()]
                if qd.static(data.has_rigid_contact_proxy):
                    baseline = baseline + data.rigid_contact_proxy_data.restoration_energy[()]
                data.energy_buf[0] = baseline
            if qd.static(data.has_rigid_forest):
                compute_merit_directional_derivative(
                    data.rigid_forest_data,
                    data.rigid_data,
                    data.rigid_contact_proxy_data,
                    data.global_linear_system_data,
                )
                initialize_merit(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                )

            if qd.static(data.has_contact):
                if qd.static(data.genesis_serial_pipeline):
                    data.broad_phase_triangle_build_action.kernel(*data.broad_phase_triangle_build_action.data)
                    data.broad_phase_edge_build_action.kernel(*data.broad_phase_edge_build_action.data)
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            data.broad_phase_triangle_build_action.kernel(*data.broad_phase_triangle_build_action.data)
                        with qd.graph.parallel():
                            data.broad_phase_edge_build_action.kernel(*data.broad_phase_edge_build_action.data)

        with qd.checkpoint(ContactCheckpoint.QUERY, yield_on=pair_overflow):
            if qd.static(data.has_contact):
                data.contact_reset_collision_counts_action.kernel(*data.contact_reset_collision_counts_action.data)
                if qd.static(data.genesis_serial_pipeline):
                    data.broad_phase_trajectory_query_action.kernel(*data.broad_phase_trajectory_query_action.data)
                    if qd.static(data.contact_data.has_halfplanes):
                        data.contact_halfplane_query_action.kernel(*data.contact_halfplane_query_action.data)
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            data.broad_phase_pt_query_action.kernel(*data.broad_phase_pt_query_action.data)
                        with qd.graph.parallel():
                            data.broad_phase_ee_query_action.kernel(*data.broad_phase_ee_query_action.data)
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_halfplanes):
                                data.contact_halfplane_query_action.kernel(*data.contact_halfplane_query_action.data)

        with qd.checkpoint(ContactCheckpoint.CCD, yield_on=data.checkpoint_never_yield):
            if qd.static(data.has_contact):
                data.contact_init_ccd_action.kernel(*data.contact_init_ccd_action.data)
                if qd.static(data.genesis_serial_pipeline):
                    data.contact_ccd_action.kernel(*data.contact_ccd_action.data)
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            data.contact_ccd_alpha_pt_action.kernel(*data.contact_ccd_alpha_pt_action.data)
                        with qd.graph.parallel():
                            data.contact_ccd_alpha_ee_action.kernel(*data.contact_ccd_alpha_ee_action.data)
                        with qd.graph.parallel():
                            if qd.static(data.contact_data.has_halfplanes):
                                data.contact_ccd_alpha_ph_action.kernel(*data.contact_ccd_alpha_ph_action.data)
                    data.contact_reduce_ccd_alpha_action.kernel(*data.contact_reduce_ccd_alpha_action.data)

        with qd.checkpoint(ContactCheckpoint.LINE_SEARCH, yield_on=data.checkpoint_never_yield):
            for _ in range(1):
                initial_alpha = qd.f64(1.0)
                if qd.static(data.has_contact):
                    initial_alpha = qd.min(initial_alpha, data.contact_data.ccd_alpha[()])
                if qd.static(data.has_rigid_contact_proxy):
                    initial_alpha = qd.min(
                        initial_alpha,
                        data.rigid_contact_proxy_data.fk_alpha[()],
                    )
                data.alpha[()] = initial_alpha
                data.ls_cond[()] = 1
                data.ls_iter[()] = 0

        # No line-search operation can yield to the host. Its child WHILE stays outside the explicit phase marker,
        # while each inlined trial stage gets a flat gate for the same qd.func suffix workaround as PCG.
        while qd.graph.do_while(data.ls_cond):
            with qd.checkpoint(ContactCheckpoint.LINE_SEARCH_TRIAL, yield_on=data.checkpoint_never_yield):
                for _ in range(1):
                    if data.pcg_solver_data.is_failed[()] != 0:
                        data.alpha[()] = qd.f64(0.0)

                if qd.static(data.has_rigid):
                    step_rigid_forward(data.rigid_data, data.alpha[()])
                    compute_rigid_energy(data.rigid_data, data.sim_config_data)
                if qd.static(data.has_rigid_forest):
                    compute_endpoint_fk(
                        data.rigid_forest_data,
                        data.rigid_data,
                        data.rigid_contact_proxy_data,
                    )
                if qd.static(data.has_fem):
                    step_fem_forward(data.fem_data, data.alpha[()])
                    forward_fem_global_vertices(data.fem_data, data.global_vertex_data)
                    reset_fem_energy(data.fem_data)
                    for action in qd.static(data.fem_energy_actions):
                        action.kernel(*(action.data + (data.sim_config_data,)))
                if qd.static(data.has_rigid_contact_proxy):
                    step_proxy_forward(
                        data.rigid_contact_proxy_data,
                        data.rigid_data,
                        data.rigid_forest_data,
                        data.alpha[()],
                    )
                    forward_proxy_global_vertices(
                        data.rigid_contact_proxy_data,
                        data.rigid_data,
                        data.rigid_forest_data,
                        data.global_vertex_data,
                    )
                    evaluate_trial_guard(
                        data.rigid_contact_proxy_data,
                        data.rigid_data,
                        data.rigid_forest_data,
                    )
                if qd.static(data.has_contact):
                    if qd.static(data.genesis_serial_pipeline):
                        data.contact_reset_contact_energy_action.kernel(*data.contact_reset_contact_energy_action.data)
                        data.contact_constitution_contact_energy_action.kernel(
                            *data.contact_constitution_contact_energy_action.data
                        )
                        data.contact_sum_contact_energy_action.kernel(*data.contact_sum_contact_energy_action.data)
                    else:
                        data.contact_reset_contact_energy_action.kernel(*data.contact_reset_contact_energy_action.data)
                        with qd.graph.parallel_context():
                            with qd.graph.parallel():
                                data.contact_constitution_filter_energy_pt_action.kernel(
                                    *data.contact_constitution_filter_energy_pt_action.data
                                )
                            with qd.graph.parallel():
                                data.contact_constitution_filter_energy_ee_action.kernel(
                                    *data.contact_constitution_filter_energy_ee_action.data
                                )
                            with qd.graph.parallel():
                                if qd.static(data.contact_data.has_halfplanes):
                                    data.contact_constitution_filter_energy_ph_action.kernel(
                                        *data.contact_constitution_filter_energy_ph_action.data
                                    )
                            with qd.graph.parallel():
                                if qd.static(data.contact_data.has_friction):
                                    data.contact_constitution_friction_energy_pt_action.kernel(
                                        *data.contact_constitution_friction_energy_pt_action.data
                                    )
                            with qd.graph.parallel():
                                if qd.static(data.contact_data.has_friction):
                                    data.contact_constitution_friction_energy_ee_action.kernel(
                                        *data.contact_constitution_friction_energy_ee_action.data
                                    )
                            with qd.graph.parallel():
                                if qd.static(data.contact_data.has_friction and data.contact_data.has_halfplanes):
                                    data.contact_constitution_friction_energy_ph_action.kernel(
                                        *data.contact_constitution_friction_energy_ph_action.data
                                    )
                        data.contact_sum_contact_energy_action.kernel(*data.contact_sum_contact_energy_action.data)
                if qd.static(data.has_rigid_contact_proxy):
                    compute_restoration_energy(
                        data.rigid_contact_proxy_data,
                        data.rigid_data,
                        data.rigid_forest_data,
                        True,
                    )

                for _ in range(1):
                    trial_energy = qd.f64(0.0)
                    if qd.static(data.has_rigid):
                        trial_energy = trial_energy + data.rigid_data.rigid_energy[()]
                    if qd.static(data.has_fem):
                        trial_energy = trial_energy + data.fem_data.fem_energy[()]
                    if qd.static(data.has_contact):
                        trial_energy = trial_energy + data.contact_data.contact_energy_value[()]
                    if qd.static(data.has_rigid_contact_proxy):
                        trial_energy = trial_energy + data.rigid_contact_proxy_data.restoration_energy[()]
                        if (
                            data.rigid_contact_proxy_data.restoration_active[()] == 0
                            and data.rigid_contact_proxy_data.merit_evaluated[()] == 0
                        ):
                            trial_energy = trial_energy + data.rigid_contact_proxy_data.test_merit_energy_bias[()]
                        if data.rigid_contact_proxy_data.restoration_triggered[()] == 0:
                            trial_energy = (
                                trial_energy + data.rigid_contact_proxy_data.ls_forensics_test_energy_bias[()]
                            )
                    data.energy_delta[()] = trial_energy - data.energy_buf[0]

                for _ in range(1):
                    guard_failed = False
                    if qd.static(data.has_rigid_contact_proxy):
                        guard_failed = data.rigid_contact_proxy_data.frame_failed[()] != 0
                    accepted = data.converged[()] != 0 or data.energy_delta[()] <= 0.0
                    if qd.static(data.has_rigid_contact_proxy):
                        if data.converged[()] == 0:
                            exhausted = data.ls_iter[()] + 1 >= data.sim_config_data.max_ls_iter[()]
                            accepted = check_proxy_line_search(
                                data.rigid_contact_proxy_data,
                                data.rigid_data,
                                data.rigid_forest_data,
                                data.energy_buf[0],
                                data.energy_buf[0] + data.energy_delta[()],
                                data.alpha[()],
                                data.ls_iter[()],
                                data.sim_config_data.max_ls_iter[()],
                                exhausted,
                                data.converged,
                            )
                    else:
                        if not accepted and data.ls_iter[()] + 1 >= data.sim_config_data.max_ls_iter[()]:
                            accepted = True
                    accepted = accepted and not guard_failed
                    if guard_failed:
                        data.alpha[()] = qd.f64(0.0)
                        data.ls_cond[()] = 0
                    elif accepted:
                        data.ls_cond[()] = 0
                    else:
                        data.alpha[()] = data.alpha[()] * 0.5
                        data.ls_iter[()] = data.ls_iter[()] + 1
                        if data.ls_iter[()] >= data.sim_config_data.max_ls_iter[()]:
                            data.ls_cond[()] = 0

        # Post-acceptance qd.func calls need an explicit gate too: QUERY may have yielded before reaching them.
        with qd.checkpoint(ContactCheckpoint.LINE_SEARCH_POST, yield_on=data.checkpoint_never_yield):
            for _ in range(1):
                if data.pcg_solver_data.is_failed[()] != 0:
                    data.alpha[()] = qd.f64(0.0)
            if qd.static(data.has_fem):
                step_fem_forward(data.fem_data, data.alpha[()])
                forward_fem_global_vertices(data.fem_data, data.global_vertex_data)
            if qd.static(data.has_rigid):
                step_rigid_forward(data.rigid_data, data.alpha[()])
            if qd.static(data.has_rigid_forest):
                compute_endpoint_fk(
                    data.rigid_forest_data,
                    data.rigid_data,
                    data.rigid_contact_proxy_data,
                )
            if qd.static(data.has_rigid_contact_proxy):
                step_proxy_forward(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.alpha[()],
                )
                forward_proxy_global_vertices(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.global_vertex_data,
                )
                finalize_restoration_step(
                    data.rigid_contact_proxy_data,
                    data.rigid_data,
                    data.rigid_forest_data,
                    data.alpha[()],
                )

            for _ in range(1):
                rejected = data.converged[()] == 0 and data.alpha[()] == 0.0
                pcg_failed = data.pcg_solver_data.is_failed[()] != 0
                linear_failed = data.global_linear_system_data.matrix.triplet_overflow[()] != 0
                if qd.static(data.has_fem):
                    linear_failed = linear_failed or data.global_linear_system_data.matrix.bcoo_valid[()] == 0
                if qd.static(data.has_contact):
                    linear_failed = linear_failed or data.contact_data.intersection_flag[()] != 0
                if qd.static(data.has_rigid_contact_proxy):
                    linear_failed = linear_failed or data.rigid_contact_proxy_data.frame_failed[()] != 0

                exhausted = data.newton_iter[()] >= data.sim_config_data.max_newton_iter[()]
                failed = rejected or pcg_failed or linear_failed or (exhausted and data.converged[()] == 0)
                data.frame_failed[()] = qd.i32(failed)
                converged = data.converged[()] != 0 and data.newton_iter[()] > 1
                data.newton_cond[()] = qd.i32(not converged and not failed and not exhausted)

            if qd.static(data.has_rigid):
                set_newton_active(data.rigid_data, data.newton_cond[()])
                build_rigid_preconditioner(data.rigid_data, compute_envelope=False)

    if qd.static(data.has_et_check):
        with qd.checkpoint(
            ContactCheckpoint.ET_OVERFLOW,
            yield_on=et_overflow,
        ):
            if qd.static(True):
                data.contact_reset_initial_intersections_action.kernel(
                    *data.contact_reset_initial_intersections_action.data
                )
                data.broad_phase_detect_initial_intersections_action.kernel(
                    *data.broad_phase_detect_initial_intersections_action.data
                )
                data.contact_flag_et_intersections_action.kernel(*data.contact_flag_et_intersections_action.data)
        with qd.checkpoint(
            ContactCheckpoint.ET_FAILURE,
            yield_on=data.contact_data.et_yield_flag,
        ):
            for _ in range(1):
                data.contact_data.et_yield_flag[()] = data.contact_data.et_yield_flag[()]

    if qd.static(data.has_rigid_contact_proxy):
        with qd.checkpoint(
            ContactCheckpoint.KKT_FAILURE,
            yield_on=data.frame_failed,
        ):
            for _ in range(1):
                data.frame_failed[()] = data.frame_failed[()]

    with qd.checkpoint(ContactCheckpoint.FINALIZE, yield_on=data.checkpoint_never_yield):
        if qd.static(data.has_rigid):
            update_rigid_velocity(data.rigid_data)
        if qd.static(data.has_fem):
            update_fem_velocity(data.fem_data, data.sim_config_data)
            copy_fem_x_prev(data.fem_data)
        if qd.static(data.has_rigid_contact_proxy):
            recover_reaction(
                data.rigid_contact_proxy_data,
                data.rigid_data,
                data.rigid_forest_data,
            )
            copy_proxy_previous_state(
                data.rigid_contact_proxy_data,
                data.rigid_data,
                data.rigid_forest_data,
            )
        if qd.static(data.has_contact):
            data.contact_shrink_assembly_padding_action.kernel(*data.contact_shrink_assembly_padding_action.data)
