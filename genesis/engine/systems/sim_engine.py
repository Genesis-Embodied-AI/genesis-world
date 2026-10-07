from __future__ import annotations

from enum import IntEnum
from typing import TypeVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .bcoo_matrix import sort_reduce_bcoo, zero_bcoo_triplets
from .consistent_ipc_contact import ConsistentIPCContactConstitution
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
from .finite_element.finite_element_method import forward_fem_scene_vertices
from .global_body_manager import GlobalBodyManager, compute_vertex_offsets
from .global_linear_system import GlobalLinearSystem
from .global_surface_manager import GlobalSurfaceManager
from .global_vertex_manager import (
    GlobalVertexManager,
    record_safe_positions,
    reset_trajectory,
    zero_in_contact,
)
from .lbvh_broad_phase import LBVHBroadPhase
from .rigid_contact_assemble import RigidContactAssemble
from .rigid_contact_proxy import RigidContactProxySystem
from .rigid_joint_forest import RigidJointForestSystem
from .rigid_system import RigidSystem
from .pcg_solver import PCGSolver
from .sim_config import SimConfig
from .sim_system import SimPipeline, SimSystem

T = TypeVar("T", bound=SimSystem)


class ContactCheckpoint(IntEnum):
    """Stable checkpoint IDs owned by the SimEngine graph."""

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


@qd.data_oriented  # WORKAROUND: Quadrants bound graph self must be data-oriented.
class SimEngine:
    """Python host coordinator for the graph-native Newton runtime."""

    def __init__(self) -> None:
        if gs.backend == gs.cpu:
            raise RuntimeError("SimEngine does not support the CPU backend")
        if not gs.use_ndarray:
            raise RuntimeError("SimEngine requires the ndarray backend")

        self.systems: dict[type, SimSystem] = {}
        self._dependencies: dict[SimSystem, bool] = {}
        self._is_built = False
        self._solver_params: tuple[int, float] | None = None
        self.genesis_serial_pipeline = False
        self.has_rigid = False
        self.has_fem = False
        self.has_contact = False
        self.has_rigid_contact_proxy = False
        self.has_rigid_contact_assemble = False
        self.has_rigid_forest = False
        self.has_et_check = False
        self._visualizer_server = None
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
        if self._is_built:
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
        if self._is_built:
            raise RuntimeError("add_system() is only valid before build_systems()")
        system_type = type(system)
        if system_type in self.systems:
            raise RuntimeError(f"SimSystem {system_type.__name__} is already registered")
        system._set_engine(self)
        # WORKAROUND: expose each data-oriented System directly from the Engine
        # graph root until Quadrants can traverse ordinary Python registries.
        setattr(self, f"_sim_system_{len(self.systems)}", system)
        self.systems[system_type] = system

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
        if self._is_built:
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
        if self._is_built:
            raise RuntimeError("SimEngine is already built")
        self.rigid = self.find(RigidSystem)
        self.fem_system = self.find(FiniteElementMethod)
        self.global_body_system = self.find(GlobalBodyManager)
        self.global_vertex_system = self.find(GlobalVertexManager)
        self.global_surface_system = self.find(GlobalSurfaceManager)
        self.contact_system = self.find(ContactSystem)
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
            self.global_body_system is None or self.global_vertex_system is None or self.global_surface_system is None
        ):
            raise RuntimeError("FiniteElementMethod requires global body, vertex, and surface managers")

        self.has_rigid = self.rigid is not None
        self.has_fem = self.fem_system is not None
        self.has_contact = self.contact_system is not None
        self.has_rigid_contact_proxy = self.rigid_contact_proxy is not None
        self.has_rigid_contact_assemble = self.rigid_contact_assemble is not None
        self.has_rigid_forest = self.rigid_forest is not None
        self.has_et_check = self.contact_system is not None and self.contact_system.intersection_check
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
        self._is_built = True

    def _make_graph_fastcache_key(self) -> int:
        """Encode Python-static graph topology that Quadrants cannot infer from ndarray arguments."""
        contact = self.contact_system
        broad_phase = self.broad_phase_system
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
            self.global_linear_system_system.data.matrix.legacy_sort_reduce,
            contact is not None and contact.genesis_legacy_sort_reduce,
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
        if not self._is_built:
            raise RuntimeError("build_systems() must run before wire_solver_params()")
        self.sim_config_system.wire(
            dt=dt,
            tol=tol,
            max_newton_iter=max_newton_iter,
            max_pcg_iter=max_pcg_iter,
            max_ls_iter=max_ls_iter,
        )
        self._solver_params = (int(max_ls_iter), float(pcg_tol_rate))

    def _initialize_runtime_fields(self, max_ls_iter: int) -> None:
        """Allocate mutable engine-owned graph state before binding pipelines."""
        self.newton_cond = qd.ndarray(qd.i32, shape=())
        self.ls_cond = qd.ndarray(qd.i32, shape=())
        self.converged = qd.ndarray(qd.i32, shape=())
        self.frame_failed = qd.ndarray(qd.i32, shape=())
        self.newton_iter = qd.ndarray(qd.i32, shape=())
        self.max_pcg_iters = qd.ndarray(qd.i32, shape=())
        self.total_pcg_iters = qd.ndarray(qd.i32, shape=())
        self.ls_iter = qd.ndarray(qd.i32, shape=())
        self.alpha = qd.ndarray(qd.f64, shape=())
        self.max_disp = qd.ndarray(qd.f64, shape=())
        self.energy_delta = qd.ndarray(qd.f64, shape=())
        self.checkpoint_never_yield = qd.ndarray(qd.i32, shape=())
        self.energy_buf = qd.ndarray(qd.f64, shape=(max_ls_iter + 1,))
        self.max_pcg_iters.from_numpy(np.array(0, dtype=np.int32))
        self.total_pcg_iters.from_numpy(np.array(0, dtype=np.int32))
        self.checkpoint_never_yield.from_numpy(np.array(0, dtype=np.int32))

    def init(self) -> None:
        if self.step_pipeline is not None:
            raise RuntimeError("SimEngine is already initialized")
        if self._solver_params is None:
            raise RuntimeError("wire_solver_params() must run before init()")
        max_ls_iter, pcg_tol_rate = self._solver_params

        self.sim_config_system.init()
        if self.global_body_system is not None:
            self.global_body_system.init()
            self.global_vertex_system.init()
            self.global_surface_system.init()
        if self.contact_system is not None:
            self.contact_system.init()
        dof_offset = 0
        if self.rigid is not None:
            self.rigid.init(dof_offset)
            dof_offset += self.rigid.storage_dof_count
        dof_block_base = dof_offset // 3

        if self.fem_system is not None:
            self.fem_system.init(dof_offset)
            dof_offset += self.fem_system.data.x.shape[0] * 3

        if self.rigid_contact_proxy is not None:
            self.rigid_contact_proxy.init()
            self.rigid_forest.init()
            dof_offset += self.rigid_contact_proxy.n_pairs * 6

        n_block_rows = dof_offset // 3
        contact_data = None if self.contact_system is None else self.contact_system.data
        max_contact_body_triplets = contact_data.unique_triplet_rows.shape[0] if contact_data is not None else 0
        if self.rigid_contact_assemble is not None:
            max_contact_body_triplets = (
                contact_data.unique_triplet_rows.shape[0] * 4 + contact_data.unique_doublet_vertices.shape[0]
            )
        self.global_linear_system_system.init()
        linear_data = self.global_linear_system_system.data
        if (
            linear_data.matrix.shape != (n_block_rows, n_block_rows)
            or linear_data.b_rhs.shape[0] != max(dof_offset, 1)
            or linear_data.matrix.triplet_row.shape[0] < max_contact_body_triplets
        ):
            raise RuntimeError("GlobalLinearSystem data was not sized for the built simulation")
        self.pcg_solver_system.init(dof_offset, n_block_rows, pcg_tol_rate)

        self._initialize_runtime_fields(max_ls_iter)
        init_yield_callbacks = {}
        step_yield_callbacks = {
            ContactCheckpoint.SORT: self.global_linear_system_system.on_triplet_overflow_yield,
        }
        if self.contact_system is not None:
            contact = self.contact_system
            init_yield_callbacks.update(
                {
                    ContactCheckpoint.INITIAL_INTERSECTION: contact.on_initial_intersection_yield,
                    ContactCheckpoint.QUERY: contact.on_query_yield,
                }
            )
            step_yield_callbacks.update(
                {
                    ContactCheckpoint.FRICTION: contact.on_friction_yield,
                    ContactCheckpoint.COUNT: contact.on_count_yield,
                    ContactCheckpoint.FILTER: contact.on_filter_yield,
                    ContactCheckpoint.QUERY: contact.on_query_yield,
                    ContactCheckpoint.ET_OVERFLOW: contact.on_et_overflow_yield,
                    ContactCheckpoint.ET_FAILURE: contact.on_et_failure_yield,
                }
            )
        if self.rigid_contact_proxy is not None:
            step_yield_callbacks[ContactCheckpoint.KKT_FAILURE] = self.rigid_contact_proxy.on_kkt_failure_yield

        never_yield = self.checkpoint_never_yield
        contact_data = None if self.contact_system is None else self.contact_system.data
        self.pair_overflow = never_yield if contact_data is None else contact_data.overflow_flag
        self.assembly_overflow = never_yield if contact_data is None else contact_data.count_overflow_flag
        self.padding_overflow = never_yield if contact_data is None else contact_data.contact_padding_overflow
        self.triplet_overflow = self.global_linear_system_system.data.matrix.triplet_overflow
        self.friction_overflow = never_yield if contact_data is None else contact_data.friction_overflow_flag
        self.et_overflow = never_yield if contact_data is None else contact_data.et_overflow_flag
        self.graph_fastcache_key = self._make_graph_fastcache_key()

        init_contact_pipeline = SimPipeline(
            self.init_contact_graph,
            yield_callbacks=init_yield_callbacks,
        )
        step_pipeline = SimPipeline(
            self.step_graph,
            yield_callbacks=step_yield_callbacks,
        )
        self.init_contact_pipeline = init_contact_pipeline
        self.step_pipeline = step_pipeline
        self.initialize_global_resources()
        if self.contact_system is not None:
            self._initialize_contact()
        self._solver_params = None

    def sync_from_solvers(self) -> None:
        """Synchronize externally authored Scene state without rebuilding graph resources."""
        self.sync_from_solvers_graph()
        if self.contact_system is not None:
            self._initialize_contact()

    def _initialize_contact(self) -> None:
        self.init_contact_pipeline.run()
        self.contact_system.raise_if_initial_intersection()

    def step(self) -> None:
        pipeline = self.step_pipeline
        if pipeline is None:
            raise RuntimeError("SimEngine.init() must run before step()")
        contact = None if self.contact_system is None else self.contact_system.data
        linear = self.global_linear_system_system.data
        pipeline.run()
        if self.fem_system is not None:
            forward_fem_scene_vertices(self.fem_system.data)
        if bool(qd_to_numpy(self.frame_failed)):
            details = [
                f"newton={int(qd_to_numpy(self.newton_iter))}",
                f"pcg={int(qd_to_numpy(self.max_pcg_iters))}",
                f"line_search={int(qd_to_numpy(self.ls_iter))}",
                f"triplet_overflow={int(qd_to_numpy(linear.matrix.triplet_overflow))}",
                f"bcoo_valid={int(qd_to_numpy(linear.matrix.bcoo_valid))}",
            ]
            if contact is not None:
                details.extend(
                    (
                        f"intersection={int(qd_to_numpy(contact.intersection_flag))}",
                        f"pairs_pt={int(qd_to_numpy(contact.n_pairs_pt))}",
                        f"pairs_ee={int(qd_to_numpy(contact.n_pairs_ee))}",
                        f"pairs_ph={int(qd_to_numpy(contact.n_pairs_ph))}",
                        f"active_pairs={int(qd_to_numpy(contact.n_active_pairs))}",
                        f"ccd_alpha={float(qd_to_numpy(contact.ccd_alpha)):.6g}",
                    )
                )
                for channel in ("pt", "ee", "ph"):
                    count = int(qd_to_numpy(getattr(contact, f"n_pairs_{channel}")))
                    if count:
                        alphas = qd_to_numpy(getattr(contact, f"ccd_alpha_{channel}"))[:count]
                        minimum_index = int(np.argmin(alphas))
                        pair = qd_to_numpy(getattr(contact, f"pairs_{channel}"))[minimum_index]
                        details.append(
                            f"ccd_{channel}=({float(alphas[minimum_index]):.6g},"
                            f" pair={tuple(int(value) for value in pair)})"
                        )
                        if channel == "ph":
                            surface = self.global_surface_system.data
                            vertex = self.global_vertex_system.data
                            surface_vertex = int(pair[0])
                            vertex_id = int(qd_to_numpy(surface.surf_verts)[surface_vertex])
                            current = qd_to_numpy(vertex.positions)[vertex_id]
                            endpoint = qd_to_numpy(vertex.trajectory_end_positions)[vertex_id]
                            thickness = float(qd_to_numpy(vertex.thicknesses)[vertex_id])
                            details.append(
                                f"ph_path=(current={tuple(float(value) for value in current)},"
                                f" endpoint={tuple(float(value) for value in endpoint)},"
                                f" thickness={thickness:.6g})"
                            )
            raise RuntimeError(f"SimEngine Newton solve failed ({', '.join(details)})")

    @qd.kernel(fastcache=True)
    def initialize_global_resources(self):
        data = self
        if qd.static(self.graph_fastcache_key < 0):
            data.frame_failed[()] = 0
        if qd.static(data.has_fem):
            self.fem_system.on_initialize_global_vertices()
            compute_vertex_offsets(self.global_body_system.data, self.fem_system.data)
        if qd.static(data.has_rigid_contact_proxy):
            self.rigid_contact_proxy.on_initialize_state()
            self.rigid_contact_proxy.on_prepare_metric()
            self.rigid_contact_proxy.on_initialize_global_vertices()

    @qd.kernel(fastcache=True)
    def sync_from_solvers_graph(self):
        data = self
        if qd.static(self.graph_fastcache_key < 0):
            data.frame_failed[()] = 0
        if qd.static(data.has_fem):
            self.fem_system.on_sync_from_scene()
        if qd.static(data.has_rigid_contact_proxy):
            self.rigid_contact_proxy.on_initialize_state()
            self.rigid_contact_proxy.on_reset_frame()
            self.rigid_contact_proxy.on_prepare_metric()
            self.rigid_contact_proxy.on_initialize_global_vertices()

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def init_contact_graph(self):
        data = self
        if qd.static(self.graph_fastcache_key < 0):
            data.frame_failed[()] = 0
        with qd.checkpoint(ContactCheckpoint.FRAME, yield_on=data.checkpoint_never_yield):
            # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
            if qd.static(True):
                self.fem_system.on_forward_global_vertices()
                reset_trajectory(self.global_vertex_system.data)
                if qd.static(data.genesis_serial_pipeline):
                    self.broad_phase_system.on_build_triangles()
                    self.broad_phase_system.on_build_edges()
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            self.broad_phase_system.on_build_triangles()
                        with qd.graph.parallel():
                            self.broad_phase_system.on_build_edges()
        if qd.static(data.has_et_check):
            with qd.checkpoint(
                ContactCheckpoint.INITIAL_INTERSECTION,
                yield_on=self.et_overflow,
            ):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.contact_system.on_reset_initial_intersections()
                    self.broad_phase_system.on_detect_initial_intersections()
        with qd.checkpoint(ContactCheckpoint.QUERY, yield_on=self.pair_overflow):
            # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
            if qd.static(True):
                self.contact_system.on_reset_collision_counts()
                if qd.static(data.genesis_serial_pipeline):
                    self.broad_phase_system.on_query_trajectory()
                    if qd.static(self.contact_system.has_halfplanes):
                        self.contact_system.on_query_halfplanes()
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            self.broad_phase_system.on_query_pt()
                        with qd.graph.parallel():
                            self.broad_phase_system.on_query_ee()
                        with qd.graph.parallel():
                            if qd.static(self.contact_system.has_halfplanes):
                                self.contact_system.on_query_halfplanes()
        with qd.checkpoint(ContactCheckpoint.COUNT, yield_on=data.checkpoint_never_yield):
            # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
            if qd.static(True):
                self.contact_system.on_reset_counted_demand()
                if qd.static(data.genesis_serial_pipeline):
                    self.contact_constitution_system.on_count_active()
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            self.contact_constitution_system.on_count_active_pt()
                        with qd.graph.parallel():
                            self.contact_constitution_system.on_count_active_ee()
                        with qd.graph.parallel():
                            if qd.static(self.contact_system.has_halfplanes):
                                self.contact_constitution_system.on_count_active_ph()
                self.contact_system.on_check_assembly_capacity()

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def step_graph(self):
        data = self
        if qd.static(self.graph_fastcache_key < 0):
            data.frame_failed[()] = 0
        with qd.checkpoint(ContactCheckpoint.FRAME, yield_on=data.checkpoint_never_yield):
            if qd.static(data.has_contact):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.contact_system.on_update_adaptive_kappa()
                    self.contact_system.on_reset_frame()
            if qd.static(data.has_rigid):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.rigid.on_predict()
                    self.rigid.on_assemble_candidate_rows()
            if qd.static(data.has_rigid_contact_proxy):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.rigid_contact_proxy.on_mark_mechanism_constrained()
            if qd.static(data.has_rigid):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.rigid.on_initialize_newton()
            if qd.static(data.has_rigid_contact_proxy):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.rigid_contact_proxy.on_reset_frame()
                    self.rigid_contact_proxy.on_prepare_metric()
                    self.rigid_contact_proxy.on_prepare_tolerance()
            if qd.static(data.has_fem):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.fem_system.on_predict()
                    self.fem_system.on_forward_global_vertices()
            for _ in range(1):
                data.newton_iter[()] = 0
                data.max_pcg_iters[()] = 0
                data.total_pcg_iters[()] = 0
                data.ls_iter[()] = 0
                data.frame_failed[()] = 0
                data.newton_cond[()] = 1

        with qd.checkpoint(ContactCheckpoint.FRICTION, yield_on=self.friction_overflow):
            if qd.static(data.has_contact):  # noqa: SIM102
                if qd.static(self.contact_system.has_friction):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        if qd.static(data.genesis_serial_pipeline):
                            self.contact_constitution_system.on_snapshot_friction()
                        else:
                            self.contact_constitution_system.on_snapshot_lagged_positions()
                            with qd.graph.parallel_context():
                                with qd.graph.parallel():
                                    self.contact_constitution_system.on_filter_friction_pairs_pt()
                                with qd.graph.parallel():
                                    self.contact_constitution_system.on_filter_friction_pairs_ee()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_halfplanes):
                                        self.contact_constitution_system.on_filter_friction_pairs_ph()

        while qd.graph.do_while(data.newton_cond):
            with qd.checkpoint(ContactCheckpoint.COUNT, yield_on=self.assembly_overflow):
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_initialize_newton()
                if qd.static(data.has_rigid_forest):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_forest.on_compute_endpoint_fk()
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_prepare_constraint()
                if qd.static(data.has_contact):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.contact_system.on_reset_counted_demand()
                        if qd.static(data.genesis_serial_pipeline):
                            self.contact_constitution_system.on_count_active()
                        else:
                            with qd.graph.parallel_context():
                                with qd.graph.parallel():
                                    self.contact_constitution_system.on_count_active_pt()
                                with qd.graph.parallel():
                                    self.contact_constitution_system.on_count_active_ee()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_halfplanes):
                                        self.contact_constitution_system.on_count_active_ph()
                        self.contact_system.on_check_assembly_capacity()

            with qd.checkpoint(ContactCheckpoint.FILTER, yield_on=self.padding_overflow):
                for _ in range(1):
                    data.newton_iter[()] = data.newton_iter[()] + 1
                if qd.static(data.has_contact):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.contact_system.on_tick_adaptive_kappa_newton()
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        zero_in_contact(self.global_vertex_system.data)
                        self.contact_system.on_reset_assembly_counts()
                        if qd.static(data.genesis_serial_pipeline):
                            self.contact_constitution_system.on_filter_assemble()
                        else:
                            with qd.graph.parallel_context():
                                with qd.graph.parallel():
                                    self.contact_constitution_system.on_filter_assemble_pt()
                                with qd.graph.parallel():
                                    self.contact_constitution_system.on_filter_assemble_ee()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_halfplanes):
                                        self.contact_constitution_system.on_filter_assemble_ph()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_friction):
                                        self.contact_constitution_system.on_friction_assemble_pt()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_friction):
                                        self.contact_constitution_system.on_friction_assemble_ee()
                                with qd.graph.parallel():
                                    if qd.static(
                                        self.contact_system.has_friction and self.contact_system.has_halfplanes
                                    ):
                                        self.contact_constitution_system.on_friction_assemble_ph()
                        self.contact_system.on_check_assembly_padding()

            with qd.checkpoint(ContactCheckpoint.SORT, yield_on=self.triplet_overflow):
                if qd.static(data.has_contact):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.contact_system.on_sort_reduce()
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.global_linear_system_system.on_report_extents()
                    self.global_linear_system_system.on_derive_extents()
                if qd.static(data.has_contact and not data.has_rigid_contact_assemble):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.global_linear_system_system.on_compute_n_triplets()

            with qd.checkpoint(ContactCheckpoint.SOLVE, yield_on=data.checkpoint_never_yield):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.global_linear_system_system.on_zero_rhs()
                    zero_bcoo_triplets(self.global_linear_system_system.data.matrix)

                if qd.static(data.has_rigid):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid.on_assemble(data.has_rigid_forest)
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.global_linear_system_system.on_assemble_subsystems()
                if qd.static(data.has_contact and not data.has_rigid_contact_assemble):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        distribute_fem_gradient_kernel(
                            self.contact_system.data,
                            self.fem_system.data,
                            self.global_linear_system_system.data,
                        )
                        distribute_fem_fem_kernel(
                            self.contact_system.data,
                            self.fem_system.data,
                            self.global_linear_system_system.data,
                        )

                if qd.static(data.has_fem):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        sort_reduce_bcoo(self.global_linear_system_system.data.matrix)
                        initialize_fem_diag_preconditioner(self.fem_preconditioner_system.data)
                        gather_fem_diag_preconditioner(
                            self.fem_preconditioner_system.data,
                            self.global_linear_system_system.data,
                        )
                        invert_fem_diag_preconditioner(self.fem_preconditioner_system.data)
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_capture_physical_gradient()
                if qd.static(data.has_rigid_forest):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_forest.on_prepare_particular()
                        self.rigid_forest.on_particular_spmv()
                        self.rigid_forest.on_project_physical_rhs()
                        self.rigid_forest.on_build_preconditioner()

            # WORKAROUND: Quadrants #956 may execute a top-level qd.func after a yielding
            # checkpoint once on the yielding launch. The explicit
            # no-yield gate keeps the complete PCG child loop behind the resume
            # boundary. Remove it after the compiler issue is fixed.
            # https://github.com/Genesis-Embodied-AI/quadrants/issues/956
            with qd.checkpoint(ContactCheckpoint.PCG_SOLVE, yield_on=data.checkpoint_never_yield):
                if qd.static(True):
                    self.pcg_solver_system.solve(
                        self.sim_config_system.data.max_pcg_iter[()],
                        data.max_pcg_iters,
                        data.total_pcg_iters,
                    )

            # Keep all post-PCG qd.func calls under a flat gate as well. Otherwise an earlier capacity yield can skip
            # direct kernel tasks but still enter one of these inlined functions.
            with qd.checkpoint(ContactCheckpoint.SOLVE_POST, yield_on=data.checkpoint_never_yield):
                if qd.static(data.has_rigid):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid.on_negate_direction(data.has_rigid_forest)
                if qd.static(data.has_fem):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.fem_system.on_negate_direction()
                if qd.static(data.has_rigid_forest):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_forest.on_expand_solution()
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_prepare_path_limit()

                for _ in range(1):
                    data.max_disp[()] = qd.f64(0.0)
                if qd.static(data.has_fem):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.fem_system.on_contribute_newton_max_displacement(data.max_disp)
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_contribute_newton_max_displacement(data.max_disp)

                for _ in range(1):
                    rigid_converged = True
                    fem_converged = True
                    if qd.static(data.has_rigid_contact_proxy):
                        rigid_converged = (
                            data.max_disp[()]
                            <= self.sim_config_system.data.tol[()] * self.sim_config_system.data.dt[()]
                        )
                    elif qd.static(data.has_rigid):
                        rigid_converged = (
                            self.rigid.data.gradient_squared[()] <= self.sim_config_system.data.tol[()] ** 2
                        )
                    if qd.static(data.has_fem and not data.has_rigid_contact_proxy):
                        fem_converged = (
                            data.max_disp[()]
                            <= self.sim_config_system.data.tol[()] * self.sim_config_system.data.dt[()]
                        )
                    data.converged[()] = qd.i32(rigid_converged and fem_converged)
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_apply_convergence(data.converged)

                if qd.static(data.has_rigid):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid.on_record_start_point()
                        self.rigid.on_energy()
                if qd.static(data.has_fem):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.fem_system.on_record_start_point()
                        self.fem_system.on_reset_energy()
                        self.fem_system.on_energy()
                        record_safe_positions(self.global_vertex_system.data)
                        self.fem_system.on_publish_trajectory_end_positions()
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_record_start_point()
                        self.rigid_contact_proxy.on_forward_global_vertices()
                        self.rigid_contact_proxy.on_publish_trajectory_end_positions()
                if qd.static(data.has_contact):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        if qd.static(data.genesis_serial_pipeline):
                            self.contact_system.on_reset_contact_energy()
                            self.contact_constitution_system.on_contact_energy()
                            self.contact_system.on_sum_contact_energy()
                        else:
                            self.contact_system.on_reset_contact_energy()
                            with qd.graph.parallel_context():
                                with qd.graph.parallel():
                                    self.contact_constitution_system.on_filter_energy_pt()
                                with qd.graph.parallel():
                                    self.contact_constitution_system.on_filter_energy_ee()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_halfplanes):
                                        self.contact_constitution_system.on_filter_energy_ph()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_friction):
                                        self.contact_constitution_system.on_friction_energy_pt()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_friction):
                                        self.contact_constitution_system.on_friction_energy_ee()
                                with qd.graph.parallel():
                                    if qd.static(
                                        self.contact_system.has_friction and self.contact_system.has_halfplanes
                                    ):
                                        self.contact_constitution_system.on_friction_energy_ph()
                            self.contact_system.on_sum_contact_energy()
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_compute_restoration_energy(False)

                for _ in range(1):
                    baseline = qd.f64(0.0)
                    if qd.static(data.has_rigid):
                        baseline = baseline + self.rigid.data.rigid_energy[()]
                    if qd.static(data.has_fem):
                        baseline = baseline + self.fem_system.data.fem_energy[()]
                    if qd.static(data.has_contact):
                        baseline = baseline + self.contact_system.data.contact_energy_value[()]
                    if qd.static(data.has_rigid_contact_proxy):
                        baseline = baseline + self.rigid_contact_proxy.data.restoration_energy[()]
                    data.energy_buf[0] = baseline
                if qd.static(data.has_rigid_forest):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_forest.on_compute_merit_directional_derivative()
                        self.rigid_contact_proxy.on_initialize_merit()

                if qd.static(data.has_contact):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        if qd.static(data.genesis_serial_pipeline):
                            self.broad_phase_system.on_build_triangles()
                            self.broad_phase_system.on_build_edges()
                        else:
                            with qd.graph.parallel_context():
                                with qd.graph.parallel():
                                    self.broad_phase_system.on_build_triangles()
                                with qd.graph.parallel():
                                    self.broad_phase_system.on_build_edges()

            with qd.checkpoint(ContactCheckpoint.QUERY, yield_on=self.pair_overflow):
                if qd.static(data.has_contact):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.contact_system.on_reset_collision_counts()
                        if qd.static(data.genesis_serial_pipeline):
                            self.broad_phase_system.on_query_trajectory()
                            if qd.static(self.contact_system.has_halfplanes):
                                self.contact_system.on_query_halfplanes()
                        else:
                            with qd.graph.parallel_context():
                                with qd.graph.parallel():
                                    self.broad_phase_system.on_query_pt()
                                with qd.graph.parallel():
                                    self.broad_phase_system.on_query_ee()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_halfplanes):
                                        self.contact_system.on_query_halfplanes()

            with qd.checkpoint(ContactCheckpoint.CCD, yield_on=data.checkpoint_never_yield):
                if qd.static(data.has_contact):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.contact_system.on_initialize_ccd()
                        if qd.static(data.genesis_serial_pipeline):
                            self.contact_system.on_ccd()
                        else:
                            with qd.graph.parallel_context():
                                with qd.graph.parallel():
                                    self.contact_system.on_compute_ccd_alpha_pt()
                                with qd.graph.parallel():
                                    self.contact_system.on_compute_ccd_alpha_ee()
                                with qd.graph.parallel():
                                    if qd.static(self.contact_system.has_halfplanes):
                                        self.contact_system.on_compute_ccd_alpha_ph()
                            self.contact_system.on_reduce_ccd_alpha()

            with qd.checkpoint(ContactCheckpoint.LINE_SEARCH, yield_on=data.checkpoint_never_yield):
                for _ in range(1):
                    initial_alpha = qd.f64(1.0)
                    if qd.static(data.has_contact):
                        initial_alpha = qd.min(initial_alpha, self.contact_system.data.ccd_alpha[()])
                    if qd.static(data.has_rigid_contact_proxy):
                        initial_alpha = qd.min(
                            initial_alpha,
                            self.rigid_contact_proxy.data.fk_alpha[()],
                        )
                    data.alpha[()] = initial_alpha
                    data.ls_cond[()] = 1
                    data.ls_iter[()] = 0

            # No line-search operation can yield to the host. Its child WHILE stays outside the explicit phase marker,
            # while each inlined trial stage gets a flat gate for the same qd.func suffix workaround as PCG.
            while qd.graph.do_while(data.ls_cond):
                with qd.checkpoint(ContactCheckpoint.LINE_SEARCH_TRIAL, yield_on=data.checkpoint_never_yield):
                    for _ in range(1):
                        if self.pcg_solver_system.data.is_failed[()] != 0:
                            data.alpha[()] = qd.f64(0.0)

                    if qd.static(data.has_rigid):
                        # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                        if qd.static(True):
                            self.rigid.on_step_forward(data.alpha[()])
                            self.rigid.on_energy()
                    if qd.static(data.has_rigid_forest):
                        # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                        if qd.static(True):
                            self.rigid_forest.on_compute_endpoint_fk()
                    if qd.static(data.has_fem):
                        # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                        if qd.static(True):
                            self.fem_system.on_step_forward(data.alpha[()])
                            self.fem_system.on_forward_global_vertices()
                            self.fem_system.on_reset_energy()
                            self.fem_system.on_energy()
                    if qd.static(data.has_rigid_contact_proxy):
                        # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                        if qd.static(True):
                            self.rigid_contact_proxy.on_step_forward(data.alpha[()])
                            self.rigid_contact_proxy.on_forward_global_vertices()
                            self.rigid_contact_proxy.on_evaluate_trial_guard()
                    if qd.static(data.has_contact):
                        # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                        if qd.static(True):
                            if qd.static(data.genesis_serial_pipeline):
                                self.contact_system.on_reset_contact_energy()
                                self.contact_constitution_system.on_contact_energy()
                                self.contact_system.on_sum_contact_energy()
                            else:
                                self.contact_system.on_reset_contact_energy()
                                with qd.graph.parallel_context():
                                    with qd.graph.parallel():
                                        self.contact_constitution_system.on_filter_energy_pt()
                                    with qd.graph.parallel():
                                        self.contact_constitution_system.on_filter_energy_ee()
                                    with qd.graph.parallel():
                                        if qd.static(self.contact_system.has_halfplanes):
                                            self.contact_constitution_system.on_filter_energy_ph()
                                    with qd.graph.parallel():
                                        if qd.static(self.contact_system.has_friction):
                                            self.contact_constitution_system.on_friction_energy_pt()
                                    with qd.graph.parallel():
                                        if qd.static(self.contact_system.has_friction):
                                            self.contact_constitution_system.on_friction_energy_ee()
                                    with qd.graph.parallel():
                                        if qd.static(
                                            self.contact_system.has_friction and self.contact_system.has_halfplanes
                                        ):
                                            self.contact_constitution_system.on_friction_energy_ph()
                                self.contact_system.on_sum_contact_energy()
                    if qd.static(data.has_rigid_contact_proxy):
                        # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                        if qd.static(True):
                            self.rigid_contact_proxy.on_compute_restoration_energy(True)

                    for _ in range(1):
                        trial_energy = qd.f64(0.0)
                        if qd.static(data.has_rigid):
                            trial_energy = trial_energy + self.rigid.data.rigid_energy[()]
                        if qd.static(data.has_fem):
                            trial_energy = trial_energy + self.fem_system.data.fem_energy[()]
                        if qd.static(data.has_contact):
                            trial_energy = trial_energy + self.contact_system.data.contact_energy_value[()]
                        if qd.static(data.has_rigid_contact_proxy):
                            trial_energy = trial_energy + self.rigid_contact_proxy.data.restoration_energy[()]
                            if (
                                self.rigid_contact_proxy.data.restoration_active[()] == 0
                                and self.rigid_contact_proxy.data.merit_evaluated[()] == 0
                            ):
                                trial_energy = trial_energy + self.rigid_contact_proxy.data.test_merit_energy_bias[()]
                            if self.rigid_contact_proxy.data.restoration_triggered[()] == 0:
                                trial_energy = (
                                    trial_energy + self.rigid_contact_proxy.data.ls_forensics_test_energy_bias[()]
                                )
                        data.energy_delta[()] = trial_energy - data.energy_buf[0]

                    for _ in range(1):
                        guard_failed = False
                        if qd.static(data.has_rigid_contact_proxy):
                            guard_failed = self.rigid_contact_proxy.data.frame_failed[()] != 0
                        accepted = data.converged[()] != 0 or data.energy_delta[()] <= 0.0
                        if qd.static(data.has_rigid_contact_proxy):
                            if data.converged[()] == 0:
                                exhausted = data.ls_iter[()] + 1 >= self.sim_config_system.data.max_ls_iter[()]
                                accepted = self.rigid_contact_proxy.on_check_line_search(
                                    data.energy_buf[0],
                                    data.energy_buf[0] + data.energy_delta[()],
                                    data.alpha[()],
                                    data.ls_iter[()],
                                    self.sim_config_system.data.max_ls_iter[()],
                                    exhausted,
                                    data.converged,
                                )
                        else:
                            if not accepted and data.ls_iter[()] + 1 >= self.sim_config_system.data.max_ls_iter[()]:
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
                            if data.ls_iter[()] >= self.sim_config_system.data.max_ls_iter[()]:
                                data.ls_cond[()] = 0

            # Post-acceptance qd.func calls need an explicit gate too: QUERY may have yielded before reaching them.
            with qd.checkpoint(ContactCheckpoint.LINE_SEARCH_POST, yield_on=data.checkpoint_never_yield):
                for _ in range(1):
                    if self.pcg_solver_system.data.is_failed[()] != 0:
                        data.alpha[()] = qd.f64(0.0)
                if qd.static(data.has_fem):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.fem_system.on_step_forward(data.alpha[()])
                        self.fem_system.on_forward_global_vertices()
                if qd.static(data.has_rigid):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid.on_step_forward(data.alpha[()])
                if qd.static(data.has_rigid_forest):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_forest.on_compute_endpoint_fk()
                if qd.static(data.has_rigid_contact_proxy):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid_contact_proxy.on_step_forward(data.alpha[()])
                        self.rigid_contact_proxy.on_forward_global_vertices()
                        self.rigid_contact_proxy.on_finalize_restoration_step(data.alpha[()])

                for _ in range(1):
                    rejected = data.converged[()] == 0 and data.alpha[()] == 0.0
                    pcg_failed = self.pcg_solver_system.data.is_failed[()] != 0
                    linear_failed = self.global_linear_system_system.data.matrix.triplet_overflow[()] != 0
                    if qd.static(data.has_fem):
                        linear_failed = (
                            linear_failed or self.global_linear_system_system.data.matrix.bcoo_valid[()] == 0
                        )
                    if qd.static(data.has_contact):
                        linear_failed = linear_failed or self.contact_system.data.intersection_flag[()] != 0
                    if qd.static(data.has_rigid_contact_proxy):
                        linear_failed = linear_failed or self.rigid_contact_proxy.data.frame_failed[()] != 0

                    exhausted = data.newton_iter[()] >= self.sim_config_system.data.max_newton_iter[()]
                    failed = rejected or pcg_failed or linear_failed or (exhausted and data.converged[()] == 0)
                    data.frame_failed[()] = qd.i32(failed)
                    converged = data.converged[()] != 0 and data.newton_iter[()] > 1
                    data.newton_cond[()] = qd.i32(not converged and not failed and not exhausted)

                if qd.static(data.has_rigid):
                    # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                    if qd.static(True):
                        self.rigid.on_set_newton_active(data.newton_cond[()])
                        self.rigid.on_build_preconditioner(compute_envelope=False)

        if qd.static(data.has_et_check):
            with qd.checkpoint(
                ContactCheckpoint.ET_OVERFLOW,
                yield_on=self.et_overflow,
            ):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.contact_system.on_reset_initial_intersections()
                    self.broad_phase_system.on_detect_initial_intersections()
                    self.contact_system.on_flag_et_intersections()
            with qd.checkpoint(
                ContactCheckpoint.ET_FAILURE,
                yield_on=self.contact_system.data.et_yield_flag,
            ):
                for _ in range(1):
                    self.contact_system.data.et_yield_flag[()] = self.contact_system.data.et_yield_flag[()]

        if qd.static(data.has_rigid_contact_proxy):
            with qd.checkpoint(
                ContactCheckpoint.KKT_FAILURE,
                yield_on=data.frame_failed,
            ):
                for _ in range(1):
                    data.frame_failed[()] = data.frame_failed[()]

        with qd.checkpoint(ContactCheckpoint.FINALIZE, yield_on=data.checkpoint_never_yield):
            if qd.static(data.has_rigid):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.rigid.on_update_velocity()
            if qd.static(data.has_fem):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.fem_system.on_update_velocity()
                    self.fem_system.on_copy_previous_positions()
            if qd.static(data.has_rigid_contact_proxy):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.rigid_contact_proxy.on_recover_reaction()
                    self.rigid_contact_proxy.on_copy_previous_state()
            if qd.static(data.has_contact):
                # WORKAROUND: Quadrants checkpoint lowering rejects bare top-level @qd.func calls.
                if qd.static(True):
                    self.contact_system.on_shrink_assembly_padding()
