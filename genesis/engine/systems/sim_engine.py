from __future__ import annotations

from enum import IntEnum
from typing import TypeVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .contact import CONTACT_CONFIG_DEFAULTS
from .contact_system import ContactSystem
from .finite_element import FEMDiagPreconditioner, FiniteElementMethod
from .finite_element.fem_contact_assemble import (
    distribute_fem_fem_kernel,
    distribute_fem_gradient_kernel,
)
from .global_body_manager import GlobalBodyManager
from .global_linear_system import GlobalLinearSystem
from .global_surface_manager import GlobalSurfaceManager
from .global_vertex_manager import GlobalVertexManager
from .rigid_contact_assemble import RigidContactAssemble
from .rigid_contact_proxy import RigidContactProxySystem
from .rigid_joint_forest import RigidJointForestSystem
from .rigid_system import RigidSystem
from .sim_config import SimConfig
from .sim_system import SimSystem
from .standard_pcg_solver import StandardPCGSolver

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
    PCG_INIT = 16
    PCG_ITERATION = 17
    PCG_FINISH = 18
    LINE_SEARCH_POST = 19


@qd.data_oriented
class SimEngine:
    """GPU graph-native Newton engine composed explicitly from CGQ systems."""

    def __init__(self) -> None:
        if gs.backend == gs.cpu:
            raise RuntimeError("SimEngine does not support the CPU backend")
        if not gs.use_ndarray:
            raise RuntimeError("SimEngine requires the ndarray backend")

        self.systems: dict[type, SimSystem] = {}
        self.is_built_host = False
        self.is_initialized_host = False
        self.params_wired_host = False
        self.genesis_serial_pipeline = False

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
        self.max_pcg_iters.from_numpy(np.array(0, dtype=np.int32))
        self.total_pcg_iters.from_numpy(np.array(0, dtype=np.int32))
        self.checkpoint_never_yield.from_numpy(np.array(0, dtype=np.int32))

        self.sim_config = SimConfig()
        self.add_system(self.sim_config)

    def configure_genesis_serial_pipeline(self, enabled: bool) -> None:
        if self.is_built_host:
            raise RuntimeError("Pipeline scheduling must be configured before build_systems()")
        self.genesis_serial_pipeline = bool(enabled)

    def add_system(self, system: SimSystem) -> None:
        if self.is_built_host:
            raise RuntimeError("add_system() is only valid before build_systems()")
        system_type = type(system)
        if system_type in self.systems:
            raise RuntimeError(f"SimSystem {system_type.__name__} is already registered")
        system._set_engine(self)
        self.systems[system_type] = system

    def find(self, system_type: type[T]) -> T | None:
        system = self.systems.get(system_type)
        if system is None or not system.is_valid:
            return None
        return system

    def require(self, system_type: type[T]) -> T:
        system = self.find(system_type)
        if system is None:
            raise RuntimeError(f"Required system {system_type.__name__} is not registered")
        return system

    def build_systems(self) -> None:
        if self.is_built_host:
            raise RuntimeError("SimEngine systems are already built")
        for system in self.systems.values():
            system.build()
        self.build()

    def build(self) -> None:
        if self.is_built_host:
            raise RuntimeError("SimEngine is already built")
        self.rigid = self.find(RigidSystem)
        self.fem = self.find(FiniteElementMethod)
        self.global_body_manager = self.find(GlobalBodyManager)
        self.global_vertex_manager = self.find(GlobalVertexManager)
        self.global_surface_manager = self.find(GlobalSurfaceManager)
        self.contact = self.find(ContactSystem)
        self.rigid_contact_proxy = self.find(RigidContactProxySystem)
        self.rigid_contact_assemble = self.find(RigidContactAssemble)
        self.rigid_forest = self.find(RigidJointForestSystem)
        self.global_linear_system = self.require(GlobalLinearSystem)
        self.pcg_solver = self.require(StandardPCGSolver)
        self.fem_preconditioner = self.find(FEMDiagPreconditioner)
        if self.fem is not None and self.fem_preconditioner is None:
            raise RuntimeError("FiniteElementMethod requires FEMDiagPreconditioner")
        if self.fem is not None and (
            self.global_body_manager is None
            or self.global_vertex_manager is None
            or self.global_surface_manager is None
        ):
            raise RuntimeError("FiniteElementMethod requires global body, vertex, and surface managers")

        self.has_rigid = self.rigid is not None
        self.has_fem = self.fem is not None
        self.has_contact = self.contact is not None
        self.has_rigid_contact_proxy = self.rigid_contact_proxy is not None
        self.has_rigid_contact_assemble = self.rigid_contact_assemble is not None
        self.has_rigid_forest = self.rigid_forest is not None
        self.has_et_check = self.contact is not None and self.contact.intersection_check_host
        if self.has_rigid_contact_proxy != self.has_rigid_forest:
            raise RuntimeError("RigidContactProxySystem and RigidJointForestSystem must be registered together")
        if self.has_rigid_contact_proxy != self.has_rigid_contact_assemble:
            raise RuntimeError("RigidContactProxySystem and RigidContactAssemble must be registered together")
        if self.has_contact and not self.has_fem:
            raise RuntimeError("The current ContactSystem milestone requires FiniteElementMethod")
        if not self.has_rigid and not self.has_fem:
            raise RuntimeError("SimEngine requires RigidSystem or FiniteElementMethod")
        self.is_built_host = True

    def _make_graph_fastcache_key(self) -> int:
        """Encode Python-static graph topology that Quadrants cannot infer from ndarray arguments."""
        contact = self.contact
        broad_phase = None if contact is None else contact.broad_phase
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
            self.global_linear_system.genesis_legacy_sort_reduce_host,
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
        self.sim_config.dt.from_numpy(np.array(dt, dtype=np.float64))
        self.sim_config.tol.from_numpy(np.array(tol, dtype=np.float64))
        self.sim_config.max_newton_iter.from_numpy(np.array(max_newton_iter, dtype=np.int64))
        self.sim_config.max_pcg_iter.from_numpy(np.array(max_pcg_iter, dtype=np.int64))
        self.sim_config.max_ls_iter.from_numpy(np.array(max_ls_iter, dtype=np.int64))
        self.max_ls_iter_host = max_ls_iter
        self.pcg_tol_rate_host = pcg_tol_rate
        self.energy_buf = qd.ndarray(qd.f64, shape=(max_ls_iter + 1,))
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
        if self.fem is not None:
            self.fem.init(dof_offset)
            dof_offset += self.fem.vert_capacity_host * 3
            n_elastic_triplets = self.fem.n_elastic_triplets()

        proxy_dof_offset = dof_offset
        if self.rigid_contact_proxy is not None:
            dof_offset += self.rigid_contact_proxy.n_pairs_host * 6

        n_block_rows = dof_offset // 3
        max_contact_body_triplets = self.contact.unique_triplet_rows.shape[0] if self.contact is not None else 0
        if self.rigid_contact_assemble is not None:
            max_contact_body_triplets = (
                self.contact.unique_triplet_rows.shape[0] * 4 + self.contact.unique_doublet_vertices.shape[0]
            )
        self.global_linear_system.init(
            n_block_rows,
            n_elastic_triplets,
            max_contact_body_triplets,
            dof_block_base,
            self.pcg_tol_rate_host,
        )
        self.pcg_solver.init(dof_offset, n_block_rows, self.pcg_tol_rate_host)
        if self.rigid_forest is not None:
            self.rigid_contact_proxy.ensure_merit_gradient_capacity(dof_offset)
            self.rigid_forest.init(
                dof_offset,
                self.rigid_contact_proxy.n_bodies_host,
                proxy_dof_offset,
            )
            self.rigid_contact_assemble.init()
        if self.fem_preconditioner is not None:
            self.fem_preconditioner.init()
        self.graph_fastcache_key = self._make_graph_fastcache_key()
        self._initialize_global_resources(self.graph_fastcache_key)
        if self.contact is not None:
            self._initialize_contact()
        self.is_initialized_host = True

    @qd.kernel(fastcache=True)
    def _initialize_global_resources(self, graph_fastcache_key: qd.template()):
        if qd.static(self.has_fem):
            self.fem.initialize_global_vertices(self.global_vertex_manager)
            self.global_body_manager.compute_vertex_offsets(self.fem)
        if qd.static(self.has_rigid_contact_proxy):
            self.rigid_contact_proxy.initialize_proxy_state()
            self.rigid_contact_proxy.prepare_metric()
            self.rigid_contact_proxy.initialize_global_vertices(self.global_vertex_manager)

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def _init_contact_kernel(
        self,
        pair_overflow: qd.types.ndarray(qd.i32, ndim=0),
        et_overflow: qd.types.ndarray(qd.i32, ndim=0),
        graph_fastcache_key: qd.template(),
    ):
        with qd.checkpoint(ContactCheckpoint.FRAME, yield_on=self.checkpoint_never_yield):
            if qd.static(True):
                self.fem.forward_global_vertices(self.global_vertex_manager)
                self.global_vertex_manager.reset_trajectory()
                if qd.static(self.genesis_serial_pipeline):
                    self.contact.bvh_triangle_build()
                    self.contact.bvh_edge_build()
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            self.contact.bvh_triangle_build()
                        with qd.graph.parallel():
                            self.contact.bvh_edge_build()
        if qd.static(self.has_et_check):
            with qd.checkpoint(
                ContactCheckpoint.INITIAL_INTERSECTION,
                yield_on=et_overflow,
            ):
                if qd.static(True):
                    self.contact.reset_initial_intersections()
                    self.contact.detect_initial_intersections()
        with qd.checkpoint(ContactCheckpoint.QUERY, yield_on=pair_overflow):
            if qd.static(True):
                self.contact.reset_collision_counts()
                if qd.static(self.genesis_serial_pipeline):
                    self.contact.trajectory_query()
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            self.contact.broad_phase.pt_query()
                        with qd.graph.parallel():
                            self.contact.broad_phase.ee_query()
                        with qd.graph.parallel():
                            if qd.static(self.contact.has_halfplanes):
                                self.contact.halfplane_query()
        with qd.checkpoint(ContactCheckpoint.COUNT, yield_on=self.checkpoint_never_yield):
            if qd.static(True):
                self.contact.reset_counted_demand()
                if qd.static(self.genesis_serial_pipeline):
                    self.contact.contact_constitution.count_active(
                        self.contact,
                        self.contact.surface,
                        self.contact.vertex,
                    )
                else:
                    with qd.graph.parallel_context():
                        with qd.graph.parallel():
                            self.contact.contact_constitution.count_active_pt(
                                self.contact,
                                self.contact.surface,
                                self.contact.vertex,
                            )
                        with qd.graph.parallel():
                            self.contact.contact_constitution.count_active_ee(
                                self.contact,
                                self.contact.surface,
                                self.contact.vertex,
                            )
                        with qd.graph.parallel():
                            if qd.static(self.contact.has_halfplanes):
                                self.contact.contact_constitution.count_active_ph(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                self.contact.check_assembly_capacity()

    def _handle_pair_overflow(self) -> None:
        self.contact.handle_broad_phase_overflow()
        self.contact.realloc_pair_buffers(
            pt=int(qd_to_numpy(self.contact.n_pairs_pt)),
            ee=int(qd_to_numpy(self.contact.n_pairs_ee)),
            pe=int(qd_to_numpy(self.contact.n_pairs_pe)),
            pp=int(qd_to_numpy(self.contact.n_pairs_pp)),
            ph=int(qd_to_numpy(self.contact.n_pairs_ph)),
        )

    def _initialize_contact(self) -> None:
        pair_overflow = self.contact.overflow_flag
        et_overflow = self.contact.et_overflow_flag
        status = self._init_contact_kernel(pair_overflow, et_overflow, self.graph_fastcache_key)
        while status.yielded:
            if status.checkpoint == ContactCheckpoint.INITIAL_INTERSECTION:
                required = int(qd_to_numpy(self.contact.n_et_pairs))
                self.contact.realloc_et_pairs(required)
                resume_from = ContactCheckpoint.INITIAL_INTERSECTION
            elif status.checkpoint == ContactCheckpoint.QUERY:
                self._handle_pair_overflow()
                resume_from = ContactCheckpoint.QUERY
            else:
                raise RuntimeError(f"Unexpected contact init checkpoint {status.checkpoint}")
            status = self._init_contact_kernel.resume(
                pair_overflow,
                et_overflow,
                self.graph_fastcache_key,
                from_checkpoint=resume_from,
            )
        if int(qd_to_numpy(self.contact.n_et_pairs)) > 0:
            message = self._et_report_message("initial state")
            gs.logger.error(message)
            raise RuntimeError(message)

    def _et_report_message(self, stage: str) -> str:
        count = int(qd_to_numpy(self.contact.n_et_pairs))
        pairs = qd_to_numpy(self.contact.et_pairs)[:count]
        edges = qd_to_numpy(self.global_surface_manager.surf_edges)
        faces = qd_to_numpy(self.global_surface_manager.surf_triangles)
        positions = qd_to_numpy(self.global_vertex_manager.positions)
        body_ids = qd_to_numpy(self.global_vertex_manager.body_id)
        geometry_ids = qd_to_numpy(self.global_vertex_manager.geometry_id)
        geometry_sources = qd_to_numpy(self.global_vertex_manager.geometry_source)
        source_geometry_ids = qd_to_numpy(self.global_vertex_manager.source_geometry_id)
        geometry_environments = qd_to_numpy(self.global_vertex_manager.geometry_environment)
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
            f"pt_pairs={int(qd_to_numpy(self.contact.n_pairs_pt))}, "
            f"ee_pairs={int(qd_to_numpy(self.contact.n_pairs_ee))}, "
            f"active_pairs={int(qd_to_numpy(self.contact.n_active_pairs))}, "
            f"ccd_alpha={float(qd_to_numpy(self.contact.ccd_alpha)):.9g}, "
            "frame_ccd_alpha="
            f"{float(qd_to_numpy(self.contact.frame_ccd_alpha)):.9g}"
        )
        if self.rigid_contact_assemble is not None:
            contact_state += f", proxy_doublets={int(qd_to_numpy(self.rigid_contact_assemble.rigid_doublet_total))}"
        broad_phase = self.contact.broad_phase
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

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def _step_kernel(
        self,
        pair_overflow: qd.types.ndarray(qd.i32, ndim=0),
        assembly_overflow: qd.types.ndarray(qd.i32, ndim=0),
        padding_overflow: qd.types.ndarray(qd.i32, ndim=0),
        triplet_overflow: qd.types.ndarray(qd.i32, ndim=0),
        friction_overflow: qd.types.ndarray(qd.i32, ndim=0),
        et_overflow: qd.types.ndarray(qd.i32, ndim=0),
        graph_fastcache_key: qd.template(),
    ):
        with qd.checkpoint(ContactCheckpoint.FRAME, yield_on=self.checkpoint_never_yield):
            if qd.static(self.has_contact):
                self.contact.adaptive_kappa_update()
                self.contact.reset_frame_ccd()
            if qd.static(self.has_rigid):
                self.rigid.predict()
                self.rigid.assemble_candidate_rows()
            if qd.static(self.has_rigid_contact_proxy):
                self.rigid_contact_proxy.mark_mechanism_constrained()
            if qd.static(self.has_rigid):
                self.rigid.initialize_newton()
            if qd.static(self.has_rigid_contact_proxy):
                self.rigid_contact_proxy.reset_frame()
                self.rigid_contact_proxy.prepare_metric()
                self.rigid_contact_proxy.prepare_tolerance(
                    self.sim_config,
                    self.contact,
                )
            if qd.static(self.has_fem):
                self.fem.predict(self.sim_config)
                self.fem.forward_global_vertices(self.global_vertex_manager)
            for _ in range(1):
                self.newton_iter[()] = 0
                self.max_pcg_iters[()] = 0
                self.total_pcg_iters[()] = 0
                self.ls_iter[()] = 0
                self.frame_failed[()] = 0
                self.newton_cond[()] = 1

        with qd.checkpoint(ContactCheckpoint.FRICTION, yield_on=friction_overflow):
            if qd.static(self.has_contact):  # noqa: SIM102
                if qd.static(self.contact.has_friction):
                    if qd.static(self.genesis_serial_pipeline):
                        self.contact.friction_snapshot()
                    else:
                        self.contact.contact_constitution.snapshot_lagged_positions(
                            self.contact,
                            self.contact.vertex,
                        )
                        with qd.graph.parallel_context():
                            with qd.graph.parallel():
                                self.contact.contact_constitution.friction_pair_filter_pt(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                            with qd.graph.parallel():
                                self.contact.contact_constitution.friction_pair_filter_ee(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_halfplanes):
                                    self.contact.contact_constitution.friction_pair_filter_ph(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )

        while qd.graph.do_while(self.newton_cond):
            with qd.checkpoint(ContactCheckpoint.COUNT, yield_on=assembly_overflow):
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.initialize_newton()
                if qd.static(self.has_rigid_forest):
                    self.rigid_forest.compute_endpoint_fk()
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.prepare_constraint()
                if qd.static(self.has_contact):
                    self.contact.reset_counted_demand()
                    if qd.static(self.genesis_serial_pipeline):
                        self.contact.contact_constitution.count_active(
                            self.contact,
                            self.contact.surface,
                            self.contact.vertex,
                        )
                    else:
                        with qd.graph.parallel_context():
                            with qd.graph.parallel():
                                self.contact.contact_constitution.count_active_pt(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                            with qd.graph.parallel():
                                self.contact.contact_constitution.count_active_ee(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_halfplanes):
                                    self.contact.contact_constitution.count_active_ph(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                    self.contact.check_assembly_capacity()

            with qd.checkpoint(ContactCheckpoint.FILTER, yield_on=padding_overflow):
                for _ in range(1):
                    self.newton_iter[()] = self.newton_iter[()] + 1
                if qd.static(self.has_contact):
                    self.contact.adaptive_kappa_newton_tick()
                    self.contact.vertex.zero_in_contact()
                    self.contact.reset_assembly_counts()
                    if qd.static(self.genesis_serial_pipeline):
                        self.contact.contact_constitution.filter_assemble(
                            self.contact,
                            self.contact.surface,
                            self.contact.vertex,
                        )
                    else:
                        with qd.graph.parallel_context():
                            with qd.graph.parallel():
                                self.contact.contact_constitution.filter_assemble_pt(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                            with qd.graph.parallel():
                                self.contact.contact_constitution.filter_assemble_ee(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_halfplanes):
                                    self.contact.contact_constitution.filter_assemble_ph(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_friction):
                                    self.contact.contact_constitution.friction_assemble_pt(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_friction):
                                    self.contact.contact_constitution.friction_assemble_ee(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_friction and self.contact.has_halfplanes):
                                    self.contact.contact_constitution.friction_assemble_ph(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                    self.contact.check_assembly_padding()

            with qd.checkpoint(ContactCheckpoint.SORT, yield_on=triplet_overflow):
                if qd.static(self.has_rigid_contact_assemble):
                    self.contact.sort_reduce()
                    self.fem.report_extent(self.global_linear_system)
                    self.rigid_contact_assemble.classify()
                    self.global_linear_system.derive_extents()
                else:
                    if qd.static(self.has_fem):
                        self.fem.report_extent(self.global_linear_system)
                    if qd.static(True):
                        self.global_linear_system.derive_extents()
                    if qd.static(self.has_contact):
                        self.contact.sort_reduce()
                        self.global_linear_system.compute_n_triplets(self.contact)

            with qd.checkpoint(ContactCheckpoint.SOLVE, yield_on=self.checkpoint_never_yield):
                if qd.static(True):
                    self.global_linear_system.zero_rhs()
                    self.global_linear_system.zero_triplet()

                if qd.static(self.has_rigid):
                    self.rigid.assemble(
                        self.sim_config,
                        self.global_linear_system,
                        self.has_rigid_forest,
                    )
                if qd.static(self.has_fem):
                    self.fem.assemble(self.sim_config, self.global_linear_system)
                if qd.static(self.has_rigid_contact_assemble):
                    self.rigid_contact_assemble.distribute()
                elif qd.static(self.has_contact):
                    distribute_fem_gradient_kernel(self.contact, self.fem, self.global_linear_system)
                    distribute_fem_fem_kernel(self.contact, self.fem, self.global_linear_system)

                if qd.static(self.has_fem):
                    self.global_linear_system.body_sort_reduce()
                    self.global_linear_system.traverse(self.fem_preconditioner, self.has_fem)
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.capture_physical_gradient(self.global_linear_system)
                if qd.static(self.has_rigid_forest):
                    self.rigid_forest.prepare_particular()
                    self.rigid_forest.particular_spmv(self.global_linear_system)
                    self.rigid_forest.project_physical_rhs(self.global_linear_system)
                    self.rigid_forest.build_preconditioner(self.global_linear_system)

            # SORT yields before an undersized triplet buffer can reach this solve. Keep the child WHILE outside the
            # explicit SOLVE checkpoint, and gate each inlined qd.func stage explicitly: checkpoint auto-wrapping does
            # not currently propagate across a top-level qd.func call that follows a yielding checkpoint
            # (Quadrants #956).
            with qd.checkpoint(ContactCheckpoint.PCG_INIT, yield_on=self.checkpoint_never_yield):
                if qd.static(True):
                    self.pcg_solver.linear_pcg.initialize(
                        self.global_linear_system,
                        self.rigid,
                        self.rigid_forest,
                        self.fem_preconditioner,
                        self.has_rigid,
                        self.has_rigid_forest,
                        self.has_fem,
                    )

            while qd.graph.do_while(self.pcg_solver.linear_pcg.condition):
                with qd.checkpoint(ContactCheckpoint.PCG_ITERATION, yield_on=self.checkpoint_never_yield):
                    if qd.static(True):
                        self.pcg_solver.linear_pcg.iteration(
                            self.global_linear_system,
                            self.rigid,
                            self.rigid_forest,
                            self.fem_preconditioner,
                            self.has_rigid,
                            self.has_rigid_forest,
                            self.has_fem,
                            self.pcg_solver.pcg_tol_rate[()],
                            self.sim_config.max_pcg_iter[()],
                        )

            with qd.checkpoint(ContactCheckpoint.PCG_FINISH, yield_on=self.checkpoint_never_yield):
                for _ in range(1):
                    self.max_pcg_iters[()] = qd.max(
                        self.max_pcg_iters[()],
                        self.pcg_solver.linear_pcg.n_iterations[()],
                    )
                    self.total_pcg_iters[()] = self.total_pcg_iters[()] + self.pcg_solver.linear_pcg.n_iterations[()]

            # Keep all post-PCG qd.func calls under a flat gate as well. Otherwise an earlier capacity yield can skip
            # direct kernel tasks but still enter one of these inlined functions.
            with qd.checkpoint(ContactCheckpoint.SOLVE_POST, yield_on=self.checkpoint_never_yield):
                if qd.static(self.has_rigid):
                    self.rigid.negate_dq(
                        self.global_linear_system,
                        self.has_rigid_forest,
                    )
                if qd.static(self.has_fem):
                    self.fem.negate_dx(self.global_linear_system)
                if qd.static(self.has_rigid_forest):
                    self.rigid_forest.expand_solution(self.global_linear_system.x_sol)
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.prepare_path_limit(self.contact)

                for _ in range(1):
                    self.max_disp[()] = qd.f64(0.0)
                if qd.static(self.has_fem):
                    self.fem.contribute_newton_max_disp(self.max_disp)
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.contribute_newton_max_disp(
                        self.global_vertex_manager,
                        self.max_disp,
                    )

                for _ in range(1):
                    rigid_converged = True
                    fem_converged = True
                    if qd.static(self.has_rigid_contact_proxy):
                        rigid_converged = self.max_disp[()] <= self.sim_config.tol[()] * self.sim_config.dt[()]
                    elif qd.static(self.has_rigid):
                        rigid_converged = self.rigid.gradient_squared[()] <= self.sim_config.tol[()] ** 2
                    if qd.static(self.has_fem and not self.has_rigid_contact_proxy):
                        fem_converged = self.max_disp[()] <= self.sim_config.tol[()] * self.sim_config.dt[()]
                    self.converged[()] = qd.i32(rigid_converged and fem_converged)
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.apply_convergence(self.converged)

                if qd.static(self.has_rigid):
                    self.rigid.record_start_point()
                    self.rigid.energy(self.sim_config)
                if qd.static(self.has_fem):
                    self.fem.record_start_point()
                    self.fem.energy(self.sim_config)
                    self.global_vertex_manager.record_safe_positions()
                    self.fem.publish_trajectory_end_positions(self.global_vertex_manager)
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.record_start_point()
                    self.rigid_contact_proxy.forward_global_vertices(self.global_vertex_manager)
                    self.rigid_contact_proxy.publish_trajectory_end_positions(self.global_vertex_manager)
                if qd.static(self.has_contact):
                    if qd.static(self.genesis_serial_pipeline):
                        self.contact.contact_energy()
                    else:
                        self.contact.reset_contact_energy()
                        with qd.graph.parallel_context():
                            with qd.graph.parallel():
                                self.contact.contact_constitution.filter_energy_pt(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                            with qd.graph.parallel():
                                self.contact.contact_constitution.filter_energy_ee(
                                    self.contact,
                                    self.contact.surface,
                                    self.contact.vertex,
                                )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_halfplanes):
                                    self.contact.contact_constitution.filter_energy_ph(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_friction):
                                    self.contact.contact_constitution.friction_energy_pt(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_friction):
                                    self.contact.contact_constitution.friction_energy_ee(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_friction and self.contact.has_halfplanes):
                                    self.contact.contact_constitution.friction_energy_ph(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                        self.contact.sum_contact_energy()
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.compute_restoration_energy(False)

                for _ in range(1):
                    baseline = qd.f64(0.0)
                    if qd.static(self.has_rigid):
                        baseline = baseline + self.rigid.rigid_energy[()]
                    if qd.static(self.has_fem):
                        baseline = baseline + self.fem.fem_energy[()]
                    if qd.static(self.has_contact):
                        baseline = baseline + self.contact.contact_energy_value[()]
                    if qd.static(self.has_rigid_contact_proxy):
                        baseline = baseline + self.rigid_contact_proxy.restoration_energy[()]
                    self.energy_buf[0] = baseline
                if qd.static(self.has_rigid_forest):
                    self.rigid_forest.compute_merit_directional_derivative(self.global_linear_system)
                    self.rigid_contact_proxy.initialize_merit()

                if qd.static(self.has_contact):
                    if qd.static(self.genesis_serial_pipeline):
                        self.contact.bvh_triangle_build()
                        self.contact.bvh_edge_build()
                    else:
                        with qd.graph.parallel_context():
                            with qd.graph.parallel():
                                self.contact.bvh_triangle_build()
                            with qd.graph.parallel():
                                self.contact.bvh_edge_build()

            with qd.checkpoint(ContactCheckpoint.QUERY, yield_on=pair_overflow):
                if qd.static(self.has_contact):
                    self.contact.reset_collision_counts()
                    if qd.static(self.genesis_serial_pipeline):
                        self.contact.trajectory_query()
                    else:
                        with qd.graph.parallel_context():
                            with qd.graph.parallel():
                                self.contact.broad_phase.pt_query()
                            with qd.graph.parallel():
                                self.contact.broad_phase.ee_query()
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_halfplanes):
                                    self.contact.halfplane_query()

            with qd.checkpoint(ContactCheckpoint.CCD, yield_on=self.checkpoint_never_yield):
                if qd.static(self.has_contact):
                    self.contact.init_ccd()
                    if qd.static(self.genesis_serial_pipeline):
                        self.contact.ccd()
                    else:
                        with qd.graph.parallel_context():
                            with qd.graph.parallel():
                                self.contact.ccd_alpha_pt_kernel()
                            with qd.graph.parallel():
                                self.contact.ccd_alpha_ee_kernel()
                            with qd.graph.parallel():
                                if qd.static(self.contact.has_halfplanes):
                                    self.contact.halfplane_ccd_alpha_kernel()
                        self.contact.reduce_ccd_alpha_final_kernel()

            with qd.checkpoint(ContactCheckpoint.LINE_SEARCH, yield_on=self.checkpoint_never_yield):
                for _ in range(1):
                    initial_alpha = qd.f64(1.0)
                    if qd.static(self.has_contact):
                        initial_alpha = qd.min(initial_alpha, self.contact.ccd_alpha[()])
                    if qd.static(self.has_rigid_contact_proxy):
                        initial_alpha = qd.min(
                            initial_alpha,
                            self.rigid_contact_proxy.fk_alpha[()],
                        )
                    self.alpha[()] = initial_alpha
                    self.ls_cond[()] = 1
                    self.ls_iter[()] = 0

            # No line-search operation can yield to the host. Its child WHILE stays outside the explicit phase marker,
            # while each inlined trial stage gets a flat gate for the same qd.func suffix workaround as PCG.
            while qd.graph.do_while(self.ls_cond):
                with qd.checkpoint(ContactCheckpoint.LINE_SEARCH_TRIAL, yield_on=self.checkpoint_never_yield):
                    for _ in range(1):
                        if self.pcg_solver.linear_pcg.is_failed[()] != 0:
                            self.alpha[()] = qd.f64(0.0)

                    if qd.static(self.has_rigid):
                        self.rigid.step_forward(self.alpha[()])
                        self.rigid.energy(self.sim_config)
                    if qd.static(self.has_rigid_forest):
                        self.rigid_forest.compute_endpoint_fk()
                    if qd.static(self.has_fem):
                        self.fem.step_forward(self.alpha[()])
                        self.fem.forward_global_vertices(self.global_vertex_manager)
                        self.fem.energy(self.sim_config)
                    if qd.static(self.has_rigid_contact_proxy):
                        self.rigid_contact_proxy.step_forward(self.alpha[()])
                        self.rigid_contact_proxy.forward_global_vertices(self.global_vertex_manager)
                        self.rigid_contact_proxy.evaluate_trial_guard()
                    if qd.static(self.has_contact):
                        if qd.static(self.genesis_serial_pipeline):
                            self.contact.contact_energy()
                        else:
                            self.contact.reset_contact_energy()
                            with qd.graph.parallel_context():
                                with qd.graph.parallel():
                                    self.contact.contact_constitution.filter_energy_pt(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                                with qd.graph.parallel():
                                    self.contact.contact_constitution.filter_energy_ee(
                                        self.contact,
                                        self.contact.surface,
                                        self.contact.vertex,
                                    )
                                with qd.graph.parallel():
                                    if qd.static(self.contact.has_halfplanes):
                                        self.contact.contact_constitution.filter_energy_ph(
                                            self.contact,
                                            self.contact.surface,
                                            self.contact.vertex,
                                        )
                                with qd.graph.parallel():
                                    if qd.static(self.contact.has_friction):
                                        self.contact.contact_constitution.friction_energy_pt(
                                            self.contact,
                                            self.contact.surface,
                                            self.contact.vertex,
                                        )
                                with qd.graph.parallel():
                                    if qd.static(self.contact.has_friction):
                                        self.contact.contact_constitution.friction_energy_ee(
                                            self.contact,
                                            self.contact.surface,
                                            self.contact.vertex,
                                        )
                                with qd.graph.parallel():
                                    if qd.static(self.contact.has_friction and self.contact.has_halfplanes):
                                        self.contact.contact_constitution.friction_energy_ph(
                                            self.contact,
                                            self.contact.surface,
                                            self.contact.vertex,
                                        )
                            self.contact.sum_contact_energy()
                    if qd.static(self.has_rigid_contact_proxy):
                        self.rigid_contact_proxy.compute_restoration_energy(True)

                    for _ in range(1):
                        trial_energy = qd.f64(0.0)
                        if qd.static(self.has_rigid):
                            trial_energy = trial_energy + self.rigid.rigid_energy[()]
                        if qd.static(self.has_fem):
                            trial_energy = trial_energy + self.fem.fem_energy[()]
                        if qd.static(self.has_contact):
                            trial_energy = trial_energy + self.contact.contact_energy_value[()]
                        if qd.static(self.has_rigid_contact_proxy):
                            trial_energy = trial_energy + self.rigid_contact_proxy.restoration_energy[()]
                            if (
                                self.rigid_contact_proxy.restoration_active[()] == 0
                                and self.rigid_contact_proxy.merit_evaluated[()] == 0
                            ):
                                trial_energy = trial_energy + self.rigid_contact_proxy.test_merit_energy_bias[()]
                            if self.rigid_contact_proxy.restoration_triggered[()] == 0:
                                trial_energy = trial_energy + self.rigid_contact_proxy.ls_forensics_test_energy_bias[()]
                        self.energy_delta[()] = trial_energy - self.energy_buf[0]

                    for _ in range(1):
                        guard_failed = False
                        if qd.static(self.has_rigid_contact_proxy):
                            guard_failed = self.rigid_contact_proxy.frame_failed[()] != 0
                        accepted = self.converged[()] != 0 or self.energy_delta[()] <= 0.0
                        if qd.static(self.has_rigid_contact_proxy):
                            if self.converged[()] == 0:
                                exhausted = self.ls_iter[()] + 1 >= self.sim_config.max_ls_iter[()]
                                accepted = self.rigid_contact_proxy.check_line_search(
                                    self.energy_buf[0],
                                    self.energy_buf[0] + self.energy_delta[()],
                                    self.alpha[()],
                                    self.ls_iter[()],
                                    self.sim_config.max_ls_iter[()],
                                    exhausted,
                                    self.converged,
                                )
                        else:
                            if not accepted and self.ls_iter[()] + 1 >= self.sim_config.max_ls_iter[()]:
                                accepted = True
                        accepted = accepted and not guard_failed
                        if guard_failed:
                            self.alpha[()] = qd.f64(0.0)
                            self.ls_cond[()] = 0
                        elif accepted:
                            self.ls_cond[()] = 0
                        else:
                            self.alpha[()] = self.alpha[()] * 0.5
                            self.ls_iter[()] = self.ls_iter[()] + 1
                            if self.ls_iter[()] >= self.sim_config.max_ls_iter[()]:
                                self.ls_cond[()] = 0

            # Post-acceptance qd.func calls need an explicit gate too: QUERY may have yielded before reaching them.
            with qd.checkpoint(ContactCheckpoint.LINE_SEARCH_POST, yield_on=self.checkpoint_never_yield):
                for _ in range(1):
                    if self.pcg_solver.linear_pcg.is_failed[()] != 0:
                        self.alpha[()] = qd.f64(0.0)
                if qd.static(self.has_fem):
                    self.fem.step_forward(self.alpha[()])
                    self.fem.forward_global_vertices(self.global_vertex_manager)
                if qd.static(self.has_rigid):
                    self.rigid.step_forward(self.alpha[()])
                if qd.static(self.has_rigid_forest):
                    self.rigid_forest.compute_endpoint_fk()
                if qd.static(self.has_rigid_contact_proxy):
                    self.rigid_contact_proxy.step_forward(self.alpha[()])
                    self.rigid_contact_proxy.forward_global_vertices(self.global_vertex_manager)
                    self.rigid_contact_proxy.finalize_restoration_step(self.alpha[()])

                for _ in range(1):
                    rejected = self.converged[()] == 0 and self.alpha[()] == 0.0
                    pcg_failed = self.pcg_solver.linear_pcg.is_failed[()] != 0
                    linear_failed = self.global_linear_system.triplet_overflow[()] != 0
                    if qd.static(self.has_fem):
                        linear_failed = linear_failed or self.global_linear_system.bcoo_valid[()] == 0
                    if qd.static(self.has_contact):
                        linear_failed = linear_failed or self.contact.intersection_flag[()] != 0
                    if qd.static(self.has_rigid_contact_proxy):
                        linear_failed = linear_failed or self.rigid_contact_proxy.frame_failed[()] != 0

                    exhausted = self.newton_iter[()] >= self.sim_config.max_newton_iter[()]
                    failed = rejected or pcg_failed or linear_failed or (exhausted and self.converged[()] == 0)
                    self.frame_failed[()] = qd.i32(failed)
                    converged = self.converged[()] != 0 and self.newton_iter[()] > 1
                    self.newton_cond[()] = qd.i32(not converged and not failed and not exhausted)

                if qd.static(self.has_rigid):
                    self.rigid.set_newton_active(self.newton_cond[()])
                    self.rigid.build_preconditioner(compute_envelope=False)

        if qd.static(self.has_et_check):
            with qd.checkpoint(
                ContactCheckpoint.ET_OVERFLOW,
                yield_on=et_overflow,
            ):
                if qd.static(True):
                    self.contact.reset_initial_intersections()
                    self.contact.detect_initial_intersections()
                    self.contact.flag_et_intersections()
            with qd.checkpoint(
                ContactCheckpoint.ET_FAILURE,
                yield_on=self.contact.et_yield_flag,
            ):
                for _ in range(1):
                    self.contact.et_yield_flag[()] = self.contact.et_yield_flag[()]

        if qd.static(self.has_rigid_contact_proxy):
            with qd.checkpoint(
                ContactCheckpoint.KKT_FAILURE,
                yield_on=self.frame_failed,
            ):
                for _ in range(1):
                    self.frame_failed[()] = self.frame_failed[()]

        with qd.checkpoint(ContactCheckpoint.FINALIZE, yield_on=self.checkpoint_never_yield):
            if qd.static(self.has_rigid):
                self.rigid.update_velocity()
            if qd.static(self.has_fem):
                self.fem.update_velocity(self.sim_config)
                self.fem.copy_x_prev()
            if qd.static(self.has_rigid_contact_proxy):
                self.rigid_contact_proxy.recover_reaction()
                self.rigid_contact_proxy.copy_previous_state()
            if qd.static(self.has_contact):
                self.contact.shrink_assembly_padding()

    def step(self) -> None:
        if not self.is_initialized_host:
            raise RuntimeError("SimEngine.init() must run before step()")
        if self.contact is None:
            pair_overflow = self.checkpoint_never_yield
            assembly_overflow = self.checkpoint_never_yield
            padding_overflow = self.checkpoint_never_yield
            friction_overflow = self.checkpoint_never_yield
        else:
            pair_overflow = self.contact.overflow_flag
            assembly_overflow = self.contact.count_overflow_flag
            padding_overflow = self.contact.contact_padding_overflow
            friction_overflow = self.contact.friction_overflow_flag
        et_overflow = self.checkpoint_never_yield if self.contact is None else self.contact.et_overflow_flag
        triplet_overflow = self.global_linear_system.triplet_overflow

        status = self._step_kernel(
            pair_overflow,
            assembly_overflow,
            padding_overflow,
            triplet_overflow,
            friction_overflow,
            et_overflow,
            self.graph_fastcache_key,
        )
        while status.yielded:
            checkpoint = status.checkpoint
            if self.contact is not None and checkpoint == ContactCheckpoint.FRICTION:
                required = {
                    channel: int(qd_to_numpy(getattr(self.contact, f"n_friction_pairs_{channel}")))
                    for channel in ("pt", "ee", "pe", "pp", "ph")
                }
                self.contact.realloc_friction_pair_buffers(required)
                friction_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = ContactCheckpoint.FRICTION
            elif self.contact is not None and checkpoint == ContactCheckpoint.COUNT:
                required_doublets = int(qd_to_numpy(self.contact.n_counted_doublets)) + int(
                    qd_to_numpy(self.contact.n_friction_demand_doublets)
                )
                required_triplets = int(qd_to_numpy(self.contact.n_counted_triplets)) + int(
                    qd_to_numpy(self.contact.n_friction_demand_triplets)
                )
                self.contact.realloc_assembly_buffers(required_doublets, required_triplets)
                if self.rigid_contact_assemble is not None:
                    self.rigid_contact_assemble.realloc_assembly_buffers()
                assembly_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = ContactCheckpoint.FILTER
            elif self.contact is not None and checkpoint == ContactCheckpoint.FILTER:
                n_doublets = int(qd_to_numpy(self.contact.n_contact_doublets))
                n_triplets = int(qd_to_numpy(self.contact.n_contact_triplets))
                grow_factor = CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"]
                padded_doublets = max(
                    int(qd_to_numpy(self.contact.padded_contact_doublets)),
                    min(
                        int(np.ceil(n_doublets * grow_factor)),
                        self.contact.contact_doublet_vertices.shape[0],
                    ),
                )
                padded_triplets = max(
                    int(qd_to_numpy(self.contact.padded_contact_triplets)),
                    min(
                        int(np.ceil(n_triplets * grow_factor)),
                        self.contact.contact_triplet_rows.shape[0],
                    ),
                )
                self.contact.set_assembly_padding(
                    padded_doublets,
                    padded_triplets,
                )
                padding_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = ContactCheckpoint.SORT
            elif checkpoint == ContactCheckpoint.SORT:
                required = int(qd_to_numpy(self.global_linear_system.n_triplets))
                grow_factor = CONTACT_CONFIG_DEFAULTS["extras/capacity_grow_factor"]
                self.global_linear_system.realloc_triplet_buffers(
                    max(
                        int(np.ceil(required * grow_factor)),
                        self.global_linear_system.triplet_row.shape[0] + 1,
                    ),
                    live_size=required,
                )
                triplet_overflow.from_numpy(np.array(0, dtype=np.int32))
                resume_from = ContactCheckpoint.SOLVE
            elif self.contact is not None and checkpoint == ContactCheckpoint.QUERY:
                self._handle_pair_overflow()
                resume_from = ContactCheckpoint.QUERY
            elif self.contact is not None and checkpoint == ContactCheckpoint.ET_OVERFLOW:
                required = int(qd_to_numpy(self.contact.n_et_pairs))
                self.contact.realloc_et_pairs(required)
                resume_from = ContactCheckpoint.ET_OVERFLOW
            elif self.contact is not None and checkpoint == ContactCheckpoint.ET_FAILURE:
                message = self._et_report_message("step")
                gs.logger.error(message)
                raise RuntimeError(message)
            elif checkpoint == ContactCheckpoint.KKT_FAILURE:
                proxy = self.rigid_contact_proxy
                rigid_dofs = self.rigid.dof_count_host
                rigid_rhs = qd_to_numpy(self.global_linear_system.b_rhs)[:rigid_dofs]
                rigid_solution = qd_to_numpy(self.global_linear_system.x_sol)[:rigid_dofs]
                rigid_preconditioned = qd_to_numpy(self.pcg_solver.linear_pcg.preconditioned_residual)[:rigid_dofs]
                rigid_search = qd_to_numpy(self.rigid.constraint_state.search).reshape(-1)[:rigid_dofs]
                edge_dofs = qd_to_numpy(self.rigid_forest.edge_dof_index)[: int(qd_to_numpy(self.rigid_forest.n_edges))]
                parent_edges = qd_to_numpy(self.rigid_forest.parent_edge)[
                    : int(qd_to_numpy(self.rigid_forest.n_mechanism_bodies))
                ]
                edge_pivots = qd_to_numpy(self.rigid_forest.edge_d)[: len(edge_dofs)]
                edge_rhs = qd_to_numpy(self.rigid_forest.precond_a)[: len(edge_dofs)]
                edge_children = qd_to_numpy(self.rigid_forest.edge_child)[: len(edge_dofs)]
                forest_depth = qd_to_numpy(self.rigid_forest.depth)
                details = (
                    f"newton={int(qd_to_numpy(self.newton_iter))}, "
                    f"pcg={int(qd_to_numpy(self.pcg_solver.linear_pcg.n_iterations))}, "
                    f"line_search={int(qd_to_numpy(self.ls_iter))}, "
                    f"alpha={float(qd_to_numpy(self.alpha)):.6g}, "
                    f"rigid_gradient_squared={float(qd_to_numpy(self.rigid.gradient_squared)):.6g}, "
                    f"rigid_rhs_norm={float(np.linalg.norm(rigid_rhs)):.6g}, "
                    f"rigid_solution_norm={float(np.linalg.norm(rigid_solution)):.6g}, "
                    f"rigid_preconditioned_norm={float(np.linalg.norm(rigid_preconditioned)):.6g}, "
                    f"rigid_search_norm={float(np.linalg.norm(rigid_search)):.6g}, "
                    f"edge_dofs={edge_dofs.tolist()}, "
                    f"edge_pivots={edge_pivots.tolist()}, "
                    f"edge_rhs={edge_rhs.tolist()}, "
                    f"edge_depths={forest_depth[edge_children].tolist()}, "
                    f"active_parent_edges={int(np.count_nonzero(parent_edges >= 0))}, "
                    f"forest_levels={int(qd_to_numpy(self.rigid_forest.n_levels))}, "
                    f"max_disp={float(qd_to_numpy(self.max_disp)):.6g}, "
                    f"residual={float(qd_to_numpy(proxy.max_surface_residual)):.6g}, "
                    f"tolerance={float(qd_to_numpy(proxy.solve_tolerance)):.6g}, "
                    f"fk_alpha={float(qd_to_numpy(proxy.fk_alpha)):.6g}, "
                    f"restoration_active={int(qd_to_numpy(proxy.restoration_active))}, "
                    f"hard_probe={int(qd_to_numpy(proxy.restoration_hard_probe))}, "
                    f"restoration_entries={int(qd_to_numpy(proxy.restoration_entries))}, "
                    f"restoration_epochs={int(qd_to_numpy(proxy.restoration_newton_epochs))}, "
                    f"hard_probes={int(qd_to_numpy(proxy.restoration_hard_probes))}, "
                    f"pcg_failed={int(qd_to_numpy(self.pcg_solver.linear_pcg.is_failed))}, "
                    f"proxy_failed={int(qd_to_numpy(proxy.frame_failed))}"
                )
                raise RuntimeError(
                    f"KKT rigid proxy solve exhausted the Newton budget before stationarity/feasibility ({details})"
                )
            else:
                raise RuntimeError(f"Unexpected timestep checkpoint {checkpoint}")

            status = self._step_kernel.resume(
                pair_overflow,
                assembly_overflow,
                padding_overflow,
                triplet_overflow,
                friction_overflow,
                et_overflow,
                self.graph_fastcache_key,
                from_checkpoint=resume_from,
            )
        if self.fem is not None:
            self.fem._forward_scene_vertices()
        if bool(qd_to_numpy(self.frame_failed)):
            details = [
                f"newton={self.get_newton_iters()}",
                f"pcg={self.get_max_pcg_iters()}",
                f"line_search={self.get_max_ls_iters()}",
                f"triplet_overflow={int(qd_to_numpy(self.global_linear_system.triplet_overflow))}",
                f"bcoo_valid={int(qd_to_numpy(self.global_linear_system.bcoo_valid))}",
            ]
            if self.contact is not None:
                details.extend(
                    (
                        f"intersection={int(qd_to_numpy(self.contact.intersection_flag))}",
                        f"pairs_pt={int(qd_to_numpy(self.contact.n_pairs_pt))}",
                        f"pairs_ee={int(qd_to_numpy(self.contact.n_pairs_ee))}",
                        f"pairs_ph={int(qd_to_numpy(self.contact.n_pairs_ph))}",
                        f"active_pairs={int(qd_to_numpy(self.contact.n_active_pairs))}",
                        f"ccd_alpha={float(qd_to_numpy(self.contact.ccd_alpha)):.6g}",
                    )
                )
                for channel in ("pt", "ee", "ph"):
                    count = int(qd_to_numpy(getattr(self.contact, f"n_pairs_{channel}")))
                    if count:
                        alphas = qd_to_numpy(getattr(self.contact, f"ccd_alpha_{channel}"))[:count]
                        minimum_index = int(np.argmin(alphas))
                        pair = qd_to_numpy(getattr(self.contact, f"pairs_{channel}"))[minimum_index]
                        details.append(
                            f"ccd_{channel}=({float(alphas[minimum_index]):.6g},"
                            f" pair={tuple(int(value) for value in pair)})"
                        )
                        if channel == "ph":
                            surface_vertex = int(pair[0])
                            vertex_id = int(qd_to_numpy(self.global_surface_manager.surf_verts)[surface_vertex])
                            current = qd_to_numpy(self.global_vertex_manager.positions)[vertex_id]
                            endpoint = qd_to_numpy(self.global_vertex_manager.trajectory_end_positions)[vertex_id]
                            thickness = float(qd_to_numpy(self.global_vertex_manager.thicknesses)[vertex_id])
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
