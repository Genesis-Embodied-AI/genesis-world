from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .consistent_ipc_contact import ConsistentIPCContactConstitution
from .contact import CONTACT_CONFIG_DEFAULTS, ContactTabular
from .contact_system import ContactSystem
from .finite_element import (
    FEMBDF1,
    FEMDiagPreconditioner,
    FiniteElement,
    FiniteElementMethod,
    QuadraticBending,
    StrainLimitBaraffWitkinShell2D,
)
from .global_body_manager import GlobalBodyManager
from .global_linear_system import GlobalLinearSystem
from .global_surface_manager import GlobalSurfaceManager
from .global_vertex_manager import GlobalVertexManager
from .lbvh_broad_phase import InfoLBVHBatchedBroadPhaseDop14, LBVHBroadPhase
from .rigid_contact_assemble import RigidContactAssemble
from .rigid_contact_proxy import RigidContactProxyGeometry, RigidContactProxySystem
from .rigid_joint_forest import RigidJointForestSystem
from .rigid_system import RigidSystem
from .sim_engine import SimEngine
from .standard_pcg_solver import StandardPCGSolver


def build_scene_engine(
    scene,
    *,
    contact_config: Mapping[str, object] | None = None,
    contact_tabular: ContactTabular | None = None,
    halfplanes: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimEngine:
    """Build the graph-native Rigid + QCloth CGQ system set."""
    finite_element = FiniteElement()
    has_fem = finite_element.init(scene)
    contact_requested = contact_config is not None
    resolved_contact_config = dict(CONTACT_CONFIG_DEFAULTS)
    if contact_config is not None:
        unknown = set(contact_config) - set(CONTACT_CONFIG_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown CGQ contact config keys: {sorted(unknown)}")
        resolved_contact_config.update(contact_config)
    enable_contact = contact_requested and bool(resolved_contact_config["contact/enable"])
    if enable_contact and not has_fem:
        raise RuntimeError("The current ContactSystem milestone requires FiniteElementMethod")
    rigid_proxy_geometry = None
    if enable_contact and scene.rigid_solver.is_active:
        staging = RigidContactProxyGeometry()
        if staging.init(scene, float(resolved_contact_config["contact/d_hat"])):
            rigid_proxy_geometry = staging

    genesis_legacy_sort_reduce = bool(int(resolved_contact_config["extras/sort_reduce/genesis_legacy"]))
    genesis_legacy_fp64_bounds = bool(int(resolved_contact_config["extras/bvh/genesis_legacy_fp64_bounds"]))
    genesis_legacy_refit = bool(int(resolved_contact_config["extras/bvh/genesis_legacy_refit"]))
    engine = SimEngine()
    engine.configure_genesis_serial_pipeline(bool(int(resolved_contact_config["extras/pipeline/genesis_serial"])))
    engine.add_system(
        GlobalLinearSystem(
            genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
        )
    )
    engine.add_system(StandardPCGSolver())
    if scene.rigid_solver.is_active:
        rigid = RigidSystem(scene.rigid_solver)
        engine.add_system(rigid)

    fem = None
    bdf1 = None
    membrane = None
    bending = None
    fem_preconditioner = None
    rigid_contact_proxy = None
    rigid_forest = None
    rigid_contact_assemble = None
    if has_fem:
        global_body_manager = GlobalBodyManager()
        global_vertex_manager = GlobalVertexManager()
        global_surface_manager = GlobalSurfaceManager()
        fem = FiniteElementMethod()
        bdf1 = FEMBDF1()
        membrane = StrainLimitBaraffWitkinShell2D()
        bending = QuadraticBending()
        fem_preconditioner = FEMDiagPreconditioner()
        for system in (
            global_body_manager,
            global_vertex_manager,
            global_surface_manager,
            fem,
            bdf1,
            membrane,
            bending,
            fem_preconditioner,
        ):
            engine.add_system(system)
        if enable_contact:
            contact_system = ContactSystem(
                intersection_check=bool(int(resolved_contact_config["contact/intersection_check"])),
                genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
            )
            bvh_type = str(resolved_contact_config["bvh/type"])
            if bvh_type == "info_lbvh_batched_dop14":
                broad_phase_system = InfoLBVHBatchedBroadPhaseDop14(
                    pt_query=str(resolved_contact_config["bvh/pt_query"]),
                    ee_query=str(resolved_contact_config["bvh/ee_query"]),
                    dual_frontier_levels=int(resolved_contact_config["bvh/dual/frontier_levels"]),
                    dual_target_waves=float(resolved_contact_config["bvh/dual/target_waves"]),
                    dual_max_levels=int(resolved_contact_config["bvh/dual/max_levels"]),
                    genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
                    genesis_legacy_fp64_bounds=genesis_legacy_fp64_bounds,
                    genesis_legacy_refit=genesis_legacy_refit,
                )
            elif bvh_type in ("lbvh", "info_lbvh", "info_lbvh_batched"):
                broad_phase_system = LBVHBroadPhase(
                    bound_type="aabb",
                    pt_query=str(resolved_contact_config["bvh/pt_query"]),
                    ee_query=str(resolved_contact_config["bvh/ee_query"]),
                    dual_frontier_levels=int(resolved_contact_config["bvh/dual/frontier_levels"]),
                    dual_target_waves=float(resolved_contact_config["bvh/dual/target_waves"]),
                    dual_max_levels=int(resolved_contact_config["bvh/dual/max_levels"]),
                    genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
                )
            else:
                raise NotImplementedError(f"Unsupported CGQ bvh/type {bvh_type!r}")
            contact_constitution = ConsistentIPCContactConstitution()
            for system in (contact_system, broad_phase_system, contact_constitution):
                engine.add_system(system)
            if rigid_proxy_geometry is not None:
                rigid_contact_proxy = RigidContactProxySystem()
                rigid_forest = RigidJointForestSystem(scene.rigid_solver)
                rigid_forest.configure(bool(int(resolved_contact_config["rigid_forest/fused"])))
                rigid_forest.configure_genesis_legacy(
                    bool(int(resolved_contact_config["extras/rigid_forest/genesis_legacy"]))
                )
                rigid_contact_assemble = RigidContactAssemble()
                for system in (
                    rigid_contact_proxy,
                    rigid_forest,
                    rigid_contact_assemble,
                ):
                    engine.add_system(system)

    engine.build_systems()

    if has_fem:
        fem.wire_data(finite_element)
        fem.receive_global_vertex_range(0, finite_element.n_verts)
        fem.receive_global_body_range(0, finite_element.n_bodies)

        proxy_vert_count = 0 if rigid_proxy_geometry is None else len(rigid_proxy_geometry.local_positions)
        total_vert_count = finite_element.n_verts + proxy_vert_count
        combined_thicknesses = finite_element.thicknesses
        combined_d_hats = np.full(
            finite_element.n_verts,
            resolved_contact_config["contact/d_hat"],
            dtype=np.float64,
        )
        combined_is_fixed = finite_element.is_fixed
        combined_geometry_ids = finite_element.geometry_ids
        combined_geometry_sources = np.zeros(
            finite_element.n_verts,
            dtype=np.int32,
        )
        combined_source_geometry_ids = finite_element.source_geometry_ids
        combined_geometry_environments = finite_element.geometry_environments
        if rigid_proxy_geometry is not None:
            rigid_contact_proxy.configure(
                str(resolved_contact_config["rigid_proxy/globalization"]),
                bool(int(resolved_contact_config["rigid_proxy/restoration"])),
                float(resolved_contact_config["rigid_proxy/test_merit_energy_bias"]),
                float(resolved_contact_config["extras/ls_forensics/test_energy_bias"]),
            )
            rigid_contact_proxy.wire_data(
                rigid_proxy_geometry.n_rigid_bodies,
                rigid_proxy_geometry.mechanism_body,
                rigid_proxy_geometry.proxy_body,
                rigid_proxy_geometry.surface_radius,
            )
            rigid_contact_proxy.wire_geometry(
                finite_element.n_verts,
                rigid_proxy_geometry,
                finite_element.n_bodies,
            )
            combined_thicknesses = np.concatenate((finite_element.thicknesses, rigid_proxy_geometry.thicknesses))
            combined_d_hats = np.concatenate((combined_d_hats, rigid_proxy_geometry.d_hats))
            combined_is_fixed = np.concatenate((finite_element.is_fixed, rigid_proxy_geometry.is_fixed))
            combined_geometry_ids = np.concatenate(
                (
                    finite_element.geometry_ids,
                    rigid_proxy_geometry.geometry_ids + finite_element.n_bodies,
                )
            )
            combined_geometry_sources = np.concatenate(
                (
                    combined_geometry_sources,
                    np.ones(proxy_vert_count, dtype=np.int32),
                )
            )
            combined_source_geometry_ids = np.concatenate(
                (
                    finite_element.source_geometry_ids,
                    rigid_proxy_geometry.source_geometry_ids,
                )
            )
            combined_geometry_environments = np.concatenate(
                (
                    finite_element.geometry_environments,
                    rigid_proxy_geometry.geometry_environments,
                )
            )

        global_vertex_manager.init(total_vert_count)
        global_vertex_manager.wire_thickness_data(combined_thicknesses)
        global_vertex_manager.wire_d_hat_data(combined_d_hats)
        global_vertex_manager.wire_is_fixed_data(combined_is_fixed)
        global_vertex_manager.wire_geometry_id_data(combined_geometry_ids)
        global_vertex_manager.wire_geometry_source_data(
            combined_geometry_sources,
            combined_source_geometry_ids,
            combined_geometry_environments,
        )

        total_body_count = finite_element.n_bodies
        body_vertex_offsets = finite_element.body_vertex_offsets
        body_self_collision = finite_element.self_collision
        ignorance = [set() for _ in range(total_body_count)]
        for body in range(finite_element.n_bodies):
            begin = finite_element.body_contact_ignorance_ranges[body]
            end = finite_element.body_contact_ignorance_ranges[body + 1]
            ignorance[body].update(int(target) for target in finite_element.body_contact_ignorance_body_ids[begin:end])

        if rigid_proxy_geometry is not None:
            total_body_count += rigid_proxy_geometry.n_rigid_bodies
            cursor = finite_element.n_verts
            offsets = list(np.asarray(finite_element.body_vertex_offsets, dtype=np.int32))
            offsets.extend([cursor] * rigid_proxy_geometry.n_mechanism_bodies)
            pair_counts = np.bincount(
                rigid_proxy_geometry.vertex_pair,
                minlength=rigid_proxy_geometry.n_pairs,
            )
            for count in pair_counts:
                cursor += int(count)
                offsets.append(cursor)
            body_vertex_offsets = np.asarray(offsets, dtype=np.int32)
            body_self_collision = np.concatenate(
                (
                    finite_element.self_collision,
                    np.ones(rigid_proxy_geometry.n_mechanism_bodies, dtype=np.int32),
                    np.zeros(rigid_proxy_geometry.n_pairs, dtype=np.int32),
                )
            )
            ignorance.extend(set() for _ in range(rigid_proxy_geometry.n_rigid_bodies))
            proxy_global_bodies = [finite_element.n_bodies + int(body) for body in rigid_proxy_geometry.proxy_body]
            for source in proxy_global_bodies:
                ignorance[source].update(target for target in proxy_global_bodies if target != source)

        ignorance_ranges = np.zeros(total_body_count + 1, dtype=np.int32)
        ignorance_ids = []
        for body, targets in enumerate(ignorance):
            ignorance_ids.extend(sorted(targets))
            ignorance_ranges[body + 1] = len(ignorance_ids)

        global_body_manager.init(total_body_count)
        global_body_manager.wire_body_layout(
            body_vertex_offsets,
            body_self_collision,
        )
        global_body_manager.wire_body_contact_ignorance(
            ignorance_ranges,
            np.asarray(ignorance_ids, dtype=np.int32),
        )

        surf_triangles = finite_element.surf_triangles
        surf_edges = finite_element.surf_edges
        surf_verts = finite_element.surf_verts
        vert_dimensions = finite_element.vert_dimensions
        vert_area_weights = finite_element.vert_area_weights
        edge_area_weights = finite_element.edge_area_weights
        face_area_weights = finite_element.face_area_weights
        if rigid_proxy_geometry is not None:
            surf_triangles = np.concatenate(
                (
                    surf_triangles,
                    rigid_proxy_geometry.surf_triangles + finite_element.n_verts,
                )
            )
            surf_edges = np.concatenate(
                (
                    surf_edges,
                    rigid_proxy_geometry.surf_edges + finite_element.n_verts,
                )
            )
            surf_verts = np.concatenate(
                (
                    surf_verts,
                    rigid_proxy_geometry.surf_verts + finite_element.n_verts,
                )
            )
            vert_dimensions = np.concatenate((vert_dimensions, rigid_proxy_geometry.vert_dimensions))
            vert_area_weights = np.concatenate((vert_area_weights, rigid_proxy_geometry.vert_area_weights))
            edge_area_weights = np.concatenate((edge_area_weights, rigid_proxy_geometry.edge_area_weights))
            face_area_weights = np.concatenate((face_area_weights, rigid_proxy_geometry.face_area_weights))

        global_surface_manager.wire_surface_data(
            surf_triangles,
            surf_edges,
            surf_verts,
        )
        global_surface_manager.wire_vert_dimensions(vert_dimensions)
        global_surface_manager.wire_area_weights(
            vert_area_weights,
            edge_area_weights,
            face_area_weights,
        )
        bdf1.wire_data(finite_element.n_verts)
        membrane.wire_data(
            tri_indices=np.arange(finite_element.n_tris, dtype=np.int32),
            mu=finite_element.membrane_mu,
            lambda_param=finite_element.membrane_lambda,
            strain_limit_multiplier=finite_element.strain_limit_multiplier,
        )
        bending.wire_data(
            hinge_indices=finite_element.hinge_indices,
            bending_stiffness=finite_element.hinge_stiffness,
            Q0=finite_element.hinge_Q0,
            vert_bend_k=finite_element.vert_bend_k,
        )

        if enable_contact:
            table = contact_tabular if contact_tabular is not None else ContactTabular()
            default_model = table.at(0, 0)
            constitution = resolved_contact_config["contact/constitution"]
            if constitution == "auto":
                constitution = "consistent_ipc"
            if constitution != "consistent_ipc":
                raise NotImplementedError(
                    f"The cloth contact milestone implements only consistent_ipc, got {constitution!r}"
                )
            contact_system.wire_params(
                d_hat=float(resolved_contact_config["contact/d_hat"]),
                kappa=default_model.resistance,
                init_pair_capacity=int(resolved_contact_config["contact/init_collision_pair_capacity"]),
                intersection_check=bool(int(resolved_contact_config["contact/intersection_check"])),
                intersection_check_capacity=int(resolved_contact_config["contact/intersection_check_capacity"]),
            )
            contact_system.set_dt_sq(scene.sim.substep_dt * scene.sim.substep_dt)
            contact_system.wire_friction_params(
                mu=default_model.friction_rate,
                eps_v=float(resolved_contact_config["friction/eps_v"]),
            )
            contact_system.wire_contact_tabular(table)
            if halfplanes is None:
                halfplane_positions = np.empty((0, 3), dtype=np.float64)
                halfplane_normals = np.empty((0, 3), dtype=np.float64)
            else:
                halfplane_positions, halfplane_normals = halfplanes
            contact_system.wire_halfplanes(halfplane_positions, halfplane_normals)
            contact_system.set_adaptive_kappa(
                str(resolved_contact_config["contact/adaptive_kappa_mode"]),
                str(resolved_contact_config["contact/adaptive_kappa_tick"]),
                total_body_count,
            )
            contact_system.init(total_vert_count)
            broad_phase_system.init_bvh(
                len(surf_triangles),
                len(surf_edges),
                0,
            )

    engine.wire_solver_params(
        dt=scene.sim.substep_dt,
        tol=5e-2 if has_fem else scene.rigid_solver._options.tolerance,
        max_newton_iter=1024 if has_fem else scene.rigid_solver._options.iterations,
        max_pcg_iter=1024 if has_fem else scene.rigid_solver._options.iterations,
        max_ls_iter=12 if has_fem else scene.rigid_solver._options.ls_iterations,
        pcg_tol_rate=(
            float(resolved_contact_config["linear_system/tol_rate"])
            if has_fem
            else scene.rigid_solver._options.tolerance
        ),
    )
    engine.init()
    return engine
