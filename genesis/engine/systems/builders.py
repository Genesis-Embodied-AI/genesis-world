from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .consistent_ipc_contact import ConsistentIPCContactConstitution
from .contact import CONTACT_CONFIG_DEFAULTS, ContactTabular
from .contact_system import ContactSystem, get_contact_assembly_capacity, get_contact_system_data
from .finite_element import (
    FEMBDF1,
    FEMDiagPreconditioner,
    FiniteElement,
    FiniteElementMethod,
    QuadraticBending,
    StrainLimitBaraffWitkinShell2D,
)
from .finite_element.fem_bdf1 import get_fem_bdf1_data
from .finite_element.fem_diag_preconditioner import get_fem_diag_preconditioner_data
from .finite_element.finite_element_method import get_finite_element_method_data
from .finite_element.quadratic_bending import get_quadratic_bending_data
from .finite_element.strain_limit_baraff_witkin_shell_2d import (
    get_strain_limit_baraff_witkin_shell_2d_data,
)
from .global_body_manager import GlobalBodyManager, get_global_body_data
from .global_linear_system import GlobalLinearSystem, get_global_linear_system_data
from .global_surface_manager import GlobalSurfaceManager, get_global_surface_data
from .global_vertex_manager import GlobalVertexManager, get_global_vertex_data
from .lbvh_broad_phase import (
    InfoLBVHBatchedBroadPhaseDop14,
    LBVHBroadPhase,
    get_info_lbvh_batched_broad_phase_dop14_data,
    get_lbvh_broad_phase_data,
)
from .linear_pcg import LinearPCG, get_linear_pcg_data
from .pcg_solver import PCGSolver
from .rigid_contact_assemble import RigidContactAssemble, get_rigid_contact_assemble_data
from .rigid_contact_proxy import (
    RigidContactProxyGeometry,
    RigidContactProxySystem,
    get_rigid_contact_proxy_data,
)
from .rigid_joint_forest import RigidJointForestSystem, get_rigid_joint_forest_data
from .rigid_system import RigidSystem, get_rigid_system_data
from .sim_engine import SimEngine


def build_scene_engine(
    scene,
    *,
    contact_config: Mapping[str, object] | None = None,
    contact_tabular: ContactTabular | None = None,
    halfplanes: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimEngine:
    """Build the graph-native Rigid + QCloth system set."""
    finite_element = FiniteElement()
    has_fem = finite_element.init(scene)
    contact_requested = contact_config is not None
    resolved_contact_config = dict(CONTACT_CONFIG_DEFAULTS)
    if contact_config is not None:
        unknown = set(contact_config) - set(CONTACT_CONFIG_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown contact config keys: {sorted(unknown)}")
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
    rigid = RigidSystem(get_rigid_system_data(scene.rigid_solver)) if scene.rigid_solver.is_active else None
    rigid_dof_count = 0 if rigid is None else rigid.storage_dof_count_host
    fem_vert_capacity = max(finite_element.n_verts, 1) if has_fem else 0
    fem_dof_count = fem_vert_capacity * 3
    proxy_pair_count = 0 if rigid_proxy_geometry is None else rigid_proxy_geometry.n_pairs
    proxy_dof_count = proxy_pair_count * 6
    total_dof = rigid_dof_count + fem_dof_count + proxy_dof_count
    n_block_rows = total_dof // 3
    n_elastic_triplets = (
        finite_element.n_verts + finite_element.n_tris * 6 + len(finite_element.hinge_indices) * 10 if has_fem else 0
    )
    extent_capacity = (3 if has_fem else 0) + (1 if rigid_proxy_geometry is not None else 0)
    max_contact_body_triplets = 0
    if enable_contact:
        assembly_capacity = get_contact_assembly_capacity()
        max_contact_body_triplets = assembly_capacity
        if rigid_proxy_geometry is not None:
            max_contact_body_triplets = assembly_capacity * 5

    engine = SimEngine()
    engine.configure_genesis_serial_pipeline(bool(int(resolved_contact_config["extras/pipeline/genesis_serial"])))
    global_linear_system = GlobalLinearSystem(
        data=get_global_linear_system_data(
            n_block_rows=n_block_rows,
            n_elastic_triplets=n_elastic_triplets,
            max_contact_body_triplets=max_contact_body_triplets,
            dof_block_base=rigid_dof_count // 3,
            extent_capacity=extent_capacity,
            genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
        )
    )
    linear_pcg = LinearPCG(get_linear_pcg_data(total_dof))
    engine.add_system(global_linear_system)
    engine.add_system(PCGSolver())
    engine.add_system(linear_pcg)
    if rigid is not None:
        engine.add_system(rigid)

    fem = None
    bdf1 = None
    membrane = None
    bending = None
    fem_preconditioner = None
    contact_system = None
    broad_phase_system = None
    contact_constitution = None
    rigid_contact_proxy = None
    rigid_forest = None
    rigid_contact_assemble = None
    if has_fem:
        if enable_contact:
            if rigid_proxy_geometry is not None:
                proxy_data = get_rigid_contact_proxy_data(
                    n_links_host=scene.rigid_solver.n_links,
                    n_instances_host=scene.rigid_solver._B,
                    n_rigid_bodies=rigid_proxy_geometry.n_rigid_bodies,
                    mechanism_body=rigid_proxy_geometry.mechanism_body,
                    proxy_body=rigid_proxy_geometry.proxy_body,
                    surface_radius=rigid_proxy_geometry.surface_radius,
                    geometry=rigid_proxy_geometry,
                    global_vert_offset=finite_element.n_verts,
                    global_body_offset=finite_element.n_bodies,
                    merit_gradient_capacity=total_dof,
                    globalization=str(resolved_contact_config["rigid_proxy/globalization"]),
                    restoration=bool(int(resolved_contact_config["rigid_proxy/restoration"])),
                    test_merit_energy_bias=float(resolved_contact_config["rigid_proxy/test_merit_energy_bias"]),
                    ls_forensics_test_energy_bias=float(
                        resolved_contact_config["extras/ls_forensics/test_energy_bias"]
                    ),
                )
                rigid_contact_proxy = RigidContactProxySystem(proxy_data)
                forest_data = get_rigid_joint_forest_data(
                    scene.rigid_solver,
                    rigid.data,
                    total_dof=total_dof,
                    n_rigid_bodies=rigid_proxy_geometry.n_rigid_bodies,
                    proxy_dof_offset=rigid_dof_count + fem_dof_count,
                    proxy_data=proxy_data,
                    fused_enabled=bool(int(resolved_contact_config["rigid_forest/fused"])),
                    genesis_legacy_enabled=bool(int(resolved_contact_config["extras/rigid_forest/genesis_legacy"])),
                )
                rigid_forest = RigidJointForestSystem(forest_data)

    if has_fem:
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

        global_vertex_data = get_global_vertex_data(
            total_vert_count,
            thicknesses=combined_thicknesses,
            d_hats=combined_d_hats,
            is_fixed=combined_is_fixed,
            geometry_ids=combined_geometry_ids,
            geometry_sources=combined_geometry_sources,
            source_geometry_ids=combined_source_geometry_ids,
            geometry_environments=combined_geometry_environments,
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

        global_body_data = get_global_body_data(
            total_body_count,
            vertex_offsets=body_vertex_offsets,
            self_collision=body_self_collision,
            body_contact_ignorance_ranges=ignorance_ranges,
            body_contact_ignorance_body_ids=np.asarray(ignorance_ids, dtype=np.int32),
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

        global_surface_data = get_global_surface_data(
            surf_triangles,
            surf_edges,
            surf_verts,
            vert_dimensions=vert_dimensions,
            surf_vert_area_weights=vert_area_weights,
            surf_edge_area_weights=edge_area_weights,
            surf_face_area_weights=face_area_weights,
        )
        fem = FiniteElementMethod(get_finite_element_method_data(finite_element))
        fem.receive_global_vertex_range(0, finite_element.n_verts)
        fem.receive_global_body_range(0, finite_element.n_bodies)
        bdf1 = FEMBDF1(get_fem_bdf1_data(finite_element.n_verts))
        membrane = StrainLimitBaraffWitkinShell2D(
            get_strain_limit_baraff_witkin_shell_2d_data(
                tri_indices=np.arange(finite_element.n_tris, dtype=np.int32),
                mu=finite_element.membrane_mu,
                lambda_param=finite_element.membrane_lambda,
                strain_limit_multiplier=finite_element.strain_limit_multiplier,
            )
        )
        bending = QuadraticBending(
            get_quadratic_bending_data(
                hinge_indices=finite_element.hinge_indices,
                bending_stiffness=finite_element.hinge_stiffness,
                Q0=finite_element.hinge_Q0,
                vert_bend_k=finite_element.vert_bend_k,
            )
        )
        fem_preconditioner = FEMDiagPreconditioner(
            get_fem_diag_preconditioner_data(
                vert_capacity=fem_vert_capacity,
                n_fem_verts=finite_element.n_verts,
                dof_offset=rigid_dof_count,
            )
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
            if halfplanes is None:
                halfplane_positions = np.empty((0, 3), dtype=np.float64)
                halfplane_normals = np.empty((0, 3), dtype=np.float64)
            else:
                halfplane_positions, halfplane_normals = halfplanes
            contact_system = ContactSystem(
                get_contact_system_data(
                    n_verts=total_vert_count,
                    n_bodies=total_body_count,
                    d_hat=float(resolved_contact_config["contact/d_hat"]),
                    kappa=default_model.resistance,
                    dt_sq=scene.sim.substep_dt * scene.sim.substep_dt,
                    init_pair_capacity=int(resolved_contact_config["contact/init_collision_pair_capacity"]),
                    contact_tabular=table,
                    friction_mu=default_model.friction_rate,
                    friction_eps_v=float(resolved_contact_config["friction/eps_v"]),
                    halfplane_positions=halfplane_positions,
                    halfplane_normals=halfplane_normals,
                    adaptive_kappa_mode=str(resolved_contact_config["contact/adaptive_kappa_mode"]),
                    adaptive_kappa_tick=str(resolved_contact_config["contact/adaptive_kappa_tick"]),
                    intersection_check=bool(int(resolved_contact_config["contact/intersection_check"])),
                    intersection_check_capacity=int(resolved_contact_config["contact/intersection_check_capacity"]),
                    genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
                )
            )
            broad_phase_kwargs = {
                "n_triangles": len(surf_triangles),
                "n_edges": len(surf_edges),
                "n_surface_vertices": len(surf_verts),
                "pt_query": str(resolved_contact_config["bvh/pt_query"]),
                "ee_query": str(resolved_contact_config["bvh/ee_query"]),
                "dual_frontier_levels": int(resolved_contact_config["bvh/dual/frontier_levels"]),
                "dual_target_waves": float(resolved_contact_config["bvh/dual/target_waves"]),
                "dual_max_levels": int(resolved_contact_config["bvh/dual/max_levels"]),
                "genesis_legacy_sort_reduce": genesis_legacy_sort_reduce,
                "genesis_legacy_fp64_bounds": genesis_legacy_fp64_bounds,
                "genesis_legacy_refit": genesis_legacy_refit,
            }
            bvh_type = str(resolved_contact_config["bvh/type"])
            if bvh_type == "info_lbvh_batched_dop14":
                broad_phase_system = InfoLBVHBatchedBroadPhaseDop14(
                    get_info_lbvh_batched_broad_phase_dop14_data(**broad_phase_kwargs)
                )
            elif bvh_type in ("lbvh", "info_lbvh", "info_lbvh_batched"):
                broad_phase_kwargs["genesis_legacy_fp64_bounds"] = False
                broad_phase_kwargs["genesis_legacy_refit"] = False
                broad_phase_system = LBVHBroadPhase(get_lbvh_broad_phase_data(bound_type="aabb", **broad_phase_kwargs))
            else:
                raise NotImplementedError(f"Unsupported bvh/type {bvh_type!r}")
            contact_constitution = ConsistentIPCContactConstitution()
            if rigid_contact_proxy is not None:
                rigid_contact_assemble = RigidContactAssemble(get_rigid_contact_assemble_data(contact_system.data))

        for system in (
            GlobalBodyManager(global_body_data),
            GlobalVertexManager(global_vertex_data),
            GlobalSurfaceManager(global_surface_data),
            fem,
            bdf1,
            membrane,
            bending,
            fem_preconditioner,
        ):
            engine.add_system(system)
        if enable_contact:
            for system in (contact_system, broad_phase_system, contact_constitution):
                engine.add_system(system)
            if rigid_contact_proxy is not None:
                for system in (
                    rigid_contact_proxy,
                    rigid_forest,
                    rigid_contact_assemble,
                ):
                    engine.add_system(system)

    engine.build_systems()

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
