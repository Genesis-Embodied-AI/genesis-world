"""
Collider module for rigid body collision detection.

This module provides collision detection functionality for the rigid body solver,
including broad-phase (sweep-and-prune), narrow-phase (convex-convex, SDF-based,
terrain), and contact management.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.kinematic_solver import _balanced_variant_mapping
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_update_geom_aabbs
from genesis.utils.misc import (
    assign_indexed_tensor,
    get_gpu_core_count,
    get_gpu_cores_per_unit,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    tensor_to_array,
)
from genesis.engine.core import Data
from genesis.utils.sdf import SDF

from . import gjk, mpr, support_field
from .broadphase import func_broad_phase
from .constants import CCD_ALGORITHM_CODE
from .contact import (
    collider_kernel_get_contacts,
    collider_kernel_reset,
    func_clamp_prune_contacts,
    func_clamp_prune_contacts_coop,
    func_set_upstream_grad,
    kernel_collider_clear,
    kernel_masked_collider_clear,
)
from .narrowphase import (
    func_fill_diff_contact_input_analytic,
    func_narrow_phase_any_vs_terrain,
    func_narrow_phase_convex_specializations,
    func_narrow_phase_convex_vs_convex,
    func_narrow_phase_diff_convex_vs_convex,
    func_narrow_phase_nonconvex_vs_nonterrain,
    func_narrowphase_contact0,
    func_narrowphase_multicontact,
)

if TYPE_CHECKING:
    from genesis.engine.systems.rigid_solver_system import RigidSolverSystem


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


NEUTRAL_COLLISION_RES_ABS = 0.01
NEUTRAL_COLLISION_RES_REL = 0.05


# WORKAROUND: the collision data belongs to RigidSolverSystem rather than to RigidRigidContactSystem, and exists whether
# or not the scene enables collisions, since the constraint solve assembles the contact constraints straight from
# collider_state and writes the contact forces back into it, while the broad phase reads the weld constraints
# registered at run time from constraint_state. Once those two exchanges are protocols between the contact and the
# constraint Systems, this data moves to RigidRigidContactSystem, its host-side tables and caches leave it, and the
# narrowphase sub-components (MPR, GJK, SupportField), which still read the rigid model through the System they are
# built with, become Data of their own.
class ColliderData(Data):
    """The collision data of the rigid geoms: the collider info, state and static configuration, the narrowphase
    sub-components, and the host-side tables and caches derived from the rigid model."""


def create_collider_data(rigid: "RigidSolverSystem", sdf: SDF) -> "ColliderData":
    """Create the collision data of the rigid geoms from the rigid model, the signed distance fields of the geoms
    being embedded in its collider info."""
    collider = ColliderData()

    collider._mc_perturbation = 1e-3 if rigid._enable_mujoco_compatibility else 3e-3
    collider._mc_tolerance = 1e-3 if rigid._enable_mujoco_compatibility else 1.5e-2
    collider._box_MAXCONPAIR = 16
    collider._diff_pos_tolerance = 1e-2
    collider._diff_normal_tolerance = 1e-2
    collider._prune_deep_penetration_ratio = 3.0
    collider._prune_max_contacts_per_link_pair = 32
    collider._prune_max_contacts_floor = 512

    # The narrowphase sub-components are built AND activated before the collision fields, whose collider info embeds
    # their description structs: the activation of the SDF and of the support field reallocates theirs at their
    # final size, so it has to run before the embedding (see ColliderInfo). Their constructors also resolve the
    # options the static configuration reads, so they run first. The signed distance fields belong to the rigid
    # geoms (see RigidSdfSystem), which other readers query too.
    collider._sdf = sdf
    collider.mpr = mpr.MPR(rigid)
    collider.gjk = gjk.GJK(rigid)
    collider._support_field = support_field.SupportField(rigid)

    _init_static_config(collider, rigid)
    collider._use_split_narrowphase = (
        collider.collider_config.has_non_box_plane_convex_convex and gs.backend != gs.cpu and not rigid._requires_grad
    )

    if collider.collider_config.has_nonconvex_nonterrain:
        collider._sdf.activate()
    # The split narrowphase launches one thread per core, its contact0 pass rounding up to whole environments. An
    # upper-bound estimate of the core count sizes these launches (Genesis-Embodied-AI/Genesis#2616).
    if collider._use_split_narrowphase:
        gpu_cores = collider.collider_config.gpu_cores
        n_contact0_threads = rigid._B * ((gpu_cores + rigid._B - 1) // rigid._B)
        collider.mpr.activate(n_contact0_threads, gpu_cores)
        collider.gjk.activate(n_contact0_threads, gpu_cores)
    elif collider.collider_config.has_non_box_plane_convex_convex:
        collider.gjk.activate()
    if collider.collider_config.has_terrain or collider.collider_config.has_non_box_plane_convex_convex:
        collider._support_field.activate()

    _init_collision_fields(collider, rigid)

    if gs.use_zerocopy:
        # Probe every view the zero-copy contact query needs (including the per-call n_contacts and
        # contact_sort_idx ones, which qd_to_torch caches on their fields). If any field sits past 2**31 bytes
        # in its SNode tree no zero-copy view exists, and get_contacts falls back to the gather-kernel path.
        collider._contact_data: dict[str, torch.Tensor] | None = {}
        try:
            qd_to_torch(collider.collider_state.n_contacts, copy=False)
            qd_to_torch(collider.collider_state.contact_sort_idx, transpose=True, copy=False)
            qd_to_torch(collider.collider_state.first_time, copy=False)
            if collider.collider_config.has_non_box_plane_convex_convex:
                qd_to_torch(collider.collider_state.contact_cache.normal, copy=False)
                qd_to_torch(collider.collider_state.contact_cache.penetration, copy=False)
            for key, name in (
                ("link_a", "link_a"),
                ("link_b", "link_b"),
                ("geom_a", "geom_a"),
                ("geom_b", "geom_b"),
                ("penetration", "penetration"),
                ("position", "pos"),
                ("normal", "normal"),
                ("force", "force"),
            ):
                collider._contact_data[key] = qd_to_torch(
                    getattr(collider.collider_state.contact_data, name), transpose=True, copy=False
                )
        except ValueError:
            collider._contact_data = None

    # Make sure that the initial state is clean
    collider_clear(collider, rigid)
    return collider


def iter_collider_data(collider: "ColliderData") -> Iterator[array_class.DataItem | array_class.ConfigItem]:
    """Yield every array and static config of the collider (see 'Solver.data')."""
    structs = [
        (collider, "collider_config"),
        (collider.gjk, "gjk_config"),
        (collider, "collider_info"),
        (collider, "collider_state"),
        (collider.mpr, "mpr_state"),
    ]
    if collider._use_split_narrowphase:
        structs += [(collider.mpr, name) for name in ("contact0_mpr_state", "multicontact_mpr_state")]
        structs += [(collider.gjk, name) for name in ("contact0_gjk_state", "multicontact_gjk_state")]
    elif collider.collider_config.has_non_box_plane_convex_convex:
        structs.append((collider.gjk, "gjk_state"))
    for owner, name in structs:
        yield from array_class.iter_data(owner, name)


def _init_static_config(collider: "ColliderData", rigid: "RigidSolverSystem") -> None:
    # Identify the convex collision detection (ccd) algorithm
    if rigid._options.use_gjk_collision:
        if rigid._enable_mujoco_compatibility:
            ccd_algorithm = CCD_ALGORITHM_CODE.MJ_GJK
        else:
            ccd_algorithm = CCD_ALGORITHM_CODE.GJK
    else:
        if rigid._enable_mujoco_compatibility:
            ccd_algorithm = CCD_ALGORITHM_CODE.MJ_MPR
        else:
            ccd_algorithm = CCD_ALGORITHM_CODE.MPR

    # The contact patch can hand back the full clipped polygon of a box-box pair, which the reference engine
    # budgets at 8 contacts (see func_clip_polygon); the convex budget must hold it or the extra witnesses of the
    # patch are silently dropped. 'enable_contact_patch' is resolved by the GJK constructor.
    if rigid.rigid_config.requires_grad:
        n_contacts_per_convex_pair = 20
    elif rigid._options.enable_contact_patch:
        n_contacts_per_convex_pair = 8
    else:
        n_contacts_per_convex_pair = 5

    # Nonconvex vertex-vs-SDF pairs and box-box pairs (via their specialized detector) emit many contacts per pair -
    # a full annular ring or face patch - unlike the handful a generic convex pair emits. They share a larger cap,
    # kept separate from the convex cap so the contact buffer is not over-allocated for ordinary convex pairs, and
    # are grouped together for the buffer sizing below. The cap is sized to keep an extended contact patch fully
    # represented: too few points and parts of the patch drop out intermittently as the geometry moves, losing
    # constraint directions and letting bodies slip.
    n_contacts_per_nonconvex_pair = 40
    if rigid._options.box_box_detection and sum(g.type == gs.GEOM_TYPE.BOX for g in rigid.geoms) > 1:
        n_contacts_per_nonconvex_pair = max(n_contacts_per_nonconvex_pair, collider._box_MAXCONPAIR)

    # Compute collision pairs and algorithm flags in a single pass
    (
        collider._n_possible_pairs,
        collider._collision_pair_idx,
        collider._valid_collision_pairs,
        has_terrain,
        has_non_box_plane_convex_convex,
        has_convex_specialization,
        has_nonconvex_nonterrain,
        collider._large_contact_pair_mask,
    ) = _compute_collision_pair_idx(collider, rigid)

    # The hull prune drops a contact only from a (link_a, link_b) bucket of at least 3 contacts, the middle one of 3
    # collinear points being the smallest case, so the pass is compiled in whenever some pair of links can emit 3
    # contacts, and leaves every contact as it is otherwise. Nonconvex geoms and terrain emit many contacts per pair
    # through their vertex-based narrowphase. A convex pair emits several through multi-contact detection or a box
    # specialization, unless it holds a sphere or an ellipsoid, which leaves it a single point. Outside performance
    # mode the pass is always compiled in, being a no-op wherever it would be left out. Composes with contact
    # islands: pruning writes a logical permutation into contact_sort_idx, which the island construction reads.
    has_prunable_contacts = gs.use_ndarray or has_nonconvex_nonterrain or has_terrain
    if not has_prunable_contacts and len(collider._valid_collision_pairs):
        geoms_type = np.array([geom.type for geom in rigid.geoms])
        geoms_link_idx = np.array([geom.link.idx for geom in rigid.geoms])
        pairs_type = geoms_type[collider._valid_collision_pairs]
        pairs_link_idx = np.sort(geoms_link_idx[collider._valid_collision_pairs], axis=1)
        is_point_pair = np.isin(pairs_type, (gs.GEOM_TYPE.SPHERE, gs.GEOM_TYPE.ELLIPSOID)).any(axis=1)
        is_multi_point_pair = ~is_point_pair & (
            rigid._options.enable_multi_contact | (pairs_type == gs.GEOM_TYPE.BOX).any(axis=1)
        )
        _, links_pair_idx = np.unique(pairs_link_idx, axis=0, return_inverse=True)
        links_pair_n_contacts = np.bincount(links_pair_idx, weights=np.where(is_multi_point_pair, 3, 1))
        has_prunable_contacts = bool(links_pair_n_contacts.max() >= 3)

    # Initialize the static config, which stores every data that are compile-time constants.
    # Note that updating any of them will trigger recompilation.
    collider.collider_config = array_class.ColliderStaticConfig(
        gpu_cores=get_gpu_core_count(),
        gpu_cores_per_unit=get_gpu_cores_per_unit(),
        has_terrain=has_terrain,
        # Outside performance mode the convex narrowphase and the contact sort it gates are always compiled in, the
        # narrowphase finding nothing in a scene whose convex pairs all take a box specialization.
        has_non_box_plane_convex_convex=has_non_box_plane_convex_convex or gs.use_ndarray,
        # Outside performance mode the box specializations are always compiled in, their pass finding nothing in a
        # scene without box pairs, which every other path leaves to it by geom type.
        has_convex_specialization=has_convex_specialization or gs.use_ndarray,
        has_nonconvex_nonterrain=has_nonconvex_nonterrain,
        has_prunable_contacts=has_prunable_contacts,
        n_contacts_per_convex_pair=n_contacts_per_convex_pair,
        n_contacts_per_nonconvex_pair=n_contacts_per_nonconvex_pair,
        ccd_algorithm=ccd_algorithm,
    )


def _init_collision_fields(collider: "ColliderData", rigid: "RigidSolverSystem") -> None:
    # Pre-compute fields, as they are needed to initialize the collider state and info.
    vert_neighbors, vert_neighbor_start, vert_n_neighbors = _compute_verts_connectivity(collider, rigid)
    n_vert_neighbors = len(vert_neighbors)
    n_valid_pairs = len(collider._valid_collision_pairs)

    # Initialize [info], which stores every data that must be considered mutable from Quadrants's perspective,
    # i.e. unknown at compile time, but IMMUTABLE from Genesis scene's perspective after build.
    collider.collider_info = array_class.get_collider_info(
        rigid,
        n_vert_neighbors,
        n_valid_pairs,
        collider.collider_config,
        collider.mpr._mpr_info,
        collider.gjk._gjk_info,
        collider._support_field._support_field_info,
        collider._sdf._sdf_info,
        mc_perturbation=collider._mc_perturbation,
        mc_tolerance=collider._mc_tolerance,
        diff_pos_tolerance=collider._diff_pos_tolerance,
        diff_normal_tolerance=collider._diff_normal_tolerance,
        contact_pruning_tolerance=rigid._options.contact_pruning_tolerance or 0.0,
        prune_deep_penetration_ratio=collider._prune_deep_penetration_ratio,
    )
    _init_collision_pair_idx(collider, rigid, collider._collision_pair_idx)
    _init_valid_pairs(collider, rigid)
    _init_verts_connectivity(collider, rigid, vert_neighbors, vert_neighbor_start, vert_n_neighbors)
    _init_verts_spatial_grid(collider, rigid)
    _init_max_contacts(collider, rigid, collider._n_possible_pairs, collider._large_contact_pair_mask)
    _init_terrain_state(collider, rigid)

    # Initialize [state], which stores every data that are may be updated at every single simulation step
    n_possible_pairs_ = max(collider._n_possible_pairs, 1)
    collider.collider_state = array_class.get_collider_state(
        rigid,
        rigid.rigid_config,
        n_possible_pairs_,
        collider.collider_info,
        collider.collider_config,
        split_narrowphase=collider._use_split_narrowphase,
    )

    # 'contact_data_cache' is not used in Quadrants kernels, so keep it outside of the collider state / info
    collider._contact_data_cache: dict[tuple[bool, bool], dict[str, torch.Tensor | tuple[torch.Tensor]]] = {}

    # The warp-per-env cooperative dedup kernel beats the one-env-per-thread fused kernel only while the GPU keeps
    # spare occupancy: past half the cores in envs, the launch oversubscribes the SMs and the fused kernel wins.
    collider._use_coop_dedup = (
        gs.backend != gs.cpu
        and not rigid.rigid_config.requires_grad
        and collider.collider_config.has_prunable_contacts
        and (rigid._options.contact_pruning_tolerance or 0.0) > 0.0
        and rigid._B <= 0.5 * collider.collider_config.gpu_cores
    )


def _compute_collision_pair_idx(collider: "ColliderData", rigid: "RigidSolverSystem"):
    """
    Compute flat indices of all valid collision pairs, plus algorithm flags.

    For each pair of geoms, determine if they can collide based on their properties and the solver configuration.
    Pairs that are already colliding at the initial configuration (qpos0) are filtered out with a warning.

    Returns (n_possible_pairs, collision_pair_idx, pair_flags) where pair_flags is a dict of booleans
    for has_terrain, has_non_box_plane_convex_convex, has_convex_specialization, has_nonconvex_nonterrain.
    """
    # Links whose contact is handled by an external solver (e.g. IPC) — exclude from GJK collision.
    # Only applies when the IPC coupler is active. Mirrors the link filtering logic in
    # IPCCoupler._add_rigid_geoms_to_ipc: for two_way_soft_constraint with a link filter,
    # only the filtered links are in IPC; for all other coupling modes, all links are in IPC.
    from genesis.engine.couplers import IPCCoupler

    n_geoms = rigid.n_geoms
    geoms = rigid.geoms

    if n_geoms == 0:
        empty_pairs = np.empty((0, 2), dtype=gs.np_int)
        empty_mask = np.zeros((0,), dtype=bool)
        return 0, np.full((0, 0), -1, dtype=gs.np_int), empty_pairs, False, False, False, False, empty_mask

    # Links delegated to IPC coupler (skip pair only when BOTH are IPC-handled)
    ipc_delegated_link_idxs = set()
    ipc_only_link_idxs = set()
    if isinstance(rigid.sim.coupler, IPCCoupler):
        for entity in rigid._entities:
            if not entity.material.needs_coup:
                continue
            mode = entity.material.coup_type
            if mode is None:
                continue
            if mode == "ipc_only":
                ipc_only_link_idxs.update(l.idx for l in entity.links)
            link_filter_names = entity.material.coup_links
            if mode == "two_way_soft_constraint" and link_filter_names is not None:
                for name in link_filter_names:
                    ipc_delegated_link_idxs.add(entity.get_link(name=name).idx)
            else:
                ipc_delegated_link_idxs.update(l.idx for l in entity.links)

    # Pre-compute per-geom properties into numpy arrays for vectorized filtering
    geom_link_idx = np.array([g.link.idx for g in geoms], dtype=np.int32)
    geom_root_idx = np.array([g.link.root_idx for g in geoms], dtype=np.int32)
    geom_is_fixed = np.array([g.link.is_fixed for g in geoms], dtype=bool)
    geom_entity_id = np.array([id(g.entity) for g in geoms], dtype=np.int64)
    geom_contype = np.array([g.contype for g in geoms], dtype=np.int64)
    geom_conaffinity = np.array([g.conaffinity for g in geoms], dtype=np.int64)
    geom_local_mask = np.array([g.entity.is_local_collision_mask for g in geoms], dtype=bool)
    geom_is_ipc_only = np.array([g.link.idx in ipc_only_link_idxs for g in geoms], dtype=bool)
    geom_is_ipc_deleg = np.array([g.link.idx in ipc_delegated_link_idxs for g in geoms], dtype=bool)
    geom_type = np.array([g.type for g in geoms], dtype=np.int32)
    geom_is_convex = np.array([g.is_convex for g in geoms], dtype=bool)

    # Build weld pairs set for O(1) lookup (use sorted tuple keys)
    weld_pairs = set()
    for eq in rigid.equalities:
        if eq.type == gs.EQUALITY_TYPE.WELD:
            a, b = eq.eq_obj1id, eq.eq_obj2id
            weld_pairs.add((min(a, b), max(a, b)))

    # --- Vectorized filtering: build upper-triangular valid-pair mask ---
    row, col = np.triu_indices(n_geoms, k=1)

    link_a = geom_link_idx[row]
    link_b = geom_link_idx[col]

    # geoms in the same link
    valid = link_a != link_b

    # Skip all pairs involving ipc_only links
    valid &= ~geom_is_ipc_only[row]
    valid &= ~geom_is_ipc_only[col]

    # Skip pairs where both links are delegated to IPC
    valid &= ~(geom_is_ipc_deleg[row] & geom_is_ipc_deleg[col])

    # pair of fixed links wrt the world
    valid &= ~(geom_is_fixed[row] & geom_is_fixed[col])

    # contype and conaffinity
    same_entity = geom_entity_id[row] == geom_entity_id[col]
    has_local_mask = geom_local_mask[row] | geom_local_mask[col]
    con_match = (geom_contype[row] & geom_conaffinity[col]) | (geom_contype[col] & geom_conaffinity[row])
    con_skip = (same_entity | ~has_local_mask) & (con_match == 0)
    valid &= ~con_skip

    # collider collision (same root) — disabled
    same_root = geom_root_idx[row] == geom_root_idx[col]
    if not rigid._enable_self_collision:
        valid &= ~same_root

    # Weld constraint filtering
    if weld_pairs:
        link_min = np.minimum(link_a, link_b)
        link_max = np.maximum(link_a, link_b)
        is_weld = np.array([(link_min[i], link_max[i]) in weld_pairs for i in range(len(row))], dtype=bool)
        valid &= ~is_weld

    # Heterogeneous variants that no environment carries together never collide, and keeping their pairs would
    # grow the pair count quadratically with the pool size. An entity dispatches the same variant to all of its
    # links in a given environment, so a pair is kept only if some environment carries the variant of both geoms.
    # Each variant of every heterogeneous entity gets its own key, and the key 0 stands for every homogeneous geom.
    geoms_variant_key = np.zeros((n_geoms,), dtype=gs.np_int)
    entities_envs_variant_key = [np.zeros((rigid._B,), dtype=gs.np_int)]
    n_variant_keys = 1
    for entity in rigid.entities:
        if not entity.desc.variants:
            continue
        for link in entity.links:
            for i_variant, (geom_start, geom_end) in enumerate(link._variant_geom_ranges):
                geoms_variant_key[geom_start:geom_end] = n_variant_keys + i_variant
        n_variants = len(entity.desc.variants)
        entities_envs_variant_key.append(n_variant_keys + _balanced_variant_mapping(n_variants, rigid._B))
        n_variant_keys += n_variants
    envs_variant_key = np.stack(entities_envs_variant_key)
    variant_keys_is_co_carried = np.zeros((n_variant_keys, n_variant_keys), dtype=bool)
    variant_keys_is_co_carried[envs_variant_key[:, None], envs_variant_key[None, :]] = True
    valid &= variant_keys_is_co_carried[geoms_variant_key[row], geoms_variant_key[col]]

    # --- Self-collision: adjacent and neutral overlap checks (Python loop, only same-root pairs) ---
    # These checks only apply when self_collision is enabled and the pair passed all vectorized filters
    self_colliding_pairs: list[tuple[int, int]] = []

    needs_self_check = rigid._enable_self_collision and np.any(valid & same_root)
    needs_neutral_check = needs_self_check and not rigid._enable_neutral_collision

    # Lazily compute geom vertices only for geoms that need neutral overlap checks
    geoms_verts: dict[int, np.ndarray] = {}
    if needs_neutral_check:
        self_root_indices = np.where(valid & same_root)[0]
        self_root_geom_idxs = np.unique(np.concatenate([row[self_root_indices], col[self_root_indices]]))
        # Compute vertices only for geoms involved in collider-collision pairs, shrunk by 0.1% to avoid false positive
        # when detecting collider-collision. The pose of a heterogeneous geom is only meaningful in the environments
        # carrying its variant. The pair filter above leaves only variants that some environment carries, and every
        # same-root pair left belongs to a single variant of one entity, so the first environment carrying a geom
        # carries the other geom of its pair as well.
        for i_g in self_root_geom_idxs:
            geom = geoms[i_g]
            i_b = geom.active_envs_idx[0] if geom.active_envs_idx is not None else 0
            env_idx = i_b if rigid.n_envs > 0 else None
            geom_pos = tensor_to_array(geom.get_pos(env_idx, relative=False).reshape((3,)))
            geom_quat = tensor_to_array(geom.get_quat(env_idx, relative=False).reshape((4,)))
            verts = gu.transform_by_trans_quat(geom.init_verts, geom_pos, geom_quat)
            centroid = verts.mean(axis=0, keepdims=True)
            verts = centroid + (1.0 - 1e-3) * (verts - centroid)
            geoms_verts[i_g] = verts

    if needs_self_check:
        self_root_indices = np.where(valid & same_root)[0]
        for idx in self_root_indices:
            i_ga, i_gb = row[idx], col[idx]
            link_ga = geoms[i_ga].link
            link_gb = geoms[i_gb].link

            # adjacent links
            # FIXME: Links should be considered adjacent if connected by only fixed joints.
            if not rigid._enable_adjacent_collision:
                is_adjacent = False
                link_a_, link_b_ = (link_ga, link_gb) if link_ga.idx < link_gb.idx else (link_gb, link_ga)
                while link_b_.parent_idx != -1:
                    if link_b_.parent_idx == link_a_.idx:
                        is_adjacent = True
                        break
                    if not all(joint.type is gs.JOINT_TYPE.FIXED for joint in link_b_.joints):
                        break
                    link_b_ = rigid.links[link_b_.parent_idx]
                if is_adjacent:
                    valid[idx] = False
                    continue

            # active in neutral configuration (qpos0)
            if needs_neutral_check:
                verts_a = geoms_verts[i_ga]
                mesh_a = trimesh.Trimesh(vertices=verts_a, faces=geoms[i_ga].init_faces, process=False)
                verts_b = geoms_verts[i_gb]
                mesh_b = trimesh.Trimesh(vertices=verts_b, faces=geoms[i_gb].init_faces, process=False)
                bounds_a, bounds_b = mesh_a.bounds, mesh_b.bounds
                if not ((bounds_a[1] < bounds_b[0]).any() or (bounds_b[1] < bounds_a[0]).any()):
                    voxels_a = mesh_a.voxelized(
                        pitch=min(NEUTRAL_COLLISION_RES_ABS, NEUTRAL_COLLISION_RES_REL * max(mesh_a.extents))
                    )
                    voxels_b = mesh_b.voxelized(
                        pitch=min(NEUTRAL_COLLISION_RES_ABS, NEUTRAL_COLLISION_RES_REL * max(mesh_b.extents))
                    )
                    coords_a = voxels_a.indices_to_points(np.argwhere(voxels_a.matrix))
                    coords_b = voxels_b.indices_to_points(np.argwhere(voxels_b.matrix))
                    if voxels_a.is_filled(coords_b).any() or voxels_b.is_filled(coords_a).any():
                        self_colliding_pairs.append((i_ga, i_gb))
                        valid[idx] = False
                        continue

    # Emit warning for collider-collision pairs
    if self_colliding_pairs:
        pairs = ", ".join((f"({i_ga}, {i_gb})") for i_ga, i_gb in self_colliding_pairs)
        gs.logger.warning(
            f"Filtered out geometry pairs causing collider-collision for the neutral configuration (qpos0): {pairs}. "
            "Consider tuning Morph option 'decompose_robot_error_threshold' or specify dedicated collision meshes. "
            "This behavior can be disabled by setting Morph option 'enable_neutral_collision=True'."
        )

    # --- Build collision_pair_idx, valid pairs list, and count ---
    valid_indices = np.where(valid)[0]
    n_possible_pairs = len(valid_indices)
    collision_pair_idx = np.full((n_geoms, n_geoms), fill_value=-1, dtype=gs.np_int)
    collision_pair_idx[row[valid_indices], col[valid_indices]] = np.arange(n_possible_pairs, dtype=gs.np_int)

    valid_collision_pairs = np.stack([row[valid_indices], col[valid_indices]], axis=1).astype(gs.np_int)

    # --- Compute algorithm flags from valid pairs ---
    valid_type_a = geom_type[row[valid_indices]]
    valid_type_b = geom_type[col[valid_indices]]
    valid_convex_a = geom_is_convex[row[valid_indices]]
    valid_convex_b = geom_is_convex[col[valid_indices]]

    has_any_vs_terrain = bool(np.any((valid_type_a == gs.GEOM_TYPE.TERRAIN) | (valid_type_b == gs.GEOM_TYPE.TERRAIN)))
    both_convex = valid_convex_a & valid_convex_b
    if np.any(both_convex):
        is_box_a = valid_type_a == gs.GEOM_TYPE.BOX
        is_box_b = valid_type_b == gs.GEOM_TYPE.BOX
        is_plane_a = valid_type_a == gs.GEOM_TYPE.PLANE
        is_plane_b = valid_type_b == gs.GEOM_TYPE.PLANE
        is_plane_box = (is_plane_a & is_box_b) | (is_box_a & is_plane_b)
        specialized = is_plane_box
        if rigid._options.box_box_detection:
            specialized = specialized | (is_box_a & is_box_b)
        has_non_box_plane_convex_convex = bool(np.any(both_convex & ~specialized))

        # Differentiable contact detection (diff_gjk) reconstructs each contact from a triangular face of the
        # Minkowski difference. A sphere or ellipsoid has no flat facet, so a pair of them yields an everywhere
        # smoothly curved Minkowski boundary on which EPA never converges, and no contact is ever generated -
        # the bodies silently tunnel. Faceted partners (box, mesh) and the analytical plane branch are unaffected,
        # and sphere-sphere pairs are reconstructed in closed form (func_differentiable_sphere_contact).
        if rigid._requires_grad:
            is_smooth_a = (valid_type_a == gs.GEOM_TYPE.SPHERE) | (valid_type_a == gs.GEOM_TYPE.ELLIPSOID)
            is_smooth_b = (valid_type_b == gs.GEOM_TYPE.SPHERE) | (valid_type_b == gs.GEOM_TYPE.ELLIPSOID)
            is_sphere_sphere = (valid_type_a == gs.GEOM_TYPE.SPHERE) & (valid_type_b == gs.GEOM_TYPE.SPHERE)
            if np.any(both_convex & ~specialized & is_smooth_a & is_smooth_b & ~is_sphere_sphere):
                gs.raise_exception(
                    "Differentiable contact detection is not supported for sphere-ellipsoid or "
                    "ellipsoid-ellipsoid collision pairs (requires_grad=True). Approximate them with a faceted "
                    "geometry (e.g. a convex mesh) or disable requires_grad."
                )
    else:
        has_non_box_plane_convex_convex = False

    if rigid._options.box_box_detection:
        spec_types = [gs.GEOM_TYPE.TERRAIN, gs.GEOM_TYPE.BOX]
        has_convex_specialization = bool(np.any(np.isin(valid_type_a, spec_types) | np.isin(valid_type_b, spec_types)))
    else:
        has_convex_specialization = bool(
            np.any(
                ((valid_type_a == gs.GEOM_TYPE.BOX) & (valid_type_b == gs.GEOM_TYPE.PLANE))
                | ((valid_type_a == gs.GEOM_TYPE.PLANE) & (valid_type_b == gs.GEOM_TYPE.BOX))
            )
        )

    # Pairs routed to the vertex-vs-SDF nonconvex narrowphase (at least one nonconvex geom, neither terrain).
    nonconvex_pair_mask = (
        ~(valid_convex_a & valid_convex_b)
        & (valid_type_a != gs.GEOM_TYPE.TERRAIN)
        & (valid_type_b != gs.GEOM_TYPE.TERRAIN)
    )
    has_nonconvex_vs_nonterrain = bool(np.any(nonconvex_pair_mask))
    # Pairs that emit many contacts per pair, used to size the contact buffer. Box-box pairs (when the specialized
    # detector is enabled) emit up to box_MAXCONPAIR contacts, so they join the nonconvex pairs in the large-cap
    # bucket rather than being sized as generic convex pairs.
    large_contact_mask = nonconvex_pair_mask
    if rigid._options.box_box_detection:
        large_contact_mask = large_contact_mask | (
            (valid_type_a == gs.GEOM_TYPE.BOX) & (valid_type_b == gs.GEOM_TYPE.BOX)
        )

    return (
        n_possible_pairs,
        collision_pair_idx,
        valid_collision_pairs,
        has_any_vs_terrain,
        has_non_box_plane_convex_convex,
        has_convex_specialization,
        has_nonconvex_vs_nonterrain,
        large_contact_mask,
    )


def _compute_verts_connectivity(collider: "ColliderData", rigid: "RigidSolverSystem"):
    """
    Compute the vertex connectivity.
    """
    vert_neighbors = []
    vert_neighbor_start = []
    vert_n_neighbors = []
    offset = 0
    for geom in rigid.geoms:
        vert_neighbors.append(geom.vert_neighbors + geom.vert_start)
        vert_neighbor_start.append(geom.vert_neighbor_start + offset)
        vert_n_neighbors.append(geom.vert_n_neighbors)
        offset = offset + len(geom.vert_neighbors)

    if rigid.n_verts > 0:
        vert_neighbors = np.concatenate(vert_neighbors, dtype=gs.np_int)
        vert_neighbor_start = np.concatenate(vert_neighbor_start, dtype=gs.np_int)
        vert_n_neighbors = np.concatenate(vert_n_neighbors, dtype=gs.np_int)

    return vert_neighbors, vert_neighbor_start, vert_n_neighbors


def _init_verts_spatial_grid(collider: "ColliderData", rigid: "RigidSolverSystem"):
    """
    Sort each geom's collision verts into a fixed 8x8x8 grid over its local AABB, as a permutation of vert
    indices ordered by grid cell (z fastest) plus per-cell vert ranges.

    The nonconvex narrowphase visits only the cells overlapping the other geom's pulled-back AABB, skipping far
    verts wholesale. Binning uses the same single-precision cell mapping as the kernel side; that mapping is
    monotone, so a vert inside a (padded) query box always lands inside the visited cell range.
    """
    if rigid.n_verts == 0:
        return
    n_geoms = len(rigid.geoms)
    verts_idx = []
    verts_pos = []
    cells_vert_start = []
    geoms_origin = np.zeros((n_geoms, 3), dtype=gs.np_float)
    geoms_inv_cell_size = np.zeros((n_geoms, 3), dtype=gs.np_float)
    offset_vert = 0
    for i_g, geom in enumerate(rigid.geoms):
        verts = geom.init_verts.astype(gs.np_float)
        if len(verts) == 0:
            verts_idx.append(np.zeros(0, dtype=gs.np_int))
            verts_pos.append(np.zeros((0, 3), dtype=gs.np_float))
            cells_vert_start.append(np.full(8**3 + 1, offset_vert, dtype=gs.np_int))
            continue
        origin = verts.min(axis=0)
        extent = verts.max(axis=0) - origin
        inv_cell_size = np.where(extent > 0.0, 8 / np.maximum(extent, gs.EPS), 0.0).astype(gs.np_float)
        verts_cell = np.clip(np.floor((verts - origin) * inv_cell_size), 0, 7).astype(gs.np_int)
        verts_cell_flat = (verts_cell[:, 0] * 8 + verts_cell[:, 1]) * 8 + verts_cell[:, 2]
        order = np.argsort(verts_cell_flat, kind="stable")
        verts_idx.append(order + geom.vert_start)
        # The positions themselves are duplicated in spatial order: the scan streams them sequentially, where
        # gathering through the permutation would defeat the prefetcher on the hot path.
        verts_pos.append(verts[order])
        counts = np.bincount(verts_cell_flat, minlength=8**3)
        cells_vert_start.append(np.concatenate(([0], counts.cumsum())) + offset_vert)
        geoms_origin[i_g] = origin
        geoms_inv_cell_size[i_g] = inv_cell_size
        offset_vert = offset_vert + len(verts)

    verts_spatial_grid = collider.collider_info.verts_spatial_grid
    verts_spatial_grid.verts_idx.from_numpy(np.concatenate(verts_idx, dtype=gs.np_int))
    verts_spatial_grid.verts_pos.from_numpy(np.concatenate(verts_pos, dtype=gs.np_float))
    verts_spatial_grid.cells_vert_start.from_numpy(np.concatenate(cells_vert_start, dtype=gs.np_int))
    verts_spatial_grid.geoms_origin.from_numpy(geoms_origin)
    verts_spatial_grid.geoms_inv_cell_size.from_numpy(geoms_inv_cell_size)


def _init_collision_pair_idx(collider: "ColliderData", rigid: "RigidSolverSystem", collision_pair_idx):
    if collider._n_possible_pairs == 0:
        collider.collider_info.collision_pair_idx.fill(-1)
        return
    collider.collider_info.collision_pair_idx.from_numpy(collision_pair_idx)


def _init_valid_pairs(collider: "ColliderData", rigid: "RigidSolverSystem"):
    if len(collider._valid_collision_pairs) > 0:
        collider.collider_info.valid_collision_pairs.from_numpy(collider._valid_collision_pairs)


def _init_verts_connectivity(
    collider: "ColliderData", rigid: "RigidSolverSystem", vert_neighbors, vert_neighbor_start, vert_n_neighbors
):
    if rigid.n_verts > 0:
        collider.collider_info.vert_neighbors.from_numpy(vert_neighbors)
        collider.collider_info.vert_neighbor_start.from_numpy(vert_neighbor_start)
        collider.collider_info.vert_n_neighbors.from_numpy(vert_n_neighbors)


def _init_max_contacts(collider: "ColliderData", rigid: "RigidSolverSystem", n_possible_pairs, large_contact_pair_mask):
    n_possible_nonconvex_pairs = int(np.count_nonzero(large_contact_pair_mask))
    max_collision_pairs = min(rigid.max_collision_pairs, n_possible_pairs)
    # Size the contact buffer per regime: nonconvex pairs each emit up to n_contacts_per_nonconvex_pair, convex and
    # terrain pairs up to n_contacts_per_convex_pair. The worst case fills the capped pair budget with as many
    # (larger-cap) nonconvex pairs as exist, then the rest with convex pairs. The budget of a nonconvex pair is
    # shared between its two vertex scans: the verification scan appends while the pair is under its cap and then
    # only displaces the pair's least-penetrating contact, so the cap holds regardless of the number of scans.
    cap_nonconvex = collider.collider_config.n_contacts_per_nonconvex_pair
    cap_convex = collider.collider_config.n_contacts_per_convex_pair
    n_nonconvex = min(n_possible_nonconvex_pairs, max_collision_pairs)
    n_convex = min(n_possible_pairs - n_possible_nonconvex_pairs, max_collision_pairs - n_nonconvex)
    max_candidate_contacts = n_nonconvex * cap_nonconvex + n_convex * cap_convex
    # The broad phase reports each possible pair at most once per environment
    max_collision_pairs_broad = min(
        max_collision_pairs * rigid._options.multiplier_collision_broad_phase, n_possible_pairs
    )

    # Post-pruning contact budget for sizing the contact constraint buffers. The physical contact buffer must hold
    # everything the narrowphase can emit (max_candidate_contacts), but the constraint solver only consumes
    # contacts surviving link-pair pruning, which keeps roughly the 2D support polygon of each (link_a, link_b)
    # contact patch. Cap each candidate link pair at _prune_max_contacts_per_link_pair points (or the sum of its
    # geom-pair caps if smaller) instead of the per-geom-pair worst case. This is a heuristic, not a hard
    # guarantee: conforming (non-coplanar) contact patches are not pruned and can exceed the cap, which clamps
    # the contact count and halts the simulation with a request to increase 'max_contacts'. Tightening only pays
    # off when the worst case is large, so the budget never goes below _prune_max_contacts_floor: under it, the
    # halt risk buys no meaningful memory savings. The constraint solver overrides this budget at build time when
    # 'max_contacts' is set.
    max_contacts = max_candidate_contacts
    if (
        collider.collider_config.has_prunable_contacts
        and not rigid._requires_grad
        and rigid._options.contact_pruning_tolerance is not None
    ):
        geoms_link_idx = np.array([geom.link.idx for geom in rigid.geoms], dtype=np.int64)
        pairs_link_idx = geoms_link_idx[collider._valid_collision_pairs]
        pairs_key = rigid.n_links * pairs_link_idx.min(axis=1) + pairs_link_idx.max(axis=1)
        _, pairs_group_idx = np.unique(pairs_key, return_inverse=True)
        pairs_n_contacts = np.where(large_contact_pair_mask, cap_nonconvex, cap_convex)
        link_pairs_n_contacts = np.bincount(pairs_group_idx, weights=pairs_n_contacts)
        max_contacts_pruned = np.minimum(link_pairs_n_contacts, collider._prune_max_contacts_per_link_pair)
        max_contacts_pruned_total = max(int(max_contacts_pruned.sum()), collider._prune_max_contacts_floor)
        max_contacts = min(max_contacts, max_contacts_pruned_total)

    collider.collider_info.max_possible_pairs[None] = n_possible_pairs
    collider.collider_info.max_collision_pairs[None] = max_collision_pairs
    collider.collider_info.max_collision_pairs_broad[None] = max_collision_pairs_broad
    collider.collider_info.max_candidate_contacts[None] = max_candidate_contacts
    collider.collider_info.max_contacts[None] = max_contacts


def _init_terrain_state(collider: "ColliderData", rigid: "RigidSolverSystem"):
    if collider.collider_config.has_terrain:
        solver = rigid
        links_idx = solver.dyn_info.geoms.link_idx.to_numpy()[
            solver.dyn_info.geoms.type.to_numpy() == gs.GEOM_TYPE.TERRAIN
        ]
        entity_idx = solver.dyn_info.links.entity_idx.to_numpy()[links_idx[0]]
        if isinstance(entity_idx, np.ndarray):
            entity_idx = entity_idx[0]
        entity = solver._entities[entity_idx]

        rc = np.array(entity.terrain_hf.shape, dtype=gs.np_int)
        xyz_maxmin = np.array(
            [
                rc[0] * entity.terrain_scale[0],
                rc[1] * entity.terrain_scale[0],
                entity.terrain_hf.max(),
                0,
                0,
                entity.terrain_hf.min() - 1.0,
            ],
            dtype=gs.np_float,
        )

        collider.collider_info.terrain_hf.from_numpy(entity.terrain_hf)
        collider.collider_info.terrain_rc.from_numpy(rc)
        collider.collider_info.terrain_scale.from_numpy(entity.terrain_scale)
        collider.collider_info.terrain_xyz_maxmin.from_numpy(xyz_maxmin)


def collider_reset(
    collider: "ColliderData", rigid: "RigidSolverSystem", envs_idx=None, *, cache_only: bool = True
) -> None:
    collider._contact_data_cache.clear()
    if gs.use_zerocopy and collider._contact_data is not None:
        envs_mask = indices_to_mask(envs_idx)
        if not cache_only:
            first_time = qd_to_torch(collider.collider_state.first_time, copy=False)
            assign_indexed_tensor(first_time, envs_mask, True)

        if collider.collider_config.has_non_box_plane_convex_convex:
            pairs_mask = (slice(None), *envs_mask)
            normal = qd_to_torch(collider.collider_state.contact_cache.normal, copy=False)
            penetration = qd_to_torch(collider.collider_state.contact_cache.penetration, copy=False)
            assign_indexed_tensor(normal, pairs_mask, 0.0)
            assign_indexed_tensor(penetration, pairs_mask, 0.0)

        if gs.backend == gs.metal:
            torch.mps.synchronize()
        return

    envs_idx = rigid._scene._sanitize_envs_idx(envs_idx)
    collider_kernel_reset(envs_idx, collider.collider_state, rigid.rigid_config, collider.collider_config, cache_only)


def collider_clear(collider: "ColliderData", rigid: "RigidSolverSystem", envs_idx=None):
    collider_reset(collider, rigid, envs_idx, cache_only=False)

    if (
        gs.use_zerocopy
        and collider._contact_data is not None
        and not rigid._use_hibernation
        and (not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool))
    ):
        n_contacts = qd_to_torch(collider.collider_state.n_contacts, copy=False)
        link_a = qd_to_torch(collider.collider_state.contact_data.link_a, copy=False)
        link_b = qd_to_torch(collider.collider_state.contact_data.link_b, copy=False)
        geom_a = qd_to_torch(collider.collider_state.contact_data.geom_a, copy=False)
        geom_b = qd_to_torch(collider.collider_state.contact_data.geom_b, copy=False)
        penetration = qd_to_torch(collider.collider_state.contact_data.penetration, copy=False)
        pos = qd_to_torch(collider.collider_state.contact_data.pos, copy=False)
        normal = qd_to_torch(collider.collider_state.contact_data.normal, copy=False)
        force = qd_to_torch(collider.collider_state.contact_data.force, copy=False)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            n_contacts.masked_fill_(envs_idx, 0)
            link_a.masked_fill_(envs_idx[None, :], -1)
            link_b.masked_fill_(envs_idx[None, :], -1)
            geom_a.masked_fill_(envs_idx[None, :], -1)
            geom_b.masked_fill_(envs_idx[None, :], -1)
            penetration.masked_fill_(envs_idx[None, :], 0.0)
            pos.masked_fill_(envs_idx[None, :, None], 0.0)
            normal.masked_fill_(envs_idx[None, :, None], 0.0)
            force.masked_fill_(envs_idx[None, :, None], 0.0)
        elif isinstance(envs_idx, torch.Tensor):
            n_contacts.scatter_(0, envs_idx, 0)
            link_a.scatter_(1, envs_idx[None, :].expand(link_a.shape[0], -1), -1)
            link_b.scatter_(1, envs_idx[None, :].expand(link_b.shape[0], -1), -1)
            geom_a.scatter_(1, envs_idx[None, :].expand(geom_a.shape[0], -1), -1)
            geom_b.scatter_(1, envs_idx[None, :].expand(geom_b.shape[0], -1), -1)
            penetration.scatter_(1, envs_idx[None, :].expand(link_a.shape[0], -1), 0.0)
            pos.scatter_(1, envs_idx[None, :, None].expand(link_a.shape[0], -1, 3), 0.0)
            normal.scatter_(1, envs_idx[None, :, None].expand(link_a.shape[0], -1, 3), 0.0)
            force.scatter_(1, envs_idx[None, :, None].expand(link_a.shape[0], -1, 3), 0.0)
        else:
            env_mask = indices_to_mask(envs_idx)
            n_contacts[env_mask] = 0
            link_a[:, envs_idx] = -1
            link_b[:, envs_idx] = -1
            geom_a[:, envs_idx] = -1
            geom_b[:, envs_idx] = -1
            penetration[:, envs_idx] = 0.0
            pos[:, envs_idx] = 0.0
            normal[:, envs_idx] = 0.0
            force[:, envs_idx] = 0.0
        if gs.backend == gs.metal:
            torch.mps.synchronize()
        return

    if not isinstance(envs_idx, torch.Tensor):
        envs_idx = rigid._scene._sanitize_envs_idx(envs_idx)
    if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
        fn = kernel_masked_collider_clear
    else:
        fn = kernel_collider_clear
    fn(
        envs_idx,
        rigid.dyn_state,
        collider.collider_state,
        rigid.dyn_info,
        rigid.rigid_info,
        rigid.rigid_config,
    )


def collider_detection(collider: "ColliderData", rigid: "RigidSolverSystem") -> None:
    kernel_detection(
        rigid.geoms_init_AABB,
        rigid.dyn_state,
        collider.collider_state,
        collider.mpr.mpr_state,
        collider.gjk.gjk_state,
        collider.gjk.gjk_state.diff_contact_input,
        collider.mpr.contact0_mpr_state,
        collider.gjk.contact0_gjk_state,
        collider.mpr.multicontact_mpr_state,
        collider.gjk.multicontact_gjk_state,
        rigid.constraint_solver.constraint_state,
        rigid.dyn_info,
        rigid.rigid_info,
        collider.collider_info,
        rigid.rigid_config,
        collider.collider_config,
        collider.gjk.gjk_config,
        collider._n_possible_pairs > 0,
        collider._use_split_narrowphase,
        collider._use_coop_dedup,
        rigid._errno,
    )
    collider._contact_data_cache.clear()


def collider_get_contacts(
    collider: "ColliderData",
    rigid: "RigidSolverSystem",
    as_tensor: bool = True,
    to_torch: bool = True,
    keep_batch_dim: bool = False,
    is_padded: bool = False,
):
    # Early return if already pre-computed
    contact_data = collider._contact_data_cache.setdefault((as_tensor, to_torch, is_padded), {})
    if contact_data:
        return contact_data.copy()

    n_envs = rigid.n_envs
    # When pruning and spatial sort are both statically disabled, contact_sort_idx is guaranteed to stay at the
    # identity permutation, so the physical layout of contact_data already matches the logical order. In that
    # case the zero-copy fast path returns torch views over contact_data storage truncated to n_contacts_max.
    # Otherwise the permutation must be applied: read both contact_data and contact_sort_idx as zero-copy torch
    # views, then materialize each field via a single torch.gather along the contact axis. This still avoids the
    # Quadrants gather kernel and produces a contiguous output suitable for downstream consumers.
    zerocopy_aligned = (
        not collider.collider_config.has_prunable_contacts
        and not collider.collider_config.has_non_box_plane_convex_convex
    )
    if gs.use_zerocopy and collider._contact_data is not None:
        n_contacts = qd_to_torch(collider.collider_state.n_contacts, copy=False)
        # 'is_padded' returns fixed-capacity tensors (plus per-env 'n_contacts'), skipping the
        # 'n_contacts.max().item()' device->host sync that trimming to the live count needs.
        padded = is_padded
        if (as_tensor or n_envs == 0) and not padded:
            n_contacts_max = (n_contacts if n_envs == 0 else n_contacts.max()).item()

        if not zerocopy_aligned:
            # Build a (_B, n_contacts_max) index tensor once, expanded to (_B, n_contacts_max, 3) for vector
            # fields. n_contacts_max comes from the max-across-envs reduction so the same index drives every
            # field; per-env trimming to n_contacts[i] happens in the ragged split below.
            if not (as_tensor or n_envs == 0) and not padded:
                n_contacts_max = n_contacts.max().item()
            sort_idx_view = qd_to_torch(collider.collider_state.contact_sort_idx, transpose=True, copy=False)
            if padded:
                # Gather the full capacity so no host-side trim (sync) is needed. Sort indices past the live
                # range may be stale; clamp (out-of-place, preserving the zero-copy view) keeps them in-bounds.
                gather_idx_flat = sort_idx_view.clamp(0, sort_idx_view.shape[1] - 1)
            else:
                gather_idx_flat = sort_idx_view[:, :n_contacts_max]
            # FIXME: MPS gathers zeros through an expanded index reading a zero-copy view of the quadrants
            # buffers, while a contiguous one reads correctly. The copy costs one small allocation per call.
            gather_idx_vec = gather_idx_flat.unsqueeze(-1).expand(-1, -1, 3).contiguous()
            # Gather indices past each env's n_contacts are stale (the permutation only fills the live range), so
            # the dense (n_envs, n_contacts_max) tensor has padding columns to reset to the per-field sentinel.
            # The mask is field-independent, so build it once and broadcast over scalar and vector fields alike.
            pad_mask = None
            if as_tensor and n_envs > 0 and not padded:
                pad_mask = torch.arange(n_contacts_max, device=sort_idx_view.device)[None, :] >= n_contacts[:, None]

        for key, data in collider._contact_data.items():
            if zerocopy_aligned:
                if n_envs == 0:
                    if padded:
                        data = data if keep_batch_dim else data[0]
                    else:
                        data = data[0, :n_contacts_max] if not keep_batch_dim else data[:, :n_contacts_max]
                elif as_tensor and not padded:
                    data = data[:, :n_contacts_max]
                if to_torch:
                    if gs.backend == gs.cpu:
                        data = data.clone()
                else:
                    data = tensor_to_array(data)
            else:
                # data shape is (_B, max_candidate_contacts) for scalars, with a trailing 3 axis for vectors.
                gidx = gather_idx_vec if data.dim() == 3 else gather_idx_flat
                data = data.gather(dim=1, index=gidx)
                if pad_mask is not None:
                    mask = pad_mask if data.dim() == 2 else pad_mask[..., None]
                    data.masked_fill_(mask, -1 if data.dtype == gs.tc_int else 0)
                if n_envs == 0 and not keep_batch_dim:
                    data = data[0]
                if not to_torch:
                    data = tensor_to_array(data)

            if n_envs > 0 and not as_tensor:
                if padded:
                    data = tuple(data[i : i + 1] if keep_batch_dim else data[i] for i in range(n_envs))
                elif keep_batch_dim:
                    data = tuple(data[i : i + 1, :j] for i, j in enumerate(n_contacts.tolist()))
                else:
                    data = tuple(data[i, :j] for i, j in enumerate(n_contacts.tolist()))

            contact_data[key] = data

        if padded:
            contact_data["n_contacts"] = n_contacts if to_torch else tensor_to_array(n_contacts)

        return contact_data.copy()

    # Find out how much dynamic memory must be allocated
    n_contacts = qd_to_numpy(collider.collider_state.n_contacts)
    n_contacts_max = n_contacts.max().item()
    # 'is_padded' is honored here too so the result matches the zero-copy path on every backend (this path
    # already syncs to size its buffers, so padding only fixes the shape). Padding needs the dense rectangular
    # layout, so force it even for a ragged (as_tensor=False) request; per-env slices are taken from it below.
    padded = is_padded
    dense = as_tensor or padded
    n_contacts_arr = n_contacts
    if dense:
        out_size = n_contacts_max * max(n_envs, 1)
    else:
        *n_contacts_starts, out_size = np.cumsum(n_contacts)
    n_contacts = n_contacts.tolist()

    # Allocate output buffer
    if to_torch:
        iout = torch.full((out_size, 4), -1, dtype=gs.tc_int, device=gs.device)
        fout = torch.zeros((out_size, 10), dtype=gs.tc_float, device=gs.device)
    else:
        iout = np.full((out_size, 4), -1, dtype=gs.np_int)
        fout = np.zeros((out_size, 10), dtype=gs.np_float)

    # Copy contact data
    if n_contacts_max > 0:
        collider_kernel_get_contacts(iout, fout, collider.collider_state, rigid.rigid_config, dense)

    # Build structured view (no copy)
    if dense:
        nb = max(n_envs, 1)
        iout = iout.reshape((nb, n_contacts_max, 4))
        fout = fout.reshape((nb, n_contacts_max, 10))
        if padded:
            # Widen from the live count to the fixed capacity the zero-copy path returns (contact buffers are
            # sized max(max_candidate_contacts, 1)). New slots keep the sentinel (-1 ints / 0 floats).
            capacity = max(collider.collider_info.max_candidate_contacts[None], 1)
            if to_torch:
                iout_full = torch.full((nb, capacity, 4), -1, dtype=gs.tc_int, device=gs.device)
                fout_full = torch.zeros((nb, capacity, 10), dtype=gs.tc_float, device=gs.device)
            else:
                iout_full = np.full((nb, capacity, 4), -1, dtype=gs.np_int)
                fout_full = np.zeros((nb, capacity, 10), dtype=gs.np_float)
            iout_full[:, :n_contacts_max] = iout
            fout_full[:, :n_contacts_max] = fout
            iout, fout = iout_full, fout_full
        if n_envs == 0 and not keep_batch_dim:
            iout, fout = iout[0], fout[0]
        if as_tensor or n_envs == 0:
            iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
            fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
            values = (*iout_chunks, *fout_chunks)
        else:
            # Ragged + padded, batched: one fixed-capacity slice per env (no host-side trim, no sync).
            iout_envs = (iout[i : i + 1] if keep_batch_dim else iout[i] for i in range(n_envs))
            fout_envs = (fout[i : i + 1] if keep_batch_dim else fout[i] for i in range(n_envs))
            iout_chunks = ((io[..., 0], io[..., 1], io[..., 2], io[..., 3]) for io in iout_envs)
            fout_chunks = ((fo[..., 0], fo[..., 1:4], fo[..., 4:7], fo[..., 7:]) for fo in fout_envs)
            values = (*zip(*iout_chunks), *zip(*fout_chunks))
    else:
        # Split smallest dimension first, then largest dimension
        if n_envs == 0:
            iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
            fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
            values = (*iout_chunks, *fout_chunks)
        elif n_contacts_max >= n_envs:
            if to_torch:
                iout_chunks = torch.split(iout, n_contacts)
                fout_chunks = torch.split(fout, n_contacts)
            else:
                iout_chunks = np.split(iout, n_contacts_starts)
                fout_chunks = np.split(fout, n_contacts_starts)
            iout_chunks = ((out[..., 0], out[..., 1], out[..., 2], out[..., 3]) for out in iout_chunks)
            fout_chunks = ((out[..., 0], out[..., 1:4], out[..., 4:7], out[..., 7:]) for out in fout_chunks)
            values = (*zip(*iout_chunks), *zip(*fout_chunks))
        else:
            iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
            fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
            if n_envs == 1:
                values = [(value,) for value in (*iout_chunks, *fout_chunks)]
            else:
                if to_torch:
                    iout_chunks = (torch.split(out, n_contacts) for out in iout_chunks)
                    fout_chunks = (torch.split(out, n_contacts) for out in fout_chunks)
                else:
                    iout_chunks = (np.split(out, n_contacts_starts) for out in iout_chunks)
                    fout_chunks = (np.split(out, n_contacts_starts) for out in fout_chunks)
                values = (*iout_chunks, *fout_chunks)

    # Store contact information in cache
    contact_data.update(
        zip(("link_a", "link_b", "geom_a", "geom_b", "penetration", "position", "normal", "force"), values)
    )

    if padded:
        contact_data["n_contacts"] = torch.as_tensor(n_contacts_arr, device=gs.device) if to_torch else n_contacts_arr

    return contact_data.copy()


def collider_backward(collider: "ColliderData", rigid: "RigidSolverSystem", dL_dposition, dL_dnormal, dL_dpenetration):
    func_set_upstream_grad(dL_dposition, dL_dnormal, dL_dpenetration, collider.collider_state)
    collider_backward_narrowphase(collider, rigid)


def collider_backward_narrowphase(collider: "ColliderData", rigid: "RigidSolverSystem"):
    func_narrow_phase_diff_convex_vs_convex.grad(
        rigid.dyn_state,
        collider.collider_state,
        collider.collider_state.diff_contact_input,
        rigid.dyn_info,
        rigid.rigid_info,
        collider.collider_info,
        rigid.rigid_config,
        rigid._errno,
    )


def forward_to_collider_data(name: str) -> property:
    """Attribute of the facade that forwards to the collision data, settable as an attribute of the collider is."""

    def get(facade):
        return getattr(facade._rigid.collider_data, name)

    def set(facade, value):
        setattr(facade._rigid.collider_data, name, value)

    return property(get, set)


class Collider:
    """User-facing access to the collision detection of the rigid geoms, whose data RigidSolverSystem owns (see
    ColliderData)."""

    collider_config = forward_to_collider_data("collider_config")
    collider_info = forward_to_collider_data("collider_info")
    collider_state = forward_to_collider_data("collider_state")
    mpr = forward_to_collider_data("mpr")
    gjk = forward_to_collider_data("gjk")
    _mc_perturbation = forward_to_collider_data("_mc_perturbation")
    _mc_tolerance = forward_to_collider_data("_mc_tolerance")
    _box_MAXCONPAIR = forward_to_collider_data("_box_MAXCONPAIR")
    _diff_pos_tolerance = forward_to_collider_data("_diff_pos_tolerance")
    _diff_normal_tolerance = forward_to_collider_data("_diff_normal_tolerance")
    _prune_deep_penetration_ratio = forward_to_collider_data("_prune_deep_penetration_ratio")
    _prune_max_contacts_per_link_pair = forward_to_collider_data("_prune_max_contacts_per_link_pair")
    _prune_max_contacts_floor = forward_to_collider_data("_prune_max_contacts_floor")
    _sdf = forward_to_collider_data("_sdf")
    _support_field = forward_to_collider_data("_support_field")
    _large_contact_pair_mask = forward_to_collider_data("_large_contact_pair_mask")
    _contact_data = forward_to_collider_data("_contact_data")
    _collision_pair_idx = forward_to_collider_data("_collision_pair_idx")
    _valid_collision_pairs = forward_to_collider_data("_valid_collision_pairs")
    _n_possible_pairs = forward_to_collider_data("_n_possible_pairs")
    _use_split_narrowphase = forward_to_collider_data("_use_split_narrowphase")
    _use_coop_dedup = forward_to_collider_data("_use_coop_dedup")
    _contact_data_cache = forward_to_collider_data("_contact_data_cache")

    def __init__(self, rigid_solver_system: "RigidSolverSystem") -> None:
        self._rigid = rigid_solver_system

    @property
    def data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        """Yield every array and static config of the collider (see 'Solver.data')."""
        return iter_collider_data(self._rigid.collider_data)

    def reset(self, envs_idx=None, *, cache_only: bool = True) -> None:
        collider_reset(self._rigid.collider_data, self._rigid, envs_idx, cache_only=cache_only)

    def clear(self, envs_idx=None) -> None:
        collider_clear(self._rigid.collider_data, self._rigid, envs_idx)

    def detection(self) -> None:
        collider_detection(self._rigid.collider_data, self._rigid)

    def get_contacts(
        self, as_tensor: bool = True, to_torch: bool = True, keep_batch_dim: bool = False, is_padded: bool = False
    ):
        return collider_get_contacts(
            self._rigid.collider_data, self._rigid, as_tensor, to_torch, keep_batch_dim, is_padded
        )

    def backward(self, dL_dposition, dL_dnormal, dL_dpenetration) -> None:
        collider_backward(self._rigid.collider_data, self._rigid, dL_dposition, dL_dnormal, dL_dpenetration)

    def backward_narrowphase(self) -> None:
        collider_backward_narrowphase(self._rigid.collider_data, self._rigid)


@qd.func
def func_detection(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    diff_contact_input: array_class.DiffContactInput,
    contact0_mpr_state: array_class.MPRState,
    contact0_gjk_state: array_class.GJKState,
    multicontact_mpr_state: array_class.MPRState,
    multicontact_gjk_state: array_class.GJKState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    has_possible_pairs: qd.template(),
    split_narrowphase: qd.template(),
    coop_dedup: qd.template(),
    errno: qd.Tensor,
):
    """Update the geom bounding boxes, then run the broad phase, the narrow phase and the contact pruning.

    has_possible_pairs is False when no geom pair can collide, in which case only the bounding boxes are updated.
    split_narrowphase runs the convex narrow phase as a contact 0 pass followed by a multi-contact pass, on their own
    scratch states. coop_dedup prunes the contacts cooperatively.
    """
    func_update_geom_aabbs(geoms_init_AABB, dyn_state, dyn_info, rigid_config)
    if qd.static(has_possible_pairs):
        func_broad_phase(
            dyn_state,
            collider_state,
            constraint_state,
            dyn_info,
            rigid_info,
            collider_info,
            rigid_config,
            collider_static_config,
            errno,
        )
        if qd.static(split_narrowphase):
            func_narrowphase_contact0(
                geoms_init_AABB,
                dyn_state,
                collider_state,
                contact0_mpr_state,
                contact0_gjk_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
                errno,
            )
            func_narrowphase_multicontact(
                geoms_init_AABB,
                dyn_state,
                collider_state,
                multicontact_mpr_state,
                multicontact_gjk_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
                gjk_static_config,
                errno,
            )
        elif qd.static(collider_static_config.has_non_box_plane_convex_convex):
            func_narrow_phase_convex_vs_convex(
                geoms_init_AABB=geoms_init_AABB,
                dyn_state=dyn_state,
                collider_state=collider_state,
                mpr_state=mpr_state,
                gjk_state=gjk_state,
                diff_contact_input=diff_contact_input,
                dyn_info=dyn_info,
                rigid_info=rigid_info,
                collider_info=collider_info,
                rigid_config=rigid_config,
                collider_static_config=collider_static_config,
                gjk_static_config=gjk_static_config,
                errno=errno,
            )
        if qd.static(collider_static_config.has_convex_specialization):
            func_narrow_phase_convex_specializations(
                geoms_init_AABB,
                dyn_state,
                collider_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
                errno,
            )
        if qd.static(collider_static_config.has_terrain):
            func_narrow_phase_any_vs_terrain(
                geoms_init_AABB,
                dyn_state,
                collider_state,
                mpr_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
                errno,
            )
        if qd.static(collider_static_config.has_nonconvex_nonterrain):
            func_narrow_phase_nonconvex_vs_nonterrain(
                geoms_init_AABB,
                dyn_state,
                collider_state,
                dyn_info,
                rigid_info,
                collider_info,
                rigid_config,
                collider_static_config,
                errno,
            )
        if qd.static(coop_dedup):
            func_clamp_prune_contacts_coop(dyn_state, collider_state, rigid_info, collider_info, errno)
        else:
            func_clamp_prune_contacts(
                dyn_state, collider_state, rigid_info, collider_info, rigid_config, collider_static_config, errno
            )
        # Plane-convex and sphere-sphere contacts come from analytic paths that leave diff_contact_input unfilled, so
        # it is filled here for the differentiable narrow-phase reverse (see func_fill_diff_contact_input_analytic)
        if qd.static(rigid_config.requires_grad):
            func_fill_diff_contact_input_analytic(dyn_state, collider_state, dyn_info, rigid_config)


@qd.kernel(fastcache=True)
def kernel_detection(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    diff_contact_input: array_class.DiffContactInput,
    contact0_mpr_state: array_class.MPRState,
    contact0_gjk_state: array_class.GJKState,
    multicontact_mpr_state: array_class.MPRState,
    multicontact_gjk_state: array_class.GJKState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    gjk_static_config: qd.template(),
    has_possible_pairs: qd.template(),
    split_narrowphase: qd.template(),
    coop_dedup: qd.template(),
    errno: qd.Tensor,
):
    """Run func_detection on its own, outside the substep graph that captures it (see func_substep_detection)."""
    func_detection(
        geoms_init_AABB=geoms_init_AABB,
        dyn_state=dyn_state,
        collider_state=collider_state,
        mpr_state=mpr_state,
        gjk_state=gjk_state,
        diff_contact_input=diff_contact_input,
        contact0_mpr_state=contact0_mpr_state,
        contact0_gjk_state=contact0_gjk_state,
        multicontact_mpr_state=multicontact_mpr_state,
        multicontact_gjk_state=multicontact_gjk_state,
        constraint_state=constraint_state,
        dyn_info=dyn_info,
        rigid_info=rigid_info,
        collider_info=collider_info,
        rigid_config=rigid_config,
        collider_static_config=collider_static_config,
        gjk_static_config=gjk_static_config,
        has_possible_pairs=has_possible_pairs,
        split_narrowphase=split_narrowphase,
        coop_dedup=coop_dedup,
        errno=errno,
    )


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.collider_decomp")
