# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Data structures of the MochiSolver: build-time model description, per-environment solver state and compile-time
configuration, all following the array_class conventions (frozen dataclasses of quadrants tensors, batch dimension
last)."""

import dataclasses
from enum import IntEnum

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, AutoInitMeta, V


class COLLIDER_TYPE(IntEnum):
    NONE = 0
    PLANE = 1
    SPHERE = 2
    BOX = 3
    GRID = 4
    POINT_CLOUD = 5


class INTEGRATOR(IntEnum):
    BACKWARD_EULER = 0
    BDF2 = 1


class FRICTION_MODEL(IntEnum):
    C1 = 0
    CINF = 1


class LINESEARCH(IntEnum):
    NONE = 0
    RESIDUAL_NORM = 1
    ARMIJO = 2


class LINEAR_TOLERANCE(IntEnum):
    CONSTANT = 0
    ADAPTIVE = 1


class SOLVE_STATUS(IntEnum):
    RUNNING = 0
    CONVERGED = 1
    STOPPED = 2
    DIVERGED = 3


# Number of previous steps kept for multistep integration (BDF2 needs two).
N_HISTORY = 2
# Batched reductions on the GPU: a 256-thread block covers 32 consecutive environments (lanes, coalesced loads) by 8
# chunks of a 64-dof tile; the chunks are summed in shared memory and each environment receives one atomic per tile.
REDUCE_LANES = 32
REDUCE_CHUNKS = 8
REDUCE_TILE = 64
REDUCE_BLOCK = REDUCE_LANES * REDUCE_CHUNKS
# Half bandwidth of a rod's Hessian in the node-interleaved ordering [x_0, theta_0, x_1, theta_1, ...]: a bending
# stencil couples three consecutive nodes and the two segments between them, i.e. rows at most 10 apart.
ROD_BAND = 10


@qd.data_oriented
class MochiStaticConfig(metaclass=AutoInitMeta):
    backend: int
    para_level: int
    integrator: int
    use_newton_euler_inertia: bool
    friction_model: int
    linesearch_type: int
    linear_tolerance: int
    use_fitted_friction_hessian: bool
    friction_with_collider_normal: bool
    fade_friction: bool
    implicit_normal_force_for_dissipation: bool
    has_dense: bool
    use_tiled_cholesky: bool
    cholesky_tile_size: int
    tiled_n_dofs: int
    has_grid_colliders: bool
    record_contacts: bool
    batch_links_info: bool
    has_soft: bool
    # tetrahedral elements present (the tetrahedron assembly is compiled only then)
    has_tets: bool
    has_equalities: bool
    # rigid-deformable vertex attachments present (their assembly is compiled only then)
    has_attachments: bool
    has_pc_colliders: bool
    # any open rod present (the banded rod preconditioner and its contact scatter are compiled only then)
    has_rod_band: bool
    has_soft_colliders: bool
    # levels of the bounding-box hierarchy of the collider tetrahedra (the refit runs one task per level)
    tet_tree_levels: int
    # bodies at rest are put to sleep (the sleep bookkeeping and the work skipping are compiled only then)
    use_sleeping: bool


# =========================================== build-time info ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiLinksInfo:
    # Whether the link carries the 6 degrees of freedom of a free root and enters the Newton system.
    is_dynamic: qd.Tensor
    has_gravity: qd.Tensor
    mass: qd.Tensor
    # Inertia about the center of mass in the inertial frame, and its second moment 0.5 tr(I) 1 - I, which weights
    # the rotation merit.
    inertia: qd.Tensor
    second_moment: qd.Tensor
    # Contact sample range and the link-frame bounding box of the sample cloud.
    sample_start: qd.Tensor
    sample_end: qd.Tensor
    samples_aabb_min: qd.Tensor
    samples_aabb_max: qd.Tensor
    # Node range of the link's contact-sample hierarchy (depth-first order, see sample_tree.py).
    tree_start: qd.Tensor
    tree_end: qd.Tensor


def get_mochi_links_info(solver):
    n_links_ = solver.n_links_
    return MochiLinksInfo(
        is_dynamic=V(dtype=gs.qd_bool, shape=(n_links_,)),
        has_gravity=V(dtype=gs.qd_bool, shape=(n_links_,)),
        mass=V(dtype=gs.qd_float, shape=(n_links_,)),
        inertia=V(dtype=gs.qd_mat3, shape=(n_links_,)),
        second_moment=V(dtype=gs.qd_mat3, shape=(n_links_,)),
        sample_start=V(dtype=gs.qd_int, shape=(n_links_,)),
        tree_start=V(dtype=gs.qd_int, shape=(n_links_,)),
        tree_end=V(dtype=gs.qd_int, shape=(n_links_,)),
        sample_end=V(dtype=gs.qd_int, shape=(n_links_,)),
        samples_aabb_min=V(dtype=gs.qd_vec3, shape=(n_links_,)),
        samples_aabb_max=V(dtype=gs.qd_vec3, shape=(n_links_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiGeomsInfo:
    collider_type: qd.Tensor
    penalty_coefficient: qd.Tensor
    penalty_smoothing_half_distance: qd.Tensor
    penalty_threshold: qd.Tensor
    friction: qd.Tensor
    friction_falloff_vel: qd.Tensor
    viscous_friction: qd.Tensor
    normal_viscous_damping: qd.Tensor
    max_alignment_normals: qd.Tensor


def get_mochi_geoms_info(solver):
    n_geoms_ = solver.n_geoms_
    return MochiGeomsInfo(
        collider_type=V(dtype=gs.qd_int, shape=(n_geoms_,)),
        penalty_coefficient=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        penalty_smoothing_half_distance=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        penalty_threshold=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        friction=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        friction_falloff_vel=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        viscous_friction=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        normal_viscous_damping=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        max_alignment_normals=V(dtype=gs.qd_float, shape=(n_geoms_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiSamplesInfo:
    # Contact sample points (surface quadrature points of the collision triangles) in the link frame, with their
    # quadrature area weight and outward normal.
    pos: qd.Tensor
    normal: qd.Tensor
    weight: qd.Tensor
    link_idx: qd.Tensor
    geom_idx: qd.Tensor
    # Bounding-sphere hierarchy of every link's samples (link frame), nodes in depth-first order: center and radius,
    # the contiguous sample range a node bounds, the depth-first index of the next node outside its subtree, and
    # whether it is a leaf (see sample_tree.py).
    tree_center: qd.Tensor
    tree_radius: qd.Tensor
    tree_first: qd.Tensor
    tree_count: qd.Tensor
    tree_escape: qd.Tensor
    tree_is_leaf: qd.Tensor


def get_mochi_samples_info(solver):
    n_samples_ = solver.n_samples_
    n_nodes_ = solver.n_tree_nodes_
    return MochiSamplesInfo(
        pos=V(dtype=gs.qd_vec3, shape=(n_samples_,)),
        normal=V(dtype=gs.qd_vec3, shape=(n_samples_,)),
        weight=V(dtype=gs.qd_float, shape=(n_samples_,)),
        link_idx=V(dtype=gs.qd_int, shape=(n_samples_,)),
        geom_idx=V(dtype=gs.qd_int, shape=(n_samples_,)),
        tree_center=V(dtype=gs.qd_vec3, shape=(n_nodes_,)),
        tree_radius=V(dtype=gs.qd_float, shape=(n_nodes_,)),
        tree_first=V(dtype=gs.qd_int, shape=(n_nodes_,)),
        tree_count=V(dtype=gs.qd_int, shape=(n_nodes_,)),
        tree_escape=V(dtype=gs.qd_int, shape=(n_nodes_,)),
        tree_is_leaf=V(dtype=gs.qd_int, shape=(n_nodes_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiInfo:
    links: MochiLinksInfo
    geoms: MochiGeomsInfo
    samples: MochiSamplesInfo
    # Whether contact between two links is enabled (layer and entity filters folded in).
    links_pair_enabled: qd.Tensor
    # Total mass of the entity owning each degree of freedom, scaling its convergence weight.
    dofs_entity_mass: qd.Tensor
    # Runtime constants
    dt: qd.Tensor
    joint_limit_stiffness: qd.Tensor
    joint_limit_damping: qd.Tensor
    equality_stiffness: qd.Tensor
    equality_damping: qd.Tensor
    gravity: qd.Tensor
    broadphase_margin: qd.Tensor
    newton_abs_tol: qd.Tensor
    newton_rel_tol: qd.Tensor
    explosion_abs_tol: qd.Tensor
    explosion_rel_tol: qd.Tensor
    linesearch_alpha: qd.Tensor
    linesearch_wolfe1: qd.Tensor
    pcg_rel_tol: qd.Tensor
    pcg_abs_tol: qd.Tensor
    n_newton_iterations: qd.Tensor
    # Sleeping of bodies at rest (see MochiOptions.use_sleeping).
    sleep_threshold: qd.Tensor
    sleep_min_steps: qd.Tensor
    sleep_max_speed: qd.Tensor
    EPS: qd.Tensor


def get_mochi_info(solver):
    options = solver._options
    return MochiInfo(
        links=get_mochi_links_info(solver),
        geoms=get_mochi_geoms_info(solver),
        samples=get_mochi_samples_info(solver),
        links_pair_enabled=V(dtype=gs.qd_bool, shape=(solver.n_links_, solver.n_links_)),
        dofs_entity_mass=V(dtype=gs.qd_float, shape=(solver.n_dofs_total_,)),
        dt=_scalar(gs.qd_float, solver._substep_dt),
        joint_limit_stiffness=_scalar(gs.qd_float, options.joint_limit_stiffness),
        joint_limit_damping=_scalar(gs.qd_float, options.joint_limit_damping),
        equality_stiffness=_scalar(gs.qd_float, options.equality_stiffness),
        equality_damping=_scalar(gs.qd_float, options.equality_damping),
        gravity=V(dtype=gs.qd_vec3, shape=(solver._B,)),
        broadphase_margin=_scalar(gs.qd_float, options.broadphase_margin),
        newton_abs_tol=_scalar(gs.qd_float, options.newton_abs_tol),
        newton_rel_tol=_scalar(gs.qd_float, options.newton_rel_tol),
        explosion_abs_tol=_scalar(gs.qd_float, options.explosion_abs_tol if options.explosion_control else 0.0),
        explosion_rel_tol=_scalar(gs.qd_float, options.explosion_rel_tol),
        linesearch_alpha=_scalar(gs.qd_float, options.linesearch_alpha),
        linesearch_wolfe1=_scalar(gs.qd_float, options.linesearch_wolfe1),
        pcg_rel_tol=_scalar(gs.qd_float, options.pcg_rel_tol),
        pcg_abs_tol=_scalar(gs.qd_float, options.pcg_abs_tol),
        n_newton_iterations=_scalar(gs.qd_int, options.n_newton_iterations),
        sleep_threshold=_scalar(gs.qd_float, options.sleep_threshold),
        sleep_min_steps=_scalar(gs.qd_int, options.sleep_min_steps),
        sleep_max_speed=_scalar(gs.qd_float, options.sleep_max_speed),
        EPS=_scalar(gs.qd_float, gs.EPS),
    )


def _scalar(dtype, value):
    data = V(dtype=dtype, shape=())
    data.fill(value)
    return data


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiIslandState:
    # Island nodes are the rigid entities (all links of an articulation move together) followed by the deformable
    # entities. Build-time maps from links and degrees of freedom to nodes (-1 for links without dofs).
    links_node: qd.Tensor
    dofs_node: qd.Tensor
    # Per-environment union-find forest over the nodes, compact island index of every node and island count.
    nodes_parent: qd.Tensor
    nodes_island: qd.Tensor
    n_islands: qd.Tensor
    # Degrees of freedom grouped by island: dofs of island k are island_dofs[island_start[k]:island_start[k + 1]].
    island_start: qd.Tensor
    island_n_dofs: qd.Tensor
    island_dofs: qd.Tensor
    dofs_island: qd.Tensor
    island_max_dofs: qd.Tensor
    # Whether the environment is solved by the island-wise direct solver (largest island within the dense limit).
    uses_dense: qd.Tensor
    # Weighted squared residual norm of every node (entity) at the current and the initial Newton iterate; an
    # environment converges when every one of its entities does, as in mochi.
    nodes_res_w_sq: qd.Tensor
    nodes_res_norm0_w: qd.Tensor
    # Sleeping (allocated only when enabled): consecutive steps at rest of every node, whether it sleeps, whether it
    # moved too fast or is actuated this step, and whether an island holds an awake node (scratch of the island-wise
    # decisions, indexed by island).
    nodes_rest_steps: qd.Tensor
    nodes_is_asleep: qd.Tensor
    nodes_is_restless: qd.Tensor
    islands_is_awake: qd.Tensor


def get_mochi_island_state(solver, links_node, dofs_node):
    _B = solver._B
    n_nodes_ = max(1, len(solver._entities) + len(solver._soft_entities))
    n_dofs_ = solver.n_dofs_total_
    n_sleep_nodes_, _B_sleep = (n_nodes_, _B) if solver._options.use_sleeping else (1, 1)
    state = MochiIslandState(
        links_node=V(dtype=gs.qd_int, shape=(solver.n_links_,)),
        dofs_node=V(dtype=gs.qd_int, shape=(n_dofs_,)),
        nodes_parent=V(dtype=gs.qd_int, shape=(n_nodes_, _B)),
        nodes_island=V(dtype=gs.qd_int, shape=(n_nodes_, _B)),
        n_islands=V(dtype=gs.qd_int, shape=(_B,)),
        island_start=V(dtype=gs.qd_int, shape=(n_nodes_ + 1, _B)),
        island_n_dofs=V(dtype=gs.qd_int, shape=(n_nodes_, _B)),
        island_dofs=V(dtype=gs.qd_int, shape=(n_dofs_, _B)),
        dofs_island=V(dtype=gs.qd_int, shape=(n_dofs_, _B)),
        island_max_dofs=V(dtype=gs.qd_int, shape=(_B,)),
        uses_dense=V(dtype=gs.qd_bool, shape=(_B,)),
        nodes_res_w_sq=V(dtype=gs.qd_float, shape=(n_nodes_, _B)),
        nodes_res_norm0_w=V(dtype=gs.qd_float, shape=(n_nodes_, _B)),
        nodes_rest_steps=V(dtype=gs.qd_int, shape=(n_sleep_nodes_, _B_sleep)),
        nodes_is_asleep=V(dtype=gs.qd_bool, shape=(n_sleep_nodes_, _B_sleep)),
        nodes_is_restless=V(dtype=gs.qd_bool, shape=(n_sleep_nodes_, _B_sleep)),
        islands_is_awake=V(dtype=gs.qd_bool, shape=(n_sleep_nodes_, _B_sleep)),
    )
    if len(links_node) > 0:
        state.links_node.from_numpy(np.asarray(links_node, dtype=gs.np_int))
    if len(dofs_node) > 0:
        state.dofs_node.from_numpy(np.asarray(dofs_node, dtype=gs.np_int))
    return state


# =========================================== runtime state ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiState:
    # Multistep history of the generalized coordinates and velocities, and of the finite-difference link velocities
    # (translational, angular, and the symmetric rotation-derivative correction): slot 0 is the previous step, slot 1
    # the one before.
    qpos_prev: qd.Tensor
    dofs_vel_prev: qd.Tensor
    links_vel_prev: qd.Tensor
    links_ang_prev: qd.Tensor
    links_vsym_prev: qd.Tensor
    # Finite-difference velocity of every link at the end of the last step. Together with the angular velocity, the
    # symmetric part of the rotation derivative reproduces the rotation increment of that step exactly, which the
    # rotation merit of the next step extrapolates from.
    links_vel: qd.Tensor
    links_ang: qd.Tensor
    links_vsym: qd.Tensor
    # Step-start extrapolation and stage-start reference of the current solve (identical for single-stage schemes).
    qpos_step_start: qd.Tensor
    qpos_stage_start: qd.Tensor
    dofs_vel_stage_start: qd.Tensor
    links_vel_stage_start: qd.Tensor
    links_ang_stage_start: qd.Tensor
    links_vsym_stage_start: qd.Tensor
    links_pos_stage_start: qd.Tensor
    links_quat_stage_start: qd.Tensor
    geoms_pos_stage_start: qd.Tensor
    geoms_quat_stage_start: qd.Tensor
    # Reference iterate of the line search (the last accepted iterate).
    qpos_ls_ref: qd.Tensor
    # Number of valid previous steps (multistep schemes fall back to backward Euler while it is short) and the stage
    # time step of the current solve.
    n_hist: qd.Tensor
    dt_stage: qd.Tensor
    # Newton system. Link-space accumulators (residual 6-vector, diagonal 6x6 block per link, off-diagonal 6x6 block
    # per contact pair) are projected onto the degrees of freedom through the link Jacobians; joint-space terms add
    # to the residual and to the diagonal of the projected Hessian directly.
    links_res: qd.Tensor
    res: qd.Tensor
    dx: qd.Tensor
    conv_w: qd.Tensor
    dofs_H_diag: qd.Tensor
    H_diag: qd.Tensor
    H_off: qd.Tensor
    H_dense: qd.Tensor
    # Per-environment Newton and line search control.
    is_active: qd.Tensor
    status: qd.Tensor
    n_iter: qd.Tensor
    n_pcg_iter: qd.Tensor
    # every environment index in order, and the batch size: the identity environment list of the step functions
    all_envs: qd.Tensor
    n_envs_all: qd.Tensor
    # control of the graph step kernel: kind of the current round and the environment-list gates of its phases
    graph_is_first: qd.Tensor
    graph_round_is_s: qd.Tensor
    graph_round_is_l: qd.Tensor
    graph_round_is_last_trial: qd.Tensor
    graph_any: qd.Tensor
    gate_ls: qd.Tensor
    gate_newton: qd.Tensor
    gate_first: qd.Tensor
    gate_post_ls: qd.Tensor
    gate_pcg: qd.Tensor
    res_norm_sq: qd.Tensor
    res_w_sq: qd.Tensor
    res_norm0: qd.Tensor
    ls_alpha: qd.Tensor
    ls_ref_norm_sq: qd.Tensor
    ls_is_done: qd.Tensor
    ls_slope: qd.Tensor
    obj: qd.Tensor
    obj_ref: qd.Tensor
    # Residual norm of the previous Newton iterate, driving the adaptive linear tolerance.
    res_norm_prev: qd.Tensor
    # Preconditioned conjugate gradient scratch. The stopping criterion monitors the preconditioned residual
    # z = M^-1 r, whose squared norm starts at pcg_zTz0; pcg_rTz_cross holds r_new . z_old for the Polak-Ribiere beta.
    pcg_rel_tol: qd.Tensor
    pcg_diag: qd.Tensor
    pcg_r: qd.Tensor
    pcg_z: qd.Tensor
    pcg_p: qd.Tensor
    pcg_Ap: qd.Tensor
    pcg_rTz: qd.Tensor
    pcg_rTz_new: qd.Tensor
    pcg_rTz_cross: qd.Tensor
    pcg_pTAp: qd.Tensor
    pcg_beta: qd.Tensor
    pcg_zTz: qd.Tensor
    pcg_zTz0: qd.Tensor
    pcg_is_active: qd.Tensor
    # Whether the node of every link sleeps (allocated only when sleeping is enabled, see MochiIslandState).
    links_is_asleep: qd.Tensor


def get_mochi_state(solver, max_pairs, has_dense):
    _B = solver._B
    n_qs_, n_links_, n_geoms_ = solver.n_qs_, solver.n_links_, solver.n_geoms_
    # Rigid degrees of freedom of the kinematic tree, then 3 per vertex of the deformable bodies.
    n_dofs_rigid_, n_dofs_ = solver.n_dofs_, solver.n_dofs_total_
    H_dense_shape = (_B, n_dofs_, n_dofs_) if has_dense else ()
    return MochiState(
        qpos_prev=V(dtype=gs.qd_float, shape=(N_HISTORY, n_qs_, _B)),
        dofs_vel_prev=V(dtype=gs.qd_float, shape=(N_HISTORY, n_dofs_rigid_, _B)),
        links_vel_prev=V(dtype=gs.qd_vec3, shape=(N_HISTORY, n_links_, _B)),
        links_ang_prev=V(dtype=gs.qd_vec3, shape=(N_HISTORY, n_links_, _B)),
        links_vsym_prev=V(dtype=gs.qd_mat3, shape=(N_HISTORY, n_links_, _B)),
        links_vel=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_ang=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_vsym=V(dtype=gs.qd_mat3, shape=(n_links_, _B)),
        qpos_step_start=V(dtype=gs.qd_float, shape=(n_qs_, _B)),
        qpos_stage_start=V(dtype=gs.qd_float, shape=(n_qs_, _B)),
        dofs_vel_stage_start=V(dtype=gs.qd_float, shape=(n_dofs_rigid_, _B)),
        links_vel_stage_start=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_ang_stage_start=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_vsym_stage_start=V(dtype=gs.qd_mat3, shape=(n_links_, _B)),
        links_pos_stage_start=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_quat_stage_start=V(dtype=gs.qd_vec4, shape=(n_links_, _B)),
        geoms_pos_stage_start=V(dtype=gs.qd_vec3, shape=(n_geoms_, _B)),
        geoms_quat_stage_start=V(dtype=gs.qd_vec4, shape=(n_geoms_, _B)),
        qpos_ls_ref=V(dtype=gs.qd_float, shape=(n_qs_, _B)),
        n_hist=V(dtype=gs.qd_int, shape=(_B,)),
        dt_stage=V(dtype=gs.qd_float, shape=(_B,)),
        links_res=V(dtype=gs.qd_vec6, shape=(n_links_, _B)),
        res=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        dx=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        conv_w=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        dofs_H_diag=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        H_diag=V_MAT(n=6, m=6, dtype=gs.qd_float, shape=(n_links_, _B)),
        H_off=V_MAT(n=6, m=6, dtype=gs.qd_float, shape=(max_pairs, _B)),
        H_dense=V(dtype=gs.qd_float, shape=H_dense_shape),
        is_active=V(dtype=gs.qd_bool, shape=(_B,)),
        status=V(dtype=gs.qd_int, shape=(_B,)),
        n_iter=V(dtype=gs.qd_int, shape=(_B,)),
        n_pcg_iter=V(dtype=gs.qd_int, shape=(_B,)),
        all_envs=V(dtype=gs.qd_int, shape=(_B,)),
        n_envs_all=_scalar(gs.qd_int, _B),
        graph_is_first=_scalar(gs.qd_int, 0),
        graph_round_is_s=_scalar(gs.qd_int, 0),
        graph_round_is_l=_scalar(gs.qd_int, 0),
        graph_round_is_last_trial=_scalar(gs.qd_int, 0),
        graph_any=_scalar(gs.qd_int, 0),
        gate_ls=_scalar(gs.qd_int, 0),
        gate_newton=_scalar(gs.qd_int, 0),
        gate_first=_scalar(gs.qd_int, 0),
        gate_post_ls=_scalar(gs.qd_int, 0),
        gate_pcg=_scalar(gs.qd_int, 0),
        res_norm_sq=V(dtype=gs.qd_float, shape=(_B,)),
        res_w_sq=V(dtype=gs.qd_float, shape=(_B,)),
        res_norm0=V(dtype=gs.qd_float, shape=(_B,)),
        ls_alpha=V(dtype=gs.qd_float, shape=(_B,)),
        ls_ref_norm_sq=V(dtype=gs.qd_float, shape=(_B,)),
        ls_is_done=V(dtype=gs.qd_bool, shape=(_B,)),
        ls_slope=V(dtype=gs.qd_float, shape=(_B,)),
        obj=V(dtype=gs.qd_float, shape=(_B,)),
        obj_ref=V(dtype=gs.qd_float, shape=(_B,)),
        res_norm_prev=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_rel_tol=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_diag=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_r=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_z=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_p=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_Ap=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_rTz=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_rTz_new=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_rTz_cross=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_pTAp=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_beta=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_zTz=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_zTz0=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_is_active=V(dtype=gs.qd_bool, shape=(_B,)),
        links_is_asleep=V(dtype=gs.qd_bool, shape=(n_links_, _B) if solver._options.use_sleeping else (1, 1)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiContactState:
    # Candidate (colliding link, collider geom) pairs of the current step and their per-pair accumulators: total
    # force, torque about the collider link origin, and the three 3x3 sums the rigid-rigid Hessian blocks are built
    # from (see kernel_pairs_to_blocks).
    n_pairs: qd.Tensor
    pair_link_a: qd.Tensor
    pair_link_b: qd.Tensor
    pair_geom_b: qd.Tensor
    acc_f: qd.Tensor
    acc_q: qd.Tensor
    acc_D: qd.Tensor
    acc_SD: qd.Tensor
    acc_SDS: qd.Tensor
    acc_obj: qd.Tensor
    n_hits: qd.Tensor
    # Conservative per-step world bounds of every link's sample cloud, and the motion padding of every link.
    links_step_aabb_min: qd.Tensor
    links_step_aabb_max: qd.Tensor
    links_step_pad: qd.Tensor


def get_mochi_contact_state(solver, max_pairs):
    _B = solver._B
    n_links_ = solver.n_links_
    return MochiContactState(
        n_pairs=V(dtype=gs.qd_int, shape=(_B,)),
        pair_link_a=V(dtype=gs.qd_int, shape=(max_pairs, _B)),
        pair_link_b=V(dtype=gs.qd_int, shape=(max_pairs, _B)),
        pair_geom_b=V(dtype=gs.qd_int, shape=(max_pairs, _B)),
        acc_f=V(dtype=gs.qd_vec3, shape=(max_pairs, _B)),
        acc_q=V(dtype=gs.qd_vec3, shape=(max_pairs, _B)),
        acc_D=V(dtype=gs.qd_vec6, shape=(max_pairs, _B)),
        acc_SD=V(dtype=gs.qd_mat3, shape=(max_pairs, _B)),
        acc_SDS=V(dtype=gs.qd_vec6, shape=(max_pairs, _B)),
        acc_obj=V(dtype=gs.qd_float, shape=(max_pairs, _B)),
        n_hits=V(dtype=gs.qd_int, shape=(max_pairs, _B)),
        links_step_aabb_min=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_step_aabb_max=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_step_pad=V(dtype=gs.qd_float, shape=(n_links_, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiHitReadback:
    """Contact points recorded for readback (positions, normals, forces, distances of every hit of the last recording
    pass, per contact kind), allocated at the first readback: the solver itself only keeps the fields its linear
    solve needs."""

    # rigid samples on rigid colliders
    n_hits_total: qd.Tensor
    hit_link_a: qd.Tensor
    hit_geom_a: qd.Tensor
    hit_link_b: qd.Tensor
    hit_geom_b: qd.Tensor
    hit_sample: qd.Tensor
    hit_pos: qd.Tensor
    hit_normal: qd.Tensor
    hit_force: qd.Tensor
    hit_distance: qd.Tensor
    hit_weight: qd.Tensor
    # deformable samples on rigid colliders
    soft_hit_geom_b: qd.Tensor
    soft_hit_pos: qd.Tensor
    soft_hit_normal: qd.Tensor
    soft_hit_force: qd.Tensor
    soft_hit_distance: qd.Tensor
    # samples on deformable (tetrahedral) colliders
    sc_hit_pos: qd.Tensor
    sc_hit_normal: qd.Tensor
    sc_hit_force: qd.Tensor
    sc_hit_distance: qd.Tensor
    # samples on point-cloud colliders
    pc_hit_pos: qd.Tensor
    pc_hit_normal: qd.Tensor
    pc_hit_force: qd.Tensor
    pc_hit_distance: qd.Tensor


def get_mochi_hit_readback(solver, max_hits, max_soft_hits, max_sc_hits, max_pc_hits):
    _B = solver._B
    return MochiHitReadback(
        n_hits_total=V(dtype=gs.qd_int, shape=(_B,)),
        hit_link_a=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_geom_a=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_link_b=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_geom_b=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_sample=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_pos=V(dtype=gs.qd_vec3, shape=(max_hits, _B)),
        hit_normal=V(dtype=gs.qd_vec3, shape=(max_hits, _B)),
        hit_force=V(dtype=gs.qd_vec3, shape=(max_hits, _B)),
        hit_distance=V(dtype=gs.qd_float, shape=(max_hits, _B)),
        hit_weight=V(dtype=gs.qd_float, shape=(max_hits, _B)),
        soft_hit_geom_b=V(dtype=gs.qd_int, shape=(max_soft_hits, _B)),
        soft_hit_pos=V(dtype=gs.qd_vec3, shape=(max_soft_hits, _B)),
        soft_hit_normal=V(dtype=gs.qd_vec3, shape=(max_soft_hits, _B)),
        soft_hit_force=V(dtype=gs.qd_vec3, shape=(max_soft_hits, _B)),
        soft_hit_distance=V(dtype=gs.qd_float, shape=(max_soft_hits, _B)),
        sc_hit_pos=V(dtype=gs.qd_vec3, shape=(max_sc_hits, _B)),
        sc_hit_normal=V(dtype=gs.qd_vec3, shape=(max_sc_hits, _B)),
        sc_hit_force=V(dtype=gs.qd_vec3, shape=(max_sc_hits, _B)),
        sc_hit_distance=V(dtype=gs.qd_float, shape=(max_sc_hits, _B)),
        pc_hit_pos=V(dtype=gs.qd_vec3, shape=(max_pc_hits, _B)),
        pc_hit_normal=V(dtype=gs.qd_vec3, shape=(max_pc_hits, _B)),
        pc_hit_force=V(dtype=gs.qd_vec3, shape=(max_pc_hits, _B)),
        pc_hit_distance=V(dtype=gs.qd_float, shape=(max_pc_hits, _B)),
    )


# =========================================== deformable bodies ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiSoftInfo:
    # Vertices: rest position, row-sum lumped mass (convergence weights), owning entity and point-cloud collider weight
    # (nodal area of shell vertices, zero elsewhere).
    verts_rest: qd.Tensor
    verts_mass: qd.Tensor
    verts_entity_idx: qd.Tensor
    verts_collider_weight: qd.Tensor
    # Tetrahedra: vertex indices, rest edge matrix and its inverse (node 3 as origin), rest volume and owning entity.
    elems_v: qd.Tensor
    elems_Dm: qd.Tensor
    elems_Dm_inv: qd.Tensor
    elems_vol: qd.Tensor
    elems_entity_idx: qd.Tensor
    # Shell triangles: vertices, opposite vertices of the neighboring triangles across the three edges (-1 across a
    # boundary edge), owning entity, rest area, inverse rest metric and rest second fundamental form.
    shell_elems_v: qd.Tensor
    shell_elems_hinge: qd.Tensor
    shell_elems_entity_idx: qd.Tensor
    shell_elems_area: qd.Tensor
    shell_elems_A_inv: qd.Tensor
    shell_elems_B: qd.Tensor
    # Rod segments: the two vertices, owning entity, reference length, rotational inertia (linear density times length),
    # reference material axis; and the interior-node stencils (three vertices, the two segments meeting at the node,
    # the reference Voronoi length and the reference curvature/twist measures).
    rod_elems_v: qd.Tensor
    rod_elems_entity_idx: qd.Tensor
    rod_elems_L: qd.Tensor
    rod_elems_rot_inertia: qd.Tensor
    rod_elems_axis_ref: qd.Tensor
    rod_stencils_v: qd.Tensor
    rod_stencils_e: qd.Tensor
    rod_stencils_L: qd.Tensor
    rod_stencils_ref: qd.Tensor
    # Smallest positive normal number of the simulation precision, flooring every rod length, tangent norm and
    # parallel-transport denominator. Resolved here rather than as a module constant because the value depends on the
    # precision selected at initialization.
    rod_tiny: qd.Tensor
    # Banded ordering of the open rods for the exact per-rod preconditioner: band row of every degree of freedom (-1
    # outside the rods), degree of freedom of every band row, and per deformable entity the row range plus the rod
    # element and stencil ranges (all zero-length for anything but an open rod).
    # Sparsity of the deformable Hessian over the deformable degrees of freedom (vertex dofs 3 i_v + k, then the rod
    # twist dofs), as a scalar CSR: row starts, columns, and the CSR index of every entry of every element block so the
    # assembly kernels scatter straight into the values (-1 for the entries of a missing shell hinge vertex).
    csr_start: qd.Tensor
    csr_col: qd.Tensor
    # position, in the shared column sequence of the rows of vertex f, of the first column of vertex g's block, per
    # element block (f, g); the scalar CSR index of entry (3 f + r, 3 g + c) is csr_start[3 f + r] + position + c
    elems_csr_block: qd.Tensor
    shell_csr_block: qd.Tensor
    rod_elems_csr: qd.Tensor
    rod_stencils_csr: qd.Tensor
    dofs_band_row: qd.Tensor
    band_rows_dof: qd.Tensor
    band_rows_entity: qd.Tensor
    entities_band_start: qd.Tensor
    entities_band_n: qd.Tensor
    entities_rod_elem_start: qd.Tensor
    entities_rod_elem_end: qd.Tensor
    entities_rod_stencil_start: qd.Tensor
    entities_rod_stencil_end: qd.Tensor
    # Contact samples: vertices and their weights (three for a boundary triangle, four for a contact-skin sample, which
    # takes the tetrahedron it is embedded in, the others then padding with a zero weight), rest area weight and owning
    # entity. The fourth vertex is allocated only when some entity has a contact skin.
    samples_verts: qd.Tensor
    samples_bary: qd.Tensor
    samples_weight: qd.Tensor
    samples_entity_idx: qd.Tensor
    # Contact-skin samples: the row of every sample in the skin tables (-1 off the skin), the sample of every row and
    # its rest normal pulled back to the edge frame of its tetrahedron (see build_skin_samples); n_skin_samples
    # bounds the loops (the arrays hold at least one row).
    samples_skin_row: qd.Tensor
    skin_samples: qd.Tensor
    skin_normal_rest: qd.Tensor
    n_skin_samples: qd.Tensor
    # Rigid-deformable attachments: the attached vertex, its link, the anchor in the link frame and the penalty
    # stiffness and damping of every attachment; n_attachments bounds the loops (the arrays hold at least one row).
    att_vert: qd.Tensor
    att_link: qd.Tensor
    att_link_is_dynamic: qd.Tensor
    att_pos_local: qd.Tensor
    att_stiffness: qd.Tensor
    att_damping: qd.Tensor
    n_attachments: qd.Tensor
    # Per deformable entity: kind (0 solid, 1 shell), mass, material, contact parameters, vertex and sample ranges.
    entities_kind: qd.Tensor
    entities_mass: qd.Tensor
    entities_has_gravity: qd.Tensor
    entities_model: qd.Tensor
    entities_mu: qd.Tensor
    entities_lam: qd.Tensor
    entities_rho: qd.Tensor
    entities_mass_damping: qd.Tensor
    entities_stiffness_damping: qd.Tensor
    entities_membrane_mu: qd.Tensor
    entities_membrane_lambda: qd.Tensor
    entities_bending_alpha: qd.Tensor
    entities_bending_beta: qd.Tensor
    entities_collider_radius: qd.Tensor
    entities_axial_stiffness: qd.Tensor
    entities_torsional_stiffness: qd.Tensor
    entities_rot_inertia: qd.Tensor
    entities_self_contact: qd.Tensor
    entities_self_contact_exclusion_ratio: qd.Tensor
    # whether the samples of the entity can hit the tetrahedra / the collider spheres of some entity at all
    entities_queries_tets: qd.Tensor
    entities_queries_spheres: qd.Tensor
    entities_penalty_coefficient: qd.Tensor
    entities_penalty_smoothing_half_distance: qd.Tensor
    entities_penalty_threshold: qd.Tensor
    entities_friction: qd.Tensor
    entities_friction_falloff_vel: qd.Tensor
    entities_viscous_friction: qd.Tensor
    entities_normal_viscous_damping: qd.Tensor
    entities_max_alignment_normals: qd.Tensor
    entities_vert_start: qd.Tensor
    entities_vert_end: qd.Tensor
    entities_sample_start: qd.Tensor
    entities_sample_end: qd.Tensor
    # Whether the entity acts as a collider, and its rest-shape signed distance grid: offset into the flattened voxel
    # array, resolution, position of the first voxel and cell size per axis (rest frame == world frame at build).
    entities_collider_type: qd.Tensor
    entities_sdf_start: qd.Tensor
    entities_sdf_res: qd.Tensor
    entities_sdf_origin: qd.Tensor
    entities_sdf_cell: qd.Tensor
    sdf_values: qd.Tensor
    # Whether contact between a deformable entity and a rigid link, or between two deformable entities, is enabled
    # (layer and entity filters).
    entities_links_pair_enabled: qd.Tensor
    entities_pair_enabled: qd.Tensor
    # Offset of the first deformable degree of freedom in the Newton system (after the rigid degrees of freedom), and
    # of the first rod twist degree of freedom (after the vertex degrees of freedom).
    dof_start: qd.Tensor
    twist_dof_start: qd.Tensor
    n_rigid_queries: qd.Tensor
    n_queries: qd.Tensor
    pc_hash_cell: qd.Tensor
    # bounding-box hierarchy of the collider tetrahedra (see tet_tree.py): per node its first leaf-ordered element and
    # count, the depth-first index of the next node outside its subtree and whether it is a leaf; the nodes listed
    # from the deepest level up (with the start of every level) for the refit; the leaf-ordered element indices; a
    # test hook that makes every query visit every node
    tet_tree_first: qd.Tensor
    tet_tree_count: qd.Tensor
    tet_tree_escape: qd.Tensor
    tet_tree_is_leaf: qd.Tensor
    tet_tree_level_nodes: qd.Tensor
    tet_tree_level_start: qd.Tensor
    tet_tree_elems: qd.Tensor
    tet_tree_brute_force: qd.Tensor


def get_mochi_soft_info(solver):
    n_sv_, n_el_, n_ss_, n_se_ = (
        solver.n_soft_verts_,
        solver.n_soft_elems_,
        solver.n_soft_samples_,
        solver.n_soft_entities_,
    )
    n_sh_ = solver.n_shell_elems_
    n_re_, n_rs_ = solver.n_rod_elems_, solver.n_rod_stencils_
    n_att_ = solver.n_attachments_
    has_skins = solver.n_skin_samples > 0
    return MochiSoftInfo(
        verts_rest=V(dtype=gs.qd_vec3, shape=(n_sv_,)),
        verts_mass=V(dtype=gs.qd_float, shape=(n_sv_,)),
        verts_entity_idx=V(dtype=gs.qd_int, shape=(n_sv_,)),
        verts_collider_weight=V(dtype=gs.qd_float, shape=(n_sv_,)),
        elems_v=V(dtype=gs.qd_ivec4, shape=(n_el_,)),
        elems_Dm=V(dtype=gs.qd_mat3, shape=(n_el_,)),
        elems_Dm_inv=V(dtype=gs.qd_mat3, shape=(n_el_,)),
        elems_vol=V(dtype=gs.qd_float, shape=(n_el_,)),
        elems_entity_idx=V(dtype=gs.qd_int, shape=(n_el_,)),
        shell_elems_v=V(dtype=gs.qd_ivec3, shape=(n_sh_,)),
        shell_elems_hinge=V(dtype=gs.qd_ivec3, shape=(n_sh_,)),
        shell_elems_entity_idx=V(dtype=gs.qd_int, shape=(n_sh_,)),
        shell_elems_area=V(dtype=gs.qd_float, shape=(n_sh_,)),
        shell_elems_A_inv=V_MAT(n=2, m=2, dtype=gs.qd_float, shape=(n_sh_,)),
        shell_elems_B=V_MAT(n=2, m=2, dtype=gs.qd_float, shape=(n_sh_,)),
        rod_elems_v=V(dtype=gs.qd_ivec2, shape=(n_re_,)),
        rod_elems_entity_idx=V(dtype=gs.qd_int, shape=(n_re_,)),
        rod_elems_L=V(dtype=gs.qd_float, shape=(n_re_,)),
        rod_elems_rot_inertia=V(dtype=gs.qd_float, shape=(n_re_,)),
        rod_elems_axis_ref=V(dtype=gs.qd_vec3, shape=(n_re_,)),
        rod_stencils_v=V(dtype=gs.qd_ivec3, shape=(n_rs_,)),
        rod_stencils_e=V(dtype=gs.qd_ivec2, shape=(n_rs_,)),
        rod_stencils_L=V(dtype=gs.qd_float, shape=(n_rs_,)),
        rod_stencils_ref=V(dtype=gs.qd_vec3, shape=(n_rs_,)),
        rod_tiny=_scalar(gs.qd_float, float(np.finfo(gs.np_float).tiny)),
        csr_start=V(dtype=gs.qd_int, shape=(solver.n_soft_dofs_ + 1,)),
        csr_col=V(dtype=gs.qd_int, shape=(solver.n_csr_,)),
        elems_csr_block=V(dtype=gs.qd_int, shape=(n_el_, 16)),
        shell_csr_block=V(dtype=gs.qd_int, shape=(solver.n_shell_elems_, 36)),
        rod_elems_csr=V(dtype=gs.qd_int, shape=(n_re_, 36)),
        rod_stencils_csr=V(dtype=gs.qd_int, shape=(n_rs_, 121)),
        dofs_band_row=V(dtype=gs.qd_int, shape=(solver.n_dofs_total_,)),
        band_rows_dof=V(dtype=gs.qd_int, shape=(solver.n_band_rows_,)),
        band_rows_entity=V(dtype=gs.qd_int, shape=(solver.n_band_rows_,)),
        entities_band_start=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_band_n=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_rod_elem_start=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_rod_elem_end=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_rod_stencil_start=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_rod_stencil_end=V(dtype=gs.qd_int, shape=(n_se_,)),
        samples_verts=V(dtype=gs.qd_ivec4 if has_skins else gs.qd_ivec3, shape=(n_ss_,)),
        samples_bary=V(dtype=gs.qd_vec4 if has_skins else gs.qd_vec3, shape=(n_ss_,)),
        samples_weight=V(dtype=gs.qd_float, shape=(n_ss_,)),
        samples_entity_idx=V(dtype=gs.qd_int, shape=(n_ss_,)),
        samples_skin_row=V(dtype=gs.qd_int, shape=(n_ss_ if has_skins else 1,)),
        skin_samples=V(dtype=gs.qd_int, shape=(solver.n_skin_samples_,)),
        skin_normal_rest=V(dtype=gs.qd_vec3, shape=(solver.n_skin_samples_,)),
        n_skin_samples=_scalar(gs.qd_int, solver.n_skin_samples),
        entities_kind=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_mass=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_has_gravity=V(dtype=gs.qd_bool, shape=(n_se_,)),
        entities_model=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_mu=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_lam=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_rho=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_mass_damping=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_stiffness_damping=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_membrane_mu=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_membrane_lambda=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_bending_alpha=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_bending_beta=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_collider_radius=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_axial_stiffness=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_torsional_stiffness=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_rot_inertia=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_self_contact=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_self_contact_exclusion_ratio=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_queries_tets=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_queries_spheres=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_penalty_coefficient=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_penalty_smoothing_half_distance=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_penalty_threshold=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_friction=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_friction_falloff_vel=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_viscous_friction=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_normal_viscous_damping=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_max_alignment_normals=V(dtype=gs.qd_float, shape=(n_se_,)),
        entities_vert_start=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_vert_end=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_sample_start=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_sample_end=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_collider_type=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_sdf_start=V(dtype=gs.qd_int, shape=(n_se_,)),
        entities_sdf_res=V(dtype=gs.qd_ivec3, shape=(n_se_,)),
        entities_sdf_origin=V(dtype=gs.qd_vec3, shape=(n_se_,)),
        entities_sdf_cell=V(dtype=gs.qd_vec3, shape=(n_se_,)),
        sdf_values=V(dtype=gs.qd_float, shape=(solver.n_soft_sdf_voxels_,)),
        entities_links_pair_enabled=V(dtype=gs.qd_bool, shape=(n_se_, solver.n_links_)),
        entities_pair_enabled=V(dtype=gs.qd_bool, shape=(n_se_, n_se_)),
        att_vert=V(dtype=gs.qd_int, shape=(n_att_,)),
        att_link=V(dtype=gs.qd_int, shape=(n_att_,)),
        att_link_is_dynamic=V(dtype=gs.qd_int, shape=(n_att_,)),
        att_pos_local=V(dtype=gs.qd_vec3, shape=(n_att_,)),
        att_stiffness=V(dtype=gs.qd_float, shape=(n_att_,)),
        att_damping=V(dtype=gs.qd_float, shape=(n_att_,)),
        n_attachments=_scalar(gs.qd_int, solver.n_attachments),
        dof_start=_scalar(gs.qd_int, solver.n_dofs),
        twist_dof_start=_scalar(gs.qd_int, solver.n_dofs + 3 * solver.n_soft_verts),
        n_rigid_queries=_scalar(gs.qd_int, solver.n_samples),
        n_queries=_scalar(gs.qd_int, solver._n_soft_queries),
        pc_hash_cell=_scalar(gs.qd_float, solver._pc_hash_cell),
        tet_tree_first=V(dtype=gs.qd_int, shape=(solver.n_tet_nodes_,)),
        tet_tree_count=V(dtype=gs.qd_int, shape=(solver.n_tet_nodes_,)),
        tet_tree_escape=V(dtype=gs.qd_int, shape=(solver.n_tet_nodes_,)),
        tet_tree_is_leaf=V(dtype=gs.qd_int, shape=(solver.n_tet_nodes_,)),
        tet_tree_level_nodes=V(dtype=gs.qd_int, shape=(solver.n_tet_nodes_,)),
        tet_tree_level_start=V(dtype=gs.qd_int, shape=(solver.n_tet_levels + 1,)),
        tet_tree_elems=V(dtype=gs.qd_int, shape=(solver.n_tet_tree_elems_,)),
        tet_tree_brute_force=_scalar(gs.qd_int, 0),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiSoftState:
    # Vertex positions (the unknowns) and finite-difference velocities, with their multistep history and the
    # step-start / stage-start references and line search reference of the current solve.
    verts_pos: qd.Tensor
    verts_vel: qd.Tensor
    verts_pos_prev: qd.Tensor
    verts_vel_prev: qd.Tensor
    verts_pos_stage_start: qd.Tensor
    verts_vel_stage_start: qd.Tensor
    verts_pos_ls_ref: qd.Tensor
    # Dirichlet flags: fixed vertices keep their position.
    verts_is_fixed: qd.Tensor
    # Prescribed end-of-step position of the fixed vertices (Dirichlet targets).
    verts_target: qd.Tensor
    # Diagonal 3x3 Hessian block of every vertex (block-Jacobi preconditioner) and net contact force for readback.
    verts_H_diag: qd.Tensor
    verts_contact_force: qd.Tensor
    # Stage-start deformation gradient (stiffness damping) and the 12x12 Hessian block of every tetrahedron.
    elems_F_stage_start: qd.Tensor
    # Values of the deformable Hessian CSR (see MochiSoftInfo.csr_start), assembled once per Newton iteration.
    csr_values: qd.Tensor
    # Shell triangles: stage-start membrane and bending strains (stiffness damping) and the 18x18 Hessian block of
    # the six-vertex stencil.
    shell_elems_eps_stage_start: qd.Tensor
    shell_elems_s_stage_start: qd.Tensor
    # Rods: material axis of every segment (current, stage start, line search reference), twist angle of the step
    # (recentered to zero at every step start) with its finite-difference rate and history, stage-start axial strain
    # and stencil measures, and the Hessian blocks (3x3 axial per segment, 11x11 per stencil over
    # [x0, theta0, x1, theta1, x2]).
    rod_elems_axis: qd.Tensor
    rod_elems_axis_stage_start: qd.Tensor
    rod_elems_axis_ls_ref: qd.Tensor
    rod_elems_twist: qd.Tensor
    rod_elems_twist_vel: qd.Tensor
    rod_elems_twist_prev: qd.Tensor
    rod_elems_twist_vel_prev: qd.Tensor
    rod_elems_twist_step_start: qd.Tensor
    rod_elems_twist_vel_stage_start: qd.Tensor
    rod_elems_twist_ls_ref: qd.Tensor
    rod_elems_strain_stage_start: qd.Tensor
    rod_elems_H: qd.Tensor
    rod_elems_inertia: qd.Tensor
    rod_elems_twist_pcg: qd.Tensor
    rod_stencils_stage_start: qd.Tensor
    rod_stencils_H: qd.Tensor
    # Lower Cholesky factor of every open rod's own Hessian block in band storage: rod_band[row, d] holds the entry
    # (row, row - d) of the node-interleaved ordering.
    rod_band: qd.Tensor
    # Conservative per-step world bounds of every entity.
    entities_step_aabb_min: qd.Tensor
    entities_step_aabb_max: qd.Tensor
    # Candidate (deformable entity, collider geom) pairs and their rigid-side accumulators (see kernel_pairs_to_blocks).
    n_pairs: qd.Tensor
    pair_entity_a: qd.Tensor
    pair_link_b: qd.Tensor
    pair_geom_b: qd.Tensor
    acc_f: qd.Tensor
    acc_q: qd.Tensor
    acc_D: qd.Tensor
    acc_SD: qd.Tensor
    acc_SDS: qd.Tensor
    acc_obj: qd.Tensor
    n_hits: qd.Tensor
    # Active contact samples of the current iterate: sample, collider link (-1 if static), lever arm about the collider
    # link origin, and the per-sample matrix D = -w df/dp, from which the vertex and coupling blocks are formed.
    n_soft_hits: qd.Tensor
    n_soft_hits_max: qd.Tensor
    hit_sample: qd.Tensor
    hit_link_b: qd.Tensor
    hit_r_b: qd.Tensor
    hit_D: qd.Tensor
    # Active samples against deformable colliders: colliding side (kind 0 = rigid link sample with lever arm r_a about
    # the link origin, kind 1 = deformable sample), collider tetrahedron with the barycentric coordinates of the point,
    # per-sample matrix D = -w df/dp, force on the colliding side and readback data.
    n_sc_hits: qd.Tensor
    n_sc_hits_max: qd.Tensor
    sc_hit_kind_a: qd.Tensor
    sc_hit_sample_a: qd.Tensor
    sc_hit_link_a: qd.Tensor
    sc_hit_r_a: qd.Tensor
    sc_hit_elem_b: qd.Tensor
    sc_hit_bary_b: qd.Tensor
    sc_hit_D: qd.Tensor
    # Active samples against the point-cloud colliders of the shells: colliding side as above, collider vertex.
    n_pc_hits: qd.Tensor
    n_pc_hits_max: qd.Tensor
    pc_hit_kind_a: qd.Tensor
    pc_hit_sample_a: qd.Tensor
    pc_hit_link_a: qd.Tensor
    pc_hit_r_a: qd.Tensor
    pc_hit_vert_b: qd.Tensor
    pc_hit_D: qd.Tensor
    # spatial hash of the collider spheres: heads of the per-bin chains (-1 empty) and the next entry of a chain; a
    # sphere has one entry (8 x item + k) per cell its bounds overlap, k encoding the cell offset
    pc_hash_heads: qd.Tensor
    pc_hash_next: qd.Tensor
    # deformed bounds of the nodes of the tetrahedron hierarchy
    tet_tree_min: qd.Tensor
    tet_tree_max: qd.Tensor

    # Attachment violation at the stage start, the reference of the penalty damping.
    att_c_start: qd.Tensor
    # Whether every deformable entity sleeps (allocated only when sleeping is enabled, see MochiIslandState).
    entities_is_asleep: qd.Tensor


def get_mochi_soft_state(solver, max_soft_pairs, max_soft_hits, max_sc_hits, max_pc_hits):
    _B = solver._B
    n_sv_, n_el_, n_se_ = solver.n_soft_verts_, solver.n_soft_elems_, solver.n_soft_entities_
    n_sh_ = solver.n_shell_elems_
    n_re_, n_rs_ = solver.n_rod_elems_, solver.n_rod_stencils_
    return MochiSoftState(
        verts_pos=V(dtype=gs.qd_vec3, shape=(n_sv_, _B)),
        verts_vel=V(dtype=gs.qd_vec3, shape=(n_sv_, _B)),
        verts_pos_prev=V(dtype=gs.qd_vec3, shape=(N_HISTORY, n_sv_, _B)),
        verts_vel_prev=V(dtype=gs.qd_vec3, shape=(N_HISTORY, n_sv_, _B)),
        verts_pos_stage_start=V(dtype=gs.qd_vec3, shape=(n_sv_, _B)),
        att_c_start=V(dtype=gs.qd_vec3, shape=(solver.n_attachments_, _B)),
        entities_is_asleep=V(dtype=gs.qd_bool, shape=(n_se_, _B) if solver._options.use_sleeping else (1, 1)),
        verts_vel_stage_start=V(dtype=gs.qd_vec3, shape=(n_sv_, _B)),
        verts_pos_ls_ref=V(dtype=gs.qd_vec3, shape=(n_sv_, _B)),
        verts_is_fixed=V(dtype=gs.qd_bool, shape=(n_sv_, _B)),
        verts_target=V(dtype=gs.qd_vec3, shape=(n_sv_, _B)),
        verts_H_diag=V(dtype=gs.qd_mat3, shape=(n_sv_, _B)),
        verts_contact_force=V(dtype=gs.qd_vec3, shape=(n_sv_, _B)),
        elems_F_stage_start=V(dtype=gs.qd_mat3, shape=(n_el_, _B)),
        csr_values=V(dtype=gs.qd_float, shape=(solver.n_csr_, _B)),
        shell_elems_eps_stage_start=V_MAT(n=2, m=2, dtype=gs.qd_float, shape=(n_sh_, _B)),
        shell_elems_s_stage_start=V_MAT(n=2, m=2, dtype=gs.qd_float, shape=(n_sh_, _B)),
        rod_elems_axis=V(dtype=gs.qd_vec3, shape=(n_re_, _B)),
        rod_elems_axis_stage_start=V(dtype=gs.qd_vec3, shape=(n_re_, _B)),
        rod_elems_axis_ls_ref=V(dtype=gs.qd_vec3, shape=(n_re_, _B)),
        rod_elems_twist=V(dtype=gs.qd_float, shape=(n_re_, _B)),
        rod_elems_twist_vel=V(dtype=gs.qd_float, shape=(n_re_, _B)),
        rod_elems_twist_prev=V(dtype=gs.qd_float, shape=(N_HISTORY, n_re_, _B)),
        rod_elems_twist_vel_prev=V(dtype=gs.qd_float, shape=(N_HISTORY, n_re_, _B)),
        rod_elems_twist_step_start=V(dtype=gs.qd_float, shape=(n_re_, _B)),
        rod_elems_twist_vel_stage_start=V(dtype=gs.qd_float, shape=(n_re_, _B)),
        rod_elems_twist_ls_ref=V(dtype=gs.qd_float, shape=(n_re_, _B)),
        rod_elems_strain_stage_start=V(dtype=gs.qd_float, shape=(n_re_, _B)),
        rod_elems_H=V(dtype=gs.qd_mat3, shape=(n_re_, _B)),
        rod_elems_inertia=V(dtype=gs.qd_float, shape=(n_re_, _B)),
        rod_elems_twist_pcg=V(dtype=gs.qd_float, shape=(n_re_, _B)),
        rod_stencils_stage_start=V(dtype=gs.qd_vec3, shape=(n_rs_, _B)),
        rod_stencils_H=V_MAT(n=11, m=11, dtype=gs.qd_float, shape=(n_rs_, _B)),
        rod_band=V(dtype=gs.qd_float, shape=(solver.n_band_rows_, ROD_BAND + 1, _B)),
        entities_step_aabb_min=V(dtype=gs.qd_vec3, shape=(n_se_, _B)),
        entities_step_aabb_max=V(dtype=gs.qd_vec3, shape=(n_se_, _B)),
        n_pairs=V(dtype=gs.qd_int, shape=(_B,)),
        pair_entity_a=V(dtype=gs.qd_int, shape=(max_soft_pairs, _B)),
        pair_link_b=V(dtype=gs.qd_int, shape=(max_soft_pairs, _B)),
        pair_geom_b=V(dtype=gs.qd_int, shape=(max_soft_pairs, _B)),
        acc_f=V(dtype=gs.qd_vec3, shape=(max_soft_pairs, _B)),
        acc_q=V(dtype=gs.qd_vec3, shape=(max_soft_pairs, _B)),
        acc_D=V(dtype=gs.qd_vec6, shape=(max_soft_pairs, _B)),
        acc_SD=V(dtype=gs.qd_mat3, shape=(max_soft_pairs, _B)),
        acc_SDS=V(dtype=gs.qd_vec6, shape=(max_soft_pairs, _B)),
        acc_obj=V(dtype=gs.qd_float, shape=(max_soft_pairs, _B)),
        n_hits=V(dtype=gs.qd_int, shape=(max_soft_pairs, _B)),
        n_soft_hits=V(dtype=gs.qd_int, shape=(_B,)),
        n_soft_hits_max=_scalar(gs.qd_int, 0),
        hit_sample=V(dtype=gs.qd_int, shape=(max_soft_hits, _B)),
        hit_link_b=V(dtype=gs.qd_int, shape=(max_soft_hits, _B)),
        hit_r_b=V(dtype=gs.qd_vec3, shape=(max_soft_hits, _B)),
        hit_D=V(dtype=gs.qd_vec6, shape=(max_soft_hits, _B)),
        n_sc_hits=V(dtype=gs.qd_int, shape=(_B,)),
        n_sc_hits_max=_scalar(gs.qd_int, 0),
        sc_hit_kind_a=V(dtype=gs.qd_int, shape=(max_sc_hits, _B)),
        sc_hit_sample_a=V(dtype=gs.qd_int, shape=(max_sc_hits, _B)),
        sc_hit_link_a=V(dtype=gs.qd_int, shape=(max_sc_hits, _B)),
        sc_hit_r_a=V(dtype=gs.qd_vec3, shape=(max_sc_hits, _B)),
        sc_hit_elem_b=V(dtype=gs.qd_int, shape=(max_sc_hits, _B)),
        sc_hit_bary_b=V(dtype=gs.qd_vec4, shape=(max_sc_hits, _B)),
        sc_hit_D=V(dtype=gs.qd_vec6, shape=(max_sc_hits, _B)),
        n_pc_hits=V(dtype=gs.qd_int, shape=(_B,)),
        n_pc_hits_max=_scalar(gs.qd_int, 0),
        pc_hit_kind_a=V(dtype=gs.qd_int, shape=(max_pc_hits, _B)),
        pc_hit_sample_a=V(dtype=gs.qd_int, shape=(max_pc_hits, _B)),
        pc_hit_link_a=V(dtype=gs.qd_int, shape=(max_pc_hits, _B)),
        pc_hit_r_a=V(dtype=gs.qd_vec3, shape=(max_pc_hits, _B)),
        pc_hit_vert_b=V(dtype=gs.qd_int, shape=(max_pc_hits, _B)),
        pc_hit_D=V(dtype=gs.qd_vec6, shape=(max_pc_hits, _B)),
        pc_hash_heads=V(dtype=gs.qd_int, shape=(solver.n_pc_bins_, _B)),
        pc_hash_next=V(dtype=gs.qd_int, shape=(8 * n_sv_, _B)),
        tet_tree_min=V(dtype=gs.qd_vec3, shape=(solver.n_tet_nodes_, _B)),
        tet_tree_max=V(dtype=gs.qd_vec3, shape=(solver.n_tet_nodes_, _B)),
    )
