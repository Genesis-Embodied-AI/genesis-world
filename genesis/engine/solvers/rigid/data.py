"""The model, state and scratch of the rigid solver."""

from enum import auto

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.base_solver import VFACES, VVERTS, ErrorCode
from genesis.utils.data import ENVS, Config, Data, DataKind, Size, field, is_opted_in
from genesis.utils.simt import ExactSums


class RigidErrorCode(ErrorCode):
    """The error flags of the rigid solver, its collider and its constraint solver (see 'Solver.register_errors')."""

    OVERFLOW_COLLISION_PAIRS = auto()
    OVERFLOW_CANDIDATE_CONTACTS = auto()
    OVERFLOW_CONTACTS = auto()
    UNSUPPORTED_COLLISION_PAIR = auto()
    INVALID_CONTACT_NAN = auto()
    INVALID_FORCE_NAN = auto()
    INVALID_ACC_NAN = auto()


QS = Size("qs")
DOFS = Size("dofs")
LINKS = Size("links")
JOINTS = Size("joints")
GEOMS = Size("geoms")
ROOTS = Size("roots")
TREES = Size("trees")
# The links of every tree level by level, and the children of every link (see trees_levels_links_idx in RigidInfo)
TREES_LEVELS_LINKS = Size("trees_levels_links")
CHILD_LINKS = Size("child_links")
VERTS = Size("verts")

# The arrays of the dynamics, which a kinematic-only solver never reads
DYNAMICS = "dynamics"
# The tables of the level sweep (see trees_levels_links_idx)
LEVEL_SWEEP = "level_sweep"
# The scratch of the register-tiled mass factor (see mass_mat_tiled_scratch)
TILED_MASS = "tiled_mass"
# The buffers of the backward pass of the dynamics and of the constraint solve
BACKWARD = "backward"


class RigidInfo(Data, kind=DataKind.CONSTANT):
    """The model parameters of the rigid solver: configuration, mass matrix and kinematic tree and root tables."""

    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO, labels=DYNAMICS)
    meaninertia: qd.Tensor = field(float, shape=(ENVS,), kind=DataKind.INFO, labels=DYNAMICS)
    # Awake dofs per env under hibernation, counted down where an island sleeps and up where one wakes. An env at zero
    # skips the passes gated on it. An env at the dof count skips the passes looking for a sleeper (the chain edges and
    # the wake pass of the island build, the pair filter of the broad phase, the contact advection, the inert rows). A
    # dof-less scene pads its dof buffers to one slot and takes the slow path.
    n_awake_dofs: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)
    qpos0: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.INFO)
    qpos: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.STATE, grad=True)
    qpos_next: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.SCRATCH, grad=True)
    geoms_init_AABB: qd.Tensor = field((float, 3), shape=(GEOMS, 8), labels=DYNAMICS)
    # Stored reversed under the cooperative constraint kernels (see RigidData), i_d1 innermost, for their readers
    # striding i_d1 across lanes. The writers striding i_b have cooperative variants under the same flag.
    mass_mat: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), kind=DataKind.DERIVED, labels=DYNAMICS, grad=True)
    # Stored in declared order whatever the kernels: its main reader, a serial back-substitution with lanes varying
    # i_b, would lose more from the reversed order than the tiled factor gains.
    mass_mat_L: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), kind=DataKind.DERIVED, labels=DYNAMICS, grad=True)
    mass_mat_D_inv: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.DERIVED, labels=DYNAMICS, grad=True)
    # Scratch of the register-tiled mass factor, environment-first like the tile operations. The constraint Hessian
    # aliases it (see nt_H in ConstraintState).
    mass_mat_tiled_scratch: qd.Tensor = field(float, shape=(ENVS, DOFS, DOFS), kind=DataKind.SCRATCH, labels=TILED_MASS)
    # Kinematic roots: the links sharing a root link (links.root_idx), static ones included. A root spans the links
    # [root, links_root_end[root]), which may interleave links of other roots, so the walks gate each link on its root.
    # The composite inertia and the center of mass are per root.
    roots_link_idx: qd.Tensor = field(int, shape=(ROOTS,))
    links_root_end: qd.Tensor = field(int, shape=(LINKS,))
    # Kinematic trees: the links a chain of moving joints connects, a static link belonging to none (links_tree_idx -1)
    # and each branch of a fixed base being a tree. Tree i_t is rooted at trees_root_idx[i_t], spans the links
    # [trees_root_idx[i_t], trees_link_end[i_t]) mapped to it and the contiguous dofs [trees_dof_start[i_t],
    # trees_dof_start[i_t] + trees_n_dofs[i_t]). The islands are built on the trees. A scene without any tree keeps
    # one padded slot at root 0 and link end 0, so every tree walk is empty.
    trees_root_idx: qd.Tensor = field(int, shape=(TREES,))
    trees_link_end: qd.Tensor = field(int, shape=(TREES,))
    trees_n_links: qd.Tensor = field(int, shape=(TREES,))
    trees_dof_start: qd.Tensor = field(int, shape=(TREES,))
    trees_n_dofs: qd.Tensor = field(int, shape=(TREES,))
    links_tree_idx: qd.Tensor = field(int, shape=(LINKS,))
    # The links of every tree level by level, for the level sweep (see func_sweep_links_by_level). The levels of tree
    # i_t start at entry trees_level_start[i_t] and number trees_n_levels[i_t], and the level of entry i_k ends before
    # entry trees_levels_links_end[i_k]. The root tables list the links of every root the same way, the level of entry
    # i_k starting at entry roots_levels_links_start[i_k].
    trees_level_start: qd.Tensor = field(int, shape=(TREES,), labels=LEVEL_SWEEP)
    trees_n_levels: qd.Tensor = field(int, shape=(TREES,), labels=LEVEL_SWEEP)
    trees_levels_links_idx: qd.Tensor = field(int, shape=(TREES_LEVELS_LINKS,), labels=LEVEL_SWEEP)
    trees_levels_links_end: qd.Tensor = field(int, shape=(TREES_LEVELS_LINKS,), labels=LEVEL_SWEEP)
    roots_level_start: qd.Tensor = field(int, shape=(ROOTS,), labels=LEVEL_SWEEP)
    roots_n_levels: qd.Tensor = field(int, shape=(ROOTS,), labels=LEVEL_SWEEP)
    roots_levels_links_idx: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    roots_levels_links_start: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    roots_levels_links_end: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    # The children of link i_l are links_child_idx[links_child_start[i_l]:links_child_start[i_l + 1]], in descending
    # order: the leaf-to-root folds of the level sweep add them into their parent in the order of the serial folds.
    links_child_start: qd.Tensor = field(int, shape=(LINKS + 1,), labels=LEVEL_SWEEP)
    links_child_idx: qd.Tensor = field(int, shape=(CHILD_LINKS,), labels=LEVEL_SWEEP)
    # Bounds of the mass block of each dof: the dofs of its branch rooted where the fixed structure ends, deeper
    # branches staying in the enclosing block, merged across entities and kept contiguous by attach(). The blocks
    # partition the dof range of a kinematic tree (an aligned free body splits into one block per dof), and the
    # assembly, the factor and the solve of the mass matrix walk them.
    dofs_mass_block_start: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    dofs_mass_block_end: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    dofs_mass_envelope_start: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    # Mass parent of each dof within its mass block, -1 for none: the previous dof of its link, else the last dof of
    # its nearest ancestor link holding one.
    dofs_mass_parent: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    # Mask of the (dof, dof) pairs the mass matrix couples: a dof with its ancestors along the kinematic chain and the
    # dofs of its own link, within its mass block.
    mass_parent_mask: qd.Tensor = field(float, shape=(DOFS, DOFS), labels=DYNAMICS)
    # Runtime constants
    substep_dt: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    ls_iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    ls_tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    noslip_iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    noslip_tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    impratio: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    n_equalities: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    n_candidate_equalities: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    hibernation_thresh_vel: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    EPS: qd.Tensor = field(float, shape=())


# The counts of the largest entity of the solver, each taken over every entity
ENTITY_QS = Size("entity_qs")
ENTITY_DOFS = Size("entity_dofs")
ENTITY_LINKS = Size("entity_links")
ENTITY_JOINTS = Size("entity_joints")


# The scratch of the kinematics queries, over the dofs of the widest entity of the solver (see ENTITY_DOFS)
KINEMATICS = "kinematics"


class KinematicsScratch(Data, kind=DataKind.SCRATCH):
    """The scratch of the kinematics queries of every entity of a solver, one column per environment.

    The Jacobian spans the widest entity, and the configuration cache spans the configuration of the solver, indexed
    by global q indices.
    """

    jacobian: qd.Tensor = field(float, shape=(6, ENTITY_DOFS, ENVS), labels=KINEMATICS)
    qpos_cache: qd.Tensor = field(float, shape=(QS, ENVS), labels=KINEMATICS)


# Sizes of the elements of the rigid solver, and of the environments of the model parameters the options batch (see
# 'batch_dofs_info', 'batch_links_info' and 'batch_joints_info' in RigidOptions)
ENTITIES = Size("entities")
FACES = Size("faces")
EDGES = Size("edges")
FREE_VERTS = Size("free_verts")
FIXED_VERTS = Size("fixed_verts")
CUSTOM_VVERTS = Size("custom_vverts")
VGEOMS = Size("vgeoms")
CANDIDATE_EQUALITIES = Size("candidate_equalities")
DOFS_INFO_ENVS = Size("dofs_info_envs")
LINKS_INFO_ENVS = Size("links_info_envs")
JOINTS_INFO_ENVS = Size("joints_info_envs")
# The backward slots of a link, one per joint plus the link itself
LINK_BW_SLOTS = Size("link_bw_slots")

# The arrays the midpoint integration of free bodies reads (see func_midpoint_free_body)
MIDPOINT = "midpoint"


class DofsInfo(Data, kind=DataKind.INFO, shape=(DOFS, DOFS_INFO_ENVS)):
    """The model parameters of each dof of the rigid solver, per environment where the options batch them."""

    stiffness: qd.Tensor = field(float)
    invweight: qd.Tensor = field(float)
    armature: qd.Tensor = field(float)
    damping: qd.Tensor = field(float)
    frictionloss: qd.Tensor = field(float)
    motion_ang: qd.Tensor = field((float, 3))
    motion_vel: qd.Tensor = field((float, 3))
    limit: qd.Tensor = field((float, 2))
    act_gain: qd.Tensor = field(float)
    act_bias: qd.Tensor = field((float, 3))
    force_range: qd.Tensor = field((float, 2))
    dof_length: qd.Tensor = field(float)


class DofsState(Data, kind=DataKind.DERIVED, shape=(DOFS, ENVS)):
    """The state of each dof of the rigid solver in each environment: its motion, forces and controls."""

    # A *_bw array keeps what the backward pass reads and the forward pass overwrites
    force: qd.Tensor = field(float, grad=True)
    qf_bias: qd.Tensor = field(float, grad=True)
    qf_passive: qd.Tensor = field(float, grad=True)
    qf_applied: qd.Tensor = field(float, grad=True)
    pos: qd.Tensor = field(float, grad=True)
    vel: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    vel_prev: qd.Tensor = field(float, kind=DataKind.SCRATCH, grad=True)
    vel_next: qd.Tensor = field(float, kind=DataKind.SCRATCH, grad=True)
    acc: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    acc_smooth: qd.Tensor = field(float, grad=True)
    acc_smooth_bw: qd.Tensor = field(float, shape=(2, DOFS, ENVS), labels=BACKWARD, grad=True)
    qf_smooth: qd.Tensor = field(float, grad=True)
    # Implicit damping (see func_implicit_damping in forward_dynamics.py): the implicit share h D qacc of the damping
    # force, D being the diagonal the damped mass factor adds, and the correction (M + hD)^-1 h D qacc subtracted from
    # the acceleration
    qf_damping_implicit: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    qacc_damping_implicit: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    cdof_ang: qd.Tensor = field((float, 3), grad=True)
    cdof_vel: qd.Tensor = field((float, 3), grad=True)
    cdofd_ang: qd.Tensor = field((float, 3), grad=True)
    cdofd_vel: qd.Tensor = field((float, 3), grad=True)
    f_vel: qd.Tensor = field((float, 3), grad=True)
    f_ang: qd.Tensor = field((float, 3), grad=True)
    ctrl_force: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    ctrl_pos: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    ctrl_vel: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    ctrl_mode: qd.Tensor = field(int, kind=DataKind.STATE)
    is_hibernated: qd.Tensor = field(int, kind=DataKind.STATE)


class LinksInfo(Data, kind=DataKind.INFO, shape=(LINKS, LINKS_INFO_ENVS)):
    """The model parameters of each link, per environment where batched: tree, ranges, pose and inertia."""

    parent_idx: qd.Tensor = field(int)
    root_idx: qd.Tensor = field(int)
    q_start: qd.Tensor = field(int)
    dof_start: qd.Tensor = field(int)
    joint_start: qd.Tensor = field(int)
    q_end: qd.Tensor = field(int)
    dof_end: qd.Tensor = field(int)
    joint_end: qd.Tensor = field(int)
    n_dofs: qd.Tensor = field(int)
    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    invweight: qd.Tensor = field((float, 2))
    is_fixed: qd.Tensor = field(bool)
    inertial_pos: qd.Tensor = field((float, 3))
    inertial_quat: qd.Tensor = field((float, 4))
    inertial_i: qd.Tensor = field((float, (3, 3)))
    inertial_mass: qd.Tensor = field(float)
    entity_idx: qd.Tensor = field(int)
    # The geoms and visual geoms of each link, which heterogeneous environments vary
    geom_start: qd.Tensor = field(int)
    geom_end: qd.Tensor = field(int)
    vgeom_start: qd.Tensor = field(int)
    vgeom_end: qd.Tensor = field(int)


class LinksState(Data, kind=DataKind.DERIVED, shape=(LINKS, ENVS)):
    """The state of each link in each environment: pose, velocities, inertias, wrenches and hibernation."""

    # A *_bw array keeps what the backward pass reads and the forward pass overwrites
    cinr_inertial: qd.Tensor = field((float, (3, 3)), grad=True)
    cinr_pos: qd.Tensor = field((float, 3), grad=True)
    cinr_quat: qd.Tensor = field((float, 4), grad=True)
    cinr_mass: qd.Tensor = field(float, grad=True)
    crb_inertial: qd.Tensor = field((float, (3, 3)), grad=True)
    crb_pos: qd.Tensor = field((float, 3), grad=True)
    crb_quat: qd.Tensor = field((float, 4), grad=True)
    crb_mass: qd.Tensor = field(float, grad=True)
    cdd_vel: qd.Tensor = field((float, 3), grad=True)
    cdd_ang: qd.Tensor = field((float, 3), grad=True)
    # The pose of a fixed root link is state, which forward kinematics leaves as set (see
    # func_forward_kinematics_link_bw), the poses of the other links being recomputed from the positions of the dofs.
    pos: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    quat: qd.Tensor = field((float, 4), kind=DataKind.STATE, grad=True)
    pos_bw: qd.Tensor = field((float, 3), shape=(LINKS, LINK_BW_SLOTS, ENVS), grad=True)
    quat_bw: qd.Tensor = field((float, 4), shape=(LINKS, LINK_BW_SLOTS, ENVS), grad=True)
    i_pos: qd.Tensor = field((float, 3), grad=True)
    i_pos_bw: qd.Tensor = field((float, 3), grad=True)
    i_quat: qd.Tensor = field((float, 4), grad=True)
    cd_ang: qd.Tensor = field((float, 3), grad=True)
    cd_vel: qd.Tensor = field((float, 3), grad=True)
    # Whether a contact or a connect or weld equality row involves the link this step, written by the constraint
    # assembly from the sorted contacts and read by the midpoint integration
    is_constrained: qd.Tensor = field(bool)
    # Midpoint angular velocity a free body eligible for midpoint integration converged to this substep, in its inertial
    # frame, and the residual of the midpoint equation there. The integration the backward pass reverses resumes from
    # them, since autodiff cannot reverse the Newton iteration (see func_midpoint_free_body).
    midpoint_ang: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, labels=MIDPOINT)
    midpoint_res: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, labels=MIDPOINT)
    cd_ang_bw: qd.Tensor = field((float, 3), shape=(LINKS, LINK_BW_SLOTS, ENVS), grad=True)
    cd_vel_bw: qd.Tensor = field((float, 3), shape=(LINKS, LINK_BW_SLOTS, ENVS), grad=True)
    mass_sum: qd.Tensor = field(float, grad=True)
    # Center of mass of the kinematic tree
    root_COM: qd.Tensor = field((float, 3), grad=True)
    root_COM_bw: qd.Tensor = field((float, 3), grad=True)
    cacc_ang: qd.Tensor = field((float, 3), grad=True)
    cacc_lin: qd.Tensor = field((float, 3), grad=True)
    cfrc_ang: qd.Tensor = field((float, 3), grad=True)
    cfrc_vel: qd.Tensor = field((float, 3), grad=True)
    cfrc_input_ang: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    cfrc_input_vel: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    # The wrench the other solvers apply the links during a substep, which the next substep consumes
    cfrc_external_ang: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    cfrc_external_vel: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    contact_force: qd.Tensor = field((float, 3), grad=True)
    # Hibernation: is_hibernated flags a sleeping link, awake_steps counts the consecutive substeps an awake link has
    # spent below the hibernation speed tolerance, up to hibernation_min_steps (see func_count_settled_step), zero
    # again the step it exceeds the tolerance or wakes.
    is_hibernated: qd.Tensor = field(int, kind=DataKind.STATE)
    awake_steps: qd.Tensor = field(int, kind=DataKind.STATE)
    # The forces and torques the other solvers apply the links over a substep, summed exactly whatever their order, slot
    # 6 i_l + i_c holding the torque then the force components of link i_l. The solvers coupled with the rigid solver
    # opt it in when the accumulation order must be fixed, and the rigid solver adds it to the external wrench of the
    # links before its next substep reads it (see kernel_flush_external_wrenches).
    cfrc_external_sums: ExactSums = field(opt_in=True, kind=DataKind.STATE, labels=DYNAMICS)


class JointsInfo(Data, kind=DataKind.INFO, shape=(JOINTS, JOINTS_INFO_ENVS)):
    """The model parameters of each joint, per environment where batched: type, ranges, position, solver."""

    type: qd.Tensor = field(int)
    sol_params: qd.Tensor = field((float, 7))
    q_start: qd.Tensor = field(int)
    dof_start: qd.Tensor = field(int)
    dof_end: qd.Tensor = field(int)
    pos: qd.Tensor = field((float, 3))


class JointsState(Data, kind=DataKind.DERIVED, shape=(JOINTS, ENVS)):
    """The world anchor and axis of each joint of the rigid solver in each environment."""

    xanchor: qd.Tensor = field((float, 3), grad=True)
    xaxis: qd.Tensor = field((float, 3), grad=True)


class GeomsInfo(Data, kind=DataKind.INFO, shape=(GEOMS,)):
    """The model parameters of each collision geom: pose on its link, shape, mesh ranges and contact properties."""

    pos: qd.Tensor = field((float, 3))
    center: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    data: qd.Tensor = field((float, 7))
    link_idx: qd.Tensor = field(int)
    type: qd.Tensor = field(int)
    friction: qd.Tensor = field(float)
    friction_torsional: qd.Tensor = field(float)
    friction_rolling: qd.Tensor = field(float)
    sol_params: qd.Tensor = field((float, 7))
    vert_start: qd.Tensor = field(int)
    vert_end: qd.Tensor = field(int)
    verts_state_start: qd.Tensor = field(int)
    face_start: qd.Tensor = field(int)
    face_end: qd.Tensor = field(int)
    edge_num: qd.Tensor = field(int)
    edge_start: qd.Tensor = field(int)
    edge_end: qd.Tensor = field(int)
    is_convex: qd.Tensor = field(bool)
    contype: qd.Tensor = field(int)
    conaffinity: qd.Tensor = field(int)
    is_fixed: qd.Tensor = field(bool)
    is_hollow: qd.Tensor = field(bool)
    needs_coup: qd.Tensor = field(int)
    coup_friction: qd.Tensor = field(float)
    coup_softness: qd.Tensor = field(float)
    coup_restitution: qd.Tensor = field(float)


class GeomsState(Data, kind=DataKind.DERIVED, shape=(GEOMS, ENVS)):
    """The pose, bounding box and hibernation flag of each collision geom of the rigid solver in each environment."""

    pos: qd.Tensor = field((float, 3), grad=True)
    quat: qd.Tensor = field((float, 4), grad=True)
    aabb_min: qd.Tensor = field((float, 3))
    aabb_max: qd.Tensor = field((float, 3))
    verts_updated: qd.Tensor = field(bool)
    is_hibernated: qd.Tensor = field(int, kind=DataKind.STATE)
    friction_ratio: qd.Tensor = field(float, kind=DataKind.INFO)


class VertsInfo(Data, kind=DataKind.CONSTANT, shape=(VERTS,)):
    """The collision vertices of the rigid geoms, in the frame of their geom."""

    init_pos: qd.Tensor = field((float, 3))
    init_normal: qd.Tensor = field((float, 3))
    geom_idx: qd.Tensor = field(int)
    verts_state_idx: qd.Tensor = field(int)
    is_fixed: qd.Tensor = field(bool)


class FacesInfo(Data, kind=DataKind.CONSTANT, shape=(FACES,)):
    """The triangles of the collision meshes of the rigid geoms."""

    verts_idx: qd.Tensor = field((int, 3))
    geom_idx: qd.Tensor = field(int)


class EdgesInfo(Data, kind=DataKind.CONSTANT, shape=(EDGES,)):
    """The edges of the collision meshes of the rigid geoms, with their length."""

    v0: qd.Tensor = field(int)
    v1: qd.Tensor = field(int)
    length: qd.Tensor = field(float)


class VertsState(Data, kind=DataKind.DERIVED):
    """The world positions of the collision vertices of the rigid geoms."""

    pos: qd.Tensor = field((float, 3))


class VVertsInfo(Data, kind=DataKind.CONSTANT, shape=(VVERTS,)):
    """The vertices of the visual meshes of the rigid geoms, in the frame of their visual geom."""

    init_pos: qd.Tensor = field((float, 3))
    init_vnormal: qd.Tensor = field((float, 3))
    vgeom_idx: qd.Tensor = field(int)
    vverts_state_idx: qd.Tensor = field(int)


class VVertsState(Data, kind=DataKind.STATE, shape=(CUSTOM_VVERTS, ENVS)):
    """The world positions of the visual vertices of the entities opted into custom visual vertices.

    They are state, since 'set_vverts' writes them from one step to the next.
    """

    pos: qd.Tensor = field(
        (float, 3),
        oversize_hint="Consider opting fewer kinematic entities into 'enable_custom_vverts=True', or reducing "
        "'n_envs'.",
    )


class VFacesInfo(Data, kind=DataKind.CONSTANT, shape=(VFACES,)):
    """The triangles of the visual meshes of the rigid geoms."""

    vverts_idx: qd.Tensor = field((int, 3))
    vgeom_idx: qd.Tensor = field(int)


class VGeomsInfo(Data, kind=DataKind.CONSTANT, shape=(VGEOMS,)):
    """The model parameters of each visual geom of the rigid solver: its pose on its link, mesh ranges and color."""

    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    link_idx: qd.Tensor = field(int)
    vvert_start: qd.Tensor = field(int)
    vvert_end: qd.Tensor = field(int)
    vface_start: qd.Tensor = field(int)
    vface_end: qd.Tensor = field(int)
    color: qd.Tensor = field((float, 4))


class VGeomsState(Data, kind=DataKind.DERIVED, shape=(VGEOMS, ENVS)):
    """The world pose of each visual geom of the rigid solver in each environment."""

    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))


class EqualitiesInfo(Data, kind=DataKind.INFO, shape=(CANDIDATE_EQUALITIES, ENVS)):
    """The equality constraints of each environment: type, bound objects, data and solver parameters."""

    eq_obj1id: qd.Tensor = field(int)
    eq_obj2id: qd.Tensor = field(int)
    # The data of a screw constraint, the largest one, sizes the entries of all equalities (see func_screw_frame in
    # constraint/solver.py)
    eq_data: qd.Tensor = field((float, 18))
    eq_type: qd.Tensor = field(int)
    sol_params: qd.Tensor = field((float, 7))


class EntitiesInfo(Data, kind=DataKind.INFO, shape=(ENTITIES,)):
    """The model parameters of each entity of the rigid solver: dof, link and geom ranges and gravity compensation."""

    dof_start: qd.Tensor = field(int)
    dof_end: qd.Tensor = field(int)
    n_dofs: qd.Tensor = field(int)
    link_start: qd.Tensor = field(int)
    link_end: qd.Tensor = field(int)
    geom_start: qd.Tensor = field(int)
    geom_end: qd.Tensor = field(int)
    gravity_compensation: qd.Tensor = field(float)


class EntitiesState(Data, kind=DataKind.STATE, shape=(ENTITIES, ENVS)):
    """The hibernation flag of each entity of the rigid solver in each environment."""

    is_hibernated: qd.Tensor = field(int)


class DynInfo(Data):
    """The model of the kinematics and dynamics, one struct per element type."""

    entities: EntitiesInfo
    links: LinksInfo
    joints: JointsInfo
    dofs: DofsInfo
    geoms: GeomsInfo = field(labels=DYNAMICS)
    verts: VertsInfo = field(labels=DYNAMICS)
    faces: FacesInfo = field(labels=DYNAMICS)
    edges: EdgesInfo = field(labels=DYNAMICS)
    vverts: VVertsInfo
    vfaces: VFacesInfo
    vgeoms: VGeomsInfo
    equalities: EqualitiesInfo = field(labels=DYNAMICS)


class DynState(Data):
    """The state of the kinematics and dynamics, one struct per element type."""

    entities: EntitiesState
    links: LinksState
    joints: JointsState
    dofs: DofsState
    geoms: GeomsState = field(labels=DYNAMICS)
    free_verts: VertsState = field(shape=(FREE_VERTS, ENVS), labels=DYNAMICS)
    fixed_verts: VertsState = field(shape=(FIXED_VERTS,), labels=DYNAMICS)
    vverts: VVertsState
    vgeoms: VGeomsState


class RigidConfig(Config):
    """The compile-time configuration of the kernels of the rigid solver."""

    backend: int
    para_level: int
    enable_collision: bool
    use_hibernation: bool
    batch_links_info: bool
    batch_dofs_info: bool
    batch_joints_info: bool
    enable_mujoco_compatibility: bool
    enable_elliptic_friction: bool
    # Whether contact friction is bounded by the disc of radius mu * f_n, f_n being the normal force at the current
    # iterate, rather than jointly with it through the coupled cone (see gs.contact_resolution). It implies
    # enable_elliptic_friction, whose rows separate the normal force from the friction disc.
    enable_signorini_contact: bool
    enable_multi_contact: bool
    enable_joint_limit: bool
    box_box_detection: bool
    sparse_solve: bool
    integrator: gs.integrator
    solver_type: gs.constraint_solver
    requires_grad: bool
    # -1 to let the dispatch pick, 0 or 1 to force it
    prefer_decomposed_solver: int = -1
    # Whether the Hessian without its cone blocks persists in the mirror slots of nt_H (see nt_H in ConstraintState)
    enable_cone_free_hessian_reuse: bool = False
    # Whether contacts carry torsional friction rows resisting relative spin about the contact normal: one extra
    # opposing pyramid pair per contact with the pyramidal cone, one extra cone row with the elliptic cone.
    enable_torsional_friction: bool = False
    # Whether contacts also carry rolling friction rows resisting relative rotation about the two tangent axes: two
    # extra opposing pyramid pairs per contact with the pyramidal cone, two extra cone rows with the elliptic cone.
    # It requires enable_torsional_friction, the rolling rows following the spin row.
    enable_rolling_friction: bool = False
    # Consecutive steps the dof velocities of a body must stay below the tolerance before it hibernates, so a body
    # momentarily slow (at the apex of a toss) stays awake
    hibernation_min_steps: int = 10
    # Whether the initialization of the solve runs over (constraint, env) pairs, for envs too few to fill the GPU
    parallel_init: bool = False
    # Whether the walks of the kinematic trees and roots sweep them level by level (see func_sweep_links_by_level)
    enable_level_sweep: bool = False
    # Lanes of the level sweep per env, a power of two dividing its blocks of 32 lanes. Four envs per block balance the
    # lanes a narrow level leaves idle against the links a wide level serializes.
    level_sweep_n_lanes_per_env: int = 8
    broadphase_traversal: gs.broadphase_traversal = gs.broadphase_traversal.SAP
    enable_tiled_cholesky_mass_matrix: bool = False
    mass_matrix_fits_shared: bool = False
    # Register-tile width of the mass-matrix Cholesky kernels, 16 or 32 (see cholesky_tile_size_for)
    cholesky_tile_size: int = 32
    # Whether func_factor_mass factors the mass block of each entity in registers with the tiled Cholesky of the
    # Hessian, where the block exceeds shared memory. It requires every entity to be one mass block, and its tiles are
    # 32 wide, a block exceeding shared memory spanning well over 48 dofs.
    enable_register_tiled_mass: bool = False
    # Whether the shared-memory mass factor also solves the smooth acceleration against the factor it holds, after the
    # force passes, in place of func_compute_qacc. See its resolution in the rigid solver.
    enable_fused_smooth_acc_solve: bool = False
    # Whether func_solve_init seeds the factor of every island with the tiled per-island kernels, else one env per
    # thread (see has_per_env_solve_seed). See its resolution in the rigid solver.
    enable_tiled_island_seed: bool = False
    # Whether the scene holds one dof-carrying kinematic tree, an env then forming one island at most, whose passes read
    # the dof and row ranges of the env
    is_single_island: bool = False
    # Whether the seed kernel assembles the Hessian of a single-island scene too large for the cooperative factor, which
    # the monolith factors with the scalar dense Cholesky (see _kernel_solve_monolith)
    has_scalar_seed_factor: bool = False
    # Whether the constraint solver runs the subgroup-cooperative GPU kernels (linesearch refinement, friction rows,
    # mass-matrix assembly) on environment-first arrays (see COOPERATIVE_ROWS)
    enable_cooperative_constraint_kernels: bool = False
    # Whether the noslip sweep of an island runs on a block of 32 lanes: its rows are colored so that the rows of a
    # color touch disjoint mass blocks, and the lanes update the rows of a color in parallel, color after color (see
    # func_noslip in noslip.py). The visit order differs from the one-thread sweep, and so do the iterates. See its
    # resolution in the rigid solver.
    enable_cooperative_noslip: bool = False
    # Whether the arrays labeled COOPERATIVE_ROWS are stored environment-first, under the cooperative constraint
    # kernels, on CPU or under serialized execution. Loops whose order follows the memory order key on it, and the
    # choice of the cooperative algorithms on enable_cooperative_constraint_kernels.
    constraint_layout_batch_first: bool = False
    # Whether the Newton system is Jacobi-equilibrated (see nt_jacobi in ConstraintState)
    enable_jacobi_equilibration: bool = False
    tiled_n_dofs_per_block: int = -1
    # Dof caps of the shared tiles of the cooperative per-island factor and solve, one launch per cap (see
    # island_tile_caps): the doublings of the first cap below the last one. The first cap is the tile size doubled up
    # to the smallest dof-carrying tree, the last one the largest multiple of the tile size fitting GPU shared memory.
    # An island above the last cap factors in global memory.
    island_tile_cap_first: int = 0
    island_tile_cap_last: int = 0
    # Whether an island can hold more dofs than the last cap, which compiles the factor paths above it
    has_island_above_tile_cap: bool = False

    @property
    def rows_per_contact(self) -> int:
        """Constraint rows per contact.

        The pyramidal cone has two opposing edges per friction axis, the elliptic cone the normal row plus one row per
        friction axis. Torsional friction adds the spin axis, and rolling friction the two tangent axes.
        """
        n_extra_axes = int(self.enable_torsional_friction) + 2 * int(self.enable_rolling_friction)
        if self.enable_elliptic_friction:
            return 3 + n_extra_axes
        return 4 + 2 * n_extra_axes

    @property
    def hessian_rank_update_batch(self) -> int:
        """Rank-1 Cholesky updates the per-island incremental factor fuses into one column sweep.

        It sizes the nt_vec slots and the unroll of func_rank_batch_update_island, and covers the two slots per cone row
        of the coupled elliptic-cone update (see func_cone_rank_update_island).
        """
        return max(8, 2 * self.rows_per_contact)

    @property
    def has_per_env_solve_seed(self) -> bool:
        """Whether func_solve_init seeds the Newton factor, the gradient and the convergence test one env per thread.

        It holds on GPU wherever the tiled seed is off or leaves the factor to the scalar dense Cholesky (see
        has_scalar_seed_factor). The decomposed arm then factors one env per thread at every iteration too, its tiled
        factor maintaining the Hessian the tiled seed assembles.
        """
        return (
            self.solver_type == gs.constraint_solver.Newton
            and self.backend != gs.cpu
            and (not self.enable_tiled_island_seed or self.has_scalar_seed_factor)
        )


class RigidData(Data):
    """Every array of a kinematic or rigid solver, allocated from the solver once its scene is built."""

    rigid_info: RigidInfo
    dyn_info: DynInfo
    dyn_state: DynState
    kinematics_scratch: KinematicsScratch
    errno: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)

    def __init__(self, solver, kinematic_only: bool):
        is_dynamic = not kinematic_only
        config, options = solver.rigid_config, solver._options
        entities = solver.entities
        entity_dofs = max([entity.n_dofs for entity in entities], default=0)
        sizes = dict(
            envs=solver._B,
            qs=solver.n_qs_,
            dofs=solver.n_dofs_,
            links=solver.n_links_,
            joints=solver.n_joints_,
            roots=solver.n_roots_,
            trees=solver.n_trees_,
            entities=solver.n_entities_,
            vverts=solver.n_vverts_,
            custom_vverts=solver.n_custom_vverts_,
            vfaces=solver.n_vfaces_,
            vgeoms=solver.n_vgeoms_,
            dofs_info_envs=solver._B if options.batch_dofs_info else (),
            links_info_envs=solver._B if options.batch_links_info else (),
            joints_info_envs=solver._B if options.batch_joints_info else (),
            link_bw_slots=max([link.n_joints for link in solver.links], default=0) + 1 if solver._requires_grad else 1,
            entity_dofs=entity_dofs,
        )
        values = dict(EPS=gs.EPS)
        if is_dynamic:
            sizes.update(
                geoms=solver.n_geoms_,
                verts=solver.n_verts_,
                faces=solver.n_faces_,
                edges=solver.n_edges_,
                free_verts=solver.n_free_verts_,
                fixed_verts=solver.n_fixed_verts_,
                candidate_equalities=solver.n_candidate_equalities_,
                trees_levels_links=np.count_nonzero(solver._links_tree_root_idx >= 0),
                child_links=np.count_nonzero(solver._links_parent_idx >= 0),
            )
            values.update(
                substep_dt=solver._substep_dt,
                iterations=options.iterations,
                tolerance=options.tolerance,
                ls_iterations=options.ls_iterations,
                ls_tolerance=options.ls_tolerance,
                noslip_iterations=options.noslip_iterations,
                noslip_tolerance=options.noslip_tolerance,
                impratio=options.impratio,
                n_equalities=solver._n_equalities,
                n_candidate_equalities=solver.n_candidate_equalities_,
                hibernation_thresh_vel=solver._hibernation_thresh_vel,
            )
        is_external_wrench_exact = is_dynamic and is_opted_in(RigidData.dyn_state.links.cfrc_external_sums)
        if is_external_wrench_exact:
            sizes.update(exact_sum_slots=6 * solver.n_links_, exact_sum_bins=8)
        super().__init__(
            sizes=sizes,
            values=dict(rigid_info=values),
            layout={
                RigidData.rigid_info.mass_mat: (2, 1, 0)
                if is_dynamic and config.enable_cooperative_constraint_kernels
                else None
            },
            unused=dict(
                dynamics=kinematic_only,
                level_sweep=not (is_dynamic and config.enable_level_sweep),
                tiled_mass=not (is_dynamic and config.enable_register_tiled_mass),
                midpoint=not (is_dynamic and config.integrator != gs.integrator.Euler),
                backward=not solver._requires_grad,
                kinematics=entity_dofs == 0,
            ),
            requires_grad=solver._requires_grad,
        )
        if is_external_wrench_exact:
            # Units from 2^-120 up to 2^104 N or N.m, 2^32 apart (see ExactSums)
            self.dyn_state.links.cfrc_external_sums.bins_unit.from_numpy(
                2.0 ** (-120 + 32 * np.arange(8, dtype=gs.np_float))
            )
