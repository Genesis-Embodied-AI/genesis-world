"""
Rigid solver dynamics kernel and function definitions.

This module contains Quadrants kernel and function definitions for rigid body dynamics
simulation, including:
- Mass matrix computation and factorization
- Force calculations (torque, passive, bias, actuation)
- Forward dynamics computation
- Velocity and acceleration updates
- Integration schemes (Euler, implicit damping)
- Cartesian space updates

These functions are used by the RigidSolver class to perform physics simulation
of articulated rigid body systems.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from .forward_kinematics import (
    LINK_SWEEP_PASS,
    func_forward_velocity_root,
    func_sweep_links_by_level,
    func_update_acc_link,
    func_update_cartesian_space_root,
)
from .misc import (
    func_add_safe_backward,
    func_is_awake_link,
    func_is_awake_tree,
    func_wakeup_island,
    linear_to_lower_tri,
)


@qd.kernel(fastcache=True)
def update_qacc_from_qvel_delta(
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    n_dofs = dyn_state.dofs.ctrl_mode.shape[0]
    _B = dyn_state.dofs.ctrl_mode.shape[1]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.dofs.is_hibernated[i_d, i_b]
        if is_awake:
            dyn_state.dofs.acc[i_d, i_b] = (
                dyn_state.dofs.vel[i_d, i_b] - dyn_state.dofs.vel_prev[i_d, i_b]
            ) / rigid_info.substep_dt[None]
            dyn_state.dofs.vel[i_d, i_b] = dyn_state.dofs.vel_prev[i_d, i_b]


@qd.kernel(fastcache=True)
def update_qvel(
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    _B = dyn_state.dofs.vel.shape[1]
    n_dofs = dyn_state.dofs.vel.shape[0]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.dofs.is_hibernated[i_d, i_b]
        if is_awake:
            dyn_state.dofs.vel_prev[i_d, i_b] = dyn_state.dofs.vel[i_d, i_b]
            dyn_state.dofs.vel[i_d, i_b] = (
                dyn_state.dofs.vel[i_d, i_b] + dyn_state.dofs.acc[i_d, i_b] * rigid_info.substep_dt[None]
            )


@qd.kernel(fastcache=True)
def kernel_compute_mass_matrix(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    decompose: qd.template(),
):
    func_compute_mass_matrix_masked(
        envs_idx, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False, is_backward=False
    )
    if decompose:
        func_factor_mass_masked(envs_idx, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False)


# @@@@@@@@@ Composer starts here
# decomposed kernels should happen in the block below. This block will be handled by composer and composed into a single kernel
@qd.func
def func_forward_dynamics(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    func_compute_mass_matrix(
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
        qd.static(rigid_config.integrator == gs.integrator.approximate_implicitfast),
        is_backward,
    )
    func_factor_mass(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False)
    func_torque_and_passive_force(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_update_acc(dyn_state, dyn_info, rigid_info, rigid_config, update_cacc=False, is_backward=is_backward)
    func_update_force(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_bias_force(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_compute_qacc(dyn_state, dyn_info, rigid_info, rigid_config)


@qd.kernel(fastcache=True)
def kernel_forward_dynamics(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    func_forward_dynamics(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(fastcache=True)
def kernel_update_acc(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    func_update_acc(dyn_state, dyn_info, rigid_info, rigid_config, update_cacc=True, is_backward=False)


@qd.func
def func_vel_at_point(link_idx, i_b, pos_world, links_state: array_class.LinksState):
    """
    Velocity of a certain point on a rigid link.
    """
    vel_rot = links_state.cd_ang[link_idx, i_b].cross(pos_world - links_state.root_COM[link_idx, i_b])
    vel_lin = links_state.cd_vel[link_idx, i_b]
    return vel_rot + vel_lin


@qd.func
def func_crb_initialize(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    is_awake = True
    if qd.static(rigid_config.use_hibernation):
        is_awake = not dyn_state.links.is_hibernated[i_l, i_b]
    if is_awake:
        dyn_state.links.crb_inertial[i_l, i_b] = dyn_state.links.cinr_inertial[i_l, i_b]
        dyn_state.links.crb_pos[i_l, i_b] = dyn_state.links.cinr_pos[i_l, i_b]
        dyn_state.links.crb_quat[i_l, i_b] = dyn_state.links.cinr_quat[i_l, i_b]
        dyn_state.links.crb_mass[i_l, i_b] = dyn_state.links.cinr_mass[i_l, i_b]


@qd.func
def func_crb_fold(
    i_r,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    """Fold the composite-rigid-body inertia of one kinematic root, from its leaves up to its root link.

    One thread handles the whole root, walking its link span in descending order so that children fold before their
    parent propagates, and gating each link on its root (see roots_link_idx in array_class.py). Mirrors the root walk of
    func_COM_root. The links of a root sleep as a unit, so the root link tells whether they are awake.
    """
    i_l_root = rigid_info.roots_link_idx[i_r]
    if func_is_awake_link(i_l_root, i_b, dyn_state, rigid_config):
        i_l_end = rigid_info.links_root_end[i_l_root]
        for k in range(i_l_end - i_l_root):
            i_l = i_l_end - 1 - k
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            i_p = dyn_info.links.parent_idx[I_l]
            I_p = [i_p, i_b]

            if dyn_info.links.root_idx[I_l] == i_l_root and i_p != -1:
                func_add_safe_backward(
                    I_p, dyn_state.links.crb_inertial[i_l, i_b], dyn_state.links.crb_inertial, is_backward
                )
                func_add_safe_backward(I_p, dyn_state.links.crb_mass[i_l, i_b], dyn_state.links.crb_mass, is_backward)
                func_add_safe_backward(I_p, dyn_state.links.crb_pos[i_l, i_b], dyn_state.links.crb_pos, is_backward)
                func_add_safe_backward(I_p, dyn_state.links.crb_quat[i_l, i_b], dyn_state.links.crb_quat, is_backward)


@qd.func
def func_mass_mat_force(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Apply the composite inertia of one link to each of its own degrees of freedom."""
    is_awake = True
    if qd.static(rigid_config.use_hibernation):
        is_awake = not dyn_state.links.is_hibernated[i_l, i_b]
    if is_awake:
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

        for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
            dyn_state.dofs.f_ang[i_d, i_b], dyn_state.dofs.f_vel[i_d, i_b] = gu.inertial_mul(
                dyn_state.links.crb_pos[i_l, i_b],
                dyn_state.links.crb_inertial[i_l, i_b],
                dyn_state.links.crb_mass[i_l, i_b],
                dyn_state.dofs.cdof_vel[i_d, i_b],
                dyn_state.dofs.cdof_ang[i_d, i_b],
            )


@qd.func
def func_mass_mat_assemble_tree_cooperative(
    tid, i_t, i_b, dyn_state: array_class.DynState, rigid_info: array_class.RigidInfo, BLOCK_DIM: qd.template()
):
    """Write the share that one warp lane owns of the mass blocks of one kinematic tree.

    Every cell of the lower triangle, diagonal included, is computed once through the compressed pair index and
    written to both [i_d, j_d, i_b] and [j_d, i_d, i_b] right away, which saves the upper-triangle dot products that a
    two-pass path computes and then overwrites, and needs no mirror pass at all. Under the flipped mass_mat layout
    (i_d stride-1) the first write coalesces, and the strided second one costs about what the mirror pass it replaces
    used to cost.
    """
    # Assemble each mass block of the tree over its full lower triangle (see dofs_mass_block_start in array_class.py).
    # mass_parent_mask zeroes the within-block ancestor gaps.
    tree_dof_start = rigid_info.trees_dof_start[i_t]
    tree_dof_end = tree_dof_start + rigid_info.trees_n_dofs[i_t]
    for block_start in range(tree_dof_start, tree_dof_end):
        if rigid_info.dofs_mass_block_start[block_start] == block_start:
            block_end = rigid_info.dofs_mass_block_end[block_start]
            n_block_dofs = block_end - block_start
            n_lower_tri = n_block_dofs * (n_block_dofs + 1) // 2
            for i_chunk_ in range((n_lower_tri + BLOCK_DIM - 1) // BLOCK_DIM):
                i_pair = i_chunk_ * BLOCK_DIM + tid
                if i_pair < n_lower_tri:
                    # Compressed lower-tri-inclusive index: i_pair = i_d_ * (i_d_ + 1) / 2 + j_d_, with j_d_ in [0,
                    # i_d_]. The fast-math-robust inversion is required: a raw sqrt drops the j=0 entry of every
                    # perfect-square row on GPU, leaving M indefinite.
                    i_d_, j_d_ = linear_to_lower_tri(i_pair)
                    i_d = block_start + i_d_
                    j_d = block_start + j_d_
                    val = (
                        dyn_state.dofs.f_ang[i_d, i_b].dot(dyn_state.dofs.cdof_ang[j_d, i_b])
                        + dyn_state.dofs.f_vel[i_d, i_b].dot(dyn_state.dofs.cdof_vel[j_d, i_b])
                    ) * rigid_info.mass_parent_mask[i_d, j_d]
                    rigid_info.mass_mat[i_d, j_d, i_b] = val
                    if i_d_ != j_d_:
                        rigid_info.mass_mat[j_d, i_d, i_b] = val


@qd.func
def func_mass_mat_assemble_tree(
    i_t, i_b, dyn_state: array_class.DynState, rigid_info: array_class.RigidInfo, rigid_config: qd.template()
):
    """Write the mass blocks of one kinematic tree, then mirror them onto their upper triangle."""
    if func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config):
        # The blocks partition the dof range of the tree (see dofs_mass_block_start in array_class.py), iterated flat
        # with per-DOF block bounds since the mass matrix couples nothing across blocks.
        blocks_dof_start = rigid_info.trees_dof_start[i_t]
        blocks_dof_end = blocks_dof_start + rigid_info.trees_n_dofs[i_t]
        for i_d in range(blocks_dof_start, blocks_dof_end):
            for j_d in range(rigid_info.dofs_mass_block_start[i_d], rigid_info.dofs_mass_block_end[i_d]):
                rigid_info.mass_mat[i_d, j_d, i_b] = (
                    dyn_state.dofs.f_ang[i_d, i_b].dot(dyn_state.dofs.cdof_ang[j_d, i_b])
                    + dyn_state.dofs.f_vel[i_d, i_b].dot(dyn_state.dofs.cdof_vel[j_d, i_b])
                ) * rigid_info.mass_parent_mask[i_d, j_d]

        for i_d in range(blocks_dof_start, blocks_dof_end):
            for j_d in range(i_d + 1, rigid_info.dofs_mass_block_end[i_d]):
                rigid_info.mass_mat[i_d, j_d, i_b] = rigid_info.mass_mat[j_d, i_d, i_b]


@qd.func
def func_compute_mass_matrix(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    is_backward: qd.template(),
):
    """Assemble the mass matrix of every environment.

    func_compute_mass_matrix_masked assembles a subset of them instead, out of the same per-index functions, so the
    two implementations differ only in the environments they visit. The step advances every environment and takes this
    one: a selection reaches a kernel as an array argument, which has to be wrapped on every launch, and the step is
    launched twice per substep.
    """
    qd.loop_config(name="crb_initialize", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(dyn_state.links.pos.shape[0], dyn_state.links.pos.shape[1]):
        func_crb_initialize(i_l, i_b, dyn_state, rigid_info, rigid_config)

    qd.loop_config(name="crb", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
        func_crb_fold(i_r, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward)

    qd.loop_config(name="mass_mat", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(dyn_state.links.pos.shape[0], dyn_state.links.pos.shape[1]):
        func_mass_mat_force(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config)

    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        BLOCK_DIM = qd.static(32)
        n_trees = rigid_info.trees_root_idx.shape[0]
        qd.loop_config(name="mass_mat_assemble", block_dim=BLOCK_DIM)
        for i_flat in range(n_trees * dyn_state.links.pos.shape[1] * BLOCK_DIM):
            tid = i_flat % BLOCK_DIM
            i_tb = i_flat // BLOCK_DIM
            i_t = i_tb % n_trees
            i_b = i_tb // n_trees
            if func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config):
                func_mass_mat_assemble_tree_cooperative(tid, i_t, i_b, dyn_state, rigid_info, BLOCK_DIM)
    else:
        qd.loop_config(name="mass_mat_assemble", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], dyn_state.links.pos.shape[1]):
            func_mass_mat_assemble_tree(i_t, i_b, dyn_state, rigid_info, rigid_config)

    qd.loop_config(name="armature", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b in qd.ndrange(dyn_state.dofs.f_ang.shape[0], dyn_state.dofs.f_ang.shape[1]):
        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
        func_add_safe_backward((i_d, i_d, i_b), dyn_info.dofs.armature[I_d], rigid_info.mass_mat, is_backward)

    if qd.static(implicit_damping):
        qd.loop_config(name="impint_order_1_corr", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d, i_b in qd.ndrange(dyn_state.dofs.f_ang.shape[0], dyn_state.dofs.f_ang.shape[1]):
            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
            h = rigid_info.substep_dt[None]
            rigid_info.mass_mat[i_d, i_d, i_b] = rigid_info.mass_mat[i_d, i_d, i_b] + dyn_info.dofs.damping[I_d] * h
            if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                # qM += d qfrc_actuator / d qvel = -act_bias[2] * dt
                rigid_info.mass_mat[i_d, i_d, i_b] = (
                    rigid_info.mass_mat[i_d, i_d, i_b] - dyn_info.dofs.act_bias[I_d][2] * h
                )


@qd.func
def func_compute_mass_matrix_masked(
    envs_idx: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    is_backward: qd.template(),
):
    """Assemble the mass matrix of the given environments. See func_compute_mass_matrix."""
    qd.loop_config(name="crb_initialize", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in qd.ndrange(dyn_state.links.pos.shape[0], envs_idx.shape[0]):
        func_crb_initialize(i_l, envs_idx[i_b_], dyn_state, rigid_info, rigid_config)

    qd.loop_config(name="crb", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        func_crb_fold(i_r, envs_idx[i_b_], dyn_state, dyn_info, rigid_info, rigid_config, is_backward)

    qd.loop_config(name="mass_mat", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in qd.ndrange(dyn_state.links.pos.shape[0], envs_idx.shape[0]):
        func_mass_mat_force(i_l, envs_idx[i_b_], dyn_state, dyn_info, rigid_info, rigid_config)

    if qd.static(rigid_config.enable_cooperative_constraint_kernels):
        BLOCK_DIM = qd.static(32)
        n_trees = rigid_info.trees_root_idx.shape[0]
        qd.loop_config(name="mass_mat_assemble", block_dim=BLOCK_DIM)
        for i_flat in range(n_trees * envs_idx.shape[0] * BLOCK_DIM):
            tid = i_flat % BLOCK_DIM
            i_tb = i_flat // BLOCK_DIM
            i_t = i_tb % n_trees
            i_b = envs_idx[i_tb // n_trees]
            if func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config):
                func_mass_mat_assemble_tree_cooperative(tid, i_t, i_b, dyn_state, rigid_info, BLOCK_DIM)
    else:
        qd.loop_config(name="mass_mat_assemble", serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_t, i_b_ in qd.ndrange(rigid_info.trees_root_idx.shape[0], envs_idx.shape[0]):
            func_mass_mat_assemble_tree(i_t, envs_idx[i_b_], dyn_state, rigid_info, rigid_config)

    qd.loop_config(name="armature", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d, i_b_ in qd.ndrange(dyn_state.dofs.f_ang.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
        func_add_safe_backward((i_d, i_d, i_b), dyn_info.dofs.armature[I_d], rigid_info.mass_mat, is_backward)

    if qd.static(implicit_damping):
        qd.loop_config(name="impint_order_1_corr", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d, i_b_ in qd.ndrange(dyn_state.dofs.f_ang.shape[0], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
            h = rigid_info.substep_dt[None]
            rigid_info.mass_mat[i_d, i_d, i_b] = rigid_info.mass_mat[i_d, i_d, i_b] + dyn_info.dofs.damping[I_d] * h
            if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                # qM += d qfrc_actuator / d qvel = -act_bias[2] * dt
                rigid_info.mass_mat[i_d, i_d, i_b] = (
                    rigid_info.mass_mat[i_d, i_d, i_b] - dyn_info.dofs.act_bias[I_d][2] * h
                )


@qd.func
def func_has_implicit_damping_tree(
    i_t,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Whether the mass factor of kinematic tree i_t of env i_b takes an implicit damping term.

    A dof with joint damping adds one, and so does, under the implicitfast integrator, a velocity-controlled dof whose
    actuator bias damps it. The tree factors as a whole (its blocks partition its dofs), so a damped dof anywhere in it,
    in an attached child entity included, gives the whole tree its damped factor. A tree without any keeps its smooth
    factor through the implicit damping pass.
    """
    EPS = rigid_info.EPS[None]
    has_damping = False
    dof_start = rigid_info.trees_dof_start[i_t]
    for i_d in range(dof_start, dof_start + rigid_info.trees_n_dofs[i_t]):
        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
        if dyn_info.dofs.damping[I_d] > EPS:
            has_damping = True
        if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
            if (
                dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY
                and qd.abs(dyn_info.dofs.act_bias[I_d][2]) > EPS
            ):
                has_damping = True
    return has_damping


@qd.func
def func_factor_mass_tree_tiled(
    tid,
    i_t,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    TileCls: qd.template(),
):
    """Factor the mass blocks of one kinematic tree by streaming tiles through registers, one warp lane's share of it.

    See func_factor_mass_tiled for how the factor the tile primitive produces maps to the one the mass solve
    consumes.
    """
    T = qd.static(rigid_config.cholesky_tile_size)
    EPS = rigid_info.EPS[None]

    # Under implicit damping only the trees carrying a damping term take a new factor, the others keep the smooth one.
    # One lane reads the dofs of the tree, the whole block takes its answer.
    is_factored = func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config)
    if qd.static(implicit_damping):
        has_damping = 0
        if tid == 0 and is_factored:
            if func_has_implicit_damping_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config):
                has_damping = 1
        is_factored = qd.simt.subgroup.broadcast(has_damping, qd.u32(0)) != 0
    if is_factored:
        # Each block owns its own scratch region [block_start, block_end), disjoint across trees too, so the scatter
        # stays race-free.
        tree_dof_start = rigid_info.trees_dof_start[i_t]
        tree_dof_end = tree_dof_start + rigid_info.trees_n_dofs[i_t]
        for block_start in range(tree_dof_start, tree_dof_end):
            if rigid_info.dofs_mass_block_start[block_start] == block_start:
                block_end = rigid_info.dofs_mass_block_end[block_start]
                n_block_dofs = block_end - block_start
                n_blocks = (n_block_dofs + T - 1) // T

                # Phase 1: copy the reverse-indexed symmetric M block (+ implicit damping) into the scratch workspace.
                # mass_mat stores the lower triangle of M, so M[ri_, rj_] with ri_ <= rj_ is read from M[rj_, ri_].
                for i_chunk_ in range((n_block_dofs + T - 1) // T):
                    i_d_ = i_chunk_ * T + tid
                    if i_d_ < n_block_dofs:
                        ri_ = n_block_dofs - 1 - i_d_
                        for j_d_ in range(i_d_ + 1):
                            rj_ = n_block_dofs - 1 - j_d_  # i_d_ >= j_d_  =>  ri_ <= rj_
                            m = rigid_info.mass_mat[block_start + rj_, block_start + ri_, i_b]
                            rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + j_d_] = m
                            rigid_info.mass_mat_tiled_scratch[i_b, block_start + j_d_, block_start + i_d_] = m
                        if qd.static(implicit_damping):
                            # Reverse-diagonal slot i_d_ holds M[ri_, ri_]; damping/act_bias index the original DOF.
                            i_d = block_start + ri_
                            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                            rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + i_d_] = (
                                rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + i_d_]
                                + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                            )
                            if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                                if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                    rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + i_d_] = (
                                        rigid_info.mass_mat_tiled_scratch[i_b, block_start + i_d_, block_start + i_d_]
                                        - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                                    )
                qd.simt.block.sync()

                # Phase 2: blocked Cholesky G_rev G_rev^T = M_rev in the scratch workspace, as the constraint Hessian's
                # func_cholesky_factor_direct_tiled does. The tile ops are warp-synchronous, so the loop needs no sync.
                for kb in range(n_blocks):
                    k0 = block_start + kb * T
                    k1 = qd.min(k0 + T, block_end)

                    L_kk = TileCls.eye(dtype=gs.qd_float)  # rows past n_block_dofs stay identity
                    L_kk[:] = rigid_info.mass_mat_tiled_scratch[i_b, k0:k1, k0:k1]
                    for jb in range(kb):
                        j0 = block_start + jb * T
                        for t in range(T):
                            v = rigid_info.mass_mat_tiled_scratch[i_b, k0:k1, j0 + t]
                            L_kk -= qd.outer(v, v)
                    # Floor each pivot relative to the row's original diagonal: a pivot carries the inertia of its DOF,
                    # which falls with the fifth power of the body size, so an absolute floor takes over the
                    # factorization below unit scale. The diagonal read here is still the original one - the
                    # left-looking factor writes this block only after the call - and 'eps' is read per lane, so a
                    # per-lane value costs nothing.
                    d_row = k0 + tid
                    diag_orig = gs.qd_float(1.0)
                    if d_row < k1:
                        diag_orig = rigid_info.mass_mat_tiled_scratch[i_b, d_row, d_row]
                    L_kk.cholesky_(EPS * qd.max(diag_orig, EPS))

                    for ib in range(kb + 1, n_blocks):
                        i0 = block_start + ib * T
                        i1 = qd.min(i0 + T, block_end)

                        L_ik = TileCls.zeros(dtype=gs.qd_float)
                        L_ik[:] = rigid_info.mass_mat_tiled_scratch[i_b, i0:i1, k0:k1]
                        for jb in range(kb):
                            j0 = block_start + jb * T
                            for t in range(T):
                                v_own = rigid_info.mass_mat_tiled_scratch[i_b, i0:i1, j0 + t]
                                v_diag = rigid_info.mass_mat_tiled_scratch[i_b, k0:k1, j0 + t]
                                L_ik -= qd.outer(v_own, v_diag)
                        L_kk.solve_triangular_(L_ik)
                        rigid_info.mass_mat_tiled_scratch[i_b, i0:i1, k0:k1] = L_ik

                    rigid_info.mass_mat_tiled_scratch[i_b, k0:k1, k0:k1] = L_kk
                qd.simt.block.sync()

                # Phase 3: scatter the LTDL factor of M from G_rev (scratch) into canonical mass_mat_L / mass_mat_D_inv.
                # Reads the scratch, writes the distinct mass_mat_L (no in-place hazard). Only the strict-lower triangle
                # and unit diagonal are meaningful to the solve, the upper triangle is left untouched.
                n_strict_lower = n_block_dofs * (n_block_dofs - 1) // 2
                for i_chunk_ in range((n_strict_lower + T - 1) // T):
                    i_pair = i_chunk_ * T + tid
                    if i_pair < n_strict_lower:
                        i_d_, j_d_ = linear_to_lower_tri(i_pair, strict=True)
                        ri_ = n_block_dofs - 1 - i_d_
                        rj_ = n_block_dofs - 1 - j_d_  # i_d_ > j_d_  =>  rj_ > ri_  (a lower G_rev entry)
                        g_num = rigid_info.mass_mat_tiled_scratch[i_b, block_start + rj_, block_start + ri_]
                        g_den = rigid_info.mass_mat_tiled_scratch[i_b, block_start + ri_, block_start + ri_]
                        rigid_info.mass_mat_L[block_start + i_d_, block_start + j_d_, i_b] = g_num / g_den

                for i_chunk_ in range((n_block_dofs + T - 1) // T):
                    i_d_ = i_chunk_ * T + tid
                    if i_d_ < n_block_dofs:
                        ri_ = n_block_dofs - 1 - i_d_
                        g_den = rigid_info.mass_mat_tiled_scratch[i_b, block_start + ri_, block_start + ri_]
                        rigid_info.mass_mat_D_inv[block_start + i_d_, i_b] = 1.0 / (g_den * g_den)
                        rigid_info.mass_mat_L[block_start + i_d_, block_start + i_d_, i_b] = 1.0
                qd.simt.block.sync()


@qd.func
def func_factor_mass_tiled(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    TileCls: qd.template(),
):
    """Factor the mass matrix of every environment by streaming tiles through registers (GPU forward only).

    Runs when the mass submatrix of a tree is too large for the shared memory the cooperative factor needs. M is
    block-diagonal per mass block (see dofs_mass_block_start in array_class.py), so one warp of T lanes factors each
    block of the tree independently through the same qd.simt.TileNxN blocked Cholesky as the constraint Hessian.

    func_solve_mass_block consumes the LTDL form M = L^T D L (L unit-lower), which comes of eliminating DOFs
    last-to-first,
    whereas the tile primitive produces the forward Cholesky M = G G^T. So what gets factored is the reverse-indexed
    matrix of each block, M_rev[a, b] = M[n-1-a, n-1-b] with n the block's DOF count, and its factor maps back to the
    LTDL factor of the block as:
      L[i,j] = G_rev[n-1-j, n-1-i] / G_rev[n-1-i, n-1-i]  (i > j),  D_inv[i] = 1 / G_rev[n-1-i, n-1-i]^2,  diag(L) = 1.

    The qd.simt tile ops are batch-first while mass_mat_L is canonical batch-last (n_dofs, n_dofs, _B), so the
    factorization runs in each block's region of the batch-first scratch rigid_info.mass_mat_tiled_scratch and is
    scattered into mass_mat_L / mass_mat_D_inv. To avoid a dedicated allocation, that scratch aliases the constraint
    Hessian buffer nt_H (same shape, and free at mass-factor time since the constraint solve only populates it later in
    the step); see get_constraint_state. The scratch and mass_mat_L are distinct buffers, so the scatter is race-free.

    func_factor_mass_tiled_masked does the same for a subset of the environments.
    """
    # Reuse the Hessian's tile width; TileCls is dispatched to match it at the call site, so T and the tile class stay
    # consistent for either value. In practice this path only runs for mass blocks exceeding shared memory (total
    # n_dofs > 48), where the rule lands on 32.
    T = qd.static(rigid_config.cholesky_tile_size)

    n_trees = rigid_info.trees_root_idx.shape[0]
    _B = rigid_info.mass_mat.shape[2]

    qd.loop_config(name="factor_mass", block_dim=T)
    for i in range(n_trees * _B * T):
        tid = i % T
        i_t = (i // T) % n_trees
        i_b = i // (T * n_trees)
        func_factor_mass_tree_tiled(
            tid, i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping, TileCls
        )


@qd.func
def func_factor_mass_tiled_masked(
    envs_idx: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    TileCls: qd.template(),
):
    """Factor the mass matrix of the given environments by streaming tiles. See func_factor_mass_tiled."""
    T = qd.static(rigid_config.cholesky_tile_size)

    n_trees = rigid_info.trees_root_idx.shape[0]
    _B = envs_idx.shape[0]

    qd.loop_config(name="factor_mass", block_dim=T)
    for i in range(n_trees * _B * T):
        tid = i % T
        i_t = (i // T) % n_trees
        i_b = envs_idx[i // (T * n_trees)]
        func_factor_mass_tree_tiled(
            tid, i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping, TileCls
        )


@qd.func
def func_factor_mass_tree_global(
    tid,
    i_t,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    BLOCK_DIM: qd.template(),
):
    """Factor the mass blocks of one kinematic tree in place in global memory, one thread's share of it.

    Each elimination step snapshots the pivot row into a small shared vector, O(n_dofs) rather than O(n_dofs^2),
    before the trailing submatrix is updated, so the parallel per-row updates only ever read the pivot row and the
    result holds whatever the scheduling. Gives the same numbers as func_factor_mass_tree, which runs the same
    elimination on a single thread.
    """
    MAX_DOFS_PER_BLOCK = qd.static(rigid_config.tiled_n_dofs_per_block)

    # Under implicit damping only the trees carrying a damping term take a new factor, the others keep the smooth one.
    # One lane reads the dofs of the tree, the whole block takes its answer.
    is_factored = func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config)
    if qd.static(implicit_damping):
        has_damping = 0
        if tid == 0 and is_factored:
            if func_has_implicit_damping_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config):
                has_damping = 1
        is_factored = qd.simt.subgroup.broadcast(has_damping, qd.u32(0)) != 0
    if is_factored:
        pivot_row = qd.simt.block.SharedArray((MAX_DOFS_PER_BLOCK,), gs.qd_float)

        # Factor each mass block of the tree in place in global memory, block-relative so shared indices stay >= 0
        tree_dof_start = rigid_info.trees_dof_start[i_t]
        tree_dof_end = tree_dof_start + rigid_info.trees_n_dofs[i_t]
        for block_start in range(tree_dof_start, tree_dof_end):
            if rigid_info.dofs_mass_block_start[block_start] == block_start:
                block_end = rigid_info.dofs_mass_block_end[block_start]
                n_block_dofs = block_end - block_start

                # Copy the block's lower triangle into mass_mat_L (+ implicit damping on the diagonal), cooperatively.
                # Restricting to the block makes the factorization cost the sum of per-block cubes instead of the cube
                # of the whole tree.
                for i_chunk_ in range((n_block_dofs + BLOCK_DIM - 1) // BLOCK_DIM):
                    i_d_ = i_chunk_ * BLOCK_DIM + tid
                    if i_d_ < n_block_dofs:
                        i_d = block_start + i_d_
                        for j_d in range(block_start, i_d + 1):
                            rigid_info.mass_mat_L[i_d, j_d, i_b] = rigid_info.mass_mat[i_d, j_d, i_b]
                        if qd.static(implicit_damping):
                            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                            rigid_info.mass_mat_L[i_d, i_d, i_b] = (
                                rigid_info.mass_mat_L[i_d, i_d, i_b]
                                + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                            )
                            if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                                if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                    rigid_info.mass_mat_L[i_d, i_d, i_b] = (
                                        rigid_info.mass_mat_L[i_d, i_d, i_b]
                                        - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                                    )
                qd.simt.block.sync()

                # In-place LDL^T, eliminating dofs from last to first (matches func_factor_mass_tree).
                for j in range(n_block_dofs):
                    i_d = block_end - j - 1
                    i_d_local = i_d - block_start
                    D_inv = 1.0 / rigid_info.mass_mat_L[i_d, i_d, i_b]
                    if tid == 0:
                        rigid_info.mass_mat_D_inv[i_d, i_b] = D_inv

                    # Phase A: snapshot the (Schur-updated) pivot-row entries below the diagonal into shared.
                    for i_chunk_ in range((i_d_local + BLOCK_DIM - 1) // BLOCK_DIM):
                        j_d_ = i_chunk_ * BLOCK_DIM + tid
                        if j_d_ < i_d_local:
                            pivot_row[j_d_] = rigid_info.mass_mat_L[i_d, block_start + j_d_, i_b]
                    qd.simt.block.sync()

                    # Phase B: each lane eliminates one column j_d, updating its own row j_d of the trailing submatrix
                    # from the read-only snapshot. Distinct rows per lane => no write conflicts, and the pivot row is
                    # only read (from shared) => no read/write race on row i_d.
                    for i_chunk_ in range((i_d_local + BLOCK_DIM - 1) // BLOCK_DIM):
                        j_d_ = i_chunk_ * BLOCK_DIM + tid
                        if j_d_ < i_d_local:
                            a = pivot_row[j_d_] * D_inv
                            j_d = block_start + j_d_
                            for k_d_ in range(j_d_ + 1):
                                rigid_info.mass_mat_L[j_d, block_start + k_d_, i_b] = (
                                    rigid_info.mass_mat_L[j_d, block_start + k_d_, i_b] - a * pivot_row[k_d_]
                                )
                            rigid_info.mass_mat_L[i_d, j_d, i_b] = a
                    qd.simt.block.sync()

                    # Diagonal coeffs of L are ignored downstream, and set to 1.0 to match the other paths.
                    if tid == 0:
                        rigid_info.mass_mat_L[i_d, i_d, i_b] = 1.0


@qd.func
def func_factor_mass_tree(
    i_t,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
):
    """Factor the mass blocks of one kinematic tree on a single thread."""
    # Under implicit damping only the trees carrying a damping term take a new factor, the others keep the smooth one
    is_factored = func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config)
    if qd.static(implicit_damping):
        if is_factored:
            is_factored = func_has_implicit_damping_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config)
    if is_factored:
        # The blocks partition the dof range of the tree (see dofs_mass_block_start in array_class.py), iterated flat
        # with per-DOF block bounds: elimination never leaves a block, so one descending scan over the independent
        # blocks stays exact.
        blocks_dof_start = rigid_info.trees_dof_start[i_t]
        blocks_dof_end = blocks_dof_start + rigid_info.trees_n_dofs[i_t]
        for i_d in range(blocks_dof_start, blocks_dof_end):
            for j_d in range(rigid_info.dofs_mass_block_start[i_d], i_d + 1):
                rigid_info.mass_mat_L[i_d, j_d, i_b] = rigid_info.mass_mat[i_d, j_d, i_b]

            if qd.static(implicit_damping):
                I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                rigid_info.mass_mat_L[i_d, i_d, i_b] = (
                    rigid_info.mass_mat_L[i_d, i_d, i_b] + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                )
                if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                    if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                        rigid_info.mass_mat_L[i_d, i_d, i_b] = (
                            rigid_info.mass_mat_L[i_d, i_d, i_b]
                            - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                        )

        for i_d_ in range(blocks_dof_end - blocks_dof_start):
            i_d = blocks_dof_end - i_d_ - 1
            block_start = rigid_info.dofs_mass_block_start[i_d]
            D_inv = 1.0 / rigid_info.mass_mat_L[i_d, i_d, i_b]
            rigid_info.mass_mat_D_inv[i_d, i_b] = D_inv

            for j_d_ in range(i_d - block_start):
                j_d = i_d - j_d_ - 1
                a = rigid_info.mass_mat_L[i_d, j_d, i_b] * D_inv
                for k_d in range(block_start, j_d + 1):
                    rigid_info.mass_mat_L[j_d, k_d, i_b] -= a * rigid_info.mass_mat_L[i_d, k_d, i_b]
                rigid_info.mass_mat_L[i_d, j_d, i_b] = a

            # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
            rigid_info.mass_mat_L[i_d, i_d, i_b] = 1.0


@qd.func
def func_factor_mass_tree_shared(
    tid,
    i_t,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
    BLOCK_DIM: qd.template(),
):
    """Factor the mass blocks of one kinematic tree in shared memory, one thread's share of it."""
    MAX_DOFS_PER_BLOCK = qd.static(rigid_config.tiled_n_dofs_per_block)
    WARP_SIZE = qd.static(32)

    # Under implicit damping only the trees carrying a damping term take a new factor, the others keep the smooth one.
    # One lane reads the dofs of the tree, the whole block takes its answer.
    is_factored = func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config)
    if qd.static(implicit_damping):
        has_damping = 0
        if tid == 0 and is_factored:
            if func_has_implicit_damping_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config):
                has_damping = 1
        is_factored = qd.simt.subgroup.broadcast(has_damping, qd.u32(0)) != 0
    if is_factored:
        mass_mat = qd.simt.block.SharedArray((MAX_DOFS_PER_BLOCK, MAX_DOFS_PER_BLOCK + 1), gs.qd_float)

        # Factor each mass block of the tree in shared memory, indexed block-relative so shared indices stay >= 0
        tree_dof_start = rigid_info.trees_dof_start[i_t]
        tree_dof_end = tree_dof_start + rigid_info.trees_n_dofs[i_t]
        for block_start in range(tree_dof_start, tree_dof_end):
            if rigid_info.dofs_mass_block_start[block_start] == block_start:
                block_end = rigid_info.dofs_mass_block_end[block_start]
                n_block_dofs = block_end - block_start
                n_lower_tri = n_block_dofs * (n_block_dofs + 1) // 2

                for i_chunk_ in range((n_lower_tri + BLOCK_DIM - 1) // BLOCK_DIM):
                    i_pair = i_chunk_ * BLOCK_DIM + tid
                    if i_pair < n_lower_tri:
                        i_d_, j_d_ = linear_to_lower_tri(i_pair)
                        mass_mat[i_d_, j_d_] = rigid_info.mass_mat[block_start + i_d_, block_start + j_d_, i_b]
                qd.simt.block.sync()

                if qd.static(implicit_damping):
                    for i_chunk_ in range((n_block_dofs + BLOCK_DIM - 1) // BLOCK_DIM):
                        i_d_ = i_chunk_ * BLOCK_DIM + tid
                        if i_d_ < n_block_dofs:
                            i_d = block_start + i_d_
                            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                            mass_mat[i_d_, i_d_] = (
                                mass_mat[i_d_, i_d_] + dyn_info.dofs.damping[I_d] * rigid_info.substep_dt[None]
                            )
                            if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                                if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                                    mass_mat[i_d_, i_d_] = (
                                        mass_mat[i_d_, i_d_]
                                        - dyn_info.dofs.act_bias[I_d][2] * rigid_info.substep_dt[None]
                                    )
                    qd.simt.block.sync()

                for j in range(n_block_dofs):
                    i_d_ = n_block_dofs - j - 1
                    i_d = block_end - j - 1

                    D_inv = 1.0 / mass_mat[i_d_, i_d_]
                    if tid == 0:
                        rigid_info.mass_mat_D_inv[i_d, i_b] = D_inv
                        # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                        rigid_info.mass_mat_L[i_d, i_d, i_b] = 1.0

                    for i_chunk_ in range((i_d_ + BLOCK_DIM - 1) // BLOCK_DIM):
                        j_d_ = i_d_ - 1 - (i_chunk_ * BLOCK_DIM + tid)
                        if j_d_ >= 0:
                            a = mass_mat[i_d_, j_d_] * D_inv
                            for k_d in range(j_d_ + 1):
                                mass_mat[j_d_, k_d] = mass_mat[j_d_, k_d] - a * mass_mat[i_d_, k_d]
                            mass_mat[i_d_, j_d_] = a
                    if qd.static(rigid_config.backend == gs.cuda):
                        if i_d_ <= WARP_SIZE:
                            qd.simt.warp.sync(qd.u32(0xFFFFFFFF))
                        else:
                            qd.simt.block.sync()
                    else:
                        qd.simt.block.sync()

                n_strict_lower_tri = n_block_dofs * (n_block_dofs - 1) // 2
                for i_chunk_ in range((n_strict_lower_tri + BLOCK_DIM - 1) // BLOCK_DIM):
                    i_pair = i_chunk_ * BLOCK_DIM + tid
                    if i_pair < n_strict_lower_tri:
                        i_d_, j_d_ = linear_to_lower_tri(i_pair, strict=True)
                        rigid_info.mass_mat_L[block_start + i_d_, block_start + j_d_, i_b] = mass_mat[i_d_, j_d_]
                qd.simt.block.sync()


@qd.func
def func_factor_mass(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
):
    """Factor the mass matrix of every environment.

    func_factor_mass_masked factors a subset of them instead, out of the same per-tree functions. See
    func_compute_mass_matrix for why both passes come in two implementations.
    """
    n_trees = rigid_info.trees_root_idx.shape[0]
    _B = rigid_info.mass_mat.shape[2]

    if qd.static(rigid_config.enable_register_tiled_mass):
        # Register-streaming tiled per-tree factor for the >shared-cap path (same primitive as the constraint Hessian).
        # Implies enable_tiled_cholesky_mass_matrix and not mass_matrix_fits_shared; see func_factor_mass_tiled.
        # Replaces the cooperative LDL^T in the elif below.
        func_factor_mass_tiled(
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            implicit_damping,
            qd.simt.Tile32x32 if qd.static(rigid_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
        )
    elif qd.static(rigid_config.enable_tiled_cholesky_mass_matrix and not rigid_config.mass_matrix_fits_shared):
        # Uncapped cooperative per-tree LDL^T, for a tree submatrix that does not fit in shared memory.
        BLOCK_DIM = qd.static(32)
        qd.loop_config(name="factor_mass", block_dim=BLOCK_DIM)
        for i in range(n_trees * _B * BLOCK_DIM):
            tid = i % BLOCK_DIM
            i_t = (i // BLOCK_DIM) % n_trees
            i_b = i // (BLOCK_DIM * n_trees)
            func_factor_mass_tree_global(
                tid, i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping, BLOCK_DIM
            )
    elif qd.static(not rigid_config.enable_tiled_cholesky_mass_matrix or rigid_config.backend == gs.cpu):
        qd.loop_config(name="factor_mass", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_t, i_b in qd.ndrange(n_trees, _B):
            func_factor_mass_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping)
    else:
        BLOCK_DIM = qd.static(32)
        qd.loop_config(name="factor_mass", block_dim=BLOCK_DIM)
        for i in range(n_trees * _B * BLOCK_DIM):
            tid = i % BLOCK_DIM
            i_t = (i // BLOCK_DIM) % n_trees
            i_b = i // (BLOCK_DIM * n_trees)
            func_factor_mass_tree_shared(
                tid, i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping, BLOCK_DIM
            )


@qd.func
def func_factor_mass_masked(
    envs_idx: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    implicit_damping: qd.template(),
):
    """Factor the mass matrix of the given environments. See func_factor_mass."""
    n_trees = rigid_info.trees_root_idx.shape[0]
    _B = envs_idx.shape[0]

    if qd.static(rigid_config.enable_register_tiled_mass):
        func_factor_mass_tiled_masked(
            envs_idx,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            implicit_damping,
            qd.simt.Tile32x32 if qd.static(rigid_config.cholesky_tile_size == 32) else qd.simt.Tile16x16,
        )
    elif qd.static(rigid_config.enable_tiled_cholesky_mass_matrix and not rigid_config.mass_matrix_fits_shared):
        BLOCK_DIM = qd.static(32)
        qd.loop_config(name="factor_mass", block_dim=BLOCK_DIM)
        for i in range(n_trees * _B * BLOCK_DIM):
            tid = i % BLOCK_DIM
            i_t = (i // BLOCK_DIM) % n_trees
            i_b = envs_idx[i // (BLOCK_DIM * n_trees)]
            func_factor_mass_tree_global(
                tid, i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping, BLOCK_DIM
            )
    elif qd.static(not rigid_config.enable_tiled_cholesky_mass_matrix or rigid_config.backend == gs.cpu):
        qd.loop_config(name="factor_mass", serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_t, i_b_ in qd.ndrange(n_trees, _B):
            func_factor_mass_tree(i_t, envs_idx[i_b_], dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping)
    else:
        BLOCK_DIM = qd.static(32)
        qd.loop_config(name="factor_mass", block_dim=BLOCK_DIM)
        for i in range(n_trees * _B * BLOCK_DIM):
            tid = i % BLOCK_DIM
            i_t = (i // BLOCK_DIM) % n_trees
            i_b = envs_idx[i // (BLOCK_DIM * n_trees)]
            func_factor_mass_tree_shared(
                tid, i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping, BLOCK_DIM
            )


@qd.func
def func_solve_mass_block(
    i_b: qd.int32,
    blocks_dof_start: qd.int32,
    blocks_dof_end: qd.int32,
    vec: qd.Tensor,
    out: qd.Tensor,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Solve M x = y for the degrees of freedom in [blocks_dof_start, blocks_dof_end), leaving the other rows as is.

    Both bounds must fall on a mass-block boundary (see dofs_mass_block_start in array_class.py), which holds for the
    degrees of freedom of a kinematic chain, so the substitutions never read a row outside the range. Restricting the
    range is what lets several chains share one pair of buffers: writing a row outside it would race with the chain
    that owns it.
    """
    # Step 1: Solve w st. L^T @ w = y. Reading out[j_d] (j_d > i_d) from the buffer being written is safe: those entries
    # were finalized in earlier (larger i_d) iterations. This func is never auto-reversed: the backward pass seeds
    # mass_mat.grad directly via the implicit function theorem (see kernel_manual_compute_qacc_bw in manual_bw.py).
    for i_d_ in range(blocks_dof_end - blocks_dof_start):
        i_d = blocks_dof_end - i_d_ - 1
        block_end = rigid_info.dofs_mass_block_end[i_d]
        curr_out = vec[i_d, i_b]
        for j_d in range(i_d + 1, block_end):
            curr_out = curr_out - rigid_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]
        out[i_d, i_b] = curr_out

    # Step 2: z = D^{-1} w
    for i_d in range(blocks_dof_start, blocks_dof_end):
        out[i_d, i_b] = out[i_d, i_b] * rigid_info.mass_mat_D_inv[i_d, i_b]

    # Step 3: Solve x st. L @ x = z
    for i_d in range(blocks_dof_start, blocks_dof_end):
        block_start = rigid_info.dofs_mass_block_start[i_d]
        curr_out = out[i_d, i_b]
        for j_d in range(block_start, i_d):
            curr_out = curr_out - rigid_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]
        out[i_d, i_b] = curr_out


@qd.func
def func_solve_mass_batch(
    i_b: qd.int32,
    vec: qd.Tensor,
    out: qd.Tensor,
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    # A block is the smallest set of coupled degrees of freedom, and one worker's whole share of the work. The degree of
    # freedom starting a block is its root, and tells whether the block's kinematic tree is hibernated, in which case
    # there is nothing to solve.
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d in range(rigid_info.mass_mat.shape[0]):
        is_hibernated = dyn_state.dofs.is_hibernated[i_d, i_b] if qd.static(rigid_config.use_hibernation) else False
        if rigid_info.dofs_mass_block_start[i_d] == i_d and not is_hibernated:
            block_end = rigid_info.dofs_mass_block_end[i_d]
            func_solve_mass_block(i_b, i_d, block_end, vec, out, rigid_info, rigid_config)


@qd.func
def func_enter_neutral_configuration(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Evaluate forward kinematics at the neutral configuration `qpos0`, and factorize the mass matrix there.

    The inverse weights are defined at the neutral configuration, and computed from the mass matrix without its
    implicit damping term, which has to be factorized (LDL^T) before it can be solved against.

    Only the environments in `envs_idx` are assembled and factorized, hence the masked implementation of those passes.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_update_cartesian_space_root(
            i_l_root,
            i_b,
            rigid_info.qpos0,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            force_update_all_geoms=False,
            is_backward=False,
        )

    func_compute_mass_matrix_masked(
        envs_idx, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False, is_backward=False
    )
    func_factor_mass_masked(envs_idx, dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=False)


@qd.func
def func_exit_neutral_configuration(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Evaluate forward kinematics again at the configuration the scene is in, which the neutral pass overwrote."""
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_update_cartesian_space_root(
            i_l_root,
            i_b,
            rigid_info.qpos,
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            force_update_all_geoms=True,
            is_backward=False,
        )


@qd.func
def func_init_meaninertia(envs_idx: qd.types.ndarray(), rigid_info: array_class.RigidInfo, rigid_config: qd.template()):
    """Compute the mean diagonal entry of the joint-space mass matrix, which scales the solver tolerances."""
    n_dofs = rigid_info.mass_mat.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        if n_dofs > 0:
            # Accumulated through the field rather than a local: a register would carry more precision than the
            # working one, and every consumer of the mean inertia is quoted on the value the field holds.
            rigid_info.meaninertia[i_b] = 0.0
            for i_d in range(n_dofs):
                rigid_info.meaninertia[i_b] = rigid_info.meaninertia[i_b] + rigid_info.mass_mat[i_d, i_d, i_b]
            rigid_info.meaninertia[i_b] = rigid_info.meaninertia[i_b] / n_dofs
        else:
            rigid_info.meaninertia[i_b] = 1.0


@qd.func
def func_init_link_invweight(
    i_l,
    i_b,
    jac_row: qd.Tensor,
    solve_out: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update: qd.template(),
):
    """Compute the two constraint inverse weights of one link, from the mass matrix at the neutral configuration.

    Each weight is the mean diagonal entry of `J M^-1 J^T` over three rows of the Jacobian of the link's center of
    mass: the translational rows for the first weight, the rotational rows for the second. Those rows are built from
    the motion subspace of every DOF between the link and the root of its tree. One row is computed at a time, and
    every operation is restricted to the mass blocks that chain spans, so the cost is that of one tree; forming M^-1
    explicitly would cost n_dofs^2 per environment instead.
    """
    EPS = rigid_info.EPS[None]
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    n_dofs = rigid_info.mass_mat.shape[0]
    dof_min = n_dofs
    dof_max = -1
    j_l = i_l
    while j_l != -1:
        J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
        if dyn_info.links.dof_end[J_l] > dyn_info.links.dof_start[J_l]:
            dof_min = qd.min(dof_min, dyn_info.links.dof_start[J_l])
            dof_max = qd.max(dof_max, dyn_info.links.dof_end[J_l] - 1)
        j_l = dyn_info.links.parent_idx[J_l]

    # A link no DOF can move keeps a zero weight, its Jacobian being zero everywhere.
    dof_start = gs.qd_int(0)
    dof_end = gs.qd_int(0)
    if dof_max >= 0:
        dof_start = rigid_info.dofs_mass_block_start[dof_min]
        dof_end = rigid_info.dofs_mass_block_end[dof_max]

    for i_c in qd.static(range(2)):
        if force_update or dyn_info.links.invweight[I_l][i_c] < EPS:
            offset = dyn_state.links.i_pos[i_l, i_b]
            weight = 0.0
            for i_r in qd.static(range(3)):
                for i_d in range(dof_start, dof_end):
                    jac_row[i_d, i_b] = 0.0
                j_l = i_l
                while j_l != -1:
                    J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
                    for i_d in range(dyn_info.links.dof_start[J_l], dyn_info.links.dof_end[J_l]):
                        if qd.static(i_c == 0):
                            cdof_lin = dyn_state.dofs.cdof_vel[i_d, i_b] + dyn_state.dofs.cdof_ang[i_d, i_b].cross(
                                offset
                            )
                            jac_row[i_d, i_b] = cdof_lin[i_r]
                        else:
                            jac_row[i_d, i_b] = dyn_state.dofs.cdof_ang[i_d, i_b][i_r]
                    j_l = dyn_info.links.parent_idx[J_l]
                func_solve_mass_block(i_b, dof_start, dof_end, jac_row, solve_out, rigid_info, rigid_config)
                for i_d in range(dof_start, dof_end):
                    weight = weight + jac_row[i_d, i_b] * solve_out[i_d, i_b]

            # A zero component is kept as is: substituting the other component to avoid degenerate constraint
            # weights over-stiffens the impedance of constraints on such links (e.g. connect constraints on a link
            # whose frame origin cannot translate) and destabilizes them.
            dyn_info.links.invweight[I_l][i_c] = weight / 3.0


@qd.func
def func_init_dofs_invweight(
    i_l,
    i_b,
    jac_row: qd.Tensor,
    solve_out: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update: qd.template(),
):
    """Compute the constraint inverse weight of every DOF of the joints of one link, at the neutral configuration.

    The Jacobian row of a DOF is the unit row, so `J M^-1 J^T` reduces to one diagonal entry of the inverse mass
    matrix. The entries are averaged over each group of DOFs a joint moves together: the translation and the rotation
    of a free joint are two such groups, a spherical joint is one, and every other movable joint has one DOF per group.
    """
    EPS = rigid_info.EPS[None]
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    for i_j in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        joint_type = dyn_info.joints.type[I_j]
        if joint_type != gs.JOINT_TYPE.FIXED:
            joint_dof_start = dyn_info.joints.dof_start[I_j]
            n_groups = gs.qd_int(1)
            group_n_dofs = gs.qd_int(1)
            if joint_type == gs.JOINT_TYPE.FREE:
                n_groups = 2
                group_n_dofs = 3
            elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                group_n_dofs = 3

            for i_group in range(n_groups):
                group_dof_start = joint_dof_start + i_group * group_n_dofs
                is_update_required = False
                for i_d_ in range(group_n_dofs):
                    i_d = group_dof_start + i_d_
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                    if force_update or dyn_info.dofs.invweight[I_d] < EPS:
                        is_update_required = True

                if is_update_required:
                    weight = 0.0
                    for i_d_ in range(group_n_dofs):
                        i_d = group_dof_start + i_d_
                        block_start = rigid_info.dofs_mass_block_start[i_d]
                        block_end = rigid_info.dofs_mass_block_end[i_d]
                        for j_d in range(block_start, block_end):
                            jac_row[j_d, i_b] = 0.0
                        jac_row[i_d, i_b] = 1.0
                        func_solve_mass_block(i_b, block_start, block_end, jac_row, solve_out, rigid_info, rigid_config)
                        weight = weight + solve_out[i_d, i_b]
                    weight = weight / group_n_dofs

                    for i_d_ in range(group_n_dofs):
                        i_d = group_dof_start + i_d_
                        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                        if force_update or dyn_info.dofs.invweight[I_d] < EPS:
                            dyn_info.dofs.invweight[I_d] = weight


@qd.func
def func_refresh_links_invweight_and_meaninertia(
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    jac_row: qd.Tensor,
    solve_out: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update: qd.template(),
    refresh_position: qd.template(),
    refresh_velocity: qd.template(),
):
    """Recompute the inverse weights of the kinematic trees the given links belong to, and the mean inertia.

    A weight is computed against the mass matrix of the whole tree, so a tree is the smallest unit that can be
    recomputed on its own, and each tree is recomputed once however many of its links are listed.
    """
    func_enter_neutral_configuration(envs_idx, dyn_state, dyn_info, rigid_info, rigid_config)

    # Unbatched weights hold one value for the whole batch, so the first environment alone has anything to do. Trees
    # are weighed side by side: a weight is solved over the blocks of its own chain (see func_solve_mass_block), so the
    # rows one tree writes are no other tree's.
    n_envs_pending = (
        envs_idx.shape[0] if qd.static(rigid_config.batch_links_info or rigid_config.batch_dofs_info) else 1
    )
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_, i_l_ in qd.ndrange(n_envs_pending, links_idx.shape[0]):
        i_b = envs_idx[i_b_]
        # An unbatched weight holds one value for the whole batch, so it is written from the first environment alone.
        is_link_pending = True if qd.static(rigid_config.batch_links_info) else i_b_ == 0
        is_dofs_pending = True if qd.static(rigid_config.batch_dofs_info) else i_b_ == 0
        J_l = [links_idx[i_l_], i_b] if qd.static(rigid_config.batch_links_info) else links_idx[i_l_]
        i_rl = dyn_info.links.root_idx[J_l]

        # Trees are refreshed as a whole, hence only from the first of their links to be listed.
        is_tree_pending = True
        for j_l_ in range(i_l_):
            K_l = [links_idx[j_l_], i_b] if qd.static(rigid_config.batch_links_info) else links_idx[j_l_]
            if dyn_info.links.root_idx[K_l] == i_rl:
                is_tree_pending = False

        if is_tree_pending:
            # The span of a tree may interleave links of other trees, which their own root excludes here.
            for i_l in range(i_rl, rigid_info.links_root_end[i_rl]):
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                if dyn_info.links.root_idx[I_l] == i_rl:
                    if is_link_pending:
                        func_init_link_invweight(
                            i_l, i_b, jac_row, solve_out, dyn_state, dyn_info, rigid_info, rigid_config, force_update
                        )
                    if is_dofs_pending:
                        func_init_dofs_invweight(
                            i_l, i_b, jac_row, solve_out, dyn_state, dyn_info, rigid_info, rigid_config, force_update
                        )

    func_init_meaninertia(envs_idx, rigid_info, rigid_config)

    # Weighing moved the poses to the neutral configuration, so bring them back to the live one when the caller had
    # them up to date, and whenever a velocity is recomputed below, since that reads the live poses. The factored mass
    # matrix is left at the neutral configuration, the next step assembling it again for the one the scene is in.
    if qd.static(refresh_position or refresh_velocity):
        func_exit_neutral_configuration(envs_idx, dyn_state, dyn_info, rigid_info, rigid_config)

    # A velocity is quoted about the center of mass this pass moves, so the same motion reads as a different velocity
    # from here. Both readings are brought back where the caller held them current, since this is what moved them.
    if qd.static(refresh_velocity):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
            i_l_root = rigid_info.roots_link_idx[i_r]
            func_forward_velocity_root(
                i_l_root, envs_idx[i_b_], dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
            )


@qd.kernel(fastcache=True)
def kernel_refresh_invweight_and_meaninertia(
    envs_idx: qd.types.ndarray(),
    jac_row: qd.Tensor,
    solve_out: qd.Tensor,
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update: qd.template(),
    refresh_position: qd.template(),
    refresh_velocity: qd.template(),
):
    """Recompute the inverse weights of every link and DOF of the solver, and the mean inertia.

    Hibernated trees are woken up first: writing an inertial property moves the equilibrium a resting body had settled
    into, and a hibernated body cannot settle into the new one.
    """
    if qd.static(rigid_config.use_hibernation):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l, i_b_ in qd.ndrange(dyn_info.links.parent_idx.shape[0], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            if dyn_state.links.is_hibernated[i_l, i_b]:
                i_is = constraint_state.island.links_island_idx[i_l, i_b]
                func_wakeup_island(i_is, i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)

    func_enter_neutral_configuration(envs_idx, dyn_state, dyn_info, rigid_info, rigid_config)

    # Weighed as in func_refresh_links_invweight_and_meaninertia, over every tree instead of the listed ones.
    n_envs_pending = (
        envs_idx.shape[0] if qd.static(rigid_config.batch_links_info or rigid_config.batch_dofs_info) else 1
    )
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_, i_r in qd.ndrange(n_envs_pending, rigid_info.roots_link_idx.shape[0]):
        i_b = envs_idx[i_b_]
        is_link_pending = True if qd.static(rigid_config.batch_links_info) else i_b_ == 0
        is_dofs_pending = True if qd.static(rigid_config.batch_dofs_info) else i_b_ == 0
        # One worker per tree, whose span may interleave links of other trees, which their own root excludes here
        i_rl = rigid_info.roots_link_idx[i_r]
        for i_l in range(i_rl, rigid_info.links_root_end[i_rl]):
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if dyn_info.links.root_idx[I_l] != i_rl:
                continue
            if is_link_pending:
                func_init_link_invweight(
                    i_l, i_b, jac_row, solve_out, dyn_state, dyn_info, rigid_info, rigid_config, force_update
                )
            if is_dofs_pending:
                func_init_dofs_invweight(
                    i_l, i_b, jac_row, solve_out, dyn_state, dyn_info, rigid_info, rigid_config, force_update
                )

    func_init_meaninertia(envs_idx, rigid_info, rigid_config)

    # Both passes are the ones func_refresh_links_invweight_and_meaninertia ends on, and are owed for the same
    # reasons, which are given there.
    if qd.static(refresh_position or refresh_velocity):
        func_exit_neutral_configuration(envs_idx, dyn_state, dyn_info, rigid_info, rigid_config)

    if qd.static(refresh_velocity):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
        for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
            i_l_root = rigid_info.roots_link_idx[i_r]
            func_forward_velocity_root(
                i_l_root, envs_idx[i_b_], dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
            )


@qd.func
def func_torque_and_passive_force(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    # Actuation forces per dof from its ctrl mode, one thread per link, so a scene of many free bodies spreads over as
    # many threads as it holds links. Every link is visited, asleep or awake: an actuated sleeping link wakes below.
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_l, i_b in qd.ndrange(dyn_info.links.parent_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
        EPS = rigid_info.EPS[None]

        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.n_dofs[I_l] > 0:
            wakeup = False
            i_j = dyn_info.links.joint_start[I_l]
            I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
            joint_type = dyn_info.joints.type[I_j]

            for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                force = gs.qd_float(0.0)
                if dyn_state.dofs.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                    force = dyn_state.dofs.ctrl_force[i_d, i_b]
                elif dyn_state.dofs.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
                    force = -dyn_info.dofs.act_bias[I_d][2] * (
                        dyn_state.dofs.ctrl_vel[i_d, i_b] - dyn_state.dofs.vel[i_d, i_b]
                    )
                elif dyn_state.dofs.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION and not (
                    joint_type == gs.JOINT_TYPE.FREE and i_d >= dyn_info.links.dof_start[I_l] + 3
                ):
                    # Unified formula for GENERAL and POSITION modes, factored for float32 stability.
                    # For PD (act_gain == -act_bias[1], act_bias[0] == 0), the residual terms vanish.
                    force = (
                        dyn_info.dofs.act_gain[I_d] * (dyn_state.dofs.ctrl_pos[i_d, i_b] - dyn_state.dofs.pos[i_d, i_b])
                        + dyn_info.dofs.act_bias[I_d][0]
                        + (dyn_info.dofs.act_gain[I_d] + dyn_info.dofs.act_bias[I_d][1]) * dyn_state.dofs.pos[i_d, i_b]
                        + dyn_info.dofs.act_bias[I_d][2]
                        * (dyn_state.dofs.vel[i_d, i_b] - dyn_state.dofs.ctrl_vel[i_d, i_b])
                    )

                dyn_state.dofs.qf_applied[i_d, i_b] = qd.math.clamp(
                    force, dyn_info.dofs.force_range[I_d][0], dyn_info.dofs.force_range[I_d][1]
                )

                if qd.abs(force) > EPS:
                    wakeup = True

            dof_start = dyn_info.links.dof_start[I_l]
            if joint_type == gs.JOINT_TYPE.FREE and (
                dyn_state.dofs.ctrl_mode[dof_start + 3, i_b] == gs.CTRL_MODE.POSITION
                or dyn_state.dofs.ctrl_mode[dof_start + 4, i_b] == gs.CTRL_MODE.POSITION
                or dyn_state.dofs.ctrl_mode[dof_start + 5, i_b] == gs.CTRL_MODE.POSITION
            ):
                xyz = qd.Vector(
                    [
                        dyn_state.dofs.pos[0 + 3 + dof_start, i_b],
                        dyn_state.dofs.pos[1 + 3 + dof_start, i_b],
                        dyn_state.dofs.pos[2 + 3 + dof_start, i_b],
                    ],
                    dt=gs.qd_float,
                )

                ctrl_xyz = qd.Vector(
                    [
                        dyn_state.dofs.ctrl_pos[0 + 3 + dof_start, i_b],
                        dyn_state.dofs.ctrl_pos[1 + 3 + dof_start, i_b],
                        dyn_state.dofs.ctrl_pos[2 + 3 + dof_start, i_b],
                    ],
                    dt=gs.qd_float,
                )

                quat = gu.qd_xyz_to_quat(xyz)
                ctrl_quat = gu.qd_xyz_to_quat(ctrl_xyz)

                q_diff = gu.qd_transform_quat_by_quat(ctrl_quat, gu.qd_inv_quat(quat))
                rotvec = gu.qd_quat_to_rotvec(q_diff, EPS)

                for j in qd.static(range(3)):
                    i_d = dof_start + 3 + j
                    I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
                    force = (
                        dyn_info.dofs.act_gain[I_d] * rotvec[j]
                        + dyn_info.dofs.act_bias[I_d][0]
                        + (dyn_info.dofs.act_gain[I_d] + dyn_info.dofs.act_bias[I_d][1]) * dyn_state.dofs.pos[i_d, i_b]
                        + dyn_info.dofs.act_bias[I_d][2]
                        * (dyn_state.dofs.vel[i_d, i_b] - dyn_state.dofs.ctrl_vel[i_d, i_b])
                    )

                    dyn_state.dofs.qf_applied[i_d, i_b] = qd.math.clamp(
                        force, dyn_info.dofs.force_range[I_d][0], dyn_info.dofs.force_range[I_d][1]
                    )

                    if qd.abs(force) > EPS:
                        wakeup = True

            if qd.static(rigid_config.use_hibernation):
                # Actuation on a sleeping link wakes its island, the unit that sleeps and wakes together (see
                # func_wakeup_island, whose atomic claim serves the links of one island waking it at once)
                if wakeup and dyn_state.links.is_hibernated[i_l, i_b]:
                    i_is = constraint_state.island.links_island_idx[i_l, i_b]
                    func_wakeup_island(i_is, i_b, dyn_state, constraint_state, dyn_info, rigid_info, rigid_config)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(dyn_state.dofs.ctrl_mode.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.dofs.is_hibernated[i_d, i_b]
        if is_awake:
            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
            dyn_state.dofs.qf_passive[i_d, i_b] = -dyn_info.dofs.damping[I_d] * dyn_state.dofs.vel[i_d, i_b]

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.links.is_hibernated[i_l, i_b]
        if is_awake:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

            if dyn_info.links.n_dofs[I_l] > 0:
                i_j = dyn_info.links.joint_start[I_l]
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                joint_type = dyn_info.joints.type[I_j]

                if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                    dof_start = dyn_info.links.dof_start[I_l]
                    dof_end = dyn_info.links.dof_end[I_l]

                    for j_d in range(dof_end - dof_start):
                        I_d = [dof_start + j_d, i_b] if qd.static(rigid_config.batch_dofs_info) else dof_start + j_d
                        # Note that using dofs_state instead of qpos here allows qpos to be pulled into qpos0
                        # instead 0: dofs_state.pos = qpos - qpos0
                        func_add_safe_backward(
                            [dof_start + j_d, i_b],
                            -dyn_state.dofs.pos[dof_start + j_d, i_b] * dyn_info.dofs.stiffness[I_d],
                            dyn_state.dofs.qf_passive,
                            BW,
                        )


@qd.func
def func_update_acc(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    update_cacc: qd.template(),
    is_backward: qd.template(),
):
    """Compute the Cartesian accelerations of every awake link of every env, per kinematic root.

    A root takes one thread, or one block that sweeps it level by level under enable_tree_level_sweep (see
    func_update_cartesian_space in forward_kinematics.py).

    A tree root reads the acceleration of its static parent, the gravity term alone, so the walk covers the static
    links of the root ahead of its trees: one walk keeps the link body inlined once in the kernel, where a static pass
    and a tree pass would inline it twice. Differentiability wants the loop outermost in its kernel, which it is.
    """
    if qd.static(rigid_config.enable_tree_level_sweep):
        func_sweep_links_by_level(
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
            sweep_pass=LINK_SWEEP_PASS.ACCELERATION,
            force_update_all_geoms=False,
            update_cacc=update_cacc,
            is_backward=is_backward,
        )
    else:
        qd.loop_config(name="update_acc", serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
            i_l_root = rigid_info.roots_link_idx[i_r]
            i_l_end = rigid_info.links_root_end[i_l_root]
            for i_l in range(i_l_root, i_l_end):
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                if dyn_info.links.root_idx[I_l] == i_l_root and func_is_awake_link(i_l, i_b, dyn_state, rigid_config):
                    func_update_acc_link(
                        i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config, update_cacc, is_backward
                    )


@qd.func
def func_update_force(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.links.pos.shape[1]):
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.links.is_hibernated[i_l, i_b]
        if is_awake:
            f1_ang, f1_vel = gu.inertial_mul(
                dyn_state.links.cinr_pos[i_l, i_b],
                dyn_state.links.cinr_inertial[i_l, i_b],
                dyn_state.links.cinr_mass[i_l, i_b],
                dyn_state.links.cdd_vel[i_l, i_b],
                dyn_state.links.cdd_ang[i_l, i_b],
            )
            f2_ang, f2_vel = gu.inertial_mul(
                dyn_state.links.cinr_pos[i_l, i_b],
                dyn_state.links.cinr_inertial[i_l, i_b],
                dyn_state.links.cinr_mass[i_l, i_b],
                dyn_state.links.cd_vel[i_l, i_b],
                dyn_state.links.cd_ang[i_l, i_b],
            )
            f3_ang, f3_vel = gu.motion_cross_force(
                dyn_state.links.cd_ang[i_l, i_b], dyn_state.links.cd_vel[i_l, i_b], f2_ang, f2_vel
            )

            dyn_state.links.cfrc_vel[i_l, i_b] = (
                f1_vel
                + f3_vel
                + dyn_state.links.cfrc_applied_vel[i_l, i_b]
                + dyn_state.links.cfrc_coupling_vel[i_l, i_b]
            )
            dyn_state.links.cfrc_ang[i_l, i_b] = (
                f1_ang
                + f3_ang
                + dyn_state.links.cfrc_applied_ang[i_l, i_b]
                + dyn_state.links.cfrc_coupling_ang[i_l, i_b]
            )

    # One thread folds the forces of a whole kinematic tree from its leaves up to its root, gating each link of the
    # span on that root, like func_crb_fold: a tree spans several entities once one is attached beneath another, and a
    # child must fold into its parent before the parent folds further up.
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_r, i_b in qd.ndrange(rigid_info.roots_link_idx.shape[0], dyn_state.links.pos.shape[1]):
        i_l_root = rigid_info.roots_link_idx[i_r]
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.links.is_hibernated[i_l_root, i_b]
        if is_awake:
            i_l_end = rigid_info.links_root_end[i_l_root]
            for k in range(i_l_end - i_l_root):
                i_l = i_l_end - 1 - k
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                i_p = dyn_info.links.parent_idx[I_l]
                I_p = [i_p, i_b]
                if dyn_info.links.root_idx[I_l] == i_l_root and i_p != -1:
                    func_add_safe_backward(I_p, dyn_state.links.cfrc_vel[i_l, i_b], dyn_state.links.cfrc_vel, BW)
                    func_add_safe_backward(I_p, dyn_state.links.cfrc_ang[i_l, i_b], dyn_state.links.cfrc_ang, BW)


@qd.func
def func_bias_force(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.links.is_hibernated[i_l, i_b]
        if is_awake:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l

            for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                dyn_state.dofs.qf_bias[i_d, i_b] = dyn_state.dofs.cdof_ang[i_d, i_b].dot(
                    dyn_state.links.cfrc_ang[i_l, i_b]
                ) + dyn_state.dofs.cdof_vel[i_d, i_b].dot(dyn_state.links.cfrc_vel[i_l, i_b])

                dyn_state.dofs.force[i_d, i_b] = (
                    dyn_state.dofs.qf_passive[i_d, i_b]
                    - dyn_state.dofs.qf_bias[i_d, i_b]
                    + dyn_state.dofs.qf_applied[i_d, i_b]
                )

                dyn_state.dofs.qf_smooth[i_d, i_b] = dyn_state.dofs.force[i_d, i_b]


@qd.func
def func_compute_qacc(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    # The smooth acceleration, one block of the mass matrix at a time over every environment (see func_solve_mass_batch
    # for what a block is), the blocks of the sleeping dofs left as they are
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(rigid_info.mass_mat.shape[0], dyn_state.dofs.acc_smooth.shape[1]):
        is_hibernated = dyn_state.dofs.is_hibernated[i_d, i_b] if qd.static(rigid_config.use_hibernation) else False
        if rigid_info.dofs_mass_block_start[i_d] == i_d and not is_hibernated:
            block_end = rigid_info.dofs_mass_block_end[i_d]
            func_solve_mass_block(
                i_b, i_d, block_end, dyn_state.dofs.force, dyn_state.dofs.acc_smooth, rigid_info, rigid_config
            )

    # A hibernated tree keeps the acceleration it was put to sleep with
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
        if func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config):
            dof_start = rigid_info.trees_dof_start[i_t]
            for i_d in range(dof_start, dof_start + rigid_info.trees_n_dofs[i_t]):
                dyn_state.dofs.acc[i_d, i_b] = dyn_state.dofs.acc_smooth[i_d, i_b]


@qd.func
def func_midpoint_eligible(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Whether the link is a free rigid body eligible for midpoint integration this step.

    Eligible: a 6-DOF free-joint link that is its own whole kinematic tree (no parent, and no DOF-bearing descendant, so
    its mass block holds its own DOFs alone, see dofs_mass_block_start in array_class.py), and unconstrained this step
    (no contact on the body, fixed children included, and no connect/weld equality, per the involvement flag the
    assembly writes, see is_constrained in array_class.py). The flag covers dynamically registered welds, and entities
    merged at build time via attach fail the tree tests. A constrained body keeps the standard update: the solver
    resolves the constraint impulse at the current configuration, and the discrete free rigid-body equation would count
    it again. In the MuJoCo compatibility mode, the body is the link alone, without fixed children.
    """
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    is_eligible = False
    if dyn_info.links.n_dofs[I_l] == 6 and dyn_info.links.parent_idx[I_l] == -1:
        i_j = dyn_info.links.joint_start[I_l]
        I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
        dof_start = dyn_info.links.dof_start[I_l]
        is_eligible = (
            dyn_info.joints.type[I_j] == gs.JOINT_TYPE.FREE
            and rigid_info.dofs_mass_block_end[dof_start] <= dyn_info.links.dof_end[I_l]
        )
        if is_eligible:
            # The assembly marks the link a constraint acts on, a fixed child included, so the whole body is scanned.
            for j_l in range(i_l, rigid_info.links_root_end[i_l]):
                J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
                if dyn_info.links.root_idx[J_l] == i_l and dyn_state.links.is_constrained[j_l, i_b]:
                    is_eligible = False
        # MuJoCo integrates a lone body only, so its compatibility mode leaves a composite on the standard update.
        if qd.static(rigid_config.enable_mujoco_compatibility):
            if is_eligible and func_midpoint_has_fixed_children(i_l, i_b, dyn_info, rigid_info, rigid_config):
                is_eligible = False
    return is_eligible


@qd.func
def func_midpoint_has_fixed_children(
    i_l,
    i_b,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Whether the body of the free root holds links other than the root itself."""
    has_fixed_children = False
    for j_l in range(i_l + 1, rigid_info.links_root_end[i_l]):
        J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
        if dyn_info.links.root_idx[J_l] == i_l:
            has_fixed_children = True
    return has_fixed_children


@qd.func
def func_midpoint_is_aligned(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Whether the center of mass of the free body sits at its joint origin.

    Alignment moves the frame of an aligned free body onto its center of mass at load, and the build shrinks the mass
    block of such a body to single DOFs, which no other body gets. A body made of one link reads its own inertial
    position. Such a body keeps the standard translation update, so the position update and the velocity recovery
    both read this (see func_midpoint_free_body).
    """
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    dof_start = dyn_info.links.dof_start[I_l]
    is_aligned = False
    if rigid_info.dofs_mass_block_end[dof_start] == dof_start + 1:
        is_aligned = True
    elif not func_midpoint_has_fixed_children(i_l, i_b, dyn_info, rigid_info, rigid_config):
        ipos = dyn_info.links.inertial_pos[I_l]
        is_aligned = ipos[0] == 0.0 and ipos[1] == 0.0 and ipos[2] == 0.0
    return is_aligned


@qd.func
def func_midpoint_free_body(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Integrate the velocities of a free rigid body over one substep with the implicit midpoint rule, so that a
    tumbling body keeps its kinetic energy and angular momentum.

    Solves the free rigid-body equation I * (w_new - w) / h = tau - w_mid x (I * w_mid) for the midpoint angular
    velocity w_mid = (w + w_new) / 2 by Newton iteration with backtracking, in a body frame where I is constant: the
    inertial frame of a lone link, or the link frame of a composite. Outside the MuJoCo compatibility mode, armature,
    damping and servo gains add to the derivative terms only, along the link axes for the angular DOFs and the world
    axes for the linear ones. The midpoint rule preserves the quadratic invariants of torque-free tumbling (kinetic
    energy, squared angular momentum). A center of mass at the joint origin keeps the standard translation update.
    Otherwise the coupled midpoint center-of-mass velocity has a closed form, with gravity applied in the accelerating
    frame, and a linear augmentation couples the rotation to it through the origin.

    Writes acc[dofs] = (new - old) / h and vel_next[dofs] = (new + old) / 2: the position update integrates with
    the midpoint velocity, and the caller recovers the true next velocity from it afterwards.
    """
    EPS = rigid_info.EPS[None]
    # Newton tolerance on the residual, relative to the momentum scale (matching MuJoCo's midpoint integration)
    tol = gs.qd_float(1e-6) if qd.static(gs.qd_float == qd.f32) else gs.qd_float(1e-13)
    h = rigid_info.substep_dt[None]
    i2h = 2.0 / h

    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    dof_start = dyn_info.links.dof_start[I_l]

    iquat = dyn_info.links.inertial_quat[I_l]
    ipos = dyn_info.links.inertial_pos[I_l]
    inertia = dyn_info.links.inertial_i[I_l]
    mass = dyn_info.links.inertial_mass[I_l]
    xquat = dyn_state.links.quat[i_l, i_b]

    # Fixed children make the link a composite. The composite inertia is constant in the link frame, so it replaces the
    # own inertial for the integration.
    if func_midpoint_has_fixed_children(i_l, i_b, dyn_info, rigid_info, rigid_config):
        rot_x = gu.qd_quat_to_R(xquat, EPS)
        iquat = gu.qd_identity_quat()
        ipos = rot_x.transpose() @ (dyn_state.links.root_COM[i_l, i_b] - dyn_state.links.pos[i_l, i_b])
        inertia = rot_x.transpose() @ dyn_state.links.crb_inertial[i_l, i_b] @ rot_x
        mass = dyn_state.links.crb_mass[i_l, i_b]
    inv_iquat = gu.qd_inv_quat(iquat)
    rot_x2i = gu.qd_quat_mul(inv_iquat, gu.qd_inv_quat(xquat))

    # Each DOF adds to the joint-space mass its armature and the first-order damping and servo terms of the implicit
    # update (see func_compute_mass_matrix). The angular ones sit along the link axes and add to the inertia in the
    # inertial frame, where both are constant. The linear ones sit along the world axes. MuJoCo integrates the inertia
    # of the links alone here, so its compatibility mode leaves them out.
    aug_ang = qd.Matrix.zero(gs.qd_float, 3, 3)
    aug_vel = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(not rigid_config.enable_mujoco_compatibility):
        for j in qd.static(range(6)):
            i_d = dof_start + j
            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
            augmentation = dyn_info.dofs.armature[I_d] + dyn_info.dofs.damping[I_d] * h
            if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                augmentation -= dyn_info.dofs.act_bias[I_d][2] * h
            if qd.static(j < 3):
                aug_vel[j] = augmentation
            else:
                aug_ang[j - 3, j - 3] = augmentation
    rot_i = gu.qd_quat_to_R(iquat, EPS)
    inertia_aug = inertia + rot_i.transpose() @ aug_ang @ rot_i

    # Angular velocity and total torque (applied + passive + constraint + external link loads) in the inertial frame.
    # The free joint's angular DOFs are body-frame. The stored bias force is re-added to make the gyroscopic and gravity
    # terms explicit. It also carries the external link and coupling loads of every link of the tree (see
    # func_bias_force), which are stripped back out so they keep their standard-path sign: the midpoint equation and its
    # accelerating-frame gravity term regenerate only the velocity products and gravity.
    ext_ang = qd.Vector.zero(gs.qd_float, 3)
    ext_vel = qd.Vector.zero(gs.qd_float, 3)
    for j_l in range(i_l, rigid_info.links_root_end[i_l]):
        J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
        if dyn_info.links.root_idx[J_l] == i_l:
            ext_ang += dyn_state.links.cfrc_applied_ang[j_l, i_b] + dyn_state.links.cfrc_coupling_ang[j_l, i_b]
            ext_vel += dyn_state.links.cfrc_applied_vel[j_l, i_b] + dyn_state.links.cfrc_coupling_vel[j_l, i_b]
    w_body = gs.qd_vec3(
        [
            dyn_state.dofs.vel[dof_start + 3, i_b],
            dyn_state.dofs.vel[dof_start + 4, i_b],
            dyn_state.dofs.vel[dof_start + 5, i_b],
        ]
    )
    tau_body = qd.Vector.zero(gs.qd_float, 3)
    for j in qd.static(range(3)):
        i_d = dof_start + 3 + j
        qf_ext = dyn_state.dofs.cdof_ang[i_d, i_b].dot(ext_ang) + dyn_state.dofs.cdof_vel[i_d, i_b].dot(ext_vel)
        tau_body[j] = dyn_state.dofs.force[i_d, i_b] + dyn_state.dofs.qf_bias[i_d, i_b] - qf_ext
    w = gu.qd_transform_by_quat(w_body, inv_iquat)
    tau_com = gu.qd_transform_by_quat(tau_body, inv_iquat)

    # A center of mass at the joint origin decouples rotation from translation
    is_aligned = func_midpoint_is_aligned(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config)

    force = qd.Vector.zero(gs.qd_float, 3)
    r_com = qd.Vector.zero(gs.qd_float, 3)
    if not is_aligned:
        force_world = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(3)):
            i_d = dof_start + j
            qf_ext = dyn_state.dofs.cdof_ang[i_d, i_b].dot(ext_ang) + dyn_state.dofs.cdof_vel[i_d, i_b].dot(ext_vel)
            force_world[j] = dyn_state.dofs.force[i_d, i_b] + dyn_state.dofs.qf_bias[i_d, i_b] - qf_ext
        force = gu.qd_transform_by_quat(force_world, rot_x2i)
        r_com = gu.qd_transform_by_quat(ipos, inv_iquat)
        tau_com = tau_com - r_com.cross(force)

    # A linear augmentation on a body whose center of mass is off the joint origin couples the rotation to the
    # translation (see the residual below). Diagonal along the world axes, it is expressed in the inertial frame.
    gravity = qd.Vector.zero(gs.qd_float, 3)
    aug_vel_mat = qd.Matrix.zero(gs.qd_float, 3, 3)
    mass_inv_mat = qd.Matrix.zero(gs.qd_float, 3, 3)
    is_coupled = False
    if not is_aligned:
        i_e = dyn_info.links.entity_idx[I_l]
        gravity = rigid_info.gravity[i_b] * (1.0 - dyn_info.entities.gravity_compensation[i_e])
        gravity = gu.qd_transform_by_quat(gravity, rot_x2i)
        is_coupled = aug_vel[0] > 0.0 or aug_vel[1] > 0.0 or aug_vel[2] > 0.0
        if is_coupled:
            rot_x2i_mat = gu.qd_quat_to_R(rot_x2i, EPS)
            for j in qd.static(range(3)):
                aug_vel_mat[j, j] = aug_vel[j]
                mass_inv_mat[j, j] = 1.0 / (mass + aug_vel[j])
            aug_vel_mat = rot_x2i_mat @ aug_vel_mat @ rot_x2i_mat.transpose()
            mass_inv_mat = rot_x2i_mat @ mass_inv_mat @ rot_x2i_mat.transpose()
    skew_r = qd.Matrix([[0.0, -r_com[2], r_com[1]], [r_com[2], 0.0, -r_com[0]], [-r_com[1], r_com[0], 0.0]])
    identity = qd.Matrix.identity(gs.qd_float, 3)

    # Newton iteration with backtracking line search on the residual of the rotational midpoint equation. The residual
    # is f(w_mid) = 2/h * (I + E) * (w_mid - w) + w_mid x (I * w_mid) - tau - r_com x (E_lin * a_origin). The last term
    # is the moment of the linear augmentation E_lin. The translation midpoint gives a_origin = a_com - u, with (m * Id
    # + E_lin) @ a_com = f + m * g + E_lin @ u and u = alpha x r_com + w_mid x (w_mid x r_com). Each candidate is
    # evaluated once. An accepted one takes a Newton step, a rejected one halves the step.
    w_mid = w
    w_try = w
    delta = qd.Vector.zero(gs.qd_float, 3)
    step = gs.qd_float(1.0)
    f_norm = gs.qd_float(-1.0)
    n_newton = 0
    n_backtracks = 0
    while n_newton < 100 and n_backtracks < 20:
        Iw = inertia @ w_try
        f = i2h * (inertia_aug @ (w_try - w)) + w_try.cross(Iw) - tau_com
        if is_coupled:
            alpha = i2h * (w_try - w)
            u = alpha.cross(r_com) + w_try.cross(w_try.cross(r_com))
            a_origin = mass_inv_mat @ (force + mass * gravity + aug_vel_mat @ u) - u
            f = f - r_com.cross(aug_vel_mat @ a_origin)
        if f_norm < 0.0 or f.norm() < f_norm:
            w_mid = w_try
            f_norm = f.norm()
            if f_norm < tol * (1.0 + i2h * Iw.norm()):
                break
            # J = 2/h * (I + E) + d(w x Iw)/dw, with d(w x Iw)/dw = skew(w_mid) @ I - skew(I @ w_mid)
            skew_w = qd.Matrix([[0.0, -w_mid[2], w_mid[1]], [w_mid[2], 0.0, -w_mid[0]], [-w_mid[1], w_mid[0], 0.0]])
            skew_Iw = qd.Matrix([[0.0, -Iw[2], Iw[1]], [Iw[2], 0.0, -Iw[0]], [-Iw[1], Iw[0], 0.0]])
            J = i2h * inertia_aug + skew_w @ inertia - skew_Iw
            if is_coupled:
                # d(a_origin)/dw = (M^-1 @ E - Id) @ du/dw with du/dw = -2/h * skew(r_com) + d(w x (w x r_com))/dw
                du_dw = -i2h * skew_r + w_mid.outer_product(r_com) + w_mid.dot(r_com) * identity
                du_dw = du_dw - 2.0 * r_com.outer_product(w_mid)
                J = J - skew_r @ aug_vel_mat @ ((mass_inv_mat @ aug_vel_mat - identity) @ du_dw)
            delta = J.inverse() @ (-f)
            step = gs.qd_float(1.0)
            n_newton += 1
            n_backtracks = 0
        else:
            step = 0.5 * step
            n_backtracks += 1
        w_try = w_mid + step * delta

    # Next angular velocity in the body frame; positions integrate with the midpoint velocity
    w_new = 2.0 * w_mid - w
    w_new_body = gu.qd_transform_by_quat(w_new, iquat)
    for j in qd.static(range(3)):
        dyn_state.dofs.acc[dof_start + 3 + j, i_b] = (w_new_body[j] - w_body[j]) / h
        dyn_state.dofs.vel_next[dof_start + 3 + j, i_b] = 0.5 * (w_new_body[j] + w_body[j])

    if not is_aligned:
        # Closed-form midpoint solve of the coupled translation, (2/h * Id + skew(w_mid)) @ vcom_mid = b
        v_world = gs.qd_vec3(
            [
                dyn_state.dofs.vel[dof_start, i_b],
                dyn_state.dofs.vel[dof_start + 1, i_b],
                dyn_state.dofs.vel[dof_start + 2, i_b],
            ]
        )
        v = gu.qd_transform_by_quat(v_world, rot_x2i)
        vcom = v + w.cross(r_com)
        a_com = force / mass + gravity
        if is_coupled:
            alpha = i2h * (w_mid - w)
            u = alpha.cross(r_com) + w_mid.cross(w_mid.cross(r_com))
            a_com = mass_inv_mat @ (force + mass * gravity + aug_vel_mat @ u)
        b = a_com + i2h * vcom
        denom = i2h * i2h + w_mid.dot(w_mid)
        vcom_mid = (i2h * b + (w_mid.dot(b) / i2h) * w_mid - w_mid.cross(b)) / denom
        v_mid = vcom_mid - w_mid.cross(r_com)
        v_new = 2.0 * v_mid - v
        # The world-frame linear velocity goes through the estimated next orientation
        w_mid_body = gu.qd_transform_by_quat(w_mid, iquat)
        qrot = gu.qd_rotvec_to_quat(w_mid_body * h, EPS)
        xquat_new = gu.qd_transform_quat_by_quat(qrot, xquat)
        v_new_world = gu.qd_transform_by_quat(gu.qd_transform_by_quat(v_new, iquat), xquat_new)
        for j in qd.static(range(3)):
            dyn_state.dofs.acc[dof_start + j, i_b] = (v_new_world[j] - v_world[j]) / h
            dyn_state.dofs.vel_next[dof_start + j, i_b] = 0.5 * (v_new_world[j] + v_world[j])


@qd.func
def func_integrate(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    BW = qd.static(is_backward)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(dyn_state.dofs.ctrl_mode.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.dofs.is_hibernated[i_d, i_b]
        if is_awake:
            dyn_state.dofs.vel_next[i_d, i_b] = (
                dyn_state.dofs.vel[i_d, i_b] + dyn_state.dofs.acc[i_d, i_b] * rigid_info.substep_dt[None]
            )

    # Standalone free bodies advance with the implicit midpoint rule under the velocity-implicit integrators: their
    # acc / vel_next are overwritten here so the position loop below integrates with the midpoint velocity, and the
    # loop after it recovers the true next velocity. Gated out of the differentiable path: the Newton iteration
    # carries no adjoint.
    if qd.static(not is_backward and not rigid_config.requires_grad and rigid_config.integrator != gs.integrator.Euler):
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
            is_awake = True
            if qd.static(rigid_config.use_hibernation):
                is_awake = not dyn_state.links.is_hibernated[i_l, i_b]
            if is_awake:
                if func_midpoint_eligible(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config):
                    func_midpoint_free_body(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config)

    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
        is_awake = True
        if qd.static(rigid_config.use_hibernation):
            is_awake = not dyn_state.links.is_hibernated[i_l, i_b]
        if is_awake:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            if dyn_info.links.n_dofs[I_l] > 0:
                EPS = rigid_info.EPS[None]

                dof_start = dyn_info.links.dof_start[I_l]
                q_start = dyn_info.links.q_start[I_l]
                q_end = dyn_info.links.q_end[I_l]

                i_j = dyn_info.links.joint_start[I_l]
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                joint_type = dyn_info.joints.type[I_j]

                if joint_type == gs.JOINT_TYPE.FREE:
                    pos = qd.Vector(
                        [
                            rigid_info.qpos[q_start, i_b],
                            rigid_info.qpos[q_start + 1, i_b],
                            rigid_info.qpos[q_start + 2, i_b],
                        ]
                    )
                    vel = qd.Vector(
                        [
                            dyn_state.dofs.vel_next[dof_start, i_b],
                            dyn_state.dofs.vel_next[dof_start + 1, i_b],
                            dyn_state.dofs.vel_next[dof_start + 2, i_b],
                        ]
                    )
                    pos = pos + vel * rigid_info.substep_dt[None]
                    for j in qd.static(range(3)):
                        rigid_info.qpos_next[q_start + j, i_b] = pos[j]
                if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                    rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                    rot0 = qd.Vector(
                        [
                            rigid_info.qpos[q_start + rot_offset + 0, i_b],
                            rigid_info.qpos[q_start + rot_offset + 1, i_b],
                            rigid_info.qpos[q_start + rot_offset + 2, i_b],
                            rigid_info.qpos[q_start + rot_offset + 3, i_b],
                        ]
                    )
                    ang = (
                        qd.Vector(
                            [
                                dyn_state.dofs.vel_next[dof_start + rot_offset + 0, i_b],
                                dyn_state.dofs.vel_next[dof_start + rot_offset + 1, i_b],
                                dyn_state.dofs.vel_next[dof_start + rot_offset + 2, i_b],
                            ]
                        )
                        * rigid_info.substep_dt[None]
                    )
                    qrot = gu.qd_rotvec_to_quat(ang, EPS)
                    rot = gu.qd_transform_quat_by_quat(qrot, rot0)
                    for j in qd.static(range(4)):
                        rigid_info.qpos_next[q_start + j + rot_offset, i_b] = rot[j]
                else:
                    for j_ in range(q_end - q_start):
                        j = q_start + j_
                        if j < q_end:
                            rigid_info.qpos_next[j, i_b] = (
                                rigid_info.qpos[j, i_b]
                                + dyn_state.dofs.vel_next[dof_start + j_, i_b] * rigid_info.substep_dt[None]
                            )

    # Recover the true next velocity of the midpoint-integrated free bodies, whose vel_next held the midpoint
    # velocity for the position update above.
    if qd.static(not is_backward and not rigid_config.requires_grad and rigid_config.integrator != gs.integrator.Euler):
        qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
            is_awake = True
            if qd.static(rigid_config.use_hibernation):
                is_awake = not dyn_state.links.is_hibernated[i_l, i_b]
            if is_awake:
                if func_midpoint_eligible(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config):
                    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                    # Linear DOFs are midpoint-integrated only when the center of mass is off the joint origin
                    # (see func_midpoint_free_body)
                    j_start = 3
                    if not func_midpoint_is_aligned(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config):
                        j_start = 0
                    for j in range(j_start, 6):
                        i_d = dyn_info.links.dof_start[I_l] + j
                        dyn_state.dofs.vel_next[i_d, i_b] = (
                            2.0 * dyn_state.dofs.vel_next[i_d, i_b] - dyn_state.dofs.vel[i_d, i_b]
                        )

    # The coupling wrench of this substep is consumed: the bias force read it during forward dynamics and the midpoint
    # pass above read it last. The coupler accumulates the next one after the substep.
    qd.loop_config(serialize=rigid_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_l, i_b in qd.ndrange(dyn_info.links.root_idx.shape[0], dyn_state.dofs.ctrl_mode.shape[1]):
        dyn_state.links.cfrc_coupling_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        dyn_state.links.cfrc_coupling_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(rigid_config.use_hibernation and not is_backward):
            if not dyn_state.links.is_hibernated[i_l, i_b]:
                func_count_settled_step(i_l, i_b, dyn_state, dyn_info, rigid_info, rigid_config)


@qd.func
def func_count_settled_step(
    i_l,
    i_b,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Count this substep in the settled-step counter of awake link i_l of env i_b.

    The counter (see awake_steps in array_class.py) grows, up to hibernation_min_steps, while the largest dof speed of
    the link stays below the hibernation tolerance, and drops to zero the step it exceeds it. The speed of a dof is its
    next velocity scaled by dof_length (1 for translation, the swept radius for rotation), so the tolerance is one
    linear speed for every kind of dof: the rotational jitter of a small body is a tiny surface speed and counts as
    rest. An actuated link stays awake, since the actuation pass wakes any sleeping link it actuates (see
    func_torque_and_passive_force).
    """
    link_I = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    EPS = rigid_info.EPS[None]
    max_vel = gs.qd_float(0.0)
    is_driven = False
    for i_d in range(dyn_info.links.dof_start[link_I], dyn_info.links.dof_end[link_I]):
        I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
        max_vel = qd.max(max_vel, dyn_info.dofs.dof_length[I_d] * qd.abs(dyn_state.dofs.vel_next[i_d, i_b]))
        if qd.abs(dyn_state.dofs.qf_applied[i_d, i_b]) > EPS:
            is_driven = True
    if max_vel < rigid_info.hibernation_thresh_vel[None] and not is_driven:
        if dyn_state.links.awake_steps[i_l, i_b] < rigid_config.hibernation_min_steps:
            dyn_state.links.awake_steps[i_l, i_b] = dyn_state.links.awake_steps[i_l, i_b] + 1
    else:
        dyn_state.links.awake_steps[i_l, i_b] = 0


@qd.kernel(fastcache=True)
def kernel_forward_dynamics_without_qacc(
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    # Backward-only kernel. func_factor_mass is omitted: its reverse is unneeded since the backward pass seeds
    # mass_mat.grad directly via the implicit function theorem (see kernel_manual_compute_qacc_bw in manual_bw.py),
    # skipping the LDL^T factor chain. func_compute_mass_matrix is kept so Quadrants autodiff auto-reverses
    # mass_mat -> links pos / quat.
    func_compute_mass_matrix(
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
        qd.static(rigid_config.integrator == gs.integrator.approximate_implicitfast),
        is_backward,
    )
    func_torque_and_passive_force(dyn_state, constraint_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_update_acc(dyn_state, dyn_info, rigid_info, rigid_config, update_cacc=False, is_backward=is_backward)
    func_update_force(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)
    func_bias_force(dyn_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.func
def func_implicit_damping(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Make the damping of the integration implicit, from the acceleration the constraint solver left in dofs.acc.

    The damped acceleration solves (M + hD) a' = M a, applied as the correction a' = a - (M + hD)^-1 (hD a) on the
    trees carrying a damping term, whose factor takes hD (see func_has_implicit_damping_tree). In MuJoCo compatibility
    mode those trees re-solve their acceleration from the forces instead, (M + hD) a' = f, and under the implicitfast
    integrator every awake tree does, as MuJoCo does.
    """
    _B = rigid_info.mass_mat.shape[2]
    n_dofs = dyn_state.dofs.acc.shape[0]

    func_factor_mass(dyn_state, dyn_info, rigid_info, rigid_config, implicit_damping=True)

    # The correction moves the damped DOFs alone, and a constraint solve short of its fixed point integrates as the
    # bounded step it took. Re-solving from the force balance integrates its residual as M^-1 r, a spurious impulse on
    # a light body, which the compatibility mode reproduces.
    if qd.static(not rigid_config.enable_mujoco_compatibility):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            I_d = [i_d, i_b] if qd.static(rigid_config.batch_dofs_info) else i_d
            damping = dyn_info.dofs.damping[I_d]
            if qd.static(rigid_config.integrator == gs.integrator.implicitfast):
                if dyn_state.dofs.ctrl_mode[i_d, i_b] <= gs.CTRL_MODE.VELOCITY:
                    damping = damping - dyn_info.dofs.act_bias[I_d][2]
            dyn_state.dofs.qf_damping_implicit[i_d, i_b] = (
                damping * rigid_info.substep_dt[None] * dyn_state.dofs.acc[i_d, i_b]
            )

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_t, i_b in qd.ndrange(rigid_info.trees_root_idx.shape[0], _B):
        # A tree without a damping term has a zero correction and keeps its smooth factor, so the pass takes the awake
        # trees whose factor carries hD alone: re-solving such a tree would only reproduce the acceleration the
        # constraint solver holds, up to rounding
        is_solved = func_is_awake_tree(i_t, i_b, dyn_state, rigid_info, rigid_config)
        if qd.static(not rigid_config.enable_mujoco_compatibility or rigid_config.integrator == gs.integrator.Euler):
            if is_solved:
                is_solved = func_has_implicit_damping_tree(i_t, i_b, dyn_state, dyn_info, rigid_info, rigid_config)
        if is_solved:
            # One solve per mass block of the tree (see func_solve_mass_batch), in a form autodiff accepts
            tree_dof_start = rigid_info.trees_dof_start[i_t]
            tree_dof_end = tree_dof_start + rigid_info.trees_n_dofs[i_t]
            for i_d in range(tree_dof_start, tree_dof_end):
                if rigid_info.dofs_mass_block_start[i_d] == i_d:
                    block_end = rigid_info.dofs_mass_block_end[i_d]
                    if qd.static(not rigid_config.enable_mujoco_compatibility):
                        func_solve_mass_block(
                            i_b,
                            i_d,
                            block_end,
                            dyn_state.dofs.qf_damping_implicit,
                            dyn_state.dofs.qacc_damping_implicit,
                            rigid_info,
                            rigid_config,
                        )
                    else:
                        func_solve_mass_block(
                            i_b, i_d, block_end, dyn_state.dofs.force, dyn_state.dofs.acc, rigid_info, rigid_config
                        )
            if qd.static(not rigid_config.enable_mujoco_compatibility):
                for i_d in range(tree_dof_start, tree_dof_end):
                    dyn_state.dofs.acc[i_d, i_b] = (
                        dyn_state.dofs.acc[i_d, i_b] - dyn_state.dofs.qacc_damping_implicit[i_d, i_b]
                    )


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.rigid_solver_dynamics_decomp")
