"""Sleeping islands of the MochiSolver: islands whose bodies stay at rest stop being simulated until an awake body
comes near them or their state is set (see MochiOptions.use_sleeping).

Sleep is decided per island node (an entity) and environment. A node sleeps with every other node of its island, and
stays frozen exactly: it skips the warm start, its residual is zeroed, so its rows of the Newton system have zero
right-hand side and the linear solvers leave it in place, and its contact and element work is skipped. Sleeping nodes
keep their conservative bounds in the broadphase, so an awake body whose bounds reach a sleeping one joins its island
and wakes the whole island before the solve.

The API wakes what it changes: a state setter wakes its environments, a control command or a degree-of-freedom
parameter wakes the nodes of its degrees of freedom, a kinematic vertex target wakes its entity, and a change of gravity
or of contact settings wakes every node of the environments it reaches. Commands, issued at every step to a driven
body, wake that body alone so that the resting bodies of its environment sleep on. A fixed link belongs to no island, so
a contact setting changed on it must reach every body.
"""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .data import SOLVE_STATUS, MochiInfo, MochiIslandState, MochiSoftInfo, MochiSoftState, MochiState


@qd.func
def func_sync_asleep_flags(
    i_b: int,
    n_rigid_entities: int,
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    has_soft: qd.template(),
):
    """Copy the sleep state of the nodes of an environment to its links and deformable entities."""
    n_links = island_state.links_node.shape[0]
    for i_l in range(n_links):
        i_n = island_state.links_node[i_l]
        is_asleep = False
        if i_n >= 0:
            is_asleep = island_state.nodes_is_asleep[i_n, i_b]
        mochi_state.links_is_asleep[i_l, i_b] = is_asleep
    if qd.static(has_soft):
        for i_e in range(soft_state.entities_is_asleep.shape[0]):
            soft_state.entities_is_asleep[i_e, i_b] = island_state.nodes_is_asleep[n_rigid_entities + i_e, i_b]


@qd.func
def func_wake_islands(
    i_b: int,
    n_rigid_entities: int,
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    has_soft: qd.template(),
):
    """Wake every sleeping node of an environment whose island also holds an awake node, which happens when the bounds
    of an awake body reach a sleeping one. Islands made of sleeping nodes alone stay asleep."""
    n_nodes = island_state.nodes_parent.shape[0]
    for i_isl in range(island_state.n_islands[i_b]):
        island_state.islands_is_awake[i_isl, i_b] = False
    for i_n in range(n_nodes):
        if not island_state.nodes_is_asleep[i_n, i_b]:
            i_isl = island_state.nodes_island[i_n, i_b]
            island_state.islands_is_awake[i_isl, i_b] = True
    for i_n in range(n_nodes):
        i_isl = island_state.nodes_island[i_n, i_b]
        if island_state.nodes_is_asleep[i_n, i_b] and island_state.islands_is_awake[i_isl, i_b]:
            island_state.nodes_is_asleep[i_n, i_b] = False
            island_state.nodes_rest_steps[i_n, i_b] = 0
    func_sync_asleep_flags(i_b, n_rigid_entities, mochi_state, soft_state, island_state, has_soft)


@qd.func
def func_update_sleep(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    n_rigid_entities: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    has_soft: qd.template(),
):
    """Count the consecutive steps at rest of every awake node after its solve, and put to sleep the islands whose
    nodes have all been at rest for `sleep_min_steps` steps.

    A step is at rest for a node when its solve converged while reducing the node's residual by less than
    `sleep_threshold` from the warm start (always so when the warm start is already within the absolute tolerance, as
    for a body at equilibrium), and when no point of it moves faster than `sleep_max_speed`. A node under an active
    actuator, or holding a rod, never counts as at rest.
    """
    n_nodes = island_state.nodes_parent.shape[0]
    n_links = island_state.links_node.shape[0]
    _B = mochi_state.is_active.shape[0]
    abs_tol = mochi_info.newton_abs_tol[None]
    rel_tol = mochi_info.newton_rel_tol[None]
    threshold = mochi_info.sleep_threshold[None]
    min_steps = mochi_info.sleep_min_steps[None]
    max_speed = mochi_info.sleep_max_speed[None]

    # The fastest point of every node (the center of mass plus the spin times the farthest sample corner of every
    # link, every vertex of a deformable body), and its actuated degrees of freedom.
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_n, i_slot in qd.ndrange(n_nodes, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_nodes, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        island_state.nodes_is_restless[i_n, i_b] = False
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        i_n = island_state.links_node[i_l]
        if i_n >= 0 and mochi_info.links.is_dynamic[i_l]:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            com = dyn_info.links.inertial_pos[I_l]
            aabb_min = mochi_info.links.samples_aabb_min[i_l]
            aabb_max = mochi_info.links.samples_aabb_max[i_l]
            radius = qd.max(qd.abs(aabb_min - com), qd.abs(aabb_max - com)).norm()
            speed = mochi_state.links_vel[i_l, i_b].norm() + mochi_state.links_ang[i_l, i_b].norm() * radius
            is_restless = speed > max_speed
            for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                ctrl_mode = dyn_state.dofs.ctrl_mode[i_d, i_b]
                if ctrl_mode != gs.CTRL_MODE.FORCE or qd.abs(dyn_state.dofs.ctrl_force[i_d, i_b]) > 0.0:
                    is_restless = True
            if is_restless:
                island_state.nodes_is_restless[i_n, i_b] = True
    if qd.static(has_soft):
        n_verts = soft_state.verts_pos.shape[0]
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_v, i_slot in qd.ndrange(n_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_verts, 1):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            i_e = soft_info.verts_entity_idx[i_v]
            is_rod = soft_info.entities_rod_elem_end[i_e] > soft_info.entities_rod_elem_start[i_e]
            if is_rod or soft_state.verts_vel[i_v, i_b].norm() > max_speed:
                island_state.nodes_is_restless[n_rigid_entities + i_e, i_b] = True

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        is_diverged = mochi_state.status[i_b] == SOLVE_STATUS.DIVERGED
        for i_n in range(n_nodes):
            if not island_state.nodes_is_asleep[i_n, i_b]:
                norm0 = island_state.nodes_res_norm0_w[i_n, i_b]
                norm = qd.sqrt(island_state.nodes_res_w_sq[i_n, i_b])
                rest_value = gs.qd_float(0.0)
                if norm0 <= abs_tol:
                    rest_value = 1.0
                elif norm <= abs_tol or norm <= rel_tol * norm0:
                    rest_value = norm / norm0
                if rest_value >= threshold and not island_state.nodes_is_restless[i_n, i_b] and not is_diverged:
                    island_state.nodes_rest_steps[i_n, i_b] = qd.min(
                        island_state.nodes_rest_steps[i_n, i_b] + 1, min_steps
                    )
                else:
                    island_state.nodes_rest_steps[i_n, i_b] = 0
        for i_isl in range(island_state.n_islands[i_b]):
            island_state.islands_is_awake[i_isl, i_b] = False
        for i_n in range(n_nodes):
            if island_state.nodes_rest_steps[i_n, i_b] < min_steps and not island_state.nodes_is_asleep[i_n, i_b]:
                i_isl = island_state.nodes_island[i_n, i_b]
                island_state.islands_is_awake[i_isl, i_b] = True
        for i_n in range(n_nodes):
            i_isl = island_state.nodes_island[i_n, i_b]
            if not island_state.islands_is_awake[i_isl, i_b]:
                island_state.nodes_is_asleep[i_n, i_b] = True
        func_sync_asleep_flags(i_b, n_rigid_entities, mochi_state, soft_state, island_state, has_soft)


@qd.kernel
def kernel_update_sleep(
    n_rigid_entities: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    has_soft: qd.template(),
):
    func_update_sleep(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        n_rigid_entities,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        soft_info,
        soft_state,
        island_state,
        rigid_config,
        has_soft,
    )


@qd.kernel
def kernel_wake_nodes(
    envs_idx: qd.types.ndarray(),
    idx: qd.types.ndarray(),
    n_rigid_entities: int,
    idx_node: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    island_state: MochiIslandState,
    rigid_config: qd.template(),
    has_soft: qd.template(),
):
    """Wake the island nodes of the given items in the given environments and restart their rest counts.

    `idx` selects items (degrees of freedom, or nodes themselves) and `idx_node` maps every item to its island node.
    The island of a woken node wakes with it at the next step (see func_wake_islands).
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_ in range(idx.shape[0]):
            i_x = idx[i_]
            i_n = idx_node[i_x]
            island_state.nodes_is_asleep[i_n, i_b] = False
            island_state.nodes_rest_steps[i_n, i_b] = 0
        func_sync_asleep_flags(i_b, n_rigid_entities, mochi_state, soft_state, island_state, has_soft)
