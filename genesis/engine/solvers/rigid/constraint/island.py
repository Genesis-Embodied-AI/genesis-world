import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from ..collider.contact import func_contact_order_key


# Partition the dof-carrying kinematic trees (see trees_root_idx in array_class.py) into islands: the connected
# components of the trees under the edges of (1) the contacts, (2) the equality constraints (CONNECT/WELD on links,
# JOINT on joints) and (3) the hibernation chains. A single Genesis entity holding several free bodies (common in MJCF)
# thus splits into one island per free body, while an articulated body's links stay one island. Each island is an
# exactly decoupled (block-diagonal) sub-problem of the constraint solve. A component is labeled by its smallest tree,
# whatever the order its edges are met in, which keeps the partition, hence the per-island solve, deterministic.


@qd.func
def func_find_tree_root(i_t, i_b, constraint_state: array_class.ConstraintState):
    # Path-halving find over the trees
    root = i_t
    while constraint_state.island.trees_parent_idx[root, i_b] != root:
        constraint_state.island.trees_parent_idx[root, i_b] = constraint_state.island.trees_parent_idx[
            constraint_state.island.trees_parent_idx[root, i_b], i_b
        ]
        root = constraint_state.island.trees_parent_idx[root, i_b]
    return root


@qd.func
def func_union_trees(i_ta, i_tb, i_b, constraint_state: array_class.ConstraintState):
    # Union by minimum index: the root of a component is its smallest tree, regardless of the order edges are processed
    root_a = func_find_tree_root(i_ta, i_b, constraint_state)
    root_b = func_find_tree_root(i_tb, i_b, constraint_state)
    if root_a < root_b:
        constraint_state.island.trees_parent_idx[root_b, i_b] = root_a
    elif root_b < root_a:
        constraint_state.island.trees_parent_idx[root_a, i_b] = root_b


@qd.func
def func_joint_link(i_joint, i_b, n_links, dyn_info: array_class.DynInfo, rigid_config: qd.template()):
    # JointsInfo carries no link mapping, so locate the link whose dof range owns the joint's first dof. Joint
    # equalities are rare and link counts are small, so the linear scan is cheap.
    I_j = [i_joint, i_b] if qd.static(rigid_config.batch_joints_info) else i_joint
    i_dof = dyn_info.joints.dof_start[I_j]
    link = -1
    for i_l in range(n_links):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if dyn_info.links.dof_start[I_l] <= i_dof < dyn_info.links.dof_end[I_l]:
            link = i_l
            break
    return link


@qd.func
def func_equality_links(i_eq, i_b, n_links, dyn_info: array_class.DynInfo, rigid_config: qd.template()):
    # Map an equality constraint to the pair of links it couples. CONNECT/WELD reference links; JOINT references joints.
    obj1 = dyn_info.equalities.eq_obj1id[i_eq, i_b]
    obj2 = dyn_info.equalities.eq_obj2id[i_eq, i_b]
    eq_type = dyn_info.equalities.eq_type[i_eq, i_b]
    i_l_a = -1
    i_l_b = -1
    if eq_type == gs.EQUALITY_TYPE.JOINT:
        i_l_a = func_joint_link(obj1, i_b, n_links, dyn_info, rigid_config)
        if obj2 >= 0:
            i_l_b = func_joint_link(obj2, i_b, n_links, dyn_info, rigid_config)
    else:
        i_l_a = obj1
        i_l_b = obj2
    return i_l_a, i_l_b


@qd.func
def func_edge_trees(
    i_e,
    i_b,
    n_contacts,
    n_equalities,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Trees coupled by edge i_e of one env: the contacts first, in contact_sort_idx order, then the equality
    constraints, then (under hibernation) the chain successor of every link.

    Either tree is -1 when the edge couples no two dof-carrying trees: a contact against a fixed body, a link with no
    chain successor.
    """
    n_links = rigid_info.links_tree_idx.shape[0]
    link_a = -1
    link_b = -1
    if i_e < n_contacts:
        i_col = collider_state.contact_sort_idx[i_e, i_b]
        link_a = collider_state.contact_data.link_a[i_col, i_b]
        link_b = collider_state.contact_data.link_b[i_col, i_b]
    elif i_e < n_contacts + n_equalities:
        link_a, link_b = func_equality_links(i_e - n_contacts, i_b, n_links, dyn_info, rigid_config)
    else:
        if qd.static(rigid_config.use_hibernation):
            link_a = i_e - n_contacts - n_equalities
            link_b = constraint_state.island.hibernated_next_link[link_a, i_b]
            if link_b >= n_links or link_b == link_a:
                link_b = -1
    i_ta = -1
    i_tb = -1
    if link_a >= 0 and link_b >= 0:
        # A static link belongs to no tree (links_tree_idx -1) and joins no island
        i_ta = rigid_info.links_tree_idx[link_a]
        i_tb = rigid_info.links_tree_idx[link_b]
    return i_ta, i_tb


@qd.func
def func_constraint_island(i_c, i_b, constraint_state: array_class.ConstraintState):
    # A constraint couples dofs of a single island, so its island is that of any dof of its support, read from the
    # sparse dof list every assembly site fills (jac_dofs_idx, see _append_relevant_dof in solver.py); a row with an
    # empty support belongs to no island.
    i_island = -1
    if constraint_state.jac_n_dofs[i_c, i_b] > 0:
        i_d = constraint_state.jac_dofs_idx[i_c, 0, i_b]
        i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
    return i_island


@qd.func
def func_group_constraints_by_island(i_b, constraint_state: array_class.ConstraintState, rigid_config: qd.template()):
    """Group the constraints of one env by island and start every island iterating.

    The island of each constraint is resolved and the constraints are listed in contiguous per-island ranges of
    constraint_id, so the per-island solve iterates its own constraints. Every island's improved flag is then raised
    (see func_exit_decision in linesearch.py). The fill walks constraints in index order, so each island's constraint
    list stays order-deterministic. The whole pass is serial per env, folded into the per-env init loop of
    func_solve_init so the partition adds no kernel launch of its own: its work is a few integer operations per
    constraint.
    """
    n_islands = constraint_state.island.n_islands[i_b]
    n_con = constraint_state.n_constraints[i_b]
    if n_islands == 1:
        # A single island spans the whole env, so every constraint belongs to island 0 in index order and the grouping
        # is the identity. A constraint touching no dof carries jac == 0, so listing it in island 0 is harmless.
        constraint_state.island.constraint_slices.start[0, i_b] = 0
        constraint_state.island.constraint_slices.n[0, i_b] = n_con
        constraint_state.island.constraint_slices.curr[0, i_b] = n_con
        for i_c in range(n_con):
            constraint_state.island.constraint_id[i_c, i_b] = i_c
            constraint_state.island.constraint_island_idx[i_c, i_b] = 0
    else:
        for i_island in range(n_islands):
            constraint_state.island.constraint_slices.n[i_island, i_b] = 0

        for i_c in range(n_con):
            i_island = func_constraint_island(i_c, i_b, constraint_state)
            constraint_state.island.constraint_island_idx[i_c, i_b] = i_island
            if i_island >= 0:
                constraint_state.island.constraint_slices.n[i_island, i_b] = (
                    constraint_state.island.constraint_slices.n[i_island, i_b] + 1
                )

        con_list_start = 0
        for i_island in range(n_islands):
            constraint_state.island.constraint_slices.start[i_island, i_b] = con_list_start
            constraint_state.island.constraint_slices.curr[i_island, i_b] = con_list_start
            con_list_start = con_list_start + constraint_state.island.constraint_slices.n[i_island, i_b]

        for i_c in range(n_con):
            i_island = constraint_state.island.constraint_island_idx[i_c, i_b]
            if i_island >= 0:
                constraint_state.island.constraint_id[
                    constraint_state.island.constraint_slices.curr[i_island, i_b], i_b
                ] = i_c
                constraint_state.island.constraint_slices.curr[i_island, i_b] = (
                    constraint_state.island.constraint_slices.curr[i_island, i_b] + 1
                )

    # Every awake island starts iterating, a constraint-free one included: a warm start leaves its acceleration at the
    # previous step's, which its own iteration brings back to the unconstrained one.
    for i_island in range(n_islands):
        is_awake = n_con > 0
        if qd.static(rigid_config.use_hibernation):
            is_awake = is_awake and constraint_state.island.is_hibernated[i_island, i_b] == 0
        constraint_state.island.improved[i_island, i_b] = is_awake


@qd.func
def func_chunk_island_rank(tid, i_island, sh_chunk):
    """Rank of a lane among the lanes of the chunk that hold the same island: the position of this lane's item within
    the island, once added to the island's running total."""
    rank = 0
    for j in range(tid):
        if sh_chunk[j] == i_island:
            rank = rank + 1
    return rank


@qd.func
def func_group_constraints_by_island_coop(
    i_b, tid, sh_chunk, constraint_state: array_class.ConstraintState, rigid_config: qd.template()
):
    """Group the constraints of one env by island with the _K lanes of its block, in constraint index order.

    The grouping of func_group_constraints_by_island: the labels resolved lane by lane, the counts by atomics, the
    starts by a subgroup scan, and the fill per chunk of _K constraints, each ranked among the chunk's constraints of
    its island through sh_chunk (one slot per lane) and placed at its island's cursor, so the lists stay in constraint
    index order.
    """
    _K = qd.static(32)
    n_islands = constraint_state.island.n_islands[i_b]
    n_con = constraint_state.n_constraints[i_b]
    if n_islands == 1:
        if tid == 0:
            constraint_state.island.constraint_slices.start[0, i_b] = 0
            constraint_state.island.constraint_slices.n[0, i_b] = n_con
            constraint_state.island.constraint_slices.curr[0, i_b] = n_con
        i_c = tid
        while i_c < n_con:
            constraint_state.island.constraint_id[i_c, i_b] = i_c
            constraint_state.island.constraint_island_idx[i_c, i_b] = 0
            i_c = i_c + _K
    else:
        i_island = tid
        while i_island < n_islands:
            constraint_state.island.constraint_slices.n[i_island, i_b] = 0
            i_island = i_island + _K
        qd.simt.block.sync()
        i_c = tid
        while i_c < n_con:
            i_island = func_constraint_island(i_c, i_b, constraint_state)
            constraint_state.island.constraint_island_idx[i_c, i_b] = i_island
            if i_island >= 0:
                qd.atomic_add(constraint_state.island.constraint_slices.n[i_island, i_b], 1)
            i_c = i_c + _K
        qd.simt.block.sync()
        carry = 0
        for i_chunk in range((n_islands + _K - 1) // _K):
            i_island = i_chunk * _K + tid
            count = 0
            if i_island < n_islands:
                count = constraint_state.island.constraint_slices.n[i_island, i_b]
            count_incl = qd.simt.subgroup.inclusive_add(count)
            if i_island < n_islands:
                constraint_state.island.constraint_slices.start[i_island, i_b] = carry + count_incl - count
                constraint_state.island.constraint_slices.curr[i_island, i_b] = carry + count_incl - count
            carry = carry + qd.simt.subgroup.broadcast(count_incl, qd.u32(_K - 1))
        qd.simt.block.sync()
        for i_chunk in range((n_con + _K - 1) // _K):
            i_c = i_chunk * _K + tid
            i_island = -1
            if i_c < n_con:
                i_island = constraint_state.island.constraint_island_idx[i_c, i_b]
            sh_chunk[tid] = i_island
            qd.simt.block.sync()
            if i_island >= 0:
                i_pos = constraint_state.island.constraint_slices.curr[i_island, i_b]
                i_pos = i_pos + func_chunk_island_rank(tid, i_island, sh_chunk)
                constraint_state.island.constraint_id[i_pos, i_b] = i_c
            qd.simt.block.sync()
            if i_island >= 0:
                qd.atomic_add(constraint_state.island.constraint_slices.curr[i_island, i_b], 1)
            qd.simt.block.sync()

    # Every awake island starts iterating, a constraint-free one included, see func_group_constraints_by_island
    i_island = tid
    while i_island < n_islands:
        is_awake = n_con > 0
        if qd.static(rigid_config.use_hibernation):
            is_awake = is_awake and constraint_state.island.is_hibernated[i_island, i_b] == 0
        constraint_state.island.improved[i_island, i_b] = is_awake
        i_island = i_island + _K


@qd.func
def func_dof_range_start(i_island, i_b, constraint_state: array_class.ConstraintState):
    """First dof of an island whose ascending dof list holds consecutive dofs, -1 otherwise (see dof_range_start in
    array_class.py).

    The lists of the build ascend, so the first and last entries decide.
    """
    dof_lo = constraint_state.island.dof_slices.start[i_island, i_b]
    n_dofs_island = constraint_state.island.dof_slices.n[i_island, i_b]
    range_start = constraint_state.island.dof_id[dof_lo, i_b]
    if constraint_state.island.dof_id[dof_lo + n_dofs_island - 1, i_b] - range_start + 1 != n_dofs_island:
        range_start = -1
    return range_start


@qd.func
def func_build_islands(
    i_b,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Build one env's island partition serially.

    The components of the trees come from a union-find over the edges, labeled in ascending order of their smallest
    tree. Each island then lists its dofs (the trees in ascending order, the dofs of each in ascending order, so an
    island's dofs ascend) and, under hibernation, which alone reads them, its links, and holds its inertia (the trace
    of the mass matrix over its dofs, the scale of its convergence tests) and, under hibernation, its sleeping flag.
    The CPU skyline path then reorders each island's dofs by contact adjacency, see func_reorder_island_dofs.
    """
    n_trees = rigid_info.trees_root_idx.shape[0]
    n_links = rigid_info.links_tree_idx.shape[0]
    n_contacts = collider_state.n_contacts[i_b]
    n_equalities = constraint_state.qd_n_equalities[i_b]
    n_edges = n_contacts + n_equalities
    if qd.static(rigid_config.use_hibernation):
        n_edges = n_edges + n_links

    for i_t in range(n_trees):
        constraint_state.island.trees_parent_idx[i_t, i_b] = i_t
    for i_e in range(n_edges):
        i_ta, i_tb = func_edge_trees(
            i_e, i_b, n_contacts, n_equalities, collider_state, constraint_state, dyn_info, rigid_info, rigid_config
        )
        if i_ta >= 0 and i_tb >= 0:
            func_union_trees(i_ta, i_tb, i_b, constraint_state)

    # The tree arrays hold one padding slot in a scene without a tree (see n_trees_), which carries no dof and labels
    # no island
    n_islands = 0
    for i_t in range(n_trees):
        constraint_state.island.trees_island_idx[i_t, i_b] = -1
        if constraint_state.island.trees_parent_idx[i_t, i_b] == i_t and rigid_info.trees_n_dofs[i_t] > 0:
            constraint_state.island.trees_island_idx[i_t, i_b] = n_islands
            n_islands = n_islands + 1
    for i_t in range(n_trees):
        constraint_state.island.trees_island_idx[i_t, i_b] = constraint_state.island.trees_island_idx[
            func_find_tree_root(i_t, i_b, constraint_state), i_b
        ]
    constraint_state.island.n_islands[i_b] = n_islands

    for i_island in range(n_islands):
        constraint_state.island.dof_slices.n[i_island, i_b] = 0
        if qd.static(rigid_config.use_hibernation):
            constraint_state.island.link_slices.n[i_island, i_b] = 0
        constraint_state.island.inertia[i_island, i_b] = 0.0
        if qd.static(rigid_config.use_hibernation):
            constraint_state.island.is_hibernated[i_island, i_b] = 1
    for i_t in range(n_trees):
        i_island = constraint_state.island.trees_island_idx[i_t, i_b]
        if i_island >= 0:
            if qd.static(rigid_config.use_hibernation):
                constraint_state.island.link_slices.n[i_island, i_b] = (
                    constraint_state.island.link_slices.n[i_island, i_b] + rigid_info.trees_n_links[i_t]
                )
            constraint_state.island.dof_slices.n[i_island, i_b] = (
                constraint_state.island.dof_slices.n[i_island, i_b] + rigid_info.trees_n_dofs[i_t]
            )
    link_list_start = 0
    dof_list_start = 0
    for i_island in range(n_islands):
        if qd.static(rigid_config.use_hibernation):
            constraint_state.island.link_slices.start[i_island, i_b] = link_list_start
            constraint_state.island.link_slices.curr[i_island, i_b] = link_list_start
            link_list_start = link_list_start + constraint_state.island.link_slices.n[i_island, i_b]
        constraint_state.island.dof_slices.start[i_island, i_b] = dof_list_start
        constraint_state.island.dof_slices.curr[i_island, i_b] = dof_list_start
        dof_list_start = dof_list_start + constraint_state.island.dof_slices.n[i_island, i_b]

    for i_t in range(n_trees):
        i_island = constraint_state.island.trees_island_idx[i_t, i_b]
        for i_l in range(rigid_info.trees_root_idx[i_t], rigid_info.trees_link_end[i_t]):
            if rigid_info.links_tree_idx[i_l] == i_t:
                constraint_state.island.links_island_idx[i_l, i_b] = i_island
                if qd.static(rigid_config.use_hibernation):
                    if i_island >= 0:
                        i_pos = constraint_state.island.link_slices.curr[i_island, i_b]
                        constraint_state.island.link_id[i_pos, i_b] = i_l
                        constraint_state.island.link_slices.curr[i_island, i_b] = i_pos + 1
        # FIXME: quadrants debug builds check a loop-invariant access before the loop runs, so the empty dof loop of a
        # dof-less tree trips on its island of -1. The guard skips that loop, which iterates over no dof anyway.
        if i_island >= 0:
            dof_start_tree = rigid_info.trees_dof_start[i_t]
            for i_d in range(dof_start_tree, dof_start_tree + rigid_info.trees_n_dofs[i_t]):
                i_pos = constraint_state.island.dof_slices.curr[i_island, i_b]
                constraint_state.island.dof_id[i_pos, i_b] = i_d
                constraint_state.island.dof_local_pos[i_d, i_b] = (
                    i_pos - constraint_state.island.dof_slices.start[i_island, i_b]
                )
                constraint_state.island.dofs_island_idx[i_d, i_b] = i_island
                constraint_state.island.dof_slices.curr[i_island, i_b] = i_pos + 1
                constraint_state.island.inertia[i_island, i_b] = (
                    constraint_state.island.inertia[i_island, i_b] + rigid_info.mass_mat[i_d, i_d, i_b]
                )
    for i_island in range(n_islands):
        constraint_state.island.dof_range_start[i_island, i_b] = func_dof_range_start(i_island, i_b, constraint_state)

    # An island is hibernated unless one of its links is awake
    if qd.static(rigid_config.use_hibernation):
        for i_l in range(n_links):
            i_island = constraint_state.island.links_island_idx[i_l, i_b]
            if i_island >= 0 and not dyn_state.links.is_hibernated[i_l, i_b]:
                constraint_state.island.is_hibernated[i_island, i_b] = 0

    if qd.static(rigid_config.sparse_solve):
        func_reorder_island_dofs(i_b, collider_state, constraint_state, rigid_info)


@qd.func
def func_build_single_island(i_b, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo):
    """Write the partition of one env of a single-island scene serially: one island holding every dof in order.

    The lists are the identity and the island's inertia the trace of the mass matrix. Reserved for scenes off the CPU
    skyline path and hibernation, which alone read the tree and link labels (see _sort_contacts_and_build_islands in
    solver.py).
    """
    n_dofs = constraint_state.island.dof_id.shape[0]
    constraint_state.island.n_islands[i_b] = 1
    constraint_state.island.dof_slices.start[0, i_b] = 0
    constraint_state.island.dof_slices.n[0, i_b] = n_dofs
    constraint_state.island.dof_slices.curr[0, i_b] = n_dofs
    constraint_state.island.dof_range_start[0, i_b] = 0
    inertia = gs.qd_float(0.0)
    for i_d in range(n_dofs):
        constraint_state.island.dof_id[i_d, i_b] = i_d
        constraint_state.island.dof_local_pos[i_d, i_b] = i_d
        constraint_state.island.dofs_island_idx[i_d, i_b] = 0
        inertia = inertia + rigid_info.mass_mat[i_d, i_d, i_b]
    constraint_state.island.inertia[0, i_b] = inertia


@qd.func
def func_build_single_island_coop(
    i_b, tid, constraint_state: array_class.ConstraintState, rigid_info: array_class.RigidInfo
):
    """Write the partition of one env of a single-island scene with the _K lanes of its block, see
    func_build_single_island."""
    _K = qd.static(32)
    n_dofs = constraint_state.island.dof_id.shape[0]
    if tid == 0:
        constraint_state.island.n_islands[i_b] = 1
        constraint_state.island.dof_slices.start[0, i_b] = 0
        constraint_state.island.dof_slices.n[0, i_b] = n_dofs
        constraint_state.island.dof_slices.curr[0, i_b] = n_dofs
        constraint_state.island.dof_range_start[0, i_b] = 0
    inertia = gs.qd_float(0.0)
    i_d = tid
    while i_d < n_dofs:
        constraint_state.island.dof_id[i_d, i_b] = i_d
        constraint_state.island.dof_local_pos[i_d, i_b] = i_d
        constraint_state.island.dofs_island_idx[i_d, i_b] = 0
        inertia = inertia + rigid_info.mass_mat[i_d, i_d, i_b]
        i_d = i_d + _K
    inertia = qd.simt.subgroup.reduce_all_add_tiled(inertia, 5)
    if tid == 0:
        constraint_state.island.inertia[0, i_b] = inertia


@qd.func
def func_tree_component(i_t, i_b, constraint_state: array_class.ConstraintState):
    # Root of a tree's component in the union-find forest, reading only
    root = i_t
    while constraint_state.island.trees_parent_idx[root, i_b] != root:
        root = constraint_state.island.trees_parent_idx[root, i_b]
    return root


@qd.func
def func_build_islands_coop(
    i_b,
    tid,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Build one env's island partition with the _K lanes of its block.

    The partition is the very one func_build_islands builds serially: the same labels and the same lists in the same
    order, so a per-island result depends on the partition alone.

    Every pass is lane-strided over the trees, the edges or the islands and fenced from the next by a block barrier,
    and n_islands holds the change flag of the hooking rounds until the label count takes its place. The components
    are found by min-label hooking: an edge between two components points the larger label at the smaller, every tree
    then jumps to its root, until a pass meets no edge across two components. The roots are then ranked in tree order
    by a subgroup scan, the counts summed by atomics, the starts scanned, and each tree writes its own links and dofs
    after those of the earlier trees of its island, so an item's position depends only on the earlier items.
    """
    _K = qd.static(32)
    n_trees = rigid_info.trees_root_idx.shape[0]
    n_links = rigid_info.links_tree_idx.shape[0]
    n_dofs = constraint_state.island.dof_id.shape[0]
    n_contacts = collider_state.n_contacts[i_b]
    n_equalities = constraint_state.qd_n_equalities[i_b]
    n_edges = n_contacts + n_equalities
    if qd.static(rigid_config.use_hibernation):
        n_edges = n_edges + n_links

    i_t = tid
    while i_t < n_trees:
        constraint_state.island.trees_parent_idx[i_t, i_b] = i_t
        constraint_state.island.dof_slices.n[i_t, i_b] = 0
        if qd.static(rigid_config.use_hibernation):
            constraint_state.island.link_slices.n[i_t, i_b] = 0
        i_t = i_t + _K
    qd.simt.block.sync()

    # Hooking rounds over the edges, until none crosses two components. A hook may overwrite one made concurrently on
    # the same root: the edge behind the lost hook stays, and the next round meets it again.
    while True:
        if tid == 0:
            constraint_state.island.n_islands[i_b] = 0
        qd.simt.block.sync()
        i_e = tid
        while i_e < n_edges:
            i_ta, i_tb = func_edge_trees(
                i_e, i_b, n_contacts, n_equalities, collider_state, constraint_state, dyn_info, rigid_info, rigid_config
            )
            if i_ta >= 0 and i_tb >= 0:
                root_a = func_tree_component(i_ta, i_b, constraint_state)
                root_b = func_tree_component(i_tb, i_b, constraint_state)
                if root_a != root_b:
                    qd.atomic_min(
                        constraint_state.island.trees_parent_idx[qd.max(root_a, root_b), i_b], qd.min(root_a, root_b)
                    )
                    constraint_state.island.n_islands[i_b] = 1
            i_e = i_e + _K
        qd.simt.block.sync()
        is_changed = constraint_state.island.n_islands[i_b]
        qd.simt.block.sync()
        if is_changed == 0:
            break
        # Every tree jumps to its root. A parent only ever moves to a smaller label, so a walk that meets another
        # lane's write still lands on an ancestor.
        i_t = tid
        while i_t < n_trees:
            constraint_state.island.trees_parent_idx[i_t, i_b] = func_tree_component(i_t, i_b, constraint_state)
            i_t = i_t + _K
        qd.simt.block.sync()

    # Label the components in ascending order of their root, a subgroup scan over the roots per chunk of _K trees. A
    # dof-less tree stays its own root, no edge reaching it, and labels no island.
    n_islands = 0
    for i_chunk in range((n_trees + _K - 1) // _K):
        i_t = i_chunk * _K + tid
        is_root = 0
        if i_t < n_trees:
            # The padding slot of a tree-less scene labels no island, see func_build_islands
            if constraint_state.island.trees_parent_idx[i_t, i_b] == i_t and rigid_info.trees_n_dofs[i_t] > 0:
                is_root = 1
        roots_incl = qd.simt.subgroup.inclusive_add(is_root)
        if is_root == 1:
            constraint_state.island.trees_island_idx[i_t, i_b] = n_islands + roots_incl - 1
        n_islands = n_islands + qd.simt.subgroup.broadcast(roots_incl, qd.u32(_K - 1))
    qd.simt.block.sync()
    i_t = tid
    while i_t < n_trees:
        constraint_state.island.trees_island_idx[i_t, i_b] = constraint_state.island.trees_island_idx[
            constraint_state.island.trees_parent_idx[i_t, i_b], i_b
        ]
        i_t = i_t + _K
    if tid == 0:
        constraint_state.island.n_islands[i_b] = n_islands
    qd.simt.block.sync()

    # Per-island link and dof counts
    i_t = tid
    while i_t < n_trees:
        i_island = constraint_state.island.trees_island_idx[i_t, i_b]
        if i_island >= 0:
            qd.atomic_add(constraint_state.island.dof_slices.n[i_island, i_b], rigid_info.trees_n_dofs[i_t])
            if qd.static(rigid_config.use_hibernation):
                qd.atomic_add(constraint_state.island.link_slices.n[i_island, i_b], rigid_info.trees_n_links[i_t])
        i_t = i_t + _K
    qd.simt.block.sync()

    # The list starts by a scan over the islands, the cursors of the fill starting there, the inertia and the sleeping
    # flag of every island from their neutral values.
    links_carry = 0
    dofs_carry = 0
    for i_chunk in range((n_islands + _K - 1) // _K):
        i_island = i_chunk * _K + tid
        n_links_island = 0
        n_dofs_island = 0
        if i_island < n_islands:
            n_dofs_island = constraint_state.island.dof_slices.n[i_island, i_b]
            if qd.static(rigid_config.use_hibernation):
                n_links_island = constraint_state.island.link_slices.n[i_island, i_b]
        links_incl = qd.simt.subgroup.inclusive_add(n_links_island)
        dofs_incl = qd.simt.subgroup.inclusive_add(n_dofs_island)
        if i_island < n_islands:
            link_list_start = links_carry + links_incl - n_links_island
            dof_list_start = dofs_carry + dofs_incl - n_dofs_island
            if qd.static(rigid_config.use_hibernation):
                constraint_state.island.link_slices.start[i_island, i_b] = link_list_start
                constraint_state.island.link_slices.curr[i_island, i_b] = link_list_start
            constraint_state.island.dof_slices.start[i_island, i_b] = dof_list_start
            constraint_state.island.dof_slices.curr[i_island, i_b] = dof_list_start
            constraint_state.island.inertia[i_island, i_b] = 0.0
            if qd.static(rigid_config.use_hibernation):
                constraint_state.island.is_hibernated[i_island, i_b] = 1
        links_carry = links_carry + qd.simt.subgroup.broadcast(links_incl, qd.u32(_K - 1))
        dofs_carry = dofs_carry + qd.simt.subgroup.broadcast(dofs_incl, qd.u32(_K - 1))
    qd.simt.block.sync()

    # The fill, per chunk of _K trees: a tree's items follow those of the earlier trees of its island, the ones of this
    # chunk counted over the chunk's trees, the ones of the earlier chunks held by the cursors. The links of a dof-less
    # tree take the label of no island.
    for i_chunk in range((n_trees + _K - 1) // _K):
        i_t = i_chunk * _K + tid
        i_island = -1
        if i_t < n_trees:
            i_island = constraint_state.island.trees_island_idx[i_t, i_b]
            i_root = rigid_info.trees_root_idx[i_t]
            i_link_pos = 0
            i_dof_pos = 0
            if i_island >= 0:
                i_dof_pos = constraint_state.island.dof_slices.curr[i_island, i_b]
                if qd.static(rigid_config.use_hibernation):
                    i_link_pos = constraint_state.island.link_slices.curr[i_island, i_b]
                for j_t in range(i_chunk * _K, i_t):
                    if constraint_state.island.trees_island_idx[j_t, i_b] == i_island:
                        i_link_pos = i_link_pos + rigid_info.trees_n_links[j_t]
                        i_dof_pos = i_dof_pos + rigid_info.trees_n_dofs[j_t]
            for i_tl in range(i_root, rigid_info.trees_link_end[i_t]):
                if rigid_info.links_tree_idx[i_tl] == i_t:
                    constraint_state.island.links_island_idx[i_tl, i_b] = i_island
                    if qd.static(rigid_config.use_hibernation):
                        if i_island >= 0:
                            constraint_state.island.link_id[i_link_pos, i_b] = i_tl
                            i_link_pos = i_link_pos + 1
            if i_island >= 0:
                dof_start_tree = rigid_info.trees_dof_start[i_t]
                dof_list_start = constraint_state.island.dof_slices.start[i_island, i_b]
                for i_d in range(dof_start_tree, dof_start_tree + rigid_info.trees_n_dofs[i_t]):
                    constraint_state.island.dof_id[i_dof_pos, i_b] = i_d
                    constraint_state.island.dof_local_pos[i_d, i_b] = i_dof_pos - dof_list_start
                    constraint_state.island.dofs_island_idx[i_d, i_b] = i_island
                    i_dof_pos = i_dof_pos + 1
        qd.simt.block.sync()
        if i_island >= 0:
            qd.atomic_add(constraint_state.island.dof_slices.curr[i_island, i_b], rigid_info.trees_n_dofs[i_t])
            if qd.static(rigid_config.use_hibernation):
                qd.atomic_add(constraint_state.island.link_slices.curr[i_island, i_b], rigid_info.trees_n_links[i_t])
        qd.simt.block.sync()
    i_island = tid
    while i_island < n_islands:
        constraint_state.island.dof_range_start[i_island, i_b] = func_dof_range_start(i_island, i_b, constraint_state)
        i_island = i_island + _K

    # An island is hibernated unless one of its links is awake
    if qd.static(rigid_config.use_hibernation):
        i_l = tid
        while i_l < n_links:
            i_island = constraint_state.island.links_island_idx[i_l, i_b]
            if i_island >= 0 and not dyn_state.links.is_hibernated[i_l, i_b]:
                constraint_state.island.is_hibernated[i_island, i_b] = 0
            i_l = i_l + _K

    # The inertia of every island, the mass diagonal summed over the dof list per chunk of _K dofs: the lanes of one
    # island reduce as one segment, whose last lane adds it into the island's total. The segments of an island come one
    # chunk after the other, so its total adds up in list order.
    for i_chunk in range((n_dofs + _K - 1) // _K):
        i_pos = i_chunk * _K + tid
        i_island = -1
        mass = gs.qd_float(0.0)
        if i_pos < n_dofs:
            i_d = constraint_state.island.dof_id[i_pos, i_b]
            i_island = constraint_state.island.dofs_island_idx[i_d, i_b]
            mass = rigid_info.mass_mat[i_d, i_d, i_b]
        i_island_prev = qd.simt.subgroup.shuffle_up(i_island, qd.u32(1))
        i_island_next = qd.simt.subgroup.shuffle_down(i_island, qd.u32(1))
        is_head = 1
        if tid > 0 and i_island_prev == i_island:
            is_head = 0
        total = qd.simt.subgroup.segmented_reduce_add_tiled(mass, is_head, 5)
        if i_island >= 0 and (tid == _K - 1 or i_island_next != i_island):
            constraint_state.island.inertia[i_island, i_b] = constraint_state.island.inertia[i_island, i_b] + total
        qd.simt.block.sync()


@qd.func
def func_contact_island(
    i_col, i_b, collider_state: array_class.ColliderState, constraint_state: array_class.ConstraintState
):
    # A contact belongs to the island of its dof-carrying endpoint: both endpoints share an island when both carry dofs,
    # since the contact unioned them, otherwise one side is a fixed body.
    i_la = collider_state.contact_data.link_a[i_col, i_b]
    i_island = constraint_state.island.links_island_idx[i_la, i_b]
    if i_island < 0:
        i_lb = collider_state.contact_data.link_b[i_col, i_b]
        i_island = constraint_state.island.links_island_idx[i_lb, i_b]
    return i_island


@qd.func
def func_contact_tree_slots(
    i_col,
    i_b,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
):
    """Island-local tree slots (rcm_tree_pos) of a contact's two endpoints, -1 for a fixed / dof-less side."""
    i_la = collider_state.contact_data.link_a[i_col, i_b]
    i_lb = collider_state.contact_data.link_b[i_col, i_b]
    i_ta = rigid_info.links_tree_idx[i_la]
    i_tb = rigid_info.links_tree_idx[i_lb]
    if i_ta >= 0 and constraint_state.island.trees_island_idx[i_ta, i_b] >= 0:
        i_ta = constraint_state.island.rcm_tree_pos[i_ta, i_b]
    else:
        i_ta = -1
    if i_tb >= 0 and constraint_state.island.trees_island_idx[i_tb, i_b] >= 0:
        i_tb = constraint_state.island.rcm_tree_pos[i_tb, i_b]
    else:
        i_tb = -1
    return i_ta, i_tb


@qd.func
def func_reorder_island_dofs(
    i_b,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
):
    """Fill-reducing DOF reordering for the CPU per-island skyline path: rebuild each island's dof_id in reverse
    Cuthill-McKee order of its kinematic trees over the contact adjacency, instead of ascending tree order.

    The build-time DOF order says nothing about which bodies end up in contact, so a settled pile otherwise produces a
    near-dense skyline; ordering trees by contact adjacency shrinks the envelope the factor, rank-1 updates and
    triangular solves sweep. Trees stay contiguous with their DOFs in original relative order, preserving the local
    contiguity of the mass blocks. Every per-island consumer addresses nt_H through (dof_id, dof_local_pos) pairs with
    island-local triangle orientation, so a non-monotonic dof_id only changes where blocks are stored.
    """
    n_trees = rigid_info.trees_root_idx.shape[0]
    n_islands = constraint_state.island.n_islands[i_b]
    n_contacts = collider_state.n_contacts[i_b]

    # Per-island contact lists (island -> contact ranges in contact_id), in contact_sort_idx order
    for i_island in range(n_islands):
        constraint_state.island.contact_slices.n[i_island, i_b] = 0
    for i_c in range(n_contacts):
        i_island = func_contact_island(collider_state.contact_sort_idx[i_c, i_b], i_b, collider_state, constraint_state)
        if i_island >= 0:
            constraint_state.island.contact_slices.n[i_island, i_b] = (
                constraint_state.island.contact_slices.n[i_island, i_b] + 1
            )
    contact_list_start = 0
    for i_island in range(n_islands):
        constraint_state.island.contact_slices.start[i_island, i_b] = contact_list_start
        constraint_state.island.contact_slices.curr[i_island, i_b] = contact_list_start
        contact_list_start = contact_list_start + constraint_state.island.contact_slices.n[i_island, i_b]
    for i_c in range(n_contacts):
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        i_island = func_contact_island(i_col, i_b, collider_state, constraint_state)
        if i_island >= 0:
            constraint_state.island.contact_id[constraint_state.island.contact_slices.curr[i_island, i_b], i_b] = i_col
            constraint_state.island.contact_slices.curr[i_island, i_b] = (
                constraint_state.island.contact_slices.curr[i_island, i_b] + 1
            )

    # The per-island scratch rows live side by side: island i_island's slots start where the earlier islands' end
    tree_base = 0
    for i_island in range(n_islands):
        con_base = constraint_state.island.contact_slices.start[i_island, i_b]
        n_island_contacts = constraint_state.island.contact_slices.n[i_island, i_b]

        # Island-local tree slots, in ascending tree order
        n_island_trees = 0
        for i_t in range(n_trees):
            if constraint_state.island.trees_island_idx[i_t, i_b] == i_island:
                constraint_state.island.rcm_tree_pos[i_t, i_b] = n_island_trees
                n_island_trees = n_island_trees + 1

        # Reordering cannot shrink the envelope of at most two trees; keep the ascending build order there so small
        # islands stay bit-identical to the unordered path (and the common tiny-island case costs nothing).
        if n_island_trees <= 2:
            tree_base = tree_base + n_island_trees
            continue

        # Contact degree per tree (duplicate contacts inflate degrees, which only biases the tie-break)
        for i_slot in range(n_island_trees):
            constraint_state.island.rcm_tree_degree[tree_base + i_slot, i_b] = 0
            constraint_state.island.rcm_tree_is_ordered[tree_base + i_slot, i_b] = False
        for i_c_ in range(n_island_contacts):
            i_col = constraint_state.island.contact_id[con_base + i_c_, i_b]
            i_ta, i_tb = func_contact_tree_slots(i_col, i_b, collider_state, constraint_state, rigid_info)
            if i_ta >= 0 and i_tb >= 0 and i_ta != i_tb:
                constraint_state.island.rcm_tree_degree[tree_base + i_ta, i_b] = (
                    constraint_state.island.rcm_tree_degree[tree_base + i_ta, i_b] + 1
                )
                constraint_state.island.rcm_tree_degree[tree_base + i_tb, i_b] = (
                    constraint_state.island.rcm_tree_degree[tree_base + i_tb, i_b] + 1
                )

        # Cuthill-McKee: BFS from the lowest-degree unordered tree, appending each frontier sorted by degree
        n_ordered = 0
        i_head = 0
        while n_ordered < n_island_trees:
            i_slot_start = -1
            for i_slot in range(n_island_trees):
                if not constraint_state.island.rcm_tree_is_ordered[tree_base + i_slot, i_b] and (
                    i_slot_start == -1
                    or constraint_state.island.rcm_tree_degree[tree_base + i_slot, i_b]
                    < constraint_state.island.rcm_tree_degree[tree_base + i_slot_start, i_b]
                ):
                    i_slot_start = i_slot
            constraint_state.island.rcm_tree_order[tree_base + n_ordered, i_b] = i_slot_start
            constraint_state.island.rcm_tree_is_ordered[tree_base + i_slot_start, i_b] = True
            n_ordered = n_ordered + 1
            while i_head < n_ordered:
                i_slot_head = constraint_state.island.rcm_tree_order[tree_base + i_head, i_b]
                n_frontier_start = n_ordered
                for i_c_ in range(n_island_contacts):
                    i_col = constraint_state.island.contact_id[con_base + i_c_, i_b]
                    i_ta, i_tb = func_contact_tree_slots(i_col, i_b, collider_state, constraint_state, rigid_info)
                    i_slot_next = -1
                    if i_ta == i_slot_head and i_tb >= 0:
                        i_slot_next = i_tb
                    elif i_tb == i_slot_head and i_ta >= 0:
                        i_slot_next = i_ta
                    if (
                        i_slot_next >= 0
                        and not constraint_state.island.rcm_tree_is_ordered[tree_base + i_slot_next, i_b]
                    ):
                        # Insert into the current frontier keeping it sorted by ascending degree
                        i_ins = n_ordered
                        while i_ins > n_frontier_start and (
                            constraint_state.island.rcm_tree_degree[
                                tree_base + constraint_state.island.rcm_tree_order[tree_base + i_ins - 1, i_b], i_b
                            ]
                            > constraint_state.island.rcm_tree_degree[tree_base + i_slot_next, i_b]
                        ):
                            constraint_state.island.rcm_tree_order[tree_base + i_ins, i_b] = (
                                constraint_state.island.rcm_tree_order[tree_base + i_ins - 1, i_b]
                            )
                            i_ins = i_ins - 1
                        constraint_state.island.rcm_tree_order[tree_base + i_ins, i_b] = i_slot_next
                        constraint_state.island.rcm_tree_is_ordered[tree_base + i_slot_next, i_b] = True
                        n_ordered = n_ordered + 1
                i_head = i_head + 1

        # Rebuild dof_id with trees in REVERSE Cuthill-McKee order, each tree's dofs in ascending order. The per-slot
        # tree scan is O(n_island_trees * n_trees), dominated by the BFS neighbor sweep above (O(n_island_trees *
        # n_island_contacts)), so it is not worth a per-slot tree buffer. The list leaves build order, so the sweeps
        # read it.
        constraint_state.island.dof_range_start[i_island, i_b] = -1
        i_dof_curr = constraint_state.island.dof_slices.start[i_island, i_b]
        for i_slot_ in range(n_island_trees):
            i_slot = constraint_state.island.rcm_tree_order[tree_base + (n_island_trees - 1 - i_slot_), i_b]
            for i_t in range(n_trees):
                if (
                    constraint_state.island.trees_island_idx[i_t, i_b] == i_island
                    and constraint_state.island.rcm_tree_pos[i_t, i_b] == i_slot
                ):
                    dof_start_tree = rigid_info.trees_dof_start[i_t]
                    for i_d in range(dof_start_tree, dof_start_tree + rigid_info.trees_n_dofs[i_t]):
                        constraint_state.island.dof_id[i_dof_curr, i_b] = i_d
                        constraint_state.island.dof_local_pos[i_d, i_b] = (
                            i_dof_curr - constraint_state.island.dof_slices.start[i_island, i_b]
                        )
                        i_dof_curr = i_dof_curr + 1
        tree_base = tree_base + n_island_trees


@qd.func
def func_sort_contacts_coop(
    i_b,
    tid,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
):
    """Sort the contacts of one env in the order of func_sort_contacts with the _K lanes of its block.

    Every contact takes the rank of its key among all of them, equal keys ranking by input position, which is the
    stable order the serial insertion sort yields.

    The input order moves to contact_id and the position keys to contact_sort_key first, so the ranks scatter straight
    into contact_sort_idx.
    """
    _K = qd.static(32)
    n = collider_state.n_contacts[i_b]
    i_c = tid
    while i_c < n:
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        geom_b = collider_state.contact_data.geom_b[i_col, i_b]
        pos = gu.qd_inv_transform_by_quat(
            collider_state.contact_data.pos[i_col, i_b] - dyn_state.geoms.pos[geom_b, i_b],
            dyn_state.geoms.quat[geom_b, i_b],
        )
        constraint_state.island.contact_id[i_c, i_b] = i_col
        collider_state.contact_sort_key[i_c, i_b] = func_contact_order_key(pos)
        i_c = i_c + _K
    qd.simt.block.sync()
    i_c = tid
    while i_c < n:
        i_col = constraint_state.island.contact_id[i_c, i_b]
        geom_a = collider_state.contact_data.geom_a[i_col, i_b]
        geom_b = collider_state.contact_data.geom_b[i_col, i_b]
        key = collider_state.contact_sort_key[i_c, i_b]
        rank = 0
        for j_c in range(n):
            j_col = constraint_state.island.contact_id[j_c, i_b]
            geom_a_j = collider_state.contact_data.geom_a[j_col, i_b]
            precedes = geom_a_j < geom_a
            if not precedes and geom_a_j == geom_a:
                geom_b_j = collider_state.contact_data.geom_b[j_col, i_b]
                if geom_b_j < geom_b:
                    precedes = True
                elif geom_b_j == geom_b:
                    key_j = collider_state.contact_sort_key[j_c, i_b]
                    precedes = key_j < key or (key_j == key and j_c < i_c)
            if precedes:
                rank = rank + 1
        collider_state.contact_sort_idx[rank, i_b] = i_col
        i_c = i_c + _K


@qd.func
def func_sort_contacts(
    i_b,
    contact_idx: qd.Tensor,
    n,
    contacts_pos: qd.Tensor,
    contacts_geom_a: qd.Tensor,
    contacts_geom_b: qd.Tensor,
    geoms_pos: qd.Tensor,
    geoms_quat: qd.Tensor,
):
    """Insertion-sort the contact indices contact_idx[0 : n] of one env by a deterministic total order.

    The order is (geom_a, geom_b, then the contact position along one direction in geom_b's own frame), a pure function
    of contact data, so it is independent of the racy atomic_add narrowphase layout. The position is taken in that frame
    because a world coordinate carries the orientation of the whole scene, which makes the order of a scene and of a
    rigidly rotated copy of it differ. geom_b carries the frame since the pair is canonically ordered by geom type and a
    plane, whose frame stays put while the scene turns, sorts first. Leading with the geom pair keeps each pair's
    contacts contiguous, so the frame is fixed across every position comparison the sort reaches; the position reduces
    to one scalar there, see func_contact_order_key.

    contact_idx is collider_state.contact_sort_idx; the contact-data tensors are passed as leaves rather than the whole
    collider_state struct so that it can be sorted in place without the struct-expansion aliasing its own field.
    """
    for i_s in range(1, n):
        i_p = contact_idx[i_s, i_b]
        geom_a_p = contacts_geom_a[i_p, i_b]
        geom_b_p = contacts_geom_b[i_p, i_b]
        pos_p = gu.qd_inv_transform_by_quat(
            contacts_pos[i_p, i_b] - geoms_pos[geom_b_p, i_b], geoms_quat[geom_b_p, i_b]
        )
        j_s = i_s - 1
        while j_s >= 0:
            i_q = contact_idx[j_s, i_b]
            geom_a_q = contacts_geom_a[i_q, i_b]
            geom_b_q = contacts_geom_b[i_q, i_b]
            precedes = geom_a_q < geom_a_p
            if not precedes and geom_a_q == geom_a_p:
                if geom_b_q < geom_b_p:
                    precedes = True
                elif geom_b_q == geom_b_p:
                    pos_q = gu.qd_inv_transform_by_quat(
                        contacts_pos[i_q, i_b] - geoms_pos[geom_b_q, i_b], geoms_quat[geom_b_q, i_b]
                    )
                    precedes = func_contact_order_key(pos_q) <= func_contact_order_key(pos_p)
            if precedes:
                break
            contact_idx[j_s + 1, i_b] = i_q
            j_s = j_s - 1
        contact_idx[j_s + 1, i_b] = i_p
