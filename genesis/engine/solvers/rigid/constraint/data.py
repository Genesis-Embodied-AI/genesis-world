"""The state of the constraint solve of the rigid solver and of its islands."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.data import BACKWARD, DOFS, LINKS, TREES
from genesis.utils.data import ENVS, Data, DataKind, Size, field


# Sizes of the constraint solve
CONSTRAINTS = Size("constraints")
CONTACTS = Size("contacts")
# The rows of each elliptic cone, and of all of them
ROWS_PER_CONTACT = Size("rows_per_contact")
CONE_ROWS = CONTACTS * ROWS_PER_CONTACT
# Fused updates of the per-island factor, one slot each (see nt_vec)
NT_VEC_SLOTS = Size("nt_vec_slots")
# Dofs a Jacobian row reaches at most (see jac_dofs_idx)
JAC_DOFS = Size("jac_dofs")
# Lanes of the cooperative noslip sweep, one without it (see noslip_MinvJT)
NOSLIP_LANES = Size("noslip_lanes")
# Size classes of the work-lists of the tiled island seed (see factor_worklist_i_b)
TILE_CLASSES = Size("tile_classes")

# The rows of the constraint state and of the islands the cooperative kernels read, one env per block, which they
# read from consecutive addresses once stored environment-first
COOPERATIVE_ROWS = "cooperative_rows"
# The Jacobian rows, read whole for one env by the cooperative kernels, CPU threads and serialized execution (see jac)
JACOBIAN_ROWS = "jacobian_rows"
# The rows CPU threads and serialized execution read env by env, in the outermost loop of every constraint kernel
SERIAL_ROWS = "serial_rows"
NEWTON = "newton"
CONE_INCREMENTAL = "cone_incremental"
SIGNORINI = "signorini"
CONE_FREE_REUSE = "cone_free_reuse"
NOSLIP = "noslip"
NOSLIP_COOPERATIVE = "noslip_cooperative"
HIBERNATION = "hibernation"
# The reverse Cuthill-McKee tree reorder of the CPU per-island skyline path (see rcm_tree_pos)
RCM = "rcm"
# The contact lists of the islands, of the skyline reorder or of the cooperative contact sort (see contact_id)
ISLAND_CONTACTS = "island_contacts"
WORKLISTS = "worklists"


class IslandSlices(Data, kind=DataKind.SCRATCH, shape=(TREES, ENVS)):
    """The slice of a packed index array that each island of each environment owns."""

    # Slices of a packed id array per (island, env): the items of island i_island are id[start[i_island, i_b] :
    # start[i_island, i_b] + n[i_island, i_b]]. curr is the write cursor of the partition build, start + n once built.
    # An island holds at least one tree, hence one slice per tree.
    curr: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)
    n: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)
    start: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)


class IslandState(Data, kind=DataKind.SCRATCH):
    """The islands of the kinematic trees of each environment, with their members and per-island solver state."""

    # Partition of the dof-carrying kinematic trees (see trees_root_idx in RigidInfo) into islands, the connected
    # components of the trees under the contact, equality and hibernation couplings, rebuilt every step by island.py.
    # trees_parent_idx is the union-find forest over the trees, trees_island_idx the island of each tree and
    # links_island_idx that of each link, -1 for a static link and for a dof-less tree and its links. link_slices maps
    # an island to its slice of link_id, dof_slices to its slice of dof_id (island-local dof -> global dof, ascending
    # unless the CPU skyline path reorders it by contact adjacency).
    trees_parent_idx: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    trees_island_idx: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    links_island_idx: qd.Tensor = field(int, shape=(LINKS, ENVS), labels=COOPERATIVE_ROWS)
    n_islands: qd.Tensor = field(int, shape=(ENVS,))
    link_slices: IslandSlices = field(labels=HIBERNATION)
    link_id: qd.Tensor = field(int, shape=(LINKS, ENVS), labels=(COOPERATIVE_ROWS, HIBERNATION))
    dof_slices: IslandSlices
    dof_id: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Inverse of dof_id: dof_local_pos[d] is the island-local position of global dof d
    dof_local_pos: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    dofs_island_idx: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # First dof of an island whose dof list holds consecutive dofs in ascending order, -1 otherwise (see func_list_item
    # in linesearch.py).
    dof_range_start: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    # Skyline envelope of the CPU per-island path: dof_env_start_local[dof_slices.start[i] + ld] is the smallest
    # island-local column structurally nonzero in local row ld of island i's Hessian block, 0 (dense) when uncomputed.
    # dof_env_col_end is its transpose, the largest local row whose envelope reaches column ld, which bounds the
    # column-oriented factor and solve sweeps to the band: a tall island of a few hundred dofs otherwise tests every
    # row below the diagonal against its envelope, an O(n^2) scan per sweep. It has no safe uncomputed default (0
    # truncates), so only the CPU per-island path reads it.
    dof_env_start_local: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    dof_env_col_end: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Contact lists of the islands, for the CPU skyline reorder (see rcm_tree_pos). contact_id also carries the input
    # order of the cooperative contact sort (see func_sort_contacts_coop in island.py).
    contact_slices: IslandSlices = field(labels=RCM)
    contact_id: qd.Tensor = field(int, shape=(CONTACTS, ENVS), labels=(COOPERATIVE_ROWS, ISLAND_CONTACTS))
    constraint_slices: IslandSlices
    constraint_id: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Island of each constraint, -1 for one touching no island
    constraint_island_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # is_hibernated marks an island whose every link is asleep. hibernated_next_link chains the links of a sleeping
    # component, which the partition build re-unions before labeling, so the component stays one island across steps
    # although its bodies generate no live contacts.
    is_hibernated: qd.Tensor = field(int, shape=(TREES, ENVS), kind=DataKind.STATE, labels=HIBERNATION)
    hibernated_next_link: qd.Tensor = field(int, shape=(LINKS, ENVS), kind=DataKind.STATE, labels=HIBERNATION)
    # (env, island) work-lists of the cooperative per-island factor+solve, one per island size class (see
    # island_tile_caps): class c holds its factor_worklist_size[c] awake islands in its own region of the two index
    # lists, of n_trees * _B slots each, in the order of the atomic reservation of the partition pass. Read where an env
    # can hold several islands (see func_island_tiled_factor_solve_all in solver.py).
    factor_worklist_i_b: qd.Tensor = field(int, shape=(TILE_CLASSES * TREES * ENVS,), labels=WORKLISTS)
    factor_worklist_i_island: qd.Tensor = field(int, shape=(TILE_CLASSES * TREES * ENVS,), labels=WORKLISTS)
    factor_worklist_size: qd.Tensor = field(int, shape=(TILE_CLASSES,), labels=WORKLISTS)
    # Scratch of the reverse Cuthill-McKee tree reorder of the CPU per-island skyline path: the island-local slot, the
    # contact degree and the ordered flag of each tree, and the resulting visit order.
    rcm_tree_pos: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_degree: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_is_ordered: qd.Tensor = field(bool, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_order: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    # Per-island Newton / CG iteration state (see linesearch.py), indexed [i_island, i_b]: inertia is the trace of the
    # mass matrix over the island's dofs, the scale of its convergence tests, and improved says whether the island
    # still iterates, every pass of an iteration being confined to the islands that do. ls_improvement carries
    # -cost(alpha_accepted) in shifted form, cost(alpha) - cost(0), which stays resolvable in float32 near convergence.
    inertia: qd.Tensor = field(float, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    improved: qd.Tensor = field(bool, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    ls_improvement: qd.Tensor = field(float, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)


class ConstraintState(Data, kind=DataKind.SCRATCH):
    """The constraint solve of each environment: rows, Jacobian, Hessian factor, iterates and gradients."""

    # /!\ The declaration order is the allocation order, which changing may reduce runtime speed by >10% /!\
    n_constraints: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    qd_n_equalities: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)
    n_constraints_equality: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    n_constraints_frictionloss: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    # Elliptic-cone contact rows, rows_per_contact per contact, zero for the pyramidal cone. They occupy
    # [ne + n_frictionloss, ne + n_frictionloss + n_cone), the start of the collision segment.
    n_constraints_cone: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    is_warmstart: qd.Tensor = field(bool, shape=(ENVS,), kind=DataKind.STATE)
    improved: qd.Tensor = field(bool, shape=(ENVS,))
    cost_ws: qd.Tensor = field(float, shape=(ENVS,))
    cost: qd.Tensor = field(float, shape=(ENVS,))
    Ma: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    Ma_ws: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    grad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    Mgrad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Cone-row residuals of the previous iteration, for the CPU elliptic cone: the incremental factor downdates the
    # previous coupled cone block by a rank-3 update, after the linesearch overwrote Jaref.
    cone_prev_jaref: qd.Tensor = field(float, shape=(CONE_ROWS, ENVS), labels=(SERIAL_ROWS, CONE_INCREMENTAL))
    # Per contact under 'signorini': the latched normal force scaling its friction discs (negative until seeded, see
    # func_relatch_cone in solver.py), and the gap between the iterate's normal force and the latch at the last update.
    cone_latch: qd.Tensor = field(float, shape=(CONTACTS, ENVS), labels=SIGNORINI)
    cone_latch_gap: qd.Tensor = field(float, shape=(CONTACTS, ENVS), labels=SIGNORINI)
    search: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    qfrc_constraint: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    qacc: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    qacc_ws: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    mv: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    cg_prev_grad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    cg_prev_Mgrad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Rank-1 vectors of the incremental Cholesky update, slot-minor at [i_d * n_slots + i_u]: one slot per fused update
    # of the per-island factor (see func_rank_batch_update_island), one for the factor of a single-island scene. The
    # flat index keeps the rank and the layout of the other dof vectors.
    nt_vec: qd.Tensor = field(float, shape=(DOFS * NT_VEC_SLOTS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    # Cholesky factor L of the Hessian H = L @ L.T, in its lower triangle. The strictly upper triangle is undefined,
    # except under enable_cone_free_hessian_reuse: the mirror of each used slot holds the Hessian without its cone
    # blocks (M + J^T D J of the active rows), kept across the Newton iterations of a step by signed flip scatters,
    # its diagonal in nt_H_cone_free_diag. A rebuild restores the mirror into the lower triangle and adds the current
    # cone blocks, skipping the full J^T D J assembly. It aliases the scratch of the register-tiled mass factor when
    # there is one (see the constraint solver).
    # TODO: Allocate the lower triangle alone.
    nt_H: qd.Tensor = field(float, shape=(ENVS, DOFS, DOFS), labels=NEWTON)
    # Jacobi scale of each dof under enable_jacobi_equilibration, s_i = 1 / sqrt(diag(M + J^T D J)_i), 1 on an empty
    # diagonal, refreshed before every rebuild (see func_jacobi_scale). The assembly writes H scaled, nt_H holds the
    # factor of S H S, the update vectors are scaled when built, and the batch solve scales grad and Mgrad, so Mgrad =
    # H^-1 grad exactly. The unit diagonal keeps the mixed units of the Hessian within float32 conditioning at any
    # geometry scale and makes the EPS pivot floor relative.
    nt_jacobi: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    # The constraints whose active state changed, which the incremental Cholesky update iterates over
    incr_changed_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    incr_n_changed: qd.Tensor = field(int, shape=(ENVS,))
    active: qd.Tensor = field(bool, shape=(CONSTRAINTS, ENVS), kind=DataKind.DERIVED, labels=COOPERATIVE_ROWS)
    prev_active: qd.Tensor = field(bool, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    diag: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    aref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    Jaref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # The friction loss of a frictionloss row. The normal row of an elliptic cone holds the sliding friction coefficient
    # of its contact and its spin row the torsional one, its tangent rows 0.
    efc_frictionloss: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    efc_force: qd.Tensor = field(
        float, shape=(CONSTRAINTS, ENVS), kind=DataKind.DERIVED, labels=COOPERATIVE_ROWS, grad=True
    )
    efc_D: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    jv: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Stored environment-first for the cooperative kernels and serialized execution (see JACOBIAN_ROWS). The lanes of
    # the cooperative J^T @ efc_force and Hessian staging then read consecutive dofs of a row, whose support spans a
    # cache line or two.
    jac: qd.Tensor = field(float, shape=(CONSTRAINTS, DOFS, ENVS), labels=JACOBIAN_ROWS)
    # The support of each row: the dofs of the kinematic chains of the two links it couples, or the two dofs of a joint
    # equality
    jac_dofs_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, JAC_DOFS, ENVS), labels=JACOBIAN_ROWS)
    jac_n_dofs: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Backward gradients
    dL_dqacc: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    dL_dM: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), labels=BACKWARD)
    dL_djac: qd.Tensor = field(float, shape=(CONSTRAINTS, DOFS, ENVS), labels=BACKWARD)
    dL_daref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    dL_defc_D: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    dL_dforce: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    # Scratch of the linear solve of the backward pass
    bw_u: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_r: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_p: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_Ap: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_Ju: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    # Profiling timers, 64-bit where the backend supports it
    timers: qd.Tensor = field(lambda: qd.i32 if gs.backend == gs.metal else qd.i64, shape=(10, ENVS))
    # Iteration of the solver loop, from 0
    solver_iter_counter: qd.Tensor = field(qd.i32, shape=())
    # An ndarray on every backend, since graph.do_while takes the same ndarray on every call
    graph_counter: qd.Tensor = field(qd.i32, shape=(), ndarray=True)
    # Iterations left to the graph loop past the current one, resumed once an env iterates on
    graph_counter_next: qd.Tensor = field(qd.i32, shape=())
    # Diagonal of the Hessian without its cone blocks (see nt_H)
    nt_H_cone_free_diag: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=CONE_FREE_REUSE)
    # M^-1 J^T of the two rows of the pair the noslip sweep updates, in the two columns of the env, or of the lane of
    # the cooperative sweep at [i_d, 2 * (i_b * 32 + tid) + i_r] (see func_noslip_update_row in noslip.py)
    noslip_MinvJT: qd.Tensor = field(float, shape=(DOFS, 2 * NOSLIP_LANES * ENVS), labels=NOSLIP)
    # Row coloring of the cooperative noslip sweep (see func_color_rows_batch in noslip.py): the color of each row, the
    # color count of each island and the next free color of the mass block starting at each dof
    noslip_rows_color: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    noslip_islands_n_colors: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    noslip_blocks_n_colors: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    island: IslandState
