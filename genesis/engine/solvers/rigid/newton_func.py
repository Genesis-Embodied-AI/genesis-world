"""Stages of Rigid's dofs as a participant of the NewtonSolver, in the joint solve with the solvers coupled with Rigid.

The Rigid unknown is x = dt^2 qacc per dof, so that a link point moves by dt v + J_pt x over the substep. Rigid's convex
primal cost (its Gauss term and constraint rows, as Rigid assembled them for the substep) times dt^4 joins the joint
energy in the displacement units of the material point method (MPM) energy. Its gradient is dt^2 times that of Rigid,
and its Hessian is the M + J^T D J of Rigid. One preconditioned conjugate gradient (PCG) solves the joint Newton system
with a block diagonal preconditioner, and Rigid's block is its island Cholesky factor of M + J^T D J corrected by the
contact stiffness of the links in contact (see func_rigid_factor). The couplings act on the links through their spatial
displacement and spatial force: a point at arm r from the root center of mass (COM) of its link moves by dr = V_lin +
V_ang x r for the link's spatial displacement V = sum over the dofs of its chain of cdof x_d, and a force g on it pushes
the dofs of that chain with the spatial force (g, r x g).
"""

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import ConstraintState, DofsState, DynInfo, DynState, LinksInfo, RigidInfo
from genesis.utils.data import ENVS, Data, DataKind, Size, field

from .abd.forward_dynamics import func_solve_mass_batch
from .constraint.island import func_group_constraints_by_island
from .constraint.solver import (
    func_cholesky_solve_batch,
    func_hessian_and_cholesky_factor_direct,
    func_update_constraint,
    initialize_Jaref,
    initialize_Ma,
)


class RigidNewtonInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """Scene values of Rigid's stages in the joint solve, as device scalars."""

    substep_dt: qd.Tensor = field(float, ndarray=True)


# The dofs, the links and the links in contact of Rigid in the joint solve, and the spatial components of the latter
DOFS = Size("dofs")
LINKS = Size("links")
CONTACT_LINKS = Size("contact_links")
CONTACT_COLS = 6 * CONTACT_LINKS


class RigidNewtonScratch(Data, kind=DataKind.SCRATCH):
    """Rigid unknowns of the joint Newton solve and the Rigid part of its conjugate gradient vectors.

    The Rigid unknown is x = dt^2 qacc per dof and environment, with Rigid's own cost scaled by dt^4 to the displacement
    units of the material point method (MPM) energy. `qacc`, `Ma` and `cost` hold Rigid's evaluation of x. `r`, `z`,
    `p`, `ap` and `sol` are the Rigid rows of the conjugate gradient residual, preconditioned residual, direction,
    operator image and solution. Per link, `link_motion_vel` / `link_motion_ang` hold the spatial displacement of a dof
    vector and `link_force_vel` / `link_force_ang` accumulate a spatial force, about the root center of mass, for the
    couplings acting on the links.
    """

    x: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    x_temp: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    dx: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    grad: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    qacc: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    Ma: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    r: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    z: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    p: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    ap: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    sol: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    cost: qd.Tensor = field(float, shape=(ENVS,), ndarray=True)
    link_motion_vel: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    link_motion_ang: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    link_force_vel: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    link_force_ang: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    # The contact stiffness the couplings put on the links, which the preconditioner takes in by the Woodbury identity
    # on Rigid's factor A. Per link, its 6 x 6 spatial stiffness K_l. Per environment, the links in contact (at most
    # `contact_links`, the first dimension of `contact_links` that the kernels read), the cdof map U of their chains
    # with one column per spatial component, W = A^-1 U, and the lower-upper (LU) factor of S = I + K U^T W with its
    # row pivots. `solve_rhs` / `solve_out` stage one column of a solve.
    link_K: qd.Tensor = field((float, (6, 6)), shape=(LINKS, ENVS), ndarray=True)
    contact_links: qd.Tensor = field(int, shape=(CONTACT_LINKS, ENVS), ndarray=True)
    n_links_in_contact: qd.Tensor = field(int, shape=(ENVS,), ndarray=True)
    woodbury_U: qd.Tensor = field(float, shape=(DOFS, CONTACT_COLS, ENVS), ndarray=True)
    woodbury_W: qd.Tensor = field(float, shape=(DOFS, CONTACT_COLS, ENVS), ndarray=True)
    capacitance_LU: qd.Tensor = field(float, shape=(CONTACT_COLS, CONTACT_COLS, ENVS), ndarray=True)
    capacitance_pivots: qd.Tensor = field(int, shape=(CONTACT_COLS, ENVS), ndarray=True)
    woodbury_y: qd.Tensor = field(float, shape=(CONTACT_COLS, ENVS), ndarray=True)
    solve_rhs: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    solve_out: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)


# ============================================ Layer 1: link kinematics =============================================


@qd.func(requires_top_level=True)
def func_link_spatial_disp(
    src: qd.template(),
    rigid_data: RigidNewtonScratch,
    dofs_state: DofsState,
    links_info: LinksInfo,
    config: qd.template(),
):
    """Set the spatial displacement of every link from a dof vector, and clear the spatial force of the link.

    The displacement is the sum over the dofs of the chain of the link of cdof src_d."""
    n_links = rigid_data.link_motion_vel.shape[0]
    n_envs = rigid_data.link_motion_vel.shape[1]
    qd.loop_config(name="mpm_link_spatial_disp")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        motion_vel = qd.Vector.zero(gs.qd_float, 3)
        motion_ang = qd.Vector.zero(gs.qd_float, 3)
        i_a = i_l
        while i_a >= 0:
            I_a = [i_a, i_b] if qd.static(config.rigid_batch_links_info) else i_a
            for i_d in range(links_info.dof_start[I_a], links_info.dof_end[I_a]):
                motion_vel += dofs_state.cdof_vel[i_d, i_b] * src[i_d, i_b]
                motion_ang += dofs_state.cdof_ang[i_d, i_b] * src[i_d, i_b]
            i_a = links_info.parent_idx[I_a]
        rigid_data.link_motion_vel[i_l, i_b] = motion_vel
        rigid_data.link_motion_ang[i_l, i_b] = motion_ang
        rigid_data.link_force_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        rigid_data.link_force_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_link_force_to_dofs(
    scale: float,
    out: qd.template(),
    rigid_data: RigidNewtonScratch,
    dofs_state: DofsState,
    links_info: LinksInfo,
    config: qd.template(),
):
    """Add the generalized force of the spatial forces accumulated on the links to a dof vector.

    out_d += scale cdof_d . f over the links whose chain holds dof d, with f the spatial force of the link."""
    n_links = rigid_data.link_force_vel.shape[0]
    n_envs = rigid_data.link_force_vel.shape[1]
    qd.loop_config(name="mpm_link_force_to_dofs")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        force_vel = rigid_data.link_force_vel[i_l, i_b]
        force_ang = rigid_data.link_force_ang[i_l, i_b]
        if force_vel.norm_sqr() + force_ang.norm_sqr() > 0.0:
            i_a = i_l
            while i_a >= 0:
                I_a = [i_a, i_b] if qd.static(config.rigid_batch_links_info) else i_a
                for i_d in range(links_info.dof_start[I_a], links_info.dof_end[I_a]):
                    value = dofs_state.cdof_vel[i_d, i_b].dot(force_vel) + dofs_state.cdof_ang[i_d, i_b].dot(force_ang)
                    qd.atomic_add(out[i_d, i_b], scale * value)
                i_a = links_info.parent_idx[I_a]


@qd.func(requires_top_level=True)
def func_clear_link_stiffness(rigid_data: RigidNewtonScratch):
    """Clear the contact stiffness of every link, before the contacts add theirs to the preconditioner."""
    qd.loop_config(name="mpm_rigid_link_K_clear")
    for i_l, i_b in qd.ndrange(rigid_data.link_K.shape[0], rigid_data.link_K.shape[1]):
        rigid_data.link_K[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)


# ============================================== Layer 1: Rigid block ===============================================


@qd.func(requires_top_level=True)
def func_rigid_begin(
    rigid_data: RigidNewtonScratch,
    dofs_state: DofsState,
    constraint_state: ConstraintState,
    newton_info: RigidNewtonInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Start the Rigid unknown from Rigid's warm start and group Rigid's assembled rows by island.

    The start point matches that of Rigid's own solve. Every island is marked as iterating, so that the factor covers
    every dof, including those of islands without rows."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    dt = newton_info.substep_dt[()]
    qd.loop_config(name="mpm_rigid_begin")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        qacc = dofs_state.acc_smooth[i_d, i_b]
        if qd.static(config.has_rigid_rows):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
                qacc = constraint_state.qacc_ws[i_d, i_b]
        rigid_data.x[i_d, i_b] = dt * dt * qacc
    if qd.static(config.has_rigid_rows):
        qd.loop_config(name="mpm_rigid_group_islands")
        for i_b in range(n_envs):
            func_group_constraints_by_island(i_b, constraint_state, rigid_config)
            for i_island in range(constraint_state.island.n_islands[i_b]):
                constraint_state.island.improved[i_island, i_b] = 1


@qd.func(requires_top_level=True)
def func_rigid_evaluate(
    rigid_data: RigidNewtonScratch,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    newton_info: RigidNewtonInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Evaluate Rigid's cost of qacc = x / dt^2 into `rigid_data.cost`, with its row activity, forces and
    qfrc_constraint.

    The cost comes from Rigid's own update of its constraint state, and reduces to the Gauss term in a scene without
    Rigid constraint rows."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    inv_dt_sq = 1.0 / (newton_info.substep_dt[()] * newton_info.substep_dt[()])
    qd.loop_config(name="mpm_rigid_qacc")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.qacc[i_d, i_b] = rigid_data.x[i_d, i_b] * inv_dt_sq
    initialize_Ma(rigid_data.Ma, rigid_data.qacc, rigid_info, rigid_config)
    if qd.static(config.has_rigid_rows):
        initialize_Jaref(qacc=rigid_data.qacc, constraint_state=constraint_state, rigid_config=rigid_config)
        func_update_constraint(
            qacc=rigid_data.qacc,
            Ma=rigid_data.Ma,
            cost=rigid_data.cost,
            dyn_state=dyn_state,
            constraint_state=constraint_state,
            rigid_config=rigid_config,
        )
    else:
        qd.loop_config(name="mpm_rigid_gauss_cost")
        for i_b in range(n_envs):
            cost = gs.qd_float(0.0)
            for i_d in range(n_dofs):
                cost += (
                    0.5
                    * (rigid_data.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b])
                    * (rigid_data.qacc[i_d, i_b] - dyn_state.dofs.acc_smooth[i_d, i_b])
                )
            rigid_data.cost[i_b] = cost


@qd.func(requires_top_level=True)
def func_rigid_energy(rigid_data: RigidNewtonScratch, energies: qd.template(), newton_info: RigidNewtonInfo):
    """Add Rigid's cost of every environment times dt^4, after func_rigid_evaluate, into its entry in `energies`."""
    dt_sq = newton_info.substep_dt[()] * newton_info.substep_dt[()]
    for i_b in range(rigid_data.x.shape[1]):
        energies[i_b] += dt_sq * dt_sq * rigid_data.cost[i_b]


@qd.func(requires_top_level=True)
def func_rigid_gradient(
    rigid_data: RigidNewtonScratch,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: RigidNewtonInfo,
    config: qd.template(),
):
    """Compute the gradient of Rigid's share of the joint energy in x, after func_rigid_evaluate at the current x.

    The gradient is dt^2 (M qacc - qf_smooth - qfrc_constraint)."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    dt_sq = newton_info.substep_dt[()] * newton_info.substep_dt[()]
    qd.loop_config(name="mpm_rigid_gradient")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        value = rigid_data.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b]
        if qd.static(config.has_rigid_rows):
            value -= constraint_state.qfrc_constraint[i_d, i_b]
        rigid_data.grad[i_d, i_b] = dt_sq * value


@qd.func
def rigid_solve(
    i_b: int,
    rhs: qd.template(),
    out: qd.template(),
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Solve the Rigid block system of one environment by Rigid's own factors.

    out = A^-1 rhs in environment i_b, with A = M + J^T D J by Rigid's island Cholesky factor, or A = M by Rigid's mass
    factor in a scene without Rigid constraint rows."""
    if qd.static(config.has_rigid_rows):
        for i_island in range(constraint_state.island.n_islands[i_b]):
            func_cholesky_solve_batch(
                i_b, i_island, rhs=rhs, out=out, constraint_state=constraint_state, rigid_config=rigid_config
            )
    else:
        func_solve_mass_batch(i_b, rhs, out, dyn_state, rigid_info, rigid_config)


@qd.func(requires_top_level=True)
def func_rigid_factor(
    rigid_data: RigidNewtonScratch,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Factor the preconditioner (A + U K U^T)^-1 of the Rigid block, after func_rigid_contact_gradient summed the
    contact stiffness K_l of each link.

    A = M + J^T D J, and U K U^T is the MPM contact stiffness of the links in contact. Rigid's island factor of A is
    computed at the current x, and the Woodbury identity applies on it. Per environment, U holds one column per spatial
    component of each link in contact (the cdofs of its chain), W = A^-1 U, and S = I + K U^T W, of order 6 times the
    number of links in contact, gets a lower-upper (LU) factorization. Without the contact stiffness, the rows of the
    many particles on a link, far stiffer than its inertia, leave the joint preconditioned conjugate gradient (PCG)
    stopping on a direction that barely moves the link."""
    if qd.static(config.has_rigid_rows):
        func_hessian_and_cholesky_factor_direct(constraint_state, rigid_info, rigid_config, compute_envelope=True)
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_contact_factor")
    for i_b in range(n_envs):
        n_links = 0
        for i_l in range(rigid_data.link_K.shape[0]):
            if rigid_data.link_K[i_l, i_b].norm_sqr() > 0.0 and n_links < rigid_data.contact_links.shape[0]:
                rigid_data.contact_links[n_links, i_b] = i_l
                n_links += 1
        rigid_data.n_links_in_contact[i_b] = n_links
        n_cols = 6 * n_links
        for i_col in range(n_cols):
            i_l = rigid_data.contact_links[i_col // 6, i_b]
            component = i_col % 6
            for i_d in range(n_dofs):
                rigid_data.solve_rhs[i_d, i_b] = 0.0
            i_a = i_l
            while i_a >= 0:
                I_a = [i_a, i_b] if qd.static(config.rigid_batch_links_info) else i_a
                for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
                    value = gs.qd_float(0.0)
                    for e in qd.static(range(3)):
                        if component == e:
                            value = dyn_state.dofs.cdof_vel[i_d, i_b][e]
                        if component == 3 + e:
                            value = dyn_state.dofs.cdof_ang[i_d, i_b][e]
                    rigid_data.solve_rhs[i_d, i_b] = value
                i_a = dyn_info.links.parent_idx[I_a]
            rigid_solve(
                i_b,
                rigid_data.solve_rhs,
                rigid_data.solve_out,
                dyn_state,
                constraint_state,
                rigid_info,
                rigid_config,
                config,
            )
            for i_d in range(n_dofs):
                rigid_data.woodbury_U[i_d, i_col, i_b] = rigid_data.solve_rhs[i_d, i_b]
                rigid_data.woodbury_W[i_d, i_col, i_b] = rigid_data.solve_out[i_d, i_b]
        # S = I + K U^T W, the rows of a link's block from its K_l and the rows of U^T W of its columns
        for b in range(n_cols):
            for j in range(n_links):
                g = qd.Vector.zero(gs.qd_float, 6)
                for k in qd.static(range(6)):
                    value = gs.qd_float(0.0)
                    for i_d in range(n_dofs):
                        value += rigid_data.woodbury_U[i_d, 6 * j + k, i_b] * rigid_data.woodbury_W[i_d, b, i_b]
                    g[k] = value
                i_l = rigid_data.contact_links[j, i_b]
                K_g = rigid_data.link_K[i_l, i_b] @ g
                for i in qd.static(range(6)):
                    value = K_g[i]
                    if 6 * j + i == b:
                        value += 1.0
                    rigid_data.capacitance_LU[6 * j + i, b, i_b] = value
        # In-place lower-upper (LU) factorization with partial pivoting. S is non-singular, K and U^T W being positive
        # semi-definite
        for k in range(n_cols):
            i_max = k
            for i in range(k + 1, n_cols):
                if qd.abs(rigid_data.capacitance_LU[i, k, i_b]) > qd.abs(rigid_data.capacitance_LU[i_max, k, i_b]):
                    i_max = i
            rigid_data.capacitance_pivots[k, i_b] = i_max
            if i_max != k:
                for j in range(n_cols):
                    entry = rigid_data.capacitance_LU[k, j, i_b]
                    rigid_data.capacitance_LU[k, j, i_b] = rigid_data.capacitance_LU[i_max, j, i_b]
                    rigid_data.capacitance_LU[i_max, j, i_b] = entry
            for i in range(k + 1, n_cols):
                rigid_data.capacitance_LU[i, k, i_b] /= rigid_data.capacitance_LU[k, k, i_b]
                factor = rigid_data.capacitance_LU[i, k, i_b]
                for j in range(k + 1, n_cols):
                    rigid_data.capacitance_LU[i, j, i_b] -= factor * rigid_data.capacitance_LU[k, j, i_b]


@qd.func(requires_top_level=True)
def func_rigid_precondition(
    rigid_data: RigidNewtonScratch,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Apply the preconditioner of the Rigid block with the factors of func_rigid_factor.

    z = (A + U K U^T)^-1 r over the Rigid dofs: z0 = A^-1 r, then z = z0 - W S^-1 K U^T z0 by the Woodbury identity."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_precondition")
    for i_b in range(n_envs):
        rigid_solve(i_b, rigid_data.r, rigid_data.z, dyn_state, constraint_state, rigid_info, rigid_config, config)
        n_links = rigid_data.n_links_in_contact[i_b]
        n_cols = 6 * n_links
        for a in range(n_cols):
            value = gs.qd_float(0.0)
            for i_d in range(n_dofs):
                value += rigid_data.woodbury_U[i_d, a, i_b] * rigid_data.z[i_d, i_b]
            rigid_data.woodbury_y[a, i_b] = value
        for j in range(n_links):
            i_l = rigid_data.contact_links[j, i_b]
            y = qd.Vector([rigid_data.woodbury_y[6 * j + k, i_b] for k in qd.static(range(6))], dt=gs.qd_float)
            K_y = rigid_data.link_K[i_l, i_b] @ y
            for i in qd.static(range(6)):
                rigid_data.woodbury_y[6 * j + i, i_b] = K_y[i]
        # S q = K U^T z0 by the lower-upper (LU) factor: the row pivots, then the unit lower and upper triangular solves
        for k in range(n_cols):
            i_p = rigid_data.capacitance_pivots[k, i_b]
            if i_p != k:
                entry = rigid_data.woodbury_y[k, i_b]
                rigid_data.woodbury_y[k, i_b] = rigid_data.woodbury_y[i_p, i_b]
                rigid_data.woodbury_y[i_p, i_b] = entry
        for i in range(n_cols):
            value = rigid_data.woodbury_y[i, i_b]
            for k in range(i):
                value -= rigid_data.capacitance_LU[i, k, i_b] * rigid_data.woodbury_y[k, i_b]
            rigid_data.woodbury_y[i, i_b] = value
        for i_rev in range(n_cols):
            i = n_cols - 1 - i_rev
            value = rigid_data.woodbury_y[i, i_b]
            for k in range(i + 1, n_cols):
                value -= rigid_data.capacitance_LU[i, k, i_b] * rigid_data.woodbury_y[k, i_b]
            rigid_data.woodbury_y[i, i_b] = value / rigid_data.capacitance_LU[i, i, i_b]
        for i_d in range(n_dofs):
            value = gs.qd_float(0.0)
            for b in range(n_cols):
                value += rigid_data.woodbury_W[i_d, b, i_b] * rigid_data.woodbury_y[b, i_b]
            rigid_data.z[i_d, i_b] -= value


# ========================================= Layer 1: joint PCG over (u, x) ==========================================


@qd.func(requires_top_level=True)
def func_rigid_pcg_init(rigid_data: RigidNewtonScratch, pcg_scalars: qd.template(), is_adjoint: qd.template()):
    """Start the Rigid part of the joint conjugate gradient from zero, with r = grad and the solution and Ap cleared.

    With `is_adjoint`, an environment left out of the solve keeps its solution (see func_init_linear_pcg_residual)."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_init")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.r[i_d, i_b] = rigid_data.grad[i_d, i_b]
        is_kept = False
        if qd.static(is_adjoint):
            is_kept = pcg_scalars.is_converged[i_b] != 0
        if not is_kept:
            rigid_data.sol[i_d, i_b] = 0.0
        rigid_data.ap[i_d, i_b] = 0.0


@qd.func(requires_top_level=True)
def func_rigid_pcg_init_direction(rigid_data: RigidNewtonScratch, pcg_scalars: qd.template()):
    """Set the first search direction p = z over the Rigid dofs, and add r . z to rz."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_init_direction")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.p[i_d, i_b] = rigid_data.z[i_d, i_b]
        qd.atomic_add(pcg_scalars.rz[i_b], rigid_data.r[i_d, i_b] * rigid_data.z[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_matvec(
    rigid_data: RigidNewtonScratch,
    constraint_state: ConstraintState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Set Ap_r = (M + J^T D J) p_r over the active rows of Rigid."""
    n_envs = rigid_data.x.shape[1]
    initialize_Ma(rigid_data.ap, rigid_data.p, rigid_info, rigid_config)
    if qd.static(config.has_rigid_rows):
        qd.loop_config(name="mpm_rigid_matvec_rows")
        for i_b in range(n_envs):
            for i_c in range(constraint_state.n_constraints[i_b]):
                if constraint_state.active[i_c, i_b]:
                    jac_p = gs.qd_float(0.0)
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        jac_p += constraint_state.jac[i_c, i_d, i_b] * rigid_data.p[i_d, i_b]
                    D_jac_p = constraint_state.efc_D[i_c, i_b] * jac_p
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        rigid_data.ap[i_d, i_b] += D_jac_p * constraint_state.jac[i_c, i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_matvec_dot(rigid_data: RigidNewtonScratch, pcg_scalars: qd.template()):
    """Add p_r . Ap_r to p . Ap, once Ap_r holds the whole joint operator."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_matvec_dot")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[i_b], rigid_data.p[i_d, i_b] * rigid_data.ap[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_pcg_update_xr(rigid_data: RigidNewtonScratch, pcg_scalars: qd.template()):
    """Update the solution and the residual over the Rigid dofs, x += alpha p and r -= alpha Ap."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_update_xr")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if pcg_scalars.is_converged[i_b] == 0:
            alpha = pcg_scalars.alpha[i_b]
            rigid_data.sol[i_d, i_b] += alpha * rigid_data.p[i_d, i_b]
            rigid_data.r[i_d, i_b] -= alpha * rigid_data.ap[i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_pcg_dot_rz(rigid_data: RigidNewtonScratch, pcg_scalars: qd.template()):
    """Add r . z over the Rigid dofs to rz_new."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_dot_rz")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.rz_new[i_b], rigid_data.r[i_d, i_b] * rigid_data.z[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_pcg_direction(rigid_data: RigidNewtonScratch, pcg_scalars: qd.template()):
    """Update the search direction p = z + beta p over the Rigid dofs, and clear Ap."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_direction")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.ap[i_d, i_b] = 0.0
        if pcg_scalars.is_converged[i_b] == 0:
            rigid_data.p[i_d, i_b] = rigid_data.z[i_d, i_b] + pcg_scalars.beta[i_b] * rigid_data.p[i_d, i_b]


# ============================================== Layer 1: Newton steps ==============================================


@qd.func(requires_top_level=True)
def func_rigid_start_line_search(rigid_data: RigidNewtonScratch, control: qd.template()):
    """Set the line-search direction dx = -solution and its start point, and take |dx| into the largest displacement of
    the Newton step, after the MPM rows."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_start_line_search")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.dx[i_d, i_b] = -rigid_data.sol[i_d, i_b]
        rigid_data.x_temp[i_d, i_b] = rigid_data.x[i_d, i_b]
        qd.atomic_max(control.max_disp[i_b], qd.abs(rigid_data.sol[i_d, i_b]))


@qd.func(requires_top_level=True)
def func_rigid_step_forward(rigid_data: RigidNewtonScratch, control: qd.template()):
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_step_forward")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.x[i_d, i_b] = rigid_data.x_temp[i_d, i_b] + control.alpha[i_b] * rigid_data.dx[i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_write_back(rigid_data: RigidNewtonScratch, constraint_state: ConstraintState, config: qd.template()):
    """Hand the converged qacc to Rigid as the solution of its constraint solve.

    Runs after func_rigid_evaluate at the converged x, which left the row activity, forces and qfrc_constraint of Rigid
    consistent with it."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_write_back")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.qacc[i_d, i_b] = rigid_data.qacc[i_d, i_b]
        if qd.static(not config.has_rigid_rows):
            constraint_state.qfrc_constraint[i_d, i_b] = 0.0


@qd.func(requires_top_level=True)
def func_rigid_adjoint_rhs(
    rigid_data: RigidNewtonScratch, constraint_state: ConstraintState, newton_info: RigidNewtonInfo
):
    """Set Rigid's right-hand side of the joint adjoint solve, in place of the gradient from which the solve starts.

    The right-hand side is the adjoint of x = dt^2 qacc, dL/dqacc / dt^2, with dL/dqacc as taken by Rigid's own adjoint
    solve (see ConstraintSolver.backward)."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    inv_dt_sq = 1.0 / (newton_info.substep_dt[()] * newton_info.substep_dt[()])
    qd.loop_config(name="mpm_rigid_adjoint_rhs")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.grad[i_d, i_b] = constraint_state.dL_dqacc[i_d, i_b] * inv_dt_sq


@qd.func(requires_top_level=True)
def func_rigid_adjoint_finish(
    rigid_data: RigidNewtonScratch, constraint_state: ConstraintState, newton_info: RigidNewtonInfo
):
    """Set the solution of Rigid's adjoint solve from that of the joint one.

    Rigid's share of the joint energy is dt^4 times its cost of x / dt^2, whose gradient in the parameters of Rigid's
    rows is dt^2 times that of its cost, so u = dt^2 lambda_x."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    dt_sq = newton_info.substep_dt[()] * newton_info.substep_dt[()]
    qd.loop_config(name="mpm_rigid_adjoint_finish")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.bw_u[i_d, i_b] = dt_sq * rigid_data.sol[i_d, i_b]
