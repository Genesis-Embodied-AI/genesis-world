"""Stages of Rigid's dofs as a participant of the NewtonSolver, in the joint solve with the solvers coupled with
Rigid."""

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.linear_pcg import PCGScalars
from genesis.engine.numerics.newton_control import NewtonControl
from genesis.utils.array_class import ConstraintState, DofsState, DynInfo, DynState, LinksInfo, RigidInfo
from genesis.utils.data import ENVS, Data, DataKind, Size, field

from .abd.forward_dynamics import func_solve_mass_batch
from .constraint.island import func_group_constraints_by_island
from .constraint.solver import (
    func_cholesky_solve_batch,
    func_update_constraint,
    initialize_Jaref,
    initialize_Ma,
)


class RigidNewtonInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """Scene values of Rigid's stages in the joint solve, as device scalars."""

    substep_dt: qd.Tensor = field(float, ndarray=True)


# The dofs and the links of Rigid in the joint solve
DOFS = Size("dofs")
LINKS = Size("links")


class RigidNewtonScratch(Data, kind=DataKind.SCRATCH):
    """Rigid unknowns of the joint Newton solve and the Rigid part of its conjugate gradient vectors."""

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
    links_motion_vel: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    links_motion_ang: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    links_force_vel: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    links_force_ang: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    # The 6 x 6 spatial stiffness K_l the couplings put on every link, which the preconditioner adds to Rigid's
    # Hessian (see func_add_links_stiffness_island_batch)
    links_stiffness: qd.Tensor = field((float, (6, 6)), shape=(LINKS, ENVS), ndarray=True)


# ============================================ Layer 1: link kinematics =============================================


@qd.func(requires_top_level=True)
def func_link_spatial_disp(
    src: qd.template(),
    rigid_data: RigidNewtonScratch,
    dofs_state: DofsState,
    links_info: LinksInfo,
    config: qd.template(),
):
    """Set the spatial displacement of every link from a dof vector, and clear the spatial force of the link."""
    n_links = rigid_data.links_motion_vel.shape[0]
    n_envs = rigid_data.links_motion_vel.shape[1]
    qd.loop_config(name="mpm_link_spatial_disp")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        motion_vel = qd.Vector.zero(gs.qd_float, 3)
        motion_ang = qd.Vector.zero(gs.qd_float, 3)
        j_l = i_l
        while j_l >= 0:
            J_l = [j_l, i_b] if qd.static(config.rigid_batch_links_info) else j_l
            for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                motion_vel += dofs_state.cdof_vel[i_d, i_b] * src[i_d, i_b]
                motion_ang += dofs_state.cdof_ang[i_d, i_b] * src[i_d, i_b]
            j_l = links_info.parent_idx[J_l]
        rigid_data.links_motion_vel[i_l, i_b] = motion_vel
        rigid_data.links_motion_ang[i_l, i_b] = motion_ang
        rigid_data.links_force_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        rigid_data.links_force_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_link_force_to_dofs(
    scale: float,
    out: qd.template(),
    rigid_data: RigidNewtonScratch,
    dofs_state: DofsState,
    links_info: LinksInfo,
    config: qd.template(),
):
    """Add the generalized force of the spatial forces accumulated on the links to a dof vector."""
    n_links = rigid_data.links_force_vel.shape[0]
    n_envs = rigid_data.links_force_vel.shape[1]
    qd.loop_config(name="mpm_link_force_to_dofs")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        force_vel = rigid_data.links_force_vel[i_l, i_b]
        force_ang = rigid_data.links_force_ang[i_l, i_b]
        if force_vel.norm_sqr() + force_ang.norm_sqr() > 0.0:
            j_l = i_l
            while j_l >= 0:
                J_l = [j_l, i_b] if qd.static(config.rigid_batch_links_info) else j_l
                for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                    value = dofs_state.cdof_vel[i_d, i_b].dot(force_vel) + dofs_state.cdof_ang[i_d, i_b].dot(force_ang)
                    qd.atomic_add(out[i_d, i_b], scale * value)
                j_l = links_info.parent_idx[J_l]


@qd.func(requires_top_level=True)
def func_clear_links_stiffness(rigid_data: RigidNewtonScratch):
    """Clear the contact stiffness of every link, before the contacts add theirs to the preconditioner."""
    qd.loop_config(name="mpm_rigid_links_stiffness_clear")
    for i_l, i_b in qd.ndrange(rigid_data.links_stiffness.shape[0], rigid_data.links_stiffness.shape[1]):
        rigid_data.links_stiffness[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)


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
    """Start the Rigid unknown from Rigid's warm start and group Rigid's assembled rows by island."""
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
    qfrc_constraint, by Rigid's own update of its constraint state."""
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
        energies[i_b] = energies[i_b] + dt_sq * dt_sq * rigid_data.cost[i_b]


@qd.func(requires_top_level=True)
def func_rigid_gradient(
    rigid_data: RigidNewtonScratch,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: RigidNewtonInfo,
    config: qd.template(),
):
    """Compute the gradient of Rigid's share of the joint energy in x, after func_rigid_evaluate at the current x."""
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
def func_rigid_solve_batch(
    i_b: int,
    rhs: qd.template(),
    out: qd.template(),
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Solve the Rigid block system of one environment by Rigid's own factors."""
    if qd.static(config.has_rigid_rows):
        for i_island in range(constraint_state.island.n_islands[i_b]):
            func_cholesky_solve_batch(
                i_b, i_island, rhs=rhs, out=out, constraint_state=constraint_state, rigid_config=rigid_config
            )
    else:
        func_solve_mass_batch(i_b, rhs, out, dyn_state, rigid_info, rigid_config)


@qd.func
def cdof_spatial(i_d: int, i_b: int, dofs_state: DofsState):
    """Spatial motion (linear, angular) of a link per unit velocity of dof `i_d`."""
    v = dofs_state.cdof_vel[i_d, i_b]
    w = dofs_state.cdof_ang[i_d, i_b]
    return qd.Vector([v[0], v[1], v[2], w[0], w[1], w[2]], dt=gs.qd_float)


@qd.func
def func_add_links_stiffness_island_batch(
    i_b: int,
    i_island: int,
    rigid_data: RigidNewtonScratch,
    dofs_state: DofsState,
    links_info: LinksInfo,
    constraint_state: ConstraintState,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Add the contact stiffness J_l^T K_l J_l of every link of an island to the lower triangle of its Hessian block,
    J_l being the cdofs of the chain of link l and K_l its spatial stiffness (see links_stiffness)."""
    for i_l in range(rigid_data.links_stiffness.shape[0]):
        stiffness = rigid_data.links_stiffness[i_l, i_b]
        if stiffness.norm_sqr() > 0.0:
            i_l1 = i_l
            while i_l1 >= 0:
                I_l1 = [i_l1, i_b] if qd.static(config.rigid_batch_links_info) else i_l1
                for i_d1 in range(links_info.dof_start[I_l1], links_info.dof_end[I_l1]):
                    if constraint_state.island.dofs_island_idx[i_d1, i_b] != i_island:
                        continue
                    stiffness_cdof1 = stiffness @ cdof_spatial(i_d1, i_b, dofs_state)
                    pos1 = constraint_state.island.dof_local_pos[i_d1, i_b]
                    i_l2 = i_l
                    while i_l2 >= 0:
                        I_l2 = [i_l2, i_b] if qd.static(config.rigid_batch_links_info) else i_l2
                        for i_d2 in range(links_info.dof_start[I_l2], links_info.dof_end[I_l2]):
                            # The lower triangle in the island-local order of the dofs
                            if pos1 >= constraint_state.island.dof_local_pos[i_d2, i_b]:
                                value = cdof_spatial(i_d2, i_b, dofs_state).dot(stiffness_cdof1)
                                if qd.static(rigid_config.enable_jacobi_equilibration):
                                    value *= (
                                        constraint_state.nt_jacobi[i_d1, i_b] * constraint_state.nt_jacobi[i_d2, i_b]
                                    )
                                # The thread of the island alone writes its block
                                constraint_state.nt_H[i_b, i_d1, i_d2] = constraint_state.nt_H[i_b, i_d1, i_d2] + value
                        i_l2 = links_info.parent_idx[I_l2]
                i_l1 = links_info.parent_idx[I_l1]


@qd.func(requires_top_level=True)
def func_rigid_precondition(
    rigid_data: RigidNewtonScratch,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """z = A^-1 r over the Rigid dofs, A being Rigid's Hessian plus the contact stiffness of the links in its factor."""
    qd.loop_config(name="mpm_rigid_precondition")
    for i_b in range(rigid_data.x.shape[1]):
        func_rigid_solve_batch(
            i_b, rigid_data.r, rigid_data.z, dyn_state, constraint_state, rigid_info, rigid_config, config
        )


# ========================================= Layer 1: joint PCG over (u, x) ==========================================


@qd.func(requires_top_level=True)
def func_rigid_pcg_init(rigid_data: RigidNewtonScratch, pcg_scalars: PCGScalars, is_adjoint: qd.template()):
    """Start the Rigid part of the joint conjugate gradient from zero, with r = grad and the solution and Ap cleared."""
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
def func_rigid_pcg_init_direction(rigid_data: RigidNewtonScratch, pcg_scalars: PCGScalars):
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
                        rigid_data.ap[i_d, i_b] = (
                            rigid_data.ap[i_d, i_b] + D_jac_p * constraint_state.jac[i_c, i_d, i_b]
                        )


@qd.func(requires_top_level=True)
def func_rigid_matvec_dot(rigid_data: RigidNewtonScratch, pcg_scalars: PCGScalars):
    """Add p_r . Ap_r to p . Ap, once Ap_r holds the whole joint operator."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_matvec_dot")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[i_b], rigid_data.p[i_d, i_b] * rigid_data.ap[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_pcg_update_xr(rigid_data: RigidNewtonScratch, pcg_scalars: PCGScalars):
    """Update the solution and the residual over the Rigid dofs, x += alpha p and r -= alpha Ap."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_update_xr")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if pcg_scalars.is_converged[i_b] == 0:
            alpha = pcg_scalars.alpha[i_b]
            rigid_data.sol[i_d, i_b] = rigid_data.sol[i_d, i_b] + alpha * rigid_data.p[i_d, i_b]
            rigid_data.r[i_d, i_b] = rigid_data.r[i_d, i_b] - alpha * rigid_data.ap[i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_pcg_dot_rz(rigid_data: RigidNewtonScratch, pcg_scalars: PCGScalars):
    """Add r . z over the Rigid dofs to rz_new."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_dot_rz")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.rz_new[i_b], rigid_data.r[i_d, i_b] * rigid_data.z[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_pcg_direction(rigid_data: RigidNewtonScratch, pcg_scalars: PCGScalars):
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
def func_rigid_start_line_search(rigid_data: RigidNewtonScratch, control: NewtonControl):
    """Set the line-search direction dx = -solution and its start point, and take |dx| into the largest displacement of
    the Newton step, after the MPM rows."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_start_line_search")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.dx[i_d, i_b] = -rigid_data.sol[i_d, i_b]
        rigid_data.x_temp[i_d, i_b] = rigid_data.x[i_d, i_b]
        qd.atomic_max(control.max_displacement[i_b], qd.abs(rigid_data.sol[i_d, i_b]))


@qd.func(requires_top_level=True)
def func_rigid_step_forward(rigid_data: RigidNewtonScratch, control: NewtonControl):
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_step_forward")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.x[i_d, i_b] = rigid_data.x_temp[i_d, i_b] + control.alpha[i_b] * rigid_data.dx[i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_write_back(rigid_data: RigidNewtonScratch, constraint_state: ConstraintState, config: qd.template()):
    """Hand the converged qacc to Rigid as the solution of its constraint solve."""
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
    """Set Rigid's right-hand side of the joint adjoint solve, in place of the gradient from which the solve starts."""
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
    """Set the solution of Rigid's adjoint solve from that of the joint one."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    dt_sq = newton_info.substep_dt[()] * newton_info.substep_dt[()]
    qd.loop_config(name="mpm_rigid_adjoint_finish")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.bw_u[i_d, i_b] = dt_sq * rigid_data.sol[i_d, i_b]
