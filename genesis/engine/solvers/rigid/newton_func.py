"""Stages of Rigid's dofs as a participant of the NewtonSolver, in the joint solve with the solvers coupled with
Rigid."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.numerics.newton_control import (
    NewtonControl,
    line_search_point,
    newton_direction,
    step_measure,
)
from genesis.utils.array_class import ConstraintState, DofsState, DynInfo, DynState, LinksInfo, RigidInfo
from genesis.utils.data import ENVS, Data, DataKind, Size, field

from .abd.forward_dynamics import func_solve_mass_batch
from .constraint.backward import func_matvec_Ap
from .constraint.island import func_group_constraints_by_island
from .constraint.solver import (
    constraint_gradient,
    func_cholesky_solve_islands_batch,
    func_update_constraint,
    func_warmstart_qacc,
    gauss_cost_term,
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
    x_start: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    dx: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    grad: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    qacc: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    Ma: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    r: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    z: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    p: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    ap: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    solution: qd.Tensor = field(float, shape=(DOFS, ENVS), ndarray=True)
    cost: qd.Tensor = field(float, shape=(ENVS,), ndarray=True)
    links_motion_vel: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    links_motion_ang: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    links_force_vel: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    links_force_ang: qd.Tensor = field((float, 3), shape=(LINKS, ENVS), ndarray=True)
    # The 6 x 6 spatial stiffness K_l the couplings put on every link, which the preconditioner adds to Rigid's
    # Hessian (see func_add_links_stiffness_island_batch)
    links_stiffness: qd.Tensor = field((float, (6, 6)), shape=(LINKS, ENVS), ndarray=True)


# =============================================== Layer 0: Rigid terms ===============================================


@qd.func
def has_spatial_force(force_vel: qd.types.vector(3), force_ang: qd.types.vector(3)):
    return force_vel.norm_sqr() + force_ang.norm_sqr() > 0.0


@qd.func
def to_displacement(qacc: float, dt: float):
    """Rigid unknown x = dt^2 qacc of a dof, in the units of a displacement like the other participants'."""
    return dt * dt * qacc


@qd.func
def to_acceleration(x: float, dt: float):
    """Acceleration qacc = x / dt^2 of a dof from its Rigid unknown (see to_displacement)."""
    return x * (1.0 / (dt * dt))


@qd.func
def constraint_force(Ma: float, qf_smooth: float):
    """Constraint force on a dof at a solved acceleration, M a - f_smooth."""
    return Ma - qf_smooth


@qd.func
def rigid_energy(cost: float, dt: float):
    """Share of Rigid's cost in the joint energy, scaled by dt^4 to the units of the other participants'."""
    return (dt * dt) * (dt * dt) * cost


@qd.func
def rigid_gradient(Ma: float, qf_smooth: float, qfrc_constraint: float, dt: float):
    """Gradient of Rigid's share of the joint energy in the unknown x of a dof."""
    return dt * dt * constraint_gradient(Ma, qf_smooth, qfrc_constraint)


@qd.func
def spatial_stiffness_column(stiffness: qd.types.matrix(6, 6), cdof: qd.types.vector(6)):
    return stiffness @ cdof


@qd.func
def dof_pair_stiffness(
    cdof: qd.types.vector(6),
    stiffness_cdof: qd.types.vector(6),
    jacobi_a: float,
    jacobi_b: float,
    is_equilibrated: qd.template(),
):
    """Entry J_a^T K J_b of the contact stiffness of a link between two dofs of its chain, equilibrated by the Jacobi
    scales of Rigid's Hessian with `is_equilibrated`."""
    value = cdof.dot(stiffness_cdof)
    if qd.static(is_equilibrated):
        value *= jacobi_a * jacobi_b
    return value


# ============================================ Layer 1: link kinematics =============================================


@qd.func
def link_chain_motion(
    i_l: int,
    i_b: int,
    src: qd.template(),
    dofs_state: DofsState,
    links_info: LinksInfo,
    config: qd.template(),
):
    """Spatial motion (linear, angular) of a link from a dof vector, summed over the dofs of its chain."""
    motion_vel = qd.Vector.zero(gs.qd_float, 3)
    motion_ang = qd.Vector.zero(gs.qd_float, 3)
    j_l = i_l
    while j_l >= 0:
        J_l = [j_l, i_b] if qd.static(config.rigid_batch_links_info) else j_l
        for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
            motion_vel_d, motion_ang_d = gu.dof_motion(
                dofs_state.cdof_vel[i_d, i_b], dofs_state.cdof_ang[i_d, i_b], src[i_d, i_b]
            )
            motion_vel += motion_vel_d
            motion_ang += motion_ang_d
        j_l = links_info.parent_idx[J_l]
    return motion_vel, motion_ang


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
        rigid_data.links_motion_vel[i_l, i_b], rigid_data.links_motion_ang[i_l, i_b] = link_chain_motion(
            i_l, i_b, src, dofs_state, links_info, config
        )
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
        if has_spatial_force(force_vel, force_ang):
            j_l = i_l
            while j_l >= 0:
                J_l = [j_l, i_b] if qd.static(config.rigid_batch_links_info) else j_l
                for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                    qd.atomic_add(
                        out[i_d, i_b],
                        scale
                        * gu.dof_force(
                            dofs_state.cdof_vel[i_d, i_b], dofs_state.cdof_ang[i_d, i_b], force_vel, force_ang
                        ),
                    )
                j_l = links_info.parent_idx[J_l]


@qd.func(requires_top_level=True)
def func_clear_links_stiffness(rigid_data: RigidNewtonScratch):
    """Clear the contact stiffness of every link, before the contacts add theirs to the preconditioner."""
    qd.loop_config(name="mpm_rigid_links_stiffness_clear")
    for i_l, i_b in qd.ndrange(rigid_data.links_stiffness.shape[0], rigid_data.links_stiffness.shape[1]):
        rigid_data.links_stiffness[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)


# ============================================== Layer 1: Rigid block ===============================================


@qd.kernel
def kernel_rigid_begin(
    rigid_data: RigidNewtonScratch,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    newton_info: RigidNewtonInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Start the Rigid unknown from Rigid's warm start and group Rigid's assembled rows by island."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_begin")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        qacc = dyn_state.dofs.acc_smooth[i_d, i_b]
        if qd.static(config.has_rigid_rows):
            qacc = func_warmstart_qacc(i_d, i_b, dyn_state, constraint_state)
        rigid_data.x[i_d, i_b] = to_displacement(qacc, newton_info.substep_dt[()])
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
    """Evaluate Rigid's cost of qacc = x / dt^2, with its row activity, forces and qfrc_constraint, by Rigid's own
    update of its constraint state."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_qacc")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.qacc[i_d, i_b] = to_acceleration(rigid_data.x[i_d, i_b], newton_info.substep_dt[()])
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
                cost += gauss_cost_term(
                    rigid_data.Ma[i_d, i_b],
                    dyn_state.dofs.qf_smooth[i_d, i_b],
                    rigid_data.qacc[i_d, i_b],
                    dyn_state.dofs.acc_smooth[i_d, i_b],
                )
            rigid_data.cost[i_b] = cost


@qd.func(requires_top_level=True)
def func_rigid_energy(energies: qd.template(), rigid_data: RigidNewtonScratch, newton_info: RigidNewtonInfo):
    """Add Rigid's cost of every environment times dt^4 into the entry of the environment, after func_rigid_evaluate."""
    for i_b in range(rigid_data.x.shape[1]):
        energies[i_b] = energies[i_b] + rigid_energy(rigid_data.cost[i_b], newton_info.substep_dt[()])


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
    qd.loop_config(name="mpm_rigid_gradient")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        qfrc_constraint = gs.qd_float(0.0)
        if qd.static(config.has_rigid_rows):
            qfrc_constraint = constraint_state.qfrc_constraint[i_d, i_b]
        rigid_data.grad[i_d, i_b] = rigid_gradient(
            rigid_data.Ma[i_d, i_b], dyn_state.dofs.qf_smooth[i_d, i_b], qfrc_constraint, newton_info.substep_dt[()]
        )


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
    """Solve the Rigid block system of one environment by Rigid's own factors, every island taking part (see
    kernel_rigid_begin)."""
    if qd.static(config.has_rigid_rows):
        func_cholesky_solve_islands_batch(
            i_b, rhs=rhs, out=out, constraint_state=constraint_state, rigid_config=rigid_config
        )
    else:
        func_solve_mass_batch(i_b, rhs, out, dyn_state, rigid_info, rigid_config)


@qd.func
def cdof_spatial(i_d: int, i_b: int, dofs_state: DofsState):
    """Spatial motion (linear, angular) of a link per unit velocity of a dof."""
    vel = dofs_state.cdof_vel[i_d, i_b]
    ang = dofs_state.cdof_ang[i_d, i_b]
    return qd.Vector([vel[0], vel[1], vel[2], ang[0], ang[1], ang[2]], dt=gs.qd_float)


@qd.func
def func_add_links_stiffness_island_batch(
    i_b: int,
    i_island: int,
    rigid_data: RigidNewtonScratch,
    dofs_state: DofsState,
    constraint_state: ConstraintState,
    links_info: LinksInfo,
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
                    stiffness_cdof1 = spatial_stiffness_column(stiffness, cdof_spatial(i_d1, i_b, dofs_state))
                    pos1 = constraint_state.island.dof_local_pos[i_d1, i_b]
                    i_l2 = i_l
                    while i_l2 >= 0:
                        I_l2 = [i_l2, i_b] if qd.static(config.rigid_batch_links_info) else i_l2
                        for i_d2 in range(links_info.dof_start[I_l2], links_info.dof_end[I_l2]):
                            # The lower triangle in the island-local order of the dofs
                            if pos1 >= constraint_state.island.dof_local_pos[i_d2, i_b]:
                                value = dof_pair_stiffness(
                                    cdof_spatial(i_d2, i_b, dofs_state),
                                    stiffness_cdof1,
                                    constraint_state.nt_jacobi[i_d1, i_b],
                                    constraint_state.nt_jacobi[i_d2, i_b],
                                    rigid_config.enable_jacobi_equilibration,
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
    """Precondition the residual over the Rigid dofs, z = A^-1 r, A being Rigid's Hessian plus the contact stiffness of
    the links in its factor."""
    qd.loop_config(name="mpm_rigid_precondition")
    for i_b in range(rigid_data.x.shape[1]):
        func_rigid_solve_batch(
            i_b, rigid_data.r, rigid_data.z, dyn_state, constraint_state, rigid_info, rigid_config, config
        )


# ========================================= Layer 1: joint PCG over (u, x) ==========================================


@qd.func(requires_top_level=True)
def func_rigid_matvec(
    rigid_data: RigidNewtonScratch,
    constraint_state: ConstraintState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Set Ap_r = (M + J^T D J) p_r over the active rows of Rigid, the product of Rigid's adjoint solve."""
    if qd.static(config.has_rigid_rows):
        qd.loop_config(name="mpm_rigid_matvec")
        for i_b in range(rigid_data.x.shape[1]):
            func_matvec_Ap(
                i_b,
                p=rigid_data.p,
                Ap=rigid_data.ap,
                constraint_state=constraint_state,
                dyn_info=dyn_info,
                rigid_info=rigid_info,
            )
    else:
        initialize_Ma(rigid_data.ap, rigid_data.p, rigid_info, rigid_config)


# ============================================== Layer 1: Newton steps ==============================================


@qd.func(requires_top_level=True)
def func_rigid_start_line_search(rigid_data: RigidNewtonScratch, control: NewtonControl):
    """Set the line-search direction dx = -solution and its start point, and take |dx| into the largest displacement of
    the Newton step, after the MPM rows."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_start_line_search")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.dx[i_d, i_b] = newton_direction(rigid_data.solution[i_d, i_b])
        rigid_data.x_start[i_d, i_b] = rigid_data.x[i_d, i_b]
        qd.atomic_max(control.max_displacement[i_b], step_measure(rigid_data.solution[i_d, i_b]))


@qd.func(requires_top_level=True)
def func_rigid_step_forward(rigid_data: RigidNewtonScratch, control: NewtonControl):
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_step_forward")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.x[i_d, i_b] = line_search_point(
            rigid_data.x_start[i_d, i_b], control.alpha[i_b], rigid_data.dx[i_d, i_b]
        )


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


@qd.kernel
def kernel_rigid_adjoint_rhs(
    rigid_data: RigidNewtonScratch, constraint_state: ConstraintState, newton_info: RigidNewtonInfo
):
    """Set Rigid's right-hand side of the joint adjoint solve, in place of the gradient from which the solve starts."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_adjoint_rhs")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        # The adjoint of qacc = x / dt^2 maps the adjoint of qacc to that of x as qacc maps from x
        rigid_data.grad[i_d, i_b] = to_acceleration(constraint_state.dL_dqacc[i_d, i_b], newton_info.substep_dt[()])


@qd.kernel
def kernel_rigid_adjoint_finish(
    rigid_data: RigidNewtonScratch, constraint_state: ConstraintState, newton_info: RigidNewtonInfo
):
    """Set the solution of Rigid's adjoint solve from that of the joint one."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_adjoint_finish")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.bw_u[i_d, i_b] = to_displacement(rigid_data.solution[i_d, i_b], newton_info.substep_dt[()])
