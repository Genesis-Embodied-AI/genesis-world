"""
The rigid solver in the global Newton solve of the implicit material point method (MPM) solver (see
'RigidSolver.is_solved_in_global_newton').

The unknowns of the dofs of the rigid solver join the Newton solve of the substep, which evaluates the cost of its
constraint solve, its gradient, its Hessian and the factor of the Hessian through the functions of its own constraint
solver. The links of the rigid solver take the forces and the stiffness of the contacts of the particles, which the
kinematic maps of the chains of the links carry to the dofs.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_solve_mass_batch
from genesis.engine.solvers.rigid.constraint.data import ConstraintState
from genesis.engine.solvers.rigid.constraint.island import func_group_constraints_by_island
from genesis.engine.solvers.rigid.constraint.solver import (
    func_cholesky_solve_batch,
    func_hessian_and_cholesky_factor_direct,
    func_update_constraint,
    initialize_Jaref,
    initialize_Ma,
)
from genesis.engine.solvers.rigid.data import DOFS, LINKS, DofsState, DynState, LinksInfo, RigidInfo
from genesis.utils.data import ENVS, Data, DataKind, field


class NewtonRigidDofs(Data, kind=DataKind.SCRATCH, shape=(DOFS, ENVS)):
    """
    The unknowns of the dofs of the rigid solver in the Newton solve, x = dt^2 qacc, in the units of the displacement of
    the nodes, with the vectors of their solve.

    `qacc` is the acceleration of x and `Ma` its product with the mass matrix, at which the rigid solver updates its
    constraint state (see 'func_rigid_update_constraint'). The other vectors are those of the rows (see 'NewtonRows' in
    numerics/newton_rows.py): the gradient, the solution, the residual, the preconditioned residual, the search direction and its
    product with the Hessian of the preconditioned conjugate gradient (PCG), and the start point and the direction of
    the line search.
    """

    x: qd.Tensor = field(float)
    x_start: qd.Tensor = field(float)
    dx: qd.Tensor = field(float)
    qacc: qd.Tensor = field(float)
    Ma: qd.Tensor = field(float)
    gradient: qd.Tensor = field(float)
    solution: qd.Tensor = field(float)
    r: qd.Tensor = field(float)
    z: qd.Tensor = field(float)
    p: qd.Tensor = field(float)
    Ap: qd.Tensor = field(float)


class NewtonRigidLinks(Data, kind=DataKind.SCRATCH, shape=(LINKS, ENVS)):
    """
    The links of the rigid solver in the Newton solve.

    `motion_vel` and `motion_ang` are the spatial displacement of a link from a vector over the dofs, linear and angular
    about the root center of mass of the link (see 'func_link_spatial_disp'), and `force_vel` and `force_ang` the
    spatial force the contacts with the particles put on it, which 'func_link_force_to_dofs' maps to the dofs.
    `stiffness` is the 6 x 6 spatial stiffness of these contacts, which the factor of the rigid solver adds to its
    Hessian (see RigidContactStiffness).
    """

    motion_vel: qd.Tensor = field((float, 3))
    motion_ang: qd.Tensor = field((float, 3))
    force_vel: qd.Tensor = field((float, 3))
    force_ang: qd.Tensor = field((float, 3))
    stiffness: qd.Tensor = field((float, (6, 6)))


@qd.data_oriented
class RigidContactStiffness:
    """
    The extra Hessian of the factor of the rigid solver: the stiffness of the contacts of the particles with the links.

    The extra Hessian is J_l^T K_l J_l summed over the links l, J_l mapping the dofs of the chain of a link to its
    spatial displacement and K_l being its spatial stiffness (see NewtonRigidLinks). The factor of the rigid solver adds
    it to its own Hessian (see 'func_hessian_and_cholesky_factor_direct').
    """

    has_extra_hessian = True

    def __init__(self, rigid_links: NewtonRigidLinks, dofs_state: DofsState, links_info: LinksInfo) -> None:
        self.rigid_links = rigid_links
        self.dofs_state = dofs_state
        self.links_info = links_info

    @qd.func
    def func_add_island_hessian_batch(
        self, i_b: int, i_island: int, constraint_state: ConstraintState, rigid_config: qd.template()
    ):
        for i_l in range(self.rigid_links.stiffness.shape[0]):
            stiffness = self.rigid_links.stiffness[i_l, i_b]
            if stiffness.norm_sqr() > 0.0:
                i_l1 = i_l
                while i_l1 >= 0:
                    I_l1 = [i_l1, i_b] if qd.static(rigid_config.batch_links_info) else i_l1
                    for i_d1 in range(self.links_info.dof_start[I_l1], self.links_info.dof_end[I_l1]):
                        if constraint_state.island.dofs_island_idx[i_d1, i_b] == i_island:
                            stiffness_cdof1 = stiffness @ func_spatial_vector(
                                self.dofs_state.cdof_vel[i_d1, i_b], self.dofs_state.cdof_ang[i_d1, i_b]
                            )
                            pos1 = constraint_state.island.dof_local_pos[i_d1, i_b]
                            i_l2 = i_l
                            while i_l2 >= 0:
                                I_l2 = [i_l2, i_b] if qd.static(rigid_config.batch_links_info) else i_l2
                                for i_d2 in range(self.links_info.dof_start[I_l2], self.links_info.dof_end[I_l2]):
                                    # The lower triangle of the block, in the island-local order of the dofs
                                    if pos1 >= constraint_state.island.dof_local_pos[i_d2, i_b]:
                                        value = stiffness_cdof1.dot(
                                            func_spatial_vector(
                                                self.dofs_state.cdof_vel[i_d2, i_b], self.dofs_state.cdof_ang[i_d2, i_b]
                                            )
                                        )
                                        if qd.static(rigid_config.enable_jacobi_equilibration):
                                            value = (
                                                value
                                                * constraint_state.nt_jacobi[i_d1, i_b]
                                                * constraint_state.nt_jacobi[i_d2, i_b]
                                            )
                                        # The thread of the island alone writes its block
                                        constraint_state.nt_H[i_b, i_d1, i_d2] = (
                                            constraint_state.nt_H[i_b, i_d1, i_d2] + value
                                        )
                                i_l2 = self.links_info.parent_idx[I_l2]
                    i_l1 = self.links_info.parent_idx[I_l1]


# ------------------------------------------------------------------------------------
# ------------------------------------ kinematics ------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_spatial_vector(lin: qd.types.vector(3), ang: qd.types.vector(3)):
    """Return the 6-vector of a spatial quantity, its linear part first."""
    return qd.Vector([lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]], dt=gs.qd_float)


@qd.func
def func_point_motion(motion_vel: qd.types.vector(3), motion_ang: qd.types.vector(3), r: qd.types.vector(3)):
    """Return the displacement of the point of a link at the arm r from its root center of mass, from the spatial
    displacement of the link."""
    return motion_vel + motion_ang.cross(r)


@qd.func
def func_point_stiffness(hessian: qd.types.matrix(3, 3), r: qd.types.vector(3)):
    """Return the spatial stiffness P^T H P of a link from the stiffness H of its point at the arm r from its root
    center of mass, P = [I, -[r]x] mapping the spatial displacement of the link to the displacement of the point."""
    P = qd.Matrix(
        [
            [1.0, 0.0, 0.0, 0.0, r[2], -r[1]],
            [0.0, 1.0, 0.0, -r[2], 0.0, r[0]],
            [0.0, 0.0, 1.0, r[1], -r[0], 0.0],
        ],
        dt=gs.qd_float,
    )
    return P.transpose() @ hessian @ P


@qd.func(requires_top_level=True)
def func_link_spatial_disp(
    src: qd.Tensor,
    rigid_links: NewtonRigidLinks,
    dofs_state: DofsState,
    links_info: LinksInfo,
    rigid_config: qd.template(),
):
    """Set the spatial displacement of every link from a vector over the dofs, summed over the dofs of its chain."""
    qd.loop_config(name="newton_link_spatial_disp")
    for i_l, i_b in qd.ndrange(rigid_links.motion_vel.shape[0], rigid_links.motion_vel.shape[1]):
        motion_vel = qd.Vector.zero(gs.qd_float, 3)
        motion_ang = qd.Vector.zero(gs.qd_float, 3)
        j_l = i_l
        while j_l >= 0:
            J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
            for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                motion_vel += dofs_state.cdof_vel[i_d, i_b] * src[i_d, i_b]
                motion_ang += dofs_state.cdof_ang[i_d, i_b] * src[i_d, i_b]
            j_l = links_info.parent_idx[J_l]
        rigid_links.motion_vel[i_l, i_b] = motion_vel
        rigid_links.motion_ang[i_l, i_b] = motion_ang


@qd.func(requires_top_level=True)
def func_link_force_to_dofs(
    out: qd.Tensor,
    rigid_links: NewtonRigidLinks,
    dofs_state: DofsState,
    links_info: LinksInfo,
    rigid_config: qd.template(),
):
    """Add to a vector over the dofs the generalized force of the spatial force on every link, over the dofs of its
    chain."""
    qd.loop_config(name="newton_link_force_to_dofs")
    for i_l, i_b in qd.ndrange(rigid_links.force_vel.shape[0], rigid_links.force_vel.shape[1]):
        force_vel = rigid_links.force_vel[i_l, i_b]
        force_ang = rigid_links.force_ang[i_l, i_b]
        if force_vel.norm_sqr() + force_ang.norm_sqr() > 0.0:
            j_l = i_l
            while j_l >= 0:
                J_l = [j_l, i_b] if qd.static(rigid_config.batch_links_info) else j_l
                for i_d in range(links_info.dof_start[J_l], links_info.dof_end[J_l]):
                    qd.atomic_add(
                        out[i_d, i_b],
                        dofs_state.cdof_vel[i_d, i_b].dot(force_vel) + dofs_state.cdof_ang[i_d, i_b].dot(force_ang),
                    )
                j_l = links_info.parent_idx[J_l]


@qd.func(requires_top_level=True)
def func_clear_link_forces(rigid_links: NewtonRigidLinks):
    qd.loop_config(name="newton_clear_link_forces")
    for i_l, i_b in qd.ndrange(rigid_links.force_vel.shape[0], rigid_links.force_vel.shape[1]):
        rigid_links.force_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        rigid_links.force_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)


# ------------------------------------------------------------------------------------
# --------------------------------- constraint solve ---------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_gauss_cost_term(Ma: float, qf_smooth: float, qacc: float, acc_smooth: float):
    """Return the term of a dof in the Gauss cost 1/2 (a - a_smooth)^T M (a - a_smooth) of the rigid solver."""
    return 0.5 * (Ma - qf_smooth) * (qacc - acc_smooth)


@qd.func
def func_rigid_gradient(Ma: float, qf_smooth: float, qfrc_constraint: float, dt: float):
    """Return the gradient of the cost of the rigid solver, scaled by dt^4 to the units of the incremental potential, in
    the unknown x = dt^2 qacc of a dof."""
    return dt * dt * (Ma - qf_smooth - qfrc_constraint)


@qd.func(requires_top_level=True)
def func_rigid_init_substep(
    rigid_dofs: NewtonRigidDofs,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    has_rigid_rows: qd.template(),
):
    """
    Initialize the dofs for the Newton solve of the substep, and start the constraint solve as the rigid solver starts
    its own.

    The dofs start from the warm start of the constraint solve, or from their smooth acceleration. The constraint rows
    are grouped by island (see 'func_solve_init' in constraint/solver.py), and without constraint rows, the links are
    free of the constraints that their assembly resets otherwise (see 'LinksState.is_constrained' in data.py).
    """
    dt = rigid_info.substep_dt[None]
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        qacc = dyn_state.dofs.acc_smooth[i_d, i_b]
        if qd.static(has_rigid_rows):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
                qacc = constraint_state.qacc_ws[i_d, i_b]
        rigid_dofs.x[i_d, i_b] = dt * dt * qacc
    if qd.static(not has_rigid_rows):
        for i_l, i_b in qd.ndrange(*dyn_state.links.is_constrained.shape):
            dyn_state.links.is_constrained[i_l, i_b] = False
    if qd.static(has_rigid_rows):
        if qd.static(rigid_config.enable_signorini_contact):
            for i_cone, i_b in qd.ndrange(*constraint_state.cone_latch.shape):
                constraint_state.cone_latch[i_cone, i_b] = -1.0
        for i_b in range(rigid_dofs.x.shape[1]):
            func_group_constraints_by_island(i_b, constraint_state, rigid_config)
            for i_island in range(constraint_state.island.n_islands[i_b]):
                constraint_state.island.improved[i_island, i_b] = True


@qd.func(requires_top_level=True)
def func_rigid_update_constraint(
    rigid_cost: qd.Tensor,
    rigid_dofs: NewtonRigidDofs,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    has_rigid_rows: qd.template(),
):
    """Update the constraint state of the rigid solver at qacc = x / dt^2 and its cost, or its Gauss cost without
    constraint rows."""
    dt = rigid_info.substep_dt[None]
    qd.loop_config(name="newton_rigid_qacc")
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        rigid_dofs.qacc[i_d, i_b] = rigid_dofs.x[i_d, i_b] / (dt * dt)
    initialize_Ma(rigid_dofs.Ma, rigid_dofs.qacc, rigid_info, rigid_config)
    if qd.static(has_rigid_rows):
        initialize_Jaref(qacc=rigid_dofs.qacc, constraint_state=constraint_state, rigid_config=rigid_config)
        func_update_constraint(
            qacc=rigid_dofs.qacc,
            Ma=rigid_dofs.Ma,
            cost=rigid_cost,
            dyn_state=dyn_state,
            constraint_state=constraint_state,
            rigid_config=rigid_config,
        )
    else:
        qd.loop_config(name="newton_rigid_gauss_cost")
        for i_b in range(rigid_dofs.x.shape[1]):
            cost = gs.qd_float(0.0)
            for i_d in range(rigid_dofs.x.shape[0]):
                cost += func_gauss_cost_term(
                    rigid_dofs.Ma[i_d, i_b],
                    dyn_state.dofs.qf_smooth[i_d, i_b],
                    rigid_dofs.qacc[i_d, i_b],
                    dyn_state.dofs.acc_smooth[i_d, i_b],
                )
            rigid_cost[i_b] = cost


@qd.func(requires_top_level=True)
def func_rigid_assemble_gradient(
    rigid_dofs: NewtonRigidDofs,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    has_rigid_rows: qd.template(),
):
    """Set the gradient of the cost of the rigid solver in the dofs at the constraint state of the current iterate (see
    'func_rigid_update_constraint')."""
    dt = rigid_info.substep_dt[None]
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        qfrc_constraint = gs.qd_float(0.0)
        if qd.static(has_rigid_rows):
            qfrc_constraint = constraint_state.qfrc_constraint[i_d, i_b]
        rigid_dofs.gradient[i_d, i_b] = func_rigid_gradient(
            rigid_dofs.Ma[i_d, i_b], dyn_state.dofs.qf_smooth[i_d, i_b], qfrc_constraint, dt
        )


@qd.func(requires_top_level=True)
def func_rigid_factor(
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    extra_hessian: qd.template(),
    has_rigid_rows: qd.template(),
):
    """Factor the Hessian of the constraint solve with an extra Hessian (see RigidContactStiffness), which preconditions
    the dofs in the linear solve, the factor of the mass matrix that the rigid solver holds already doing so without
    constraint rows."""
    if qd.static(has_rigid_rows):
        func_hessian_and_cholesky_factor_direct(
            constraint_state, rigid_info, rigid_config, compute_envelope=True, extra_hessian=extra_hessian
        )


@qd.func
def func_rigid_solve_batch(
    i_b: int,
    rhs: qd.Tensor,
    out: qd.Tensor,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    has_rigid_rows: qd.template(),
):
    """
    Solve the block of the dofs of the rigid solver of an environment against a vector over them.

    The solve of `rhs` into `out` goes through the factor of the rigid solver, of its block of the Hessian (see
    'func_rigid_factor') or of its mass matrix without constraint rows.
    """
    if qd.static(has_rigid_rows):
        for i_island in range(constraint_state.island.n_islands[i_b]):
            func_cholesky_solve_batch(
                i_b, i_island, rhs=rhs, out=out, constraint_state=constraint_state, rigid_config=rigid_config
            )
    else:
        func_solve_mass_batch(i_b, rhs, out, dyn_state, rigid_info, rigid_config)


@qd.func(requires_top_level=True)
def func_rigid_matvec(
    rigid_dofs: NewtonRigidDofs,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    has_rigid_rows: qd.template(),
):
    """Set Ap = (M + J^T D J) p over the dofs of the rigid solver, J and D those of its active constraint rows."""
    initialize_Ma(rigid_dofs.Ap, rigid_dofs.p, rigid_info, rigid_config)
    if qd.static(has_rigid_rows):
        qd.loop_config(name="newton_rigid_matvec_rows")
        for i_b in range(rigid_dofs.p.shape[1]):
            for i_c in range(constraint_state.n_constraints[i_b]):
                if constraint_state.active[i_c, i_b]:
                    jac_p = gs.qd_float(0.0)
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        jac_p += constraint_state.jac[i_c, i_d, i_b] * rigid_dofs.p[i_d, i_b]
                    efc_D_jac_p = constraint_state.efc_D[i_c, i_b] * jac_p
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        rigid_dofs.Ap[i_d, i_b] = (
                            rigid_dofs.Ap[i_d, i_b] + constraint_state.jac[i_c, i_d, i_b] * efc_D_jac_p
                        )


# ------------------------------------------------------------------------------------
# ------------------------------------ line search -----------------------------------
# ------------------------------------------------------------------------------------


@qd.func(requires_top_level=True)
def func_rigid_init_line_search(
    envs_mask: qd.Tensor,
    max_displacement: qd.Tensor,
    is_measured: qd.Tensor,
    rigid_dofs: NewtonRigidDofs,
    has_measure: qd.template(),
):
    """
    Set the Newton direction of the dofs of every environment selected, minus the solution of the linear solve, from
    their start point.

    The largest update of a dof adds to the entry of its environment in `max_displacement`, with `has_measure` that of
    the dofs `is_measured` selects alone, the others being measured otherwise (see 'IPCProxyDofsInfo' in
    rigid/contact_proxy.py).
    """
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        if envs_mask[i_b]:
            dx = -rigid_dofs.solution[i_d, i_b]
            rigid_dofs.dx[i_d, i_b] = dx
            rigid_dofs.x_start[i_d, i_b] = rigid_dofs.x[i_d, i_b]
            is_kept = True
            if qd.static(has_measure):
                is_kept = is_measured[i_d]
            if is_kept:
                qd.atomic_max(max_displacement[i_b], qd.abs(dx))


@qd.func(requires_top_level=True)
def func_rigid_step_forward(rigid_dofs: NewtonRigidDofs, envs_mask: qd.Tensor, alpha: qd.Tensor):
    """
    Move the dofs of every environment selected to the trial point of its step along the Newton direction.

    The step of an environment is its entry in `alpha`.
    """
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        if envs_mask[i_b]:
            rigid_dofs.x[i_d, i_b] = rigid_dofs.x_start[i_d, i_b] + alpha[i_b] * rigid_dofs.dx[i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_finish(
    rigid_cost: qd.Tensor,
    rigid_dofs: NewtonRigidDofs,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    has_rigid_rows: qd.template(),
):
    """
    Hand the accelerations of the dofs solved to the rigid solver, which finishes its substep from them (see
    'RigidSolver.substep_post_coupling').

    The constraint state takes their values, and its constraint force the whole force beyond the smooth one, the forces
    of the contacts of the particles included. Without constraint rows, the dofs take their acceleration and their force
    at once, which the constraint solve sets otherwise (see 'func_update_qacc' in constraint/solver.py). A frozen
    environment (see 'RigidInfo.is_env_frozen') takes the smooth acceleration of its dofs, which keeps every value the
    rigid solver computes from them finite.
    """
    dt = rigid_info.substep_dt[None]
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        if rigid_info.is_env_frozen[i_b]:
            rigid_dofs.x[i_d, i_b] = dt * dt * dyn_state.dofs.acc_smooth[i_d, i_b]
    func_rigid_update_constraint(
        rigid_cost, rigid_dofs, dyn_state, constraint_state, rigid_info, rigid_config, has_rigid_rows
    )
    for i_d, i_b in qd.ndrange(rigid_dofs.x.shape[0], rigid_dofs.x.shape[1]):
        if qd.static(has_rigid_rows):
            constraint_state.qacc[i_d, i_b] = rigid_dofs.qacc[i_d, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = rigid_dofs.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b]
        else:
            dyn_state.dofs.acc[i_d, i_b] = rigid_dofs.qacc[i_d, i_b]
            dyn_state.dofs.force[i_d, i_b] = rigid_dofs.Ma[i_d, i_b]
