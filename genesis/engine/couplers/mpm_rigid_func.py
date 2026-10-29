"""MPM-Rigid coupling stages of the implicit MPM solver: one Newton solve over the MPM grid displacement and the Rigid
dofs.

Rigid geoms enter the MPM energy as soft contact rows with Rigid's own contact model: the deepest geom of a particle
within its soft range gives the pyramid rows of its friction cone, with the geom's friction and solref / solimp
impedance, and the summed inverse weight of the link and of the particle in the implicit system (the material it is
embedded in, see `particle_inv_weight`). A row acts on the displacement of the particle relative to its contact point on
the link.

The Rigid unknown is x = dt^2 qacc per dof, so that the link point moves by dt v + J_pt x over the substep; Rigid's
convex primal cost (its Gauss term and constraint rows, as Rigid assembled them for the substep) times dt^4 joins the MPM
energy in the same displacement units. Its gradient is dt^2 times Rigid's, its Hessian Rigid's own M + J^T D J. The
joint Newton system is solved by one PCG over (u, x), whose preconditioner is block diagonal: the MPM brick
preconditioner, and Rigid's island Cholesky factor of M + J^T D J. The contact rows couple the blocks through the
spatial force and displacement of each link: a particle at arm r from the root COM of its link moves with
dr = V_lin + V_ang x r for the link's spatial displacement V = sum over the dofs of its chain of cdof x_d, and its row
gradient g pushes the dofs of that chain with the spatial force (g, r x g).
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.solvers.mpm_data import (
    CONTACT_ROWS,
    PYRAMID_ROWS,
    WALL_ROWS,
    MpmParticles,
)
from genesis.engine.solvers.mpm_func import (
    band_row,
    contact_band_width,
    particle_inv_weight,
    particle_of_sorted,
    q1_stencil,
    stencil_node_slots,
)
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_solve_mass_batch, func_vel_at_point
from genesis.engine.solvers.rigid.constraint.island import func_group_constraints_by_island
from genesis.engine.solvers.rigid.constraint.solver import (
    func_cholesky_solve_batch,
    func_hessian_and_cholesky_factor_direct,
    func_update_constraint,
    initialize_Jaref,
    initialize_Ma,
)
from genesis.utils import array_class


# ============================================ Layer 0: spatial algebra =============================================


@qd.func
def link_spatial(
    coupling: qd.template(),  # MpmRigidCoupling
    i_l,
    i_b,
):
    """Linear and angular part of the spatial displacement of link i_l held in `coupling.link_V`."""
    lin = qd.Vector([coupling.link_V[i_l, i_b, d] for d in qd.static(range(3))], dt=gs.qd_float)
    ang = qd.Vector([coupling.link_V[i_l, i_b, 3 + d] for d in qd.static(range(3))], dt=gs.qd_float)
    return lin, ang


@qd.func
def particle_arm(
    particles: MpmParticles,
    links_state: array_class.LinksState,
    f,
    i_p,
    i_b,
    i_l,
):
    """Arm of a particle from the root COM of its contact link, the reference point of the link's spatial vectors."""
    return particles.pos[f, i_p, i_b] - links_state.root_COM[i_l, i_b]


@qd.func
def particle_stencil(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    params: qd.template(),  # MpmParams
    f,
    s,
):
    """Particle index, environment, stencil node slots and weights of sorted particle s."""
    i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
    slots = stencil_node_slots(grid, env, cell)
    w, grad_w = q1_stencil(particles.pos[f, i_p, i_b] * params.inv_dx[()] - cell.cast(gs.qd_float), params.inv_dx[()])
    return i_p, i_b, slots, w


# =============================================== Layer 0: geom distance ===============================================


@qd.func
def geom_sdf_grid(
    i_g,
    i_b,
    x,
    geoms_state: array_class.GeomsState,
    sdf_info: array_class.SDFInfo,
):
    """Signed distance and its gradient in the mesh frame of a geom with an SDF grid, continuous everywhere.

    Inside the grid, the trilinear interpolant of the node distances and its analytic gradient, consistent with each
    other. Outside, the value at the point clamped onto the grid plus the distance to it: an upper bound of the distance
    that joins the interpolant on the grid boundary. Rigid's own query returns a value above every in-grid one there,
    for comparisons only, which a contact range reaching past the grid would see as a jump.
    """
    pos_mesh = gu.qd_inv_transform_by_trans_quat(x, geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b])
    pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
    res = sdf_info.geoms_info.sdf_res[i_g]
    cs = sdf_info.geoms_info.sdf_cell_size[i_g]
    upper = (res - 1).cast(gs.qd_float)
    q = qd.min(qd.max(pos_sdf, 0.0), upper)
    base = qd.max(qd.min(qd.floor(q, gs.qd_int), res - 2), 0)
    value = gs.qd_float(0.0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        node = base + offset
        w = 1.0 - qd.abs(q - node.cast(gs.qd_float))
        val = sdf_info.geoms_sdf_val[sdf.sdf_func_ravel_cell_idx(node, i_g, res, sdf_info)]
        value += w[0] * w[1] * w[2] * val
        grad[0] += (2 * offset[0] - 1) * w[1] * w[2] * val / cs[0]
        grad[1] += w[0] * (2 * offset[1] - 1) * w[2] * val / cs[1]
        grad[2] += w[0] * w[1] * (2 * offset[2] - 1) * val / cs[2]
    outside = qd.Vector([(pos_sdf[d] - q[d]) * cs[d] for d in qd.static(range(3))], dt=gs.qd_float)
    distance = outside.norm()
    if distance > 0.0:
        for d in qd.static(range(3)):
            if outside[d] != 0.0:
                grad[d] = outside[d] / distance
    return value + distance, grad


@qd.func
def geom_sdf(
    i_g,
    i_b,
    x,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
):
    """Signed distance of a world point to a geom, by `geom_sdf_grid` for the geoms with an SDF grid."""
    sd = gs.qd_float(0.0)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.PLANE or geom_type == gs.GEOM_TYPE.TERRAIN:
        sd = sdf.sdf_func_world(i_g, i_b, x, geoms_state, geoms_info, sdf_info)
    else:
        sd, grad = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info)
    return sd


@qd.func
def geom_normal(
    i_g,
    i_b,
    x,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """Outward unit normal of a geom at a world point: the gradient of the distance `geom_sdf` measures."""
    normal = qd.Vector.zero(gs.qd_float, 3)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.PLANE or geom_type == gs.GEOM_TYPE.TERRAIN:
        normal = sdf.sdf_func_normal_world(
            i_g, i_b, x, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
        )
    else:
        sd, grad_mesh = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info)
        normal = gu.qd_normalize(gu.qd_transform_by_quat(grad_mesh, geoms_state.quat[i_g, i_b]), rigid_info.EPS[None])
    return normal


# =============================================== Layer 1: contact rows ===============================================


@qd.func(requires_top_level=True)
def func_build_rigid_rows(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
    params: qd.template(),  # MpmParams
    f,
    dt,
):
    """Pyramid contact rows of every valid particle within the soft range of a coupled geom, after
    `func_build_wall_rows` reset the rows.

    The geom's `coup_softness` is a contact margin: a row starts at that distance from the surface, and its constraint
    position is the signed distance minus the margin, so the resting penetration of the soft row is taken up outside
    the geom. As in Rigid's contact assembly, the friction coefficient is `max(friction * friction_ratio, 0.01)`, the
    timeconst is floored at twice the substep, and every row `normal + mu d_k` of the pyramid gets
    `diag = 2 mu_reg^2 (1 + mu_reg^2) invweight (1 - imp) / imp`, with `mu_reg^2 = mu^2 / impratio`. The row target
    `dt J.v_p + dt^2 aref` is on the displacement of the particle relative to the motion J_pt x of its contact point
    beyond the link's velocity at the substep start. With `params.contact_band`, the nearest geom also gives rows to a
    particle short of its range by less than `contact_band_width`, by `band_row`.
    """
    n_geoms = geoms_info.link_idx.shape[0]
    use_band = params.contact_band[()] != 0
    qd.loop_config(name="mpm_build_rigid_rows")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
        x_p = particles.pos[f, i_p, i_b]
        # The nearest coupled geom, by its gap to the contact range
        i_g_contact = -1
        phi = gs.qd_float(float("inf"))
        for i_g in range(n_geoms):
            if geoms_info.needs_coup[i_g]:
                phi_g = geom_sdf(i_g, i_b, x_p, geoms_state, geoms_info, sdf_info) - geoms_info.coup_softness[i_g]
                if phi_g < phi:
                    phi = phi_g
                    i_g_contact = i_g
        i_l = -1
        v_r = qd.Vector.zero(gs.qd_float, 3)
        v_rel = qd.Vector.zero(gs.qd_float, 3)
        is_candidate = False
        if i_g_contact >= 0:
            i_l = geoms_info.link_idx[i_g_contact]
            v_r = func_vel_at_point(i_l, i_b, x_p, links_state)
            v_rel = particles.vel[f, i_p, i_b] - v_r
            # The relative speed bounds the normal approach speed, which needs the normal
            is_candidate = phi < 0.0 or (use_band and phi < contact_band_width(params, v_rel.norm(), dt))
        normal = qd.Vector.zero(gs.qd_float, 3)
        in_range = False
        if is_candidate:
            normal = geom_normal(
                i_g_contact, i_b, x_p, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
            )
            in_range = phi < 0.0 or (use_band and phi < contact_band_width(params, -normal.dot(v_rel), dt))
        if in_range:
            I_l = [i_l, i_b] if qd.static(config.rigid_batch_links_info) else i_l
            friction = qd.max(geoms_info.friction[i_g_contact] * geoms_state.friction_ratio[i_g_contact, i_b], 1e-2)
            friction_sq_reg = friction * friction / rigid_info.impratio[None]
            slots = stencil_node_slots(grid, env, cell)
            w, grad_w = q1_stencil(x_p * params.inv_dx[()] - cell.cast(gs.qd_float), params.inv_dx[()])
            invweight = particle_inv_weight(grid, slots, w, normal) + links_info.invweight[I_l][0]
            diag_coef = invweight * (1.0 + friction_sq_reg) * 2.0 * friction_sq_reg
            sol_params = geoms_info.sol_params[i_g_contact]
            sol_params[0] = qd.max(sol_params[0], 2.0 * dt)
            d1, d2 = gu.qd_orthogonals(normal)
            for i in qd.static(range(PYRAMID_ROWS)):
                d = d1
                if qd.static(i >= 2):
                    d = d2
                J = normal + (qd.static(2 * (i % 2) - 1) * friction) * d
                D, t = band_row(sol_params, phi, J.dot(v_rel), diag_coef, dt)
                k = CONTACT_ROWS * s + WALL_ROWS + i
                for e in qd.static(range(3)):
                    newton.contact_dir[3 * k + e] = J[e]
                newton.contact_D[k] = D
                newton.contact_t[k] = t + dt * J.dot(v_r)
            newton.contact_link[s] = i_l


# ============================================ Layer 1: link kinematics =============================================


@qd.func(requires_top_level=True)
def func_link_spatial_disp(
    coupling: qd.template(),  # MpmRigidCoupling
    src: qd.template(),  # qd.Ndarray (n_dofs, n_envs), a dof vector in units of x
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    config: qd.template(),  # MpmStaticConfig
):
    """link_V = sum over the dofs of the link's chain of cdof src_d, for every link; link_F is cleared alongside."""
    n_links = coupling.link_V.shape[0]
    n_envs = coupling.link_V.shape[1]
    qd.loop_config(name="mpm_link_spatial_disp")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        lin = qd.Vector.zero(gs.qd_float, 3)
        ang = qd.Vector.zero(gs.qd_float, 3)
        i_a = i_l
        while i_a >= 0:
            I_a = [i_a, i_b] if qd.static(config.rigid_batch_links_info) else i_a
            for i_d in range(links_info.dof_start[I_a], links_info.dof_end[I_a]):
                lin += dofs_state.cdof_vel[i_d, i_b] * src[i_d, i_b]
                ang += dofs_state.cdof_ang[i_d, i_b] * src[i_d, i_b]
            i_a = links_info.parent_idx[I_a]
        for d in qd.static(range(3)):
            coupling.link_V[i_l, i_b, d] = lin[d]
            coupling.link_V[i_l, i_b, 3 + d] = ang[d]
            coupling.link_F[i_l, i_b, d] = 0.0
            coupling.link_F[i_l, i_b, 3 + d] = 0.0


@qd.func(requires_top_level=True)
def func_link_force_to_dofs(
    coupling: qd.template(),  # MpmRigidCoupling
    out: qd.template(),  # qd.Ndarray (n_dofs, n_envs)
    scale,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    config: qd.template(),  # MpmStaticConfig
):
    """out_d += scale cdof_d . link_F over the links whose chain holds dof d: the generalized force of the spatial
    forces accumulated on the links."""
    n_links = coupling.link_V.shape[0]
    n_envs = coupling.link_V.shape[1]
    qd.loop_config(name="mpm_link_force_to_dofs")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        F_lin = qd.Vector([coupling.link_F[i_l, i_b, d] for d in qd.static(range(3))], dt=gs.qd_float)
        F_ang = qd.Vector([coupling.link_F[i_l, i_b, 3 + d] for d in qd.static(range(3))], dt=gs.qd_float)
        if F_lin.norm_sqr() + F_ang.norm_sqr() > 0.0:
            i_a = i_l
            while i_a >= 0:
                I_a = [i_a, i_b] if qd.static(config.rigid_batch_links_info) else i_a
                for i_d in range(links_info.dof_start[I_a], links_info.dof_end[I_a]):
                    value = dofs_state.cdof_vel[i_d, i_b].dot(F_lin) + dofs_state.cdof_ang[i_d, i_b].dot(F_ang)
                    qd.atomic_add(out[i_d, i_b], scale * value)
                i_a = links_info.parent_idx[I_a]


@qd.func(requires_top_level=True)
def func_contact_point_disp(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    coupling: qd.template(),  # MpmRigidCoupling
    links_state: array_class.LinksState,
    params: qd.template(),  # MpmParams
    f,
    out: qd.template(),  # qd.Ndarray, 3 per sorted particle
):
    """Displacement J_pt y of the contact point of every particle with Rigid rows, for the dof vector y whose link
    displacements `link_V` holds."""
    qd.loop_config(name="mpm_contact_point_disp")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        dr = qd.Vector.zero(gs.qd_float, 3)
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            lin, ang = link_spatial(coupling, i_l, i_b)
            dr = lin + ang.cross(particle_arm(particles, links_state, f, i_p, i_b, i_l))
        for d in qd.static(range(3)):
            out[3 * s + d] = dr[d]


@qd.func(requires_top_level=True)
def func_rigid_line_model(
    coupling: qd.template(),  # MpmRigidCoupling
    newton: qd.template(),  # MpmNewton
):
    """The Rigid dofs' sol.b and sol.r in the sums of the one-dimensional line-search model, after func_line_model."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_line_model")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        qd.atomic_add(newton.ls_sums[0], coupling.sol[i_d, i_b] * coupling.grad[i_d, i_b])
        qd.atomic_add(newton.ls_sums[1], coupling.sol[i_d, i_b] * coupling.r[i_d, i_b])


# ============================================== Layer 1: Rigid block ===============================================


@qd.func(requires_top_level=True)
def func_rigid_begin(
    coupling: qd.template(),  # MpmRigidCoupling
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
    params: qd.template(),  # MpmParams
):
    """Start the Rigid unknown from Rigid's warm start, as Rigid's own solve does, and group Rigid's assembled rows by
    island with every island iterating, so that its factor covers every dof whether or not the island has rows."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    dt = params.substep_dt[()]
    qd.loop_config(name="mpm_rigid_begin")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        qacc = dofs_state.acc_smooth[i_d, i_b]
        if qd.static(config.has_rigid_rows):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.is_warmstart[i_b]:
                qacc = constraint_state.qacc_ws[i_d, i_b]
        coupling.x[i_d, i_b] = dt * dt * qacc
    if qd.static(config.has_rigid_rows):
        qd.loop_config(name="mpm_rigid_group_islands")
        for i_b in range(n_envs):
            func_group_constraints_by_island(i_b, constraint_state, rigid_config)
            for i_island in range(constraint_state.island.n_islands[i_b]):
                constraint_state.island.improved[i_island, i_b] = 1


@qd.func(requires_top_level=True)
def func_rigid_evaluate(
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
    params: qd.template(),  # MpmParams
):
    """Rigid's cost of qacc = x / dt^2 in `coupling.cost`, with its row activity, forces and qfrc_constraint, by Rigid's
    own update of its constraint state; the Gauss term alone in a scene without Rigid constraint rows."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    inv_dt_sq = 1.0 / (params.substep_dt[()] * params.substep_dt[()])
    qd.loop_config(name="mpm_rigid_qacc")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        coupling.qacc[i_d, i_b] = coupling.x[i_d, i_b] * inv_dt_sq
    initialize_Ma(coupling.Ma, coupling.qacc, dyn_info, rigid_info, rigid_config)
    if qd.static(config.has_rigid_rows):
        initialize_Jaref(qacc=coupling.qacc, constraint_state=constraint_state, rigid_config=rigid_config)
        func_update_constraint(
            qacc=coupling.qacc,
            Ma=coupling.Ma,
            cost=coupling.cost,
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
                    * (coupling.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b])
                    * (coupling.qacc[i_d, i_b] - dyn_state.dofs.acc_smooth[i_d, i_b])
                )
            coupling.cost[i_b] = cost


@qd.func(requires_top_level=True)
def func_rigid_energy(
    coupling: qd.template(),  # MpmRigidCoupling
    control: qd.template(),  # NewtonControl
    params: qd.template(),  # MpmParams
    is_trial,
):
    """Add Rigid's cost times dt^4, after func_rigid_evaluate, to the start or trial energy of the MPM energy pass."""
    n_envs = coupling.x.shape[1]
    dt_sq = params.substep_dt[()] * params.substep_dt[()]
    for _ in range(1):
        total = gs.qd_float(0.0)
        for i_b in range(n_envs):
            total += coupling.cost[i_b]
        if is_trial:
            control.energy_trial[()] += dt_sq * dt_sq * total
        else:
            control.energy_start[()] += dt_sq * dt_sq * total


@qd.func(requires_top_level=True)
def func_rigid_gradient(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
    params: qd.template(),  # MpmParams
    f,
):
    """Gradient of the joint energy in x, after func_rigid_evaluate at the current x and the MPM assembly that staged
    the row gradients: dt^2 (M qacc - qf_smooth - qfrc_constraint) minus the generalized force of the Rigid rows of the
    particles on their links. Rigid's island factor of M + J^T D J is refreshed alongside, for the preconditioner."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    dt_sq = params.substep_dt[()] * params.substep_dt[()]
    qd.loop_config(name="mpm_rigid_gradient")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        value = coupling.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b]
        if qd.static(config.has_rigid_rows):
            value -= constraint_state.qfrc_constraint[i_d, i_b]
        coupling.grad[i_d, i_b] = dt_sq * value
    func_link_spatial_disp(coupling, coupling.x, dyn_info.links, dyn_state.dofs, config)
    qd.loop_config(name="mpm_rigid_row_forces")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            g = qd.Vector([newton.g_rigid[3 * s + d] for d in qd.static(range(3))], dt=gs.qd_float)
            if g.norm_sqr() > 0.0:
                r = particle_arm(particles, dyn_state.links, f, i_p, i_b, i_l)
                torque = r.cross(g)
                for d in qd.static(range(3)):
                    qd.atomic_add(coupling.link_F[i_l, i_b, d], g[d])
                    qd.atomic_add(coupling.link_F[i_l, i_b, 3 + d], torque[d])
    func_link_force_to_dofs(coupling, coupling.grad, -1.0, dyn_info.links, dyn_state.dofs, config)
    if qd.static(config.has_rigid_rows):
        func_hessian_and_cholesky_factor_direct(
            constraint_state, dyn_info, rigid_info, rigid_config, compute_envelope=True
        )


@qd.func(requires_top_level=True)
def func_rigid_precondition(
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
):
    """z = (M + J^T D J)^-1 r over the Rigid dofs by Rigid's island Cholesky factor, or z = M^-1 r by Rigid's mass
    factor in a scene without Rigid constraint rows. The MPM contact stiffness on the links is left to the iterations:
    its coupling of the blocks is of the rank of the dofs of the contacting chains at most."""
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_precondition")
    for i_b in range(n_envs):
        if qd.static(config.has_rigid_rows):
            for i_island in range(constraint_state.island.n_islands[i_b]):
                func_cholesky_solve_batch(
                    i_b,
                    i_island,
                    rhs=coupling.r,
                    out=coupling.z,
                    constraint_state=constraint_state,
                    rigid_config=rigid_config,
                )
        else:
            func_solve_mass_batch(i_b, coupling.r, coupling.z, dyn_state, rigid_info, rigid_config)


# ========================================= Layer 1: joint PCG over (u, x) ==========================================


@qd.func(requires_top_level=True)
def func_rigid_pcg_init(
    coupling: qd.template(),  # MpmRigidCoupling
):
    """Rigid part of the PCG start from 0: r = grad, solution and Ap cleared."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_init")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        coupling.r[i_d, i_b] = coupling.grad[i_d, i_b]
        coupling.sol[i_d, i_b] = 0.0
        coupling.ap[i_d, i_b] = 0.0


@qd.func(requires_top_level=True)
def func_rigid_pcg_init_direction(
    coupling: qd.template(),  # MpmRigidCoupling
    ws: qd.template(),  # PCGWorkspace
):
    """p = z over the Rigid dofs, and r.z added to rz before the threshold is re-derived, after
    func_init_linear_pcg_direction did the same over the MPM rows."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_init_direction")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        coupling.p[i_d, i_b] = coupling.z[i_d, i_b]
        qd.atomic_add(ws.rz_slots[ws.parity[()]], coupling.r[i_d, i_b] * coupling.z[i_d, i_b])
    for _ in range(1):
        ws.tol_sq[()] = ws.tol_rate[()] * qd.abs(ws.rz_slots[ws.parity[()]])


@qd.func(requires_top_level=True)
def func_rigid_pcg_matvec(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
    params: qd.template(),  # MpmParams
    f,
):
    """The joint operator beyond the MPM matrix, after func_linear_pcg_spmv, on p = (p_u, p_r).

    A particle with Rigid rows of Hessian H, trial displacement dxp = sum_a w_a p_u,a and contact point displacement dr of
    p_r adds -w_a H dr to Ap_u,a and the generalized force of the spatial force (h, r x h), h = H (dr - dxp), to Ap_r;
    Ap_r also takes (M + J^T D J) p_r over Rigid's active rows. p.Ap takes p_u.(-sum w_a H dr) and p_r.Ap_r.
    """
    ws = linear.pcg
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    func_link_spatial_disp(coupling, coupling.p, dyn_info.links, dyn_state.dofs, config)
    qd.loop_config(name="mpm_rigid_matvec_particles")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0 and ws.converged[()] == 0:
            i_p, i_b, slots, w = particle_stencil(particles, grid, params, f, s)
            H = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i, j in qd.static(qd.ndrange(3, 3)):
                H[i, j] = newton.h_rigid[9 * s + 3 * i + j]
            r = particle_arm(particles, dyn_state.links, f, i_p, i_b, i_l)
            lin, ang = link_spatial(coupling, i_l, i_b)
            dr = lin + ang.cross(r)
            dxp = qd.Vector.zero(gs.qd_float, 3)
            for a in qd.static(range(8)):
                row = 3 * (linear.row_offset + slots[a])
                dxp += w[a] * qd.Vector([ws.p[row + d] for d in qd.static(range(3))], dt=gs.qd_float)
            H_dr = H @ dr
            for a in qd.static(range(8)):
                row = 3 * (linear.row_offset + slots[a])
                for d in qd.static(range(3)):
                    qd.atomic_add(ws.ap[row + d], -w[a] * H_dr[d])
            h = H @ (dr - dxp)
            torque = r.cross(h)
            for d in qd.static(range(3)):
                qd.atomic_add(coupling.link_F[i_l, i_b, d], h[d])
                qd.atomic_add(coupling.link_F[i_l, i_b, 3 + d], torque[d])
            qd.atomic_add(ws.p_dot_ap[()], -dxp.dot(H_dr))
    initialize_Ma(coupling.ap, coupling.p, dyn_info, rigid_info, rigid_config)
    if qd.static(config.has_rigid_rows):
        qd.loop_config(name="mpm_rigid_matvec_rows")
        for i_b in range(n_envs):
            for i_c in range(constraint_state.n_constraints[i_b]):
                if constraint_state.active[i_c, i_b]:
                    jp = gs.qd_float(0.0)
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        jp += constraint_state.jac[i_c, i_d, i_b] * coupling.p[i_d, i_b]
                    coef = constraint_state.efc_D[i_c, i_b] * jp
                    for i_d_ in range(constraint_state.jac_n_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_dofs_idx[i_c, i_d_, i_b]
                        coupling.ap[i_d, i_b] += coef * constraint_state.jac[i_c, i_d, i_b]
    func_link_force_to_dofs(coupling, coupling.ap, 1.0, dyn_info.links, dyn_state.dofs, config)
    qd.loop_config(name="mpm_rigid_matvec_dot")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if ws.converged[()] == 0:
            qd.atomic_add(ws.p_dot_ap[()], coupling.p[i_d, i_b] * coupling.ap[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_pcg_update_xr(
    coupling: qd.template(),  # MpmRigidCoupling
    ws: qd.template(),  # PCGWorkspace
):
    """x += alpha p and r -= alpha Ap over the Rigid dofs, with the alpha of func_linear_pcg_update_xr."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_update_xr")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if ws.converged[()] == 0:
            p_dot_ap = ws.p_dot_ap[()]
            alpha = ws.rz_slots[ws.parity[()]] / p_dot_ap if p_dot_ap != 0.0 else gs.qd_float(0.0)
            coupling.sol[i_d, i_b] += alpha * coupling.p[i_d, i_b]
            coupling.r[i_d, i_b] -= alpha * coupling.ap[i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_pcg_dot_rz(
    coupling: qd.template(),  # MpmRigidCoupling
    ws: qd.template(),  # PCGWorkspace
):
    """r.z over the Rigid dofs into the rz_new slot, before func_linear_pcg_direction_update adds the MPM rows."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_dot_rz")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if ws.converged[()] == 0:
            qd.atomic_add(ws.rz_slots[1 - ws.parity[()]], coupling.r[i_d, i_b] * coupling.z[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_pcg_direction(
    coupling: qd.template(),  # MpmRigidCoupling
    ws: qd.template(),  # PCGWorkspace
):
    """p = z + beta p over the Rigid dofs and Ap cleared, after func_linear_pcg_direction_update flipped the parity:
    rz_new now sits in the parity slot and rz in the other."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_direction")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        coupling.ap[i_d, i_b] = 0.0
        if ws.converged[()] == 0:
            rz = ws.rz_slots[1 - ws.parity[()]]
            beta = ws.rz_slots[ws.parity[()]] / rz if rz != 0.0 else gs.qd_float(0.0)
            coupling.p[i_d, i_b] = coupling.z[i_d, i_b] + beta * coupling.p[i_d, i_b]


# ============================================== Layer 1: Newton steps ==============================================


@qd.func(requires_top_level=True)
def func_rigid_start_line_search(
    coupling: qd.template(),  # MpmRigidCoupling
    control: qd.template(),  # NewtonControl
):
    """dx = -solution, the line-search start point, and |dx| into the Newton measure of the step, after the MPM rows."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_start_line_search")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        coupling.dx[i_d, i_b] = -coupling.sol[i_d, i_b]
        coupling.x_temp[i_d, i_b] = coupling.x[i_d, i_b]
        qd.atomic_max(control.max_disp[()], qd.abs(coupling.sol[i_d, i_b]))


@qd.func(requires_top_level=True)
def func_rigid_step_forward(
    coupling: qd.template(),  # MpmRigidCoupling
    control: qd.template(),  # NewtonControl
):
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_step_forward")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        coupling.x[i_d, i_b] = coupling.x_temp[i_d, i_b] + control.alpha[()] * coupling.dx[i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_write_back(
    coupling: qd.template(),  # MpmRigidCoupling
    constraint_state: array_class.ConstraintState,
    config: qd.template(),  # MpmStaticConfig
):
    """Hand the converged qacc to Rigid as the solution of its constraint solve, after func_rigid_evaluate at the
    converged x left Rigid's row activity, forces and qfrc_constraint consistent with it."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_write_back")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.qacc[i_d, i_b] = coupling.qacc[i_d, i_b]
        if qd.static(not config.has_rigid_rows):
            constraint_state.qfrc_constraint[i_d, i_b] = 0.0
