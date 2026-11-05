"""Stages of the coupling between the implicit material point method (MPM) and Rigid: one Newton solve over the grid
displacement and the Rigid dofs.

Rigid geoms enter the MPM energy as soft contact rows with Rigid's own contact model: the deepest geom of a particle
within its soft range gives the pyramid rows of its friction cone, with the geom's friction and solref / solimp
impedance, and the summed inverse weight of the link and of the particle in the implicit system (the material it is
embedded in, see `func_contact_invweight`). A row acts on the displacement of the particle relative to its contact point
on the link.

The Rigid unknown is x = dt^2 qacc per dof, so that the link point moves by dt v + J_pt x over the substep. Rigid's
convex primal cost (its Gauss term and constraint rows, as Rigid assembled them for the substep) times dt^4 joins the
MPM energy in the same displacement units. Its gradient is dt^2 times that of Rigid, and its Hessian is the M + J^T D J
of Rigid. One preconditioned conjugate gradient (PCG) over (u, x) solves the joint Newton system, with a block diagonal
preconditioner: the MPM brick preconditioner, and Rigid's island Cholesky factor of M + J^T D J corrected by the contact
stiffness of the links in contact (see func_rigid_factor). The contact rows couple the blocks through the spatial force
and displacement of each link: a particle at arm r from the root center of mass (COM) of its link moves by dr = V_lin +
V_ang x r for the link's spatial displacement V = sum over the dofs of its chain of cdof x_d, and its row gradient g
pushes the dofs of that chain with the spatial force (g, r x g).
"""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import (
    BVHQueryResults,
    ConstraintState,
    DofsState,
    DynInfo,
    DynState,
    GeomsInfo,
    GeomsState,
    LinksInfo,
    LinksState,
    RigidInfo,
    SDFInfo,
)
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import CONTACT_ROWS, PYRAMID_ROWS, WALL_ROWS, MpmParticles, MpmParticlesInfo
from genesis.engine.solvers.mpm_func import (
    EMPTY_BOX_HALF,
    band_row,
    contact_band_width,
    contact_cross_matvec,
    contact_invweight,
    contact_row_terms,
    particle_of_sorted,
    particle_trial_disp,
    pyramid_dirs,
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


# ============================================ Layer 0: spatial algebra =============================================


@qd.func
def particle_arm(f: int, i_p: int, i_b: int, i_l: int, particles: MpmParticles, links_state: LinksState):
    """Arm of a particle from the root center of mass (COM) of its contact link, the reference point of the link's
    spatial vectors."""
    return particles.pos[f, i_p, i_b] - links_state.root_COM[i_l, i_b]


# =============================================== Layer 0: geom distance ===============================================


@qd.func
def geom_sdf_grid(i_g: int, i_b: int, x: qd.types.vector(3), geoms_state: GeomsState, sdf_info: SDFInfo):
    """Continuous signed distance and its gradient in the mesh frame of a geom with a signed distance field (SDF) grid.

    Inside the grid, it returns the trilinear interpolant of the node distances and its analytic gradient, consistent
    with each other. Outside, it returns the value at the point clamped onto the grid plus the distance to it, an upper
    bound of the distance that joins the interpolant on the grid boundary. Rigid's own query returns there a value above
    every in-grid one, meant for comparisons only, which a contact range reaching past the grid would see as a jump.
    """
    pos_mesh = gu.qd_inv_transform_by_trans_quat(x, geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b])
    pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
    sdf_res = sdf_info.geoms_info.sdf_res[i_g]
    cell_size = sdf_info.geoms_info.sdf_cell_size[i_g]
    upper = (sdf_res - 1).cast(gs.qd_float)
    q = qd.min(qd.max(pos_sdf, 0.0), upper)
    base = qd.max(qd.min(qd.floor(q, gs.qd_int), sdf_res - 2), 0)
    value = gs.qd_float(0.0)
    grad = qd.Vector.zero(gs.qd_float, 3)
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        node = base + offset
        w = 1.0 - qd.abs(q - node.cast(gs.qd_float))
        sdf_val = sdf_info.geoms_sdf_val[sdf.sdf_func_ravel_cell_idx(node, i_g, sdf_res, sdf_info)]
        value += w[0] * w[1] * w[2] * sdf_val
        grad[0] += (2 * offset[0] - 1) * w[1] * w[2] * sdf_val / cell_size[0]
        grad[1] += w[0] * (2 * offset[1] - 1) * w[2] * sdf_val / cell_size[1]
        grad[2] += w[0] * w[1] * (2 * offset[2] - 1) * sdf_val / cell_size[2]
    outside = qd.Vector([(pos_sdf[d] - q[d]) * cell_size[d] for d in qd.static(range(3))], dt=gs.qd_float)
    distance = outside.norm()
    if distance > 0.0:
        for d in qd.static(range(3)):
            if pos_sdf[d] < 0.0 or pos_sdf[d] > upper[d]:
                grad[d] = outside[d] / distance
    return value + distance, grad


@qd.func
def geom_sdf(
    i_g: int,
    i_b: int,
    x: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    sdf_info: SDFInfo,
):
    """Signed distance of a world point to a geom, by `geom_sdf_grid` for the geoms with a distance grid."""
    signed_distance = gs.qd_float(0.0)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.PLANE or geom_type == gs.GEOM_TYPE.TERRAIN:
        signed_distance = sdf.sdf_func_world(i_g, i_b, x, geoms_state, geoms_info, sdf_info)
    else:
        signed_distance, grad = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info)
    return signed_distance


@qd.func
def geom_normal(
    i_g: int,
    i_b: int,
    x: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """Outward unit normal of a geom at a world point: the gradient of the distance measured by `geom_sdf`."""
    normal = qd.Vector.zero(gs.qd_float, 3)
    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE or geom_type == gs.GEOM_TYPE.PLANE or geom_type == gs.GEOM_TYPE.TERRAIN:
        normal = sdf.sdf_func_normal_world(i_g, i_b, x, geoms_state, geoms_info, rigid_info, sdf_info, collider_config)
    else:
        signed_distance, grad_mesh = geom_sdf_grid(i_g, i_b, x, geoms_state, sdf_info)
        normal = gu.qd_normalize(gu.qd_transform_by_quat(grad_mesh, geoms_state.quat[i_g, i_b]), rigid_info.EPS[None])
    return normal


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class RigidGeomQuery:
    """Linear bounding volume hierarchy (LBVH) over the coupled Rigid geoms of every environment.

    The tree comes from genesis.engine.bvh. The query also holds the data read by the filter of its queries: the geoms,
    and the query point and nearest geom of each sorted particle in `newton`. An inactive query holds no geom and skips
    the build."""

    def __init__(self, rigid, newton, n_envs, is_active):
        coupled_geoms = [geom.idx for geom in rigid.geoms if geom.needs_coup] if is_active else []
        self.n_coupled = len(coupled_geoms)
        self.coupled_geoms = qd.ndarray(gs.qd_int, shape=(max(self.n_coupled, 1),))
        if self.n_coupled:
            self.coupled_geoms.from_numpy(np.array(coupled_geoms, dtype=gs.np_int))
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(self.n_coupled, 1), is_active=self.n_coupled > 0)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        # Every query keeps its nearest geom in `newton` and records no pair
        self.results = BVHQueryResults(sizes=dict(query_results=1))
        self.geoms_state = rigid.dyn_state.geoms
        self.geoms_info = rigid.dyn_info.geoms
        self.geoms_init_AABB = rigid.geoms_init_AABB
        self.links_state = rigid.dyn_state.links
        self.links_info = rigid.dyn_info.links
        self.batch_links_info = rigid._options.batch_links_info
        self.sdf_info = rigid.collider.collider_info.sdf
        self.newton = newton

    def build(self, params):
        """Fit the leaf boxes to the geoms of the substep and build the trees."""
        if self.n_coupled:
            kernel_fit_geom_leaves(self, params)
            build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)


@qd.func(requires_top_level=True)
def func_fit_geom_leaves(dt: float, query: qd.template()):
    """Fit the leaf box of every coupled geom, grown by its contact margin and the distance its points may travel.

    The box is the world box of the posed initial box of the geom (as `func_update_geom_aabbs` fits it, a stage Rigid
    skips when collisions are disabled), grown by the contact margin of the geom and by twice the distance its points
    may travel over the substep, bounded by the link velocity at the box center and the spin over its radius. A geom
    that an environment does not carry gets an empty box with a finite center, for the Morton code."""
    qd.loop_config(name="mpm_fit_geom_leaves")
    for i_b, i_a in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        i_g = query.coupled_geoms[i_a]
        i_l = query.geoms_info.link_idx[i_g]
        is_carried = True
        if qd.static(query.batch_links_info):
            is_carried = query.links_info.geom_start[i_l, i_b] <= i_g < query.links_info.geom_end[i_l, i_b]
        if is_carried:
            geom_pos = query.geoms_state.pos[i_g, i_b]
            geom_quat = query.geoms_state.quat[i_g, i_b]
            box_min = gu.qd_vec3(qd.math.inf)
            box_max = gu.qd_vec3(-qd.math.inf)
            for i_corner in qd.static(range(8)):
                corner = gu.qd_transform_by_trans_quat(query.geoms_init_AABB[i_g, i_corner], geom_pos, geom_quat)
                box_min = qd.min(box_min, corner)
                box_max = qd.max(box_max, corner)
            center = 0.5 * (box_min + box_max)
            radius = 0.5 * (box_max - box_min).norm()
            speed = func_vel_at_point(i_l, i_b, center, query.links_state).norm()
            speed += query.links_state.cd_ang[i_l, i_b].norm() * radius
            margin = qd.max(query.geoms_info.coup_softness[i_g], 0.0) + 2.0 * dt * speed
            query.leaves.aabbs_min[i_b, i_a] = box_min - margin
            query.leaves.aabbs_max[i_b, i_a] = box_max + margin
        else:
            query.leaves.aabbs_min[i_b, i_a] = qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
            query.leaves.aabbs_max[i_b, i_a] = qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)


@qd.kernel
def kernel_fit_geom_leaves(query: qd.template(), params: qd.template()):
    func_fit_geom_leaves(params.substep_dt[()], query)


@qd.func
def func_nearest_geom_filter(i_t: int, i_a: int, i_q: int, query: qd.template()) -> bool:
    """Narrow phase of a box query of a sorted particle, which keeps the coupled geom of the nearest contact range.

    The nearest geom of sorted particle i_q in environment i_t goes to `newton.query_phi` and `newton.query_idx`, and
    the query records no pair."""
    i_g = query.coupled_geoms[i_a]
    x = query.newton.query_pos[i_q]
    phi = (
        geom_sdf(i_g, i_t, x, query.geoms_state, query.geoms_info, query.sdf_info) - query.geoms_info.coup_softness[i_g]
    )
    if phi < query.newton.query_phi[i_q]:
        query.newton.query_phi[i_q] = phi
        query.newton.query_idx[i_q] = i_g
    return True


# =============================================== Layer 1: contact rows ===============================================


@qd.func(requires_top_level=True)
def func_build_rigid_rows(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    geoms_state: GeomsState,
    links_state: LinksState,
    query: qd.template(),
    particles_info: MpmParticlesInfo,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    params: qd.template(),
    config: qd.template(),
):
    """Build the pyramid contact rows of every valid particle within the soft range of a coupled geom, after
    `func_build_wall_rows` reset the rows.

    A box query of the trees of the coupled geoms finds the nearest geom. The geom's `coup_softness` is a contact
    margin: a row starts at that distance from the surface, and its constraint position is the signed distance minus the
    margin, so the resting penetration of the soft row is taken up outside the geom. As in Rigid's contact assembly, the
    friction coefficient is `max(friction * friction_ratio, 0.01)`, the time constant is floored at twice the substep,
    and every row `normal + mu d_k` of the pyramid gets `diag = 2 mu_reg^2 (1 + mu_reg^2) invweight (1 - imp) / imp`,
    with `mu_reg^2 = mu^2 / impratio`. The row target `dt J.v_p + dt^2 aref` is on the displacement of the particle
    relative to the motion J_pt x of its contact point beyond the link's velocity at the substep start. The nearest geom
    gives rows, by `band_row`, to a particle within its range or short of it by less than `contact_band_width`.
    """
    qd.loop_config(name="mpm_build_rigid_rows")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
        x_p = particles.pos[f, i_p, i_b]
        # The nearest coupled geom, by its gap to the contact range, among those whose grown box meets the particle's
        # band: the two grown boxes together cover the band of the relative velocity
        newton.query_pos[s] = x_p
        newton.query_phi[s] = float("inf")
        newton.query_idx[s] = -1
        reach = params.dx[()] + 2.0 * dt * particles.vel[f, i_p, i_b].norm()
        func_bvh_query_aabb(
            i_b, s, x_p - reach, x_p + reach, query.tree, query.results, func_nearest_geom_filter, query
        )
        i_g_contact = newton.query_idx[s]
        phi = newton.query_phi[s]
        i_l = -1
        v_r = qd.Vector.zero(gs.qd_float, 3)
        v_rel = qd.Vector.zero(gs.qd_float, 3)
        is_candidate = False
        if i_g_contact >= 0:
            i_l = geoms_info.link_idx[i_g_contact]
            v_r = func_vel_at_point(i_l, i_b, x_p, links_state)
            v_rel = particles.vel[f, i_p, i_b] - v_r
            # The relative speed bounds the normal approach speed, which needs the normal
            is_candidate = phi < contact_band_width(v_rel.norm(), dt, params)
        normal = qd.Vector.zero(gs.qd_float, 3)
        is_in_range = False
        if is_candidate:
            normal = geom_normal(i_g_contact, i_b, x_p, geoms_state, geoms_info, rigid_info, sdf_info, config)
            is_in_range = phi < contact_band_width(-normal.dot(v_rel), dt, params)
        if is_in_range:
            I_l = [i_l, i_b] if qd.static(config.rigid_batch_links_info) else i_l
            friction = qd.max(geoms_info.friction[i_g_contact] * geoms_state.friction_ratio[i_g_contact, i_b], 1e-2)
            friction_sq_reg = friction * friction / rigid_info.impratio[None]
            invweight = contact_invweight(f, i_p, i_b, normal, particles, particles_info) + links_info.invweight[I_l][0]
            diag_coef = invweight * (1.0 + friction_sq_reg) * 2.0 * friction_sq_reg
            sol_params = geoms_info.sol_params[i_g_contact]
            sol_params[0] = qd.max(sol_params[0], 2.0 * dt)
            dirs = pyramid_dirs(normal, friction)
            for i in qd.static(range(PYRAMID_ROWS)):
                J = dirs[i, :]
                D, t = band_row(sol_params, phi, J.dot(v_rel), diag_coef, dt)
                k = CONTACT_ROWS * s + WALL_ROWS + i
                newton.contact_dir[k] = J
                newton.contact_D[k] = D
                newton.contact_t[k] = t + dt * J.dot(v_r)
            newton.contact_link[s] = i_l


# ============================================ Layer 1: link kinematics =============================================


@qd.func(requires_top_level=True)
def func_link_spatial_disp(
    src: qd.template(),
    rigid_data: qd.template(),
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
    rigid_data: qd.template(),
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
def func_contact_point_disp(
    f: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    rigid_data: qd.template(),
    links_state: LinksState,
    params: qd.template(),
):
    """Set the displacement J_pt x of the contact point of every particle with Rigid rows.

    x is the dof vector whose link displacements are held in `link_motion_vel` / `link_motion_ang`. The other particles
    keep theirs, from `func_reset_contact_rows` or from the coupling that gave them rows."""
    qd.loop_config(name="mpm_contact_point_disp")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            r = particle_arm(f, i_p, i_b, i_l, particles, links_state)
            newton.contact_dr[s] = rigid_data.link_motion_vel[i_l, i_b] + rigid_data.link_motion_ang[i_l, i_b].cross(r)


# ============================================== Layer 1: Rigid block ===============================================


@qd.func(requires_top_level=True)
def func_rigid_begin(
    rigid_data: qd.template(),
    dofs_state: DofsState,
    constraint_state: ConstraintState,
    params: qd.template(),
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Start the Rigid unknown from Rigid's warm start and group Rigid's assembled rows by island.

    The start point matches that of Rigid's own solve. Every island is marked as iterating, so that the factor covers
    every dof, including those of islands without rows."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    dt = params.substep_dt[()]
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
    rigid_data: qd.template(),
    dyn_state: DynState,
    constraint_state: ConstraintState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    params: qd.template(),
    rigid_config: qd.template(),
    config: qd.template(),
):
    """Evaluate Rigid's cost of qacc = x / dt^2 into `rigid_data.cost`, with its row activity, forces and
    qfrc_constraint.

    The cost comes from Rigid's own update of its constraint state, and reduces to the Gauss term in a scene without
    Rigid constraint rows."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    inv_dt_sq = 1.0 / (params.substep_dt[()] * params.substep_dt[()])
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
def func_rigid_energy(
    rigid_data: qd.template(), control: qd.template(), params: qd.template(), is_trial: qd.template()
):
    """Add Rigid's cost of every environment times dt^4, after func_rigid_evaluate, to its start or trial energy."""
    n_envs = rigid_data.x.shape[1]
    dt_sq = params.substep_dt[()] * params.substep_dt[()]
    for i_b in range(n_envs):
        if is_trial:
            control.energy_trial[i_b] += dt_sq * dt_sq * rigid_data.cost[i_b]
        else:
            control.energy_start[i_b] += dt_sq * dt_sq * rigid_data.cost[i_b]


@qd.func(requires_top_level=True)
def func_rigid_gradient(
    rigid_data: qd.template(),
    dyn_state: DynState,
    constraint_state: ConstraintState,
    params: qd.template(),
    config: qd.template(),
):
    """Compute the gradient of Rigid's share of the joint energy in x, after func_rigid_evaluate at the current x.

    The gradient is dt^2 (M qacc - qf_smooth - qfrc_constraint)."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    dt_sq = params.substep_dt[()] * params.substep_dt[()]
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
    rigid_data: qd.template(),
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
            if rigid_data.link_K[i_l, i_b].norm_sqr() > 0.0 and n_links < rigid_data.contact_links.shape[1]:
                rigid_data.contact_links[i_b, n_links] = i_l
                n_links += 1
        rigid_data.n_links_in_contact[i_b] = n_links
        n_cols = 6 * n_links
        for i_col in range(n_cols):
            i_l = rigid_data.contact_links[i_b, i_col // 6]
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
                rigid_data.woodbury_U[i_d, i_b, i_col] = rigid_data.solve_rhs[i_d, i_b]
                rigid_data.woodbury_W[i_d, i_b, i_col] = rigid_data.solve_out[i_d, i_b]
        # S = I + K U^T W, the rows of a link's block from its K_l and the rows of U^T W of its columns
        for b in range(n_cols):
            for j in range(n_links):
                g = qd.Vector.zero(gs.qd_float, 6)
                for k in qd.static(range(6)):
                    value = gs.qd_float(0.0)
                    for i_d in range(n_dofs):
                        value += rigid_data.woodbury_U[i_d, i_b, 6 * j + k] * rigid_data.woodbury_W[i_d, i_b, b]
                    g[k] = value
                i_l = rigid_data.contact_links[i_b, j]
                K_g = rigid_data.link_K[i_l, i_b] @ g
                for i in qd.static(range(6)):
                    value = K_g[i]
                    if 6 * j + i == b:
                        value += 1.0
                    rigid_data.capacitance_LU[i_b, 6 * j + i, b] = value
        # In-place lower-upper (LU) factorization with partial pivoting. S is non-singular, K and U^T W being positive
        # semi-definite
        for k in range(n_cols):
            i_max = k
            for i in range(k + 1, n_cols):
                if qd.abs(rigid_data.capacitance_LU[i_b, i, k]) > qd.abs(rigid_data.capacitance_LU[i_b, i_max, k]):
                    i_max = i
            rigid_data.capacitance_pivots[i_b, k] = i_max
            if i_max != k:
                for j in range(n_cols):
                    entry = rigid_data.capacitance_LU[i_b, k, j]
                    rigid_data.capacitance_LU[i_b, k, j] = rigid_data.capacitance_LU[i_b, i_max, j]
                    rigid_data.capacitance_LU[i_b, i_max, j] = entry
            for i in range(k + 1, n_cols):
                rigid_data.capacitance_LU[i_b, i, k] /= rigid_data.capacitance_LU[i_b, k, k]
                factor = rigid_data.capacitance_LU[i_b, i, k]
                for j in range(k + 1, n_cols):
                    rigid_data.capacitance_LU[i_b, i, j] -= factor * rigid_data.capacitance_LU[i_b, k, j]


@qd.func(requires_top_level=True)
def func_rigid_contact_gradient(
    f: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    rigid_data: qd.template(),
    dyn_state: DynState,
    dyn_info: DynInfo,
    params: qd.template(),
    config: qd.template(),
):
    """Add the share of the Rigid contact rows to the gradient in x, and sum the spatial contact stiffness of each link.

    Runs after func_rigid_gradient and after the MPM assembly that staged the row gradients and Hessians. The gradient
    takes minus the generalized force of the rows of the particles on their links. The spatial stiffness K_l = sum_s
    P_s^T H_s P_s of each link feeds the preconditioner, with P_s = [I, -[r_s]x] mapping the spatial displacement of the
    link to that of the contact point at arm r_s."""
    func_link_spatial_disp(rigid_data.x, rigid_data, dyn_state.dofs, dyn_info.links, config)
    qd.loop_config(name="mpm_rigid_link_K_clear")
    for i_l, i_b in qd.ndrange(rigid_data.link_K.shape[0], rigid_data.link_K.shape[1]):
        rigid_data.link_K[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)
    qd.loop_config(name="mpm_rigid_row_forces")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            g = newton.g_rigid[s]
            H = newton.h_rigid[s]
            if g.norm_sqr() > 0.0 or H.norm_sqr() > 0.0:
                r = particle_arm(f, i_p, i_b, i_l, particles, dyn_state.links)
                qd.atomic_add(rigid_data.link_force_vel[i_l, i_b], g)
                qd.atomic_add(rigid_data.link_force_ang[i_l, i_b], r.cross(g))
                R = gu.qd_vec_to_skew(r)
                P = qd.Matrix.zero(gs.qd_float, 3, 6)
                for i in qd.static(range(3)):
                    P[i, i] = 1.0
                    for j in qd.static(range(3)):
                        P[i, 3 + j] = -R[i, j]
                qd.atomic_add(rigid_data.link_K[i_l, i_b], P.transpose() @ H @ P)
    func_link_force_to_dofs(-1.0, rigid_data.grad, rigid_data, dyn_state.dofs, dyn_info.links, config)


@qd.func(requires_top_level=True)
def func_rigid_precondition(
    rigid_data: qd.template(),
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
                value += rigid_data.woodbury_U[i_d, i_b, a] * rigid_data.z[i_d, i_b]
            rigid_data.woodbury_y[i_b, a] = value
        for j in range(n_links):
            i_l = rigid_data.contact_links[i_b, j]
            y = qd.Vector([rigid_data.woodbury_y[i_b, 6 * j + k] for k in qd.static(range(6))], dt=gs.qd_float)
            K_y = rigid_data.link_K[i_l, i_b] @ y
            for i in qd.static(range(6)):
                rigid_data.woodbury_y[i_b, 6 * j + i] = K_y[i]
        # S q = K U^T z0 by the lower-upper (LU) factor: the row pivots, then the unit lower and upper triangular solves
        for k in range(n_cols):
            i_p = rigid_data.capacitance_pivots[i_b, k]
            if i_p != k:
                entry = rigid_data.woodbury_y[i_b, k]
                rigid_data.woodbury_y[i_b, k] = rigid_data.woodbury_y[i_b, i_p]
                rigid_data.woodbury_y[i_b, i_p] = entry
        for i in range(n_cols):
            value = rigid_data.woodbury_y[i_b, i]
            for k in range(i):
                value -= rigid_data.capacitance_LU[i_b, i, k] * rigid_data.woodbury_y[i_b, k]
            rigid_data.woodbury_y[i_b, i] = value
        for i_rev in range(n_cols):
            i = n_cols - 1 - i_rev
            value = rigid_data.woodbury_y[i_b, i]
            for k in range(i + 1, n_cols):
                value -= rigid_data.capacitance_LU[i_b, i, k] * rigid_data.woodbury_y[i_b, k]
            rigid_data.woodbury_y[i_b, i] = value / rigid_data.capacitance_LU[i_b, i, i]
        for i_d in range(n_dofs):
            value = gs.qd_float(0.0)
            for b in range(n_cols):
                value += rigid_data.woodbury_W[i_d, i_b, b] * rigid_data.woodbury_y[i_b, b]
            rigid_data.z[i_d, i_b] -= value


# ========================================= Layer 1: joint PCG over (u, x) ==========================================


@qd.func(requires_top_level=True)
def func_rigid_pcg_init(rigid_data: qd.template(), pcg_scalars: qd.template(), is_adjoint: qd.template()):
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
def func_rigid_pcg_init_direction(rigid_data: qd.template(), pcg_scalars: qd.template()):
    """Set the first search direction p = z over the Rigid dofs, and add r . z to rz."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_init_direction")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.p[i_d, i_b] = rigid_data.z[i_d, i_b]
        qd.atomic_add(pcg_scalars.rz[i_b], rigid_data.r[i_d, i_b] * rigid_data.z[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_contact_matvec(
    f: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    rigid_data: qd.template(),
    pcg_scalars: qd.template(),
    dyn_state: DynState,
    dyn_info: DynInfo,
    params: qd.template(),
    config: qd.template(),
):
    """Apply the Rigid contact rows' share of the joint operator, after the own operators of MPM and Rigid set their Ap.

    The contact point of a particle moves by dr of p_r along the direction. The rows add their cross term on the MPM
    rows (see contact_cross_matvec), and the spatial force (h, r x h) to the link in `link_force_vel` /
    `link_force_ang`, which func_rigid_contact_matvec_post hands to Ap_r."""
    func_link_spatial_disp(rigid_data.p, rigid_data, dyn_state.dofs, dyn_info.links, config)
    qd.loop_config(name="mpm_rigid_matvec_particles")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
        if i_l >= 0 and pcg_scalars.is_converged[i_b] == 0:
            r = particle_arm(f, i_p, i_b, i_l, particles, dyn_state.links)
            dr = rigid_data.link_motion_vel[i_l, i_b] + rigid_data.link_motion_ang[i_l, i_b].cross(r)
            h = contact_cross_matvec(f, s, dr, particles, grid, newton, linear, pcg_scalars, params)
            qd.atomic_add(rigid_data.link_force_vel[i_l, i_b], h)
            qd.atomic_add(rigid_data.link_force_ang[i_l, i_b], r.cross(h))


@qd.func(requires_top_level=True)
def func_rigid_matvec(
    rigid_data: qd.template(),
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
def func_rigid_contact_matvec_post(
    rigid_data: qd.template(), dyn_state: DynState, dyn_info: DynInfo, config: qd.template()
):
    """Add to Ap_r the generalized force of the spatial forces accumulated on the links by func_rigid_contact_matvec."""
    func_link_force_to_dofs(1.0, rigid_data.ap, rigid_data, dyn_state.dofs, dyn_info.links, config)


@qd.func(requires_top_level=True)
def func_rigid_matvec_dot(rigid_data: qd.template(), pcg_scalars: qd.template()):
    """Add p_r . Ap_r to p . Ap, once Ap_r holds the whole joint operator."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_matvec_dot")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[i_b], rigid_data.p[i_d, i_b] * rigid_data.ap[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_pcg_update_xr(rigid_data: qd.template(), pcg_scalars: qd.template()):
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
def func_rigid_pcg_dot_rz(rigid_data: qd.template(), pcg_scalars: qd.template()):
    """Add r . z over the Rigid dofs to rz_new."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_pcg_dot_rz")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.rz_new[i_b], rigid_data.r[i_d, i_b] * rigid_data.z[i_d, i_b])


@qd.func(requires_top_level=True)
def func_rigid_pcg_direction(rigid_data: qd.template(), pcg_scalars: qd.template()):
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
def func_rigid_start_line_search(rigid_data: qd.template(), control: qd.template()):
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
def func_rigid_step_forward(rigid_data: qd.template(), control: qd.template()):
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_rigid_step_forward")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.x[i_d, i_b] = rigid_data.x_temp[i_d, i_b] + control.alpha[i_b] * rigid_data.dx[i_d, i_b]


@qd.func(requires_top_level=True)
def func_rigid_write_back(rigid_data: qd.template(), constraint_state: ConstraintState, config: qd.template()):
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
def func_rigid_adjoint_rhs(rigid_data: qd.template(), constraint_state: ConstraintState, params: qd.template()):
    """Set Rigid's right-hand side of the joint adjoint solve, in place of the gradient from which the solve starts.

    The right-hand side is the adjoint of x = dt^2 qacc, dL/dqacc / dt^2, with dL/dqacc as taken by Rigid's own adjoint
    solve (see ConstraintSolver.backward)."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    inv_dt_sq = 1.0 / (params.substep_dt[()] * params.substep_dt[()])
    qd.loop_config(name="mpm_rigid_adjoint_rhs")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        rigid_data.grad[i_d, i_b] = constraint_state.dL_dqacc[i_d, i_b] * inv_dt_sq


@qd.func(requires_top_level=True)
def func_rigid_adjoint_finish(rigid_data: qd.template(), constraint_state: ConstraintState, params: qd.template()):
    """Set the solution of Rigid's adjoint solve from that of the joint one.

    Rigid's share of the joint energy is dt^4 times its cost of x / dt^2, whose gradient in the parameters of Rigid's
    rows is dt^2 times that of its cost, so u = dt^2 lambda_x."""
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    dt_sq = params.substep_dt[()] * params.substep_dt[()]
    qd.loop_config(name="mpm_rigid_adjoint_finish")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.bw_u[i_d, i_b] = dt_sq * rigid_data.sol[i_d, i_b]


@qd.func(requires_top_level=True)
def func_mpm_reaction(
    f: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    rigid_data: qd.template(),
    dyn_state: DynState,
    constraint_state: ConstraintState,
    params: qd.template(),
):
    """Apply the force of the MPM contact rows on Rigid at the converged solve, after func_rigid_write_back.

    qfrc_constraint becomes M qacc - qf_smooth, Rigid's own and the MPM rows' generalized force together, so that
    Rigid's dofs.force is M qacc of the converged qacc as after its own solve. The rows' spatial force on each link
    stays in `link_force_vel` / `link_force_ang` (in units of dt^2 force) for func_add_mpm_contact_force: the contact
    force of a link is not recoverable from the generalized force of a chain.

    A link pushed by the rows is marked constrained, as Rigid's assembly marks the links of its own contacts, since the
    midpoint integration of a free link (see func_midpoint_eligible) replaces qacc by the solution of the unconstrained
    equation of motion."""
    n_links = rigid_data.link_force_vel.shape[0]
    n_envs = rigid_data.link_force_vel.shape[1]
    qd.loop_config(name="mpm_reaction_clear")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        rigid_data.link_force_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
        rigid_data.link_force_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
    qd.loop_config(name="mpm_reaction_rows")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            slots = stencil_node_slots(env, cell, grid)
            dx, w = particle_trial_disp(f, i_p, i_b, cell, slots, particles, newton, params)
            dr = newton.contact_dr[s]
            energy, g, hess = contact_row_terms(s, dx - dr, newton, WALL_ROWS, CONTACT_ROWS)
            if g.norm_sqr() > 0.0:
                r = particle_arm(f, i_p, i_b, i_l, particles, dyn_state.links)
                qd.atomic_add(rigid_data.link_force_vel[i_l, i_b], g)
                qd.atomic_add(rigid_data.link_force_ang[i_l, i_b], r.cross(g))
                dyn_state.links.is_constrained[i_l, i_b] = True
    n_dofs = rigid_data.x.shape[0]
    qd.loop_config(name="mpm_reaction_qfrc")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.qfrc_constraint[i_d, i_b] = rigid_data.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b]


@qd.func(requires_top_level=True)
def func_add_mpm_contact_force(rigid_data: qd.template(), links_state: LinksState, params: qd.template()):
    """Add the MPM rows' force on each link (func_mpm_reaction) to `links_state.contact_force`, after Rigid's own
    contact forces were written by the post-solve stage of its substep."""
    n_links = rigid_data.link_force_vel.shape[0]
    n_envs = rigid_data.link_force_vel.shape[1]
    dt = params.substep_dt[()]
    qd.loop_config(name="mpm_add_contact_force")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        force = rigid_data.link_force_vel[i_l, i_b]
        links_state.contact_force[i_l, i_b] = links_state.contact_force[i_l, i_b] + force / (dt * dt)
