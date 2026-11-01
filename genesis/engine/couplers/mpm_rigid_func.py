"""MPM-Rigid coupling stages of the implicit MPM solver: one Newton solve over the MPM grid displacement and the Rigid
dofs.

Rigid geoms enter the MPM energy as soft contact rows with Rigid's own contact model: the deepest geom of a particle
within its soft range gives the pyramid rows of its friction cone, with the geom's friction and solref / solimp
impedance, and the summed inverse weight of the link and of the particle in the implicit system (the material it is
embedded in, see `func_contact_invweight`). A row acts on the displacement of the particle relative to its contact point on
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

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import (
    CONTACT_ROWS,
    PYRAMID_ROWS,
    WALL_ROWS,
    MpmParticles,
    MpmParticlesInfo,
)
from genesis.engine.solvers.mpm_func import (
    band_row,
    contact_band_width,
    contact_invweight,
    contact_row_terms,
    particle_of_sorted,
    particle_trial_disp,
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

# Half extent of the inverted box of a leaf that must meet no query, finite so that its center gets a morton code
EMPTY_BOX_HALF = 1e10


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


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class RigidGeomQuery:
    """The LBVH over the coupled Rigid geoms of every environment (see genesis.engine.bvh), and what the filter of its
    queries reads: the geoms, and the query point and nearest geom of each sorted particle in `newton`. A query that is
    not active holds no geom and builds nothing."""

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
        self.results = array_class.get_bvh_query_results(1)
        self.geoms_state = rigid.dyn_state.geoms
        self.geoms_info = rigid.dyn_info.geoms
        self.geoms_init_AABB = rigid.geoms_init_AABB
        self.links_state = rigid.dyn_state.links
        self.links_info = rigid.dyn_info.links
        self.batch_links_info = rigid._options.batch_links_info
        self.sdf_info = rigid.collider.sdf.sdf_info
        self.newton = newton

    def build(self, params):
        """Fit the leaf boxes to the geoms of the substep and build the trees."""
        if self.n_coupled:
            kernel_fit_geom_leaves(self, params)
            build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)


@qd.func(requires_top_level=True)
def func_fit_geom_leaves(
    query: qd.template(),  # RigidGeomQuery
    dt,
):
    """The box of every coupled geom, the world box of its posed initial box (as `func_update_geom_aabbs` fits it,
    which Rigid skips without collisions), grown by the geom's contact margin and by twice the distance its points may
    travel over the substep, bounded by the link velocity at the box center and the spin over its radius. A geom an
    environment does not carry gets an empty box with a finite center, for the morton code."""
    qd.loop_config(name="mpm_fit_geom_leaves")
    for i_b, i_a in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        i_g = query.coupled_geoms[i_a]
        i_l = query.geoms_info.link_idx[i_g]
        is_carried = True
        if qd.static(query.batch_links_info):
            is_carried = query.links_info.geom_start[i_l, i_b] <= i_g < query.links_info.geom_end[i_l, i_b]
        if is_carried:
            g_pos = query.geoms_state.pos[i_g, i_b]
            g_quat = query.geoms_state.quat[i_g, i_b]
            lo = gu.qd_vec3(qd.math.inf)
            hi = gu.qd_vec3(-qd.math.inf)
            for i_corner in qd.static(range(8)):
                corner = gu.qd_transform_by_trans_quat(query.geoms_init_AABB[i_g, i_corner], g_pos, g_quat)
                lo = qd.min(lo, corner)
                hi = qd.max(hi, corner)
            center = 0.5 * (lo + hi)
            radius = 0.5 * (hi - lo).norm()
            speed = func_vel_at_point(i_l, i_b, center, query.links_state).norm()
            speed += query.links_state.cd_ang[i_l, i_b].norm() * radius
            margin = qd.max(query.geoms_info.coup_softness[i_g], 0.0) + 2.0 * dt * speed
            query.leaves.aabbs_min[i_b, i_a] = lo - margin
            query.leaves.aabbs_max[i_b, i_a] = hi + margin
        else:
            query.leaves.aabbs_min[i_b, i_a] = qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
            query.leaves.aabbs_max[i_b, i_a] = qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)


@qd.kernel
def kernel_fit_geom_leaves(query: qd.template(), params: qd.template()):
    func_fit_geom_leaves(query, params.substep_dt[()])


@qd.func
def func_nearest_geom_filter(i_t: int, i_a: int, i_q: int, query: qd.template()) -> bool:
    """The narrow phase of a box query of sorted particle i_q in environment i_t: keep the coupled geom of the nearest
    contact range in `newton.query_phi` / `query_idx`, and record no pair."""
    i_g = query.coupled_geoms[i_a]
    x = qd.Vector([query.newton.query_pos[3 * i_q + d] for d in qd.static(range(3))], dt=gs.qd_float)
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
    particles: MpmParticles,
    particles_info: MpmParticlesInfo,
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
    query: qd.template(),  # RigidGeomQuery, its trees built for the substep
    f,
    dt,
):
    """Pyramid contact rows of every valid particle within the soft range of a coupled geom, after
    `func_build_wall_rows` reset the rows. The nearest geom is found by a box query of the coupled geoms' trees.

    The geom's `coup_softness` is a contact margin: a row starts at that distance from the surface, and its constraint
    position is the signed distance minus the margin, so the resting penetration of the soft row is taken up outside
    the geom. As in Rigid's contact assembly, the friction coefficient is `max(friction * friction_ratio, 0.01)`, the
    timeconst is floored at twice the substep, and every row `normal + mu d_k` of the pyramid gets
    `diag = 2 mu_reg^2 (1 + mu_reg^2) invweight (1 - imp) / imp`, with `mu_reg^2 = mu^2 / impratio`. The row target
    `dt J.v_p + dt^2 aref` is on the displacement of the particle relative to the motion J_pt x of its contact point
    beyond the link's velocity at the substep start. The nearest geom gives rows to a particle within its range or short
    of it by less than `contact_band_width`, by `band_row`.
    """
    qd.loop_config(name="mpm_build_rigid_rows")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
        x_p = particles.pos[f, i_p, i_b]
        # The nearest coupled geom, by its gap to the contact range, among those whose grown box meets the particle's
        # band: the two grown boxes together cover the band of the relative velocity
        for d in qd.static(range(3)):
            newton.query_pos[3 * s + d] = x_p[d]
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
            is_candidate = phi < contact_band_width(params, v_rel.norm(), dt)
        normal = qd.Vector.zero(gs.qd_float, 3)
        in_range = False
        if is_candidate:
            normal = geom_normal(
                i_g_contact, i_b, x_p, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
            )
            in_range = phi < contact_band_width(params, -normal.dot(v_rel), dt)
        if in_range:
            I_l = [i_l, i_b] if qd.static(config.rigid_batch_links_info) else i_l
            friction = qd.max(geoms_info.friction[i_g_contact] * geoms_state.friction_ratio[i_g_contact, i_b], 1e-2)
            friction_sq_reg = friction * friction / rigid_info.impratio[None]
            invweight = contact_invweight(particles, particles_info, f, i_p, i_b, normal) + links_info.invweight[I_l][0]
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
):
    """Displacement J_pt x of the contact point of every particle with Rigid rows, for the dof vector x whose link
    displacements `link_V` holds. The other particles keep theirs, from `func_reset_contact_rows` or from the
    coupling that gave them rows."""
    qd.loop_config(name="mpm_contact_point_disp")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            lin, ang = link_spatial(coupling, i_l, i_b)
            dr = lin + ang.cross(particle_arm(particles, links_state, f, i_p, i_b, i_l))
            for d in qd.static(range(3)):
                newton.contact_dr[3 * s + d] = dr[d]


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
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    config: qd.template(),  # MpmStaticConfig
    params: qd.template(),  # MpmParams
):
    """Gradient of Rigid's share of the joint energy in x, after func_rigid_evaluate at the current x:
    dt^2 (M qacc - qf_smooth - qfrc_constraint)."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    dt_sq = params.substep_dt[()] * params.substep_dt[()]
    qd.loop_config(name="mpm_rigid_gradient")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        value = coupling.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b]
        if qd.static(config.has_rigid_rows):
            value -= constraint_state.qfrc_constraint[i_d, i_b]
        coupling.grad[i_d, i_b] = dt_sq * value


@qd.func
def rigid_solve(
    i_b,
    rhs: qd.template(),  # qd.Ndarray (n_dofs, n_envs)
    out: qd.template(),  # qd.Ndarray (n_dofs, n_envs)
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
):
    """out = A^-1 rhs in environment i_b, A = M + J^T D J by Rigid's island Cholesky factor, or A = M by Rigid's mass
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
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
):
    """The preconditioner of the Rigid block, (A + U K U^T)^-1 with A = M + J^T D J and the MPM contact stiffness
    U K U^T of the links in contact, after func_rigid_contact_gradient summed their K_l.

    Rigid's island factor of A at the current x, then the Woodbury identity on it: per environment, U holds one column
    per spatial component of each link in contact, the cdofs of its chain, W = A^-1 U, and S = I + K U^T W is factored
    by LU, its order 6 times the links in contact. Without the contact stiffness, the rows of the many particles on a
    link, far stiffer than its inertia, leave the joint PCG stopping on a direction that barely moves the link."""
    if qd.static(config.has_rigid_rows):
        func_hessian_and_cholesky_factor_direct(
            constraint_state, dyn_info, rigid_info, rigid_config, compute_envelope=True
        )
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_contact_factor")
    for i_b in range(n_envs):
        n_links = 0
        for i_l in range(coupling.link_K.shape[0]):
            k_sum = gs.qd_float(0.0)
            for k in range(36):
                k_sum += qd.abs(coupling.link_K[i_l, i_b, k])
            if k_sum > 0.0 and n_links < qd.static(coupling.n_contact_links):
                coupling.contact_links[i_b, n_links] = i_l
                n_links += 1
        coupling.n_links_in_contact[i_b] = n_links
        n_cols = 6 * n_links
        for col in range(n_cols):
            i_l = coupling.contact_links[i_b, col // 6]
            c = col % 6
            for i_d in range(n_dofs):
                coupling.solve_rhs[i_d, i_b] = 0.0
            i_a = i_l
            while i_a >= 0:
                I_a = [i_a, i_b] if qd.static(config.rigid_batch_links_info) else i_a
                for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
                    value = gs.qd_float(0.0)
                    for e in qd.static(range(3)):
                        if c == e:
                            value = dyn_state.dofs.cdof_vel[i_d, i_b][e]
                        if c == 3 + e:
                            value = dyn_state.dofs.cdof_ang[i_d, i_b][e]
                    coupling.solve_rhs[i_d, i_b] = value
                i_a = dyn_info.links.parent_idx[I_a]
            rigid_solve(
                i_b,
                coupling.solve_rhs,
                coupling.solve_out,
                dyn_state,
                constraint_state,
                rigid_info,
                rigid_config,
                config,
            )
            for i_d in range(n_dofs):
                coupling.wb_U[i_d, i_b, col] = coupling.solve_rhs[i_d, i_b]
                coupling.wb_W[i_d, i_b, col] = coupling.solve_out[i_d, i_b]
        # S = I + K U^T W, the rows of a link's block from its K_l and the rows of U^T W of its columns
        for b in range(n_cols):
            for j in range(n_links):
                g = qd.Vector.zero(gs.qd_float, 6)
                for k in qd.static(range(6)):
                    value = gs.qd_float(0.0)
                    for i_d in range(n_dofs):
                        value += coupling.wb_U[i_d, i_b, 6 * j + k] * coupling.wb_W[i_d, i_b, b]
                    g[k] = value
                i_l = coupling.contact_links[i_b, j]
                for i in qd.static(range(6)):
                    value = gs.qd_float(1.0) if 6 * j + i == b else gs.qd_float(0.0)
                    for k in qd.static(range(6)):
                        value += coupling.link_K[i_l, i_b, 6 * i + k] * g[k]
                    coupling.wb_S[i_b, 6 * j + i, b] = value
        # LU with partial pivoting, in place; S is non-singular, K and U^T W being positive semi-definite
        for k in range(n_cols):
            i_max = k
            for i in range(k + 1, n_cols):
                if qd.abs(coupling.wb_S[i_b, i, k]) > qd.abs(coupling.wb_S[i_b, i_max, k]):
                    i_max = i
            coupling.wb_pivot[i_b, k] = i_max
            if i_max != k:
                for jj in range(n_cols):
                    tmp = coupling.wb_S[i_b, k, jj]
                    coupling.wb_S[i_b, k, jj] = coupling.wb_S[i_b, i_max, jj]
                    coupling.wb_S[i_b, i_max, jj] = tmp
            for i in range(k + 1, n_cols):
                coupling.wb_S[i_b, i, k] = coupling.wb_S[i_b, i, k] / coupling.wb_S[i_b, k, k]
                for jj in range(k + 1, n_cols):
                    coupling.wb_S[i_b, i, jj] -= coupling.wb_S[i_b, i, k] * coupling.wb_S[i_b, k, jj]


@qd.func(requires_top_level=True)
def func_rigid_contact_gradient(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    config: qd.template(),  # MpmStaticConfig
    params: qd.template(),  # MpmParams
    f,
):
    """The Rigid contact rows' share of the gradient in x, after func_rigid_gradient and the MPM assembly that staged
    the row gradients and Hessians: minus the generalized force of the rows of the particles on their links, and the
    spatial stiffness K_l = sum_s P_s^T H_s P_s of each link for the preconditioner, P_s = [I, -[r_s]x] mapping the
    link's spatial displacement to that of the contact point at arm r_s."""
    func_link_spatial_disp(coupling, coupling.x, dyn_info.links, dyn_state.dofs, config)
    qd.loop_config(name="mpm_rigid_link_K_clear")
    for i_l, i_b, k in qd.ndrange(coupling.link_K.shape[0], coupling.link_K.shape[1], 36):
        coupling.link_K[i_l, i_b, k] = 0.0
    qd.loop_config(name="mpm_rigid_row_forces")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            g = qd.Vector([newton.g_rigid[3 * s + d] for d in qd.static(range(3))], dt=gs.qd_float)
            H = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i, j in qd.static(qd.ndrange(3, 3)):
                H[i, j] = newton.h_rigid[9 * s + 3 * i + j]
            if g.norm_sqr() > 0.0 or H.norm_sqr() > 0.0:
                r = particle_arm(particles, dyn_state.links, f, i_p, i_b, i_l)
                torque = r.cross(g)
                for d in qd.static(range(3)):
                    qd.atomic_add(coupling.link_F[i_l, i_b, d], g[d])
                    qd.atomic_add(coupling.link_F[i_l, i_b, 3 + d], torque[d])
                R = qd.Matrix([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]], dt=gs.qd_float)
                HR = H @ R
                RH = R @ H
                RHR = RH @ R
                for i, j in qd.static(qd.ndrange(3, 3)):
                    qd.atomic_add(coupling.link_K[i_l, i_b, 6 * i + j], H[i, j])
                    qd.atomic_add(coupling.link_K[i_l, i_b, 6 * i + 3 + j], -HR[i, j])
                    qd.atomic_add(coupling.link_K[i_l, i_b, 6 * (3 + i) + j], RH[i, j])
                    qd.atomic_add(coupling.link_K[i_l, i_b, 6 * (3 + i) + 3 + j], -RHR[i, j])
    func_link_force_to_dofs(coupling, coupling.grad, -1.0, dyn_info.links, dyn_state.dofs, config)


@qd.func(requires_top_level=True)
def func_rigid_precondition(
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
):
    """z = (A + U K U^T)^-1 r over the Rigid dofs, with the factors of func_rigid_factor: z0 = A^-1 r, then
    z = z0 - W S^-1 K U^T z0 by the Woodbury identity."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    qd.loop_config(name="mpm_rigid_precondition")
    for i_b in range(n_envs):
        rigid_solve(i_b, coupling.r, coupling.z, dyn_state, constraint_state, rigid_info, rigid_config, config)
        n_links = coupling.n_links_in_contact[i_b]
        n_cols = 6 * n_links
        for a in range(n_cols):
            value = gs.qd_float(0.0)
            for i_d in range(n_dofs):
                value += coupling.wb_U[i_d, i_b, a] * coupling.z[i_d, i_b]
            coupling.wb_y[i_b, a] = value
        for j in range(n_links):
            i_l = coupling.contact_links[i_b, j]
            y = qd.Vector([coupling.wb_y[i_b, 6 * j + k] for k in qd.static(range(6))], dt=gs.qd_float)
            for i in qd.static(range(6)):
                value = gs.qd_float(0.0)
                for k in qd.static(range(6)):
                    value += coupling.link_K[i_l, i_b, 6 * i + k] * y[k]
                coupling.wb_y[i_b, 6 * j + i] = value
        # S q = K U^T z0 by the LU factor: the row pivots, then the unit lower and the upper triangular solves
        for k in range(n_cols):
            i_p = coupling.wb_pivot[i_b, k]
            if i_p != k:
                tmp = coupling.wb_y[i_b, k]
                coupling.wb_y[i_b, k] = coupling.wb_y[i_b, i_p]
                coupling.wb_y[i_b, i_p] = tmp
        for i in range(n_cols):
            value = coupling.wb_y[i_b, i]
            for k in range(i):
                value -= coupling.wb_S[i_b, i, k] * coupling.wb_y[i_b, k]
            coupling.wb_y[i_b, i] = value
        for i_rev in range(n_cols):
            i = n_cols - 1 - i_rev
            value = coupling.wb_y[i_b, i]
            for k in range(i + 1, n_cols):
                value -= coupling.wb_S[i_b, i, k] * coupling.wb_y[i_b, k]
            coupling.wb_y[i_b, i] = value / coupling.wb_S[i_b, i, i]
        for i_d in range(n_dofs):
            value = gs.qd_float(0.0)
            for b in range(n_cols):
                value += coupling.wb_W[i_d, i_b, b] * coupling.wb_y[i_b, b]
            coupling.z[i_d, i_b] -= value


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
def func_rigid_contact_matvec(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    config: qd.template(),  # MpmStaticConfig
    params: qd.template(),  # MpmParams
    f,
):
    """The Rigid contact rows' share of the joint operator beyond the MPM matrix, after func_linear_pcg_spmv, on
    p = (p_u, p_r), and before func_rigid_matvec overwrites Ap_r.

    A particle with Rigid rows of Hessian H, trial displacement dxp = sum_a w_a p_u,a and contact point displacement dr of
    p_r adds -w_a H dr to Ap_u,a and the spatial force (h, r x h), h = H (dr - dxp), to its link in `link_F`, which
    func_rigid_contact_matvec_post hands to Ap_r. p.Ap takes p_u.(-sum w_a H dr).
    """
    ws = linear.pcg
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


@qd.func(requires_top_level=True)
def func_rigid_matvec(
    coupling: qd.template(),  # MpmRigidCoupling
    constraint_state: array_class.ConstraintState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    config: qd.template(),  # MpmStaticConfig
):
    """Ap_r = (M + J^T D J) p_r over Rigid's active rows."""
    n_envs = coupling.x.shape[1]
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


@qd.func(requires_top_level=True)
def func_rigid_contact_matvec_post(
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    config: qd.template(),  # MpmStaticConfig
):
    """Ap_r += the generalized force of the spatial forces func_rigid_contact_matvec accumulated on the links."""
    func_link_force_to_dofs(coupling, coupling.ap, 1.0, dyn_info.links, dyn_state.dofs, config)


@qd.func(requires_top_level=True)
def func_rigid_matvec_dot(
    coupling: qd.template(),  # MpmRigidCoupling
    ws: qd.template(),  # PCGWorkspace
):
    """p.Ap += p_r.Ap_r, once Ap_r holds the whole joint operator."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
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


@qd.func(requires_top_level=True)
def func_rigid_adjoint_rhs(
    coupling: qd.template(),  # MpmRigidCoupling
    constraint_state: array_class.ConstraintState,
    params: qd.template(),  # MpmParams
):
    """Rigid's right-hand side of the joint adjoint solve, in place of the gradient the PCG starts from: the adjoint
    of x = dt^2 qacc, dL/dqacc / dt^2, with dL/dqacc as Rigid's own adjoint solve takes it (see
    ConstraintSolver.backward)."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    inv_dt_sq = 1.0 / (params.substep_dt[()] * params.substep_dt[()])
    qd.loop_config(name="mpm_rigid_adjoint_rhs")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        coupling.grad[i_d, i_b] = constraint_state.dL_dqacc[i_d, i_b] * inv_dt_sq


@qd.func(requires_top_level=True)
def func_rigid_adjoint_finish(
    coupling: qd.template(),  # MpmRigidCoupling
    constraint_state: array_class.ConstraintState,
    params: qd.template(),  # MpmParams
):
    """The solution of Rigid's adjoint solve from the joint one: Rigid's joint energy is dt^4 times its cost of
    x / dt^2, whose gradient in the parameters of Rigid's rows is dt^2 times that of its cost, so u = dt^2 lambda_x."""
    n_dofs = coupling.x.shape[0]
    n_envs = coupling.x.shape[1]
    dt_sq = params.substep_dt[()] * params.substep_dt[()]
    qd.loop_config(name="mpm_rigid_adjoint_finish")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.bw_u[i_d, i_b] = dt_sq * coupling.sol[i_d, i_b]


@qd.func(requires_top_level=True)
def func_mpm_reaction(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    coupling: qd.template(),  # MpmRigidCoupling
    dyn_state: array_class.DynState,
    constraint_state: array_class.ConstraintState,
    params: qd.template(),  # MpmParams
    f,
):
    """The force of the MPM contact rows on Rigid at the converged solve, after func_rigid_write_back.

    qfrc_constraint becomes M qacc - qf_smooth, Rigid's own and the MPM rows' generalized force together, so that
    Rigid's dofs.force is M qacc of the converged qacc as after its own solve. The rows' spatial force on each link
    stays in `link_F` (in units of dt^2 force) for func_add_mpm_contact_force: a link's contact force is not recoverable
    from the generalized force of a chain.

    A link the rows push is marked constrained, as Rigid's assembly marks the links of its own contacts: the midpoint
    integration of a free body (see func_midpoint_is_eligible) replaces qacc by the solution of the unconstrained
    body equation."""
    n_links = coupling.link_V.shape[0]
    n_envs = coupling.link_V.shape[1]
    qd.loop_config(name="mpm_reaction_clear")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        for d in qd.static(range(6)):
            coupling.link_F[i_l, i_b, d] = 0.0
    qd.loop_config(name="mpm_reaction_rows")
    for s in range(grid.n_valid[()]):
        i_l = newton.contact_link[s]
        if i_l >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            slots = stencil_node_slots(grid, env, cell)
            dx, w = particle_trial_disp(particles, newton, params, f, i_p, i_b, cell, slots)
            dr = qd.Vector([newton.contact_dr[3 * s + d] for d in qd.static(range(3))], dt=gs.qd_float)
            energy, g, hess = contact_row_terms(newton, s, dx - dr, WALL_ROWS, CONTACT_ROWS)
            if g.norm_sqr() > 0.0:
                torque = particle_arm(particles, dyn_state.links, f, i_p, i_b, i_l).cross(g)
                for d in qd.static(range(3)):
                    qd.atomic_add(coupling.link_F[i_l, i_b, d], g[d])
                    qd.atomic_add(coupling.link_F[i_l, i_b, 3 + d], torque[d])
                dyn_state.links.is_constrained[i_l, i_b] = True
    n_dofs = coupling.x.shape[0]
    qd.loop_config(name="mpm_reaction_qfrc")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.qfrc_constraint[i_d, i_b] = coupling.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b]


@qd.func(requires_top_level=True)
def func_add_mpm_contact_force(
    coupling: qd.template(),  # MpmRigidCoupling
    links_state: array_class.LinksState,
    params: qd.template(),  # MpmParams
):
    """Add the MPM rows' force on each link (func_mpm_reaction) to `links_state.contact_force`, after Rigid's own
    contact forces were written by the post-solve stage of its substep."""
    n_links = coupling.link_V.shape[0]
    n_envs = coupling.link_V.shape[1]
    dt = params.substep_dt[()]
    qd.loop_config(name="mpm_add_contact_force")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        force = qd.Vector([coupling.link_F[i_l, i_b, d] for d in qd.static(range(3))], dt=gs.qd_float)
        links_state.contact_force[i_l, i_b] = links_state.contact_force[i_l, i_b] + force / (dt * dt)
