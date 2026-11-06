"""Contact pairs between the implicit material point method (MPM) and Rigid in the joint Newton solve over the grid
displacement and the Rigid dofs.

Rigid geoms enter the MPM energy as soft contact rows with Rigid's own contact model: the deepest geom of a particle
within its soft range gives a pair of the pyramid rows of its friction cone, with the geom's friction and solref /
solimp impedance, and the summed inverse weight of the link and of the particle in the implicit system (the material it
is embedded in, see `func_contact_invweight`). A row acts on the displacement of the particle relative to its contact
point on the link.

The Rigid unknown and its stages are those of the Rigid participant (see rigid/newton_func.py). The contact point of
a pair is a contact vertex of the Newton solve (see newton_contact.py), whose vertex map is the coupling: a particle at
arm r from the root center of mass (COM) of its link moves by dr = V_lin + V_ang x r for the link's spatial displacement
V = sum over the dofs of its chain of cdof x_d, and a vector y on the point pushes the dofs of that chain with the
spatial force (y, r x y). A Rigid without dofs gives contact points of no displacement, and its pairs act on the
particles alone.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, func_no_filter, get_bvh_data
from genesis.engine.solvers.mpm_data import PYRAMID_ROWS, MpmGridScratch, MpmInfo, MpmParticles, MpmParticlesInfo
from genesis.engine.solvers.mpm_func import (
    EMPTY_BOX_HALF,
    band_row,
    contact_band_width,
    contact_invweight,
    particle_of_sorted,
    pyramid_dirs,
    pyramid_terms,
)
from genesis.engine.solvers.newton_contact import SRC_ITERATE, append_pair, write_doublet, write_triplet
from genesis.engine.solvers.newton_linear import PAIRS
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.newton_func import RigidNewtonScratch
from genesis.engine.solvers.rigid.newton_participant import RigidNewtonParticipant
from genesis.utils.array_class import (
    ConstraintState,
    DynState,
    GeomsInfo,
    GeomsState,
    LinksInfo,
    LinksState,
    RigidInfo,
    SDFInfo,
)
from genesis.utils.data import Data, DataKind, field

from .contact_candidates import (
    ContactCandidates,
    func_report_candidates,
    func_reset_candidates,
    func_select_nearest,
    set_candidate_score,
)


class MpmRigidPairs(Data, kind=DataKind.SCRATCH):
    """The pairs of the particles with their nearest coupled Rigid geom (see func_build_rigid_pairs).

    A pair holds its sorted particle `s`, its environment `i_b`, its geom `geom` and the link `link` of the geom, the
    gap `phi` of the particle to the contact range of the geom at the substep start, and the rows of its friction
    pyramid around `normal` with friction coefficient `friction`, row i of stiffness `D[i]` and target `t[i]`. `n_pairs`
    counts the pairs asked for, above the capacity on overflow."""

    n_pairs: qd.Tensor = field(int, ndarray=True)
    s: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    i_b: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    geom: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    link: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    phi: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    normal: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    friction: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    D: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)
    t: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)


def build_rigid_pairs(max_pairs: int) -> MpmRigidPairs:
    """Build the MPM-Rigid pairs of a capacity, with no pair."""
    pairs = MpmRigidPairs(sizes={PAIRS: max_pairs})
    pairs.n_pairs.fill(0)
    return pairs


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

    The tree comes from genesis.engine.bvh. The query also holds the Rigid data its narrow phase reads, from the Rigid
    participant `rigid`."""

    def __init__(self, rigid: RigidNewtonParticipant, n_envs: int):
        self.n_coupled = len(rigid.coupled_geoms_idx)
        self.coupled_geoms = qd.ndarray(gs.qd_int, shape=(max(self.n_coupled, 1),))
        if self.n_coupled:
            self.coupled_geoms.from_numpy(rigid.coupled_geoms_idx)
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(self.n_coupled, 1), is_active=self.n_coupled > 0)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        self.geoms_state = rigid.dyn_state.geoms
        self.geoms_info = rigid.dyn_info.geoms
        self.geoms_init_AABB = rigid.geoms_init_AABB
        self.links_state = rigid.dyn_state.links
        self.links_info = rigid.dyn_info.links
        self.batch_links_info = rigid.config.rigid_batch_links_info
        self.sdf_info = rigid.sdf_info

    def build(self, mpm_info):
        """Fit the leaf boxes to the geoms of the substep and build the trees."""
        if self.n_coupled:
            kernel_fit_geom_leaves(self, mpm_info)
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
def kernel_fit_geom_leaves(query: qd.template(), mpm_info: MpmInfo):
    func_fit_geom_leaves(mpm_info.substep_dt[()], query)


@qd.func(requires_top_level=True)
def func_query_geoms(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    query: qd.template(),
    candidates: ContactCandidates,
    mpm_info: MpmInfo,
):
    """Record the coupled geoms whose grown box meets the band of every valid particle as candidates: the two grown
    boxes together cover the band of the relative velocity."""
    func_reset_candidates(candidates)
    qd.loop_config(name="mpm_query_geoms")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        x_p = particles.pos[f, i_p, i_b]
        reach = mpm_info.dx[()] + 2.0 * dt * particles.vel[f, i_p, i_b].norm()
        func_bvh_query_aabb(i_b, s, x_p - reach, x_p + reach, query.tree, candidates.results, func_no_filter, query)


@qd.func(requires_top_level=True)
def func_score_geoms(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    query: qd.template(),
    candidates: ContactCandidates,
    mpm_info: MpmInfo,
):
    """Score every candidate geom by the gap of the particle to its contact range, its signed distance minus its
    contact margin `coup_softness`."""
    func_report_candidates(candidates)
    qd.loop_config(name="mpm_score_geoms")
    for i_cand in range(candidates.order.n_triplets[()]):
        triplet = candidates.results.triplets[i_cand]
        i_b = triplet[0]
        i_g = query.coupled_geoms[triplet[1]]
        s = triplet[2]
        i_p, i_b_sorted, env, cell = particle_of_sorted(s, grid, mpm_info)
        x_p = particles.pos[f, i_p, i_b]
        phi = geom_sdf(i_g, i_b, x_p, query.geoms_state, query.geoms_info, query.sdf_info)
        set_candidate_score(i_cand, s, phi - query.geoms_info.coup_softness[i_g], 0.0, candidates)


# =============================================== Layer 1: contact pairs ===============================================


@qd.func(requires_top_level=True)
def func_build_rigid_pairs(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    pairs: MpmRigidPairs,
    geoms_state: GeomsState,
    links_state: LinksState,
    query: qd.template(),
    candidates: ContactCandidates,
    particles_info: MpmParticlesInfo,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    mpm_info: MpmInfo,
    config: qd.template(),
):
    """Build the pair of every valid particle with its nearest coupled geom within its soft range, once the candidates
    are sorted by particle.

    The geom's `coup_softness` is a contact margin: a row starts at that distance from the surface, and its constraint
    position is the signed distance minus the margin, so the resting penetration of the soft row is taken up outside
    the geom. As in Rigid's contact assembly, the friction coefficient is `max(friction * friction_ratio, 0.01)`, the
    time constant is floored at twice the substep, and every row `normal + mu d_k` of the pyramid gets
    `diag = 2 mu_reg^2 (1 + mu_reg^2) invweight (1 - imp) / imp`, with `mu_reg^2 = mu^2 / impratio`. The row target `dt J.v_p + dt^2 aref`
    is on the displacement of the particle relative to the motion J_pt x of its contact point beyond the link's velocity
    at the substep start. The nearest geom gives a pair, by `band_row`, to a particle within its range or short of it
    by less than `contact_band_width`."""
    func_select_nearest(0.0, candidates)
    for _ in range(1):
        pairs.n_pairs[()] = 0
    qd.loop_config(name="mpm_build_rigid_pairs")
    for i_seg in range(candidates.order.bcoo_nnz[()]):
        i_cand = candidates.best_candidate[i_seg]
        if i_cand < candidates.secondary.shape[0]:
            triplet = candidates.results.triplets[i_cand]
            i_b = triplet[0]
            i_g = query.coupled_geoms[triplet[1]]
            s = triplet[2]
            phi = candidates.order.triplet_val[i_cand]
            i_p, i_b_sorted, env, cell = particle_of_sorted(s, grid, mpm_info)
            x_p = particles.pos[f, i_p, i_b]
            i_l = geoms_info.link_idx[i_g]
            v_r = func_vel_at_point(i_l, i_b, x_p, links_state)
            v_rel = particles.vel[f, i_p, i_b] - v_r
            # The relative speed bounds the normal approach speed, which needs the normal
            if phi < contact_band_width(v_rel.norm(), dt, mpm_info):
                normal = geom_normal(i_g, i_b, x_p, geoms_state, geoms_info, rigid_info, sdf_info, config)
                if phi < contact_band_width(-normal.dot(v_rel), dt, mpm_info):
                    i_pair = append_pair(pairs.n_pairs, pairs.s.shape[0])
                    if i_pair >= 0:
                        I_l = [i_l, i_b] if qd.static(config.rigid_batch_links_info) else i_l
                        friction = qd.max(geoms_info.friction[i_g] * geoms_state.friction_ratio[i_g, i_b], 1e-2)
                        friction_sq_reg = friction * friction / rigid_info.impratio[None]
                        invweight = (
                            contact_invweight(f, i_p, i_b, normal, particles, particles_info)
                            + links_info.invweight[I_l][0]
                        )
                        diag_coef = invweight * (1.0 + friction_sq_reg) * 2.0 * friction_sq_reg
                        sol_params = geoms_info.sol_params[i_g]
                        sol_params[0] = qd.max(sol_params[0], 2.0 * dt)
                        dirs = pyramid_dirs(normal, friction)
                        D = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
                        t = qd.Vector.zero(gs.qd_float, PYRAMID_ROWS)
                        for i in qd.static(range(PYRAMID_ROWS)):
                            J = dirs[i, :]
                            D_i, t_i = band_row(sol_params, phi, J.dot(v_rel), diag_coef, dt)
                            D[i] = D_i
                            t[i] = t_i + dt * J.dot(v_r)
                        pairs.s[i_pair] = s
                        pairs.i_b[i_pair] = i_b
                        pairs.geom[i_pair] = i_g
                        pairs.link[i_pair] = i_l
                        pairs.phi[i_pair] = phi
                        pairs.normal[i_pair] = normal
                        pairs.friction[i_pair] = friction
                        pairs.D[i_pair] = D
                        pairs.t[i_pair] = t


@qd.func
def rigid_pair_terms(f: int, i_pair: int, mpm: qd.template(), contact: qd.template()):
    """Energy, gradient and Hessian of the rows of a pair in the displacement of its particle relative to its contact
    point at the current iterate, which a Rigid without dofs leaves at rest."""
    r = mpm.func_vertex_value(f, contact.pairs.s[i_pair])
    if qd.static(contact.has_rigid_dofs):
        r -= contact.func_vertex_value(f, i_pair, SRC_ITERATE)
    return pyramid_terms(
        contact.pairs.normal[i_pair],
        contact.pairs.friction[i_pair],
        contact.pairs.D[i_pair],
        contact.pairs.t[i_pair],
        r,
    )


@qd.func(requires_top_level=True)
def func_rigid_pair_terms(
    f: int, i_d0: int, i_t0: int, mpm: qd.template(), contact: qd.template(), newton_contact: qd.template()
):
    """Write the gradient and Hessian of the rows of every pair at the current iterate, from doublet slot `i_d0` and
    triplet slot `i_t0` on: on the particle, and with Rigid dofs on the contact point and the block between them."""
    qd.loop_config(name="mpm_rigid_pair_terms")
    for i_pair in range(contact.pairs.n_pairs[()]):
        _, g, H = rigid_pair_terms(f, i_pair, mpm, contact)
        i_vp = mpm.vertex_offset[()] + contact.pairs.s[i_pair]
        if qd.static(contact.has_rigid_dofs):
            i_vc = contact.vertex_offset[()] + i_pair
            write_doublet(i_d0 + 2 * i_pair, i_vp, g, newton_contact)
            write_doublet(i_d0 + 2 * i_pair + 1, i_vc, -g, newton_contact)
            write_triplet(i_t0 + 3 * i_pair, i_vp, i_vp, H, newton_contact)
            write_triplet(i_t0 + 3 * i_pair + 1, i_vp, i_vc, -H, newton_contact)
            write_triplet(i_t0 + 3 * i_pair + 2, i_vc, i_vc, H, newton_contact)
        else:
            write_doublet(i_d0 + i_pair, i_vp, g, newton_contact)
            write_triplet(i_t0 + i_pair, i_vp, i_vp, H, newton_contact)


@qd.func(requires_top_level=True)
def func_rigid_pair_energy(f: int, mpm: qd.template(), contact: qd.template(), energies: qd.template()):
    """Add the energy of the rows of every pair at the current iterate into the entry of its environment."""
    qd.loop_config(name="mpm_rigid_pair_energy")
    for i_pair in range(contact.pairs.n_pairs[()]):
        energy, grad, hess = rigid_pair_terms(f, i_pair, mpm, contact)
        qd.atomic_add(energies[contact.pairs.i_b[i_pair]], energy)


# ============================================ Layer 1: contact points ============================================


@qd.func
def contact_point_arm(
    f: int,
    i_pair: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    pairs: MpmRigidPairs,
    links_state: LinksState,
    mpm_info: MpmInfo,
):
    """Link, environment and arm of the contact point of a pair, at the position of its particle at the substep
    start."""
    i_p, i_b_sorted, env, cell = particle_of_sorted(pairs.s[i_pair], grid, mpm_info)
    i_l = pairs.link[i_pair]
    i_b = pairs.i_b[i_pair]
    return i_l, i_b, particle_arm(f, i_p, i_b, i_l, particles, links_state)


@qd.func
def link_point_jacobian(r: qd.types.vector(3)):
    """Map P = [I, -[r]x] of the spatial displacement of a link to the displacement of its point at arm r."""
    R = gu.qd_vec_to_skew(r)
    P = qd.Matrix.zero(gs.qd_float, 3, 6)
    for i in qd.static(range(3)):
        P[i, i] = 1.0
        for j in qd.static(range(3)):
            P[i, 3 + j] = -R[i, j]
    return P


# ============================================ Layer 1: reaction on Rigid ============================================


@qd.func(requires_top_level=True)
def func_mpm_reaction(
    f: int,
    mpm: qd.template(),
    contact: qd.template(),
    rigid_data: RigidNewtonScratch,
    dyn_state: DynState,
    constraint_state: ConstraintState,
    config: qd.template(),
):
    """Apply the force of the MPM pairs on Rigid at the converged solve, after func_rigid_write_back.

    qfrc_constraint becomes M qacc - qf_smooth, Rigid's own and the MPM rows' generalized force together, so that
    Rigid's dofs.force is M qacc of the converged qacc as after its own solve. The rows' spatial force on each link
    stays in `link_force_vel` / `link_force_ang` (in units of dt^2 force) for func_add_mpm_contact_force: the contact
    force of a link is not recoverable from the generalized force of a chain.

    A link pushed by the rows is marked constrained, as Rigid's assembly marks the links of its own contacts, since the
    midpoint integration of a free link (see func_midpoint_eligible) replaces qacc by the solution of the unconstrained
    equation of motion."""
    contact.func_vertex_begin(f, SRC_ITERATE)
    qd.loop_config(name="mpm_reaction_pairs")
    for i_pair in range(contact.pairs.n_pairs[()]):
        energy, g, hess = rigid_pair_terms(f, i_pair, mpm, contact)
        if g.norm_sqr() > 0.0:
            i_l, i_b, r = contact_point_arm(
                f, i_pair, mpm.particles, mpm.grid, contact.pairs, dyn_state.links, mpm.mpm_info
            )
            qd.atomic_add(rigid_data.link_force_vel[i_l, i_b], g)
            qd.atomic_add(rigid_data.link_force_ang[i_l, i_b], r.cross(g))
            dyn_state.links.is_constrained[i_l, i_b] = True
    n_dofs = rigid_data.x.shape[0]
    n_envs = rigid_data.x.shape[1]
    qd.loop_config(name="mpm_reaction_qfrc")
    for i_d, i_b in qd.ndrange(n_dofs, n_envs):
        constraint_state.qfrc_constraint[i_d, i_b] = rigid_data.Ma[i_d, i_b] - dyn_state.dofs.qf_smooth[i_d, i_b]


@qd.func(requires_top_level=True)
def func_add_mpm_contact_force(rigid_data: RigidNewtonScratch, links_state: LinksState, mpm_info: MpmInfo):
    """Add the MPM rows' force on each link (func_mpm_reaction) to `links_state.contact_force`, after Rigid's own
    contact forces were written by the post-solve stage of its substep."""
    n_links = rigid_data.link_force_vel.shape[0]
    n_envs = rigid_data.link_force_vel.shape[1]
    dt = mpm_info.substep_dt[()]
    qd.loop_config(name="mpm_add_contact_force")
    for i_l, i_b in qd.ndrange(n_links, n_envs):
        force = rigid_data.link_force_vel[i_l, i_b]
        links_state.contact_force[i_l, i_b] = links_state.contact_force[i_l, i_b] + force / (dt * dt)
