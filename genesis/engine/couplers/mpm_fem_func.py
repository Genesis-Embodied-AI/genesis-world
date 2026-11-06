"""Contact pairs of the material point method (MPM) with the implicit finite element method (FEM) in the joint Newton
solve over the grid displacement and the FEM vertex positions (see fem_newton.py for the FEM unknown and its stages).

An MPM particle gets a pair of penalty rows against its nearest FEM surface triangle, at its signed distance minus a
contact thickness: the triangles have no volume of their own, and a coarse surface would otherwise let the material slip
between its vertices. Penetration is allowed, in proportion to the contact force over the penalty stiffness. The rows
act on the particle displacement relative to the displacement of its closest point, the barycentric combination of the
triangle's vertex displacements over the whole substep. The pairs are contact pairs of the Newton solve (see
newton_contact.py), over the particle and the three vertices of the triangle.
"""

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, func_no_filter, get_bvh_data
from genesis.engine.solvers.mpm_data import MpmGridScratch, MpmInfo, MpmParticles, MpmParticlesInfo
from genesis.engine.solvers.mpm_func import contact_band_width, contact_invweight, particle_of_sorted, pyramid_terms
from genesis.engine.solvers.newton_contact import PAIRS, SRC_ITERATE, append_pair, write_doublet, write_triplet
from genesis.utils.data import Data, DataKind, field

from .contact_candidates import (
    ContactCandidates,
    func_report_candidates,
    func_reset_candidates,
    func_select_nearest,
    set_candidate_score,
)

# Smallest friction coefficient of a contact row, the lower bound of the Rigid contacts too
MIN_FRICTION = 1e-2
# Distances to two triangles within this fraction of the grid spacing are those to a closest point they share
TIE_DIST_RATIO = 1e-6
# The upper blocks (k, l), k <= l, between the particle (0) and the triangle vertices (1 to 3) of a pair
_PAIR_K = tuple(k for k in range(4) for l in range(k, 4))
_PAIR_L = tuple(l for k in range(4) for l in range(k, 4))


class MpmFemPairs(Data, kind=DataKind.SCRATCH):
    """The pairs of the particles with their nearest FEM surface triangle (see func_build_fem_pairs).

    A pair holds its sorted particle `s`, its environment `i_b`, its triangle `tri` and the barycentric coordinates
    `bary` of the closest point, and the rows of its friction pyramid around `normal` with friction coefficient
    `friction`, of stiffness `D` and target `t` each. `n_pairs` counts the pairs asked for, above the capacity on
    overflow."""

    n_pairs: qd.Tensor = field(int, ndarray=True)
    s: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    i_b: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    tri: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    bary: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    normal: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    friction: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    D: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    t: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)


def build_fem_pairs(max_pairs: int) -> MpmFemPairs:
    """Build the MPM-FEM pairs of a capacity, with no pair."""
    pairs = MpmFemPairs(sizes={PAIRS: max_pairs})
    pairs.n_pairs.fill(0)
    return pairs


# =============================================== Layer 0: geometry ===============================================


@qd.func
def closest_point_barycentric(
    p: qd.types.vector(3), a: qd.types.vector(3), b: qd.types.vector(3), c: qd.types.vector(3)
):
    """Barycentric coordinates of the closest point of a triangle to a point (Ericson 2005, Real-Time Collision
    Detection, Sec. 5.1.5).

    The coordinates are those of the point of triangle abc closest to p, in the order a, b, c."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    bary = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    if d1 > 0.0 or d2 > 0.0:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        if d3 >= 0.0 and d4 <= d3:
            bary = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                v = d1 / (d1 - d3)
                bary = qd.Vector([1.0 - v, v, 0.0], dt=gs.qd_float)
            else:
                cp = p - c
                d5 = ab.dot(cp)
                d6 = ac.dot(cp)
                if d6 >= 0.0 and d5 <= d6:
                    bary = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        bary = qd.Vector([1.0 - w, 0.0, w], dt=gs.qd_float)
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            bary = qd.Vector([0.0, 1.0 - w, w], dt=gs.qd_float)
                        else:
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            bary = qd.Vector([1.0 - v - w, v, w], dt=gs.qd_float)
    return bary


@qd.func
def is_in_element(i_e: int, i_b: int, f: int, x: qd.types.vector(3), fem: qd.template()):
    """Whether a point lies in a tetrahedral element, by its barycentric coordinates.

    The point is x and the element i_e of environment i_b, at frame f. A degenerate element holds no point."""
    el2v = fem.elements_i[i_e].el2v
    v3 = fem.elements_v[f, el2v[3], i_b].pos
    D = qd.Matrix.cols([fem.elements_v[f, el2v[k], i_b].pos - v3 for k in qd.static(range(3))])
    is_inside = False
    if qd.abs(D.determinant()) > gs.EPS:
        lam = D.inverse() @ (x - v3)
        is_inside = (lam >= 0.0).all() and lam.sum() <= 1.0
    return is_inside


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class FemContactParams:
    """Scene values of the MPM-FEM contact pairs, as device scalars.

    The FEM surface takes the thickness `thickness`, in metres, half a grid cell of MPM: a triangle has no volume of its
    own. A row takes the penalty stiffness `stiffness` in units of its effective mass, the inverse of its inverse
    weight: the row energy is 1/2 stiffness m_eff c^2 in the joint dt^2-scaled units."""

    def __init__(self, dx: float) -> None:
        self.thickness = qd.ndarray(gs.qd_float, shape=())
        self.thickness.from_numpy(np.array(0.5 * dx, dtype=gs.np_float))
        self.stiffness = qd.ndarray(gs.qd_float, shape=())
        self.stiffness.from_numpy(np.array(10.0, dtype=gs.np_float))


@qd.data_oriented
class FemTriangleQuery:
    """The linear bounding volume hierarchy (LBVH) over the finite element method (FEM) surface triangles of every
    environment (see genesis.engine.bvh), with the FEM solver, the values of the material point method (MPM) kernels
    `mpm_info` and the values of the contact pairs `contact`."""

    def __init__(self, fem, mpm_info, contact: FemContactParams, n_envs):
        self.fem = fem
        self.mpm_info = mpm_info
        self.contact = contact
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, fem.n_surfaces)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves

    def build(self, f):
        """Fit the leaf boxes to the triangles at the substep start and build the trees."""
        kernel_fit_triangle_leaves(f, self)
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)


@qd.kernel
def kernel_fit_triangle_leaves(f: int, query: qd.template()):
    """The box of every surface triangle over the element behind it, so that a particle that penetrated the surface
    element still finds the triangle.

    The box grows by the contact thickness and by twice the distance the fastest vertex of the element travels over the
    substep."""
    dt = query.mpm_info.substep_dt[()]
    thickness = query.contact.thickness[()]
    for i_b, i_s in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        lo = gu.qd_vec3(qd.math.inf)
        hi = gu.qd_vec3(-qd.math.inf)
        speed = gs.qd_float(0.0)
        i_e = query.fem.surface[i_s].tri2el
        for k in qd.static(range(4)):
            i_v = query.fem.elements_i[i_e].el2v[k]
            lo = qd.min(lo, query.fem.elements_v[f, i_v, i_b].pos)
            hi = qd.max(hi, query.fem.elements_v[f, i_v, i_b].pos)
            speed = qd.max(speed, query.fem.elements_v[f, i_v, i_b].vel.norm())
        margin = thickness + 2.0 * dt * speed
        query.leaves.aabbs_min[i_b, i_s] = lo - margin
        query.leaves.aabbs_max[i_b, i_s] = hi + margin


@qd.func(requires_top_level=True)
def func_query_triangles(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    query: qd.template(),
    candidates: ContactCandidates,
    mpm_info: MpmInfo,
):
    """Record the triangles of every valid particle as candidates.

    The particle's box spans its band at its own speed, and the grown triangle boxes span the thickness and the vertex
    motion, so together they cover the band of the relative velocity."""
    func_reset_candidates(candidates)
    qd.loop_config(name="mpm_query_triangles")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        x_p = particles.pos[f, i_p, i_b]
        reach = contact_band_width(particles.vel[f, i_p, i_b].norm(), dt, mpm_info)
        func_bvh_query_aabb(i_b, s, x_p - reach, x_p + reach, query.tree, candidates.results, func_no_filter, query)


@qd.func
def triangle_closest_point(f: int, s: int, i_b: int, i_s: int, x: qd.types.vector(3), fem: qd.template()):
    """The closest point of a FEM surface triangle to the position `x` of sorted particle `s`, at frame `f`.

    Returns whether the triangle may hold the particle, its unsigned distance, the alignment of its normal with the
    direction to the particle, the signed distance outward along its normal, the barycentric coordinates of the closest
    point, and the contact normal, the direction from the closest point to a particle outside.

    A particle behind the plane of a triangle is inside the entity only if it lies in the element behind the triangle,
    whose box the leaf of the triangle spans. Behind any other found triangle, the particle may lie far outside the
    entity, where the grown box of that triangle reaches it and the box of its nearest triangle misses it, so the
    triangle is rejected. Penetration deeper than the surface elements gives no pair."""
    tri2v = fem.surface[i_s].tri2v
    a = fem.elements_v[f, tri2v[0], i_b].pos
    b = fem.elements_v[f, tri2v[1], i_b].pos
    c = fem.elements_v[f, tri2v[2], i_b].pos
    bary = closest_point_barycentric(x, a, b, c)
    d = x - (bary[0] * a + bary[1] * b + bary[2] * c)
    dist = d.norm()
    n_tri = gu.qd_normalize((b - a).cross(c - a), gs.EPS)
    side = gs.qd_float(1.0)
    is_valid = True
    if d.dot(n_tri) < 0.0:
        side = -1.0
        is_valid = is_in_element(fem.surface[i_s].tri2el, i_b, f, x, fem)
    align = qd.abs(d.dot(n_tri)) / qd.max(dist, gs.EPS)
    normal = n_tri
    if side > 0.0 and dist > gs.EPS:
        normal = d / dist
    return is_valid, dist, align, side * dist, bary, normal


@qd.func(requires_top_level=True)
def func_score_triangles(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    fem: qd.template(),
    candidates: ContactCandidates,
    mpm_info: MpmInfo,
):
    """Score every candidate triangle by its unsigned distance to the particle and break the ties by the alignment of
    its normal (see triangle_closest_point).

    Among the triangles whose closest point is shared on an edge or a vertex, the one whose normal is most aligned with
    the direction to the particle is kept. The sign of the distance to the other ones may be wrong on a closed surface,
    as for the bottom edge of a side face of a box seen from below its bottom face."""
    func_report_candidates(candidates)
    qd.loop_config(name="mpm_score_triangles")
    for i_cand in range(candidates.order.n_triplets[()]):
        triplet = candidates.results.triplets[i_cand]
        i_b = triplet[0]
        i_s = triplet[1]
        s = triplet[2]
        i_p, i_b_sorted, env, cell = particle_of_sorted(s, grid, mpm_info)
        is_valid, dist, align, phi, bary, normal = triangle_closest_point(
            f, s, i_b, i_s, particles.pos[f, i_p, i_b], fem
        )
        score = qd.math.inf
        if is_valid:
            score = dist
        set_candidate_score(i_cand, s, score, align, candidates)


# ============================================== Layer 1: contact pairs ==============================================


@qd.func(requires_top_level=True)
def func_build_fem_pairs(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    pairs: MpmFemPairs,
    fem: qd.template(),
    query: qd.template(),
    candidates: ContactCandidates,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Build the pair of every valid particle with its nearest FEM surface triangle, within its contact range or short
    of it by less than `contact_band_width`, once the candidates are sorted by particle.

    The gap phi is the signed distance to the triangle, outward along its normal, minus `query.contact.thickness`.
    Every row of the friction pyramid J = n +- mu d is a penalty on the gap at the end of the substep, phi + J.r for the
    relative displacement r. Its target is -phi and its stiffness `query.contact.stiffness` times the row's
    effective mass, the inverse of the particle's inverse weight (see func_contact_invweight) plus that of the triangle
    point, sum_k b_k^2 / m_k. The row energy depends on the end positions alone, so the rows add no energy, as a
    reference acceleration from the start gap would. The next substep builds its pairs anew and drops the tangential
    part of a pyramid row, which makes it a dissipative friction bounded by the cone. The friction coefficient is the
    element's, at least MIN_FRICTION."""
    func_select_nearest(TIE_DIST_RATIO * mpm_info.dx[()], candidates)
    for _ in range(1):
        pairs.n_pairs[()] = 0
    qd.loop_config(name="mpm_build_fem_pairs")
    for i_seg in range(candidates.order.bcoo_nnz[()]):
        i_cand = candidates.best_candidate[i_seg]
        if i_cand < candidates.secondary.shape[0]:
            triplet = candidates.results.triplets[i_cand]
            i_b = triplet[0]
            i_s = triplet[1]
            s = triplet[2]
            i_p, i_b_sorted, env, cell = particle_of_sorted(s, grid, mpm_info)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            is_valid, dist, align, signed_dist, bary, normal = triangle_closest_point(f, s, i_b, i_s, x_p, fem)
            phi = signed_dist - query.contact.thickness[()]
            v_q = qd.Vector.zero(gs.qd_float, 3)
            inv_mass = gs.qd_float(0.0)
            for k in qd.static(range(3)):
                i_v = fem.surface[i_s].tri2v[k]
                v_q += bary[k] * fem.elements_v[f, i_v, i_b].vel
                inv_mass += bary[k] * bary[k] / fem.elements_v_info[i_v].mass
            if phi < contact_band_width(-normal.dot(v_p - v_q), dt, mpm_info):
                i_pair = append_pair(pairs.n_pairs, pairs.s.shape[0])
                if i_pair >= 0:
                    i_e = fem.surface[i_s].tri2el
                    invweight = contact_invweight(f, i_p, i_b, normal, particles, particles_info) + inv_mass
                    pairs.s[i_pair] = s
                    pairs.i_b[i_pair] = i_b
                    pairs.tri[i_pair] = i_s
                    pairs.bary[i_pair] = bary
                    pairs.normal[i_pair] = normal
                    pairs.friction[i_pair] = qd.max(fem.elements_i[i_e].friction_mu, MIN_FRICTION)
                    pairs.D[i_pair] = query.contact.stiffness[()] / invweight
                    pairs.t[i_pair] = -phi


@qd.func
def fem_pair_vertices(i_pair: int, mpm: qd.template(), fem: qd.template(), pairs: MpmFemPairs):
    """Contact vertices of a pair, the particle then the vertices of the triangle, with their weights in the relative
    displacement r = dx_p - sum_k b_k dx_k, and the index of the triangle vertices in the FEM vertex map."""
    i_b = pairs.i_b[i_pair]
    bary = pairs.bary[i_pair]
    tri2v = fem.surface[pairs.tri[i_pair]].tri2v
    i_fv = qd.Vector([i_b * fem.n_vertices + tri2v[k] for k in qd.static(range(3))], dt=gs.qd_int)
    verts = qd.Vector(
        [mpm.vertex_offset[()] + pairs.s[i_pair]] + [fem.vertex_offset[()] + i_fv[k] for k in qd.static(range(3))],
        dt=gs.qd_int,
    )
    weights = qd.Vector([1.0, -bary[0], -bary[1], -bary[2]], dt=gs.qd_float)
    return verts, weights, i_fv


@qd.func
def fem_pair_terms(f: int, i_pair: int, mpm: qd.template(), fem: qd.template(), pairs: MpmFemPairs):
    """Energy, gradient and Hessian of the rows of a pair in its relative displacement at the current iterate, with
    the vertices and weights of the pair (see fem_pair_vertices)."""
    verts, weights, i_fv = fem_pair_vertices(i_pair, mpm, fem, pairs)
    r = mpm.func_vertex_value(f, pairs.s[i_pair], SRC_ITERATE)
    for k in range(3):
        r += weights[k + 1] * fem.func_vertex_value(f, i_fv[k], SRC_ITERATE)
    D = pairs.D[i_pair]
    t = pairs.t[i_pair]
    energy, g, H = pyramid_terms(
        pairs.normal[i_pair], pairs.friction[i_pair], qd.Vector([D, D, D, D]), qd.Vector([t, t, t, t]), r
    )
    return energy, g, H, verts, weights


@qd.func(requires_top_level=True)
def func_fem_pair_terms(
    f: int, i_d0: int, i_t0: int, mpm: qd.template(), fem: qd.template(), pairs: MpmFemPairs, contact: qd.template()
):
    """Write the gradient and Hessian of the rows of every pair at the current iterate, from doublet slot `i_d0` and
    triplet slot `i_t0` on: 4 doublets on the particle and the triangle vertices, and the 10 upper blocks between
    them."""
    pair_k = qd.Vector(_PAIR_K, dt=gs.qd_int)
    pair_l = qd.Vector(_PAIR_L, dt=gs.qd_int)
    qd.loop_config(name="mpm_fem_pair_terms")
    for i_pair in range(pairs.n_pairs[()]):
        _, g, H, verts, weights = fem_pair_terms(f, i_pair, mpm, fem, pairs)
        for k in range(4):
            write_doublet(i_d0 + 4 * i_pair + k, verts[k], weights[k] * g, contact)
        for i_block in range(10):
            k = pair_k[i_block]
            k_col = pair_l[i_block]
            write_triplet(
                i_t0 + 10 * i_pair + i_block, verts[k], verts[k_col], (weights[k] * weights[k_col]) * H, contact
            )


@qd.func(requires_top_level=True)
def func_fem_pair_energy(f: int, mpm: qd.template(), fem: qd.template(), pairs: MpmFemPairs, energies: qd.template()):
    """Add the energy of the rows of every pair at the current iterate into the entry of its environment."""
    qd.loop_config(name="mpm_fem_pair_energy")
    for i_pair in range(pairs.n_pairs[()]):
        energy, grad, hess, verts, weights = fem_pair_terms(f, i_pair, mpm, fem, pairs)
        qd.atomic_add(energies[pairs.i_b[i_pair]], energy)
