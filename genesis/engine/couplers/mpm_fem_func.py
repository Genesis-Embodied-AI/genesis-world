"""Contact pairs of the material point method (MPM) with the implicit finite element method (FEM) in the joint Newton
solve (see fem_newton.py for the FEM unknown and its stages)."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import build_bvh, get_bvh_data
from genesis.engine.solvers.fem_newton import FemNewtonParticipant
from genesis.engine.solvers.mpm_data import PYRAMID_ROWS
from genesis.engine.solvers.mpm_func import contact_band_width, contact_invweight, particle_of_sorted
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_linear import PAIRS
from genesis.utils.data import Data, DataItem, DataKind, field, iter_data

from .contact_candidates import CouplingPairs


class MpmFemPairs(Data, kind=DataKind.SCRATCH):
    """The pairs of the particles with their nearest FEM surface triangle (see MpmFemContact.build_pair)."""

    n_pairs: qd.Tensor = field(int, ndarray=True)
    sorted_idx: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    i_b: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    triangle_idx: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    barycentric: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    normal: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    friction: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    stiffness: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)
    target: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)


# =============================================== Layer 0: geometry ===============================================


@qd.func
def closest_point_barycentric(
    p: qd.types.vector(3), a: qd.types.vector(3), b: qd.types.vector(3), c: qd.types.vector(3)
):
    """Barycentric coordinates of the closest point of a triangle to a point (Ericson 2005, Real-Time Collision
    Detection, Sec. 5.1.5)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    barycentric = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    if d1 > 0.0 or d2 > 0.0:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        if d3 >= 0.0 and d4 <= d3:
            barycentric = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                v = d1 / (d1 - d3)
                barycentric = qd.Vector([1.0 - v, v, 0.0], dt=gs.qd_float)
            else:
                cp = p - c
                d5 = ab.dot(cp)
                d6 = ac.dot(cp)
                if d6 >= 0.0 and d5 <= d6:
                    barycentric = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        barycentric = qd.Vector([1.0 - w, 0.0, w], dt=gs.qd_float)
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            barycentric = qd.Vector([0.0, 1.0 - w, w], dt=gs.qd_float)
                        else:
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            barycentric = qd.Vector([1.0 - v - w, v, w], dt=gs.qd_float)
    return barycentric


@qd.func
def is_in_element(i_e: int, i_b: int, f: int, x: qd.types.vector(3), fem: qd.template()):
    """Whether a point lies in a tetrahedral element, by its barycentric coordinates."""
    el2v = fem.solver.elements_i[i_e].el2v
    v3 = fem.solver.elements_v[f, el2v[3], i_b].pos
    edges = qd.Matrix.cols([fem.solver.elements_v[f, el2v[k], i_b].pos - v3 for k in qd.static(range(3))])
    is_inside = False
    if qd.abs(edges.determinant()) > gs.EPS:
        barycentric = edges.inverse() @ (x - v3)
        is_inside = (barycentric >= 0.0).all() and barycentric.sum() <= 1.0
    return is_inside


# ============================================== Layer 1: broad phase ==============================================


class FemContactParams(Data, kind=DataKind.CONSTANT, shape=()):
    """Scene values of the MPM-FEM contact pairs, as device scalars.

    `thickness` is the surface thickness in metres, and `stiffness` the penalty stiffness of a row in units of its
    effective mass, the inverse of its inverse weight."""

    thickness: qd.Tensor = field(float, ndarray=True)
    stiffness: qd.Tensor = field(float, ndarray=True)


@qd.data_oriented
class FemTriangleQuery:
    """The linear bounding volume hierarchy (LBVH) over the finite element method (FEM) surface triangles of every
    environment (see genesis.engine.bvh).

    It holds the FEM solver, the values of the material point method (MPM) kernels `mpm_info` and the values of the
    contact pairs `contact`."""

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
    """The box of every surface triangle over the element behind it, so that a particle that penetrated the element
    still finds the triangle, grown by the contact thickness and twice the substep travel of its fastest vertex."""
    dt = query.mpm_info.substep_dt[()]
    thickness = query.contact.thickness[()]
    for i_b, i_s in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        lo = gu.qd_vec3(qd.math.inf)
        hi = gu.qd_vec3(-qd.math.inf)
        speed = gs.qd_float(0.0)
        i_e = query.fem.solver.surface[i_s].tri2el
        for k in qd.static(range(4)):
            i_v = query.fem.solver.elements_i[i_e].el2v[k]
            lo = qd.min(lo, query.fem.solver.elements_v[f, i_v, i_b].pos)
            hi = qd.max(hi, query.fem.solver.elements_v[f, i_v, i_b].pos)
            speed = qd.max(speed, query.fem.solver.elements_v[f, i_v, i_b].vel.norm())
        margin = thickness + 2.0 * dt * speed
        query.leaves.aabbs_min[i_b, i_s] = lo - margin
        query.leaves.aabbs_max[i_b, i_s] = hi + margin


@qd.func
def triangle_closest_point(f: int, i_b: int, i_s: int, x: qd.types.vector(3), fem: qd.template()):
    """The closest point of a FEM surface triangle to a position.

    The triangle is i_s at its vertex positions of frame `f`, and the position is `x`."""
    tri2v = fem.solver.surface[i_s].tri2v
    a = fem.solver.elements_v[f, tri2v[0], i_b].pos
    b = fem.solver.elements_v[f, tri2v[1], i_b].pos
    c = fem.solver.elements_v[f, tri2v[2], i_b].pos
    barycentric = closest_point_barycentric(x, a, b, c)
    separation = x - (barycentric[0] * a + barycentric[1] * b + barycentric[2] * c)
    distance = separation.norm()
    n_tri = gu.qd_normalize((b - a).cross(c - a), gs.EPS)
    side = gs.qd_float(1.0)
    is_valid = True
    if separation.dot(n_tri) < 0.0:
        side = -1.0
        is_valid = is_in_element(fem.solver.surface[i_s].tri2el, i_b, f, x, fem)
    align = qd.abs(separation.dot(n_tri)) / qd.max(distance, gs.EPS)
    normal = n_tri
    if side > 0.0 and distance > gs.EPS:
        normal = separation / distance
    return is_valid, distance, align, side * distance, barycentric, normal


# ============================================== Layer 2: pair kind ===============================================


@qd.data_oriented
class MpmFemContact(CouplingPairs):
    """Pair kind of the contacts between the material point method (MPM) and the finite element method (FEM), the pair
    of a particle and its nearest surface triangle, over the particle and the vertices of the triangle."""

    n_doublets = 4
    n_triplets = 10
    pairs_type = MpmFemPairs
    # Distances to two triangles within this fraction of the grid spacing are those to a closest point they share
    tie_ratio = 1e-6

    def __init__(self, mpm: MpmNewtonParticipant, fem: FemNewtonParticipant, query: FemTriangleQuery):
        self.fem = fem
        self.query = query
        super().__init__(mpm)

    @property
    def data(self) -> Iterator[DataItem]:
        yield from super().data
        yield from iter_data(self.query, "contact")

    def build_query(self, f: int) -> None:
        self.query.build(f)

    @qd.func
    def candidate_score(self, f: int, i_b: int, i_s: int, i_ps: int):
        """Unsigned distance of a sorted particle to a triangle, infinite for a triangle that cannot hold the particle,
        and the alignment of the normal of the triangle.

        The particle is `i_ps` and the triangle i_s. The alignment breaks the ties (see triangle_closest_point), since
        the sign of the distance to a triangle sharing the closest edge or vertex may be wrong on a closed surface."""
        i_p, _, _ = particle_of_sorted(i_ps, self.mpm.grid, self.mpm.mpm_info)
        x_p = self.mpm.particles.pos[f, i_p, i_b]
        is_valid, distance, align, signed_distance, barycentric, normal = triangle_closest_point(
            f, i_b, i_s, x_p, self.fem
        )
        score = qd.math.inf
        if is_valid:
            score = distance
        return score, align

    @qd.func
    def build_pair(self, f: int, i_b: int, i_s: int, i_ps: int, score: float):
        """Pair a sorted particle with its nearest FEM surface triangle if it lies within its contact range or short of
        it by less than `contact_band_width`.

        The particle is `i_ps` and the triangle i_s, whose signed distance the pair measures again."""
        i_p, _, _ = particle_of_sorted(i_ps, self.mpm.grid, self.mpm.mpm_info)
        x_p = self.mpm.particles.pos[f, i_p, i_b]
        is_valid, distance, align, signed_distance, barycentric, normal = triangle_closest_point(
            f, i_b, i_s, x_p, self.fem
        )
        phi = signed_distance - self.query.contact.thickness[()]
        v_q = qd.Vector.zero(gs.qd_float, 3)
        inv_mass = gs.qd_float(0.0)
        for i_v_ in qd.static(range(3)):
            i_v = self.fem.solver.surface[i_s].tri2v[i_v_]
            v_q += barycentric[i_v_] * self.fem.solver.elements_v[f, i_v, i_b].vel
            inv_mass += barycentric[i_v_] * barycentric[i_v_] / self.fem.solver.elements_v_info[i_v].mass
        v_rel = self.mpm.particles.vel[f, i_p, i_b] - v_q
        if phi < contact_band_width(-normal.dot(v_rel), self.mpm.mpm_info.substep_dt[()], self.mpm.mpm_info):
            invweight = contact_invweight(f, i_p, i_b, normal, self.mpm.particles, self.mpm.particles_info) + inv_mass
            i_e = self.fem.solver.surface[i_s].tri2el
            friction = qd.max(self.fem.solver.elements_i[i_e].friction_mu, self.mpm.mpm_info.contact_min_friction[()])
            stiffness = qd.Vector([self.query.contact.stiffness[()] / invweight] * PYRAMID_ROWS)
            i_pair = self.add_pair(i_ps, i_b, normal, friction, stiffness, qd.Vector([-phi] * PYRAMID_ROWS))
            if i_pair >= 0:
                self.pairs.triangle_idx[i_pair] = i_s
                self.pairs.barycentric[i_pair] = barycentric

    @qd.func
    def pair_vertices(self, f: int, i_pair: int):
        """The particle then the vertices of the triangle of a pair, of weights 1 and -b_k in the relative displacement
        u_rel = dx_p - sum_k b_k dx_k, with u_rel at the current iterate."""
        i_b = self.pairs.i_b[i_pair]
        barycentric = self.pairs.barycentric[i_pair]
        i_s = self.pairs.triangle_idx[i_pair]
        tri2v = self.fem.solver.surface[i_s].tri2v
        verts_fem = qd.Vector([i_b * self.fem.n_vertices + tri2v[i_v_] for i_v_ in qd.static(range(3))], dt=gs.qd_int)
        i_v_mpm = self.mpm.vertex_range.offset[()] + self.pairs.sorted_idx[i_pair]
        verts = qd.Vector(
            [i_v_mpm] + [self.fem.vertex_range.offset[()] + verts_fem[i_v_] for i_v_ in qd.static(range(3))],
            dt=gs.qd_int,
        )
        weights = qd.Vector([1.0, -barycentric[0], -barycentric[1], -barycentric[2]], dt=gs.qd_float)
        u_rel = self.mpm.func_vertex_value(f, self.pairs.sorted_idx[i_pair])
        for i_v_ in range(3):
            u_rel += weights[i_v_ + 1] * self.fem.func_vertex_value(f, verts_fem[i_v_])
        return verts, weights, u_rel
