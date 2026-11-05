"""Stages of the contact rows of the material point method (MPM) with the implicit finite element method (FEM), the
coupling of the joint Newton solve over the grid displacement and the FEM vertex positions (see fem_newton.py for the
FEM unknown and its stages).

An MPM particle gets penalty rows against the nearest FEM surface triangle, at its signed distance minus a contact
thickness: the triangles have no volume of their own, and a coarse surface would otherwise let the material slip between
its vertices. Penetration is allowed, in proportion to the contact force over the penalty stiffness. The rows act on the
particle displacement relative to the displacement of its closest point, the barycentric combination of the triangle's
vertex displacements over the whole substep.
"""

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import CONTACT_ROWS, PYRAMID_ROWS, WALL_ROWS, MpmParticles, MpmParticlesInfo
from genesis.engine.solvers.mpm_func import (
    contact_band_width,
    contact_cross_matvec,
    contact_invweight,
    particle_of_sorted,
    pyramid_dirs,
)
from genesis.utils.array_class import BVHQueryResults

# Smallest friction coefficient of a contact row, the lower bound of the Rigid contacts too
MIN_FRICTION = 1e-2
# Distances to two triangles within this fraction of the grid spacing are those to a closest point they share
TIE_DIST_RATIO = 1e-6


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


@qd.func
def triangle_point_disp(i_s: int, i_b: int, f: int, bary: qd.types.vector(3), fem: qd.template()):
    """Displacement over the substep of a point of a FEM surface triangle.

    The point is that of surface triangle i_s of environment i_b at barycentric coordinates bary, displaced from frame f
    to f + 1."""
    disp = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        i_v = fem.surface[i_s].tri2v[k]
        disp += bary[k] * (fem.elements_v[f + 1, i_v, i_b].pos - fem.elements_v[f, i_v, i_b].pos)
    return disp


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class FemContactParams:
    """Scene values of the MPM-FEM contact rows, as device scalars.

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
    """The linear bounding volume hierarchy (LBVH) over the FEM surface triangles of every environment (see
    genesis.engine.bvh).

    It also holds the data the filter of its queries reads: the FEM solver, the values of the MPM kernels `params`, the
    values of the contact rows `contact`, and the query point, frame and nearest triangle of each sorted particle in
    `newton`."""

    def __init__(self, fem, newton, params, contact: FemContactParams, n_envs):
        self.fem = fem
        self.newton = newton
        self.params = params
        self.contact = contact
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, fem.n_surfaces)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        # Every query keeps its nearest triangle in `newton` and records no pair
        self.results = BVHQueryResults(sizes=dict(query_results=1))

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
    dt = query.params.substep_dt[()]
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


@qd.func
def func_nearest_triangle_filter(i_t: int, i_a: int, i_q: int, query: qd.template()) -> bool:
    """The narrow phase of a box query of a sorted particle, which keeps the nearest FEM surface triangle in `newton`.

    The query is that of sorted particle i_q in environment i_t, against surface triangle i_a. The filter keeps the
    nearest triangle in `newton.query_idx`, with its gap in `query_phi` (its signed distance outward along the triangle
    normal minus the thickness), the barycentric coordinates of the closest point, and the contact normal, the direction
    from the closest point to a particle outside. It records no pair.

    The nearest triangle is that of the smallest unsigned distance, and among the triangles whose closest point is
    shared on an edge or a vertex, the one whose normal is most aligned with the direction to the particle. The sign of
    the distance to the other ones may be wrong on a closed surface, as for the bottom edge of a side face of a box seen
    from below its bottom face.

    A particle behind the plane of a triangle is inside the entity only if it lies in the element behind the triangle,
    whose box the leaf of the triangle spans. Behind any other found triangle, the particle may lie far outside the
    entity, where the grown box of that triangle reaches it and the box of its nearest triangle misses it, so the filter
    drops that triangle. Penetration deeper than the surface elements gives no row."""
    f = query.newton.query_frame[()]
    x = query.newton.query_pos[i_q]
    tri2v = query.fem.surface[i_a].tri2v
    a = query.fem.elements_v[f, tri2v[0], i_t].pos
    b = query.fem.elements_v[f, tri2v[1], i_t].pos
    c = query.fem.elements_v[f, tri2v[2], i_t].pos
    bary = closest_point_barycentric(x, a, b, c)
    d = x - (bary[0] * a + bary[1] * b + bary[2] * c)
    dist = d.norm()
    n_tri = gu.qd_normalize((b - a).cross(c - a), gs.EPS)
    side = gs.qd_float(1.0)
    is_valid = True
    if d.dot(n_tri) < 0.0:
        side = -1.0
        is_valid = is_in_element(query.fem.surface[i_a].tri2el, i_t, f, x, query.fem)
    align = qd.abs(d.dot(n_tri)) / qd.max(dist, gs.EPS)
    best_dist = query.newton.query_dist[i_q]
    is_tie = qd.abs(dist - best_dist) <= TIE_DIST_RATIO * query.params.dx[()]
    if is_valid and ((dist < best_dist and not is_tie) or (is_tie and align > query.newton.query_align[i_q])):
        normal = n_tri
        if side > 0.0 and dist > gs.EPS:
            normal = d / dist
        query.newton.query_dist[i_q] = dist
        query.newton.query_align[i_q] = align
        query.newton.query_phi[i_q] = side * dist - query.contact.thickness[()]
        query.newton.query_idx[i_q] = i_a
        query.newton.query_bary[i_q] = bary
        query.newton.query_normal[i_q] = normal
    return True


# ============================================== Layer 1: contact rows ==============================================


@qd.func(requires_top_level=True)
def func_build_fem_rows(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    fem: qd.template(),
    query: qd.template(),
    particles_info: MpmParticlesInfo,
    params: qd.template(),
):
    """Pyramid contact rows of every valid particle without Rigid rows against the nearest FEM surface triangle within
    its contact range or short of it by less than `contact_band_width`.

    The rows follow the wall and Rigid rows. A box query of the triangle trees finds the nearest triangle: the
    particle's box spans its band at its own speed, and the grown triangle boxes span the thickness and the vertex
    motion, so together they cover the band of the relative velocity.

    The gap phi is the signed distance to the triangle, outward along its normal, minus `query.contact.thickness`.
    Every row of the friction pyramid J = n +- mu d is a penalty on the gap at the end of the substep, phi + J.r for the
    relative displacement r. Its target is -phi and its stiffness `query.contact.stiffness` times the row's
    effective mass, the inverse of the particle's inverse weight (see func_contact_invweight) plus that of the triangle
    point, sum_k b_k^2 / m_k. The row energy depends on the end positions alone, so the rows add no energy, as a
    reference acceleration from the start gap would. The next substep builds its rows anew and drops the tangential part
    of a pyramid row, which makes it a dissipative friction bounded by the cone. The friction coefficient is the
    element's, at least MIN_FRICTION."""
    for _ in range(1):
        newton.query_frame[()] = f
    qd.loop_config(name="mpm_build_fem_rows")
    for s in range(grid.n_valid[()]):
        if newton.contact_link[s] < 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            newton.query_pos[s] = x_p
            newton.query_phi[s] = float("inf")
            newton.query_idx[s] = -1
            newton.query_dist[s] = float("inf")
            newton.query_align[s] = -1.0
            reach = contact_band_width(v_p.norm(), dt, params)
            func_bvh_query_aabb(
                i_b, s, x_p - reach, x_p + reach, query.tree, query.results, func_nearest_triangle_filter, query
            )
            best_tri = newton.query_idx[s]
            best_phi = newton.query_phi[s]
            best_bary = newton.query_bary[s]
            best_normal = newton.query_normal[s]
            if best_tri >= 0:
                v_q = qd.Vector.zero(gs.qd_float, 3)
                inv_mass = gs.qd_float(0.0)
                for k in qd.static(range(3)):
                    i_v = fem.surface[best_tri].tri2v[k]
                    v_q += best_bary[k] * fem.elements_v[f, i_v, i_b].vel
                    inv_mass += best_bary[k] * best_bary[k] / fem.elements_v_info[i_v].mass
                v_rel = v_p - v_q
                if best_phi < contact_band_width(-best_normal.dot(v_rel), dt, params):
                    i_e = fem.surface[best_tri].tri2el
                    friction = qd.max(fem.elements_i[i_e].friction_mu, MIN_FRICTION)
                    invweight = contact_invweight(f, i_p, i_b, best_normal, particles, particles_info) + inv_mass
                    D = query.contact.stiffness[()] / invweight
                    dirs = pyramid_dirs(best_normal, friction)
                    for i in qd.static(range(PYRAMID_ROWS)):
                        k = CONTACT_ROWS * s + WALL_ROWS + i
                        newton.contact_dir[k] = dirs[i, :]
                        newton.contact_D[k] = D
                        newton.contact_t[k] = -best_phi
                    newton.contact_tri[s] = best_tri
                    newton.contact_bary[s] = best_bary


@qd.func(requires_top_level=True)
def func_fem_point_disp(f: int, grid: qd.template(), newton: qd.template(), fem: qd.template(), params: qd.template()):
    """Displacement of the contact point of every particle with FEM rows at the current FEM iterate."""
    qd.loop_config(name="mpm_fem_point_disp")
    for s in range(grid.n_valid[()]):
        i_s = newton.contact_tri[s]
        if i_s >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            newton.contact_dr[s] = triangle_point_disp(i_s, i_b, f, newton.contact_bary[s], fem)


@qd.func(requires_top_level=True)
def func_fem_contact_gradient(grid: qd.template(), newton: qd.template(), fem: qd.template(), params: qd.template()):
    """Add the share of the FEM rows to the FEM gradient and to its preconditioner.

    It runs after FEM's own force and diagonal blocks, and after the MPM assembly that staged the row gradients g and
    Hessians H. The vertices of the closest point take the gradient -b_k g, entered in FEM's force (the negative
    gradient over dt^2) as b_k g / dt^2, and b_k^2 H / dt^2 on their diagonal block.

    The rows of the many particles against one triangle are far stiffer than the inertia of its vertices, so a FEM
    preconditioner without them would leave the joint preconditioned conjugate gradient (PCG) stopping on a direction
    that barely moves FEM."""
    inv_dt_sq = 1.0 / (params.substep_dt[()] * params.substep_dt[()])
    qd.loop_config(name="mpm_fem_contact_gradient")
    for s in range(grid.n_valid[()]):
        i_s = newton.contact_tri[s]
        if i_s >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            g = newton.g_rigid[s]
            H = newton.h_rigid[s]
            bary = newton.contact_bary[s]
            for k in qd.static(range(3)):
                i_v = fem.surface[i_s].tri2v[k]
                b_k = bary[k]
                qd.atomic_add(fem.elements_v_energy[i_b, i_v].force, b_k * inv_dt_sq * g)
                fem.pcg_state_v[i_b, i_v].diag3x3 += (b_k * b_k * inv_dt_sq) * H


@qd.func(requires_top_level=True)
def func_fem_contact_matvec(
    f: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    fem: qd.template(),
    pcg_scalars: qd.template(),
    params: qd.template(),
):
    """The FEM rows' share of the joint operator, after the own operators of MPM and FEM set their Ap.

    The closest point of a particle moves by dq = sum_k b_k p_v,k along the direction. The rows add their cross term on
    the MPM rows (see contact_cross_matvec), and b_k h to Ap_v,k of the vertices of the triangle."""
    qd.loop_config(name="mpm_fem_contact_matvec")
    for s in range(grid.n_valid[()]):
        i_s = newton.contact_tri[s]
        i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
        if i_s >= 0 and pcg_scalars.is_converged[i_b] == 0:
            bary = newton.contact_bary[s]
            tri2v = fem.surface[i_s].tri2v
            dq = qd.Vector.zero(gs.qd_float, 3)
            for k in qd.static(range(3)):
                i_v = tri2v[k]
                dq += bary[k] * fem.pcg_state_v[i_b, i_v].p
            h = contact_cross_matvec(f, s, dq, particles, grid, newton, linear, pcg_scalars, params)
            for k in qd.static(range(3)):
                i_v = tri2v[k]
                qd.atomic_add(fem.pcg_state_v[i_b, i_v].Ap, bary[k] * h)
