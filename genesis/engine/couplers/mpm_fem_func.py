"""Stages of the implicit finite element method (FEM) as a participant of the NewtonSolver, and of the contact rows of
the material point method (MPM) with FEM as a coupling.

The FEM unknown is the end-of-substep vertex position x, starting at the substep start. Its share of the joint energy is
FEM's own incremental potential (inertia, elasticity and the linearized damping of its implicit solver) times dt^2, in
the displacement units of the MPM energy. The joint solve replaces FEM's own solve and takes its preconditioned
conjugate gradient (PCG) vectors: pcg_state_v.r, z, p, Ap and x, the solution, whose negative is the Newton direction.

An MPM particle gets penalty rows against the nearest FEM surface triangle, at its signed distance minus a contact
thickness: the triangles have no volume of their own, and a coarse surface would otherwise let the material slip between
its vertices. Penetration is allowed, in proportion to the contact force over the penalty stiffness. The rows act on the
particle displacement relative to the displacement of its closest point, the barycentric combination of the triangle's
vertex displacements over the whole substep.
"""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import BVHQueryResults
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
def func_fem_factor(fem: qd.template()):
    """FEM's preconditioner, the inverse of every vertex's diagonal block, once the contact rows added theirs."""
    qd.loop_config(name="mpm_fem_factor")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].prec = fem.pcg_state_v[i_b, i_v].diag3x3.inverse()


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
        if i_s >= 0 and pcg_scalars.is_converged[()] == 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
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


# ============================================ Layer 1: the FEM block ============================================


@qd.func(requires_top_level=True)
def func_fem_pcg_init(dt: float, fem: qd.template()):
    """The FEM part of the preconditioned conjugate gradient (PCG) start from 0: r = grad = -dt^2 force, solution and Ap
    cleared."""
    qd.loop_config(name="mpm_fem_pcg_init")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].r = -dt * dt * fem.elements_v_energy[i_b, i_v].force
        fem.pcg_state_v[i_b, i_v].x = qd.Vector.zero(gs.qd_float, 3)
        fem.pcg_state_v[i_b, i_v].Ap = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_fem_precondition(dt: float, fem: qd.template()):
    """z = M^-1 r by FEM's inverse 3x3 diagonal blocks, scaled to the joint energy.

    A vertex FEM holds at its target gets z = 0, so that its direction stays 0 and the contact rows leave it in
    place."""
    qd.loop_config(name="mpm_fem_precondition")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        z = (fem.pcg_state_v[i_b, i_v].prec @ fem.pcg_state_v[i_b, i_v].r) / (dt * dt)
        if qd.static(fem._enable_vertex_constraints):
            if fem.vertex_constraints.is_constrained[i_v, i_b]:
                z = qd.Vector.zero(gs.qd_float, 3)
        fem.pcg_state_v[i_b, i_v].z = z


@qd.func(requires_top_level=True)
def func_fem_pcg_init_direction(fem: qd.template(), pcg_scalars: qd.template()):
    """Set the first search direction p = z over the FEM vertices, and add r . z to rz."""
    qd.loop_config(name="mpm_fem_pcg_init_direction")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].p = fem.pcg_state_v[i_b, i_v].z
        qd.atomic_add(pcg_scalars.rz[()], fem.pcg_state_v[i_b, i_v].r.dot(fem.pcg_state_v[i_b, i_v].z))


@qd.func(requires_top_level=True)
def func_fem_matvec(dt: float, fem: qd.template()):
    """Ap = dt^2 H p over the FEM vertices with FEM's own operator, before the FEM rows add their term."""
    fem.compute_Ap()
    qd.loop_config(name="mpm_fem_matvec_scale")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].Ap = dt * dt * fem.pcg_state_v[i_b, i_v].Ap


@qd.func(requires_top_level=True)
def func_fem_matvec_dot(fem: qd.template(), pcg_scalars: qd.template()):
    """Add p . Ap over the FEM vertices to p . Ap, once Ap holds the whole joint operator."""
    qd.loop_config(name="mpm_fem_matvec_dot")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if pcg_scalars.is_converged[()] == 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[()], fem.pcg_state_v[i_b, i_v].p.dot(fem.pcg_state_v[i_b, i_v].Ap))


@qd.func(requires_top_level=True)
def func_fem_pcg_update_xr(fem: qd.template(), pcg_scalars: qd.template()):
    """Update the solution and the residual over the FEM vertices, x += alpha p and r -= alpha Ap."""
    qd.loop_config(name="mpm_fem_pcg_update_xr")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if pcg_scalars.is_converged[()] == 0:
            alpha = pcg_scalars.alpha[()]
            fem.pcg_state_v[i_b, i_v].x = fem.pcg_state_v[i_b, i_v].x + alpha * fem.pcg_state_v[i_b, i_v].p
            fem.pcg_state_v[i_b, i_v].r = fem.pcg_state_v[i_b, i_v].r - alpha * fem.pcg_state_v[i_b, i_v].Ap


@qd.func(requires_top_level=True)
def func_fem_pcg_dot_rz(fem: qd.template(), pcg_scalars: qd.template()):
    """Add r . z over the FEM vertices to rz_new."""
    qd.loop_config(name="mpm_fem_pcg_dot_rz")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if pcg_scalars.is_converged[()] == 0:
            qd.atomic_add(pcg_scalars.rz_new[()], fem.pcg_state_v[i_b, i_v].r.dot(fem.pcg_state_v[i_b, i_v].z))


@qd.func(requires_top_level=True)
def func_fem_pcg_direction(fem: qd.template(), pcg_scalars: qd.template()):
    """Update the search direction p = z + beta p over the FEM vertices, and clear Ap."""
    qd.loop_config(name="mpm_fem_pcg_direction")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].Ap = qd.Vector.zero(gs.qd_float, 3)
        if pcg_scalars.is_converged[()] == 0:
            beta = pcg_scalars.beta[()]
            fem.pcg_state_v[i_b, i_v].p = fem.pcg_state_v[i_b, i_v].z + beta * fem.pcg_state_v[i_b, i_v].p


@qd.func(requires_top_level=True)
def func_fem_start_line_search(f: int, fem: qd.template(), control: qd.template()):
    """The line-search start point, and the largest component of the step -x into the Newton measure."""
    qd.loop_config(name="mpm_fem_start_line_search")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.linesearch_state_v[i_b, i_v].x_prev = fem.elements_v[f + 1, i_v, i_b].pos
        step = fem.pcg_state_v[i_b, i_v].x
        qd.atomic_max(control.max_disp[()], qd.max(qd.abs(step[0]), qd.abs(step[1]), qd.abs(step[2])))


@qd.func(requires_top_level=True)
def func_fem_step_forward(f: int, fem: qd.template(), control: qd.template()):
    qd.loop_config(name="mpm_fem_step_forward")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.elements_v[f + 1, i_v, i_b].pos = (
            fem.linesearch_state_v[i_b, i_v].x_prev - control.alpha[()] * fem.pcg_state_v[i_b, i_v].x
        )


@qd.func(requires_top_level=True)
def func_fem_energy(f: int, dt: float, fem: qd.template(), control: qd.template(), is_trial: bool):
    """Add FEM's incremental potential at the end-of-substep positions, times dt^2, to the start or trial energy.

    The potential holds the inertia and the linearized damping of the vertices, and the elastic energy and damping of
    the elements, at the positions of frame f + 1."""
    dt_sq = dt * dt
    damping_alpha_dt = fem._damping_alpha * fem._substep_dt
    qd.loop_config(name="mpm_fem_energy_vertices")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        diff = fem.elements_v[f + 1, i_v, i_b].pos - fem.elements_v_energy[i_b, i_v].inertia
        energy = 0.5 * fem.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
        if qd.static(fem._damping_alpha > 0.0):
            step = fem.elements_v[f + 1, i_v, i_b].pos - fem.elements_v[f, i_v, i_b].pos
            energy += 0.5 * fem.elements_v_info[i_v].mass_over_dt2 * step.dot(step) * damping_alpha_dt
        if is_trial:
            qd.atomic_add(control.energy_trial[()], dt_sq * energy)
        else:
            qd.atomic_add(control.energy_start[()], dt_sq * energy)
    fem._func_compute_ele_energy(f)
    qd.loop_config(name="mpm_fem_energy_elements")
    for i_b, i_e in qd.ndrange(fem._B, fem.n_elements):
        energy = fem.elements_el_energy[i_b, i_e].energy * fem.elements_i[i_e].V
        if is_trial:
            qd.atomic_add(control.energy_trial[()], dt_sq * energy)
        else:
            qd.atomic_add(control.energy_start[()], dt_sq * energy)
