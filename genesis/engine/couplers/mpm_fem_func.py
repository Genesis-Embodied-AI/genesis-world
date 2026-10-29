"""Stages of implicit FEM as a secondary participant of the NewtonSolver, and of the MPM-FEM contact rows as a coupling.

The FEM unknown is the end-of-substep vertex position x, starting at the substep start; its share of the joint energy
is FEM's own incremental potential (inertia, elasticity and the linearized damping of its implicit solver) times dt^2,
in the displacement units of the MPM energy. Its PCG vectors are those of FEM's own solve, which the joint solve
replaces: pcg_state_v.r, z, p, Ap and x, the solution, whose negative is the Newton direction.

An MPM particle gets penalty rows against the nearest FEM surface triangle, at its signed distance minus a contact
thickness: the triangles have no volume of their own, and a coarse surface would otherwise let the material slip
between its vertices. Penetration is allowed, in proportion to the contact force over the penalty stiffness. The rows
act on the particle displacement relative to the displacement of its closest point, the barycentric combination of the
triangle's vertex displacements over the whole substep.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import CONTACT_ROWS, PYRAMID_ROWS, WALL_ROWS, MpmParticles
from genesis.engine.solvers.mpm_func import (
    contact_band_width,
    particle_inv_weight,
    particle_of_sorted,
    q1_stencil,
    stencil_node_slots,
)
from genesis.utils import array_class

from .mpm_rigid_func import EMPTY_BOX_HALF, particle_stencil

# Smallest friction coefficient of a contact row, as Rigid floors its contacts
MIN_FRICTION = 1e-2
# Distances to two triangles within this fraction of the grid spacing are those to a closest point they share
TIE_DIST_RATIO = 1e-6


# =============================================== Layer 0: geometry ===============================================


@qd.func
def closest_point_barycentric(p, a, b, c):
    """Barycentric coordinates of the point of triangle abc closest to p (Ericson, Real-Time Collision Detection,
    5.1.5)."""
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
def is_in_element(fem: qd.template(), i_e, i_b, x, f):
    """Whether point x lies in tet i_e at frame f, by its barycentric coordinates."""
    v3 = fem.elements_v[f, fem.elements_i[i_e].el2v[3], i_b].pos
    D = qd.Matrix.cols(
        [fem.elements_v[f, fem.elements_i[i_e].el2v[k], i_b].pos - v3 for k in qd.static(range(3))]
    )
    is_inside = False
    if qd.abs(D.determinant()) > gs.EPS:
        lam = D.inverse() @ (x - v3)
        is_inside = (lam >= 0.0).all() and lam.sum() <= 1.0
    return is_inside


@qd.func
def triangle_point_disp(fem: qd.template(), i_s, i_b, bary, f):
    """Displacement over the substep of the point of surface triangle i_s at barycentric coordinates bary."""
    disp = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        i_v = fem.surface[i_s].tri2v[k]
        disp += bary[k] * (fem.elements_v[f + 1, i_v, i_b].pos - fem.elements_v[f, i_v, i_b].pos)
    return disp


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class FemTriangleQuery:
    """The LBVH over the FEM surface triangles of every environment (see genesis.engine.bvh), and what the filter of
    its queries reads: the FEM solver, the thickness, and the query point, frame and nearest triangle of each sorted
    particle in `newton`."""

    def __init__(self, fem, newton, params, n_envs):
        self.fem = fem
        self.newton = newton
        self.params = params
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, fem.n_surfaces)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        # Every query keeps its nearest triangle in `newton` and records no pair
        self.results = array_class.get_bvh_query_results(1)

    def build(self, f):
        """Fit the leaf boxes to the triangles at the substep start and build the trees."""
        kernel_fit_triangle_leaves(self, f)
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)


@qd.kernel
def kernel_fit_triangle_leaves(query: qd.template(), f: qd.i32):
    """The box of every surface triangle with the element behind it, so that a particle that penetrated the surface
    element still finds the triangle, grown by the contact thickness and by twice the distance the fastest vertex of
    the element travels over the substep. An inactive triangle gets an empty box with a finite center, for the morton
    code."""
    dt = query.params.substep_dt[()]
    thickness = query.params.fem_contact_thickness[()]
    for i_b, i_s in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        if query.fem.surface[i_s].active:
            lo = gu.qd_vec3(qd.math.inf)
            hi = gu.qd_vec3(-qd.math.inf)
            speed = gs.qd_float(0.0)
            for k in qd.static(range(4)):
                i_v = query.fem.elements_i[query.fem.surface[i_s].tri2el].el2v[k]
                lo = qd.min(lo, query.fem.elements_v[f, i_v, i_b].pos)
                hi = qd.max(hi, query.fem.elements_v[f, i_v, i_b].pos)
                speed = qd.max(speed, query.fem.elements_v[f, i_v, i_b].vel.norm())
            margin = thickness + 2.0 * dt * speed
            query.leaves.aabbs_min[i_b, i_s] = lo - margin
            query.leaves.aabbs_max[i_b, i_s] = hi + margin
        else:
            query.leaves.aabbs_min[i_b, i_s] = qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
            query.leaves.aabbs_max[i_b, i_s] = qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)


@qd.func
def func_nearest_triangle_filter(i_t: int, i_a: int, i_q: int, query: qd.template()) -> bool:
    """The narrow phase of a box query of sorted particle i_q in environment i_t: keep the nearest surface triangle in
    `newton.query_idx`, with its gap in `query_phi`, its signed distance outward along the triangle normal minus the
    thickness, the barycentric coordinates of the closest point and the contact normal, the direction from it to a
    particle outside, and record no pair.

    The nearest triangle is that of the smallest unsigned distance, and the one of its normal most aligned with the
    direction to the particle among those whose closest point is shared, on an edge or a vertex: the sign of the
    distance to any other triangle of a closed surface may be wrong, as the bottom edge of a side face of a box seen
    from below its bottom face.

    A particle behind the plane of a triangle is inside the body only if it lies in the element behind the triangle,
    whose box the triangle's leaf spans; behind any other found triangle, it may be far outside, when the grown box of
    that triangle reaches it and the box of its nearest one does not, so the triangle is dropped. Penetration deeper
    than the surface elements gives no row."""
    f = query.newton.query_frame[()]
    x = qd.Vector([query.newton.query_pos[3 * i_q + d] for d in qd.static(range(3))], dt=gs.qd_float)
    a = query.fem.elements_v[f, query.fem.surface[i_a].tri2v[0], i_t].pos
    b = query.fem.elements_v[f, query.fem.surface[i_a].tri2v[1], i_t].pos
    c = query.fem.elements_v[f, query.fem.surface[i_a].tri2v[2], i_t].pos
    bary = closest_point_barycentric(x, a, b, c)
    d = x - (bary[0] * a + bary[1] * b + bary[2] * c)
    dist = d.norm()
    n_tri = gu.qd_normalize((b - a).cross(c - a), gs.EPS)
    side = gs.qd_float(1.0)
    is_valid = True
    if d.dot(n_tri) < 0.0:
        side = -1.0
        is_valid = is_in_element(query.fem, query.fem.surface[i_a].tri2el, i_t, x, f)
    align = qd.abs(d.dot(n_tri)) / qd.max(dist, gs.EPS)
    best_dist = query.newton.query_dist[i_q]
    is_tie = qd.abs(dist - best_dist) <= TIE_DIST_RATIO * query.params.dx[()]
    if is_valid and ((dist < best_dist and not is_tie) or (is_tie and align > query.newton.query_align[i_q])):
        normal = n_tri
        if side > 0.0 and dist > gs.EPS:
            normal = d / dist
        query.newton.query_dist[i_q] = dist
        query.newton.query_align[i_q] = align
        query.newton.query_phi[i_q] = side * dist - query.params.fem_contact_thickness[()]
        query.newton.query_idx[i_q] = i_a
        for k in qd.static(range(3)):
            query.newton.query_bary[3 * i_q + k] = bary[k]
            query.newton.query_normal[3 * i_q + k] = normal[k]
    return True


# ============================================== Layer 1: contact rows ==============================================


@qd.func(requires_top_level=True)
def func_build_fem_rows(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    fem: qd.template(),  # FEMSolver
    params: qd.template(),  # MpmParams
    query: qd.template(),  # FemTriangleQuery, its trees built for the substep
    f,
    dt,
):
    """Pyramid contact rows of every valid particle without Rigid rows against the nearest FEM surface triangle within
    its contact range or short of it by less than `contact_band_width`, after the wall and Rigid rows. The nearest
    triangle is found by a box query of the triangle trees: the particle's box spans its band at its own speed, the
    grown triangle boxes the thickness and the vertex motion, so together they cover the band of the relative velocity.

    The gap phi is the signed distance to the triangle, outward along its normal, minus
    `params.fem_contact_thickness`. Every row of the friction pyramid J = n +- mu d is a penalty on the gap at the end
    of the substep, phi + J.r for the relative displacement r: its target is -phi and its stiffness
    `params.fem_contact_stiffness` times the row's effective mass, the inverse of the particle's inverse weight in the
    implicit system plus that of the triangle point, sum_k b_k^2 / m_k. The row energy depends on the end positions
    alone, so it adds no energy the way a reference acceleration from the start gap does; the tangential part of a
    pyramid row is dropped with the rows of the next substep, which makes it a dissipative friction bounded by the cone.
    The friction coefficient is the element's."""
    for _ in range(1):
        newton.query_frame[()] = f
    qd.loop_config(name="mpm_build_fem_rows")
    for s in range(grid.n_valid[()]):
        if newton.contact_link[s] < 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            for d in qd.static(range(3)):
                newton.query_pos[3 * s + d] = x_p[d]
            newton.query_phi[s] = float("inf")
            newton.query_idx[s] = -1
            newton.query_dist[s] = float("inf")
            newton.query_align[s] = -1.0
            reach = contact_band_width(params, v_p.norm(), dt)
            func_bvh_query_aabb(
                i_b, s, x_p - reach, x_p + reach, query.tree, query.results, func_nearest_triangle_filter, query
            )
            best_tri = newton.query_idx[s]
            best_phi = newton.query_phi[s]
            best_bary = qd.Vector([newton.query_bary[3 * s + k] for k in qd.static(range(3))], dt=gs.qd_float)
            best_normal = qd.Vector([newton.query_normal[3 * s + k] for k in qd.static(range(3))], dt=gs.qd_float)
            if best_tri >= 0:
                v_q = qd.Vector.zero(gs.qd_float, 3)
                inv_mass = gs.qd_float(0.0)
                for k in qd.static(range(3)):
                    i_v = fem.surface[best_tri].tri2v[k]
                    v_q += best_bary[k] * fem.elements_v[f, i_v, i_b].vel
                    inv_mass += best_bary[k] * best_bary[k] / fem.elements_v_info[i_v].mass
                v_rel = v_p - v_q
                if best_phi < contact_band_width(params, -best_normal.dot(v_rel), dt):
                    friction = qd.max(fem.elements_i[fem.surface[best_tri].tri2el].friction_mu, MIN_FRICTION)
                    slots = stencil_node_slots(grid, env, cell)
                    w, grad_w = q1_stencil(x_p * params.inv_dx[()] - cell.cast(gs.qd_float), params.inv_dx[()])
                    invweight = particle_inv_weight(grid, slots, w, best_normal) + inv_mass
                    D = params.fem_contact_stiffness[()] / invweight
                    d1, d2 = gu.qd_orthogonals(best_normal)
                    for i in qd.static(range(PYRAMID_ROWS)):
                        d = d1
                        if qd.static(i >= 2):
                            d = d2
                        J = best_normal + (qd.static(2 * (i % 2) - 1) * friction) * d
                        k = CONTACT_ROWS * s + WALL_ROWS + i
                        for e in qd.static(range(3)):
                            newton.contact_dir[3 * k + e] = J[e]
                        newton.contact_D[k] = D
                        newton.contact_t[k] = -best_phi
                    newton.contact_tri[s] = best_tri
                    for k in qd.static(range(3)):
                        newton.contact_bary[3 * s + k] = best_bary[k]


@qd.func(requires_top_level=True)
def func_fem_point_disp(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    fem: qd.template(),  # FEMSolver
    params: qd.template(),  # MpmParams
    f,
):
    """Displacement of the contact point of every particle with FEM rows at the current FEM iterate."""
    qd.loop_config(name="mpm_fem_point_disp")
    for s in range(grid.n_valid[()]):
        i_s = newton.contact_tri[s]
        if i_s >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            bary = qd.Vector([newton.contact_bary[3 * s + k] for k in qd.static(range(3))], dt=gs.qd_float)
            dr = triangle_point_disp(fem, i_s, i_b, bary, f)
            for d in qd.static(range(3)):
                newton.contact_dr[3 * s + d] = dr[d]


@qd.func(requires_top_level=True)
def func_fem_contact_gradient(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    fem: qd.template(),  # FEMSolver
    params: qd.template(),  # MpmParams
):
    """The FEM rows' share of the FEM gradient and of its preconditioner, after FEM's own force and diagonal blocks
    and the MPM assembly that staged the row gradients g and Hessians H: the closest point's vertices take -b_k g,
    entered in FEM's force, minus the gradient over dt^2, and b_k^2 H over dt^2 on their diagonal block.

    The rows of the many particles against one triangle are far stiffer than the inertia of its vertices, so a FEM
    preconditioner without them would leave the joint PCG stopping on a direction that barely moves FEM."""
    inv_dt_sq = 1.0 / (params.substep_dt[()] * params.substep_dt[()])
    qd.loop_config(name="mpm_fem_contact_gradient")
    for s in range(grid.n_valid[()]):
        i_s = newton.contact_tri[s]
        if i_s >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            g = qd.Vector([newton.g_rigid[3 * s + d] for d in qd.static(range(3))], dt=gs.qd_float)
            H = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i, j in qd.static(qd.ndrange(3, 3)):
                H[i, j] = newton.h_rigid[9 * s + 3 * i + j]
            for k in qd.static(range(3)):
                i_v = fem.surface[i_s].tri2v[k]
                b_k = newton.contact_bary[3 * s + k]
                qd.atomic_add(fem.elements_v_energy[i_b, i_v].force, b_k * inv_dt_sq * g)
                fem.pcg_state_v[i_b, i_v].diag3x3 += (b_k * b_k * inv_dt_sq) * H


@qd.func(requires_top_level=True)
def func_fem_factor(fem: qd.template()):
    """FEM's preconditioner, the inverse of every vertex's diagonal block, once the contact rows added theirs."""
    qd.loop_config(name="mpm_fem_factor")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].prec = fem.pcg_state_v[i_b, i_v].diag3x3.inverse()


@qd.func
def fem_contact_terms(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    fem: qd.template(),  # FEMSolver
    params: qd.template(),  # MpmParams
    f,
    s,
):
    """The row Hessian H of sorted particle s, its stencil, its trial displacement dxp = sum_a w_a p_u,a along the
    direction and the displacement dq = sum_k b_k p_v,k of its closest point."""
    ws = linear.pcg
    i_p, i_b, slots, w = particle_stencil(particles, grid, params, f, s)
    H = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        H[i, j] = newton.h_rigid[9 * s + 3 * i + j]
    dxp = qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(8)):
        row = 3 * (linear.row_offset + slots[a])
        dxp += w[a] * qd.Vector([ws.p[row + d] for d in qd.static(range(3))], dt=gs.qd_float)
    dq = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        dq += newton.contact_bary[3 * s + k] * fem.pcg_state_v[i_b, fem.surface[newton.contact_tri[s]].tri2v[k]].p
    return i_b, slots, w, H, dxp, dq


@qd.func(requires_top_level=True)
def func_fem_contact_matvec(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    fem: qd.template(),  # FEMSolver
    params: qd.template(),  # MpmParams
    f,
):
    """The FEM rows' cross term on the MPM rows, after func_linear_pcg_spmv: Ap_u,a -= w_a H dq, and p.Ap takes
    p_u.(-sum w_a H dq)."""
    ws = linear.pcg
    qd.loop_config(name="mpm_fem_matvec_particles")
    for s in range(grid.n_valid[()]):
        if newton.contact_tri[s] >= 0 and ws.converged[()] == 0:
            i_b, slots, w, H, dxp, dq = fem_contact_terms(particles, grid, newton, linear, fem, params, f, s)
            H_dq = H @ dq
            for a in qd.static(range(8)):
                row = 3 * (linear.row_offset + slots[a])
                for d in qd.static(range(3)):
                    qd.atomic_add(ws.ap[row + d], -w[a] * H_dq[d])
            qd.atomic_add(ws.p_dot_ap[()], -dxp.dot(H_dq))


@qd.func(requires_top_level=True)
def func_fem_contact_matvec_post(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    fem: qd.template(),  # FEMSolver
    params: qd.template(),  # MpmParams
    f,
):
    """The FEM rows' term on the FEM rows, after FEM's own operator overwrote its Ap: Ap_v,k += b_k H (dq - dxp)."""
    ws = linear.pcg
    qd.loop_config(name="mpm_fem_matvec_vertices")
    for s in range(grid.n_valid[()]):
        if newton.contact_tri[s] >= 0 and ws.converged[()] == 0:
            i_b, slots, w, H, dxp, dq = fem_contact_terms(particles, grid, newton, linear, fem, params, f, s)
            h = H @ (dq - dxp)
            for k in qd.static(range(3)):
                i_v = fem.surface[newton.contact_tri[s]].tri2v[k]
                qd.atomic_add(fem.pcg_state_v[i_b, i_v].Ap, newton.contact_bary[3 * s + k] * h)


# ============================================ Layer 1: the FEM block ============================================


@qd.func(requires_top_level=True)
def func_fem_pcg_init(fem: qd.template(), dt):
    """The FEM part of the PCG start from 0: r = grad = -dt^2 force, solution and Ap cleared."""
    qd.loop_config(name="mpm_fem_pcg_init")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].r = -dt * dt * fem.elements_v_energy[i_b, i_v].force
        fem.pcg_state_v[i_b, i_v].x = qd.Vector.zero(gs.qd_float, 3)
        fem.pcg_state_v[i_b, i_v].Ap = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_fem_precondition(fem: qd.template(), dt):
    """z = M^-1 r by FEM's inverse 3x3 diagonal blocks, scaled to the joint energy. A vertex FEM holds at its target
    gets z = 0, so that its direction stays 0 and no contact row moves it."""
    qd.loop_config(name="mpm_fem_precondition")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        z = (fem.pcg_state_v[i_b, i_v].prec @ fem.pcg_state_v[i_b, i_v].r) / (dt * dt)
        if qd.static(fem._enable_vertex_constraints):
            if fem.vertex_constraints.is_constrained[i_v, i_b]:
                z = qd.Vector.zero(gs.qd_float, 3)
        fem.pcg_state_v[i_b, i_v].z = z


@qd.func(requires_top_level=True)
def func_fem_pcg_init_direction(fem: qd.template(), ws: qd.template()):
    """p = z over the FEM vertices, and r.z added to rz before the threshold is re-derived, after the primary's
    func_init_linear_pcg_direction."""
    qd.loop_config(name="mpm_fem_pcg_init_direction")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].p = fem.pcg_state_v[i_b, i_v].z
        qd.atomic_add(ws.rz_slots[ws.parity[()]], fem.pcg_state_v[i_b, i_v].r.dot(fem.pcg_state_v[i_b, i_v].z))
    for _ in range(1):
        ws.tol_sq[()] = ws.tol_rate[()] * qd.abs(ws.rz_slots[ws.parity[()]])


@qd.func(requires_top_level=True)
def func_fem_matvec(fem: qd.template(), dt):
    """Ap = dt^2 H p over the FEM vertices with FEM's own operator, before the FEM rows add their term."""
    fem.compute_Ap()
    qd.loop_config(name="mpm_fem_matvec_scale")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].Ap = dt * dt * fem.pcg_state_v[i_b, i_v].Ap


@qd.func(requires_top_level=True)
def func_fem_matvec_dot(fem: qd.template(), ws: qd.template()):
    qd.loop_config(name="mpm_fem_matvec_dot")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if ws.converged[()] == 0:
            qd.atomic_add(ws.p_dot_ap[()], fem.pcg_state_v[i_b, i_v].p.dot(fem.pcg_state_v[i_b, i_v].Ap))


@qd.func(requires_top_level=True)
def func_fem_pcg_update_xr(fem: qd.template(), ws: qd.template()):
    """x += alpha p and r -= alpha Ap over the FEM vertices, with the alpha of func_linear_pcg_update_xr."""
    qd.loop_config(name="mpm_fem_pcg_update_xr")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if ws.converged[()] == 0:
            p_dot_ap = ws.p_dot_ap[()]
            alpha = ws.rz_slots[ws.parity[()]] / p_dot_ap if p_dot_ap != 0.0 else gs.qd_float(0.0)
            fem.pcg_state_v[i_b, i_v].x = fem.pcg_state_v[i_b, i_v].x + alpha * fem.pcg_state_v[i_b, i_v].p
            fem.pcg_state_v[i_b, i_v].r = fem.pcg_state_v[i_b, i_v].r - alpha * fem.pcg_state_v[i_b, i_v].Ap


@qd.func(requires_top_level=True)
def func_fem_pcg_dot_rz(fem: qd.template(), ws: qd.template()):
    """r.z over the FEM vertices into the rz_new slot, before the primary's func_linear_pcg_direction_update."""
    qd.loop_config(name="mpm_fem_pcg_dot_rz")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        if ws.converged[()] == 0:
            qd.atomic_add(
                ws.rz_slots[1 - ws.parity[()]], fem.pcg_state_v[i_b, i_v].r.dot(fem.pcg_state_v[i_b, i_v].z)
            )


@qd.func(requires_top_level=True)
def func_fem_pcg_direction(fem: qd.template(), ws: qd.template()):
    """p = z + beta p over the FEM vertices and Ap cleared, after func_linear_pcg_direction_update flipped the parity."""
    qd.loop_config(name="mpm_fem_pcg_direction")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.pcg_state_v[i_b, i_v].Ap = qd.Vector.zero(gs.qd_float, 3)
        if ws.converged[()] == 0:
            rz = ws.rz_slots[1 - ws.parity[()]]
            beta = ws.rz_slots[ws.parity[()]] / rz if rz != 0.0 else gs.qd_float(0.0)
            fem.pcg_state_v[i_b, i_v].p = fem.pcg_state_v[i_b, i_v].z + beta * fem.pcg_state_v[i_b, i_v].p


@qd.func(requires_top_level=True)
def func_fem_start_line_search(fem: qd.template(), control: qd.template(), f):
    """The line-search start point, and the largest component of the step -x into the Newton measure."""
    qd.loop_config(name="mpm_fem_start_line_search")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.linesearch_state_v[i_b, i_v].x_prev = fem.elements_v[f + 1, i_v, i_b].pos
        step = fem.pcg_state_v[i_b, i_v].x
        qd.atomic_max(control.max_disp[()], qd.max(qd.abs(step[0]), qd.abs(step[1]), qd.abs(step[2])))


@qd.func(requires_top_level=True)
def func_fem_step_forward(fem: qd.template(), control: qd.template(), f):
    qd.loop_config(name="mpm_fem_step_forward")
    for i_b, i_v in qd.ndrange(fem._B, fem.n_vertices):
        fem.elements_v[f + 1, i_v, i_b].pos = (
            fem.linesearch_state_v[i_b, i_v].x_prev - control.alpha[()] * fem.pcg_state_v[i_b, i_v].x
        )


@qd.func(requires_top_level=True)
def func_fem_energy(fem: qd.template(), control: qd.template(), f, is_trial, dt):
    """Add FEM's incremental potential at the positions f + 1 times dt^2 to the start or trial energy: the inertia and
    the linearized damping of its vertices, and the elastic energy and damping of its elements."""
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
