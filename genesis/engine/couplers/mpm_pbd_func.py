"""Coupling of the implicit MPM solver with PBD: the PBD particles as a secondary participant of the NewtonSolver with
their inertia alone, and the MPM-PBD contact rows as a coupling.

PBD has projected its own constraints for the substep before the solve: its particles hold their start position
`ipos` and the end position `pos` of the projection. The joint solve takes the end positions as unknowns, starting at
the projection x~, with the energy 1/2 m |x - x~|^2 of the inertia around it in the dt^2-scaled units of the MPM
energy; the contact rows push them off it. PBD's constraints stay with PBD, which projects them again the next
substep, so the coupling is strong in the contact and staggered in the PBD constraints. After the solve, a PBD
particle takes the velocity of its displacement over the substep.

An MPM particle gets penalty rows against the nearest PBD particle, a sphere of the PBD radius plus a contact
thickness. The rows act on the particle displacement relative to the PBD particle's displacement over the substep,
and solve the contact by the masses on both sides, so that a light cloth is held by a heavy block and the block does
not yield to it.
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


@qd.data_oriented
class PbdNewtonData:
    """The PBD rows of the joint solve, per reordered PBD particle and environment, three scalars per particle at
    3 (i_b n + i_q): the projection x~ of PBD's own substep, the gradient, the PCG residual, preconditioned residual,
    direction, operator image and solution, the line-search start, and the inverse of the 3 x 3 diagonal block, nine
    scalars per particle."""

    def __init__(self, pbd, n_envs):
        self.pbd = pbd
        self.n = max(pbd.n_particles, 1)
        size = 3 * self.n * n_envs
        for name in ("target", "grad", "r", "z", "p", "ap", "sol", "x_prev"):
            setattr(self, name, qd.ndarray(gs.qd_float, shape=(size,)))
        self.diag_inv = qd.ndarray(gs.qd_float, shape=(3 * size,))


# ================================================ Layer 0: access ================================================


@qd.func
def vec_at(a: qd.template(), k):
    return qd.Vector([a[3 * k + d] for d in qd.static(range(3))], dt=gs.qd_float)


@qd.func
def set_vec_at(a: qd.template(), k, v):
    for d in qd.static(range(3)):
        a[3 * k + d] = v[d]


@qd.func
def is_moving(pbd: qd.template(), i_q, i_b):
    """Whether the PBD particle is an unknown of the solve, active and free; an attached one keeps its position."""
    return pbd.particles_ng_reordered[i_q, i_b].active and pbd.particles_reordered[i_q, i_b].free


# ============================================== Layer 1: the PBD block ==============================================


@qd.func(requires_top_level=True)
def func_pbd_begin(pd: qd.template()):
    """The projection of PBD's substep as the target of the inertia, the end positions starting at it."""
    qd.loop_config(name="mpm_pbd_begin")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        set_vec_at(pd.target, i_b * pd.n + i_q, pd.pbd.particles_reordered[i_q, i_b].pos)


@qd.func(requires_top_level=True)
def func_pbd_gradient(pd: qd.template()):
    """The gradient m (x - x~) of the inertia and its diagonal block m I, before the contact rows add theirs."""
    qd.loop_config(name="mpm_pbd_gradient")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        grad = qd.Vector.zero(gs.qd_float, 3)
        m = gs.qd_float(1.0)
        if is_moving(pd.pbd, i_q, i_b):
            m = pd.pbd.particles_info_reordered[i_q, i_b].mass
            grad = m * (pd.pbd.particles_reordered[i_q, i_b].pos - vec_at(pd.target, k))
        set_vec_at(pd.grad, k, grad)
        for i, j in qd.static(qd.ndrange(3, 3)):
            pd.diag_inv[9 * k + 3 * i + j] = m if i == j else 0.0


@qd.func(requires_top_level=True)
def func_pbd_factor(pd: qd.template()):
    """The inverse of every particle's diagonal block, once the contact rows added theirs."""
    qd.loop_config(name="mpm_pbd_factor")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        A = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i, j in qd.static(qd.ndrange(3, 3)):
            A[i, j] = pd.diag_inv[9 * k + 3 * i + j]
        A_inv = A.inverse()
        for i, j in qd.static(qd.ndrange(3, 3)):
            pd.diag_inv[9 * k + 3 * i + j] = A_inv[i, j]


@qd.func(requires_top_level=True)
def func_pbd_pcg_init(pd: qd.template()):
    """The PBD part of the PCG start from 0: r = grad, solution and Ap cleared."""
    qd.loop_config(name="mpm_pbd_pcg_init")
    for i in range(pd.r.shape[0]):
        pd.r[i] = pd.grad[i]
        pd.sol[i] = 0.0
        pd.ap[i] = 0.0


@qd.func(requires_top_level=True)
def func_pbd_precondition(pd: qd.template()):
    """z = M^-1 r by the inverse diagonal blocks; an attached particle gets z = 0, so that it does not move."""
    qd.loop_config(name="mpm_pbd_precondition")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        z = qd.Vector.zero(gs.qd_float, 3)
        if is_moving(pd.pbd, i_q, i_b):
            A_inv = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i, j in qd.static(qd.ndrange(3, 3)):
                A_inv[i, j] = pd.diag_inv[9 * k + 3 * i + j]
            z = A_inv @ vec_at(pd.r, k)
        set_vec_at(pd.z, k, z)


@qd.func(requires_top_level=True)
def func_pbd_pcg_init_direction(pd: qd.template(), ws: qd.template()):
    """p = z over the PBD rows, and r.z added to rz before the threshold is re-derived."""
    qd.loop_config(name="mpm_pbd_pcg_init_direction")
    for i in range(pd.r.shape[0]):
        pd.p[i] = pd.z[i]
        qd.atomic_add(ws.rz_slots[ws.parity[()]], pd.r[i] * pd.z[i])
    for _ in range(1):
        ws.tol_sq[()] = ws.tol_rate[()] * qd.abs(ws.rz_slots[ws.parity[()]])


@qd.func(requires_top_level=True)
def func_pbd_matvec(pd: qd.template()):
    """Ap = m p over the PBD rows, before the contact rows add their term."""
    qd.loop_config(name="mpm_pbd_matvec")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        m = gs.qd_float(0.0)
        if is_moving(pd.pbd, i_q, i_b):
            m = pd.pbd.particles_info_reordered[i_q, i_b].mass
        set_vec_at(pd.ap, k, m * vec_at(pd.p, k))


@qd.func(requires_top_level=True)
def func_pbd_matvec_dot(pd: qd.template(), ws: qd.template()):
    qd.loop_config(name="mpm_pbd_matvec_dot")
    for i in range(pd.p.shape[0]):
        if ws.converged[()] == 0:
            qd.atomic_add(ws.p_dot_ap[()], pd.p[i] * pd.ap[i])


@qd.func(requires_top_level=True)
def func_pbd_pcg_update_xr(pd: qd.template(), ws: qd.template()):
    """x += alpha p and r -= alpha Ap over the PBD rows, with the alpha of func_linear_pcg_update_xr."""
    qd.loop_config(name="mpm_pbd_pcg_update_xr")
    for i in range(pd.p.shape[0]):
        if ws.converged[()] == 0:
            p_dot_ap = ws.p_dot_ap[()]
            alpha = ws.rz_slots[ws.parity[()]] / p_dot_ap if p_dot_ap != 0.0 else gs.qd_float(0.0)
            pd.sol[i] += alpha * pd.p[i]
            pd.r[i] -= alpha * pd.ap[i]


@qd.func(requires_top_level=True)
def func_pbd_pcg_dot_rz(pd: qd.template(), ws: qd.template()):
    """r.z over the PBD rows into the rz_new slot, before the primary's func_linear_pcg_direction_update."""
    qd.loop_config(name="mpm_pbd_pcg_dot_rz")
    for i in range(pd.r.shape[0]):
        if ws.converged[()] == 0:
            qd.atomic_add(ws.rz_slots[1 - ws.parity[()]], pd.r[i] * pd.z[i])


@qd.func(requires_top_level=True)
def func_pbd_pcg_direction(pd: qd.template(), ws: qd.template()):
    """p = z + beta p over the PBD rows and Ap cleared, after func_linear_pcg_direction_update flipped the parity."""
    qd.loop_config(name="mpm_pbd_pcg_direction")
    for i in range(pd.p.shape[0]):
        pd.ap[i] = 0.0
        if ws.converged[()] == 0:
            rz = ws.rz_slots[1 - ws.parity[()]]
            beta = ws.rz_slots[ws.parity[()]] / rz if rz != 0.0 else gs.qd_float(0.0)
            pd.p[i] = pd.z[i] + beta * pd.p[i]


@qd.func(requires_top_level=True)
def func_pbd_start_line_search(pd: qd.template(), control: qd.template()):
    """The line-search start point, and the largest component of the step -x into the Newton measure."""
    qd.loop_config(name="mpm_pbd_start_line_search")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        set_vec_at(pd.x_prev, k, pd.pbd.particles_reordered[i_q, i_b].pos)
        step = vec_at(pd.sol, k)
        qd.atomic_max(control.max_disp[()], qd.max(qd.abs(step[0]), qd.abs(step[1]), qd.abs(step[2])))


@qd.func(requires_top_level=True)
def func_pbd_step_forward(pd: qd.template(), control: qd.template()):
    qd.loop_config(name="mpm_pbd_step_forward")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        if is_moving(pd.pbd, i_q, i_b):
            k = i_b * pd.n + i_q
            pd.pbd.particles_reordered[i_q, i_b].pos = vec_at(pd.x_prev, k) - control.alpha[()] * vec_at(pd.sol, k)


@qd.func(requires_top_level=True)
def func_pbd_energy(pd: qd.template(), control: qd.template(), is_trial):
    """Add the inertia 1/2 m |x - x~|^2 of the PBD particles to the start or trial energy."""
    qd.loop_config(name="mpm_pbd_energy")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        if is_moving(pd.pbd, i_q, i_b):
            diff = pd.pbd.particles_reordered[i_q, i_b].pos - vec_at(pd.target, i_b * pd.n + i_q)
            energy = 0.5 * pd.pbd.particles_info_reordered[i_q, i_b].mass * diff.dot(diff)
            if is_trial:
                qd.atomic_add(control.energy_trial[()], energy)
            else:
                qd.atomic_add(control.energy_start[()], energy)


@qd.func(requires_top_level=True)
def func_pbd_finish(pd: qd.template(), dt):
    """The velocity of every moving PBD particle from its displacement over the substep."""
    qd.loop_config(name="mpm_pbd_finish")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        if is_moving(pd.pbd, i_q, i_b):
            pd.pbd.particles_reordered[i_q, i_b].vel = (
                pd.pbd.particles_reordered[i_q, i_b].pos - pd.pbd.particles_reordered[i_q, i_b].ipos
            ) / dt


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class PbdParticleQuery:
    """The LBVH over the PBD particles of every environment (see genesis.engine.bvh), in PBD's reordered layout, and
    what the filter of its queries reads: the PBD solver, the contact range, and the query point and nearest PBD
    particle of each sorted MPM particle in `newton`."""

    def __init__(self, pbd, newton, params, n_envs):
        self.pbd = pbd
        self.newton = newton
        self.params = params
        self.radius = pbd.particle_radius
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(pbd.n_particles, 1))
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        # Every query keeps its nearest PBD particle in `newton` and records no pair
        self.results = array_class.get_bvh_query_results(1)

    def build(self):
        """Fit the leaf boxes to the PBD particles' sweeps over the substep and build the trees."""
        kernel_fit_pbd_leaves(self)
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)


@qd.kernel
def kernel_fit_pbd_leaves(query: qd.template()):
    """The box of every PBD particle over its sweep from its start to its projected end position, grown by its radius
    and the contact thickness. An inactive particle gets an empty box with a finite center, for the morton code."""
    reach = qd.static(query.radius) + query.params.pbd_contact_thickness[()]
    for i_b, i_q in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        if query.pbd.particles_ng_reordered[i_q, i_b].active:
            x0 = query.pbd.particles_reordered[i_q, i_b].ipos
            x1 = query.pbd.particles_reordered[i_q, i_b].pos
            query.leaves.aabbs_min[i_b, i_q] = qd.min(x0, x1) - reach
            query.leaves.aabbs_max[i_b, i_q] = qd.max(x0, x1) + reach
        else:
            query.leaves.aabbs_min[i_b, i_q] = qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
            query.leaves.aabbs_max[i_b, i_q] = qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)


@qd.func
def func_nearest_pbd_filter(i_t: int, i_a: int, i_q: int, query: qd.template()) -> bool:
    """The narrow phase of a box query of sorted MPM particle i_q in environment i_t: keep the PBD particle of the
    smallest gap at the substep start, its distance minus its radius and the contact thickness, in
    `newton.query_phi` / `query_idx`, with the normal from it to the MPM particle, and record no pair."""
    x = qd.Vector([query.newton.query_pos[3 * i_q + d] for d in qd.static(range(3))], dt=gs.qd_float)
    d = x - query.pbd.particles_reordered[i_a, i_t].ipos
    dist = d.norm()
    phi = dist - qd.static(query.radius) - query.params.pbd_contact_thickness[()]
    if phi < query.newton.query_phi[i_q] and dist > gs.EPS:
        query.newton.query_phi[i_q] = phi
        query.newton.query_idx[i_q] = i_a
        for k in qd.static(range(3)):
            query.newton.query_normal[3 * i_q + k] = d[k] / dist
    return True


# ============================================== Layer 1: contact rows ==============================================


@qd.func(requires_top_level=True)
def func_build_pbd_rows(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    pbd: qd.template(),  # PBDSolver
    params: qd.template(),  # MpmParams
    query: qd.template(),  # PbdParticleQuery, its trees built for the substep
    f,
    dt,
):
    """Pyramid penalty rows of every valid particle without Rigid or FEM rows against the nearest PBD particle within
    its contact range or short of it by less than `contact_band_width`.

    The particle's box spans its band at its own speed, the PBD leaves the sweep, radius and thickness, so together
    they cover the band of the relative velocity. Every row of the pyramid J = n +- mu d is a penalty on the gap at the
    end of the substep, phi + J.r for the relative displacement r: its target is -phi and its stiffness
    `params.pbd_contact_stiffness` times the row's effective mass, the inverse of the particle's inverse weight in the
    implicit system plus that of the PBD particle."""
    friction = params.pbd_contact_friction[()]
    qd.loop_config(name="mpm_build_pbd_rows")
    for s in range(grid.n_valid[()]):
        if newton.contact_link[s] < 0 and newton.contact_tri[s] < 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            for d in qd.static(range(3)):
                newton.query_pos[3 * s + d] = x_p[d]
            newton.query_phi[s] = float("inf")
            newton.query_idx[s] = -1
            reach = contact_band_width(params, v_p.norm(), dt)
            func_bvh_query_aabb(
                i_b, s, x_p - reach, x_p + reach, query.tree, query.results, func_nearest_pbd_filter, query
            )
            i_q = newton.query_idx[s]
            if i_q >= 0:
                phi = newton.query_phi[s]
                normal = qd.Vector([newton.query_normal[3 * s + k] for k in qd.static(range(3))], dt=gs.qd_float)
                v_rel = v_p - pbd.particles_reordered[i_q, i_b].vel
                if phi < contact_band_width(params, -normal.dot(v_rel), dt):
                    slots = stencil_node_slots(grid, env, cell)
                    w, grad_w = q1_stencil(x_p * params.inv_dx[()] - cell.cast(gs.qd_float), params.inv_dx[()])
                    invweight = particle_inv_weight(grid, slots, w, normal)
                    if is_moving(pbd, i_q, i_b):
                        invweight += 1.0 / pbd.particles_info_reordered[i_q, i_b].mass
                    D = params.pbd_contact_stiffness[()] / invweight
                    d1, d2 = gu.qd_orthogonals(normal)
                    for i in qd.static(range(PYRAMID_ROWS)):
                        d = d1
                        if qd.static(i >= 2):
                            d = d2
                        J = normal + (qd.static(2 * (i % 2) - 1) * friction) * d
                        k = CONTACT_ROWS * s + WALL_ROWS + i
                        for e in qd.static(range(3)):
                            newton.contact_dir[3 * k + e] = J[e]
                        newton.contact_D[k] = D
                        newton.contact_t[k] = -phi
                    newton.contact_pbd[s] = i_q


@qd.func(requires_top_level=True)
def func_pbd_point_disp(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    pbd: qd.template(),  # PBDSolver
    params: qd.template(),  # MpmParams
):
    """Displacement over the substep of the PBD particle of every particle with PBD rows, at the current iterate."""
    qd.loop_config(name="mpm_pbd_point_disp")
    for s in range(grid.n_valid[()]):
        i_q = newton.contact_pbd[s]
        if i_q >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            dq = pbd.particles_reordered[i_q, i_b].pos - pbd.particles_reordered[i_q, i_b].ipos
            for d in qd.static(range(3)):
                newton.contact_dr[3 * s + d] = dq[d]


@qd.func(requires_top_level=True)
def func_pbd_contact_gradient(
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    pd: qd.template(),  # PbdNewtonData
    params: qd.template(),  # MpmParams
):
    """The PBD rows' share of the PBD gradient and diagonal blocks, after the inertia's and the MPM assembly that
    staged the row gradients g and Hessians H: the PBD particle takes -g and H."""
    qd.loop_config(name="mpm_pbd_contact_gradient")
    for s in range(grid.n_valid[()]):
        i_q = newton.contact_pbd[s]
        if i_q >= 0:
            i_p, i_b, env, cell = particle_of_sorted(grid, params, s)
            k = i_b * pd.n + i_q
            for d in qd.static(range(3)):
                qd.atomic_add(pd.grad[3 * k + d], -newton.g_rigid[3 * s + d])
            for i, j in qd.static(qd.ndrange(3, 3)):
                qd.atomic_add(pd.diag_inv[9 * k + 3 * i + j], newton.h_rigid[9 * s + 3 * i + j])


@qd.func
def pbd_contact_terms(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    pd: qd.template(),  # PbdNewtonData
    params: qd.template(),  # MpmParams
    f,
    s,
):
    """The row Hessian H of sorted particle s, its stencil, its trial displacement dxp = sum_a w_a p_u,a along the
    direction, the PBD row index and the PBD particle's direction dq."""
    ws = linear.pcg
    i_p, i_b, slots, w = particle_stencil(particles, grid, params, f, s)
    H = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        H[i, j] = newton.h_rigid[9 * s + 3 * i + j]
    dxp = qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(8)):
        row = 3 * (linear.row_offset + slots[a])
        dxp += w[a] * qd.Vector([ws.p[row + d] for d in qd.static(range(3))], dt=gs.qd_float)
    k = i_b * pd.n + newton.contact_pbd[s]
    return slots, w, H, dxp, k, vec_at(pd.p, k)


@qd.func(requires_top_level=True)
def func_pbd_contact_matvec(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    pd: qd.template(),  # PbdNewtonData
    params: qd.template(),  # MpmParams
    f,
):
    """The PBD rows' cross term on the MPM rows, after func_linear_pcg_spmv: Ap_u,a -= w_a H dq, and p.Ap takes
    p_u.(-sum w_a H dq)."""
    ws = linear.pcg
    qd.loop_config(name="mpm_pbd_matvec_particles")
    for s in range(grid.n_valid[()]):
        if newton.contact_pbd[s] >= 0 and ws.converged[()] == 0:
            slots, w, H, dxp, k, dq = pbd_contact_terms(particles, grid, newton, linear, pd, params, f, s)
            H_dq = H @ dq
            for a in qd.static(range(8)):
                row = 3 * (linear.row_offset + slots[a])
                for d in qd.static(range(3)):
                    qd.atomic_add(ws.ap[row + d], -w[a] * H_dq[d])
            qd.atomic_add(ws.p_dot_ap[()], -dxp.dot(H_dq))


@qd.func(requires_top_level=True)
def func_pbd_contact_matvec_post(
    particles: MpmParticles,
    grid: qd.template(),  # MpmGrid
    newton: qd.template(),  # MpmNewton
    linear: qd.template(),  # MpmLinearSystem
    pd: qd.template(),  # PbdNewtonData
    params: qd.template(),  # MpmParams
    f,
):
    """The PBD rows' term on the PBD rows, after the inertia overwrote their Ap: Ap_q += H (dq - dxp)."""
    ws = linear.pcg
    qd.loop_config(name="mpm_pbd_matvec_pbd")
    for s in range(grid.n_valid[()]):
        if newton.contact_pbd[s] >= 0 and ws.converged[()] == 0:
            slots, w, H, dxp, k, dq = pbd_contact_terms(particles, grid, newton, linear, pd, params, f, s)
            h = H @ (dq - dxp)
            for d in qd.static(range(3)):
                qd.atomic_add(pd.ap[3 * k + d], h[d])


# =============================================== Layer 2: participants ===============================================


@qd.data_oriented
class PbdNewtonParticipant:
    """Device stages of the PBD end positions in the joint solve."""

    def __init__(self, pd: PbdNewtonData):
        self.pd = pd

    @qd.func(requires_top_level=True)
    def func_evaluate(self, f):
        pass

    @qd.func(requires_top_level=True)
    def func_gradient(self, f):
        func_pbd_gradient(self.pd)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        func_pbd_factor(self.pd)

    @qd.func(requires_top_level=True)
    def func_pcg_init(self):
        func_pbd_pcg_init(self.pd)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        func_pbd_precondition(self.pd)

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, ws: qd.template()):
        func_pbd_pcg_init_direction(self.pd, ws)

    @qd.func(requires_top_level=True)
    def func_matvec(self):
        func_pbd_matvec(self.pd)

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, ws: qd.template()):
        func_pbd_matvec_dot(self.pd, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, ws: qd.template()):
        func_pbd_pcg_update_xr(self.pd, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, ws: qd.template()):
        func_pbd_pcg_dot_rz(self.pd, ws)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, ws: qd.template()):
        func_pbd_pcg_direction(self.pd, ws)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f):
        func_pbd_start_line_search(self.pd, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f):
        func_pbd_step_forward(self.pd, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f, is_trial):
        func_pbd_energy(self.pd, control, is_trial)


@qd.data_oriented
class MpmPbdContact:
    """Device stages of the MPM-PBD contact rows in the joint solve: their share of the PBD gradient, their cross
    terms in both blocks of the operator, and the PBD particle displacement at every iterate."""

    def __init__(self, mpm, pbd_participant: PbdNewtonParticipant, query: PbdParticleQuery):
        self.particles = mpm.particles
        self.grid = mpm.grid
        self.newton = mpm.newton
        self.linear = mpm.linear
        self.params = mpm.params
        self.pd = pbd_participant.pd
        self.query = query

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f):
        func_pbd_contact_gradient(self.grid, self.newton, self.pd, self.params)

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f):
        func_pbd_contact_matvec(self.particles, self.grid, self.newton, self.linear, self.pd, self.params, f)

    @qd.func(requires_top_level=True)
    def func_contact_matvec_post(self, f):
        func_pbd_contact_matvec_post(self.particles, self.grid, self.newton, self.linear, self.pd, self.params, f)

    @qd.func(requires_top_level=True)
    def func_after_step(self, f):
        func_pbd_point_disp(self.grid, self.newton, self.pd.pbd, self.params)


@qd.kernel
def kernel_pbd_begin(pd: qd.template()):
    func_pbd_begin(pd)


@qd.kernel
def kernel_pbd_finish(pd: qd.template(), params: qd.template()):
    func_pbd_finish(pd, params.substep_dt[()])


@qd.kernel
def kernel_pbd_rows(contact: qd.template(), f: qd.i32):
    func_build_pbd_rows(
        contact.particles,
        contact.grid,
        contact.newton,
        contact.pd.pbd,
        contact.params,
        contact.query,
        f,
        contact.params.substep_dt[()],
    )
    contact.func_after_step(f)


class PbdNewtonHost:
    """Host stages of PBD around the joint solve: the projection of its substep as the inertia target before, its
    velocities from the solved positions after."""

    def __init__(self, pd: PbdNewtonData, params):
        self._pd = pd
        self._params = params

    def newton_prepare(self, f):
        kernel_pbd_begin(self._pd)

    def newton_finish(self, f):
        kernel_pbd_finish(self._pd, self._params)

    def newton_after(self, f):
        pass


class MpmPbdContactHost:
    """Host stages of the MPM-PBD contact rows before the joint solve: the PBD trees, then the rows and the PBD
    particle displacement at the start point."""

    def __init__(self, device: MpmPbdContact):
        self._device = device

    def newton_prepare(self, f):
        self._device.query.build()
        kernel_pbd_rows(self._device, f)

    def newton_finish(self, f):
        pass

    def newton_after(self, f):
        pass
