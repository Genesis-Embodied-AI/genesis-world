"""Coupling of the implicit material point method (MPM) solver with position based dynamics (PBD).

The PBD particles enter the NewtonSolver as a participant with their inertia alone, and the MPM-PBD contact
rows as a coupling. PBD projects its own constraints for the substep before the solve, so its particles hold their start
position `ipos` and the end position `pos` of the projection. The joint solve takes the end positions as unknowns,
starting at the projection x~, with the inertia energy 1/2 m |x - x~|^2 around it in the dt^2-scaled units of the MPM
energy. The contact rows push them off it. PBD's constraints stay with PBD, which projects them again the next substep,
so the coupling is strong in the contact and staggered in the PBD constraints. After the solve, a PBD particle takes the
velocity of its displacement over the substep.

An MPM particle gets penalty rows against the nearest PBD particle, a sphere of the PBD radius plus a contact thickness.
The rows act on the particle displacement relative to the PBD particle's displacement over the substep, and solve the
contact by the masses on both sides, so that a heavy block holds a light cloth and the cloth barely moves the block.
"""

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import CONTACT_ROWS, PYRAMID_ROWS, WALL_ROWS, MpmParticles, MpmParticlesInfo
from genesis.engine.solvers.mpm_func import (
    EMPTY_BOX_HALF,
    contact_band_width,
    contact_invweight,
    particle_of_sorted,
    particle_stencil,
)
from genesis.engine.solvers.newton_solver import NewtonHost, kernel_assemble_participant
from genesis.utils import array_class

# Obstacle candidates of a PBD particle over a substep: the nearest OBSTACLE_NEIGHBORS MPM particles within
# OBSTACLE_REACH contact distances of its start or predicted position
OBSTACLE_NEIGHBORS = 32
OBSTACLE_REACH = 1.5


@qd.data_oriented
class PbdNewtonData:
    """The PBD rows of the joint solve, per reordered PBD particle and environment.

    Every vector holds three scalars per particle at 3 (i_b n + i_q): the projection x~ of PBD's own substep, the
    gradient, the preconditioned conjugate gradient (PCG) residual, preconditioned residual, direction, operator image
    and solution, and the line-search start. The inverse of the 3 x 3 diagonal block holds nine scalars per particle."""

    def __init__(self, pbd, n_envs):
        self.pbd = pbd
        self.substep_dt = qd.ndarray(gs.qd_float, shape=())
        self.substep_dt.from_numpy(np.array(pbd.substep_dt, dtype=gs.np_float))
        self.n = max(pbd.n_particles, 1)
        size = 3 * self.n * n_envs
        self.target = qd.ndarray(gs.qd_float, shape=(size,))
        self.grad = qd.ndarray(gs.qd_float, shape=(size,))
        self.r = qd.ndarray(gs.qd_float, shape=(size,))
        self.z = qd.ndarray(gs.qd_float, shape=(size,))
        self.p = qd.ndarray(gs.qd_float, shape=(size,))
        self.ap = qd.ndarray(gs.qd_float, shape=(size,))
        self.sol = qd.ndarray(gs.qd_float, shape=(size,))
        self.x_prev = qd.ndarray(gs.qd_float, shape=(size,))
        self.diag_inv = qd.ndarray(gs.qd_float, shape=(3 * size,))


# ================================================ Layer 0: access ================================================


@qd.func
def vec_at(k: int, a: qd.template()):
    return qd.Vector([a[3 * k + d] for d in qd.static(range(3))], dt=gs.qd_float)


@qd.func
def set_vec_at(k: int, v: qd.types.vector(3), a: qd.template()):
    for d in qd.static(range(3)):
        a[3 * k + d] = v[d]


@qd.func
def is_moving(i_q: int, i_b: int, pbd: qd.template()):
    """Whether a PBD particle is an unknown of the solve: active and free, an attached particle keeping its position."""
    return pbd.particles_ng_reordered[i_q, i_b].active and pbd.particles_reordered[i_q, i_b].free


# ============================================== Layer 1: the PBD block ==============================================


@qd.func(requires_top_level=True)
def func_pbd_begin(pd: qd.template()):
    """The projection of PBD's substep as the target of the inertia, the end positions starting at it."""
    qd.loop_config(name="mpm_pbd_begin")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        set_vec_at(i_b * pd.n + i_q, pd.pbd.particles_reordered[i_q, i_b].pos, pd.target)


@qd.func(requires_top_level=True)
def func_pbd_gradient(pd: qd.template()):
    """The gradient m (x - x~) of the inertia and its diagonal block m I, before the contact rows add theirs."""
    qd.loop_config(name="mpm_pbd_gradient")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        grad = qd.Vector.zero(gs.qd_float, 3)
        m = gs.qd_float(1.0)
        if is_moving(i_q, i_b, pd.pbd):
            m = pd.pbd.particles_info_reordered[i_q, i_b].mass
            grad = m * (pd.pbd.particles_reordered[i_q, i_b].pos - vec_at(k, pd.target))
        set_vec_at(k, grad, pd.grad)
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
    """The PBD part of the preconditioned conjugate gradient (PCG) start from 0: r = grad, solution and Ap cleared."""
    qd.loop_config(name="mpm_pbd_pcg_init")
    for i in range(pd.r.shape[0]):
        pd.r[i] = pd.grad[i]
        pd.sol[i] = 0.0
        pd.ap[i] = 0.0


@qd.func(requires_top_level=True)
def func_pbd_precondition(pd: qd.template()):
    """z = M^-1 r by the inverse diagonal blocks, with z = 0 for an attached particle so that it keeps its position."""
    qd.loop_config(name="mpm_pbd_precondition")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        z = qd.Vector.zero(gs.qd_float, 3)
        if is_moving(i_q, i_b, pd.pbd):
            A_inv = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i, j in qd.static(qd.ndrange(3, 3)):
                A_inv[i, j] = pd.diag_inv[9 * k + 3 * i + j]
            z = A_inv @ vec_at(k, pd.r)
        set_vec_at(k, z, pd.z)


@qd.func(requires_top_level=True)
def func_pbd_pcg_init_direction(pd: qd.template(), pcg_scalars: qd.template()):
    """Set the first search direction p = z over the PBD rows, and add r . z to rz."""
    qd.loop_config(name="mpm_pbd_pcg_init_direction")
    for i in range(pd.r.shape[0]):
        pd.p[i] = pd.z[i]
        qd.atomic_add(pcg_scalars.rz[()], pd.r[i] * pd.z[i])


@qd.func(requires_top_level=True)
def func_pbd_matvec(pd: qd.template()):
    """Ap = m p over the PBD rows, before the contact rows add their term."""
    qd.loop_config(name="mpm_pbd_matvec")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        m = gs.qd_float(0.0)
        if is_moving(i_q, i_b, pd.pbd):
            m = pd.pbd.particles_info_reordered[i_q, i_b].mass
        set_vec_at(k, m * vec_at(k, pd.p), pd.ap)


@qd.func(requires_top_level=True)
def func_pbd_matvec_dot(pd: qd.template(), pcg_scalars: qd.template()):
    """Add p . Ap over the PBD rows to p . Ap, once Ap holds the whole joint operator."""
    qd.loop_config(name="mpm_pbd_matvec_dot")
    for i in range(pd.p.shape[0]):
        if pcg_scalars.is_converged[()] == 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[()], pd.p[i] * pd.ap[i])


@qd.func(requires_top_level=True)
def func_pbd_pcg_update_xr(pd: qd.template(), pcg_scalars: qd.template()):
    """Update the solution and the residual over the PBD rows, x += alpha p and r -= alpha Ap."""
    qd.loop_config(name="mpm_pbd_pcg_update_xr")
    for i in range(pd.p.shape[0]):
        if pcg_scalars.is_converged[()] == 0:
            alpha = pcg_scalars.alpha[()]
            pd.sol[i] += alpha * pd.p[i]
            pd.r[i] -= alpha * pd.ap[i]


@qd.func(requires_top_level=True)
def func_pbd_pcg_dot_rz(pd: qd.template(), pcg_scalars: qd.template()):
    """Add r . z over the PBD rows to rz_new."""
    qd.loop_config(name="mpm_pbd_pcg_dot_rz")
    for i in range(pd.r.shape[0]):
        if pcg_scalars.is_converged[()] == 0:
            qd.atomic_add(pcg_scalars.rz_new[()], pd.r[i] * pd.z[i])


@qd.func(requires_top_level=True)
def func_pbd_pcg_direction(pd: qd.template(), pcg_scalars: qd.template()):
    """Update the search direction p = z + beta p over the PBD rows, and clear Ap."""
    qd.loop_config(name="mpm_pbd_pcg_direction")
    for i in range(pd.p.shape[0]):
        pd.ap[i] = 0.0
        if pcg_scalars.is_converged[()] == 0:
            pd.p[i] = pd.z[i] + pcg_scalars.beta[()] * pd.p[i]


@qd.func(requires_top_level=True)
def func_pbd_start_line_search(pd: qd.template(), control: qd.template()):
    """The line-search start point, and the largest component of the step -x into the Newton measure."""
    qd.loop_config(name="mpm_pbd_start_line_search")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        k = i_b * pd.n + i_q
        set_vec_at(k, pd.pbd.particles_reordered[i_q, i_b].pos, pd.x_prev)
        step = vec_at(k, pd.sol)
        qd.atomic_max(control.max_disp[()], qd.max(qd.abs(step[0]), qd.abs(step[1]), qd.abs(step[2])))


@qd.func(requires_top_level=True)
def func_pbd_step_forward(pd: qd.template(), control: qd.template()):
    qd.loop_config(name="mpm_pbd_step_forward")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        if is_moving(i_q, i_b, pd.pbd):
            k = i_b * pd.n + i_q
            pd.pbd.particles_reordered[i_q, i_b].pos = vec_at(k, pd.x_prev) - control.alpha[()] * vec_at(k, pd.sol)


@qd.func(requires_top_level=True)
def func_pbd_energy(pd: qd.template(), control: qd.template(), is_trial: bool):
    """Add the inertia 1/2 m |x - x~|^2 of the PBD particles to the start or trial energy."""
    qd.loop_config(name="mpm_pbd_energy")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        if is_moving(i_q, i_b, pd.pbd):
            diff = pd.pbd.particles_reordered[i_q, i_b].pos - vec_at(i_b * pd.n + i_q, pd.target)
            energy = 0.5 * pd.pbd.particles_info_reordered[i_q, i_b].mass * diff.dot(diff)
            if is_trial:
                qd.atomic_add(control.energy_trial[()], energy)
            else:
                qd.atomic_add(control.energy_start[()], energy)


@qd.func(requires_top_level=True)
def func_pbd_finish(dt: float, pd: qd.template()):
    """The velocity of every moving PBD particle from its displacement over the substep."""
    qd.loop_config(name="mpm_pbd_finish")
    for i_q, i_b in qd.ndrange(pd.pbd._n_particles, pd.pbd._B):
        if is_moving(i_q, i_b, pd.pbd):
            pd.pbd.particles_reordered[i_q, i_b].vel = (
                pd.pbd.particles_reordered[i_q, i_b].pos - pd.pbd.particles_reordered[i_q, i_b].ipos
            ) / dt


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class PbdContactParams:
    """Scene values of the MPM-PBD contact rows, which the PBD obstacles share, as device scalars.

    A PBD particle is a sphere of its radius plus the thickness `thickness`, in metres, half a grid cell of MPM. A row
    takes the penalty stiffness `stiffness` in units of its effective mass, the inverse of its inverse weight, and the
    friction coefficient `friction`, which PBD particles do not hold."""

    def __init__(self, dx: float) -> None:
        self.thickness = qd.ndarray(gs.qd_float, shape=())
        self.thickness.from_numpy(np.array(0.5 * dx, dtype=gs.np_float))
        self.stiffness = qd.ndarray(gs.qd_float, shape=())
        self.stiffness.from_numpy(np.array(10.0, dtype=gs.np_float))
        self.friction = qd.ndarray(gs.qd_float, shape=())
        self.friction.from_numpy(np.array(0.15, dtype=gs.np_float))


@qd.data_oriented
class PbdParticleQuery:
    """The linear bounding volume hierarchy (LBVH) over the PBD particles of every environment, in PBD's reordered
    layout (see genesis.engine.bvh).

    It also holds the data the filter of its queries reads: the PBD solver, the values of the MPM kernels `params`, the
    values of the contact rows `contact`, and the query point and nearest PBD particle of each sorted MPM particle in
    `newton`."""

    def __init__(self, pbd, newton, params, contact: PbdContactParams, n_envs):
        self.pbd = pbd
        self.newton = newton
        self.params = params
        self.contact = contact
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
    and the contact thickness.

    An inactive particle gets an empty box with a finite center, for the Morton code."""
    reach = qd.static(query.radius) + query.contact.thickness[()]
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
    """The narrow phase of a box query of a sorted MPM particle, which keeps the PBD particle of the smallest gap at the
    substep start.

    The query is that of sorted MPM particle i_q in environment i_t, against PBD particle i_a. The gap is the distance
    minus the PBD radius and the contact thickness. The filter keeps it in `newton.query_phi`, the PBD particle in
    `newton.query_idx` and the normal from it to the MPM particle in `newton.query_normal`, and records no pair."""
    x = qd.Vector([query.newton.query_pos[3 * i_q + d] for d in qd.static(range(3))], dt=gs.qd_float)
    d = x - query.pbd.particles_reordered[i_a, i_t].ipos
    dist = d.norm()
    phi = dist - qd.static(query.radius) - query.contact.thickness[()]
    if phi < query.newton.query_phi[i_q] and dist > gs.EPS:
        query.newton.query_phi[i_q] = phi
        query.newton.query_idx[i_q] = i_a
        for k in qd.static(range(3)):
            query.newton.query_normal[3 * i_q + k] = d[k] / dist
    return True


# ============================================== Layer 1: contact rows ==============================================


@qd.func(requires_top_level=True)
def func_build_pbd_rows(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    pbd: qd.template(),
    query: qd.template(),
    particles_info: MpmParticlesInfo,
    params: qd.template(),
):
    """Pyramid penalty rows of every valid particle without Rigid or FEM rows against the nearest PBD particle within
    its contact range or short of it by less than `contact_band_width`.

    A box query of the PBD trees finds the nearest PBD particle: the particle's box spans its band at its own speed, and
    the PBD leaves span the sweep, radius and thickness, so together they cover the band of the relative velocity. Every
    row of the pyramid J = n +- mu d is a penalty on the gap at the end of the substep, phi + J.r for the relative
    displacement r. Its target is -phi and its stiffness `query.contact.stiffness` times the row's effective mass, the
    inverse of the particle's inverse weight (see func_contact_invweight) plus that of the PBD particle."""
    friction = query.contact.friction[()]
    qd.loop_config(name="mpm_build_pbd_rows")
    for s in range(grid.n_valid[()]):
        if newton.contact_link[s] < 0 and newton.contact_tri[s] < 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            for d in qd.static(range(3)):
                newton.query_pos[3 * s + d] = x_p[d]
            newton.query_phi[s] = float("inf")
            newton.query_idx[s] = -1
            reach = contact_band_width(v_p.norm(), dt, params)
            func_bvh_query_aabb(
                i_b, s, x_p - reach, x_p + reach, query.tree, query.results, func_nearest_pbd_filter, query
            )
            i_q = newton.query_idx[s]
            if i_q >= 0:
                phi = newton.query_phi[s]
                normal = qd.Vector([newton.query_normal[3 * s + k] for k in qd.static(range(3))], dt=gs.qd_float)
                v_rel = v_p - pbd.particles_reordered[i_q, i_b].vel
                if phi < contact_band_width(-normal.dot(v_rel), dt, params):
                    invweight = contact_invweight(f, i_p, i_b, normal, particles, particles_info)
                    if is_moving(i_q, i_b, pbd):
                        invweight += 1.0 / pbd.particles_info_reordered[i_q, i_b].mass
                    D = query.contact.stiffness[()] / invweight
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
def func_pbd_point_disp(grid: qd.template(), newton: qd.template(), pbd: qd.template(), params: qd.template()):
    """Displacement over the substep of the PBD particle of every particle with PBD rows, at the current iterate."""
    qd.loop_config(name="mpm_pbd_point_disp")
    for s in range(grid.n_valid[()]):
        i_q = newton.contact_pbd[s]
        if i_q >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            dq = pbd.particles_reordered[i_q, i_b].pos - pbd.particles_reordered[i_q, i_b].ipos
            for d in qd.static(range(3)):
                newton.contact_dr[3 * s + d] = dq[d]


@qd.func(requires_top_level=True)
def func_pbd_contact_gradient(grid: qd.template(), newton: qd.template(), pd: qd.template(), params: qd.template()):
    """Add the share of the PBD rows to the PBD gradient and diagonal blocks.

    It runs after the inertia terms and after the MPM assembly that staged the row gradients g and Hessians H. The PBD
    particle takes -g and H."""
    qd.loop_config(name="mpm_pbd_contact_gradient")
    for s in range(grid.n_valid[()]):
        i_q = newton.contact_pbd[s]
        if i_q >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            k = i_b * pd.n + i_q
            for d in qd.static(range(3)):
                qd.atomic_add(pd.grad[3 * k + d], -newton.g_rigid[3 * s + d])
            for i, j in qd.static(qd.ndrange(3, 3)):
                qd.atomic_add(pd.diag_inv[9 * k + 3 * i + j], newton.h_rigid[9 * s + 3 * i + j])


@qd.func
def pbd_contact_terms(
    f: int,
    s: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    pd: qd.template(),
    params: qd.template(),
):
    """The operator terms of the PBD rows of a sorted particle: its stencil, row Hessian and trial displacement, and the
    direction of its PBD particle.

    For sorted particle s, the func returns its stencil slots and weights w, its row Hessian H, its trial displacement
    dxp = sum_a w_a p_u,a along the direction, the PBD row index k and the direction dq of the PBD particle."""
    pcg_workspace = linear.pcg
    i_p, i_b, slots, w = particle_stencil(f, s, particles, grid, params)
    H = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        H[i, j] = newton.h_rigid[9 * s + 3 * i + j]
    dxp = qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(8)):
        row = 3 * slots[a]
        dxp += w[a] * qd.Vector([pcg_workspace.p[row + d] for d in qd.static(range(3))], dt=gs.qd_float)
    k = i_b * pd.n + newton.contact_pbd[s]
    return slots, w, H, dxp, k, vec_at(k, pd.p)


@qd.func(requires_top_level=True)
def func_pbd_contact_matvec(
    f: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    pd: qd.template(),
    pcg_scalars: qd.template(),
    params: qd.template(),
):
    """The PBD rows' cross term on the MPM rows, after func_linear_pcg_spmv: Ap_u,a -= w_a H dq, and p.Ap takes
    p_u.(-sum w_a H dq)."""
    pcg_workspace = linear.pcg
    qd.loop_config(name="mpm_pbd_matvec_particles")
    for s in range(grid.n_valid[()]):
        if newton.contact_pbd[s] >= 0 and pcg_scalars.is_converged[()] == 0:
            slots, w, H, dxp, k, dq = pbd_contact_terms(f, s, particles, grid, newton, linear, pd, params)
            H_dq = H @ dq
            for a in qd.static(range(8)):
                row = 3 * slots[a]
                for d in qd.static(range(3)):
                    qd.atomic_add(pcg_workspace.ap[row + d], -w[a] * H_dq[d])
            qd.atomic_add(pcg_scalars.p_dot_ap[()], -dxp.dot(H_dq))


@qd.func(requires_top_level=True)
def func_pbd_contact_matvec_post(
    f: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    pd: qd.template(),
    pcg_scalars: qd.template(),
    params: qd.template(),
):
    """The PBD rows' term on the PBD rows, after the inertia overwrote their Ap: Ap_q += H (dq - dxp)."""
    qd.loop_config(name="mpm_pbd_matvec_pbd")
    for s in range(grid.n_valid[()]):
        if newton.contact_pbd[s] >= 0 and pcg_scalars.is_converged[()] == 0:
            slots, w, H, dxp, k, dq = pbd_contact_terms(f, s, particles, grid, newton, linear, pd, params)
            h = H @ (dq - dxp)
            for d in qd.static(range(3)):
                qd.atomic_add(pd.ap[3 * k + d], h[d])


# ======================================= Layer 1: MPM obstacles of the PBD loop =======================================


@qd.data_oriented
class MpmParticleObstacles:
    """The MPM particles at the start of the substep as immovable obstacles of the PBD particles (see
    PBDSolver.set_particle_obstacles).

    The obstacles act in the contact group of PBD's unified loop and in its pre-stabilization, at the contact distance
    of the MPM-PBD rows of the joint solve: the PBD radius plus `contact.thickness`. They keep PBD's own
    projection from pushing its particles into the MPM material, where the rows of the joint solve would push them back
    every substep. Those rows alone exchange the momentum between the two.

    A linear bounding volume hierarchy (LBVH) over the MPM particles of every environment finds the obstacle candidates
    of each PBD particle once per substep, at (i_b n + i_q) OBSTACLE_NEIGHBORS + k, and every iteration of PBD's loop
    reuses them."""

    def __init__(self, mpm_solver, pbd, contact: PbdContactParams, n_envs):
        self.mpm_particles = mpm_solver.particles
        self.mpm_particles_ng = mpm_solver.particles_ng
        self.pbd = pbd
        self.contact = contact
        self.radius = pbd.particle_radius
        self.n = max(pbd.n_particles, 1)
        self.frame = qd.ndarray(gs.qd_int, shape=())
        self.nbr_idx = qd.ndarray(gs.qd_int, shape=(self.n * n_envs * OBSTACLE_NEIGHBORS,))
        self.nbr_dist = qd.ndarray(gs.qd_float, shape=(self.n * n_envs * OBSTACLE_NEIGHBORS,))
        self.nbr_count = qd.ndarray(gs.qd_int, shape=(self.n * n_envs,))
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(mpm_solver.n_particles, 1))
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        self.results = array_class.get_bvh_query_results(1)

    def build(self, f):
        """Fit the leaves to the MPM particles, build the trees, and gather the obstacle candidates of every PBD
        particle, once PBD predicted its positions.

        The leaves sit at the MPM particles of frame f."""
        kernel_fit_mpm_obstacle_leaves(f, self)
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)
        kernel_gather_obstacle_neighbors(self)

    @qd.func(requires_top_level=True)
    def func_project(self, pbd: qd.template(), f: int):
        """The contacts of every PBD particle at its current position with the MPM particles, with friction."""
        func_pbd_obstacles(self, is_stabilization=False)

    @qd.func(requires_top_level=True)
    def func_stabilize(self, pbd: qd.template(), f: int):
        """The non-penetration of every PBD particle at its start position with the MPM particles."""
        func_pbd_obstacles(self, is_stabilization=True)


@qd.kernel
def kernel_fit_mpm_obstacle_leaves(f: int, obstacles: qd.template()):
    """A point leaf at every active MPM particle, and an empty box with a finite center at the others.

    The particles are those of frame f, which the kernel stores in `obstacles.frame` for the filters."""
    for _ in range(1):
        obstacles.frame[()] = f
    for i_b, i_p in qd.ndrange(obstacles.leaves.aabbs_min.shape[0], obstacles.leaves.aabbs_min.shape[1]):
        if obstacles.mpm_particles_ng.active[f, i_p, i_b]:
            x = obstacles.mpm_particles.pos[f, i_p, i_b]
            obstacles.leaves.aabbs_min[i_b, i_p] = x
            obstacles.leaves.aabbs_max[i_b, i_p] = x
        else:
            obstacles.leaves.aabbs_min[i_b, i_p] = qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
            obstacles.leaves.aabbs_max[i_b, i_p] = qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)


@qd.func
def func_obstacle_contact_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """The contact of a PBD particle with an MPM particle: the projection out to the contact distance along their
    separation, and the friction.

    The PBD particle is i_q of environment i_t and the MPM particle i_a. The friction is that of Macklin et al. 2014
    (Eq. 23-24) on the tangential displacement of the PBD particle over the substep, the obstacle standing still. The
    filter records no pair."""
    f = obstacles.frame[()]
    x_q = obstacles.pbd.particles_reordered[i_q, i_t].pos
    d_vec = x_q - obstacles.mpm_particles.pos[f, i_a, i_t]
    dist = d_vec.norm()
    target = qd.static(obstacles.radius) + obstacles.contact.thickness[()]
    if dist < target and dist > gs.EPS:
        n = d_vec / dist
        depth = target - dist
        dx = x_q - obstacles.pbd.particles_reordered[i_q, i_t].ipos
        dx_t = dx - n * n.dot(dx)
        mu_s = obstacles.pbd.particles_info_reordered[i_q, i_t].mu_s
        mu_k = obstacles.pbd.particles_info_reordered[i_q, i_t].mu_k
        friction = -dx_t
        if dx_t.norm() >= mu_s * depth:
            friction = friction * qd.min(1.0, mu_k * depth / dx_t.norm(gs.EPS))
        obstacles.pbd.particles_reordered[i_q, i_t].dpos += depth * n + friction
        obstacles.pbd.particles_reordered[i_q, i_t].dcount += 1.0
    return True


@qd.func
def func_obstacle_start_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """The non-penetration of a PBD particle at its start position with an MPM particle, for the pre-stabilization.

    The PBD particle is i_q of environment i_t and the MPM particle i_a. The filter records no pair."""
    f = obstacles.frame[()]
    d_vec = obstacles.pbd.particles_reordered[i_q, i_t].ipos - obstacles.mpm_particles.pos[f, i_a, i_t]
    dist = d_vec.norm()
    target = qd.static(obstacles.radius) + obstacles.contact.thickness[()]
    if dist < target and dist > gs.EPS:
        obstacles.pbd.particles_reordered[i_q, i_t].dpos += (target - dist) / dist * d_vec
        obstacles.pbd.particles_reordered[i_q, i_t].dcount += 1.0
    return True


@qd.func
def func_obstacle_gather_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """Keep an MPM particle among the nearest OBSTACLE_NEIGHBORS obstacle candidates of a PBD particle, replacing the
    farthest one once they are full.

    The filter keeps MPM particle i_a for PBD particle i_q of environment i_t if it lies within OBSTACLE_REACH contact
    distances of its start or predicted position, and records no pair."""
    f = obstacles.frame[()]
    x_m = obstacles.mpm_particles.pos[f, i_a, i_t]
    dist = qd.min(
        (obstacles.pbd.particles_reordered[i_q, i_t].pos - x_m).norm(),
        (obstacles.pbd.particles_reordered[i_q, i_t].ipos - x_m).norm(),
    )
    reach = qd.static(OBSTACLE_REACH) * (qd.static(obstacles.radius) + obstacles.contact.thickness[()])
    if dist < reach:
        k = i_t * obstacles.n + i_q
        base = k * qd.static(OBSTACLE_NEIGHBORS)
        n = obstacles.nbr_count[k]
        if n < qd.static(OBSTACLE_NEIGHBORS):
            obstacles.nbr_idx[base + n] = i_a
            obstacles.nbr_dist[base + n] = dist
            obstacles.nbr_count[k] = n + 1
        else:
            i_worst = 0
            for j in range(1, qd.static(OBSTACLE_NEIGHBORS)):
                if obstacles.nbr_dist[base + j] > obstacles.nbr_dist[base + i_worst]:
                    i_worst = j
            if dist < obstacles.nbr_dist[base + i_worst]:
                obstacles.nbr_idx[base + i_worst] = i_a
                obstacles.nbr_dist[base + i_worst] = dist
    return True


@qd.kernel
def kernel_gather_obstacle_neighbors(obstacles: qd.template()):
    """The obstacle candidates of every free PBD particle for the substep, by a box query of the MPM trees around its
    start and predicted positions."""
    reach = qd.static(OBSTACLE_REACH) * (qd.static(obstacles.radius) + obstacles.contact.thickness[()])
    qd.loop_config(name="pbd_mpm_obstacle_neighbors")
    for i_q, i_b in qd.ndrange(obstacles.pbd._n_particles, obstacles.pbd._B):
        obstacles.nbr_count[i_b * obstacles.n + i_q] = 0
        if is_moving(i_q, i_b, obstacles.pbd):
            x0 = obstacles.pbd.particles_reordered[i_q, i_b].ipos
            x1 = obstacles.pbd.particles_reordered[i_q, i_b].pos
            func_bvh_query_aabb(
                i_b,
                i_q,
                qd.min(x0, x1) - reach,
                qd.max(x0, x1) + reach,
                obstacles.tree,
                obstacles.results,
                func_obstacle_gather_filter,
                obstacles,
            )


@qd.func(requires_top_level=True)
def func_pbd_obstacles(obstacles: qd.template(), is_stabilization: qd.template()):
    """Accumulate the obstacle corrections of every free PBD particle against its obstacle candidates of the substep, at
    its current or start position."""
    qd.loop_config(name="pbd_mpm_obstacles")
    for i_q, i_b in qd.ndrange(obstacles.pbd._n_particles, obstacles.pbd._B):
        k = i_b * obstacles.n + i_q
        for j in range(obstacles.nbr_count[k]):
            i_a = obstacles.nbr_idx[k * qd.static(OBSTACLE_NEIGHBORS) + j]
            if qd.static(is_stabilization):
                func_obstacle_start_filter(i_b, i_a, i_q, obstacles)
            else:
                func_obstacle_contact_filter(i_b, i_a, i_q, obstacles)


# =============================================== Layer 2: participants ===============================================


@qd.data_oriented
class PbdNewtonParticipant:
    """Device stages of the PBD end positions in the joint solve, which own the PBD data of the solve."""

    def __init__(self, pbd, n_envs: int):
        self.pd = PbdNewtonData(pbd, n_envs)

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int):
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
    def func_pcg_init_direction(self, pcg_scalars: qd.template()):
        func_pbd_pcg_init_direction(self.pd, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_matvec(self, pcg_scalars: qd.template()):
        func_pbd_matvec(self.pd)

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, pcg_scalars: qd.template()):
        func_pbd_matvec_dot(self.pd, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, pcg_scalars: qd.template()):
        func_pbd_pcg_update_xr(self.pd, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, pcg_scalars: qd.template()):
        func_pbd_pcg_dot_rz(self.pd, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, pcg_scalars: qd.template()):
        func_pbd_pcg_direction(self.pd, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        func_pbd_start_line_search(self.pd, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        func_pbd_step_forward(self.pd, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f: int, is_trial: bool):
        func_pbd_energy(self.pd, control, is_trial)


@qd.data_oriented
class MpmPbdContact:
    """Device stages of the MPM-PBD contact rows in the joint solve: their share of the PBD gradient, their cross terms
    in both blocks of the operator, and the PBD particle displacement at every iterate."""

    def __init__(self, mpm, pbd_participant: PbdNewtonParticipant, query: PbdParticleQuery):
        self.particles = mpm.particles
        self.particles_info = mpm.particles_info
        self.grid = mpm.grid
        self.newton = mpm.newton
        self.linear = mpm.linear
        self.params = mpm.params
        self.pd = pbd_participant.pd
        self.query = query

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f: int):
        func_pbd_contact_gradient(self.grid, self.newton, self.pd, self.params)

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f: int, pcg_scalars: qd.template()):
        func_pbd_contact_matvec(
            f, self.particles, self.grid, self.newton, self.linear, self.pd, pcg_scalars, self.params
        )
        func_pbd_contact_matvec_post(
            f, self.particles, self.grid, self.newton, self.linear, self.pd, pcg_scalars, self.params
        )

    @qd.func(requires_top_level=True)
    def func_after_step(self, f: int):
        func_pbd_point_disp(self.grid, self.newton, self.pd.pbd, self.params)


@qd.kernel
def kernel_pbd_begin(pd: qd.template()):
    func_pbd_begin(pd)


@qd.kernel
def kernel_pbd_finish(pd: qd.template()):
    func_pbd_finish(pd.substep_dt[()], pd)


@qd.kernel
def kernel_pbd_rows(f: int, contact: qd.template()):
    func_build_pbd_rows(
        f,
        contact.params.substep_dt[()],
        contact.particles,
        contact.grid,
        contact.newton,
        contact.pd.pbd,
        contact.query,
        contact.particles_info,
        contact.params,
    )
    contact.func_after_step(f)


class PbdNewtonHost(NewtonHost):
    """Host stages of PBD around the joint solve: the projection of its substep as the inertia target before, and its
    velocities from the solved positions after."""

    def __init__(self, device: PbdNewtonParticipant):
        self._device = device

    def newton_prepare(self, f):
        kernel_pbd_begin(self._device.pd)

    def newton_assemble(self, f, is_exact=False):
        kernel_assemble_participant(f, self._device)

    def newton_finish(self, f):
        kernel_pbd_finish(self._device.pd)


class MpmPbdContactHost(NewtonHost):
    """Host stages of the MPM-PBD contact rows before the joint solve: the PBD trees, then the rows and the PBD particle
    displacement at the start point."""

    def __init__(self, device: MpmPbdContact):
        self._device = device

    def newton_prepare(self, f):
        self._device.query.build()
        kernel_pbd_rows(f, self._device)
