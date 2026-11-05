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
from genesis.utils.array_class import BVHQueryResults
import genesis.utils.geom as gu
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import CONTACT_ROWS, PYRAMID_ROWS, WALL_ROWS, MpmParticles, MpmParticlesInfo
from genesis.engine.solvers.mpm_func import (
    EMPTY_BOX_HALF,
    contact_band_width,
    contact_cross_matvec,
    contact_invweight,
    particle_of_sorted,
    pyramid_dirs,
)
from genesis.engine.solvers.newton_solver import NewtonHost, kernel_assemble_participant

# Obstacle candidates of a PBD particle over a substep: the nearest OBSTACLE_NEIGHBORS MPM particles within
# OBSTACLE_REACH contact distances of its start or predicted position
OBSTACLE_NEIGHBORS = 32
OBSTACLE_REACH = 1.5


@qd.data_oriented
class PbdNewtonData:
    """The PBD rows of the joint solve, per reordered PBD particle and environment, indexed [i_q, i_b].

    The vectors are the projection x~ of PBD's own substep, the gradient, the preconditioned conjugate gradient (PCG)
    residual, preconditioned residual, direction, operator image and solution, and the line-search start. `diag_inv`
    holds the 3 x 3 diagonal block of every particle, then its inverse.

    `pos_begin` and `vel_begin` hold the position and velocity of every PBD particle at the start of the substep,
    indexed [i_p, i_b] like PBD's own particles, which the particles of a failed environment return to."""

    def __init__(self, pbd, n_envs):
        self.pbd = pbd
        self.substep_dt = qd.ndarray(gs.qd_float, shape=())
        self.substep_dt.from_numpy(np.array(pbd.substep_dt, dtype=gs.np_float))
        shape = (max(pbd.n_particles, 1), n_envs)
        self.pos_begin = qd.ndarray(gs.qd_vec3, shape=shape)
        self.vel_begin = qd.ndarray(gs.qd_vec3, shape=shape)
        self.target = qd.ndarray(gs.qd_vec3, shape=shape)
        self.grad = qd.ndarray(gs.qd_vec3, shape=shape)
        self.r = qd.ndarray(gs.qd_vec3, shape=shape)
        self.z = qd.ndarray(gs.qd_vec3, shape=shape)
        self.p = qd.ndarray(gs.qd_vec3, shape=shape)
        self.ap = qd.ndarray(gs.qd_vec3, shape=shape)
        self.sol = qd.ndarray(gs.qd_vec3, shape=shape)
        self.x_prev = qd.ndarray(gs.qd_vec3, shape=shape)
        self.diag_inv = qd.ndarray(gs.qd_mat3, shape=shape)


# ================================================ Layer 0: access ================================================


@qd.func
def is_moving(i_q: int, i_b: int, pbd: qd.template()):
    """Whether a PBD particle is an unknown of the solve: active and free, an attached particle keeping its position."""
    return pbd.particles_ng_reordered[i_q, i_b].active and pbd.particles_reordered[i_q, i_b].free


# ============================================== Layer 1: the PBD block ==============================================


@qd.func(requires_top_level=True)
def func_pbd_begin(pbd_data: qd.template()):
    """The projection of PBD's substep as the target of the inertia, the end positions starting at it."""
    qd.loop_config(name="mpm_pbd_begin")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        pbd_data.target[i_q, i_b] = pbd_data.pbd.particles_reordered[i_q, i_b].pos


@qd.func(requires_top_level=True)
def func_pbd_gradient(pbd_data: qd.template()):
    """The gradient m (x - x~) of the inertia and its diagonal block m I, before the contact rows add theirs."""
    qd.loop_config(name="mpm_pbd_gradient")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        grad = qd.Vector.zero(gs.qd_float, 3)
        m = gs.qd_float(1.0)
        if is_moving(i_q, i_b, pbd_data.pbd):
            m = pbd_data.pbd.particles_info_reordered[i_q, i_b].mass
            grad = m * (pbd_data.pbd.particles_reordered[i_q, i_b].pos - pbd_data.target[i_q, i_b])
        pbd_data.grad[i_q, i_b] = grad
        pbd_data.diag_inv[i_q, i_b] = m * qd.Matrix.identity(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_pbd_factor(pbd_data: qd.template()):
    """The inverse of every particle's diagonal block, once the contact rows added theirs."""
    qd.loop_config(name="mpm_pbd_factor")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        pbd_data.diag_inv[i_q, i_b] = pbd_data.diag_inv[i_q, i_b].inverse()


@qd.func(requires_top_level=True)
def func_pbd_pcg_init(pbd_data: qd.template(), pcg_scalars: qd.template(), is_adjoint: qd.template()):
    """The PBD part of the preconditioned conjugate gradient (PCG) start from 0: r = grad, solution and Ap cleared.

    With `is_adjoint`, an environment left out of the solve keeps its solution (see func_init_linear_pcg_residual)."""
    qd.loop_config(name="mpm_pbd_pcg_init")
    for i_q, i_b in qd.ndrange(pbd_data.r.shape[0], pbd_data.r.shape[1]):
        pbd_data.r[i_q, i_b] = pbd_data.grad[i_q, i_b]
        is_kept = False
        if qd.static(is_adjoint):
            is_kept = pcg_scalars.is_converged[i_b] != 0
        if not is_kept:
            pbd_data.sol[i_q, i_b] = qd.Vector.zero(gs.qd_float, 3)
        pbd_data.ap[i_q, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func(requires_top_level=True)
def func_pbd_precondition(pbd_data: qd.template()):
    """z = M^-1 r by the inverse diagonal blocks, with z = 0 for an attached particle so that it keeps its position."""
    qd.loop_config(name="mpm_pbd_precondition")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        z = qd.Vector.zero(gs.qd_float, 3)
        if is_moving(i_q, i_b, pbd_data.pbd):
            z = pbd_data.diag_inv[i_q, i_b] @ pbd_data.r[i_q, i_b]
        pbd_data.z[i_q, i_b] = z


@qd.func(requires_top_level=True)
def func_pbd_pcg_init_direction(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Set the first search direction p = z over the PBD rows, and add r . z to rz."""
    qd.loop_config(name="mpm_pbd_pcg_init_direction")
    for i_q, i_b in qd.ndrange(pbd_data.r.shape[0], pbd_data.r.shape[1]):
        pbd_data.p[i_q, i_b] = pbd_data.z[i_q, i_b]
        qd.atomic_add(pcg_scalars.rz[i_b], pbd_data.r[i_q, i_b].dot(pbd_data.z[i_q, i_b]))


@qd.func(requires_top_level=True)
def func_pbd_matvec(pbd_data: qd.template()):
    """Ap = m p over the PBD rows, before the contact rows add their term."""
    qd.loop_config(name="mpm_pbd_matvec")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        m = gs.qd_float(0.0)
        if is_moving(i_q, i_b, pbd_data.pbd):
            m = pbd_data.pbd.particles_info_reordered[i_q, i_b].mass
        pbd_data.ap[i_q, i_b] = m * pbd_data.p[i_q, i_b]


@qd.func(requires_top_level=True)
def func_pbd_matvec_dot(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Add p . Ap over the PBD rows to p . Ap, once Ap holds the whole joint operator."""
    qd.loop_config(name="mpm_pbd_matvec_dot")
    for i_q, i_b in qd.ndrange(pbd_data.p.shape[0], pbd_data.p.shape[1]):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.p_dot_ap[i_b], pbd_data.p[i_q, i_b].dot(pbd_data.ap[i_q, i_b]))


@qd.func(requires_top_level=True)
def func_pbd_pcg_update_xr(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Update the solution and the residual over the PBD rows, x += alpha p and r -= alpha Ap."""
    qd.loop_config(name="mpm_pbd_pcg_update_xr")
    for i_q, i_b in qd.ndrange(pbd_data.p.shape[0], pbd_data.p.shape[1]):
        if pcg_scalars.is_converged[i_b] == 0:
            alpha = pcg_scalars.alpha[i_b]
            pbd_data.sol[i_q, i_b] += alpha * pbd_data.p[i_q, i_b]
            pbd_data.r[i_q, i_b] -= alpha * pbd_data.ap[i_q, i_b]


@qd.func(requires_top_level=True)
def func_pbd_pcg_dot_rz(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Add r . z over the PBD rows to rz_new."""
    qd.loop_config(name="mpm_pbd_pcg_dot_rz")
    for i_q, i_b in qd.ndrange(pbd_data.r.shape[0], pbd_data.r.shape[1]):
        if pcg_scalars.is_converged[i_b] == 0:
            qd.atomic_add(pcg_scalars.rz_new[i_b], pbd_data.r[i_q, i_b].dot(pbd_data.z[i_q, i_b]))


@qd.func(requires_top_level=True)
def func_pbd_pcg_direction(pbd_data: qd.template(), pcg_scalars: qd.template()):
    """Update the search direction p = z + beta p over the PBD rows, and clear Ap."""
    qd.loop_config(name="mpm_pbd_pcg_direction")
    for i_q, i_b in qd.ndrange(pbd_data.p.shape[0], pbd_data.p.shape[1]):
        pbd_data.ap[i_q, i_b] = qd.Vector.zero(gs.qd_float, 3)
        if pcg_scalars.is_converged[i_b] == 0:
            pbd_data.p[i_q, i_b] = pbd_data.z[i_q, i_b] + pcg_scalars.beta[i_b] * pbd_data.p[i_q, i_b]


@qd.func(requires_top_level=True)
def func_pbd_start_line_search(pbd_data: qd.template(), control: qd.template()):
    """The line-search start point, and the largest component of the step -x into the Newton measure."""
    qd.loop_config(name="mpm_pbd_start_line_search")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        pbd_data.x_prev[i_q, i_b] = pbd_data.pbd.particles_reordered[i_q, i_b].pos
        qd.atomic_max(control.max_disp[i_b], qd.abs(pbd_data.sol[i_q, i_b]).max())


@qd.func(requires_top_level=True)
def func_pbd_step_forward(pbd_data: qd.template(), control: qd.template()):
    qd.loop_config(name="mpm_pbd_step_forward")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if is_moving(i_q, i_b, pbd_data.pbd):
            step = control.alpha[i_b] * pbd_data.sol[i_q, i_b]
            pbd_data.pbd.particles_reordered[i_q, i_b].pos = pbd_data.x_prev[i_q, i_b] - step


@qd.func(requires_top_level=True)
def func_pbd_energy(pbd_data: qd.template(), control: qd.template(), is_trial: qd.template()):
    """Add the inertia 1/2 m |x - x~|^2 of the PBD particles to the start or trial energy."""
    qd.loop_config(name="mpm_pbd_energy")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if is_moving(i_q, i_b, pbd_data.pbd):
            diff = pbd_data.pbd.particles_reordered[i_q, i_b].pos - pbd_data.target[i_q, i_b]
            energy = 0.5 * pbd_data.pbd.particles_info_reordered[i_q, i_b].mass * diff.dot(diff)
            if is_trial:
                qd.atomic_add(control.energy_trial[i_b], energy)
            else:
                qd.atomic_add(control.energy_start[i_b], energy)


@qd.func(requires_top_level=True)
def func_pbd_finish(dt: float, pbd_data: qd.template(), control: qd.template()):
    """The velocity of every moving PBD particle from its displacement over the substep, then the position and velocity
    of the substep start for every particle of an environment whose solve failed."""
    qd.loop_config(name="mpm_pbd_finish")
    for i_q, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if is_moving(i_q, i_b, pbd_data.pbd):
            pbd_data.pbd.particles_reordered[i_q, i_b].vel = (
                pbd_data.pbd.particles_reordered[i_q, i_b].pos - pbd_data.pbd.particles_reordered[i_q, i_b].ipos
            ) / dt

    qd.loop_config(name="mpm_pbd_freeze")
    for i_p, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        if control.has_failed_non_finite[i_b] != 0 and pbd_data.pbd.particles_ng[i_p, i_b].active:
            i_q = pbd_data.pbd.particles_ng[i_p, i_b].reordered_idx
            pbd_data.pbd.particles_reordered[i_q, i_b].pos = pbd_data.pos_begin[i_p, i_b]
            pbd_data.pbd.particles_reordered[i_q, i_b].vel = pbd_data.vel_begin[i_p, i_b]


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
        self.results = BVHQueryResults(sizes=dict(query_results=1))

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
    d = query.newton.query_pos[i_q] - query.pbd.particles_reordered[i_a, i_t].ipos
    dist = d.norm()
    phi = dist - qd.static(query.radius) - query.contact.thickness[()]
    if phi < query.newton.query_phi[i_q] and dist > gs.EPS:
        query.newton.query_phi[i_q] = phi
        query.newton.query_idx[i_q] = i_a
        query.newton.query_normal[i_q] = d / dist
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
            newton.query_pos[s] = x_p
            newton.query_phi[s] = float("inf")
            newton.query_idx[s] = -1
            reach = contact_band_width(v_p.norm(), dt, params)
            func_bvh_query_aabb(
                i_b, s, x_p - reach, x_p + reach, query.tree, query.results, func_nearest_pbd_filter, query
            )
            i_q = newton.query_idx[s]
            if i_q >= 0:
                phi = newton.query_phi[s]
                normal = newton.query_normal[s]
                v_rel = v_p - pbd.particles_reordered[i_q, i_b].vel
                if phi < contact_band_width(-normal.dot(v_rel), dt, params):
                    invweight = contact_invweight(f, i_p, i_b, normal, particles, particles_info)
                    if is_moving(i_q, i_b, pbd):
                        invweight += 1.0 / pbd.particles_info_reordered[i_q, i_b].mass
                    D = query.contact.stiffness[()] / invweight
                    dirs = pyramid_dirs(normal, friction)
                    for i in qd.static(range(PYRAMID_ROWS)):
                        k = CONTACT_ROWS * s + WALL_ROWS + i
                        newton.contact_dir[k] = dirs[i, :]
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
            newton.contact_dr[s] = pbd.particles_reordered[i_q, i_b].pos - pbd.particles_reordered[i_q, i_b].ipos


@qd.func(requires_top_level=True)
def func_pbd_contact_gradient(
    grid: qd.template(), newton: qd.template(), pbd_data: qd.template(), params: qd.template()
):
    """Add the share of the PBD rows to the PBD gradient and diagonal blocks.

    It runs after the inertia terms and after the MPM assembly that staged the row gradients g and Hessians H. The PBD
    particle takes -g and H."""
    qd.loop_config(name="mpm_pbd_contact_gradient")
    for s in range(grid.n_valid[()]):
        i_q = newton.contact_pbd[s]
        if i_q >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
            qd.atomic_add(pbd_data.grad[i_q, i_b], -newton.g_rigid[s])
            qd.atomic_add(pbd_data.diag_inv[i_q, i_b], newton.h_rigid[s])


@qd.func(requires_top_level=True)
def func_pbd_contact_matvec(
    f: int,
    particles: MpmParticles,
    grid: qd.template(),
    newton: qd.template(),
    linear: qd.template(),
    pbd_data: qd.template(),
    pcg_scalars: qd.template(),
    params: qd.template(),
):
    """The PBD rows' share of the joint operator, after the own operators of MPM and PBD set their Ap.

    The PBD particle of a particle moves by its direction dq = p_q. The rows add their cross term on the MPM rows (see
    contact_cross_matvec), and h to Ap_q."""
    qd.loop_config(name="mpm_pbd_contact_matvec")
    for s in range(grid.n_valid[()]):
        i_q = newton.contact_pbd[s]
        i_p, i_b, env, cell = particle_of_sorted(s, grid, params)
        if i_q >= 0 and pcg_scalars.is_converged[i_b] == 0:
            dq = pbd_data.p[i_q, i_b]
            h = contact_cross_matvec(f, s, dq, particles, grid, newton, linear, pcg_scalars, params)
            qd.atomic_add(pbd_data.ap[i_q, i_b], h)


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
    of each PBD particle once per substep, at [i_q, j, i_b] for the j-th of them, and every iteration of PBD's loop
    reuses them."""

    def __init__(self, mpm_solver, pbd, contact: PbdContactParams, n_envs):
        self.mpm_particles = mpm_solver.particles
        self.mpm_particles_ng = mpm_solver.particles_ng
        self.pbd = pbd
        self.contact = contact
        self.radius = pbd.particle_radius
        n_pbd = max(pbd.n_particles, 1)
        self.frame = qd.ndarray(gs.qd_int, shape=())
        self.neighbors_idx = qd.ndarray(gs.qd_int, shape=(n_pbd, OBSTACLE_NEIGHBORS, n_envs))
        self.neighbors_dist = qd.ndarray(gs.qd_float, shape=(n_pbd, OBSTACLE_NEIGHBORS, n_envs))
        self.n_neighbors = qd.ndarray(gs.qd_int, shape=(n_pbd, n_envs))
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(mpm_solver.n_particles, 1))
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        self.results = BVHQueryResults(sizes=dict(query_results=1))

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
        n = obstacles.n_neighbors[i_q, i_t]
        if n < qd.static(OBSTACLE_NEIGHBORS):
            obstacles.neighbors_idx[i_q, n, i_t] = i_a
            obstacles.neighbors_dist[i_q, n, i_t] = dist
            obstacles.n_neighbors[i_q, i_t] = n + 1
        else:
            j_worst = 0
            for j in range(1, qd.static(OBSTACLE_NEIGHBORS)):
                if obstacles.neighbors_dist[i_q, j, i_t] > obstacles.neighbors_dist[i_q, j_worst, i_t]:
                    j_worst = j
            if dist < obstacles.neighbors_dist[i_q, j_worst, i_t]:
                obstacles.neighbors_idx[i_q, j_worst, i_t] = i_a
                obstacles.neighbors_dist[i_q, j_worst, i_t] = dist
    return True


@qd.kernel
def kernel_gather_obstacle_neighbors(obstacles: qd.template()):
    """The obstacle candidates of every free PBD particle for the substep, by a box query of the MPM trees around its
    start and predicted positions."""
    reach = qd.static(OBSTACLE_REACH) * (qd.static(obstacles.radius) + obstacles.contact.thickness[()])
    qd.loop_config(name="pbd_mpm_obstacle_neighbors")
    for i_q, i_b in qd.ndrange(obstacles.pbd._n_particles, obstacles.pbd._B):
        obstacles.n_neighbors[i_q, i_b] = 0
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
        for j in range(obstacles.n_neighbors[i_q, i_b]):
            i_a = obstacles.neighbors_idx[i_q, j, i_b]
            if qd.static(is_stabilization):
                func_obstacle_start_filter(i_b, i_a, i_q, obstacles)
            else:
                func_obstacle_contact_filter(i_b, i_a, i_q, obstacles)


# =============================================== Layer 2: participants ===============================================


@qd.data_oriented
class PbdNewtonParticipant:
    """Device stages of the PBD end positions in the joint solve, which own the PBD data of the solve."""

    def __init__(self, pbd, n_envs: int):
        self.pbd_data = PbdNewtonData(pbd, n_envs)

    @qd.func(requires_top_level=True)
    def func_assemble(self, f: int):
        func_pbd_gradient(self.pbd_data)

    @qd.func(requires_top_level=True)
    def func_factor(self):
        func_pbd_factor(self.pbd_data)

    @qd.func(requires_top_level=True)
    def func_pcg_init(self, pcg_scalars: qd.template(), is_adjoint: qd.template()):
        func_pbd_pcg_init(self.pbd_data, pcg_scalars, is_adjoint)

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        func_pbd_precondition(self.pbd_data)

    @qd.func(requires_top_level=True)
    def func_pcg_init_direction(self, pcg_scalars: qd.template()):
        func_pbd_pcg_init_direction(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_matvec(self, pcg_scalars: qd.template()):
        func_pbd_matvec(self.pbd_data)

    @qd.func(requires_top_level=True)
    def func_matvec_dot(self, pcg_scalars: qd.template()):
        func_pbd_matvec_dot(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_update_xr(self, pcg_scalars: qd.template()):
        func_pbd_pcg_update_xr(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_dot_rz(self, pcg_scalars: qd.template()):
        func_pbd_pcg_dot_rz(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_pcg_direction(self, pcg_scalars: qd.template()):
        func_pbd_pcg_direction(self.pbd_data, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_start_line_search(self, control: qd.template(), f: int):
        func_pbd_start_line_search(self.pbd_data, control)

    @qd.func(requires_top_level=True)
    def func_step_forward(self, control: qd.template(), f: int):
        func_pbd_step_forward(self.pbd_data, control)

    @qd.func(requires_top_level=True)
    def func_energy(self, control: qd.template(), f: int, is_trial: qd.template()):
        func_pbd_energy(self.pbd_data, control, is_trial)


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
        self.pbd_data = pbd_participant.pbd_data
        self.query = query

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f: int):
        func_pbd_contact_gradient(self.grid, self.newton, self.pbd_data, self.params)

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f: int, pcg_scalars: qd.template()):
        func_pbd_contact_matvec(
            f, self.particles, self.grid, self.newton, self.linear, self.pbd_data, pcg_scalars, self.params
        )

    @qd.func(requires_top_level=True)
    def func_after_step(self, f: int):
        func_pbd_point_disp(self.grid, self.newton, self.pbd_data.pbd, self.params)


@qd.kernel
def kernel_pbd_save(pbd_data: qd.template()):
    for i_p, i_b in qd.ndrange(pbd_data.pbd._n_particles, pbd_data.pbd._B):
        pbd_data.pos_begin[i_p, i_b] = pbd_data.pbd.particles[i_p, i_b].pos
        pbd_data.vel_begin[i_p, i_b] = pbd_data.pbd.particles[i_p, i_b].vel


@qd.kernel
def kernel_pbd_begin(pbd_data: qd.template()):
    func_pbd_begin(pbd_data)


@qd.kernel
def kernel_pbd_finish(pbd_data: qd.template(), control: qd.template()):
    func_pbd_finish(pbd_data.substep_dt[()], pbd_data, control)


@qd.kernel
def kernel_pbd_rows(f: int, contact: qd.template()):
    func_build_pbd_rows(
        f,
        contact.params.substep_dt[()],
        contact.particles,
        contact.grid,
        contact.newton,
        contact.pbd_data.pbd,
        contact.query,
        contact.particles_info,
        contact.params,
    )
    contact.func_after_step(f)


class PbdNewtonHost(NewtonHost):
    """Host stages of PBD around the joint solve: the projection of its substep as the inertia target before, and its
    velocities from the solved positions after.

    PBD runs its own substep before the solve, so the host saves the start of the substep first."""

    def __init__(self, device: PbdNewtonParticipant):
        self._device = device

    def newton_begin(self, f):
        kernel_pbd_save(self._device.pbd_data)

    def newton_prepare(self, f):
        kernel_pbd_begin(self._device.pbd_data)

    def newton_assemble(self, f, is_exact=False):
        kernel_assemble_participant(f, self._device)

    def newton_finish(self, f, control):
        kernel_pbd_finish(self._device.pbd_data, control)


class MpmPbdContactHost(NewtonHost):
    """Host stages of the MPM-PBD contact rows before the joint solve: the PBD trees, then the rows and the PBD particle
    displacement at the start point."""

    def __init__(self, device: MpmPbdContact):
        self._device = device

    def newton_prepare(self, f):
        self._device.query.build()
        kernel_pbd_rows(f, self._device)
