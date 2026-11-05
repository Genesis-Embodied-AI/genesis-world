"""Stages of the contact rows of the implicit material point method (MPM) with position based dynamics (PBD), the
coupling of the joint Newton solve over the grid displacement and the PBD particle positions (see pbd_newton.py for the
PBD unknown and its stages). PBD's own constraints stay with PBD, so the coupling is strong in the contact and staggered
in the constraints.

An MPM particle gets penalty rows against the nearest PBD particle, a sphere of the PBD radius plus a contact thickness.
The rows act on the particle displacement relative to the PBD particle's displacement over the substep, and solve the
contact by the masses on both sides, so that a heavy block holds a light cloth and the cloth barely moves the block.
"""

from collections.abc import Iterator

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import (
    CONTACT_ROWS,
    PYRAMID_ROWS,
    SORTED_PARTICLES,
    WALL_ROWS,
    MpmGridScratch,
    MpmInfo,
    MpmLinearScratch,
    MpmNewtonScratch,
    MpmParticles,
    MpmParticlesInfo,
)
from genesis.engine.solvers.mpm_func import (
    EMPTY_BOX_HALF,
    contact_band_width,
    contact_cross_matvec,
    contact_invweight,
    particle_of_sorted,
    pyramid_dirs,
    pyramid_row_terms,
)
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_solver import NewtonDevice, NewtonHost
from genesis.engine.solvers.pbd_newton import PbdNewtonParticipant, is_moving
from genesis.utils.array_class import BVHQueryResults
from genesis.utils.data import Data, DataItem, DataKind, field, iter_data

# Obstacle candidates of a PBD particle over a substep: the nearest OBSTACLE_NEIGHBORS MPM particles within
# OBSTACLE_REACH contact distances of its start or predicted position
OBSTACLE_NEIGHBORS = 32
OBSTACLE_REACH = 1.5


class MpmPbdContactScratch(Data, kind=DataKind.SCRATCH, shape=(SORTED_PARTICLES,)):
    """Position based dynamics (PBD) side of the pyramid rows of every sorted particle."""

    # The PBD particle of the particle's pyramid rows, in PBD's reordered layout, -1 if PBD gave it none
    particle: qd.Tensor = field(int, ndarray=True)
    # The Hessian of those rows on the displacement of the PBD particle
    h: qd.Tensor = field((float, (3, 3)), ndarray=True)


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
    """The linear bounding volume hierarchy (LBVH) over the position based dynamics (PBD) particles of every
    environment, in PBD's reordered layout (see genesis.engine.bvh).

    It also holds the data the filter of its queries reads: the PBD participant, the values of the material point
    method (MPM) kernels `mpm_info`, the values of the contact rows `contact`, and the query point and nearest PBD
    particle of each sorted MPM particle in `newton`."""

    def __init__(
        self, pbd: PbdNewtonParticipant, mpm: MpmNewtonParticipant, mpm_info, contact: PbdContactParams, n_envs
    ):
        self.pbd = pbd
        self.mpm = mpm
        self.mpm_info = mpm_info
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
    d = query.mpm.newton.query_pos[i_q] - query.pbd.particles_reordered[i_a, i_t].ipos
    dist = d.norm()
    phi = dist - qd.static(query.radius) - query.contact.thickness[()]
    if phi < query.mpm.newton.query_phi[i_q] and dist > gs.EPS:
        query.mpm.newton.query_phi[i_q] = phi
        query.mpm.newton.query_idx[i_q] = i_a
        query.mpm.newton.query_normal[i_q] = d / dist
    return True


# ============================================== Layer 1: contact rows ==============================================


@qd.func(requires_top_level=True)
def func_build_pbd_rows(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    rows: MpmPbdContactScratch,
    pbd: qd.template(),
    query: qd.template(),
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Pyramid penalty rows of every valid particle without Rigid or finite element method (FEM) rows against the
    nearest position based dynamics (PBD) particle within its contact range or short of it by less than
    `contact_band_width`.

    A particle that got rows is marked in `newton.has_pyramid_rows` and its PBD particle is in `rows.particle`, which
    holds -1 for the others. The rows go to the particles that the Rigid and FEM rows left free, which relies on the
    Rigid and FEM row builders of the substep running before this one.

    A box query of the PBD trees finds the nearest PBD particle: the particle's box spans its band at its own speed, and
    the PBD leaves span the sweep, radius and thickness, so together they cover the band of the relative velocity. Every
    row of the pyramid J = n +- mu d is a penalty on the gap at the end of the substep, phi + J.r for the relative
    displacement r. Its target is -phi and its stiffness `query.contact.stiffness` times the row's effective mass, the
    inverse of the particle's inverse weight (see func_contact_invweight) plus that of the PBD particle."""
    friction = query.contact.friction[()]
    qd.loop_config(name="mpm_build_pbd_rows")
    for s in range(grid.n_valid[()]):
        rows.particle[s] = -1
        if not newton.has_pyramid_rows[s]:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
            x_p = particles.pos[f, i_p, i_b]
            v_p = particles.vel[f, i_p, i_b]
            newton.query_pos[s] = x_p
            newton.query_phi[s] = float("inf")
            newton.query_idx[s] = -1
            reach = contact_band_width(v_p.norm(), dt, mpm_info)
            func_bvh_query_aabb(
                i_b, s, x_p - reach, x_p + reach, query.tree, query.results, func_nearest_pbd_filter, query
            )
            i_q = newton.query_idx[s]
            if i_q >= 0:
                phi = newton.query_phi[s]
                normal = newton.query_normal[s]
                v_rel = v_p - pbd.particles_reordered[i_q, i_b].vel
                if phi < contact_band_width(-normal.dot(v_rel), dt, mpm_info):
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
                    rows.particle[s] = i_q
                    newton.has_pyramid_rows[s] = True


@qd.func(requires_top_level=True)
def func_pbd_point_disp(
    grid: MpmGridScratch, newton: MpmNewtonScratch, rows: MpmPbdContactScratch, pbd: qd.template(), mpm_info: MpmInfo
):
    """Displacement over the substep of the PBD particle of every particle with PBD rows, at the current iterate."""
    qd.loop_config(name="mpm_pbd_point_disp")
    for s in range(grid.n_valid[()]):
        i_q = rows.particle[s]
        if i_q >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
            newton.contact_point_disp[s] = (
                pbd.particles_reordered[i_q, i_b].pos - pbd.particles_reordered[i_q, i_b].ipos
            )


@qd.func(requires_top_level=True)
def func_pbd_contact_gradient(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    newton: MpmNewtonScratch,
    rows: MpmPbdContactScratch,
    pbd_data: qd.template(),
    mpm_info: MpmInfo,
):
    """Add the share of the position based dynamics (PBD) rows to the PBD gradient and diagonal blocks.

    It runs after the inertia terms and after the material point method (MPM) assembly, at the same iterate. The
    gradient g and Hessian H of the rows of each particle come from `pyramid_row_terms`, and H stays in `rows.h` for
    the matvec. The PBD particle takes -g and H."""
    qd.loop_config(name="mpm_pbd_contact_gradient")
    for s in range(grid.n_valid[()]):
        i_q = rows.particle[s]
        if i_q >= 0:
            i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
            _, g, H = pyramid_row_terms(f, s, particles, grid, newton, mpm_info)
            rows.h[s] = H
            qd.atomic_add(pbd_data.grad[i_q, i_b], -g)
            qd.atomic_add(pbd_data.diag_inv[i_q, i_b], H)


@qd.func(requires_top_level=True)
def func_pbd_contact_matvec(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    linear: MpmLinearScratch,
    rows: MpmPbdContactScratch,
    pbd_data: qd.template(),
    pcg_scalars: qd.template(),
    mpm_info: MpmInfo,
):
    """The PBD rows' share of the joint operator, after the own operators of MPM and PBD set their Ap.

    The PBD particle of a particle moves by its direction dq = p_q. The rows add their cross term on the MPM rows (see
    contact_cross_matvec), and h to Ap_q."""
    qd.loop_config(name="mpm_pbd_contact_matvec")
    for s in range(grid.n_valid[()]):
        i_q = rows.particle[s]
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        if i_q >= 0 and pcg_scalars.is_converged[i_b] == 0:
            dq = pbd_data.p[i_q, i_b]
            h = contact_cross_matvec(f, s, rows.h[s], dq, particles, grid, linear, pcg_scalars, mpm_info)
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


# ============================================== Layer 2: contact stages ===============================================


@qd.data_oriented
class MpmPbdContact(NewtonDevice):
    """Device stages of the contact rows between the material point method (MPM) and position based dynamics (PBD).

    They add their share of the PBD gradient, their cross terms in both blocks of the operator, and the PBD particle
    displacement at every iterate. `pbd_rows` holds the PBD side of the rows of the sorted particles."""

    def __init__(
        self,
        mpm: MpmNewtonParticipant,
        pbd: PbdNewtonParticipant,
        query: PbdParticleQuery,
        pbd_rows: MpmPbdContactScratch,
    ):
        self.mpm = mpm
        self.pbd = pbd
        self.query = query
        self.pbd_rows = pbd_rows

    @property
    def data(self) -> Iterator[DataItem]:
        """Yield the rows of the sorted particles on the position based dynamics (PBD) side, rebuilt every substep."""
        yield from iter_data(self, "pbd_rows")

    @qd.func(requires_top_level=True)
    def func_contact_gradient(self, f: int):
        func_pbd_contact_gradient(
            f, self.mpm.particles, self.mpm.grid, self.mpm.newton, self.pbd_rows, self.pbd.pbd_data, self.mpm.mpm_info
        )

    @qd.func(requires_top_level=True)
    def func_contact_matvec(self, f: int, pcg_scalars: qd.template()):
        func_pbd_contact_matvec(
            f,
            self.mpm.particles,
            self.mpm.grid,
            self.mpm.linear,
            self.pbd_rows,
            self.pbd.pbd_data,
            pcg_scalars,
            self.mpm.mpm_info,
        )

    @qd.func(requires_top_level=True)
    def func_after_step(self, f: int):
        func_pbd_point_disp(self.mpm.grid, self.mpm.newton, self.pbd_rows, self.pbd, self.mpm.mpm_info)


@qd.kernel
def kernel_pbd_rows(f: int, contact: qd.template()):
    func_build_pbd_rows(
        f,
        contact.mpm.mpm_info.substep_dt[()],
        contact.mpm.particles,
        contact.mpm.grid,
        contact.mpm.newton,
        contact.pbd_rows,
        contact.pbd,
        contact.query,
        contact.mpm.particles_info,
        contact.mpm.mpm_info,
    )
    contact.func_after_step(f)


class MpmPbdContactHost(NewtonHost):
    """Host stages of the MPM-PBD contact rows before the joint solve: the PBD trees, then the rows and the PBD particle
    displacement at the start point."""

    def __init__(self, device: MpmPbdContact):
        self._device = device

    def newton_prepare(self, f):
        self._device.query.build()
        kernel_pbd_rows(f, self._device)
