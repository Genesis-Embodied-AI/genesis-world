"""Contact pairs of the implicit material point method (MPM) with position based dynamics (PBD) in the joint Newton
solve (see pbd_newton.py for the PBD unknown and its stages).

An MPM particle gets a pair of penalty rows against its nearest PBD particle, a sphere of the PBD radius plus a contact
thickness, on its displacement relative to that of the PBD particle. The pairs are contact pairs of the Newton solve
(see newton_contact.py)."""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, func_no_filter, get_bvh_data
from genesis.engine.solvers.mpm_data import MpmGridScratch, MpmInfo, MpmParticles, MpmParticlesInfo
from genesis.engine.solvers.mpm_func import (
    EMPTY_BOX_HALF,
    contact_band_width,
    contact_invweight,
    particle_of_sorted,
    pyramid_terms,
)
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_contact import append_pair, write_doublet, write_triplet
from genesis.engine.solvers.newton_linear import PAIRS
from genesis.engine.solvers.pbd_newton import PbdNewtonParticipant, is_moving
from genesis.utils.array_class import BVHQueryResults
from genesis.utils.data import Data, DataKind, field

from .contact_candidates import (
    ContactCandidates,
    CouplingPairs,
    func_report_candidates,
    func_reset_candidates,
    func_select_nearest,
    set_candidate_score,
)

# Obstacle candidates of a PBD particle over a substep: the nearest OBSTACLE_NEIGHBORS MPM particles within
# OBSTACLE_REACH contact distances of its start or predicted position
OBSTACLE_NEIGHBORS = 32
OBSTACLE_REACH = 1.5


class MpmPbdPairs(Data, kind=DataKind.SCRATCH):
    """The pairs of the particles with their nearest PBD particle (see func_build_pbd_pairs).

    A pair holds its sorted particle `s`, its environment `i_b`, its PBD particle `particle` in PBD's reordered layout
    and the stiffness `D` and target `t` of the rows of its friction pyramid around `normal`. `n_pairs` counts the pairs
    asked for, above the capacity on overflow."""

    n_pairs: qd.Tensor = field(int, ndarray=True)
    s: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    i_b: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    particle: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    normal: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    D: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    t: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)


def build_pbd_pairs(max_pairs: int) -> MpmPbdPairs:
    """Build the MPM-PBD pairs of a capacity, with no pair."""
    pairs = MpmPbdPairs(sizes={PAIRS: max_pairs})
    pairs.n_pairs.fill(0)
    return pairs


# ============================================== Layer 1: broad phase ==============================================


@qd.data_oriented
class PbdContactParams:
    """Scene values of the MPM-PBD contact pairs, as device scalars: the thickness `thickness` added to the PBD radius
    in metres, the penalty stiffness `stiffness` of a row in units of its effective mass and the friction `friction`."""

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
    environment, in PBD's reordered layout (see genesis.engine.bvh), with the PBD participant, the values of the
    material point method (MPM) kernels `mpm_info` and the values of the contact pairs `contact`."""

    def __init__(self, pbd: PbdNewtonParticipant, mpm_info, contact: PbdContactParams, n_envs):
        self.pbd = pbd
        self.mpm_info = mpm_info
        self.contact = contact
        self.radius = pbd.particle_radius
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(pbd.n_particles, 1))
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves

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


@qd.func(requires_top_level=True)
def func_query_pbd_particles(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    query: qd.template(),
    candidates: ContactCandidates,
    mpm_info: MpmInfo,
):
    """Record the PBD particles of every valid particle as candidates: the particle's box spans its band at its own
    speed, and the PBD leaves span the sweep, radius and thickness, so together they cover the band of the relative
    velocity."""
    func_reset_candidates(candidates)
    qd.loop_config(name="mpm_query_pbd_particles")
    for s in range(grid.n_valid[()]):
        i_p, i_b, env, cell = particle_of_sorted(s, grid, mpm_info)
        x_p = particles.pos[f, i_p, i_b]
        reach = contact_band_width(particles.vel[f, i_p, i_b].norm(), dt, mpm_info)
        func_bvh_query_aabb(i_b, s, x_p - reach, x_p + reach, query.tree, candidates.results, func_no_filter, query)


@qd.func(requires_top_level=True)
def func_score_pbd_particles(
    f: int,
    particles: MpmParticles,
    grid: MpmGridScratch,
    query: qd.template(),
    candidates: ContactCandidates,
    mpm_info: MpmInfo,
):
    """Score every candidate PBD particle by the gap of the particle to its sphere at the substep start, its distance
    minus the PBD radius and the contact thickness. A PBD particle at the position of the particle gives no normal and
    is rejected."""
    func_report_candidates(candidates)
    qd.loop_config(name="mpm_score_pbd_particles")
    for i_cand in range(candidates.order.n_triplets[()]):
        triplet = candidates.results.triplets[i_cand]
        i_b = triplet[0]
        i_q = triplet[1]
        s = triplet[2]
        i_p, i_b_sorted, env, cell = particle_of_sorted(s, grid, mpm_info)
        dist = (particles.pos[f, i_p, i_b] - query.pbd.particles_reordered[i_q, i_b].ipos).norm()
        score = qd.math.inf
        if dist > gs.EPS:
            score = dist - qd.static(query.radius) - query.contact.thickness[()]
        set_candidate_score(i_cand, s, score, 0.0, candidates)


# ============================================== Layer 1: contact pairs ==============================================


@qd.func(requires_top_level=True)
def func_build_pbd_pairs(
    f: int,
    dt: float,
    particles: MpmParticles,
    grid: MpmGridScratch,
    pairs: MpmPbdPairs,
    pbd: qd.template(),
    query: qd.template(),
    candidates: ContactCandidates,
    particles_info: MpmParticlesInfo,
    mpm_info: MpmInfo,
):
    """Build the pair of every valid particle with its nearest PBD particle, within its contact range or short of it by
    less than `contact_band_width`, once the candidates are sorted by particle.

    Every row J = n +- mu d of the pyramid is a penalty on the end gap phi + J.r of the relative displacement r, of
    target -phi and stiffness `query.contact.stiffness` times its effective mass, the inverse of the inverse weights of
    the two particles."""
    func_select_nearest(0.0, candidates)
    for _ in range(1):
        pairs.n_pairs[()] = 0
    qd.loop_config(name="mpm_build_pbd_pairs")
    for i_seg in range(candidates.order.bcoo_nnz[()]):
        i_cand = candidates.best_candidate[i_seg]
        if i_cand < candidates.secondary.shape[0]:
            triplet = candidates.results.triplets[i_cand]
            i_b = triplet[0]
            i_q = triplet[1]
            s = triplet[2]
            phi = candidates.order.triplet_val[i_cand]
            i_p, i_b_sorted, env, cell = particle_of_sorted(s, grid, mpm_info)
            d = particles.pos[f, i_p, i_b] - pbd.particles_reordered[i_q, i_b].ipos
            normal = d / d.norm()
            v_rel = particles.vel[f, i_p, i_b] - pbd.particles_reordered[i_q, i_b].vel
            if phi < contact_band_width(-normal.dot(v_rel), dt, mpm_info):
                i_pair = append_pair(pairs.n_pairs, pairs.s.shape[0])
                if i_pair >= 0:
                    invweight = contact_invweight(f, i_p, i_b, normal, particles, particles_info)
                    if is_moving(i_q, i_b, pbd):
                        invweight += 1.0 / pbd.particles_info_reordered[i_q, i_b].mass
                    pairs.s[i_pair] = s
                    pairs.i_b[i_pair] = i_b
                    pairs.particle[i_pair] = i_q
                    pairs.normal[i_pair] = normal
                    pairs.D[i_pair] = query.contact.stiffness[()] / invweight
                    pairs.t[i_pair] = -phi


@qd.func
def pbd_pair_terms(f: int, i_pair: int, mpm: qd.template(), pbd: qd.template(), contact: qd.template()):
    """Energy, gradient and Hessian of the rows of a pair in the displacement of its particle relative to its PBD
    particle at the current iterate, with the contact vertices of both."""
    i_vq = contact.pairs.i_b[i_pair] * pbd.n_vertex_particles + contact.pairs.particle[i_pair]
    r = mpm.func_vertex_value(f, contact.pairs.s[i_pair]) - pbd.func_vertex_value(f, i_vq)
    D = contact.pairs.D[i_pair]
    t = contact.pairs.t[i_pair]
    energy, g, H = pyramid_terms(
        contact.pairs.normal[i_pair],
        contact.query.contact.friction[()],
        qd.Vector([D, D, D, D]),
        qd.Vector([t, t, t, t]),
        r,
    )
    return energy, g, H, mpm.vertex_offset[()] + contact.pairs.s[i_pair], pbd.vertex_offset[()] + i_vq


@qd.func(requires_top_level=True)
def func_pbd_pair_terms(
    f: int,
    i_d0: int,
    i_t0: int,
    mpm: qd.template(),
    pbd: qd.template(),
    contact: qd.template(),
    newton_contact: qd.template(),
):
    """Write the gradient and Hessian of the rows of every pair at the current iterate, from doublet slot `i_d0` and
    triplet slot `i_t0` on: on both particles, and the block between them."""
    qd.loop_config(name="mpm_pbd_pair_terms")
    for i_pair in range(contact.pairs.n_pairs[()]):
        _, g, H, i_vp, i_vq = pbd_pair_terms(f, i_pair, mpm, pbd, contact)
        write_doublet(i_d0 + 2 * i_pair, i_vp, g, newton_contact)
        write_doublet(i_d0 + 2 * i_pair + 1, i_vq, -g, newton_contact)
        write_triplet(i_t0 + 3 * i_pair, i_vp, i_vp, H, newton_contact)
        write_triplet(i_t0 + 3 * i_pair + 1, i_vp, i_vq, -H, newton_contact)
        write_triplet(i_t0 + 3 * i_pair + 2, i_vq, i_vq, H, newton_contact)


@qd.func(requires_top_level=True)
def func_pbd_pair_energy(
    f: int, mpm: qd.template(), pbd: qd.template(), contact: qd.template(), energies: qd.template()
):
    """Add the energy of the rows of every pair at the current iterate into the entry of its environment."""
    qd.loop_config(name="mpm_pbd_pair_energy")
    for i_pair in range(contact.pairs.n_pairs[()]):
        energy, grad, hess, i_vp, i_vq = pbd_pair_terms(f, i_pair, mpm, pbd, contact)
        qd.atomic_add(energies[contact.pairs.i_b[i_pair]], energy)


# ======================================= Layer 1: MPM obstacles of the PBD loop =======================================


@qd.data_oriented
class MpmParticleObstacles:
    """The MPM particles at the start of the substep as immovable obstacles of the PBD particles (see
    PBDSolver.set_particle_obstacles).

    The obstacles act in the contact group of PBD's loop and in its pre-stabilization, at the contact distance of the
    MPM-PBD rows, so that PBD's own projection does not push its particles into the MPM material. A linear bounding
    volume hierarchy (LBVH) over the MPM particles finds the candidates of each PBD particle once per substep, at [i_q,
    j, i_b] for the j-th of them."""

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
    """Project PBD particle i_q of environment i_t out of MPM particle i_a to the contact distance, with the friction of
    Macklin et al. 2014 (Eq. 23-24) on its tangential displacement over the substep."""
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
    """Keep MPM particle i_a among the nearest OBSTACLE_NEIGHBORS obstacle candidates of PBD particle i_q of environment
    i_t if it lies within OBSTACLE_REACH contact distances of its start or predicted position."""
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


# ============================================== Layer 2: pair kind ===============================================


@qd.data_oriented
class MpmPbdContact(CouplingPairs):
    """Pair kind of the contacts between the material point method (MPM) and position based dynamics (PBD), the pair of
    a particle and its nearest PBD particle. A pair writes 2 doublets on its particle and its PBD particle, and the 3
    upper blocks between them."""

    n_doublets = 2
    n_triplets = 3
    build_pairs = staticmethod(build_pbd_pairs)

    def __init__(self, mpm: MpmNewtonParticipant, pbd: PbdNewtonParticipant, query: PbdParticleQuery):
        self.pbd = pbd
        self.query = query
        super().__init__(mpm)

    def build_query(self, f: int) -> None:
        self.query.build()

    @qd.func(requires_top_level=True)
    def func_query(self, f: int):
        func_query_pbd_particles(
            f,
            self.mpm.mpm_info.substep_dt[()],
            self.mpm.particles,
            self.mpm.grid,
            self.query,
            self.candidates,
            self.mpm.mpm_info,
        )

    @qd.func(requires_top_level=True)
    def func_score(self, f: int):
        func_score_pbd_particles(f, self.mpm.particles, self.mpm.grid, self.query, self.candidates, self.mpm.mpm_info)

    @qd.func(requires_top_level=True)
    def func_build_pairs(self, f: int):
        func_build_pbd_pairs(
            f,
            self.mpm.mpm_info.substep_dt[()],
            self.mpm.particles,
            self.mpm.grid,
            self.pairs,
            self.pbd,
            self.query,
            self.candidates,
            self.mpm.particles_info,
            self.mpm.mpm_info,
        )

    @qd.func(requires_top_level=True)
    def func_pair_terms(self, f: int, i_kind: qd.template(), contact: qd.template()):
        func_pbd_pair_terms(
            f,
            contact.matrices.doublet_offsets[i_kind],
            contact.matrices.triplet_offsets[i_kind],
            self.mpm,
            self.pbd,
            self,
            contact,
        )

    @qd.func(requires_top_level=True)
    def func_pair_energy(self, f: int, energies: qd.template()):
        func_pbd_pair_energy(f, self.mpm, self.pbd, self, energies)
