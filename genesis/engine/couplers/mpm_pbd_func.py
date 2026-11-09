"""Contact pairs of the implicit material point method (MPM) with position based dynamics (PBD) in the joint Newton
solve (see pbd_newton.py for the PBD unknown and its stages)."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import PYRAMID_ROWS
from genesis.engine.solvers.mpm_func import EMPTY_BOX_HALF, contact_band_width, contact_invweight, particle_of_sorted
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_linear import PAIRS
from genesis.engine.solvers.pbd_newton import PBD_PARTICLES, PbdNewtonParticipant, is_moving
from genesis.utils.array_class import BVHQueryResults
from genesis.utils.data import ENVS, Data, DataItem, DataKind, field, iter_data

from .contact_candidates import CouplingPairs
from .mpm_sph_func import kernel_fit_mpm_particle_leaves

# Obstacle candidates of a PBD particle over a substep: the nearest OBSTACLE_NEIGHBORS MPM particles within
# OBSTACLE_REACH contact distances of its start or predicted position
OBSTACLE_NEIGHBORS = 32
OBSTACLE_REACH = 1.5


class MpmPbdPairs(Data, kind=DataKind.SCRATCH):
    """The pairs of the particles with their nearest PBD particle (see MpmPbdContact.build_pair)."""

    n_pairs: qd.Tensor = field(int, ndarray=True)
    s: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    i_b: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    particle: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    normal: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    friction: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    D: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)
    t: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)


# ============================================== Layer 1: broad phase ==============================================


class PbdContactParams(Data, kind=DataKind.CONSTANT, shape=()):
    """Scene values of the MPM-PBD contact pairs, as device scalars: the thickness `thickness` added to the PBD radius
    in metres, the penalty stiffness `stiffness` of a row in units of its effective mass and the friction `friction`."""

    thickness: qd.Tensor = field(float, ndarray=True)
    stiffness: qd.Tensor = field(float, ndarray=True)
    friction: qd.Tensor = field(float, ndarray=True)


@qd.data_oriented
class PbdParticleQuery:
    """The linear bounding volume hierarchy (LBVH) over the position based dynamics (PBD) particles of every
    environment, in PBD's reordered layout (see genesis.engine.bvh), with the PBD participant, the values of the
    material point method (MPM) kernels `mpm_info` and the values of the contact pairs `contact`."""

    def __init__(self, pbd: PbdNewtonParticipant, mpm_info, contact: PbdContactParams, n_envs):
        self.pbd = pbd
        self.mpm_info = mpm_info
        self.contact = contact
        self.radius = pbd.solver.particle_radius
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(pbd.solver.n_particles, 1))
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves

    def build(self):
        """Fit the leaf boxes to the PBD particles' sweeps over the substep and build the trees."""
        kernel_fit_pbd_leaves(self)
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)


@qd.kernel
def kernel_fit_pbd_leaves(query: qd.template()):
    """The box of every PBD particle over its sweep from its start to its projected end position, grown by its radius
    and the contact thickness."""
    reach = qd.static(query.radius) + query.contact.thickness[()]
    for i_b, i_q in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        if query.pbd.solver.particles_ng_reordered[i_q, i_b].active:
            x0 = query.pbd.solver.particles_reordered[i_q, i_b].ipos
            x1 = query.pbd.solver.particles_reordered[i_q, i_b].pos
            query.leaves.aabbs_min[i_b, i_q] = qd.min(x0, x1) - reach
            query.leaves.aabbs_max[i_b, i_q] = qd.max(x0, x1) + reach
        else:
            query.leaves.aabbs_min[i_b, i_q] = qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float)
            query.leaves.aabbs_max[i_b, i_q] = qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)


# ======================================= Layer 1: MPM obstacles of the PBD loop =======================================


class MpmObstaclesScratch(Data, kind=DataKind.SCRATCH):
    """The frame of the MPM particles in the tree of MpmParticleObstacles, and the number of obstacle candidates of
    every PBD particle over the substep, with their indices and distances."""

    frame: qd.Tensor = field(int, ndarray=True)
    n_neighbors: qd.Tensor = field(int, shape=(PBD_PARTICLES, ENVS), ndarray=True)
    neighbors_idx: qd.Tensor = field(int, shape=(PBD_PARTICLES, OBSTACLE_NEIGHBORS, ENVS), ndarray=True)
    neighbors_dist: qd.Tensor = field(float, shape=(PBD_PARTICLES, OBSTACLE_NEIGHBORS, ENVS), ndarray=True)


@qd.data_oriented
class MpmParticleObstacles:
    """The MPM particles at the start of the substep as immovable obstacles of the PBD particles at the contact distance
    of the MPM-PBD rows, so that PBD's own projection does not push its particles into the MPM material (see
    PBDSolver.set_particle_obstacles)."""

    def __init__(self, mpm_solver, pbd, contact: PbdContactParams, n_envs):
        self.mpm_particles = mpm_solver.particles
        self.mpm_particles_ng = mpm_solver.particles_ng
        self.pbd = pbd
        self.contact = contact
        self.radius = pbd.particle_radius
        self.scratch = MpmObstaclesScratch(sizes={PBD_PARTICLES: pbd.n_particles, ENVS: n_envs})
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(mpm_solver.n_particles, 1))
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves
        self.results = BVHQueryResults(sizes=dict(query_results=1))

    def build(self, f):
        """Fit the leaves to the MPM particles, build the trees, and gather the obstacle candidates of every PBD
        particle, once PBD predicted its positions."""
        kernel_fit_mpm_particle_leaves(f, self)
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


@qd.func
def func_obstacle_contact_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """Project PBD particle i_q of environment i_t out of MPM particle i_a to the contact distance, with the friction of
    Macklin et al. 2014 (Eq. 23-24) on its tangential displacement over the substep."""
    f = obstacles.scratch.frame[()]
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
        particle = obstacles.pbd.particles_reordered[i_q, i_t]
        obstacles.pbd.particles_reordered[i_q, i_t].dpos = particle.dpos + depth * n + friction
        obstacles.pbd.particles_reordered[i_q, i_t].dcount = particle.dcount + 1.0
    return True


@qd.func
def func_obstacle_start_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """The non-penetration of PBD particle i_q of environment i_t at its start position with MPM particle i_a, for the
    pre-stabilization."""
    f = obstacles.scratch.frame[()]
    d_vec = obstacles.pbd.particles_reordered[i_q, i_t].ipos - obstacles.mpm_particles.pos[f, i_a, i_t]
    dist = d_vec.norm()
    target = qd.static(obstacles.radius) + obstacles.contact.thickness[()]
    if dist < target and dist > gs.EPS:
        particle = obstacles.pbd.particles_reordered[i_q, i_t]
        obstacles.pbd.particles_reordered[i_q, i_t].dpos = particle.dpos + (target - dist) / dist * d_vec
        obstacles.pbd.particles_reordered[i_q, i_t].dcount = particle.dcount + 1.0
    return True


@qd.func
def func_obstacle_gather_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """Keep MPM particle i_a among the nearest OBSTACLE_NEIGHBORS obstacle candidates of PBD particle i_q of environment
    i_t if it lies within OBSTACLE_REACH contact distances of its start or predicted position."""
    f = obstacles.scratch.frame[()]
    x_m = obstacles.mpm_particles.pos[f, i_a, i_t]
    dist = qd.min(
        (obstacles.pbd.particles_reordered[i_q, i_t].pos - x_m).norm(),
        (obstacles.pbd.particles_reordered[i_q, i_t].ipos - x_m).norm(),
    )
    reach = qd.static(OBSTACLE_REACH) * (qd.static(obstacles.radius) + obstacles.contact.thickness[()])
    if dist < reach:
        n = obstacles.scratch.n_neighbors[i_q, i_t]
        if n < qd.static(OBSTACLE_NEIGHBORS):
            obstacles.scratch.neighbors_idx[i_q, n, i_t] = i_a
            obstacles.scratch.neighbors_dist[i_q, n, i_t] = dist
            obstacles.scratch.n_neighbors[i_q, i_t] = n + 1
        else:
            j_worst = 0
            for j in range(1, qd.static(OBSTACLE_NEIGHBORS)):
                if obstacles.scratch.neighbors_dist[i_q, j, i_t] > obstacles.scratch.neighbors_dist[i_q, j_worst, i_t]:
                    j_worst = j
            if dist < obstacles.scratch.neighbors_dist[i_q, j_worst, i_t]:
                obstacles.scratch.neighbors_idx[i_q, j_worst, i_t] = i_a
                obstacles.scratch.neighbors_dist[i_q, j_worst, i_t] = dist
    return True


@qd.kernel
def kernel_gather_obstacle_neighbors(obstacles: qd.template()):
    """The obstacle candidates of every free PBD particle for the substep, by a box query of the MPM trees around its
    start and predicted positions."""
    reach = qd.static(OBSTACLE_REACH) * (qd.static(obstacles.radius) + obstacles.contact.thickness[()])
    qd.loop_config(name="pbd_mpm_obstacle_neighbors")
    for i_q, i_b in qd.ndrange(obstacles.pbd._n_particles, obstacles.pbd._B):
        obstacles.scratch.n_neighbors[i_q, i_b] = 0
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
        for j in range(obstacles.scratch.n_neighbors[i_q, i_b]):
            i_a = obstacles.scratch.neighbors_idx[i_q, j, i_b]
            if qd.static(is_stabilization):
                func_obstacle_start_filter(i_b, i_a, i_q, obstacles)
            else:
                func_obstacle_contact_filter(i_b, i_a, i_q, obstacles)


# ============================================== Layer 2: pair kind ===============================================


@qd.data_oriented
class MpmPbdContact(CouplingPairs):
    """Pair kind of the contacts between the material point method (MPM) and position based dynamics (PBD), the pair of
    a particle and its nearest PBD particle, over the particle and the PBD particle."""

    n_doublets = 2
    n_triplets = 3
    pairs_type = MpmPbdPairs

    def __init__(self, mpm: MpmNewtonParticipant, pbd: PbdNewtonParticipant, query: PbdParticleQuery):
        self.pbd = pbd
        self.query = query
        super().__init__(mpm)

    @property
    def data(self) -> Iterator[DataItem]:
        yield from super().data
        yield from iter_data(self.query, "contact")

    def build_query(self, f: int) -> None:
        self.query.build()

    @qd.func
    def candidate_score(self, f: int, i_b: int, i_q: int, s: int):
        """Gap of sorted particle `s` to the sphere of PBD particle i_q at the substep start, its distance minus the PBD
        radius and the contact thickness, infinite for a PBD particle at the position of the particle."""
        i_p, i_b_sorted, env, cell = particle_of_sorted(s, self.mpm.grid, self.mpm.mpm_info)
        dist = (self.mpm.particles.pos[f, i_p, i_b] - self.pbd.solver.particles_reordered[i_q, i_b].ipos).norm()
        score = qd.math.inf
        if dist > gs.EPS:
            score = dist - qd.static(self.query.radius) - self.query.contact.thickness[()]
        return score, 0.0

    @qd.func
    def build_pair(self, f: int, i_b: int, i_q: int, s: int, phi: float):
        """Pair sorted particle `s` with PBD particle i_q at gap phi if it lies within its contact range or short of it
        by less than `contact_band_width`."""
        i_p, i_b_sorted, env, cell = particle_of_sorted(s, self.mpm.grid, self.mpm.mpm_info)
        d = self.mpm.particles.pos[f, i_p, i_b] - self.pbd.solver.particles_reordered[i_q, i_b].ipos
        normal = d / d.norm()
        v_rel = self.mpm.particles.vel[f, i_p, i_b] - self.pbd.solver.particles_reordered[i_q, i_b].vel
        if phi < contact_band_width(-normal.dot(v_rel), self.mpm.mpm_info.substep_dt[()], self.mpm.mpm_info):
            invweight = contact_invweight(f, i_p, i_b, normal, self.mpm.particles, self.mpm.particles_info)
            if is_moving(i_q, i_b, self.pbd.solver):
                invweight += 1.0 / self.pbd.solver.particles_info_reordered[i_q, i_b].mass
            D = qd.Vector([self.query.contact.stiffness[()] / invweight] * PYRAMID_ROWS)
            i_pair = self.add_pair(s, i_b, normal, self.query.contact.friction[()], D, qd.Vector([-phi] * PYRAMID_ROWS))
            if i_pair >= 0:
                self.pairs.particle[i_pair] = i_q

    @qd.func
    def pair_vertices(self, f: int, i_pair: int):
        """The particle and the PBD particle of a pair, of weights 1 and -1, with their relative displacement."""
        i_vq = self.pairs.i_b[i_pair] * self.pbd.n_vertex_particles + self.pairs.particle[i_pair]
        r = self.mpm.func_vertex_value(f, self.pairs.s[i_pair]) - self.pbd.func_vertex_value(f, i_vq)
        verts = qd.Vector(
            [self.mpm.vertex_range.offset[()] + self.pairs.s[i_pair], self.pbd.vertex_range.offset[()] + i_vq],
            dt=gs.qd_int,
        )
        return verts, qd.Vector([1.0, -1.0], dt=gs.qd_float), r
