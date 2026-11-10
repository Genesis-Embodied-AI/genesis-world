"""Contact pairs of the implicit material point method (MPM) with position based dynamics (PBD) in the joint Newton
solve (see pbd_newton.py for the PBD unknown and its stages)."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, get_bvh_data
from genesis.engine.solvers.mpm_data import PYRAMID_ROWS
from genesis.engine.solvers.mpm_func import (
    EMPTY_BOX_HALF,
    contact_gap,
    is_within_band,
    particle_invweight,
    particle_of_sorted,
    penalty_rows,
)
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_linear import PAIRS
from genesis.engine.solvers.pbd_newton import PBD_PARTICLES, PbdNewtonParticipant, is_moving
from genesis.engine.solvers.pbd_solver import friction_correction, func_insert_nearest, separation_correction
from genesis.utils.array_class import BVHQueryResults
from genesis.utils.data import ENVS, Data, DataItem, DataKind, field, iter_data

from .contact_candidates import CouplingPairs, pair_relative_displacement
from .mpm_sph_func import kernel_fit_mpm_particle_leaves

# Obstacle candidates of a PBD particle over a substep: the nearest OBSTACLE_NEIGHBORS MPM particles within its obstacle
# reach (see PbdContactParams)
OBSTACLE_NEIGHBORS = 32


class MpmPbdPairs(Data, kind=DataKind.SCRATCH):
    """The pairs of the particles with their nearest PBD particle (see MpmPbdContact.build_pair)."""

    n_pairs: qd.Tensor = field(int, ndarray=True)
    sorted_idx: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    i_b: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    pbd_particle_idx: qd.Tensor = field(int, shape=(PAIRS,), ndarray=True)
    normal: qd.Tensor = field((float, 3), shape=(PAIRS,), ndarray=True)
    friction: qd.Tensor = field(float, shape=(PAIRS,), ndarray=True)
    stiffness: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)
    target: qd.Tensor = field((float, PYRAMID_ROWS), shape=(PAIRS,), ndarray=True)


# =============================================== Layer 0: geometry ===============================================


@qd.func
def contact_distance(radius: float, thickness: float):
    """Distance between an MPM particle and the center of a PBD particle at which their contact starts."""
    return radius + thickness


@qd.func
def obstacle_reach(reach_ratio: float, radius: float, thickness: float):
    """Distance from a PBD particle within which an MPM particle is an obstacle candidate, in contact distances."""
    return reach_ratio * contact_distance(radius, thickness)


@qd.func
def sweep_box(x0: qd.types.vector(3), x1: qd.types.vector(3), reach: float):
    """Box of the sweep of a point from x0 to x1, grown by a reach."""
    return qd.min(x0, x1) - reach, qd.max(x0, x1) + reach


@qd.func
def sweep_distance(x0: qd.types.vector(3), x1: qd.types.vector(3), point: qd.types.vector(3)):
    """Distance of a point to the nearer end of a sweep."""
    return qd.min((x1 - point).norm(), (x0 - point).norm())


@qd.func
def obstacle_overlap(x_q: qd.types.vector(3), x_m: qd.types.vector(3), target: float):
    """Whether a PBD particle overlaps an MPM particle within the contact distance, with their separation and its
    length."""
    separation = x_q - x_m
    distance = separation.norm()
    return distance < target and distance > gs.EPS, separation, distance


@qd.func
def sphere_contact(x_p: qd.types.vector(3), center: qd.types.vector(3)):
    """Distance of a point to the center of a sphere, and the outward normal of the sphere at its closest point."""
    separation = x_p - center
    distance = separation.norm()
    return distance, separation / distance


@qd.func
def partner_inverse_mass(mass: float, is_unknown: bool):
    """Inverse mass of a PBD particle in the inverse weight of a pair, zero for a particle the solve holds."""
    inv_mass = gs.qd_float(0.0)
    if is_unknown:
        inv_mass = 1.0 / mass
    return inv_mass


# ============================================== Layer 1: broad phase ==============================================


class PbdContactParams(Data, kind=DataKind.CONSTANT, shape=()):
    """Scene values of the MPM-PBD contact pairs, as device scalars: lengths in metres, the penalty stiffness of a row
    per unit of its effective mass, and the obstacle reach in contact distances."""

    radius: qd.Tensor = field(float, ndarray=True)
    thickness: qd.Tensor = field(float, ndarray=True)
    stiffness: qd.Tensor = field(float, ndarray=True)
    friction: qd.Tensor = field(float, ndarray=True)
    obstacle_reach: qd.Tensor = field(float, ndarray=True)


@qd.data_oriented
class PbdParticleQuery:
    """The linear bounding volume hierarchy (LBVH) over the position based dynamics (PBD) particles of every
    environment, in PBD's reordered layout (see genesis.engine.bvh)."""

    def __init__(self, pbd: PbdNewtonParticipant, mpm_info, contact: PbdContactParams, n_envs):
        self.pbd = pbd
        self.mpm_info = mpm_info
        self.contact = contact
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
    for i_b, i_q in qd.ndrange(query.leaves.aabbs_min.shape[0], query.leaves.aabbs_min.shape[1]):
        if query.pbd.solver.particles_ng_reordered[i_q, i_b].active:
            lo, hi = sweep_box(
                query.pbd.solver.particles_reordered[i_q, i_b].ipos,
                query.pbd.solver.particles_reordered[i_q, i_b].pos,
                contact_distance(query.contact.radius[()], query.contact.thickness[()]),
            )
            query.leaves.aabbs_min[i_b, i_q] = lo
            query.leaves.aabbs_max[i_b, i_q] = hi
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
    """The MPM particles at the start of the substep as immovable obstacles of the PBD particles, at the contact
    distance of the MPM-PBD rows so that PBD's own projection keeps its particles out of the MPM material."""

    def __init__(self, mpm_solver, pbd, contact: PbdContactParams, n_envs):
        self.mpm_particles = mpm_solver.particles
        self.mpm_particles_ng = mpm_solver.particles_ng
        self.pbd = pbd
        self.contact = contact
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
    def func_project(self, f: int, pbd: qd.template()):
        """The contacts of every PBD particle at its current position with the MPM particles, with friction."""
        func_pbd_obstacles(self, is_stabilization=False)

    @qd.func(requires_top_level=True)
    def func_stabilize(self, f: int, pbd: qd.template()):
        """The non-penetration of every PBD particle at its start position with the MPM particles."""
        func_pbd_obstacles(self, is_stabilization=True)


@qd.func
def func_obstacle_contact_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """Project a PBD particle out of an MPM particle to the contact distance, with the friction of Macklin et al. 2014
    (Eq. 23-24) on its tangential displacement over the substep."""
    f = obstacles.scratch.frame[()]
    particle = obstacles.pbd.particles_reordered[i_q, i_t]
    x_m = obstacles.mpm_particles.pos[f, i_a, i_t]
    target = contact_distance(obstacles.contact.radius[()], obstacles.contact.thickness[()])
    is_overlapping, separation, distance = obstacle_overlap(particle.pos, x_m, target)
    if is_overlapping:
        correction = separation_correction(separation, distance, target, 1.0)
        friction = friction_correction(
            particle.pos,
            particle.ipos,
            x_m,
            x_m,
            separation,
            distance,
            target,
            obstacles.pbd.particles_info_reordered[i_q, i_t].mu_s,
            obstacles.pbd.particles_info_reordered[i_q, i_t].mu_k,
            1.0,
        )
        obstacles.pbd.particles_reordered[i_q, i_t].dpos = particle.dpos + correction + friction
        obstacles.pbd.particles_reordered[i_q, i_t].dcount = particle.dcount + 1.0
    return True


@qd.func
def func_obstacle_start_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """The non-penetration of a PBD particle at its start position with an MPM particle, for the pre-stabilization."""
    f = obstacles.scratch.frame[()]
    particle = obstacles.pbd.particles_reordered[i_q, i_t]
    target = contact_distance(obstacles.contact.radius[()], obstacles.contact.thickness[()])
    is_overlapping, separation, distance = obstacle_overlap(
        particle.ipos, obstacles.mpm_particles.pos[f, i_a, i_t], target
    )
    if is_overlapping:
        correction = separation_correction(separation, distance, target, 1.0)
        obstacles.pbd.particles_reordered[i_q, i_t].dpos = particle.dpos + correction
        obstacles.pbd.particles_reordered[i_q, i_t].dcount = particle.dcount + 1.0
    return True


@qd.func
def func_obstacle_gather_filter(i_t: int, i_a: int, i_q: int, obstacles: qd.template()) -> bool:
    """Keep an MPM particle among the nearest OBSTACLE_NEIGHBORS obstacle candidates of a PBD particle if it lies within
    its obstacle reach, in contact distances of its start or predicted position."""
    f = obstacles.scratch.frame[()]
    distance = sweep_distance(
        obstacles.pbd.particles_reordered[i_q, i_t].ipos,
        obstacles.pbd.particles_reordered[i_q, i_t].pos,
        obstacles.mpm_particles.pos[f, i_a, i_t],
    )
    reach = obstacle_reach(
        obstacles.contact.obstacle_reach[()], obstacles.contact.radius[()], obstacles.contact.thickness[()]
    )
    if distance < reach:
        func_insert_nearest(
            i_q,
            i_t,
            i_a,
            distance,
            obstacles.scratch.n_neighbors,
            obstacles.scratch.neighbors_idx,
            obstacles.scratch.neighbors_dist,
            OBSTACLE_NEIGHBORS,
        )
    return True


@qd.kernel
def kernel_gather_obstacle_neighbors(obstacles: qd.template()):
    """The obstacle candidates of every free PBD particle for the substep, by a box query of the MPM trees around its
    start and predicted positions."""
    qd.loop_config(name="pbd_mpm_obstacle_neighbors")
    for i_q, i_b in qd.ndrange(obstacles.pbd._n_particles, obstacles.pbd._B):
        obstacles.scratch.n_neighbors[i_q, i_b] = 0
        if is_moving(i_q, i_b, obstacles.pbd):
            lo, hi = sweep_box(
                obstacles.pbd.particles_reordered[i_q, i_b].ipos,
                obstacles.pbd.particles_reordered[i_q, i_b].pos,
                obstacle_reach(
                    obstacles.contact.obstacle_reach[()], obstacles.contact.radius[()], obstacles.contact.thickness[()]
                ),
            )
            func_bvh_query_aabb(
                i_b, i_q, lo, hi, obstacles.tree, obstacles.results, func_obstacle_gather_filter, obstacles
            )


@qd.func(requires_top_level=True)
def func_pbd_obstacles(obstacles: qd.template(), is_stabilization: qd.template()):
    """Accumulate the obstacle corrections of every free PBD particle against its obstacle candidates of the substep, at
    its current or start position."""
    qd.loop_config(name="pbd_mpm_obstacles")
    for i_q, i_b in qd.ndrange(obstacles.pbd._n_particles, obstacles.pbd._B):
        for i_nb in range(obstacles.scratch.n_neighbors[i_q, i_b]):
            i_a = obstacles.scratch.neighbors_idx[i_q, i_nb, i_b]
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
    def candidate_score(self, f: int, i_b: int, i_q: int, i_ps: int):
        """Gap of a sorted particle to the sphere of a PBD particle at the substep start, infinite for a PBD particle at
        the position of the particle."""
        i_p, _, _ = particle_of_sorted(i_ps, self.mpm.grid, self.mpm.mpm_info)
        distance, normal = sphere_contact(
            self.mpm.particles.pos[f, i_p, i_b], self.pbd.solver.particles_reordered[i_q, i_b].ipos
        )
        score = qd.math.inf
        if distance > gs.EPS:
            score = contact_gap(
                distance, contact_distance(self.query.contact.radius[()], self.query.contact.thickness[()])
            )
        return score, 0.0

    @qd.func
    def build_pair(self, f: int, i_b: int, i_q: int, i_ps: int, phi: float):
        """Pair a sorted particle with a PBD particle if it lies within its contact range or short of it by less than
        `contact_band_width`."""
        i_p, _, _ = particle_of_sorted(i_ps, self.mpm.grid, self.mpm.mpm_info)
        distance, normal = sphere_contact(
            self.mpm.particles.pos[f, i_p, i_b], self.pbd.solver.particles_reordered[i_q, i_b].ipos
        )
        if is_within_band(
            phi,
            normal,
            self.mpm.particles.vel[f, i_p, i_b],
            self.pbd.solver.particles_reordered[i_q, i_b].vel,
            self.mpm.mpm_info.substep_dt[()],
            self.mpm.mpm_info.dx[()],
        ):
            stiffness, target = penalty_rows(
                self.query.contact.stiffness[()],
                particle_invweight(f, i_p, i_b, normal, self.mpm.particles, self.mpm.particles_info),
                partner_inverse_mass(
                    self.pbd.solver.particles_info_reordered[i_q, i_b].mass, is_moving(i_q, i_b, self.pbd.solver)
                ),
                phi,
            )
            i_pair = self.add_pair(i_ps, i_b, normal, self.query.contact.friction[()], stiffness, target)
            if i_pair >= 0:
                self.pairs.pbd_particle_idx[i_pair] = i_q

    @qd.func
    def pair_vertices(self, f: int, i_pair: int):
        """The particle and the PBD particle of a pair, of weights 1 and -1, with their relative displacement."""
        i_v_pbd = self.pairs.i_b[i_pair] * self.pbd.n_vertex_particles + self.pairs.pbd_particle_idx[i_pair]
        verts = qd.Vector(
            [
                self.mpm.vertex_range.offset[()] + self.pairs.sorted_idx[i_pair],
                self.pbd.vertex_range.offset[()] + i_v_pbd,
            ],
            dt=gs.qd_int,
        )
        weights = qd.Vector([1.0, -1.0], dt=gs.qd_float)
        values = qd.Matrix.rows(
            [self.mpm.func_vertex_value(f, self.pairs.sorted_idx[i_pair]), self.pbd.func_vertex_value(f, i_v_pbd)]
        )
        return verts, weights, pair_relative_displacement(weights, values)
