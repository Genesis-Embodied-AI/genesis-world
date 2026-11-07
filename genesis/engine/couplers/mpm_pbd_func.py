"""Contact pairs of the implicit material point method (MPM) with position based dynamics (PBD) in the joint Newton
solve (see pbd_newton.py for the PBD unknown and its stages)."""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import build_bvh, get_bvh_data
from genesis.engine.solvers.mpm_data import PYRAMID_ROWS
from genesis.engine.solvers.mpm_func import EMPTY_BOX_HALF, contact_band_width, contact_invweight, particle_of_sorted
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_linear import PAIRS
from genesis.engine.solvers.pbd_newton import PbdNewtonParticipant, is_moving
from genesis.utils.data import Data, DataKind, field

from .contact_candidates import CouplingPairs


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
            [self.mpm.vertex_offset[()] + self.pairs.s[i_pair], self.pbd.vertex_offset[()] + i_vq], dt=gs.qd_int
        )
        return verts, qd.Vector([1.0, -1.0], dt=gs.qd_float), r
