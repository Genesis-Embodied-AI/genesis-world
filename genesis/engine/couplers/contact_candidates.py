"""Nearest contact candidate of every particle of the material point method (MPM) in one coupling."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import build_bvh, func_bvh_query_aabb, func_no_filter, get_bvh_data
from genesis.engine.numerics.bcoo_matrix import (
    BCOOMatrixScratch,
    bcoo_matrix_sizes,
    func_extract_unique_bcoo_entries,
    func_report_triplet_demand,
    init_bcoo_matrix,
    sort_bcoo,
)
from genesis.engine.solvers.mpm_data import PYRAMID_ROWS, MpmInfo
from genesis.engine.solvers.mpm_func import contact_reach, particle_of_sorted, pyramid_terms
from genesis.engine.solvers.newton_contact import PairKind, append_pair
from genesis.engine.solvers.newton_linear import MIN_SLOTS, grown_capacity, retire_scratch
from genesis.engine.solvers.newton_solver import NewtonHost
from genesis.utils.array_class import QUERY_RESULTS, BVHQueryResults
from genesis.utils.data import Data, DataItem, DataKind, Size, field, iter_data
from genesis.utils.misc import qd_to_numpy

CANDIDATES = Size("candidates")

# Half extent of the inverted box of a leaf that must meet no query of a coupling's tree, finite so that its center
# gets a Morton code
EMPTY_BOX_HALF = 1e10


@qd.data_oriented
class LeafTrees:
    """The linear bounding volume hierarchies (LBVH) of a coupling, one per environment over the same leaves (see
    genesis.engine.bvh), whose boxes a kernel of the coupling fits before every build."""

    def __init__(self, n_envs: int, n_leaves: int, is_active: bool = True):
        self.bvh_state, self.bvh_config = get_bvh_data(n_envs, max(n_leaves, 1), is_active=is_active)
        self.tree = self.bvh_state.tree
        self.leaves = self.bvh_state.leaves

    def build_trees(self):
        build_bvh(self.bvh_state, self.bvh_config, eps=gs.EPS)


@qd.func
def empty_box():
    """Box of a leaf that meets no query."""
    return qd.Vector([EMPTY_BOX_HALF] * 3, dt=gs.qd_float), qd.Vector([-EMPTY_BOX_HALF] * 3, dt=gs.qd_float)


class ContactCandidates(Data, kind=DataKind.SCRATCH):
    """The candidates that the box queries of the particles of a coupling find, and the nearest one of every
    particle."""

    results: BVHQueryResults
    order: BCOOMatrixScratch
    secondary: qd.Tensor = field(float, shape=(CANDIDATES,), ndarray=True)
    best_score: qd.Tensor = field(float, shape=(CANDIDATES,), ndarray=True)
    best_secondary: qd.Tensor = field(float, shape=(CANDIDATES,), ndarray=True)
    best_candidate: qd.Tensor = field(int, shape=(CANDIDATES,), ndarray=True)
    # Fraction of the grid spacing within which the scores of two candidates tie, their secondary score breaking the tie
    # (see func_select_nearest)
    tie_ratio: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT, ndarray=True)


def build_contact_candidates(max_candidates: int, tie_ratio: float) -> ContactCandidates:
    """Build the candidates of a coupling at a capacity, with no candidate, and the tie ratio of its scores."""
    candidates = ContactCandidates(
        sizes={
            CANDIDATES: max_candidates,
            ContactCandidates.results: {QUERY_RESULTS: max_candidates},
            ContactCandidates.order: bcoo_matrix_sizes(max_candidates, block_shape=(1, 1)),
        },
        values={"tie_ratio": tie_ratio},
    )
    candidates.results.count.fill(0)
    init_bcoo_matrix(candidates.order, n_triplets=0)
    return candidates


# =============================================== Layer 0: selection ================================================


@qd.func
def is_within_tie(score: float, best_score: float, tie_ratio: float, dx: float):
    """Whether a candidate ties with the best score of its particle, within the tie ratio of the grid spacing."""
    return score < qd.math.inf and score <= best_score + tie_ratio * dx


@qd.func
def pair_relative_displacement(weights, values):
    """Relative displacement of a pair from the displacements of its vertices, row k of values, and their weights."""
    return values.transpose() @ weights


# ============================================== Layer 1: candidates ================================================


@qd.func
def set_candidate_score(i_cand: int, i_ps: int, score: float, secondary: float, candidates: ContactCandidates):
    """Record the scores of a candidate of a sorted particle, an infinite score rejecting it."""
    candidates.order.triplets_row[i_cand] = i_ps
    candidates.order.triplets_col[i_cand] = 0
    candidates.order.triplets_val[i_cand] = score
    candidates.secondary[i_cand] = secondary


@qd.func(requires_top_level=True)
def func_select_nearest(candidates: ContactCandidates, mpm_info: MpmInfo):
    """Keep in `best_candidate` the nearest candidate of every particle, or the capacity if every candidate of the
    particle was rejected, once the candidates are sorted by particle (see sort_bcoo)."""
    # The unique entries leave the count of segments alone when there is no candidate
    for _ in range(1):
        candidates.order.bcoo_nnz[()] = 0
    func_extract_unique_bcoo_entries(candidates.order)
    qd.loop_config(name="contact_candidates_reset")
    for i_nz in range(candidates.order.bcoo_nnz[()]):
        candidates.best_score[i_nz] = qd.math.inf
        candidates.best_secondary[i_nz] = -qd.math.inf
        candidates.best_candidate[i_nz] = candidates.secondary.shape[0]
    qd.loop_config(name="contact_candidates_score")
    for i_t in range(candidates.order.n_triplets[()]):
        i_cand = candidates.order.triplets_source_idx[i_t]
        i_nz = candidates.order.triplets_segment_idx[i_t]
        score = candidates.order.triplets_val[i_cand]
        if score < qd.math.inf:
            qd.atomic_min(candidates.best_score[i_nz], score)
    qd.loop_config(name="contact_candidates_secondary")
    for i_t in range(candidates.order.n_triplets[()]):
        i_cand = candidates.order.triplets_source_idx[i_t]
        i_nz = candidates.order.triplets_segment_idx[i_t]
        score = candidates.order.triplets_val[i_cand]
        if is_within_tie(score, candidates.best_score[i_nz], candidates.tie_ratio[()], mpm_info.dx[()]):
            qd.atomic_max(candidates.best_secondary[i_nz], candidates.secondary[i_cand])
    qd.loop_config(name="contact_candidates_winner")
    for i_t in range(candidates.order.n_triplets[()]):
        i_cand = candidates.order.triplets_source_idx[i_t]
        i_nz = candidates.order.triplets_segment_idx[i_t]
        score = candidates.order.triplets_val[i_cand]
        if (
            is_within_tie(score, candidates.best_score[i_nz], candidates.tie_ratio[()], mpm_info.dx[()])
            and candidates.secondary[i_cand] >= candidates.best_secondary[i_nz]
        ):
            qd.atomic_min(candidates.best_candidate[i_nz], i_cand)


@qd.data_oriented
class CouplingPairs(PairKind):
    """Pair kind of the contacts of the material point method (MPM) particles with the partners of one coupling, the
    nearest partner of every particle (see NewtonContact).

    A coupling provides the `query` of its partners, `pairs_type`, `n_doublets` and `n_triplets`, and three hooks: the
    narrow phase `candidate_score`, `build_pair`, and `pair_vertices`, the vertices of a pair with their weights in its
    relative displacement."""

    tie_ratio = 0.0

    def __init__(self, mpm):
        self.mpm = mpm
        self.candidates = build_contact_candidates(MIN_SLOTS, self.tie_ratio)
        super().__init__()

    @property
    def data(self) -> Iterator[DataItem]:
        yield from iter_data(self, "candidates")
        yield from super().data

    @qd.func
    def add_pair(
        self,
        i_ps: int,
        i_b: int,
        normal: qd.types.vector(3),
        friction: float,
        stiffness: qd.types.vector(PYRAMID_ROWS),
        target: qd.types.vector(PYRAMID_ROWS),
    ):
        """Append a pair with the rows of its friction pyramid, and return its slot, or -1 past the capacity."""
        i_pair = append_pair(self.pairs.n_pairs, self.pairs.sorted_idx.shape[0])
        if i_pair >= 0:
            self.pairs.sorted_idx[i_pair] = i_ps
            self.pairs.i_b[i_pair] = i_b
            self.pairs.normal[i_pair] = normal
            self.pairs.friction[i_pair] = friction
            self.pairs.stiffness[i_pair] = stiffness
            self.pairs.target[i_pair] = target
        return i_pair

    @qd.func
    def pair_terms(self, f: int, i_pair: int):
        """Environment, energy, gradient and Hessian of the rows of a pair in its relative displacement at the current
        iterate, with its vertices and their weights in that displacement (see pair_vertices)."""
        verts, weights, u_rel = self.pair_vertices(f, i_pair)
        energy, g, H = pyramid_terms(
            self.pairs.normal[i_pair],
            self.pairs.friction[i_pair],
            self.pairs.stiffness[i_pair],
            self.pairs.target[i_pair],
            u_rel,
        )
        return self.pairs.i_b[i_pair], energy, g, H, verts, weights

    def prepare(self, f: int) -> None:
        self.build_query(f)
        kernel_query_candidates(f, self)
        n_candidates = int(qd_to_numpy(self.candidates.results.count)[0])
        if n_candidates > self.candidates.secondary.shape[0]:
            retire_scratch(self.candidates, self.scratches_replaced)
            self.candidates = build_contact_candidates(grown_capacity(n_candidates), self.tie_ratio)
            kernel_query_candidates(f, self)
        kernel_score_candidates(f, self)
        sort_bcoo(self.candidates.order)
        self.fill_pairs(lambda: kernel_build_pairs(f, self))


class CouplingPairsHost(NewtonHost):
    """Host stages of a coupling of MPM: its pairs of the substep, built before the joint solve."""

    def __init__(self, device: CouplingPairs):
        self._device = device

    def newton_prepare(self, f):
        self._device.prepare(f)


@qd.kernel
def kernel_query_candidates(f: int, pair_kind: qd.template()):
    """Record as candidates the partners whose leaf box meets the band of every valid particle at its own speed (see
    contact_band_width), the leaves spanning the motion of the partners over the substep."""
    for _ in range(1):
        pair_kind.candidates.results.count[0] = 0
    qd.loop_config(name="mpm_query_partners")
    for i_ps in range(pair_kind.mpm.grid.n_valid[()]):
        i_p, i_b, cell = particle_of_sorted(i_ps, pair_kind.mpm.grid, pair_kind.mpm.mpm_info)
        x_p = pair_kind.mpm.particles.pos[f, i_p, i_b]
        reach = contact_reach(
            pair_kind.mpm.particles.vel[f, i_p, i_b],
            pair_kind.mpm.mpm_info.substep_dt[()],
            pair_kind.mpm.mpm_info.dx[()],
        )
        func_bvh_query_aabb(
            i_b,
            i_ps,
            x_p - reach,
            x_p + reach,
            pair_kind.query.tree,
            pair_kind.candidates.results,
            func_no_filter,
            pair_kind.query,
        )


@qd.kernel
def kernel_score_candidates(f: int, pair_kind: qd.template()):
    """Score every candidate by the narrow phase of the coupling (see candidate_score)."""
    # The candidates to sort are those the results hold
    for _ in range(1):
        func_report_triplet_demand(pair_kind.candidates.results.count[0], pair_kind.candidates.order)
    qd.loop_config(name="mpm_score_partners")
    for i_cand in range(pair_kind.candidates.order.n_triplets[()]):
        triplet = pair_kind.candidates.results.triplets[i_cand]
        score, secondary = pair_kind.candidate_score(f, triplet[0], triplet[1], triplet[2])
        set_candidate_score(i_cand, triplet[2], score, secondary, pair_kind.candidates)


@qd.kernel
def kernel_build_pairs(f: int, pair_kind: qd.template()):
    """Pair every valid particle with its nearest candidate (see build_pair), once the candidates are sorted by
    particle."""
    func_select_nearest(pair_kind.candidates, pair_kind.mpm.mpm_info)
    for _ in range(1):
        pair_kind.pairs.n_pairs[()] = 0
    qd.loop_config(name="mpm_build_pairs")
    for i_nz in range(pair_kind.candidates.order.bcoo_nnz[()]):
        i_cand = pair_kind.candidates.best_candidate[i_nz]
        if i_cand < pair_kind.candidates.secondary.shape[0]:
            triplet = pair_kind.candidates.results.triplets[i_cand]
            pair_kind.build_pair(f, triplet[0], triplet[1], triplet[2], pair_kind.candidates.order.triplets_val[i_cand])
