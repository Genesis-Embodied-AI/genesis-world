"""Nearest contact candidate of every particle of the material point method (MPM) in one coupling."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import func_bvh_query_aabb, func_no_filter
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
from genesis.engine.solvers.newton_contact import append_pair, weighted_block, weighted_vector
from genesis.engine.solvers.newton_linear import MIN_SLOTS, PAIRS, grown_capacity, retire_scratch
from genesis.engine.solvers.newton_solver import NewtonDevice, NewtonHost
from genesis.utils.array_class import QUERY_RESULTS, BVHQueryResults
from genesis.utils.data import Data, DataItem, DataKind, Size, field, iter_data
from genesis.utils.misc import qd_to_numpy

CANDIDATES = Size("candidates")


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
class CouplingPairs(NewtonDevice):
    """Pair kind of the contacts of the material point method (MPM) particles with the partners of one coupling, the
    nearest partner of every particle (see NewtonContact).

    A coupling provides the `query` of its partners, `pairs_type`, `n_doublets` and `n_triplets`, and three hooks: the
    narrow phase `candidate_score`, `build_pair`, and `pair_vertices`, the vertices of a pair with their weights in its
    relative displacement."""

    tie_ratio = 0.0

    def __init__(self, mpm):
        self.mpm = mpm
        self.candidates = build_contact_candidates(MIN_SLOTS, self.tie_ratio)
        self.pairs = self.build_pairs(MIN_SLOTS)
        # The candidates and pairs that grew out of, kept referenced (see retire_scratch)
        self.scratches_replaced = []
        self.block_rows = tuple(i_v_ for i_v_ in range(self.n_doublets) for _ in range(i_v_, self.n_doublets))
        self.block_cols = tuple(j_v_ for i_v_ in range(self.n_doublets) for j_v_ in range(i_v_, self.n_doublets))

    def build_pairs(self, capacity: int):
        """Build the pairs of the coupling at a capacity, with no pair."""
        pairs = self.pairs_type(sizes={PAIRS: capacity})
        pairs.n_pairs.fill(0)
        return pairs

    @property
    def data(self) -> Iterator[DataItem]:
        yield from iter_data(self, "candidates")
        yield from iter_data(self, "pairs")

    @qd.func
    def func_pair_count(self):
        return self.pairs.n_pairs[()]

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
        """Energy, gradient and Hessian of the rows of a pair in its relative displacement at the current iterate, with
        its vertices and their weights in that displacement (see pair_vertices)."""
        verts, weights, u_rel = self.pair_vertices(f, i_pair)
        energy, g, H = pyramid_terms(
            self.pairs.normal[i_pair],
            self.pairs.friction[i_pair],
            self.pairs.stiffness[i_pair],
            self.pairs.target[i_pair],
            u_rel,
        )
        return energy, g, H, verts, weights

    @qd.func(requires_top_level=True)
    def func_pair_terms(self, f: int, i_kind: qd.template(), contact: qd.template()):
        """Write the gradient w_k g of every vertex of every pair at the current iterate, and the upper blocks w_k w_l H
        between them."""
        qd.loop_config(name="mpm_pair_terms")
        for i_pair in range(self.pairs.n_pairs[()]):
            _, g, H, verts, weights = self.pair_terms(f, i_pair)
            i_db_start = contact.matrices.doublet_offsets[i_kind] + qd.static(self.n_doublets) * i_pair
            i_t_start = contact.matrices.triplet_offsets[i_kind] + qd.static(self.n_triplets) * i_pair
            for i_v_ in range(qd.static(self.n_doublets)):
                contact.write_doublet(i_db_start + i_v_, verts[i_v_], weighted_vector(weights[i_v_], g))
            block_rows = qd.Vector(self.block_rows, dt=gs.qd_int)
            block_cols = qd.Vector(self.block_cols, dt=gs.qd_int)
            for i_t_ in range(qd.static(self.n_triplets)):
                i_v_ = block_rows[i_t_]
                j_v_ = block_cols[i_t_]
                contact.write_triplet(
                    i_t_start + i_t_, verts[i_v_], verts[j_v_], weighted_block(weights[i_v_], weights[j_v_], H)
                )

    @qd.func(requires_top_level=True)
    def func_pair_energy(self, f: int, energies: qd.template()):
        """Add the energy of the rows of every pair at the current iterate into the entry of its environment."""
        qd.loop_config(name="mpm_pair_energy")
        for i_pair in range(self.pairs.n_pairs[()]):
            energy, g, H, verts, weights = self.pair_terms(f, i_pair)
            i_b = self.pairs.i_b[i_pair]
            qd.atomic_add(energies[i_b], energy)

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
        kernel_build_pairs(f, self)
        n_pairs = int(qd_to_numpy(self.pairs.n_pairs))
        if n_pairs > self.pairs.sorted_idx.shape[0]:
            self.grow_pairs(grown_capacity(n_pairs))
            kernel_build_pairs(f, self)

    def grow_pairs(self, capacity: int) -> None:
        retire_scratch(self.pairs, self.scratches_replaced)
        self.pairs = self.build_pairs(capacity)


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
