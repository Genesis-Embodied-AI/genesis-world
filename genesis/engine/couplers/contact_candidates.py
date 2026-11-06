"""Nearest contact candidate of every particle of the material point method (MPM) in one coupling.

The box query of a sorted particle against the tree of the partners of a coupling records every partner it meets as a
candidate pair. The narrow phase of the coupling scores each candidate (its distance to the particle, infinite for a
candidate it rejects) and gives it a secondary score that breaks the ties. Sorted by particle, the candidates of a
particle form a segment, and the nearest one of every segment is kept: the smallest score, then among the candidates
within `tie_tol` of it the largest secondary score, then the first candidate. Every stage runs one thread per candidate.
"""

from collections.abc import Iterator

import quadrants as qd

from genesis.engine.numerics.bcoo_matrix import (
    BCOOMatrixScratch,
    bcoo_matrix_sizes,
    func_extract_unique_bcoo_entries,
    func_report_triplet_demand,
    init_bcoo_matrix,
    sort_bcoo,
)
from genesis.engine.solvers.newton_linear import MIN_SLOTS, grown_capacity, retire_scratch
from genesis.engine.solvers.newton_solver import NewtonDevice, NewtonHost
from genesis.utils.array_class import QUERY_RESULTS, BVHQueryResults
from genesis.utils.data import Data, DataItem, DataKind, Size, field, iter_data
from genesis.utils.misc import qd_to_numpy

CANDIDATES = Size("candidates")


class ContactCandidates(Data, kind=DataKind.SCRATCH):
    """The candidate pairs that the box queries of the particles of a coupling find, and the nearest one of every
    particle.

    `results` holds the (environment, partner, sorted particle) triplet of every candidate. `order` sorts the
    candidates by particle, the block of a candidate being its score. `secondary` holds the secondary score of every
    candidate, and the `best_*` arrays the best score, secondary score and candidate of every segment of a particle."""

    results: BVHQueryResults
    order: BCOOMatrixScratch
    secondary: qd.Tensor = field(float, shape=(CANDIDATES,), ndarray=True)
    best_score: qd.Tensor = field(float, shape=(CANDIDATES,), ndarray=True)
    best_secondary: qd.Tensor = field(float, shape=(CANDIDATES,), ndarray=True)
    best_candidate: qd.Tensor = field(int, shape=(CANDIDATES,), ndarray=True)


def build_contact_candidates(max_candidates: int) -> ContactCandidates:
    """Build the candidates of a coupling at a capacity, with no candidate."""
    candidates = ContactCandidates(
        sizes={
            CANDIDATES: max_candidates,
            ContactCandidates.results: {QUERY_RESULTS: max_candidates},
            ContactCandidates.order: bcoo_matrix_sizes(max_candidates, block_shape=(1, 1)),
        }
    )
    candidates.results.count.fill(0)
    init_bcoo_matrix(candidates.order, n_triplets=0)
    return candidates


@qd.func(requires_top_level=True)
def func_reset_candidates(candidates: ContactCandidates):
    for _ in range(1):
        candidates.results.count[0] = 0


@qd.func
def set_candidate_score(i_cand: int, s: int, score: float, secondary: float, candidates: ContactCandidates):
    """Record the scores of candidate `i_cand` of sorted particle `s`, an infinite score rejecting it."""
    candidates.order.triplet_row[i_cand] = s
    candidates.order.triplet_col[i_cand] = 0
    candidates.order.triplet_val[i_cand] = score
    candidates.secondary[i_cand] = secondary


@qd.func(requires_top_level=True)
def func_report_candidates(candidates: ContactCandidates):
    """Set the count of the candidates to sort, those the results hold."""
    for _ in range(1):
        func_report_triplet_demand(candidates.results.count[0], candidates.order)


@qd.func(requires_top_level=True)
def func_select_nearest(tie_tol: float, candidates: ContactCandidates):
    """Keep the nearest candidate of every particle, once the candidates are sorted by particle (see sort_bcoo).

    The segment of every particle holds its best candidate in `best_candidate`, or the capacity if every candidate of
    the particle was rejected."""
    # The unique entries leave the count of segments alone when there is no candidate
    for _ in range(1):
        candidates.order.bcoo_nnz[()] = 0
    func_extract_unique_bcoo_entries(candidates.order)
    qd.loop_config(name="contact_candidates_reset")
    for i_seg in range(candidates.order.bcoo_nnz[()]):
        candidates.best_score[i_seg] = qd.math.inf
        candidates.best_secondary[i_seg] = -qd.math.inf
        candidates.best_candidate[i_seg] = candidates.secondary.shape[0]
    qd.loop_config(name="contact_candidates_score")
    for i_sorted in range(candidates.order.n_triplets[()]):
        i_cand = candidates.order.triplet_perm[i_sorted]
        score = candidates.order.triplet_val[i_cand]
        if score < qd.math.inf:
            qd.atomic_min(candidates.best_score[candidates.order.seg_ids[i_sorted]], score)
    qd.loop_config(name="contact_candidates_secondary")
    for i_sorted in range(candidates.order.n_triplets[()]):
        i_cand = candidates.order.triplet_perm[i_sorted]
        i_seg = candidates.order.seg_ids[i_sorted]
        score = candidates.order.triplet_val[i_cand]
        if score < qd.math.inf and score <= candidates.best_score[i_seg] + tie_tol:
            qd.atomic_max(candidates.best_secondary[i_seg], candidates.secondary[i_cand])
    qd.loop_config(name="contact_candidates_winner")
    for i_sorted in range(candidates.order.n_triplets[()]):
        i_cand = candidates.order.triplet_perm[i_sorted]
        i_seg = candidates.order.seg_ids[i_sorted]
        score = candidates.order.triplet_val[i_cand]
        if (
            score < qd.math.inf
            and score <= candidates.best_score[i_seg] + tie_tol
            and candidates.secondary[i_cand] >= candidates.best_secondary[i_seg]
        ):
            qd.atomic_min(candidates.best_candidate[i_seg], i_cand)


@qd.data_oriented
class CouplingPairs(NewtonDevice):
    """Pair kind of the contacts of the material point method (MPM) particles with the partners of one coupling, the
    nearest partner of every particle (see NewtonContact).

    Before the Newton solve of every substep, `prepare` rebuilds the query of the partners (`build_query`), records the
    candidates of the box query of every particle (`func_query`), scores them by the narrow phase of the coupling
    (`func_score`), and writes the pair of the nearest one (`func_build_pairs`) into `pairs`, which `build_pairs` builds
    at a capacity. The candidates and the pairs grow on overflow. A coupling also provides the `n_doublets`,
    `n_triplets`, `func_pair_terms` and `func_pair_energy` of a pair kind."""

    def __init__(self, mpm):
        self.mpm = mpm
        self.candidates = build_contact_candidates(MIN_SLOTS)
        self.pairs = self.build_pairs(MIN_SLOTS)
        # The candidates and pairs that grew out of, kept referenced (see retire_scratch)
        self.scratches_replaced = []

    @property
    def data(self) -> Iterator[DataItem]:
        yield from iter_data(self, "candidates")
        yield from iter_data(self, "pairs")

    @qd.func
    def func_pair_count(self):
        return self.pairs.n_pairs[()]

    def prepare(self, f: int) -> None:
        self.build_query(f)
        kernel_query_candidates(f, self)
        n_candidates = int(qd_to_numpy(self.candidates.results.count)[0])
        if n_candidates > self.candidates.secondary.shape[0]:
            retire_scratch(self.candidates, self.scratches_replaced)
            self.candidates = build_contact_candidates(grown_capacity(n_candidates))
            kernel_query_candidates(f, self)
        kernel_score_candidates(f, self)
        sort_bcoo(self.candidates.order)
        kernel_build_pairs(f, self)
        n_pairs = int(qd_to_numpy(self.pairs.n_pairs))
        if n_pairs > self.pairs.s.shape[0]:
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
    pair_kind.func_query(f)


@qd.kernel
def kernel_score_candidates(f: int, pair_kind: qd.template()):
    pair_kind.func_score(f)


@qd.kernel
def kernel_build_pairs(f: int, pair_kind: qd.template()):
    pair_kind.func_build_pairs(f)
