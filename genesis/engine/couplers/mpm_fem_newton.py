"""The contact pairs of the material point method (MPM) with the implicit finite element method (FEM) as a pair kind of
the NewtonSolver (see newton_contact.py).

The stage functions are those of `mpm_fem_func`, and this module binds them to the data of one scene, on the
participants of MPM and FEM (see FEMSolver.build_newton_participant).
"""

from collections.abc import Iterator

import quadrants as qd

from genesis.engine.numerics.bcoo_matrix import sort_bcoo
from genesis.engine.solvers.fem_newton import FemNewtonParticipant
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant
from genesis.engine.solvers.newton_linear import MIN_SLOTS, grown_capacity, retire_scratch
from genesis.engine.solvers.newton_solver import NewtonDevice, NewtonHost
from genesis.utils.data import DataItem, iter_data
from genesis.utils.misc import qd_to_numpy

from . import mpm_fem_func
from .contact_candidates import build_contact_candidates


@qd.data_oriented
class MpmFemContact(NewtonDevice):
    """Pair kind of the contacts between the material point method (MPM) and the finite element method (FEM).

    `pairs` holds the pairs of the substep, which `candidates` selects from the box queries of `query`. A pair writes 4
    doublets on its particle and the vertices of its triangle, and the 10 upper blocks between them."""

    n_doublets = 4
    n_triplets = 10

    def __init__(self, mpm: MpmNewtonParticipant, fem: FemNewtonParticipant, query: mpm_fem_func.FemTriangleQuery):
        self.mpm = mpm
        self.fem = fem
        self.query = query
        self.fem_candidates = build_contact_candidates(MIN_SLOTS)
        self.fem_pairs = mpm_fem_func.build_fem_pairs(MIN_SLOTS)
        # The candidates and pairs that grew out of, kept referenced (see retire_scratch)
        self.scratches_replaced = []

    @property
    def data(self) -> Iterator[DataItem]:
        """Yield the candidates and the pairs, rebuilt every substep."""
        yield from iter_data(self, "fem_candidates")
        yield from iter_data(self, "fem_pairs")

    @qd.func
    def func_pair_count(self):
        return self.fem_pairs.n_pairs[()]

    @qd.func(requires_top_level=True)
    def func_pair_terms(self, f: int, i_kind: qd.template(), contact: qd.template()):
        mpm_fem_func.func_fem_pair_terms(
            f,
            contact.matrices.doublet_offsets[i_kind],
            contact.matrices.triplet_offsets[i_kind],
            self.mpm,
            self.fem,
            self.fem_pairs,
            contact,
        )

    @qd.func(requires_top_level=True)
    def func_pair_energy(self, f: int, control: qd.template(), is_trial: qd.template()):
        if qd.static(is_trial):
            mpm_fem_func.func_fem_pair_energy(f, self.mpm, self.fem, self.fem_pairs, control.energy_trial)
        else:
            mpm_fem_func.func_fem_pair_energy(f, self.mpm, self.fem, self.fem_pairs, control.energy_start)


@qd.kernel
def kernel_query_triangles(f: int, contact: qd.template()):
    mpm_fem_func.func_query_triangles(
        f,
        contact.mpm.mpm_info.substep_dt[()],
        contact.mpm.particles,
        contact.mpm.grid,
        contact.query,
        contact.fem_candidates,
        contact.mpm.mpm_info,
    )


@qd.kernel
def kernel_score_triangles(f: int, contact: qd.template()):
    mpm_fem_func.func_score_triangles(
        f, contact.mpm.particles, contact.mpm.grid, contact.fem, contact.fem_candidates, contact.mpm.mpm_info
    )


@qd.kernel
def kernel_build_fem_pairs(f: int, contact: qd.template()):
    mpm_fem_func.func_build_fem_pairs(
        f,
        contact.mpm.mpm_info.substep_dt[()],
        contact.mpm.particles,
        contact.mpm.grid,
        contact.fem_pairs,
        contact.fem,
        contact.query,
        contact.fem_candidates,
        contact.mpm.particles_info,
        contact.mpm.mpm_info,
    )


class MpmFemContactHost(NewtonHost):
    """Host stages of the MPM-FEM contact pairs before the joint solve: the triangle trees, the candidates of every
    particle, and the pairs of the nearest triangles, each grown on overflow."""

    def __init__(self, device: MpmFemContact):
        self._device = device

    def newton_prepare(self, f):
        device = self._device
        device.query.build(f)
        kernel_query_triangles(f, device)
        n_candidates = int(qd_to_numpy(device.fem_candidates.results.count)[0])
        if n_candidates > device.fem_candidates.secondary.shape[0]:
            retire_scratch(device.fem_candidates, device.scratches_replaced)
            device.fem_candidates = build_contact_candidates(grown_capacity(n_candidates))
            kernel_query_triangles(f, device)
        kernel_score_triangles(f, device)
        sort_bcoo(device.fem_candidates.order)
        kernel_build_fem_pairs(f, device)
        n_pairs = int(qd_to_numpy(device.fem_pairs.n_pairs))
        if n_pairs > device.fem_pairs.s.shape[0]:
            retire_scratch(device.fem_pairs, device.scratches_replaced)
            device.fem_pairs = mpm_fem_func.build_fem_pairs(grown_capacity(n_pairs))
            kernel_build_fem_pairs(f, device)
