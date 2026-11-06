"""The contact pairs of the material point method (MPM) with the implicit finite element method (FEM) as a pair kind of
the NewtonSolver (see newton_contact.py).

The stage functions are those of `mpm_fem_func`, and this module binds them to the data of one scene, on the
participants of MPM and FEM (see FEMSolver.build_newton_participant).
"""

import quadrants as qd

from genesis.engine.solvers.fem_newton import FemNewtonParticipant
from genesis.engine.solvers.mpm_solver import MpmNewtonParticipant

from . import mpm_fem_func
from .contact_candidates import CouplingPairs


@qd.data_oriented
class MpmFemContact(CouplingPairs):
    """Pair kind of the contacts between the material point method (MPM) and the finite element method (FEM), the pair
    of a particle and its nearest surface triangle. A pair writes 4 doublets on its particle and the vertices of its
    triangle, and the 10 upper blocks between them."""

    n_doublets = 4
    n_triplets = 10
    build_pairs = staticmethod(mpm_fem_func.build_fem_pairs)

    def __init__(self, mpm: MpmNewtonParticipant, fem: FemNewtonParticipant, query: mpm_fem_func.FemTriangleQuery):
        self.fem = fem
        self.query = query
        super().__init__(mpm)

    def build_query(self, f: int) -> None:
        self.query.build(f)

    @qd.func(requires_top_level=True)
    def func_query(self, f: int):
        mpm_fem_func.func_query_triangles(
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
        mpm_fem_func.func_score_triangles(
            f, self.mpm.particles, self.mpm.grid, self.fem, self.candidates, self.mpm.mpm_info
        )

    @qd.func(requires_top_level=True)
    def func_build_pairs(self, f: int):
        mpm_fem_func.func_build_fem_pairs(
            f,
            self.mpm.mpm_info.substep_dt[()],
            self.mpm.particles,
            self.mpm.grid,
            self.pairs,
            self.fem,
            self.query,
            self.candidates,
            self.mpm.particles_info,
            self.mpm.mpm_info,
        )

    @qd.func(requires_top_level=True)
    def func_pair_terms(self, f: int, i_kind: qd.template(), contact: qd.template()):
        mpm_fem_func.func_fem_pair_terms(
            f,
            contact.matrices.doublet_offsets[i_kind],
            contact.matrices.triplet_offsets[i_kind],
            self.mpm,
            self.fem,
            self.pairs,
            contact,
        )

    @qd.func(requires_top_level=True)
    def func_pair_energy(self, f: int, energies: qd.template()):
        mpm_fem_func.func_fem_pair_energy(f, self.mpm, self.fem, self.pairs, energies)
