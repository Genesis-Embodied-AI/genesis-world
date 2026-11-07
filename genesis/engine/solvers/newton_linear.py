"""Linear system of the joint Newton solve over the rows of every participant that assembles one."""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.bcoo_matrix import (
    BCOOMatrixScratch,
    bcoo_matrix_sizes,
    func_discard_triplet,
    func_read_bcoo,
    func_report_triplet_demand,
    func_write_triplet,
    init_bcoo_matrix,
)
from genesis.engine.numerics.linear_pcg import PCG_ROWS, PCGWorkspaceScratch
from genesis.utils.data import Data, DataKind, Size, field, iter_state_arrays
from genesis.utils.misc import qd_destroy_data

# Block rows of a block-diagonal preconditioner
JACOBI_ROWS = Size("jacobi_rows")
# Capacity of the pairs of a contact pair kind (see newton_contact.py)
PAIRS = Size("pairs")
# Smallest capacity of the storage of the solve that grows on demand: triplets, slots and pairs
MIN_SLOTS = 64
# Slack a capacity gets when it grows on overflow: the new capacity is the demand times this factor
CAPACITY_GROW_FACTOR = 1.2


def retire_scratch(scratch: Data, retired: list) -> None:
    """Free the arrays of a struct that a larger one replaces, and keep the struct referenced in `retired`."""
    # FIXME: quadrants#1074 - the launch of a kernel is cached under the identity of its struct arguments, and the
    # entry outlives the struct, so a new struct taking the identity of a dead one would run on its arrays
    for _, array in iter_state_arrays(scratch):
        qd_destroy_data(array)
    retired.append(scratch)


def grown_capacity(demand: int) -> int:
    """Capacity of a struct grown to a demand, with the slack of CAPACITY_GROW_FACTOR."""
    return int(np.ceil(demand * CAPACITY_GROW_FACTOR))


class NewtonLinearScratch(Data, kind=DataKind.SCRATCH):
    """The matrix, right-hand side, solution and conjugate gradient vectors of the joint Newton solve."""

    matrix: BCOOMatrixScratch
    pcg: PCGWorkspaceScratch
    b_rhs: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)
    solution: qd.Tensor = field((float, 3), shape=(PCG_ROWS,), ndarray=True)


def build_newton_linear(max_rows: int, max_triplets: int) -> NewtonLinearScratch:
    """Build the linear system of a capacity of block rows and triplets, with no live row and no live triplet."""
    linear = NewtonLinearScratch(
        sizes={
            PCG_ROWS: max_rows,
            NewtonLinearScratch.matrix: bcoo_matrix_sizes(max_triplets, block_shape=(3, 3)),
            NewtonLinearScratch.pcg: {PCG_ROWS: max_rows},
        }
    )
    init_bcoo_matrix(linear.matrix, n_triplets=0)
    linear.pcg.n_rows.fill(0)
    linear.pcg.row_env.fill(0)
    return linear


class LinearBlock(Data, kind=DataKind.SCRATCH):
    """The share of one owner in the linear system: the counts of block rows and triplets it reports for the next
    assembly, and the start row and start triplet the layout gives it."""

    n_rows: qd.Tensor = field(int, ndarray=True)
    n_triplets: qd.Tensor = field(int, ndarray=True)
    row_start: qd.Tensor = field(int, ndarray=True)
    triplet_start: qd.Tensor = field(int, ndarray=True)


def build_linear_block(n_rows: int, n_triplets: int) -> LinearBlock:
    """Build the share of an owner reporting fixed counts of block rows and triplets."""
    block = LinearBlock(sizes={})
    block.n_rows.fill(n_rows)
    block.n_triplets.fill(n_triplets)
    block.row_start.fill(0)
    block.triplet_start.fill(0)
    return block


class BlockJacobiScratch(Data, kind=DataKind.SCRATCH):
    """The 3 x 3 diagonal blocks of the rows of one owner, then their inverses, a block Jacobi preconditioner."""

    blocks: qd.Tensor = field((float, (3, 3)), shape=(JACOBI_ROWS,), ndarray=True)


def build_block_jacobi(n_rows: int) -> BlockJacobiScratch:
    return BlockJacobiScratch(sizes={JACOBI_ROWS: max(n_rows, 1)})


@qd.data_oriented
class BlockJacobiOwner:
    """Block Jacobi preconditioner of an owner of rows of the linear system, which holds its rows in `linear_block` and
    their diagonal blocks in `jacobi` (see BlockJacobiScratch)."""

    @qd.func(requires_top_level=True)
    def func_factor_begin(self):
        qd.loop_config(name="newton_block_jacobi_reset")
        for i_r in range(self.linear_block.n_rows[()]):
            self.jacobi.blocks[i_r] = qd.Matrix.zero(gs.qd_float, 3, 3)

    @qd.func
    def func_linear_visit_block(self, i_r: int, i_c: int, H: qd.types.matrix(3, 3)):
        # The reduced blocks of a row are distinct, so that its diagonal block comes once
        if i_r == i_c:
            self.jacobi.blocks[i_r] = H

    @qd.func(requires_top_level=True)
    def func_factor(self):
        """Invert every diagonal block, equilibrated to a unit diagonal."""
        qd.loop_config(name="newton_block_jacobi_invert")
        for i_r in range(self.linear_block.n_rows[()]):
            D = self.jacobi.blocks[i_r]
            d = qd.Vector([D[0, 0], D[1, 1], D[2, 2]], dt=gs.qd_float)
            scale = qd.select(d > 0.0, 1.0 / qd.sqrt(qd.select(d > 0.0, d, 1.0)), 1.0)
            scaling = scale.outer_product(scale)
            S_inv = qd.Matrix.identity(gs.qd_float, 3)
            if (D * scaling).determinant() > 0.0:
                S_inv = (D * scaling).inverse()
            self.jacobi.blocks[i_r] = S_inv * scaling

    @qd.func(requires_top_level=True)
    def func_precondition(self):
        """z = M^-1 r over the rows of the owner, M^-1 being its inverted diagonal blocks."""
        qd.loop_config(name="newton_block_jacobi_apply")
        for i_r in range(self.linear_block.n_rows[()]):
            row = self.linear_block.row_start[()] + i_r
            self.linear.pcg.z[row] = self.jacobi.blocks[i_r] @ self.linear.pcg.r[row]


# ============================================== Layer 1: stages ==============================================


@qd.func(requires_top_level=True)
def func_linear_layout(participants: qd.template()):
    """Lay the rows and triplets of every owner of `participants` (see NewtonParticipants), then the triplets of the
    contacts, out in order, from the counts they reported, and report the live rows and the triplet demand of the
    matrix."""
    for _ in range(1):
        n_rows = gs.qd_int(0)
        n_triplets = gs.qd_int(0)
        for owner in qd.static(participants.owners):
            owner.linear_block.row_start[()] = n_rows
            owner.linear_block.triplet_start[()] = n_triplets
            n_rows += owner.linear_block.n_rows[()]
            n_triplets += owner.linear_block.n_triplets[()]
        if qd.static(participants.has_contact):
            participants.contact.linear_block.row_start[()] = n_rows
            participants.contact.linear_block.triplet_start[()] = n_triplets
            n_triplets += participants.contact.linear_block.n_triplets[()]
        participants.linear.pcg.n_rows[()] = qd.min(n_rows, participants.linear.pcg.row_env.shape[0])
        func_report_triplet_demand(n_triplets, participants.linear.matrix)


@qd.func(requires_top_level=True)
def func_linear_visit(linear: NewtonLinearScratch, owners: qd.template()):
    """Hand every block of the matrix whose row and column one owner holds to the visitor of that owner, with the row
    and column within its range (see func_linear_visit_block of an owner)."""
    qd.loop_config(name="newton_linear_visit")
    for i_nz in range(linear.matrix.bcoo_nnz[()]):
        i_r, i_c, H = func_read_bcoo(i_nz, linear.matrix, block_rows=3, block_cols=3)
        for owner in qd.static(owners):
            i_r_local = i_r - owner.linear_block.row_start[()]
            i_c_local = i_c - owner.linear_block.row_start[()]
            n_rows = owner.linear_block.n_rows[()]
            if 0 <= i_r_local and i_r_local < n_rows and 0 <= i_c_local and i_c_local < n_rows:
                owner.func_linear_visit_block(i_r_local, i_c_local, H)


@qd.func
def write_block(i_t: int, i_r: int, i_c: int, H: qd.types.matrix(3, 3), linear: NewtonLinearScratch):
    """Write a 3 x 3 block of the symmetric matrix into triplet slot `i_t`, mirrored into the upper triangle."""
    func_write_triplet(i_t, i_r, i_c, H, linear.matrix, is_symmetric=True)


@qd.func
def discard_block(i_t: int, linear: NewtonLinearScratch):
    """Discard triplet slot `i_t`, that of a block on an eliminated row."""
    func_discard_triplet(i_t, linear.matrix, block_rows=3, block_cols=3)
