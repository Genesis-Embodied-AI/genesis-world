"""Linear system of the joint Newton solve over the rows of every participant that assembles one.

The Newton solver owns one block coordinate (BCOO) matrix of 3 x 3 blocks, its right-hand side `b_rhs`, its solution
and the vectors of its preconditioned conjugate gradient (PCG). Every participant that assembles its operator (MPM, FEM,
PBD) owns a range of the block rows and a range of the triplets, both laid out at every assembly from the counts the
participant reports in its `LinearBlock` (see func_linear_layout): it writes the triplets of its blocks, its gradient
into `b_rhs` and the environment of its rows, all from its start row and start triplet. The contacts between them write
their blocks into a range of their own (see newton_contact.py). The PCG then runs once over all the rows, and the
preconditioner is the block diagonal of the participants: one traverse of the matrix hands every block whose row and
column a participant owns to its visitor, between the start and the end of its factorization.

A row whose unknown an owner eliminates, such as a fixed MPM node, a constrained FEM vertex or an attached PBD particle,
takes an identity diagonal block and a zero right-hand side, and every other block on it is discarded (see
discard_block), as CGQ eliminates a fixed degree of freedom: the row decouples and its solution is exactly zero.
"""

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
    b_rhs: qd.Tensor = field(float, shape=(3 * PCG_ROWS,), ndarray=True)
    solution: qd.Tensor = field(float, shape=(3 * PCG_ROWS,), ndarray=True)


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


@qd.func(requires_top_level=True)
def func_block_jacobi_reset(n_rows: int, jacobi: BlockJacobiScratch):
    qd.loop_config(name="newton_block_jacobi_reset")
    for i_r in range(n_rows):
        jacobi.blocks[i_r] = qd.Matrix.zero(gs.qd_float, 3, 3)


@qd.func
def block_jacobi_visit(i_r: int, i_c: int, H: qd.types.matrix(3, 3), jacobi: BlockJacobiScratch):
    """Keep a diagonal block of the matrix, the reduced blocks of a row being distinct."""
    if i_r == i_c:
        jacobi.blocks[i_r] = H


@qd.func(requires_top_level=True)
def func_block_jacobi_invert(n_rows: int, jacobi: BlockJacobiScratch):
    qd.loop_config(name="newton_block_jacobi_invert")
    for i_r in range(n_rows):
        jacobi.blocks[i_r] = jacobi.blocks[i_r].inverse()


@qd.func(requires_top_level=True)
def func_block_jacobi_apply(block: LinearBlock, linear: NewtonLinearScratch, jacobi: BlockJacobiScratch):
    """z = M^-1 r over the rows of an owner, M^-1 being its inverted diagonal blocks."""
    qd.loop_config(name="newton_block_jacobi_apply")
    for i_r in range(block.n_rows[()]):
        row = 3 * (block.row_start[()] + i_r)
        r = qd.Vector([linear.pcg.r[row + d] for d in qd.static(range(3))], dt=gs.qd_float)
        z = jacobi.blocks[i_r] @ r
        for d in qd.static(range(3)):
            linear.pcg.z[row + d] = z[d]


@qd.func
def write_block(i_t: int, i_r: int, i_c: int, H: qd.types.matrix(3, 3), linear: NewtonLinearScratch):
    """Write a 3 x 3 block of the symmetric matrix into triplet slot `i_t`, mirrored into the upper triangle."""
    func_write_triplet(i_t, i_r, i_c, H, linear.matrix, is_symmetric=True)


@qd.func
def discard_block(i_t: int, linear: NewtonLinearScratch):
    """Discard triplet slot `i_t`, that of a block on an eliminated row (see the module docstring)."""
    func_discard_triplet(i_t, linear.matrix, block_rows=3, block_cols=3)


@qd.func
def row_vector(i_r: int, values: qd.template()):
    """The 3-vector of block row `i_r` of a vector of the linear system."""
    return qd.Vector([values[3 * i_r + d] for d in qd.static(range(3))], dt=gs.qd_float)
