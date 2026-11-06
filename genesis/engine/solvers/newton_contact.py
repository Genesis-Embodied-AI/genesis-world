"""Contacts of the joint Newton solve as a sparse matrix over contact vertices.

A contact vertex is a point whose displacement over the substep is linear in the unknowns of one participant: a
material point method (MPM) particle, a finite element method (FEM) vertex, a position based dynamics (PBD) particle or
a contact point on a Rigid link. The device owning a set of vertices is their vertex map, and every vertex has a global
index, its map's `vertex_offset` plus its index in the map, below its `vertex_count`.

A participant that owns rows of the linear system of the solve (see newton_linear.py) maps its vertices onto them: it
provides `func_vertex_value(f, i_v)`, the displacement of a vertex at the current iterate, and `func_vertex_rows(i_v)`,
the rows it moves and their weights, at most its `n_vertex_rows`. The row of an unknown its owner eliminates has a zero
weight, so that no contact reaches it (see newton_linear.py). The contact points on Rigid links, whose dofs solve
their own rows, have a map of their own (see MpmRigidContact), which reads and writes them through the links.

A pair kind holds the contact pairs of one kind that its owner builds before the Newton solve of a substep, which stay
fixed over the solve. Every pair writes the gradient and the Hessian of its energy over the vertices it involves into
the vertex-space matrices of `NewtonContactScratch`: doublets (vertex, 3-vector) and upper triangular triplets (row
vertex, column vertex, 3 x 3 block), `n_doublets` and `n_triplets` of them per pair at fixed slots. A pair kind
provides `func_pair_count()`, `func_pair_terms(f, i_kind, contact)` and `func_pair_energy(f, energies)`, which adds
the energy of every pair into the entry of its environment in `energies`.

The keys of the slots are sorted once per substep and their values reduced at every assembly. A reduced entry between
vertices of the linear system expands into the blocks of their rows, which join the matrix (as CGQ distributes its
contacts), and its doublets into the right-hand side. An entry on a Rigid contact point stays in vertex space: the
product of the conjugate gradient, the diagonal blocks of Rigid's preconditioner and the gradient of Rigid go through
its map, one thread per entry.
"""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
from genesis.engine.numerics.bcoo_matrix import (
    BCOOMatrixScratch,
    bcoo_matrix_sizes,
    func_read_bcoo,
    func_reduce_bcoo,
    func_report_triplet_demand,
    func_write_triplet,
    init_bcoo_matrix,
    kernel_reduce_bcoo,
    sort_bcoo,
)
from genesis.engine.numerics.dynamic_exclusive_sum import (
    DynamicExclusiveSumScratch,
    dynamic_exclusive_sum_sizes,
    func_dynamic_exclusive_sum,
)
from genesis.utils.data import Data, DataItem, DataKind, Size, field, iter_data
from genesis.utils.misc import qd_to_numpy

from .newton_linear import (
    MIN_SLOTS,
    NewtonLinearScratch,
    build_linear_block,
    discard_block,
    grown_capacity,
    retire_scratch,
    write_block,
)
from .newton_solver import NewtonDevice, participant_group

# What a Rigid contact point reads in a stage: its displacement at the current iterate, or the direction of the
# conjugate gradient
SRC_ITERATE = 0
SRC_DIRECTION = 1
# What a Rigid contact point writes in a stage: the gradient, or the operator image of the direction
DST_GRADIENT = 0
DST_PRODUCT = 1
# Rows a contact vertex moves at most, the 8 nodes of the stencil of an MPM particle
MAX_VERTEX_ROWS = 8

PAIR_KINDS = Size("pair_kinds")
HESSIAN_ENTRIES = Size("hessian_entries")
# The global vertex index is a 32-bit row or column of the vertex-space matrices
MAX_VERTICES = 2**31 - 1


class NewtonContactScratch(Data, kind=DataKind.SCRATCH):
    """Vertex-space gradient (doublets) and Hessian (upper triangular triplets) of the contact pairs, the offsets of
    the slots of every pair kind in them, then their totals, and the first triplet of the linear system of every reduced
    Hessian entry, then their total."""

    gradient: BCOOMatrixScratch
    hessian: BCOOMatrixScratch
    doublet_offsets: qd.Tensor = field(int, shape=(PAIR_KINDS + 1,), ndarray=True)
    triplet_offsets: qd.Tensor = field(int, shape=(PAIR_KINDS + 1,), ndarray=True)
    entry_blocks: qd.Tensor = field(int, shape=(HESSIAN_ENTRIES,), ndarray=True)
    entry_starts: qd.Tensor = field(int, shape=(HESSIAN_ENTRIES,), ndarray=True)
    entry_scan: DynamicExclusiveSumScratch


def build_newton_contact(n_kinds: int, max_doublets: int, max_triplets: int) -> NewtonContactScratch:
    """Build the vertex-space matrices of a number of pair kinds at a capacity of doublets and triplets."""
    scratch = NewtonContactScratch(
        sizes={
            PAIR_KINDS: n_kinds,
            HESSIAN_ENTRIES: max_triplets,
            NewtonContactScratch.gradient: bcoo_matrix_sizes(max_doublets, block_shape=(3, 1)),
            NewtonContactScratch.hessian: bcoo_matrix_sizes(max_triplets, block_shape=(3, 3)),
            NewtonContactScratch.entry_scan: dynamic_exclusive_sum_sizes(max_triplets),
        }
    )
    init_bcoo_matrix(scratch.gradient, n_triplets=0)
    init_bcoo_matrix(scratch.hessian, n_triplets=0)
    scratch.doublet_offsets.fill(0)
    scratch.triplet_offsets.fill(0)
    return scratch


# =============================================== Layer 0: dispatch ===============================================


@qd.func
def append_pair(n_pairs: qd.template(), capacity: int):
    """Take the next slot of a pair list, or -1 past its capacity, the count keeping on to report the demand."""
    i_pair = qd.atomic_add(n_pairs[()], 1)
    if i_pair >= capacity:
        i_pair = -1
    return i_pair


@qd.func
def write_doublet(i_d: int, i_vg: int, g: qd.types.vector(3), contact: qd.template()):
    """Write the gradient `g` on contact vertex `i_vg` into doublet slot `i_d`."""
    block = qd.Matrix([[g[0]], [g[1]], [g[2]]], dt=gs.qd_float)
    func_write_triplet(i_d, i_vg, 0, block, contact.matrices.gradient, is_symmetric=False)


@qd.func
def write_triplet(i_t: int, i_r: int, i_c: int, H: qd.types.matrix(3, 3), contact: qd.template()):
    """Write the Hessian block `H` of contact vertices (`i_r`, `i_c`) into triplet slot `i_t`, mirrored into the upper
    triangle."""
    func_write_triplet(i_t, i_r, i_c, H, contact.matrices.hessian, is_symmetric=True)


@qd.func
def vertex_rows(i_vg: int, contact: qd.template()):
    """Rows of the linear system that contact vertex `i_vg` moves, their weights and their count, zero for a Rigid
    contact point."""
    n_rows = gs.qd_int(0)
    rows = qd.Vector.zero(gs.qd_int, MAX_VERTEX_ROWS)
    weights = qd.Vector.zero(gs.qd_float, MAX_VERTEX_ROWS)
    for vertex_map in qd.static(contact.linear_maps):
        i_v = i_vg - vertex_map.vertex_offset[()]
        if 0 <= i_v and i_v < vertex_map.vertex_count[()]:
            n_map, rows_map, weights_map = vertex_map.func_vertex_rows(i_v)
            n_rows = n_map
            for a in qd.static(range(vertex_map.n_vertex_rows)):
                rows[a] = rows_map[a]
                weights[a] = weights_map[a]
    return n_rows, rows, weights


@qd.func
def is_rigid_vertex(i_vg: int, contact: qd.template()):
    is_rigid = False
    if qd.static(contact.has_rigid_map):
        is_rigid = contact.rigid_map.vertex_offset[()] <= i_vg
    return is_rigid


@qd.func
def entry_block_count(i_r: int, i_c: int, contact: qd.template()):
    """Blocks of the linear system a reduced Hessian entry expands into, the upper triangle of a diagonal one, and
    none on a Rigid contact point."""
    n_blocks = gs.qd_int(0)
    if not is_rigid_vertex(i_c, contact):
        n_r, rows_r, weights_r = vertex_rows(i_r, contact)
        n_c, rows_c, weights_c = vertex_rows(i_c, contact)
        n_blocks = n_r * n_c
        if i_r == i_c:
            n_blocks = n_r * (n_r + 1) // 2
    return n_blocks


# ============================================== Layer 1: stages ==============================================


@qd.func(requires_top_level=True)
def func_contact_layout(contact: qd.template()):
    """Lay the slots of every pair kind out in the vertex-space matrices, and report the demand of both."""
    for _ in range(1):
        n_doublets = gs.qd_int(0)
        n_triplets = gs.qd_int(0)
        for i_kind in qd.static(range(len(contact.pair_kinds))):
            contact.matrices.doublet_offsets[i_kind] = n_doublets
            contact.matrices.triplet_offsets[i_kind] = n_triplets
            n_pairs = contact.pair_kinds[i_kind].func_pair_count()
            n_doublets += qd.static(contact.pair_kinds[i_kind].n_doublets) * n_pairs
            n_triplets += qd.static(contact.pair_kinds[i_kind].n_triplets) * n_pairs
        contact.matrices.doublet_offsets[qd.static(len(contact.pair_kinds))] = n_doublets
        contact.matrices.triplet_offsets[qd.static(len(contact.pair_kinds))] = n_triplets
        func_report_triplet_demand(n_doublets, contact.matrices.gradient)
        func_report_triplet_demand(n_triplets, contact.matrices.hessian)


@qd.func(requires_top_level=True)
def func_contact_terms(f: int, contact: qd.template()):
    """Write the doublets and triplets of every pair at the current iterate."""
    if qd.static(contact.has_rigid_map):
        contact.rigid_map.func_vertex_begin(f, SRC_ITERATE)
    for i_kind in qd.static(range(len(contact.pair_kinds))):
        contact.pair_kinds[i_kind].func_pair_terms(f, i_kind, contact)


@qd.func(requires_top_level=True)
def func_contact_entry_layout(contact: qd.template()):
    """Number the blocks of the linear system of every reduced Hessian entry, and report their count in the share of
    the contacts (see entry_block_count). The entries hold over the substep."""
    qd.loop_config(name="newton_contact_entry_blocks")
    for i_nz in range(contact.matrices.hessian.bcoo_nnz[()]):
        i_r, i_c, H = func_read_bcoo(i_nz, contact.matrices.hessian, block_rows=3, block_cols=3)
        contact.matrices.entry_blocks[i_nz] = entry_block_count(i_r, i_c, contact)
    func_dynamic_exclusive_sum(
        contact.matrices.entry_blocks,
        contact.matrices.entry_starts,
        contact.matrices.hessian.bcoo_nnz,
        contact.matrices.entry_scan,
    )
    for _ in range(1):
        n_entries = contact.matrices.hessian.bcoo_nnz[()]
        n_blocks = gs.qd_int(0)
        if n_entries > 0:
            n_blocks = contact.matrices.entry_starts[n_entries - 1] + contact.matrices.entry_blocks[n_entries - 1]
        contact.linear_block.n_triplets[()] = n_blocks


@qd.func(requires_top_level=True)
def func_contact_assemble(f: int, contact: qd.template(), linear: NewtonLinearScratch):
    """Add the gradient and the Hessian of the contacts at the current iterate to the linear system and to Rigid.

    Runs once every participant assembled its own share. A doublet adds w_a g to every row a of its vertex. A Hessian
    entry (r, c) between vertices of the linear system writes the blocks w_a w_b H of their rows a and b into the
    triplets of the contacts, discarding those of zero weight. A doublet on a Rigid contact point goes to its link."""
    func_contact_terms(f, contact)
    func_reduce_bcoo(contact.matrices.gradient, block_scalars=3)
    func_reduce_bcoo(contact.matrices.hessian, block_scalars=9)
    qd.loop_config(name="newton_contact_gradient")
    for i_nz in range(contact.matrices.gradient.bcoo_nnz[()]):
        i_vg, i_zero, block = func_read_bcoo(i_nz, contact.matrices.gradient, block_rows=3, block_cols=1)
        g = qd.Vector([block[0, 0], block[1, 0], block[2, 0]], dt=gs.qd_float)
        if is_rigid_vertex(i_vg, contact):
            if qd.static(contact.has_rigid_map):
                contact.rigid_map.func_vertex_add_force(f, i_vg - contact.rigid_map.vertex_offset[()], g)
        else:
            n_rows, rows, weights = vertex_rows(i_vg, contact)
            for a in range(n_rows):
                qd.atomic_add(linear.b_rhs[rows[a]], weights[a] * g)
    if qd.static(contact.has_rigid_map):
        contact.rigid_map.func_vertex_end(DST_GRADIENT)
    qd.loop_config(name="newton_contact_blocks")
    for i_nz in range(contact.matrices.hessian.bcoo_nnz[()]):
        i_r, i_c, H = func_read_bcoo(i_nz, contact.matrices.hessian, block_rows=3, block_cols=3)
        if not is_rigid_vertex(i_c, contact):
            n_r, rows_r, weights_r = vertex_rows(i_r, contact)
            n_c, rows_c, weights_c = vertex_rows(i_c, contact)
            i_t = contact.linear_block.triplet_start[()] + contact.matrices.entry_starts[i_nz]
            for a in range(n_r):
                b_start = 0
                if i_r == i_c:
                    b_start = a
                for b in range(b_start, n_c):
                    weight = weights_r[a] * weights_c[b]
                    if weight == 0.0:
                        discard_block(i_t, linear)
                    else:
                        block = weight * H
                        # Two vertices on one row: the block and its transpose land on the diagonal of the row
                        if i_r != i_c and rows_r[a] == rows_c[b]:
                            block += block.transpose()
                        write_block(i_t, rows_r[a], rows_c[b], block, linear)
                    i_t += 1


@qd.func(requires_top_level=True)
def func_contact_visit_rigid(f: int, contact: qd.template()):
    """Add the diagonal blocks of the contacts on the Rigid contact points to Rigid's preconditioner, between the start
    and the end of its factorization."""
    qd.loop_config(name="newton_contact_visit_rigid")
    for i_nz in range(contact.matrices.hessian.bcoo_nnz[()]):
        i_r, i_c, H = func_read_bcoo(i_nz, contact.matrices.hessian, block_rows=3, block_cols=3)
        if i_r == i_c and is_rigid_vertex(i_r, contact):
            contact.rigid_map.func_vertex_visit_diag(f, i_r - contact.rigid_map.vertex_offset[()], H)


@qd.func(requires_top_level=True)
def func_contact_rigid_product(f: int, contact: qd.template(), linear: NewtonLinearScratch, pcg_scalars: qd.template()):
    """Add the product of the contact entries on a Rigid contact point with the direction of the conjugate gradient,
    once the matrix and Rigid set the products of their own.

    One thread per stored block B of rows (r, c): it adds B p_c to the image of r and, off the diagonal, B^T p_r to the
    image of c, the rows of the linear system adding their p . Ap. The environments that converged are skipped."""
    contact.rigid_map.func_vertex_begin(f, SRC_DIRECTION)
    qd.loop_config(name="newton_contact_rigid_product")
    for i_nz in range(contact.matrices.hessian.bcoo_nnz[()]):
        i_r, i_c, B = func_read_bcoo(i_nz, contact.matrices.hessian, block_rows=3, block_cols=3)
        # The Rigid contact points come last, so an entry on one has it as its column
        i_b = gs.qd_int(0)
        is_running = False
        if is_rigid_vertex(i_c, contact):
            i_b = contact.rigid_map.func_vertex_env(i_c - contact.rigid_map.vertex_offset[()])
            is_running = pcg_scalars.is_converged[i_b] == 0
        if is_running:
            p_ends = qd.Matrix.zero(gs.qd_float, 2, 3)
            # The two ends go through one inlined dispatch
            for i_end in range(2):
                i_vg = i_r
                if i_end == 1:
                    i_vg = i_c
                p = qd.Vector.zero(gs.qd_float, 3)
                if is_rigid_vertex(i_vg, contact):
                    p = contact.rigid_map.func_vertex_value(
                        f, i_vg - contact.rigid_map.vertex_offset[()], SRC_DIRECTION
                    )
                else:
                    n_rows, rows, weights = vertex_rows(i_vg, contact)
                    for a in range(n_rows):
                        p += weights[a] * linear.pcg.p[rows[a]]
                for d in qd.static(range(3)):
                    p_ends[i_end, d] = p[d]
            p_r = qd.Vector([p_ends[0, d] for d in qd.static(range(3))], dt=gs.qd_float)
            p_c = qd.Vector([p_ends[1, d] for d in qd.static(range(3))], dt=gs.qd_float)
            n_ends = 2
            if i_r == i_c:
                n_ends = 1
            for i_end in range(n_ends):
                i_vg = i_r
                y = B @ p_c
                p = p_r
                if i_end == 1:
                    i_vg = i_c
                    y = B.transpose() @ p_r
                    p = p_c
                if is_rigid_vertex(i_vg, contact):
                    contact.rigid_map.func_vertex_add_force(f, i_vg - contact.rigid_map.vertex_offset[()], y)
                else:
                    n_rows, rows, weights = vertex_rows(i_vg, contact)
                    for a in range(n_rows):
                        qd.atomic_add(linear.pcg.ap[rows[a]], weights[a] * y)
                    # Rigid adds the p . Ap of its own rows (see func_matvec_dot)
                    qd.atomic_add(pcg_scalars.p_dot_ap[i_b], p.dot(y))
    contact.rigid_map.func_vertex_end(DST_PRODUCT)


@qd.func(requires_top_level=True)
def func_contact_energy(f: int, contact: qd.template(), energies: qd.template()):
    """Add the energy of every pair at the current iterate into the entry of its environment in `energies`."""
    if qd.static(contact.has_rigid_map):
        contact.rigid_map.func_vertex_begin(f, SRC_ITERATE)
    for pair_kind in qd.static(contact.pair_kinds):
        pair_kind.func_pair_energy(f, energies)


@qd.kernel
def kernel_contact_layout(contact: qd.template()):
    func_contact_layout(contact)


@qd.kernel
def kernel_contact_terms(f: int, contact: qd.template()):
    func_contact_terms(f, contact)


@qd.kernel
def kernel_contact_entry_layout(contact: qd.template()):
    func_contact_entry_layout(contact)


# ============================================== Layer 2: device ==============================================


@qd.data_oriented
class NewtonContact(NewtonDevice):
    """The contacts of the joint Newton solve, over the vertices of `linear_maps` and `rigid_map` and the pairs of
    `pair_kinds`.

    Every pair kind is built by its owner before the solve. `prepare` lays the slots out once they all are, grows the
    vertex-space matrices to them, sorts their keys, which hold over the substep, and numbers the blocks their reduced
    entries add to the linear system, the share of the contacts `linear_block`. The stages of the solve are
    `func_contact_assemble`, `func_contact_visit_rigid`, `func_contact_rigid_product` and `func_contact_energy`."""

    def __init__(self, linear_maps: list, rigid_map, pair_kinds: list):
        self.linear_maps = participant_group("NewtonContactLinearMaps", linear_maps)
        self.rigid_map = rigid_map
        self.has_rigid_map = rigid_map is not None
        self.pair_kinds = participant_group("NewtonContactPairKinds", pair_kinds)
        self.matrices = build_newton_contact(len(pair_kinds), MIN_SLOTS, MIN_SLOTS)
        self.linear_block = build_linear_block(n_rows=0, n_triplets=0)
        self._scratches_replaced = []
        self.lay_out_vertices()

    @property
    def data(self) -> Iterator[DataItem]:
        yield from iter_data(self, "matrices")
        yield from iter_data(self, "linear_block")

    def lay_out_vertices(self) -> None:
        """Set the offset of every vertex map, in order, from the vertex counts of those before it.

        The map of the Rigid contact points comes last, its count growing with the pairs, so that its growth moves no
        other offset."""
        offset = 0
        for vertex_map in [*self.linear_maps, *([] if self.rigid_map is None else [self.rigid_map])]:
            vertex_map.vertex_offset.fill(offset)
            offset += int(qd_to_numpy(vertex_map.vertex_count))
        if offset > MAX_VERTICES:
            gs.raise_exception(
                f"The contacts of the implicit solve span {offset} vertices, past the {MAX_VERTICES} that a 32-bit "
                "index addresses: reduce the number of environments or of particles."
            )

    def prepare(self, f: int) -> None:
        """Lay out, size and sort the vertex-space matrices for the pairs of the substep, once every pair kind is
        built, and number the blocks of their reduced entries in the linear system."""
        kernel_contact_layout(self)
        matrices = (self.matrices.gradient, self.matrices.hessian)
        if any(int(qd_to_numpy(matrix.has_triplet_overflow)) for matrix in matrices):
            capacities = [
                max(grown_capacity(int(qd_to_numpy(matrix.triplet_demand))), matrix.triplet_row.shape[0])
                for matrix in matrices
            ]
            retire_scratch(self.matrices, self._scratches_replaced)
            self.matrices = build_newton_contact(len(self.pair_kinds), *capacities)
            kernel_contact_layout(self)
        kernel_contact_terms(f, self)
        sort_bcoo(self.matrices.gradient)
        sort_bcoo(self.matrices.hessian)
        # The reduced entries of the substep, whose blocks in the linear system are numbered once
        kernel_reduce_bcoo(self.matrices.hessian, block_scalars=9)
        kernel_contact_entry_layout(self)

    @qd.func(requires_top_level=True)
    def func_contact_assemble(self, f: int, linear: NewtonLinearScratch):
        func_contact_assemble(f, self, linear)

    @qd.func(requires_top_level=True)
    def func_contact_visit_rigid(self, f: int):
        func_contact_visit_rigid(f, self)

    @qd.func(requires_top_level=True)
    def func_contact_rigid_product(self, f: int, linear: NewtonLinearScratch, pcg_scalars: qd.template()):
        func_contact_rigid_product(f, self, linear, pcg_scalars)

    @qd.func(requires_top_level=True)
    def func_contact_energy(self, f: int, energies: qd.template()):
        func_contact_energy(f, self, energies)
