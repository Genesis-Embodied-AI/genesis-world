"""Contacts of the joint Newton solve as a sparse matrix over contact vertices.

A contact vertex is a point whose displacement over the substep is linear in the unknowns of one participant: a
material point method (MPM) particle, a finite element method (FEM) vertex, a position based dynamics (PBD) particle or
a contact point on a Rigid link. The device owning a set of vertices is their vertex map, and every vertex has a global
index, its map's `vertex_offset` plus its index in the map. A vertex map provides, as qd.func methods:

- `func_vertex_begin(f, src)`, run before the vertices of a stage are read, and `func_vertex_end(dst)`, after they
  are written, both top-level stages;
- `func_vertex_env(i_v)`, the environment of a vertex;
- `func_vertex_value(f, i_v, src)`, the displacement of a vertex at the current iterate (SRC_ITERATE) or its share of
  the direction of the conjugate gradient (SRC_DIRECTION);
- `func_vertex_add_gradient(f, i_v, y)`, which adds a vertex vector to the gradient of the unknowns,
  `func_vertex_add_product(f, i_v, y, p, pcg_scalars)`, which adds it to their operator image (`p` being the value of
  the direction at the vertex), and `func_vertex_visit_diag(f, i_v, H)`, which adds a diagonal block to their
  preconditioner;
- `vertex_offset` and `vertex_count`, scalar ndarrays.

A pair kind holds the contact pairs of one kind that its owner builds before the Newton solve of a substep, which stay
fixed over the solve. Every pair writes the gradient and the Hessian of its energy over the vertices it involves into
the vertex-space matrices of `NewtonContactScratch`: doublets (vertex, 3-vector) and upper triangular triplets (row
vertex, column vertex, 3 x 3 block), `n_doublets` and `n_triplets` of them per pair at fixed slots. A pair kind
provides `func_pair_count()`, `func_pair_terms(f, i_kind, contact)` and `func_pair_energy(f, control, is_trial)`.

The keys of the slots are sorted once per substep and their values reduced at every assembly, so that the stages of the
solve apply the contacts one block coordinate (BCOO) entry per thread, through the vertex maps: the gradient, the
operator product, the diagonal blocks of the preconditioners and the line-search energy.
"""

from collections.abc import Iterator

import numpy as np

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
    sort_bcoo,
)
from genesis.utils.data import Data, DataItem, DataKind, Size, field, iter_data, iter_state_arrays
from genesis.utils.misc import qd_destroy_data, qd_to_numpy

from .newton_solver import NewtonDevice, participant_group

# What the vertex maps read in a stage: the displacement at the current iterate, or the direction of the conjugate
# gradient
SRC_ITERATE = 0
SRC_DIRECTION = 1
# What the vertex maps write in a stage: the gradient, or the operator image of the direction
DST_GRADIENT = 0
DST_PRODUCT = 1

PAIR_KINDS = Size("pair_kinds")
# Capacity of the pairs of a pair kind
PAIRS = Size("pairs")
# Smallest capacity of the vertex-space matrices in slots and of a pair kind in pairs
MIN_SLOTS = 64
# Slack a capacity gets when it grows on overflow: the new capacity is the demand times this factor
CAPACITY_GROW_FACTOR = 1.2
# The global vertex index is a 32-bit row or column of the vertex-space matrices
MAX_VERTICES = 2**31 - 1


class NewtonContactScratch(Data, kind=DataKind.SCRATCH):
    """Vertex-space gradient (doublets) and Hessian (upper triangular triplets) of the contact pairs, with the offsets
    of the slots of every pair kind in them, then their totals."""

    gradient: BCOOMatrixScratch
    hessian: BCOOMatrixScratch
    doublet_offsets: qd.Tensor = field(int, shape=(PAIR_KINDS + 1,), ndarray=True)
    triplet_offsets: qd.Tensor = field(int, shape=(PAIR_KINDS + 1,), ndarray=True)


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


def build_newton_contact(n_kinds: int, max_doublets: int, max_triplets: int) -> NewtonContactScratch:
    """Build the vertex-space matrices of a number of pair kinds at a capacity of doublets and triplets."""
    scratch = NewtonContactScratch(
        sizes={
            PAIR_KINDS: n_kinds,
            NewtonContactScratch.gradient: bcoo_matrix_sizes(max_doublets, block_shape=(3, 1)),
            NewtonContactScratch.hessian: bcoo_matrix_sizes(max_triplets, block_shape=(3, 3)),
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
def vertex_env(i_vg: int, contact: qd.template()):
    """Environment of the contact vertex of global index `i_vg`."""
    i_b = gs.qd_int(0)
    for vertex_map in qd.static(contact.vertex_maps):
        i_v = i_vg - vertex_map.vertex_offset[()]
        if 0 <= i_v and i_v < vertex_map.vertex_count[()]:
            i_b = vertex_map.func_vertex_env(i_v)
    return i_b


@qd.func
def vertex_value(f: int, i_vg: int, contact: qd.template(), src: qd.template()):
    """Value of the contact vertex of global index `i_vg` (see func_vertex_value)."""
    value = qd.Vector.zero(gs.qd_float, 3)
    for vertex_map in qd.static(contact.vertex_maps):
        i_v = i_vg - vertex_map.vertex_offset[()]
        if 0 <= i_v and i_v < vertex_map.vertex_count[()]:
            value = vertex_map.func_vertex_value(f, i_v, src)
    return value


# ============================================== Layer 1: stages ==============================================


@qd.func(requires_top_level=True)
def func_contact_layout(contact: qd.template()):
    """Lay the slots of every pair kind out in the vertex-space matrices, and report the demand of both."""
    for _ in range(1):
        n_doublets = gs.qd_int(0)
        n_triplets = gs.qd_int(0)
        for i_kind in qd.static(range(len(contact.pair_kinds))):
            pair_kind = contact.pair_kinds[i_kind]
            contact.matrices.doublet_offsets[i_kind] = n_doublets
            contact.matrices.triplet_offsets[i_kind] = n_triplets
            n_pairs = pair_kind.func_pair_count()
            n_doublets += qd.static(pair_kind.n_doublets) * n_pairs
            n_triplets += qd.static(pair_kind.n_triplets) * n_pairs
        contact.matrices.doublet_offsets[qd.static(len(contact.pair_kinds))] = n_doublets
        contact.matrices.triplet_offsets[qd.static(len(contact.pair_kinds))] = n_triplets
        func_report_triplet_demand(n_doublets, contact.matrices.gradient)
        func_report_triplet_demand(n_triplets, contact.matrices.hessian)


@qd.func(requires_top_level=True)
def func_contact_terms(f: int, contact: qd.template()):
    """Write the doublets and triplets of every pair at the current iterate."""
    for vertex_map in qd.static(contact.vertex_maps):
        vertex_map.func_vertex_begin(f, SRC_ITERATE)
    for i_kind in qd.static(range(len(contact.pair_kinds))):
        contact.pair_kinds[i_kind].func_pair_terms(f, i_kind, contact)


@qd.func(requires_top_level=True)
def func_contact_assemble(f: int, contact: qd.template()):
    """Add the gradient of the contacts at the current iterate to the gradients of the participants.

    Runs once every participant assembled its own gradient. The Hessian is reduced too, which the preconditioner and
    the operator product read until the next assembly."""
    func_contact_terms(f, contact)
    func_reduce_bcoo(contact.matrices.gradient, block_scalars=3)
    func_reduce_bcoo(contact.matrices.hessian, block_scalars=9)
    qd.loop_config(name="newton_contact_gradient")
    for i_nz in range(contact.matrices.gradient.bcoo_nnz[()]):
        i_vg, _, block = func_read_bcoo(i_nz, contact.matrices.gradient, block_rows=3, block_cols=1)
        g = qd.Vector([block[0, 0], block[1, 0], block[2, 0]], dt=gs.qd_float)
        for vertex_map in qd.static(contact.vertex_maps):
            i_v = i_vg - vertex_map.vertex_offset[()]
            if 0 <= i_v and i_v < vertex_map.vertex_count[()]:
                vertex_map.func_vertex_add_gradient(f, i_v, g)
    for vertex_map in qd.static(contact.vertex_maps):
        vertex_map.func_vertex_end(DST_GRADIENT)


@qd.func(requires_top_level=True)
def func_contact_visit_diag(f: int, contact: qd.template()):
    """Add the diagonal blocks of the contact Hessian to the preconditioners of the participants, between the start
    and the end of their factorization."""
    qd.loop_config(name="newton_contact_visit_diag")
    for i_nz in range(contact.matrices.hessian.bcoo_nnz[()]):
        i_r, i_c, H = func_read_bcoo(i_nz, contact.matrices.hessian, block_rows=3, block_cols=3)
        if i_r == i_c:
            for vertex_map in qd.static(contact.vertex_maps):
                i_v = i_r - vertex_map.vertex_offset[()]
                if 0 <= i_v and i_v < vertex_map.vertex_count[()]:
                    vertex_map.func_vertex_visit_diag(f, i_v, H)


@qd.func(requires_top_level=True)
def func_contact_product(f: int, pcg_scalars: qd.template(), contact: qd.template()):
    """Add the product of the contact Hessian with the direction of the conjugate gradient to the operator images of
    the participants, once they set the product of their own operators.

    One thread per stored block B of rows (r, c): it adds B p_c to the image of r and, off the diagonal, B^T p_r to the
    image of c. The environments that converged are skipped."""
    for vertex_map in qd.static(contact.vertex_maps):
        vertex_map.func_vertex_begin(f, SRC_DIRECTION)
    qd.loop_config(name="newton_contact_product")
    for i_nz in range(contact.matrices.hessian.bcoo_nnz[()]):
        i_r, i_c, B = func_read_bcoo(i_nz, contact.matrices.hessian, block_rows=3, block_cols=3)
        i_b = vertex_env(i_r, contact)
        if pcg_scalars.is_converged[i_b] == 0:
            p_r = qd.Vector.zero(gs.qd_float, 3)
            p_c = qd.Vector.zero(gs.qd_float, 3)
            # The two ends go through one inlined dispatch
            for i_end in range(2):
                i_vg = i_r
                if i_end == 1:
                    i_vg = i_c
                p = vertex_value(f, i_vg, contact, SRC_DIRECTION)
                if i_end == 0:
                    p_r = p
                else:
                    p_c = p
            y_r = B @ p_c
            y_c = B.transpose() @ p_r
            n_ends = 2
            if i_r == i_c:
                n_ends = 1
            for i_end in range(n_ends):
                i_vg = i_r
                y = y_r
                p = p_r
                if i_end == 1:
                    i_vg = i_c
                    y = y_c
                    p = p_c
                for vertex_map in qd.static(contact.vertex_maps):
                    i_v = i_vg - vertex_map.vertex_offset[()]
                    if 0 <= i_v and i_v < vertex_map.vertex_count[()]:
                        vertex_map.func_vertex_add_product(f, i_v, y, p, pcg_scalars)
    for vertex_map in qd.static(contact.vertex_maps):
        vertex_map.func_vertex_end(DST_PRODUCT)


@qd.func(requires_top_level=True)
def func_contact_energy(f: int, control: qd.template(), contact: qd.template(), is_trial: qd.template()):
    """Add the energy of every pair at the current iterate to the start or trial energy of its environment."""
    for vertex_map in qd.static(contact.vertex_maps):
        vertex_map.func_vertex_begin(f, SRC_ITERATE)
    for pair_kind in qd.static(contact.pair_kinds):
        pair_kind.func_pair_energy(f, control, is_trial)


@qd.kernel
def kernel_contact_layout(contact: qd.template()):
    func_contact_layout(contact)


@qd.kernel
def kernel_contact_terms(f: int, contact: qd.template()):
    func_contact_terms(f, contact)


@qd.kernel
def kernel_contact_assemble(f: int, contact: qd.template()):
    func_contact_assemble(f, contact)


# ============================================== Layer 2: device ==============================================


@qd.data_oriented
class NewtonContact(NewtonDevice):
    """The contacts of the joint Newton solve, over the vertices of `vertex_maps` and the pairs of `pair_kinds`.

    Every pair kind is built by its owner before the solve. `prepare` lays the slots out once they all are, grows the
    vertex-space matrices to them and sorts their keys, which hold over the substep. The stages of the solve are
    `func_contact_assemble`, `func_contact_visit_diag`, `func_contact_product` and `func_contact_energy`."""

    def __init__(self, vertex_maps: list, pair_kinds: list):
        self.vertex_maps = participant_group("NewtonContactVertexMaps", vertex_maps)
        self.pair_kinds = participant_group("NewtonContactPairKinds", pair_kinds)
        self.matrices = build_newton_contact(len(pair_kinds), MIN_SLOTS, MIN_SLOTS)
        self._scratches_replaced = []
        self.lay_out_vertices()

    @property
    def data(self) -> Iterator[DataItem]:
        yield from iter_data(self, "matrices")

    def lay_out_vertices(self) -> None:
        """Set the offset of every vertex map, in order, from the vertex counts of those before it.

        A vertex map whose count changes at runtime comes last, so that its growth moves no other offset."""
        offset = 0
        for vertex_map in self.vertex_maps:
            vertex_map.vertex_offset.fill(offset)
            offset += int(qd_to_numpy(vertex_map.vertex_count))
        if offset > MAX_VERTICES:
            gs.raise_exception(
                f"The contacts of the implicit solve span {offset} vertices, past the {MAX_VERTICES} that a 32-bit "
                "index addresses: reduce the number of environments or of particles."
            )

    def prepare(self, f: int) -> None:
        """Lay out, size and sort the vertex-space matrices for the pairs of the substep, once every pair kind is
        built."""
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

    @qd.func(requires_top_level=True)
    def func_contact_assemble(self, f: int):
        func_contact_assemble(f, self)

    @qd.func(requires_top_level=True)
    def func_contact_visit_diag(self, f: int):
        func_contact_visit_diag(f, self)

    @qd.func(requires_top_level=True)
    def func_contact_product(self, f: int, pcg_scalars: qd.template()):
        func_contact_product(f, pcg_scalars, self)

    @qd.func(requires_top_level=True)
    def func_contact_energy(self, control: qd.template(), f: int, is_trial: qd.template()):
        func_contact_energy(f, control, self, is_trial)
