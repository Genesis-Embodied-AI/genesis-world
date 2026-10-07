from __future__ import annotations

import quadrants as qd

from .distance_flag import _popcount4, flag_active_offsets


@qd.func
def cipc_pair_area_weight(wa, wb, d_hat):
    return qd.min(wa, wb) * d_hat / 4.0


@qd.func
def cipc_scatter_doublets(
    contact: qd.template(),  # ContactSystem.Data
    gradient: qd.template(),  # qd.Vector
    flag,
    global_ids: qd.template(),  # qd.Vector
):
    count = _popcount4(flag)
    offsets = qd.Vector.zero(qd.i32, 4)
    flag_active_offsets(flag, offsets)
    begin = qd.atomic_add(contact.n_contact_doublets[()], count)
    for local_index in qd.static(range(4)):
        if local_index < count:
            offset = offsets[local_index]
            output = begin + local_index
            contact.contact_doublet_vertices[output] = global_ids[offset]
            for axis in qd.static(range(3)):
                contact.contact_doublet_gradients[output, axis] = gradient[offset * 3 + axis]


@qd.func
def cipc_fill_triplet_entry(
    contact: qd.template(),  # ContactSystem.Data
    output,
    hessian: qd.template(),  # qd.Vector
    offsets: qd.template(),  # qd.Vector
    global_ids: qd.template(),  # qd.Vector
    left,
    right,
):
    left_offset = offsets[left]
    right_offset = offsets[right]
    row = global_ids[left_offset]
    column = global_ids[right_offset]
    transpose = False
    if row > column:
        swap = row
        row = column
        column = swap
        transpose = True
    contact.contact_triplet_rows[output] = row
    contact.contact_triplet_cols[output] = column
    for block_row in qd.static(range(3)):
        for block_column in qd.static(range(3)):
            value = hessian[(left_offset * 3 + block_row) * 12 + right_offset * 3 + block_column]
            if transpose:
                value = hessian[(right_offset * 3 + block_row) * 12 + left_offset * 3 + block_column]
            contact.contact_triplet_values[output, block_row, block_column] = value


@qd.func
def cipc_scatter_triplets_upper(
    contact: qd.template(),  # ContactSystem.Data
    hessian: qd.template(),  # qd.Vector
    flag,
    global_ids: qd.template(),  # qd.Vector
):
    count = _popcount4(flag)
    triplet_count = count * (count + 1) // 2
    begin = qd.atomic_add(contact.n_contact_triplets[()], triplet_count)
    offsets = qd.Vector.zero(qd.i32, 4)
    flag_active_offsets(flag, offsets)
    output_offset = qd.i32(0)
    for left in qd.static(range(4)):
        for right in qd.static(range(left, 4)):
            if left < count and right < count:
                cipc_fill_triplet_entry(
                    contact,
                    begin + output_offset,
                    hessian,
                    offsets,
                    global_ids,
                    left,
                    right,
                )
                output_offset = output_offset + 1


@qd.func
def cipc_fill_triplet_entry_rank1(
    contact: qd.template(),  # ContactSystem.Data
    output,
    values: qd.template(),  # qd.Vector
    coefficient,
    global_ids: qd.template(),  # qd.Vector
    left,
    right,
):
    row = global_ids[left]
    column = global_ids[right]
    row_offset = left
    column_offset = right
    if row > column:
        swap = row
        row = column
        column = swap
        row_offset = right
        column_offset = left
    contact.contact_triplet_rows[output] = row
    contact.contact_triplet_cols[output] = column
    for block_row in qd.static(range(3)):
        for block_column in qd.static(range(3)):
            contact.contact_triplet_values[output, block_row, block_column] = (
                coefficient * values[row_offset * 3 + block_row] * values[column_offset * 3 + block_column]
            )


@qd.func
def cipc_scatter_triplets_upper_rank1(
    contact: qd.template(),  # ContactSystem.Data
    values: qd.template(),  # qd.Vector
    coefficient,
    global_ids: qd.template(),  # qd.Vector
):
    begin = qd.atomic_add(contact.n_contact_triplets[()], 10)
    output_offset = qd.i32(0)
    for left in qd.static(range(4)):
        for right in qd.static(range(left, 4)):
            cipc_fill_triplet_entry_rank1(
                contact,
                begin + output_offset,
                values,
                coefficient,
                global_ids,
                left,
                right,
            )
            output_offset = output_offset + 1
