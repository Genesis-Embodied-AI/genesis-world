from __future__ import annotations

import quadrants as qd


@qd.func(requires_top_level=True)
def distribute_fem_gradient_kernel(
    contact: qd.template(),
    fem: qd.template(),
    global_linear_system: qd.template(),
):
    global_begin = fem.global_vert_offset[()]
    global_end = global_begin + fem.n_fem_verts[()]
    for index in range(contact.n_unique_doublets[()]):
        global_vertex = contact.unique_doublet_vertices[index]
        if global_vertex >= global_begin and global_vertex < global_end:
            local_vertex = global_vertex - global_begin
            if fem.is_fixed[local_vertex] == 0:
                for axis in qd.static(range(3)):
                    qd.atomic_add(
                        global_linear_system.b_rhs[fem.dof_offset[()] + local_vertex * 3 + axis],
                        contact.unique_doublet_gradients[index, axis],
                    )


@qd.func(requires_top_level=True)
def distribute_fem_fem_kernel(
    contact: qd.template(),
    fem: qd.template(),
    global_linear_system: qd.template(),
):
    global_begin = fem.global_vert_offset[()]
    global_end = global_begin + fem.n_fem_verts[()]
    output_begin = global_linear_system.n_elastic[()]
    for index in range(contact.n_unique_triplets[()]):
        global_row = contact.unique_triplet_rows[index]
        global_column = contact.unique_triplet_cols[index]
        if (
            global_row >= global_begin
            and global_row < global_end
            and global_column >= global_begin
            and global_column < global_end
        ):
            local_row = global_row - global_begin
            local_column = global_column - global_begin
            block = qd.Matrix.zero(qd.f64, 3, 3)
            if fem.is_fixed[local_row] == 0 and fem.is_fixed[local_column] == 0:
                for row in qd.static(range(3)):
                    for column in qd.static(range(3)):
                        block[row, column] = contact.unique_triplet_values[index, row, column]
            global_linear_system.set_sym(
                output_begin + index,
                fem.dof_offset[()] // 3 + local_row,
                fem.dof_offset[()] // 3 + local_column,
                block,
            )
