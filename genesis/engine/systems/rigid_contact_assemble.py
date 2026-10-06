from __future__ import annotations

import numpy as np
import quadrants as qd
from quadrants.algorithms import exclusive_scan_add, exclusive_scan_scratch_slots

from .bcoo_matrix import write_bcoo_block
from .contact_system import ContactSystem
from .dynamic_exclusive_sum import DynamicExclusiveSum, dynamic_exclusive_sum
from .finite_element import FiniteElementMethod
from .global_linear_system import GlobalLinearSystem
from .global_vertex_manager import GlobalVertexManager
from .rigid_contact_proxy import RigidContactProxySystem
from .rigid_contact_proxy_kkt import rigid_contact_proxy_skew
from .rigid_joint_forest import RigidJointForestSystem
from .sim_system import SimData, SimSystem


class RigidContactAssemble(SimSystem):
    """Organize rigid/proxy contact assembly state and dependencies."""

    @qd.data_oriented
    class Data(SimData):
        """Complete mutable scan and distribution state."""

        scan_log256_max_n: int
        is_initialized_host: bool
        extent_slot: int
        triplet_multipliers: qd.Ndarray
        triplet_offsets: qd.Ndarray
        doublet_flags: qd.Ndarray
        doublet_offsets: qd.Ndarray
        triplet_scan_scratch: qd.Ndarray
        doublet_scan_scratch: qd.Ndarray
        pair_triplet_total: qd.Ndarray
        rigid_doublet_total: qd.Ndarray
        triplet_scanner: DynamicExclusiveSum
        doublet_scanner: DynamicExclusiveSum

    def __init__(self, data: Data) -> None:
        super().__init__()
        self.data = data

    def build(self) -> None:
        self.contact_system = self.require(ContactSystem)
        self.fem_system = self.require(FiniteElementMethod)
        self.vertex_system = self.require(GlobalVertexManager)
        self.linear_system_system = self.require(GlobalLinearSystem)
        self.proxy = self.require(RigidContactProxySystem)
        self.forest = self.require(RigidJointForestSystem)
        self.data.extent_slot = self.linear_system_system.register_extent_slot()

    def realloc_assembly_buffers(self, contact) -> None:
        realloc_rigid_contact_assembly_buffers(self.data, contact)


def get_rigid_contact_assemble_data(contact) -> RigidContactAssemble.Data:
    """Construct scan buffers before the assembly system is built."""
    data = RigidContactAssemble.Data()
    data.scan_log256_max_n = 4
    data.is_initialized_host = False
    data.extent_slot = -1
    if data.is_initialized_host:
        raise RuntimeError("RigidContactAssemble is already initialized")
    triplet_capacity = contact.unique_triplet_rows.shape[0]
    doublet_capacity = contact.unique_doublet_vertices.shape[0]
    data.triplet_multipliers = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.triplet_offsets = qd.ndarray(qd.i32, shape=(triplet_capacity,))
    data.triplet_scanner = DynamicExclusiveSum(
        triplet_capacity,
    )
    data.doublet_flags = qd.ndarray(qd.i32, shape=(doublet_capacity,))
    data.doublet_offsets = qd.ndarray(qd.i32, shape=(doublet_capacity,))
    data.doublet_scanner = DynamicExclusiveSum(
        doublet_capacity,
    )
    data.triplet_scan_scratch = qd.ndarray(
        qd.i32,
        shape=(
            max(
                exclusive_scan_scratch_slots(
                    triplet_capacity,
                    data.scan_log256_max_n,
                ),
                1,
            ),
        ),
    )
    data.doublet_scan_scratch = qd.ndarray(
        qd.i32,
        shape=(
            max(
                exclusive_scan_scratch_slots(
                    doublet_capacity,
                    data.scan_log256_max_n,
                ),
                1,
            ),
        ),
    )
    data.pair_triplet_total = qd.ndarray(qd.i32, shape=())
    data.rigid_doublet_total = qd.ndarray(qd.i32, shape=())
    data.pair_triplet_total.from_numpy(np.array(0, dtype=np.int32))
    data.rigid_doublet_total.from_numpy(np.array(0, dtype=np.int32))
    data.is_initialized_host = True
    return data


def realloc_rigid_contact_assembly_buffers(data: RigidContactAssemble.Data, contact) -> None:
    """Grow only assembly buffers whose capacities are insufficient."""
    if not data.is_initialized_host:
        raise RuntimeError("RigidContactAssemble must be initialized before reallocation")

    triplet_capacity = contact.unique_triplet_rows.shape[0]
    if triplet_capacity > data.triplet_multipliers.shape[0]:
        data.triplet_multipliers = qd.ndarray(
            qd.i32,
            shape=(triplet_capacity,),
        )
        data.triplet_offsets = qd.ndarray(
            qd.i32,
            shape=(triplet_capacity,),
        )
        data.triplet_scanner = DynamicExclusiveSum(
            triplet_capacity,
        )
        data.triplet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(
                max(
                    exclusive_scan_scratch_slots(
                        triplet_capacity,
                        data.scan_log256_max_n,
                    ),
                    1,
                ),
            ),
        )

    doublet_capacity = contact.unique_doublet_vertices.shape[0]
    if doublet_capacity > data.doublet_flags.shape[0]:
        data.doublet_flags = qd.ndarray(
            qd.i32,
            shape=(doublet_capacity,),
        )
        data.doublet_offsets = qd.ndarray(
            qd.i32,
            shape=(doublet_capacity,),
        )
        data.doublet_scanner = DynamicExclusiveSum(
            doublet_capacity,
        )
        data.doublet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(
                max(
                    exclusive_scan_scratch_slots(
                        doublet_capacity,
                        data.scan_log256_max_n,
                    ),
                    1,
                ),
            ),
        )


@qd.func(requires_top_level=True)
def classify(
    data: qd.template(),
    proxy: qd.template(),
    forest: qd.template(),
    vertex: qd.template(),
    contact: qd.template(),
    linear_system_data: qd.template(),
):
    proxy_vertex_begin = proxy.global_vert_offset[()]
    for index in range(contact.n_unique_triplets[()]):
        row = contact.unique_triplet_rows[index]
        col = contact.unique_triplet_cols[index]
        multiplier = qd.i32(1)
        if row < proxy_vertex_begin and col >= proxy_vertex_begin:
            multiplier = 2
        elif row >= proxy_vertex_begin:
            multiplier = 4
        data.triplet_multipliers[index] = multiplier
    if qd.static(contact.genesis_legacy_sort_reduce_host):
        exclusive_scan_add(
            data.triplet_multipliers,
            data.triplet_offsets,
            data.triplet_scan_scratch,
            contact.n_unique_triplets[()],
            qd.i32,
            data.scan_log256_max_n,
        )
    else:
        dynamic_exclusive_sum(
            data.triplet_scanner,
            data.triplet_multipliers,
            data.triplet_offsets,
            contact.n_unique_triplets[()],
        )

    for index in range(contact.n_unique_doublets[()]):
        data.doublet_flags[index] = qd.i32(contact.unique_doublet_vertices[index] >= proxy_vertex_begin)
    if qd.static(contact.genesis_legacy_sort_reduce_host):
        exclusive_scan_add(
            data.doublet_flags,
            data.doublet_offsets,
            data.doublet_scan_scratch,
            contact.n_unique_doublets[()],
            qd.i32,
            data.scan_log256_max_n,
        )
    else:
        dynamic_exclusive_sum(
            data.doublet_scanner,
            data.doublet_flags,
            data.doublet_offsets,
            contact.n_unique_doublets[()],
        )

    for _ in range(1):
        n_triplets = contact.n_unique_triplets[()]
        pair_total = qd.i32(0)
        if n_triplets > 0:
            last = n_triplets - 1
            pair_total = data.triplet_offsets[last] + data.triplet_multipliers[last]
        n_doublets = contact.n_unique_doublets[()]
        doublet_total = qd.i32(0)
        if n_doublets > 0:
            last = n_doublets - 1
            doublet_total = data.doublet_offsets[last] + data.doublet_flags[last]
        data.pair_triplet_total[()] = pair_total
        data.rigid_doublet_total[()] = doublet_total
        linear_system_data.extent_slots[data.extent_slot] = pair_total + doublet_total


@qd.func
def _proxy_vertex_data(
    data: qd.template(), proxy: qd.template(), forest: qd.template(), vertex: qd.template(), global_vertex
):
    local_vertex = global_vertex - proxy.global_vert_offset[()]
    pair = proxy.vertex_pair[local_vertex]
    lever = qd.Vector.zero(qd.f64, 3)
    for axis in qd.static(range(3)):
        lever[axis] = vertex.positions[global_vertex, axis] - proxy.t[pair, axis]
    return qd.Vector([qd.f64(pair), lever[0], lever[1], lever[2]])


@qd.func
def _set_triplet(
    data: qd.template(),
    proxy: qd.template(),
    forest: qd.template(),
    vertex: qd.template(),
    linear_system_data: qd.template(),
    slot,
    row,
    col,
    block: qd.template(),
):
    write_bcoo_block(linear_system_data.matrix, slot, row, col, block)


@qd.func(requires_top_level=True)
def distribute_gradient(
    data: qd.template(),
    proxy: qd.template(),
    forest: qd.template(),
    vertex: qd.template(),
    contact: qd.template(),
    fem_data: qd.template(),
    linear_system_data: qd.template(),
):
    proxy_vertex_begin = proxy.global_vert_offset[()]
    proxy_block_base = forest.proxy_dof_offset[()] // 3
    geometric_base = linear_system_data.extent_offsets[data.extent_slot] + data.pair_triplet_total[()]
    for index in range(contact.n_unique_doublets[()]):
        global_vertex = contact.unique_doublet_vertices[index]
        if global_vertex < proxy_vertex_begin:
            if fem_data.is_fixed[global_vertex] == 0:
                offset = fem_data.dof_offset[()] + global_vertex * 3
                for axis in qd.static(range(3)):
                    qd.atomic_add(
                        linear_system_data.b_rhs[offset + axis],
                        contact.unique_doublet_gradients[index, axis],
                    )
        else:
            proxy_data = _proxy_vertex_data(data, proxy, forest, vertex, global_vertex)
            pair = qd.i32(proxy_data[0])
            lever = qd.Vector([proxy_data[1], proxy_data[2], proxy_data[3]])
            gradient = qd.Vector(
                [
                    contact.unique_doublet_gradients[index, 0],
                    contact.unique_doublet_gradients[index, 1],
                    contact.unique_doublet_gradients[index, 2],
                ]
            )
            if vertex.is_fixed[global_vertex] == 0:
                angular_gradient = lever.cross(gradient)
                offset = forest.proxy_dof_offset[()] + pair * 6
                for axis in qd.static(range(3)):
                    qd.atomic_add(
                        linear_system_data.b_rhs[offset + axis],
                        gradient[axis],
                    )
                    qd.atomic_add(
                        linear_system_data.b_rhs[offset + axis + 3],
                        angular_gradient[axis],
                    )

                geometric = 0.5 * (gradient.outer_product(lever) + lever.outer_product(gradient)) - gradient.dot(
                    lever
                ) * qd.Matrix.identity(qd.f64, 3)
                geometric = qd.make_spd(geometric, qd.f64)
                slot = geometric_base + data.doublet_offsets[index]
                _set_triplet(
                    data,
                    proxy,
                    forest,
                    vertex,
                    linear_system_data,
                    slot,
                    proxy_block_base + pair * 2 + 1,
                    proxy_block_base + pair * 2 + 1,
                    geometric,
                )
            else:
                slot = geometric_base + data.doublet_offsets[index]
                _set_triplet(
                    data,
                    proxy,
                    forest,
                    vertex,
                    linear_system_data,
                    slot,
                    proxy_block_base + pair * 2 + 1,
                    proxy_block_base + pair * 2 + 1,
                    qd.Matrix.zero(qd.f64, 3, 3),
                )


@qd.func(requires_top_level=True)
def distribute_triplets(
    data: qd.template(),
    proxy: qd.template(),
    forest: qd.template(),
    vertex: qd.template(),
    contact: qd.template(),
    fem_data: qd.template(),
    linear_system_data: qd.template(),
):
    proxy_vertex_begin = proxy.global_vert_offset[()]
    proxy_block_base = forest.proxy_dof_offset[()] // 3
    fem_block_base = fem_data.dof_offset[()] // 3
    contact_base = linear_system_data.extent_offsets[data.extent_slot]
    for index in range(contact.n_unique_triplets[()]):
        left_vertex = contact.unique_triplet_rows[index]
        right_vertex = contact.unique_triplet_cols[index]
        multiplier = data.triplet_multipliers[index]
        output = contact_base + data.triplet_offsets[index]
        hessian = qd.Matrix.zero(qd.f64, 3, 3)
        for row in qd.static(range(3)):
            for column in qd.static(range(3)):
                hessian[row, column] = contact.unique_triplet_values[
                    index,
                    row,
                    column,
                ]

        if multiplier == 1:
            left = fem_block_base + left_vertex
            right = fem_block_base + right_vertex
            if fem_data.is_fixed[left_vertex] == 0 and fem_data.is_fixed[right_vertex] == 0:
                _set_triplet(data, proxy, forest, vertex, linear_system_data, output, left, right, hessian)
            else:
                _set_triplet(
                    data,
                    proxy,
                    forest,
                    vertex,
                    linear_system_data,
                    output,
                    left,
                    right,
                    qd.Matrix.zero(qd.f64, 3, 3),
                )
        elif multiplier == 2:
            proxy_data = _proxy_vertex_data(data, proxy, forest, vertex, right_vertex)
            pair = qd.i32(proxy_data[0])
            lever = qd.Vector([proxy_data[1], proxy_data[2], proxy_data[3]])
            left = fem_block_base + left_vertex
            proxy_translation = proxy_block_base + pair * 2
            proxy_rotation = proxy_translation + 1
            if fem_data.is_fixed[left_vertex] == 0 and vertex.is_fixed[right_vertex] == 0:
                skew = rigid_contact_proxy_skew(lever)
                _set_triplet(data, proxy, forest, vertex, linear_system_data, output, left, proxy_translation, hessian)
                _set_triplet(
                    data,
                    proxy,
                    forest,
                    vertex,
                    linear_system_data,
                    output + 1,
                    left,
                    proxy_rotation,
                    -(hessian @ skew),
                )
            else:
                zero = qd.Matrix.zero(qd.f64, 3, 3)
                _set_triplet(data, proxy, forest, vertex, linear_system_data, output, left, proxy_translation, zero)
                _set_triplet(data, proxy, forest, vertex, linear_system_data, output + 1, left, proxy_rotation, zero)
        else:
            left_data = _proxy_vertex_data(data, proxy, forest, vertex, left_vertex)
            right_data = _proxy_vertex_data(data, proxy, forest, vertex, right_vertex)
            left_pair = qd.i32(left_data[0])
            right_pair = qd.i32(right_data[0])
            left_lever = qd.Vector([left_data[1], left_data[2], left_data[3]])
            right_lever = qd.Vector([right_data[1], right_data[2], right_data[3]])
            left_skew = rigid_contact_proxy_skew(left_lever)
            right_skew = rigid_contact_proxy_skew(right_lever)
            block = qd.Matrix.zero(qd.f64, 6, 6)
            block[:3, :3] = hessian
            block[:3, 3:] = -(hessian @ right_skew)
            block[3:, :3] = left_skew @ hessian
            block[3:, 3:] = -(left_skew @ hessian @ right_skew)
            if left_pair == right_pair and left_vertex != right_vertex:
                transpose_hessian = hessian.transpose()
                block[:3, :3] += transpose_hessian
                block[:3, 3:] += -(transpose_hessian @ left_skew)
                block[3:, :3] += right_skew @ transpose_hessian
                block[3:, 3:] += -(right_skew @ transpose_hessian @ left_skew)
            left_base = proxy_block_base + left_pair * 2
            right_base = proxy_block_base + right_pair * 2
            slot_offset = qd.i32(0)
            for block_row in qd.static(range(2)):
                for block_column in qd.static(range(2)):
                    value = qd.Matrix.zero(qd.f64, 3, 3)
                    row_id = left_base + block_row
                    column_id = right_base + block_column
                    if not (left_pair == right_pair and block_row > block_column):
                        for row in qd.static(range(3)):
                            for column in qd.static(range(3)):
                                value[row, column] = block[
                                    block_row * 3 + row,
                                    block_column * 3 + column,
                                ]
                    else:
                        column_id = row_id
                    if vertex.is_fixed[left_vertex] != 0 or vertex.is_fixed[right_vertex] != 0:
                        value = qd.Matrix.zero(qd.f64, 3, 3)
                    _set_triplet(
                        data,
                        proxy,
                        forest,
                        vertex,
                        linear_system_data,
                        output + slot_offset,
                        row_id,
                        column_id,
                        value,
                    )
                    slot_offset = slot_offset + 1


@qd.func(requires_top_level=True)
def distribute(
    data: qd.template(),
    proxy: qd.template(),
    forest: qd.template(),
    vertex: qd.template(),
    contact: qd.template(),
    fem_data: qd.template(),
    linear_system_data: qd.template(),
):
    distribute_gradient(data, proxy, forest, vertex, contact, fem_data, linear_system_data)
    distribute_triplets(data, proxy, forest, vertex, contact, fem_data, linear_system_data)
