from __future__ import annotations

import numpy as np
import quadrants as qd
from quadrants.algorithms import exclusive_scan_add, exclusive_scan_scratch_slots

from .contact_system import ContactSystem
from .dynamic_exclusive_sum import (
    DynamicExclusiveSum,
    dynamic_exclusive_sum,
)
from .finite_element import FiniteElementMethod
from .global_linear_system import GlobalLinearSystem
from .global_vertex_manager import GlobalVertexManager
from .rigid_contact_proxy import RigidContactProxySystem
from .rigid_contact_proxy_kkt import rigid_contact_proxy_skew
from .rigid_joint_forest import RigidJointForestSystem
from .sim_system import SimSystem


@qd.data_oriented
class RigidContactAssemble(SimSystem):
    """Rigid/proxy contact gradient and BCOO distribution routes."""

    def __init__(self) -> None:
        super().__init__()
        self.scan_log256_max_n = 4
        self.is_initialized_host = False

    def build(self) -> None:
        self.contact = self.require(ContactSystem)
        self.fem = self.require(FiniteElementMethod)
        self.vertex = self.require(GlobalVertexManager)
        self.linear_system = self.require(GlobalLinearSystem)
        self.proxy = self.require(RigidContactProxySystem)
        self.forest = self.require(RigidJointForestSystem)
        self.extent_slot = self.linear_system.register_extent_slot()

    def init(self) -> None:
        if self.is_initialized_host:
            raise RuntimeError("RigidContactAssemble is already initialized")
        triplet_capacity = self.contact.unique_triplet_rows.shape[0]
        doublet_capacity = self.contact.unique_doublet_vertices.shape[0]
        self.triplet_multipliers = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.triplet_offsets = qd.ndarray(qd.i32, shape=(triplet_capacity,))
        self.triplet_scanner = DynamicExclusiveSum(
            triplet_capacity,
        )
        self.doublet_flags = qd.ndarray(qd.i32, shape=(doublet_capacity,))
        self.doublet_offsets = qd.ndarray(qd.i32, shape=(doublet_capacity,))
        self.doublet_scanner = DynamicExclusiveSum(
            doublet_capacity,
        )
        self.triplet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(
                max(
                    exclusive_scan_scratch_slots(
                        triplet_capacity,
                        self.scan_log256_max_n,
                    ),
                    1,
                ),
            ),
        )
        self.doublet_scan_scratch = qd.ndarray(
            qd.i32,
            shape=(
                max(
                    exclusive_scan_scratch_slots(
                        doublet_capacity,
                        self.scan_log256_max_n,
                    ),
                    1,
                ),
            ),
        )
        self.pair_triplet_total = qd.ndarray(qd.i32, shape=())
        self.rigid_doublet_total = qd.ndarray(qd.i32, shape=())
        self.pair_triplet_total.from_numpy(np.array(0, dtype=np.int32))
        self.rigid_doublet_total.from_numpy(np.array(0, dtype=np.int32))
        self.is_initialized_host = True

    def realloc_assembly_buffers(self) -> None:
        if not self.is_initialized_host:
            raise RuntimeError("RigidContactAssemble must be initialized before reallocation")

        triplet_capacity = self.contact.unique_triplet_rows.shape[0]
        if triplet_capacity > self.triplet_multipliers.shape[0]:
            self.triplet_multipliers = qd.ndarray(
                qd.i32,
                shape=(triplet_capacity,),
            )
            self.triplet_offsets = qd.ndarray(
                qd.i32,
                shape=(triplet_capacity,),
            )
            self.triplet_scanner = DynamicExclusiveSum(
                triplet_capacity,
            )
            self.triplet_scan_scratch = qd.ndarray(
                qd.i32,
                shape=(
                    max(
                        exclusive_scan_scratch_slots(
                            triplet_capacity,
                            self.scan_log256_max_n,
                        ),
                        1,
                    ),
                ),
            )

        doublet_capacity = self.contact.unique_doublet_vertices.shape[0]
        if doublet_capacity > self.doublet_flags.shape[0]:
            self.doublet_flags = qd.ndarray(
                qd.i32,
                shape=(doublet_capacity,),
            )
            self.doublet_offsets = qd.ndarray(
                qd.i32,
                shape=(doublet_capacity,),
            )
            self.doublet_scanner = DynamicExclusiveSum(
                doublet_capacity,
            )
            self.doublet_scan_scratch = qd.ndarray(
                qd.i32,
                shape=(
                    max(
                        exclusive_scan_scratch_slots(
                            doublet_capacity,
                            self.scan_log256_max_n,
                        ),
                        1,
                    ),
                ),
            )

    @qd.func(requires_top_level=True)
    def classify(self):
        proxy_vertex_begin = self.proxy.global_vert_offset[()]
        for index in range(self.contact.n_unique_triplets[()]):
            row = self.contact.unique_triplet_rows[index]
            col = self.contact.unique_triplet_cols[index]
            multiplier = qd.i32(1)
            if row < proxy_vertex_begin and col >= proxy_vertex_begin:
                multiplier = 2
            elif row >= proxy_vertex_begin:
                multiplier = 4
            self.triplet_multipliers[index] = multiplier
        if qd.static(self.contact.genesis_legacy_sort_reduce_host):
            exclusive_scan_add(
                self.triplet_multipliers,
                self.triplet_offsets,
                self.triplet_scan_scratch,
                self.contact.n_unique_triplets[()],
                qd.i32,
                self.scan_log256_max_n,
            )
        else:
            dynamic_exclusive_sum(
                self.triplet_scanner,
                self.triplet_multipliers,
                self.triplet_offsets,
                self.contact.n_unique_triplets[()],
            )

        for index in range(self.contact.n_unique_doublets[()]):
            self.doublet_flags[index] = qd.i32(self.contact.unique_doublet_vertices[index] >= proxy_vertex_begin)
        if qd.static(self.contact.genesis_legacy_sort_reduce_host):
            exclusive_scan_add(
                self.doublet_flags,
                self.doublet_offsets,
                self.doublet_scan_scratch,
                self.contact.n_unique_doublets[()],
                qd.i32,
                self.scan_log256_max_n,
            )
        else:
            dynamic_exclusive_sum(
                self.doublet_scanner,
                self.doublet_flags,
                self.doublet_offsets,
                self.contact.n_unique_doublets[()],
            )

        for _ in range(1):
            n_triplets = self.contact.n_unique_triplets[()]
            pair_total = qd.i32(0)
            if n_triplets > 0:
                last = n_triplets - 1
                pair_total = self.triplet_offsets[last] + self.triplet_multipliers[last]
            n_doublets = self.contact.n_unique_doublets[()]
            doublet_total = qd.i32(0)
            if n_doublets > 0:
                last = n_doublets - 1
                doublet_total = self.doublet_offsets[last] + self.doublet_flags[last]
            self.pair_triplet_total[()] = pair_total
            self.rigid_doublet_total[()] = doublet_total
            self.linear_system.extent_slots[self.extent_slot] = pair_total + doublet_total

    @qd.func
    def _proxy_vertex_data(self, global_vertex):
        local_vertex = global_vertex - self.proxy.global_vert_offset[()]
        pair = self.proxy.vertex_pair[local_vertex]
        lever = qd.Vector.zero(qd.f64, 3)
        for axis in qd.static(range(3)):
            lever[axis] = self.vertex.positions[global_vertex, axis] - self.proxy.t[pair, axis]
        return qd.Vector([qd.f64(pair), lever[0], lever[1], lever[2]])

    @qd.func
    def _set_triplet(self, slot, row, col, block: qd.template()):
        self.linear_system.set_sym(slot, row, col, block)

    @qd.func(requires_top_level=True)
    def distribute_gradient(self):
        proxy_vertex_begin = self.proxy.global_vert_offset[()]
        proxy_block_base = self.forest.proxy_dof_offset[()] // 3
        geometric_base = self.linear_system.extent_offsets[self.extent_slot] + self.pair_triplet_total[()]
        for index in range(self.contact.n_unique_doublets[()]):
            global_vertex = self.contact.unique_doublet_vertices[index]
            if global_vertex < proxy_vertex_begin:
                if self.fem.is_fixed[global_vertex] == 0:
                    offset = self.fem.dof_offset[()] + global_vertex * 3
                    for axis in qd.static(range(3)):
                        qd.atomic_add(
                            self.linear_system.b_rhs[offset + axis],
                            self.contact.unique_doublet_gradients[index, axis],
                        )
            else:
                data = self._proxy_vertex_data(global_vertex)
                pair = qd.i32(data[0])
                lever = qd.Vector([data[1], data[2], data[3]])
                gradient = qd.Vector(
                    [
                        self.contact.unique_doublet_gradients[index, 0],
                        self.contact.unique_doublet_gradients[index, 1],
                        self.contact.unique_doublet_gradients[index, 2],
                    ]
                )
                if self.vertex.is_fixed[global_vertex] == 0:
                    angular_gradient = lever.cross(gradient)
                    offset = self.forest.proxy_dof_offset[()] + pair * 6
                    for axis in qd.static(range(3)):
                        qd.atomic_add(
                            self.linear_system.b_rhs[offset + axis],
                            gradient[axis],
                        )
                        qd.atomic_add(
                            self.linear_system.b_rhs[offset + axis + 3],
                            angular_gradient[axis],
                        )

                    geometric = 0.5 * (gradient.outer_product(lever) + lever.outer_product(gradient)) - gradient.dot(
                        lever
                    ) * qd.Matrix.identity(qd.f64, 3)
                    geometric = qd.make_spd(geometric, qd.f64)
                    slot = geometric_base + self.doublet_offsets[index]
                    self._set_triplet(
                        slot,
                        proxy_block_base + pair * 2 + 1,
                        proxy_block_base + pair * 2 + 1,
                        geometric,
                    )
                else:
                    slot = geometric_base + self.doublet_offsets[index]
                    self._set_triplet(
                        slot,
                        proxy_block_base + pair * 2 + 1,
                        proxy_block_base + pair * 2 + 1,
                        qd.Matrix.zero(qd.f64, 3, 3),
                    )

    @qd.func(requires_top_level=True)
    def distribute_triplets(self):
        proxy_vertex_begin = self.proxy.global_vert_offset[()]
        proxy_block_base = self.forest.proxy_dof_offset[()] // 3
        fem_block_base = self.fem.dof_offset[()] // 3
        contact_base = self.linear_system.extent_offsets[self.extent_slot]
        for index in range(self.contact.n_unique_triplets[()]):
            left_vertex = self.contact.unique_triplet_rows[index]
            right_vertex = self.contact.unique_triplet_cols[index]
            multiplier = self.triplet_multipliers[index]
            output = contact_base + self.triplet_offsets[index]
            hessian = qd.Matrix.zero(qd.f64, 3, 3)
            for row in qd.static(range(3)):
                for column in qd.static(range(3)):
                    hessian[row, column] = self.contact.unique_triplet_values[
                        index,
                        row,
                        column,
                    ]

            if multiplier == 1:
                left = fem_block_base + left_vertex
                right = fem_block_base + right_vertex
                if self.fem.is_fixed[left_vertex] == 0 and self.fem.is_fixed[right_vertex] == 0:
                    self._set_triplet(output, left, right, hessian)
                else:
                    self._set_triplet(output, left, right, qd.Matrix.zero(qd.f64, 3, 3))
            elif multiplier == 2:
                proxy_data = self._proxy_vertex_data(right_vertex)
                pair = qd.i32(proxy_data[0])
                lever = qd.Vector([proxy_data[1], proxy_data[2], proxy_data[3]])
                left = fem_block_base + left_vertex
                proxy_translation = proxy_block_base + pair * 2
                proxy_rotation = proxy_translation + 1
                if self.fem.is_fixed[left_vertex] == 0 and self.vertex.is_fixed[right_vertex] == 0:
                    skew = rigid_contact_proxy_skew(lever)
                    self._set_triplet(output, left, proxy_translation, hessian)
                    self._set_triplet(
                        output + 1,
                        left,
                        proxy_rotation,
                        -(hessian @ skew),
                    )
                else:
                    zero = qd.Matrix.zero(qd.f64, 3, 3)
                    self._set_triplet(output, left, proxy_translation, zero)
                    self._set_triplet(output + 1, left, proxy_rotation, zero)
            else:
                left_data = self._proxy_vertex_data(left_vertex)
                right_data = self._proxy_vertex_data(right_vertex)
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
                        if self.vertex.is_fixed[left_vertex] != 0 or self.vertex.is_fixed[right_vertex] != 0:
                            value = qd.Matrix.zero(qd.f64, 3, 3)
                        self._set_triplet(
                            output + slot_offset,
                            row_id,
                            column_id,
                            value,
                        )
                        slot_offset = slot_offset + 1

    @qd.func(requires_top_level=True)
    def distribute(self):
        self.distribute_gradient()
        self.distribute_triplets()
