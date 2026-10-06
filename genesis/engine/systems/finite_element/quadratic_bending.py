from __future__ import annotations

import numpy as np
import quadrants as qd

from ..global_linear_system import GlobalLinearSystem
from .fem_constitution import FEMConstitution


@qd.data_oriented
class QuadraticBending(FEMConstitution):
    """Bergou quadratic shell-bending constitution."""

    def __init__(self) -> None:
        super().__init__()
        self.n_hinges_host = 0
        self.extent_slot = -1

    def do_build_constitution(self) -> None:
        self.extent_slot = self.require(GlobalLinearSystem).register_extent_slot()

    def wire_data(
        self,
        hinge_indices: np.ndarray,
        bending_stiffness: np.ndarray,
        Q0: np.ndarray,
        vert_bend_k: np.ndarray,
    ) -> None:
        self.n_hinges_host = len(hinge_indices)
        if len(bending_stiffness) != self.n_hinges_host or len(Q0) != self.n_hinges_host:
            raise ValueError("QuadraticBending wire-data lengths must match")

        capacity = max(self.n_hinges_host, 1)
        self.n_hinges = qd.ndarray(qd.i32, shape=())
        self.hinge_indices = qd.ndarray(qd.i32, shape=(capacity, 4))
        self.k = qd.ndarray(qd.f64, shape=(capacity,))
        self.Q0 = qd.ndarray(qd.f64, shape=(capacity, 16))
        self.vert_bend_k = qd.ndarray(qd.f64, shape=(max(len(vert_bend_k), 1),))
        self.n_hinges.from_numpy(np.array(self.n_hinges_host, dtype=np.int32))
        if self.n_hinges_host:
            self.hinge_indices.from_numpy(np.asarray(hinge_indices, dtype=np.int32))
            self.k.from_numpy(np.asarray(bending_stiffness, dtype=np.float64))
            self.Q0.from_numpy(np.asarray(Q0, dtype=np.float64).reshape(self.n_hinges_host, 16))
        else:
            self.hinge_indices.from_numpy(np.zeros((capacity, 4), dtype=np.int32))
            self.k.from_numpy(np.zeros(capacity, dtype=np.float64))
            self.Q0.from_numpy(np.zeros((capacity, 16), dtype=np.float64))
        self.vert_bend_k.from_numpy(np.asarray(vert_bend_k, dtype=np.float64))

    def triplet_count(self) -> int:
        return self.n_hinges_host * 10

    @qd.func(requires_top_level=True)
    def report_extent(self, fem: qd.template(), global_linear_system: qd.template()):
        for _ in range(1):
            global_linear_system.extent_slots[self.extent_slot] = self.n_hinges[()] * 10

    @qd.func(requires_top_level=True)
    def assemble(
        self,
        fem: qd.template(),
        sim_config: qd.template(),
        global_linear_system: qd.template(),
    ):
        triplet_offset = global_linear_system.extent_offsets[self.extent_slot]
        dt2 = sim_config.dt[()] * sim_config.dt[()]
        for i in range(self.n_hinges[()]):
            if global_linear_system.triplet_overflow[()] == 0:
                verts = qd.Vector(
                    [
                        self.hinge_indices[i, 0],
                        self.hinge_indices[i, 1],
                        self.hinge_indices[i, 2],
                        self.hinge_indices[i, 3],
                    ],
                    dt=qd.i32,
                )
                stiffness = self.k[i]

                for left in qd.static(range(4)):
                    if fem.is_fixed[verts[left]] == 0:
                        for axis in qd.static(range(3)):
                            gradient = qd.f64(0.0)
                            for right in qd.static(range(4)):
                                gradient = gradient + self.Q0[i, left * 4 + right] * fem.x[verts[right], axis]
                            qd.atomic_add(
                                global_linear_system.b_rhs[fem.dof_offset[()] + verts[left] * 3 + axis],
                                stiffness * dt2 * gradient,
                            )

                slot = triplet_offset + i * 10
                for left in qd.static(range(4)):
                    for right in qd.static(range(left, 4)):
                        block = qd.Matrix.zero(qd.f64, 3, 3)
                        if fem.is_fixed[verts[left]] == 0 and fem.is_fixed[verts[right]] == 0:
                            value = stiffness * dt2 * self.Q0[i, left * 4 + right]
                            for axis in qd.static(range(3)):
                                block[axis, axis] = value
                        global_linear_system.set_sym(
                            slot,
                            fem.dof_offset[()] // 3 + verts[left],
                            fem.dof_offset[()] // 3 + verts[right],
                            block,
                        )
                        slot = slot + 1

    @qd.func(requires_top_level=True)
    def energy(self, fem: qd.template(), sim_config: qd.template(), energy: qd.template()):
        dt2 = sim_config.dt[()] * sim_config.dt[()]
        for i in range(self.n_hinges[()]):
            value = qd.f64(0.0)
            for left in qd.static(range(4)):
                left_vertex = self.hinge_indices[i, left]
                for right in qd.static(range(4)):
                    right_vertex = self.hinge_indices[i, right]
                    dot = qd.f64(0.0)
                    for axis in qd.static(range(3)):
                        dot = dot + fem.x[left_vertex, axis] * fem.x[right_vertex, axis]
                    value = value + self.Q0[i, left * 4 + right] * dot
            qd.atomic_add(energy[()], 0.5 * self.k[i] * value * dt2)
