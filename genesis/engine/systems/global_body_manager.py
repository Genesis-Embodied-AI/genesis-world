from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimSystem


@qd.data_oriented
class GlobalBodyManager(SimSystem):
    """Own CGQ body ranges and contact-ignorance data."""

    def __init__(self) -> None:
        super().__init__()
        self.is_initialized_host = False

    def do_build(self) -> None:
        from .finite_element import FiniteElementMethod
        from .global_vertex_manager import GlobalVertexManager

        self.fem = self.require(FiniteElementMethod)
        self.vertex = self.require(GlobalVertexManager)

    def init(self, n_bodies: int) -> None:
        if self.is_initialized_host:
            raise RuntimeError("GlobalBodyManager is already initialized")
        if n_bodies < 0:
            raise ValueError("GlobalBodyManager body count must be non-negative")

        capacity = max(n_bodies, 1)
        self.n_bodies = qd.ndarray(qd.i32, shape=())
        self.halfplane_body_id_offset = qd.ndarray(qd.i32, shape=())
        self.n_body_contact_ignorance = qd.ndarray(qd.i32, shape=())
        self.self_collision = qd.ndarray(qd.i32, shape=(capacity,))
        self.vertex_offsets = qd.ndarray(qd.i32, shape=(capacity + 1,))
        self.body_contact_ignorance_ranges = qd.ndarray(qd.i32, shape=(capacity + 1,))
        self.body_contact_ignorance_body_ids = qd.ndarray(qd.i32, shape=(1,))

        self.n_bodies.from_numpy(np.array(n_bodies, dtype=np.int32))
        self.halfplane_body_id_offset.from_numpy(np.array(n_bodies, dtype=np.int32))
        self.n_body_contact_ignorance.from_numpy(np.array(0, dtype=np.int32))
        self.self_collision.from_numpy(np.ones(capacity, dtype=np.int32))
        self.vertex_offsets.from_numpy(np.zeros(capacity + 1, dtype=np.int32))
        self.body_contact_ignorance_ranges.from_numpy(np.zeros(capacity + 1, dtype=np.int32))
        self.body_contact_ignorance_body_ids.from_numpy(np.zeros(1, dtype=np.int32))
        self.is_initialized_host = True

    def wire_body_layout(
        self,
        vertex_offsets: np.ndarray,
        self_collision: np.ndarray,
    ) -> None:
        offsets = np.ascontiguousarray(vertex_offsets, dtype=np.int32).reshape(-1)
        collision = np.ascontiguousarray(self_collision, dtype=np.int32).reshape(-1)
        if len(offsets) != self.vertex_offsets.shape[0]:
            raise ValueError("GlobalBodyManager vertex offsets must have n_bodies + 1 entries")
        if len(collision) != self.self_collision.shape[0]:
            raise ValueError("GlobalBodyManager self-collision flags must match n_bodies")
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("GlobalBodyManager vertex offsets must be non-decreasing from zero")
        if np.any((collision != 0) & (collision != 1)):
            raise ValueError("GlobalBodyManager self-collision flags must be zero or one")
        self.vertex_offsets.from_numpy(offsets)
        self.self_collision.from_numpy(collision)

    def wire_body_contact_ignorance(
        self,
        ranges: np.ndarray,
        body_ids: np.ndarray,
    ) -> None:
        range_values = np.ascontiguousarray(ranges, dtype=np.int32).reshape(-1)
        id_values = np.ascontiguousarray(body_ids, dtype=np.int32).reshape(-1)
        if len(range_values) != self.vertex_offsets.shape[0]:
            raise ValueError("GlobalBodyManager ignorance ranges must have n_bodies + 1 entries")
        if range_values[0] != 0 or range_values[-1] != len(id_values):
            raise ValueError("GlobalBodyManager ignorance ranges must span the body-id array")
        if np.any(range_values[1:] < range_values[:-1]):
            raise ValueError("GlobalBodyManager ignorance ranges must be non-decreasing")

        self.body_contact_ignorance_ranges.from_numpy(range_values)
        self.body_contact_ignorance_body_ids = qd.ndarray(qd.i32, shape=(max(len(id_values), 1),))
        if len(id_values) == 0:
            self.body_contact_ignorance_body_ids.from_numpy(np.zeros(1, dtype=np.int32))
        else:
            self.body_contact_ignorance_body_ids.from_numpy(id_values)
        self.n_body_contact_ignorance.from_numpy(np.array(len(id_values), dtype=np.int32))

    @qd.func(requires_top_level=True)
    def compute_vertex_offsets(self, fem: qd.template()):
        for i_body in range(fem.n_bodies[()] + 1):
            self.vertex_offsets[i_body] = fem.body_vertex_offsets[i_body]
        for i_body in range(fem.n_bodies[()]):
            self.self_collision[i_body] = fem.self_collision[i_body]

    @qd.func
    def is_body_contact_ignored(self, source, target):
        ignored = False
        begin = self.body_contact_ignorance_ranges[source]
        end = self.body_contact_ignorance_ranges[source + 1]
        for index in range(begin, end):
            if self.body_contact_ignorance_body_ids[index] == target:
                ignored = True
        return ignored
