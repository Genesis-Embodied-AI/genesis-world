from __future__ import annotations

import numpy as np
import quadrants as qd

from .sim_system import SimData, SimSystem


class GlobalBodyManager(SimSystem):
    """Organize global body data and its host wiring lifecycle."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible body ranges and contact-ignorance data."""

        n_bodies: qd.Ndarray
        halfplane_body_id_offset: qd.Ndarray
        n_body_contact_ignorance: qd.Ndarray
        self_collision: qd.Ndarray
        vertex_offsets: qd.Ndarray
        body_contact_ignorance_ranges: qd.Ndarray
        body_contact_ignorance_body_ids: qd.Ndarray

    def __init__(self, data: Data | None = None) -> None:
        super().__init__()
        self.data = get_global_body_data(0) if data is None else data

    def build(self) -> None:
        from .finite_element import FiniteElementMethod
        from .global_vertex_manager import GlobalVertexManager

        self.fem_system = self.require(FiniteElementMethod)
        self.vertex_system = self.require(GlobalVertexManager)


def get_global_body_data(
    n_bodies: int,
    *,
    vertex_offsets: np.ndarray | None = None,
    self_collision: np.ndarray | None = None,
    body_contact_ignorance_ranges: np.ndarray | None = None,
    body_contact_ignorance_body_ids: np.ndarray | None = None,
    halfplane_body_id_offset: int | None = None,
) -> GlobalBodyManager.Data:
    if n_bodies < 0:
        raise ValueError("Global body count must be non-negative")
    capacity = max(n_bodies, 1)
    if vertex_offsets is None:
        offset_values = np.zeros(capacity + 1, dtype=np.int32)
    else:
        offset_values = np.ascontiguousarray(vertex_offsets, dtype=np.int32).reshape(-1)
        if len(offset_values) != n_bodies + 1:
            raise ValueError("Global body vertex offsets must have n_bodies + 1 entries")
        if offset_values[0] != 0 or np.any(offset_values[1:] < offset_values[:-1]):
            raise ValueError("Global body vertex offsets must be non-decreasing from zero")
        if n_bodies == 0:
            offset_values = np.zeros(capacity + 1, dtype=np.int32)
    if self_collision is None:
        collision_values = np.ones(capacity, dtype=np.int32)
    else:
        collision_values = np.ascontiguousarray(self_collision, dtype=np.int32).reshape(-1)
        if len(collision_values) != n_bodies:
            raise ValueError("Global body self-collision flags must match n_bodies")
        if np.any((collision_values != 0) & (collision_values != 1)):
            raise ValueError("Global body self-collision flags must be zero or one")
        if n_bodies == 0:
            collision_values = np.ones(capacity, dtype=np.int32)

    id_values = (
        np.empty(0, dtype=np.int32)
        if body_contact_ignorance_body_ids is None
        else np.ascontiguousarray(body_contact_ignorance_body_ids, dtype=np.int32).reshape(-1)
    )
    if body_contact_ignorance_ranges is None:
        range_values = np.zeros(capacity + 1, dtype=np.int32)
    else:
        range_values = np.ascontiguousarray(body_contact_ignorance_ranges, dtype=np.int32).reshape(-1)
        if len(range_values) != n_bodies + 1:
            raise ValueError("Global body ignorance ranges must have n_bodies + 1 entries")
        if range_values[0] != 0 or range_values[-1] != len(id_values):
            raise ValueError("Global body ignorance ranges must span the body-id array")
        if np.any(range_values[1:] < range_values[:-1]):
            raise ValueError("Global body ignorance ranges must be non-decreasing")
        if n_bodies == 0:
            range_values = np.zeros(capacity + 1, dtype=np.int32)

    def array(dtype, shape, values):
        result = qd.ndarray(dtype, shape=shape)
        result.from_numpy(values)
        return result

    id_storage = id_values if len(id_values) else np.zeros(1, dtype=np.int32)
    halfplane_offset = n_bodies if halfplane_body_id_offset is None else int(halfplane_body_id_offset)
    if halfplane_offset < 0:
        raise ValueError("Halfplane body ID offset must be non-negative")
    data = GlobalBodyManager.Data()
    data.n_bodies = array(qd.i32, (), np.array(n_bodies, dtype=np.int32))
    data.halfplane_body_id_offset = array(qd.i32, (), np.array(halfplane_offset, dtype=np.int32))
    data.n_body_contact_ignorance = array(qd.i32, (), np.array(len(id_values), dtype=np.int32))
    data.self_collision = array(qd.i32, (capacity,), collision_values)
    data.vertex_offsets = array(qd.i32, (capacity + 1,), offset_values)
    data.body_contact_ignorance_ranges = array(qd.i32, (capacity + 1,), range_values)
    data.body_contact_ignorance_body_ids = array(qd.i32, (max(len(id_values), 1),), id_storage)
    return data


@qd.func(requires_top_level=True)
def compute_vertex_offsets(data: qd.template(), fem: qd.template()):
    for i_body in range(fem.n_bodies[()] + 1):
        data.vertex_offsets[i_body] = fem.body_vertex_offsets[i_body]
    for i_body in range(fem.n_bodies[()]):
        data.self_collision[i_body] = fem.self_collision[i_body]


@qd.func
def is_body_contact_ignored(data: qd.template(), source, target):
    ignored = False
    begin = data.body_contact_ignorance_ranges[source]
    end = data.body_contact_ignorance_ranges[source + 1]
    for index in range(begin, end):
        if data.body_contact_ignorance_body_ids[index] == target:
            ignored = True
    return ignored
