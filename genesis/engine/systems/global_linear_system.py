from __future__ import annotations

import numpy as np
import quadrants as qd

from genesis.utils.misc import qd_to_numpy

from .bcoo_matrix import BCOOMatrix
from .bcoo_operations import sym_bcoo_spmv_naive
from .sim_system import ActionKind, SimAction, SimData, SimSystem, validate_action_protocol


@qd.data_oriented  # WORKAROUND: Quadrants bound @qd.func self must be data-oriented.
class GlobalLinearSystem(SimSystem):
    """Global linear problem layout, vectors, and solver entry point."""

    @qd.data_oriented
    class Data(SimData):
        """Device-visible global matrix, vector, and extent state."""

        n_block_rows: qd.Ndarray
        total_dof: qd.Ndarray
        dof_block_base: qd.Ndarray
        n_extent_slots: qd.Ndarray
        n_elastic: qd.Ndarray
        required_block_rows: qd.Ndarray
        extent_slots: qd.Ndarray
        extent_offsets: qd.Ndarray
        matrix: BCOOMatrix
        x_sol: qd.Ndarray
        b_rhs: qd.Ndarray

        @qd.func
        def set_subsystem_extent(self, subsystem_id, extent):
            """Publish one subsystem's live triplet demand."""
            self.extent_slots[subsystem_id] = extent

        @qd.func
        def subsystem_offset(self, subsystem_id):
            """Return the derived global triplet offset for one subsystem."""
            return self.extent_offsets[subsystem_id]

        @qd.func
        def read_rhs(self, dof):
            return self.b_rhs[dof]

        @qd.func
        def write_rhs(self, dof, value):
            self.b_rhs[dof] = value

        @qd.func
        def atomic_add_rhs(self, dof, value):
            qd.atomic_add(self.b_rhs[dof], value)

        @qd.func
        def read_solution(self, dof):
            return self.x_sol[dof]

        @qd.func
        def write_solution(self, dof, value):
            self.x_sol[dof] = value

        @qd.func
        def add_solution(self, dof, value):
            self.x_sol[dof] = self.x_sol[dof] + value

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self.capacity_grow_factor: float = 1.2
        self.extent_actions = self.create_action_collection()
        self.assemble_actions = self.create_action_collection()
        self.extent_schedule = ()
        self.assemble_schedule = ()
        self._n_block_rows: int | None = None
        self._n_elastic_triplets: int | None = None
        self._max_contact_body_triplets: int | None = None
        self._dof_block_base: int | None = None
        self._legacy_sort_reduce: bool | None = None

    def wire_data(
        self,
        *,
        n_block_rows: int,
        n_elastic_triplets: int,
        max_contact_body_triplets: int,
        dof_block_base: int,
        genesis_legacy_sort_reduce: bool = False,
        capacity_grow_factor: float = 1.2,
    ) -> None:
        if min(n_block_rows, n_elastic_triplets, max_contact_body_triplets, dof_block_base) < 0:
            raise ValueError("Global linear system sizes must be non-negative")
        if capacity_grow_factor <= 1.0:
            raise ValueError("GlobalLinearSystem capacity grow factor must be greater than one")
        self.capacity_grow_factor = float(capacity_grow_factor)
        self._n_block_rows = n_block_rows
        self._n_elastic_triplets = n_elastic_triplets
        self._max_contact_body_triplets = max_contact_body_triplets
        self._dof_block_base = dof_block_base
        self._legacy_sort_reduce = genesis_legacy_sort_reduce

    def build(self) -> None:
        from .contact_system import ContactSystem
        from .pcg_solver import PCGSolver
        from .sim_config import SimConfig

        self.contact_system = self.find(ContactSystem)
        self.pcg_solver_system = self.require(PCGSolver)
        self.sim_config_system = self.require(SimConfig)
        self.pcg_operator_action = self.create_action(pcg_apply_operator, self.data)
        self.pcg_solver_system.on_primary_operator(self.pcg_operator_action)

    def on_subsystem(self, *, extent: SimAction, assemble: SimAction) -> None:
        validate_action_protocol(
            extent,
            protocol="GlobalLinearSystem.on_subsystem.extent",
            expected_kind=ActionKind.TOP_LEVEL_FUNC,
            transient_arity=2,
        )
        validate_action_protocol(
            assemble,
            protocol="GlobalLinearSystem.on_subsystem.assemble",
            expected_kind=ActionKind.TOP_LEVEL_FUNC,
            transient_arity=3,
        )
        if extent.rank != assemble.rank:
            raise RuntimeError("GlobalLinearSystem extent and assemble Actions must have the same rank")
        self.extent_actions.register(extent)
        self.assemble_actions.register(assemble)

    def on_triplet_overflow_yield(self, _status):
        """Grow triplet storage and select the safe resume checkpoint."""
        from .sim_engine import ContactCheckpoint

        matrix = self.data.matrix
        required = int(qd_to_numpy(matrix.n_triplets))
        matrix.grow(
            max(
                int(np.ceil(required * self.capacity_grow_factor)),
                matrix.triplet_row.shape[0] + 1,
            ),
            live_size=required,
        )
        matrix.triplet_overflow.from_numpy(np.array(0, dtype=np.int32))
        return ContactCheckpoint.SOLVE

    def init(self) -> None:
        if (
            self._n_block_rows is None
            or self._n_elastic_triplets is None
            or self._max_contact_body_triplets is None
            or self._dof_block_base is None
            or self._legacy_sort_reduce is None
        ):
            raise RuntimeError("GlobalLinearSystem data has not been wired")
        n_block_rows = self._n_block_rows
        n_elastic_triplets = self._n_elastic_triplets
        max_contact_body_triplets = self._max_contact_body_triplets
        dof_block_base = self._dof_block_base
        genesis_legacy_sort_reduce = self._legacy_sort_reduce
        total_dof = n_block_rows * 3
        self.extent_schedule = self.extent_actions.actions
        self.assemble_schedule = self.assemble_actions.actions
        extent_storage = max(len(self.extent_schedule), 1)
        dof_storage = max(total_dof, 1)

        def scalar(value):
            result = qd.ndarray(qd.i32, shape=())
            result.from_numpy(np.array(value, dtype=np.int32))
            return result

        self.data.n_block_rows = scalar(n_block_rows)
        self.data.total_dof = scalar(total_dof)
        self.data.dof_block_base = scalar(dof_block_base)
        self.data.n_extent_slots = scalar(len(self.extent_schedule))
        self.data.n_elastic = scalar(n_elastic_triplets)
        self.data.required_block_rows = scalar(n_block_rows)
        self.data.extent_slots = qd.ndarray(qd.i32, shape=(extent_storage,))
        self.data.extent_offsets = qd.ndarray(qd.i32, shape=(extent_storage,))
        self.data.extent_slots.from_numpy(np.zeros(extent_storage, dtype=np.int32))
        self.data.extent_offsets.from_numpy(np.full(extent_storage, n_elastic_triplets, dtype=np.int32))
        self.data.matrix = BCOOMatrix(
            shape=(n_block_rows, n_block_rows),
            block_shape=(3, 3),
            value_type=qd.f64,
            symmetric=True,
            initial_triplets=n_elastic_triplets,
            max_triplets=n_elastic_triplets + max_contact_body_triplets,
            genesis_legacy_sort_reduce=genesis_legacy_sort_reduce,
        )
        self.data.x_sol = qd.ndarray(qd.f64, shape=(dof_storage,))
        self.data.b_rhs = qd.ndarray(qd.f64, shape=(dof_storage,))
        self.data.x_sol.from_numpy(np.zeros(dof_storage, dtype=np.float64))
        self.data.b_rhs.from_numpy(np.zeros(dof_storage, dtype=np.float64))
        self._n_block_rows = None
        self._n_elastic_triplets = None
        self._max_contact_body_triplets = None
        self._dof_block_base = None
        self._legacy_sort_reduce = None

    @qd.func(requires_top_level=True)
    def on_derive_extents(self):
        derive_extents(self.data)

    @qd.func(requires_top_level=True)
    def on_report_extents(self):
        for linear_system_id, action in qd.static(enumerate(self.extent_schedule)):
            action.invoke((self.data, linear_system_id))

    @qd.func(requires_top_level=True)
    def on_assemble_subsystems(self):
        for linear_system_id, action in qd.static(enumerate(self.assemble_schedule)):
            action.invoke((self.sim_config_system.data, self.data, linear_system_id))

    @qd.func(requires_top_level=True)
    def on_compute_n_triplets(self):
        compute_n_triplets(self.data, self.contact_system.data)

    @qd.func(requires_top_level=True)
    def on_zero_rhs(self):
        zero_rhs(self.data)


@qd.func(requires_top_level=True)
def derive_extents(
    data: qd.template(),  # GlobalLinearSystem.Data
):
    for _ in range(1):
        total = qd.i32(0)
        for slot in range(data.n_extent_slots[()]):
            data.extent_offsets[slot] = total
            total = total + data.extent_slots[slot]
        data.n_elastic[()] = total
        data.matrix.set_n_triplets(total)


@qd.func(requires_top_level=True)
def compute_n_triplets(
    data: qd.template(),  # GlobalLinearSystem.Data
    contact_data: qd.template(),  # ContactSystem.Data
):
    for _ in range(1):
        total = data.n_elastic[()] + contact_data.n_unique_triplets[()]
        data.matrix.report_triplet_demand(total)


@qd.func(requires_top_level=True)
def zero_rhs(
    data: qd.template(),  # GlobalLinearSystem.Data
):
    for i in range(data.total_dof[()]):
        data.b_rhs[i] = qd.f64(0.0)


@qd.func(requires_top_level=True)
def pcg_apply_operator(
    data: qd.template(),  # GlobalLinearSystem.Data
    _linear_system_data: qd.template(),  # GlobalLinearSystem.Data
    direction: qd.template(),  # qd.Ndarray
    output: qd.template(),  # qd.Ndarray
):
    sym_bcoo_spmv_naive(data.matrix, direction, output)
