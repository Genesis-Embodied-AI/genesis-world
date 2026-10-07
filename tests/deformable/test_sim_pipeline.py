from dataclasses import dataclass
import inspect
from pathlib import Path

import numpy as np
import pytest
import quadrants as qd

from genesis.engine.systems import (
    ActionKind,
    ConsistentIPCContactConstitution,
    ContactSystem,
    GlobalLinearSystem,
    LBVHBroadPhase,
    PCGSolver,
    RigidContactAssemble,
    RigidContactProxySystem,
    RigidJointForestSystem,
    SimData,
    SimPipeline,
    SimSystem,
    validate_action_protocol,
)
from genesis.engine.systems.finite_element import (
    FiniteElementMethod,
    QuadraticBending,
    StrainLimitBaraffWitkinShell2D,
)
from genesis.engine.systems.rigid_system import RigidSystem


@dataclass(frozen=True)
class FakeGraphStatus:
    yielded: bool
    checkpoint: int


class FakeCheckpointGraph:
    def __init__(self) -> None:
        self.launches = []
        self.resumes = []

    def __call__(self, *args, **kwargs):
        self.launches.append((args, kwargs))
        return FakeGraphStatus(yielded=True, checkpoint=3)

    def resume(self, *args, from_checkpoint, **kwargs):
        self.resumes.append((args, from_checkpoint, kwargs))
        return FakeGraphStatus(yielded=False, checkpoint=-1)


@qd.data_oriented
class FakeActionData(SimData):
    value: int


@qd.func
def combine_action_data(
    left: qd.template(),  # FakeActionData
    right: qd.template(),  # FakeActionData
    scale: int,
) -> int:
    return left.value * scale + right.value


@qd.data_oriented
class MultiDataActionSystem(SimSystem):
    def __init__(self) -> None:
        super().__init__()
        self.left = FakeActionData()
        self.right = FakeActionData()

    def wire_data(self, left: FakeActionData, right: FakeActionData) -> None:
        self.left = left
        self.right = right

    def build(self) -> None:
        self.action = self.create_action(combine_action_data, self.left, self.right)


def test_sim_system_rejects_indirect_and_multiple_inheritance():
    missing = [
        system_type.__name__
        for system_type in SimSystem.__subclasses__()
        if not system_type.__dict__.get("_data_oriented", False)
    ]
    assert not missing, f"Concrete SimSystem classes must be explicitly @qd.data_oriented: {missing}"

    with pytest.raises(TypeError, match="must inherit directly and only from SimSystem"):

        class IndirectSystem(MultiDataActionSystem):
            pass

    class AdditionalBase:
        pass

    with pytest.raises(TypeError, match="must inherit directly and only from SimSystem"):

        class MultipleSystem(SimSystem, AdditionalBase):
            def build(self) -> None:
                pass


@pytest.mark.parametrize(
    ("system_type", "method_arities"),
    [
        (
            RigidSystem,
            {
                "on_predict": 0,
                "on_assemble_candidate_rows": 0,
                "on_initialize_newton": 0,
                "on_assemble": 1,
                "on_negate_direction": 1,
                "on_record_start_point": 0,
                "on_energy": 0,
                "on_step_forward": 1,
                "on_set_newton_active": 1,
                "on_build_preconditioner": 1,
                "on_update_velocity": 0,
            },
        ),
        (
            FiniteElementMethod,
            {
                "on_initialize_global_vertices": 0,
                "on_sync_from_scene": 0,
                "on_forward_global_vertices": 0,
                "on_predict": 0,
                "on_extent": 0,
                "on_assemble": 0,
                "on_energy": 0,
                "on_negate_direction": 0,
                "on_contribute_newton_max_displacement": 1,
                "on_record_start_point": 0,
                "on_reset_energy": 0,
                "on_step_forward": 1,
                "on_update_velocity": 0,
                "on_copy_previous_positions": 0,
                "on_publish_trajectory_end_positions": 0,
            },
        ),
        (
            GlobalLinearSystem,
            {
                "on_derive_extents": 0,
                "on_compute_n_triplets": 0,
                "on_zero_rhs": 0,
            },
        ),
        (
            PCGSolver,
            {
                "solve": 3,
            },
        ),
        (
            RigidContactProxySystem,
            {
                "on_initialize_state": 0,
                "on_prepare_metric": 0,
                "on_reset_frame": 0,
                "on_mark_mechanism_constrained": 0,
                "on_prepare_tolerance": 0,
                "on_initialize_newton": 0,
                "on_prepare_constraint": 0,
                "on_capture_physical_gradient": 0,
                "on_prepare_path_limit": 0,
                "on_contribute_newton_max_displacement": 1,
                "on_apply_convergence": 1,
                "on_record_start_point": 0,
                "on_forward_global_vertices": 0,
                "on_publish_trajectory_end_positions": 0,
                "on_compute_restoration_energy": 1,
                "on_initialize_merit": 0,
                "on_step_forward": 1,
                "on_evaluate_trial_guard": 0,
                "on_finalize_restoration_step": 1,
                "on_recover_reaction": 0,
                "on_copy_previous_state": 0,
                "on_initialize_global_vertices": 0,
            },
        ),
        (
            RigidJointForestSystem,
            {
                "on_compute_endpoint_fk": 0,
                "on_prepare_particular": 0,
                "on_particular_spmv": 0,
                "on_project_physical_rhs": 0,
                "on_build_preconditioner": 0,
                "on_expand_solution": 0,
                "on_compute_merit_directional_derivative": 0,
            },
        ),
        (
            RigidContactAssemble,
            {
                "on_classify": 0,
                "on_distribute": 0,
            },
        ),
        (
            ContactSystem,
            {
                "on_reset_initial_intersections": 0,
                "on_flag_et_intersections": 0,
                "on_reset_counted_demand": 0,
                "on_update_adaptive_kappa": 0,
                "on_tick_adaptive_kappa_newton": 0,
                "on_reset_collision_counts": 0,
                "on_query_halfplanes": 0,
                "on_initialize_ccd": 0,
                "on_reset_frame": 0,
                "on_compute_ccd_alpha_pt": 0,
                "on_compute_ccd_alpha_ee": 0,
                "on_compute_ccd_alpha_ph": 0,
                "on_reduce_ccd_alpha": 0,
                "on_ccd": 0,
                "on_reset_contact_energy": 0,
                "on_sum_contact_energy": 0,
                "on_check_assembly_capacity": 0,
                "on_check_assembly_padding": 0,
                "on_shrink_assembly_padding": 0,
                "on_reset_assembly_counts": 0,
                "on_sort_reduce": 0,
            },
        ),
        (
            LBVHBroadPhase,
            {
                "on_build_triangles": 0,
                "on_build_edges": 0,
                "on_query_pt": 0,
                "on_query_ee": 0,
                "on_query_trajectory": 0,
                "on_detect_initial_intersections": 0,
            },
        ),
        (
            ConsistentIPCContactConstitution,
            {
                "on_snapshot_lagged_positions": 0,
                "on_filter_friction_pairs_pt": 0,
                "on_filter_friction_pairs_ee": 0,
                "on_filter_friction_pairs_ph": 0,
                "on_snapshot_friction": 0,
                "on_count_active_pt": 0,
                "on_count_active_ee": 0,
                "on_count_active_ph": 0,
                "on_count_active": 0,
                "on_filter_assemble_pt": 0,
                "on_filter_assemble_ee": 0,
                "on_filter_assemble_ph": 0,
                "on_friction_assemble_pt": 0,
                "on_friction_assemble_ee": 0,
                "on_friction_assemble_ph": 0,
                "on_filter_assemble": 0,
                "on_filter_energy_pt": 0,
                "on_filter_energy_ee": 0,
                "on_filter_energy_ph": 0,
                "on_friction_energy_pt": 0,
                "on_friction_energy_ee": 0,
                "on_friction_energy_ph": 0,
                "on_contact_energy": 0,
            },
        ),
    ],
)
def test_fixed_lifecycle_apis_are_named_top_level_bound_funcs(system_type, method_arities):
    data_type = getattr(system_type, "Data", None)
    if system_type is ConsistentIPCContactConstitution:
        assert data_type is None
    else:
        assert issubclass(data_type, SimData)
    for method_name, transient_arity in method_arities.items():
        method = system_type.__dict__[method_name]
        assert method._is_quadrants_function
        assert method._qd_requires_top_level
        assert len(inspect.signature(method.fn).parameters) == transient_arity + 1


def test_fixed_inline_bound_func_metadata():
    method = RigidContactProxySystem.__dict__["on_check_line_search"]
    assert method._is_quadrants_function
    assert not getattr(method, "_qd_requires_top_level", False)
    assert len(inspect.signature(method.fn).parameters) == 8


def test_system_template_parameters_document_concrete_types():
    systems_root = Path(__file__).parents[2] / "genesis" / "engine" / "systems"
    undocumented = []
    for path in systems_root.rglob("*.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            marker = ": qd.template()"
            if marker not in line:
                continue
            parameter, suffix = line.split(marker, maxsplit=1)
            if parameter.strip().isidentifier():
                comment = suffix.partition("#")[2].strip()
                if not comment or comment.startswith(("noqa", "type: ignore")):
                    undocumented.append(f"{path.relative_to(systems_root)}:{line_number}")
    assert undocumented == []


def test_constitution_wire_validation_is_atomic():
    bending = QuadraticBending()
    with pytest.raises(ValueError, match="wire-data lengths"):
        bending.wire_data(
            np.zeros((1, 4), dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            np.zeros((1, 16), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )
    assert bending._hinge_indices is None
    assert bending._bending_stiffness is None
    assert bending._Q0 is None
    assert bending._vert_bend_k is None

    membrane = StrainLimitBaraffWitkinShell2D()
    with pytest.raises(ValueError, match="wire-data lengths"):
        membrane.wire_data(
            np.zeros(1, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )
    assert membrane._tri_indices is None
    assert membrane._mu is None
    assert membrane._lambda is None
    assert membrane._strain_limit_multiplier is None


def test_action_invokes_pure_function_with_ordered_data_inputs():
    left = FakeActionData()
    left.value = 4
    right = FakeActionData()
    right.value = 7
    system = MultiDataActionSystem()
    system.wire_data(left, right)

    system._begin_build()
    system.build()
    system._end_build()

    assert system.action.data == (left, right)
    assert system.action.kind is ActionKind.INLINE_FUNC
    assert system.action.transient_arity == 1
    assert system.action.kernel.fn(left, right, 10) == 47


def test_action_protocol_validates_actual_quadrants_metadata():
    left = FakeActionData()
    right = FakeActionData()
    system = MultiDataActionSystem()
    system.wire_data(left, right)
    system._begin_build()
    system.build()
    system._end_build()

    validate_action_protocol(
        system.action,
        protocol="inline_combine",
        expected_kind=ActionKind.INLINE_FUNC,
        transient_arity=1,
    )
    with pytest.raises(TypeError, match="requires TOP_LEVEL_FUNC"):
        validate_action_protocol(
            system.action,
            protocol="scheduled_combine",
            expected_kind=ActionKind.TOP_LEVEL_FUNC,
            transient_arity=0,
        )


def test_pipeline_dispatches_yield_callback_and_resumes_from_returned_checkpoint():
    graph = FakeCheckpointGraph()
    callback_statuses = []

    def handle_yield(status):
        callback_statuses.append(status)
        return 7

    pipeline = SimPipeline(graph, yield_callbacks={3: handle_yield})
    final_status = pipeline.run("argument", option=True)

    assert not final_status.yielded
    assert callback_statuses == [FakeGraphStatus(yielded=True, checkpoint=3)]
    assert graph.launches == [(("argument",), {"option": True})]
    assert graph.resumes == [(("argument",), 7, {"option": True})]


def test_pipeline_rejects_unknown_checkpoint_and_allows_callback_rebinding():
    pipeline = SimPipeline(FakeCheckpointGraph())
    with pytest.raises(RuntimeError, match="no yield callback for checkpoint 3"):
        pipeline.run()

    resumed = []

    def resume_from_three(_status):
        resumed.append(3)
        return 3

    pipeline.yield_callbacks[3] = resume_from_three
    pipeline.run()
    assert resumed == [3]

    def resume_from_seven(_status):
        resumed.append(7)
        return 7

    pipeline.yield_callbacks[3] = resume_from_seven
    pipeline.run()
    assert resumed == [3, 7]
