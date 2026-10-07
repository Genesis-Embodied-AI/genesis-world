from __future__ import annotations

import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.sim_system import (
    ActionKind,
    SimAction,
    SimData,
    SimPipeline,
    SimSystem,
    validate_action_protocol,
)


@qd.data_oriented
class ScalarData(SimData):
    value: qd.Ndarray


@qd.data_oriented
class PipelineRoot:
    action_data_count: int


def make_scalar(value: int) -> ScalarData:
    data = ScalarData()
    data.value = qd.ndarray(qd.i32, shape=())
    data.value.from_numpy(np.array(value, dtype=np.int32))
    return data


@qd.func(requires_top_level=True)
def add_action_data(
    left: qd.template(),  # ScalarData
    right: qd.template(),  # ScalarData
    output: qd.template(),  # qd.Ndarray
):
    for _ in range(1):
        output[()] = left.value[()] + right.value[()]


@qd.kernel(graph=True, fastcache=True)
def run_action_with_registered_data(
    pipeline: qd.template(),
    action: qd.template(),
):
    if qd.static(pipeline.action_data_count == len(action.data)):
        action.invoke()


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_action_uses_multiple_data_registered_by_graph_pipeline():
    left = make_scalar(13)
    right = make_scalar(29)
    output = qd.ndarray(qd.i32, shape=())
    owner = ProbeSystem()
    action = SimAction(
        owner=owner,
        data=(left, right, output),
        kernel=add_action_data,
        kind=ActionKind.TOP_LEVEL_FUNC,
        transient_arity=0,
    )
    pipeline = PipelineRoot()
    pipeline.action_data_count = 3
    pipeline.action_data_0 = left
    pipeline.action_data_1 = right
    pipeline.action_data_2 = output

    run_action_with_registered_data(pipeline, action)

    assert int(output.to_numpy()) == 42


@qd.data_oriented
class ProbeSystem(SimSystem):
    @qd.data_oriented
    class Data(SimData):
        values: qd.Ndarray
        live_count: qd.Ndarray

    def __init__(self) -> None:
        super().__init__()
        self.data = ProbeSystem.Data()

    def init(self) -> None:
        self.data.values = qd.ndarray(qd.i32, shape=(4,))
        self.data.live_count = qd.ndarray(qd.i32, shape=())
        self.data.values.from_numpy(np.zeros(4, dtype=np.int32))
        self.data.live_count.from_numpy(np.array(4, dtype=np.int32))

    def build(self) -> None:
        self.increment_action = self.create_action(self.increment)
        self.step_action = self.create_action(self.on_step)

    @qd.func
    def increment(self, index):
        self.data.values[index] = self.data.values[index] + 1

    @qd.func(requires_top_level=True)
    def on_step(self):
        for index in range(self.data.live_count[()]):
            self.increment_action.invoke((index,))


@qd.data_oriented
class ProbeEngine:
    def __init__(self, system: ProbeSystem) -> None:
        self.system = system
        self.resize_required = qd.ndarray(qd.i32, shape=())
        self.never_yield = qd.ndarray(qd.i32, shape=())
        self.resize_required.from_numpy(np.array(0, dtype=np.int32))
        self.never_yield.from_numpy(np.array(0, dtype=np.int32))

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def step_graph(self):
        with qd.checkpoint(10, yield_on=self.resize_required):
            for _ in range(1):
                self.never_yield[()] = 0
        with qd.checkpoint(20, yield_on=self.never_yield):
            # WORKAROUND: Quadrants checkpoint lowering rejects a bare top-level
            # @qd.func call. Remove this static wrapper when that restriction is fixed.
            if qd.static(True):
                self.system.on_step()


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_bound_engine_graph_composes_top_level_and_inline_actions():
    system = ProbeSystem()
    system._begin_build()
    system.build()
    system._end_build()
    system.init()

    validate_action_protocol(
        system.step_action,
        protocol="step",
        expected_kind=ActionKind.TOP_LEVEL_FUNC,
        transient_arity=0,
    )
    validate_action_protocol(
        system.increment_action,
        protocol="increment",
        expected_kind=ActionKind.INLINE_FUNC,
        transient_arity=1,
    )

    engine = ProbeEngine(system)

    def grow_on_yield(_status):
        system.data.values = qd.ndarray(qd.i32, shape=(8,))
        system.data.live_count = qd.ndarray(qd.i32, shape=())
        system.data.values.from_numpy(np.zeros(8, dtype=np.int32))
        system.data.live_count.from_numpy(np.array(8, dtype=np.int32))
        engine.resize_required.from_numpy(np.array(0, dtype=np.int32))
        return 20

    pipeline = SimPipeline(engine.step_graph, yield_callbacks={10: grow_on_yield})
    pipeline.run()

    np.testing.assert_array_equal(system.data.values.to_numpy(), np.ones(4, dtype=np.int32))

    engine.resize_required.from_numpy(np.array(1, dtype=np.int32))
    pipeline.run()

    np.testing.assert_array_equal(system.data.values.to_numpy(), np.ones(8, dtype=np.int32))
