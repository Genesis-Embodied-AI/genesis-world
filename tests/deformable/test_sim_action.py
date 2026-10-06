from __future__ import annotations

import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.systems.sim_system import ActionInvocation, SimData


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
def add_action_data(left: qd.template(), right: qd.template(), output: qd.template()):
    for _ in range(1):
        output[()] = left.value[()] + right.value[()]


@qd.kernel(graph=True, fastcache=True)
def run_action_with_registered_data(
    pipeline: qd.template(),
    action: qd.template(),
    output: qd.types.ndarray(qd.i32, ndim=0),
):
    if qd.static(pipeline.action_data_count == len(action.data)):
        action.kernel(*(action.data + (output,)))


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("backend", [gs.gpu])
def test_action_uses_multiple_data_registered_by_graph_pipeline():
    left = make_scalar(13)
    right = make_scalar(29)
    action = ActionInvocation(
        data=(left, right),
        kernel=add_action_data,
    )
    pipeline = PipelineRoot()
    pipeline.action_data_count = 2
    pipeline.action_data_0 = left
    pipeline.action_data_1 = right
    output = qd.ndarray(qd.i32, shape=())

    run_action_with_registered_data(pipeline, action, output)

    assert int(output.to_numpy()) == 42
