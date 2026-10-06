from dataclasses import dataclass

import pytest

from genesis.engine.systems import SimData, SimPipeline, SimSystem


@dataclass(frozen=True)
class FakeGraphStatus:
    yielded: bool
    checkpoint: int


class FakeCheckpointGraph:
    def __init__(self) -> None:
        self.launches = []
        self.resumes = []

    def __call__(self, data, *args, **kwargs):
        self.launches.append((data, args, kwargs))
        return FakeGraphStatus(yielded=True, checkpoint=3)

    def resume(self, data, *args, from_checkpoint, **kwargs):
        self.resumes.append((data, args, from_checkpoint, kwargs))
        return FakeGraphStatus(yielded=False, checkpoint=-1)


class FakeActionData(SimData):
    value: int


def combine_action_data(left: FakeActionData, right: FakeActionData, scale: int) -> int:
    return left.value * scale + right.value


class MultiDataActionSystem(SimSystem):
    def __init__(self, left: FakeActionData, right: FakeActionData) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def build(self) -> None:
        self.action = self.create_action(combine_action_data, self.left, self.right)


def test_action_invokes_pure_function_with_ordered_data_inputs():
    left = FakeActionData()
    left.value = 4
    right = FakeActionData()
    right.value = 7
    system = MultiDataActionSystem(left, right)

    system._begin_build()
    system.build()
    system._end_build()

    assert system.action.data == (left, right)
    assert system.action.invocation(10) == 47


def test_pipeline_exposes_all_ordered_action_data_on_graph_root():
    left = FakeActionData()
    right = FakeActionData()
    system = MultiDataActionSystem(left, right)
    system._begin_build()
    system.build()
    system._end_build()

    root = type("PipelineRoot", (), {})()
    pipeline = SimPipeline(root, FakeCheckpointGraph())
    pipeline.bind_actions(system.action)

    assert root._sim_pipeline_action_data_0 is left
    assert root._sim_pipeline_action_data_1 is right


def test_pipeline_dispatches_yield_callback_and_resumes_from_returned_checkpoint():
    data = object()
    graph = FakeCheckpointGraph()
    pipeline = SimPipeline(data, graph)
    callback_statuses = []

    def handle_yield(status):
        callback_statuses.append(status)
        return 7

    pipeline.register_yield_callback(3, handle_yield)
    final_status = pipeline.run("argument", option=True)

    assert not final_status.yielded
    assert callback_statuses == [FakeGraphStatus(yielded=True, checkpoint=3)]
    assert graph.launches == [(data, ("argument",), {"option": True})]
    assert graph.resumes == [(data, ("argument",), 7, {"option": True})]


def test_pipeline_rejects_unknown_checkpoint_and_late_registration():
    pipeline = SimPipeline(object(), FakeCheckpointGraph())
    with pytest.raises(RuntimeError, match="no yield callback for checkpoint 3"):
        pipeline.run()
    with pytest.raises(RuntimeError, match="before SimPipeline.run"):
        pipeline.register_yield_callback(3, lambda status: status.checkpoint)
