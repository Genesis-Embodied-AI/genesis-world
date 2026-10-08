import importlib
from pathlib import Path

import numpy as np
import pytest

import quadrants as qd

import genesis as gs
from genesis.engine.core import (
    Action,
    Engine,
    Find,
    HostAction,
    InlineAction,
    Pipeline,
    Require,
    StageAction,
    System,
    register_system,
    unregister_system,
)
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


pytestmark = [
    pytest.mark.required,
]


# An Engine has three deterministic admission paths: explicit instances, registered factories and recursive Require
# dependencies. Find observes only admitted Systems and never creates one.
@pytest.mark.parametrize("backend", [None])
def test_registered_manual_and_required_systems():
    scene = object()

    class Coordinator(System):
        pass

    class Manager(System):
        coordinator = Require(Coordinator)

    class Optional(System):
        pass

    class Manual(System):
        manager = Require(Manager)
        optional = Find(Optional)

    class Registered(System):
        manager = Require(Manager)

    class TestEngine(Engine):
        manager = Require(Manager)

    class OtherEngine(Engine):
        manager = Require(Manager)

    def create_registered(factory_scene):
        assert factory_scene is scene
        return Registered(factory_scene)

    def create_nothing(factory_scene):
        assert factory_scene is scene
        return None

    register_system(TestEngine)(create_registered)
    register_system(TestEngine)(create_nothing)
    try:
        engine = TestEngine(scene)
        manual = engine.add_system(Manual(scene))

        with pytest.raises(gs.GenesisException):
            _ = manual.manager
        with pytest.raises(TypeError):
            Manual.deps["invalid"] = Require(Optional)

        engine.build()

        assert list(engine.systems) == [Manual, Registered, Manager, Coordinator]
        assert engine.manager is manual.manager
        assert manual.manager.coordinator is engine.systems[Coordinator]
        assert manual.optional is None
        assert Manual.deps["manager"].label == "required"
        assert Manual.deps["optional"].label == "optional"

        with pytest.raises(gs.GenesisException):
            engine.add_system(Manual(scene))

        engine_with_optional = TestEngine(scene)
        optional = engine_with_optional.add_system(Optional(scene))
        manual_with_optional = engine_with_optional.add_system(Manual(scene))
        engine_with_optional.build()
        assert manual_with_optional.optional is optional

        other_engine = OtherEngine(scene)
        other_engine.build()
        assert list(other_engine.systems) == [Manager, Coordinator]
    finally:
        unregister_system(TestEngine, create_registered)
        unregister_system(TestEngine, create_nothing)


# System and Engine admit one concrete inheritance layer. Action admits exactly one final class for each of its three
# kinds, Pipeline is final, and register_system accepts a final Engine class alone.
@pytest.mark.parametrize("backend", [None])
def test_inheritance_and_action_kinds():
    class DirectSystem(System):
        pass

    class DirectEngine(Engine):
        pass

    with pytest.raises(gs.GenesisException):
        type("IndirectSystem", (DirectSystem,), {})
    with pytest.raises(gs.GenesisException):
        type("MixedSystem", (System, object), {})
    with pytest.raises(gs.GenesisException):
        type("IndirectEngine", (DirectEngine,), {})
    with pytest.raises(gs.GenesisException):
        type("MixedEngine", (Engine, object), {})
    with pytest.raises(gs.GenesisException):
        type("SpecialHostAction", (HostAction,), {})
    with pytest.raises(gs.GenesisException):
        type("UnknownAction", (Action,), {"kind": "compute"})
    with pytest.raises(gs.GenesisException):
        type("UnclassifiedAction", (Action,), {})
    with pytest.raises(gs.GenesisException):
        type("SecondHostAction", (Action,), {"kind": "host"})
    with pytest.raises(gs.GenesisException):
        type("SpecialPipeline", (Pipeline,), {})
    with pytest.raises(gs.GenesisException):
        register_system(Engine)
    with pytest.raises(gs.GenesisException):
        register_system(int)


# @System.action is a build-time declaration whose first access creates and caches one partial application. The declared
# kind matches the wrapped Quadrants callable, and a kernel is a Pipeline entry rather than an Action.
@pytest.mark.parametrize("backend", [None])
def test_action_binds_arguments_and_is_created_once_during_build():
    def add(left, right):
        return left + right

    @qd.func
    def add_inline(left: int, right: int):
        return left + right

    @qd.func(requires_top_level=True)
    def add_stage(left: int, right: int):
        return left + right

    @qd.kernel
    def kernel_entry():
        pass

    class Producer(System):
        @System.action(kind="host")
        def host(self):
            return add, 2

        @System.action(kind="inline")
        def inline(self):
            return add_inline, 3

        @System.action(kind="stage")
        def stage(self):
            return add_stage, 4

        def build(self):
            self.host_built = self.host
            self.inline_built = self.inline
            self.stage_built = self.stage

    class ProducerEngine(Engine):
        pass

    producer = Producer(object())
    with pytest.raises(gs.GenesisException):
        _ = producer.host

    engine = ProducerEngine(object())
    engine.add_system(producer)
    engine.build()

    assert producer.host.invoke((5,)) == 7
    assert producer.host is producer.host_built
    assert producer.inline is producer.inline_built
    assert producer.stage is producer.stage_built
    assert producer.host.kind == "host"
    assert producer.inline.kind == "inline"
    assert producer.stage.kind == "stage"

    class Mislabeled(System):
        @System.action(kind="stage")
        def action(self):
            return add_inline, 1

        def build(self):
            _ = self.action

    class MislabeledEngine(Engine):
        pass

    mislabeled_engine = MislabeledEngine(object())
    mislabeled_engine.add_system(Mislabeled(object()))
    with pytest.raises(gs.GenesisException):
        mislabeled_engine.build()

    class KernelAction(System):
        @System.action(kind="stage")
        def action(self):
            return kernel_entry

        def build(self):
            _ = self.action

    class KernelActionEngine(Engine):
        pass

    kernel_action_engine = KernelActionEngine(object())
    kernel_action_engine.add_system(KernelAction(object()))
    with pytest.raises(gs.GenesisException):
        kernel_action_engine.build()


# A collection freezes in deterministic ranked-then-unranked order. Call argument names and Action kind annotations are
# part of its protocol, whose build-time boundary closes before the collection freezes.
@pytest.mark.parametrize("backend", [None])
def test_protocol_and_action_collection_enforce_the_schedule_contract():
    @qd.func(requires_top_level=True)
    def stage(marker: qd.template(), segment: qd.template()):
        pass

    @qd.func(requires_top_level=True)
    def stage_with_wrong_name(marker: qd.template(), other: qd.template()):
        pass

    @qd.func
    def inline(marker: qd.template(), segment: qd.template()):
        pass

    def variadic(*args):
        return args

    class Manager(System):
        @System.action_collection(kind="stage", call_args=("segment",))
        def stages(self):
            pass

        @System.protocol(doc="protocol.md", references=("paper",))
        def on_stage(self, action: StageAction, rank: int | None = None):
            self.stages.add(action, rank=rank)

    manager = Manager(object())
    actions = [StageAction(stage, (object(),)) for _ in range(4)]

    with pytest.raises(gs.GenesisException):
        _ = manager.stages.actions
    with pytest.raises(gs.GenesisException):
        manager.on_stage(actions[0])

    manager.is_building = True
    with pytest.raises(gs.GenesisException):
        manager.on_stage(InlineAction(inline, (object(),)))
    with pytest.raises(gs.GenesisException):
        manager.stages.add(stage)
    with pytest.raises(gs.GenesisException):
        manager.stages.add(HostAction(lambda segment: None, ()))
    with pytest.raises(gs.GenesisException):
        manager.stages.add(StageAction(variadic, ()))
    with pytest.raises(gs.GenesisException):
        manager.stages.add(StageAction(stage_with_wrong_name, (object(),)))
    with pytest.raises(gs.GenesisException):
        manager.stages.add(actions[0], rank=-1)
    with pytest.raises(gs.GenesisException):
        manager.stages.add(actions[0], rank=True)

    manager.on_stage(actions[0])
    manager.on_stage(actions[1], rank=2)
    manager.on_stage(actions[2], rank=0)
    manager.on_stage(actions[3])
    with pytest.raises(gs.GenesisException):
        manager.on_stage(StageAction(stage, (object(),)), rank=0)

    manager.is_building = False
    manager.stages.freeze()

    assert manager.stages.actions == (actions[2], actions[1], actions[0], actions[3])
    assert manager.on_stage.__doc__.endswith("Guide: protocol.md\nReference: paper")


# A Pipeline has one entry point and accepts either a host function or a device kernel. Checkpoint callbacks resume a
# yielded graph from the callback-selected label with the same bound arguments.
@pytest.mark.parametrize("backend", [gs.cpu])
def test_pipeline_distinguishes_host_func_and_checkpointed_kernel_entries():
    host_result = [0]

    def host_entry(left, right):
        host_result[0] = left + right

    host_pipeline = Pipeline(host_entry, 2, 3)
    host_pipeline.run()
    assert host_pipeline.kind == "host"
    assert host_result[0] == 5

    @qd.func
    def func_entry():
        pass

    with pytest.raises(gs.GenesisException):
        Pipeline(func_entry)
    with pytest.raises(gs.GenesisException):
        Pipeline(host_entry, yield_callbacks={0: lambda _status: None})

    @qd.data_oriented
    class CheckpointData:
        value: qd.Tensor
        yield_required: qd.Tensor
        never_yield: qd.Tensor

    @qd.kernel(graph=True, checkpoints=True)
    def checkpoint_graph(data: qd.template()):
        with qd.checkpoint(3, yield_on=data.yield_required):
            for _ in range(1):
                data.value[()] = 10
        with qd.checkpoint(7, yield_on=data.never_yield):
            for _ in range(1):
                data.value[()] += 1

    data = CheckpointData()
    data.value = V(dtype=gs.qd_int, shape=())
    data.yield_required = V(dtype=gs.qd_int, shape=())
    data.never_yield = V(dtype=gs.qd_int, shape=())
    data.value.fill(0)
    data.yield_required.fill(1)
    data.never_yield.fill(0)

    with pytest.raises(gs.GenesisException):
        Pipeline(checkpoint_graph, data).run()

    def handle_yield(status):
        assert status.checkpoint == 3
        data.yield_required.fill(0)
        return 3

    pipeline = Pipeline(checkpoint_graph, data, yield_callbacks={3: handle_yield})
    pipeline.run()

    assert pipeline.kind == "device"
    assert qd_to_numpy(data.value) == 11


# The split example is the end-to-end contract: registration selects Rigid from the scene, Require adds Integrator,
# every participant exchanges state through Actions, and the device Pipeline advances the assembled DOFs.
@pytest.mark.parametrize("backend", [gs.cpu])
def test_registered_system_actions_integrate_global_dofs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples" / "engine"))

    Cloth = importlib.import_module("cloth_system").Cloth
    integrator_module = importlib.import_module("integrator_engine")
    Integrator = integrator_module.Integrator
    IntegratorEngine = integrator_module.IntegratorEngine
    Rigid = importlib.import_module("rigid_system").Rigid

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.5), show_viewer=False)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    engine = IntegratorEngine(scene)
    cloth = engine.add_system(Cloth(scene))
    engine.build()
    engine.init_pipeline.run()
    engine.step_pipeline.run()
    engine.step_pipeline.run()

    assert list(engine.systems) == [Cloth, Rigid, Integrator]
    assert_allclose(
        qd_to_numpy(engine.integrator.state.dofs_pos),
        np.array((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 110.0, 0.0, 0.0, 210.0, 0.0, 0.0), dtype=gs.np_float),
        tol=gs.EPS,
    )
    assert_allclose(
        qd_to_numpy(cloth.state.verts_pos),
        np.array(((110.0, 0.0, 0.0), (210.0, 0.0, 0.0)), dtype=gs.np_float),
        tol=gs.EPS,
    )
