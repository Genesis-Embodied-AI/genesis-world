import dataclasses

import pytest

import quadrants as qd

import genesis as gs
from genesis.engine.core import (
    Action,
    Data,
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
from genesis.utils.qd_ext import invoke_stage_actions

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


# Admission is idempotent: a function registered twice is registered once, and a registered function returning a class
# already added by hand adds nothing, so importing the module of a System class for the class alone is harmless.
@pytest.mark.parametrize("backend", [None])
def test_registration_and_admission_are_idempotent():
    scene = object()

    class Registered(System):
        pass

    class IdempotentEngine(Engine):
        pass

    def create_registered(factory_scene):
        return Registered(factory_scene)

    register_system(IdempotentEngine)(create_registered)
    register_system(IdempotentEngine)(create_registered)
    try:
        registered_engine = IdempotentEngine(scene)
        registered_engine.build()
        assert list(registered_engine.systems) == [Registered]

        manual_engine = IdempotentEngine(scene)
        manual = manual_engine.add_system(Registered(scene))
        manual_engine.build()
        assert list(manual_engine.systems) == [Registered]
        assert manual_engine.systems[Registered] is manual
    finally:
        unregister_system(IdempotentEngine, create_registered)

    unregistered_engine = IdempotentEngine(scene)
    unregistered_engine.build()
    assert list(unregistered_engine.systems) == []


# Engine creates every declared Action before closing the build window, so a manager invokes one its participant never
# accessed during build() directly afterwards.
@pytest.mark.parametrize("backend", [None])
def test_every_declared_action_is_created_by_build():
    def add(left, right):
        return left + right

    class Participant(System):
        @System.action(kind="host")
        def host(self):
            return add, 2

    class ParticipantEngine(Engine):
        pass

    participant = Participant(object())
    engine = ParticipantEngine(object())
    engine.add_system(participant)
    engine.build()

    assert participant.host.invoke((5,)) == 7
    assert participant.host is participant.host


# A dependency that a class-level Require / Find would turn into an import cycle is looked up during build() instead.
# The lookup observes the fixed System set: it returns what the Engine holds and adds nothing.
@pytest.mark.parametrize("backend", [None])
def test_build_time_find_and_require():
    scene = object()

    class Held(System):
        pass

    class Absent(System):
        pass

    class Looker(System):
        def build(self):
            self.held = self.require(Held)
            self.absent = self.find(Absent)

    class Demanding(System):
        def build(self):
            self.require(Absent)

    class LookupEngine(Engine):
        def build(self):
            super().build()
            self.looker = self.require(Looker)

    engine = LookupEngine(scene)
    held = engine.add_system(Held(scene))
    looker = engine.add_system(Looker(scene))
    with pytest.raises(gs.GenesisException):
        engine.find(Held)
    engine.build()
    assert looker.held is held
    assert looker.absent is None
    assert engine.looker is looker
    assert engine.find(Absent) is None
    assert list(engine.systems) == [Held, Looker]
    with pytest.raises(gs.GenesisException):
        looker.find(Held)

    demanding_engine = LookupEngine(scene)
    demanding_engine.add_system(Looker(scene))
    demanding_engine.add_system(Held(scene))
    demanding_engine.add_system(Demanding(scene))
    with pytest.raises(gs.GenesisException):
        demanding_engine.build()


# System and Engine admit one concrete inheritance layer. Action admits exactly one final class for each of its three
# kinds, Pipeline is final, and register_system accepts a final Engine class alone and registers functions, not classes.
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
    with pytest.raises(gs.GenesisException):
        register_system(DirectEngine)(DirectSystem)


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

    def host_with_keyword_only(marker, *, segment):
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

    class HostManager(System):
        @System.action_collection(kind="host", call_args=("segment",))
        def stages(self):
            pass

    manager = Manager(object())
    host_manager = HostManager(object())
    actions = [StageAction(stage, (object(),)) for _ in range(4)]

    with pytest.raises(gs.GenesisException):
        _ = manager.stages.actions
    with pytest.raises(gs.GenesisException):
        manager.on_stage(actions[0])

    manager.is_building = True
    host_manager.is_building = True
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
        host_manager.stages.add(HostAction(host_with_keyword_only, (object(),)))
    with pytest.raises(gs.GenesisException):
        manager.stages.add(StageAction(stage, (object(), object(), object())))
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
    with pytest.raises(gs.GenesisException):
        manager.stages.add(StageAction(stage, (object(),)))
    assert manager.on_stage.__doc__.endswith("Guide: protocol.md\nReference: paper")


@pytest.mark.parametrize("backend", [gs.cpu])
def test_global_data_context_collects_inherited_and_declared_data_for_actions():
    @qd.data_oriented
    class AutomaticData(Data):
        def __init__(self):
            self.value = qd.ndarray(qd.i32, shape=(1,))

    @qd.data_oriented
    class CustomData:
        def __init__(self):
            self.value = qd.ndarray(qd.i32, shape=(1,))

    class Participant(System):
        def __init__(self, scene):
            super().__init__(scene)
            self.automatic = AutomaticData()
            self.custom = CustomData()

        @System.data
        def custom_parameter(self):
            return self.custom

    class InvalidParticipant(System):
        @System.data
        def combined(self):
            return AutomaticData(), CustomData()

    @dataclasses.dataclass(frozen=True)
    class CustomDataclass:
        value: int

    class InvalidClassParticipant(System):
        @System.data
        def data_class(self):
            return CustomDataclass

    @qd.func(requires_top_level=True)
    def write(system: qd.template(), automatic: int, custom: int):
        for _ in range(1):
            system.automatic.value[0] = automatic
            system.custom.value[0] = custom

    @qd.kernel(graph=True)
    def run(context: qd.template(), action: qd.template()):
        action.invoke()

    participant = Participant(object())
    context_engine = Engine(object())
    context_engine.add_system(participant)
    context = context_engine.create_context()
    action = StageAction(write, (participant, 3, 5))
    run(context, action)
    qd.sync()

    assert participant.automatic.value.to_numpy().tolist() == [3]
    assert participant.custom.value.to_numpy().tolist() == [5]
    with pytest.raises(gs.GenesisException):
        invalid_engine = Engine(object())
        invalid_engine.add_system(InvalidParticipant(object()))
        invalid_engine.create_context()
    with pytest.raises(gs.GenesisException):
        invalid_engine = Engine(object())
        invalid_engine.add_system(InvalidClassParticipant(object()))
        invalid_engine.create_context()


# A Pipeline has one entry point and accepts either a host function or a device kernel. Checkpoint callbacks resume a
# yielded graph from the callback-selected label with the same bound arguments.
@pytest.mark.parametrize("backend", [gs.cpu])
def test_pipeline_distinguishes_host_func_and_checkpointed_kernel_entries():
    host_result = [0]

    def host_entry(left, right):
        host_result[0] = left + right

    host_pipeline = Pipeline(host_entry, 2)
    host_pipeline.run(3)
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


# The end-to-end contract: registration selects a participant from the scene, a participant is added by hand, Require
# adds their manager, every participant exchanges state through its Actions, and the device Pipeline advances it.
@pytest.mark.parametrize("backend", [gs.cpu])
def test_registered_system_actions_integrate_global_dofs():
    @qd.data_oriented
    class TotalState(Data):
        value: qd.Tensor

    class Total(System):
        def __init__(self, scene):
            super().__init__(scene)
            self.state = TotalState()

        @System.action_collection(kind="host")
        def inits(self):
            """Host Actions that fill the Data of each participant."""

        @System.action_collection(kind="stage", call_args=("total",))
        def contributions(self):
            """Stage Actions that add the share of each participant to `total`."""

        @System.protocol
        def on_total(self, *, init: HostAction, contribute: StageAction):
            """Join the total with an `init` HostAction and a `contribute` StageAction taking `total`."""
            self.inits += init
            self.contributions += contribute

        def init(self):
            self.state.value = V(dtype=gs.qd_float, shape=())
            self.state.value.fill(0.0)
            for action in self.inits.actions:
                action.invoke()

    @qd.data_oriented
    class ShareState(Data):
        value: qd.Tensor

    @qd.func(requires_top_level=True)
    def func_contribute(share_state: qd.template(), total: qd.template()):
        for _ in range(1):
            total.value[()] += share_state.value[()]

    def make_share_system(share):
        class Share(System):
            total = Require(Total)

            def __init__(self, scene):
                super().__init__(scene)
                self.state = ShareState()

            @System.action(kind="host")
            def init(self):
                return self.fill

            @System.action(kind="stage")
            def contribute(self):
                return func_contribute, self.state

            def build(self):
                self.total.on_total(init=self.init, contribute=self.contribute)

            def fill(self):
                self.state.value = V(dtype=gs.qd_float, shape=())
                self.state.value.fill(share)

        return Share

    Registered = make_share_system(1.0)
    Manual = make_share_system(2.0)

    class TotalEngine(Engine):
        total = Require(Total)

        def build(self):
            super().build()
            self.init_pipeline = Pipeline(self.total.init)

    @qd.kernel(graph=True)
    def kernel_step(context: qd.template(), engine: qd.template()):
        invoke_stage_actions(engine.total.contributions.actions, (engine.total.state,))

    def create_registered(scene):
        return Registered(scene) if scene.sim.rigid_solver.is_active else None

    register_system(TotalEngine)(create_registered)
    try:
        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
        engine = TotalEngine(scene)
        manual = engine.add_system(Manual(scene))
        engine.build()
        engine.init_pipeline.run()
        engine.step_pipeline = Pipeline(
            kernel_step,
            engine.create_context(),
            engine,
        )
        engine.step_pipeline.run()
        engine.step_pipeline.run()

        assert list(engine.systems) == [Manual, Registered, Total]
        assert engine.total is manual.total
        assert_allclose(qd_to_numpy(engine.total.state.value), 6.0, tol=gs.EPS)
    finally:
        unregister_system(TotalEngine, create_registered)
