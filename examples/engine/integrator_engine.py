"""Integrator manager of the global general DOFs, and the Engine that arranges it.

Participants count their DOFs, receive a range of the global DOF vectors through an IntegratorSegment, and hand their
state to the Integrator, which advances every DOF with the explicit Euler method inside one graph-captured step.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.core import Engine, HostAction, InlineAction, Pipeline, Require, StageAction, System
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy

# WORKAROUND: @qd.data_oriented on Systems, Data and Segments only lets Quadrants discover their tensors from a kernel
# parameter. Systems hold no kernels or funcs: device code is free functions, bound to Data through Actions. A view
# handed to participants, such as IntegratorSegment, exposes its accessors as funcs.


@qd.data_oriented
class IntegratorInfo:
    dt: qd.Tensor
    segments_n_dofs: qd.Tensor
    segments_dof_start: qd.Tensor
    n_dofs: qd.Tensor


@qd.data_oriented
class IntegratorState:
    dofs_pos: qd.Tensor
    dofs_vel: qd.Tensor


@qd.data_oriented
class IntegratorSegment:
    """View of one participant on the global DOFs, and the only interface participants see.

    The DOF index `i_d_` of every accessor is relative to the start of the range of the participant.
    """

    def __init__(self, info: IntegratorInfo, state: IntegratorState, i_s: int) -> None:
        self._info = info
        self._state = state
        self._i_s = i_s

    @qd.func
    def set_n_dofs(self, n_dofs: int):
        self._info.segments_n_dofs[self._i_s] = n_dofs

    @qd.func
    def set_dof_position(self, i_d_: int, position: float):
        i_d = self._info.segments_dof_start[self._i_s] + i_d_
        self._state.dofs_pos[i_d] = position

    @qd.func
    def set_dof_velocity(self, i_d_: int, velocity: float):
        i_d = self._info.segments_dof_start[self._i_s] + i_d_
        self._state.dofs_vel[i_d] = velocity

    @qd.func
    def get_dof_position(self, i_d_: int):
        i_d = self._info.segments_dof_start[self._i_s] + i_d_
        return self._state.dofs_pos[i_d]


@qd.data_oriented
class Integrator(System):
    """Manager of the global general DOFs: participants count their DOFs, receive an offset, and exchange state."""

    def __init__(self, scene: gs.Scene) -> None:
        super().__init__(scene)
        self.info = IntegratorInfo()
        self.state = IntegratorState()
        self.segments: tuple[IntegratorSegment, ...] = ()

    @System.action_collection(kind="host")
    def inits(self):
        """Host Actions that initialize the Data of each participant."""

    @System.action_collection(kind="inline", call_args=("segment",))
    def counts(self):
        """Inline device functions that report the DOF count of each participant through its Segment."""

    @System.action_collection(kind="stage", call_args=("segment",))
    def gathers(self):
        """Stage device functions that write the state of each participant into its Segment."""

    @System.action_collection(kind="stage", call_args=("segment",))
    def scatters(self):
        """Stage device functions that read the state of each participant back from its Segment."""

    @System.protocol(doc="integrator_protocol.md", references=("https://en.wikipedia.org/wiki/Euler_method",))
    def on_integrate(
        self,
        *,
        init: HostAction,
        count: InlineAction,
        gather: StageAction,
        scatter: StageAction,
        rank: int | None = None,
    ):
        """Join the global general-DOF integration, calling it once from build() with all four Actions.

        init     HostAction    fill your own Data
        count    InlineAction  report your DOF count through `segment`
        gather   StageAction   write your state into `segment` before integration
        scatter  StageAction   read the integrated state back from `segment`
        rank     int | None    your position in the Integrator, None keeping the add order

        The guide holds the full contract: model, offsets, invocation order, and what each Action must and must not do.
        """
        # The same rank in all four collections keeps their frozen orders, and therefore the slot, identical
        self.inits.add(init, rank=rank)
        self.counts.add(count, rank=rank)
        self.gathers.add(gather, rank=rank)
        self.scatters.add(scatter, rank=rank)

    def init(self):
        """Initialize every participant, count DOFs in one kernel, read back the total once, then allocate."""
        n_segments = len(self.counts.actions)
        self.info.dt = V(dtype=gs.qd_float, shape=())
        self.info.dt.fill(self._scene.options.sim.dt)
        self.info.segments_n_dofs = V(dtype=gs.qd_int, shape=(n_segments,))
        self.info.segments_dof_start = V(dtype=gs.qd_int, shape=(n_segments,))
        self.info.n_dofs = V(dtype=gs.qd_int, shape=())
        self.segments = tuple(IntegratorSegment(self.info, self.state, i_s) for i_s in range(n_segments))
        for action in self.inits.actions:
            action.invoke()
        kernel_count_dofs(self)
        n_dofs = int(qd_to_numpy(self.info.n_dofs))
        self.state.dofs_pos = V(dtype=gs.qd_float, shape=(n_dofs,))
        self.state.dofs_vel = V(dtype=gs.qd_float, shape=(n_dofs,))


# One-time sizing uses a plain kernel, and a graph with yield checkpoints is reserved for growth during stepping
@qd.kernel
def kernel_count_dofs(integrator: qd.template()):
    # One serial task calls every inline count Action
    for _ in range(1):
        for action, segment in qd.static(tuple(zip(integrator.counts.actions, integrator.segments))):
            action.invoke((segment,))
    func_derive_dof_start(integrator.info)


@qd.func(requires_top_level=True)
def func_derive_dof_start(integrator_info: qd.template()):
    for _ in range(1):
        n_dofs = 0
        for i_s in range(integrator_info.segments_n_dofs.shape[0]):
            integrator_info.segments_dof_start[i_s] = n_dofs
            n_dofs += integrator_info.segments_n_dofs[i_s]
        integrator_info.n_dofs[()] = n_dofs


@qd.func(requires_top_level=True)
def func_integrate_dofs(integrator_state: qd.template(), integrator_info: qd.template()):
    for i_d in range(integrator_info.n_dofs[()]):
        integrator_state.dofs_pos[i_d] += integrator_info.dt[()] * integrator_state.dofs_vel[i_d]


# Device phase called from the step entry of the Engine. qd.static unrolls the frozen collections at compile time.
@qd.func(requires_top_level=True)
def func_step_integrator(integrator: qd.template()):
    for action, segment in qd.static(tuple(zip(integrator.gathers.actions, integrator.segments))):
        action.invoke((segment,))
    func_integrate_dofs(integrator.state, integrator.info)
    for action, segment in qd.static(tuple(zip(integrator.scatters.actions, integrator.segments))):
        action.invoke((segment,))


@qd.data_oriented
class IntegratorEngine(Engine):
    """The engine that arranges the Integrator, which in turn arranges its participants."""

    integrator = Require(Integrator)

    def build(self):
        super().build()
        self.init_pipeline = Pipeline(self.init)
        self.step_pipeline = Pipeline(kernel_step, self)

    def init(self):
        self.integrator.init()


# The Engine is the kernel argument, since Quadrants discovers the Data of every participant through it
@qd.kernel(graph=True)
def kernel_step(engine: qd.template()):
    func_step_integrator(engine.integrator)
