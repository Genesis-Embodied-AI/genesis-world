import functools
import weakref
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.grad.bridge import LaunchRecorder, Operation, RolloutGraph
from genesis.options.morphs import Morph
from genesis.options.solvers import IPCCouplerOptions, LegacyCouplerOptions, SAPCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import DataItem, DataKind, ErrorCode, iter_data
from genesis.utils.misc import indices_to_mask, qd_to_numpy, qd_to_torch
from genesis.utils.tools import StepTimer

from .couplers import IPCCoupler, LegacyCoupler, SAPCoupler
from .entities import HybridEntity
from .sensors import SensorManager
from .solvers import (
    FEMSolver,
    KinematicSolver,
    MPMSolver,
    PBDSolver,
    RigidSolver,
    SFSolver,
    SPHSolver,
)
from .solvers.base_solver import GravityMixin, TimeBasedMixin, merge_errno
from .state_change import StateChange, Subscriber
from .states.solvers import SimState, SimulatorCheckpoint, SolverCheckpoint

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity, EntityDescription
    from genesis.engine.scene import Scene
    from genesis.options.scene import SceneOptions

    from .solvers.base_solver import Solver


# The kinds of the state a step or a substep starts from, from which its backward pass replays it
STATE_KINDS = frozenset((DataKind.STATE,))


def restore_checkpoints(solvers_checkpoints: Iterable[tuple["Solver", SolverCheckpoint]]) -> None:
    """Write back the boundary checkpoint of every given solver, then flush the zero-copy writes of the restore on
    Metal."""
    for solver, checkpoint in solvers_checkpoints:
        solver.load_checkpoint(checkpoint, boundary_only=True)
    if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
        torch.mps.synchronize()


class Simulator(RBC):
    """
    A simulator is a scene-level simulation manager, which manages all simulation-related operations in the scene, including multiple solvers and the inter-solver coupler.

    Parameters
    ----------
    scene : gs.Scene
        The scene object that the simulator is associated with.
    options : SceneOptions
        Every option the scene was created with. The simulator keeps the one that configures itself and hands each
        solver, the coupler and the visualizer the one that configures it. All of them stay reachable as
        ``sim.scene.options``.
    """

    def __init__(self, scene: "Scene", options: "SceneOptions"):
        self._scene = scene
        self._step_timer = StepTimer()

        # options
        self.options = options.sim

        self._dt: float = self.options.dt
        self._substep_dt: float = self.options.dt / self.options.substeps
        self._substeps: int = self.options.substeps
        self._requires_grad: bool = self.options.requires_grad

        self._cur_substep_global = 0

        # Autograd graph of the rollout (see genesis.grad.bridge), which a reset starts anew (see 'is_forward_ready' for
        # the steps a backward pass forbids)
        def begin_backward_pass():
            # The pass rewinds the state of the simulation, so stepping forward stays forbidden until the next reset,
            # even after a pass that raises. The errors the rollout flagged raise before any step is reversed, rather
            # than with the context of the step being reversed when a deferred check reads them. The adjoints start
            # from zero.
            self._is_forward_ready = False
            for solver in self._active_solvers:
                solver.update_errno()
                solver.check_errno()
                solver.reset_grad()

        def end_backward_pass():
            # The errors the solvers flagged while replaying the rollout are raised once the pass is over
            for solver in self._active_solvers:
                solver.check_errno()

        self._rollout_graph = RolloutGraph(
            self._requires_grad, on_pass_start=begin_backward_pass, on_pass_end=end_backward_pass
        )
        self._is_forward_ready = True

        # solvers
        self.rigid_solver = RigidSolver(self.scene, self, options.rigid)
        self.kinematic_solver = KinematicSolver(self.scene, self, options.kinematic)
        self.mpm_solver = MPMSolver(self.scene, self, options.mpm)
        self.sph_solver = SPHSolver(self.scene, self, options.sph)
        self.pbd_solver = PBDSolver(self.scene, self, options.pbd)
        self.fem_solver = FEMSolver(self.scene, self, options.fem)
        self.sf_solver = SFSolver(self.scene, self, options.sf)

        self._solvers: list["Solver"] = gs.List(
            [
                self.rigid_solver,
                self.kinematic_solver,
                self.mpm_solver,
                self.sph_solver,
                self.pbd_solver,
                self.fem_solver,
                self.sf_solver,
            ]
        )

        self._active_solvers: list["Solver"] = gs.List()

        # coupler
        if isinstance(options.coupler, SAPCouplerOptions):
            self._coupler = SAPCoupler(self, options.coupler)
        elif isinstance(options.coupler, LegacyCouplerOptions):
            self._coupler = LegacyCoupler(self, options.coupler)
        elif isinstance(options.coupler, IPCCouplerOptions):
            self._coupler = IPCCoupler(self, options.coupler)
        else:
            gs.raise_exception(
                f"Coupler options {options.coupler} not supported. Please use SAPCouplerOptions, "
                "LegacyCouplerOptions, or IPCCouplerOptions."
            )

        # entities
        self._entities: list["Entity"] = gs.List()

        # sensors
        self._sensor_manager = SensorManager(self)

    def _add_entity(
        self,
        morph: Morph | None = None,
        material=None,
        surface=None,
        visualize_contact=False,
        name: str | None = None,
        desc: "EntityDescription | None" = None,
    ):
        if desc is not None:
            material = desc.material
        if visualize_contact and not isinstance(material, gs.materials.Rigid):
            gs.raise_exception("'visualize_contact' only applies to rigid entities.")
        if isinstance(material, gs.materials.Hybrid):
            # Note that adding to solver is handled in the hybrid entity
            entity = HybridEntity(self.n_entities, self.scene, material, morph, surface, name=name)
        else:
            # Several solvers may declare a class the material belongs to, since 'Rigid' derives from 'Kinematic'. The
            # one declaring the most derived class simulates it (see 'Solver.material_cls').
            solver = None
            for candidate in self._solvers:
                if candidate.material_cls is None or not isinstance(material, candidate.material_cls):
                    continue
                if solver is None or issubclass(candidate.material_cls, solver.material_cls):
                    solver = candidate
            if solver is None:
                gs.raise_exception(f"No solver simulates entities of material {type(material).__name__}.")
            entity = solver.add_entity(
                self.n_entities, material, morph, surface, visualize_contact, name=name, desc=desc
            )
        self._entities.append(entity)
        if entity.desc is not None:
            self.scene._desc.entities.append(entity.desc)
        return entity

    def _add_force_field(self, force_field):
        for solver in self._solvers:
            solver._add_force_field(force_field)

    def build(self):
        self.n_envs = self.scene.n_envs
        self._B = self.scene._B
        self._para_level = self.scene._para_level

        # Settled before any solver builds: a solver fills its own buffers with the interval it holds, so a rate
        # settled afterwards would leave those buffers describing an interval the loop does not run at. Whether a
        # solver is active follows from the entities added to it, which is already known at this point. A solver that
        # integrates over no interval of its own has no rate to agree on.
        integrating_solvers = [
            solver for solver in self._solvers if isinstance(solver, TimeBasedMixin) and solver.is_active
        ]
        # A solver given an interval asks for a rate of its own, whatever that interval works out to. A solver left
        # unset takes the rate the shorthand of SimOptions decides, and has nothing to reconcile.
        asking_solvers = [solver for solver in integrating_solvers if "dt" in solver._options.model_fields_set]
        for solver in asking_solvers:
            interval = solver._options.dt
            n_substeps = solver.substeps
            if abs(n_substeps * interval - self._dt) > gs.EPS * self._dt:
                gs.raise_exception(
                    f"{type(solver).__name__} dt={interval} does not divide the step dt={self._dt} an integer number "
                    "of times."
                )
            # Asked for, rather than left at its default, which is what tells a count of one from no count at all.
            if "substeps" in self.options.model_fields_set and n_substeps != self._substeps:
                gs.raise_exception(
                    f"{type(solver).__name__} dt={interval} implies {n_substeps} substep(s) per step, conflicting with "
                    f"the requested substeps={self._substeps}. Set one or the other."
                )
        rates = {solver.substeps for solver in asking_solvers}
        if len(rates) > 1:
            gs.raise_exception(
                "Solvers integrating at different rates are not supported yet, got "
                + ", ".join(f"{type(solver).__name__} at dt={solver._options.dt}" for solver in asking_solvers)
                + "."
            )
        substeps = next(iter(rates), self._substeps)
        self._substeps = substeps
        self._substep_dt = self._dt / self._substeps
        # The substep loop advances every active solver once per iteration, so the rate one of them asks for is the
        # rate they all take, the ones asking for nothing included.
        for solver in integrating_solvers:
            solver._substeps = self._substeps
            solver._substep_dt = self._substep_dt

        # Counted per environment to support per-env reset, as steps rather than seconds to avoid drifting over time.
        self._steps = torch.zeros((self._B,), dtype=gs.tc_int, device=gs.device)

        # solvers
        # IPCCoupler needs full substep flow for pre/post coupling phases
        self._rigid_only = self.rigid_solver.is_active and not isinstance(self._coupler, (SAPCoupler, IPCCoupler))
        for solver in self._solvers:
            solver.build()
            if solver.is_active:
                self._active_solvers.append(solver)
                if not isinstance(solver, RigidSolver):
                    self._rigid_only = False

        # A coupler exchanges state once per substep, so it is built once the rate that loop runs at is known.
        self._coupler.build()

        # Every active solver and the coupler of a differentiable scene support gradients, which they declare. The
        # coupler declares it for the couplings it activated at build.
        if self._requires_grad:
            unsupported_names = [
                type(solver).__name__ for solver in self._active_solvers if not solver.is_differentiable
            ]
            if not self._coupler.is_differentiable:
                unsupported_names.append(type(self._coupler).__name__)
            if unsupported_names:
                gs.raise_exception(
                    f"Differentiable simulation ('requires_grad=True') is not supported by {unsupported_names}."
                )

        if self.n_envs > 0 and self.sf_solver.is_active:
            gs.raise_exception("Batching is not supported for SF solver as of now.")

        # hybrid
        for entity in self._entities:
            if isinstance(entity, HybridEntity):
                entity.build()

        # A change of the model invalidates every operation of the rollout (see 'RolloutGraph.invalidate'), while a
        # change of the state is a call the graph records (see 'mutates'). The state of a solver tracking no gradient
        # stays outside the graph.
        for solver in self._active_solvers:
            if solver.requires_grad:
                solver.subscribe(Subscriber(frozenset((StateChange.MODEL,)), callback=self._rollout_graph.invalidate))

        # On CPU, the flags of the solvers are merged straight from their numpy views (see 'merge_errno'), and whether
        # any env is flagged spares the calls of 'check_errno' at every step but those that raise
        self._errno_views = None
        self._has_errno_flags = False
        if gs.backend == gs.cpu and gs.use_zerocopy.is_supported:
            self._errno_views = [
                (solver._errno_latched.numpy(), qd_to_numpy(solver._errno, copy=False))
                for solver in self._active_solvers
                if solver._errno is not None
            ]

        self._step_timer.add_phases("physics", "sensors")
        self._sensor_manager.build()

    def destroy(self):
        self._sensor_manager.destroy()

    def reset(self, state: SimState, envs_idx=None):
        for solver, solver_state in zip(self._solvers, state):
            if solver.n_entities > 0:
                solver.set_state(0, solver_state, envs_idx)

        if envs_idx is None:
            self._steps.zero_()
        else:
            self._steps[indices_to_mask(envs_idx)] = 0
        self._restart(envs_idx)

    def _restart(self, envs_idx=None):
        """Restart the coupler, the autograd graph of the rollout and the sensors."""
        self._coupler.reset(envs_idx=envs_idx)

        # The adjoints of the state start from zero, and a new rollout of the autograd graph begins
        for solver in self._active_solvers:
            solver.reset_grad()
        self._rollout_graph.restart()
        self._is_forward_ready = True
        # The cursor is a position in the frames the solvers hold, not a clock, so it rewinds whole.
        self._cur_substep_global = 0

        # reset sensors state
        self._sensor_manager.reset(envs_idx=envs_idx)

    def data(self, kinds: frozenset[DataKind]) -> Iterator[DataItem]:
        """Yield every item of the requested kinds that the simulator and its active solvers hold.

        The items of the simulator are named under 'Simulator', and the ones of a solver under its class name.
        """
        if isinstance(self._coupler, IPCCoupler):
            gs.raise_exception(
                "A scene coupled by IPC cannot be checkpointed yet: the IPC world holds state of its own."
            )
        for item in iter_data(self, "steps", DataKind.STATE):
            if item.kind in kinds:
                yield item._replace(name=f"Simulator.{item.name}")
        for solver in self._active_solvers:
            prefix = type(solver).__name__
            for item in solver.data:
                if isinstance(item, DataItem) and item.kind in kinds:
                    yield item._replace(name=f"{prefix}.{item.name}")

    def __getstate__(self) -> SimulatorCheckpoint:
        """Return a SimulatorCheckpoint of the simulation, for '__setstate__' to restore."""
        if isinstance(self._coupler, IPCCoupler):
            gs.raise_exception(
                "A scene coupled by IPC cannot be checkpointed yet: the IPC world holds state of its own."
            )
        return SimulatorCheckpoint(
            steps=self._steps.clone(),
            solvers={type(solver).__name__: solver.__getstate__() for solver in self._active_solvers},
        )

    def __setstate__(self, state: SimulatorCheckpoint) -> None:
        """Put the built simulation back in a state '__getstate__' read.

        Everything around the state restarts as under a reset.

        The clock of each environment comes back as recorded, since simulated time is state. The autograd graph of the
        rollout, the gradients, the coupler and the sensors are restarted (see 'reset'): they describe the run that led
        here.
        """
        # The record was checked against every solver by the scene (see 'Scene.__setstate__'). The restart precedes the
        # state, whose gradients it would zero otherwise.
        self._restart()
        for solver in self._active_solvers:
            solver.__setstate__(state.solvers[type(solver).__name__])
        self._steps[:] = torch.as_tensor(state.steps, device=gs.device)
        # Flush the zero-copy writes of fill_data on Metal.
        if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
            torch.mps.synchronize()

    def check_errno(self, blocking: bool = True):
        """Raise on the first error a solver flagged since the last reset of each env (see 'Solver.check_errno')."""
        for solver in self._active_solvers:
            solver.check_errno(blocking)

    def get_error_envs_mask(self) -> torch.Tensor:
        """Return whether each env flagged an error in any solver since its last reset."""
        envs_mask = torch.zeros((self._B,), dtype=torch.bool, device=gs.device)
        for solver in self._active_solvers:
            envs_mask |= solver.get_error_envs_mask()
        return envs_mask

    def clear_errno(self, envs_idx=None):
        """Clear the errors every solver flagged for the given envs, all of them if None."""
        for solver in self._active_solvers:
            solver.clear_errno(envs_idx)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def step(self):
        # The errors the solvers flagged are raised at the beginning of a step, never between substeps, so that the envs
        # reset between two steps carry on (see 'Scene.get_error_envs_mask'). The check does not block where the device
        # allows it (see 'Solver.check_errno').
        if self._errno_views is None:
            self.check_errno(blocking=False)
        elif self._has_errno_flags:
            self.check_errno()
            self._has_errno_flags = False

        with self._step_timer.phase("physics"):
            if self._rigid_only and not self._requires_grad:  # "Only Advance!" --Thomas Wade :P
                self._steps += 1
                for _ in range(self._substeps):
                    self.rigid_solver.substep(self.cur_substep_local)
                    self._cur_substep_global += 1
                self.rigid_solver.clear_external_force()
            elif self._rollout_graph.is_recording:
                self._rollout_graph.record_step(StepOperation(self))
            else:
                # A step the graph does not record overwrites the state the last recorded operation returned
                self._rollout_graph.mark_state_changed()
                self._step()

        # The flags of the step are merged once, as it is over, which leaves the checks and the mask of the flagged envs
        # mere reads of them (see 'Solver.update_errno')
        if self._errno_views is not None:
            for errno_out, errno_in in self._errno_views:
                if merge_errno(errno_out, errno_in):
                    self._has_errno_flags = True
        else:
            for solver in self._active_solvers:
                solver.update_errno()

        # The sensors read the state for themselves, outside of the autograd graph of a differentiable rollout
        with self._step_timer.phase("sensors"), torch.no_grad():
            self._sensor_manager.step()

    def _step(self):
        """Run the substeps of a step, rolling the frames the solvers hold over at the end of each window of them."""
        self._steps += 1
        for _ in range(self._substeps):
            self.substep(self.cur_substep_local)
            self._cur_substep_global += 1
            if self.cur_substep_local == 0:
                for solver in self._active_solvers:
                    if solver.has_frames:
                        solver.copy_frame(solver.n_frames - 1, 0)
        if self.rigid_solver.is_active:
            self.rigid_solver.clear_external_force()

    def _reverse(self, substep_start: int, steps_start: torch.Tensor, checkpoints: list[SolverCheckpoint]) -> None:
        """
        Run the backward pass of the step starting at the given global substep, from the boundary checkpoints of the
        differentiable solvers.

        Each substep, last first, is replayed from its start while its kernels are recorded, and the recorded kernels
        reversed (see LaunchRecorder), the end of the step with the last one. The substeps before the last are replayed
        once beforehand, to rebuild the frames of the step and save the start of each of them. The clocks of the
        environments go back to `steps_start`, the steps they counted before it.
        """
        # A solver holding its state in frames keeps the state of every substep of the step. The state of the others is
        # saved at the start of every substep but the first, whose start the boundary checkpoints hold.
        differentiable_solvers = [solver for solver in self._active_solvers if solver.requires_grad]
        boundary_checkpoints = list(zip(differentiable_solvers, checkpoints))
        substeps_checkpoints = [boundary_checkpoints]
        self._cur_substep_global = substep_start
        restore_checkpoints(boundary_checkpoints)
        for _ in range(self._substeps - 1):
            self.substep(self.cur_substep_local)
            self._cur_substep_global += 1
            substeps_checkpoints.append(
                [
                    (solver, solver.save_checkpoint(STATE_KINDS, boundary_only=True))
                    for solver in differentiable_solvers
                    if not solver.has_frames
                ]
            )
        # The roll of the frames overwrites frame 0 in arrays the recorder cannot see, which the boundary checkpoints of
        # the solvers holding their state in frames restore
        framed_checkpoints = [(solver, checkpoint) for solver, checkpoint in boundary_checkpoints if solver.has_frames]
        restore_frames = functools.partial(restore_checkpoints, framed_checkpoints)
        for i_substep in reversed(range(self._substeps)):
            self._cur_substep_global = substep_start + i_substep
            is_last_substep = i_substep == self._substeps - 1
            if not is_last_substep:
                restore_checkpoints(substeps_checkpoints[i_substep])
            with LaunchRecorder() as recorder:
                self.substep(self.cur_substep_local)
                if is_last_substep:
                    with recorder.entry(restore=restore_frames, autodiff=True):
                        for solver in self._active_solvers:
                            if solver.has_frames:
                                solver.copy_frame(solver.n_frames - 1, 0)
                    if self.rigid_solver.is_active:
                        self.rigid_solver.clear_external_force()
            recorder.reverse()
        self._cur_substep_global = substep_start
        self._steps.copy_(steps_start)

    def substep(self, f):
        self._coupler.preprocess(f)
        self.substep_pre_coupling(f)
        self._coupler.couple(f)
        self.substep_post_coupling(f)

    # -------------- pre coupling --------------
    def substep_pre_coupling(self, f):
        for solver in self._active_solvers:
            solver.substep_pre_coupling(f)

    # -------------- post coupling --------------
    def substep_post_coupling(self, f):
        for solver in self._active_solvers:
            solver.substep_post_coupling(f)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self):
        state = SimState(
            scene=self.scene,
            s_global=self.cur_step_global,
            f_local=self.cur_substep_local,
            solvers=self._solvers,
        )

        return state

    def set_gravity(self, gravity, envs_idx=None):
        for solver in self._solvers:
            if solver.is_active and isinstance(solver, GravityMixin):
                solver.set_gravity(gravity, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def steps(self) -> torch.Tensor:
        """The number of steps each environment has run since its last reset, of shape [B]."""
        return self._steps

    @property
    def dt(self) -> float:
        """The time duration for each simulation step."""
        return self._dt

    @property
    def substeps(self):
        """The number of substeps per simulation step."""
        return self._substeps

    @property
    def substep_dt(self) -> float:
        """Duration of one substep, in seconds, which is the interval every solver integrates over."""
        return self._substep_dt

    @property
    def scene(self):
        """The scene object that the simulator is associated with."""
        return self._scene

    @property
    def requires_grad(self):
        """Whether the simulator requires gradients."""
        return self._requires_grad

    @property
    def is_forward_ready(self) -> bool:
        """Whether the simulation may step forward, which a backward pass rewinding it forbids until the next reset."""
        return self._is_forward_ready

    @property
    def n_entities(self) -> int:
        """The number of entities in the simulator."""
        return len(self._entities)

    @property
    def entities(self):
        """The list of entities in the simulator."""
        return self._entities

    @property
    def cur_substep_global(self):
        """The current substep of the simulation."""
        return self._cur_substep_global

    @property
    def cur_substep_local(self):
        """
        The frame the current substep starts from, among those the solvers hold.

        The solvers of a differentiable scene hold the frames of the substeps of one step, those of any other scene the
        frames of one substep (see 'Solver.has_frames').
        """
        return self._cur_substep_global % (self._substeps if self._requires_grad else 1)

    @property
    def step_timer(self) -> StepTimer:
        """The timer of the phases of the steps."""
        return self._step_timer

    @property
    def cur_step_global(self):
        """Number of `scene.step()` calls, counted for the whole batch.

        Use it to tell that the simulation moved on, for instance to invalidate a cache. For the simulated time of an
        environment, use `get_time`.
        """
        return self._cur_substep_global // self._substeps

    def get_time(self, envs_idx=None):
        """The simulated time of each environment, in seconds.

        Environments are stepped and reset independently, so simulated time is per environment, and this is where it
        is read from.
        """
        time = self._steps[indices_to_mask(envs_idx)].to(dtype=gs.tc_float) * self._dt
        return time[0] if self.n_envs == 0 else time

    @property
    def cur_t(self):
        """How far the substep loop has advanced, in seconds: the number of substeps run times the substep interval.

        Shared by every environment, so it tells that the simulation moved on rather than what one environment has
        simulated, which is `get_time`.
        """
        return self._cur_substep_global * self._substep_dt

    @property
    def coupler(self):
        """The coupler object that manages the inter-solver coupling."""
        return self._coupler

    @property
    def solvers(self):
        """The list of solvers in the simulator."""
        return self._solvers

    @property
    def active_solvers(self):
        """The list of active solvers in the simulator."""
        return self._active_solvers


class StepOperation(Operation):
    """A step of a differentiable scene, as one operation of its autograd graph (see genesis.grad.bridge)."""

    def __init__(self, sim: Simulator):
        # See OperationNode in genesis.grad.bridge for the weak reference
        self._sim = weakref.proxy(sim)
        self._substep_start = sim.cur_substep_global
        self._steps_start = sim.steps.clone()
        # The state each differentiable solver holds at the start of the step, from which its backward pass replays it
        self._checkpoints = None

    def forward(self) -> tuple[torch.Tensor, ...]:
        self._checkpoints = [
            solver.save_checkpoint(STATE_KINDS, boundary_only=True)
            for solver in self._sim._active_solvers
            if solver.requires_grad
        ]
        self._sim._step()
        return ()

    def backward(self) -> tuple[torch.Tensor | None, ...]:
        self._sim._reverse(self._substep_start, self._steps_start, self._checkpoints)
        # The adjoint of every earlier step derives from this one, so an adjoint that is not finite raises during the
        # pass, naming the step a shorter horizon would have to stop before. It is read from every array of the state
        # that carries one, so every differentiable solver is covered. The arrays are those the boundary checkpoint of
        # the step holds, whose gradients the solver lists once (see 'BoundaryItems'). The check does not block where
        # the device allows it, so an error raises one step later, and at the end of the pass for the first step.
        for solver in self._sim._active_solvers:
            if not solver.requires_grad:
                continue
            grads = [
                *solver._boundary_items.grads,
                *[
                    qd_to_torch(item.array.grad, transpose=True)[..., None]
                    for item in solver._boundary_items.items_grad_copied
                ],
            ]
            if grads:
                envs_finite = torch.stack([grad.isfinite().all(dim=tuple(range(1, grad.ndim))) for grad in grads])
                solver.flag_errno(ErrorCode.INVALID_GRAD_NAN, ~envs_finite.all(dim=0))
            solver.update_errno()
            solver.check_errno(
                blocking=False, context=f"Backward pass of step {self._substep_start // self._sim._substeps}"
            )
        return ()
