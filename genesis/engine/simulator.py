from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.options.morphs import Morph
from genesis.options.solvers import IPCCouplerOptions, LegacyCouplerOptions, SAPCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import DataItem, DataKind
from genesis.utils.misc import indices_to_mask

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
    ToolSolver,
)
from .solvers.base_solver import GravityMixin, TimeBasedMixin
from .states.cache import QueriedStates
from .states.solvers import SimState, SimulatorCheckpoint

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity, EntityDescription
    from genesis.engine.scene import Scene
    from genesis.options.scene import SceneOptions

    from .solvers.base_solver import Solver


RATE_CHECK_ERRNO = 10


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

        # options
        self.options = options.sim

        self._dt: float = self.options.dt
        self._substep_dt: float = self.options.dt / self.options.substeps
        self._substeps: int = self.options.substeps
        self._substeps_local: int | None = self.options.substeps_local
        self._requires_grad: bool = self.options.requires_grad
        self._steps_local: int | None = self.options._steps_local

        self._cur_substep_global = 0

        # solvers
        self.tool_solver = ToolSolver(self.scene, self, options.tool)
        self.rigid_solver = RigidSolver(self.scene, self, options.rigid)
        self.kinematic_solver = KinematicSolver(self.scene, self, options.kinematic)
        self.mpm_solver = MPMSolver(self.scene, self, options.mpm)
        self.sph_solver = SPHSolver(self.scene, self, options.sph)
        self.pbd_solver = PBDSolver(self.scene, self, options.pbd)
        self.fem_solver = FEMSolver(self.scene, self, options.fem)
        self.sf_solver = SFSolver(self.scene, self, options.sf)

        self._solvers: list["Solver"] = gs.List(
            [
                self.tool_solver,
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

        # states
        self._queried_states = QueriedStates()

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
        # The differentiable window was sized from the authored substeps before the solvers allocated their buffers,
        # so a rate derived afterwards would leave the tape describing a step of a different length than the one run.
        if self._requires_grad and substeps != self._substeps:
            gs.raise_exception(
                f"A solver dt deriving {substeps} substep(s) per step is not supported in differentiable mode, which "
                f"is recording {self._substeps}. Ask for the rate through `SimOptions.substeps={substeps}` instead."
            )
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

        if self.n_envs > 0 and self.sf_solver.is_active:
            gs.raise_exception("Batching is not supported for SF solver as of now.")

        # hybrid
        for entity in self._entities:
            if isinstance(entity, HybridEntity):
                entity.build()

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
        """Restart the coupler, the gradient tape and the sensors."""
        self._coupler.reset(envs_idx=envs_idx)

        # TODO: keeping as is for now
        self.reset_grad()
        # The tape cursor is a position in the recorded window, not a clock, so it rewinds whole.
        self._cur_substep_global = 0

        # reset sensors state
        self._sensor_manager.reset(envs_idx=envs_idx)

    def data(self, kinds: frozenset[DataKind]) -> Iterator[DataItem]:
        """Yield every item of the given kinds the active solvers hold, each under the class name of its solver."""
        if isinstance(self._coupler, IPCCoupler):
            gs.raise_exception(
                "A scene coupled by IPC cannot be checkpointed yet: the IPC world holds state of its own."
            )
        for solver in self._active_solvers:
            prefix = type(solver).__name__
            for name, value, kind in solver.data:
                if kind in kinds:
                    yield DataItem(f"{prefix}.{name}", value, kind)

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

        The clock of each environment comes back as recorded, since simulated time is state. The tape cursor, the
        gradients, the coupler and the sensors are restarted (see 'reset'): they describe the run that led here.
        """
        # The record was checked against every solver by the scene (see 'Scene.__setstate__'). The restart precedes the
        # state, whose gradients it would zero otherwise.
        self._restart()
        for solver in self._active_solvers:
            solver.__setstate__(state.solvers[type(solver).__name__])
        self._steps[:] = torch.as_tensor(state.steps, device=gs.device)
        # Flush the zero-copy writes of fill_data on Metal.
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def reset_grad(self):
        for solver in self._active_solvers:
            solver.reset_grad()

        # clear up all queried scene states and free up memory
        self._queried_states.clear()

    # ------------------------------------------------------------------------------------
    # -------------------------------- step computation ----------------------------------
    # ------------------------------------------------------------------------------------
    """
    We use f to represent substep, and s to represent step.
    """

    def f_global_to_f_local(self, f_global):
        f_local = f_global % self._substeps_local
        return f_local

    def f_local_to_s_local(self, f_local):
        f_local = f_local // self._substeps
        return f_local

    def f_global_to_s_local(self, f_global):
        f_local = self.f_global_to_f_local(f_global)
        s_local = self.f_local_to_s_local(f_local)
        return s_local

    def f_global_to_s_global(self, f_global):
        s_global = f_global // self._substeps
        return s_global

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def step(self, in_backward=False):
        # Check errno at the very beginning of the step.
        # This will trigger GPU sync, but it is not a big deal at the point, since we are going to enqueue very large
        # kernel right away. Moreover, if computations are still not done at this point, then the queue will just
        # continue growing endlessly, which will not make the simulation faster either.
        if self.rigid_solver.is_active and self._cur_substep_global % RATE_CHECK_ERRNO == 0:
            self.rigid_solver.check_errno()

        # Reconstructing a checkpoint window replays steps the environments already simulated, so only a forward step
        # advances their clock. The backward pass winds it down again through `_step_grad`.
        if not in_backward:
            self._steps += 1

        if self._rigid_only and not self._requires_grad:  # "Only Advance!" --Thomas Wade :P
            for _ in range(self._substeps):
                self.rigid_solver.substep(self.cur_substep_local)
                self._cur_substep_global += 1
        else:
            self.process_input(in_backward=in_backward)
            for _ in range(self._substeps):
                self.substep(self.cur_substep_local)

                self._cur_substep_global += 1
                if self.cur_substep_local == 0 and not in_backward:
                    self.save_ckpt()

        if self.rigid_solver.is_active:
            self.rigid_solver.clear_external_force()

        self._sensor_manager.step()

    def _step_grad(self):
        self._steps -= 1
        for _ in range(self._substeps - 1, -1, -1):
            if self.cur_substep_local == 0:
                self.load_ckpt()
            self._cur_substep_global -= 1

            self.sub_step_grad(self.cur_substep_local)

        self.process_input_grad()

    def process_input(self, in_backward=False):
        """
        setting _tgt state using external commands
        note that external inputs are given at step level, not substep
        """
        for solver in self._active_solvers:
            solver.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for solver in reversed(self._active_solvers):
            solver.process_input_grad()

    def substep(self, f):
        self._coupler.preprocess(f)
        self.substep_pre_coupling(f)
        self._coupler.couple(f)
        self.substep_post_coupling(f)

    def sub_step_grad(self, f):
        self.substep_post_coupling_grad(f)
        self._coupler.couple_grad(f)
        self.substep_pre_coupling_grad(f)

    # -------------- pre coupling --------------
    def substep_pre_coupling(self, f):
        for solver in self._active_solvers:
            solver.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        for solver in reversed(self._active_solvers):
            solver.substep_pre_coupling_grad(f)

    # -------------- post coupling --------------
    def substep_post_coupling(self, f):
        for solver in self._active_solvers:
            solver.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        for solver in reversed(self._active_solvers):
            solver.substep_post_coupling_grad(f)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def add_grad_from_state(self, state):
        for solver, solver_state in zip(self._solvers, state):
            solver.add_grad_from_state(solver_state)

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """

        # simulator-level states
        if self.cur_step_global in self._queried_states:
            # one step could have multiple states
            for state in self._queried_states[self.cur_step_global]:
                self.add_grad_from_state(state)

        # each solver will have their own entities, each of which stores a set of _queried_states
        for solver in self._active_solvers:
            solver.collect_output_grads()

    def save_ckpt(self):
        """
        This function refreshes the gpu memory (copy the last frame to the first frame in the local memory), and then saves the checkpoint.
        This function is called every `substeps_local` steps, which means it's called only once per step when `requires_grad` is True.
        """
        ckpt_start_substep = self._cur_substep_global - self._substeps_local
        ckpt_end_step = self._cur_substep_global - 1
        ckpt_name = f"{ckpt_start_substep}"

        for solver in self._active_solvers:
            solver.save_ckpt(ckpt_name)

        if self._requires_grad:
            gs.logger.debug(
                f"Forward: Saved checkpoint for global substep {ckpt_start_substep} to {ckpt_end_step}. Now starts from substep {self._cur_substep_global}."
            )

    def load_ckpt(self):
        ckpt_start_substep = self._cur_substep_global - self._substeps_local
        ckpt_end_step = self._cur_substep_global - 1
        ckpt_name = f"{ckpt_start_substep}"

        for solver in self._active_solvers:
            solver.load_ckpt(ckpt_name)

        # now that we loaded the first frame, we do a forward pass to fill up the rest
        self._cur_substep_global = ckpt_start_substep
        for _ in range(self._steps_local):
            self.step(in_backward=True)

        gs.logger.debug(
            f"Backward: Loaded checkpoint for global substep {ckpt_start_substep} to {ckpt_end_step}. Now starts from substep {ckpt_start_substep}."
        )

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

        # `SimState.__init__` calls `solver.get_state` on every solver, and solvers that maintain a per-solver queue
        # (kinematic/rigid) push the returned state there as a within-step cache. The SimState itself is registered just
        # below for grad collection at the simulator level, so the per-solver entry would cause `collect_output_grads`
        # to dispatch `kernel_get_state_grad` twice on the same state and double the adjoint via atomic_add. Lift those
        # entries here. Solvers without solver-state registration (mpm, fem, sph, pbd, sf, tool) leave their queue
        # empty, so `discard` is a no-op for them.
        for solver, solver_state in zip(self._solvers, state.solvers_state):
            if solver_state is not None:
                solver._queried_states.discard(solver_state)

        # store all queried states to track gradient flow
        self._queried_states.append(state)

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
    def n_entities(self) -> int:
        """The number of entities in the simulator."""
        return len(self._entities)

    @property
    def entities(self):
        """The list of entities in the simulator."""
        return self._entities

    @property
    def substeps_local(self):
        """The number of substeps stored in local memory."""
        return self._substeps_local

    @property
    def cur_substep_global(self):
        """The current substep of the simulation."""
        return self._cur_substep_global

    @property
    def cur_substep_local(self):
        """The current substep of the simulation in local memory."""
        return self.f_global_to_f_local(self._cur_substep_global)

    @property
    def cur_step_local(self):
        """The current step of the simulation in local memory."""
        return self.f_global_to_s_local(self._cur_substep_global)

    @property
    def cur_step_global(self):
        """Number of `scene.step()` calls, counted for the whole batch.

        Use it to tell that the simulation moved on, for instance to invalidate a cache. For the simulated time of an
        environment, use `get_time`.
        """
        return self.f_global_to_s_global(self._cur_substep_global)

    def get_time(self, envs_idx=None):
        """The simulated time of each environment, in seconds.

        Environments are stepped and reset independently, so simulated time is per environment, and this is where it
        is read from.
        """
        time = self._steps[indices_to_mask(envs_idx)] * self._dt
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
