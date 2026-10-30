import dataclasses

import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.utils.array_class import DataKind


@dataclasses.dataclass
class SolverCheckpoint:
    """The arrays of one solver of the given kinds (see 'DataKind'), for 'Solver.__setstate__' to write back.

    A checkpoint '__getstate__' reads holds every kind a step reads from one step to the next (see 'CHECKPOINT_KINDS'),
    so stepping on from it lands where the original went. A trajectory frame may hold the state alone, and writing it
    back recomputes what it leaves out. Each array stands under the name 'Solver.data' gives it, copied where it lives:
    on the device as a torch tensor when zero-copy views exist, as a numpy array otherwise. The static configs come
    along as provenance and are never written back.
    """

    arrays: dict[str, np.ndarray | torch.Tensor]
    configs: dict[str, object]
    kinds: frozenset[DataKind]


@dataclasses.dataclass
class SimulatorCheckpoint:
    """The state a simulation stands in.

    How many steps each environment has run, and what every active solver reads from one step to the next, under the
    name of its class.
    """

    steps: np.ndarray | torch.Tensor
    solvers: dict[str, SolverCheckpoint]


class SimState(RBC):
    """
    Dynamic state queried from a Scene's Simulator.
    """

    def __init__(
        self,
        scene,
        s_global,
        f_local,
        solvers,
    ):
        self._scene = scene
        self._s_global = s_global
        self._solvers_state = list()
        for solver in solvers:
            self._solvers_state.append(solver.get_state(f_local))

    @property
    def scene(self):
        return self._scene

    @property
    def s_global(self):
        return self._s_global

    @property
    def solvers_state(self):
        return self._solvers_state

    def __iter__(self):
        return iter(self._solvers_state)


class KinematicSolverState:
    """
    Dynamic state queried from a KinematicSolver.

    Only stores position-related fields (qpos, link poses). Physics fields
    (velocity, acceleration, mass, friction) are omitted since kinematic entities have no dynamics.
    """

    def __init__(self, scene, s_global):
        self.scene = scene
        self._s_global = s_global

        _B = scene.sim.kinematic_solver._B
        self.qpos = torch.zeros((_B, scene.sim.kinematic_solver.n_qs), dtype=gs.tc_float, device=gs.device)
        self.dofs_vel = torch.zeros((_B, scene.sim.kinematic_solver.n_dofs), dtype=gs.tc_float, device=gs.device)
        self.links_pos = torch.zeros((_B, scene.sim.kinematic_solver.n_links, 3), dtype=gs.tc_float, device=gs.device)
        self.links_quat = torch.zeros((_B, scene.sim.kinematic_solver.n_links, 4), dtype=gs.tc_float, device=gs.device)

    @property
    def s_global(self):
        return self._s_global


class RigidSolverState:
    """
    Dynamic state queried from a RigidSolver.
    """

    def __init__(self, scene, s_global, qpos, dofs_vel, dofs_acc, links_pos, links_quat, friction_ratio):
        self.scene = scene
        self._s_global = s_global
        self.qpos = qpos
        self.dofs_vel = dofs_vel
        self.dofs_acc = dofs_acc
        self.links_pos = links_pos
        self.links_quat = links_quat
        self.friction_ratio = friction_ratio

    @property
    def s_global(self):
        return self._s_global


class MPMSolverState(RBC):
    """
    Dynamic state queried from a MPMSolver.
    """

    def __init__(self, scene, pos, vel, C, F, Jp, actu, active):
        self._scene = scene
        self._pos = pos
        self._vel = vel
        self._C = C
        self._F = F
        self._Jp = Jp
        self._actu = actu
        self._active = active

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def Jp(self):
        return self._Jp

    @property
    def actu(self):
        return self._actu

    @property
    def active(self):
        return self._active


class SPHSolverState:
    """
    Dynamic state queried from a SPHSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        self._pos = torch.zeros(
            (scene.sim._B, scene.sim.sph_solver.n_particles, 3), dtype=gs.tc_float, device=gs.device
        )
        self._vel = torch.zeros(
            (scene.sim._B, scene.sim.sph_solver.n_particles, 3), dtype=gs.tc_float, device=gs.device
        )
        self._active = torch.zeros((scene.sim._B, scene.sim.sph_solver.n_particles), dtype=gs.tc_bool, device=gs.device)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class PBDSolverState:
    """
    Dynamic state queried from a PBDSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        self._pos = torch.zeros(
            (scene.sim._B, scene.sim.pbd_solver.n_particles, 3), dtype=gs.tc_float, device=gs.device
        )
        self._vel = torch.zeros(
            (scene.sim._B, scene.sim.pbd_solver.n_particles, 3), dtype=gs.tc_float, device=gs.device
        )
        self._free = torch.zeros((scene.sim._B, scene.sim.pbd_solver.n_particles), dtype=gs.tc_bool, device=gs.device)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def free(self):
        return self._free


class FEMSolverState:
    def __init__(self, scene, pos, vel):
        self._scene = scene
        self._pos = pos
        self._vel = vel

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel
