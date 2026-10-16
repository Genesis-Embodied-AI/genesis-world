"""Rigid participant of the Integrator, created for every IntegratorEngine whose scene contains a rigid entity.

Every Action function follows `function(*bound_args, *call_args)`: Rigid binds its own Data, and the Integrator
supplies `segment`.
"""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.core import Data, Require, System, register_system
from genesis.utils.array_class import V

from integrator_engine import Integrator, IntegratorEngine


@qd.data_oriented
class RigidState(Data):
    dofs_pos: qd.Tensor
    dofs_vel: qd.Tensor


def init_rigid(rigid_state: RigidState, scene: gs.Scene):
    """Allocate the DOF state of every rigid entity of the scene, at rest at zero with increasing velocities."""
    n_dofs = scene.sim.rigid_solver.n_dofs
    rigid_state.dofs_pos = V(dtype=gs.qd_float, shape=(n_dofs,))
    rigid_state.dofs_vel = V(dtype=gs.qd_float, shape=(n_dofs,))
    rigid_state.dofs_pos.fill(0.0)
    rigid_state.dofs_vel.from_numpy(np.arange(n_dofs, dtype=gs.np_float))


@qd.func
def func_count_rigid_dofs(rigid_state: qd.template(), segment: qd.template()):
    segment.set_n_dofs(rigid_state.dofs_pos.shape[0])


@qd.func(requires_top_level=True)
def func_gather_rigid_dofs(rigid_state: qd.template(), segment: qd.template()):
    for i_d_ in range(rigid_state.dofs_pos.shape[0]):
        segment.set_dof_position(i_d_, rigid_state.dofs_pos[i_d_])
        segment.set_dof_velocity(i_d_, rigid_state.dofs_vel[i_d_])


@qd.func(requires_top_level=True)
def func_scatter_rigid_dofs(rigid_state: qd.template(), segment: qd.template()):
    for i_d_ in range(rigid_state.dofs_pos.shape[0]):
        rigid_state.dofs_pos[i_d_] = segment.get_dof_position(i_d_)


@qd.data_oriented
class Rigid(System):
    """Participant holding the DOFs of the rigid entities of the scene."""

    integrator = Require(Integrator)

    def __init__(self, scene: gs.Scene) -> None:
        super().__init__(scene)
        self.state = RigidState()

    @System.action(kind="host")
    def init(self):
        return init_rigid, self.state, self._scene

    @System.action(kind="inline")
    def count(self):
        return func_count_rigid_dofs, self.state

    @System.action(kind="stage")
    def gather(self):
        return func_gather_rigid_dofs, self.state

    @System.action(kind="stage")
    def scatter(self):
        return func_scatter_rigid_dofs, self.state

    # rank=0 places the rigid DOFs before the cloth (rank=1) in the Integrator, whatever the add order
    def build(self):
        self.integrator.on_integrate(init=self.init, count=self.count, gather=self.gather, scatter=self.scatter, rank=0)


@register_system(IntegratorEngine)
def create_rigid(scene: gs.Scene) -> Rigid | None:
    """Create the Rigid System whenever the scene contains a rigid entity."""
    return Rigid(scene) if scene.sim.rigid_solver.is_active else None
