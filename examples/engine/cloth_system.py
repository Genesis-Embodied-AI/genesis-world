"""Cloth participant of the Integrator, whose vertices each carry three DOFs.

Every Action function follows `function(*bound_args, *call_args)`: Cloth binds its own Data, and the Integrator
supplies `segment`.
"""

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.core import Find, Require, System
from genesis.utils.array_class import V_VEC

from integrator_engine import Integrator
from rigid_system import Rigid


@qd.data_oriented
class ClothState:
    verts_pos: qd.Tensor
    verts_vel: qd.Tensor


def init_cloth(cloth_state: ClothState):
    """Allocate the state of two vertices, moving along x at 10 m/s from x=100 and x=200."""
    cloth_state.verts_pos = V_VEC(3, dtype=gs.qd_float, shape=(2,))
    cloth_state.verts_vel = V_VEC(3, dtype=gs.qd_float, shape=(2,))
    cloth_state.verts_pos.from_numpy(np.array(((100.0, 0.0, 0.0), (200.0, 0.0, 0.0)), dtype=gs.np_float))
    cloth_state.verts_vel.from_numpy(np.array(((10.0, 0.0, 0.0), (10.0, 0.0, 0.0)), dtype=gs.np_float))


# The cloth maps vertex i_v to the three DOFs 3 * i_v + j of its range, so the Segment stays a plain DOF range
@qd.func
def func_count_cloth_dofs(cloth_state: qd.template(), segment: qd.template()):
    segment.set_n_dofs(3 * cloth_state.verts_pos.shape[0])


@qd.func(requires_top_level=True)
def func_gather_cloth_dofs(cloth_state: qd.template(), segment: qd.template()):
    for i_v in range(cloth_state.verts_pos.shape[0]):
        for j in qd.static(range(3)):
            segment.set_dof_position(3 * i_v + j, cloth_state.verts_pos[i_v][j])
            segment.set_dof_velocity(3 * i_v + j, cloth_state.verts_vel[i_v][j])


@qd.func(requires_top_level=True)
def func_scatter_cloth_dofs(cloth_state: qd.template(), segment: qd.template()):
    for i_v in range(cloth_state.verts_pos.shape[0]):
        for j in qd.static(range(3)):
            cloth_state.verts_pos[i_v][j] = segment.get_dof_position(3 * i_v + j)


@qd.data_oriented
class Cloth(System):
    """Participant whose vertices each carry three DOFs, and which finds the rigid DOFs when the scene has any."""

    integrator = Require(Integrator)
    rigid = Find(Rigid)

    def __init__(self, scene: gs.Scene) -> None:
        super().__init__(scene)
        self.state = ClothState()

    @System.action(kind="host")
    def init(self):
        return init_cloth, self.state

    @System.action(kind="inline")
    def count(self):
        return func_count_cloth_dofs, self.state

    @System.action(kind="stage")
    def gather(self):
        return func_gather_cloth_dofs, self.state

    @System.action(kind="stage")
    def scatter(self):
        return func_scatter_cloth_dofs, self.state

    def build(self):
        self.integrator.on_integrate(init=self.init, count=self.count, gather=self.gather, scatter=self.scatter, rank=1)
