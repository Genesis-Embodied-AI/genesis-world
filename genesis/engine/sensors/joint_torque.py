from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.options.sensors import JointTorque as JointTorqueOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field

from .base_sensor import SimpleSensor, SimpleSensorMetadata

if TYPE_CHECKING:
    from .sensor_manager import SensorManager


@dataclass
class JointTorqueSensorMetadata(SimpleSensorMetadata):
    """Shared state for all JointTorqueSensor instances in a scene."""

    solver: "gs.RigidSolver | None" = None
    # Global DOF indices concatenated across all sensor instances in scene order.
    dofs_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)


class JointTorqueSensor(SimpleSensor[JointTorqueOptions, None, JointTorqueSensorMetadata]):
    """
    Measures the generalized effort transmitted from each actuator to its joint output shaft (torque for revolute
    DOFs, force for prismatic DOFs).

    The reading is the commanded actuator effort minus the gearbox losses, derived from Newton's 3rd law at the
    gearbox interface:

        actuator_force = tau_control - armature * qacc + tau_frictionloss + tau_damping

    where ``tau_damping = -damping * vel`` is the viscous passive effort. Gravity, Coriolis and contact loads are
    captured implicitly through the constraint-solved acceleration ``qacc``.
    """

    def __init__(
        self,
        options: JointTorqueOptions,
        idx: int,
        shared_context: None,
        shared_metadata: JointTorqueSensorMetadata,
        manager: "SensorManager",
    ):
        # Resolve dofs_idx_local=None to the full DOF range before super().__init__(), because _get_return_format() is
        # called inside the base constructor and needs the final DOF count.
        if options.dofs_idx_local is None:
            entity = manager._sim.entities[options.entity_idx]
            options.dofs_idx_local = tuple(range(entity.n_dofs))
        super().__init__(options, idx, shared_context, shared_metadata, manager)

    def _get_return_format(self) -> tuple[int, ...]:
        return (len(self._options.dofs_idx_local),)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        entity = self._manager._sim.entities[self._options.entity_idx]
        dofs_idx = torch.tensor(
            [entity.dof_start + i for i in self._options.dofs_idx_local],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self._shared_metadata.dofs_idx = concat_with_tensor(self._shared_metadata.dofs_idx, dofs_idx, dim=0)

    @classmethod
    def _update_raw_data(
        cls,
        shared_context: None,
        shared_metadata: JointTorqueSensorMetadata,
        raw_data_T: torch.Tensor,  # shape (n_sensor_dofs, B)
    ):
        solver = shared_metadata.solver
        actuator_force = solver.get_dofs_actuator_force(shared_metadata.dofs_idx)
        if solver.n_envs == 0:
            actuator_force = actuator_force[None]
        raw_data_T.copy_(actuator_force.T)
