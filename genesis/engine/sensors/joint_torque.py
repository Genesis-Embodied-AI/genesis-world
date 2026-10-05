import torch

import genesis as gs
from genesis.options.sensors import JointTorque as JointTorqueOptions

from .base_sensor import SimpleSensor, SimpleSensorArray


class JointTorqueSensorArray(SimpleSensorArray[JointTorqueOptions]):
    """Array of every joint torque sensor of the scene."""

    def add_sensor(self, sensor: "JointTorqueSensor") -> None:
        # A sensor naming no DOF measures every DOF of its entity, which fixes its return format
        options = sensor.options
        if options.dofs_idx_local is None:
            options.dofs_idx_local = tuple(range(self._sim.entities[options.entity_idx].n_dofs))
        super().add_sensor(sensor)

    def build(self) -> None:
        super().build()

        # Joint torque sensors are attached to an entity rather than a link, so the rigid solver is bound here
        self.solver = self._sim.rigid_solver
        # Global DOF indices concatenated across all sensors in scene order
        dofs_idx = [
            self._sim.entities[sensor.options.entity_idx].dof_start + i_d
            for sensor in self._sensors
            for i_d in sensor.options.dofs_idx_local
        ]
        self.dofs_idx = torch.tensor(dofs_idx, dtype=gs.tc_int, device=gs.device)

    def _get_return_format(self, options: JointTorqueOptions) -> tuple[int, ...]:
        return (len(options.dofs_idx_local),)

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def _update_raw_data(self, raw_data: torch.Tensor) -> None:
        actuator_force = self.solver.get_dofs_actuator_force(self.dofs_idx)
        if self.solver.n_envs == 0:
            actuator_force = actuator_force[None]
        raw_data.copy_(actuator_force)


class JointTorqueSensor(SimpleSensor[JointTorqueOptions, JointTorqueSensorArray]):
    """
    Measures the generalized effort transmitted from each actuator to its joint output shaft (torque for revolute
    DOFs, force for prismatic DOFs).

    The reading is the commanded actuator effort minus the gearbox losses, derived from Newton's 3rd law at the
    gearbox interface:

        actuator_force = tau_control - armature * qacc + tau_frictionloss + tau_damping

    where ``tau_damping = -damping * vel`` is the viscous passive effort. Gravity, Coriolis and contact loads are
    captured implicitly through the constraint-solved acceleration ``qacc``.
    """
