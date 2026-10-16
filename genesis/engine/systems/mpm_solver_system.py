"""System wrapping the MPM solver, whose phases the Substepper calls.

The MPM solver keeps owning its data and running its own kernels; the System hands its phases to the Substepper as host
Actions.
"""

from typing import TYPE_CHECKING

from genesis.engine.core import Require, System, register_system

from .rigid_sdf_system import RigidSdfSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# FIXME: thin wrapper. The MPM solver still owns its data and runs each phase as a host method launching its own
# kernels. The System owns that data and exposes its phases as Actions over it, the solver forwarding to it, once the
# solver is inverted as the rigid solver is (see RigidSolverSystem).
class MPMSolverSystem(System):
    """System wrapping the MPM solver of the scene."""

    rigid_sdf_system = Require(RigidSdfSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.mpm_solver

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.solver.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.solver.substep_post_coupling

    @System.action(kind="host")
    def initialize(self):
        return self.activate_rigid_sdf

    def activate_rigid_sdf(self) -> None:
        """Activate the signed distance fields of the rigid geoms, by which CPIC separates the particles from them."""
        if self.solver._enable_CPIC and self.rigid_sdf_system.rigid_solver_system.n_geoms:
            self.rigid_sdf_system.activate()


@register_system(SimulatorEngine)
def create_mpm_solver_system(scene: "Scene") -> MPMSolverSystem | None:
    """Create the System wrapping the MPM solver whenever it simulates an entity."""
    return MPMSolverSystem(scene) if scene.sim.mpm_solver.is_active else None
