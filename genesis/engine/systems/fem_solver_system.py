"""System wrapping the FEM solver, whose phases the Substepper calls.

The FEM solver keeps owning its data and running its own kernels; the System hands its phases to the Substepper as host
Actions.
"""

from typing import TYPE_CHECKING

from genesis.engine.core import System, register_system

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# FIXME: thin wrapper. The FEM solver still owns its data and runs each phase as a host method launching its own
# kernels. The System owns that data and exposes its phases as Actions over it, the solver forwarding to it, once the
# solver is inverted as the rigid solver is (see RigidSolverSystem).
class FEMSolverSystem(System):
    """System wrapping the FEM solver of the scene."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.fem_solver

    @System.action(kind="host")
    def process_input(self):
        return self.solver.process_input

    @System.action(kind="host")
    def pre_coupling(self):
        return self.solver.substep_pre_coupling

    @System.action(kind="host")
    def post_coupling(self):
        return self.solver.substep_post_coupling


@register_system(SimulatorEngine)
def create_fem_solver_system(scene: "Scene") -> FEMSolverSystem | None:
    """Create the System wrapping the FEM solver whenever it simulates an entity."""
    return FEMSolverSystem(scene) if scene.sim.fem_solver.is_active else None
