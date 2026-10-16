"""System wrapping the stable fluid solver, whose phases the Substepper calls.

The stable fluid solver keeps owning its data and running its own kernels; the System hands its phases to the Substepper
as host Actions.
"""

from typing import TYPE_CHECKING

from genesis.engine.core import System, register_system

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# FIXME: thin wrapper. The stable fluid solver still owns its data and runs each phase as a host method launching its
# own kernels. The System owns that data and exposes its phases as Actions over it, the solver forwarding to it, once
# the solver is inverted as the rigid solver is (see RigidSolverSystem).
class SFSolverSystem(System):
    """System wrapping the stable fluid solver of the scene."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.solver = scene.sim.sf_solver

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
def create_sf_solver_system(scene: "Scene") -> SFSolverSystem | None:
    """Create the System wrapping the stable fluid solver whenever it simulates an entity."""
    return SFSolverSystem(scene) if scene.sim.sf_solver.is_active else None
