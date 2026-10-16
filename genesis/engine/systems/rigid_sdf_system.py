"""System of the signed distance fields of the rigid geoms."""

from typing import TYPE_CHECKING

from genesis.engine.core import System, register_system
from genesis.utils.sdf import SDF

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

    from .rigid_solver_system import RigidSolverSystem


class RigidSdfSystem(System):
    """System owning the signed distance fields of the rigid geoms, a geometric description of each geom that does not
    depend on what queries it.

    The rigid-rigid contact queries them for non-convex geoms, through the collider info that embeds them. The couplings
    of particles with rigid geoms, the CPIC of the MPM solver and the tactile sensors query them too. The fields are
    allocated with the rigid data whether or not anything queries them, and filled by activate, which every reader
    calls from its own initialization once it knows it needs them.
    """

    @System.action(kind="host")
    def initialize(self):
        return self.allocate

    def build(self):
        # The rigid data embeds these fields in its collision detection, so its module imports this one
        from .rigid_solver_system import RigidSolverSystem

        self.rigid_solver_system: "RigidSolverSystem" = self.require(RigidSolverSystem)

    def allocate(self) -> None:
        """Allocate the fields from the model of the rigid geoms, ahead of the collision detection that embeds
        them."""
        self.sdf = SDF(self.rigid_solver_system)

    def activate(self) -> None:
        """Fill the fields, once a reader needs them. Idempotent."""
        self.sdf.activate()


@register_system(SimulatorEngine)
def create_rigid_sdf_system(scene: "Scene") -> RigidSdfSystem:
    """Create the System owning the signed distance fields of the rigid geoms, which every scene has."""
    return RigidSdfSystem(scene)
