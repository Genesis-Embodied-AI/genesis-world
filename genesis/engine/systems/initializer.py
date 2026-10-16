"""Manager System that initializes the data of the Systems once the solvers have built their entities."""

from typing import TYPE_CHECKING

from genesis.engine.core import Find, Require, System, register_system

from .constraint_solver_system import ConstraintSolverSystem
from .kinematic_solver_system import KinematicSolverSystem
from .mpm_solver_system import MPMSolverSystem
from .rigid_rigid_contact_system import RigidRigidContactSystem
from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class Initializer(System):
    """Manager of the initialization, which invokes the initialize Action of every System in the order initialize
    states, once the Engine is built and before the coupler reads the data of the solvers."""

    kinematic = Require(KinematicSolverSystem)
    rigid_sdf = Require(RigidSdfSystem)
    rigid = Require(RigidSolverSystem)
    rigid_rigid_contact = Find(RigidRigidContactSystem)
    constraint = Find(ConstraintSolverSystem)
    mpm = Find(MPMSolverSystem)

    def initialize(self) -> None:
        """Initialize the data of every System."""
        # The kinematic data depends on no other System
        self.kinematic.initialize.invoke()
        # The signed distance fields of the rigid geoms are sized from the rigid model, and the collision detection
        # that the rigid data initialization constructs embeds them, so they are allocated in between
        self.rigid.initialize_model.invoke()
        self.rigid_sdf.initialize.invoke()
        self.rigid.initialize.invoke()
        # The rigid-rigid contact and the constraint solve record the configuration of the collider and of the
        # constraint solver that the rigid initialization constructs
        if self.rigid_rigid_contact is not None:
            self.rigid_rigid_contact.initialize.invoke()
        if self.constraint is not None:
            self.constraint.initialize.invoke()
        # The MPM solver fills the signed distance fields against the static configuration of the rigid data
        if self.mpm is not None:
            self.mpm.initialize.invoke()


@register_system(SimulatorEngine)
def create_initializer(scene: "Scene") -> Initializer:
    """Create the manager of the initialization, which every scene has."""
    return Initializer(scene)
