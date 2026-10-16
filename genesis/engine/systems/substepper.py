"""Manager System that runs the phases of the solvers and of the coupler in the order of one simulator substep."""

from typing import TYPE_CHECKING

from genesis.engine.core import Find, Require, System, register_system

from .coupler_system import CouplerSystem
from .fem_solver_system import FEMSolverSystem
from .kinematic_solver_system import KinematicSolverSystem
from .mpm_solver_system import MPMSolverSystem
from .pbd_solver_system import PBDSolverSystem
from .rigid_solver_system import RigidSolverSystem
from .sf_solver_system import SFSolverSystem
from .simulator_engine import SimulatorEngine
from .sph_solver_system import SPHSolverSystem
from .tool_solver_system import ToolSolverSystem

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class Substepper(System):
    """Manager of one substep: the preprocess of the coupler, every pre-coupling phase, the coupling, then every
    post-coupling phase, each in the order substep states.

    A solver System takes part when it has entities to advance: the rigid and kinematic Systems own their data whether
    or not their solver holds an entity, the other solver Systems only exist when theirs does.
    """

    tool = Find(ToolSolverSystem)
    rigid = Require(RigidSolverSystem)
    kinematic = Require(KinematicSolverSystem)
    mpm = Find(MPMSolverSystem)
    sph = Find(SPHSolverSystem)
    pbd = Find(PBDSolverSystem)
    fem = Find(FEMSolverSystem)
    sf = Find(SFSolverSystem)
    coupler = Require(CouplerSystem)

    def process_input(self, in_backward: bool) -> None:
        """Apply the external commands of one step to every solver, each of which applies its own."""
        if self.tool is not None:
            self.tool.process_input.invoke((in_backward,))
        if self.rigid.is_active:
            self.rigid.process_input.invoke((in_backward,))
        if self.kinematic.is_active:
            self.kinematic.process_input.invoke((in_backward,))
        if self.mpm is not None:
            self.mpm.process_input.invoke((in_backward,))
        if self.sph is not None:
            self.sph.process_input.invoke((in_backward,))
        if self.pbd is not None:
            self.pbd.process_input.invoke((in_backward,))
        if self.fem is not None:
            self.fem.process_input.invoke((in_backward,))
        if self.sf is not None:
            self.sf.process_input.invoke((in_backward,))

    # FIXME: the substep runs on the host, since the phases it calls launch their own kernels around host logic. The
    # DFSPH and implicit FEM iterations read their convergence back on the host every iteration, the rigid constraint
    # solve selects its kernels on the host and one of them is a graph itself, the SAP coupler iterates Newton, PCG and
    # line search on the host, and the phases of the legacy coupler are kernels rather than funcs. The substep becomes
    # one graph once every phase is a stage Action whose loops run as device-side graph loops, and once quadrants#966
    # lets such a graph use fastcache.
    def substep(self, f: int) -> None:
        """Advance every solver by substep f. A scene simulating rigid entities alone advances through
        SimulatorEngine.rigid_substep_pipeline instead."""
        self.coupler.preprocess.invoke((f,))

        if self.tool is not None:
            self.tool.pre_coupling.invoke((f,))
        if self.rigid.is_active:
            self.rigid.pre_coupling.invoke((f,))
        if self.kinematic.is_active:
            self.kinematic.pre_coupling.invoke((f,))
        # The p2g of the MPM solver reads the rigid geoms and links that the rigid substep has just moved
        if self.mpm is not None:
            self.mpm.pre_coupling.invoke((f,))
        if self.sph is not None:
            self.sph.pre_coupling.invoke((f,))
        if self.pbd is not None:
            self.pbd.pre_coupling.invoke((f,))
        if self.fem is not None:
            self.fem.pre_coupling.invoke((f,))
        if self.sf is not None:
            self.sf.pre_coupling.invoke((f,))

        self.coupler.couple.invoke((f,))

        if self.tool is not None:
            self.tool.post_coupling.invoke((f,))
        if self.rigid.is_active:
            self.rigid.post_coupling.invoke((f,))
        if self.kinematic.is_active:
            self.kinematic.post_coupling.invoke((f,))
        # The g2p of the MPM solver reads the rigid links that the rigid post-coupling phase has completed
        if self.mpm is not None:
            self.mpm.post_coupling.invoke((f,))
        if self.sph is not None:
            self.sph.post_coupling.invoke((f,))
        if self.pbd is not None:
            self.pbd.post_coupling.invoke((f,))
        if self.fem is not None:
            self.fem.post_coupling.invoke((f,))
        if self.sf is not None:
            self.sf.post_coupling.invoke((f,))


@register_system(SimulatorEngine)
def create_substepper(scene: "Scene") -> Substepper:
    """Create the manager of the substep, which every scene stepped by the engine has."""
    return Substepper(scene)
