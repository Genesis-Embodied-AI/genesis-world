"""System wrapping the coupler of the scene, whose phases the Substepper calls."""

from typing import TYPE_CHECKING

from genesis.engine.core import System, register_system

from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


# FIXME: thin wrapper. The coupler still owns its data and runs its preprocess and its coupling as host methods over the
# data of every solver. Each coupling becomes a System defined between the two Systems it couples, joining them through
# their protocols, once the solvers it reads are inverted.
class CouplerSystem(System):
    """System wrapping the coupler of the scene, whichever coupler the scene options select."""

    def __init__(self, scene: "Scene") -> None:
        super().__init__(scene)
        self.coupler = scene.sim.coupler

    @System.action(kind="host")
    def preprocess(self):
        return self.coupler.preprocess

    @System.action(kind="host")
    def couple(self):
        return self.coupler.couple


@register_system(SimulatorEngine)
def create_coupler_system(scene: "Scene") -> CouplerSystem:
    """Create the System wrapping the coupler, which every scene stepped by the engine has."""
    return CouplerSystem(scene)
