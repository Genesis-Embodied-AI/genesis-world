"""Engine of the simulator, which steps the legacy solvers and coupler through Systems."""

from genesis.engine.core import Engine, Pipeline


class SimulatorEngine(Engine):
    """The engine of one simulator, which runs the substep phases of its solvers and of its coupler.

    Each System registers the function that creates it for this Engine from its own module (see register_system). The
    simulator keeps the step loop, the clock and the checkpoints, and launches one pipeline per entry point.
    """

    def build(self):
        super().build()
        # The Systems import this module to register, so they are imported here rather than declared on the class
        from .initializer import Initializer
        from .rigid_solver_system import RigidSolverSystem
        from .substepper import Substepper

        self.init_pipeline = Pipeline(self.require(Initializer).initialize)
        substepper = self.require(Substepper)
        self.input_pipeline = Pipeline(substepper.process_input)
        self.substep_pipeline = Pipeline(substepper.substep)
        # A scene simulating rigid entities alone advances the rigid solver directly, skipping the inputs, which only
        # feed the differentiable tape, and the coupler, which has nothing to exchange
        self.rigid_substep_pipeline = Pipeline(self.require(RigidSolverSystem).substep)
