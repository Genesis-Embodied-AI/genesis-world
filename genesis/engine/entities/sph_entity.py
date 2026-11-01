import quadrants as qd

import genesis as gs
from genesis.engine.states.entities import SPHEntityState

from .particle_entity import ParticleEntity


@qd.data_oriented
class SPHEntity(ParticleEntity):
    """
    SPH-based particle entity.

    Parameters
    ----------
    scene : Scene
        The simulation scene.
    solver : Solver
        The solver handling the simulation logic.
    material : Material
        Material properties (e.g., density, stiffness).
    morph : Morph
        Morphological configuration.
    surface : Surface
        Surface constraints or geometry.
    particle_size : float
        The size of each particle.
    idx : int
        Index of this entity in the scene.
    particle_start : int
        Start index for the particles belonging to this entity.
    """

    def __init__(
        self, scene, solver, material, morph, surface, particle_size, idx, particle_start, name: str | None = None
    ):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False, name=name
        )

    def init_sampler(self):
        """
        Initialize the particle sampler based on the material's sampling method.

        Raises
        ------
        GenesisException
            If the sampler is not one of the supported types: 'regular', 'pbs', or 'pbs-sdf_res'.
        """
        self.sampler = self._material.sampler

        match self.sampler.split("-"):
            case ["regular"]:
                pass
            case ["random"]:
                pass
            case ["pbs"]:
                # using default sdf_res=32
                self.sampler += "-32"
            case ["pbs", num] if num.isnumeric():
                pass
            case _:
                gs.raise_exception(
                    "Only one of the following samplers is supported: [`regular`, `random`, `pbs`, `pbs-sdf_res`]. "
                    f"Got: {self.sampler}."
                )

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.rho,
            self._material.stiffness,
            self._material.exponent,
            self._material.mu,
            self._material.gamma,
            self._particles,
        )

    @gs.assert_built
    def get_state(self):
        """
        Get the current state of the SPHEntity, a view of the state of the solver.

        Returns
        -------
        state : SPHEntityState
            The current particle state for the entity.
        """
        solver_state = self._solver.get_state(self._sim.cur_substep_local)
        particles_slice = slice(self._particle_start, self._particle_start + self._n_particles)
        return SPHEntityState(
            self, self._sim.cur_step_global, solver_state.pos[:, particles_slice], solver_state.vel[:, particles_slice]
        )

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    def get_position(self, envs_idx=None):
        return self.get_particles_pos(envs_idx)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        return f"sph_{super()._get_morph_identifier()}"
