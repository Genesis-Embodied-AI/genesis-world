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
    material : gs.materials.Liquid
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
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        particle_size,
        idx,
        particle_start,
        name: str | None = None,
        *,
        options,
    ):
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            need_skinning=False,
            name=name,
            options=options,
        )

    def _add_particles_to_solver(self):
        material_params = self._options.material_params
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.density,
            material_params.stiffness,
            self._options.solver_options.exponent,
            material_params.viscosity,
            material_params.surface_tension,
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
