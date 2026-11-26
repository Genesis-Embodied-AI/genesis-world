from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
from genesis.options.surfaces import Surface

from .particle_entity import ParticleEntity

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.sph_solver import SPHSolver


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
    spec : gs.entities.SPH
        The entity spec the entity is constructed from, whose morph gives its geometry.
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
        scene: "Scene",
        solver: "SPHSolver",
        surface: Surface,
        particle_size: float,
        idx: int,
        particle_start: int,
        spec: "gs.entities.SPH",
    ) -> None:
        super().__init__(
            scene,
            solver,
            surface,
            particle_size,
            idx,
            particle_start,
            need_skinning=False,
            spec=spec,
        )

    def _add_particles_to_solver(self):
        self._solver.add_particles(self.active, self._particle_start, self.material, self._particles)

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
