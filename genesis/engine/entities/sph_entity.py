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
    options : gs.entities.SPH
        The entity options the entity is constructed from, whose morph gives its geometry.
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
        options: "gs.entities.SPH",
    ) -> None:
        super().__init__(
            scene,
            solver,
            surface,
            particle_size,
            idx,
            particle_start,
            need_skinning=False,
            options=options,
        )

    def init_sampler(self):
        """
        Initialize the particle sampler based on the material's sampling method.

        Raises
        ------
        GenesisException
            If the sampler is not one of the supported types: 'regular', 'pbs', or 'pbs-sdf_res'.
        """
        self.sampler = self.material.sampler

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
