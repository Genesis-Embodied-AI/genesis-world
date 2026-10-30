from genesis.engine.entities.particle_entity import ParticleEntity


class SFParticleEntity(ParticleEntity):
    """
    PBD-based entity represented solely by particles.
    """

    def __init__(
        self, scene, solver, material, morph, surface, particle_size, idx, particle_start, name: str | None = None
    ):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False, name=name
        )

    def _add_particles_to_solver(self):
        # Nothing to add here because particles in SF are purely for visualization
        pass

    def sample(self):
        pass

    def update_particles(self, particles):
        self._particles = particles
        self._n_particles = len(particles)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        return f"sf_{super()._get_morph_identifier()}"
