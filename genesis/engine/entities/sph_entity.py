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
    material : MaterialOptions
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
        self._solver._kernel_add_particles(
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

    def _reset_grad(self):
        pass

    def add_grad_from_state(self, state):
        """
        Apply gradients from a given state.

        Parameters
        ----------
        state : SPHEntityState
            The state from which to compute gradients.
        """
        pass

    @qd.kernel
    def get_frame(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
    ):
        """
        Retrieve particle positions and velocities for the given frame.

        Parameters
        ----------
        f : int
            Frame index.
        pos : ndarray
            Output array for positions (n_envs, n_particles, 3).
        vel : ndarray
            Output array for velocities (n_envs, n_particles, 3).
        """
        for i_p_, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_p = i_p_ + self._particle_start
            for j in qd.static(range(3)):
                pos[i_b, i_p_, j] = self.solver.particles[i_p, i_b].pos[j]
                vel[i_b, i_p_, j] = self.solver.particles[i_p, i_b].vel[j]

    @gs.assert_built
    def get_state(self):
        """
        Get the current state of the SPHEntity including positions, velocities, .

        Returns
        -------
        state : SPHEntityState
            The current particle state for the entity.
        """
        state = SPHEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel)

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        poss = self._sanitize_particles_tensor(poss, gs.tc_float, particles_idx, envs_idx, (3,))
        self.solver._kernel_set_particles_pos(particles_idx, envs_idx, poss)

    def get_particles_pos(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        poss = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self.solver._kernel_get_particles_pos(self._particle_start, self.n_particles, envs_idx, poss)
        if self._scene.n_envs == 0:
            poss = poss[0]
        return poss

    def get_position(self, envs_idx=None):
        return self.get_particles_pos(envs_idx)

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        vels = self._sanitize_particles_tensor(vels, gs.tc_float, particles_idx, envs_idx, (3,))
        self.solver._kernel_set_particles_vel(particles_idx, envs_idx, vels)

    def get_particles_vel(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vels = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self.solver._kernel_get_particles_vel(self._particle_start, self.n_particles, envs_idx, vels)
        if self._scene.n_envs == 0:
            vels = vels[0]
        return vels

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        actives = self._sanitize_particles_tensor(actives, gs.tc_bool, particles_idx, envs_idx)
        self.solver._kernel_set_particles_active(particles_idx, envs_idx, actives)

    def get_particles_active(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actives = self._sanitize_particles_tensor(None, gs.tc_bool, None, envs_idx)
        self.solver._kernel_get_particles_active(self._particle_start, self.n_particles, envs_idx, actives)
        if self._scene.n_envs == 0:
            actives = actives[0]
        return actives

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        return f"sph_{super()._get_morph_identifier()}"
