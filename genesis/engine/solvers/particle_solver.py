import quadrants as qd

import genesis as gs
from genesis.engine.state_change import StateChange, mutates

from .base_solver import GravityMixin, Solver, TimeBasedMixin


class ParticleSolver(GravityMixin, TimeBasedMixin, Solver):
    """Base class of the particle-based solvers, holding what they share: the accessors of the state of the particles.

    The accessors take global particle indices, given once for every environment or one row per environment (see
    '_sanitize_io_variables'), and the kernels of each solver read and write the state the particles hold between two
    steps.
    """

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_particles_pos(self, poss, particles_idx=None, envs_idx=None):
        """
        Set the position of some particles, at rest.

        Parameters
        ----------
        poss : array_like, shape ([M,] [N,] 3)
            Target position of each particle.
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles. If None, all particles will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        poss, particles_idx, envs_idx = self._sanitize_io_variables(
            poss, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_set_particles_pos(particles_idx, envs_idx, poss)
        # Overwriting the positions of the particles clears the errors their former state flagged
        self.clear_errno(envs_idx)

    def get_particles_pos(self, particles_idx=None, envs_idx=None):
        """Get the position of some particles (see 'set_particles_pos')."""
        poss, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_get_particles_pos(particles_idx, envs_idx, poss)
        return poss[0] if self._scene.n_envs == 0 else poss

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_particles_vel(self, vels, particles_idx=None, envs_idx=None):
        """Set the velocity of some particles (see 'set_particles_pos')."""
        vels, particles_idx, envs_idx = self._sanitize_io_variables(
            vels, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_set_particles_vel(particles_idx, envs_idx, vels)

    def get_particles_vel(self, particles_idx=None, envs_idx=None):
        """Get the velocity of some particles (see 'set_particles_pos')."""
        vels, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_get_particles_vel(particles_idx, envs_idx, vels)
        return vels[0] if self._scene.n_envs == 0 else vels

    @gs.assert_built
    @mutates(StateChange.GEOMETRY, differentiable=False)
    def set_particles_active(self, actives, particles_idx=None, envs_idx=None):
        """Set whether some particles take part in the simulation (see 'set_particles_pos')."""
        actives, particles_idx, envs_idx = self._sanitize_io_variables(
            actives, particles_idx, self.n_particles, "particles_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        self._kernel_set_particles_active(particles_idx, envs_idx, actives)

    def get_particles_active(self, particles_idx=None, envs_idx=None):
        """Get whether some particles take part in the simulation (see 'set_particles_pos')."""
        actives, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        self._kernel_get_particles_active(particles_idx, envs_idx, actives)
        return actives[0] if self._scene.n_envs == 0 else actives

    def get_particles_mass(self, particles_idx=None, envs_idx=None):
        """Get the mass of some particles (see 'set_particles_pos')."""
        masses, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True
        )
        self._kernel_get_particles_mass(particles_idx, envs_idx, masses)
        return masses[0] if self._scene.n_envs == 0 else masses

    @qd.kernel
    def add_vverts(
        self,
        vvert_start: int,
        particle_start: int,
        vverts: qd.types.ndarray(element_dim=1),
        particles: qd.types.ndarray(element_dim=1),
        P_invs: qd.types.ndarray(element_dim=2),
        support_idxs_local: qd.types.ndarray(),
    ):
        """Bind the given vverts to their supporting particles, whose positions skin them for rendering."""
        for i_vv_ in range(vverts.shape[0]):
            i_vv = i_vv_ + vvert_start
            v = vverts[i_vv_]
            P_inv = P_invs[i_vv_]
            i_p_last = support_idxs_local[i_vv_, self._n_vvert_supports - 1]
            b = P_inv @ (v - particles[i_p_last])
            for j in range(self._n_vvert_supports - 1):
                self.vverts_info.support_weights[i_vv][j] = b[j]
            self.vverts_info.support_weights[i_vv][self._n_vvert_supports - 1] = 1.0 - b.sum()
            for j in range(self._n_vvert_supports):
                self.vverts_info.support_idxs[i_vv][j] = support_idxs_local[i_vv_, j] + particle_start
