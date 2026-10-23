import functools

import torch

import quadrants as qd

import genesis as gs
from genesis.engine.states.entities import MPMEntityState
from genesis.utils.misc import drop_repeated_index, to_gs_tensor

from .particle_entity import ParticleEntity, assert_active


def assert_muscle(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self.material, gs.materials.MPM.Muscle):
            gs.raise_exception("This method is only supported by entities with 'MPM.Muscle' material.")
        return method(self, *args, **kwargs)

    return wrapper


@qd.data_oriented
class MPMEntity(ParticleEntity):
    """
    MPM-based particle entity.

    Parameters
    ----------
    scene : Scene
        Scene object this entity belongs to.
    solver : Solver
        The solver responsible for simulating this entity.
    material : Material
        Material used to determine physical behavior (e.g., Snow, Sand, Muscle).
    morph : Morph
        Shape description used for particle sampling.
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Particle size for discretization.
    idx : int
        Unique index of the entity.
    particle_start : int
        Starting particle index.
    vvert_start : int
        Start index for visual vertices (unused if no skinning).
    vface_start : int
        Start index for visual faces (unused if no skinning).
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
        vvert_start,
        vface_start,
        name: str | None = None,
    ):
        need_skinning = not isinstance(
            material, (gs.materials.MPM.Liquid, gs.materials.MPM.Sand, gs.materials.MPM.Snow)
        )
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            vvert_start,
            vface_start,
            need_skinning=need_skinning,
            name=name,
        )

    def _add_to_solver(self):
        super()._add_to_solver()
        if isinstance(self.material, gs.materials.MPM.Muscle) and isinstance(self._morph, gs.options.morphs.MeshSet):
            self.set_muscle_group(self.mesh_set_group_ids)

    def _add_particles_to_solver(self):
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.idx,
            self._material._default_Jp,
            self._material.rho,
            self._particles,
        )

    @qd.kernel
    def add_grad_from_pos(self, f: qd.i32, poss_grad: qd.types.ndarray()):
        """
        Accumulate gradients to particle positions at the given substep.

        Parameters
        ----------
        f : int
            Local substep index to update.
        poss_grad : ndarray
            Gradient of particle positions, shape (B, n_particles, 3).
        """
        for i_p, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in qd.static(range(3)):
                self._solver.particles.grad[f, i_global, i_b].pos[j] += poss_grad[i_b, i_p, j]

    @qd.kernel
    def add_grad_from_vel(self, f: qd.i32, vels_grad: qd.types.ndarray()):
        """
        Accumulate gradients to particle velocities at the given substep.

        Parameters
        ----------
        f : int
            Local substep index to update.
        vels_grad : ndarray
            Gradient of particle velocities, shape (B, n_particles, 3).
        """
        for i_p, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in qd.static(range(3)):
                self._solver.particles.grad[f, i_global, i_b].vel[j] += vels_grad[i_b, i_p, j]

    @qd.kernel
    def add_grad_from_C(self, f: qd.i32, C_grad: qd.types.ndarray()):
        """
        Accumulate gradients to affine matrices C at the given substep.

        Parameters
        ----------
        f : int
            Local substep index to update.
        C_grad : ndarray
            Gradient of C matrices, shape (B, n_particles, 3, 3).
        """
        for i_p, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in qd.static(range(3)):
                for k in qd.static(range(3)):
                    self._solver.particles.grad[f, i_global, i_b].C[j, k] += C_grad[i_b, i_p, j, k]

    @qd.kernel
    def add_grad_from_F(self, f: qd.i32, F_grad: qd.types.ndarray()):
        """
        Accumulate gradients to deformation gradients F at the given substep.

        Parameters
        ----------
        f : int
            Local substep index to update.
        F_grad : ndarray
            Gradient of F matrices, shape (B, n_particles, 3, 3).
        """
        for i_p, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in qd.static(range(3)):
                for k in qd.static(range(3)):
                    self._solver.particles.grad[f, i_global, i_b].F[j, k] += F_grad[i_b, i_p, j, k]

    @qd.kernel
    def add_grad_from_Jp(self, f: qd.i32, Jp_grad: qd.types.ndarray()):
        """
        Accumulate gradients to plastic volume ratios Jp at the given substep.

        Parameters
        ----------
        f : int
            Local substep index to query.
        Jp_grad : ndarray
            Gradient of Jp values, shape (B, n_particles).
        """
        for i_p, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            self._solver.particles.grad[f, i_global, i_b].Jp += Jp_grad[i_b, i_p]

    @gs.assert_built
    def get_state(self):
        """
        Get the current physical state of the particle entity.

        Returns
        -------
        state : MPMEntityState
            The current state of all physical properties of the entity.
        """
        # TODO: merge with self._solver.get_state?!
        state = MPMEntityState(self, self._sim.cur_step_global)
        self.get_frame(
            f=self._sim.cur_substep_local,
            pos=state.pos,
            vel=state.vel,
            C=state.C,
            F=state.F,
            Jp=state.Jp,
            active=state.active,
        )

        # The gradient of the state goes to the particles of this entity (see 'MPMSolver.add_grad_from_state')
        self._solver.record_read(state.grad_tensors, self)

        return state

    @qd.kernel
    def get_frame(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_particles, 3]
        vel: qd.types.ndarray(),  # shape [B, n_particles, 3]
        C: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: qd.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: qd.types.ndarray(),  # shape [B, n_particles]
        active: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        """
        Extract the state of particles at the given substep.

        Parameters
        ----------
        f : int
            Local substep index to query.
        pos : ndarray
            Particle positions, shape (B, n_particles, 3).
        vel : ndarray
            Particle velocities, shape (B, n_particles, 3).
        C : ndarray
            Affine matrix C, shape (B, n_particles, 3, 3).
        F : ndarray
            Deformation gradient F, shape (B, n_particles, 3, 3).
        Jp : ndarray
            Volume ratio, shape (B, n_particles).
        active : ndarray
            Particle activeness state, shape (B, n_particles).
        """
        for i_p, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            # Copy pos, vel
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self._solver.particles[f, i_global, i_b].pos[j]
                vel[i_b, i_p, j] = self._solver.particles[f, i_global, i_b].vel[j]
                # Copy C, F
                for k in qd.static(range(3)):
                    C[i_b, i_p, j, k] = self._solver.particles[f, i_global, i_b].C[j, k]
                    F[i_b, i_p, j, k] = self._solver.particles[f, i_global, i_b].F[j, k]
            # Copy Jp, active
            Jp[i_b, i_p] = self._solver.particles[f, i_global, i_b].Jp
            active[i_b, i_p] = self._solver.particles_ng[f, i_global, i_b].active

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_written_particles(self, tensor, dtype, particles_idx_local, envs_idx, element_shape=()):
        """
        Sanitize the particles a setter writes and the values written, as the global indices of the particles.

        An environment or a particle given several times keeps its last occurrence (see 'drop_repeated_index').
        """
        envs_idx_ = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx = self._sanitize_particles_idx_local(particles_idx_local, envs_idx_) + self._particle_start
        tensor = self._sanitize_particles_tensor(tensor, dtype, particles_idx, envs_idx_, element_shape)
        envs_idx_, (tensor, particles_idx) = drop_repeated_index(
            envs_idx, envs_idx_, (tensor, particles_idx), 0, self._scene.n_envs
        )
        particles_idx, (tensor,) = drop_repeated_index(
            particles_idx_local, particles_idx, (tensor,), 1, self._n_particles
        )
        return tensor, particles_idx, envs_idx_

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None):
        poss, particles_idx, envs_idx = self._sanitize_written_particles(
            poss, gs.tc_float, particles_idx_local, envs_idx, (3,)
        )
        # Between two steps, the state sits in frame 0 (see 'MPMSolver.roll_frame_window')
        with self._solver.recording_input():
            self._solver._kernel_set_particles_pos(0, particles_idx, envs_idx, poss)

    def get_particles_pos(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        poss = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self._solver._kernel_get_particles_pos(
            self._sim.cur_substep_local, self._particle_start, self.n_particles, envs_idx, poss
        )
        if self._scene.n_envs == 0:
            poss = poss[0]
        return poss

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None):
        vels, particles_idx, envs_idx = self._sanitize_written_particles(
            vels, gs.tc_float, particles_idx_local, envs_idx, (3,)
        )
        # See 'set_particles_pos'
        with self._solver.recording_input():
            self._solver._kernel_set_particles_vel(0, particles_idx, envs_idx, vels)

    def get_particles_vel(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vels = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self._solver._kernel_get_particles_vel(
            self._sim.cur_substep_local, self._particle_start, self.n_particles, envs_idx, vels
        )
        if self._scene.n_envs == 0:
            vels = vels[0]
        return vels

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None):
        actives, particles_idx, envs_idx = self._sanitize_written_particles(
            actives, gs.tc_bool, particles_idx_local, envs_idx
        )

        # FIXME: This check is too expensive
        # if not torch.isin(actives, torch.Tensor([False, True], dtype=gs.tc_bool, device=gs.device)).all():
        #     gs.raise_exception("Elements of `actives' must be either True or False.")

        # See 'set_particles_pos'
        with self._solver.recording_input(differentiable=False):
            self._solver._kernel_set_particles_active(0, particles_idx, envs_idx, actives)

    def get_particles_active(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actives = self._sanitize_particles_tensor(None, gs.tc_bool, None, envs_idx)
        self._solver._kernel_get_particles_active(
            self._sim.cur_substep_local, self._particle_start, self.n_particles, envs_idx, actives
        )
        if self._scene.n_envs == 0:
            actives = actives[0]
        return actives

    @assert_muscle
    def set_actuation(self, actus, envs_idx=None):
        """
        Set actuation values for each muscle group individually.

        Parameters
        ----------
        actus : int | array_like, shape ([M,] [n_particles,] [n_groups,])
            Tensor of actuation values.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        actus = to_gs_tensor(actus)
        if actus.ndim == 0:
            actus = actus.reshape((1,)).expand((self.material.n_groups,))
        self._set_particles_state(self.set_particles_actu, (self.material.n_groups,), gs.tc_float, actus, envs_idx)

    @assert_muscle
    @gs.assert_built
    def set_particles_actu(self, actus, particles_idx_local=None, envs_idx=None):
        """
        Set particle actuation values.

        Parameters
        ----------
        actus: torch.Tensor, shape (M, N, 3)
            Activation value of each particle.
        particles_idx_local : torch.Tensor, shape (M, N)
            Index of the particles relative to this entity.
        envs_idx : torch.Tensor, shape (M,)
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        actus, particles_idx, envs_idx = self._sanitize_written_particles(
            actus, gs.tc_float, particles_idx_local, envs_idx, (self.material.n_groups,)
        )
        # See 'set_particles_pos'
        with self._solver.recording_input():
            self._solver._kernel_set_particles_actu(0, self.material.n_groups, particles_idx, envs_idx, actus)

    def get_particles_actu(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actus = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx)
        self._solver._kernel_get_particles_actu(
            self._sim.cur_substep_local, self._particle_start, self.n_particles, envs_idx, actus
        )
        if self._scene.n_envs == 0:
            actus = actus[0]
        return actus

    @assert_muscle
    def set_muscle_group(self, muscle_group):
        """
        Set the muscle group index for each particle.

        Parameters
        ----------
        muscle_group : torch.Tensor, shape ([n_particles,])
            A tensor with integer group IDs.
        """
        particles_idx_local = self._sanitize_particles_idx_local(None)
        particles_idx = particles_idx_local + self._particle_start
        muscle_group = self._sanitize_particles_tensor(muscle_group, gs.tc_int, particles_idx, batched=False)

        # FIXME: This check is too expensive
        # if not (0 <= muscle_group & muscle_group < self.material.n_groups).all():
        #     gs.raise_exception("Elements of `muscle_group' are out-of-range.")

        self._solver._kernel_set_particles_muscle_group(particles_idx, muscle_group)

    @assert_muscle
    @assert_active
    def get_muscle_group(self):
        """
        Retrieve the muscle group index for each particle.

        Returns
        -------
        muscle_group : torch.Tensor, shape (n_particles,)
            A tensor containing the muscle group ID of each particle.
        """
        muscle_group = torch.zeros((self._n_particles,), dtype=gs.tc_int, device=gs.device)
        self._solver._kernel_get_particles_muscle_group(self._particle_start, self._n_particles, muscle_group)
        return muscle_group

    @assert_muscle
    @assert_active
    def set_muscle_direction(self, muscle_direction):
        """
        Set the muscle fiber direction for each particle.

        Parameters
        ----------
        muscle_direction : torch.Tensor, shape ([n_particles,] 3)
            A tensor with unit vectors representing muscle directions.
        """
        particles_idx_local = self._sanitize_particles_idx_local(None)
        particles_idx = particles_idx_local + self._particle_start
        muscle_direction = self._sanitize_particles_tensor(
            muscle_direction, gs.tc_float, particles_idx, None, (3,), batched=False
        )

        # FIXME: This check is too expensive
        # if not ((muscle_direction.norm(dim=-1) - 1.0).abs() < gs.EPS).all():
        #     gs.raise_exception("Last dimension of `muscle_direction' must be normalized.")

        self._solver._kernel_set_particles_muscle_direction(particles_idx, muscle_direction)

    def set_muscle(self, muscle_group=None, muscle_direction=None):
        """
        Set both the muscle group indices and direction vectors.

        Parameters
        ----------
        muscle_group : torch.Tensor, shape ([n_particles,]), optional
            A tensor  with group indices.
        muscle_direction : torch.Tensor, shape ([n_particles,] 3), optional
            A tensor with unit vectors.
        """
        if muscle_group is not None:
            self.set_muscle_group(muscle_group)

        if muscle_direction is not None:
            self.set_muscle_direction(muscle_direction)

    @assert_active
    def set_free(self, free):
        """
        Set particles as free or constrained.

        Parameters
        ----------
        free : torch.Tensor, shape ([n_particles,])
            A tensor indicating if each particle is free (1) or fixed (0).
        """
        particles_idx_local = self._sanitize_particles_idx_local(None)
        particles_idx = particles_idx_local + self._particle_start
        free = self._sanitize_particles_tensor(free, gs.tc_bool, particles_idx, batched=False)

        # FIXME: This check is too expensive
        # if not torch.isin(free, torch.Tensor([False, True], dtype=gs.tc_bool, device=gs.device)).all():
        #     gs.raise_exception("Elements of `free' must be either True or False.")

        self._solver._kernel_set_particles_free(particles_idx, free)

    @assert_active
    def get_free(self):
        """
        Get free/fixed status for all particles.

        Returns
        -------
        free : torch.Tensor, shape (n_particles,)
            A tensor indicating free (1) or fixed (0) status.
        """
        free = self._sanitize_particles_tensor(None, gs.tc_bool)
        self._solver._kernel_get_particles_free(self._particle_start, self._n_particles, free)
        return free

    # ------------------------------------------------------------------------------------
    # ------------------------------ particle constraints --------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_particles_in_bbox(self, bbox_min, bbox_max):
        """
        Get boolean mask for particles within a bounding box.

        Parameters
        ----------
        bbox_min : array_like, shape (3,)
            Minimum corner of the bounding box [x, y, z].
        bbox_max : array_like, shape (3,)
            Maximum corner of the bounding box [x, y, z].

        Returns
        -------
        mask : torch.Tensor, shape (n_envs, n_particles)
            Boolean mask where True indicates particle is within the bounding box.
        """
        bbox_min = torch.as_tensor(bbox_min, dtype=gs.tc_float, device=gs.device)
        bbox_max = torch.as_tensor(bbox_max, dtype=gs.tc_float, device=gs.device)

        # Get particle positions: shape (n_envs, n_particles, 3)
        poss = self.get_particles_pos()
        if poss.ndim == 2:
            poss = poss.unsqueeze(0)  # (1, n_particles, 3)

        # Vectorized bbox check: (n_envs, n_particles)
        mask = ((bbox_min <= poss) & (poss <= bbox_max)).all(dim=-1)
        return mask

    @gs.assert_built
    def set_particle_constraints(self, particles_mask, link_idx, stiffness):
        """
        Attach MPM particles to a rigid link using soft constraints.

        The particles will be pulled toward their relative position on the link
        using spring forces with critical damping.

        Parameters
        ----------
        particles_mask : torch.Tensor, shape (n_envs, n_particles)
            Boolean mask indicating which particles to constrain.
        link_idx : int
            Index of the rigid link to attach particles to.
        stiffness : float
            Spring stiffness for the constraint.
        """
        if not isinstance(link_idx, int):
            gs.raise_exception("link_idx must be an integer.")

        if not self._solver._constraints_initialized:
            self._solver.init_constraints()

        # Get link position and quaternion for all envs
        rigid_solver = self._sim.coupler.rigid_solver
        link_pos = rigid_solver.get_links_pos(links_idx=[link_idx])  # (n_envs, 1, 3)
        link_quat = rigid_solver.get_links_quat(links_idx=[link_idx])  # (n_envs, 1, 4)
        if link_pos.ndim == 2:
            link_pos = link_pos.unsqueeze(0)
            link_quat = link_quat.unsqueeze(0)
        link_pos = link_pos[:, 0, :]  # (n_envs, 3)
        link_quat = link_quat[:, 0, :]  # (n_envs, 4)

        self._solver._kernel_set_particle_constraints(
            self._sim.cur_substep_local,
            particles_mask,
            self._particle_start,
            stiffness,
            link_idx,
            link_pos,
            link_quat,
        )

    @gs.assert_built
    def remove_particle_constraints(self, particles_mask=None):
        """
        Remove constraints from specified particles, or all if None.

        Parameters
        ----------
        particles_mask : torch.Tensor, shape (n_envs, n_particles), optional
            Boolean mask indicating which particles to unconstrain. If None, removes all constraints for this entity.
        """
        if not self._solver._constraints_initialized:
            return

        # Remove all constraints for this entity if mask not specified
        if particles_mask is None:
            particles_mask = torch.ones((self._sim._B, self.n_particles), dtype=torch.bool, device=gs.device)

        self._solver._kernel_remove_particle_constraints(particles_mask, self._particle_start)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        return f"mpm_{super()._get_morph_identifier()}"
