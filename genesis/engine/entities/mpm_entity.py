import functools

import torch

import quadrants as qd

import genesis as gs
from genesis.engine.states.entities import MPMEntityState
from genesis.utils.misc import to_gs_tensor

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
    material : MaterialOptions
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

    def init_tgt_keys(self):
        """
        Initialize target keys used for buffer-based state tracking.

        Sets up the list of keys for target states, including velocity, position, activeness, and finally actuation (for
        muscle only).
        """
        if isinstance(self.material, gs.materials.MPM.Muscle):
            self._tgt_keys = ("pos", "vel", "act", "actu")
        else:
            self._tgt_keys = ("pos", "vel", "act")

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

    def _reset_grad(self):
        """
        Clear all gradients for particle properties.
        """
        self._reset_frame_grad(self._sim.cur_substep_local)

    @qd.kernel
    def _reset_frame_grad(self, f: qd.i32):
        """
        Clear all gradients for particle properties at the given substep.

        Parameters
        ----------
        f : int
            The current substep index.
        """
        for i_p, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            self._solver.particles.grad[f, i_global, i_b].pos = 0
            self._solver.particles.grad[f, i_global, i_b].vel = 0
            self._solver.particles.grad[f, i_global, i_b].C = 0
            self._solver.particles.grad[f, i_global, i_b].F = 0
            self._solver.particles.grad[f, i_global, i_b].F_tmp = 0
            self._solver.particles.grad[f, i_global, i_b].Jp = 0
            self._solver.particles.grad[f, i_global, i_b].U = 0
            self._solver.particles.grad[f, i_global, i_b].V = 0
            self._solver.particles.grad[f, i_global, i_b].S = 0
            self._solver.particles.grad[f, i_global, i_b].actu = 0

    def add_grad_from_state(self, state):
        """
        Accumulate gradients from a recorded state back into the solver.

        Parameters
        ----------
        state : MPMEntityState
            The state object containing gradients for physical quantities.
        """
        if state.pos.grad is not None:
            state.pos.assert_contiguous()
            self._kernel_add_frame_particles_pos_grad(self._sim.cur_substep_local, state.pos.grad)

        if state.vel.grad is not None:
            state.vel.assert_contiguous()
            self._kernel_add_frame_particles_vel_grad(self._sim.cur_substep_local, state.vel.grad)

        if state.C.grad is not None:
            state.C.assert_contiguous()
            self._kernel_add_frame_particles_C_grad(self._sim.cur_substep_local, state.C.grad)

        if state.F.grad is not None:
            state.F.assert_contiguous()
            self._kernel_add_frame_particles_F_grad(self._sim.cur_substep_local, state.F.grad)

        if state.Jp.grad is not None:
            state.Jp.assert_contiguous()
            self._kernel_add_frame_particles_Jp_grad(self._sim.cur_substep_local, state.Jp.grad)

    @qd.kernel
    def _kernel_add_frame_particles_pos_grad(self, f: qd.i32, poss_grad: qd.types.ndarray()):
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
    def _kernel_add_frame_particles_vel_grad(self, f: qd.i32, vels_grad: qd.types.ndarray()):
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
    def _kernel_add_frame_particles_C_grad(self, f: qd.i32, C_grad: qd.types.ndarray()):
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
    def _kernel_add_frame_particles_F_grad(self, f: qd.i32, F_grad: qd.types.ndarray()):
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
    def _kernel_add_frame_particles_Jp_grad(self, f: qd.i32, Jp_grad: qd.types.ndarray()):
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

    def process_input(self, in_backward=False):
        if isinstance(self.material, gs.materials.MPM.Muscle) and self._tgt["actu"] is not None:
            self._tgt["actu"].assert_contiguous()
            self._tgt["actu"].assert_sceneless()
            particles_idx_local = self._sanitize_particles_idx_local(envs_idx=self._scene._envs_idx)
            self.set_particles_actu(self._tgt["actu"], particles_idx_local)

        super().process_input(in_backward)

    def process_input_grad(self):
        """
        Process gradients for buffered inputs and backpropagate using custom kernels.
        """
        if isinstance(self.material, gs.materials.MPM.Muscle):
            _tgt_actu = self._tgt_buffer["actu"].pop()
            if _tgt_actu is not None and _tgt_actu.requires_grad:
                _tgt_actu._backward_from_qd(self._set_particles_actu_grad)

        super().process_input_grad()

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

        # Store all queried states to track gradient flow
        self._queried_states.append(state)

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

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        poss = self._sanitize_particles_tensor(poss, gs.tc_float, particles_idx, envs_idx, (3,))
        self.solver._kernel_set_particles_pos(self._sim.cur_substep_local, particles_idx, envs_idx, poss)

    @gs.assert_built
    def _set_particles_pos_grad(self, poss_grad):
        self.solver._kernel_set_particles_pos_grad(
            self._sim.cur_substep_local, self._particle_start, self._n_particles, poss_grad
        )

    def get_particles_pos(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        poss = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self.solver._kernel_get_particles_pos(
            self._sim.cur_substep_local, self._particle_start, self.n_particles, envs_idx, poss
        )
        if self._scene.n_envs == 0:
            poss = poss[0]
        return poss

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        vels = self._sanitize_particles_tensor(vels, gs.tc_float, particles_idx, envs_idx, (3,))
        self.solver._kernel_set_particles_vel(self._sim.cur_substep_local, particles_idx, envs_idx, vels)

    @gs.assert_built
    def _set_particles_vel_grad(self, vels_grad):
        self.solver._kernel_set_particles_vel_grad(
            self._sim.cur_substep_local, self._particle_start, self._n_particles, vels_grad
        )

    def get_particles_vel(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vels = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self.solver._kernel_get_particles_vel(
            self._sim.cur_substep_local, self._particle_start, self.n_particles, envs_idx, vels
        )
        if self._scene.n_envs == 0:
            vels = vels[0]
        return vels

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        actives = self._sanitize_particles_tensor(actives, gs.tc_bool, particles_idx, envs_idx)

        # FIXME: This check is too expensive
        # if not torch.isin(actives, torch.Tensor([False, True], dtype=gs.tc_bool, device=gs.device)).all():
        #     gs.raise_exception("Elements of `actives' must be either True or False.")

        self.solver._kernel_set_particles_active(self._sim.cur_substep_local, particles_idx, envs_idx, actives)

    def get_particles_active(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actives = self._sanitize_particles_tensor(None, gs.tc_bool, None, envs_idx)
        self.solver._kernel_get_particles_active(
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
        self._set_particles_target_state("actu", "actuation", (self.material.n_groups,), gs.tc_float, actus, envs_idx)

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
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        actus = self._sanitize_particles_tensor(actus, gs.tc_float, particles_idx, envs_idx, (self.material.n_groups,))
        self.solver._kernel_set_particles_actu(
            self._sim.cur_substep_local, self.material.n_groups, particles_idx, envs_idx, actus
        )

    @gs.assert_built
    def _set_particles_actu_grad(self, actu_grad):
        """
        Set gradients for particle actuation values.

        Parameters
        ----------
        actu_grad : torch.Tensor
            A tensor containing gradients for actuation inputs.
        """
        self.solver._kernel_set_particles_actu_grad(
            self._sim.cur_substep_local, self._particle_start, self._n_particles, actu_grad
        )

    def get_particles_actu(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actus = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (self.material.n_groups,))
        self.solver._kernel_get_particles_actu(
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

        self.solver._kernel_set_particles_muscle_group(particles_idx, muscle_group)

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
        muscle_group = gs.zeros((self._n_particles,), dtype=gs.tc_int, requires_grad=False, scene=self._scene)
        self.solver._kernel_get_particles_muscle_group(self._particle_start, self._n_particles, muscle_group)
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

        self.solver._kernel_set_particles_muscle_direction(particles_idx, muscle_direction)

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

        self.solver._kernel_set_particles_free(particles_idx, free)

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
        self.solver._kernel_get_particles_free(self._particle_start, self._n_particles, free)
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
