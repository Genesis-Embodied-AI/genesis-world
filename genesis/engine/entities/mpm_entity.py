import functools

import torch

import quadrants as qd

import genesis as gs
from genesis.engine.states.entities import MPMEntityState
from genesis.options.materials import Liquid, Muscle, Sand, Snow
from genesis.utils.misc import to_gs_tensor

from .particle_entity import ParticleEntity, assert_active


def assert_muscle(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self.material, Muscle):
            gs.raise_exception("This method is only supported by entities with a 'Muscle' material.")
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
    model_idx : int
        Index of the constitutive model of the entity in the solver.
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
        model_idx,
        particle_start,
        vvert_start,
        vface_start,
        name: str | None = None,
        *,
        options,
    ):
        need_skinning = not isinstance(material, (Liquid, Sand, Snow))
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
            options=options,
        )
        self._model_idx = model_idx
        # The number of muscle groups, which the hybrid entity owning this one as its skin sets to its number of links
        self._n_groups = options.solver_options.n_groups

    def init_tgt_keys(self):
        """
        Initialize target keys used for buffer-based state tracking.

        Sets up the list of keys for target states, including velocity, position, activeness, and finally actuation (for
        muscle only).
        """
        if isinstance(self.material, Muscle):
            self._tgt_keys = ("pos", "vel", "act", "actu")
        else:
            self._tgt_keys = ("pos", "vel", "act")

    def _add_to_solver(self):
        super()._add_to_solver()
        if isinstance(self.material, Muscle) and isinstance(self._morph, gs.options.morphs.MeshSet):
            self.set_muscle_group(self.mesh_set_group_ids)

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._model_idx,
            self._solver._models[self._model_idx].default_Jp,
            self._material.density,
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
            assert state.pos.grad.is_contiguous()
            self._kernel_add_frame_particles_pos_grad(self._sim.cur_substep_local, state.pos.grad)

        if state.vel.grad is not None:
            assert state.vel.grad.is_contiguous()
            self._kernel_add_frame_particles_vel_grad(self._sim.cur_substep_local, state.vel.grad)

        if state.C.grad is not None:
            assert state.C.grad.is_contiguous()
            self._kernel_add_frame_particles_C_grad(self._sim.cur_substep_local, state.C.grad)

        if state.F.grad is not None:
            assert state.F.grad.is_contiguous()
            self._kernel_add_frame_particles_F_grad(self._sim.cur_substep_local, state.F.grad)

        if state.Jp.grad is not None:
            assert state.Jp.grad.is_contiguous()
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

    def process_input_grad(self):
        """
        Process gradients for buffered inputs and backpropagate using custom kernels.

        Returns the pairs (input, gradient) of the inputs applied at this step that track gradients.
        """
        inputs_grad = []
        if isinstance(self.material, Muscle):
            _tgt_actu = self._tgt_buffer["actu"].pop()
            if _tgt_actu is not None and _tgt_actu.requires_grad:
                actu_grad = torch.zeros_like(_tgt_actu)
                self._set_particles_actu_grad(actu_grad)
                inputs_grad.append((_tgt_actu, actu_grad))

        return inputs_grad + super().process_input_grad()

    @gs.assert_built
    def get_state(self):
        """
        Get the current physical state of the particle entity.

        Returns
        -------
        state : MPMEntityState
            The current state of all physical properties of the entity, a view of the state of the solver.
        """
        solver_state = self._solver.get_state(self._sim.cur_substep_local)
        particles_slice = slice(self._particle_start, self._particle_start + self._n_particles)
        return MPMEntityState(
            self,
            self._sim.cur_step_global,
            solver_state.pos[:, particles_slice],
            solver_state.vel[:, particles_slice],
            solver_state.C[:, particles_slice],
            solver_state.F[:, particles_slice],
            solver_state.Jp[:, particles_slice],
            solver_state.active[:, particles_slice],
        )

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

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
            actus = actus.reshape((1,)).expand((self._n_groups,))
        self._set_particles_target_state("actu", "actuation", (self._n_groups,), gs.tc_float, actus, envs_idx)

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
        actus = self._sanitize_particles_tensor(actus, gs.tc_float, particles_idx, envs_idx, (self._n_groups,))
        self.solver._kernel_set_particles_actu(
            self._sim.cur_substep_local, self._n_groups, particles_idx, envs_idx, actus
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
        self._solver.set_particles_actu(actus, particles_idx, envs_idx, n_groups=self.material.n_groups)

    def get_particles_actu(self, envs_idx=None):
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        return self._solver.get_particles_actu(particles_idx, envs_idx)

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
        # if not (0 <= muscle_group & muscle_group < self._n_groups).all():
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
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        return self._solver.get_particles_muscle_group(particles_idx)

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
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        self._solver.set_particles_muscle_direction(muscle_direction, particles_idx)

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
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        self._solver.set_particles_free(free, particles_idx)

    @assert_active
    def get_free(self):
        """
        Get free/fixed status for all particles.

        Returns
        -------
        free : torch.Tensor, shape (n_particles,)
            A tensor indicating free (1) or fixed (0) status.
        """
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        return self._solver.get_particles_free(particles_idx)

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
        Attach some particles to a rigid link with critically damped springs.

        Each particle is pulled toward the position it holds now, expressed in the frame of the link.

        Parameters
        ----------
        particles_mask : array_like, shape ([n_envs,] n_particles)
            Boolean mask indicating which particles to constrain.
        link_idx : int
            Global index of the rigid link to attach particles to.
        stiffness : float
            Spring stiffness for the constraint.
        """
        particles_mask = broadcast_tensor(
            particles_mask, gs.tc_bool, (self._sim._B, self._n_particles), ("envs_idx", "particles_idx")
        )
        self._solver.set_particle_constraints(particles_mask, link_idx, stiffness, particles_start=self._particle_start)

    @gs.assert_built
    def remove_particle_constraints(self, particles_mask=None):
        """
        Remove constraints from specified particles, or all if None.

        Parameters
        ----------
        particles_mask : None | array_like, shape ([n_envs,] n_particles), optional
            Boolean mask indicating which particles to unconstrain. If None, removes all constraints for this entity.
        """
        particles_mask = broadcast_tensor(
            True if particles_mask is None else particles_mask,
            gs.tc_bool,
            (self._sim._B, self._n_particles),
            ("envs_idx", "particles_idx"),
        )
        self._solver.remove_particle_constraints(particles_mask, particles_start=self._particle_start)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        return f"mpm_{super()._get_morph_identifier()}"
