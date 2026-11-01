import functools

import torch

import quadrants as qd

import genesis as gs
from genesis.engine.states.entities import MPMEntityState
from genesis.utils.misc import broadcast_tensor

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
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.idx,
            self._material._default_Jp,
            self._material.rho,
            self._particles,
        )

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
        self.set_particles_actu(actus, envs_idx=envs_idx)

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
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
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
        particles_idx = self._get_global_idx(None, self._n_particles, self._particle_start, unsafe=True)
        self._solver.set_particles_muscle_group(muscle_group, particles_idx)

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
