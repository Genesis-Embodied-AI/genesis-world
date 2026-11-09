from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.state_change import StateChange, mutates
from genesis.options.solvers import MPMOptions, PBDOptions, SPHOptions
from genesis.utils.data import ENVS, Data, DataKind, Size, eoa, field
from genesis.utils.geom import PARTICLES, SpatialHashGrid, func_hash_grid_to_slot, func_hash_pos_to_grid
from genesis.utils.misc import indices_to_mask, qd_to_torch

from .base_solver import VVERTS, GravityMixin, Solver, TimeBasedMixin, kernel_set_values

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


# The particles supporting each visual vertex
VVERT_SUPPORTS = Size("vvert_supports")


class ParticlesRender(Data, kind=DataKind.SCRATCH, shape=(PARTICLES, ENVS)):
    """The particles as drawn: their position and velocity, an inactive particle moved out of sight."""

    pos: qd.Tensor = field((float, 3))
    vel: qd.Tensor = field((float, 3))
    is_active: qd.Tensor = field(bool)


class VVertsSupport(Data, kind=DataKind.CONSTANT, shape=(VVERTS,)):
    """The visual vertices skinned by particles: each one the weighted sum of the positions of its support particles."""

    support_idxs: qd.Tensor = field((int, VVERT_SUPPORTS))
    support_weights: qd.Tensor = field((float, VVERT_SUPPORTS))


class VVertsRender(Data, kind=DataKind.SCRATCH, shape=(VVERTS, ENVS)):
    """The visual vertices as drawn, active when their first support particle is."""

    pos: qd.Tensor = field((float, 3))
    is_active: qd.Tensor = field(bool)


class ParticleSolver(GravityMixin, TimeBasedMixin, Solver):
    """Base class of the particle-based solvers, holding what they share: the accessors of the state of the particles.

    The accessors take global particle indices, given once for every environment or one row per environment (see
    '_sanitize_io_variables'), and the kernels of each solver read and write the state the particles hold between two
    steps. A solver assigns the arrays of the position, velocity and activity of its particles, the array of their mass,
    and the particles and visual vertices as drawn, at build time, which the accessors and the update of the rendering
    then read and write.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: MPMOptions | PBDOptions | SPHOptions) -> None:
        super().__init__(scene, sim, options)

        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._n_vvert_supports = scene.options.vis.n_support_neighbors
        # The distance from the walls of the domain to the boundary confining the particles
        self.boundary_padding = 0.0

        self._n_particles: int | None = None
        self._n_vverts: int | None = None
        self._n_vfaces: int | None = None

        # The arrays of the particles the accessors read and write, held in frames by a solver that keeps its state in
        # frames (see 'Solver.has_frames'), and those the rendering reads.
        self._particles_pos: qd.Tensor | None = None
        self._particles_vel: qd.Tensor | None = None
        self._particles_is_active: qd.Tensor | None = None
        self._particles_mass: qd.Tensor | None = None
        self._vverts_support: VVertsSupport | None = None
        self.particles_render: ParticlesRender | None = None
        self.vverts_render: VVertsRender | None = None

    def build(self) -> None:
        super().build()
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

    def _build_render(self, has_vverts: bool) -> None:
        """
        Allocate the particles as drawn, and the visual vertices as drawn with their support.

        The visual vertices are allocated only with `has_vverts`.
        """
        is_unused = not self.is_active
        self.particles_render = ParticlesRender(
            sizes=dict(envs=self._B, particles=self._n_particles), env_first=False, unused=is_unused
        )
        if has_vverts:
            self._vverts_support = VVertsSupport(
                sizes=dict(vverts=self._n_vverts, vvert_supports=self._n_vvert_supports), unused=is_unused
            )
            self.vverts_render = VVertsRender(
                sizes=dict(envs=self._B, vverts=self._n_vverts), env_first=False, unused=is_unused
            )

    def add_vverts(
        self,
        vvert_start: int,
        particle_start: int,
        vverts: np.ndarray,
        particles: np.ndarray,
        P_invs: np.ndarray,
        support_idxs_local: np.ndarray,
    ) -> None:
        """Bind the given vverts to their supporting particles (see 'kernel_add_vverts')."""
        kernel_add_vverts(
            vvert_start, particle_start, support_idxs_local, vverts, particles, P_invs, self._vverts_support
        )

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_particles_pos(self, poss, particles_idx=None, envs_idx=None) -> None:
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
        kernel_set_particles_pos(
            particles_idx, envs_idx, poss, self._particles_pos, self._particles_vel, self.has_frames
        )
        self._reset_particles(particles_idx, envs_idx, reset_deformation=True)
        # Overwriting the positions of the particles clears the errors their former state flagged
        self.clear_errno(envs_idx)

    def get_particles_pos(self, particles_idx=None, envs_idx=None) -> torch.Tensor:
        """Get the position of some particles, the same in every environment (see 'set_particles_pos')."""
        tensor = self._read_data(self._particles_pos, envs_idx, particles_idx, copy=True)
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_particles_vel(self, vels, particles_idx=None, envs_idx=None) -> None:
        """Set the velocity of some particles (see 'set_particles_pos')."""
        self._write_data(
            self._particles_vel, vels, particles_idx, self.n_particles, "particles_idx", envs_idx, element_shape=(3,)
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def get_particles_vel(self, particles_idx=None, envs_idx=None) -> torch.Tensor:
        """Get the velocity of some particles, the same in every environment (see 'set_particles_pos')."""
        tensor = self._read_data(self._particles_vel, envs_idx, particles_idx, copy=True)
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @gs.assert_built
    @mutates(StateChange.GEOMETRY, differentiable=False)
    def set_particles_active(self, actives, particles_idx=None, envs_idx=None) -> None:
        """Set whether some particles take part in the simulation (see 'set_particles_pos')."""
        actives, particles_idx, envs_idx = self._sanitize_io_variables(
            actives, particles_idx, self.n_particles, "particles_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        kernel_set_values(particles_idx, envs_idx, actives, self._particles_is_active, self.has_frames)
        self._reset_particles(particles_idx, envs_idx, reset_deformation=False)

    def get_particles_active(self, particles_idx=None, envs_idx=None) -> torch.Tensor:
        """Get whether some particles take part in the simulation (see 'set_particles_pos')."""
        tensor = self._read_data(self._particles_is_active, envs_idx, particles_idx, copy=True)
        return tensor[0] if self._scene.n_envs == 0 else tensor

    def get_particles_mass(self, particles_idx=None, envs_idx=None) -> torch.Tensor:
        """Get the mass of some particles (see 'set_particles_pos')."""
        # The particles store their mass multiplied by 'particle_mass_scale', the same in every environment
        masses = (qd_to_torch(self._particles_mass) / self.particle_mass_scale)[None].expand(self._B, -1)
        masses = masses[indices_to_mask(envs_idx, particles_idx, pair_rows=True)]
        return masses[0] if self._scene.n_envs == 0 else masses

    def _reset_particles(self, particles_idx: torch.Tensor, envs_idx: torch.Tensor, reset_deformation: bool) -> None:
        """
        Reset the state of the given particles beside their position and velocity, after they moved or changed activity.

        `reset_deformation` asks a deformable material to restart undeformed.
        """
        pass

    def update_render_fields(self) -> None:
        """Update the particles and the visual vertices as drawn from the state of the particles."""
        f = self._sim.cur_substep_local
        kernel_update_particles_render(
            f,
            self._particles_pos,
            self._particles_vel,
            self._particles_is_active,
            self.particles_render,
            self.has_frames,
        )
        if self.vverts_render is not None:
            kernel_update_vverts_render(
                f, self._particles_pos, self.particles_render, self.vverts_render, self._vverts_support, self.has_frames
            )

    @property
    def is_active(self) -> bool:
        return self.n_particles > 0

    @property
    def particle_mass_scale(self) -> float:
        """The factor the particles store their mass multiplied by, for numerical robustness."""
        return 1.0

    @property
    def n_particles(self) -> int:
        if self.is_built:
            return self._n_particles
        return sum([entity.n_particles for entity in self._entities])

    @property
    def n_vvert_supports(self) -> int:
        """Number of particles supporting each visual vertex."""
        return self._n_vvert_supports

    @property
    def n_vverts(self) -> int:
        if self.is_built:
            return self._n_vverts
        return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_vfaces(self) -> int:
        if self.is_built:
            return self._n_vfaces
        return sum([entity.n_vfaces for entity in self._entities])

    @property
    def particle_size(self) -> float:
        return self._particle_size

    @property
    def particle_radius(self) -> float:
        return self._particle_size / 2.0

    @property
    def upper_bound(self) -> np.ndarray:
        return self._upper_bound

    @property
    def lower_bound(self) -> np.ndarray:
        return self._lower_bound

    @property
    def boundary_lower(self) -> np.ndarray:
        """The lower corner of the boundary confining the particles, inside the domain by the padding."""
        return self._lower_bound + self.boundary_padding

    @property
    def boundary_upper(self) -> np.ndarray:
        """The upper corner of the boundary confining the particles, inside the domain by the padding."""
        return self._upper_bound - self.boundary_padding


@qd.kernel(fastcache=True)
def kernel_add_vverts(
    vvert_start: int,
    particle_start: int,
    support_idxs_local: qd.types.ndarray(),
    vverts: qd.types.ndarray(element_dim=1),
    particles: qd.types.ndarray(element_dim=1),
    P_invs: qd.types.ndarray(element_dim=2),
    vverts_support: VVertsSupport,
):
    """
    Bind the given vverts to their supporting particles, whose positions skin them for rendering.

    The weights of a vvert are its barycentric coordinates in the simplex of its support particles, `P_invs` mapping its
    offset from the last support to the others.
    """
    n_supports = support_idxs_local.shape[1]
    for i_vv_ in range(vverts.shape[0]):
        i_vv = i_vv_ + vvert_start
        v = vverts[i_vv_]
        P_inv = P_invs[i_vv_]
        i_p_last = support_idxs_local[i_vv_, n_supports - 1]
        b = P_inv @ (v - particles[i_p_last])
        for i_s in range(n_supports - 1):
            vverts_support.support_weights[i_vv][i_s] = b[i_s]
        vverts_support.support_weights[i_vv][n_supports - 1] = 1.0 - b.sum()
        for i_s in range(n_supports):
            vverts_support.support_idxs[i_vv][i_s] = support_idxs_local[i_vv_, i_s] + particle_start


@qd.kernel(fastcache=True)
def kernel_set_particles_pos(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    has_frames: qd.template(),
):
    """Set the position of some particles, at rest."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(particles_pos))):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_values' in base_solver.py
        if i_p >= 0:
            I_p = [0, i_p, i_b] if qd.static(has_frames) else [i_p, i_b]
            particles_pos[I_p] = qd.Vector([poss[i_b_, i_p_, 0], poss[i_b_, i_p_, 1], poss[i_b_, i_p_, 2]])
            particles_vel[I_p] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_update_particles_render(
    f: int,
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_is_active: qd.Tensor,
    particles_render: ParticlesRender,
    has_frames: qd.template(),
):
    """Update the particles as drawn from their state, at the given frame for a solver holding its state in frames."""
    for i_p, i_b in qd.ndrange(*particles_render.pos.shape, axes=qd.static(eoa(particles_render.is_active))):
        I_p = [f, i_p, i_b] if qd.static(has_frames) else [i_p, i_b]
        if particles_is_active[I_p]:
            particles_render.pos[i_p, i_b] = particles_pos[I_p]
            particles_render.vel[i_p, i_b] = particles_vel[I_p]
        else:
            particles_render.pos[i_p, i_b] = gu.qd_nowhere()
        particles_render.is_active[i_p, i_b] = particles_is_active[I_p]


@qd.kernel(fastcache=True)
def kernel_update_vverts_render(
    f: int,
    particles_pos: qd.Tensor,
    particles_render: ParticlesRender,
    vverts_render: VVertsRender,
    vverts_support: VVertsSupport,
    has_frames: qd.template(),
):
    """Update the visual vertices as drawn from the particles supporting them, once the particles are."""
    for i_vv, i_b in qd.ndrange(*vverts_render.pos.shape, axes=qd.static(eoa(vverts_render.pos))):
        support_idxs = vverts_support.support_idxs[i_vv]
        support_weights = vverts_support.support_weights[i_vv]
        vvert_pos = qd.Vector.zero(gs.qd_float, 3)
        for k in qd.static(range(support_idxs.n)):
            i_p = support_idxs[k]
            I_p = [f, i_p, i_b] if qd.static(has_frames) else [i_p, i_b]
            vvert_pos += particles_pos[I_p] * support_weights[k]
        vverts_render.pos[i_vv, i_b] = vvert_pos
        i_p_first = support_idxs[0]
        vverts_render.is_active[i_vv, i_b] = particles_render.is_active[i_p_first, i_b]


# FIXME: quadrants#1036 - a func parameter accepts no subclass of its dataclass annotation, so the funcs shared by the
# solvers take the arrays of their particles one by one.
@qd.func
def func_get_particles_mean_vel(
    i_b: int,
    pos: qd.types.vector(3),
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    hash_grid: SpatialHashGrid,
    half_size: float,
    n_cells: int,
):
    """
    Return the mean velocity of the sorted particles of a solver within a cube around a position, and their count.

    The cube is centered at `pos` with half side `half_size`. The particles are searched for among the cells of the hash
    grid of the solver, `n_cells` along each axis from the one holding the lowest corner of the cube.
    """
    base = func_hash_pos_to_grid(pos - half_size, hash_grid)
    vel_sum = qd.Vector.zero(gs.qd_float, 3)
    n_particles = 0
    for offset in qd.grouped(qd.ndrange(n_cells, n_cells, n_cells)):
        i_s = func_hash_grid_to_slot(base + offset, hash_grid)
        i_p_start = hash_grid.slot_start[i_s, i_b]
        for i_p in range(i_p_start, i_p_start + hash_grid.slot_size[i_s, i_b]):
            if qd.abs(pos - particles_pos[i_p, i_b]).max() < half_size:
                vel_sum += particles_vel[i_p, i_b]
                n_particles += 1
    return vel_sum / qd.max(n_particles, 1), n_particles


@qd.func
def func_add_particles_momentum(
    i_b: int,
    pos: qd.types.vector(3),
    momentum: qd.types.vector(3),
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_is_moved: qd.Tensor,
    particles_mass: qd.Tensor,
    hash_grid: SpatialHashGrid,
    half_size: float,
    n_cells: int,
):
    """
    Add a momentum to the sorted particles of a solver in a cube around a position (see func_get_particles_mean_vel).

    Each particle that `particles_is_moved` selects gains the velocity `momentum` over its mass, which `particles_mass`
    holds by particle index.
    """
    base = func_hash_pos_to_grid(pos - half_size, hash_grid)
    for offset in qd.grouped(qd.ndrange(n_cells, n_cells, n_cells)):
        i_s = func_hash_grid_to_slot(base + offset, hash_grid)
        i_p_start = hash_grid.slot_start[i_s, i_b]
        for i_p in range(i_p_start, i_p_start + hash_grid.slot_size[i_s, i_b]):
            if qd.abs(pos - particles_pos[i_p, i_b]).max() < half_size and particles_is_moved[i_p, i_b]:
                i_o = hash_grid.particles_idx[i_p, i_b]
                particles_vel[i_p, i_b] = particles_vel[i_p, i_b] + momentum / particles_mass[i_o]
