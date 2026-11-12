from collections.abc import Iterator, Sequence
from enum import auto
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.boundaries import CubeBoundary, func_impose_cube_boundary
from genesis.engine.entities import MPMEntity
from genesis.engine.materials import MPM
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.abd.misc import func_add_external_wrench
from genesis.engine.solvers.rigid.data import GEOMS, GeomsInfo, GeomsState, LinksState, RigidData, RigidInfo
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import SolverCheckpoint
from genesis.grad.bridge import no_grad, register_custom_backward, reversed_with
from genesis.options.solvers import MPMOptions
from genesis.utils.data import (
    ENVS,
    Config,
    ConfigItem,
    Data,
    DataItem,
    DataKind,
    Size,
    eoa,
    field,
    iter_data,
    opt_in,
)
from genesis.utils.geom import PARTICLES, SpatialHashGrid, func_sort_particles_by_slot, hash_grid_sizes
from genesis.utils.misc import broadcast_tensor, qd_isfinite, qd_to_torch, qd_zero_grad
from genesis.utils.sdf import SDFInfo
from genesis.utils.simt import requires_deterministic_accumulation

from .base_solver import ErrorCode, kernel_set_actu
from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class MPMErrorCode(ErrorCode):
    """The error flags the material point method (MPM) solver writes in its errno (see 'Solver.register_errors')."""

    INVALID_VELOCITY_NAN = auto()


# The cells of the grid of the material point method (MPM) solver along each axis
GRID_X = Size("grid_x")
GRID_Y = Size("grid_y")
GRID_Z = Size("grid_z")
# The tiles of cells of the grid, and the groups of environments a warp gathers the cells of a tile for (see MPMTiles)
TILES = Size("tiles")
ENVS_GROUPS = Size("envs_groups")
# The arrays of the sparse reset of the grid, which the backward pass leaves out (see MPMGrid)
SPARSE_GRID_RESET = "sparse_grid_reset"
# The arrays of the compatible particle-in-cell (CPIC) separation of the particles from the cells by the rigid geoms
CPIC = "cpic"
# The arrays of the deterministic transfer of the particles to the grid (see MPMConfig)
DETERMINISTIC = "deterministic"
# The arrays the backward pass alone reads
GRAD = "grad"


class MPMParticles(Data, kind=DataKind.STATE, shape=(PARTICLES, ENVS)):
    """Runtime state of the material point method (MPM) particles, one value per particle and environment."""

    pos: qd.Tensor = field((float, 3), grad=True)
    vel: qd.Tensor = field((float, 3), grad=True)
    # The affine velocity field
    C: qd.Tensor = field((float, (3, 3)), grad=True)
    # The deformation gradient
    F: qd.Tensor = field((float, (3, 3)), grad=True)
    # The state a substep integrates, which it copies to the state once over (see 'kernel_copy_next_to_curr')
    pos_next: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, grad=True)
    vel_next: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, grad=True)
    C_next: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    F_next: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    Jp_next: qd.Tensor = field(float, kind=DataKind.SCRATCH, grad=True)
    # The deformation gradient advected over the substep, before the plasticity of the material applies, and its
    # singular value decomposition
    F_tmp: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    U: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    V: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    S: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    # The actuation of the muscles
    actu: qd.Tensor = field(float, grad=True)
    # The plastic volume ratio
    Jp: qd.Tensor = field(float, grad=True)
    is_active: qd.Tensor = field(bool)
    # The momentum per unit offset a particle gives the cells of its stencil, its stress and affine velocity together
    affine: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, labels=DETERMINISTIC)
    # The velocity a particle gathers from the grid, before the boundary applies (see kernel_transfer_grid_to_particles)
    vel_tmp: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, grad=True, labels=GRAD)
    # The flat index of the lowest cell of the stencil of a particle, which tells apart the cells sharing a slot
    cell_idx: qd.Tensor = field(int, kind=DataKind.SCRATCH, labels=DETERMINISTIC)


class MPMParticlesInfo(Data, kind=DataKind.INFO, shape=(PARTICLES,)):
    """
    Model parameters of the material point method (MPM) particles, one value per particle.

    `default_Jp` is the plastic volume ratio a particle takes back when its position is set, and `is_free` whether it
    moves or acts as a boundary condition of the grid.
    """

    material_idx: qd.Tensor = field(int, kind=DataKind.CONSTANT)
    mass: qd.Tensor = field(float, kind=DataKind.CONSTANT)
    default_Jp: qd.Tensor = field(float, kind=DataKind.CONSTANT)
    is_free: qd.Tensor = field(bool)
    muscle_group: qd.Tensor = field(int)
    muscle_direction: qd.Tensor = field((float, 3))


class MPMGrid(Data, kind=DataKind.SCRATCH, shape=(GRID_X, GRID_Y, GRID_Z, ENVS)):
    """
    The background grid of the material point method (MPM), one value per cell and environment.

    `vel_in` holds the momentum the particles scatter, and `vel_out` the velocity of the cells once the grid is updated.
    Without gradient, the cells touched by a substep are zeroed right after it, listed once across the environments:
    `dirty_flag` marks a listed cell, and `dirty_list` holds the flat indices of the `dirty_count` listed cells. With
    gradient, the grid is zeroed whole with its adjoint at the start of every substep instead.
    """

    mass: qd.Tensor = field(float, grad=True)
    vel_in: qd.Tensor = field((float, 3), grad=True)
    vel_out: qd.Tensor = field((float, 3), grad=True)
    dirty_flag: qd.Tensor = field(int, shape=(GRID_X, GRID_Y, GRID_Z), labels=SPARSE_GRID_RESET)
    dirty_list: qd.Tensor = field(int, shape=(GRID_X * GRID_Y * GRID_Z,), labels=SPARSE_GRID_RESET)
    dirty_count: qd.Tensor = field(int, shape=(1,), labels=SPARSE_GRID_RESET)


class MPMParticlesCPIC(Data, kind=DataKind.STATE, shape=(PARTICLES, GEOMS, ENVS)):
    """
    The separation of the material point method (MPM) particles from the cells of the grid by the thin rigid geoms
    (CPIC).

    `rigid_normal` is the normal of each rigid geom the particle last saw on its side, kept from one substep to the next
    while the particle does not cross the geom. `cell_geom_idx` is the geom separating the particle from each cell of
    its stencil, -1 for none, which the transfer to the grid finds and the integration of the particles reads.
    """

    rigid_normal: qd.Tensor = field((float, 3))
    cell_geom_idx: qd.Tensor = field(int, shape=(PARTICLES, 3, 3, 3, ENVS), kind=DataKind.SCRATCH)


class MPMTiles(Data, kind=DataKind.SCRATCH, shape=(TILES * ENVS_GROUPS,)):
    """The cubic tiles of cells of the grid that the stencils of the particles cover, by group of environments."""

    # Whether the lowest cell of the stencil of some particle of the group lies in the tile
    is_occupied: qd.Tensor = field(bool, shape=(TILES, ENVS_GROUPS))
    # The tiles some stencil of the group covers, `active_tile` being the flat index of each of the `n_active` tiles of
    # group `active_envs_group`, in no particular order
    n_active: qd.Tensor = field(int, shape=(1,))
    active_tile: qd.Tensor = field(int)
    active_envs_group: qd.Tensor = field(int)


class MPMState(Data, kind=DataKind.STATE):
    """Runtime state of the material point method (MPM) solver."""

    particles: MPMParticles
    grid: MPMGrid
    particles_cpic: MPMParticlesCPIC = field(labels=CPIC)
    # The particles sorted by the lowest cell of their stencil, a slot of the hash grid hashing each cell
    hash_grid: SpatialHashGrid = field(kind=DataKind.SCRATCH)
    tiles: MPMTiles = field(labels=DETERMINISTIC)


class MPMInfo(Data, kind=DataKind.CONSTANT):
    """Model parameters of the material point method (MPM) solver."""

    particles: MPMParticlesInfo
    boundary: CubeBoundary
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float, shape=())
    # The size of a cell of the grid, which spans the cells from `grid_offset` on
    dx: qd.Tensor = field(float, shape=())
    inv_dx: qd.Tensor = field(float, shape=())
    grid_offset: qd.Tensor = field((int, 3), shape=())
    # The volume a particle stands for, scaled up by `particle_volume_scale` for numerical robustness, which the masses
    # exchanged with the other solvers divide out
    particle_volume: qd.Tensor = field(float, shape=())
    particle_volume_scale: qd.Tensor = field(float, shape=())
    # The factor turning the stress of a particle into the momentum it gives the grid over a substep
    stress_scale: qd.Tensor = field(float, shape=())
    # The bound away from zero of the gaps between singular values that the gradient of the decomposition divides by
    EPS: qd.Tensor = field(float, shape=())


# TODO: 'scene.reset' restores the state alone, so constraints set after a saved state survive its restore. To revisit,
# as state carried by the checkpoints or as an explicit reset of the parameters the setters configure.
class MPMParticleConstraints(Data, kind=DataKind.INFO, shape=(PARTICLES, ENVS)):
    """
    The critically damped springs pulling some particles toward their targets, one per particle and environment.

    A constrained particle is pulled toward `target_pos`, or toward the point `link_local_pos` of the rigid link of
    index `link_idx` when it is not -1.
    """

    is_constrained: qd.Tensor = field(bool)
    target_pos: qd.Tensor = field((float, 3))
    stiffness: qd.Tensor = field(float)
    link_idx: qd.Tensor = field(int)
    link_local_pos: qd.Tensor = field((float, 3))


class MPMConfig(Config):
    """The compile-time configuration of the kernels of the material point method (MPM) solver."""

    # Whether the singular value decomposition of the advected deformation gradient is computed, which a material
    # reading it and the backward pass require. Otherwise J is the determinant of the advected deformation gradient.
    is_svd_needed: bool
    # Whether the grid is zeroed whole with its adjoint before every substep, instead of sparsely after it (see MPMGrid)
    requires_grad: bool
    # Whether the rigid geoms separate the particles from the cells of the grid (see MPMParticlesCPIC)
    is_cpic: bool
    # Whether the transfer to the grid and the coupling forces sum in a fixed order across threads, with
    # `use_deterministic_algorithms` (see kernel_transfer_particles_to_grid and 'cfrc_external_sums' in LinksState)
    is_deterministic: bool


class MPMData(Data):
    """
    Every array of the material point method (MPM) solver but its particle constraints.

    The solver allocates it once its scene is built.
    """

    mpm_info: MPMInfo
    mpm_state: MPMState
    errno: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)

    def __init__(self, solver: "MPMSolver"):
        grid_x, grid_y, grid_z = solver._grid_res.tolist()
        # See TILE_SIZE and ENVS_GROUP_SIZE in 'func_sort_particles'
        n_tiles = int(np.prod((solver._grid_res + 3) // 4))
        values = {}
        unused = {(MPMData.mpm_info, MPMData.mpm_state): True}
        if solver.is_active:
            info_values = dict(
                boundary=dict(lower=solver.boundary_lower, upper=solver.boundary_upper, restitution=0.0),
                substep_dt=solver.substep_dt,
                dx=solver._dx,
                inv_dx=solver._inv_dx,
                grid_offset=tuple(solver._lower_bound_cell.tolist()),
                particle_volume=solver._particle_volume,
                particle_volume_scale=solver._particle_volume_scale,
                stress_scale=-solver.substep_dt * solver._particle_volume * 4 * solver._inv_dx * solver._inv_dx,
                EPS=gs.EPS,
            )
            values = dict(mpm_info=info_values)
            unused = {
                SPARSE_GRID_RESET: solver._requires_grad,
                CPIC: not solver.mpm_config.is_cpic,
                DETERMINISTIC: not solver.mpm_config.is_deterministic,
                GRAD: not solver._requires_grad,
                MPMData.mpm_state.hash_grid: not solver.mpm_config.is_deterministic,
                # The flat index of a cell picks its slot (see func_sort_particles), so the grid of the hash goes unread
                (MPMData.mpm_state.hash_grid.cell_size, MPMData.mpm_state.hash_grid.grid_res): True,
            }
        super().__init__(
            sizes=dict(
                particles=solver._n_particles,
                envs=solver._B,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=grid_z,
                geoms=solver._sim.rigid_solver.n_geoms_,
                # The slots of the hash grid hash the cells of the grid, one per particle
                **hash_grid_sizes(max(solver._n_particles, 1)),
                tiles=n_tiles,
                envs_groups=(solver._B + 31) // 32,
            ),
            values=values,
            # The particles and the cells of the grid keep the environments innermost, as the kernels loop them
            env_first={
                (
                    MPMData.mpm_state.particles,
                    MPMData.mpm_state.particles_cpic,
                    MPMData.mpm_state.grid.mass,
                    MPMData.mpm_state.grid.vel_in,
                    MPMData.mpm_state.grid.vel_out,
                    MPMData.mpm_state.hash_grid,
                ): False
            },
            unused=unused,
            requires_grad=solver._requires_grad,
        )


@qd.data_oriented
class MPMMaterials:
    """
    The distinct materials of the material point method (MPM) entities.

    The kernels dispatch the particles to them by material index.
    """

    def __init__(self, materials: tuple[MPM.Base, ...]) -> None:
        self.materials = materials

    @qd.func
    def update_F_S_Jp(
        self,
        i_mat: int,
        J: float,
        F_tmp: qd.types.matrix(3, 3),
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        Jp: float,
        eps: float,
    ):
        """
        Return the deformation gradient, its singular values and the plastic volume ratio once plasticity applies.

        The particle is of the material `i_mat`, the other arguments being those of 'MPM.Base.update_F_S_Jp'.
        """
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        Jp_new = gs.qd_float(1.0)
        for j_mat in qd.static(range(len(self.materials))):
            if i_mat == j_mat:
                F_new, S_new, Jp_new = self.materials[j_mat].update_F_S_Jp(J, F_tmp, U, S, V, Jp, eps)
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(
        self,
        i_mat: int,
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        F_tmp: qd.types.matrix(3, 3),
        F_new: qd.types.matrix(3, 3),
        J: float,
        Jp: float,
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        """
        Return the stress of a particle (see 'MPM.Base.update_stress').

        The particle is of the material `i_mat`.
        """
        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        for j_mat in qd.static(range(len(self.materials))):
            if i_mat == j_mat:
                stress = self.materials[j_mat].update_stress(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir)
        return stress


class MPMSolver(ParticleSolver):
    # Its whole state lives in the arrays 'data' yields (see 'CheckpointMixin')
    is_checkpointable = True

    material_cls = MPM.Base

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: MPMOptions) -> None:
        super().__init__(scene, sim, options)

        self._grid_density = options.grid_density
        self._enable_CPIC = options.enable_CPIC
        # The rigid geoms separate the particles from the cells by their SDF (see MPMParticlesCPIC)
        opt_in(SDFInfo, cond=lambda: self.is_active and self._enable_CPIC)
        # The grid and the particles sum their forces on the rigid links exactly when the order of the sums is fixed
        opt_in(
            RigidData.dyn_state.links.cfrc_external_sums,
            cond=lambda: self.is_active and requires_deterministic_accumulation(),
        )

        # The actual volume of a particle may be very small, which its mass would suffer from numerically. The volume
        # is scaled up, which changes nothing to the MPM simulation itself, and the couplings scale the mass back.
        self._particle_volume_real = self._particle_size**3
        self._particle_volume_scale = 1e3
        self._particle_volume = self._particle_volume_real * self._particle_volume_scale

        # Other derived parameters
        self._dx = 1.0 / self._grid_density
        self._inv_dx = self._grid_density
        self._lower_bound_cell = np.round(self._grid_density * self._lower_bound).astype(gs.np_int)
        self._upper_bound_cell = np.round(self._grid_density * self._upper_bound).astype(gs.np_int)
        # Both corners included
        self._grid_res = self._upper_bound_cell - self._lower_bound_cell + 1
        self._grid_offset = qd.Vector(self._lower_bound_cell)
        if np.prod(self._grid_res) > 1e9:
            gs.raise_exception(
                """Grid size larger than 1e9 not supported by MPM solver.
Please reduce 'grid_density', or set tighter boundaries via 'lower_bound' / 'upper_bound'."""
            )

        # The distinct materials of the entities, each one dispatched to by its index (see 'add_material')
        self._materials: list[MPM.Base] = []

        # The particles stay a few cells away from the walls, where the stencil of a particle reaches outside the grid
        self.boundary_padding = 3 * self._dx

        self.mpm_state: MPMState | None = None
        self.mpm_info: MPMInfo | None = None
        self.mpm_config: MPMConfig | None = None
        self.particle_constraints: MPMParticleConstraints | None = None
        self.data_manager: MPMData | None = None
        self._mpm_materials: MPMMaterials | None = None
        self._is_constrained = False

    def build(self) -> None:
        super().build()

        if self.is_active and self._enable_CPIC and self._requires_grad:
            gs.raise_exception(
                "CPIC is not supported in differentiable mode yet. Submit a feature request if you need it."
            )

        # When no material reads U, S, V, the transfer to the grid takes J from the determinant of the advected
        # deformation gradient, which spares the singular value decomposition. The backward pass reverses it alone.
        # TODO: validate the J = det(F_tmp) shortcut composes with quadrants autograd before enabling it there.
        self.mpm_config = MPMConfig(
            is_svd_needed=self._requires_grad or any([material.needs_svd for material in self._materials]),
            requires_grad=self._requires_grad,
            is_cpic=self._enable_CPIC and self._sim.rigid_solver.is_active,
            is_deterministic=requires_deterministic_accumulation(),
        )
        self._mpm_materials = MPMMaterials(tuple(self._materials))

        self.data_manager = MPMData(self)
        self.mpm_state = self.data_manager.mpm_state
        self.mpm_info = self.data_manager.mpm_info
        self.particle_constraints = MPMParticleConstraints(unused=True)
        # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
        self.register_errors(
            self.data_manager.errno,
            {
                MPMErrorCode.INVALID_VELOCITY_NAN: lambda: (
                    """Invalid MPM particle velocities causing 'nan' or 'inf'.
Please decrease the simulation timestep or adjust the material parameters."""
                ),
            },
        )

        self._particles_pos = self.mpm_state.particles.pos
        self._particles_vel = self.mpm_state.particles.vel
        self._particles_is_active = self.mpm_state.particles.is_active
        self._particles_mass = self.mpm_info.particles.mass
        self._build_render(has_vverts=True)

        if self.is_active:
            for entity in self._entities:
                entity._add_to_solver()

            # See: https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L84
            suggested_dt = 2e-2 * self._dx
            if self.substep_dt > suggested_dt:
                gs.logger.warning(
                    f"""Current `substep_dt` ({self.substep_dt:.6g}) is greater than suggested_dt
({suggested_dt:.6g}, calculated based on `grid_density`). Simulation might be unstable."""
                )

        self._build_gravity(self.mpm_info.gravity)

    def reset_grad(self) -> None:
        if self._requires_grad:
            qd_zero_grad(self.mpm_state.particles)
            qd_zero_grad(self.mpm_state.grid)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "MPMEntity":
        self.add_material(material)

        entity = MPMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )
        self._entities.append(entity)
        return entity

    def add_material(self, material: MPM.Base) -> None:
        """Give a material its index among the distinct materials of the solver, registering it unless one equals it.

        A material equal to one already registered shares its index, and so the branches of the kernels it compiles.
        """
        for material_registered in self._materials:
            if material == material_registered:
                material.idx = material_registered.idx
                return
        material.idx = len(self._materials)
        self._materials.append(material)

    def add_particles(self, is_active: bool, particle_start: int, material: MPM.Base, pos: np.ndarray) -> None:
        """Add the particles of an entity at rest, with the parameters of its material, between two steps."""
        kernel_add_particles(
            particle_start,
            material.idx,
            is_active,
            material._default_Jp,
            material.rho,
            pos,
            self.mpm_state,
            self.mpm_info,
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self) -> None:
        if self._requires_grad:
            kernel_reset_grid_and_grad(self.mpm_state)
            kernel_compute_F_tmp(self.mpm_state, self.mpm_info, compute_svd=False)
            kernel_svd(self.mpm_state, self.mpm_info)
        else:
            # The sparse reset at the end of the previous substep leaves the grid zeroed for this one
            kernel_compute_F_tmp(self.mpm_state, self.mpm_info, self.mpm_config.is_svd_needed)
        rigid_solver = self._sim.rigid_solver
        kernel_transfer_particles_to_grid(
            self.mpm_state,
            rigid_solver.dyn_state.geoms,
            self.mpm_info,
            rigid_solver.dyn_info.geoms,
            rigid_solver.rigid_info,
            rigid_solver.collider.sdf.sdf_info,
            self.mpm_config,
            rigid_solver.collider.collider_config,
            self._mpm_materials,
            is_backward=False,
        )

    def substep_post_coupling(self) -> None:
        rigid_solver = self._sim.rigid_solver
        # With gradient, the advection of the particles reads back the velocity they gathered, which the backward pass
        # of the gather reverses alone (see kernel_transfer_grid_to_particles_grad)
        if self._requires_grad:
            kernel_transfer_grid_to_particles(
                self.mpm_state,
                rigid_solver.dyn_state.links,
                self.mpm_info,
                rigid_solver.dyn_info.geoms,
                rigid_solver.rigid_info,
                self.mpm_config,
            )
            kernel_advect_particles(self.mpm_state, self.mpm_info, self._errno)
        else:
            kernel_integrate_particles(
                self.mpm_state,
                rigid_solver.dyn_state.links,
                self.mpm_info,
                rigid_solver.dyn_info.geoms,
                rigid_solver.rigid_info,
                self.mpm_config,
                self._errno,
            )

        # The particle constraints apply once the particles left the grid
        if self._is_constrained:
            kernel_apply_particle_constraints(
                self.mpm_state, rigid_solver.dyn_state.links, self.particle_constraints, self.mpm_info
            )

        # The grid is not read after the integration of the particles, so the cells the substep touched are zeroed now,
        # across all envs, and the list of them emptied (see MPMGrid).
        if not self._requires_grad:
            kernel_reset_dirty_cells(self.mpm_state)
            if gs.use_zerocopy:
                dirty_count = qd_to_torch(self.mpm_state.grid.dirty_count, copy=False)
                dirty_count.zero_()
                # The next kernel reads the count (see qd_zero_grad in misc.py)
                if gs.backend == gs.metal:
                    torch.mps.synchronize()
            else:
                kernel_clear_dirty_count(self.mpm_state)

        kernel_copy_next_to_curr(self.mpm_state, self._errno)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_differentiable(self) -> bool:
        return not self._enable_CPIC

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        yield from iter_data(self, "mpm_config")
        yield from iter_data(self.data_manager, "mpm_info")
        yield from iter_data(self, "particle_constraints")
        yield from iter_data(self.data_manager, "mpm_state")
        yield from iter_data(self.data_manager, "errno", DataKind.STATE)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def _reset_particles(self, particles_idx: torch.Tensor, envs_idx: torch.Tensor, reset_deformation: bool) -> None:
        if reset_deformation:
            kernel_reset_particles_deformation(particles_idx, envs_idx, self.mpm_state, self.mpm_info)

    @gs.assert_built
    @mutates()
    def set_particles_actu(self, actus, particles_idx=None, envs_idx=None, *, n_groups: int = 1) -> None:
        """
        Set the actuation of some particles, each particle taking the actuation of its muscle group.

        Parameters
        ----------
        actus : array_like, shape ([M,] [N,] [n_groups,])
            Actuation of each muscle group, for each particle.
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles (see 'set_particles_pos'). If None, all particles will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        n_groups : int, optional
            The number of muscle groups. Defaults to 1.
        """
        actus, particles_idx, envs_idx = self._sanitize_io_variables(
            actus, particles_idx, self.n_particles, "particles_idx", envs_idx, (n_groups,), batched_idx=True
        )
        kernel_set_actu(
            particles_idx, envs_idx, actus, self.mpm_info.particles.muscle_group, self.mpm_state.particles.actu
        )

    def get_particles_actu(self, particles_idx=None, envs_idx=None) -> torch.Tensor:
        """Get the actuation of some particles (see 'set_particles_actu')."""
        tensor = self._read_data(self.mpm_state.particles.actu, envs_idx, particles_idx, copy=True)
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @mutates(StateChange.MODEL)
    def set_particles_muscle_group(self, muscle_group, particles_idx=None) -> None:
        """Set the muscle group of some particles, shared by every environment."""
        self._write_data(
            self.mpm_info.particles.muscle_group,
            muscle_group,
            particles_idx,
            self.n_particles,
            "particles_idx",
            dtype=gs.tc_int,
            is_batched=False,
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def get_particles_muscle_group(self, particles_idx=None) -> torch.Tensor:
        """Get the muscle group of some particles (see 'set_particles_muscle_group')."""
        return self._read_data(self.mpm_info.particles.muscle_group, None, particles_idx, copy=True, is_batched=False)

    @mutates(StateChange.MODEL)
    def set_particles_muscle_direction(self, muscle_direction, particles_idx=None) -> None:
        """Set the unit direction of the muscle fibers of some particles, shared by every environment."""
        self._write_data(
            self.mpm_info.particles.muscle_direction,
            muscle_direction,
            particles_idx,
            self.n_particles,
            "particles_idx",
            element_shape=(3,),
            is_batched=False,
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    @mutates(StateChange.MODEL)
    def set_particles_free(self, free, particles_idx=None) -> None:
        """Set whether some particles move freely or stay fixed, shared by every environment."""
        self._write_data(
            self.mpm_info.particles.is_free,
            free,
            particles_idx,
            self.n_particles,
            "particles_idx",
            dtype=gs.tc_bool,
            is_batched=False,
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def get_particles_free(self, particles_idx=None) -> torch.Tensor:
        """Get whether some particles move freely (see 'set_particles_free')."""
        return self._read_data(self.mpm_info.particles.is_free, None, particles_idx, copy=True, is_batched=False)

    # ------------------------------------------------------------------------------------
    # -------------------------------- particle constraints ------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_particle_constraints(
        self, particles_mask, link_idx: int, stiffness: float, *, particles_start: int = 0
    ) -> None:
        """
        Attach some particles to a rigid link with critically damped springs.

        Each particle is pulled toward the position it holds now, expressed in the frame of the link. A spring applies the
        force -k (x - x_t) - 2 sqrt(k m) (v - v_t) to a particle of mass m, x_t and v_t being the position and velocity
        of its target.

        Parameters
        ----------
        particles_mask : array_like, shape ([n_envs,] n_particles)
            Boolean mask over consecutive particles of the solver, selecting those to attach.
        link_idx : int
            Global index of the rigid link.
        stiffness : float
            Stiffness of the springs.
        particles_start : int, optional
            Global index of the particle the mask starts at. Defaults to 0.
        """
        if not isinstance(link_idx, int):
            gs.raise_exception("link_idx must be an integer.")
        if not self._is_constrained:
            self._init_particle_constraints()
        particles_mask = self._broadcast_particles_mask(particles_mask, particles_start)
        rigid_solver = self._sim.rigid_solver
        # The attachment is a configuration of the constraint, which carries no gradient
        with torch.no_grad():
            link_pos = rigid_solver.get_links_pos(links_idx=link_idx)[..., 0, :]
            link_quat = rigid_solver.get_links_quat(links_idx=link_idx)[..., 0, :]
        link_pos = broadcast_tensor(link_pos, gs.tc_float, (self._B, 3)).contiguous()
        link_quat = broadcast_tensor(link_quat, gs.tc_float, (self._B, 4)).contiguous()
        kernel_set_particle_constraints(
            particles_start,
            link_idx,
            stiffness,
            particles_mask,
            link_pos,
            link_quat,
            self.mpm_state,
            self.particle_constraints,
        )

    def _broadcast_particles_mask(self, particles_mask, particles_start: int) -> torch.Tensor:
        # A missing mask selects every particle from `particles_start` on
        n_particles = -1
        if particles_mask is None:
            particles_mask = True
            n_particles = self.n_particles - particles_start
        particles_mask = broadcast_tensor(
            particles_mask, gs.tc_bool, (self._B, n_particles), ("envs_idx", "particles_idx")
        ).contiguous()
        if particles_start < 0 or particles_start + particles_mask.shape[1] > self.n_particles:
            gs.raise_exception("The particle mask overruns the particles of the solver.")
        return particles_mask

    def _init_particle_constraints(self) -> None:
        # Allocated once some particles are constrained, which most scenes never do
        self.particle_constraints = MPMParticleConstraints(
            sizes=dict(particles=self._n_particles, envs=self._B), env_first=False
        )
        kernel_clear_particle_constraints(self.particle_constraints)
        self._is_constrained = True

    def load_checkpoint(
        self,
        checkpoint: SolverCheckpoint,
        *,
        boundary_only: bool = False,
        envs_idx: int | Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        # The constraints are allocated on first use, so a record holds them only if they existed when it was made: a
        # record holding them allocates them before they are written back, and one without them leaves none.
        is_constrained_record = checkpoint.arrays.get("particle_constraints.is_constrained")
        if is_constrained_record is not None and is_constrained_record.ndim > 0 and not self._is_constrained:
            self._init_particle_constraints()
        super().load_checkpoint(checkpoint, boundary_only=boundary_only, envs_idx=envs_idx)
        if is_constrained_record is not None and is_constrained_record.ndim == 0 and self._is_constrained:
            kernel_clear_particle_constraints(self.particle_constraints)

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def remove_particle_constraints(self, particles_mask=None, *, particles_start: int = 0) -> None:
        """
        Detach some particles from the rigid link they are attached to (see 'set_particle_constraints').

        Parameters
        ----------
        particles_mask : None | array_like, shape ([n_envs,] n_particles), optional
            Boolean mask over consecutive particles of the solver, selecting those to detach. If None, all particles
            from `particles_start` on will be detached. Defaults to None.
        particles_start : int, optional
            Global index of the particle the mask starts at. Defaults to 0.
        """
        if not self._is_constrained:
            return
        particles_mask = self._broadcast_particles_mask(particles_mask, particles_start)
        kernel_remove_particle_constraints(particles_start, particles_mask, self.particle_constraints)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def particle_mass_scale(self) -> float:
        return self._particle_volume_scale

    @property
    def grid_density(self) -> float:
        return self._grid_density

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def inv_dx(self) -> float:
        return self._inv_dx

    @property
    def particle_volume_real(self) -> float:
        return self._particle_volume_real

    @property
    def particle_volume(self) -> float:
        return self._particle_volume

    @property
    def particle_volume_scale(self) -> float:
        return self._particle_volume_scale

    @property
    def lower_bound_cell(self) -> np.ndarray:
        return self._lower_bound_cell

    @property
    def upper_bound_cell(self) -> np.ndarray:
        return self._upper_bound_cell

    @property
    def grid_res(self) -> np.ndarray:
        return self._grid_res

    @property
    def grid_offset(self) -> qd.Vector:
        return self._grid_offset

    @property
    def enable_CPIC(self) -> bool:
        return self._enable_CPIC


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_quadratic_bspline_stencil(pos: qd.types.vector(3), mpm_info: MPMInfo):
    """
    Return the lowest cell of the stencil of a position, its offset from it in cells, and the weights of the stencil.

    The weights are those of the quadratic B-spline along each axis, one row per cell offset 0, 1, 2 of the stencil.
    """
    inv_dx = mpm_info.inv_dx[None]
    base = qd.floor(pos * inv_dx - 0.5).cast(gs.qd_int)
    fx = pos * inv_dx - base.cast(gs.qd_float)
    w = qd.Matrix.rows([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2])
    return base, fx, w


@qd.kernel(fastcache=True)
def kernel_compute_F_tmp(mpm_state: MPMState, mpm_info: MPMInfo, compute_svd: qd.template()):
    """
    Advect the deformation gradient of the active particles over the substep, by their affine velocity field.

    With `compute_svd`, its singular value decomposition follows in the same pass, which keeps the advected deformation
    gradient in registers. The backward pass reverses the decomposition apart, through 'kernel_svd'.
    """
    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.F_tmp))):
        if mpm_state.particles.is_active[i_p, i_b]:
            F_tmp = (
                qd.Matrix.identity(gs.qd_float, 3) + mpm_info.substep_dt[None] * mpm_state.particles.C[i_p, i_b]
            ) @ mpm_state.particles.F[i_p, i_b]
            mpm_state.particles.F_tmp[i_p, i_b] = F_tmp
            if qd.static(compute_svd):
                U, S, V = gu.qd_svd(F_tmp)
                mpm_state.particles.U[i_p, i_b] = U
                mpm_state.particles.S[i_p, i_b] = S
                mpm_state.particles.V[i_p, i_b] = V


@qd.kernel(fastcache=True)
def kernel_svd_grad(mpm_state: MPMState, mpm_info: MPMInfo):
    """Reverse the singular value decomposition of the substep, consuming the adjoint of its factors like autodiff."""
    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.F_tmp.grad))):
        if mpm_state.particles.is_active[i_p, i_b]:
            mpm_state.particles.F_tmp.grad[i_p, i_b] = mpm_state.particles.F_tmp.grad[i_p, i_b] + func_backward_svd(
                mpm_state.particles.U.grad[i_p, i_b],
                mpm_state.particles.S.grad[i_p, i_b],
                mpm_state.particles.V.grad[i_p, i_b],
                mpm_state.particles.U[i_p, i_b],
                mpm_state.particles.S[i_p, i_b],
                mpm_state.particles.V[i_p, i_b],
                mpm_info.EPS[None],
            )
            mpm_state.particles.U.grad[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            mpm_state.particles.S.grad[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            mpm_state.particles.V.grad[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)


# The decomposition is reversed in closed form, which autodiff cannot derive. The backward takes the arguments of the
# launch, the info it reads included.
@register_custom_backward(kernel_svd_grad)
@qd.kernel(fastcache=True)
def kernel_svd(mpm_state: MPMState, mpm_info: MPMInfo):
    """Compute the singular value decomposition of the advected deformation gradient of the active particles."""
    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.U))):
        if mpm_state.particles.is_active[i_p, i_b]:
            U, S, V = gu.qd_svd(mpm_state.particles.F_tmp[i_p, i_b])
            mpm_state.particles.U[i_p, i_b] = U
            mpm_state.particles.S[i_p, i_b] = S
            mpm_state.particles.V[i_p, i_b] = V


@reversed_with(is_backward=True)
@qd.kernel
def kernel_transfer_particles_to_grid(
    mpm_state: MPMState,
    geoms_state: GeomsState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    mpm_config: qd.template(),
    collider_config: qd.template(),
    mpm_materials: qd.template(),
    is_backward: qd.template(),
):
    """
    Update the deformation of the active particles, then transfer their momentum and mass to the grid.

    The material of each particle updates its deformation gradient and plastic volume ratio into the next state and
    gives its stress, which enters the momentum. Under compatible particle-in-cell (CPIC), a cell separated from a
    particle by a rigid geom gets nothing from it. A cell in the stencil of a particle held in place gets no momentum,
    as a boundary condition of the grid. Without gradient, the cells touched are listed for the sparse reset (see
    MPMGrid).

    The particles scatter into the cells with atomic additions, whose order follows the threads. With
    `is_deterministic` (see MPMConfig), the particles are sorted instead, and each cell gathers them in a fixed order
    (see func_gather_particles), which gives the same sums bit for bit from one run to the next and across the
    environments holding the same particles. The variant autodiff reverses (`is_backward`) scatters, whose reverse
    accumulates the adjoint of each particle in its own thread.
    """
    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.F))):
        if mpm_state.particles.is_active[i_p, i_b]:
            # The determinant of F_tmp equals that of S, since U and V of the singular value decomposition are proper
            # rotations (see MPMConfig). J is declared before the branches, which quadrants scopes locally.
            J = gs.qd_float(0.0)
            if qd.static(mpm_config.is_svd_needed):
                J = mpm_state.particles.S[i_p, i_b].determinant()
            else:
                J = mpm_state.particles.F_tmp[i_p, i_b].determinant()
            i_mat = mpm_info.particles.material_idx[i_p]
            F_new, S_new, Jp_new = mpm_materials.update_F_S_Jp(
                i_mat,
                J,
                mpm_state.particles.F_tmp[i_p, i_b],
                mpm_state.particles.U[i_p, i_b],
                mpm_state.particles.S[i_p, i_b],
                mpm_state.particles.V[i_p, i_b],
                mpm_state.particles.Jp[i_p, i_b],
                mpm_info.EPS[None],
            )
            mpm_state.particles.F_next[i_p, i_b] = F_new
            mpm_state.particles.Jp_next[i_p, i_b] = Jp_new

            # The stress takes both the advected deformation gradient and the one once plasticity applies: the elastic
            # materials read the latter, while a viscous liquid reads the former, whose shear the plasticity removes.
            # The plastic volume ratio is the one the substep starts from, which the snow alone reads.
            stress = mpm_materials.update_stress(
                i_mat,
                mpm_state.particles.U[i_p, i_b],
                S_new,
                mpm_state.particles.V[i_p, i_b],
                mpm_state.particles.F_tmp[i_p, i_b],
                F_new,
                J,
                mpm_state.particles.Jp[i_p, i_b],
                mpm_state.particles.actu[i_p, i_b],
                mpm_info.particles.muscle_direction[i_p],
            )
            mass = mpm_info.particles.mass[i_p]
            affine = mpm_info.stress_scale[None] * stress + mass * mpm_state.particles.C[i_p, i_b]
            if qd.static(mpm_config.is_deterministic and not is_backward):
                mpm_state.particles.affine[i_p, i_b] = affine

            if qd.static(mpm_config.is_cpic or not mpm_config.is_deterministic or is_backward):
                base, fx, w = func_quadratic_bspline_stencil(mpm_state.particles.pos[i_p, i_b], mpm_info)
                grid_offset = mpm_info.grid_offset[None]
                for offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                    i_g_sep = -1
                    if qd.static(mpm_config.is_cpic):
                        # The cell is separated from the particle by the first geom whose sides they lie on differ
                        cell_pos = (base + offset) * mpm_info.dx[None]
                        for i_g in range(mpm_state.particles_cpic.rigid_normal.shape[1]):
                            if i_g_sep == -1 and geoms_info.needs_coup[i_g]:
                                sdf_normal_cell = sdf.sdf_func_normal_world(
                                    i_g, i_b, cell_pos, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
                                )
                                if mpm_state.particles_cpic.rigid_normal[i_p, i_g, i_b].dot(sdf_normal_cell) < 0:
                                    i_g_sep = i_g
                        mpm_state.particles_cpic.cell_geom_idx[i_p, offset[0], offset[1], offset[2], i_b] = i_g_sep
                    if qd.static(not mpm_config.is_deterministic or is_backward):
                        if i_g_sep == -1:
                            dpos = (offset.cast(gs.qd_float) - fx) * mpm_info.dx[None]
                            weight = gs.qd_float(1.0)
                            for d in qd.static(range(3)):
                                weight *= w[offset[d], d]
                            cell = base - grid_offset + offset
                            qd.atomic_add(
                                mpm_state.grid.vel_in[cell[0], cell[1], cell[2], i_b],
                                weight * (mass * mpm_state.particles.vel[i_p, i_b] + affine @ dpos),
                            )
                            mass_contrib = weight * mass
                            mass_prev = qd.atomic_add(mpm_state.grid.mass[cell[0], cell[1], cell[2], i_b], mass_contrib)
                            # The masses are non-negative, so the first contribution of an env to a cell claims the cell
                            # in the dirty flag shared by the envs, and the first env claiming it lists it (see MPMGrid)
                            if qd.static(not mpm_config.requires_grad):
                                if mass_prev <= 0.0 and mass_contrib > 0.0:
                                    if qd.atomic_or(mpm_state.grid.dirty_flag[cell[0], cell[1], cell[2]], 1) == 0:
                                        i_slot = qd.atomic_add(mpm_state.grid.dirty_count[0], 1)
                                        grid_y, grid_z = mpm_state.grid.mass.shape[1], mpm_state.grid.mass.shape[2]
                                        i_cell = (cell[0] * grid_y + cell[1]) * grid_z + cell[2]
                                        mpm_state.grid.dirty_list[i_slot] = i_cell

    if qd.static(mpm_config.is_deterministic and not is_backward):
        func_sort_particles(mpm_state, mpm_info, mpm_config)
        func_gather_particles(mpm_state, mpm_info, mpm_config, is_g2p_backward=False)
    else:
        # The particles held in place act as boundary conditions of the grid, zeroing the cells of their stencils once
        # every particle scattered into them, as the gather does (see func_gather_particles)
        for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.F))):
            if mpm_state.particles.is_active[i_p, i_b] and not mpm_info.particles.is_free[i_p]:
                base, fx, w = func_quadratic_bspline_stencil(mpm_state.particles.pos[i_p, i_b], mpm_info)
                for offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                    cell = base - mpm_info.grid_offset[None] + offset
                    mpm_state.grid.vel_in[cell[0], cell[1], cell[2], i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_sort_particles(mpm_state: MPMState, mpm_info: MPMInfo, mpm_config: qd.template()):
    """
    Sort the active particles by the lowest cell of their stencil, and list the tiles their stencils cover.

    The particles of a slot of the hash grid follow each other in ascending order of index, the cells sharing a slot
    telling their particles apart by `cell_idx`. Called at the top level of a kernel, whose loops it runs in parallel
    (see MPMTiles).
    """
    # The cells per side of a tile of the grid, and the environments the lanes of a warp gather a cell for
    TILE_SIZE = qd.static(4)
    ENVS_GROUP_SIZE = qd.static(32)
    grid_res = qd.Vector([mpm_state.grid.mass.shape[0], mpm_state.grid.mass.shape[1], mpm_state.grid.mass.shape[2]])
    tiles_res = (grid_res + TILE_SIZE - 1) // TILE_SIZE

    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.pos))):
        i_s = -1
        i_cell = -1
        if mpm_state.particles.is_active[i_p, i_b]:
            base, fx, w = func_quadratic_bspline_stencil(mpm_state.particles.pos[i_p, i_b], mpm_info)
            cell = base - mpm_info.grid_offset[None]
            i_cell = (cell[0] * grid_res[1] + cell[1]) * grid_res[2] + cell[2]
            i_s = i_cell % mpm_state.hash_grid.slot_size.shape[0]
            tile = cell // TILE_SIZE
            i_t = (tile[0] * tiles_res[1] + tile[1]) * tiles_res[2] + tile[2]
            mpm_state.tiles.is_occupied[i_t, i_b // ENVS_GROUP_SIZE] = True
        mpm_state.hash_grid.particles_slot[i_p, i_b] = i_s
        mpm_state.particles.cell_idx[i_p, i_b] = i_cell
        if i_p == 0 and i_b == 0:
            mpm_state.tiles.n_active[0] = 0

    func_sort_particles_by_slot(mpm_state.hash_grid)

    # A tile is active when it or a tile below it along some axes holds the lowest cell of a stencil
    for i_t, i_g in qd.ndrange(*mpm_state.tiles.is_occupied.shape):
        tile = qd.Vector([i_t // (tiles_res[1] * tiles_res[2]), i_t // tiles_res[2] % tiles_res[1], i_t % tiles_res[2]])
        is_active = False
        for i_src in range(8):
            tile_src = tile - qd.Vector([i_src // 4, i_src // 2 % 2, i_src % 2])
            if (tile_src >= 0).all():
                i_t_src = (tile_src[0] * tiles_res[1] + tile_src[1]) * tiles_res[2] + tile_src[2]
                if mpm_state.tiles.is_occupied[i_t_src, i_g]:
                    is_active = True
        if is_active:
            i_a = qd.atomic_add(mpm_state.tiles.n_active[0], 1)
            mpm_state.tiles.active_tile[i_a] = i_t
            mpm_state.tiles.active_envs_group[i_a] = i_g

    for i_t, i_g in qd.ndrange(*mpm_state.tiles.is_occupied.shape):
        mpm_state.tiles.is_occupied[i_t, i_g] = False


@qd.func
def func_gather_particles(
    mpm_state: MPMState, mpm_info: MPMInfo, mpm_config: qd.template(), is_g2p_backward: qd.template()
):
    """
    Gather into each cell of the active tiles the mass and momentum of the particles whose stencil covers it.

    With `is_g2p_backward`, the cells add the adjoint of their velocity instead, which the particles give them through
    the adjoint of the velocity and affine velocity field they gathered (see kernel_transfer_grid_to_particles_grad).

    A thread sums the particles of one cell and environment, cell of its stencil by cell, in ascending order of index
    within each of them. The lanes of a warp hold the same cell for consecutive environments, which read the same
    particles at neighbouring addresses while the environments hold them in the same cells. Called at the top level of
    a kernel, after func_sort_particles.
    """
    # See TILE_SIZE and ENVS_GROUP_SIZE in 'func_sort_particles'
    TILE_SIZE = qd.static(4)
    ENVS_GROUP_SIZE = qd.static(32)
    N_TILE_CELLS = qd.static(TILE_SIZE**3)
    _B = mpm_state.particles.pos.shape[1]
    grid_res = qd.Vector([mpm_state.grid.mass.shape[0], mpm_state.grid.mass.shape[1], mpm_state.grid.mass.shape[2]])
    tiles_res = (grid_res + TILE_SIZE - 1) // TILE_SIZE
    for i_flat in range(mpm_state.tiles.n_active[0] * N_TILE_CELLS * ENVS_GROUP_SIZE):
        i_a = i_flat // (ENVS_GROUP_SIZE * N_TILE_CELLS)
        i_c = i_flat // ENVS_GROUP_SIZE % N_TILE_CELLS
        i_b = mpm_state.tiles.active_envs_group[i_a] * ENVS_GROUP_SIZE + i_flat % ENVS_GROUP_SIZE
        i_t = mpm_state.tiles.active_tile[i_a]
        tile = qd.Vector([i_t // (tiles_res[1] * tiles_res[2]), i_t // tiles_res[2] % tiles_res[1], i_t % tiles_res[2]])
        cell = tile * TILE_SIZE + qd.Vector(
            [i_c // (TILE_SIZE * TILE_SIZE), i_c // TILE_SIZE % TILE_SIZE, i_c % TILE_SIZE]
        )
        if i_b < _B and (cell < grid_res).all():
            mass_cell = gs.qd_float(0.0)
            momentum = qd.Vector.zero(gs.qd_float, 3)
            is_fixed = False
            for i_o in range(27):
                offset = qd.Vector([i_o // 9, i_o // 3 % 3, i_o % 3])
                base = cell - offset
                if (base >= 0).all():
                    i_cell = (base[0] * grid_res[1] + base[1]) * grid_res[2] + base[2]
                    i_s = i_cell % mpm_state.hash_grid.slot_size.shape[0]
                    i_r_start = mpm_state.hash_grid.slot_start[i_s, i_b]
                    for i_r in range(i_r_start, i_r_start + mpm_state.hash_grid.slot_size[i_s, i_b]):
                        i_p = mpm_state.hash_grid.particles_idx[i_r, i_b]
                        if mpm_state.particles.cell_idx[i_p, i_b] == i_cell:
                            _, fx, w = func_quadratic_bspline_stencil(mpm_state.particles.pos[i_p, i_b], mpm_info)
                            weight = gs.qd_float(1.0)
                            for d in qd.static(range(3)):
                                weight *= qd.select(
                                    offset[d] == 0, w[0, d], qd.select(offset[d] == 1, w[1, d], w[2, d])
                                )
                            if qd.static(is_g2p_backward):
                                dpos = offset.cast(gs.qd_float) - fx
                                momentum += weight * (
                                    mpm_state.particles.vel_tmp.grad[i_p, i_b]
                                    + 4 * mpm_info.inv_dx[None] * (mpm_state.particles.C_next.grad[i_p, i_b] @ dpos)
                                )
                            else:
                                is_separated = False
                                if qd.static(mpm_config.is_cpic):
                                    i_g_sep = mpm_state.particles_cpic.cell_geom_idx[
                                        i_p, offset[0], offset[1], offset[2], i_b
                                    ]
                                    is_separated = i_g_sep != -1
                                if not is_separated:
                                    mass = mpm_info.particles.mass[i_p]
                                    dpos = (offset.cast(gs.qd_float) - fx) * mpm_info.dx[None]
                                    momentum += weight * (
                                        mass * mpm_state.particles.vel[i_p, i_b]
                                        + mpm_state.particles.affine[i_p, i_b] @ dpos
                                    )
                                    mass_cell += weight * mass
                                if not mpm_info.particles.is_free[i_p]:
                                    is_fixed = True
            if qd.static(is_g2p_backward):
                mpm_state.grid.vel_out.grad[cell[0], cell[1], cell[2], i_b] = (
                    mpm_state.grid.vel_out.grad[cell[0], cell[1], cell[2], i_b] + momentum
                )
            else:
                if is_fixed:
                    momentum = qd.Vector.zero(gs.qd_float, 3)
                mpm_state.grid.mass[cell[0], cell[1], cell[2], i_b] = mass_cell
                mpm_state.grid.vel_in[cell[0], cell[1], cell[2], i_b] = momentum
                # The first env holding mass in a cell lists it (see MPMGrid)
                if qd.static(not mpm_config.requires_grad):
                    if mass_cell > 0.0:
                        if qd.atomic_or(mpm_state.grid.dirty_flag[cell[0], cell[1], cell[2]], 1) == 0:
                            i_slot = qd.atomic_add(mpm_state.grid.dirty_count[0], 1)
                            i_cell = (cell[0] * grid_res[1] + cell[1]) * grid_res[2] + cell[2]
                            mpm_state.grid.dirty_list[i_slot] = i_cell


@qd.func
def func_point_vel_after_collision_with_rigid_geom(
    i_g: int,
    vel: qd.types.vector(3),
    vel_rigid: qd.types.vector(3),
    normal_rigid: qd.types.vector(3),
    influence: float,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
):
    """
    Return the velocity of a point bouncing off a rigid geom, and whether it moves into the geom.

    The velocity relative to the geom loses its normal part by the restitution and its tangential part by the friction
    of the geom, `influence` blending the result with the velocity the point had.
    """
    # v w.r.t rigid
    rvel = vel - vel_rigid
    # Negative if inward
    rvel_normal_magnitude = rvel.dot(normal_rigid)

    is_colliding = rvel_normal_magnitude < 0
    if is_colliding:
        # tangential component
        rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
        rvel_tan_norm = rvel_tan.norm(rigid_info.EPS[None])

        # tangential component after friction
        rvel_tan = (
            rvel_tan / rvel_tan_norm * qd.max(0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[i_g])
        )

        # normal component after collision
        rvel_normal = -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[i_g]

        # normal + tangential component
        rvel_new = rvel_tan + rvel_normal

        # apply influence
        vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)
    return vel, is_colliding


@qd.func
def func_point_vel_after_two_way_collision_with_rigid_geom(
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    normal_rigid: qd.types.vector(3),
    influence: float,
    links_state: LinksState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
):
    """
    Return the velocity of a point of the given mass colliding with a rigid geom, and the force its momentum change
    applies the link of the geom, if it collides.

    The normal of the geom and its influence at the point are given (see
    'func_point_vel_after_collision_with_rigid_geom').
    """
    vel_rigid = func_vel_at_point(geoms_info.link_idx[i_g], i_b, pos_world, links_state)
    vel_new, is_colliding = func_point_vel_after_collision_with_rigid_geom(
        i_g, vel, vel_rigid, normal_rigid, influence, geoms_info, rigid_info
    )
    force = -mass * (vel_new - vel) / rigid_info.substep_dt[None]
    return vel_new, force, is_colliding


@qd.func
def func_add_particle_vel(i_p: int, i_b: int, vel: qd.types.vector(3), mpm_state: MPMState):
    """Add a velocity to particle i_p of env i_b, as another coupling drives it."""
    mpm_state.particles.vel[i_p, i_b] = mpm_state.particles.vel[i_p, i_b] + vel


@qd.func
def func_gather_grid_velocity(
    i_p: int,
    i_b: int,
    mpm_state: MPMState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
):
    """
    Return the velocity and the affine velocity field an active particle gathers from the cells of its stencil.

    Under compatible particle-in-cell (CPIC), a cell separated from the particle by a rigid geom gives it the velocity
    of the collision of the particle with the geom instead, whose reaction pushes the link of the geom (see
    func_add_external_wrench).
    """
    inv_dx = mpm_info.inv_dx[None]
    base, fx, w = func_quadratic_bspline_stencil(mpm_state.particles.pos[i_p, i_b], mpm_info)
    vel = qd.Vector.zero(gs.qd_float, 3)
    C = qd.Matrix.zero(gs.qd_float, 3, 3)
    # A runtime loop over the cells of the stencil keeps a single inlined copy of the collision with the rigid geoms
    for i_o in range(27):
        offset = qd.Vector([i_o // 9, i_o // 3 % 3, i_o % 3])
        dpos = offset.cast(gs.qd_float) - fx
        cell = base - mpm_info.grid_offset[None] + offset
        grid_vel = mpm_state.grid.vel_out[cell[0], cell[1], cell[2], i_b]
        weight = gs.qd_float(1.0)
        for d in qd.static(range(3)):
            weight *= qd.select(offset[d] == 0, w[0, d], qd.select(offset[d] == 1, w[1, d], w[2, d]))

        if qd.static(mpm_config.is_cpic):
            i_g_sep = mpm_state.particles_cpic.cell_geom_idx[i_p, offset[0], offset[1], offset[2], i_b]
            if i_g_sep != -1:
                pos = mpm_state.particles.pos[i_p, i_b]
                grid_vel, force, is_colliding = func_point_vel_after_two_way_collision_with_rigid_geom(
                    i_g_sep,
                    i_b,
                    pos,
                    mpm_state.particles.vel[i_p, i_b],
                    mass=mpm_info.particles.mass[i_p] * weight / mpm_info.particle_volume_scale[None],
                    normal_rigid=mpm_state.particles_cpic.rigid_normal[i_p, i_g_sep, i_b],
                    influence=1.0,
                    links_state=links_state,
                    geoms_info=geoms_info,
                    rigid_info=rigid_info,
                )
                if is_colliding:
                    func_add_external_wrench(
                        geoms_info.link_idx[i_g_sep],
                        i_b,
                        pos,
                        force,
                        torque=qd.Vector.zero(gs.qd_float, 3),
                        links_state=links_state,
                        is_exact=mpm_config.is_deterministic,
                    )

        vel += weight * grid_vel
        C += 4 * inv_dx * weight * grid_vel.outer_product(dpos)
    return vel, C


@qd.func
def func_advect_particle(
    i_p: int, i_b: int, vel: qd.types.vector(3), mpm_state: MPMState, mpm_info: MPMInfo, errno: qd.Tensor
):
    """Advect an active particle by the velocity it gathered from the grid into the next state, within the boundary."""
    # A velocity that is not finite flags its env, whose state kernel_copy_next_to_curr leaves as it was
    if not qd_isfinite(vel):
        errno[i_b] = errno[i_b] | MPMErrorCode.INVALID_VELOCITY_NAN

    pos = mpm_state.particles.pos[i_p, i_b] + mpm_info.substep_dt[None] * vel

    # The boundary keeps an exploding simulation from reading cells out of the grid
    pos, vel = func_impose_cube_boundary(pos, vel, mpm_info.boundary)

    mpm_state.particles.vel_next[i_p, i_b] = vel
    mpm_state.particles.pos_next[i_p, i_b] = pos


@qd.kernel(fastcache=True)
def kernel_integrate_particles(
    mpm_state: MPMState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
    errno: qd.Tensor,
):
    """
    Gather the velocity of the grid back to the active particles and advect them into the next state.

    The inactive particles carry their state unchanged to the next one.
    """
    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.is_active))):
        if mpm_state.particles.is_active[i_p, i_b]:
            vel, C = func_gather_grid_velocity(
                i_p, i_b, mpm_state, links_state, mpm_info, geoms_info, rigid_info, mpm_config
            )
            mpm_state.particles.C_next[i_p, i_b] = C
            func_advect_particle(i_p, i_b, vel, mpm_state, mpm_info, errno)
        else:
            mpm_state.particles.vel_next[i_p, i_b] = mpm_state.particles.vel[i_p, i_b]
            mpm_state.particles.pos_next[i_p, i_b] = mpm_state.particles.pos[i_p, i_b]
            mpm_state.particles.C_next[i_p, i_b] = mpm_state.particles.C[i_p, i_b]
            mpm_state.particles.F_next[i_p, i_b] = mpm_state.particles.F[i_p, i_b]
            mpm_state.particles.Jp_next[i_p, i_b] = mpm_state.particles.Jp[i_p, i_b]


@qd.kernel(fastcache=True)
def kernel_transfer_grid_to_particles_grad(
    mpm_state: MPMState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
):
    """
    Reverse the gather of the velocity of the grid, consuming the adjoint of what it wrote like autodiff.

    Each cell of the grid sums the adjoint of its velocity from the particles whose stencil covers it, gathered in a
    fixed order with `is_deterministic` (see func_gather_particles) and added by each particle otherwise. Each particle
    adds the adjoint of its position, through the weights of its stencil, in its own thread.
    """
    if qd.static(mpm_config.is_deterministic):
        func_sort_particles(mpm_state, mpm_info, mpm_config)
        func_gather_particles(mpm_state, mpm_info, mpm_config, is_g2p_backward=True)

    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.is_active))):
        if mpm_state.particles.is_active[i_p, i_b]:
            inv_dx = mpm_info.inv_dx[None]
            vel_grad = mpm_state.particles.vel_tmp.grad[i_p, i_b]
            C_grad = mpm_state.particles.C_next.grad[i_p, i_b]
            base, fx, w = func_quadratic_bspline_stencil(mpm_state.particles.pos[i_p, i_b], mpm_info)
            # The derivatives of the weights of the quadratic B-spline along each axis, one row per cell offset
            dw = qd.Matrix.rows([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5])
            fx_grad = qd.Vector.zero(gs.qd_float, 3)
            for offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                dpos = offset.cast(gs.qd_float) - fx
                cell = base - mpm_info.grid_offset[None] + offset
                grid_vel = mpm_state.grid.vel_out[cell[0], cell[1], cell[2], i_b]
                weight = gs.qd_float(1.0)
                for d in qd.static(range(3)):
                    weight *= w[offset[d], d]
                weight_grad = qd.Vector(
                    [
                        dw[offset[0], 0] * w[offset[1], 1] * w[offset[2], 2],
                        w[offset[0], 0] * dw[offset[1], 1] * w[offset[2], 2],
                        w[offset[0], 0] * w[offset[1], 1] * dw[offset[2], 2],
                    ]
                )
                C_grad_dpos = C_grad @ dpos
                fx_grad += (grid_vel.dot(vel_grad) + 4 * inv_dx * grid_vel.dot(C_grad_dpos)) * weight_grad
                fx_grad -= 4 * inv_dx * weight * (C_grad.transpose() @ grid_vel)
                if qd.static(not mpm_config.is_deterministic):
                    qd.atomic_add(
                        mpm_state.grid.vel_out.grad[cell[0], cell[1], cell[2], i_b],
                        weight * (vel_grad + 4 * inv_dx * C_grad_dpos),
                    )
            mpm_state.particles.pos.grad[i_p, i_b] = mpm_state.particles.pos.grad[i_p, i_b] + inv_dx * fx_grad
            mpm_state.particles.vel_tmp.grad[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        else:
            mpm_state.particles.C.grad[i_p, i_b] = (
                mpm_state.particles.C.grad[i_p, i_b] + mpm_state.particles.C_next.grad[i_p, i_b]
            )
            mpm_state.particles.F.grad[i_p, i_b] = (
                mpm_state.particles.F.grad[i_p, i_b] + mpm_state.particles.F_next.grad[i_p, i_b]
            )
            mpm_state.particles.Jp.grad[i_p, i_b] = (
                mpm_state.particles.Jp.grad[i_p, i_b] + mpm_state.particles.Jp_next.grad[i_p, i_b]
            )
            mpm_state.particles.F_next.grad[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            mpm_state.particles.Jp_next.grad[i_p, i_b] = 0.0
        mpm_state.particles.C_next.grad[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)


# The gather is reversed by hand, which reduces the adjoint of the cells of the grid in a fixed order where autodiff
# would add it with atomic additions. The backward takes the arguments of the launch.
@register_custom_backward(kernel_transfer_grid_to_particles_grad)
@qd.kernel(fastcache=True)
def kernel_transfer_grid_to_particles(
    mpm_state: MPMState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    mpm_config: qd.template(),
):
    """
    Gather the velocity of the grid back to the active particles, ahead of their advection (kernel_advect_particles).

    The inactive particles carry their affine velocity field, deformation and plastic volume ratio to the next state.
    """
    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.is_active))):
        if mpm_state.particles.is_active[i_p, i_b]:
            vel, C = func_gather_grid_velocity(
                i_p, i_b, mpm_state, links_state, mpm_info, geoms_info, rigid_info, mpm_config
            )
            mpm_state.particles.vel_tmp[i_p, i_b] = vel
            mpm_state.particles.C_next[i_p, i_b] = C
        else:
            mpm_state.particles.C_next[i_p, i_b] = mpm_state.particles.C[i_p, i_b]
            mpm_state.particles.F_next[i_p, i_b] = mpm_state.particles.F[i_p, i_b]
            mpm_state.particles.Jp_next[i_p, i_b] = mpm_state.particles.Jp[i_p, i_b]


@qd.kernel(fastcache=True)
def kernel_advect_particles(mpm_state: MPMState, mpm_info: MPMInfo, errno: qd.Tensor):
    """Advect the active particles by the velocity they gathered from the grid, the inactive ones staying in place."""
    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.is_active))):
        if mpm_state.particles.is_active[i_p, i_b]:
            func_advect_particle(i_p, i_b, mpm_state.particles.vel_tmp[i_p, i_b], mpm_state, mpm_info, errno)
        else:
            mpm_state.particles.vel_next[i_p, i_b] = mpm_state.particles.vel[i_p, i_b]
            mpm_state.particles.pos_next[i_p, i_b] = mpm_state.particles.pos[i_p, i_b]


@qd.kernel(fastcache=True)
def kernel_copy_next_to_curr(mpm_state: MPMState, errno: qd.Tensor):
    """
    Copy the state the substep integrated to the state of the particles, unless their velocities are not finite.

    The state of an env flagged with a velocity that is not finite stays as it was before the substep, so it stays
    finite. The flag holds until the error raises at the beginning of the next step, which freezes the environment for
    the rest of its step, as in the rigid solver.
    """
    for i_p, i_b in qd.ndrange(*mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles.pos))):
        if (errno[i_b] & MPMErrorCode.INVALID_VELOCITY_NAN) == 0:
            mpm_state.particles.pos[i_p, i_b] = mpm_state.particles.pos_next[i_p, i_b]
            mpm_state.particles.vel[i_p, i_b] = mpm_state.particles.vel_next[i_p, i_b]
            mpm_state.particles.C[i_p, i_b] = mpm_state.particles.C_next[i_p, i_b]
            mpm_state.particles.F[i_p, i_b] = mpm_state.particles.F_next[i_p, i_b]
            mpm_state.particles.Jp[i_p, i_b] = mpm_state.particles.Jp_next[i_p, i_b]


@no_grad
@qd.kernel(fastcache=True)
def kernel_reset_grid_and_grad(mpm_state: MPMState):
    """Zero the grid of a substep, its adjoint included."""
    for i_x, i_y, i_z, i_b in qd.ndrange(*mpm_state.grid.mass.shape):
        mpm_state.grid.mass[i_x, i_y, i_z, i_b] = 0.0
        mpm_state.grid.vel_in[i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)
        mpm_state.grid.vel_out[i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)
        mpm_state.grid.mass.grad[i_x, i_y, i_z, i_b] = 0.0
        mpm_state.grid.vel_in.grad[i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)
        mpm_state.grid.vel_out.grad[i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_reset_dirty_cells(mpm_state: MPMState):
    """
    Zero the cells the transfer to the grid touched, across all envs, with their dirty flags.

    The list of the cells is shared by the envs (see MPMGrid), so the threads past its length idle, and the first env
    alone clears the flag of a cell, for the next substep.
    """
    grid_y, grid_z = mpm_state.grid.mass.shape[1], mpm_state.grid.mass.shape[2]
    for i_slot, i_b in qd.ndrange(
        mpm_state.grid.dirty_list.shape[0], mpm_state.grid.mass.shape[3], axes=qd.static(eoa(mpm_state.grid.mass))
    ):
        if i_slot < mpm_state.grid.dirty_count[0]:
            i_cell = mpm_state.grid.dirty_list[i_slot]
            k = i_cell % grid_z
            j = i_cell // grid_z % grid_y
            i = i_cell // grid_z // grid_y
            mpm_state.grid.mass[i, j, k, i_b] = 0.0
            mpm_state.grid.vel_in[i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
            mpm_state.grid.vel_out[i, j, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
            if i_b == 0:
                mpm_state.grid.dirty_flag[i, j, k] = 0


@qd.kernel(fastcache=True)
def kernel_clear_dirty_count(mpm_state: MPMState):
    mpm_state.grid.dirty_count[0] = 0


# ------------------------------------------------------------------------------------
# --------------------------------------- io -----------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_add_particles(
    particle_start: int,
    material_idx: int,
    is_active: bool,
    default_Jp: float,
    rho: float,
    pos: qd.types.ndarray(),
    mpm_state: MPMState,
    mpm_info: MPMInfo,
):
    """Add the particles of an entity at rest, with the parameters of its material."""
    for i_p_ in range(pos.shape[0]):
        i_p = i_p_ + particle_start
        mpm_info.particles.material_idx[i_p] = material_idx
        mpm_info.particles.default_Jp[i_p] = default_Jp
        mpm_info.particles.mass[i_p] = mpm_info.particle_volume[None] * rho
        mpm_info.particles.is_free[i_p] = True
        mpm_info.particles.muscle_group[i_p] = 0
        mpm_info.particles.muscle_direction[i_p] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

    for i_p_, i_b in qd.ndrange(
        pos.shape[0], mpm_state.particles.pos.shape[1], axes=qd.static(eoa(mpm_state.particles.is_active))
    ):
        i_p = i_p_ + particle_start
        mpm_state.particles.is_active[i_p, i_b] = is_active
        mpm_state.particles.pos[i_p, i_b] = qd.Vector([pos[i_p_, 0], pos[i_p_, 1], pos[i_p_, 2]])
        mpm_state.particles.vel[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        mpm_state.particles.F[i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
        mpm_state.particles.C[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        mpm_state.particles.Jp[i_p, i_b] = default_Jp
        mpm_state.particles.actu[i_p, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_reset_particles_deformation(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), mpm_state: MPMState, mpm_info: MPMInfo
):
    """Reset the deformation of some particles between two steps, as they are at rest."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(mpm_state.particles.F))):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_values' in base_solver.py
        if i_p >= 0:
            mpm_state.particles.F[i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
            mpm_state.particles.C[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            mpm_state.particles.Jp[i_p, i_b] = mpm_info.particles.default_Jp[i_p]


# ------------------------------------------------------------------------------------
# -------------------------------- particle constraints ------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_clear_particle_constraints(particle_constraints: MPMParticleConstraints):
    for i_p, i_b in qd.ndrange(
        *particle_constraints.link_idx.shape, axes=qd.static(eoa(particle_constraints.is_constrained))
    ):
        particle_constraints.is_constrained[i_p, i_b] = False
        particle_constraints.link_idx[i_p, i_b] = -1


@qd.kernel(fastcache=True)
def kernel_set_particle_constraints(
    particles_start: int,
    link_idx: int,
    stiffness: float,
    particles_mask: qd.types.ndarray(),
    link_pos: qd.types.ndarray(),
    link_quat: qd.types.ndarray(),
    mpm_state: MPMState,
    particle_constraints: MPMParticleConstraints,
):
    """Attach the particles a mask selects to a link, pulled toward where they are now, in the frame of the link."""
    for i_p_, i_b in qd.ndrange(
        particles_mask.shape[1], particles_mask.shape[0], axes=qd.static(eoa(particle_constraints.is_constrained))
    ):
        if particles_mask[i_b, i_p_]:
            i_p = i_p_ + particles_start
            pos = mpm_state.particles.pos[i_p, i_b]
            l_pos = qd.Vector([link_pos[i_b, 0], link_pos[i_b, 1], link_pos[i_b, 2]], dt=gs.qd_float)
            l_quat = qd.Vector(
                [link_quat[i_b, 0], link_quat[i_b, 1], link_quat[i_b, 2], link_quat[i_b, 3]], dt=gs.qd_float
            )
            particle_constraints.is_constrained[i_p, i_b] = True
            particle_constraints.target_pos[i_p, i_b] = pos
            particle_constraints.stiffness[i_p, i_b] = stiffness
            particle_constraints.link_idx[i_p, i_b] = link_idx
            particle_constraints.link_local_pos[i_p, i_b] = gu.qd_inv_transform_by_trans_quat(pos, l_pos, l_quat)


@qd.kernel(fastcache=True)
def kernel_remove_particle_constraints(
    particles_start: int, particles_mask: qd.types.ndarray(), particle_constraints: MPMParticleConstraints
):
    for i_p_, i_b in qd.ndrange(
        particles_mask.shape[1], particles_mask.shape[0], axes=qd.static(eoa(particle_constraints.is_constrained))
    ):
        if particles_mask[i_b, i_p_]:
            i_p = i_p_ + particles_start
            particle_constraints.is_constrained[i_p, i_b] = False
            particle_constraints.link_idx[i_p, i_b] = -1


@qd.kernel(fastcache=True)
def kernel_apply_particle_constraints(
    mpm_state: MPMState,
    links_state: LinksState,
    particle_constraints: MPMParticleConstraints,
    mpm_info: MPMInfo,
):
    """Pull the constrained particles toward their target with a critically damped spring, in the next state."""
    for i_p, i_b in qd.ndrange(
        *mpm_state.particles.pos.shape, axes=qd.static(eoa(particle_constraints.is_constrained))
    ):
        if particle_constraints.is_constrained[i_p, i_b]:
            # The target of a particle attached to a link follows the link. It stays a local, since autodiff cannot
            # reverse an array read back after it is overwritten in the same kernel.
            target_pos = particle_constraints.target_pos[i_p, i_b]
            target_vel = qd.Vector.zero(gs.qd_float, 3)
            i_l = particle_constraints.link_idx[i_p, i_b]
            if i_l >= 0:
                target_pos = gu.qd_transform_by_trans_quat(
                    particle_constraints.link_local_pos[i_p, i_b], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
                target_vel = links_state.cd_vel[i_l, i_b] + links_state.cd_ang[i_l, i_b].cross(
                    target_pos - links_state.root_COM[i_l, i_b]
                )

            # The spring is integrated by backward Euler on the offset to the target, the offset moving with the
            # relative velocity it gives, which keeps it stable whatever its stiffness. The rigid solver has already
            # stepped and g2p has already advected the particle, so both offset and relative velocity are taken at the
            # end of the substep: m (v' - v) = -dt (k (x - x_t + dt (v' - v_t)) + c (v' - v_t)).
            stiffness = particle_constraints.stiffness[i_p, i_b]
            mass = mpm_info.particles.mass[i_p] / mpm_info.particle_volume_scale[None]
            damping = 2.0 * qd.math.sqrt(stiffness * mass)
            dt = mpm_info.substep_dt[None]
            dt_damping_stiffness = dt * damping + dt**2 * stiffness

            vel_error = mpm_state.particles.vel_next[i_p, i_b] - target_vel
            pos_error = mpm_state.particles.pos_next[i_p, i_b] - target_pos
            vel_delta = -(dt_damping_stiffness * vel_error + dt * stiffness * pos_error) / (mass + dt_damping_stiffness)
            # An accumulation, which autodiff reverses, where an overwrite of the velocity it reads is not
            qd.atomic_add(mpm_state.particles.vel_next[i_p, i_b], vel_delta)


@qd.func
def func_signmax(a: float, eps: float):
    """Return a value of the sign of `a` and the magnitude of `a`, at least `eps`."""
    sign = qd.select(a >= 0, 1.0, -1.0)
    return sign * qd.max(qd.abs(a), eps)


@qd.func
def func_backward_svd(
    grad_U: qd.types.matrix(3, 3),
    grad_S: qd.types.matrix(3, 3),
    grad_V: qd.types.matrix(3, 3),
    U: qd.types.matrix(3, 3),
    S: qd.types.matrix(3, 3),
    V: qd.types.matrix(3, 3),
    eps: float,
):
    """Return the adjoint of a matrix from those of the factors U, S, V of its singular value decomposition."""
    vt = V.transpose()
    ut = U.transpose()
    S_term = U @ grad_S @ vt

    s = qd.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
    F = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if qd.static(i != j):
            F[i, j] = 1.0 / func_signmax(s[j] - s[i], eps)
    u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
    v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
    return u_term + v_term + S_term
